# PAPI: PARALLEL PICARD-DUHAMEL INTEGRATION FOR FAST NEURO-SPECTRAL ARCHITECTURES

Arthur Bizzi Imperial College London London, United Kingdom a.bizzi@imperial.ac.uk

Kleyton Da Costa   
Computer Science   
University College London   
London, United Kingdom   
kleyton.costa.25@ucl.ac.uk   
Tiago Novello   
Instituto de Matemática Pura e Aplicada   
Rio de Janeiro, RJ, Brazil   
tiago.novello@impa.br   
Daniel Yukimura   
Instituto de Matemática Pura e Aplicada   
Rio de Janeiro, Brazil   
yukimura@impa.br

Márcio Marques Instituto de Matemática Pura e Aplicada Rio de Janeiro, RJ, Brazil marcio.marques@impa.br

Vitor Balestro   
Instituto de Matemática e Estatística   
Universidade Federal Fluminense   
Niterói, RJ, Brazil   
vitorbalestro@id.uff.br

João M Pereira Department of Mathematics University of Georgia Athens, GA, USA jpereira@uga.edu

## ABSTRACT

Neuro-spectral architectures (NeuSA) are a class of physics-informed neural networks (PINNs) that have proven effective for solving time-dependent PDEs in the frequency domain. However, training and inference often rely on costly numerical integration of the learned vector field, typically using sequential Runge–Kutta time stepping, which limits temporal parallelism. We introduce PaPi, an integration scheme inspired by decomposable linear/nonlinear flows that enables time-Parallel inference through exponential integration and a Picard-Duhamel scheme. We show that the Picard-Duhamel iteration converges exponentially for all flows, and provide explicit error bounds that depend on how close the flow is to linear. Experiments show that in some instances PaPi accelerates training and inference by up to three orders of magnitude, achieving training times of under one second for low-dimensional PDE learning problems, while maintaining competitive predictive accuracy.

## 1 INTRODUCTION

Physics-informed neural networks (PINNs) approximate solutions of partial differential equations (PDEs) by training neural representations against the governing equations, initial and boundary conditions, and available data. This formulation supports both forward simulation and inverse problems, including parameter identification and reconstruction from sparse measurements (Raissi et al., 2019). PINNs have been applied across science and engineering, including fluid mechanics, transport processes, and more (Karniadakis et al., 2021; Lu et al., 2021).

For time-dependent problems, PINNs can struggle to capture spatial detail (spectral bias) and respect the temporal causality imposed by the initial condition. Neuro-spectral architectures (NeuSA) (Bizzi et al., 2025) approach this problem by combining a spatial spectral representation with a system of

Neural ODEs (Chen et al., 2018). This reformulation allows them to achieve state-of-the-art accuracy with substantially shorter training times.

The network parameterizes the dynamics of the spectral coefficients, and the solution is obtained by integrating these dynamics from the prescribed initial state. This formulation makes time integration a computational bottleneck. Each training step requires evaluating a trajectory and differentiating a physics-informed loss through the numerical solver. The original NeuSA implementation uses explicit fourth-order Runge–Kutta (RK4) integration, whose time steps are evaluated sequentially. This dependence limits temporal parallelism even when spatial operations are performed on a GPU. Moreover, rapidly oscillating or strongly dissipative linear modes can restrict the stable step size of an explicit solver, increasing the number of steps required for each training update.

We introduce PaPi, a parallel integration and training scheme for neuro-spectral dynamics with a tractable linear component. Rather than propagating the solution sequentially through time, we construct successive approximations to the entire trajectory using a Picard–Duhamel formulation of the underlying ODE. Each iteration incorporates the linear dynamics exactly and updates the nonlinear correction through affine recurrences that can be evaluated using a parallel scan. This allows temporal propagation within each iteration to be carried out in parallel, substantially reducing the sequential depth of the ODE solve.

Our analysis establishes convergence of the continuous Picard–Duhamel iteration and provides explicit error bounds for the iteration and its temporal discretization. The implementation is tailored to the training of Neuro-Spectral Architectures, reusing trajectories across optimizer steps to reduce the number of iterations needed during training.

We evaluate PaPi on six forward PDE benchmarks and a coefficient-identification problem, covering heterogeneous wave propagation, nonlinear transport, and dispersive dynamics. Our experiments demonstrate substantial reductions in training time over the original sequential NeuSA formulation, with competitive or improved accuracy on most problems. For one-dimensional problems, the proposed models can reach sub-second training times, illustrating how parallel-in-time integration can fundamentally change the computational regime in which PINNs operate.

Our main contribution is to accelerate the training and evaluation of Neuro-Spectral Architectures through parallel time integration. Specifically:

• We introduce PaPi, a Picard–Duhamel integrator that combines exact linear propagation with parallel updates of the nonlinear correction, reducing the sequential depth of each trajectory update from O(N) to O(log N).

• We demonstrate substantial reductions in optimization time across six forward PDE benchmarks and a coefficient-identification problem, achieving sub-second optimization in a one-dimensional benchmark while maintaining competitive accuracy on most problems.

• We establish convergence of the continuous iteration and derive explicit iteration and quadrature error bounds, clarifying how the nonlinear correction affects the number of iterations required.

## 2 RELATED WORK

Physics-informed neural networks. Early neural methods for differential equations used trial solutions satisfying initial or boundary conditions by design (Lagaris et al., 1998). PINNs popularized flexible neural approximations with physical constraints imposed through residuals evaluated by automatic differentiation (Raissi et al., 2019), becoming central to physics-informed machine learning for forward and inverse problems (Karniadakis et al., 2021; Lu et al., 2021).

Gradient imbalances among PINN loss terms can produce pathological training (Wang et al., 2021a), while neural-tangent-kernel analyses reveal large disparities in their convergence rates (Wang et al., 2022). Krishnapriyan et al. (2021) linked failures on convection-, reaction-, and diffusion-dominated problems to PDE-residual optimization difficulties despite sufficient network expressivity. Remedies include adaptive weighting (McClenny & Braga-Neto, 2023), adaptive sampling and residual refinement (Lu et al., 2021; Yu et al., 2022), domain decomposition (Jagtap & Karniadakis, 2020; Shukla et al., 2021), and architectures tailored to physics-informed optimization (Wang et al., 2024a).

Spectral bias and causality. Coordinate-based multilayer perceptrons favor low frequencies, hindering high-frequency and multiscale solutions (Tancik et al., 2020). For PINNs, Wang et al. (2021b) analyzed this bias through the neural tangent kernel eigenspectrum and proposed Fourier feature embeddings to improve high-frequency representation. Conventional PINN objectives also do not explicitly enforce temporal causality: Wang et al. (2024b) showed severe training difficulties on multiscale, chaotic, and turbulent systems and introduced temporally weighted residuals to propagate information progressively from earlier to later times.

NeuSA (Bizzi et al., 2025) addresses both issues architecturally rather than only adjusting the objective: spatial spectral discretization reduces the PDE to a finite-dimensional dynamical system with a neural vector field integrated as a Neural ODE. The spectral representation exposes spatial modes, and the initial-value formulation imposes temporal causality. Subsequent work has extended NeuSA through temporal-domain decomposition in D-NeuSA, which specializes the learned dynamics across temporal subdomains Balestro et al. (2026), while Perturbative-NeuSA introduces a structured residual formulation that decomposes the solution into a low-fidelity background and a learned perturbation Zhu & Yin (2026). We retain this representation and address its main computational bottleneck: repeated integration during training.

Parallel time integration. Parallel-in-time methods address sequential time stepping by exposing temporal concurrency beyond spatial parallelism. Parareal (Lions et al., 2001) partitions time and iteratively combines inexpensive coarse propagation with accurate parallel fine solves; Gander & Vandewalle (2007) analyzed its convergence and connections to multiple shooting and multigrid.

Picard iteration for Diffusion Models. ParaDiGMS (Shih et al., 2023) uses Picard iteration to refine a diffusion model’s denoising trajectory, evaluating updates in parallel across time and accumulating them through prefix sums. A sliding temporal window controls memory usage and advances as earlier portions converge, trading additional computation for lower sampling latency.

Our approach uses the integral initial-value formulation: Picard iteration defines a fixed-point problem for the time-integrated vector field, allowing contributions across temporal intervals to be reorganized. For decomposable linear/nonlinear systems, the variation-of-constants (Duhamel) formula separates a tractable linear evolution from the nonlinear remainder, evaluated at the previous iterate. Propagation across the temporal grid then becomes associative composition, enabling parallel scans. We exploit this structure in NeuSA training, where repeated integration of learned high-dimensional spectral dynamics makes temporal parallelism particularly valuable.

## 3 METHOD

## 3.1 BACKGROUND

We consider the following initial-boundary value problem for $t \in [ 0 , T ]$ and $\mathbf { x } \in \Omega \subset \mathbb { R } ^ { d }$

$$
\frac { d } { d t } u ( t , x ) = { \bf F } \left( t , x , u , \nabla u , \nabla \nabla u \right) , \quad u ( 0 , x ) = u _ { 0 } ( x )\tag{1}
$$

where $u : [ 0 , T ] \times \Omega  \mathbb { R } ^ { n }$ denotes the (vector-valued) solution, ∇u and ∇∇u denote its first- and second-order spatial derivatives, and F is a smooth function. $u _ { 0 } ( \mathrm { x } )$ is the initial condition (Cauchy data), and we assume some boundary conditions are imposed on ∂Ω.

Neuro-spectral architectures describe the time evolution of a PDE as an ODE evolving in a Hilbert space via a spectral decomposition, approximating $\boldsymbol { u } ( \boldsymbol { x } , t )$ by its projection onto the Fourier basis Boyd (2013); Canuto et al. (2006); Trefethen (2000). The solution u is represented as a truncated Fourier expansion, stored as the coefficients tensor $\hat { \mathrm { u } } ( t ) : [ 0 , T ]  \mathbb { C } ^ { c _ { 1 } \times \cdots \times \hat { c } _ { d } \times n }$ , leading to an ODE in the Fourier coefficients themselves:

$$
\begin{array} { r } { { \bf u } ( t , \mathrm { x } ) = \sum _ { k } \hat { \bf u } _ { k } ( t ) e ^ { i \omega _ { k } } , } \end{array}\tag{2}
$$

$$
\begin{array} { r l } { \implies } & { { } \frac { d } { d t } \hat { \mathrm { \mathbf { u } } } = \hat { \mathbf { F } } ( \hat { \mathrm { \mathbf { u } } } ) , \quad \hat { \mathrm { u } } ( 0 ) = \hat { \mathrm { \mathbf { u } } } _ { 0 } , } \end{array}\tag{3}
$$

where k denotes a d-dimensional multi-index, and $\hat { \mathbf { F } } : \mathbb { C } ^ { c _ { 1 } \times \dots \times c _ { d } \times n }  \mathbb { C } ^ { c _ { 1 } \times \dots \times c _ { d } \times n }$ denotes the finite-dimensional projection of the operator F in the Fourier basis.

Importantly, $\hat { \mathbf { F } }$ does not generally admit a simple closed-form expression in the Fourier basis, and must instead be learned as a parameterized network ${ \widehat { \mathbf { F } } } _ { \theta }$ . Nevertheless, it often has structure: a

large class of canonical quasi-linear differential operators may be split into homogeneous linear and nonlinear components. In the Fourier basis, the former diagonalizes, leading to:

$$
\hat { \mathbf { F } } ( \hat { \mathbf { u } } ) = M \hat { \mathbf { u } } + \varepsilon \tilde { \mathbf { F } } ( \hat { \mathbf { u } } ) ,\tag{4}
$$

where M is a diagonal linear operator representing frequencies being multiplied by scalar factors.   
For example, the Laplacian $\Delta$ reduces to $\mathrm { \Delta } M \mathrm { \hat { u } } = | \omega | ^ { 2 } \odot \mathrm { \hat { i } }$ ˆu.

NeuSA models the temporal evolution of a PDE solution through a dynamical system in spectral space. In the original formulation, this system is integrated using standard time-stepping methods, such as Runge–Kutta schemes. Since these methods advance the solution sequentially, time integration dominates the cost of both training and inference.

## 3.2 PICARD ITERATIONS

Consider the initial value problem

$$
\frac { \mathrm { d } } { \mathrm { d } t } X ( t ) = G ( t , X ( t ) ) , \qquad X ( 0 ) = X _ { 0 } ,\tag{5}
$$

Given a temporal grid $0 = t _ { 0 } < t _ { 1 } < \cdot \cdot \cdot < t _ { N } = T$ , a standard time-stepping method introduces a strict dependency of future times on the past, meaning that while operations within each step may be parallelized, the time steps themselves cannot be evaluated simultaneously. Integrating (5) we get:

$$
X ( t ) = X _ { 0 } + \int _ { 0 } ^ { t } G ( s , X ( s ) ) \mathrm { d } s .\tag{6}
$$

The solution to (5) is a fixed point of the integral equation (6), suggesting the Picard iteration

$$
X ^ { ( k + 1 ) } ( t ) = X _ { 0 } + \int _ { 0 } ^ { t } G ( s , X ^ { ( k ) } ( s ) ) \mathrm { d } s , \qquad k = 0 , 1 , \dots ,\tag{7}
$$

Here, k is an iteration index rather than a time index: $X ^ { ( k ) }$ approximates the entire trajectory on $[ 0 , T ]$ Hence, given $X ^ { ( k ) }$ , all evaluations $G ( t _ { j } , X ^ { ( k ) } ( t _ { j } ) ) , j = 0 , \dots , N$ , can be performed simultaneously. After temporal discretization, the remaining prefix integrals reduce to cumulative operations that can be implemented using a parallel scan. Picard iteration therefore replaces the sequential chain of time steps by a small number of global trajectory updates.

However, time parallelism alone does not guarantee an efficient integrator. Consider the linear autonomous system $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } t } X ( t ) = A X ( t ) , X ( 0 ) \stackrel { \smile } { = } X _ { 0 } ; } \end{array}$ Picard iteration yields the Taylor expansion for the matrix exponential, reconstructing the flow through successive polynomial approximations. When the linear dynamics are mild, only a few iterations may suffice; in general, however, power-series methods scale poorly. Moreover, they are generally numerically unstable.

## 3.3 PICARD–DUHAMEL INTEGRATION

We consider systems of the form

$$
\frac { \mathrm { d } } { \mathrm { d } t } X ( t ) = A ( t ) X ( t ) + \varepsilon F ( X ( t ) ) , \qquad X ( 0 ) = X _ { 0 } ,\tag{8}
$$

where $A ( t )$ describes the linear dynamics and $\varepsilon F$ the nonlinear contribution. Let $\Phi _ { s } ^ { t }$ denote the linear propagator defined by the linear component:

$$
\frac { \partial } { \partial t } \Phi _ { s } ^ { t } = A ( t ) \Phi _ { s } ^ { t } , \qquad \Phi _ { s } ^ { s } = I .\tag{9}
$$

Thus, $\Phi _ { s } ^ { t } v$ propagates a state v from time s to t according to the homogeneous linear dynamics. For numerical evaluation, we approximate the propagator over the temporal grid by composing local matrix exponentials. The Duhamel principle yields the analytical solution

$$
X ( t ) = \Phi _ { 0 } ^ { t } X _ { 0 } + \varepsilon \int _ { 0 } ^ { t } \Phi _ { s } ^ { t } F ( X ( s ) ) \mathrm { d } s .\tag{10}
$$

The linear evolution is therefore incorporated directly through $\Phi _ { s } ^ { t }$ , rather than reconstructed through successive Picard iterations. Applying Picard iteration to (10) gives our Picard–Duhamel scheme,

$$
X ^ { ( k + 1 ) } ( t ) = \Phi _ { 0 } ^ { t } X _ { 0 } + \varepsilon \int _ { 0 } ^ { t } \Phi _ { s } ^ { t } F ( X ^ { ( k ) } ( s ) ) \mathrm { d } s , \qquad X ^ { ( 0 ) } ( t ) = \Phi _ { 0 } ^ { t } X _ { 0 } .\tag{11}
$$

The initialization already captures the complete linear dynamics and is exact when $\varepsilon = 0$ . Subsequent iterations therefore update only the nonlinear correction while retaining the time-parallel structure of classical Picard iteration.

## 3.4 DISCRETIZATION AND PARALLEL EVALUATION

In practice, we evaluate the Picard–Duhamel scheme (11) using a numerical integration method. Let $0 = t _ { 0 } < t _ { 1 } < \cdot \cdot \cdot < t _ { N } = T$ , with $\begin{array} { r } { h = \frac { T } { N } } \end{array}$ and $t _ { i } = i h$ . We use the trapezoidal method

$$
\bar { X } _ { j } ^ { ( k + 1 ) } : = \Phi _ { 0 } ^ { t _ { j } } X _ { 0 } + \frac { \varepsilon h } { 2 } \left( \Phi _ { 0 } ^ { t _ { j } } g _ { 0 } ^ { ( k ) } + 2 \sum _ { i = 1 } ^ { j - 1 } \Phi _ { t _ { i } } ^ { t _ { j } } g _ { i } ^ { ( k ) } + g _ { j } ^ { ( k ) } \right) ,\tag{12}
$$

where $g _ { j } ^ { ( k ) } = F ( \bar { X } _ { j } ^ { ( k ) } )$ . A crucial property of this numerical integration scheme is that it can be evaluated in parallel over time. All nonlinear terms $F ( X _ { i } ^ { ( k ) } )$ depend only on the previous trajectory and can therefore be evaluated simultaneously. Furthermore, the following recursion allows us to evaluate the cumulative integration as a parallel scan. Letting $P _ { j } = \Phi _ { t _ { j } } ^ { t _ { j + 1 } }$ , we have $\bar { X } _ { i + 1 } ^ { ( k + 1 ) } = P _ { j } \bar { X } _ { i } ^ { ( k + 1 ) } + b _ { j }$ with $\begin{array} { r } { b _ { j } = \frac { \varepsilon h } { 2 } ( P _ { j } g _ { j } ^ { ( k ) } + g _ { j + 1 } ^ { ( k ) } ) } \end{array}$ . Two such maps compose according to $( \tilde { P _ { b } } , b _ { b } ) \circ ( P _ { a } , \tilde { b _ { a } } ) = ( P _ { b } P _ { a } , \ P _ { b } b _ { a } + b _ { b } )$ . Since composition of affine maps is associative,

we can evaluate the iteration with a parallel scan, constructed through a tree of partial sums with O(log N) sequential depth and $O ( N )$ total work. See Appendix G.1 for more details.

## 3.5 CONVERGENCE AND ERROR BOUNDS

In the following theorem we establish convergence of the continuous Duhamel–Picard iteration (11), and show that successive iterates approximate the solution with a factorially decaying error.

Theorem 1. Consider (8) and let F be any Lipschitz-continuous function with constant L and suppose $\| A ( t ) \| _ { o p } \leq \alpha \forall t$ . Then the Duhamel-Picard scheme (11) converges to X, the solution of (8). Moreover, there exists a constant C depending only on $X _ { 0 } , F ( X _ { 0 } )$ and L such that the following error bound holds:

$$
\| X ^ { ( k ) } ( t ) - X ( t ) \| \leq C { \frac { L ^ { k } ( \varepsilon t ) ^ { k + 1 } } { ( k + 1 ) ! } } e ^ { ( \alpha + \varepsilon L ) t }
$$

We next consider the numerical integration scheme that evaluates the integrals in the Duhamel–Picard iteration. The total error is decomposed into the error of the continuous Duhamel–Picard iteration and the quadrature error introduced at each time step. Combining Theorem 1 with the quadrature error for trapezoidal rules yields the following convergence estimates.

Theorem 2. Consider (8) under the same conditions of Theorem 1 and let $C _ { 2 } : =$ sup $\begin{array} { r l } {  { \operatorname* { i } _ { 0 \leq s \geq 0 \atop 0 \leq s \leq t \leq T } \| \frac { d ^ { 2 } } { d s ^ { 2 } } ( \Phi _ { s } ^ { t } F ( X ^ { ( k ) } ( s ) ) ) \| } . } \end{array}$ . Then the error ofthe discrete Duhamel–Picard is bounded as follows.

$$
\| \bar { X } _ { j } ^ { ( k ) } - X ( t _ { j } ) \| \leq \underbrace { C \frac { L ^ { k } ( \varepsilon t _ { j } ) ^ { k + 1 } } { ( k + 1 ) ! } e ^ { ( \alpha + \varepsilon L ) t _ { j } } } _ { D u h a m e l - P i c a n d u e r a t i o n \ e r r o r } + \underbrace { \frac { \varepsilon h ^ { 3 } C _ { 2 } } { 1 2 } e ^ { \alpha t _ { j - 1 } } \frac { ( 1 + \varepsilon L h ) ^ { j + k - 1 } - 1 } { \varepsilon L h } } _ { Q u a d r a t u r e e r r o r }
$$

The factor $h ^ { 3 }$ is the local truncation-error factor of the trapezoidal method. Furthermore, the quadrature error increases with $j$ at least at a linear rate, which leads to the expected quadrature error of order $h ^ { 2 }$ of the trapezoidal method. The proof of Theorems 1 and 2 is covered in Appendix A.

Algorithm 1 One optimiser step of PaPi (warm-started Picard–Duhamel training).   
Require: parameters $\theta ,$ carried trajectory $\bar { X } = ( \bar { X } _ { 0 } , \ldots , \bar { X } _ { N } )$ , propagators $P _ { j } = \Phi _ { j } ^ { j + 1 }$ , sweeps $K$   
weights $w _ { j }$   
1: $X ^ { ( 0 ) } $ stop\_gradient $( { \bar { X } } )$   
2: for $k = 1 , \cdot \cdot \cdot , \mathsf { \bar { K } }$ do   
3: $g _ { j }  \varepsilon F _ { \theta } ( X _ { j } ^ { ( k - 1 ) } )$ for all $j$ (parallel over j)   
4: b<sub>j</sub> ← quadrature of $\begin{array} { r } { \int _ { t _ { j } } ^ { t _ { j + 1 } } \Phi _ { s } ^ { t _ { j + 1 } } g ( s ) } \end{array}$ ds from $g _ { j } , g _ { j + 1 }$   
5: $X ^ { ( k ) }$ ← scan $\left( ( P _ { j } , b _ { j } ) _ { j } \right)$ from $X _ { 0 }$ (parallel prefix)   
6: end for   
7: $\begin{array} { r } { \mathcal { L } ( \boldsymbol { \theta } )  \sum _ { j } w _ { j } } \end{array}$ ∥residual $_ j ( \theta , X ^ { ( K ) } ) \| ^ { 2 } / \sum _ { j } w _ { j }$   
8: θ ← Adam $( \theta , \nabla _ { \theta } \mathcal { L } )$ ; $\bar { X }  X ^ { ( K ) }$

## 3.6 TRAINING AND INFERENCE WITH PAPI

We now describe how the scheme is used inside a neuro-spectral model. The state X is the vector of spectral coefficients ˆu (or (ˆu, ˆv) for second-order problems), the linear part A is the analytic operator of (2) that NeuSA initialises around, and $\varepsilon F$ is the learned correction $\varepsilon \mathbf { F } _ { \theta } .$ . In every problem we consider, A is diagonal or block-diagonal in the basis, so $\Phi _ { s } ^ { t }$ is a closed-form exponential and is cheap to evaluate.

Linear part. Theorem 2 indicates that the speed of convergence of the Picard-Duhamel scheme may be substantially improved by reducing ε, the magnitude of the nonlinear part of the dynamics. Any linear term of the PDE whose propagator is available in closed form can then be moved from the network into the linear component; by Theorem 1 this shrinks the constant εk that controls the number of sweeps, and it also lowers the effective stiffness the network has to represent.

Warm-start training. Training minimises NeuSA’s PDE residual along the trajectory, with the RK4 rollout replaced by the iteration (11). Consecutive optimiser steps change θ only slightly, so the trajectory produced at the previous step is already close to the fixed point of the current one. We therefore carry the trajectory across steps: instead of starting the iteration 11 from the beginning at each step, the previous trajectory is taken as $X ^ { ( 0 ) }$ (with no gradient flowing into it):

$$
X ^ { ( k + 1 ) } ( t ) = \Phi _ { 0 } ^ { t } X _ { 0 } + \varepsilon \int _ { 0 } ^ { t } \Phi _ { s } ^ { t } F ( X ^ { ( k ) } ( s ) ) \mathrm { d } s , \qquad X ^ { ( 0 ) } ( t ) = X _ { n - 1 } ^ { ( 0 ) } ( t ) .\tag{13}
$$

K new sweeps are computed from it, and the loss is evaluated on $X ^ { ( K ) }$ and differentiated through those K sweeps only (Algorithm 1). Two sweeps per step suffice on all but the most strongly nonlinear problem we consider. Since the residual of the iterate is smallest at early times and grows with t like $( \varepsilon k t ) ^ { K + 1 }$ (Theorem 1), we weight the residual in time by $w ( t ) = \mathbf { \bar { \rho } } ( 1 + t ) ^ { - 2 }$ , which anchors the loss on the part of the trajectory that is already converged while the later part catches up over subsequent steps.

Exponential quadrature rules. We use an exponential variant of the trapezoidal method (12) that integrates Φ<sup>t</sup> exactly against a piecewise-linear interpolant of g. For brevity, we introduce it in Appendix G.2.

Inference. The trained model is deployed with a cold iteration from the linear flow, $X ^ { ( 0 ) } = \Phi _ { 0 } ^ { t } X _ { 0 }$ for a fixed number of sweeps ; eight sweeps suffice on all but the stiffest problem we consider, where 32 are used. Each sweep is one batched network evaluation and one scan of depth ${ \cal O } ( \log N )$ , so inference has sequential depth $O ( K \log N )$ rather than $O ( N )$

## 4 EXPERIMENTS

We evaluate PaPi on the 2D wave equation in a three-layer medium (2D Layers) and in the Marmousi model, the 3D wave equation in a two-layer medium (3D Layers), the coupled 2D Burgers’ equation, the 1D sine-Gordon equation, and the Korteweg–de Vries (KdV) equation, whose dispersive linear term makes the spectral ODE stiff. We also test PaPi on an inverse Burgers’ problem to identify PDE coefficients from data efficiently. We describe the PDEs and experimental setup in Appendix B.

Baselines. We follow the benchmarks of Bizzi et al. (2025), comparing against PINN (Raissi et al., 2019), QRes (Bu & Karpatne, 2021), FLS (Wong et al., 2024), and NeuSA. Appendix B provides the baseline configurations and adaptations for the additional KdV and inverse Burgers’ problems.

Implementation details. PaPi is implemented in JAX (Bradbury et al., 2018). All experiments were run on an NVIDIA RTX 5090 GPU. For comparison with NeuSA, PaPi replaces the integrator but retains each problem’s other parameters, including the basis, mode count, dimension-wise MLP with two hidden layers and residual loss, unless stated otherwise. The linear part A matches NeuSA’s initial linear operator; Appendix B lists exceptions, including the sine-Gordon mass term. Training follows Section 3.6 with $\dot { K } = 2$ warm-started Picard–Duhamel sweeps per optimiser step (six on Marmousi), residual weight $ ( 1 + t ) ^ { - 2 }$ , and Adam with the reported spectral-model learning rates, decay schedules and step counts unless stated otherwise. Inference uses 8 cold sweeps (32 on KdV; Appendix D). Training time is the wall-clock time of the optimisation loop, excluding JAX compilation (5–20 s per run). Forward results are means over five seeds per configuration; Appendix C gives standard deviations. We report relative $L ^ { 2 }$ (rMSE) and $L ^ { 1 }$ (rMAE) errors against the reference solution on the evaluation grid, plus relative coefficient errors for the inverse problem.

## 4.1 COMPARISONS

Table 1 reports rMSE and training time for the six forward problems; Figure 1 shows the Marmousi solution, and Appendix C gives rMAE, standard deviations, the error as a function of time and a time-against-error plot for every benchmark.

Table 1: Quantitative comparison. Mean rMSE and training time over five seeds for each model. TT denotes training time in seconds. PaPi has the shortest training time on all six problems and the lowest rMSE on five. Appendix C reports standard deviations and rMAE.
<table><tr><td></td><td colspan="2">2D Layers</td><td colspan="2">Marmousi</td><td colspan="2">3D Layers</td><td colspan="2">2D Burgers</td><td colspan="2">Sine-Gordon</td><td colspan="2">KdV</td></tr><tr><td>Model</td><td>rMSE</td><td>TT</td><td>rMSE</td><td>TT</td><td>rMSE</td><td>TT</td><td>rMSE</td><td>TT</td><td>rMSE</td><td>TT</td><td>rMSE</td><td>TT</td></tr><tr><td>PINN</td><td>0.549</td><td>420</td><td>0.618</td><td>427</td><td>0.062</td><td>4000</td><td>0.183</td><td>605</td><td>0.130</td><td>569</td><td>6.1e-3</td><td>908</td></tr><tr><td>QRes</td><td>0.150</td><td>765</td><td>0.401</td><td>707</td><td>0.026</td><td>5496</td><td>0.073</td><td>1178</td><td>0.013</td><td>781</td><td>9.5e-3</td><td>1240</td></tr><tr><td>FLS</td><td>0.492</td><td>408</td><td>0.712</td><td>408</td><td>0.071</td><td>4050</td><td>0.174</td><td>578</td><td>0.129</td><td>594</td><td>1.5e-2</td><td>935</td></tr><tr><td>NeuSA</td><td>0.069</td><td>508</td><td>0.266</td><td>604</td><td>0.008</td><td>577</td><td>0.053</td><td>119</td><td>9.0e-4</td><td>234</td><td>3.6e-3</td><td>1322</td></tr><tr><td>PaPi (ours)</td><td>0.068</td><td>12</td><td>0.154</td><td>66</td><td>0.018</td><td>251</td><td>0.028</td><td>30</td><td>5.6e-4</td><td>0.3</td><td>1.2e-3</td><td>3.2</td></tr></table>

![](images/f5352842a7effbec2918d471f6c5345cf61879f8be76c3e614d138a61d0e763b.jpg)  
Figure 1: Wave equation in the Marmousi medium at t = 2 (seed 0): the displacement u on the region of interest $[ - 2 , 2 ] ^ { 2 }$ (top) and the absolute error against the reference (bottom), with the relative $L ^ { 2 }$ error at this time in each error tile. PaPi resolves the reflected and refracted fronts that NeuSA blurs and the PINN misses.

Wave equation. On 2D Layers PaPi matches the accuracy of NeuSA (rMSE 0.068 against 0.069) while training in 12 s instead of 508 s. On Marmousi, the most heterogeneous medium, it is more accurate (0.15 against 0.27)<sup>\*</sup> and almost 10.0× faster. The speedup is smaller here for two reasons: the learned term, which represents the whole heterogeneity $( c ^ { 2 } - 1 ) \dot { \Delta }$ u with speeds between 0.57 and 2.18, is not weak, so six sweeps per step are needed; and the RK4 rollout is already compute-bound on $\phantom { - } 1 2 0 1 ^ { 2 } \times 2$ state, so the parallel scan removes depth but not arithmetic. On 3D Layers PaPi trains $2 . 3 \times$ faster than NeuSA but is less accurate (0.018 against 0.0076), which we attribute to the coarse training grid (56 of the 111 time nodes) and to the large learned term $( \epsilon = 1 0$ in NeuSA’s setting); the PINN baselines are worse by a further factor of 1.5–4 and take 7–20 times longer.

Sine-Gordon. This is the problem on which the integrator matters most. With the mass term carried by the linear part, PaPi reaches an rMSE of $5 . 6 \times 1 0 ^ { - { \overline { { 4 } } } }$ , below $\mathrm { N e u S A ^ { * } s 9 . 0 \times 1 0 ^ { - 4 } }$ , in 0.3 s of training against 234 s, i.e. over two orders of magnitude faster, and three orders of magnitude faster than the PINN baselines, which are also one to two orders of magnitude less accurate. The optimisation loop takes less than a second because each optimiser step is two parallel sweeps over a 402-dimensional state on 201 nodes, and the 1,000 steps are compiled into one program.

2D Burgers. NeuSA is trained for 200 steps, as in its paper; PaPi is trained for 1,000 steps, which still takes a quarter of the time (30 s against 119 s) and halves the error (rMSE 0.028 against 0.053). The best PINN baseline, QRes, reaches 0.073 in 1,178 s.

## 4.2 KORTEWEG–DE VRIES EQUATION

The KdV equation

$$
\partial _ { t } u + 6 u \partial _ { x } u + \partial _ { x x x } u = 0 , \qquad u ( 0 , x ) = \frac { c } { 2 } \mathrm { s e c h } ^ { 2 } \bigg ( \frac { \sqrt { c } } { 2 } ( x - x _ { 0 } ) \bigg ) ,\tag{14}
$$

on $[ - 1 0 , 1 0 ] \times [ 0 , 1 ]$ with periodic boundary conditions, $c = 4$ and $x _ { 0 } = - 6 ,$ has the one-soliton solution $u ( t , x ) = u ( 0 , x - c t )$ , which we use as reference. In a Fourier basis the linear term is the dispersion $- \partial _ { x x x } ,$ whose eigenvalues $\pm i k ^ { 3 }$ grow with the cube of the wavenumber; the spectral ODE is therefore stiff, leading to NeuSA’s collapse (Bizzi et al., 2025, App. E), as explicit RK4 is stable only for $\Delta t | k | _ { \operatorname* { m a x } } ^ { 3 } \lesssim 2 . 8$ . PaPi treats the dispersion exactly through $\Phi _ { s } ^ { t } = e ^ { i k ^ { 3 } ( t - s ) }$ , so the grid only has to resolve the smooth soliton and the learned correction. We use 81 modes for both methods so that they can be compared, $\epsilon = 0 . 1$ , and train PaPi for 10,000 steps with a slower learning-rate decay; the converged Picard solution requires 32 sweeps at inference (Appendix D).

PaPi reaches an rMSE of $1 . 2 \times 1 0 ^ { - 3 }$ in 3.2 s. NeuSA, which must take 1,001 sequential RK4 steps per optimiser step, takes $1 { , } 3 2 2 \mathrm { s }$ for 1,000 steps and an rMSE of $3 . 6 \times 1 0 ^ { - 3 }$ ; the PINN baselines lie between $6 \times 1 0 ^ { - 3 }$ and $1 . 5 \times 1 0 ^ { - 2 }$ after 900–1,200 s. At 101 modes, where RK4 diverges, PaPi trains in 3.4 s to an rMSE of $1 . 4 \times 1 0 ^ { - 3 }$

## 4.3 INVERSE PROBLEM: COEFFICIENT IDENTIFICATION IN BURGERS’ EQUATION

Since the Picard–Duhamel rollout is differentiable with respect to the linear operator as well as to the network, PaPi applies without change to inverse problems in which coefficients of the PDE are unknown. As a test of its ability to identify such coefficients from sparse data we take the parameter-identification problem for Burgers’ equation as in Raissi et al. (2019), given by:

$$
\partial _ { t } u + \lambda _ { 1 } u \partial _ { x } u = \lambda _ { 2 } \partial _ { x x } u , \qquad u ( 0 , x ) = - \sin ( \pi x ) ,\tag{15}
$$

on $[ - 1 , 1 ] \times [ 0 , 1 ]$ with periodic boundary conditions, where $\lambda _ { 1 }$ and $\lambda _ { 2 }$ are to be learned and the true values are $( \bar { \lambda _ { 1 } } , \bar { \lambda _ { 2 } } ) = ( 1 , 0 . 1 )$ . The model sees two snapshots of the solution: the initial condition, which it satisfies by construction, and the final state $u ( 1 , x )$ on the 199-node spatial grid, either clean or corrupted with Gaussian noise of standard deviation equal to 1% of that of the data. Both coefficients are trained jointly with the network by Adam. Since the outcome depends on where the coefficients start, the experiment is repeated for ten random initial guesses $( \lambda _ { 1 } ^ { 0 } , \lambda _ { 2 } ^ { \bar { 0 } } )$ drawn uniformly from $[ 0 . 1 , 2 ] \times [ 0 . 0 1 , 0 . { \overset { . } { 2 } } ]$ , one seed each, and the results are averaged.

Table 2: Inverse Burgers’ problem: mean ± std over the ten starting pairs of the relative $L ^ { 2 }$ error on [0, 1], the relative errors of the identified coefficients and the training time per run. The observation is the final frame only; noise is relative to the standard deviation of the data.
<table><tr><td>Noise</td><td>Model</td><td>rMSE</td><td> $| \lambda _ { 1 } - 1 |$ </td><td> $| \lambda _ { 2 } - 0 . 1 | / 0 . 1$ </td><td>TT (s)</td></tr><tr><td>0%</td><td>PINN (5,000 steps)</td><td> $9 . 9 \mathrm { e } \mathrm { - } 2 \pm 4 . 1 \mathrm { e } \mathrm { - } 2$ </td><td> $4 . 9 \mathrm { e } - 1 \pm 2 . 0 \mathrm { e } - 1$ </td><td> $8 . 2 \mathrm { e } \mathrm { - } 2 \pm 3 . 5 \mathrm { e } \mathrm { - } 2$ </td><td>419</td></tr><tr><td></td><td>QRes (5,000 steps)</td><td> $4 . 0 \mathrm { e } { - 2 } \pm 2 . 1 \mathrm { e } { - 2 }$ </td><td> $2 . 0 \mathrm { e } { - 1 } \pm 1 . 0 \mathrm { e } { - 1 }$ </td><td> $3 . 7 \mathrm { e } { - 2 } \pm 1 . 9 \mathrm { e } { - 2 }$ </td><td>561</td></tr><tr><td></td><td>FLS (5,000 steps)</td><td> $9 . 9 \mathrm { e } \mathrm { - } 2 \pm 4 . 1 \mathrm { e } \mathrm { - } 2$ </td><td> $4 . 9 \mathrm { e } - 1 \pm 2 . 0 \mathrm { e } - 1$ </td><td> $8 . 1 \mathrm { e } \mathrm { - } 2 \pm 3 . 5 \mathrm { e } \mathrm { - } 2$ </td><td>411</td></tr><tr><td></td><td>PaPi (500 steps)</td><td> $1 . 0 \mathrm { e } { - 2 } \pm 8 . 4 \mathrm { e } { - 3 }$ </td><td> $4 . 7 \mathrm { e } { - 2 } \pm 4 . 1 \mathrm { e } { - 2 }$ </td><td> $9 . 4 \mathrm { e } \mathrm { - } 3 \pm 8 . 2 \mathrm { e } \mathrm { - } 3$ </td><td>0.6</td></tr><tr><td></td><td>PaPi (5,000 steps, float64)</td><td> $3 . 8 \mathrm { e } { - 5 } \pm 1 . 2 \mathrm { e } { - 5 }$ </td><td> ${ \bf 1 . 1 e . 4 \pm 5 . 3 e . 5 }$ </td><td> ${ \bf 3 . 6 e - 5 \pm 2 . 2 e - 5 }$ </td><td>66</td></tr><tr><td>1%</td><td>PINN (5,000 steps)</td><td> $9 . 8 \mathrm { e } - 2 \pm 4 . 1 \mathrm { e } - 2$ </td><td> $4 . 9 \mathrm { e } - 1 \pm 2 . 0 \mathrm { e } - 1$ </td><td> $8 . 0 \mathrm { e } \mathrm { - } 2 \pm 3 . 5 \mathrm { e } \mathrm { - } 2$ </td><td>432</td></tr><tr><td></td><td>QRes (5,000 steps)</td><td> $4 . 0 \mathrm { e } { - 2 } \pm 2 . 2 \mathrm { e } { - 2 }$ </td><td> $1 . 9 \mathrm { e } - 1 \pm 1 . 0 \mathrm { e } - 1$ </td><td> $3 . 6 \mathrm { e } - 2 \pm 1 . 9 \mathrm { e } - 2$ </td><td>565</td></tr><tr><td></td><td>FLS (5,000 steps)</td><td> $9 . 8 \mathrm { e } - 2 \pm 4 . 2 \mathrm { e } - 2$ </td><td> $4 . 9 \mathrm { e } - 1 \pm 2 . 0 \mathrm { e } - 1$ </td><td> $8 . 0 \mathrm { e } \mathrm { - } 2 \pm 3 . 5 \mathrm { e } \mathrm { - } 2$ </td><td>406</td></tr><tr><td></td><td>PaPi (500 steps)</td><td> $9 . 8 \mathrm { e } . 3 \pm 8 . 5 \mathrm { e } . 3$ </td><td> $4 . 7 \mathrm { e } { - 2 } \pm 4 . 1 \mathrm { e } { - 2 }$ </td><td> $9 . 4 \mathrm { e } \mathrm { - } 3 \pm 8 . 3 \mathrm { e } \mathrm { - } 3$ </td><td>0.7</td></tr><tr><td></td><td>PaPi (5,000 steps, float64)</td><td> $1 . 4 \mathrm { e } { - 3 } \pm 5 . 0 \mathrm { e } { - 4 }$ </td><td> $4 . 2 \mathrm { e } { - 3 } \pm 2 . 7 \mathrm { e } { - 3 }$ </td><td> $2 . 0 \mathbf { e } { } { } { } { } { } { } 3 \pm 1 . 4 \mathbf { e } { } { } { } { } { } 3$ </td><td>65</td></tr></table>

As with KdV, the problem is stiff: $\lambda _ { 2 } k _ { \mathrm { m a x } } ^ { 2 } \approx 1 0 ^ { 4 }$ at the true viscosity, so on the data grid an explicit RK4 rollout is unstable before training even begins (it would need a time step ten to twenty times smaller), which is why sequential NeuSA does not appear in Table 2; the exact propagator is unaffected. The PINN, QRes and FLS baselines treat $( \lambda _ { 1 } , \lambda _ { 2 } )$ as trainable scalars in the residual and are trained for 5,000 steps. We report PaPi at a budget of 500 steps and, since it is cheap, at 5,000 steps in double precision.

Table 2 reports relative solution errors on [0, 1], relative coefficient errors and training time. Without temporal causality, the MLP baselines struggle to reconstruct the solution and recover coefficients from two widely separated snapshots. QRes performs best among them, with errors of 20% for $\lambda _ { 1 }$ and 4% for $\lambda _ { 2 } .$ At 500 steps, PaPi reaches 5% and 1%, respectively, in under a second, over two orders of magnitude faster. At 5,000 steps with clean data, both coefficient errors reach $1 0 ^ { - 4 }$ in about a minute; with 1% noise, they settle at $\bar { 4 } \times 1 0 ^ { - 3 }$ for $\lambda _ { 1 }$ and $2 \times 1 0 ^ { - 3 }$ for $\lambda _ { 2 }$

Evolving the identified equation with the reference solver on (1, 2], beyond the observed interval, gives relative solution errors of $1 . 0 \times 1 0 ^ { - 2 }$ after 500 training steps and $5 \times 1 0 ^ { - 5 }$ (clean) or $2 \times 1 0 ^ { - 3 }$ (noisy) after 5,000 steps. As in NeuSA, this extrapolation comes from the architecture’s causal structure.

## 4.4 DISCUSSION

Across the forward problems PaPi matches or improves NeuSA’s accuracy on five of six benchmarks while reducing training time by a factor that ranges from 2.3 (3D Layers) to about 800 (sine-Gordon), and it reaches errors one to two orders of magnitude below the MLP-based PINNs at a small fraction of their cost. The size of the speedup follows the sequential depth that is removed: for the 1D problems, NeuSA’s cost is dominated by the 4 × (time steps) dependent kernel launches of the RK4 rollout, and replacing them with a handful of parallel sweeps brings training below a few seconds; for the 2D and 3D problems, each step is already compute-bound on the spectral state, and the gain is the ratio between the arithmetic of an RK4 rollout and that of K sweeps, which is between 2 and 10. Part of the difference between NeuSA and PaPi on the 1D problems is due to the framework rather than to the integrator; Appendix E quantifies this with an RK4 rollout implemented in JAX, and shows that the remaining factor of 50–200 on the 1D problems, and of 2–10 on the 2D and 3D problems, is due to PaPi. Finally, KdV and the inverse Burgers’ problem illustrate the second advantage of the method: since the linear part is propagated exactly, the stiffness that constrains explicit time stepping in NeuSA, and forces a trade-off between spatial resolution and time step (Bizzi et al., 2025, App. D.1), does not enter PaPi, whose time grid only needs to resolve the nonlinear correction.

## 5 CONCLUSIONS

We introduce PaPi, a time-parallel integration scheme designed to accelerate the training and inference of Neuro-Spectral Architectures. By combining exact linear propagation with Picard–Duhamel iterations and parallel scans, our approach replaces sequential time stepping with parallel trajectory updates, addressing a major computational bottleneck of the original NeuSA formulation without modifying its underlying neural representation. We further established convergence of the proposed iteration and derived explicit error bounds for its numerical approximation.

Our experiments demonstrate substantial computational gains across a range of time-dependent PDEs, including forward and inverse problems. In several cases, PaPi reduces training times by orders of magnitude while maintaining competitive predictive accuracy, with sub-second training achieved on low-dimensional benchmarks. The computational benefits nevertheless depend on the number of iterations required to resolve the nonlinear dynamics. Improving the efficiency of these iterations, particularly for strongly nonlinear problems, is a natural direction for future work.

## 6 AI USE STATEMENT

We used generative AI tools to aid and polish writing, in particular editing the paper for readability (grammar, clarity, and phrasing of text written by the authors), and to create and edit auxiliary code.

## 7 ACKNOWLEDGMENTS

Arthur Bizzi is supported by UK Research and Innovation (UKRI) through the Engineering and Physical Sciences Research Council (EPSRC). Tiago Novello is thankful for financial support from Google. João M Pereira is thankful for a start-up grant from the University of Georgia. Vitor Balestro gratefully acknowledges Centro Pi/IMPA for providing computational resources used in this work.

## REFERENCES

Vitor Balestro, Márcio Marques, Leonardo Mendonça, Leonardo M. Moreira, Christian Júnior de Oliveira, Francisco Ganacim, Tiago Novello, Pavel Petrov, Daniel Yukimura, and Lucas Nissenbaum. Neuro-spectral architectures with time-domain decomposition. In ICLR 2026 Workshop on AI and Partial Differential Equations, 2026.

Arthur Bizzi, Leonardo Moreira, Márcio Marques, Leonardo Mendonça, Christian de Oliveira, Vitor Balestro, Lucas dos Santos Fernandez, Daniel Yukimura, Pavel Petrov, João Pereira, Tiago Novello, and Lucas Nissenbaum. Neuro-spectral architectures for causal physicsinformed networks. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, Main Conference, pp. 54419–54449. Curran Associates, Inc., 2025. doi: 10.52202/ 085713-1818. URL https://proceedings.neurips.cc/paper\_files/paper/ 2025/file/4e52bbb99690d1e05c7ef7b4c8b3569a-Paper-Conference.pdf.

J.P. Boyd. Chebyshev and Fourier Spectral Methods: Second Revised Edition. Dover Books on Mathematics. Dover Publications, 2013. ISBN 9780486141923. URL https://books. google.com.br/books?id=b4TCAgAAQBAJ.

James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Yash Katariya, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs, 2018. URL http://github.com/jax-ml/jax.

Jie Bu and Anuj Karpatne. Quadratic residual networks: A new class of neural networks for solving forward and inverse problems in physics involving pdes. In Proceedings of the 2021 SIAM International Conference on Data Mining (SDM), pp. 675–683. SIAM, 2021.

Claudio Canuto, M. Yousuff Hussaini, Alfio Quarteroni, and Thomas A. Zang. Spectral Methods: Fundamentals in Single Domains. Springer, 2006. doi: 10.1007/978-3-540-30726-6.

Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018. URL https://proceedings.neurips.cc/paper\_files/paper/ 2018/file/69386f6bb1dfed68692a24c8686939b9-Paper.pdf.

DeepMind, Igor Babuschkin, Kate Baumli, Alison Bell, Surya Bhupatiraju, Jake Bruce, Peter Buchlovsky, David Budden, Trevor Cai, Aidan Clark, et al. The DeepMind JAX ecosystem, 2020. URL http://github.com/google-deepmind.

Martin J. Gander and Stefan Vandewalle. Analysis of the Parareal time-parallel time-integration method. SIAM Journal on Scientific Computing, 29(2):556–578, 2007. doi: 10.1137/05064607X.

Marlis Hochbruck and Alexander Ostermann. Exponential integrators. Acta Numerica, 19:209–286, 2010. doi: 10.1017/S0962492910000048.

Ameya D. Jagtap and George Em Karniadakis. Extended physics-informed neural networks (XPINNs): A generalized space-time domain decomposition based deep learning framework for nonlinear partial differential equations. Communications in Computational Physics, 28(5): 2002–2041, 2020. doi: 10.4208/cicp.OA-2020-0164.

George Em Karniadakis, Ioannis G. Kevrekidis, Lu Lu, Paris Perdikaris, Sifan Wang, and Liu Yang. Physics-informed machine learning. Nature Reviews Physics, 3:422–440, 2021. doi: 10.1038/s42254-021-00314-5.

Patrick Kidger and Cristian Garcia. Equinox: neural networks in jax via callable pytrees and filtered transformations. arXiv preprint arXiv:2111.00254, 2021.

Aditi Krishnapriyan, Amir Gholami, Shandian Zhe, Robert Kirby, and Michael Mahoney. Characterizing possible failure modes in physics-informed neural networks. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan (eds.), Advances in Neural Information Processing Systems, volume 34, pp. 26548–26560. Curran Associates, Inc., 2021. URL https://proceedings.neurips.cc/paper\_files/paper/ 2021/file/df438e5206f31600e6ae4af72f2725f1-Paper.pdf.

Isaac E. Lagaris, Aristidis Likas, and Dimitrios I. Fotiadis. Artificial neural networks for solving ordinary and partial differential equations. IEEE Transactions on Neural Networks, 9(5):987–1000, 1998. doi: 10.1109/72.712178.

Ernest Lindelöf. Sur l’application des méthodes d’approximations successives à l’étude des intégrales réelles des équations différentielles ordinaires. Journal de mathématiques pures et appliquées, 10: 117–128, 1894.

Jacques-Louis Lions, Yvon Maday, and Gabriel Turinici. Résolution d’EDP par un schéma en temps “pararéel”. Comptes Rendus de l’Académie des Sciences - Series I - Mathematics, 332(7):661–668, 2001. doi: 10.1016/S0764-4442(00)01793-6.

Lu Lu, Xuhui Meng, Zhiping Mao, and George Em Karniadakis. DeepXDE: A deep learning library for solving differential equations. SIAM Review, 63(1):208–228, 2021. doi: 10.1137/19M1274067.

Levi D. McClenny and Ulisses M. Braga-Neto. Self-adaptive physics-informed neural networks. Journal ofComputational Physics, 474:111722, 2023. doi: https://doi.org/10.1016/j.jcp. 2022.111722. URL https://www.sciencedirect.com/science/article/pii/ S0021999122007859.

B. G. Pachpatte. Inequalities for Differential and Integral Equations, volume 197 of Mathematics in Science and Engineering. Academic Press, San Diego, 1998. ISBN 0-12-543430-8.

Maziar Raissi, Paris Perdikaris, and George E. Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378:686–707, 2019.

Andy Shih, Suneel Belkhale, Stefano Ermon, Dorsa Sadigh, and Nima Anari. Parallel sampling of diffusion models. Advances in Neural Information Processing Systems, 36:4263–4276, 2023.

Khemraj Shukla, Ameya D. Jagtap, and George Em Karniadakis. Parallel physics-informed neural networks via domain decomposition. Journal of Computational Physics, 447:110683, 2021. doi: 10.1016/j.jcp.2021.110683.

Matthew Tancik, Pratul Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan Barron, and Ren Ng. Fourier features let networks learn high frequency functions in low dimensional domains. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 7537–7547. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper\_files/paper/2020/ file/55053683268957697aa39fba6f231c68-Paper.pdf.

Lloyd N. Trefethen. Spectral Methods in MATLAB. SIAM, 2000. doi: 10.1137/1.9780898719598.

Roelof Versteeg. The Marmousi experience: Velocity model determination on a synthetic complex data set. The Leading Edge, 13(9):927–936, 1994.

Sifan Wang, Yujun Teng, and Paris Perdikaris. Understanding and mitigating gradient flow pathologies in physics-informed neural networks. SIAM Journal on Scientific Computing, 43(5):A3055–A3081, 2021a. doi: 10.1137/20M1318043.

Sifan Wang, Hanwen Wang, and Paris Perdikaris. On the eigenvector bias of Fourier feature networks: From regression to solving multi-scale PDEs with physics-informed neural networks. Computer Methods in Applied Mechanics and Engineering, 384:113938, 2021b. doi: 10.1016/j.cma.2021. 113938.

Sifan Wang, Xinling Yu, and Paris Perdikaris. When and why PINNs fail to train: A neural tangent kernel perspective. Journal ofComputational Physics, 449:110768, 2022. doi: 10.1016/j.jcp.2021. 110768.

Sifan Wang, Bowen Li, Yuhan Chen, and Paris Perdikaris. PirateNets: Physics-informed deep learning with residual adaptive networks. Journal of Machine Learning Research, 25(402):1–51, 2024a.

Sifan Wang, Shyam Sankaran, and Paris Perdikaris. Respecting causality for training physicsinformed neural networks. Computer Methods in Applied Mechanics and Engineering, 421: 116813, 2024b. doi: 10.1016/j.cma.2024.116813.

Jian Cheng Wong, Chin Chun Ooi, Abhishek Gupta, and Yew-Soon Ong. Learning in sinusoidal spaces with physics-informed neural networks. IEEE Transactions on Artificial Intelligence, 5(3): 985–1000, 2024. doi: 10.1109/TAI.2022.3192362.

Jeremy Yu, Lu Lu, Xuhui Meng, and George Em Karniadakis. Gradient-enhanced physics-informed neural networks for forward and inverse PDE problems. Computer Methods in Applied Mechanics and Engineering, 393:114823, 2022. doi: 10.1016/j.cma.2022.114823.

Xianli Zhu and Jia Yin. Perturbative-neusa: A structured spectral framework for time-dependent pdes. arXiv preprint arXiv:2607.24345, 2026. doi: 10.48550/arXiv.2607.24345.

## A PROOF OF CONVERGENCE AND ERROR BOUNDS

We first state and prove some necessary lemmas.

Lemma 3. Let $\Phi _ { s } ^ { t }$ be defined as in (9), with $s \leq t a n d \| A ( t ) \| _ { o p } \leq \alpha \forall t .$ . Then $\| \Phi _ { s } ^ { t } \| _ { o p } \leq e ^ { \alpha ( t - s ) }$ and $\| \Phi _ { s } ^ { t } - I \| _ { o p } \leq e ^ { \alpha ( t - s ) } - 1$ , where I is the identity matrix.

Proof. Let z be any vector, and define $Z ( t ) = \Phi _ { s } ^ { t } z . \ Z ( t )$ admits the integral form:

$$
Z ( t ) = Z ( s ) + \int _ { s } ^ { t } A ( \tau ) Z ( \tau ) d \tau .
$$

Taking the norm, we obtain

$$
\begin{array} { l } { \displaystyle \| Z ( t ) \| \leq \| Z ( s ) \| + \int _ { s } ^ { t } \| A ( \tau ) Z ( \tau ) \| d \tau , } \\ { \displaystyle \leq \| Z ( s ) \| + \int _ { s } ^ { t } \alpha \| Z ( \tau ) \| d \tau . } \end{array}\tag{16}
$$

Applying Grönwall’s inequality (Pachpatte, 1998, Theorem 1.2.2) to (16), we obtain $\forall s , t$

$$
\begin{array} { c } { \| \Phi _ { s } ^ { t } z \| = \| Z ( t ) \| \leq e ^ { \alpha ( t - s ) } \| Z ( s ) \| } \\ { = e ^ { \alpha ( t - s ) } \| z \| } \end{array}
$$

Using this inequality, we obtain

$$
\begin{array} { r l } {  { \| ( \Phi _ { s } ^ { t } - I ) z \| = \| Z ( t ) - Z ( s ) \| } } \\ & { \leq \displaystyle \int _ { s } ^ { t } \alpha \| Z ( \tau ) \| d \tau } \\ & { \leq \alpha \displaystyle \int _ { s } ^ { t } e ^ { \alpha ( \tau - s ) } \| z \| d \tau } \\ & { = ( e ^ { \alpha ( t - s ) } - 1 ) \| z \| } \end{array}
$$

Since this holds for any vector z, the proof follows.

Lemma 4. Let $q \geq 1 , r , s \geq 0 ,$ , and suppose $\{ m _ { j } ^ { ( k ) } \}$ follows the recurrence

$$
m _ { j + 1 } ^ { ( k + 1 ) } \leq q m _ { j } ^ { ( k + 1 ) } + r \big ( m _ { j + 1 } ^ { ( k ) } + q m _ { j } ^ { ( k ) } \big ) + s\tag{17}
$$

for all $j , k \geq 0$ , with $m _ { 0 } ^ { ( k ) } = m _ { j } ^ { ( 0 ) } = 0 .$ . Then, for every $j \geq 1$ and $k \geq 1$

$$
m _ { j } ^ { ( k ) } \leq s q ^ { j - 1 } \frac { ( 1 + 2 r ) ^ { j + k - 1 } - 1 } { 2 r } ,\tag{18}
$$

with $\begin{array} { r } { \frac { ( 1 + 2 r ) ^ { j + k - 1 } - 1 } { 2 r } = j + k - 1 i f r = 0 . } \end{array}$

Proof. Define the rescaled sequence

$$
a _ { j } ^ { ( k ) } : = q ^ { 1 - j } m _ { j } ^ { ( k ) } .
$$

Dividing (17) by $q ^ { j }$ gives

$$
a _ { j + 1 } ^ { ( k + 1 ) } \leq a _ { j } ^ { ( k + 1 ) } + r \big ( a _ { j + 1 } ^ { ( k ) } + a _ { j } ^ { ( k ) } \big ) + s q ^ { - j } .
$$

Since $q \geq 1$ , we have $q ^ { - j } \leq 1$ , and therefore

$$
a _ { j + 1 } ^ { ( k + 1 ) } \leq a _ { j } ^ { ( k + 1 ) } + r \big ( a _ { j + 1 } ^ { ( k ) } + a _ { j } ^ { ( k ) } \big ) + s .\tag{19}
$$

Summing (19) over $j$ index from 0 to $j - 1$ and using $a _ { 0 } ^ { ( k + 1 ) } = a _ { 0 } ^ { ( k ) } = 0$ , we obtain

$$
\begin{array} { r l } { \displaystyle a _ { j } ^ { ( k + 1 ) } \leq r \sum _ { i = 0 } ^ { j - 1 } \bigl ( a _ { i + 1 } ^ { ( k ) } + a _ { i } ^ { ( k ) } \bigr ) + j s } \\ { \displaystyle = r \left( a _ { j } ^ { ( k ) } + 2 \sum _ { i = 1 } ^ { j - 1 } a _ { i } ^ { ( k ) } \right) + j s , } \end{array}\tag{20}
$$

We now prove by induction on k that

$$
a _ { j } ^ { ( k + 1 ) } \leq s \sum _ { \ell = 0 } ^ { k } ( 2 r ) ^ { \ell } { \binom { j + \ell } { \ell + 1 } }\tag{21}
$$

for every $j \geq 1$

For $k = 0$ , the condition $a _ { j } ^ { ( 0 ) } = 0$ and (20) imply $\begin{array} { r } { a _ { j } ^ { ( 1 ) } \leq j s = s \binom { j } { 1 } } \end{array}$ , and for $k > 0$ , substituting (21) into (20) yields

$$
a _ { j } ^ { ( k + 1 ) } \leq j s + r s \sum _ { \ell = 0 } ^ { k - 1 } ( 2 r ) ^ { \ell } \left[ { \binom { j + \ell } { \ell + 1 } } + 2 \sum _ { i = 1 } ^ { j - 1 } { \binom { i + \ell } { \ell + 1 } } \right] .
$$

The hockey-stick identity gives

$$
\sum _ { i = 1 } ^ { j - 1 } { \binom { i + \ell } { \ell + 1 } } = { \binom { j + \ell } { \ell + 2 } } .
$$

Moreover, by nonnegativity and Pascal’s identity,

$$
\begin{array} { c } { { \displaystyle \binom { j + \ell } { \ell + 1 } + 2 \binom { j + \ell } { \ell + 2 } \leq 2 \left[ \binom { j + \ell } { \ell + 1 } + \binom { j + \ell } { \ell + 2 } \right] } } \\ { { = 2 \binom { j + \ell + 1 } { \ell + 2 } . } } \end{array}
$$

Consequently,

$$
\begin{array} { l } { { \displaystyle a _ { j } ^ { ( k + 1 ) } \leq j s + s \sum _ { \ell = 0 } ^ { k - 1 } ( 2 r ) ^ { \ell + 1 } \binom { j + \ell + 1 } { \ell + 2 } } } \\ { ~ } \\ { { \displaystyle ~ = j s + s \sum _ { \ell = 1 } ^ { k } ( 2 r ) ^ { \ell } \binom { j + \ell } { \ell + 1 } } } \\ { { \displaystyle ~ = s \sum _ { \ell = 0 } ^ { k } ( 2 r ) ^ { \ell } \binom { j + \ell } { \ell + 1 } , } } \end{array}
$$

This completes the induction and since $m _ { j } ^ { ( k + 1 ) } = q ^ { j - 1 } a _ { j } ^ { ( k + 1 ) }$

$$
m _ { j } ^ { ( k ) } \leq s q ^ { j - 1 } \sum _ { \ell = 0 } ^ { k - 1 } ( 2 r ) ^ { \ell } { \binom { j + \ell } { \ell + 1 } } .
$$

Equation (18) follows

$$
\begin{array} { r l } & { m _ { j } ^ { ( k ) } \le s q ^ { j - 1 } \displaystyle \sum _ { \ell = 0 } ^ { k - 1 } ( 2 r ) ^ { \ell } \left( \small j + \ell \right) } \\ & { \quad \le s q ^ { j - 1 } \displaystyle \sum _ { \ell = 0 } ^ { k - 1 } ( 2 r ) ^ { \ell } \left( \small j + k - 1 \right) } \\ & { \quad \qquad \quad \vdots + s q ^ { j - 1 } } \\ & { \quad \le s q ^ { j - 1 } \displaystyle \sum _ { \ell = 0 } ^ { j + k - 2 } ( 2 r ) ^ { \ell } \left( \small j + k - 1 \right) } \\ & { \quad \quad \ell = s q ^ { j - 1 } \displaystyle \sum _ { \ell = 0 } ^ { \left( 1 + 2 r \right) ^ { j + k - 1 } - 1 } } \end{array}
$$

Proof of Theorem 1. The proof is inspired by the classical convergence argument for Picard iteration due to Lindelöf Lindelöf (1894). Let $Y ^ { ( k ) } = X ^ { ( k + 1 ) } - X ^ { ( k ) }$ . For $k = 0$ , we have the following

estimate:

$$
\begin{array} { r l } & { \| Y ^ { ( 0 ) } \| = \| X ^ { ( 1 ) } - X ^ { ( 0 ) } \| \le \varepsilon \int _ { 0 } ^ { t } \Big | \Big | \Phi _ { s } ^ { * } \big ( F ( X ^ { ( 0 ) } ( s ) ) \big | \Big | \ d s , } \\ & { \qquad \le \varepsilon \int _ { 0 } ^ { t } e ^ { \varepsilon ( t - s ) } \Big \| F \big ( X ^ { ( 0 ) } ( s ) \big ) \Big \| , } \\ & { \qquad \le \varepsilon \int _ { 0 } ^ { t } e ^ { \varepsilon ( t - s ) } \Big ( \big \| F \big ( X ^ { ( 0 ) } ( s ) \big ) - F ^ { \prime } ( X _ { 0 } ) \big \| + \| F ( X _ { 0 } ) \| \Big ) , } \\ & { \qquad \le \varepsilon \int _ { 0 } ^ { t } e ^ { \varepsilon ( t - s ) } \Big ( L \| X ^ { ( 0 ) } ( s ) - X _ { 0 } \| + \| F ( X _ { 0 } ) \| \Big ) , } \\ & { \qquad \le \varepsilon \int _ { 0 } ^ { t } e ^ { \varepsilon ( t - s ) } \big ( L \big ( e ^ { \varepsilon \alpha s } - 1 \big ) \| X _ { 0 } \| + \| F ( X _ { 0 } ) \| \big ) , } \\ & { \qquad \le \frac { \varepsilon } { \alpha } \Big ( ( \{ ( \varepsilon - 1 ) e ^ { \alpha } + 1 \} \varepsilon ] \| X _ { 0 } \| + \| F ( X _ { 0 } ) \| \big ) , } \\ & { \qquad \le \varepsilon ( L \| X _ { 0 } \| + \| F ( X _ { 0 } ) \| \varepsilon ^ { \alpha , \alpha } \big ) } \end{array}\tag{22}
$$

where we use Lemma 3 in lines 1 and 5. For $k > 0 ,$ , we obtain the following recurrence:

$$
\begin{array} { l } { \displaystyle \| Y ^ { ( k + 1 ) } ( t ) \| \le \varepsilon \int _ { 0 } ^ { t } \left\| \Phi _ { s } ^ { t } \big ( F ( X ^ { ( k + 1 ) } ( s ) ) - F ( X ^ { ( k ) } ( s ) ) \big ) \right\| \mathrm { d } s } \\ { \displaystyle \le \varepsilon \int _ { 0 } ^ { t } e ^ { \alpha ( t - s ) } \left\| \big ( F ( X ^ { ( k + 1 ) } ( s ) ) - F ( X ^ { ( k ) } ( s ) ) \big ) \right\| \mathrm { d } s } \\ { \displaystyle \le \varepsilon L \int _ { 0 } ^ { t } e ^ { \alpha ( t - s ) } \left\| X ^ { ( k + 1 ) } ( s ) - X ^ { ( k ) } ( s ) \right\| \mathrm { d } s } \\ { \displaystyle \le \varepsilon L \int _ { 0 } ^ { t } e ^ { \alpha ( t - s ) } \left\| Y ^ { ( k ) } ( s ) \right\| \mathrm { d } s } \end{array}
$$

Letting $C = L \| X _ { 0 } \| + \| F ( X _ { 0 } ) \|$ , we now show by induction over k that:

$$
\| Y ^ { ( k ) } ( t ) \| \leq C \frac { L ^ { k } ( \varepsilon t ) ^ { k + 1 } } { ( k + 1 ) ! } e ^ { \alpha t }\tag{23}
$$

The case $k = 0$ follows from (22), while for $k + 1$

$$
\begin{array} { r l } {  { \| Y ^ { ( k + 1 ) } ( t ) \| \le \varepsilon L \int _ { 0 } ^ { t } e ^ { \alpha ( t - s ) } \| Y ^ { ( k ) } ( s ) \| \mathrm { d } s } } \\ & { \le C \frac { L ^ { k + 1 } \varepsilon ^ { k + 2 } } { ( k + 1 ) ! } \int _ { 0 } ^ { t } e ^ { \alpha ( t - s ) } e ^ { \alpha s } s ^ { k + 1 } \mathrm { d } s } \\ & { = C \frac { L ^ { k + 1 } ( \varepsilon t ) ^ { k + 2 } } { ( k + 2 ) ! } e ^ { \alpha t } } \end{array}
$$

We have $\textstyle \sum _ { k = 0 } ^ { \infty } | Y ^ { ( k ) } | < \infty$ , which implies that $X ^ { ( k ) }$ is a Cauchy sequence, hence it converges, and we have

$$
\begin{array} { r l } { \| X ^ { ( k ) } ( t ) - X ( t ) \| \le \displaystyle \sum _ { i = k } ^ { \infty } \| X ^ { ( i + 1 ) } ( t ) - X ^ { ( i ) } ( t ) \| = \displaystyle \sum _ { i = k } ^ { \infty } \| Y ^ { ( i ) } ( t ) \| } & { } \\ { \le \displaystyle \frac { C } { L } e ^ { \alpha t } \displaystyle \sum _ { i = k } ^ { \infty } \frac { ( \varepsilon L t ) ^ { i + 1 } } { ( i + 1 ) ! } . } \end{array}
$$

Finally, the error bound follows from $( k + i ) ! \geq k ! i !$

$$
\sum _ { i = k } ^ { \infty } \frac { ( \varepsilon L t ) ^ { i + 1 } } { ( i + 1 ) ! } = \frac { ( \varepsilon L t ) ^ { k + 1 } } { ( k + 1 ) ! } \sum _ { i = 0 } ^ { \infty } \frac { ( k + 1 ) ! } { ( i + k + 1 ) ! } ( \varepsilon L t ) ^ { i } \leq \frac { ( \varepsilon L t ) ^ { k + 1 } } { ( k + 1 ) ! } \sum _ { i = 0 } ^ { \infty } \frac { 1 } { i ! } ( \varepsilon L t ) ^ { i } = \frac { ( \varepsilon L t ) ^ { k + 1 } } { ( k + 1 ) ! } e ^ { \varepsilon L t } .
$$

ProofofTheorem 2. Let $m _ { j } ^ { ( k ) } = \| \bar { X } _ { j } ^ { ( k ) } - X ^ { ( k ) } ( t _ { j } ) \|$ . Applying the trapezoidal bounds and the triangle inequality we obtain

$$
\begin{array} { r l } & { m _ { j + 1 } ^ { ( k + 1 ) } = \| \bar { X } _ { j + 1 } ^ { ( k + 1 ) } - X ^ { ( k + 1 ) } ( t _ { j + 1 } ) \| } \\ & { \qquad \leq \underbrace { \| P _ { j } \bar { X } _ { j } ^ { ( k + 1 ) } - P _ { j } X ^ { ( k + 1 ) } ( t _ { j } ) \| } _ { ( 1 ) } + \underbrace { \| b _ { j } - \tilde { b } _ { j } \| } _ { ( 2 ) } + \underbrace { \| \Delta X ^ { ( k + 1 ) } ( t _ { j } ) - \tilde { b } _ { j } \| } _ { ( 3 ) } , } \end{array}
$$

where

$$
\begin{array} { c } { { b _ { j } = \displaystyle \frac { \varepsilon h } { 2 } ( P _ { j } g _ { j } ^ { ( k ) } + g _ { j + 1 } ^ { ( k ) } ) , } } \\ { { { } } } \\ { { g _ { j } ^ { ( k ) } = F ( \bar { X } _ { j } ^ { ( k ) } ) , } } \\ { { { } \tilde { b } _ { j } = \displaystyle \frac { \varepsilon h } { 2 } ( P _ { j } \tilde { g } _ { j } ^ { ( k ) } + \tilde { g } _ { j + 1 } ^ { ( k ) } ) , } } \\ { { { } } } \\ { { \tilde { g } _ { j } ^ { ( k ) } = F ( X ^ { ( k ) } ( t _ { j } ) ) , } } \\ { { \Delta X ^ { ( k + 1 ) } ( t _ { j } ) = X ^ { ( k + 1 ) } ( t _ { j + 1 } ) - P _ { j } X ^ { ( k + 1 ) } ( t _ { j } ) . } } \end{array}
$$

Using $\| \boldsymbol { P } _ { j } \| _ { o p } \leq e ^ { \alpha h }$ , we have

$$
\| P _ { j } \bar { X } _ { j } ^ { ( k + 1 ) } - P _ { j } X ^ { ( k + 1 ) } ( t _ { j } ) \| \leq e ^ { \alpha h } \| \bar { X } _ { j } ^ { ( k + 1 ) } - X ^ { ( k + 1 ) } ( t _ { j } ) \| = e ^ { \alpha h } m _ { j } ^ { ( k + 1 ) }
$$

Similarly, we have

$$
\begin{array} { l } { \displaystyle \| b _ { j } - \tilde { b } _ { j } \| \le \frac { \varepsilon h } { 2 } \left( e ^ { \alpha h } \| g _ { j } ^ { ( k ) } - \tilde { g } _ { j } ^ { ( k ) } \| + \| g _ { j + 1 } ^ { ( k ) } - \tilde { g } _ { j + 1 } ^ { ( k ) } \| \right) } \\ { \displaystyle \le \frac { \varepsilon h } { 2 } \left( e ^ { \alpha h } \| \bar { X } _ { j } ^ { ( k ) } - X ^ { ( k ) } ( t _ { j } ) \| + \| \bar { X } _ { j + 1 } ^ { ( k ) } - X ^ { ( k ) } ( t _ { j + 1 } ) \| \right) } \\ { \displaystyle = \frac { \varepsilon L h } { 2 } ( e ^ { \alpha h } m _ { j } ^ { ( k ) } + m _ { j + 1 } ^ { ( k ) } ) , } \end{array}
$$

Finally, (11) and the truncation error of the trapezoidal method implies

$$
\begin{array} { r l } & { \| \Delta X ^ { ( k + 1 ) } ( t _ { j } ) - \tilde { b } _ { j } \| = \left\| \varepsilon \int _ { t _ { j } } ^ { t _ { j + 1 } } \Phi _ { s } ^ { t _ { j + 1 } } F ( X ^ { ( k ) } ( s ) ) \mathrm { d } s - \tilde { b } _ { j } \right\| } \\ & { \qquad = \varepsilon \left\| \int _ { t _ { j } } ^ { t _ { j + 1 } } \Phi _ { s } ^ { t _ { j + 1 } } F ( X ^ { ( k ) } ( s ) ) \mathrm { d } s - \frac { \varepsilon h } { 2 } ( P _ { j } \tilde { g } _ { j } ^ { ( k ) } + \tilde { g } _ { j + 1 } ^ { ( k ) } ) \right\| } \\ & { \qquad \le \frac { \varepsilon h ^ { 3 } C _ { 2 } } { 1 2 } } \end{array}
$$

We obtain

$$
m _ { j + 1 } ^ { ( k + 1 ) } \leq e ^ { \alpha h } m _ { j } ^ { ( k + 1 ) } + \frac { \varepsilon L h } { 2 } ( e ^ { \alpha h } m _ { j } ^ { ( k ) } + m _ { j + 1 } ^ { ( k ) } ) + \frac { \varepsilon h ^ { 3 } C _ { 2 } } { 1 2 }
$$

Applying Lemma 4 we obtain

$$
m _ { j } ^ { ( k ) } \leq \frac { \varepsilon h ^ { 3 } C _ { 2 } } { 1 2 } e ^ { \alpha t _ { j - 1 } } \frac { ( 1 + \varepsilon L h ) ^ { j + k - 1 } - 1 } { \varepsilon L h }
$$

Finally, applying Theorem 1,

$$
\begin{array} { l } { \displaystyle \| \bar { X } _ { j } ^ { ( k ) } - X ( t _ { j } ) \| \le \| X ^ { ( k ) } ( t _ { j } ) - X ( t _ { j } ) \| + \| \bar { X } _ { j } ^ { ( k ) } - X ^ { ( k ) } ( t _ { j } ) \| , } \\ { \displaystyle \le C \frac { L ^ { k } ( \varepsilon t _ { j } ) ^ { k + 1 } } { ( k + 1 ) ! } e ^ { ( \alpha + \varepsilon L ) t _ { j } } + \frac { \varepsilon h ^ { 3 } C _ { 2 } } { 1 2 } e ^ { \alpha t _ { j - 1 } } \frac { ( 1 + \varepsilon L h ) ^ { j + k - 1 } - 1 } { \varepsilon L h } . } \end{array}
$$

## B BENCHMARK PROBLEMS AND EXPERIMENTAL DETAILS

## B.1 PROBLEMS

The five forward benchmarks are those of NeuSA (Bizzi et al., 2025), with the same discretisations, networks and reference solutions; we restate them here for completeness, together with the choices specific to PaPi on each. KdV and the inverse problem are defined in Sections 4.2 and 4.3.

Wave equation: 2D Layers, 3D Layers and Marmousi. The acoustic wave equation in a heterogeneous medium,

$$
\partial _ { t t } u = c ( \mathbf { x } ) ^ { 2 } \Delta u , \qquad u ( 0 , \mathbf { x } ) = \exp \left( - \frac { | \mathbf { x } - \mathbf { x } _ { 0 } | ^ { 2 } } { 2 \sigma ^ { 2 } } \right) , \qquad \partial _ { t } u ( 0 , \mathbf { x } ) = 0 ,\tag{24}
$$

in three settings: two dimensions with three horizontal layers of speeds 1, 1.25 and 1.5 (2D Layers, $\sigma = 0 . 1 )$ , two dimensions with the Marmousi velocity model (Versteeg, 1994) (c between 0.57 and 2.18), and three dimensions with two layers of speeds 1 and 0.5 (3D Layers, $\sigma = 0 . 1 5 )$ . As in NeuSA, the 2D problems are posed on $[ - 4 , 4 ] ^ { \dot { 2 } }$ , twice the region of interest $[ - 2 , 2 ] ^ { 2 }$ , in a cosine basis, so that reflections from the artificial boundary do not reach the scored region within $T = 2 ;$ they use $2 0 1 ^ { 2 }$ modes and 201 time steps. The 3D problem is posed on $[ - 1 . 5 , 1 . 5 ] ^ { 3 }$ with $1 0 1 ^ { 3 }$ modes and 111 time steps up to $T = 1 . ^ { \dagger }$ The vector field is NeuSA’s first-order system in $( \hat { u } , \hat { v } ) , v = \partial _ { t } u \colon$ the linear part is the homogeneous wave operator with unit speed, $A = \hat { ( } _ { \Delta \mathrm { ~ 0 ~ } } ^ { \mathrm { ~ 0 ~ } I } \hat { ) }$ , and the network is left with the whole effect of the heterogeneity, $( c ^ { 2 } - 1 ) \Delta u ,$ , with $\epsilon = 1$ (2D Layers), 2 (Marmousi) and 10 (3D Layers). Two settings are specific to PaPi. On 2D and 3D Layers the training rollout uses the exponential trapezoidal rule on a coarser grid than the evaluation one (101 of the 201 nodes, and 56 of the 111). On Marmousi, whose learned term is not weak, two warm-started sweeps per step do not track the trajectory; we use six, with 2,500 Adam steps under a warm-up–cosine learning-rate schedule (linear warm-up to $1 0 ^ { - 2 }$ over 150 steps, cosine decay to $1 0 ^ { - 4 } )$ and the residual imposed on a window 15 grid points wider than the scored one, as in the authors’ script. The references are the authors’ data: eighth-order finite differences for the 2D problems and their pseudo-spectral RK4 generator for the 3D one.

Sine-Gordon. NeuSA’s 1D sine-Gordon problem,

$$
\partial _ { t t } u = \partial _ { x x } u - 1 0 \sin u , \qquad u ( 0 , x ) = { \frac { 1 } { \sqrt { 2 \pi } \sigma } } \exp \left( - { \frac { x ^ { 2 } } { 2 \sigma ^ { 2 } } } \right) , \qquad \partial _ { t } u ( 0 , x ) = 0 ,\tag{25}
$$

on $[ - 4 , 4 ] \times [ 0 , 3 ]$ with homogeneous Dirichlet conditions and $\sigma = 0 . 1$ , in a sine basis with 201 modes and 201 time steps, trained for 1,000 steps with $\epsilon = 0 . 1$ . NeuSA initialises around the Laplacian. For PaPi we move the mass term into the linear part, so that A is the Klein–Gordon operator $\partial _ { x x } - 1 0$ whose mode-k block has eigenvalues $\pm i \sqrt { k ^ { 2 } \pi ^ { 2 } / L ^ { 2 } + 1 0 }$ , and the network is left with $1 0 ( u - \sin u )$ which is cubic at small amplitude (Section 3.6). The reference is the authors’ pseudo-spectral RK4 solution.

2D Burgers. The coupled system

$$
\partial _ { t } \mathbf { u } = - ( \mathbf { u } \cdot \nabla ) \mathbf { u } + \nu \Delta \mathbf { u } , \qquad \mathbf { u } ( 0 , \mathbf { x } ) = { \bigl ( } \sin ( \pi x _ { 1 } ) \sin ( \pi x _ { 2 } ) , \cos ( \pi x _ { 1 } ) \cos ( \pi x _ { 2 } ) { \bigr ) } ,\tag{26}
$$

with $\mathbf { u } = ( u , v ) , ( \mathrm { x } , t ) \in [ 0 , 4 ] ^ { 2 } \times [ 0 , 1 ] , \nu = 0 . 0 1$ and periodic boundary conditions, in a Fourier basis with $2 0 1 ^ { 2 }$ modes and 201 time steps. The linear part is the heat operator $\nu \Delta$ , as in NeuSA’s initialisation, and $\epsilon = 0 . 1$ . NeuSA is trained for 200 steps, as in its paper; PaPi for 1,000, since the loss was still decreasing at that budget and the extra steps are cheap. The reference is the authors’ pseudo-spectral RK4 solution.

KdV and inverse Burgers. For KdV (Section 4.2) the linear $\mathrm { p a r t } - \partial _ { x x x }$ couples each $\left( \cos _ { k } , \sin _ { k } \right)$ pair of the real Fourier basis through a $2 \times 2$ rotation generator of frequency $k ^ { 3 }$ , which is exponentiated in closed form; both NeuSA and PaPi use 81 modes and 1,001 time steps, the finest grid on which RK4 is stable at $\Delta t = 1 0 ^ { - 3 } ( \Delta t k _ { \operatorname* { m a x } } ^ { 3 } \approx 2 . 0 ;$ ; at 101 modes it is 3.9 and RK4 returns $\bar { \mathrm { N a N } } ) , \epsilon = 0 . 1$ and PaPi is trained for 10,000 steps with decay 0.9995 per step instead of 0.999. The reference is a pseudo-spectral RK4 solution on the model’s grid, which agrees with the exact soliton to $4 \times 1 0 ^ { - 4 }$ For the inverse problem (Section 4.3) the linear part is $\lambda _ { 2 } \Delta$ with the current value of $\lambda _ { 2 } ;$ the reference is the clean solution with the true coefficients from NETD’s pseudo-spectral solver on an 801-node grid, and the identified equation is rolled out with the same solver on (1, 2].

## B.2 SETTINGS

Table 3 lists the discretisation and training settings of every problem. Basis, modes, time grid, network, ϵ, learning rate and, unless stated, number of steps are those of NeuSA’s released scripts; where the paper text and the scripts differ, we used the configuration that reproduces the numbers of the paper’s tables.

Table 3: Settings per problem. $n _ { t }$ is the number of time nodes on $[ 0 , T ] ;$ ; “train nodes” the coarser grid the training rollout uses with the exponential trapezoidal rule (–: full grid); K the warm-started sweeps per optimiser step; “deploy” the number of Picard sweeps at inference. Steps are PaPi’s, with NeuSA’s in parentheses when different.
<table><tr><td>Problem</td><td>Basis</td><td>Modes</td><td> $\smash { n _ { t } ~ ( T ) }$ </td><td>€</td><td>Steps</td><td>lr</td><td>K</td><td>Quadrature</td><td>Train nodes</td><td>Deploy</td><td>Params</td></tr><tr><td>2D Layers</td><td>cosine</td><td> $2 0 1 ^ { 2 }$ </td><td>201 (2)</td><td>1</td><td>2,000</td><td> $1 0 ^ { - 2 }$ </td><td>2</td><td>exp. trapezoid</td><td>101</td><td>8</td><td>3.2e5</td></tr><tr><td>Marmousi</td><td>cosine</td><td> $2 0 1 ^ { 2 }$ </td><td>201 (2)</td><td>2</td><td>2,500 (2,000)</td><td>cosine</td><td>6</td><td>trapezoid</td><td>1</td><td>8</td><td>3.2e5</td></tr><tr><td>3D Layers</td><td>cosine</td><td> $1 0 1 ^ { 3 }$ </td><td>111 (1)</td><td>10</td><td>4,000 (2,000)</td><td> $1 0 ^ { - 2 }$ </td><td>2</td><td>exp. trapezoid</td><td>56</td><td>8</td><td>6.1e4</td></tr><tr><td>2D Burgers</td><td>Fourier</td><td> $2 0 1 ^ { 2 }$ </td><td>201 (1)</td><td>0.1</td><td>1,000 (200)</td><td> $5 \cdot 1 0 ^ { - 3 }$ </td><td>2</td><td>trapezoid</td><td></td><td>8</td><td>8.1e5</td></tr><tr><td>Sine-Gordon</td><td>sine</td><td>201</td><td>201 (3)</td><td>0.1</td><td>1,000</td><td> $2 \cdot 1 0 ^ { - 2 }$ </td><td>2</td><td>trapezoid</td><td></td><td>8</td><td>9.7e5</td></tr><tr><td>KdV</td><td>Fourier</td><td>81</td><td>1,001 (1)</td><td>0.1</td><td>10,000 (1,000)</td><td>10⁻2</td><td>2</td><td>trapezoid</td><td></td><td>32</td><td>1.6e5</td></tr><tr><td>Inverse Burgers</td><td>Fourier</td><td>199</td><td>401 (1)</td><td>0.1</td><td>500 / 5,000</td><td> $1 0 ^ { - 2 }$ </td><td>8 (cold)</td><td>exp. trapezoid</td><td></td><td>8</td><td>9.5e5</td></tr></table>

Hardware and software. All runs, ours and the baselines, were executed on machines with an NVIDIA GeForce RTX 5090 (32 GB, 575 W power limit) and an Intel i9-13900K. PaPi is implemented in JAX 0.11 with Equinox (Kidger & Garcia, 2021) and Optax (DeepMind et al., 2020), CUDA 13, with the default TF32 matrix multiplications; the baselines and NeuSA run the authors’ PyTorch 2.9.1 code (CUDA 12.8) with torchdyn 1.0.6, as pinned by their lock file. Training time is the wall-clock time of the optimiser loop; for JAX the ahead-of-time compilation (5–20 s per run) is excluded. Every run records the host, GPU, library versions and commit, and no partial runs are included.

Baselines. PINN, QRes and FLS use the authors’ scripts and the settings of the NeuSA paper: learning rate $1 0 ^ { - 3 } .$ , initial-condition weight $\mathrm { 1 0 ^ { 3 } ~ ( 2 D }$ Burgers: $1 0 ^ { 2 }$ , boundary weight 1), 20,000 steps with 10,000 residual, 1,000 initial and 500 boundary points resampled at every step for the 2D problems (3D Layers: 70,000 / 10,000 / 2,000), and 10,000 steps on the full $2 0 1 \times 2 0 1$ space-time grid for the 1D problems. NeuSA runs the authors’ scripts unchanged; for KdV, which the authors do not provide, we forked their 1D Burgers script, keeping model, trainer and integrator, and verified that with the network switched off the fork and our implementation agree on the initial coefficients to $1 0 ^ { - 1 5 }$ , on the RK4 trajectory to $1 0 ^ { - 6 }$ and on the loss to seven digits. Marmousi NeuSA is run at the paper’s configuration $( 2 0 1 ^ { 2 }$ modes, 2,000 steps, residual window and subsampling as in the authors NETD script); it reaches an rMSE of $0 . 2 7 \pm 0 . 0 9$ against the $0 . 1 7 1 \pm 0 . 0 3 5$ reported in the paper, whereas on the other four benchmarks the released code reproduces the paper’s tables (Table 5). For the inverse problem the PINN, QRes and FLS baselines are those of NETD’s protocol, with $( \lambda _ { 1 } , \lambda _ { 2 } )$ as trainable scalars, 5,000 steps, data weight 10 and initial-condition weight 10.

## C FULL RESULTS

Table 4 gives the mean and standard deviation over five seeds of rMSE, rMAE and training time for the six benchmarks of Table 1. Figure 2 plots training time against rMSE for every method on each benchmark, and Figure 3 shows how the error accumulates over time on three of them: both neuro-spectral models start from the exact initial condition and their error grows causally, whereas the error of the best MLP-based baseline is spread over the whole interval; on Marmousi the gap between PaPi and NeuSA widens with time.

## D NUMBER OF PICARD SWEEPS AT INFERENCE

Training uses two warm-started sweeps per optimiser step (six on Marmousi), but the trained model is deployed with a cold Picard–Duhamel solve starting from the linear flow $\Phi _ { 0 } ^ { t } X _ { 0 }$ . Figure 4 scores every trained model after $k = 1 , 2 , \ldots$ . sweeps. On all problems but KdV the error is stationary from eight sweeps on, so eight were used throughout; at four sweeps the iteration has not converged on the problems with the largest learned term (2D Layers, Marmousi, 2D Burgers). KdV, whose nonlinear term $6 u \partial _ { x } u$ is not small relative to the horizon, needs 32 sweeps; the cold iteration converges between 16 and 32 sweeps and does not change afterwards. This is consistent with Theorem 1: the number of sweeps required grows with εkT and is independent of the stiffness of A. A sweep costs a fraction of a millisecond on the 1D problems and up to a few hundred milliseconds on Marmousi, so inference is dominated by the 8 to 32 parallel sweeps rather than by 201 sequential RK4 steps.

![](images/6f84c93aab59b73562c0207c7ab402be4a2eef48fa8789c5c0a84c96c3808a41.jpg)  
Figure 2: Training time against rMSE on every benchmark of Table 1 (means over five seeds, one RTX 5090). The arrow joins NeuSA to PaPi: the same model, trained through the parallel Picard– Duhamel rollout instead of the sequential RK4 one.

![](images/20bfa3e4df78304bbb451ba80e8b19d3bacec55aaae00f168a06cda1b7850a53.jpg)

![](images/016e8d9af766668a04b79127491228466522327145a2882884bd7b0051dcd7ef.jpg)

![](images/b2239ed103a4b3679ebfcb90f6303c1dbfc07169223f38585537da7652fcaaf1.jpg)  
Figure 3: Relative $L ^ { 2 }$ error at each time t (mean over five seeds, band from best to worst seed) for PaPi, NeuSA and the best MLP-based baseline, QRes.

Table 4: Main benchmarks: mean ± std over five seeds, all on the same RTX 5090.
<table><tr><td>Problem</td><td>Model</td><td>rMSE</td><td>rMAE</td><td> $\mathrm { T T } \left( \mathrm { s } \right)$ </td></tr><tr><td>2D Layers</td><td>PINN</td><td> $5 . 5 \mathrm { e } \mathrm { - } 1 \pm 8 . 9 \mathrm { e } \mathrm { - } 2$ </td><td> $7 . 8 \mathsf { e } - 1 \pm 1 . 3 \mathsf { e } - 1$ </td><td> $4 2 0 \pm 5 6$ </td></tr><tr><td></td><td>QRes</td><td> $1 . 5 \mathrm { e } \mathrm { - } 1 \pm 7 . 2 \mathrm { e } \mathrm { - } 2$ </td><td> $1 . 9 \mathrm { e } \mathrm { - } 1 \pm 7 . 5 \mathrm { e } \mathrm { - } 2$ </td><td> $7 6 5 \pm 2 1 2$ </td></tr><tr><td></td><td>FLS</td><td> $4 . 9 \mathrm { e } \mathrm { - } 1 \pm 5 . 2 \mathrm { e } \mathrm { - } 2$ </td><td> $7 . 2 \mathrm { e } \mathrm { - } 1 \pm 8 . 3 \mathrm { e } \mathrm { - } 2$ </td><td> $4 0 8 \pm 6 0$ </td></tr><tr><td></td><td>NeuSA</td><td> $6 . 9 \mathrm { e } \mathrm { - } 2 \pm 2 . 0 \mathrm { e } \mathrm { - } 2$ </td><td> $9 . 4 \mathrm { e } \mathrm { - } 2 \pm 3 . 1 \mathrm { e } \mathrm { - } 2$ </td><td> $5 0 8 \pm 1 9$ </td></tr><tr><td></td><td>PaPi</td><td> $6 . 8 \mathsf { e } - 2 \pm 1 . 8 \mathsf { e } - 2$ </td><td> $9 . 1 \mathrm { e } \mathrm { - } 2 \pm 2 . 7 \mathrm { e } \mathrm { - } 2$ </td><td> $1 1 . 8 \pm 0 . 0$ </td></tr><tr><td>Marmousi</td><td>PINN</td><td> $6 . 2 \mathrm { e } \mathrm { - } 1 \pm 3 . 5 \mathrm { e } \mathrm { - } 2$ </td><td> $9 . 2 \mathrm { e } \mathrm { - } 1 \pm 8 . 1 \mathrm { e } \mathrm { - } 2$ </td><td> $4 2 7 \pm 6 8$ </td></tr><tr><td></td><td>QRes</td><td> $4 . 0 \mathrm { e } \mathrm { - } 1 \pm 1 . 5 \mathrm { e } \mathrm { - } 2$ </td><td> $5 . 4 \mathrm { e } \mathrm { - } 1 \pm 2 . 0 \mathrm { e } \mathrm { - } 2$ </td><td> $7 0 7 \pm 1 8 1$ </td></tr><tr><td></td><td>FLS</td><td> $7 . 1 \mathrm { e } \mathrm { - } 1 \pm 6 . 3 \mathrm { e } \mathrm { - } 2$ </td><td> $1 . 0 6 \pm 1 . 2 \mathrm { e } \mathrm { - } 1$ </td><td> $4 0 8 \pm 5 4$ </td></tr><tr><td></td><td>NeuSA</td><td> $2 . 7 \mathrm { e } \mathrm { - } 1 \pm 9 . 4 \mathrm { e } \mathrm { - } 2$ </td><td> $3 . 4 \mathrm { e } \mathrm { - } 1 \pm 1 . 0 \mathrm { e } \mathrm { - } 1$ </td><td> $6 0 4 \pm 1 9 1$ </td></tr><tr><td></td><td>PaPi</td><td> $1 . 5 \mathrm { e } \mathrm { - } 1 \pm 1 . 5 \mathrm { e } \mathrm { - } 2$ </td><td> $2 . 0 \mathrm { e } \mathrm { - } 1 \pm 1 . 5 \mathrm { e } \mathrm { - } 2$ </td><td> $6 5 . 7 \pm 0 . 3$ </td></tr><tr><td>3D Layers</td><td>PINN</td><td> $6 . 2 \mathrm { e } { - 2 } \pm 6 . 4 \mathrm { e } { - 3 }$ </td><td> $1 . 4 \mathrm { e } \mathrm { - } 1 \pm 2 . 1 \mathrm { e } \mathrm { - } 2$ </td><td> $4 0 0 0 \pm 1 7 2$ </td></tr><tr><td></td><td>QRes</td><td> $2 . 6 \mathrm { e } - 2 \pm 1 . 6 \mathrm { e } - 2$ </td><td> $4 . 4 \mathrm { e } { - 2 } \pm 2 . 1 \mathrm { e } { - 2 }$ </td><td> $5 4 9 6 \pm 4 3$ </td></tr><tr><td></td><td>FLS</td><td> $7 . 1 \mathrm { e } \mathrm { - } 2 \pm 1 . 5 \mathrm { e } \mathrm { - } 2$ </td><td> $1 . 8 \mathrm { e } \mathrm { - } 1 \pm 4 . 9 \mathrm { e } \mathrm { - } 2$ </td><td> $4 0 5 0 \pm 2 3$ </td></tr><tr><td></td><td>NeuSA</td><td> $7 . 6 \mathrm { e } { - 3 } \pm 7 . 7 \mathrm { e } { - 4 }$ </td><td> $1 . 1 \mathrm { e } { - 2 } \pm 2 . 7 \mathrm { e } { - 3 }$ </td><td> $5 7 7 \pm 1 3$ </td></tr><tr><td></td><td>PaPi</td><td> $1 . 8 \mathrm { e } { - 2 } \pm 1 . 1 \mathrm { e } { - 2 }$ </td><td> $2 . 9 \mathrm { e } - 2 \pm 1 . 7 \mathrm { e } - 2$ </td><td> $2 5 1 \pm 1$ </td></tr><tr><td>2D Burgers</td><td>PINN</td><td> $1 . 8 \mathrm { e } \mathrm { - } 1 \pm 7 . 2 \mathrm { e } \mathrm { - } 2$ </td><td> $1 . 3 \mathrm { e } \mathrm { - } 1 \pm 6 . 8 \mathrm { e } \mathrm { - } 2$ </td><td> $6 0 5 \pm 8 7$ </td></tr><tr><td></td><td>QRes</td><td> $7 . 3 \mathrm { e } \mathrm { - } 2 \pm 1 . 3 \mathrm { e } \mathrm { - } 3$ </td><td> $2 . 3 \mathrm { e } { - 2 } \pm 1 . 6 \mathrm { e } { - 3 }$ </td><td> $1 1 7 8 \pm 3 4 8$ </td></tr><tr><td></td><td>FLS</td><td> $1 . 7 \mathrm { e } \mathrm { - } 1 \pm 7 . 0 \mathrm { e } { - 2 }$ </td><td> $1 . 2 \mathrm { e } \mathrm { - } 1 \pm 6 . 8 \mathrm { e } \mathrm { - } 2$ </td><td> $5 7 8 \pm 6 8$ </td></tr><tr><td></td><td>NeuSA</td><td> $5 . 3 \mathrm { e } { - 2 } \pm 5 . 4 \mathrm { e } { - 3 }$ </td><td> $4 . 3 \mathrm { e } { - 2 } \pm 4 . 3 \mathrm { e } { - 3 }$ </td><td> $1 1 9 \pm 4$ </td></tr><tr><td></td><td>PaPi</td><td> $2 . 8 \mathrm { e } { - 2 } \pm 1 . 7 \mathrm { e } { - 3 }$ </td><td> $2 . 1 \mathrm { e } { - 2 } \pm 2 . 4 \mathrm { e } { - 3 }$ </td><td> $2 9 . 6 \pm 2 . 1$ </td></tr><tr><td>Sine-Gordon</td><td>PINN</td><td> $1 . 3 \mathrm { e } \mathrm { - } 1 \pm 2 . 3 \mathrm { e } \mathrm { - } 2$ </td><td> $1 . 4 \mathrm { e } \mathrm { - } 1 \pm 2 . 4 \mathrm { e } \mathrm { - } 2$ </td><td> $5 6 9 \pm 1 6$ </td></tr><tr><td></td><td>QRes</td><td> $1 . 3 \mathrm { e } { - 2 } \pm 7 . 1 \mathrm { e } { - 3 }$ </td><td> $1 . 8 \mathrm { e } { - 2 } \pm 1 . 1 \mathrm { e } { - 2 }$ </td><td> $7 8 1 \pm 2 8$ </td></tr><tr><td></td><td>FLS</td><td> $1 . 3 \mathrm { e } \mathrm { - } 1 \pm 3 . 0 \mathrm { e } \mathrm { - } 2$ </td><td> $1 . 4 \mathrm { e } \mathrm { - } 1 \pm 2 . 9 \mathrm { e } \mathrm { - } 2$ </td><td> $5 9 4 \pm 1 7$ </td></tr><tr><td></td><td>NeuSA</td><td> $9 . 0 \mathrm { e } . 4 \pm 1 . 7 \mathrm { e } . 5$ </td><td> $1 . 2 \mathrm { e } { \cdot } 3 \pm 3 . 2 \mathrm { e } { \cdot } 5$ </td><td> $2 3 4 \pm 9$ </td></tr><tr><td></td><td>PaPi</td><td> $5 . 6 \mathrm { e } { - 4 } \pm 7 . 0 \mathrm { e } { - 5 }$ </td><td> $8 . 1 \mathrm { e } \mathrm { - } 4 \pm 8 . 1 \mathrm { e } \mathrm { - } 5$ </td><td> $0 . 3 \pm 0 . 0$ </td></tr><tr><td>KdV</td><td>PINN</td><td> $6 . 1 \mathrm { e } \substack { - 3 \pm 7 . 6 \mathrm { e } - 4 }$ </td><td> $9 . 0 { \mathrm { e } } { - } 3 \pm 1 . 1 { \mathrm { e } } { - } 3$ </td><td> $9 0 8 \pm 1 7$ </td></tr><tr><td></td><td>QRes</td><td> $9 . 5 \mathrm { e } \mathrm { - } 3 \pm 1 . 0 \mathrm { e } \mathrm { - } 2$ </td><td> $1 . 7 \mathrm { e } \mathrm { - } 2 \pm 1 . 9 \mathrm { e } \mathrm { - } 2$ </td><td> $1 2 4 0 \pm 2 9$ </td></tr><tr><td></td><td>FLS</td><td> $1 . 5 \mathrm { e } { - 2 } \pm 1 . 3 \mathrm { e } { - 2 }$ </td><td> $2 . 9 \mathrm { e } - 2 \pm 3 . 2 \mathrm { e } - 2$ </td><td> $9 3 5 \pm 1 6$ </td></tr><tr><td></td><td>NeuSA</td><td> $3 . 6 \mathrm { e } { - 3 } \pm 1 . 2 \mathrm { e } { - 3 }$ </td><td> $6 . 5 \mathrm { e } { - 3 } \pm 1 . 9 \mathrm { e } { - 3 }$ </td><td> $1 3 2 2 \pm 1 0 9$ </td></tr><tr><td></td><td>PaPi</td><td> $1 . 2 \mathrm { e } { \cdot } 3 \pm 4 . 7 \mathrm { e } { \cdot } 4$ </td><td> $2 . 2 \mathrm { e } \mathrm { } . 3 \pm 7 . 5 \mathrm { e } . 4$ </td><td> $3 . 2 \pm 0 . 0$ </td></tr><tr><td></td><td>PaPi, 101 modes</td><td> $1 . 4 \mathrm { e } { - 3 } \pm 7 . 7 \mathrm { e } { - 4 }$ </td><td> $2 . 5 \mathrm { e } { - 3 } \pm 1 . 3 \mathrm { e } { - 3 }$ </td><td> $3 . 4 \pm 0 . 0$ </td></tr></table>

## E JAX IMPLEMENTATION AND COMPARISON WITH THE PYTORCH RK4 ROLLOUT

The port. PaPi is implemented in a JAX library that also contains a port of NeuSA: the same spectral bases (analysis and synthesis matrices, derivative operators), the same dimension-wise MLP, the same first-order $( \hat { u } , \hat { v } )$ formulation of the second-order problems, the same residual losses and metrics, and an explicit RK4 rollout. Two things differ from the PyTorch original. First, the basis matrices are assembled in double precision and then cast, which reduces the round-trip error of the transforms at 201 modes by a factor of six to ten; second, the RK4 rollout is a single lax.scan over the time grid and the optimiser steps are fused into scanned chunks, instead of torchdyn’s Python loop over stages and steps. The arithmetic is otherwise identical. Equivalence with the original is checked by a parity test suite: with identical weights copied across, the two implementations agree on the basis matrices, the projected initial condition, the RK4 trajectory, the residual and its gradient, and the 2D pipeline is pinned by scoring the NeuSA authors’ released Marmousi checkpoint at its reported error. The parallel scans of PaPi are the associative\_scan primitive of JAX, with the propagator applied mode by mode.

How much of the speedup is the framework? Table 5 compares, on every problem and the same hardware, the PyTorch NeuSA (torchdyn RK4), the JAX RK4 rollout and PaPi, with the time per optimiser step, since the step counts differ on some problems. The JAX RK4 rollout is 14–20 times faster per step than torchdyn on the 1D problems, whose per-step cost in PyTorch is dominated by the roughly $4 n _ { t }$ dependent kernel launches of the rollout (800 for sine-Gordon, 4,000 for KdV), and 1–5 times faster on the 2D and 3D problems, where each RK4 stage is a large tensor contraction and the framework overhead is negligible (1.7× on 3D Layers, 4–5× on the 2D problems). The remaining factor, RK4 in JAX against PaPi in JAX, isolates the integrator: 56× on sine-Gordon, 10× on 2D Layers, 4× on 2D Burgers, and 2–3× on Marmousi and 3D Layers, where a Picard sweep costs a sizeable fraction of an RK4 rollout because the state is large enough for the GPU to be saturated either way. On KdV the per-step factor is 210×, the largest of all, because the stiff dispersion forces the RK4 rollout through 1,001 steps while the exact propagator makes the cost of a sweep independent of the stiffness.

![](images/12553d2cd3e094bc739069e830b095c69fa6f561682b7abcef274e63063b790b.jpg)  
Figure 4: rMSE of the trained PaPi models after k cold Picard–Duhamel sweeps from the linear flow, mean over five seeds. Every problem but KdV is converged by eight sweeps.

The JAX RK4 rollout also reproduces the accuracy of the PyTorch NeuSA within seed variability (Table 5), and on the four benchmarks where the authors’ code reproduces their paper, both do; the JAX rollout is slightly more accurate on sine-Gordon, which we attribute to the double-precision basis assembly.

Table 5: Same model, three integrators, same RTX 5090: mean over five seeds of rMSE, optimiser steps, training time and time per step. “NeuSA” is the authors’ PyTorch code (torchdyn RK4); “RK4 (JAX)” our port of the same rollout; both ratios are of times per step.
<table><tr><td rowspan="2">Problem</td><td colspan="3">NeuSA (PyTorch)</td><td colspan="3">RK4 (JAX)</td><td colspan="3">PaPi (JAX)</td><td colspan="2">speedup / step</td></tr><tr><td>rMSE</td><td>steps</td><td>TT (s)</td><td>rMSE</td><td>steps</td><td>TT (s)</td><td>rMSE</td><td>steps</td><td>TT (s)</td><td>torch/JAX</td><td>RK4/PaPi</td></tr><tr><td>2D Layers</td><td>6.9e-2</td><td>2,000</td><td>508</td><td>9.1e-2</td><td>2,000</td><td>117</td><td>6.8e-2</td><td>2,000</td><td>11.8</td><td>4.4</td><td>9.9</td></tr><tr><td>Marmousi</td><td>2.7e-1</td><td>2,000</td><td>604</td><td>1.9e-1</td><td>5,000</td><td>290</td><td>1.5e-1</td><td>2,500</td><td>66</td><td>5.2</td><td>2.2</td></tr><tr><td>3D Layers</td><td>7.6e-3</td><td>2,000</td><td>577</td><td>5.3e-3</td><td>4,000</td><td>676</td><td>1.8e-2</td><td>4,000</td><td>251</td><td>1.7</td><td>2.7</td></tr><tr><td>2D Burgers</td><td>5.3e-2</td><td>200</td><td>119</td><td>2.5e-2</td><td>1,000</td><td>123</td><td>2.8e-2</td><td>1,000</td><td>29.6</td><td>4.8</td><td>4.2</td></tr><tr><td>Sine-Gordon</td><td>9.0e-4</td><td>1,000</td><td>234</td><td>7.3e-4</td><td>1,000</td><td>16.7</td><td>5.6e-4</td><td>1,000</td><td>0.3</td><td>14</td><td>56</td></tr><tr><td>KdV</td><td>3.6e-3</td><td>1,000</td><td>1322</td><td>3.3e-3</td><td>1,000</td><td>67.0</td><td>1.2e-3</td><td>10,000</td><td>3.2</td><td>20</td><td>210</td></tr></table>

Stability. The explicit rollout constrains the discretisation in a way PaPi does not. On the padded Marmousi grid the largest eigenvalue of the wave operator gives $\Delta t \omega _ { \mathrm { m a x } } \approx 2 . 2$ at NeuSA’s ∆t = 0.01, just inside RK4’s stability limit of about 2.8 on the imaginary axis; a coarser time grid produces NaN rather than a worse answer. On KdV the same limit caps the number of modes at 81 for $\Delta t = 1 0 ^ { - 3 }$ , and doubling the resolution requires eight times more steps. For the inverse Burgers problem the viscous term gives $\Delta t \lambda _ { \mathrm { m a x } } \approx 2 4$ on the data grid, so the RK4 baseline of our port has to sub-step the rollout by a factor of ten to twenty. In all three cases the exact propagator of PaPi is unconditionally stable, and its time grid is chosen to resolve the nonlinear correction alone; with the exponential trapezoidal rule the training grid can even be coarser than the evaluation grid, as on 2D and 3D Layers.

## F SINE-GORDON ABLATION STUDY

Two parts of our method (the solver and training) are explored in this section considering a single controlled problem (the Sine-Gordon equation). We observe that exact linear propagation is what makes the fixed-point iteration converge at all, and training accuracy is set mainly by the step size, with Adam-type methods at $\eta _ { 0 } \approx 0 . 0 1$ reaching errors below $1 0 ^ { - 3 }$

## F.1 SETUP

We solve $u _ { t t } = u _ { x x } - 1 0$ sin u on $[ - 4 , 4 ] \times [ 0 , 3 ]$ with $u ( \pm 4 , t ) = 0 , u _ { t } ( x , 0 ) = 0$ , and a Gaussian pulse $u ( x , 0 ) = e ^ { - x ^ { 2 } / ( 2 \sigma ^ { 2 } ) } / ( \sqrt { 2 \pi } \sigma ) , \sigma = 0 . 1$ , which splits into two counter-propagating waves that the nonlinearity keeps reshaping. A sine expansion in $M = 2 0 1$ modes enforces the boundary conditions; the nonlinearity is evaluated at the nodes without dealiasing. Writing $y = ( a , \dot { a } )$ for the coefficients and velocities gives $\dot { y } = L y + N ( y )$ , with wave operator $L$ and nonlinear force $N$ We report the relative Frobenius error $E = \| U - \mathrm { \ ' { U } } _ { \star } \| _ { \mathrm { F } } / \| U _ { \star } \| _ { \mathrm { F } }$ over the time–space grid against a reference $U _ { \star }$ from adaptive eighth-order Dormand-Prince integration of the same system (accurate to $6 \times 1 0 ^ { - 1 2 } )$ . Since the spatial discretization is shared, E measures only time-integration and, later, learning error.

## F.2 SOLVER: EXACT LINEAR PROPAGATION DRIVES CONVERGENCE

The two fixed-point iterations we compare differ in what they leave to quadrature. Plain Picard iteration integrates the whole right-hand side, whereas Picard–Duhamel (PD) propagates the linear part exactly through $\Phi ( t ) = e ^ { t \bar { L } }$ and iterates only on the nonlinear force:

$$
y _ { \mathrm { P } } ^ { k + 1 } ( t ) = y _ { 0 } + \int _ { 0 } ^ { t } [ L y _ { \mathrm { P } } ^ { k } + N ( y _ { \mathrm { P } } ^ { k } ) ] d s , \qquad y _ { \mathrm { P D } } ^ { k + 1 } ( t ) = \Phi ( t ) y _ { 0 } + \int _ { 0 } ^ { t } \Phi ( t - s ) N ( y _ { \mathrm { P D } } ^ { k } ( s ) ) d s ,\tag{27}
$$

starting from $y _ { 0 }$ and $\Phi ( t ) y _ { 0 }$ , respectively. A single PD sweep is the classical Duhamel approximation. In both cases the integrals are evaluated with the trapezoidal rule and parallel prefix sums, without damping or time windowing. We run both iterations on 101, 201, and 401 time nodes for $K \in$ $\{ 1 , \ldots , 3 2 \}$ sweeps.

Table 6: Only PD converges. Relative error $E$ at fixed sweep budgets; times are medians of 20 warmed float64 GPU solves, excluding compilation.
<table><tr><td>Time nodes</td><td>Picard,  $K = 1 6$ </td><td>Duhamel,  $K = 1$ </td><td> $\mathrm { P D } , K = 1 6$ </td><td>PD time (ms)</td></tr><tr><td>101</td><td> $7 . 5 5 \times 1 0 ^ { 1 7 }$ </td><td>5.025</td><td> $8 . 6 3 \times 1 0 ^ { - 4 }$ </td><td>3.50</td></tr><tr><td>201</td><td> $7 . 0 4 \times 1 0 ^ { 1 7 }$ </td><td>5.021</td><td> $2 . 1 6 \times 1 0 ^ { - 4 }$ </td><td>5.10</td></tr><tr><td>401</td><td> $6 . 8 8 \times 1 0 ^ { 1 7 }$ </td><td>5.018</td><td> $5 . 3 9 \times 1 0 ^ { - 5 }$ </td><td>7.76</td></tr></table>

The outcome is not close (Table 6). Plain Picard never produces a usable solution at any of these budgets; by $K = 1 6$ its error has grown past $1 0 ^ { 1 7 }$ . A single Duhamel sweep at least stays bounded, but it is still off by a factor of five. Repeating the Duhamel step is what closes the gap: on the 401-node grid, PD reaches $E = 5 . 3 9 \times \mathrm { { \bar { 1 0 } } ^ { - 5 } }$ in 7.8 ms. Figure 5 shows why. Picard loses accuracy almost as soon as it starts, whereas each PD sweep pushes the accurate region further along the time axis, until after 16 sweeps it covers the whole horizon. Beyond that point additional sweeps only move the fifth significant digit, and what error remains is due to the quadrature: halving $\Delta t$ cuts it by a factor of four, with observed orders of 2.002 and 2.000. Nor is this a matter of PD starting from a better guess: crossing both updates with both starting paths, PD from the constant path reaches the same $5 . 3 9 \times 1 0 ^ { - 5 }$ , and Picard from the free wave still fails (Figure 7a).

![](images/1184447a97886c8b42188a05a846041fd67389a8855d709f633d46b8c53d4bd2.jpg)  
Figure 5: Each PD sweep extends the accurate time window, while Picard fails early at every budget. (a,b) Error at each time t on the 401-node grid (gray: above one); the bottom PD row is the Duhamel solution. (c) Once converged $( K = 3 2 )$ , PD error falls at exactly second order in $\Delta t .$

## F.3 TRAINING: STEP SIZE DOMINATES THE CHOICE OF OPTIMIZER

For the training study we replace the exact force by a learned one, $0 . 1 N _ { \theta } ( a )$ , where $N _ { \theta }$ is a two-layer LeakyReLU network of width 804 with 971,433 parameters. Training rolls the model out with RK4 on 201 time nodes and minimizes the mean squared residual of $u _ { t t } - u _ { x x } + 1 0$ sin u along that rollout, optionally with a penalty $\frac { \lambda } { 2 } \lVert \boldsymbol { \theta } \rVert _ { 2 } ^ { 2 }$ added; gradients flow through the entire rollout. Every run consists of 1,000 full-batch float32 updates and is repeated with three initialization seeds. We report the mean and sample standard deviation over seeds, which should be read as the spread due to initialization rather than as a confidence interval. The comparison covers Adam, AdamW (weight decay $1 0 ^ { - 4 } )$ RMSProp, momentum SGD, and Muon at initial learning rates $\eta _ { 0 } \in \{ 0 . 0 0 1 , 0 . 0 1 , 0 . 0 2 \}$ with decay $0 . 9 9 9 ^ { k }$ ; Polyak SGD with step caps $\{ 0 . 0 1 , 0 . 1 , 1 \}$ ; and L-BFGS with a zoom line search. A separate sweep adds $\mathsf { \bar { \lambda } } \in \{ 1 0 ^ { - 6 } , 1 0 ^ { - 4 } , 1 0 ^ { - 2 } \}$ to Adam at $\eta _ { 0 } = 0 . 0 1$ . Altogether this gives 22 configurations and 66 runs.

Step size matters more than the optimizer family. Only the Adam-type methods get below $1 0 ^ { - 3 }$ , and they do so only once the learning rate is large enough (Figure 6a). The most accurate configuration is Adam at $\eta _ { 0 } = 0 . 0 1$ , with $E \bar { = } ( 7 . 1 7 \pm \bar { 0 . 2 7 } ) \times \bar { 1 } 0 ^ { - 4 } ,$ ; AdamW lands within its seed spread. Drop the rate to $\eta _ { 0 } = 0 . 0 0 1$ and both become thirty times worse. Muon reaches $8 . 2 \times 1 0 ^ { - 3 }$ and RMSProp $3 . 7 \times 1 0 ^ { - \dot { 2 } }$ , while momentum SGD and Polyak SGD never get below 0.5 and improve only as their step is allowed to grow (panel c). L-BFGS stalls around $7 \times 1 0 ^ { - 2 }$ , and in float32 22% of its line searches terminate without satisfying both acceptance criteria.

Explicit regularization only hurts. Since the objective is a physics residual on a single trajectory, there is nothing to overfit, and an L2 penalty can only pull the solution away from the one that satisfies the equation. That is what happens: $\lambda = { \overset { \cdot } { 1 0 ^ { - 6 } } } , 1 { \overset { \cdot } { 0 ^ { - 4 } } }$ , and $1 0 ^ { - 2 }$ raise the error by factors of 1.09, 13.9, and 374 (Figure 6b). The decoupled weight decay of $1 0 ^ { - 4 }$ in AdamW is mild enough to do no harm.

Cost is nearly uniform. All first-order methods take about 24 s per run (23–26 ms per update), so the three orders of magnitude separating the configurations in Figure 6(d) are not bought with compute. The exception is L-BFGS, whose line search costs 1,662 gradient evaluations and 46 s.

## F.4 ROBUSTNESS CHECKS

Three questions remain: how much of the training error is really integration error, whether the weaker optimizers were merely under-tuned or under-trained, and whether the L-BFGS failures are a float32 artifact. Each gets a targeted control on the same problem and architecture.

![](images/96e52858c0defb4d4e58d4d77df5587f22441b90a5d2ed60bfe2279d43ab95b1.jpg)

![](images/8faf099f04b8cb33710e1fb2ada952c763b53c0478118ffee83e6baa74593b35.jpg)

![](images/0068f30be5747f2e2de76d9a0d1730125dcb7c945cd5d1efebd9293111cdaf2f.jpg)

(d) Decades of error at equal cost  
![](images/51257e6a32402529a2aa13805bd2d038a736be6cfc7e109377685f1a5def735a.jpg)  
Figure 6: Step size and optimizer family determine accuracy; regularization only adds error. (a) Mean error per optimizer and initial learning rate. (b) L2 penalty for Adam at $\eta _ { 0 } = 0 . 0 1$ . (c) Polyak SGD step-size cap. (d) All configurations against training time; the star marks Adam at $\eta _ { 0 } = 0 . 0 1$ . Error bars are standard deviations over three seeds.

Integration error. Even with the exact force, RK4 on the training grid already incurs $E =$ $4 . 9 3 \times 1 0 ^ { - 4 }$ in float64. Refining to four substeps per interval brings this down to $1 . { \overset { \cdot } { 9 4 } } \times 1 0 ^ { - 6 }$ , but in float32 the same refinement stalls at $2 . 6 \times \dot { 1 } 0 ^ { - 4 }$ (Figure 7b). The learned models are affected the same way: freezing the trained Adam and AdamW models and evaluating them with finer RK4 lowers their error from $\mathbf { 7 . 3 3 } \times 1 0 ^ { - 4 } \mathrm { t o } \ 5 . 6 \mathbf { - } 5 . 7 \times 1 0 ^ { - 4 }$ (panel c). A good part of the reported training error is therefore integration error, not a deficiency of the learned force.

Tuning and budget. To check whether the weaker methods were held back by their step-size grid or by the training budget, we extended the grids, picked the best setting for each method by its final training residual (never by the reference error), and continued training to 5,000 updates (Figure 8a,b). For Muon the optimum lies inside the grid at $\eta _ { 0 } = 0 . 0 2$ , and with the longer budget it catches up with Adam, at $5 . 9 6 \times 1 0 ^ { - 4 }$ against $6 . 2 7 \times 1 0 ^ { - 4 }$ . Momentum SGD and Polyak SGD, by contrast, keep improving right up to the largest steps we tested and still only reach $2 { - } 4 \dot { \times } 1 0 ^ { - 2 }$ . The ranking at 1,000 updates therefore says as much about how fast each method converges as about where it ends up.

Precision. Rerunning L-BFGS in float64 from the same initial parameters eliminates the line-search failures entirely, yet the error only improves to $( 4 . 3 5 \pm 1 . 4 2 ) \times \mathrm { { \dot { 1 } } 0 ^ { - 2 } }$ , and for one seed it gets worse (Figure 8c). Precision explains the failed line searches, but not why L-BFGS does poorly here.

(a) PD, either start  
![](images/f0552e1833c4bef2eb88205fb677e809501feb1c1ace278db287b6f3ae88d4cb.jpg)

![](images/e4eba2493b8c5529be39fd52339ec2aea5f0e35055baed8ac7ae0e624e9f6dbb.jpg)

![](images/e92715e4b44b260d3c16ffeb3f3bd55b7c7197b4346d4ca1d601809d6a134c76.jpg)  
Figure 7: The PD gain comes from the update, and part of the training error comes from time integration. (a) Both updates from both initial paths (401 nodes). (b) Exact-force RK4 in float32 and float64. (c) Frozen Adam and AdamW models evaluated with finer RK4. Panels $^ { ( \mathrm { b , c } ) }$ score the original 201 nodes.

![](images/b744f40ec82b006bb26b1992ca1af2b6dcd4a006b4099f081ec198c640064fd3.jpg)

![](images/fd05ff84a3491b78a489238c0eede1d2ab57c4092f1a415f097cd3a46a447b86.jpg)

![](images/305265985599fcca15e80ccd9e8c1ecf3a01c7669e5131291115816ca6a0f6d0.jpg)  
Figure 8: Given enough updates Muon matches Adam, while float64 does not rescue L-BFGS. (a) Extended step-size ranges at 1,000 updates. (b) Residuals continued to 5,000 updates: seed median with min–max band, smoothed by a 51-update running median (dotted line: original budget). (c) L-BFGS in float32 and float64 from identical initial parameters.

## F.5 CONCLUSION

Exact linear propagation is the decisive design choice. It costs a few milliseconds, and it turns an iteration that fails at every budget we tried into one that converges in 16 sweeps to the quadrature floor, leaving only the second-order error set by the time step. On the training side, Adam or AdamW at $\eta _ { 0 } \approx 0 . 0 1$ with no explicit regularization is the most accurate choice at the standard budget, and the gaps between optimizers say more about how quickly they converge than how far they can go, as Muon’s catching up at 5,000 updates shows. The best learned models settle around $6 \times 1 0 ^ { - 4 }$ the same order as the exact-force RK4 error on the training grid $( 4 . 9 \times 1 0 ^ { - 4 } )$ and ten times the converged PD solver. Further gains are therefore more likely to come from the time integrator used during training than from more optimizer tuning, which argues for pairing the learned force with the Picard–Duhamel solver itself.

## G OTHER IMPLEMENTATION DETAILS

## G.1 PARALLEL SCANS

A parallel scan is an efficient way of evaluating a recurrence of the form

$$
h _ { t } = A _ { t } h _ { t - 1 } + b _ { t } , \qquad t = 1 , \ldots , T ,\tag{28}
$$

without explicitly stepping through the sequence one element at a time. The key observation is that each step can be represented as an affine transformation

$$
h \mapsto A _ { t } h + b _ { t } ,
$$

and that affine transformations can be composed associatively. Writing

$$
( A _ { 2 } , b _ { 2 } ) \circ ( A _ { 1 } , b _ { 1 } ) = ( A _ { 2 } A _ { 1 } , ; A _ { 2 } b _ { 1 } + b _ { 2 } ) ,
$$

the entire sequence can therefore be evaluated using a standard parallel prefix-scan algorithm.

Conceptually, the sequential computation

$$
h _ { 0 } \longrightarrow h _ { 1 } \longrightarrow h _ { 2 } \longrightarrow h _ { 3 } \longrightarrow \cdots \longrightarrow h _ { T }
$$

is replaced by a tree of compositions:

$$
\begin{array} { c c c c c c } { { x _ { 1 } } } & { { x _ { 2 } } } & { { x _ { 3 } } } & { { x _ { 4 } } } & { { x _ { 5 } } } & { { x _ { 6 } } } & { { x _ { 7 } } } & { { x _ { 8 } } } \\ { { \downarrow } } & { { \downarrow } } & { { \downarrow } } & { { \downarrow } } & { { \downarrow } } & { { \downarrow } } & { { \downarrow } } & { { \downarrow } } \\ { { x _ { 1 } \circ x _ { 2 } } } & { { x _ { 3 } \circ x _ { 4 } } } & { { x _ { 5 } \circ x _ { 6 } } } & { { x _ { 7 } \circ x _ { 8 } } } & { { } } & { { } } \\ { { } } & { { } } & { { } } & { { } } & { { } } & { { } } \\ { { \left( x _ { 1 } \stackrel { \times } { \circ } x _ { 2 } \right) \circ \left( x _ { 3 } \stackrel { \times } { \circ } x _ { 4 } \right) } } & { { \left( x _ { 5 } \stackrel { \times } { \circ } x _ { 6 } \right) \circ \left( x _ { 7 } \stackrel { \times } { \circ } x _ { 8 } \right) } } & { { } } & { { } } & { { } } \\ { { } } & { { } } & { { } } & { { } } & { { } } & { { \times } } & { { } } \\ { { \times } } & { { { \ x _ { 1 } \circ x _ { 2 } \circ x _ { 3 } \circ x _ { 4 } \circ x _ { 5 } \circ x _ { 6 } \circ x _ { 7 } \circ x _ { 8 } } } } & { { } } & { { } } & { { } } \end{array}\tag{29}
$$

so that the computational depth is reduced from $\mathcal { O } ( T )$ to ${ \mathcal { O } } ( \log T )$ , at the cost of $\mathcal { O } ( T )$ total work.

This structure is particularly well suited to GPUs. Although the scan performs more operations than a strictly sequential recurrence, many of these operations are independent and can be executed simultaneously across thousands of GPU threads. Moreover, the computation consists primarily of regular tensor operations with predictable memory access patterns, allowing modern GPU kernels to exploit massive parallelism and high memory bandwidth. In practice, this can make a parallel scan substantially faster than a sequential recurrence for long sequences, even when the underlying recurrence itself is unchanged.

The same idea extends beyond scalar or diagonal recurrences to more general associative state-update operators. This is one reason parallel scans are useful for state-space models and other sequence architectures: the temporal structure is retained, while the sequential dependency that normally prevents efficient GPU utilization is replaced by an associative reduction.

## G.2 EXPONENTIAL TRAPEZOIDAL QUADRATURE

The trapezoidal rule introduced in (12) approximates the integrand in (11), $\Phi _ { s } ^ { t } F ( X ^ { ( k ) } ( s ) )$ , by a piecewise-linear function. However in the setting when $\Phi _ { s } ^ { t }$ has a closed formula, for instance when $A ( t ) = A$ is constant and $\Phi _ { s } ^ { t } = \mathrm { e x p } ( A ( t - \bar { s } ) )$ , it is often preferable to instead approximate $F ( X ^ { ( k ) } ( s )$ by a piecewise-linear function $\iota ( s )$ , and evaluate $\textstyle \int _ { 0 } ^ { t } \Phi _ { s } ^ { t } \iota ( s )$ ds exactly. This leads to the exponential trapezoidal rule. The construction below follows the standard framework of exponential quadrature; see, for example, Hochbruck & Ostermann (2010).

We first recall the $\varphi \cdot$ functions used in exponential integration. We may define them by their Taylor expansion

$$
\varphi _ { \ell } ( z ) : = \sum _ { n = 0 } ^ { \infty } \frac { z ^ { n } } { ( n + \ell ) ! }
$$

With most relevance to our setting are the first three φ-functions,

$$
\varphi _ { 0 } ( z ) = e ^ { z } , \qquad \varphi _ { 1 } ( z ) = \frac { e ^ { z } - 1 } z , \qquad \varphi _ { 2 } ( z ) = \frac { e ^ { z } - 1 - z } { z ^ { 2 } } .
$$

The Taylor expansion allows to define the $\varphi$ functions also for matrices, in a similar way to the definition of matrix exponential.

Now let $\Phi _ { s } ^ { t } = \exp ( A ( t - s ) )$ ) then it the definition of $\varphi$ implies that

$$
\int _ { 0 } ^ { t } \Phi _ { s } ^ { t } ( a ( t - s ) + b s ) \mathrm { d } s = \left( \varphi _ { 1 } ( t A ) - \varphi _ { 2 } ( t A ) \right) a + \varphi _ { 2 } ( A ) b .\tag{30}
$$

Letting $P = P _ { j } = e ^ { h A } , B _ { 0 } = \varphi _ { 1 } ( h A ) - \varphi _ { 2 } ( h A )$ and $B _ { 1 } = \varphi _ { 2 } ( h A )$ , (30) leads to the exponential trapezoidal scheme, analog to (12):

$$
\bar { X } _ { \mathrm { E T } , j } ^ { ( k + 1 ) } : = \Phi _ { 0 } ^ { t _ { j } } X _ { 0 } + \varepsilon h \sum _ { i = 0 } ^ { j - 1 } \Phi _ { t _ { i + 1 } } ^ { t _ { j } } \left( B _ { 0 } g _ { \mathrm { E T } , i } ^ { ( k ) } + B _ { 1 } g _ { \mathrm { E T } , i + 1 } ^ { ( k ) } \right) ,\tag{31}
$$

where $g _ { \mathrm { E T } , i } ^ { ( k ) } = F ( \bar { X } _ { \mathrm { E T } , i } ^ { ( k ) } )$ . Finally, and most crucially, this integration scheme also enjoys a recursion similar to that of the trapezoidal method, that allows it to be calculated with a parallel scan:

$$
\overline { { X } } _ { \mathrm { E T } , j + 1 } ^ { ( k + 1 ) } = P \overline { { X } } _ { \mathrm { E T } , j } ^ { ( k + 1 ) } + b _ { \mathrm { E T } , j } ^ { ( k ) } , \quad \mathrm { w h e r e } \quad \quad b _ { \mathrm { E T } , j } ^ { ( k ) } : = \varepsilon \left( B _ { 0 } g _ { E T , j } ^ { ( k ) } + B _ { 1 } g _ { E T , j + 1 } ^ { ( k ) } \right) .\tag{32}
$$