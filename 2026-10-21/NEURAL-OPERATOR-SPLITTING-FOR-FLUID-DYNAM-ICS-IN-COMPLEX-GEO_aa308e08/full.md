# NEURAL OPERATOR SPLITTING FOR FLUID DYNAM-ICS IN COMPLEX GEOMETRY

Jianxiang Li<sup>∗</sup>   
Peking University   
lijianxiang0619@stu.pku.edu.cn

Hong Chul Nam<sup>∗</sup> New York University hn2557@nyu.edu

Xi Deng California Institute of Technology xideng94@caltech.edu

Anima Anandkumar California Institute of Technology anima@caltech.edu

## ABSTRACT

Solving dynamical systems in complex and diverse geometries remains challenging for both numerical solvers and learning-based surrogate models. We propose Neural Operator Splitting, a framework for learning fluid dynamics while preserving the physical structure of fractional-step numerical schemes. Instead of learning the full-step evolution monolithically, we decompose each time step into physically interpretable steps, and learn each operator separately using intermediate states from the underlying numerical solver. Guided by the linearity properties of the fractional steps, we introduce a conditional-linear Fourier neural operator (CL-FNO) whose linear state mappings are modulated by nonlinear conditioning on inputs such as boundary geometry. We further introduce two complementary strategies, symmetry sampling and corrective cycling, to mitigate error accumulation during autoregressive rollout. Experiments on 2D and 3D fluid flows with varying geometries and initial conditions show that our full model reduces mean relative $\breve { L } ^ { 2 }$ rollout error by 68% in 2D and 40% in 3D compared with a single full-step neural operator. At comparable error levels, neural-operator inference is approximately $\mathrm { 1 0 ^ { 3 } }$ times faster per step than the Monte Carlo solver.

## 1 INTRODUCTION

Traditional numerical solvers for nonlinear dynamics in diverse geometries often decompose the system into operators, enabling faster predictions (MacNamara & Strang, 2017). Neural operators (Li et al., 2021) have been proposed as learning-based methods for solving complex PDEs by learning function-to-function mappings. However, most neural operators predict the next state with a single operator that must capture the full physical evolution, making generalization to unseen conditions and long-horizon rollout more challenging. Furthermore, existing operators do not respect the possible underlying structure of such an operator and do not fully exploit physical structure.

Fluid dynamics provides a natural setting for studying this issue, particularly in complex and diverse geometries, where both numerical solvers and learning-based surrogate models face distinct challenges (Sugimoto et al., 2024; Li et al., 2023b). Numerical methods often require specialized meshing or boundary treatment, while learning-based surrogates must generalize across unseen geometries.

Many traditional solvers for the Navier–Stokes equations solve the governing partial differential equations (PDEs) via operator splitting, in which a complex single time step is decomposed into advection, diffusion, and pressure projection (Chorin, 1968; Orszag & Kells, 1980). Advection and diffusion evolve the velocity field, while pressure projection enforces the divergence-free constraint on the velocity. For static geometries with homogeneous boundary conditions, only advection is nonlinear; diffusion and projection are linear in the evolving state. A monolithic next-step operator must rediscover all three from data, along with the linear structure of two of them.

![](images/655e753169dda2ac9315ac74fb7f882e6f648f8c3dab0291e74aaa58a6dfb8f7.jpg)  
(a)

3D Geo.  
![](images/006be2cf9b7c69147c4e7b4593360c921b00451eedfa590c30128a0d605ed136.jpg)

Init.  
Ref. (step 40)  
Ours (step 40)  
![](images/0f9876250a6cca9f933e694b61e06bbc5601731089171797501771be1ebb26ba.jpg)  
(b)  
Figure 1: (a) Improved accuracy. We compare our neural operator splitting approach against a single operator with a matched network parameter count. Across four datasets, our method achieves ${ \bf a } 2 - 3 \times$ reduction in relative $L ^ { 2 }$ error. (b) 3D fluid rollout visualization. We visualize the predicted 3D velocity field using particle trajectories on a representative slice of the domain. Across unseen geometries, our model produces rollout dynamics that remain spatially close to the reference solution.

The direct way to use this decomposition is to learn one operator per substep, supervised by the intermediate states the solver already produces. Table 1 shows that this alone reduces rollout error by 10.7% in 2D and 23.8% in 3D over a single full-step operator. But the three substeps differ in what they demand of a learned model, and supervising them as three interchangeable regression problems leaves that difference unused.

We therefore propose neural operator splitting as shown in Figure 2, which learns each substep with an architecture matched to its structure. Its central component is a conditional-linear Fourier neural operator (CL-FNO), a two-branch design that separates a linear state-evolution branch from a nonlinear conditioning branch that incorporates geometry. For fixed geometry, the diffusion and projection operators are exactly linear in the evolving state, matching their true structure, while advection recovers its nonlinearity by receiving the state through the conditioning branch as well.

We further introduce two strategies for autoregressive rollout. Symmetry sampling resamples an element of the problem’s spatial symmetry group at each operator application, disrupting the temporal coherence of prediction errors and recovering most of the benefit of full group averaging at a small fraction of the cost. Corrective cycling trains an ordered sequence of operators each on states carrying the errors of its predecessors, without backpropagating through the rollout, and cycles through them during rollout to absorb distribution shift.

We evaluate on 2D and 3D incompressible flows with diverse geometries and initial conditions, testing generalization to unseen geometries and long rollout horizons. We primarily use a velocity-based Monte Carlo fluid solver (Sugimoto et al., 2024) that handles varied geometries without body-fitted volume meshing and exposes the intermediate states needed for substep supervision. Additional spectral and Basilisk benchmarks demonstrate the framework’s applicability to deterministic solvers.

## Our contributions include

• Solver-supervised fractional-step operators. We learn the advection, diffusion, and pressureprojection substeps of a solver as separate neural operators, supervised directly by the intermediate states of the corresponding solver trajectory. On 2D and 3D flows with varying geometries, this reduces mean rollout error by 11% and 24% relative to a single full-step operator of the same architecture (Table 1).

• A conditional-linear operator architecture. We separate the operator into a linear stateevolution branch and a nonlinear conditioning branch, allowing the learned map to remain exactly linear in the evolving state when the conditioning depends only on geometry. This matches the structure of diffusion and pressure projection, while leaving advection free to depend on the state. The design reduces rollout error by 18–61% across settings (Table 2).

![](images/ddb1e1c64f0a2629f66e444d0d176c2ae5c1ef934eb67df2771d0264848686e5.jpg)  
Figure 2: Overview of learned operator splitting. (a) Geometry-conditioned advection, diffusion, and projection operators advance the velocity field from frame $u ^ { ( n ) }$ to frame $u ^ { ( n + 1 ) }$ . (b) Monte Carlo solver targets supervise each stage during training; no solves are required at inference. (c) Each operator uses a geometry encoder to generate gates that modulate spectral convolution blocks processing the input velocity field.

• Symmetry sampling. We resample a symmetry element from the problem’s finite symmetry group at every operator application during rollout, reducing long-horizon error by 14–51% without additional training (Table 3). Stochastic sampling recovers nearly all of the benefit of full group averaging at a fraction of the forward passes (Fig. 6), indicating that symmetry transformations help not only by enforcing equivariance but by disrupting the temporal coherence of prediction errors.

• Corrective cycling. We train an ordered sequence of operators, each on states carrying the errors of its predecessors, and cycle through them during rollout. This addresses autoregressive distribution shift without backpropagating through a rollout, reducing error by up to 21% (Table 4).

## 2 RELATED WORKS

The Fourier neural operator (FNO) (Li et al., 2021) parameterizes global interactions in Fourier space and has been widely used for learning fluid dynamics, including Navier–Stokes flows. However, the standard FNO formulation is built around regular-grid discretizations, which makes complex and varying geometries challenging.

Existing approaches differ in how geometric information is incorporated into the operator. Geo FNO (Li et al., 2023a) learns a coordinate deformation from irregular domains to regular latent grids. DAFNO (Liu et al., 2023) incorporates a smoothed characteristic function of the domain into Fourier integral layers. GINO (Li et al., 2023b) uses graph operators to couple irregular physical discretizations with a regular latent grid, where an FNO acts on the latent field. PCNO (Zeng et al., 2025) instead defines neural operators directly on point-cloud discretizations. These works primarily focus on geometric representation and generalization across domains, whereas our setting additionally emphasizes long-horizon fluid evolution across varying geometries.

Recent work has begun to explore operator splitting for decomposing complex PDE dynamics into subproblems. DOSnet (Lan et al., 2023) embeds classical operator splitting directly into a neural architecture composed of multiple learnable operator-splitting blocks. Within neural operators, Gopakumar et al. decompose PDE dynamics into learned and numerically discretized physica operators whose contributions are combined before time integration. Sun et al. uses a contextconditioned Strang-splitting backbone to combine learned global and local mechanisms for multiphysics pretraining. Another approach pretrains operators for elementary or single-physics dynamics and subsequently combines them for multiphysics systems through a learned adaptation block (Hmida et al., 2026). Serrano et al. instead compose operators from a pretrained dictionary through test-time operator splitting. In contrast, our neural operator splitting follows the solver’s fractional-step sequence and directly supervises each suboperator with consecutive intermediate states from the same solver trajectory.

A separate line of work has explored operator architectures that distinguish different input roles and structure their interactions accordingly. VarMiON (Patel et al., 2024) derives this structure from variational discretizations. Neural Green’s Operators (Melchers et al., 2026) and Neural Green’s Functions (Yoo et al., 2025) use Green-function formulations to preserve linear dependence on forcing and boundary data, while learning how the solution operator varies with PDE coefficients and domain geometry, respectively. Separately, HyperFNO (Alesiani et al., 2022) generates FNO parameters from physical PDE parameters, while Jing et al. generate parameter-dependent shifts for individual FNO layers. Our conditional-linear architecture uses FNO-generated layer-wise gates to condition the operator while preserving linearity in the evolving state.

Symmetry has been incorporated into neural PDE solvers through equivariant architectures and group-based symmetrization. G-FNO (Helwig et al., 2023) builds rotation and reflection equivariance directly into Fourier neural operator layers. Beyond architectural constraints, frame averaging (Puny et al., 2022) averages predictions over input-dependent frames, while probabilistic symmetrization (Kim et al., 2023) learns an input-conditioned distribution over group transformations. Foit et al. apply test-time group averaging to autoregressive physical surrogates, using exact averaging for finite groups and Monte Carlo estimates for continuous symmetries. Our approach uses single-sample symmetrization for finite groups and refreshes the sampled realization across operator applications during rollout to mitigate coherent error accumulation.

Finally, autoregressive neural PDE solvers face distribution shift as model predictions are recursively fed back as inputs. Pushforward training (Brandstetter et al., 2022) mitigates this shift by training on self-generated states. Our corrective cycling instead distributes adaptation to rollout-induced states across a sequence of operator instances that are cycled during rollout.

## 3 NEURAL OPERATOR SPLITTING

In this section, we introduce the Navier–Stokes problem setup, our neural operator splitting framework and a conditional linear architecture motivated by the structure of the split operators, and associated techniques for improving long-horizon rollout performance.

## 3.1 NEURAL OPERATOR SPLITTING OF THE NAVIER–STOKES EQUATIONS

We consider incompressible fluid evolution governed by

$$
\frac { \partial { \bf { u } } } { \partial t } + ( { \bf { u } } \cdot \nabla ) { \bf { u } } = - \frac { 1 } { \rho } \nabla p + \nu \nabla ^ { 2 } { \bf { u } } , \qquad \nabla \cdot { \bf { u } } = 0 .\tag{1}
$$

Here u is the velocity field, p is the pressure, t is time, $\rho$ is the fluid density, and ν is the kinematic viscosity. The operators $\bar { \nabla , \nabla . }$ , and $\nabla ^ { 2 }$ denote the spatial gradient, divergence, and Laplacian, respectively.

We use the operator splitting induced by the fractional-step solver, which decomposes one solver step into the advection, diffusion, and pressure-projection substeps:

$$
\mathbf { u } _ { * } ^ { n + 1 } = A ( \mathbf { u } ^ { n } ) , \qquad \quad \partial _ { \tau } \mathbf { u } + ( \mathbf { u } \cdot \nabla ) \mathbf { u } = 0 , \quad \tau \in [ 0 , \Delta t ] ,\tag{2}
$$

$$
\mathbf { u } _ { * * } ^ { n + 1 } = { \cal D } ( \mathbf { u } _ { * } ^ { n + 1 } ) , \qquad \partial _ { \tau } \mathbf { u } = \nu \nabla ^ { 2 } \mathbf { u } , \quad \tau \in [ 0 , \Delta t ] ,\tag{3}
$$

$$
\mathbf { u } ^ { n + 1 } = P ( \mathbf { u } _ { * * } ^ { n + 1 } ) , \qquad \nabla ^ { 2 } \overline { { p } } = \nabla \cdot \mathbf { u } _ { * * } ^ { n + 1 } , \quad \mathbf { u } ^ { n + 1 } = \mathbf { u } _ { * * } ^ { n + 1 } - \nabla \overline { { p } } .\tag{4}
$$

Here n indexes the solver time steps, $ { \mathbf { u } } ^ { n }$ denotes the velocity at time $t ^ { n } , \Delta t = t ^ { n + 1 } - t ^ { n }$ is the time-step size, and $\tau \ \mathrm { i s }$ the local time within each substep. The fields $\mathbf { u } _ { * } ^ { n + 1 }$ and $\mathbf { u } _ { * * } ^ { n + 1 }$ are the intermediate velocities after the advection and diffusion substeps, respectively, and $A , D$ , and $P$ denote the corresponding physical substep operators. The scaled pressure is $\begin{array} { r } { \overline { { p } } \doteq \frac { \Delta t } { \rho } p } \end{array}$

On solid boundaries, the velocity satisfies

$$
\mathbf { u } = \mathbf { u } _ { s } ,\tag{5}
$$

where $\mathbf { u } _ { s }$ denotes the prescribed solid-boundary velocity, with $\mathbf { u } _ { s } = \mathbf { 0 }$ for stationary boundaries. Here $\overline { { p } }$ satisfies the Neumann condition

$$
\frac { \partial \overline { { p } } } { \partial n } = \mathbf { n } \cdot \left( \mathbf { u } _ { * * } ^ { n + 1 } - \mathbf { u } _ { s } \right) ,\tag{6}
$$

where n is the outward unit normal from the fluid domain and $\partial / \partial n = { \bf n } \cdot \nabla$ denotes the normal derivative.

Equivalently,

$$
{ \bf u } ^ { n } \stackrel { A } {  } { \bf u } _ { \ast } ^ { n + 1 } \stackrel { D } { \longrightarrow } { \bf u } _ { \ast \ast } ^ { n + 1 } \stackrel { P } { \longrightarrow } { \bf u } ^ { n + 1 } .\tag{7}
$$

The solver provides $ { \mathbf { u } } ^ { n } ,  { \mathbf { u } } _ { \ast } ^ { n + 1 } ,  { \mathbf { u } } _ { \ast \ast } ^ { n + 1 }$ , and $\mathbf { u } ^ { n + 1 }$ . We therefore learn three neural operators: ${ \mathcal { F } } _ { A }$ for the advection map $A , { \mathcal { F } } _ { D }$ for the diffusion map D, and $\mathcal { F } _ { P }$ for the pressure-projection map $P .$ Each operator is trained independently using direct supervision from the corresponding solver input– output pair, with each operator learning a single physical substep. At inference, they are applied in sequence,

$$
\hat { \mathbf { u } } ^ { n + 1 } = \mathcal { F } _ { P } \circ \mathcal { F } _ { D } \circ \mathcal { F } _ { A } [ \hat { \mathbf { u } } ^ { n } ] ,\tag{8}
$$

where $\hat { \mathbf { u } } ^ { n }$ denotes the predicted velocity at time step n, and ◦ denotes operator composition.

This setup enables a direct comparison between operator splitting and a single neural operator trained directly on the full-step map $\mathbf u ^ { n } \mapsto \mathbf u ^ { n + 1 }$ in autoregressive rollout.

## 3.2 CONDITIONAL LINEAR ARCHITECTURE

We separate the neural operator into a linear branch for the evolving state u and a nonlinear branch for the conditioning input c. The conditioning branch processes c with an FNO and collects the feature fields after lifting and after each FNO layer. These features are fused pointwise by an MLP into a shared conditioning representation. For each layer ℓ of the state branch, separate heads then produce input and output gates,

$$
\Gamma _ { \ell } ^ { \mathrm { i n } } ( \mathbf { c } ) , \qquad \Gamma _ { \ell } ^ { \mathrm { o u t } } ( \mathbf { c } ) .\tag{9}
$$

These gates modulate the state features before and after the corresponding linear FNO block, so the conditioning branch affects the state evolution only through multiplicative modulation.

Let $\mathbf { z } _ { \ell }$ denote the input feature to layer ℓ of the state branch, and define its linear FNO block as

$$
\begin{array} { r } { L _ { \ell } ( \mathbf { z } _ { \ell } ) = K _ { \ell } ( \mathbf { z } _ { \ell } ) + W _ { \ell } ( \mathbf { z } _ { \ell } ) , } \end{array}\tag{10}
$$

where $K _ { \ell }$ is a bias-free spectral convolution and $W _ { \ell }$ is a pointwise linear map. The corresponding state update is

$$
\begin{array} { r } { \mathbf { z } _ { \ell + 1 } = \Gamma _ { \ell } ^ { \mathrm { o u t } } ( \mathbf { c } ) \odot L _ { \ell } \big ( \Gamma _ { \ell } ^ { \mathrm { i n } } ( \mathbf { c } ) \odot \mathbf { z } _ { \ell } \big ) , } \end{array}\tag{11}
$$

where ⊙ denotes pointwise multiplication.

Therefore, for fixed c, the resulting neural operator $\mathcal { F } ( \mathbf { u } ; \mathbf { c } )$ is linear in the evolving state u.

This design reflects the operator structure of the split dynamics. Diffusion,

$$
\partial _ { \tau } \mathbf { u } = \nu \nabla ^ { 2 } \mathbf { u } ,\tag{12}
$$

and pressure projection,

$$
\nabla ^ { 2 } \overline { { p } } = \nabla \cdot \mathbf { u } , \qquad P ( \mathbf { u } ) = \mathbf { u } - \nabla \overline { { p } } ,\tag{13}
$$

are linear in u for fixed geometry. Their conditioning branches therefore depend only on geometric representations and spatial coordinates.

Advection is nonlinear overall,

$$
\partial _ { \tau } \mathbf { u } + ( \mathbf { u } \cdot \nabla ) \mathbf { u } = 0 ,\tag{14}
$$

but is linear in the advected field when the advecting velocity is held fixed. In self-advection, u serves as both the advected field and the advecting velocity, so we additionally include u in the conditioning input c. Since advection is the only nonlinear component of the split dynamics considered here, the same state-dependent conditioning is used for the single-step operator.

## 3.3 SYMMETRY SAMPLING

We consider finite spatial symmetry groups that preserve the governing equations under the corresponding transformation of the spatial domain and vector-valued flow fields. Specifically, we use the dihedral group $D _ { 4 }$ , consisting of the rotations and reflections of the square, for the 2D problems, and the full octahedral group $O _ { h }$ , consisting of the rotational and reflection symmetries of the cube, for the 3D problem. We denote either group by G, and let $\mathcal { T } _ { g }$ denote the action of $g \in G$ on spatially defined inputs and outputs, including the corresponding transformation of vector components for vector-valued fields.

For a learned operator ${ \mathcal { F } } _ { : }$ its symmetry-transformed operator under $g$ is

$$
\mathcal { F } ^ { ( g ) } = \mathcal { T } _ { g } ^ { - 1 } \circ \mathcal { F } \circ \mathcal { T } _ { g } ,\tag{15}
$$

where $\mathcal { T } _ { g } ^ { - 1 }$ denotes the inverse symmetry action.

Rather than using the same symmetry transform throughout a rollout, we independently sample

$$
g _ { k } \sim \operatorname { U n i f } ( G )\tag{16}
$$

at each operator application $k ,$ where Unif(G) denotes the uniform distribution over $G .$ . At each application, the sampled $g _ { k }$ is used to form the corresponding transformed version of the learned operator.

Beyond the commonly discussed train–rollout distribution shift, autoregressive prediction can also amplify correlated approximation errors, as repeatedly applying the same operator realization to successive states may produce spatially aligned errors that accumulate coherently over time. Random symmetry sampling varies the symmetry transform across applications and can mitigate this coherent error accumulation.

## 3.4 CORRECTIVE CYCLING

Corrective cycling uses an ordered sequence of M learned operators, $( \mathcal { F } _ { 1 } , \ldots , \mathcal { F } _ { M } )$ . Let $\mathbf { u } _ { k }$ and $\hat { \mathbf { u } } _ { k }$ denote the ground-truth and predicted states at operator application k, respectively. Starting from $\hat { \mathbf { u } } _ { k } = \mathbf { u } _ { k }$ , we recursively define

$$
\hat { \mathbf { u } } _ { k + j } = \mathcal { F } _ { j } \big ( \hat { \mathbf { u } } _ { k + j - 1 } \big ) , \qquad j = 1 , \dotsc , M .\tag{17}
$$

The operators are trained sequentially. After $\mathcal { F } _ { 1 } , \ldots , \mathcal { F } _ { j - 1 }$ have been trained and frozen, the training inputs for ${ \mathcal { F } } _ { j }$ are generated offline by recursively applying these preceding operators starting from ground-truth states. Thus, ${ \mathcal { F } } _ { j }$ is trained to predict ${ \mathbf { u } } _ { k + j }$ from $\hat { \mathbf { u } } _ { k + j - 1 }$ , which carries their accumulated errors.

During rollout, the operators are applied cyclically,

$$
{ \mathcal { F } } _ { 1 } \to { \mathcal { F } } _ { 2 } \to \cdots \to { \mathcal { F } } _ { M } \to { \mathcal { F } } _ { 1 } \to \cdots .\tag{18}
$$

For split models, the corrective-cycle index advances after each substep application, continuing across advection, diffusion and pressure projection rather than restarting at each solver time step.

## 4 EXPERIMENT DESIGN AND RESULTS

In this section, we explain our experiment design for our method as well as final results.

## 4.1 DATASETS

We use four fluid benchmarks for training and evaluation: Spectral-2D, Basilisk-2D, MC-2D, and MC-3D. We use VelMCFluids (Sugimoto et al., 2024) as our primary data-generation solver because it handles varied solid geometries through ray–boundary intersection queries without requiring a body-fitted volume mesh, while allowing velocity fields to be stored on the regular grids used by our neural operators. Its explicit advection, diffusion, and pressure-projection stages provide intermediate states that directly match our supervision targets. By comparison, standard PISO/PIMPLEbased OpenFOAM solvers combine convection and viscous terms within a momentum equation and enforce pressure–velocity coupling through correction loops (Weller et al., 1998; CFD Direct, 2025), so obtaining the same three physical substep targets would require additional adaptation of the numerical scheme. These properties make VelMCFluids convenient for our experiments, but the learning framework does not require stochastic supervision. To demonstrate its applicability to de terministic solvers, we additionally implement a spectral solver with separate advection, diffusion, and pressure-projection stages and evaluate a modified Basilisk solver (Popinet, 2009). The resulting Spectral-2D and Basilisk-2D benchmarks test the framework without solid geometry and with embedded obstacles, respectively, complementing the MC benchmarks with deterministic evaluations. Full dataset and solver details are provided in Appendix A.

Spectral-2D uses standard held-out trajectories, while the other three benchmarks evaluate zeroshot generalization to unseen geometries. Basilisk-2D is trained on circular obstacles and tested on squares; MC-2D and MC-3D are tested on unseen NACA airfoils and ABC geometries (Koch et al., 2019), respectively. Spectral-2D, Basilisk-2D, and MC-2D are trained on 100-step trajectories and evaluated with 200-step rollouts, additionally testing temporal extrapolation. Due to its higher computational cost, MC-3D uses 50 steps for both training and evaluation.

## 4.2 TRAINING AND EVALUATION PROTOCOL

All learned operators predict residual velocity updates,

$$
\mathcal { F } ( \mathbf { u } ; \mathbf { c } ) = \mathbf { u } + \mathcal { R } ( \mathbf { u } ; \mathbf { c } ) ,\tag{19}
$$

where u is the input state, and c denotes the physical and geometric conditioning variables. For the single-operator model, these are consecutive full-step states; for Neural Operator Splitting, the same formulation is applied to each supervised fractional step pair. All operators are trained with relative $L ^ { 2 }$ loss over fluid grid points. Normalization, padding, data augmentation, model architecture, and other training details are provided in Appendix B.

We evaluate autoregressive rollouts over the fluid region using pooled relative $L ^ { 2 }$ rollout error, referred to as rollout error below. Tables pool squared errors and reference energies across trajectories and time; curves pool them across trajectories at each step. For MC benchmarks, tables report variance-corrected errors, while rollout curves show standard errors together with MC noise; deterministic benchmarks use standard errors throughout. Appendix C defines both error metrics and the MC noise estimates. Additional evaluation metrics are provided in Appendix D.

Figure 3 shows representative 2D and 3D rollouts of the single-operator baseline and the full model.

## 4.3 NEURAL OPERATOR SPLITTING

For both deterministic and Monte Carlo benchmarks, we generate training trajectories and record the velocity fields after each advection, diffusion, and projection substep. The single operator is supervised by the full-step pairs $( \mathbf { u } ^ { n } , \mathbf { u } ^ { n + 1 } )$ , while the split operators are independently supervised by the corresponding substep pairs $( \mathbf { u } ^ { n } , \mathbf { u } _ { \ast } ^ { n } ) , \ : ( \mathbf { u } _ { \ast } ^ { n } , \mathbf { u } _ { \ast \ast } ^ { n } )$ , and $( \mathbf { u } _ { * * } ^ { n } , \mathbf { u } ^ { n + 1 } )$ . At inference time, the split operators are applied sequentially

Although splitting does not improve performance in the simpler setting without geometry, it consistently outperforms the single operator for fluid dynamics with geometry (Table 1), with lower errors both within the training horizon and in temporal extrapolation in Fig. 5. To control for the increased parameter count introduced by using three split operators, we additionally train single-operator baselines with three times the number of FNO layers, yielding a comparable parameter count. As shown in Table 1, this threefold increase in model size provides essentially no performance improvement, indicating that the gains from splitting do not arise from increased model capacity.

A qualitative example is the vortex ring impinging on the bunny, where the evolution of the vortex shape is visibly closer to the MC reference for the split model than for the single operator, as shown in Fig. 4.

## 4.4 CONDITIONAL LINEAR ARCHITECTURE

For geometry-dependent datasets, geometry and physical coordinates are provided only to the conditioning branch. For the single and advection operators, velocity enters both branches, whereas for diffusion and projection it enters only the linear branch. Spectral-2D uses neither geometry nor coordinate inputs; its diffusion and projection operators therefore reduce to the linear FNO. As shown in Fig. 5 and Table 2, this conditional linear (CL) design reduces long-horizon rollout error for the split models across all benchmarks, whereas the benefit for single models depends on the dataset.

![](images/a760da361220561f5dc9672414538eda3cf655229cacc0f8ede0e512bd4f09d9.jpg)

Figure 3: Qualitative rollout comparison in 2D and 3D. Rows show the MC reference, a single full-step operator (Single), and our combined model (Ours). Ours uses splitting, CL, SS and CC. Columns show t = 0.8, 1.6, and 3.2. (a) Flow around a circular obstacle, colored by signed vorticity with a shared scale. (b) Flow around a bunny-shaped obstacle. Blue volumes show vortex regions identified by the $\lambda _ { 2 }$ criterion. All 3D panels use the same rendering parameters.  
![](images/5ccef3f08d218994421169c7ad3532dbfe1351cf7db969184dd2cd7285a1daf5.jpg)  
Figure 4: Vortex-ring interaction with a bunny-shaped obstacle. Rows show a single full-step operator (single), an operator-splitting model (split), and the MC reference. Columns show t = 0.0, 0.4, 0.8, 1.2, and 1.6. Red volumes show the magnitude of azimuthal vorticity; brown surfaces show the solid obstacle.

## 4.5 SYMMETRY SAMPLING

We first isolate the effect of inference-time symmetry on the MC-2D single-operator baseline, with model weights fixed across all variants. We compare plain rollout, stochastic $D _ { 4 }$ symmetry sampling, and full $D _ { 4 }$ group averaging. As shown in Fig. 6, stochastic sampling substantially reduces rolslout error, while group averaging provides only a small additional improvement despite requir ing many more forward passes per step. This supports our hypothesis that independently varying the symmetry realization can disrupt coherent error accumulation without requiring full group averaging.

Table 1: Relative $L ^ { 2 }$ rollout error of single-operator and split models. Single $3 \times$ uses three times as many FNO layers as Single. Percentages are relative to Single.
<table><tr><td>Dataset</td><td>Single</td><td>Single 3×</td><td>Split</td></tr><tr><td>Spectral-2D</td><td> $\mathbf { 1 . 6 2 6 \times 1 0 ^ { - 3 } }$ </td><td> $1 . 6 3 4 \times 1 0 ^ { - 3 } \ : ( + 0 . 5 \% )$ </td><td> $2 . 6 3 0 \times 1 0 ^ { - 3 } \ : ( + 6 1 . 8 \% )$ </td></tr><tr><td>Basilisk-2D</td><td> $5 . 2 8 6 \times 1 0 ^ { - 2 }$ </td><td> $5 . 2 5 1 \times 1 0 ^ { - 2 } \ : ( - 0 . 7 \% )$ </td><td> $\mathbf { 4 . 9 1 9 \times 1 0 ^ { - 2 } } \left( - 6 . 9 \% \right)$ </td></tr><tr><td>MC-2D</td><td> $7 . 7 8 \times 1 0 ^ { - 2 }$ </td><td> $7 . 9 6 \times 1 0 ^ { - 2 } \ : ( + 2 . 2 \% )$ </td><td> ${ \bf 6 . 9 5 \times 1 0 ^ { - 2 } } \left( - 1 0 . 7 \% \right)$ </td></tr><tr><td>MC-3D</td><td> $5 . 0 7 \times 1 0 ^ { - 2 }$ </td><td> $5 . 1 4 \times 1 0 ^ { - 2 } \ : ( + 1 . 3 \% )$ </td><td> $\mathbf { 3 . 8 6 \times 1 0 ^ { - 2 } } \ ( - 2 3 . 8 \% )$ </td></tr></table>

![](images/8917e296e123ee8cb1f453b174a25b9a5361bf8ae6fb6725663f1e490fe744d6.jpg)

![](images/a93fdc24cb486b26fc367db5f51d43a11baba22d73e82492a5990ef2b1f61701.jpg)

![](images/b05346f8ed17a66e3556400d2500b97014c182af63bf8ff79805af10d4e9f540.jpg)  
Figure 5: Relative $L ^ { 2 }$ rollout error of standard and conditional-linear (CL) single-operator and split models. Black dotted curves show MC noise; shaded regions mark extrapolation beyond the training horizon.

We further evaluate symmetry sampling (SS) across both single-operator and split models on all benchmarks, using $D _ { 4 }$ in 2D and $O _ { h }$ in 3D. A symmetry element is sampled at every operator call: once per rollout step for the single model and independently for each $\mathcal { F } _ { A } , \mathcal { F } _ { D }$ , or $\mathcal { F } _ { P }$ call for the split model. As summarized in Table 3, SS consistently reduces rollout error across all benchmarks.

## 4.6 CORRECTIVE CYCLING

We first compare corrective cycling (CC) with unrolled and pushforward training in the MC-2D single-operator setting. CC approaches the rollout accuracy of unrolled training with substantially lower additional training time and peak GPU memory, and outperforms pushforward training in accuracy and cost under matched fine-tuning (Appendix G).

We then evaluate CC with $M = 2$ operators across the benchmarks. For the single-operator model, $\mathcal { F } _ { 1 }$ and $\mathcal { F } _ { 2 }$ are alternated across successive rollout steps. For the split model, the alternation continues across the advection, diffusion, and projection stages, following $\mathcal { F } _ { A 1 } \to \mathcal { F } _ { D 2 } \to \mathcal { F } _ { P 1 } \to$ $\mathcal { F } _ { A 2 } \to \mathcal { F } _ { D 1 } \to \mathcal { F } _ { P 2 } \to \cdot \cdot \cdot$ . As shown in Table 4, this strategy provides no improvement on the simpler dataset, but consistently reduces rollout error for both the single and split operators on the MC dataset with complex geometries.

## 4.7 COMBINED PERFORMANCE

Finally, we combine the proposed components to evaluate their cumulative effect. As shown in Figure 7, the full combination substantially reduces rollout error relative to the single-operator baseline across all benchmarks. These results show that the proposed components remain effective when used together, yielding consistent improvements in long-horizon rollout accuracy.

Table 2: Relative $L ^ { 2 }$ rollout error of the standard and conditional-linear (CL) single and split models. Percentages indicate the relative change from the corresponding standard model.
<table><tr><td>Dataset</td><td>Standard Single</td><td>CL Single</td><td>Standard Split</td><td>CL Split</td></tr><tr><td>Spectral-2D</td><td> $1 . 6 2 6 \times 1 0 ^ { - 3 }$ </td><td> $2 . 4 1 6 \times 1 0 ^ { - 3 } \ : ( + 4 8 . 6 \% )$ </td><td> $2 . 6 3 0 \times 1 0 ^ { - 3 }$ </td><td> $\mathbf { 1 . 0 3 1 \times 1 0 ^ { - 3 } ( - 6 0 . 8 \% ) }$ </td></tr><tr><td>Basilisk-2D</td><td> $5 . 2 8 6 \times 1 0 ^ { - 2 }$ </td><td> $4 . 9 9 9 \times 1 0 ^ { - 2 } \ : ( - 5 . 4 \% )$ </td><td> $4 . 9 1 9 \times 1 0 ^ { - 2 }$ </td><td> $\mathbf { 1 . 9 0 1 \times 1 0 ^ { - 2 } } \left( - 6 1 . 3 \% \right)$ </td></tr><tr><td>MC-2D</td><td> $7 . 7 8 \times 1 0 ^ { - 2 }$ </td><td> $7 . 5 9 \times 1 0 ^ { - 2 } \ : ( - 2 . 5 \% )$ </td><td> $6 . 9 5 \times 1 0 ^ { - 2 }$ </td><td> $\mathbf { 3 . 9 7 \times 1 0 ^ { - 2 } } \ ( \mathbf { - 4 2 . 9 \% } )$ </td></tr><tr><td>MC-3D</td><td> $5 . 0 7 \times 1 0 ^ { - 2 }$ </td><td> $4 . 0 9 \times 1 0 ^ { - 2 } \ : ( - 1 9 . 3 \% )$ </td><td> $3 . 8 6 \times 1 0 ^ { - 2 }$ </td><td> $\mathbf { 3 . 1 7 \times 1 0 ^ { - 2 } } \left( - 1 7 . 9 \% \right)$ </td></tr></table>

![](images/d3744d812a48871e4178baf16806ca308713155149f10e4b3e36689614414f41.jpg)  
Figure 6: Relative $L ^ { 2 }$ rollout error of the single-operator model with plain inference, symmetry sampling (SS), and full group averaging. The symmetry group is $D _ { 4 }$ in 2D and $O _ { h }$ in 3D. Black dotted curves show MC noise; shaded regions mark extrapolation beyond the training horizon.

## 5 DISCUSSION

Figure 8 compares the runtime–accuracy tradeoff of the learned models with VelMCFluids on MC-2D and MC-3D. MC error arises from stochastic sampling and decreases with increasing computational cost. At comparable error levels, neural-operator inference is roughly three orders of magnitude faster per step. By further lowering surrogate error, Neural Operator Splitting and the additional components make learned models viable at stricter accuracy targets, where achieving the same accuracy with MC becomes increasingly expensive.

Despite these gains, our study has several limitations. First, our experiments focus on incompressible Navier–Stokes flows with advection–diffusion–projection splitting; extending neural operator splitting to other PDE systems and fractional-step decompositions remains to be explored. The approach also relies on access to meaningful intermediate solver states for substep supervision, which may not be available in all numerical solvers. Second, the conditional-linear architecture is motivated by the structure exposed through operator splitting and is not uniformly advantageous as a standalone architecture. For nonlinear mappings such as advection and the full-step operator, we also observe reduced stability when extrapolating to Courant–Friedrichs–Lewy (CFL) numbers substantially larger than those seen during training. Its behavior under broader nonlinear and extrapolative regimes remains to be further investigated. Finally, although we evaluate across spectral, Basilisk, and Monte Carlo solvers in both 2D and 3D, broader validation across solver families and geometry distributions is left for future work.

## 6 CONCLUSION

We introduced Neural Operator Splitting, a framework that follows the physical substeps of a fractional-step fluid solver and learns the corresponding advection, diffusion, and pressureprojection operators from intermediate solver states. Across 2D and 3D fluid problems with complex geometries, the operator-splitting formulation consistently improves long-horizon rollout accuracy over a single full-step neural operator, while this advantage does not appear in the simpler geometry free setting.

Table 3: Relative $L ^ { 2 }$ rollout error with and without symmetry sampling (SS). Percentages indicate the relative change from the corresponding model without SS.
<table><tr><td>Dataset</td><td>Single</td><td>Single+SS</td><td>Split</td><td>Split+SS</td></tr><tr><td>Spectral-2D</td><td> $1 . 6 2 6 \times 1 0 ^ { - 3 }$ </td><td> $\mathbf { 8 . 4 2 4 \times 1 0 ^ { - 4 } } \left( - 4 8 . 2 \% \right)$ </td><td> $2 . 6 3 0 \times 1 0 ^ { - 3 }$ </td><td> $1 . 3 0 3 \times 1 0 ^ { - 3 } \ : ( - 5 0 . 5 \% )$ </td></tr><tr><td>Basilisk-2D</td><td> $5 . 2 8 6 \times 1 0 ^ { - 2 }$ </td><td> $4 . 1 7 6 \times 1 0 ^ { - 2 } \ : ( - 2 1 . 0 \% )$ </td><td> $4 . 9 1 9 \times 1 0 ^ { - 2 }$ </td><td> $\mathbf { 3 . 8 3 8 \times 1 0 ^ { - 2 } } \left( - 2 2 . 0 \% \right)$ </td></tr><tr><td>MC-2D</td><td> $7 . 7 8 \times 1 0 ^ { - 2 }$ </td><td> $5 . 0 8 \times 1 0 ^ { - 2 } \left( - 3 4 . 7 \% \right)$ </td><td> $6 . 9 5 \times 1 0 ^ { - 2 }$ </td><td> $\mathbf { 4 . 7 3 \times 1 0 ^ { - 2 } } \ ( - 3 2 . 0 \% )$ </td></tr><tr><td>MC-3D</td><td> $5 . 0 7 \times 1 0 ^ { - 2 }$ </td><td> $4 . 1 8 \times 1 0 ^ { - 2 } \ : ( - 1 7 . 6 \% )$ </td><td> $3 . 8 6 \times 1 0 ^ { - 2 }$ </td><td> $\mathbf { 3 . 3 3 \times 1 0 ^ { - 2 } } \ ( - 1 3 . 9 \% )$ </td></tr></table>

Table 4: Relative $L ^ { 2 }$ rollout error with and without corrective cycling (CC). Percentages indicate the relative change from the corresponding baseline without CC.
<table><tr><td>Dataset</td><td>Single</td><td>Single+CC</td><td>Split</td><td>Split+CC</td></tr><tr><td>Spectral-2D</td><td> $\mathbf { 1 . 6 2 6 \times 1 0 ^ { - 3 } }$ </td><td> $1 . 8 0 1 \times 1 0 ^ { - 3 } \ : ( + 1 0 . 8 \% )$ </td><td> $2 . 6 3 0 \times 1 0 ^ { - 3 }$ </td><td> $2 . 2 3 2 \times 1 0 ^ { - 3 } ( - 1 5 . 1 \% )$ </td></tr><tr><td>Basilisk-2D</td><td> $5 . 2 8 6 \times 1 0 ^ { - 2 }$ </td><td> $5 . 4 4 6 \times 1 0 ^ { - 2 } \ : ( + 3 . 0 \% )$ </td><td> $\mathbf { 4 . 9 1 9 \times 1 0 ^ { - 2 } }$ </td><td> $5 . 0 6 5 \times 1 0 ^ { - 2 } \ : ( + 3 . 0 \% )$ </td></tr><tr><td>MC-2D</td><td> $7 . 7 8 \times 1 0 ^ { - 2 }$ </td><td> $6 . 2 4 \times 1 0 ^ { - 2 } \ : ( - 1 9 . 9 \% )$ </td><td> $6 . 9 5 \times 1 0 ^ { - 2 }$ </td><td> $\mathbf { 5 . 8 1 \times 1 0 ^ { - 2 } } \ ( - 1 6 . 5 \% )$ </td></tr><tr><td>MC-3D</td><td> $5 . 0 7 \times 1 0 ^ { - 2 }$ </td><td> $4 . 0 0 \times 1 0 ^ { - 2 } \left( - 2 1 . 0 \% \right)$ </td><td> $3 . 8 6 \times 1 0 ^ { - 2 }$ </td><td> $\mathbf { 3 . 8 0 \times 1 0 ^ { - 2 } } \left( - 1 . 7 \% \right)$ </td></tr></table>

Building on this formulation, we further introduced a structure-matched conditional-linear architecture together with two complementary strategies for autoregressive rollout. The conditional-linear architecture matches each learned suboperator to its physical structure, incorporating the known linear structure of selected substeps while allowing the operator to adapt to geometry. Symmetry sampling varies the symmetry-transformed operator across applications to reduce coherent error accumulation. Corrective cycling trains a sequence of operators on states containing errors from pre ceding operators. These components provide complementary reductions in rollout error and, when combined, reduce mean rollout error by 68% in 2D and 40% in 3D relative to the single full-step operator (Fig 7).

Overall, our results suggest that a full solver step need not be treated as a monolithic learning target in neural PDE solvers. Neural Operator Splitting decomposes this target into physically distinct sub-processes and provides a framework in which substep-level structural designs and rollout-level strategies can be combined. Together, these mechanisms substantially improve long-horizon prediction of fluid dynamics across complex and varying geometries. Future work could extend the evaluation to additional solver families, including OpenFOAM, explore alternative operator-splitting schemes, and extend the framework to other PDE systems.

## AI USE STATEMENT

In this work, we used generative AI tools to assist with parts of method implementation and experimental work. Additionally, we used generative AI tools to polish writing and identify relevant literature. We did not use generative AI tools for other applicable tasks requiring disclosure. All AIassisted work was reviewed and verified by the authors. We take responsibility for the final content of this work, including text, claims, and artifacts produced with the aid of generative AI.

## ACKNOWLEDGEMENTS

Anima Anandkumar and Xi Deng are both supported in part by the Bren endowed chair and the AI2050 senior fellow program at Schmidt Sciences. Hong Chul Nam is supported by a PhD fellowship for the Center for Data Science at NYU. Jianxiang Li is supported by Caltech Summer Undergraduate Research Fellowships (SURF) program.

## REFERENCES

Francesco Alesiani, Makoto Takamoto, and Mathias Niepert. Hyperfno: Improving the generalization behavior of fourier neural operators. In NeurIPS 2022 Workshop on Machine Learning and

![](images/f4e993582d8d8547fb944a26ebb2c5cb35ad5e5c0b645d4e717b4a4e7bf41e3b.jpg)

![](images/04018e792f237959e010a88b5a9d296e38e64aaaa0aabd1c16353eecfb558fd7.jpg)

Figure 7: Combined rollout improvements. Relative $L ^ { 2 }$ rollout error divided by the singleoperator baseline error on each benchmark. Starting from Split, successive bars add the conditionallinear architecture (CL), symmetry sampling (SS), and corrective cycling (CC). Single 3× uses three times as many FNO layers as Single. The dashed line marks the baseline ratio of 1.  
![](images/5041be65ac29efe9ebbe6d54601a84ccf8d84623f175c8c0a2cea0adba339839.jpg)  
Figure 8: Runtime–accuracy tradeoff. Per-step runtime on an RTX 4090 versus relative $L ^ { 2 }$ error, with logarithmic axes. Filled circles show variance-corrected neural-operator rollout errors. Open circles estimate the first-step sampling error of one MC realization from independent seeds (Appendix C). MC path counts vary; other estimator settings follow Table 6. Lines show Pareto fronts.

## Physical Sciences, 2022.

Johannes Brandstetter, Daniel E. Worrall, and Max Welling. Message passing neural PDE solvers. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview.net/forum?id= vSix3HPYKSU.

CFD Direct. OpenFOAM v13 User Guide: Solution and Algorithm Control, 2025. URL https:// doc.cfd.direct/openfoam/user-guide-v13/fvsolution. Section 4.6, accessed September 25, 2026.

Alexandre Joel Chorin. Numerical solution of the Navier–Stokes equations. Mathematics of Computation, 22(104):745–762, 1968. doi: 10.1090/S0025-5718-1968-0242392-2.

Valentino F. Foit, David W. Hogg, and Soledad Villar. Group averaging for physics applications: Accuracy improvements at zero training cost. CoRR, abs/2511.09573, 2025. doi: 10.48550/ ARXIV.2511.09573. URL https://doi.org/10.48550/arXiv.2511.09573.

Vignesh Gopakumar, Ander Gray, Daniel Giles, Lorenzo Zanisi, Matt J. Kusner, Timo Betcke, Stanislas Pamela, and Marc Peter Deisenroth. Learning physical operators using neural operators. In Emtiyaz Khan, Yingzhen Li, Arno Solin, and Aaditya Ramdas (eds.), Proceedings ofThe 29th International Conference on Artificial Intelligence and Statistics, volume 300 of Proceedings of Machine Learning Research, pp. 3223–3231. PMLR, 02–05 May 2026. URL https:// proceedings.mlr.press/v300/gopakumar26a.html.

Jacob Helwig, Xuan Zhang, Cong Fu, Jerry Kurtin, Stephan Wojtowytsch, and Shuiwang Ji. Group equivariant Fourier neural operators for partial differential equations. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 12907–12930. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/v202/helwig23a.html.

Hamda Hmida, Hsiu-Wen Chang, and Youssef Mesri. Compositional neural operators for multidimensional fluid dynamics. In International Conference on Learning Representations, 2026.

Cheng Jing, Uvini Balasuriya Mudiyanselage, Abhishek Verma, Kallol Bera, Shahid Rauf, and Kookjin Lee. Extending fourier neural operators for modeling parameterized and coupled pdes. In C. Vondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust (eds.), International Conference on Learning Representations, volume 2026, pp. 119621–119651, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ c253f6fe5cc2bd1905427da63ac37917-Paper-Conference.pdf.

Jinwoo Kim, Dat Nguyen, Ayhan Suleymanzade, Hyeokjun An, and Seunghoon Hong. Learning probabilistic symmetrization for architecture agnostic equivariance. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), Advances in Neural Information Processing Systems, volume 36, pp. 18582–18612. Curran Associates, Inc., 2023. doi: 10.52202/ 075280-0816. URL https://proceedings.neurips.cc/paper\_files/paper/ 2023/file/3b5c7c9c5c7bd77eb73d0baec7a07165-Paper-Conference.pdf.

Sebastian Koch, Albert Matveev, Zhongshi Jiang, Francis Williams, Alexey Artemov, Evgeny Burnaev, Marc Alexa, Denis Zorin, and Daniele Panozzo. Abc: A big cad model dataset for geometric deep learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.

Yuan Lan, Zhen Li, Jie Sun, and Yang Xiang. Dosnet as a non-black-box pde solver: When deep learning meets operator splitting. Journal of Computational Physics, 491:112343, 2023. ISSN 0021-9991. doi: https://doi.org/10.1016/j.jcp.2023.112343. URL https://www. sciencedirect.com/science/article/pii/S0021999123004382.

Zongyi Li, Nikola Borislavov Kovachki, Kamyar Azizzadenesheli, Burigede liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial differential equations. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=c8P9NQVtmnO.

Zongyi Li, Daniel Zhengyu Huang, Burigede Liu, and Anima Anandkumar. Fourier neural operator with learned deformations for pdes on general geometries. Journal of Machine Learning Research, 24(388):1–26, 2023a. URL http://jmlr.org/papers/v24/23-0064.html.

Zongyi Li, Nikola Kovachki, Chris Choy, Boyi Li, Jean Kossaifi, Shourya Otta, Mohammad Amin Nabian, Maximilian Stadler, Christian Hundt, Kamyar Azizzadenesheli, and Animashree Anandkumar. Geometry-informed neural operator for large-scale 3d pdes. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), Advances in Neural Information Processing Systems, volume 36, pp. 35836–35854. Curran Associates, Inc., 2023b. doi: 10.52202/ 075280-1556. URL https://proceedings.neurips.cc/paper\_files/paper/ 2023/file/70518ea42831f02afc3a2828993935ad-Paper-Conference.pdf.

Ning Liu, Siavash Jafarzadeh, and Yue Yu. Domain agnostic fourier neural operators. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), Advances in Neural Information Processing Systems, volume 36, pp. 47438– 47450. Curran Associates, Inc., 2023. doi: 10.52202/075280-2053. URL https://proceedings.neurips.cc/paper\_files/paper/2023/file/ 940a7634dab556b67af15bacd337f7db-Paper-Conference.pdf.

Shev MacNamara and Gilbert Strang. Operator splitting. In Splitting methods in communication, imaging, science, and engineering, pp. 95–114. Springer, 2017.

H.A. Melchers, J.H.M. Prins, and M.R.A. Abdelmalik. Neural green’s operators for parametric partial differential equations. Computer Methods in Applied Mechanics and Engineering, 455, June 2026. ISSN 0045-7825. doi: 10.1016/j.cma.2026.118893.

Steven A. Orszag and Lawrence C. Kells. Transition to turbulence in plane poiseuille and plane couette flow. Journal ofFluid Mechanics, 96(1):159–205, 1980. doi: 10.1017/S0022112080002066.

Dhruv Patel, Deep Ray, {Michael R.A.} Abdelmalik, {Thomas J.R.} Hughes, and {Assad A.} Oberai. Variationally mimetic operator networks. Computer Methods in Applied Mechanics and Engineering, 419, February 2024. ISSN 0045-7825. doi: 10.1016/j.cma.2023.116536.

Stephane Popinet. An accurate adaptive solver for surface-tension-driven interfacial flows.´ Journal of Computational Physics, 228(16):5838–5866, 2009. ISSN 0021-9991. doi: https://doi. org/10.1016/j.jcp.2009.04.042. URL https://www.sciencedirect.com/science/ article/pii/S002199910900240X.

Omri Puny, Matan Atzmon, Edward J. Smith, Ishan Misra, Aditya Grover, Heli Ben-Hamu, and Yaron Lipman. Frame averaging for invariant and equivariant network design. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum? id=zIUyj55nXR.

Louis Serrano, Jiequn Han, Edouard Oyallon, Shirley Ho, and Rudy Morel. Test-time generalization for physics through neural operator splitting. In International Conference on Machine Learning (ICML), 2026.

Ryusuke Sugimoto, Christopher Batty, and Toshiya Hachisuka. Velocity-based monte carlo fluids. In ACM SIGGRAPH 2024 Conference Papers, SIGGRAPH ’24, New York, NY, USA, 2024. Association for Computing Machinery. ISBN 9798400705250. doi: 10.1145/3641519.3657405. URL https://doi.org/10.1145/3641519.3657405.

Li Sun, Hongbo Lv, Zhikai Jiang, Zhongtian Sun, Lanxu Yang, and Philip S. Yu. Origo: Interpretable multi-physics PDE foundation model through neural operator splitting. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/ forum?id=nBDuY2PjTY.

Henry G Weller, Gavin Tabor, Hrvoje Jasak, and Christer Fureby. A tensorial approach to computational continuum mechanics using object-oriented techniques. Computers in physics, 12(6): 620–631, 1998.

Seungwoo Yoo, Kyeongmin Yeo, Jisung Hwang, and Minhyuk Sung. Neural green’s functions. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, Main Conference, pp. 47961–47982. Curran Associates, Inc., 2025. doi: 10.52202/ 085713-1599. URL https://proceedings.neurips.cc/paper\_files/paper/ 2025/file/448abd486677165ceedfa790e9a61802-Paper-Conference.pdf.

Chenyu Zeng, Yanshu Zhang, Jiayi Zhou, Yuhan Wang, Zilin Wang, Yuhao Liu, Lei Wu, and Daniel Zhengyu Huang. Point cloud neural operator for parametric pdes on complex and variable geometries. Computer Methods in Applied Mechanics and Engineering, 443:118022, 2025. ISSN 0045-7825. doi: https://doi.org/10.1016/j.cma.2025.118022. URL https: //www.sciencedirect.com/science/article/pii/S0045782525002944.

## A DATASETS

Table 5: Grid resolutions, trajectory counts, and trajectory lengths (in time steps).
<table><tr><td></td><td>Spectral-2D</td><td>Basilisk-2D</td><td>MC-2D</td><td>MC-3D</td></tr><tr><td>Grid</td><td> $6 4 ^ { 2 }$ </td><td> $6 4 ^ { 2 }$ </td><td> $6 4 ^ { 2 }$ </td><td> $6 4 ^ { 3 }$ </td></tr><tr><td>Train/validation/test</td><td>500/50/50</td><td>500/50/50</td><td>2000/200/100</td><td>300/30/30</td></tr><tr><td>Training trajectory length</td><td>100</td><td>100</td><td>100</td><td>50</td></tr><tr><td>Test trajectory length</td><td>200</td><td>200</td><td>200</td><td>50</td></tr></table>

All four datasets use a domain of side length 1, time step $\Delta t = 0 . 0 2$ , and kinematic viscosity $\nu = 1 0 ^ { - 3 }$ . Spectral and Basilisk are doubly periodic, whereas the MC simulations use an unbounded exterior domain.

For datasets with solids, the models receive a static signed distance function (SDF) on the regular grid, negative in the solid and positive in the fluid. Basilisk evaluates its analytic solid representation at cell centers. For 2D MC, this distance is computed directly to the polygonal boundary; for 3D MC, to the normalized triangular mesh used by the solver, both at cell centers without interpolation. The velocity inside the solid is set to zero. Initial fields are lifted to approximately satisfy u · n = 0 at the boundary, using a harmonic stream-function lift for Basilisk and 2D MC and a potential-band lift for 3D MC. For MC, the solver applies one projection before the advection–diffusion–projection sequence.

Spectral-2D. There is no solid. The initial velocity is the curl of a stream function solved from periodic Gaussian vorticity, with vortices placed uniformly, as dipoles, or in clusters, and is scaled to a peak speed of 0.5.

Basilisk-2D. Each trajectory has one stationary solid: a circle in training and validation, and an axis-aligned square at test. The embedded boundary is no-slip. The initial velocity is the curl of a stream function combining Fourier modes, compact vortices, and an optional uniform stream. The fluid RMS speed is drawn uniformly from [0.10, 0.30] and capped at 0.45, and the projection tolerance is $1 0 ^ { - 3 }$

MC-2D. Each scene has one stationary solid. Training and validation geometries are analytic closed curves—circles, polygons, stars, and radial shapes—placed by a random affine transform. The test geometries are NACA airfoils outside those families. The initial velocity is the curl of a stream function combining a uniform stream, Fourier modes, vortices, shear, and wakes. The target fluid RMS speed is drawn uniformly from [0.75, 1.05], with speed capped at 2.

MC-3D. Each scene has one ABC part, scaled to that its longest bounding-box side is 0.6. The initial velocity is the curl of a vector potential assembled from a few atomic fields. The fluid RMS speed is 0.25 and the speed is capped at 0.5.

Table 6: MC estimator settings. Separate training and test values are shown where they differ.
<table><tr><td></td><td>MC-2D</td><td>MC-3D</td></tr><tr><td>Path length</td><td>4</td><td>4</td></tr><tr><td>MC paths (train/test)</td><td> $6 . 2 5 \times 1 0 ^ { 4 }$ </td><td> $1 0 ^ { 5 } / 5 \times 1 0 ^ { 6 }$ </td></tr><tr><td>Direct volume / pseudo-boundary samples</td><td> $1 . 2 5 \times 1 0 ^ { 5 }$ </td><td> $1 0 ^ { 5 }$ </td></tr><tr><td>Indirect volume samples</td><td>10</td><td>100</td></tr><tr><td>Indirect pseudo-boundary samples</td><td>10</td><td>10</td></tr><tr><td>Winding samples</td><td>1</td><td>0</td></tr><tr><td>Initial winding samples</td><td>1024</td><td>1024</td></tr></table>

Each MC test scene has a paired reference generated with the same geometry, initial condition, and numerical parameters but an independent MC seed. These pairs estimate sampling noise as described in Appendix C.

## B TRAINING SETUP

## B.1 TRAINING LOSS

For each learned operator, let ub and u denote the predicted and target velocity fields, respectively. Let Ω contain the fluid grid points for geometry-dependent data and the full grid for Spectral-2D, and let x denote a grid point. Define the discrete norm

$$
\| \mathbf { u } \| _ { \ell ^ { 2 } ( \Omega ) } : = \left( \sum _ { \mathbf { x } \in \Omega } \| \mathbf { u } ( \mathbf { x } ) \| _ { 2 } ^ { 2 } \right) ^ { 1 / 2 } ,\tag{20}
$$

where $\| \cdot \| _ { 2 }$ denotes the Euclidean norm over the velocity components. With $\varepsilon = 1 0 ^ { - 8 }$ , we use the relative discrete $\ell ^ { 2 }$ loss

$$
\mathscr { L } = \frac { \| \widehat { \mathbf { u } } - \mathbf { u } \| _ { \ell ^ { 2 } ( \Omega ) } } { \| \mathbf { u } \| _ { \ell ^ { 2 } ( \Omega ) } + \varepsilon } .\tag{21}
$$

The minibatch loss is the mean of $\mathcal { L }$ over the minibatch.

## B.2 INPUT PREPROCESSING

Normalization. For the velocity field, we use an isotropic normalization that commutes with the symmetry transformations used for data augmentation. Let $d \in \{ 2 , 3 \}$ be the spatial dimension, $u _ { c }$ the velocity component indexed by $c = 1 , \ldots , d ,$ , and $\mu _ { c }$ and $\sigma _ { c }$ its training-set mean and standard deviation. With $\dot { \varepsilon } _ { \mathrm { n o r m } } = 1 0 ^ { - 5 }$ , define the shared scale $\sigma _ { \mathrm { i s o } }$ and normalized component $\widetilde { u } _ { c }$ by

$$
\sigma _ { \mathrm { i s o } } = \left[ \frac { 1 } { d } \sum _ { c = 1 } ^ { d } \left( \mu _ { c } ^ { 2 } + \sigma _ { c } ^ { 2 } \right) \right] ^ { 1 / 2 } , \qquad \widetilde { u } _ { c } = \frac { u _ { c } } { \sigma _ { \mathrm { i s o } } + \varepsilon _ { \mathrm { n o r m } } } .\tag{22}
$$

Let s denote the SDF geometry channel and $\mu _ { s }$ and $\sigma _ { s }$ its training-set mean and standard deviation. Its normalized value se is defined separately by

$$
\widetilde s = \frac { s - \mu _ { s } } { \sigma _ { s } + \varepsilon _ { \mathrm { n o r m } } } .\tag{23}
$$

Domain padding. Spectral-2D uses no domain padding, Basilisk-2D uses zero-valued padding with a padding ratio of 0.125, and MC-2D/3D use replicate padding with the same ratio. For the MC datasets, replicate padding matches VelMCFluids’ clamp-to-edge treatment of velocity queries beyond the finite computational grid used to represent the unbounded physical domain.

## B.3 DATA AUGMENTATION

Symmetry augmentation. During training, we independently sample a random element of $D _ { 4 }$ in 2D or $O _ { h }$ in 3D for each learned operator and input–target pair, and apply it to the velocity fields, SDF, and coordinates, treating velocity as a vector field. All models use this augmentation online.

Combination with corrective cycling. When generating corrective-cycling inputs offline, each preceding operator is applied with independent symmetry sampling. The resulting input–target pairs receive the same online augmentation during training.

## B.4 CONDITIONAL-LINEAR IMPLEMENTATION

The conditioning branch retains the lifted feature and the output of each FNO layer. For each state branch layer ℓ, these features are mapped by layer-specific bias-free 1×1 convolutions and summed. A two-layer pointwise GELU MLP with expansion ratio 2 maps this sum to the gate-head outputs $\delta _ { \ell } ^ { \mathrm { i n } }$ and $\bar { \delta } _ { \ell } ^ { \mathrm { { o u t } } }$ . Each $\eta _ { \ell }$ is learned and initialized to 0.2. The gates are

$$
\Gamma _ { \ell } ^ { \mathrm { i n } } ( { \mathbf { c } } ) = \mathrm { e x p } \left[ \eta _ { \ell } \mathrm { t a n h } \Big ( \delta _ { \ell } ^ { \mathrm { i n } } \Big ) \right] , \qquad \Gamma _ { \ell } ^ { \mathrm { o u t } } ( { \mathbf { c } } ) = \mathrm { e x p } \left[ \eta _ { \ell } \mathrm { t a n h } \big ( \delta _ { \ell } ^ { \mathrm { o u t } } \big ) \right] .\tag{24}
$$

The final gate-head layer is initialized with zero-mean weights with standard deviation $1 0 ^ { - 3 }$ and zero bias.

The state branch uses bias-free pointwise lifting and projection, bias-free spectral convolutions, and linear skip connections, without activation.

## B.5 MODEL CONFIGURATION AND PARAMETER COUNTS

Table 7 summarizes the architecture settings that differ across the standard FNO, conditional-linear state branch, and conditioner in the primary MC experiments. All use four Fourier layers and a projection ratio of 2. MC inputs comprise velocity, SDF, and cell coordinates, and every learned operator predicts a residual velocity update.

Table 7: Architecture settings for the standard FNO and the two conditional-linear branches in the MC experiments. Fourier mode counts are per spatial dimension.
<table><tr><td colspan="2">Standard FNO</td><td>Linear state branch</td><td>Conditioning branch</td></tr><tr><td>Hidden width</td><td>64</td><td>48</td><td>48</td></tr><tr><td>Fourier modes</td><td>24</td><td>24</td><td>20</td></tr><tr><td>Nonlinearity</td><td>GELU</td><td>None</td><td>GELU</td></tr><tr><td>Channel MLP</td><td>Yes, expansion 0.5</td><td>No</td><td>Yes, expansion 0.5</td></tr><tr><td>Stabilizer</td><td>None</td><td>None</td><td>tanh</td></tr></table>

Table 8 reports trainable parameters per operator. The conditional-linear models have comparable or fewer parameters than the standard FNO, and their accuracy gains therefore cannot be attributed to a larger parameter count.

Table 8: Trainable parameters per learned operator. Counts include both state and conditioning branches for conditional-linear models. Single 3× is the 12-layer standard FNO from Table 1.
<table><tr><td>Dataset</td><td>Standard FNO</td><td>Single 3×</td><td>CL single/advection</td><td>CL diffusion/projection</td></tr><tr><td>Spectral-2D</td><td>5.163M</td><td>15.454M</td><td>5.048M</td><td>5.145M</td></tr><tr><td>Basilisk-2D</td><td>5.163M</td><td>15.454M</td><td>5.048M</td><td>5.048M</td></tr><tr><td>MC-2D</td><td>5.163M</td><td>15.454M</td><td>5.048M</td><td>5.048M</td></tr><tr><td>MC-3D</td><td>122.735M</td><td>368.169M</td><td>109.705M</td><td>109.705M</td></tr></table>

## B.6 OPTIMIZATION

We train with AdamW using a learning rate of $5 \times 1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 3 }$ , and batch sizes of 100 in 2D and 20 in 3D. ReduceLROnPlateau uses a factor of 0.5, patience 20, and an absolute threshold of $1 0 ^ { - 5 }$ . Training runs for at most 2000 epochs; after epoch 50, early stopping uses patience 50 and a minimum improvement of $1 0 ^ { - 5 }$ . Compared variants use matched random seeds, and evaluation uses the checkpoint with the lowest validation loss.

## C VARIANCE-CORRECTED ERROR AND MC NOISE

For each MC test sample i, representing a trajectory–time-step pair, let $\widehat { \mathbf { u } } _ { i }$ be the model prediction and let $\mathbf { u } _ { i } ^ { ( 1 ) }$ and $\mathbf { u } _ { i } ^ { ( 2 ) }$ be references generated with the same geometry, initial condition, and numerical parameters but independent MC seeds. Let $\Omega _ { i }$ be the fluid grid points of that sample. Using the discrete norm defined in Appendix $\mathbf { B } ,$ define the variance-corrected squared discrepancy $q _ { i }$ and squared signal $s _ { i }$ by

$$
q _ { i } = \frac { 1 } { 2 } \left( \left. \widehat { \mathbf { u } } _ { i } - \mathbf { u } _ { i } ^ { ( 1 ) } \right. _ { \ell ^ { 2 } ( \Omega _ { i } ) } ^ { 2 } + \left. \widehat { \mathbf { u } } _ { i } - \mathbf { u } _ { i } ^ { ( 2 ) } \right. _ { \ell ^ { 2 } ( \Omega _ { i } ) } ^ { 2 } - \left. \mathbf { u } _ { i } ^ { ( 1 ) } - \mathbf { u } _ { i } ^ { ( 2 ) } \right. _ { \ell ^ { 2 } ( \Omega _ { i } ) } ^ { 2 } \right) ,\tag{25}
$$

$$
s _ { i } = \frac { 1 } { 2 } \left( \left\| \mathbf { u } _ { i } ^ { ( 1 ) } \right\| _ { \ell ^ { 2 } ( \Omega _ { i } ) } ^ { 2 } + \left\| \mathbf { u } _ { i } ^ { ( 2 ) } \right\| _ { \ell ^ { 2 } ( \Omega _ { i } ) } ^ { 2 } - \left\| \mathbf { u } _ { i } ^ { ( 1 ) } - \mathbf { u } _ { i } ^ { ( 2 ) } \right\| _ { \ell ^ { 2 } ( \Omega _ { i } ) } ^ { 2 } \right) .\tag{26}
$$

Let $\mathbf { u } _ { i }$ be the common expectation of the two MC references. If their sampling errors are mutually uncorrelated and uncorrelated with the model error, then

$$
\mathbb { E } [ q _ { i } ] = \lVert \widehat { \mathbf { u } } _ { i } - \mathbf { u } _ { i } \rVert _ { \ell ^ { 2 } ( \Omega _ { i } ) } ^ { 2 } , \qquad \mathbb { E } [ s _ { i } ] = \lVert \mathbf { u } _ { i } \rVert _ { \ell ^ { 2 } ( \Omega _ { i } ) } ^ { 2 } .\tag{27}
$$

The correction removes the contribution of MC sampling variance in expectation but not bias shared by the two references.

Let $N$ be the number of test samples in the evaluation window and $\Delta V _ { i }$ the grid-cell area in 2D or volume in 3D for sample i. We report the pooled absolute and relative errors $E _ { \mathrm { a b s } }$ and $E _ { \mathrm { r e l } }$ respectively:

$$
E _ { \mathrm { a b s } } = \sqrt { \operatorname* { m a x } \left( \frac { 1 } { N } \sum _ { i } \Delta V _ { i } q _ { i } , 0 \right) } , \qquad E _ { \mathrm { r e l } } = \sqrt { \frac { \operatorname* { m a x } \left( \sum _ { i } q _ { i } , 0 \right) } { \sum _ { i } s _ { i } } } .\tag{28}
$$

The sums include all N records with equal weight, and the relative error is reported only when $\textstyle \sum _ { i } s _ { i } > 0$ . Within each benchmark, all records share the same grid-cell area or volume, which cancels in the relative error. Signed values of $q _ { i }$ and $s _ { i }$ are retained during aggregation; the nonnegative clamp is applied to the pooled squared error before taking its square root.

Standard rollout error and MC noise. For each time step in Figs. 5 and 6, we pool the squared norms over the test trajectories. With sums restricted to records at that step, the standard relative error is

$$
\sqrt { \frac { \sum _ { i } \left\| \widehat { \mathbf { u } } _ { i } - \mathbf { u } _ { i } ^ { ( 1 ) } \right\| _ { \ell ^ { 2 } ( \Omega _ { i } ) } ^ { 2 } } { \sum _ { i } \left\| \mathbf { u } _ { i } ^ { ( 1 ) } \right\| _ { \ell ^ { 2 } ( \Omega _ { i } ) } ^ { 2 } } } ,\tag{29}
$$

and MC noise is

$$
\sqrt { \frac { \sum _ { i } \left\| \mathbf { u } _ { i } ^ { ( 1 ) } - \mathbf { u } _ { i } ^ { ( 2 ) } \right\| _ { \ell ^ { 2 } ( \Omega _ { i } ) } ^ { 2 } } { \sum _ { i } \left\| \mathbf { u } _ { i } ^ { ( 1 ) } \right\| _ { \ell ^ { 2 } ( \Omega _ { i } ) } ^ { 2 } } } .\tag{30}
$$

This discrepancy includes the sampling variance of both references. The displayed curves use a three-point temporal median filter, with endpoints unchanged.

For the open circles in Fig. 8, we use first-step records to estimate the relative sampling error of one MC realization:

$$
\sqrt { \frac { \sum _ { i } \left\| \mathbf { u } _ { i } ^ { ( 1 ) } - \mathbf { u } _ { i } ^ { ( 2 ) } \right\| _ { \ell ^ { 2 } ( \Omega _ { i } ) } ^ { 2 } } { 2 \sum _ { i } s _ { i } } } , \qquad \sum _ { i } s _ { i } > 0 .\tag{31}
$$

The factor of $1 / 2$ converts the squared difference of independent references with equal sampling budgets to the sampling variance of one realization; the denominator uses the variance-corrected signal defined above.

## D ADDITIONAL EVALUATION METRICS

Tables 9–13 give a counterpart to each results table in the main text. The Split and CL tables additionally report one-step relative $L ^ { 2 }$ prediction error, evaluated from ground-truth input states. All five tables report rollout mean squared error (MSE) and absolute divergence over the same test trajectories and rollout horizons as their main-text counterparts.

We retain the sample index i, sample count N, fluid grid $\Omega _ { i } ,$ , and cell area or volume $\Delta V _ { i }$ from Appendix C. For deterministic benchmarks, $\mathbf { u } _ { i }$ denotes the deterministic reference field, and rollout MSE is the average physical squared velocity error:

$$
\mathrm { M S E } = \frac { 1 } { N } \sum _ { i } \Delta V _ { i } \left. \widehat { \mathbf { u } } _ { i } - \mathbf { u } _ { i } \right. _ { \ell ^ { 2 } ( \Omega _ { i } ) } ^ { 2 } .\tag{32}
$$

For MC benchmarks, we replace the squared discrepancy with the paired-reference corrected quantity $q _ { i }$ from Appendix C, and apply the nonnegative clamp after averaging:

$$
\mathrm { M S E } = \operatorname* { m a x } \left( \frac { 1 } { N } \sum _ { i } \Delta V _ { i } q _ { i } , 0 \right) = E _ { \mathrm { a b s } } ^ { 2 } .\tag{33}
$$

Thus, MSE has units of squared velocity integrated over the unit domain.

Absolute divergence is

$$
\left( \frac { 1 } { N } \sum _ { i } \left. \nabla \cdot \widehat { \mathbf { u } } _ { i } \right. _ { L ^ { 2 } ( \Omega _ { i } ) } ^ { 2 } \right) ^ { 1 / 2 } ,\tag{34}
$$

where the norm is evaluated on the common valid stencil support within $\Omega _ { i }$ . This measures the prediction’s own divergence, rather than a divergence error against the reference, and no MC correction is applied.

The one-step relative $L ^ { 2 }$ prediction error follows the pooled convention of Appendix C for MC data and the corresponding pooled error-to-signal ratio for deterministic data.

Table 9: One-step relative $L ^ { 2 }$ error, rollout MSE, and absolute divergence for the models in Table 1.
<table><tr><td>Dataset</td><td>Metric</td><td>Single</td><td>Single 3×</td><td>Split</td></tr><tr><td rowspan="3">Spectral-2D</td><td>One-step rel.  $L ^ { 2 }$ </td><td> $5 . 2 0 6 \times 1 0 ^ { - 5 }$ </td><td> $\mathbf { 4 . 9 4 4 \times 1 0 ^ { - 5 } }$ </td><td> $7 . 7 2 3 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Rollout MSE</td><td> $\mathbf { 1 . 0 0 5 \times 1 0 ^ { - 7 } }$ </td><td> $1 . 0 1 5 \times 1 0 ^ { - 7 }$ </td><td> $2 . 6 3 1 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>Rollout abs div</td><td> $5 . 0 2 9 \times 1 0 ^ { - 3 }$ </td><td> $\mathbf { 3 . 6 6 3 \times 1 0 ^ { - 3 } }$ </td><td> $5 . 1 5 7 \times 1 0 ^ { - 3 }$ </td></tr><tr><td rowspan="3">Basilisk-2D</td><td>One-step rel.  $L ^ { 2 }$ </td><td> $1 . 5 3 3 \times 1 0 ^ { - 3 }$ </td><td> $\mathbf { 1 . 4 9 9 \times 1 0 ^ { - 3 } }$ </td><td> $2 . 1 3 8 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Rollout MSE</td><td> $3 . 0 9 9 \times 1 0 ^ { - 5 }$ </td><td> $3 . 0 5 8 \times 1 0 ^ { - 5 }$ </td><td> $\mathbf { 2 . 6 8 3 \times 1 0 ^ { - 5 } }$ </td></tr><tr><td>Rollout abs div</td><td> $5 . 8 2 6 \times 1 0 ^ { - 2 }$ </td><td> $\mathbf { 5 . 6 0 5 \times 1 0 ^ { - 2 } }$ </td><td> $6 . 6 0 2 \times 1 0 ^ { - 2 }$ </td></tr><tr><td rowspan="3">MC-2D</td><td>One-step rel.  $L ^ { 2 }$ </td><td> $4 . 1 3 2 \times 1 0 ^ { - 3 }$ </td><td> $\mathbf { 4 . 0 6 8 \times 1 0 ^ { - 3 } }$ </td><td> $4 . 1 3 7 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Rollout MSE</td><td> $1 . 7 7 3 \times 1 0 ^ { - 3 }$ </td><td> $1 . 8 5 2 \times 1 0 ^ { - 3 }$ </td><td> $\mathbf { 1 . 4 1 5 \times 1 0 ^ { - 3 } }$ </td></tr><tr><td>Rollout abs div</td><td> $1 . 1 4 6 \times 1 0 ^ { - 1 }$ </td><td> $\mathbf { 1 . 1 2 8 \times 1 0 ^ { - 1 } }$ </td><td> $1 . 1 5 3 \times 1 0 ^ { - 1 }$ </td></tr><tr><td rowspan="3">MC-3D</td><td>One-step rel.  $L ^ { 2 }$ </td><td> $6 . 0 5 8 \times 1 0 ^ { - 3 }$ </td><td> $5 . 6 0 2 \times 1 0 ^ { - 3 }$ </td><td> $\mathbf { 5 . 8 2 8 \times 1 0 ^ { - 3 } }$ </td></tr><tr><td>Rollout MSE</td><td> $2 . 6 2 3 \times 1 0 ^ { - 5 }$ </td><td> $2 . 6 9 4 \times 1 0 ^ { - 5 }$ </td><td> $\mathbf { 1 . 5 2 5 \times 1 0 ^ { - 5 } }$ </td></tr><tr><td>Rollout abs div</td><td> $2 . 5 2 1 \times 1 0 ^ { - 2 }$ </td><td> $2 . 7 9 2 \times 1 0 ^ { - 2 }$ </td><td> $\mathbf { 2 . 2 8 8 \times 1 0 ^ { - 2 } }$ </td></tr></table>

Table 10: One-step relative $L ^ { 2 }$ error, rollout MSE, and absolute divergence for the models in Table 2.
<table><tr><td>Dataset</td><td>Metric</td><td>Single</td><td>CL Single</td><td>Split</td><td>CL Split</td></tr><tr><td rowspan="3">Spectral-2D</td><td>One-step rel.  $L ^ { 2 }$ </td><td> $5 . 2 0 6 \times 1 0 ^ { - 5 }$ </td><td> $\mathbf { 4 . 9 8 4 \times 1 0 ^ { - 5 } }$ </td><td> $7 . 7 2 3 \times 1 0 ^ { - 5 }$ </td><td> $6 . 5 6 1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Rollout MSE</td><td> $1 . 0 0 5 \times 1 0 ^ { - 7 }$ </td><td> $2 . 2 2 0 \times 1 0 ^ { - 7 }$ </td><td> $2 . 6 3 1 \times 1 0 ^ { - 7 }$ </td><td> $\mathbf { 4 . 0 4 5 \times 1 0 ^ { - 8 } }$ </td></tr><tr><td>Rollout abs div</td><td> $5 . 0 2 9 \times 1 0 ^ { - 3 }$ </td><td> $5 . 5 1 9 \times 1 0 ^ { - 3 }$ </td><td> $5 . 1 5 7 \times 1 0 ^ { - 3 }$ </td><td> $\mathbf { 4 . 2 7 6 \times 1 0 ^ { - 3 } }$ </td></tr><tr><td rowspan="3">Basilisk-2D</td><td>One-step rel.  $L ^ { 2 }$ </td><td> $1 . 5 3 3 \times 1 0 ^ { - 3 }$ </td><td> $\mathbf { 1 . 1 5 1 \times 1 0 ^ { - 3 } }$ </td><td> $2 . 1 3 8 \times 1 0 ^ { - 3 }$ </td><td> $2 . 3 0 5 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Rollout MSE</td><td> $3 . 0 9 9 \times 1 0 ^ { - 5 }$ </td><td> $2 . 7 7 2 \times 1 0 ^ { - 5 }$ </td><td> $2 . 6 8 3 \times 1 0 ^ { - 5 }$ </td><td> $\mathbf { 4 . 0 1 0 \times 1 0 ^ { - 6 } }$ </td></tr><tr><td>Rollout abs div</td><td> $5 . 8 2 6 \times 1 0 ^ { - 2 }$ </td><td> $5 . 5 5 5 \times 1 0 ^ { - 2 }$ </td><td> $6 . 6 0 2 \times 1 0 ^ { - 2 }$ </td><td> $\mathbf { 4 . 1 6 0 \times 1 0 ^ { - 2 } }$ </td></tr><tr><td rowspan="3">MC-2D</td><td>One-step rel.  $L ^ { 2 }$ </td><td> $4 . 1 3 2 \times 1 0 ^ { - 3 }$ </td><td> $3 . 1 0 3 \times 1 0 ^ { - 3 }$ </td><td> $4 . 1 3 7 \times 1 0 ^ { - 3 }$ </td><td> $\mathbf { 2 . 6 6 1 \times 1 0 ^ { - 3 } }$ </td></tr><tr><td>Rollout MSE</td><td> $1 . 7 7 3 \times 1 0 ^ { - 3 }$ </td><td> $1 . 6 8 7 \times { { 1 0 } ^ { - 3 } }$ </td><td> $1 . 4 1 5 \times 1 0 ^ { - 3 }$ </td><td> $\mathbf { 4 . 6 1 1 \times 1 0 ^ { - 4 } }$ </td></tr><tr><td>Rollout abs div</td><td> $1 . 1 4 6 \times 1 0 ^ { - 1 }$ </td><td> $9 . 8 6 3 \times 1 0 ^ { - 2 }$ </td><td> $1 . 1 5 3 \times 1 0 ^ { - 1 }$ </td><td> $\mathbf { 9 . 8 3 6 \times 1 0 ^ { - 2 } }$ </td></tr><tr><td rowspan="3">MC-3D</td><td>One-step rel.  $L ^ { 2 }$ </td><td> $6 . 0 5 8 \times 1 0 ^ { - 3 }$ </td><td> $5 . 3 6 2 \times 1 0 ^ { - 3 }$ </td><td> $5 . 8 2 8 \times 1 0 ^ { - 3 }$ </td><td> $\mathbf { 5 . 1 2 2 \times 1 0 ^ { - 3 } }$ </td></tr><tr><td>Rollout MSE</td><td> $2 . 6 2 3 \times 1 0 ^ { - 5 }$ </td><td> $1 . 7 0 9 \times 1 0 ^ { - 5 }$ </td><td> $1 . 5 2 5 \times 1 0 ^ { - 5 }$ </td><td> $\mathbf { 1 . 0 2 8 \times 1 0 ^ { - 5 } }$ </td></tr><tr><td>Rollout abs div</td><td> $2 . 5 2 1 \times 1 0 ^ { - 2 }$ </td><td> $1 . 9 9 8 \times 1 0 ^ { - 2 }$ </td><td> $2 . 2 8 8 \times 1 0 ^ { - 2 }$ </td><td> $\mathbf { 1 . 7 6 0 \times 1 0 ^ { - 2 } }$ </td></tr></table>

Table 11: Rollout MSE and absolute divergence with and without symmetry sampling, corresponding to Table 3.
<table><tr><td>Dataset</td><td>Metric</td><td>Single</td><td>Single+SS</td><td>Split</td><td>Split+SS</td></tr><tr><td rowspan="2">Spectral-2D</td><td>Rollout MSE</td><td> $1 . 0 0 5 \times 1 0 ^ { - 7 }$ </td><td> $\mathbf { 2 . 6 9 9 \times 1 0 ^ { - 8 } }$ </td><td> $2 . 6 3 1 \times 1 0 ^ { - 7 }$ </td><td> $6 . 4 5 6 \times 1 0 ^ { - 8 }$ </td></tr><tr><td>Rollout abs div</td><td> $5 . 0 2 9 \times 1 0 ^ { - 3 }$ </td><td> $\mathbf { 3 . 9 5 4 \times 1 0 ^ { - 3 } }$ </td><td> $5 . 1 5 7 \times 1 0 ^ { - 3 }$ </td><td> $3 . 9 9 8 \times 1 0 ^ { - 3 }$ </td></tr><tr><td rowspan="2">Basilisk-2D</td><td>Rollout MSE</td><td> $3 . 0 9 9 \times 1 0 ^ { - 5 }$ </td><td> $1 . 9 3 4 \times 1 0 ^ { - 5 }$ </td><td> $2 . 6 8 3 \times 1 0 ^ { - 5 }$ </td><td> $\mathbf { 1 . 6 3 4 \times 1 0 ^ { - 5 } }$ </td></tr><tr><td>Rollout abs div</td><td> $5 . 8 2 6 \times 1 0 ^ { - 2 }$ </td><td> $\mathbf { 3 . 1 4 5 \times 1 0 ^ { - 2 } }$ </td><td> $6 . 6 0 2 \times 1 0 ^ { - 2 }$ </td><td> $3 . 9 5 2 \times 1 0 ^ { - 2 }$ </td></tr><tr><td rowspan="2">MC-2D</td><td>Rollout MSE</td><td> $1 . 7 7 3 \times 1 0 ^ { - 3 }$ </td><td> $7 . 5 6 6 \times 1 0 ^ { - 4 }$ </td><td> $1 . 4 1 5 \times 1 0 ^ { - 3 }$ </td><td> $\mathbf { 6 . 5 4 7 \times 1 0 ^ { - 4 } }$ </td></tr><tr><td>Rollout abs div</td><td> $1 . 1 4 6 \times 1 0 ^ { - 1 }$ </td><td> $\mathbf { 1 . 0 8 2 \times 1 0 ^ { - 1 } }$ </td><td> $1 . 1 5 3 \times 1 0 ^ { - 1 }$ </td><td> $1 . 1 1 9 \times 1 0 ^ { - 1 }$ </td></tr><tr><td rowspan="2">MC-3D</td><td>Rollout MSE</td><td> $2 . 6 2 3 \times 1 0 ^ { - 5 }$ </td><td> $1 . 7 8 3 \times 1 0 ^ { - 5 }$ </td><td> $1 . 5 2 5 \times 1 0 ^ { - 5 }$ </td><td> $\mathbf { 1 . 1 3 0 \times 1 0 ^ { - 5 } }$ </td></tr><tr><td>Rollout abs div</td><td> $2 . 5 2 1 \times 1 0 ^ { - 2 }$ </td><td> $1 . 9 2 5 \times 1 0 ^ { - 2 }$ </td><td> $2 . 2 8 8 \times 1 0 ^ { - 2 }$ </td><td> $\mathbf { 1 . 9 2 2 \times 1 0 ^ { - 2 } }$ </td></tr></table>

Table 12: Rollout MSE and absolute divergence with and without corrective cycling, corresponding to Table 4.
<table><tr><td>Dataset</td><td>Metric</td><td>Single</td><td>Single+CC</td><td>Split</td><td>Split+CC</td></tr><tr><td rowspan="2">Spectral-2D</td><td>Rollout MSE</td><td> $\mathbf { 1 . 0 0 5 \times 1 0 ^ { - 7 } }$ </td><td> $1 . 2 3 3 \times 1 0 ^ { - 7 }$ </td><td> $2 . 6 3 1 \times 1 0 ^ { - 7 }$ </td><td> $1 . 8 9 4 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>Rollout abs div</td><td> $5 . 0 2 9 \times 1 0 ^ { - 3 }$ </td><td> $5 . 2 8 6 \times 1 0 ^ { - 3 }$ </td><td> $5 . 1 5 7 \times 1 0 ^ { - 3 }$ </td><td> $\mathbf { 4 . 8 8 6 \times 1 0 ^ { - 3 } }$ </td></tr><tr><td rowspan="2">Basilisk-2D</td><td>Rollout MSE</td><td> $3 . 0 9 9 \times 1 0 ^ { - 5 }$ </td><td> $3 . 2 8 9 \times 1 0 ^ { - 5 }$ </td><td> $\mathbf { 2 . 6 8 3 \times 1 0 ^ { - 5 } }$ </td><td> $2 . 8 4 6 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Rollout abs div</td><td> $5 . 8 2 6 \times 1 0 ^ { - 2 }$ </td><td> $\mathbf { 5 . 3 3 5 \times 1 0 ^ { - 2 } }$ </td><td> $6 . 6 0 2 \times 1 0 ^ { - 2 }$ </td><td> $6 . 4 1 7 \times 1 0 ^ { - 2 }$ </td></tr><tr><td rowspan="2">MC-2D</td><td>Rollout MSE</td><td> $1 . 7 7 3 \times 1 0 ^ { - 3 }$ </td><td> $1 . 1 3 9 \times 1 0 ^ { - 3 }$ </td><td> $1 . 4 1 5 \times 1 0 ^ { - 3 }$ </td><td> $\mathbf { 9 . 8 7 4 \times 1 0 ^ { - 4 } }$ </td></tr><tr><td>Rollout abs div</td><td> $1 . 1 4 6 \times 1 0 ^ { - 1 }$ </td><td> $\mathbf { 1 . 1 0 5 \times 1 0 ^ { - 1 } }$ </td><td> $1 . 1 5 3 \times 1 0 ^ { - 1 }$ </td><td> $1 . 1 3 8 \times 1 0 ^ { - 1 }$ </td></tr><tr><td rowspan="2">MC-3D</td><td>Rollout MSE</td><td> $2 . 6 2 3 \times 1 0 ^ { - 5 }$ </td><td> $1 . 6 3 7 \times 1 0 ^ { - 5 }$ </td><td> $1 . 5 2 5 \times 1 0 ^ { - 5 }$ </td><td> $\mathbf { 1 . 4 7 3 \times 1 0 ^ { - 5 } }$ </td></tr><tr><td>Rollout abs div</td><td> $2 . 5 2 1 \times 1 0 ^ { - 2 }$ </td><td> $\mathbf { 2 . 0 2 7 \times 1 0 ^ { - 2 } }$ </td><td> $2 . 2 8 8 \times 1 0 ^ { - 2 }$ </td><td> $2 . 0 8 4 \times 1 0 ^ { - 2 }$ </td></tr></table>

Table 13: Relative $L ^ { 2 }$ rollout error, MSE, and absolute divergence for the model configurations in Fig. 7.
<table><tr><td>Dataset</td><td>Metric</td><td>Single</td><td>Split</td><td>CL Split</td><td>Split+CL +SS</td><td>Split+CL +SS+CC</td></tr><tr><td>Spectral-2D</td><td>Rollout rel.  $L ^ { 2 }$ </td><td> $1 . 6 2 6 \times 1 0 ^ { - 3 }$ </td><td> $2 . 6 3 0 \times 1 0 ^ { - 3 }$ </td><td> $1 . 0 3 1 \times 1 0 ^ { - 3 }$ </td><td> $5 . 9 0 8 \times 1 0 ^ { - 4 }$ </td><td> $\mathbf { 5 . 5 1 1 \times 1 0 ^ { - 4 } }$ </td></tr><tr><td></td><td>Rollout MSE</td><td> $1 . 0 0 5 \times 1 0 ^ { - 7 }$ </td><td> $2 . 6 3 1 \times 1 0 ^ { - 7 }$ </td><td> $4 . 0 4 5 \times 1 0 ^ { - 8 }$ </td><td> $1 . 3 2 7 \times 1 0 ^ { - 8 }$ </td><td> $\mathbf { 1 . 1 5 5 \times 1 0 ^ { - 8 } }$ </td></tr><tr><td></td><td>Rollout abs div</td><td> $5 . 0 2 9 \times 1 0 ^ { - 3 }$ </td><td> $5 . 1 5 7 \times 1 0 ^ { - 3 }$ </td><td> $4 . 2 7 6 \times 1 0 ^ { - 3 }$ </td><td> $4 . 1 0 8 \times 1 0 ^ { - 3 }$ </td><td> $\mathbf { 3 . 6 2 9 \times 1 0 ^ { - 3 } }$ </td></tr><tr><td>Basilisk-2D</td><td>Rollout rel.  $L ^ { 2 }$ </td><td> $5 . 2 8 6 \times 1 0 ^ { - 2 }$ </td><td> $4 . 9 1 9 \times 1 0 ^ { - 2 }$ </td><td> $1 . 9 0 1 \times 1 0 ^ { - 2 }$ </td><td> $1 . 6 6 0 \times 1 0 ^ { - 2 }$ </td><td> $\mathbf { 1 . 6 1 6 \times 1 0 ^ { - 2 } }$ </td></tr><tr><td></td><td>Rollout MSE</td><td> $3 . 0 9 9 \times 1 0 ^ { - 5 }$ </td><td> $2 . 6 8 3 \times 1 0 ^ { - 5 }$ </td><td> $4 . 0 1 0 \times 1 0 ^ { - 6 }$ </td><td> $3 . 0 5 7 \times 1 0 ^ { - 6 }$ </td><td> $\mathbf { 2 . 8 9 7 \times 1 0 ^ { - 6 } }$ </td></tr><tr><td></td><td>Rollout abs div</td><td> $5 . 8 2 6 \times 1 0 ^ { - 2 }$ </td><td> $6 . 6 0 2 \times 1 0 ^ { - 2 }$ </td><td> $4 . 1 6 0 \times 1 0 ^ { - 2 }$ </td><td> $3 . 9 5 5 \times 1 0 ^ { - 2 }$ </td><td> $\mathbf { 3 . 7 1 9 \times 1 0 ^ { - 2 } }$ </td></tr><tr><td>MC-2D</td><td>Rollout rel.  $L ^ { 2 }$ </td><td> $7 . 7 8 4 \times 1 0 ^ { - 2 }$ </td><td> $6 . 9 5 3 \times 1 0 ^ { - 2 }$ </td><td> $3 . 9 6 9 \times 1 0 ^ { - 2 }$ </td><td> $2 . 9 3 4 \times 1 0 ^ { - 2 }$ </td><td> $\mathbf { 2 . 5 2 0 \times 1 0 ^ { - 2 } }$ </td></tr><tr><td></td><td>Rollout MSE</td><td> $1 . 7 7 3 \times 1 0 ^ { - 3 }$ </td><td> $1 . 4 1 5 \times 1 0 ^ { - 3 }$ </td><td> $4 . 6 1 1 \times 1 0 ^ { - 4 }$ </td><td> $2 . 5 1 9 \times 1 0 ^ { - 4 }$ </td><td> $\mathbf { 1 . 8 5 8 \times 1 0 ^ { - 4 } }$ </td></tr><tr><td></td><td>Rollout abs div</td><td> $1 . 1 4 6 \times 1 0 ^ { - 1 }$ </td><td> $1 . 1 5 3 \times 1 0 ^ { - 1 }$ </td><td> $9 . 8 3 6 \times 1 0 ^ { - 2 }$ </td><td> $9 . 7 1 1 \times 1 0 ^ { - 2 }$ </td><td> $\mathbf { 9 . 6 0 6 \times 1 0 ^ { - 2 } }$ </td></tr><tr><td>MC-3D</td><td>Rollout rel.  $L ^ { 2 }$ </td><td> $5 . 0 6 9 \times 1 0 ^ { - 2 }$ </td><td> $3 . 8 6 5 \times 1 0 ^ { - 2 }$ </td><td> $\phantom { + } 3 . 1 7 4 \times 1 0 ^ { - 2 }$ </td><td> $\mathbf { 2 . 9 9 5 \times 1 0 ^ { - 2 } }$ </td><td> $3 . 0 3 2 \times 1 0 ^ { - 2 }$ </td></tr><tr><td></td><td>Rollout MSE</td><td> $2 . 6 2 3 \times 1 0 ^ { - 5 }$ </td><td> $1 . 5 2 5 \times 1 0 ^ { - 5 }$ </td><td> $1 . 0 2 8 \times 1 0 ^ { - 5 }$ </td><td> $\mathbf { 9 . 1 5 7 \times 1 0 ^ { - 6 } }$ </td><td> $9 . 3 8 6 \times 1 0 ^ { - 6 }$ </td></tr><tr><td></td><td>Rollout abs div</td><td> $2 . 5 2 1 \times 1 0 ^ { - 2 }$ </td><td> $2 . 2 8 8 \times 1 0 ^ { - 2 }$ </td><td> $1 . 7 6 0 \times 1 0 ^ { - 2 }$ </td><td> $1 . 6 7 5 \times 1 0 ^ { - 2 }$ </td><td> $\mathbf { 1 . 6 2 6 \times 1 0 ^ { - 2 } }$ </td></tr></table>

## E ADJACENT OPERATOR-ERROR CORRELATION

To examine how symmetry sampling affects errors during rollout, we compare the Spectral-2D single FNO with and without SS, keeping the weights fixed, and also evaluate two-operator CC. Let S denote one step of the reference solver. The local operator error at call k is

$$
\mathbf { e } _ { k } = \hat { \mathbf { u } } _ { k + 1 } - S ( \hat { \mathbf { u } } _ { k } ) .\tag{35}
$$

The model and reference solver thus start from the same predicted state $\hat { \mathbf { u } } _ { k }$ , and the rollout continues with the model output $\hat { \mathbf { u } } _ { k + 1 }$ . Unlike the one-step prediction error evaluated from ground-truth inputs in Appendix D, this local operator error is evaluated along the predicted trajectory. We measure its magnitude and the cosine similarity of successive errors,

$$
\operatorname { m e a n } \| \mathbf { e } _ { k } \| , \qquad \operatorname { m e a n } { \frac { \left. \mathbf { e } _ { k - 1 } , \mathbf { e } _ { k } \right. } { \left\| \mathbf { e } _ { k - 1 } \right\| \left\| \mathbf { e } _ { k } \right\| } } ,\tag{36}
$$

where the averages run over test trajectories and operator calls. Norms and inner products are Euclidean over all grid points and velocity components. The cosine measures how consistently successive calls produce errors in the same spatial direction.

Table 14 reports results on 50 test trajectories over 100 steps, with SS sampling independently at each call. SS reduces error correlation and mean local relative error, the latter possibly reflecting less rollout distribution shift. CC lowers correlation but increases local error, potentially because its second operator is harder to optimize on predicted inputs; this trade-off yields worse rollout accuracy here.

Table 14: Local operator errors and rollout error on 50 Spectral-2D test trajectories over 100 steps. Mean local error averages 100 operator calls per trajectory; mean cosine similarity averages 99 consecutive error pairs. Rollout error is the pooled relative $L ^ { \mathbf { \bar { 2 } } }$ error over these trajectories and steps.
<table><tr><td>Model</td><td>Mean local operator error</td><td>Mean cosine</td><td>Rollout error</td></tr><tr><td>Single</td><td> $5 . 0 3 \times 1 0 ^ { - 3 }$ </td><td>0.997</td><td> $1 . 1 7 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Single+SS</td><td> $2 . 9 2 \times 1 0 ^ { - 3 }$ </td><td>0.947</td><td> $6 . 8 6 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Single+CC</td><td> $5 . 6 3 \times 1 0 ^ { - 3 }$ </td><td>0.990</td><td> $1 . 2 7 \times 1 0 ^ { - 3 }$ </td></tr></table>

Figure 9 shows two successive local errors on one trajectory. The baseline produces nearly identical error patterns across the two calls, while both SS and CC alter the spatial distribution of the error.

![](images/fff997c7b744d2ac788cf57b740034350e40467bcac63c7a3dde170fc5f9d2e9.jpg)  
Figure 9: The x-component of two successive local operator errors, with a shared color scale. All variants share the first call, which uses no symmetry transform. The second call uses the baseline, SS, or CC. For this illustration, SS uses the 180<sup>◦</sup> rotation that minimizes the absolute cosine similarity between consecutive error fields among non-identity $D _ { 4 }$ transforms; Table 14 uses independent random symmetry samples.

## F TRAINING WITH MONTE CARLO NOISE

We generate paired MC and deterministic trajectories for a two-dimensional problem without solids in an unbounded domain. As in the other benchmarks, we use $\mathrm { ~ a ~ } 6 4 ^ { 2 }$ grid on the unit square, $\Delta t =$ 0.02, and $\nu = 1 0 ^ { - 3 }$ . Initial fields combine background flow, Fourier modes, vortices, shear, and wakes. Both solvers clamp velocity queries outside the grid and project the initial field before the first time step. The dataset contains 500/50/50 train/validation/test initial fields, with 100 training steps and 200 test steps. Models receive velocity and coordinates, without a geometry channel.

The deterministic BIE solver follows the MC formulation, replacing stochastic integral estimates with deterministic quadrature. Both solvers use the same semi-Lagrangian RK3 advection scheme. For pressure projection, the BIE solver evaluates the volume and pseudo-boundary integrals using grid convolution and boundary quadrature. For diffusion, it convolves the heat kernel with the MC estimator’s bilinear interpolation basis. Over the 200-step test trajectories, its pooled variancecorrected relative $L ^ { 2 }$ error against the MC references is 1.57% (Appendix C).

We train the same single-step FNO on (i) MC trajectories, (ii) BIE trajectories, and (iii) BIE trajectories with Gaussian input noise. In the third setting, we add independent Gaussian noise to each input velocity component and leave the targets unperturbed. We set the noise standard deviation to $4 . { \overset { - } { 6 } } \times 1 0 ^ { - 3 }$ in physical velocity units, matching the pooled per-component RMS MC sampling noise estimated from paired references over the first 100 steps.

All runs use $D _ { 4 }$ training augmentation; the 50- and 10-trajectory subsets select matching initial fields from the 500-trajectory set. Validation and test inputs receive no added Gaussian noise. We evaluate 200-step autoregressive rollouts on both test sets, sampling a $D _ { 4 }$ transform at each prediction step.

Figure 10 shows that adding Gaussian input noise to BIE training brings its rollout errors much closer to those of MC training. MC training performs best with 10 trajectories. Gaussian input noise substantially improves BIE training with limited data but introduces a modest disadvantage with the full training set, consistent with a regularization effect. These trends are similar on the BIE and MC test sets.

## G CORRECTIVE CYCLING VERSUS UNROLLED AND PUSHFORWARD TRAINING

We compare two-operator CC with two-step unrolled and single-step pushforward training on MC-2D. Unrolled training recursively applies the same operator for two time steps and minimizes the average loss against the corresponding ground-truth states, backpropagating through both steps. Pushforward training uses equal numbers of ground-truth inputs and inputs generated by advancing ground-truth states by one step. For both input types, it minimizes a one-step loss against the corresponding next-step ground-truth state. The predicted inputs are regenerated at the start of each epoch using a frozen model snapshot, without backpropagating through their generation. Both baselines apply a single operator throughout inference rollouts.

![](images/b44116262c5ae4bb7eafb04e8240277b6eaa4a8b3e4beb474f731cb6db513683.jpg)  
Figure 10: Relative $L ^ { 2 }$ rollout error after training on 500, 50, or 10 trajectories. MC test errors use the paired-reference pooled variance correction of Appendix $\mathrm { C } ;$ BIE test errors use standard relative $L ^ { 2 }$ error.

Full training follows the standard optimization schedule of Appendix B, with a separately trained base operator for CC. Fine-tuning initializes the trainable operator from the same pretrained base and runs for 100 epochs at one-tenth of the original learning rate, with cosine decay and no early stopping. Table 15 reports 200-step rollout error and training cost. All runs use $D _ { 4 }$ symmetry augmentation during training and are evaluated without symmetry sampling (SS). Errors use the pooled variance-corrected metric of Appendix C.

Table 15: Training cost on one NVIDIA B200 and 200-step relative $L ^ { 2 }$ rollout error on MC-2D NACA. Times are in hours:minutes:seconds. Fine-tuning and CC times exclude base training; CC times include data export. Full CC takes 8:59:55 including base training, still less than unrolled training. GPU memory is the peak training allocation. Bold marks the lowest values within each comparison.
<table><tr><td>Comparison</td><td>Method</td><td>Time</td><td></td><td>Rollout error Peak GPU (GB)</td></tr><tr><td>Reference</td><td>Base</td><td>4:14:16</td><td>0.078</td><td>6.727</td></tr><tr><td rowspan="2">Full training</td><td>Unrolled</td><td>9:13:05</td><td>0.055</td><td>11.157</td></tr><tr><td>CC</td><td>4:45:39</td><td>0.062</td><td>6.727</td></tr><tr><td rowspan="3">Fine-tuning</td><td>Unrolled</td><td>2:51:35</td><td>0.073</td><td>11.073</td></tr><tr><td>CC</td><td>1:38:19</td><td>0.074</td><td>6.727</td></tr><tr><td>Pushforward</td><td>12:18:18</td><td>0.076</td><td>12.801</td></tr></table>

CC approaches the rollout accuracy of unrolled training with lower training time and substantially less peak GPU memory; in fine-tuning, it also improves on pushforward in accuracy and cost. CC trains each successive operator on states generated by its frozen predecessors, allowing it to adapt to the accumulated prediction errors in its inputs.