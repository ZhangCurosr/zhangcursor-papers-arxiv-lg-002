# Identifying the Shear Piezoelectric Coeficient using Physics-Informed Neural Networks under Exact Boundary and Initial Constraints

Suhas Suresh Bharadwaj<sup>∗</sup>

Department of Electrical and Electronics Engineering

BITS Pilani, Dubai Campus, Dubai, United Arab Emirates

(Dated: November 3, 2026)

The shear piezoelectric coeficient e governs the shear-mode electromechanical coupling in piezoelectric micromachined ultrasonic transducers (PMUTs) and surface acoustic wave (SAW) devices, yet its measurement typically requires destructive resonance testing on dedicated specimens. This paper shows, on simulated data, that $e _ { \mathrm { 1 5 } }$ can instead be recovered non-destructively from sparse, noisy subsurface wavefield observations using a constrained physics-informed neural network (PINN). By building exact spatial distance functions and temporal multipliers directly into the network output, the clamped, grounded boundary conditions and the resting initial state are satisfied automatically, leaving the loss function with only the interior equations of coupled electroelastodynamics and the data misfit. Along with a non-dimensional scaling that brings all governing equation coeficients within a single decade of O(1), this setup removes the boundary and initialcondition penalty weights that soft-constraint methods have to tune. In a six-way comparison on a plane-strain PZT-5H problem with a manufactured wavefield and 30 dB noise, a soft-penalty baseline with adaptive weights converges to an erroneous $e _ { \mathrm { 1 5 } }$ in four of the five runs (51.7% mean error), and fixed penalties reach an error of 0.85% only after a second-order optimization stage. In contrast, the constrained formulations proposed here reach mean $e _ { 1 5 }$ errors between 0.15% and 0.51% with held-out displacement errors below 1%. Within these variants, the spatial multiplier gives rapid first-order convergence, and the quadratic temporal multiplier lowers the early-time electrostatic residual by a factor of 6.8 compared with a linear one. Tested on finite-element data along a single subsurface line, the full formulation recovers $e _ { \mathrm { 1 5 } }$ with a mean error of 1.19% at 30 dB and keeps the mean error between 0.65% and 2.71% across a noise sweep from 40 dB down to 0 dB. Finally, since the internal electric potential cannot be measured in practice, training on mechanical displacements alone still recovers $e _ { 1 5 }$ with mean errors of 0.45% on finite-element data and 1.64% on manufactured data.

## I. INTRODUCTION

Characterizing the electromechanical coupling tensor of anisotropic thin films typically requires destructive resonance testing. For transversely isotropic media such as Lead Zirconate Titanate (PZT) and Scandium Aluminum Nitride (ScAlN) [1], the diagonal piezoelectric stress components are easy to extract from standard thickness-stretch and radial resonant modes [2, 3]. The shear coeficient $e _ { 1 5 }$ is harder. It controls how eficiently lateral shear waves are excited and sets the shearhorizontal electromechanical coupling, and it is conventionally isolated through pure shear resonances in specifically diced, laterally poled samples [4]. Inverse procedures that match simulated and measured electrical impedance can identify complete sets of piezoelectric parameters [5], but they also work on dedicated specimens and not on embedded or in-situ structures. This limits quality control during the fabrication of piezoelectric micromachined ultrasonic transducers (PMUTs) [6] and surface acoustic wave (SAW) devices [7], where wafer-towafer variation in $e _ { 1 5 }$ changes the resonant frequency, the coupling eficiency, and the bandwidth of the finished transducer. Embedded subsurface sensing, such as fiber Bragg grating arrays in structural composites [8, 9], shows that internal fields can be interrogated without destructive access. This paper therefore tests whether $e _ { 1 5 }$ can be recovered from sparse, one-sided, subsurface observations instead of full-field measurements, which would allow in-situ, non-destructive characterization.

Existing NDE approaches for thin-film piezoelectric characterization include laser interferometry [10], electrical impedance spectroscopy [11], and Brillouin light scattering [12]. While these methods can resolve certain elastic and piezoelectric constants with high accu racy, isolating $e _ { 1 5 }$ non-destructively is generally dificult, since its signature in the surface displacement spectrum is coupled to, and can be partially masked by, the dominant longitudinal constants $( c _ { 3 3 } , \ e _ { 3 3 } )$ . Decoupling $e _ { 1 5 }$ from an acoustic measurement therefore requires solving a nonlinear inverse parameter identification problem over a strongly coupled, anisotropic, stif transient wavefield [13, 14].

Finite element (FE) based inversion [15] combined with adjoint-state gradients provides a well-established route to full-waveform inversion [16, 17], but each iteration requires forward and adjoint wavefield solves over many time steps, and the cost grows with mesh density and with the number of unknown fields. Mesh-free surrogate models have therefore been explored as alternatives, particularly when several fields must be inferred jointly from sparse observations [18]. In the present work, the recovery of a single coeficient serves as a controlled test bed for the constrained architecture, which accommodates further unknown coeficients as additional trainable parameters.

Scientific Machine Learning (SciML) ofers a mesh-free alternative for problems of this kind. Physics-Informed Neural Networks (PINNs) [19–21] write the governing partial diferential equations (PDEs) into the training objective of a neural network and compute the required derivatives by reverse-mode Automatic Diferentiation (AD) [22]. They have been applied to forward and inverse problems in fluid dynamics [23], linear elastodynamics [24, 25], and acoustic NDE [26]. For piezoelectric systems, PINNs have been used for plate modelling [27] and shunted piezoelectric systems [28], but, to the author’s knowledge, they do not recover the shear coeficient e<sub>15</sub> from sparse transient wavefield data.

Despite these successes, standard PINN formulations often struggle with stif, strongly coupled multiphysics problems such as piezoelectricity, largely because the physical scales of the mechanical and electrical terms differ so much. In standard continuous PINNs, for example those built with the DeepXDE framework [29], the training objective is a sum of the interior PDE residuals and penalty terms for the boundary conditions (BC) and initial conditions (IC):

$$
\begin{array} { r } { \mathcal { L } ( \theta ) = \mathcal { L } _ { \mathrm { P D E } } ( \theta ) + \lambda _ { \mathrm { B C } } \mathcal { L } _ { \mathrm { B C } } ( \theta ) + \lambda _ { \mathrm { I C } } \mathcal { L } _ { \mathrm { I C } } ( \theta ) , } \end{array}\tag{1}
$$

where θ represents the network weights and λ are penalty hyperparameters. Wang et al. [30] and Krishnapriyan et al. [31] showed that enforcing boundary conditions through soft penalties can create gradient-flow pathologies in stif systems, and piezoelectricity sits toward the extreme end of that spectrum. The elastic stif nesses $( c _ { i j } ^ { E } \sim 1 0 ^ { 1 0 } \ \mathrm { P a } )$ and the dielectric permittivities $( \varepsilon _ { i j } ^ { S } \sim 1 0 ^ { - 8 } ~ \mathrm { F / m } )$ difer by many orders of magnitude as SI numbers, and the mechanical and electrical residuals carry diferent units $\mathrm { { ( N / m ^ { 3 } } }$ and $\mathrm { C / m ^ { 3 } } )$ , so in an unscaled mean squared error their raw magnitudes can be many orders apart. Wang et al. analyzed imbalances of this kind for general PINNs through gradient statistics [30] and Neural Tangent Kernel (NTK) eigenvalues [32], and a comparable imbalance is expected here.

A common remedy is self-adaptive loss weighting or NTK-based balancing, which adjusts the λ values during training [32–34]. These methods help in many problems, but Sec. IV A shows that a gradient-norm version of them is not reliable for the inverse problem studied here. Soft penalties also have a second weakness in this problem. The boundaries are only approximately satisfied early in training, so the simulated wave picks up phase errors, and one possible consequence is that the optimizer lowers the data loss by pushing $e _ { 1 5 }$ upward instead of fixing the boundaries. Through piezoelectric stifening, $c _ { 4 4 } ^ { D } \approx c _ { 4 4 } ^ { E } + e _ { 1 5 } ^ { 2 } / \varepsilon _ { 1 1 } \ [ 2 ]$ , a larger $e _ { 1 5 }$ raises the efective shear wave speed and could bring the phase back into line at the observation plane, which would resemble the deceptive convergence described by Krishnapriyan et al. [31]. Sec. IV A treats this as a hypothesis that the experiments support but do not prove. The alternative is to enforce the boundaries exactly instead of through penalties. This goes back to Lagaris, Likas, and Fotiadis [35], who showed that trial solutions satisfying the boundary conditions by construction reduce the optimization to the interior of the domain.

Sukumar and Srivastava [36] later showed that distance functions can impose spatial boundary conditions exactly on general geometries, and hard enforcement of initial and boundary conditions has also been used for transient elastodynamics [25]. The present work builds on these ideas and on a one-dimensional electro-elastodynamic formulation with exact geometric constraints reported earlier by the author [37]. It extends them to a twodimensional coupled hyperbolic-elliptic piezoelectric system, examines how the quasi-static potential behaves near $t = 0$ , and, to the author’s knowledge, uses a hardconstrained network for the first time to identify a piezoelectric coeficient, $e _ { 1 5 } .$ , from noisy transient observations.

The piezoelectric system in the acoustic regime is a coupled hyperbolic-elliptic problem. The mechanical fields obey wave equations, while the electric potential is fixed at every instant by Gauss’s law, which acts as an instantaneous constraint on the mechanical strain. This makes the temporal behavior of the potential near $t = 0$ matter. If the displacement grows as $\mathcal { O } ( t ^ { 2 } )$ from rest, Gauss’s law requires the potential to grow at the same order. A linear multiplier on the potential generically leaves an $\mathcal { O } ( t )$ residual in the Poisson equation at early times, whereas a quadratic multiplier leaves only $\mathcal { O } ( t ^ { 2 } )$ This compatibility between the temporal orders of the mechanical and electrical fields is referred to here as the initial-state electrostatic inconsistency, and it is examined in Sec. III A and measured in Sec. IV B.

The framework is tested in four steps. The first uses the Method of Manufactured Solutions (MMS). Because MMS supplies an exact solution, recovery of $e _ { 1 5 }$ under 30 dB additive Gaussian noise can be judged without any finite-element discretization error. In this step, six training formulations are compared on identical data: two soft-penalty baselines and four constrained variants that separate what the spatial and the temporal factors each contribute (Sec. IV B). The second step checks the full formulation against a finite-element solution of the same forced problem (Sec. III C). The third is a sweep over signal-to-noise ratios that shows how the identification degrades as the noise grows (Sec. IV F). The fourth removes the potential from the data and keeps only the displacement observations (Sec. IV G).

The main contributions of this work are summarized as follows:

• Controlled comparison on identical data. Two soft-penalty baselines and four constrained variants are trained on the same five seeds, the same noisy observations, and the same initial weights. Adaptive penalties fail in four of five runs $( 5 1 . 7 \% \pm 3 1 . 3 \%$ mean error), fixed penalties reach $0 . 8 4 9 \% \pm 0 . 2 4 5 \%$ only after a long second-order stage, and the constrained variants reach 0.15% to 0.51% with held-out $u _ { 3 }$ errors below 1%.

• What each constraint does. The spatial factor alone gives the cleanest convergence (0.29% error after the first-order stage). The temporal factor lowers the early-time Gauss residual by a factor of 6.8 when the potential carries $t ^ { 2 }$ instead of $t ,$ but it does not change the accuracy of $e _ { 1 5 }$ noticeably.

• Compatibility analysis of the temporal factor. The temporal orders of the mechanical and electrical fields near $t = 0$ are analyzed for the coupled hyperbolic-elliptic system. A linear multiplier on the quasi-static potential generically leaves an $\mathcal { O } ( t )$ Poisson residual, whereas a quadratic multiplier leaves an $\mathcal { O } ( t ^ { 2 } )$ residual, and the measured residuals follow this ordering.

• Non-dimensional scaling. The scaling brings all PDE coeficients to within about one decade of $\mathcal { O } ( 1 )$ , even though the stifnesses are of order $1 0 ^ { 1 0 }$ Pa and the permittivities of order $1 0 ^ { - 8 } \ \mathrm { F / m }$

• FEM cross-check and noise sweep. Applied to observations on the line $\bar { x } _ { 3 } = 0 . 9$ from a finiteelement solution of the same forced problem, with 30 dB noise, the full formulation recovers $e _ { 1 5 }$ with a mean error of $1 . 1 9 \% \pm 0 . 6 5 \%$ . A noise sweep from 40 dB to 0 dB $( N = 5$ runs per level, with the initialization, collocation points, and noise redrawn each time) keeps the mean error below 1.5% down to 5 dB.

• Displacement-only observations. With the potential removed from the data, $e _ { 1 5 }$ is still recovered, with mean errors of 0.45% on the finite-element data and 1.64% on the manufactured data, although the first stage alone is then much less accurate and the recovery depends on L-BFGS.

The rest of the paper follows the same order. $\mathrm { S e c - }$ tion II states the governing equations and their nondimensional form, Section III describes the constrained network, the test problems, and the training setups, Sec tion IV presents the results, Section V states the limitations, and Section VI concludes.

## II. THEORETICAL FORMULATION

This section defines the physical problem before any network is introduced. It states the kinematics, justifies the quasi-static approximation, writes out the coupled equations, and then rescales them, because the large gap between the mechanical and electrical coeficients comes from the PDEs themselves and not from the network.

## A. Infinitesimal Strain and Plane-Strain Kinematics

The transient acoustic wavefield is evaluated in a twodimensional continuous domain $\Omega \subset \mathbb { R } ^ { 2 }$ , representing the $x _ { 1 } - x _ { 3 }$ cross-section of a class 6mm transversely isotropic crystal (Lead Zirconate Titanate, PZT-5H), poled along the x -axis (see Fig. 1). Under the assumption of infinitesimal deformation, the Green-Lagrange strain tensor linearizes to the Cauchy strain tensor $\begin{array} { r } { S _ { i j } = \frac { 1 } { 2 } ( u _ { i , j } + } \end{array}$ $u _ { j , i } )$ , where $\mathbf { u } ( x , t )$ is the displacement vector field.

![](images/ae35beda5fdde9ccb6ce021805487d4228abb2f1822d29eec6eb61eb4ad9ff57.jpg)  
FIG. 1. Schematic of the two-dimensional plane-strain computational domain for the PZT-5H medium. The perimeter is fully clamped $( u _ { i } = 0 )$ and electrically grounded $( \phi = 0 )$ enforcing strict Dirichlet boundary conditions across the entire domain boundary. The physical orientation defines the bottom boundary at $x _ { 3 } = 0$ and the top boundary at $x _ { 3 } = 1$ For the non-destructive inverse evaluation, acoustic data are taken exclusively from a subsurface observation plane at normalized height $x _ { 3 } = 0 . 9 \ ( $ 1 mm below the top surface of the 10 mm domain), with $N _ { s } = 5 \small { , } 0 0 0$ space–time observation samples (not physical sensors) in the manufactured-solution tests.

To model bulk wave mechanics while mitigating outof-plane scattering modes, a strict state of plane strain $( u _ { 2 } = 0 , \ \partial / \partial x _ { 2 } = 0 )$ is enforced. While SAW devices heavily exploit the out-of-plane shear-horizontal (SH) mode, the identical tensor coeficient $e _ { 1 5 }$ remains fully active in the in-plane (P-SV) system through the engineering shear stress $T _ { 5 } = c _ { 4 4 } ( u _ { 1 , 3 } + u _ { 3 , 1 } ) + e _ { 1 5 } \phi ,$ <sub>1</sub> and the electric displacement $D _ { 1 }$ . Because $e _ { 1 5 }$ is a material property, the in-plane (P-SV) system recovers the same value that governs the SH mode in later 3D device design, with a simpler and cheaper computation.

The mechanical state is completely described by the in-plane displacement vector $\mathbf { u } = ( u _ { 1 } , u _ { 3 } ) ^ { \top }$ . The nontrivial strain components in Voigt notation reduce to the longitudinal strains $S _ { 1 } = u _ { 1 , 1 }$ and $S _ { 3 } = u _ { 3 , 3 }$ , alongside the engineering shear strain $S _ { 5 } = u _ { 1 , 3 } + u _ { 3 , 1 }$ Plane strain leaves three unknowns $( u _ { 1 } , u _ { 3 } , \phi )$ and keeps the 2D validation inexpensive. Going to three dimensions adds a displacement component but does not change the constrained architecture. The fully clamped, electrically grounded boundary $( u _ { i } = 0 , \phi = 0$ on ∂Ω) is the simplest homogeneous Dirichlet setup and keeps the boundary conditions from complicating the identification of $e _ { 1 5 }$ Traction-free (Neumann) boundaries, which are relevant for free-standing resonators and SAW substrates, need a diferent construction within the distance-function framework [36] and are the natural next step.

## B. Electrodynamic Decoupling via Asymptotic Scaling

As the crystal lattice deforms dynamically, an internal polarization field is created. To justify using the electrostatic approximation for this dynamic system, the full electrodynamic potentials are examined. The electric field E is defined by the scalar potential ϕ and the magnetic vector potential A via $\mathbf { E } = - \nabla \phi - \partial \mathbf { A } / \partial t$

The dimensionless parameter $\beta = v _ { a } / c$ is introduced, representing the ratio of the bulk acoustic wave velocity to the speed of light in the dielectric medium. Because the acoustic wave speed is orders of magnitude slower than the speed of light in the dielectric medium, the magnetic vector potential is eliminated via standard quasi-static electrodynamic scaling [4, 38].

For piezoceramics like PZT-5H, the quasi-transverse and quasi-longitudinal acoustic wave speeds are approximately $v _ { t } \approx 1 7 5 1$ m/s and $v _ { L } \approx 3 9 5 0 ~ \mathrm { m / s }$ . The speed of light in the dielectric medium is $c \approx c _ { \mathrm { v a c } } / \sqrt { \varepsilon _ { r } } \sim$ $7 . 3 \times 1 0 ^ { 6 } ~ \mathrm { m / s }$ . Using the upper bound $v _ { L }$ yields a maximum ratio of $\beta \sim 5 . 4 \times 1 0 ^ { - 4 } \ll 1$ . Expanding the electromagnetic wave equation through asymptotic perturbation reveals that the magnetic vector potential scales as $\mathcal { O } ( \beta ^ { 2 } )$ . Taking the limit $\beta  0$ mathematically eliminates the transient magnetic term, reducing the electric field entirely to its irrotational component, $E _ { i } = - \phi _ { , i }$

## C. The Coupled Multiphysics Boundary Value Problem

Operating under this irrotational (quasi-static) limit, the macroscopic behavior of the continuum is governed by the linear IEEE stress-charge constitutive relations [3]. Expanding these relations for the 6mm crystal class under the derived kinematic and electrodynamic restrictions yields the following expressions for the coupled Cauchy stress (T) and electric displacement (D) tensors:

$$
T _ { 1 } = c _ { 1 1 } u _ { 1 , 1 } + c _ { 1 3 } u _ { 3 , 3 } + e _ { 3 1 } \phi _ { , 3 } ,\tag{2}
$$

$$
T _ { 3 } = c _ { 1 3 } u _ { 1 , 1 } + c _ { 3 3 } u _ { 3 , 3 } + e _ { 3 3 } \phi _ { , 3 } ,\tag{3}
$$

$$
T _ { 5 } = c _ { 4 4 } ( u _ { 1 , 3 } + u _ { 3 , 1 } ) + e _ { 1 5 } \phi _ { , 1 } ,\tag{4}
$$

$$
D _ { 1 } = e _ { 1 5 } ( u _ { 1 , 3 } + u _ { 3 , 1 } ) - \varepsilon _ { 1 1 } \phi _ { , 1 } ,\tag{5}
$$

$$
D _ { 3 } = e _ { 3 1 } u _ { 1 , 1 } + e _ { 3 3 } u _ { 3 , 3 } - \varepsilon _ { 3 3 } \phi _ { , 3 } ,\tag{6}
$$

where $c _ { i j }$ is the elastic stifness at constant electric field, $e _ { i j }$ is the piezoelectric stress constant, and $\varepsilon _ { i j }$ is the dielectric permittivity at constant strain.

Substituting these constitutive laws into the conservation of linear momentum $( \rho \ddot { u } _ { i } = T _ { i j , j } )$ and the charge conservation for an insulating dielectric $( D _ { i , i } = 0 )$ yields the governing PDE system over $\Omega \times ( 0 , T ]$

$$
\rho \ddot { u } _ { 1 } = c _ { 1 1 } u _ { 1 , 1 1 } + c _ { 4 4 } u _ { 1 , 3 3 } + \left( c _ { 1 3 } + c _ { 4 4 } \right) u _ { 3 , 1 3 }
$$

$$
+ \left( e _ { 3 1 } + e _ { 1 5 } \right) \phi _ { , 1 3 } ,\tag{7}
$$

$$
\rho \ddot { u } _ { 3 } = c _ { 4 4 } u _ { 3 , 1 1 } + c _ { 3 3 } u _ { 3 , 3 3 } + \left( c _ { 1 3 } + c _ { 4 4 } \right) u _ { 1 , 1 3 }
$$

$$
+ e _ { 1 5 } \phi _ { , 1 1 } + e _ { 3 3 } \phi _ { , 3 3 } ,\tag{8}
$$

$$
\begin{array} { c } { 0 = \left( e _ { 1 5 } + e _ { 3 1 } \right) u _ { 1 , 1 3 } + e _ { 1 5 } u _ { 3 , 1 1 } + e _ { 3 3 } u _ { 3 , 3 3 } } \\ { - \varepsilon _ { 1 1 } \phi _ { , 1 1 } - \varepsilon _ { 3 3 } \phi _ { , 3 3 } . } \end{array}\tag{9}
$$

This system establishes a strictly coupled hyperbolicelliptic diferential operator. The two momentum equations are hyperbolic (wave type), while Gauss’s law (9) is elliptic and couples the mixed partial derivatives of the mechanical fields $( u _ { 1 , 1 3 } , u _ { 3 , 1 1 } )$ to the electrostatic potential, requiring simultaneous resolution of wave propagation and the instantaneous electrostatic constraint.

## D. Dimensional Scaling and Stabilization

When the original boundary value problem is turned into a training objective, the loss landscape is shaped by the sizes of the diferential coeficients. For PZT-5H the stifnesses are $\mathcal { O } ( 1 0 ^ { 1 0 } )$ Pa and the permittivities $\mathcal { O } ( 1 0 ^ { - 8 } )$ $\mathrm { F / m } ,$ , so the SI coeficients in the governing PDEs difer by many orders of magnitude, although they carry diferent units and cannot be compared directly. The mechan ical and electrical residuals also carry diferent units, so their raw magnitudes in an unscaled mean squared error can difer by many orders of magnitude, with the exact ratio depending on the unit system and the field scales. Such imbalances are expected to make the Hessian of the objective poorly conditioned and to hinder both second-order optimizers (L-BFGS) and first-order methods (Adam).

To resolve this scale gap, the equations are nondimensionalized. Spatial coordinates are normalized by a characteristic macroscopic length L (equal to the dimensional side length of the domain), defining $x _ { i } = L \bar { x } _ { i }$ The temporal coordinate is normalized by the bulk quasitransverse wave transit time $t _ { 0 } = L \sqrt { \rho / c _ { 4 4 } }$ (5.71 µs for $L = 1 0 ~ \mathrm { m m } )$ , yielding $t = t _ { 0 } \bar { t }$ The reference stifness $c _ { 4 4 }$ is deliberately selected as the normalization anchor because it directly governs the shear wave mechanics from which $e _ { 1 5 }$ must be extracted. The mechanical displacement is scaled by a reference amplitude $U \left( U = 1 \right.$ nm in all computations), defining $u _ { i } = U \bar { u } _ { i }$

To enforce energetic symmetry between the hyperbolic and elliptic operators, the electric potential is scaled proportionally to the mechanical strain-induced polarization. The reference potential is defined using the primary longitudinal piezoelectric constant $e _ { 3 3 }$ and the outof-plane permittivity $\varepsilon _ { 3 3 }$ as $\Phi _ { 0 } ~ = ~ U ( e _ { 3 3 } / \varepsilon _ { 3 3 } )$ , setting $\phi = \Phi _ { 0 } \bar { \phi } .$ . This choice ensures that the piezoelectric coupling terms in the dimensionless Poisson equation reduce to $\mathcal { O } ( 1 )$ coeficients.

Applying the chain rule, the diferential operators transform as $\partial _ { x _ { i } } = L ^ { - 1 } \partial _ { \bar { x } _ { i } }$ and $\partial _ { t t } = ( c _ { 4 4 } / \rho L ^ { 2 } ) \partial _ { \overline { { t t } } }$ . Substituting these into the PDE system isolates the dimensionless normalization parameter:

$$
k _ { 0 } ^ { 2 } = \frac { e _ { 3 3 } ^ { 2 } } { c _ { 4 4 } \varepsilon _ { 3 3 } } .\tag{10}
$$

Note that $k _ { 0 } ^ { 2 }$ is a mathematical scaling ratio defined specifically to balance the $\mathcal { O } ( 1 )$ dimensionless Poisson equation. It should not be conflated with the standard physical electromechanical coupling coeficient $k _ { 3 3 } ^ { 2 } ,$ which is strictly bounded below unity by thermodynamic constraints. For PZT-5H, using the values $e _ { 3 3 } = 2 3 . 3 \ : \mathrm { C / m ^ { 2 } }$ $c _ { 4 4 } = 2 . 3 0 \times 1 0 ^ { 1 0 }$ Pa, and $\dot { \varepsilon } _ { 3 3 } ^ { S } = 1 . 3 0 \times 1 0 ^ { - 8 } ~ \mathrm { F / m }$ , this evaluates to $k _ { 0 } ^ { 2 } \approx 1 . 8 2$ . With this scaling, the dimensionless PDE coeficients of Eqs. (11)–(13) span about one decade, from $\bar { e } _ { 1 5 } + \bar { e } _ { 3 1 }$ ≈ 0.45 up to $\bar { c } _ { 1 1 } \approx 5 . 5$ This is not perfectly uniform, but it is a range in which first- and second-order optimizers are generally expected to work without per-term adaptive weighting.

The complete set of PZT-5H material constants used in this work, adapted from the IEEE piezoelectricity standard [3], is listed in Table I.

TABLE I. Material constants for PZT-5H used in the 2D plane-strain model, adapted from the IEEE piezoelectricity standard [3], with $c _ { 1 3 }$ set to the value used throughout this work (in the network, the manufactured sources, and the COMSOL model). While $c _ { 1 2 } ^ { E } \ = \ 8 . 0 2 \times 1 0 ^ { 1 0 }$ Pa is a nonzero property of the 6mm crystal class, its corresponding term identically vanishes from the governing equations under the strict plane-strain kinematic constraint $( S _ { 2 } = 0 )$
<table><tr><td>Parameter</td><td>Symbol</td><td>Value</td></tr><tr><td>Density</td><td> $\rho$ </td><td> $7 5 0 0 ~ \mathrm { k g / m ^ { 3 } }$ </td></tr><tr><td>Stiffness (11)</td><td> $c _ { 1 1 } ^ { E }$ </td><td> $1 2 . 6 \times 1 0 ^ { 1 0 } \ \mathrm { P a }$ </td></tr><tr><td>Stiffness (12)</td><td> $c _ { 1 2 } ^ { E }$ </td><td> $8 . 0 2 \times 1 0 ^ { 1 0 } \ \mathrm { P a }$ </td></tr><tr><td>Stiffness (13)</td><td> $c _ { 1 3 } ^ { E }$ </td><td> $5 . 3 \times 1 0 ^ { 1 0 } \ \mathrm { P a }$ </td></tr><tr><td>Stiffness (33)</td><td> $c _ { 3 3 } ^ { E }$ </td><td> $1 1 . 7 \times 1 0 ^ { 1 0 }$  Pa</td></tr><tr><td>Stiffness (44)</td><td> $c _ { 4 4 } ^ { E }$ </td><td> $2 . 3 0 \times 1 0 ^ { 1 0 } \ \mathrm { P a }$ </td></tr><tr><td>Piezo. stress (15)</td><td> $e _ { \mathrm { 1 5 } }$ </td><td> $1 7 . 0 ~ \mathrm { C / m ^ { 2 } }$ </td></tr><tr><td>Piezo. stress (31)</td><td> $e _ { 3 1 }$ </td><td> $- 6 . 5 ~ \mathrm { C } / \mathrm { m } ^ { 2 }$ </td></tr><tr><td>Piezo. stress (33)</td><td> $e _ { 3 3 }$ </td><td> $2 3 . 3 ~ \mathrm { C } / \mathrm { m } ^ { 2 }$ </td></tr><tr><td>Permittivity (11)</td><td> $\varepsilon _ { 1 1 } ^ { S }$ </td><td> $1 . 5 0 3 \times 1 0 ^ { - 8 } ~ \mathrm { F / m }$ </td></tr><tr><td>Permittivity (33)</td><td> $\varepsilon _ { 3 3 } ^ { S }$ </td><td> $1 . 3 0 0 \times 1 0 ^ { - 8 } \ \mathrm { F / m }$ </td></tr></table>

The finalized, dimensionally stabilized PDEs are governed entirely by dimensionless tensor ratios $\begin{array} { r l } { ( \bar { c } _ { i j } } & { { } = } \end{array}$ $c _ { i j } / c _ { 4 4 } , \bar { e } _ { i j } = e _ { i j } / e _ { 3 3 } , \bar { \varepsilon } _ { i j } = \varepsilon _ { i j } / \varepsilon _ { 3 3 } )$

$$
\begin{array} { r l } & { \ddot { \bar { u } } _ { 1 } = \bar { c } _ { 1 1 } \bar { u } _ { 1 , 1 1 } + \bar { u } _ { 1 , 3 3 } + \left( \bar { c } _ { 1 3 } + 1 \right) \bar { u } _ { 3 , 1 3 } } \\ & { ~ + k _ { 0 } ^ { 2 } ( \bar { e } _ { 3 1 } + \bar { e } _ { 1 5 } ) \bar { \phi } _ { , 1 3 } , } \end{array}
$$

$$
\ddot { \bar { u } } _ { 3 } = \bar { u } _ { 3 , 1 1 } + \bar { c } _ { 3 3 } \bar { u } _ { 3 , 3 3 } + \left( \bar { c } _ { 1 3 } + 1 \right) \bar { u } _ { 1 , 1 3 }\tag{11}
$$

$$
+ k _ { 0 } ^ { 2 } \bar { e } _ { 1 5 } \bar { \phi } _ { , 1 1 } + k _ { 0 } ^ { 2 } \bar { \phi } _ { , 3 3 } ,\tag{12}
$$

$$
0 = \left( \bar { e } _ { 1 5 } + \bar { e } _ { 3 1 } \right) \bar { u } _ { 1 , 1 3 } + \bar { e } _ { 1 5 } \bar { u } _ { 3 , 1 1 } + \bar { u } _ { 3 , 3 3 }
$$

$$
- \bar { \varepsilon } _ { 1 1 } \bar { \phi } _ { , 1 1 } - \bar { \phi } _ { , 3 3 } .\tag{13}
$$

In this form every coeficient of the PDEs is $\mathcal { O } ( 1 )$ within about one decade. This removes one leading source of ill-conditioning in the loss landscape and gives the optimizer a better-scaled gradient field, and the experiments below converge without adaptive loss weights.

The next section turns to how the boundary and initial conditions enter the network.

## III. CONSTRAINED NETWORK, TEST PROBLEMS, AND TRAINING SETUP

## A. Exact Boundary and Temporal Constraints

Standard PINNs enforce boundary and initial conditions by sampling points on the space–time boundary and adding their mean squared error to the objective. In strongly coupled electro-elastodynamics this leaves the initial state only approximately enforced, which is the source of the initial-state electrostatic inconsistency discussed in Sec. I.

To avoid this, the initial displacement, the initial potential, and the initial rates of change of all fields are set to zero algebraically, so that Gauss’s law holds trivially at $t = 0$ before any gradient is evaluated. This is done by building analytical distance functions and temporal multipliers into the forward pass.

For notational simplicity in Eqs. (14)–(20), the overbars on the normalized coordinates $( \bar { x } _ { 1 } , \bar { x } _ { 3 } , \bar { t } ) \in [ 0 , 1 ] ^ { 3 }$ and dimensionless fields are omitted where no ambiguity arises. For a normalized two-dimensional spatial domain clamped and electrically grounded at all external boundaries $( x _ { 1 } \in [ 0 , 1 ] , x _ { 3 } \in [ 0 , 1 ] )$ , the continuous spatial distance boundary multiplier is defined as:

$$
\begin{array} { r } { \mathcal { D } _ { \mathrm { b c } } ( x _ { 1 } , x _ { 3 } ) = x _ { 1 } ( 1 - x _ { 1 } ) x _ { 3 } ( 1 - x _ { 3 } ) . } \end{array}\tag{14}
$$

This function is zero on the perimeter and positive inside the domain. At the center $( x _ { 1 } ~ = ~ x _ { 3 } ~ = ~ 0 . 5 )$ it equals 0.0625 (1/16), so the network weights must scale up to compensate for the reduced amplitude.

Some PINN formulations use a linear temporal multiplier (t). The trial solution $\hat { \boldsymbol { u } } _ { i } ~ = ~ \boldsymbol { t } \cdot \mathcal { N } _ { \theta }$ then has a nonzero initial velocity, $\partial _ { t } ( \hat { u } _ { i } ) | _ { t = 0 } ~ = ~ \mathcal { N } _ { \theta } ~ \neq ~ 0$ , and for hyperbolic wave equations an unphysical initial velocity produces a large transient residual. For the quasi-static potential, a linear multiplier already gives $\hat { \phi } ( x , 0 ) = 0$ and $\nabla \widehat { \phi } ( x , 0 ) = 0$ , so Gauss’s law holds exactly at $t = 0$ . The issue is compatibility at early times. With $\hat { u } _ { i } = \mathcal { O } ( t ^ { 2 } )$ the displacement terms in Gauss’s law are $\mathcal { O } ( t ^ { 2 } )$ while a linear potential generically adds an $\mathcal { O } ( t )$ term through $\varepsilon _ { i j } \hat { \phi } _ { , i j }$ , which leaves a residual of lower order than the mechanical terms. To place all fields at the same order and to give the potential a zero initial rate of change, a second-order temporal multiplier $( t ^ { 2 } )$ is used.

![](images/66ae080e985aa07f110f06abdbd8a987625b4b3006ac3c85d651dfed103c27a1.jpg)  
FIG. 2. Topology of the constrained multiphysics neural network. The raw network outputs $\left( \mathcal { N } _ { \theta } \right)$ are algebraically multiplied by the exact spatio-temporal distance function $( \mathcal { D } _ { b c } \cdot t ^ { 2 } )$ . This forward-pass constraint mathematically guarantees the enforcement of the clamped Dirichlet boundaries and a strictly zero-velocity initial state prior to gradient evaluation, completely removing boundary and initial conditions from the loss function.

Remark on the temporal constraint applied to ϕ. The electric potential follows from Gauss’s law, an instantaneous elliptic (quasi-static) equation that has no initial condition of its own. Since the displacement grows as $\mathcal { O } ( t ^ { 2 } )$ near $t = 0$ , the induced potential does too, and so does the manufactured charge source used below, which is proportional to $[ 1 - \cos ( 2 \pi t ) ]$ . The $t ^ { 2 }$ multiplier on $\hat { \phi }$ therefore matches the asymptotic order of the potential to that of the displacement and sets $\partial _ { t } \hat { \phi } | _ { t = 0 } = 0$ . It is a compatibility condition and not a physical initial condition, and the Gauss residual for $t > 0$ remains a soft interior term in the loss.

The functional approximation of the unconstrained neural network is thus transformed into the exact, geometrically constrained ansatz, as illustrated in Fig. 2.

$$
\hat { u } _ { 1 } ( x _ { 1 } , x _ { 3 } , t ) = \mathcal { D } _ { \mathrm { b c } } ( x _ { 1 } , x _ { 3 } ) \cdot t ^ { 2 } \cdot \mathcal { N } _ { u _ { 1 } } ( x _ { 1 } , x _ { 3 } , t ) ,\tag{15}
$$

$$
\hat { u } _ { 3 } ( x _ { 1 } , x _ { 3 } , t ) = \mathcal { D } _ { \mathrm { b c } } ( x _ { 1 } , x _ { 3 } ) \cdot t ^ { 2 } \cdot \mathcal { N } _ { u _ { 3 } } ( x _ { 1 } , x _ { 3 } , t ) ,\tag{16}
$$

$$
\hat { \phi } ( x _ { 1 } , x _ { 3 } , t ) = \mathcal { D } _ { \mathrm { b c } } ( x _ { 1 } , x _ { 3 } ) \cdot t ^ { 2 } \cdot \mathcal { N } _ { \phi } ( x _ { 1 } , x _ { 3 } , t ) .\tag{17}
$$

This structure guarantees three things: (i) The spatial factor $\mathcal { D } _ { \mathrm { b c } }$ enforces homogeneous Dirichlet conditions on all fields for all $t \geq 0$ . (ii) At $t = 0 , \hat { u } _ { i } = 0$ and $\hat { \phi } = 0$ 7 so the medium starts at rest with zero potential. (iii) $\partial _ { t } ( \hat { u } _ { i } ) | _ { t = 0 } = 0$ and $\partial _ { t } ( \hat { \phi } ) | _ { t = 0 } = 0$ , a zero initial rate of change for the mechanical and electrical fields alike, so Gauss’s law holds at $t ~ = ~ 0$ whatever the random initialization of the network. Because these constraints are built in, the boundary and initial conditions disappear from the loss, and the optimization acts only on the interior coupled physics.

The two factors can also be used on their own to test what each of them does. Sec. III D defines the four constrained variants C1 to C4, and Sec. IV B compares them.

## B. Manufactured-Solution Validation of Inverse Parameter Discovery

The first test of the constrained network on the inverse problem of recovering $e _ { 1 5 }$ of PZT-5H from sparse, onesided subsurface data uses a manufactured solution, for which the exact answer is known.

To keep discretization error out of the first test, the Method of Manufactured Solutions (MMS) [39, 40] is used, which prescribes an exact transient wavefield over the domain. The prescribed solution is chosen to stress the constraints: its envelope and spatial structure differ from the network’s own ansatz ${ \mathcal { D } } _ { \mathrm { b c } } t ^ { 2 } { \mathcal { N } }$ , and only its leading small-time behavior is proportional to $t ^ { 2 }$ . It is a non-polynomial, spatially asymmetric wavefield with distinct spatial wavenumbers and an independent amplitude for each field component. The manufactured fields are prescribed as:

$$
\begin{array} { r } { u _ { 1 } ^ { * } ( x _ { 1 } , x _ { 3 } , t ) = 0 . 0 8 \left[ 1 - \cos ( 2 \pi t ) \right] \sin ( 2 \pi x _ { 1 } ) \sin ( 4 \pi x _ { 3 } ) , } \end{array}\tag{18}
$$

$$
\begin{array} { r } { u _ { 3 } ^ { * } ( x _ { 1 } , x _ { 3 } , t ) = 0 . 1 0 \left[ 1 - \cos ( 2 \pi t ) \right] \sin ( 4 \pi x _ { 1 } ) \sin ( 2 \pi x _ { 3 } ) , } \end{array}\tag{19}
$$

$$
\phi ^ { * } ( x _ { 1 } , x _ { 3 } , t ) = 0 . 1 2 \left[ 1 - \cos ( 2 \pi t ) \right] \sin ( 2 \pi x _ { 1 } ) \sin ( 2 \pi x _ { 3 } ) .\tag{20}
$$

The temporal profile $[ 1 - \cos ( 2 \pi t ) ]$ is non-polynomial, but near $t = 0$ it behaves as $( 2 \pi t ) ^ { 2 } / \bar { 2 }$ . The network output $\mathcal { N } _ { \theta }$ is free to represent the full sinusoidal envelope, so the $t ^ { 2 }$ factor does not restrict the solution to quadratic time dependence. The distinct wavenumbers and amplitudes also keep the three fields from being proportional to one another, so the cross-coupling terms that contain $e _ { 1 5 }$ do not collapse onto a single spatial mode. Because $e _ { \mathrm { 1 5 } }$ enters Eqs. (11)–(13) linearly, the PDE loss is quadratic in $e _ { 1 5 }$ for fixed fields and has a single minimizer whenever the sensitivity terms $k _ { 0 } ^ { 2 } \bar { \phi } _ { , 1 3 } , k _ { 0 } ^ { 2 } \bar { \phi } _ { , 1 1 }$ , and $\bar { u } _ { 1 , 1 3 } + \bar { u } _ { 3 , 1 1 }$ do not vanish identically, which holds for the manufactured fields. Over $\Omega \times [ 0 , 1 ]$ the root-mean-square values of these three terms are 5.3, 5.3, and 10.4, so $e _ { 1 5 }$ is strongly constrained by the PDE residual.

Substituting these fields into the dimensionally stabilized PDEs (11)–(13) via reverse-mode Automatic Differentiation generates the spatio-temporal MMS bodyforce terms $f _ { 1 } ( x _ { 1 } , x _ { 3 } , t ) , f _ { 3 } ( x _ { 1 } , x _ { 3 } , t )$ , and charge density $q ( x _ { 1 } , x _ { 3 } , t )$ required to sustain the manufactured wavefield.

$$
f _ { 1 } \equiv \ddot { \bar { u } } _ { 1 } ^ { * } - { \bar { c } } _ { 1 1 } { \bar { u } } _ { 1 , 1 1 } ^ { * } - { \bar { u } } _ { 1 , 3 3 } ^ { * } - ( { \bar { c } } _ { 1 3 } + 1 ) { \bar { u } } _ { 3 , 1 3 } ^ { * }
$$

$$
- k _ { 0 } ^ { 2 } ( \bar { e } _ { 3 1 } + \bar { e } _ { 1 5 } ^ { \mathrm { t r u e } } ) \bar { \phi } _ { , 1 3 } ^ { * } ,\tag{21}
$$

$$
f _ { 3 } \equiv \ddot { \bar { u } } _ { 3 } ^ { * } - \bar { u } _ { 3 , 1 1 } ^ { * } - \bar { c } _ { 3 3 } \bar { u } _ { 3 , 3 3 } ^ { * } - ( \bar { c } _ { 1 3 } + 1 ) \bar { u } _ { 1 , 1 3 } ^ { * }
$$

$$
- k _ { 0 } ^ { 2 } \bar { e } _ { 1 5 } ^ { \mathrm { t r u e } } \bar { \phi } _ { , 1 1 } ^ { * } - k _ { 0 } ^ { 2 } \bar { \phi } _ { , 3 3 } ^ { * } ,\tag{22}
$$

$$
q \equiv - ( \bar { e } _ { 1 5 } ^ { \mathrm { t r u e } } + \bar { e } _ { 3 1 } ) \bar { u } _ { 1 , 1 3 } ^ { * } - \bar { e } _ { 1 5 } ^ { \mathrm { t r u e } } \bar { u } _ { 3 , 1 1 } ^ { * } - \bar { u } _ { 3 , 3 3 } ^ { * }
$$

$$
+ \bar { \varepsilon } _ { 1 1 } \bar { \phi } _ { , 1 1 } ^ { * } + \bar { \phi } _ { , 3 3 } ^ { * } .\tag{23}
$$

Since $\phi ^ { * }$ is prescribed independently of the mechanical fields, substituting the manufactured fields into Gauss’s law gives a nonzero residual $q ( x _ { 1 } , x _ { 3 } , t )$ . In line with standard MMS practice [39, 40], q is treated as a synthetic source that closes the equations. This lets the test check the optimizer and the parameter recovery on the coupled PDE operators before any finite-element efects enter, without requiring the manufactured fields to belong to the charge-free physical solution manifold. The finiteelement cross-check of Sec. III C keeps the same charge source q and is therefore also a charged manufactured problem.

These forcing terms are computed once from the true parameter $\bar { e } _ { 1 5 } ^ { \mathrm { t r u e } } = 0 . 7 2 9 6 1$ and held fixed during training. The manufactured fields are evaluated over $\Omega \times [ 0 , \bar { T } ]$ with $\bar { T } = 1$ , one full cycle of the [1 − cos] envelope, whereas the finite-element data of Sec. III C cover $0 \leq \bar { t } \leq 0 . 8 7 6$ (0 to $5 ~ \mu \mathrm { { s } ) }$

The PINN receives these forcing terms together with a sparse set of synthetic observations $( u _ { 1 } , u _ { 3 } , \phi )$ collected only on the internal plane $x _ { 3 } = 0 . 9$ . The plane is a simple access model for the subsurface interrogation motivated in Sec. I. The 5,000 observations are space–time samples of $u _ { 1 } , u _ { 3 }$ , and $\phi$ along this plane. Because the potential is fixed to zero on the whole perimeter, the interior potential is kept as an observation in the coupled data loss, and boundary electrode voltages cannot take its place.

All material constants other than $e _ { 1 5 } .$ , including the longitudinal piezoelectric components $( e _ { 3 1 } , ~ e _ { 3 3 } )$ are treated as known, and the shear coeficient $e _ { 1 5 }$ is a trainable scalar initialized at $\bar { e } _ { 1 5 } ^ { ( 0 ) } = 0 . 1 0 \ ( 2 . 3 3 \mathrm { C } / \mathrm { m } ^ { 2 } )$ , an 86% error in magnitude. Each tabulated result is based on five runs with diferent random seeds. The observations are corrupted with 30 dB additive white Gaussian noise, a moderate level used for both the MMS and the FEM tests. The signal-to-noise ratio is defined on a power basis for each field component, so 30 dB and 5 dB correspond to root-mean-square noise-to-signal ratios of about 3.2% and 56%.

Because the distance function enforces the Dirichlet boundaries exactly in both the MMS and the FEM tests (Sec. III C), the objective reduces to the mean squared error of the interior PDE residuals plus the data misfit at the noisy observations:

$$
\mathcal { L } _ { \mathrm { T o t a l } } ( \theta , e _ { 1 5 } ) = \mathcal { L } _ { \mathrm { P D E } } + \lambda \mathcal { L } _ { \mathrm { D a t a } } ,\tag{24}
$$

where the individual loss components are defined as

$$
\mathcal { L } _ { \mathrm { P D E } } = \frac { 1 } { N _ { c } } \sum _ { k = 1 } ^ { N _ { c } } \Bigl [ \bigl ( \mathcal { R } _ { 1 } ^ { ( k ) } \bigr ) ^ { 2 } + \bigl ( \mathcal { R } _ { 3 } ^ { ( k ) } \bigr ) ^ { 2 } + \bigl ( \mathcal { R } _ { \phi } ^ { ( k ) } \bigr ) ^ { 2 } \Bigr ] ,\tag{25}
$$

$$
\mathcal { L } _ { \mathrm { { D a t a } } } = \frac { 1 } { N _ { s } } \sum _ { j = 1 } ^ { N _ { s } } \Bigl [ \bigl ( \hat { u } _ { 1 } ^ { ( j ) } - u _ { 1 } ^ { * ( j ) } \bigr ) ^ { 2 } + \bigl ( \hat { u } _ { 3 } ^ { ( j ) } - u _ { 3 } ^ { * ( j ) } \bigr ) ^ { 2 }
$$

$$
+ ( \hat { \phi } ^ { ( j ) } - \phi ^ { \ast ( j ) } ) ^ { 2 } ] ,\tag{26}
$$

where $\mathcal { R } _ { 1 } ^ { ( k ) } , \mathcal { R } _ { 3 } ^ { ( k ) }$ , and ${ \mathcal R } _ { \phi } ^ { ( k ) }$ are the residuals of Eqs. $( 1 1 ) ‐ ( 1 3 )$ at collocation point $k ,$ evaluated with the predicted fields. The momentum residuals subtract the forcing terms $f _ { 1 }$ and $f _ { 3 } ,$ , and the Gauss residual adds the charge density q of Eqs. (21)–(23). The data weight is fixed at $\lambda = 5 0 0 0$ , a non-adaptive choice that gives the observations a dominant weight in the total loss. Sec. IV D reports how the recovered parameter depends on this choice.

## C. FE Cross-Validation Under Additive Observation Noise and Discretization Error

This subsection applies the constrained network to a transient dataset computed with the COMSOL Multiphysics finite-element solver, using the same plane-strain PZT-5H geometry, clamped-and-grounded boundaries, and material constants (Table I).

The network is again asked to reproduce the asymmetric, non-polynomial 8-lobe wavefield of Eqs. (18)–(20). Since the manufactured fields satisfy the clamped boundaries and the zero initial conditions, they are the exact solution of the forced problem, and the finite-element output is that wavefield plus discretization error. Over all exported points and times this error is $1 . 9 7 \% \left( u _ { 1 } \right)$ , 1.49% $\left( u _ { 3 } \right)$ , and 1.00% (ϕ) in relative $L ^ { 2 }$ norm.

The MMS study works entirely in the normalized $\mathcal { O } ( 1 )$ space, whereas the FEM study uses dimensional fields. The dimensionless body-force and charge-density terms $( f _ { 1 } , f _ { 3 } , q )$ derived from Eqs. (18)–(20) are converted with the scales $c _ { 4 4 } U / L ^ { 2 } = 2 . 3 0 \times 1 0 ^ { 5 } ~ \mathrm { N / m ^ { 3 } }$ for the body forces and $e _ { 3 3 } U / L ^ { 2 } = 2 . 3 3 \times 1 0 ^ { - 4 } ~ \mathrm { C } / \mathrm { m } ^ { 3 }$ for the charge density, with $U = 1$ nm and $L = 1 0$ mm, and are applied as volumetric sources in the COMSOL solid mechanics and electrostatics modules. The body forces are applied as $f _ { 1 }$ and $f _ { 3 }$ times the force scale, and the space charge density is applied as −q times the charge scale, where q is defined in Eq. (23).

The model was built in COMSOL Multiphysics 6.3 as a transient study on a 10 mm × 10 mm square, with the solid mechanics and electrostatics interfaces coupled through the piezoelectric efect and with zero initial displacement, velocity, and potential. The solution was written out every 25 ns from 0 to $5 \mu \mathrm { s }$ . The mesh and the time-dependent solver use the physics-controlled default settings of COMSOL. The mesh has 24,912 triangular elements with quadratic Lagrange shape functions, which gives 150,675 degrees of freedom.

The fields are exported along the line $x _ { 3 } = 9$ mm $( { \bar { x } } _ { 3 } =$ 0.9) at 5,000 equispaced points $( x _ { 1 } = 0 . 0 0 1$ to 9.999 mm, 0.002 mm apart) and 201 equispaced times from 0 to $5 \mu \mathrm { s }$ (normalized $\bar { T } \approx 0 . 8 7 6 )$ , which gives $1 . 0 0 5 \times 1 0 ^ { 6 }$ samples per field component. COMSOL writes the displacements in mm and the potential in volts, and both are converted to the normalized units of Sec. III D before training.

Additive white Gaussian noise at 30 dB is added independently to each exported field component, and the noisy fields $( u _ { 1 } ^ { \delta } , u _ { 3 } ^ { \delta } , \phi ^ { \delta } )$ are the targets of $\mathcal { L } _ { \mathrm { D a t a } }$ . As in the MMS test, the observations lie on the plane $\bar { x } _ { 3 } = 0 . 9$ Unlike the MMS test, which uses $5 , 0 0 0$ random space– time samples, the finite-element record is dense along this line, so the data are sparse only in the sense that they come from a single plane near the top surface. Minibatches of the exported samples are drawn during training. The analytical sources $f _ { 1 } , \ f _ { 3 }$ , and $q$ stay in the PDE residuals of Eqs. (11)–(13), so $\mathcal { L } _ { \mathrm { P D E } } = 0$ at the exact manufactured solution. Any remaining PDE residual therefore comes from the discretization error of the FE data, from the noise carried into the network weights through the data loss, and from the finite capacity of the network.

The network, the training procedure, $\lambda = 5 0 0 0$ , and the initial value $\bar { e } _ { 1 5 } ^ { ( 0 ) } = 0 . 1 0$ are the same as in the MMS study (Sec. III D).

## D. Computational Setup, Baselines, and Network Architecture

The setup shared by all of the studies is as follows. The network is a single fully connected network with 7 hidden layers of 150 neurons and three output heads $( \hat { u } _ { 1 } , \hat { u } _ { 3 } , \hat { \phi } )$ 2 which gives about 137,000 trainable weights in $\mathcal { N } _ { \theta }$ (not counting the single material parameter $\bar { e } _ { 1 5 } )$ . It uses hyperbolic tangent (Tanh) activations and Glorot uniform initialization [41], and $N _ { c } = 2 5 { , } 0 0 0$ collocation points are sampled uniformly in the interior, over $0 \leq \bar { t } \leq 1$ in the MMS runs and over $0 \leq \bar { t } \leq 0 . 8 7 5$ in the FEM runs (the 5 µs record ends at $\bar { t } \approx 0 . 8 7 6 )$ . The non-dimensional scaling is also intended to put the gradient with respect to $\bar { e } _ { 1 5 }$ on a scale comparable to that of the network weights, so that no separate learning rate is needed for $\bar { e } _ { 1 5 } \ [ 2 4 ]$ Once trained, the network gives the fields at any space– time query point with a single forward pass $( \sim \mathrm { 2 . 7 \times 1 0 ^ { 5 } }$ floating-point operations per point), independent of any mesh and without further solves.

Two soft-penalty baselines serve as the point of com parison. Both are trained on exactly the data used for the constrained network: for each seed, the same collocation points, the same 5,000 noisy observations, the same noise realization, and the same initial weights. The spatiotemporal constraints are removed from the forward pass and enforced through the penalties $\mathcal { L } _ { \mathrm { B C } }$ (squared fields on the four boundary faces) and $\mathcal { L } _ { \mathrm { I C } }$ (squared fields and squared time derivatives at $t = 0 )$ . Baseline A adapts the weights of the mechanical and electrical PDE terms, of $\mathcal { L } _ { \mathrm { B C } }$ , and of the two $\mathcal { L } _ { \mathrm { I C } }$ terms every 100 steps from the squared gradient norms of the individual loss terms, in the spirit of the NTK-based balancing of Wang et al. [32], with an exponential moving average of rate $0 . 1$ , and its data weight is kept at or above 5000. Baseline B uses no adaptation. All its penalty weights, including those of $\mathcal { L } _ { \mathrm { B C } }$ and $\mathcal { L } _ { \mathrm { I C } }$ , are fixed at 5000, the same value as the data weight. Both baselines use the same network and the same initial guess $\bar { e } _ { 1 5 } ^ { ( 0 ) } = 0 . 1 0$ as the constrained model, the same Adam schedule (15,000 steps, cosine decay from $1 0 ^ { - 3 }$ to $1 0 ^ { - 5 } )$ , no gradient clipping, and a full batch of 25,000 collocation points, 5,000 boundary points, 5,000 initial-time points, and 5,000 observations at every step. The L-BFGS stage that follows uses the same full batch.

The constrained variants use the same network and the same observations. They difer in which constraints are built into the forward pass. C1 is the full ansatz of Eqs. (15)–(17). C2 changes only the temporal factor of the potential from $t ^ { 2 }$ to $t ,$ with no extra penalty, so it isolates the efect of the temporal order on the electrostatics. C3 keeps the spatial factor only and adds soft penalties of weight 5000 on the initial value and the initial rate. C4 keeps the factor $t ^ { 2 }$ only and adds a soft boundary penalty of weight 5000. The constrained variants draw a minibatch of 5,000 collocation points at each Adam step, whereas the baselines use the full batch.

Training has two stages for every formulation. Stage 1 uses the Adam optimizer [42] (default $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } =$ 0.999) for 15,000 steps with a cosine annealing learning rate [43] that decays from $1 0 ^ { - 3 } \mathrm { t o } 1 0 ^ { - 5 }$ . In the MMS runs of the constrained network, each step uses a minibatch of 5,000 collocation points drawn from the 25,000-point pool together with all 5,000 observations, and in the FEM runs it uses 5,000 collocation points and a minibatch of 2,000 observation samples drawn from the exported record.

Stage 2 refines the estimate with L-BFGS and a strong-Wolfe line search [44], using the default PyTorch tolerances $( 1 0 ^ { - 7 }$ on the gradient and $1 0 ^ { - 9 }$ on the function change) and a history size of 100. In the MMS runs of the constrained network, including arms C1 to C4, this stage works on the first 15,000 collocation points of the pool and all 5,000 observations, and in the FEM runs on 15,000 collocation points and a fixed random subset of 5,000 observation samples.

The L-BFGS stage runs for at most 1,500 iterations (1,875 function evaluations) with a learning rate of 0.1 in the MMS study and in the baseline and ablation studies, and for at most 1,250 iterations with a learning rate of 1 in the FEM runs of Secs. IV E and IV G.

The noise sweep (Sec. IV F), including its 30 dB entry, uses 12,000 Adam steps (the cosine schedule is stretched over these 12,000 steps) with minibatches of 4,000 collocation points and 2,000 observations, followed by at most 500 L-BFGS iterations with a learning rate of 1. All computations use float32 precision. All runs were executed on NVIDIA Tesla T4 GPUs, and the full noise sweep took about 28 hours.

To measure run-to-run variability, each formulation was trained five times. In the baseline and ablation studies of Secs. IV A and IV B, the five seeds fix the initial network weights, the collocation points, the observations, and the noise, and the same seeds are used for every formulation, so diferences between formulations within a seed come only from the formulation. In the MMS and FEM studies of Secs. IV C and IV E, the seed in the script was changed by hand between runs, the runs of Sec. IV G use the fixed seeds 1 to $5 ,$ and in the noise sweep it comes from the system clock, so in every run the initial weights, the collocation points, and the noise are redrawn. Results are given as mean ± one sample standard deviation (N − 1 normalization) over the five runs, and the error columns report the mean of the individual absolute percentage errors.

## IV. RESULTS AND DISCUSSION

The results follow the order of the studies described above. The two soft-penalty baselines come first, because they show what penalty enforcement does in this problem.

## A. Soft-Penalty Baselines

The two soft-penalty baselines of Sec. III D are each trained on five seeds. Fig. 3 shows their $e _ { 1 5 }$ trajectories, and Table II collects the final numbers.

Baseline A, with adaptive weights, ends at the wrong $e _ { 1 5 }$ in four of five runs, with final errors of 54.3%, 50.8%, 63.9%, and 87.5% (Fig. 3(a)). The fifth run (seed 3) reaches 1.9%, so the failure depends on the seed and is not guaranteed. The total loss gives no warning. It drops to about $5 \times 1 0 ^ { - 2 }$ in every run, which with a data weight of 5000 means that the observations are fitted well. The fields tell a diferent story. At held-out space–time points, the relative $L ^ { 2 }$ errors are 113–179% for $u _ { 1 }$ , 67–131% for $u _ { 3 }$ , and 74–148% for $\phi ,$ and the boundary values are of by about 0.06–0.10 in dimensionless units. Even the run with a good $e _ { 1 5 }$ has field errors above 100%, so a correct parameter does not mean correct fields. In addition, the L-BFGS stage of A stops after two to four iterations in four of the five runs and leaves $e _ { 1 5 }$ unchanged.

Baseline B, with all penalties fixed at 5000, behaves diferently (Fig. 3(b)). The parameter first overshoots to between about 59 and 79 $\mathrm { C } / \mathrm { m } ^ { 2 }$ and then relaxes, and Adam alone leaves a mean error of $9 . 4 \% \pm 1 . 6 \%$ (final values between 18.3 and 19.0 $\mathrm { C } / \mathrm { m } ^ { 2 } )$ . The L-BFGS stage, which runs for 1,123 to 1,214 iterations and always reaches its cap of 1,875 function evaluations, brings this down to 0.849% ± 0.245% (range 0.58–1.24%). The heldout field errors are 0.9–2.4% and the boundary values are of order $1 0 ^ { - 3 }$ . So penalty enforcement can work in this problem when the weights are chosen well and a secondorder stage is available. It is the adaptive weighting of A that fails, and its result depends on the seed.

The upward bias of $e _ { 1 5 }$ in A is consistent with a mechanism referred to here as a “stifness trap”. During early training the Dirichlet boundaries are violated, which puts phase errors into the predicted wavefield, and a larger $e _ { 1 5 }$ compensates for these errors in the data loss on the observation plane, because it makes the medium stifer and the waves faster. Piezoelectric stifening gives the optimizer the means to do this, since the open-circuit stifened shear modulus $c _ { 4 4 } ^ { D } = c _ { 4 4 } ^ { E } + e _ { 1 5 } ^ { 2 } / \varepsilon _ { 1 1 }$ grows with $e _ { 1 5 }$ . At the mean value of A, $e _ { 1 5 } \approx 2 5 . 8 \ : \mathrm { C / m ^ { 2 } }$ , this modulus is about 1.6 times its value at $1 7 . 0 ~ \mathrm { C / m ^ { 2 } }$ , which corresponds to an efective shear wave speed about 26% higher. The large overshoots of B point in the same direction. The experiments here do not isolate this mech anism, but the comparison in the next subsection gives some evidence, because the upward overshoot of $e _ { 1 5 }$ is largest in the formulations with soft boundaries.

![](images/12e43288226f22418472aa4a350ae9e65d3da53199e040c95e6f3e98ed391dc2.jpg)

![](images/5f0ad7ccce5892db05da17afd32798472fe01a01fb547a10afec3da28f161a57.jpg)  
FIG. 3. Trajectories of the recovered $e _ { \mathrm { 1 5 } }$ for the two soft-penalty baselines on five seeds (30 dB noise, true value $1 7 . 0 ~ \mathrm { C / m ^ { 2 } } ,$ dashed line). The horizontal axis shows the Adam steps (0–15,000) followed by the L-BFGS function evaluations. (a) Baseline A with adaptive weights: four runs settle between $2 5 . 6$ and $\mathrm { 3 1 . 9 ~ C / m ^ { 2 } }$ and one reaches $1 7 . 3 ~ \mathrm { C / m ^ { 2 } }$ . (b) Baseline B with all penalties fixed at 5000: the parameter overshoots to between about 59 and 79 $\mathrm { C / m ^ { 2 } }$ , relaxes to 18.3–19.0 $\mathrm { { C / m ^ { 2 } } }$ under Adam, and ends at 17.1–17.2 $\mathrm { { C / m ^ { 2 } } }$ after L-BFGS.

## B. Controlled Comparison of Constraint Types

To separate what the spatial factor and the temporal factor each contribute, and to check whether the temporal order matters, the four constrained variants of Sec. III D are trained on the same five seeds as the baselines. Table II puts all six formulations side by side, and Fig. 4 shows the e e15 $e _ { 1 5 }$ trajectories of the constrained variants.

All four constrained variants recover $e _ { 1 5 }$ to better than $0 . 6 \%$ on average. The full formulation C1 reaches $0 . 4 5 8 \% \pm 0 . 1 7 1 \%$ , which is close to the $0 . 4 4 4 \% \pm 0 . 2 2 9 \%$ of the MMS study of Sec. IV C, whose five runs used diferent seeds. C1, C3, and C4 beat baseline B in all five seeds, and C2 does so in four of five. The fields are also more accurate. The relative $L ^ { 2 }$ error of $u _ { 3 }$ is 0.6–0.9% for the constrained variants and 1.5% for B, and the boundary values of the variants with the spatial factor are exactly zero. The convergence behavior difers even more. After Adam alone, C3 is already at $0 . 2 9 \% \pm 0 . 1 2 \%$ , whereas B is at $9 . 4 \%$ , so the spatial factor removes most of the need for a long second-order stage. The variants with the factor $t ^ { 2 }$ (C1, C2, and C4) are at 3.4–5.1% after Adam, which is still lower than B.

The full formulation is not the best of the four, which is the less obvious result. C3 and C4, which use one exact constraint and a soft penalty for the other, reach $0 . 1 6 3 \% \pm 0 . 0 5 4 \%$ and $0 . 1 4 5 \% \pm 0 . 1 3 0 \%$ , and both beat C1 in all five seeds. The trajectories in Fig. 4 suggest where the diference comes from. In $\mathrm { C 3 } , e _ { 1 5 }$ rises smoothly to the true value within about 2,500 steps. In the three variants that contain the factor $t ^ { 2 }$ (C1, C2, and C4), the trajectories show large excursions. The parameter dips to roughly $- 1 5 \mathrm { C } / \mathrm { m } ^ { 2 }$ or lower in some runs of C1 and C2 and overshoots in others before it settles. One plausible reason is that the factor $t ^ { 2 }$ makes the network output small near $t = 0$ , which makes the early gradients poorly scaled. This reading was not tested separately.

The temporal factor does what the compatibility analysis of Sec. III A predicts. At early times $( t ~ \leq ~ 0 . 0 5 )$ and with the true manufactured sources, the root-meansquare Gauss-law residual is 0.0127 with the factor $t ^ { 2 }$ on the potential (C1) and 0.0864 with the factor t (C2), a ratio of 6.8. The variant without any temporal factor (C3) gives 0.046, and the variant with $t ^ { 2 }$ only (C4) gives 0.0091. This diference hardly changes $e _ { 1 5 } .$ , though. C1 and C2 end 0.06 percentage points apart on average (0.458% against 0.515%), and C2 is better than C1 in one of the five seeds.

The boundary treatment shows up in the overshoot of $e _ { 1 5 }$ . In the three formulations with soft boundaries $\mathrm { ( A , ~ B , ~ }$ , and C4) the trajectory peaks at roughly 28 to 79 $\mathrm { C / m ^ { 2 } }$ before it relaxes. With the exact spatial factor the peak is at most about 27 $\mathrm { C / m ^ { 2 } }$ (C1 and C2) or absent (C3). This is the direction expected from the stifness mechanism above, but it is an observation and not a test of that mechanism.

Taken together, the comparison supports a simple conclusion. Exact constraints, whether spatial or temporal, give more accurate and more reliable recovery of $e _ { 1 5 }$ than penalty enforcement, and the full formulation removes the need to tune or adapt boundary and initial penalty weights, which is where baseline A breaks down.

Arm C4: Temporal factor $t ^ { 2 }$ only  
![](images/110ab6d61219b7f96f861d5a7079a84a9fa8aaa4106ab747fc85880ab5109a67.jpg)

![](images/83ca2cae84493c1a8cdb3c72848e4c2360cc8981593a92f5572aeb37ddc2a5d7.jpg)

(c)  
(d)  
![](images/4b5d3e1db52b90a84fc6508f3618eb24037ca5734deac8051d014af82a095822.jpg)

![](images/6a6651f6a2c7c187f3bf7527fb58d25236c4a7234acc92fcaa63ddf22bd58ddb.jpg)  
FIG. 4. Trajectories of the recovered $e _ { \mathrm { 1 5 } }$ for the four constrained variants (panels (a)–(d) show C1–C4, five seeds each, 30 dB noise). The dashed line is the true value 17.0 $\mathrm { { C / m ^ { 2 } } }$ . The horizontal axis shows the Adam steps followed by the L-BFGS function evaluations, and the vertical axis is cut $\mathrm { a t \_ - 1 5 }$ and 85 $\mathrm { C / m ^ { 2 } } .$ . C3, with the spatial factor only, converges smoothly. The variants that contain the factor $t ^ { 2 }$ (C1, C2, and C4) show large excursions before they settle.

TABLE II. Controlled comparison of six training formulations on identical data (MMS, 30 dB noise, five seeds). Errors are mean ± sample standard deviation $( N - 1 )$ of the absolute percentage error in $e _ { 1 5 } .$ , and “after Adam” is the error before the L-BFGS stage. The u<sub>3</sub> error is the mean relative $L ^ { 2 }$ error on 20,000 held-out space–time points. The boundary column is the mean root-mean-square value of ¯u<sub>3</sub> on the boundary faces, where the exact value is zero. The early-time residual is the root-mean-square Gauss-law residual for $t \leq 0 . 0 5$ , evaluated with the true manufactured sources, and is not computed for A and B. A and B are soft-penalty baselines, C1 uses $\mathcal { D } _ { \mathrm { b c } } t ^ { 2 }$ on all fields, C2 uses $\mathcal { D } _ { \mathrm { b c } } \mathrm { i }$ t on $\phi$ and $\dot { \mathcal { D } } _ { \mathrm { b c } } t ^ { 2 }$ on u<sub>1</sub> and $u _ { 3 } ,$ C3 uses $\mathcal { D } _ { \mathrm { b c } }$ only, and C4 uses $t ^ { 2 }$ only.
<table><tr><td>Arm Built into the network</td><td></td><td>Final error (%) After Adam (%) u3 error (%) Boundary rms Early-time residual</td><td></td><td></td><td></td><td></td></tr><tr><td>A</td><td>none (adaptive penalties)</td><td> $5 1 . 7 \pm 3 1 . 3$ </td><td> $5 1 . 7 \pm 3 1 . 3$ </td><td>85.8</td><td> $6 . 9 \times 1 0 ^ { - 2 }$ </td><td></td></tr><tr><td>B</td><td>none (fixed penalties)</td><td> $0 . 8 4 9 \pm 0 . 2 4 5$ </td><td> $9 . 3 9 \pm 1 . 6 3$ </td><td>1.50</td><td> $1 . 1 \times 1 0 ^ { - 3 }$ </td><td></td></tr><tr><td>C1</td><td> $\mathcal { D } _ { \mathrm { b c } } t ^ { 2 }$  on all fields</td><td> $0 . 4 5 8 \pm 0 . 1 7 1$ </td><td> $5 . 1 4 \pm 4 . 4 0$ </td><td>0.74</td><td> $0 \ \mathrm { ( e x a c t ) }$ </td><td> $1 . 2 7 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>C2</td><td> $\mathcal { D } _ { \mathrm { b c } } t ^ { 2 }$  on  $u _ { 1 } , u _ { 3 } ; \mathcal { D } _ { \mathrm { b c } }$  t on</td><td> $\phi$   $0 . 5 1 5 \pm 0 . 1 6 5$ </td><td> $3 . 9 0 \pm 4 . 0 1$ </td><td>0.90</td><td> $0 \ \mathrm { ( e x a c t ) }$ </td><td> $8 . 6 4 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>C3</td><td> $\mathcal { D } _ { \mathrm { b c } } ~ \mathrm { o n l y }$ </td><td> $0 . 1 6 3 \pm 0 . 0 5 4$ </td><td> $0 . 2 8 5 \pm 0 . 1 1 5$ </td><td>0.69</td><td> $0 \ \mathrm { ( e x a c t ) }$ </td><td> $4 . 6 2 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>C4</td><td> $t ^ { 2 } ~ \mathrm { o n l y }$ </td><td> $0 . 1 4 5 \pm 0 . 1 3 0$ </td><td> $3 . 4 2 \pm 0 . 6 0$ </td><td>0.61</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> $9 . 1 \times 1 0 ^ { - 3 }$ </td></tr></table>

The spatial factor gives the cleanest convergence. The temporal factor adds exact initial conditions and a small early-time electrostatic residual, at the price of a harder optimization path in these runs. The full formulation C1 is used for the finite-element cross-check and the noise sweep below, because it satisfies both the boundary and the initial conditions exactly.

Three points limit how far these numbers can be pushed. The baselines use full batches while the constrained variants use minibatches of 5,000 points, so run times are not compared here. The penalty weight of 5000 in B, C3, and C4 was set equal to the data weight and was not tuned. With five runs per formulation, diferences of a few hundredths of a percentage point should not be interpreted.

## C. Exact-Constrained Parameter Recovery

The constrained network removes the boundary and initial conditions from the loss, so the optimizer acts only on the interior residual and the data. Fig. 5 shows a representative training run on the asymmetric $[ 1 - \cos ( 2 \pi t ) ]$ manufactured wavefield with 30 dB noise.

The predicted displacement ${ { \bar { u } } _ { 3 } }$ at $\bar { t } = 0 . 5 \ : ( \mathrm { F i g . \ : 5 ( a ) } )$ reproduces the asymmetric multi-lobe structure set by the wavenumber assignment sin(4πx ) sin(2πx ). The multiplier $\mathcal { D } _ { \mathrm { b c } }$ forces all amplitudes to zero at the four edges, so the boundary condition holds even though the interior field is strongly oscillatory and non-polynomial in time. The cross-section in Fig. 5(b) shows the three independently structured fields, which is consistent with the shared network representation resolving the diferent spatial frequencies assigned to each variable.

Both $\mathcal { L } _ { \mathrm { P D E } }$ and $\mathcal { L } _ { \mathrm { D a t a } }$ decrease throughout training $\left( { \mathrm { F i g . ~ 5 ( c ) } } \right)$ The L-BFGS stage then fine-tunes the parameter. Over five independent runs the mean error is 0.444% ± 0.229% relative to 17.0 $\mathrm { C / m ^ { 2 } }$ , and arm C1 of Sec. IV B, which uses diferent seeds, gives 0.458% ± 0.171%, so the number is stable across seed sets. With the initial and boundary conditions fixed analytically, the network cannot lower the loss by introducing errors at the domain edges, so it has to satisfy the interior physics to fit the heavily weighted, noisy observations. This is consistent with the absence of the boundary-related bias seen for baseline A.

## D. Data-Weighting Sensitivity

The data weight λ is the one weight set by hand in the loss of the full formulation, so this subsection checks how the recovered parameter depends on it. The test uses the noisy asymmetric manufactured wavefield at 30 dB.

Table III lists the recovered $e _ { 1 5 }$ over a decade of $\lambda$ values. The mean error falls monotonically with λ over the tested range, from 1.497% at $\lambda = 1 0 0 0$ to 0.175% at $\lambda = 1 0 0 0 0$ , and every value stays within 1.5% of the true parameter. The value $\lambda = 5 0 0 0$ is an intermediate setting inside this range and was kept for the FEM crosscheck and the SNR sweep. A larger weight would lower the error further at this noise level, but 5000 was kept so that the other studies stay comparable.

TABLE III. Sensitivity of the recovered $e _ { 1 5 }$ to the dataweighting hyperparameter λ. Results are reported as mean ± 1σ over five independent training runs evaluated on the asymmetric MMS under 30 dB additive Gaussian noise. The $\lambda = 5 0 0 0$ row is the five-run MMS study of Sec. IV C. The true dimensional reference value is $e _ { 1 5 } ^ { \mathrm { t r u e } } = 1 7 . 0 \mathrm { C / m ^ { 2 } }$
<table><tr><td>λ Mean  $\mathrm { ( C / m ^ { 2 } ) }$ </td><td> $e _ { 1 5 }$  Mean Absolute Error (%) ± 1σ</td></tr><tr><td>1000</td><td>17.254  $1 . 4 9 7 \pm 0 . 7 5 0$ </td></tr><tr><td>2500 17.039</td><td> $0 . 9 5 2 \pm 0 . 2 1 9$ </td></tr><tr><td>5000 16.981</td><td> $0 . 4 4 4 \pm 0 . 2 2 9$ </td></tr><tr><td>10000 16.970</td><td> $0 . 1 7 5 \pm 0 . 0 4 5$ </td></tr></table>

## E. FEM Cross-Check at 30 dB Noise

Fig. 6 shows a representative run of the FEM crosscheck. The network reproduces the asymmetric 8-lobe transient displacement field (Fig. 6(a)). The data come from a finite-element computation, yet the displacement goes cleanly to zero at all four edges, as expected: the $\mathcal { D } _ { \mathrm { b c } }$ multiplier applies the exact boundary conditions, which the manufactured solution satisfies.

The cross-section in Fig. 6(b), taken at the spatial antinode $x _ { 1 } = 1 . 2 5$ mm to avoid nodal lines, shows the displacement and the potential in phase. The dotted curves are the exact manufactured fields at the same position and time. $\mathrm { A t } ~ t = 5 . 0 ~ \mu \mathrm { s }$ the wavefield is small, and the largest diference between the PINN and the exact curve is 5.5% of the peak for $u _ { 3 }$ and 11.4% of the peak for $\phi .$ The potential is therefore reproduced with a larger relative error than the displacement at this time. A single late snapshot also gives larger relative errors than the relative $L ^ { 2 }$ errors over the whole record reported below, because the amplitude is small and part of the diference comes from the finite-element data, which difer from the exact field by 1–2% in relative $L ^ { 2 }$ norm.

Both $\mathcal { L } _ { \mathrm { P D E } }$ and $\mathcal { L } _ { \mathrm { D a t a } }$ decrease steadily apart from the minibatch noise of Adam (Fig. 6(c)). The PDE loss levels of at a nonzero floor, as expected when continuous analytical source terms are reconciled with finite-element target data.

In the representative run of Fig. 6(d), $e _ { 1 5 }$ converges rapidly during the Adam phase and L-BFGS refines it. The recovered value stays accurate despite the noise that remains in the displacement field.

Fig. 7 separates the clean finite-element field, the noisy observations, and the PINN prediction along the observation line $x _ { 3 } = 9$ mm at $t = 5 . 0 ~ \mu \mathrm { s }$ . At this time the added noise has a root-mean-square value of about 10% of the peak of each field, because the 30 dB level is set by the signal power of the whole record. The PINN deviates from the clean field by 1.6–4.3% of the peak in root mean square, and by at most 9.2% for $u _ { 3 } ~ ( \mathrm { F i g . ~ 7 ( c ) } )$ . Over all 5,000 points and 201 times of the record, the relative $L ^ { 2 }$ errors against the clean finite-element data are 1.01%, 1.17%, and 0.87% for $u _ { 1 } , u _ { 3 }$ , and ϕ in this run.

## Constrained PINN Performance (MMS Data + 30 dB Noise)

(a)  
![](images/beb8c90b42762ce8bc48bc106a042e3513dbe0af544a5a2de11574b48a3f331c.jpg)  
(b)

(c)  
(d)  
![](images/3a9aab29bb6ff82ce39540826a91b26e86f2b595f9f4a90cc8d15cadabcfad1e.jpg)

![](images/57a24962ff4c6ff0cb630a16b4da73cc13b0319b9aad0533dfc229ab564d938a.jpg)

![](images/fab0deb4dbf0a5b4e91ac879b3d6a945d85ae467d795e9f7e2c9365e77f6ada6.jpg)  
FIG. 5. Parameter discovery using the topologically constrained multiphysics PINN, evaluated against an asymmetric manufactured wavefield subject to 30 dB additive Gaussian noise. (a) The predicted dimensionless displacement field $\bar { u } _ { 3 }$ at $\bar { t } = 0 . 5$ captures the asymmetric spatial structure arising from the independent wavenumber assignment sin $\left( 4 \pi x _ { 1 } \right) \sin ( 2 \pi x _ { 3 } )$ , which distributes four antinodal columns across two antinodal rows, with all amplitudes exactly zero at the domain perimeter. (b) A cross-sectional profile at $\bar { x } _ { 1 } = 0 . 1 2 5 \ ( x _ { 1 } = 1 . 2 5 \ \mathrm { m m } )$ ), where $\displaystyle \bar { u } _ { 3 }$ has an antinode in $x _ { 1 } ,$ , shows the reconstruction of the coupled electromechanical response across the independent spatial frequencies assigned to each field. The dotted curves are the exact manufactured fields. (c) The optimization dynamics show both the internal physics loss $( \mathcal { L } _ { \mathrm { P D E } } )$ and the noisy observation data loss $\left( \mathcal { L } _ { \mathrm { D a t a } } \right)$ minimizing continuously through the sequential Adam and L-BFGS stages. (d) The inverse parameter trajectory for the shear coeficient $e _ { \mathrm { 1 5 } }$ . The network converges to $1 6 . 9 7 5 ~ \mathrm { C / m ^ { 2 } }$ , representing a 0.15% error relative to the true target of $1 7 . 0 ~ \mathrm { C / m ^ { 2 } }$ . The five-run mean error is $0 . 4 4 4 \% \pm 0 . 2 2 9 \%$

Table IV compares the MMS and FEM runs of the full formulation. The first FEM row is the 30 dB entry of the noise sweep (Sec. IV F), so it comes from the shorter sweep schedule of 12,000 Adam steps, whereas the representative run in Fig. 6 uses the 15,000-step schedule of Sec. III D. The MMS study stays below 0.5%. The error rises to $1 . 1 8 9 \% \pm 0 . 6 5 1 \%$ for the finite-element data at the same noise level. The finite-element fields difer from the exact manufactured fields by 1.0–2.0% in relative $L ^ { 2 }$ norm, of the same order as the 3.2% noise amplitude at 30 dB, and the two tests also difer in observation set and time window, so the increase reflects all of these factors together. In the MMS and FEM studies, each run redraws the network initialization, the collocation points, and the noise, so the standard deviation is the combined run-to-run variability.

The FEM cross-check used a single noise level, and Sec. IV F varies this noise level.

TABLE IV. Inverse $e _ { \mathrm { 1 5 } }$ identification with the full constrained formulation on the manufactured (MMS) and finite-element (FEM) data. Results are reported as mean ± 1σ over $N = 5$ independent training runs evaluated at 30 dB additive Gaussian noise. The error column reports the mean of individual absolute percentage errors $\langle \bar { | } e _ { 1 5 , i } - e _ { 1 5 } ^ { \mathrm { t r u e } } | / e _ { 1 5 } ^ { \mathrm { t r u e } } \rangle$ across the $N \ = \ 5$ runs. The true dimensional reference value is $e _ { 1 5 } ^ { \mathrm { t r u e } } = 1 7 . 0 \ \mathrm { C / m ^ { 2 } }$
<table><tr><td>Dataset Mean  $\mathrm { ( C / m ^ { 2 } ) }$ </td><td> $e _ { 1 5 }$  Mean Error (%) ±1σ</td></tr><tr><td>MMS 16.981</td><td> $0 . 4 4 4 \pm 0 . 2 2 9$ </td></tr><tr><td>FEM, sweep schedule 16.810</td><td> $1 . 1 8 9 \pm 0 . 6 5 1$ </td></tr><tr><td>FEM, long schedule 16.837</td><td> $0 . 9 5 9 \pm 0 . 3 0 9$ </td></tr></table>

## F. Noise Sensitivity Across the Full SNR Spectrum

To see how the identification depends on noise, the same constrained network, training pipeline, and finiteelement wavefield are applied at nine SNR levels from 40 dB down to 0 dB. Before each training run a fresh,

## Constrained PINN Performance (FEM Data + 30 dB Noise)

(a)  
![](images/a735f3049e30f0977659dc55ab925134d43e94cf945bc57339871030d4a57b0c.jpg)  
(c)

(b)  
(d)  
![](images/1b075ceaaaf32b2e2c77120faa826737e261cd6976bcb7cdf04de7828b5b64a4.jpg)

![](images/0dc7ab22477703d6c261de2a4e0f8e00cc4a83dcfaa31ea31d978a60a0af7ffd.jpg)

![](images/ee7c9cf7ce0b0031a0d5fe4343c59a8d62dc1ac5e1a011dda24643e6fea737fa.jpg)  
FIG. 6. Parameter discovery using the identical constrained multiphysics PINN applied to $\mathrm { C O M S O L }$ FEM transient data of the same forced problem with 30 dB additive Gaussian noise. (a) The predicted displacement field $u _ { 3 }$ at $t = 5 . 0 ~ \mu \mathrm { s }$ (normalized $\bar { t } \approx 0 . 8 7 6 )$ reproduces the asymmetric 8-lobe standing wave pattern across the domain and vanishes at the clamped boundaries, as enforced by $\mathcal { D } _ { \mathrm { b c } } \cdot t ^ { 2 }$ . (b) A cross-sectional profile evaluated explicitly at $x _ { 1 } = 1 . 2 5$ mm (the primary spatial antinode of the $u _ { 3 }$ wavefield) shows co-phase behavior of the displacement and the potential. The dotted curves are the exact manufactured fields. The dual axes show the physical amplitudes $( u _ { 3 }$ in pm, ϕ in V). (c) Both $\mathcal { L } _ { \mathrm { P D E } }$ and $\mathcal { L } _ { \mathrm { D a t a } }$ decrease through the Adam phase, apart from minibatch noise, and stabilize under L-BFGS. (d) The inverse parameter trajectory for the shear coeficient $e _ { 1 5 }$ . The network converges from an initial estimate of 2.33 $\mathrm { C / m ^ { 2 } }$ to $1 6 . 8 7 3 ~ \mathrm { C / m ^ { 2 } }$ , representing a 0.75% error relative to the true target of $1 7 . 0 ~ \mathrm { C / m ^ { 2 } }$

independent noise realization is added to all three field components. Table V lists the recovered $e _ { 1 5 }$ and its absolute error at each level, and Fig. 8 shows the error curve on a logarithmic scale.

With $N = 5$ runs per level, the spread shows how much the result varies from run to run at each noise level. Since the initialization, the collocation points, and the noise are all redrawn in every run, the spread is the combined variability. As Fig. 8 and Table V show, the mean error does not rise monotonically between 40 dB and 10 dB and stays in overlapping bands of about 0.6% to 1.2%. This plateau is consistent with an accuracy floor set by the 1.0–2.0% discretization error of the finite-element data. This explanation was not tested separately.

At lower SNR the noise energy approaches the signal energy and the mean error drifts upward, to 2.71% at 0 dB. The spread grows markedly too: at 5 dB the standard deviation (1.768%) exceeds the mean (1.383%), and at 0 dB it is 3.594%. The run errors are therefore heavytailed, and the means should not be read as bounds on individual runs. Even so, the mean error stays at the percent level down to equal signal and noise power.

TABLE V. Statistical noise sensitivity of the constrained PINN inverse $^ { e _ { 1 5 } }$ discovery across nine SNR levels. Results are reported as mean $\pm 1 \sigma$ over $N = 5$ independent training runs per noise level, each evaluating a completely reinitialized network against a freshly sampled Gaussian noise realization. The error column reports the mean of individual absolute percentage errors $\langle | e _ { 1 5 , i } - e _ { 1 5 } ^ { \mathrm { t r u e } } | / e _ { 1 5 } ^ { \mathrm { t r u e } } \rangle$ across the $N = 5$ runs. The true reference value is $e _ { 1 5 } ^ { \mathrm { t r u \dot { e } } } = 1 \dot { 7 } . 0 \ \mathrm { C / m ^ { 2 } }$
<table><tr><td colspan="2">SNR (dB) Mean  $e _ { \mathrm { 1 5 } }$   $\mathrm { ( C / m ^ { 2 } ) }$  Error (%) ± 1σ</td></tr><tr><td>40 16.973</td><td> $0 . 6 4 6 \pm 0 . 2 8 0$ </td></tr><tr><td>35 16.838</td><td> $1 . 2 4 9 \pm 0 . 6 1 5$ </td></tr><tr><td>30 16.810</td><td> $1 . 1 8 9 \pm 0 . 6 5 1$ </td></tr><tr><td>25 16.952</td><td> $0 . 7 7 9 \pm 0 . 5 3 2$ </td></tr><tr><td>20 16.861</td><td> $0 . 9 5 4 \pm 0 . 3 9 9$ </td></tr><tr><td>15 16.856</td><td> $0 . 8 4 7 \pm 0 . 5 0 0$ </td></tr><tr><td>10 17.021</td><td> $1 . 0 1 9 \pm 0 . 8 7 3$ </td></tr><tr><td>5 16.870</td><td> $1 . 3 8 3 \pm 1 . 7 6 8$ </td></tr><tr><td>0 17.237</td><td> $2 . 7 0 5 \pm 3 . 5 9 4$ </td></tr></table>

![](images/8e786e844ffe63e4fa00847f8491069fc606520d1d46de3de7f840cb927d1b9f.jpg)

![](images/68c09c68f4e8b58612580b849a0f288a47838151bf5488e72bf9ca4f311c4297.jpg)

![](images/6c4908a907512e27f06dfd6778c74478c49a9be37042b1b1390896aeb31c5b56.jpg)  
FIG. 7. Fields along the observation line $x _ { 3 } = 9$ mm at $t = 5 . 0 ~ \mu \mathrm { s }$ for the representative finite-element run. (a), (b) Clean FEM field, every tenth noisy observation, and the PINN prediction for $u _ { 3 }$ and ϕ. (c) Pointwise diference between the PINN and the clean FEM field, as a percentage of the peak value of each field.

$$
e _ { 1 5 }
$$

![](images/3d72cbf01b9cee951246a045109da1b2f6e3030f808381b1770cc937873c16e3.jpg)  
FIG. 8. Statistical noise sensitivity of the constrained PINN inverse $e _ { \mathrm { 1 5 } }$ discovery across nine SNR levels from 40 dB down to 0 dB. The y-axis displays the absolute percentage error in the recovered $e _ { \mathrm { 1 5 } }$ on a logarithmic scale, and the x-axis is inverted to reflect increasing noise. The solid line represents the mean absolute error evaluated over $N = 5$ training runs per noise level, while the shaded region denotes the ±1σ band, which is truncated at the lower axis limit where the mean minus one standard deviation is not positive.

## G. Displacement-Only Observations and Field Accuracy

The electric potential cannot be measured inside a real device (Sec. V), so the full formulation was also trained on displacement observations only. The data term $\mathcal { L } _ { \mathrm { D a t a } }$ then contains only $u _ { 1 }$ and $u _ { 3 }$ . Everything else is unchanged: the PDE residuals of all three equations, the exact constraints, λ = 5000, and the schedules. The runs use seeds 1 to 5 at 30 dB on the manufactured data and on the finite-element data. On the manufactured data the seeds, collocation points, and noise are those of arm C1 in Sec. IV B, so C1 is the reference with all three fields observed. On the finite-element data, the reference is a new set of runs with all three fields and the same seeds, which uses the schedule of the representative run (15,000

Adam steps and at most 1,250 L-BFGS iterations). Table VI lists the results and Fig. 9 shows the trajectories.

Without the potential, $e _ { 1 5 }$ is still recovered. On the finite-element data the mean error is $0 . 4 5 2 \% \pm 0 . 7 2 7 \%$ against $0 . 9 5 9 \% \pm 0 . 3 0 9 \%$ with all fields. The spread is wider in the displacement-only case because of seed 3 (1.75%), while the other four seeds give 0.05–0.21%. With five seeds the two means should not be ranked. On the manufactured data the displacement-only error is $1 . 6 4 \% \pm 0 . 3 0 \%$ , larger than the $0 . 4 5 8 \% \pm 0 . 1 7 1 \%$ of C1, and larger in each of the five seeds (1.29–2.01% against 0.26–0.69%). The potential data therefore help in the manufactured test, and their efect on the finite-element data is not clear at this sample size.

The first stage behaves diferently without the potential. After Adam alone the error is $1 9 . 0 \% \pm 7 . 1 \%$ on the finite-element data and $3 1 . 5 \% \pm 1 0 . 3 \%$ on the manufactured data, against 5.7% and 5.1% with all fields, and in most runs the trajectory is still below the true value when Adam ends (Fig. 9(b), (c)). The L-BFGS stage then completes the recovery, so the displacementonly identification depends on the second-order stage, as baseline B does. The held-out field errors change little (Table VI). The potential, which is no longer observed, is still reproduced to 0.9% on the finite-element data and 1.4% on the manufactured data, so the equations carry it from the displacement.

For the field accuracy with all three fields observed on the finite-element data, the relative $L ^ { 2 }$ errors against the exact wavefield at held-out points are $0 . 9 6 \% , 0 . 8 8 \%$ and 0.64% for $u _ { 1 } , \ u _ { 3 } ,$ , and $\phi .$ These are smaller than the diferences between the finite-element solution and the exact field (1.97%, 1.49%, and 1.00%), which is consistent with the PDE residuals being evaluated with the exact forcing, so that the network is pulled toward the continuous solution and not toward the discretization error. This was not tested separately. Against the noisefree finite-element export on the observation line, the errors are 1.07%, 1.21%, and 0.88%. Both sets of numbers are means over seeds 1 to 5. Their mean $e _ { 1 5 }$ error of $0 . 9 5 9 \% \pm 0 . 3 0 9 \%$ is the second FEM row of Table ${ \mathrm { I V } } ,$ and it difers from the $1 . 1 8 9 \% \pm 0 . 6 5 1 \%$ of the first row because the schedule and the seeds difer.

## V. LIMITATIONS AND OUTLOOK

The study rests on several choices that limit what it shows. This section states each one and the next step it suggests.

## A. Boundaries and Excitation

The formulation handles only homogeneous Dirichlet boundaries, a rest initial state, and a volumetric source. Real excitations such as electrode voltage pulses, surface tractions, or laser-generated loading act through the boundaries. The next step is to extend the exact construction to non-homogeneous boundary data and, for free surfaces, to Neumann conditions.

## B. Observations and Data

In the main runs the observations include the interior electric potential, which cannot be measured inside a real device. The displacement-only test of Sec. IV G removes it and $e _ { 1 5 }$ is still recovered, but it uses the same simulated data and noise model. Because the perimeter potential is fixed to zero here, boundary electrode voltages cannot stand in for it, so a measurement model built on quantities that can be recorded is needed. The finite-element data also solve the same forced problem as the manufactured solution, so that test checks the solver and not a diferent excitation, and no experimental data are used.

## C. Scope of the Comparison

The comparison of formulations uses one noise level, one manufactured field, and five seeds per formulation, and the soft penalty weights were set to 5000 without further tuning. Repeating it with more seeds, other fields, and tuned weights would show how general the ranking is. The finite-element cross-check and the noise sweep use only the full formulation, and running them for the spatial-only variant is the obvious next test.

## D. Material Constants and Geometry

The constants $c _ { 4 4 } , e _ { 3 1 } , e _ { 3 3 }$ , and the dielectric constants are treated as known, so a sensitivity study of how errors in these constants carry over to $e _ { 1 5 }$ is also needed. The test geometry is a millimeter-scale block of bulk PZT-5H, and micrometer-scale PMUT stacks and traction-free SAW substrates still have to be tried.

## VI. CONCLUSION

This paper presents a constrained multiphysics PINN for identifying the shear piezoelectric coeficient $e _ { 1 5 }$ from sparse, noisy wavefield observations, and compares it with penalty-based alternatives on identical data. A soft-penalty baseline with adaptive loss weights failed in four of five runs $( 5 1 . 7 \% \pm 3 1 . 3 \%$ mean error) and returned fields with relative $L ^ { 2 }$ errors above 65% at heldout points. With fixed penalties, the baseline reached $0 . 8 4 9 \% \pm 0 . 2 4 5 \%$ , but only after a long second-order stage, since the first-order stage alone left 9.4% error. The constrained variants reached mean errors between 0.15% and 0.51% with held-out $u _ { 3 }$ errors below 1%.

![](images/93970f39e00c1a17c9f2419b39b8443cd6f4a64afbc4ec0326cab57c9df801e8.jpg)

![](images/d65b672f3260c8032ff660b3f9836c6a6878418e33fc98cd995a680226b2cdae.jpg)

![](images/f7633a8bb37b65e7e9541718304355aa0df62d7960d6089460203b92a9bead17.jpg)  
FIG. 9. Trajectories of the recovered $e _ { 1 5 }$ for five seeds at 30 dB noise. (a) FEM data with all three fields observed. (b) FEM data with displacement observations only. (c) MMS data with displacement observations only. The horizontal axis shows the Adam steps followed by the L-BFGS function evaluations, and the Adam stage ends at step 15,000. The dashed line is the true value of $\mathrm { 1 \bar { 7 } . 0 ~ C / m ^ { 2 } }$ , and the vertical axis is cut at −15 and 85 $\mathrm { { C / m ^ { 2 } } }$

TABLE VI. Recovery of $e _ { \mathrm { 1 5 } }$ with and without the potential in the data (30 dB, seeds 1 to 5). Errors are mean ± sample standard deviation $( N - 1 )$ of the absolute percentage error in $e _ { \mathrm { 1 5 } }$ , and “after $\mathrm { A d a m } ^ { \prime \prime }$ is the error before the L-BFGS stage. The last three columns are the mean relative $\dot { L } ^ { 2 }$ errors of the fields (%) against the exact wavefield at 20,000 held-out space–time points. The MMS rows with all fields are arm C1 of Table II. The FEM rows use the schedule of the representative run.
<table><tr><td colspan="4">Data Observed fields Final error (%) After Adam (%) u1</td></tr><tr><td></td><td> $0 . 4 5 8 \pm 0 . 1 7 1$ </td><td> $5 . 1 4 \pm 4 . 4 0$ </td><td></td><td>U3  $\phi$  0.99 0.74 0.71</td></tr><tr><td>MMS  $u _ { 1 } , u _ { 3 } , \phi$  MMS</td><td> $1 . 6 4 \pm 0 . 3 0$ </td><td> $3 1 . 5 \pm 1 0 . 3$ </td><td></td><td>1.57 1.15 1.43</td></tr><tr><td> $u _ { 1 } , u _ { 3 }$  FEM  $u _ { 1 } , u _ { 3 } , \phi$ </td><td> $0 . 9 5 9 \pm 0 . 3 0 9$ </td><td> $5 . 6 8 \pm 3 . 2 0$ </td><td></td><td>0.960.880.64</td></tr><tr><td>FEM  $u _ { 1 } , u _ { 3 }$ </td><td> $0 . 4 5 2 \pm 0 . 7 2 7$ </td><td> $1 9 . 0 \pm 7 . 1$ </td><td></td><td>1.00 0.88 0.89</td></tr></table>

The constrained formulation has three parts. First, exact spatial Dirichlet enforcement through boundaryconforming distance functions [35, 36], together with hard enforcement of initial conditions as shown for elastodynamics [25] and in the author’s earlier one-dimensional electro-elastodynamic work [37], is extended to a twodimensional coupled piezoelectric system. Second, a second-order temporal multiplier $( t ^ { 2 } )$ places the displacement and the quasi-static potential at the same asymptotic order near $t = 0$ and sets their initial rates of change to zero. Third, a non-dimensional scaling brings the PDE coeficients to within about one decade of $\mathcal { O } ( 1 )$ . The controlled comparison adds two findings about these parts. The spatial factor alone gives the cleanest convergence, and the temporal factor lowers the early-time electrostatic residual by a factor of about seven relative to a linear factor without noticeably changing the accuracy of $e _ { 1 5 } .$

On a PZT-5H plane-strain system with a spatially asymmetric manufactured wavefield and 30 dB Gaussian noise, the full formulation recovers $e _ { \mathrm { 1 5 } }$ from 5,000 space– time observations on a subsurface plane with a mean error of $0 . 4 4 4 \% \pm 0 . 2 2 9 \%$ , and arm C1 of the controlled comparison, with diferent seeds, gives $0 . 4 5 8 \% \pm 0 . 1 7 1 \%$ On a finite-element solution of the same forced problem at the same noise level, the mean error is $1 . 1 9 \% \pm 0 . 6 5 \%$

When only the displacement is observed, the mean error is $0 . 4 5 \% \pm 0 . 7 3 \%$ on the finite-element data and $1 . 6 4 \% \pm 0 . 3 0 \%$ on the manufactured data, and the recovery then depends on the L-BFGS stage. A noise sweep with $N = 5$ runs per level keeps the mean error at the percent level from 40 dB down to 0 dB, reaching 2.71% at 0 dB, where the run-to-run spread is largest.

Within the limits of Sec. V, the results show that exact constraints combined with a consistent nondimensionalization identify a single piezoelectric coefi cient from sparse, noisy field data more accurately and more reliably than penalty enforcement does, and to about one percent from 40 dB down to 10 dB. The full formulation also has no boundary or initial penalty weights to tune, which is where the adaptive baseline failed. Closing the gaps listed in Sec. V, in particular boundary-driven excitation, measurable observations, and device-scale geometries, is the path toward in-situ characterization of piezoelectric transducers.

## DATA AVAILABILITY

The Python/PyTorch code and the data needed to reproduce the results of this work are available in the Exact-Constrained-Piezo-PINN repository on

GitHub at https://github.com/suhasbharadwajx/Exact-Constrained-Piezo-PINN. The repository has one folder for each study: MMS-framework for the manufactured solution run, FEM-framework for the finite-element run, SNR-sweep for the noise sweep, and Baseline-study and Ablation-study for the soft-penalty baselines and the constrained variants of Secs. IV A and IV B, and Extended-validation for the displacement-only and field-accuracy runs of Sec. IV G.

[1] F. Tasn´adi, B. Alling, C. H¨oglund, G. Wingqvist, J. Birch, L. Hultman, and I. A. Abrikosov, Origin of the anomalous piezoelectric response in wurtzite $\mathrm { S c } _ { x } \mathrm { A l } _ { 1 - x } \mathrm { N }$ alloys, Phys. Rev. Lett. 104, 137601 (2010).

[2] B. A. Auld, Acoustic Fields and Waves in Solids (John Wiley & Sons, New York, 1973) 2 volumes.

[3] IEEE, IEEE standard on piezoelectricity, ANSI/IEEE Std. 176-1987 10.1109/IEEESTD.1988.79638 (1988).

[4] H. F. Tiersten, Linear Piezoelectric Plate Vibrations (Plenum Press, New York, 1969).

[5] L. Claes, K. Koch, O. Friesen, and L. Meihost, Machine learning-supported inverse measurement procedure for broadband, temperature dependent piezoelectric material parameters, Acta Acustica 9, 65 (2025).

[6] P. Muralt, Recent progress in materials issues for piezoelectric mems, Journal of the American Ceramic Society 91, 1385 (2008).

[7] P. Delsing, A. N. Cleland, M. J. A. Schuetz, J. Kn¨orzer, G. Giedke, J. I. Cirac, K. Srinivasan, M. Wu, K. C. Balram, C. B¨auerle, T. Meunier, C. J. B. Ford, P. V. Santos, E. Cerda-M´endez, H. Wang, H. J. Krenner, E. D. S. Nysten, M. Weiß, G. R. Nash, L. Thevenard, C. Gourdon, P. Rovillain, M. Marangolo, J.-Y. Duquesne, G. Fischerauer, W. Ruile, A. Reiner, B. Paschke, D. Denysenko, D. Volkmer, A. Wixforth, H. Bruus, M. Wiklund, J. Reboud, J. M. Cooper, Y. Fu, M. S. Brugger, F. Rehfeldt, and C. Westerhausen, The 2019 surface acoustic waves roadmap, Journal of Physics D: Applied Physics 52, 353001 (2019).

[8] B. Glisic and D. Inaudi, Fibre optic methods for structural health monitoring (John Wiley & Sons, 2007).

[9] K. S. C. Kuang, R. Kenny, M. P. Whelan, W. J. Cantwell, and P. R. Chalker, Embedded fibre bragg grating sensors in advanced composite materials, Composites Science and Technology 61, 1379 (2001).

[10] A. L. Kholkin, C. W¨utchrich, D. V. Taylor, and N. Setter, Interferometric measurements of electric field-induced displacements in piezoelectric thin films, Review of Scientific Instruments 67, 1935 (1996).

[11] E. Barsoukov and J. R. Macdonald, Impedance Spectroscopy: Theory, Experiment, and Applications, 3rd ed.

## ACKNOWLEDGEMENTS

The author humbly thanks Prerana Ramkumar for the original idea that led to this work. During the preparation of this work, the author used Claude (Anthropic) to support code development and formatting (including the scripts released on GitHub), figure preparation, numerical cross-checks, and language editing. The author reviewed and verified all outputs, ran the experiments, and takes full responsibility for the content of the manuscript.

## CONFLICT OF INTEREST STATEMENT

The author declares no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

(John Wiley & Sons, 2018).

[12] G. Carlotti, D. Fioretto, L. Palmieri, G. Socino, L. Verdini, and E. Verona, Brillouin scattering by surface acoustic modes for elastic characterization of ZnO films, IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control 38, 56 (1991).

[13] C. B. Scruby and L. E. Drain, Laser Ultrasonics: Techniques and Applications (Adam Hilger, Bristol, 1990).

[14] D. Royer and E. Dieulesaint, Elastic Waves in Solids I: Free and Guided Propagation (Springer Science & Business Media, Berlin, 1999).

[15] H. Allik and T. J. R. Hughes, Finite element method for piezoelectric vibration, International Journal for Numerical Methods in Engineering 2, 151 (1970).

[16] J. Virieux and S. Operto, An overview of full-waveform inversion in exploration geophysics, Geophysics 74, WCC1 (2009).

[17] A. Tarantola, Inverse Problem Theory and Methods for Model Parameter Estimation (SIAM, Philadelphia, 2005).

[18] S. Markidis, The old and the new: Can physics-informed deep-learning replace traditional linear solvers?, Frontiers in Big Data 4, 669097 (2021).

[19] M. Raissi, P. Perdikaris, and G. E. Karniadakis, Physicsinformed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations, Journal of Computational Physics 378, 686 (2019).

[20] G. E. Karniadakis, I. G. Kevrekidis, L. Lu, P. Perdikaris, S. Wang, and L. Yang, Physics-informed machine learn ing, Nature Reviews Physics 3, 422 (2021).

[21] S. Cuomo, V. S. Di Cola, F. Giampaolo, G. Rozza, M. Raissi, and F. Piccialli, Scientific machine learning through physics-informed neural networks: Where we are and what’s next, Journal of Scientific Computing 92, 88 (2022).

[22] A. G. Baydin, B. A. Pearlmutter, A. A. Radul, and J. M. Siskind, Automatic diferentiation in machine learning: a survey, Journal of Machine Learning Research 18, 1 (2018).

[23] S. Cai, Z. Mao, Z. Wang, M. Yin, and G. E. Karniadakis, Physics-informed neural networks (PINNs) for fluid mechanics: a review, Acta Mechanica Sinica 37, 1727 (2021).

[24] E. Haghighat, M. Raissi, A. Moure, H. Gomez, and R. Juanes, A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics, Computer Methods in Applied Mechanics and Engineering 379, 113741 (2021).

[25] C. Rao, H. Sun, and Y. Liu, Physics-informed deep learning for computational elastodynamics without labeled data, Journal of Engineering Mechanics 147, 04021043 (2021).

[26] K. Shukla, P. C. Di Leoni, J. Blackshire, D. Sparkman, and G. E. Karniadakis, Physics-informed neural network for ultrasound nondestructive quantification of surface breaking cracks, Journal of Nondestructive Evaluation 39, 61 (2020).

[27] N. Tandis and E. Tandis, A physics-informed machine learning approach to piezoelectric plate modelling, Engineering Applications of Artificial Intelligence 160, 111847 (2025).

[28] A. D. Mouratidou, M.-S. Daraki, G. A. Drosopoulos, G. Foutsitzi, W. Larbi, J.-F. De¨u, R. Ohayon, and G. E. Stavroulakis, Physics-informed neural networks for shunted piezoelectric systems, Acta Mechanica 237, 2745 (2026).

[29] L. Lu, X. Meng, Z. Mao, and G. E. Karniadakis, Deep-XDE: A deep learning library for solving diferential equations, SIAM Review 63, 208 (2021).

[30] S. Wang, Y. Teng, and P. Perdikaris, Understanding and mitigating gradient flow pathologies in physics-informed neural networks, SIAM Journal on Scientific Computing 43, A3055 (2021).

[31] A. Krishnapriyan, A. Gholami, S. Zhe, R. Kirby, and M. W. Mahoney, Characterizing possible failure modes in physics-informed neural networks, Advances in Neural Information Processing Systems 34, 26548 (2021).

[32] S. Wang, X. Yu, and P. Perdikaris, When and why PINNs fail to train: A neural tangent kernel perspective, Journal of Computational Physics 449, 110768 (2022).

[33] L. D. McClenny and U. M. Braga-Neto, Self-adaptive physics-informed neural networks using a soft attention

mechanism (2021), arXiv:2009.04544 [cs.LG].

[34] R. Bischof and M. A. Kraus, Multi-objective loss balancing for physics-informed deep learning, Computer Methods in Applied Mechanics and Engineering 439, 117914 (2025).

[35] I. E. Lagaris, A. Likas, and D. I. Fotiadis, Artificial neural networks for solving ordinary and partial diferential equations, IEEE Transactions on Neural Networks 9, 987 (1998).

[36] N. Sukumar and A. Srivastava, Exact imposition of boundary conditions with distance functions in physicsinformed deep neural networks, Computer Methods in Applied Mechanics and Engineering 389, 114333 (2022).

[37] S. S. Bharadwaj, Thermodynamically admissible neural solvers for stif electro-elastodynamics via exact geometric constraints, Journal of Applied Physics 140, 064502 (2026).

[38] R. D. Mindlin, High frequency vibrations of piezoelectric crystal plates, International Journal of Solids and Structures 8, 895 (1972).

[39] P. J. Roache, Verification of codes and calculations, AIAA Journal 36, 696 (1998).

[40] K. Salari and P. Knupp, Code Verification by the Method of Manufactured Solutions, Tech. Rep. SAND2000-1444 (Sandia National Laboratories, Albuquerque, NM and Livermore, CA, 2000).

[41] X. Glorot and Y. Bengio, Understanding the dificulty of training deep feedforward neural networks, in Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics (AISTATS), JMLR Workshop and Conference Proceedings, Vol. 9 (2010) pp. 249–256.

[42] D. P. Kingma and J. Ba, Adam: A method for stochastic optimization, in 3rd International Conference on Learning Representations (ICLR) (2015) arXiv:1412.6980.

[43] I. Loshchilov and F. Hutter, SGDR: Stochastic gradient descent with warm restarts, in 5th International Conference on Learning Representations (ICLR) (2017) arXiv:1608.03983.

[44] J. Nocedal and S. J. Wright, Numerical Optimization, 2nd ed. (Springer, New York, 2006).