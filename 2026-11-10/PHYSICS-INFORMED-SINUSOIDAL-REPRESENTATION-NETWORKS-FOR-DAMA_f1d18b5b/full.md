# PHYSICS-INFORMED SINUSOIDAL REPRESENTATION NETWORKS FOR DAMAGE IDENTIFICATION IN ULTRASONIC GUIDED WAVE STRUCTURAL HEALTH MONITORING

Lukas Jilke<sup>∗</sup>

Caspar Wasle<sup>†</sup>

Norbert Hosters<sup>‡</sup>

Florian Raddatz<sup>†</sup>

Marek Behr<sup>‡</sup>

Gerko Wende<sup>†</sup>

## ABSTRACT

Ultrasonic guided wave (UGW) inspection has become an effective method in structural health monitoring (SHM). While machine learning has enhanced damage detection capabilities, purely data-driven models remain constrained by their reliance on extensive labeled datasets and limited physical interpretability. Standard multi-layer perceptrons (MLPs) further suffer from spectral bias (F-Principle) in high-frequency regimes, prioritizing low-frequency components while failing to resolve the high-frequency oscillations critical for UGW reconstruction. Physics-Informed Sinusoidal Representation Networks (PI-SIRENs) address these limitations by embedding governing partial differential equations directly into the training process. Building upon SIRENs, which utilize periodic activation functions to inherently capture high-frequency spectral content, the proposed framework is designed to operate effectively in sparse ultrasonic measurement regimes. This work introduces a frequency-domain PI-SIREN framework specifically tailored for the Mindlin–Reissner plate theory, embedding the coupled elastodynamic equations to reconstruct ultrasonic wavefields without temporal discretization. The model parameters are optimized using a hybrid strategy comprising three sequential phases: first-order Adam optimization, residual-adaptive refinement (RAR) of the collocation set and quasi-Newton L-BFGS. Numerical and experimental evaluations confirm that the trained model serves as a reliable physics-consistent baseline derived from sparse data, functioning as a computationally efficient surrogate for damage identification in SHM and non-destructive testing (NDT) applications. Integrating the trained surrogate into a proposed damage identification pipeline detects and localizes a milled notch from sparse experimental measurements with a localization error of 1 % of the plate diagonal. In contrast, purely data-driven benchmarks and MLPs utilizing hyperbolic tangent activation functions, with and without Fourier feature input embedding, are unable to identify damage due to higher reconstruction errors.

Keywords Physics-Informed Neural Networks · Sinusoidal Representation Networks · Guided Waves · Structural Health Monitoring · Non-Destructive Testing

## 1 Introduction

Ultrasonic guided wave (UGW) inspection has emerged as a central technique in structural health monitoring (SHM) applications, particularly for thin-walled aerospace structures, where the early detection of material degradation is essential [1]. UGWs are elastic waves propagating within bounded media, characterized by low attenuation over long distances and high sensitivity to localized perturbations such as cracks, delaminations or corrosion [2, 3]. The increasing complexity of structural systems and the rising costs of maintenance, repair and overhaul (MRO) activities require the development of rapid, reliable and quantitative damage identification methods [4]. Conventional physics-based simulations, such as the finite element method (FEM) and boundary element method (BEM), provide high-fidelity wavefield simulations but remain computationally expensive for real-time monitoring or inverse analyses [5]. In contrast, purely data-driven machine learning approaches provide rapid training but require extensive labeled datasets and often lack physical interpretability, limiting their robustness in sparse-measurement regimes [6, 7, 8]. Hybrid modeling frameworks that embed physical laws into data driven architectures have therefore gained traction, providing a balance between computational efficiency and physical consistency. Physics-informed neural networks (PINNs) have emerged as hybrid frameworks that incorporate physical knowledge during the training phase [9] By enforcing partial differential equations as constraints within the loss function, PINNs provide essential regular ization that enables reliable predictions in regimes with limited observational data [10]. In the context of SHM and ultrasonic non-destructive testing (NDT) applications, several studies have applied PINNs to damage inference in various tasks. Al-Adly et al. [11] demonstrate the applica tion of a PINN model for quasi-static plate bending under Kirchhoff–Love (KL) assumptions, focusing on state estimation and virtual sensing. The authors show that this approach can accurately predict unmeasured structural deflections and internal bending moments even under variable loading and noisy data conditions. Since this approach operates strictly within a spatial and quasi-static regime, it does not account for temporal dynamics, making it inapplicable to real-time tracking, dynamic anomaly detection or ultrasonic wave-based damage identification. Zhou and Xu [12] introduce a baseline-free damage identification method for plate structures designed to isolate localized defects using UGW without requiring experimental historical baselines. Their approach applies multi-layer perceptron (MLP)-based PINNs governed by the KL plate theory to UGW data, constructing a pseudo-pristine model for damage identification based on a two-dimensional Teager energy operator metric. However, the framework remains constrained to lowerfrequency excitations and leaves boundary conditions explicitly unenforced, which requires higher-density spatial measurement data and computationally expensive model retraining for every encountered measurement. Song et al. [13] pursue a different baseline-free approach, training a physics-guided multiscale network on wavelet coefficients of time-domain Lamb wave signals to eliminate boundary reflections and extract scattered-wave arriva times without explicit wavefield reconstruction. Shukla et al. [14] localize surface-breaking cracks in aluminum plates by training a PINN with adaptive activation functions on ultrasonic laser vibrometry data. The framework relies on the 2D scalar acoustic wave equation to model surface displacements and is primarily limited to modeling the behavior of Rayleigh waves in thick plates. The model reconstructs local sound-speed maps that indicate defect geometries. However, it lacks both the elastodynamic formulations required to capture dispersive guided waves in thin plates and the generalization capability to handle new inspection scenarios without model retraining Huang et al. [15] achieve reliable micro-crack localiza tion and depth characterization by mapping equivalent velocity reductions from out-of-plane surface wave displacements, demonstrating robust performance with only 50% of the required training data. However, the framework is restricted to an idealized one-dimensional scan path and employs a decoupled empirical linear regression to infer crack depth rather than performing direct physical parametric inversion. Additionally, the model requires re-optimization for new measurement configurations. Kulkarni and Sabato [16] combine a variational autoencoder with physics-informed constraints to expand sparse thermal measurements to full-field temperature distributions for defect detection on metal plates.

A fundamental limitation of conventional PINN architectures is spectral bias, also referred to as the F-Principle [17, 18]. Standard MLPs tend to prioritize learn ing the low-frequency components of the target function, making them prone to converging to suboptimal local minima [19]. In UGW modeling, this bias manifests as smoothed wavefronts, attenuated scattering signatures and failure to capture phase-dependent interference patterns essential for damage quantification. To address spectral bias, modifications to network architectures have been explored in various studies. Sitzmann et al. [20] introduce Sinusoidal Representation Networks (SIRENs), which employ periodic activation functions that inherently support multi-scale frequency representation. Recent applica tions in wave propagation analysis have further improved spectral resolution through multi-scale Fourier feature embeddings [21, 22, 23]. By embedding Fourier features into a dual-network PINN architecture, Chen et al. [24] mitigate spectral bias to invert high-resolution phase velocity drops from sparse laser ultrasonic data and charac terize densely distributed near-surface defects. However, the proposed framework leaves the full elastodynamic equations unexamined, remains restricted to idealized cylindrical cavity geometries in isotropic media and requires a separate model re-optimization cycle for each unique measurement wavefield. Recent work by Yang et al. [25] demonstrates that defect-parameterized PINN surrogates can effectively reconstruct scattered wavefields from sparse laser ultrasonic measurements, enabling fast inverse characterization of crack geometry. However, their framework relies on time-domain training and empirical transition steepness tuning, which complicates de ployment in practical SHM workflows where rapid baseline generation and standardized damage identification pipelines are required. Despite these advances, the majority of PINN formulations for UGWs are formulated in the time domain, which adds a temporal dimension and introduces known causality issues during training [26]. A dedicated frequency-domain framework operating on the coupled spatial fields remains underexplored in UGW analysis. Frequency-domain PINNs offer distinct advantages for UGW analysis by eliminating temporal derivatives and initial condition dependencies, reducing the problem domain to a stationary boundary value problem [27]. At typical UGW excitation frequencies, shear deformation and rotational inertia cannot be neglected. The Mindlin–Reissner (MR) plate theory [28, 29] provides the kinematic framework to capture these effects through coupled transverse displacement and sectional rotation fields.

The present study introduces a frequency-domain PI-SIREN architecture formulated for the MR plate theory to address these limitations. The model embeds the coupled elastodynamic equations as constraints within the training loss of a SIREN-based neural network, enabling the reconstruction of steady-state guided wavefields without temporal discretization or explicit initial condition constraints. By employing periodic activations, the architecture counteracts spectral bias and thereby resolves the high-frequency ultrasonic oscillations that standard MLPs utilizing tanh activation functions fail to capture. The network parameters are optimized using a hybrid three-phase strategy: first-order Adam [30], residual-adaptive refinement (RAR) [31] and quasi-Newton L-BFGS [32, 33]. The primary contributions of this work are:

• A frequency-domain PI-SIREN framework formulated for the coupled MR plate equations, enabling steady-state wavefield reconstruction from sparse measurements without temporal discretization.

• Numerical and experimental validation demonstrating that the trained model functions as a physics-consistent baseline surrogate, outperforming data-only and no-RAR variants in ablation studies and tanh MLP baselines with and without Fourier feature embedding.

• A two-stage damage identification pipeline integrating the surrogate model with coarse residual scanning and adaptive local refinement, achieving high localization accuracy on experimental ultrasonic wavefield measurements.

## 2 Problem Formulation

Reliable damage detection and localization in UGW SHM require establishing a physically consistent pristine baseline for comparison against measured structural responses [7]. While Rytter’s hierarchy [34] establishes the theoretical steps from damage detection to severity quantification, practical deployments face a fundamental challenge: acquiring high-fidelity baseline measurements across all operational conditions is rarely feasible. Finite element (FE) simulations can provide such baselines but are computationally expensive for repeated evaluations. Furthermore, experimental measurements are inherently sparse due to limited sensor placement, making purely data-driven approaches insufficient for reconstructing the full wavefield without physical regularization. To address these limitations, this work formulates the forward wave propagation problem as a physics-informed regression task. The structure under consideration is a homogeneous, isotropic, linearly elastic plate with lateral dimensions L and thickness h, excited with a five-cycle Hann-windowed burst at a driving frequency of 100 kHz. At this frequency, shear deformation and rotational inertia effects become significant, necessitating the use of first-order shear deformation theory to accurately capture the dispersive UGW modes [28, 35]. The objective is to learn a continuous, differentiable surrogate model $\mathcal { N } _ { \theta } : ( \bar { x } , \bar { y } ) \mapsto $ wˆ that predicts the steady-state transverse displacement wavefield at arbitrary spatial coordinates from sparse measurement data. The parameters θ are obtained by minimizing a composite loss function that enforces the physical constraints while ensuring data fidelity.

Training such surrogates introduces significant computational challenges. First, standard neural networks suffer from spectral bias (the F-Principle), inherently struggling to approximate high-frequency oscillatory solutions characteristic of wave propagation [19]. Second, the MR formulation involves three coupled second-order partial differential equations (PDEs), which can yield an illconditioned loss landscape. These factors combine into a highly non-convex optimization problem prone to convergence to suboptimal local minima. To overcome these challenges, this work employs a network architecture with sinusoidal activation functions that capture the high frequency oscillations and stabilize gradient propagation during training [20].

By embedding the coupled elastodynamic equations of MR plate theory into the network training, the framework enables accurate wavefield reconstruction from sparse inputs while reducing measurement effort. The resulting physics-consistent baseline serves as a computationally efficient alternative to repeated FE simulations and functions as the reference state in the proposed two-stage damage identification pipeline, in which localized scattering signatures are isolated through adaptive comparison against the pristine-state surrogate prediction.

## 3 Governing Equations

The propagation of elastic waves within a continuous, homogeneous and isotropic solid is governed by the Navier– Cauchy equations for elastodynamics. Assuming steadystate time-harmonic excitation of the form $e ^ { \mathrm { i } \omega t }$ , temporal derivatives transform as $\ddot { u }  - \omega ^ { 2 } \hat { u }$ , reducing the problem to the frequency domain. Derived from the conservation of linear momentum and Hooke’s law, the general three-dimensional equation of motion in the absence of volumetric body forces is expressed as [36]:

$$
- \rho \omega ^ { 2 } \hat { \pmb u } = \left( \lambda + \mu \right) \nabla ( \nabla \cdot \hat { \pmb u } ) + \mu \nabla ^ { 2 } \hat { \pmb u } ,\tag{1}
$$

where $\omega = 2 \pi f$ is the angular frequency, $\hat { \pmb u } ( \pmb { x } , \omega ) \in \mathbb { C } ^ { 3 }$ denotes the complex displacement amplitude, with x = $( x , y , z )$ as the spatial coordinate vector, $\rho$ is the mass density and λ and $\mu$ are the Lamé constants defined as:

$$
\lambda = \frac { E \nu } { ( 1 + \nu ) ( 1 - 2 \nu ) } , \qquad \mu = \frac { E } { 2 ( 1 + \nu ) } .\tag{2}
$$

The formulation omits external forces, since the present study focuses on the steady-state response to the excitation, resulting in a source-free homogeneous system of equations. Solving the full three-dimensional elastodynamic equations (1) is computationally expensive. Plate theories reduce the computational effort for wave propagation in plate-like structures where the thickness h is significantly smaller than the lateral dimensions L. To reduce the problem to two dimensions, the system of equations is simplified by means of the MR first-order shear deformation theory. This theory relaxes the strict normality constraint of classical thin-plate formulations by allowing cross-sections to shear. The fundamental kinematic assumption states that straight lines perpendicular to the undeformed mid-surface remain straight and inextensible during deformation but are not required to remain perpendicular to the deformed mid-surface [28, 29]. This introduces two independent rotational degrees of freedom, $\hat { \psi } _ { x }$ and $\hat { \psi } _ { y } ,$ , alongside the transverse displacement wˆ. The three-dimensional displacement vector field in $( x , y , z )$ -direction $\hat { \pmb { u } } = ( \hat { u } , \hat { v } , \hat { w } ) ^ { \top }$ is approximated as:

$$
\hat { u } ( x , y , z , \omega ) = z \hat { \psi } _ { x } ( x , y , \omega ) ,\tag{3}
$$

$$
\hat { v } ( x , y , z , \omega ) = z \hat { \psi } _ { y } ( x , y , \omega ) ,\tag{4}
$$

$$
\hat { w } ( x , y , z , \omega ) = \hat { w } ( x , y , \omega ) ,\tag{5}
$$

where $z \in [ - h / 2 , h / 2 ]$ is the through-thickness coordinate. Substituting this kinematic Ansatz into the threedimensional strain-displacement and constitutive relations and integrating the resulting stresses over the plate thickness yields the steady-state equations of motion. Expressed in terms of complex stress resultants, bending moments $\hat { M } _ { ( \cdot ) }$ and transverse shear forces $\hat { Q } _ { ( \cdot ) }$ , the three coupled equilibrium equations are given by [35]:

$$
\frac { \partial \hat { M } _ { x x } } { \partial x } + \frac { \partial \hat { M } _ { x y } } { \partial y } - \hat { Q } _ { x } + I _ { 2 } \omega ^ { 2 } \hat { \psi } _ { x } = 0 ,\tag{6}
$$

$$
\frac { \partial \hat { M } _ { x y } } { \partial x } + \frac { \partial \hat { M } _ { y y } } { \partial y } - \hat { Q } _ { y } + I _ { 2 } \omega ^ { 2 } \hat { \psi } _ { y } = 0 ,\tag{7}
$$

$$
\frac { \partial \hat { Q } _ { x } } { \partial x } + \frac { \partial \hat { Q } _ { y } } { \partial y } + \rho h \omega ^ { 2 } \hat { w } = 0 ,\tag{8}
$$

where $I _ { 2 } = \rho h ^ { 3 } / 1 2$ is the rotary inertia per unit area. The constitutive relations linking the stress resultants to the

kinematic variables are:

$$
\hat { M } _ { x x } = D \left( \frac { \partial \hat { \psi } _ { x } } { \partial x } + \nu \frac { \partial \hat { \psi } _ { y } } { \partial y } \right) ,\tag{9}
$$

$$
\hat { M } _ { y y } = D \left( \frac { \partial \hat { \psi } _ { y } } { \partial y } + \nu \frac { \partial \hat { \psi } _ { x } } { \partial x } \right) ,\tag{10}
$$

$$
\hat { M } _ { x y } = D \frac { 1 - \nu } { 2 } \left( \frac { \partial \hat { \psi } _ { x } } { \partial y } + \frac { \partial \hat { \psi } _ { y } } { \partial x } \right) ,\tag{11}
$$

$$
\hat { Q } _ { x } = \kappa G h \left( \hat { \psi } _ { x } + \frac { \partial \hat { w } } { \partial x } \right) ,\tag{12}
$$

$$
\hat { Q } _ { y } = \kappa G h \left( \hat { \psi } _ { y } + \frac { \partial \hat { w } } { \partial y } \right) ,\tag{13}
$$

where $D = E h ^ { 3 } / [ 1 2 ( 1 - \nu ^ { 2 } ) ]$ is the flexural rigidity, $G = E / [ 2 ( 1 + \nu ) ]$ is the shear modulus and $\kappa = \bar { 5 } / 6$ is the shear correction factor for rectangular cross-sections. The sign convention follows from the Ansatz $\hat { u } = z \hat { \psi } _ { x }$ for which the transverse shear $\hat { Q } _ { x }$ vanishes in the KL limit $\hat { \psi } _ { x }  - \partial \hat { w } / \partial x$ , consistent with the classical constraint that normals remain perpendicular to the deformed midsurface [28]. At the plate surface $z = h / 2 ,$ , the linear through-thickness displacement field relates the in-plane surface displacements $\hat { u } _ { \mathrm { s u r f } }$ and $\hat { v } _ { \mathrm { s u r f } }$ to the rotations as:

$$
\hat { \psi } _ { x } = \frac { 2 } { h } \hat { u } _ { \mathrm { s u r f } } , \qquad \mathrm { a n d } \qquad \hat { \psi } _ { y } = \frac { 2 } { h } \hat { v } _ { \mathrm { s u r f } } .\tag{14}
$$

## 4 Methodology

The proposed frequency-domain PI-SIREN framework, designed to solve the coupled elastodynamic equations of MR plate theory, is detailed in the following subsections. These subsections present the neural network architecture for high-frequency spectral representation, the formulation of the tailored physics-informed loss function and the proposed hybrid optimization strategy integrating Adam, RAR and L-BFGS fine-tuning.

## 4.1 PI-SIREN Architecture

The surrogate model is constructed as a PINN based on the SIREN architecture. Unlike conventional architectures that utilize standard non-periodic activation functions such as hyperbolic tangent, SIRENs employ periodic sinusoidal activations of the form $\phi ( \xi ) = \sin ( \omega _ { 0 } \xi )$ which are particularly effective for learning the highfrequency oscillations and complex spectral representations inherent in UGW propagation [20]. Here ξ denotes the pre-activation input, defined as the affine transformation of the outputs from the previous layer. Conventional MLPs with smooth, non-periodic activations such as tanh fail to represent high-frequency information and, more critically, its higher-order derivatives, even under dense supervision [20]. The frequency-domain formulation eliminates the temporal dimension, reducing the network input to normalized two-dimensional spatial coordinates $( \bar { x } , \bar { y } ) \in [ - 1 , 1 ] ^ { 2 }$ . The network outputs the three complex surface-displacement amplitudes (u, ˆ¯ v, ˆ¯ wˆ¯) at $f _ { 0 } ,$ normalized by the coordinate map and field scales, each decomposed into its real and imaginary parts, yielding a six-dimensional output vector:

$$
\begin{array} { r } { x _ { \mathrm { o u t } } = \left[ \begin{array} { l } { \mathrm { R e } ( \hat { \bar { u } } ) } \\ { \mathrm { I m } ( \hat { \bar { u } } ) } \\ { \mathrm { R e } ( \hat { \bar { v } } ) } \\ { \mathrm { I m } ( \hat { \bar { v } } ) } \\ { \mathrm { R e } ( \hat { \bar { w } } ) } \\ { \mathrm { I m } ( \hat { \bar { w } } ) } \end{array} \right] . } \end{array}\tag{15}
$$

The rotational fields $\hat { \psi } _ { x }$ and $\hat { \psi } _ { y }$ required by the MR plate theory are derived analytically from the predicted displacements using first-order shear deformation kinematics according to Eq. (14). During training, only the transverse displacement w<sup>ˆ</sup>¯ is constrained by measurement data. The in-plane displacements and the derived rotational fields are learned implicitly through the implemented governing PDEs and boundary conditions (BCs). This parameterization matches the quantity that is directly observable in the experiments.

The model employs a fully connected neural network with $N _ { L } = 5$ hidden layers, each containing $N = 2 5 6$ neurons per layer. Automatic differentiation (AD) is used to compute the spatial derivatives required for the PDE residuals. To preserve activation variance and ensure stable gradient flow across the network, the weights are initialized according to Sitzmann et al. [20]. Specifically, the first layer uses $\mathcal { U } ( - 1 / n _ { i n } , 1 / n _ { i n } )$ , while all subsequent layers use $\mathcal { U } ( - \sqrt { 6 / n _ { i n } } / \omega _ { 0 } , \sqrt { 6 / n _ { i n } } / \omega _ { 0 } )$ , where $n _ { i n }$ denotes the number of input neurons to each layer. The SIREN frequency hyperparameter is set to $\omega _ { 0 } = 3 0$ . Figure 1 provides a schematic representation of the proposed PI-SIREN architecture.

## 4.2 Loss Function

The network parameters $\pmb \theta$ are optimized by minimizing the composite loss function that enforces the governing physics, boundary conditions and measurement data. The overall objective function to be minimized is defined as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t o t a l } } ( \pmb { \theta } ) = \lambda _ { \mathrm { P D E } } \mathcal { L } _ { \mathrm { P D E } } + \lambda _ { \mathrm { B C } } \mathcal { L } _ { \mathrm { B C } } + \lambda _ { \mathrm { d a t a } } \mathcal { L } _ { \mathrm { d a t a } } , } \end{array}\tag{16}
$$

where $\lambda _ { \mathrm { P D E } } , \lambda _ { \mathrm { B C } }$ and $\lambda _ { \mathrm { d a t a } }$ are constant weighting coefficients that balance the relative contribution of each loss term during optimization. The PDE residual terms are formulated in normalized coordinates and scaled with field variables to ensure stable optimization. The physical domain $( x , y ) \in [ 0 , L ] ^ { 2 }$ is mapped to the computational domain $( \bar { x } , \bar { y } ) \in [ - 1 , \bar { 1 } ] ^ { 2 }$ by

$$
\bar { x } = 2 \Bigl ( \frac { x } { L } \Bigr ) - 1 , \qquad \bar { y } = 2 \Bigl ( \frac { y } { L } \Bigr ) - 1 ,\tag{17}
$$

which introduces the chain-rule scaling factors $c _ { x } =$ $c _ { y } = 2 / L$ into the spatial derivatives: $\bar { \partial } _ { x } = c _ { x } \partial _ { \bar { x } }$ and $\partial _ { x } ^ { n } = c _ { x } ^ { n } \partial _ { \bar { x } } ^ { n }$ . The physical fields are scaled to $O ( 1 )$ as $\hat { w } = \bar { W } _ { s } \hat { \bar { w } } , \hat { u } _ { \mathrm { s u r f } } = U _ { s } \hat { \bar { u } }$ and $\hat { v } _ { \mathrm { s u r f } } = V _ { s } \hat { \bar { v } }$ , where $W _ { s }$ is the maximum complex-magnitude amplitude of the transverse displacement in the reference data. For experimental data, $U _ { s }$ and $V _ { s }$ are not directly measurable and are set relative to W<sub>s</sub> from dispersion analysis and parametric studies as $U _ { s } = V _ { s } = \dot { W _ { s } } / 2$ . Combining the surface scaling with the kinematic relation of Eq. (14), the rotations scale as

$$
\hat { \psi } _ { x } = \Psi _ { s } \hat { \bar { \psi } } _ { x } , \quad \hat { \psi } _ { y } = \Psi _ { s } \hat { \bar { \psi } } _ { y } , \qquad \Psi _ { s } = \frac { 2 U _ { s } } { h } ,\tag{18}
$$

and the normalized rotations are identified with the inplane network outputs, $\bar { \psi } _ { x } \equiv \hat { \bar { u } }$ and $\ddot { \psi } _ { y } \equiv \hat { v }$ . Substituting the constitutive laws, the scaling and the normalized coordinates into Eqs. (6)–(8) yields the PDE residuals R:

$$
\mathcal { R } _ { \psi _ { x } } = D c _ { x } ^ { 2 } \Psi _ { s } \left[ \frac { \partial ^ { 2 } \hat { \bar { \psi } } _ { x } } { \partial \bar { x } ^ { 2 } } + \frac { 1 - \nu } { 2 } \frac { \partial ^ { 2 } \hat { \bar { \psi } } _ { x } } { \partial \bar { y } ^ { 2 } } + \frac { 1 + \nu } { 2 } \frac { \partial ^ { 2 } \hat { \bar { \psi } } _ { y } } { \partial \bar { x } \partial \bar { y } } \right]
$$

$$
- C _ { s } \bigg [ \Psi _ { s } \hat { \bar { \psi } } _ { x } + c _ { x } W _ { s } \frac { \partial \hat { \bar { w } } } { \partial \bar { x } } \bigg ] + I _ { 2 } \omega ^ { 2 } \Psi _ { s } \hat { \bar { \psi } } _ { x } ,\tag{19}
$$

$$
\mathcal { R } _ { \psi _ { y } } = D c _ { y } ^ { 2 } \Psi _ { s } \left[ \frac { \partial ^ { 2 } \hat { \bar { \psi } } _ { y } } { \partial \bar { y } ^ { 2 } } + \frac { 1 - \nu } { 2 } \frac { \partial ^ { 2 } \hat { \bar { \psi } } _ { y } } { \partial \bar { x } ^ { 2 } } + \frac { 1 + \nu } { 2 } \frac { \partial ^ { 2 } \hat { \bar { \psi } } _ { x } } { \partial \bar { x } \partial \bar { y } } \right]
$$

$$
- C _ { s } \left[ \Psi _ { s } \hat { \bar { \psi } } _ { y } + c _ { y } W _ { s } \frac { \partial \hat { \bar { w } } } { \partial \bar { y } } \right] + I _ { 2 } \omega ^ { 2 } \Psi _ { s } \hat { \bar { \psi } } _ { y } ,\tag{20}
$$

$$
\begin{array} { l } { \displaystyle { \mathcal { R } _ { w } = C _ { s } c _ { x } \left[ \Psi _ { s } \left( \frac { \partial \hat { \bar { \psi } } _ { x } } { \partial \bar { x } } + \frac { \partial \hat { \bar { \psi } } _ { y } } { \partial \bar { y } } \right) + c _ { x } W _ { s } \nabla _ { \bar { x } \bar { y } } ^ { 2 } \hat { \bar { w } } \right] } } \\ { \displaystyle { + \rho h \omega ^ { 2 } W _ { s } \hat { \bar { w } } , } } \end{array}\tag{21}
$$

where $C _ { s } ~ = ~ \kappa G h$ is the transverse shear stiffness and $\nabla _ { \bar { x } \bar { y } } ^ { 2 } = \mathrm { ~ } \breve { \partial } ^ { 2 } / \partial \bar { x } ^ { 2 } + \partial ^ { 2 } / \partial \bar { y } ^ { 2 }$ the two-dimensional Laplacian. The PDE loss L penalizes violations of the three coupled MR equations in Eqs. (19)–(21), which are evaluated at $N _ { f }$ collocation points sampled randomly across the normalized spatial domain $( \bar { x } , \bar { y } ) \in [ - 1 , 1 ] ^ { 2 }$ . Defining $\mathcal { R } ^ { ( j ) } = [ \mathcal { R } _ { \psi _ { x } } ^ { ( j ) } , \mathcal { R } _ { \psi _ { y } } ^ { ( j ) } , \mathcal { R } _ { w } ^ { ( j ) } ] ^ { \top }$ as the complex residual vector at the $j \mathrm { - t h }$ collocation point (encompassing both real and imaginary parts), the PDE loss is formulated as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { P D E } } = \displaystyle \frac { 1 } { N _ { f } } \sum _ { j = 1 } ^ { N _ { f } } { \left( s _ { \mathrm { b e n d } } ^ { 2 } \left| \mathcal { R } _ { \psi _ { x } } ^ { ( j ) } \right| ^ { 2 } + s _ { \mathrm { b e n d } } ^ { 2 } \left| \mathcal { R } _ { \psi _ { y } } ^ { ( j ) } \right| ^ { 2 } \right. } } \\ { \left. + s _ { \mathrm { v e r t } } ^ { 2 } \left| \mathcal { R } _ { w } ^ { ( j ) } \right| ^ { 2 } \right) } ,  \end{array}\tag{22}
$$

where $s _ { \mathrm { v e r t } } = 1 / \mathcal { M } _ { \mathrm { v e r t } }$ and $s _ { \mathrm { b e n d } } ~ = ~ 1 / \mathcal { M } _ { \mathrm { b e n d } }$ are perequation scaling factors, ensuring that each squared residual contributes a dimensionless term of order $\mathcal { O } ( 1 )$ at initialization. The corresponding characteristic magnitudes $\mathcal { M } _ { \mathrm { b e n d } }$ and ${ \mathcal { M } } _ { \mathrm { v e r t } }$ are estimated from the dominant residual terms at the excitation frequency as:

![](images/51848cc0c17b57f7823f6ccb15f5f058a95343f5788e477295f3b47ec937bf57.jpg)  
Figure 1: Schematic representation of the frequency-domain PI-SIREN architecture for solving the normalized MR plate theory equations.

$$
\begin{array} { r l } & { \mathcal { M } _ { \mathrm { b e n d } } = \operatorname* { m a x } \bigr ( D c _ { x } ^ { 2 } \Psi _ { s } , \ C _ { s } \Psi _ { s } , } \\ & { \qquad \quad C _ { s } c _ { x } W _ { s } , \ I _ { 2 } \omega ^ { 2 } \Psi _ { s } \bigr ) , } \\ & { \mathcal { M } _ { \mathrm { v e r t } } = \operatorname* { m a x } \bigr ( C _ { s } c _ { x } \Psi _ { s } , \ C _ { s } c _ { x } ^ { 2 } W _ { s } , \ \rho h \omega ^ { 2 } W _ { s } \bigr ) . } \end{array}\tag{23}
$$

(24)

The boundary condition loss $\mathcal { L } _ { \mathrm { B C } }$ enforces free-edge conditions on all four plate edges. Specifically, for edges normal to the x¯-direction $( \bar { x } \ = \ - 1$ and $\dot { \bar { x } } = 1 )$ , the boundary residuals are defined by the vanishing of the bending moment $\hat { M } _ { x x }$ , twisting moment $\hat { M } _ { x y }$ and transverse shear force ${ \hat { Q } } _ { x } .$ . For edges normal to the y¯-direction $( \bar { y } = - 1$ and $\bar { y } = 1 )$ , the corresponding residuals involve $\hat { M } _ { y y } , \hat { M } _ { x y }$ and $\hat { Q } _ { y }$ :

$$
\begin{array} { r l r } & { \mathcal { R } _ { M _ { n n } } ^ { x } = \hat { M } _ { x x } , } & { \mathcal { R } _ { M _ { n s } } ^ { x } = \hat { M } _ { x y } , \mathcal { R } _ { Q _ { n } } ^ { x } = \hat { Q } _ { x } , } \\ & { \mathcal { R } _ { M _ { n n } } ^ { y } = \hat { M } _ { y y } , } & { \mathcal { R } _ { M _ { n s } } ^ { y } = \hat { M } _ { x y } , \mathcal { R } _ { Q _ { n } } ^ { y } = \hat { Q } _ { y } , } \end{array}\tag{25}
$$

(26)

where the stress resultants $\hat { M } _ { ( \cdot ) }$ and $\hat { Q } _ { ( \cdot ) }$ are evaluated from the constitutive relations in Eqs. (9)–(13) using the normalized fields and the scaling factors defined above. The boundary loss evaluated at $N _ { b }$ points is then defined as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { B C } } = \displaystyle \frac { 1 } { N _ { b } } \sum _ { k = 1 } ^ { N _ { b } } { \left( s _ { M } ^ { 2 } \left| \mathcal { R } _ { M _ { n n } } ^ { ( k ) } \right| ^ { 2 } + s _ { M } ^ { 2 } \left| \mathcal { R } _ { M _ { n s } } ^ { ( k ) } \right| ^ { 2 } \right. } } \\ { \left. + s _ { Q } ^ { 2 } \left| \mathcal { R } _ { Q _ { n } } ^ { ( k ) } \right| ^ { 2 } \right) } ,  \end{array}\tag{27}
$$

where $s _ { M } = 1 / \mathcal { M } _ { M }$ and $s _ { Q } = 1 / \mathcal { M } _ { Q }$ are scaling factors derived from the characteristic magnitudes of bending moments and shear forces, respectively. These magnitudes are estimated based on the bending stiffness and transverse shear stiffness contributions:

$$
\begin{array} { l } { { \mathcal { M } _ { M } \approx D c _ { x } \Psi _ { s } , } } \\ { { \mathcal { M } _ { Q } \approx C _ { s } \operatorname* { m a x } ( \Psi _ { s } , c _ { x } W _ { s } ) . } } \end{array}\tag{28}
$$

(29)

This ensures balanced gradient signals across moment and shear constraints. The global loss weights are set to $\lambda _ { \mathrm { P D E } } = 1 , \lambda _ { \mathrm { d a t a } } = 1 0$ and $\lambda _ { \mathrm { B C } } = 1 0$ to account for the stricter enforcement required by boundary conditions relative to the domain residuals.

Finally, the data loss ${ \mathcal { L } } _ { \mathrm { d a t a } }$ ensures fidelity to the numerical and experimental benchmark by comparing the network’s predictions at $N _ { d }$ spatial locations against FE or experimental data. Data supervision is applied exclusively to the transverse displacement w<sup>ˆ</sup>¯. The data loss is computed as:

$$
\mathcal { L } _ { \mathrm { d a t a } } = \frac { 1 } { N _ { d } } \sum _ { i = 1 } ^ { N _ { d } } \left| \hat { \bar { w } } _ { \mathrm { p r e d } } ^ { ( i ) } - \hat { \bar { w } } _ { \mathrm { r e f } } ^ { ( i ) } \right| ^ { 2 } ,\tag{30}
$$

where $\hat { \bar { w } } _ { \mathrm { p r e d } }$ and ${ \hat { \bar { w } } _ { \mathrm { r e f } } }$ denote the predicted and reference normalized transverse displacement vectors, respectively.

## 4.3 Optimization Strategy

For the training of the proposed PI-SIREN architecture, a hybrid optimization strategy is employed. The methodology sequentially combines first-order optimization with Adam [30], RAR [31] of the collocation set and quasi-Newton fine-tuning with L-BFGS [32, 33]. The optimization process is summarized in Algorithm 1 and detailed below. In the first optimization stage, the Adam optimizer [30] minimizes the composite loss function $\bar { \mathcal { L } } _ { \mathrm { t o t a l } } ( \pmb { \theta } )$ defined in Eq. (16). Adam employs randomized resampling of collocation points at a constant learning rate of $\eta = 2 \times 1 0 ^ { - 5 }$ to ensure stable convergence. At every optimization step, a new set of $N _ { f }$ spatial points ${ \mathcal { T } } ^ { ( n ) }$ is randomly sampled from the normalized domain $( \bar { x } , \bar { y } ) \in [ - 1 , 1 ] ^ { 2 }$ to evaluate the PDE residuals. The stochastic sampling prevents the network from adapting to a fixed spatial pattern. To concentrate collocation points in regions of high residual error while maintaining a bounded point budget for L-BFGS optimization, a hybrid RAR strategy is employed. RAR evaluates the squared PDE residual magnitude across a dense, fixed candidate set $S _ { \mathrm { c a n d } } \subset [ - 1 , \overline { { { 1 } } } ] ^ { 2 }$ using the parameters learned during Phase 1. The points with the largest residual errors are identified and merged with a subset of points drawn uniformly from the domain. This prioritizes regions where physical constraints are most violated. The collocation points identified in Phase 2 serve as the fixed evaluation set for Phase 3 fine-tuning. Bounding the collocation point budget is what makes the subsequent full-batch quasi-Newton step tractable, since standard L-BFGS requires the residual to be evaluated on a fixed set at every iteration. The final optimization phase employs the L-BFGS quasi-Newton algorithm for finetuning. L-BFGS approximates the inverse Hessian of the loss function, enabling faster convergence to high precision compared to first-order methods, at increased computational costs. This phase is essential since spectral bias is a dynamical property of the training process and not solely a limitation of the model representation. Resolving high-frequency content requires curvature-aware optimization, where first-order methods tend to stall on the high-frequency components of the solution [18]. The quasi-Newton update in L-BFGS relies on accurate gradients. The MR residuals contain second-order operators whose spectral contributions span several orders of magnitude. In single precision arithmetic (FP32) the resulting gradient noise can satisfy the L-BFGS convergence criterion prematurely. Therefore, all model parameters and arithmetic operations during the L-BFGS phase are cast to double precision (FP64) [37, 38].

```tcl
Algorithm 1 Optimization strategy
Require: Collocation budget $K ,$ , Candidate set size
$\bar { N } _ { \mathrm { c a n d } } .$ Gradient-norm tolerance $\tau ,$ Learning rate $\eta ,$
Adam iterations $N _ { \mathrm { A d a m } } .$ max L-BFGS iterations $M _ { \mathrm { m a x } }$
Initialize: Network parameters $\pmb { \theta } _ { 0 }$
Phase 1: Adam Optimization (FP32)
for $n = 0 , \ldots , \bar { N _ { \mathrm { A d a m } } } - 1$ do
Sample coll. set $\begin{array} { r } { \mathcal { T } ^ { ( n ) } \sim \mathcal { U } ( [ - 1 , 1 ] ^ { 2 } ) , | \mathcal { T } ^ { ( n ) } | = } \end{array}$
$N _ { f }$
Evaluate ${ \mathcal { L } } _ { \mathrm { t o t a l } } ( \theta _ { n } ; \mathcal { T } ^ { ( n ) } )$ , update via Adam:
$\theta _ { n + 1 } \gets \mathrm { A d a m } \big ( \theta _ { n } , \nabla _ { \theta } \mathcal { L } _ { \mathrm { t o t a l } } ( \theta _ { n } ; \mathcal { T } ^ { ( n ) } ) , \eta \big )$
end for
Phase 2: Residual-Adaptive Refinement (RAR)
Generate candidate set $S _ { \mathrm { c a n d } } \subset [ - 1 , 1 ] ^ { 2 }$ of size
$N _ { \mathrm { c a n d } } ,$ evaluate residuals $r _ { i } = \lVert \dot { \mathcal { R } } ( \pmb { \xi } _ { i } ; \pmb { \theta } _ { N _ { \mathrm { A d a m } } } ) \rVert ^ { 2 }$
for $\pmb { \xi } _ { i } \in \mathcal { S } _ { \mathrm { c a n d } } .$ . Construct optimized collocation set:
$\mathcal { T } _ { \mathrm { L - B F G S } }  \mathrm { t o p } _ { K _ { \mathrm { h a r d } } } ( \{ r _ { i } \} ) \cup \mathrm { r a n d } _ { K _ { \mathrm { r a n d } } } ( \mathcal { T } ^ { ( n ) } )$
Phase 3: L-BFGS fine-tuning (FP64)
Cast $\pmb { \theta } _ { N _ { \mathrm { A d a m } } }  \pmb { \theta } _ { 0 } ^ { ( 6 4 ) }$ , set m $\gets 0$
while m $< M _ { \mathrm { m a x } }$ and $\begin{array} { r l } {  { \big \| \nabla _ { \theta } \mathcal { L } _ { \mathrm { t o t a l } } ( \pmb { \theta } _ { m } ^ { ( 6 4 ) } ; \mathcal { T } _ { \mathrm { L - B F G S } } ) \big \| } } & { { } \ge } \end{array}$
τ do
Update $\pmb { \theta } ^ { ( 6 4 ) }$ using L-BFGS step:
$\begin{array} { r } { \pmb { \theta } _ { m + 1 } ^ { ( 6 4 ) }  \mathrm { L . B F G S } ( \pmb { \theta } _ { m } ^ { ( 6 4 ) } , \mathcal { L } _ { \mathrm { t o t a l } } ( \cdot ; \mathcal { T } _ { \mathrm { L \mathrm { - B F G S } } } ) ) , } \end{array}$
m $\dot {  } m + 1$
end while
return $\theta _ { \mathrm { f i n a l } }  \theta _ { m } ^ { ( 6 4 ) }$
```

## 4.4 Numerical Dataset

The numerical dataset is generated through threedimensional FE simulations in Abaqus/Explicit [39]. The structure under consideration is a homogeneous, isotropic, linearly elastic aluminum plate of dimensions $( L \times \dot { L ^ { \times } } h ) = 2 0 \dot { 0 } \times 2 0 0 \times 3$ mm (EN AW-5083) with $E =$ $7 0 \mathrm { G P a } , \nu = 0 . 3 3$ and $\rho = 2 6 6 0 \mathrm { k g / m ^ { 3 } }$ . The plate is ex cited with a five-cycle Hann-windowed burst at a driving frequency of 100 kHz. The dominant response frequency, determined from the measured wavefield spectrum is $f _ { 0 } = 9 6 . 4 \mathrm { k H z } .$ A mesh independence study confirmed convergence at 40 elements per shortest wavelength of the $A _ { 0 }$ mode at the excitation frequency. Therefore, the plate is discretized with reduced-integration, eight-node brick elements (C3D8R) at this resolution, yielding a total of 980,000 elements. All four plate edges are simulated traction-free. Parametric studies varying the Rayleigh damping coefficients (mass-proportional and stiffnessproportional) showed that an undamped model reproduces the experimental wavefield amplitudes and phases most accurately. The explicit time-domain simulation runs for $T = 3 0 0 \mu \mathrm { s }$ utilizing automatic time increments with $\Delta t \approx 0 . 0 \dot { 5 } \mu \mathrm { s }$ The excitation is applied as a concentrated force in the z-direction at the actuator position $( x , y ) = ( 1 0 0 , 1 0 0 )$ mm. The time-domain force history is taken from the experiment: the actuator displacement trace measured at the excitation location is spatially aver aged over a 5 mm-radius ring centered at (100, 100) mm, bandpass filtered between 30 and 150 kHz and normalized before being loaded into the FE model. The actuator is driven with a five-cycle Hann-windowed sinusoidal burst at a center frequency of 100 kHz. The complex amplitude fields $( \hat { \bar { u } } , \hat { \bar { v } } , \hat { \bar { w } } )$ are extracted at the dominant response frequency $f _ { 0 } = 8 9 . 9$ kHz from the steady-state portion of the simulation. The resulting dataset provides the complex-valued transverse displacement wˆ(x, f<sub>0</sub>) on the FE mesh nodes over the full $2 0 0 \times 2 0 0$ mm plate domain. From this field the normalized transverse displacement w<sup>ˆ</sup>¯ is obtained through the amplitude scaling $W _ { s }$ . This reference wavefield serves as the ground-truth baseline for training the PI-SIREN surrogate for the numerical validation of the damage identification pipeline and for comparison against the experimental dataset.

![](images/7a38c11409dbbfe62f95db9c9b95b96c24e2974051a980926ccf713b8ba56fdd.jpg)  
Figure 2: Schematic representation of the experimental robotic measurement setup.

## 4.5 Experimental Dataset

The experimental campaign is performed on EN AW-5083 aluminum plates matching the numerical specimen material and geometry. Figure 2 illustrates the technical measurement setup used for this study. Excitation is achieved using a DuraAct piezoelectric transducer (cylin drical, 10 mm diameter, 0.25 mm thickness) vacuummounted at the plate center (100, 100) mm. The actuator is driven through a dedicated power amplifier, delivering a 100 kHz, five-cycle Hann-windowed sinusoidal burst. Surface vibrations are captured non-contactly using a Xarion ETA450 T-Shape optical microphone mounted on a UR-10 robotic manipulator. The microphone measures the acoustic pressure $p ( { \pmb x } , t )$ in the near field above the plate $( h _ { \mathrm { s u r f } } \approx 5$ mm), which is proportional to the normal surface velocity. The transverse displacement $w ( \pmb { x } , t )$ is obtained by time-integrating the pressure trace and dividing by the acoustic impedance of air $Z = \rho _ { 0 } c _ { 0 }$ . The incident angle between the microphone and plate surface is set to $9 0 °$ . According to Snell’s law, however, the plate radiates a leaky plane wave into the surrounding air at an angle of $\theta _ { p } = \arcsin ( c _ { \mathrm { a i r } } / c _ { p } )$ from the surface normal [36]. With $c _ { \mathrm { a i r } } = 3 4 3 \mathrm { m } / \mathrm { s }$ as the speed of sound at room temperature and $c _ { p } \approx 1 5 2 9 \mathrm { m / s }$ as the phase velocity of the $A _ { 0 }$ -mode, obtained from dispersion relations, $\theta _ { p }$ is calculated to $1 3 ^ { \circ }$ and $\cos ( \theta _ { p } )$ is used as amplitude correction factor. The microphone output is recorded via a PicoScope 4842A oscilloscope. The raw measurement traces undergo the following pre-processing before frequency-domain analysis. Since the measurements extend beyond the plate edges, the spatial scan grid is co-registered with the plate geometry and cropped to the 200 × 200 mm physical domain. The global time offset is estimated by averaging the traces at the grid points closest to the actuator and locating the onset of the firstarriving wave. Then, all traces are shifted so that $t = 0$ coincides with the start of the excitation signal. Each trace undergoes bandpass filtering with a fourth-order

Butterworth filter (passband 30–150 kHz) and truncation to $3 0 0 \mu \mathrm { s }$ to match the numerical simulation duration. The pre-processed dataset comprises $N _ { d }$ time-domain displacement traces recorded on a uniform spatial grid over the $2 0 0 \times 2 0 0$ mm plate surface. From these traces the complex amplitude field $\hat { w } ( { \pmb x } , f _ { 0 } )$ of the harmonic component at $f _ { 0 } = 9 6 . \nonumber$ 4 kHz is obtained from the discrete Fourier transform (DFT), providing the experimental reference for training and validation.

## 4.6 Damage Identification Methodology

The trained pristine PI-SIREN provides a continuous, physics-consistent representation of the undamaged wavefield that can be evaluated at arbitrary spatial locations without retraining. Damage is detected by subtracting this baseline from a new measurement. The full procedure is illustrated in Figure 3 and comprises an offline training phase followed by a sequential online pipeline: a coarse-grid evaluation and an adaptive refinement evaluation of the surrogate. Let ${ \hat { w } } _ { \mathrm { d a m } } ( { \boldsymbol { x } } )$ denote the complex transverse displacement amplitude of the damaged plate at $\boldsymbol { x } = ( x , y )$ , obtained from measurement or FE simulations and let $\hat { w } _ { \mathrm { p r i } } ( \pmb { x } )$ be the corresponding pristine PI-SIREN prediction. Minor global amplitude and phase discrepancies are removed by fitting the pristine field to the damaged field through a single complex scalar α:

$$
\begin{array} { r l r } {  { \alpha = \underset { \alpha \in \mathbb { C } } { \arg \operatorname* { m i n } } \| \hat { w } _ { \mathrm { d a m } } - \alpha \hat { w } _ { \mathrm { p r i } } \| _ { 2 } ^ { 2 } } } \\ & { } & { = \frac { \displaystyle \sum _ { x \in \Omega _ { v } } \hat { w } _ { \mathrm { d a m } } ( \pmb { x } ) \overline { { \hat { w } _ { \mathrm { p r i } } ( \pmb { x } ) } } } { \displaystyle \sum _ { x \in \Omega _ { v } } | \hat { w } _ { \mathrm { p r i } } ( \pmb { x } ) | ^ { 2 } } , } \end{array}\tag{31}
$$

where $\Omega _ { v }$ is the set of valid measurement points and $\overline { { ( \cdot ) } }$ denotes complex conjugation. The scattered field is the

![](images/c2aaf6179b90a92e20babfb464188fccce951dc4f17546acd79db5744da6fd4c.jpg)  
Figure 3: Sequential damage identification pipeline on experimental data: The pristine baseline model is trained once offline and reused as pre-trained model in every online inspection for inference. Online phases I–IV: coarse field acquisition, baseline subtraction and candidate detection, adaptive resampling and refined damage localization.

residual

$$
\hat { w } _ { \mathrm { s c a t } } ( \pmb { x } _ { i } ) = \hat { w } _ { \mathrm { d a m } } ( \pmb { x } _ { i } ) - \alpha \hat { w } _ { \mathrm { p r i } } ( \pmb { x } _ { i } ) ,\tag{32}
$$

with $i = 1 , \ldots , N _ { s }$ , evaluated at $N _ { s }$ measurement points before interpolation. This subtraction suppresses the dominant incident wavefield, retaining the weaker scattering signature of the defect.

The first evaluation, phase (I) of Figure 3, operates on a sparse uniform measurement grid. The scattered values are interpolated onto a Cartesian grid via Clough– Tocher [40] cubic interpolation, $\widetilde { w } _ { \mathrm { s c a t } } \overline { { \mathbf { \Gamma } } } = \mathbf { \Gamma } \mathcal { T } [ \hat { w } _ { \mathrm { s c a t } } ]$ and smoothed with an isotropic Gaussian kernel of width σ to yield $W _ { \sigma } = G _ { \sigma } * \widetilde { w } _ { \mathrm { s c a t } }$ . Three complementary indicators are then evaluated. The first two act on the scattered field, while the third compares damaged and pristine fields locally:

$$
\mathcal { D } _ { \mathrm { m a g } } ( \pmb { x } ) = \big | W _ { \sigma } ( \pmb { x } ) \big | ,\tag{33}
$$

$$
{ \mathcal D } _ { { \mathcal H } } ( { \pmb x } ) = \big | \mathrm { I m } \{ \overline { { W _ { \sigma } } } \nabla ^ { 2 } { W _ { \sigma } } \} \big | ,\tag{34}
$$

$$
\mathcal { D } _ { \mathrm { i n c } } ( \pmb { x } ) = 1 - \frac { \left| \left. \hat { w } _ { \mathrm { d a m } } \overline { { \hat { w } _ { \mathrm { p r i } } } } \right. _ { B _ { R } } ( \pmb { x } ) \right| } { \left. \left| \hat { w } _ { \mathrm { d a m } } \right| \right. _ { B _ { R } } ( \pmb { x } ) \left. \left| \hat { w } _ { \mathrm { p r i } } \right| \right. _ { B _ { R } } ( \pmb { x } ) } ,\tag{35}
$$

where $\langle \cdot \rangle _ { B _ { R } }$ denotes a box average over a square window of half-width R and $\hat { w } _ { \mathrm { p r i } }$ is the aligned pristine field α-scaled according to Eq. (31). The three indicators are complementary: $\mathcal { D } _ { \mathrm { m a g } }$ is robust to noise but poorly localized, $\mathcal { D _ { H } }$ is sharply localized but derivativebased and noise-sensitive and $\dot { \mathcal { D } } _ { \mathrm { i n c } }$ depends only on local phase coherence and is insensitive to amplitude calibration. The Helmholtz-type indicator is proportional to the divergence of the time-averaged flexural intensity flux, $\nabla \cdot \mathbf { I } \propto \operatorname { I m } \{ \overline { { \hat { w } } } \nabla ^ { 2 } \hat { w } \}$ , which for the MR plate reduces to $- \mathrm { I m } \{ \overline { { \hat { w } } } \left( \nabla \cdot \boldsymbol { \psi } \right) \}$ . Fusing the three indicators reduces the risk that a defect is missed due to measurement conditions that degrade a single indicator.

To suppress smooth background residuals, each indicator is normalized via a rectified z-score over a sliding box window of half-width R. The rectified z-score

$$
Z ( { \pmb x } ) = \operatorname* { m a x } \left( \frac { \mathcal { D } ( { \pmb x } ) - \mu _ { R } ( { \pmb x } ) } { \varsigma _ { R } ( { \pmb x } ) } , 0 \right) \chi _ { v } ( { \pmb x } )\tag{36}
$$

retains only positive deviations from the local background, where $\chi _ { v }$ masks the valid measurement domain. The three z-scored indicators are rescaled to [0, 1] and combined into a single fused detection score via an equal-weighted mean. In phase (II), candidate damage regions are identified as local maxima of this fused score that exceed a quantile threshold. A non-maximum suppression step enforces a minimum spatial separation between candidates to avoid clustered duplicates. The remaining candidates are ranked by score and define rectangular regions of interest. These regions are adaptively rescanned on a finer grid in phase (III), while the original coarse points are retained. This concentrates the measurement effort where anomalies were initially detected.

In the second evaluation, phase (III) and (IV), the pretrained PI-SIREN is reevaluated at all measurement points (coarse and resampled) and the scattered field is recomputed. The fused score serves exclusively for candidate detection in the first pass. For the final localization, only $\mathcal { D _ { H } }$ is evaluated, since at the refined sampling density its derivative-based sharpness resolves the defect centroid more precisely than the fused score. The dominant response region is isolated by thresholding $\mathcal { D } _ { \mathcal { H } } .$ yielding the region $\Omega _ { b }$ . The damage geometry within $\Omega _ { b }$ is characterized by intensity-weighted image moments. With $s ( \pmb { x } )$ denoting the normalized indicator and $\begin{array} { r } { m _ { 0 } = \sum _ { { \pmb x } \in \Omega _ { b } } \dot { \pmb S } ( { \pmb x } ) } \end{array}$ its integrated mass, the centroid and covariance matrix are

Table 1: Reconstruction ablation study on the $2 5 \times 2 5$ grid. Relative test error $L _ { 2 } ( \hat { \bar { w } } )$ on a fixed dense test set and the final PDE residual $\mathcal { L } _ { \mathrm { P D E } }$
<table><tr><td>Tag</td><td>Configuration</td><td> $L _ { 2 } ( \hat { \overline { { w } } } )$ </td><td> $\mathcal { L } _ { \mathrm { P D E } }$ </td></tr><tr><td>M1</td><td>Data-only  $( \lambda _ { \mathrm { P D E } } = \lambda _ { \mathrm { B C } } = 0 )$ </td><td>0.4932</td><td></td></tr><tr><td>M2</td><td> $\mathrm { P I N N , A d a m { \mathrm { - } } o n l y ~ ( 7 0 , 0 0 0 ~ i t e r . ) }$ </td><td>0.2881</td><td> $2 . 7 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>M3</td><td> $\mathrm { P I N N } , \mathrm { A d a m + L { \cdot } B F G S }$ </td><td>0.1351</td><td> $5 . 2 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>M4</td><td> $\mathrm { P I N N , A d a m + R A R + L \mathrm { - } B F G S }$ </td><td>0.1037</td><td> $5 . 6 \times 1 0 ^ { - 6 }$ </td></tr></table>

$$
\pmb { x } _ { c } = \frac { 1 } { m _ { 0 } } \sum _ { \pmb { x } \in \Omega _ { b } } \pmb { S } ( \pmb { x } ) \pmb { x } ,\tag{37}
$$

$$
\pmb { \Sigma } = \frac { 1 } { m _ { 0 } } \sum _ { \pmb { x } \in \Omega _ { b } } S ( \pmb { x } ) \left( \pmb { x } - \pmb { x } _ { c } \right) \left( \pmb { x } - \pmb { x } _ { c } \right) ^ { \top } .\tag{38}
$$

The centroid $\scriptstyle { \mathbf { { \mathit { x } } } } _ { c }$ gives the damage location. The eigenvectors of Σ yield the principal orientation $\theta _ { c }$ and the eigenvalues $\varrho _ { 1 } ~ \geq ~ \varrho _ { 2 }$ give the characteristic extents $\ell _ { k } \ = \ 2 \sqrt { 3 } \sqrt { \varrho _ { k } }$ . The pipeline outputs $\mathbf { \delta } _ { \mathbf { \boldsymbol { x } } _ { c } , \mathbf { \delta } } \theta _ { c }$ and the aspect ratio $\dot { \ell } _ { 1 } / \ell _ { 2 }$ of the damage.

## 5 Results

This section evaluates the proposed PI-SIREN against the numerical and experimental datasets. The proposed optimization strategy (Adam, RAR, L-BFGS) for pristine wavefield reconstruction using FE simulation data is established in the subsection on numerical wavefield reconstruction. Next, the minimum spatial sensor density required for reliable damage localization is examined using the same FE dataset. The contribution of the network architecture is then isolated by applying the resulting configuration to conventional and Fourier-feature MLP-tanh baselines. The resulting sensor density and architecture are used in the experimental campaign, where the pristine reconstruction is validated against measured wavefields and the two-stage damage identification pipeline is applied to experimental data containing an introduced defect.

## 5.1 Numerical Wavefield Reconstruction

In this section, the proposed optimization strategy (Algorithm 1) is analyzed and evaluated in terms of its effectiveness as part of an ablation study. The strategy is evaluated using four training configurations labeled M1–M4 in Table 1, which differ by a single component. This allows the attribution of any change in accuracy to the additional component. M1 removes the physics $( \lambda _ { \mathrm { P D E } } = \lambda _ { \mathrm { B C } } = 0 )$ reducing the model to a data-driven interpolant. M2 uses the full physics-informed loss but optimizes with Adam only, isolating the quasi-Newton (L-BFGS) contribution. M3 adds the L-BFGS phase, but the L-BFGS collocation points are drawn uniformly at random to isolate the RAR contribution. M4 corresponds to the proposed pipeline: Adam, RAR and L-BFGS in FP64. All four models share the same data, architecture, loss weights and number of collocation points. The model is trained on a $2 5 \times 2 5$ spatial grid $( \Delta x \approx 8 \mathrm { m m } )$ , which provides approximately two measurement points per A<sub>0</sub>-mode wavelength at $f _ { 0 } ,$ the Nyquist-limit data density. The sufficiency of this grid for the downstream localization task is established in the following density study. The training configuration is held constant across all variants (Table 2). All variants are evaluated on the same fixed test dataset.

Table 1 shows the ablation results. The full optimization pipeline (M4) reaches a test $L _ { 2 } ( \hat { w } )$ )-error of 0.104, representing reductions of 79 %, 64 % and 23 % relative to the Data-only (M1, 0.493), Adam-only (M2, 0.288) and no-RAR (M3, 0.135) runs, respectively. The M1 result indicates that the PDE residual provides dominant regularization. With 625 measurement points, the Data-only model is insufficient to constrain the ultrasonic wavefield at $f _ { 0 }$ without physical regularization. The final PDE residual of M2 remains nearly two orders of magnitude above the M3–M4 values $( \sim \mathrm { ~ \bar { ~ } } \times 1 0 ^ { - 6 } )$ due to the L-BFGS phase. The step from M2 to M3 halves the $L _ { 2 }$ error, identifying the L-BFGS phase as the main source of increased accuracy. The FP64 second-order refinement reaches minima that the first-order Adam does not approach within the same number of iterations. The final step, from M3 to M4, attributes a 23 % reduction to RAR. One plausible explanation is that L-BFGS collocation concentrates on high-residual regions, providing more informative curvature samples than uniform sampling, which allocates points to already-converged, low-residual regions.

Table 2: Training configuration held constant across all numerical and experimental tests unless stated otherwise.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Hidden layers / neurons</td><td> $5 \times 2 5 6$ </td></tr><tr><td>Base frequency ω0 (all layers)</td><td>30</td></tr><tr><td>Data supervision</td><td>transverse  only</td></tr><tr><td>Loss weights  $( \lambda _ { \mathrm { P D E } } , \lambda _ { \mathrm { B C } } , \lambda _ { \mathrm { d a t a } } )$ </td><td>(1.0, 10.0, 10.0)</td></tr><tr><td>Adam learning rate η</td><td>2 × 10−5</td></tr><tr><td>Adam iterations (FP32)</td><td>20,000</td></tr><tr><td>RAR: candidate set size  $N _ { \mathrm { c a n d } }$ </td><td>20,000</td></tr><tr><td>RAR: hard / random split</td><td>50 % / 50 %</td></tr><tr><td>L-BFGS max iterations (FP64)  $M _ { \mathrm { m a x } }$ </td><td>50,000</td></tr></table>

Figure 4 shows the convergence trajectories of the relative $L _ { 2 }$ error versus the number of iterations. During the Adam phase, M3 and M4 decrease at a comparable rate and plateau near $L _ { 2 } \approx 0 . 1 – 0 . 1 5$ . The Data-only model M1 plateaus near $L _ { 2 } \approx 0 . 3 – 0 . 5$ . M2 tracks the physicsinformed variants through the Adam phase. The transition to L-BFGS in M3 and M4 produces a significant drop in L<sub>2</sub>-error driven by quasi Newton optimization with FP64 precision. Figure 5 compares the reconstructed field

![](images/37ac05a0ee310f7a21234764be041334700f8eb2fed2c48539b0609c7741465f.jpg)  
Figure 4: Test $L _ { 2 }$ versus training iteration for all four ablation variants on the $2 5 ^ { 2 }$ grid.

against the FE reference for model M4 (full optimization strategy) and M1 (Data-only). M4 is able to reconstruct the emitted wavefield with low error. The real part of the transverse displacement and pointwise error are shown on a common, normalized scale by the amplitude scaling factor $W _ { s }$ . The pointwise normalized error, defined as $| \Delta w | / W _ { s }$ , remains below 0.1 over most of the domain. Minor locally elevated errors occur near the plate edges, where the free-boundary condition is most sensitive to the collocation density. The full optimization strategy (M4: Adam, RAR and L-BFGS) is adopted and held constant for all subsequent results. Only the training grid changes in the following sensor density study, which establishes the data requirement of the surrogate. All hyperparameters and loss weights are the same as those specified in Table 2.

## 5.2 Sensor Density Requirement

The architecture, loss formulation and optimization strategy defined in the Methodology and validated in the preceding subsection determine how the PI-SIREN is trained, but not how many measurement points the training requires for sufficient damage identification. In a practical ultrasonic inspection task, the scan time scales linearly with the number of sampled points, making the spatial sensor density one of the dominant constraints. This section identifies the coarsest uniform training grid for which the surrogate can reliably localize damage using FE data. This differs from the question of how small the global $L _ { 2 }$ error can be made for purely reconstruction purposes: a wavefield can be reconstructed with a moderate global error yet still preserve the local scattering signature of a defect. The PI-SIREN is retrained on a series of uniform grids from $1 5 \times 1 5$ to $4 0 \times 4 0$ points over the $2 0 0 \times 2 0 0$ mm plate domain, using the optimization strategy and hyperparameters of M4 fixed in Table 2. The grid spacing is $\dot { \Delta x } = L / ( N - 1 )$ and the sampling density is expressed as points per wavelength $( \mathrm { p p w } = \lambda / \Delta x )$ , where $\lambda \approx 1 6 . { \bar { 6 } }$ mm is the A -mode wavelength extracted from the FEM reference field at $f _ { 0 } = 8 9 . 9 \mathrm { k H z }$ . The Nyquist limit corresponds to 2 ppw, approximated by the $2 5 \times 2 5$ grid. Two metrics are tracked simultaneously: (i) the relative $L _ { 2 }$ error of the transverse displacement on a fixed test dataset, measuring global reconstruction accuracy and (ii) the localization error, measuring damage identification accuracy. The damaged reference fields are generated by removing all FEM elements within a line region of 8 mm at four (x, y) positions: (130, 50), (150, 50), (170, 100) and (30, 30) mm, to cover different scattering geometries relative to the centrally mounted actuator. Three random initialization seeds are used per grid to account for optimization variability. A localization is counted as successful when the localization error is below 2 mm. Table 3 and Figure 6 report

Table 3: Sensor density study: ppw and pristine-field reconstruction error $L _ { 2 } ( \hat { \overline { { w } } } )$ (mean ± std over three seeds) for each uniform training grid.
<table><tr><td>Grid ppw</td><td> $L _ { 2 } ( \hat { w } )$ </td></tr><tr><td> $1 5 \times 1 5$ </td><td>1.16  $0 . 2 6 2 \pm 0 . 0 2 0 8$ </td></tr><tr><td> $1 8 \times 1 8$ </td><td>1.41  $0 . 1 9 8 \pm 0 . 0 8 3 4$ </td></tr><tr><td> $2 0 \times 2 0$ </td><td>1.58  $0 . 2 6 5 \pm 0 . 0 0 6 2$ </td></tr><tr><td> $2 5 \times 2 5$   $3 0 \times 3 0$ </td><td>1.99  $0 . 1 0 3 \pm 0 . 0 0 3 2$ </td></tr><tr><td></td><td>2.41  $0 . 0 2 1 \pm 0 . 0 0 1 1$ </td></tr><tr><td> $3 5 \times 3 5$ </td><td>2.83  $0 . 0 1 1 \pm 0 . 0 0 0 1$ </td></tr><tr><td> $4 0 \times 4 0$ </td><td>3.24  $0 . 0 1 2 \pm 0 . 0 0 6 3$ </td></tr></table>

the reconstruction error as a function of sensor density. The $L _ { 2 } ( \hat { \overline { { w } } } )$ error decreases from approximately 26 % at 1.2 ppw to about 10 % at the Nyquist limit $( 2 5 ^ { 2 } , 2 . 0$ ppw) and continues to fall past Nyquist to $\sim 2 \%$ at 2.4 ppw $( 3 0 ^ { 2 } )$ ) and ∼ 1 % at 2.8 ppw (35<sup>2</sup>).

Beyond 2.8 ppw the error curve flattens: the $4 0 ^ { 2 }$ grid (3.2 ppw) yields $L _ { 2 } \approx 1 . 2 \%$ , a slightly increased error in comparison to the $3 5 ^ { 2 }$ result. The reconstruction error therefore saturates near 2.8 ppw and further densification yields no measurable gain in error reduction. Notably, this saturation point lies above the Nyquist density: the full-field reconstruction continues to improve for nearly one additional ppw beyond the density at which damage localization already becomes reliable (cf. Figure 7). For each trained model on FE simulation data, the localization error is determined from the final single-pass evaluation of the damage identification pipeline (phase (III) and (IV) in Figure 3). The pristine surrogate prediction is aligned to the damaged field via the complex scalar α (Eq. (31)), the scattered field is obtained by subtraction and the damage location is extracted from the peak of the Gaussian-smoothed intensity-divergence indicator ${ \mathcal { D } } = \left| { \mathrm { I m } } \{ { \overline { { W _ { \sigma } } } } \nabla ^ { 2 } W _ { \sigma } \} \right|$ <sup></sup>. The resulting damage localization error as a function of sensor density (ppw) is shown in Figure 7. At and above $2 5 ^ { 2 }$ points (2.0 ppw), the median error for all four positions remains below the 2 mm acceptance threshold across all seeds. Below this density, the median error becomes position-dependent and non-monotonic in grid size. For instance, the defect at (170, 100) mm stays below 2 mm at $1 5 ^ { 2 }$ (1.16 ppw) but exceeds it at $1 8 ^ { 2 }$ and $2 0 ^ { 2 }$ , while the defect at (30, 30) mm shows the reverse trend (Figure 7). The non-monotonic success pattern is physically consistent with spatial aliasing. A sub-Nyquist grid cannot uniquely represent the continuous wavefield: high spatial frequencies fold back into the baseband and the resulting aliased field depends on the relative phase between the sampling grid and the wavefront at each location. For a given defect, the aliased reconstruction may either preserve or suppress the local scattering contrast, depending on this phase relationship. Below the Nyquist limit, the localization outcome is governed by the geometric relationship between the defect position and the sampling grid rather than by the sampling density itself. A clearer picture emerges when reconstruction and localization accuracy are plotted against each other. Figure 8 shows the localization error as a function of the reconstruction $L _ { 2 } ( \hat { \overline { { w } } } )$ for the defect at (130, 50) mm, with color encoding the sensor density. A sharp bimodal separation is observed: runs with $\dot { L _ { 2 } } \lesssim 0 . 1 0$ yield localization errors below the 2 mm acceptance threshold, whereas runs with $L _ { 2 } \gtrsim 0 . 2 4$ miss by more than 25 mm. The vertical line at $L _ { 2 } = 0 . 1 0$ separates the two clusters. The few sub-Nyquist successes

![](images/382bf0acb7f3343d21e20c15b7d7a1a7ebbf53c089eda9de6354220b0c0409c8.jpg)

![](images/906de994fc756b9b9bc078cc231285d7958756d2cf22e9d1592b05333af71606.jpg)

![](images/3fd9f967fb48256aae1f38abfc724b6afc6d79fc264e9b9c5fc8051b4a0160e6.jpg)

![](images/7d04bb194822be773479cf3e0611179c2a744dffd063e8643ba5a28567911d42.jpg)

![](images/863ff5cc9b490162c92eba50223b8d6f78d484dcc01130a1cae60af386f0fd28.jpg)  
Figure 5: Pristine wavefield reconstruction on the $2 5 ^ { 2 }$ grid: FEM reference (left), M1 with $L _ { 2 } = 0 . 4 9$ (center) and M4 PI-SIREN prediction with $L _ { 2 } = 0 . 1 0$ (right). Top row: normalized transverse displacement $\mathrm { R e } ( \hat { \bar { w } } ) / W _ { s }$ , bottom row: pointwise error $| \Delta \hat { w } | / W _ { s }$

![](images/9250f23de4d08c8400c385d8b80c0fd27a8db4f6ff0679f384981d5e606635f4.jpg)  
Figure 6: Reconstruction $L _ { 2 } ( \hat { \bar { w } } )$ versus sensor density in ppw (mean ± std).

![](images/be4143cb0a8ae1b21f98dacde404d8a9cd0d9bd3af4ca35043823fdb408a3a70.jpg)  
Figure 7: Median localization error versus sensor density in ppw for four simulated damage positions with 2 mm acceptance threshold. Above the Nyquist limit (dashed line, 2 ppw) all positions converge $\mathrm { t o } < 1$ mm.

![](images/44887b3317eb8d11b0c6674773484d64b723bf8529903b576895321cefc04f72.jpg)  
Figure 8: Localization error versus reconstruction $L _ { 2 } ( \hat { \overline { { w } } } )$ with damage at (130, 50) mm. Successful localizations occur predominantly for $L _ { 2 } \lesssim 0 . 1 0$

correspond to model runs in which the aliasing pattern happened to preserve the scattered-field contrast, yielding a per-run reconstruction $L _ { 2 }$ error at the lower end of the inter-seed spread. Conversely, the sub-Nyquist failures are runs in which aliasing degraded the reconstruction accuracy, with an $L _ { 2 }$ error above the threshold. Sensor density is thus a measure of reconstruction accuracy. It determines whether the reconstruction error is low enough for the localization pipeline to extract the defect signature. The contrast between the two saturation points is the central finding of this section: localization becomes reliable at the Nyquist density $( 2 . 0 \mathrm { p p w } , 2 5 ^ { 2 } )$ , while the full-field reconstruction continues to improve until approximately 2.8 ppw $( 3 5 ^ { 2 } )$ . The damage-identification task therefore requires less reconstruction accuracy than the global $L _ { 2 }$ metric suggests, because the pipeline extracts a local scattering signature rather than reproducing the entire wavefield.

For the experimental demonstration, the nominal sensor density is set at the Nyquist limit (625 points, $2 5 ^ { 2 }$ 2.0 ppw), with spatial robustness ensured through stratified subsampling of a coarse scan rather than a rigid grid. Although the $L _ { \mathrm { 2 } } { \mathrm { - p p w } }$ threshold varies with defect size, excitation frequency and plate geometry, the diagnostic from numerical data is transferable to experiments.

## 5.3 Architectural Comparison

The proposed PI-SIREN (M4) is compared against two MLP baselines with hyperbolic tangent activation functions (M5 and M6) of comparable size. Both models are trained using the same $\bar { 2 5 } \times 2 5$ grid with the same collocation and boundary budgets, the same loss weights and the same three-phase optimization strategy of Algorithm 1. Three initialization seeds are used per model. Training parameters and results are summarized in Table 4. M5 is a fully connected network with tanh activation functions and no input embedding. M6 augments the input with a fixed random Fourier feature embedding [41] $\phi ( \zeta ) = [ \sin ( 2 \pi \mathbf { B } \zeta ) , \cos ( 2 \pi \mathbf { B } \zeta ) ]$ , where the $6 4 \times 2$ matrix B is drawn in two bands of 32 frequencies each from $\textstyle \mathcal { N } ( 0 , \sigma _ { f } ^ { 2 } )$ , with $\sigma _ { f }$ set to 2.0 and 6.0 cycles per computational unit. The embedding input $\zeta = ( \bar { x } , \bar { y } )$ is the normalized coordinate vector. The embedding itself yields 128 input channels. The network width is reduced from 256 to 241 neurons per layer, so that the number of trainable parameters $N _ { \mathrm { p a r a m s } }$ remains comparable to those of M4 and M5. Under the input normalization to $[ - 1 , 1 ] ^ { 2 }$ one computational unit corresponds to a physical length of $L / 2$ , so the plate spans two computational units. The A -mode wavelength of λ ≈ 16.6 mm corresponds to $L / ( 2 \lambda ) \approx 6$ cycles per computational unit, which the $\sigma _ { f } = 6 . 0$ band targets at one standard deviation, corresponding to $k _ { A _ { 0 } } \approx 0 . 3 8 \mathrm { r a d / m m }$ , while the $\sigma _ { f } = 2 . 0$ band covers the wave envelope. Both comparison models M5 and M6 use the learning rate that yields the lowest test error for the Adam phase in an independent grid search, yielding $\eta = 1 0 ^ { - 3 }$ . Additionally, M5 and M6 employ a 1,000-step warm-up during the Adam phase followed by a cosine decay to 10 % of the initial learning rate.

As shown in Table 4, the PI-SIREN reduces the reconstruction error by 58 % relative to the Fourier-feature baseline (M6) and by 79 % relative to the plain MLPtanh network (M5). The error of M5 matches that of the data-only PI-SIREN M1, which reaches 0.4932 with the physics entirely removed (Table 1). A tanh network equipped with the full MR residual thus performs on par with a SIREN trained without any physical constraint. M6 halves the error of M5, confirming that enriching the input spectrum is effective, yet its error still exceeds that of M4 by a factor of 2.4.

The trained models are evaluated as pristine baselines on the same four defect positions as shown in Figure 7. As in the sensor density study, the damage indicator is computed in a single pass on a dense grid of 10,000 sampled points without adaptive refinement. All M4 inferences localize the defects within the 2 mm threshold with a median localization error of 0.45 mm, whereas neither of the MLP baselines is able to localize a single defect. The failures of M5 are largely independent of the defect position, as a given seed returns the same predicted location for all four damage cases. The detected peak therefore responds to the systematic reconstruction error of the architecture rather than to a scattering signature that the residual field no longer contains. This is supported by the magnitude of the alignment factor in Eq. (31), which stays within 0.95–1.00 for M4 and 1.00–1.05 for M6 but drops to 0.83–0.93 for M5. This indicates a systematic, spatially varying amplitude bias that the single scalar alignment cannot remove. Unlike the sub-Nyquist failures identified in the sensor density study, which arise from spatial aliasing of a field that still resolves the $A _ { 0 }$ mode, these failures indicate a reconstruction that does not reproduce the $A _ { 0 }$ mode with sufficient accuracy. To identify the spectral origin of this failure, the reconstruction error is evaluated in two wavenumber bands around $k _ { A _ { 0 } } = 2 \pi / \lambda _ { A _ { 0 } } \mathrm { . }$ the $A _ { 0 }$ band at $0 . 8 5 k _ { A _ { 0 } } \leq | \mathbf { k } | \leq 1 . 1 5 k _ { A _ { 0 } } \mathrm { . }$ , which contains the propagating wave and the short-wavelength band at $1 . 1 \bar { 5 } k _ { A _ { 0 } } < | \mathbf { k } | \leq 2 . 0 k _ { A _ { 0 } }$ , which contains the higher spatial frequencies. For a band B the relative error is defined as

Table 4: Architecture comparison on the $2 5 \times 2 5$ grid. $L _ { 2 }$ values are mean ± std over three seeds. The localization error is the median over the 12 runs (three seeds × four defect positions).
<table><tr><td>Tag</td><td>Model</td><td>Activation</td><td>Input, [Layers × Neurons]</td><td> $N _ { \mathrm { p a r a m s } }$ </td><td>η</td><td> $L _ { 2 } ( \hat { \overline { { w } } } )$ </td><td>Loc.  $\| \Delta \pmb { x } _ { c } \|$  [mm]</td></tr><tr><td>M4</td><td>PI-SIREN</td><td> $\sin ( \omega _ { 0 } \xi )$ </td><td>coordinates,  $[ 5 \times 2 5 6 ]$ </td><td>265,478</td><td> $2 \times 1 0 ^ { - 5 }$ </td><td> $\mathbf { 0 . 1 0 3 \pm 0 . 0 0 3 }$ </td><td>0.45</td></tr><tr><td>M5</td><td>MLP-tanh</td><td>tanh</td><td>coordinates,  $[ 5 \times 2 5 6 ]$ </td><td>265,478</td><td> $1 0 ^ { - 3 }$ </td><td> $0 . 4 9 0 \pm 0 . 0 3 9$ </td><td>110.15</td></tr><tr><td>M6</td><td> $\mathrm { F F - M L P - t a n h }$ </td><td>tanh</td><td>Fourier Feature,  $[ 5 \times 2 4 1 ]$ </td><td>265,829</td><td> $1 0 ^ { - 3 }$ </td><td> $0 . 2 4 5 \pm 0 . 0 0 7$ </td><td>58.85</td></tr></table>

![](images/7d89e337325a5bfc68890cfc0a9895225f5b8eae0db1a9ad754c561629e3ca64.jpg)  
Figure 9: Relative band error $\varepsilon _ { B }$ against training iteration: (a) $A _ { 0 }$ band, 0.85 $k _ { A _ { 0 } } \leq | \mathbf { k } | \leq 1 . 1 5 k _ { A _ { 0 } }$ , and (b) short-wavelength band above the $A _ { 0 }$ band, $1 . 1 5 k _ { A _ { 0 } } < | \mathbf { k } | \leq 2 . 0 k _ { A _ { 0 } }$

$$
\varepsilon _ { \boldsymbol { \mathsf { B } } } = \frac { \left\| \hat { \bar { w } } _ { \mathrm { p r e d } } - \hat { \bar { w } } _ { \mathrm { r e f } } \right\| _ { \boldsymbol { \mathsf { B } } } } { \left\| \hat { \bar { w } } _ { \mathrm { r e f } } \right\| _ { \boldsymbol { \mathsf { B } } } } , \qquad \| \hat { f } \| _ { \boldsymbol { \mathsf { B } } } ^ { 2 } = \sum _ { \mathbf { k } \in \mathcal { B } } \vert \hat { f } ( \mathbf { k } ) \vert ^ { 2 } .\tag{39}
$$

Figure 9 shows the resulting wavenumber-band error ε<sub>B</sub> over the training iterations. After 5,000 Adam iterations, M5 has reduced the $A _ { 0 } .$ -band error from 1.00 to 0.97, while M4 reaches 0.31 and M6 0.58. The L-BFGS phase then reduces the A<sub>0</sub>-band error by 76 % for M4, against 42 % and 10 % for the M6 and M5 baselines. Above the $A _ { 0 }$ band, in the short-wavelength band $1 . 1 5 k _ { A _ { 0 } } < | \mathbf { k } | \leq$ $2 . 0 k _ { A _ { 0 } }$ , the error of M5 increases during training and is not recovered. The tanh architecture therefore generates spatial content that the reference solution does not contain. M6 performs better, but only M4 reduces the relative error in the short-wavelength band below 0.2.

To separate the bias of the network architecture from the influence of optimization and training, the neural tangent kernel (NTK) [42] is computed for each architecture on a $3 0 \times 3 0$ uniform test grid in FP64. The NTK is the Gram matrix of the parameter gradients of the network output at the query points, assembled from the real and imaginary parts. It measures how sensitively the predicted wavefield responds to a parameter update at each spatial location. Diagonalizing it yields eigenpairs $( \lambda _ { i } , \phi _ { i } )$ whose eigenvectors are spatial modes of the network output and whose eigenvalues act as the convergence rates of those modes. Under gradient flow, the error component along eigenvector i decays as $\exp ( - \lambda _ { i } t )$ , so a steep decay of the spectrum with wavenumber implies that high-wavenumber content is learned only very slowly [19, 41]. Figure 10 illustrates $\lambda _ { i } / \lambda _ { 1 }$ against the dominant wavenumber $k _ { i }$ at network initialization and after training. Each spectrum is normalized by its own largest eigenvalue. At initialization the plain MLP-tanh spectrum has collapsed to $\lesssim 1 0 ^ { - 1 1 }$ by $\dot { k } _ { i } \approx 0 . 2 \mathrm { r a d / m m }$ , below the $A _ { 0 }$ band, in which no eigenmode exceeds $\mathrm { i } 0 ^ { - 1 2 } \lambda _ { 1 }$ PI-SIREN M4 and the Fourier-embedded baseline M6 both retain $\lambda _ { i } / \lambda _ { 1 } \approx 1 0 ^ { - 3 } \mathrm { a t } k _ { A _ { 0 } } ,$ a gap of approximately nine orders of magnitude. Because this is a comparison at initialization, it is independent of the architecture-specific training parameter and isolates the contribution of the network architecture to spectral bias. Comparing the SIREN and Fourier-embedded baseline results at $k _ { A _ { 0 } }$ reveals that the remaining performance gap $( L _ { 2 }$ of 0.103 versus 0.245) is not due to a lack of sensitivity in the $A _ { 0 }$ band. Rather, it is due to the kernel shape beyond this band. The Fourierembedded kernel remains flat in the short wavelength band, while the PI-SIREN kernel decays. The flat tail of the Fourier-embedded kernel is consistent with its residual short-wavelength error in Figure 9 (b), while the collapsed M5 kernel explains why that network reduces neither band. Only the fully periodic SIREN drives the $A _ { 0 }$ band to convergence, while the Fourier-embedded tanh MLP retains a higher band error. The Fourier embedding of M6 acts on the input only. It removes the spectral bias of the tanh network but fixes a broadband, approximately stationary kernel with a bandwidth set by σ. In the SIREN architecture, every layer applies a periodic activation function. The resulting kernel is able to cover the $A _ { 0 }$ band, while also decaying in short-wavelength bands.

![](images/cbe985bf08bba91c590c7397d7619142b44decd263dd644a4143833b8038ba32.jpg)  
Figure 10: NTK eigenvalue spectrum $\lambda _ { i } / \lambda _ { 1 }$ against the dominant wavenumber $k _ { i }$ of each eigenvector at initialization and after training. The dashed line marks $k _ { A _ { 0 } } = 2 \pi / \lambda _ { A _ { 0 } } \approx 0 . 3 8 \mathrm { r a d / m m }$

## 5.4 Reconstruction and Damage Identification on Experimental Data

The proposed framework is applied to experimental wavefield measurements acquired with the optical microphone setup described in the “Experimental Dataset” section. The dominant response frequency is determined from the spectral peak of the full scan at 96.4 kHz. The sensor density study performed above established that reliable damage localization requires sampling at or above the Nyquist limit $( \approx \mathrm { ~ 2 ~ p p w ) }$ . For the experimental data, this density is achieved with 625 points obtained through stratified subsampling of a dense scan rather than a structured grid. The structured grids used in synthetic studies assume noise-free samples, whereas experimental measurements carry optical and coupling noise. A rigid grid may place points at locations of locally degraded signal quality without spatial redundancy. Stratified sampling distributes points across spatial cells while selecting the most reliable measurement per cell, improving robustness at equivalent nominal density.

The network architecture and three-phase optimization strategy (Adam, RAR, L-BFGS) follow Algorithm 1. The boundary-condition weight is reduced to $\lambda _ { \mathrm { B C } } = 1$ because the idealized free-edge condition is only approximately satisfied by the physically supported plate. The RAR hard-point fraction is reduced from 50% to 10%, since the highest-residual collocation points concentrate on the free-edge boundaries, where the model mismatch dominates (cf. Figure 11). The remaining 90% random points preserve uniform domain coverage. No synthetic noise is added during the Adam phase: the measurement noise in the experimental data already provides sufficient stochastic regularization. Figure 11 illustrates the comparison of measurement, the PI-SIREN prediction and the pointwise error. The model reproduces the dominant $A _ { \mathrm { 0 } } { \cdot } \mathrm { m o d e }$ interference pattern in the plate interior, including the radial wavefront from the central actuator. Reconstruction error concentrates near the boundaries (Figure 11, right): the global error is $L _ { 2 } ( \hat { \bar { w } } ) = 2 9 . 4 \%$ , while over the interior region (> 15 mm from the edge) it decreases to 18.5 % (Table 5). Both values exceed the synthetic result at the same density (10.3 % for $2 5 ^ { 2 }$ , Table 3). This increase is attributable to (i) optical measurement noise, (ii) boundary-condition differences between the idealized free-edge model and the physically supported plate, which dominate the edge region and (iii) model-form error, as the single-frequency MR residual does not capture all physical features of the experimentally measured field. However, the physics-informed loss provides effective regularization on this noisy data, consistent with the dominant regularization effect observed in the performed synthetic ablation study.

Table 5: Experimental pristine reconstruction error.
<table><tr><td>Region</td><td>Relative  $\overline { { L _ { 2 } ( \hat { w } ) } }$ </td></tr><tr><td>Full plate</td><td>29.4%</td></tr><tr><td>Interior (&gt; 15 mm)</td><td>18.5%</td></tr></table>

The trained surrogate serves as the pristine baseline in the damage identification pipeline described in Figure 3. The specimen contains a milled notch on the back side of the plate with dimensions $8 \times 2 \times 1 . 6$ mm (length × depth × in-plane thickness), centered at (150, 150) mm. The damaged wavefield is measured on the same plate and the pristine prediction is aligned to it using the complex scalar α (Eq. (31)), obtained via least-squares projection of the damaged field onto the pristine prediction. Here $| \alpha | =$ 0.87 and $\arg ( \alpha ) ~ = ~ - \bar { 1 1 } . 6 ^ { \circ }$ The scalar α reconciles global amplitude scaling and phase reference between the PI-SIREN baseline and optical measurement prior to subtraction.

The scattered field $\hat { w } _ { \mathrm { s c a t } } = \hat { w } _ { \mathrm { d a m } } - \alpha \hat { w } _ { \mathrm { p r i } }$ in Eq. (32) is interpolated onto a regular grid, Gaussian-smoothed $( \sigma = 1 . 5 \mathrm { m m } )$ and mapped through the indicator $\mathcal { D } =$ $\left| \operatorname { I m } \{ { \overline { { W _ { \sigma } } } } \nabla ^ { 2 } W _ { \sigma } \} \right|$ , which responds to localized scattering introduced by the defect. The proposed sequential pipeline (cf. Figure 3) is applied to the experimental data as illustrated in Figure 12. A coarse pass over the full grid identifies candidate regions, after which an adaptive refinement pass densely resamples all candidate regions and recomputes the indicator. The true-defect region emerges as the dominant response. The refined localization places the damage centroid at (148, 148) mm, corresponding to a distance of 2.63 mm from the true position $( \approx 1$ % of the plate diagonal). Localization sensitivity to postprocessing choices is assessed by varying the smoothing width σ over [1.0, 3.5] mm. The localization error varies by less than 0.35 mm across this range (2.51–2.83 mm, decreasing weakly with increasing $\sigma )$ , confirming that the result is robust to the kernel-width choice.

![](images/a9c6c65a32232aa3013a866e3edb0b523c730522bcf0d5659b1bdf4e4808fcb4.jpg)

![](images/f9027510cfb949aacc2c8685b7e5decea29c28b4c8bbb46311f9926fefb11325.jpg)

![](images/a203770c23d114a24460f0f5ecacc396da3774b7838bb7801d17881f54993285.jpg)  
Figure 11: Experimental pristine reconstruction: Experimental measurement (left), M4 PI-SIREN prediction (center), pointwise error $| \Delta \hat { w } | / W _ { s } $ (right). The dashed box indicates the interior region (> 15 mm) used for the interior $L _ { 2 }$ calculation.

Damage localization succeeds (2.63 mm) despite the reconstruction error being above the synthetic threshold $( L _ { 2 } \lesssim 0 . 1 0 ) \colon$ the indicator responds to localized scattering, whereas reconstruction error is dominated by smooth, global mismatch that the α-alignment and subtraction largely cancel. The task therefore tolerates larger reconstruction error on noisy experimental data than the noisefree synthetic threshold implies and that threshold should be regarded as a conservative design guideline. The adaptive, two-stage evaluation method restricts dense measurements to candidate regions identified by the coarse pass. This reduces the inspection scan from a full-plate, dense grid to approximately one-seventh of the measurement points. Combined with the inference-based pristine baseline, the complete inspection cycle therefore avoids the dense scanning that conventional baseline-subtraction methods require for both the reference and the damaged state.

## 5.5 Computational Cost

The framework separates computation into a one-time offline training phase and an online inspection phase (inference). To leverage accelerated automatic differentiation and XLA compilation, all model variants and ablation studies are implemented using JAX [43] library. All models and ablation variants are trained on a compute cluster containing a single NVIDIA A100 GPU (40 GB VRAM). As illustrated in Table 6, training time varies from 90–96 min, dominated by the double-precision L-BFGS refinement. Once trained, the frozen model evaluates 40,000 query points in < 1 ms, enabling baseline reconstruction by inference rather than physical measurement. This offline-online split is practical because the dominant cost in wavefield-based inspection is measurement, not computation. At a robotic acquisition rate of 0.1 s per point (travel, settling and measurement), a dense baseline scan of 40,000 points requires approx. 1.1 h of robotic acquisition, while the sparse scan used here (625 points) takes approx. 1 min. The two-stage adaptive damage identification pipeline applies the same principle at inspection, densifying only candidate regions rather than resampling the full plate.

Table 6: Computational cost overview. Values for M4 trained on synthetic (FEM) and experimental (EXP) data.
<table><tr><td>Quantity</td><td>FEM</td><td>EXP</td></tr><tr><td>M4 trainable parameters</td><td colspan="2">265,478</td></tr><tr><td>Training (one-time)</td><td>89.6 min</td><td>96.1 min</td></tr><tr><td>Adam / L-BFGS [min]</td><td> $2 2 . 6 / 6 6 . 9$ </td><td>18.7/77.5</td></tr><tr><td>Inference (40,000 pts)</td><td>0.77 ms</td><td>0.76 ms</td></tr></table>

## 6 Conclusion

This work introduces a frequency-domain PI-SIREN framework for UGW inspection in isotropic thin-walled structures. By embedding the coupled MR plate equations directly into a SIREN-based model architecture, the method enables high-fidelity reconstruction of steadystate wavefields without temporal discretization. The sinusoidal activation functions allow the network to represent the high-frequency content that limits conventional MLP-based PINNs with tanh activation functions at high ultrasonic frequencies. The hybrid optimization strategy combining first-order Adam, RAR and quasi-Newton L-BFGS fine-tuning drives the reconstruction to low errors. Numerical ablation studies on synthetic FE data established that the full optimization strategy (Adam, RAR, L-BFGS) reduces the reconstruction $L _ { 2 }$ error by 79 % relative to a Data-only baseline and by 23 % relative to the Adam + L-BFGS variant with uniform collocation. The RAR phase concentrates collocation points in the high-residual regions prior to quasi-Newton L-BFGS refinement. A sensor density study revealed that damage localization becomes reliable at the Nyquist sampling limit (2 ppw), while full-field reconstruction accuracy continues to improve until approximately 2.8 ppw.

![](images/c9acc7d82ef34ed246be7794b52e734fd9567f9b3a3814a472cef7373e952557.jpg)

![](images/b3e037aa620bc89faed99ebcc58be0c319f54a356b8cc75dcdaa4d4943d6237c.jpg)  
Figure 12: Demonstration of damage identification pipeline on experimental data. Left: pass-1 candidate regions (◦, #1–10) and adaptive resampled distribution. The true damage is among the identified regions. Right: Damage indicator D recomputed on the adaptively refined sampling, with the true-defect region showing the strongest response (2.63 mm localization error).

On experimental data acquired with a robotic optical microphone setup, the trained PI-SIREN surrogate serves as a robust pristine baseline despite higher global reconstruction error $( L _ { 2 } = 2 9 . 4 \%$ versus interior $L _ { 2 } \ : = \ : 1 8 . 5 \% )$ compared to the noise-free synthetic case at equivalent density. Although the physics-informed loss effectively regularized noisy measurements within the domain, the residual error is concentrated near the boundary edges. The implemented idealized free-edge boundary condition only approximates a physically supported plate, resulting in a less reliable reconstruction of the experimental data. Integrated into the two-stage damage identification pipeline (coarse candidate detection followed by adaptive resampling), the model localizes a back-side milled notch with an absolute error of 2.63 mm (≈ 1 % of the plate diagonal).

The proposed framework addresses three key limitations of existing SHM/NDT approaches: (i) the computational cost of FE-based baselines for repeated evaluations (ii) the data requirements and lack of interpretability of purely data-driven models and (iii) the spectral bias of conventional MLP-tanh-based PINNs at ultrasonic frequencies. This is achieved through periodic activations of the SIREN architecture [20]. The present study demonstrates the framework for a single plate configuration, material and excitation frequency. The practical advantage scales with the number of configurations covered. The one-time training cost is amortized across all subsequent inspections of the same or related structures. Operator-based architectures such as DeepONets [44, 45] represent a promising approach towards a single multi-configuration surrogate spanning multiple materials and geometries. Another approach is to directly utilize the PINN as inverse model for defect parameters. Rather than subtracting a pristine baseline, the damage location and extent would be included in the physics residual as trainable quantities. While this approach eliminates the need for pristine measurements of each specimen, it requires retraining for every new damaged dataset, shifting the cost from acquisition to computation. Further research involves applying the model framework to anisotropic and composite plate structures.

## Acknowledgments

The authors gratefully acknowledge the computational and data resources provided through the joint high-performance data an alytics (HPDA) project “terrabyte” of the German Aerospace Center (DLR) and the Leibniz Supercomputing Center (LRZ). During the preparation of this publication, Qwen3.8-27B and DeepL Write were utilized for language and grammar refinement and assistance in research data visualization. After utilizing these tools, the authors thoroughly reviewed and edited the content and accept full responsibility for the accuracy and integrity of the final publication.

## ORCID iDs

Lukas Jilke https://orcid.org/0000-0001-6879-3279   
Caspar Wasle https://orcid.org/0009-0003-7198-9186   
Norbert Hosters https://orcid.org/0000-0003-1174-4446   
Florian Raddatz https://orcid.org/0000-0002-0660-7650   
Marek Behr https://orcid.org/0000-0003-4257-8276   
Gerko Wende https://orcid.org/0000-0003-2368-2369

## CRediT authorship contribution

L.J. performed the roles of Conceptualization, Data curation, Formal analysis, Investigation, Methodology, Project administration, Software, Validation, Visualization, Funding acquisition, Writing – original draft, Writing – review & editing. C.W. performed the roles of Investigation, Writing – review & editing. N.H. performed the roles of Conceptualization, Supervision, Writing – review & editing. F.R. performed the roles of Funding acquisition, Writing – review & editing. M.B. performed the roles of Supervision, Writing – review & editing. G.W. performed the roles of Supervision, Writing – review & editing.

## Funding

The author(s) received no financial support for the research, authorship, and/or publication of this article.

## References

[1] L. Jilke, E. Arts, F. Raddatz, N. Hosters, M. Behr, and G. Wende, “Federated two-stage machine learning model for ultrasonic guided wave structural health monitoring of composite structures,” in Digital Twins, AI, and NDE for Industry Applications and Energy Systems 2025 (Proc. SPIE 13438, ed.), p. 19, SPIE, 17.03.2025 - 20.03.2025. doi: 10.1117/12.3051068.

[2] C. Willberg, S. Duczek, J. M. Vivar-Perez, and Z. A. B. Ahmad, “Simulation methods for guided wave-based structural health monitoring: A review,” Applied Mechanics Reviews, vol. 67, no. 1, 2015. doi: 10.1115/1.4029539.

[3] M. Abbas and M. Shafiee, “Structural health monitoring (shm) and determination of surface defects in large metallic structures using ultrasonic guided waves,” Sensors (Basel, Switzerland), vol. 18, no. 11, 2018. doi: 10.3 390/s18113958.

[4] L. Jilke, F. Raddatz, and G. Wende, “Investigation of degradation modeling for aircraft structures: A systematic literature review,” in AIAA AVIATION 2023 Forum, (Reston, Virginia), American Institute of Aeronautics and Astronautics, 2023. doi: 10.2514/6.2023-3550.

[5] S. Huang, S. Wang, W. Li, and Q. Wang, “Finite element simulation of ultrasonic guided waves,” in Electromagnetic Ultrasonic Guided Waves (S. Huang, S. Wang, W. Li, and Q. Wang, eds.), Springer Series in Measurement Science and Technology, pp. 237–270, Singapore: Springer Singapore, 2016. doi: 10.1007/978-981-10-0564-0\_6.

[6] V. Nerlikar, O. Mesnil, R. Miorelli, and O. D’Almeida, “Damage detection with ultrasonic guided waves using machine learning and aggregated baselines,” Structural Health Monitoring, vol. 23, no. 1, pp. 443–462, 2024. doi: 10.1177/14759217231169719.

[7] M. Rautela and S. Gopalakrishnan, “Ultrasonic guided wave based structural damage detection and localization using model assisted convolutional and recurrent neural networks,” Expert Systems with Applications, vol. 167, p. 114189, 2021. doi: 10.1016/j.eswa.2020.114189.

[8] Z. Yang, H. Yang, T. Tian, D. Deng, M. Hu, J. Ma, D. Gao, J. Zhang, S. Ma, L. Yang, H. Xu, and Z. Wu, “A review on guided-ultrasonic-wave-based structural health

monitoring: From fundamental theory to machine learning techniques,” Ultrasonics, vol. 133, p. 107014, 2023. doi: 10.1016/j.ultras.2023.107014.

[9] M. Raissi, P. Perdikaris, and G. E. Karniadakis, “Physicsinformed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,” Journal ofComputational Physics, vol. 378, pp. 686–707, 2019. doi: 10.1016/j.jcp. 2018.10.045.

[10] V. Trávníková, E. von Lieres, and M. Behr, “Quantifying data needs in surrogate modeling for flow fields in twodimensional stirred tanks with physics-informed neural networks,” Physics ofFluids, vol. 37, no. 11, 2025. doi: 10 .1063/5.0291054.

[11] A. I. F. Al-Adly and P. Kripakaran, “Physics-informed neural networks for structural health monitoring: a case study for kirchhoff–love plates,” Data-Centric Engineering, vol. 5, 2024. doi: 10.1017/dce.2024.4.

[12] W. Zhou and Y. F. Xu, “Damage identification for plate structures using physics-informed neural networks,” Mechanical Systems and Signal Processing, vol. 209, p. 111111, 2024. doi: 10.1016/j.ymssp.2024.111111.

[13] Y. Song, S. Shan, Y. Zhang, and L. Cheng, “Physicsguided neural network for structural health monitoring with lamb waves through boundary reflection elimination,” Structural Health Monitoring, vol. 25, no. 2, pp. 1219– 1236, 2026. doi: 10.1177/14759217241305050.

[14] K. Shukla, P. C. Di Leoni, J. Blackshire, D. Sparkman, and G. E. Karniadakis, “Physics-informed neural network for ultrasound nondestructive quantification of surface breaking cracks,” Journal ofNondestructive Evaluation, vol. 39, no. 3, 2020. doi: 10.1007/s10921-020-00705-1.

[15] W. Huang, K. Ying, X. Yan, and C. Ni, “Research on laser ultrasonic surface micro-crack detection based on physics-informed neural networks,” Journal of Physics: Conference Series, vol. 2966, no. 1, p. 012019, 2025. doi: 10.1088/1742-6596/2966/1/012019.

[16] N. N. Kulkarni and A. Sabato, “Full-field expansion and damage detection from sparse measurements using physics-informed variational autoencoders,” Structural Health Monitoring, vol. 25, no. 1, pp. 607–629, 2026. doi: 10.1177/14759217241289575.

[17] Z.-Q. J. Xu, Y. Zhang, and T. Luo, “Overview frequency principle/spectral bias in deep learning,” Communications on Applied Mathematics and Computation, vol. 7, no. 3, pp. 827–864, 2025. doi: 10.1007/s42967-024-00398-7.

[18] S. Khodakarami, V. Oommen, N. A. Daryakenari, M. Beekenkamp, and G. E. Karniadakis, “Spectral bias in physics-informed and operator learning: Analysis and mitigation guidelines,” Computer Methods in Applied Mechanics and Engineering, vol. 461, p. 119156, 2026. doi: 10.1016/j.cma.2026.119156.

[19] N. Rahaman, A. Baratin, D. Arpit, F. Draxler, M. Lin, F. Hamprecht, Y. Bengio, and A. Courville, “On the spectral bias of neural networks,” in Proceedings of the 36th International Conference on Machine Learning (K. Chaudhuri and R. Salakhutdinov, eds.), vol. 97 of Proceedings ofMachine Learning Research, pp. 5301–5310, PMLR, 2019.

[20] V. Sitzmann, J. Martel, A. Bergman, D. Lindell, and G. Wetzstein, “Implicit neural representations with periodic activation functions,” in Advances in Neural Information Processing Systems (H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, eds.), vol. 33, pp. 7462– 7473, Curran Associates, Inc, 2020.

[21] C. Song and Y. Wang, “Simulating seismic multifrequency wavefields with the fourier feature physics-informed neural network,” Geophysical Journal International, vol. 232, no. 3, pp. 1503–1514, 2022. doi: 10.1093/gji/ggac399.

[22] Y. Ding, C. Su, H. Miyake, and X. Li, Physics-Informed Neural Networks with Fourier Featuresfor Seismic Wavefield Simulation in Time-Domain Nonsmooth Complex Media. 2024. doi: 10.2139/ssrn.4902158.

[23] S. Schoder, “Physics-informed neural networks for modal wave field predictions in 3d room acoustics,” Applied Sciences, vol. 15, no. 2, p. 939, 2025. doi: 10.3390/app15020 939.

[24] Z. Chen, H. Chen, S. Yan, C. Lu, M. Yuan, L. Ding, X. Xu, X. Yan, M. Lu, and Y. Chen, “Multi-scale physicsinformed inversion of densely distributed near-surface defects using laser ultrasonics,” Optics & Laser Technology, vol. 203, p. 115534, 2026. doi: 10.1016/j.optlastec.2026.1 15534.

[25] L. Yang, P. Liu, K. Yi, T. van Huynh, H. Byun, J. Kim, H. Hwang, and H. Sohn, “Defect-parameterized physicsinformed neural network for forward and inverse modeling of laser ultrasonic wavefield,” Ultrasonics, vol. 161, p. 107952, 2026. doi: 10.1016/j.ultras.2026.107952.

[26] S. Wang, S. Sankaran, and P. Perdikaris, “Respecting causality for training physics-informed neural networks,” Computer Methods in Applied Mechanics and Engineering, vol. 421, p. 116813, 2024. doi: 10.1016/j.cma.2024.1 16813.

[27] L. Jilke, M. Schubert, N. Hosters, F. Raddatz, M. Behr, and G. Wende, “Physics-informed neural networks for ultrasonic lamb wave reconstruction in non-destructive testing,” in Proceedings of the 35th Congress of the International Council ofthe Aeronautical Sciences (ICAS), 2026. doi: 10.71945/icas2026\_0573.

[28] R. D. Mindlin, “Influence of rotatory inertia and shear on flexural motions of isotropic, elastic plates,” Journal of Applied Mechanics, vol. 18, no. 1, pp. 31–38, 1951. doi: 10.1115/1.4010217.

[29] E. Reissner, “The effect of transverse shear deformation on the bending of elastic plates,” Journal of Applied Mechanics, vol. 12, no. 2, pp. A69–A77, 1945. doi: 10.1115/1.4009435.

[30] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” 3rd International Conferencefor Learning Representations, 2015. doi: 10.48550/arXiv.1412.6980.

[31] L. Lu, X. Meng, Z. Mao, and G. E. Karniadakis, “Deepxde: A deep learning library for solving differential equations,” SIAM Review, vol. 63, no. 1, pp. 208–228, 2021. doi: 10.1 137/19M1274067.

[32] D. C. Liu and J. Nocedal, “On the limited memory bfgs method for large scale optimization,” Mathematical Programming, vol. 45, no. 1-3, pp. 503–528, 1989. doi: 10.1007/BF01589116.

[33] R. Bollapragada, J. Nocedal, D. Mudigere, H.-J. Shi, and P. T. P. Tang, “A progressive batching l-bfgs method for machine learning,” in Proceedings ofthe 35th International Conference on Machine Learning (J. Dy and A. Krause, eds.), vol. 80 of Proceedings of Machine Learning Research, pp. 620–629, PMLR, 2018.

[34] A. Rytter, Vibrational Based Inspection ofCivil Engineering Structures. PhD thesis, Dept. of Building Technology and Structural Engineering, Aalborg University, Denmark, 1993.

[35] M. Lints and M. Ratassepp, “Accuracy of the mindlin model in a0 lamb mode scattering from partial throughthickness damage,” Applied Sciences, vol. 14, no. 20, p. 9351, 2024. doi: 10.3390/app14209351.

[36] J. L. Rose, Ultrasonic guided waves in solid media. New York, NY: Cambridge University Press, 2014. doi: 10.101 7/CBO9781107273610.

[37] E. Kiyani, K. Shukla, J. F. Urbán, J. Darbon, and G. E. Karniadakis, “Optimizing the optimizer for physics-informed neural networks and kolmogorov-arnold networks,” Computer Methods in Applied Mechanics and Engineering, vol. 446, p. 118308, 2025. doi: 10.1016/j.cma.2025.11830 8.

[38] C. Xu, D. Liu, A. Nassereldine, and J. Xiong, “Fp64 is all you need: Rethinking failure modes in physics-informed neural networks,” in Advances in Neural Information Processing Systems 38, (San Diego, California, USA), pp. 158613–158634, 2025. doi: 10.52202/085713-4785.

[39] Dassault Systèmes Simulia Corp., “Abaqus/explicit and abaqus/cae,” 2024.

[40] S. Mann, “Cubic precision clough-tocher interpolation,” Computer Aided Geometric Design, vol. 16, no. 2, pp. 85– 88, 1999. doi: 10.1016/S0167-8396(98)00038-7.

[41] M. Tancik, P. Srinivasan, B. Mildenhall, S. Fridovich-Keil, N. Raghavan, U. Singhal, R. Ramamoorthi, J. Barron, and R. Ng, “Fourier features let networks learn high frequency functions in low dimensional domains,” in Advances in Neural Information Processing Systems (NeurIPS 2020), vol. 33, pp. 7537–7547, Curran Associates, Inc., 2020.

[42] A. Jacot, F. Gabriel, and C. Hongler, “Neural tangent kernel: Convergence and generalization in neural networks,” in Advances in Neural Information Processing Systems (NeurIPS 2018), vol. 31, Curran Associates, Inc., 2018.

[43] J. Bradbury, R. Frostig, P. Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang, “Jax: composable machine learning and scientific computing,” 2018.

[44] L. Lu, P. Jin, G. Pang, Z. Zhang, and G. E. Karniadakis, “Learning nonlinear operators via deeponet based on the universal approximation theorem of operators,” Nature Machine Intelligence, vol. 3, no. 3, pp. 218–229, 2021. doi: 10.1038/s42256-021-00302-5.

[45] S. Goswami, A. Bora, Y. Yu, and G. E. Karniadakis, “Physics-informed deep neural operator networks,” in Machine Learning in Modeling and Simulation (T. Rabczuk and K.-J. Bathe, eds.), Computational Methods in Engineering & the Sciences, pp. 219–254, Cham: Springer International Publishing, 2023. doi: 10.1007/978-3-031-3 6644-4\_6.