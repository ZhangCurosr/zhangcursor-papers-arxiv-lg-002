# PINNing the pion: conformal deep learning for $F _ { \pi } ( s )$ and the $( g - 2 ) _ { \mu }$ hadronic contribution

Mayank Goel,<sup>1,</sup> <sup>∗</sup> Subhadip Mitra,<sup>1,</sup> <sup>2,</sup> <sup>†</sup> and Monalisa Patra<sup>3,</sup> <sup>‡</sup>

<sup>1</sup>Center for Computational Natural Sciences and Bioinformatics,

International Institute of Information Technology, Hyderabad 500 032, India

<sup>2</sup>Center for Quantum Science and Technology, International Institute of Information Technology, Hyderabad 500 032, India <sup>3</sup>iHub-Data, International Institute of Information Technology, Hyderabad 500 032, India

Extracting the pion electromagnetic form factor $F _ { \pi } ( s )$ through phenomenological curve-fitting models introduces model dependence, unphysical artefacts, and kinematic inconsistencies. We introduce a Physics-Informed Neural Network (PINN) embedded in a conformal z-plane that constructs $F _ { \pi } ( s )$ directly from first principles across spacelike and timelike domains: charge normalisation and Schwarz reflection are enforced by construction, while Cauchy-Riemann analyticity, dispersion relations, Watson’s theorem, and perturbative QCD asymptotics enter through the loss functional. Thus, the fundamental S-matrix principles dictate the form factor’s behaviour while data act as constraints. Mapping the cut complex plane onto the unit disk bounds the Hessian norm and prevents Neural Tangent Kernel spectral starvation, two known failure modes of deep-learning optimisation. Besides $e ^ { + } e ^ { - }$ scattering data, we also incorporate τ-decay data through a switch that isolates the pure isovector form factor natively, bypassing model-dependent isospin-breaking pre-corrections. The network organically yields an interior zero-free form factor, while the framework tests experimental tensions around the $\rho ( 7 7 0 )$ peak against analyticity and dispersion constraints. We obtain model-independent estimates of the pion charge radius, $\langle r _ { \pi } ^ { 2 } \rangle = 0 . 4 3 5 \pm 0 . 0 0 8 _ { \mathrm { s t a t } } \pm 0 . 0 0 7 _ { \mathrm { c a l i } } \ \mathrm { f m } ^ { 2 }$ , the second-sheet pole parameters, $m _ { \rho } ^ { \mathrm { p o l e } } = 7 6 1 . 7 2 \pm 1 . 0 4$ MeV and $\Gamma _ { \rho } ^ { \mathrm { p o l e } } = 1 3 5 . 9 9 \pm 1 . 2 0 \ \mathrm { M e V } ,$ and the two-pion contribution to the muon anomalous magnetic moment, $a _ { \mu } ^ { \dot { \pi } \pi } = ( 5 0 6 . 4 8 \pm 2 . 0 2 _ { \mathrm { s t a t } } \pm 1 . 7 0 _ { \mathrm { c a l i } } ) \times 1 0 ^ { - 1 0 }$

## I. INTRODUCTION

The pion electromagnetic form factor $F _ { \pi } ( q ^ { 2 } )$ is a fundamental quantity in hadron physics. It maps the spatial distribution of valence quarks within the lightest hadron. As the pion is the pseudo-Goldstone boson of broken chiral symmetry, its structure provides a direct window into the transition between perturbative QCD at high energies and the QCD vacuum. It also sits at the heart of the muon $g - 2$ anomaly because the dominant twopion hadronic vacuum polarisation (HVP) contribution to $a _ { \mu } = ( g - 2 ) _ { \mu } / 2$ is dictated by an integral over $| F _ { \pi } ( q ^ { 2 } ) | ^ { 2 }$ The historical 3 − 4σ tension with the Standard Model (SM) is fundamentally tied to uncertainties in this form factor [1, 2]. However, recent high-precision lattice QCD calculations [3] and the CMD-3 cross-section measurement [4] have upended this landscape by suggesting a higher HVP value that largely defuses the $g - 2$ discrepancy. But in the process, they have triggered a new crisis: a contradiction between theory and the traditional data-driven consensus. Resolving this internal HVP puzzle now hinges entirely on how reliably $F _ { \pi } ( q ^ { 2 } )$ can be continuously mapped across both the spacelike and timelike domains.

Several theoretical approaches to study $F _ { \pi } ( q ^ { 2 } )$ exist in the literature, but these usually struggle to bridge the domains while satisfying all constraints simultaneously. Chiral perturbation theory (ChPT) is precise near $q ^ { 2 }$ 0 [5] but breaks down approaching the ρ(770) resonance region because it lacks the vector meson degrees of freedom. Vector meson dominance (VMD) models [6] capture this resonance peak but violate the $1 / q ^ { 2 }$ power-law scaling at high energies required by perturbative quantum chromodynamics (pQCD) without any connection to underlying quark-gluon dynamics. Dispersive frameworks based on the Omnès representation [7] rely on experimental inputs for ππ scattering phase shifts obtained via the Roy equations [8, 9]. These methods accumulate systematic uncertainties, as Leplumey and Stofer recently identified, depending on how they treat complex zeros in the form factor [10]. Imposing a zero-free condition reduces these systematic errors but rigidifies the model, exposing severe contradictions between the CMD-3 [4] and BaBar [11, 12] datasets. Lattice QCD offers a first-principles alternative but remains confined to the Euclidean region, where continuing these numerical results to Minkowski space remains limited by the incompleteness problem [13].

Traditional fitting procedures assume some physicsmotivated functional form of the form factor and thus suffer from modelling bias. Modern machine learning and data-driven interpolations [14] can bypass explicit model parameterisation, but they do not guarantee global Smatrix consistency and are susceptible to unphysical numerical artefacts. Moreover, as we demonstrate here, standard neural network fits face a structural obstruction: training directly with the physical momentumtransfer variable s $( \equiv \dot { q } ^ { 2 } )$ causes gradient descent to fail beyond a region. The unbounded kinematic domain forces the minimum eigenvalue of the neural tangent kernel (NTK), the functional matrix governing the optimisation dynamics in the network’s parameter space, to collapse to zero. This structural pathology causes spectral starvation, meaning the network fails to learn the form factor in the $\mathrm { h i g h } { - \cdot } q ^ { 2 }$ region, rendering the asymptotic high-energy constraints unlearnable regardless of the network architecture or training time.

The resolution of this inherent obstruction reveals a correspondence between S-matrix theory and deep learning optimisation. The conformal mapping z(s) used in dispersive physics (see, e.g., [15]) to compactify the cut complex plane and enforce analyticity matches the transformation required to stabilise gradient descent and preserve the NTK expressivity. Compactifying the physical domain into the unit disk simultaneously satisfies the topological demands of the S-matrix and eliminates the coordinate-induced gradient starvation in the optimisation space. The same geometry addresses the physics and the optimisation.

We use this geometric transformation to build a physics-informed neural network (PINN) that operates natively in the conformal $\mathcal { Z }$ plane and constructs the form factor from the data by relying on the S-matrix principles rather than simple data fitting. Our architecture structurally enforces normalisation and the Schwarz reflection principle, while loss functions enforce analyticity, the dispersion relation, Watson’s theorem, and perturbative QCD asymptotics. We don’t fix any specific functional form for the imaginary part Im $F _ { \pi }$ . When trained on spacelike and timelike data simultaneously, the reconstructed form factor emerges free of complex zeros in the interior. Because this zero-free structure is organic rather than imposed, the framework retains the flexibility to accommodate tensions in the data. The result is a stable, model-independent reconstruction of $F _ { \pi } ( q ^ { 2 } )$ across all kinematic regions, providing independent evidence for a zero-free form factor. From the numerical form factor, we estimate the pion charge radius, unphysical second-sheet resonance pole parameters, and the twopion HVP contribution to $a _ { \mu }$

The rest of this paper is organised as follows. Section II presents the theoretical background and the fundamental principles governing the form factor. Section III establishes the conformal z mapping, outlines the PINN architecture, including the details of the composite loss functional that enforces physics constraints, and explains how conformal preconditioning resolves deep-learning optimisation pathologies. Section IV describes the experimental datasets covering spacelike and timelike channels used to train and constrain the model. In Section V, we present our primary results, including the extracted pion form factor, the network’s role as an analytical arbiter for dataset tensions around the $\rho ( 7 7 0 )$ peak, and precise extractions of the pion charge radius, resonance pole parameters, and $a _ { \mu } ^ { \pi \pi }$ . Section VI provides a compact conformal Padé approximant that compresses the continuous network prediction into a usable closed-form expression for phenomenological applications. Finally, Section VII summarises our findings, discusses the framework’s main limitations, and outlines potential future directions.

## II. THE THEORETICAL BACKGROUND

## A. The low-energy expansion

The pion form factor can be formally defined via the matrix element of the electromagnetic current operator $J _ { \mathrm { e m } } ^ { \mu }$ evaluated between charged pion states:

$$
\langle \pi ^ { + } ( p ^ { \prime } ) | J _ { \mathrm { e m } } ^ { \mu } ( 0 ) | \pi ^ { + } ( p ) \rangle = ( p + p ^ { \prime } ) ^ { \mu } F _ { \pi } ( s ) ,\tag{1}
$$

where $s \equiv q ^ { 2 } = ( p ^ { \prime } - p ) ^ { 2 }$ is the squared momentum transfer. The conservation of the vector current associated with the unbroken $S U ( 2 ) _ { V }$ isospin subgroup of the global chiral symmetry demands the charge normalisation $F _ { \pi } ( 0 ) = 1$ . The low-energy expansion of the form factor serves as a fundamental probe of the QCD vacuum:

$$
F _ { \pi } ( s ) = 1 + { \frac { 1 } { 3 ! } } \langle r _ { \pi } ^ { 2 } \rangle s + { \frac { 1 } { 5 ! } } \langle r _ { \pi } ^ { 4 } \rangle s ^ { 2 } + { \frac { 1 } { 7 ! } } \langle r _ { \pi } ^ { 6 } \rangle s ^ { 3 } + { \mathcal O } ( s ^ { 4 } ) ,\tag{2}
$$

where $\langle r _ { \pi } ^ { 2 } \rangle$ is the charge radius of the pion, and the rest captures the higher-order curvature. Any physically admissible parameterisation must reproduce both the unit charge normalisation and the observed slope at the origin.

## B. Analyticity and the S matrix

Causality and the analytic structure of the S matrix mandate that $F _ { \pi } ( s )$ is a holomorphic function throughout the complex s plane except for the branch cuts and branch points. Because physical signals cannot propagate outside the forward light cone, the commutators of local currents vanish for spacelike separations. In momentum space, this microcausality demands that the singularities of the form factor are strictly confined to regions corresponding to physical intermediate hadronic states. The lightest such state in the isovector vector channel is the two-pion state, producing a branch cut along the positive real axis beginning at $s _ { \mathrm { t h } } = 4 m _ { \pi } ^ { 2 }$

The Schwarz reflection principle, $F _ { \pi } ( s ^ { * } ) = F _ { \pi } ^ { * } ( s )$ follows from analyticity. It implies that $F _ { \pi } ( s )$ is real on the real axis wherever the function is analytic, in particular in the spacelike region and the unphysical region $0 < s < 4 m _ { \pi } ^ { 2 }$ . On the physical cut, the upper and lower rim values are related by

$$
F _ { \pi } ( s - i 0 ) = F _ { \pi } ^ { * } ( s + i 0 ) , \quad s > 4 m _ { \pi } ^ { 2 } ,\tag{3}
$$

and the discontinuity is therefore

$$
\mathrm { D i s c } F _ { \pi } ( s ) = F _ { \pi } ( s + i 0 ) - F _ { \pi } ( s - i 0 ) = 2 i \mathrm { I m } F _ { \pi } ( s + i 0 ) .\tag{4}
$$

Therefore, the absorptive part of the form factor completely determines the discontinuity across the physical branch cut.

## C. Isospin breaking, the inelastic threshold, and Watson’s theorem

Below the inelastic four-pion threshold $( s _ { \mathrm { t h } } ^ { \mathrm { i n e l a s t i c } } = 1 6 m _ { \pi } ^ { 2 }$ ≈ $0 . 3 1 \mathrm { G e V } ^ { 2 } )$ , the final state in $e ^ { + } e ^ { - }$ scattering is dominated by the elastic ππ channel. Since the two-pion state is an isovector state, unitarity implies that Watson’s finalstate interaction theorem is applicable in this region. The theorem fixes the phase of the pion form factor to the isospin-1, P-wave ππ scattering phase shift,

$$
\arg \left[ F _ { \pi } ^ { I = 1 } ( s + i 0 ) \right] = \delta _ { 1 } ^ { 1 } ( s ) , \quad 4 m _ { \pi } ^ { 2 } < s < s _ { \mathrm { t h } } ^ { \mathrm { i n e l a s t i c } } .\tag{5}
$$

Equivalently, for isovector dominance,

$$
\mathrm { I m } F _ { \pi } ( s ) = \mathrm { R e } F _ { \pi } ( s ) \tan \delta _ { 1 } ^ { 1 } ( s ) .\tag{6}
$$

This condition strongly constrains the phase of $F _ { \pi } ( s )$

In reality, isovector dominance continues much beyond $1 6 m _ { \pi } ^ { 2 }$ . In particular, the rapid increase of $\delta _ { 1 } ^ { 1 } ( s )$ through approximately $\pi / 2$ in the $\rho ( 7 7 0 )$ region forces the form factor to exhibit the characteristic resonant enhancement observed in the timelike cross section. Beyond the ρ resonance, electromagnetic interactions induce small isospinbreaking mixing between the $I = 1 ~ ( \rho )$ and $I = 0$ (mostly $\omega ,$ , but also the heavier φ) states that manifests as minor (compared to the isovector resonance) dip-bump interference patterns in the timelike cross-section data. In the $e ^ { + } e ^ { - } \to \pi ^ { + } \pi ^ { - }$ process, this isospin breaking is quantified by $\rho { - } \omega$ mixing. The physical electromagnetic form factor is parameterised as

$$
F _ { \pi } ^ { e ^ { + } e ^ { - } } ( s ) = F _ { \pi } ^ { I = 1 } ( s ) \left( 1 + \frac { \alpha _ { \rho \to \omega } e ^ { i \phi _ { \omega } } m _ { \omega } ^ { 2 } } { m _ { \omega } ^ { 2 } - s - i m _ { \omega } \Gamma _ { \omega } } \right) ,\tag{7}
$$

where $\alpha _ { \rho - \omega }$ controls the mixing strength and $\phi _ { \omega } \approx 1 . 7 4$ rad is the Orsay phase, producing the characteristic interference near s ≈ $m _ { \omega } ^ { 2 }$ . In contrast, τ-decay data measure $F _ { \pi } ^ { I = 1 } ( s )$ directly, free of this contamination up to calculable isospin-breaking corrections.

The first physically dominant inelastic channel breaking the elastic phase-locking is $\omega \pi ^ { 0 }$ production. This shifts the efective phenomenological inelastic threshold to $s _ { \mathrm { e f f . } } ^ { \mathrm { i n e l a s t i c } } = ( m _ { \omega } + \bar { m _ { \pi } } ) ^ { 2 } \approx 0 . 8 4 \mathrm { G e V ^ { 2 } } [ 1 6 ]$ . However, since even this efect is small, the efective inelastic threshold is generally set at the $K \overline { { K } }$ threshold at $4 m _ { K } ^ { 2 } \approx 1 ~ \mathrm { G e V ^ { 2 } ~ [ 1 7 ] }$ We therefore consider $s _ { \mathrm { e f f . } } ^ { \mathrm { i n e l a s t i c } } \approx 1 ~ \mathrm { G e } \ddot { \ V } ^ { 2 }$ as the upper boundary of the elastic Watson constraint in Eq. (6) in the loss functional of our neural network.

## D. Asymptotic pQCD scaling

At large spacelike momentum transfers, perturbative QCD (pQCD) imposes an absolute asymptotic boundary. In the deep Euclidean region $( s \to - \infty )$ , the virtual photon resolves the pion’s partonic structure. Here, the Brodsky-Farrar quark-counting rules [18] dictate that the form factor must decouple, scaling as $F _ { \pi } ( s ) \sim 1 / s$ up to logarithmic corrections from the running strong coupling and the pion distribution amplitude [19]:

$$
F _ { \pi } ( - Q ^ { 2 } ) \sim \frac { \mathcal { A } } { Q ^ { 2 } } , \quad Q ^ { 2 } = - s \to \infty ,\tag{8}
$$

where

$$
\mathcal { A } \approx 8 \pi f _ { \pi } ^ { 2 } \alpha _ { S } ( \mu _ { R } ^ { 2 } ) \Bigg [ 1 + \frac { \alpha _ { S } ( \mu _ { R } ^ { 2 } ) } { \pi } \Big ( \frac { \beta _ { 0 } } { 4 } \ln \Big ( \frac { \mu _ { R } ^ { 2 } } { Q ^ { 2 } } \Big ) + 6 . 4 1 \Big ) \Bigg ] ,
$$

at the next-to-leading order (NLO) in pQCD with $\mu _ { R } ^ { 2 }$ ≈ $Q ^ { 2 } / 2 1$ being the renormalisation scale (see Appendix B), and $f _ { \pi } = 0 . 1 3 1$ GeV is the pion decay constant. This condition forbids any parametrisation that grows unboundedly at high energies, ensuring consistency with the short-distance limits of QCD.

Traditionally, accommodating all of unit normalisation, phase tracking, analytic continuity, and asymptotic scaling within a single phenomenological model has proven remarkably dificult. However, our framework does not need any form factor model to enforce these S-matrix and QCD mandates; instead, these enter via a loss functional that governs the neural network’s optimisation.

## III. NEURAL NETWORK ARCHITECTURE: STABILITY AND EXPRESSIVITY

## A. Using conformal mapping to bypass geometric instability

For a neural network, the primary numerical challenge comes from unbounded physical kinematic variables, which may contain several nontrivial boundary components associated with diferent thresholds. Hadronic form factors possess several threshold singularities associated with distinct physical channels. These thresholds generate branch points, and the corresponding branch cuts determine the boundary structure of the physical sheet. Since the physical momentum of newly created massive particles scales as $\sqrt { s - s _ { \mathrm { t h } } }$ each independent threshold locally splits the complex s-plane into a twosheeted Riemann surface. Furthermore, if massless exchanges are present, they introduce logarithmic singularities that splinter the domain into infinitely many sheets. To address this, we first replace this complicated physical geometry with a bounded canonical domain:

$$
\mathcal { S } = \hat { \mathbb { C } } \backslash \bigcup _ { i = 1 } ^ { N _ { \Gamma } } \Gamma _ { i } ,\tag{9}
$$

where ${ \hat { \mathbb { C } } } = \mathbb { C } \cup \{ \infty \}$ is the Riemann sphere and Γ ’s are disjoint branch cuts on the physical sheet. As the cuts define non-degenerate boundary components, $\mathcal { S }$ is a finitely connected planar domain. However, as we discuss below, even the canonical physical domain is not ideal for training our neural network.

For optimisers such as Adam or L-BFGS, optimisation is most stable when the local loss landscape is close to isotropic, i.e., approximately bowl-shaped with comparable curvature in all directions. This is hard to achieve everywhere in the physical space. However, a transformation to a conformal coordinate can act as a geometric preconditioner and remove the coordinate-induced source of large curvature associated with the unbounded physical momentum-transfer variable. To see that, let us first define the loss function. Let $\begin{array} { r l } { f ( x ; \theta ) = \sum _ { k = 1 } ^ { K } \nu _ { k } \sigma ( a _ { k } x + b _ { k } ) } & { { } } \end{array}$ be a single-hidden-layer neural network with K neurons, $P =$ $3 K$ parameters: $\mathbf { \bar { \theta } } = \{ \nu _ { k } , a _ { k } , b _ { k } \} _ { k = 1 } ^ { K }$ (where x denotes the input coordinate), and σ a smooth and bounded activation function $( \mathrm { i } . { \mathrm e } . , | \sigma ( u ) | \leqslant C _ { \sigma }$ and $| \sigma ^ { \prime } ( u ) | \leqslant C _ { \sigma ^ { \prime } } , \forall u \in \mathbb { R } )$ The empirical mean-squared-error loss over N collocation points is given as

$$
\mathcal { L } ( \theta ) = \frac { 1 } { 2 N } \sum _ { i = 1 } ^ { N } \left( f ( x _ { i } ; \theta ) - y _ { i } \right) ^ { 2 } = \frac { 1 } { 2 N } \sum _ { i = 1 } ^ { N } r _ { i } ^ { 2 } ( \theta ) ,\tag{10}
$$

where $r _ { i } ( \theta ) = f ( x _ { i } ; \theta ) - y _ { i }$ denotes the ith residual. The Hessian matrix of the loss, $H \in \mathbb { R } ^ { P \times P }$ , is the collection of

![](images/dc37744a879e2065c0b84629f84e6d5af30baeea8e76c8159c94605e893da358.jpg)  
FIG. 1. The conformal map $z ( s )$ compactifying the physical $\mathcal { S }$ plane onto the unit disk. The physical branch cut $[ 4 m _ { \pi } ^ { 2 } , \infty )$ maps to the unit circle boundary. The $\rho ( 7 7 0 )$ resonance pole lies on the second Riemann sheet, outside the unit disk at $z _ { \rho ( 7 7 0 ) } \approx 0 . 8 - 0 . 7 i .$ ensuring $F _ { \pi }$ is holomorphic throughout the interior $| z | < 1$

all second derivatives, given as

$$
\begin{array} { r l r } {  { H _ { j k } = \partial _ { \theta _ { j } } \partial _ { \theta _ { k } } \mathcal { L } } } \\ & { } & { = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } [ \partial _ { \theta _ { j } } f ( x _ { i } ; \theta ) \partial _ { \theta _ { k } } f ( x _ { i } ; \theta ) + r _ { i } ( \theta ) \partial _ { \theta _ { j } } \partial _ { \theta _ { k } } f ( x _ { i } ; \theta ) ] } \\ & { } & { = \frac { 1 } { N } [ ( J ^ { T } J ) _ { j k } + \sum _ { i = 1 } ^ { N } r _ { i } ( \theta ) \partial _ { \theta _ { j } } \partial _ { \theta _ { k } } f ( x _ { i } ; \theta ) ] , \qquad ( 1 1 ; } \end{array}
$$

where $J \in \mathbb { R } ^ { N \times P }$ is the Jacobian, $J _ { i j } = \partial _ { \theta _ { i } } f ( x _ { i } ; \theta )$

Since near a small-residual solution $( r _ { i } ( \theta ) \approx 0 )$ the Hessian is approximated by the Gauss-Newton matrix, $H _ { \mathrm { G N } } = ( J ^ { T } J ) / N ;$ , we can extract a limit on the largest eigenvalue: $\lambda _ { \operatorname* { m a x } } ( H _ { \mathrm { G N } } ) \leqslant \mathrm { T r } ( H _ { \mathrm { G N } } ) = \| J \| _ { F } ^ { 2 } / N$ . This upper bound on the eigenvalue dictates the maximum curvature of our optimisation landscape. It also points to the crucial dependence on the choice of coordinate system: if the network is trained directly in physical momentum space, this curvature can become unbounded. However, if we map the problem into a conformal space, it need not be. Let’s analyse the behaviour of the network gradient in these two spaces.

In the physical domain, the input coordinate, $x = s \in$ $\left( - \infty , s _ { \mathrm { t h } } = 4 m _ { \pi } ^ { 2 } \right]$ The deep spacelike region extends to negative infinity $( s \to - \infty )$ . If we calculate the derivative of the network’s prediction with respect to an internal frequency weight $a _ { k }$ (assuming its corresponding outer weight $\nu _ { k } \neq 0 )$ , the chain rule yields: $\partial _ { a _ { k } } f ( s ; \theta )$ = $\nu _ { k } \sigma ^ { \prime } ( a _ { k } s + b _ { k } ) s$ . As we evaluate the loss using data from the deep spacelike region, the magnitude of s grows unboundedly. Provided the activation function’s slope does not vanish at these extremes (which is generically true for the periodic sine activations, $\left| \sigma ^ { \prime } \right| \geqslant c _ { \sigma } > 0 )$ , this gradient can diverge uncontrollably, forcing the Frobenius norm $\| J \| _ { F } ^ { 2 }$ and thus $\lambda _ { \operatorname* { m a x } } ( H _ { \mathrm { G N } } )$ to explode. The gradient descent becomes inherently unstable due to steep ravines in the loss landscape. This unbounded curvature reveals a critical vulnerability in standard neural network architectures. If we attempt to avoid this explosion by using traditional activation functions like tanh, their exponentially decaying derivatives $( \sigma ^ { \prime } ( u ) = \operatorname { s e c h } ^ { 2 } ( u ) )$ lead to vanishing gradients and spectral starvation, which means the network does not learn the form factor in the high $q ^ { 2 }$ region. Conversely, highly expressive periodic activations, like the adaptive sine [20], where $| { \boldsymbol { \sigma } } ^ { \prime } | \geqslant c _ { \sigma } > 0 ;$ , preserve gradient signals but trigger the uncontrolled explosion in the unbounded $\mathcal { S }$ space.

Now, instead of operating in the physical ${ \mathcal { S } } .$ -space, let us consider a conformal bijection into the complex ${ \mathcal { L } } .$ plane, where the mapping z(s) compactifies the entire unbounded physical domain $( \mathcal { S }$ -space) into a finite-sized one, like, $\mathrm { e . g . }$ , the unit disk, $| z | \leqslant 1$ . For hadronic form factors in general, we can define this bounded domain generally as

$$
\mathcal { Z } = \mathbb { D } \backslash \bigcup _ { j = 1 } ^ { N _ { \Gamma } - 1 } \overline { { D _ { j } ( c _ { j } , r _ { j } ) } } ,\tag{12}
$$

where the closed disks are mutually disjoint and contained in $\mathbb { D } = \left\{ z \in \mathbb { C } : | z | < 1 \right\}$ . However, for the pion form factor, it simplifies beautifully. Since there is only one primary branch cut starting at $4 m _ { \pi } ^ { 2 }$ , the conformal bijection [15]

$$
z ( s ) = { \frac { \sqrt { 4 m _ { \pi } ^ { 2 } - s } - 2 m _ { \pi } } { \sqrt { 4 m _ { \pi } ^ { 2 } - s } + 2 m _ { \pi } } }\tag{13}
$$

smoothly compactifies the entire first Riemann sheet directly into the simple unit disk $| z | \leqslant 1$ (see Fig. 1).

Let our network in this new conformal domain be parametrised as: $\begin{array} { r l } { f ( z ; \theta ) = \sum _ { k = 1 } ^ { K } \nu _ { k } \sigma ( a _ { k } z + b _ { k } ) } & { { } } \end{array}$ . Because z is strictly bounded, the catastrophic gradient growth we observed earlier is mathematically eliminated, as the activation derivative $\vert \sigma ^ { \prime } \vert$ is already capped. As long as the network’s weights remain finite during optimisation, the magnitude of the gradient is guaranteed to be bounded by a finite constant. We can extend this logic to the second derivatives. Every term within the exact Hessian matrix relies solely on products of inherently bounded quantities such as the bounded input coordinate $z ,$ the finite network weights, bounded experimental data targets $( | y _ { i } | \leqslant Y )$ , and the bounded activation derivatives $( \sigma ^ { \prime }$ and $\sigma ^ { \prime \prime } )$ . Because no single element can diverge, there must exist a finite constant $C _ { H }$ such that the Hessian of the loss is bounded (see Appendix A for the proof):

$$
\| H \| _ { 2 } \leqslant C _ { H } < \infty .\tag{14}
$$

The bound depends on the network size and weight bounds, but not on the original unbounded physical coordinate s. By acting as a geometric preconditioner, the conformal mapping ensures the loss landscape possesses a well-defined maximum curvature. This implies that the loss gradient is globally Lipschitz continuous with a Lipschitz constant (maximum rate of change) $L = C _ { H }$ . Effectively, this establishes a curvature speed limit for the network’s optimisation dynamics. According to the standard descent lemma [21], taking a step along the gradient yields the following bound on the new loss:

$$
\mathcal { L } ( \theta - \eta \partial _ { \theta } \mathcal { L } ( \theta ) ) \leqslant \mathcal { L } ( \theta ) - \eta \left( 1 - \frac { \eta L } { 2 } \right) \| \partial _ { \theta } \mathcal { L } ( \theta ) \| _ { 2 } ^ { 2 } .\tag{15}
$$

The bracketed term on the right remains positive as long as the learning rate satisfies $\begin{array} { r } { \mathrm { ~ \bar { ~ } { ~ 0 ~ } ~ } < \eta < \frac { 2 } { \bar { L } } } \end{array}$ . Therefore, the loss decreases at every step unless a stationary point $( \partial _ { \theta } \mathcal { L } ( \theta ) = 0 )$ is reached. In addition to bypassing the instabilities that plague the physical momentum space, this approach ensures robust convergence for gradientdescent-based optimisation.

For the neural network, the crucial point is simply that its inputs remain bounded. Unlike traditional dispersive approaches, where the choice of conformal map encodes physical information about the branch cut structure, our framework is agnostic to this choice: any bijection that compactifies the unbounded physical domain into a finite region – a disk, an ellipse, or any smoothly bounded set – would eliminate the catastrophic gradient explosions and guarantee Hessian boundedness.<sup>1</sup> We adopt the map in Eq. (13), well-known in the literature [22], as a convenient and physically motivated choice.

## B. Preserving the Neural Tangent Kernel

In addition to successful convergence, we also require that the network maintain the expressivity needed to fit the data. We examine the network’s training dynamics in the parameter space through the empirical NTK [23], defined as:

$$
\Theta = \frac { 1 } { P } J J ^ { T } \ \in \mathbb { R } ^ { N \times N } .\tag{16}
$$

The diagonal elements of this kernel represent the local gradient magnitude for a given data point: $\Theta _ { i i } =$ $\lVert \partial _ { \theta } \bar { f } ( x _ { i } ; \theta ) \rVert _ { 2 } ^ { 2 } / P .$ . For a continuous-time gradient descent

with the mean-squared error (MSE) loss, the residual dynamics are governed by (see the appendix for proof):

$$
\frac { 1 } { \mathcal { L } } \frac { d \mathcal { L } } { d t } \leqslant - \frac { 2 P } { N } \lambda _ { \mathrm { m i n } } ( \Theta ) .\tag{17}
$$

Thus, the NTK must remain positive definite $( \lambda _ { \mathrm { m i n } } ( \Theta ) >$ 0) for $\mathcal { L }$ to converge to a global minimum successfully. If the minimum eigenvalue approaches zero, the network undergoes spectral starvation and loses the ability to learn the corresponding eigen-directions of the dataset.

This exposes another limitation of using the physical coordinates as direct inputs. The physical form factor must vanish in the deep Euclidean limit $( F _ { \pi } ( s ) \to 0$ as $s \to - \infty )$ . If the network is trained in the unbounded $\mathcal { S }$ -space using a periodic activation such as the adaptive sine, $\sigma ( a _ { k } s + b _ { k } )$ has no limit as $s \to - \infty$ for any fixed $a _ { k } \neq 0 ,$ as it oscillates indefinitely rather than settling to a fixed value. The only way for the network to approach a finite, tunable value in this limit is for the optimiser to shrink the internal weights $( a _ { k } \to 0 )$ . Hence, the parameter gradients evaluated at these deep spacelike points vanish:

$$
\operatorname* { l i m } _ { s _ { i }  - \infty } \lVert \partial _ { \theta } f ( s _ { i } ; \theta ) \rVert _ { 2 } ^ { 2 } = 0 \implies \Theta _ { i i }  0 .\tag{18}
$$

Because the minimum eigenvalue of a positive semidefinite matrix is upper-bounded by its minimum diagonal element, evaluating the network in the unbounded domain forces lim $\iota _ { s _ { i } \to - \infty } \lambda _ { \mathrm { m i n } } ( \Theta _ { s } ) = 0 .$ Thus, the diminishing form factor induces spectral starvation, flattening the loss landscape and stalling the optimisation.

Again, mapping the domain to the conformal Z -space eradicates this null space. Because the conformal coordinate maps the infinite spacelike tail to a finite boundary point $( z  1 )$ , the network parameters are no longer forced to collapse to zero to satisfy the asymptotic behaviour. For generic, non-polynomial activations like the adaptive sine function, the parameter Jacobian $J _ { z }$ naturally maintains full row rank. We can state it formally as the following theorem.

Theorem 1 (Full rank of the empirical NTK). Let $z _ { 1 } , \dots , z _ { N }$ be distinct points with $\left| z _ { i } \right| \leqslant 1 ,$ , and let $f ( z ; \theta )$ be a feedforward neural network with a real analytic, nonpolynomial activationfunction σ and a total of N hidden neurons across all layers. If $\mathcal { N } \geqslant N ,$ then for a generic choice of parameters $\theta ,$ the parameter Jacobian $J _ { z } \in \mathbb { R } ^ { N \times P }$ has row rank $N ,$ and the empirical NTK $\Theta _ { z } = J _ { z } J _ { z } ^ { T } / P$ is positive definite: $\lambda _ { \operatorname* { m i n } } ( \Theta _ { z } ) > 0 ;$ , where P is the total number of parameters, satisfying $P \geqslant { \mathcal { N } }$ since every neuron carries at least one parameter.

Check Appendix A for the proof of this theorem. Physically, the NTK governs the network’s expressive bandwidth, i.e., its ability to simultaneously resolve sharp features like the $\rho ( 7 7 0 )$ resonance and the smooth, asymptotic spacelike tail without sufering from spectral starvation. Because the conformal mapping prevents the parameter gradients from vanishing in the deep Euclidean limit, the empirical NTK in the conformal space remains strictly positive definite, $\lambda _ { \operatorname* { m i n } } ( \Theta _ { z } ) \geqslant \lambda _ { 0 } > 0$ , throughout the entire training interval. This ensures that no region of the kinematic phase space becomes a dead zone for learning. Under the linearised NTK gradient-flow dynamics, the residual error decays at a rate proportionally bounded by this minimum eigenvalue:

$$
\frac { d \mathcal { L } } { d t } \leqslant - \frac { 2 P \lambda _ { 0 } } { N } \mathcal { L } .\tag{19}
$$

This guarantees an exponential decay of the loss: $\mathcal { L } ( t ) \leqslant$ $\begin{array} { r } { \mathcal { L } ( 0 ) \exp \left( - \frac { 2 P \lambda _ { 0 } } { N } t \right) } \end{array}$ . Because no diagonal entry of $\Theta _ { z }$ collapses asymptotically, the trace of the NTK scales with the size of the dataset N:

$$
\mathrm { T r } ( \Theta _ { z } ) = \sum _ { i = 1 } ^ { N } \Theta _ { i i } \geqslant N \operatorname* { m i n } _ { 1 \leqslant i \leqslant N } ( \Theta _ { i i } ) > 0 .\tag{20}
$$

Finally, we note that the neural network can still arbitrarily approximate the true physical form factor $F ( s )$ in this new domain. Because the composition of the form factor and the inverse conformal map, $F ( s ( z ) ) =$ $F ( - 4 s _ { \mathrm { t h } } z / ( 1 - z ) ^ { 2 } )$ , remains complex analytic (holomorphic). The physical target is guaranteed to remain perfectly smooth and free of coordinate-induced singularities. When the infinite physical space is mapped into a closed unit disk, this subset shields the network from the divergent parameter errors from fitting asymptotic tails to infinity. With a smooth target on a finite canvas, the universal approximation theorem guarantees that our neural network can approximate the true pion form factor with arbitrary precision. Crucially, this theorem requires non-polynomial activations; a polynomial network would algebraically collapse and fail to map complex, sharp resonance peaks. The adaptive sine function $( \sigma ( x ) = \sin ( a x )$ , where a is a learnable vector) safely bypasses this restriction. In short, the conformal mapping delivers the best of both worlds: the geometric stability required for gradient descent to find a minimum, and the expressivity needed to ensure that minimum accurately reflects the true underlying QCD dynamics.

## C. Enforcement of physical constraints

We show the schematic of the simple four-layered, feedforward network we use in Fig. 2. We impose the physical and analytical constraints through the loss function. However, we also structurally encode two fundamental constraints directly into the architecture. First, we make a structural ansatz:

$$
F _ { \pi } ( z ) = 1 + s ( z ) \mathcal { N } ( z )\tag{21}
$$

where $\mathcal { N } ( z )$ is the complex-valued output of the neural network and $s ( z )$ is the inverse of the conformal map. This formulation guarantees that the charge normalisation, $F _ { \pi } ( 0 ) = 1$ , is satisfied identically, independent of any trainable parameters. Second, the pion form factor must be strictly real on the real axis below the threshold, i.e., Im $F _ { \pi } ( s ) = 0$ for $\ s < 4 m _ { \pi } ^ { 2 }$ , by the Schwarz reflection principle. Because our conformal mapping preserves this complex conjugation, the reality condition holds identically for z on the real axis. We impose this as an exact, architectural hard constraint. For any complex kinematic input z, the network evaluates the amplitude using only the absolute value of the imaginary part, |Im(z)|, efectively folding the lower half-plane onto the upper half-plane. To recover the analytic structure, the predicted imaginary part of the form factor is multiplied by sgn(Im(z)). This forward-pass routing inherently guarantees exact Schwarz reflection globally.

We construct a loss functional that acts as a regularising boundary in the loss landscape, preventing the network from settling into mathematically degenerate solutions while fitting the experimental data. It is a weighted superposition of experimental data losses and physicsmotivated losses:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \sum _ { d \in d a t a } \lambda _ { d } \mathcal { L } _ { d } + \sum _ { p \in p h y s } \lambda _ { p } \mathcal { L } _ { p } ,\tag{22}
$$

with $d a t a \equiv$ {spacelike, timelike} and $p h y s \equiv \{ \mathrm { a n a l y t i c } .$ ity, dispersion, moment, positivity, Watson, monotonicity, pQCD, t-asymptotic}. The weights $\lambda _ { i } ^ { \prime }$ s are empirically determined and fixed. We define the individual physics constraints below. For brevity, we show each loss term as a squared residual at a single evaluation point; the full loss is its mean over all relevant points for that constraint.

1. Analyticity $( { \mathcal { L } } _ { \mathrm { a n a l y t i c i t y } } ) { \mathrm { : } }$ We enforce the form factor to be smooth and diferentiable in the $\mathcal { Z }$ plane by minimising the violation of the Cauchy-Riemann (CR) conditions:

$$
r \frac { \partial u } { \partial r } = \frac { \partial \nu } { \partial \theta } , \quad r \frac { \partial \nu } { \partial r } = - \frac { \partial u } { \partial \theta } ,\tag{23}
$$

where $z = r e ^ { i \theta }$ with $r \leqslant R _ { \mathrm { m a x } }$ and $\theta \in [ - \pi , \pi ]$ , and $F _ { \pi } ( z ) = u ( z ) + i \nu ( z )$ , i.e.,

$$
\mathcal { L } _ { \mathrm { a n a l y t i c i t y } } = \left[ \left( r \frac { \partial u } { \partial r } - \frac { \partial \nu } { \partial \theta } \right) ^ { 2 } + \left( r \frac { \partial \nu } { \partial r } + \frac { \partial u } { \partial \theta } \right) ^ { 2 } \right] .\tag{24}
$$

We evaluate all spatial gradients exactly via automatic diferentiation, thereby eliminating finitediference truncation errors. Instead of evaluating the loss on a static, high-density lattice – which scales poorly $( \mathcal { O } ( N ^ { 2 } ) )$ and allows the neural network to interpolate unphysically or memorise analyticity solely at the grid nodes – we implement a continuous stochastic collocation method. At each training epoch, a new coordinate cloud $\{ z _ { j } \} _ { j = 1 } ^ { N _ { \mathrm { c o l } } }$ is sampled uniformly from the continuous interior domain $| z | \leqslant R _ { \mathrm { m a x } } = 0 . 9 9 9$ , avoiding boundary singularities from the physical branch cut. By dynamically shifting these evaluation points at each step, this Monte Carlo sampling approximates the continuous volume integral over the disk for a large number of epochs $( E \to \infty )$ . This strategy requires only a small $ { \mathcal { O } } ( N _ { \mathrm { c o l } } )$ point set per iteration (e.g., $1 0 ^ { 3 }$ points over $1 0 ^ { 4 }$ epochs), efectively scanning $\sim 1 0 ^ { 7 }$ unique points. This guarantees suficient spatial resolution across the domain and prevents localised violations of analyticity.

2. Dispersion relation $( \mathcal { L } _ { \mathrm { d i s p e r s i o n } } )$ : This relation essentially functions as an analytic holographic principle for the pion. It dictates that the entire complex interior is completely determined by its absorptive imaginary part (the physical cross-section)

![](images/cce93ede8bdc36c6dfa4ca6662a31f816ac1c0bf271ad7bb559284ea00a4755a.jpg)  
FIG. 2. Network schematic: PINN architecture for the pion form factor $F _ { \pi } ( s )$ in the conformal $\mathcal { Z }$ plane. The physical variable s is first mapped to the unit disk via the conformal bijection $z ( s )$ . The network takes three inputs – Rez, Imz, and a channel switch distinguishing the $e ^ { + } e ^ { - }$ <sup>−</sup> and τ-decay datasets – and passes them through four hidden layers of 128 neurons each with adaptive sine activations. The output $\mathcal { N } ( z )$ is combined with the structural ansatz $F _ { \pi } ( z ) = 1 + s ( z ) \mathcal { N } ( z )$ , which hardwires the charge normalisation $F _ { \pi } ( 0 ) = 1$ . Training minimises a composite loss $\begin{array} { r } { \mathcal { L } = \lambda _ { \mathrm { d a t a } } \mathcal { L } _ { \mathrm { d a t a } } + \lambda _ { \mathrm { p h y s } } \mathcal { L } _ { \mathrm { p h y s } } } \end{array}$ , where the physics loss enforces unitarity (Watson’s theorem), analyticity (Cauchy-Riemann conditions via stochastic collocation), the Schwarz reflection principle, the twice-subtracted dispersion relation, and perturbative QCD asymptotics simultaneously.

on the boundary branch cut. However, directly applying an unsubtracted dispersion integral exposes the network to severe ultraviolet (UV) instabilities from noisy high-energy extrapolations. To insulate the model, we perform two subtractions at the origin. Because we map the infinite physical branch cut onto the lower conformal boundary arc $( e ^ { i \theta }$ with $\theta \in [ - \pi , 0 ] )$ , we can evaluate this causality constraint entirely within the conformal plane. Enforcing charge normalisation $( u ( 0 ) = 1 )$ , the dispersive reconstruction for a spacelike point s is given by:

$$
u _ { \mathrm { d i s p } } ( s ) = 1 + { \frac { 1 } { 6 } } \langle r _ { \pi } ^ { 2 } \rangle s + { \frac { s ^ { 2 } } { \pi } } \int _ { 4 m _ { \pi } ^ { 2 } } ^ { \infty } { \frac { \nu ( s ^ { \prime } ) } { ( s ^ { \prime } ) ^ { 2 } ( s ^ { \prime } - s ) } } d s ^ { \prime } ,
$$

which, in the $\mathcal { Z }$ plane, can be written as,

$$
u _ { \mathrm { d i s p } } ( z ) = 1 + s ( z ) \left. \frac { d F _ { \pi } ( s ) } { d s } \right| _ { s = 0 } + \Delta _ { \mathrm { d i s p } } ( z ) .\tag{25}
$$

Rather than fixing the slope with an empirical parameter, we extract $F _ { \pi } ^ { \prime } ( 0 )$ dynamically from the network via automatic diferentiation to guarantee self-consistency and evaluate $\Delta _ { \mathrm { d i s p } } ( z )$ as,

$$
\Delta _ { \mathrm { d i s p } } ( z ) = \frac { s ^ { 2 } ( z ) } { 2 \pi s _ { \mathrm { t h } } } \int _ { - \pi } ^ { 0 } \frac { \nu ( e ^ { i \theta ^ { \prime } } ) | \sin \theta ^ { \prime } | ( 1 - \cos \theta ^ { \prime } ) } { 2 s _ { \mathrm { t h } } - s ( z ) ( 1 - \cos \theta ^ { \prime } ) } d \theta ^ { \prime } .
$$

We thus minimise the following squared loss:

$$
\mathcal { L } _ { \mathrm { d i s p e r s i o n } } = \left[ u ( z ) - u _ { \mathrm { d i s p } } ( z ) \right] ^ { 2 } .\tag{26}
$$

The suppression in the integration kernel heavily damps high-energy contributions, stabilising the loss landscape and focusing the network’s expressive power on the resonant physics.

3. Higher-moment sum-rules $( { \mathcal { L } } _ { \mathrm { m o m e n t } } )$ : Neural networks can sufer from spectral bias, a tendency for gradient descent to preferentially learn smooth, low-frequency functions while severely underestimating sharp, high-frequency features. In our context, this bias can cause the network to artificially flatten the local curvature at the origin, introducing unphysical parameter degeneracies (such as allowing the model to compensate incorrectly for the ρ − ω mixing phase). To break this degeneracy, we explicitly constrain the network’s dynamically generated local derivatives to match the global dispersive sum rules.

Along the physical real axis, the analyticity of the pion form factor lets us evaluate the generalised nth spectral moment $I _ { n }$ (n $\geqslant 0 )$ as:

$$
I _ { n } = \left. \frac { 1 } { n ! } \frac { d ^ { n } } { d s ^ { n } } F _ { \pi } ( s ) \right| _ { s = 0 } = \frac { 1 } { \pi } \int _ { s _ { \mathrm { t h } } } ^ { \infty } \frac { \ d \nu ( s ^ { \prime } ) } { ( s ^ { \prime } ) ^ { n + 1 } } d s ^ { \prime } ,\tag{27}
$$

with $I _ { 0 } = F _ { \pi } ( 0 ) = 1$ . This provides tight topological constraints to control how rapidly the spectral density flattens out as it transitions into the perturbative QCD continuum. As with the dispersion relation, integrating to infinity numerically is unstable. We therefore define ${ \mathcal { L } } _ { \mathrm { m o m e n t } }$ by compactifying the integral after projecting it onto the conformal boundary $z ^ { \prime } = e ^ { i \theta ^ { \prime } }$ , i.e.,

$$
\mathcal { L } _ { \mathrm { m o m e n t } } = \sum _ { n = 1 } ^ { 4 } \left[ \frac { 1 } { \pi } \int _ { - \pi } ^ { 0 } \frac { \nu ( e ^ { i \theta ^ { \prime } } ) s ( \theta ^ { \prime } ) | \sin \theta ^ { \prime } | } { s ^ { n + 1 } ( \theta ^ { \prime } ) ( 1 - \cos \theta ^ { \prime } ) } d \theta ^ { \prime } - \frac { F _ { \pi } ^ { ( n ) } ( 0 ) } { n ! } \right] ^ { 2 } ,\tag{28}
$$

where $F _ { \pi } ^ { ( n ) }$ denotes the nth derivative of $F _ { \pi }$ with respect to $s ( z )$ . These integral bounds restrict the network from introducing unphysical, wide spectral artefacts in deep inelastic regions where experimental data is sparse.

4. Spectral positivity $( \mathcal { L } _ { \mathrm { p o s i t i v i t y } } )$ : Unitarity dictates that since the absorptive part of the form factor on the branch cut corresponds to physical crosssections, it must be strictly non-negative:

$$
\nu ( s ) \geqslant 0 , \quad \mathrm { f o r } s > 4 m _ { \pi } ^ { 2 } .\tag{29}
$$

which translates to the lower unit-semicircle boundary in the $\mathcal { Z }$ plane. Since an inequality boundary is dificult to enforce dynamically, we implement it as a soft topological penalty in the loss function using a rectified linear unit (ReLU) functional (one-sided hinge loss):

$$
\mathcal { L } _ { \mathrm { p o s i t i v i t y } } = \left[ \operatorname* { m a x } \left( 0 , - \nu ( z ) \right) \right] ^ { 2 } ,\tag{30}
$$

The penalty is zero when the physics is respected, but grows quadratically when the network strays into unphysical values.

5. Watson’s theorem $( \mathcal { L } _ { \mathrm { W a t s o n } } ) \colon$ In the region where elastic scattering dominates $( 4 m _ { \pi } ^ { 2 } < s \stackrel { \textstyle - } { \sim } 1 \mathrm { G e V } ^ { 2 } )$ Watson’s theorem sets arg $F _ { \pi } ( z ) = \delta _ { 1 } ^ { 1 } ( s ( z ) )$ [see Eq. (6)]. However, computing tan $\mathfrak { l } ^ { - 1 } ( \nu ( \bar { z } ) / u ( z ) )$ can introduce coordinate singularities for small amplitudes and artificial discontinuous jumps across branch cuts. To bypass these, we reformulate the constraint as a geometric projection in the complex plane, where we minimise the perpendicular projection of the form factor relative to the target phase angle as,

$$
\begin{array} { r } { \mathcal { R } _ { \phi } = u ( z _ { k } ) \sin \delta _ { 1 } ^ { 1 } ( s _ { k } ) - \nu ( z _ { k } ) \cos \delta _ { 1 } ^ { 1 } ( s _ { k } ) = 0 . } \end{array}\tag{31}
$$

This scales smoothly as $\vert F _ { \pi } \vert \sin \bigl ( \delta _ { 1 } ^ { 1 } - \phi \bigr )$ , eliminating any singular denominators. Because $\mathcal { R } _ { \phi } ( s ) = 0$ is satisfied for both $\delta _ { 1 } ^ { 1 }$ and $\delta _ { 1 } ^ { 1 } + \pi$ , we enforce a directional constraint to avoid the wrong branch. We require the parallel projection to be strictly positive:

$$
\mathcal { P } _ { \phi } = u ( z ) \cos \delta _ { 1 } ^ { 1 } ( s ( z ) ) + \nu ( z ) \sin \delta _ { 1 } ^ { 1 } ( s ( z ) ) ,\tag{32}
$$

We apply the following loss functional:

$$
\mathcal { L } _ { \mathrm { W a t s o n } } = \mathcal { R } _ { \phi } ^ { 2 } + \left[ \operatorname* { m a x } ( 0 , - \mathcal { P } _ { \phi } ) \right] ^ { 2 } .\tag{33}
$$

By optimising these linear combinations of the form factor components $\left( u , \nu \right)$ , the PINN stably tracks the rapid phase variations across the $\rho ( 7 7 0 )$ resonance.

6. Phase monotonicity $( { \mathcal { L } } _ { \mathrm { m o n o t o n i c i t y } } ) { : }$ : To prevent unphysical phase oscillations or numerical backtracking in regions where experimental phase-shift data is sparse or plagued by inelastic channel openings, we implement a phase monotonicity loss along the lower unit-semicircle boundary in the conformal space as,

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { \mathrm { m o n o t o n i c i t y } } = \left[ \operatorname* { m a x } \left( 0 , - \frac { d } { d \theta } \left( \arg F _ { \pi } ( e ^ { i \theta } ) \right) \right) \right] ^ { 2 } } \\ { = \left[ \operatorname* { m a x } \left( 0 , \nu \frac { d u } { d \theta } - u \frac { d \nu } { d \theta } \right) \right] ^ { 2 } , } \end{array}\tag{34}
$$

for $\theta \in [ - \pi , 0 ]$ . It forces the angular derivative of the form factor’s phase to be non-negative.

7. Asymptotic pQCD (L<sub>PQCD</sub>): We also consider the asymptotic pQCD scaling behaviour in the deep spacelike region. From Eq. (8) (and Appendix B), we observe that,

$$
\frac { d } { d Q ^ { 2 } } \left[ \frac { Q ^ { 2 } u ( Q ^ { 2 } ) } { \alpha _ { s } ( Q ^ { 2 } / 2 1 ) \left[ 1 + 0 . 1 8 \alpha _ { s } \left( Q ^ { 2 } / 2 1 \right) \right] } \right] \approx 0 ,
$$

in the domain of pQCD. Thus, we minimise the following loss functional in the conformal space for $z \geqslant 0 . 8 2$ (or, $Q ^ { 2 } \gtrsim 8 \mathrm { G e V } ^ { 2 } )$

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { \mathrm { p Q C D } } = \left( \frac { ( 1 - z ) ^ { 3 } } { ( 1 + z ) } \frac { d } { d z } \left[ \frac { z } { ( 1 - z ) ^ { 2 } } \frac { u ( z ) } { \alpha _ { s } ( Q ^ { 2 } ( z ) / 2 1 ) } \right. \right. } \\ { \displaystyle \left. \left. \times \frac { 1 } { \left[ 1 + 0 . 1 8 \alpha _ { s } \left( Q ^ { 2 } ( z ) / 2 1 \right) \right] } \right] \right) ^ { 2 } , } \end{array}\tag{35}
$$

This way, we don’t bias the network by the pQCD normalisation, only the scaling.

8. Timelike asymptotic $( { \mathcal { L } } _ { \mathrm { t - a s y m p t o t i c } } )$ : Power counting rules of pQCD imply that the real part of the form factor drops as $1 / s$ as $s \to \infty , { \mathrm { i } } . { \mathrm { e } } . , u \sim 1 / s$ deep along the edge of the branch cut. With $s ( \theta ^ { \prime } ) =$ $8 m _ { \pi } ^ { 2 } / ( 1 - \cos \theta ^ { \prime } )$ on the $z ^ { \prime } = e ^ { i \theta ^ { \prime } }$ circle in the conformal plane, we have $\theta ^ { \prime 2 }$ ≈ $1 6 m _ { \pi } ^ { 2 } / s$ for small $\theta ^ { \prime } .$ Hence, we demand that, u grows as $\theta ^ { \prime 2 }$ near $\theta ^ { \prime } = 0$ $( \mathrm { i } . \mathbf { e } , \ s \to \infty )$ . In other words, we require $u ( 1 ) =$ $u ^ { \prime } ( 1 ) = 0$ so that we have

$$
u ( z ^ { \prime } ) \approx - { \frac { 1 } { 2 } } u ^ { \prime \prime } ( 1 ) ( z ^ { \prime } - 1 ) ^ { 2 } \approx - { \frac { 1 } { 2 } } u ^ { \prime \prime } ( 1 ) \theta ^ { \prime 2 }\tag{36}
$$

near $z ^ { \prime } = 1$ . Thus, we use the following loss function

$$
\mathcal { L } _ { \mathrm { t - a s y m p t o t i c } } = \left[ u ^ { 2 } + \left( \frac { d u } { d z ^ { \prime } } \right) ^ { 2 } \right] _ { z ^ { \prime } = 1 } .\tag{37}
$$

## IV. EXPERIMENTAL DATA

We train our network on both spacelike (primarily from electroproduction data) and timelike (including highprecision cross-section results from BaBar, CMD-3, and BESIII collaborations) datasets (see Table I for the references). Spacelike data are normally expressed in terms of $Q ^ { 2 } = - q ^ { 2 } ; F _ { \pi } ( Q ^ { 2 } )$ is well-measured up to $Q ^ { 2 }$ values as low as 0.28 GeV<sup>2</sup> by elastic electron-pion scattering. At large $Q ^ { 2 } , F _ { \pi } ( Q ^ { 2 } )$ is determined through pion electroproduction from a nucleon target, $e ^ { - } p \to e ^ { - } \pi ^ { + } n$ . The longitudinal part of the pion electroproduction cross section, $\sigma _ { L }$ , contains the pion exchange process, where a virtual photon couples to a virtual pion within the nucleon, from which these experiments extracted $F _ { \pi }$

TABLE I. Experimental datasets used for training the neural network.
<table><tr><td>Scattering channel</td><td>Experiment</td><td>Data points</td><td>Energy range (in  $\mathsf { G e V } ^ { 2 } )$ </td></tr><tr><td colspan="2">Spacelike data</td><td></td><td> $Q ^ { 2 } = - s$ </td></tr><tr><td rowspan="2">eπ → eπ</td><td>NA7 [24]</td><td>45</td><td>0.015-0.253</td></tr><tr><td>Fermilab [25]</td><td>20</td><td>0.03-0.07</td></tr><tr><td rowspan="4"> $e ^ { - } p \to e ^ { - } \pi ^ { + } n$ </td><td>CEA [26]</td><td>5</td><td>0.176-1.188</td></tr><tr><td>Cornell ’71 data [27]</td><td>6</td><td>0.62-2.015</td></tr><tr><td>JLab [28]</td><td>8</td><td>0.60-2.45</td></tr><tr><td>WSL at Cornell University [29]</td><td>6</td><td>1.2-4.0</td></tr><tr><td>Timelike data</td><td></td><td></td><td> $q ^ { 2 } = s$ </td></tr><tr><td> $\tau ^ { - } \to \pi ^ { - } \pi ^ { 0 } \nu _ { \tau }$ </td><td>Belle Collaboration [30]</td><td>62</td><td>0.088-3.125</td></tr><tr><td rowspan="8"></td><td>CLEO Collaboration [31]</td><td>41</td><td>0.11-2.81</td></tr><tr><td>CMD-2 Collaboration [32]</td><td>29</td><td>0.36-0.94</td></tr><tr><td>CMD-3 Collaboration [4]</td><td>209</td><td>0.11-1.44</td></tr><tr><td>DM2 detector [33]</td><td>17</td><td>1.82-4.52</td></tr><tr><td>CLEO-c [34]</td><td>2</td><td>14.2,17.4</td></tr><tr><td>ADONE storage ring at Frascati [35]</td><td>12</td><td>1.44-9</td></tr><tr><td>VEPP-2M, OLYA- 85 [36]</td><td>79</td><td>0.16-1.95</td></tr><tr><td>VEPP-2M, OLYA- 78 [37]</td><td>29</td><td>0.61-1.77</td></tr><tr><td rowspan="3"> $e ^ { + } e ^ { - } \to \pi ^ { + } \pi ^ { - } \gamma$ </td><td>SND detector [38]</td><td>36</td><td>0.28-0.78</td></tr><tr><td>KLOE detector [39]</td><td>75</td><td>0.1-0.85</td></tr><tr><td>BaBar [11, 12]</td><td>337</td><td>0.09-9</td></tr><tr><td rowspan="2"></td><td></td><td></td><td></td></tr><tr><td>BESIII detector [40]</td><td>60</td><td>0.36-0.81</td></tr></table>

For the timelike region, we use $e ^ { + } e ^ { - }$ scattering data: the direct energy-scan measurements from SND, CMD-2, and CMD-3, and the initial-state radiation (ISR)-based radiative return measurements at fixed baseline energies from BaBar, KLOE, and BESIII. For the $e ^ { + } e ^ { - }$ datasets, we use either the bare form factor, undressed of vacuum polarisation efects, or the one obtained after correcting for final-state radiation (FSR) efects, as applicable, by the following formula

$$
| F _ { \pi } ( s ) | ^ { 2 } = | F _ { \pi } ^ { \mathrm { e x p } } | ^ { 2 } | 1 - \Pi ( s ) | ^ { 2 } \left( \frac { \pi } { \pi + \alpha _ { e m } \mathfrak { f } ( s ) } \right) ,\tag{38}
$$

where

$$
\begin{array} { l } { { \displaystyle { \mathfrak { f } } ( s ) = \frac { 3 ( 1 + \sigma _ { \pi } ^ { 2 } ( s ) ) } { 2 \sigma _ { \pi } ^ { 2 } ( s ) } - 4 \log \sigma _ { \pi } ( s ) + 6 \log \frac { 1 + \sigma _ { \pi } ( s ) } { 2 } \ ~ } } \\ { { \displaystyle ~ + \frac { 1 + \sigma _ { \pi } ^ { 2 } ( s ) } { \sigma _ { \pi } ( s ) } F ( \sigma _ { \pi } ( s ) ) - \frac { \left( 1 - \sigma _ { \pi } ( s ) \right) } { 4 \sigma _ { \pi } ^ { 3 } ( s ) } \ ~ } } \\ { { \displaystyle ~ \times \left( 3 + 3 \sigma _ { \pi } ( s ) - 7 \sigma _ { \pi } ^ { 2 } ( s ) + 5 \sigma _ { \pi } ^ { 3 } ( s ) \right) \log \frac { 1 + \sigma _ { \pi } ( s ) } { 1 - \sigma _ { \pi } ( s ) } , } } \end{array}
$$

$$
\begin{array} { l } { \displaystyle \sigma _ { \pi } ( s ) = \sqrt { 1 - \frac { 4 m _ { \pi } ^ { 2 } } { s } } , } \\ { \displaystyle F ( x ) = - 4 \mathrm { L i } _ { 2 } ( x ) + 4 \mathrm { L i } _ { 2 } ( - x ) + 2 \log x \log \frac { 1 + x } { 1 - x } } \\ { \displaystyle ~ + 3 \mathrm { L i } _ { 2 } \Big ( \frac { 1 + x } { 2 } \Big ) - 3 \mathrm { L i } _ { 2 } \Big ( \frac { 1 - x } { 2 } \Big ) + \frac { \pi ^ { 2 } } { 2 } , } \\ { \displaystyle \mathrm { L i } _ { 2 } ( x ) = - \int _ { 0 } ^ { x } d t \frac { \log ( 1 - t ) } { t } . } \end{array}
$$

The term $| 1 - \Pi ( s ) | ^ { 2 }$ with the polarisation operator $\Pi ( s )$ excludes the efect of leptonic and hadronic vacuum polarisation [41], so that one obtains the bare cross section.

It is well-known that there is a systematic tension between the high-precision $e ^ { + } e ^ { - } \to \pi ^ { + } \pi ^ { - }$ cross-section measurements from BaBar and CMD-3, with CMD-3 yielding a significantly higher form factor magnitude $( | F _ { \pi } ( s ) | )$ across the $\rho$ resonance. BaBar uses ISR at a fixed centre-of-mass energy, and CMD-3 employs a step-bystep energy scan. Older direct-scan measurements like CMD-2 agree well with BaBar and τ-decay data, indicating the shift is specific to CMD-3 rather than an inherent flaw in the energy-scan methodology. This persistent discrepancy hints towards some unaccounted systematic errors in at least one framework, likely originating from sub-leading radiative photon corrections or integrated beam luminosity calibrations.

To evaluate the sensitivity of the learned representation to dataset-specific tensions, we train the network on five baseline sets. Two on the $e ^ { + } e ^ { - }$ data:

![](images/0f02aa62e7408301eca4511474916e810a435aac14441e1696febae38dda3d76.jpg)

![](images/05fe2f9d7e5c8a369fe14287fe56f60fba4fcf20f60b56549153850eca03d3c1.jpg)  
(a) |F<sub>π</sub>(s)|

![](images/c8ddd8ce314b19f18315561d3bbafc050baea47ea8c8badbfd2c90d69956b450.jpg)

![](images/fecb5027bddea3b4177f88632610f8e746e2c3265a8bc758d19002bf970320cd.jpg)  
(b) $\mathtt { A r g } ( F _ { \pi } ( s ) )$  
FIG. 3. (a) The pion form factor analytically continued on the complex $\mathcal { S }$ plane; (b) its argument validating the Schwarz relation. The peak is at the $\rho ( 7 7 0 )$ resonance. The discontinuity across the branch cut is clearly visible in the argument.

1. Set A (spacelike $\textit { + e } ^ { + } e ^ { - }$ data excluding CMD-3) and

2. Set B (spacelike + only CMD-3 data).

The form factor that appears in the $\tau ^ { - } \to \pi ^ { - } \pi ^ { 0 } \nu _ { \tau }$ decay is diferent from the electromagnetic form factor $F _ { \pi } ( s )$ in the $e ^ { + } e ^ { - }$ annihilation. The τ decay probes the magnitude of the weak form factor $f _ { \pi } ^ { - } ( s )$ . These two functions are related as [42]:

$$
| F _ { \pi } ( s ) | ^ { 2 } = | f _ { \pi } ^ { - } ( s ) | ^ { 2 } \times R _ { \mathrm { I B } } ( s ) ,\tag{39}
$$

where $R _ { \mathrm { I B } } ( s )$ compensates for the isospin-breaking:

$$
R _ { \mathrm { I B } } ( s ) = { \frac { 1 } { G _ { \mathrm { E M } } ( s ) } } { \frac { \beta _ { \pi ^ { + } \pi ^ { - } } ^ { 3 } ( s ) } { \beta _ { \pi ^ { - } \pi ^ { 0 } } ^ { 3 } ( s ) } } \left| { \frac { F _ { V } ( s ) } { f _ { + } ( s ) } } \right| ^ { 2 } .
$$

Here, $\beta _ { \pi \pi } ^ { 3 } ( s )$ accounts for the phase-space diference arising from the $\pi ^ { \pm } - \pi ^ { 0 }$ mass splitting, $G _ { \mathrm { E M } } ( s )$ isolates long-distance electromagnetic radiative corrections, and the form factor ratio $| \bar { F } _ { V } ( s ) / f _ { + } ( s ) | ^ { 2 }$ captures intrinsic hadronic isospin-breaking efects, including $\rho ^ { \pm } - \rho ^ { 0 }$ mass and width diferences $( \Delta m _ { \rho } , \Delta \Gamma _ { \rho } )$ as well as $\rho - \omega$ mixing present in the neutral channel.

3-4. Set C (spacelike + τ decay data): We train in two ways:

– Without channel switch (Set $\mathrm { C } _ { R _ { \mathrm { I B } } } )$ : we correct the τ data for isospin-violation explicitly by $R _ { \mathrm { { I B } } }$ and use that for training, and

– With channel switch (Set $\mathrm { C } _ { \mathrm { r a w } } )$ : we use the architectural flexibility of the PINN by making the neural network directly approximate the pure isovector form factor, $F _ { \pi } ^ { \bar { I } = \mathrm { { \bar { l } } } } ( s )$ , from the raw τ data. We handle the dataset diferences by using the mixing parameter $\alpha _ { \rho - \omega }$ in Eq. (7) as a conditional architectural switch in the forward pass. This way, the network handles the two data streams seamlessly, creating a joint representation, with $\alpha _ { \rho - \omega } = 0$ for the τ-decay channel and $\alpha _ { \rho - \omega } \neq 0$ for the $e ^ { + } e ^ { - }$ annihilation channel (see Fig. 2).

We also train on all datasets combined.

5. Set D (spacelike $+ \ e ^ { + } e ^ { - }$ data including CMD-3 + raw τ decay data with channel switch): We combine all available datasets to evaluate the overall model consistency. Similar to Set $\mathrm { { C } _ { \ r a w ; \Omega } }$ , the τ decay data is incorporated without explicit $R _ { \mathrm { { I B } } }$ corrections, relying on the conditional mixing parameter $\alpha _ { \rho - \omega }$ to handle the channel transition directly within the network architecture.

## V. RESULTS

## A. Global analytic structure and the zero-free landscape

We first recover the global analytic structure of the extracted pion electromagnetic form factor, $F _ { \pi } ( s )$ , across the complex s plane. Fig. 3a shows its absolute magnitude across the first Riemann sheet. The multiparticle branch cut lies along the positive real axis, starting from the two-pion threshold across the peak on the cut near $\mathsf { R e } ( s ) \approx 0 . 6 ~ \mathsf { G e V } ^ { 2 }$ , reproducing the profile of the $\rho ( 7 7 0 )$ vector resonance.<sup>2</sup> Constrained by ${ \mathcal { L } } _ { \mathrm { a n a l y t i c i t y } } ,$ the PINN yields a stable reconstruction without introducing unphysical poles or numerical artefacts away from the real axis.

The phase topology mapped in Fig. 3b confirms that the network output adheres to the Schwarz reflection principle. Approaching the cut from the upper half-plane yields a positive phase, whereas approaching from the lower half-plane produces a mirror-image negative value. Below the elastic threshold, the phase vanishes identically on the real axis. Beyond the threshold, the branch cut opens, splitting the phase and revealing a sharp, steplike clif across the real axis. This is enforced by design, as we only calculate the form factor in the upper half of the complex plane; the form factor in the lower half is the complex conjugate of the upper half. This behaviour is explained by Watson’s final-state interaction theorem. In the low-energy elastic region, the phase is locked to the scattering phase shift of two interacting pions, rising smoothly with energy, passing through $\pi / 2$ around the $\rho ( 7 7 0 )$ resonance and continuing toward π. The phase monotonicity loss $( \mathcal { L } _ { \mathrm { m o n o t o n i c i t y } } )$ ensures this stable angular trajectory in both elastic and inelastic domains.

Fig. 4 illustrates the logarithmic magnitude landscape over a wide kinematic range. Since the network has not been trained on any zero-free constraint, the absence of unphysical destructive interference patterns provides an independent, data-driven support for the zero-free hypothesis. We see a continuous drop in the deep-spacelike and complex-UV regions, aligning with pQCD scaling expectations $( 1 / s )$

![](images/2cc8f57611dca6e8274499472edd63b1fdb2941ca869e0b84c1349ef79a04e20.jpg)  
FIG. 4. Heat map of $| F _ { \pi } ( s ) |$ confirming that the predicted form factor contains no zeros in the interior of the $\mathcal { S }$ plane.

![](images/a46b53f3c7805d6276e0132cd05a69dfc7cc6d8aa0753fd595c413651e43f60b.jpg)  
FIG. 5. Convergence of the dispersion relation loss.

We also test the network’s prediction against the dispersion relation [Eq. (25)]. Fig. 5 illustrates the real component, Re $\left[ F _ { \pi } ( s ) \right]$ up to the two-pion threshold, $s \in \left[ - 2 . 0 , 4 m _ { \pi } ^ { 2 } \right]$ . We see that the absolute residual, $| \mathrm { R e } [ F _ { \pi } ( s ) ] _ { \mathrm { P I N N } } - \mathrm { R e } [ F _ { \pi } ( s ) ] _ { \mathrm { d i s p . } } |$ , remains tightly bounded. This adds further evidence that the network has constructed a holistic amplitude on the first Riemann sheet where the local real and imaginary predictions are holomorphic.

## B. Form factor across spacelike and timelike regions

Fig. 6 illustrates the predicted pion form factor $| F _ { \pi } ( s ) | ^ { 2 }$ across the spacelike and timelike regions alongside the experimental datasets. Using the $e ^ { + } e ^ { - }$ annihilation data for the timelike region (Set A), the network produces a curve that satisfies the charge normalisation and smoothly passes through the low-energy spacelike points, reproducing the $\rho ( 7 7 0 )$ vector resonance. (Unless specifically noted otherwise, all results presented in Sec. V are computed using dataset Set A.) The smoothness stems from the dispersion relation that acts as an analytical anchor; the shape of the curve in the timelike region dictates the spacelike behaviour, forcing the network to maintain a holomorphic and physically plausible transition between the two regions.

![](images/92cdc313e5c9dda3e86a03b334d942c1d01e5fbb36f68276e536ae219c5f34ba.jpg)

FIG. 6. The predicted pion electromagnetic form factor squared $| F _ { \pi } ( s ) | ^ { 2 }$ (black curve) with its associated 1σ uncertainty band (imperceptible at this scale as $\sigma \sim 1 0 ^ { - 2 } )$ , plotted alongside experimental data across both spacelike $( s < 0 )$ and timelike $( s > 4 m _ { \pi } ^ { 2 } )$ kinematic regions. The plot highlights the continuous transition through the low-energy region, the primary $\rho ( 7 7 0 )$ resonance peak, and the narrow $\rho - \omega$ interference structure around $s \approx \stackrel { \textstyle - } { 0 } . 6 \mathsf { G e V } ^ { 2 }$  
![](images/58f97c4a6efd5fd7d26572802de1f2903d4352d0ee79d41882c1df671c8fbab2.jpg)  
FIG. 7. PINN prediction for the timelike region with the $e ^ { + } e ^ { - }$ datasets showing the ρ resonance along with the dip-bump pattern from the $\rho { - } \omega$ interference.

The network natively isolates the strong-interaction physics (the isovector $I = 1$ state) and, at the same time, captures the interference pattern from $\rho { - } \omega$ mixing. As shown in Fig. 6, adjusting the $\alpha _ { \rho - \omega }$ switch from Eq. (7) allows the architecture to seamlessly incorporate the isoscalar $\left( I = 0 \right) \rho - \omega$ mixing mechanism without requiring any changes to the underlying neural network core. This mixing leaves a highly localised dip-bump ripple directly on the steep slope of the dominant $\rho$ peak.

![](images/e170669b507bb6cf3a31ff74d277f68fd2bdb16cfe2edc8fe7ed78d669942110.jpg)  
FIG. 8. PINN prediction for the spacelike region, overlaid on the experimental data.

Fig. 7 shows this explicitly for the electron-positron annihilation data $[ e ^ { + } e ^ { - }  \pi ^ { + } \pi ^ { - } ( \gamma ) ]$ With the switch active $( \alpha _ { \rho - \omega } \neq 0 )$ , the network tracks the abrupt phase swing and the small (less than a per cent) interference ripple with high precision, demonstrating that it has learned a universal, underlying representation of the form factor rather than memorising individual datasets. Setting $\alpha _ { \rho - \omega } = 0$ instead makes the network correctly ignore all interference, tracking the pure vector resonance smoothly over the data points; this is explored further in Sec. V H.

To complete the physical bridge, Fig. 8 illustrates the spacelike region of $F _ { \pi }$ alongside the experimental electroproduction data. The network produces a stable, monotonic damping that efectively matches the experimental points. The absence of unphysical oscillations or sudden upturns is consistent with the short-distance limits of quantum field theory

## C. The spacelike asymptotics

We analyse the asymptotic scaling of the charged pion electromagnetic form factor in Fig. 9, plotted as $\bar { Q } ^ { 2 } | F _ { \pi } ( Q ^ { 2 } ) |$ versus $Q ^ { 2 }$ . This range illustrates the transition between non-perturbative and perturbative QCD regimes, a region targeted by experiments such as JLab E12-19-006. We benchmark the neural network’s prediction against four theoretical bounds.

– Curve A represents the full, non-perturbative prediction derived from the Dyson-Schwinger equation (DSE) framework [43], which accounts for the dressed-quark mass function and dynamical chiral symmetry breaking (DCSB) across all energy scales. It transitions smoothly from the $\mathsf { l o w } { - } Q ^ { 2 }$ nonperturbative regime into the $\mathrm { \ h i g h } { - Q ^ { 2 } }$ asymptotic regime, serving as the theoretical benchmark for this analysis.

– Curve B shows the empirical monopole form $( F _ { \pi } ( Q ^ { 2 } ) = 1 / ( 1 + Q ^ { 2 } / m _ { \rho } ^ { 2 } ) )$ , representing the classic VMD extension into high energies. While this parametrisation matches low-energy data, it acts as an upper bound that overestimates the nonperturbative QCD behaviour at higher $Q ^ { 2 }$

The remaining curves map the short-distance, pQCD limits using the leading-order, leading-twist formula, $Q ^ { 2 } F _ { \pi } ( Q ^ { 2 } )$ ≈ $8 \pi \alpha _ { s } ( Q ^ { 2 } ) f _ { \pi } ^ { 2 } \bar { \omega } _ { \varphi } ^ { 2 }$ , with $\begin{array} { r } { \omega _ { \varphi } = \bar { \int _ { 0 } ^ { 1 } d x } \varphi _ { \pi } ( x ) / ( 3 x ) } \end{array}$

– Curve C is the pQCD prediction obtained using a simplified, semi-analytical pion distribution amplitude $\mathrm { ( D A ) } , \phi _ { \pi } ( x ; Q ^ { 2 } = 4 { \mathrm { G e V } } ^ { 2 } ) \approx x ^ { \rho } ( 1 - x ) ^ { \rho } \Gamma [ 2 ( \rho +$ $1 ) ] / \Gamma [ ( \rho + 1 ) ] ^ { 2 }$ , with $\rho = 0 . 3 .$ Instead of computing the full non-perturbative machinery across all scales, this curve isolates how a realistic, flat DA profile, structured by DCSB at a standard hadronic initialisation scale of $4 ~ \mathrm { G e V ^ { 2 } }$ , modulates the valence-quark gluon exchange kernel. Because it relies strictly on a hard-scattering factorisation framework, Curve C is initialised exclusively in the deep-spacelike domain $( Q ^ { 2 } \gtrsim 7 \ { \mathrm { G e V } } ^ { 2 } )$ where the strong coupling constant $\alpha _ { s } ( Q ^ { 2 } )$ is suficiently small to ensure perturbative convergence.

– Curve D shows the leading-order textbook formula for asymptotic pQCD using both the asymptotic DA, $\phi _ { a s } ( x ) = 6 x ( 1 - x )$ (the one we use in $\mathsf { A p - }$ pendix B), and a standard running $\alpha _ { s } ( Q ^ { 2 } )$ . This limit illustrates the expected behaviour if local, non-perturbative physics and dynamical mass generation vanished instantly outside the low-energy regime.

The PINN prediction (solid black curve) successfully bridges the non-perturbative and perturbative domains up to $Q ^ { 2 } = 2 0 \bar { \mathrm { G e V } } ^ { 2 }$ . At low-to-intermediate momentum transfers $( Q ^ { 2 } \lesssim 4 ~ \mathrm { G e V } ^ { 2 } )$ , the network smoothly interpolates existing spacelike measurements while staying well below the empirical monopole bound (Curve B), which lacks the QCD-driven logarithmic suppression at higher momentum transfer. In the high- $\cdot Q ^ { 2 }$ regime $( Q ^ { 2 } \gtrsim 7 \mathrm { G e V } ^ { 2 } )$ , the PINN prediction closely tracks the continuum DSE benchmark (Curve A) and approaches the broad-scale hard-scattering trajectory (Curve C). This close agreement stems from the fact that both the PINN (via its loss constraints) and the DSE/Curve C formulations preserve the non-perturbative dressing efects and the broad pion distribution amplitude generated by DCSB. Conversely, the PINN curve remains well above the asymptotic textbook limit (Curve D). Curve D assumes an immediate collapse to the asymptotic distribution amplitude $\phi _ { a s } ( x )$ , underestimating the persistent non-perturbative mass generation that survives at intermediate scales $( Q ^ { 2 } \sim 1 0 – 2 0 \mathrm { G e V } ^ { 2 } )$ . By enforcing asymptotic $1 / s$ power-law fallof and Cauchy-Riemann analyticity without imposing a rigid parameterisation, the network organically favours the realistic, non-perturbative continuum QCD trajectory over rigid empirical extensions or premature asymptotic assumptions.

## D. The complex phase and Watson’s theorem

We evaluate the PINN’s reconstruction of the complex phase of the pion electromagnetic form factor in Fig. 10, to benchmark the prediction directly against the rigorous solutions of the Roy equations. The extracted phase tracks the Roy equation solution very closely across the full window, successfully capturing the localised derivative $( d \delta / d s )$ that defines the physical decay width of the ρ meson. The absence of any systematic ofset between the two curves reflects the explicit Watson’s theorem constraint $( \mathcal { L } _ { \mathrm { { W a t s o n } } } ,$ Eq. 33) built into the training loss, which directly locks the phase of the isovector form factor to the elastic ππ scattering phase shift $\delta _ { 1 } ^ { 1 } ( s )$ in this kinematic region. To isolate the role of this constraint directly, Fig. 10 also shows the phase obtained when Watson’s loss is switched of during training $( \lambda _ { \mathsf { W a t s o n } } = 0 )$ . The resulting curve (blue dotted) visibly departs from both the full

![](images/af3b1430879a4e013cc49318af644645e382c4b18cab993fd8c4bdba4ca3489d.jpg)  
FIG. 9. Asymptotic scaling of the pion form factor multiplied by the squared momentum transfer, $Q ^ { 2 } | F _ { \pi } ( Q ^ { 2 } )$ |, in the spacelike domain up to $Q ^ { 2 } = - s = 2 0 { \mathrm { G e V } } ^ { 2 }$ . The solid black curve shows the PINN framework prediction. For comparison, Curve A (blue dashed) represents a QCD theoretical model bridging large and short distance scales; Curve B (orange dash-dotted) corresponds to a standard monopole parametrisation; and Curves C (green dash-dotted) and D (red dotted) illustrate high-energy short-distance quark-gluon approaches valid for $Q ^ { 2 } \gtrsim 7 \mathrm { \ G e V } ^ { 2 } .$

![](images/129a56265277257553969493568e5c45d106218af1b0d18bf3212c579f477e92.jpg)  
FIG. 10. The extracted phase of the form factor (black-solid) compared with the Roy equation data (orange-dashed) from Ref. [44]. The blue dotted curve shows the PINN prediction with Watson loss disabled $( \lambda _ { \mathrm { W a t s o n } } = 0 )$ , illustrating the systematic phase drift that results once the loss is removed.

PINN prediction and the Roy-equation solution, with the deviation growing with energy rather than remaining a static ofset. This is the same efect quantified in the ablation study (Table IV). Analyticity and dispersion alone constrain the global holomorphic structure of $F _ { \pi } ( s )$ , but they do not by themselves fix the functional form of the low-energy phase; that information must come from Watson’s theorem. Without it, the network still produces a smooth, monotonic phase (since $\mathcal { L } _ { \mathrm { m o n o t o n i c i t y } }$ is still active), but one that is systematically displaced from the physical ππ scattering phase shift.

Unlike a purely data-driven fit, the full PINN is architecturally constrained to satisfy elastic unitarity, rather than merely interpolating sparse experimental phase inputs. This agreement is nontrivial as standard Royequation solutions are anchored by sensitive low-energy boundary conditions such as the experimental S-wave scattering length $( a _ { 0 } ^ { 0 } )$ to which the PINN has no direct access. Instead, these global physics loss terms collectively constrain the low-energy phase; Watson’s theorem sets the target phase alignment in the elastic regime, while analyticity and dispersion enforce its smooth continuation across the entire kinematic domain.

Since the full PINN’s phase reconstruction in Fig. 10 tracks the Roy-equation baseline with no discernible deviation (in contrast to the $\lambda _ { \mathrm { W a t s o n } } = 0$ curve), we exploit the network’s direct access to the second Riemann sheet to extract the $\rho ( 7 7 0 )$ mass and width. A resonance is defined, process-independently, as a pole of the ππ scattering amplitude on the second (unphysical) Riemann sheet [45, 46]. We extract $m _ { \rho }$ and $\Gamma _ { \rho }$ by locating this pole directly, rather than relying on the conventional Breit-Wigner-motivated extraction from the realaxis phase (the $9 0 ^ { \circ }$ phase-crossing point and its local slope), which implicitly neglects any non-resonant background phase. Sheet II is reached from the network’s Sheet-I output $F _ { \pi } ^ { I } ( s )$ via the elastic-unitarity continuation,

$$
F _ { \pi } ^ { I I } ( s ) = F _ { \pi } ^ { I } ( s ) e ^ { - 2 i \delta _ { 1 } ^ { 1 } ( s ) } ,
$$

valid below the inelastic threshold where Watson’s theorem [Eq. 6] fixes the phase of $F _ { \pi } ^ { I } ( s )$ to the P-wave ππ phase shift $\delta _ { 1 } ^ { 1 } ( s )$ . This is the standard technique used to extract light-meson poles from dispersive phase-shift solutions [47, 48]. Writing the inverse amplitude as $D ( s ) \equiv 1 / F _ { \pi } ^ { I I } ( s ) = S _ { 1 } ^ { 1 } ( s ) / F _ { \pi } ^ { I } ( s )$ , where $S _ { 1 } ^ { 1 } ( s ) = e ^ { 2 i \delta _ { 1 } ^ { 1 } ( s ) }$ is the elastic ππ S-matrix element, makes explicit that a zero of $D ( s )$ is equivalent to a zero of $S _ { 1 } ^ { 1 } ( s )$ itself, i.e., a genuine ππ resonance pole, provided $F _ { \pi } ^ { \dot { I _ { ( s ) } } }$ contributes no spurious zeros of its own. This is supported by the zero-free analytic structure established in Fig. 4. Searching for the zero of $D ( s )$ rather than the pole of $F _ { \pi } ^ { I I } ( s )$ directly also converts an ill-posed numerical divergence into a wellbehaved root-finding problem.

A two-dimensional Newton-Raphson search over $s =$ $s _ { R } + i s _ { I } $ , initialised near the $\rho$ peak, converges to

$$
s _ { \mathrm { p o l e } } = ( 0 . 5 7 5 6 - 0 . 1 0 3 6 i ) ~ \mathrm { G e V } ^ { 2 } , \quad | D ( s _ { \mathrm { p o l e } } ) | ^ { 2 } \sim 2 \times 1 0 ^ { - 1 7 } ,
$$

confirming that the root is resolved to machine precision rather than sitting at a shallow local minimum. Using the standard parameterisation $\sqrt { s _ { \mathrm { p o l e } } } = m _ { \rho } ^ { \mathrm { p o l e } } - i \Gamma _ { \rho } ^ { \mathrm { p o l e } } / 2 .$

we extract the conventional pole mass and width,

$$
\begin{array} { r l } & { m _ { \rho } ^ { \mathrm { p o l e } } = \mathrm { R e } \sqrt { s _ { \mathrm { p o l e } } } = 7 6 1 . 7 2 \pm 1 . 0 4 \mathrm { M e V } , } \\ & { \Gamma _ { \rho } ^ { \mathrm { p o l e } } = - 2 \mathrm { I m } \sqrt { s _ { \mathrm { p o l e } } } = 1 3 5 . 9 9 \pm 1 . 2 0 \mathrm { M e V } . } \end{array}
$$

As a cross-check, we also extract the resonance parameters from the conventional real-axis phase-crossing definition, $\delta _ { 1 } ^ { 1 } ( m _ { \rho } ^ { 2 } ) = \pi / 2$ , with the width inferred from the local phase slope, $\Gamma _ { \rho } = \lceil m _ { \rho } ( d \delta _ { 1 } ^ { 1 } / d s ) \rceil _ { s = m _ { \rho } ^ { 2 } } \rceil ^ { - 1 }$ , giving $m _ { \rho } =$ $7 7 3 . 3 6 \pm 0 . 3 8$ MeV and $\Gamma _ { \rho } \approx 1 4 9 . 5 9 \pm \bar { 2 } . 4 2$ MeV. This agrees with the pole extraction to within $\sim 1 . 5 \%$ in mass, while the width difers by about 10%, a statistically significant spread given the sub-MeV/few-MeV uncertainties on each extraction. This is expected as a slowly-varying non-resonant background phase barely shifts the steep $9 0 ^ { \circ }$ crossing point, but contributes directly, and without suppression, to the local slope used to infer $\Gamma _ { \rho ; }$ , a bias the background-free pole extraction does not inherit. Both extractions are in the same ballpark as the PDG T-matrix pole, ${ \sqrt { s _ { \mathrm { p o l e } } } } = ( 7 6 1 { \cdot } 7 6 5 ) - i ( 7 1 { \cdot } 7 4 )$ MeV [49]. Our pole mass agrees to within $1 \%$ , while our extracted pole width sits approximately 4-8% below the PDG world average. This minor width deficit is a natural target for future multi-channel extensions of the framework.

## E. Uncertainty quantification

To quantify the reliability of the predicted observables, such as the pion charge radius or the two-pion hadronic contribution to the muon anomalous magnetic moment, we decouple statistical data-driven fluctuations from systematic model choices by decomposing the total uncertainty budget into independent statistical $( \sigma _ { \mathrm { { s t a t } } } )$ and calibration $( \sigma _ { \mathrm { c a l i } } )$ components, quoting observables as $\hat { \mathcal { O } } \pm$ $\sigma _ { \mathrm { s t a t } } \pm \sigma _ { \mathrm { c a l i } }$ . The two are estimated from independent ensembles and combined in quadrature.

To estimate the statistical uncertainty $( \sigma _ { \mathrm { { s t a t } } } )$ , we employ a Monte Carlo bootstrapping procedure on the experimental dataset. We first generate pseudo-datasets by sampling individual measurements from Gaussian distributions defined by their reported central values and standard errors. For high-statistics datasets providing published covariance matrices, namely KLOE, BaBar, Belle, and $\mathrm { C L E O } ,$ data points are drawn jointly from the corresponding multivariate Gaussian distribution to preserve bin-to-bin correlations; for all remaining datasets lacking published covariances, measurements are sampled independently. We construct $N = 5 0 0$ such pseudo-datasets and train an independent network on each realisation, where the bootstrap seed simultaneously governs the data sampling, network weight initialisations, and minibatch sequencing, so that $\pmb { \sigma } _ { \mathrm { s t a t } }$ reflects both data resampling and ordinary training variation. Final central values and their associated statistical uncertainties $( \sigma _ { \mathrm { { s t a t } } } )$ for all observables are evaluated as the ensemble mean and standard deviation across these bootstrap realisations, respectively.

The loss weighting hyperparameters $\lambda _ { i }$ carry no physical units and do not correspond to measurable observables or physical couplings; they are akin to calibration parameters in an experiment. As long as the network satisfies the corresponding constraint via $\mathcal { L } _ { i } ,$ the physical observables will ideally satisfy $\partial \mathcal { O } / \partial \lambda _ { i } = 0$ . In practice, the $\lambda _ { i }$ act as numerical hyperparameters that scale gradient step sizes during backpropagation, preventing any single constraint from dominating the optimisation landscape, and the residual dependence of $\mathcal { O }$ on $\lambda _ { i }$ away from this ideal is precisely what we quantify below as $\sigma _ { \mathrm { c a l i } }$

Because the relative weights (λ<sub>i</sub>) represent modelling choices with no unique theoretical prescription, we systematically sample the ten-dimensional hyperparameter space to quantify the resulting variation in our observables. Instead of varying parameters one at a time, we vary each hyperparameter λ over a factor of two, spanning the interval $[ \lambda / 2 , 2 \lambda ]$ ], using a Latin Hypercube Sampling scheme. To guarantee uniform, space-filling coverage across the parameter domain that random sampling cannot achieve, the hyperparameter space is sampled at $N = 6 4$ points generated via a centred-discrepancyoptimised Latin hypercube. For each configuration, a neural network is trained using an identical adaptive learning-rate schedule and convergence criterion. The calibration uncertainty on each observable is then defined as the standard deviation across these $N = 6 4$ converged network configurations. Because each calibration trial represents a fully converged fit to the unperturbed central experimental data, the statistical and calibration ensembles are computationally independent and probe orthogonal sources of uncertainty. Consequently, we report the statistical and calibration uncertainties separately throughout, combining them in quadrature $( \sigma _ { \mathrm { t o t } } ^ { 2 }$ = $\sigma _ { \mathrm { s t a t } } ^ { 2 } + \sigma _ { \mathrm { c a l i } } ^ { 2 } )$ when determining the total uncertainty.

## F. Charge radius of the pion

The Taylor expansion of $F _ { \pi } ( s )$ at low momentum transfer $( s \to 0 )$ is defined in Eq. (2). The physical consistency of our PINN framework is further benchmarked by extracting the pion squared charge radius, $\langle r _ { \pi } ^ { 2 } \rangle$ , which is obtained from the first derivative of the form factor at zero momentum transfer:

$$
\langle r _ { \pi } ^ { 2 } \rangle = 3 ! \kappa ^ { 2 } \left. \frac { d F _ { \pi } ( s ) } { d s } \right| _ { s = 0 } ,\tag{40}
$$

where $\kappa = 0 . 1 9 7 3 2 7$ GeV fm is the natural unit conversion factor from $\mathrm { G e V ^ { - 1 } }$ to fm. As summarised in Table II, our results for the pion form-factor moments are in strong agreement with established theoretical and experimental values in the literature. For the squared charge radius $\langle r _ { \pi } ^ { 2 } \rangle$ , our result of $0 . 4 3 5 \pm 0 . 0 0 8 _ { \mathrm { s t a t } } \mathrm { \bar { \pm } 0 . 0 0 7 _ { \mathrm { c a l i } } \mathrm { \bar { f m } ^ { 2 } } }$ is consistent with the PDG world average [49]. Furthermore, it remains fully compatible with the lattice QCD prediction [50] at $0 . 6 7 \sigma ,$ , while reducing the statistical uncertainty relative to the lattice extraction by more than a factor of two. These precision gains are supported by the global nature of our framework, which constrains the derivative at $s = 0$ using the full spacelike and timelike spectrum through dispersive integrals rather than local polynomial approximations.

For the higher-order curvature terms, our prediction for the fourth-order moment $\langle r _ { \pi } ^ { 4 } \rangle = 0 . 7 4 8 \pm \bar { 0 } . 0 5 7 _ { \mathrm { s t a t } }$ ± $ { 0 . 0 4 8 _ { \mathrm { c a l i } } }$ fm<sup>4</sup> shows excellent agreement (0.39σ) with two-loop ChPT [51] and sits just above the range of the dispersive bounds obtained in Ref. [52]. Notably, our network constrains the uncertainty of $\langle r _ { \pi } ^ { 4 } \rangle$ ⟩ to about twothirds of that from two-loop ChPT.

TABLE II. Comparison of the extracted pion charge radius squared $\langle r _ { \pi } ^ { 2 } \rangle$ , fourth-order radius $\langle r _ { \pi } ^ { 4 } \rangle$ , and sixth-order radius $\langle \bar { r _ { \pi } ^ { 6 } } \rangle$ from our PINN framework against existing theoretical, lattice QCD, ChPT, and empirical estimates from the literature. All values are given in units of $\mathbf { f m } ^ { n }$ (where $n = 2 , 4 , 6 )$ . Errors for our work are presented in the form $\hat { \sigma } \pm \sigma _ { \mathrm { s t a t } } \pm \sigma _ { \mathrm { c a l i } }$ as defined in Sec. V E
<table><tr><td>Observable Previous estimation (in fmⁿ)</td><td></td><td>Our result (in fmⁿ)</td></tr><tr><td> $\langle r _ { \pi } ^ { 2 } \rangle$ </td><td>Lattice  $[ 5 0 ] \colon 0 . 4 2 \pm 0 . 0 2$  PDG  $[ 4 9 ] \colon 0 . 4 3 4 \pm 0 . 0 0 5$ </td><td> $0 . 4 3 5 \pm 0 . 0 0 8 \pm 0 . 0 0 7$ </td></tr><tr><td> $\langle r _ { \pi } ^ { 4 } \rangle$ </td><td>2-loop ChPT  $[ 5 1 ] \colon 0 . 7 \pm 0 . 1 1$  Ref.  $[ 5 2 ] \colon 0 . 6 8 - 0 . 7 2$ </td><td> $0 . 7 4 8 \pm 0 . 0 5 7 \pm 0 . 0 4 8$ </td></tr><tr><td> $\langle r _ { \pi } ^ { 6 } \rangle$ </td><td>Ref.  $[ 5 2 ] \colon 2 . 9 5 - 3 . 1 1$  Ref.  $[ 5 3 ] \colon 2 . 8 9 \pm 0 . 1 2$ </td><td> $4 . 0 1 \pm 0 . 9 5 \pm 0 . 9 0$ </td></tr></table>

Finally, our sixth-order moment $\langle r _ { \pi } ^ { 6 } \rangle = 4 . 0 1 \pm 0 . 9 5 \mathrm { s t a t } \pm$ $0 . 9 0 _ { \mathrm { c a l i } } \ : \mathrm { f m } ^ { 6 }$ suggests a slightly enhanced low-energy curvature compared to earlier estimates, remaining consistent at the 0.8σ level with both the dispersive bounds of Ref. [52] and the estimate of Ref. [53]. Such minor shifts are physically expected given that higher-order derivatives at $s = 0$ act as sensitive probes of the complexplane geometry, receiving non-local contributions from the timelike resonance spectrum and the high-s asymptotic tail, both of which are dynamically coupled in our network via exact S-matrix constraints.

## G. Implications for the muon anomalous magnetic moment

The hadronic contribution to the anomalous magnetic moment of the muon can be written as [44, 54]:

$$
a _ { \mu } ^ { \pi \pi } = \left( \frac { \alpha _ { e m } m _ { \mu } } { 3 \pi } \right) ^ { 2 } \int _ { 4 m _ { \pi } ^ { 2 } } ^ { \infty } d s \frac { K ( s ) } { s ^ { 2 } } R _ { \pi \pi } ( s ) ,\tag{41}
$$

where $\alpha _ { e m } = e ^ { 2 } / 4 \pi _ { : }$ , is the fine structure constant, and $K ( s )$ is the kernel function dominating at low energies:

$$
\begin{array} { c } { { { \displaystyle K ( s ) = \frac { 3 s } { m _ { \mu } ^ { 2 } } \left[ \frac { x ^ { 2 } } { 2 } ( 2 - x ^ { 2 } ) + \frac { ( 1 + x ^ { 2 } ) ( 1 + x ) ^ { 2 } } { x ^ { 2 } } \right. } } } \\ { { { \left. \qquad \times \left( \log ( 1 + x ) - x + \frac { x ^ { 2 } } { 2 } \right) + \frac { 1 + x } { 1 - x } x ^ { 2 } \log x \right] } , } } \end{array}\tag{42}
$$

with

$$
x = \frac { 1 - \sigma _ { \mu } ( s ) } { 1 + \sigma _ { \mu } ( s ) } , \sigma _ { \mu } ( s ) = \sqrt { 1 - \frac { 4 m _ { \mu } ^ { 2 } } { s } } ,
$$

and $R _ { \pi \pi }$ is related to the specific case of two-pion production $( e ^ { + } e ^ { - } \to \pi ^ { + } \pi ^ { - } )$ , and is directly proportional to

TABLE III. Comparison of the two-pion hadronic vacuum polarisation contribution to the muon anomalous magnetic moment, $a _ { \mu } ^ { \pi \pi } \times 1 0 ^ { 1 0 }$ , integrated over various centre-of-mass energy intervals $( \sqrt { s }$ in GeV). Our PINN estimations are benchmarked against existing dispersive and data-driven determinations from the literature. Errors for our work are presented in the form $\hat { \sigma } \pm \sigma _ { \mathrm { s t a t } } \pm \sigma _ { \mathrm { c a l i } }$ as defined in Sec. V E.
<table><tr><td>Energy range (GeV)</td><td>Previous estimations</td><td>Our estimation</td></tr><tr><td> $[ 2 m \pi , 0 . 6 3 ]$ </td><td> $1 3 3 . 2 5 8 \pm 0 . 7 2 3$  [55]  $1 3 0 . 4 7 \pm 0 . 6 4$  [44]</td><td> $1 3 0 . 6 6 \pm 0 . 6 7 \pm 1 . 2 0$ </td></tr><tr><td> $[ 2 m \tau , 1 . 0 ]$ </td><td> $4 9 5 . 0 \pm 2 . 5 8$  [44]</td><td> $4 9 6 . 3 7 \pm 2 . 0 8 \pm 1 . 7 0$ </td></tr><tr><td> $[ 2 m _ { \pi } , \infty ]$ </td><td> $5 1 3 . 2 \pm 3 . 8 \ [ 5 7 ]$   $5 0 7 . 9 \pm 3 . 2 9 \ [ 5 6 ]$ </td><td> $5 0 6 . 4 8 \pm 2 . 0 2 \pm 1 . 7 0$ </td></tr></table>

the square of the pion form factor.

$$
R _ { \pi \pi } ( s ) = \frac { 1 } { 4 } \left( 1 - \frac { 4 m _ { \pi } ^ { 2 } } { s } \right) ^ { 3 / 2 } | F _ { \pi } ( s ) | ^ { 2 } .
$$

In the presence of FSR, the above equation is modified to [44]:

$$
R _ { \pi \pi } ^ { \mathrm { F S R } } ( s ) = \left[ 1 + \frac { \alpha _ { e m } } { \pi } \mathfrak { f } ( s ) \right] R _ { \pi \pi } ( s )\tag{43}
$$

where ${ \mathfrak { f } } ( s )$ is defined in Eq. (38). In Table III, we present our results for $a _ { \mu } ^ { \pi \pi }$ integrated over three characteristic energy domains alongside prominent estimations in the literature. In the low-energy region below $0 . 6 3 \mathrm { G e V } ,$ our result $( 1 3 0 . 6 6 \pm 0 . 6 7 _ { \mathrm { s t a t } } \pm 1 . 2 0 _ { \mathrm { c a l i } } )$ is consistent with the dispersive evaluation of Colangelo et al. [44] within 0.14σ (evaluating total uncertainties in quadrature), while differing from Ananthanarayan et al. [55] by 1.67σ. For the primary $\rho$ -resonance region $[ 2 m _ { \pi } , 1 . 0 \mathsf { G e V } ]$ , our value of $4 9 6 . 3 7 \pm 2 . 0 8 _ { \mathrm { s t a t } } \pm 1 . 7 0 _ { \mathrm { c a l i } }$ is consistent with Colangelo et al. [44], difering by 0.37σ. Finally, extending the integration across the entire spectrum $[ 2 m _ { \pi } , \infty ]$ yields $a _ { \mu } ^ { \pi \pi } \times 1 0 ^ { 1 0 } = 5 0 6 . 4 8 \pm 2 . 0 2 _ { \mathrm { s t a t } } \pm 1 . 7 0 _ { \mathrm { c a l i } }$ , compatible with the benchmark evaluation of the Muon $g - 2$ Theory Initiative, Ref. [56], within 0.34σ, and 1.45σ lower than Ref. [57].

Our physics-informed framework achieves a refined total uncertainty (±2.64 in quadrature) compared to traditional evaluations (from ±3.29 to ±3.8). Standard datadriven integrations directly propagate point-to-point experimental noise and local dataset discrepancies through the $K ( s ) / s ^ { 2 }$ integration kernel. In contrast, the PINN provides a globally consistent analytic framework by enforcing Cauchy-Riemann analyticity, Watson’s final-state interaction theorem, and $1 / s$ pQCD asymptotic fallof within the conformal z-plane. It suppresses unphysical local fluctuations and resolves sharp $\rho \mathbf { - } \omega$ interference features without relying on rigid functional parameterisations. The resulting representation of $F _ { \pi } ( s )$ reproduces established low-energy moments and $a _ { \mu } ^ { \pi \pi }$ integrals while establishing a robust, physics-constrained baseline for hadronic vacuum polarisation.

## H. Hyperparameter loss weighting and structural ablation

We probe the sensitivity of the loss functionals through ablation $( \lambda _ { i } = 0 \mathrm { v s . } \ \lambda _ { i } \neq 0 )$ . By selectively setting individual weights to zero, we isolate and test the structural necessity of each theoretical principle (such as Watson’s theorem, pQCD asymptotics, or dispersion relations) by observing their efect on various physical quantities (see Table IV). Physical observables such as $a _ { \mu } ^ { \pi \pi }$ do not directly enter the loss functional but instead emerge a posteriori as outcomes evaluated via dispersion integrals over $F _ { \pi } ( s ) _ { \mathit { \pi } }$ ; nonetheless, as the results below show, they remain highly sensitive to which constraints and datasets are active during training.

Significance of first-principles learning: Our results indicate that data-driven constraints alone guarantee only localised numerical interpolation, whereas the global uniqueness and structural integrity of the pion form factor rely on the physics losses. The numbers in Table IV reveal a clear hierarchy among the losses based on their role in shaping $F _ { \pi } ( s )$ . Watson’s elastic unitarity theorem emerges as the single most vital physics loss in the framework. Removing this constraint $( \lambda _ { \mathsf { W a t s o n } } = 0 )$ destabilises the network: $\langle \bar { r } _ { \pi } ^ { 2 } \rangle , \langle r _ { \pi } ^ { 4 } \rangle$ , and $\langle r _ { \pi } ^ { 6 } \rangle$ inflate by about 25%, 80%, and 179%, respectively, and the phase shifts show marked deviations of $7 . 7 0 ^ { \circ }$ at $s _ { 0 }$ and $1 1 . 4 5 ^ { \circ }$ at $s _ { 1 }$ . Watson’s theorem ties the phase of $F _ { \pi } ( s )$ directly to elastic ππ scattering phase shifts below the inelastic threshold; without it, the network loses its directional phase alignment, causing distortions in the imaginary spectrum, and spoils the low-energy slope.

On the other hand, disabling the Cauchy-Riemann analyticity constraint $( \lambda _ { \mathrm { a n a l y t i c i t y } } = 0 )$ or the explicit dispersion relation $( \lambda _ { \mathrm { d i s p e r s i o n } } \dot { = } 0 )$ causes modest shifts in the HVP integral and low-energy moments, but introduces growing phase-shift errors at higher energies $( | \Delta \delta _ { 1 } ^ { 1 } ( s _ { 1 } ) |$ ≈ $2 . 5 ^ { \circ } { - 2 . 9 ^ { \circ } } )$ . Intuitively, global analyticity and Cauchy integrals enforce smooth connectivity across the complex plane; removing them allows local gradient noise to unbind the real and imaginary parts of the form factor above the $\rho ( 7 7 0 )$ peak. Similar phase deviations are seen by setting $\lambda _ { \mathrm { m o n o t o n i c i t y } } = 0 ,$ , which requires the P-wave phase shift to increase monotonically with energy $( \partial \delta _ { 1 } ^ { 1 } / \partial s > 0 )$ in the elastic domain. Disabling monotonicity enhances the phase error at higher energies $( \sim 2 . 2 0 ^ { \circ } \mathrm { a t } s _ { 1 } )$ . Monotonicity prevents numerical backtracking and unphysical phase oscillations, particularly near resonance thresholds where phase shifts change rapidly.

The moment constraint $( \lambda _ { \mathrm { { m o m e n t } } } = 0 )$ plays a diferent, more localised role. Omitting this curvature sum rule causes $\langle r _ { \pi } ^ { 6 } \rangle$ to explode to about $1 2 . 7 \mathrm { f m } ^ { 6 }$ (more than triple the baseline value), while leaving $a _ { \mu } ^ { \pi \pi }$ virtually untouched. This happens because neural networks normally sufer from spectral bias, a tendency to smooth out high-frequency curvature near $s = 0$ . The moment loss acts as an essential low-energy anchor penalising unphysical local flattening.

The network output shows only a mild sensitivity towards spectral positivity $( \lambda _ { \mathrm { p o s i t i v i t y } } = 0 )$ , as the dense experimental data in the elastic resonance region naturally force Im $F _ { \pi } ( s ) \geqslant 0$ . The loss mainly serves as an unphysical-sign regulator that prevents the spectral function from becoming negative in sparse kinematic regions or high-energy tails. Similarly, the asymptotic constraints $( \lambda _ { \mathrm { { p Q C D } } } = 0$ and $\lambda _ { t - \mathrm { a s y m p t o t i c } } = 0 )$ only act as high-energy boundary controls; disabling them allows the network’s high-energy tail to drift freely, degrading the scattering phase at $s _ { 1 }$ up to about $5 ^ { \circ }$ without afecting the other quantities significantly.

TABLE IV. Performance comparison of pion form factor moments $\langle r _ { \pi } ^ { 2 n } \rangle$ , integrated HVP contributions $a _ { \mu } ^ { \pi \pi } \times 1 0 ^ { 1 0 } ;$ , and P-wave phaseshift deviations $| \Delta \delta _ { 1 } ^ { 1 } ( s ) |$ from the Roy equation at $s _ { 0 } = ( 0 . 8 \mathrm { G e V } ) ^ { 2 }$ and $s _ { 1 } = ( 1 . 1 5 \mathrm { G e V } ) ^ { 2 }$ across dataset variations (Sets A-D) and physics-loss ablation studies on Set A.
<table><tr><td rowspan="2">Configuration</td><td rowspan="2"> $\langle r _ { \pi } ^ { 2 } \rangle ~ ( \mathrm { f m } ^ { 2 } )$ </td><td rowspan="2"> $\langle r _ { \pi } ^ { 4 } \rangle ~ ( \mathrm { f m ^ { 4 } } )$ </td><td rowspan="2"> $\langle r _ { \pi } ^ { 6 } \rangle \ ( \mathrm { f m } ^ { 6 } )$ </td><td colspan="3"> $a _ { \mu } ^ { \pi \pi } \times 1 0 ^ { 1 0 }$ </td><td colspan="2"> $| \Delta \delta _ { 1 } ^ { 1 } ( s ) |$  in degrees</td></tr><tr><td> $[ 2 m _ { \pi } , 0 . 6 3 ]$ </td><td> $[ 2 m _ { \pi } , 1 . 0 ]$ </td><td> $[ 2 m _ { \pi } , \infty ]$ </td><td> $s = s _ { 0 }$ </td><td> $s = s _ { 1 }$ </td></tr><tr><td colspan="7">Data ablation (baseline sets)</td><td colspan="2"></td></tr><tr><td>Set A (no CMD-3)</td><td>0.435</td><td>0.748</td><td>4.01</td><td>130.66</td><td>496.37</td><td>506.48</td><td>0.25</td><td>1.72</td></tr><tr><td>Set B (CMD-3 only)</td><td>0.447</td><td>0.738</td><td>3.335</td><td>137.71</td><td>520.08</td><td>532.27</td><td>0.08</td><td>5.08</td></tr><tr><td>Set  $\mathrm { C _ { r a w } } \left( \tau { \cdot } \mathrm { d e c a y } \right)$ </td><td>0.421</td><td>0.665</td><td>2.709</td><td>131.36</td><td>502.34</td><td>512.79</td><td>1.15</td><td>6.65</td></tr><tr><td>Set  $\mathrm { C } _ { \mathrm { R _ { I B } } } \left( \tau { \cdot } \mathrm { d e c a y } \right)$ </td><td>0.437</td><td>0.817</td><td>5.458</td><td>127.40</td><td>492.51</td><td>502.83</td><td>0.19</td><td>6.28</td></tr><tr><td>Set D (all combined)</td><td>0.421</td><td>0.655</td><td>2.515</td><td>132.69</td><td>504.76</td><td>515.30</td><td>0.42</td><td>3.46</td></tr><tr><td> $\lambda _ { \mathrm { s p a c e l i k e } } = 0 ( \mathrm { S e t } \mathrm { A } )$ </td><td>0.436</td><td>0.786</td><td>4.486</td><td>132.67</td><td>496.12</td><td>505.68</td><td>0.19</td><td>0.60</td></tr><tr><td> $\lambda _ { \mathrm { t i m e l i k e } } = 0 ( \mathsf { S e t A } )$ </td><td>0.392</td><td>2.075</td><td>31.560</td><td>17.72</td><td>20.18</td><td>20.83</td><td>55.99</td><td>116.20</td></tr><tr><td colspan="9">Physics-loss ablation (trained on Set A)</td></tr><tr><td>Set A baseline</td><td>0.435</td><td>0.748</td><td>4.01</td><td>130.66</td><td>496.37</td><td>506.48</td><td>0.25</td><td>1.72</td></tr><tr><td> $\lambda _ { \mathrm { a n a l y t i c i t y } } = 0$ </td><td>0.431</td><td>0.700</td><td>2.956</td><td>132.76</td><td>496.07</td><td>506.48</td><td>0.90</td><td>2.47</td></tr><tr><td> $\lambda _ { \mathrm { d i s p e r s i o n } } = 0$ </td><td>0.448</td><td>0.836</td><td>5.464</td><td>130.65</td><td>498.28</td><td>508.28</td><td>0.36</td><td>2.91</td></tr><tr><td> $\lambda _ { \mathrm { m o m e n t } } = 0$ </td><td>0.436</td><td>0.705</td><td>12.701</td><td>130.77</td><td>497.37</td><td>507.33</td><td>0.19</td><td>1.90</td></tr><tr><td> $\lambda _ { \mathrm { p o s i t i v i t y } } = 0$ </td><td>0.436</td><td>0.758</td><td>4.053</td><td>130.04</td><td>497.36</td><td>507.54</td><td>0.34</td><td>2.72</td></tr><tr><td> $\lambda _ { \mathrm { W a t s o n } } = 0$ </td><td>0.546</td><td>1.337</td><td>11.188</td><td>135.01</td><td>502.78</td><td>512.73</td><td>7.70</td><td>11.45</td></tr><tr><td> $\lambda _ { \mathrm { m o n o t o n i c i t y } } = 0$ </td><td>0.438</td><td>0.759</td><td>4.119</td><td>129.80</td><td>496.68</td><td>506.67</td><td>0.77</td><td>2.20</td></tr><tr><td> $\lambda _ { \mathrm { p Q C D } } = 0$ </td><td>0.435</td><td>0.745</td><td>3.986</td><td>130.82</td><td>495.01</td><td>505.11</td><td>0.28</td><td>3.75</td></tr><tr><td> $\lambda _ { \mathrm { t - a s y m p t o t i c } } = 0$ </td><td>0.431</td><td>0.729</td><td>3.746</td><td>130.49</td><td>495.46</td><td>505.76</td><td>0.42</td><td>5.19</td></tr><tr><td> $\lambda _ { p h y s } = 0$ </td><td>0.445</td><td>-0.640</td><td>-57.210</td><td>131.74</td><td>498.50</td><td>509.27</td><td>43.60</td><td>57.20</td></tr><tr><td>Padé approximation</td><td>0.453</td><td>0.834</td><td>4.758</td><td>130.14</td><td>492.52</td><td>502.30</td><td>0.48</td><td>3.33</td></tr></table>

![](images/9d481bf389ba2f5ce92e570b88ae980470ac8d40874ecc1a11281539ec422ac7.jpg)  
(a) Set $\mathrm { C } _ { \mathrm { r a w } }$

![](images/5742a36bcf49e22b6a914343a0ce4b38ea28b07384326f01e7049c4043625500.jpg)  
(b) Set $\mathrm { C } _ { R _ { \mathrm { I B } } }$  
FIG. 11. Reconstruction of $| F _ { \pi } ( s ) | ^ { 2 }$ using τ-decay data in the timelike region under two distinct treatment schemes: (a) Set $C _ { \mathrm { r a w } } ,$ where raw τ-decay data are directly fitted while isolating the pure $I = 1$ isovector form factor $F _ { \pi } ^ { I = 1 } ( s )$ by setting the $\rho - \omega$ mixing parameter to zero $( \alpha _ { \rho - \omega } = 0 ) ;$ ; and (b) Set $C _ { R _ { \mathrm { I B } } } .$ , where raw τ data are explicitly pre-corrected for isospin-breaking efects using the point-by-point $R _ { I B } ( s )$ factor before training. Both panels display the resulting PINN predictions overlaid against experimental data points from BELLE and CLEO.

If we switch of all physics constraints $( \lambda _ { p h y s } = 0 )$ we get a purely data-driven model. However, it suffers a complete physical breakdown even though it easily fits discrete timelike data points by introducing highfrequency numeric oscillations. While these local wrinkles average out in broad spectral integrals like $a _ { \mu } ^ { \pi \pi }$ , taking sequential derivatives magnifies them catastrophically, forcing $\langle r _ { \pi } ^ { 4 } \rangle$ and $\langle r _ { \pi } ^ { 6 } \rangle$ to collapse to unphysical negative numbers and destroying the phase alignment completely. Incorporating physics losses is therefore indispensable: they act as non-local regularisers that guarantee holomorphy, enforce unitarity, and yield reliable physical derivatives that unconstrained interpolations cannot achieve.

The tension in the data: We test the roles of diferent datasets similarly through ablations. Removing spacelike data $( \lambda _ { \mathrm { s p a c e l i k e } } = 0 )$ causes minor degradation in the low-energy moments, while removing timelike data $( \lambda _ { \mathrm { t i m e l i k e } } = 0 )$ causes a catastrophic collapse in $a _ { \mu } ^ { \pi \pi }$ and a massive phase shift $( \sim 1 1 6 ^ { \circ } )$ , reflecting the dominant role of the timelike dataset near the $\rho ( 7 7 0 )$ resonance in fixing the form factor.

Training on various data subsets exposes the wellknown tension across experiments. Training exclusively on CMD-3 data along with the spacelike data (Set B) pushes $a _ { \mu } ^ { \pi \pi } [ 2 m _ { \pi } , \infty ]$ to $5 3 2 . 2 7 \times 1 0 ^ { - 1 0 }$ from our baseline value (Set A: $5 0 6 . 4 8 \times 1 0 ^ { - 1 0 } )$ This correlates with the higher cross-section normalisation reported by CMD-3 near the $\rho ( 7 7 0 )$ resonance peak, which also induces a noticeable phase shift (5.08<sup>◦</sup> at $s _ { 1 } = ( 1 . 1 5 \mathrm { G e V } ) ^ { 2 } )$ relative to the Roy equation solution. This indicates that the CMD-3 data creates a subtle structural tension with elastic unitarity and dispersion constraints at higher energies, forcing the model to distort its phase slope to accommodate the excess spectral weight.

We next evaluate the network’s sensitivity to τ-decay data via Set C (Fig. 11). In Set $C _ { \mathrm { r a w } ; }$ the PINN isolates the pure $I = 1$ form factor $F _ { \pi } ^ { I = 1 } ( s )$ natively by setting $\alpha _ { \rho - \omega } =$ 0 (using the channel switch), bypassing external precorrections. Retaining this unsuppressed $I = 1$ normalisation yields $a _ { \mu } ^ { \pi \pi } [ 2 m _ { \pi } , \infty ] = 5 1 2 . 7 9 \times 1 0 ^ { - 1 0 }$ , demonstrating that embedding channel-aware parameterisations within the PINN provides a self-consistent alternative to modeldependent point-by-point factors. Conversely, in Set $C _ { R _ { \mathrm { I B } } } ,$ the point-by-point $R _ { \mathrm { I B } } ( s )$ pre-corrections [Eq. 39] disrupt local derivative structures and distort the resonance curvature. Enforcing S-matrix analyticity and dispersion relations causes the physics loss to smooth through these gradient mismatches, slightly underfitting the peak near $\overline { { s } } \approx 0 . 6 \ : \mathrm { G e V } ^ { 2 }$ (Fig. 11b) and pulling $a _ { \mu } ^ { \pi \pi }$ down to $5 0 2 . 8 3 \times 1 0 ^ { - 1 0 }$ . Due to reduced high-mass precision in τ data, both Set C configurations exhibit larger phase shift mismatches $( | \Delta \delta _ { 1 } ^ { 1 } ( s _ { 1 } ) | = 6 . 6 5 ^ { \circ }$ for Set $C _ { \mathrm { r a w } }$ and $6 . 2 8 ^ { \circ }$ for Set $C _ { R _ { \mathrm { I B } } } )$ compared to Set $\mathrm { ~ A ~ } ( 1 . 7 2 ^ { \circ } )$ , underscoring the reliance on high-energy $e ^ { + } e ^ { - }$ data for phase consistency above 1 $\mathrm { G e V } ^ { 2 }$

In Set D, which synthesises all $e ^ { + } e ^ { - }$ and raw τ data streams via the conditional channel switch, the PINN acts as an analytical mediator. The network integrates the conflicting CMD-3 and $e ^ { + } e ^ { - } / \tau$ normalizations to yield a balanced HVP contribution of $a _ { \mu } ^ { \pi \pi } [ 2 m _ { \pi } , \infty ] = 5 1 5 . 3 0 \times$ $1 0 ^ { - 1 0 }$ . This upward shift relative to Set $\mathrm { ~ A ~ } ( 5 0 6 . 4 8 \times 1 0 ^ { - 1 0 } )$ reflects how raw τ normalisation and CMD-3 data jointly pull the global fit toward higher spectral density. Despite the added input tension, Set D maintains strong physical self-consistency, achieving a high-energy phase shift deviation of $| \Delta \delta _ { 1 } ^ { 1 } ( s _ { 1 } ) | = 3 . 4 \bar { 6 } ^ { \circ }$ . This is an improvement over standalone τ sets $( > 6 ^ { \circ } )$ , demonstrating how $e ^ { + } e ^ { - }$ highenergy precision stabilises S-matrix analyticity while accommodating raw τ dynamics natively.

## VI. CLOSED-FORM APPROXIMATION OF THE FORM FACTOR

Although the PINN gives a numerically stable, modelindependent form factor, its output exists only as a trained network, not as an expression that can be quoted or used in other calculations. We find a compact conformal Padé fraction in $z = z ( s )$ that closely follows the PINN prediction near the real axis and the $\rho ( 7 7 0 )$ pole, useful for phenomenological calculations and further model building,

$$
F _ { \pi } ( s ) = { \frac { 1 + a _ { 1 } z + a _ { 2 } z ^ { 2 } + a _ { 3 } z ^ { 3 } + a _ { 4 } z ^ { 4 } } { \left( 1 - { \frac { z } { z _ { \rho } } } \right) \left( 1 - { \frac { z } { z _ { \rho } ^ { * } } } \right) \left( 1 + b _ { 1 } z + b _ { 2 } z ^ { 2 } \right) } } ,\tag{44}
$$

with best-fit coeficients

$$
\begin{array} { r l r l } & { a _ { 1 } = - 3 . 5 2 7 6 4 7 , } & & { a _ { 2 } = 4 . 7 5 5 6 4 5 , } \\ & { a _ { 3 } = - 2 . 9 3 7 7 0 9 , } & & { a _ { 4 } = 0 . 7 0 7 6 6 0 , } \\ & { b _ { 1 } = - 1 . 5 5 4 0 4 8 , } & & { b _ { 2 } = 0 . 7 5 8 9 7 5 , } \\ & { z _ { \rho } = 0 . 7 9 0 3 5 3 - 0 . 7 2 8 0 3 3 i . } \end{array}
$$

The complex-conjugate pole pair $z _ { \rho } , z _ { \rho } ^ { \ast }$ is factored out explicitly as $( 1 - z / z _ { \rho } ) ( 1 - z / z _ { \rho } ^ { * } )$ in the denominator. This structural choice exactly enforces the Schwarz reflection property of the network output (any pole at $z _ { \rho }$ is automatically accompanied by its mirror pole at $z _ { \rho } ^ { \ast } )$ . The remaining quadratic $\left( 1 + b _ { 1 } z + b _ { 2 } z ^ { 2 } \right)$ supplies the additional rational structure needed to match the network beyond the leading resonance behaviour. The fit converges to all four denominator roots lying outside the unit disk $( | z | > 1 )$ ). This means the approximant is automatically holomorphic throughout $| z | \le 1$ , consistent with the analyticity enforced by $\mathcal { L } _ { \mathrm { a n a l y t i c i t y } }$ during training, without this having been enforced by hand.

We show in Figs. 12a, 12b and $^ { 1 3 , }$ how well Eq. (44) reproduces the full network prediction. Each panel shows the discrepancy between the full PINN prediction and its Padé approximation over the complex s-plane as both a 3D surface (left) and a 2D heat map (right). Fig. 12a shows the absolute diference in magnitude, $\left| | \mathsf { \bar { F } } _ { \pi } ^ { \mathrm { P I N N } } ( s ) | - | F _ { \pi } ^ { \mathrm { P a d } \acute { e } } ( s ) | \right|$ , with the heat map colour-scaled to the relative error. Fig. 12b shows the corresponding diference in phase, $\left| \overset { \ J } { \operatorname { A r g } } ( F _ { \pi } ^ { \mathrm { P I N N } } ) - \operatorname { A r g } ( F _ { \pi } ^ { \mathrm { P a d } \acute { \mathrm { e } } } ) \right|$ , in degrees. Both quantities remain small near the real axis and around the $\rho ( 7 7 0 )$ pole, which is the region of interest for most physical calculations; the discrepancy grows to several per cent in magnitude and a few degrees in phase further into the complex plane.

We next show a direct comparison of the form factor along the physical kinematic line in Fig. 13. The form factor $| F _ { \pi } ( s ) |$ from the PINN (solid black) and from the Padé approximant (red dashed) are overlaid on a logarithmic scale across both the spacelike $( s < 0 )$ and timelike $\textstyle ( s > 4 m _ { \pi } ^ { 2 }$ , shaded pink) regions. The two curves are visually indistinguishable through the $\rho ( 7 7 0 )$ peak, with a small, growing separation appearing only at the largest timelike s shown, confirming that the closed-form expression faithfully reproduces the network’s behaviour where it matters most for the extracted observables.

(a)  
![](images/e0fad9488df58fabeeef7fdc214e144fad5f3789b26cd72f3e6cc218134a869d.jpg)

![](images/410c853c56963afe239c9cf214b352dbcc8fa72a0177773e29fb6448d78857c0.jpg)

![](images/2e71faa8863a2246d77de6abf1930fd9a2a2ca91ed142c0a2e289a68bc9e4347.jpg)

![](images/23af3f007610d9d0ae799e4b9cc1198a85d06b2387f5ddecdca4be7eda94338c.jpg)  
(b) <sup></sup>Arg(F<sup>PINN</sup><sub>π</sub> ) − Arg(F <sup>Padé</sup><sub>π</sub> ) <sup></sup>  
FIG. 12. Discrepancy between the full PINN prediction and its [4,4] Padé approximation across the complex s-plane. (a) Absolute diference in magnitude, $\left| | F _ { \pi } ^ { \mathrm { P I N N } } ( s ) | - | F _ { \pi } ^ { \mathrm { P a d } \acute { \mathrm { e } } } ( \acute { s } ) | \right|$ : the 3D surface (left) zooms into the region marked by the yellow box in the 2D heat map (right), which shows the relative error over the full sampled domain. (b) Corresponding phase diference, $\left| \mathrm { A r g } ( F _ { \pi } ^ { \mathrm { P I N N } } ) \right.$ $\mathrm { A r g } ( F _ { \pi } ^ { \mathrm { \tiny ~ P a d \acute { e } } } ) \vert$ in degrees, with the same left/right correspondence.

Because the second sheet is reached from the first via $z _ { I I } ( s ) = 1 / z _ { I } ( s )$ , the fitted pole $z _ { \rho }$ maps directly onto the $\rho ( 7 7 0 )$ pole position by evaluating the inverse conformal map at $z _ { I I , \rho } = 1 / z _ { I , \rho }$

$$
s _ { \rho } = 4 m _ { \pi } ^ { 2 } \left[ 1 - \left( \frac { 1 + z _ { \rho } ^ { - 1 } } { 1 - z _ { \rho } ^ { - 1 } } \right) ^ { 2 } \right] = 0 . 5 7 3 7 - 0 . 1 0 6 5 i \mathrm { G e V } ^ { 2 } .
$$

This gives us,

$$
\begin{array} { c } { { \sqrt { s _ { \rho } } = \left( 7 6 0 . 7 - \frac { i } { 2 } 1 4 0 . 1 \right) \ \mathrm { { M e V } } } } \\ { { \Rightarrow \ m _ { \rho } = 7 6 0 . 7 \ \mathrm { { M e V } } , \quad \Gamma _ { \rho } = 1 4 0 . 1 \ \mathrm { { M e V } } . } } \end{array}
$$

This is an independent extraction, obtained purely from the algebraic pole of the closed-form fit rather than from the Newton–Raphson root search on the full network output (Sec. V D). It agrees with the direct extraction to within ∼ 0.1% in mass and ∼ 3% in width. The residual diference reflects the finite order of the Padé truncation rather than any inconsistency in the underlying network, and confirms that a low-order rational function already captures the resonance analytic structure faithfully.

To systematically select the appropriate order for this closed-form surrogate, we refit conformal Padé approximants of orders [1,1] through [6,6] directly to the PINN output and track the mean relative deviation on the timelike cut. The deviation drops rapidly from 124% at [1,1] to 7.1% at [2,2] and 3.4% at [4,4], after which it saturates (3.4% at [5, 5], 3.2% at [6, 6]); the spacelike axis and complex plane exhibit a matching plateau (2.8–3.3% and

![](images/9fafeca20513066f67f443228e81dc28c376533719c853a233f58b38306d8fab.jpg)  
FIG. 13. Comparison of $| F _ { \pi } ( s ) |$ from the full PINN (solid black) and the [4/4] Padé approximant (red dashed) along the physical axis, across the spacelike and timelike regions. The two curves are visually indistinguishable through the $\rho ( 7 7 0 )$ peak.

2.8–3.1%, respectively). The extracted resonance parameters likewise stabilise from [4,4] onward. We therefore select the $[ 4 / 4 ]$ order as the minimal approximant that fully saturates the achievable accuracy, balancing mathematical compactness against numerical fidelity.

Evaluating the physical observables directly from the $[ 4 / 4 ]$ Padé approximant yields $a _ { \mu } ^ { \pi \pi }$ values in close agreement with the continuous PINN baseline (the last line of Table IV), though the higher moments $\langle r _ { \pi } ^ { n } \rangle$ and the phase deviation at $s _ { 1 }$ show somewhat larger residual shifts. This indicates that while the PINN solves the ill-posed inverse problem by mapping noisy, multi-channel experimental data into a smooth, physically constrained representation across the real energy axis, the Padé approximant acts as an analytic tool that compresses this continuous representation into a rational form, granting direct access to unphysical-sheet resonance poles and low-energy derivative expansions. The slight shift in the low-energy moments $( \langle r _ { \pi } ^ { n } \rangle )$ reflects the minimal-order truncation redistributing far-UV asymptotic tails, confirming that the Padé fit serves as an analytic bridge from the PINN’s continuous representation to exact S-matrix properties, rather than as a solver in its own right.

## VII. DISCUSSION AND FUTURE DIRECTIONS

We have presented a Physics-Informed Neural Network embedded in a conformal z-plane that extracts the pion electromagnetic form factor $F _ { \pi } ( s )$ across the full spacelike and timelike spectrum directly from first principles, rather than from a chosen functional template. The conformal mapping resolves a structural pathology we identify in standard neural-network training on unbounded kinematic variables. Embedding charge normalisation, analyticity, dispersion relations, and Watson’s theorem into the loss functional then yields a form factor free of complex zeros without this being separately imposed, along with precise extractions of the pion charge radius, the $\rho ( 7 7 0 )$ pole, and $a _ { \mu } ^ { \pi \pi }$ (Secs. III–V).

A few points are worth discussing further. First, our default result excludes CMD-3: the network’s global holomorphy and dispersion constraints fit CMD-3 less comfortably than BaBar and τ-decay data, evidence bearing on the ongoing tension between these measurements, though the tension is not resolved by this framework alone. Second, the extracted pole width sits 4–8% below the PDG world average, a small but consistent deficit likely tied to the elastic-unitarity approximation used to reach the second sheet; resolving it will need an explicit treatment of inelastic channels. Third, the zero-free structure and the CMD-3/BaBar preference both emerge without being directly targeted by any loss term, which is the kind of result most useful for adjudicating between datasets, but it should be read as evidence from one particular architecture and constraint set, not as an independent physical proof.

Extending this framework to multi-channel form factors (such as $\pi \pi \to K \bar { K }$ and $\pi \pi \to \omega \pi ^ { 0 } )$ would let it capture inelastic efects beyond $1 \mathsf { G e V } ^ { 2 }$ directly, addressing the width deficit noted above. A unified treatment of momentum-dependent $\rho { - } \omega$ mixing and $\pi ^ { \pm } \mathrm { - } \pi ^ { 0 }$ mass splitting within the network would further reconcile the τ-decay and $e ^ { + } e ^ { - }$ datasets without relying on external isospin-breaking corrections. Finally, adapting the conformal PINN scheme to the pion transition form factor $F _ { \pi ^ { 0 } \gamma ^ { * } \gamma ^ { * } }$ would provide a model-independent baseline for the dominant remaining theoretical uncertainty in Hadronic Light-by-Light scattering for $( g - 2 ) _ { \mu }$

## SUPPLEMENTARY MATERIAL

The code, model weights & plotting scripts are all available at our GitHub repository.

## ACKNOWLEDGMENTS

We thank B. Ananthanarayan for insightful discussions and valuable suggestions during the development of this work.

## Appendix A: Proofs of theorems

We prove the NTK rank theorem from Section III.

Theorem 1 (Full rank of the empirical NTK). Let $z _ { 1 } , \dots , z _ { N }$ be distinct points with $\left| z _ { i } \right| \leqslant 1 ,$ , and let $f ( z ; \theta )$ be a feedforward neural network with a real analytic, nonpolynomial activationfunction σ and a total of N hidden neurons across all layers. If $\mathcal { N } \geqslant N ,$ then for a generic choice ofparameters $\theta ,$ , the parameter Jacobian $J _ { z } \in \mathbb { R } ^ { N \times P }$ has row rank $N ,$ and the empirical NTK $\Theta _ { z } = J _ { z } J _ { z } ^ { T } / P$ is positive definite: $\lambda _ { \operatorname* { m i n } } ( \Theta _ { z } ) > 0 ;$ , where P is the total number of parameters, satisfying $P \geqslant { \mathcal { N } }$ since every neuron carries at least one parameter.

Proof. It is enough to show that there exists at least one parameter choice for which $J _ { z }$ has row rank N. Since the entries of $J _ { z }$ are analytic functions of the parameters, the vanishing of every $N \times N$ minor defines an analytic variety. If one such minor is not identically zero, then its zero set has an empty interior and measure zero. Hence, full row rank holds generically. Since $\Theta _ { z } = J _ { z } J _ { z } ^ { T } / P$ is positive definite exactly when $J _ { z }$ has full row rank, positivedefiniteness of the NTK then also holds generically.

Let us first consider a network with a single hidden layer of $\mathcal { N }$ neurons, $\begin{array} { r l } { f ( z ; \theta ) = \sum _ { k = 1 } ^ { \mathcal { N } } \nu _ { k } \sigma ( a _ { k } z + b _ { k } ) } & { { } } \end{array}$ . If we fix $\{ a _ { k } , b _ { k } \} _ { k = 1 } ^ { \mathcal { N } }$ at generic values, so that the $\mathcal { N }$ numbers $a _ { k } z _ { i } + b _ { k }$ are pairwise distinct across neurons, the classical interpolation result for non-polynomial activations [58] tells us that, once $\mathcal { N } \geqslant N$ , the outer weights $\left\{ \nu _ { k } \right\}$ alone can be chosen so that the feature matrix $\Phi _ { i k } = \sigma ( a _ { k } z _ { i } +$ $b _ { k } )$ has rank N. Since $\partial _ { \nu _ { k } } f ( z _ { i } ; \theta ) = \Phi _ { i k }$ , the Jacobian $J _ { z }$ contains Φ as a submatrix, so $J _ { z }$ already has row rank N at this parameter choice, establishing the theorem for a network with a single hidden layer.

To extend this to a network of any depth, the key observation is that adding more layers cannot reduce the freedom available to the network. Once one layer of neurons already gives the network some rank, adding a further layer that does nothing to the output leaves the rank untouched. Since its own neurons are also free to add further independent directions on top, the rank only increases. We now make this precise.

Let us express a network with L hidden layers as $h ^ { ( 0 ) } ( z ) = z$ and $h ^ { ( l ) } ( z ) = \sigma ( W ^ { ( l ) } h ^ { ( l - 1 ) } ( z ) + b ^ { ( l ) } )$ for $l =$ $1 , \ldots , L ,$ with output $\begin{array} { r } { f ( z ; \theta ) = \sum _ { k = 1 } ^ { \mathcal { N } _ { L } } \nu _ { k } h _ { k } ^ { ( L ) } ( z ) } \end{array}$ , where $\mathcal { N }$ is the width of the last hidden layer. Let $\mathcal { N } _ { \leqslant l }$ denote the number of neurons in the first l layers, and let $J _ { z } ^ { ( \leqslant l ) }$ be the Jacobian restricted to the parameters of these layers. Suppose that, at some parameter choice, $J _ { z } ^ { ( \leqslant l ) }$ has rank $r = \operatorname* { m i n } ( \mathcal { N } _ { \leqslant l } , N )$ . If we now attach layer $l + 1$ and tune its weights close to the values that make $h ^ { ( l + 1 ) } ( z )$ simply reproduce $h ^ { ( l ) } ( z )$ , the rank already achieved is preserved, since the network at this point is only a small, generic deformation of the one we started with, and rank cannot drop under such a deformation. If r is still less than $N ,$ we can then apply the single-layer argument above once more, this time to layer $l + 1 \ ' s$ own weights and to the inputs $h ^ { ( l ) } ( z _ { i } )$ , which are generically distinct; this supplies further independent columns, raising the rank up to min $( \mathcal { N } _ { \leqslant l + 1 } , N )$ . Starting from $l = 1$ and repeating this step up to $l = L ,$ we find that a generic choice of θ gives $J _ { z }$ rank min $( \mathcal { N } , N )$ , where $\mathcal { N } = \mathcal { N } _ { \leqslant L }$ is the total neuron count of the network. Once $\mathcal { N } \geqslant N$ , this is exactly the row rank N we set out to prove. □

The condition $\mathcal { N } \geqslant N$ above only makes use of each neuron’s outer weight, and is therefore a conservative one: for the sine activation we used, a single neuron in fact carries more than one useful direction. We illustrate this with a simple example.

Let us take a single neuron, $f ( z ; \theta ) = \nu \sin ( a z + b )$ , and three distinct points $z _ { 1 } , z _ { 2 } , z _ { 3 }$ . Its three parameters contribute the columns $\partial _ { \nu } f = \sin ( a z _ { i } + b ) , \partial _ { b } f = \nu \cos ( a z _ { i } + b )$ and $\partial _ { a } f = \nu z _ { i } \cos ( a z _ { i } + b )$ . Note that a cannot be zero here, since then $\partial _ { \nu } f$ and $\partial _ { b } f$ both become constant across $i ,$ making the resulting $3 \times 3$ determinant vanish identically. If instead we take a small but non-zero and $b = 0 ,$ expanding the determinant in powers of a gives, to lead-

ing order,

$$
D ( a , 0 ) = - { \frac { \nu } { 3 } } a ^ { 3 } \operatorname* { d e t } { \left( \begin{array} { l l l } { z _ { 1 } } & { 1 } & { z _ { 1 } ^ { 3 } } \\ { z _ { 2 } } & { 1 } & { z _ { 2 } ^ { 3 } } \\ { z _ { 3 } } & { 1 } & { z _ { 3 } ^ { 3 } } \end{array} \right) } + { \mathcal O } ( a ^ { 5 } ) .\tag{A1}
$$

The determinant on the right is not identically zero, so it can only vanish for special, symmetric choices of the three points. For any other choice, $D ( a , 0 ) \neq 0 .$ , and the single neuron already supplies rank 3 from its three parameters. The same mechanism should extend to larger N. Our network used $\mathcal { N } = 5 1 2$ neurons spread over four hidden layers, comfortably above the size of any collocation batch N used in training, so the conservative bound already covers our case.

Theorem 2 (Boundedness of Hessian in ${ \mathcal { Z } } { \mathrm { - } } s { \mathrm { p a c e } } )$ . For $\begin{array} { r l } { f ( z ; \theta ) = \sum _ { k = 1 } ^ { K } \nu _ { k } \sigma ( a _ { k } z + b _ { k } ) } & { { } } \end{array}$ with $| z | \leqslant 1 .$ , assume that the activation satisfies,

$$
| \sigma ( u ) | \leqslant C _ { 0 } , | \sigma ^ { \prime } ( u ) | \leqslant C _ { 1 } , | \sigma ^ { \prime \prime } ( u ) | \leqslant C _ { 2 } ,\tag{A2}
$$

for all $u \in \mathbb { R }$ . Assume further that along the optimisation trajectory, $| \nu _ { k } | \leqslant V ,$ and that the data labels are bounded $\left| y _ { i } \right| \leqslant Y .$ Then the Hessian of the MSE loss is uniformly bounded $\| H \| _ { 2 } \le C _ { H } < \infty ,$ where $C _ { H }$ is independent of the original physical coordinate s.

Proof. Following Eq. (11), the Hessian is

$$
H = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \nabla _ { \theta } f ( z _ { i } ; \theta ) \nabla _ { \theta } f ( z _ { i } ; \theta ) ^ { T } + \frac { 1 } { N } \sum _ { i = 1 } ^ { N } r _ { i } ( \theta ) \nabla _ { \theta } ^ { 2 } f ( z _ { i } ; \theta ) .\tag{A3}
$$

We bound both terms separately. First, using $| z | \leqslant 1$ and the boundedness assumptions on the parameter derivatives, we get,

$$
\left| \partial _ { \nu _ { k } } f \right| \leqslant C _ { 0 } , \ \left| \partial _ { a _ { k } } f \right| \leqslant V C _ { 1 } , \ \mathrm { a n d } \ \left| \partial _ { b _ { k } } f \right| \leqslant V C _ { 1 } .\tag{A4}
$$

Therefore there exists a constant $M _ { 1 } <$ ∞ such that

$$
\| \nabla _ { \theta } f ( z _ { i } ; \theta ) \| _ { 2 } \leqslant M _ { 1 } ,\tag{A5}
$$

for every data point i. Now, because $| \sigma ^ { \prime \prime } | \leqslant C _ { 2 }$ and the input and the parameters are bounded, every second parameter derivative of $f$ is bounded by a constant depending only on $V , C _ { 1 } , C _ { 2 }$ . Hence, there exists $M _ { 2 } < \infty$ such that

$$
\| \nabla _ { \theta } ^ { 2 } f ( z _ { i } ; \theta ) \| _ { 2 } \leqslant M _ { 2 } .\tag{A6}
$$

It remains to bound the residuals. Since

$$
| f ( z _ { i } ; \theta ) | \leqslant \sum _ { k = 1 } ^ { K } | \nu _ { k } | | \sigma ( a _ { k } z _ { i } + b _ { k } ) | \leqslant K V C _ { 0 } ,\tag{A7}
$$

and $| y _ { i } | \leqslant Y ,$ , we have

$$
\vert r _ { i } ( \theta ) \vert \leqslant K V C _ { 0 } + Y \equiv R _ { \mathrm { m a x } } .\tag{A8}
$$

Therefore,

$$
\begin{array} { r l r } {  { \| H \| _ { 2 } \leqslant \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \| \nabla _ { \theta } f ( z _ { i } ; \theta ) \| _ { 2 } ^ { 2 } + \frac { 1 } { N } \sum _ { i = 1 } ^ { N } | r _ { i } ( \theta ) | \| \nabla _ { \theta } ^ { 2 } f ( z _ { i } ; \theta ) \| _ { 2 } } } \\ & { } & { \leqslant M _ { 1 } ^ { 2 } + R _ { \operatorname* { m a x } } M _ { 2 } \equiv C _ { H } < \infty . \qquad ( \& { \mathrm { o } } ) } \end{array}
$$

The bound depends on the network-size and weight bounds, but not on the original physical coordinate s.

Appendix B: The NLO pQCD constraint for the asymptotic loss functional

To anchor the PINN in the deep Euclidean region $( Q ^ { 2 } =$ $- s \to \infty )$ , we enforce the NLO pQCD prediction for the pion electromagnetic form factor. Following Melić et al. [19], the NLO form factor can be written as:

$$
\begin{array} { r l r } { \quad } & { } & { F _ { \pi } ( Q ^ { 2 } , \mu _ { R } ^ { 2 } , \mu _ { F } ^ { 2 } ) = F _ { \pi } ^ { ( 0 ) } ( Q ^ { 2 } , \mu _ { R } ^ { 2 } , \mu _ { F } ^ { 2 } ) + F _ { \pi } ^ { ( 1 a ) } ( Q ^ { 2 } , \mu _ { R } ^ { 2 } , \mu _ { F } ^ { 2 } ) } \\ & { } & { + F _ { \pi } ^ { ( 1 b ) } ( Q ^ { 2 } , \mu _ { R } ^ { 2 } , \mu _ { F } ^ { 2 } ) , \qquad ( { \mathrm { B } } } \end{array}\tag{1}
$$

where $\mu _ { F }$ is the factorisation scale, $F _ { \pi } ^ { ( 0 ) }$ is the leadingorder (LO) form factor, $F _ { \pi } ^ { ( 1 a ) }$ , the NLO correction to the hard-scattering amplitude, and $F _ { \pi } ^ { ( 1 b ) }$ , the NLO evolutional correction to the pion DA. To extract a closed-form analytic constraint for our loss functional, we choose the asymptotic DA [19] to model the pion, $\phi _ { a s } ( x ) = 6 x ( 1 - x )$ ， where x is the longitudinal momentum fraction carried by the quark. For the asymptotic DA, the $\mu _ { F }$ dependence disappears and the NLO evolutional correction $F _ { \pi } ^ { ( 1 b ) }$ becomes negligible (∼ 1%). We can thus safely omit it when establishing a deep-spacelike boundary. Under this asymptotic assumption, the LO and the net NLO contri-

[1] G. W. Bennett et al. (Muon $g { - } 2 ) _ { : }$ , “Final Report of the Muon E821 Anomalous Magnetic Moment Measurement at BNL,” Phys. Rev. D 73, 072003 (2006), arXiv:hepex/0602035.

[2] B. Abi et al. (Muon g-2), “Measurement of the Positive Muon Anomalous Magnetic Moment to 0.46 ppm,” Phys. Rev. Lett. 126, 141801 (2021), arXiv:2104.03281 [hepex].

[3] Sz. Borsanyi et al., “Leading hadronic contribution to the muon magnetic moment from lattice QCD,” Nature 593, 51–55 (2021), arXiv:2002.12347 [hep-lat].

[4] F. V. Ignatov et al. (CMD-3), “Measurement of the Pion Form Factor with CMD-3 Detector and its Implication to the Hadronic Contribution to Muon (g-2),” Phys. Rev. Lett. 132, 231903 (2024), arXiv:2309.12910 [hep-ex].

[5] J. Gasser and H. Leutwyler, “Chiral Perturbation Theory to One Loop,” Annals Phys. 158, 142 (1984); “Chiral Perturbation Theory: Expansions in the Mass of the Strange Quark,” Nucl. Phys. B 250, 465–516 (1985).

[6] G. J. Gounaris and J. J. Sakurai, “Finite width corrections to the vector meson dominance prediction for $\rho  e ^ { + } e ^ { - } , \dot  $ Phys. Rev. Lett. 21, 244–247 (1968).

[7] R. Omnes, “On the Solution of certain singular integral equations of quantum field theory,” Nuovo Cim. 8, 316– 326 (1958).

butions simplify to:

$$
\begin{array} { l } { { \displaystyle Q ^ { 2 } F _ { \pi } ^ { ( 0 ) } ( Q ^ { 2 } , { \mu } _ { R } ^ { 2 } ) = 8 \pi f _ { \pi } ^ { 2 } \alpha _ { S } ( { \mu } _ { R } ^ { 2 } ) , \eqno ( 8 2 ) } } \\ { { \displaystyle Q ^ { 2 } F _ { \pi } ^ { ( 1 ) } ( Q ^ { 2 } , { \mu } _ { R } ^ { 2 } ) \approx 8 f _ { \pi } ^ { 2 } \alpha _ { S } ^ { 2 } ( { \mu } _ { R } ^ { 2 } ) \left[ \frac { \beta _ { 0 } } { 4 } \ln \left( \frac { { \mu } _ { R } ^ { 2 } } { Q ^ { 2 } } \right) + 6 . 8 3 - \frac { n _ { f } } { 1 2 } \right] , } } \end{array}\tag{B3}
$$

where $n _ { f }$ denotes the number of active flavours. Melić et al. set $n _ { f } = 3$ to estimate their final results. However, since our spacelike data extends to $Q ^ { 2 }$ values above the charm and bottom thresholds, we adopt the fiveflavour scheme, setting $n _ { f } = 5$ and $\beta _ { 0 } = 1 1 - 2 n _ { f } / 3 =$ $2 3 / 3$ , and evaluate $\alpha _ { S } ( \mu _ { R } ^ { 2 } )$ using the corresponding fiveflavour renormalisation group evolution. Because the $\mu _ { F }$ dependence vanishes for the asymptotic DA, this scheme change is implemented directly with no additional corrections.

Summing the LO and NLO hard-scattering contributions, we obtain the scale-dependent NLO prediction:

$$
F _ { \pi } ( Q ^ { 2 } , \mu _ { R } ^ { 2 } ) \approx { \frac { 8 \pi f _ { \pi } ^ { 2 } \alpha _ { S } ( \mu _ { R } ^ { 2 } ) } { Q ^ { 2 } } } \left[ 1 + { \frac { \alpha _ { S } ( \mu _ { R } ^ { 2 } ) } { \pi } } \Big ( { \frac { \beta _ { 0 } } { 4 } } \ln \Big ( { \frac { \mu _ { R } ^ { 2 } } { Q ^ { 2 } } } \Big ) \right.\tag{4}
$$

As pointed out by Melić et al., setting $\mu _ { R } ^ { 2 } = Q ^ { 2 }$ is physically unsuited because the essential virtualities of the particles in the parton subprocess are considerably smaller than the overall momentum transfer $Q ^ { 2 }$ . Following the Brodsky-Lepage-Mackenzie procedure adopted by Melić et al., setting $\mu _ { R } ^ { 2 }$ to the characteristic virtualities of the parton subprocess yields the physical renormalisation scale $\mu _ { R } ^ { 2 } = \mu _ { \mathrm { B L M } } ^ { 2 } \approx Q ^ { 2 } / 2 1$ for the asymptotic distribution amplitude.

[8] S. M. Roy, “Exact integral equation for pion pion scattering involving only physical region partial waves,” Phys. Lett. B 36, 353–356 (1971).

[9] G. Colangelo, J. Gasser, and H. Leutwyler, “ππ scattering,” Nucl. Phys. B 603, 125–179 (2001), arXiv:hepph/0103088.

[10] Thomas P. Leplumey and Peter Stofer, “Dispersive analysis of the pion vector form factor without zeros,” (2025), arXiv:2501.09643 [hep-ph].

[11] Bernard Aubert et al. (BaBar), “Precise measurement of the e+ e- —> pi+ pi- (gamma) cross section with the Initial State Radiation method at BABAR,” Phys. Rev. Lett. 103, 231801 (2009), arXiv:0908.3589 [hep-ex].

[12] J. P. Lees et al. (BaBar), “Precise Measurement of the $e ^ { + } e ^ { - } \to \pi ^ { + } \pi ^ { - } \left( \gamma \right)$ Cross Section with the Initial-State Radiation Method at BABAR,” Phys. Rev. D 86, 032013 (2012), arXiv:1205.2228 [hep-ex].

[13] Enrique Ruiz Arriola, Pablo Sanchez-Puertas, and Wojciech Broniowski, “Hadronic form factors in QCD and the incompleteness problem in the time-like region,” in Excited QCD 2026 Workshop (2026) arXiv:2604.09185 [hepph].

[14] Diogo Buarque Franzosi et al., “Vector boson scattering processes: Status and prospects,” Rev. Phys. 8, 100071 (2022), arXiv:2106.01393 [hep-ph].

[15] Irinel Caprini, “Dispersive and chiral symmetry constraints on the light meson form-factors,” Eur. Phys. J. C 13, 471–484 (2000), arXiv:hep-ph/9907227.

[16] B. Ananthanarayan, Irinel Caprini, and Diganta Das, “Pion electromagnetic form factor at high precision with implications to $\bar { a } _ { \mu } ^ { \pi \pi }$ and the onset of perturbative QCD,” Phys. Rev. D 98, 114015 (2018), arXiv:1810.09265 [hepph].

[17] B. Ananthanarayan, G. Colangelo, J. Gasser, and H. Leutwyler, “Roy equation analysis of pi pi scattering,” Phys. Rept. 353, 207–279 (2001), arXiv:hepph/0005297.

[18] Stanley J. Brodsky and Glennys R. Farrar, “Scaling Laws at Large Transverse Momentum,” Phys. Rev. Lett. 31, 1153– 1156 (1973); “Scaling Laws for Large Momentum Transfer Processes,” Phys. Rev. D 11, 1309 (1975).

[19] B. Melić, B. Nižić, and K. Passek, “Complete next-toleading order perturbative QCD prediction for the pion form-factor,” Phys. Rev. D 60, 074004 (1999), arXiv:hepph/9802204; “On the PQCD prediction for the pion form-factor,” in 6th INT / Jlab Workshop on Exclusive and Semiexclusive Processes at High Momentum Transfer (1999) pp. 279–286, arXiv:hep-ph/9908510.

[20] Vincent Sitzmann, Julien N.P. Martel, Alexander W. Bergman, David B. Lindell, and Gordon Wetzstein, “Implicit neural representations with periodic activation functions,” in Advances in Neural Information Processing Systems, Vol. 33 (Curran Associates, Inc., 2020).

[21] Yurii Nesterov, “Introductory lectures on convex optimization: A basic course,” in Applied Optimization (Springer, New York, 2004).

[22] C. Glenn Boyd, Benjamin Grinstein, and Richard F. Lebed, “Constraints on form-factors for exclusive semileptonic heavy to light meson decays,” Phys. Rev. Lett. 74, 4603–4606 (1995), arXiv:hep-ph/9412324.

[23] Arthur Jacot, Franck Gabriel, and Clément Hongler, “Neural tangent kernel: Convergence and generalization in neural networks,” in Advances in Neural Information Processing Systems (NeurIPS), Vol. 31 (2018) arXiv:1806.07572 [cs.LG].

[24] S. R. Amendolia et al. (NA7), “A Measurement of the Space - Like Pion Electromagnetic Form-Factor,” Nucl. Phys. B 277, 168 (1986).

[25] E. B. Dally et al., “Measurement of the π<sup>−</sup> Form-factor,” Phys. Rev. D 24, 1718–1735 (1981).

[26] C. N. Brown, C. R. Canizares, W. E. Cooper, A. M. Eisner, G. J. Feldmann, C. A. Lichtenstein, L. Litt, W. Loceretz, V. B. Montana, and F. M. Pipkin, “Coincidence electroproduction of charged pions and the pion form-factor,” Phys. Rev. D 8, 92–135 (1973).

[27] C. J. Bebek et al., “Further measurements of forwardcharged-pion electroproduction at large κ<sup>2</sup>,” Phys. Rev. D 9, 1229–1242 (1974).

[28] H. P. Blok et al. (Jeferson Lab), “Charged pion form factor between $Q ^ { 2 } { = } 0 . 6 0$ and 2.45 GeV<sup>2</sup>. I. Measurements of the cross section for the $^ 1 \mathrm { H } ( e , e ^ { \prime } \pi ^ { + } )$ n reaction,” Phys. Rev. C 78, 045202 (2008), arXiv:0809.3161 [nucl-ex].

[29] C. J. Bebek, C. N. Brown, M. Herzlinger, Stephen D. Holmes, C. A. Lichtenstein, F. M. Pipkin, S. Raither, and L. K. Sisterson, “Measurement of the pion form-factor up to $q ^ { 2 } = 4 – \mathrm { G e V } ^ { 2 } ,$ ” Phys. Rev. D 13, 25 (1976).

[30] M. Fujikawa et al. (Belle), “High-Statistics Study of the tau- —> pi- pi0 nu(tau) Decay,” Phys. Rev. D 78, 072006 (2008), arXiv:0805.3773 [hep-ex].

[31] S. Anderson et al. (CLEO), “Hadronic structure in the decay tau- —> pi- pi0 neutrino(tau),” Phys. Rev. D 61, 112002 (2000), arXiv:hep-ex/9910046.

[32] R. R. Akhmetshin et al. (CMD-2), “High-statistics mea-

surement of the pion form factor in the rho-meson energy range with the CMD-2 detector,” Phys. Lett. B 648, 28–38 (2007), arXiv:hep-ex/0610021.

[33] D. Bisello et al. (DM2), “The Pion Electromagnetic Formfactor in the Timelike Energy Range 1 $. 3 5 { \mathrm { - G e V } } \leq { \sqrt { s } } \leq 2 . 4 -$ GeV,” Phys. Lett. B 220, 321–327 (1989).

[34] Kamal K. Seth, S. Dobbs, Z. Metreveli, A. Tomaradze, T. Xiao, and G. Bonvicini, “Electromagnetic Structure of the Proton, Pion, and Kaon by High-Precision Form Factor Measurements at Large Timelike Momentum Transfers,” Phys. Rev. Lett. 110, 022002 (2013), arXiv:1210.1596 [hep-ex].

[35] D. Bollini, P. Giusti, T. Massam, L. Monari, F. Palmonari, G. Valenti, and A. Zichichi, “The Pion Electromagnetic Form-Factor in the Timelike Range 1.44-GeV\*\*2-9.0- GeV\*\*2,” Lett. Nuovo Cim. 14, 418 (1975).

[36] L. M. Barkov et al., “Electromagnetic Pion Form-Factor in the Timelike Region,” Nucl. Phys. B 256, 365–384 (1985).

[37] A.D. Bukin, I.B. Vasserman, I.A. Koop, L.M. Kurdadze, V.A. Sidorov, A.N. Skrinsky, G.M. Tumaikin, A.G. Khabakhpashev, A.G. Chilingarov, Yu.M. Shatunov, B.A. Schwartz, and S.I. Eidelman, “Pion form factor measurement by $e ^ { + } e ^ { - } \to \pi ^ { + } \pi ^ { - }$ in the energy range 2E from 0.78 up to 1.34 GeV,” Phys. Lett. B 73, 226–228 (1978).

[38] M. N. Achasov et al. (SND), “Measurement of the $e ^ { + } e ^ { - } \to$ π<sup>+</sup>π<sup>−</sup> process cross section with the SND detector at the VEPP-2000 collider in the energy region $0 . 5 2 5 < \sqrt { s } <$ 0.883 GeV,” JHEP 01, 113 (2021), arXiv:2004.00263 [hep-ex].

[39] F. Ambrosino et al. (KLOE), “Measurement of $\sigma ( e ^ { + } e ^ { - } $ π<sup>+</sup>π<sup>−</sup>) from threshold to 0.85 GeV<sup>2</sup> using Initial State Radiation with the KLOE detector,” Phys. Lett. B 700, 102– 110 (2011), arXiv:1006.5313 [hep-ex].

[40] M. Ablikim et al. (BESIII), “Measurement of the $e ^ { + } e ^ { - } \to$ $\pi ^ { + } \pi ^ { - }$ cross section between 600 and 900 MeV using initial state radiation,” Phys. Lett. B 753, 629–638 (2016), [Erratum: Phys.Lett.B 812, 135982 (2021)], arXiv:1507.08188 [hep-ex].

[41] F. Jegerlehner, “The Running fine structure constant alpha(E) via the Adler function,” Nucl. Phys. B Proc. Suppl. 181-182, 135–140 (2008), arXiv:0807.4206 [hep-ph].

[42] V. Cirigliano, G. Ecker, and H. Neufeld, “Radiative tau decay and the magnetic moment of the muon,” JHEP 08, 002 (2002), arXiv:hep-ph/0207310.

[43] L. Chang, I. C. Cloët, C. D. Roberts, S. M. Schmidt, and P. C. Tandy, “Pion electromagnetic form factor at spacelike momenta,” Phys. Rev. Lett. 111, 141802 (2013), arXiv:1307.0026 [nucl-th].

[44] Gilberto Colangelo, Martin Hoferichter, and Peter Stoffer, “Two-pion contribution to hadronic vacuum polarization,” JHEP 02, 006 (2019), arXiv:1810.00007 [hep-ph].

[45] Richard John Eden, Peter V. Landshof, David I. Olive, and John Charlton Polkinghorne, The analytic S-matrix (Cambridge Univ. Press, Cambridge, 1966).

[46] A.M. Badalyan, L.P. Kok, M.I. Polikarpov, and Yu.A. Simonov, “Resonances in coupled channels in nuclear and particle physics,” Physics Reports 82, 31–177 (1982).

[47] J. A. Oller, “Lectures on scattering theory in partial-wave amplitudes,” (2024), arXiv:2409.16790 [hep-ph].

[48] Irinel Caprini, Gilberto Colangelo, and Heinrich Leutwyler, “Mass and width of the lowest resonance in QCD,” Phys. Rev. Lett. 96, 132001 (2006), arXiv:hepph/0512364.

[49] S. Navas et al. (Particle Data Group), “Review of particle physics,” Phys. Rev. D 110, 030001 (2024).

[50] Xiang Gao, Nikhil Karthik, Swagato Mukherjee, Peter Petreczky, Sergey Syritsyn, and Yong Zhao, “Pion form factor and charge radius from lattice QCD at

the physical point,” Phys. Rev. D 104, 114515 (2021), arXiv:2102.06047 [hep-lat].

[51] J. Bijnens, G. Colangelo, and P. Talavera, “The Vector and scalar form-factors of the pion to two loops,” JHEP 05, 014 (1998), arXiv:hep-ph/9805389.

[52] B. Ananthanarayan, Irinel Caprini, and I. Sentitemsu Imsong, “Implications of the recent high statistics determination of the pion electromagnetic form factor in the timelike region,” Phys. Rev. D 83, 096002 (2011), arXiv:1102.3299 [hep-ph].

[53] Tran N. Truong, “Taylor’s series and dispersion relation analyses of the vector pion form-factor and their comparison with perturbative and nonperturbative calculations,” (1998), arXiv:hep-ph/9809476.

[54] Andrzej Czarnecki and William J. Marciano, “The Muon anomalous magnetic moment: A Harbinger for ’new physics’,” Phys. Rev. D 64, 013014 (2001), arXiv:hep-

ph/0102122.

[55] B. Ananthanarayan, Irinel Caprini, Diganta Das, and I. Sentitemsu Imsong, “Precise determination of the lowenergy hadronic contribution to the muon g − 2 from analyticity and unitarity: An improved analysis,” Phys. Rev. D 93, 116007 (2016), arXiv:1605.00202 [hep-ph].

[56] T. Aoyama et al., “The anomalous magnetic moment of the muon in the Standard Model,” Phys. Rept. 887, 1–166 (2020), arXiv:2006.04822 [hep-ph].

[57] Michel Davier, Andreas Hoecker, Bogdan Malaescu, and Zhiqing Zhang, “Reevaluation of the hadronic vacuum polarisation contributions to the Standard Model predictions of the muon g − 2 and $\alpha ( m _ { Z } ^ { 2 } )$ using newest hadronic cross-section data,” Eur. Phys. J. C 77, 827 (2017), arXiv:1706.09436 [hep-ph].

[58] Allan Pinkus, “Approximation theory of the MLP model in neural networks,” Acta Numerica 8, 143–195 (1999).