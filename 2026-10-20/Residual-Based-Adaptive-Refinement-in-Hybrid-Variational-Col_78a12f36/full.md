# Residual-Based Adaptive Refinement in Hybrid Variational-Collocation PINNs for Singular Elliptic PDEs with Mixed Dirichlet-Neumann Boundary Conditions

Somnath Giri<sup>a</sup>, Muslim Malik<sup>a,∗</sup> and Rishikesh Yadav<sup>a</sup>

<sup>a</sup>School ofMathematical & Statistical Sciences, Indian Institute ofTechnology Mandi, Mandi, 175005, Himachal Pradesh, India

A R T I C L E I N F O

Keywords:   
Hybrid variational-collocation PINNs   
Residual-based adaptive refinement   
Adaptive collocation   
Geometric singularities   
Mixed boundary conditions   
Petrov-Galerkin methods   
Singular elliptic PDEs

## A BS T R AC T

Solutions of second-order elliptic PDE problems on nonconvex or cracked domains with mixed Dirichlet-Neumann conditions typically lie outside $H ^ { 2 } ( \Omega )$ . Near re-entrant corners, crack tips and points where the boundary condition changes type, the gradient grows like $r ^ { \lambda - 1 }$ with $\lambda < 1$ . We study residual-based adaptive refinement in hybrid variational-collocation PINNs. A multiscale Fourier-feature network is trained on a composite loss. The loss combines a variational residual tested against Legendre bubble functions, a strong-form collocation term and penalties for the Dirichlet and Neumann data. The variational term needs only first derivatives of the network, whereas the collocation term requires second derivatives and provides a residual that can be evaluated at any point. After each training stage, candidate points selected by this residual are added to the collocation set. The method is tested on five two dimensional benchmarks, namely a Mode III crack, a cross-shaped domain with four re-entrant corners with unequal amplitude, a reaction-difusion problem with four localized peaks, a monotone semilinear problem with a boundary layer and the Motz problem. Compared with uniform point insertion under the same collocation and iteration budget, residual-based refinement reduced the relative $L ^ { 2 }$ error by up to a factor of 32, the relative $L ^ { \infty }$ error by up to a factor of 38 and the relative $H ^ { 1 }$ semi-norm error by up to a factor of 15.

## 1. Introduction

Many physical phenomena, including steady heat transfer, mass transport, electrostatic fields, linear elasticity and incompressible potential flows, are governed by second-order elliptic boundary value problems [1, 2]. In many applications, the primary quantities of interest are gradient, such as heat fluxes, electric fields and stress intensity factors. For such quantities, the accuracy of a numerical method in approximating ∇� is often as important as its accuracy in approximating � [3]. Mixed Dirichlet-Neumann boundary conditions frequently arise in elliptic problems, with the Neumann component presenting the greater dificulty. As it prescribes normal derivatives, its accurate<sup>.</sup> enforcement requires resolving gradients that may be singular near crack faces and re-entrant corners. The change of boundary condition type at a Dirichlet-Neumann junction introduces a singularity even in the absence of geometric singularities [4]. The classical Motz problem illustrates this efect on a straight boundary with smooth data [5].

Near a corner of interior angle �, Kondrat’ev-Grisvard theory [4, 6] provides the decomposition $u = u _ { \mathrm { r e g } } +$ $\begin{array} { r l } { \sum _ { j } c _ { j } \chi ( r ) r ^ { \lambda _ { j } } \phi _ { j } ( \theta ) } \end{array}$ , where $u _ { \mathrm { r e g } } \in H ^ { 2 } ( \Omega )$ and the singular exponents $\lambda _ { j }$ depend on both the interior angle and the two adjacent boundary conditions and equal $\frac { j \pi } { \omega }$ when both conditions are of the same type. With $\lambda = \lambda _ { 1 }$ denoting the leading singular exponent, $u \in H ^ { s } ( \Omega )$ for every $s < 1 + \lambda$ and $\left| \nabla u \right| \sim r ^ { \lambda - 1 }$ is unbounded whenever $\lambda < 1$ , as at re-entrant corners $( \omega > \pi )$ and at Dirichlet-Neumann junctions. This loss of regularity is intrinsic to the underlying problem and cannot be eliminated by the choice of approximation space. A uniform discretization fine enough to resolve the singularity over resolves the rest of the domain. Adaptive and $h p$ -adaptive finite element methods instead use a posteriori error estimators [7] to concentrate resolution near corners and crack tips. They recover optimal [8, 9] or even exponential [10, 11] convergence rates, at the cost of mesh generation and local refinement.

PINNs enforce the governing PDE by minimizing the strong-form residual together with boundary condition penalties, with derivatives computed by automatic diferentiation [12–14]. Their meshfree formulation enables the direct treatment of complex geometries [15, 16]. However, three factors limit the accuracy of PINNs for the class of problems considered here. First, the strong formulation requires the evaluation of second-order derivatives. Second, the optimization process exhibits a bias toward smooth, low-frequency solution components, leading to poor resolution of singularities and sharp layers [17–19]. Third, Neumann boundary conditions are typically enforced either through penalty terms, whose weights must be balanced against the residual term [20] or through specially designed trial spaces [21–23]. For corner singularities and mixed boundary conditions, singularity enriched PINNs [24] add the leading singular functions, with trainable coeficients to the network.

In variational PINNs (VPINNs) [25], the strong-form residual is replaced by a variational residual of Petrov-Galerkin type, with the neural network representing the trial solution and local polynomial basis functions defining the test space [26, 27]. Integration by parts transfers derivatives to the test functions, making only first-order derivatives of the network necessary. The mathematical theory of VPINNs discretizations is now well developed. It covers a priori energy norm estimates under inf-sup stability conditions, the efects of quadrature accuracy and test-space enrichment [28] and reliable and eficient a posteriori error estimation [29]. Further advances include basis independent residual minimization through the Riesz representation of the discrete dual norm [30] and its spectral realization in the Deep Fourier Residual framework [31, 32].

Adaptive strategies for VPINNs and related weak-form neural solvers typically act on the test space or on an underlying mesh. For example, the meshfree VPINN (MF-VPINN) generates additional test functions in regions identified by a posteriori error indicators [33], the robust VPINN (RVPINN) adaptively enriches the test space using information derived from the mismatch between the discrete and exact Riesz representatives [34] and finite elementinterpolated neural networks with adaptive refinement modify the underlying mesh used to represent the network approximation [35]. However, the variational residual does not provide a pointwise measure of error and therefore cannot be evaluated or ranked at arbitrary locations $\boldsymbol { x } \in \Omega$ . Consequently, pure VPINNs formulations admit no point level adaptivity and the residual-based sampling strategies developed for strong-form PINNs [36–40] cannot be applied directly.

A hybrid loss restores this capability. Specifically, augmenting the variational residual with the strong-form collocation term

$$
\mathcal { L } _ { f } = \frac { 1 } { N _ { f } } \sum _ { k = 1 } ^ { N _ { f } } \Bigl | R ( \hat { u } ) ( \pmb { x } _ { k } ) \Bigr | ^ { 2 } ,
$$

where $R ( \hat { u } )$ is the strong-form residual (2.12) of the network approximation ̂� and $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { N _ { f } }$ are the collocation points, provides a pointwise residual that can be evaluated throughout Ω. This residual may then be computed on a dense candidate pool and ranked to guide residual-based adaptive refinement (RAR). In this way, adaptivity acts directly on the collocation support, whereas variational adaptivity acts through the test space, the two mechanisms are complementary and address distinct aspects of the approximation. At the same time, the variational component depends only on first-order derivatives of ̂�, thereby avoiding the $H ^ { 2 } ( \Omega )$ regularity requirements that are violated near crack tips, re-entrant corners and other singular features. The proposed method is neither a pure weak-form method nor is it entirely free of second-order derivatives, as both the collocation term $ { \mathcal { L } } _ { f }$ and the RAR indicator require the evaluation of $\Delta \hat { u } .$ . Moreover, the Neumann boundary condition is not imposed naturally, the Legendre bubble functions that serve as test functions vanish on element boundaries, so the boundary functional drops out of the discrete variational residual. The prescribed flux is therefore enforced through an explicit penalty term, natural imposition would require a modified test space on elements adjacent to the Neumann boundary [41]. In addition, no adaptive ℎ- or �-refinement is performed, both the mesh partition and the local polynomial degree remain fixed throughout training. The underlying static discretization is that of ℎ�-VPINNs [26] and we call the resulting framework a hybrid variational-collocation PINNs (VC-PINNs), or VC-PINNs-RAR when it is combined with RAR. Collocation also enters collocation-based robust VPINNs (CRVPINNs) [42], where the robust variational loss is discretized by a point collocation scheme with a sparse Gram matrix. Here, by contrast, the collocation points carry a separate strong-form term beside a quadraturebased variational residual.

The contributions of this work are as follows. First, we observe that a strong-form collocation term kept beside the variational residual is what allows point insertion to act on training, its residual can be evaluated and ranked at any candidate point, whereas each entry of the variational residual belongs to a test function, not to a point. Second, we adapt RAR [36] to this hybrid loss for elliptic problems with mixed Dirichlet-Neumann conditions. Where needed, candidates within a guard radius of a singular point are excluded, as there the pointwise residual is not a reliable refinement indicator. Third, we compare every RAR run with a uniform insertion run on five two dimensional benchmarks. The first is a Mode III crack on a non-Lipschitz slit domain with traction-free crack faces, whose interior angle $\omega = 2 \pi$ gives the strongest singularity for a corner with the same type of condition on both edges. The next three are a cross-shaped domain with four re-entrant corners whose singularities difer in amplitude, an Allen-Cahn type reaction-difusion equation with steep localized peaks and a semilinear problem on the L-shaped domain with a boundary layer along the Dirichlet edges. The fifth is the Motz problem, which is singular since the boundary condition changes type, not due to the geometry. The two runs of each comparison, the uniform arm and the RAR arm, share the architecture, quadrature rule, collocation schedule and optimizer schedule, point placement is the only diference. The relative $L ^ { 2 } , L ^ { \infty }$ and $H ^ { 1 }$ semi-norm errors are reported problem by problem together with the training budgets.

The paper is organized as follows. Section 2 states the model problem, its weak formulation and the regularity results used later and it separates the element indexed variational residual from the pointwise strong-form residual, a distinction on which the rest of the paper builds. Section 3 develops the hybrid discretization, the composite loss, the RAR procedure and the training algorithm, together with the budget under which the two arms are compared. Section 4 assembles these into the architecture of the framework. Five numerical examples are presented in section 5, followed by concluding remarks in section 6.

## 2. Preliminaries

This section fixes the model problem and its weak form, records the regularity imposed by the geometry and the boundary conditions and distinguishes the two residuals on which the method of section 3 is built.

## 2.1. Model problem and weak formulation

Let $\Omega \subset \mathbb { R } ^ { 2 }$ be a bounded, connected open set whose boundary is a finite union of closed straight segments, with vertex set  and interior angles $\omega _ { v } \in ( 0 , 2 \pi ]$ . The value $\omega _ { v } = 2 \pi$ occurs only at the tip of a slit, the one case in which Ω is not Lipschitz, in the terminology of [4], Ω is a polygon with cracks. The two faces of a slit are treated as distinct boundary portions with opposite outward normals, so that a function in $H ^ { 1 } ( \Omega )$ has two independent traces there.

The boundary is split into disjoint, relatively open sets $\Gamma _ { D }$ and $\Gamma _ { N }$ with $\partial \Omega = \overline { { \Gamma _ { D } \cup \Gamma _ { N } } }$ and either $\mathrm { ( D ) } \left| \Gamma _ { D } \right| > 0$ or $( \mathrm { N } ) \Gamma _ { D } = \emptyset$ , where $| \cdot |$ denotes one-dimensional measure. Case (N) occurs only in section 5.3. With $( \cdot , \cdot ) _ { \Omega }$ the $L ^ { 2 } ( \Omega )$ inner product, let

$$
\boldsymbol { V } = \{ \boldsymbol { v } \in \boldsymbol { H } ^ { 1 } ( \Omega ) : \boldsymbol { v } | _ { \Gamma _ { D } } = 0 \} , \qquad \boldsymbol { H } _ { g } ^ { 1 } = \{ \boldsymbol { w } \in \boldsymbol { H } ^ { 1 } ( \Omega ) : \boldsymbol { w } | _ { \Gamma _ { D } } = g \} ,\tag{2.1}
$$

so that $V = H _ { \sigma } ^ { 1 } = H ^ { 1 } ( \Omega )$ in case (N) and $H _ { g } ^ { 1 } = u _ { g } + V$ for any $u _ { g } \in H ^ { 1 } ( \Omega )$ with trace � on $\Gamma _ { D }$ . The dual space of � is denoted by $\Breve { V ^ { \prime } }$ and $\langle \cdot , \cdot \rangle$ denotes the duality pairing between $V ^ { \prime }$ and $V .$

Assumption 2.1 (Data). $A = \kappa I$ with constant $\kappa > 0 , \kappa = 1$ for the crack, cross and Motz problems and $\kappa = \varepsilon ^ { 2 }$ for the reaction-difusion and semilinear problems, with $\varepsilon = 0 . 0 1$ and $\varepsilon = 0 . 0 5$ , respectively. The data satisfy $f \in L ^ { 2 } ( \Omega )$ and $h \in L ^ { 2 } ( \Gamma _ { N } )$ and $g$ is the trace on $\Gamma _ { D }$ of a function in $H ^ { 1 } ( \Omega )$ . The reaction term $\mathcal { N } : \mathbb { R } $ ℝ is continuous with polynomial growth, so $\mathcal { N } ( w ) \in L ^ { 2 } ( \Omega )$ for every $w \in H ^ { 1 } ( \Omega )$ by the embedding $H ^ { 1 } ( \Omega ) \hookrightarrow L ^ { q } ( \Omega ) , q < \infty$ , which holds since $\Omega$ coincides, up to a null set, with a finite union of Lipschitz rectangles. Three cases occur: ${ \mathcal { N } } \equiv 0$ for the crack, cross and Motz problems, $\mathcal { N } ( t ) = t ^ { 3 } - t$ for the reaction-difusion problem and $\mathcal { N } ( t ) = t ^ { 3 }$ for the semilinear problem.

Square integrability of $h ,$ rather than boundedness, is required as, in the cross problem, the prescribed flux behaves like $r ^ { - 1 / 3 }$ at the re-entrant corners. With � the outward unit normal, the model problem is

$$
\begin{array} { r } { - \nabla \cdot ( A \nabla u ) + \mathcal { N } ( u ) = f } \\ { u = g } \\ { ( A \nabla u ) \cdot n = h } \end{array}
$$

$$
{ \mathrm { i n ~ } } \Omega ,\tag{2.2}
$$

$$
{ \mathrm { o n } } \Gamma _ { D } ,\tag{2.3}
$$

$$
\quad \mathrm { o n } \Gamma _ { N } .\tag{2.4}
$$

Equation (2.2) holds in the sense of distributions and pointwise only where $u \in H _ { \mathrm { l o c } } ^ { 2 }$ , this fails at the singular points of section 2.2. Testing (2.2) with $v \in V$ and applying Green’s formula together with (2.4) gives the weak problem: find $u \in H _ { g } ^ { 1 }$ such that

$$
a ( u , v ) + \left( { \mathcal { N } } ( u ) , v \right) _ { \Omega } = \ell ( v ) \qquad \forall v \in V ,\tag{2.5}
$$

where

$$
a ( u , v ) = \int _ { \Omega } A \nabla u \cdot \nabla v d \mathbf { x } , \qquad \ell ( v ) = \int _ { \Omega } f v d \mathbf { x } + \int _ { \Gamma _ { N } } h v d S .\tag{2.6}
$$

Problem (2.5) involves only first derivatives of �. The Dirichlet condition is essential, the Neumann condition is natura and enters through $\ell .$ . Section 3.2 shows that the discrete test space annihilates this term and section 3.3 restores it by a penalty.

## Proposition 2.2 (Well-posedness). Let Assumption 2.1 hold.

(i) $I f \mathcal { N } \equiv 0$ and case (D) holds, � is bounded and coercive on � and (2.5) has a unique solution by the Lax-Milgram theorem [2].

(ii) If <sup></sup> is nondecreasing and case (D) holds, the operator $\mathcal { G } : V  V ^ { \prime } , \langle \mathcal { G } ( z ) , v \rangle = a ( u _ { g } + z , v ) + ( \mathcal { N } ( u _ { g } +$ $z ) , v ) _ { \Omega } - \ell ( v )$ , is bounded, hemicontinuous, coercive and strictly monotone, with

$$
\langle \mathcal { G } ( z _ { 1 } ) - \mathcal { G } ( z _ { 2 } ) , z _ { 1 } - z _ { 2 } \rangle \geq \kappa \left\| \nabla ( z _ { 1 } - z _ { 2 } ) \right\| _ { L ^ { 2 } ( \Omega ) } ^ { 2 } .\tag{2.7}
$$

By the Browder-Minty theorem [43], (2.5) has a unique solution, independent of the lift $u _ { g } .$

(iii) $I f \mathcal { N } ( t ) = t ^ { 3 } - t$ and case (N) holds, <sup></sup> is decreasing on $( - \frac { 1 } { \sqrt { 3 } } , \frac { 1 } { \sqrt { 3 } } )$ and for $f \equiv h \equiv 0$ the constants $u \equiv 0$ and $u \equiv \pm 1$ all solve (2.5). Uniqueness fails and the boundary data do not select a branch.

Coercivity in (i) and (ii) rests on the Poincaré-Friedrichs inequality on �. On the slit domain it follows from $| \Gamma _ { D } | > 0$ , the connectedness of Ω and the compactness of $H ^ { 1 } ( \Omega ) \stackrel { - } { \hookrightarrow } L ^ { \hat { 2 } } ( \Omega )$ . The compactness holds as Ω coincides, up to a null set, with a finite union of Lipschitz rectangles.

Remark 2.3 (Branch selection). In view of proposition 2.2(iii), the reaction-difusion problem is supplemented by twenty interior values of the exact solution, placed at the four peak centers and at four neighbors of each. They enter the loss (3.21) as an additional quadratic term. They select the branch and also supply the locations of the features that the adaptive scheme resolve, hence no conclusion about feature discovery is drawn from that example. The semilinear problem has a unique solution by proposition 2.2(ii) and nothing beyond �, �, ℎ and Ω is supplied to the solver.

## 2.2. Regularity

Call $p \in \partial \Omega \setminus \mathcal { V }$ a Dirichlet-Neumann junction if $p \in \overline { { \Gamma _ { D } } } \cap \overline { { \Gamma _ { N } } }$ , let  be the finite set of such points and define the singular set $S = \mathcal { V } \cup \mathcal { I }$ . Its subset $S ^ { * } = \{ p \in S : \lambda _ { 1 } ( p ) < 1 \}$ , with $\lambda _ { 1 } ( p )$ the leading exponent at � given by (2.9), collects the points at which the expansion (2.8) below can contain singular terms. On every compact subset of $\Omega \setminus S ,$ $u \in H ^ { 2 } ( \Omega )$ and � is $C ^ { \infty }$ there for smooth data [2, 4]. Near $p \in S$ with interior angle � and local polar coordinates $( r , \theta )$

$$
u = u _ { \mathrm { r e g } } + \sum _ { 0 < \lambda _ { j } < 1 } c _ { j } \chi ( r ) r ^ { \lambda _ { j } } \phi _ { j } ( \theta ) , \qquad u _ { \mathrm { r e g } } \in H ^ { 2 } ( \Omega ) ,\tag{2.8}
$$

where $\chi$ is a smooth cut of and the coeficients � are not known a priori. The pairs $c _ { j }$ $( \lambda _ { j } , \phi _ { j } )$ solve $\phi ^ { \prime \prime } + \lambda ^ { 2 } \phi = 0$ on $( 0 , \omega )$ under the boundary conditions of the two rays meeting at � [4],

$$
\lambda _ { j } = { \frac { j \pi } { \omega } } \mathrm { ( D , D ) , ( N , N ) } , \qquad \lambda _ { j } = { \frac { ( 2 j - 1 ) \pi } { 2 \omega } } \mathrm { ( D , N ) , ( N , D ) } , \qquad j \geq 1 ,\tag{2.9}
$$

with $\phi _ { j } = \sin ( \lambda _ { j } \theta )$ if the ray $\theta = 0$ is Dirichlet and $\phi _ { j } = \cos ( \lambda _ { j } \theta )$ if it is Neumann. A change of boundary condition type shifts the spectrum by half an index, so $\begin{array} { r } { \lambda _ { 1 } = \frac { 1 } { 2 } } \end{array}$ already at $\omega = \pi ;$ this is the mechanism of the Motz problem [5]. Terms with $\lambda _ { j } \in \mathbb { N }$ whose angular part makes $r ^ { \lambda _ { j } } \phi _ { j }$ a harmonic polynomial are smooth and belong to $u _ { \mathrm { r e g } }$ . For the two nonlinear problems, $- \kappa \Delta u = f - \mathcal { N } ( u ) \in L ^ { 2 } ( \Omega )$ , so (2.8) holds with the same $\lambda _ { j }$ and diferent $c _ { j }$

Let $\lambda = \operatorname* { m i n } _ { p \in S } \lambda _ { 1 } ( p )$ and suppose $\lambda < 1$ . Since, for any $\begin{array} { r } { r _ { 0 } > 0 , \int _ { 0 } ^ { r _ { 0 } } r ^ { 2 ( \lambda - s ) } r d r < } \end{array}$ ∞ exactly when $s < 1 + \lambda$

$$
u \in H ^ { s } ( \Omega ) \ \forall s < 1 + \lambda , \qquad \mathrm { i f } \ c _ { 1 } \neq 0 \ \mathrm { a n d } \ u \notin H ^ { 1 + \lambda } ( \Omega ) , \ | \nabla u | \sim r ^ { \lambda - 1 } \ \mathrm { a s } \ r \to 0 .\tag{2.10}
$$

Leading singular exponents from (2.9). In the cross problem the corner singularity enters through the flux data $h ,$ the exponent is the same for (D,D) and (N,N).
<table><tr><td>Problem</td><td>Points of S that govern λ</td><td>Rays</td><td>ω</td><td> $\lambda _ { 1 }$ </td></tr><tr><td>Crack (§5.1)</td><td>crack tip</td><td>(N,N)</td><td> $2 \pi$ </td><td> $1 / 2$ </td></tr><tr><td>Cross  $( \ S 5 . 2 )$ </td><td>four re-entrant corners</td><td>(N,N)</td><td> $3 \pi / 2$ </td><td> $2 / 3$ </td></tr><tr><td>Reaction-diffusion (§5.3)</td><td>vertices</td><td>(N,N)</td><td> $\pi / 2$ </td><td>2</td></tr><tr><td>Semilinear (§5.4)</td><td>re-entrant corner</td><td>(D,D)</td><td> $3 \pi / 2$ </td><td> $2 / 3$ </td></tr><tr><td>Motz (§5.5)</td><td>junction at (0, 0)</td><td>(N,D)</td><td>π</td><td> $1 / 2$ </td></tr></table>

In that case $\nabla u \in L ^ { 2 } ( \Omega )$ is unbounded and � $\notin \ : H ^ { 2 } ( \Omega )$ . Table 1 lists the leading exponents of the five benchmark problems. For the four benchmarks with $\lambda < 1$ , the second derivatives required by (2.12) are not in $L ^ { 2 } ( \Omega )$ . In the reaction-difusion problem every vertex is a right angle with Neumann data on both rays, so $\lambda _ { j } = 2 j$ , no singular term arises and $u \in H ^ { 2 } ( \Omega )$ . Its dificulty lies in resolving steep peaks, not in regularity. The semilinear problem also carries a boundary layer of width $\frac { \varepsilon } { \sqrt { 3 } }$ along $\Gamma _ { D } .$ , analyzed in section 5.4.

## 2.3. Two notions of residual

The variational residual of $w \in H ^ { 1 } ( \Omega )$ is

$$
\mathcal { R } ( w ; v ) = a ( w , v ) + \left( \mathcal { N } ( w ) , v \right) _ { \Omega } - \ell ( v ) , \qquad v \in V ,\tag{2.11}
$$

so $\mathcal { R } ( w ; \cdot ) \in V ^ { \prime }$ . It involves only first derivatives and is therefore well defined under (2.10). A VPINNs [25, 26] is a Petrov-Galerkin method that takes its trial functions from a class of neural networks and its test functions from a finite-dimensional space $V _ { K } \subset V$ . It drives  to zero on a basis of $V _ { K }$ . The space $V _ { K }$ used here is constructed in section 3.2.

The strong-form residual is

$$
R ( w ) ( \pmb { x } ) = - \nabla \cdot \big ( \pmb { A } \nabla w ( \pmb { x } ) \big ) + \mathcal { N } \big ( w ( \pmb { x } ) \big ) - f ( \pmb { x } ) ,\tag{2.12}
$$

and it requires second derivatives. As observed in [25, 26], the choice $v = \delta _ { x _ { 0 } }$ , the Dirac measure at a point $\boldsymbol { x } _ { 0 } \in \Omega$ formally turns the variational residual into the strong-form residual at $\mathbf { \boldsymbol { x } } _ { 0 } .$ . For $w \in H ^ { 1 } ( \Omega )$ this identification is not available. In two dimensions $H ^ { 1 } ( \Omega ) \not \hookrightarrow C ( \overline { { \Omega } } )$ , with log | log �| as the standard witness, so point evaluation is unbounded on $H ^ { 1 } ( \Omega ) , \delta _ { x _ { 0 } } \notin V ^ { \prime }$ and $\mathcal { R } ( w ; \delta _ { x _ { 0 } } )$ is undefined. Consequently,  can be tested yet cannot be evaluated, ranked or compared at a point. Conversely, $R ( u )$ is not defined near any $p \in S ^ { * }$ , as � $\notin H ^ { 2 } ( \Omega )$ there. In section 3.3, � is applied to the network trial function $\hat { u } ( \cdot ; \theta )$ , which is twice diferentiable away from $S ,$ therefore $R ( \hat { u } )$ is a function of position that can be evaluated at any candidate point and it is the only loss component with this property.

Remark 2.4 (The trial function is not globally smooth). The input enrichment of section 3.1 contains features $r ^ { \lambda _ { 1 } ( p ) }$ centered at points $p \in S ^ { * }$ . Away from these points ̂� is $C ^ { \infty }$ . At them, $| D ^ { 2 } r ^ { \lambda _ { 1 } } | \sim r ^ { \lambda _ { 1 } - 2 }$ , where $D ^ { 2 }$ denotes the Hessian, so ̂� inherits the behavior (2.10) it is built to represent. This has two consequences for (2.12) evaluated at ̂�. First, the individual second derivatives grow like $r ^ { \lambda _ { 1 } - 2 }$ while their combination is the small quantity being measured, so near � the computed value is eventually dominated by floating point cancellation. Second, the $\bar { r ^ { \lambda _ { 1 } - 2 } }$ envelope is a genuine property of the solution, so a ranking by $| R ( { \hat { u } } ) |$ favors proximity to � independently of the local error. Candidates within a guard radius of $S ^ { * }$ are therefore excluded from refinement (section 3.4).

## 3. Hybrid variational-collocation discretization with residual-based adaptive refinement

This section discretizes the residual formulations of section 2.3 and introduces the adaptive procedure built on them. The trial function (section 3.1) is a residual tanh network on a multiscale Fourier-feature embedding. For the singular benchmarks it is preceded by an enrichment that encodes the known behavior at the singular points in $S ^ { * }$ The variational residual (section 3.2) is tested against Legendre bubble functions on a fixed rectangular partition and evaluated by Gauss-Legendre quadrature. As these functions vanish on every element boundary, including �Ω, the Neumann data are imposed by a penalty in the composite loss (section 3.3). The loss also contains a collocation term built on the strong-form residual. This term makes the method hybrid. Unlike the variational residual, the strong-form residual can be evaluated at arbitrary points, so it drives the insertion of collocation points by RAR (section 3.4). Section 3.5 states the complete algorithm, in which only the collocation set is adapted, together with the optimization settings and the training budgets.

The enrichment map Ψ of (3.1). Here $r _ { i }$ is the distance to the �-th re-entrant corner. For the crack, $( r , \theta )$ are polar coordinates centered at the tip, with $\theta \in ( 0 , 2 \pi )$ and the branch cut on the slit. For the Motz problem, $( r , \theta )$ are centered at the Dirichlet-Neumann junction, with $\theta \in [ 0 , \pi ]$ . In the semilinear row, � is the distance to the re-entrant corner at the origin. Every radius is evaluated through (3.2).
<table><tr><td>Problem</td><td> $\Psi ( { \pmb x } )$   $n _ { \mathrm { i n } }$ </td></tr><tr><td>Crack (§5.1)</td><td> $\begin{array} { r l } { \small } & { { } ( x , y , r ^ { 1 / 2 } \cos \frac { \theta } { ) } , r ^ { 1 / 2 } \sin \frac { \theta } { ) } \big ) } \end{array}$  4</td></tr><tr><td>Cross (§5.2)</td><td> $\left( x , y , r _ { 1 } ^ { 2 / 3 } , r _ { 2 } ^ { 2 / 5 } , r _ { 3 } ^ { 2 / 3 } , r _ { 4 } ^ { 2 / 3 } \right)$  6</td></tr><tr><td>Reaction-diffusion (§5.3) (x, y)</td><td>2</td></tr><tr><td>Semilinear (§5.4)</td><td> $\left( x , y , r ^ { 2 / 3 } \right)$  3</td></tr><tr><td>Motz (§5.5)</td><td> $\scriptstyle \left( x , y , r ^ { 1 / 2 } \cos { \frac { \theta } { 2 } } \right)$  3</td></tr></table>

## 3.1. Trial function and input enrichment

As in PINNs and VPINNs [12, 26], the trial functions are neural networks. Here the trial function is the composition

$$
\begin{array} { r } { \hat { u } ( \cdot ; \theta ) = c _ { u } \mathcal { F } _ { \theta } \circ \gamma \circ \Psi , \qquad \Psi : \overline { { \Omega } }  \mathbb { R } ^ { n _ { \mathrm { i n } } } , \quad \gamma : \mathbb { R } ^ { n _ { \mathrm { i n } } }  \mathbb { R } ^ { 2 S n _ { F } } , \quad \mathcal { F } _ { \theta } : \mathbb { R } ^ { 2 S n _ { F } }  \mathbb { R } , } \end{array}\tag{3.1}
$$

of an enrichment map Ψ, a multiscale Fourier-feature embedding � [18, 44] and a residual tanh network ${ \mathcal { F } } _ { \theta }$ . The constant $c _ { u }$ fixes the output scale. It equals 500 for the Motz problem, where the network represents $u / 5 0 0$ and 1 otherwise.

Enrichment: The maps Ψ are listed in table 2. Without enrichment, Ψ = id and $n _ { \mathrm { i n } } = 2$ . Every fractional power is taken of a floored radius,

$$
r _ { \tau } = \big ( \mathrm { m a x } \{ r ^ { 2 } , \tau \} \big ) ^ { 1 / 2 } , \qquad \tau \leq 1 0 ^ { - 2 8 } ,\tag{3.2}
$$

which keeps Ψ finite and diferentiable at the singular points themselves. This floor only safeguards the evaluation of Ψ. It is unrelated to and many orders of magnitude smaller than, the guard radius $r _ { \mathrm { g } }$ of (3.24).

Embedding and network: Fix scales $\sigma _ { 1 } , \dots , \sigma _ { S } > 0$ and matrices $\pmb { B } ^ { ( s ) } \in \mathbb { R } ^ { n _ { \mathrm { i n } } \times n _ { F } }$ whose entries are independent draws from the normal distribution with mean 0 and variance $\sigma _ { s } ^ { 2 }$ . They are sampled once and then held fixed, they are not trained. With sin and cos acting componentwise,

$$
\begin{array} { r } { \gamma ( z ) = \left( \sin \bigl ( B ^ { ( 1 ) \top } z \bigr ) ^ { \top } , \cos \bigl ( B ^ { ( 1 ) \top } z \bigr ) ^ { \top } , \ldots , \sin \bigl ( B ^ { ( S ) \top } z \bigr ) ^ { \top } , \cos \bigl ( B ^ { ( S ) \top } z \bigr ) ^ { \top } \right) ^ { \top } . } \end{array}\tag{3.3}
$$

In every experiment $n _ { F } = 6 4$ and $S = 3$ , giving $\gamma ( z ) \in \mathbb { R } ^ { 3 8 4 }$ . The network has � hidden layers of uniform width $n _ { H }$ With $\pmb { h } ^ { ( 0 ) } = \gamma ( \Psi ( \pmb { x } ) )$ ,

$$
\begin{array} { r } { \boldsymbol { h } ^ { ( 1 ) } = \operatorname { t a n h } \bigl ( \boldsymbol { W } ^ { ( 1 ) } \boldsymbol { h } ^ { ( 0 ) } + \boldsymbol { b } ^ { ( 1 ) } \bigr ) , \qquad \boldsymbol { h } ^ { ( i ) } = \boldsymbol { h } ^ { ( i - 1 ) } + \operatorname { t a n h } \bigl ( \boldsymbol { W } ^ { ( i ) } \boldsymbol { h } ^ { ( i - 1 ) } + \boldsymbol { b } ^ { ( i ) } \bigr ) , \quad i = 2 , \dots , L , } \end{array}\tag{3.4}
$$

$$
\mathcal { F } _ { \theta } ( { h } ^ { ( 0 ) } ) = W ^ { ( L + 1 ) } { h } ^ { ( L ) } + b ^ { ( L + 1 ) } ,\tag{3.5}
$$

where ${ \pmb W } ^ { ( 1 ) } \in \mathbb { R } ^ { n _ { H } \times 2 S n _ { F } }$ $W ^ { ( i ) } \in \mathbb { R } ^ { n _ { H } \times n _ { H } }$ for $2 \leq i \leq L , \pmb { b } ^ { ( i ) } \in \mathbb { R } ^ { n _ { H } }$ for $1 \le i \le L , { \cal W } ^ { ( L + 1 ) } \in \mathbb { R } ^ { 1 \times n _ { H } }$ and $b ^ { ( L + 1 ) } \in \mathbb { R }$ The trainable parameters are $\pmb { \theta } = \{ \pmb { W } ^ { ( i ) } , \pmb { b } ^ { ( i ) } \} _ { i = 1 } ^ { L } \cup \{ \pmb { W } ^ { ( L + 1 ) } , \pmb { b } ^ { ( L + 1 ) } \}$ . The skip connection acts only between hidden layers of equal width. Weights are initialized by the Xavier normal rule [44] and biases by zero. All computations are in double precision. A single fixed seed governs $\{ B ^ { ( s ) } \}$ , the initialization of � and every random point set of sections 3.3 and 3.4.

Since � and ${ \mathcal { F } } _ { \theta }$ are $C ^ { \infty }$ , the regularity of ̂� is that of Ψ. Hence $\hat { u } \in C ^ { \infty } ( \overline { { \Omega } } \backslash S ^ { * } )$ and near a singular point $p \in S ^ { * }$ the trial function carries the algebraic behavior $r ^ { \lambda _ { 1 } ( p ) }$ of (2.10). The consequences for the strong-form residual are recorded in remark 2.4.

Remark 3.1 (Status of the enrichment). The crack pair $( s _ { 1 } , s _ { 2 } ) = ( r ^ { 1 / 2 } \cos { \frac { \theta } { 2 } } , r ^ { 1 / 2 } \sin { \frac { \theta } { 2 } } )$ serves two purposes, which should be kept apart. First, $s _ { 2 }$ removes a topological obstruction. The exact solution takes diferent values on the two faces of the slit, whereas any function continuous in $( x , y )$ takes arbitrarily close values at points on opposite faces. As $\theta \to 0 ^ { + }$ and $\theta  2 \pi ^ { - } , s _ { 2 }  0 ^ { + }$ and $s _ { 2 }  0 ^ { - }$ , so the map $\pmb { x } \mapsto ( s _ { 1 } , s _ { 2 } )$ is continuous on $\Omega$ and separates the faces. This is the discrete counterpart of the two independent traces of section 2.1. The function $s _ { 2 }$ is not a mode of the (N,N) wedge at $\omega = 2 \pi$ , whose angular functions are cosines by (2.9), it is supplied for the reason just given. Second, $s _ { 1 }$ is the leading singular mode at the tip and for the exact solution of section 5.1 it coincides with �. The enrichment is therefore not a neutral change of coordinates rather, it is a singular function enrichment of the trial space, as in the singular function method [45], extended finite element methods [46] and enriched neural solvers for crack problems [47, 48]. It difers from those constructions, including the singularity enriched PINNs of [24], in one respect: the mode is not added as a separate term with its own coeficient instead, it is passed through (3.3) and (3.4), so ̂� depends on it nonlinearly.

The corner features $r ^ { 2 / 3 }$ have the same status. The Motz feature supplies only the leading mode of (2.8), the higher modes must be learned. Section 5 measures the efect of adaptive point placement given the enrichment, which is identical in the two compared arms.

## 3.2. Test space, variational residual and quadrature

Mesh and reference element: Let $\mathcal { E } = \left\{ \Omega _ { e } \right\} _ { e = 1 } ^ { N _ { e } }$ be a partition of Ω into $N _ { e }$ non-overlapping, axis-aligned open rectangles $\Omega _ { e } = ( x _ { a } ^ { e } , x _ { b } ^ { e } ) \times ( y _ { a } ^ { e } , y _ { b } ^ { e } )$ with $\begin{array} { r } { \overline { { \Omega } } = \bigcup _ { e = 1 } ^ { N _ { e } } \overline { { \Omega } } _ { e } . \operatorname { L e t } \hat { \Omega } = [ - 1 , 1 ] ^ { 2 } } \end{array}$ denote the reference element with coordinates $( \xi , \eta )$ . Each element is the image of $\hat { \Omega }$ under the afine map

$$
F _ { e } ( \xi , \eta ) = \big ( x _ { a } ^ { e } + J _ { x } ^ { e } ( 1 + \xi ) , ~ y _ { a } ^ { e } + J _ { y } ^ { e } ( 1 + \eta ) \big ) , \qquad J _ { x } ^ { e } = \textstyle { \frac { 1 } { 2 } } ( x _ { b } ^ { e } - x _ { a } ^ { e } ) , \quad J _ { y } ^ { e } = \textstyle { \frac { 1 } { 2 } } ( y _ { b } ^ { e } - y _ { a } ^ { e } ) ,\tag{3.6}
$$

so that $F _ { e } ( \hat { \Omega } ) = \overline { { \Omega } } _ { e }$ . The Jacobian matrix of $F _ { e }$ is $\mathrm { d i a g } ( J _ { x } ^ { e } , J _ { y } ^ { e } )$ where $J _ { x } ^ { e }$ and $J _ { y } ^ { e }$ are half the side lengths of $\Omega _ { e }$ and $d { \pmb x } = J _ { \ b x } ^ { e } J _ { \ b v } ^ { e } d \xi d \eta$

The partition is chosen so that no point of $S ^ { * }$ lies in an element interior. The re-entrant corner of the semilinear problem and the Motz junction are element vertices, the four re-entrant corners of the cross are vertices of the two side elements and lie inside the vertical edges of the central bar. For the crack, the slit $\Sigma = \{ ( x , 0 ) : 0 \leq x \leq 1 \}$ of section 5.1 lies on an element interface. Hence no element straddles the slit and the polar angle � in the exact solution is smooth on every element. We use $N _ { e } = 3$ for the cross and semilinear problems and $N _ { e } = 4$ for the crack, reaction-difusion and Motz problems.

Test functions: On [−1, 1] we use the Legendre bubble functions of [26],

$$
\varphi _ { i } ( \xi ) = P _ { i + 1 } ( \xi ) - P _ { i - 1 } ( \xi ) , \qquad i = 1 , \dots , K ,\tag{3.7}
$$

where $P _ { j }$ is the Legendre polynomial of degree �, thus $\varphi _ { i }$ is a polynomial of degree $i + 1$ . Since $P _ { j } ( 1 ) \ = \ 1$ $P _ { j } ( - 1 ) { \stackrel { \_ } { = } } ( - 1 ) ^ { j } { \mathrm { ~ a n d ~ } } P _ { i + 1 } ^ { \prime } - P _ { i - 1 } ^ { \prime } = ( 2 i + 1 ) P _ { i }$

$$
\varphi _ { i } ( \pm 1 ) = 0 , \qquad \varphi _ { i } ^ { \prime } = ( 2 i + 1 ) P _ { i } .\tag{3.8}
$$

The derivatives of the test functions are therefore available in closed form and the basis is never diferentiated numerically. On the reference element we define the tensor-product functions

$$
\hat { v } _ { i j } ( \xi , \eta ) = \varphi _ { i } ( \xi ) \varphi _ { j } ( \eta ) , \qquad 1 \leq i , j \leq K ,\tag{3.9}
$$

and transfer them to $\Omega _ { e } .$ , extended by zero outside the element,

$$
\begin{array} { r } { v _ { i j } ^ { e } ( { \pmb x } ) = \left\{ \begin{array} { l l } { \hat { v } _ { i j } \big ( F _ { e } ^ { - 1 } ( { \pmb x } ) \big ) , } & { { \pmb x } \in \Omega _ { e } , } \\ { 0 , } & { { \pmb x } \in \Omega \setminus \Omega _ { e } . } \end{array} \right. } \end{array}\tag{3.10}
$$

The test space is spanned by all such functions,

$$
V _ { K } = \mathrm { s p a n } \big \{ v _ { i j } ^ { e } ~ : ~ 1 \leq e \leq N _ { e } , 1 \leq i , j \leq K \big \} .\tag{3.11}
$$

Since distinct elements have disjoint interiors, the functions $v _ { i j } ^ { e }$ are linearly independent and dim $V _ { K } ~ = ~ N _ { e } K ^ { 2 }$ Equivalently, $V _ { K }$ consists of the functions whose restriction to each $\Omega _ { e } ,$ , pulled back by $F _ { e }$ , is a polynomial of degree at most $K + 1$ in each variable vanishing on $\partial \hat { \Omega } .$ In every experiment $K = 8$ , which gives $K ^ { 2 } = 6 4$ test functions per element and $N _ { e } K ^ { 2 } \in \{ 1 9 2 , 2 5 6 \}$ in total.

Lemma 3.2 (Admissibility). $V _ { K } \subset H _ { 0 } ^ { 1 } ( \Omega ) \subset V$

Proof. By (3.8), $\hat { v } _ { i j }$ vanishes on $\partial \hat { \Omega } .$ , so $v _ { i j } ^ { e }$ vanishes on $\partial \Omega _ { e }$ . Its zero extension is therefore continuous and piecewise polynomial, hence lies in $H ^ { 1 } ( \Omega )$ with zero trace on �Ω. The inclusion $H _ { 0 } ^ { 1 } ( \Omega ) \subset V$ holds by (2.1) in both case (D) and case (N). □

Variational residual: By Lemma 3.2, the variational residual (2.11) can be tested with $V _ { K }$ directly, without interface or boundary terms and requires only $\hat { u } \in H ^ { 1 } ( \Omega )$ . For $1 \leq e \leq N _ { e }$ and $1 \leq i , j \leq K$

$$
\mathcal { R } _ { i j } ^ { e } ( \theta ) = \mathcal { R } \big ( \hat { u } ; v _ { i j } ^ { e } \big ) = \int _ { \Omega _ { e } } \Big ( A \nabla \hat { u } \cdot \nabla v _ { i j } ^ { e } + \mathcal { N } ( \hat { u } ) v _ { i j } ^ { e } - f v _ { i j } ^ { e } \Big ) d x .\tag{3.12}
$$

The integral is over $\Omega _ { e }$ alone, since $v _ { i j } ^ { e }$ vanishes outside it. The nonlinear term is present only in the reaction-difusion and semilinear problems.

Remark 3.3 (Loss of the natural Neumann term). Every $v \in V _ { K }$ vanishes on $\Gamma _ { N }$ . The boundary functional of (2.6),

$$
B _ { i j } ^ { e } = \int _ { \Gamma _ { N } } h v _ { i j } ^ { e } d S ,\tag{3.13}
$$

is therefore identically zero and the Neumann data do not enter the discrete variational residual. This is a property of the bubble basis (3.7), not of the weak form (2.5). A basis that is active on $\Gamma _ { N }$ , such as the odd sine and cosine modes of [41], retains the term. We keep the bubble basis and impose ℎ through the penalty (3.19), which is then the only route by which ℎ enters the loss.

Quadrature: Let $\{ ( \xi _ { p } , w _ { p } ) \} _ { p = 1 } ^ { Q }$ be the nodes and weights of the �-point Gauss-Legendre rule on [−1, 1]. Set $w _ { p q } = w _ { p } w _ { q }$ and $\pmb { x } _ { p q } ^ { e } = F _ { e } ( \xi _ { p } , \xi _ { q } ^ { \mathrm { ~ } } )$ and define the set of quadrature points

$$
\mathcal { T } _ { Q } = \Bigl \{ \pmb { x } _ { p q } ^ { e } : 1 \leq e \leq N _ { e } , 1 \leq p , q \leq Q \Bigr \} , \qquad | \mathcal { T } _ { Q } | = N _ { e } Q ^ { 2 } .\tag{3.14}
$$

By the chain rule and (3.6), for $\pmb { x } \in \Omega _ { e }$ with $( \xi , \eta ) = F _ { e } ^ { - 1 } ( x )$

$$
\partial _ { x } v _ { i j } ^ { e } ( { \pmb x } ) = \frac { \varphi _ { i } ^ { \prime } ( { \xi } ) \varphi _ { j } ( \eta ) } { J _ { x } ^ { e } } , \qquad \partial _ { y } v _ { i j } ^ { e } ( { \pmb x } ) = \frac { \varphi _ { i } ( { \xi } ) \varphi _ { j } ^ { \prime } ( \eta ) } { J _ { y } ^ { e } } .\tag{3.15}
$$

With $\mathbf { A } \ : = \ : \kappa I$ (assumption 2.1), one Jacobian factor cancels in each stifness term and (3.12) is approximated, as in [26], by

$$
\begin{array} { l } { { \displaystyle \widetilde { \mathcal { R } } _ { i j } ^ { e } ( \theta ) = \kappa J _ { y } ^ { e } \sum _ { p , q = 1 } ^ { Q } w _ { p q } \partial _ { x } \hat { u } ( { \pmb x } _ { p q } ^ { e } ) \varphi _ { i } ^ { \prime } ( \xi _ { p } ) \varphi _ { j } ( \xi _ { q } ) + \kappa J _ { x } ^ { e } \sum _ { p , q = 1 } ^ { Q } w _ { p q } \partial _ { y } \hat { u } ( { \pmb x } _ { p q } ^ { e } ) \varphi _ { i } ( \xi _ { p } ) \varphi _ { j } ^ { \prime } ( \xi _ { q } ) } } \\ { { \displaystyle \qquad + J _ { x } ^ { e } J _ { y } ^ { e } \sum _ { p , q = 1 } ^ { Q } w _ { p q } \Big [ \mathcal { N } \big ( \hat { u } ( { \pmb x } _ { p q } ^ { e } ) \big ) - f ( { \pmb x } _ { p q } ^ { e } ) \Big ] \varphi _ { i } ( \xi _ { p } ) \varphi _ { j } ( \xi _ { q } ) . } } \end{array}\tag{3.16}
$$

Only first derivatives of ̂� enter (3.16), they are obtained by automatic diferentiation at the points of $\tau _ { Q }$ . For the three linear benchmarks, $f \equiv 0$ and ${ \mathcal { N } } \equiv 0$ , so the last line vanishes.

The quadrature rule uses $Q = 2 4$ for the crack, cross and Motz problems, $Q = 4 8$ for the semilinear problem, whose boundary layer of width $\frac { \varepsilon } { \sqrt { 3 } }$ must be integrated accurately and $Q = 8 0$ for the reaction-difusion problem. There the elements have unit side length and the peaks have standard deviation $\sigma _ { \mathrm { p } } = 0 . 0 5$ , so an interval of length $3 \sigma _ { \mathfrak { p } }$ centered at any of the four peaks contains roughly eight to ten nodes per direction. The rule integrates exactly the polynomial factors $\varphi _ { i } \varphi _ { j }$ , whose degree in each variable is at most $2 K + 2 \leq 2 Q - 1$ . The full integrands contain ̂� and are not integrated exactly, the resulting quadrature error is the variational crime analyzed for VPINNs in [28].

## 3.3. Collocation term, constraints and composite loss

The family $\{ \widetilde { \mathcal { R } } _ { i j } ^ { e } \}$ is indexed by $( e , i , j )$ . It is the action of an element of $V _ { K } ^ { \prime }$ , not a function of position. Consistent with section 2.3, there is no map ${ \pmb x } \mapsto \widetilde { \pmb R } ( { \pmb x } )$ to sample. A pure VPINNs therefore admits refinement of the test space (more elements, higher degree, additional patches) however, it has no notion of inserting a point at which the equation is poorly satisfied. This is why adaptive weak-form methods refine test spaces [32–34] or meshes [35] rather than point sets.

We therefore retain the strong-form residual (2.12) on a set of collocation points (the penalizing points of [25]), $\mathcal { T } _ { f } = \{ \mathbf { x } _ { k } ^ { f } \} _ { k = 1 } ^ { N _ { f } } \subset \Omega .$

$$
\mathcal { L } _ { f } ( \theta ; \mathcal { T } _ { f } ) = \frac { 1 } { N _ { f } } \sum _ { k = 1 } ^ { N _ { f } } \Bigl | R ( \hat { u } ) ( \mathbf { x } _ { k } ^ { f } ) \Bigr | ^ { 2 } , \qquad R ( \hat { u } ) = - \kappa \Delta \hat { u } + \mathcal { N } ( \hat { u } ) - f .\tag{3.17}
$$

The second derivatives are obtained by nested automatic diferentiation, so the method is not second-derivative free. The term plays two roles. It penalizes pointwise violation of (2.2), supplementing the element averaged information in (3.16). As it is built on a function of �, it makes point level adaptivity possible, without $ { \mathcal { L } } _ { f }$ , the indicator of section 3.4 would not exist.

It also changes the size of the discrete system. The variational term supplies $N _ { e } K ^ { 2 } \in \{ 1 9 2 , 2 5 6 \}$ scalar conditions against $8 . 9 \times 1 0 ^ { 4 }$ to $2 . 0 \times 1 0 ^ { 5 }$ trainable parameters. With the collocation term the count becomes $N _ { e } K ^ { 2 } + N _ { f }$ , which at the final round lies between $4 . 2 \times 1 0 ^ { 3 }$ (crack and Motz) and $1 . 3 \times 1 0 ^ { 4 }$ (reaction-difusion). The system remains underdetermined in every case.

Boundary and data constraints: With $\mathcal { T } _ { D } = \{ \boldsymbol { x } _ { i } ^ { D } \} _ { i = 1 } ^ { N _ { D } } \subset \Gamma _ { D }$ and $\mathcal { T } _ { N } = \{ \pmb { x } _ { j } ^ { N } \} _ { j = 1 } ^ { N _ { N } } \subset \Gamma _ { N }$ , define

$$
\mathcal { L } _ { D } ( \boldsymbol { \theta } ; \mathcal { T } _ { D } ) = \frac { 1 } { N _ { D } } \sum _ { i = 1 } ^ { N _ { D } } \Bigl | \hat { u } ( \boldsymbol { x } _ { i } ^ { D } ; \boldsymbol { \theta } ) - g ( \boldsymbol { x } _ { i } ^ { D } ) \Bigr | ^ { 2 } ,\tag{3.18}
$$

$$
\mathcal { L } _ { N } ( \pmb { \theta } ; \mathcal { T } _ { N } ) = \frac { 1 } { N _ { N } } \sum _ { j = 1 } ^ { N _ { N } } \Bigl | \kappa \partial _ { n } \hat { u } ( \pmb { x } _ { j } ^ { N } ; \pmb { \theta } ) - h ( \pmb { x } _ { j } ^ { N } ) \Bigr | ^ { 2 } .\tag{3.19}
$$

These enforce (2.3) and restore (2.4), which was lost in remark 3.3. Both sets are drawn once, uniformly along each boundary segment (150-300 points per segment, depending on the problem) and are never resampled. Since (3.19) penalizes the flux, its efective weight on $\partial _ { n } \hat { u }$ is $w _ { N } \kappa ^ { 2 }$ . For the semilinear problem, where $h \equiv 0$ and $\kappa = \varepsilon ^ { 2 }$ , this is small, about $3 \times 1 0 ^ { - 4 }$

For the reaction-difusion problem, $\Gamma _ { D } ~ = ~ \emptyset$ and $\mathcal { L } _ { D }$ is vacuous. In its place, with $N _ { A } = 2 0$ interior anchors $\mathcal { T } _ { A } = \{ \mathbf { x } _ { k } ^ { A } \} _ { k = 1 } ^ { N _ { A } } \subset \Omega$

$$
\mathcal { L } _ { A } ( \pmb { \theta } ; \mathcal { T } _ { A } ) = \frac { 1 } { N _ { A } } \sum _ { k = 1 } ^ { N _ { A } } \Bigl | \hat { u } ( \pmb { x } _ { k } ^ { A } ; \pmb { \theta } ) - u ( \pmb { x } _ { k } ^ { A } ) \Bigr | ^ { 2 } ,\tag{3.20}
$$

where � is the exact solution of section 5.3. The anchors sit at the four peak centers and at the four points one peak width $\sigma _ { \mathfrak { p } }$ from each center along the coordinate directions. They select the branch (remark 2.3) and $\mathcal { L } _ { A }$ enters (3.21) with weight $w _ { D }$

Composite loss: The training objective is

$$
\begin{array} { r } { \mathcal { L } ( \boldsymbol { \theta } ) = { w _ { v } } \mathcal { L } _ { v } + { w _ { D } } \mathcal { L } _ { D } + { w _ { N } } \mathcal { L } _ { N } + { w _ { f } } \mathcal { L } _ { f } , } \end{array}\tag{3.21}
$$

$$
\mathcal { L } _ { v } ( \pmb { \theta } ; \mathcal { T } _ { Q } ) = \frac { 1 } { N _ { e } K ^ { 2 } } \sum _ { e = 1 } ^ { N _ { e } } \sum _ { i , j = 1 } ^ { K } \left| \widetilde { \mathcal { R } } _ { i j } ^ { e } ( \pmb { \theta } ) \right| ^ { 2 } ,\tag{3.22}
$$

with $\mathcal { L } _ { D }$ replaced by $\mathcal { L } _ { A }$ for the reaction-difusion problem. Each term is a mean rather than a sum, so the weights keep their meaning as $N _ { f }$ grows during refinement. The weights are $( w _ { v } , w _ { D } , w _ { N } , w _ { f } ) = ( 1 , 5 0 , 5 0 , 1 0 )$ for all problems except reaction-difusion, where they are (1, 100, 50, 5). They were fixed by preliminary experimentation, were not tuned per run and are identical in the two compared arms.

## 3.4. Residual-based adaptive refinement

Indicator: After each training stage, a candidate pool $c \subset \Omega$ with $| C | \gg N _ { f }$ is drawn uniformly at random and the indicator

$$
\rho ( { \pmb x } ; { \pmb \theta } ) = \Big | R ( { \hat { \boldsymbol u } } ) ( { \pmb x } ) \Big | = \Big | - \kappa \Delta { \hat { \boldsymbol u } } ( { \pmb x } ; { \pmb \theta } ) + \mathcal { N } \big ( { \hat { \boldsymbol u } } ( { \pmb x } ; { \pmb \theta } ) \big ) - f ( { \pmb x } ) \Big | , \qquad { \pmb x } \in C ,\tag{3.23}
$$

is evaluated on it. The indicator requires no derivatives beyond those in (3.17). It is not an error estimator and no reliability or eficiency property is claimed for it. Its justification is that it can be evaluated at an arbitrary point, which (3.16) cannot. Near $S ^ { * }$ , by remark 2.4, large values of $\rho$ reflect both approximation error and the $r ^ { \lambda _ { 1 } - 2 }$ growth of the second derivatives and $\rho$ alone does not separate the two.

Guard: Where this growth is severe, candidates close to $S ^ { * }$ are excluded,

$$
C _ { \mathrm { a d m } } = \Bigl \{ { \pmb x } \in { \cal C } : \mathrm { d i s t } ( { \pmb x } , { \pmb S } ^ { * } ) \geq r _ { \mathrm { g } } \Bigr \} .\tag{3.24}
$$

For the crack, $r _ { \mathrm { g } } = 2 \times 1 0 ^ { - 3 }$ . At this radius $| \partial _ { x x } \hat { u } | , | \partial _ { \nu \nu } \hat { u } | \sim r ^ { - 3 / 2 } \approx 1 . 1 \times 1 0 ^ { 4 }$ , so the round of error in their sum is of order $1 0 ^ { - 1 2 }$ . Inside the disk, the computed $\rho$ would rank points by round of error.

For the Motz problem the growth is genuine rather than numerical. On a trained network, the median of $\rho$ rises monotonically from $4 . 5 \times 1 0 ^ { - 4 }$ at $r = 0 . 5$ to 56 at $r = 1 0 ^ { - 2 }$ . Ranking against this five decade spread sends the whole insertion budget to the smallest admissible radius. The pool is therefore restricted by $r _ { \mathrm { g } } = 5 \times 1 0 ^ { - 2 }$ , placed where the median indicator (about 0.5) is comparable to the 90th percentile at larger radii, this compresses the spread to about three decades. All other interior samples of the Motz problem, namely the initial set and the insertions of the uniform arm, satisfy only $r \geq 2 \times 1 0 ^ { - 3 }$ . The two Motz arms therefore difer in the region from which insertions may be drawn.

For the semilinear problem, $r _ { \mathrm { g } } = 2 \times 1 0 ^ { - 3 }$ is applied to every interior sample in both arms. No guard is applied to the cross problem, whose second derivatives grow only like $r ^ { - 4 / 3 }$ , nor to the reaction-difusion problem, whose solution is smooth. Where no guard is used, ${ \mathcal { C } } _ { \mathrm { a d m } } = { \mathcal { C } }$

Selection: Let $\pmb { x } _ { ( 1 ) } , \ldots , \pmb { x } _ { ( n ) }$ , with $n = | C _ { \mathrm { a d m } } |$ , enumerate ${ \mathcal { C } } _ { \mathrm { a d m } }$ in nonincreasing order of $\rho .$ Given the insertion count �, set

$$
\mathcal { M } = \{ \pmb { x } _ { ( 1 ) } , \hdots , \pmb { x } _ { ( m ) } \} , \qquad \mathcal { T } _ { f } \gets \mathcal { T } _ { f } \cup \mathcal { M } .\tag{3.25}
$$

The pool is redrawn in every round and the procedure is repeated for $n _ { \mathrm { R A R } }$ rounds.

Radius-normalized selection for the crack: At the crack tip, $\rho$ grows approximately like $r ^ { - 2 . 3 }$ over four decades on a trained network. In a preliminary run, the ranking (3.25) placed 92% of the first round insertions in $r < 0 . 1$ , in tight clusters and the relative $L ^ { 2 }$ error increased. For this problem the ranking is therefore replaced by three steps.

First (radial normalization), with $r ( { \pmb x } )$ the distance from � to the crack tip, ${ \mathcal { C } } _ { \mathrm { a d m } }$ is divided into $n _ { b } = 1 2$ radial bands in � with logarithmically spaced edges between $r _ { \mathrm { g } }$ and 1.5. Each indicator value is normalized by the median over its own band,

$$
{ \hat { \rho } } ( { \boldsymbol { \mathbf { \mathit { x } } } } ) = \rho ( { \boldsymbol { \mathbf { \mathit { x } } } } ) { \big / } \operatorname { m e d } \{ \rho ( { \boldsymbol { \mathbf { \mathit { y } } } } ) : { \boldsymbol { \mathbf { \mathit { y } } } } \in C _ { \mathrm { a d m } } , { \boldsymbol { \mathbf { \mathit { y } } } } { \mathrm { ~ i n ~ t h e ~ b a n d ~ o f ~ } } { \boldsymbol { \mathbf { \mathit { x } } } } \} .\tag{3.26}
$$

Second (density sampling), min $n , \operatorname* { m a x } \{ 8 m , m + 1 0 0 0 \} \}$ candidates are drawn without replacement with probability proportional to $\hat { \rho } ( \pmb { x } ) / \overline { { \hat { \rho } } } + 1$ , where $\overline { { \hat { \rho } } }$ is the mean of $\hat { \rho }$ over ${ \mathcal { C } } _ { \mathrm { a d m } }$ . This is the residual-based adaptive distribution of [37] with exponent 1 and ofset 1. Third (minimum-separation filter), the drawn candidates are visited in decreasing order of ${ \hat { \rho } } . \mathrm { A }$ candidate is accepted if it lies at distance at least 0.35 max $\{ r ( \pmb { x } ) , r _ { \mathrm { g } } \}$ from every point already accepted. If fewer than � points are accepted, the remaining drawn candidates fill the set.

Uniform arm: The uniform arm, against which the RAR arm is compared, replaces the selection step by � points drawn uniformly from the same admissible region and leaves everything else unchanged; the indicator is not evaluated. The only exception to identical admissible regions is the Motz problem, as described under Guard.

Algorithm 1 VC-PINNs with residual-based adaptive refinement (VC-PINNs-RAR)   
Require: data $f , g , h ;$ network $( 3 . 1 ) \ – ( 3 . 5 ) ;$ partition  and test space $V _ { K }$ (3.11); quadrature set $\tau _ { Q }$ ; point sets   
$\mathcal { T } _ { D } , \mathcal { T } _ { N } , \mathcal { T } _ { f }$ (and $\tau _ { A } ) ;$ weights $w _ { v } , w _ { D } , w _ { N } , w _ { f } ;$ rounds $n _ { \mathrm { R A R } } ;$ insertion count �; pool size ||; guard radius $r _ { \mathrm { g } } ;$   
iteration counts $N _ { \mathrm { A d a m } } ^ { 0 } , N _ { \mathrm { A d a m } } , N _ { \mathrm { L B F G S } } .$   
1: Initialize �.   
2: Train $N _ { \mathrm { A d a m } } ^ { 0 }$ Adam iterations on (3.21), then up to $N _ { \mathrm { L B F G S } }$ L-BFGS iterations.   
3: for $k = \overbrace { 1 , \ldots } ^ { } , n _ { \mathrm { R A R } }$ do   
4: Draw $c \subset \Omega$ and form ${ \mathcal { C } } _ { \mathrm { a d m } }$ by (3.24).   
5: Evaluate $\rho$ on ${ \mathcal { C } } _ { \mathrm { a d m } }$ by (3.23).   
6: Select  by (3.25) (crack: by (3.26) and the steps following it); set $\mathcal { T } _ { f }  \mathcal { T } _ { f } \cup \mathcal { M } .$   
7: Train $N _ { \mathrm { A d a m } }$ Adam iterations, then up to $N _ { \mathrm { L B F G S } }$ L-BFGS iterations, on the updated (3.21).   
8: end for   
9: return $\hat { u } ( \cdot ; \theta ) .$

## 3.5. Algorithm, scope and training budget

Algorithm 1 is written in execution order. In each round the pool is scored and $\tau _ { f }$ enlarged, using the parameters left by the previous stage. The algorithm acts on $\boldsymbol { \tau } _ { f }$ and hence on $\scriptstyle { \mathcal { L } } _ { f }$ and on nothing else. The partition , the number � of test functions per direction, the quadrature set $\tau _ { Q }$ , the sets $\mathcal { T } _ { D } , \mathcal { T } _ { N } , \mathcal { T } _ { A }$ , the architecture and the weights are fixed for the whole run. In particular, for given $\theta , \mathcal { L } _ { v }$ is the same function in the last round as in the first. The method performs neither ℎ- nor �-refinement.

The loss (3.21) is minimized by Adam [49] followed by L-BFGS [50] with a strong Wolfe line search [51] in every stage. Adam is restarted at each stage, with learning rate $1 0 ^ { - 3 }$ in the first stage and $1 0 ^ { - 4 }$ thereafter, exponential decay by a factor of 0.9 per 5000 iterations and gradient norm clipping at 1. L-BFGS uses a history of 50, a gradient tolerance of $1 0 ^ { - 1 4 }$ and a change tolerance of $1 0 ^ { - 2 0 }$ and is capped at $N _ { \mathrm { L B F G S } }$ iterations per stage. Training ends when the prescribed stages are complete. There is no early stopping and all reported errors are those of the final iterate.

The two arms of every benchmark share the seed, the architecture, the weights, the partition, the test space $V _ { K }$ and the quadrature rule. They also share the collocation schedule: $N _ { f } ^ { 0 }$ initial points, � insertions in each of $n _ { \mathrm { R A R } }$ rounds and hence the same $N _ { f }$ in every round. Finally, they share the optimizer schedule: $N _ { \mathrm { A d a m } } ^ { 0 } + n _ { \mathrm { R A R } } N _ { \mathrm { A d a m } }$ Adam iterations and $1 + n _ { \mathrm { R A R } } \mathrm { L } \mathrm { - } \mathrm { B i } \mathrm { F G S }$ stages. The arms therefore difer only in how the � points of each round are chosen and for the Motz problem, in the admissible region. The problem specific values are given with each benchmark in section 5.

## 4. Architecture of the VC-PINNs-RAR framework

Figure 1 summarizes the proposed VC-PINNs-RAR framework. It assembles the four components of section 3 a neural trial function, a discrete variational residual tested against local polynomials on a partitioned domain, a composite loss carrying the mixed boundary data and the refinement loop. In the forward pass, the coordinate $\boldsymbol { x } \in \Omega$ is enriched, mapped by the multiscale Fourier-feature embedding and propagated to $\hat { u } ( { \pmb x } ; { \pmb \theta } )$ . Automatic diferentiation supplies ∇ ̂� for the variational residual and $\Delta \hat { u }$ for the strong-form residual. The weighted sum of both residuals and the boundary penalties is minimized by Adam followed by L-BFGS. Between stages, the loop augments the interior collocation set and training continues from the current parameters.

The trial function $\hat { u } = c _ { u } \mathcal { F } _ { \theta ^ { \circ } } \gamma \circ \Psi$ composes a problem specific enrichment Ψ, the multiscale Fourier-feature embedding � of (3.3) and the residual tanh network ${ \mathcal { F } } _ { \theta }$ of $( 3 . 4 ) \ – ( 3 . 5 )$ , whose hidden layers carry skip connections $\pmb { h } ^ { ( i ) } = \pmb { h } ^ { ( i - 1 ) } + \mathrm { t a n h } ( \cdot )$ . The frequency matrices $\pmb { B } ^ { ( s ) }$ , whose entries are independent draws from the normal distribution with mean 0 and variance $\sigma _ { s } ^ { 2 }$ , are sampled once at $S = 3$ scales and then held fixed, so they belong to the architecture rather than to �. The multiple scales reduce the spectral bias of the network [17, 18, 44] and sharpen its resolution of localized features. All problems are steady, so the input is the spatial coordinate alone. Where the corner behavior is known, Ψ appends the leading singular mode: $r ^ { 2 / 3 }$ on the cross-shaped and L-shaped domains, $r ^ { 1 / 2 } \cos ( \theta / 2 )$ for the Motz problem and the pair $( r ^ { \top } { } ^ { 2 } \cos ( \theta / 2 ) , r ^ { 1 / 2 }$ <sub>sin(�∕2)</sub>) <sub>on the slit domain. The pair also carries the jump across the</sub> crack faces, which the plain coordinates cannot represent.

Following the Petrov-Galerkin construction, the test functions are drawn from $V _ { K } \subset V$ of (3.11). The domain is partitioned into nonoverlapping elements, $\overline { { \Omega } } ~ = ~ \bigcup _ { e } \overline { { \Omega } } _ { e }$ and on each element the test functions are the tensor products (3.10) of the one-dimensional Legendre bubble functions $\varphi _ { 1 } , \ldots , \varphi _ { K }$ of (3.7). The partition, the number of test functions per direction $K = 8$ and the quadrature order are fixed for the entire run. Since $\varphi _ { i } ( \pm 1 ) = 0$ , every test function vanishes on $\partial \Omega _ { e } ,$ so no interelement or boundary term survives and the natural boundary functional vanishes identically, $B _ { i j } ^ { e } \equiv 0$ on $V _ { K }$ (remark 3.3). Testing therefore leaves the three term residual (3.16), split as

$$
\widetilde { \mathscr { R } } _ { i j } ^ { e } ( \pmb { \theta } ) = \widetilde { \mathscr { R } } _ { i j } ^ { e , \mathrm { d i f f } } ( \pmb { \theta } ) + \widetilde { \mathscr { R } } _ { i j } ^ { e , \mathrm { r e a c } } ( \pmb { \theta } ) - \widetilde { \mathscr { R } } _ { i j } ^ { e , \mathrm { s r c } } ,\tag{4.1}
$$

with $\widetilde { \mathcal { R } } _ { i j } ^ { e , \mathrm { d i f f } }$ the difusion contribution, $\mathcal { \widetilde { R } } _ { i j } ^ { e , \mathrm { r e a c } }$ that of the nonlinear reaction term (absent when the problem is linear) and ${ \widetilde { \mathcal { R } } } _ { i j } ^ { e , \mathrm { { s r c } } }$ the source. Element integrals use the tensor-product Gauss-Legendre rule (3.14) with $Q > K$ nodes per direction: $Q = 2 4$ by default, $Q = 4 8$ where a boundary layer must be integrated and $Q = 8 0$ for the localized peaks of the reaction-difusion problem.

Neither boundary condition is natural in the discrete formulation, so both are imposed by explicit penalties in (3.21): $\mathcal { L } _ { D }$ on $\tau _ { D }$ and $\mathcal { L } _ { N }$ on $\tau _ { N }$ , the latter penalizing the flux � $\partial _ { n } { \hat { u } } - h$ . With $w _ { v }$ for the variational residual and $w _ { f }$ for the collocation residual, the formulation carries four weights, given in section 3.3. The parameters are obtained by minimizing (3.21) with Adam followed by L-BFGS with a strong Wolfe line search, one such pair per training stage.

The quadrature set $\tau _ { Q }$ is fixed and the only object the refinement modifies is the interior collocation set $\boldsymbol { \tau } _ { f }$ . After each stage, the indicator $\rho = | R ( { \hat { u } } ) |$ of (3.23) is evaluated on a candidate pool $c \subset \Omega$ to identify where the governing equation is least well satisfied. The strong-form residual is used since it can be evaluated at an arbitrary point, whereas the variational residual is indexed by element and test mode moreover, points inserted into $\boldsymbol { \tau } _ { f }$ act on training through $ { \mathcal { L } } _ { f }$ . Where the singular growth of the second derivatives is severe, candidates inside the guard radius $r _ { \mathrm { g } }$ of the singular point are discarded, as there the indicator grows toward the singular point regardless of the local error (remark 2.4 and section 3.4). The � highest ranked admissible points, selected by (3.25), are appended to $\tau _ { f }$ and training continues for $n _ { \mathrm { R A R } }$ rounds (algorithm 1). The uniform arm difers only in this selection step: it draws the same number of points uniformly from the same admissible region and reaches the same final $N _ { f }$ under the same optimizer schedule. The two exceptions, the admissible region of the Motz problem and the uniform arm of the reaction-difusion problem, are stated in section 3.5. section 5 compares the two arms.

## 5. Numerical examples

We compare the VC-PINNs with uniform insertion (the uniform arm) and with residual-based adaptive refinement (VC-PINNs-RAR, the RAR arm) on five second-order elliptic benchmarks whose solutions are singular, nearly singular, or strongly localized. The first problem is the Laplace equation on a slit square with traction-free crack faces, whose tip has interior angle 2�. The second is the Laplace equation on a cross-shaped domain with four re-entrant corner singularities and mixed Dirichlet-Neumann boundary conditions. The third is an Allen-Cahn type reactiondifusion problem [53] on the square with homogeneous Neumann data and four localized Gaussian peaks. The fourth is a monotone semilinear problem on an L-shaped domain, in which a re-entrant corner singularity exists with a thin boundary layer. The fifth is the classical Motz problem, whose singularity arises where the boundary condition changes type on a straight edge. Exact solutions, closed form or manufactured, are available for the first three problems, for the fourth and fifth, a high resolution finite element reference is used. In every benchmark problem the uniform collocation and RAR arms are budget matched, with the same final collocation count, the same total Adam iterations and the same number of L-BFGS stages, they difer in how the inserted points are chosen and for the Motz problem, in the region from which they are drawn.

All experiments are run in PyTorch [54] in double precision on a single NVIDIA RTX 5080 GPU, with Adam followed by L-BFGS (section 3.5). The loss weights were $( w _ { v } , w _ { D } , w _ { N } , w _ { f } ) = ( 1 , 5 0 , 5 0 , 1 0 )$ for all benchmarks except the reaction-difusion problem, where $( w _ { v } , w _ { D } , w _ { N } , w _ { f } ) = ( 1 , 1 0 0 , 5 0 , 5 )$ was used, there $w _ { D }$ acts on interior anchor points rather than on a Dirichlet boundary.

## 5.1. Laplace equation on a slit domain with traction-free crack faces

We consider the Laplace equation on the slit square

$$
\Delta u = 0 \mathrm { i n } \Omega = ( - 1 , 1 ) ^ { 2 } \setminus \Sigma \mathrm { w h e r e } \Sigma = \{ ( x , 0 ) : 0 \leq x \leq 1 \} ,\tag{5.1}
$$

with homogeneous Neumann data prescribed on the two faces of the slit and Dirichlet data taken from the exact solution on the outer boundary. This problem models antiplane shear (Mode III) deformation of a cracked elastic body [55]. In

![](images/d33b3eea2003d1e97b7e0d9b55d5a83905fb822b9dde763b1145f5ccab6b27a1.jpg)  
Fig. 1. Architecture of the proposed framework. A multiscale Fourier-feature embedding $\gamma ,$ whose frequency matrices are drawn once and held fixed, acts on the enriched coordinate $z = \Psi ( x )$ and feeds a residual tanh network whose output, scaled by $c _ { u } ,$ , is the trial function ̂�(�; �), Ψ appends the leading singular mode where the corner behavior is known and is the identity otherwise. One application of automatic diferentiation supplies ∇ ̂� for the variationa residual $\widetilde { \mathcal { R } } _ { i j } ^ { e }$ , assembled on the fixed quadrature set $\tau _ { Q }$ against the Legendre bubble test functions $v _ { i j } ^ { e }$ , a second, nested application supplies $\Delta \hat { u }$ for the strong-form residual �( ̂�) on the collocation set $\tau _ { f }$ . As every $v _ { i j } ^ { e }$ vanishes on $\partial \Omega _ { e }$ , the natural boundary functional $B _ { i j } ^ { e }$ vanishes identically and the Neumann data reach the loss only through the penalty $\mathcal { L } _ { N }$ . The composite loss is the weighted sum of the variational residual, the collocation term and the two boundary penalties. The RAR loop branches from the strong-form residual, the only loss component that can be evaluated at an arbitrary interior point. It scores a candidate pool $^ { c , }$ discards candidates inside the guard radius $r _ { \mathrm { g } } ,$ inserts the � highest ranked points into $\tau _ { f }$ and continues training from the current parameters. The element partition, the number $K$ of test functions per direction and the quadrature set $\tau _ { Q }$ are fixed throughout, so the scheme performs neither ℎ- nor �-refinement, adaptivity acts on $\tau _ { f }$ alone.

crack-tip polar coordinates with the branch cut aligned with the slit, $\theta \in ( 0 , 2 \pi )$ , the exact solution is

$$
u ( r , \theta ) = r ^ { 1 / 2 } \cos \Big ( \frac { \theta } { 2 } \Big ) .\tag{5.2}
$$

The crack tip has interior angle $\omega = 2 \pi$ , so the leading singular exponent is $\begin{array} { r } { \lambda _ { 1 } = \frac { \pi } { \omega } = \frac { 1 } { 2 } ( \mathrm { t a b l e } 1 ) } \end{array}$ . Hence $u \in H ^ { s } ( \Omega )$ for every $s < 3 / 2$ , however, $u \not \in H ^ { 2 } ( \Omega )$ and $\left| \nabla u \right| \sim r ^ { - 1 / 2 }$ as $r  0$

This problem presents two major challenges for neural-network solvers. First, smooth activation functions struggle to approximate the non smooth behavior $r ^ { 1 / 2 }$ near the crack tip, causing the approximation error to localize around the singularity while the optimization process remains dominated by the smooth bulk region. Second, � jumps across the slit, taking the values $\pm \sqrt { x }$ on the two faces and no network that is continuous in $( x , y )$ can represent this jump, whatever its capacity (remark 3.1).

Crack problem on the slit square (Mode III). All errors are reported in percent $( \% )$ and bold marks the more accurate arm in each row. Both arms use the same collocation schedule $( \vert \mathcal { T } _ { f } \vert = 1 5 0 0  3 9 0 0 )$ , 150,000 Adam iterations and thirteen L-BFGS stages, one training run per arm. Errors are evaluated on a uniform $3 0 1 \times 3 0 1$ grid on $[ - 1 , 1 ] ^ { 2 }$ . The relative $L ^ { \infty }$ error is max $\vert u - \hat { u } \vert /$ max � over the grid, where max $| u | = 2 ^ { 1 / 4 } \cos ( \pi / 8 ) \approx 1 . 0 9 9$ is attained at the corners (1, ±1), the $H ^ { 1 }$ semi-norm excludes grid points with $r \leq 1 0 ^ { - 3 }$
<table><tr><td>Metric</td><td>VC-PINNs (uniform)</td><td>VC-PINNs-RAR</td></tr><tr><td>Relative  $L ^ { 2 }$  error</td><td>0.4138</td><td>0.3545</td></tr><tr><td>Relative  $L ^ { \infty }$  error</td><td>2.462</td><td>1.889</td></tr><tr><td>Relative  $H ^ { 1 }$  semi-norm error</td><td>2.374</td><td>1.442</td></tr></table>

To address these issues, the computational domain is partitioned into four rectangular elements that meet on the coordinate axes, with the slit on an element interface, so that no element crosses the discontinuity. In addition, the harmonic crack-tip coordinates

$$
s _ { 1 } = r ^ { 1 / 2 } \cos \left( \frac { \theta } { 2 } \right) , \qquad s _ { 2 } = r ^ { 1 / 2 } \sin \left( \frac { \theta } { 2 } \right) ,\tag{5.3}
$$

are appended to the network input. These enriched features encode both the leading singular mode and the branch cut discontinuity that cannot be represented using (�, �) alone.

The trial space is parameterized by a multiscale Fourier-feature embedding with $n _ { F } = 6 4$ frequencies at each of the scales $\{ \sigma _ { 1 } , \sigma _ { 2 } , \sigma _ { 3 } \} = \{ 1 , 5 , 1 5 \}$ , followed by six tanh hidden layers of width 100, the last five with residual connections. The test space uses $K = 8$ Legendre bubble functions and $Q = 2 4$ Gauss-Legendre quadrature points per direction. The training objective combines the variational residual, boundary penalties and a strong-form collocation term,

$$
\mathcal { L } = \mathcal { L } _ { v } + w _ { D } \mathcal { L } _ { D } + w _ { N } \mathcal { L } _ { N } + w _ { f } \mathcal { L } _ { f } ,\tag{5.4}
$$

with weights $( w _ { v } , w _ { D } , w _ { N } , w _ { f } ) = ( 1 , 5 0 , 5 0 , 1 0 )$ . Training runs for thirteen stages, an initial stage followed by twelve rounds, with each stage combining an Adam phase and a subsequent L-BFGS phase. The collocation set starts with 1500 points and gains 200 points per round, ending at $| \mathscr { T } _ { f } | = 3 9 0 0$ after 150,000 Adam iterations in total.

The uniform and RAR arms difer only in how the newly inserted collocation points are selected. In the uniform arm, points are sampled uniformly throughout the admissible region. The RAR arm instead scores a candidate pool of $1 0 ^ { 5 }$ points by the strong-form residual. Direct ranking by $| R |$ is inefective. The exact solution is harmonic, so the residual equals, up to sign, the Laplacian of the approximation error, which grows approximately like $r ^ { - 2 . 3 }$ near the crack tip (section 3.4). The largest residuals therefore mark proximity to the singularity rather than large local error.

To mitigate this efect, points are selected by the three step rule of section 3.4, radial normalization of the residual by its median in twelve logarithmically spaced bands (3.26), density sampling and a minimum-separation filter, here with $r _ { \mathrm { g } } = 2 \times 1 0 ^ { - 3 }$

Both arms resolve a solution with an unbounded gradient and a jump across the slit, with relative $L ^ { 2 }$ errors below 0.5% (table 3). This indicates that the element partition and the crack-tip enrichment together capture the singular structure of the solution; as $s _ { 1 }$ coincides with the exact solution (5.2), the remaining error reflects the discretization rather than the representation of the singularity. At identical collocation and iteration budgets, RAR reduces the relative $L ^ { 2 }$ error by 14% (a factor of 1.17), the relative $L ^ { \infty }$ error by 23% (a factor of 1.30) and the relative $H ^ { 1 }$ semi-norm error by 39% (a factor of 1.65). The largest improvement occurs in the $H ^ { 1 }$ semi-norm, which measures the error in the gradient, where the singularity resides.

The spatial error distributions show where these improvements occur. With uniform insertion, the error remains concentrated around the crack tip and extends into a broad lobe away from the singular region (Fig. 2C). RAR suppresses these secondary structures and produces a nearly radially symmetric error pattern around the crack tip (Fig. 2I). This is consistent with the adaptive strategy allocating collocation points to regions where the approximation error is most significant and thus making more efective use of the same collocation budget.

The adaptive collocation set is graded toward the crack tip while maintaining adequate coverage of the far field (Fig. 2F). This behavior closely resembles mesh grading in adaptive finite element methods. In the loss histories (Fig. 2J and K), the jump in $ { \mathcal { L } } _ { f }$ at an insertion is larger in the RAR arm in most rounds; its inserted points are chosen for their large residual.

A  
![](images/14d31a288b9c1abb660b5adf334a238c6d235bec5ed839a51d929e442d743bdb.jpg)

B  
![](images/1a651c23e2cd26ea1c1c553d3741fd62f717ca84008d81f5d39548a35bca3834.jpg)

C  
![](images/ccf88b26420b2259a0012e340121ac2eaf6949b48bd841a23363c53dd011e247.jpg)

D  
![](images/be121916b58265ece02aba900b543300470ce3ee260c158f01117dd07e0e0cce.jpg)

E  
![](images/26baa0b48e97b3fd5d850915bed710b0dc55d1adadf4feb176890b6fa01c02b7.jpg)

F  
![](images/a998eb7b859f70600a42c13a3905928d9ff95829fba58f17f61ce5aa48fa19c1.jpg)

G  
![](images/155a424439b5c4c736ce9fbfc42baaa4a4eb77e9c3ed2d2293d5690a8495e350.jpg)

H  
![](images/95f5908d225ef5254ca161084707703e13bcb1d07278558b8e14a6229849fee2.jpg)

I  
![](images/6f23ea9c563c1db3df95bafe8016a52279929b3a23e149385bf538b15bb1579e.jpg)

J  
![](images/c96371dc74451fa93bb19233a534d54bc42e584126ecf2c69125db335443e5ac.jpg)

K  
![](images/b6fc2885505f0e5a2ad584ac2fbb73c98974e9f41e72ac3071c091da8c016658.jpg)  
Fig. 2. Crack problem on the slit square (Mode III). (A-C) Exact solution, VC-PINNs (uniform) prediction and pointwise absolute error. (D-F) Domain with boundary conditions and element partition, uniform collocation set and adaptive collocation set after twelve rounds, $| \mathcal { T } _ { f } | = 1 5 0 0  3 9 0 0$ . (G-I) Exact solution, VC-PINNs-RAR prediction and pointwise absolute error; panels C and I have separate color scales. (J-K) Adam loss components for the uniform and the RAR arm.

Cross problem. All errors are reported in percent (%) and bold marks the more accurate arm in each row. Both arms use the same collocation schedule $( | T _ { f } | = 8 0 0 0 \to 1 0 , \dot { 4 } 0 \dot { 0 } )$ , 150,000 Adam iterations and thirteen L-BFGS stages; one training run per arm. Errors are evaluated at the points of a uniform $3 0 1 \times 3 0 1$ grid on $[ - 1 , 1 ] ^ { 2 }$ that lie in $\overline { { \Omega } } .$ The relative $L ^ { \infty }$ error is max $\vert u - \hat { u } \vert /$ max |�| over these points, where max $| u | \approx 3 . 5 7 7 ;$ the $H ^ { 1 }$ semi-norm uses the network gradient from automatic diferentiation and excludes points within $1 0 ^ { - 3 }$ of a corner.
<table><tr><td>Metric</td><td>VC-PINNs (uniform)</td><td>VC-PINNs-RAR</td></tr><tr><td>Relative  $L ^ { 2 }$  error</td><td>5.078</td><td>0.1601</td></tr><tr><td>Relative  $L ^ { \infty }$  error</td><td>13.85</td><td>1.283</td></tr><tr><td>Relative  $H ^ { 1 }$  semi-norm error</td><td>32.36</td><td>2.248</td></tr></table>

## 5.2. Laplace equation on a cross-shaped domain with four re-entrant corners

This example tests whether RAR can resolve several singularities simultaneously and more stringently, rank them: the four corners of the cross share the same exponent yet carry diferent amplitudes, so an efective adaptive scheme must distribute points in proportion to their strength rather than merely locate the singular points.

We consider $\Delta u = 0$ on the cross-shaped domain $\Omega = \left( ( - 1 , 1 ) \times ( - \frac { 1 } { 3 } , \frac { 1 } { 3 } ) \right) \cup \left( ( - \frac { 1 } { 3 } , \frac { 1 } { 3 } ) \times ( - 1 , 1 ) \right)$ , the union of a horizontal and a vertical bar. Its four re-entrant corners $\begin{array} { l l } { { C _ { 1 } = ( - { \frac { 1 } { 3 } } , { \frac { 1 } { 3 } } ) , C _ { 2 } = ( { \frac { 1 } { 3 } } , { \frac { 1 } { 3 } } ) , C _ { 3 } = ( { \frac { 1 } { 3 } } , - { \frac { 1 } { 3 } } ) } } \end{array}$ and $C _ { 4 } = ( - \frac { 1 } { 3 } , - \frac { 1 } { 3 } )$ each have interior angle $\begin{array} { r } { \omega = \frac { 3 \pi } { 2 } } \end{array}$ ; by classical corner regularity theory [4], the leading singular exponent is therefore $\begin{array} { r } { \lambda _ { 1 } = \frac { \pi } { \omega } = \frac { 2 } { 3 } ( { \mathrm { t a b l e } } 1 ) } \end{array}$ . Dirichlet data are prescribed on the four bar ends and Neumann data on the eight side walls, both read from the exact solution

$$
u = \sum _ { i = 1 } ^ { 4 } c _ { i } r _ { i } ^ { 2 / 3 } \sin \Bigl ( \frac { 2 } { 3 } \bigl ( \theta _ { i } - \theta _ { 0 , i } \bigr ) \Bigr ) , \qquad ( c _ { 1 } , c _ { 2 } , c _ { 3 } , c _ { 4 } ) = ( 1 , 0 . 5 , 2 , 0 . 7 5 ) ,\tag{5.5}
$$

where $( r _ { i } , \theta _ { i } )$ are polar coordinates at $C _ { i }$ with the angular branch cut placed along the bisector of the removed quadrant and $\theta _ { 0 , i }$ is the angle of one adjacent edge. Each mode is harmonic, so $f \equiv 0$ exactly; each also vanishes on both edges meeting at its own corner, since sin $\begin{array} { r } { ( \frac { 2 } { 3 } \cdot \frac { 3 \pi } { 2 } ) = 0 } \end{array}$

Near $C _ { i }$ the remaining three terms are analytic, so the local behavior is governed by the �-th mode alone. Hence $u \in H ^ { s } ( \Omega )$ for every $s < 5 / 3 ;$ however, $u \not \in H ^ { 2 } ( \Omega )$ , with $\begin{array} { r } { | \nabla u | \sim \frac { 2 } { 3 } c _ { i } r _ { i } ^ { - 1 / 3 } } \end{array}$ as $r _ { i } \to 0$ . The regularity is lost at four singular points at once and each corner’s gradient blow-up scales with its own amplitude. Unlike benchmarks whose dificulty is manufactured through a localized source, the obstruction here comes from the geometry rather than the data. The problem models steady heat conduction in a cross-shaped section with a prescribed heat flux on the side walls and prescribed temperatures at the bar ends, the re-entrant corners being the sites of flux concentration.

For uniform collocation, the dificulty is that of the single-corner case multiplied by four: a uniform budget assigns each corner neighborhood a share proportional only to its area and the four singular regions divide that share among them. Since $f \equiv 0$ , the strong-form residual measures purely the failure of the network to be harmonic and therefore peaks at the corners. The unequal amplitudes turn this into a ranking problem: the strongest corner $C _ { 3 }$ contributes most to the global loss and the weakest, $C _ { 2 }$ , least.

The domain is tiled by three rectangular elements, the full vertical bar together with the left and right side bars, so that no element leaves Ω. The network input is enriched with the powers $\bar { r } _ { i } ^ { 2 / 3 }$ of the four corner distances, which expose the radial rate of every singularity while leaving the angular structure to be learned. The trial space is a multiscale Fourier-feature embedding followed by six tanh hidden layers of width 100, the last five with residua connections; the test space uses $K = 8$ Legendre bubble functions and $Q = 2 4$ Gauss-Legendre nodes per direction. The objective combines the variational residual, boundary penalties and a strong-form collocation term with weights $( w _ { v } , w _ { D } , w _ { N } , w _ { f } ) = ( 1 , 5 0 , 5 0 , 1 0 )$ , using 300 Dirichlet points per bar end and 200 Neumann points per side wall. Both arms start from the same 8000 collocation points, insert 200 points in each of twelve rounds to reach $| \mathcal { T } _ { f } | = 1 0 \small { , } 4 0 0$ and are trained for 150,000 Adam iterations and thirteen L-BFGS stages. They difer only in placement: the uniform arm draws the inserted points uniformly and the RAR arm takes the 200 largest-residual points from a candidate pool of 25,000 points.

RAR reduces the relative $L ^ { 2 }$ error from 5.078% to 0.1601% (a factor of about 32), the relative $H ^ { 1 }$ semi-norm error by a factor of about 14 and the relative $L ^ { \infty }$ error by a factor of about 11 (table 4; Fig. 3C and I). The remaining error of the RAR arm is largest around $C _ { 1 }$ . The inserted points (Fig. 3F) reach all four corners, most densely $C _ { 3 }$ , the corner of largest amplitude; however, a large share of them lies along the bar ends and side walls and in small clusters in the interior. Since the corner locations are already encoded in the input features, this example tests how the residual distributes points among the four corners rather than whether it finds them.

## 5.3. Nonlinear reaction-difusion with four localized peaks

The preceding examples carry singularities whose locations are dictated by the geometry and are supplied to the network through input enrichment. This example removes both advantages: the solution structure is localized in the interior rather than at the boundary, its position is not encoded anywhere in the architecture and the equation is nonlinear. It therefore tests whether residual-driven refinement can locate and resolve sharp features that the trial space knows nothing about.

We consider the steady reaction-difusion equation

$$
- \varepsilon ^ { 2 } \Delta u + u ^ { 3 } - u = f \quad \mathrm { i n } \Omega = ( - 1 , 1 ) ^ { 2 } , \qquad \partial _ { n } u = 0 \mathrm { o n } \partial \Omega ,\tag{5.6}
$$

with $\varepsilon = 0 . 0 1$ , the homogeneous Neumann condition modeling a closed system with zero flux across the boundary. The source � is obtained by substituting into (5.6) the exact solution

$$
u = \sum _ { k = 1 } ^ { 4 } A _ { k } \exp \Big ( - { \frac { \| x - \bar { x } _ { k } \| ^ { 2 } } { 2 \sigma _ { _ { \mathrm { p } } } ^ { 2 } } } \Big ) , \qquad \sigma _ { \mathrm { p } } = 0 . 0 5 , \quad ( A _ { 1 } , A _ { 2 } , A _ { 3 } , A _ { 4 } ) = ( 1 , - 1 , 1 , - 1 ) ,\tag{5.7}
$$

with centers $\bar { \pmb { x } } _ { 1 } = ( - 0 . 6 , 0 . 4 ) , \bar { \pmb { x } } _ { 2 } = ( 0 . 5 , - 0 . 3 ) , \bar { \pmb { x } } _ { 3 } = ( 0 . 2 , 0 . 7 )$ and $\bar { \pmb { x } } _ { 4 } = ( - 0 . 4 , - 0 . 6 )$ , one per quadrant. Since every center lies at least 0.3 from $\partial \Omega .$ , the exact solution satisfies the homogeneous Neumann condition to within $1 0 ^ { - 6 }$ , which is negligible at the accuracy levels reported below. The problem is a steady state of the Allen-Cahn equation, each peak representing a localized phase domain embedded in an otherwise homogeneous medium; the solution is nearly zero over the bulk and rises $\mathrm { t o } \pm 1$ only within narrow neighborhoods of the centers, where it carries very steep gradients.

Three features make this problem hard for uniform collocation. The four $\sigma _ { \mathfrak { p } } .$ -radius disks cover under 1% of the domain, so a uniform collocation set places almost all of its points in the flat region where the residual is already small, while the few points near a peak are too sparse to resolve its gradient. Since the reaction term $u ^ { 3 } - u$ is bistable, the optimization landscape admits the spurious low-energy state $u \approx 0$ , toward which an under-resolved model is naturally drawn. Bistability under purely Neumann data also leaves the solution branch undetermined, so a small set of interior anchors (the center of each peak and four neighbors one peak width away, twenty points in all) is PINNs based to the values of the exact solution with weight $w _ { D } = 1 0 0$ to select the intended branch. The anchors localize the peaks yet do not resolve them: given the branch, an under-resolved peak still produces a large strong-form residual and RAR exploits that signal.

The square is partitioned into four rectangular elements, its quadrants, each carrying $K = 8$ Legendre bubble functions per direction. The quadrature order is raised to $Q = 8 0$ nodes per direction, giving eight to ten nodes across the $3 \sigma _ { \mathfrak { p } }$ support of each peak; at the order $Q = 2 4$ used for the singular benchmarks, the peaks would be invisible to the variational residual. Unlike in the preceding examples, the network input is not enriched: the trial space is a multiscale Fourier-feature network acting on (�, �) alone, with four tanh hidden layers of width 200, the last three with residual connections. The objective combines the variational residual, the Neumann penalty, the anchor term and the collocation term with weights $( w _ { v } , w _ { D } , w _ { N } , w _ { f } ) = ( 1 , 1 0 0 , 5 0 , 5 )$ , using 200 Neumann points per edge. The RAR arm begins from 10,000 uniformly drawn collocation points and, in each of ten rounds, inserts the 300 points with the largest strong-form residual from a candidate pool of 30,000 uniformly drawn points. It reaches $| \mathcal { T } _ { f } | = 1 3 \small { , } 0 0 0$ after 140,000 Adam iterations and eleven L-BFGS stages. Table 5 reports the errors.

The diagnostic result is the gap between the moderate $L ^ { 2 }$ error of the uniform arm and its very large $H ^ { 1 }$ error. That network locates all four peaks and recovers their approximate amplitudes (Fig. 4B); however, it does not recover their gradients, so its pointwise error concentrates in the peak cores (Fig. 4C, up to about 0.27) while the smooth background is accurate. Redirecting the same final number of collocation points with the residual indicator reduces the $\bar { L ^ { 2 } } , H ^ { 1 }$ and $L ^ { \infty }$ errors by factors of roughly 16, 14 and 38, respectively (table 5). The remaining error collapses to a low-amplitude pattern spread over the domain rather than a set of cores (Fig. 4I, up to about 0.007). Successive rounds cluster points at the four centers (Fig. 4F). That the largest gain appears in the gradient measure confirms that the refinement resolves the peak structure itself rather than merely lowering the global error and since both runs end at 13,000 points the diference is attributable to placement alone.

A  
![](images/33239c8f6a1c4941126f11377c38036c8a33799d0598fb5fedf164d040a328e8.jpg)

![](images/91fc524a2d79321c509ad87b97c6e7e34bac3b8396308d8d865f6e2da3642e9b.jpg)

C  
![](images/8ce11890bacd72f5849687e4e5f3497bb094de105b60e5875602539b173d95cd.jpg)

D  
![](images/2357dbcd0c8d978965f992704bdeb216cfa4287419a0cc4c01136e35d0ff7307.jpg)

E  
![](images/f55ba10a85a0b47aa7dbdd16b249d43bc46a161567335e5c6ec9a6e0bf4646de.jpg)

F  
![](images/7ec56b8ea967ac750f028510be25a82957ef371bb80fd40635b1a08859855f1d.jpg)

G  
![](images/85d1415d81c01af3f4c58d1425987ec3d1eb3d2e26f46755936941fe5da7375f.jpg)

H  
![](images/c387c189d19993ee7153da1c69a1d8e5fbf9b0b765da364d8c0d9005b1bf5f23.jpg)

I  
![](images/1acf4f65a8c5cc5cf223217331c3938a56ff5ea791f06ea334d13c7f3021a5b8.jpg)

J  
![](images/206d5a6b9c5f44cd2a614fe150bb6dbe89021e7c500877824a9f375ce6689b7e.jpg)

K  
![](images/55aa8e687e76145461e82d0bc753c6e4ae6f089ae6bc84f66c58fa37f63cef5a.jpg)  
Fig. 3. Cross problem: Laplace equation on the cross-shaped domain. (A–C) Exact solution, VC-PINNs (uniform) prediction and pointwise absolute error. (D-F) Domain with boundary conditions and element partition, uniform collocation set and adaptive collocation set after twelve rounds, $| \mathcal { T } _ { f } | = 8 0 0 0 \to 1 0 , 4 0 0$ . (G-I) Exact solution, VC-PINNs-RAR prediction and pointwise absolute error; panels C and I have separate color scales. (J-K) Adam loss components for the uniform and the RAR arm.

A  
![](images/23cdb67b6cafeb2bb9c87b843ef8605f64cba4bfdc5db9e943bf0810d4cc6d4c.jpg)

![](images/5a61d36663dab2a08927fb7d26ec82c48db9b61b48b33eb52c8dcfbcfe30c424.jpg)

C  
![](images/379bed423e364c508d862ec48ce07b9e1907a9793360b87a311355b6b2fc70ad.jpg)

D  
![](images/82edc156ca1c19639f04f4671db4d92058f09f59de249f3805a6647f9658f802.jpg)

E  
![](images/25b37b3732c6ba9a986aeea7421fa57dbf947a7b5830354633331f97dcaf3293.jpg)

F  
![](images/f1b72597e3f27ea7c290e0eb38131e0d25e53693545a64e202acc676fb50b9ae.jpg)

G  
![](images/1b73eaa7bb65256c32555f304c45d1421f429ee94bbeebf60de4dc814b5a7403.jpg)

H  
![](images/8434ef040a2e77eabfc7f78b6b8f0bff23b626d2538d0a09d720846481dcb5cd.jpg)

I  
![](images/5c03939648497eabe8e6e1450bc70da26cae391cdd2d9677d5e9c721de8f71a1.jpg)  
K

J  
![](images/fd277cd0203db8ebf4496ada7a82b9a916a717a987bd20035f76c75a407eb11b.jpg)

![](images/4806e5f186cba4a1ce6f99845cbe573715011a84fc3c39383877ff6b7a67a7be.jpg)  
Fig. 4. Reaction-difusion problem with four localized peaks. (A-C) Exact solution, VC-PINNs (uniform) prediction and pointwise absolute error. (D-F) Domain with the four peak centers and element partition, uniform collocation set and adaptive collocation set after ten rounds, $| \mathcal { T } _ { f } | = 1 0 , 0 0 0 \to 1 3 , 0 0 0$ . (G-I) Exact solution, VC-PINNs-RAR prediction and pointwise absolute error. (J-K) Adam loss components for the uniform and the RAR arm.

Reaction-difusion problem. All errors are reported in percent $( \% )$ and bold marks the more accurate arm in each row. The RAR arm grows its collocation set from 10,000 to 13,000 points over ten rounds of 300 insertions, with 140,000 Adam iterations and eleven L-BFGS stages. The uniform arm is trained on a fixed set of uniformly drawn collocation points without insertion rounds. One training run per arm. Errors are evaluated on a uniform $4 0 1 \times 4 0 1$ grid on $[ - 1 , 1 ] ^ { 2 }$ and the $H ^ { 1 }$ semi-norm is computed with central diferences on the same grid. As max $| u | = 1$ , the relative $L ^ { \infty }$ error equals the absolute one.

<table><tr><td>Metric</td><td>VC-PINNs (uniform)</td><td>VC-PINNs-RAR</td></tr><tr><td>Relative  $L ^ { 2 }$  error</td><td>16.00</td><td>1.015</td></tr><tr><td>Relative  $L ^ { \infty }$  error</td><td>26.50</td><td>0.6986</td></tr><tr><td>Relative  $H ^ { 1 }$  semi-norm error</td><td>64.46</td><td>4.738</td></tr></table>

## 5.4. Monotone semilinear reaction-difusion on an L-shaped domain

The preceding examples each carry one kind of singular feature. This one carries two of diferent dimension at the same time and admits no closed from solution, so it tests whether a single scalar indicator can distribute refinement between a curve and a point without any manufactured solution to guide it.

We solve the singularly perturbed semilinear problem

$$
- \varepsilon ^ { 2 } \Delta u + u ^ { 3 } = 1 \mathrm { i n } \Omega , \qquad u = 0 \mathrm { o n } \Gamma _ { D } , \qquad \partial _ { n } u = 0 \mathrm { o n } \Gamma _ { N } ,\tag{5.8}
$$

with $\varepsilon = 0 . 0 5$ on $\Omega = ( - 1 , 1 ) ^ { 2 } \setminus ( [ 0 , 1 ] \times [ 0 , 1 ] )$ , where $\Gamma _ { D } = \{ ( x , 0 ) : 0 \leq x \leq 1 \} \cup \{ ( 0 , y ) : 0 \leq y \leq 1 \}$ are the two re-entrant edges and $\Gamma _ { N }$ the outer boundary. The right hand side is the constant $f \equiv 1$ . Nothing is manufactured, the network is given only $f ,$ , the boundary data and the geometry.

Writing $A ( u ) = - \varepsilon ^ { 2 } \Delta u + u ^ { 3 }$ on $V = \{ v \in H ^ { 1 } ( \Omega ) \ : \ v | _ { \Gamma _ { n } } = 0 \}$ , the pointwise identity $( u ^ { 3 } - v ^ { 3 } ) ( u - v ) =$ $( u - v ) ^ { 2 } ( u ^ { 2 } + u v + v ^ { 2 } ) \geq 0$ gives $\langle A ( u ) - A ( v ) , u - v \rangle \geq \varepsilon ^ { 2 } | u - v | _ { H ^ { 1 } } ^ { 2 }$ , so � is strictly monotone and being coercive and hemi continuous it admits a unique weak solution by the Browder-Minty theorem. No branch has to be selected, so this example uses no interior anchors, the refinement locates the features from the residual alone.

The solution has two distinct sharp features. Away from $\Gamma _ { D } .$ , the difusion term is negligible and � approaches the root $u \equiv 1$ of $u ^ { 3 } = 1$ . Along $\Gamma _ { D } .$ , the homogeneous data forces a boundary layer of width $\begin{array} { r } { \frac { \overline { { \varepsilon } } } { \sqrt { 3 } } \approx 0 . 0 2 9 } \end{array}$ , a codimension one feature of total length 2. At the origin, the interior angle $\frac { 3 \pi } { 2 }$ produces the corner behavior $u \sim r ^ { 2 / 3 }$ , a point feature. A scalar indicator must allocate points between a curve and a point using one number per candidate. As in section 5.2, the leading corner mode is supplied by appending $r ^ { 2 / 3 }$ to the input, so the network acts on $( x , y , r ^ { 2 / 3 } )$ , the layer receives no analogous treatment and must be learned.

Errors are measured against a quadratic $( \mathbb { P } _ { 2 } )$ finite element reference computed on a mesh graded toward $\Gamma _ { D }$ and the corner. It is obtained by Newton’s method with continuation in � from 0.4 down to 0.05, extrapolated across refinemen levels and sampled on a $6 0 1 \times 6 0 1$ grid. The reference is used only post hoc and never enters the loss, grid points within $1 0 ^ { - 3 }$ of the corner are excluded from the $H ^ { 1 }$ semi-norm, where the reference gradient is itself unbounded.

A second check bypasses the reference entirely. Along a straight portion of $\Gamma _ { D }$ and away from the corner, the solution depends only on the wall distance $\zeta$ and $( 5 . 8 )$ reduces $\mathrm { t o } - \varepsilon ^ { 2 } u ^ { \prime \prime } + u ^ { 3 } = 1$ with $u ( 0 ) = 0 , u ( \infty ) = 1$ . Multiplying by $u ^ { \prime }$ and integrating once gives the first integra

$$
\begin{array} { r } { \frac { 1 } { 2 } \varepsilon ^ { 2 } ( u ^ { \prime } ) ^ { 2 } = \frac { 1 } { 4 } \big ( u ^ { 4 } - 4 u + 3 \big ) = \frac { 1 } { 4 } ( 1 - u ) ^ { 2 } \big ( u ^ { 2 } + 2 u + 3 \big ) , } \end{array}\tag{5.9}
$$

which gives the exact wall gradient $u ^ { \prime } ( 0 ) = \sqrt { 3 / 2 } / \varepsilon = 2 4 . 4 9 4 9 .$ … and the tail $1 - u \sim 1 . 4 5 5 8 e ^ { - \sqrt { 3 } \zeta / \varepsilon }$ . This profile solves the two dimensional equation exactly in the strip, so away from the corner both the reference and the network must reproduce it.

Both arms use a multiscale Fourier-feature network with six residual hidden layers of width 128 and Fourier scales $\{ \sigma _ { 1 } , \sigma _ { 2 } , \sigma _ { 3 } \} ~ = ~ \{ 1 , 1 0 , 3 0 \}$ , the L-shaped domain is tiled by three rectangular elements (unit squares), each carrying $K \ = \ 8$ Legendre bubble functions and $Q \ = \ 4 8$ Gauss-Legendre nodes per direction. The loss weights are $( w _ { v } , w _ { D } , w _ { N } , w _ { f } ) = ( 1 , 5 0 , 5 0 , 1 0 )$ , with 600 Dirichlet and 1000 Neumann points. Both arms begin from 4000 collocation points, insert 300 per round over twelve rounds to $| \mathcal { T } _ { f } | = 7 6 0 0$ and train for 150,000 Adam iterations and thirteen L-BFGS stages, drawing candidates from the same admissible region $\lbrace r \geq 2 \times 1 0 ^ { - 3 } \rbrace$ . They difer only in selection, the uniform arm draws its 300 points at random from that region, whereas the RAR arm scores a candidate pool of 60,000 points by $\rho = | - \varepsilon ^ { 2 } \Delta \hat { u } + \hat { u } ^ { 3 } - 1 |$ and retains the largest 300.

Semilinear problem. All errors are reported in percent (%) and bold marks the more accurate arm in each row. Both arms use the same collocation schedule $( | T _ { f } | = 4 0 0 0 \to 7 6 0 0 )$ , 150,000 Adam iterations and thirteen L-BFGS stages, with no interior anchors; one training run per arm. Errors are measured against the finite element reference on its $6 0 1 \times 6 0 1$ grid and the $H ^ { 1 }$ semi-norm excludes grid points within $1 0 ^ { - 3 }$ of the corner. As $0 \leq u \leq 1$ with max $u = 1$ , the relative $L ^ { \infty }$ error equals the absolute one.
<table><tr><td>Metric</td><td>VC-PINNs (uniform)</td><td>VC-PINNs-RAR</td></tr><tr><td>Relative  $L ^ { 2 }$  error</td><td>2.403</td><td>0.1904</td></tr><tr><td>Relative  $L ^ { \infty }$  error</td><td>55.85</td><td>3.546</td></tr><tr><td>Relative  $H ^ { 1 }$  semi-norm error</td><td>39.00</td><td>3.366</td></tr></table>

The spread between the three error measures of the uniform arm is diagnostic rather than incidental. As $u \approx 1$ over most of Ω and departs from 1 only in a strip of width $O ( \varepsilon )$ , a network that reproduces the constant state yet fails to resolve the layer (Fig. 5B and C) already attains a small $L ^ { 2 }$ error, that norm does not weight a set of measure $O ( \varepsilon )$ whereas the $\dot { H } ^ { 1 }$ semi-norm does and it reports a relative gradient error of 39.00%. Over twelve rounds the uniform arm reduces its relative $L ^ { 2 }$ error from 4.382% to 2.403% and its $H ^ { 1 }$ semi-norm error from 50.35% to 39.00% however, only 135 of its 3600 inserted points lie within 0.05 of $\Gamma _ { D }$ (Fig. 5E), close to the 3.4% share of the domain that this strip occupies, residual-based insertion places 909 points there.

The layer check based on (5.9) is more direct than any norm. At the stations $x = 0 . 3 , 0 . 5$ and 0.7 on $\{ ( x , 0 ) : 0 <$ $x < 1 \}$ , the uniform arm returns wall gradients of 12.62, 16.75 and 15.03 against the exact 24.4949, too low by 48.5%, 31.6% and 38.6% and its wall normal profile deviates from the exact one by up to 25.93%. The RAR arm returns 24.50, 24.51 and 24.47, within 0.12% of the exact value and its profile deviates by less than 0.42%. With the same point and iteration budget, it also reduces the relative $L ^ { 2 } , L ^ { \infty }$ and $H ^ { 1 }$ semi-norm errors by factors of roughly 13, 16 and 12 (table 6, Fig. 5C and I). Since the layer check does not use the finite element reference, it is evidence that the network has resolved the layer rather than matched a particular discretization.

The point distribution shows how one scalar handles two feature types. Early rounds are dominated by the layer, inserted points forming a dense band along both segments of $\Gamma _ { D }$ together with a tight cluster at the origin (Fig. 5F), later rounds spread into the bulk once the layer residual has fallen to the background level. This is the amplitude ordered behavior of section 5.2, here operating across features of diferent dimension. Inserted points afect training only since the objective retains a collocation term alongside the variational one, in a purely variational formulation, points ranked by |�( ̂�)| would have nothing to act on and refinement would require an element level estimator whose spatial resolution is coarser than the layer itself.

## 5.5. The classical Motz problem

Each preceding singular example draws its singularity from the geometry, a slit or a re-entrant corner. The Motz problem removes that source entirely. The domain is a convex rectangle and the solution is not in $H ^ { 2 } ( \Omega )$ , since the boundary condition changes type at an interior point of a straight edge. It is a standard benchmark for singularity resolving numerical methods [56, 57]. Here it tests whether RAR responds to the singular behavior of the solution rather than to geometric cues.

We consider $\Delta u = 0$ on $\Omega = ( - 1 , 1 ) \times ( 0 , 1 )$ with

$$
u = 0 ~ \mathrm { o n } ~ \{ ( x , 0 ) : - 1 \leq x \leq 0 \} , \quad \quad u = 5 0 0 ~ \mathrm { o n } ~ \{ ( 1 , y ) : 0 \leq y \leq 1 \} , \quad \quad \partial _ { n } u = 0 ~ \mathrm { o n } \Gamma _ { N } ,\tag{5.10}
$$

where $\Gamma _ { D }$ is the union of the two Dirichlet segments and $\Gamma _ { N }$ comprises the right half of the bottom edge $\{ ( x , 0 ) : 0 \leq$ $x \leq 1 \}$ , the top edge and the left edge. The boundary condition changes from Dirichlet to Neumann at the origin, which is an interior point of the bottom edge, the interior angle there is $\pi .$ Separation of variables about that point admits solutions $r ^ { \lambda } \cos ( \lambda \theta )$ with $\theta \in [ 0 , \pi ]$ satisfying $u = 0$ at $\theta = \pi$ and $\begin{array} { r } { \frac { \partial u } { \partial \theta } = 0 } \end{array}$ at $\theta = 0 .$ , which forces $\lambda \in \{ \frac { 1 } { 2 } , \frac { \bar { 3 } } { 2 } , \frac { 5 } { 2 } , \dots \}$ and gives the Williams expansion [58]

$$
u ( r , \theta ) = \sum _ { k \geq 0 } a _ { k } r ^ { k + 1 / 2 } \cos \bigl ( ( k + \textstyle \frac 1 2 ) \theta \bigr ) .\tag{5.11}
$$

A  
![](images/ff0c3284ed659e1f59fd95046cb7d67a67457f85a97c4ed54f5e131e2055d9b5.jpg)

B  
![](images/be8877b983d101149b590445cb08c2526ac3d90cbd9df7b3c8750cf73045aaa6.jpg)

C  
![](images/9b2fa325b328c74ae78b6836d71f7f3ef8b190adedcef319ca628b3bbac34bca.jpg)

D  
![](images/7853de3cf7daa447e2c0ee102e3b4b2065a2e8a6eb35c79a5856cf034e302296.jpg)

E  
![](images/94ddf9e026d60c12efb454d2898bd2369809624ad49577cb1c29d47fbcef73a8.jpg)

F  
![](images/fc1d6bcc09c6f8133dfa6c4b015603aa008a202de53f17dd076fd5b4ffbb06b7.jpg)

G  
![](images/0a66a9f3c5295a7535a7d5287e759522519ab0904d642569babc83895512d47c.jpg)

H  
![](images/a89dfd9464fed7c8f91857a14d5dfc5d7924aa9461a99cad5c2c91d83044ab72.jpg)

I  
![](images/2ffb2e838cf464c3f2bf8351c6ce75ed321a0c0d22c5698cba05e6cd2f0af344.jpg)

J  
![](images/5587bb54d74dc1e32dea4bbaa33fcc8bc20a7d719a077dd28e42e54b86ac4c7b.jpg)

K  
![](images/537011878856ad159c31f690395ae53baeea5d4cd2dc417709cf4c0138396b10.jpg)  
Fig. 5. Semilinear problem on the L-shaped domain. (A-C) Finite element reference, VC-PINNs (uniform) prediction and pointwise absolute error against the reference. (D-F) Domain with boundary conditions and element partition, uniform collocation set and adaptive collocation set after twelve rounds, $| \mathcal { T } _ { f } | = 4 0 0 0  7 6 0 0 .$ (G-I) Finite element reference, VC-PINNs-RAR prediction and pointwise absolute error. (J-K) Adam loss components for the uniform and the RAR arm.

Motz problem. The first three rows are relative errors in percent $( \% )$ , the last row is the leading coeficient $\boldsymbol { a } _ { 0 } .$ . Bold marks the more accurate arm in each row. Both arms use the same collocation schedule $( \vert \mathcal { T } _ { f } \vert = 1 5 0 0  3 9 0 0 )$ , 150,000 Adam iterations and thirteen L-BFGS stages, one training run per arm. Errors are measured against the quadratic $\left( \mathbb { P } _ { 2 } \right)$ finite element reference, whose own error floor is 0.000079%. ${ \sf A s } \mathrm { ~ 0 ~ } \leq u \leq 5 0 0$ with max $u = 5 0 0$ , the relative $L ^ { \infty }$ error is the absolute error divided by 500. The leading coeficient is extracted by (5.12) at $r _ { 0 } = 0 . 5 ,$ , its reference value is $a _ { 0 } = 4 0 1 . 1 6 1 9 1$
<table><tr><td>Metric</td><td>VC-PINNs (uniform)</td><td>VC-PINNs-RAR</td></tr><tr><td>Relative  $L ^ { 2 }$  error</td><td>0.2407</td><td>0.2096</td></tr><tr><td>Relative  $L ^ { \infty }$  error</td><td>0.7872</td><td>0.4458</td></tr><tr><td>Relative  $H ^ { 1 }$  semi-norm error</td><td>1.666</td><td>0.9920</td></tr><tr><td>Leading coefficient  $a _ { 0 }$ </td><td>399.92050</td><td>400.15007</td></tr></table>

The leading term has $\begin{array} { r } { \lambda _ { 1 } = \frac { 1 } { 2 } . } \end{array}$ , so $u \in H ^ { s } ( \Omega )$ for every $s < 3 / 2$ however, $u \not \in H ^ { 2 } ( \Omega )$ and $\left| \nabla u \right| \sim r ^ { - 1 / 2 }$ as $r  0$ This is the same rate as in the crack problem (section 5.1), produced here by a change of boundary condition type on a straight edge rather than by a slit. The problem originates in electrostatics, � being the potential in a rectangular region held at fixed potentials on part of the boundary and insulated on the rest.

The dificulty is that nothing about Ω marks the origin as special. A refinement criterion driven by geometry has no signal to act on and a uniform collocation set treats the neighborhood of the origin exactly as it treats the bulk, whereas the error concentrates there. Since $f \equiv 0$ , the strong-form residual measures purely the failure of the network to be harmonic and is therefore available as an indicator that depends only on the solution.

The lines $x = 0$ and $y = 0 . 5$ partition the rectangle into four rectangular elements, so that the singular point lies on an element interface. The network input is enriched with the leading mode alone, so that ̂� acts on $( x , y , r ^ { 1 / 2 } \cos ( \theta / 2 ) )$ , since $\theta \in [ 0 , \pi ]$ on this domain, the map is single valued and needs no branch cut. Only one term of $( 5 . 1 1 )$ is supplied and the rest of the series must be learned, this distinguishes the example from the crack problem of section 5.1, where the enrichment reproduces the exact solution (remark 3.1). As stated in section 3.1, the network represents $u / 5 0 0$ $( c _ { u } = 5 0 0 )$ , which is $O ( 1 )$ and the scaling is undone before any error is reported. The two arms are budget matched in collocation count, Adam iterations and L-BFGS stages, each run recording its budget so that a drift between them is detected rather than assumed absent, they difer only in whether the points inserted each round are drawn uniformly or taken as the largest strong-form residuals from a candidate pool.

With no closed from solution available, errors are measured against a quadratic $( \mathbb { P } _ { 2 } )$ finite element reference that plays the role of the exact solution. Accuracy at the singularity itself is assessed independently of any pointwise norm by extracting the first four coeficients of (5.11). Using the orthogonality $\begin{array} { r } { \int _ { 0 } ^ { \pi } \cos ( ( k + \frac { 1 } { 2 } ) \theta ) \cos ( ( j + \frac { 1 } { 2 } ) \theta ) d \theta = \frac { \pi } { 2 } \delta _ { k j } } \end{array}$

$$
a _ { k } = \frac { 2 } { \pi r _ { 0 } ^ { k + 1 / 2 } } \int _ { 0 } ^ { \pi } u ( r _ { 0 } , \theta ) \cos \bigl ( ( k + { \textstyle { \frac { 1 } { 2 } } } ) \theta \bigr ) d \theta ,\tag{5.12}
$$

evaluated at $r _ { 0 } = 0 . 5$ by 400 node Gauss-Legendre quadrature. The same projection is applied to the finite element reference, so network coeficients, reference coeficients and the published collocation Treftz values [56] are directly comparable.

## 6. Concluding remarks

This work studied residual-based adaptive refinement within a hybrid variational-collocation PINNs for secondorder elliptic boundary value problems with Dirichlet and Neumann data. The study covers geometric and boundary induced singularities, localized interior features and a boundary layer. The approach rests on a structural observation. A Petrov-Galerkin weak form yields one functional per element and no pointwise residual, so point level adaptivity cannot be driven by the variational term alone. The collocation term, built on the strong-form residual, supplies the missing pointwise quantity. Refinement therefore acts on the collocation set only and the element partition, the test space and the quadrature rule are left unchanged.

The scheme was assessed on five benchmarks, each chosen to isolate a distinct dificulty. In every comparison the adaptive and uniform runs share the final collocation count, the number of Adam iterations and the L-BFGS schedule. The two arms therefore difer only in how the collocation set is placed and grown. The comparison is matched in budget.

A  
![](images/766a003e721901410ce5565d10694597a27a06542222878a506f5c9386ccd5d6.jpg)  
D

B  
![](images/d994d5ca893224a1865bec8e80bd91be2ed0985d65e4654a5c8debd192a872df.jpg)  
E

C  
![](images/9905490541b6b72ae4093bec9d967eb4e600d793ead008a5c7a337efb8c526c5.jpg)

![](images/568323d9ab0ebfd1ccb04129702be70f0f3ad8d7d7f5b6a47272842f1f77d2ec.jpg)

F  
![](images/1c28d86bb5c25afe2069b27e38b8c9444f44bf84e201c39036104a4ffc55d2b6.jpg)

![](images/e0fd0172b0668430b6132dd9126e78e35b854baac14d84e6ca3d525224e466f4.jpg)

G  
![](images/34e4112b935e810ed97a2a84f76f6bda39dae1b01496ce44b770b6c1c1660948.jpg)

H  
![](images/d60c849890b3cace074be6c678f296df867b1baef55418602caf7378e6a9151d.jpg)

I  
![](images/b39c633e7426c7871b661d64e86412ac5550a2e1a38b8634c38c3a0843880ed4.jpg)

J  
K  
![](images/7af03f140d01cd16342f2c359c563938ccc98a8e67b73d32cf1fbc5a5bd7ca42.jpg)

![](images/c4e92362c5095b96b4c5295dbb41e9983de8c65dcb891ac475ad38b4d9382afe.jpg)  
Fig. 6. The Motz problem. (A-C) Finite element reference solution, VC-PINNs (uniform) prediction and pointwise absolute error against the reference. (D-F) Domain with the mixed boundary conditions and element partition, uniform collocation set and adaptive collocation set after twelve rounds, $| \mathcal { T } _ { f } | = 1 5 0 0  3 9 0 0$ . (G-I) Finite element reference solution, VC-PINNs-RAR prediction and pointwise absolute error. (J-K) Adam loss components for the uniform and the RAR arm.

The relative $H ^ { 1 }$ semi-norm error decreased on all five problems, by factors between 1.7 and 16, which is consistent with refinement concentrating where the gradient is steep or unbounded. The $L ^ { 2 }$ error was less uniform. It ranged from a 32 fold reduction on the cross-shaped domain to a slight increase on the Motz problem. The individual benchmarks lead to the following conclusions.

Mode III crack. The slit makes the domain non-Lipschitz and the solution lies only in $H ^ { 3 / 2 - \epsilon }$ near the $\mathrm { t i p . }$ Refinement graded the collocation set toward the crack tip and reduced both the $L ^ { 2 }$ and the $H ^ { 1 }$ errors. This problem motivated three selection devices, radial normalization of the indicator, density based sampling and a minimum separation filter. Together they prevent successive rounds from collapsing onto a small cluster of candidates. The RAR arm reduced the relative $H ^ { 1 }$ seminorm error by 1.65 times compared with uniform arm.

Cross-shaped domain. Four re-entrant corners generate $r ^ { 2 / 3 }$ singularities of unequal amplitude, which compete for the same collocation budget. The indicator ranked the corners by amplitude from a single scalar per candidate point, with no knowledge of the singular coeficients. This benchmark gave the largest $L ^ { 2 }$ reduction, a factor of 32.

Reaction-difusion with localized peaks. Although the solution is smooth on a convex square, it has four sharp interior peaks with $O ( 1 / \sigma )$ gradients. The problem therefore tests whether the indicator can resolve interior features of finite yet large gradient, rather than geometric singularities. RAR arm reduces the relative $L ^ { \infty }$ error by a factor of 38 times compared with uniform arm.

Semilinear singularly perturbed problem. For $- \varepsilon ^ { 2 } \Delta u + u ^ { 3 } = 1$ with $\varepsilon = 0 . 0 5$ on the L-shaped domain, a boundary layer along $\Gamma _ { D }$ coexists with the $r ^ { 2 / 3 }$ singularity at the re-entrant corner. Accuracy was measured against a graded-mesh $P _ { 2 }$ finite element reference. The indicator divided refinement between a one-dimensional layer and a point singularity, again from one scalar per candidate. RAR arm reduces the relative $L ^ { \infty }$ error by a factor of 16 times compared with uniform arm.

Motzproblem. Here the singularity arises from the change from Dirichlet to Neumann data along a straight edge, not from the geometry. Under the bare residual ranking, refinement improved every measure weighted toward the singular point, the $H ^ { 1 }$ semi-norm included. The domain-averaged $L ^ { 2 }$ error, however, increased slightly. The mechanism is identifiable. Each round placed its new points in a single neighborhood instead of grading them toward the singularity, so resolution near the singular point was gained at the expense of the rest of the domain. RAR arm reduces the relative $L ^ { \infty }$ error by a factor of 2 times compared with uniform arm.

In both the crack and the Motz problems the indicator located the singularity correctly. What difered was how ranked candidates were turned into new collocation points. The evidence points to the selection rule, at least as much as the indicator, as the component that makes residual driven refinement reliable. The radial normalization, density sampling and minimum separation filter were designed to prevent exactly the collapse observed on the Motz problem. That they do so there remains to be confirmed.

Three limitations remain. First, the strong-form residual requires second derivatives of the network. Second, Neumann data are imposed by penalty. Third, only the collocation set is adapted and no ℎ- or �-refinement of the test space is performed. Finally, all results are outcome from a single random seed run.

## CRediT authorship contribution statement

Somnath Giri: Conceptualization, Methodology, Software, Investigation, Writing - original draft. Muslim Malik: Supervision, Methodology, Writing - review & editing. Rishikesh Yadav: Supervision, Methodology, Validation, Writing - review & editing.

## Disclosure statement

No potential conflict of interest was reported by the author(s).

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Data availability

No data was used for the research described in the article.

## References

[1] Ciarlet, P. G. (2002). The finite element method for elliptic problems. Society for Industrial and Applied Mathematics.

[2] Brenner, S. C., & Scott, L. R. (2008). The mathematical theory of finite element methods. New York, NY: Springer New York.

[3] LeVeque, R. J. (2007). Finite diference methods for ordinary and partial diferential equations: steady-state and time-dependent problems. Society for Industrial and Applied Mathematics.

[4] Grisvard, P. (2011). Elliptic problems in nonsmooth domains. Society for Industrial and Applied Mathematics.

[5] Motz, H. (1947). The treatment of singularities of partial diferential equations by relaxation methods. Quarterly of Applied Mathematics, 4(4), 371 − 377.

[6] Kondrat’ev, V. A. (1967). Boundary value problems for elliptic equations in domains with conical or angular points. Trudy Moskovskogo Matematicheskogo Obshchestva, 16, 209 − 292.

[7] Verfürth, R. (2013). A posteriori error estimation techniques for finite element methods. OUP Oxford.

[8] Binev, P., Dahmen, W., & DeVore, R. (2004). Adaptive finite element methods with convergence rates. Numerische Mathematik, 97(2), 219 − 268.

[9] Cascon, J. M., Kreuzer, C., Nochetto, R. H., & Siebert, K. G. (2008). Quasi-optimal convergence rate for an adaptive finite element method. SIAM Journal on Numerical Analysis, 46(5), 2524 − 2550.

[10] Szabó, B., & Babuška, I. (2021). Finite Element Analysis: Method. Verification and Validation, 2023(23), 980.

[11] Mitchell, W. F., & McClain, M. A. (2011). A survey of hp-adaptive strategies for elliptic partial diferential equations. In Recent advances in computational and applied mathematics (pp. 227 − 258). Dordrecht: Springer Netherlands.

[12] Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations. Journal of Computational physics, 378, 686 − 707.

[13] Lagaris, I. E., Likas, A., & Fotiadis, D. I. (1998). Artificial neural networks for solving ordinary and partial diferential equations. IEEE transactions on neural networks, 9(5), 987 − 1000.

[14] Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018). Automatic diferentiation in machine learning: a survey. Journal of machine learning research, 18(153), 1 − 43.

[15] Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021). Physics-informed machine learning. Nature Reviews Physics, 3(6), 422 − 440.

[16] Noor, M., & Malik, M. (2025). Deep neural network approach to solve coupled diferential equations with multiple delays. Physica Scripta, 100(7), 075259.

[17] Rahaman, N., Baratin, A., Arpit, D., Draxler, F., Lin, M., Hamprecht, F., ... & Courville, A. (2019, ���). On the spectral bias of neural networks. In International conference on machine learning (��.5301 − 5310).����.

[18] Wang, S., Wang, H., & Perdikaris, P. (2021). On the eigenvector bias of Fourier feature networks: From regression to solving multi-scale PDEs with physics-informed neural networks. Computer Methods in Applied Mechanics and Engineering, 384, 113938.

[19] Krishnapriyan, A., Gholami, A., Zhe, S., Kirby, R., & Mahoney, M. (2021). Characterizing possible failure modes in physics-informed neural networks. Advances in neural information processing systems, 34, 26548 − 26560.

[20] Wang, S., Teng, Y., & Perdikaris, P. (2021). Understanding and mitigating gradient flow pathologies in physics-informed neural networks. SIAM Journal on Scientific Computing, 43(5), �3055 − �3081.

[21] McFall, K. S., & Mahan, J. R. (2009). Artificial neural network method for solution of boundary value problems with exact satisfaction of arbitrary boundary conditions. IEEE transactions on neural networks, 20(8), 1221 − 1233.

[22] Sukumar, N., & Srivastava, A. (2022). Exact imposition of boundary conditions with distance functions in physics-informed deep neura networks. Computer Methods in Applied Mechanics and Engineering, 389, 114333.

[23] Beidokhti, R. S., & Malek, A. (2009). Solving initial-boundary value problems for systems of partial diferential equations using neural networks and optimization techniques. Journal of the franklin institute, 346(9), 898 − 913.

[24] Hu, T., Jin, B., & Zhou, Z. (2024). Solving Poisson problems in polygonal domains with singularity enriched physics informed neural networks. SIAM Journal on Scientific Computing, 46(4), �369 − �398.

[25] Kharazmi, E., Zhang, Z., & Karniadakis, G. E. (2019). Variational physics-informed neural networks for solving partial diferential equations. arXiv preprint arXiv:1912.00873.

[26] Kharazmi, E., Zhang, Z., & Karniadakis, G. E. (2021). hp-VPINNs: Variational physics-informed neural networks with domain decomposition. Computer Methods in Applied Mechanics and Engineering, 374, 113547.

[27] Grifiths, D. F., & Lorenz, J. (1978). An analysis of the petrov—galerkin finite element method. Computer Methods in Applied Mechanics and Engineering, 14(1), 39 − 64.

[28] Berrone, S., Canuto, C., & Pintore, M. (2022). Variational physics informed neural networks: the role of quadratures and test functions. Journa of Scientific Computing, 92(3), 100.

[29] Berrone, S., Canuto, C., & Pintore, M. (2022). Solving PDEs by variational physics-informed neural networks: an a posteriori error analysis: S. Berrone et al. ANNALI DELL’UNIVERSITA’DI FERRARA, 68(2), 575 − 595.

[30] Rojas, S., Maczuga, P., Muñoz-Matute, J., Pardo, D., & Paszyński, M. (2024). Robust variational physics-informed neural networks. Computer Methods in Applied Mechanics and Engineering, 425, 116904.

[31] Taylor, J. M., Pardo, D., & Muga, I. (2023). A deep Fourier residual method for solving PDEs using neural networks. Computer Methods in Applied Mechanics and Engineering, 405, 115850.

[32] Taylor, J. M., Bastidas, M., Calo, V. M., & Pardo, D. (2024). Adaptive Deep Fourier Residual method via overlapping domain decomposition. Computer Methods in Applied Mechanics and Engineering, 427, 116997.

[33] Berrone, S., & Pintore, M. (2024). Meshfree Variational-Physics-Informed neural networks (MF-VPINN): An adaptive training strategy. Algorithms, 17(9), 415.

[34] Udomworarat, T., Brevis, I., van der Zee, K. G., & Rojas, S. (2026). Neural network approximation in discrete dual norms with adaptive test spaces. arXiv preprint arXiv:2606.20972.

[35] Badia, S., Li, W., & Martín, A. F. (2025). Adaptive finite element interpolated neural networks. Computer Methods in Applied Mechanics and Engineering, 437, 117806.

[36] Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E. (2021). DeepXDE: A deep learning library for solving diferential equations. SIAM review, 63(1), 208 − 228.

[37] Wu, C., Zhu, M., Tan, Q., Kartha, Y., & Lu, L. (2023). A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks. Computer Methods in Applied Mechanics and Engineering, 403, 115671.

[38] Mao, Z., & Meng, X. (2023). Physics-informed neural networks with residual/gradient-based adaptive sampling methods for solving partia diferential equations with sharp solutions. Applied Mathematics and Mechanics, 44(7), 1069 − 1084.

[39] Tang, K., Wan, X., & Yang, C. (2023). DAS-PINNs: A deep adaptive sampling method for solving high-dimensional partial diferentia equations. Journal of Computational Physics, 476, 111868.

[40] Gao, Z., Yan, L., & Zhou, T. (2023). Failure-informed adaptive sampling for PINNs. SIAM Journal on Scientific Computing, 45(4), �1971 − �1994.

[41] Dodge, S., Formisano, A., & Barmada, S. (2026). INI-VPINN: A variational physics-informed neural network with implicit Neumann and interface handling for multi-material domains with geometric singularities. Journal of Computational Physics, 565, 115328.

[42] Łoś, M., Służalec, T., Maczuga, P., Vilkha, A., Uriarte, C., & Paszyński, M. (2025). Collocation-based robust variational physics-informed neural networks (CRVPINNs). Computers & Structures, 316, 107839.

[43] Zeidler, E. (2013). Nonlinear functional analysis and its applications: II/B: nonlinear monotone operators. Springer Science & Business Media

[44] Tancik, M., Srinivasan, P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., ... & Ng, R. (2020). Fourier features let networks learn high frequency functions in low dimensional domains. Advances in neural information processing systems, 33, 7537 − 7547.

[45] Fix, G. J., Gulati, S., & Wakof, G. I. (1973). On the use of singular functions with finite element approximations. Journal of Computational Physics, 13(2), 209 − 228.

[46] Moës, N., Dolbow, J., & Belytschko, T. (1999). A finite element method for crack growth without remeshing. Internationaljournal for numerical methods in engineering, 46(1), 131 − 150.

[47] Gu, Y., Zhang, C., Zhang, P., Golub, M. V., & Yu, B. (2023). Enriched physics-informed neural networks for 2� in-plane crack analysis: Theory and MATLAB code. International Journal of Solids and Structures, 276, 112321.

[48] Zhu, B., Li, H., & Zhang, Q. (2025). Extended physics-informed extreme learning machine for linear elastic fracture mechanics. Computer Methods in Applied Mechanics and Engineering, 435, 117655.

[49] Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.

[50] Liu, D. C., & Nocedal, J. (1989). On the limited memory BFGS method for large scale optimization. Mathematical programming, 45(1), 503 − 528.

[51] Nocedal, J., & Wright, S. (2006). Numerical Optimization.

[52] Mitchell, W. F. (2013). A collection of 2� elliptic problems for testing adaptive grid refinement algorithms. Applied mathematics and computation, 220, 350 − 364.

[53] Allen, S. M., & Cahn, J. W. (1979). A microscopic theory for antiphase boundary motion and its application to antiphase domain coarsening. Acta metallurgica, 27(6), 1085 − 1095.

[54] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32.

[55] Anderson, T. L. (2017). Fracture mechanics: Fundamentals and applications (4th ed.). Boca Raton, FL: CRC Press.

[56] Lu, T. T., Hu, H. Y., & Li, Z. C. (2004). Highly accurate solutions of Motz’s and the cracked beam problems. Engineering Analysis with Boundary Elements, 28(11), 1387 − 1403.

[57] Li, Z. C., Chan, Y. L., Georgiou, G. C., & Xenophontos, C. (2006). Special boundary approximation methods for Laplace equation problems with boundary singularities—applications to the Motz problem. Computers & Mathematics with Applications, 51(1), 115-142.

[58] Williams, M. L. (1952). Stress singularities resulting from various boundary conditions in angular corners of plates in extension.