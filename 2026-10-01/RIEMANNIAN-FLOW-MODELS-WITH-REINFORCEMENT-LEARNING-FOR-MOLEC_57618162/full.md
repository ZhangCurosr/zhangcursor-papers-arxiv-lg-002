# RIEMANNIAN FLOW MODELS WITH REINFORCEMENT LEARNING FOR MOLECULAR CRYSTAL STRUCTURE PREDICTION

Thomas Egg<sup>1,2\*</sup>, Harry Winston Sullivan<sup>3\*</sup>, Maya M. Martirossyan<sup>1,2</sup>, Philipp Höllmer<sup>1,2</sup>, Cheng Zeng<sup>4,5</sup>, Adrian Roitberg<sup>4,5</sup>, Mingjie Liu<sup>4,5</sup>, Richard Hennig<sup>5,6</sup>, Sapna Sarupria<sup>7</sup>, Ellad B. Tadmor<sup>8</sup>, and Stefano Martiniani<sup>1,2,9,10</sup>

<sup>1</sup>Centerfor Soft Matter Research, Department ofPhysics, New York University, New York 10003, USA <sup>2</sup>Simons Center for Computational Physical Chemistry, Department of Chemistry, New York University, New York 10003, USA <sup>3</sup>Department ofChemical Engineering and Materials Science, University ofMinnesota, Minneapolis, MN 55455, USA <sup>4</sup>Department ofChemistry, University ofFlorida, Gainesville, FL 32611, USA <sup>5</sup>Quantum Theory Project, University of Florida, Gainesville, FL 32611, USA <sup>6</sup>Department ofMaterials Science & Engineering, University ofFlorida, Gainesville, FL 32611, USA <sup>7</sup>Department ofChemistry, University ofMinnesota, Minneapolis, MN 55455, USA <sup>8</sup>Department of Aerospace Engineering and Mechanics, University of Minnesota, Minneapolis, MN 55455, USA <sup>9</sup>Courant Institute ofMathematical Sciences, New York University, New York 10003, USA <sup>10</sup>Centerfor Neural Science, New York University, New York 10003, USA

May 6, 2026

## ABSTRACT

Crystal structure governs material properties, making crystal structure prediction (CSP) a fundamental problem in materials science. Generative models are a promising approach for solving this problem, but the prevalence of polymorphism, coupled with large unit cells and complex packing geometry, makes the molecular CSP task challenging for existing models. To address this, we introduce Coarse-Grained Open Materials Generation (CG-OMatG), an equivariant Riemannian flow-based generative model. CG-OMatG predicts molecular crystal structures via a coarse-grained, hierarchical representation. CG-OMatG treats molecules as rigid bodies—performing both inter- and intramolecular message passing to construct a geometric representation for molecular packings—and learns to reconstruct molecule centroid positions, orientations, and lattice parameters, conditioned on chemical species and conformer geometry. We train the model on subsets of the Open Molecular Crystals (OMC25) and Cambridge Structural Database (CSD) datasets. Further, we fine-tune the model via policy gradient reinforcement learning to steer the model towards generating low-energy candidate structures. We validate the generated structures on the CSP blind test benchmark, assessing agreement with experimentally determined crystals using COMPACK packing-similarity analysis. CG-OMatG exhibits strong performance for generative molecular crystal structure prediction, paving the way for accelerated polymorph screening and organic solid-state materials discovery.

## 1 Introduction

An outstanding challenge in materials science is that of molecular (organic) crystal structure prediction (CSP), where one seeks to determine the energetically favorable structures into which organic molecules crystallize [1]. Unlike in atomic crystals, the weak non-bonded interactions holding molecules together in a crystal packing give rise to a proliferation of many stable energy minima, whose corresponding crystal structures can lie very close in absolute free energy [2] while remaining separated by large kinetic barriers. Molecular crystals, therefore, display a propensity for polymorphism, whereby multiple metastable crystal structures can co-exist for a given molecule, making CSP significantly more challenging than for inorganic materials. Different crystal polymorphs of the same compound can exhibit markedly different physical properties, only some of which may be suitable for a given application [3–

![](images/4496daf6513601ed0b9ef0fb75abede6dd7fcea23c021970323d6e0e25fce9f3.jpg)  
Figure 1: Comparison of generated and ground-truth crystal packings for three OMC25-MCF targets. COMPACK alignments are shown with RDKit molecular diagrams and computed RMSD<sub>NMatches</sub>. CSD refcodes, left to right: NIYDOO, PEMWOT, JURRET. No relaxation was performed.

6]. Correctly identifying the ground-state structure is thus critical: phase transformations away from a metastable polymorph carry nonzero probability and can compromise performance or, in pharmaceutical applications, alter drug bioavailability with direct consequences for patient safety [7]. Beyond pharmaceuticals, molecular crystals find broad application across agrochemistry, food science, and electronics, making reliable polymorph prediction a problem of broad practical importance.

Given the cost and difficulty of experimental crystal structure determination, substantial effort has gone into the development of computational tools for predicting these phases in silico [8, 9]. Ranking crystal structures by their energetic stability—either using quantum-mechanical calculations via density functional theory (DFT) [10] or approximating these calculations with machine learning interatomic potentials (MLIPs) [11, 12]—can provide insight into the available crystal structures and their relative stabilities. These methods typically involve many rounds of iterative optimization and expensive calculations, limiting their ability to exhaustively explore all low-energy phases.

Recent advances in MLIPs and generative models offer a more scalable path, enabling faster exploration of the energy landscape and, through guidance [13–16] and reinforcement learning [17, 18], the ability to steer generation toward desired properties. Applying these tools to molecular CSP, however, introduces distinct challenges. All-atom generative models lack explicit knowledge of the different length scales and bond types present in organic molecules, and there is no constraint enforcing that intramolecular bonds remain coherent during or at the end of generation. Unit cell sizes compound the problem: organic crystals are typically far larger than the inorganic structures for which most generative models have been designed and trained [19]. Existing approaches must therefore be substantially tailored for molecular crystal structure prediction.

We present a coarse-grained generative model for molecular crystal structure prediction, following a line of work that treats molecules as rigid-body building blocks [20–22]. By separating the strong intramolecular interactions that define molecular shape from the weak intermolecular interactions that drive the crystal packing, this approach directly addresses the multi-scale character of molecular crystals, focusing generation on the placement and orientation of rigid molecules within the unit cell. Galanakis and Tuckerman [23] have shown that molecular centers of mass tend to occupy well-defined packing positions, suggesting that crystal packings can be learned effectively at this coarser scale. Our model learns local-coordinate representations and constrains the generative process to building block positions and orientations, encoding this multi-scale inductive bias by design.

## Our contributions

• We introduce the Coarse-Grained Open Materials Generator (CG-OMatG), an equivariant generative model for molecular CSP.

• We mathematically formulate the Riemannian manifold of molecular crystal configurations and construct an affineinvariant geodesic unit-cell interpolation using polar decomposition on $\bar { S } O ( 3 ) \times \mathrm { { S y m _ { 3 } ^ { + } } }$ , ensuring every point along the path corresponds to a nondegenerate, positive-volume unit cell—in contrast to prior parameterizations on $\mathbb { R } ^ { 3 \times \breve { 3 } }$

• We formulate a group-relative policy optimization (GRPO) scheme on the manifold of molecular crystal configurations and apply it to reward energetically stable packings.

## 2 Related Work

## 2.1 Molecular CSP

Predicting energetically favorable molecular crystal packings is a long-standing challenge with far-reaching implications for materials design, pharmaceutical development, and discovery of functional molecular solids [1]. Due to the considerable cost, effort, and time required to identify crystal structures experimentally, there is significant interest in computational approaches that can accelerate the CSP pipeline. Conventional methods rely on iterative rounds of expensive quantum chemical calculations [24, 25]. Recent work seeks to bypass expensive energy function evaluations entirely: Galanakis and Tuckerman [23] introduced CrystalMath, an optimization procedure that optimizes the crystal structure with respect to simple order parameters and reports rapid structure prediction for systems with one or two molecules $Z ^ { \prime } \in \mathsf { \bar { \{ 1 , 2 \} } }$ in the asymmetric unit. Other tools like Genarris and FastCSP couple random structure generation with physical constraints or energy evaluations to predict stable configurations of close-packed molecular crystals [26, 27, 11].

## 2.2 Generative Models

Developments in machine learning have spurred the advent of generative models which accelerate crystal structure prediction by learning to sample from the distribution of known crystal structures, obtained either through experimental determination or first-principles calculations. A wealth of models have been devised to predict inorganic condensed phases conditioned on a target chemistry [28–34], and adaptations for larger and more complex systems have followed, including models for metal-organic frameworks [21, 35, 36]

Generative models for molecular CSP are more recent. AssembleFlow uses inertial frames to decompose molecular SE(3) transformations into separate translation and rotation flows for finite molecular clusters [20]. OXtal, an AlphaFold3-style diffusion model, learns crystalline packings of molecular conformers in Cartesian space, foregoing the learning of the unit-cell lattice [37]. Both of these approaches, however, require post-hoc Patterson analysis [38] to recover the unit cell, limiting their utility for property-based guidance based on energy calculations.

Closest to our work, MolCrystalFlow [22] and PackFlow [39] also apply flow-based generative modeling to molecular CSP. MolCrystalFlow shares the rigid-body decomposition and Riemannian treatment of molecular degrees of freedom with CG-OMatG, but parameterizes the lattice as an unconstrained matrix in R<sup>3×3</sup> and does not include RL post-training. PackFlow generates per-atom Cartesian coordinates for all heavy atoms in the unit cell without exploiting the rigid-body structure of molecular crystals, jointly sampling these with lattice parameters in Euclidean space. Related lattice decompositions are used by MatterGen [31] and DiffCSP++ [40], the latter preserving positive definiteness through diffusion in a symmetric logarithmic representation. CG-OMatG instead combines affine-invariant geodesic unit-cell interpolation with coarse-grained building blocks and GRPO on $S O ( 3 ) \times \mathrm { S y m _ { 3 } ^ { + } }$

## 2.3 Reinforcement Learning

Post-training via reinforcement learning (RL) provides a way to align generative models with downstream objectives by optimizing neural network weights against a reward function. RL post-training has begun to show success in generative modeling for inorganic materials [41–46]. Höllmer and Martiniani [18] demonstrated the potential of RL to steer pretrained inorganic CSP models toward low-energy structures in OMatG-IRL, and Subramanian et al. [39] pursued an analogous objective in the molecular setting in PackFlow. Both works address the central challenge of applying policy-gradient RL to ODE-based generative models, but through distinct constructions. OMatG-IRL introduces stochasticity into the ODE dynamics, yielding a surrogate SDE with tractable step-wise transition likelihoods that provide exact importance ratios and KL terms for policy-gradient updates. PackFlow retains deterministic ODE sampling and instead approximates per-sample policy scores using the flow-matching pretraining loss evaluated at a single time point, yielding surrogate importance ratios and KL regularization terms that do not correspond to exact likelihoods of the generative process. The RL setup in CG-OMatG builds on the construction in OMatG-IRL, extending it from Euclidean space to the Riemannian manifold of molecular crystal configurations, where stochastic exploration is introduced in the tangent space of this manifold, enabling policy-gradient RL with exact transition probabilities while preserving the geometry of the generative dynamics.

## 3 Methods

## 3.1 Molecular Crystal Structure Prediction

Crystal Structure Prediction The CSP task can be framed as a sampling problem targeting a conditional distribution

$$
p \left( L , \{ c ^ { ( j ) } \} _ { j = 1 } ^ { N } \mid { \mathcal { A } } \right) = p ( y \mid { \mathcal { A } } ) ,\tag{1}
$$

where $L \in G L ^ { + } ( 3 , \mathbb { R } )$ is a row-major matrix of lattice vectors, $\boldsymbol { c } ^ { ( j ) } \in \mathbb { R } ^ { 3 }$ is the Cartesian position of atom $j ,$ and $\mathcal { A } : = \{ a ^ { ( j ) } \} _ { i = 1 } ^ { N }$ where $a ^ { ( j ) } \in \{ 0 , 1 \} ^ { T }$ is a one-hot vector encoding its atomic type. The molecular crystal definition is formulated rigorously in Definition B.1. In principle, this distribution is induced by the laws of quantum mechanics and thermodynamics (the latter only for nonzero temperatures). For the purposes of generative model training we use a dataset $\bar { \mathcal { D } }$ of energetically stable or experimentally realizable crystals as a proxy for Equation 1.

Factorization of the Joint Distribution In this work we exploit the hierarchical structure of molecular crystals to sample from a factorization of Equation 1. Splitting the joint distribution into intra- and inter-molecular factors lets us enforce strict molecular validity without sacrificing probabilistic consistency. To do so we consider a coarse-graining map (Definition $\mathbf { A . } 1 )$ that removes all intramolecular degrees of freedom, replacing each molecule i with a centroid $\boldsymbol { q } ^ { ( i ) } \in \mathbb { R } ^ { 3 }$ and an orientation $Q ^ { ( i ) } \in S O ( 3 )$ relative to a set of canonical coordinates $\{ \widetilde { c } ^ { ( j ) } \} _ { j \in S _ { i } }$ . Partitioning the atoms into M disjoint molecular subsets $\{ S _ { i } \} _ { i = 1 } ^ { M }$ and applying the coarse-graining map to each allows us to rewrite the target in Equation 1

$$
\begin{array} { r } { p \left( L , \{ c ^ { ( j ) } \} _ { j = 1 } ^ { N } \Big | \mathcal { A } \right) = p \left( L , \{ q ^ { ( i ) } , Q ^ { ( i ) } \} _ { i = 1 } ^ { M } \Big | C , \mathcal { A } \right) p ( C | \mathcal { A } ) , } \end{array}\tag{2}
$$

where $C : = \big ( \{ \tilde { c } ^ { ( j ) } \} _ { j \in S _ { 1 } } , \dots , \{ \tilde { c } ^ { ( j ) } \} _ { j \in S _ { M } } \big )$ collects the local atomic coordinates of each molecule in its frame. The first factor is the inter-molecular distribution over cell and rigid-body placements; the second is the intra-molecular conformer prior.

We further assume local atomic coordinates are mutually independent across molecules,

$$
p ( C \mid A ) \approx \prod _ { i = 1 } ^ { M } p \left( \{ { \tilde { c } } ^ { ( j ) } \} _ { j \in S _ { i } } \Big | \{ a ^ { ( j ) } \} _ { j \in S _ { i } } \right) ,\tag{3}
$$

which is reasonable when molecules are sufficiently rigid and only weakly perturbed by their environment. Additionally, we assume each factor is concentrated around an a priori known conformer, so that C may be treated as fixed (rigid-body assumption). Together, these reduce the learning problem to the inter-molecular factor alone. Both assumptions can fail for flexible molecules, and we leave their relaxation to future work. In the absence of the true conformer, one must estimate it by another method before applying the current iteration of CG-OMatG.

We parameterize molecular centroid translations in fractional coordinates $f ^ { ( i ) } = \mathrm { w r a p } ( q ^ { ( i ) } L ^ { - 1 } ) \in \mathbb { T } ^ { 3 }$ , which simplifies the implementation of periodic boundary conditions [30]. The lattice matrix L itself also requires a parameterization suitable for Riemannian flow matching. Naively treating L as an element of $\mathbb { R } ^ { 3 \times 3 }$ does not guarantee that intermediate points along a flow remain valid unit cells. To address this, we decompose L via the polar decomposition [47]. Considering, for simplicity, the column-major representation of L which we write $L ^ { \prime } = \bar { L } ^ { \top }$ , this factors $L ^ { \prime } = U P$ uniquely into a symmetric positive-definite $P = ( L ^ { \prime \top } L ^ { \prime } ) ^ { 1 / 2 }$ and an orthogonal $U = L ^ { \prime } P ^ { - 1 }$ . Taking determinants gives det $\mathbf { \bar { \boldsymbol { L } } ^ { \prime } } = \operatorname* { d e t } \boldsymbol { U }$ det $P ;$ since det $P > 0$ and det $L ^ { \prime } > 0$ (as $L ^ { \prime } \in G L ^ { + } ( 3 , \mathbb { R } ) )$ ), we have det $U = 1$ , so $U \in { \tilde { S O } } ( 3 )$ This gives the manifold of unit cells as $G L ^ { + } ( 3 , \mathbb { R } ) \cong S O ( 3 ) \times \mathrm { S y m _ { 3 } ^ { + } }$ , where $\mathrm { S y m _ { 3 } ^ { + } }$ is the set of symmetric positivedefinite $3 \times 3$ matrices. This choice ensures that at all times during flow the unit cell is nondegenerate and has positive volume; we comment briefly on this point in Appendix E. For $L ^ { \zeta } = U P$ , the metric tensor is $G = L ^ { \prime } ^ { \top } L ^ { \prime } = \dot { P } ^ { 2 }$ . We retain $U$ because the cell and molecular orientations share a Cartesian frame, and we do not pre-rotate the OMC or CSD data during preprocessing. Together these identities allow us to write the target probability data distribution as

$$
\varphi ( x ) : = p \left( U , P , \{ f ^ { ( i ) } , Q ^ { ( i ) } \} _ { i = 1 } ^ { M } \Big | C , A \right) .\tag{4}
$$

Molecular Crystal Manifold The variable $x : = ( U , P , \{ f ^ { ( i ) } , Q ^ { ( i ) } \} _ { i = 1 } ^ { M } )$ lives on a partially curved product space rather than Euclidean space. To apply Riemannian flow matching, we collect these variables into a single product manifold M defined a

$$
\mathcal { M } : = S O ( 3 ) \times \mathrm { S y m } _ { 3 } ^ { + } \times \left( \mathbb { T } ^ { 3 } \times S O ( 3 ) \right) ^ { M } .\tag{5}
$$

Each point x specifies a molecular crystal configuration. The complete definition of the manifold along with its Riemannian metric is provided in Appendix C. By summing the metric on each sub-manifold, M is trivially a Riemannian manifold (see [48, Eq. 3.3] and [49, Examples 1.8 and 13.2]). The induced distance, logarithm, and exponential maps are given in Appendix D; they define the closed-form geodesics used to construct the conditional velocity field in Section 3.2.

## 3.2 Learning Crystal Packings with Riemannian Flow Models

Geometric Flow Modeling Assuming access to samples $x _ { 1 }$ from some unknown data distribution $\varphi : \mathcal { M } \to \mathbb { R } _ { \geq 0 }$ along with an easy-to-sample prior $p _ { 0 } : { \mathcal { M } } \to \mathbb { R } _ { \geq 0 }$ , the goal is to learn a bijection $\Psi : \mathcal { M }  \mathcal { M }$ which pushes $p _ { 0 }$ forward to closely approximate $\varphi$ . We learn this bijection by borrowing ideas from dynamical measure transport [50–52]. Specifically, we aim to parameterize a time-dependent velocity field $u _ { t } : [ 0 , 1 ] \times \mathbf { \bar { \mathcal { M } } }  T _ { x } { \mathcal { M } }$ in the manifold ODE

$$
\begin{array} { r } { \frac { d } { d t } \psi _ { t } ( x ) = u _ { t } ( \psi _ { t } ( x ) ) , \qquad \psi _ { 0 } ( x ) = x . } \end{array}\tag{6}
$$

with solution $\psi _ { t }$ . The flow map induces a family of pushed forward densities of the form

$$
\log p _ { t } ( x ) = \log p _ { 0 } \big ( \psi _ { t } ^ { - 1 } ( x ) \big ) - \int _ { 0 } ^ { t } \mathrm { d i v } _ { g } \big ( u _ { s } ( x _ { s } ) \big ) d s .\tag{7}
$$

As the solution to the ODE is deterministic and may be time-reversed, we know it is invertible. This lets us define the bijection Ψ as the $t = 1$ solution of this ODE, $i . e . , \Psi : = \psi _ { 1 }$ . The goal then is to approximate $u _ { t }$ by some neural network $b _ { t } ^ { \theta } : \mathcal { M }  T _ { x } \mathcal { M }$ , which in turn induces an approximate bijection which can be used for generating molecular crystal configurations.

Riemannian Conditional Flow Matching Chen and Lipman [53] show that the minimizer of the Riemannian conditional flow matching (RCFM) objective provides a training target for the model velocity $b _ { t } ^ { \theta }$ such that the pushforward satisfies $p _ { 1 } \approx \varphi$ when optimized. The RCFM loss is

$$
\mathcal { L } [ b _ { t } ^ { \theta } ] = \mathbb { E } \left[ \left. b _ { t } ^ { \theta } ( x ) - u _ { t } ( x \mid x _ { 1 } ) \right. _ { g } ^ { 2 } \right] ,\tag{8}
$$

where the expectation is taken over $t \sim U ( [ 0 , 1 ] ) , x _ { 1 } \sim \varphi ( x _ { 1 } )$ , and $x \sim p _ { t } ( x \mid x _ { 1 } )$ . The function $p _ { t } ( \cdot \mid x _ { 1 } )$ $\mathcal { M } \to \mathbb { R } _ { > 0 }$ denotes a conditional probability path satisfying the boundary conditions $p _ { 1 } ( x \mid x _ { 1 } ) = \delta _ { x _ { 1 } } ( x )$ and $p _ { 0 } ( x \mid x _ { 1 } { \overline { { ) } } } = p _ { 0 } ( x )$ , meaning that at $t = 1$ it is tightly distributed about the conditioning point, while at $t = 0$ it reproduces the easy-to-sample prior. This density is induced by a conditional velocity $\boldsymbol { u } _ { t } ( \boldsymbol { x } \mid \boldsymbol { x } _ { 1 } )$ ) whose ODE solution is $\psi _ { t } ( x \mid x _ { 1 } )$ , with initial condition $\psi _ { 0 } ( x \mid x _ { 1 } ) = x$ , generates the conditional density via the push-forward.

Parameterization of Conditional Velocity We parametrize the conditional velocity using geodesics on M defined through the Riemannian logarithm and exponential maps. Specifically we set

$$
x _ { t } : = \psi _ { t } ( x _ { 0 } | x _ { 1 } ) = \exp _ { x _ { 0 } } ^ { \mathcal { M } } \left( t \log _ { x _ { 0 } } ^ { \mathcal { M } } ( x _ { 1 } ) \right) .\tag{9}
$$

where $x _ { t }$ is just shorthand for the conditional ODE solution $\psi _ { t } ( x _ { 0 } | x _ { 1 } )$ as visualized in Figure 3 in Appendix section C. Since M is a product manifold, this path is obtained by evolving each component along its corresponding geodesic. For the fractional coordinates this gives the minimum-image straight line on the torus [33],

$$
f _ { t } = \psi _ { t } ^ { \mathbb { T } ^ { 3 } } ( f _ { 0 } \mid f _ { 1 } ) = f _ { 0 } + t ( f _ { 1 } - f _ { 0 } - n ^ { \star } ) \bmod \mathbb { Z } ^ { 3 } ,\tag{10}
$$

where $\begin{array} { r } { n ^ { \star } \in \arg \operatorname* { m i n } _ { n \in \mathbb { Z } ^ { 3 } } \| ( f _ { 1 } - f _ { 0 } ) - n \| } \end{array}$ is the integer translation that minimizes the distance between $f _ { 0 }$ and $f _ { 1 }$ under periodic boundary conditions. For rotations, the same geodesic applies to both the molecular orientation $Q _ { t }$ and the cell orientation $U _ { t }$ . Writing generically $R _ { t } \in S O ( 3 )$ with endpoints $R _ { 0 }$ and $R _ { 1 }$ , and defining $\Omega = \log _ { R _ { 0 } } ^ { S O ( 3 ) } ( R _ { 1 } )$ and $\begin{array} { r } { \theta = ( - \frac { 1 } { 2 } \operatorname { T r } \left( ( R _ { 0 } ^ { \top } \Omega ) ^ { 2 } \right) ) ^ { 0 . 5 } } \end{array}$ , the geodesic is evaluated using the Rodrigues formula [54]

$$
R _ { t } = \psi _ { t } ^ { S O ( 3 ) } ( R _ { 0 } \mid R _ { 1 } ) = R _ { 0 } \left( I + { \frac { \sin ( t \theta ) } { \theta } } ( R _ { 0 } ^ { \top } \Omega ) + 2 { \frac { \sin ^ { 2 } ( t \theta / 2 ) } { \theta ^ { 2 } } } ( R _ { 0 } ^ { \top } \Omega ) ^ { 2 } \right) .\tag{11}
$$

Finally, the symmetric positive-definite cell component follows the affine-invariant geodesic [55]

$$
P _ { t } = \psi _ { t } ^ { \mathrm { S y m } _ { 3 } ^ { + } } ( P _ { 0 } \mid P _ { 1 } ) = P _ { 0 } ^ { \frac { 1 } { 2 } } \exp \left( t \log \left( P _ { 0 } ^ { - { \frac { 1 } { 2 } } } P _ { 1 } P _ { 0 } ^ { - { \frac { 1 } { 2 } } } \right) \right) P _ { 0 } ^ { \frac { 1 } { 2 } } ,\tag{12}
$$

where the matrix logarithm, matrix exponential, and principal square root are each evaluated by diagonalizing the argument, applying the corresponding scalar function to the eigenvalues, and reconstructing the matrix from the resulting spectrum.

## 3.3 Constraints on the Velocity $b _ { t } ^ { \theta }$

Tangency Constraints A standard flow model parametrizes a map $\mathbb { R } ^ { d } \to \mathbb { R } ^ { d }$ , but on a manifold the velocity must satisfy $b _ { t } ^ { \theta } ( x ) \in T _ { x } { \mathcal { M } }$ . Two common methods to enforce this are projecting an ambient prediction onto the tangent space via ${ P _ { x } } : { \mathbb { R } } ^ { \mathrm { { e m b } } ( \mathcal { M } ) }  T _ { x } { \mathcal { M } } \ [ 5 6 \mathrm { - } 5 8 , 5 3 ]$ , or predicting an endpoint $\hat { x } _ { 1 } \in \mathcal { M }$ and recovering the velocity as $b _ { t } ^ { \theta } ( x _ { t } ) = \log _ { x _ { t } } ^ { \mathcal { M } } ( \hat { x } _ { 1 } ) \left[ 5 9 , 2 1 , 2 2 \right]$ . Following the generator-based construction of Falorsi and Forré [60, Appendix B.2], we use a third parameterization based on the Lie group structure: the network outputs an unconstrained $\omega \in \mathbb { R } ^ { \dim \mathfrak { g } }$ which is hat-mapped to the Lie algebra ${ \widehat { \omega } } \in { \mathfrak { g } }$ and left-translated to give $b _ { t } ^ { \theta } ( x _ { t } ) \stackrel { ^ { \bullet } } { = } x _ { t } \widehat { \omega }$ . Since $T _ { x } { \mathcal { M } } = x { \mathfrak { g } }$ for any matrix Lie group, tangency holds by construction. Our model uses all three parameterizations. The cell rotation head takes the Lie-algebra route, outputting $\boldsymbol \omega \in \mathbb { R } ^ { 3 }$ and hat-mapping to $\textcircled { \omega } \in \mathfrak { s o } ( \bar { 3 } )$ to obtain a velocity in $T _ { U _ { t } } \mathrm { S O ( 3 ) }$ . The per-molecule orientation head uses the log-map, the lattice shape head uses projection via Voigt-vector readout, and the centroid head is Euclidean. Full per-head details are in Appendix H.

Symmetry Constraints Crystal structures, like many physical systems, are known to exhibit symmetries correspond ing to conservation laws [61]. It is well established that accounting for symmetries leads to higher-quality samples and networks that generalize better [62–67]. Molecular crystals exhibit a very rich symmetry structure: translations, rotations, lattice basis changes, PCA sign ambiguity [68], conformer point group operations, space group operations, and permutations of both atom and molecule labels. A complete mathematical treatment is deferred to Appendix G; here we focus on the subset of symmetries that our generative framework must actively handle. Abstractly, a symmetry group G acts on the manifold $\dot { \mathcal { M } }$ . For $x \in \mathcal { M }$ , its orbit is $G x = \{ g x \mid g \in G \}$ , and the quotient $\mathcal { M } / G$ identifies configurations that lie in the same orbit. It would therefore be natural to define the velocity directly on the quotient, with $\overline { { b } } _ { t } ^ { \theta } ( [ x ] ) \in T _ { [ x ] } ( \mathcal { M } / G )$ [69]. This strategy has been applied to space-group-constrained crystal generation [40], where the Wyckoff position provides a natural representative of each orbit, and to pose prediction [70] by selecting the rotation closest to the identity as a canonical representative. In both cases, the symmetry is quotiented out by choosing a representative $x _ { \mathrm { r e p } } ( [ x ] ) \in [ x ] \subset \mathcal { M }$ prior to training. Such canonicalization would require choosing an orbit representative before training; we do not impose that preprocessing convention here.

Equivariant Networks Although we could work on the quotient space so that each state corresponds to a unique molecular crystal, we instead adopt an equivariant modeling approach. Kohler et al. [71] proved that the pushforward of a G-equivariant $\psi _ { t }$ produces a G-invariant density as long as the base density is at least G-invariant. Corresponding extensions to Riemannian flow models have been proven as well [72]. Their theorems state that the velocity network $b _ { t } ^ { \forall }$ must be equivariant. Letting $\Phi _ { g } ( x ) = g x$ denote the left group action, we require for each symmetry $g \in G$ and for each point on the manifold $x \in \mathcal { M }$

$$
b _ { t } ^ { \theta } ( \Phi _ { g } ( x ) ) = ( d \Phi _ { g } ) _ { x } ( b _ { t } ^ { \theta } ( x ) ) \in T _ { \Phi _ { g } ( x ) } { \mathcal { M } }\tag{13}
$$

where the differential acts as $( d \Phi _ { g } ) _ { x } : T _ { x } \mathcal { M }  T _ { \Phi _ { a } ( x ) } \mathcal { M }$ and represents the action of the symmetry on the velocity. This is the differential geometric version of the rather intuitive statement: when the system is rotated, the velocity vectors rotate with it. In Appendix G we describe the differential for the rotation and translation groups $( S O ( 3 )$ and $\mathbb { R } ^ { 3 }$ respectively) acting on configurations $x \in \mathcal { M }$ , and we further apply an equivariant neural network to enforce this symmetry directly [73]. A full prescription of the architecture with numerical estimates of the equivariance error [67] is reported in Appendix H.

We also apply a new type of data augmentation that handles both the frame ambiguity and the point group ambiguity of the CG pose assigned to a given molecule. We perturb atomic positions with small isotropic noise $( \approx 0 . 0 1 \mathring { \mathrm { A } } )$ , apply the coarse-graining map in Definition A.1 to obtain a unique orientation from the noisy positions, and then express the original un-noised positions in the resulting orientation to obtain local coordinates. This procedure spans the set of possible poses that can be assigned to a molecule with non-trivial symmetries, allowing the training procedure to be robust to the inherent ambiguity in orientation assignment and removing the need for canonicalization present in othe point cloud pose estimation schemes [70]. Rather, we simply train on all equivalent orientations.

## 3.4 Reinforcement Learning on the Molecular Crystal Manifold

Flow models can be post-trained with Flow-GRPO [74], which converts a flow ODE into a marginally equivalent SDE. However, its derivation requires a Gaussian base distribution and a linear interpolant, neither of which is available for our manifold-valued flow. We therefore cannot apply Flow-GRPO directly and instead build on the more general surrogate-SDE framework of Höllmer and Martiniani [18], extending it to dynamics on M. The resulting log-likelihood factorizes into a tangent-space Gaussian and a parameter-independent Jacobian that cancels from the importance ratio and KL term. Alternative RL formulations for materials generation include Reinforce Adjoint Matching, used by OMatG-flash to post-train flow maps [75].

Promotion of the ODE to an SDE To enable exploration during RL, Höllmer and Martiniani [18] replace a velocitybased ODE with stochastic surrogate dynamics obtained by adding isotropic Gaussian noise to the numerical integration increment. For sufficiently small noise, this leaves evaluation metrics unchanged. In our case, we inject isotropic Gaussian noise with noise schedule $\sigma _ { t }$ in the tangent space $T _ { x _ { t } } { \mathcal { M } }$ before mapping back to the manifold via the exponential map:

$$
x _ { t + \Delta t } = \exp _ { x _ { t } } \left( \Delta t b _ { t } ^ { \theta } ( x _ { t } ) + \sigma _ { t } \sqrt { \Delta t } \xi _ { t } \right) , \qquad T _ { x _ { t } } \mathcal { M } \ni \xi _ { t } \sim \mathcal { N } ( 0 , I _ { \dim } \mathcal { M } ) .\tag{14}
$$

The resulting conditional probability distribution $\pi ^ { \theta } ( x _ { t + \Delta t } \mid x _ { t } )$ is a wrapped Gaussian on M [76], whose loglikelihood factors into a Euclidean Gaussian in the tangent space plus a θ-independent Jacobian term (see Appendix F for the factorization; the nontrivial $S O ( 3 )$ and Sym<sup>+</sup> Jacobians are given in Appendix D).

GRPO The iterative stochastic process in Equation 14 can be understood as a Markov decision process to enable policy-gradient RL [77], which aims to optimize the policy $\pi ^ { \theta } ( x _ { t + \Delta t } \mid x _ { t } )$ so that the expected terminal-only reward $r ( x _ { 1 } )$ is maximized. As the reward in our setting, we use the negative all-atom energy from an MLIP—UMA or, in our ablation, Orb—so that generated structures are biased towards smaller energies [78, 79]. GRPO samples G trajectories $\tau ^ { 1 : G }$ under identical conditioning—in our case, for the same molecular crystal—and maximizes the following surrogate objective [17]:

$$
\mathcal { L } _ { \mathrm { G R P O } } ( \theta ) = \frac { 1 } { S G K } \mathbb { E } _ { \tau ^ { 1 : G } \sim \pi ^ { \theta _ { \mathrm { o l d } } } } \left[ \sum _ { i = 1 } ^ { G } \sum _ { k = 0 } ^ { K - 1 } \operatorname* { m i n } \bigl ( \rho _ { i , k } ( \theta ) \hat { A } _ { i } , \exp ( \rho _ { i , k } ( \theta ) , 1 - \varepsilon , 1 + \varepsilon ) \hat { A } _ { i } \bigr ) \right] .\tag{15}
$$

Here, K is the number of integration time steps, S is an optional normalization factor that accounts for the variable atom count across different molecular crystals [18], and ε is a clipping hyperparameter. We further used the group-normalized advantages $\hat { A } _ { i } = ( r ( x _ { 1 } ^ { i } ) - \mathrm { m e a n } \{ r ( x _ { 1 } ^ { j } ) \} ) / \mathrm { s t d } \{ r ( x _ { 1 } ^ { j } ) \}$ , and the one-step ratio $\rho _ { i , k } ( \theta ) = \pi ^ { \theta } ( x _ { t _ { k + 1 } } ^ { i } \mid x _ { t _ { k } } ^ { i } ) / \pi ^ { \theta _ { \mathrm { o l d } } } ( x _ { t _ { k + 1 } } ^ { i } \mid$ $x _ { t _ { k } } ^ { i } )$ between the updated policy $\pi ^ { \theta }$ and the old policy $\pi ^ { \theta _ { \mathrm { o l d } } }$ that generated the trajectories. This one-step ratio reduces to a ratio of Euclidean Gaussians in $T _ { \boldsymbol { x } _ { t _ { k } } } \mathcal { M } \cong \mathbb { R } ^ { 6 M + 9 }$ because, for policies compared at the same base point $x _ { t _ { k } }$ , the θ-independent Jacobian terms cancel. Besides the objective in Equation 15, we use a KL-regularization with respect to the pretrained policy $\pi ^ { \theta _ { \mathrm { r e f } } }$ , evaluated in closed form between the corresponding tangent-space transitions (Appendix F).

## 4 Results

Datasets We train our model on two separate molecular crystal datasets: OMC25- MCF [80], a subset of the Open Molecular Crystals dataset curated by Zeng et al. [22] containing 46, 120 structures; and CSD [81], a proprietary dataset of 400, 057 experimentally validated molecular crystal structures maintained by the CCDC and available through the purchase of a license. In both cases, we restrict ourselves to homomolecular crystals and leave cocrystalline and solvated materials as an avenue for future investigation. A complete summary of how the data is preprocessed is provided in Appendix M.

Table 1: CCDC packing-similarity metrics on the first 128 structures of the OMC25-MCF test set $( k = 3 0$ inference). Values are rates; ↑ higher is better and ↓ lower is better.
<table><tr><td>Metric</td><td>MCF†</td><td>Base</td><td>Orb-IRL</td><td>UMA-IRL</td></tr><tr><td>Solved ↑</td><td>0.0391</td><td> $0 . 0 7 4 2 \pm 0 . 0 0 3 9$ </td><td> $0 . 1 0 0 8 \pm 0 . 0 0 4 6$ </td><td> $\mathbf { 0 . 1 2 7 3 \pm 0 . 0 0 8 1 }$ </td></tr><tr><td>Solved (coll. allowed) ↑ 0.1797</td><td></td><td> $0 . 2 1 0 9 \pm 0 . 0 0 6 4$ </td><td> $0 . 2 2 2 7 \pm 0 . 0 0 5 5$ </td><td> $\mathbf { 0 . 2 5 9 4 \pm 0 . 0 0 6 4 }$ </td></tr><tr><td>Packing match ↑</td><td>0.4062</td><td> $0 . 4 6 8 0 \pm 0 . 0 1 1 5$ </td><td> $0 . 5 0 3 1 \pm 0 . 0 0 8 5$ </td><td> $\mathbf { 0 . 5 7 6 6 \pm 0 . 0 0 7 9 }$ </td></tr><tr><td>Packing match/draw ↑</td><td>0.0534</td><td> $0 . 0 6 7 9 \pm 0 . 0 0 1 3$ </td><td> $0 . 0 7 4 9 \pm 0 . 0 0 0 9$ </td><td> $\mathbf { 0 . 0 8 6 5 \pm 0 . 0 0 0 9 }$ </td></tr><tr><td>Clash ↓</td><td>0.6182</td><td> $0 . 3 8 9 6 \pm 0 . 0 0 2 2$ </td><td> $0 . 2 7 1 6 \pm 0 . 0 0 1 6$ </td><td> $\mathbf { 0 . 1 7 3 2 \pm 0 . 0 0 1 5 }$ </td></tr></table>

For CG-OMatG variants, values are the mean ± SEM over ten blocks of 30 draws per target (300 draws total). <sup>†</sup>MCF has one K = 30 block and therefore no error bars.

Open Molecular Crystals On the first 128 structures of OMC25-MCF—the lowest-uma-s-1p1-energy subset of Open Molecular Crystals [80, 78] introduced by Zeng et al. [22]—we compare MCF with the base model and models reinforced using UMA or Orb rewards. The stronger, longer-trained UMA run supplies the main CG-OMatG-IRL results; the shorter Orb run and reward-circularity analysis are discussed in Appendix K. The Orb gains show that the improvement is not specific to UMA.

Results are presented in Table 1. To quantify substantial violations of physical interactions, we report clash rates, where a clash occurs if the distance between two intermolecular heavy atoms is less than 0.75 times the sum of their covalent radii [39]. We define packing similarity using COMPACK, where a packing match is recorded when at least eight of fifteen molecules in the packing shell can be aligned [82]. Finally, a target is considered solved when the structures are packing similar, possess $\mathrm { R M S D } _ { N _ { \mathrm { M a t c h e s } } } \leq 2 . 0 \mathring \mathrm { A }$ , and contain no collisions. Collisions are stricter than clashes and occur when the distance between two intermolecular atoms falls below the sum of their van der Waals radii minus 0.7 Å [37]. “Collisions allowed” applies the same packing and RMSD criteria without the collision screen. Packing match (per draw) and clash are the respective fractions of the 30 draws satisfying the packing-match and clash criteria. The base model nearly doubles MCF’s solved rate, while UMA-RL exceeds three times it and further improves the collision-screened solve rate relative to the base model. Rewards rise and clashes fall during RL on both datasets (Appendix Figure 10), empirically validating the manifold policy-gradient construction and supporting the idea that it distills MLIP physicality into the generator.

![](images/0c00fd0ac59028696a8af0d2dcb9cbcb01f6f960850f75ec7ef5c75ad62805d5.jpg)  
Figure 2: UMA-driven relaxation for the three homomolecular CSP blind-test 6 targets (NACJAF, XAFPAY, XAFQIH), showing the ten lowest-energy draws per generator. Top: energy deviation from the UMA-relaxed ground truth across the three-stage BFGS relaxation; bottom: final deviations sorted by energy. Energies are in eV/atom. Relaxation details and complementary energy–density analyses are in Appendix I and Appendix K.

Cambridge Structural Database We benchmark CG-OMatG on CSP blind-test data [81]. The CSP blind test refers to an annual competition hosted by the CCDC in which scientists aim to predict experimentally validated, yet previously unseen, crystal structures. We train CG-OMatG on a large dataset curated from the CSD (Appendix Section M). In Figure 2, we demonstrate the performance of CG-OMatG on three crystal targets from the sixth CCDC blind test. We benchmark CG-OMatG against MCF and Genarris 3.0, a popular statistical algorithm for proposing molecular crystal structures [27]. Quantitatively, CG-OMatG matches NACJAF before relaxation (8/15 molecules at 1.86 Å) and after relaxation (11/15 at 0.37 Å), and XAFPAY after relaxation (8/15 at 1.51 Å). No method solves XAFQIH, although relaxation improves the best CG-OMatG-IRL match from 2.35 to 2.03 Å. Appendix K reports full blind-test metrics and further studies of velocity annealing, conformer choice, polymorph diversity, and additional benchmark comparisons. In the conformer ablation, ETKDG/MMFF94s sampling recovers conformers within 1 Å of the experimental structures, although inference from the generated conformers does not yield an additional solved target.

## 5 Discussion

Methodological Contributions To our knowledge, this work is the first to formulate policy-gradient optimization of flow models on a non-trivial Riemannian manifold. This construction circumvents the need to evaluate probability densities using the computationally expensive divergence. We additionally introduce a data augmentation scheme that accounts for the symmetries of rigid bodies; because these symmetries arise generically in pose prediction problems, we suspect the approach has implications beyond molecular CSP. Lastly, this work provides the first formal construction of the Riemannian manifold of all possible molecular crystal configurations. These contributions are primarily theoretical and provide a rigorous geometric foundation on which future generative models for molecular crystals and, more generally, periodic rigid-body systems can build.

Limitations A current limitation of our approach is that the number of molecules in the unit cell, M, must be specified at inference time. In a true blind-test setting where only the molecular graph is given, this requires running the full generation-plus-relaxation pipeline for each candidate number M. Additionally, we assume that conformer degrees of freedom factorize and are delta distributed from the crystal packing problem in equation 3. This assumption can be insidious for highly flexible molecules and warrants reexamination in future work. Rigidity applies only during proposal generation: subsequent unconstrained atomistic relaxation can correct moderate conformational errors, but cannot replace explicit conformational sampling. Another limitation of this work is the observed high clash rates. The rigid body approximation, which we exploit in this work, makes molecular crystal systems highly sensitive to improperly learned rotations, which manifest as clashes. These effects are dramatic in systems featuring long, rod-like molecules. Finally, we restrict our study to homomolecular crystals, whereas cocrystals containing multiple distinct molecular species are common in nature and represent an important extension for future work.

## Acknowledgments

The authors thank the NYU IT High Performance Computing team for their provision of computational resources and general support. The authors acknowledge funding from NSF Grant OAC-2311632. S. M. acknowledges support from the Simons Center for Computational Physical Chemistry (Simons Foundation grant 839534, MT). The authors gratefully acknowledge use of the research computing resources of the Empire AI Consortium, Inc., with support from the State of New York, the Simons Foundation, and the Secunda Family Foundation. We thank Shenglong Wang for reserving compute nodes and helping resolve a CCDC license-validation issue.

This material is based upon work supported by the National Science Foundation under Grant Number 2345719. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.

Large language models assisted with manuscript drafting and editing and research code development; the author verified all results and claims and take full responsibility for the work.

## References

[1] Sarah L. Price. Control and prediction of the organic solid state: a challenge to theory and experiment†. Proceedings ofthe Royal Society A: Mathematical, Physical and Engineering Sciences, 474(2217):20180351, September 2018. ISSN 1364-5021. doi: 10.1098/rspa.2018.0351. URL https://doi.org/10.1098/rspa. 2018.0351.

[2] Jonas Nyman and Graeme M. Day. Static and lattice vibrational energy differences between polymorphs. CrystEngComm, 17(28):5154–5165, 2015. ISSN 1466-8033. doi: 10.1039/C5CE00045A. URL https://xlink. rsc.org/?DOI=C5CE00045A.

[3] Yongbo Yuan, Gaurav Giri, Alexander L. Ayzner, Arjan P. Zoombelt, Stefan C. B. Mannsfeld, Jihua Chen, Dennis Nordlund, Michael F. Toney, Jinsong Huang, and Zhenan Bao. Ultra-high mobility transparent organic thin film transistors grown by an off-centre spin-coating method. Nature Communications, 5(1):3005, January 2014. ISSN 2041-1723. doi: 10.1038/ncomms4005. URL https://www.nature.com/articles/ncomms4005.

[4] Marcus A. Neumann and Jacco Van De Streek. How many ritonavir cases are there still out there? Faraday Discussions, 211:441–458, 2018. ISSN 1359-6640, 1364-5498. doi: 10.1039/C8FD00069G. URL https: //xlink.rsc.org/?DOI=C8FD00069G.

[5] Huanhuan Zhao and Bryony J. James. Fat bloom formation on model chocolate stored under steady and cycling temperatures. Journal ofFood Engineering, 249:9–14, May 2019. ISSN 02608774. doi: 10.1016/j.jfoodeng.2018. 12.008. URL https://linkinghub.elsevier.com/retrieve/pii/S0260877418305272.

[6] Jingxiang Yang, Bryan Erriah, Chunhua T. Hu, Ethan Reiter, Xiaolong Zhu, Vilmalí López-Mejías, Isis Paola Carmona-Sepúlveda, Michael D. Ward, and Bart Kahr. A deltamethrin crystal polymorph for more effective malaria control. Proceedings ofthe National Academy ofSciences, 117(43):26633–26638, October 2020. ISSN 0027-8424, 1091-6490. doi: 10.1073/pnas.2013390117. URL https://pnas.org/doi/full/10.1073/pnas. 2013390117.

[7] Jack D. Dunitz and Joel Bernstein. Disappearing Polymorphs. Accounts ofChemical Research, 28(4):193–200, April 1995. ISSN 0001-4842, 1520-4898. doi: 10.1021/ar00052a005.

[8] J. P. M. Lommerse, W. D. S. Motherwell, H. L. Ammon, J. D. Dunitz, A. Gavezzotti, D. W. M. Hofmann, F. J. J. Leusen, W. T. M. Mooij, S. L. Price, B. Schweizer, M. U. Schmidt, B. P. van Eijck, P. Verwer, and D. E. Williams. A test of crystal structure prediction of small organic molecules. Acta Crystallographica Section B: Structural Science, 56(4):697–714, August 2000. ISSN 0108-7681. doi: 10.1107/S0108768100004584. URL https://journals.iucr.org/b/issues/2000/04/00/bk0070/. Publisher: International Union of Crystallography.

[9] Lily M. Hunnisett, Jonas Nyman, Nicholas Francia, Nathan S. Abraham, Claire S. Adjiman, Srinivasulu Aitipamula, Tamador Alkhidir, Mubarak Almehairbi, Andrea Anelli, Dylan M. Anstine, John E. Anthony, Joseph E. Arnold, Faezeh Bahrami, Michael A. Bellucci, Rajni M. Bhardwaj, Imanuel Bier, Joanna A. Bis, A. Danie Boese, David H. Bowskill, James Bramley, Jan Gerit Brandenburg, Doris E. Braun, Patrick W. V. Butler, Joseph Cadden, Stephen Carino, Eric J. Chan, Chao Chang, Bingqing Cheng, Sarah M. Clarke, Simon J. Coles, Richard I. Cooper, Ricky Couch, Ramon Cuadrado, Tom Darden, Graeme M. Day, Hanno Dietrich, Yiming Ding, Antonio DiPasquale, Bhausaheb Dhokale, Bouke P. Van Eijck, Mark R. J. Elsegood, Dzmitry Firaha, Wenbo Fu, Kaori Fukuzawa, Joseph Glover, Hitoshi Goto, Chandler Greenwell, Rui Guo, Jürgen Harter, Julian Helfferich, Detlef W. M. Hofmann, Johannes Hoja, John Hone, Richard Hong, Geoffrey Hutchison, Yasuhiro Ikabata, Olexandr Isayev, Ommair Ishaque, Varsha Jain, Yingdi Jin, Aling Jing, Erin R. Johnson, Ian Jones, K. V. Jovan Jose, Elena A. Kabova, Adam Keates, Paul F. Kelly, Dmitry Khakimov, Stefanos Konstantinopoulos, Liudmila N. Kuleshova, He Li, Xiaolu Lin, Alexander List, Congcong Liu, Yifei Michelle Liu, Zenghui Liu, Zhi-Pan Liu, Joseph W. Lubach, Noa Marom, Alexander A. Maryewski, Hiroyuki Matsui, Alessandra Mattei, R. Alex Mayo, John W. Melkumov, Sharmarke Mohamed, Zahrasadat Momenzadeh Abardeh, Hari S. Muddana, Naofumi Nakayama, Kamal Singh Nayal, Marcus A. Neumann, Rahul Nikhar, Shigeaki Obata, Dana O’Connor, Artem R. Oganov, Koji Okuwaki, Alberto Otero-de-la Roza, Constantinos C. Pantelides, Sean Parkin, Chris J. Pickard, Luca Pilia, Tatyana Pivina, Rafał Podeszwa, Alastair J. A. Price, Louise S. Price, Sarah L. Price, Michael R. Probert, Angeles Pulido, Gunjan Rajendra Ramteke, Atta Ur Rehman, Susan M. Reutzel-Edens, Jutta Rogal, Marta J. Ross, Adrian F. Rumson, Ghazala Sadiq, Zeinab M. Saeed, Alireza Salimi, Matteo Salvalaglio, Leticia Sanders De Almada, Kiran Sasikumar, Sivakumar Sekharan, Cheng Shang, Kenneth Shankland, Kotaro Shinohara, Baimei Shi, Xuekun Shi, A. Geoffrey Skillman, Hongxing Song, Nina Strasser, Jacco Van De Streek, Isaac J. Sugden, Guangxu Sun, Krzysztof Szalewicz, Benjamin I. Tan, Lu Tan, Frank Tarczynski, Christopher R. Taylor, Alexandre Tkatchenko, Rithwik Tom, Mark E. Tuckerman, Yohei Utsumi, Leslie Vogt-Maranto, Jake Weatherston, Luke J. Wilkinson, Robert D. Willacy, Lukasz Wojtas, Grahame R. Woollam, Zhuocen Yang, Etsuo Yonemochi, Xin Yue, Qun Zeng, Yizu Zhang, Tian Zhou, Yunfei Zhou, Roman Zubatyuk, and Jason C. Cole. The seventh blind test of crysta structure prediction: structure generation methods. Acta Crystallographica Section B Structural Science, Crystal Engineering and Materials, 80(6):517–547, December 2024. ISSN 2052-5206. doi: 10.1107/S2052520624007492. URL https://journals.iucr.org/paper?S2052520624007492.

[10] Farren Curtis, Xiayue Li, Timothy Rose, Álvaro Vázquez-Mayagoitia, Saswata Bhattacharya, Luca M. Ghiringhelli, and Noa Marom. GAtor: A First Principles Genetic Algorithm for Molecular Crystal Structure Prediction. Faraday Discussions, 211:61–77, 2018. ISSN 1359-6640, 1364-5498. doi: 10.1039/C8FD00067K. URL http://arxiv.org/abs/1802.08602. arXiv:1802.08602 [cond-mat].

[11] Vahe Gharakhanyan, Yi Yang, Luis Barroso-Luque, Muhammed Shuaibi, Daniel S. Levine, Kyle Michel, Viachaslau Bernat, Misko Dzamba, Xiang Fu, Meng Gao, Xingyu Liu, Keian Noori, Lafe J. Purvis, Tingling Rao, Brandon M. Wood, Ammar Rizvi, Matt Uyttendaele, Andrew J. Ouderkirk, Chiara Daraio, C. Lawrence Zitnick, Arman Boromand, Noa Marom, Zachary W. Ulissi, and Anuroop Sriram. FastCSP: Accelerated Molecular Crystal Structure Prediction with Universal Model for Atoms, August 2025. URL http://arxiv.org/abs/2508.02641. arXiv:2508.02641 [physics].

[12] Kamal Singh Nayal, Dana O’Connor, Roman Zubatyuk, Dylan M. Anstine, Yi Yang, Rithwik Tom, Wenda Deng, Kehan Tang, Noa Marom, and Olexandr Isayev. Efficient Molecular Crystal Structure Prediction and Stability Assessment with AIMNet2 Neural Network Potentials, June 2025. URL https://chemrxiv.org/engage/ chemrxiv/article-details/685969b11a8f9bdab5210dc7.

[13] Jonathan Ho and Tim Salimans. Classifier-Free Diffusion Guidance, July 2022. URL http://arxiv.org/abs/ 2207.12598. arXiv:2207.12598 [cs].

[14] Pawan Prakash, Jason B. Gibson, Zhongwei Li, Gabriele Di Gianluca, Juan Esquivel, Eric Fuemmeler, Benjamin Geisler, Jung Soo Kim, Adrian Roitberg, Ellad B. Tadmor, Mingjie Liu, Stefano Martiniani, Gregory R. Stewart, James J. Hamlin, Peter J. Hirschfeld, and Richard G. Hennig. Guided Diffusion for the Discovery of New Superconductors, September 2025. URL http://arxiv.org/abs/2509.25186. arXiv:2509.25186 [condmat].

[15] Marta Skreta, Tara Akhound-Sadegh, Viktor Ohanesian, Roberto Bondesan, Alán Aspuru-Guzik, Arnaud Doucet, Rob Brekelmans, Alexander Tong, and Kirill Neklyudov. Feynman-Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts, June 2025. URL http://arxiv.org/abs/2503.02819. arXiv:2503.02819 [cs].

[16] Carles Domingo-Enrich, Michal Drozdzal, Brian Karrer, and Ricky T. Q. Chen. Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control, January 2025.

[17] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models, April 2024. URL http://arxiv.org/abs/2402.03300. arXiv:2402.03300 [cs].

[18] Philipp Höllmer and Stefano Martiniani. Open Materials Generation with Inference-Time Reinforcement Learning, January 2026. URL http://arxiv.org/abs/2602.00424. arXiv:2602.00424 [cs].

[19] Maya M. Martirossyan, Thomas Egg, Philipp Höllmer, George Karypis, Mark Transtrum, Adrian Roitberg, Mingjie Liu, Richard G. Hennig, Ellad B. Tadmor, and Stefano Martiniani. All that structure matches does not glitter, December 2025. URL http://arxiv.org/abs/2509.12178. arXiv:2509.12178 [cs].

[20] Hongyu Guo, Yoshua Bengio, and Shengchao Liu. AssembleFlow: Rigid Flow Matching with Inertial Frames for Molecular Assembly. October 2024. URL https://openreview.net/forum?id=jckKNzYYA6.

[21] Nayoung Kim, Seongsu Kim, Minsu Kim, Jinkyoo Park, and Sungsoo Ahn. MOFFlow: Flow Matching for Structure Prediction of Metal-Organic Frameworks, March 2025. URL http://arxiv.org/abs/2410.17270. arXiv:2410.17270 [q-bio].

[22] Cheng Zeng, Harry W. Sullivan, Thomas Egg, Maya M. Martirossyan, Philipp Höllmer, Jirui Jin, Richard G. Hennig, Adrian Roitberg, Stefano Martiniani, Ellad B. Tadmor, and Mingjie Liu. MolCrystalFlow: Molecular Crystal Structure Prediction via Flow Matching, March 2026. URL http://arxiv.org/abs/2602.16020. arXiv:2602.16020 [cs].

[23] Nikolaos Galanakis and Mark E. Tuckerman. Rapid prediction of molecular crystal structures using simple topological and physical descriptors. Nature Communications, 15(1):9757, November 2024. ISSN 2041-1723. doi: 10.1038/s41467-024-53596-5. URL https://www.nature.com/articles/s41467-024-53596-5.

[24] Sarah L. Price. Predicting crystal structures of organic compounds. Chemical Society Reviews, 43(7):2098–2111, 2014. doi: 10.1039/C3CS60279F.

[25] Gregory J. O. Beran. Modeling Polymorphic Molecular Crystals with Electronic Structure Theory. Chemical Reviews, 116(9):5567–5613, May 2016. ISSN 0009-2665. doi: 10.1021/acs.chemrev.5b00648.

[26] Xiayue Li, Farren S. Curtis, Timothy Rose, Christoph Schober, Alvaro Vazquez-Mayagoitia, Karsten Reuter, Harald Oberhofer, and Noa Marom. Genarris: Random Generation of Molecular Crystal Structures and Fast Screening with a Harris Approximation. The Journal ofChemical Physics, 148(24):241701, June 2018. ISSN 0021- 9606, 1089-7690. doi: 10.1063/1.5014038. URL http://arxiv.org/abs/1803.02145. arXiv:1803.02145 [physics].

[27] Yi Yang, Rithwik Tom, Jose A. G. L. Wui, Jonathan E. Moussa, and Noa Marom. Genarris 3.0: Generating Close-Packed Molecular Crystal Structures with Rigid Press, June 2025. URL https://chemrxiv.org/doi/ full/10.26434/chemrxiv-2025-046zn.

[28] Tian Xie, Xiang Fu, Octavian-Eugen Ganea, Regina Barzilay, and Tommi Jaakkola. Crystal Diffusion Variational Autoencoder for Periodic Material Generation, March 2022. URL http://arxiv.org/abs/2110.06197. arXiv:2110.06197 [cs].

[29] Rui Jiao, Wenbing Huang, Peijia Lin, Jiaqi Han, Pin Chen, Yutong Lu, and Yang Liu. Crystal Structure Prediction by Joint Equivariant Diffusion, March 2024. URL http://arxiv.org/abs/2309.04475. arXiv:2309.04475 [cond-mat].

[30] Benjamin Kurt Miller, Ricky T. Q. Chen, Anuroop Sriram, and Brandon M. Wood. FlowMM: Generating Materials with Riemannian Flow Matching, June 2024. URL http://arxiv.org/abs/2406.04713. arXiv:2406.04713 [cs].

[31] Claudio Zeni, Robert Pinsler, Daniel Zügner, Andrew Fowler, Matthew Horton, Xiang Fu, Zilong Wang, Aliaksandra Shysheya, Jonathan Crabbé, Shoko Ueda, Roberto Sordillo, Lixin Sun, Jake Smith, Bichlien Nguyen, Hannes Schulz, Sarah Lewis, Chin-Wei Huang, Ziheng Lu, Yichi Zhou, Han Yang, Hongxia Hao, Jielan Li, Chunlei Yang, Wenjie Li, Ryota Tomioka, and Tian Xie. A generative model for inorganic materials design. Nature, 639(8055):624–632, March 2025. ISSN 1476-4687. doi: 10.1038/s41586-025-08628-5. URL https://www.nature.com/articles/s41586-025-08628-5.

[32] Luis M. Antunes, Keith T. Butler, and Ricardo Grau-Crespo. Crystal structure generation with autoregressive large language modeling. Nature Communications, 15(1):10570, December 2024. ISSN 2041-1723. doi: 10. 1038/s41467-024-54639-7. URL https://www.nature.com/articles/s41467-024-54639-7. Publisher: Nature Publishing Group.

[33] Philipp Höllmer, Thomas Egg, Maya M. Martirossyan, Eric Fuemmeler, Zeren Shui, Amit Gupta, Pawan Prakash, Adrian Roitberg, Mingjie Liu, George Karypis, Mark Transtrum, Richard G. Hennig, Ellad B. Tadmor, and Stefano Martiniani. Open Materials Generation with Stochastic Interpolants, July 2025. URL http: //arxiv.org/abs/2502.02582. arXiv:2502.02582 [cs].

[34] Tin Hadži Veljkovic, Joshua Rosenthal, Ivor Lon ´ cari ˇ c, and Jan-Willem van de Meent. Crystalite: A Lightweight´ Transformer for Efficient Crystal Modeling, April 2026.

[35] Nayoung Kim, Seongsu Kim, and Sungsoo Ahn. Flexible MOF Generation with Torsion-Aware Flow Matching, May 2025. URL http://arxiv.org/abs/2505.17914. arXiv:2505.17914 [q-bio].

[36] Vaidotas Simkus, Anders Christensen, Steven Bennett, Ian Johnson, Mark Neumann, James Gin, Jonathan Godwin, and Benjamin Rhodes. Mofasa: A Step Change in Metal-Organic Framework Generation, December 2025.

[37] Emily Jin, Andrei Cristian Nica, Mikhail Galkin, Jarrid Rector-Brooks, Kin Long Kelvin Lee, Santiago Miret, Frances H. Arnold, Michael Bronstein, Avishek Joey Bose, Alexander Tong, and Cheng-Hao Liu. OXtal: An All-Atom Diffusion Model for Organic Crystal Structure Prediction, December 2025. URL http://arxiv.org/ abs/2512.06987. arXiv:2512.06987 [cs].

[38] A. L. Patterson. A Fourier Series Method for the Determination of the Components of Interatomic Distances in Crystals. Physical Review, 46(5):372–376, September 1934. doi: 10.1103/PhysRev.46.372.

[39] Akshay Subramanian, Elton Pan, Juno Nam, Maurice Weiler, Shuhui Qu, Cheol Woo Park, Tommi S. Jaakkola, Elsa Olivetti, and Rafael Gomez-Bombarelli. PackFlow: Generative Molecular Crystal Structure Prediction via Reinforcement Learning Alignment, February 2026. URL http://arxiv.org/abs/2602.20140. arXiv:2602.20140 [physics].

[40] Rui Jiao, Wenbing Huang, Yu Liu, Deli Zhao, and Yang Liu. Space Group Constrained Crystal Generation, April 2024. URL http://arxiv.org/abs/2402.03992. arXiv:2402.03992 [cs].

[41] Christopher Karpovich, Elton Pan, and Elsa A. Olivetti. Deep reinforcement learning for inverse inorganic materials design. npj Computational Materials, 10(1):287, Dec 2024. ISSN 2057-3960. doi: 10.1038/s41524-024-01474-5. URL https://doi.org/10.1038/s41524-024-01474-5.

[42] Trupti Mohanty, Maitrey Mehta, Hasan M. Sayeed, Bat-El Oded, Itay Pitussi, Arie Borenstein, Vivek Srikumar, and Taylor D. Sparks. CrysText: A generative ai approach for text-conditioned crystal structure generation using LLM. ChemRxiv, 2025(1103), 2025. doi: 10.26434/chemrxiv-2024-gjhpq-v2. URL https://chemrxiv.org/ doi/abs/10.26434/chemrxiv-2024-gjhpq-v2.

[43] Andy Xu, Rohan Desai, Larry Wang, Gabriel Hope, and Ethan Ritz. PLaID++: A preference aligned language model for targeted inorganic materials design, 2025. URL https://arxiv.org/abs/2509.07150.

[44] Hyunsoo Park and Aron Walsh. Guiding Generative Models to Uncover Diverse and Novel Crystals via Reinforcement Learning. https://arxiv.org/abs/2511.07158v1, November 2025.

[45] Junwu Chen, Jeff Guo, Edvin Fako, and Philippe Schwaller. Accelerating inverse materials design using generative diffusion models with reinforcement learning, November 2025.

[46] Zhendong Cao and Lei Wang. Reinforcement fine-tuning for materials design. Physical Review B, 113(2), January 2026. ISSN 2469-9969. doi: 10.1103/45zh-44bg. URL http://dx.doi.org/10.1103/45zh-44bg.

[47] Roger A. Horn and Charles R. Johnson. Matrix analysis. Cambridge University Press, Cambridge ; New York, 2nd ed edition, 2012. ISBN 978-0-521-83940-2.

[48] John M. Lee. Introduction to Riemannian Manifolds, volume 176 of Graduate Texts in Mathematics. Springer International Publishing, Cham, 2018. ISBN 978-3-319-91754-2 978-3-319-91755-9. doi: 10.1007/ 978-3-319-91755-9. URL http://link.springer.com/10.1007/978-3-319-91755-9.

[49] John M. Lee. Introduction to Smooth Manifolds, volume 218 of Graduate Texts in Mathematics. Springer New York, New York, NY, 2012. ISBN 978-1-4419-9981-8 978-1-4419-9982-5. doi: 10.1007/978-1-4419-9982-5. URL https://link.springer.com/10.1007/978-1-4419-9982-5.

[50] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow, September 2022.

[51] Michael S. Albergo and Eric Vanden-Eijnden. Building Normalizing Flows with Stochastic Interpolants, March 2023. URL http://arxiv.org/abs/2209.15571. arXiv:2209.15571 [cs].

[52] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow Matching for Generative Modeling, February 2023. URL http://arxiv.org/abs/2210.02747. arXiv:2210.02747 [cs].

[53] Ricky T. Q. Chen and Yaron Lipman. Flow Matching on General Geometries, February 2024. URL http: //arxiv.org/abs/2302.03660. arXiv:2302.03660 [cs].

[54] Olinde Rodrigues. Des lois géométriques qui régissent les déplacements d’un système solide dans l’espace, et de la variation des coordonnées provenant de ces déplacements considérés indépendamment des causes qui peuvent les produire.

[55] Xavier Pennec, Pierre Fillard, and Nicholas Ayache. A Riemannian Framework for Tensor Computing.

[56] Mevlana C. Gemici, Danilo Rezende, and Shakir Mohamed. Normalizing Flows on Riemannian Manifolds, November 2016. URL http://arxiv.org/abs/1611.02304. arXiv:1611.02304 [stat].

[57] Aaron Lou, Derek Lim, Isay Katsman, Leo Huang, Qingxuan Jiang, Ser-Nam Lim, and Christopher De Sa. Neural Manifold Ordinary Differential Equations, June 2020. URL http://arxiv.org/abs/2006.10254. arXiv:2006.10254 [stat].

[58] Emile Mathieu and Maximilian Nickel. Riemannian Continuous Normalizing Flows. arXiv.org, June 2020.

[59] Jason Yim, Brian L. Trippe, Valentin De Bortoli, Emile Mathieu, Arnaud Doucet, Regina Barzilay, and Tommi Jaakkola. SE(3) diffusion model with application to protein backbone generation, May 2023. URL http: //arxiv.org/abs/2302.02277. arXiv:2302.02277 [cs].

[60] Luca Falorsi and Patrick Forré. Neural Ordinary Differential Equations on Manifolds, June 2020. URL http: //arxiv.org/abs/2006.06663. arXiv:2006.06663 [stat.ML].

[61] E. Noether. Invariante Variationsprobleme. Nachrichten von der Gesellschaft der Wissenschaften zu Göttingen, Mathematisch-Physikalische Klasse, 1918:235–257, 1918. URL https://eudml.org/doc/59024.

[62] Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. Backpropagation Applied to Handwritten Zip Code Recognition. Neural Computation, 1(4):541–551, December 1989. ISSN 0899-7667, 1530-888X. doi: 10.1162/neco.1989.1.4.541. URL https://direct.mit.edu/neco/article/ 1/4/541-551/5515.

[63] Kristof T. Schütt, Oliver T. Unke, and Michael Gastegger. Equivariant message passing for the prediction of tensorial properties and molecular spectra, June 2021. URL http://arxiv.org/abs/2102.03150. arXiv:2102.03150 [physics].

[64] Avishek Joey Bose, Marcus Brubaker, and Ivan Kobyzev. Equivariant Finite Normalizing Flows, August 2022. URL http://arxiv.org/abs/2110.08649. arXiv:2110.08649 [cs].

[65] Maurice Weiler, Patrick Forré, Erik Verlinde, and Max Welling. Equivariant and Coordinate Independent Convolutional Networks. 2023. URL https://maurice-weiler.gitlab.io/cnn\_book/ EquivariantAndCoordinateIndependentCNNs.pdf.

[66] Liang Hou, Yuan Gao, Boyuan Jiang, Xin Tao, Qi Yan, Renjie Liao, Pengfei Wan, Di Zhang, and Kun Gai. Score Augmentation for Diffusion Models, August 2025. URL http://arxiv.org/abs/2508.07926. arXiv:2508.07926 [cs].

[67] Michelangelo Domina, Joseph William Abbott, Paolo Pegolo, Filippo Bigi, and Michele Ceriotti. How unconstrained machine-learning models learn physical symmetries, March 2026. URL http://arxiv.org/abs/ 2603.24638. arXiv:2603.24638 [cs].

[68] Feiran Li, Kent Fujiwara, Fumio Okura, and Yasuyuki Matsushita. A Closer Look at Rotation-invariant Deep Point Cloud Analysis. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 16198–16207, October 2021. doi: 10.1109/ICCV48922.2021.01591. URL https://ieeexplore.ieee.org/document/ 9710091. ISSN: 2380-7504.

[69] Cai Zhou, Zijie Chen, Zian Li, Jike Wang, Kaiyi Jiang, Pan Li, Rose Yu, Muhan Zhang, Stephen Bates, and Tommi Jaakkola. Rethinking Diffusion Models with Symmetries through Canonicalization with Applications to Molecular Graph Generation, February 2026. URL http://arxiv.org/abs/2602.15022. arXiv:2602.15022 [cs].

[70] Alona Levy-Jurgenson, Alvaro Prat, James Cuin, and Yee Whye Teh. Manifold Aware Denoising Score Matching (MAD), March 2026. URL http://arxiv.org/abs/2603.02452. arXiv:2603.02452 [cs].

[71] Jonas Köhler, Leon Klein, and Frank Noé. Equivariant Flows: Exact Likelihood Generative Learning for Symmetric Densities, October 2020. URL http://arxiv.org/abs/2006.02425. arXiv:2006.02425 [stat].

[72] Isay Katsman, Aaron Lou, Derek Lim, Qingxuan Jiang, Ser-Nam Lim, and Christopher De Sa. Equivariant Manifold Flows, January 2022. URL http://arxiv.org/abs/2107.08596. arXiv:2107.08596 [stat].

[73] Mario Geiger and Tess Smidt. e3nn: Euclidean Neural Networks, July 2022. URL http://arxiv.org/abs/ 2207.09453. arXiv:2207.09453 [cs].

[74] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-GRPO: Training Flow Matching Models via Online RL, October 2025. URL http: //arxiv.org/abs/2505.05470. arXiv:2505.05470 [cs.CV].

[75] Thomas Egg, Harry Winston Sullivan, Ellad B. Tadmor, and Stefano Martiniani. OMatG-flash: An all-atom flow map with reinforce adjoint matching for scalable materials discovery. arXiv preprint arXiv:2609.26402, 2026. doi: 10.48550/arXiv.2609.26402. URL https://arxiv.org/abs/2609.26402.

[76] Thibault de Surrel, Fabien Lotte, Sylvain Chevallier, and Florian Yger. Wrapped Gaussian on the manifold of Symmetric Positive Definite Matrices, February 2025. URL https://arxiv.org/abs/2502.01512v4.

[77] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=YCWjhGrJFD.

[78] Brandon M. Wood, Misko Dzamba, Xiang Fu, Meng Gao, Muhammed Shuaibi, Luis Barroso-Luque, Kareem Abdelmaqsoud, Vahe Gharakhanyan, John R. Kitchin, Daniel S. Levine, Kyle Michel, Anuroop Sriram, Taco Cohen, Abhishek Das, Ammar Rizvi, Sushree Jagriti Sahoo, Zachary W. Ulissi, and C. Lawrence Zitnick. UMA: A Family of Universal Models for Atoms, March 2026. URL http://arxiv.org/abs/2506.23971. arXiv:2506.23971 [cs].

[79] Mark Neumann, James Gin, Benjamin Rhodes, Steven Bennett, Zhiyi Li, Hitarth Choubisa, Arthur Hussey, and Jonathan Godwin. Orb: A Fast, Scalable Neural Network Potential, October 2024. URL http://arxiv.org/ abs/2410.22570. arXiv:2410.22570 [cond-mat.mtrl-sci] version: 1.

[80] Vahe Gharakhanyan, Luis Barroso-Luque, Yi Yang, Muhammed Shuaibi, Kyle Michel, Daniel S. Levine, Misko Dzamba, Xiang Fu, Meng Gao, Xingyu Liu, Haoran Ni, Keian Noori, Brandon M. Wood, Matt Uyttendaele, Arman Boromand, C. Lawrence Zitnick, Noa Marom, Zachary W. Ulissi, and Anuroop Sriram. Open Molecular Crystals 2025 (OMC25) Dataset and Models, August 2025. URL http://arxiv.org/abs/2508.02651. arXiv:2508.02651 [physics].

[81] C. R. Groom, I. J. Bruno, M. P. Lightfoot, and S. C. Ward. The Cambridge Structural Database. Acta Crystallographica Section B: Structural Science, Crystal Engineering and Materials, 72(2):171–179, April 2016. ISSN 2052-5206. doi: 10.1107/S2052520616003954. URL //journals.iucr.org/paper?bm5086. Publisher: International Union of Crystallography.

[82] Sam Motherwell and James Alexander Chisholm. COMPACK: A program for identifying crystal structure similarity using distances. Journal of Applied Crystallography, 38(1):228–231, 2005. ISSN 1600-5767. doi: 10.1107/S0021889804027074.

[83] Rajendra Bhatia. Positive Definite Matrices. Princeton University Press, January 2007. ISBN 978-0-691-12918-1.

[84] Emmanuel Chevallier and Nicolas Guigui. Wrapped statistical models on manifolds: Motivations, the case se(n), and generalization to symmetric spaces. In Frédéric Barbaresco and Frank Nielsen, editors, Geometric Structures of Statistical Physics, Information Geometry, and Learning, pages 96–106, Cham, 2021. Springer International Publishing. ISBN 978-3-030-77957-3.

[85] Gregory S. Chirikjian. Stochastic Models, Information Theory, and Lie Groups, Volume 2: Analytic Methods and Modern Applications. Applied and Numerical Harmonic Analysis. Birkhäuser, Boston, 2012. ISBN 978-0- 8176-4943-2 978-0-8176-4944-9. doi: 10.1007/978-0-8176-4944-9. URL https://link.springer.com/10. 1007/978-0-8176-4944-9.

[86] Joan Solà, Jeremie Deray, and Dinesh Atchuthan. A micro Lie theory for state estimation in robotics, December 2021. URL http://arxiv.org/abs/1812.01537. arXiv:1812.01537 [cs].

[87] Nicholas Gao and Stephan Günnemann. Ab-Initio Potential Energy Surfaces by Pairing GNNs with Neural Wave Functions, March 2022. URL http://arxiv.org/abs/2110.05064. arXiv:2110.05064 [cs].

[88] M. Arndt, V. Sorkin, and E. B. Tadmor. Efficient algorithms for discrete lattice calculations. Journal of Computational Physics, 228(13):4858–4880, July 2009. ISSN 0021-9991. doi: 10.1016/j.jcp.2009.03.039. URL https://www.sciencedirect.com/science/article/pii/S0021999109001703.

[89] Allan dos Santos Costa, Ilan Mitnikov, Franco Pellegrini, Ameya Daigavane, Mario Geiger, Zhonglin Cao, Karsten Kreis, Tess Smidt, Emine Kucukbenli, and Joseph Jacobson. EquiJump: Protein Dynamics Simulation via SO(3)-Equivariant Stochastic Interpolants, December 2024. URL http://arxiv.org/abs/2410.09667. arXiv:2410.09667 [cs].

[90] Yi-Lun Liao and Tess Smidt. Equiformer: Equivariant Graph Attention Transformer for 3D Atomistic Graphs, February 2023. URL http://arxiv.org/abs/2206.11990. arXiv:2206.11990 [cs].

[91] Ask Hjorth Larsen, Jens Jørgen Mortensen, Jakob Blomqvist, Ivano E Castelli, Rune Christensen, Marcin Dułak, Jesper Friis, Michael N Groves, Bjørk Hammer, Cory Hargus, Eric D Hermes, Paul C Jennings, Peter Bjerre Jensen, James Kermode, John R Kitchin, Esben Leonhard Kolsbjerg, Joseph Kubal, Kristen Kaasbjerg, Steen Lysgaard, Jón Bergmann Maronsson, Tristan Maxson, Thomas Olsen, Lars Pastewka, Andrew Peterson, Carsten Rostgaard, Jakob Schiøtz, Ole Schütt, Mikkel Strange, Kristian S Thygesen, Tejs Vegge, Lasse Vilhelmsen, Michael Walter, Zhenhua Zeng, and Karsten W Jacobsen. The atomic simulation environment—a Python library for working with atoms. Journal ofPhysics: Condensed Matter, 29(27):273002, jun 2017. doi: 10.1088/1361-648X/aa680e. URL https://doi.org/10.1088/1361-648X/aa680e.

A The Coarse Graining Map 17   
B Mathematical Specification of a Molecular Crystal 17   
C Mathematical Definition of a Molecular Crystal Manifold 18   
D Distances, Logarithms, and Exponentials on the Molecular Crystal Manifold 19   
E Insufficiency of a Euclidean Unit Cell 20   
F Reinforcement Learning on The Molecular Crystal Manifold 20   
G Symmetries of Molecular Crystals 22   
H Neural Network Architecture 27   
I UMA Relaxation Scheme 31   
J Hyperparameters 32   
K Additional Results 35   
L Loss Curves 40   
M Data Availability, Preprocessing, and Resources 41

## A The Coarse Graining Map

Coarse Graining Letting an index set $S _ { i }$ specify a subset of atoms constituting a molecule, we reduce the naive degrees of freedom from $\mathbb { R } ^ { \breve { 3 } \times N _ { i } }$ to a coarse-grained descriptor. On each disjoint subset of atoms, the ‘coarse graining’ (CG) map is taken to act equivariantly, producing a pair consisting of the molecular centroid $q ^ { ( i ) } \in \mathbb { R } ^ { 3 }$ and orientation $Q ^ { ( i ) } \in S O ( 3 )$ . Following Kim et al. [21], we define the CG mapping in the following way

Definition A.1. (Coarse Graining Map)

The coarse graining map C is defined as

$$
\begin{array} { r l } & { \mathcal { C } : \mathbb { R } ^ { 3 \times N _ { i } } \to \mathbb { R } ^ { 3 } \times S O ( 3 ) , } \\ & { \{ c ^ { ( j ) } \} _ { j \in S _ { i } } \mapsto ( q ^ { ( i ) } , Q ^ { ( i ) } ) , } \end{array}
$$

where

$$
q ^ { ( i ) } = \frac { 1 } { N _ { i } } \sum _ { j \in S _ { i } } c ^ { ( j ) } , \qquad Q ^ { ( i ) } = \pi \left( \{ c ^ { ( j ) } \} _ { j \in S _ { i } } \right) ,
$$

such that, writing $C = [ c ^ { ( 1 ) } | \cdot \cdot \cdot | c ^ { ( N _ { i } ) } ] \in \mathbb { R } ^ { 3 \times N _ { i } }$ and

$$
\left[ e _ { 1 } ( C ) \mid e _ { 2 } ( C ) \mid e _ { 3 } ( C ) \right] : = \mathrm { E i g } \left( \frac { 1 } { N _ { i } } \sum _ { n = 1 } ^ { N _ { i } } \left( c ^ { ( n ) } - q ^ { ( i ) } \right) \left( c ^ { ( n ) } - q ^ { ( i ) } \right) ^ { \top } \right) ,
$$

where $\mathrm { E i g ( \cdot ) }$ returns the orthonormal eigenbasis ordered by decreasing eigenvalues, the orientation map is given by

$$
\begin{array} { r } { \mathcal { R } ( C ) : = [ \tilde { e } _ { 1 } ( C ) \mid \tilde { e } _ { 2 } ( C ) \mid \tilde { e } _ { 1 } ( C ) \times \tilde { e } _ { 2 } ( C ) ] , \qquad \tilde { e } _ { k } ( C ) : = \mathrm { s g n } \left( v ( C ) ^ { \top } e _ { k } ( C ) \right) e _ { k } ( C ) , } \end{array}
$$

where $v : \mathbb { R } ^ { 3 \times N _ { i } }  \mathbb { R } ^ { 3 }$ is an auxiliary $S O ( 3 )$ -equivariant vector function. Note that this definition is valid only for point clouds in $\mathbb { R } ^ { 3 \times N _ { i } }$ that do not admit a stabilizer under any type-preserving orientation, i.e., a non-trivial point group. See Section G for our resolution.

The mapping R is (nearly) the principal axis of the molecule; Appendix G discusses the associated frame ambiguities. The ‘canonical’ or ‘local’ coordinates $\tilde { c } ^ { ( j ) }$ of atoms in molecule i are then obviously given by

$$
\tilde { c } ^ { ( j ) } = ( Q ^ { ( i ) } ) ^ { \top } ( c ^ { ( j ) } - q ^ { ( i ) } )\tag{16}
$$

which shifts the molecule to the origin (so its centroid is identically zero) and undoes the rotation $Q ^ { ( i ) }$ so the principal components are ordered along the conventional $x , y , z$ axes.

## B Mathematical Specification of a Molecular Crystal

Crystals and CSP We first define a crystal structure mathematically. This is a key step as we will use this as a foundation to rigorously define a molecular crystal.

Definition B.1. (Crystal)

A crystal y is a tuple

$$
y : = \left( L , \{ c ^ { ( j ) } \} _ { j = 1 } ^ { N } , \{ a ^ { ( j ) } \} _ { j = 1 } ^ { N } \right) ,
$$

where N is the total number of atoms in the unit cell, $L \in G L ^ { + } ( 3 , \mathbb { R } )$ is a row-major matrix specifying the lattice vectors, $\boldsymbol { c } ^ { ( j ) } \in \mathbb { R } ^ { 3 }$ is the Cartesian position of atom $j ,$ , and $a ^ { ( j ) } \in \{ 0 , 1 \} ^ { T }$ is a one-hot vector over $T$ possible atomic types (e.g., carbon, hydrogen, chlorine). For brevity, we write

$$
\boldsymbol { \mathcal { A } } : = \{ a ^ { ( j ) } \} _ { j = 1 } ^ { N } .
$$

Molecular Crystals For molecular crystals we partition the atoms into M disjoint subsets, each constituting a “molecular conformer” or “molecule” with $N _ { i }$ atoms. Letting $S _ { i }$ denote the set of atoms in molecule $i ,$ we have the

restrictions

$$
\{ c ^ { ( j ) } \} _ { j = 1 } ^ { N } = \bigcup _ { i = 1 } ^ { M } \{ c ^ { ( j ) } \} _ { j \in S _ { i } } , \qquad \{ a ^ { ( j ) } \} _ { j = 1 } ^ { N } = \bigcup _ { i = 1 } ^ { M } \{ a ^ { ( j ) } \} _ { j \in S _ { i } } , \qquad \sum _ { i = 1 } ^ { M } N _ { i } = N ,\tag{17}
$$

where F is the disjoint union. The first two conditions ensure that the subsets reconstruct the entire crystal; the last condition ensures that all atoms appear once. The subsets must be disjoint so that no atom belongs to two conformers. It is then natural to define an $\hat { M } \cdot$ -molecule coarse-grained crystal as the image of an N-atom crystal y under the coarse-graining map, where each set $S _ { i } \in \{ S _ { i } \} _ { i = 1 } ^ { M }$ is taken to be a connected component of the geometric connectivity graph

$$
\mathcal { G } : = ( \mathcal { V } , \mathcal { E } ) = \big ( \{ c ^ { ( j ) } \} _ { j = 1 } ^ { N } , \mathcal { E } \big ) ,\tag{18}
$$

with V as the set of Cartesian atomic positions defining the nodes and $\mathcal { E }$ as the set of edges encoding chemical bonds.   
The set of connected components may be denoted $\mathfrak { S } _ { m } : = \{ S _ { i } ^ { ( m ) } \} _ { i = 1 } ^ { M }$ <sub>1</sub>.

## Definition B.2. (Coarse Grained Crystal)

A coarse-grained crystal x is a tuple

$$
x : = \big ( L , \{ q ^ { ( i ) } , Q ^ { ( i ) } \} _ { i = 1 } ^ { M } \big ) ,
$$

where M is the number of molecules in the unit cell, $L \in G L ^ { + } ( 3 , \mathbb { R } )$ is a row-major matrix whose rows are the lattice vectors, $q ^ { ( i ) } \in \mathbb { R } ^ { 3 }$ is the Cartesian position of molecule $i ,$ and $Q ^ { ( i ) } \in S O ( 3 )$ is its orientation.

The number of degrees of freedom in a coarse-grained crystal is typically smaller than that of the corresponding atomistic crystal. To retain invertibility of the coarse graining map, it is useful to augment this representation with the local coordinates and atomic species. Accordingly, define

$$
\tilde { x } : = \big ( L , \{ q ^ { ( i ) } , Q ^ { ( i ) } \} _ { i = 1 } ^ { M } , C , \mathcal { A } \big ) ,
$$

where

$$
C : = \big ( \{ \tilde { c } ^ { ( j ) } \} _ { j \in S _ { 1 } } , \dots , \{ \tilde { c } ^ { ( j ) } \} _ { j \in S _ { M } } \big ) ,
$$

and where A denotes the corresponding atomic species.

## C Mathematical Definition of a Molecular Crystal Manifold

## Definition C.1. (Molecular Crystal Manifold)

The molecular crystal manifold M is the (6M + 9)-dimensional product manifold

$$
\mathcal { M } : = S O ( 3 ) \times \mathrm { S y m } _ { 3 } ^ { + } \times \left( \mathbb { T } ^ { 3 } \times S O ( 3 ) \right) ^ { M } ,
$$

endowed with the Riemannian metric $g _ { x } ^ { \mathcal { M } } : T _ { x } \mathcal { M } \times T _ { x } \mathcal { M } $ R which is additive

$$
g _ { x } ( v _ { 1 } , v _ { 2 } ) : = g _ { U } ^ { S O ( 3 ) } ( \mathcal { U } _ { 1 } , \mathcal { U } _ { 2 } ) + g _ { P } ^ { \mathrm { S y m } _ { 3 } ^ { + } } ( \mathcal { P } _ { 1 } , \mathcal { P } _ { 2 } ) + \sum _ { i = 1 } ^ { M } \left( g _ { f ^ { ( i ) } } ^ { \sf T ^ { 3 } } ( \mathcal { F } _ { 1 } ^ { ( i ) } , \mathcal { F } _ { 2 } ^ { ( i ) } ) + g _ { Q ^ { ( i ) } } ^ { S O ( 3 ) } ( \mathcal { Q } _ { 1 } ^ { ( i ) } , \mathcal { Q } _ { 2 } ^ { ( i ) } ) \right) ,
$$

where the metric on each sub-manifold is given by

$$
\begin{array} { r l } & { g _ { U } ^ { S O ( 3 ) } ( \mathcal { U } _ { 1 } , \mathcal { U } _ { 2 } ) : = \frac { 1 } { 2 } \operatorname { T r } ( \mathcal { U } _ { 1 } ^ { \top } \mathcal { U } _ { 2 } ) , \qquad g _ { P } ^ { \mathrm { S y m } _ { 3 } ^ { + } } ( \mathcal { P } _ { 1 } , \mathcal { P } _ { 2 } ) : = \operatorname { T r } ( P ^ { - 1 / 2 } \mathcal { P } _ { 1 } P ^ { - 1 } \mathcal { P } _ { 2 } P ^ { - 1 / 2 } ) , } \\ & { \quad g _ { f } ^ { \mathbb { T } ^ { 3 } } ( \mathcal { F } _ { 1 } , \mathcal { F } _ { 2 } ) : = \mathcal { F } _ { 1 } ^ { \top } \mathcal { F } _ { 2 } , \qquad g _ { Q } ^ { S O ( 3 ) } ( \mathcal { Q } _ { 1 } , \mathcal { Q } _ { 2 } ) : = \frac { 1 } { 2 } \operatorname { T r } ( \mathcal { Q } _ { 1 } ^ { \top } \mathcal { Q } _ { 2 } ) . } \end{array}
$$

Here $v _ { 1 } , v _ { 2 } \in T _ { x } { \mathcal { M } }$ are arbitrary tangent vectors at $x ,$ written as

$$
\boldsymbol { v } _ { k } = \big ( \mathcal { U } _ { k } , \mathcal { P } _ { k } , \mathcal { F } _ { k } ^ { ( 1 ) } , \ldots , \mathcal { F } _ { k } ^ { ( M ) } , \mathcal { Q } _ { k } ^ { ( 1 ) } , \ldots , \mathcal { Q } _ { k } ^ { ( M ) } \big ) ,
$$

where

$$
\mathcal { U } _ { k } \in T _ { U } S O ( 3 ) , \qquad \mathcal { P } _ { k } \in T _ { P } \mathrm { S y m } _ { 3 } ^ { + } , \qquad Q _ { k } ^ { ( i ) } \in T _ { Q ^ { ( i ) } } S O ( 3 ) , \qquad \mathcal { F } _ { k } ^ { ( i ) } \in T _ { f ^ { ( i ) } } \mathbb { T } ^ { 3 } .
$$

The metric on $S O ( 3 )$ is bi-invariant; the metric on $\mathrm { S y m _ { 3 } ^ { + } }$ is affine-invariant. Both admit closed-form geodesics — Rodrigues’ formula [54] on $S O ( 3 )$ and Pennec’s formula [55] on $\mathrm { S y m _ { 3 } ^ { + } - }$ and the metric on $\mathbb { T } ^ { 3 }$ follows FlowMM [30]. A cartoon of the molecular crystal manifold is shown in figure 3.

![](images/d287f017eff2b1efcba6ae8dfc9300646847a242267ea08f53b66241c6e5edc9.jpg)  
Figure 3: A cartoon of the molecular crystal manifold $\mathcal { M } .$ . The blue curve represents a geodesic $x _ { t } = \exp _ { x _ { 0 } } ( t \log _ { x _ { 0 } } { ( x _ { 1 } ) } )$ connecting an initially random rigid body positions $x _ { 0 }$ to an optimal set of positions $x _ { 1 }$ . The initial velocity v is an element of the tangent space $T _ { x _ { 0 } } \mathcal { M }$ at the point $x _ { 0 }$ . The abstract blobs are rigid bodies with PCA frames attached to them which represent molecules being crystallized.

## D Distances, Logarithms, and Exponentials on the Molecular Crystal Manifold

Distances The distance $d _ { \mathcal { M } } : \mathcal { M } \times \mathcal { M } \to \mathbb { R } _ { \ge 0 }$ between molecular crystal configurations is additive in its square over the product manifold like

$$
d ^ { M } ( x _ { 1 } , x _ { 2 } ) ^ { 2 } = d ^ { \mathrm { S y m } _ { 3 } ^ { + } } ( P _ { 1 } , P _ { 2 } ) ^ { 2 } + d ^ { S O ( 3 ) } ( U _ { 1 } , U _ { 2 } ) ^ { 2 } + \sum _ { i = 1 } ^ { M } \Big ( d ^ { \pi ^ { 3 } } ( f _ { 1 } ^ { ( i ) } , f _ { 2 } ^ { ( i ) } ) ^ { 2 } + d ^ { S O ( 3 ) } ( Q _ { 1 } ^ { ( i ) } , Q _ { 2 } ^ { ( i ) } ) ^ { 2 } \Big )\tag{19}
$$

where each distance function is given by

$$
\begin{array} { r } { d ^ { S O ( 3 ) } ( Q _ { 1 } , Q _ { 2 } ) = \operatorname { a r c c o s } \big ( \frac { 1 } { 2 } \left( \mathrm { T r } ( Q _ { 1 } ^ { \top } Q _ { 2 } ) - 1 \right) \big ) , \qquad d ^ { \mathbb { T } ^ { 3 } } ( f _ { 1 } , f _ { 2 } ) = \underset { k \in \mathbb Z ^ { 3 } } { \operatorname* { m i n } } \| f _ { 1 } - f _ { 2 } + k \| , } \end{array}\tag{20}
$$

$$
d ^ { \mathrm { S y m } _ { 3 } ^ { + } } ( P _ { 1 } , P _ { 2 } ) = \mathrm { T r } \Big ( \log \big ( P _ { 1 } ^ { - 1 / 2 } P _ { 2 } P _ { 1 } ^ { - 1 / 2 } \big ) \ \log \big ( P _ { 1 } ^ { - 1 / 2 } P _ { 2 } P _ { 1 } ^ { - 1 / 2 } \big ) \Big ) ^ { 1 / 2 } .\tag{21}
$$

These correspond respectively to the minimum-image fractional distance, the relative angle, and the size of the multiplicative deformation taking one positive-definite form to another. Here log denotes the standard matrix logarithm, defined by spectral decomposition log $P = \Lambda$ diag(log $\lambda _ { 1 } , . . . , \log \lambda _ { n } ^ { \intercal }$

Logarithms The Riemannian logarithm $\log _ { x } ^ { \mathcal { M } } : \mathcal { O } \to T _ { x } \mathcal { M }$ is a componentwise map from an open set $\mathcal { O }$ containing base and target points $x _ { 1 }$ and $x _ { 2 }$ to the tangent space at the base point. It is given by

$$
\begin{array} { r } { \log _ { x _ { 1 } } ^ { \mathcal { M } } ( x _ { 2 } ) = \left( \log _ { U _ { 1 } } ^ { S O ( 3 ) } ( U _ { 2 } ) , \log _ { P _ { 1 } } ^ { \mathrm { S y m } _ { 3 } ^ { + } } ( P _ { 2 } ) , \{ \log _ { f _ { 1 } ^ { ( i ) } } ^ { \mathbb { T } ^ { 3 } } ( f _ { 2 } ^ { ( i ) } ) , \log _ { Q _ { 1 } ^ { ( i ) } } ^ { S O ( 3 ) } ( Q _ { 2 } ^ { ( i ) } ) \} _ { i = 1 } ^ { M } \right) . } \end{array}\tag{22}
$$

where

$$
\log _ { Q _ { 1 } } ^ { S O ( 3 ) } ( Q _ { 2 } ) = Q _ { 1 } \left( \frac { d ^ { S O ( 3 ) } ( Q _ { 1 } , Q _ { 2 } ) } { 2 \sin d ^ { S O ( 3 ) } ( Q _ { 1 } , Q _ { 2 } ) } \left( Q _ { 1 } ^ { \top } Q _ { 2 } - Q _ { 2 } ^ { \top } Q _ { 1 } \right) \right) , \qquad \log f _ { 1 } ^ { ^ { 3 } } ( f _ { 2 } ) = f _ { 2 } - f _ { 1 } - n ^ { \star } ,\tag{23}
$$

$$
\log _ { P _ { 1 } } ^ { \mathrm { S y m _ { 3 } ^ { + } } } ( P _ { 2 } ) = P _ { 1 } ^ { \frac { 1 } { 2 } } \log \bigl ( P _ { 1 } ^ { - \frac { 1 } { 2 } } P _ { 2 } P _ { 1 } ^ { - \frac { 1 } { 2 } } \bigr ) P _ { 1 } ^ { \frac { 1 } { 2 } } ,\tag{24}
$$

The value $n ^ { \star } \in \arg$ min $\mathfrak { l } _ { n \in \mathbb { Z } ^ { 3 } } \| ( f _ { 2 } - f _ { 1 } ) - n \|$ , which implies logarithm on the torus always points from $f _ { 1 }$ to the nearest periodic image of $f _ { 2 } ,$ , meaning it can be multivalued when they are half a period apart. Similarly, the logarithm on $S O ( \mathrm { { 3 } ) }$ is multivalued near relative angles π.

Exponentials The Riemannian exponential $\exp _ { x } ^ { \mathcal { M } } : T _ { x } \mathcal { M } \longrightarrow \mathcal { M }$ gives the result of following the geodesic defined by the tangent vector for one unit of time. This is given by

$$
\begin{array} { r } { \exp _ { x } ^ { \mathcal { M } } ( v ) = \left( \exp _ { U } ^ { S O ( 3 ) } ( \mathcal { U } ) , \exp _ { P } ^ { \mathrm { { S y m } _ { 3 } ^ { + } } } ( \mathcal { P } ) , \{ \exp _ { f ^ { ( i ) } } ^ { \mathbb { T } ^ { 3 } } ( \mathcal { F } ^ { ( i ) } ) , \exp _ { Q ^ { ( i ) } } ^ { S O ( 3 ) } ( \mathcal { Q } ^ { ( i ) } ) \} _ { i = 1 } ^ { M } \right) , } \end{array}\tag{25}
$$

where on each submanifold we have

$$
\begin{array} { r } { \exp _ { Q } ^ { S O ( 3 ) } ( \mathcal { Q } ) = Q \left( I + \frac { \sin \theta } { \theta } ( Q ^ { \top } \mathcal { Q } ) + 2 \frac { \sin ^ { 2 } ( \theta / 2 ) } { \theta ^ { 2 } } ( Q ^ { \top } \mathcal { Q } ) ^ { 2 } \right) , \qquad \theta = \sqrt { - \frac { 1 } { 2 } \operatorname { T r } ( ( Q ^ { \top } \mathcal { Q } ) ^ { 2 } ) } , } \end{array}\tag{26}
$$

$$
\begin{array} { r } { \exp _ { P } ^ { \mathrm { S y m } _ { 3 } ^ { + } } ( \mathcal { P } ) = P ^ { \frac { 1 } { 2 } } \exp \bigl ( P ^ { - \frac { 1 } { 2 } } \mathcal { P } P ^ { - \frac { 1 } { 2 } } \bigr ) P ^ { \frac { 1 } { 2 } } , \qquad \exp _ { f } ^ { \mathbb { T } ^ { 3 } } ( \mathcal { F } ) = f + \mathcal { F } \bmod \mathbb { Z } ^ { 3 } . } \end{array}\tag{27}
$$

Jacobians The Jacobian determinant of $\exp _ { P } ^ { \mathrm { S y m _ { 3 } ^ { + } } }$ is [83, 76, 84]

$$
J _ { P } ^ { \mathrm { S y m } _ { 3 } ^ { + } } ( \mathcal { P } ) = 2 ^ { d ( d - 1 ) / 2 } \prod _ { i < j } \frac { \sinh \bigl ( \frac { s _ { i } - s _ { j } } { 2 } \bigr ) } { s _ { i } - s _ { j } } , \qquad s _ { i } = \lambda _ { i } \Bigl ( P ^ { - \frac { 1 } { 2 } } \mathcal { P } P ^ { - \frac { 1 } { 2 } } \Bigr ) .\tag{28}
$$

The Jacobian determinant of $\exp _ { Q } ^ { S O ( 3 ) }$ is [85, 86]

$$
J _ { Q } ^ { S O ( 3 ) } ( Q ) = \frac { 2 ( 1 - \cos \theta ) } { \theta ^ { 2 } } , \qquad \theta = \sqrt { - \textstyle { \frac { 1 } { 2 } } \mathrm { T r } ( ( Q ^ { \top } Q ) ^ { 2 } ) } .\tag{29}
$$

On $\mathbb { T } ^ { 3 } , J _ { f } ^ { \mathbb { T } ^ { 3 } } ( \mathcal { F } ) = 1$ as it is flat.

## E Insufficiency of a Euclidean Unit Cell

Euclidean Description. We briefly comment on the possibility for a flow prescribed by interpolation on $\mathbb { R } ^ { 3 \times 3 }$ to leave the manifold ${ \overline { { G L ^ { + } } } } ( 3 , \mathbb { R } )$ . Consider the two cell matrices

$$
L _ { 0 } = \left[ { \begin{array} { c c c } { 1 } & { 0 } & { 0 } \\ { 0 } & { 1 } & { 0 } \\ { 0 } & { 0 } & { 1 } \end{array} } \right] , \quad L _ { 1 } = \left[ { \begin{array} { c c c } { - 1 } & { 0 } & { 0 } \\ { 0 } & { - 2 } & { 0 } \\ { 0 } & { 0 } & { 3 } \end{array} } \right] .\tag{30}
$$

In either case the determinant is positive: det $L _ { 0 } = 1$ , det $L _ { 1 } = 6$ , so both are valid endpoints for our interpolation. Now consider their linear interpolation:

$$
\begin{array}{c} L _ { t } = ( 1 - t ) L _ { 0 } + t L _ { 1 } \implies L _ { t = 0 . 4 } = { \binom { 0 . 2 \quad \quad 0 \quad 0 } { 0 } } \ { - 0 . 2 \quad \quad 0 } \\ { 0 \quad \quad 0 \quad 1 . 8 \} } \end{array} .\tag{31}
$$

This yields det $L _ { t = 0 . 4 } = - 0 . 0 7 2 < 0$ , meaning the interpolant has left $G L ^ { + } ( 3 , \mathbb { R } )$ and no longer defines a valid unit cell. Naive linear interpolation in $\mathbb { R } ^ { 3 \times 3 }$ does not respect the topology of the constraint set. One alternative is to canonicalize all cell matrices into a standard orientation before interpolation, e.g. by extracting lattice parameters and reconstructing a lower-triangular cell in a fixed frame as in Crystalite [34] or applying a Niggli reduction to all the data during preprocessing.

Our Solution. We decompose the cell via $L = U P$ into a rotation $U \in S O ( 3 )$ and a symmetric positive-definite stretch $P \in \mathrm { S y m _ { 3 } ^ { + } }$ , and interpolate each factor along geodesics of its intrinsic Riemannian metric. Since both $S O ( 3 )$ and $\mathrm { S y m _ { 3 } ^ { + } }$ are geodesically complete, the interpolant remains on the manifold for all $t \in [ 0 , 1 ]$ by construction. Every intermediate point is a valid rotation composed with a valid positive-definite stretch, and hence a valid element of $G L ^ { + } ( 3 , \mathbb { R } )$ .

## F Reinforcement Learning on The Molecular Crystal Manifold

This appendix expands the manifold-RL construction summarized in Section 3.4

Markov Decision Process Following Höllmer and Martiniani [18], we use reinforcement learning (RL) as a posttraining fine-tuning step to improve transferability and to bias generation toward energetically stable crystal structures. Our setting extends their construction from Euclidean generative dynamics to the manifold-valued dynamics of molecular crystals.

First, we cast the time-discretized dynamics on M as a Markov decision process

$$
( x , y , \mu _ { 0 } , K , R ) ,\tag{32}
$$

with state space $\mathcal { X } = [ 0 , 1 ] \times \mathcal { M }$ , action space Y, initial-state distribution $\mu _ { 0 } = ( \delta _ { 0 } , p _ { 0 } )$ , transition kernel K, and reward function R. At time t, the state is $s _ { t } = ( t , x _ { t } ) \in \mathcal { X }$ and the initial state is drawn as $s _ { 0 } \sim \mu _ { 0 }$ , so every trajectory begins at $t = 0$ from a prior sample $x _ { 0 } \sim p _ { 0 }$ . The agent samples an action from the stochastic policy

$$
\pi ^ { \theta } ( a _ { t } \mid s _ { t } ) : = \pi ^ { \theta } ( x _ { t + \Delta t } \mid x _ { t } ) ,\tag{33}
$$

which we identify with the next configuration, $a _ { t } : = x _ { t + \Delta t }$ . The transition kernel is deterministic given the action,

$$
K ( s _ { t + \Delta t } \mid s _ { t } , a _ { t } ) = \delta _ { ( t + \Delta t , a _ { t } ) } ,\tag{34}
$$

so all stochasticity in the trajectory comes from the policy itself. We take the reward to be terminal-only,

$$
R ( s _ { t } , a _ { t } ) : = { \left\{ \begin{array} { l l } { r ( x _ { 1 } ) } & { { \mathrm { i f ~ } } t = 1 , } \\ { 0 } & { { \mathrm { o t h e r w i s e } } , } \end{array} \right. }\tag{35}
$$

and choose $r ( x _ { 1 } ) = - E ( x _ { 1 } )$ , where E is the all-atom energy of the final crystal computed using UMA [78]. This energy calculation includes periodic boundary conditions from the predicted unit cell L.

Promotion of the ODE to an SDE In the base formulation, the learned generative dynamics define a deterministic ODE, so the induced policy is likewise deterministic. This is undesirable for RL, where some degree of stochasticity is needed for exploration. Höllmer and Martiniani [18] address this in the flat setting by augmenting the dynamics with controlled noise. Here we extend that idea to the curved manifold M by introducing noise directly in the tangent space $T _ { x _ { t } } { \mathcal { M } }$

Given a trained velocity field $b _ { t } ^ { \theta } .$ , deterministic integration is performed by a Riemannian Euler step with step size $\Delta t$ using the exponential map as the local chart:

$$
x _ { t + \Delta t } = \exp _ { x _ { t } } \left( \Delta t b _ { t } ^ { \theta } ( x _ { t } ) \right) .\tag{36}
$$

To obtain a stochastic policy, we instead add isotropic Gaussian noise in the tangent space before mapping the update back to the manifold,

$$
\begin{array} { r } { x _ { t + \Delta t } = \exp _ { x _ { t } } \left( \Delta t b _ { t } ^ { \theta } ( x _ { t } ) + \sigma _ { t } \sqrt { \Delta t } \xi _ { t } \right) , \qquad \xi _ { t } \sim \mathcal { N } ( 0 , I _ { \dim } \mathcal { M } ) . } \end{array}\tag{37}
$$

Equivalently, if we denote the tangent-space increment by $w _ { t } : = \Delta t b _ { t } ^ { \theta } ( x _ { t } ) + \sigma _ { t } \sqrt { \Delta t } \xi _ { t } \in T _ { x _ { t } } \mathcal { M }$ , then $w _ { t }$ is Gaussian in $T _ { x { _ t } } { \mathcal { M } }$ with mean $\Delta t b _ { t } ^ { \theta } ( x _ { t } )$ and covariance $\sigma _ { t } ^ { 2 } \Delta t I _ { \mathrm { d i m } , M } .$ , and the next state is obtained by the pushforward $\begin{array} { r } { x _ { t + \Delta t } = \exp _ { x _ { t } } ^ { \mathcal { M } } ( w _ { t } ) } \end{array}$ . The resulting policy is therefore a wrapped Gaussian on $\mathcal { M } \left[ 7 6 \right]$

Using the inverse map $w _ { t } = \log _ { x _ { t } } ^ { \mathcal { M } } ( x _ { t + \Delta t } )$ , the usual change-of-variables formula first gives

$$
\pi ^ { \theta } ( x _ { t + \Delta t } \mid x _ { t } ) = \mathcal { N } \left( \log _ { x _ { t } } ^ { \mathcal { M } } ( x _ { t + \Delta t } ) \Big \vert \Delta t b _ { t } ^ { \theta } ( x _ { t } ) , \sigma _ { t } ^ { 2 } \Delta t I _ { \mathrm { d i m } } \mathcal { M } \right) \left. \operatorname* { d e t } d \log _ { x _ { t } } ^ { \mathcal { M } } ( x _ { t + \Delta t } ) \right. .\tag{38}
$$

Since $d \log _ { x _ { t } } ^ { \mathcal { M } }$ is the inverse of $d \exp _ { x _ { t } } ^ { \mathcal { M } }$ , this may equivalently be written as

$$
\pi ^ { \theta } ( x _ { t + \Delta t } \mid x _ { t } ) = \frac { \mathcal { N } \left( \log _ { x _ { t } } ^ { \mathcal { M } } ( x _ { t + \Delta t } ) \Big \vert \Delta t b _ { t } ^ { \theta } ( x _ { t } ) , \sigma _ { t } ^ { 2 } \Delta t I _ { \dim \mathcal { M } } \right) } { J _ { x _ { t } } ^ { \mathcal { M } } \left( \log _ { x _ { t } } ^ { \mathcal { M } } ( x _ { t + \Delta t } ) \right) } ,\tag{39}
$$

where $J _ { x _ { t } } ^ { \mathcal { M } } ( w ) = | \operatorname* { d e t } d \exp _ { x _ { t } } ^ { \mathcal { M } } ( w ) |$ is the Jacobian of the exponential map at base point $x _ { t }$ . Taking logarithms yields

$$
\begin{array} { r l r } { \log \pi ^ { \theta } ( x _ { t + \Delta t } \mid x _ { t } ) = - \frac { \dim \mathcal { M } } { 2 } \log \left( 2 \pi \sigma _ { t } ^ { 2 } \Delta t \right) - \frac { \left\| \log _ { x _ { t } } ^ { \mathcal { M } } ( x _ { t + \Delta t } ) - \Delta t b _ { t } ^ { \theta } ( x _ { t } ) \right\| ^ { 2 } } { 2 \sigma _ { t } ^ { 2 } \Delta t } } & { } & \\ { \qquad - \log J _ { x _ { t } } ^ { \mathcal { M } } \left( \log _ { x _ { t } } ^ { \mathcal { M } } ( x _ { t + \Delta t } ) \right) . } & { } & \end{array}\tag{40}
$$

Because M is a product manifold, this Jacobians determinant factorizes over its component manifolds. In particular, the torus contribution is trivially the identity as it is flat, while the nontrivial geometric corrections come from the $S O ( 3 )$ and $\mathrm { S y m _ { 3 } ^ { + } }$ factors. These factors are provided in Appendix D.

Reinforcement Learning Objective Policy-gradient RL aims to maximize the expected terminal reward under trajectories τ generated by the policy,

$$
\mathcal { I } [ \pi ^ { \theta } ] = \mathbb { E } _ { \tau \sim \pi ^ { \theta } } \left[ r ( x _ { 1 } ) \right] .\tag{41}
$$

As discussed in the main text, in practice, GRPO samples G trajectories $\tau ^ { 1 : G }$ from the stochastic policy under identical conditioning and maximizes the following clipped surrogate objective [17]:

$$
\mathcal { L } _ { \mathrm { G R P O } } ( \theta ) = \frac { 1 } { S G K } \mathbb { E } _ { \tau ^ { 1 : G } \sim \pi ^ { \theta _ { \mathrm { o l d } } } } \left[ \sum _ { i = 1 } ^ { G } \sum _ { k = 0 } ^ { K - 1 } \operatorname* { m i n } \left( \rho _ { i , k } ( \theta ) \hat { A } _ { i } , \mathrm { c l i p } \left( \rho _ { i , k } ( \theta ) , 1 - \varepsilon , 1 + \varepsilon \right) \hat { A } _ { i } \right) \right] .\tag{42}
$$

Here, K is the number of integration time steps, $S$ is an optional normalization factor that accounts for different system sizes across different GRPO groups [18], and ε is a clipping hyperparameter. For $G$ sampled trajectories with rewards $\{ r _ { i } \} _ { i = 1 } ^ { G } = \{ r ( x _ { 1 } ^ { i } ) \} _ { i = 1 } ^ { G }$ , the group-relative advantages are given by

$$
\hat { A } _ { i } = \frac { r _ { i } - \operatorname* { m e a n } ( \{ r _ { j } \} _ { j = 1 } ^ { G } ) } { \mathrm { s t d } ( \{ r _ { j } \} _ { j = 1 } ^ { G } ) } .\tag{43}
$$

Since the reward is terminal-only, this same normalized score is used across all steps of trajectory $\ v x _ { t } ^ { i }$ in the clipped surrogate objective. The one-step likelihood ratio between the updated policy $\pi ^ { \theta }$ and the old policy $\pi ^ { \theta _ { \mathrm { o l d } } }$ that generated the trajectories is given by

$$
\rho _ { i , k } ( \theta ) : = \frac { \pi ^ { \theta } ( x _ { t _ { k + 1 } } ^ { i } \mid x _ { t _ { k } } ^ { i } ) } { \pi ^ { \theta _ { \mathrm { o u d } } } ( x _ { t _ { k + 1 } } ^ { i } \mid x _ { t _ { k } } ^ { i } ) } = \frac { \mathcal { N } \left( \log _ { x _ { t _ { k } } ^ { i } } ^ { \mathcal { M } } ( x _ { t _ { k + 1 } } ^ { i } ) \left. \Delta t b _ { t } ^ { \theta } ( x _ { t _ { k } } ^ { i } ) , \sigma _ { t } ^ { 2 } \Delta t I _ { \mathrm { d i m } , \mathcal { M } } \right. \right) } { \mathcal { N } \left( \log _ { x _ { t _ { k } } ^ { i } } ^ { \mathcal { M } } ( x _ { t _ { k + 1 } } ^ { i } ) \left. \Delta t b _ { t } ^ { \theta _ { \mathrm { o u d } } } ( x _ { t _ { k } } ^ { i } ) , \sigma _ { t } ^ { 2 } \Delta t I _ { \mathrm { d i m } , \mathcal { M } } \right. \right) } .\tag{44}
$$

Here, the Jacobian factor from the wrapped Gaussian policy cancels exactly, since it depends only on the manifold geometry and not on θ due to the tangent covariance of the policy being fixed.

Kullback–Leibler Regularization To prevent the updated policy from drifting too far from the reference policy $\pi ^ { \theta _ { \mathrm { r e f } } }$ of the pretrained model, we additionally include a KL regularization term penalizing large deviations of the updated drift $b _ { t } ^ { \bar { \theta } }$ from the reference at each step. Writing the KL contribution for a single rollout trajectory, we add the following term to the maximized objective:

$$
\mathcal { L } _ { \mathrm { K L } } ( \theta ) = - \beta \mathbb { E } _ { \tau \sim \pi ^ { \theta _ { \mathrm { o l d } } } } \left[ \sum _ { k = 0 } ^ { K - 1 } D _ { \mathrm { K L } } \left( \pi ^ { \theta } ( \cdot \vert x _ { t _ { k } } ) \vert \vert \pi ^ { \theta _ { \mathrm { r e f } } } ( \cdot \vert x _ { t _ { k } } ) \right) \right] ,\tag{45}
$$

where $\beta$ controls the strength of the regularization. For a single step $k ,$ substituting the wrapped Gaussian form of the policy and writing $w = \log _ { x _ { t _ { k } } } ^ { \overline { { \mathcal { M } } } } ( x )$ gives

$$
D _ { \mathrm { K L } } \left( \pi ^ { \theta } ( \cdot \mid x _ { t _ { k } } ) \parallel \pi ^ { \theta _ { \mathrm { r e f } } } ( \cdot \mid x _ { t _ { k } } ) \right)\tag{46}
$$

$$
= \int _ { \mathcal { M } } \frac { \mathcal { N } ( w \mid \Delta t b _ { t _ { k } } ^ { \theta } ( x _ { t _ { k } } ) , \sigma _ { t _ { k } } ^ { 2 } \Delta t I ) } { J _ { x _ { t _ { k } } } ^ { M } ( w ) } \log \frac { \mathcal { N } ( w \mid \Delta t b _ { t _ { k } } ^ { \theta } ( x _ { t _ { k } } ) , \sigma _ { t _ { k } } ^ { 2 } \Delta t I ) } { \mathcal { N } ( w \mid \Delta t b _ { t _ { k } } ^ { \theta _ { \mathrm { r e f } } } ( x _ { t _ { k } } ) , \sigma _ { t _ { k } } ^ { 2 } \Delta t I ) } d \mathrm { v o l } _ { \mathcal { M } } ( x )\tag{47}
$$

$$
= \int _ { T _ { x _ { t _ { k } } } \mathcal { M } } \mathcal { N } ( w \mid \Delta t b _ { t _ { k } } ^ { \theta } ( x _ { t _ { k } } ) , \sigma _ { t _ { k } } ^ { 2 } \Delta t I ) \log \frac { \mathcal { N } ( w \mid \Delta t b _ { t _ { k } } ^ { \theta } ( x _ { t _ { k } } ) , \sigma _ { t _ { k } } ^ { 2 } \Delta t I ) } { \mathcal { N } ( w \mid \Delta t b _ { t _ { k } } ^ { \theta _ { \mathrm { r e f } } } ( x _ { t _ { k } } ) , \sigma _ { t _ { k } } ^ { 2 } \Delta t I ) } d w ,\tag{48}
$$

where in the second line we used $d \mathrm { v o l } _ { \mathcal { M } } ( x ) = J _ { x _ { t _ { k } } } ^ { \mathcal { M } } ( w )$ dw. Thus, for policies compared at the same base point $x _ { t _ { k } }$ the manifold Jacobian cancels exactly, and the KL reduces to the ordinary Euclidean KL between the corresponding tangent-space Gaussians.

## G Symmetries of Molecular Crystals

Lattice Translations The defining feature of a crystal is its translational symmetry, which presents as a discrete translational symmetry via the lattice vectors of the crystalline unit cell, or periodic repeating unit. Any crystal is invariant under the action of the lattice translation group

$$
\mathcal { T } ( L ) = \left\{ n _ { 1 } l _ { x } + n _ { 2 } l _ { y } + n _ { 3 } l _ { z } \ | \ n \in \mathbb { Z } ^ { 3 } \right\}\tag{49}
$$

Consequently, positions are only physically meaningful modulo the lattice, i.e. as fractional coordinates $f ^ { ( i ) } =$ $\mathrm { w r a p } ( \dot { c } ^ { ( i ) } L ^ { - 1 } ) \dot { \in } \mathbb { T } ^ { 3 }$

In principle, this means the fractional centroid of the lattice point can be represented with any real number—as long as it is understood that points in 3D space are equivalent under lattice translations. To see this, let ∼ be a relation on the set $\mathbb { R } ^ { 3 }$ given by

$$
\forall x , y \in \mathbb { R } ^ { 3 } : x \sim y \iff ( x - y ) L ^ { - 1 } \in \mathbb { Z } ^ { 3 }\tag{50}
$$

where $L \in G L ^ { + } ( 3 , \mathbb { R } )$ are the unit-cell lattice vectors.

Claim. The lattice translation ∼ is a valid equivalence relation.

Proof. To be a valid equivalence relation it must be reflexive, symmetric, and transitive.

• Reflexivity: Observe tha $\forall x \in \mathbb { R } ^ { 3 } : x - x = 0$ . For any matrix the following holds: $\forall A \in \mathbb { R } ^ { 3 \times 3 } : 0 A = 0$ . This implies that $( x - x ) L ^ { - 1 } = 0 \in \mathbb { Z } ^ { 3 }$ which in turn implies that $\forall x \in \mathbb { R } ^ { 3 } : x \overset { \cdot } { \sim } x ,$ , showing the relation is indeed reflexive.

• Symmetry: If $x \sim y$ then $( x - y ) L ^ { - 1 } \in \mathbb { Z } ^ { 3 }$ . The negation $- ( x - y ) L ^ { - 1 } \in \mathbb { Z } ^ { 3 }$ is true because $\mathbb { Z } ^ { 3 }$ with + operation forms a group, and group elements have inverses. Accordingly $- ( x - y ) L ^ { - 1 } = ( y - x ) L ^ { - 1 } \iff y \sim x ,$ , showing that the relation is indeed symmetric.

• Transitivity: Supposing $( x - y ) L ^ { - 1 } = m \in \mathbb { Z } ^ { 3 }$ and $( y - z ) L ^ { - 1 } = n \in \mathbb { Z } ^ { 3 }$ implies $y = n L + z$ . Substituting this into the relation $x \sim y$ gives $( x - ( n L + z ) ) L ^ { - 1 } = ( x - z ) L ^ { - 1 } - n = m \ { \overset { \cdot } { \Longrightarrow } } \ ( x - z ) L ^ { - 1 } = n + m \in \mathbb { Z } ^ { \frac { \cdot } { 3 } }$ showing that even after substitution this remains in $\mathbb { Z } ^ { 3 }$ . Because $\mathbb { Z } ^ { 3 }$ forms a group, $m + n \in \mathbb { Z } ^ { 3 }$ . Therefore $x \sim y \wedge y \sim z \implies x \sim z$ , showing that the relation is indeed transitive.

Therefore the relation defined by Equation 50 is an equivalence relation.

Define the set

$$
[ x ] = \{ y \in \mathbb { R } ^ { 3 } | x \sim y \}\tag{51}
$$

as the lattice equivalence class. Similarly, a location in the lattice is defined as an element of the quotient set<sup>2</sup>

$$
\mathbb { R } ^ { 3 } / L : = \left\{ [ x ] \in { \mathcal { P } } ( \mathbb { R } ^ { 3 } ) | x \in \mathbb { R } ^ { 3 } \right\}\tag{52}
$$

which emphasizes that “a location in the lattice” is a set of points representing the infinite periodic point pattern termed a crystal. This is the same point made before Equation 50.

In this work we choose the unit cell as the representative. The choice is degenerate because many unit cells can map to the same crystal. We express this choice in fractional coordinates via the wrapping function:

$$
f ^ { ( i ) } = \mathrm { w r a p } ( c ^ { ( i ) } L ^ { - 1 } ) \in \mathbb { T } ^ { 3 }\tag{53}
$$

where the wrap always gives $f ^ { ( i ) } \in [ 0 , 1 )$ . We will call the following map

$$
\pi : \mathbb { R } ^ { 3 } \to \mathbb { R } ^ { 3 } / L\tag{54}
$$

$$
c ^ { ( j ) } \mapsto \pi ( c ^ { ( j ) } ) = [ c ^ { ( j ) } ]\tag{55}
$$

the quotient map. This map takes a given point $c ^ { ( j ) }$ to its equivalence class. By choosing a consistent representative in this way we can ensure the generative model always trains on examples from the same type of representative. Mathematically this is similar to the canonicalization choice made for symmetric point clouds in the paper [70].

With respect to this group, if we were to apply its symmetry operation (which in this case is associated with the integer translation $n \in \mathbb { Z } ^ { 3 } )$ the resulting transformation $\Phi _ { g }$ on each component of a point $x = \left( U , P , \{ f ^ { ( i ) } , Q ^ { ( i ) } \} _ { i = 1 } ^ { M } \right) \in \mathbf { \bar { \mathcal { M } } }$ would be

$$
\Phi _ { g } : U  U , \quad P  P , \quad f ^ { ( i ) }  f ^ { ( i ) } + n , \quad Q ^ { ( i ) }  Q ^ { ( i ) }\tag{56}
$$

Since $\Phi _ { g }$ shifts $f ^ { ( i ) }$ by a constant $n \in \mathbb { Z } ^ { 3 }$ and acts as the identity on all other components, the differential acts trivially on the tangent space:

$$
d \Phi _ { g } : \dot { U }  \dot { U } , \dot { P }  \dot { P } , \dot { f } ^ { ( i ) }  \dot { f } ^ { ( i ) } , \dot { Q } ^ { ( i ) }  \dot { Q } ^ { ( i ) }\tag{57}
$$

This means our network must be invariant to integer translations. We achieve this by depending strictly on Cartesianspace differences $c ^ { ( j ) } - c ^ { ( j ^ { \prime } ) }$ and fractional-coordinate differences wrap $( f ^ { ( j ) } - f ^ { ( j ^ { \prime } ) } )$

Translations Beyond the discrete lattice, a physically correct model must also respect global (rigid-body) continuous translations $c ^ { ( j ) } \mapsto \overset { \cdot } { c } ^ { ( j ) } + \tau , \tau \in \mathbb { R } ^ { 3 }$ , which shift the entire crystal without changing interatomic distances. The group of global translations is $\left( \mathbb { R } ^ { 3 } , + \right)$ , and it acts trivially on our parameterization of the manifold point. Applying $\Phi _ { g }$ for $g = \tau \in \mathbb { R } ^ { 3 }$

$$
\Phi _ { g } : U  U , \quad P  P , \quad f ^ { ( i ) }  f ^ { ( i ) } + \tau L ^ { - 1 } , \quad Q ^ { ( i ) }  Q ^ { ( i ) }\tag{58}
$$

Since $\tau L ^ { - 1 }$ is a constant shift in fractional coordinates, the differential again acts trivially:

$$
d \Phi _ { g } : \dot { U }  \dot { U } , \dot { P }  \dot { P } , \dot { f } ^ { ( i ) }  \dot { f } ^ { ( i ) } , \dot { Q } ^ { ( i ) }  \dot { Q } ^ { ( i ) }\tag{59}
$$

Invariance to continuous translations is achieved in the same manner as lattice translations: through a dependence on strictly pairwise differences $c ^ { ( j ) } - c ^ { ( j ^ { \prime } ) }$ and $\mathrm { w r a p } ( f ^ { ( j ) } - f ^ { ( j ^ { \prime } ) } )$ , in which the uniform shift τ cancels identically.

Rotations In addition to translations, physically meaningful properties of a crystal are invariant under global rotations. Because L is row-major, a global rotation $R \in S O ( 3 )$ acts as $\mathbf { \bar { \Phi } } _ { c } ( j ) \mapsto c ^ { ( j ) } \dot { R } ^ { \intercal }$ , or equivalently $L \mapsto L R ^ { \top }$ . Since $L ^ { \top } = U P$ is the polar decomposition with $U \in S O ( 3 )$ and $P$ symmetric positive-definite, this gives

$$
L ^ { \top } \mapsto ( L R ^ { \top } ) ^ { \top } = R L ^ { \top } = R U P\tag{60}
$$

Since $R U \in S O ( 3 )$ and $P$ is unchanged, uniqueness of the polar decomposition implies $U \mapsto R U$ and $P \mapsto P$ . The fractional coordinates are also unchanged:

$$
f ^ { ( i ) } = c ^ { ( i ) } L ^ { - 1 } \mapsto c ^ { ( i ) } R ^ { \top } ( L R ^ { \top } ) ^ { - 1 } = c ^ { ( i ) } R ^ { \top } R L ^ { - 1 } = c ^ { ( i ) } L ^ { - 1 } = f ^ { ( i ) }\tag{61}
$$

Applying $\Phi _ { g }$ for $g = R \in S O ( 3 )$

$$
\Phi _ { g } : U  R U , \quad P  P , \quad f ^ { ( i ) }  f ^ { ( i ) } , \quad Q ^ { ( i ) }  R Q ^ { ( i ) }\tag{62}
$$

The differential is:

$$
d \Phi _ { g } : \dot { U }  R \dot { U } , \quad \dot { P }  \dot { P } , \quad \dot { f } ^ { ( i ) }  \dot { f } ^ { ( i ) } , \quad \dot { Q } ^ { ( i ) }  R \dot { Q } ^ { ( i ) }\tag{63}
$$

The rotation acts nontrivially on the orientation factors $U$ and $Q ^ { ( i ) }$ , while leaving the invariant components $P$ and $f ^ { ( i ) }$ unchanged. Our network is constructed to be explicitly equivariant with respect to these transformation laws. See Section H for details.

Molecular Point Group Symmetries To give an example before stating the formal proofs, consider water $\mathrm { ( H _ { 2 } O ) }$ : its two hydrogens are related by a $1 8 0 ^ { \circ }$ rotation $( C _ { 2 } )$ about the bisector axis. Suppose we extract an orientation frame $Q$ by PCA on the atomic positions. Now rotate the entire molecule by this $C _ { 2 }$ rotation. The two hydrogens swap, but since they are identical atoms, the resulting configuration is physically indistinguishable from the original. Yet the frame has rotated: because PCA is equivariant [68], the rotated configuration yields $g Q \neq Q$ . Both $\bar { Q }$ and $g Q$ are equally valid orientations of the same physical molecule, and there is no principled way to prefer one over the other. Any single-valued map that tries to do so will violate equivariance. The impossibility proof below makes this precise; the resolution is to return both orientations (more generally, the full orbit under the molecular point group) rather than choosing one.

Formally, let $\boldsymbol { c } ^ { ( j ) } \in \mathbb { R } ^ { 3 }$ for $j \in S _ { i }$ denote the Cartesian positions of the $N _ { i }$ atoms in molecule i. Define the rotational point group $\mathcal { G } _ { i } \subset S O ( 3 )$ as the set of rotations under which the molecular configuration is physically indistinguishable:

$$
\mathcal { G } _ { i } = \{ g \in S O ( 3 ) ~ | ~ g \cdot C ~ \mathrm { i s ~ i d e n t i c a l ~ t o ~ } C \mathrm { ~ u p ~ t o ~ r e l a b e l i n g ~ o f ~ i d e n t i c a l ~ a t o m s } \} .\tag{64}
$$

This is precisely the stabilizer of the physical configuration: the subgroup of $S O ( 3 )$ that fixes the molecule as an unordered point cloud of labeled species. For water, $\bar { \mathcal { G } } _ { i } = \{ I , C _ { 2 } \}$

Claim. $\operatorname { I f } { \mathcal { G } } _ { i }$ is nontrivial, there is no single-valued $S O ( 3 )$ -equivariant map from the physical configuration to $S O ( 3 )$

Proof. Suppose such a map R exists and denote its value $Q ^ { ( i ) } = \mathcal { R } ( C ) \in S O ( 3 )$ . Let $g \in { \mathcal { G } } _ { i }$ with $g \neq I .$ Since $g$ fixes the physical configuration, $\mathcal { R } ( g \cdot C ) = \mathcal { R } ( C )$ . By equivariance, $\mathcal { R } ( g \cdot C ) = g Q ^ { ( i ) }$ . Together:

$$
g Q ^ { ( i ) } = Q ^ { ( i ) } .\tag{65}
$$

Right-multiplying by $( Q ^ { ( i ) } ) ^ { - 1 }$ yields $g = I$ , contradicting $g \neq I .$ Therefore the map is either not equivariant or not single-valued. □

The preceding result implies that any molecule admitting a nontrivial permutation stabilizer realized by a rotation cannot have a single-valued equivariant orientation map. The coarse-graining map in Def. A.1 must instead be set-valued, returning the orbit of equivalent orientations under the molecular point group. Since $\mathcal { G } _ { i }$ is nontrivial, the coarse-graining map must be set-valued. If $Q ^ { ( i ) } = { \mathcal { R } } ( C )$ is a valid PCA frame, then for every $g \in { \mathcal { G } } _ { i }$ the frame $g Q ^ { ( i ) }$ is equally valid, since $g$ fixes the physical configuration. The set-valued map is therefore the orbit of $Q ^ { ( i ) }$ under $\mathcal { G } _ { i } \mathrm { : }$

$$
{ \hat { \mathcal { R } } } ( C ) = \{ g Q ^ { ( i ) } \mid g \in { \mathcal { G } } _ { i } \} = { \mathrm { O r b } } _ { { \mathcal { G } } _ { i } } ( Q ^ { ( i ) } ) .\tag{66}
$$

Claim. The set-valued map $\hat { \mathcal { R } }$ is $S O ( 3 )$ -equivariant: $\mathcal { \hat { R } } ( h \cdot C ) = h \mathcal { \hat { R } } ( C )$ for all $h \in S O ( 3 )$

Proof. The proof has two steps: first we identify the point group of a rotated molecule, then we compute the orbit.

Step 1: Conjugation of the point group. If $g \in { \mathcal { G } } _ { i }$ fixes the physical configuration of C, then $h g h ^ { - 1 }$ fixes that of $h \cdot C \colon$

$$
( h g h ^ { - 1 } ) \cdot ( h \cdot C ) = h ( g \cdot C ) ,\tag{67}
$$

which is physically indistinguishable from $h \cdot C$ because $g \cdot C$ is physically indistinguishable from C. This gives an isomorphism $\mathcal { G } _ { i }  \mathcal { G } _ { h \cdot C }$ via $g \mapsto h g h ^ { - 1 }$ , so the point group of the rotated configuration is $\mathcal { G } _ { h \cdot C } = h \mathcal { G } _ { i } h ^ { - 1 }$

Step 2: Equivariance of the orbit. By equivariance of PCA, the frame of $h \cdot C$ is $h Q ^ { ( i ) }$ . From Step 1, the point group of $\mathbf { \bar { \boldsymbol { h } } } \cdot \boldsymbol { C } \operatorname { i s } \lambda \mathcal { G } _ { i } \lambda ^ { - 1 }$ . Applying the set-valued map to $h \cdot C$

$$
\begin{array} { r l } & { \hat { \mathscr { R } } ( h \cdot C ) = \{ g ^ { \prime } h Q ^ { ( i ) } \mid g ^ { \prime } \in h \mathscr { G } _ { i } h ^ { - 1 } \} } \\ & { \quad \quad \quad = \{ \left( h g h ^ { - 1 } \right) h Q ^ { ( i ) } \mid g \in \mathscr { G } _ { i } \} } \\ & { \quad \quad \quad = \{ h g Q ^ { ( i ) } \mid g \in \mathscr { G } _ { i } \} = h \hat { \mathscr { R } } ( C ) , } \end{array}\tag{68}
$$

where the first line expands the definition of $\hat { \mathcal { R } }$ using the frame and point group of $h \cdot C$ , the second substitutes $g ^ { \prime } = h g h ^ { - 1 }$ , and the third cancels $h ^ { - 1 } h = I$ □

We resolve the need for a multivalued coarse graining map via data augmentation described below.

PCA Degeneracy The PCA-based orientation assignment suffers from two well-known sources of degeneracy (see [68] for a thorough review). The first is sign ambiguity: each eigenvector $e _ { k }$ is determined only up to a sign flip $e _ { k } \mapsto - e _ { k }$ , giving $2 ^ { 3 } = 8$ possible sign assignments. However, only 4 of these preserve det $( Q ^ { ( i ) } ) > 0$ , i.e. correspond to proper rotations in $S O ( 3 )$ ; the remaining 4 produce improper rotations with $\operatorname* { d e t } = - 1$ . The valid sign combinations (assuming that the determinant of the $+ , + , +$ combination is $> 0 )$ are

$$
( + e _ { 1 } , + e _ { 2 } , + e _ { 3 } ) , \quad ( - e _ { 1 } , - e _ { 2 } , + e _ { 3 } ) , \quad ( + e _ { 1 } , - e _ { 2 } , - e _ { 3 } ) , \quad ( - e _ { 1 } , + e _ { 2 } , - e _ { 3 } )\tag{69}
$$

corresponding to flipping zero or two axes. The second source is order ambiguity: permuting the three eigenvectors yields $\bar { 3 } ! = 6$ valid orderings, each defining a distinct frame. In total, this gives $\bar { 4 } \times \bar { 6 } = 2 4$ ambiguities of the PCA-based canonical pose.

Data augmentation More discussion of the insufficiency of plain PCA can be found in the appendix of Gao and Günnemann [87]. To resolve this issue we apply a simple data augmentation that spans the orbit $\hat { \mathcal { R } } ( C )$ over the course of training. Given a molecular configuration $\{ c ^ { ( j ) } \} _ { j \in S } $ , we perturb the atomic positions with small isotropic noise

$$
\bar { c } ^ { ( j ) } = c ^ { ( j ) } + \epsilon ^ { ( j ) } , \qquad \epsilon ^ { ( j ) } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I _ { 3 } ) , \qquad \sigma = 0 . 0 1 \mathring { \mathrm { \ A } }\tag{70}
$$

and apply the coarse graining map to obtain a perturbed frame

$$
( \bar { q } ^ { ( i ) } , \bar { Q } ^ { ( i ) } ) = \mathcal { C } ( \{ \bar { c } ^ { ( j ) } \} _ { j \in S _ { i } } )\tag{71}
$$

The local coordinates are then computed using the perturbed frame but the original positions

$$
\tilde { c } ^ { ( j ) } = ( \bar { Q } ^ { ( i ) } ) ^ { \top } ( c ^ { ( j ) } - q ^ { ( i ) } )\tag{72}
$$

The noise breaks the exact symmetry of the molecule, so the PCA eigenbasis is generically non-degenerate and $\bar { Q } ^ { ( i ) }$ is single-valued. Different noise realizations produce frames near different elements of the orbit $\hat { \mathcal { R } } ( C )$ , so over training the model sees all equivalent poses. Since the noise enters only through $\bar { Q } ^ { ( i ) }$ and not the local coordinates $\tilde { c } ^ { ( j ) }$ , the body-frame geometry is preserved. This removes the need for an explicit canonicalization of the molecular orientation and allows the model to see the breadth of the orbits.

In our setting, we sort the eigenvalues in decreasing order, which fixes the ordering and eliminates the 6 order ambiguities. To handle the residual 4 sign ambiguities, we augment each training example by randomly sampling one of the four valid sign combinations. Near-degenerate eigenvalues (which would reintroduce order ambiguity via floating point errors) are resolved by the noise perturbation described above, which generically lifts the degeneracy and ensures the eigenvalue ordering is well-defined.

Cell Transformations As mentioned previously, the lattice vectors of a crystal are not unique. Two sets of lattice vectors L and L<sup>¯</sup> span the same lattice if and only if [88]

$$
\bar { \cal L } = M { \cal L } , \qquad M \in G { \cal L } ( 3 , \mathbb { Z } ) , \quad | \operatorname* { d e t } M | = 1\tag{73}
$$

where $G L ( 3 , \mathbb { Z } )$ is the group of $3 \times 3$ integer matrices with determinant ±1 (unimodular matrices). Under thi transformation the fractional coordinates transform as $f ^ { ( i ) } \mapsto f ^ { ( i ) } M ^ { - 1 }$ so that the Cartesian positions $c ^ { ( i ) } = f ^ { ( i ) } L$ are unchanged. The molecular orientations $Q ^ { ( i ) }$ are similarly unaffected. More generally, an integer matrix M with | det $M | = m > 1$ produces a supercell containing m copies of the original unit cell [37]. In this work we neglect invariance to both unimodular basis changes and supercell equivalences, training on a single cell choice present in the dataset. This has been effective in practice for inorganic crystal structure prediction, and we leave explicit treatment of these symmetries to future work. We note that a cluster-based description avoids the need to account for this because it has no lattice and uses Cartesian coordinates.

Permutations A crystal is invariant under permutation of molecule indices $i = 1 , \dots , M$ and atom indices $j =$ $1 , \ldots , N$

$$
\Phi _ { g } : U  U , \quad P  P , \quad f ^ { ( \sigma ( i ) ) }  f ^ { ( i ) } , \quad Q ^ { ( \sigma ( i ) ) }  Q ^ { ( i ) }\tag{74}
$$

for $\sigma \in S _ { M }$ . We handle this symmetry by choosing our network to be permutation invariant with respect to both molecule and atom reorderings. See Section H for details.

Space groups and impact of molecular coarse-graining The space group $\mathcal { G }$ of a molecular crystal structure is the group of all Seitz operations $\{ R \mid \mathbf { t } \}$ that map the crystal to itself while preserving atomic types:

$$
\mathcal { G } = \left\{ \{ R \mid \mathbf { t } \} \in E ( 3 ) \mid \{ R \mid \mathbf { t } \} \cdot \{ c ^ { ( j ) } , a ^ { ( j ) } \} _ { j = 1 } ^ { N } = \{ c ^ { ( j ) } , a ^ { ( j ) } \} _ { j = 1 } ^ { N } \right\} .\tag{75}
$$

The space group of the coarse-grained molecular crystal is the analogous stabilizer acting on the CG descriptors:

$$
\mathcal { G } _ { \mathrm { C G } } = \{ \{ R \ : | \ : \mathbf { t } \} \in E ( 3 ) \ : | \ : \{ R \ : | \ : \mathbf { t } \} \cdot \{ q ^ { ( i ) } , Q ^ { ( i ) } \} _ { i = 1 } ^ { M } = \{ q ^ { ( i ) } , Q ^ { ( i ) } \} _ { i = 1 } ^ { M } \} .\tag{76}
$$

We suspect that coarse-graining can thus only “increase” the space group symmetry or leave it unchanged: $\mathcal { G } \le \mathcal { G } _ { \mathrm { C G } }$ The mechanism is geometric: molecular centroids tend to occupy high-symmetry packing positions. This empirical observation is a simpler version of those made in CrystalMath [23], and the CG descriptor $( q ^ { ( i ) } , Q ^ { ( i ) } )$ is a low-resolution summary that retains less information. It is the molecular shape that breaks the higher G symmetry down to $\mathcal { G } .$

We do not explicitly enforce space group symmetry in the generative model. Instead, we follow the common approach in crystal structure prediction of learning in P1 (the trivial space group with no non-trivial symmetry operations) and relying on the training data distribution to implicitly capture the statistics of higher-symmetry structures. The coarse-graining further simplifies this: since $\mathcal { G } \leq \bar { \mathcal { G } } _ { \mathrm { C G } }$ , the CG representation is at least as symmetric as the atomistic one, and a model that generates valid CG packings will tend to respect the dominant space group motifs present in the data. The fine-grained space group G is then recovered upon reconstruction of the full atomistic structure from the CG descriptors and the stored local coordinates $\tilde { c } ^ { ( j ) }$

## H Neural Network Architecture

![](images/5603b9edde5238251540051d9b0f583dc4c6629d0c3a5dc30434e39203cf307a.jpg)  
Figure 4: Geometric molecule embedding. Rotated atomic coordinates define a radius graph; edge lengths are expanded with a Gaussian radial basis and edge directions with spherical harmonics, then combined with species and time embeddings to form equivariant node features to be fed into a deep message passing network whose final hidden states are averaged to produce a molecule embedding.

Geometric Molecule Embedding The CG-OMatG network operates in two stages. In the first stage, it constructs a molecule embedding via the Geometric Molecule Embedding module (Figure 4). The module takes as input the tuple

$$
( \{ Q _ { t } ^ { ( i ) } \} _ { i = 1 } ^ { M } , \{ \tilde { c } ^ { ( j ) } \} _ { j = 1 } ^ { N } , t , \mathcal { A } ) .\tag{77}
$$

Each canonical atomic coordinate is rotated by the time-dependent rotation matrices:

$$
c ^ { ( j ) } = Q _ { t } ^ { ( i ) } \tilde { c } ^ { ( j ) } + q ^ { ( i ) } .\tag{78}
$$

This produces a time-dependent atomic position.

A radius graph is then built using the rotated coordinates. Two atoms m and n are connected if

$$
\| c ^ { ( n ) } - c ^ { ( m ) } \| \leq r _ { \mathrm { c u t } } \quad { \mathrm { ~ a n d ~ } } \quad n , m \in S _ { i }\tag{79}
$$

in which case the adjacency matrix has entry $a _ { n m } = 1$ . This means that the atoms must be nearby and within the same molecule. For each edge $e _ { k }$ (with $k = 1 , \ldots , E )$ , we form radial features by expanding the edge length $\| e _ { k } \|$ in a set of radial basis functions using e3nn’s soft\_one\_hot\_linspace. This can be viewed as a projection onto a basis:

$$
y _ { l } ( e _ { k } ) = Z ^ { - 1 } f _ { l } ( \| e _ { k } \| ) , \qquad \mathrm { w i t h } \qquad \left. \sum _ { l = 1 } ^ { l _ { \operatorname* { m a x } } } y _ { l } ( e _ { k } ) ^ { 2 } \right. _ { e _ { k } } \approx 1 ,\tag{80}
$$

where $\langle \cdot \rangle _ { e _ { k } }$ denotes an average over edges.

In this work, at the intramolecular message passing stage, we use the Gaussian basis with cutoff=True. Let $l _ { \mathrm { m a x } }$ denote the number of radial basis functions and define the spacing and centers (excluding endpoints) by

$$
\Delta = \frac { r _ { \mathrm { c u t } } } { l _ { \mathrm { m a x } } + 1 } , \qquad c _ { l } = l \Delta , \qquad \ell = 1 , \dots , l _ { \mathrm { m a x } } .\tag{81}
$$

![](images/2511f156190c8ed20a71a330f0db4974a83e1da001a05d7d91737c0d2a66e7a0.jpg)  
Figure 5: Overall architecture. Geometric inputs are embedded and propagated on a periodic radius graph over centroid coordinates (via ghost centroids), yielding per-centroid hidden states for lattice, fractional-coordinate, and rotational-velocity prediction.

Then the lth radial basis component is

$$
y _ { l } ( e _ { k } ) = \frac { 1 } { 1 . 1 2 } \exp \left( - \left( \frac { \| e _ { k } \| - c _ { l } } { \Delta } \right) ^ { 2 } \right)\tag{82}
$$

For more details, see the e3nn documentation.

In addition to radial features, we compute angular features by applying spherical harmonics to the normalized edge directions $\boldsymbol { e } _ { k } / \lVert \boldsymbol { e } _ { k } \rVert$ . For each ℓ, the spherical harmonics define a map $Y ^ { \ell } : \mathbb { R } ^ { 3 }  \mathbb { R } ^ { 2 \ell + 1 }$ satisfying rotation equivariance:

$$
Y ^ { \ell } ( R x ) = D ^ { \ell } ( R ) Y ^ { \ell } ( x ) ,\tag{83}
$$

where $D ^ { \ell } ( R )$ is the Wigner-D matrix for rank-ℓ irreducible representations. We normalize them such that $\| Y ^ { \ell } ( x ) \| =$ $2 \ell + 1$ . By equivariance, applying the time-dependent rotations ${ Q } _ { t } ^ { \left( i \right) }$ in (78) corresponds to rotating the spherical harmonic features by the same transformation.

After embedding positional information, we embed the atomic species A using a learned lookup, which is equivalent to applying a linear layer without bias to a one-hot encoding. The flow time t is embedded via a sinusoidal time embedding. All non-positional features are concatenated and passed through a linear layer to obtain a consistent feature shape across tensor ranks. The resulting per-atom features $s ^ { ( \bar { j } ) }$ contain irreducible components of ranks $\ell = 0 , 1 , \ldots , \ell _ { \mathrm { m a x } }$ , with C channels per rank. These features are then processed by the deep message passing network, to be described later. To obtain a molecule-level hidden state, we sum the final per-atom geometric features across all atoms in the molecule:

$$
h ^ { ( i ) } = \sum _ { j \in \mathrm { m o l e c u l e i } } ^ { N } s ^ { ( j ) } ,\tag{84}
$$

where $s ^ { ( j ) }$ denotes the final geometric hidden feature of atom $j$ and $N _ { i }$ is the number of atoms in molecule i.

![](images/672539999ed6ade42919ba66d3d96bd5e6814f4b82e65a71709f357b9e0948fb.jpg)  
Figure 6: Prediction heads. Per-centroid hidden states feed four heads: fractional translation, cell rotation, cell stretch, and molecular orientation. Outputs are mapped to tangent vectors on each factor of M via Riemannian logarithm or Lie-algebra left-translation.

Overall Architecture The overall architecture (Figure 5) closely mirrors the molecule embedding module, but it acts on a different set of inputs. Rather than using sinusoidal time embeddings and chemical species, it uses geometric features derived from the time-dependent rotations and the time-dependent unit cell, the PCA eigenvalues of the rigid body, together with the geometric molecule embedding; these features are passed through a linear layer to obtain a consistent shape across spherical tensor ranks (meaning they share the same number of channels for all ranks). Message passing is then performed on a periodic radius graph constructed from centroid coordinates (not the fractional coordinates). We implement periodicity by duplicating the structure to create ghost centroids, with enough replicas so that every centroid in the fundamental cell can access all neighbors within the cutoff radius $r _ { \mathrm { c u t } } .$ . This is then fed into a regular radius graph afterwards. The resulting geometric graph is then processed by a deep message passing network.

Prediction Heads The prediction heads act on the per-centroid hidden state at the output of the crystal branch and emit one tangent vector per modeled field of $\mathcal { M }$ . The fractional-position head outputs a per-centroid translational tangent vector $\mathcal { F } ^ { ( i ) } \in T _ { f ^ { ( i ) } } \mathbb { T } ^ { 3 } \cong \mathbb { R } ^ { 3 }$ . The cell is decomposed as $L _ { t } = ( U _ { t } P _ { t } ) ^ { \top }$ with $U _ { t } \in S O ( 3 )$ and $P _ { t } \in \mathrm { S y m _ { 3 } ^ { + } }$ , and is modeled by two heads. The cell-rotation head reads out $\omega _ { U } ^ { ( i ) } \in \mathbb { R } ^ { 3 }$ at every centroid, mean-pools over the centroids in the unit cell to a single $\omega _ { U } \in \mathfrak { s o } ( 3 ) = T _ { I } S O ( 3 )$ per cell, and lifts it to $T _ { U _ { t } } { \bar { S } } O ( 3 )$ by left translation,

$$
{ \mathcal { U } } _ { t } = U _ { t } \omega _ { U } ^ { \wedge } .\tag{85}
$$

![](images/f186d5d7eeebc77ea0b14f8ccbb11480d3e8283271732162cb7697134f31a296.jpg)  
Figure 7: Deep message-passing network. Left: two message-update blocks with a residual-sum skip and a final concatenation skip. Right: the message layer uses an RBF-conditioned weighted tensor product with spherical harmonics; the update layer applies tensor augmentation, scalar gating, and geometric layer normalization.

The cell-stretch head reads out the six independent Voigt components of a symmetric $3 \times 3$ matrix at every centroid, mean-pools over the centroids, and reshapes to a tangent vector $\mathrm { \dot { \mathcal { P } } } _ { t } \in T _ { P _ { t } } \mathrm { S y m _ { 3 } ^ { + } } = \mathrm { S y m _ { 3 } } ;$ the full cell velocity follows from the product rule,

$$
\begin{array} { r } { \dot { L } _ { t } = \left( \mathcal { U } _ { t } P _ { t } + U _ { t } \mathcal { P } _ { t } \right) ^ { \top } . } \end{array}\tag{86}
$$

For the per-molecule rotations ${ Q } _ { t } ^ { \left( i \right) }$ , we consider two interchangeable heads that both produce a tangent vector $\mathcal { Q } _ { t } ^ { ( i ) } \in T _ { Q _ { t } ^ { ( i ) } } S O ( 3 )$ . The first predicts a denoised endpoint $\hat { Q } _ { t = 1 } ^ { ( i ) }$ and maps it back via the Riemannian logarithm,

$$
\mathcal { Q } _ { t } ^ { ( i ) } = \log _ { Q _ { t } ^ { ( i ) } } \big ( \hat { Q } _ { t = 1 } ^ { ( i ) } \big ) .\tag{87}
$$

To enforce $\hat { Q } _ { t = 1 } ^ { ( i ) } \in S O ( 3 )$ the head outputs two $\ell = 1$ vectors which are orthonormalized via Gram–Schmidt, with their cross product completing the rotation matrix; molecules whose two predicted vectors are nearly collinear are masked out of the rotation loss.

Deep Message Passing Network Our message-passing network closely follows EquiJump $\left[ 8 9 \right] ;$ see that work for further details. For completeness, we describe it here. Each deep message-passing network is built from a repeated sequence of message-update blocks with two skip connections. Starting from state (a), a message-update block produces an intermediate representation (b) (the node state immediately before the first + in the diagram). This intermediate is combined with (a) through a residual summation to yield the post-skip state (b) (immediately after the +). A second message-update block is then applied to produce the current state (the node state immediately before ⊕). Finally, a concatenation skip forms [(a), (b), current], which is projected back to the hidden dimension by a linear layer. This full pattern is repeated some number of times (as indicated by the repeat symbol), with independent parameters in each repetition. The overall block structure is shown on the left of Figure 7.

In the message layer, each node’s features are split into a scalar stream and a spherical-tensor stream. The scalar stream is concatenated with the edge radial basis embedding and passed through an MLP to produce mixing weights. These weights parameterize a weighted tensor product between the node’s spherical-tensor features and the edge spherical harmonics, yielding edge messages. Messages are summed over each node’s neighborhood, concatenated with the node’s original state, and mapped back to the hidden dimension with a linear layer. In the update layer, tensor features are augmented by concatenating the tensor-square with the original tensor features. In parallel, the scalar stream is split to produce scalar gates that multiplicatively modulate the tensor features. The resulting tensor features are then passed through a linear layer followed by geometric layer normalization [90]. The message and update operations are shown on the right of Figure 7.

## I UMA Relaxation Scheme

For every candidate molecular crystal, we evaluate the uma-s-1p2 universal foundation potential energy model [78] via its FAIRChemCalculator ASE interface, using the molecular-crystal task head trained on the OMC25 dataset [80]. The structure is then relaxed under a three-stage BFGS protocol implemented in ASE [91] that mirrors the relaxation pipeline of MolCrystalFlow [22]:

1. Rigid-body warm-up. Each molecular building block is held internally rigid by a custom ASE constraint: at every BFGS step, positions are re-projected onto the rigid-body manifold by Kabsch alignment to the block’s reference geometry, and per-atom forces are replaced by the corresponding net-force / net-torque contributions of that block. We run BFGS for $N _ { \mathrm { r i g i d } } = 1 0 0$ iterations, allowing centroids and orientations to relax while intramolecular geometries are preserved.

2. Coupled cell + atomic relax. Lattice and atomic degrees of freedom are co-optimised by wrapping the system in an ASE FrechetCellFilter, whose coordinates are the atomic positions in the undeformed cell together with the matrix logarithm of the deformation gradient. BFGS is run on the filtered system until the maximum atomic force component falls below $f _ { \mathrm { m a x } } = 0 . 0 1 \mathrm { e V } / \mathring { \mathrm { A } } \mathrm { o r }$ a 1000-step cap is reached.

3. Atomic-only relax. The cell is then fixed and a final BFGS pass on the atomic coordinates re-converges them to the same $f _ { \mathrm { m a x } } = 0 . 0 1 \mathrm { e V } / \mathring { \mathrm { A } }$ tolerance, eliminating any residual atomic forces left over from the previous stage.

For each structure we record the initial and relaxed UMA energies, the relaxation gain $\Delta E = E _ { \mathrm { i n i t } } - E _ { \mathrm { r e l a x e d } }$ , the input-to-final atomic RMSD, the maximum lattice-vector drift, per-stage step counts, and convergence flags (a stage is convergent iff it exits before the step cap, i.e. on the $f _ { \mathrm { m a x } }$ criterion). Per-step trajectories of all three stages can optionally be stitched together for inspection.

## J Hyperparameters

Table 2: MolCrystalFlow inference hyperparameters used with the OMC25-MCF checkpoint, following the values recommended in the project README.

Parameter Value   
Inference   
config-name omc25\_inference.yaml   
ckpt\_path model-checkpoints/omc25-mcf/best.ckpt   
num\_samples 30   
Interpolant — sampling   
num\_timesteps 50   
Interpolant — translations   
scaling 9.0 Å   
Interpolant — rotations   
exp\_rate 3.0   
Model — backbone embedder   
num\_atom\_types 12   
Aggregation   
Draws per crystal K 30   
Sampling strategy Independent draws from the flow ODE

Table 3: Genarris hyperparameters used for the molecular-crystal generation baseline.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Master</td><td></td></tr><tr><td>Z</td><td>Per-crystal (#unique bb_indices)</td></tr><tr><td>MPI ranks</td><td>8</td></tr><tr><td>Workflow</td><td></td></tr><tr><td>tasks</td><td>generation, symm_rigid_press</td></tr><tr><td>Generation</td><td></td></tr><tr><td>generation_type</td><td>crystal</td></tr><tr><td>spg_distribution_type</td><td>standard</td></tr><tr><td>num_structures_per_spg</td><td>30</td></tr><tr><td>unit_cell_volume_mean</td><td>predict</td></tr><tr><td>volume_mult</td><td>1.5</td></tr><tr><td>sr</td><td>0.85</td></tr><tr><td>natural_cutoff_mult</td><td>1.2</td></tr><tr><td>tol</td><td>0.01</td></tr><tr><td>max_attempts_per_spg</td><td>107</td></tr><tr><td>max_attempts_per_volume</td><td>107</td></tr><tr><td>Symmetric rigid-body relaxation</td><td></td></tr><tr><td>method</td><td>BFGS</td></tr><tr><td>sr</td><td>0.85</td></tr><tr><td>natural_cutoff_mult</td><td>1.2</td></tr><tr><td>tol</td><td>0.01</td></tr><tr><td>Aggregation</td><td></td></tr><tr><td>Draws per crystal K</td><td>30</td></tr><tr><td>Sampling strategy</td><td>W/o replacement; w/ replacement if #converged &lt; K</td></tr></table>

Table 4: CG-OMatG pre-training hyperparameters. Each MLP has a single hidden layer with the listed width.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td colspan="2">Training</td></tr><tr><td>Batch size (global)</td><td> $4 \times 3 2 0 = 1 2 8 0$ </td></tr><tr><td>Optimizer</td><td> $\mathrm { A d a m W } , \mathrm { l r } = 5 \times 1 0 ^ { - 4 } .$  weight decay = 0.00818</td></tr><tr><td>LR schedule</td><td>Cosine annealing, 1500 epochs,  $\eta _ { \mathrm { m i n } } = 1 0 ^ { - 7 }$ </td></tr><tr><td>Gradient clipping</td><td>0.5, per-element</td></tr><tr><td>Time sampling</td><td>Logit-normal:  $t = \sigma ( 1 . 7 Z + 0 . 8 ) , Z \sim \mathcal { N } ( 0 , 1 )$ </td></tr><tr><td>PCA-frame augmentation scale</td><td>Isotropic Gaussian noise  $( \sigma = 0 . 0 1 \mathring \mathrm { A } )$ </td></tr><tr><td colspan="2">Molecule branch — intra-molecule message passing</td></tr><tr><td>Layers / channels / irrep rank</td><td> $1 / 1 6 / 1$ </td></tr><tr><td>Edge construction</td><td>Spherical harmonics  $\ell _ { \mathrm { m a x } } = 2 ,$  64 Gaussian radial bases, cutoff  $r _ { c } = $ </td></tr><tr><td>Edge-weight / node-update MLPs</td><td>8.0 Å, max 100 neighbors [512]/[512]</td></tr><tr><td>Species embedding</td><td>Dimension 64, vocabulary size 100</td></tr><tr><td>Atom → molecule pooling</td><td>Sum reduction</td></tr><tr><td colspan="2">Crystal branch — inter-molecular message passing</td></tr><tr><td colspan="2">Layers / channels / irrep rank</td></tr><tr><td>Edge construction</td><td> $5 / 1 6 / 2$  Spherical harmonics  $\ell _ { \mathrm { m a x } } = 2 ,$  64 Gaussian radial bases + 64-frequency</td></tr><tr><td>Time-conditioned cutoff  $r _ { c } ( t )$ </td><td>sin/cos Fourier features on fractional BB-BB displacements Lagrange interpolation through  $( t , \ r _ { c } )$  二</td></tr><tr><td>Edge-weight / node-update MLPs</td><td>{(0, 9.0), (0.5, 10.0), (1, 11.5)} Å; max 100 neighbors [1024] / [256]</td></tr><tr><td>Rotation readout</td><td>Gram-Schmidt orthogonalization  $+ \log _ { S O ( 3 ) }$ </td></tr><tr><td colspan="2">Flow matching — per-field interpolants</td></tr><tr><td> $\mathbb { T } ^ { 3 }$  fractional positions</td><td>Periodic linear interpolant, Euler ODE; center-of-mass motion subtracted before loss</td></tr><tr><td>SO(3) molecule orientations</td><td>Riemannian geodesic interpolant, MODE</td></tr><tr><td>Sym+ (R) lattice shape</td><td>Riemannian geodesic interpolant on SPD manifold  $\scriptstyle ( \mathrm { E x p } / \mathrm { L o g } / \langle \cdot , \cdot \rangle _ { \mathrm { S P D } } ) ,$  MODE</td></tr><tr><td>SO(3) cell orientation</td><td>Riemannian geodesic interpolant, MODE</td></tr><tr><td colspan="2">Loss weights  $\left( \mathbb { T } ^ { 3 } / S O ( 3 ) _ { \mathrm { m o l } } / S y m _ { 3 } ^ { + } / S O ( 3 ) _ { \mathrm { c e l l } } \right)$ </td></tr><tr><td></td><td> $1 5 . 0 / 3 . 0 / 1 . 0 / 1 . 0$ </td></tr><tr><td colspan="2">Priors (t = 0)</td></tr><tr><td>Fractional positions</td><td>Uniform on  $[ 0 , 1 ) ^ { 3 }$ </td></tr><tr><td>Orientations (mol. + cell)</td><td>Haar-uniform on  $S O ( 3 )$ </td></tr><tr><td>Lattice</td><td>Log-normal lengths + uniform angles on  $[ 6 0 ^ { \circ } , 1 2 0 ^ { \circ } ]$  , parameters fit from CSD</td></tr><tr><td colspan="2">Inference</td></tr><tr><td>ODE integration</td><td>500 Euler steps</td></tr><tr><td>Velocity annealing</td><td> $\hat { b } ( t ) = ( 1 + \alpha _ { v } t ) b ( t ) ; \alpha _ { v } \colon$  positions 8.0, mol. orient. 8.0, cell rot. 0.1, lattice shape 0.0</td></tr></table>

Table 5: CG-OMatG RL fine-tuning hyperparameters (GRPO with PPO clipping).
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td colspan="2">Training</td></tr><tr><td>Optimizer</td><td> $\mathrm { A d a m } , \mathrm { l r } = 1 0 ^ { - 4 }$ </td></tr><tr><td>Max steps</td><td>5000</td></tr><tr><td>Gradient clipping</td><td>1.0, global norm</td></tr><tr><td colspan="2">GRPO /PPO</td></tr><tr><td>Group size / num. groups</td><td>64 / 5</td></tr><tr><td>Shared xo within group</td><td>Yes</td></tr><tr><td>PPO clip €</td><td>0.1</td></tr><tr><td>PPO epochs per step</td><td>1</td></tr><tr><td colspan="2">Exploration noise  $( \sigma ( t ) = \sigma _ { 0 } \sqrt { t } )$ </td></tr><tr><td>T³  $/ S O ( 3 ) _ { \mathrm { m o l } } / S y m _ { 3 } ^ { + }$ </td><td>0.1 / 0.1 / 0.1</td></tr><tr><td> $S O ( 3 ) _ { \mathrm { c e l l } }$ </td><td>0.01</td></tr><tr><td colspan="2">Policy loss weights (pos / mol. rot / lattice / cell rot)</td></tr><tr><td>Policy</td><td> $1 . 0 / 1 . 0 / 0 . 5 / 0 . 2 5$ </td></tr><tr><td>KL regularization</td><td> $1 0 ^ { - 3 }$  (all fields)</td></tr><tr><td colspan="2">Reward (UMA energy)</td></tr><tr><td>Scale</td><td>1.0</td></tr><tr><td>Invalid-structure penalty</td><td>3.0 eV/atom</td></tr><tr><td>Volume-check cutoff</td><td>0.1</td></tr><tr><td>Polar-sine cutoff</td><td>0.001</td></tr></table>

## K Additional Results

Table 6: CCDC packing-similarity metrics for the three homomolecular sixth CSD blind-test targets (k = 30 inference). Values are rates; ↑ higher is better and ↓ lower is better.
<table><tr><td>Target</td><td>Method</td><td>Solved ↑</td><td>Solved (collisions allowed) ↑</td><td>Packing match ↑</td><td>Packing match (per draw) ↑</td><td>Clash ↓</td></tr><tr><td>NACJAF</td><td>OXtal</td><td> $0 . 3 0 \pm 0 . 1 5$ </td><td> $0 . 3 0 \pm 0 . 1 5$ </td><td> $\mathbf { 1 . 0 0 \pm 0 . 0 0 }$ </td><td> $\mathbf { 0 . 0 8 \pm 0 . 0 1 }$ </td><td> $\mathbf { 0 . 0 0 \pm 0 . 0 0 }$ </td></tr><tr><td></td><td>MCF</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.93</td></tr><tr><td></td><td>CG-OMatG</td><td> $0 . 1 0 \pm 0 . 1 0$ </td><td> $0 . 1 0 \pm 0 . 1 0$ </td><td> $0 . 4 0 \pm 0 . 1 6$ </td><td> $0 . 0 2 \pm 0 . 0 1$ </td><td> $0 . 2 0 \pm 0 . 0 2$ </td></tr><tr><td></td><td>CG-OMatG-IRL</td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 5 0 \pm 0 . 1 7$ </td><td> $0 . 0 3 \pm 0 . 0 1$ </td><td> $0 . 0 7 \pm 0 . 0 1$ </td></tr><tr><td></td><td>CG-OMatG (relaxed)</td><td> ${ \bf 0 . 6 0 \pm 0 . 1 6 }$ </td><td> ${ \bf 0 . 6 0 \pm 0 . 1 6 }$ </td><td> $0 . 8 0 \pm 0 . 1 3$ </td><td> $0 . 0 5 \pm 0 . 0 1$ </td><td> $0 . 0 2 \pm 0 . 0 1$ </td></tr><tr><td></td><td>CG-OMatG-IRL (relaxed)</td><td> $0 . 1 0 \pm 0 . 1 0$ </td><td> $0 . 1 0 \pm 0 . 1 0$ </td><td> $0 . 5 0 \pm 0 . 1 7$ </td><td> $0 . 0 2 \pm 0 . 0 1$ </td><td> $\mathbf { 0 . 0 0 \pm 0 . 0 0 }$ </td></tr><tr><td>XAFPAY</td><td>OXtal</td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $\mathbf { 1 . 0 0 \pm 0 . 0 0 }$ </td><td> $\mathbf { 0 . 0 9 \pm 0 . 0 1 }$ </td><td> $\mathbf { 0 . 0 0 \pm 0 . 0 0 }$ </td></tr><tr><td></td><td>MCF</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.93</td></tr><tr><td></td><td>CG-OMatG</td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td></td><td> $0 . 1 0 \pm 0 . 1 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 5 1 \pm 0 . 0 2$ </td></tr><tr><td></td><td>CG-OMatG-IRL</td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $\begin{array} { c } { 0 . 0 0 \pm 0 . 0 0 } \\ { 0 . 0 0 \pm 0 . 0 0 } \end{array}$ </td><td> $0 . 3 0 \pm 0 . 1 5$ </td><td> $0 . 0 1 \pm 0 . 0 1$ </td><td> $0 . 4 1 \pm 0 . 0 3$ </td></tr><tr><td></td><td>CG-OMatG (relaxed)</td><td> ${ \bf 0 . 1 0 \pm 0 . 1 0 }$ </td><td> ${ \bf 0 . 1 0 \pm 0 . 1 0 }$ </td><td> $0 . 1 0 \pm 0 . 1 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $\mathbf { 0 . 0 0 \pm 0 . 0 0 }$ </td></tr><tr><td></td><td>CG-OMatG-IRL (relaxed)</td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $\mathbf { 0 . 0 0 \pm 0 . 0 0 }$ </td></tr><tr><td>XAFQIH</td><td>OXtal</td><td> $\mathbf { 0 . 0 0 \pm 0 . 0 0 }$ </td><td> $\mathbf { 0 . 0 0 \pm 0 . 0 0 }$ </td><td> $0 . 3 0 \pm 0 . 1 5$ </td><td> $\mathbf { 0 . 0 1 \pm 0 . 0 1 }$ </td><td> $\mathbf { 0 . 0 0 \pm 0 . 0 0 }$ </td></tr><tr><td></td><td>MCF</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.73</td></tr><tr><td></td><td>CG-OMatG</td><td> $\mathbf { 0 . 0 0 \pm 0 . 0 0 }$ </td><td> $\mathbf { 0 . 0 0 \pm 0 . 0 0 }$ </td><td> $0 . 3 0 \pm 0 . 1 5$ </td><td> $\mathbf { 0 . 0 1 \pm 0 . 0 1 }$ </td><td> $0 . 5 5 \pm 0 . 0 3$ </td></tr><tr><td></td><td>CG-OMatG-IRL</td><td> $\mathbf { 0 . 0 0 \pm 0 . 0 0 }$ </td><td> $\mathbf { 0 . 0 0 \pm 0 . 0 0 }$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 4 0 \pm 0 . 0 2$ </td></tr><tr><td></td><td>CG-OMatG (relaxed)</td><td> $\mathbf { 0 . 0 0 \pm 0 . 0 0 }$ </td><td> $\mathbf { 0 . 0 0 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 4 0 \pm 0 . 1 6 }$ </td><td> $\mathbf { 0 . 0 1 \pm 0 . 0 1 }$ </td><td> $\mathbf { 0 . 0 0 \pm 0 . 0 0 }$ </td></tr><tr><td></td><td>CG-OMatG-IRL (relaxed)</td><td> $\mathbf { 0 . 0 0 \pm 0 . 0 0 }$ </td><td> $\mathbf { 0 . 0 0 \pm 0 . 0 0 }$ </td><td> $0 . 3 0 \pm 0 . 1 5$ </td><td> $\mathbf { 0 . 0 1 \pm 0 . 0 1 }$ </td><td> $\mathbf { 0 . 0 0 \pm 0 . 0 0 }$ </td></tr></table>

Error bars are SEM over ten blocks of 30 draws. MCF has one K = 30 block and therefore no error bars. Relaxed rows use the UMA relaxation procedure in Appendix I.

CG-OMatG matches NACJAF before relaxation (8/15 at 1.86 Å) and after relaxation (11/15 at 0.37 Å), and XAFPAY after relaxation (8/15 at 1.51 Å); no method solves XAFQIH.

OXtal comparison on a training-disjoint OMC subset OXtal’s shipped checkpoint was trained on other members of the OMC test set used in Table 1, so that comparison is not apples-to-apples. We therefore selected all structures in our held-out 1,000-structure OMC test set that were absent from both OXtal’s and CG-OMatG’s training data, leaving 37 structures that were processed with OXtal’s released routine. Table 7 reports the resulting comparison.

Table 7: CCDC packing-similarity metrics on the 37 OMC targets absent from both models’ training data. Values are mean ± SEM over ten K = 30 blocks.
<table><tr><td>Method</td><td>Solved</td><td>Collisions allowed</td><td>Packing match</td><td>Per draw</td><td>Clash</td></tr><tr><td>OXtal</td><td> ${ \bf 0 . 2 0 \pm 0 . 0 2 }$ </td><td> ${ \bf 0 . 2 0 \pm 0 . 0 2 }$ </td><td> $\mathbf { 0 . 5 0 \pm 0 . 0 2 }$ </td><td> $\mathbf { 0 . 1 0 \pm 0 . 0 0 }$ </td><td> $\mathbf { 0 . 0 0 \pm 0 . 0 0 }$ </td></tr><tr><td>CG-OMatG</td><td> $0 . 0 5 \pm 0 . 0 0$ </td><td> $0 . 0 8 \pm 0 . 0 0$ </td><td> $0 . 3 9 \pm 0 . 0 1$ </td><td> $0 . 0 5 \pm 0 . 0 0$ </td><td> $0 . 4 3 \pm 0 . 0 0$ </td></tr><tr><td>CG-OMatG-IRL</td><td> $0 . 0 6 \pm 0 . 0 0$ </td><td> $0 . 1 0 \pm 0 . 0 0$ </td><td> $0 . 4 9 \pm 0 . 0 1$ </td><td> $0 . 0 6 \pm 0 . 0 0$ </td><td> $0 . 2 5 \pm 0 . 0 0$ </td></tr></table>

OXtal has the higher solved rate on this subset, while CG-OMatG-IRL nearly closes the packing-match gap and reduces clashes relative to CG-OMatG. This does not invalidate CG-OMatG: a true apples-to-apples comparison remains difficult because of differing data splits and benchmarking pipelines. A fair comparison would require fully retraining OXtal on our split, which is currently impossible without a valid released training setup.

Energy and density before and after relaxation Figure 8 compares the sampled structures with the same structures after UMA relaxation.

![](images/a50672ba9a0f82873deef8a7f430e1b80604d32d630563b192495a9c42ada057.jpg)

## Generated crystals as sampled, before relaxation

![](images/6d2afd96e4f8a7caf71d6718128c05ba867637f6b5a4f6cd001e7d6f93b2b479.jpg)  
The same crystals after UMA relaxation

![](images/b80c85b604756b8b7b5c3866813172601e56f8c5e47a19d26ecafafa5098e578.jpg)

o GT (experiment)o CG-OMatG CG-OMatG-RLo CG-OMatG-RL + conformer  
![](images/0b0fa27b64dc15be2da22f978f3f6f65a2138d6f1a7f050efd1702d4dc724025.jpg)

![](images/f343cfe4353a2b506692d6affc8039b783d86928d56ee126c95dff2b619f8cae.jpg)

![](images/e50ed7bd786cb5b19247a07c83e7117065bef78084551c46083feac043263dd3.jpg)  
Figure 8: Energy versus density for generated sixth CSD blind-test structures before relaxation (top) and after UMA relaxation (bottom). Energies are reported relative to the UMA-relaxed experimental target. Relaxation sharpens the energy–density distributions for NACJAF, XAFPAY, and XAFQIH; for all three targets, the best CG-OMatG sample lies within 0.01 eV/atom of the lowest-energy relaxed ground truth with low density error.

DFT would be required to make claims about small energy differences; the UMA evaluations here are intended to evaluate the methodology.

Velocity annealing We swept the positional and rotational velocity-annealing parameters over $s _ { \mathrm { p o s } } ^ { \prime } , s _ { \mathrm { r o t } } ^ { \prime } \in$ {0, 2, 4, 6, 8, 10} using $s ( t ) = 1 + s ^ { \prime } t .$ . Figure 9 reports the full sweep.

OMC velocity-annealing sweep — CCDC metrics (K=30, first 128 structures) BASE RL
<table><tr><td>10</td><td>0.055</td><td>0.047</td><td>0.031</td><td>0.055</td><td>0.047</td><td>0.062</td></tr><tr><td>8</td><td>0.055</td><td>0.047</td><td>0.031</td><td>0.055</td><td>0.047</td><td>0.070</td></tr><tr><td>Posb  so) sor 6</td><td>0.047</td><td>0.055</td><td>0.039</td><td>0.055</td><td>0.055</td><td>0.062</td></tr><tr><td>4</td><td>0.047</td><td>0.062</td><td>0.047</td><td>0.062</td><td>0.055</td><td>0.047</td></tr><tr><td>2</td><td>0.055</td><td>0.055</td><td>0.047</td><td>0.062</td><td>0.055</td><td>0.055</td></tr><tr><td>0</td><td>0.047</td><td>0.047</td><td>0.047</td><td>0.070</td><td>0.047</td><td>0.055</td></tr><tr><td></td><td>0</td><td>2</td><td>4 rot (bb_rot VA factor)</td><td>6</td><td>8</td><td>10</td></tr></table>

<table><tr><td rowspan=3 colspan=1>108</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=9 colspan=1>0.160.14(t btrter0.120.10Ap oved0.080.060.0410</td></tr><tr><td rowspan=1 colspan=1>0.109</td><td rowspan=1 colspan=1>0.133</td><td rowspan=1 colspan=1>0.148</td><td rowspan=1 colspan=1>0.156</td><td rowspan=1 colspan=1>0.141</td><td rowspan=1 colspan=2></td><td rowspan=6 colspan=1></td></tr><tr><td rowspan=1 colspan=1>0.133</td><td rowspan=1 colspan=1>0.125</td><td rowspan=1 colspan=1>0.141</td><td rowspan=1 colspan=1>0.164</td><td rowspan=1 colspan=1>0.148</td><td rowspan=1 colspan=2>0.141</td></tr><tr><td rowspan=6 colspan=1>6420</td><td rowspan=2 colspan=1>0.1410.141</td><td rowspan=1 colspan=1>0.148</td><td rowspan=1 colspan=1>0.156</td><td rowspan=1 colspan=1>0.148</td><td rowspan=1 colspan=1>0.148</td><td rowspan=1 colspan=2>0.156</td></tr><tr><td rowspan=1 colspan=1>0.148</td><td rowspan=1 colspan=1>0.125</td><td rowspan=1 colspan=1>0.156</td><td rowspan=1 colspan=1>0.148</td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1>0.148</td><td rowspan=1 colspan=1>0.133</td><td rowspan=1 colspan=1>0.141</td><td rowspan=1 colspan=1>0.164</td><td rowspan=1 colspan=1>0.148</td><td rowspan=1 colspan=2>0.148</td></tr><tr><td rowspan=1 colspan=1>0.117</td><td rowspan=1 colspan=1>0.141</td><td rowspan=1 colspan=1>0.125</td><td rowspan=1 colspan=1>0.133</td><td rowspan=1 colspan=1>0.141</td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=2 colspan=1>0</td><td rowspan=2 colspan=1>2</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>6</td><td rowspan=2 colspan=1>8</td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>rot (bb_rot</td><td rowspan=1 colspan=1>VA factor)</td><td></td><td></td></tr></table>

<table><tr><td rowspan="2"></td><td>10</td><td>0.510</td><td>0.425</td><td>0.405</td><td>0.397</td><td>0.394</td><td>0.388</td></tr><tr><td>8</td><td>0.512</td><td>0.434</td><td>0.406</td><td>0.397</td><td>0.394</td><td>0.395</td></tr><tr><td>Clash rate (erps ple)</td><td>Pos  s) stor 6</td><td>0.514</td><td>0.437</td><td>0.415</td><td>0.406</td><td>0.404</td><td>0.405</td></tr><tr><td></td><td>4</td><td>0.521</td><td>0.446</td><td>0.423</td><td>0.415</td><td>0.413</td><td>0.409</td></tr><tr><td></td><td>2</td><td>0.534</td><td>0.462</td><td>0.440</td><td>0.433</td><td>0.430</td><td>0.429</td></tr><tr><td></td><td>0</td><td>0.604</td><td>0.528</td><td>0.514</td><td>0.511</td><td>0.512</td><td>0.514</td></tr><tr><td></td><td></td><td>0</td><td>2</td><td>4 rot (bb_rot VA factor)</td><td>6</td><td>8</td><td>10</td></tr></table>

<table><tr><td rowspan="2">10</td><td rowspan="2">0.180</td><td rowspan="2">0.166</td><td rowspan="2">0.168</td><td rowspan="2">0.170</td><td rowspan="2">0.174</td><td rowspan="2">0.175</td><td>0.60 0.55</td></tr><tr><td rowspan="6">0.50 Iowtr bter) 0.45 0.40 こ 0.35 Clash rate 0.30 0.25 0.20</td></tr><tr><td>8</td><td>0.180</td><td>0.161</td><td>0.166</td><td>0.166</td><td>0.170</td><td>0.171</td></tr><tr><td>6</td><td>0.179</td><td>0.160</td><td>0.161</td><td>0.168</td><td>0.172</td><td>0.175</td></tr><tr><td>4</td><td>0.180</td><td>0.162</td><td>0.165</td><td>0.171</td><td>0.175</td><td>0.174</td></tr><tr><td>2</td><td>0.183</td><td>0.164</td><td>0.168</td><td>0.172</td><td>0.175</td><td>0.173</td></tr><tr><td>0</td><td>0.187</td><td>0.174</td><td>0.175</td><td>0.183</td><td>0.185</td><td>0.188</td></tr><tr><td></td><td>0</td><td>2</td><td>4</td><td>6</td><td>8</td><td>10</td></tr></table>

![](images/9a51706d8d1ba7c0400876142d66326b2e92f9e1633dddfe18f5308075bba77f.jpg)

![](images/7cc10640d7147cae1c238ff845e81b21fada7f08e396bf33c064a9f2dc5f5cec.jpg)  
Figure 9: Full OMC128 velocity-annealing sweep for the base and reinforced models at $K = 3 0$ . Rows vary positional annealing and columns vary rotational annealing. Shown are solved rate, clash rate per draw, and target-level packing similarity.

Reward choice and circularity The use of UMA in our reinforcement learning pipeline is partly circular, since OMC25-MCF was relaxed with UMA and UMA is also used to define the reward. To test how our results hinge on UMA as a reward, we reinforced the same pretrained checkpoint using Orb instead. The choice of Orb versus UMA as a reward, in this setting, does not markedly affect the robustness of the CG-OMatG-IRL strategy, as shown in Table 1. Most energy-function rewards may be expected to improve performance, especially on OMC. However, the performance gap between reinforced models on OMC and CSD data stems less from UMA’s reward–data alignment with OMC and more from the difficulty of modeling experimental CSD data using an energy reward. The most reliable reward and assessment signal would be DFT, which is not scalable for reinforcement learning. Foundational MLIPs such as UMA, MACE, SevenNet, or Orb are reasonable proxies for evaluating the quality of proposal structures.

More broadly, using the same model for relaxation and reward does not invalidate the experiment. Crystal structure generation is a rare-event sampling problem because low-energy basins occupy only a small part of configuration space. Alignment with RL is intended to shift the proposal distribution toward these regions. Our experiment asks whether alignment reduces the number of proposals needed to recover held-out reference minima. This is an important practical goal of computational crystal structure prediction and serves as a key step toward a practical tool. Ideally, CSP models should also reproduce experimental structures; that is, however, not the specific question addressed in this work. Our aim is to determine whether alignment can make the sampling of relevant low-energy structures more efficient, which leaves the accuracy of these minima to those fitting MLIPs.

![](images/e2073760644e217876eb35a9127a1c248dd9389a6d8cd94ebe722a4cf4c9bb3b.jpg)  
Figure 10: Reinforcement-learning trajectories for CSD with a UMA reward (left), OMC with a UMA reward (center), and OMC with an Orb reward (right). Top: per-step reward −E/N and its 50-step rolling mean. Bottom: clashes per generated structure. Reward increases and clashes decrease across all three runs.

Generated conformer inputs In practice, the conformer is not known a priori. To mimic this setting, we generated 500 ETKDGv3 conformers per molecule with RDKit, optimized and ranked them in vacuum using MMFF94s, and clustered the final geometries at a 0.50 Å heavy-atom RMSD threshold. We retained the lowest-energy representative from each cluster and relaxed it with UMA. Table 8 reports the closest recovered conformers; inference then used the top ten generated conformers.

Table 8: Recovery of experimental blind-test conformers from ETKDGv3/MMFF94s sampling.
<table><tr><td>CSD ID</td><td>Rotatable bonds</td><td>Energy rank</td><td>RMSD after MMFF94s</td><td>RMSD after UMA</td></tr><tr><td>XAFQIH</td><td>5</td><td>10</td><td>0.571 Å</td><td>0.648 Å</td></tr><tr><td>XAFPAY</td><td>6</td><td>24</td><td>0.401 Å</td><td>0.348 Å</td></tr><tr><td>NACJAF</td><td>0</td><td>1</td><td>0.057 Å</td><td>0.040 Å</td></tr></table>

Table 9: Blind-test metrics using generated conformer inputs. Each row has one K = 30 block and therefore no error bars.
<table><tr><td>Target</td><td>Pipeline</td><td>Solved</td><td>Collisions allowed</td><td>Packing match</td><td>Per draw</td><td>Clash</td></tr><tr><td>NACJAF</td><td>IRL + conformer</td><td>0.00</td><td>0.00</td><td>1.00</td><td>0.03</td><td>0.07</td></tr><tr><td></td><td>IRL + conformer (relaxed)</td><td>0.00</td><td>0.00</td><td>1.00</td><td>0.03</td><td>0.00</td></tr><tr><td>XAFPAY</td><td>IRL + conformer</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.37</td></tr><tr><td></td><td>IRL + conformer (relaxed)</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>XAFQIH</td><td>IRL + conformer</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.53</td></tr><tr><td></td><td>IRL + conformer (relaxed)</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr></table>

The inexpensive conformer sampling produces candidate ensembles containing conformers close to the experimentally observed structures, but these inputs do not yield an additional solved blind-test target.

Polymorph diversity To quantify diversity after RL post-training, we construct a graph over each set of $K = 3 0$ generated crystals. Nodes i and $j$ are connected when COMPACK aligns at least eight of fifteen molecules with $\mathrm { R M S D } _ { N } < 2 . 0 \mathring \mathrm { A }$ . The number of connected components C is the first diversity measure. Because C can overstate diversity when many components are singletons, we also report the effective number of components ef $= 1 / \textstyle \sum _ { k } p _ { k } ^ { 2 }$ where $p _ { k } = s _ { k } / 3 0$ , and the dominant-mode fraction max<sub>k</sub> $p _ { k }$ . Results in Table 10 are averaged over ten independent sets of 30 samples per target and reported as mean ± standard error.

Table 10: Diversity of generated blind-test structures under COMPACK component clustering.
<table><tr><td>Target</td><td>Method</td><td>Distinct components C</td><td>Effective components eff</td><td>Dominant-mode fraction maxk pk</td></tr><tr><td>NACJAF</td><td>CG-OMatG</td><td> ${ \bf 2 3 . 0 \pm 1 . 0 }$ </td><td> ${ \bf 1 8 . 4 \pm 1 . 2 }$ </td><td> $\mathbf { 0 . 1 3 \pm 0 . 0 1 }$ </td></tr><tr><td>NACJAF</td><td>CG-OMatG-IRL</td><td> $2 1 . 6 \pm 1 . 2$ </td><td> $1 5 . 1 \pm 1 . 5$ </td><td> $0 . 1 7 \pm 0 . 0 1$ </td></tr><tr><td>XAFPAY</td><td>CG-OMatG</td><td> ${ \bf 2 7 . 5 \pm 0 . 6 }$ </td><td> ${ \bf 2 5 . 2 \pm 1 . 2 }$ </td><td> $\mathbf { 0 . 0 8 \pm 0 . 0 1 }$ </td></tr><tr><td>XAFPAY</td><td>CG-OMatG-IRL</td><td> $2 6 . 9 \pm 0 . 4$ </td><td> $2 4 . 4 \pm 0 . 8$ </td><td> $\mathbf { 0 . 0 8 \pm 0 . 0 1 }$ </td></tr><tr><td>XAFQIH</td><td>CG-OMatG</td><td> ${ \bf 2 7 . 1 \pm 0 . 8 }$ </td><td> ${ \bf 2 3 . 9 \pm 1 . 9 }$ </td><td> ${ \bf 0 . 1 0 \pm 0 . 0 2 }$ </td></tr><tr><td>XAFQIH</td><td>CG-OMatG-IRL</td><td> $2 1 . 5 \pm 1 . 0$ </td><td> $1 2 . 9 \pm 1 . 7$ </td><td> $0 . 2 3 \pm 0 . 0 4$ </td></tr></table>

While CG-OMatG-IRL does exhibit a reduced diversity score, we do not believe that this number is indicative of mode collapse but rather suggests more refined inference with respect to the UMA energy landscape.

## L Loss Curves

![](images/ad1cbbc571623bcda13877221e5727c8e2522c8e2f6c865b2d577ee7ee746cd7.jpg)  
Figure 11: CG-OMatG training and validation losses on the CSD database, decomposed by manifold component: lattice shape $\mathrm { S y m _ { 3 } ^ { + } }$ , cell and molecule orientations on $S O ( 3 )$ , and fractional coordinates on $\mathbb { T } ^ { 3 } .$ . Weight norm is the global L2 norm of all trainable parameters. Note that the constant term is dropped from this loss, so it can be below zero, unlike the usual normalizing-flow presentation.

![](images/41084368897f1613e1772e73bbc1760655567c240523c133382b1196515d4ede.jpg)  
Figure 12: CG-OMatG training and validation losses on the OMC database, decomposed by manifold component: lattice shape $\mathrm { { S y m _ { 3 } ^ { + } } }$ , cell and molecule orientations on $S O ( 3 )$ , and fractional coordinates on ${ \overline { { \mathbb { T } } } } ^ { 3 }$ . Weight norm is the global L2 norm of all trainable parameters. Note that the constant term is dropped from this loss, so it can be below zero, unlike the usual normalizing-flow presentation. Occasional jumps in the loss curves reflect checkpoint restarts after improper GPU resource allocation on the cluster.

## M Data Availability, Preprocessing, and Resources

OMC25-MCF This OMC subset is processed by Zeng et al. [22], as described in Section 4. Specifically, all cocrystals are filtered out, leaving only homomolecular crystals. Subsequently, for each crystal family, the uma-s-1p1 MLIP [78] is used to retain the crystal polymorph with the lowest energy per conformer. After filtering, the dataset is reduced to 46, 120 molecular crystal structures.

CSD The CSD dataset is processed similarly to [37, 39]. First, we filter out any crystal unit cells containing more than 250 heavy atoms. Then, we prescribe that no member of the CSD blind test crystal families may be present in the training data and that the SMILES are indeed valid SMILES strings using RDKit. We ensure that the crystal unit cells possess 3-D coordinates and an R-factor < 0.9. The crystal must have a space group symbol. We resolve disorder by selecting the disorder group with the highest occupancy. Lastly, we split the data into training, validation, and test sets such that crystal polymorphs of the same family belong to the data split. To resolve degeneracy, we compare polymorphs and ensure that the RMSD between them does not fall below 0.25 Å; if it does, we retain the polymorph with the lowest R-factor.

Resources Used All training and inference were carried out on NVIDIA A100 GPUs (80 GB HBM2e) on a shared SLURM cluster. Each training run used 4×A100 in a single node under PyTorch Lightning DDP at FP32 precision. For evaluation, generation is carried out on the same hardware. Downstream CCDC packing-similarity and relaxation-based metrics run on CPU-only nodes (16 cores, 60 GB RAM, ≤ 3 h walltime per dataset), parallelized across structures with a single L40 GPU for UMA energy calculations.