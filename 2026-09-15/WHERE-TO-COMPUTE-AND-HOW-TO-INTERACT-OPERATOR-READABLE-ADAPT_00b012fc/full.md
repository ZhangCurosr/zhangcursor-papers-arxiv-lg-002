# WHERE TO COMPUTE AND HOW TO INTERACT:OPERATOR-READABLE ADAPTATION WITH GAUGE-AWARE TRANSPORT

Zixuan Shen<sup>1∗</sup> Quanxu Wan<sup>1∗</sup> Bingchuan Wang<sup>1</sup> Zhi Wang<sup>2</sup> Biao Luo<sup>1</sup> <sup>1</sup>Central South University <sup>2</sup>Nanjing University

## ABSTRACT

Adaptive meshes enable neural operators for partial differential equations (PDEs) to allocate spatial samples and computational resources according to local physical structures. Existing approaches, however, primarily focus on where to compute, while paying less attention to how to interact after node relocation. Mesh adaptation changes local sampling scales, neighborhood structures, and geometric contexts, making representations formed at different nodes not necessarily directly comparable. Direct aggregation may therefore entangle genuine physical variation with discretization-induced representation variation. Moreover, because allocation and interaction are jointly optimized through the same output objective, their individual roles are difficult to distinguish from final errors alone. We introduce operator readability, which requires an adaptive operator to explicitly account for and test why computation is allocated to particular locations and how representations interact under the resulting nonuniform discretization. Based on this principle, we propose the Gauge-Aware Adaptive Mesh Neural Operator (GA-AMNO). Physics-informed adaptive allocation answers where to compute, while geometry-conditioned low-rank Gauge transport maps source features into target representation contexts before aggregation, answering how to interact. This design turns the otherwise implicit mesh-to-solver information exchange into an inspectable and intervenable computational process. We further establish sufficient conditions for representation-consistent aggregation and analyze approximate transport errors and continuity under topology-preserving mesh deformations. Experiments across five PDE benchmarks demonstrate improved predictive accuracy, while controlled interventions and geometric-mismatch analyses verify the computational roles of allocation and interaction and show that Gauge transport improves cross-discretization representation compatibility under strong geometric mismatch.

## 1 INTRODUCTION

Neural operators aim to learn solution mappings for partial differential equations (PDEs) between function spaces, while their numerical realizations operate on physical fields sampled over grids or meshes (Huang et al., 2025; Kovachki et al., 2023). Adaptive discretization makes this computational representation state dependent by relocating nodes toward dynamically informative regions (Berger & Oliger, 1984). Existing adaptive operators are therefore commonly organized around one question: where should computation be allocated? However, node relocation changes not only the spatial distribution of resolution, but also the local sampling scales, edge directions, neighborhood structures, and mesh Jacobians under which features are formed. The resulting representations must subsequently be exchanged by the operator under these unequal local geometric conditions. Adaptive computation must therefore answer a second question: how should these representations interact? These two decisions are connected through the learned geometry. Allocation determines the local discretization contexts presented to the interaction mechanism, while interaction determines whether the representations created by that allocation can be combined effectively. Making the mesh adaptive without explicitly accounting for the resulting information exchange consequently leaves an important part of adaptive computation unspecified.

![](images/cf98ca3673d422ae4a9b4fcc918b8ab616caaf24f4134861e1162190b82ab4be.jpg)

(b)  
![](images/8511cdf8fe3699600b193abab1db181af3558d644fb555b43f31ca33a77b0079.jpg)

(c)  
![](images/6da61d29c6e9c8294d4f6a605312811d0d5a2f5d6f23765454ee197dd49f0de4.jpg)  
Figure 1: Preliminary diagnosis of the interaction effects induced by adaptive discretization on Navier–Stokes. (a) For the same physical state, increasing the topology-preserving deformation strength λ produces substantially larger representation and prediction discrepancies than input reconstruction discrepancy. (b) Representation drift emerges at direct aggregation and accumulates through subsequent message passing, where $\mathbf { h } ^ { ( \ell ) }$ denotes the node representation at layer ℓ and $\mathbf { m } ^ { ( \ell ) }$ the aggregated message. (c) Message inconsistency increases with local geometric mismatch. Together, these results show that direct aggregation amplifies differences induced by unequal local discretization contexts.

This missing account of information exchange is difficult to expose under standard end-to-end train ing, because allocation and interaction are jointly optimized through the same prediction objective. An expressive solver may compensate for an uninformative mesh, while a strong allocation mechanism may reduce the apparent need for geometry-aware interaction. Different allocation–interaction configurations may therefore achieve similar predictive performance, even though they use the adaptive geometry in substantially different ways. To determine whether this concern has observable consequences, we conduct a controlled diagnosis in Figure 1. We keep the physical state, model parameters, node identities, and neighborhood topology fixed, while progressively deforming only the sampling geometry. A small input reconstruction discrepancy develops into substantially larger representation and prediction discrepancies. Layerwise measurements further locate the onset of this drift at direct message aggregation, while message inconsistency increases with local geometric mismatch. These results show that the problem is not irregular sampling alone; it emerges when representations formed under unequal local discretization contexts are directly combined.

These observations motivate operator readability as an operational and experimentally testable computational criterion. An operator-readable adaptive model should make the roles of its adaptive decisions explicit and allow each role to be examined through controlled interventions. In this work, we use the following conditions as sufficient operational criteria for the two aspects of readability. Allocation readability is established when node movement is explicitly linked to declared physical quantities through a clear computational path and responds predictably to controlled modifications of the allocation signal. Interaction readability is established when source-to-target information transformation is explicitly conditioned on local geometric relations, can be examined at the edge level, and exhibits measurable responses to geometric mismatch and controlled interventions. Readability therefore does not refer to whether the resulting mesh is visually intuitive. It refers to whether the model can account for both why computational resolution is allocated to particular regions and how information is exchanged under the resulting discretization.

Based on this formulation, we propose the Gauge-Aware Adaptive Mesh Neural Operator (GA-AMNO) as one realization of operator-readable adaptation. As summarized in Figure 2, physicsinformed indicators construct an importance map that determines where nodes are relocated, while the resulting relative coordinates, local mesh Jacobians, and state descriptors condition an edge-wise low-rank transport that determines how their features interact. We use Gauge transport as a design principle rather than assume that every discretization change induces an exact gauge transformation. By mapping each source feature into its target representation context before aggregation, it explicitly accounts for differences in local sampling scale, orientation, and neighborhood geometry, making the otherwise implicit mesh-to-solver interaction inspectable. Gauge transport therefore provides a structured realization of interaction readability, while the predictive capability is supplied by the complete GA-AMNO architecture. We further establish sufficient conditions for representationconsistent aggregation, characterize the propagation of approximate transport error, and analyze continuity under topology-preserving mesh deformation.

![](images/7efadfb592587188dba62a14423a076f3f028b77495d2297febc13a91ed2e165.jpg)  
Figure 2: Conceptual overview of operator-readable adaptive computation. Physics-informed allocation explains where adaptive resolution is placed. The resulting nonuniform mesh produces unequal local discretization contexts, under which direct aggregation leaves the effects of geometric mismatch on cross-node interaction implicit. Gauge-aware transport instead maps each source feature into a target-conditioned representation before aggregation, making explicit how information is exchanged after the discretization changes.

The main contributions are summarized as follows:

• Conceptual contribution. We reformulate adaptive neural operator learning as two connected problems: resource allocation and cross-discretization interaction. Based on this formulation, we introduce an operational notion of operator readability and identify, through controlled re-discretization, a reproducible failure mode in which direct aggregation amplifies discretization-induced representation drift.

• Architectural and theoretical contribution. We develop GA-AMNO as a concrete realization of operator-readable adaptation. Physics-informed allocation determines where computational resolution is placed, while geometry-conditioned low-rank transport makes explicit how representations formed under unequal local discretizations interact. We additionally provide consistency conditions, an approximate-transport error bound, and a continuity analysis under topology-preserving mesh deformation.

• Evaluation contribution. We evaluate both predictive utility and mechanism readability across multiple PDE families. Beyond standard prediction benchmarks, allocation counterfactuals, edge interventions, geometric-mismatch stratification, and controlled mesh deformations test whether the allocation and interaction mechanisms fulfill their declared computational roles.

## 2 RELATED WORK

We briefly review the three lines of research most relevant to our work. A more comprehensive discussion is provided in Appendix B.

Neural operators. Neural operators learn mappings between function spaces using spectral transformations, coordinate-conditioned kernels, graph interactions, or physics-aware attention (Li et al., 2020a;b; Wu et al., 2024; Bie et al., 2025). While these methods support regular or irregular discretizations, they do not primarily examine how input-dependent mesh adaptation changes the representation contexts in which features interact.

Adaptive mesh methods. Classical and learned adaptive methods allocate spatial resolution through error indicators, local refinement, node movement, or learned policies (Berger & Oliger, 1984; Berger & Colella, 1989; Rudd et al., 2014; Pfaff et al., 2020). They mainly determine where computation should be concentrated, whereas the interaction between features formed under the resulting nonuniform local discretizations is typically delegated to the downstream solver.

Gauge-aware representations. Gauge-aware geometric networks transform features between local reference frames to support consistent information exchange (Bronstein et al., 2021; Cohen et al., 2019). Motivated by this principle, we use geometry-conditioned transport without assuming exact Gauge equivariance, making feature interaction across adaptive local discretizations explicit before aggregation.

## 3 METHOD

## 3.1 PROBLEM FORMULATION

Let $\Omega \subset \mathbb { R } ^ { d }$ be the spatial domain and let u denote a $C \mathrm { \cdot }$ -channel physical field. We consider PDE systems that can be written in the generic form

$$
\mathcal { F } _ { \mathrm { P D E } } \left( \mathbf { u } , \partial _ { t } \mathbf { u } , \nabla \mathbf { u } , \nabla ^ { 2 } \mathbf { u } ; \pmb { \mu } \right) = 0 .\tag{1}
$$

Here, $\mathcal { F } _ { \mathrm { P D E } }$ is the governing differential operator, $\partial _ { t }$ is the temporal derivative, $\nabla$ and $\nabla ^ { 2 }$ denote first- and second-order spatial derivatives, and $\pmb { \mu }$ collects physical parameters, coefficient fields, or forcing terms. For steady-state PDEs, the temporal derivative is omitted.

The corresponding solution operator is

$$
\mathcal { G } ^ { \dagger } : \mathcal { A }  \mathcal { U } , \qquad \mathbf { u } = \mathcal { G } ^ { \dagger } ( \mathbf { a } ) ,\tag{2}
$$

where $\mathcal { A }$ is the input function space, $\mathcal { U }$ is the solution function space, a denotes the input condition, and $\mathcal { G } ^ { \dagger }$ is the unknown exact operator Kovachki et al. (2023). For temporal prediction, a contains a short history of states; for steady-state prediction, it may contain coefficients, forcing terms, or boundary information.

GA-AMNO learns an approximation

$$
\widehat { \mathbf { u } } = \mathcal G _ { \boldsymbol \theta } ( \mathbf { a } ) \approx \mathcal G ^ { \dagger } ( \mathbf { a } ) ,\tag{3}
$$

where $\widehat { \mathbf { u } }$ is the predicted field and $\mathcal { G } _ { \theta }$ is the neural operator with trainable parameters $\theta .$

The input field is first represented on a reference discretization $\mathcal X ^ { r } = \{ \mathbf x _ { i } ^ { r } \} _ { i = 1 } ^ { N }$ . GA-AMNO then constructs an input-dependent adaptive mesh

$$
\begin{array} { r } { \mathcal { X } ^ { a } = \{ \mathbf { x } _ { i } ^ { a } = \Pi _ { \Omega } \left( \mathbf { x } _ { i } ^ { r } + \Delta \mathbf { x } _ { i } \right) \} _ { i = 1 } ^ { N } . } \end{array}\tag{4}
$$

Here, $\mathcal { X } ^ { a }$ is the adaptive mesh, $\mathbf { x } _ { i } ^ { r }$ and $\mathbf { x } _ { i } ^ { a }$ are the reference and adaptive coordinates of node i, N is the number of nodes, $\Delta \mathbf { x } _ { i } \in \mathbb { R } ^ { d }$ is the learned displacement, and $\Pi _ { \Omega }$ keeps the displaced node inside the computational domain.

The adaptive coordinates define an input-dependent local discretization environment for each node. Once constructed for an input a, the adaptive mesh ${ \mathcal { X } } ^ { a } ( \mathbf { a } )$ remains fixed during the subsequent operator computation but may vary across inputs, leading to changes in local spacing, neighborhood geometry, and mesh Jacobians. Consequently, nodal features describing related physical content may be formed under different local discretization environments and need not be directly comparable. To address this mismatch, GA-AMNO does not assume an analytically specified coordinate transformation. Instead, it learns local feature-transport mappings that are conditioned on mesh geometry and input descriptors and are applied before aggregation to mitigate the resulting representation offsets.

![](images/77b84105585259fc360df6d1e65abdbe13fa067844950b3adcb016adfe8fc9e0.jpg)  
Figure 3: Overall architecture of GA-AMNO. Physics-informed indicators guide adaptive allocation, low-rank gauge transport aligns features before aggregation on the resulting mesh, and the reconstructed grid features are refined by spectral and differential residual corrections.

## 3.2 OVERALL ARCHITECTURE

Given an input a on $\mathcal { X } ^ { r }$ , GA-AMNO encodes local state descriptors and physics-informed indicators, predicts an importance map, and constructs ${ \mathcal { X } } ^ { a } ( \mathbf { a } )$ . Gauge-aware operator layers then use adaptive coordinates, local mesh Jacobians, and state descriptors to transport each source feature into its target context before aggregation. After the nodal updates, a mesh-to-grid bridge reconstructs the adaptive features on the reference grid, where spectral and differential residual corrections produce the final output, as summarized in Figure 3. The complete forward procedure is given in Appendix A, while the reconstruction details are provided in Section D.3.

## 3.3 PHYSICS-INFORMED ADAPTIVE ALLOCATION

This module determines where representation capacity is allocated. Given the latest input slice v, we compute four physically meaningful indicators: the gradient magnitude $Q _ { g } ,$ Laplacian magnitude $Q _ { \Delta }$ , local energy $Q _ { E }$ , and vorticity-related response $\bar { Q } _ { \omega }$ . Their complete definitions and scalar-field treatment are provided in Appendix D.1. The indicators are combined with the learned spatial state descriptor P = Encoder(a) to predict the importance map:

$$
\begin{array} { r l } & { \widetilde { \mathbf { Q } } ( \mathbf { x } ) = \mathrm { C o n c a t } ( \mathcal { N } ( Q _ { g } ) , \mathcal { N } ( Q _ { \Delta } ) , \mathcal { N } ( Q _ { E } ) , \mathcal { N } ( Q _ { \omega } ) ) ( \mathbf { x } ) , } \\ & { I ( \mathbf { x } ) = \sigma \left( \eta _ { \theta } \left( \mathrm { C o n c a t } ( \mathbf { P } , \widetilde { \mathbf { Q } } ) \right) ( \mathbf { x } ) \right) . } \end{array}\tag{5}
$$

Here, x is a spatial location, $\widetilde { \mathbf { Q } } ( \mathbf { x } )$ is the concatenated physical-indicator vector at x, $\mathcal { N } ( \cdot )$ denotes per-sample channel normalization, and $\mathrm { { C o n c a t } ( \cdot ) }$ denotes channel-wise concatenation. The quantity $I ( \mathbf { x } )$ is the scalar importance value, η<sub>θ</sub> is a convolutional predictor parameterized by θ, and σ is the sigmoid function. The input a and its encoded descriptor P have been defined above.

The predicted importance modulates a bounded node displacement:

$$
\begin{array} { l } { \displaystyle { \mathbf { o } _ { i } = \operatorname { t a n h } ( \psi _ { \theta } ( \mathbf { P } ) _ { i } ) , } } \\ { \displaystyle { w _ { i } = \frac { I _ { \operatorname* { m i n } } + s _ { I } I _ { i } } { N ^ { - 1 } \sum _ { k = 1 } ^ { N } ( I _ { \operatorname* { m i n } } + s _ { I } I _ { k } ) + \epsilon } , } } \\ { \Delta \mathbf { x } _ { i } = \delta _ { \operatorname* { m a x } } w _ { i } \mathbf { o } _ { i } . } \end{array}\tag{6}
$$

Here, $\mathbf { o } _ { i }$ is the bounded raw displacement of node i, ψ is the convolutional displacement head, and $I _ { i } = I ( \mathbf { x } _ { i } ^ { r } )$ is the importance value at that node. The quantity $w _ { i }$ is its positive mean-normalized importance weight, $I _ { \operatorname* { m i n } } > 0$ prevents complete suppression of a node, $s _ { I }$ controls the modulation range, k indexes the $N$ nodes, and $\epsilon > 0$ ensures numerical stability. Finally, $\delta _ { \mathrm { m a x } }$ bounds the displacement scale and $\Delta { { \bf { x } } _ { i } }$ is the resulting displacement. The reference coordinate $\mathbf { x } _ { i } ^ { r }$ and the number of nodes N have been defined in Section 3.1.

The resulting path $Q _ { g } , Q _ { \Delta } , Q _ { E } , Q _ { \omega }  I ( { \bf x } )  w _ { i }  \Delta { \bf x } _ { i }$ explicitly connects physical structure to node relocation, thereby supporting allocation readability. The nonuniform local discretizations produced by this allocation motivate the interaction mechanism introduced next.

## 3.4 GAUGE-AWARE MESH OPERATOR

Adaptive allocation changes local spacing, orientation, neighborhood geometry, and mesh Jacobians. GA-AMNO therefore applies an edge-conditioned feature transport before aggregating information from neighboring adaptive nodes. For each directed edge $j \  \ i ,$ the edge descriptor is

$$
\mathbf { e } _ { i j } = \mathrm { C o n c a t } \left( \mathbf { x } _ { j } ^ { a } - \mathbf { x } _ { i } ^ { a } , \mathrm { v e c } ( \mathbf { J } _ { j } ^ { a } ) , \mathrm { v e c } ( \mathbf { J } _ { i } ^ { a } ) , \mathbf { p } _ { j } , \mathbf { p } _ { i } \right) .\tag{7}
$$

Here, ${ \bf e } _ { i j }$ is the descriptor of the edge from source node $j$ to target node $i ,$ and $\mathbf { x } _ { i } ^ { a } - \mathbf { x } _ { i } ^ { a }$ is their relative adaptive coordinate. The matrices $\mathbf { J } _ { j } ^ { a }$ and $\mathbf { J } _ { i } ^ { a }$ are the estimated local mesh Jacobians at the source and target nodes, respectively, $\mathrm { v e c } ( \cdot )$ flattens a matrix into a vector, and $\mathbf { p } _ { j }$ and $\mathbf { p } _ { i }$ are their learned state descriptors. The adaptive coordinates $\mathbf { x } _ { i } ^ { a }$ have been defined in Section 3.1.

The transport uses a diagonal map with a gated low-rank correction:

$$
\mathbf { T } _ { i  j } ^ { ( \ell ) } = \mathrm { d i a g } ( \mathbf { d } _ { i j } ^ { ( \ell ) } ) + \frac { \gamma _ { i j } ^ { ( \ell ) } } { \sqrt { C _ { h } r } } \mathbf { U } _ { i j } ^ { ( \ell ) } ( \mathbf { V } _ { i j } ^ { ( \ell ) } ) ^ { \top } .\tag{8}
$$

Here, $\mathbf { T } _ { i  j } ^ { ( \ell ) } \in \mathbb { R } ^ { C _ { h } \times C _ { h } }$ is the source-to-target transport matrix in layer $\ell , C _ { h }$ is the hidden feature width, and $r$ is the transport rank. The vector $\mathbf { d } _ { i j } ^ { ( \ell ) } \in \mathbb { R } ^ { C _ { h } }$ determines diagonal channel scaling, $\mathbf { U } _ { i j } ^ { ( \ell ) } , \mathbf { V } _ { i j } ^ { ( \ell ) } \in \mathbb { R } ^ { C _ { h } \times r }$ determine the low-rank correction, and $\gamma _ { i j } ^ { ( \ell ) }$ is an edge-conditioned sigmoid gate. The superscript ⊤ denotes matrix transpose. These quantities are predicted from $\mathbf { e } _ { i j }$ by separate learned heads.

Each source feature is transported into its target-conditioned representation before weighted aggregation:

$$
\begin{array} { r l } & { \alpha _ { i j } ^ { ( \ell ) } = \frac { \exp ( a _ { i j } ^ { ( \ell ) } ) } { \sum _ { k \in \mathcal { N } ( i ) } \exp ( a _ { i k } ^ { ( \ell ) } ) } , } \\ & { \mathbf { m } _ { i } ^ { ( \ell ) } = \displaystyle \sum _ { j \in \mathcal { N } ( i ) } \alpha _ { i j } ^ { ( \ell ) } \mathbf { T } _ { i  j } ^ { ( \ell ) } \mathbf { h } _ { j } ^ { ( \ell ) } . } \end{array}\tag{9}
$$

Here, $a _ { i j } ^ { ( \ell ) }$ is the learned compatibility score of edge $j  i ,$ , and $\alpha _ { i j } ^ { ( \ell ) }$ is its normalized attention weight. The set $\mathcal { N } ( i )$ contains the neighbors of target node $i ,$ while k indexes these neighbors in the softmax denominator. The vector $\mathbf { h } _ { i } ^ { ( \ell ) }$ is the source-node feature in layer $\ell ,$ and $\mathbf { m } _ { i } ^ { ( \ell ) }$ is the transported message aggregated at target node i.

The target feature is then updated through

$$
\mathbf { h } _ { i } ^ { ( \ell + 1 ) } = \mathbf { h } _ { i } ^ { ( \ell ) } + \rho _ { \theta } ^ { ( \ell ) } \left( \mathbf { m } _ { i } ^ { ( \ell ) } \right) .\tag{10}
$$

Here, $\mathbf { h } _ { i } ^ { ( \ell ) }$ and $\mathbf { h } _ { i } ^ { ( \ell + 1 ) }$ are the target-node features before and after the update, respectively, and $\rho _ { \theta } ^ { ( \ell ) }$ is the layer-specific update network. The residual form preserves the current target feature while incorporating transported neighborhood information.

The transport is interpreted as a learned, geometry-conditioned feature correction rather than an analytically prescribed coordinate transformation. Its representation-level motivation and complete implementation details are provided in Appendix D.2, while its theoretical analysis is given in Appendix E. The subsequent residual prediction is described in Appendix D.4.

Table 1: Relative $\ell _ { 2 }$ error (lower is better). All the models report mean ± standard deviation over three seeds. Classic denotes adaptive mesh learning with direct aggregation and the shared base solver. Best results are bold.
<table><tr><td>Dataset</td><td>GA-AMNO</td><td>Classic</td><td>MMPDE</td><td>Transolver</td><td>FNO</td><td>GNO</td><td>WNO</td><td>UNet</td></tr><tr><td>Navier-Stokes</td><td> $\mathbf { \overline { { 0 . 0 0 3 7 9 4 \pm 0 . 0 0 0 6 2 4 } } }$ </td><td> $\overline { { 0 . 0 1 1 1 7 5 \pm 0 . 0 0 0 0 6 5 } }$ </td><td> $\overline { { 0 . 0 0 5 3 0 0 \pm 0 . 0 0 0 5 3 5 } }$ </td><td>0.019768</td><td>0.006781</td><td>0.033800</td><td>0.016903</td><td>0.005054</td></tr><tr><td>Rayleigh-Bénard</td><td> $\mathbf { 0 . 0 0 0 9 2 7 \pm 0 . 0 0 0 3 6 3 }$ </td><td> $0 . 0 0 1 4 7 2 \pm 0 . 0 0 0 2 1 3$ </td><td> $0 . 0 0 3 0 3 2 \pm 0 . 0 0 0 4 7 7$ </td><td>0.002664</td><td>0.006327</td><td>0.003954</td><td>0.007910</td><td>0.010840</td></tr><tr><td>KS</td><td> $\mathbf { 0 . 0 0 0 3 8 0 \pm 0 . 0 0 0 0 6 1 }$ </td><td> $0 . 0 0 0 8 9 1 \pm 0 . 0 0 0 1 2 8$ </td><td> $0 . 0 0 3 9 8 8 \pm 0 . 0 0 1 1 8 4$ </td><td>0.013923</td><td>0.009289</td><td>0.010016</td><td>0.009549</td><td>0.010877</td></tr><tr><td>Kolmogorov</td><td> $\mathbf { 0 . 0 2 0 2 2 1 } \pm \mathbf { 0 . 0 0 1 5 2 9 }$ </td><td> $0 . 0 8 1 6 7 1 \pm 0 . 0 0 1 8 6 0$ </td><td> $0 . 0 6 9 6 1 4 \pm 0 . 0 0 3 0 3 1$ </td><td>0.070090</td><td>0.030034</td><td>0.090519</td><td>0.121220</td><td>0.028476</td></tr><tr><td>Darcy</td><td> $\mathbf { 0 . 0 2 8 1 5 7 \pm 0 . 0 0 1 7 0 1 }$ </td><td> $0 . 3 4 1 6 1 4 \pm 0 . 0 0 4 7 7 3$ </td><td> $0 . 0 3 0 0 8 2 \pm 0 . 0 0 1 9 0 7$ </td><td>0.046329</td><td>0.028557</td><td>0.254866</td><td>0.118636</td><td>0.030238</td></tr></table>

![](images/a7df158087e6962bfc9c7f1bc0e22b11d8a0468ef1a97383f11f21ed95d28b97.jpg)  
Figure 4: Ablation of the two core mechanisms. Identity Transport retains the adaptive mesh but removes learned representation transport, while w/o Adaptive Mesh retains the remaining solver without learned node relocation. Lower test relative $\ell _ { 2 }$ error is better.

## 4 EXPERIMENTAL RESULTS

## 4.1 OVERALL PREDICTIVE PERFORMANCE

Table 1 compares GA-AMNO with seven representative baselines covering spectral operators, graph operators, convolutional architectures, transformer-based operators, and adaptive-mesh solvers. GA-AMNO achieves the lowest central relative $\ell _ { 2 }$ error on all five benchmarks. This consistent advantage across substantially different computational paradigms demonstrates that the proposed architecture is not specialized to outperform only one particular operator family. Instead, physicsinformed adaptive allocation, low-rank gauge transport, and residual field refinement form an effective operator-learning framework that improves solution approximation across regular-grid, graphbased, and adaptive-mesh alternatives.

The comparisons with Classic and MMPDE are particularly important to the central claim of this work. Classic uses an adaptive mesh with direct feature aggregation, while MMPDE introduces a dedicated moving-mesh mechanism. GA-AMNO consistently outperforms both, showing that determining where nodes should be allocated is only part of successful adaptive operator learning. Explicitly transporting source features into the target representation context before aggregation provides an additional and practically useful capability for processing heterogeneous local discretizations. The strong results across Navier-Stokes, Rayleigh-Benard, KS, Kolmogorov, and Darcy further´ demonstrate that this benefit extends from evolutionary and chaotic dynamics to static coefficientto-solution mappings. Overall, the main experiment supports the central design principle of operator readability: the predictive benefit emerges when adaptive discretization is operator-readable, while broad applicability is retained across distinct PDE regimes.

## 4.2 COMPONENT ABLATION

We isolate the two mechanisms central to operator-readable adaptation while retaining the remaining architecture and evaluation protocol. Identity Transport replaces every learned source-to-target transport with the identity map, whereas w/o Adaptive Mesh removes the learned node relocation. Figure 4 reports the resulting test relative $\ell _ { 2 }$ errors on Navier-Stokes, Kolmogorov, and Darcy.

Full GA-AMNO achieves the lowest error on all three datasets. Replacing learned transport with the identity increases error, most substantially on Kolmogorov, while removing adaptive allocation also consistently degrades performance. The comparison directly verifies that both where representations are allocated and how they are aligned contribute to the complete model. Targeted ablations of the remaining components are provided in Figure 8 in the extended results.

![](images/8e2cd50c3e33306ce01efad8ad93daa3f904291d40d84ed7c2221d41b2018e52.jpg)

![](images/ce47b581552c2a5d10a02d2729a4620bcde9b8c59b952bf2918b8962ebc06ef0.jpg)  
Figure 5: Causal tests of allocation and interaction readability. (a) Relative $\ell _ { 2 }$ increase after replacing the learned importance map with uniform, shuffled, or shifted alternatives. (b) Prediction change after replacing transport with the identity on top-ranked, random, or bottom-ranked edge subsets.

## 4.3 INTERPRETING ADAPTIVE COMPUTATION

This subsection examines whether the learned importance map is used by the predictor, whether the learned connection identifies influential edges, and whether transport converts source messages into target-compatible representations under strong local geometric mismatch. All experiments use frozen checkpoints and held-out samples, with no parameter updates during evaluation. Predic tion effects in high-adaptation-demand regions are reported in the targeted component ablation of Figure 8, avoiding duplicate tabulation of the same quantities.

## 4.3.1 IMPORTANCE ALLOCATION FAITHFULNESS

We intervene on the learned importance map while leaving the remaining network unchanged. The learned map is replaced by a spatially uniform map, randomly shuffled within each sample, or cyclically shifted. If the map merely visualized activity without controlling useful computation, these counterfactuals would leave the prediction nearly unchanged.

As shown in Figure 5(a), the learned map is best in all three cases, but the impact varies across PDEs. Navier-Stokes exhibits a strong counterfactual gap, Darcy exhibits a moderate gap, and Kolmogorov is only weakly sensitive. The counterfactual design is important because it distinguishes a visually plausible importance map from one that actually controls prediction: preserving its values while destroying their spatial correspondence degrades performance. This experiment therefore supports allocation readability by showing that the learned spatial allocation is functionally used, although it does not prove that every highlighted region has a unique physical interpretation or that importance identifiability is equally strong for every PDE.

## 4.3.2 GAUGE EDGE FAITHFULNESS

Using the correction magnitude and prediction-change metric defined in Appendix C, we replace the learned transport on a selected 10% of edges by the identity map. Top-ranked, random, and bottom-ranked edge subsets are evaluated under the same intervention budget.

As shown in Figure 5(b), top-ranked interventions produce 2.99–9.92 times the prediction change of random interventions, while bottom-ranked edges have little influence. The correction magnitude is therefore functionally faithful to the trained operator: edges assigned a large representation repair are also the edges on which the output causally depends. The purpose of this intervention is to establish a causal link between an interpretable internal quantity and the final prediction, which cannot be obtained from feature visualization alone. It supports interaction readability by showing that the learned connection identifies consequential information transfers rather than acting as a decorative latent variable. The correction magnitude is not itself claimed to be a physical observable.

![](images/ea03d318c00dae0bc9b55b3fadfc4e96096e893197d16903fa3fcbad056e32c3.jpg)

![](images/7ab1893af81470186f795ffb0115342c455264b7bed776662e3df88ca13a00bb.jpg)

![](images/b917aba878af7bb8b2365f44e1ead8a16197f71139b926d664e6c9b4d39fc987.jpg)  
Figure 6: Source–target and transported-message–target discrepancy stratified by local geometric mismatch. The reduction measures message compatibility before aggregation, not the covariance defect of the learned connection.

## 4.3.3 MESSAGE COMPATIBILITY BY LOCAL GEOMETRIC MISMATCH

We use the geometric-mismatch groups and compatibility metrics defined in Appendix C to compare source–target discrepancy with transported-message–target discrepancy. This evaluation asks whether the learned correction is most active where the adaptive geometry is most nonuniform; it does not by itself separate physical feature variation from discretization-induced variation.

As summarized in Table 4, transport reduces message–target discrepancy by 48.32%–81.93% in the high-geometric-mismatch group. The full stratification in Figure 6 shows that this compatibility gain is concentrated where the adaptive geometry is most nonuniform. This concentration is the key mechanism-level result: the learned correction is strongest where direct comparison is least justified by the local discretization, which is the regime targeted by interaction readability. Together with the edge intervention and regional prediction ablation, the result connects geometric mismatch, feature correction, and output utility. Because the metric also contains actual source–target physical variation, it remains a compatibility diagnostic rather than a standalone proof of gauge covariance.

## 5 CONCLUSION

We introduced GA-AMNO to address a structural issue in adaptive neural operators: state-dependent discretization changes not only where information is sampled, but also the local representation context in which information is encoded and exchanged. GA-AMNO therefore treats adaptive computation as two coupled problems, using physics-informed adaptive allocation to determine where resolution should be allocated and low-rank gauge transport to determine how features associated with unequal local discretizations should interact before aggregation. The theoretical analysis explains why shared direct aggregation cannot generally remove discretization-induced representation ambiguity, establishes sufficient conditions for consistent transported aggregation, and bounds the effect of approximate learned connections without claiming exact gauge equivariance. The counterfactual, intervention, mismatch-stratified, and deformation experiments further show that the learned allocation and transport are functionally used rather than merely producing visually plausible meshes or latent variables. The broader significance of this work is to reposition adaptive discretization from an opaque preprocessing or resource-allocation mechanism into a mechanism that operationalizes operator readability, with allocation decisions and information transfers that can both be inspected and tested. Future work should extend this formulation to explicitly equivariant connection parameterizations, topology-changing and anisotropic mesh adaptation, three-dimensional unstructured domains, and scalable sparse transport, while developing stronger links between representation consistency, numerical approximation error, and the convergence properties of learned PDE operators.

## REFERENCES

Ferran Alet, Adarsh K. Jeewajee, Maria Bauza Villalonga, Alberto Rodriguez, Tomas Lozano-Perez, and Leslie P. Kaelbling. Graph element networks: Adaptive, structured computation and memory. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings ofMachine Learning Research, pp. 212–222. PMLR, 2019.

Francesca Bartolucci, Emmanuel de Bezenac, Bogdan Raonic, Roberto Molinaro, Siddhartha Mishra, and Rima Alaifari. Representation equivalent neural operators: a framework for aliasfree operator learning. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=7LSEkvEGCM.

Marsha J Berger and Phillip Colella. Local adaptive mesh refinement for shock hydrodynamics. Journal ofcomputational Physics, 82(1):64–84, 1989.

Marsha J Berger and Joseph Oliger. Adaptive mesh refinement for hyperbolic partial differential equations. Journal ofcomputational Physics, 53(3):484–512, 1984.

Pengfei Bie, Ning Song, Nuoqing Zhang, Jie Nie, Min Ye, Xinyue Liang, and Qi Wen. Space– frequency cross-attention node feature optimization graph neural operator for partial differential equations. IEEE Transactions on Neural Networks and Learning Systems, 2025.

Johannes Brandstetter, Daniel E. Worrall, and Max Welling. Message passing neural PDE solvers. In International Conference on Learning Representations, 2022.

Michael M Bronstein, Joan Bruna, Taco Cohen, and Petar Velickovi ˇ c. Geometric deep learning:´ Grids, groups, graphs, geodesics, and gauges. arXiv preprint arXiv:2104.13478, 2021.

Keaton J Burns, Geoffrey M Vasil, Jeffrey S Oishi, Daniel Lecoanet, and Benjamin P Brown. Dedalus: A flexible framework for numerical simulations with spectral methods. Physical Review Research, 2(2):023068, 2020.

Taco Cohen, Maurice Weiler, Berkay Kicanaoglu, and Max Welling. Gauge equivariant convolutional networks and the icosahedral cnn. In International conference on Machine learning, pp. 1321–1330. PMLR, 2019.

Filipe de Avila Belbute-Peres, Thomas Economon, and Zico Kolter. Combining differentiable PDE solvers and graph neural networks for fluid flow prediction. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pp. 2402–2411. PMLR, 2020.

Pim de Haan, Maurice Weiler, Taco Cohen, and Max Welling. Gauge equivariant mesh CNNs: Anisotropic convolutions on geometric graphs. In International Conference on Learning Representations, 2021.

Carl Eckart and Gale Young. The approximation of one matrix by another of lower rank. Psychometrika, 1(3):211–218, 1936.

Fabian B. Fuchs, Daniel E. Worrall, Volker Fischer, and Max Welling. SE(3)-transformers: 3d roto-translation equivariant attention networks. In Advances in Neural Information Processing Systems, volume 33, 2020.

John Guibas, Morteza Mardani, Zongyi Li, Andrew Tao, Anima Anandkumar, and Bryan Catanzaro. Efficient token mixing for transformers via adaptive fourier neural operators. In International Conference on Learning Representations, 2022.

Zhongkai Hao, Zhengyi Wang, Hang Su, Chengyang Ying, Yinpeng Dong, Songming Liu, Ze Cheng, Jian Song, and Jun Zhu. GNOT: A general neural operator transformer for operator learning. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 12556–12569. PMLR, 2023.

Lingshen He, Yiming Dong, Yisen Wang, Dacheng Tao, and Zhouchen Lin. Gauge equivariant transformer. Advances in Neural Information Processing Systems, 34:27331–27343, 2021.

Jacob Helwig, Xuan Zhang, Cong Fu, Jerry Kurtin, Stephan Wojtowytsch, and Shuiwang Ji. Group equivariant fourier neural operators for partial differential equations. arXiv preprint arXiv:2306.05697, 2023.

Maximilian Herde, Bogdan Raonic, Tobias Rohner, Roger K´ appeli, Roberto Molinaro, Emmanuel¨ De Bezenac, and Siddhartha Mishra. Poseidon: Efficient foundation models for pdes. Advances in Neural Information Processing Systems, 37:72525–72624, 2024.

Peiyan Hu, Yue Wang, and Zhi-Ming Ma. Better neural pde solvers through data-free mesh movers. In International Conference on Learning Representations, volume 2024, pp. 4550–4576, 2024.

Shudong Huang, Wentao Feng, Chenwei Tang, Zhenan He, Caiyang Yu, and Jiancheng Lv. Partial differential equations meet deep neural networks: A survey. IEEE Transactions on Neural Networks and Learning Systems, 36(8):13649–13669, 2025. doi: 10.1109/TNNLS.2025.3545967.

Erik Jenner and Maurice Weiler. Steerable partial differential operators for equivariant neural networks. arXiv preprint arXiv:2106.10163, 2021.

Nikola Kovachki, Zongyi Li, Burigede Liu, Kamyar Azizzadenesheli, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Neural operator: Learning maps between function spaces with applications to PDEs. Journal ofMachine Learning Research, 24(89):1–97, 2023.

Zijie Li, Saurabh Patil, Francis Ogoke, Dule Shu, Wilson Zhen, Michael Schneier, John R Buchanan Jr, and Amir Barati Farimani. Latent neural pde solver: A reduced-order modeling framework for partial differential equations. Journal of Computational Physics, 524:113705, 2025.

Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial differential equations. arXiv preprint arXiv:2010.08895, 2020a.

Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Neural operator: Graph kernel network for partial differential equations. arXiv preprint arXiv:2003.03485, 2020b.

Zongyi Li, Daniel Zhengyu Huang, Burigede Liu, and Anima Anandkumar. Fourier neural operator with learned deformations for PDEs on general geometries. Journal of Machine Learning Research, 24(388):1–26, 2023a.

Zongyi Li, Nikola Kovachki, Chris Choy, Boyi Li, Jean Kossaifi, Shourya Otta, Mohammad Amin Nabian, Maximilian Stadler, Christian Hundt, Kamyar Azizzadenesheli, et al. Geometryinformed neural operator for large-scale 3d pdes. Advances in Neural Information Processing Systems, 36:35836–35854, 2023b.

Zongyi Li, Hongkai Zheng, Nikola Kovachki, David Jin, Haoxuan Chen, Burigede Liu, Kamyar Azizzadenesheli, and Anima Anandkumar. Physics-informed neural operator for learning partial differential equations. ACM/IMS Journal of Data Science, 1(3):1–27, 2024. doi: 10.1145/3648506.

Phillip Lippe, Bas Veeling, Paris Perdikaris, Richard Turner, and Johannes Brandstetter. Pde-refiner: Achieving accurate long rollouts with neural pde solvers. Advances in Neural Information Processing Systems, 36:67398–67433, 2023.

Shouyi Liu, Xiaokang Yang, and Yuntian Chen. Riesz neural operator for solving partial differential equations. In The Fourteenth International Conference on Learning Representations, 2026.

Lu Lu, Pengzhan Jin, Guofei Pang, Zhongqiang Zhang, and George Em Karniadakis. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature Machine Intelligence, 3(3):218–229, 2021. doi: 10.1038/s42256-021-00302-5.

Leon Mirsky. Symmetric gauge functions and unitarily invariant norms. The Quarterly Journal of Mathematics, 11(1):50–59, 1960.

Jung Yeon Park, Lawson Wong, and Robin Walters. Modeling dynamics over meshes with gauge equivariant nonlinear message passing. Advances in Neural Information Processing Systems, 36: 15277–15302, 2023.

Tobias Pfaff, Meire Fortunato, Alvaro Sanchez-Gonzalez, and Peter W Battaglia. Learning meshbased simulation with graph networks. arXiv preprint arXiv:2010.03409, 2020.

Md Ashiqur Rahman, Zachary E. Ross, and Kamyar Azizzadenesheli. U-NO: U-shaped neural operators. Transactions on Machine Learning Research, 2023.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computerassisted intervention, pp. 234–241. Springer, 2015.

Franc¸ois Rozet and Gilles Louppe. Score-based data assimilation. Advances in Neural Information Processing Systems, 36:40521–40541, 2023.

Keith Rudd, Gianluca Di Muro, and Silvia Ferrari. A constrained backpropagation approach for the adaptive solution of partial differential equations. IEEE Transactions on Neural Networks and Learning Systems, 25(3):571–584, 2014. doi: 10.1109/TNNLS.2013.2277601.

Alvaro Sanchez-Gonzalez, Jonathan Godwin, Tobias Pfaff, Rex Ying, Jure Leskovec, and Peter W. Battaglia. Learning to simulate complex physics with graph networks. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pp. 8459–8468. PMLR, 2020.

Victor Garcia Satorras, Emiel Hoogeboom, and Max Welling. E(n) equivariant graph neural networks. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pp. 9323–9332. PMLR, 2021.

Aliaksandra Shysheya, Cristiana Diaconu, Federico Bergamin, Paris Perdikaris, Jose M Hern´ andez-´ Lobato, Richard E Turner, and Emile Mathieu. On conditional diffusion models for PDE simulations. Advances in Neural Information Processing Systems, 37:23246–23300, 2024.

Wenbin Song, Mingrui Zhang, Joseph G Wallwork, Junpeng Gao, Zheng Tian, Fanglei Sun, Matthew Piggott, Junqing Chen, Zuoqiang Shi, Xiang Chen, et al. M2n: Mesh movement networks for pde solvers. Advances in Neural Information Processing Systems, 35:7199–7210, 2022.

Alasdair Tran, Alexander Mathews, Lexing Xie, and Cheng Soon Ong. Factorized fourier neural operators. In International Conference on Learning Representations, 2023.

Tapas Tripura and Souvik Chakraborty. Wavelet neural operator for solving parametric partial differential equations in computational mechanics problems. Computer Methods in Applied Mechanics and Engineering, 404:115783, 2023. doi: 10.1016/j.cma.2022.115783.

Zhichao Wang, Xinhai Chen, Qinglin Wang, Xiang Gao, Qingyang Zhang, Menghan Jia, Xiang Zhang, and Jie Liu. Ugm2n: An unsupervised and generalizable mesh movement network via m-uniform loss. arXiv preprint arXiv:2508.08615, 2025.

Maurice Weiler, Mario Geiger, Max Welling, Wouter Boomsma, and Taco S. Cohen. 3d steerable CNNs: Learning rotationally equivariant features in volumetric data. In Advances in Neural Information Processing Systems, volume 31, 2018.

Haixu Wu, Huakun Luo, Haowen Wang, Jianmin Wang, and Mingsheng Long. Transolver: A fast transformer solver for pdes on general geometries. arXiv preprint arXiv:2402.02366, 2024.

Tailin Wu, Takashi Maruyama, Qingqing Zhao, Gordon Wetzstein, and Jure Leskovec. Learning controllable adaptive simulation for multi-resolution physics. arXiv preprint arXiv:2305.01122, 2023.

## A ALGORITHMIC SUMMARY

Table 2 summarizes the forward procedure of GA-AMNO. All learnable components are optimized jointly during training.

## B EXPANDED RELATED WORK

## B.1 NEURAL OPERATORS FOR PDE SOLVING

Neural operators learn mappings between function spaces and provide a practical route to surrogate PDE solvers that can share parameters across discretization resolutions Huang et al. (2025); Kovachki et al. (2023). Deep Operator Network (DeepONet) represents an input function and its query coordinates through branch and trunk networks Lu et al. (2021), whereas Fourier Neural Operator (FNO) parameterizes global integral operators in the spectral domain Li et al. (2020a). This basic formulation has subsequently been extended through factorized spectral layers Tran et al. (2023), adaptive Fourier token mixing Guibas et al. (2022), U-shaped multiscale architectures Rahman et al. (2023), wavelet decompositions Tripura & Chakraborty (2023), and physics-informed objectives Li et al. (2024). Graph Neural Operator (GNO) Li et al. (2020b) and Message Passing Neural PDE Solvers Brandstetter et al. (2022) support local interactions on irregular samples, while Geo-FNO Li et al. (2023a) learns a deformation between irregular physical domains and a regular latent grid. Geometry-Informed Neural Operator further combines graph-based lifting with spectral processing for large-scale PDEs on varying geometries Li et al. (2023b). Transformer-based operators, including GNOT Hao et al. (2023) and Transolver Wu et al. (2024), model long-range physical interactions, while explicit space–frequency coupling improves the recovery of multiscale solution features Bie et al. (2025). More recent work has addressed stable long-horizon prediction through iterative spectral refinement Lippe et al. (2023), general-purpose PDE pretraining through the Poseidon foundation model Herde et al. (2024), and the integration of global spectra with local derivative information through the Riesz Neural Operator Liu et al. (2026). Despite these advances, the numerical realization of a neural operator still acts on discretized field representations. Representation Equivalent Neural Operators show that changes in sampling and reconstruction can introduce operator aliasing and break consistency between continuous operators and their discrete realizations Bartolucci et al. (2023). This observation is closely related to our motivation, but our focus is different: rather than addressing global continuous–discrete equivalence alone, we study how features produced under unequal, input-dependent local discretizations should interact inside an adaptive-mesh operator.

<table><tr><td colspan="2">Table 2: Forward procedure of the proposed GA-AMNO.</td></tr><tr><td>Step</td><td>Operation</td></tr><tr><td>Input</td><td>Receive the temporal history or steady conditioning field a on the reference discretization  $\mathcal { X } ^ { r }$ </td></tr><tr><td>1</td><td>Encode the input into a state descriptor:  $\mathbf { P } = \operatorname { E n c o d e r } ( \mathbf { a } )$ </td></tr><tr><td>2</td><td>Compute gradient, Laplacian, energy, and vorticity-related indicators from the latest or con- ditioning field v.</td></tr><tr><td>3</td><td>Predict the importance map from the encoded state and normalized physical indicators:  $I =$   $\sigma \Bigl ( \eta _ { \theta } ( \mathrm { C o n c a t } ( { \bf P } , \widetilde { { \bf Q } } ) ) \Bigr )$ </td></tr><tr><td>4</td><td>Predict bounded node directions and construct importance-modulated adaptive coordinates:  $\mathbf { x } _ { i } ^ { a } = \Pi _ { \Omega } \big ( \mathbf { x } _ { i } ^ { r } + \delta _ { \operatorname* { m a x } } w _ { i } \mathbf { o } _ { i } \big )$ </td></tr><tr><td>5</td><td>Estimate local mesh Jacobians  $\mathbf { J } _ { i } ^ { a }$  and form each directed edge descriptor 4  $\mathbf { e } _ { i j }$  from relative coordinates, source and target Jacobians, and source and target state descriptors.</td></tr><tr><td>6</td><td>Predict the edge attention score and diagonal-plus-low-rank transport:  $\mathbf { T } _ { i  j } ^ { ( \ell ) } = \mathrm { d i a g } ( \mathbf { d } _ { i j } ^ { ( \ell ) } ) +$   $\gamma _ { i j } ^ { ( \ell ) } \mathbf { U } _ { i j } ^ { ( \ell ) } ( \mathbf { V } _ { i j } ^ { ( \ell ) } ) ^ { \top } / \sqrt { C _ { h } r } .$ </td></tr><tr><td>7</td><td> $\widetilde { \mathbf { h } } _ { i  j } ^ { ( \ell ) } = \mathbf { T } _ { i  j } ^ { ( \ell ) } \mathbf { h } _ { j } ^ { ( \ell ) }$  Transport every source feature into its target-conditioned representation:</td></tr><tr><td>8</td><td>Aggregate transported messages and update the target feature:  $\begin{array} { r } { \mathbf { m } _ { i } ^ { ( \ell ) } = \sum _ { j \in \mathcal { N } ( i ) } \alpha _ { i j } ^ { ( \ell ) } \widetilde { \mathbf { h } } _ { i  j } ^ { ( \ell ) } } \end{array}$  and  $\mathbf { h } _ { i } ^ { ( \ell + 1 ) } = \mathbf { h } _ { i } ^ { ( \ell ) } + \rho _ { \theta } ^ { ( \ell ) } ( \mathbf { m } _ { i } ^ { ( \ell ) } )$ </td></tr><tr><td>9</td><td>Repeat Steps  $_ { 6 - 8 }$  for all  $L _ { g }$  low-rank gauge transport layers and modulate the final nodal features by the learned importance weights.</td></tr><tr><td>10</td><td>Reconstruct the adaptive nodal features on the canonical grid using coarse-grid upsampling or distance-weighted interpolation, producing  $\mathbf { F } _ { \mathrm { g r i d } }$ </td></tr><tr><td>11</td><td>Predict the local residual  $\delta _ { \mathrm { r a w } }$  and the truncated spectral correction  $\delta _ { \mathrm { s p e c } } ,$  and set  $\delta _ { 1 } \ =$   $\delta _ { \mathrm { r a w } } + \delta _ { \mathrm { s p e c } } .$ </td></tr><tr><td>12</td><td>Construct the provisional field  $\mathbf { u _ { \mathrm { b a s e } } }$  , compute its finite-difference features, and obtain the differential correction and final residual δ.</td></tr><tr><td>Output</td><td>Return  $\widehat { \mathbf { u } } = \mathbf { u } _ { t } + \boldsymbol \delta$  for temporal residual prediction, or  $\widehat { \mathbf { u } } = \delta$  for steady or direct-output problems.</td></tr></table>

## B.2 DATA-DRIVEN ADAPTIVE MESH METHODS

Adaptive meshing concentrates computational resolution near sharp gradients, discontinuities, and dynamically complex structures. Classical adaptive mesh refinement uses equation-dependent error indicators and hierarchical refinement rules Berger & Oliger (1984); Berger & Colella (1989), while constrained neural adaptation provides an early example of learning spatial resolution for PDE approximation Rudd et al. (2014). Graph Element Networks introduce adaptive structured computation over spatial elements Alet et al. (2019), and graph-based simulators subsequently support physical prediction on mesh and particle representations Sanchez-Gonzalez et al. (2020). MeshGraph Nets learn dynamics directly on simulation meshes and adapt connectivity during rollout Pfaff et al. (2020), whereas differentiable combinations of PDE solvers and graph networks preserve a stronger connection to conventional numerical computation de Avila Belbute-Peres et al. (2020). More ex plicit learned-mesh approaches treat node placement itself as an optimization problem. M2N learns end-to-end mesh movement for PDE solvers Song et al. (2022), and LAMP jointly learns physical evolution with a controllable refinement and coarsening policy for multiresolution simulation Wu et al. (2023). Data-Free Mesh Movers construct an r-adaptive mesh through a Monge–Ampere-\` based objective and integrate the resulting mesh into a learned PDE solver Hu et al. (2024). UGM2N further introduces an unsupervised mesh-movement objective based on local equidistribution Wang et al. (2025). Related operator-learning methods handle geometric variation through learned global coordinate transformations: Geo-FNO maps irregular geometries to a uniform latent domain Li et al. (2023a), while GINO transfers information between irregular point sets and a regular latent grid Li et al. (2023b). These methods substantially improve where computation is allocated or how irregular data are embedded. However, the adapted mesh is still commonly treated as a sampling, interpolation, or message-passing structure once it has been generated. Node relocation changes local density, orientation, spatial support, and neighborhood geometry, but conventional aggregation does not explicitly account for how these changes affect the representation convention of each node feature. Consequently, physical variation and discretization-induced representation variation may be mixed during information exchange. GA-AMNO addresses this complementary problem by coupling physics-informed adaptive allocation with source-to-target feature transport before aggregation.

## B.3 EQUIVARIANT AND GAUGE-AWARE REPRESENTATION LEARNING

Equivariant representation learning provides a principled framework for processing features under transformations of coordinates, orientations, or local reference frames. Geometric deep learning unifies many of these constructions across grids, groups, graphs, and manifolds Bronstein et al. (2021). Steerable three-dimensional convolutions Weiler et al. (2018), SE(3)-Transformers Fuchs et al. (2020), and E(n)-equivariant graph networks Satorras et al. (2021) preserve prescribed transformation laws under global rotations and translations. In operator learning, Group Equivariant Fourier Neural Operators extend rotational, translational, and reflectional symmetries to spectral operator layers Helwig et al. (2023). Steerable partial differential operators provide a complementary characterization of when differential mappings between feature fields are equivariant Jenner & Weiler (2021). These global symmetry constructions improve physical consistency, but they do not directly resolve the comparison of features expressed in independently varying local frames. Gauge-equivariant convolutional networks instead associate each location with a local representation frame and require feature transformations to follow a prescribed gauge action Cohen et al. (2019). Gauge Equivariant Mesh CNNs extend this principle to manifold meshes by parallel-transporting neighboring features before applying anisotropic convolution de Haan et al. (2021). Gauge Equivariant Transformer incorporates parallel transport into self-attention on triangular meshes He et al. (2021), while gauge-equivariant nonlinear message passing applies the same geometric principle to nonlinear dynamics and PDE evolution on surfaces Park et al. (2023). These studies establish the central principle that features expressed under different local conventions should be transported into compatible representations before interaction. However, they generally assume a fixed geometric domain, explicitly constructed local frames, and predefined transformation laws. GA-AMNO considers a different setting in which the discretization itself depends on the input. The resulting local representation changes are not known a priori and are not claimed to obey exact gauge equivariance. Instead, relative coordinates, adaptive-mesh Jacobians, and state descriptors condition a learned diagonal-plus-low-rank transport that maps each source feature into a target-conditioned representation before aggregation. The gauge interpretation therefore provides a model for interaction readability under input-dependent discretization, while the learned connection is treated as an approximate, functionally testable representation-alignment mechanism.

## C EXPERIMENTAL SETTING

## C.1 DATASETS AND DATA PREPARATION

We evaluate GA-AMNO on five PDE benchmarks: Navier-Stokes Li et al. (2025), Rayleigh-Benard´ Burns et al. (2020), KS Shysheya et al. (2024), Kolmogorov Rozet & Louppe (2023), and Darcy Li et al. (2025). The benchmarks cover scalar and multi-channel temporal prediction as well as a steady-state coefficient-to-solution mapping.

The Navier-Stokes dataset contains 5,000 trajectories of two-dimensional incompressible flow at viscosity $1 0 ^ { - 3 }$ Each trajectory contains 50 scalar flow-state snapshots on a 64 × 64 grid. We use trajectories 1–1,000 for training, 1,001–1,100 for validation, and the final 100 trajectories fo testing. The Rayleigh-Benard dataset contains vorticity and temperature-anomaly fields on a´ $6 4 \times 6 4$ grid. Its training, validation, and test splits contain 256, 64, and 128 trajectories, respectively, with 41 snapshots per trajectory. The KS dataset contains one-dimensional scalar trajectories with 256 spatial points and 641 temporal snapshots. The training, validation, and test splits contain 800, 100, and 50 trajectories, respectively. The Kolmogorov dataset contains two-channel forced-flow states on a 64 × 64 grid. It provides 819 training trajectories, 102 validation trajectories, and 103 test trajectories, with 64 snapshots per trajectory. The Darcy benchmark is a steady-state operatorlearning problem on a 64×64 grid. Its training, validation, and test splits contain 900, 124, and 1,024 samples, respectively. Each sample stores a coefficient field followed by its normalized solution field.

For the four temporal benchmarks, the input is a history of four consecutive snapshots, $\begin{array} { r l } { \mathcal { U } _ { t } } & { { } = } \end{array}$ $\left\{ u _ { t - 3 } , u _ { t - 2 } , u _ { t - 1 } , u _ { t } \right\}$ , and the target is the immediately following state $u _ { t + 1 }$ . The starting indices of successive windows are separated by 8 snapshots for Navier-Stokes, Rayleigh-Benard, and KS,´ and by 4 snapshots for Kolmogorov. Thus, the stride controls window extraction and does not introduce gaps between the four states inside a history window. For Darcy, the coefficient field is used as a single conditioning input and the corresponding solution is the target. No additional dataset-level normalization is applied during training; the pre-normalized Darcy files are loaded directly.

## C.2 BASELINES AND COMPARISON PROTOCOL

We compare GA-AMNO with seven representative baselines. 1) FNO Li et al. (2020a) applies global spectral convolution on a regular grid. 2) Transolver Wu et al. (2024) uses physics-aware slice attention to model long-range spatial interactions. 3) GNO Li et al. (2020b) performs coordinateconditioned local message passing. 4) WNO Tripura & Chakraborty (2023) uses Haar-wavelet operator blocks to capture localized multiresolution components. 5) UNet Ronneberger et al. (2015) is a convolutional encoder–decoder with multiscale skip connections. 6) Classic is a controlled adaptive-mesh baseline that learns node displacements but directly aggregates node features using the shared base solver, without physics-informed adaptive allocation, low-rank gauge transport, spectral residual correction, or PDE-aware correction. 7) MMPDE Hu et al. (2024) combines a datafree differentiable mesh mover with regular-mesh and moving-mesh solution branches to perform adaptive PDE prediction. Our implementations of Transolver, GNO, WNO, and MMPDE follow the core mechanisms of the cited architectures and are adapted to the unified input–output protocol, spatial resolution, and training configuration used in this study. Classic shares the corresponding base prediction protocol with GA-AMNO and serves as a controlled comparison for evaluating the benefit of operator-readable adaptive discretization beyond adaptive node movement alone.

All methods use the same dataset files, trajectory splits, history windows, prediction targets, spatial resolutions, learning rate, epoch count, and number of updates per epoch. Models that use coordinates receive the same regular physical-coordinate encoding. Checkpoints are selected by the same validation relative $\ell _ { 2 }$ criterion. The comparison controls the data protocol and optimizationstep budget but does not enforce parameter- or FLOP-matched architectures. Parameter counts and counted FLOPs are therefore reported separately when computational cost is discussed.

## C.3 EVALUATION METRICS

We use relative $\ell _ { 2 }$ error as the primary metric for predictive comparison. RMSE and spectral-energy error are additionally reported for rollout evaluation, while gradient, differential, and high-band

spectral errors are used for targeted mechanism analyses.

$$
\begin{array} { r l r } { \mathrm { R e l L 2 } ( \widehat { u } , u ) = \displaystyle \frac { \| \widehat { u } - u \| _ { 2 } } { \| u \| _ { 2 } + \epsilon } , } & { } & \\ { \mathrm { R M S E } ( \widehat { u } , u ) = \sqrt { \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( \widehat { u } _ { i } - u _ { i } ) ^ { 2 } } , } & \\ { \mathrm { S p e c t r u m E r r o r } ( \widehat { u } , u ) = \displaystyle \frac { \| | \mathcal { F } _ { \mathrm { D F T } } ( \widehat { u } ) | _ { ^ { \prime } c } ^ { 2 } - | \mathcal { F } _ { \mathrm { D F T } } ( u ) | _ { ^ { \prime } c } ^ { 2 } \| _ { 2 } } { \| | \mathcal { F } _ { \mathrm { D F T } } ( u ) | | _ { ^ { \prime } c } ^ { 2 } \| _ { 2 } + \epsilon } . } & \end{array}\tag{11}
$$

Here, $\widehat { u }$ and u are the prediction and ground truth, N is the number of scalar field entries, $\mathcal { F } _ { \mathrm { D F T } }$ is the orthonormal discrete Fourier transform, $\overline { { ( \cdot ) } } _ { c }$ denotes averaging over channels, and $\epsilon = 1 0 ^ { - 8 }$ The overall SpectrumError measures spectral-energy discrepancy over all represented frequencies and is used for rollout evaluation. For the Spectral Residual targeted ablation, we separately report high-band spectral error. Let

$$
K _ { \mathrm { h i } } = \left\{ k : \frac { \nu ( k ) } { \nu _ { \mathrm { m a x } } } \geq 0 . 6 \right\} ,
$$

where $\nu ( k )$ is the radial frequency of coefficient k and $\nu _ { \mathrm { m a x } }$ is the maximum represented radius. The high-band spectral error is the same channel-averaged spectral-energy discrepancy restricted to $\kappa _ { \mathrm { h i } } .$

$$
\mathrm { H i g h B a n d E r r o r } ( \widehat { u } , u ) = \frac { \left\| \overline { { | \mathcal { F } _ { \mathrm { D F T } } ( \widehat { u } ) | ^ { 2 } } } _ { c , \mathcal { K } _ { \mathrm { h i } } } - \overline { { | \mathcal { F } _ { \mathrm { D F T } } ( u ) | ^ { 2 } } } _ { c , \mathcal { K } _ { \mathrm { h i } } } \right\| _ { 2 } } { \left\| \overline { { | \mathcal { F } _ { \mathrm { D F T } } ( u ) | ^ { 2 } } } _ { c , \mathcal { K } _ { \mathrm { h i } } } \right\| _ { 2 } + \epsilon } .
$$

Here, the subscript $c , K _ { \mathrm { h i } }$ denotes channel averaging followed by restriction to the high-band coefficients. Gradient error is used in mechanism and regional analyses:

$$
\mathrm { G r a d E r r o r } ( \widehat { u } , u ) = \frac { \| [ D _ { x } \widehat { u } - D _ { x } u , D _ { y } \widehat { u } - D _ { y } u ] \| _ { 2 } } { \| [ D _ { x } u , D _ { y } u ] \| _ { 2 } + \epsilon } ,\tag{12}
$$

where $D _ { x }$ and $D _ { y }$ are centered finite differences in grid-index units with one-sided boundary differences. For one-dimensional KS data, the inactive vertical derivative is zero.

For Darcy, the current auxiliary differential diagnostic is reported as Laplacian discrepancy,

$$
E _ { \Delta } = \frac { \| \Delta \widehat { u } - \Delta u \| _ { 2 } } { \| \Delta u \| _ { 2 } + \epsilon } ,\tag{13}
$$

rather than as an equation residual, because it does not explicitly evaluate $- \nabla \cdot ( a \nabla \widehat { u } ) - f .$ . For scalar Navier-Stokes and KS fields, divergence, vector vorticity, and enstrophy are not reported because they are not defined by the stored single-channel representation. Physics-specific vectorfield diagnostics are used only when the stored channels have the required physical meaning.

To study input-dependent adaptation demand, we use

$$
\begin{array} { r l } & { d ( \mathbf { x } ) = 0 . 4 \mathcal { N } ( g ( \mathbf { x } ) ) + 0 . 3 \mathcal { N } ( c ( \mathbf { x } ) ) } \\ & { \quad \quad + 0 . 2 \mathcal { N } ( m ( \mathbf { x } ) ) + 0 . 1 \mathcal { N } ( I ( \mathbf { x } ) ) , } \end{array}\tag{14}
$$

where $g \ = \ \| \nabla u _ { t } \|$ is input-gradient magnitude, $c = | \Delta u _ { t } |$ is input-curvature magnitude, m is the norm of the local finite-difference variation of the learned mesh offset, and I is the learned importance map. $\mathcal { N }$ denotes per-sample robust normalization using the 5th and 95th percentiles. The lower, middle, and upper terciles of d define low-, medium-, and high-demand masks. The masks are constructed once from the input and the full GA-AMNO geometry, without using targets or predictions, and the same masks are applied to every compared model.

Importance allocation faithfulness is evaluated by replacing the learned importance map with a spatially uniform, randomly shuffled, or cyclically shifted map and then reporting the resulting relative $\ell _ { 2 }$ error. Gauge edge faithfulness is measured using the transport-correction magnitude

$$
c _ { i j } = \| \mathbf { T } _ { i  j } \mathbf { h } _ { j } - \mathbf { h } _ { j } \| _ { 2 } ,\tag{15}
$$

where $\mathbf { h } _ { j }$ is the source feature and $\mathbf { T } _ { i  j }$ is the learned transport from node $j$ to node i. After replacing the transport on an edge subset S by the identity, its causal influence is quantified as

$$
\Delta _ { \mathrm { p r e d } } ( S ) = \frac { \| \widehat { u } _ { S } - \widehat { u } _ { \mathrm { f u l l } } \| _ { 2 } } { \| \widehat { u } _ { \mathrm { f u l l } } \| _ { 2 } + \epsilon } ,\tag{16}
$$

where $\widehat { u } _ { S }$ is the intervened prediction and $\widehat { u } _ { \mathrm { f u l l } }$ is the original prediction.

Message–target compatibility is evaluated after dividing directed edges into low-, medium-, and high-geometric-mismatch groups using local edge-scale ratio, mesh-Jacobian variation, and importance difference. For each group $\mathcal { E } _ { b } ,$ , we compute

$$
E _ { \mathrm { s r c } } ^ { ( b ) } = \frac { 1 } { | \mathcal { E } _ { b } | } \sum _ { ( i , j ) \in \mathcal { E } _ { b } } \frac { \| \mathbf { h } _ { i } - \mathbf { h } _ { j } \| _ { 2 } } { \| \mathbf { h } _ { i } \| _ { 2 } + \epsilon } ,
$$

$$
E _ { \mathrm { m s g } } ^ { ( b ) } = \frac { 1 } { | \mathcal { E } _ { b } | } \sum _ { ( i , j ) \in \mathcal { E } _ { b } } \frac { \Vert \mathbf { h } _ { i } - \mathbf { T } _ { i  j } \mathbf { h } _ { j } \Vert _ { 2 } } { \Vert \mathbf { h } _ { i } \Vert _ { 2 } + \epsilon } ,
$$

$$
G _ { \mathrm { c o m p } } ^ { ( b ) } = \frac { E _ { \mathrm { s r c } } ^ { ( b ) } - E _ { \mathrm { m s g } } ^ { ( b ) } } { E _ { \mathrm { s r c } } ^ { ( b ) } + \epsilon } .\tag{17}
$$

Here, b indexes the geometric-mismatch group, $E _ { \mathrm { s r c } } ^ { ( b ) }$ is the source–target feature discrepancy, $E _ { \mathrm { m s g } } ^ { ( b ) }$ is the transported-message–target discrepancy, and $G _ { \mathrm { c o m p } } ^ { ( b ) }$ is their relative reduction. A positive value means that the message supplied to aggregation is closer to the current target feature. Because neighboring nodes may contain actually different physical content, this quantity is a messagecompatibility diagnostic rather than a direct estimate of the covariance defect in Equation 36. The training-time loop diagnostic constrains only the diagonal transport component around sampled local triangular cycles and is therefore termed diagonal loop consistency, not exact holonomy of the complete low-rank transport. Mesh inversion rate, minimum cell angle, and aspect ratio are reported only as mesh-health diagnostics.

## C.4 IMPLEMENTATION DETAILS

GA-AMNO uses a hidden width of 32, two low-rank gauge transport layers, six indexed neighbors per node, and a low-rank transport rank of $r = 4$ . The operator stride is 2 for the two-dimensional datasets and 1 for KS. The maximum latent-mesh displacement is 0.08 for the temporal benchmarks and is reduced to 0.02 for Darcy. Dataset-specific spectral modes and auxiliary-loss weights are selected using only the validation split. The reference connectivity is fixed, while latent coordinates, local Jacobians, edge descriptors, and transport matrices depend on the input state.

All methods are optimized with AdamW for 300 fixed-budget epochs using an initial learning rate of $1 0 ^ { - 3 }$ . Each epoch consists of 100 mini-batch updates, giving 30,000 parameter updates per run rather than 300 complete passes through each windowed dataset. The default training batch size is 8. Owing to GPU-memory constraints, GA-AMNO uses a batch size of 4 on Rayleigh-Benard,´ whereas the corresponding baseline runs use batch size 8. Validation is performed every 10 epochs, in addition to the first and final epochs, and the checkpoint with the lowest validation relative $\ell _ { 2 }$ error is selected for testing. GA-AMNO is evaluated using three random seeds, 42, 43, and 44, and its main results are reported as the mean and standard deviation across these runs. Unless otherwise specified, the baseline and mechanism-analysis experiments use random seed 42.

The GA-AMNO objective combines field MSE with supervised gradient, Laplacian, and spectral losses, together with diagonal loop consistency, transport alignment, mesh-quality, physics-informed allocation, frame-consistency, cross-mesh, and mesh-perturbation regularizers. The weights are tuned separately for each dataset on its validation split and remain fixed during test evaluation. These auxiliary terms regularize the learned latent geometry and feature transport; they are not treated as exact gauge-equivariance constraints.

## C.5 EFFICIENCY AND MEMORY SCALABILITY

We evaluate computational scalability using the Navier-Stokes GA-AMNO checkpoint on an NVIDIA GeForce RTX 4070 Laptop GPU. The model contains 177.4K trainable parameters. At

![](images/ba2b8c57bd224ccda4fd5fb7acd5f5a7dbd06192b6bf3966fea9c79cd5e09fbd.jpg)

![](images/ed023e88b4823e24e9d590f1ca002565e60df7e4a21166c9db219e91983975a9.jpg)  
Figure 7: Computational scalability of GA-AMNO. (a) Counted FLOPs and synchronized forward latency at different grid resolutions. (b) Peak allocated forward memory. The same Navier-Stokes checkpoint and batch size 8 are used throughout.

$6 4 \times 6 4$ resolution with batch size 8, the hook-based estimator reports 8.943G counted FLOPs for convolutional and linear operations. The measured forward latency is 23.38 ms per batch and peak allocated forward memory is 144.3 MiB. Latency is measured after 15 warm-up passes and averaged over 30 synchronized inference passes. The FLOP estimate counts a multiply–add as two operations and does not include every FFT, distance, interpolation, normalization, softmax, or tensor-contraction operation; it is therefore reported as an implementation-level estimate rather than an exact hardware-independent operation count.

As shown in Figure 7, the same checkpoint is evaluated from 16×16 to 128×128 without changing model parameters. Inputs are bilinearly resized solely for this computational-scaling study. The counted FLOPs increase from 0.559G at $1 6 \times 1 6$ to 35.771G at $1 2 \bar { 8 } \times 1 2 8$ , while peak forward memory increases from 19.7 MiB to 542.8 MiB. Latency is dominated by launch overhead at resolutions below 64 × 64 and increases to 51.28 ms per batch at 128 × 128. Figure 7 is a computational scalability analysis and should not be interpreted as a cross-resolution accuracy experiment.

## D ADDITIONAL METHOD DETAILS

## D.1 PHYSICS-INFORMED ADAPTIVE ALLOCATION

For the latest input slice v, the physical indicators used by the importance predictor are

$$
\begin{array} { l } { \displaystyle { { \cal Q } _ { g } ( { \bf x } ) = \left[ \frac { 1 } { C } \sum _ { c = 1 } ^ { C } \left( | D _ { x } v ^ { ( c ) } ( { \bf x } ) | ^ { 2 } + | D _ { y } v ^ { ( c ) } ( { \bf x } ) | ^ { 2 } \right) \right] ^ { 1 / 2 } , } } \\ { \displaystyle { { \cal Q } _ { \Delta } ( { \bf x } ) = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } | D _ { x } D _ { x } v ^ { ( c ) } ( { \bf x } ) + D _ { y } D _ { y } v ^ { ( c ) } ( { \bf x } ) | , } } \\ { \displaystyle { { \cal Q } _ { E } ( { \bf x } ) = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } | v ^ { ( c ) } ( { \bf x } ) | ^ { 2 } , } } \\ { \displaystyle { { \cal Q } _ { \omega } ( { \bf x } ) = | D _ { x } v ^ { ( 2 ) } ( { \bf x } ) - D _ { y } v ^ { ( 1 ) } ( { \bf x } ) | . } } \end{array}\tag{18}
$$

Here, x is a spatial location, v is the latest input field, C is its number of physical channels, c is the channel index, and $v ^ { ( c ) } ( \mathbf { x } )$ is the value of channel $c \ a t \ \mathbf { x } .$ . The operators $D _ { x }$ and $D _ { y }$ are finite-difference derivatives along the two spatial directions. The quantities $Q _ { g } , Q _ { \Delta } , Q _ { E }$ , and $Q _ { \omega }$ denote the gradient magnitude, Laplacian magnitude, local energy, and vorticity-related response, respectively.

The four indicators provide complementary local information. Specifically, $Q _ { g }$ measures first-order variation, $Q _ { \Delta }$ captures second-order local structure, $Q _ { E }$ measures field strength or activity, and $Q _ { \omega }$ provides a cue related to rotation or shear. For scalar fields, where the two-component response required by $Q _ { \omega }$ is unavailable, the implementation uses the Laplacian-magnitude channel instead.

As defined in Equation 5, the indicators are normalized and concatenated with the learned descriptor $\mathbf { P } .$ . They serve as explicit conditioning variables rather than an assumed unique or complete description of the governing dynamics. The nonlinear predictor may therefore adjust their relative contributions according to the current input state.

In Equation $^ { 6 , }$ the mean normalization makes $w _ { i }$ a relative positive importance weight. The lower bound $I _ { \mathrm { m i n } }$ prevents any node from being completely suppressed, while $s _ { I }$ and $\delta _ { \mathrm { m a x } }$ control the importance-modulation range and maximum displacement scale, respectively. Consequently, the indicators directly condition node relocation rather than serving only as auxiliary prediction targets.

## D.2 GAUGE-AWARE MESH OPERATOR

To motivate source-to-target transport, let $\mathbf { g } _ { i }$ denote the local discretization context at node i and let $\mathbf { z } _ { i }$ denote its latent physical content. We model the encoded feature as

$$
\mathbf { h } _ { i } = \mathbf { R } ( \mathbf { g } _ { i } ) \mathbf { z } _ { i } ,\tag{19}
$$

where $\mathbf { h } _ { i }$ is the encoded feature at node i, g<sub>i</sub> collects its local discretization variables, $\mathbf { z } _ { i }$ represents the underlying latent physical content, and $\bar { \bf R } ( { \bf g } _ { i } )$ is an invertible representation map associated with that local context.

Under this representation model, the ideal source-to-target transport is

$$
\mathbf { T } _ { i  j } ^ { \star } = \mathbf { R } ( \mathbf { g } _ { i } ) \mathbf { R } ( \mathbf { g } _ { j } ) ^ { - 1 } .\tag{20}
$$

Here, $\mathbf { T } _ { i  j } ^ { \star }$ denotes the ideal transport from source node $j$ to target node $i ,$ the superscript ⋆ distinguishes the ideal transport from its learned approximation, and ${ \mathbf { R } } ( { \bf g } _ { j } ) ^ { - 1 }$ is the inverse representation map of the source context. The representation maps and local contexts have been defined in Equation 19.

Applying this ideal transport to $\mathbf { h } _ { j }$ gives $\mathbf { R } ( \mathbf { g } _ { i } ) \mathbf { z } _ { j }$ , which expresses the source content under the target representation context. This relation motivates the learned transport but is not imposed as an exact equality on the implemented network.

The learned edge descriptor in Equation 7 includes relative adaptive coordinates, source and target mesh Jacobians, and their state descriptors. It therefore conditions information exchange on both the geometry produced by adaptive allocation and the current encoded state.

The diagonal component in Equation 8 supports efficient channel-wise adjustment, while the gated low-rank component permits cross-channel correction without predicting an unrestricted $\bar { C } _ { h } \ \bar { \times } \ \bar { C } _ { h }$ matrix for every edge. Separate prediction heads process ${ \bf e } _ { i j }$ to generate $\mathbf { d } _ { i j } ^ { ( \ell ) } , \mathbf { U } _ { i j } ^ { ( \ell ) } , \mathbf { V } _ { i j } ^ { ( \ell ) }$ , and $\gamma _ { i j } ^ { ( \ell ) }$

In Equation $^ { 9 , }$ transport is applied before the weighted sum. The operator therefore exposes how the source–target geometry conditions each incoming message instead of leaving this dependence implicit in direct aggregation. The residual update in Equation 10 then combines the transported message with the existing target representation.

The learned transformation is intended to reduce representation incompatibility within the considered adaptive-mesh regime. It is not claimed to recover a unique physical frame, represent an exact analytic coordinate transformation, or impose exact Gauge equivariance. The corresponding sufficient conditions, approximation analysis, and deformation-continuity results are presented in Appendix E.

## D.3 MESH-TO-GRID BRIDGE

The gauge-aware operator produces features on adaptive nodes, whereas the output head expects a regular spatial tensor. When the operator and output grids have the same resolution, the bridge assigns distance-based weights to nearby adaptive nodes for each target location $\mathbf { y } _ { q } ^ { r } \mathrm { ; }$

$$
\kappa _ { q i } = \frac { \exp ( - \| \mathbf { x } _ { i } ^ { a } - \mathbf { y } _ { q } ^ { r } \| _ { 2 } / \tau _ { b } ) } { \sum _ { j \in \mathcal { N } _ { b } ( q ) } \exp ( - \| \mathbf { x } _ { j } ^ { a } - \mathbf { y } _ { q } ^ { r } \| _ { 2 } / \tau _ { b } ) } .\tag{21}
$$

Here, $\kappa _ { q i }$ is the reconstruction weight from adaptive node $i$ to target grid point $q , \mathbf { y } _ { q } ^ { r }$ is the qth regular-grid coordinate, $\mathcal { N } _ { b } ( q )$ is the bridge neighborhood, and $\tau _ { b }$ is the temperature controlling distance sensitivity.

The reconstructed feature at grid point q is

$$
\mathbf { z } _ { q } = \sum _ { i \in \mathcal { N } _ { b } ( q ) } \kappa _ { q i } \widehat { \mathbf { h } } _ { i } , \qquad \widehat { \mathbf { h } } _ { i } = w _ { i } \mathbf { h } _ { i } ^ { ( L _ { g } ) } .\tag{22}
$$

Here, $\mathbf { z } _ { q }$ is the interpolated grid feature, $\widehat { \mathbf { h } } _ { i }$ is the mean-normalized importance-modulated final nodal feature, $w _ { i }$ is defined in the adaptive-mesh module, $L _ { g }$ is the number of gauge-aware layers, and $\mathbf { h } _ { i } ^ { ( L _ { g } ) }$ is the final feature of adaptive node i.

The regular tensor is then obtained by projection and reshaping:

$$
\mathbf { F } _ { \mathrm { g r i d } } = \mathrm { R e s h a p e } _ { H , W } \left( \left[ \rho _ { b } ( \mathbf { z } _ { 1 } ) , \ldots , \rho _ { b } ( \mathbf { z } _ { M } ) \right] \right) .\tag{23}
$$

Here, $\rho _ { b }$ is a learnable channel projection, H and W are the target grid height and width, and $M = H W$ is the number of regular-grid locations. In the default two-dimensional configuration, the gauge operator uses stride 2 and therefore follows the coarse-grid path: its feature map is bilinearly upsampled and then passed through a channel projection. The distance-weighted bridge above is used when the operator and output grids have equal resolution. We consequently use mesh-to-grid reconstruction as the general name for these two implementation paths.

## D.4 SPECTRAL AND DIFFERENTIAL RESIDUAL CORRECTION

The prediction head combines local convolutional prediction with spectral and physical refinement. The local solver input is

$$
\mathbf { Z } _ { \mathrm { s o l } } = \operatorname { C o n c a t } ( \mathbf { F } _ { \mathrm { g r i d } } , \mathbf { v } ) ,\tag{24}
$$

where $\mathbf { Z } _ { \mathrm { s o l } }$ is the solver input and v denotes the latest input field for temporal prediction or the conditioning field for steady-state prediction. The local solver first produces $ { \delta _ { \mathrm { r a w } } } =  { \mathcal { H } _ { \mathrm { l o c a l } } } (  { \mathbf { Z } } _ { \mathrm { s o l } } )$ The spectral branch receives the complete input history, reconstructed grid features, and canonical coordinates:

$$
\begin{array} { r l } & { \mathbf { X } _ { \mathrm { s p e c } } = \mathrm { C o n c a t } ( \mathrm { F l a t t e n } _ { t } ( \mathbf { a } ) , \mathbf { F } _ { \mathrm { g r i d } } , \mathbf { X } ^ { r } ) , } \\ & { \mathbf { Z } _ { \mathrm { s p e c } } = \mathcal { L } _ { \mathrm { s p e c } } ( \mathbf { X } _ { \mathrm { s p e c } } ) , } \\ & { \delta _ { \mathrm { s p e c } } = s _ { \mathrm { s p e c } } \mathcal { P } _ { \mathrm { s p e c } } ( \varphi ( \mathcal { N } _ { \mathrm { s p e c } } ( \mathcal { C } _ { \mathrm { F } } ( \mathbf { Z } _ { \mathrm { s p e c } } ) + \mathcal { C } _ { 1 \times 1 } ( \mathbf { Z } _ { \mathrm { s p e c } } ) ) ) ) . } \end{array}\tag{25}
$$

Here, $\mathcal { L } _ { \mathrm { s p e c } }$ and $\mathcal { P } _ { \mathrm { s p e c } }$ are the spectral lift and output projection, $\mathcal { C } _ { \mathrm { F } }$ is a truncated Fourier convolution retaining a prescribed set of low-order modes, $\mathcal { C } _ { 1 \times 1 }$ is a pointwise spatial path, $\mathcal { N } _ { \mathrm { s p e c } }$ is group normalization, $\varphi$ is GELU, and $s _ { \mathrm { s p e c } } = 0 . 2 \operatorname { t a n h } ( \beta _ { \mathrm { s p e c } } )$ is a learned bounded scale. This module is therefore a global spectral correction, not an explicit high-pass filter.

The intermediate residual $\delta _ { 1 } = \delta _ { \mathrm { r a w } } + \delta _ { \mathrm { s p e c } }$ is then refined using finite-difference features of its provisional field:

$$
\begin{array} { r l } & { \mathbf { u } _ { \mathrm { b a s e } } = \mathbf { v } + \delta _ { 1 } , } \\ & { \mathbf { Z } _ { \mathrm { p h y } } = \mathrm { C o n c a t } ( \mathbf { v } , \delta _ { 1 } , D _ { x } \mathbf { u } _ { \mathrm { b a s e } } , D _ { y } \mathbf { u } _ { \mathrm { b a s e } } , } \\ & { \qquad \quad \Delta \mathbf { u } _ { \mathrm { b a s e } } , E ( \mathbf { u } _ { \mathrm { b a s e } } ) ) , } \\ & { \qquad \quad \delta = \delta _ { 1 } + s _ { \mathrm { p h y } } \mathcal { H } _ { \mathrm { p h y } } ( \mathbf { Z } _ { \mathrm { p h y } } ) . } \end{array}\tag{26}
$$

Here, ${ \mathcal { H } } _ { \mathrm { p h y } }$ is the convolutional differential-correction network, $s _ { \mathrm { p h y } } ~ = ~ 0 . 2 \operatorname { t a n h } ( \beta _ { \mathrm { p h y } } )$ is its learned bounded scale, $\Delta$ is the discrete Laplacian, and E is the channel-averaged squared amplitude. For direct-output tasks in which the conditioning and output channel counts differ, $\mathbf { u } _ { \mathrm { b a s e } }$ is set to $\delta _ { 1 }$

The final prediction is

$$
\widehat { \mathbf { u } } = \left\{ \begin{array} { l l } { \mathbf { u } _ { t } + \delta , } & { \mathrm { f o r t e m p o r a l ~ r e s i d u a l ~ p r e d i c t i o n } , } \\ { \delta , } & { \mathrm { f o r ~ s t e a d y - s t a t e ~ p r e d i c t i o n ~ o r ~ d i r e c t ~ o u t p u t } . } \end{array} \right.\tag{27}
$$

In this equation, $\mathbf { u } _ { t }$ is the latest observed state and $\widehat { \mathbf { u } }$ is the predicted future or steady-state field. All modules are trained end to end, with mesh-quality and transport-consistency terms used as auxiliary regularization during training.

## E THEORETICAL ANALYSIS OF OPERATOR READABILITY

We formalize operator readability relative to a declared family of nondegenerate adaptive discretizations. It has two requirements. First, allocation readability requires the node displacement to be an explicit function of specified input and geometry variables, so that their influence can be inspected or intervened on. Second, interaction readability requires a source message to have the same physical meaning after an admissible change of its local representation convention. The first requirement concerns where computation is allocated; the analysis below focuses on the second requirement, which concerns how the resulting representations are exchanged.

Let $\mathbf { g } _ { i }$ collect the local discretization variables at node i, including local coordinates, mesh Jacobians, and state descriptors. The representation model in Equation 19 adopts the standard gaugeequivariant viewpoint that the same underlying content may have different coordinate representations under changes of local frame (Cohen et al., 2019; de Haan et al., 2021). Here, we adapt this viewpoint to input-dependent discretizations without assuming that the implemented GA-AMNO satisfies exact gauge equivariance. Specifically, $\mathbf { z } _ { i } ~ \in ~ \mathbb { R } ^ { C _ { h } }$ is fixed latent physical content and ${ \bf R } ( { \bf g } _ { i } ) \in \mathfrak { G } \subseteq \bar { \mathrm { G L } } ( C _ { h } )$ is invertible. The admissible group $\mathfrak { G }$ contains only representation changes induced by the considered nondegenerate mesh family; it is not identified with every element of $\operatorname { G L } ( C _ { h } )$ . We assume that R and $\mathbf { \bar { R } } ^ { - 1 }$ are bounded on this family. Unless the Frobenius norm is explicitly indicated, matrix norms below are induced $\ell _ { 2 }$ norms. This representation model is our formalization for the stated adaptive-discretization setting; it is sufficient for the analysis and does not claim that the frame is unique or directly observable.

Definition 1 (Discretization-induced representation ambiguity). For fixed latent physical content $\mathbf { z } _ { i } .$ , define its admissible representation set as

$$
O ( \mathbf { z } _ { i } ) = \left\{ { \bf R } ( \mathbf { g } ) { \bf z } _ { i } : \mathbf { g } \in \mathfrak { D } \right\} ,\tag{28}
$$

where D is the declared family of nondegenerate local discretization contexts. Discretizationinduced representation ambiguity occurs when ${ \mathcal { O } } ( \mathbf { z } _ { i } )$ contains more than one element, so that the same physical content admits different encoded features under different admissible local discretizations. For two contexts $\mathbf { g } _ { i } , \mathbf { g } _ { i } ^ { \prime } \in \mathfrak { D }$ , the corresponding representations satisfy

$$
\mathbf { h } _ { i } ^ { \prime } = \mathbf { S } _ { i } \mathbf { h } _ { i } , \qquad \mathbf { S } _ { i } = \mathbf { R } ( \mathbf { g } _ { i } ^ { \prime } ) \mathbf { R } ( \mathbf { g } _ { i } ) ^ { - 1 } .\tag{29}
$$

We refer to this local nonuniqueness of feature coordinates, with $\mathbf { z } _ { i }$ unchanged, as a hidden gauge ambiguity. This is a modeling description of admissible representation changes, not a claim that the implemented network is exactly gauge equivariant.

Definition 2 (Interaction readability). A directed interaction $j  i$ satisfies interaction readability if its message depends on the source physical content $\mathbf { z } _ { j }$ and the target context $\mathbf { g } _ { i }$ , but not on the chosen admissible source-frame convention used to encode $\mathbf { z } _ { j }$ . Equivalently, changing local representation conventions must transform the aggregated message according to the target action $\mathbf { S } _ { i }$ rather than according to the independently varying source actions. An adaptive mesh is operatorreadable relative to the declared discretization family when this interaction property is combined with allocation readability.

Theorem 1 (Limitation of shared direct aggregation). Consider the uncorrected linear message

$$
\overline { { \mathbf { m } } } _ { i } = \sum _ { j \in \mathcal { N } ( i ) } \alpha _ { i j } \mathbf { W } \mathbf { h } _ { j } ,\tag{30}
$$

where W is shared across edges. Consider the subset of local reparameterizations $\mathbf { h } _ { j } ^ { \prime } = \mathbf { S } _ { j } \mathbf { h } _ { j }$ that leaves the scalar weights unchanged. If $\overline { { \mathbf { m } } } _ { i } ^ { \prime } = \mathbf { S } _ { i } \overline { { \mathbf { m } } } _ { i }$ must hold for all source features, then every edge with $\alpha _ { i j } \neq 0$ must satisfy

$$
\mathbf { W } \mathbf { S } _ { j } = \mathbf { S } _ { i } \mathbf { W } .\tag{31}
$$

Consequently, a single shared map cannot guarantee operator-readable aggregation for independently varying local frames.

Proof: The required equality is

$$
\sum _ { j } \alpha _ { i j } \mathbf { W } \mathbf { S } _ { j } \mathbf { h } _ { j } = \sum _ { j } \alpha _ { i j } \mathbf { S } _ { i } \mathbf { W } \mathbf { h } _ { j } .\tag{32}
$$

Because the source features can vary independently, each nonzero edge must satisfy Equation 31. For W = I, this reduces to $\mathbf { S } _ { j } = \mathbf { S } _ { i } ;$ for invertible W, it imposes the same fixed conjugacy relation on every neighbor. Neither condition accommodates independently changing local discretization frames in general.

Theorem 1 concerns shared uncorrected aggregation, not every irregular-mesh operator. A geometry-conditioned edge map may avoid this limitation; such a map already acts as a form of representation transport.

Theorem 2 (Sufficient condition for operator-readable transport). Let $\alpha _ { i j } \geq 0$ and $\textstyle \sum _ { j } \alpha _ { i j } = 1$ $\mathbf { I f } \ \mathbf { T } _ { i  j } = \mathbf { T } _ { i  j } ^ { \star }$ , with the ideal transport defined in Equation 20, then the transported aggregation satisfies

$$
\mathbf { m } _ { i } = \sum _ { j } \alpha _ { i j } \mathbf { T } _ { i  j } \mathbf { h } _ { j } = \mathbf { R } ( \mathbf { g } _ { i } ) \sum _ { j } \alpha _ { i j } \mathbf { z } _ { j } .\tag{33}
$$

Moreover, under a local coordinate change $\mathbf { h } _ { i } ^ { \prime } = \mathbf { S } _ { i } \mathbf { h } _ { i } , \mathrm { i f } \alpha _ { i j } ^ { \prime } = \alpha _ { i j }$ and

$$
\mathbf { T } _ { i  j } ^ { \prime } = \mathbf { S } _ { i } \mathbf { T } _ { i  j } \mathbf { S } _ { j } ^ { - 1 } ,\tag{34}
$$

then $\mathbf { m } _ { i } ^ { \prime } = \mathbf { S } _ { i } \mathbf { m } _ { i }$

Proof: Substituting Equation 19 and Equation 20 cancels the source-frame factors before summation and gives Equation 33. Under the coordinate change,

$$
\mathbf { m } _ { i } ^ { \prime } = \sum _ { j } \alpha _ { i j } \mathbf { S } _ { i } \mathbf { T } _ { i  j } \mathbf { S } _ { j } ^ { - 1 } \mathbf { S } _ { j } \mathbf { h } _ { j } = \mathbf { S } _ { i } \mathbf { m } _ { i } .\tag{35}
$$

Thus, all incoming physical contents are represented in one target frame, and changing local coordinates changes only the coordinates of the target message.

Theorem 2 is an exact sufficient condition. The learned diagonal-plus-low-rank map is not asserted to satisfy it identically. To cover the learned case, define the attention defect and connection defect as

$$
\begin{array} { l } { { \displaystyle \delta _ { \alpha , i } = \sum _ { j } \big \vert \alpha _ { i j } ^ { \prime } - \alpha _ { i j } \big \vert , } } \\ { { \displaystyle \varepsilon _ { i j } = \big \Vert \mathbf { T } _ { i  j } ^ { \prime } \mathbf { S } _ { j } - \mathbf { S } _ { i } \mathbf { T } _ { i  j } \big \Vert _ { 2 } . } } \end{array}\tag{36}
$$

Theorem 3 (Approximate interaction-consistency bound). Assume $\| \mathbf { h } _ { j } \| _ { 2 } \quad \leq \quad H$ and $\| \mathbf { T } _ { i  j } ^ { \prime } \mathbf { S } _ { j } \| _ { 2 } \leq \bar { B }$ . Then

$$
\left\| \mathbf { m } _ { i } ^ { \prime } - \mathbf { S } _ { i } \mathbf { m } _ { i } \right\| _ { 2 } \leq B H \delta _ { \alpha , i } + H \sum _ { j } \alpha _ { i j } \varepsilon _ { i j } .\tag{37}
$$

Proof: Add and subtract $\begin{array} { r } { \sum _ { j } \alpha _ { i j } \mathbf { T } _ { i  j } ^ { \prime } \mathbf { S } _ { j } \mathbf { h } _ { j } } \end{array}$ to obtain

$$
\begin{array} { l } { { \displaystyle { \bf m } _ { i } ^ { \prime } - { \bf S } _ { i } { \bf m } _ { i } = \sum _ { j } ( \alpha _ { i j } ^ { \prime } - \alpha _ { i j } ) { \bf T } _ { i  j } ^ { \prime } { \bf S } _ { j } { \bf h } _ { j } } } \\ { { \displaystyle ~ + \sum _ { j } \alpha _ { i j } ( { \bf T } _ { i  j } ^ { \prime } { \bf S } _ { j } - { \bf S } _ { i } { \bf T } _ { i  j } ) { \bf h } _ { j } . } } \end{array}\tag{38}
$$

The triangle inequality and submultiplicativity bound the first sum by $B H \delta _ { \alpha , i }$ and the second by $H \sum _ { j } { \alpha _ { i j } \varepsilon _ { i j } }$

The bound separates two failure modes of a learned layer: the scalar neighbor weighting may change, and the feature transport may deviate from connection covariance. It therefore applies even when attention is produced from learned state descriptors rather than assumed to be exactly gauge invariant.

Proposition 1 (Approximation capacity of diagonal-plus-low-rank transport). Let $\mathbf { T } ^ { \star } \in$ $\mathbb { R } ^ { \boldsymbol { C _ { h } ^ { * } \times C _ { h } } }$ denote an ideal transport matrix for one edge, and let D be any admissible diagonal matrix.

Define $\mathbf { E } = \mathbf { T ^ { \star } } - \mathbf { D }$ , with singular values $\sigma _ { 1 } ( \mathbf { E } ) \geq \cdot \cdot \cdot \geq \sigma _ { C _ { h } } ( \mathbf { E } )$ . Then the best rank-r correction satisfies

$$
\operatorname* { m i n } _ { \mathbf { r a n k } ( \mathbf { L } ) \leq r } \left\| \mathbf { T } ^ { \star } - ( \mathbf { D } + \mathbf { L } ) \right\| _ { F } ^ { 2 } = \sum _ { k > r } \sigma _ { k } ^ { 2 } ( \mathbf { E } ) .\tag{39}
$$

Consequently, the parameterization $\mathbf { D } + \mathbf { U } \mathbf { V } ^ { \top }$ can represent the optimal rank-r approximation of the non-diagonal remainder for the selected D.

Proof: For fixed D, the problem is equivalent to finding the best rank-r approximation to E. Equation 39 follows from the Eckart–Young–Mirsky theorem (Eckart & Young, 1936; Mirsky, 1960), and a truncated singular value decomposition supplies factors U and V. The edgewise scalar gate and normalization used in the implementation can be absorbed into these unconstrained factors whenever the gate is nonzero.

Proposition 1 is an expressivity statement, not an optimization guarantee. It explains why the diagonal term can model channelwise rescaling while a small rank captures dominant cross-channel frame mixing. The residual tail in Equation 39 contributes to the connection defect $\varepsilon _ { i j }$ in Theorem 3; the learned edge network is not assumed to attain the best approximation.

Proposition 2 (Topology-preserving deformation stability). Consider two discretizations X and $\mathcal { X } ^ { \prime }$ of the same physical input, with the same node identities and neighborhood topology, and let $\delta { x } = \operatorname* { m a x } _ { i } \| \mathbf { x } _ { i } ^ { \prime } - \mathbf { x } _ { i } \| _ { 2 }$ . If all geometry-dependent learned features, normalized edge weights, and induced transport matrices are locally Lipschitz functions of the coordinates on a compact family of nondegenerate meshes, then there is a finite $L _ { m }$ such that

$$
\| \mathbf { m } _ { i } ^ { \prime } - \mathbf { m } _ { i } \| _ { 2 } \leq L _ { m } \delta _ { \mathcal { X } } .\tag{40}
$$

Justification: For the same physical sample, decompose the message difference into feature, weight, and transport changes and apply their local Lipschitz bounds. This gives Equation 40. The statement applies only while connectivity is fixed and cells remain nondegenerate; it does not cover folding or topology changes.

Corollary 1 (Propagation to the operator output). Let $\Phi _ { \ell }$ and $\Phi _ { \ell } ^ { \prime }$ denote the ℓth complete featureupdate maps under two admissible descriptions of the same physical input, and let $\bar { \mathbf { S } } _ { \ell }$ denote the corresponding action on layer-ℓ features. Assume

$$
\| \Phi _ { \ell } ^ { \prime } ( \mathbf { S } _ { \ell - 1 } \mathbf { h } ) - \mathbf { S } _ { \ell } \Phi _ { \ell } ( \mathbf { h } ) \| _ { 2 } \leq b _ { \ell }\tag{41}
$$

on the considered compact domain, where $b _ { \ell }$ can be bounded using Theorem 3 together with the deformation contribution in Proposition 2. Suppose $\Phi _ { q } ^ { \prime }$ is $K _ { q ^ { - } }$ Lipschitz. For the complete reconstruction and prediction maps D and $\mathcal { D } ^ { \prime } .$ , assume that $\mathcal { D } ^ { j }$ is $K _ { \mathrm { o u t } ^ { - } }$ Lipschitz and

$$
\begin{array} { r } { \left\| \mathcal { D } ^ { \prime } ( \mathbf { S } _ { L _ { g } } \mathbf { h } ) - \mathbf { S } _ { \mathrm { o u t } } \mathcal { D } ( \mathbf { h } ) \right\| _ { 2 } \leq b _ { \mathrm { o u t } } . } \end{array}\tag{42}
$$

Then the final predictions satisfy

$$
\| \widehat { \mathbf { u } } ^ { \prime } - \mathbf { S } _ { \mathrm { o u t } } \widehat { \mathbf { u } } \| _ { 2 } \leq K _ { \mathrm { o u t } } \sum _ { \ell = 1 } ^ { L _ { g } } \left( \prod _ { q = \ell + 1 } ^ { L _ { g } } K _ { q } \right) b _ { \ell } + b _ { \mathrm { o u t } } .\tag{43}
$$

Here, $\mathbf { S _ { \mathrm { o u t } } }$ is the prescribed output action; it is the identity after reconstruction to the same canonical grid. An empty product in Equation 43 is defined as one.

Proof: Apply Equation 41 at the first layer and propagate its defect through each later Lipschitz map. Repeating this argument for every layer and using the triangle inequality gives the weighted sum in Equation 43. Applying Equation 42 contributes the final term $b _ { \mathrm { o u t } }$

Corollary 1 connects local transport quality to final operator stability, but it also exposes the remaining requirements: small edgewise defects alone are insufficient if later mappings have large Lipschitz constants or are incompatible with the canonical output representation. In the implemented architecture, the transport-continuity and loop terms are auxiliary regularizers rather than exact constraints. Moreover, for the low-rank mode the loop term regularizes the diagonal channel-scaling component only; no exact holonomy claim is made for the full diagonal-plus-low-rank transport.

This analysis is performed on the induced matrix $\mathbf { T } _ { i  j }$ , not on the individual diagonal and low-rank factors. It establishes the limitation of shared direct aggregation, an exact sufficient condition for operator-readable transport, an approximation-capacity result for the implemented parameterization, and local-to-output bounds for the learned approximate case. It does not claim unique recovery of a physical frame or exact invariance of the implemented network. The intervention and deformation studies in Section 4 examine whether the learned connection is functionally used and remains stable, while the message–target discrepancy reported there is treated as a compatibility diagnostic rather than a direct measurement of the latent covariance defect in Equation 36.

## F EXPANDED RESULTS

## F.1 TARGETED COMPONENT ABLATIONS

We remove one component at a time while retaining the remaining training protocol and evaluate each ablation through the failure mode that the removed component is intended to address. All comparisons use frozen checkpoints. To make quantities with different scales visually comparable, Figure 8 reports the ratio between the Full GA-AMNO error and the corresponding w/o-module error. The dashed line at one is the ablated reference; a value below one means that the complete model is better.

1) Adaptive Mesh: difficult-region allocation. We use the target-free high-demand masks defined in Appendix C and measure regional relative $\ell _ { 2 }$ and gradient errors. This test asks whether adaptive allocation improves the regions that exhibit strong input variation and nonuniform geometric demand, rather than whether it uniformly changes the whole field.

2) Spectral Residual: spectral correction. We jointly measure full-field relative $\ell _ { 2 }$ error and highband spectral-energy error. The high band contains Fourier coefficients whose normalized radial frequency is at least 0.6 of the maximum represented radius. Although the implemented Fourier convolution retains truncated low-order modes rather than applying an explicit high-pass mask, this diagnostic tests whether the complete spectral-plus-pointwise correction empirically improves frequency content that is difficult for the mesh-domain prediction alone.

3) Differential Correction: local-structure repair. We measure gradient error and the datasetspecific differential residual used in the evaluation pipeline. For Darcy, the latter is the Laplacian discrepancy defined in Appendix C, not a claim of evaluating the complete coefficient-weighted equation residual. These metrics match the derivative features supplied to the correction head.

4) Low-rank Gauge Transport: geometry-conditioned message correction. We measure highdemand regional relative $\ell _ { 2 }$ error together with residual message–target discrepancy after transport. For the compatibility test, Full GA-AMNO and w/o Low-rank Gauge Transport receive the same GA-AMNO adaptive mesh and identical smooth perturbations, isolating the low-rank correction from mesh allocation.

All bars in Figure 8 are below one. Adaptive allocation reduces difficult-region errors by approximately 1%–12%. The spectral correction gives the largest targeted effect, reducing relative $\ell _ { 2 }$ and high-band errors by 45%–99%; this is an empirical ablation result rather than a claim that the branch is structurally restricted to high frequencies. Differential correction reduces its two local-structure diagnostics by 10%–15%. Low-rank Gauge Transport reduces high-demand prediction error by 4%– 13% and residual message–target discrepancy by 39%–53%. The significance of this experiment is the observed module–failure-mode correspondence: removing a component primarily degrades the quantity that the component was designed to control. This correspondence shows that the architecture is not merely an undifferentiated stack of accuracy modules and links Physics-Informed Adaptive Allocation and Low-rank Gauge Transport to their intended computational roles. It does not imply that every component lowers every global metric on every dataset.

## F.2 SMOOTH ADAPTIVE-MESH PERTURBATIONS

We next test whether the learned message-correction mechanism remains stable after controlled changes to the internally learned mesh. Starting from mesh G, we construct

$$
G _ { k , \ell } = G + \ell \eta d _ { k } ( G ) , \qquad \ell \in \{ 0 , 0 . 2 5 , 0 . 5 0 , 0 . 7 5 , 1 . 0 0 \} ,\tag{44}
$$

where k indexes one of six smooth boundary-tapered transformations, ℓ is the perturbation strength, $d _ { k }$ is the displacement field, and η equals 12% of the mean edge length. The transformations com-

(a) Adaptive Mesh: difficult regions

![](images/9d3cc2577bbe1202f1d6a6741ed969d9e6c145a39f208ec4028994b314ea24c2.jpg)  
(b) Spectral Residual: spectral correction

![](images/a39bba4aea38675727c7265c4c5c1ee8ae520508fe37d3b073eb40f9857d841a.jpg)  
(c) Correction: differential errors  
(d) Gauge: message correction

![](images/1685acd9d56ece413a8ee6c5ecd5fedb7891b675577a8ad7cf29e33f97c646ec.jpg)

![](images/8452f37e4dd3679f310f7bfb29bc7865c9df7e5ece6d453e2868bbd2af31a92e.jpg)  
Figure 8: Targeted component ablations using frozen checkpoints. Each bar is the Full GA-AMNO metric divided by the corresponding w/o-module metric; lower is better and the dashed line denotes the ablated reference. Adaptive Mesh is evaluated in high-demand regions, Spectral Residual through field and high-band spectral errors, Differential Correction through gradient and differential errors, and Low-rank Gauge Transport through regional prediction and residual message–target dis crepancy. Panel (b) uses a logarithmic vertical axis.

Table 3: Mean message-compatibility gain over six smooth transformations and four nonzero strengths.
<table><tr><td>Dataset</td><td>GA-AMNO (%)</td><td>Adaptive Mesh (%)</td><td>Difference</td></tr><tr><td>Navier-Stokes</td><td>83.15</td><td>64.13</td><td>+19.02</td></tr><tr><td>Kolmogorov</td><td>48.05</td><td>6.24</td><td>+41.81</td></tr><tr><td>Darcy</td><td>39.52</td><td>1.35</td><td>+38.17</td></tr></table>

prise smooth horizontal translation, diagonal translation, rotation, sine-modulated rotation, cosinemodulated rotation, and a sine–cosine twist. Connectivity is fixed and every evaluated mesh has a positive cell Jacobian.

Full GA-AMNO is compared with the trained adaptive-mesh ablation without Low-rank Gauge Transport. Both models receive the same GA-AMNO adaptive mesh and the same perturbation, isolating the effect of Low-rank Gauge Transport from resource allocation.

As shown in Table 3, GA-AMNO has higher message-compatibility gain in all 72 dataset– transformation–strength combinations. It also has lower prediction error than the adaptive-mesh ablation in all 24 perturbation conditions on Navier-Stokes and Kolmogorov, but not on Darcy. Moreover, it is not uniformly less sensitive according to a target-free output-change metric. Unlike the previous experiment, which analyzes mismatch already present in the learned mesh, this controlled perturbation test changes the geometry while holding the physical sample and mesh allocation shared between the two models. It therefore verifies that the advantage of low-rank transport persists under multiple smooth, topology-preserving changes of the discretization and is not only a correlation observed on the original mesh. The result supports deformation-robust interaction correction, rather than universal invariance to arbitrary geometry changes.

Table 4: Message–target compatibility on high-geometric-mismatch edges. Lower $E _ { \mathrm { s r c } }$ and $E _ { \mathrm { m s g } }$ indicate smaller discrepancy, whereas a larger positive Gain indicates a stronger improvement produced by low-rank Gauge transport.
<table><tr><td>Dataset</td><td> $E _ { \mathrm { s r c } }$ </td><td> $E _ { \mathrm { m s g } }$ </td><td>Gain (%)</td></tr><tr><td>Navier-Stokes</td><td>0.052823</td><td>0.009543</td><td>81.93</td></tr><tr><td>Kolmogorov</td><td>0.127806</td><td>0.066045</td><td>48.32</td></tr><tr><td>Darcy</td><td>0.287031</td><td>0.123934</td><td>56.82</td></tr></table>

Table 5: Geometric quality of the learned adaptive mesh.
<table><tr><td>Dataset</td><td>Inversion Rate</td><td>Min. Angle</td><td>Aspect Ratio</td></tr><tr><td>Navier-Stokes</td><td>0.0000</td><td>89.019°</td><td>1.0036</td></tr><tr><td>Rayleigh-Bénard</td><td>0.0000</td><td>84.308°</td><td>1.0360</td></tr><tr><td>Kolmogorov</td><td>0.0000</td><td> $8 9 . 7 1 6 ^ { \circ }$ </td><td>1.0038</td></tr><tr><td>Darcy</td><td>0.0000</td><td> $8 9 . 6 6 9 ^ { \circ }$ </td><td>1.0014</td></tr></table>

## F.3 HIGH-MISMATCH MESSAGE COMPATIBILITY

To complement the mismatch-stratified results in Figure 6, we report the absolute compatibility measurements for the high-geometric-mismatch group in Table 4. Here, $E _ { \mathrm { s r c } }$ measures the discrepancy between the untransported source feature and the target representation, while $E _ { \mathrm { m s g } }$ measures the discrepancy after source-to-target transport. The compatibility gain is computed as

$$
\mathrm { G a i n } = \frac { E _ { \mathrm { s r c } } - E _ { \mathrm { m s g } } } { E _ { \mathrm { s r c } } } \times 1 0 0 \% .
$$

Transport consistently lowers message–target discrepancy, with gains ranging from 48.32% to 81.93%. These results provide the absolute values underlying the stratified comparison in the main text.

## F.4 MESH QUALITY

Finally, Table 5 verifies that the learned allocation is realized on valid geometries. KS is omitted because it is one-dimensional. All evaluated two-dimensional meshes have zero inversion rate, minimum cell angles above $8 4 ^ { \circ }$ , and aspect ratios close to one. This experiment rules out a degenerate explanation in which prediction gains are obtained through folded cells, invalid neighborhoods, or extreme element anisotropy. These quantities are geometric feasibility controls rather than evidence of operator readability by themselves; the counterfactual and transport experiments above establish whether the valid adaptive geometry and its interactions are functionally meaningful.

Taken together, the experiments support a layered conclusion rather than relying on prediction error alone. The main comparison establishes operator utility; component ablation assigns improvements to intended failure modes; importance counterfactuals test allocation readability; edge intervention and mismatch stratification test whether representation correction is both consequential and geometrically targeted; controlled deformation evaluates whether that mechanism persists when the discretization changes; and mesh diagnostics exclude invalid geometry as an explanation. This chain of evidence supports the central claim that operator readability increases when both node allocation and cross-node interaction can be inspected and functionally tested. It does not establish exact gauge covariance, unique recovery of latent physical frames, or invariance to arbitrary external meshes.

## F.5 CROSS-RESOLUTION GENERALIZATION

Table 6 evaluates the proposed model under resolution changes. The model is trained at the canonical 64 × 64 resolution and directly evaluated at resolutions from 16 to 128. Only relative $\ell _ { 2 }$ error is reported to focus on the overall reconstruction robustness under discretization changes.

The lowest error is consistently obtained at the native resolution. Nevertheless, the model remains accurate at nearby resolutions. For example, the Navier-Stokes error is 0.013443 at resolution 48 and 0.009215 at resolution 96, compared with 0.003219 at the native resolution. Similar trends are observed for Rayleigh-Benard convection, Kuramoto–Sivashinsky, and Kolmogorov flow.´

Table 6: Cross-resolution generalization of GA-AMNO. The model is trained at $6 4 \times 6 4$ . Relative $\ell _ { 2 }$ error is reported; lower is better.
<table><tr><td>Dataset</td><td>16</td><td>32</td><td>48</td><td>64</td><td>96</td><td>128</td></tr><tr><td>Navier-Stokes</td><td>0.175885</td><td>0.041609</td><td>0.013443</td><td>0.003219</td><td>0.009215</td><td>0.012195</td></tr><tr><td>Rayleigh-Bénard</td><td>0.202487</td><td>0.068442</td><td>0.033256</td><td>0.000680</td><td>0.019584</td><td>0.020235</td></tr><tr><td>KS</td><td>0.172811</td><td>0.077675</td><td>0.045117</td><td>0.000387</td><td>0.013415</td><td>0.007378</td></tr><tr><td>Kolmogorov</td><td>0.431518</td><td>0.169014</td><td>0.064861</td><td>0.018796</td><td>0.058290</td><td>0.076400</td></tr><tr><td>Darcy</td><td>0.351837</td><td>0.132896</td><td>0.061258</td><td>0.027743</td><td>0.083502</td><td>0.113239</td></tr></table>

![](images/0e5ee03bc6cf546cdb24380449c79dc2cf9f2b238d5bc5969b562baee6704073.jpg)  
Figure 9: GA-AMNO predictions under cross-resolution evaluation. Columns show the ground truth and predictions at mesh resolutions 16, 32, 48, 64, 96, and 128; rows correspond to the five PDE benchmarks. The native training resolution is 64, and colors are independently scaled within each panel.

The error increases more noticeably at extremely coarse resolutions, especially for Kolmogorov flow and Darcy flow. This behavior is expected because 16 × 16 sampling cannot represent localized vortices, sharp gradients, or heterogeneous permeability patterns adequately. The results indicate that the learned gauge-aware representation is stable under moderate resolution shifts, while its accuracy remains fundamentally tied to the information available in the input discretization.

To complement the quantitative results in Table 6, Figure 9 shows representative predictions across the evaluated resolutions. The visualizations confirm the same resolution-dependent pattern: severe downsampling removes localized and high-frequency structures, whereas resolutions close to or finer than the training resolution preserve the dominant field morphology. Because each panel is scaled independently, the figure is intended to illustrate structural preservation rather than absolute amplitude agreement or resolution invariance.

## F.6 ROLLOUT STABILITY

Table 7 reports autoregressive rollout results of GA-AMNO. Darcy flow is not included because it is a static PDE operator-learning problem. We report prediction errors at one, five, and ten rollout steps.

For Navier-Stokes, the relative $\ell _ { 2 }$ error increases from 0.003213 at one step to 0.030052 after ten steps. Rayleigh-Benard convection remains particularly stable, with a ten-step error of only´

Table 7: Autoregressive rollout results of GA-AMNO. Lower is better. ‘Spec.’ denotes the overall spectral-energy error SpectrumError.
<table><tr><td>Dataset</td><td>Step</td><td>Rel. L2</td><td>RMSE</td><td>Spec.</td></tr><tr><td>Navier-Stokes</td><td>1</td><td>0.003213</td><td>0.002604</td><td>0.001566</td></tr><tr><td>Navier-Stokes</td><td>5</td><td>0.016251</td><td>0.014324</td><td>0.011963</td></tr><tr><td>Navier-Stokes</td><td>10</td><td>0.030052</td><td>0.027607</td><td>0.024469</td></tr><tr><td>Rayleigh-Bénard</td><td>1</td><td>0.000692</td><td>0.000085</td><td>0.000195</td></tr><tr><td>Rayleigh-Bénard</td><td>5</td><td>0.004747</td><td>0.000612</td><td>0.001478</td></tr><tr><td>Rayleigh-Bénard</td><td>10</td><td>0.012066</td><td>0.001595</td><td>0.004401</td></tr><tr><td>KS</td><td>1</td><td>0.000407</td><td>0.000223</td><td>0.000339</td></tr><tr><td>KS</td><td>5</td><td>0.002140</td><td>0.001146</td><td>0.001753</td></tr><tr><td>KS</td><td>10</td><td>0.004134</td><td>0.002108</td><td>0.003505</td></tr><tr><td>Kolmogorov</td><td>1</td><td>0.018522</td><td>0.018065</td><td>0.006738</td></tr><tr><td>Kolmogorov</td><td>5</td><td>0.100077</td><td>0.097531</td><td>0.064080</td></tr><tr><td>Kolmogorov</td><td>10</td><td>0.253117</td><td>0.247677</td><td>0.165009</td></tr></table>

![](images/b27d6fb8cb696c6de33e1643fd0e80de41b69b53b2c4f6ed57b24524ede45268.jpg)  
Figure 10: Selected GA-AMNO autoregressive predictions at rollout steps 1, 3, 5, 7, and 10 for Navier-Stokes, Rayleigh-Benard, KS, and Kolmogorov. Colors are independently scaled within´ each panel, and the one-dimensional KS profile is repeated vertically for visualization.

0.012066. Kuramoto–Sivashinsky also maintains a low ten-step error of 0.004134, despite its chaotic temporal evolution. Kolmogorov flow has the largest long-horizon error, increasing from 0.018522 to 0.253117, which reflects the accumulation of phase and vortex-location errors in forced turbulent dynamics.

These results suggest that the proposed model preserves stable temporal evolution for moderate rollout horizons. The differential residual correction helps suppress local high-frequency drift, while Physics-Informed Adaptive Allocation and Low-rank Gauge Transport reduce the propagation of representation mismatch across time. The remaining degradation on Kolmogorov flow highlight that strongly forced turbulence remains challenging for purely autoregressive neural operators.

The qualitative rollouts in Figure 10 provide a visual counterpart to the error growth reported in Table 7. Navier-Stokes, Rayleigh-Benard, and KS retain their dominant structures over the displayed´ horizon, whereas Kolmogorov flow exhibits increasingly visible phase and vortex-location deviations. This behavior is consistent with its faster quantitative error accumulation. Darcy is excluded because it defines a steady coefficient-to-solution mapping rather than a temporal rollout task.

![](images/8fb775633398d6aec3302b2d62b90cd0db8f39e9c15f17f02518edf97439ca44.jpg)  
Figure 11: Representative native-resolution predictions. Columns compare the ground truth, GA AMNO, Transolver, FNO, GNO, WNO, and UNet; rows correspond to Navier-Stokes, Rayleigh-Benard, KS, Kolmogorov, and Darcy. Colors are independently scaled within each panel to empha-´ size spatial structure.

## F.7 QUALITATIVE COMPARISON AT NATIVE RESOLUTION

Figure 11 complements the aggregate errors in Table 1 with one representative prediction at the native evaluation resolution. GA-AMNO preserves the dominant spatial structures across all five PDE families, including localized flow patterns and heterogeneous Darcy responses. Each panel is scaled independently to [0, 1] for structural comparison, so color should not be interpreted as a shared physical amplitude across models or datasets. The one-dimensional KS profile is repeated vertically only to maintain a common field-map layout; quantitative conclusions remain those of Table 1.

## G NOTATION

The following notation summarizes the symbols used in the method, evaluation, and theoretical analysis. Superscripts r and a denote reference and adaptive discretizations, respectively. A prime denotes a quantity after an admissible change of local representation convention.

<table><tr><td> $\Omega$ </td><td>Spatial domain.</td></tr><tr><td> $d$ </td><td>Spatial dimension.</td></tr><tr><td> $\mathcal { F } _ { \mathrm { P D E } }$ </td><td>Governing differential operator.</td></tr><tr><td> $\pmb { \mu }$ </td><td>Physical parameters, coefficient fields, or forcing terms.</td></tr><tr><td> $\mathcal { A }$ </td><td>Input function space of the PDE solution operator.</td></tr><tr><td> $\mathcal { U }$ </td><td>Output solution function space of the PDE solution opera- tor.</td></tr><tr><td>a</td><td>Temporal input history or steady conditioning field.</td></tr><tr><td> $\mathcal { U } _ { t }$ </td><td>Input history ending at time t.</td></tr><tr><td>V</td><td>Latest input state or conditioning field.</td></tr><tr><td>u</td><td>Ground-truth physical field.</td></tr><tr><td> $\widehat { \mathbf { u } }$ </td><td>Predicted physical field.</td></tr><tr><td> $\mathcal { G } ^ { \dagger }$ </td><td>Ground-truth PDE solution operator.</td></tr><tr><td> $\mathcal { G } _ { \theta }$ </td><td>Learned neural operator parameterized by θ.</td></tr><tr><td></td><td>Adaptive discretization and state encoding</td></tr><tr><td> $\mathcal { X } ^ { r }$ </td><td>Reference discretization or canonical grid.</td></tr><tr><td> $\mathbf { X } ^ { r }$ </td><td>Canonical coordinate tensor used by the grid-based branches.</td></tr><tr><td> $\mathcal { X } ^ { a }$ </td><td>Input-dependent adaptive discretization.</td></tr><tr><td> $\mathbf { x } _ { i } ^ { r }$ </td><td>Coordinate of reference node i.</td></tr><tr><td> $\mathbf { x } _ { i } ^ { a }$ </td><td>Coordinate of adaptive node ¿.</td></tr><tr><td> $\mathbf { P }$ </td><td>Encoded state descriptor or physics-proxy feature map.</td></tr><tr><td> $\mathbf { p } _ { i }$ </td><td>State descriptor at node i.</td></tr><tr><td> $Q _ { g }$ </td><td>Gradient-magnitude physical indicator.</td></tr><tr><td> $Q _ { \Delta }$ </td><td>Laplacian-magnitude physical indicator.</td></tr><tr><td> $Q _ { E }$ </td><td>Local energy indicator.</td></tr><tr><td> $Q _ { \omega }$ </td><td>Vorticity-related indicator.</td></tr><tr><td> $\widetilde { \mathbf { Q } } ( \mathbf { x } )$ </td><td>Concatenated normalized physical-indicator vector at loca- tion x.</td></tr><tr><td> $I ( \mathbf { x } )$ </td><td>Predicted scalar importance map at spatial location x.</td></tr><tr><td> $\mathcal { N } ( \cdot )$ </td><td>Per-sample channel-normalization operator.</td></tr><tr><td> $\eta _ { \theta }$ </td><td>Convolutional importance-map predictor.</td></tr><tr><td> $\sigma$ </td><td>Sigmoid activation used for importance prediction.</td></tr><tr><td></td><td>Mesh dimensions and adaptive movement</td></tr></table>

<table><tr><td>N</td><td>Number of nodes or scalar field entries, according to con- text.</td></tr><tr><td> $C$ </td><td>Number of physical channels.</td></tr><tr><td> $C _ { h }$ </td><td>Hidden feature width.</td></tr><tr><td> $I _ { i }$ </td><td>Learned importance value at node ¿.</td></tr><tr><td> $w _ { i }$ </td><td>Positive mean-normalized importance weight.</td></tr><tr><td> $\mathbf { o } _ { i }$ </td><td>Bounded raw displacement direction.</td></tr><tr><td> $\Delta { { \bf { x } } _ { i } }$ </td><td>Importance-modulated node displacement.</td></tr><tr><td> $\delta _ { \mathrm { m a x } }$ </td><td>Maximum allowed displacement magnitude</td></tr><tr><td> $\psi _ { \boldsymbol { \theta } }$ </td><td>Convolutional head that predicts the bounded raw node dis- placement.</td></tr><tr><td> $I _ { \mathrm { m i n } }$ </td><td>Positive lower bound used in importance modulation.</td></tr><tr><td> $s _ { I }$ </td><td>Scale controlling the range of importance modulation.</td></tr><tr><td> $\Pi _ { \Omega }$ </td><td>Projection that keeps displaced nodes inside the spatial do- mainΩ.</td></tr><tr><td> $\mathbf { J } _ { i } ^ { a }$ </td><td>Local adaptive-mesh Jacobian.</td></tr><tr><td></td><td>Gauge-connection parameterization</td></tr><tr><td> ${ \bf e } _ { i j }$ </td><td>Edge descriptor for directed edge  $j  i .$ </td></tr><tr><td> $r$ </td><td>Rank of the low-rank transport correction.</td></tr><tr><td> $\mathbf { d } _ { i j } ^ { ( \ell ) }$ </td><td>Diagonal transport factors at layer l.</td></tr><tr><td> $\mathbf { U } _ { i j } ^ { ( \ell ) }$ </td><td>Left low-rank transport factor.</td></tr><tr><td> $\mathbf { V } _ { i j } ^ { ( \ell ) }$ </td><td>Right low-rank transport factor.</td></tr><tr><td> $\gamma _ { i j } ^ { ( \ell ) }$ </td><td>Gate controlling the low-rank correction.</td></tr><tr><td> $\mathbf { T } _ { i  j } ^ { ( \ell ) }$ </td><td>Learned source-to-target feature transport.</td></tr><tr><td></td><td>Attention and transported aggregation</td></tr><tr><td> $a _ { i j } ^ { ( \ell ) }$ </td><td>Unnormalized edge-attention score.</td></tr><tr><td> $\alpha _ { i j } ^ { ( \ell ) }$ </td><td>Softmax-normalized edge-attention weight.</td></tr><tr><td> $\mathbf { h } _ { i } ^ { ( \ell ) }$ </td><td>Feature of node i at layer l.</td></tr><tr><td> $\mathbf { m } _ { i } ^ { ( \ell ) }$ </td><td>Aggregated transported message at node i.</td></tr><tr><td> $\mathcal { N } ( i )$ </td><td>Neighborhood of node i.</td></tr><tr><td> $\rho _ { \theta } ^ { ( \ell ) }$ </td><td>Learned residual feature-update network at layer l.</td></tr><tr><td> $L _ { g }$ </td><td>Number of low-rank gauge transport layers.</td></tr><tr><td>€</td><td></td></tr><tr><td></td><td>Positive numerical-stabilization constant. Mesh-to-grid reconstruction</td></tr><tr><td> $\mathbf { y } _ { q } ^ { r }$ </td><td>Coordinate of canonical output point  $q .$ </td></tr><tr><td> $\mathcal { N } _ { b } ( q )$ </td><td>Adaptive-node neighborhood used for reconstruction.</td></tr><tr><td> $\kappa _ { q i }$ </td><td>Interpolation weight from adaptive node i to output point  $q .$ </td></tr><tr><td> $\tau _ { b }$ </td><td>Distance temperature of the mesh-to-grid bridge.</td></tr><tr><td> $\mathbf { z } _ { q }$ </td><td>Reconstructed feature at canonical point  $q .$ </td></tr><tr><td> $\mathbf { F } _ { \mathrm { g r i d } }$ </td><td>Canonical grid feature tensor.</td></tr><tr><td></td><td>Residual prediction and differential operators</td></tr><tr><td> $\delta _ { \mathrm { r a w } }$ </td><td>Residual predicted by the local branch.</td></tr><tr><td> $\delta _ { \mathrm { s p e c } }$ </td><td>Residual predicted by the spectral branch.</td></tr><tr><td> $\delta _ { 1 }$ </td><td>Sum of local and spectral residuals.</td></tr><tr><td> $\mathbf { u } _ { \mathrm { b a s e } }$ </td><td>Provisional physical field used for differential correction.</td></tr><tr><td> $\mathbf { Z } _ { \mathrm { p h y } }$ </td><td>Input tensor of the differential correction head.</td></tr><tr><td> $\pmb { \delta }$ </td><td>Final corrected residual.</td></tr><tr><td> $D _ { x }$ </td><td>Discrete derivative in the first spatial direction.</td></tr><tr><td> $D _ { y }$ </td><td>Discrete derivative in the second spatial direction.</td></tr><tr><td> $\Delta$ </td><td>Discrete Laplacian operator.</td></tr><tr><td> $\mathcal { F } _ { \mathrm { D F T } }$ </td><td>Orthonormal discrete Fourier transform.</td></tr><tr><td></td><td>Feature-compatibility evaluation</td></tr><tr><td> $\mathcal { E } _ { b }$ </td><td>Directed-edge set in geometric-mismatch group b.</td></tr><tr><td> $E _ { \mathrm { s r c } } ^ { ( b ) }$ </td><td>Source-to-target feature discrepancy in group  $b .$ </td></tr><tr><td> $E _ { \mathrm { m s g } } ^ { ( b ) }$ </td><td>Transported-message-to-target discrepancy in group b.</td></tr><tr><td> $G _ { \mathrm { c o m p } } ^ { ( b ) }$ </td><td>Relative compatibility gain produced by transport.</td></tr><tr><td> $\mathbf { g } _ { i }$ </td><td>Local discretization context at node i.</td></tr><tr><td> $\mathbf { z } _ { i }$ </td><td>Latent physical content at node i.</td></tr><tr><td> $\mathbf { R } ( \mathbf { g } _ { i } )$ </td><td>Representation map induced by local context  $\mathbf { g } _ { i } .$ </td></tr><tr><td></td><td>Local representation analysis</td></tr><tr><td>G</td><td>Admissible group of local representation changes.</td></tr><tr><td> $\mathfrak { D }$ </td><td>Family of admissible nondegenerate discretization con- texts.</td></tr><tr><td> ${ \mathcal { O } } ( \mathbf { z } _ { i } )$ </td><td>Admissible representation set of physical content  $\mathbf { z } _ { i } .$  </td></tr><tr><td> $\mathbf { S } _ { i }$ </td><td>Local representation action at node i.</td></tr><tr><td> $\overline { { \mathbf { m } } } _ { i }$ </td><td>Message produced by uncorrected direct aggregation.</td></tr><tr><td>W</td><td>Shared linear map in direct aggregation.</td></tr><tr><td></td><td>Connection consistency and ideal transport</td></tr><tr><td> $\delta _ { \alpha , i }$ </td><td>Attention-consistency defect at node i.</td></tr><tr><td> $\varepsilon _ { i j }$ </td><td>Connection-consistency defect on edge  $j  i .$ </td></tr><tr><td> $\mathbf { T } _ { i  j } ^ { \star }$ </td><td>Ideal source-to-target transport matrix.</td></tr><tr><td> $\mathbf { T } ^ { \star }$ </td><td>Ideal transport matrix for a representative edge in the ap- proximation analysis.</td></tr><tr><td>D</td><td>Diagonal component of an ideal transport.</td></tr><tr><td>E</td><td>Non-diagonal remainder  $\mathbf { T } ^ { \star } - \mathbf { D } .$ </td></tr><tr><td> $\sigma _ { k } ( { \bf E } )$ </td><td>The kth singular value of E.</td></tr><tr><td></td><td>Mesh and operator stability constants</td></tr><tr><td> $\delta _ { X }$ </td><td>Maximum coordinate change between two meshes.</td></tr><tr><td> $L _ { m }$ </td><td>Local Lipschitz constant for message variation.</td></tr><tr><td> $\Phi _ { \ell }$ </td><td>Complete feature-update map at layer l.</td></tr><tr><td> $b _ { \ell }$ </td><td>Representation-consistency defect at layer l.</td></tr><tr><td> $K _ { \ell }$ </td><td>Lipschitz constant of layer l.</td></tr><tr><td> $\mathcal { D }$ </td><td>Canonical reconstruction and prediction map.</td></tr><tr><td> $K _ { \mathrm { o u t } }$ </td><td>Lipschitz constant of the output map.</td></tr><tr><td> $\mathbf { S _ { \mathrm { o u t } } }$ </td><td>Prescribed representation action on the output.</td></tr><tr><td> $b _ { \mathrm { o u t } }$ </td><td>Representation-consistency defect of the output map.</td></tr></table>

Mesh-to-grid reconstruction

## AI USE STATEMENT

In this work, we used generative AI tools to assist with language polishing and limited programming support. All scientific ideas, methodology, experimental design, implementation decisions, analysis, and conclusions were developed and verified by the authors. All AI-assisted content was carefully reviewed and revised by the authors. We take full responsibility for the final content of this work.