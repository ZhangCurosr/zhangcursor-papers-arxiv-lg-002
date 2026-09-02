# WHY MULTI-LAYER MESSAGE PASSING WORKS: COMPLETENESS THEORY FOR GRAPH NEURAL NETWORK INTERATOMIC POTENTIALS

PINGBING MING<sup>∗</sup> AND HAN WANG<sup>†</sup>

Abstract. We prove that the Hypergraph Neural Network, an invariant architecture with 3-body message passing, is a universal approximator for potential energy surfaces. Our main contribution is a multi-layer completeness theory. We show that L layers of message passing on sparse, cutof-based graphs achieve the same representational power as having access to the full L-hop neighborhood, provided the configurations are generic, satisfy an overlap condition and a connectivity condition. This provides the first rigorous justification for the common practice of using multi-layer message passing with a per-layer cutof smaller than the physical interaction range, the setting used by virtually all practical graph neural network based machine-learned interatomic potentials. As immediate consequences, we show that both DPA3 and CHGNet architectures inherit universal approximation.

Key words. machine-learned interatomic potentials, graph neural networks, hypergraph neural networks, universal approximation

MSC codes. 68T07, 68R10, 41A65, 92E10

1. Introduction. Machine-learned interatomic potentials (MLIPs) [6, 2, 42] have become an essential tool in computational chemistry and materials science, enabling molecular dynamics simulations that achieve near-quantum-mechanical accuracy at a fraction of the computational cost. Applications range from drug discovery [37] and catalysis [21] to materials design [9] and phase diagram exploration [38]. Recent advances in atomistic foundation models [41, 3, 24, 10] have further expanded the scope of MLIPs to universal potentials covering much of the periodic table. This raises a fundamental question: can a multi-layer GNN-based MLIP approximate an arbitrary potential energy surface (PES) while rigorously preserving the required physical symmetries and continuity constraints? The answer to this question is of central importance, as it determines whether the predictive accuracy of such models is inherently limited by the architectural design, or whether it ultimately depends only on the training data and optimization procedure.

A closely related concept is completeness. A model is said to be complete if it produces distinct outputs for any two local environments that are not related by symmetry, that is, two environments yield the same output only if they are related by a rotation or reflection combined with a permutation of atoms of the same species. Completeness is a necessary condition for universal approximation, yet whether it is also suficient is a separate question that we shall address in this work. The concept was introduced by Bart´ok et al. [1] in the context of invariant representations, and brought into sharp focus by Pozdnyakov et al. [29], who demonstrated that the widely used 3-body representations, such as SOAP power spectra, are incomplete, in that there exist geometrically distinct environments that share identical sets of pairwise distances and triplet angles. Incompleteness thus imposes a fundamental ceiling on approximation accuracy that cannot be overcome, regardless of the training data.

Further incompleteness results have since revealed the scope of the problem. Pozdnyakov and Ceriotti [27] showed that distance-only GNNs, such as SchNet [30], are incomplete for 3D point clouds even with arbitrary depth and nonlinearity, so angular information is not merely helpful but necessary. Cen et al. [8] showed that equivariant GNNs with fixed tensor degree l degenerate to zero on symmetric structures for specific values of l, e.g., l = 1 fails on all symmetric graphs, demonstrating that low-degree equivariant features are also insuficient.

Several approaches have been developed to achieve completeness. The Atomic Cluster Expansion (ACE) [14, 15] and Moment Tensor Potentials (MTP) [32] construct systematically improvable polynomial bases for invariant functions. These are linear models for which basis density directly yields universal approximation. Equivariant message-passing architectures such as Tensor Field Networks [34], NequIP [5], PaiNN [31], SEGNN [7], and Equivariant Transformers [33] use E(3)-equivariant convolutions to propagate vector and tensor features, achieving high data eficiency but without a formal completeness guarantee. Building on this line of work, MACE [4] achieves completeness by constructing ACE basis functions within the GNN framework: tensor products of 2-body equivariant features are symmetrized via Clebsch– Gordan coeficients to build (ν+1)-body features that are mathematically equivalent to the ACE basis, so completeness follows directly from ACE theory. A diferent approach processes invariant scalar features, namely distances and bond angles, with nonlinear networks. Models in this class include DimeNet [17], GemNet [16], SphereNet [23], ALIGNN [11], M3GNet [9], CHGNet [13], and DPA3 [41]. Finally, coordinate-frame methods construct local reference frames from pairs of neighbors and apply nonlinear functions before symmetrization: Nigam et al. [25] proposed 3- center-1-neighbor representations, and Pozdnyakov and Ceriotti [28] introduced the Equivariant Coordinate System Ensemble (ECSE).

On the theoretical side, several foundational results underpin the analysis of model expressiveness. The DeepSets universality theorem [40, 36, 18] provides established that $\phi \circ ( \sum _ { j } g )$ is a universal approximator for permutation-invariant functions. Villar et al. [35] proved that any function invariant under translations, rotations, reflections, and permutations can be expressed as a permutation-invariant function of the Gram matrix of displacement vectors, a result known as the “scalars are universal” theorem. Li et al. [22] demonstrated that several invariant GNN architectures, such as DimeNet, SphereNet and GemNet, achieve E(3)-completeness on fully connected graphs. Joshi et al. [20] introduced the Geometric Weisfeiler–Leman test, covering both invariant and equivariant models, and proved an equivalence between pairwise discrimination within a model class and universal approximation.

Despite this progress, significant gaps remain. The completeness results of Villar et al. [35] and Li et al. [22] are established only for fully connected graphs; for the sparse, cutof-based graphs commonly employed in practice, completeness remains an open question. Villar et al. characterize what form invariant functions must take, but do not show any specific architecture can actually approximate them. Nigam et al. [25] identify the correct geometric insight, namely that three-center features with nonlinearity applied before summation achieve completeness, yet their argument is presented as a proof sketch rather than a rigorous derivation. The discrimination– UAT equivalence of Joshi et al. [20] operates on the function class, requiring only that for each pair of environments some function in the class separates them. This is conceptually diferent from completeness of a single model with fixed weights. Moreover, to the best of our knowledge, all existing theoretical analyses operate within a single cutof neighborhood. In practice, however, multi-layer message passing extends the receptive field from the per-layer cutof $r _ { c }$ to a larger physical interaction range $R _ { c } > r _ { c } ,$ and the impact of this extended range on completeness has not yet been addressed.

We close these gaps by introducing the Hypergraph Neural Network (HGNN), an invariant 3-body architecture, and proving the following multi-layer completeness theorem for HGNN:

Main result (see Theorem 5.1). On generic configurations satisfying an overlap and a connectivity condition, and for L large enough that the L-hop neighborhood covers the physical interaction range $R _ { c }$ , the L-layer HGNN is a universal approximator for the PES at range $R _ { c }$

The above result extends completeness from the per-layer cutof $r _ { c }$ to the physical range $R _ { c }$ via multi-layer message passing, a setting not addressed in prior work. The proof operates on sparse cutof-based graphs, provides a rigorous continuous approximation guarantee for the HGNN, and characterizes a single fixed-weight representation rather than a function class. Beyond this, the HGNN serves as a reference theoretical architecture: any practical architecture that can simulate the HGNN inherits its universal approximation property. We demonstrate this implication explicitly with DPA3 and CHGNet, i.e., Corollaries 5.2 and 5.4. The proofs further reveal one key architectural requirement, that message functions must be Multi-Layer Perceptrons (MLPs) [12, 19], rather than single linear layers, to ensure the desired expressiveness.

The paper is organized as follows. §2 introduces the PES properties, symmetry framework, and the GNN-based interatomic potential architectures considered in this work, namely, HGNN, ALIGNN, CHGNet, DPA3. §3 proves the completeness–UAT equivalence and establishes the existence of complete representations via Gram matrix reconstruction. §4 proves that the HGNN can approximate complete representations. $\ S 5$ states the UAT and derives corollaries for DPA3 and CHGNet, with the simulation proofs given in the supplementary materials. §6 discusses implications, assumptions, limitations, and open problems.

## 2. Graph Neural Network Interatomic Potentials.

2.1. Potential energy surface and target function class. Consider a system of N atoms with types $z _ { i } \in \mathcal { Z } = \{ 1 , . . . , T \}$ and pairwise distinct positions $\pmb { r } _ { i } \in \mathbb { R } ^ { 3 }$ Write $B _ { r } = \{ r \in \mathbb { R } ^ { 3 } : | r | < r \}$ for the open ball of radius $^ r .$ We assume the density is bounded, in that the number of atoms in any ball of fixed radius admits a uniform bound. We write M for such a bound at the largest cutof appearing below.

Definition 2.1 (Local environment). The local environment of atom j within cutof r is

$$
\mathcal { D } _ { j } ^ { ( r ) } = \big ( z _ { j } , \{ ( \Delta r _ { j k } , z _ { k } ) : k \neq j , \Delta r _ { j k } \in B _ { r } \} \big ) , \qquad \Delta r _ { j k } = r _ { k } - r _ { j } ,\tag{2.1}
$$

where the neighbor set is unordered (a multiset). That is, $\mathcal { D } _ { i } ^ { ( r ) }$ consists of the center type $z _ { j }$ together with the displacement vectors and types of all neighbors within r. We write $\dot { \mathcal { D } } _ { j } \equiv \mathcal { D } _ { j } ^ { ( r _ { c } ) }$ when the cutof is the default $r _ { c }$

A PES is a map $E : \bigcup _ { N = 1 } ^ { \infty } ( \mathbb { R } ^ { 3 } \times \mathcal { Z } ) ^ { N } \to \mathbb { R }$ subject to four structural properties:

(P1) Symmetry. E is invariant under translations, rotations/reflections, and permutation of same-type atoms:

(2.2)

$$
E ( \{ r _ { i } + t , z _ { i } \} ) = E ( \{ r _ { i } , z _ { i } \} ) \quad \qquad \forall t \in \mathbb { R } ^ { 3 } ,\tag{2.3}
$$

$$
E ( \{ R r _ { i } , z _ { i } \} ) = E ( \{ r _ { i } , z _ { i } \} ) \quad \qquad \forall R \in O ( 3 ) ,\tag{2.4}
$$

$$
E \bigl ( \{ r _ { \pi ( i ) } , z _ { i } \} \bigr ) = E \bigl ( \{ r _ { i } , z _ { i } \} \bigr ) \qquad \forall \pi \in S _ { N _ { 1 } } \times \cdot \cdot \cdot \times S _ { N _ { T } } ,
$$

where $N _ { t } = | \{ i : z _ { i } = t \} | , S _ { n }$ denotes the symmetric group on n elements, and $\pi$ permutes only positions of same-type atoms. Each atom’s type is an intrinsic attribute that travels with it under permutation. Swapping two atoms exchanges their positions while each retains its type, so only same-type swaps leave the configuration unchanged.

(P2) Extensivity. E decomposes as a sum of atomic contributions:

$$
E ( \{ r _ { i } , z _ { i } \} _ { i = 1 } ^ { N } ) = \sum _ { i = 1 } ^ { N } \varepsilon _ { i } .\tag{2.5}
$$

(P3) Locality. There exists a physical interaction range $R _ { c } ~ > ~ 0$ such that $\varepsilon _ { i }$ depends only on atoms within $R _ { c }$ of atom i: $\varepsilon _ { i } = \varepsilon ( D _ { i } ^ { ( R _ { c } ) } )$

$( P \ 4 )$ Smoothness. ε is at least $C ^ { k } \left( k \geq 2 \right)$ with respect to the neighbor positions, including at the interaction-range boundary, where neighbors enter and leave the environment. Since ε is $C ^ { k }$ , the forces are $\scriptstyle { \dot { C } } ^ { k - 1 }$ and the Hessian is $C ^ { k - 2 }$ . Requiring $k \geq 2$ therefore makes all three continuous.

The extensivity and locality properties reduce the problem from a function on a 3N-dimensional configuration space to a function on the space of local environments, which has bounded dimensionality. We now formalize this space.

Definition 2.2 (Environment space). For a fixed cutof $r > 0$ and a type signature $( z _ { 0 } , z )$ with $z _ { 0 } \in { \mathcal { Z } } , z \in { \mathcal { Z } } _ { < } ^ { n } = \{ ( z _ { 1 } , . . . , z _ { n } ) \in { \mathcal { Z } } ^ { n } : z _ { 1 } \leq \dots \leq z _ { n } \}$ , and $n \leq M$ , the stratum $\mathcal { E } _ { z _ { 0 } , z } ^ { ( r ) }$ is the set of ordered lists of the neighbors of a local environment within cutof r (Definition 2.1) whose center has type $z _ { 0 }$ , arranged so that the k-th neighbor has type $z _ { k }$ . Since the types are fixed by the signature, such a list is determined by its tuple of displacement vectors, written

$$
\begin{array} { r } { \mathcal { D } = ( \Delta \pmb { r } _ { 1 } , \dots , \Delta \pmb { r } _ { n } ) \in \left( B _ { r } \setminus \{ 0 \} \right) ^ { n } \subset \mathbb { R } ^ { 3 n } . } \end{array}\tag{2.6}
$$

Every such tuple occurs, so the stratum is an open subset of $\mathbb { R } ^ { 3 n }$ and inherits its Euclidean metric, standard topology, and $C ^ { \infty }$ diferentiable structure. The type signature $( z _ { 0 } , z )$ is index data rather than set data, so strata with diferent signatures are disjoint even when they have equal dimension. The environment space with cutof r is the disjoint union over all center types and neighbor compositions:

$$
\mathcal { E } ^ { ( r ) } = \bigsqcup _ { z _ { 0 } \in \mathcal { Z } } \ \bigsqcup _ { n = 0 } ^ { M } \bigsqcup _ { z \in \mathcal { Z } _ { \leq } ^ { n } } \mathcal { E } _ { z _ { 0 } , z } ^ { ( r ) } .\tag{2.7}
$$

We write $\mathcal { E } \equiv \mathcal { E } ^ { ( r _ { c } ) }$ and $\mathcal { E } _ { z _ { 0 } , z } \equiv \mathcal { E } _ { z _ { 0 } , z } ^ { ( r _ { c } ) }$ when the cutof is the default $r _ { c }$

The Euclidean metric on each stratum is

$$
d ( \mathcal D , \mathcal D ^ { \prime } ) = \Bigl ( \sum _ { k = 1 } ^ { n } \| \Delta \pmb { r } _ { k } - \Delta \pmb { r } _ { k } ^ { \prime } \| ^ { 2 } \Bigr ) ^ { 1 / 2 } , \qquad \mathcal D , \mathcal D ^ { \prime } \in \mathscr E _ { z _ { 0 } , z } ^ { ( r ) } .\tag{2.8}
$$

Two tuples in the same stratum list the same local environment precisely when they difer by a type-preserving permutation of the neighbors, which is part of the symmetry group of Definition 2.3. We refer to elements of ${ \mathcal { E } } ^ { ( r ) }$ simply as local environments. Since the type tuple is nondecreasing, every local environment has all of its representatives in a single stratum. The mathematical structure of $\mathcal { E }$ is stratum-wise, in that continuity, compactness, and $C ^ { k }$ -smoothness are defined with respect to the Euclidean metric and diferentiable structure on each stratum $\mathcal { E } _ { z _ { 0 } , z } ^ { ( r ) } \subset \mathbb { R } ^ { 3 n }$ . Globally, ${ \mathcal { E } } ^ { ( r ) }$ is a disjoint union of strata of diferent dimensions, while it is not a linear space.

The symmetry group acting on a stratum of $\mathcal { E }$ with n neighbors of types $_ { z }$ is

$$
G = O ( 3 ) \times \bigl ( S _ { n _ { 1 } } \times \cdot \cdot \cdot \times S _ { n _ { T } } \bigr ) ,\tag{2.9}
$$

where $S _ { n _ { t } }$ permutes the $n _ { t }$ neighbors of type t with $\textstyle \sum _ { t } n _ { t } \ = \ n$ . We write $\Gamma =$ $S _ { n _ { 1 } } \times \cdots \times S _ { n _ { T } }$ for the permutation part. Only type-preserving permutations appear, since atom types are intrinsic attributes, cf. (P1). The action of $( R , \pi ) \in G$ on a stratum element $\mathcal { D } = ( \Delta \boldsymbol { r } _ { 1 } , \ldots , \Delta \boldsymbol { r } _ { n } ) \in \mathcal { E } _ { z _ { 0 } , z }$ is

$$
( R , \pi ) \cdot ( \Delta \boldsymbol { r } _ { 1 } , \ldots , \Delta \boldsymbol { r } _ { n } ) = ( R \Delta \pmb { r } _ { \pi ^ { - 1 } ( 1 ) } , \ldots , R \Delta \pmb { r } _ { \pi ^ { - 1 } ( n ) } ) .\tag{2.10}
$$

Definition 2.3 (Symmetry equivalence). Two environments $\mathcal { D } , \mathcal { D } ^ { \prime }$ , possibly of diferent atoms or diferent configurations, are equivalent, written $\mathcal { D } \sim \mathcal { D } ^ { \prime }$ , if they lie in the same stratum and there exists $( R , \pi ) \in G$ such that $( R , \pi ) \cdot { \mathcal { D } } = { \mathcal { D } } ^ { \prime }$ , that is,

$$
R \Delta r _ { k } = \Delta r _ { \pi ( k ) } ^ { \prime } \qquad f o r \ a l l \ k .\tag{2.11}
$$

Write $d _ { j k } = | \pmb { r } _ { k } - \pmb { r } _ { j } |$ for the distance between atoms j and $k ,$ and $\theta _ { i j k }$ for the bond angle at $j$ between neighbors i and k.

Rotations and reflections change the displacement vectors but leave their mutual inner products unchanged. Collecting those inner products therefore retains exactly the $O ( 3 )$ invariants of the geometry. Two environments have the same inner products if and only if they lie in the same $O ( 3 )$ orbit.

Definition 2.4 (Gram matrix). The Gram matrix of a local environment is

$$
G _ { i k } = \Delta \pmb { r } _ { i } \cdot \Delta \pmb { r } _ { k } = d _ { i } d _ { k } \cos \theta _ { i j k } , \qquad i , k = 1 , \dots , n ,\tag{2.12}
$$

where $d _ { i } = | \Delta \boldsymbol { r } _ { i } |$ and $\theta _ { i j k }$ is the angle between $\Delta { \pmb r } _ { i }$ and $\Delta \boldsymbol { r } _ { k }$

Within a stratum $\mathcal { E } _ { z _ { 0 } , z }$ , the type tuple z is fixed, so the Gram matrix G alone encodes all geometric information. The type-preserving permutation group Γ is determined by the stratum.

For any atom $k ,$ we write $G ^ { ( k ) }$ for the Gram matrix of atom k’s own local environment. Its entries are $G _ { a b } ^ { ( k ) } = \Delta \pmb { r } _ { a } ^ { ( k ) } \cdot \Delta \pmb { r } _ { b } ^ { ( k ) }$ with $\Delta { r } _ { a } ^ { ( k ) } = { r } _ { a } - { r } _ { k }$

Definition 2.5 (Invariant function). A function $\varepsilon : \mathcal { E } ^ { ( r ) } \to \mathbb { R }$ is invariant under the symmetry group $G \ i f \varepsilon ( \mathcal { D } ) = \varepsilon ( \mathcal { D } ^ { \prime } )$ whenever $\mathcal { D } \sim \mathcal { D } ^ { \prime }$

Definition 2.6 (Target function space). Denote by $\mathcal { F } _ { R _ { c } } ^ { k }$ the set of functions $\varepsilon : \mathcal { E } ^ { ( R _ { c } ) }  \mathbb { R }$ satisfying:

1. G-invariance;

2. $C ^ { k }$ -smoothness on each stratum: the restriction $o f \varepsilon$ to each $\mathcal { E } _ { z _ { 0 } , z } ^ { ( R _ { c } ) } \subset \mathbb { R } ^ { 3 n }$ is $C ^ { k }$ function with respect to the Euclidean diferentiable structure;

3. cross-stratum $C ^ { k }$ matching: if $\mathcal { D } = ( \Delta \boldsymbol { r } _ { 1 } , \ldots , \Delta \boldsymbol { r } _ { n } )$ lies in $\mathcal { E } _ { z _ { 0 } , z } ^ { ( R _ { c } ) }$ and $\mathcal { D } ^ { - }$ denotes the environment with the m-th neighbor deleted, then $\varepsilon ( \mathcal { D } ) \to \varepsilon ( \mathcal { D } ^ { - } )$ together with all partial derivatives of order $\leq k$ in the retained coordinates, while all derivatives involving $\Delta \pmb { r } _ { m }$ tend to zero, as $| \Delta \pmb { r } _ { m } |  R _ { c }$

Condition 3 of Definition 2.6 ensures that $\varepsilon$ is well defined and is $C ^ { k }$ across the strata of diferent dimension that $\mathcal { E } ^ { ( R _ { c } ) }$ comprises. It is met, for instance, by $\begin{array} { r } { \varepsilon ( \mathcal { D } ) = \sum _ { i = 1 } ^ { n } s ( | \Delta \pmb { r } _ { i } | ) u ( z _ { 0 } , z _ { i } , | \Delta \pmb { r } _ { i } | ) } \end{array}$ with u a $C ^ { \bar { k } }$ function and $s : [ 0 , R _ { c } ] \to [ 0 , 1 ]$ satisfying $s ^ { ( m ) } ( R _ { c } ) = 0$ for $0 \leq m \leq k$

2.2. Machine learning interatomic potentials. An MLIP approximates the local energy $\varepsilon ( \mathcal { D } _ { j } ^ { ( R _ { c } ) } )$ by

$$
\varepsilon _ { \theta } ( \mathcal { D } _ { j } ^ { ( R _ { c } ) } ) = f _ { \theta } \big ( \Phi _ { \theta } ( \mathcal { D } _ { j } ^ { ( R _ { c } ) } ) \big ) ,\tag{2.13}
$$

where $\Phi _ { \theta } : \mathcal { E } ^ { ( R _ { c } ) }  \mathbb { R } ^ { d }$ is a representation that maps the local environment to a d-dimensional feature vector, and $f _ { \theta } : \mathbb { R } ^ { d }  \mathbb { R }$ is a readout MLP that maps the representation to an atomic energy contribution. Both carry learnable parameters, collectively denoted $\theta ,$ and the total energy is

$$
E _ { \theta } ( \{ r _ { i } , z _ { i } \} _ { i = 1 } ^ { N } ) = \sum _ { i = 1 } ^ { N } \varepsilon _ { \theta } ( \boldsymbol { D } _ { i } ^ { ( R _ { c } ) } ) .\tag{2.14}
$$

The ansatz builds (P2) and (P3) into the architecture. The total energy is a sum of atomic contributions, each depending only on the $R _ { c }$ -environment. The representation must be G-invariant so that (P1) holds. Smoothness of $\Phi _ { \theta }$ and $f _ { \theta }$ , including at the cutof boundary, yields (P4). The quality of the MLIP hinges on the expressiveness of the representation.

Two architectural families instantiate this ansatz. For linear models such as ACE [14, 15] and MTP [32], $\Phi _ { \theta }$ provides a complete polynomial basis $\{ B _ { \alpha } \}$ and $\begin{array} { r } { f _ { \theta } = \sum _ { \alpha } c _ { \alpha } B _ { \alpha } } \end{array}$ is a linear combination. Universal approximation then follows from the density of the basis. For nonlinear models, $f _ { \theta }$ is an MLP and the relevant criterion is not basis density but injectivity of $\Phi _ { \theta }$ on symmetry orbits. §2.3 describes specific GNN-based realizations of $\Phi _ { \theta }$

2.3. Graph neural networks for atomistic systems. Fix a center atom $j$ with neighbor set $\mathcal { N } ( j ) = \{ k : k \neq j , | \pmb { r } _ { k } - \pmb { r } _ { j } | < r _ { c } \}$ , where $r _ { c }$ is the per-layer cutof. Each atom k carries a node feature vector ${ \boldsymbol { n } } _ { k } \in  { \mathbb { R } } ^ { d }$

An atomic system is naturally represented as a graph, see Figure $\mathrm { 1 ( a ) }$ , in which atoms are nodes and pairs within cutof $r _ { c }$ are connected by edges. A GNN updates node features by message passing. At each layer, every node $j$ collects information from its neighbors, aggregates it, and updates its own feature. A single messagepassing layer has the general form

$$
{ \pmb n } _ { j } ^ { ( l + 1 ) } = \psi \Bigl ( { \pmb n } _ { j } ^ { ( l ) } , \bigoplus _ { k \in \mathcal { N } ( j ) } s _ { a } ( d _ { j k } ) \phi \bigl ( { \pmb n } _ { j } ^ { ( l ) } , { \pmb n } _ { k } ^ { ( l ) } , d _ { j k } \bigr ) \Bigr ) ,\tag{2.15}
$$

where $\phi$ is a message function, $\oplus$ is a permutation-invariant aggregation, typically a summation, $\psi$ is an update function, and $s _ { a } \colon [ 0 , r _ { c } ] \to [ 0 , 1 ]$ is a smooth switch function, positive on $[ 0 , r _ { c } )$ with $s _ { a } ( r _ { c } ) = s _ { a } ^ { \prime } ( r _ { c } ) = 0$ , so that a neighbor’s contribution leaves the energy and the forces continuous as it crosses the cutof boundary. Both $\phi$ and $\psi$ are parametrized by MLPs. The representation must be invariant under both $O ( 3 )$ and Γ. In (2.15), O(3)-invariance is achieved by using only the scalar distance $d _ { j k }$ as geometric input, and Γ-invariance by the permutation-invariant aggregation.

In the context of MLIPs, GNNs serve as representations. The output feature ${ n } _ { j } ^ { ( L ) }$ after L layers serves as the representation, written $\Phi ^ { ( L ) }$ . The key advantage of stacking layers is that the receptive field grows. After L layers, the feature of atom $j$ depends on all atoms reachable by L hops, each shorter than $r _ { c }$ . This allows the model to capture interactions at range $R _ { c } > r _ { c }$ without using a large single-layer cutof, which would be computationally expensive.

![](images/0c29c60afd8b65e7cb1d0f4ada3d061300c33b417985323396a1b03bb4a066b7.jpg)  
Fig. 1. Graph structures for atomistic modeling. (a) A graph with 2-body edges, in which each edge $e _ { j i }$ connects a pair of atoms and carries the interatomic distance. (b) A hypergraph with 3-body hyperedges, in which each hyperedge $\mathbf { \ } _ { \mathbf { \lambda } _ { a _ { i j k } } }$ (shaded triangle) connects a triplet (j, i, k) and carries the distances and bond angle. The three hyperedges are shown in orange, blue, and green. Dashed lines indicate the outer edge of each triangle. (c) The line graph transform ${ \mathcal { L } } ,$ , under which each edge in the atom graph G becomes a node (square) in $\mathcal { L } ( G )$ , and two nodes in ${ \mathcal { L } } ( G )$ are connected if the corresponding edges share an atom. The edges in L(G) use the same colors as the hyperedges in (b), showing the correspondence between 3-body hyperedges and line graph edges. ALIGNN, CHGNet, and DPA3 use this two-graph structure to implement 3-body message passing.

However, the framework (2.15) uses only pairwise geometric information: the scalar distance $d _ { j k }$ along each edge. Pozdnyakov et al. [29] showed that such two-body representations are incomplete, and that even adding three-body invariant correlations does not fully resolve the problem, since certain configurations with identical distance and angle distributions are geometrically distinct.

2.4. Hypergraph neural networks. A hypergraph, see Figure 1(b), generalizes a graph by allowing hyperedges that connect more than two nodes simultaneously. An HGNN passes messages along such hyperedges. In the architecture considered here, each hyperedge contains a center atom $j$ and two of its neighbors i and k. It carries the 3-body information, namely the bond-angle cosine cos $\theta _ { i j k }$ in addition to the distances $d _ { j i }$ and $d _ { j k }$ . The message-passing update aggregates over all hyperedges incident to a given node, analogous to (2.15) but with each message depending on a triplet rather than a pair. This gives the HGNN access to angular information that standard two-body GNNs lack. Unlike the GNN framework of §2.3, which maintains only node and edge features, the HGNN computes three-body angle features at each layer. However, only node features are carried across layers, and the angle features are recomputed from scratch at each layer. The HGNN layer comprises an angle-feature calculation and an aggregation operation, defined as follows.

Angle features. For each ordered pair i, $k \in \mathcal { N } ( j )$ with $i \neq k$

$$
\begin{array} { r } { \pmb { a } _ { j ; i , k } = \rho _ { 3 } \Big ( \pmb { n } _ { j } ^ { \mathrm { ( i n ) } } , \pmb { n } _ { i } ^ { \mathrm { ( i n ) } } , \pmb { n } _ { k } ^ { \mathrm { ( i n ) } } , d _ { j i } , d _ { j k } , \cos \theta _ { i j k } \Big ) \in \mathbb { R } ^ { d _ { a } } , } \end{array}\tag{2.16}
$$

where $\rho _ { 3 }$ is an MLP and $d _ { a }$ is the angle-feature dimension.

Aggregation.

$$
{ \pmb n } _ { j } ^ { ( \mathrm { o u t } ) } = \psi _ { 3 } \Big ( \sum _ { \stackrel { i , k \in \mathcal { N } ( j ) } { i \not = k } } s _ { a } ( d _ { j i } ) s _ { a } ( d _ { j k } ) { \pmb a } _ { j ; i , k } \Big ) ,\tag{2.17}
$$

where $\psi _ { 3 }$ is an MLP and $s _ { a }$ is the switch function of (2.15).

Multi-layer stacking. An L-layer HGNN stacks L such layers. The input node features for layer 1 are type embeddings: ${ \pmb n } _ { k } ^ { ( 0 ) } = \mathrm { t y p e \mathrm { . e m b e d } } ( z _ { k } )$ . The output of layer l provides the input for layer $l + 1$ , in that ${ \pmb n } _ { k } ^ { ( l ) } \equiv { \pmb n } _ { k } ^ { ( \mathrm { o u t } , l ) }$ serves as ${ \pmb n } _ { k } ^ { \mathrm { ( i n ) } }$ in (2.16) for the next layer. The L-layer HGNN representation $\Phi ^ { ( L ) }$ maps the L-hop surroundings of atom $j$ to ${ n } _ { j } ^ { ( L ) }$ . Its domain is formalized in Definition 4.5. The total energy is $\begin{array} { r } { E _ { \theta } = \sum _ { j } f _ { \theta } ( { \pmb n } _ { j } ^ { ( L ) } ) } \end{array}$ , where $f _ { \theta }$ is a readout MLP.

2.5. DPA3: GNN on a line graph series. Three architectures used in practice fall within the scope of our results: ALIGNN [11], CHGNet [13], and DPA3 [41]. All three maintain node, edge, and angle features and update them in the order angle $ \mathrm { e d g e } $ node. They difer, however, in the specific form of the message functions and in how angular information is propagated to the nodes. Among these diferences, the last one is particularly consequential. As we show in $\ S 5$ , it determines which of the three architectures inherits universal approximation. Specifically, DPA3 and CHGNet build their messages from MLPs, whereas ALIGNN relies on single gated linear layers within its edge-gated convolutions. We present DPA3 in detail here, as it contains the core ingredients. CHGNet and ALIGNN are described in sections SM2 and SM3 of the supplementary materials, respectively.

DPA3 [41] adopts the same angle → edge → node aggregation pattern as ALIGNN and CHGNet, formalizing it through a line graph series (LiGS). Starting from the atomic graph $G ^ { ( 1 ) } \equiv G$ , one recursively applies the line graph transform to obtain a series $G ^ { ( 1 ) } , G ^ { ( 2 ) } , G ^ { ( 3 ) } , \dots .$ , where $\grave { G } ^ { ( k ) } = \mathcal { L } ( G ^ { ( k - 1 ) } )$ In particular, $G ^ { ( 2 ) } =$ ${ \mathcal { L } } ( G )$ . The vertices and edges of successive graphs correspond to progressively higherorder geometric entities: $\Breve { G } ^ { ( 1 ) }$ has atoms/bonds, $G ^ { ( 2 ) }$ has bonds/angles, $G ^ { ( 3 ) }$ has angles/dihedrals, and so forth. The LiGS thus naturally extends to arbitrary order K. In practice, DPA3 uses $K = 2$ , limiting itself to 3-body angular features, although $K = 3$ with 4-body dihedral features has also been explored in [41].

DPA3 uses directed edges, so each bond $\{ j , i \}$ gives rise to two directed edges $( j , i )$ and $( i , j )$ , which carry independent features. DPA3 maintains three feature types: node features ${ n } _ { j } ^ { ( l ) } \in \mathbb { R } ^ { d }$ , edge features $e _ { j i } ^ { ( l ) } \in \mathbb { R } ^ { d }$ (which serve as vertex features of the graph $G ^ { ( 2 ) } )$ , and angle features ${ \pmb a } _ { j ; i , k } ^ { ( l ) } \in \mathbb { R } ^ { d }$ (which serve as edge features of the graph $G ^ { ( 2 ) } )$ ). The initial representations are constructed from type embeddings for nodes, distance embeddings for edges, and angle embeddings for angles.

Each DPA3 layer applies four updates in sequence, each consuming the most recent value of its inputs.<sup>1</sup>

1: Edge self-message. The edge features first load the node features of both endpoints:

$$
e _ { j i } ^ { ( l + \frac { 1 } { 2 } ) } = e _ { j i } ^ { ( l ) } + \delta _ { s } ^ { ( 1 ) } \phi _ { s } ^ { ( 1 ) } \big ( { \pmb n } _ { j } ^ { ( l ) } , { \pmb n } _ { i } ^ { ( l ) } , { \pmb e } _ { j i } ^ { ( l ) } \big ) .\tag{2.18}
$$

2: Angle self-message. The angle features are updated from the two node-enriched edges that span them:

$$
{ \pmb a } _ { j ; i , k } ^ { ( l + 1 ) } = { \pmb a } _ { j ; i , k } ^ { ( l ) } + \delta _ { s } ^ { ( 2 ) } \phi _ { s } ^ { ( 2 ) } \big ( e _ { j i } ^ { ( l + \frac { 1 } { 2 } ) } , e _ { j k } ^ { ( l + \frac { 1 } { 2 } ) } , { \pmb a } _ { j ; i , k } ^ { ( l ) } \big ) .\tag{2.19}
$$

3: Angle → edge. Each edge aggregates over the angles centered at its tail atom:

$$
e _ { j i } ^ { ( l + 1 ) } = e _ { j i } ^ { ( l + \frac { 1 } { 2 } ) } + \delta _ { u } ^ { ( 2 ) } \phi _ { u } ^ { ( 2 ) } \Big ( \sum _ { k \in \mathcal { N } ^ { ( 3 ) } ( j ) \backslash \{ i \} } w _ { j i k } \phi _ { c } ^ { ( 2 ) } \big ( e _ { j i } ^ { ( l + \frac { 1 } { 2 } ) } , e _ { j k } ^ { ( l + \frac { 1 } { 2 } ) } , a _ { j ; i , k } ^ { ( l + 1 ) } \big ) \Big ) ,\tag{2.20}
$$

where $\mathcal { N } ^ { ( 3 ) } ( j )$ denotes the angle-cutof neighbors of $j ~ ( d _ { j k } < r _ { c } ^ { ( 3 ) } ) , ~ w _ { j i k }$ is a switch prefactor, and $\delta _ { u } ^ { ( 2 ) }$ is a trainable step size.

$\begin{array} { r } { \begin{array} { r l } { \psi : \ E d g e \to } \end{array} } \end{array}$ node. Each node aggregates over its edges:

$$
{ \pmb n } _ { j } ^ { ( l + 1 ) } = { \pmb n } _ { j } ^ { ( l ) } + \delta _ { u } ^ { ( 1 ) } \phi _ { u } ^ { ( 1 ) } \Bigl ( \sum _ { i \in \mathcal { N } ^ { ( 2 ) } ( j ) } w _ { j i } \phi _ { c } ^ { ( 1 ) } \bigl ( { \pmb n } _ { j } ^ { ( l ) } , { \pmb n } _ { i } ^ { ( l ) } , { \pmb e } _ { j i } ^ { ( l + 1 ) } \bigr ) \Bigr ) ,\tag{2.21}
$$

where $\mathcal { N } ^ { ( 2 ) } ( j )$ denotes the edge-cutof neighbors of $j ~ ( d _ { j i } < r _ { c } ^ { ( 2 ) } )$ . DPA3 uses two cutof radii: $r _ { c } ^ { ( 3 ) } \leq r _ { c } ^ { ( 2 ) }$ . All ϕ functions $( \bar { \phi _ { s } ^ { ( 1 ) } } , \phi _ { s } ^ { ( 2 ) } , \bar { \phi _ { c } ^ { ( 2 ) } } , \bar { \phi _ { u } ^ { ( 2 ) } } , \phi _ { c } ^ { ( 1 ) } , \phi _ { u } ^ { ( 1 ) } )$ are MLPs. The prefactors are products of switch-function values, $w _ { j i k } = s _ { a } ^ { ( 3 ) } ( d _ { j i } ) s _ { a } ^ { ( 3 ) } ( d _ { j k } )$ and $w _ { j i } = s _ { a } ^ { ( 2 ) } ( d _ { j i } )$ , where $s _ { a } ^ { ( 2 ) }$ and $s _ { a } ^ { ( 3 ) }$ are switch functions as in (2.15) for the cutofs $r _ { c } ^ { ( 2 ) }$ and $r _ { c } ^ { ( 3 ) } .$ . The $\delta$ step sizes are trainable scalars. The total energy is $E _ { \theta } ~ =$ $\begin{array} { r } { \sum _ { j } \mathrm { M L P } ( n _ { j } ^ { ( L ) } ) } \end{array}$

As noted above, the LiGS extends to arbitrary order K. Empirically, the DPA3 paper found that $K = 2$ yields optimal performance, while $K = 3$ ofers no further improvement and may even degrade accuracy. Our completeness theory provides an explanation for this observation. On generic configurations, 3-body information already sufices for universal approximation, so higher body-order features are theoretically unnecessary.

## 3. Representation Completeness.

3.1. Completeness and universal approximation. As described in §2.2, most MLIP architectures decompose the local energy as $\varepsilon ( \mathcal { D } ) \ = \ f ( \Phi ( \mathcal { D } ) )$ , where $\Phi : \mathcal { E } ^ { ( R _ { c } ) }  \mathbb { R } ^ { d }$ is a G-invariant representation and $f : \mathbb { R } ^ { d } $ R is a readout network.

Definition 3.1 (Completeness). A complete representation at cutof r is a continuous G-invariant map $\Phi : \mathcal { E } ^ { ( r ) }  \mathbb { R } ^ { d }$ that separates orbits: for any two local environments $\mathcal { D } _ { 1 } , \mathcal { D } _ { 2 } \in \mathcal { E } ^ { ( r ) }$

$$
\begin{array} { r } { \Phi ( \mathcal { D } _ { 1 } ) = \Phi ( \mathcal { D } _ { 2 } ) \quad i m p l i e s \quad \mathcal { D } _ { 1 } \sim \mathcal { D } _ { 2 } . } \end{array}\tag{3.1}
$$

Since ∼ includes both rotations/reflections and type-preserving permutations, completeness requires separating orbits under the full symmetry group G.

Definition 3.2 (Universal approximation). A parametrized family $\left\{ \varepsilon _ { \theta } \right\}$ of continuous functions on $\dot { \mathcal { E } } ^ { ( R _ { c } ) }$ is a universal approximator for $\mathcal { F } _ { R } ^ { 0 }$ on a subset $A \subseteq \mathcal { E } ^ { ( R _ { c } ) }$ if for every $\varepsilon \in \mathcal { F } _ { R _ { c } } ^ { 0 }$ , every compact $K \subseteq A$ , and every $\delta > 0$ there is a parameter θ with $\| \varepsilon _ { \theta } - \varepsilon \| _ { C ^ { 0 } ( K ) } < \delta$ . We omit the subset when $A = \mathcal { E } ^ { ( R _ { c } ) }$

Completeness constrains the representation alone, whereas universal approximation constrains the architecture $f _ { \boldsymbol { \theta } } \circ \Phi$ as a whole. That an incomplete representation caps the achievable accuracy is immediate. The substance of the next result is the converse, that completeness sufices.

Theorem 3.3 (Completeness ⇔ UAT). Let $\Phi : \mathcal { E } ^ { ( R _ { c } ) }  \mathbb { R } ^ { d }$ be a continuous G-invariant representation, and let $\{ f _ { \theta } \}$ be a family of MLPs, which by $\it { \Omega } / \mathrm { 1 2 } , \ \it { 1 9 } \mathrm { ] }$ approximate every continuous function on $\mathbb { R } ^ { d }$ uniformly on compact sets. Then the architecture $\varepsilon _ { \theta } = f _ { \theta } \circ \Phi$ is a universal approximator for $\mathcal { F } _ { R _ { c } } ^ { 0 }$ in the sense of Definition 3.2 if and only $i f \Phi$ is complete.

Continuity of Φ and compactness of test sets are understood with respect to the Euclidean metric on each stratum.

Proof. Completeness $\Rightarrow \ U A T$ . Let $\varepsilon \in \mathcal { F } _ { R _ { c } } ^ { 0 }$ and $K \subset \mathcal { E } ^ { ( R _ { c } ) }$ be compact. Since ε is G-invariant and Φ is complete, $\Phi ( \mathcal { D } _ { 1 } ) ~ = ~ \Phi ( \mathcal { D } _ { 2 } )$ forces $\mathcal { D } _ { 1 } ~ \sim ~ \mathcal { D } _ { 2 }$ and hence $\varepsilon ( \mathcal { D } _ { 1 } ) = \varepsilon ( \mathcal { D } _ { 2 } )$ , so $f ( \Phi ( \mathcal { D } ) ) = \varepsilon ( \mathcal { D } )$ defines a function f on the compact set $\Phi ( K )$ The restriction $\Phi | _ { K }$ is a continuous surjection from a compact space onto a Hausdorf one, hence a quotient map. The composition $f \circ \Phi | _ { K } = \varepsilon | _ { K }$ is continuous, so $f$ is continuous. By the Tietze extension theorem, f extends to a continuous function on $\mathbb { R } ^ { d }$ , and by MLP universality, for any $\delta > 0$ , there exists $f _ { \theta }$ that approximates that extension on $\Phi ( K )$ with $\| f _ { \theta } \circ \Phi - \varepsilon \| _ { C ^ { 0 } ( K ) } < \delta$

Converse (incompleteness ⇒ not UAT). If Φ is incomplete, there exist $\mathcal { D } _ { 1 } ~ \nearrow ~ \mathcal { D } _ { 2 }$ with $\Phi ( \mathcal { D } _ { 1 } ) = \Phi ( \mathcal { D } _ { 2 } )$ . For any readout network $f , f ( \Phi ( { \mathcal { D } } _ { 1 } ) ) = f ( \Phi ( { \mathcal { D } } _ { 2 } ) )$ , so the architecture cannot distinguish $\mathcal { D } _ { 1 }$ and $\mathcal { D } _ { 2 }$ . It remains to exhibit a target $\varepsilon \in \mathcal { F } _ { R _ { c } } ^ { 0 }$ with $\varepsilon ( \mathcal { D } _ { 1 } ) \neq \varepsilon ( \mathcal { D } _ { 2 } )$

Let $\Phi ^ { * }$ be a complete representation whose components lie in $\mathcal { F } _ { R _ { c } } ^ { 0 }$ . Such a representation is constructed in Proposition 3.7 via the switched Gram matrix, without relying on the present theorem, so no circularity arises. Set ε(D) = min $\{ 1 , \| \Phi ^ { * } ( \mathcal { D } ) -$ $\Phi ^ { * } ( D _ { 2 } ) \Vert \}$ . Then $\varepsilon$ is G-invariant and continuous because $\Phi ^ { * } \ \mathrm { i s } ,$ and it satisfies the cross-stratum matching of Definition 2.6 because the components of $\Phi ^ { * }$ do, so $\varepsilon \in \mathcal { F } _ { R _ { c } } ^ { 0 }$ Moreover $\varepsilon ( \mathcal { D } _ { 2 } ) ~ = ~ 0$ , while $\varepsilon ( \mathcal { D } _ { 1 } ) > 0$ , because completeness and $\mathcal { D } _ { 1 } ~ \nearrow ~ \mathcal { D } _ { 2 }$ give $\Phi ^ { * } ( { \mathcal { D } } _ { 1 } ) \neq \Phi ^ { * } ( { \mathcal { D } } _ { 2 } )$

Taking $K = \{ \mathcal { D } _ { 1 } , \mathcal { D } _ { 2 } \}$ (compact) and $\delta = \varepsilon ( D _ { 1 } ) / 2$ , every model $f _ { \theta } \circ \Phi$ satisfies $\| f _ { \theta } \circ \Phi - \varepsilon \| _ { C ^ { 0 } ( K ) } \geq \varepsilon ( \mathcal { D } _ { 1 } ) / 2 > 0$ , so the architecture is not a universal approximator.

3.2. Generic configurations and Gram matrix reconstruction. This subsection is purely information-theoretic. We show that complete representations exist, without reference to any neural network architecture. Consider a local environment D with center type z and n neighbors at displacements $\Delta \boldsymbol { r } _ { 1 } , . . . , \Delta \boldsymbol { r } _ { n } \in B _ { r _ { c } } \setminus \{ 0 \}$ with types $z _ { 1 } , \ldots , z _ { n }$

Definition 3.4 (Generic configuration). A local environment $\mathcal { D } \in \mathcal { E } ^ { ( r ) }$ is generic if no two same-type neighbors are equidistant from the center, that is, $i f z _ { i } = z _ { k }$ and $i \neq k$ imply $\left| \Delta \boldsymbol { r } _ { i } \right| \neq \left| \Delta \boldsymbol { r } _ { k } \right|$ . We write ${ \mathcal { E } } ^ { * } \subset { \mathcal { E } } ^ { ( r ) }$ for the set of generic environments.

On each stratum, the complement of $\mathcal { E } ^ { * }$ is the union, over the finitely many sametype pairs $i \neq k$ , of the zero sets of the nonzero polynomials $| \Delta \boldsymbol { r } _ { i } | ^ { 2 } - | \Delta \boldsymbol { r } _ { k } | ^ { 2 }$ . The zero set of a polynomial that does not vanish identically is closed and has zero Lebesgue measure in $( \mathbb { R } ^ { 3 } ) ^ { n }$ . The complement of $\mathcal { E } ^ { * }$ is a finite union of such sets, so it is closed and has zero Lebesgue measure. Hence $\mathcal { E } ^ { * }$ is open and dense in ${ \mathcal { E } } ^ { ( r ) }$

A complete representation has two requirements, invariance under G and separation of distinct G-orbits. The Gram matrix meets the first for the $O ( 3 )$ factor, and the second amounts to the converse, that it determines the displacement vectors up to a single orthogonal transformation. That is the content of Lemma 3.5. The permutation factor $\Gamma$ is handled separately in Lemma 3.6.

Lemma 3.5 (Gram matrix reconstruction). If A, $\boldsymbol { B } \in \mathbb { R } ^ { n \times 3 }$ satisfy $A A ^ { T } =$

$B B ^ { T }$ , then there exists $R \in O ( 3 )$ such that $B = A R$

Proof. It follows from the thin SVD and $A A ^ { T } = B B ^ { T }$ that

$$
A = P U B = P V
$$

with $P = ( A A ^ { T } ) ^ { 1 / 2 }$ and $U ^ { T } U = V ^ { T } V = I _ { 3 }$ . When rank $A \ : < \ : 3$ , the factors are not unique. The column space col U of U is a 3-dimensional subspace containing col A, determined by an arbitrary orthonormal completion in the thin SVD, and likewise for V. Since col $A = \operatorname { c o l } P = \operatorname { c o l } B$ , we choose the two completions so that col $U = \mathrm { c o l } V$ . We claim that

$$
U U ^ { T } = V V ^ { T } .
$$

Using the fact that for any $x \in \mathbb { R } ^ { 3 }$ and $y \in \mathbb { R } ^ { n }$

$$
\begin{array} { r } { ( ( I - U U ^ { T } ) y , U x ) = ( U ^ { T } y , x ) - ( U ^ { T } U U ^ { T } y , x ) = ( U ^ { T } y , x ) - ( U ^ { T } y , x ) = 0 . } \end{array}
$$

We conclude that $U U ^ { T }$ is the orthogonal projection onto col U. Similarly, $V V ^ { T }$ is the orthogonal projection onto col V . The two column spaces coincide by our choice, which gives $U \hat { U ^ { T } } \overset { \vartriangle } { = } V V ^ { T }$

Let $R = U ^ { T } V$ . It is straightforward to verify that $R \in O ( 3 )$ and $B = A R$

## 3.3. Existence of a complete representation.

Lemma 3.6 (Gram matrix orbit characterization). On a stratum $\mathcal { E } _ { z _ { 0 } , z }$ , the Gram matrix map $\Psi ( { \mathcal { D } } ) = G$ is O(3)-invariant and Γ-equivariant, where the permutation group $\Gamma = S _ { n _ { 1 } } \times \cdot \cdot \cdot \times S _ { n _ { T } }$ acts on the indices via $( \pi \cdot G ) _ { i k } = G _ { \pi ^ { - 1 } ( i ) , \pi ^ { - 1 } ( k ) }$

The map Ψ determines the local environment up to O(3)-equivalence. More precisely, two environments $\mathcal { D } _ { 1 } , \mathcal { D } _ { 2 }$ in the same stratum satisfy $\mathcal { D } _ { 1 } \sim \mathcal { D } _ { 2 }$ if and only if

$$
\exists \pi \in \Gamma : G _ { \pi ( i ) , \pi ( k ) } ^ { ( 2 ) } = G _ { i k } ^ { ( 1 ) } \qquad \forall i , k .\tag{3.2}
$$

Proof. If $\mathcal { D } _ { 1 } \sim \mathcal { D } _ { 2 }$ via $\Delta { r } _ { \pi ( i ) } ^ { ( 2 ) } = R \Delta { r } _ { i } ^ { ( 1 ) }$ , then $G _ { \pi ( i ) , \pi ( k ) } ^ { ( 2 ) } = R \Delta \pmb { r } _ { i } ^ { ( 1 ) } \cdot R \Delta \pmb { r } _ { k } ^ { ( 1 ) } = G _ { i k } ^ { ( 1 ) }$ Suppose a type-preserving π gives $G _ { \pi ( i ) , \pi ( k ) } ^ { ( 2 ) } = G _ { i k } ^ { ( 1 ) }$ for all i, k. Let $X ^ { ( 1 ) }$ have rows $\Delta { r } _ { i } ^ { ( 1 ) }$ and Y have rows $\Delta r _ { \pi ( i ) } ^ { ( 2 ) }$ . Then $\pmb { Y } \pmb { Y } ^ { T } = \pmb { X } ^ { ( 1 ) } ( \pmb { X } ^ { ( 1 ) } ) ^ { T }$ . By Lemma 3.5, there exists $R \in O ( 3 )$ such that $Y = X ^ { ( 1 ) } R$ , hence $\mathcal { D } _ { 1 } \sim \mathcal { D } _ { 2 }$ □

Lemma 3.6 reduces completeness to separating the Γ-orbits of Gram matrices. Since Γ is finite, invariant theory supplies finitely many polynomials that do so on each stratum.

Physical applications require continuity with respect to the atomic coordinates, in particular when an atom crosses the boundary at cutof radius R and the environment passes from one stratum to another. A representation built from the Gram matrix alone is discontinuous at such a crossing, because the entries involving a departing neighbor do not decay to zero as $| \Delta \pmb { r } _ { k } |  R$

To restore continuity, we damp each displacement with a switch function that vanishes smoothly at the cutof radius. Let $s \colon [ 0 , R ] \to [ 0 , 1 ]$ be a smooth function, positive on [0, R), satisfying $s ( R ) = s ^ { \prime } ( R ) = 0$ , and such that $r \mapsto s ( r ) r$ is injective on (0, R). We then define the switched Gram matrix by replacing each displacement vector by its damping counterpart $\Delta \tilde { r } _ { k } = s ( | \Delta r _ { k } | ) \Delta r _ { k } \mathrm { . }$

$$
\tilde { G } _ { i k } = \Delta \tilde { r } _ { i } \cdot \Delta \tilde { r } _ { k } = s ( d _ { i } ) s ( d _ { k } ) G _ { i k } .
$$

The switched Gram matrix glues the strata together in the proof of the following proposition.

Proposition 3.7 (Existence of a complete representation). For any cutof $R >$ 0 there exists a complete representation $\Phi ^ { * } : \mathcal { E } ^ { ( R ) } \overset { } { \to } \mathbb { R } ^ { d }$ whose components satisfy the cross-stratum matching of condition 3 in Definition 2.6, with $R _ { c }$ replaced by R.

Proof. Step 1: a single stratum. Work on a fixed stratum $\mathcal { E } _ { z _ { 0 } , z } ^ { ( R ) }$ with n neighbors. The Gram matrix $\Psi ( { \mathcal { D } } ) = G$ takes values in the space of $n \times n$ real symmetric matrices $\gamma \cong \mathbb { R } ^ { n ( n + 1 ) / 2 }$ . By Lemma 3.6, $\mathcal { D } _ { 1 } \sim \mathcal { D } _ { 2 }$ if and only if $\Psi ( D _ { 1 } )$ and $\Psi ( D _ { 2 } )$ lie in the same Γ-orbit under the action $G \mapsto P _ { \pi } G P _ { \pi } ^ { T }$

Since Γ is a finite group acting on V, by Noether’s theorem [26] the ring of Γ-invariant polynomials $\bar { \mathbb { R } } [ \mathcal { V } ] ^ { \Gamma }$ is finitely generated, so there exist Γ-invariant polynomials $p _ { 1 } , . . . , p _ { d } : \mathcal { V }  \mathbb { R }$ such that every Γ-invariant polynomial is a polynomial in $p _ { 1 } , \ldots , p _ { d } .$ . Moreover, these generators separate Γ-orbits, in the sense that if $p _ { i } ( G ^ { ( 1 ) } ) = p _ { i } ( G ^ { ( 2 ) } )$ for all $i = 1 , \ldots , d .$ , then $G ^ { ( \bar { 1 } ) }$ and $G ^ { ( 2 ) }$ lie in the same Γ-orbit. Indeed, for any $G ^ { ( 2 ) } \notin \Gamma \cdot G ^ { ( 1 ) }$ , the polynomial $\begin{array} { r } { P ( X ) = \prod _ { M \in \Gamma \cdot G ^ { ( 1 ) } } \| X - M \| ^ { 2 } } \end{array}$ is Γ-invariant, vanishes on $\Gamma \cdot G ^ { ( 1 ) }$ , and is strictly positive on $\Gamma \cdot G ^ { ( 2 ) }$ . Since $P$ is Γ- invariant, it is expressible in the generators, so at least one generator difers between the two orbits.

Define

$$
\Phi ( \mathcal D ) = \big ( p _ { 1 } ( \Psi ( \mathcal D ) ) , \mathrm { ~ . ~ . ~ . ~ } , p _ { d } ( \Psi ( \mathcal D ) ) \big ) \in \mathbb { R } ^ { d } .
$$

Φ is continuous because its components are polynomials in the entries of the continuous map $\Psi , O ( 3 )$ -invariant because Ψ is, and Γ-invariant because each $p _ { i }$ is, hence G-invariant. If $\Phi ( \mathcal { D } _ { 1 } ) = \Phi ( \mathcal { D } _ { 2 } )$ , then all generators agree on $\Psi ( D _ { 1 } )$ and $\Psi ( D _ { 2 } )$ , so $\Psi ( \mathcal { D } _ { 1 } ) \sim _ { \Gamma } \Psi ( \mathcal { D } _ { 2 } )$ , hence $\mathcal { D } _ { 1 } \sim \mathcal { D } _ { 2 }$ by Lemma 3.6.

Step 2: assembly across strata. The per-stratum maps are now glued through the switched Gram matrix. Pad $\tilde { G }$ by zeros to a fixed size, reserving for each type t as many index slots as the maximal number $M _ { t }$ of type-t neighbors, and write $\hat { G } ( \mathcal { D } )$ for the result. When a neighbor leaves through the boundary, its switched entries vanish, so $\hat { G } ( { \cal D } )  \hat { G } ( { \cal D } ^ { - } )$ along the transitions of condition 3. The pair $( z _ { 0 } , \hat { G } )$ still determines the environment. The positive diagonal entries reveal the number of neighbors of each type, hence the stratum, and the injectivity of $r \mapsto s ( r ) r$ recovers the distances, then the values $s ( d _ { i } ) \ > \ 0$ , and finally $G _ { i k } \ = \ \hat { G } _ { i k } / ( s ( d _ { i } ) s ( d _ { k } ) )$ , so Lemmas 3.5 and 3.6 apply. Repeating Step 1 for the finite group $\begin{array} { r } { \bar { \Gamma } = \prod _ { t } S _ { M _ { t } } } \end{array}$ , which permutes the slots within each type block of ${ \hat { G } } .$ , and appending the center type $z _ { 0 }$ yields $\Phi ^ { * }$ . Its components are polynomials in the entries of ${ \hat { G } } _ { : }$ so they inherit the continuity and the boundary matching, and the center type is constant along every condition-3 transition. □

$\Phi ^ { * }$ is the representation invoked in the converse direction of Theorem 3.3.

4. HGNN Approximation of Complete Representations. We have thus established the existence of complete representations, yet the construction of Proposition 3.7 is information-theoretic. It assembles $\Phi ^ { * }$ from invariant polynomials rather than from a trainable network. We now turn to the question of whether such a representation can be realized in practice. Specifically, we show that the HGNN architecture is expressive enough to approximate any such complete representation to arbitrary accuracy on compact sets.

4.1. DeepSets universality. We first record a form of the DeepSets universality theorem [40, 36] that covers vector-valued elements. It is the tool underlying every approximation result below. The HGNN aggregation (2.17) is a sum of per-triplet messages, and DeepSets universality is what lets such a sum represent an arbitrary symmetric function of the triplet multiset.

For a compact $X \subset \mathbb { R } ^ { d }$ , let $\mathcal { M } _ { M } ( X ) \equiv ( \ O _ { < M } ^ { X } ) ^ { 2 }$ denote the set of all multisets of elements of X having size at most M. Each fixed-size stratum $\mathcal { M } _ { m } ( \boldsymbol { X } ) = \boldsymbol { X } ^ { m } / S _ { m }$ carries the quotient metric $d ( S , S ^ { \prime } ) = \operatorname* { m i n } _ { \pi \in S _ { m } } \operatorname* { m a x } _ { i } \left\| x _ { i } - y _ { \pi ( i ) } \right\|$ , where $S = \{ x _ { 1 } , \dots , x _ { m } \}$ and $S ^ { \prime } = \{ y _ { 1 } , \dots , y _ { m } \}$ are any ordered representatives, and $\begin{array} { r } { \mathcal { M } _ { M } ( X ) = \bigcup _ { m = 0 } ^ { M } \mathcal { M } _ { m } ( X ) } \end{array}$ has the disjoint-union topology. Since $X$ is compact, $X ^ { m }$ is compact as a finite product of compact sets, and $\mathcal { M } _ { m } ( X )$ is compact as the continuous image of $X ^ { m }$ under the quotient map.

Lemma 4.1 (DeepSets universality). Let $X \subset \mathbb { R } ^ { d }$ be compact. There is an integer m depending only on M and d such that, for any continuous function $f \colon { \mathcal { M } } _ { M } ( X ) \to$ R, there exist continuous functions $\phi \colon X  \mathbb { R } ^ { m }$ and $\rho \colon \mathbb { R } ^ { m }  \mathbb { R }$ such that $f ( S ) =$ $\textstyle \rho { \bigl ( } \sum _ { x \in S } \phi ( x ) { \bigr ) }$ for every $S \in \mathcal { M } _ { M } ( X )$ , where $\textstyle \sum _ { x \in S }$ denotes the sum over elements of S with multiplicity. One may take $\phi ( x ) = \left( x ^ { \alpha } \right) _ { 0 \leq | \alpha | \leq M }$ , the monomials of degree at most M, so that $m = { \binom { M + d } { d } }$

Proof. The coordinate $\alpha = 0$ recovers the size of S, and for each fixed size the sums $\textstyle \sum _ { x \in S } x ^ { \alpha }$ with $1 \leq | \alpha | \leq M$ generate the polynomial invariants of the symmetric group acting on tuples of vectors [39], which separate its orbits as in Step 1 of the proof of Proposition 3.7. Hence $\begin{array} { r } { S \stackrel { \cdot } { \mapsto } \sum _ { x \in S } \phi ( x ) } \end{array}$ is injective, and, being a continuous bijection from a compact space onto its image, a homeomorphism. Composing f with the inverse and extending by the Tietze theorem yields $\rho .$ □

Lemma 4.1 produces continuous $\phi$ and $\rho ,$ whereas the HGNN’s message functions are MLPs. The following lemma may be viewed as an approximate version of DeepSets universality.

Lemma 4.2 (DeepSets MLP approximation). Let $X \subset \mathbb { R } ^ { d }$ be compact, $K \subset$ $\mathcal { M } _ { M } ( \boldsymbol { X } )$ compact, and $f \in C ^ { 0 } ( K )$ a continuous symmetric function. For every $\delta > 0$ there exist MLPs $\phi _ { \theta } \colon X \to \mathbb { R } ^ { m }$ and $\rho _ { \theta } \colon \mathbb { R } ^ { m }  \mathbb { R }$ , with m as in Lemma 4.1, such that

$$
\begin{array} { r } { \left. f - \rho _ { \theta } \circ \left( \sum _ { i } \phi _ { \theta } \right) \right. _ { C ^ { 0 } ( K ) } < \delta . } \end{array}
$$

Proof. For a multiset $S \in \mathcal { M } _ { M } ( X )$ , we write elts $( S ) \subset X$ for the set of elements appearing in $S$ (forgetting multiplicities) and $\textstyle \sum _ { x \in S }$ for the sum over elements with multiplicity.

By Lemma 4.1, $\begin{array} { r } { f ( S ) = \rho \bigl ( \sum _ { x \in S } \phi ( x ) \bigr ) } \end{array}$ for continuous $\phi \colon X  \mathbb { R } ^ { m }$ and $\rho \colon \mathbb { R } ^ { m } $ R. Define $\begin{array} { r } { X _ { K } = \bigcup _ { S \in { K } } \operatorname { e l t s } ( \bar { S } ) \subset \mathbf { \bar {  { X } } } } \end{array}$ , which is compact because K is compact, and $\begin{array} { r } { \Sigma = \left\{ \sum _ { x \in S } \phi ( x ) : S \in K \right\} \subset \mathbb { R } ^ { m } } \end{array}$ , which is also compact because it is a continuous image of $K$

By MLP universality on $X _ { K }$ , there exists $\phi _ { \theta }$ with $\| \phi _ { \theta } - \phi \| _ { C ^ { 0 } ( X _ { K } ) } < \epsilon$ . Then for every $\begin{array} { r } { S \in K , \ \left\| \sum _ { x \in S } \phi _ { \theta } ( x ) - \sum _ { x \in S } \phi ( x ) \right\| \le M \epsilon } \end{array}$ . Let $\Sigma _ { \epsilon }$ be the closed $M \epsilon -$ neighborhood of $\Sigma .$ which is compact. By MLP universality on $\Sigma _ { \epsilon }$ , there exists $\rho _ { \theta }$ with $\| \rho _ { \theta } - \rho \| _ { C ^ { 0 } ( \Sigma _ { \epsilon } ) } < \epsilon$ . Since $\rho$ is continuous on the compact set $\Sigma _ { \epsilon }$ , it is uniformly continuous: for any $\delta ^ { \prime } > 0$ there exists $\eta > 0$ such that $\| s - s ^ { \prime } \| < \eta \Rightarrow | \rho ( s ) - \rho ( s ^ { \prime } ) | < \delta ^ { \prime }$ for all $s , s ^ { \prime } \in \Sigma _ { \epsilon }$ . Set $\delta ^ { \prime } = \delta / 2$ , let η be the corresponding modulus, and choose ϵ small enough that $M \epsilon < \eta$ and $\epsilon < \delta / 2$ . For every $S \in K$ , writing $\begin{array} { r } { s = \sum _ { x \in S } \phi ( x ) \in \Sigma } \end{array}$ and

$$
\begin{array} { r } { \tilde { s } = \sum _ { x \in S } \phi _ { \theta } ( x ) \in \Sigma _ { \epsilon } , } \end{array}
$$

$$
\begin{array} { r } { | \rho _ { \theta } ( \tilde { s } ) - \rho ( s ) | \le | \rho _ { \theta } ( \tilde { s } ) - \rho ( \tilde { s } ) | + | \rho ( \tilde { s } ) - \rho ( s ) | < \frac { \delta } { 2 } + \frac { \delta } { 2 } = \delta , } \end{array}
$$

where the first term is bounded by the choice of $\rho _ { \theta }$ and the second by uniform continuity, since $\| \tilde { s } - s \| \leq M \epsilon < \eta$ . Since the bound holds for every $S \ \in \ K .$ $\| f - \rho _ { \boldsymbol { \theta } } \circ \big ( \sum \phi _ { \boldsymbol { \theta } } \big ) \big \| _ { C ^ { 0 } ( K ) } < \delta .$ □

4.2. Single-layer approximation. Lemma 4.2 shows that a pair of MLPs can approximate any symmetric function of a multiset. To apply this result to a single HGNN layer, we require that the multiset over which the layer aggregates encodes the same information as the Gram matrix. The following identities demonstrate that this is indeed the case. The map from the Gram matrix $G$ to distances and angles,

$$
d _ { i } = \sqrt { G _ { i i } } , \qquad \cos { \theta _ { i j k } } = \frac { G _ { i k } } { d _ { i } d _ { k } } \quad ( i \neq k ) ,\tag{4.1}
$$

is a bijection with inverse

$$
G _ { i k } = d _ { i } d _ { k } \cos \theta _ { i j k } , \qquad G _ { i i } = d _ { i } ^ { 2 } .\tag{4.2}
$$

The two descriptions of the local geometry are thus interchangeable. The computation performed by a single HGNN layer can therefore be equivalently formulated in terms of the Gram matrix. The following lemma establishes that, under the condition that distinct neighbors carry distinct labels, this layer can approximate any continuous Γ-invariant function of that Gram matrix and the incoming node features.

Lemma 4.3 (Architectural expressiveness). On a stratum $\mathcal { E } _ { z _ { 0 } , z } ^ { ( r _ { c } ) }$ with $n \geq 2$ neighbors and permutation group Γ, consider a single HGNN layer $( ( 2 . 1 6 ) \ – ( 2 . 1 7 ) )$ . Suppose the pairs $( \pmb { n } _ { k } ^ { \mathrm { ( i n ) } } , d _ { j k } ) , k \in \mathcal { N } ( j )$ , are pairwise distinct. Suppose further that the node features of neighbors of diferent types take values in disjoint sets. Then, with suficient MLP capacity and suficiently large angle-feature dimension $d _ { a }$ , the HGNN layer can approximate any continuous Γ-invariant function of the labeled data $\mathcal { L } = ( G ^ { ( j ) } , ( \pmb { n } _ { 1 } ^ { \mathrm { ( i n ) } } , \dots , \pmb { n } _ { n } ^ { \mathrm { ( i n ) } } ) )$ , where $G ^ { ( j ) }$ is the Gram matrix (2.12) of j’s environment and Γ acts by $\pi \cdot ( G , \pmb { n } ) = ( P _ { \pi } G P _ { \pi } ^ { T } , \pmb { n } _ { \pi ^ { - 1 } } )$

The argument proceeds in three steps. First, a single HGNN layer has DeepSets form over the multiset of neighbor triplets, so Lemma 4.2 applies. Second, every continuous Γ-invariant function of the labeled data factors through that multiset. Third, the two combine by composition.

Before delving into the proof, we first fix notation. Let $\mathcal { V } = \mathbb { R } _ { \mathrm { s y m } } ^ { n ( n + 1 ) / 2 }$ denote the space of Gram matrices and $\mathcal { W } = \mathcal { V } \times ( \mathbb { R } ^ { d } ) ^ { n }$ the space of labeled data ${ \mathcal { L } } =$ $( G , ( n _ { 1 } , \ldots , n _ { n } ) )$ . The group Γ acts on W by $\pi \cdot ( G , \pmb { n } ) = ( P _ { \pi } G P _ { \pi } ^ { T } , \pmb { n } _ { \pi ^ { - 1 } } )$ . Let $X =$ $\mathbb { R } ^ { d } \times \mathbb { R } ^ { d } \times ( 0 , r _ { c } ) \times ( 0 , r _ { c } ) \times [ - 1 , 1 ]$ be the space of a single tuple $( { \pmb n } _ { i } , { \pmb n } _ { k } , d _ { j i } , d _ { j k }$ , cos $\theta _ { i j k } )$ Define the map µ $\iota \colon { \mathcal { W } }  { \mathcal { M } } _ { M ( M - 1 ) } ( X )$ by

$$
\mu ( G , \pmb { n } ) = \big \{ ( \pmb { n } _ { i } ^ { \mathrm { ( i n ) } } , \pmb { n } _ { k } ^ { \mathrm { ( i n ) } } , d _ { j i } , d _ { j k } , \cos \theta _ { i j k } ) \big \} _ { i \neq k } ,\tag{4.3}
$$

where distances and angles are obtained from G via (4.1) and the index pair $( i , k )$ runs over all ordered pairs of distinct neighbors. Write $C _ { \mathrm { s y m } } ^ { 0 } ( \mathcal { M } _ { M ( M - 1 ) } ( X ) )$ for the continuous functions on the multiset space, $C _ { \Gamma } ( \mathcal { W } )$ for the continuous Γ-invariant functions on W, and $\mu ^ { * } \colon g \mapsto g \circ \mu$ for the pullback along $\mu .$

## Proof. Step 1: DeepSets universality. The HGNN computes

$$
{ \pmb { n } } _ { j } ^ { \mathrm { ( o u t ) } } = \psi _ { 3 } \Big ( \sum _ { \stackrel { i , k \in \mathcal { N } ( j ) } { i \not = k } } \tilde { \rho } _ { 3 } \big ( { \pmb { n } } _ { j } ^ { \mathrm { ( i n ) } } , { \pmb { n } } _ { i } ^ { \mathrm { ( i n ) } } , { \pmb { n } } _ { k } ^ { \mathrm { ( i n ) } } , d _ { j i } , d _ { j k } , \cos \theta _ { i j k } \big ) \Big ) ,
$$

where ${ \tilde { \rho } } _ { 3 }$ absorbs the switch factors $s _ { a } ( d _ { j i } ) s _ { a } ( d _ { j k } )$ into $\rho _ { 3 }$ , both being continuous functions of the inputs $d _ { j i }$ and $d _ { j k }$ of $\rho _ { 3 }$ . Conversely, since $s _ { a }$ is positive on $[ 0 , r _ { c } )$ , on compact sets the desired summand is realized by letting $\rho _ { 3 }$ approximate the quotient by the switch factors. This has DeepSets form $\psi _ { 3 } \circ \sum \circ \tilde { \rho } _ { 3 }$ over $\mu ( \mathcal { L } )$ , where each summand ${ \tilde { \rho } } _ { 3 }$ acts on one element of the multiset. The multiset has at most $M ( M - 1 )$ elements. For any $g \in C _ { \mathrm { s y m } } ^ { 0 } ( \mathcal { M } _ { M ( M - 1 ) } ( X ) )$ , compact $K ^ { \prime } \subset \mathcal { M } _ { M ( M - 1 ) } ( X )$ , and $\delta > 0$ Lemma 4.2, applied with suficiently large $d _ { a }$ and with X replaced by a compact subset containing the tuple values over $K ^ { \prime }$ , yields MLP weights for $\rho _ { 3 }$ and $\psi _ { 3 }$ such that $\| \pmb { n } _ { j } ^ { \mathrm { ( o u t ) } } - g \| _ { C ^ { 0 } ( K ^ { \prime } ) } < \delta$

Step 2: $\mu ^ { * }$ maps $C _ { \mathrm { s y m } } ^ { 0 } ( \mathcal { M } _ { M ( M - 1 ) } ( X ) )$ into $C _ { \Gamma } ( \mathcal { W } )$ and is surjective.

$\mu ^ { * }$ is well-defined. The map µ is continuous and Γ-invariant. Continuity holds because each element of the multiset depends continuously on L via (4.1). For invariance, any $\pi \in \Gamma \subset S _ { n }$ permutes the ordered pairs $( i , k )$ with $i \neq k$ among themselves, and the multiset is unordered, so $\mu ( { \boldsymbol { \pi } } \cdot { \boldsymbol { \mathcal { L } } } ) = \mu ( { \boldsymbol { \mathcal { L } } } )$ . Hence for $g \in C _ { \mathrm { s y m } } ^ { 0 } ( \mathcal { M } _ { M ( M - 1 ) } ( X ) )$ the pullback $g \circ \mu$ is continuous (composition of continuous maps) and Γ-invariant, $\mathrm { i . e . , ~ } g \circ \mu \in C _ { \Gamma } ( \mathcal { W } )$

$\mu ^ { * }$ is surjective. It sufices to show that $\mu$ is injective modulo Γ, i.e., $\mu ( \mathcal { L } _ { 1 } ) =$ $\mu ( { \mathcal { L } } _ { 2 } ) \Rightarrow { \mathcal { L } } _ { 1 } \sim _ { \Gamma } { \mathcal { L } } _ { 2 }$ . Since $( n _ { k } ^ { \mathrm { ( i n ) } } , d _ { j k } )$ is distinct for distinct k (by hypothesis), each element $( { \pmb n } _ { i } , { \pmb n } _ { k } , d _ { j i } , d _ { j k } , \cos \theta _ { i j k } )$ of the multiset uniquely identifies the atoms i and k. Equal multisets therefore imply a permutation $\pi \in S _ { n }$ with ${ \pmb n } _ { \pi ( k ) } ^ { ( 2 ) } = { \pmb n } _ { k } ^ { ( 1 ) }$ $d _ { j , \pi ( k ) } ^ { ( 2 ) } = d _ { j k } ^ { ( 1 ) }$ , and cos $\theta _ { \pi ( i ) j \pi ( k ) } ^ { ( 2 ) } = \cos \theta _ { i j k } ^ { ( 1 ) }$ for all $i \neq k$ , where the superscripts refer to $\mathcal { L } _ { 1 }$ and $\mathcal { L } _ { 2 }$ . Since the node features of neighbors of diferent types lie in disjoint sets, the matching ${ \pmb n } _ { \pi ( k ) } ^ { ( 2 ) } = { \pmb n } _ { k } ^ { ( 1 ) }$ forces $\pi$ to be type-preserving, that is, $\pi \in \Gamma$ . By (4.2), the matched distances and angles give $G _ { \pi ( i ) , \pi ( k ) } ^ { ( 2 ) } = G _ { i k } ^ { ( 1 ) }$ , so $\mathcal { L } _ { 1 } \sim _ { \Gamma } \mathcal { L } _ { 2 }$

Now let $f \in C _ { \Gamma } ( \mathcal { W } )$ and $K \subset \mathcal { W }$ compact. Since µ is injective modulo Γ, $\mu ( \mathcal { L } _ { 1 } ) =$ $\mu ( \mathcal { L } _ { 2 } ) \Rightarrow f ( \mathcal { L } _ { 1 } ) = f ( \mathcal { L } _ { 2 } )$ , so f factors as $f = g _ { 0 } \circ \mu$ for a well-defined function g on the image $\mu ( K )$ . The function $g _ { 0 }$ is continuous on $\mu ( K )$ . The restriction $\mu | _ { K }$ is a continuous map from the compact set K to the metrizable space $\mathscr { M } _ { M ( M - 1 ) } ( X )$ hence a closed map, and for any closed $C \subset \mathbb { R } , g _ { 0 } ^ { - 1 } ( C ) = \mu ( f ^ { - 1 } ( C ) \cap K )$ is closed in $\mu ( K )$ . Since $\mu ( K )$ is a compact subset of the metrizable space $\mathscr { M } _ { M ( M - 1 ) } ( X )$ , the Tietze extension theorem provides a continuous extension $g \colon { \mathcal { M } } _ { M ( M - 1 ) } ( X ) \to$ R with $g | _ { \mu ( K ) } = g _ { 0 }$

Step 3: approximation. Let $f \in C _ { \Gamma } ( \mathcal { W } ) , K \subset \mathcal { W }$ compact, and $\delta > 0$ . By Step $2 ,$ $f = g \circ \mu$ on K with $g \in C _ { \mathrm { s y m } } ^ { 0 } ( \mathcal { M } _ { M ( M - 1 ) } ( X ) )$ . By Step 1, there exist MLP weights such that the HGNN output ˜g satisfies $\| \tilde { g } - g \| _ { C ^ { 0 } ( \mu ( K ) ) } < \delta$ . For every ${ \mathcal { L } } \in K$

$$
| ( \tilde { g } \circ \mu ) ( \mathcal { L } ) - f ( \mathcal { L } ) | = | \tilde { g } ( \mu ( \mathcal { L } ) ) - g ( \mu ( \mathcal { L } ) ) | < \delta ,
$$

so the HGNN approximates $f$ uniformly on $K$

Applying this procedure to each component of the complete representation of Proposition 3.7, we prove that a single HGNN layer approximates the representation on generic environments.

Proposition 4.4 (Single-layer HGNN approximates a complete representation).

Let the HGNN have a single layer with $r _ { c } ^ { ( 3 ) } = r _ { c } ^ { ( 2 ) } = r _ { c }$ and suficient feature dimensions $d _ { a }$ and d. For any compact $K \subset \mathcal { E } ^ { ( r _ { c } ) } \cap \mathcal { E } ^ { * }$ whose environments have at least two neighbors, and any $\delta > 0$ , there exist MLP weights such that the HGNN output approximates the complete representation $\Phi ^ { * }$ of Proposition 3.7 with $\| \Phi -$ $\Phi ^ { * } \big \| _ { C ^ { 0 } ( K ) } < \delta$

Proof. Work on a fixed stratum $\mathcal { E } _ { z _ { 0 } , z } ^ { ( r _ { c } ) }$ . The cross-stratum case follows because type embeddings place each stratum’s data in a disjoint region of the MLP input space, and by MLP universality a single set of weights can approximate independent targets on each of the finitely many strata.

By Proposition 3.7, the components of the complete representation $\Phi ^ { * }$ , restricted to the stratum, are polynomials in the entries of the switched Gram matrix. These entries are continuous functions of the distances and angles, so each component is a continuous Γ-invariant function of the labeled data.

For the single-layer case, the input node features are type embeddings $n _ { k } ^ { ( 0 ) } =$ type embed $\left( z _ { k } \right)$ . For generic configurations, distinct same-type atoms have distinct distances $d _ { j k }$ , so the pair $( n _ { k } ^ { ( 0 ) } , d _ { j k } )$ is distinct for distinct k. The type embeddings are chosen distinct, so the features of neighbors of diferent types lie in disjoint sets, and both hypotheses of Lemma 4.3 are satisfied.

By Lemma 4.3 applied to each component, there exist MLP weights such that the HGNN output $\Phi ( \mathcal D )$ approximates $\Phi ^ { * } ( { \mathcal { D } } )$ to within δ uniformly on $K$ □

4.3. Multi-layer approximation. We now extend the single-layer result to the multi-layer HGNN. We first introduce the L-hop neighborhoods and environments that an L-layer network sees, then the two conditions on the configuration that the main theorem requires. These are connectivity, which makes $L$ layers reach the physical range $\textstyle R _ { c } ,$ and overlap, which lets each layer reconstruct the extended geometry from local data. We then show that this reconstruction is possible in Lemma 4.10 and prove the multi-layer approximation theorem in Theorem 4.11. Two collections of Gram matrices recur in what follows. For a center atom $j .$

$$
\mathcal { T } _ { j } = \left( G ^ { ( j ) } , \{ G ^ { ( k ) } \} _ { k \in \mathcal { N } ( j ) } \right)\tag{4.4}
$$

gathers the Gram matrices of $j$ and of each of its neighbors, while $\bar { G } ^ { ( j ) }$ denotes the Gram matrix of the extended environment obtained by merging all the neighbors’ environments.

Definition 4.5 (L-hop neighborhood and environment). For cutof $r _ { c }$ , define the 1-hop neighbor set $\mathcal { N } ( j ) = \{ k : k \neq j , | \pmb { r } _ { k } - \pmb { r } _ { j } | < r _ { c } \}$ . The L-hop neighbor set is defined inductively by $\mathcal { N } ^ { 1 } ( j ) = \mathcal { N } ( j )$ and $\mathcal { N } ^ { l + 1 } ( j ) \overset { \cdot } { = } \left( \mathcal { N } ( j ) \cup \bigcup _ { k \in \mathcal { N } ( j ) } \mathcal { N } ^ { l } ( k ) \right) \backslash \{ j \}$ , so that the center never counts among its own neighbors. The L-hop local environment is

$$
\mathcal { D } _ { j } ^ { [ r _ { c } , L ] } = \big ( z _ { j } , \{ ( \Delta r _ { j k } , z _ { k } ) : k \in \mathcal { N } ^ { L } ( j ) \} \big ) , \qquad \Delta r _ { j k } = r _ { k } - r _ { j } .\tag{4.5}
$$

The L-hop neighborhood $\mathcal { N } ^ { L } ( j )$ contains every atom reachable from $j$ by a chain of at most L steps, each of length less than $r _ { c }$ . Note that $\mathcal { N } ^ { L } ( j )$ is in general a subset of the ball $\{ k : | \boldsymbol { r } _ { k } - \boldsymbol { r } _ { j } | < L \boldsymbol { r } _ { c } \}$ . An atom within distance $L r _ { c }$ may not be reachable if there are gaps in the neighbor structure.

Definition 4.6 (L-hop environment space). The L-hop environment space $\mathcal { E } ^ { [ r _ { c } , L ] }$ is the set of all L-hop local environments $\mathcal { D } _ { j } ^ { [ r _ { c } , L ] }$ , as $j$ ranges over the atoms of all configurations.

![](images/7b0fe5bff720601bef300ee9e4e8ce9a1df0cd3d11c59cadbb00e37b61eff069.jpg)  
Fig. 2. Extending the receptive field via multi-layer message passing. The central atom j (star) has a per-layer cutof $r _ { c } .$ The dashed circle marks the physical interaction range $\begin{array} { r l } { R _ { c } . } & { { } ( a ) } \end{array} \overbrace { A f t e r }$ one layer, only the 1-hop neighbors (blue) are reached. (b) After two layers, the 2-hop neighborhood (green) extends further, but atoms near the boundary of $R _ { c }$ remain unreached. (c) After three layers, the 3-hop neighborhood (orange) covers all atoms within $R _ { c }$ . Note that the L-hop neighborhood is in general a subset of the ball of radius $L r _ { c }$ (dotted circles), since an atom within distance $L r _ { c }$ may not be reachable in L graph hops if intermediate neighbors are absent. The connectivity condition guarantees that suficiently many layers cover $R _ { c }$ . The overlap condition ensures that each layer faithfully reconstructs the extended geometry from local data.

Since every atom reachable from $j$ in at most L hops of length less than $r _ { c }$ lies within distance $L r _ { c }$ of $j ,$ , we have the inclusion $\mathcal { E } ^ { [ r _ { c } , L ] } \mathsf { \bar { \Omega } } _ { \subset } ^ { } \mathcal { E } ^ { ( L r _ { c } ) }$ . The membership condition is intrinsic. An environment $\mathcal { D } \in \mathcal { E } ^ { ( L r _ { c } ) }$ lies in $\mathcal { E } ^ { [ r _ { c } , L ] }$ if and only if every neighbor of the central atom is connected to it by a chain of at most L steps, with each step linking two atoms at distance less than $r _ { c } .$ . This condition can be expressed as a finite disjunction over all such chains of finitely many strict inequalities. Hence $\mathcal { E } ^ { [ r _ { c } , L ] }$ is an open subset of $\mathcal { E } ^ { ( L r _ { c } ) }$ , inheriting its stratification, metric, and diferentiable structure. We call an L-hop environment generic if the local environment of every atom in it is generic in the sense of Definition 3.4 and no two of its atoms lie at distance exactly $r _ { c }$

Definition 4.7 (Connectivity condition). A configuration satisfies the connectivity condition with respect to cutof $r _ { c }$ and physical range $R _ { c }$ if, for every atom $j$ and every atom k $\neq j$ with $| r _ { k } - r _ { j } | < R _ { c }$ , there exists $L \geq 1$ such that $k \in \mathcal { N } ^ { L } ( j )$

Since a configuration has finitely many atoms, the hop depths required for the individual pairs admit a finite maximum. Hence, a single $L$ sufices for all pairs simultaneously. Consequently, $\mathcal { D } _ { j } ^ { [ r _ { c } , L ] } \supseteq \mathcal { D } _ { j } ^ { ( R _ { c } ) }$ for every atom $j$ , meaning that an L-layer HGNN efectively covers the entire physical interaction range $R _ { c } .$ . See Figure 2 for an illustration. In condensed matter systems, where atoms typically form a connected network at cutof $r _ { c }$ , this condition is generally satisfied.

Definition 4.8 (Overlap condition). Write $\mathcal { N } [ j ] = \mathcal { N } ( j ) \cup \{ j \}$ for the closed neighborhood of atom $j$ . A configuration satisfies the overlap condition with respect to $r _ { c }$ if every atom has a neighbor and, for every ordered pair of atoms $( j , k )$ with $| r _ { j } - r _ { k } | < r _ { c }$ , the displacements $\{ \pmb { r } _ { m } - \pmb { r } _ { k } : m \in \mathcal { N } [ j ] \cap \mathcal { N } [ k ] \}$ span $\mathbb { R } ^ { 3 }$

Both $j$ and k lie in $\mathcal { N } [ j ] \cap \mathcal { N } [ k ]$ , since $| r _ { j } - r _ { k } | < r _ { c }$ . Spanning $\mathbb { R } ^ { 3 }$ is equivalent to the existence of three atoms $m _ { 1 } , m _ { 2 } , m _ { 3 }$ in $\mathcal { N } [ j ] \cap \mathcal { N } [ k ]$ whose displacements relative to k are linearly independent, which is the form used in Lemma 4.10. Three is exactly the number required there, since the relative orientation of the local frames of j and k is recovered as a linear map of $\mathbb { R } ^ { 3 }$ , and such a map is determined by its values on a basis.

The overlap condition requires $r _ { c }$ to be large enough that two neighboring atoms share enough common neighbors to span $\mathbb { R } ^ { 3 }$ . In condensed matter systems, a cutof slightly above the nearest-neighbor distance typically sufices.

Definition 4.9 (Overlap condition for L-hop environments). The hop depth of an atom in an L-hop environment is the smallest number of hops connecting it to the center. An L-hop environment satisfies the overlap condition if its center has a neighbor and the spanning requirement of Definition 4.8 holds for every ordered pair of its atoms $( j , k )$ with $| r _ { j } - r _ { k } | < r _ { c }$ in which j has hop depth at most $L - 1$ . Here the neighborhoods are computed within the environment.

The depth restriction exempts the outermost atoms, whose shared neighbors may lie beyond the L-hop horizon. For a pair whose first atom has hop depth at most $L - 1$ , the shared neighbors lie within one hop of that atom, hence at depth at most L. Consequently, every L-hop environment of a configuration satisfying the overlap condition satisfies Definition 4.9.

The inductive step of the multi-layer theorem needs a center atom to recover its extended environment from what its neighbors already encode. At the level of Gram matrices, we have

Lemma 4.10 (Information suficiency). Under the overlap condition at $r _ { c . }$ , the data $\mathcal { T } _ { j }$ uniquely determines the extended Gram matrix $\bar { G } ^ { ( j ) }$ . Moreover, $\bar { G } ^ { ( j ) }$ is a continuous function of $\mathcal { T } _ { j }$ .

Proof. Every atom a in the extended environment belongs to the environment of at least one $k _ { a } \in \mathcal { N } ( j )$ . We compute each entry $\bar { G } _ { a b } ^ { ( j ) }$ for two cases.

Case 1: a, b belong to a common neighbor’s environment. Let $k \in \mathcal { N } ( j )$ have both a and b in its environment. Since $k \in \mathcal { N } ( j )$ implies $j \in \mathcal { N } ( k )$ , atom j appears in $G ^ { ( k ) }$ Direct computation gives

$$
\bar { G } _ { a b } ^ { ( j ) } = G _ { a b } ^ { ( k ) } - G _ { a j } ^ { ( k ) } - G _ { b j } ^ { ( k ) } + d _ { j k } ^ { 2 } ,\tag{4.6}
$$

where $d _ { j k } ^ { 2 } = G _ { k k } ^ { ( j ) }$ . All quantities on the right-hand side are known from $\mathcal { T } _ { j }$

Case 2: a and b belong to diferent neighbors’ environments with no common environment containing both. Let a belong to $k _ { a } \mathrm { ^ { * } s }$ environment and b to $k _ { b } \mathrm { ^ { * } s }$ , with $k _ { a } \ \ne \ k _ { b }$ . This requires determining the relative orientation between $k _ { a } \mathrm { { ' s } }$ and $k _ { b }$ ’s local environments via frame alignment.

Gram matrix reconstruction. From $G ^ { ( j ) } = X ^ { ( j ) } ( X ^ { ( j ) } ) ^ { T }$ , choose a position matrix $\pmb { X } ^ { ( j ) }$ (determined up to right-multiplication by $R \in O ( 3 ) )$ . Row k gives a vector $\mathbf { \Delta } \mathbf { u } _ { k }$ with $\Delta \pmb { r } _ { k } ^ { ( j ) } = R \pmb { u } _ { k }$ . Similarly, from $G ^ { ( k _ { a } ) }$ , choose $X ^ { ( k _ { a } ) }$ (up to $R _ { k _ { a } } ~ \in ~ O ( 3 ) )$ . Throughout, the center of each frame is assigned the zero vector, so that Gram entries carrying the index of the center vanish.

Frame alignment via shared atoms. By the overlap condition, $\mathcal { N } [ j ] \cap \mathcal { N } [ k _ { a } ]$ contains atoms $m _ { 1 } , m _ { 2 } , m _ { 3 }$ whose displacements relative to $k _ { a }$ are linearly independent. Each shared atom m satisfies $R _ { k _ { a } } \mathbf { v } _ { m } ^ { ( k _ { a } ) } = R ( \mathbf { u } _ { m } - \mathbf { u } _ { k _ { a } } )$ . Define $V _ { k _ { a } }$ and $U _ { k _ { a } }$ as the $3 \times 3$ matrices with columns ${ \pmb v } _ { m _ { i } } ^ { ( k _ { a } ) }$ and ${ \pmb u } _ { m _ { i } } - { \pmb u } _ { k _ { a } }$ , respectively. Linear independence implies

$V _ { k _ { a } }$ is non-singular. Then

$$
T _ { k _ { a } } \equiv R ^ { - 1 } R _ { k _ { a } } = U _ { k _ { a } } V _ { k _ { a } } ^ { - 1 } .\tag{4.7}
$$

Expansion coeficients. Define $\pmb { c } _ { a } = V _ { k _ { a } } ^ { - 1 } \pmb { v } _ { a } ^ { ( k _ { a } ) } \in \mathbb { R } ^ { 3 }$ , so that the position of a in $j ^ { \prime }$ s frame is $\mathbf { } { \pmb u } _ { a } = { \pmb u } _ { k _ { a } } + U _ { k _ { a } } { \pmb c } _ { a }$ . The coeficients $c _ { a }$ satisfy the normal equations

$$
\begin{array} { r } { \left( V _ { k _ { a } } ^ { T } V _ { k _ { a } } \right) \pmb { c } _ { a } = V _ { k _ { a } } ^ { T } \pmb { v } _ { a } ^ { ( k _ { a } ) } , } \end{array}\tag{4.8}
$$

where $( V _ { k _ { a } } ^ { T } V _ { k _ { a } } ) _ { i j } = G _ { m _ { i } , m _ { j } } ^ { ( k _ { a } ) }$ and $( V _ { k _ { a } } ^ { T } { \pmb v } _ { a } ^ { ( k _ { a } ) } ) _ { i } = G _ { m _ { i } , a } ^ { ( k _ { a } ) }$ are entries of $G ^ { ( k _ { a } ) }$ . Since $V _ { k _ { a } }$ is non-singular,

$$
\pmb { c } _ { a } = \left( G _ { m _ { i } , m _ { j } } ^ { ( k _ { a } ) } \right) ^ { - 1 } \left( G _ { m _ { i } , a } ^ { ( k _ { a } ) } \right) ,
$$

a continuous function of Gram entries of $G ^ { ( k _ { a } ) }$ . Similarly for $c _ { b }$ via $G ^ { ( k _ { b } ) }$

Inner product formula. Expanding ${ \pmb u } _ { a } \cdot { \pmb u } _ { b } = ( { \pmb u } _ { k _ { a } } + { \pmb U } _ { k _ { a } } { \pmb c } _ { a } ) \cdot ( { \pmb u } _ { k _ { b } } + { \pmb U } _ { k _ { b } } { \pmb c } _ { b } )$ and using $\pmb { u } _ { \alpha } \cdot \pmb { u } _ { \beta } = G _ { \alpha , \beta } ^ { ( j ) }$ :

$$
\bar { G } _ { a b } ^ { ( j ) } = G _ { k _ { a } , k _ { b } } ^ { ( j ) } + c _ { a } ^ { T } g _ { k _ { a } \to k _ { b } } + c _ { b } ^ { T } g _ { k _ { b } \to k _ { a } } + c _ { a } ^ { T } M _ { k _ { a } , k _ { b } } c _ { b } ,\tag{4.9}
$$

where

(4.10)

$$
( g _ { k _ { a }  k _ { b } } ) _ { i } = G _ { m _ { i } , k _ { b } } ^ { ( j ) } - G _ { k _ { a } , k _ { b } } ^ { ( j ) } ,\tag{4.11}
$$

$$
( M _ { k _ { a } , k _ { b } } ) _ { i j } = G _ { m _ { i } , m _ { j } ^ { \prime } } ^ { ( j ) } - G _ { m _ { i } , k _ { b } } ^ { ( j ) } - G _ { k _ { a } , m _ { j } ^ { \prime } } ^ { ( j ) } + G _ { k _ { a } , k _ { b } } ^ { ( j ) } .
$$

Here $m _ { 1 } ^ { \prime } , m _ { 2 } ^ { \prime } , m _ { 3 } ^ { \prime }$ denote the shared atoms chosen for the pair $( j , k _ { b } )$ , in analogy with the $m _ { i }$ for $( j , k _ { a } )$ . Every term is a continuous function of entries of $\mathcal { T } _ { j }$

Independence $o f$ the choice of shared atoms. The coeficients $\mathbf { } c _ { a } , g _ { k _ { a }  k _ { b } }$ , and $M _ { k _ { a } , k _ { b } }$ all depend on the choice of shared atoms m<sub>1</sub>, m<sub>2</sub>, m<sub>3</sub>. However, the final value $\bar { G } _ { a b } ^ { ( j ) }$ does not: since ${ \pmb u } _ { a } = { \pmb u } _ { k _ { a } } + U _ { k _ { a } } { \pmb c } _ { a }$ and $T _ { k _ { a } } = U _ { k _ { a } } V _ { k _ { a } } ^ { - 1 }$ is the unique linear map sending $\pmb { v } _ { m } ^ { ( k _ { a } ) } \mapsto \pmb { u } _ { m } - \pmb { u } _ { k _ { a } }$ for all shared atoms m, the product $U _ { k _ { a } } \mathbf { c } _ { a } = T _ { k _ { a } } \mathbf { v } _ { a } ^ { ( k _ { a } ) }$ is independent of which triple defines $V _ { k _ { a } }$ and $U _ { k _ { a } }$ . Hence ${ \pmb u } _ { a }$ is the same for any valid triple, and $\bar { G } _ { a b } ^ { ( j ) } = \pmb { u } _ { a } \cdot \pmb { u } _ { b }$ is well-defined.

In both cases, $\bar { G } _ { a b } ^ { ( j ) }$ is a well-defined continuous function of $\mathcal { T } _ { j }$ . Since this holds for every pair $( a , b )$ , the entire extended Gram matrix $\bar { G } ^ { ( j ) }$ is uniquely and continuously determined by $\mathcal { T } _ { j }$ □

We are ready to prove the multi-layer theorem. The argument proceeds by induction on L. At each layer, Lemma 4.3 provides the required approximation capabilities, while Lemma 4.10 ensures that the information reaching that layer is suficient to determine the extended environment it must represent. The induction then composes these layerwise approximations to establish the full L-layer completeness result.

Theorem 4.11 (Multi-layer HGNN approximates a complete representation). Let the L-layer HGNN have equal cutofs $r _ { c } ^ { ( 3 ) } = r _ { c } ^ { ( 2 ) } = r _ { c }$ and suficient feature dimensions $d _ { a }$ and d. Then there is a continuous complete representation $\Phi ^ { * , ( L ) }$ on $\mathcal { E } ^ { [ r _ { c } , L ] }$ such that for every compact $K \subset \mathcal { E } ^ { [ r _ { c } , L ] }$ consisting of generic environments that satisfy the overlap condition, and every $\delta > 0$ , there exist MLP weights for which the L-layer HGNN representation $\Phi ^ { ( L ) } : \bar { \mathcal { D } } _ { j } \mapsto n _ { j } ^ { ( L ) }$ satisfies $\lVert \Phi ^ { ( L ) } - \Phi ^ { * , ( L ) } \rVert _ { C ^ { 0 } ( K ) } < \delta$

Proof. By induction on L. We prove the statement $P ( L )$ : there is a continuous complete representation $\Phi ^ { * , ( L ) } o n \ \bar { \mathcal { E } } ^ { [ r _ { c } , L ] }$ such that for every compact K as in the statement of the theorem and every $\delta > 0$ , there exist MLP weights for the L-layer HGNN with $\lVert \Phi ^ { ( L ) } - \Phi ^ { * , ( L ) } \rVert _ { C ^ { 0 } ( K ) } < \delta$

The overlap condition gives every center in such a K at least two neighbors, since the pair formed with one neighbor has a spanning triple containing at least two atoms of $\mathcal { N } ( j )$

Base case $P ( 1 )$ follows from Proposition 4.4.

Assume $P ( L )$ holds. Fix the weights for layers $1 , \ldots , L$ as guaranteed by $P ( L )$ at an accuracy that may be taken as small as needed. We find weights for layer $L + 1$

For $k \in \mathcal { N } ( j )$ , genericity keeps every interatomic distance away from $r _ { c } .$ , so the hop structure is locally constant and the sub-environment $\mathcal { D } _ { k } ^ { [ r _ { c } , L ] }$ depends continuously on $\mathcal { D } _ { i } ^ { [ r _ { c } , L + 1 ] }$ . The set $K _ { L }$ of all sub-environments arising from K is therefore compact. Genericity passes to sub-environments. The overlap condition passes as well. An atom of hop depth at most $L - 1$ in a sub-environment $\mathsf { \bar { D } } _ { k } ^ { [ r _ { c } , L ] }$ has hop depth at most L in $\mathcal { D } _ { j } ^ { [ r _ { c } , L + 1 ] }$ , so the overlap condition of the parent applies to its pairs, and the shared neighbors thus provided lie within one hop of that atom, hence at depth at most L and inside the sub-environment. Each sub-environment center k retains its neighbor $j ,$ , as Definition 4.9 requires. Hence $K _ { L }$ qualifies for $P ( L )$

Let $\mathcal { W } ^ { ( L ) } = \mathcal { V } \times ( \mathbb { R } ^ { d } ) ^ { n }$ denote the labeled data space for layer $L + 1$ , and define the map ν : $\mathcal { D } _ { i } ^ { [ r _ { c } , L + 1 ] } \mapsto \mathcal { W } _ { j } = ( G ^ { ( j ) } , ( \Phi ^ { * , ( L ) } ( \mathcal { D } _ { 1 } ^ { [ r _ { c } , L ] } ) , \dots , \Phi ^ { * , ( L ) } ( \mathcal { D } _ { n } ^ { [ r _ { c } , L ] } ) ) ) \in \mathcal { W } ^ { ( L ) }$ , which labels each neighbor with the exact complete representation of its L-hop environment supplied by $P ( L )$ . The node features computed by the first L layers approximate these labels, and we account for the diference at the end of the proof.

We first show that ν is injective modulo G on the environments of K. Since $\begin{array} { r } { \mathcal { N } ^ { L + 1 } ( j ) = \left( \bigcup _ { k \in \mathcal { N } ( j ) } ( \{ k \} \cup \mathcal { N } ^ { L } ( k ) ) \right) \backslash \{ j \} } \end{array}$ , every atom in the (L+1)-hop environment appears in some $k ^ { \mathrm { ^ { \circ } s } } \ L \mathrm { - h o p }$ environment. The k-th label is a complete invariant of $\mathcal { D } _ { k } ^ { [ \bar { r } _ { c } , L ] }$ , so it determines the Gram matrix $G ^ { ( k ) }$ and the atom types of that environment up to a relabeling of its atoms. Genericity fixes the relabeling where it matters. For $k \in \mathcal { N } ( j )$ , the distances from k to its neighbors are pairwise distinct, and the distance from $k$ to any atom of $\mathcal { N } [ j ]$ is computable from $G ^ { ( j ) }$ . Hence every atom of $\mathcal { N } [ j ] \cap \mathcal { N } [ k ]$ , and in particular the center $j ,$ , occupies a determined index in $G ^ { ( k ) }$ . The labeled data ${ \mathcal { W } } _ { j }$ therefore determines $\bar { \mathcal { T } _ { j } } \overset { \cdot } { = } ( G ^ { ( j ) } , \{ G ^ { ( k ) } \} _ { k \in \mathcal { N } ( j ) } )$ together with these identifications. Suppose two environments of K have the same image under ν up to Γ. Then they have the same collection $\mathcal { T } _ { j }$ of Gram matrices, so by Lemma 4.10 they have the same extended Gram matrix $\bar { G } ^ { ( j ) }$ , and the labels give their atoms the same types. By Lemma 3.5 the positions then agree up to an orthogonal transformation, so the two environments are symmetry-equivalent.

This lets the target representation be pulled back to the labeled data. By Proposition 3.7 applied at cutof $( L + 1 ) r _ { c }$ , a continuous complete representation exists on $\mathcal { E } ^ { ( ( L + 1 ) r _ { c } ) }$ . Since $\mathcal { E } ^ { [ r _ { c } , L + 1 ] }$ is a G-invariant subset, its restriction $\bar { \Phi } ^ { * , ( L + 1 ) }$ is continuous and complete there. Because ν is injective modulo G on $K , \Phi ^ { * , ( L + 1 ) }$ factors on K as $\Phi ^ { * , ( L + 1 ) } = g \circ \nu$ for a continuous Γ-invariant function g on ${ \mathcal { W } } ^ { ( L ) }$ , by the same factorization and Tietze extension argument used in the proof of Lemma 4.3.

It remains to approximate g with a single layer. $\mathrm { A s }$ in the proof of Proposition 4.4, it sufices to work on a fixed stratum. The cross-stratum case follows from the label separation established below. We verify the hypotheses of Lemma 4.3 for the labels $( \Phi ^ { * , ( L ) } ( \mathcal { D } _ { k } ^ { [ r _ { c } , L ] } ) , d _ { j k } ) , k \in \mathcal { N } ( j )$ . Let $k \neq k ^ { \prime }$ in $\mathcal { N } ( j )$ . If $z _ { k } = z _ { k ^ { \prime } }$ , genericity gives $d _ { j k } \neq d _ { j k ^ { \prime } }$ , so the pairs difer in their second component. If $z _ { k } \neq z _ { k ^ { \prime } }$ , the environments $\mathcal { D } _ { k } ^ { [ r _ { c } , L ] }$ and $\mathcal { D } _ { k ^ { \prime } } ^ { [ r _ { c } , L ] }$ have diferent center types, hence lie in diferent strata and are not symmetry-equivalent, so the complete representation $\Phi ^ { * , ( L ) }$ separates them. The separation $\sigma =$ min $\| \Phi ^ { * , ( L ) } ( \mathcal { D } _ { k } ^ { [ r _ { c } , L ] } ) - \Phi ^ { * , ( L ) } ( \mathcal { D } _ { k ^ { \prime } } ^ { [ r _ { c } , L ] } ) \|$ , taken over $K$ and over the finitely many pairs with $z _ { k } \neq z _ { k ^ { \prime } }$ , is positive because $K$ is compact and $\Phi ^ { * , ( L ) }$ is continuous. The labels of neighbors of diferent types therefore lie in disjoint sets, and in either case the pairs are distinct. Both hypotheses of Lemma 4.3 are thus satisfied, so there exist MLP weights for layer $L + 1$ whose realized map $\tilde { g }$ approximates $g$ to within $\delta / 2$ on $\nu ( K )$

Finally, we account for the diference between the exact labels and the computed node features. The realized map $\tilde { g }$ is uniformly continuous on a compact neighborhood of $\nu ( K )$ , so there is an $\epsilon > 0$ such that inputs within ϵ of each other produce outputs within $\delta / 2$ . The exact labels and the computed features difer only in the feature slots, by at most max $_ { k \in { \cal N } ( j ) } \| n _ { k } ^ { ( L ) } - \Phi ^ { * , ( L ) } ( { \cal { D } } _ { k } ^ { [ r _ { c } , L ] } ) \|$ , which is the accuracy of the first $L$ layers on $K _ { L }$ . Invoking $P ( L )$ on $K _ { L }$ at accuracy ϵ therefore keeps the layer- $( L + 1 )$ output on the computed features within $\delta / 2$ of ${ \tilde { g } } \circ \nu .$ . Hence $\Phi ^ { \tilde { ( } L + 1 ) }$ approximates $g \circ \overset { \cdot } { \nu } = \Phi ^ { * , ( L + 1 ) }$ to within $\delta$ on $K$ , establishing $P ( L + 1 )$

By induction, $P ( L )$ holds for all $L \geq 1$

The induction shows that suitable weights exist but does not say what the network computes at each layer. For the atoms that share a neighbor’s environment this can be made explicit, and the resulting formula shows how Gram information propagates one hop at a time.

Remark 4.12 (Constructive Gram propagation). The Case 1 formula (4.6) provides constructive insight into how the HGNN extends Gram information layer by layer. For atoms $a , b$ in a common neighbor’s L-hop environment $\mathcal { D } _ { k } ^ { [ r _ { c } , L ] }$ , the extended Gram entry $\bar { G } _ { a b } ^ { ( j ) } = G _ { a b } ^ { ( k ) } - G _ { a j } ^ { ( k ) } - G _ { b j } ^ { ( k ) } + d _ { j k } ^ { 2 }$ involves only entries of $k \mathrm { { s } }$ Gram matrix and $d _ { j k }$ . The HGNN can evaluate this through $\rho _ { 3 }$ , since by $P ( L )$ the node feature ${ \mathbf { } } n _ { k } ^ { ( L ) }$ approximately encodes a complete invariant of $\mathcal { D } _ { k } ^ { [ r _ { c } , L ] }$ , from which any continuous function of $G ^ { ( \check { k } ) }$ can be extracted by MLP universality. Setting $b = a$ yields $d _ { j a } ^ { 2 } = G _ { a a } ^ { ( k ) } - 2 G _ { a j } ^ { ( k ) } + d _ { j k } ^ { 2 } ,$ , showing that distances to all $( L + 1 )$ -hop atoms are constructively determined. The cross-environment Case $2$ requires the universality argument (Lemma 4.3) rather than an explicit formula.

5. Universal Approximation Theorem. Theorem 4.11 supplies a complete representation of the L-hop environment, and Theorem 3.3 turns completeness into universal approximation. A combination of two results gives the main theorem.

Theorem 5.1 (UAT for HGNN). Let the L-layer HGNN have equal cutofs $r _ { c } ^ { ( 3 ) } = r _ { c } ^ { ( 2 ) } = r _ { c }$ and suficient feature dimensions and MLP capacity per layer. Let $K \subset \mathcal { E } ^ { [ r _ { c } , L ] }$ be a compact set of generic environments that satisfy the overlap condition, each drawn from a configuration satisfying the connectivity condition, with L large enough that $\dot { \mathcal { D } } _ { j } ^ { [ r _ { c } , L ] } \supseteq \dot { \mathcal { D } _ { j } ^ { ( \tilde { R } _ { c } ) } }$ . Then for every $\varepsilon \in \mathcal { F } _ { R _ { c } } ^ { 0 }$ and every $\delta > 0$ there exist parameters θ with $\| f _ { \theta } \circ \Phi ^ { ( L ) } - \varepsilon \| _ { C ^ { 0 } ( K ) } < \delta$

Proof. Let $\Phi ^ { * , ( L ) }$ be the complete representation of Theorem 4.11, whose hypotheses on K are part of the present assumptions. The target $\varepsilon$ depends only on the neighbors within $\mathit { R } _ { c } ,$ so it defines a G-invariant function on $K$ , continuous by condition 3 of Definition 2.6. By completeness of $\Phi ^ { * , ( L ) }$ and the argument of the forward implication of Theorem 3.3, applied on $\mathcal { E } ^ { [ r _ { c } , L ] }$ in place of $\mathcal { E } ^ { ( R _ { c } ) }$ , there exists a readout $f _ { \theta }$ with $\| f _ { \theta } \circ \Phi ^ { * , ( L ) } - \varepsilon \| _ { C ^ { 0 } ( K ) } < \delta / 2$ . The readout $f _ { \theta }$ is uniformly continuous on a compact neighborhood of $\Phi ^ { * , ( L ) } ( K )$ , so there is an $\epsilon > 0$ such that inputs within ϵ produce outputs within $\delta / 2$ . By Theorem 4.11 there exist layer weights with $\| \Phi ^ { ( L ) } - \bar { \Phi } ^ { \ast , ( L ) } \| _ { C ^ { 0 } ( K ) } \overline { { { < \epsilon } } }$ . Combining the two bounds gives $\| f _ { \theta } \circ \Phi ^ { ( L ) } - \varepsilon \| _ { C ^ { 0 } ( K ) } < \delta . \Pi$

The theorem is stated for the HGNN, which was introduced as a reference architecture. Any network that can reproduce its layer inherits the conclusion. We verify this for DPA3 and CHGNet in the appendices.

Corollary 5.2 (UAT for DPA3). Under the same conditions as Theorem 5.1, the L-layer DPA3 architecture with $\begin{array} { r } { \hat { r } _ { c } ^ { ( 3 ) } = r _ { c } ^ { ( 2 ) } = r _ { c } } \end{array}$ , suficient feature dimension, suficient MLP capacity, and readout network $f _ { \theta }$ approximates every $\varepsilon \in \mathcal { F } _ { R _ { c } } ^ { 0 }$ to arbitrary accuracy on $K$

The proof simulates one HGNN layer by a single DPA3 layer and is given in §SM1 of the supplementary materials.

Definition 5.3 (Strong genericity). A local environment $\mathcal { D } \in \mathcal { E } ^ { ( r ) }$ is strongly generic $i f$ no two neighbors are equidistant from the center, that is, $i f i \ne k$ implies $\left| \Delta \boldsymbol { r } _ { i } \right| \neq \left| \Delta \boldsymbol { r } _ { k } \right|$ . An L-hop environment is strongly generic if every local environment within it is strongly generic.

As with Definition 3.4, the excluded set is a finite union of zero sets of nonzero polynomials, so the strongly generic environments form an open dense subset of full measure in each stratum.

Corollary 5.4 (UAT for CHGNet). Under the same conditions as Theorem 5.1, with the atom graph and the bond graph sharing the cutof $r _ { c }$ , and with every environment in K strongly generic in the sense of Definition 5.3, the L-layer CHGNet architecture of §SM2, with suficient feature dimension, suficient MLP capacity, and readout network $f _ { \theta }$ , approximates every $\varepsilon \in \mathcal { F } _ { R _ { c } } ^ { 0 }$ to arbitrary accuracy on K.

The proof follows the same line as that of Corollary 5.2 and is given in §SM2 of the supplementary materials. The strengthening of genericity is needed because the bond message function of CHGNet receives only the center’s node features, so the proof recovers each neighbor’s features from its distance by a lookup, which requires all neighbor distances to be distinct. ALIGNN shares the same angle → edge → node topology but difers crucially in that it constructs its messages from single gated linear layers rather than MLPs, and the previous simulation argument does not carry over. We shall describe the architecture and explain where the argument fails in §SM3 of the supplementary materials.

6. Discussion and perspective. The multi-layer theory justifies the practice of stacking layers with a per-layer cutof smaller than the physical interaction range. The connectivity condition sets how many layers are needed to cover that range, and the overlap condition tells one how small the per-layer cutof may be, for condensed matter systems typically slightly above the nearest-neighbor distance.

At the architectural level, the results show that an invariant HGNN relying solely on scalar geometric inputs attains universal approximation, so raising the LiGS order beyond $K \ : = \ : 2$ , the 3-body angular level, is not necessary for theoretical expressiveness. It is important to emphasize, however, that this is a statement about expressiveness alone. The HGNN is intended as a theoretical reference architecture rather than a practical proposal; universal approximation, in itself, does not address data eficiency, computational cost, or the optimization landscape. Its primary role is to serve as a target for simulation: once a practical architecture can reproduce an HGNN layer, universal approximation follows without the need to re-establish the completeness analysis from scratch.

We carry this out for DPA3 and CHGNet in Corollary 5.2 and Corollary 5.4, respectively. The proofs identify a concrete architectural criterion for guaranteed universal approximation, that the message functions in the aggregation steps must be MLPs, which are universal approximators, not single linear layers. Both architectures satisfy it. ALIGNN [11], which shares the same angle → edge → node topology but uses single gated linear layers in its convolutions, does not (Theorem SM3.1). This distinction has practical implications. Replacing the single gated linear layers in ALIGNN’s convolutions with MLPs would remove the obstruction identified in Theorem SM3.1, although establishing universality for the modified architecture would still require a simulation argument along the lines of §SM2.

The theorems rely on genericity. Completeness is valid on the generic environments of Definition 3.4, which excludes highly symmetric arrangements where sametype atoms are equidistant from the center. The restriction is in fact sharp. The Pozdnyakov counterexample [29] exhibits four same-species atoms on a circle, all sharing identical multisets of distances and triplet angles, yet possessing distinct Gram matrices. No single-layer 3-body invariant can separate such configurations. In practice, thermal fluctuations break exact symmetries, so sampled configurations are generic with probability one. Moreover, in the multi-layer HGNN, the node features of earlier layers help break residual equidistance. The remaining case, zero-temperature crystals, remains a fundamental open problem. Switching to equivariant architectures does not resolve the issue, since equivariant GNNs with a fixed tensor degree l also degenerate on symmetric graphs for specific values of l [8]. Both families also face hierarchies of degeneracies. Raising the body order or increasing the tensor degree resolves progressively more symmetric cases, yet counterexamples are known already at low levels, and no finite level has been shown to achieve completeness across all configurations.

Several other directions remain open. The DeepSets arguments require the anglefeature dimension $d _ { a }$ , which serves as the latent dimension of the aggregation, to grow at least with the number of neighbor triplets, far beyond the typical practical value of 64. Tightening this gap remains an open problem. The present results are also qualitative rather than quantitative. Bounding the approximation rate, that is, how the error decays with network size, would bridge the gap between theory and practical model selection. Finally, extending the analysis from $C ^ { 0 }$ to $C ^ { k }$ approximation, so as to match the smoothness of the potential energy surface including forces and Hessians, requires a more detailed study of the diferentiability of the DeepSets representation. The smooth symmetrization of Pozdnyakov and Ceriotti [28] ofers a promising direction in this regard.

Acknowledgments. The AI-driven experiments, simulations and model training were performed on the robotic AI-Scientist platform of Chinese Academy of Sciences. Pingbing Ming is supported by the National Key R&D Program of China (Grant no. 2024YFA1012502) and by National Natural Science Foundation of China (Grant No. 12371438). Han Wang is supported by the National Key R&D Program of China (Grant No. 2022YFA1004300) and by the National Natural Science Foundation of China (Grants No. 12525113 and No. 12561160120). The authors used AI tools to assist with the development of proofs, manuscript editing, and reference verification. All mathematical proofs have been checked by the authors to the best of their ability, and the authors assume responsibility for all content.

## REFERENCES

[1] A. P. Bartok, R. Kondor, and G. Cs <sup>´</sup> anyi <sup>´</sup> , On representing chemical environments, Phys. Rev. B, 87 (2013), p. 184115.

[2] A. P. Bartok, M. C. Payne, R. Kondor, and G. Cs <sup>´</sup> anyi <sup>´</sup> , Gaussian approximation potentials: The accuracy of quantum mechanics, without the electrons, Phys. Rev. Lett., 104 (2010), p. 136403.

[3] I. Batatia et al., A foundation model for atomistic simulations, arXiv:2401.00096, (2024).

[4] I. Batatia, D. P. Kovacs, G. N. C. Simm, C. Ortner, and G. Cs<sup>´</sup> anyi<sup>´</sup> , MACE: Higher order equivariant message passing neural networks for fast and accurate force fields, in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 11423–11436.

[5] S. Batzner, A. Musaelian, L. Sun, M. Geiger, J. P. Mailoa, M. Kornbluth, N. Molinari, T. E. Smidt, and B. Kozinsky, E(3)-equivariant graph neural networks for data-eficient and accurate interatomic potentials, Nat. Commun., 13 (2022), p. 2453.

[6] J. Behler and M. Parrinello, Generalized neural-network representation of high-dimensional potential-energy surfaces, Phys. Rev. Lett., 98 (2007), p. 146401.

[7] J. Brandstetter, R. Hesselink, E. van der Pol, E. J. Bekkers, and M. Welling, Geometric and physical quantities improve E(3) equivariant message passing, in Proc. ICLR, 2022.

[8] J. Cen, A. Li, N. Lin, Y. Ren, Z. Wang, and W. Huang, Are high-degree representations really unnecessary in equivariant graph neural networks?, in Advances in Neural Information Processing Systems, vol. 37, 2024.

[9] C. Chen and S. P. Ong, A universal graph deep learning interatomic potential for the periodic table, Nat. Comput. Sci., 2 (2022), pp. 718–728.

[10] B. Cheng et al., Foundation models for atomistic simulation of chemistry and materials, Nat. Rev. Chem., (2025). arXiv:2503.10538.

[11] K. Choudhary and B. DeCost, Atomistic line graph neural network for improved materials property predictions, npj Comput. Mater., 7 (2021), p. 185.

[12] G. Cybenko, Approximation by superpositions of a sigmoidal function, Math. Control Signals Syst., 2 (1989), pp. 303–314.

[13] B. Deng, P. Zhong, K. Jun, J. Riebesell, K. Han, C. J. Bartel, and G. Ceder, CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling, Nat. Mach. Intell., 5 (2023), pp. 1031–1041.

[14] R. Drautz, Atomic cluster expansion for accurate and transferable interatomic potentials, Phys. Rev. B, 99 (2019), p. 014104.

[15] G. Dusson, M. Bachmayr, G. Csanyi, R. Drautz, S. Etter, C. van der Oord, and C. Or-<sup>´</sup> tner, Atomic cluster expansion: Completeness, eficiency and stability, J. Comput. Phys., 454 (2022), p. 110946.

[16] J. Gasteiger, F. Becker, and S. Gunnemann<sup>¨</sup> , GemNet: Universal directional graph neural networks for molecules, in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 6790–6802.

[17] J. Gasteiger, J. Groß, and S. Gunnemann <sup>¨</sup> , Directional message passing for molecular graphs, in Proc. ICLR, 2020.

[18] J. Han, Y. Li, L. Lin, J. Lu, J. Zhang, and L. Zhang, Universal approximation of symmetric and anti-symmetric functions, Commun. Math. Sci., 18 (2020), pp. 1157–1180.

[19] K. Hornik, M. Stinchcombe, and H. White, Multilayer feedforward networks are universal approximators, Neural Networks, 2 (1989), pp. 359–366.

[20] C. K. Joshi, C. Bodnar, S. V. Mathis, T. Cohen, and P. Lio<sup>\`</sup>, On the expressive power of geometric graph neural networks, in Proc. ICML, 2023, pp. 15330–15355.

[21] J. Lan, A. Palizhati, M. Shuaibi, B. M. Wood, B. Wander, A. Das, M. Uber, C. L. Zitnick, and Z. W. Ulissi, AdsorbML: A leap in eficiency for adsorption energy calculations using generalizable machine learning potentials, npj Comput. Mater., 9 (2023), p. 172.

[22] Z. Li, X. Wang, S. Kang, and M. Zhang, On the completeness of invariant geometric deep learning models, in Proc. ICLR, 2025. arXiv:2402.04836.

[23] Y. Liu, L. Wang, M. Liu, Y. Lin, X. Zhang, B. Oztekin, and S. Ji, Spherical message passing for 3D molecular graphs, in Proc. ICLR, 2022.

[24] A. Merchant et al., Scaling deep learning for materials discovery, Nature, 624 (2023), pp. 80– 85.

[25] J. Nigam, S. N. Pozdnyakov, K. K. Huguenin-Dumittan, and M. Ceriotti, Completeness

of atomic structure representations, APL Mach. Learn., 2 (2024), p. 016110.

[26] E. Noether, Der Endlichkeitssatz der Invarianten endlicher Gruppen, Math. Ann., 77 (1915), pp. 89–92.

[27] S. N. Pozdnyakov and M. Ceriotti, Incompleteness of graph convolutional neural networks for points clouds in three dimensions, Mach. Learn.: Sci. Technol., 3 (2022), p. 045020.

[28] S. N. Pozdnyakov and M. Ceriotti, Smooth, exact rotational symmetrization for deep learning on point clouds, in Advances in Neural Information Processing Systems, vol. 36, 2023.

[29] S. N. Pozdnyakov, M. J. Willatt, A. P. Bartok, C. Ortner, G. Cs<sup>´</sup> anyi, and M. Ceri-<sup>´</sup> otti, Incompleteness of atomic structure representations, Phys. Rev. Lett., 125 (2020), p. 166001.

[30] K. T. Schutt, H. E. Sauceda, P.-J. Kindermans, A. Tkatchenko, and K.-R. M <sup>¨</sup> uller <sup>¨</sup> , SchNet – a deep learning architecture for molecules and materials, J. Chem. Phys., 148 (2018), p. 241722.

[31] K. T. Schutt, O. T. Unke, and M. Gastegger <sup>¨</sup> , Equivariant message passing for the prediction of tensorial properties and molecular spectra, in Proc. ICML, 2021, pp. 9377–9388.

[32] A. V. Shapeev, Moment tensor potentials: A class of systematically improvable interatomic potentials, Multiscale Model. Simul., 14 (2016), pp. 1153–1173.

[33] P. Tholke and G. D. Fabritiis<sup>¨</sup> , Equivariant transformers for neural network based molecular potentials, in Proc. ICLR, 2022.

[34] N. Thomas, T. Smidt, S. Kearnes, L. Yang, L. Li, K. Kohlhoff, and P. Riley, Tensor field networks: Rotation- and translation-equivariant neural networks for 3D point clouds, arXiv:1802.08219, (2018).

[35] S. Villar, D. W. Hogg, K. Storey-Fisher, W. Yao, and B. Blum-Smith, Scalars are universal: Equivariant machine learning, structured like classical physics, in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 28848–28863.

[36] E. Wagstaff, F. Fuchs, M. Engelcke, I. Posner, and M. A. Osborne, On the limitations of representing functions on sets, in Proc. ICML, 2019, pp. 6487–6494.

[37] Y. Wang et al., Scientific discovery in the age of artificial intelligence, Nature, 620 (2023), pp. 47–60.

[38] T. Wen et al., Specialising and analysing instruction-tuned and byte-level language models for organic reaction prediction, npj Comput. Mater., 10 (2024), p. 1.

[39] H. Weyl, The Classical Groups: Their Invariants and Representations, Princeton University Press, Princeton, 2nd ed., 1946.

[40] M. Zaheer, S. Kottur, S. Ravanbakhsh, B. Poczos, R. Salakhutdinov, and A. Smola<sup>´</sup> , Deep sets, in Advances in Neural Information Processing Systems, vol. 30, 2017.

[41] D. Zhang, J. Zeng, W. Jia, H. Wang, and L. Zhang, DPA-3: A scalable model for the era of large atomistic models, arXiv:2506.01686, (2025).

[42] L. Zhang, J. Han, H. Wang, R. Car, and W. E, End-to-end symmetry preserving interatomic potential energy model for finite and extended systems, in Advances in Neural Information Processing Systems, vol. 31, 2018, pp. 4436–4446.

## Supplementary Materials

## SM1. Proof of the DPA3 corollary.

Proof of Corollary 5.2. It sufices to show that a single DPA3 layer can simulate one HGNN layer. An L-layer HGNN is then simulated by an L-layer DPA3, and the result follows from Theorem 5.1. Each simulated layer realizes a map uniformly continuous on compact sets, so an input error entering a layer propagates to a controlled output error. Since there are finitely many layers, the per-layer simulation accuracies can be chosen so that the final node features are within any prescribed tolerance of the HGNN output on K.

Set $r _ { c } ^ { ( 3 ) } = r _ { c } ^ { ( 2 ) } = r _ { c } .$ . Write ${ \tilde { \rho } } _ { 3 }$ for $s _ { a } ( d _ { j k } ) \rho _ { 3 } ( { \pmb n } _ { j } , { \pmb n } _ { i } , { \pmb n } _ { k } , d _ { j i } , d _ { j k } , \cos \theta _ { i j k } )$ , and define the target intermediate function as

$$
F _ { j i } = \sum _ { k \in \mathcal { N } ( j ) \backslash \{ i \} } \tilde { \rho } _ { 3 } ( n _ { j } , n _ { i } , n _ { k } , d _ { j i } , d _ { j k } , \cos \theta _ { i j k } ) ,
$$

so that the HGNN output (2.17) is $\begin{array} { r } { \psi _ { 3 } \left( \sum _ { i \in \mathcal { N } ( j ) } s _ { a } ( d _ { j i } ) F _ { j i } \right) } \end{array}$ . Fix a layer index l and write $\mathbf { \Omega } _ { n } ( l )$ for its input node features. We exhibit weights for which the four updates (2.18)–(2.21) carry $\mathbf { \Omega } _ { n } ( l )$ to the output of one HGNN layer. Throughout, we maintain the invariant that a designated coordinate of $e _ { j i } ^ { ( l ) }$ carries the distance $d _ { j i }$ and a designated coordinate of $\pmb { a } _ { j ; i , k } ^ { ( l ) }$ carries cos $\theta _ { i j k }$ . At initialization this holds by choosing distance and angle embeddings that write $d _ { j i }$ and cos $\theta _ { i j k }$ into those coordinates.

Step 1: the self-message loads node features into the edges. At initialization, the edge features depend only on distances, $e _ { j i } ^ { ( 0 ) } = \phi _ { d } ( d _ { j i } )$ . The self-message (2.18) gives

$$
e _ { j i } ^ { ( l + \frac { 1 } { 2 } ) } = e _ { j i } ^ { ( l ) } + \delta _ { s } ^ { ( 1 ) } \phi _ { s } ^ { ( 1 ) } \big ( { \pmb n } _ { j } ^ { ( l ) } , { \pmb n } _ { i } ^ { ( l ) } , { \pmb e } _ { j i } ^ { ( l ) } \big ) .\tag{SM1.1}
$$

The MLP $\phi _ { s } ^ { ( 1 ) }$ receives $e _ { j i } ^ { ( l ) }$ along with both endpoint features, so by MLP universality it can approximate $\pmb { x } \mapsto ( \pmb { x } - \pmb { e } _ { j i } ^ { ( l ) } ) / \delta _ { s } ^ { ( 1 ) }$ for any continuous target x. With suficiently large feature dimension d, we may choose the weights so that $e _ { j i } ^ { ( l + \frac { 1 } { 2 } ) }$ encodes $( { \pmb n } _ { j } ^ { ( l ) } , { \pmb n } _ { i } ^ { ( l ) } , d _ { j i } )$ in designated coordinates.

Step 2: the angle self-message. The angle update (2.19) gives ${ \pmb a } _ { j ; i , k } ^ { ( l + 1 ) } = { \pmb a } _ { j ; i , k } ^ { ( l ) } +$ $\delta _ { s } ^ { ( 2 ) } \phi _ { s } ^ { ( 2 ) } ( e _ { j i } ^ { ( l + \frac { 1 } { 2 } ) } , e _ { j k } ^ { ( l + \frac { 1 } { 2 } ) } , { \bf \alpha } _ { j ; i , k } ^ { ( l ) } )$ . We take $\phi _ { s } ^ { ( 2 ) } = 0$ , so the angle features pass through unchanged and the invariant is preserved. By Step 1 and the invariant, the three arguments of $\phi _ { c } ^ { ( 2 ) }$ in the next step then carry all six arguments of ${ \tilde { \rho } } _ { 3 }$

Step 3: the angle aggregation produces $F _ { j i }$ . The edge update (2.20) reads

$$
e _ { j i } ^ { ( l + 1 ) } = e _ { j i } ^ { ( l + \frac { 1 } { 2 } ) } + \delta _ { u } ^ { ( 2 ) } \phi _ { u } ^ { ( 2 ) } \biggl ( \sum _ { k \in \mathcal { N } ( j ) \setminus \{ i \} } w _ { j i k } \phi _ { c } ^ { ( 2 ) } \bigl ( e _ { j i } ^ { ( l + \frac { 1 } { 2 } ) } , e _ { j k } ^ { ( l + \frac { 1 } { 2 } ) } , a _ { j ; i , k } ^ { ( l + 1 ) } \bigr ) \biggr ) .
$$

By Steps 1 and 2 each summand receives all six arguments of ${ \tilde { \rho } } _ { 3 }$ through a continuous encoding, so by MLP universality $\phi _ { c } ^ { ( 2 ) }$ can approximate any continuous function of them. The prefactor $w _ { j i k } = s _ { a } ^ { ( 3 ) } ( d _ { j i } ) s _ { a } ^ { ( 3 ) } ( d _ { j k } )$ is positive and bounded below on K, because genericity keeps all distances below $r _ { c } .$ the set K is compact, and $s _ { a } ^ { ( 3 ) }$ is positive on $[ 0 , r _ { c } )$ . Letting $\dot { \phi } _ { c } ^ { ( 2 ) }$ approximate the quotient of the desired summand by $w _ { j i k }$ whose arguments are available through the edge features, the product $w _ { j i k } \phi _ { c } ^ { ( 2 ) }$ realizes the summand. The sum over k composed with the outer MLP $\phi _ { u } ^ { ( 2 ) }$ has DeepSets form, so by Lemma 4.2 it approximates any continuous symmetric function of the multiset $\{ ( { \pmb n } _ { k } ^ { ( l ) } , d _ { j k } , \cos \theta _ { i j k } ) \} _ { k \neq i }$ . The target below also depends on the parameters $( { \pmb n } _ { j } ^ { ( l ) } , { \pmb n } _ { i } ^ { ( l ) } , d _ { j i } )$ , which the outer MLP recovers from the sum as follows. The inner MLP writes the parameters and the constant 1 into designated coordinates, the sum accumulates the parameters times the summand count and the count itself, and the quotient of the two returns the parameters. The count is at least one, because the spanning triple that the overlap condition provides for the pair $( j , i )$ contains an atom of $\mathcal { N } ( j ) \backslash \{ i \}$ . Choose the target $( { \pmb x } - e _ { j i } ^ { ( l + \frac { 1 } { 2 } ) } ) / \delta _ { u } ^ { ( 2 ) }$ with x encoding $( F _ { j i } , d _ { j i } )$ in designated coordinates, the distance coordinate taken unchanged from $e _ { j i } ^ { ( l + \frac { 1 } { 2 } ) }$ . The target is symmetric because $e _ { j i } ^ { ( l + \frac { 1 } { 2 } ) }$ is constant across the sum over $k ,$ so the residual update gives $e _ { j i } ^ { ( l + 1 ) }$ ≈ x and the invariant is preserved.

Step $\mathit { 4 } \mathit { : }$ the edge aggregation produces the HGNN output. The node update (2.21) reads

$$
{ \pmb n } _ { j } ^ { ( l + 1 ) } = { \pmb n } _ { j } ^ { ( l ) } + \delta _ { u } ^ { ( 1 ) } \phi _ { u } ^ { ( 1 ) } \bigg ( \sum _ { i \in \mathcal { N } ( j ) } w _ { j i } \phi _ { c } ^ { ( 1 ) } \big ( { \pmb n } _ { j } ^ { ( l ) } , { \pmb n } _ { i } ^ { ( l ) } , { \pmb e } _ { j i } ^ { ( l + 1 ) } \big ) \bigg ) .
$$

Since $e _ { j i } ^ { ( l + 1 ) }$ encodes $( F _ { j i } , d _ { j i } )$ , the prefactor $w _ { j i } = s _ { a } ^ { ( 2 ) } ( d _ { j i } )$ is likewise positive and bounded below on $K$ , and $\phi _ { c } ^ { ( 1 ) }$ may approximate the quotient $s _ { a } ( d _ { j i } ) F _ { j i } / w _ { j i }$ , so that $w _ { j i } \phi _ { c } ^ { ( 1 ) } ( . . . ) \approx s _ { a } ( d _ { j i } ) F _ { j i }$ . The sum then gives $\begin{array} { r } { \sum _ { i } w _ { j i } \phi _ { c } ^ { ( 1 ) } ( \dots ) \approx \sum _ { i } s _ { a } ( d _ { j i } ) F _ { j i } } \end{array}$ . The target of the outer MLP $\phi _ { u } ^ { ( 1 ) }$ is $\textstyle ( \psi _ { 3 } ( \sum _ { i } s _ { a } ( d _ { j i } ) F _ { j i } ) - { \pmb n } _ { j } ^ { ( l ) } ) / \delta _ { u } ^ { ( 1 ) }$ , which depends on the parameter ${ n } _ { j } ^ { ( l ) }$ in addition to the aggregate. The device of Step 3 recovers ${ n } _ { j } ^ { ( l ) }$ from the sum, whose summand count $\left| \mathcal { N } ( j ) \right|$ | is positive because the overlap condition grants every center a neighbor. Hence $\begin{array} { r } { { \pmb n } _ { j } ^ { ( l + 1 ) } \approx \psi _ { 3 } ( \sum _ { i } s _ { a } ( d _ { j i } ) F _ { j i } ) } \end{array}$ , the output of one HGNN layer.

Multi-layer extension. Steps 1–4 consume the layer’s input node features $\mathbf { \Omega } _ { n } ( l )$ together with the geometric data maintained by the invariant, produce the output $\mathbf { \Omega } _ { n } ( l { + } 1 )$ , and preserve the invariant, while the self-message reloads the current node features at the start of every layer. Stacking therefore needs no separate loading layer, and an L-layer HGNN is simulated by an L-layer DPA3. □

SM2. CHGNet. CHGNet [SM13] uses the same atom graph and bond graph as DPA3 with $K = 2 \colon$ : atoms as nodes, bonds within $r _ { \mathrm { c u t } }$ as edges, and the line graph ${ \mathcal { L } } ( G )$ encoding angles. We formalize the architecture as written in Eq. (2) of [SM13], with directed edges, so that each bond $\{ j , i \}$ carries two features $e _ { j i }$ and $e _ { i j } . ^ { 3 }$ CHGNet maintains three feature types: node features ${ \boldsymbol n } _ { j } ^ { l }$ , edge features $e _ { j i } ^ { l }$ on $G _ { \ l }$ , and angle features $\pmb { a } _ { i j k } ^ { l }$ on ${ \mathcal { L } } ( G )$ . The initial features are element embeddings (nodes), smooth radial Bessel function expansions (edges), and Fourier basis expansions of angles.

Each layer updates all three feature types. The node update aggregates over

neighbors:

$$
{ n } _ { j } ^ { l + 1 } = { n } _ { j } ^ { l } + L _ { v } \Big [ \sum _ { i \in \mathcal { N } ( j ) } { e } _ { j i } ^ { 0 } \odot \phi _ { v } \left( { n } _ { i } ^ { l } , { n } _ { j } ^ { l } , { e } _ { j i } ^ { l } \right) \Big ] .\tag{SM2.1}
$$

The bond update aggregates over the angles centered at the tail atom $j \colon$

$$
e _ { j i } ^ { l + 1 } = e _ { j i } ^ { l } + L _ { e } \Big [ \sum _ { k \in \mathcal { N } ( j ) \backslash \{ i \} } e _ { j i } ^ { 0 } \odot e _ { j k } ^ { 0 } \odot \phi _ { e } \big ( e _ { j k } ^ { l } , e _ { j i } ^ { l } , a _ { k j i } ^ { l } , n _ { j } ^ { l + 1 } \big ) \Big ] .\tag{SM2.2}
$$

Both edge arguments of $\phi _ { e }$ are the copies directed away from the shared center $j .$ The angle update is:

$$
\begin{array} { r } { \pmb { a } _ { i j k } ^ { l + 1 } = \pmb { a } _ { i j k } ^ { l } + \phi _ { a } \big ( \pmb { e } _ { j i } ^ { l + 1 } , \pmb { e } _ { j k } ^ { l + 1 } , \pmb { a } _ { i j k } ^ { l } , \pmb { n } _ { j } ^ { l + 1 } \big ) . } \end{array}\tag{SM2.3}
$$

Here $L _ { v } , L _ { e }$ are linear layers, $\phi _ { v } , \phi _ { e } , \phi _ { a }$ are gated MLPs (MLPs with sigmoid gating on intermediate activations), and $\odot$ denotes element-wise multiplication. The gate $e _ { j i } ^ { 0 }$ is a trainable linear image of a smooth radial basis expansion of the bond distance, multiplied by a fixed envelope that vanishes at the cutof. By default, CHGNet stacks three full interaction blocks followed by one block with only the node update, and uses a feature dimension of 64. The total energy is $\begin{array} { r } { E _ { \theta } = \sum _ { j } \mathrm { M L P } ( n _ { j } ^ { L } ) } \end{array}$

Proof of Corollary 5.4. We show that two CHGNet layers can approximate the HGNN’s per-layer computation up to the outer readout $\psi _ { 3 }$ , which is absorbed into subsequent blocks or the readout network. As in the proof of Corollary 5.2, write ${ \tilde { \rho } } _ { 3 }$ for $s _ { a } ( d _ { j k } ) \rho _ { 3 } ( { \pmb n } _ { j } , { \pmb n } _ { i } , { \pmb n } _ { k } , d _ { j i } , d _ { j k } , \cos \theta _ { i j k } )$ and $\begin{array} { r } { F _ { j i } = \sum _ { k \in \mathcal { N } ( j ) \backslash \{ i \} } \tilde { \rho } _ { 3 } } \end{array}$ , so that the HGNN layer output is $\begin{array} { r } { \psi _ { 3 } \left( \sum _ { i \in \mathcal { N } ( i ) } s _ { a } \left( d _ { j i } \right) F _ { j i } \right) } \end{array}$

The gates are handled once and for all. Since K is compact and the displacement vectors range over $B _ { r _ { c } } \ \backslash \ \{ 0 \}$ , all interatomic distances in K lie in an interval $[ \delta , r _ { c } -$ $\eta ]$ with $\delta , \eta > 0$ . Choose the trainable coeficients of every gate $e ^ { 0 }$ so that each component equals the first radial basis function times the envelope, which is bounded below on $\left[ \delta , r _ { c } - \eta \right]$ . Absorbing a gate into a message function then means letting the MLP approximate the quotient of the target by the gate, which is continuous on K.

At initialization, edge features carry only distance information: $e _ { j i } ^ { ( 0 ) } = \phi _ { d } ( d _ { j i } )$ 2 and node features are type embeddings: ${ \pmb n } _ { k } ^ { ( 0 ) } = \mathrm { t y p e \_ e m b e d } ( z _ { k } )$

Step 1: node update. The node update (SM2.1) computes

$$
{ \pmb n } _ { j } ^ { ( 1 ) } = { \pmb n } _ { j } ^ { ( 0 ) } + L _ { v } \Big [ \sum _ { m \in \mathcal { N } ( j ) } { \pmb e } _ { j m } ^ { ( 0 ) } \odot \phi _ { v } \big ( { \pmb n } _ { m } ^ { ( 0 ) } , { \pmb n } _ { j } ^ { ( 0 ) } , { \pmb e } _ { j m } ^ { ( 0 ) } \big ) \Big ] ,
$$

where $L _ { v }$ is a linear layer and $\phi _ { v }$ is an MLP. Since $L _ { v }$ is linear, ${ n } _ { j } ^ { ( 1 ) }$ alone does not necessarily encode the full neighbor multiset. However, ${ n } _ { j } ^ { ( 1 ) }$ will serve as input to the subsequent bond update $\mathrm { M L P } \phi _ { e }$ , and it is the composition that provides universality. Step 2: bond update approximates ${ \tilde { \rho } } _ { 3 }$ per summand. The bond update (SM2.2) for the directed edge $( j , i )$ is computed with $L _ { e } = I$

$$
e _ { j i } ^ { ( 1 ) } = e _ { j i } ^ { ( 0 ) } + \sum _ { k \in \mathcal { N } ( j ) \backslash \{ i \} } e _ { j i } ^ { ( 0 ) } \odot e _ { j k } ^ { ( 0 ) } \odot \phi _ { e } \big ( e _ { j k } ^ { ( 0 ) } , e _ { j i } ^ { ( 0 ) } , a _ { k j i } ^ { ( 0 ) } , n _ { j } ^ { ( 1 ) } \big ) .\tag{SM2.4}
$$

Substituting the expression for ${ n } _ { j } ^ { ( 1 ) }$ , we obtain that each summand has the form

$$
\begin{array} { r } { e _ { j i } ^ { ( 0 ) } \odot e _ { j k } ^ { ( 0 ) } \odot \phi _ { e } \Big ( \phi _ { d } ( d _ { j k } ) , ~ \phi _ { d } ( d _ { j i } ) , ~ { \bf a } _ { k j i } ^ { ( 0 ) } , ~ { \bf n } _ { j } ^ { ( 0 ) } + L _ { v } \left[ \sum _ { m } e _ { j m } ^ { ( 0 ) } \odot \phi _ { v } ( { \bf n } _ { m } ^ { ( 0 ) } , ~ { \bf n } _ { j } ^ { ( 0 ) } , ~ e _ { j m } ^ { ( 0 ) } ) \right] \Big ) . } \end{array}
$$

This is a function of the per-angle data $( d _ { j k } , d _ { j i } , \cos \theta _ { k j i } )$ and the multiset ${ \cal { S } } _ { j } ~ =$ $\{ ( \pmb { n } _ { m } ^ { ( 0 ) } , d _ { j m } ) \} _ { m \in \mathcal { N } ( j ) }$ . The dependence on $S _ { j }$ enters through $\pmb { n } _ { j } ^ { ( 0 ) } + L _ { v } [ \sum _ { m } g _ { m } ]$ , where $g _ { m } = e _ { j m } ^ { ( 0 ) } \odot \phi _ { v } ( { \pmb n } _ { m } ^ { ( 0 ) } , { \pmb n } _ { j } ^ { ( 0 ) } , e _ { j m } ^ { ( 0 ) } )$ . The composition $\begin{array} { r } { \phi _ { e } ( . . . , \ n _ { j } ^ { ( 0 ) } + L _ { v } [ \sum _ { m } g _ { m } ] ) } \end{array}$ has the extended DeepSets form. The inner function $g _ { m }$ is parametrized by an MLP $\phi _ { v } .$ With suficiently large feature dimension, the type embeddings are chosen supported on one coordinate block and the image of $L _ { v }$ on the complementary block, so the fourth argument of $\phi _ { e }$ carries the center feature ${ \pmb n } _ { j } ^ { ( 0 ) }$ and the aggregate $L _ { v } [ \sum _ { m } g _ { m } ]$ in disjoint blocks. By Lemma 4.2, there exist weights for $\phi _ { v }$ and $\phi _ { e }$ such that each summand approximates any continuous function of $( d _ { j k } , d _ { j i }$ , cos $\theta _ { k j i } )$ , the center feature, and the multiset $S _ { j }$ , symmetric in the multiset argument. The lemma applies in this parametrized form because the per-angle data and the center feature enter the outer MLP as additional inputs, and the approximation is uniform on their compact range.

The target per-angle function is $\tilde { \rho } _ { 3 } ( { \pmb n } _ { j } ^ { ( 0 ) } , { \pmb n } _ { i } ^ { ( 0 ) } , { \pmb n } _ { k } ^ { ( 0 ) } , d _ { j i } , d _ { j k } , \cos \theta _ { i j k } )$ , depending on specific entries ${ n } _ { i } ^ { ( 0 ) } , { n } _ { k } ^ { ( 0 ) }$ of the multiset. Under strong genericity (Definition 5.3), all neighbors of $j$ have distinct distances. Define the lookup function $\lambda : \mathcal { M } _ { \leq M } ( \mathbb { R } ^ { d } \times$ $\mathbb { R } _ { > 0 } ) \times \mathbb { R } _ { > 0 }  \mathbb { R } ^ { d }$ by $\lambda ( S _ { j } , d ) = n _ { m } ^ { ( 0 ) }$ , where m is the unique element of $\mathcal { N } ( j )$ satisfying $d _ { j m } = d .$ . On strongly generic configurations this is well-defined and symmetric in $S _ { j }$ On the compact set $K$ the distances of distinct neighbors are separated by a positive gap, so λ is continuous. Rewriting:

$$
\tilde { \rho } _ { 3 } ( { \pmb n } _ { j } ^ { ( 0 ) } , { \pmb n } _ { i } ^ { ( 0 ) } , { \pmb n } _ { k } ^ { ( 0 ) } , \ldots ) = \tilde { \rho } _ { 3 } \big ( { \pmb n } _ { j } ^ { ( 0 ) } , \lambda ( S _ { j } , d _ { j i } ) , \lambda ( S _ { j } , d _ { j k } ) , \ldots \big ) ,
$$

which is a continuous symmetric function of $S _ { j }$ and the per-angle data. The gate prefactors $e _ { j i } ^ { ( 0 ) } \odot e _ { j k } ^ { ( 0 ) } \odot ( \cdot )$ are absorbed by the gate convention fixed at the beginning of the proof, and $\check { \theta } _ { k j i } = \theta _ { i j k }$ . Therefore, each summand in (SM2.4) approximates ${ \tilde { \rho } } _ { 3 }$ Since the sum in (SM2.4) is a direct linear sum of per-angle terms, we obtain

$$
e _ { j i } ^ { ( 1 ) } \approx \phi _ { d } ( d _ { j i } ) + \sum _ { k \in \mathcal { N } ( j ) \backslash \{ i \} } \tilde { \rho } _ { 3 } ( n _ { j } ^ { ( 0 ) } , n _ { i } ^ { ( 0 ) } , n _ { k } ^ { ( 0 ) } , d _ { j i } , d _ { j k } , \cos \theta _ { i j k } ) .
$$

That is, $e _ { j i } ^ { ( 1 ) } \approx \phi _ { d } ( d _ { j i } ) + F _ { j i }$ , and the oppositely directed copy stores $e _ { i j } ^ { ( 1 ) } \approx \phi _ { d } ( d _ { j i } ) +$ $F _ { i j }$ . The summands are written into coordinates on which $\phi _ { d }$ vanishes, so $e _ { j i } ^ { ( 1 ) }$ carries $\phi _ { d } ( d _ { j i } )$ and $F _ { j i }$ in disjoint blocks.

Step 3: node update computes the HGNN double sum. The node update (SM2.1) with $L _ { v } = I$ computes

$$
{ \pmb n } _ { j } ^ { ( 2 ) } = { \pmb n } _ { j } ^ { ( 1 ) } + \sum _ { i \in \mathcal { N } ( j ) } { \pmb e } _ { j i } ^ { ( 0 ) } \odot \phi _ { v } \bigl ( { \pmb n } _ { i } ^ { ( 1 ) } , { \pmb n } _ { j } ^ { ( 1 ) } , { \pmb e } _ { j i } ^ { ( 1 ) } \bigr ) .
$$

Each summand receives $e _ { j i } ^ { ( 1 ) }$ , the copy directed away from $j .$ , which stores the inner sum $F _ { j i }$ by Step 2. By MLP universality, $\phi _ { v }$ can extract this inner sum from $e _ { j i } ^ { ( 1 ) }$ and multiply by $s _ { a } ( d _ { j i } )$ , where the distance $d _ { j i }$ is available through the retained block $\phi _ { d } ( d _ { j i } )$ of $e _ { j i } ^ { ( 1 ) }$ , and the gating factor $e _ { j i } ^ { ( 0 ) } \odot ( \cdot )$ is absorbed by the gate convention. The sum over i then gives

$$
{ n } _ { j } ^ { ( 2 ) } \approx \pmb { n } _ { j } ^ { ( 1 ) } + \sum _ { \stackrel { i , k \in \mathcal { N } ( j ) } { i \not = k } } s _ { a } ( d _ { j i } ) \tilde { \rho } _ { 3 } ( \pmb { n } _ { j } ^ { ( 0 ) } , \pmb { n } _ { i } ^ { ( 0 ) } , \pmb { n } _ { k } ^ { ( 0 ) } , d _ { j i } , d _ { j k } , \cos \theta _ { i j k } ) ,
$$

which is the HGNN double sum (2.17) without the outer readout $\psi _ { 3 }$

Step $\it 4 .$ absorbing $\psi _ { 3 }$ . The HGNN layer output is $\begin{array} { r } { \psi _ { 3 } \big ( \sum _ { i , k } s _ { a } \big ( d _ { j i } \big ) \tilde { \rho } _ { 3 } \big ) } \end{array}$ , but the CHGNet layer produces $\begin{array} { r } { \pmb { n } _ { j } ^ { ( 1 ) } + \sum _ { i , k } s _ { a } ( d _ { j i } ) \tilde { \rho } _ { 3 } } \end{array}$ . With suficiently large feature dimension, the increment of Step 3 is written into coordinates on which ${ n } _ { j } ^ { ( 1 ) }$ vanishes, so the double sum is recoverable from ${ n } _ { j } ^ { ( 2 ) }$ The missing nonlinearity $\psi _ { 3 }$ is handled diferently in intermediate and final layers. In an intermediate layer, the next CHGNet block begins with a node update (SM2.1), whose MLP $\phi _ { v } ( \pmb { \mathscr { n } } _ { m } ^ { ( 2 ) } , \pmb { \mathscr { n } } _ { j } ^ { ( 2 ) } , e _ { j m } ^ { ( 2 ) } )$ receives each neighbor’s double-sum feature and the center’s own. Being an MLP, it can apply ψ<sub>3</sub> to these inputs as part of its computation, so ${ n } _ { j } ^ { ( 3 ) }$ encodes ψ<sub>3</sub>-transformed neighbor information, and the subsequent bond update can compute the next layer’s ${ \tilde { \rho } } _ { 3 }$ from it by the extended DeepSets argument of Step 2. In the final layer, the readout network is an MLP applied after the last CHGNet layer, so it can approximate the composition of the extraction of the double sum, $\psi _ { 3 }$ , and the HGNN readout $f _ { \theta }$

Multi-layer extension. Each HGNN layer is simulated by two CHGNet layers. The first performs the node and bond updates of Steps 1–2, computing the per-angle ${ \tilde { \rho } } _ { 3 }$ and storing the inner sums in the edge features, absorbing the previous layer’s $\psi _ { 3 }$ when $l \geq 2$ as in Step 4. The second performs the node update of Step 3, which accumulates the double sum into the node features. The angle updates are set to approximate zero, so the angle features retain the initial encoding of cos $\theta _ { i j k }$ at every layer. The bond update of the second layer of each pair is not used and is suppressed by setting $L _ { e } ~ = ~ 0$ Each simulated layer realizes a map uniformly continuous on compact sets, so an input error entering a layer propagates to a controlled output error. Since there are finitely many layers, the per-layer simulation accuracies can be chosen so that the final node features are within any prescribed tolerance of the HGNN output on $K$ . Therefore, an L-layer HGNN is simulated by 2L CHGNet layers, with the final $\psi _ { 3 }$ absorbed into the readout, and the result follows from Theorem 5.1.

SM3. ALIGNN. ALIGNN [SM11] operates on two graphs: an atom graph $G ,$ whose nodes are atoms and whose edges are the bonds joining pairs of atoms within the cutof $r _ { c } ,$ , and its line graph $\mathcal { L } ( G ) . ^ { 4 }$ As in §SM2, edges are directed, and each bond $\{ j , i \}$ carries two features $e _ { j i }$ and $e _ { i j }$ , matching the reference implementation. The line graph, see Figure 1(c), is constructed by making each directed edge of G a node in ${ \mathcal { L } } ( G )$ , and connecting two nodes in ${ \mathcal { L } } ( G )$ if the head of the first edge is the tail of the second. The edges of ${ \mathcal { L } } ( G )$ thus correspond to bond angles. ALIGNN maintains three feature types: node features ${ \boldsymbol n } _ { j } ^ { l }$ , edge features $e _ { j i } ^ { l }$ on $G ,$ , and angle features $\pmb { a } _ { i j k } ^ { l }$ on ${ \mathcal { L } } ( G )$ . The initial node features are element embeddings, the initial edge features are radial basis function (RBF) expansions of bond distances, and the initial angle features are RBF expansions of bond-angle cosines.

Each ALIGNN layer performs two sequential edge-gated graph convolutions: first on ${ \mathcal { L } } ( G )$ to update the bond and angle features using angular information, then on $G$ to update the node and bond features using the enriched bonds. We write the two convolutions out in turn, since the update order determines which features reach the node aggregation.

1: Line graph convolution. The angle features are updated using the two adjacent

bond features, then the bond features are aggregated with angle-derived gates:

(SM3.1)

$$
\begin{array} { r } { \pmb { a } _ { i j k } ^ { l } = \pmb { a } _ { i j k } ^ { l - 1 } + \mathrm { S i L U } \big ( \mathrm { N o r m } ( A _ { a } \pmb { e } _ { j i } ^ { l - 1 } + B _ { a } \pmb { e } _ { j k } ^ { l - 1 } + C _ { a } \pmb { a } _ { i j k } ^ { l - 1 } ) \big ) , } \end{array}\tag{SM3.2}
$$

$$
\tilde { e } _ { j i } ^ { l } = e _ { j i } ^ { l - 1 } + \mathrm { S i L U } \Bigl ( \mathrm { N o r m } \bigl ( W _ { s , a } e _ { j i } ^ { l - 1 } + \sum _ { k \in \mathcal { N } ( j ) \backslash \{ i \} } \sigma _ { i j k } W _ { d , a } e _ { j k } ^ { l - 1 } \bigr ) \Bigr ) ,
$$

where $A _ { a } , B _ { a } , C _ { a } , W _ { s , a } , W _ { d , a }$ are trainable weight matrices, Norm is a feature normalization layer, σ denotes the sigmoid function, and $\mathrm { S i L U } ( x ) = x \sigma ( x )$ is the sigmoid linear unit. All edge features entering these updates are the copies directed away from the center $j .$ . Both convolutions use the same normalized gate: for features $\mathbf { \delta } _ { \mathbf { x } _ { s } }$ aggregated over an index set S,

$$
\sigma [ { \pmb x } _ { s } ] = \frac { \sigma ( { \pmb x } _ { s } ) } { \sum _ { s ^ { \prime } \in S } \sigma ( { \pmb x } _ { s ^ { \prime } } ) + \epsilon } ,\tag{SM3.3}
$$

with $\epsilon > 0$ a small constant for numerical stability. In (SM3.2) the gate is $\sigma _ { i j k } =$ $\sigma [ { a } _ { i j k } ^ { l } ]$ , aggregated over the angles at the bond $( j , i )$ . In (SM3.5) it is $\sigma _ { j i } = \sigma [ e _ { j i } ^ { l } ]$ aggregated over the neighbors of $j$

2: Atom graph convolution. The edge update loads both endpoint node features into the bond, then the node update aggregates over neighbors with edge-derived gates:

(SM3.4)

$$
e _ { j i } ^ { l } = \tilde { e } _ { j i } ^ { l } + \mathrm { S i L U } \big ( \mathrm { N o r m } ( A n _ { j } ^ { l - 1 } + B n _ { i } ^ { l - 1 } + C \tilde { e } _ { j i } ^ { l } ) \big ) ,\tag{SM3.5}
$$

$$
n _ { j } ^ { l } = n _ { j } ^ { l - 1 } + \mathrm { S i L U } \Bigl ( \operatorname { N o r m } \bigl ( W _ { s } { n } _ { j } ^ { l - 1 } + \sum _ { i \in \mathcal { N } ( j ) } \sigma _ { j i } W _ { d } { n } _ { i } ^ { l - 1 } \bigr ) \Bigr ) ,
$$

where $A , B , C , W _ { s } , W _ { d }$ are trainable weight matrices. The total energy is $E _ { \theta } ~ =$ $\begin{array} { r } { \sum _ { j } \mathrm { M L P } ( n _ { j } ^ { L } ) } \end{array}$

A key architectural feature is that the edge update (SM3.4) takes both endpoint node features $\mathbf { \nabla } n _ { i }$ and $\mathbf { \Delta } _ { n _ { j } }$ directly.

Remark SM3.1. ALIGNN shares the same angle → edge → node topology as DPA3 and CHGNet, and its edge update (SM3.4) incorporates both endpoint node features, analogous to DPA3’s self-message. However, the simulation proofs above rely on the message functions being MLPs, which are universal approximators. This is needed for the DeepSets universality argument and for the lookup decoding in the proof of Corollary 5.4. In ALIGNN, the message functions are single nonlinear layers. The inner functions in the aggregation are sigmoid-gated linear maps, whose only nonlinearity is the scalar gate, and the outer functions are SiLU ◦ Norm ◦ linear rather than MLPs. Consequently, Lemma 4.2 does not directly apply to a single ALIGNN layer, and the HGNN simulation argument does not carry over. Whether ALIGNN achieves UAT through suficient depth remains an open question. We note that replacing these gated linear messages with MLPs would remove this obstruction.