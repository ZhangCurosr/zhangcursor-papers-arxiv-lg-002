# OSCILLATORY NEURAL DYNAMICS OVER SHEAVES

Jan-Willem Van Looy<sup>1,∗</sup> Alessandro Trenta<sup>2,∗</sup> Alessio Gravina<sup>2,∗</sup> Alessio Borgi<sup>3,4∗</sup> Ferdinando Zanchetta<sup>1</sup> Pietro Lio\`<sup>4</sup> Davide Bacciu<sup>2</sup> Rita Fioresi<sup>1</sup>

<sup>1</sup>FaBiT, University of Bologna

<sup>2</sup>Department of Computer Science, University of Pisa

<sup>3</sup>Department of Computer, Control and Management Engineering, Sapienza University of Rome

<sup>4</sup>Department of Computer Science and Technology, University of Cambridge

## ABSTRACT

Effective long-range propagation remains a central challenge in graph neural networks, as increasing a model’s propagation depth does not guarantee that distant nodes effectively influence each other. Sheaf neural networks enrich graph propagation through matrix-valued transport between stalks; still, this expressivity alone does not automatically imply effective long-range communication. We introduce ONDA, a long-range graph learning framework based on operator-valued information waves. Stalk-valued representations evolve through second-order dynamics governed by learned sheaf transport operators, combining wave-like propagation with expressive local geometry. We characterize long-range influence through a stalk-wise sensitivity analysis and show that the cross-influence never vanishes. Across long-range propagation, severe graph bottlenecks, graph transfer, and heterophilic benchmarks, ONDA consistently improves over scalar wave propagation, diffusive sheaf baselines, and state-of-the-art models, demonstrating the benefit of coupling wave dynamics with matrix-valued transport.

## 1 INTRODUCTION

Effectively modeling long-range dependencies remains a central challenge in deep learning for graphs (Shi et al., 2023; Arroyo et al., 2025). Graph Neural Networks (GNNs) (Micheli, 2009; Scarselli et al., 2009; Defferrard et al., 2016; Gilmer et al., 2017; Kipf & Welling, 2017) usually rely on local message-passing to exchange information between nodes, meaning that dependencies between non-adjacent regions must be mediated through repeated propagation steps. However, while increasing depth expands the receptive field (i.e., the region from which information can be gathered), it does not guarantee that distant nodes communicate effectively. Indeed, challenges such as over-smoothing (Cai & Wang, 2020; Oono & Suzuki, 2020; Rusch et al., 2023), over-squashing (Alon & Yahav, 2021; Topping et al., 2022; Di Giovanni et al., 2023; Mishayev et al., 2025), and, more generally, vanishing gradients (Arroyo et al., 2025) prevent relevant information exchange between distant nodes. For this reason, learning how to propagate information effectively over long graph distances has become a critical problem in graph representation learning.

Existing approaches address this problem from several directions. Rewiring methods modify the graph to shorten communication paths (Topping et al., 2022; Gutteridge et al., 2023), while graph Transformers allow direct message passing between any pair of nodes via attention mechanisms (Shi et al., 2021a; Rampa´sek et al., 2022). A complementary line of work instead changes the dynamicsˇ of propagation while retaining sparse local interactions. In particular, differential-equation-inspired GNNs (DE-GNNs) have explored stable and non-dissipative dynamics as a way to maintain information over depth (Rusch et al., 2022; Gravina et al., 2023; Heilig et al., 2025; Hariri et al., 2025). Oscillatory DE-GNNs (Rusch et al., 2022; Trenta et al., 2025) follow this direction by modeling information propagation through second-order wave dynamics on the original graph.

While these approaches primarily focus on improving how far information can propagate, another important aspect is how information is exchanged between neighboring nodes. Many widely used GNNs generally propagate features through local aggregation operators based on scalar edge weights and shared feature transformations. Cellular sheaves provide a richer framework by associating local vector spaces (or stalks) with nodes and edges, together with linear restriction maps that relate node representations within a common edge space (Curry, 2014; Hansen & Ghrist, 2019; Hansen & Gebhart, 2020). Sheaf Neural Networks (SNNs) learn these maps, inducing matrix-valued couplings between neighboring node stalks and substantially enlarging the class of graph-supported propagation operators (Bodnar et al., 2022). This additional expressivity, however, does not by itself guarantee effective long-range communication between nodes. Indeed, neural sheaf models still rely on repeated dissipative local propagation, while information originating from distant regions must therefore traverse multiple successive sheaf transformations before reaching a target node. Consequently, maintaining informative signals over depth remains a largely underexplored challenge in SNNs, complementary to the problem of learning richer local transport. Recent works have started investigating how to extend neural sheaf architectures through mechanisms like polynomial filters (Borgi et al., 2025), large-scale diffusion (Bamberger et al., 2025) and directional routing (Ribeiro et al., 2026) to broaden the radius of information propagation across the graph.

Rather than extending the spatial range of individual updates or controlling which messages tra verse the graph, in this work, we seek to design a principled sheaf neural network whose dynamics enable useful stalk-valued information to remain active as it traverses the graph. This motivates ONDA (oscillatory neural dynamics on sheaves), a framework for long-range information propagation based on operator-valued information waves. In ONDA, stalk-valued representations evolve through an oscillatory second-order dynamical system governed by a learned sheaf Laplacian. The learned sheaf shapes the propagation of these information waves: its restriction maps determine how information is transformed between stalks and how different propagated contributions reinforce or cancel when they interact. In this sense, the learned sheaf acts as a propagation medium whose local properties govern how information waves travel across the graph. This couples the persistence of wave propagation with the expressive local geometry of cellular sheaves, allowing information to remain influential over distance while being non-trivially transformed along the graph.

Our contributions are threefold: (i) we introduce ONDA, a new class of sheaf neural networks based on second-order oscillatory dynamics, providing a non-trivial extension of wave-like graph propagation to matrix-valued sheaf operators; (ii) we theoretically investigate the long-range propagation properties of ONDA, showing that its oscillatory dynamics preserve the influence of distant information; (iii) we demonstrate strong and consistent empirical performance across a diverse suite of tasks, including long-range propagation, over-squashing mitigation, and heterophilic graph learning.

## 2 RELATED WORK

Long-Range Propagation on Graphs. Although stacking message-passing layers progressively enlarges the receptive field, information from distant nodes can rapidly lose influence due to phenomena such as over-smoothing (Cai & Wang, 2020; Oono & Suzuki, 2020; Rusch et al., 2023) and over-squashing (Alon & Yahav, 2021; Topping et al., 2022; Di Giovanni et al., 2023; Mishayev et al., 2025), both closely related to vanishing gradients (Arroyo et al., 2025). Various approaches aim to improve long-range information propagation. Graph rewiring methods (Topping et al., 2022; Gasteiger et al., 2019; Gutteridge et al., 2023; Black et al., 2023) modify the graph topology to create more favorable communication paths, whereas Graph Transformers (Kreuzer et al., 2021; Rampa´sekˇ et al., 2022; Shi et al., 2021a; Ying et al., 2021; Ma et al., 2023) use attention to learn direct interactions between nodes. Other methods modify the propagation dynamics through weight-space regularization (Gravina et al., 2023; 2025), port-Hamiltonian dynamics (Heilig et al., 2025), message filtering (Errica et al., 2024; Finkelshtein et al., 2024), or virtual nodes (Southern et al., 2025). Oscillatory GNNs (Rusch et al., 2022; Trenta et al., 2025) exploit second-order oscillatory propagation dynamics to facilitate communication between distant nodes, but operate within conventional GNN frameworks and therefore rely on standard graph-based interactions. In contrast, ONDA introduces oscillatory dynamics into sheaf neural networks, combining persistent wave-like propagation with learned matrix-valued transport operators that can control how information is transformed and transmitted along individual edges. Together, these mechanisms allow ONDA to combine effective long-range propagation with richer, geometry-aware information transport.

![](images/39db0e180db697f3c7174bcd7a6c317e73983b81abb5fd84588295372e1b0a43.jpg)  
Figure 1: Schematic view of ONDA on a single edge. Orange curves show the oscillatory evolution of node states in the stalks $\mathcal { F } ( u )$ and $\mathcal { F } ( v )$ ; learned restriction maps express them in the shared edge stalk $\mathcal { F } ( e )$ (dashed orange). The blue curve shows their superposition, highlighting constructive and destructive interference between the mapped signals.

Sheaf Neural Networks. Cellular sheaves are used to extend graph-based learning by associating vector spaces with graph cells and linear restriction maps with their incidences, giving rise to matrix-valued generalizations of classical graph operators (Hansen & Gebhart, 2020). Building on this framework, Bodnar et al. (2022) introduced learnable sheaf structures into GNNs, showing that the resulting diffusion can provide favourable properties in heterophilic settings and alleviate over-smoothing. Subsequent works have explored different ways of parametrizing and learning the underlying sheaf, including connection-Laplacian constructions (Barbero et al., 2022), polynomial filters (Borgi et al., 2025), inductive biases inspired by joint diffusion processes (Hernandez Caralt et al., 2024), and adaptive nonlinear sheaf Laplacians (Zaghen et al., 2024). More recently, Bamberger et al. (2025) proposed using continuous diffusion over flat vector bundles to operate over larger spatial scales, while other approaches consider directional routing (Ribeiro et al., 2026) specifically designed to better exploit deep sheaf propagation. Nevertheless, most existing SNNs remain centered around diffusion-like dynamics, which are inherently dissipative and struggle to capture long-range interactions. In contrast, ONDA introduces second-order oscillatory dynamics into sheaf neural networks, interpreting learned matrix-valued restriction maps as the wave propagation medium whose local properties govern how information waves travel across the graph.

## 3 ONDA

We introduce Oscillatory Neural Dynamics on Sheaves (ONDA), a new class of sheaf neural networks driven by second-order oscillatory dynamics, allowing stalk-valued information to remain relevant as it traverses the graph. A schematic illustration of ONDA is provided in Figure 1.

## 3.1 NOTATION AND PRELIMINARIES

Let $\mathcal { G } ~ = ~ ( \nu , \mathcal { E } )$ be an undirected graph, where V is the finite set of n nodes (or vertices) and $\mathcal { E } \subseteq \mathcal { V } \times \dot { \mathcal { V } }$ the set of edges. Given two nodes $u , v \in \mathcal { V }$ we denote an edge between them with the unordered pair {u, v}. Graph connectivity is encoded by the adjacency matrix $\ b { A } \in \mathbb { R } ^ { n \times n }$ with degree matrix $\bar { \boldsymbol { D } } \in \mathbb { R } ^ { n \times n }$ defined by $\begin{array} { r } { \dot { D } _ { i i } = \sum _ { i = 1 } ^ { n } \dot { A _ { i j } } } \end{array}$ . A cellular sheaf on G (Shepard, 1985; Curry, 2014) assigns a finite-dimensional vector space $\mathcal { F } ( v )$ to each vertex $v \in \mathcal V$ , a finitedimensional vector space $\mathcal { F } ( e )$ to each edge $e \in { \mathcal { E } }$ , and a linear restriction map $\mathcal { F } _ { v , e } : \mathcal { F } ( v )  \mathcal { F } ( e )$ to each incidence $v \in e .$ . We refer to $\mathcal { F } ( v )$ and $\mathcal { F } ( e )$ as the vertex and edge stalks, respectively. In this paper, all stalks share a common dimension $d \geq 1$ . The space of 0-cochains, or stalk-valued node states is $C ^ { 0 } ( \mathcal { G } ; \mathcal { F } ) : = \bigoplus _ { v \in \mathcal { V } } \mathcal { F } ( v )$ . We write $\pmb { X } = ( \pmb { X } _ { v } ) _ { v \in \mathcal { V } } \in C ^ { 0 } ( \mathcal { G } ; \mathcal { F } )$ , where $\boldsymbol { X _ { v } } \in \mathcal { F } ( \boldsymbol { v } )$ For c feature channels, node features lie in $C ^ { 0 } ( \mathcal { G } ; \mathcal { F } ) \otimes \mathbb { R } ^ { c }$ . The SheafLaplacian (Hansen & Ghrist, 2019; Hansen, 2020) $L _ { \mathcal { F } } : C ^ { 0 } ( \mathcal { G } ; \mathcal { F } ) \to C ^ { 0 } ( \dot { \mathcal { G } } ; \mathcal { F } )$ acts nodewise as:

$$
( L _ { \mathcal { F } } \pmb { X } ) _ { v } = \sum _ { e = \{ v , u \} \in \mathcal { E } } \mathcal { F } _ { v , e } ^ { * } \big ( \mathcal { F } _ { v , e } \pmb { X } _ { v } - \mathcal { F } _ { u , e } \pmb { X } _ { u } \big )\tag{1}
$$

where each stalk is equipped with an inner product and $\mathcal { F } _ { v , e } ^ { * }$ denotes the adjoint of $\mathcal { F } _ { v , e }$ . The resulting sheaf Laplacian is self-adjoint and positive semidefinite (Hansen & Ghrist, 2019). Along each edge, the restriction maps place neighbouring node representations in a common space, where their discrepancy is computed; the adjoint then maps this discrepancy back to the incident node, where it is aggregated with the remaining neighbour contributions. For $d = 1$ , Euclidean inner products, and identity restriction maps, $L _ { \mathcal { F } }$ reduces to the standard graph Laplacian $\pmb { L } _ { \mathcal { G } } = \pmb { D } - \pmb { A }$ (Brouwer & Haemers, 2012). We also use the symmetrically normalized sheaf Laplacian $\Delta _ { \mathcal { F } } : =$ $\mathcal { D } _ { \mathcal { F } } ^ { - 1 / 2 } L _ { \mathcal { F } } \mathcal { D } _ { \mathcal { F } } ^ { - 1 / 2 }$ , where $\mathcal { D } _ { \mathcal { F } }$ is the block diagonal part of $L _ { \mathcal { F } }$ (Hansen & Ghrist, 2019).

Given its oscillatory dynamics, ONDA is inherently a differential-equation-based architecture, where node representations are interpreted as the evolving states of a dynamical system defined over the graph (Poli et al., 2019; Gravina et al., 2023; Han et al., 2024). Under this view, we denote the collection of all node states at time t as $\pmb { X } ( t ) \in C ^ { 0 } ( \mathcal { G } ; \mathcal { F } ) \otimes \mathbb { R } ^ { c }$ , stacked as an nd × c matrix. As in DE-GNNs, the continuous dynamics are instantiated as a neural architecture through numerical discretization (Gravina et al., 2023). Therefore, given an initial condition $\pmb { X } ( 0 ) = \overline { { \pmb { X } } }$ and a step size $h > 0 ,$ , we denote by $X ^ { ( \ell ) }$ the node features after ℓ integration steps, initialized with $\mathbf { X } ^ { ( 0 ) } = \overline { { \mathbf { X } } }$ Thus, $X ^ { ( \ell ) } \approx X ( t _ { \ell } )$ , where $t _ { \ell } = h \ell$ and the parenthesized superscript indexes the discrete steps.

## 3.2 OSCILLATORY NEURAL DYNAMICS ON SHEAVES

Neural sheaf models replace scalar graph interactions with matrix-valued transport between local stalks, transforming information as it crosses each edge. Yet, expressive transport alone does not guarantee that information remains influential over long propagation ranges. ONDA addresses both aspects jointly: the learned sheaf controls how information moves between neighbouring stalks, while second-order oscillatory dynamics govern how it evolves with depth.

Specifically, we model the evolution of the stalk-valued representations $\pmb { X } ( t ) \in C ^ { 0 } ( \mathcal { G } ; \mathcal { F } ) \otimes \mathbb { R } ^ { c }$ through the following second-order differential equation

$\ddot { X } ( t ) = - L _ { \mathcal { F } } ( I _ { n } \otimes W _ { 1 } ) X ( t ) W _ { 2 } - R _ { \theta } ( X ( t ) ) \odot \dot { X } ( t ) + F _ { \theta } ( X ( t ) ) , \quad X ( 0 ) = \overline { { X } }$ (2) where $L _ { \mathcal { F } }$ is the sheaf Laplacian from Equation $( 1 ) , I _ { n } \in \mathbb { R } ^ { n \times n }$ is the identity matrix, $W _ { 1 } \in \mathbb { R } ^ { d \times d }$ is the weight matrix acting on stalk coordinates, and $W _ { 2 } \in \mathbb { R } ^ { c \times c }$ transforms feature channels. The terms $R _ { \theta } ( \mathbf { \bar { X } } ( t ) ) \in \mathbb { R } _ { > 0 } ^ { n d \times c }$ and $F _ { \theta } ( X ( t ) ) \in \mathbb { R } ^ { n d \times c }$ provide state-dependent dissipation and external forcing, respectively, while ⊙ denotes the Hadamard product.

In Equation (2), the sheaf operator shapes the information waves: restriction maps determine how signals transform across edges and how incoming wave contributions combine within each stalk. This mirrors classical wave equations (Evans, 2010), where the spatial Laplacian operator encoded the propagation properties of the underlying medium. ONDA therefore couples the propagation bias of wave dynamics with learned matrix-valued transport.

With zero dissipation and forcing and identity weight matrices, Equation (2) reduces to a cellularsheaf generalization of the graph wave equation $\ddot { X } ( t ) = - L _ { \mathcal { G } } X ( t )$ . Here, $\scriptstyle { L _ { \mathcal { G } } }$ is the graph Laplacian which governs spatial coupling in the graph wave equation and induces conservative oscillatory dynamics.

Purely conservative wave dynamics can nevertheless be restrictive for downstream learning, because not all information should necessarily be preserved indefinitely. We therefore augment the oscillatory term with the state-dependent dissipation $R _ { \theta } ( { \mathbf { } } X ( t ) )$ and external forcing $\bar { F } _ { \theta } ( \pmb { X } ( t ) )$ . Dissipation adaptively attenuates propagated information, while forcing supplies an additional learned drive. Together, they let ONDA balance conservative and non-conservative behavior and adapt its propagation dynamics to the task at hand.

Discretization and Implementation Details. We obtain a neural architecture from Equation (2) by numerically discretizing its continuous dynamics. Introducing the velocity $V ( t ) = \dot { X } ( t )$ and applying a finite-difference scheme (Ferziger & Peric, 2001) yields the update

$$
\begin{array} { r l } & { { \pmb V } ^ { ( \ell + 1 ) } = { \pmb V } ^ { ( \ell ) } - h L _ { { \mathcal F } } \big ( ( I _ { n } \otimes { \pmb W } _ { 1 } ) { \pmb X } ^ { ( \ell ) } { \pmb W } _ { 2 } \big ) + h F _ { \theta } \big ( { \pmb X } ^ { ( \ell ) } \big ) - h R _ { \theta } \big ( { \pmb X } ^ { ( \ell ) } \big ) \odot { \pmb V } ^ { ( \ell ) } , } \\ & { { \pmb X } ^ { ( \ell + 1 ) } = { \pmb X } ^ { ( \ell ) } + h { \pmb V } ^ { ( \ell + 1 ) } } \end{array}\tag{3}
$$

where $h > 0$ is the integration step size, and $\ell = 0 , \ldots , \ell - 1$

To increase expressivity, we stack multiple instances of the propagation dynamics in Equation (3) and interleave them with nonlinear feature transformations, following the general principle of separating recurrent propagation from channel-wise nonlinear mixing (Gu et al., 2021; Orvieto et al.,

2023; Ceni et al., 2025; Trenta et al., 2025). A block consists of $\mathcal { L }$ propagation steps from Equation (3), followed by an MLP transformation. In block b, the features after $\mathcal { L }$ propagation steps initialize block (b + 1) through:

$$
\begin{array} { r } { \pmb { X } ^ { ( b + 1 , 0 ) } = \mathrm { M L P } _ { b } ( \pmb { X } ^ { ( b , \mathcal { L } ) } ) , \qquad b = 0 , \ldots , B - 1 } \end{array}\tag{4}
$$

The velocity of the new block is again initialized from its input representation and the final block output is passed to a task-specific decoder. Propagation within each block can be either fixed or adaptive. In the fixed setting, restriction maps are inferred once from the block input and held constant for all $\mathcal { L }$ steps: each block first learns the sheaf governing transport, then evolves node representations according to Equation (3). In the adaptive setting, restriction maps are recomputed at each step from the current features, allowing the evolving node states to reshape the propagation medium. The number of blocks B, the number of propagation steps ${ \mathcal { L } } ,$ and the integration step size h are treated as architectural hyperparameters.

Following standard SNN practice (Bodnar et al., 2022; Bamberger et al., 2025; Ribeiro et al., 2026), we parametrize restriction maps with neural networks. We consider both symmetrically normalized sheaf Laplacians and a directed variant. In the latter, restriction maps are learned independently for each edge orientation, and each discrepancy is applied only at the receiving endpoint. For simplicity, we retain the notation $L _ { \mathcal { F } }$ . Further details on sheaf normalization and the directed variant are provided in Appendix A.2; the restriction-map parametrizations are described in Appendix B.1.

## 4 THEORETICAL ANALYSIS

In this section, we study the evolution of node states in a single ONDA block, with particular focus on providing theoretical grounding for long-range influence across nodes. We start from the continuous definition of the model in Equation (2) without dissipation and external forcing, showing that the oscillatory dynamics provide explicit guarantees on the cross-influence between nodes. In Theorem 1, we translate these results to the discrete case.

To isolate the benefits of the oscillatory dynamics, we fix the restriction maps within one block and set the weights to the identity. For simplicity, we also work with a single feature channel and write $\mathcal { H } : = C ^ { 0 } ( \mathcal { G } ; \mathcal { F } ) = \bigoplus _ { \mathfrak { v } \in \mathcal { V } } \mathcal { F } ( \mathfrak { v } ) \cong \dot { \mathbb { R } } ^ { n d }$ for the feature space. With a fixed sheaf Laplacian $L _ { \mathcal { F } } : \mathcal { H } \to \mathcal { H }$ , the conservative form of ONDA is given by:

$$
\ddot { X } ( t ) + L _ { \mathcal { F } } X ( t ) = 0 , \qquad X ( 0 ) = \overline { { X } } , \qquad \dot { X } ( 0 ) = \overline { { V } }\tag{5}
$$

where $\overline { { \boldsymbol { X } } }$ and $\overline { V }$ are respectively the initial node features and their velocities. We now define the equivalent of the sine and cosine functions for the sheaf Laplacian through the corresponding power series expansion (Higham, 2008):

$$
\mathcal { C } _ { \mathcal { F } } ( t ) = \sum _ { k = 0 } ^ { \infty } ( - 1 ) ^ { k } \frac { t ^ { 2 k } } { ( 2 k ) ! } L _ { \mathcal { F } } ^ { k } , \qquad \mathcal { S } _ { \mathcal { F } } ( t ) = \sum _ { k = 0 } ^ { \infty } ( - 1 ) ^ { k } \frac { t ^ { 2 k + 1 } } { ( 2 k + 1 ) ! } L _ { \mathcal { F } } ^ { k }\tag{6}
$$

Analogously to the wave equation in physics (Evans, 2010), the general solution to Equation (5) is given by $\mathbf { } X ( t ) = \mathcal { C } _ { \mathcal { F } } ( t ) \overline { { \mathbf { } X } } + \mathcal { S } _ { \mathcal { F } } ( t ) \overline { { \mathbf { } V } }$ , as shown in Appendix A.3. This describes the node features at time t through a combination of the sine and cosine operators.

We now seek to measure how different nodes influence each other through the common framework of sensitivity analysis (Di Giovanni et al., 2023). Concretely, we measure how changes in the features of node u affect a target node v. For each $u \in \mathcal V .$ , we define $\iota _ { u } : \mathcal { F } ( u )  \mathcal { H }$ and $\pi _ { u } : \mathcal { H } \to \mathcal { F } ( u )$ as the inclusion and projection operators, respectively. We define $\dot { W } \in \mathcal { F } ( u )$ as a perturbation of the initial state at node $u .$ Thus, the perturbed initial condition of the whole graph is $\overline { { \boldsymbol { X } } } + \iota _ { u } \boldsymbol { W }$ , where the components relative to u are shifted by a vector $W$ . Given the closed-form solution of Equation (5), we can calculate the difference between the perturbed and unperturbed trajectories as:

$$
\Delta \boldsymbol { X } ( t ) = ( \mathcal { C } _ { \mathcal { F } } ( t ) ( \overline { { \boldsymbol { X } } } + \iota _ { u } \boldsymbol { W } ) + \mathcal { S } _ { \mathcal { F } } ( t ) \overline { { \boldsymbol { V } } } ) - ( \mathcal { C } _ { \mathcal { F } } ( t ) \overline { { \boldsymbol { X } } } + \mathcal { S } _ { \mathcal { F } } ( t ) \overline { { \boldsymbol { V } } } ) = \mathcal { C } _ { \mathcal { F } } ( t ) \iota _ { u } \boldsymbol { W }\tag{7}
$$

This equation already shows that changes in the stalk of a single node u affect the whole trajectory indefinitely, as $\mathscr { C } _ { \mathcal { F } } ( t )$ does not vanish in the limit of $t  \infty$ . We refer to the evolving response

$\Delta X ( t )$ generated by a perturbation at node u as its information wave. Projecting this wave onto a target node v gives $\Delta \bar { X } _ { v } ( t ) = \pi _ { v } \mathcal { C } _ { \mathcal { F } } ( t ) \iota _ { u } W$ . Independent of the particular perturbation $W$ , we therefore define the source-to-target influence at time t as:

$$
J _ { v  u } ^ { \mathcal { F } } ( t ) : = \pi _ { v } { \mathcal { C } } _ { \mathcal { F } } ( t ) \iota _ { u } = ( { \mathcal { C } } _ { \mathcal { F } } ( t ) ) _ { v u } : { \mathcal { F } } ( u ) \to { \mathcal { F } } ( v )\tag{8}
$$

In contrast to the scalar case, the map $J _ { v  u } ^ { \mathcal { F } } ( t )$ is a matrix-valued transformation between the source and target stalks, describing both the strength of the interaction and how directions in $\mathcal { F } ( u )$ are transformed in $\mathcal { F } ( v )$ . The fact that the operators $\mathcal { C } _ { \mathcal { F } } , \iota _ { u } , \pi _ { \iota }$ are all linear allows for the linear superposition of waves and their effects, similar to how physical waves linearly compose and decompose (Evans, 2010). In the small perturbation limit, the source-to-target response in Equation (8) can be thought of as the equivalent of the Jacobian matrix in sensitivity analysis on standard GNNs (Di Giovanni et al., 2023) to the sheaf case.

To understand whether the non-vanishing influence of node u on the whole trajectory enables longrange propagation, we first examine the behaviour of Equation (8) depending on the target node v in terms of the graph structure. Since $( L _ { \mathcal { F } } ) _ { v u } = 0$ whenever distinct nodes $u , v$ are not adjacent, applying $L _ { \mathcal { F } }$ once only couples the stalk of node u to itself or adjacent ones. Subsequent powers $L _ { \mathcal { F } } ^ { k }$ in the definition of the cosine operator $\mathcal { C } _ { \mathcal { F } }$ can couple only nodes connected by a path of length at most k. For $k \geq 0 .$ , we formalise this idea (see also Appendix A.4) by defining $\mathcal { W } _ { k } ( u , v )$ as the set of length-k walks $\omega = ( v _ { 0 } = u , v _ { 1 } , \ldots , v _ { k } = v )$ , where each step either follows an edge or remains at the current node. From this we can construct an operator $\bar { ( L _ { \mathcal { F } } ) } _ { \omega }$ as the ordered composition of Laplacian blocks along $\omega ,$ that is, $( L _ { \mathcal { F } } ) _ { \omega } : = ( L _ { \mathcal { F } } ) _ { v _ { k } v _ { k - 1 } } \cdot \cdot \cdot ( \dot { L } _ { \mathcal { F } } ) _ { v _ { 1 } v _ { 0 } } : \mathcal { F } ( u )  \mathcal { F } ( v )$

For $k = 0$ , we define the empty product as the identity on $\mathcal { F } ( u )$ . In particular, this operator describes the stalk-to-stalk transformation accumulated along the walk ω and lets us decompose the sourceto-target response across all different walks that connect u to v.

Proposition 1 (Pathwise source-to-target response). Let $u , v \in \mathcal { V }$ be two nodes connected by a shortest path oflength r. Then:

$$
J _ { v  u } ^ { \mathcal { F } } ( t ) = \sum _ { k = r } ^ { \infty } ( - 1 ) ^ { k } \frac { t ^ { 2 k } } { ( 2 k ) ! } \sum _ { \omega \in \mathcal { W } _ { k } ( u , v ) } ( L _ { \mathcal { F } } ) _ { \omega }\tag{9}
$$

Ifthere is no path connecting u and v in the graph, then $J _ { v  u } ^ { \mathcal { F } } ( t ) = 0 f o r e \nu e r y t .$

The proof is provided in Appendix A.4. Thus, if u and v are r edges apart, the first possible contribution comes from shortest paths of length r, with longer walks contributing through higher powers of $L _ { \mathcal { F } }$ . A key observation is that the propagation time t weighs the contribution of different walks through the coefficients $\frac { ( - 1 ) ^ { k } t ^ { 2 k } } { ( 2 k ) ! }$ . Hence, the contribution of walks of different lengths to the source-to-target response depends on the propagation time t, and graph distance does not prevent communication between connected nodes. At early times, shorter walks dominate the response, while increasing t allows contributions from progressively longer walks to become relevant. Intuitively, the longer the wave evolves, the more opportunities it has to propagate through the graph and influence the target through different routes.

While Proposition 1 describes how the source-to-target response is assembled from local interactions, we now describe its evolution as time varies. For the self-adjoint sheaf Laplacian, this temporal behaviour can be characterised through its spectral theory (Hansen $\&$ Ghrist, 2019; Hansen, 2020). By the finite-dimensional spectral theorem (Halmos, 1974), we can decompose $\begin{array} { r } { L _ { \mathcal { F } } = \sum _ { \lambda } \lambda P _ { \lambda } } \end{array}$ where the sum runs over distinct eigenvalues $\lambda \geq 0$ and $P _ { \lambda }$ is the orthogonal projector onto the corresponding eigenspace. As before, we write $( P _ { \lambda } ) _ { v u } : = \pi _ { v } P _ { \lambda } \iota _ { u } : { \mathcal { F } } ( u ) \to { \mathcal { F } } ( v )$ for the associated source-to-target block.

Proposition 2 (Spectral source-to-target response). For the sheaf Laplacian $L _ { \mathcal { F } }$ , the source-totarget sensitivity satisfies:

$$
J _ { v  u } ^ { \mathcal { F } } ( t ) = ( P _ { 0 } ) _ { v u } + \sum _ { \lambda > 0 } \cos ( t \sqrt { \lambda } ) ( P _ { \lambda } ) _ { v u }\tag{10}
$$

The proof is provided in Appendix A.5. This statement makes the oscillatory nature of the response explicit. Each block $( P _ { \lambda } ) _ { v u }$ gives the contribution of the corresponding wave frequency and eigenspace, while the factor $\cos ( t \sqrt \lambda )$ determines how that contribution changes over time. The term $( P _ { 0 } ) _ { v u }$ remains constant, whereas the positive-eigenvalue contributions oscillate. Waves at different frequencies can reinforce or cancel, so a non-trivial source-to-target response can be weak or strong depending on time, but cannot decay to zero. This fundamentally distinguishes ONDA from sheaf models based on heat-flow dynamics, where positive-eigenvalue components decay exponentially (Bodnar et al., 2022; Bamberger et al., 2025).

The continuous-time analysis shows that injecting oscillatory biases into feature dynamics sustains long-range influence on sheaves. We now explain how the above results directly translate into the discrete updates from Equation (3), under the same assumptions of no dissipation or external forcing. Starting from the same perturbation $W \in { \mathcal { F } } ( u )$ at the source node $u ,$ let:

$$
\Delta \boldsymbol { X } ^ { ( \ell ) } : = \boldsymbol { X } ^ { ( \ell ) } ( \overline { { \boldsymbol { X } } } + \iota _ { u } \boldsymbol { W } ) - \boldsymbol { X } ^ { ( \ell ) } ( \overline { { \boldsymbol { X } } } )\tag{11}
$$

denote the resulting change in the node features after ℓ discrete updates. Its component at a target node v is $\Delta X _ { v , \ell } : = \pi _ { v } \Delta X ^ { ( \ell ) } \in \mathcal { F } ( v )$ . As in the continuous case, we define the discrete source-totarget sensitivity $J _ { v  u } ^ { \mathcal { F } } [ \ell , h ]$ by the linear map satisfying $\begin{array} { r } { \Delta { X } _ { v , \ell } = J _ { v  u } ^ { \mathcal { F } } [ \ell , h ] W } \end{array}$ . Thus, $J _ { v  u } ^ { \mathcal { F } } [ \ell , h ]$ describes how a perturbation in the source stalk is transformed into a response at the target after ℓ discrete updates with step size h. The following result gives this discrete source-to-target sensitivity a closed-form expression in terms of the spectral projectors, under a mild stability condition. In particular, it applies to the symmetrically normalized sheaf Laplacian $\Delta _ { \mathcal { F } }$ for $0 < h < { \sqrt { 2 } }$ , since its spectrum is contained in [0, 2], as shown in Appendix A.2.

Theorem 1 (Discrete source-to-target response). Suppose $L _ { \mathcal { F } }$ is self-adjoint and positive semidefinite and $h ^ { 2 } \lambda _ { \operatorname* { m a x } } ( L _ { \mathcal { F } } ) < 4$ . Then,for every integer $\ell \geq 0 .$

$$
J _ { v  u } ^ { \mathcal { F } } [ \ell , h ] = \sum _ { \lambda } \frac { \cos \bigl ( ( \ell + \frac { 1 } { 2 } ) \theta _ { h } ( \lambda ) \bigr ) } { \cos \bigl ( \theta _ { h } ( \lambda ) / 2 \bigr ) } ( P _ { \lambda } ) _ { v u }\tag{12}
$$

where $\theta _ { h } ( \lambda ) = 2$ arcsin $\left( { \frac { h { \sqrt { \lambda } } } { 2 } } \right)$ . Moreover, $J _ { v  u } ^ { \mathcal { F } } [ \ell , h ] = 0$ whenever there is no path from u to v of length at most ℓ. Finally, for each node pair $u , v ,$ the response is bounded and, if it is non-zero at some step, it does not converge to zero as $\ell \to \infty$

The proof is provided in Appendix A.6. Thus, both features and properties on long-range propagation and non-vanishing of responses we discussed previously still hold for the discretized dynamics that define ONDA. First, the local nature of message-passing updates enables communication between nodes at distance r only after $\ell \geq r$ iterations. At the same time, each positive-eigenvalue contribution remains oscillatory across depth and time, and does not vanish, making the discrete dynamics consistent with the continuous ones. Indeed, as shown in Appendix $_ { \mathrm { A . 6 , } }$ for each fixed eigenvalue $\lambda ,$ the coefficient multiplying $( P _ { \lambda } ) _ { v u }$ in Equation (12) converges to cos $( T \sqrt { \lambda } )$ as $\ell \to \infty$ and $h  0$ with $\ell h = T$ fixed. Taking this limit in the spectral expansion therefore recovers the continuous-time expression at time T. Thus, for sufficiently small step sizes, ONDA inherits the persistent oscillatory behaviour of the continuous dynamics, leveraging matrix-valued sheaf operators for long-range propagation.

## 5 EXPERIMENTS

We evaluate ONDA across complementary settings that isolate distinct aspects of effective information propagation on graphs. Section 5.1 examines long-range propagation across structurally challenging topologies with severe information bottlenecks. Section 5.2 provides a controlled test of short-range over-squashing, requiring densely connected regions to exchange aggregated information through a narrow communication channel. Section 5.3 evaluates transfer across varying sourceto-target distances and graph topologies. We further test ONDA beyond explicitly propagationoriented tasks on heterophilic benchmarks in Appendix C.2. Appendix B details the sheaf learners and hyperparameter search space, Appendix C.1 reports an ablation of the restriction-map parametrization, and Appendix D analyzes computational cost and runtime.

## 5.1 LONG-RANGE PROPAGATION: ECHO BENCHMARK

We first evaluate ONDA on ECHO (Miglior et al., 2026), a benchmark specifically designed to assess long-range propagation capabilities. Its synthetic suite, ECHO-Synth, comprises single-source shortest path (SSSP), node eccentricity, and graph diameter prediction, tasks whose solutions depend on information travelling across distant regions of the graph. Models with limited propagation range therefore incur high error. We follow data splits, hyperparameters search space, and experimental protocol of (Miglior et al., 2026).

Table 1: ECHO-Synth benchmark results. Performance comparison on MAE (↓). Baseline values are reported directly from Miglior et al. (2026). For each column, the best, second-best, and thirdbest results are highlighted.
<table><tr><td>Model</td><td>sssp ↓</td><td>ecc ↓</td><td>diam ↓</td></tr><tr><td>A-DGN</td><td> $1 . 1 7 6 { \scriptstyle \pm 0 . 1 4 0 }$ </td><td> $4 . 9 8 1 { \scriptstyle \pm 0 . 0 3 7 }$ </td><td> $\mathbf { 1 . 1 5 1 { \scriptstyle \pm 0 . 0 3 8 } }$ </td></tr><tr><td>DRew</td><td> $1 . 2 7 9 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $4 . 6 5 1 { \scriptstyle \pm 0 . 0 2 0 }$ </td><td> $1 . 2 4 3 _ { \pm 0 . 0 4 7 }$ </td></tr><tr><td>GCN</td><td> $2 . 1 0 2 _ { \pm 0 . 0 9 4 }$ </td><td> $5 . 2 3 3 { \scriptstyle \pm 0 . 0 3 4 }$ </td><td> $3 . 8 3 2 _ { \pm 0 . 2 6 2 }$ </td></tr><tr><td>GCNII</td><td> $2 . 1 2 8 { \scriptstyle \pm 0 . 4 2 9 }$ </td><td> $5 . 2 4 1 { \scriptstyle \pm 0 . 0 3 0 }$ </td><td> $2 . 0 0 5 { \scriptstyle \pm 0 . 0 9 3 }$ </td></tr><tr><td>GIN/GINE</td><td> $2 . 2 3 4 { \scriptstyle \pm 0 . 2 7 1 }$ </td><td> $4 . 8 6 9 _ { \pm 0 . 0 9 2 }$ </td><td> $1 . 6 3 0 { \scriptstyle \pm 0 . 1 6 1 }$ </td></tr><tr><td>GPS</td><td> $0 . 4 7 2 _ { \pm 0 . 0 5 0 }$ </td><td> $4 . 7 5 8 { \scriptstyle \pm 0 . 0 2 1 }$ </td><td> $2 . 1 6 0 _ { \pm 0 . 0 9 8 }$ </td></tr><tr><td>GraphCON</td><td> $5 . 7 3 4 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $5 . 4 7 4 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $2 . 9 6 9 _ { \pm 0 . 1 8 9 }$ </td></tr><tr><td>GRIT</td><td> $0 . 1 2 1 { \scriptstyle \pm 0 . 0 1 3 }$ </td><td> $5 . 0 9 1 _ { \pm 0 . 1 5 8 }$ </td><td> $1 . 0 1 4 { \scriptstyle \pm 0 . 0 4 6 }$ </td></tr><tr><td>PH-DGN</td><td> $1 . 3 2 3 { \scriptstyle \pm 0 . 4 8 5 }$ </td><td> $5 . 0 6 8 _ { \pm 0 . 1 2 6 }$ </td><td> $1 . 6 2 7 _ { \pm 0 . 3 9 8 }$ </td></tr><tr><td>SWAN</td><td> $0 . 8 9 6 _ { \pm 0 . 2 3 2 }$ </td><td> $4 . 8 4 0 { \scriptstyle \pm 0 . 0 4 5 }$ </td><td> $1 . 1 2 1 { \scriptstyle \pm 0 . 0 7 0 }$ </td></tr><tr><td>SONAR</td><td> $0 . 5 2 7 { \scriptstyle \pm 0 . 2 5 4 }$ </td><td> $2 . 3 9 3 { \scriptstyle \pm 0 . 8 0 8 }$ </td><td> $1 . 2 3 9 { \scriptstyle \pm 0 . 1 5 1 }$ </td></tr><tr><td>NSD</td><td> $\mathbf { 0 . 2 3 1 { \scriptstyle \pm 0 . 0 3 8 } }$ </td><td> $4 . 6 7 2 _ { \pm 0 . 3 1 5 }$ </td><td> $1 . 4 4 8 _ { \pm 0 . 0 9 7 }$ </td></tr><tr><td>CSNN</td><td> $0 . 4 2 1 { \scriptstyle \pm 0 . 1 6 5 }$ </td><td> $\mathbf { 4 . 5 2 2 { \scriptstyle \pm 0 . 0 4 8 } }$ </td><td> $\mathbf { 1 . 1 3 4 { \scriptstyle \pm 0 . 0 9 4 } }$ </td></tr><tr><td>ONDA</td><td> $0 . 0 8 5 { \scriptstyle \pm 0 . 0 1 9 }$ </td><td> $0 . 8 8 5 { \scriptstyle \pm 0 . 1 5 8 }$ </td><td> $1 . 1 8 3 _ { \pm 0 . 0 2 0 }$ </td></tr></table>

Results. As shown in Table 1, ONDA achieves the lowest error on both SSSP and eccentricity, substantially outperforming the strongest competing methods, while remaining competitive on diameter prediction. It also consistently surpasses SONAR and GraphCON, the two most closely related oscillatory GNN baselines, as well as established sheaf models, including NSD and CSNN. These results show that ONDA’s oscillatory sheaf dynamics are particularly effective on tasks designed to stress long-range information propagation.

## 5.2 GRAPH BOTTLENECK INFORMATION TRANSFER: BARBELL BENCHMARK

Following (Bamberger et al., 2025), we consider the Barbell benchmark to assess robustness to oversquashing. This synthetic node-regression task consists of two cliques connected by a single bridge edge and requires information to be transferred between them. Node features are independently sampled from two disjoint distributions, one per clique, and each node must predict the empirical feature mean of the opposite subgraph. Despite the graph diameter being three, every cross-clique signal must traverse the same bridge edge. The benchmark therefore isolates whether a model can aggregate a clique-level statistic and transport it through a severe communication bottleneck. We report nodewise mean-squared error, averaged over test graphs. Under the standard benchmark scaling, an MSE near 1 indicates no effective cross-clique transfer, values between 0.4 and 0.6 indicate partial transfer, and values below 0.25 are considered to solve the task (Bamberger et al., 2025; Hariri et al., 2025).

Results. Table 6 summarizes Barbell results, with an extended comparison in Appendix C.1. Standard NSD variants fail to achieve effective cross-clique transfer at any tested graph sizes, underscoring the need for an explicit propagation mechanism inn SNNs. Even CSNN and SONAR, the closest competitors from the sheaf and oscillatory perspectives, respectively, achieve only partial transfer. By contrast, ONDA solves the task near-perfectly at $N = 1 0$ and $N = 2 0$ . The gap is most revealing at $N = 5 0$ , where ONDA is the only method to satisfy the solving criterion, while all baselines exhibit only limited information transfer.

## 5.3 GRAPH TRANSFER TASK

We consider the graph transfer task introduced by (Di Giovanni et al., 2023), following the regression setup of (Gravina et al., 2025), with the aim of assessing model capabilities to propagate information at increasing distances with different topologies and bottlenecks. Nodes are initialized with random features, except for the source and target nodes, whose values are set to 1 and 0, respectively. These nodes are separated by $\kappa \in \{ 3 , 5 , 1 0 , 5 0 \}$ hops, and the objective is to swap their values while leaving all other node features unchanged. The task considers line, ring, and crossed-ring graphs, which induce increasing levels of propagation difficulty. We use the same data, hyperparameter search space, and experimental protocol as in (Gravina et al., 2025).

Table 2: Barbell-clique Results. Test MSE $( \times 1 0 ^ { - 3 } ; \downarrow )$ Results are reported as $\mathrm { \ m e a n _ { \pm s t d } }$ over 5 seeds using the best configuration selected by validation. For each column, the best, second-best, and third-best results are highlighted.
<table><tr><td>Model</td><td> ${ \cal N } = { \bf 1 0 } \downarrow$ </td><td> $\mathbf { N } = \mathbf { 2 0 } \downarrow$ </td><td> $\mathbf { N } = \mathbf { 5 0 } \downarrow$ </td></tr><tr><td>GCN</td><td> $8 8 4 . 2 0 0 { \scriptstyle \pm 6 . 9 0 0 }$ </td><td> $9 6 1 . 1 0 0 { \scriptstyle \pm 1 2 . 3 0 0 }$ </td><td> $8 6 0 . 0 0 0 { \scriptstyle \pm 3 5 . 0 0 0 }$ </td></tr><tr><td>GAT</td><td> $7 . 9 0 0 { \scriptstyle \pm 2 . 1 0 0 }$ </td><td> $4 8 7 . 0 0 0 { \scriptstyle \pm 3 4 2 . 2 0 0 }$ </td><td> $8 5 2 . 0 0 0 { \scriptstyle \pm 2 9 . 0 0 0 }$ </td></tr><tr><td>GraphSAGE</td><td> $4 5 5 . 5 0 0 { \scriptstyle \pm 1 9 9 . 6 0 0 }$ </td><td> $9 9 8 . 2 0 0 { \scriptstyle \pm 6 . 5 0 0 }$ </td><td> $8 7 6 . 0 0 0 { \scriptstyle \pm 3 2 . 0 0 0 }$ </td></tr><tr><td>GIN</td><td> $9 0 0 . 5 0 0 { \scriptstyle \pm 4 7 . 9 0 0 }$ </td><td> $9 7 0 . 6 0 0 { \scriptstyle \pm 3 . 9 0 0 }$ </td><td> $9 8 5 . 0 0 0 { \scriptstyle \pm 1 2 3 . 0 0 0 }$ </td></tr><tr><td>GraphTransformer</td><td> $6 7 0 . 3 0 0 { \scriptstyle \pm 3 0 6 . 8 0 0 }$ </td><td> $1 0 4 1 . 2 0 0 { \scriptstyle \pm 4 6 . 9 0 0 }$ </td><td> $8 3 1 . 0 0 0 { \scriptstyle \pm 1 3 . 0 0 0 }$ </td></tr><tr><td>ChebNet</td><td> $\mathbf { 1 0 . 6 0 0 { \scriptstyle \pm 1 . 2 0 0 } }$ </td><td> $9 8 3 . 6 0 0 { \scriptstyle \pm 5 . 1 0 0 }$ </td><td> $9 2 2 . 0 0 0 { \scriptstyle \pm 4 7 . 0 0 0 }$ </td></tr><tr><td>SONAR</td><td> $4 6 2 . 8 0 0 { \scriptstyle \pm 2 5 0 . 3 0 0 }$ </td><td> $\mathbf { 7 1 7 . 5 0 0 { \scriptstyle \pm 2 4 9 . 6 0 0 } }$ </td><td> $\mathbf { 8 0 9 . 0 0 0 { \scriptstyle \pm 6 8 . 0 0 0 } }$ </td></tr><tr><td>NSD</td><td> $9 3 7 . 2 0 0 { \scriptstyle \pm 1 1 . 4 0 0 }$ </td><td> $1 0 0 5 . 1 0 0 { \scriptstyle \pm 7 . 8 0 0 }$ </td><td> $8 2 5 . 0 0 0 { \scriptstyle \pm 3 2 . 0 0 0 }$ </td></tr><tr><td>CSNN</td><td> $4 7 3 . 9 0 0 { \scriptstyle \pm 1 0 6 . 2 0 0 }$ </td><td> $8 9 0 . 8 0 0 { \scriptstyle \pm 2 6 5 . 4 0 0 }$ </td><td> $7 4 2 . 0 0 0 { \scriptstyle \pm 1 6 8 . 0 0 0 }$ </td></tr><tr><td>ONDA</td><td> $0 . 7 0 0 { \scriptstyle \pm 0 . 3 0 0 }$ </td><td> $0 . 9 0 0 { \scriptstyle \pm 0 . 5 0 0 }$ </td><td> $8 3 . 0 0 0 { \scriptstyle \pm 8 5 . 0 0 0 }$ </td></tr></table>

![](images/e115705b962cd967a65eba4b57f025d0e8bfa2c20d9775e34b3a8d251f90d374.jpg)  
(a) Line

![](images/1d5dbbb75f0973a86359e2c5a18e4b136a60706fb4db5567ee84ca15a0633bd5.jpg)  
(b) Ring

![](images/4b6d802865b36017d3aad747c8104f34aa9dcbef43767028ccf6f164c8bc3830.jpg)  
(c) Crossed-Ring  
Figure 2: Information transfer performance on (a) Line, (b) Ring, and (c) Crossed-Ring graphs.

Results. Figure 2 reports the mean test MSE and standard deviation over three seeds as a function of the distance between source and target nodes. ONDA achieves the lowest error at almost all evaluated distances across the three topologies, while most competing methods progressively degrade as the propagation distance increases. In particular, ONDA consistently improves over non-dissipative GNNs such as A-DGN, SWAN, and PH-DGN, while also outperforming GraphCON and SONAR, which share its oscillatory inductive bias but lack the richer matrix-valued transport provided by the sheaf structure. It also compares favorably with the graph Transformer GPS, despite relying only on local propagation. Overall, these results show that ONDA maintains effective information transfer as the distance between source and target nodes increases.

## 6 CONCLUSION

We introduced ONDA, a sheaf neural network that couples learned matrix-valued transport with second-order oscillatory dynamics for effective long-range information propagation. Rather than relying on richer sheaf transport alone, ONDA uses the learned sheaf to determine how information is transformed between stalks, while wave-like dynamics allow these transformed signals to remain influential as they propagate across the graph. Our theoretical analysis shows that connected nodes can influence one another regardless of their distance, and our experiments demonstrate the effectiveness of ONDA across a large variety of tasks, including long-range propagation and shortrange over-squashing, where our model sets a new bar for state-of-the-art performance, as well as heterophilic benchmarks.

## 7 ACKNOWLEDGEMENTS

Jan-Willem Van Looy was supported by MSCA-DN CaLiForNIA, Project ID: 101119552. Rita Fioresi and Ferdinando Zanchetta acknowledge support by GNSAGA-Indam, INFN Gast Initiative, PNRR MNESYS, PNRR National Center for HPC, Big Data and Quantum Computing CUP J33C22001170001, PNNR SIMQuSEC CUP J13C22000680006. This work was also supported by Horizon Europe EU projects MSCA-SE CaLIGOLA, Project ID: 101086123, COST Action CaL ISTA CA21109.

## REFERENCES

Uri Alon and Eran Yahav. On the Bottleneck of Graph Neural Networks and its Practical Implications. In International Conference on Learning Representations, 2021. URL https: //openreview.net/forum?id=i80OPhOCVH2.

Alvaro Arroyo, Alessio Gravina, Benjamin Gutteridge, Federico Barbero, Claudio Gallicchio, Xi-<sup>´</sup> aowen Dong, Michael M. Bronstein, and Pierre Vandergheynst. On Vanishing Gradients, Over-Smoothing, and Over-Squashing in GNNs: Bridging Recurrent and Graph Learning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

Jacob Bamberger, Federico Barbero, Xiaowen Dong, and Michael Bronstein. Bundle neural network for message diffusion on graphs. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=scI9307PLG.

Federico Barbero, Cristian Bodnar, Haitz Saez de Oc´ ariz Borde, Michael Bronstein, Petar´ Velickoviˇ c, and Pietro Li´ o. Sheaf neural networks with connection laplacians. In\` Proceedings of Topological, Algebraic, and Geometric Learning Workshops 2022, volume 196 of Proceedings of Machine Learning Research, pp. 28–36. PMLR, 25 Feb–22 Jul 2022.

Mitchell Black, Zhengchao Wan, Amir Nayyeri, and Yusu Wang. Understanding oversquashing in gnns through the lens of effective resistance. In Proceedings ofthe 40th International Conference on Machine Learning, 2023.

Deyu Bo, Xiao Wang, Chuan Shi, and Huawei Shen. Beyond low-frequency information in graph convolutional networks. Proceedings of the AAAI Conference on Artificial Intelligence, 35(5): 3950–3957, May 2021. doi: 10.1609/aaai.v35i5.16514. URL https://ojs.aaai.org/ index.php/AAAI/article/view/16514.

Cristian Bodnar, Francesco Di Giovanni, Benjamin Chamberlain, Pietro Lio, and Michael Bronstein. Neural sheaf diffusion: A topological perspective on heterophily and oversmoothing in gnns. Advances in Neural Information Processing Systems, 35:18527–18541, 2022.

Alessio Borgi, Fabrizio Silvestri, and Pietro Lio. Polynomial neural sheaf diffusion: A spectral\` filtering approach on cellular sheaves. arXiv preprint arXiv:2512.00242, 2025.

Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule, Fabrizio Silvestri, and Pietro Lio.\` sheaf-mpnn: A pytorch implementation of sheaf neural networks as message passing. https:// github.com/alessioborgi/sheaf-mpnn, 2026. GitHub repository, accessed 22 June 2026.

Xavier Bresson and Thomas Laurent. Residual Gated Graph ConvNets. arXiv preprint arXiv:1711.07553, 2018.

Andries E Brouwer and Willem H Haemers. Spectra of graphs. Springer, New York, NY, 2012. ISBN 978-1-4614-1939-6. doi: 10.1007/978-1-4614-1939-6.

Chen Cai and Yusu Wang. A note on over-smoothing for graph neural networks. arXiv preprint arXiv:2006.13318, 2020.

Andrea Ceni, Alessio Gravina, Claudio Gallicchio, Davide Bacciu, Carola-Bibiane Schonlieb, and Moshe Eliasof. Message-Passing State-Space Models: Improving Graph Learning with Modern Sequence Modeling. arXiv preprint arXiv:2505.18728, 2025.

Jinsong Chen, Kaiyuan Gao, Gaichao Li, and Kun He. NAGphormer: A tokenized graph transformer for node classification in large graphs. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=8KYeilT3Ow.

Ming Chen, Zhewei Wei, Zengfeng Huang, Bolin Ding, and Yaliang Li. Simple and Deep Graph Convolutional Networks. In Hal Daume III and Aarti Singh (eds.),´ Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings ofMachine Learning Research, pp. 1725–1735. PMLR, 13–18 Jul 2020.

Eli Chien, Jianhao Peng, Pan Li, and Olgica Milenkovic. Adaptive universal generalized pagerank graph neural network. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=n6jl7fLxrP.

John B Conway. A course in functional analysis. Graduate Texts in Mathematics. Springer, New York, NY, 2 edition, 2019. ISBN 978-1-4757-4383-8. doi: 10.1007/978-1-4757-4383-8.

Justin Michael Curry. Sheaves, cosheaves and applications. PhD thesis, University of Pennsylvania, 2014.

Michael Defferrard, Xavier Bresson, and Pierre Vandergheynst. Convolutional Neural Networks on¨ Graphs with Fast Localized Spectral Filtering. In Advances in Neural Information Processing Systems, volume 29. Curran Associates, Inc., 2016.

Francesco Di Giovanni, Lorenzo Giusti, Federico Barbero, Giulia Luise, Pietro Lio, and Michael\` Bronstein. On over-squashing in message passing neural networks: the impact of width, depth, and topology. In Proceedings of the 40th International Conference on Machine Learning, ICML’23. JMLR.org, 2023.

Lun Du, Xiaozhou Shi, Qiang Fu, Xiaojun Ma, Hengyu Liu, Shi Han, and Dongmei Zhang. Gbkgnn: Gated bi-kernel graph neural networks for modeling both homophily and heterophily. In Proceedings ofthe ACM Web Conference 2022, WWW ’22, pp. 1550–1558, New York, NY, USA, 2022. Association for Computing Machinery. ISBN 9781450390965. doi: 10.1145/3485447. 3512201. URL https://doi.org/10.1145/3485447.3512201.

Vijay Prakash Dwivedi and Xavier Bresson. A Generalization of Transformer Networks to Graphs. AAAI Workshop on Deep Learning on Graphs: Methods and Applications, 2021.

Federico Errica, Henrik Christiansen, Viktor Zaverkin, Takashi Maruyama, Mathias Niepert, and Francesco Alesiani. Adaptive message passing: A general framework to mitigate oversmoothing, oversquashing, and underreaching. arXiv preprint arXiv:2312.16560, 2024.

Lawrence C Evans. Partial Differential Equations. Graduate studies in mathematics. American Mathematical Society, Providence, RI, 2 edition, March 2010.

Joel H Ferziger and Milovan Peric. Computational methods for fluid dynamics. Springer, Berlin, Germany, 3 edition, November 2001.

Ben Finkelshtein, Xingyue Huang, Michael M. Bronstein, and Ismail Ilkan Ceylan. Cooperative Graph Neural Networks. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview.net/forum?id=ZQcqXCuoxD.

Johannes Gasteiger, Stefan Weiß enberger, and Stephan Gunnemann. Diffusion Improves Graph¨ Learning. In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.

Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, and George E. Dahl. Neural message passing for Quantum chemistry. In Proceedings ofthe 34th International Conference on Machine Learning, volume 70 of ICML’17, pp. 1263–1272. JMLR.org, 2017.

Alessio Gravina, Davide Bacciu, and Claudio Gallicchio. Anti-Symmetric DGN: a stable architecture for Deep Graph Networks. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=J3Y7cgZOOS.

Alessio Gravina, Moshe Eliasof, Claudio Gallicchio, Davide Bacciu, and Carola-Bibiane Schonlieb.¨ On Oversquashing in Graph Neural Networks Through the Lens of Dynamical Systems. Proceedings of the AAAI Conference on Artificial Intelligence, 39(16):16906–16914, Apr. 2025. doi: 10.1609/aaai.v39i16.33858. URL https://ojs.aaai.org/index.php/AAAI/ article/view/33858.

Albert Gu, Karan Goel, and Christopher Re. Efficiently modeling long sequences with structured´ state spaces. arXiv preprint arXiv:2111.00396, 2021.

Benjamin Gutteridge, Xiaowen Dong, Michael M Bronstein, and Francesco Di Giovanni. Drew: Dynamically rewired message passing with delay. In International Conference on Machine Learning, pp. 12252–12267. PMLR, 2023.

Paul R. Halmos. Finite-Dimensional Vector Spaces. Undergraduate Texts in Mathematics. Springer, New York, NY, 1974. ISBN 978-1-4612-6387-6. doi: 10.1007/978-1-4612-6387-6.

William L. Hamilton, Rex Ying, and Jure Leskovec. Inductive representation learning on large graphs. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, pp. 1025–1035. Curran Associates Inc., 2017. ISBN 9781510860964.

Andi Han, Dai Shi, Lequan Lin, and Junbin Gao. From Continuous Dynamics to Graph Neural Networks: Neural Diffusion and Beyond. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=fPQSxjqa2o. Survey Certification.

Jakob Hansen. Laplacians of cellular sheaves: Theory and applications. PhD thesis, University of Pennsylvania, 2020.

Jakob Hansen and Thomas Gebhart. Sheaf neural networks. In NeurIPS 2020 Workshop on TDA and Beyond, 2020. URL https://arxiv.org/abs/2012.06333.

Jakob Hansen and Robert Ghrist. Toward a spectral theory of cellular sheaves. Journal ofApplied and Computational Topology, 3(4):315–358, 2019.

Ali Hariri, Alvaro Arroyo, Alessio Gravina, Moshe Eliasof, Carola-Bibiane Schonlieb, Davide¨ Bacciu, Xiaowen Dong, Kamyar Azizzadenesheli, and Pierre Vandergheynst. Return of Cheb-Net: Understanding and Improving an Overlooked GNN on Long Range Tasks. In The Thirtyninth Annual Conference on Neural Information Processing Systems, 2025. URL https: //openreview.net/forum?id=oLyfML1Qze.

Simon Heilig, Alessio Gravina, Alessandro Trenta, Claudio Gallicchio, and Davide Bacciu. Port-Hamiltonian Architectural Bias for Long-Range Propagation in Deep Graph Networks. In The Thirteenth International Conference on Learning Representations, 2025. URL https: //openreview.net/forum?id=03EkqSCKuO.

Ferran Hernandez Caralt, Guillermo Bernardez Gil, Iulia Duta, Pietro Li´ o, and Eduard Alarc\` on Cot.´ Joint diffusion processes as an inductive bias in sheaf neural networks. In Proceedings of the Geometry-grounded Representation Learning and Generative Modeling Workshop (GRaM), volume 251 of Proceedings ofMachine Learning Research, pp. 249–263. PMLR, 29 Jul 2024.

Nicholas J Higham. Functions of matrices: theory and computation. SIAM, 2008. ISBN 978-0- 89871-777-8. doi: 10.1137/1.9780898717778.

Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, and Jure Leskovec. Strategies for Pre-training Graph Neural Networks. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id= HJlWWJSFDH.

Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations, 2017. URL https: //openreview.net/forum?id=SJU4ayYgl.

Kezhi Kong, Jiuhai Chen, John Kirchenbauer, Renkun Ni, C. Bayan Bruss, and Tom Goldstein. GOAT: A global transformer on large-scale graphs. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 17375–17390. PMLR, 23–29 Jul 2023. URL https: //proceedings.mlr.press/v202/kong23a.html.

Devin Kreuzer, Dominique Beaini, Will Hamilton, Vincent Letourneau, and Prudencio Tossou. Re-´ thinking graph transformers with spectral attention. Advances in Neural Information Processing Systems, 34:21618–21629, 2021.

Xiang Li, Renyu Zhu, Yao Cheng, Caihua Shan, Siqiang Luo, Dongsheng Li, and Weining Qian. Finding global homophily in graph neural networks when meeting heterophily. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pp. 13242–13256. PMLR, 17–23 Jul 2022. URL https://proceedings.mlr.press/v162/li22ad.html.

Sitao Luan, Chenqing Hua, Qincheng Lu, Liheng Ma, Lirong Wu, Xinyu Wang, Minkai Xu, Xiao-Wen Chang, Doina Precup, Rex Ying, Stan Z. Li, Jian Tang, Guy Wolf, and Stefanie Jegelka. The heterophilic graph learning handbook: Benchmarks, models, theoretical analysis, applications and challenges. arXiv preprint arXiv:2407.09618, 2024.

Liheng Ma, Chen Lin, Derek Lim, Adriana Romero-Soriano, Puneet K. Dokania, Mark Coates, Philip Torr, and Ser-Nam Lim. Graph inductive biases in transformers without message passing. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pp. 23321–23337. PMLR, 23–29 Jul 2023.

Sunil Kumar Maurya, Xin Liu, and Tsuyoshi Murata. Simplifying approach to node classification in graph neural networks. Journal of Computational Science, 62:101695, 2022. ISSN 1877-7503. doi: https://doi.org/10.1016/j.jocs.2022.101695. URL https://www.sciencedirect. com/science/article/pii/S1877750322000990.

Alessio Micheli. Neural Network for Graphs: A Contextual Constructive Approach. IEEE Transactions on Neural Networks, 20(3):498–511, 2009.

Luca Miglior, Matteo Tolloso, Alessio Gravina, and Davide Bacciu. Can You Hear Me Now? A Benchmark for Long-Range Graph Propagation. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id= DgkWFPZMPp.

Yaaqov Mishayev, Yonatan Sverdlov, Tal Amir, and Nadav Dym. Short-range oversquashing. In The Fourth Learning on Graphs Conference, 2025. URL https://openreview.net/forum? id=rmX8Jamnyg.

Luis Muller, Mikhail Galkin, Christopher Morris, and Ladislav Ramp¨ a´sek. Attending to graphˇ transformers. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=HhbqHBBrfZ.

Kenta Oono and Taiji Suzuki. Graph Neural Networks Exponentially Lose Expressive Power for Node Classification. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=S1ldO2EFPr.

Antonio Orvieto, Samuel L Smith, Albert Gu, Anushan Fernando, Caglar Gulcehre, Razvan Pascanu, and Soham De. Resurrecting recurrent neural networks for long sequences. In International Conference on Machine Learning, pp. 26670–26698. PMLR, 2023.

Oleg Platonov, Denis Kuznedelev, Michael Diskin, Artem Babenko, and Liudmila Prokhorenkova. A critical look at the evaluation of GNNs under heterophily: Are we really making progress? In The Eleventh International Conference on Learning Representations, 2023. URL https: //openreview.net/forum?id=tJbbQfw-5wv.

Michael Poli, Stefano Massaroli, Junyoung Park, Atsushi Yamashita, Hajime Asama, and Jinkyoo Park. Graph neural ordinary differential equations. arXiv preprint arXiv:1911.07532, 2019.

Ladislav Rampa´sek, Mikhail Galkin, Vijay Prakash Dwivedi, Anh Tuan Luu, Guy Wolf, and Do-ˇ minique Beaini. Recipe for a General, Powerful, Scalable Graph Transformer. Advances in Neural Information Processing Systems, 35, 2022.

Andre Ribeiro, Ana Luiza Tenorio, Juan Belieni, Amauri H Souza, and Diego Mesquita. Cooperative´ sheaf neural networks. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=AHpexliCTM.

T Konstantin Rusch, Ben Chamberlain, James Rowbottom, Siddhartha Mishra, and Michael Bronstein. Graph-coupled oscillator networks. In International Conference on Machine Learning, pp. 18888–18909. PMLR, 2022.

T. Konstantin Rusch, Michael M. Bronstein, and Siddhartha Mishra. A Survey on Oversmoothing in Graph Neural Networks. arXiv preprint arXiv:2303.10993, 2023.

Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. The Graph Neural Network Model. IEEE Transactions on Neural Networks, 20(1):61–80, 2009.

Allen Dudley Shepard. A cellular description of the derived category of a stratified space. Brown University, 1985.

Dai Shi, Andi Han, Lequan Lin, Yi Guo, and Junbin Gao. Exposition on over-squashing problem on GNNs: Current Methods, Benchmarks and Challenges, 2023.

Yunsheng Shi, Zhengjie Huang, Shikun Feng, Hui Zhong, Wenjing Wang, and Yu Sun. Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification. In Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21, pp. 1548–1554. International Joint Conferences on Artificial Intelligence Organization, 8 2021a. doi: 10.24963/ijcai.2021/214. URL https://doi.org/10.24963/ijcai.2021/214.

Yunsheng Shi, Zhengjie Huang, Shikun Feng, Hui Zhong, Wenjing Wang, and Yu Sun. Masked label prediction: Unified message passing model for semi-supervised classification. In Zhi-Hua Zhou (ed.), Proceedings ofthe Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21, pp. 1548–1554. International Joint Conferences on Artificial Intelligence Organization, 8 2021b. doi: 10.24963/ijcai.2021/214. URL https://doi.org/10.24963/ijcai. 2021/214. Main Track.

Hamed Shirzad, Ameya Velingker, Balaji Venkatachalam, Danica J Sutherland, and Ali Kemal Sinop. Exphormer: Sparse transformers for graphs. In International Conference on Machine Learning, pp. 31613–31632. PMLR, 2023.

Joshua Southern, Francesco Di Giovanni, Michael M. Bronstein, and Johannes F. Lutzeyer. Understanding virtual nodes: Oversquashing and node heterogeneity. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum? id=NmcOAwRyH5.

Gerald Teschl. Ordinary differential equations and dynamical systems, volume 140 of Graduate Studies in Mathematics. American Mathematical Soc., 2012. doi: 10.1090/gsm/140.

Jake Topping, Francesco Di Giovanni, Benjamin Paul Chamberlain, Xiaowen Dong, and Michael M. Bronstein. Understanding over-squashing and bottlenecks on graphs via curvature. In International Conference on Learning Representations, 2022. URL https://openreview.net/ forum?id=7UmjRGzp-A.

Alessandro Trenta, Alessio Gravina, and Davide Bacciu. SONAR: Long-Range Graph Propagation Through Information Waves. In Advances in Neural Information Processing Systems, volume 38, Main Conference, pp. 145054–145087. Curran Associates, Inc., 2025. doi: 10.52202/085713-4854.

A. Vaswani et al. Attention is all you need. Advances in Neural Information Processing Systems, 30, 2017.

Petar Velickoviˇ c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Li´ o, and Yoshua\` Bengio. Graph Attention Networks. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=rJXMpikCZ.

Xiyuan Wang and Muhan Zhang. How powerful are spectral graph neural networks. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pp. 23341–23362. PMLR, 17–23 Jul 2022. URL https://proceedings.mlr.press/v162/wang22am.html.

Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How Powerful are Graph Neural Networks? In International Conference on Learning Representations, 2019. URL https: //openreview.net/forum?id=ryGs6iA5Km.

Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin $\mathrm { K e , }$ Di He, Yanming Shen, and Tie-Yan Liu. Do transformers really perform badly for graph representation? Advances in Neural Information Processing Systems, 34:28877–28888, 2021.

Olga Zaghen, Antonio Longa, Steve Azzolin, Lev Telyatnikov, Andrea Passerini, and Pietro Lio.\` Sheaf diffusion goes nonlinear: Enhancing gnns with adaptive sheaf laplacians. In Proceedings of the Geometry-grounded Representation Learning and Generative Modeling Workshop (GRaM), volume 251 of Proceedings ofMachine Learning Research, pp. 264–276. PMLR, 29 Jul 2024.

Jiong Zhu, Yujun Yan, Lingxiao Zhao, Mark Heimann, Leman Akoglu, and Danai Koutra. Beyond homophily in graph neural networks: Current limitations and effective designs. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 7793–7804. Curran Associates, Inc., 2020.

Jiong Zhu, Ryan A. Rossi, Anup Rao, Tung Mai, Nedim Lipka, Nesreen K. Ahmed, and Danai Koutra. Graph neural networks with heterophily. Proceedings of the AAAI Conference on Artificial Intelligence, 35(12):11168–11176, May 2021. doi: 10.1609/aaai.v35i12.17332. URL https://ojs.aaai.org/index.php/AAAI/article/view/17332.

## A MATHEMATICAL BACKGROUND AND PROOFS

This appendix provides the mathematical background and proofs for the theoretical results presented throughout Section 4. First, Appendix A.1 gathers some additional notation and conventions on inner products. Section A.2 establishes the spectral bound for the normalized sheaf Laplacian and includes a description of the directed sheaf operator. Appendix A.3 derives the closed-form solution of the continuous sheaf wave equation, while Appendices A.4 and A.5 prove the pathwise and spectral representations of the source-to-target response, respectively. Finally, Appendix A.6 proves the discrete-time result from Theorem 1, and Corollary 1 shows that the discrete response converges to the continuous response as the step size tends to zero. Throughout, all vector spaces are over R.

## A.1 PRELIMINARIES ON INNER PRODUCTS AND NORMS

We recall basic definitions and facts concerning inner products and norms on finite-dimensional spaces. This material can be found in standard introductory textbooks to functional analysis, see for instance (Conway, 2019). For each vertex $u \in \mathcal V$ , denote by $\langle \cdot , \cdot \rangle _ { \mathscr { F } ( u ) }$ a fixed real inner product on $\mathcal { F } ( u )$ and write $\| \boldsymbol { X } \| _ { \mathcal { F } ( \boldsymbol { u } ) } ^ { 2 } = \langle \boldsymbol { X } , \boldsymbol { X } \rangle _ { \mathcal { F } ( \boldsymbol { u } ) }$ for $\pmb { X } \in \mathcal { F } ( \boldsymbol { u } )$ . The cochain space ${ \mathcal { H } } = C ^ { 0 } ( { \mathcal { G } } ; { \mathcal { F } } )$ carries the direct sum inner product, defined by $\begin{array} { r } { \langle X , Y \rangle _ { \mathcal { H } } : = \sum _ { u \in \mathcal { V } } \langle X _ { u } , Y _ { u } \rangle _ { \mathcal { F } ( u ) } } \end{array}$ and $\| X \| _ { \mathcal { H } } ^ { 2 } =$ $\textstyle \sum _ { u \in \mathcal { V } } \| X _ { u } \| _ { \mathcal { F } ( u ) } ^ { 2 }$ . All matrix representations are taken with respect to fixed orthonormal bases of the vertex and edge stalks.

For a linear map $A : F  G$ between finite-dimensional inner-product spaces, we use the induced operator norm, defined as

$$
\| A \| : = \operatorname* { s u p } _ { \| X \| _ { F } \leq 1 } \| A X \| _ { G } .
$$

Since we consider finite-dimensional vector spaces, this is a finite quantity and satisfies $\| A X \| _ { G } \leq$ $\| A \| \| X \| _ { F }$ . Additionally, for $c \in \mathbb { R } , A , B ^ { \widehat { } } \in \operatorname { H o m } _ { \mathbb { R } } ( F , G ) , D ^ { \widehat { } } \in \operatorname { H o m } _ { \mathbb { R } } ( E , F )$ , the following properties hold:

$$
\| c A \| = | c | \| A \| , \qquad \| A + B \| \leq \| A \| + \| B \| , \qquad \| A D \| \leq \| A \| \| D \| .
$$

Recall that a sequence $A _ { n } : F \to G$ converges to A in operator norm if $\| A _ { n } - A \| \to 0$ . An operator series $\bar { \sum { \mathfrak { n } } } { = } 0 ^ { \infty } A _ { n }$ converges absolutely in operator norm if $\textstyle \sum _ { n = 0 } ^ { \infty } \| A _ { n } \| < \infty$ . Since F and $\bar { G }$ are finite-dimensional, the space of linear maps $\dot { F }  G$ is complete in the operator norm. Thus, absolute convergence guarantees convergence of the partial sums in operator norm.

For $A , B \in \operatorname { H o m } _ { \mathbb { R } } ( F , G )$ , we also use the Hilbert–Schmidt inner product

$$
\begin{array} { r } { \langle A , B \rangle _ { \mathrm { H S } } : = \mathrm { t r } ( A ^ { * } B ) , \qquad \| A \| _ { \mathrm { H S } } ^ { 2 } : = \langle A , A \rangle _ { \mathrm { H S } } . } \end{array}
$$

Since ${ \mathrm { H o m } } _ { \mathbb { R } } ( F , G )$ is finite-dimensional, all norms on this space are equivalent. Thus, convergence of operator series is the same in the operator and Hilbert–Schmidt norms, as is absolute convergence.

## A.2 SHEAF LAPLACIAN AND SPECTRAL BOUNDS

Recall that we denote by $\mathcal { D } _ { \mathcal { F } }$ the block-diagonal part of $L _ { \mathcal { F } }$ , given by $\begin{array} { r } { ( \mathcal { D } _ { \mathcal { F } } ) _ { v v } = \sum _ { e \ni v } \mathcal { F } _ { v , e } ^ { * } \mathcal { F } _ { v , e } , } \end{array}$ The symmetrically normalized sheaf Laplacian is given by $\Delta _ { \mathcal { F } } = \mathcal { D } _ { \mathcal { F } } ^ { - 1 / 2 } L _ { \mathcal { F } } \mathcal { D } _ { \mathcal { F } } ^ { - 1 / 2 }$ . Here, $\mathcal { D } _ { \mathcal { F } } ^ { - 1 / 2 }$ denotes the Moore–Penrose pseudoinverse of $\mathcal { D } _ { \mathcal { F } } ^ { 1 / 2 }$ . It acts by $\mu ^ { - 1 / 2 }$ on each eigenspace of $\mathcal { D } _ { \mathcal { F } }$ with eigenvalue $\mu > 0 .$ , and by zero on ker $\mathcal { D } _ { \mathcal { F } }$ . We recall the following well-known fact, presented here with a self-contained elementary proof. A more general version of this result can be found in (Hansen & Ghrist, 2019).

Lemma 1. The symmetrically normalized sheaf Laplacian $\Delta { _ { \mathcal { F } } }$ is self-adjoint and all of its eigenvalues lie in [0, 2].

Proof. Since $L _ { \mathcal { F } }$ and $\mathcal { D } _ { \mathcal { F } } ^ { - 1 / 2 }$ are self-adjoint, also $\Delta _ { \mathcal { F } } ^ { * } = \mathcal { D } _ { \mathcal { F } } ^ { - 1 / 2 } L _ { \mathcal { F } } ^ { * } \mathcal { D } _ { \mathcal { F } } ^ { - 1 / 2 } = \Delta _ { \mathcal { F } }$

Let $( q _ { j } ) _ { j }$ be an orthonormal eigenbasis of $\mathcal { D } _ { \mathcal { F } }$ , with eigenvalues $\mu _ { j } \geq 0$ . For $\begin{array} { r } { X = \sum _ { j } \alpha _ { j } q _ { j } } \end{array}$ , we have

$$
Y = \mathcal { D } _ { \mathcal { F } } ^ { - 1 / 2 } X = \sum _ { \mu _ { j } > 0 } \frac { \alpha _ { j } } { \sqrt { \mu _ { j } } } q _ { j } , \qquad \langle Y , \mathcal { D } _ { \mathcal { F } } Y \rangle = \sum _ { \mu _ { j } > 0 } | \alpha _ { j } | ^ { 2 } \leq \sum _ { j } | \alpha _ { j } | ^ { 2 } = \| X \| ^ { 2 } .
$$

Substituting $Y = \mathcal { D } _ { \mathcal { F } } ^ { - 1 / 2 } X$ , we obtain

$$
\begin{array} { r l } & { 0 \leq \langle X , \Delta _ { \mathcal { F } } X \rangle = \langle Y , L _ { \mathcal { F } } Y \rangle = \displaystyle \sum _ { e = \{ u , v \} \in \mathcal { E } } \| \mathcal { F } _ { u , e } Y _ { u } - \mathcal { F } _ { v , e } Y _ { v } \| ^ { 2 } } \\ & { \qquad \leq 2 \displaystyle \sum _ { e = \{ u , v \} \in \mathcal { E } } \big ( \| \mathcal { F } _ { u , e } Y _ { u } \| ^ { 2 } + \| \mathcal { F } _ { v , e } Y _ { v } \| ^ { 2 } \big ) } \\ & { \qquad = 2 \displaystyle \sum _ { v \in \mathcal { V } } \displaystyle \sum _ { e \ni v } \langle Y _ { v } , \mathcal { F } _ { v , e } ^ { * } \mathcal { F } _ { v , e } Y _ { v } \rangle = 2 \langle Y , \mathcal { D } _ { \mathcal { F } } Y \rangle \leq 2 \| X \| ^ { 2 } , } \end{array}
$$

where the inequality in the second line uses $\| a - b \| ^ { 2 } \leq 2 \| a \| ^ { 2 } + 2 \| b \| ^ { 2 }$ , and the final inequality uses the bound on $\langle \bar { Y } , \bar { \mathcal { D } } \mathcal { F } Y \rangle$ established above. Hence, for any eigenvalue λ with non-zero eigenvector X, we find $\begin{array} { r } { 0 \le \lambda = \frac { \langle \pmb { X } , \Delta _ { \mathcal { F } } \pmb { X } \rangle } { \| \pmb { X } \| ^ { 2 } } \le 2 } \end{array}$ □

Directed variant. Here we give the precise description of the directed version of the sheaf Laplacian as mentioned in Section 3.2. For each edge $\mathit { \Pi } _ { e } ~ = ~ \{ u , v \}$ consider two oriented copies $e _ { u } ~ = ~ ( u  v )$ and $e _ { v } \ = \ ( v \  \ u )$ . Each copy $e _ { z } , z \in \{ u , v \}$ , carries an edge stalk $\mathcal { F } ( e _ { z } )$ and independently learned restriction maps $\mathcal { F } _ { w , e _ { z } } : \mathcal { F } ( w )  \mathcal { F } ( e _ { z } ) , w \in \{ u , v \}$ . The nodewise expression is given by

$$
( L _ { \mathcal { F } } ^ { \mathrm { d i r } } \pmb { X } ) _ { v } = \sum _ { e = \{ u , v \} \in \mathcal { E } } \mathcal { F } _ { v , e _ { v } } ^ { * } \big ( \mathcal { F } _ { v , e _ { v } } \pmb { X } _ { v } - \mathcal { F } _ { u , e _ { v } } \pmb { X } _ { u } \big ) .
$$

This operator is not guaranteed to be self-adjoint. Identifying $\mathcal { F } ( e _ { u } ) = \mathcal { F } ( e _ { v } ) = \mathcal { F } ( e )$ and forcing $\mathcal { F } _ { w , e _ { u } } \dot { = } \mathcal { F } _ { w , e _ { v } } = \dot { \mathcal { F } _ { w , e } }$ for $w \in \{ u , v \}$ recovers $L _ { \mathcal { F } }$

## A.3 SOLUTION OF THE SHEAF WAVE EQUATION

We derive the solution of Equation (5) by rewriting the second-order dynamics as a first-order linear system and computing its operator exponential.

Lemma 2. Let $L _ { \mathcal { F } }$ be fixed. The unique solution of

$$
\ddot { \mathbf { X } } ( t ) + L \mathcal { F } \pmb { X } ( t ) = 0 ,
$$

with initial conditions $\begin{array} { r } { { \cal X } ( 0 ) = \overline { { \cal X } } , \dot { \cal X } ( 0 ) = \overline { { \cal V } } } \end{array}$ , is given by $\begin{array} { r } { \pmb { X } ( t ) = \mathcal { C } _ { \mathcal { F } } ( t ) \overline { { \pmb { X } } } + \mathcal { S } _ { \mathcal { F } } ( t ) \overline { { \pmb { V } } } . } \end{array}$

Proof. For a time-dependent node state $\pmb { X } ( t ) \in C ^ { 0 } ( \mathcal { G } ; \mathcal { F } )$ with velocity $V ( t ) = \dot { \mathbf { X } } ( t )$ , we follow usual terminology of second-order dynamics (Teschl, 2012) and introduce the phase-space state

$$
Z ( t ) : = { \binom { X ( t ) } { \dot { X } ( t ) } } \in { \mathcal { H } } \oplus { \mathcal { H } } .
$$

In this notation, our problem becomes equivalent to the first-order system

$$
\dot { \pmb { Z } } ( t ) = G _ { \mathcal { F } } \pmb { Z } ( t ) , \qquad G _ { \mathcal { F } } : = \left( \begin{array} { c c } { 0 } & { I _ { \mathcal { H } } } \\ { - L _ { \mathcal { F } } } & { 0 } \end{array} \right) , \qquad \pmb { Z } ( 0 ) = \left( \frac { \pmb { X } } { \pmb { V } } \right) .\tag{13}
$$

Since $\mathcal { H } \oplus \mathcal { H }$ is finite-dimensional, $G _ { \mathcal { F } }$ is bounded. Its exponential is defined by the absolutely convergent power series $\begin{array} { r } { e ^ { t G _ { \mathcal { F } } } : = \sum _ { n = 0 } ^ { \infty } \frac { t ^ { n } } { n ! } G _ { \mathcal { F } } ^ { n } } \end{array}$ . Since

$$
G _ { \mathcal { F } } ^ { 2 } = - \left( \begin{array} { c c } { L _ { \mathcal { F } } } & { 0 } \\ { 0 } & { L _ { \mathcal { F } } } \end{array} \right) ,
$$

the even and odd powers in the above power series give

$$
\begin{array} { l } { { e ^ { \displaystyle t G _ { \mathcal { F } } } = \sum _ { k = 0 } ^ { \infty } ( - 1 ) ^ { k } \displaystyle \frac { t ^ { 2 k } } { ( 2 k ) ! } \left( \begin{array} { c c } { { L _ { \mathcal { F } } ^ { k } } } & { { 0 } } \\ { { 0 } } & { { L _ { \mathcal { F } } ^ { k } } } \end{array} \right) + \sum _ { k = 0 } ^ { \infty } ( - 1 ) ^ { k } \displaystyle \frac { t ^ { 2 k + 1 } } { ( 2 k + 1 ) ! } \left( \begin{array} { c c } { { 0 } } & { { L _ { \mathcal { F } } ^ { k } } } \\ { { - L _ { \mathcal { F } } ^ { k + 1 } } } & { { 0 } } \end{array} \right) } } \\ { { = \left( \begin{array} { c c } { { \mathcal { C } _ { \mathcal { F } } ( t ) } } & { { \mathcal { S } _ { \mathcal { F } } ( t ) } } \\ { { - L _ { \mathcal { F } } \mathcal { S } _ { \mathcal { F } } ( t ) } } & { { \mathcal { C } _ { \mathcal { F } } ( t ) } } \end{array} \right) . } } \end{array}
$$

The power series defining $\mathcal { C } _ { \mathcal { F } }$ and $\mathit { S } _ { \mathcal { F } }$ have infinite radius of convergence and can therefore be differentiated termwise. Hence

$$
\begin{array} { l } { { \displaystyle { \mathcal C } _ { \mathcal F } ^ { \prime } ( t ) = \sum _ { k = 1 } ^ { \infty } ( - 1 ) ^ { k } \frac { t ^ { 2 k - 1 } } { ( 2 k - 1 ) ! } L _ { \mathcal F } ^ { k } = - L _ { \mathcal F } \sum _ { j = 0 } ^ { \infty } ( - 1 ) ^ { j } \frac { t ^ { 2 j + 1 } } { ( 2 j + 1 ) ! } L _ { \mathcal F } ^ { j } = - L _ { \mathcal F } \mathcal S _ { \mathcal F } ( t ) , } } \\ { { \displaystyle { \mathcal S } _ { \mathcal F } ^ { \prime } ( t ) = \sum _ { k = 0 } ^ { \infty } ( - 1 ) ^ { k } \frac { t ^ { 2 k } } { ( 2 k ) ! } L _ { \mathcal F } ^ { k } = \mathcal C _ { \mathcal F } ( t ) . } } \end{array}
$$

Therefore, for $\pmb { X } ( t ) : = \mathcal { C } _ { \mathcal { F } } ( t ) \overline { { \pmb { X } } } + \mathcal { S } _ { \mathcal { F } } ( t ) \overline { { \pmb { V } } }$ , we find that

$$
\ddot { \pmb X } ( t ) = - L _ { \mathcal { F } } \mathcal { C } _ { \mathcal { F } } ( t ) \overline { { \pmb X } } - L _ { \mathcal { F } } \mathcal { S } _ { \mathcal { F } } ( t ) \overline { { \pmb V } } = - L _ { \mathcal { F } } \pmb X ( t ) .
$$

Moreover, the above identities at $t = 0$ give that $\mathbf { \boldsymbol { X } } ( 0 ) = \overline { { \mathbf { \boldsymbol { X } } } } , ~ \dot { \mathbf { \boldsymbol { X } } } ( 0 ) = \overline { { \mathbf { \boldsymbol { V } } } }$ . Hence $X ( t )$ solves our problem. Uniqueness follows by the standard existence-and-uniqueness theorem for finitedimensional linear ordinary differential equations. □

## A.4 PROOF OF PROPOSITION 1

We first express the blocks of $L _ { \mathcal { F } } ^ { k }$ as sums of compositions along length-k walks.

Recall that, for $k \geq 0$ and vertices $u , v \in \mathcal { V }$ , we denoted by $\mathcal { W } _ { k } ( u , v )$ the set of length-k walks $\omega = ( v _ { 0 } = u , v _ { 1 } , \ldots , v _ { k } = v )$ , where either $v _ { j } = v _ { j - 1 } \mathrm { ~ o r ~ } \{ v _ { j - 1 } , v _ { j } \} \in \mathcal { E }$ at each step. To each such $\omega \in \mathcal { W } _ { k } ( u , v )$ , we associated

$$
( L _ { \mathcal { F } } ) _ { \omega } = ( L _ { \mathcal { F } } ) _ { v _ { k } v _ { k - 1 } } \ldots ( L _ { \mathcal { F } } ) _ { v _ { 1 } v _ { 0 } } : \mathcal { F } ( u )  \mathcal { F } ( v ) .
$$

Allowing steps with $v _ { j } = v _ { j - 1 }$ accounts for the diagonal blocks of $L _ { \mathcal { F } }$ in these compositions. For $k = 0$ , we put ${ \mathcal W } _ { 0 } ( u , \dot { u } ) = \dot { \{ }  ( u ) \dot  \}$ }, with empty composition equal to $I _ { \mathcal { F } ( u ) }$

Lemma 3. For every $k \geq 0$ and $u , v \in \mathcal { V } ,$

$$
( L _ { \mathcal { F } } ^ { k } ) _ { v u } = \sum _ { \omega \in \mathcal { W } _ { k } ( u , v ) } ( L _ { \mathcal { F } } ) _ { \omega } : \mathcal { F } ( u ) \to \mathcal { F } ( v ) .
$$

Proof. We proceed by induction on k.

For $k = 0$ , the result is clear. Suppose the result holds for some $k \geq 0$ . Then

$$
( L _ { \mathcal { F } } ^ { k + 1 } ) _ { v u } = \sum _ { a \in \mathcal { V } } ( L _ { \mathcal { F } } ) _ { v a } ( L _ { \mathcal { F } } ^ { k } ) _ { a u } = \sum _ { a \in \mathcal { V } } \sum _ { \omega \in \mathcal { W } _ { k } ( u , a ) } ( L _ { \mathcal { F } } ) _ { v a } ( L _ { \mathcal { F } } ) _ { \omega } .
$$

Since $( L _ { \mathcal { F } } ) _ { v a } = 0$ whenever $a \neq v$ and $\{ a , v \} \not \in { \mathcal { E } }$ , only those a for which $a = v \mathrm { o r } \{ a , v \} \in \mathcal { E }$ can contribute. For every such term, if $\omega = ( v _ { 0 } = u , \ldots , v _ { k } = a ) \in \mathcal { W } _ { k } ( u , a )$ , appending v gives the walk $\widetilde { \boldsymbol { \omega } } = ( v _ { 0 } = u , \ldots , v _ { k } = a , v _ { k + 1 } = v ) \in \mathcal { W } _ { k + 1 } ( \boldsymbol { u } , \boldsymbol { v } )$ , with associated composition

$$
\begin{array} { r } { ( L _ { \mathcal { F } } ) _ { \widetilde { \omega } } = ( L _ { \mathcal { F } } ) _ { v a } ( L _ { \mathcal { F } } ) _ { \omega } . } \end{array}
$$

Hence, $\begin{array} { r } { ( L _ { \mathcal { F } } ^ { k + 1 } ) _ { v u } = \sum _ { \omega \in \mathcal { W } _ { k + 1 } ( u , v ) } ( L _ { \mathcal { F } } ) _ { \omega } } \end{array}$ , completing the induction.

For two different nodes $v , u ,$ we define their distance, denoted by $\operatorname { d i s t } _ { \mathcal { G } } ( u , v )$ , as the minimum $k \geq 0$ such that $\mathcal { W } _ { k } ( u , v ) \neq \emptyset$ if they lie in the same connected component, and $\mathrm { d i s t } _ { \mathcal { G } } ( u , v ) = \infty$ otherwise. Also, dis ${ \bf \nabla } _ { \mathcal { G } } ( u , u ) = 0$

Proof of Proposition 1 . For $r = \mathrm { d i s t } _ { \mathcal { G } } ( u , v ) < \infty$ , we have $\mathcal { W } _ { k } ( u , v ) = \emptyset$ for $k < r ,$ and hence $( L _ { \mathcal { F } } ^ { k } ) _ { v u } = 0$ for these values of k. Taking the $( v , u )$ -block of the series defining $\mathscr { C } _ { \mathcal { F } } ( t )$ gives

$$
J _ { v  u } ^ { \mathcal { F } } ( t ) = \sum _ { k = 0 } ^ { \infty } ( - 1 ) ^ { k } \frac { t ^ { 2 k } } { ( 2 k ) ! } ( L _ { \mathcal { F } } ^ { k } ) _ { v u } = \sum _ { k = r } ^ { \infty } ( - 1 ) ^ { k } \frac { t ^ { 2 k } } { ( 2 k ) ! } \sum _ { \omega \in \mathcal { W } _ { k } ( u , v ) } ( L _ { \mathcal { F } } ) _ { \omega } .
$$

At $k = r ,$ , these walks are precisely the shortest paths. Indeed, a stationary step or a repeated vertex would allow a segment to be removed and thereby produce a shorter walk from u to v, contrary to the definition of r.

For the last statement, if u and v lie in different connected components, then $\mathcal { W } _ { k } ( u , v ) = \emptyset$ for every k. Every block $( L _ { \mathcal { F } } ^ { k } ) _ { v u }$ therefore vanishes, and $J _ { v  u } ^ { \mathcal { F } } ( t ) = 0$ for all t. □

## A.5 PROOF OF PROPOSITION 2

This result applies to the positive semidefinite self-adjoint sheaf Laplacian and uses the orthogonal projectors onto the eigenspaces $P _ { \lambda }$ indexed by distinct eigenvalues given by the finite-dimensional spectral theorem (Halmos, 1974). These projectors satisfy $L _ { \mathcal { F } } P _ { \lambda } \overset { = } { = } \lambda P _ { \lambda }$ , as well as the identities $P _ { \lambda } ^ { * } = P _ { \lambda } , P _ { \lambda } P _ { \mu } = \delta _ { \lambda \mu } P _ { \lambda }$ , and $\begin{array} { r } { \sum _ { \lambda \in \mathrm { s p e c } ( L _ { \mathcal { F } } ) } P _ { \lambda } = I _ { \mathcal { H } } } \end{array}$

ProofofProposition 2 . Substituting the above identities into the power series gives

$$
\begin{array} { l } { { \displaystyle { \mathcal C } _ { { \mathcal F } } ( t ) P _ { \lambda } = \sum _ { k = 0 } ^ { \infty } ( - 1 ) ^ { k } \frac { t ^ { 2 k } } { ( 2 k ) ! } L _ { { \mathcal F } } ^ { k } P _ { \lambda } } } \\ { ~ } \\ { { \displaystyle ~ = \left( \sum _ { k = 0 } ^ { \infty } ( - 1 ) ^ { k } \frac { t ^ { 2 k } \lambda ^ { k } } { ( 2 k ) ! } \right) P _ { \lambda } = \cos ( t \sqrt \lambda ) P _ { \lambda } } . } \end{array}
$$

Since the spectral projectors sum to $I _ { \mathcal { H } }$ , also

$$
\mathcal { C } _ { \mathcal { F } } ( t ) = \mathcal { C } _ { \mathcal { F } } ( t ) \sum _ { \lambda } P _ { \lambda } = P _ { 0 } + \sum _ { \lambda > 0 } \cos ( t \sqrt \lambda ) P _ { \lambda } ,
$$

where $P _ { 0 }$ is the projector onto ker $L _ { \mathcal { F } }$ , with $P _ { 0 } = 0$ when $0 \not \in \mathrm { s p e c } ( L _ { \mathcal { F } } )$ . Applying $\pi _ { v }$ on the left and $\iota _ { u }$ on the right proves the result. □

## A.6 PROOF OF THEOREM 1

For the proof of this result, we start with two preliminary lemmas. First, Lemma 4 expresses the discrete response as a polynomial in $L _ { \mathcal { F } }$ . Afterwards, we solve the corresponding scalar recurrence to obtain the spectral formula and boundedness in Lemma 5. The spectral formula and boundedness in Theorem 1 then follow directly. To prove that a response which is non-zero at some step cannot converge to zero, we compute the Cesaro mean of its squared Hilbert–Schmidt norm. Finally, \` Corollary 1 shows that the discrete spectral coefficients converge to their continuous counterparts as $h  0$ with $\ell h = T$ fixed.

Let $( X ^ { ( \ell ) } , V ^ { ( \ell ) } )$ denote the underlying trajectory of Equation (3) with fixed restriction maps, identity weights, and no dissipation or external forcing, i.e.,

$$
{ \pmb V } ^ { ( \ell + 1 ) } = { \pmb V } ^ { ( \ell ) } - h L _ { \mathcal { F } } { \pmb X } ^ { ( \ell ) } , \qquad { \pmb X } ^ { ( \ell + 1 ) } = { \pmb X } ^ { ( \ell ) } + h { \pmb V } ^ { ( \ell + 1 ) } ,\tag{14}
$$

and let $( \widetilde { X } ^ { \left( \ell \right) } , \widetilde { V } ^ { \left( \ell \right) } )$ denote the trajectory obtained by replacing $\overline { { \boldsymbol { X } } }$ with $\overline { { \boldsymbol { X } } } + \iota _ { u } \boldsymbol { W }$ , while keeping the initial velocity fixed. Write

$$
\Delta { \cal X } ^ { ( \ell ) } : = \widetilde { { \cal X } } ^ { ( \ell ) } - { \cal X } ^ { ( \ell ) } , \qquad \Delta { \cal V } ^ { ( \ell ) } : = \widetilde { { \cal V } } ^ { ( \ell ) } - { \cal V } ^ { ( \ell ) } ,
$$

so that $\Delta \overline { { X } } = \iota _ { u } W$ and $\Delta \overline { { V } } = 0$ . The following lemma expresses the resulting response as a polynomial in $L _ { \mathcal { F } }$

Lemma 4. $L e t p _ { 0 , h } ( z ) = 1 , p _ { 1 , h } ( z ) = 1 - h ^ { 2 } z , a n d , f o r \ell \ge 1$

$$
p _ { \ell + 1 , h } ( z ) = ( 2 - h ^ { 2 } z ) p _ { \ell , h } ( z ) - p _ { \ell - 1 , h } ( z ) .
$$

Then, for every integer $\ell \geq 0 , \Delta { X } ^ { ( \ell ) } = p _ { \ell , h } ( L _ { \mathcal { F } } ) \iota _ { u } W$ , and $J _ { v  u } ^ { \mathcal { F } } [ \ell , h ] = ( p _ { \ell , h } ( L _ { \mathcal { F } } ) ) _ { v u }$ , where deg $p _ { \ell , h } \leq \ell .$ In particular, $i f \mathrm { d i s t } _ { \mathcal { G } } ( u , v ) > \ell ,$ then $J _ { v  u } ^ { \mathcal { F } } [ \ell , h ] = 0$

Proof. Taking differences in the update Equation (14) gives

$$
\Delta { \cal V } ^ { ( \ell + 1 ) } = \Delta { \cal V } ^ { ( \ell ) } - h L _ { \mathcal { F } } \Delta { \cal X } ^ { ( \ell ) } , \qquad \Delta { \cal X } ^ { ( \ell + 1 ) } = \Delta { \cal X } ^ { ( \ell ) } + h \Delta { \cal V } ^ { ( \ell + 1 ) } .
$$

For $\ell = 0$ , this becomes $\Delta \boldsymbol { X } ^ { ( 1 ) } = ( I _ { \mathcal { H } } - h ^ { 2 } L _ { \mathcal { F } } ) \Delta \boldsymbol { X } ^ { ( 0 ) } = p _ { 1 , h } ( L _ { \mathcal { F } } ) \iota _ { u } \boldsymbol { W } .$

For $\ell \geq 1$ , the above update gives $h \Delta V ^ { ( \ell ) } = \Delta X ^ { ( \ell ) } - \Delta X ^ { ( \ell - 1 ) }$ . We substitute this into the next one, giving

$$
\Delta { \pmb X } ^ { ( \ell + 1 ) } = \Delta { \pmb X } ^ { ( \ell ) } + h \Delta { \pmb V } ^ { ( \ell ) } - h ^ { 2 } L _ { \mathcal { F } } \Delta { \pmb X } ^ { ( \ell ) } = ( 2 I _ { \mathcal { H } } - h ^ { 2 } L _ { \mathcal { F } } ) \Delta { \pmb X } ^ { ( \ell ) } - \Delta { \pmb X } ^ { ( \ell - 1 ) } .
$$

Together with the initial values, this recurrence implies by induction that $\Delta { X ^ { ( \ell ) } } = p _ { \ell , h } ( L _ { \mathcal { F } } ) \iota _ { u } W$ After applying $\pi _ { v } .$ , and using that the identity holds for every $W \in { \mathcal { F } } ( u )$ , this gives $J _ { v  u } ^ { \mathcal { F } } [ \ell , h ] =$ $\pi _ { v } p _ { \ell , h } ( L _ { \mathcal { F } } ) \iota _ { u } = \left( p _ { \ell , h } ( L _ { \mathcal { F } } ) \right) _ { v u }$

The recurrence defining $p _ { \ell , h }$ directly implies that deg $p _ { \ell , h } \leq \ell .$

If dis $\mathrm { t } _ { \mathcal { G } } ( u , v ) > \ell ,$ then $\mathcal { W } _ { j } ( u , v ) = \emptyset$ for every $j \leq \ell .$ . By Lemma $3 , ( L _ { \mathcal { F } } ^ { j } ) _ { v u } = 0$ for all such $j ,$ and hence $J _ { v  u } ^ { \mathcal { F } } [ \ell , h ] = ( \bar { p _ { \ell , h } } ( L _ { \mathcal { F } } ) ) _ { v u } = 0$ □

We next solve the scalar recurrence defining $p _ { \ell , h }$ . For this and the subsequent proof, we use the following trigonometric identities, valid for $a , b , \theta \in \mathbb { R }$

$$
\cos ( a + b ) + \cos ( a - b ) = 2 \cos ( a ) \cos ( b ) ,\tag{15}
$$

$$
\sin ( a + b ) - \sin ( a - b ) = 2 \cos ( a ) \sin ( b ) ,\tag{16}
$$

$$
\cos \theta = 1 - 2 \sin ^ { 2 } \left( { \frac { \theta } { 2 } } \right) , \qquad \cos ^ { 2 } \theta = { \frac { 1 + \cos ( 2 \theta ) } { 2 } } .\tag{17}
$$

Lemma 5. Let $h > 0$ and $\lambda \geq 0$ satisfy $h ^ { 2 } \lambda < 4 ,$ , and define $\theta _ { h } ( \lambda ) : = 2$ arcsin $\left( { \frac { h { \sqrt { \lambda } } } { 2 } } \right)$ . Then, for every integer $\ell \geq 0$

$$
p _ { \ell , h } ( \lambda ) = \frac { \cos \big ( ( \ell + \frac { 1 } { 2 } ) \theta _ { h } ( \lambda ) \big ) } { \cos ( \theta _ { h } ( \lambda ) / 2 ) } .
$$

Moreover, $\begin{array} { r } { \left| p _ { \ell , h } ( \lambda ) \right| \le \frac { 1 } { \sqrt { 1 - h ^ { 2 } \lambda / 4 } } } \end{array}$

Proof. Since $0 \leq h \sqrt { \lambda } / 2 < 1$ , we have $\theta _ { h } ( \lambda ) \in [ 0 , \pi )$ . By definition of $\theta _ { h } ( \lambda )$ , also sin $\begin{array} { r l r } {  { ( \frac { \theta _ { h } ( \lambda ) } { 2 } ) = } } \end{array}$ $\frac { h \sqrt { \lambda } } { 2 }$ . Using Equation (17), we find

$$
2 \cos \theta _ { h } ( \lambda ) = 2 \left( 1 - 2 \sin ^ { 2 } \left( \frac { \theta _ { h } ( \lambda ) } { 2 } \right) \right) = 2 - h ^ { 2 } \lambda .\tag{18}
$$

Define

$$
q _ { \ell } : = \frac { \cos ( ( \ell + \frac { 1 } { 2 } ) \theta _ { h } ( \lambda ) ) } { \cos ( \theta _ { h } ( \lambda ) / 2 ) } .
$$

For $\ell = 0 , q _ { 0 } = 1 = p _ { 0 , h } ( \boldsymbol { \lambda } )$

Applying Equation (15) to $a = \theta _ { h } ( \lambda ) , b = \theta _ { h } ( \lambda ) / 2$ and dividing by cos $( \theta _ { h } ( \lambda ) / 2 )$ gives

$$
\frac { \cos ( 3 \theta _ { h } ( \lambda ) / 2 ) } { \cos ( \theta _ { h } ( \lambda ) / 2 ) } = 2 \cos \theta _ { h } ( \lambda ) - 1 ,
$$

so that $q _ { 1 } = 2 \cos \theta _ { h } ( \lambda ) - 1 = 1 - h ^ { 2 } \lambda = p _ { 1 , h } ( \lambda )$ , where we used Equation (18) and the definition of $p _ { 1 , h } ( \lambda )$

For $\ell \geq 1$ , again applying Equation (15) with $\begin{array} { r } { a = ( \ell + \frac 1 2 ) \theta _ { h } ( \lambda ) } \end{array}$ and $b = \theta _ { h } ( \lambda )$ gives

$$
\begin{array} { r } { \cos \left( ( \ell + \frac { 3 } { 2 } ) \theta _ { h } ( \lambda ) \right) + \cos \left( ( \ell - \frac { 1 } { 2 } ) \theta _ { h } ( \lambda ) \right) = 2 \cos \theta _ { h } ( \lambda ) \cos \left( ( \ell + \frac { 1 } { 2 } ) \theta _ { h } ( \lambda ) \right) . } \end{array}
$$

Dividing this expression by $\cos ( \theta _ { h } ( \lambda ) / 2 )$ , and using the definition of $q _ { \ell } ,$ , we find

$$
q _ { \ell + 1 } + q _ { \ell - 1 } = 2 \cos \theta _ { h } ( \lambda ) q _ { \ell } .
$$

By Equation (18), this can be rearranged to $q _ { \ell + 1 } = 2 \cos { \theta _ { h } } ( \lambda ) q _ { \ell } - q _ { \ell - 1 } = ( 2 - h ^ { 2 } \lambda ) q _ { \ell } - q _ { \ell - 1 }$

In particular, $q _ { \ell }$ has the same initial values and recurrence as $p _ { \ell , h } ( \boldsymbol { \lambda } )$ , and hence

$$
p _ { \ell , h } ( \lambda ) = \frac { \cos \big ( ( \ell + \frac { 1 } { 2 } ) \theta _ { h } ( \lambda ) \big ) } { \cos ( \theta _ { h } ( \lambda ) / 2 ) } .
$$

For the final statement, we use Equations (17) and (18) to find

$$
\cos ^ { 2 } ( \theta _ { h } ( \lambda ) / 2 ) = \frac { 1 + \cos \theta _ { h } ( \lambda ) } { 2 } = 1 - \frac { h ^ { 2 } \lambda } { 4 } .\tag{19}
$$

Hence, $\begin{array} { r } { \cos \left( \frac { \theta _ { h } ( \lambda ) } { 2 } \right) = \sqrt { 1 - \frac { h ^ { 2 } \lambda } { 4 } } > 0 , \mathrm { a n d } \left| p _ { \ell , h } ( \lambda ) \right| \leq \frac { 1 } { \cos ( \theta _ { h } ( \lambda ) / 2 ) } = \frac { 1 } { \sqrt { 1 - h ^ { 2 } \lambda / 4 } } . } \end{array}$

Proof of Theorem 1 . By Lemma 4 and the spectral decomposition of $L _ { \mathcal { F } }$ ,

$$
J _ { v  u } ^ { \mathcal { F } } [ \ell , h ] = ( p _ { \ell , h } ( L _ { \mathcal { F } } ) ) _ { v u } = \sum _ { \lambda } p _ { \ell , h } ( \lambda ) ( P _ { \lambda } ) _ { v u } .
$$

Since $h ^ { 2 } \lambda _ { \operatorname* { m a x } } ( L _ { \mathcal { F } } ) < 4 ,$ Lemma 5 gives Equation (12).

The fact that $J _ { v  u } ^ { \mathcal { F } } [ \ell , h ] = 0$ whenever dis $\begin{array} { r } { \mathrm { t } _ { \mathcal { G } } ( u , v ) > \ell } \end{array}$ follows again from Lemma 4.

The bound in Lemma 5 gives

$$
\| J _ { v  u } ^ { \mathcal { F } } [ \ell , h ] \| \leq \sum _ { \lambda } \frac { \| ( P _ { \lambda } ) _ { v u } \| } { \sqrt { 1 - h ^ { 2 } \lambda / 4 } } .
$$

The right-hand side is finite under the step-size assumption and independent of ℓ, showing that the response is bounded as the propagation depth varies.

For the final statement, fix $u , v \in \mathcal { V }$ and write $J _ { \ell } : = J _ { v  u } ^ { \mathcal { F } } [ \ell , h ]$ . To exclude that $J _ { \ell } \longrightarrow 0$ , it suffices to show that the Cesaro average\` $\begin{array} { r } { \frac { 1 } { N } \sum _ { \ell = 0 } ^ { N - 1 } \| J _ { \ell } \| _ { \mathrm { H S } } ^ { 2 } } \end{array}$ has a strictly positive limit whenever the response is non-zero at some step.

We use the Hilbert–Schmidt norm as introduced in Section A.1 to expand the inner products

$$
\begin{array} { r l } { \displaystyle \| J _ { \ell } \| _ { \mathrm { H S } } ^ { 2 } = \| ( P _ { 0 } ) _ { v u } \| _ { \mathrm { H S } } ^ { 2 } + 2 \displaystyle \sum _ { \lambda > 0 } p _ { \ell , h } ( \lambda ) \left. ( P _ { 0 } ) _ { v u } , ( P _ { \lambda } ) _ { v u } \right. _ { \mathrm { H S } } } & { } \\ { + \displaystyle \sum _ { \lambda , \mu > 0 } p _ { \ell , h } ( \lambda ) p _ { \ell , h } ( \mu ) \left. ( P _ { \lambda } ) _ { v u } , ( P _ { \mu } ) _ { v u } \right. _ { \mathrm { H S } } } & { } \\ { = \| ( P _ { 0 } ) _ { v u } \| _ { \mathrm { H S } } ^ { 2 } + 2 \displaystyle \sum _ { \lambda > 0 } p _ { \ell , h } ( \lambda ) \left. ( P _ { 0 } ) _ { v u } , ( P _ { \lambda } ) _ { v u } \right. _ { \mathrm { H S } } + \displaystyle \sum _ { \lambda > 0 } p _ { \ell , h } ( \lambda ) ^ { 2 } \| ( P _ { \lambda } ) _ { v u } \| _ { \mathrm { H S } } ^ { 2 } } & { } \\ { + \displaystyle 2 \displaystyle \sum _ { 0 \le \lambda \le \mu } p _ { \ell , h } ( \lambda ) p _ { \ell , h } ( \mu ) \left. ( P _ { \lambda } ) _ { v u } , ( P _ { \mu } ) _ { v u } \right. _ { \mathrm { H S } } . } & { } \end{array}
$$

The last equality separates the diagonal terms $\lambda = \mu$ and uses symmetry of the real Hilbert–Schmidt inner product.

We analyze the three oscillatory sums separately, using a shared averaging identity. For fixed $\omega \not \in$ $2 \pi \mathbb { Z } ,$ applying Equation (16) with $a = ( \ell + { \frac { 1 } { 2 } } )$ ω and $b = \omega / 2$ gives

$$
\begin{array} { r } { \sin ( ( \ell + 1 ) \omega ) - \sin ( \ell \omega ) = 2 \sin ( \omega / 2 ) \cos ( ( \ell + \frac { 1 } { 2 } ) \omega ) . } \end{array}
$$

Summing over $\ell = 0 , \dots , N - 1$ , the left-hand side telescopes. Consequently,

$$
T _ { N } ( \omega ) : = \frac { 1 } { N } \sum _ { \ell = 0 } ^ { N - 1 } \cos ( ( \ell + \textstyle \frac 1 2 ) \omega ) = \frac { \sin ( N \omega ) } { 2 N \sin ( \omega / 2 ) } \longrightarrow 0 .\tag{20}
$$

Here si $\iota ( \omega / 2 ) \neq 0$ is fixed, while $| \sin ( N \omega ) | \leq 1$

Term 1. For $\lambda > 0$ , we have $\theta _ { h } ( \lambda ) \in ( 0 , \pi )$ . Lemma 5 and Equation (20) therefore give

$$
\frac { 1 } { N } \sum _ { \ell = 0 } ^ { N - 1 } p _ { \ell , h } ( \lambda ) = \frac { T _ { N } ( \theta _ { h } ( \lambda ) ) } { \cos ( \theta _ { h } ( \lambda ) / 2 ) } \longrightarrow 0 .
$$

Term 2. For $\lambda > 0$ , using Equation (17), we obtain

$$
\begin{array} { r l r } & { } & { \displaystyle \frac { 1 } { N } \sum _ { \ell = 0 } ^ { N - 1 } p _ { \ell , h } ( \lambda ) ^ { 2 } = \frac { 1 } { N \cos ^ { 2 } ( \theta _ { h } ( \lambda ) / 2 ) } \sum _ { \ell = 0 } ^ { N - 1 } \cos ^ { 2 } ( ( \ell + \frac { 1 } { 2 } ) \theta _ { h } ( \lambda ) ) } \\ & { } & { \displaystyle = \frac { 1 + T _ { N } ( 2 \theta _ { h } ( \lambda ) ) } { 2 \cos ^ { 2 } ( \theta _ { h } ( \lambda ) / 2 ) } \longrightarrow \frac { 1 } { 2 \cos ^ { 2 } ( \theta _ { h } ( \lambda ) / 2 ) } . } \end{array}
$$

Here $T _ { N } ( 2 \theta _ { h } ( \lambda ) )  0$ because of Equation (20) as $2 \theta _ { h } ( \lambda ) \in ( 0 , 2 \pi )$

Term 3. For distinct positive eigenvalues $\lambda , \mu ,$ applying Equation (15) gives

$$
\frac { 1 } { N } \sum _ { \ell = 0 } ^ { N - 1 } p _ { \ell , h } ( \lambda ) p _ { \ell , h } ( \mu ) = \frac { T _ { N } ( \theta _ { h } ( \lambda ) - \theta _ { h } ( \mu ) ) + T _ { N } ( \theta _ { h } ( \lambda ) + \theta _ { h } ( \mu ) ) } { 2 \cos ( \theta _ { h } ( \lambda ) / 2 ) \cos ( \theta _ { h } ( \mu ) / 2 ) } \longrightarrow 0 .
$$

Indeed, as $\theta _ { h }$ is strictly increasing,

$$
\theta _ { h } ( \lambda ) - \theta _ { h } ( \mu ) \notin 2 \pi \mathbb { Z } , \qquad \theta _ { h } ( \lambda ) + \theta _ { h } ( \mu ) \in ( 0 , 2 \pi ) .
$$

So Equation (20) applies to both cosine terms.

Since the spectrum is finite, combining these limits gives

$$
\operatorname* { l i m } _ { N \to \infty } \frac { 1 } { N } \sum _ { \ell = 0 } ^ { N - 1 } \| J _ { \ell } \| _ { \mathrm { H S } } ^ { 2 } = \| ( P _ { 0 } ) _ { v u } \| _ { \mathrm { H S } } ^ { 2 } + \sum _ { \lambda > 0 } \frac { \| ( P _ { \lambda } ) _ { v u } \| _ { \mathrm { H S } } ^ { 2 } } { 2 \cos ^ { 2 } ( \theta _ { h } ( \lambda ) / 2 ) } .
$$

If $J _ { \ell } \neq 0$ at some step, its spectral representation implies that at least one block $( P _ { \lambda } ) _ { v u }$ is non-zero.   
The right-hand side is therefore strictly positive.

If $J _ { \ell }$ converged to zero in operator norm, the equivalence of norms in Appendix A.1 would imply $\Vert J _ { \ell } \Vert _ { \mathrm { H S } } ^ { 2 }  \tilde { 0 }$ . Its averages would then also converge to zero, contradicting the positive limit above. Thus, a response that is non-zero at some step cannot converge to zero as $\ell  \infty .$ □

To conclude, we show that the discrete spectral response converges to its continuous counterpart when the step size tends to zero at fixed propagation time.

Corollary 1. Let $L _ { \mathcal { F } }$ be fixed, self-adjoint, and positive semidefinite. Fix $T > 0$ and set $h = T / \ell .$ Then, for every eigenvalue λ of L<sub>F</sub>, lim $_ { \ell \to \infty } p _ { \ell , h } ( \lambda ) = \cos ( T \sqrt { \lambda } )$ . Consequently, for every pair of nodes $u , v ,$

$$
\operatorname* { l i m } _ { \ell \to \infty } \| J _ { v  u } ^ { \mathcal { F } } [ \ell , T / \ell ] - J _ { v  u } ^ { \mathcal { F } } ( T ) \| = 0 .
$$

Proof. Since $h = T / \ell \to 0$ , the stability condition $h ^ { 2 } \lambda _ { \operatorname* { m a x } } ( L _ { \mathcal { F } } ) < 4$ holds for all sufficiently large ℓ. We may therefore use the spectral coefficient from Lemma 5.

For $\lambda = 0 .$ , the coefficient equals 1, so the limit statement holds.

For $\lambda > 0 .$ , we have $\begin{array} { r } { \frac { \theta _ { h } ( \lambda ) } { h } \ = \ \sqrt { \lambda } \frac { \arcsin ( h \sqrt { \lambda } / 2 ) } { h \sqrt { \lambda } / 2 } \ \longrightarrow \ \sqrt { \lambda } ; } \end{array}$ , where the last step uses the identity lim $. x {  } 0 ~ \frac { \arcsin ( x ) } { x } = 1$ . Since $\ell h = T$ , it follows that $\begin{array} { r } { ( \ell + \frac 1 2 ) \theta _ { h } ( \lambda ) = ( T + \frac h 2 ) \frac { \theta _ { h } ( \lambda ) } { h } \longrightarrow T \sqrt { \lambda } . } \end{array}$ Using Equation (19), also cos $\left( \theta _ { h } ( \lambda ) / 2 \right) = \sqrt { 1 - h ^ { 2 } \lambda / 4 } \longrightarrow 1$ . Continuity of cosine therefore gives

$$
p _ { \ell , h } ( \lambda ) = \frac { \cos \left( ( \ell + \frac { 1 } { 2 } ) \theta _ { h } ( \lambda ) \right) } { \cos \left( \theta _ { h } ( \lambda ) / 2 \right) } \longrightarrow \cos ( T \sqrt { \lambda } ) .
$$

For the final statement, we have that $\| J _ { v  u } ^ { \mathcal { F } } [ \ell , h ] - J _ { v  u } ^ { \mathcal { F } } ( T ) \|$ $\leq$ $\begin{array} { r l } { \sum _ { \lambda } \left| p _ { \ell , h } ( \lambda ) - \cos ( T \sqrt { \lambda } ) \right| \| ( P _ { \lambda } ) _ { v u } \| } & { { } } \end{array}$ . For $\ell \ \longrightarrow \ \infty$ , this last term converges to zero by the above. □

## B ADDITIONAL EXPERIMENTAL DETAILS

This section provides Section 5 with details on restriction-map parametrizations, training protocols, and hyperparameter selection.

## B.1 RESTRICTION-MAP PARAMETRIZATIONS

We consider four restriction-map parametrizations for ONDA, following (Bodnar et al., 2022; Borgi et al., 2026). The maps $\mathcal { F } _ { v , e } : \bar { \mathcal { F } } ( v )  \mathcal { F } ( e )$ act between d-dimensional node and edge stalks and take the following forms:

• Diagonal: The restriction map is constrained to $\mathcal { F } _ { v , e } = \mathrm { d i a g } ( \alpha _ { v , e } )$ . Each stalk coordinate is therefore scaled independently, without mixing information across coordinates. This requires d scalar coefficients per restriction map.

• Orthogonal. The restriction map satisfies $\mathcal { F } _ { v , e } ^ { \top } \mathcal { F } _ { v , e } = I _ { d }$ . These maps allow for coordinate mixing while preserving Euclidean norms and inner products.

• Low-Rank. The restriction map is represented in factorized form, $\mathcal { F } _ { v , e } = U _ { v , e } V _ { v , e } ^ { \top } ,$ with $U _ { v , e } , V _ { v , e } \in \mathbb { R } ^ { d \times r }$ and $1 \leq r < d .$ . This permits coordinate mixing through an intermediate space of dimension r, with rank $( \mathcal { F } _ { v , e } ) \leq r$

• General. The restriction map is an unconstrained matrix $\mathcal { F } _ { v , e } \in \mathbb { R } ^ { d \times d }$ . This provides the most flexible local transport, allowing arbitrary linear transformations between stalks, at the cost of $\mathcal { O } ( d ^ { 2 } )$ storage and application complexity.

We stress that these parametrizations only affect the restriction maps used to construct $L _ { \mathcal { F } }$ , while retaining the same ONDA update rule and oscillatory dynamics.

## B.2 EMPLOYED BASELINES

In our experiments, the performance of our method is compared with various state-of-the-art GNN and SNN baselines from the literature. Specifically, we consider:

• Classical MPNN-based methods, i.e., ChebNet (Defferrard et al., 2016), GCN (Kipf & Welling, 2017), GraphSAGE (Hamilton et al., 2017), GAT (Velickoviˇ c et al., 2018), Gat-´ edGCN (Bresson & Laurent, 2018), GIN (Xu et al., 2019), GINE (Hu et al., 2020), GC-NII (Chen et al., 2020), and CoGNN (Finkelshtein et al., 2024);

• Heterophily-specific models, i.e., H2GCN (Zhu et al., 2020), CPGNN (Zhu et al., 2021), FAGCN (Bo et al., 2021), GPR-GNN (Chien et al., 2021), FSGNN (Maurya et al., 2022), GloGNN (Li et al., 2022), GBK-GNN (Du et al., 2022), and JacobiConv (Wang & Zhang, 2022);

• DE-GNNs, i.e., GraphCON (Rusch et al., 2022), A-DGN (Gravina et al., 2023), SWAN (Gravina et al., 2025), PH-DGN (Heilig et al., 2025), SONAR (Trenta et al., 2025);

• Graph Transformers, i.e., Transformer (Vaswani et al., 2017; Dwivedi & Bresson, 2021), GT (Shi et al., 2021b), GPS (Rampa´sek et al., 2022), GOAT (Kong et al., 2023), Ex-ˇ phormer (Shirzad et al., 2023), NAGphormer Chen et al. (2023), GRIT(Ma et al., 2023);

• Higher-Order GNNs, i.e., DRew (Gutteridge et al., 2023);

• Sheaf Neural Networks, i.e., NSD (Bodnar et al., 2022), BuNN (Bamberger et al., 2025), and CSNN (Ribeiro et al., 2026).

## B.3 HYPERPARAMETER SEARCH SPACES FOR THE EXPERIMENTS IN SECTION 5

The hyperparameter search spaces used for ONDA are reported in Table 3 for the ECHO benchmark, Table 4 for the Barbell benchmark, and Table 5 for the Graph Transfer benchmark. To select the best hyperparameters, we perform a Bayesian optimization in the chosen hyperparameter space.

Table 3: Hyperparameter search space for the ECHO Benchmark.
<table><tr><td>Hyperparameter</td><td>Candidate values</td></tr><tr><td>Hidden dimension H</td><td>[36,252]</td></tr><tr><td>Stalk dimension d</td><td>{2, 3, 4}</td></tr><tr><td>ONDA Blocks B Inner iterations L</td><td>{1, 2, 3, 4}</td></tr><tr><td>Step size h</td><td>[1, 40]  $[ 1 0 ^ { - 4 } , 0 . 5 ]$ </td></tr><tr><td>Learning rate</td><td> $[ 1 0 ^ { - 5 } , 1 0 ^ { - 2 } ]$ </td></tr><tr><td>Operator</td><td>{classical, normalized, directional }</td></tr><tr><td>Sheaf type</td><td></td></tr><tr><td></td><td>{general, diagonal, orthogonal}</td></tr><tr><td>Dissipation</td><td>{False, True}</td></tr><tr><td>Forcing</td><td>{False, True}</td></tr><tr><td>Fixed Restriction maps</td><td>{False, True}</td></tr></table>

Table 4: Hyperparameter search space for the Barbell benchmark.
<table><tr><td>Hyperparameter</td><td>Candidate values</td></tr><tr><td>Hidden dimension H</td><td>{128, 192}</td></tr><tr><td>Stalk dimension d</td><td>{2,6}</td></tr><tr><td>ONDA Blocks B</td><td>{1, 2, 4}</td></tr><tr><td>Inner iterations L</td><td>{8, 10, 12, 14, 16, 20}</td></tr><tr><td>Step size h</td><td>{0.1, 0.2, 0.25, 0.33, 0.5, 0.6, 0.66, 0.75, 1.0}</td></tr><tr><td>Learning rate</td><td>{3 × 10−4}</td></tr><tr><td>Operator</td><td>{classical}</td></tr><tr><td>Sheaf type</td><td>{general, diagonal, orthogonal, low-rank}</td></tr><tr><td>Dissipation</td><td>{False}</td></tr><tr><td>Forcing</td><td>{False}</td></tr><tr><td>Fixed Restriction maps</td><td>{True}</td></tr></table>

## C ADDITIONAL RESULTS

## C.1 RESTRICTION MAP PARAMETRIZATION ABLATION

In this section, we investigate how the parametrization of the restriction maps affects information propagation on the Barbell benchmark from Section 5.2. In Table 2, we compare four parametrizations, i.e., diagonal, orthogonal, low-rank, and general maps, which differ in the expressivity of the learned transport between stalks. To disentangle the effect of the restriction-map parametrization from that of the oscillatory dynamics, we evaluate the same variants for both ONDA and NSD. This allows us to assess how the choice of sheaf transport affects information transfer across the severe topological bottleneck of the Barbell graphs. To focus on the propagation mechanism, we consider ONDA configurations with a single-layer MLP-based restriction-map learner, without forcing or dissipation.

Table 5: Hyperparameter search space for the graph transfer task.
<table><tr><td>Hyperparameter</td><td>Candidate values</td></tr><tr><td>Hidden dimension H</td><td>64 (Gravina et al., 2025)</td></tr><tr><td>Stalk dimension d</td><td>{1, 2, 4}</td></tr><tr><td>ONDA Blocks B</td><td>{1, 2}</td></tr><tr><td>Inner iterations L</td><td> $\mathrm { d i s t a n c e \times \left\{ 1 , 2 \right\} } / B$ </td></tr><tr><td>Step size h</td><td> $\{ 0 . 0 5 , 0 . 1 , \mathrm { { \dot { 0 } } . 5 , \mathrm { { \dot { 1 } } . 0 } } \}$ </td></tr><tr><td>Learning rate</td><td> $\bar { 1 } 0 ^ { - 3 } \ : ( \mathrm { G r a v i n a e t \ a l . , } 2 0 2 5 )$ </td></tr><tr><td>Operator</td><td>{classical, normalized, directional }</td></tr><tr><td>Sheaf type</td><td>{general, diagonal}</td></tr><tr><td>Dissipation</td><td>{False, True}</td></tr><tr><td>Forcing</td><td>{False, True}</td></tr><tr><td>Fixed Restriction maps</td><td></td></tr><tr><td></td><td>{False, True}</td></tr></table>

Results. Across all restriction-map parametrizations and graph sizes, ONDA substantially improves over the corresponding NSD variant, showing that its effectiveness is robust to the particular choice of sheaf transport. All four ONDA parametrizations solve the task at $N = 1 0$ , with the diagonal and orthogonal variants remaining below the solving threshold across all tested sizes. Among them, diagonal maps provide the most consistent behavior, achieving the lowest mean error at both $N = 2 0$ and $N = 5 0$ . The remaining parametrizations exhibit a stronger dependence on graph size. Notably, the most expressive general maps are not systematically preferable to more constrained parametrizations. Overall, these results indicate that richer restriction-map expressivity alone is not sufficient for effective information transfer across severe bottlenecks, while the oscillatory propagation of ONDA consistently improves upon diffusion-based NSD under matched restriction-map parametrizations.

Table 6: Barbell additional results. Test MSE $( \times 1 0 ^ { - 3 } ; \downarrow )$ on the Barbell task for graph sizes $N \in$ {10, 20, 50}. All models report the results over the best configuration selected by validation, as mean<sub>±</sub> over 5 seeds. For each column, we report the best, second-best, and third-best results.
<table><tr><td>Model</td><td> $N = 1 0 \downarrow$ </td><td> $\mathbf { N } = \mathbf { 2 0 } \downarrow$ </td><td> $\mathbf { N } = \mathbf { 5 0 } \downarrow$ </td></tr><tr><td>GCN</td><td> $8 8 4 . 2 0 0 { \scriptstyle \pm 6 . 9 0 0 }$ </td><td> $9 6 1 . 1 0 0 { \scriptstyle \pm 1 2 . 3 0 0 }$ </td><td> $8 6 0 . 0 0 0 { \scriptstyle \pm 3 5 . 0 0 0 }$ </td></tr><tr><td>GAT</td><td> $7 . 9 0 0 { \scriptstyle \pm 2 . 1 0 0 }$ </td><td> $4 8 7 . 0 0 0 { \scriptstyle \pm 3 4 2 . 2 0 0 }$ </td><td> $8 5 2 . 0 0 0 { \scriptstyle \pm 2 9 . 0 0 0 }$ </td></tr><tr><td>GraphSAGE</td><td> $4 5 5 . 5 0 0 { \scriptstyle \pm 1 9 9 . 6 0 0 }$ </td><td> $9 9 8 . 2 0 0 { \scriptstyle \pm 6 . 5 0 0 }$ </td><td> $8 7 6 . 0 0 0 { \scriptstyle \pm 3 2 . 0 0 0 }$ </td></tr><tr><td>GIN</td><td> $9 0 0 . 5 0 0 { \scriptstyle \pm 4 7 . 9 0 0 }$ </td><td> $9 7 0 . 6 0 0 { \scriptstyle \pm 3 . 9 0 0 }$ </td><td> $9 8 5 . 0 0 0 { \scriptstyle \pm 1 2 3 . 0 0 0 }$ </td></tr><tr><td>GraphTransformer</td><td> $6 7 0 . 3 0 0 { \scriptstyle \pm 3 0 6 . 8 0 0 }$ </td><td> $1 0 4 1 . 2 0 0 { \scriptstyle \pm 4 6 . 9 0 0 }$ </td><td> $8 3 1 . 0 0 0 { \scriptstyle \pm 1 3 . 0 0 0 }$ </td></tr><tr><td>ChebNet</td><td> $1 0 . 6 0 0 { \scriptstyle \pm 1 . 2 0 0 }$ </td><td> $9 8 3 . 6 0 0 { \scriptstyle \pm 5 . 1 0 0 }$ </td><td> $9 2 2 . 0 0 0 { \scriptstyle \pm 4 7 . 0 0 0 }$ </td></tr><tr><td>SONAR</td><td> $4 6 2 . 8 0 0 { \scriptstyle \pm 2 5 0 . 3 0 0 }$ </td><td> $7 1 7 . 5 0 0 { \scriptstyle \pm 2 4 9 . 6 0 0 }$ </td><td> $8 0 9 . 0 0 0 { \scriptstyle \pm 6 8 . 0 0 0 }$ </td></tr><tr><td>Diag-NSD</td><td> $9 3 7 . 2 0 0 { \scriptstyle \pm 1 1 . 4 0 0 }$ </td><td> $1 0 0 6 . 6 0 0 { \scriptstyle \pm 2 2 . 0 0 0 }$ </td><td> $8 5 8 . 0 0 0 { \scriptstyle \pm 4 3 . 0 0 0 }$ </td></tr><tr><td>Ortho-NSD</td><td> $9 6 5 . 3 0 0 { \scriptstyle \pm 2 3 . 0 0 0 }$ </td><td> $1 0 0 5 . 1 0 0 { \scriptstyle \pm 7 . 8 0 0 }$ </td><td> $8 9 1 . 0 0 0 { \scriptstyle \pm 3 8 . 0 0 0 }$ </td></tr><tr><td>LowRank-NSD</td><td> $9 5 5 . 1 0 0 { \scriptstyle \pm 6 . 3 0 0 }$ </td><td> $1 0 0 8 . 1 0 0 { \scriptstyle \pm 1 6 . 4 0 0 }$ </td><td> $8 2 5 . 0 0 0 { \scriptstyle \pm 3 2 . 0 0 0 }$ </td></tr><tr><td>General-NSD</td><td> $9 3 8 . 1 0 0 { \scriptstyle \pm 1 2 . 1 0 0 }$ </td><td> $1 0 3 4 . 2 0 0 { \scriptstyle \pm 2 3 . 5 0 0 }$ </td><td> $8 6 0 . 0 0 0 { \scriptstyle \pm 3 1 . 0 0 0 }$ </td></tr><tr><td>CSNN</td><td> $4 7 3 . 9 0 0 { \scriptstyle \pm 1 0 6 . 2 0 0 }$ </td><td> $8 9 0 . 8 0 0 { \scriptstyle \pm 2 6 5 . 4 0 0 }$ </td><td> $7 4 2 . 0 0 0 { \scriptstyle \pm 1 6 8 . 0 0 0 }$ </td></tr><tr><td>Diag-ONDA</td><td> $1 . 2 0 0 { \scriptstyle \pm 0 . 4 0 0 }$ </td><td> $0 . 9 0 0 { \scriptstyle \pm 0 . 5 0 0 }$ </td><td> $8 3 . 0 0 0 { \scriptstyle \pm 8 5 . 0 0 0 }$ </td></tr><tr><td>Ortho-ONDA</td><td> $0 . 7 0 0 { \scriptstyle \pm 0 . 3 0 0 }$ </td><td> $\mathbf { 6 8 . 9 0 0 { \scriptstyle \pm 1 6 5 . 9 0 0 } }$ </td><td> $9 2 . 0 0 0 { \scriptstyle \pm 1 2 7 . 0 0 0 }$ </td></tr><tr><td> $\mathrm { L o w R a n k  – O N D A }$ </td><td> $2 2 . 8 0 0 { \scriptstyle \pm 1 9 . 5 0 0 }$ </td><td> $8 4 0 . 1 0 0 { \scriptstyle \pm 3 0 7 . 9 0 0 }$ </td><td> $\mathbf { 1 2 8 . 0 0 0 { \scriptstyle \pm 1 5 4 . 0 0 0 } }$ </td></tr><tr><td> $\mathrm { G e n e r a l - O N D A }$ </td><td> $\mathbf { 1 . 3 0 0 { \scriptstyle \pm 0 . 3 0 0 } }$ </td><td> $2 . 2 0 0 { \scriptstyle \pm 1 . 2 0 0 }$ </td><td> $6 1 6 . 0 0 0 { \scriptstyle \pm 1 2 1 . 0 0 0 }$ </td></tr></table>

## C.2 HETEROPHILIC TASKS

Although ONDA is primarily designed to improve long-range information propagation, we also evaluate its effectiveness on heterophilic tasks. We consider the heterophilic benchmarks from Platonov et al. (2023), and compare ONDA against conventional MPNNs, graph Transformers, heterophilyspecific architectures, and recent sheaf neural networks (see also Section B.2).

Hyperparameter Space. Table 7 reports the hyperparameter values considered for the heterophilic benchmarks. Hyperparameters were selected through sequential local searches over subsets of these candidate values as well as Bayesian optimization. Final configurations were selected according to validation performance. Similarly to Bamberger et al. (2025); Ribeiro et al. (2026), for these tasks we optionally learn restriction maps from node representations produced by a GNN encoder. Specifically, in Table 7, GNN depth denotes the number of message-passing layers, and GNN hidden dimension denotes the encoder’s hidden width. At depth zero, an MLP learns the restriction maps directly from the endpoint node features, similar to (Bodnar et al., 2022).

Table 7: Hyperparameter search space for the heterophilic benchmarks.
<table><tr><td>Hyperparameter</td><td>Candidate values</td></tr><tr><td>Hidden dimension H</td><td>{64, 128, 256, 512}</td></tr><tr><td>Stalk dimension d</td><td>{2, 4, 8}</td></tr><tr><td>ONDA Blocks B</td><td>{2, 3, 4, 6, 8, 10, 11}</td></tr><tr><td>Inner iterations L</td><td>{1, 2, 4, 6, 8, 10}</td></tr><tr><td>Step size ε</td><td> $\{ 0 . 0 1 , 0 . 0 2 5 , 0 . 0 4 , 0 . 0 5 , 0 . 0 7 5 , 0 . 1 \}$ </td></tr><tr><td>Learning rate</td><td> $\{ 5 \times 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 2 \times 1 0 ^ { - 3 } , 2 \times 1 \bar { 0 } ^ { - 2 } \}$ </td></tr><tr><td>Operator</td><td>{classical, normalized, directional}</td></tr><tr><td>Sheaf type</td><td>{general}</td></tr><tr><td>Dissipation</td><td>{False, True}</td></tr><tr><td>Forcing</td><td>{False, True}</td></tr><tr><td>Fixed Restriction maps</td><td>{True}</td></tr><tr><td>GNN depth</td><td>{0, 1, 2, 3, 4, 5, 6}</td></tr><tr><td>GNN hidden dim</td><td>{64, 128, 256, 512}</td></tr></table>

Results. Table 8 reports the performance on the heterophilic benchmarks. ONDA achieves highly competitive performance across the heterophilic benchmarks despite not being specifically designed for these tasks, ranking third overall based on the average rank across the five datasets. Notably, ONDA consistently improves over SONAR across all five datasets, showing that extending oscillatory propagation to the sheaf setting is also beneficial beyond explicitly long-range tasks. Similarly, ONDA substantially outperforms NSD on every dataset, indicating that richer sheaf transport alone is not sufficient to match the performance obtained with the proposed oscillatory dynamics. Compared with more recent sheaf architectures, ONDA remains competitive with BuNN and CSNN. Overall, these results show that ONDA maintains strong performance on heterophilic graphs while improving over both its closest oscillatory and standard sheaf counterparts.

## D TIME AND MEMORY COMPLEXITY ANALYSIS

In this section, we provide both a theoretical and experimental analysis of the time and space complexity of ONDA. Let $n = | \nu |$ and $m = | \mathcal { E } |$ , let d denote the stalk dimension, c the number of feature channels, L the number of internal propagation steps per block, and B the number of blocks. Since ONDA operates through graph-local propagation, its computational cost is determined by the stalk dimension, the number of feature channels, and the parameterization of the restriction maps.

Time Complexity. Consider a single propagation step. The state and velocity V each contain ndc scalar entries. The stalk transformation $( { \bar { \cal I } } _ { n } \not \otimes { \cal W } _ { 1 } ) X$ costs $\mathcal { O } ( n d ^ { 2 } c )$ , while the channel mixer $W _ { 2 }$ costs $\mathcal { O } ( n d c ^ { 2 } )$ . The cost of applying the sheaf operator depends on the restriction-map family (see Appendix B.1). Writing $q ( d )$ for the cost of applying one restriction map to one feature channel, we

Table 8: Mean test set score and std on heterophilic datasets. The higher, the better. First, second, and third best results for each task are color-coded. Baseline results are reported from Finkelshtein et al. (2024); Platonov et al. (2023); Muller et al. (2024); Luan et al. (2024); Ribeiro et al. (2026).¨
<table><tr><td>Model</td><td> $\operatorname { A c c } \uparrow$ </td><td>Roman-empire Amazon-ratings Minesweeper Acc ↑</td><td>AUC↑</td><td>Tolokers AUC↑</td><td>Questions AUC ↑</td></tr><tr><td>Luan et al. (2024)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MLP-2</td><td> $6 6 . 0 4 _ { \pm 0 . 7 1 }$ </td><td> $4 9 . 5 5 _ { \pm 0 . 8 1 }$ </td><td> $5 0 . 9 2 _ { \pm 1 . 2 5 }$ </td><td> $7 4 . 5 8 _ { \pm 0 . 7 5 }$ </td><td> $6 9 . 9 7 _ { \pm 1 . 1 6 }$ </td></tr><tr><td>SGC-1</td><td> $4 4 . 6 0 _ { \pm 0 . 5 2 }$ </td><td> $4 0 . 6 9 _ { \pm 0 . 4 2 }$ </td><td> $8 2 . 0 4 _ { \pm 0 . 7 7 }$ </td><td> $7 3 . 8 0 _ { \pm 1 . 3 5 }$ </td><td> $7 1 . 0 6 _ { \pm 0 . 9 2 }$ </td></tr><tr><td>MLP-1</td><td> $6 4 . 1 2 _ { \pm 0 . 6 1 }$ </td><td> $3 8 . 6 0 _ { \pm 0 . 4 1 }$ </td><td> $5 0 . 5 9 _ { \pm 0 . 8 3 }$ </td><td> $7 1 . 8 9 _ { \pm 0 . 8 2 }$ </td><td> $7 0 . 3 3 _ { \pm 0 . 9 6 }$ </td></tr><tr><td>Graph-agnostic</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ResNet</td><td> $6 5 . 8 8 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $4 5 . 9 0 _ { \pm 0 . 5 2 }$ </td><td> $5 0 . 8 9 _ { \pm 1 . 3 9 }$ </td><td> $7 2 . 9 5 _ { \pm 1 . 0 6 }$ </td><td> $7 0 . 3 4 _ { \pm 0 . 7 6 }$ </td></tr><tr><td>ResNet+SGC</td><td> $7 3 . 9 0 _ { \pm 0 . 5 1 }$ </td><td> $5 0 . 6 6 _ { \pm 0 . 4 8 }$ </td><td> $7 0 . 8 8 _ { \pm 0 . 9 0 }$ </td><td> $8 0 . 7 0 { \scriptstyle \pm 0 . 9 7 }$ </td><td> $7 5 . 8 1 _ { \pm 0 . 9 6 }$ </td></tr><tr><td>ResNet+adj</td><td> $5 2 . 2 5 _ { \pm 0 . 4 0 }$ </td><td> $5 1 . 8 3 { \scriptstyle \pm 0 . 5 7 }$ </td><td> $5 0 . 4 2 { \scriptstyle \pm 0 . 8 3 }$ </td><td> $7 8 . 7 8 { \scriptstyle \pm 1 . 1 1 }$ </td><td> $7 5 . 7 7 { \scriptstyle \pm 1 . 2 4 }$ </td></tr><tr><td>MPNNs</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GAT</td><td> $8 0 . 8 7 _ { \pm 0 . 3 0 }$ </td><td> $4 9 . 0 9 _ { \pm 0 . 6 3 }$ </td><td> $9 2 . 0 1 _ { \pm 0 . 6 8 }$ </td><td> $8 3 . 7 0 _ { \pm 0 . 4 7 }$ </td><td> $7 7 . 4 3 _ { \pm 1 . 2 0 }$ </td></tr><tr><td>GAT-sep</td><td> $8 8 . 7 5 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $5 2 . 7 0 { \scriptstyle \pm 0 . 6 2 }$ </td><td> $9 3 . 9 1 _ { \pm 0 . 3 5 }$ </td><td> $8 3 . 7 8 { \scriptstyle \pm 0 . 4 3 }$ </td><td> $7 6 . 7 9 { \scriptstyle \pm 0 . 7 1 }$ </td></tr><tr><td>GAT (LapPE)</td><td> $8 4 . 8 0 _ { \pm 0 . 4 6 }$ </td><td> $4 4 . 9 0 _ { \pm 0 . 7 3 }$ </td><td> $9 3 . 5 0 { \scriptstyle \pm 0 . 5 4 }$ </td><td> $8 4 . 9 9 _ { \pm 0 . 5 4 }$ </td><td> $7 6 . 5 5 _ { \pm 0 . 8 4 }$ </td></tr><tr><td>GAT (RWSE)</td><td> $8 6 . 6 2 _ { \pm 0 . 5 3 }$ </td><td> $4 8 . 5 8 _ { \pm 0 . 4 1 }$ </td><td> $9 2 . 5 3 _ { \pm 0 . 6 5 }$ </td><td> $8 5 . 0 2 _ { \pm 0 . 6 7 }$ </td><td> $7 7 . 8 3 _ { \pm 1 . 2 2 }$ </td></tr><tr><td>GAT (DEG)</td><td> $8 5 . 5 1 { \scriptstyle \pm 0 . 5 6 }$ </td><td> $5 1 . 6 5 { \scriptstyle \pm 0 . 6 0 }$ </td><td> $9 3 . 0 4 { \scriptstyle \pm 0 . 6 2 }$ </td><td> $8 4 . 2 2 _ { \pm 0 . 8 1 }$ </td><td> $7 7 . 1 0 { \scriptstyle \pm 1 . 2 3 }$ </td></tr><tr><td>GatedGCN</td><td> $7 4 . 4 6 _ { \pm 0 . 5 4 }$ </td><td> $4 3 . 0 0 _ { \pm 0 . 3 2 }$ </td><td> $8 7 . 5 4 _ { \pm 1 . 2 2 }$ </td><td> $7 7 . 3 1 { \scriptstyle \pm 1 . 1 4 }$ </td><td> $7 6 . 6 1 _ { \pm 1 . 1 3 }$ </td></tr><tr><td>GCN</td><td> $7 3 . 6 9 _ { \pm 0 . 7 4 }$ </td><td> $4 8 . 7 0 { \scriptstyle \pm 0 . 6 3 }$ </td><td> $8 9 . 7 5 _ { \pm 0 . 5 2 }$ </td><td> $8 3 . 6 4 _ { \pm 0 . 6 7 }$ </td><td> $7 6 . 0 9 _ { \pm 1 . 2 7 }$ </td></tr><tr><td>GCN (LapPE)</td><td> $8 3 . 3 7 { \scriptstyle \pm 0 . 5 5 }$ </td><td> $4 4 . 3 5 _ { \pm 0 . 3 6 }$ </td><td> $9 4 . 2 6 { \scriptstyle \pm 0 . 4 9 }$ </td><td> $8 4 . 9 5 _ { \pm 0 . 7 8 }$ </td><td> $7 7 . 7 9 _ { \pm 1 . 3 4 }$ </td></tr><tr><td>GCN (RWSE)</td><td> $8 4 . 8 4 _ { \pm 0 . 5 5 }$ </td><td> $4 6 . 4 0 { \scriptstyle \pm 0 . 5 5 }$ </td><td> $9 3 . 8 4 _ { \pm 0 . 4 8 }$ </td><td> $\mathbf { 8 5 . 1 1 { \scriptstyle \pm 0 . 7 7 } }$ </td><td> $7 7 . 8 1 _ { \pm 1 . 4 0 }$ </td></tr><tr><td>GCN (DEG)</td><td> $8 4 . 2 1 _ { \pm 0 . 4 7 }$ </td><td> $5 0 . 0 1 _ { \pm 0 . 6 9 }$ </td><td> $9 4 . 1 4 _ { \pm 0 . 5 0 }$ </td><td> $8 2 . 5 1 _ { \pm 0 . 8 3 }$ </td><td> $7 6 . 9 6 _ { \pm 1 . 2 1 }$ </td></tr><tr><td> $\mathrm { C o G N N } ( \Sigma , \Sigma )$ </td><td> $9 1 . 5 7 { \scriptstyle \pm 0 . 3 2 }$ </td><td> $5 1 . 2 8 { \scriptstyle \pm 0 . 5 6 }$ </td><td> $9 5 . 0 9 { \scriptstyle \pm 1 . 1 8 }$ </td><td> $8 3 . 3 6 { \scriptstyle \pm 0 . 8 9 }$ </td><td>80.02±0.86</td></tr><tr><td> $\operatorname { C o G N N } ( \mu , \mu )$ </td><td> $9 1 . 3 7 { \scriptstyle \pm 0 . 3 5 }$ </td><td> ${ 5 4 . 1 7 \pm 0 . 3 7 }$ </td><td> $9 7 . 3 1 _ { \pm 0 . 4 1 }$ </td><td> $8 4 . 4 5 _ { \pm 1 . 1 7 }$ </td><td> $7 6 . 5 4 _ { \pm 0 . 9 5 }$ </td></tr><tr><td>GraphSAGE</td><td> $8 5 . 7 4 _ { \pm 0 . 6 7 }$ </td><td> ${ \pm 3 . 6 3 \mathrm { _ { \pm 0 . 3 9 } } }$ </td><td> $9 3 . 5 1 _ { \pm 0 . 5 7 }$ </td><td> $8 2 . 4 3 _ { \pm 0 . 4 4 }$ </td><td> $7 6 . 4 4 _ { \pm 0 . 6 2 }$ </td></tr><tr><td>SONAR</td><td> $8 9 . 8 2 { \scriptstyle \pm 0 . 5 7 }$ </td><td> $5 2 . 2 2 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $9 6 . 2 9 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $8 3 . 5 7 { \scriptstyle \pm 1 . 4 4 }$ </td><td> $7 4 . 9 6 { \scriptstyle \pm 1 . 1 0 }$ </td></tr><tr><td>Graph Transformers</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Exphormer</td><td> $8 9 . 0 3 _ { \pm 0 . 3 7 }$ </td><td> $5 3 . 5 1 _ { \pm 0 . 4 6 }$ </td><td> $9 0 . 7 4 _ { \pm 0 . 5 3 }$ </td><td> $8 3 . 7 7 _ { \pm 0 . 7 8 }$ </td><td> $7 3 . 9 4 _ { \pm 1 . 0 6 }$ </td></tr><tr><td>NAGphormer GOAT</td><td> $7 4 . 3 4 { \scriptstyle \pm 0 . 7 7 }$ </td><td> $5 1 . 2 6 { \scriptstyle \pm 0 . 7 2 }$ </td><td> $8 4 . 1 9 { \scriptstyle \pm 0 . 6 6 }$ </td><td> $7 8 . 3 2 { \scriptstyle \pm 0 . 9 5 }$ </td><td> $6 8 . 1 7 { \scriptstyle \pm 1 . 5 3 }$ </td></tr><tr><td>GPS</td><td> $7 1 . 5 9 _ { \pm 1 . 2 5 }$ </td><td> $4 4 . 6 1 _ { \pm 0 . 5 0 }$ </td><td> $8 1 . 0 9 _ { \pm 1 . 0 2 }$ </td><td> $8 3 . 1 1 _ { \pm 1 . 0 4 }$ </td><td> $7 5 . 7 6 { \scriptstyle \pm 1 . 6 6 }$ </td></tr><tr><td>GPSGCN+Performer (LapPE)</td><td> $8 2 . 0 0 { \scriptstyle \pm 0 . 6 1 }$ </td><td> $5 3 . 1 0 _ { \pm 0 . 4 2 }$ </td><td> $9 0 . 6 3 _ { \pm 0 . 6 7 }$ </td><td>83.71±0.48</td><td> $7 1 . 7 3 _ { \pm 1 . 4 7 }$ </td></tr><tr><td>(RWSE)</td><td> $8 3 . 9 6 _ { \pm 0 . 5 3 }$ </td><td> $4 8 . 2 0 { \scriptstyle \pm 0 . 6 7 }$ </td><td> $9 3 . 8 5 _ { \pm 0 . 4 1 }$ </td><td> $8 4 . 7 2 { \scriptstyle \pm 0 . 7 7 }$ </td><td> $7 7 . 8 5 { \scriptstyle \pm 1 . 2 5 }$ </td></tr><tr><td> $\mathrm { G P S } _ { \mathrm { G C N + P e r f o r m e r } }$  GPSGCN+Performer (DEG)</td><td> $8 4 . 7 2 _ { \pm 0 . 6 5 }$ </td><td> $4 8 . 0 8 { \scriptstyle \pm 0 . 8 5 }$ </td><td> $9 2 . 8 8 _ { \pm 0 . 5 0 }$ </td><td> $8 4 . 8 1 _ { \pm 0 . 8 6 }$ </td><td> $7 6 . 4 5 _ { \pm 1 . 5 1 }$ </td></tr><tr><td> $\mathrm { G P S _ { G A T + P e r f o r m e r } }$  (LapPE)</td><td> $8 3 . 3 8 _ { \pm 0 . 6 8 }$ </td><td> $4 8 . 9 3 _ { \pm 0 . 4 7 }$ </td><td> $9 3 . 6 0 _ { \pm 0 . 4 7 }$ </td><td> $8 0 . 4 9 _ { \pm 0 . 9 7 }$ </td><td> $7 4 . 2 4 _ { \pm 1 . 1 8 }$ </td></tr><tr><td> $\mathrm { G P S } _ { \mathrm { G A T + P e r f o r m e r } }$  (RWSE)</td><td> $8 5 . 9 3 { \scriptstyle \pm 0 . 5 2 }$ </td><td> $4 8 . 8 6 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $9 2 . 6 2 { \scriptstyle \pm 0 . 7 9 }$ </td><td> $8 4 . 6 2 { \scriptstyle \pm 0 . 5 4 }$ </td><td> $7 6 . 7 1 { \scriptstyle \pm 0 . 9 8 }$ </td></tr><tr><td></td><td> $8 7 . 0 4 _ { \pm 0 . 5 8 }$ </td><td> $4 9 . 9 2 _ { \pm 0 . 6 8 }$ </td><td> $9 1 . 0 8 { \scriptstyle \pm 0 . 5 8 }$ </td><td> $8 4 . 3 8 { \scriptstyle \pm 0 . 9 1 }$ </td><td>77.14±1.49</td></tr><tr><td>GPSGAT+Performer (DEG) GT</td><td> $8 5 . 5 4 _ { \pm 0 . 5 8 }$ </td><td> $5 1 . 0 3 _ { \pm 0 . 6 0 }$ </td><td> $9 1 . 5 2 _ { \pm 0 . 4 6 }$ </td><td> $8 2 . 4 5 _ { \pm 0 . 8 9 }$ </td><td>76.51±1.19</td></tr><tr><td>GT-sep</td><td> $8 6 . 5 1 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $5 1 . 1 7 { \scriptstyle \pm 0 . 6 6 }$ </td><td> $9 1 . 8 5 { \scriptstyle \pm 0 . 7 6 }$ </td><td> $8 3 . 2 3 { \scriptstyle \pm 0 . 6 4 }$ </td><td> $7 7 . 9 5 _ { \pm 0 . 6 8 }$ </td></tr><tr><td></td><td> $8 7 . 3 2 _ { \pm 0 . 3 9 }$ </td><td> $5 2 . 1 8 _ { \pm 0 . 8 0 }$ </td><td> $9 2 . 2 9 _ { \pm 0 . 4 7 }$ </td><td> $8 2 . 5 2 _ { \pm 0 . 9 2 }$ </td><td> $7 8 . 0 5 _ { \pm 0 . 9 3 }$ </td></tr><tr><td colspan="6">Heterophily-Designated GNNs</td></tr><tr><td>CPGNN</td><td> $6 3 . 9 6 { \scriptstyle \pm 0 . 6 2 }$ </td><td> $3 9 . 7 9 _ { \pm 0 . 7 7 }$ </td><td> $5 2 . 0 3 { \scriptstyle \pm 5 . 4 6 }$ </td><td> $7 3 . 3 6 { \scriptstyle \pm 1 . 0 1 }$ </td><td> $6 5 . 9 6 { \scriptstyle \pm 1 . 9 5 }$ </td></tr><tr><td>FAGCN</td><td> $6 5 . 2 2 _ { \pm 0 . 5 6 }$ </td><td> $4 4 . 1 2 _ { \pm 0 . 3 0 }$ </td><td> $8 8 . 1 7 _ { \pm 0 . 7 3 }$ </td><td> $7 7 . 7 5 _ { \pm 1 . 0 5 }$ </td><td> $7 7 . 2 4 _ { \pm 1 . 2 6 }$ </td></tr><tr><td>FSGNN</td><td> $7 9 . 9 2 _ { \pm 0 . 5 6 }$ </td><td> $5 2 . 7 4 _ { \pm 0 . 8 3 }$ </td><td> $9 0 . 0 8 _ { \pm 0 . 7 0 }$ </td><td> $8 2 . 7 6 _ { \pm 0 . 6 1 }$ </td><td> $\mathbf { 7 8 . 8 6 _ { \pm 0 . 9 2 } }$ </td></tr><tr><td>GBK-GNN</td><td> $7 4 . 5 7 { \scriptstyle \pm 0 . 4 7 }$ </td><td> $4 5 . 9 8 { \scriptstyle \pm 0 . 7 1 }$ </td><td> $9 0 . 8 5 { \scriptstyle \pm 0 . 5 8 }$ </td><td> $8 1 . 0 1 { \scriptstyle \pm 0 . 6 7 }$ </td><td> $7 4 . 4 7 { \scriptstyle \pm 0 . 8 6 }$ </td></tr><tr><td>GloGNN</td><td> $5 9 . 6 3 _ { \pm 0 . 6 9 }$ </td><td> $3 6 . 8 9 _ { \pm 0 . 1 4 }$ </td><td> $5 1 . 0 8 _ { \pm 1 . 2 3 }$ </td><td> $7 3 . 3 9 _ { \pm 1 . 1 7 }$ </td><td> $6 5 . 7 4 _ { \pm 1 . 1 9 }$ </td></tr><tr><td>GPR-GNN</td><td> $6 4 . 8 5 _ { \pm 0 . 2 7 }$ </td><td> $4 4 . 8 8 _ { \pm 0 . 3 4 }$ </td><td> $8 6 . 2 4 _ { \pm 0 . 6 1 }$ </td><td> $7 2 . 9 4 _ { \pm 0 . 9 7 }$ </td><td> $5 5 . 4 8 _ { \pm 0 . 9 1 }$ </td></tr><tr><td>H2GCN</td><td> $6 0 . 1 1 { \scriptstyle \pm 0 . 5 2 }$ </td><td> $3 6 . 4 7 { \scriptstyle \pm 0 . 2 3 }$ </td><td> $8 9 . 7 1 { \scriptstyle \pm 0 . 3 1 }$ </td><td> $7 3 . 3 5 { \scriptstyle \pm 1 . 0 1 }$ </td><td> $6 3 . 5 9 { \scriptstyle \pm 1 . 4 6 }$ </td></tr><tr><td>JacobiConv</td><td> $7 1 . 1 4 _ { \pm 0 . 4 2 }$ </td><td> $4 3 . 5 5 _ { \pm 0 . 4 8 }$ </td><td> $8 9 . 6 6 _ { \pm 0 . 4 0 }$ </td><td> $6 8 . 6 6 { \scriptstyle \pm 0 . 6 5 }$ </td><td> $7 3 . 8 8 _ { \pm 1 . 1 6 }$ </td></tr><tr><td>Sheaf Neural Networks</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>NSD</td><td> $8 0 . 4 1 { \scriptstyle \pm 0 . 7 2 }$ </td><td> $4 2 . 7 6 { \scriptstyle \pm 0 . 5 4 }$ </td><td> $9 2 . 1 5 { \scriptstyle \pm 0 . 8 4 }$ </td><td> $7 8 . 8 3 { \scriptstyle \pm 0 . 7 6 }$ </td><td> $6 9 . 6 9 { \scriptstyle \pm 1 . 4 6 }$ </td></tr><tr><td>BuNN</td><td> $\mathbf { 9 1 . 7 5 { \scriptstyle \pm 0 . 3 9 } }$ </td><td> $5 3 . 7 4 { \scriptstyle \pm 0 . 5 1 }$ </td><td> $9 8 . 9 9 2 \mathrm { { \fontfamily { q p l } \selectfont { 0 . 1 6 } } }$ </td><td> $8 4 . 7 8 { \scriptstyle \pm 0 . 8 0 }$ </td><td> $7 8 . 7 5 { \scriptstyle \pm 1 . 0 9 }$ </td></tr><tr><td>CSNN</td><td> $9 2 . 6 3 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $5 2 . 0 7 { \scriptstyle \pm 1 . 0 0 }$ </td><td> $9 9 . 0 7 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $8 5 . 4 5 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $7 9 . 3 1 { \scriptstyle \pm 1 . 2 2 }$ </td></tr><tr><td>Ours</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ONDA</td><td> $9 1 . 8 6 _ { \pm 0 . 3 1 }$ </td><td> $5 2 . 7 6 { \scriptstyle \pm 0 . 4 8 }$ </td><td> $\mathbf { 9 8 . 9 5 { \scriptstyle \pm 0 . 5 4 } }$ </td><td> $8 5 . 2 1 { \scriptstyle \pm 0 . 9 }$ </td><td> $7 6 . 1 4 { \scriptstyle \pm 1 . 3 0 }$ </td></tr></table>

have:

$$
\begin{array} { r } { q ( d ) = \left\{ \begin{array} { l l } { d , } & { \mathrm { d i a g o n a l , } } \\ { d ^ { 2 } , } & { \mathrm { g e n e r a l o r o r t h o g o n a l , } } \\ { r d , } & { \mathrm { r a n k - } r \mathrm { l o w - r a n k , } } \end{array} \right. } \end{array}\tag{21}
$$

where the low-rank complexity assumes that the restriction maps are applied in factorized form. A sparse application of the sheaf operator therefore costs $\mathcal { O } ( m q ( \bar { d } ) c )$ . Since the restriction maps may be recomputed from the current state at each propagation step, we denote the corresponding learner cost by $\bar { \mathcal { C } } _ { \mathrm { m a p } } .$ The velocity update uses the sheaf propagation term already counted above. The remaining updates of $V$ and $\dot { x }$ involve only elementwise scaling and addition, contributing $\mathcal { O } ( n d c )$ operations per step. This contribution is absorbed by the node-wise transformation costs, so the combined cost of one ONDA propagation step is:

$$
T _ { \mathrm { O N D A } } = \mathcal { O } \big ( \mathcal { C } _ { \mathrm { m a p } } + m q ( d ) c + n d ^ { 2 } c + n d c ^ { 2 } \big ) .\tag{22}
$$

Across $B$ blocks, each with $\mathcal { L }$ propagation steps, the total propagation cost is:

$$
T _ { \mathrm { O N D A } } ^ { \mathrm { n e t } } = \left\{ \begin{array} { l l } { \mathcal { O } \big ( B \mathcal { C } _ { \mathrm { m a p } } + B \mathcal { L } ( m q ( d ) c + n d ^ { 2 } c + n d c ^ { 2 } ) \big ) , } & { \mathrm { w h e n ~ m a p s ~ f i x e d ~ w i t h i n ~ e a c h ~ b l o c k , } } \\ { \mathcal { O } \big ( B \mathcal { L } \left( \mathcal { C } _ { \mathrm { m a p } } + m q ( d ) c + n d ^ { 2 } c + n d c ^ { 2 } \right) \big ) , } & { \mathrm { w h e n ~ m a p s ~ r e c o m p u t e d ~ a t ~ e a c h ~ s t e p . } } \end{array} \right.
$$

In particular, for fixed stalk dimension $d ,$ feature dimension c, and restriction-map parameterization, ONDA retains linear scaling with the graph size $n + m$ . Thus, its matrix-valued sheaf transport preserves the locality and sparse graph scaling of message-passing architectures (Gilmer et al., 2017). In contrast, Graph Transformers based on dense global self-attention typically incur quadratic complexity in the number of nodes, $\mathcal { O } ( n ^ { 2 } )$ , due to all-pairs interactions. Thus, despite its richer local transport, ONDA preserves the sparse scaling characteristic of MPNNs and avoids the quadratic dependence on n of dense Graph Transformers.

Memory Complexity. The instantaneous state X and velocity $V$ together contain 2ndc scalar entries and therefore require $\mathcal { O } ( n d c )$ memory. The learned restriction maps additionally require $\mathcal { O } ( m s ( d ) )$ storage, where:

$$
s ( d ) = \left\{ \begin{array} { l l } { d , } & { \mathrm { d i a g o n a l } , } \\ { d ^ { 2 } , } & { \mathrm { g e n e r a l o r o r t h o g o n a l } , } \\ { r d , } & { \mathrm { r a n k } \ – r \mathrm { l o w } \mathrm { - r a n k } , } \end{array} \right.\tag{23}
$$

up to constant factors associated with the chosen parameterization. Thus, excluding trainable parameters and common graph storage, inference requires:

$$
M _ { \mathrm { O N D A } } ^ { \mathrm { i n f e r } } = \mathcal { O } ( n d c + m s ( d ) ) .\tag{24}
$$

During training, backpropagation through the unrolled second-order dynamics requires retaining intermediate activations. Without activation checkpointing, this gives the leading-order bound:

$$
M _ { \mathrm { O N D A } } ^ { \mathrm { t r a i n } } = \mathcal { O } ( B \mathcal { L } \left( n d c + m s ( d ) \right) ) .\tag{25}
$$

For fixed $d$ and restriction-map parameterization, the memory requirements of ONDA also scale linearly with $n + m$ , as in MPNNs. This contrasts with dense Graph Transformers, whose attention matrices require $\mathcal { O } ( n ^ { 2 } )$ memory per layer. Hence, both the computational and memory complexity of ONDA remain compatible with sparse graph processing, with the additional cost arising from the stalk-valued representation and matrix-valued restriction maps rather than from dense pairwise interactions.

Matched-parameter Empirical Cost. To complement the asymptotic analysis, we measure wallclock time and peak GPU memory on the Barbell benchmark under a controlled matched-parameter setting. We compare ONDA with SONAR (Trenta et al., 2025), a message-passing architecture based on oscillatory dynamics, to assess the practical computational overhead introduced by the sheaf-based components of ONDA. All models use B = 4 blocks, $\mathcal { L } = 1 2$ internal propagation steps per block, and approximately 100k trainable parameters. Measurements are obtained with batch size one (to avoid confounding effects from batch-level computational optimizations) on an RTX 5080, with one exclusive GPU process at a time. Table 9 reports results on multiple settings of the problem, N = 10, 20, 50, showing results for different implementations of ONDA’s restrictionmap parametrization (see Appendix B.1). Despite the richer per-edge computation predicted by Equation (22), the measured runtime remains comparable to SONAR under a matched parameter budget. At N = 50, Diag-ONDA and Ortho-ONDA reduce training time by 15.1% and 9.1%, respectively, relative to SONAR, while their inference latency is lower by 39.6% and 28.8%. The more expressive LowRank and General variants incur only a 2.3% and 2.0% increase in training time, respectively, while still reducing inference latency by approximately 22.7%. Across N ∈ {10, 20, 50}, all four ONDA variants exhibit lower measured inference latency than SONAR. At N = 50, peak training memory ranges from 159.9 to 241.8 MB for the sheaf variants, compared with 120.7 MB for SONAR. This increase is consistent with Equation (25): ONDA maintains ddimensional stalk states and learned edge-wise restriction maps throughout the unrolled dynamics. Importantly, this additional memory buys a strictly richer propagation mechanism while preserving sparse, linear-in-n+m graph scaling. Hence, for the small stalk dimensions used in practice, ONDA retains the computational profile of local SONAR propagation while replacing scalar communication with expressive matrix-valued transport.

Table 9: Matched-parameter computational cost on Barbell. All models use approximately 100k trainable parameters, four blocks, and 12 internal propagation steps per block. We report peak training memory, training time per epoch, and inference latency per graph for $N \in \{ 1 0 , 2 0 , \hat { 5 } 0 \}$ corresponding to graphs with 20, 40, and 100 nodes, respectively. Values are means over five seeds; lower is better.
<table><tr><td rowspan="2">Model</td><td rowspan="2"></td><td colspan="3">N = 10</td><td colspan="3">N = 20</td><td colspan="3">N = 50</td></tr><tr><td>#Params Mem. MB</td><td>Train ms/ep.</td><td>Infer. ms/g</td><td>Mem. MB</td><td>Train ms/ep.</td><td>Infer. ms/g</td><td>Mem. MB</td><td>Train ms/ep.</td><td>Infer. ms/g</td></tr><tr><td>SONAR</td><td>100,079</td><td>23.1</td><td>2400.1</td><td>10.245</td><td>35.1</td><td>2452.0</td><td>10.800</td><td>120.7</td><td>2675.2</td><td>11.138</td></tr><tr><td>Diag-ONDA</td><td>101,405</td><td>26.8</td><td>2186.3</td><td>6.716</td><td>50.9</td><td>2193.7</td><td>6.795</td><td>223.0</td><td>2270.4</td><td>6.722</td></tr><tr><td>Ortho-ONDA</td><td>103,700</td><td>24.5</td><td>2229.1</td><td>7.849</td><td>41.1</td><td>2237.4</td><td>7.962</td><td>159.9</td><td>2432.3</td><td>7.925</td></tr><tr><td>LowRank-ONDA</td><td>101,540</td><td>27.6</td><td>2464.3</td><td>8.371</td><td>53.9</td><td>2459.3</td><td>8.290</td><td>241.8</td><td>2736.4</td><td>8.610</td></tr><tr><td>General-ONDA</td><td>104,814</td><td>27.4</td><td>2404.6</td><td>8.150</td><td>53.1</td><td>2447.9</td><td>8.138</td><td>236.5</td><td>2729.5</td><td>8.604</td></tr></table>