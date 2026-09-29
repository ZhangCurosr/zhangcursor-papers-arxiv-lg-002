# Hierarchical Clustering and Signal Denoising on Digraphs

Yi Wang, Sippanon Kitimoon, Hrushikesh N. Mhaskar, Xiaosheng Zhuang

## Abstract

In this paper, we propose a representation of a digraph (directed graph) as a Hermitian matrix derived from its adjacency matrix. This representation characterizes both the connectivity and the edge orientation of the digraph. Based on the spectral decomposition of the Hermitian matrix, a digraph clustering algorithm with k-means is introduced to produce a partition on the graph. Applying this algorithm (bottom-up) recursively to a digraph with partially labeled vertices yields a spectral hierarchical digraph clustering (SpecHDC) algorithm that produces consistent nested partitions of the digraph, or equivalently, a tree structure. Furthermore, based on the in-degree and out-degree of each cluster in the digraph clustering, a pair of hierarchical interval partitions (filtrations) can be derived in a top-down manner to produce a pair of nested knot sequences. These knot sequences facilitate the construction of multilevel spline quasi-interpolants, enabling a noisy graph signal to be decomposed into a coarse approximation and inter-level details, followed by adaptive thresholding and reconstruction. Experiments on synthetic and real-world digraphs demonstrate the superiority of our SpecHDC algorithm for digraph clustering across diverse graph structural properties (homophily and heterophily) and supervision settings. Moreover, experiments on digraph signal processing using multilevel spline quasi-interpolants further demonstrate the effectiveness of signal recovery on digraphs in terms of RMSE and SNR.

## Index Terms

Graph signal processing, Digraph clustering, Spline approximation, Quasi-interpolants.

## I. INTRODUCTION

IGNALS defined on graphs arise in a broad range of applications [1], [2], [3], [4], including social, citation, web, sensor, and biological networks, where observations are associated with vertices and their dependencies are encoded by edges. Graph signal processing (GSP) extends classical signal processing from regular Euclidean domains to graph domains. In GSP, graph operators—most notably the graph Laplacian, adjacency matrix, and graph-shift operator are used to define graph-domain notions of frequency and signal variation and to support fundamental tasks such as filtering [5], [6], sampling [7], [8], and recovery [9], [10], [11].

Clustering can be viewed as a special form of graph signal recovery in GSP, where cluster labels constitute a discrete graph signal whose recovery reveals latent structures in graph-structured data. Early graph clustering methods formulated this task through combinatorial objectives such as graph cuts and normalized cuts [12], [13]. Spectral clustering subsequently relaxed these discrete objectives and embedded vertices using selected eigenvectors of the graph Laplacian, followed by k-means or related partitioning procedures [14], [15]. From a GSP perspective, these eigenvectors represent slowly varying structural modes, while the resulting cluster assignments constitute a piecewise-coherent graph signal. These classical formulations are primarily developed for undirected graphs, whereas many real-world networks are inherently directed (i.e., digraphs). In digraphs, edge orientations encode asymmetric relations such as influence, transition, migration, and information flow, making directionality an essential component of the clustering structure. Existing digraph clustering methods address such asymmetry from several perspectives. Symmetrization-based methods construct undirected similarities from directed connectivity patterns [16], [17], while source-target methods distinguish the sending and receiving roles of vertices. For example, DI-SIM uses the left and right singular vectors of a regularized directed Laplacian to identify asymmetric block structures [18]. Random-walk and flow-based approaches instead characterize clusters through directed transition behavior, including stationary-distribution-based Laplacians [19], [20] and description-length objectives [21]. More recently, Hermitian spectral methods have encoded edge orientation in complex-valued operators while retaining real eigenvalues and an orthonormal eigenbasis [22]; related skew-symmetric formulations reduce complex-domain computation while preserving the relevant directed-cut structure [23].

On the other hand, it is well known that spline approximation provides an efficient and effective tool in classical signal processing. Splines are piecewise polynomials over a knot sequence that can be evaluated fast [24]. Moreover, in contrast to exact interpolation, spline quasi-interpolation determines approximation coefficients from local samples or bounded functionals, avoiding a global interpolation system while retaining polynomial reproduction and favorable approximation orders [25], [26]. Spline constructions have also been closely connected with multiresolution representations, for example, spline-wavelet filter banks provide efficient multirate decomposition and reconstruction [27], [28], while hierarchical and truncated hierarchical B-splines enable adaptive local refinement through nested spline spaces [29], [30] [31].

These two research threads suggest a natural but unexplored connection. Multi-level digraph clustering reveals how fine-grained vertex groups are progressively merged into coarse structural units, while spline approximation requires an ordered knot configuration that determines local support and resolution. Establishing this connection poses two coupled challenges. First, recursive clustering must preserve asymmetric inter-cluster interactions, incorporate limited supervision, and maintain consistent relations across successive coarse digraphs. Second, the discrete partitions and their level-wise directional connectivity must be transformed into ordered intervals suitable for spline construction. These considerations lead to two closely related questions:

## • How can digraphs be clustered consistently across levels with limited supervision?

## • How can multi-level clusters induce valid knot sequences for spline approximation?

The first challenge requires a direction-aware spectral representation that incorporates available labels. The second requires an interval construction that combines the partition and directed adjacency information at every level. Addressing them jointly enables the resulting spline approximation to adapt to the hierarchical and asymmetric structures revealed by digraph clustering.

To address these challenges, we propose SpecHDC, a semi-supervised spectral hierarchical digraph clustering algorithm, together with a hierarchy-induced spline construction. SpecHDC is based on a generalized Hermitian matrix that jointly encodes reciprocal connectivity and asymmetric edge orientation, yielding a direction-sensitive spectral representation with real eigenvalues and an orthonormal eigenbasis. Unlike single-level methods that independently compute partitions at prescribed granularities, SpecHDC constructs a consistent hierarchy through recursive digraph coarsening. At each level, the current clusters are contracted into coarse vertices, and the directed adjacency weight between two coarse vertices is obtained by aggregating the edge weights between their constituent clusters. A new generalized Hermitian matrix is then constructed from the resulting coarse adjacency matrix, and its spectral representation is used to determine the partition at the next level. Repeating this procedure yields a sequence of nested partitions and their corresponding coarse directed graphs, with the available labels anchoring the finestlevel partition and propagating through the hierarchical construction. Based on the partition and adjacency matrix obtained at each level, we further derive in-degree and out-degree intervals that characterize the directional connectivity of the corresponding clusters. The intervals collected across the hierarchy are organized into multi-level knot sequences for spline construction. Consequently, the resulting splines incorporate both the hierarchical organization of the clusters and the asymmetric inflow and outflow patterns of the digraph, providing a direction-aware and structure-adaptive representation beyond the discrete clustering results. Empirically, controlled experiments on synthetic homophilic and heterophilic digraphs characterize the behavior of SpecHDC under different directed structural regimes. Comparisons with representative clustering baselines on five labeled and two unlabeled real-world networks further demonstrate its practical effectiveness, while additional denoising experiments validate the utility of the hierarchy-induced in-degree and out-degree knot sequences for structure-adaptive signal recovery.

## A. Related work

Digraph Clustering. From a graph signal processing perspective, digraph clustering seeks a directionaware graph operator whose spectral representation supports the recovery of cluster-label signals over asymmetric topologies. Existing methods mainly model directionality through symmetric transformations, source-target representations [16], [18], [32], random-walk dynamics [19], [21], and Hermitian operators [22], [23].

Satuluri and Parthasarathy [16] constructed symmetric similarity matrices from products involving the directed adjacency matrix and its transpose, enabling conventional spectral clustering through commonpredecessor and common-successor relations. Rohe et al. [18] proposed DI-SIM, which uses the left and right singular vectors of a regularized directed Laplacian to characterize the sending and receiving roles of vertices; related methods similarly derive source- and target-side representations from incoming and outgoing connectivity patterns [32]. Random-walk-based methods instead identify clusters through transition behavior. Chung's directed Laplacian constructs a spectral operator from a random-walk transition matrix and its stationary distribution [19], and was subsequently applied to learning from labeled and unlabeled digraph data [20]. InfoMap [21] detects modules by minimizing the description length of a random walk. Hermitian formulations provide a direction-preserving spectral alternative. Cucuringu et al. [22] encoded edge orientation in a complex Hermitian matrix with real eigenvalues and an orthonormal eigenbasis, allowing conventional spectral embedding to retain asymmetric information. Hayashi et al. [23] further related this representation to a real skew-symmetric formulation, reducing complex-domain computation while preserving the relevant directed-cut structure. Other studies have explored flow imbalance and higher-order relations among directed clusters [33], [34].

Despite these advances, most existing methods produce single-level partitions at prescribed granularities Direction-aware spectral hierarchical clustering remains less explored, particularly when partial labels must be incorporated and asymmetric inter-cluster interactions must be preserved during recursive graph coarsening.

Spline Approximation. Spline approximation [35], [24], [36], [37] is a classical framework for representing, interpolating, and smoothing functions through piecewise-polynomial basis functions defined over a knot sequence. The compact support and controllable smoothness of B-splines make them particularly suitable for local approximation, geometric modeling, numerical computation, and signal reconstruction [24]. Unlike classical interpolation, which enforces exact sample reproduction and often requires solving a global system, spline quasi-interpolation constructs coefficients from local function values or bounded linear functionals, providing an efficient and stable approximation while retaining locality and polynomial reproduction. Fundamental approximation properties of spline quasi-interpolants were established in [25], with further developments for scattered-data approximation given in [38]. Spline methods are also closely related to multiresolution and adaptive representation. Spline-wavelet constructions support efficient multiscale decomposition [28], while hierarchical spline spaces enable local refinement through nested approximation spaces [29], [30]. These methods rely strongly on the underlying knot sequence or refinement hierarchy, which is typically specified in advance. Although variational splines extend interpolation and recovery to graph-supported signals through graph Laplacians [39], they do not explicitly derive spline knots from the directional and hierarchical structure of a digraph. In this work, the in-degree and out-degree intervals obtained at successive levels of digraph clustering are assembled into multi-level knot sequences, allowing the quasi-interpolatory spline to adapt its resolution to both the hierarchical organization and asymmetric connectivity of the underlying digraph.

## B. Contributions

The main contributions of this work are summarized as follows:

• We introduce a generalized Hermitian matrix that jointly models reciprocal connectivity and asymmetric edge orientation, providing a flexible direction-sensitive spectral representation with real eigenvalues and an orthonormal eigenbasis.

• We develop SpecHDC, which recursively constructs coarse directed graphs by aggregating intercluster edge weights, recomputes the generalized Hermitian representation at each level, and produces nested partitions while incorporating the available label information.

• We derive level-wise in-degree and out-degree intervals from the clustering results and their corresponding adjacency matrices and organize these intervals into multi-level knot sequences, enabling direction-aware and structure-adaptive spline approximation.

• Since the knots are not equidistant, the de Boor inequality yielding stability bounds for splines does not hold. We obtain the necessary stability bounds with respect to a doubling measure. (See Section IV for details).

• Controlled experiments on synthetic homophilic and heterophilic digraphs analyze the structural behavior of SpecHDC; comparisons on five labeled and two unlabeled real-world networks demonstrate its clustering effectiveness; and denoising experiments confirm the value of the proposed knot sequences for graph signal recovery.

## C. Structure of the paper

The remainder of this paper is organized as follows. Section II presents the notation and preliminaries for digraph clustering, develops the proposed spectral hierarchical clustering algorithm, and provides illustrative examples. Section III evaluates the clustering performance on synthetic directed stochastic block models and real-world digraphs across diverse graph structural properties (homophily and heterophily) Section IV introduces splines with general measures and establishes the stability of the proposed quasiinterpolation scheme. Section V connects the obtained digraph hierarchy with graph signal processing by constructing multi-level spline representations and examining their performance in signal denoising. Finally, Section VI concludes the paper.

## II. DIGRAPH CLUSTERING

In this section, we introduce our SpecHDC algorithm based on a proposed generalized Hermitian matrix and present its induced hierarchical interval partitions. Some synthetic examples are used to demonstrate our approach.

## A. Notation and Preliminary on Graph Clustering

Undirected and Directed Graphs. Let $G \ = \ ( V , E , A )$ be a weighted graph with vertex set $V =$ $\{ v _ { 1 } , \dotsc , v _ { N } \}$ , edge set $E \subset V \times V$ , and weighted adjacency matrix $A = \mathbf { \bar { \rho } } [ a _ { i j } ] _ { 1 \leq i , j \leq N } \in \mathbb { R } _ { + } ^ { N \times N }$ , where $a _ { i i } = 0$ and $a _ { i , j } \neq 0 \mathrm { ~ i f ~ } ( v _ { i } , v _ { j } ) \in E$ . For an undirected graph, $A = A ^ { \top }$ , the degree of $v _ { i }$ is $\begin{array} { r } { d _ { i } = \sum _ { j = 1 } ^ { N } a _ { i j } , } \end{array}$ and the degree matrix $D = \operatorname* { d i a g } ( d _ { 1 } , \dots , d _ { N } )$ is diagonal. For a digraph, $a _ { i j } > 0$ denotes an edge from $v _ { i }$ to $v _ { j }$ , and generally $a _ { i j } \neq a _ { j i }$ . The out-degree and in-degree of $v _ { i }$ are given by

$$
d _ { i } ^ { \mathrm { o u t } } = \sum _ { j = 1 } ^ { N } a _ { i j } , \qquad d _ { i } ^ { \mathrm { i n } } = \sum _ { j = 1 } ^ { N } a _ { j i } ,\tag{2.1}
$$

and with the degree matrices $D _ { \mathrm { o u t } } = \mathrm { d i a g } ( d _ { 1 } ^ { \mathrm { o u t } } , \dots , d _ { N } ^ { \mathrm { o u t } } )$ and $D _ { \mathrm { i n } } = \mathrm { d i a g } ( d _ { 1 } ^ { \mathrm { i n } } , \dots , d _ { N } ^ { \mathrm { i n } } )$ . The graph volume is

$$
\operatorname { v o l } ( G ) = \sum _ { i = 1 } ^ { N } d _ { i } ^ { \mathrm { o u t } } = \sum _ { i = 1 } ^ { N } d _ { i } ^ { \mathrm { i n } } = \sum _ { i , j = 1 } ^ { N } a _ { i j } .\tag{2.2}
$$

A k-way clustering is a partition $\mathcal { C } = \{ C _ { 1 } , \ldots , C _ { k } \}$ of $V$ so that $V = \cup _ { j = 1 } ^ { k } C _ { j }$ , equivalently represented by the label signal $y : V \to \{ 1 , \dots , k \}$ with $\begin{array} { r } { C _ { j } = y ^ { - 1 } ( j ) } \end{array}$ . In the semi-supervised setting, labels are known only for vertices indexed by ${ \mathcal { T } } _ { \mathrm { k n w } } \subseteq \{ 1 , \dots , N \}$ so that $Y _ { \mathrm { k n w } } ( i ) \in \{ 1 , \dots , k \}$ is given in advance for $i \in \mathcal { T } _ { \mathrm { k n w } } . \ \mathrm { H } \ \mathcal { T } _ { \mathrm { k n w } } = \emptyset$ , then it corresponds to the unsupervised setting. The remaining assignments are inferred subject to these known labels.

k-Means. Given N observations $z _ { 1 } , z _ { 2 } , \dots , z _ { N } \in \mathbb { R } ^ { n }$ of n-dimensional vectors associated with vertices $v _ { 1 } , \ldots , v _ { N }$ of $V ,$ respectively, k-means clustering [40] aims to partition V into a k-way clustering ${ \mathcal { C } } =$ $\{ C _ { 1 } , C _ { 2 } , \ldots , C _ { k } \}$ of $V ,$ equivalently, to obtain a label signal $y : V \to \{ 1 , \dots , k \}$ , so as to minimize the within-cluster sum of squares. Formally, the objective function is given by

$$
\operatorname* { m i n } _ { \mathcal { C } = \{ C _ { j } = y ^ { - 1 } ( j ) \} _ { j = 1 } ^ { k } } \sum _ { j = 1 } ^ { k } \sum _ { v _ { i } \in C _ { j } } \left. z _ { i } - c _ { j } \right. _ { 2 } ^ { 2 } , \mathrm { ~ s . t . ~ } y | _ { \mathcal { T } _ { \mathrm { k n w } } } = Y _ { \mathrm { k n w } } ,\tag{2.3}
$$

where $\begin{array} { r } { c _ { j } = \frac { 1 } { | C _ { i } | } \sum _ { v _ { i } \in C _ { j } } z _ { i } } \end{array}$ is the centroid of the vertices in $C _ { j }$ . See Supplementary Material Part A.6 for more details. For a graph, the set $\{ z _ { 1 } , z _ { 2 } , \dots , z _ { N } \}$ is typically obtained via spectral embedding deduced from a graph operator, e.g., the graph Laplacian [15]. For digraphs, defining such a graph operator, reflecting both the connectivity and the edge orientation of the digraph, is key to accurate embedding for the subsequent k-means algorithm.

## B. Digraph Clustering

To preserve both reciprocal connectivity and asymmetric edge orientation, we propose a generalized Hermitian representation from the weighted adjacency matrix A. Specifically, the generalized Hermitian matrix is defined as

$$
\boldsymbol { W } = \alpha ( \boldsymbol { A } + \boldsymbol { A } ^ { \top } ) + \beta ( \boldsymbol { A } - \boldsymbol { A } ^ { \top } ) \mathrm { i } ,\tag{2.4}
$$

where $\mathrm { i } = \sqrt { - 1 } , \alpha \in [ 0 , 1 ]$ , and $\beta \in ( 0 , 1 ]$ . The symmetric component $A + A ^ { \top }$ characterizes the overall connectivity between two vertices, whereas the skew-symmetric component $A { - } A ^ { \intercal }$ captures the directional imbalance between opposite edges. Thus, α and $\beta$ control the contributions of reciprocal connectivity and edge directionality, respectively. Since $A + A ^ { \top }$ is symmetric and $A - A ^ { \top }$ is skew-symmetric, we have $\begin{array} { r } { W ^ { \mathbf { \bar { * } } } = \alpha ( A + A ^ { \top } ) ^ { \top } - \hat { \beta ( } A - A ^ { \top } ) ^ { \top } \mathbf { i } = W } \end{array}$ , where $( \cdot ) ^ { * }$ denotes the complex conjugate transpose. Therefore, W is Hermitian and admits an eigendecomposition

$$
{ \cal W } = U \Lambda U ^ { * } ,\tag{2.5}
$$

where $U = [ g _ { 1 } , \dots , g _ { N } ] \in \mathbb { C } ^ { N \times N }$ contains an orthonormal set of eigenvectors, and $\boldsymbol { \Lambda } = \operatorname { d i a g } ( \lambda _ { 1 } , \ldots , \lambda _ { N } )$ contains the corresponding real eigenvalues.

Remark 2.1: Satuluri and Parthasarathy in [16] used the bibliometric symmetrization $W = A A ^ { \top } + A ^ { \top } A$ for digraph representation. Chui et al. in [17] jointly use the bibliographic coupling matrix $A A ^ { \top }$ and the co-citation strength matrix $A ^ { \top } A$ as a pair $( A A ^ { \top } , A ^ { \top } A )$ to explore digraph signal representation as a 2-dimensional embedding. Cucuringu et al. in [22] proposed the Hermitian adjacency matrix $W =$ $( A - A ^ { \top } ) \mathrm { i }$ . Hayashi et al. [23] focus on the real skew-symmetric part $A - A ^ { \top }$ . Despite the advantages of symmetrization, most of the above methods lack a complete characterization of a digraph in terms of both connectivity and directional orientation. Our formulation of the generalized Hermitian matrix $W = \alpha ( A { + } A ^ { \top } ) { + } \bar { \beta } ( A { - } A ^ { \top } ) \mathrm { i }$ not only characterizes both connectivity $A + { \bar { A } } ^ { \top }$ and directional orientation $( A - A ^ { \top } ) \mathrm { i }$ , but also provides weights between the two characteristics, thereby enhancing the representation power of the a digraph for subsequent tasks.

Given a threshold $\epsilon > 0$ , we retain the eigenvectors whose eigenvalues satisfy $| \lambda _ { j } | > \epsilon$ . The corresponding spectral projection matrix is constructed as

$$
P = \sum _ { j \in \mathcal { I } _ { \epsilon } } g _ { j } \boldsymbol { g } _ { j } ^ { * } ,\tag{2.6}
$$

Algorithm 1 Spectral Clustering for Digraphs with the Generalized Hermitian Matrix   
Require: Digraph $\overline { { G } } = ( V , E , A ) ; k \geq 2 ; ( \mathbb { Z } _ { \mathrm { k n w } } , Y _ { \mathrm { k n w } } ) ; \epsilon > 0 ;$ hyperparameters $\alpha \in [ 0 , 1 ]$ and $\beta \in ( 0 , 1 ]$   
1: Construct the generalized Hermitian matrix:   
$W = \alpha ( A + \breve { A ^ { \top } } ) + \beta ( A - A ^ { \top } ) \mathrm { i }$   
2: Compute the eigenpairs $\{ ( \lambda _ { j } , g _ { j } ) \} _ { j \in \mathcal { T } _ { \epsilon } }$ of W with $\mathcal { T } _ { \epsilon } = \{ j : | \lambda _ { j } | > \epsilon \}$   
3: $P \gets \sum g _ { j } g _ { j } ^ { * }$   
$j \in \mathcal { I } _ { \epsilon }$   
4: Apply k-means on the matrix $[ \mathrm { R e } ( P )$ , Im $( P ) ]$ subject to $( \mathcal { T } _ { \mathrm { k n w } } , Y _ { \mathrm { k n w } } )$ as in (3)   
5: return A partition of V based on the output of k-means

Algorithm 2 SpecHDC: A Semi-supervised Spectral Hierarchical Digraph Clustering Algorithm.   
a) Input: A digraph $\overline { { G = ( V , E , A ) ; ( \mathcal { T } _ { \mathrm { k n w } } , Y _ { \mathrm { k n w } } ) } }$ if any; $\overline { { K = ( k _ { 1 } , k _ { 2 } , \ldots , k _ { L } ) } }$ satisfying $1 < k _ { 1 } <$   
$k _ { 2 } < \cdots < k _ { L } < N$ , where $N = | V | ;$ hyperparameters $\alpha \in [ 0 , 1 ] , \beta \in ( 0 , 1 ] .$ and $\epsilon > 0 .$   
b) Output: A hierarchical tree of levels $0 , 1 , \ldots , L + 1$ , where level 0 is the root $\{ V \}$ , level $L + 1$   
consists of the original vertices as leaves, and level l contains a kl-way clustering of $V$ for $l = 1 , \ldots , L .$   
c) Main Steps:   
1: Initialization: $l  L , V ^ { ( L + 1 ) }  V , A ^ { ( L + 1 ) }  A ;$   
2: while $l \geq 1$ do   
3: Apply Algorithm 1 to the current digraph with adjacency matrix $A ^ { ( l + 1 ) }$ and $k = k _ { l }$ , subject to the   
known pair $( \mathcal { T } _ { \mathrm { k n w } } , Y _ { \mathrm { k n w } } )$ resulting a $k _ { l } \mathrm { - w a y }$ clustering $\mathcal { C } _ { l } \overset { \cdot } { = } \{ C _ { 1 } ^ { ( l ) } , \ldots , C _ { k _ { l } } ^ { ( l ) } \}$ of V.   
4: Construct the coarse digraph $G ^ { ( l ) } = ( V ^ { ( l ) } , E ^ { ( l ) } , A ^ { ( l ) } )$ with $V ^ { ( l ) } = \{ C _ { 1 } ^ { ( l ) } , \ldots , C _ { k _ { l } } ^ { ( l ) } \}$ and adjacency   
matrix $A ^ { ( l ) } \in \mathbb { R } ^ { k _ { l } \times k _ { l } }$ given by (2.8).   
5: $l  l - 1 .$   
6: end while

where $\mathcal { T } _ { \epsilon } = \{ j : | \lambda _ { j } | > \epsilon \}$ . Each row of P provides a direction-aware spectral representation of a vertex. The rows of $P$ are used as $\{ z _ { 1 } , \dots , z _ { N } \}$ for the k-means algorithm to obtain a k-way clustering of the digraph. When a real-valued implementation is required, each complex row $P _ { i , : }$ can be represented as

$$
\widetilde { P } _ { i , : } = \left[ \mathrm { R e } ( P _ { i , : } ) , \mathrm { I m } ( P _ { i , : } ) \right] ,\tag{2.7}
$$

which preserves the Euclidean distance between the original complex representations.

The complete digraph clustering procedure is summarized in Algorithm 1. It serves as the basic single-level clustering operation in the proposed hierarchical framework. In the following subsection, this procedure is recursively applied to successively coarsened digraphs to construct nested partitions.

## C. Spectral Hierarchical Digraph Clustering

Algorithm 1 produces a digraph partition at a prescribed granularity. To capture structures at multiple granularities, we recursively apply it to successively coarsened digraphs. The resulting semi-supervised spectral hierarchical digraph clustering method, termed SpecHDC, is summarized in Algorithm 2.

Let $K = ( k _ { 1 } , \ldots , k _ { L } )$ satisfy $1 = k _ { 0 } < k _ { 1 } < \cdots < k _ { L } < k _ { L + 1 } = N$ The original vertices form level $L + 1$ form N clusters of singletons, level l contains $k _ { l }$ clusters from the level $l + 1$ , and level 0 is the root representing a single cluster {V} of all vertices. Starting from level $l = L + 1$ with input the initial digraph $G ^ { ( \breve { L } + 1 ) } ~ = ~ \stackrel {  } { G } ~$ Algorithm 1 is recursively applied to the current digraph $\boldsymbol { G } ^ { ( l + 1 ) ^ { - } } =$ $( V ^ { ( l + 1 ) } , E ^ { ( l + 1 ) } , \dot { A } ^ { ( l + 1 ) } )$ with $k = k _ { l }$ , yielding a $k _ { l } \mathrm { - w a y }$ clustering $\mathcal { C } _ { l }$ of $V .$ Note that the known labels in the pair $( \mathcal { T } _ { \mathrm { k n w } } , Y _ { \mathrm { k n w } } )$ with $Y _ { \mathrm { k n w } } ( \mathbb { Z } _ { \mathrm { k n w } } ) \subseteq \{ 1 , \dots , k ^ { \prime } \}$ for some $k ^ { \prime } \leq k _ { 1 }$ are imposed as fixed clusterassignment constraints during the k-means clustering. Specifically, let $\mathcal { C } _ { l } = \{ C _ { 1 } ^ { ( l ) } , \cdot \cdot \cdot , C _ { k _ { l } } ^ { ( l ) } \}$ be the kl-way clustering of $V = \{ v _ { 1 } , \ldots , v _ { N } \}$ obtained at level l. Then, it must satisfy $C _ { j } ^ { ( l ) } \supseteq Y _ { \mathrm { k n w } } ^ { - 1 } ( j )$ for $j = 1 , \ldots , k ^ { \prime }$

When $\mathcal { T } _ { \mathrm { k n w } } = \emptyset$ , SpecHDC reduces to the unsupervised hierarchical clustering. The partition $\mathcal { C } _ { l }$ is then contracted to construct the next coarser digraph $G ^ { ( l ) } = ( V ^ { ( l ) } , E ^ { ( l ) } , A ^ { ( l ) } )$ with $V ^ { ( l ) } = \mathcal { C } _ { l }$ . Each cluster becomes a coarse vertex, and the directed weight from $C _ { i } ^ { ( l ) }$ to $\dot { C } _ { j } ^ { ( l ) }$ is defined as

$$
A ^ { ( l ) } ( i , j ) = \frac { 1 } { \mathrm { v o l } ( G ) } \sum _ { C _ { u } ^ { ( l + 1 ) } \subseteq C _ { i } ^ { ( l ) } } \sum _ { C _ { v } ^ { ( l + 1 ) } \subseteq C _ { j } ^ { ( l ) } } A ^ { ( l + 1 ) } ( u , v ) ,\tag{2.8}
$$

for $1 \leq i , j \leq k _ { l }$ . Since the aggregation is performed separately for each ordered pair, $A ^ { ( l ) } ( i , j )$ and $A ^ { ( l ) } ( j , i )$ may differ, thereby preserving asymmetric inter-cluster interactions.

The resulting clusterings satisfy

$$
\mathcal { C } _ { L + 1 } \preceq \mathcal { C } _ { L } \preceq \mathcal { C } _ { L - 1 } \preceq \cdots \preceq \mathcal { C } _ { 1 } \preceq \mathcal { C } _ { 0 } ,\tag{2.9}
$$

where $\preceq$ denotes partition containment. These containment relations by the above bottom-up approach define a hierarchical tree (filtration), with the original vertices as leaves at the bottom and $\{ V \}$ as the root at the top. See Fig. 1 (top row, left to right).

![](images/73cae0fbea4fc660826f5ae1809968863502ae92fd295c4daaec33c5f26bcbbd.jpg)

![](images/a55bb52bd591dbdbf23768b8db1cbfb6313320581972f0f5131e848778e81d1c.jpg)

![](images/b69cb0517b782621bc5256affead8cd42bdd1920b115558d12835ecb75409b81.jpg)

![](images/bc78ef089a082c52c02f571c9690b4802dc4444cfcef3bafdd3e9d3c07cd3a27.jpg)

![](images/b81a5fcc3339a92d190755a0cf53551ab959c7173d00bbd1d562558a82425f47.jpg)

![](images/84548e54a85815acebd45e9873f6d7f283612e7026bbc069a2c1cc6d21340e2b.jpg)  
Fig. 1. The top presents the original digraph and the hierarchical clustering results obtained with our algorithm using $\alpha ( A { + } A ^ { T } ) { + } \beta ( A { - } A ^ { T } ) \mathrm { i }$ The bottom displays the corresponding rectangular partition of $I ^ { 2 } = [ 0 , \breve { 1 } ] \times [ 0 , 1 ]$ , derived from the hierarchical structure of the trees above.

## D. Hierarchy-Induced Interval Partitions

We now associate the hierarchy in (2.9) to a pair of nested interval partitions in a top-down manner. Specifically, along the in-degree and out-degree coordinates, respectively, from the root level 0 to the leaf level $L + 1$ , we associate each cluster $C _ { j } ^ { ( l ) }$ at each level with a block $\dot { R _ { j } ^ { ( l ) } } = I _ { j , \mathrm { i n } } ^ { ( l ) } \times I _ { j , \mathrm { o u t } } ^ { ( l ) }$ of sub-intervals in [0, 1]. Such a procedure also provides a sequence of nested knot sequences for the definition of splines in Section IV.

More precisely, at level 0, the root cluster $\{ V \}$ is assigned the full unit square $[ 0 , 1 ] \times [ 0 , 1 ]$ , that is, $I _ { \mathrm { i n } } ^ { ( 0 ) } : = [ a _ { \mathrm { i n } } ^ { ( 0 ) } , \dot { b } _ { \mathrm { i n } } ^ { ( 0 ) } ] = [ 0 , 1 ] = [ a _ { \mathrm { o u t } } ^ { ( 0 ) } , b _ { \mathrm { o u t } } ^ { ( 0 ) } ] = : \dot { I _ { \mathrm { o u t } } } ^ { ( 0 ) }$ . Now, from the top level going down along the tree in (2.9) for $l = 1 , \ldots , L + 1$ , suppose that a parent cluster $C ^ { ( l - 1 ) }$ at level l — 1 is associated with $[ a _ { \mathrm { i n } } ^ { ( l - 1 ) } , \dot { b _ { \mathrm { i n } } ^ { ( l - 1 ) } } ] \times [ a _ { \mathrm { o u t } } ^ { ( l - 1 ) } , \dot { b _ { \mathrm { o u t } } ^ { ( l - 1 ) } } ]$ at level l — 1 and it is formed from the ordered children clusters $C _ { 1 } ^ { ( l ) } , \ldots , C _ { k } ^ { ( l ) }$ at level l. That is, $C ^ { ( l - 1 ) } = \cup _ { j } C _ { j } ^ { ( l ) }$ . Let $G ^ { ( l ) } = ( V ^ { ( l ) } , E ^ { ( l ) } , A ^ { ( l ) } )$ be the digraph obtained at level l. Based on the adjacency matrix $A ^ { ( l ) } = [ a _ { u v } ^ { ( l ) } ]$ , the aggregate in-degree and out-degree for $u = C _ { j } ^ { ( l ) }$ are computed as

$$
d _ { \mathrm { i n } } \left( C _ { j } ^ { ( l ) } \right) = \sum _ { v \in V ^ { ( l ) } } a _ { v u } ^ { ( l ) } , \ d _ { \mathrm { o u t } } \left( C _ { j } ^ { ( l ) } \right) = \sum _ { v \in V ^ { ( l ) } } a _ { u v } ^ { ( l ) } .
$$

Then, the directional proportion (relative weight) of $C _ { j } ^ { ( l ) }$ among all the ordered children clusters can be defined as $\begin{array} { r } { \rho _ { j , \iota } ^ { ( l ) } = d _ { \iota } \left( C _ { j } ^ { ( l ) } \right) / \sum _ { i = 1 } ^ { j } d _ { \iota } \left( C _ { i } ^ { ( l ) } \right) } \end{array}$ for $\iota \in \{ \mathrm { i n } , \mathrm { o u t } \}$ . The parent intervals are then subdivided according to these proportions (weights). Specifically, setting $t _ { 0 , \iota } ^ { ( l ) } = a _ { \iota } ^ { ( l - 1 ) }$ , we define

$$
t _ { j , \iota } ^ { ( l ) } = a _ { \iota } ^ { ( l - 1 ) } + \big ( b _ { \iota } ^ { ( l - 1 ) } - a _ { \iota } ^ { ( l - 1 ) } \big ) \sum _ { i = 1 } ^ { j } \rho _ { i , \iota } ^ { ( l ) } , \iota \in \{ \mathrm { i n } , \mathrm { o u t } \} ,
$$

for $j = 1 , \dots , k$ . The cluster $C _ { i } ^ { ( l ) }$ is then associated with a block $I _ { j , \mathrm { i n } } ^ { ( l ) } \times I _ { j , \mathrm { o u t } } ^ { ( l ) }$ , where $I _ { j , \iota } ^ { ( l ) } = [ t _ { j - 1 , \iota } ^ { ( l ) } , t _ { j , \iota } ^ { ( l ) } ]$ for $\iota \in \{ \mathrm { i n } , \mathrm { o u t } \}$ . Applying this subdivision procedure to all parent clusters at level $( l - \overset { \vartriangle } { 1 } )$ produces the set $\mathcal { R } _ { l } = \{ R _ { j } ^ { ( l ) } = I _ { i , \mathrm { i n } } ^ { ( l ) } \times I _ { j , \mathrm { o u t } } ^ { ( l ) } : j = 1 , \dots , | V ^ { ( l ) } | \}$ of blocks at level l.

Note that $[ 0 , 1 ] = \cup _ { j = 1 } ^ { | V ^ { ( l ) } | } I _ { j , \iota } ^ { ( l ) }$ for any $\iota \in \{ \mathrm { i n } , \mathrm { o u t } \}$ and any $l = 0 , \ldots , L { + } 1$ . The resulting block sequence satisfies

$$
\mathcal { R } _ { 0 } \succeq \mathcal { R } _ { 1 } \succeq \mathcal { R } _ { 2 } \succeq \dots \succeq \mathcal { R } _ { L } \succeq \mathcal { R } _ { L + 1 } ,\tag{2.10}
$$

where $\succeq$ denotes partition inclusion. See Fig. 1 (bottom row, right to left).

## E. Illustration and Synthetic Examples

To illustrate the hierarchical clustering process in Section II, we use a synthetic digraph generated by the directed stochastic block model (DSBM). The DSBM was introduced by Cucuringu et al. in [22] (see also Supplementary Material Part A.5), where a directed graph $G = ( V , E , A )$ drawn from the model $\mathcal { G } ( k , n , p , q , F )$ has k clusters and $\vert V \vert = N = k \cdot n { \mathrm { ~ } } ( { \mathrm { i . e } }$ , each cluster has n vertices). The parameter $p$ controls the probability that there is an edge between two vertices within the same cluster, the parameter $q \in [ 0 , 1 ]$ controls the probability that there is an edge between two vertices belonging to two different clusters, and $F \in [ 0 , 1 ] ^ { k \times k }$ is a matrix where $F _ { j , j ^ { \prime } }$ denotes the probability of an edge from cluster $j$ to cluster $j ^ { \prime } { } .$ Necessarily, $F _ { j , j ^ { \prime } } + F _ { j ^ { \prime } , j } = 1$ , and $F _ { j , j } = 1 / 2$ . The pair $( p , q )$ affects the homophily ratio of the directed graph (see (3.1) and Section III-B).

As an example, we generate a graph from the model with $p = q = 0 . 5$ and $N = 3 0$ , using $k = 2$ and $n = 1 5$ . The cyclic pattern matrix is $\bar { \boldsymbol { F } } = \bigl [ \begin{array} { l l } { \frac { 1 } { 2 } } & { 1 - \eta } \\ { \eta } & { \frac { 1 } { 2 } } \end{array} \bigr ]$ , where $\begin{array} { r } { 0 \leq \eta \leq \frac { 1 } { 2 } } \end{array}$ controls the directional randomness between the two clusters. Smaller η indicates a stronger directional preference, while $\begin{array} { r } { \eta = \frac { 1 } { 2 } } \end{array}$ corresponds to no directional bias. We set $\eta = 0$ , giving $F = \left[ { 1 / 2 \atop 0 } { 1 / 2 } \right]$ . We then apply SpecHDC in the unsupervised setting with $\mathcal { T } _ { \mathrm { k n w } } = \emptyset$ and set $K = ( k _ { 1 } , k _ { 2 } ) = ( 2 , 5 )$ . Thus, Levels 1 and 2 contain 2 and 5 clusters, respectively, while Level 3 contains the 30 original vertices.

As shown in Fig. 1, the top row (from left to right) presents the original digraph at level 3 and the clustering results at levels 2 and 1, respectively. Vertices with the same color belong to the same cluster. The $N = 3 0$ vertices at level 3 are aggregated to 5 clusters at level 2, and they are further grouped into two larger clusters at level 1, demonstrating the nested structure $V = \mathcal { C } _ { 3 } \preceq \mathcal { C } _ { 2 } \preceq \mathcal { C } _ { 1 } \preceq \mathcal { C } _ { 0 } = \{ \bar { V } \}$ produced by the SpecHDC alogirithm (as a bottom-up procedure). Note that the root level $\mathcal { C } _ { 0 }$ is omitted as it is the single cluster $\{ V \}$ . The bottom row of Fig. 1 (from right to left) shows the induced hierarchy block partitions $\mathcal { R } _ { 0 } \succeq \mathcal { R } _ { 1 } \succeq \mathcal { R } _ { 2 } \succeq \mathcal { R } _ { 3 }$ (by a top-down procedure). At each level, the in-degree and out-degree intervals assigned to the same cluster are paired to form a rectangular cell. The level-1 partition $\mathcal { R } _ { 1 }$ (bottom right), therefore, contains two rectangle blocks, while the level-2 partition $\mathcal { R } _ { 2 }$ (bottom middle) contains five. At the finest level $\mathcal { R } _ { 3 }$ (bottom left), each original vertex is associated with an individual block cell. The successive subdivisions illustrate how the clustering hierarchy induces increasingly refined directional partitions and nested knot sequences for subsequent spline quasi-interpolation.

## III. EXPERIMENTS ON DIGRAPH CLUSTERING

We evaluate our method against representative spectral clustering approaches on both synthetic and realworld directed graphs. The experiments further include parameter sensitivity analysis and a comparison of different normalization strategies. Section III-B reports results on synthetic DSBM datasets, examining how clustering performance changes with key model parameters and revealing the structural conditions under which each method performs best. Section III-C presents evaluations on real-world networks under different training settings, comparing our method with the baselines across diverse graph types. Additional experimental details and results are provided in Supplementary Material Parts A and B.

## A. Experimental Setup

Datasets. The experiments are conducted on both synthetic and real-world datasets. Synthetic digraphs are generated using the DSBM with the five parameters: number of nodes N, number of clusters k, intracluster link probability $p ,$ inter-cluster link probability q, and the direction probability matrix $F ,$ allowing us to systematically control their structural properties. The entries of $F$ are restricted to $1 / 2 , \eta ,$ and $1 - \eta ,$ , corresponding to random, preferred, and reversed edge directions, respectively. Unless otherwise stated, we set $N \ = \ 5 0 0 0$ and $k = 5$ by default. In addition, we evaluate our model on seven realworld directed graph datasets with diverse scales, structural properties, and homophily levels (see (3.1)) These datasets span diverse application domains, including the Cora citation network [41], the Squirrel Wikipedia webpage network [42], the Telegram influence network [43], [33], the Cornell, Wisconsin, and Texas WebKB networks [44], and the political Blog network [45]. Since Telegram and Blog are unlabeled, their class counts and homophily ratios are reported as N/A. All datasets are publicly available and have been widely used in directed graph learning, supporting fair and reproducible evaluation. Due to space limitations, the experimental results for Squirrel, Cornell, and Texas are provided in Supplementary Material Part B.2, while detailed statistics and descriptions of all datasets are summarized in Supplementary Material Part A.2.

For a directed graph $G = ( V , E , A )$ with each node v with a label $y _ { v } \in \{ 1 , \ldots , k \}$ , its homophily ratio [46] is defined by

$$
\mathcal { H } _ { G } = \frac { 1 } { | V | } \sum _ { v \in V } h _ { v } ,\tag{3.1}
$$

with the node homophily ratio at v given by

$$
h _ { v } = \frac { 1 } { 2 } \Bigg ( \frac { | \{ u \in \stackrel { \left. } { \mathcal { N } } ( v ) : y _ { v } = y _ { u } \} | } { | \stackrel { \left. } { \mathcal { N } } ( v ) | } + \frac { | \{ u \in \stackrel { \right. } { \mathcal { N } } ( v ) : y _ { v } = y _ { u } \} | } { | \stackrel { \right. } { \mathcal { N } } ( v ) | } \Bigg ) ,
$$

where $\overleftarrow { \mathcal { N } } ( v )$ denotes the set of nodes that have edges pointing into node $v \in V$ , and $\vec { \mathcal { N } } ( v )$ is the set of nodes that have edges pointing out of node v.

Competitors. We benchmark our approach against five state-of-the-art spectral clustering methods designed for directed graphs: Bi-Sym [16], DD-Sym [16], DI-SIM [18], Herm [22], and Skew [23]. These unsupervised baselines primarily rely on matrix symmetrization or spectral decomposition to learn node embeddings. All methods are strictly evaluated under their original hyperparameter settings across synthetic DSBM graphs and the seven real-world directed networks. Additional details are provided in Supplementary Material Parts A.3.

Evaluation Metrics. We evaluate algorithm performance using three metrics: Adjusted Rand Index (ARI) [47], Modularity [48], and F-measure [16]. On synthetic graphs generated from the DSBM, we use ARI as the ground-truth communities are known. On real-world datasets, modularity and F-measure metrics are adopted for adapting both unsupervised and semi-supervised settings. Modularity does not require labels, whereas ARI and F-measure do. All metrics reflect how closely the recovered clusters match the ground truth: values near 1 indicate nearly perfect recovery, while values near 0 suggest nearly random partitioning.

![](images/c1d9758998e65bda7341f8b83ea3cbb2104817ff19f780386142c720b2e68d8d.jpg)

![](images/2380e30d61030b0ef4444e1ecb52ea3b51ceef4df1e01703a4e4934801d4821d.jpg)  
Fig. 2. Sensitivity of clustering performance to three key parameters $p , q , \eta$ on DSBM graphs. The top panel varies the within-cluster probability $p$ while fixing $q = 0 . 0 0 4 5$ , producing graphs with increasing homophily ratios. The bottom panel varies the between-cluster probability q while fixing $p = 0 . 0 0 4 5$ , producing graphs with decreasing homophily ratios

Implementation Details. All experiments are conducted on a workstation equipped with an NVIDIA GeForce RTX 2080 Ti GPU and 24 GB of memory. The proposed method is implemented in Python, with the eigendecomposition and clustering procedures performed using standard numerical and machinelearning libraries. Unless otherwise specified, the parameters $\alpha , \beta ,$ and € are selected through grid search on the validation data. For semi-supervised experiments, the labeled vertices are randomly sampled from each class to ensure that all classes are represented. Each experiment is repeated over multiple random splits, and the mean performance is reported.

## B. Results for the DSBM

Results and Analysis. We conduct experiments on graphs generated randomly from the DSBM with different values of $n , p , q .$ and matrix $F ,$ , as shown in Fig. 2. In these graphs, the gap between $p$ and $q$ ranges from twice to ten times. The generated directed graphs are either homophilic or heterophilic. Under both scenarios, we observe how other DSBM parameters affect the results and obtain several key findings. All reported results are averaged over 10 independently generated graphs for each fixed parameter set. Additional results are reported in Supplementary Material Part B.1.

Observation 1: Clear homophilic or heterophilic structures facilitate accurate clustering. Fig. 2 examines the performance of SpecHDC under different combinations of the DSBM parameters $p$ and $q .$ When $p \gg q .$ , edges are mainly formed within clusters, producing a pronounced homophilic structure. Conversely, $q \gg p$ yields a clear heterophilic structure dominated by inter-cluster connections. In both cases, the ARI increases as the difference between $p$ and $q$ becomes larger. When $p$ and $q$ are close, the cluster boundaries become less distinguishable and the performance decreases. Therefore, SpecHDC performs most reliably when the generated digraph exhibits a clear homophilic or heterophilic pattern.

Observation 2: Directional randomness mainly affects graphs with ambiguous structural patterns. Fig. 2 also shows the performance of SpecHDC as the directionality parameter $\eta$ increases. A larger $\eta$ introduces more randomness into edge directions and weakens the directional distinction between clusters. In the top panel, strongly homophilic graphs maintain ARI values close to 1 over the entire range of $\eta ,$ whereas the ARI values for weakly or moderately homophilic graphs deteriorate more rapidly. A similar trend is observed in the bottom panel: clearly, heterophilic graphs remain stable, while graphs with weaker inter-cluster separation become increasingly sensitive to directional randomness. These results indicate that structural separation can compensate for noisy edge directions, whereas ambiguous structures rely more strongly on consistent directional information.

Observation 3: The appropriate balance between $\alpha$ and $\beta$ depends on graph structure. Fig. 8 studies the sensitivity of the unnormalized SpecHDC to α and $\beta ,$ which control the symmetric and asymmetric components of the generalized Hermitian matrix, respectively. When $p = q$ , the model generally benefits from a stronger asymmetric component, although the overall performance remains limited by the weak cluster separation. A similar tendency is observed when $q \gg p .$ , where directional inter-cluster interactions provide the main clustering information. In contrast, when $p \gg q ,$ larger values of α or a more balanced combination of $\alpha$ and $\beta$ are preferable because the within-cluster connectivity itself becomes highly informative. Thus, no single parameter combination is optimal for all digraphs; α and $\beta$ should be balanced according to the relative importance of connectivity and directionality.

![](images/2f98760ffcd8c5f14924b75ea95088164634c21d76153c9a06a33d682ee58010.jpg)  
(a) η = 0, p = 0.0045, q = 0.0045

![](images/43d0014a646c380c5b967d413d7e6c0460fd2f55c097423b9e8451485ab85958.jpg)  
(b) η = 0, p = 0.0045, q = 0.05

![](images/f709e9d11a07b07f5d87d32ff7245176711424c92f53fc173c1af48dee1f33f9.jpg)  
(c) η = 0, p = 0.05, q = 0.0045  
Fig. 3. Investigating the effect of α and β: Model parameter sensitivity on different directed graph structures.

TABLE I  
CLUSTERING ALGORITHMS ON THE WISCONSIN DATASET: A COMPARATIVE ANALYSIS OF MODULARITY (M), AND F-MEASURE (F). THE BEST-PERFORMING MODEL IS HIGHLIGHTED IN BOLD, AND THE SECOND-BEST IS MARKED WITH AN UNDERLINE.
<table><tr><td>M</td><td>Trains (%)</td><td>0 (USL)</td><td>10</td><td>20</td><td>30</td><td>40</td><td>50</td><td>60</td><td>70</td><td>80</td><td>90</td></tr><tr><td>Bi-Sym</td><td>Level 2 (10) Level 1 (5)</td><td>0.2262 0.1524</td><td>0.2505 0.0971</td><td>0.2484 0.0848</td><td>0.1880</td><td>0.1549</td><td>0.0923 0.2787</td><td>0.0933</td><td>0.0327</td><td>0.0593</td><td>0.0330</td></tr><tr><td>DD-Sym</td><td>Level 2 (10)</td><td>0.5171</td><td>0.5031</td><td>0.3864</td><td>0.3868 0.1777</td><td>0.1712 0.2956</td><td>0.1919</td><td>0.1995 0.0829</td><td>0.0940 0.0299</td><td>0.2439 0.0415</td><td>0.0071 0.0048</td></tr><tr><td>DI-SIM</td><td>Level 1 (5) Level 2 (10)</td><td>0.5305 0.5192</td><td>0.2839 0.3690</td><td>0.3454 0.5065</td><td>0.2318 0.3297</td><td>0.0751 0.2122</td><td>0.0999 0.1023</td><td>0.2017 0.1046</td><td>0.1596 0.0007</td><td>0.2086 0.0012</td><td>0.1823 0.0421</td></tr><tr><td rowspan="2">Herm</td><td>Level 1 (5)</td><td>0.1281</td><td>0.3061</td><td>0.1822</td><td>0.3266</td><td>0.1528</td><td>0.1686</td><td>0.2333</td><td>0.1802</td><td>0.3917</td><td>0.0638</td></tr><tr><td>Level 2 (10)</td><td>0.1968</td><td>0.1769</td><td>0.1675</td><td>0.1613</td><td>0.1050</td><td>0.0622</td><td>0.0791</td><td>0.0302</td><td>0.0554</td><td>0.0340</td></tr><tr><td rowspan="2">Skew</td><td>Level 1 (5)</td><td>0.0635</td><td>0.0205</td><td>0.1077</td><td>0.1328</td><td>0.2164</td><td>0.1179</td><td>0.2759</td><td>0.1696</td><td>0.1617</td><td>0.1540</td></tr><tr><td>Level 2 (10)</td><td>0.1968</td><td>0.1769</td><td>0.1675</td><td>0.1613</td><td>0.1050</td><td>0.0622</td><td>0.0791</td><td>0.0302</td><td>0.0554</td><td>0.0340</td></tr><tr><td></td><td>Level 1 (5)</td><td>0.0116</td><td>0.0303</td><td>0.1265</td><td>0.1124</td><td>0.1933</td><td>0.2340</td><td>0.1506</td><td>0.0625</td><td>0.1497</td><td>0.0250</td></tr><tr><td rowspan="2">SpecHDC</td><td>Level 2 (10)</td><td>0.2478</td><td>0.4298</td><td>0.2378</td><td>0.1794</td><td>0.0892</td><td>0.2086</td><td>0.0993</td><td>0.0565</td><td>0.0075</td><td>0.0310</td></tr><tr><td>Level 1 (5)</td><td>0.4169</td><td>0.1416</td><td>0.1954</td><td>0.1938</td><td>0.2403</td><td>0.1024</td><td>0.1101</td><td>0.2356</td><td>0.2589</td><td>0.0873</td></tr><tr><td>F</td><td>Trains (%)</td><td>0 (USL)</td><td>10</td><td>20</td><td>30</td><td>40</td><td>50</td><td>60</td><td>70</td><td>80</td><td>90</td></tr><tr><td>Bi-Sym</td><td>Level 2 (10)</td><td>0.1718</td><td>0.1847</td><td>0.2339</td><td>0.3744</td><td>0.3522</td><td>0.4259</td><td>0.6316</td><td>0.6682</td><td>0.7444</td><td>0.8856</td></tr><tr><td rowspan="2">DD-Sym</td><td>Level 1 (5)</td><td>0.2698</td><td>0.5105</td><td>0.1763</td><td>0.2946</td><td>0.5118</td><td>0.6117</td><td>0.7082</td><td>0.7673</td><td>0.8487</td><td>0.9021</td></tr><tr><td>Level 2 (10)</td><td>0.0571</td><td>0.3820</td><td>0.1490</td><td>0.3276</td><td>0.4825</td><td>0.4485</td><td>0.5995</td><td>0.6905</td><td>0.7514</td><td>0.8688</td></tr><tr><td></td><td>Level 1 (5)</td><td>0.3835</td><td>0.1299</td><td>0.2125</td><td>0.3677</td><td>0.5833</td><td>0.5866</td><td>0.6478</td><td>0.7470</td><td>0.8169</td><td>0.8989</td></tr><tr><td rowspan="2">DI-SIM</td><td>Level 2 (10)</td><td>0.0650</td><td>0.4695</td><td>0.4715</td><td>0.2089</td><td>0.5614</td><td>0.4042</td><td>0.5201</td><td>0.6869</td><td>0.7465</td><td>0.9269</td></tr><tr><td>Level 1 (5)</td><td>0.1267</td><td>0.4978</td><td>0.5509</td><td>0.5017</td><td>0.6362</td><td>0.5877</td><td>0.7396</td><td>0.6890</td><td>0.8859</td><td>0.9477</td></tr><tr><td>Herm</td><td>Level 2 (10) Level 1 (5)</td><td>0.1336 0.3696</td><td>0.2222 0.3101</td><td>0.1930 0.2783</td><td>0.2501 0.4236</td><td>0.3164</td><td>0.5370</td><td>0.6029</td><td>0.6283</td><td>0.7458</td><td>0.8889</td></tr><tr><td rowspan="2">Skew</td><td>Level 2 (10)</td><td>0.1336</td><td>0.2222</td><td>0.1930</td><td>0.2501</td><td>0.6599 0.3164</td><td>0.6194 0.5370</td><td>0.6584</td><td>0.8237</td><td>0.8311</td><td>0.9016</td></tr><tr><td>Level 1 (5)</td><td>0.3904</td><td>0.3193</td><td>0.3540</td><td>0.4255</td><td>0.4989</td><td>0.5688</td><td>0.6029 0.7009</td><td>0.6283 0.8116</td><td>0.7458</td><td>0.8889</td></tr><tr><td rowspan="2">SpecHDC</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.8535</td><td>0.9111</td></tr><tr><td>Level 2 (10) Level 1 (5)</td><td>0.0493 0.5836</td><td>0.3010 0.4923</td><td>0.1377 0.5980</td><td>0.3245 0.6198</td><td>0.4856 0.6785</td><td>0.4323 0.7141</td><td>0.6300 0.7940</td><td>0.7305 0.7681</td><td>0.7672 0.8408</td><td>0.8982 0.9358</td></tr></table>

## C. Results for Real-World Data

We evaluate the competing methods on both labeled and unlabeled real-world digraphs. For labeled datasets, Modularity (M) and F-measure (F) are reported for labeled-node ratios ranging from 0% to 90%, where 0% corresponds to the unsupervised setting (ULS). Results are evaluated at two hierarchical resolutions: Level 1 contains five coarse clusters, while Level 2 contains ten finer clusters. The two metrics provide complementary views: M measures structural cohesion, whereas $\mathcal { F }$ evaluates agreement with the ground-truth classes.

Table I reports the results on Wisconsin. At Level 1, SpecHDC achieves the best $\mathcal { F }$ at labelednode ratios of 0%, 20%, 30%, 40%, 50%, and 60%, and obtains the second-best results at 10% and

TABLE ⅡI  
CLUSTERING ALGORITHMS ON THE TWO UNLABELED DATASETS: A COMPARATIVE ANALYSIS OF ARI (A). THE REPORTED ARI MEASURES CLUSTERING STABILITY ACROSS REPEATED UNSUPERVISED RUNS. THE BEST RESULTS ARE SHOWN IN BOLD, AND THE SECOND-BEST ARE UNDERLINED
<table><tr><td>Datasets</td><td>Bi-Sym</td><td>DD-Sym</td><td>DI-SIM</td><td>Herm</td><td>Skew</td><td>SpecHDC</td></tr><tr><td>telegram</td><td>0.91</td><td>0.95</td><td>0.47</td><td>0.89</td><td>0.89</td><td>0.97</td></tr><tr><td>blog</td><td>0.82</td><td>0.88</td><td>0.64</td><td>0.84</td><td>0.84</td><td>0.88</td></tr></table>

90%. Its score increases from 0.5836 without labels to 0.9358 at the 90% setting, showing that the inherited label constraints effectively improve the class consistency of the coarse partition. At Level 2, SpecHDC becomes increasingly competitive as more labels are provided, attaining the best $\mathcal { F }$ at 70% and 80% and the second-best result at 60% and 90%. This indicates that stronger supervision is particularly useful for distinguishing finer substructures. The modularity results exhibit a less regular trend because structurally cohesive communities do not necessarily coincide with the ground-truth classes. Nevertheless, SpecHDC achieves the highest M at the 50% and 70% settings of Level 2 and at the 70% setting of Level 1, while remaining competitive in several other cases. The results therefore show that the two hierarchy levels capture complementary properties: Level 1 is generally more consistent with the class partition, whereas Level 2 provides a finer description of the graph structure.

Table II reports ARI (A) on the unlabeled Telegram and Blog networks. Since no reference labels are available, ARI is used here to measure the consistency of the partitions across repeated unsupervised runs rather than their agreement with ground-truth classes. SpecHDC achieves the highest stability on Telegram with an ARI of 0.97, followed by DD-Sym at 0.95. On Blog, SpecHDC and DD-Sym jointly obtain the best score of 0.88. These results indicate that SpecHDC produces highly repeatable partitions on both unlabeled networks.

Overall, SpecHDC shows strong performance across unsupervised and semi-supervised settings, particularly for class-aligned coarse partitions, high-supervision fine partitions, and clustering stability on unlabeled digraphs. Due to space limitations, the results on Cora, Squirrel, Cornell, and Texas are reported in Supplementary Material Part B.2.

## IV. STABILITY OF SPLINE QUASI-INTERPOLATION WITH DOUBLING MEASURES

In this section, we give a brief introduction to B-splines and the quasi-interpolation operator. We then present the stability results for the quasi-interpolatory representation of a spline, which plays a key role for our digraph signal processing in Section V.

## A. Notation

We define $x _ { + } = \operatorname* { m a x } ( x , 0 )$ for $x \in \mathbb { R }$ , and for $r > 0 , x _ { + } ^ { r }$ $( x _ { + } ) ^ { r }$ . If I is a real interval, $| I |$ denotes the length of I. If $I = [ c - \ell , c + \ell ] , c \in \mathbb { R } , \ell > 0$ , and $\alpha > 0$ , we denote

$$
\alpha * I = [ c - \alpha \ell , c + \alpha \ell ] .\tag{4.1}
$$

For integer $k \geq 1$ , the space of all polynomials of degree $< k$ is denoted by $\Pi _ { k }$ (so that the dimension of $\Pi _ { k }$ is k).

For any non-empty set $K$ , (possibly signed) measure ν on $K$ , and ν-measurable function $f : K \to \mathbb { R }$ we define

$$
\| f \| _ { \nu , K ; p } = \left\{ \begin{array} { l l } { \left\{ \displaystyle \int _ { K } | f ( t ) | ^ { p } d | \nu | ( t ) \right\} ^ { 1 / p } , } & { \mathrm { ~ i f ~ } 1 \leq p < \infty , \medskip } \\ { | \nu | - \displaystyle \operatorname* { s s s } _ { t \in K } | f ( t ) | , } & { \mathrm { ~ i f ~ } p = \infty , } \end{array} \right.\tag{4.2}
$$

where $| \nu |$ denotes the total variation measure of ν. When $\nu$ is the Lebesgue measure or $p = \infty$ , we will drop the mention of $\nu .$ Likewise, we will drop the mention of K if $K = \mathbb { R }$ . If K is a topological space, support of a measure $\nu ,$ denoted by supp(ν) is defined to be the set of all points $x \in K$ such that $| \nu | ( B ) > 0$ for every neighborhood B of x.

For any sequence a (finite or infinite), we define

$$
\| \pmb { a } \| _ { p } = \left\{ \left\{ \sum _ { j } | a _ { j } | ^ { p } \right\} ^ { 1 / p } , \quad \mathrm { ~ i f ~ } 1 \leq p < \infty , \right.\tag{4.3}
$$

In the sequel, $m \geq 1$ is a fixed integer, and $\mu$ is a probability measure supported on $[ 0 , 1 ]$ . The notation $C \lesssim D$ denotes $C \le c D$ , where c is a generic positive constant depending only on $\mu , m$ , and the norms. The notation $C \gtrsim D$ means $D \lesssim C$ and $C \sim D$ means $C \lesssim D \lesssim \overline { { C } }$

## B. Splines

The material in this section is based on [49], [31], where the notation is different.

For real numbers $t _ { 1 } \leq \cdots \leq t _ { m }$ and a function g defined at these points, there exists a unique $P \in \Pi _ { m }$ such that $P ( t _ { k } ) = g ( t _ { k } )$ for $k = 1 , \ldots , m$ , where a derivative interpolation is understood when a knot $t _ { i }$ is repeated. The leading coefficient of this $P$ is denoted by $[ t _ { 1 } , \ldots , t _ { m } ] g$

Let $\xi _ { 1 } = 0$ and $\xi _ { n + 1 } = 1$ and

$$
\pm = ( \xi _ { 1 } = \cdot \cdot \cdot = \xi _ { m } < \xi _ { m + 1 } < \cdot \cdot \cdot < \xi _ { n + 1 } = \cdot \cdot \cdot = \xi _ { n + m } )\tag{4.4}
$$

be a non-decreasing sequence on [0, 1]. Although we will consider a nested sequence of such sequences later on, we consider $\pmb { \xi }$ to be a fixed sequence for the time being. The notation means that apart from $\xi _ { 1 } , \ldots , \xi _ { m - 1 }$ , and $\xi _ { n + 2 } , \ldots , \xi _ { n + m }$ , all other knots are distinct

Definition 1: The B-spline $B _ { j }$ based on these knots is defined by

$$
B _ { j } ( x ) = ( \xi _ { j + m } - \xi _ { j } ) [ \xi _ { j } , \dots , \xi _ { j + m } ] ( \xi _ { j } - x ) _ { + } ^ { m - 1 } , \qquad x \in \mathbb { R } .\tag{4.5}
$$

Let

$$
\mathbb { S } = \mathbb { S } _ { m , \pm } = \mathsf { s p a n } \{ B _ { j } \} _ { j = 1 } ^ { n + 1 } .\tag{4.6}
$$

A member of $\mathbb { S } _ { m , \xi }$ is called a spline of order m with the knot sequence $\xi .$

The following proposition summarizes some of the properties of B-splines.

Proposition 4.1: (a) The restriction $B _ { i ; j }$ of $B _ { j }$ to any $[ \xi _ { i } , \xi _ { i + 1 } ]$ is in $\Pi _ { m }$ . For any i, the restrictions $B _ { i ; j }$ $j = i - m + 1 , \ldots , i ,$ are a basis for $\Pi _ { m }$ (b) $B _ { j } ( x ) \geq 0$ for all $x \in \mathbb { R } . \ B _ { j } ( x ) = 0$ if $x \not \in [ \xi _ { j } , \xi _ { j + m } ]$ and $B _ { j } ( x ) > 0 { \mathrm { ~ i f ~ } } x \in ( \xi _ { j } , \xi _ { j + m } )$ . In particular, for any $j ,$ only $B _ { j - m + 1 } , \ldots , B _ { j }$ are non-zero on $[ \xi _ { j } , \xi _ { j + 1 } ]$ (c) $\begin{array} { r } { \sum _ { i } B _ { j } ( x ) = 1 } \end{array}$ for all $x \in [ \xi _ { m } , \xi _ { n } ]$

The splines $B _ { 1 } , \ldots , B _ { n }$ are linearly independent on $[ \xi _ { m } , \xi _ { n + 1 } ]$

Definition 2: An operator $\mathcal { Q }$ of the form

$$
\mathcal { Q } ( f ) = \sum _ { j } \lambda _ { j } ( f ) B _ { j }\tag{4.7}
$$

is called a quasi-interpolation operator if

(a) Each $\lambda _ { j }$ is a linear functional supported on $[ \xi _ { j } , \xi _ { j + m } ]$

(b) For every $P \in \Pi _ { m } , \mathcal { Q } ( P ) = P .$

Proposition 4.2: (a) If $\mathcal { Q }$ is a quasi-interpolation operator, where each $\lambda _ { j }$ is supported on only one of the intervals $[ \xi _ { i } , \xi _ { i + 1 } ]$ for some i, has the property that

$$
\mathcal { Q } ( S ) = S , \qquad S \in \mathbb { S } .\tag{4.8}
$$

In particular, if $\begin{array} { r } { S = \sum _ { j } c _ { j } B _ { j } } \end{array}$ , then $c _ { j } = \lambda _ { j } ( \boldsymbol { S } )$

(b) There exists a quasi-interpolatory operator $\begin{array} { r } { \mathcal { Q } ^ { * } ( f ) = \sum _ { j } \Lambda _ { j } ( f ) B _ { j } } \end{array}$ , which satisfies the conditions in part (a), so that (4.8) holds. Moreover,

$$
| \Lambda _ { j } ( f ) | \lesssim \| f \| _ { [ \xi _ { j } , \xi _ { j + m } ] } , \qquad j = 1 , \ldots , n .\tag{4.9}
$$

## C. Orthogonal polynomials

In the remainder of this section, let $\mu$ be a probability measure on $[ \xi _ { m } , \xi _ { n + 1 } ] \subseteq [ 0 , 1 ]$ . For an interval $I \subset [ 0 , 1 ]$ , let $\mu _ { I }$ be the restriction of $\mu$ to $I ,$ normalized again to be a probability measure. If there are at least $m + 1$ points in $\mathsf { s u p p } ( \mu ) \cap I ,$ , then there exists a unique basis $\{ p _ { k , I } \} _ { k = 0 } ^ { m - 1 }$ for $\Pi _ { m }$ such that each $p _ { k , I }$ has a positive leading coefficient and

$$
\int _ { I } p _ { k , I } p _ { j , I } d \mu _ { I } = { \frac { 1 } { \mu ( I ) } } \int _ { I } p _ { k , I } p _ { j , I } d \mu = { \left\{ \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ } } j = k , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }\tag{4.10}
$$

We say that $\mu$ is a doubling measure if for any $x \in [ 0 , 1 ] , 0 < y < 1 / 2$

$$
\mu \left( \left[ x - 2 y , x + 2 y \right] \right) \lesssim \mu \left( \left[ x - y , x + y \right] \right) .\tag{4.11}
$$

Lemma $4 . l \cdot$ Let $I \subseteq [ 0 , 1 ]$ be an interval, $\mu$ be a doubling measure with at least $m + 1$ points in $\mathsf { s u p p } ( \mu ) \cap I$ Then

$$
\sum _ { k = 0 } ^ { m - 1 } p _ { k , I } ^ { 2 } ( x ) \lesssim 1 , \qquad x \in I .\tag{4.12}
$$

The proof of this lemma depends upon two well-known facts, summarized in the following proposition. Part (a) is known as Markov inequality (cf., [50, Vol. 1, Chapter VI.§6]). Part (b) is a simple consequence of the Schwarz inequality and Parseval identity (cf. [51, Chapter I, Theorem 4.1]). Part (c) is a simple consequence of a conformal mapping argument (cf. [52, Chapter $^ { 4 , }$ formula (2.10)].

Proposition $4 . 3 \colon$ (a) For $P \in \Pi _ { m } , - \infty < a < b < \infty$ , we have

$$
\| P ^ { \prime } \| _ { \infty , [ a , b ] } \leq \frac { 2 m ^ { 2 } } { b - a } \| P \| _ { \infty , [ a , b ] } .\tag{4.13}
$$

(b) For any $x \in \mathbb { R }$

$$
\operatorname* { m a x } _ { P \in \Pi _ { m } } \int | P ( t ) | ^ { 2 } d \mu _ { I } ( t ) = \sum _ { k = 0 } ^ { m - 1 } p _ { k , I } ^ { 2 } ( x ) .\tag{4.14}
$$

(c) If $I \subseteq J \subset \mathbb { R }$ are compact intervals, then for any $P \in \Pi _ { m }$

$$
\begin{array} { r } { \| P \| _ { \infty , I } \leq \| P \| _ { \infty , J } \lesssim ( | J | / | I | ) ^ { m } \| P \| _ { \infty , I } . } \end{array}\tag{4.15}
$$

PROOF OF LEMMA 4.1.

Let $P \in \Pi _ { m }$ satisfy $\| P \| _ { \infty , I } = P ( x _ { 0 } )$ for some $x _ { 0 } \in I$ Since $P ^ { 2 } \in \Pi _ { 2 m }$ , the Markov inequality (4.13) implies that

$$
\begin{array} { r l r } & { } & { \displaystyle { \left| | P ( t ) | ^ { 2 } - | P ( x _ { 0 } ) | ^ { 2 } \right| \le \frac { 8 m ^ { 2 } } { | I | } | t - x _ { 0 } | \| P \| _ { \infty , I } ^ { 2 } } } \\ & { } & { \displaystyle { = \frac { 8 m ^ { 2 } } { | I | } | t - x _ { 0 } | | P ( x _ { 0 } ) | ^ { 2 } } . } \end{array}
$$

Consequently, if $I ^ { \prime } = ( ( 1 6 m ^ { 2 } ) ^ { - 1 } * I ) \cap I .$ then $| P ( t ) | ^ { 2 } \geq \| P \| _ { \infty , I } ^ { 2 } / 2$ for all $t \in I ^ { \prime }$ . Hence,

$$
\int _ { I } | P ( t ) | ^ { 2 } d \mu _ { I } ( t ) \geq \int _ { I ^ { \prime } } | P ( t ) | ^ { 2 } d \mu _ { I } ( t ) \geq \frac { \| P \| _ { \infty , I } ^ { 2 } } { 2 } \mu _ { I } ( I ^ { \prime } ) .\tag{4.16}
$$

Since $\mu$ is a doubling measure, we observe that

$$
\mu _ { I } ( I ^ { \prime } ) = \frac { \mu ( I ^ { \prime } ) } { \mu ( I ) } \gtrsim 1 .
$$

$\mathrm { S o } ,$ we have proved that for any $P \in \Pi _ { m }$

$$
\int _ { I } | P ( t ) | ^ { 2 } d \mu _ { I } ( t ) \gtrsim \| P \| _ { \infty , I } ^ { 2 } .
$$

The extremal principle (4.14) now leads to the estimate (4.12).

## D. Stability theorem

The purpose of this section is to prove the following generalization of the well-known de Boor stability theorem.

Theorem 4.1: Let $\begin{array} { r } { S = \sum _ { j \in \mathbb { Z } } c _ { j } B _ { j } \in \mathbb { S } . } \end{array}$ and $\mu$ be a doubling measure. then for $1 \leq p < \infty$

$$
\sum _ { j \in \mathbb { Z } } \mu ( I _ { j } ) | c _ { j } | ^ { p } \sim \int _ { \xi _ { m } } ^ { \xi _ { n + 1 } } | f ( t ) | ^ { p } d \mu ( t ) ,\tag{4.17}
$$

with an obvious modification in the case when $p = \infty$

Following [31], we first obtain a quasi-interpolatory representation of a spline. For $m \le j \le n$ , we denote $I _ { j } = [ \xi _ { j } , \xi _ { j + 1 } ]$ , and Let $j \leq j ^ { * } < j + m$ be chosen so that

$$
| I _ { j ^ { * } } | = \operatorname* { m a x } _ { j \leq \ell < j + m } | I _ { \ell } | .
$$

We note that

$$
I _ { j } ^ { \ast } \subseteq [ \xi _ { j } , \xi _ { j + m } ] \subset ( 2 m ) \ast I _ { j ^ { \ast } } .\tag{4.18}
$$

With the notation as in Proposition 4.2(b), we define

$$
\lambda _ { j } ( f ) = \int _ { I _ { j ^ { * } } } \left\{ \sum _ { k = 0 } ^ { m - 1 } \Lambda _ { j } ( p _ { k , I _ { j ^ { * } } } ) p _ { k , I _ { j ^ { * } } } ( x ) \right\} f ( x ) d \mu _ { I _ { j ^ { * } } } ( x ) ,\tag{4.19}
$$

and

$$
\mathcal { Q } ( f ) ( x ) = \sum _ { j } \lambda _ { j } ( f ) B _ { j } ( x ) , \qquad x \in [ x _ { m } , x _ { n + 1 } ] .\tag{4.20}
$$

Lemma $4 . 2 \colon ( \mathrm { a } ) \ \mathcal { Q }$ defined in (4.20) is a quasi-interpolation operator in the sense of Definition 2. (b) We have $Q ( S ) = S$ for all $S \in \mathbb S$

(c) For $1 \leq p < \infty$ , we have

$$
\sum _ { j = m } ^ { n + 1 } \mu ( I _ { j } ) | \lambda _ { j } ( f ) | ^ { p } \lesssim \int _ { \xi _ { m } } ^ { \xi _ { n + 1 } } | f ( x ) | ^ { p } d \mu ( x ) .\tag{4.21}
$$

For $p = \infty$ , we have

$$
\operatorname* { m a x } _ { m \leq j \leq n + 1 } \lambda _ { j } ( f ) \vert \lesssim \| f \| _ { \infty ; [ \xi _ { m } , \xi _ { n + 1 } ] } .\tag{4.22}
$$

PROOF OF LEMMA 4.2

Obviously,

$$
\lambda _ { j } ( p _ { \ell , I _ { j ^ { * } } } ) = \Lambda _ { j } ( p _ { \ell , I _ { j ^ { * } } } ) , \qquad \ell = 0 , \cdot \cdot \cdot , m - 1 .
$$

It follows from Proposition 4.2(b) that

$$
\begin{array} { r } { \boldsymbol { \mathcal { Q } } ( \boldsymbol { P } ) = \boldsymbol { P } , \qquad \boldsymbol { P } \in \Pi _ { m } . } \end{array}
$$

Thus, $\mathcal { Q }$ is a quasi-interpolation operator. This proves part (a).

Since $\Lambda _ { j }$ is supported on only one interval $I _ { i ^ { * } }$ , Proposition 4.2 implies part (b).

In this part of the proof, let $\begin{array} { r } { R _ { x } ( y ) = \sum _ { k = 0 } ^ { m - 1 } p _ { k , I _ { j ^ { * } } } ( y ) p _ { k , I _ { j ^ { * } } } ( x ) } \end{array}$ . The Schwarz inequality yields that

$$
| R _ { x } ( y ) | \leq \left\{ \sum _ { k = 0 } ^ { m - 1 } p _ { k , I _ { j ^ { * } } } ( y ) ^ { 2 } \right\} ^ { 1 / 2 } \left\{ \sum _ { k = 0 } ^ { m - 1 } p _ { k , I _ { j ^ { * } } } ( x ) ^ { 2 } \right\} ^ { 1 / 2 } ,
$$

so that, in view of Lemma 4.1,

$$
\operatorname* { m a x } _ { x \in I _ { j ^ { * } } } \| R _ { x } \| _ { \infty , I _ { j ^ { * } } } \lesssim 1 .
$$

In view of Proposition 4.3(c) and (4.18), we deduce that

$$
\operatorname* { m a x } _ { x \in I _ { j ^ { * } } } \| R _ { x } \| _ { \infty , [ \xi _ { j } , \xi _ { j + m } ] } \lesssim 1 .
$$

Consequently, (4.9) implies that for all $x \in I _ { j ^ { * } }$

$$
\left| \sum _ { k = 0 } ^ { m - 1 } \Lambda _ { j } ( p _ { k , I _ { j ^ { * } } } ) p _ { k , I _ { j ^ { * } } } ( x ) \right| = | \Lambda _ { j } \left( R _ { x } \right) | \lesssim 1 .
$$

Therefore, a straightforward application of Hölder inequality in the definition (4.19), taking into account the fact that $\mu$ is a doubling measure, implies that for $1 \leq p < \infty$

$$
\begin{array} { l } { { \displaystyle | \lambda _ { j } ( f ) | ^ { p } \lesssim \int _ { I _ { j } ^ { * } } | f ( x ) | ^ { p } d \mu _ { I _ { j } ^ { * } } ( x ) = \frac { 1 } { \mu ( I _ { j ^ { * } } ) } \int _ { I _ { j } ^ { * } } | f ( x ) | ^ { p } d \mu ( x ) } } \\ { { \displaystyle \lesssim \frac { 1 } { \mu ( I _ { j } ) } \int _ { I _ { j } ^ { * } } | f ( x ) | ^ { p } d \mu ( x ) } . } \end{array}\tag{4.23}
$$

The estimate (4.21) is obtained by adding these inequalities, $j = m , \cdots , n + 1$ . The estimate (4.22) is simpler.■

PROOF OF THEOREM 4.1

Since $\{ B _ { j } \}$ is a basis for S, it follows from Lemma 4.2(b) that for $\begin{array} { r } { S = \sum _ { j } c _ { j } B _ { j } , c _ { j } = \lambda _ { j } ( S ) } \end{array}$ . Hence, Lemma 4.2(c) leads immediately to the bound in (4.17) for estimating the discrete norm of $\mathbf { c } = \{ c _ { j } \}$ in terms of the norm of S. In the reverse direction, we note that since $B _ { j } \geq 0$ and $\textstyle \sum _ { j } B _ { j } \equiv 1$

$$
\| S \| _ { \infty } \leq \left\| \sum _ { j } c _ { j } B _ { j } \right\| \leq \| \mathbf { c } \| _ { \infty } .\tag{4.24}
$$

Further, since each $B _ { j }$ is supported on $[ \xi _ { j } , \xi _ { j + m } ] , 0 \leq B _ { j } \leq 1$ , and $\mu$ is a doubling measure (cf. (4.18)),

$$
\int _ { \xi _ { m } } ^ { \xi _ { n + 1 } } | s | d \mu \leq \sum _ { j } | c _ { j } | \int _ { \xi _ { j } } ^ { \xi _ { j + m } } B _ { j } d \mu \lesssim \sum _ { j } | c _ { j } | \mu ( I _ { j } ) .
$$

Together with (4.24), this proves the other inequality in (4.17) for $p = 1 , \infty$ . The general case follows by applying the Riesz-Thorin interpolation theorem.

## V. DIGRAPH SIGNAL PROCESSING VIA SPLINE QUASI-INTERPOLANTS

The hierarchical partitions produced by the SpecHDC algorithm induce a hierarchy of block partitions. In this section, we shall apply this hierarchy to construct the splines and their associated quasi-interpolation operators for digraph signal processing.

## A. Digraph Signal Processing

Specifically, given a digraph signal $f : V \to \mathbb { R }$ defined on a vertex set V of a digraph $G = ( V , E , A )$ we first apply our SpecHDC algorithm to form a hierarchical tree (filtration) and then it induces the hierarchy $\mathcal { R } _ { 0 } \succeq \mathcal { R } _ { 1 } \succeq \dots \succeq \mathcal { R } _ { L } \succeq \mathcal { R } _ { L + 1 }$ of blocks, where each block in $\mathcal { R } _ { l }$ is given by a pair of directional intervals. At each level l, we convert $\mathcal { R } _ { l }$ into a pair $( \pmb { \xi } _ { l , \mathrm { i n } } , \pmb { \xi } _ { l , \mathrm { o u t } } )$ of knot sequences, which are then used to construct the pair $( \{ B _ { j , \mathrm { i n } } ^ { ( l ) } \} _ { j = 0 } ^ { k _ { l } } , \{ B _ { j , \mathrm { o u t } } ^ { ( l ) } \} _ { j = 0 } ^ { k _ { l } } )$ of spline bases and the corresponding pair $( \mathcal { Q } _ { \mathrm { i n } } ^ { ( l ) } , \mathcal { Q } _ { \mathrm { o u t } } ^ { ( l ) } )$ of quasi-interpolants at level l, as described in Section IV. Due to the hierarchy property, the knot sequences are nested for each direction. The differences $\mathcal { Q } _ { \iota } ^ { ( l + 1 ) } f - \mathcal { Q } _ { \iota } ^ { ( l ) } f , \iota \in \{ \mathrm { i n } , \mathrm { o u t } \}$ , between successive approximations provide detailed information about the graph signal $f$ and can be used for graph signal denoising.

Knot Sequençes from Multi-Level Clusters. Fix m to be the dimension of the polynomial space $\Pi _ { m }$ Let $\mathcal { R } _ { l } \stackrel { \cdot } { = } \{ R _ { j } ^ { ( l ) } = I _ { j , \mathrm { i n } } ^ { ( l ) } \times I _ { j , \mathrm { o u t } } ^ { ( l ) } : j = 1 , \ldots , k _ { l } \}$ be the rectangular blocks at level l. Note that $[ 0 , 1 ] =$ $\cup _ { i = 1 } ^ { k _ { l } } I _ { i , \iota } ^ { ( l ) }$ for any $\iota \in \{ \mathrm { i n } , \mathrm { o u t } \}$ and any $l = 0 , \ldots , L + 1$ . We collect and order the distinct endpoints of $\{ \check { I } _ { j , \iota } : \dot { j } = 1 , \dots , k _ { l } \}$ in Section II-D as

$$
\pmb { \xi } _ { l , \iota } = \big ( \xi _ { 0 , \iota } ^ { ( l ) } = \cdot \cdot \cdot = \xi _ { 0 , \iota } ^ { ( l ) } < \xi _ { 1 , \iota } ^ { ( l ) } < \cdot \cdot \cdot < \xi _ { k _ { l } , \iota } ^ { ( l ) } = \cdot \cdot \cdot = \xi _ { k _ { l } , \iota } ^ { ( l ) } \big )
$$

for $\iota \in \{ \mathrm { i n } , \mathrm { o u t } \}$ , where $\xi _ { 0 , \iota } ^ { ( l ) } = 0$ and $\xi _ { k _ { l } , \iota } ^ { ( l ) } = 1$ are repeated $m + 1$ times for the construction of B-splines of order m. Since the intervals at level l are obtained by subdividing those at level $l - 1$ , the knot sequences satisfy

$$
\pmb { \xi } _ { 0 , \iota } \subseteq \pmb { \xi } _ { 1 , \iota } \subseteq \cdots \subseteq \pmb { \xi } _ { L , \iota } \subseteq \pmb { \xi } _ { L + 1 , \iota } .\tag{5.1}
$$

Each knot sequence $\xi _ { l , \iota }$ then induces a spline basis $\{ B _ { j , \iota } ^ { ( l ) } \} _ { j = 0 } ^ { k _ { l } }$ for the space $\mathbb { S } _ { m , \pmb { \xi } _ { l , \iota } }$

For the finest level $L + 1$ , we associate vertex $v _ { i }$ with the coordinate $x _ { i , \ i } = \xi _ { i , \ i } ^ { ( L + 1 ) } , i = 1 , \ i . . . , N$ . The graph signal $f : V \to \mathbb { R }$ can be represented as a spline function $\begin{array} { r } { f _ { \iota } ( t ) = \sum _ { j = 0 } ^ { k _ { l } } c _ { j , \iota } ^ { ( L + 1 ) } B _ { j , \iota } ^ { ( L + 1 ) } ( t ) \in \mathbb { S } _ { m , \xi _ { L + 1 , 4 } } } \end{array}$ for $t \in [ 0 , 1 ]$ via the spline interpolation algorithm [49]. That is, $f _ { \iota } ( x _ { i , \iota } ) = ^ { \prime } f ( v _ { i } )$ for $\dot { \iota } \in \{ \mathrm { i n } , \mathrm { o u t } \}$ Spline Quasi-Interpolation for Graph Signals. Given $f _ { \iota }$ and the spline basis $\{ B _ { j , \iota } ^ { ( l ) } \} _ { j = 0 } ^ { k _ { l } }$ , by Definition 2, the corresponding spline quasi-interpolant for $f _ { \iota }$ at level l is given by

$$
\mathcal { Q } _ { \iota } ^ { ( l ) } f _ { \iota } ( t ) = \sum _ { j = 0 } ^ { k _ { l } } \lambda _ { j , \iota } ^ { ( l ) } ( f _ { \iota } ) B _ { j , \iota } ^ { ( l ) } ( t ) , \quad l = 1 , \ldots , L + 1 ,\tag{5.2}
$$

where $\lambda _ { j , \iota } ^ { ( l ) }$ is a local linear functional determined by the signal samples within the support of $B _ { j , \iota } ^ { ( l ) }$ . The compact support of the B-spline basis allows the coefficients to be computed locally without solving a global interpolation system (see [31]). Evaluating the quasi-interpolant $\mathcal { Q } _ { \iota } ^ { ( l ) } f _ { \iota }$ at the (finest) vertex coordinates $x _ { i , \iota }$ gives a quasi-interpolant graph signal $\mathbf { \mathit { L } } _ { l , \iota }$ as

$$
\pmb { L } _ { l , \iota } = \left[ \mathcal { Q } _ { \iota } ^ { ( l ) } f ( x _ { 1 , \iota } ) , \ldots , \mathcal { Q } _ { \iota } ^ { ( l ) } f ( x _ { N , \iota } ) \right] ^ { \top } .\tag{5.3}
$$

Note that a finer knot sequence captures more local variations, whereas a coarser sequence produces a smoother structural approximation.

Multi-level Digraph Signal Denoising. Let $\pmb { y } _ { T } = [ f ( v _ { 1 } ) , \dots , f ( v _ { N } ) ] ^ { \top }$ denote the underlying clean digraph signal (ground truth). Its noisy observation is modeled as

$$
{ \pmb y } _ { O } = { \pmb y } _ { T } + \sigma { \pmb \varepsilon } ,\tag{5.4}
$$

TABLE III  
DENOISING PERFORMANCE OF GRAPH SIGNALS UNDER VARYING NOISE RATIOS, COMPARING RAW CORRUPTED SIGNALS WITH SPLINE-SMOOTHED RECONSTRUCTIONS USING MULTI-RESOLUTION HIERARCHICAL STRUCTURES.
<table><tr><td>Noise Level</td><td>Signal Pairs</td><td>Metrics</td><td>Cora</td><td>Cornell</td><td>Texas</td><td>Wisconsin</td><td>Squirrel</td></tr><tr><td rowspan="5">5%</td><td rowspan="2">yT and yo</td><td>RMSE</td><td>0.0053</td><td>0.0078</td><td>0.0078</td><td>0.0068</td><td>0.0046</td></tr><tr><td>SNR</td><td>26.1595</td><td>26.4977</td><td>26.4977</td><td>26.3257</td><td>26.0635</td></tr><tr><td rowspan="2">yT and  $\pm \tilde { \mathbf { { d } } } _ { 2 } + \tilde { \mathbf { { d } } } _ { 2 }$ </td><td>RMSE</td><td>0.0041</td><td>0.0056</td><td>0.0063</td><td>0.0044</td><td>0.0023</td></tr><tr><td>SNR</td><td>28.2786</td><td>29.3331</td><td>28.3221</td><td>30.0617</td><td>31.9974</td></tr><tr><td rowspan="2">yT and  ${ \pmb L } _ { 1 } + \tilde { { \pmb d } } _ { 1 } + \tilde { { \pmb d } } _ { 2 }$ </td><td>RMSE</td><td>0.0035</td><td>0.0056</td><td>0.0063</td><td>0.0044</td><td>0.0023</td></tr><tr><td>SNR</td><td>29.6455</td><td>29.3331</td><td>28.3221</td><td>30.0617</td><td>31.9974</td></tr><tr><td>Noise Level</td><td>Signal Pairs</td><td>Metrics</td><td>Cora</td><td>Cornell</td><td>Texas</td><td>Wisconsin</td><td>Squirrel</td></tr><tr><td rowspan="5">10%</td><td>yT and yo</td><td>RMSE</td><td>0.0080</td><td>0.0156</td><td>0.0156</td><td>0.0136</td><td>0.0073</td></tr><tr><td rowspan="2">yT and</td><td>SNR</td><td>20.1389</td><td>20.4771</td><td>20.4771</td><td>20.3051</td><td>20.0429</td></tr><tr><td>RMSE</td><td>0.0082</td><td>0.0113</td><td>0.0127</td><td>0.0089</td><td>0.0037</td></tr><tr><td rowspan="2"> $\pm \tilde { \mathbf { { d } } } _ { 2 } + \tilde { \mathbf { { d } } } _ { 2 }$ </td><td>SNR</td><td>22.2583</td><td>23.3126</td><td>22.3091</td><td>24.0513</td><td>26.0014</td></tr><tr><td>RMSE</td><td>0.0070</td><td>0.0113</td><td>0.0127</td><td>0.0089</td><td>0.0037</td></tr><tr><td>Noise Level</td><td>yT and  ${ \pmb L } _ { 1 } + \tilde { d } _ { 1 } + \tilde { d } _ { 2 }$  Signal Pairs</td><td>SNR</td><td>23.6250</td><td>23.3126</td><td>22.3091</td><td>24.0513</td><td>26.0014</td></tr><tr><td></td><td></td><td>Metrics</td><td>Cora</td><td>Cornell</td><td>Texas</td><td>Wisconsin</td><td>Squirrel</td></tr><tr><td rowspan="5">15%</td><td>yT and yo</td><td>RMSE</td><td>0.0158</td><td>0.0235</td><td>0.0235</td><td>0.0204</td><td>0.0139</td></tr><tr><td rowspan="2">yT and  $\pm \hat { \mathbf { { d } } } _ { 2 } + \tilde { \mathbf { { d } } } _ { 2 }$ </td><td>SNR</td><td>16.6171</td><td>16.9552</td><td>16.9552</td><td>16.7833</td><td>16.5211</td></tr><tr><td>RMSE</td><td>0.0124</td><td>0.0169</td><td>0.0190</td><td>0.0133</td><td>0.0070</td></tr><tr><td rowspan="2"> ${ \pmb L } _ { 1 } + \tilde { d } _ { 1 } + \tilde { d } _ { 2 }$ </td><td>SNR</td><td>18.7367</td><td>19.7906</td><td>18.7943</td><td>20.5393</td><td>22.4556</td></tr><tr><td>RMSE</td><td>0.0106</td><td>0.0169</td><td>0.0190</td><td>0.0133</td><td>0.0070</td></tr><tr><td>Noise Level</td><td>yT and Signal Pairs</td><td>SNR</td><td>20.1034</td><td>19.7906</td><td>18.7943</td><td>20.5393</td><td>22.4556</td></tr><tr><td rowspan="6">20%</td><td></td><td>Metrics</td><td>Cora</td><td>Cornell</td><td>Texas</td><td>Wisconsin</td><td>Squirrel</td></tr><tr><td rowspan="2">yT and  $_ { \mathbf { \nabla } \mathbf { \boldsymbol { y } } _ { O } }$ </td><td>RMSE</td><td>0.0211</td><td>0.0313</td><td>0.0313</td><td>0.0273</td><td>0.0185</td></tr><tr><td>SNR</td><td>14.1183</td><td>14.4565</td><td>14.4565</td><td>14.2845</td><td>14.0223</td></tr><tr><td rowspan="2">yT and  $\pm \hat { \mathbf { { d } } } _ { 2 } + \tilde { \mathbf { { d } } } _ { 2 }$ </td><td>RMSE</td><td>0.0165</td><td>0.0226</td><td>0.0253</td><td>0.0177</td><td>0.0093</td></tr><tr><td>SNR</td><td>16.2381</td><td>17.2915</td><td>16.3010</td><td>18.0501</td><td>19.9570</td></tr><tr><td rowspan="2">yT and  ${ \pmb L } _ { 1 } + \tilde { d } _ { 1 } + \tilde { d } _ { 2 }$ </td><td>RMSE</td><td>0.0141</td><td>0.0226</td><td>0.0253</td><td>0.0177</td><td>0.0093</td></tr><tr><td>SNR</td><td>17.6025</td><td>17.2915</td><td>16.3010</td><td>18.0501</td><td>19.9570</td></tr></table>

where $\sigma = \mathrm { m a x } ( | y _ { T } | ) r , r$ is the (relative) noise level, and $\varepsilon$ is a standard Gaussian noise vector satisfying $\varepsilon \sim \mathcal { N } ( 0 , 1 )$ . Only $_ { \mathbf { \nabla } \mathbf {  } \mathbf {  } \mathbf {  } \mathbf {  } \mathbf {  } \mathbf {  } \mathbf {  } \mathbf { \sigma }   }$ is available during reconstruction, while ${ \mathbf { } } ^ { y _ { T } }$ is used solely to evaluate the recovery quality. Applying the spline quasi-interpolation operator to $_ { \mathbf { \nabla } \mathbf {  } \mathbf {  } \mathbf {  } \mathbf {  } \mathbf {  } \mathbf {  } \mathbf {  } \mathbf { \sigma }   }$ at each resolution gives the approximation vectors $\pmb { L } _ { 1 , \iota } , \dots , \pmb { L } _ { L + 1 , \iota }$ in $\mathbb { R } ^ { \bar { N } }$ for $\iota \in \{ \mathrm { i n } , \mathrm { o u t } \}$ . The detail between two consecutive resolutions is defined as $d _ { j , \iota } = L _ { j + 1 , \iota } - L _ { j , \iota } , j = 1 , \ldots , L$ . Each $d _ { j , \iota }$ contains the details at resolution $j .$ Note that for any $j _ { 0 } \in \{ 2 , \dots , J \}$ , we have the decomposition

$$
\pmb { L } _ { L + 1 , \iota } = \pmb { L } _ { j _ { 0 } , \iota } + \sum _ { j = j _ { 0 } } ^ { L } d _ { j , \iota } .\tag{5.5}
$$

For the noisy observation, these detail components $d _ { j , \iota }$ are typically sparse and contain both useful local features and noise. To suppress noise, we apply an adaptive thresholding operator $\tau _ { j }$ on $d _ { j , \iota }$ to obtain $\widetilde { \pmb { d } } _ { j , \iota } = \mathcal { T } _ { j } ( \pmb { d } _ { j , \iota } ) , j = 1 , \dots , L$ . The threshold is adjusted according to the estimated noise level and local residual activity, thereby suppressing weak noisy fluctuations while preserving significant signal variations. From $\mathbf { L } _ { j _ { 0 } , N }$ as the coarse structural component, the denoised signal is reconstructed as

$$
\widetilde { \pmb { y } } _ { j _ { 0 } : L + 1 , \iota } = \pmb { L } _ { j _ { 0 } , \iota } + \sum _ { j = j _ { 0 } } ^ { L } \widetilde { d } _ { j , \iota } , \quad j _ { 0 } = 1 , \ldots , L .\tag{5.6}
$$

The final denoised signal is given by

$$
\widetilde { \pmb { y } } _ { j _ { 0 } : L + 1 } = \frac { 1 } { 2 } ( \widetilde { \pmb { y } } _ { j _ { 0 } : L + 1 , \mathrm { i n } } + \widetilde { \pmb { y } } _ { j _ { 0 } : L + 1 , \mathrm { o u t } } ) .
$$

That is, the average of the denoised results with respect to the in-degree and out-degree spline quasiinterpolants. The objective is to obtain $\widetilde { \pmb { y } } _ { j _ { 0 } : L + 1 } \approx \pmb { y } _ { T }$ from the noisy observation $_ { \mathbf { \nabla } \mathbf {  } \mathbf {  } \mathbf {  } \mathbf {  } \mathbf {  } \mathbf {  } \mathbf { \mu }   }$ . In this process,

the coarse quasi-interpolant captures the dominant smooth structure, while the thresholded details restore informative variations at finer resolutions.

## B. Denoising Performance Analysis

We evaluate the hierarchy-induced knot sequences on graph signal denoising. Let ${ \mathbf { } } _ { { \mathbf { } } _ { 3 \mathrm { { T } } } }$ denote the groundtruth signal and $\mathbf { \Omega } _ { \mathbf { \mathcal { Y } } _ { N } }$ its noisy observation, with noise ratios r ranging from 5% to 20%. Following the multi-level reconstruction defined above, we report the results from the one-level denoising $\bar { \mathbf { L } } _ { 2 , \iota } + \bar { d } _ { 2 , \iota }$ and the two-level denoising ${ \pmb { L } } _ { 1 , \iota } + \widetilde { d } _ { 1 , \iota } + \widetilde { d } _ { 2 , \iota }$ . Performance is measured by RMSE (root mean square error, the smaller the better) and SNR (signal to noise ratio, the larger the better) between ${ \bf Y } _ { T }$ and $\widetilde { \pmb { y } } _ { j _ { 0 } : L + 1 }$ for $L = 2$ and $j _ { 0 } = 1 , 2$

As shown in Table III, increasing the noise ratio consistently degrades the raw signal, whereas splinebased reconstruction generally reduces RMSE and improves SNR across all datasets. At a 20% noise ratio, the best reconstruction reduces the RMSE from 0.0211 to 0.0141 on Cora and from 0.0185 to 0.0093 on Squirrel, with corresponding SNR improvements from 14.1183 to 17.6025 dB and from 14.0223 to 19.9570 dB. The three-level reconstruction further improves the results on Cora, indicating that the additional detail component contains useful fine-scale information. On the other datasets, the two reconstruction depths yield similar or identical results, suggesting that the coarser approximation already captures most of the recoverable signal structure. Overall, these results confirm that the cluster-derived knot sequences provide effective and stable supports for structure-adaptive graph signal denoising.

## VI. CONCLUSIONS

This work connected hierarchical digraph clustering with spline-based graph signal processing. We proposed SpecHDC, a semi-supervised spectral hierarchical clustering framework based on a generalized Hermitian matrix. By recursively clustering and coarsening the digraph, SpecHDC preserves directional interactions and produces consistent nested partitions under both supervised and unsupervised settings. The resulting hierarchy is mapped to in-degree and out-degree intervals, whose endpoints form nested nonuniform knot sequences. These knots define multilevel spline quasi-interpolants, allowing noisy graph signals to be reconstructed from a coarse approximation and adaptively thresholded inter-level details. Experiments on synthetic and real-world digraphs demonstrate the effectiveness of SpecHDC under different structural and supervision settings. The denoising results further show that the hierarchy-induced knots provide structure-adaptive supports for signal recovery, yielding lower RMSE and higher SNR. Future work will focus on scalable implementations and extensions to more complex directed graph structures.

## REFERENCES

[1] F. Gama, A. G. Marques, G. Leus, and A. Ribeiro, “Convolutional neural network architectures for signals supported on graphs," IEEE Transactions on Signal Processing, vol. 67, no. 4, pp. 1034–1049, 2019.

[2] A. Sandryhaila and J. M. Moura, "Discrete signal processing on graphs," IEEE Transactions on Signal Processing, vol. 61, no. 7, pp. 1644–1656, 2013.

[3] A. Ortega, P. Frossard, J. Kovačević, J. M. Moura, and P. Vandergheynst, "Graph signal processing: Overview, challenges, and applications," Proceedings of the IEEE, vol. 106, no. 5, pp. 808–828, 2018.

[4] G. Leus, A. G. Marques, J. M. Moura, A. Ortega, and D. I. Shuman, "Graph signal processing: History, development, impact, and outlook," IEEE Signal Processing Magazine, vol. 40, no. 4, pp. 49–60, 2023.

[5] D. I. Shuman, "Localized spectral graph filter frames: A unifying framework, survey of design considerations, and numerical comparison," IEEE Signal Processing Magazine, vol. 37, no. 6, pp. 43–63, 2020.

[6] D. Wei and S. Yuan, "Vertex-frequency analysis on directed graphs," IEEE Transactions on Signal Processing, vol. 73, pp. 2255–2270, 2025.

[7] Y. Tanaka, Y. C. Eldar, A. Ortega, and G. Cheung, “Sampling signals on graphs: From theory to applications," IEEE Signal Processing Magazine, vol. 37, no. 6, pp. 14–30, 2020.

[8] Y. Tanaka and Y. C. Eldar, "Generalized sampling on graphs with subspace and smoothness priors," IEEE Transactions on Signal Processing, vol. 68, pp. 2272–2286, 2020.

[9] A. Jung and N. Tran, “Localized linear regression in networked data," IEEE Signal Processing Letters, vol. 26, no. 7, pp. 1090–1094, 2019.

[10] A. Jung, A. O. Hero III, A. C. Mara, S. Jahromi, A. Heimowitz, and Y. C. Eldar, “Semi-supervised learning in network-structured data via total variation minimization," IEEE Transactions on Signal Processing, vol. 67, no. 24, pp. 6256–6269, 2019.

[11] S. Rey, V. M. Tenorio, and A. G. Marqués, "Robust graph filter identification and graph denoising from signal observations," IEEE Transactions on Signal Processing, vol. 71, pp. 3651–3666, 2023.

[12] J. Shi and J. Malik, "Normalized cuts and image segmentation," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 22, no. 8, pp. 888–905, 2000.

[13] G. W. Flake, R. E. Tarjan, and K. Tsioutsiouliklis, "Graph clustering and minimum cut trees," Internet Mathematics, vol. 1, no. 4, pp. 385–408, 2004.

[14] A. Ng, M. Jordan, and Y. Weiss, "On spectral clustering: Analysis and an algorithm," Advances in Neural Information Processing Systems, vol. 14, 2001.

[15] U. Von Luxburg, “A tutorial on spectral clustering," Statistics and Computing, vol. 17, no. 4, pp. 395–416, 2007.

[16] V. Satuluri and S. Parthasarathy, "Symmetrizations for clustering directed graphs," in Proceedings of the 14th International Conference on Extending Database Technology, 2011, pp. 343–354.

[17] C. K. Chui, H. Mhaskar, and X. Zhuang, "Representation of functions on big data associated with directed graphs," Applied and Computational Harmonic Analysis, vol. 44, no. 1, pp. 165–188, 2018.

[18] K. Rohe, T. Qin, and B. Yu, "Co-clustering directed graphs to discover asymmetries and directional communities," Proceedings of the National Academy of Sciences, vol. 113, no. 45, pp. 12 679–12 684, 2016.

[19] F. Chung, "Laplacians and the cheeger inequality for directed graphs," Annals of Combinatorics, vol. 9, no. 1, pp. 1–19, 2005

[20] D. Zhou, J. Huang, and B. Schölkopf, "Learning from labeled and unlabeled data on a directed graph," in Proceedings of the International Conference on Machine Learning (ICML), 2005, pp. 1036–1043.

[21] M. Rosvall and C. T. Bergstrom, "Maps of random walks on complex networks reveal community structure," Proceedings of the National Academy of Sciences, vol. 105, no. 4, pp. 1118–1123, 2008.

[22] M. Cucuringu, H. Li, H. Sun, and L. Zanetti, "Hermitian matrices for clustering directed graphs: Insights and applications," in International Conference on Artificial Intelligence and Statistics. PMLR, 2020, pp. 983–992.

[23] K. Hayashi, S. G. Aksoy, and H. Park, “Skew-symmetric adjacency matrices for clustering directed graphs," in Proceedings of the IEEE International Conference on Big Data. IEEE, 2022, pp. 555–564.

[24] C. De Boor, A Practical Guide to Splines, ser. Applied Mathematical Sciences. New York: Springer-Verlag, 2001, vol. 27.

[25] C. De Boor and G. J. Fix, "Spline approximation by quasiinterpolants," Journal of Approximation Theory, vol. 8, no. 1, pp. 19–45, 1973.

[26] C. K. Chui and H. Diamond, “A general framework for local interpolation," Numerische Mathematik, vol. 58, no. 1, pp. 569–581, 1990.

[27] C. K. Chui, F. Filbir, and H. N. Mhaskar, "Representation of functions on big data: graphs and trees," Applied and Computational Harmonic Analysis, vol. 38, no. 3, pp. 489–509, 2015.

[28] C. K. Chui, J. De Villiers, and X. Zhuang, “Multirate systems with shortest spline-wavelet filters," Applied and Computational Harmonic Analysis, vol. 41, no. 1, pp. 266–296, 2016.

[29] H. Speleers, “Hierarchical spline spaces: Quasi-interpolants and local approximation estimates," Advances in Computational Mathematics, vol. 43, no. 2, pp. 235–255, 2017.

[30] C. Bracco, C. Giannelli, F. Mazzia, and A. Sestini, "Bivariate hierarchical Hermite spline quasi-interpolation," BIT Numerical Mathematics, vol. 56, no. 4, pp. 1165–1188, 2016.

[31] A. Kunoth, T. Lyche, G. Sangalli, S. Serra-Capizzano, T. Lyche, C. Manni, and H. Speleers, "Foundations of spline theory: B-splines, spline approximation, and hierarchical refinement," Splines and PDEs: From Approximation Theory to Numerical Linear Algebra: Cetraro, Italy 2017, pp. 1–76, 2018.

[32] J. Zhang, X. He, and J. Wang, "Directed community detection with network embedding," Journal of the American Statistical Association, vol. 117, no. 540, pp. 1809–1819, 2022.

[33] Y. He, G. Reinert, and M. Cucuringu, "Digrac: Digraph clustering based on flow imbalance," in Proceedings of Learning on Graphs Conference. PMLR, 2022.

[34] S. Laenen and H. Sun, “Higher-order spectral clustering of directed graphs," in Advances in Neural Information Processing Systems, 2020, pp. 941–951.

[35] W. Dahmen, R. De Vore, and K. Scherer, "Multidimensional spline approximation," SIAM Journal on Numerical Analysis, vol. 17, no. 3, pp. 380–402, 1980.

[36] M. G. Cox, "Practical spline approximation,"’ in Topics in Numerical Analysis: Proceedings of the SERC Summer School. Springer, 2006, pp. 79–112.

[37] M. S. Hasan, M. N. Alam, M. Fayz-Al-Asad, N. Muhammad, and C. Tunç, "B-spline curve theory: An overview and applications in real life," Nonlinear Engineering, vol. 13, no. 1, p. 20240054, 2024.

[38] H. N. Mhaskar, F. J. Narcowich, and J. D. Ward, "Quasi-interpolation in shift invariant spaces," Journal of mathematical analysis and applications, vol. 251, no. 1, pp. 356–363, 2000.

[39] I. Pesenson, "Variational splines and paley-wiener spaces on combinatorial graphs," Constructive Approximation, vol. 29, no. 1, pp. 1–21, 2009.

[40] A. K. Jain, "Data clustering: 50 years beyond k-means," Pattern recognition letters, vol. 31, no. 8, pp. 651–666, 2010.

[41] P. Sen, G. Namata, M. Bilgic, L. Getoor, B. Galligher, and T. Eliassi-Rad, "Collective classification in network data," AI Magazine, vol. 29, no. 3, pp. 93–93, 2008.

[42] B. Rozemberczki, C. Allen, and R. Sarkar, “Multi-scale attributed node embedding," Journal of Complex Networks, vol. 9, no. 2, p. cnab014, 2021.

[43] X. Zhang, Y. He, N. Brugnone, M. Perlmutter, and M. Hirn, "Magnet: A neural network for directed graphs," in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 27 003–27 015.

[44] H. Pei, B. Wei, K. C.-C. Chang, Y. Lei, and B. Yang, "Geom-gcn: Geometric graph convolutional networks," in International Conference on Learning Representations, 2020.

[45] L. A. Adamic and N. Glance, "The political blogosphere and the 2004 us election: divided they blog," in Proceedings of the International Workshop on Link Discovery, 2005, pp. 36–43.

[46] C. Huang, Y. Wang, Y. Jiang, M. Li, X. Huang, S. Wang, S. Pan, and C. Zhou, "Flow2GNN: Flexible two-way flow message passing for enhancing gnns beyond homophily," IEEE Transactions on Cybernetics, vol. 54, no. 11, pp. 6607–6618, 2024.

[47] A. J. Gates and Y.-Y. Ahn, "The impact of random models on clustering similarity," Journal of Machine Learning Research, vol. 18, no. 87, pp. 1–28, 2017.

[48] F. D. Malliaros and M. Vazirgiannis, “Clustering and community detection in directed networks: A survey," Physics Reports, vol. 533, no. 4, pp. 95–142, 2013.

[49] C. De Boor, A practical guide to splines. springer New York, 1978, vol. 27.

[50] I. P. Natanson, Constructive function theory. Ungar, 1964.

[51] G. Freud, Orthogonal polynomials. Elsevier, 2014.

[52] R. A. DeVore and G. G. Lorentz, Constructive approximation. Springer Science & Business Media, 1993, vol. 303.

## SUPPLEMENTARY MATERIALS

## A. Training Details

1) Code Realisation: The detailed experimental code is available at

https://github.com/kellysylvia77/Digraph.

2) Datasets: We evaluate the proposed method on seven real-world directed graphs with different scales and structural characteristics. As summarized in Table IV, the datasets contain between 183 and 5201 vertices and between 298 and 217073 directed edges. We report the numbers of vertices, edges, and classes, together with the the homophily ratio H. The five labeled datasets exhibit different degrees of heterophily, while Telegram and Blog contain no node labels; therefore, their class counts and homophily ratios are reported as unavailable.

TABLE IV  
STATISTICS OF REAL-WORLD DIRECTED GRAPHS.
<table><tr><td>Datasets</td><td>Cora</td><td>Squirrel</td><td>Cornell</td><td>Wisconsin</td><td>Texas</td><td>Telegram</td><td>Blog</td></tr><tr><td>#Nodes, |ν|</td><td>2708</td><td>5201</td><td>183</td><td>251</td><td>183</td><td>245</td><td>5201</td></tr><tr><td>#Edges, |ε|</td><td>5429</td><td>217073</td><td>298</td><td>515</td><td>325</td><td>8912</td><td>19024</td></tr><tr><td>#Classes, c</td><td>7</td><td>5</td><td>5</td><td>5</td><td>5</td><td></td><td></td></tr><tr><td>Hom. Ratio, H</td><td>0.3347</td><td>0.0854</td><td>0.1153</td><td>0.1325</td><td>0.0695</td><td>N/A</td><td>N/A</td></tr></table>

• Citation Network. Cora is a citation network in which vertices represent scientific publications and directed edges denote citation relations. Node features are derived from document contents, and labels indicate research categories [41].

• Wikipedia and Webpage Networks. Squirrel is a Wikipedia webpage network whose directed edges represent hyperlinks between pages [42]. Cornell, Wisconsin, and Texas are WebKB networks collected from university websites, where vertices correspond to webpages and directed edges represent hyperlinks [44]. Their labels describe webpage categories.

• Social and Information Networks. Telegram is a directed influence network between Telegram channels. It is used to evaluate community discovery without supervision [43], [33]. Blog is a political blog network in which directed edges represent hyperlinks between blogs [45]. Both datasets are treated as unlabeled networks in our experiments.

The labeled datasets have homophily ratios ranging from 0.0695 to 0.3347, indicating predominantly heterophilic connectivity. Together with the two unlabeled networks, these datasets provide diverse settings for evaluating both semi-supervised hierarchical clustering and unsupervised community discovery on directed graphs.

3) Competitors: We compare SpecHDC with five representative spectral clustering methods for directed graphs. These baselines cover several major strategies for handling directionality, including matrix symmetrization, singular-vector decomposition, and Hermitian/skew-symmetric spectral formulations.

• Bi-Sym [16] transforms the directed adjacency matrix into a symmetric similarity matrix through products involving A and $A ^ { \top }$ , so that vertices are compared according to shared predecessor and successor relationships before applying conventional spectral clustering.

• DD-Sym [16] is a degree-discounted symmetrization method that further reweights the predecessorsuccessor similarities to reduce the dominance of high-degree vertices in the resulting spectral representation.

• DI-SIM [18] constructs a regularized directed Laplacian and uses its left and right singular vectors to characterize the distinct sending and receiving roles of vertices, respectively.

• Herm [22] encodes edge orientation in a complex Hermitian matrix. Its real eigenvalues and orthonormal eigenvectors enable standard spectral embedding while preserving asymmetric connectivity information.

• Skew [23] represents directed connectivity through a real skew-symmetric formulation closely related to the Hermitian construction, preserving the relevant directed-cut information while reducing the need for complex-domain computation.

All baselines are evaluated using the hyperparameter settings recommended in their original implementations or publications. The same experimental protocol is applied to the synthetic DSBM graphs and all seven real-world directed networks to ensure a consistent comparison.

4) Metrics: We evaluate clustering performance using Modularity (M), Adjusted Rand Index $( \mathcal { A } )$ , and F-measure $( \mathcal { F } )$ . Higher values indicate better clustering quality. The mathematical formulations of these metrics are provided in Definitions 3-5.

Definition 3: [Modularity] Let $k _ { i } ^ { o u t }$ and $k _ { j } ^ { i n }$ denote the outdegree of node i and the indegree of node j in W, respectively. We assume that in a random directed graph with the same connectivity, the expected probability of an edge from i to $j$ is $k _ { i } ^ { o u t } k _ { j } ^ { i n } / m$ . Here, $m \ = \ k _ { i } ^ { i n } + k _ { i } ^ { o u t }$ is the total weight of the incoming/outgoing edges in the directed graph, which equals the sum of all entries in W. Then, the modularity metric is defined as:

$$
\mathcal { M } = \frac { 1 } { m } \sum _ { i , j } \left( W _ { i j } - \frac { k _ { i } ^ { o u t } k _ { j } ^ { i n } } { m } \right) \delta ( C _ { i } , C _ { j } ) ,\tag{6.1}
$$

where $\delta ( C _ { i } , C _ { j } )$ is 1 if the nodes i and $j$ both belong to the same cluster $C = C _ { i } = C _ { j }$ , and 0 otherwise. This metric quantifies the difference between the actual number of edges within clusters and the expected number in a random graph with the same degree distribution.

Definition 4: [Adjusted Rand Index (ARI)] Suppose we have two partitions for N individuals: $C _ { 1 }$ (with K classes) and $C _ { 2 }$ (with M classes). Each individual i has two labels: $c _ { i } ^ { 1 }$ from $C _ { 1 }$ and $c _ { i } ^ { 2 }$ from $C _ { 2 }$ . We define two indicator variables, $c _ { i j } ^ { 1 }$ and $c _ { i j } ^ { 2 }$ , to show if the pair $( i , j )$ is grouped together in $C _ { 1 }$ and $C _ { 2 } ,$ respectively:

$$
c _ { i j } ^ { 1 } = \left\{ \begin{array} { l l } { 1 } & { \mathrm { i f ~ } c _ { i } ^ { 1 } = c _ { j } ^ { 1 } = k , } \\ { 0 } & { \mathrm { o t h e r w i s e } , } \end{array} \right. \quad \mathrm { a n d } \quad c _ { i j } ^ { 2 } = \left\{ \begin{array} { l l } { 1 } & { \mathrm { i f ~ } c _ { i } ^ { 2 } = c _ { j } ^ { 2 } = \ell , } \\ { 0 } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{6.2}
$$

The values $c _ { i j } ^ { 1 }$ and $c _ { i j } ^ { 2 }$ are realizations of Bernoulli random variables. A pair is consistent in similarity if $c _ { i j } ^ { 1 } c _ { i j } ^ { 2 } = 1$ , and consistent in difference if $( 1 - c _ { i j } ^ { 1 } ) ( 1 - c _ { i j } ^ { 2 } ) = 1$ . The Adjusted Rand Index (ARI) is then defined over all pairs as:

$$
A { = } \frac { \sum _ { i j } \binom { c _ { i j } } { 2 } - \left[ \sum _ { i } \binom { c _ { i } } { 2 } \sum _ { j } \binom { c _ { j } } { 2 } \right] / \binom { n } { 2 } } { \frac { 1 } { 2 } \left[ \sum _ { i } \binom { c _ { i } } { 2 } + \sum _ { j } \binom { c _ { j } } { 2 } \right] - \left[ \sum _ { i } \binom { c _ { i } } { 2 } \sum _ { j } \binom { c _ { j } } { 2 } \right] / \binom { n } { 2 } } .\tag{6.3}
$$

Definition 5: [F-measure Score] Let $C _ { 1 } , C _ { 2 } , \cdots , C _ { K }$ be the clusters obtained from a clustering algorithm, and let $L _ { 1 } , L _ { 2 } , \cdots , L _ { M }$ be the ground-truth partition of nodes by class labels $( \mathrm { i } . \mathrm { e } . , L _ { j }$ contains all nodes in W with label j). The metric is defined as:

$$
F ( C _ { i } ) = 2 \operatorname* { m a x } _ { 1 \leq j \leq M } { \frac { | C _ { i } \cap L _ { j } | } { | C _ { i } | + | L _ { j } | } } ,\tag{6.4}
$$

the (micro-averaged) F-measure is then defined by

$$
\mathcal { F } = \frac { \sum _ { i } \left| C _ { i } \right| F ( C _ { i } ) } { \sum _ { i } \left| C _ { i } \right| } .\tag{6.5}
$$

5) Directed Stochastic Block Model: The directed stochastic block model (DSBM) extends the classical stochastic block model (SBM) by incorporating directional interactions between clusters. Besides the within-cluster and between-cluster connection probabilities $p$ and $q ,$ the DSBM introduces an orientation matrix $F \in [ 0 , 1 ] ^ { k \times k }$ to characterize the preferred direction of edges between different clusters. Hence, $F$ can be interpreted as the weighted adjacency matrix of a directed meta-graph whose vertices represent clusters.

Let $\mathcal { C } = \{ C _ { 1 } , \ldots , C _ { k } \}$ be a partition of $N = k \cdot n$ vertices into k clusters, each containing n vertices. For a pair of vertices $u \in C _ { i }$ and $v \in C _ { j }$ , an edge is generated with probability

$$
\operatorname* { P r } ( \{ u , v \} \in E ) = { \left\{ \begin{array} { l l } { p , } & { i = j , } \\ { q , } & { i \neq j . } \end{array} \right. }\tag{6.6}
$$

Once an edge is generated, its orientation is determined by $F$ Specifically,

$$
\operatorname* { P r } ( u \to v \mid \{ u , v \} \in E ) = F _ { i j } , \qquad \operatorname* { P r } ( v \to u \mid \{ u , v \} \in E ) = F _ { j i } ,\tag{6.7}
$$

where

$$
F _ { i j } + F _ { j i } = 1 , \qquad F _ { i i } = \frac { 1 } { 2 } .\tag{6.8}
$$

Thus, $F _ { i j } > 1 / 2$ indicates a preferred direction from cluster $C _ { i }$ to cluster $C _ { j }$ , whereas $F _ { i j } ~ = ~ 1 / 2$ corresponds to no directional preference. In our synthetic settings, the off-diagonal entries of $F$ are parameterized by $\eta \in [ 0 , \frac { 1 } { 2 } ]$ , with the preferred and reverse directions assigned probabilities $1 - \eta$ and $\eta ,$ respectively. Hence, a smaller $\eta$ indicates stronger directional preference, while $\begin{array} { r } { \eta = \frac { 1 } { 2 } } \end{array}$ removes the directional bias. The resulting random digraph is denoted by $G ( k , n , p , q , F )$

Example. Let $k = 4 , p = q$ and

$$
F = \left( \begin{array} { l l l l } { { { \frac { 1 } { 2 } } } } & { { { \frac { 2 } { 3 } } } } & { { { \frac { 2 } { 3 } } } } & { { { \frac { 1 } { 3 } } } } \\ { { { \frac { 1 } { 3 } } } } & { { { \frac { 1 } { 2 } } } } & { { { \frac { 2 } { 3 } } } } & { { { \frac { 2 } { 3 } } } } \\ { { { \frac { 1 } { 3 } } } } & { { { \frac { 1 } { 3 } } } } & { { { \frac { 1 } { 2 } } } } & { { { \frac { 2 } { 3 } } } } \\ { { { \frac { 2 } { 3 } } } } & { { { \frac { 1 } { 3 } } } } & { { { \frac { 1 } { 3 } } } } & { { { \frac { 1 } { 2 } } } } \end{array} \right) .\tag{6.9}
$$

![](images/86e074786e6643d312519e64f43ab38708043004c7eaec329af806af2b3cde3d.jpg)  
Fig. 4. Illustration of the asymmetric orientation pattern.

In this case, $G$ consists of four equal-sized clusters $C _ { 1 } , C _ { 2 } , C _ { 3 }$ , and $C _ { 4 } .$ , and any pair of vertices is connected with the same probability $p .$ Hence, the cluster structure cannot be identified from edge density alone. Within each cluster, edge directions are chosen uniformly at random since $F _ { i i } = 1 / 2$ . In contrast, the directions of edges between different clusters are determined non-uniformly by $F .$ For every pair $( C _ { i } , C _ { j } )$ with $F _ { i j } = 2 / 3$ , approximately two-thirds of the edges are expected to be directed from $C _ { i }$ to $C _ { j } .$ while the remaining one-third are directed in the reverse direction. Specifically, the preferred inter-cluster directions are $C _ { 1 }  C _ { 2 } , C _ { 1 }  C _ { 3 } , C _ { 2 }  C _ { 3 } , C _ { 2 }  C _ { 4 } , C _ { 3 }  C _ { 4 }$ , and $C _ { 4 }  C _ { 1 }$ . Therefore, although $p = q$ eliminates density-based separation, the asymmetric directional pattern encoded by F still provides informative cluster structure. This example illustrates how the DSBM can generate directed communities whose distinctions arise primarily from edge orientation rather than connectivity density.

6) k-Means Clustering: Given N observations $z _ { 1 } , z _ { 2 } , \ldots , z _ { N } \in \mathbb { R } ^ { n }$ of n-dimensional vectors associated with vertices $v _ { 1 } , \ldots , v _ { N }$ of a vertex set V, k-means clustering [40] aims to partition the N observations into a k-way clustering ${ \mathcal { C } } = \{ C _ { 1 } , C _ { 2 } , \ldots , C _ { k } \}$ of V, equivalently, to obtain a label signal $y : V \to \{ 1 , \dots , k \}$ so as to minimize the within-cluster sum of squares. In the semi-supervised setting, labels are known only for vertices indexed by ${ \mathcal { T } } _ { \mathrm { k n w } } \subseteq \{ 1 , \dots , N \}$ so that $Y _ { \mathrm { k n w } } ( i ) \in \{ 1 , \dots , k \}$ is given in advance for $i \in \mathcal { T } _ { \mathrm { k n w } } .$ If $\mathcal { T } _ { \mathrm { k n w } } = \emptyset$ , then it corresponds to the unsupervised setting. The assignments of labels for the remaining unknown vertices are inferred subject to the known labels. Formally, the objective is to find:

$$
\operatorname* { m i n } _ { \mathcal { C } = \{ C _ { j } = y ^ { - 1 } ( j ) \} _ { j = 1 } ^ { k } } \sum _ { j = 1 } ^ { k } \sum _ { v _ { i } \in C _ { j } } \left. z _ { i } - c _ { j } \right. _ { 2 } ^ { 2 } , \mathrm { ~ s . t . ~ } y | _ { \mathcal { T } _ { \mathrm { k n w } } } = Y _ { \mathrm { k n w } } ,\tag{6.10}
$$

where each

$$
c _ { j } = { \frac { 1 } { | C _ { j } | } } \sum _ { v _ { i } \in C _ { j } } z _ { i }
$$

is the centroid of the vertices in the class $C _ { j }$ for $j = 1 , \dots , k$ , the norm $\| \cdot \| _ { 2 }$ is the Euclidean distance, but can be replaced by other general metric.

The most common algorithm uses an iterative refinement technique. Given an initial set of k centroids, the algorithm proceeds by alternating between two steps:

1) Assignment step: Assign each observation to the cluster with the nearest mean (centroid). Mathematically, this means partitioning the observations according to the Voronoi diagram generated by the means.

2) Update step: Recalculate means (centroids) for observations assigned to each cluster. This is also called refitting.

After each iteration, the within-cluster sum of squares decreases monotonically, yielding a nonnegative, monotonically decreasing sequence. This guarantees that the k-means always converges, but not necessarily to the global optimum. Algorithm 3 presents the details of the implementations of k-means in our setting.

Algorithm 3 k-means Clustering Algorithm.   
a) Input: $\overline { { V \ = \ \{ v _ { 1 } , . . . , v _ { n } \} } }$ with each vertex v associated with a d-dimensional vector $z _ { v } \in \mathbb { R } ^ { n }$   
respectively; the known pair $( \mathcal { T } _ { \mathrm { k n w } } , Y _ { \mathrm { k n w } } )$ of labels; number k of clusters; the maximum number   
of iterations   
b) Output: A k-way clustering $\mathcal { C } = \{ C _ { 1 } , \ldots , C _ { k } \}$ of V   
c) Main Steps:   
1: Initialization: Randomly choose k vertices $u _ { 1 } , \ldots , u _ { k }$ from V as centers subject to $u _ { j } \in Y _ { \mathrm { k n w } } ^ { - 1 } ( j )$   
respectively. The initial centroids $c _ { 1 } , \ldots , c _ { k }$ are then the vectors $z _ { u 1 } , \ldots , z _ { u _ { k } }$ , respectively   
2: while true do   
3: Construct cluster $C _ { j }$ for $j = 1 , \dots , k .$ That is, $v \in V$ belongs to $C _ { j }$ if either $v \in Y _ { \mathrm { k n w } } ^ { - 1 } ( j )$ or   
$j = \mathrm { a r g m i n } _ { 1 \le j ^ { \prime } \le k } \| z _ { v } - c _ { j ^ { \prime } } \| _ { 2 }$   
4: Update the centers: for each $C _ { j }$ , find a new center $u \in C _ { j }$ such that $\begin{array} { r } { \sum _ { v \in C _ { j } } \| z _ { u } - z _ { v } \| _ { 2 } } \end{array}$ is minimal   
and the centroid is updated to $z _ { u } ,$ respectively   
5: Break if all centers remain the same or the maximum number of iterations is reached.   
6: end while

## B. Supplementary Experiments

1) Results for the DSBM: This section provides a further analysis of model performance using an expanded variable set in DSBM including N, k and the pattern of F matrix, as well as under more conditions than those covered in the main text.

![](images/27d458a223deb887b7dee07601788a0b8e6327a663b3230955adf1d6f0c0d421.jpg)  
Fig. 5. The models’ performance is compared across varying numbers of nodes under three scenarios: p = q (left), $p \gg q$ (middle), and q  p (right).

![](images/5c90f9a22059d5a702d04643e37d5d58052fca92dfac5ada60a7d28c740cd05b.jpg)  
Fig. 6. The models’ performance is compared across varying numbers of clusters under three scenarios: p = q (left), $p \gg q$ (middle), and $q \gg p \ \mathrm { ( r i g h t ) }$

Parameter N-Graph Scale. Figure 5 examines the effect of graph size on clustering performance under three structural settings: $p = q , p \gg q .$ and $q \ \gg \ p .$ Overall, increasing N provides more structural evidence and generally improves the ARI of all methods. When $p = q .$ , the cluster structure is relatively ambiguous, and most methods remain ineffective on small graphs. Their performance improves only when the graph becomes sufficiently large, with ARI values rising markedly at $N = 5 0 0 0$ and $N = 1 0 0 0 0$ Clearer structural separation leads to faster improvement. Under $q \gg p ,$ all methods achieve high ARI at $N = 5 0 0$ and nearly perfect clustering once $N \geq 1 0 0 0$ . Under $p \gg q ,$ however, the methods respond differently: DD-Sym performs well from N = 500, while Bi-Sym, DI-SIM, and SpecHDC reach nearperfect performance at approximately $N \ = \ 1 0 0 0 .$ In contrast, Herm and Skew remain less effective even on larger graphs in this setting. These results show that graph scale and structural pattern jointly determine clustering difficulty. Larger graphs generally improve recoverability, while strong homophilic or heterophilic separation reduces the amount of data required. SpecHDC exhibits stable scalability and achieves nearly perfect clustering on sufficiently large graphs with clearly distinguishable structures.

Parameter k-Cluster Size. Figure 6 examines the effect of the number of clusters under three structural settings: $p = q , p \gg q .$ , and $q \gg p .$ When $p = q$ , the community structure is weak, and the performance of all methods decreases rapidly as k increases. Most ARI values approach zero for large k, indicating that finer partitions are difficult to recover without clear structural separation. When $p \gg q .$ SpecHDC, Bi-Sym, and DD-Sym remain nearly perfect across the tested values of k, while DI-SIM exhibits only a small fluctuation. In contrast, Herm and Skew perform poorly in this setting. Under $q \gg p ,$ almost all methods maintain high ARI, although slight degradation appears for some methods when k becomes large. Overall, increasing the number of clusters amplifies clustering difficulty mainly when the underlying block structure is ambiguous. Clear homophilic or heterophilic patterns substantially reduce this sensitivity, and SpecHDC shows strong robustness across different clustering granularities.

Pattern Type for F Matrix. The parameter $F \in [ 0 , 1 ] ^ { k \times k }$ dictates the edge direction probability matrix between clusters, satisfying $F _ { i , j } + F _ { j , i } = 1$ . Figure 7 investigates the influence of this meta-graph structure on clustering robustness against the directional noise parameter $\eta .$ We explicitly evaluate two distinct structural definitions:

![](images/ed05bfd87a0fa40a7424c73afdc238f70b545c6692aa6164bd77780a4e53003f.jpg)  
(b) Complete Meta-Graph  
Fig. 7. Performance comparison of the models across varying levels of η under two distinct F matrix patterns: (a) Circular Pattern and (b) Complete Meta-Graph.

(1) Cyclic Pattern: Directional biases are strictly confined to adjacent clusters forming a directed cycle $( \mathbf { e . g . , } F _ { i , i + 1 } = 1 - \eta$ and $F _ { i + 1 , i } = \eta )$ , while all other non-adjacent cluster pairs maintain perfectly symmetric, unbiased connections $( F _ { i , j } = 1 / 2 )$

(2) Complete Pattern: Every distinct pair of clusters possesses a strict directional bias $( F _ { i , j } \in \{ \eta , 1 - \eta \}$ for all $i \neq j )$ , forming a densely connected and highly complex asymmetric meta-graph.

To provide a comprehensive evaluation, we expand the analysis across three distinct structural scenarios: $p = q$ (left), $q > p$ (middle), and $p > q \mathrm { ~ ( r i g h t ) }$ . The results reveal a stark contrast between the two topologies. Under the Cyclic Pattern (Figure 7(a)), our proposed methods exhibit extraordinary tolerance to directional noise whenever explicit structural signals exist $( p \neq q )$ . Specifically, in the heterophilic setting $( q \gg p )$ , SpecHDC degrades at a significantly slower rate than baselines. In the homophilic setting $( p \gg q )$ , our methods establish and maintain a commanding lead from the outset. Conversely, under the Complete Pattern (Figure 7(b)), the densely connected and conflicting directional biases make the extraction of stable signals exceptionally difficult. This leads to severe performance oscillations across all algorithms, regardless of the relationship between $p$ and $q .$ Ultimately, these findings highlight that our methods are exceptionally adept at capturing sparsely structured meta-graph signals (such as cyclic patterns) and offer outstanding resilience against structural noise.

Parameter $\alpha$ and β-Spectral Balance. Figure 8 examines the sensitivity of SpecHDC to $\alpha$ and $\beta ,$ which control the symmetric connectivity term and the asymmetric directional term in the generalized Hermitian matrix, respectively. The results show that their appropriate balance depends strongly on the underlying graph structure. When $p = q .$ , the graph provides little separation through edge density. $\mathbf { A } \mathbf { t } ~ \eta = 0$ , useful clustering is obtained only within a limited parameter region, and the overall ARI decreases rapidly as η increases. This indicates that, without clear homophilic or heterophilic structure, the model relies heavily on informative edge directions and is therefore sensitive to directional perturbations. When $q \gg p ,$ a stronger contribution from the asymmetric component is generally beneficial at $\eta = 0$ . For $\eta = 0 . 1 2$ and 0.24, however, the ARI remains close to 1 over almost the entire parameter space, suggesting that the pronounced inter-cluster connectivity makes the clustering result largely insensitive to the precise choice of α and $\beta .$ In contrast, when $p \gg q ,$ high ARI is obtained over a broad region where the symmetric connectivity component is sufficiently emphasized. Performance drops sharply when the asymmetric term dominates excessively, showing that within-cluster connectivity is the primary source of information in strongly homophilic graphs. Overall, no single parameter pair is optimal for all settings: $\beta$ is more important for direction-dominated structures, whereas α should receive greater weight when homophilic connectivity provides the main clustering signal.

![](images/cd5f12c9c9f45bec214d66afbab7fc59f4bffd5bdce1a92c337354410268674f.jpg)  
(a) η = 0, p = 0.0045, q = 0.0045

![](images/3b546583df28c2d7ca87d70bd9c0bb6cbb68e5e2d7dc95bf37251170cc26d5d7.jpg)  
(b) η = 0.12, p = 0.0045, q = 0.0045

![](images/852e54ae5c04465b7cda0f59a190bf8cab1874b55bd2e5f9aff6172dda0efba9.jpg)  
(c) η = 0.24, p = 0.0045, q = 0.0045

![](images/bf6950072f60e2f5b56fef37b2cbfd7de6c3214d0baf7fc2c1236198c46b89d1.jpg)  
(d) $\eta = 0 , p = 0 . 0 0 4 5 , q = 0 . 0 5$

![](images/e916eb10928536a5d9a7129120622fade4971b3f066d2f08700f4a5dab412f73.jpg)  
(e) η = 0.12, p = 0.0045, q = 0.05

![](images/08ab9a09bf23c1952f7f910232707da0c6d9d88c9d86812a40930e482a4aceda.jpg)  
(f) η = 0.24, p = 0.0045, q = 0.05

![](images/658d9018fdd249e64268fe7b146837fddc2c2d1ca9c1b836ea32d479162ba261.jpg)  
(g) $\eta = 0 , p = 0 . 0 5 , q = 0 . 0 0 4 5$

![](images/560374c6ee1645fa5bf3a29a9e820e40ffac2d94fd9ef404bd33e9591104fedb.jpg)  
(h) η = 0.12, p = 0.05, q = 0.0045

![](images/ac90a1832b8afb3f1e5a936871de2ea184040746e96b17286c707a355a8604b1.jpg)  
(i) η = 0.24, p = 0.05, q = 0.0045  
Fig. 8. Investigating the effect of α and β: Model parameter sensitivity on different directed graph structures.

2) Results for Real-World Data: Tables V–VIII report the clustering results on Cora, Squirrel, Cornell, and Texas under different supervision ratios and hierarchical resolutions. Level 1 contains the same number of clusters as the ground-truth classes, whereas Level 2 provides a finer partition. We separately analyze Modularity (M) and F-measure (F), since they measure structural cohesion and class consistency, respectively.

Structural Quality. The modularity results vary noticeably across datasets and hierarchical levels. On partition captures structurally cohesive subcommunities when sufficient label constraints are available. Its Level 1 modularity is less competitive, suggesting that the coarse class partition does not always coincide with the densest graph structure. On Squirrel, the advantage of SpecHDC is mainly observed at Level 1. It achieves the best modularity at the 30%, 40%, 70%, 80%, and 90% settings, and remains competitive at several other ratios. In particular, its Level 1 modularity reaches 0.2453 at 90%, whereas the Level 2 scores of all methods remain relatively low. This suggests that the coarse partition provides a more meaningful structural description of this strongly heterophilic network than the finer partition. The modularity results on Cornell and Texas are more irregular because these networks are small and their structural communities may not align closely with the class labels. On Cornell, SpecHDC obtains the best Level 1 modularity at 10%, 30%, and 80%, while remaining competitive at several other settings. On Texas, it achieves the best result at Level 2 with 10% supervision and at Level 1 with 80% supervision. Although SpecHDC does not dominate every setting on these two datasets, the results show that it can identify structurally cohesive partitions at selected resolutions and supervision levels.

TABLE V  
CLUSTERING ALGORITHMS ON THE CORA DATASET: A COMPARATIVE ANALYSIS OF MODULARITY (M), AND F-MEASURE (F). THE BEST-PERFORMING MODEL IS HIGHLIGHTED IN BOLD, AND THE SECOND-BEST IS MARKED WITH AN UNDERLINE.
<table><tr><td>M</td><td>Trains (%)</td><td>0 (USL)</td><td>10</td><td>20</td><td>30</td><td>40</td><td>50</td><td>60</td><td>70</td><td>80</td><td>90</td></tr><tr><td rowspan="2">Bi-Sym</td><td>Level 2 (70)</td><td>0.4392</td><td>0.3759</td><td>0.3760</td><td>0.4318</td><td>0.4313</td><td>0.4651</td><td>0.5622</td><td>0.5964</td><td>0.6482</td><td>0.7083</td></tr><tr><td>Level 1 (7)</td><td>0.2624</td><td>0.2546</td><td>0.2233</td><td>0.2625</td><td>0.1476</td><td>0.2048</td><td>0.2752</td><td>0.3033</td><td>0.0907</td><td>0.0309</td></tr><tr><td rowspan="2">DD-Sym</td><td>Level 2 (70)</td><td>0.4983</td><td>0.4089</td><td>0.3501</td><td>0.4219</td><td>0.4076</td><td>0.4667</td><td>0.5405</td><td>0.5944</td><td>0.6365</td><td>0.7078</td></tr><tr><td>Level 1 (7)</td><td>0.4789</td><td>0.3494</td><td>0.3652</td><td>0.3888</td><td>0.4120</td><td>0.3817</td><td>0.2040</td><td>0.4070</td><td>0.2990</td><td>0.1125</td></tr><tr><td rowspan="2">DI-SIM</td><td>Level 2 (70)</td><td>0.6091</td><td>0.4782</td><td>0.4076</td><td>0.4730</td><td>0.4436</td><td>0.4679</td><td>0.5671</td><td>0.6081</td><td>0.6464</td><td>0.7089</td></tr><tr><td>Level 1 (7)</td><td>0.4456</td><td>0.4293</td><td>0.3047</td><td>0.2351</td><td>0.2629</td><td>0.2622</td><td>0.2127</td><td>0.2855</td><td>0.2282</td><td>0.1297</td></tr><tr><td rowspan="2">Herm</td><td>Level 2 (70)</td><td>0.4091</td><td>0.3492</td><td>0.3591</td><td>0.4286</td><td>0.4020</td><td>0.4671</td><td>0.5674</td><td>0.6101</td><td>0.6316</td><td>0.7116</td></tr><tr><td>Level 1 (7)</td><td>0.1376</td><td>0.1400</td><td>0.1206</td><td>0.1575</td><td>0.1017</td><td>0.0698</td><td>0.1162</td><td>0.1109</td><td>0.0600</td><td>0.0680</td></tr><tr><td rowspan="2">Skew</td><td>Level 2 (70)</td><td>0.4091</td><td>0.3492</td><td>0.3591</td><td>0.4286</td><td>0.4020</td><td>0.4671</td><td>0.5674</td><td>0.6101</td><td>0.6316</td><td>0.7116</td></tr><tr><td>Level 1 (7)</td><td>0.1876</td><td>0.1485</td><td>0.1383</td><td>0.1810</td><td>0.0831</td><td>0.0685</td><td>0.0561</td><td>0.0814</td><td>0.0270</td><td>0.0646</td></tr><tr><td rowspan="2">SpecHDC</td><td>Level 2 (70)</td><td>0.4565</td><td>0.3765</td><td>0.3756</td><td>0.4429</td><td>0.4370</td><td>0.4705</td><td>0.5716</td><td>0.6173</td><td>0.6492</td><td>0.7157</td></tr><tr><td>Level 1 (7)</td><td>0.2509</td><td>0.2903</td><td>0.3888</td><td>0.3576</td><td>0.2601</td><td>0.3819</td><td>0.3035</td><td>0.2071</td><td>0.0800</td><td>0.0544</td></tr><tr><td>F</td><td>Trains (%)</td><td>0 (USL)</td><td>10</td><td>20</td><td>30</td><td>40</td><td>50</td><td>60</td><td>70</td><td>80</td><td>90</td></tr><tr><td rowspan="2">Bi-Sym</td><td>Level 2 (70)</td><td>0.1211</td><td>0.1536</td><td>0.2474</td><td>0.3110</td><td>0.3929</td><td>0.4174</td><td>0.5801</td><td>0.5845</td><td>0.7847</td><td>0.8560</td></tr><tr><td>Level 1 (7)</td><td>0.1556</td><td>0.2360</td><td>0.2885</td><td>0.3418</td><td>0.4360</td><td>0.5559</td><td>0.6190</td><td>0.7354</td><td>0.8266</td><td>0.9079</td></tr><tr><td rowspan="2">DD-Sym</td><td>Level 2 (70)</td><td>0.0915</td><td>0.0455</td><td>0.0834</td><td>0.1664</td><td>0.2456</td><td>0.4153</td><td>0.4589</td><td>0.5883</td><td>0.7739</td><td>0.8562</td></tr><tr><td>Level 1 (7)</td><td>0.4154</td><td>0.2590</td><td>0.3098</td><td>0.3851</td><td>0.4176</td><td>0.4993</td><td>0.6563</td><td>0.7614</td><td>0.8188</td><td>0.9152</td></tr><tr><td rowspan="2">DI-SIM</td><td>Level 2 (70)</td><td>0.2573</td><td>0.0317</td><td>0.1989</td><td>0.2470</td><td>0.3834</td><td>0.3438</td><td>0.5714</td><td>0.6687</td><td>0.7835</td><td>0.8790</td></tr><tr><td>Level 1 (7)</td><td>0.0959</td><td>0.1212</td><td>0.3144</td><td>0.3917</td><td>0.4736</td><td>0.4858</td><td>0.6320</td><td>0.7100</td><td>0.8014</td><td>0.9071</td></tr><tr><td rowspan="2">Herm</td><td>Level 2 (70)</td><td>0.1838</td><td>0.0254</td><td>0.2160</td><td>0.3293</td><td>0.2469</td><td>0.4139</td><td>0.5601</td><td>0.6830</td><td>0.7170</td><td>0.8822</td></tr><tr><td>Level 1 (7)</td><td>0.1919</td><td>0.1904</td><td>0.2601</td><td>0.3753</td><td>0.4623</td><td>0.5239</td><td>0.6287</td><td>0.7239</td><td>0.8151</td><td>0.9077</td></tr><tr><td rowspan="2">Skew</td><td>Level 2 (70)</td><td>0.1838</td><td>0.0254</td><td>0.2160</td><td>0.3293</td><td>0.2469</td><td>0.4139</td><td>0.5601</td><td>0.6830</td><td>0.7170</td><td>0.8822</td></tr><tr><td>Level 1 (7)</td><td>0.1972</td><td>0.1723</td><td>0.3473</td><td>0.3846</td><td>0.4681</td><td>0.5291</td><td>0.5944</td><td>0.7080</td><td>0.8590</td><td>0.9019</td></tr><tr><td rowspan="2">SpecHDC</td><td>Level 2 (70)</td><td>0.1848</td><td>0.0245</td><td>0.0837</td><td>0.3194</td><td>0.2456</td><td>0.4176</td><td>0.5866</td><td>0.6853</td><td>0.7876</td><td>0.8874</td></tr><tr><td>Level 1 (7)</td><td>0.1592</td><td>0.3197</td><td>0.4129</td><td>0.3288</td><td>0.5420</td><td>0.5200</td><td>0.6466</td><td>0.7096</td><td>0.8114</td><td>0.9112</td></tr></table>

TABLE VI

CLUSTERING ALGORITHMS ON THE SQUIRREL DATASET: A COMPARATIVE ANALYSIS OF MODULARITY (M), AND F-MEASURE (F). THE BEST-PERFORMING MODEL IS HIGHLIGHTED IN BOLD, AND THE SECOND-BEST IS MARKED WITH AN UNDERLINE.
<table><tr><td>M</td><td>Trains (%)</td><td>0 (USL)</td><td>10</td><td>20</td><td>30</td><td>40</td><td>50</td><td>60</td><td>70</td><td>80</td><td>90</td></tr><tr><td rowspan="2">Bi-Sym</td><td>Level 2 (50)</td><td>0.1110</td><td>0.1043</td><td>0.0942</td><td>0.0934</td><td>0.0734</td><td>0.0730</td><td>0.0457</td><td>0.0426</td><td>0.0474</td><td>0.0518</td></tr><tr><td>Level 1 (5)</td><td>0.1365</td><td>0.1379</td><td>0.1328</td><td>0.2097</td><td>0.1385</td><td>0.2942</td><td>0.1899</td><td>0.1746</td><td>0.1866</td><td>0.2097</td></tr><tr><td rowspan="2">DD-Sym</td><td>Level 2 (50)</td><td>0.2289</td><td>0.2137</td><td>0.1929</td><td>0.2062</td><td>0.1400</td><td>0.1280</td><td>0.1172</td><td>0.0756</td><td>0.0606</td><td>0.0792</td></tr><tr><td>Level 1 (5)</td><td>0.4736</td><td>0.2888</td><td>0.2644</td><td>0.1445</td><td>0.2056</td><td>0.1856</td><td>0.1916</td><td>0.1852</td><td>0.1686</td><td>0.0962</td></tr><tr><td rowspan="2">DI-SIM</td><td>Level 2 (50)</td><td>0.1743</td><td>0.1643</td><td>0.1409</td><td>0.1237</td><td>0.2747</td><td>0.0844</td><td>0.0753</td><td>0.0672</td><td>0.0511</td><td>0.0587</td></tr><tr><td>Level 1 (5)</td><td>0.2091</td><td>0.2107</td><td>0.1437</td><td>0.1579</td><td>0.1681</td><td>0.1937</td><td>0.1546</td><td>0.1504</td><td>0.1160</td><td>0.0653</td></tr><tr><td rowspan="2">Herm</td><td>Level 2 (50)</td><td>0.1103</td><td>0.0843</td><td>0.0955</td><td>0.0938</td><td>0.0792</td><td>0.0571</td><td>0.0539</td><td>0.0416</td><td>0.0400</td><td>0.0554</td></tr><tr><td>Level 1 (5)</td><td>0.1756</td><td>0.1514</td><td>0.1778</td><td>0.1650</td><td>0.1404</td><td>0.1305</td><td>0.1247</td><td>0.1055</td><td>0.1607</td><td>0.1530</td></tr><tr><td rowspan="2">Skew</td><td>Level 2 (50)</td><td>0.1103</td><td>0.0843</td><td>0.0955</td><td>0.0938</td><td>0.0792</td><td>0.0571</td><td>0.0539</td><td>0.0416</td><td>0.0400</td><td>0.0554</td></tr><tr><td>Level 1 (5)</td><td>0.1604</td><td>0.1344</td><td>0.1907</td><td>0.1896</td><td>0.2049</td><td>0.0886</td><td>0.1031</td><td>0.0905</td><td>0.1826</td><td>0.1342</td></tr><tr><td rowspan="2">SpecHDC</td><td>Level 2 (50)</td><td>0.1201</td><td>0.1155</td><td>0.0975</td><td>0.0922</td><td>0.1260</td><td>0.0629</td><td>0.0487</td><td>0.0450</td><td>0.0507</td><td>0.0572</td></tr><tr><td>Level 1 (5)</td><td>0.1946</td><td>0.1856</td><td>0.2210</td><td>0.2303</td><td>0.2758</td><td>0.2610</td><td>0.1487</td><td>0.2074</td><td>0.2087</td><td>0.2453</td></tr><tr><td>F</td><td>Trains (%)</td><td>0 (USL)</td><td>10</td><td>20</td><td>30</td><td>40</td><td>50</td><td>60</td><td>70</td><td>80</td><td>90</td></tr><tr><td rowspan="2">Bi-Sym</td><td>Level 2 (50)</td><td>0.1412</td><td>0.1650</td><td>0.2223</td><td>0.3045</td><td>0.2721</td><td>0.5132</td><td>0.4807</td><td>0.6036</td><td>0.7883</td><td>0.8572</td></tr><tr><td>Level 1 (5)</td><td>0.2284</td><td>0.2684</td><td>0.3795</td><td>0.4330</td><td>0.5250</td><td>0.5860</td><td>0.6754</td><td>0.7455</td><td>0.8337</td><td>0.9168</td></tr><tr><td rowspan="2">DD-Sym</td><td>Level 2 (50)</td><td>0.0028</td><td>0.0782</td><td>0.0963</td><td>0.1939</td><td>0.2572</td><td>0.3957</td><td>0.5504</td><td>0.6601</td><td>0.7330</td><td>0.8647</td></tr><tr><td>Level 1 (5)</td><td>0.2636</td><td>0.2829</td><td>0.3621</td><td>0.4204</td><td>0.4629</td><td>0.5794</td><td>0.6637</td><td>0.7494</td><td>0.8369</td><td>0.9208</td></tr><tr><td rowspan="2">DI-SIM</td><td>Level 2 (50)</td><td>0.0023</td><td>0.2225</td><td>0.0857</td><td>0.1602</td><td>0.3963</td><td>0.3535</td><td>0.5935</td><td>0.5874</td><td>0.7202</td><td>0.8552</td></tr><tr><td>Level 1 (5)</td><td>0.2664</td><td>0.3221</td><td>0.3627</td><td>0.4492</td><td>0.5065</td><td>0.5697</td><td>0.6642</td><td>0.7447</td><td>0.8303</td><td>0.9196</td></tr><tr><td rowspan="2">Herm</td><td>Level 2 (50)</td><td>0.1002</td><td>0.1841</td><td>0.0841</td><td>0.1488</td><td>0.4032</td><td>0.4830</td><td>0.5842</td><td>0.5970</td><td>0.7214</td><td>0.8966</td></tr><tr><td>Level 1 (5)</td><td>0.3031</td><td>0.3031</td><td>0.3696</td><td>0.3925</td><td>0.5291</td><td>0.5999</td><td>0.6632</td><td>0.7479</td><td>0.8314</td><td>0.9176</td></tr><tr><td rowspan="2">Skew</td><td>Level 2 (50)</td><td>0.1002</td><td>0.1841</td><td>0.0841</td><td>0.1488</td><td>0.4032</td><td>0.4830</td><td>0.5842</td><td>0.5970</td><td>0.7214</td><td>0.8966</td></tr><tr><td>Level 1 (5)</td><td>0.2526</td><td>0.3020</td><td>0.4047</td><td>0.4347</td><td>0.5340</td><td>0.5842</td><td>0.6794</td><td>0.7437</td><td>0.8301</td><td>0.9190</td></tr><tr><td rowspan="2">SpecHDC</td><td>Level 2 (50)</td><td>0.1250</td><td>0.2048</td><td>0.1107</td><td>0.3206</td><td>0.2559</td><td>0.4872</td><td>0.4605</td><td>0.5896</td><td>0.7933</td><td>0.8572</td></tr><tr><td>Level 1 (5)</td><td>0.2946</td><td>0.2653</td><td>0.3667</td><td>0.4471</td><td>0.4982</td><td>0.5887</td><td>0.6591</td><td>0.7479</td><td>0.8310</td><td>0.9201</td></tr></table>

Cora, SpecHDC is particularly effective at Level 2 under moderate-to-high supervision. It achieves the highest M from 50% to 90% labeled vertices, increasing from 0.4705 to 0.7157. This indicates that its finer

Class Consistency. Compared with modularity, the F-measure generally benefits more consistently from increasing supervision, especially at Level 1, where the number of clusters matches the number of classes. On Cora, SpecHDC achieves the best Level 1 F at the 10%, 20%, and 40% settings and the second-best result at 60% and 90%. It also attains the best Level 2 score at 50%. These results demonstrate that SpecHDC can effectively use limited labels to recover class-aligned partitions, particularly under low-tomoderate supervision. On Squirrel, the differences among the leading methods become relatively small as the supervision ratio increases. Nevertheless, SpecHDC remains competitive across both levels and reaches a Level 1 F-measure of 0.9201 at 90%, which is close to the best result of 0.9208. Its strong results at several intermediate ratios further indicate that the hierarchical constraints remain effective despite the highly heterophilic structure of the network. On Cornell, SpecHDC performs strongly in both unsupervised and supervised settings. It achieves the best F-measure at Level 2 under 0% and 30% supervision and at Level 1 under 70% and 80% supervision. At 90%, its Level 1 score of 0.9039 is the second best. This shows that the method can capture both fine structural groups and coarse class-aligned clusters, depending on the available supervision. The clearest F-measure advantage is observed on Texas. At Level 1, SpecHDC achieves the best result at 10%, 40%, 50%, 60%, 70%, and 90% supervision. It also obtains the best Level 2 score at 70%. Its Level 1 F-measure reaches 0.9179 at 90%, outperforming all baselines. These results confirm that the inherited label constraints are particularly effective for recovering the semantic class structure of Texas.

Overall, the two metrics reveal complementary properties of the proposed hierarchy. The modularity advantage is more dataset- and resolution-dependent, reflecting differences between structural communities and semantic classes. By contrast, SpecHDC shows more consistent improvements in F-measure, especially at Level 1, demonstrating its effectiveness in incorporating limited supervision for class-consistent clustering.

TABLE VII  
CLUSTERING ALGORITHMS ON THE CORNELL DATASET: A COMPARATIVE ANALYSIS OF MODULARITY (M), AND F-MEASURE (F). THE BEST-PERFORMING MODEL IS HIGHLIGHTED IN BOLD, AND THE SECOND-BEST IS MARKED WITH AN UNDERLINE.
<table><tr><td>M</td><td>Trains (%)</td><td>0 (USL)</td><td>10</td><td>20</td><td>30</td><td>40</td><td>50</td><td>60</td><td>70</td><td>80</td><td>90</td></tr><tr><td rowspan="2">Bi-Sym</td><td>Level 2 (10)</td><td>0.1615</td><td>0.1126</td><td>0.1359</td><td>0.0933</td><td>0.0814</td><td>0.0835</td><td>0.1410</td><td>0.0724</td><td>0.1280</td><td>0.1811</td></tr><tr><td>Level 1 (5)</td><td>0.0781</td><td>0.1328</td><td>0.0589</td><td>0.0954</td><td>0.1048</td><td>0.1523</td><td>0.0246</td><td>0.1480</td><td>0.0187</td><td>0.1809</td></tr><tr><td rowspan="2">DD-Sym</td><td>Level 2 (10)</td><td>0.2014</td><td>0.1592</td><td>0.1187</td><td>0.2003</td><td>0.1826</td><td>0.1282</td><td>0.2048</td><td>0.0779</td><td>0.0896</td><td>0.2090</td></tr><tr><td>Level 1 (5)</td><td>0.0597</td><td>0.2724</td><td>0.2941</td><td>0.0461</td><td>0.2301</td><td>0.1811</td><td>0.3907</td><td>0.0608</td><td>0.0027</td><td>0.1023</td></tr><tr><td rowspan="2">DI-SIM</td><td>Level 2 (10)</td><td>0.3187</td><td>0.2820</td><td>0.4102</td><td>0.2320</td><td>0.1026</td><td>0.0637</td><td>0.1783</td><td>0.1409</td><td>0.1112</td><td>0.1924</td></tr><tr><td>Level 1 (5)</td><td>0.1017</td><td>0.1436</td><td>0.3105</td><td>0.1153</td><td>0.1512</td><td>0.3072</td><td>0.0758</td><td>0.1284</td><td>0.0679</td><td>0.1857</td></tr><tr><td rowspan="2">Herm</td><td>Level 2 (10)</td><td>0.1124</td><td>0.0833</td><td>0.0460</td><td>0.0482</td><td>0.0451</td><td>0.0636</td><td>0.1483</td><td>0.1128</td><td>0.1350</td><td>0.1931</td></tr><tr><td>Level 1 (5)</td><td>0.1979</td><td>0.1442</td><td>0.1651</td><td>0.1059</td><td>0.0086</td><td>0.1933</td><td>0.0780</td><td>0.1683</td><td>0.0395</td><td>0.1182</td></tr><tr><td rowspan="2">Skew</td><td>Level 2 (10)</td><td>0.1124</td><td>0.0833</td><td>0.0460</td><td>0.0482</td><td>0.0451</td><td>0.0636</td><td>0.1483</td><td>0.1128</td><td>0.1350</td><td>0.1931</td></tr><tr><td>Level 1 (5)</td><td>0.1970</td><td>0.1866</td><td>0.0921</td><td>0.0614</td><td>0.0289</td><td>0.0944</td><td>0.0259</td><td>0.0815</td><td>0.0395</td><td>0.0356</td></tr><tr><td rowspan="2">SpecHDC</td><td>Level 2 (10)</td><td>0.1139</td><td>0.0881</td><td>0.0573</td><td>0.0426</td><td>0.0613</td><td>0.0276</td><td>0.1369</td><td>0.0755</td><td>0.1305</td><td>0.1951</td></tr><tr><td>Level 1 (5)</td><td>0.0761</td><td>0.3215</td><td>0.0410</td><td>0.2642</td><td>0.1939</td><td>0.1399</td><td>0.0603</td><td>0.0910</td><td>0.1759</td><td>0.1424</td></tr><tr><td>F</td><td>Trains (%)</td><td>0 (USL)</td><td>10</td><td>20</td><td>30</td><td>40</td><td>50</td><td>60</td><td>70</td><td>80</td><td>90</td></tr><tr><td rowspan="2">Bi-Sym</td><td>Level 2 (10)</td><td>0.0869</td><td>0.1268</td><td>0.1562</td><td>0.1810</td><td>0.4310</td><td>0.3964</td><td>0.5648</td><td>0.6135</td><td>0.7751</td><td>0.9032</td></tr><tr><td>Level 1 (5)</td><td>0.1578</td><td>0.3965</td><td>0.6000</td><td>0.3334</td><td>0.4645</td><td>0.4982</td><td>0.5733</td><td>0.7362</td><td>0.7797</td><td>0.8927</td></tr><tr><td rowspan="2">DD-Sym</td><td>Level 2 (10)</td><td>0.0952</td><td>0.1845</td><td>0.4563</td><td>0.3146</td><td>0.5943</td><td>0.4519</td><td>0.6455</td><td>0.7349</td><td>0.7531</td><td>0.8873</td></tr><tr><td>Level 1 (5)</td><td>0.2174</td><td>0.1822</td><td>0.3072</td><td>0.3154</td><td>0.5200</td><td>0.6030</td><td>0.6756</td><td>0.6984</td><td>0.7929</td><td>0.8905</td></tr><tr><td rowspan="2">DI-SIM</td><td>Level 2 (10)</td><td>0.2255</td><td>0.2772</td><td>0.1020</td><td>0.3791</td><td>0.2689</td><td>0.4604</td><td>0.6230</td><td>0.6041</td><td>0.7315</td><td>0.8923</td></tr><tr><td>Level 1 (5)</td><td>0.2260</td><td>0.2911</td><td>0.1386</td><td>0.3768</td><td>0.3886</td><td>0.5757</td><td>0.5816</td><td>0.7129</td><td>0.7619</td><td>0.8897</td></tr><tr><td rowspan="2">Herm</td><td>Level 2 (10)</td><td>0.0970</td><td>0.2428</td><td>0.1040</td><td>0.3376</td><td>0.4176</td><td>0.3815</td><td>0.5650</td><td>0.6114</td><td>0.7730</td><td>0.8927</td></tr><tr><td>Level 1 (5)</td><td>0.0737</td><td>0.2045</td><td>0.3102</td><td>0.2425</td><td>0.4879</td><td>0.4629</td><td>0.6180</td><td>0.7022</td><td>0.7936</td><td>0.9101</td></tr><tr><td rowspan="2">Skew</td><td>Level 2 (10) Level 1 (5)</td><td>0.0970 0.0813</td><td>0.2428 0.3095</td><td>0.1040 0.2753</td><td>0.3376 0.2615</td><td>0.4176 0.5009</td><td>0.3815</td><td>0.5650</td><td>0.6114</td><td>0.7730</td><td>0.8927</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>0.4697</td><td>0.6349</td><td>0.6677</td><td>0.8450</td><td>0.8991</td></tr><tr><td rowspan="2">SpecHDC</td><td>Level 2 (10)</td><td>0.2440</td><td>0.1386</td><td>0.3448</td><td>0.4261</td><td>0.3839</td><td>0.4357</td><td>0.5635</td><td>0.5962</td><td>0.7933</td><td>0.8926</td></tr><tr><td>Level 1 (5)</td><td>0.2293</td><td>0.2079</td><td>0.4524</td><td>0.4130</td><td>0.4384</td><td>0.5050</td><td>0.6014</td><td>0.7638</td><td>0.8556</td><td>0.9039</td></tr></table>

TABLE VIII  
CLUSTERING ALGORITHMS ON THE TEXAS DATASET: A COMPARATIVE ANALYSIS OF MODULARITY (M), AND F-MEASURE (F). THE BEST-PERFORMING MODEL IS HIGHLIGHTED IN BOLD, AND THE SECOND-BEST IS MARKED WITH AN UNDERLINE.
<table><tr><td>M</td><td>Trains (%)</td><td>0 (USL)</td><td>10</td><td>20</td><td>30</td><td>40</td><td>50</td><td>60</td><td>70</td><td>80</td><td>90</td></tr><tr><td rowspan="2">Bi-Sym</td><td>Level 2 (10)</td><td>0.2244</td><td>0.1420</td><td>0.1308</td><td>0.1396</td><td>0.0791</td><td>0.1368</td><td>0.1125</td><td>0.0252</td><td>0.0166</td><td>0.0059</td></tr><tr><td>Level 1 (5)</td><td>0.1530</td><td>0.1084</td><td>0.1091</td><td>0.1560</td><td>0.1151</td><td>0.0090</td><td>0.1340</td><td>0.0426</td><td>0.0828</td><td>0.1817</td></tr><tr><td rowspan="2">DD-Sym</td><td>Level 2 (10)</td><td>0.3540</td><td>0.2166</td><td>0.1529</td><td>0.2291</td><td>0.0915</td><td>0.1632</td><td>0.1231</td><td>0.0240</td><td>0.0086</td><td>0.0051</td></tr><tr><td>Level 1 (5)</td><td>0.2601</td><td>0.2492</td><td>0.2350</td><td>0.2553</td><td>0.0586</td><td>0.2800</td><td>0.3242</td><td>0.1634</td><td>0.1351</td><td>0.2755</td></tr><tr><td rowspan="2">DI-SIM</td><td>Level 2 (10)</td><td>0.3876</td><td>0.2241</td><td>0.2727</td><td>0.2515</td><td>0.2709</td><td>0.1580</td><td>0.1096</td><td>0.0043</td><td>0.0058</td><td>0.0017</td></tr><tr><td>Level 1 (5)</td><td>0.3251</td><td>0.2191</td><td>0.1324</td><td>0.0864</td><td>0.1995</td><td>0.0180</td><td>0.1261</td><td>0.1324</td><td>0.2356</td><td>0.3472</td></tr><tr><td rowspan="2">Herm</td><td>Level 2 (10)</td><td>0.1660</td><td>0.1336</td><td>0.0714</td><td>0.1655</td><td>0.0014</td><td>0.1378</td><td>0.1233</td><td>0.0199</td><td>0.0020</td><td>0.0071</td></tr><tr><td>Level 1 (5)</td><td>0.1544</td><td>0.0852</td><td>0.2178</td><td>0.1955</td><td>0.1876</td><td>0.1024</td><td>0.0889</td><td>0.1577</td><td>0.1123</td><td>0.1791</td></tr><tr><td rowspan="2">Skew</td><td>Level 2 (10)</td><td>0.1660</td><td>0.1336</td><td>0.0714</td><td>0.1655</td><td>0.0014</td><td>0.1378</td><td>0.1233</td><td>0.0199</td><td>0.0020</td><td>0.0071</td></tr><tr><td>Level 1 (5)</td><td>0.1395</td><td>0.1052</td><td>0.1806</td><td>0.1129</td><td>0.1807</td><td>0.0175</td><td>0.0556</td><td>0.1763</td><td>0.0988</td><td>0.0089</td></tr><tr><td rowspan="2">SpecHDC</td><td>Level 2 (10)</td><td>0.2578</td><td>0.2551</td><td>0.2076</td><td>0.1802</td><td>0.0284</td><td>0.1208</td><td>0.1055</td><td>0.0190</td><td>0.0155</td><td>0.0025</td></tr><tr><td>Level 1 (5)</td><td>0.1915</td><td>0.2154</td><td>0.0454</td><td>0.0556</td><td>0.0885</td><td>0.0719</td><td>0.1646</td><td>0.1133</td><td>0.3336</td><td>0.1114</td></tr><tr><td>F</td><td>Trains (%)</td><td>0 (USL)</td><td>10</td><td>20</td><td>30</td><td>40</td><td>50</td><td>60</td><td>70</td><td>80</td><td>90</td></tr><tr><td rowspan="2">Bi-Sym</td><td>Level 2 (10)</td><td>0.1014</td><td>0.0298</td><td>0.1724</td><td>0.1617</td><td>0.5147</td><td>0.4158</td><td>0.6130</td><td>0.6490</td><td>0.7542</td><td>0.8647</td></tr><tr><td>Level 1 (5)</td><td>0.2125</td><td>0.1344</td><td>0.3098</td><td>0.3900</td><td>0.3970</td><td>0.4951</td><td>0.6795</td><td>0.6754</td><td>0.7830</td><td>0.8803</td></tr><tr><td rowspan="2">DD-Sym</td><td>Level 2 (10)</td><td>0.2015</td><td>0.3793</td><td>0.1310</td><td>0.1940</td><td>0.4068</td><td>0.4572</td><td>0.5531</td><td>0.6294</td><td>0.7733</td><td>0.8738</td></tr><tr><td>Level 1 (5)</td><td>0.1256</td><td>0.3345</td><td>0.1893</td><td>0.3889</td><td>0.4156</td><td>0.5595</td><td>0.6625</td><td>0.7194</td><td>0.8194</td><td>0.8991</td></tr><tr><td rowspan="2">DI-SIM</td><td>Level 2 (10)</td><td>0.2398</td><td>0.0466</td><td>0.3560</td><td>0.5918</td><td>0.4634</td><td>0.5470</td><td>0.6277</td><td>0.6089</td><td>0.7315</td><td>0.8779</td></tr><tr><td>Level 1 (5)</td><td>0.1463</td><td>0.1822</td><td>0.2526</td><td>0.4041</td><td>0.3953</td><td>0.4580</td><td>0.5868</td><td>0.6320</td><td>0.7609</td><td>0.8897</td></tr><tr><td rowspan="2">Herm</td><td>Level 2 (10)</td><td>0.0161</td><td>0.1508</td><td>0.1384</td><td>0.3140</td><td>0.4368</td><td>0.4704</td><td>0.5100</td><td>0.6063</td><td>0.7312</td><td>0.8673</td></tr><tr><td>Level 1 (5)</td><td>0.1810</td><td>0.4223</td><td>0.1118</td><td>0.4043</td><td>0.3571</td><td>0.4878</td><td>0.5548</td><td>0.6453</td><td>0.8339</td><td>0.8905</td></tr><tr><td rowspan="2">Skew</td><td>Level 2 (10)</td><td>0.0161</td><td>0.1508</td><td>0.1384</td><td>0.3140</td><td>0.4368</td><td>0.4704</td><td>0.5100</td><td>0.6063</td><td>0.7312</td><td>0.8673</td></tr><tr><td>Level 1 (5)</td><td>0.1630</td><td>0.2985</td><td>0.1008</td><td>0.4210</td><td>0.3554</td><td>0.4827</td><td>0.5575</td><td>0.6231</td><td>0.8499</td><td>0.8923</td></tr><tr><td rowspan="2">SpecHDC</td><td>Level 2 (10)</td><td>0.0290</td><td>0.0308</td><td>0.3290</td><td>0.1755</td><td>0.4303</td><td>0.4572</td><td>0.6234</td><td>0.7722</td><td>0.7500</td><td>0.8621</td></tr><tr><td>Level 1 (5)</td><td>0.0856</td><td>0.6265</td><td>0.1936</td><td>0.3348</td><td>0.7460</td><td>0.5920</td><td>0.6868</td><td>0.7416</td><td>0.8055</td><td>0.9179</td></tr></table>