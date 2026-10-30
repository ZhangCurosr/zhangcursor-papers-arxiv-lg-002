# Topology Survives Shuffling: Robust Long-Context Memory Access Prediction for Graph Analytics

Dongyan Sun<sup>\*</sup>, Neelesh Gupta<sup>\*†</sup>, Rajgopal Kannan<sup>‡</sup>, Viktor K. Prasanna

University of Southern California, USA

<sup>‡</sup>DEVCOM Army Research Office, USA

Email: {dongyans, neeleshg, prasanna}@usc.edu, rajgopal.kannan.civ@army.mil

Abstract—Graph analytics over structured complex networks generates irregular, topology-dependent memory-access streams via indirect traversals, limiting the effectiveness of data prefetching. Accurate prediction enables timely prefetches that hide memory latency. However, existing machine-learning predictors couple short histories with fixed horizons, fitting the training trace’s node layout rather than learning transferable graph patterns. We propose LoCoMAP, a long-context representation-learning framework for memory-access prediction with independently controlled observation history and prediction horizon. We evaluate Connected Components (CC), PageRank (PR), and Single-Source Shortest Paths (SSSP) on six synthetic graph families, with Erdos–R˝ enyi (ER) and Grid as random and deterministic controls.´ We test two interventions: same-family regeneration changes connectivity, while node shuffling disrupts layout but preserves connectivity. Across 18 graph–workload pairs, extending history from 8 to 128 accesses improves F1 by 7.8%, while extending prediction horizon from 8 to 512 improves F1 by 88.3%. For CC, regeneration is less disruptive than shuffling: without retraining, long context retains 95.4% of original F1 and outperforms short context in six families. Under full shuffle, long context improves CC F1 by 33.6% across ER, Barabasi–Albert, Copying Model,´ and Stochastic Block Model (SBM), with a 2.06× gain on SBM, but not on Watts–Strogatz or Grid, where near-zero delta mass shifts into heavy tails beyond the predictor’s range. Long context outperforms short context on six real-world CC graphs. At full shuffle, long-context PR retains 57.6% of original F1 and averages 11.0% above short context. Both SSSP models retain about 30% of original F1 under full shuffle as distance updates access vertex data in a new order. Within these bounds, long-context predictors are robust across related graphs and layouts without retraining, enabling a generalizable graph prefetcher.

Index Terms—data prefetching, graph analytics, graph topology, memory access prediction, long-context modeling

## I. INTRODUCTION

Data prefetching hides memory latency by predicting future memory accesses and fetching data into the cache hierarchy before the processor requests it [1]–[3]. Its effectiveness rests on accurate memory access prediction: correct and timely predictions overlap latency with computation, while incorrect ones waste bandwidth and pollute caches [4]. Rule-based hardware prefetchers exploit simple regularities such as streams and strides but fail when accesses become irregular [5]–[8]. Machine learning (ML) prefetchers instead learn access patterns as sequence models and achieve substantially higher accuracy on irregular workloads [9]–[11], yet graph analytics remain among the hardest applications to predict [12], [13].

Graph analytics underpin web search, recommendation systems, and social networks [14], [15], but they execute through indirect memory accesses (e.g., row[col[i]]) whose targets depend on previously loaded values [16], [17], producing sparse access streams with poor locality that leave graph processing memory-bound [12], [18], [19]. This irregularity, however, is not arbitrary [20], [21]: real-world networks exhibit well-understood structure, such as power-law degree distributions [22], [23], small-world connectivity [24], and community organization [25], all of which can be reproduced under controlled conditions by synthetic graph models [22], [24], [26]–[28]. A predictor that captures this structure can use topology as an exploitable prior for memory access prediction.

Existing ML predictors do not exploit this prior. State-ofthe-art approaches such as TransFetch [11], SHARP [29], and MPGraph [13] apply a single short-context configuration, with fewer than ten accesses of history, uniformly to all graph workloads, implicitly assuming that all predictive signal lies within a short window and that one configuration fits every graph structure. Neither assumption holds: different topologies induce fundamentally different delta distributions and temporal dependencies. Scale-free graphs [22] revisit hubs at long, irregular intervals; community-structured graphs [25], [28] produce clustered access bursts that exceed any short window; small-world graphs [24] interleave local traversal with longrange jumps; and spatial grids [30] are nearly deterministic. A short-context predictor cannot distinguish these regimes and instead memorizes the vertex layout of its training trace.

Hence, we propose LoCoMAP, a long-context memory access prediction framework for graph analytics. LoCoMAP executes Single-Source Shortest Paths, PageRank, and Connected Components from the GAP benchmark suite [19] on six synthetic graph families: Erdos–R˝ enyi graphs and spatial grids´ as controls at the extremes of pure randomness and perfect regularity, and Barabasi–Albert, Copying Model, stochastic ´ block model, and Watts–Strogatz graphs as representatives of the major structural classes of real-world networks. Lo-CoMAP formulates prediction as a long-context representation learning problem in which the look-back length L, the lookforward window W, and the delta range D are independent parameters, separating how much history a predictor needs from how far its predictions should reach. Because longer contexts raise inference cost under the strict latency budgets of ML prefetching [10], [31]–[33], we characterize the latency and FLOPs cost of extending context alongside its accuracy gains. Finally, we test whether the learned signal is genuinely topological by evaluating trained predictors under progressively shuffled node orderings, which destroy memory locality while preserving connectivity [34], [35]. We summarize our main contributions as follows:

• We propose LoCoMAP, a long-context memory-access prediction framework that maps access histories to multi-hot bitmaps of future address deltas. Independently controlling observation history, prediction horizon, and delta range separates the context used to form the representation from the temporal and spatial scope of its predictions.

• We construct a topology-controlled evaluation of CC, PR, and SSSP across six synthetic graph families. Erdos–R ˝ enyi´ and Grid provide random and deterministic controls; Barabasi–Albert, Copying Model, stochastic block model,´ and Watts–Strogatz capture hub-dominated growth, link copying, community organization, and small-world connectivity across biological, web, social, and infrastructure networks. Same-family regeneration tests unseen connectivity, while progressive node shuffling tests disrupted layouts. Gnutella peer-to-peer, Skitter Internet, Google web, DBLP collaboration, FlyWire connectome, and roadCA networks extend the CC evaluation to real graphs.

• We quantify the accuracy–cost tradeoff of long context. At fixed prediction horizon, increasing history from 8 to 128 accesses improves geometric-mean F1 by 7.8% across 18 graph–workload pairs and by up to 36.8%, while requiring 21× the per-inference FLOPs and 6.3× the latency. At fixed history, increasing the prediction horizon from 8 to 512 accesses improves geometric-mean F1 by 88.3%.

• We identify the limits of predictor reuse without retraining. Long-context CC outperforms short-context on all held-out same-family instances and six fully shuffled real graphs, improving mean F1 on the latter by 10.2%. Under full shuffle, it improves geometric-mean F1 by 33.6% on ER, BA, CM, and SBM and reaches 2.06× short-context F1 on SBM, but loses its advantage on WS and Grid, where delta mass moves beyond the output range. PR retains a long-context advantage on average, but SSSP does not.

## II. BACKGROUND

## A. Graph Analytics and Synthetic Graph Topologies

Graph analytics computes structural properties, such as shortest-path distances, vertex importance, and connectivity, by traversing a graph according to its connectivity [36], [37]. Because traversal order is determined by topology rather than memory layout, the resulting access streams are indirect, with locality that varies across applications and graph structures [12], [16], [19]. Synthetic graph models isolate the influence of topology on memory behavior while avoiding the confounding factors of real-world datasets [20], [38]. As illustrated in Fig. 1, we consider six families, ordered from independent random connectivity to deterministic spatial regularity:

![](images/c9ec8c1afc5d914d5dcc2fe1916deee78f099998fca48b0b00d04c86dbec13df.jpg)  
Fig. 1. Graph Topologies of Synthetic Network Models.

Erdos–R˝ enyi (ER):´ Erdos–R˝ enyi graphs serve as baseline´ models for random communication networks and randomized distributed systems [3]. The $G ( n , p )$ model includes each possible edge independently with probability $p$ [26], yielding a binomial degree distribution:

$$
P ( k ) = { \binom { n - 1 } { k } } p ^ { k } ( 1 - p ) ^ { n - 1 - k }\tag{1}
$$

where $n$ is the number of vertices and $p$ is the edge inclusion probability. The result is homogeneous random connectivity with no hubs and no community structure.

Barabasi–Albert (BA):´ Barabasi–Albert graphs model scale-´ free networks with hub-dominated connectivity, including protein interaction networks, citation networks, and Internet autonomous systems [15]. New vertices attach to existing vertices with probability proportional to their degree [22]:

$$
\Pi ( k _ { i } ) = \frac { k _ { i } } { \sum _ { j } k _ { j } }\tag{2}
$$

where $\Pi ( k _ { i } )$ is the probability that a new vertex connects to vertex i of current degree $k _ { i }$ . Preferential attachment yields a power-law degree distribution $P ( k ) \propto k ^ { - 3 }$ dominated by a small number of high-degree hubs [22].

Copying Model (CM): Copying Model graphs capture the hyperlink structure of the World Wide Web and related information networks, including citation graphs and online document collections [27]. Each arriving vertex selects a random prototype u and forms its i-th out-link either uniformly at random or by copying the prototype $\mathrm { ^ \circ s }$ i-th link:

$$
\operatorname* { P r } ( v _ { i } = w \mid u _ { i } ) = { \frac { \alpha } { n } } + ( 1 - \alpha ) \mathbf { 1 } \{ w = u _ { i } \} .\tag{3}
$$

where $v _ { i }$ is the destination of the new vertex’s i-th link, $u _ { i }$ is the i-th neighbor of the prototype $u , n$ is the current number of vertices, and $\alpha$ is the probability of selecting a destination uniformly at random, while $1 - \alpha$ is the probability of copying $u _ { i } .$ . The indicator function ${ \bf 1 } \{ w = u _ { i } \}$ equals one when $w = u _ { i }$ and zero otherwise. Copying yields power-law degrees and locally dense link patterns that emerge stochastically rather than from an explicit global partition [27].

Stochastic Block Model (SBM): Stochastic Block Model graphs model community-structured networks such as online social communities, scientific collaboration networks, and protein interaction networks [25]. Vertices are partitioned into $K$ communities by an assignment $z \ : \ V \ \to \ \{ 1 , \ldots , K \}$ and each edge is included independently with probability determined by the community memberships of its endpoints:

$$
\operatorname* { P r } \left[ ( u , v ) \in E \right] = B _ { z ( u ) z ( v ) }\tag{4}
$$

where $\operatorname* { P r } \left[ ( u , v ) \in E \right]$ is the probability of an edge between vertices u and v, and $B _ { z ( u ) z ( v ) }$ is the block connection probability for their communities $z ( u )$ and $z ( v )$ . Setting $B _ { i i } > B _ { i j }$ for $i \neq j$ produces densely connected clusters joined by sparse inter-community bridges [28], [39].

Watts–Strogatz (WS): Watts–Strogatz graphs model smallworld networks, including power grids, neuronal networks, and social networks [24]. Starting from a ring lattice, edges are rewired with probability $p ;$ for intermediate $p ,$ the graph combines short paths of a random graph with local clustering:

$$
L ( p ) \gtrsim L _ { \mathrm { r a n d o m } } , \qquad C ( p ) \gg C _ { \mathrm { r a n d o m } }\tag{5}
$$

where $L ( p )$ and $C ( \boldsymbol p )$ denote the path length and clustering coefficient at rewiring probability $p ,$ and $L _ { \mathrm { r a n d o m } }$ and $C _ { \mathrm { r a n d o m } }$ the corresponding quantities for an equivalent random graph. Spatial Grid: Grid graphs represent spatially structured systems such as image pixels, sensor arrays, road networks, and integrated circuit layouts [12]. Vertices lie on a two-dimensional lattice with coordinates $( x _ { u } , y _ { u } )$ and are adjacent exactly at unit Manhattan distance:

$$
( u , v ) \in E \iff | x _ { u } - x _ { v } | + | y _ { u } - y _ { v } | = 1\tag{6}
$$

where $( u , v ) \in E$ denotes an edge between vertices u and v.   
The result has uniform degree and strong spatial locality [30].

Together, these families span the structural spectrum relevant to our evaluation, from independent randomness to increasingly explicit local and mesoscopic organization [20], [21].

## B. Memory Access Prediction

Memory access prediction estimates future memory references from execution history [9], [29]. Following recent MLbased prefetching work [10], [11], we represent each access by its program counter (PC), memory block address, and address delta relative to previous accesses: the PC identifies the referencing instruction, block addresses reduce granularity to cache-line resolution, and deltas encode relative movement that exposes recurring patterns less apparent in absolute addresses [9], [31]. Together, these features provide spatial and program-semantic cues for learning irregular access behavior.

Graph-processing frameworks improve locality through graph partitioning, vertex reordering, and cache-aware data layouts [34], [35], [40]–[44], but largely ignore the accesspattern differences induced by graph topology. We instead use topology-controlled graph families to study whether long context captures topology-induced regularities.

## III. RELATED WORK

## A. Rule-Based and ML-Based Data Prefetching

Data prefetching hides memory latency by predicting future memory accesses and bringing data into the cache hierarchy before it is requested by the processor [2], [3], [5]. Effective prefetching therefore depends on accurate memory access prediction [4]. For graph analytics, sparse graph structures and indirect memory references generate irregular access patterns that significantly reduce the effectiveness of conventional hardware prefetchers [12], [17], [45].

1) Rule-Based Prefetchers: Existing data prefetching approaches can be broadly divided into rule-based and machine learning (ML)-based methods. Rule-based prefetchers, including stride, stream, and Best-Offset (BO) [6], [8], [46], exploit spatial or temporal locality using predefined heuristics [7], [47] and achieve good performance on regular workloads. However, graph analytics generate sparse and indirect memory accesses that exhibit little predictable locality, limiting the effectiveness of these rule-based designs [12], [19]. To address this challenge, graph-aware prefetchers such as the Indirect Memory Prefetcher (IMP) exploit pointer-based memory references to improve prediction accuracy for graph workloads [48], [49]. However, IMP is primarily designed for L1 cache prefetching and does not address memory access prediction for last-level cache (LLC) prefetching, where accurate prediction matters most: LLC prefetching hides DRAM accesses costing hundreds of cycles, whereas L1/L2 prefetching hides comparatively fast L2/LLC accesses. Consequently, rule-based approaches remain limited in capturing the complex long-context memory access behavior of irregular graph analytics.

2) ML-Based Prefetchers: Recently, ML-based prefetchers formulate memory access prediction as a sequence modeling problem [9]. Representative approaches include Voyager [10], TransFetch [11], and MPGraph [13], which learn memory access patterns directly from execution traces instead of relying on fixed heuristics. These methods have demonstrated significantly higher prediction accuracy than traditional prefetchers on irregular workloads [13], [29]. However, their inference latency and storage requirements exceed the strict timeliness constraints of hardware prefetching, especially at the LLC; recent efforts such as Net2Tab narrow this gap by distilling attention-based predictors into compact hierarchies of tables [31], [33].

Existing ML-based approaches, however, fix one shortcontext model for all workloads and do not examine how topology shapes access behavior. In this work, we treat execution history as a controllable parameter, evaluated across topology-controlled graph families, and use node shuffling to separate learned structure from memorized layout.

## IV. APPROACH

LoCoMAP formulates memory access prediction as a longcontext sequence-learning problem and evaluates how much information longer context provides across graph topologies.

![](images/eedea75215a9595ea8ad079f25fa7d22eec7ca490f26ad1951162ba43a5b5b70.jpg)  
Fig. 2. Overall Architecture of LoCoMAP .

## A. Problem Formulation.

Let $X _ { t } = \{ x _ { 1 } , x _ { 2 } , \ldots , x _ { L } \}$ be the sequence of L history block addresses at time t, where $x _ { i } ~ = ~ a _ { i } ~ \gg ~ \beta$ discards the β-bit block offset from the raw address $a _ { i }$ . Let $P _ { t } \ =$ $\{ \mathrm { p c } _ { 1 } , \mathrm { p c } _ { 2 } , \ldots , \mathrm { p c } _ { L } \}$ be the sequence of L history program counters at time $t ,$ and let $I _ { t } = \{ ( x _ { 1 } , \mathrm { p c } _ { 1 } ) , \dots , ( x _ { L } , \mathrm { p c } _ { L } ) \}$ be the input of the prediction model. Let $O _ { t } = \{ o _ { 1 } , o _ { 2 } , . . . , o _ { 2 D } \}$ be the multi-hot label over the future block deltas within range $[ - D , + D ]$ occurring across the next W accesses. Our goal is to learn a meaningful mapping from $I _ { t }$ to $O _ { t }$ for data prefetching; final address predictions are the addition of current block address and predicted deltas. Unlike prior work that fixes a short history, we treat look-back length L as a long context to expose long-range dependencies in the access stream.

## B. Overall Design

Fig. 2 illustrates the overall workflow of LoCoMAP. A synthetic graph is executed using a GAP workload to generate an LLC memory-access trace. Each memory access is converted to a cache block address and represented using fine-grained address segmentation and execution context. An attention-based predictor takes the previous L accesses as input and predicts a bitmap of future address deltas within a look-forward window W and delta bound D. The predicted deltas are then added to the current block address to recover multiple future memory accesses. The L history block addresses are processed by fine grained address segmentation, and the program counters are hashed and embedded as context. The segmented addresses are mapped by a linear projection, combined with position and context embeddings, and encoded by Transformer layers [50], whose self attention relates every access to the entire L-access history in a single step. A classification head with sigmoid activation outputs the probability $p _ { i }$ of each of the 2D deltas, trained with binary cross entropy over the independent labels:

$$
\mathcal { L } = - \frac { 1 } { 2 D } \sum _ { i = 1 } ^ { 2 D } \left[ o _ { i } \log p _ { i } + ( 1 - o _ { i } ) \log ( 1 - p _ { i } ) \right] .\tag{7}
$$

At inference, the output probabilities are binarized with a validation selected threshold that maximizes F1-score, each

predicted delta yields the prefetch address $x _ { L }$ plus that delta, so a single inference produces multiple prefetch candidates.

![](images/68c432f5376d764f4570f21724c22d79b84e6a51ccdae820017b9246331fcfd7.jpg)  
Fig. 3. Visualization of memory access patterns in the benchmark applications.

## V. EVALUATION

## A. Experimental Setup

1) Datasets: We evaluate LoCoMAP using six synthetic graph families that capture structural properties observed in real-world complex networks as well as six real-world graphs as summarized in Table I. The graphs are represented in compressed sparse row (CSR) format for algorithm processing.

2) Algorithms: We implemented three widely used graph processing algorithms based on GPOP framework [51].

• Single-Source Shortest Paths (SSSP) computes the shortest-path distance from a source vertex to all other vertices by propagating tentative distances along edges until convergence [52].

• PageRank (PR) iteratively computes a vertex-importance score by distributing rank mass from each vertex to its neighbors until the scores converge [36].

• Connected Components (CC) labels each vertex by its component using label propagation, where each vertex adopts the minimum label among its neighbors [53].

TABLE I GRAPH STATISTICS.
<table><tr><td rowspan=1 colspan=1>Topology</td><td rowspan=1 colspan=1># Vertices</td><td rowspan=1 colspan=1># Edges</td><td rowspan=1 colspan=1>Avg. Degree</td><td rowspan=1 colspan=1> $\overline { { \mathbf { S } _ { \mathrm { o r d e r } } } }$ </td></tr><tr><td rowspan=1 colspan=1>ER</td><td rowspan=1 colspan=1>1,000,000</td><td rowspan=1 colspan=1>2,400,000</td><td rowspan=1 colspan=1>4.8</td><td rowspan=1 colspan=1>0.0007</td></tr><tr><td rowspan=1 colspan=1>BA</td><td rowspan=1 colspan=1>1,000,000</td><td rowspan=1 colspan=1>4,999,985</td><td rowspan=1 colspan=1>10.0</td><td rowspan=1 colspan=1>0.0016</td></tr><tr><td rowspan=1 colspan=1>CM</td><td rowspan=1 colspan=1>1,000,000</td><td rowspan=1 colspan=1>2,952,366</td><td rowspan=1 colspan=1>5.9</td><td rowspan=1 colspan=1>0.2088</td></tr><tr><td rowspan=1 colspan=1>SBM</td><td rowspan=1 colspan=1>1,000,000</td><td rowspan=1 colspan=1>2,500,000</td><td rowspan=1 colspan=1>5.0</td><td rowspan=1 colspan=1>0.5188</td></tr><tr><td rowspan=1 colspan=1>WS</td><td rowspan=1 colspan=1>1,000,000</td><td rowspan=1 colspan=1>2,000,000</td><td rowspan=1 colspan=1>4.0</td><td rowspan=1 colspan=1>0.9002</td></tr><tr><td rowspan=1 colspan=1>Grid</td><td rowspan=1 colspan=1>1,000,000</td><td rowspan=1 colspan=1>1,998,000</td><td rowspan=1 colspan=1>4.0</td><td rowspan=1 colspan=1>0.9985</td></tr><tr><td rowspan=1 colspan=1>FlyWire</td><td rowspan=1 colspan=1>134,013</td><td rowspan=1 colspan=1>2,313,472</td><td rowspan=1 colspan=1>34.53</td><td rowspan=1 colspan=1>0.0083</td></tr><tr><td rowspan=1 colspan=1>Google</td><td rowspan=1 colspan=1>875,713</td><td rowspan=1 colspan=1>4,322,051</td><td rowspan=1 colspan=1>9.87</td><td rowspan=1 colspan=1>0.0100</td></tr><tr><td rowspan=1 colspan=1>DBLP</td><td rowspan=1 colspan=1>317,080</td><td rowspan=1 colspan=1>1,049,866</td><td rowspan=1 colspan=1>6.62</td><td rowspan=1 colspan=1>0.3632</td></tr><tr><td rowspan=1 colspan=1>Gnutella</td><td rowspan=1 colspan=1>62,586</td><td rowspan=1 colspan=1>147,892</td><td rowspan=1 colspan=1>4.73</td><td rowspan=1 colspan=1>0.4644</td></tr><tr><td rowspan=1 colspan=1>Skitter</td><td rowspan=1 colspan=1>1,696,415</td><td rowspan=1 colspan=1>11,095,298</td><td rowspan=1 colspan=1>13.08</td><td rowspan=1 colspan=1>0.4730</td></tr><tr><td rowspan=1 colspan=1>roadCA</td><td rowspan=1 colspan=1>1,965,206</td><td rowspan=1 colspan=1>2,766,607</td><td rowspan=1 colspan=1>2.82</td><td rowspan=1 colspan=1>0.9937</td></tr></table>

## B. Metrics

We report $\mathrm { F 1 } = 2 P R / ( P + R )$ , the harmonic mean of precision P and recall R [54].

Unless stated otherwise, aggregate improvements are geometric means of per-topology–workload F1 ratios, and “up to” refers to the maximum single-pair ratio.

To quantify the organization of the native vertex ordering, we first compute the normalized mean edge span

$$
\bar { s } = \frac { 1 } { | E | } \sum _ { ( u , v ) \in E } \frac { | u - v | } { | V | - 1 } .\tag{8}
$$

We then compare the native edge span with its randomly relabeled value and compute the ordering score as

$$
\begin{array} { r l } & { R _ { \mathrm { e d g e } } = \frac { \overline { { s } } _ { \mathrm { n a t i v e } } } { \overline { { s } } _ { \mathrm { r a n d o m } } } , } \\ & { S _ { \mathrm { o r d e r } } = \frac { | R _ { \mathrm { e d g e } } - 1 | } { \operatorname* { m a x } ( R _ { \mathrm { e d g e } } , 1 ) } . } \end{array}\tag{9}
$$

Here, $\bar { s } _ { \mathrm { n a t i v e } }$ and ${ \bar { s } } _ { \mathrm { r a n d o m } }$ denote the normalized mean edge spans before and after complete random relabeling. An $S _ { \mathrm { o r d e r } }$ near zero indicates a random-like assignment, whereas a value closer to one indicates organization in the CSR layout.

## C. Controlled Graph and Trace Generation

We generate all graphs synthetically using the standard generative model of each family, with a fixed target size and comparable average degree, so families differ in structure rather than scale; each graph is stored in CSR format for processing.

To obtain memory-access traces, we develop a Pintool using Intel Pin [55] that detects the memory-access operations executed by each graph workload and records the corresponding context information, including the accessed virtual address and program counter. We then replay the collected accesses through a four-core ChampSim [56] model with a shared last-level cache (LLC) and record the resulting per-core LLC access stream, which forms the input to memory-access prediction.

Across the 18 graph-workload traces, we observe 14.1K-5.19M LLC accesses, 276-131.0k unique block addresses, 600- 4.1K pages, and 1.8k-613.5K unique deltas.

TABLE II  
MODEL CONFIGURATION.
<table><tr><td></td><td>Configuration</td><td>Value</td><td>Configuration</td><td>Value</td></tr><tr><td>Input/output</td><td>Delta bound History N</td><td>±128 9</td><td>Delta bitmap B Look-forward W</td><td>256 256</td></tr><tr><td rowspan="2">Attention</td><td>Dimension D</td><td>64</td><td>MLP head layer</td><td>1</td></tr><tr><td>Head number</td><td>4</td><td>Layer L</td><td>1</td></tr></table>

## D. Long-Context Configuration and Cost

We investigate long-context memory access prediction by varying the look-back length L and the look-forward window W. Increasing the look-back length L lengthens the input sequence, regularities at any temporal distance within the window, such as hub revisitation and community dwell intervals, become directly reachable by attention. Increasing the look-forward window W aggregates deltas over a longer future horizon, so the bitmap becomes denser and long-range predictive information is converted into additional coverage.

However, the representation gains of long context are not free. Fig. 4 reports inference FLOPs and latency as the lookback length scales from L=8 to L=128. FLOPs per inference grow from 8M to 169.6M, a 21× increase, and latency grows from 11,357 to 71,345 [cycles], a 6.3× increase, reflecting the quadratic cost of self attention at large L. Setting these costs against Table III yields the practical tradeoff: for several topology and workload pairs the accuracy gains largely arrive by L=64, while cost continues to grow quadratically beyond it, so intermediate context lengths capture much of the benefit at a fraction of the cost. Established deployment techniques for ML prefetchers, including model compression and distance prefetching [13], [31]–[33], [57], apply directly to our models and can offset the remaining latency.

![](images/b200f66e52af3e76c441edf6d3b2f2130d7d572d4e014cc13b86f334d885af4b.jpg)

![](images/aa9aaade13392771df018601d4ec862a15306dee54af4b1e7d7631dfa44c6cad.jpg)

![](images/e9e07bddcafaf40381fba3842378275acc97e2eac6c5f821ff8cce5832e68c75.jpg)  
Fig. 4. Latency, FLOPs and memory cost as the look-back length L scales from 8 to 128.

## E. Training and Evaluation

For each graph–workload pair, we split the collected memory trace into training and testing sets and train a delta predictor to map the access history to a bitmap of future address deltas.

## VI. RESULTS

## A. Ablation Study

We conduct ablation studies on the two long-context parameters, the look-back length L and the look-forward window W, and report the results in Table III.

1) Effectiveness of look-back Length: Increasing L from 8 to 128 consistently improves prediction accuracy across all workloads. Averaged over the six topologies, the F1-score improves by 5.0% for CC, 3.6% for PR, and 15.0% for SSSP.

TABLE III  
ABLATION OF LOOK-BACK LENGTH L AND LOOK-FORWARD WINDOW W ACROSS GRAPH TOPOLOGIES AND APPLICATIONS (F1-SCORE)
<table><tr><td rowspan=3 colspan=1>Parameter</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=18>Applications</td></tr><tr><td rowspan=1 colspan=1>Value</td><td rowspan=1 colspan=6>CC</td><td rowspan=1 colspan=6>PR</td><td rowspan=1 colspan=6>SSSP</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ER</td><td rowspan=1 colspan=1>BA</td><td rowspan=1 colspan=1>CM</td><td rowspan=1 colspan=1>SBM</td><td rowspan=1 colspan=1>WS</td><td rowspan=1 colspan=1>Grid</td><td rowspan=1 colspan=1>ER</td><td rowspan=1 colspan=1>BA</td><td rowspan=1 colspan=1>CM</td><td rowspan=1 colspan=1>SBM</td><td rowspan=1 colspan=1>WS</td><td rowspan=1 colspan=1>Grid</td><td rowspan=1 colspan=1>ER</td><td rowspan=1 colspan=1>BA</td><td rowspan=1 colspan=1>CM</td><td rowspan=1 colspan=1>SBM</td><td rowspan=1 colspan=1>WS</td><td rowspan=1 colspan=1>Grid</td></tr><tr><td rowspan=5 colspan=1>Look-back L(W=256, B=256)</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>0.466</td><td rowspan=1 colspan=1>0.561</td><td rowspan=1 colspan=1>0.660</td><td rowspan=1 colspan=1>0.590</td><td rowspan=1 colspan=1>0.690</td><td rowspan=1 colspan=1>0.641</td><td rowspan=1 colspan=1>0.472</td><td rowspan=1 colspan=1>0.658</td><td rowspan=1 colspan=1>0.684</td><td rowspan=1 colspan=1>0.609</td><td rowspan=1 colspan=1>0.733</td><td rowspan=1 colspan=1>0.668</td><td rowspan=1 colspan=1>0.075</td><td rowspan=1 colspan=1>0.256</td><td rowspan=1 colspan=1>0.683</td><td rowspan=1 colspan=1>0.057</td><td rowspan=1 colspan=1>0.363</td><td rowspan=1 colspan=1>0.639</td></tr><tr><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>0.468</td><td rowspan=1 colspan=1>0.574</td><td rowspan=1 colspan=1>0.661</td><td rowspan=1 colspan=1>0.592</td><td rowspan=1 colspan=1>0.691</td><td rowspan=1 colspan=1>0.642</td><td rowspan=1 colspan=1>0.475</td><td rowspan=1 colspan=1>0.669</td><td rowspan=1 colspan=1>0.688</td><td rowspan=1 colspan=1>0.613</td><td rowspan=1 colspan=1>0.738</td><td rowspan=1 colspan=1>0.671</td><td rowspan=1 colspan=1>0.083</td><td rowspan=1 colspan=1>0.273</td><td rowspan=1 colspan=1>0.684</td><td rowspan=1 colspan=1>0.065</td><td rowspan=1 colspan=1>0.354</td><td rowspan=1 colspan=1>0.646</td></tr><tr><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>0.503</td><td rowspan=1 colspan=1>0.568</td><td rowspan=1 colspan=1>0.664</td><td rowspan=1 colspan=1>0.593</td><td rowspan=1 colspan=1>0.714</td><td rowspan=1 colspan=1>0.645</td><td rowspan=1 colspan=1>0.492</td><td rowspan=1 colspan=1>0.661</td><td rowspan=1 colspan=1>0.690</td><td rowspan=1 colspan=1>0.616</td><td rowspan=1 colspan=1>0.740</td><td rowspan=1 colspan=1>0.665</td><td rowspan=1 colspan=1>0.086</td><td rowspan=1 colspan=1>0.303</td><td rowspan=1 colspan=1>0.696</td><td rowspan=1 colspan=1>0.066</td><td rowspan=1 colspan=1>0.365</td><td rowspan=1 colspan=1>0.653</td></tr><tr><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>0.515</td><td rowspan=1 colspan=1>0.620</td><td rowspan=1 colspan=1>0.669</td><td rowspan=1 colspan=1>0.596</td><td rowspan=1 colspan=1>0.731</td><td rowspan=1 colspan=1>0.650</td><td rowspan=1 colspan=1>0.503</td><td rowspan=1 colspan=1>0.670</td><td rowspan=1 colspan=1>0.691</td><td rowspan=1 colspan=1>0.622</td><td rowspan=1 colspan=1>0.748</td><td rowspan=1 colspan=1>0.679</td><td rowspan=1 colspan=1>0.093</td><td rowspan=1 colspan=1>0.311</td><td rowspan=1 colspan=1>0.704</td><td rowspan=1 colspan=1>0.074</td><td rowspan=1 colspan=1>0.358</td><td rowspan=1 colspan=1>0.655</td></tr><tr><td rowspan=1 colspan=1>128</td><td rowspan=1 colspan=1>0.514</td><td rowspan=1 colspan=1>0.635</td><td rowspan=1 colspan=1>0.674</td><td rowspan=1 colspan=1>0.587</td><td rowspan=1 colspan=1>0.722</td><td rowspan=1 colspan=1>0.648</td><td rowspan=1 colspan=1>0.520</td><td rowspan=1 colspan=1>0.682</td><td rowspan=1 colspan=1>0.693</td><td rowspan=1 colspan=1>0.619</td><td rowspan=1 colspan=1>0.757</td><td rowspan=1 colspan=1>0.681</td><td rowspan=1 colspan=1>0.094</td><td rowspan=1 colspan=1>0.312</td><td rowspan=1 colspan=1>0.716</td><td rowspan=1 colspan=1>0.078</td><td rowspan=1 colspan=1>0.369</td><td rowspan=1 colspan=1>0.663</td></tr><tr><td rowspan=5 colspan=1>Look-forward W(L=8, B=256)</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>0.375</td><td rowspan=1 colspan=1>0.209</td><td rowspan=1 colspan=1>0.369</td><td rowspan=1 colspan=1>0.255</td><td rowspan=1 colspan=1>0.150</td><td rowspan=1 colspan=1>0.263</td><td rowspan=1 colspan=1>0.355</td><td rowspan=1 colspan=1>0.435</td><td rowspan=1 colspan=1>0.372</td><td rowspan=1 colspan=1>0.394</td><td rowspan=1 colspan=1>0.325</td><td rowspan=1 colspan=1>0.373</td><td rowspan=1 colspan=1>0.049</td><td rowspan=1 colspan=1>0.235</td><td rowspan=1 colspan=1>0.594</td><td rowspan=1 colspan=1>0.066</td><td rowspan=1 colspan=1>0.170</td><td rowspan=1 colspan=1>0.537</td></tr><tr><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>0.361</td><td rowspan=1 colspan=1>0.391</td><td rowspan=1 colspan=1>0.486</td><td rowspan=1 colspan=1>0.428</td><td rowspan=1 colspan=1>0.350</td><td rowspan=1 colspan=1>0.402</td><td rowspan=1 colspan=1>0.389</td><td rowspan=1 colspan=1>0.572</td><td rowspan=1 colspan=1>0.464</td><td rowspan=1 colspan=1>0.513</td><td rowspan=1 colspan=1>0.551</td><td rowspan=1 colspan=1>0.527</td><td rowspan=1 colspan=1>0.065</td><td rowspan=1 colspan=1>0.243</td><td rowspan=1 colspan=1>0.600</td><td rowspan=1 colspan=1>0.057</td><td rowspan=1 colspan=1>0.339</td><td rowspan=1 colspan=1>0.621</td></tr><tr><td rowspan=1 colspan=1>128</td><td rowspan=1 colspan=1>0.431</td><td rowspan=1 colspan=1>0.514</td><td rowspan=1 colspan=1>0.589</td><td rowspan=1 colspan=1>0.549</td><td rowspan=1 colspan=1>0.607</td><td rowspan=1 colspan=1>0.581</td><td rowspan=1 colspan=1>0.432</td><td rowspan=1 colspan=1>0.619</td><td rowspan=1 colspan=1>0.601</td><td rowspan=1 colspan=1>0.582</td><td rowspan=1 colspan=1>0.694</td><td rowspan=1 colspan=1>0.605</td><td rowspan=1 colspan=1>0.092</td><td rowspan=1 colspan=1>0.251</td><td rowspan=1 colspan=1>0.648</td><td rowspan=1 colspan=1>0.075</td><td rowspan=1 colspan=1>0.369</td><td rowspan=1 colspan=1>0.625</td></tr><tr><td rowspan=1 colspan=1>256</td><td rowspan=1 colspan=1>0.466</td><td rowspan=1 colspan=1>0.561</td><td rowspan=1 colspan=1>0.660</td><td rowspan=1 colspan=1>0.590</td><td rowspan=1 colspan=1>0.690</td><td rowspan=1 colspan=1>0.641</td><td rowspan=1 colspan=1>0.472</td><td rowspan=1 colspan=1>0.658</td><td rowspan=1 colspan=1>0.684</td><td rowspan=1 colspan=1>0.609</td><td rowspan=1 colspan=1>0.733</td><td rowspan=1 colspan=1>0.668</td><td rowspan=1 colspan=1>0.075</td><td rowspan=1 colspan=1>0.256</td><td rowspan=1 colspan=1>0.683</td><td rowspan=1 colspan=1>0.057</td><td rowspan=1 colspan=1>0.363</td><td rowspan=1 colspan=1>0.639</td></tr><tr><td rowspan=1 colspan=1>512</td><td rowspan=1 colspan=1>0.507</td><td rowspan=1 colspan=1>0.606</td><td rowspan=1 colspan=1>0.710</td><td rowspan=1 colspan=1>0.623</td><td rowspan=1 colspan=1>0.735</td><td rowspan=1 colspan=1>0.696</td><td rowspan=1 colspan=1>0.501</td><td rowspan=1 colspan=1>0.693</td><td rowspan=1 colspan=1>0.736</td><td rowspan=1 colspan=1>0.620</td><td rowspan=1 colspan=1>0.740</td><td rowspan=1 colspan=1>0.713</td><td rowspan=1 colspan=1>0.095</td><td rowspan=1 colspan=1>0.259</td><td rowspan=1 colspan=1>0.682</td><td rowspan=1 colspan=1>0.118</td><td rowspan=1 colspan=1>0.388</td><td rowspan=1 colspan=1>0.654</td></tr></table>

This shows that additional execution history exposes topologyinduced regularities that are not visible within a short context window, motivating long-context prediction for graph analytics.

2) Effectiveness of look-forward Window: The look-forward window is the dominant factor: extending W from 8 to 512 improves the F1-score by 2.49× for CC, 76.7% for PR, and 51.7% for SSSP. Because a larger output range converts predictive information into additional coverage, this result confirms that much of the exploitable structure in graph access streams lies beyond a short prediction horizon.

## B. Cross-Instance Generalization

To evaluate if learned access patterns transfer beyond a single graph instance, we apply each trained CC predictor to a heldout graph from the same family; Table IV reports the results. The long-context model generalizes without retraining to graphs with unseen connectivity, retaining an average of 94.9% of its original-graph F1-score across the five stochastic families. Across all six families, it achieves higher held-out F1-score and raises mean retention from 92.5% to 95.4%. Retention improves for five families, with BA as the sole exception.

TABLE IV  
CROSS-INSTANCE CC GENERALIZATION WITHOUT RETRAINING. EACH ENTRY REPORTS INSTANCE A F1/HELD-OUT SAME-FAMILY INSTANCE B F1
<table><tr><td></td><td>Original (L = 8) A/B (ret.)</td><td>Long (L = 128) A/B (ret.)</td></tr><tr><td>Family ER</td><td>.466/.393 (84.3%)</td><td>.514/.481 (93.6%)</td></tr><tr><td>BA</td><td>.561/.546 (97.3%)</td><td>.635/.589 (92.8%)</td></tr><tr><td>CM</td><td>.660/.560 (84.8%)</td><td>.674/.622 (92.3%)</td></tr><tr><td>SBM</td><td>.590/.569 (96.4%)</td><td>.587/.576 (98.1%)</td></tr><tr><td>WS</td><td>.690/.656 (95.1%)</td><td>.722/.705 (97.6%)</td></tr><tr><td>Grid</td><td>.641/.622 (97.0%)</td><td>.648/.634 (97.8%)</td></tr><tr><td>Mean</td><td>.601/.558 (92.5%)</td><td>.630/.601 (95.4%)</td></tr></table>

## C. Generalization Under Node Shuffling

To test whether long context enables the predictor to learn graph topology rather than a particular vertex ordering, we evaluate each trained predictor, without retraining, on CC traces generated after permuting a controlled fraction of vertex identifiers, comparing a short context model (L=8) against a long context model (L=128); results are shown in Table V.

Fig. 5 shows that shuffling reshapes each family’s delta distribution in a different way: CM and SBM are nearly unchanged; ER sharpens toward small deltas; BA lowers its peak while staying in the same band; and for WS and Grid, the narrow peak of layout induced locality collapses. Because the changes in distribution span from invariant, centralized to decentralized, long context is needed since our delta ranges are wide and differ across families. A short window of eight accesses exposes too few deltas to characterize a family’s range, whereas a long history samples the distribution broadly enough.

Under complete (100%) shuffling, the long context model retains higher accuracy on all four families representing realworld network classes, with the largest gap on communitystructured SBM graphs, and ER remains the most robust overall, consistent with its homogeneous random connectivity. These are exactly the families whose delta distributions survive shuffling in Fig. 5: structure that persists after the layout is destroyed is carried by connectivity, and long context gives the model enough history to exploit it. In contrast, for WS and Grid, whose accuracy derives from layout-induced neighborhood locality, both models collapse and long context confers no advantage. The robustness benefit of long context is therefore specifically a benefit of learning topology structure.

TABLE V  
F1-SCORE RETAINED UNDER FULL NODE SHUFFLING
<table><tr><td rowspan=1 colspan=1>Graph</td><td rowspan=1 colspan=1>Original Model</td><td rowspan=1 colspan=1>Long Context Model</td></tr><tr><td rowspan=1 colspan=1>ER</td><td rowspan=1 colspan=1>92.5%</td><td rowspan=1 colspan=1>93.8%</td></tr><tr><td rowspan=1 colspan=1>BA</td><td rowspan=1 colspan=1>62.4%</td><td rowspan=1 colspan=1>63.9%</td></tr><tr><td rowspan=1 colspan=1>CM</td><td rowspan=1 colspan=1>57.7%</td><td rowspan=1 colspan=1>67.5%</td></tr><tr><td rowspan=1 colspan=1>SBM</td><td rowspan=1 colspan=1>34.2%</td><td rowspan=1 colspan=1>70.7%</td></tr><tr><td rowspan=1 colspan=1>WS</td><td rowspan=1 colspan=1>31.6%</td><td rowspan=1 colspan=1>25.6%</td></tr><tr><td rowspan=1 colspan=1>Grid</td><td rowspan=1 colspan=1>40.9%</td><td rowspan=1 colspan=1>24.5%</td></tr></table>

## D. When Long Context Improves Robustness

Long-context robustness under node shuffling depends on how each workload’s memory accesses change. CC robustness requires deltas to stay within the predictor’s output range. PR remains predictable as shuffling preserves its full-graph sweep, whereas SSSP loses its data-dependent visited-set access order. Figs. 5–10 compare delta distributions and F1 retention.

![](images/c34de963ac01ac4a014479b1185fbed4850af57982febae54f8d0f4cbcddf20a.jpg)

Fig. 5. Memory-access delta distributions per graph family, before (blue) and after (orange) node shuffling for Connected Components (CC) workload.  
![](images/d9a685b115e657a65debbfe57405f4a2a3467270d5375ace175544073baf9e0a.jpg)  
Fig. 6. F1-score under node shuffling (L=8 vs. L=128, CC workload).

Connected Components: CC propagates minimum component labels across adjacent vertices. Relabeling preserves connectivity but remaps vertex and adjacency addresses, so retention tracks whether deltas remain within the predictor’s output range. Long context achieves higher retention than short context on ER, BA, CM, and SBM, with ER retaining 93.8% of its F1- score. On WS and Grid, near-zero delta mass shifts into heavier tails, reducing long-context retention to 25.6% and 24.5%.

PageRank: PR processes every vertex and edge in each iteration. Relabeling changes vertex and neighbor addresses but preserves this full-graph execution schedule. Under complete shuffling, long context retains 57.6% of its baseline F1-score and improves mean F1-score by 11.0% over short context.

Single-Source Shortest Paths: SSSP advances through a frontier generated by conditional distance relaxations. Relabeling remaps temporal address order of frontier accesses, which the marginal delta distribution does not capture. Both context lengths retain only about 30% of their baseline F1, showing that longer history cannot recover the lost temporal predictability.

![](images/3f5f7368613d13ec6cbbabefc0c790f6ef85a32c2c0f183f131fd3912f8e6181.jpg)

Fig. 7. Memory-access delta distributions per graph family, before (blue) and after (orange) node shuffling for PageRank (PR) workload.  
![](images/61715a6ade44a03aaa484f9f1de4044606c5e88872fa7f7a363b1a007727419c.jpg)  
Fig. 8. F1-score under node shuffling (L=8 vs. L=128, PR workload).

## E. Real-Graph Validation

The synthetic results show that random-like native vertex orderings are more robust to shuffling, and the same trend holds across the six real-world graphs in Fig. 12. For CC, the synthetic results further show that F1 retention decreases as shuffling moves more deltas outside the prediction range. The real-graph delta distributions reproduce this relationship. FlyWire’s memory-access delta distribution remains largely unchanged, whereas the more strongly ordered Skitter and roadCA graphs lose near-zero concentration and shift more delta mass outside the prediction range, corresponding to larger F1 degradation. Under complete shuffling, the long-context model outperforms the short-context model on all six graphs and improves mean F1-score by 10.2%.

![](images/6ea421986080d0366319a5ff0b6bf1d760838d5d2f5c5f03048e7da9317337de.jpg)

Fig. 9. Memory-access delta distributions per graph family, before (blue) and after (orange) node shuffling for Single-Source Shortest Paths (SSSP) workload.  
![](images/7b3fbf907588b58fe6bc48c3932e402ab0535528c0f77dfe39a7a11f474fe841.jpg)  
Fig. 10. F1-score under node shuffling (L=8 vs. L=128, SSSP workload).

For CC, long-context F1 retention decreases as $S _ { \mathrm { o r d e r } }$ approaches one on both synthetic and real graphs, showing that structured vertex orderings are more sensitive to shuffling. FlyWire has a random-like native ordering with $S _ { \mathrm { o r d e r } } = 0 . 0 0 8$ and retains 96.2% of its long-context F1-score, whereas roadCA has a strongly ordered layout with $S _ { \mathrm { o r d e r } } = 0 . 9 9 4$ and retains only 72.5%. Together, these results show that the relationship among random-like ordering, range of delta preservation, and F1 robustness transfers from synthetic to real graphs.

## VII. CONCLUSION

LoCoMAP decouples look-back length from prediction horizon to study topology-dependent memory-access prediction across three GAP workloads and six graph families. Increasing L from 8 to 128 improves F1 by 7.8% and by up to 36.8%, while increasing W from 8 to 512 raises F1 by 88.3%. Under complete relabeling, long context improves F1 by 33.6% across ER, BA, CM, and SBM—including a 2.06× gain on SBM—but not WS or Grid. Same-family and real-graph results show that robustness emerges when relabeling preserves characteristic delta ranges, indicating connectivity learning rather than layout memorization. In future work, we plan to reduce the cost of long context prediction with subquadratic sequence architectures to enable low-latency ML-based prefetchers.

![](images/5b8acaa408692bd502d3f83f03ca81a569f021cae70a422a2c2bbb177bda5290.jpg)

Fig. 11. Memory-access delta distributions for CC on six real-world graphs under progressive node shuffling.  
![](images/d986765eae08465657c84092a7e8c7bcdd29ccc308cb2a847b0816a49897dd9e.jpg)  
Fig. 12. F1-score on six real-world graphs under node shuffling(L=8 vs. L=128, CC workload).

## ACKNOWLEDGMENT

This work was supported by the National Science Foundation (NSF) under Award Nos. CSSI-2311870 and OAC-2411446, and by the NSF IDEAS IUCRC under Award No. ASUB00001781.

Distribution Statement A: Approved for public release. Distribution is unlimited.

[1] M. Dubois, M. Annavaram, and P. Stenstrom,¨ Parallel computer organization and design. Cambridge University Press, 2012.

[2] S. P. Vanderwiel and D. J. Lilja, “Data prefetch mechanisms,” ACM Computing Surveys (CSUR), vol. 32, no. 2, pp. 174–199, 2000.

[3] S. Byna, Y. Chen, and X.-H. Sun, “A taxonomy of data prefetching mechanisms,” in 2008 International Symposium on Parallel Architectures, Algorithms, and Networks (i-span 2008). IEEE, 2008, pp. 19–24.

[4] S. Srinath, O. Mutlu, H. Kim, and Y. N. Patt, “Feedback directed prefetching: Improving the performance and bandwidth-efficiency of hardware prefetchers,” in 2007 IEEE 13th International Symposium on High Performance Computer Architecture (HPCA). IEEE, 2007, pp. 63–74.

[5] A. J. Smith, “Sequential program prefetching in memory hierarchies,” Computer, vol. 11, no. 12, pp. 7–21, 1978.

[6] N. P. Jouppi, “Improving direct-mapped cache performance by the addition of a small fully-associative cache and prefetch buffers,” ACM SIGARCH Computer Architecture News, vol. 18, no. 2SI, pp. 364–373, 1990.

[7] S. Somogyi, T. F. Wenisch, A. Ailamaki, B. Falsafi, and A. Moshovos, “Spatial memory streaming,” ACM SIGARCH Computer Architecture News, vol. 34, no. 2, pp. 252–263, 2006.

[8] P. Michaud, “Best-offset hardware prefetching,” in 2016 IEEE Interna tional Symposium on High Performance Computer Architecture (HPCA). IEEE, 2016, pp. 469–480.

[9] M. Hashemi, K. Swersky, J. R. Smith, G. Ayers, H. Litz, J. Chang, C. Kozyrakis, and P. Ranganathan, “Learning memory access patterns,” arXiv preprint arXiv:1803.02329, 2018. [Online]. Available: https://arxiv.org/abs/1803.02329

[10] Z. Shi, A. Jain, K. Swersky, M. Hashemi, P. Ranganathan, and C. Lin, “A hierarchical neural model of data prefetching,” in Proceedings of the 26th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, 2021, pp. 861–873.

[11] P. Zhang, A. Srivastava, A. V. Nori, R. Kannan, and V. K. Prasanna, “Fine-grained address segmentation for attention-based variable-degree prefetching,” in Proceedings of the 19th ACM International Conference on Computing Frontiers, ser. CF ’22. New York, NY, USA: Association for Computing Machinery, 2022, pp. 103–112. [Online]. Available: https://doi-org.libproxy1.usc.edu/10.1145/3528416.3530236

[12] A. Basak, S. Li, X. Hu, S. M. Oh, X. Xie, L. Zhao, X. Jiang, and Y. Xie, “Analysis and optimization of the memory hierarchy for graph processing workloads,” in 2019 IEEE International Symposium on High Performance Computer Architecture (HPCA). IEEE, 2019, pp. 373–386.

[13] P. Zhang, R. Kannan, and V. K. Prasanna, “Phases, modalities, spatial and temporal locality: Domain specific ml prefetcher for accelerating graph analytics,” in Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, 2023, pp. 1–15.

[14] A. Drosou, I. Kalamaras, S. Papadopoulos, and D. Tzovaras, “An enhanced graph analytics platform (gap) providing insight in big network data,” Journal of Innovation in Digital Ecosystems, vol. 3, no. 2, pp. 83–97, 2016.

[15] R. Meusel, S. Vigna, O. Lehmberg, and C. Bizer, “The graph structure in the web–analyzed on different aggregation levels,” The Journal of Web Science, vol. 1, 2015.

[16] A. Lumsdaine, D. Gregor, B. Hendrickson, and J. Berry, “Challenges in parallel graph processing,” Parallel Processing Letters, vol. 17, no. 01, pp. 5–20, 2007.

[17] S. Ainsworth and T. M. Jones, “Graph prefetching using data structure knowledge,” in Proceedings of the 2016 International Conference on Supercomputing, 2016, pp. 1–11.

[18] W. A. Wulf and S. A. McKee, “Hitting the memory wall: Implications of the obvious,” ACM SIGARCH Computer Architecture News, vol. 23, no. 1, pp. 20–24, 1995.

[19] S. Beamer, K. Asanovic, and D. Patterson, “The gap benchmark suite,”´ arXiv preprint arXiv:1508.03619, 2017.

[20] M. E. J. Newman, “The structure and function of complex networks,” SIAM Review, vol. 45, no. 2, pp. 167–256, 2003.

[21] R. Albert and A.-L. Barabasi, “Statistical mechanics of complex networks,”´ Reviews of Modern Physics, vol. 74, no. 1, pp. 47–97, 2002.

[22] A.-L. Barabasi and R. Albert, “Emergence of scaling in random networks,”´ Science, vol. 286, no. 5439, pp. 509–512, 1999.

[23] A. Broder, R. Kumar, F. Maghoul, P. Raghavan, S. Rajagopalan, R. Stata, A. Tomkins, and J. Wiener, “Graph structure in the web,” Computer Networks, vol. 33, no. 1–6, pp. 309–320, 2000.

[24] D. J. Watts and S. H. Strogatz, “Collective dynamics of ‘small-world networks,” Nature, vol. 393, no. 6684, pp. 440–442, 1998.

[25] M. Girvan and M. E. J. Newman, “Community structure in social and biological networks,” Proceedings of the National Academy of Sciences, vol. 99, no. 12, pp. 7821–7826, 2002.

[26] P. Erdos and A. R ˝ enyi, “On random graphs. i,” ´ Publicationes Mathematicae Debrecen, vol. 6, no. 3–4, pp. 290–297, 1959.

[27] R. Kumar, P. Raghavan, S. Rajagopalan, D. Sivakumar, A. Tomkins, and E. Upfal, “Stochastic models for the web graph,” in Proceedings of the 41st Annual Symposium on Foundations of Computer Science (FOCS), 2000, pp. 57–65.

[28] E. Abbe, “Community detection and stochastic block models: recent developments,” The Journal of Machine Learning Research, vol. 18, no. 1, pp. 6446–6531, 2017. [Online]. Available: https://arxiv.org/abs/1703.10146

[29] P. Zhang, R. Kannan, X. Tong, A. V. Nori, and V. K. Prasanna, “Sharp: Software hint-assisted memory access prediction for graph analytics,” in 2022 IEEE High Performance Extreme Computing Conference (HPEC), 2022, pp. 1–8.

[30] M. Barthelemy, “Spatial networks,”´ Physics Reports, vol. 499, no. 1–3, pp. 1–101, 2011.

[31] P. Zhang, N. Gupta, R. Kannan, and V. K. Prasanna, “Attention, distillation, and tabularization: Towards practical neural network-based prefetching,” arXiv preprint arXiv:2401.06362, 2024.

[32] N. Gupta, P. Zhang, R. Kannan, and V. K. Prasanna, “PaCKD: Pattern clustered knowledge distillation for compressing memory access prediction models,” in 2023 IEEE High Performance Extreme Computing Conference (HPEC). IEEE, 2023, pp. 1–7.

[33] P. Zhang, N. Gupta, R. Kannan, and V. K. Prasanna, “Net2Tab: Tabularizing neural networks with applications to data prefetching,” Journal of Parallel and Distributed Computing, 2026, to appear.

[34] H. Wei, J. X. Yu, C. Lu, and X. Lin, “Speedup graph processing by graph ordering,” in Proceedings of the 2016 International Conference on Management of Data (SIGMOD), 2016, pp. 1803–1818.

[35] V. Balaji and B. Lucia, “When is graph reordering an optimization? Studying the effect of lightweight graph reordering,” in 2018 IEEE International Symposium on Workload Characterization (IISWC). IEEE, 2018, pp. 203–214.

[36] L. Page, S. Brin, R. Motwani, and T. Winograd, “The PageRank citation ranking: Bringing order to the web,” Stanford InfoLab, Tech. Rep. 1999- 66, 1999.

[37] G. Malewicz, M. H. Austern, A. J. C. Bik, J. C. Dehnert, I. Horn, N. Leiser, and G. Czajkowski, “Pregel: A system for large-scale graph processing,” in Proceedings of the 2010 ACM SIGMOD International Conference on Management of Data, 2010, pp. 135–146.

[38] D. Chakrabarti, Y. Zhan, and C. Faloutsos, “R-MAT: A recursive model for graph mining,” in Proceedings of the 2004 SIAM International Conference on Data Mining (SDM), 2004, pp. 442–446.

[39] P. W. Holland, K. B. Laskey, and S. Leinhardt, “Stochastic blockmodels: First steps,” Social Networks, vol. 5, no. 2, pp. 109–137, 1983.

[40] J. E. Gonzalez, Y. Low, H. Gu, D. Bickson, and C. Guestrin, “Powergraph: Distributed graph-parallel computation on natural graphs,” in 10th USENIX Symposium on Operating Systems Design and Implementation (OSDI 12), 2012, pp. 17–30.

[41] J. Shun and G. E. Blelloch, “Ligra: A lightweight graph processing framework for shared memory,” in Proceedings of the 18th ACM SIG-PLAN Symposium on Principles and Practice of Parallel Programming (PPoPP), 2013, pp. 135–146.

[42] A. Kyrola, G. Blelloch, and C. Guestrin, “Graphchi: Large-scale graph computation on just a pc,” in 10th USENIX Symposium on Operating Systems Design and Implementation (OSDI 12), 2012, pp. 31–46.

[43] S. Beamer, K. Asanovic, and D. Patterson, “Locality exists in graph´ processing: Workload characterization on an ivy bridge server,” in 2015 IEEE International Symposium on Workload Characterization (IISWC). IEEE, 2015, pp. 56–65.

[44] A. Mukkara, N. Beckmann, M. Abeydeera, X. Ma, and D. Sanchez, “Exploiting locality in graph analytics through hardware-accelerated traversal scheduling,” in 2018 51st Annual IEEE/ACM International Symposium on Microarchitecture (MICRO). IEEE, 2018, pp. 1–14.

[45] S. Ainsworth and T. M. Jones, “Software prefetching for indirect memory accesses: A microarchitectural perspective,” ACM Transactions on Computer Systems (TOCS), vol. 36, no. 3, pp. 1–34, 2018.

[46] T.-F. Chen and J.-L. Baer, “Effective hardware-based data prefetching for high-performance processors,” IEEE Transactions on Computers, vol. 44, no. 5, pp. 609–623, 1995.

[47] A. Jain and C. Lin, “Linearizing irregular memory accesses for improved correlated prefetching,” in Proceedings of the 46th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO), 2013, pp. 247– 259.

[48] J. Yu, H. Lu, C. Ding, and C. Dwyer, “Imp: Indirect memory prefetcher,” in Proceedings of the 50th Annual IEEE/ACM International Symposium on Microarchitecture, 2017, pp. 178–190.

[49] P. Zhang, R. Kannan, and V. K. Prasanna, “GraFetch: Accelerating graph applications through domain-specific hierarchical hybrid prefetching,” IEEE Transactions on Parallel and Distributed Systems, vol. 36, no. 8, pp. 1542–1559, 2025.

[50] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” in Advances in Neural Information Processing Systems (NeurIPS), 2017.

[51] K. Lakhotia, R. Kannan, S. Pati, and V. Prasanna, “Gpop: A scalable cache-and memory-efficient framework for graph processing over parts,” ACM Transactions on Parallel Computing (TOPC), vol. 7, no. 1, pp.

1–24, 2020.

[52] E. W. Dijkstra, “A note on two problems in connexion with graphs,” Numerische Mathematik, vol. 1, no. 1, pp. 269–271, 1959.

[53] Y. Shiloach and U. Vishkin, “An o(log n) parallel connectivity algorithm,” Journal of Algorithms, vol. 3, no. 1, pp. 57–67, 1982.

[54] D. M. W. Powers, “Evaluation: from precision, recall and f-measure to roc, informedness, markedness and correlation,” arXiv preprint arXiv:2010.16061, 2020. [Online]. Available: https://arxiv.org/abs/2010.16061

[55] C.-K. Luk, R. Cohn, R. Muth, H. Patil, A. Klauser, G. Lowney, S. Wallace, V. J. Reddi, and K. Hazelwood, “Pin: Building customized program analysis tools with dynamic instrumentation,” in Proceedings of the 2005 ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI), 2005, pp. 190–200.

[56] N. Gober, G. Chacon, L. Wang, P. V. Gratz, D. A. Jimenez, E. Teran, ´ S. Pugsley, and J. Kim, “The championship simulator: Architectural simulation for education and competition,” 2022. [Online]. Available: https://arxiv.org/abs/2210.14324

[57] N. Gupta, N. Kannan, P. Zhang, and V. K. Prasanna, “TabConv: Lowcomputation CNN inference via table lookups,” in Proceedings of the 21st ACM International Conference on Computing Frontiers, ser. CF ’24. New York, NY, USA: Association for Computing Machinery, 2024.