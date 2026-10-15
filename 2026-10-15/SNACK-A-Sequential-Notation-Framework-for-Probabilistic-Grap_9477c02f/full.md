# SNACK: A Sequential Notation Framework for Probabilistic Graph Generation

Hohyun Kim<sup>∗</sup> Hyesung Kim<sup>∗</sup> Min-hwan Oh<sup>†</sup> Seunggeun Lee<sup>†</sup>

Seoul National University

## Abstract

We introduce SNACK, a string representation for probabilistic graph generation that enables sequence models to generate and model graphs with discrete node and edge features. SNACK linearizes graphs by encoding the nonzero entries of the lower-triangular adjacency matrix, establishing a bijective correspondence between valid sequences and ordered adjacency matrices. This design enables invalid logit masking to enforce structural constraints—such as chemical valence and ordering rules—guaranteeing 100% valid generated sequences. SNACK also enables principled string-based GFlowNet training by explicitly accounting for node-ordering symmetries. Across standard graph-generation tasks and molecular distribution-learning benchmarks, SNACK achieves competitive sample quality, while substantially improving the throughput of GFlowNet training relative to graph-based generation. Probing analyses further show that models trained on SNACK encode meaningful local and global graph structure despite receiving only sequential supervision. These results position SNACK as a practical framework for probabilistic graph generation with deep sequence models.<sup>1</sup>

## 1 Introduction

Two dominant paradigms have been widely explored for generative models for graphs with discrete features: graph-based methods [1, 2, 3], which operate directly on discrete structures, and string-based methods [4, 5, 6], which linearize graphs into sequences. While graph-based methods naturally handle valid actions or trajectories, they often demand specialized architectures and incur high computationa costs. In contrast, string-based representations enable efficient training using standard sequence models. However, existing string-based methods fall short as flexible graph learning frameworks, as they typically address only a subset of key capabilities—such as guaranteed validity, general graph applicability, flexible conditional generation, or goal-directed exploration. Furthermore, a single graph can correspond to multiple valid string representations, which poses a significant challenge to probabilistic modeling. While this issue has been extensively studied in graph-based frameworks [7, 8], it has received comparatively little attention in the context of string-based approaches. Consequently, graph-based methods remain the predominant choice in frameworks such as Generative Flow Networks (GFlowNets) [9].

In this work, we address these challenges by introducing Sequential Notation of Adjacent Connectivity FrameworK (SNACK), a novel string representation that addresses the aforementioned challenges by enabling flexible graph learning. The encoding process of SNACK mirrors the iterative construction found in adjacency matrix-based methods, where a graph is linearized row-wise. This design allows SNACK to inherit core advantages of graph-based methods, while utilizing efficient sequence training pipelines. Specifically, our learning framework provides three practical advantages: first, each representation decodes into a unique adjacency matrix, facilitating string-based GFlowNet training by introducing a node ordering policy; second, any sequence prefix can be readily decoded into a partial graph state, which enables conditioned optimization and direct logit masking during autoregressive generation; and third, SNACK maintains a compact representation, utilizing exactly ${ \bar { N } } + 2 { \bar { M } }$ tokens for an attributed graph with N nodes and M edges, thereby avoiding redundancy while preserving structural correspondence.

We demonstrate the efficacy of SNACK across various graph-related tasks, including distribution learning and property optimization. Our results show that SNACK reliably generates high-quality samples, achieving 100% validity across all benchmarks. Furthermore, SNACK facilitates the implementation of string-based GFlowNets, significantly enhancing the scalability of GFlowNet training on graphs. We also introduce a conditioned Trajectory Balance (TB) loss specifically tailored for constrained generation. Beyond performance gains, we provide empirical evidence that SNACK internalizes graph structural information within its representation despite its sequential nature—a finding that is of independent interest. We summarize our contributions as follows.

• We introduce SNACK, a simple and efficient string representation that maintains a one-toone correspondence with adjacency matrices.

• We propose a training methodology for string-based GFlowNets applicable to both de novo and conditioned settings, which significantly improves the scalability.

• We evaluate SNACK both on distribution and goal-directed learning tasks, showing compet itive generation quality, guaranteed 100% validity, and superior exploration capabilities.

• We employ bilinear probing to show that SNACK-based models internalize graph structures within their representations from purely sequential data without explicit supervision.

## 2 Sequential Notation of Adjacent Connectivity Framework

This section introduces the SNACK framework. We analyze its core properties and define the training objectives necessary for distribution learning and GFlowNet.

## 2.1 Overview

SNACK follows a similar procedure to adjacency matrix-based generative models for undirected graphs, where a graph is constructed node-by-node, along with its corresponding edges to existing nodes. For instance, 2=1-[C] specifies that a node of type [C] is connected to the nodes located two positions and one position earlier, using edge types = and -, respectively.

Formally, consider an undirected graph $G = ( V , E )$ in which each node and edge is associated with a type label. To encode the graph, we assign a unique index to each node via a mapping $\pi : V  [ 1 , N ]$ Let $v _ { i }$ denote the label of the i-th node, and let $e _ { i , j }$ denote the label of the edge between nodes i and j. We define the distance between two nodes i and j as $| i - j |$ . If the k-th node is connected to nodes i and j with $i < j < k$ , the corresponding encoding is

$$
[ \ . \ . . , \ v _ { i } , \ . . . , \ v _ { j } , \ . . . , \ | k - i | , \ e _ { k , i } , \ | k - j | , \ e _ { k , j } , \quad v _ { k } \ ]
$$

where $| k - i |$ and $| k - j |$ represent the distances from node k.

We derive the encoding procedure from the graph’s adjacency matrix, as illustrated in Figure 1. The adjacency matrix A is defined as a symmetric $N \times \bar { N }$ matrix, where off-diagonal entries specify edge types and diagonal entries specify node types. Since all structural information is contained in the lower triangular part of the matrix, it suffices to encode only the lower triangular portion, including the diagonal. We encode the graph row by row in ascending order of node IDs, recording non-zero entries. We adopt the following rules to encode each row k: (1) each off-diagonal entry $( k , i )$ is encoded as two tokens, $( k - i )$ and $e _ { k , i } ,$ , while the node itself is represented by its diagonal entry $v _ { k } ; ( 2 )$ within each row, non-zero entries are encoded from left to right. Left-to-right order prioritizes neighbors that were introduced earlier, thereby reducing the required context window during decoding. We evaluate other design choices for SNACK in the experimental section.

![](images/5343dd0adbb66bce999a0943c47474cc22fc72ad69aed0ebc756f2b66c79c1c5.jpg)  
Figure 1: Overview of the encoding process. The molecule in the figure is represented as [C]1=[C]2-[C]2-[N]2=1-[C]1-[O]. By reordering the node indices, the same graph can also be represented as [O][C][N]3-2=1-[C]3-[C]3-1=[C], resulting in a larger bandwidth.

## 2.2 Bandwidth-restricted encoding

In the current scheme, the distance between two nodes can be as large as N − 1, which requires N − 1 distinct distance tokens in the vocabulary. To reduce the vocabulary size, we aim to keep distances small by choosing an appropriate node ordering π. The largest distance induced by an ordering is known as its bandwidth. As illustrated in Figure 1, the ordering function π can be chosen so that all nonzero entries are concentrated within a narrow band around the diagonal.

Restricting the bandwidth of the adjacency matrix has been shown to substantially improve the performance of graph generative models [10, 6]. Random breadth-first search (BFS) ordering is a standard approach for bandwidth reduction [3, 11, 12, 13], while the Cuthill–McKee (CM) algorithm [14], a BFS variant, has been proposed to further minimize bandwidth [10]. However, in our molecular experiments, BFS ordering consistently outperforms CM. We therefore adopt BFS ordering to control the vocabulary size in SNACK. See Appendix H for a detailed discussion.

## 2.3 Invalid logit masking

Each element in a SNACK sequence corresponds directly to a component of the graph, such as a node or an edge, making the representation easy to interpret. As a result, an intermediate graph can be readily decoded from a partially generated sequence. This prefix-decodability is central to controlled generation: at each step, we are allowed to inspect the partial graph to determine the exact set of admissible next operations, such as valid node connections or compliance with BFS ordering rules. We then convert these operational constraints into a logit mask applied directly to the model’s output distribution, strictly preventing invalid sequence completions before token selection. Specifically, we employ three masking schemes:

• Grammar: We mask tokens that violate syntactic constraints, ensuring that the generated graphs are always valid SNACK representations. This includes enforcing termination by allowing only the [eos] token when appropriate, preventing infinitely long sequences.

• Valence: For molecular graphs, we mask tokens that violate chemical validity, ensuring that the generated molecules are always chemically valid.

• BFS: We mask tokens that do not respect the BFS ordering.

We provide the rules for enforcing BFS masking in Appendix A.2. The BFS mask is crucial, as restricting the training examples to a specific ordering does not guarantee that the model will adhere to the same ordering during generation. Enforcing BFS ordering reduces the modeling space and maintains consistency with the training data, which helps mitigate exposure bias [15]. To avoid confusion, note that BFS masking restricts the set of valid vocabulary tokens the model can predict, whereas BFS ordering determines the training sequence in which the graph is linearized.

## 2.4 Sequence modeling objective

Let a tokenized sequence be denoted as $\mathbf { x } = [ x _ { 1 } , x _ { 2 } , \ldots , x _ { L } ]$ , where $x _ { i }$ is the i-th token. The negative log-likelihood of a sequence under model p<sub>θ</sub> is given by

$$
\mathcal { L } _ { \mathrm { N L L } } ( \mathbf { x } ; \theta ) = - \log p _ { \theta } ( \mathbf { x } ) = - \sum _ { i = 1 } ^ { L } \log p _ { \theta } ( x _ { i } | \mathbf { x } _ { < i } ) ,\tag{1}
$$

where the next-token probability is conditioned on the previous tokens $\mathbf { x } _ { < i }$ , and invalid tokens are masked out. The distribution under which the loss is computed is induced by the graph data and the node ordering policy. Let q(x|G) denote a distribution over sequences given a graph G. As noted in several prior works [1, 7, 16], Equation 1 optimizes a variational lower bound on the graph log-likelihood, given by

$$
\begin{array} { r } { \log p _ { \theta } ( G ) \geq \mathbb { E } _ { \mathbf { x } \sim q ( \mathbf { x } \mid G ) } \left[ \log p _ { \theta } ( \mathbf { x } ) - \log q ( \mathbf { x } \vert G ) \right] . } \end{array}\tag{2}
$$

While we focus on the setting where $q$ is fixed to a random BFS ordering throughout this paper, the following section describes how to explicitly compute $q ,$ as it is essential for GFlowNet training.

## 2.5 Computing the variational distribution

To compute $q ( \mathbf { x } | G )$ , we first note the one-to-one mapping between adjacency matrices and SNACK representations. That is, each SNACK sequence uniquely determines an ordered adjacency matrix, and each such matrix has a unique SNACK representation (proof in Appendix A.4).

Proposition 1. The SNACK representation is bijective with respect to the adjacency matrix.

This implies that $q ( \mathbf { x } | G )$ can be computed by evaluating the probability assigned to the corresponding adjacency matrix A. This property distinguishes SNACK from other string-based representations, in which multiple sequences may map to the same adjacency matrix. For instance, in SELFIES, the strings [C][N] and [Ring1][C][N] yield the same adjacency matrix.

Let $q ^ { \mathrm { o r d } } ( \pi | G )$ denote the ordering policy (e.g., random BFS ordering) that assigns probability to each node ordering π. While a node ordering determines an adjacency matrix A—and a SNACK sequence x—multiple node orderings may induce the same matrix A. Consequently, $q ( \mathbf { x } | G )$ is obtained by summing $q ^ { \mathrm { o r d } } ( \pi | G )$ over all node orderings that preserve A. The set of all node orderings that result in the same adjacency matrix is known as the automorphism group of the matrix, denoted by $\operatorname { A u t } ( \mathbf { A } )$ Thus, $q ( \mathbf { x } | G )$ is given by:

$$
q ( \mathbf { x } | G ) = \sum _ { \pi \in \mathrm { A u t } ( \mathbf { A } ) } q ^ { \mathrm { o r d } } ( \pi | G ) = | \mathrm { A u t } ( \mathbf { A } ) | \cdot q ^ { \mathrm { o r d } } ( \pi | G ) .\tag{3}
$$

The second equality holds because the ordering policy assigns equal probability to all node orderings that yield A. This is a natural consequence of symmetry: node orderings within the same automorphism group induce the same adjacency matrix and are thus structurally indistinguishable.

The ability to evaluate q opens up several possibilities, including: (1) training GFlowNets, (2) substructure-conditioned optimization, (3) learning q, and (4) computing the marginal log-likelihood. We focus specifically on GFlowNets in the next section and discuss the remaining applications in detail in Appendix D.

## 2.6 Training GFlowNets with SNACK

Beyond generating graphs that resemble the training examples, property optimization aims to generate graphs with desired properties. To this end, GFlowNets have gained popularity, as they encourage the generative model to sample high-reward molecular graphs while maintaining diversity [9]. Previous GFlowNet approaches have typically relied on graph-based representations [9, 17, 18, 19, 8], possibly because computing $q ( \mathbf { x } | G )$ for string-based representations is challenging. Although SMILES-based GFlowNets have been adopted owing to the simplicity of string formats [20, 21, 22], this formulation inherently biases generation toward molecules with multiple valid SMILES strings [9]. Consequently, how to properly align string representations with GFlowNets remains largely unexplored. In this section, we demonstrate that GFlowNets can be successfully trained using SNACK.

Let X denote the set of all SNACK representations and $\chi _ { G }$ denote the set of all SNACK representations of graph G. The marginal probability of generating G is then given by $\textstyle \sum _ { \mathbf { x } \in \mathcal { X } _ { G } } p _ { \theta } ( \bar { \mathbf { x } } )$ . The goal of GFlowNets is to learn a policy such that $\begin{array} { r } { \sum _ { \mathbf { x } \in \mathcal { X } _ { G } } p _ { \boldsymbol \theta } ( \mathbf { x } ) = R ( G ) / Z } \end{array}$ , where $\bar { R } : \mathcal G \to \mathbb { R } _ { > 0 }$ denotes a positive reward defined over the set of graphs $\operatorname { \mathcal { G } } , G$ is the graph decoded from any sequence $\mathbf { x } \in \mathcal { X } _ { G }$ , and $Z$ is a normalizing constant.

We define the state space as $S = { \mathcal { X } } \cup { \mathcal { G } }$ . Starting from the empty string (or the [bos] token), the policy generates the next token $x _ { i }$ with probability $p _ { \theta } ( x _ { i } | \mathbf { x } _ { < i } )$ . Once the sequence is complete, a reward ${ \overline { { R } } } ( G )$ is given, where G is the graph decoded from the sequence x. A complete forward trajectory is thus given by $[ \mathbf { x } _ { \leq 1 } , \dotsc , \mathbf { x } _ { \leq L - 1 } , \mathbf { x } , G ]$ , where the transition from x to G is deterministic. Training via GFlowNets also requires evaluating backward trajectory probabilities. In our setting, intermediate backward transitions ${ \bf x } _ { \le i + 1 } \to { \bf x } _ { \le i }$ <sub>i</sub> are deterministically defined by removing the last token, while the initial step $G \to \mathbf { x }$ is governed by $q ( \mathbf { x } | G )$ . Under this formulation, the Trajectory Balance (TB) objective [17] matches two trajectory distributions, $p _ { \theta } ( \tau )$ and $q ( \mathbf { x } | G ) R ( G ) \bar { / } Z ,$ , by minimizing squared error in the log domain:

$$
\mathcal { L } _ { \mathrm { T B } } ( \mathbf { x } , G ; \theta , \gamma ) = \left[ \log Z _ { \gamma } + \log p _ { \theta } ( \mathbf { x } ) - \log q ( \mathbf { x } | G ) - \log R ( G ) \right] ^ { 2 } ,\tag{4}
$$

where $\gamma$ is jointly optimized with $\theta .$ The role of $q ( \mathbf { x } | G )$ is to distribute the reward across all SNACK representations corresponding to $G _ { \ast }$ . The TB loss aims to train the model that samples trajectories in proportion to their assigned rewards, such that $p _ { \theta } ( \mathbf { x } ) \propto q ( \mathbf { x } | G ) R ( G )$ , where log $Z _ { \gamma }$ acts as a variance reduction term [23]. We further develop a conditioned version of GFlowNets that integrates initial state distributions for predefined substructures. The full derivation is provided in Appendix E.

We use random BFS ordering, for which $q ^ { \mathrm { o r d } } ( \pi | G )$ can be evaluated during the BFS traversal, as detailed in Algorithm 1. Since our logit masking scheme ensures that the generation process adheres to a BFS ordering, the supports of $p _ { \theta }$ and $q$ are perfectly aligned. This alignment guarantees that every generated sequence x possesses a positive backward flow, $q ( \mathbf { x } | G ) \bar { R ( G ) }$ . While alternative masking schemes for $p _ { \theta }$ or different ordering policies q could be employed, they may assign zero probability to certain generated samples, potentially resulting in numerical instability.

## 2.7 Overall computation

Although computing $\lvert \operatorname { A u t } ( \mathbf { A } ) \rvert$ is asymptotically expensive [24], in practice, the graphs considered in existing generative models are typically limited to a few hundred nodes. This allows the nauty package [25] to compute $\lvert \operatorname { A u t } ( \mathbf { A } )$ efficiently within a fraction of a millisecond (see Appendix J of Kim et al. [8]). This cost is also inherent to graph-based GFlowNets, which must evaluate $\lvert \operatorname { A u t } ( \mathbf { A } ) \rvert$ to account for equivalent actions [8].

For attributed graphs, SNACK uses $N + 2 M$ tokens, and its vocabulary size scales linearly with bandwidth. SNACK computes trajectory-level log-probabilities in a single forward pass over the full sequence. In contrast, graph-based methods evaluate action probabilities at each intermediate state, requiring $N + M$ forward passes. This leads to higher computational cost and memory usage due to repeated processing of intermediate states. When both methods use standard dense attention [26], this corresponds to a reduction in both computational and memory complexity from cubic to quadratic in the number of nodes (see Appendix C for details).

## 3 Related work

Graph generative models GraphRNN [3] generates graphs by constructing the adjacency matrix row-by-row. While GraphRNN explicitly predicts the presence or absence of every potential edge, our proposed method, SNACK, only encodes non-zero entries, significantly improving efficiency for sparse graphs. The utility of graph bandwidth in improving generative performance was first explored by Diamant et al. [10]. Our treatment of symmetry aligns with Chen et al. [7], who derived an identical automorphism formula for adjacency matrix-based graph generation. While invalid logit masking is commonly utilized in graph-based models to enforce chemical valency [27, 11, 12], these techniques primarily focus on improving the validity of generated molecules rather than reducing the output space. Similarly, recent string-based approaches like GEEL [6] and STGG [5] employ logit masking to satisfy syntactic and chemical constraints. Our work advances this line of research by enforcing a strict BFS ordering during generation through masking.

String representations of graphs For molecular graphs, SMILES [28] is the most widely adopted string representation. Despite being human-readable, SMILES often suffers from low syntactic validity in generative tasks, such as unmatched ring closures. To address this, several robust alternatives have been proposed, such as DeepSMILES [29], SELFIES [4], and STGG [5]. Recent methods such as G2PT [16], GEEL [6] and AutoGraph [30] have also been proposed as general edge list-based graph string representations. Specifically, GEEL uses CM ordering to restrict the vocabulary size; however, it differs from ours in that it constructs graphs edge by edge, and its vocabulary size scales quadratically with the bandwidth. While these approaches are similar to SNACK in their use of graph linearization, their scope and primary objectives fundamentally differ from our work. SNACK is specifically designed to facilitate effective goal-directed learning by integrating graph symmetry and specialized masking schemes. While certain existing methods could potentially be extended to achieve bijectivity relative to the adjacency matrix, to the best of our knowledge, we are the first to explicitly formalize and realize this bijective property within a string-based representation, providing a rigorous foundation for GFlowNet training. See Appendix I for further comparisons.

GFlowNets While prior GFlowNet applications in molecular design have predominantly relied on graph-based approaches [9, 31, 18, 19], we adopt string-based representations to substantially improve scalability. Consistent with recent findings that exact graph GFlowNets require automorphism correction [8], our method integrates graph symmetries to guarantee unbiased sampling. Existing string-based GFlowNets operate on natural language [32, 33, 34] or SMILES [20, 21, 22]; however, SMILES encoding inherently biases the model toward over-sampling larger molecules [9]. Our formulation moves beyond standard autoregressive generation by inducing a directed acyclic graph (DAG) over the state space, allowing multiple distinct strings to transition to the same terminal graph.

## 4 Experiments

In this section, we evaluate SNACK in three settings: generic graph generation, molecular string generation, and goal-directed learning, with a particular focus on molecular generation as a practically relevant application. For distribution learning, we employ 8.9M-parameter, 6-layer Transformers [26] based on the LLaMA architecture [35]. For goal-directed learning tasks, a more compact 4-layer variant is utilized. Further implementation and experimental details are provided in Appendix F.

## 4.1 Generic graph generation

We evaluate SNACK on eight widely used generic graph datasets that include both synthetic and real graphs, enabling us to assess its ability to model diverse graph distributions. Since these are non-attributed graphs, we omit edge-type tokens in the SNACK representation. Following You et al. [3], we use squared Maximum Mean Discrepancy (MMD), which compares the distributions of graph statistics between generated and test graphs. We use the same set of hyperparameters across all datasets without dataset-specific tuning. Table 1 presents a subset of our results, with the full experimental data available in Appendix G. SNACK generates samples whose statistics closely match the training distribution, approaching the benchmark’s performance ceiling. These findings demonstrate SNACK’s ability to model diverse graph structures without dataset-specific tuning.

Table 1: MMD scores for degree (Deg.), clustering (Clus.), and orbit (Orb.) statistics on selected datasets. Results within one standard deviation of the training graphs are highlighted in bold. The baseline results are obtained from Jang et al. [6].
<table><tr><td rowspan="2">Method</td><td colspan="3">Planar</td><td colspan="3">Enzymes</td><td colspan="3">SBM</td><td colspan="3">Grid</td></tr><tr><td>Deg.</td><td>Clus.</td><td>Orb.</td><td>Deg.</td><td>Clus.</td><td>Orb.</td><td>Deg.</td><td>Clus.</td><td>Orb.</td><td>Deg.</td><td>Clus.</td><td>Orb.</td></tr><tr><td>Training</td><td>0.001</td><td>0.004</td><td>0.000</td><td>0.010</td><td>0.019</td><td>0.009</td><td>0.015</td><td>0.001</td><td>0.037</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td>GraphRNN</td><td>0.005</td><td>0.278</td><td>1.254</td><td>0.017</td><td>0.062</td><td>0.046</td><td>0.006</td><td>0.058</td><td>0.079</td><td>0.011</td><td>0.000</td><td>0.021</td></tr><tr><td>GDSS DiGress</td><td>0.250 0.000</td><td>0.393 0.002</td><td>0.587 0.008</td><td>0.026 0.011</td><td>0.061 0.039</td><td>0.009 0.010</td><td>0.496 0.006</td><td>0.456 0.051</td><td>0.717 0.058</td><td>0.111 0.016</td><td>0.005 0.000</td><td>0.070 0.004</td></tr><tr><td>GEEL</td><td>0.001</td><td>0.010</td><td>0.001</td><td>0.005</td><td>0.018</td><td>0.006</td><td>0.025</td><td>0.003</td><td>0.026</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td>SNACK</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>0.001</td><td>0.005</td><td>0.003</td><td>0.003</td><td>0.013</td><td>0.001</td><td>0.018</td><td>0.013</td><td>0.039</td><td>0.000</td><td>0.000</td><td>0.000</td></tr></table>

Table 2: Performance on molecular distribution learning. Results within one standard deviation of the best score are highlighted in bold.
<table><tr><td></td><td colspan="5">QM9</td><td colspan="5">MOSES</td></tr><tr><td>Method</td><td>Valid ↑</td><td>FCD↓</td><td>SNN↑</td><td>Frag ↑</td><td>Scaf ↑</td><td>Valid ↑</td><td>FCD↓</td><td>SNN↑</td><td>Frag ↑</td><td>Scaf ↑</td></tr><tr><td>SMILES</td><td>0.995</td><td>0.053</td><td>0.630</td><td>0.995</td><td>0.976</td><td>0.993</td><td>0.100</td><td>0.528</td><td>1.000</td><td>0.898</td></tr><tr><td>SELFIES</td><td>1.000</td><td>0.037</td><td>0.625</td><td>0.995</td><td>0.975</td><td>1.000</td><td>0.119</td><td>0.514</td><td>1.000</td><td>0.896</td></tr><tr><td>SAFE</td><td>0.969</td><td>0.113</td><td>0.611</td><td>0.992</td><td>0.956</td><td>0.989</td><td>0.258</td><td>0.496</td><td>0.999</td><td>0.877</td></tr><tr><td>STGG</td><td>1.000</td><td>0.121</td><td>0.602</td><td>0.994</td><td>0.962</td><td>1.000</td><td>0.114</td><td>0.528</td><td>1.000</td><td>0.891</td></tr><tr><td>GEEL</td><td>0.995</td><td>0.058</td><td>0.618</td><td>0.993</td><td>0.974</td><td>0.997</td><td>0.104</td><td>0.531</td><td>1.000</td><td>0.894</td></tr><tr><td>G2PT</td><td>0.936</td><td>0.164</td><td>0.585</td><td>0.992</td><td>0.951</td><td>0.919</td><td>0.536</td><td>0.488</td><td>0.998</td><td>0.877</td></tr><tr><td>SNACK (None)</td><td>0.998</td><td>0.046</td><td>0.627</td><td>0.995</td><td>0.977</td><td>0.996</td><td>0.106</td><td>0.516</td><td>1.000</td><td>0.893</td></tr><tr><td>SNACK (V)</td><td>1.000</td><td>0.040</td><td>0.628</td><td>0.994</td><td>0.977</td><td>1.000</td><td>0.105</td><td>0.517</td><td>1.000</td><td>0.900</td></tr><tr><td>SNACK (V+B)</td><td>1.000</td><td>0.037</td><td>0.627</td><td>0.996</td><td>0.977</td><td>1.000</td><td>0.103</td><td>0.518</td><td>1.000</td><td>0.895</td></tr></table>

## 4.2 Molecular string generation

In this section, we focus our comparison specifically on molecular strings using the QM9 [36] and MOSES [37] dataset. We compare SNACK against established molecular string representations, such as SMILES [28], SELFIES [4], STGG [5], and SAFE [38], alongside domain-agnostic methods including GEEL [6] and G2PT [16]. These baselines encompass both BFS- and DFS-based formulations. For our method, we evaluate three logit masking variants: no masking (None), grammar and valence masking (V), and a combination of grammar, valence and BFS masking (V+B). We report standard distribution-learning metrics, as proposed in the MOSES benchmark [37]. By standardizing the underlying architecture and excluding method-specific tricks, we ensure that any observed performance gains are directly attributable to the representations themselves. Additional comparisons across more baselines and metrics are provided in Appendix G.2.

SNACK achieves competitive performance in molecular distribution learning across both datasets. While it remains strong even without masking, adding logit masking enables SNACK to reach 100% validity. This effect is even more pronounced on the more challenging Peptides-func dataset [39], as detailed in Appendix G.2.1. Nevertheless, other string representations—such as SMILES, SELFIES, and GEEL—perform within one standard deviation across several metrics. Overall, SNACK proves to be a highly competitive string-based method for graph distribution learning.

To demonstrate the structural properties of SNACK, we evaluate substructure-conditioned generation without fine-tuning under predefined constraints. SNACK accommodates this by simply appending tokens to the substructure’s string representation. For the 129 smallest MOSES test molecules, 100 random sequences were provided as inputs to the pretrained models. In most cases, baseline models tend to terminate immediately, as intermediate substructures are already valid molecules (or because syntax enforces early stopping, as in GEEL). In contrast, SNACK (None) yields a high success rate (Table 3), where success requires a generated molecule to be valid, connected, contain the target substructure, and possess strictly more heavy atoms. We attribute this to two factors: (1) SNACK allows the model to flexibly attach new atoms to any prior atom within the bandwidth, much like graph-based methods, whereas

Table 3: Success Rate in conditioned generation.
<table><tr><td>Method</td><td>Success</td></tr><tr><td>SMILES</td><td>0.152</td></tr><tr><td>SELFIES</td><td>0.181</td></tr><tr><td>SAFE</td><td>0.057</td></tr><tr><td>STGG</td><td>0.003</td></tr><tr><td>GEEL</td><td>0.000</td></tr><tr><td>G2PT</td><td>0.000</td></tr><tr><td>SNACK</td><td>0.390</td></tr></table>

SMILES can only extend the sequence from the final atom. (2) For SNACK, every subsequence ending with a node-type token constitutes a valid prefix. We discuss this further in Appendix G.4.

## 4.3 Goal-directed learning

We evaluate whether SNACK can effectively support goal-directed graph generation within the GFlowNet framework, comparing it against the 4-layer graph Transformer following Jain et al. [18]. Both SNACK and graph-based baselines use automorphism correction following Kim et al. [8]. HOMO–LUMO gap optimization assesses whether SNACK can match the performance of a graph policy on a standard, small-graph benchmark (max N = 9). In addition, we include two similarity-based optimization tasks using albuterol and aripiprazole as reference molecules; these

Table 4: Results for HOMO–LUMO gap (Gap), Aripiprazole similarity (Ari), and Albuterol similarity (Alb) optimization tasks. Bold indicates results within one standard deviation of the best.
<table><tr><td></td><td>Reward Method</td><td>Avg. Reward</td><td>Diversity</td><td>Top 100 reward Top 100 div. Div. Top 100</td><td></td><td></td><td>Uniq. Frac.</td></tr><tr><td>Gap</td><td>Graph</td><td> $0 . 9 8 4 { \scriptstyle \pm 0 . 0 1 0 }$ </td><td> $0 . 9 5 8 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $\mathbf { 1 . 1 1 8 { \scriptstyle \pm 0 . 0 0 6 } }$ </td><td> $\mathbf { 0 . 9 4 8 _ { \pm 0 . 0 0 6 } }$ </td><td> $\mathbf { 1 . 1 1 8 { \scriptstyle \pm 0 . 0 0 6 } }$ </td><td> $0 . 9 8 2 { \scriptstyle \pm 0 . 0 1 6 }$ </td></tr><tr><td></td><td>SNÁCK (V)</td><td> $0 . 9 8 8 _ { \pm 0 . 0 0 1 }$ </td><td> $0 . 9 5 7 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> ${ \bf 1 . 1 1 7 { \scriptstyle \pm 0 . 0 0 5 } }$ </td><td> $0 . 9 4 0 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> ${ \bf 1 . 1 1 7 { \scriptstyle \pm 0 . 0 0 5 } }$ </td><td> $\mathbf { 1 . 0 0 0 } _ { \pm 0 . 0 0 0 }$ </td></tr><tr><td></td><td>SNACK (V+B)</td><td> $\mathbf { 0 . 9 9 7 { \scriptstyle \pm 0 . 0 0 1 } }$ </td><td> $\mathbf { 0 . 9 5 9 } _ { \pm 0 . 0 0 0 }$ </td><td> $\mathbf { 1 . 1 1 8 { \scriptstyle \pm 0 . 0 0 2 } }$ </td><td> $\mathbf { 0 . 9 4 7 { \scriptstyle \pm 0 . 0 0 2 } }$ </td><td> $\mathbf { 1 . 1 1 8 { \scriptstyle \pm 0 . 0 0 2 } }$ </td><td> $\mathbf { 1 . 0 0 0 } _ { \pm 0 . 0 0 0 }$ </td></tr><tr><td>Ari</td><td>Graph</td><td> $0 . 1 5 9 _ { \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 9 5 6 _ { \pm 0 . 0 0 2 } }$ </td><td> $0 . 2 0 4 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $\mathbf { 0 . 9 6 0 _ { \pm 0 . 0 0 0 } }$ </td><td> $0 . 2 0 4 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $0 . 9 9 7 _ { \pm 0 . 0 0 2 }$ </td></tr><tr><td></td><td>SNACK (V)</td><td> $0 . 1 8 7 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 9 4 2 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 2 5 9 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $0 . 9 3 9 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $0 . 2 5 9 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $\mathbf { 1 . 0 0 0 } _ { \pm 0 . 0 0 0 }$ </td></tr><tr><td></td><td>SNACK (V+B)</td><td> $\mathbf { 0 . 2 9 2 _ { \pm 0 . 0 2 2 } }$ </td><td> $0 . 9 4 5 _ { \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 4 6 9 _ { \pm 0 . 0 0 9 } }$ </td><td> $0 . 9 1 6 _ { \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 4 6 9 _ { \pm 0 . 0 0 9 } }$ </td><td> $\mathbf { 1 . 0 0 0 } _ { \pm 0 . 0 0 0 }$ </td></tr><tr><td>Alb</td><td>Graph</td><td> $0 . 1 1 6 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $\mathbf { 0 . 9 6 0 { \scriptstyle \pm 0 . 0 0 1 } }$ </td><td> $0 . 1 7 1 { \scriptstyle \pm 0 . 0 1 7 }$ </td><td> $\mathbf { 0 . 9 6 0 { \scriptstyle \pm 0 . 0 0 5 } }$ </td><td> $0 . 1 7 1 { \scriptstyle \pm 0 . 0 1 7 }$ </td><td> $0 . 9 9 8 { \scriptstyle \pm 0 . 0 0 1 }$ </td></tr><tr><td></td><td>SNACK (V)</td><td> $0 . 3 5 3 { \scriptstyle \pm 0 . 1 0 1 }$ </td><td> $0 . 9 3 7 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 5 5 2 { \scriptstyle \pm 0 . 1 5 2 }$ </td><td> $0 . 9 2 1 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $0 . 5 5 2 { \scriptstyle \pm 0 . 1 5 2 }$ </td><td> $\mathbf { 1 . 0 0 0 } _ { \pm 0 . 0 0 0 }$ </td></tr><tr><td></td><td>SNACK (V+B)</td><td> $\mathbf { 0 . 5 2 9 } _ { \pm 0 . 0 0 7 }$ </td><td> $0 . 9 3 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $\mathbf { 0 . 7 3 6 { \scriptstyle \pm 0 . 0 0 7 } }$  </td><td> $0 . 9 1 2 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 7 3 6 { \scriptstyle \pm 0 . 0 0 7 } }$ </td><td> $\mathbf { 1 . 0 0 0 } _ { \pm 0 . 0 0 0 }$ </td></tr></table>

tasks probe the scalability of each method as the action space and trajectory length expand (max $N = \bar { 1 } 6 )$

Table 4 shows that SNACK is competitive with the graph-based GFlowNet on the HOMO–LUMO gap task and substantially outperforms it on both similarity rewards, while maintaining a unique fraction of 1.0. In contrast, the graph-based GFlowNet did not reach a unique fraction of 1.0, and its markedly lower Top 100 reward (0.171 versus 0.736 on albuterol) suggests that the graph policy struggles to scale as the action space expands. Notably, results further improve when BFS masking is enforced. This constraint is particularly beneficial for GFlowNet training, as it ensures that every generated sequence possesses a strictly positive backward flow, thereby enhancing overall training stability.

We also trained GFlowNets directly on SMILES, but they failed to make meaningful progress. This failure is primarily attributed to the inability of SMILES to guarantee structural validity; consequently, early-stage samples were dominated by invalid, zero-reward strings. This shows that SNACK provides a more robust foundation for training GFlowNet models, while SMILES-based models inherently require pretraining to mitigate the complexities of their string representation.

Table 5: Conditioned optimization

Conditioned optimization In this experiment, we focus on optimizing molecular structures around a given substructure, a task of critical importance in drug discovery for lead optimization. Following previous work [40, 2], we constrain the generated molecules to contain one of the 800 molecules with low penalized logP, and train with the conditioned TB objective detailed in Appendix E. Since every prefix of a valid SNACK sequence that ends with a node-type token corresponds to a graph, the task amounts to sequentially appending tokens to the initial string representation of the given scaffold. Table 5 demonstrates that SNACK (V+B) effectively optimizes target properties while strictly preserving the required fragment. This suggests that SNACK is fully compatible with principled GFlowNet objectives for conditioned generation, despite its string-based nature.

<table><tr><td>Method</td><td colspan="3">Improvement Similarity Success</td></tr><tr><td>JT-VAE</td><td> $1 . 9 1 _ { \pm 2 . 0 }$ </td><td> $0 . 2 8 { \scriptstyle \pm . 1 }$ </td><td>97.5%</td></tr><tr><td>GCPN</td><td> $4 . 2 0 { \scriptstyle \pm 1 . 3 }$ </td><td> $0 . 3 2 { \scriptstyle \pm . 1 }$ </td><td>100%</td></tr><tr><td>GraphDF</td><td> $5 . 9 3 { \scriptstyle \pm 0 . 8 }$ </td><td> $0 . 3 0 { \scriptstyle \pm . 1 }$ </td><td>100%</td></tr><tr><td>SNACK</td><td> ${ \bf 6 . 4 5 { \scriptstyle \pm 1 . 6 } }$  </td><td> $\mathbf { 0 . 4 6 _ { \pm . 1 5 } }$ </td><td>100%</td></tr></table>

Scalability analysis We further evaluate the empirical scalability of SNACK and graph-based GFlowNet by measuring training throughput across molecules of varying sizes, ensuring a fair comparison by maintaining consistent model capacities and hardware configurations. Figure 2 demonstrates that SNACK-GFlowNet exhibits superior computational efficiency as molecular size increases, significantly outperforming the graph-based baseline in terms of training throughput. Additional results regarding inference latency and memory consumption are provided in Appendix G.3. Together, these findings indicate that SNACK enables effective goal-directed GFlowNet learning without the need for graph-native operations, while offering substantial scalability advantages.

![](images/04fea7c75ce39b29c7e70dc8d101a7ea6ec82fcc87bcfa597eaa125755f78ecf.jpg)  
Figure 2: GFlowNet training time comparison. The $\mathbf { \epsilon } ^ { \star } \mathbf { \times } \mathbf { \epsilon }$ indicates the maximum scale within GPU memory constraints.

## 4.4 Ablation studies

We investigate two alternative design choices for SNACK on molecular datasets. Further ablation studies, including investigations into the impact of edge ordering (e.g., left-to-right encoding) and variations in model scale, are provided in Appendix G.

Different node orderings We evaluate three node ordering strategies: uniform, BFS, and CM. Table 6 summarizes results. BFS ordering outperforms both uniform and CM orderings on both datasets. We hypothesize that BFS strikes a balance between consistency and diversity of the training data: Uniform ordering produces sequences with substantially different semantics, whereas CM ordering does not provide sufficient diversity. Importantly, this finding contrasts with prior studies [10, 6] that favor CM ordering. We attribute this discrepancy to two factors. First, the relative advantage of CM ordering is likely dataset-dependent. Second, prior conclusions were often based on evaluation metrics that do not fully capture generative quality. A more comprehensive analysis is provided in Appendix H.

Table 6: Comparison of FCD scores across different node ordering strategies.
<table><tr><td colspan="2">QM9</td><td>MOSES</td></tr><tr><td>Unif.</td><td> $0 . 2 0 3 _ { \pm 0 . 0 1 3 }$ </td><td> $6 . 5 1 3 { \scriptstyle \pm 0 . 7 1 0 }$ </td></tr><tr><td>CM</td><td> $0 . 0 4 4 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $0 . 1 1 3 { \scriptstyle \pm 0 . 0 1 0 }$ </td></tr><tr><td>BFS</td><td>_  $\mathbf { 0 . 0 3 7 { \scriptstyle \pm 0 . 0 0 3 } }$  </td><td> $\mathbf { 0 . 1 0 3 _ { \pm 0 . 0 0 2 } }$  </td></tr></table>

Different edge orderings A SNACK string can be decomposed into what we refer to as “blocks,” each consisting of a node and its edges to previously generated nodes. For example, 2=1-[C] forms a block that connects the current node to two distinct preceding nodes. We evaluate four alternative within-block design choices, including ascending distance ordering (1-2=[C]) and node-first generation ([C]2=1-). Node-first generation is used in GEEL and also common in graph-based methods [1, 2, 12, 8]. Perhaps surprisingly, however, we find that it diminishes performance consistently in our experiments as shown in Figure 3. This finding suggests that placing connectivity information before the node token can provide a better inductive bias for autoregressive modeling, justifying the current SNACK representation.

![](images/cce2f661e7eb7168b35dc1e8c6b5b5d5bdbd446241cc794dd4443da641d410ab.jpg)  
Figure 3: Validation loss of SNACK (None) on QM9 dataset for each block design.

## 5 Probing graph structure in learned representations

In this section, we use bilinear probing to investigate whether the model’s hidden representations encode underlying graph structure despite the absence of explicit supervision. This is of particular interest for two reasons: First, while prior works often inject structural information directly into models via attention biases [41, 5], the extent to which a model can internalize such information without these explicit biases merits closer examination. Second, recent studies demonstrate that generative models trained solely on linearized representations can be effective for downstream prediction tasks [16, 42]. This raises an important question regarding what structural information is captured by the model during next-token prediction.

Specifically, we assess whether the adjacency matrix and shortest path distance (SPD) matrix can be recovered from the model’s hidden states that correspond to node-type tokens. Let $\mathbf { h } _ { i } ^ { \ell } \in \mathbb { R } ^ { d }$ denote the representation of the i-th node at layer $\ell ,$ with ${ \bf { h } } _ { i } ^ { 0 }$ being the non-contextual embedding. We normalize and aggregate them via a learned linear combination: $\begin{array} { r } { \mathbf { h } _ { i } = \sum _ { \ell } a ^ { \ell } \mathbf { h } _ { i } ^ { \ell } / \| \mathbf { h } _ { i } ^ { \ell } \| } \end{array}$ , where $a ^ { \ell } \in \mathbb { R }$ are learned weights. We then train a bilinear probe parameterized by $\mathbf { W } _ { K } , \mathbf { W } _ { Q } \in \mathbb { R } ^ { d \times d }$

$$
\mathbf { k } _ { i } = \mathbf { W } _ { K } \mathbf { h } _ { i } \quad \mathbf { q } _ { j } = \mathbf { W } _ { Q } \mathbf { h } _ { j } \quad \hat { y } _ { i , j } = \sigma \big ( \mathbf { k } _ { i } ^ { \top } \mathbf { q } _ { j } \big ) , \quad 1 \leq i < j \leq N\tag{5}
$$

For edge prediction, σ denotes the sigmoid function; for SPD estimation, it is the identity. While alternative normalization schemes exist [43, 44], we adopt this formulation for its simplicity and effectiveness. Note that we are essentially training a single weight matrix; overparameterizing it with two full-rank matrices was chosen because we observed that it speeds up training [45].

Using Equation 5, we probe whether a SNACK (V+B) model trained on MOSES encodes graph structure in its hidden representations. Figure 4 compares model predictions on test molecules with ground truth. The near-perfect reconstruction of adjacency matrices (AUROC: 0.9998) indicates that the SNACK-Transformer encodes local connectivity, while the alignment in SPDs (MAE: 0.9250) suggests that it also captures aspects of the graph’s global geometry. This suggests that explicit graph biases are not strictly necessary for capturing complex connectivity; rather, autoregressive pre-training enables the Transformer to internalize the graph’s structure. Table 7 highlights key performance trade-offs between the two approaches: SNACK excels at edge prediction, whereas SMILES is superior at SPD prediction. This contrast suggests that SNACK prioritizes local connectivity, whereas SMILES representations preserve long-range pairwise distances.

![](images/2b556e8f527ce5d066f7d42e1b538e3b6e87e241f56e085f72580858573f7852.jpg)  
Figure 4: Adjacency and SPD matrices recovered from the model’s hidden representations.

Table 7: Model comparison on probing. “Random” denotes a randomly initialized SNACK model.
<table><tr><td rowspan="2">Model</td><td colspan="2">Adjacency</td><td colspan="2">SPD</td></tr><tr><td>Accuracy ↑</td><td>AUROC ↑</td><td>MSE↓</td><td>MAE↓</td></tr><tr><td>Random</td><td>0.9191</td><td>0.9362</td><td>3.0564</td><td>1.2897</td></tr><tr><td>SMILES</td><td>0.9894</td><td>0.9970</td><td>1.2722</td><td>0.8148</td></tr><tr><td>SNACK</td><td>0.9976</td><td>0.9998</td><td>1.6659</td><td>0.9250</td></tr></table>

## 6 Conclusion

In this work, we introduced SNACK, a simple and efficient string representation for graphs that captures structural information through a construction process analogous to adjacency matrix generation. We showed that SNACK enables the direct enforcement of structural constraints, supports GFlowNet training, and achieves competitive performance across several benchmarks. Furthermore, through probing analyses, we demonstrated that models trained with SNACK can recover meaningful graph structure from their internal representations, even without explicit supervision. A limitation of our approach is that the ordering policy is fixed and manually designed. Important directions for future work include extending SNACK to incorporate stereochemistry and chirality, learning the ordering policy jointly with the model, and adapting the representation for fragment-level generation to better capture higher-level structural patterns.

## Acknowledgments

This research was supported by the Advanced GPU Utilization Support Program funded by the Korean government (Ministry of Science and ICT) (No. 02-26-01-0303), the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korean government (MSIT) (No. RS-2025-25463302, Global Talent Recruitment for AI-powered Drug Discovery), the Korea Health Technology R&D Project through the Korea Health Industry Development Institute (KHIDI) funded by the Ministry of Health & Welfare, Republic of Korea (No. RS-2025-02307233), the National Research Foundation of Korea (NRF) grant funded by the Korean government (MSIT) (No. RS-2023-00222663), the SNU AI-Bio Research Fund, and the Advanced Medical and Pharmaceutical Researcher Training Support Program through Seoul National University.

## References

[1] Yujia Li, Oriol Vinyals, Chris Dyer, Razvan Pascanu, and Peter Battaglia. Learning deep generative models of graphs. arXiv preprint arXiv:1803.03324, 2018. 1, 4, 9

[2] Jiaxuan You, Bowen Liu, Zhitao Ying, Vijay Pande, and Jure Leskovec. Graph convolutional policy network for goal-directed molecular graph generation. Advances in neural information processing systems, 31, 2018. 1, 8, 9, 20, 22

[3] Jiaxuan You, Rex Ying, Xiang Ren, William Hamilton, and Jure Leskovec. Graphrnn: Generating realistic graphs with deep auto-regressive models. In International conference on machine learning, pages 5708–5717. PMLR, 2018. 1, 3, 5, 6, 19, 20, 27

[4] Mario Krenn, Florian Häse, AkshatKumar Nigam, Pascal Friederich, and Alan Aspuru-Guzik. Self-referencing embedded strings (selfies): A 100% robust molecular string representation. Machine Learning: Science and Technology, 1(4):045024, 2020. 1, 6, 7, 29

[5] Sungsoo Ahn, Binghong Chen, Tianzhe Wang, and Le Song. Spanning tree-based graph generation for molecules. In International Conference on Learning Representations, 2021. 1, 5, 6, 7, 9, 24, 25, 29

[6] Yunhui Jang, Seul Lee, and Sungsoo Ahn. A simple and scalable representation for graph generation. In 12th International Conference on Learning Representations, ICLR 2024, 2024. 1, 3, 5, 6, 7, 9, 19, 20, 23, 24, 25, 27, 29

[7] Xiaohui Chen, Xu Han, Jiajing Hu, Francisco Ruiz, and Liping Liu. Order matters: Probabilistic modeling of node sequence for graph generation. In International Conference on Machine Learning, pages 1630–1639. PMLR, 2021. 1, 4, 5, 18, 28

[8] Hohyun Kim, Seunggeun Lee, and Min-Hwan Oh. Symmetry-aware gflownets. In International Conference on Machine Learning, pages 30334–30362. PMLR, 2025. 1, 4, 5, 6, 7, 9, 19, 22

[9] Emmanuel Bengio, Moksh Jain, Maksym Korablyov, Doina Precup, and Yoshua Bengio. Flow network based generative models for non-iterative diverse candidate generation. Advances in neural information processing systems, 34:27381–27394, 2021. 1, 4, 6, 21

[10] Nathaniel Lee Diamant, Alex M Tseng, Kangway V Chuang, Tommaso Biancalani, and Gabriele Scalia. Improving graph generation by restricting graph bandwidth. In International Conference on Machine Learning, pages 7939–7959. PMLR, 2023. 3, 5, 9, 20, 24, 27, 29

[11] Mariya Popova, Mykhailo Shvets, Junier Oliva, and Olexandr Isayev. Molecularrnn: Generating realistic molecular graphs with optimized properties. arXiv preprint arXiv:1905.13372, 2019. 3, 5

[12] Chence Shi, Minkai Xu, Zhaocheng Zhu, Weinan Zhang, Ming Zhang, and Jian Tang. Graphaf: a flow-based autoregressive model for molecular graph generation. arXiv preprint arXiv:2001.09382, 2020. 3, 5, 9, 20

[13] Youzhi Luo, Keqiang Yan, and Shuiwang Ji. Graphdf: A discrete flow model for molecular graph generation. In International conference on machine learning, pages 7192–7203. PMLR, 2021. 3, 20, 22

[14] Elizabeth Cuthill and James McKee. Reducing the bandwidth of sparse symmetric matrices. In Proceedings of the 1969 24th national conference, pages 157–172, 1969. 3

[15] Marc’Aurelio Ranzato, Sumit Chopra, Michael Auli, and Wojciech Zaremba. Sequence level training with recurrent neural networks. arXiv preprint arXiv:1511.06732, 2015. 3

[16] Xiaohui Chen, Yinkai Wang, Jiaxing He, Yuanqi Du, Soha Hassoun, Xiaolin Xu, and Liping Liu. Graph generative pre-trained transformer. In International Conference on Machine Learning, pages 9176–9197. PMLR, 2025. 4, 6, 7, 9, 19, 24, 25, 29

[17] Nikolay Malkin, Moksh Jain, Emmanuel Bengio, Chen Sun, and Yoshua Bengio. Trajectory balance: Improved credit assignment in gflownets. Advances in Neural Information Processing Systems, 35:5955–5967, 2022. 4, 5, 22

[18] Moksh Jain, Sharath Chandra Raparthy, Alex Hernández-Garcıa, Jarrid Rector-Brooks, Yoshua Bengio, Santiago Miret, and Emmanuel Bengio. Multi-objective gflownets. In International conference on machine learning, pages 14631–14653. PMLR, 2023. 4, 6, 7, 21, 25

[19] Mohit Pandey, Gopeshh Subbaraj, Artem Cherkasov, Martin Ester, and Emmanuel Bengio. Pretraining generative flow networks with inexpensive rewards for molecular graph generation. In International Conference on Machine Learning, pages 47781–47819. PMLR, 2025. 4, 6

[20] Hyeonah Kim, Minsu Kim, Sanghyeok Choi, and Jinkyoo Park. Genetic-guided gflownets for sample efficient molecular optimization. Advances in Neural Information Processing Systems, 37:42618–42648, 2024. 4, 6, 22

[21] Hyeonah Kim, Minsu Kim, Celine Roget, Dionessa Biton, Louis Vaillancourt, Yves V Brun, Yoshua Bengio, and Alex Hernandez-Garcia. Synthesizable molecular generation via softconstrained gflownets with rich chemical priors. arXiv preprint arXiv:2602.04119, 2026. 4, 6

[22] Xi Wang, Wenbo Lu, and Shengjie Wang. Rooted absorbed prefix trajectory balance with submodular replay for gflownet training. arXiv preprint arXiv:2603.00454, 2026. 4, 6

[23] Nikolay Malkin, Salem Lahlou, Tristan Deleu, Xu Ji, Edward Hu, Katie Everett, Dinghuai Zhang, and Yoshua Bengio. Gflownets and variational inference. arXiv preprint arXiv:2210.00580, 2022. 5

[24] László Babai, William M Kantor, and Eugene M Luks. Computational complexity and the classification of finite simple groups. In 24th Annual Symposium on Foundations of Computer Science (Sfcs 1983), pages 162–171. IEEE, 1983. 5

[25] Brendan D McKay and Adolfo Piperno. nauty and traces user’s guide (version 2.9.3). 2025. 5

[26] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 5, 6, 18, 19

[27] Qi Liu, Miltiadis Allamanis, Marc Brockschmidt, and Alexander Gaunt. Constrained graph variational autoencoders for molecule design. Advances in neural information processing systems, 31, 2018. 5

[28] David Weininger. Smiles, a chemical language and information system. 1. introduction to methodology and encoding rules. Journal of chemical information and computer sciences, 28 (1):31–36, 1988. 6, 7, 29

[29] Noel O’Boyle and Andrew Dalke. Deepsmiles: an adaptation of smiles for use in machinelearning of chemical structures. 2018. 6

[30] Dexiong Chen, Markus Krimmel, and Karsten Borgwardt. Flatten graphs as sequences: Transformers are scalable graph generators. In Conference on Neural Information Processing Systems (NeurIPS 2025), 2025. 6, 29

[31] Moksh Jain, Emmanuel Bengio, Alex Hernandez-Garcia, Jarrid Rector-Brooks, Bonaventure FP Dossou, Chanakya Ajit Ekbote, Jie Fu, Tianyu Zhang, Michael Kilgour, Dinghuai Zhang, et al. Biological sequence design with gflownets. In International conference on machine learning, pages 9786–9801. PMLR, 2022. 6

[32] Edward J Hu, Moksh Jain, Eric Elmoznino, Younesse Kaddar, Guillaume Lajoie, Yoshua Bengio, and Nikolay Malkin. Amortizing intractable inference in large language models. In The Twelfth International Conference on Learning Representations, 2024. 6

[33] Minchan Kwon, Sunghyun Baek, Minseo Kim, Jaemyung Yu, Dongyoon Han, and Junmo Kim. Stable-gflownet: Toward diverse and robust llm red-teaming via contrastive trajectory balance. arXiv preprint arXiv:2605.00553, 2026. 6

[34] Dohyung Kim, Minbeom Kim, Jeonghye Kim, Sangmook Lee, Sojeong Rhee, and Kyomin Jung. Beyond normalization: Rethinking the partition function as a difficulty scheduler for rlvr. arXiv preprint arXiv:2602.12642, 2026. 6

[35] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023. URL https://arxiv.org/abs/2302.13971. 6, 19

[36] Raghunathan Ramakrishnan, Pavlo O Dral, Matthias Rupp, and O Anatole Von Lilienfeld. Quantum chemistry structures and properties of 134 kilo molecules. Scientific data, 1(1):1–7, 2014. 7

[37] Daniil Polykovskiy, Alexander Zhebrak, Benjamin Sanchez-Lengeling, Sergey Golovanov, Oktai Tatanov, Stanislav Belyaev, Rauf Kurbanov, Aleksey Artamonov, Vladimir Aladinskiy, Mark Veselov, et al. Molecular sets (moses): a benchmarking platform for molecular generation models. Frontiers in pharmacology, 11:565644, 2020. 7, 24, 25

[38] Emmanuel Noutahi, Cristian Gabellini, Michael Craig, Jonathan SC Lim, and Prudencio Tossou. Gotta be safe: a new framework for molecular design. Digital Discovery, 3(4):796–804, 2024. 7, 26, 29

[39] Vijay Prakash Dwivedi, Ladislav Rampášek, Michael Galkin, Ali Parviz, Guy Wolf, Anh Tuan Luu, and Dominique Beaini. Long range graph benchmark. Advances in Neural Information Processing Systems, 35:22326–22340, 2022. 7

[40] Wengong Jin, Regina Barzilay, and Tommi Jaakkola. Junction tree variational autoencoder for molecular graph generation. In International conference on machine learning, pages 2323–2332. PMLR, 2018. 8, 22, 24, 25

[41] Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di He, Yanming Shen, and Tie-Yan Liu. Do transformers really perform badly for graph representation? Advances in neural information processing systems, 34:28877–28888, 2021. 9

[42] Eric Wang, Samuel Schmidgall, Paul F Jaeger, Fan Zhang, Rory Pilgrim, Yossi Matias, Joelle Barral, David Fleet, and Shekoofeh Azizi. Txgemma: Efficient and agentic llms for therapeutics. arXiv preprint arXiv:2504.06196, 2025. 9

[43] Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. Deep contextualized word representations. In Marilyn Walker, Heng Ji, and Amanda Stent, editors, Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 2227–2237, New Orleans, Louisiana, June 2018. Association for Computational Linguistics. doi: 10.18653/v1/N18-1202. URL https://aclanthology. org/N18-1202/. 9

[44] Ian Tenney, Dipanjan Das, and Ellie Pavlick. Bert rediscovers the classical nlp pipeline. In Proceedings of the 57th annual meeting of the association for computational linguistics, pages 4593–4601, 2019. 9

[45] Sanjeev Arora, Nadav Cohen, and Elad Hazan. On the optimization of deep networks: Implicit acceleration by overparameterization. In International conference on machine learning, pages 244–253. PMLR, 2018. 9

[46] Lorenz Richter, Ayman Boustati, Nikolas Nüsken, Francisco Ruiz, and Omer Deniz Akyildiz. Vargrad: a low-variance gradient estimator for variational inference. Advances in Neural Information Processing Systems, 33:13481–13492, 2020. 19

[47] David W Zhang, Corrado Rainone, Markus Peschl, and Roberto Bondesan. Robust scheduling with gflownets. arXiv preprint arXiv:2302.05446, 2023. 19

[48] Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in neural information processing systems, 32, 2019. 19

[49] Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020. 19

[50] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. 19

[51] Jaehyeong Jo, Seul Lee, and Sung Ju Hwang. Score-based generative modeling of graphs via the system of stochastic differential equations. In International conference on machine learning, pages 10362–10383. PMLR, 2022. 19, 20

[52] Ida Schomburg, Antje Chang, Christian Ebeling, Marion Gremse, Christian Heldt, Gregor Huhn, and Dietmar Schomburg. Brenda, the enzyme database: updates and major new developments. Nucleic acids research, 32(suppl\_1):D431–D433, 2004. 20

[53] Prithviraj Sen, Galileo Namata, Mustafa Bilgic, Lise Getoor, Brian Galligher, and Tina Eliassi-Rad. Collective classification in network data. AI magazine, 29(3):93–93, 2008. 20

[54] Martin Simonovsky and Nikos Komodakis. Graphvae: Towards generation of small graphs using variational autoencoders. In International conference on artificial neural networks, pages 412–422. Springer, 2018. 20

[55] Renjie Liao, Yujia Li, Yang Song, Shenlong Wang, Will Hamilton, David K Duvenaud, Raquel Urtasun, and Richard Zemel. Efficient graph generation with graph recurrent attention networks. Advances in neural information processing systems, 32, 2019. 20

[56] Chenhao Niu, Yang Song, Jiaming Song, Shengjia Zhao, Aditya Grover, and Stefano Ermon. Permutation invariant graph generation via score-based generative modeling. In International conference on artificial intelligence and statistics, pages 4474–4484. PMLR, 2020. 20

[57] Nikhil Goyal, Harsh Vardhan Jain, and Sayan Ranu. Graphgen: A scalable approach to domain-agnostic labeled graph generation. In Proceedings ofthe web conference 2020, pages 1253–1263, 2020. 20

[58] Davide Bacciu and Marco Podda. Graphgen-redux: a fast and lightweight recurrent model for labeled graph generation. In 2021 International Joint Conference on Neural Networks (IJCNN), pages 1–8. IEEE, 2021. 20

[59] Hanjun Dai, Azade Nazi, Yujia Li, Bo Dai, and Dale Schuurmans. Scalable deep generative modeling for sparse graphs. In International conference on machine learning, pages 2302–2312. PMLR, 2020. 20

[60] Clément Vignac, Igor Krawczuk, Antoine Siraudin, Bohan Wang, Volkan Cevher, and Pascal Frossard. Digress: Discrete denoising diffusion for graph generation. In Proceedings of the 11th International Conference on Learning Representations, 2023. 20

[61] Tianze Luo, Zhanfeng Mo, and Sinno Jialin Pan. Fast graph generative model via spectral diffusion. arXiv preprint arXiv:2211.08892, 3(4):5, 2022. 20

[62] Yunhui Jang, Dongwoo Kim, and Sungsoo Ahn. Graph generation with k<sup>2</sup>-trees. In International Conference on Learning Representations, volume 2024, pages 27621–27644, 2024. 20

[63] Jorg Degen, Christof Wegscheid-Gerlach, Andrea Zaliani, and Matthias Rarey. On the art of compiling and using ‘drug-like’ chemical fragment spaces. ChemMedChem, 3(10):1503, 2008. 20

[64] Josep Arús-Pous, Simon Viet Johansson, Oleksii Prykhodko, Esben Jannik Bjerrum, Christian Tyrchan, Jean-Louis Reymond, Hongming Chen, and Ola Engkvist. Randomized smiles strings improve the quality of molecular generative models. Journal of cheminformatics, 11(1):71, 2019. 20

[65] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. 20, 22

[66] Shuo Zhang, Yang Liu, and Lei Xie. Molecular mechanics-driven graph neural network with multiplex graph for molecular structures. arXiv preprint arXiv:2011.07457, 2020. 21

[67] Tairan Liu, Misagh Naderi, Chris Alvin, Supratik Mukhopadhyay, and Michal Brylinski. Break down in order to build up: decomposing small molecules for fragment-based drug design with e molfrag. Journal ofchemical information and modeling, 57(4):627–631, 2017. 24, 25

[68] Marwin HS Segler, Thierry Kogej, Christian Tyrchan, and Mark P Waller. Generating focused molecule libraries for drug discovery with recurrent neural networks. ACS central science, 4(1): 120–131, 2018. 24, 25, 29

[69] Oleksii Prykhodko, Simon Viet Johansson, Panagiotis-Christos Kotsias, Josep Arús-Pous, Esben Jannik Bjerrum, Ola Engkvist, and Hongming Chen. A de novo molecular generation method using latent vector based generative adversarial network. Journal ofcheminformatics, 11(1):74, 2019. 24, 25

[70] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 24

[71] Tuomas Kynkäänniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved precision and recall metric for assessing generative models. Advances in neural information processing systems, 32, 2019. 29

[72] Michael A Skinnider. Invalid smiles are beneficial rather than detrimental to chemical language models. Nature Machine Intelligence, 6(4):437–448, 2024. 29

## A Additional details on SNACK

## A.1 Definition

In Section 2.1, we described SNACK through the encoding of an adjacency matrix. Here, we present an independent formal definition of the SNACK representation, starting with its primary structural unit: the block.

Definition (Block). A block consists ofa single node token preceded by zero or more distance tokens. The distance tokens are integer-valued and must be arranged in strictly descending order. Each distance token is immediately followed by its corresponding edge-type token.

To illustrate, consider the SNACK sequence [C]1-[C]2-1-[C]. According to the definition, this sequence is partitioned into three distinct blocks: [C], 1-[C], and $2 - 1 - [ \mathsf { C } ]$ . This block-based structure ensures that every node carries its own connectivity information to the previously generated subgraph.

Definition (SNACK). A SNACK string is defined as a concatenation of N blocks, denoted as $[ B _ { 1 } , B _ { 2 } , \ldots , B _ { N } ]$ . For each $i \in \{ 1 , \ldots , N \}$ , the i-th block $B _ { i }$ contains zero or more distinct distance tokens drawnfrom $\{ 1 , \ldots , \bar { i } - 1 \}$ , arranged in strictly descending order.

Distance tokens in i-th block represent connections to the preceding i − 1 nodes in the sequence.

## A.2 Valid masking rule

In this subsection, we describe the rules for valid SNACK strings. For simplicity, we focus on SNACK strings that are valid under connected search ordering, in which each newly generated node must be connected to at least one previously generated node. This ordering disallows two cases. (1) Disconnected graphs: SNACK represents such graphs by concatenating the strings of each component. For example, [C][C]1-[C] denotes a disconnected graph with two components, [C] and [C]1-[C]. (2) Uniform ordering: under uniform ordering, a three-node graph can be represented as [C][C]2-1-[C], which corresponds to first generating two disconnected nodes and then connecting them via a third node. In both cases, a node-type token may follow another node-type token. In contrast, in orderings such as BFS, nodes are generated by expanding from previously visited nodes. As a result, a node-type token cannot directly follow another node-type token, allowing only sequences such as [C]1-[C]1-[C] or [C]1-[C]2-[C] for a three-node graph.

Below, we describe the rules for SNACK strings under this connected search ordering. To allow disconnected graphs, node-type tokens are permitted to follow another node-type token directly. BFS ordering can be further enforced using the rules highlighted in green, which track the expansion node, i.e., the node currently being processed.

• Initialize: The first token (except for [bos]) is always a node-type token; initialize the number of nodes as $N \gets 0$ . Initialize a pointer $P \gets 1$ to track the current expansion node.

• Node type: After a node-type token, either a distance token or [eos] follows. Increment $N \gets N + 1$ . Distance tokens take values in $\{ 1 , \ldots , N { - } ( P - 1 ) \}$ }. If N reaches the maximum number of nodes, only the [eos] token is allowed.

• Distance: After a distance token, an edge-type token follows. Denote the current distance by D. If this is the first distance token in the current block, set $P  N - ( D - 1 )$ ).

• Edge type: An edge-type token can be followed by either a node-type token or a distance token. Distance tokens take values in $\{ 1 , \ldots , D - 1 \}$ to enforce a descending order. If $D = 1$ , no further distance tokens are allowed, and only node-type tokens may follow.

Example Under BFS ordering, the possible next tokens after the sequence [C]1-[C]2-1-[C] are {1, 2, 3, [eos]}. The maximum distance token is 3 (i.e., the previous distance token 2 plus 1).

## A.3 The number of distance tokens

The number of distance tokens (i.e., the maximum bandwidth) can be estimated from the training data. However, the model may generate sequences that use the maximum distance token, for which larger

distances could arise when reordering node IDs. To address this, we allocate additional distance tokens beyond the observed maximum. Because the maximum distance is bounded by the total number of nodes, this expansion remains within a strictly finite range.

## A.4 Bijectivity with the adjacency matrix

In this section, we provide a formal proof of Proposition 1.

Proposition 1 The SNACK representation is bijective with respect to the adjacency matrix.

Proof. Let A be the space of adjacency matrices and X be the space of valid SNACK sequences. To establish bijectivity, we show that both the encoding function $f : { \mathcal { A } }  { \mathcal { X } }$ and the decoding function $g : \mathcal { X }  \mathcal { A }$ are injective.

The injectivity of f is evident from its construction: the deterministic row-by-row traversal of the lower triangle ensures that distinct matrices are mapped to distinct sequences. For the decoding function $^ { g , }$ consider any $\mathbf { x } \in { \mathcal { X } } . g ( \mathbf { x } )$ constructs an adjacency matrix $\mathbf { A } \in { \mathcal { A } }$ as follows:

1. x is uniquely partitioned into N blocks $[ B _ { 1 } , \ldots , B _ { N } ]$ based on node tokens.

2. For each block $B _ { k }$ , the node token defines the diagonal entry $\mathbf { A } _ { k , k }$

3. For each distance-edge pair $( d _ { j } , e _ { j } )$ within $B _ { k }$ , we set the off-diagonal entries $\mathbf { A } _ { k , k - d _ { j } } =$ $\mathbf { A } _ { k - d _ { j } , k } = e _ { j } .$

By definition of SNACK, each distance $d _ { j }$ satisfies $1 \leq d _ { j } < k ,$ which guarantees that the index $k - d _ { j }$ is a valid preceding node index $( 1 \leq k - \ ' d _ { j } < k )$ . Since each node token and each distance–edge pair in x corresponds to a unique entry in the lower triangle of A, and the descending order of distance tokens ensures a deterministic mapping without overlap, any two distinct sequences x $\neq { \mathbf { x } } ^ { \prime }$ will result in at least one differing entry in their reconstructed matrices $g ( \mathbf { x } ) \neq g ( \mathbf { x } ^ { \prime } )$ . This establishes the injectivity of $g .$ □

As a final note, we emphasize that ambiguous tokens may break the bijectivity of SNACK. For example, if both [O] and [OH2] are processed to represent the same molecule $\mathrm { ( H _ { 2 } O ) }$ , the one-to-one mapping between sequences and graphs is lost. We evaluate this variant in Appendix G.2.3.

## B Computing random BFS ordering

Throughout our GFlowNet experiments, we employ a fixed probability distribution for the random BFS ordering, denoted as $q ^ { \mathrm { o r } \hat { \mathbf { d } } } ( \pi | G )$ . The procedure for computing this probability is detailed in Algorithm 1. Note that the resulting probability is independent of any specific node indexing and depends solely on the graph structure. Thus, node orderings belonging to the same automorphism group yield identical probability outputs.

Algorithm 1 Computation of log-probability of random BFS ordering   
Require: Graph $G = ( V , E )$ , BFS ordering $\pi = ( v _ { 1 } , \ldots , v _ { n } )$   
Ensure: log q(π|G)   
1: Queue ← [v ], i ← 2, P ← n ▷ n choices for the BFS root   
2: while Queue is not empty do   
3: u ← POPFRONT(Queue)   
4: $C \gets \{ w \in N e i g h \bar { b } o r ( u ) : w \notin \{ v _ { 1 } , \dots , v _ { i - 1 } \} \}$ ▷ set of newly discovered neighbors   
5: P ← P · |C|!   
6: append $v _ { i } , \ldots , v _ { i + | C | - 1 }$ to Queue ▷ the next |C| nodes in the given BFS order   
7: i ← i + |C|   
8: end while   
9: return − log P

## C Computational complexity of trajectory evaluation

We compare the computational cost of evaluating forward-policy probabilities along a complete construction trajectory. This setting arises when computing trajectory-level training objectives after the trajectory has been collected. Let N and M denote the numbers of nodes and edges in the terminal graph, respectively.

Sequence-based evaluation. Let $\mathbf { x } _ { \le L } = ( x _ { 1 } , \dots , x _ { L } )$ be a complete SNACK representation. Its length is $L = N + 2 M + O ( 1 )$ , where the constant term accounts for special tokens. A causal Transformer jointly evaluates $p _ { \theta } ( x _ { \ell } | \mathbf { x } _ { < \ell } )$ for all $\ell = 1 , \ldots , L$ in a single forward pass: causal masking ensures that each prediction depends only on the corresponding preceding prefix [26]. Thus, the intermediate prefixes need not be encoded separately. Under standard dense self-attention, the resulting arithmetic complexity is

$$
C _ { \mathrm { S N A C K } } = \Theta ( L ^ { 2 } ) = \Theta \big ( ( N + M ) ^ { 2 } \big ) .\tag{6}
$$

Graph-based evaluation. Consider a graph construction trajectory $( G _ { 0 } , \dots , G _ { T } )$ that starts from a fixed-size initial graph and adds one node or one edge at each transition. Its length is $T = \Theta ( N + M )$ Write $n _ { t } = | V ( G _ { t } )$ | and $m _ { t } = | E ( G _ { t } )$ | for the size of each intermediate graph. A graph-based policy that recomputes its representations at each state requires separate encodings of $G _ { 0 } , \ldots , G _ { T - 1 }$ . For a graph Transformer with dense attention over all node pairs, evaluating $G _ { t }$ costs $\Theta ( n _ { t } ^ { 2 } )$ , giving

$$
C _ { \mathrm { g r a p h } } = \Theta \left( \sum _ { t = 0 } ^ { T - 1 } n _ { t } ^ { 2 } \right) = O ( T N ^ { 2 } ) = O \big ( ( N + M ) N ^ { 2 } \big ) .\tag{7}
$$

Once the trajectory is available, the intermediate graphs can be processed in batches. Such batching permits parallel execution but does not eliminate their separate encodings or the associated arithmetic work. Consequently, $T$ counts state-specific evaluations, not necessarily sequential model calls.

Comparison under edge sparsity. Under the assumption $M = O ( N )$ , both construction lengths are $\bar { \Theta ( N ) }$ . Equations 6 and 7 then yield quadratic and cubic costs, respectively. The cubic bound is tight for the dense-attention graph policy: adding nodes individually from a fixed-size initial graph entails $\begin{array} { r } { \sum _ { t = 0 } ^ { T - 1 } n _ { t } ^ { 2 } = \Omega ( N ^ { 3 } ) } \end{array}$ . Therefore,

$$
C _ { \mathrm { S N A C K } } = \Theta ( N ^ { 2 } ) , \qquad C _ { \mathrm { g r a p h } } = \Theta ( N ^ { 3 } ) .\tag{8}
$$

## D Applications of graph-to-sequence transition probability

As alluded to in the main text, the ability to compute $q ( \mathbf { x } | G )$ offers several utilities, including: (1) training GFlowNets, (2) substructure-conditioned optimization, (3) learning $q ,$ and (4) computing the marginal log-likelihood. We already discussed GFlowNet training in Section 2.6. This section elaborates on the remaining applications.

Substructure-conditioned optimization The task is to train a model that optimizes a given molecule by expanding upon its structure. Let $G _ { 0 }$ be the initial molecular graph. A sequence model optimizes $\dot { G } _ { 0 }$ by first converting it into a string representation, $\mathbf { x } _ { 0 } ,$ , and then continuing the generation process. In many reinforcement learning algorithms, computing the action probability is essential. While the sequence model typically treats $\mathbf { x } _ { \mathrm { 0 } }$ as a fixed starting point, the conversion $G _ { 0 } \to \mathbf { x } _ { 0 }$ is actually governed by the underlying graph-to-sequence dynamics. The methodology for this computation is detailed in Section $2 . 6 ,$ and we further discuss its application within the GFlowNet framework in Appendix E.

Variational inference While we employ a fixed random BFS for $q ( \mathbf { x } | G )$ throughout this work, we observe that the optimal ordering strategy can vary depending on the dataset, as discussed in Appendix H. Therefore, performance could be further enhanced by jointly optimizing $q ( \mathbf { x } | G )$ and $p _ { \theta }$ via Equation 2, following the approach of Chen et al. [7].

Approximating marginal log-likelihood The ability to compute the marginal log-likelihood provides several critical advantages for model evaluation and selection. However, exactly computing the log-likelihood is intractable, as it requires marginalizing over all possible node orderings. Instead, we can approximate $p _ { \theta } ( G )$ using an importance sampling approach, similar to Kim et al. [8]:

$$
p _ { \theta } ( G ) = \sum _ { \mathbf { x } \in \mathcal { X } _ { G } } p _ { \theta } ( \mathbf { x } ) = \mathbb { E } _ { \mathbf { x } \sim q ( \mathbf { x } | G ) } \left[ \frac { p _ { \theta } ( \mathbf { x } ) } { q ( \mathbf { x } | G ) } \right] \approx \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \frac { p _ { \theta } ( \mathbf { x } _ { i } ) } { q ( \mathbf { x } _ { i } | G ) }\tag{9}
$$

where $\mathbf { x } _ { i }$ are sequences sampled from the variational distribution $q ( \mathbf { x } | G )$

## E Training Objectives for Conditioned GFlowNets

We present a framework for training GFlowNets in a conditioned setting, where the model generates molecules by expanding upon a predefined scaffold. This approach directly addresses lead optimization, a pivotal task in drug discovery.

Formally, given an initial graph $G _ { 0 }$ , the objective is to train the model such that the conditional probability $p ( G | G _ { 0 } )$ is proportional to the reward function $R ( G | G _ { 0 } )$ . Here, $R ( G | G _ { 0 } )$ is defined to assign zero reward to any graph $G$ that is unreachable from $G _ { 0 } ,$ , effectively constraining the search space to valid expansions of the scaffold.

We decompose the conditional probability $p ( G | G _ { 0 } )$ into a three-stage transition: $G _ { 0 } \to \mathbf { x } _ { 0 } , \mathbf { x } _ { 0 } \to \mathbf { x }$ and $\mathbf { x }  G$ . The sequence model $p _ { \theta } ( \mathbf { x } | \mathbf { x } _ { 0 } )$ handles the intermediate transitions $\mathbf { x } _ { 0 } \to \mathbf { x } .$ , while the terminating action $\mathbf { x }  G$ is deterministic. To complete the path, we must account for the initial transition $G _ { 0 } \to \mathbf { x } _ { 0 }$ . To this end, we introduce an initial policy $p _ { 0 } ( \mathbf { x } _ { 0 } | G _ { 0 } )$ that assigns probabilities to each $\operatorname { S N A C K }$ representation corresponding to $G _ { 0 }$ . Analogous to the variational distribution $q ( \mathbf { x } | G )$ the initial policy $p _ { 0 } ( \mathbf { x } _ { 0 } | G _ { 0 } )$ is computed as:

$$
p _ { 0 } ( \mathbf { x } _ { 0 } | G _ { 0 } ) = | \mathrm { A u t } ( \mathbf { A } _ { 0 } ) | \cdot p _ { 0 } ^ { \mathrm { o r d } } ( \pi | G _ { 0 } )\tag{10}
$$

for any $\pi \in \mathrm { A u t } ( \mathbf { A } _ { 0 } )$ , where ${ \bf A } _ { 0 }$ is the adjacency matrix encoded by $\mathbf { x } _ { \mathrm { 0 } }$ , and $p _ { 0 } ^ { \mathrm { o r d } }$ denotes the ordering policy for the initial step. Incorporating the initial state distribution $\mu ( G _ { 0 } )$ , the conditioned Trajectory Balance (TB) objective is formulated as:

$$
\big [ \log Z + \log \mu ( G _ { 0 } ) + \log p _ { 0 } ( \mathbf { x } _ { 0 } | G _ { 0 } ) + \log p _ { \theta } ( \mathbf { x } | \mathbf { x } _ { 0 } ) - \log q ( \mathbf { x } | G ) - \log R ( G | G _ { 0 } ) \big ] ^ { 2 } .\tag{11}
$$

In our experiments, the initial states are sampled from a set of 800 molecules under a uniform distribution; thus, log $\mu ( G _ { 0 } )$ remains constant and can be omitted during optimization. We also set $p _ { 0 }$ to be the fixed random BFS in the experiment for convenience; however, the initial policy $p _ { 0 }$ must be learned to ensure that the marginal property $\begin{array} { r } { p ( G | G _ { 0 } ) = \sum _ { \mathbf { x } \in \mathcal { X } _ { G } } p _ { \theta } ( \mathbf { x } | \mathbf { x } _ { 0 } ) p _ { 0 } ( \mathbf { x } _ { 0 } | G _ { 0 } ) \propto \bar { R } ( G | G _ { 0 } ) } \end{array}$ is strictly satisfied. Finally, in our experiment, we employed a VarGrad variant [46, 47] that estimates log $Z$ for each mini-batch.

## F Experimental details

## F.1 Model architecture

For simplicity, we use a unified model architecture across all experiments. Specifically, we train Transformer models [26] following the LLaMA architecture [35], with RMSNorm pre-normalization [48], SwiGLU activations [49], and rotary positional embeddings [50]. For distribution learning tasks, our model has 6 layers, an embedding dimension of 384, and 6 attention heads, resulting in approximately 8.9M parameters. The exact parameter count varies slightly with the tokenizer, as the vocabulary size determines the output projection layer. For goal-directed learning, we use a 4-layer variant.

## F.2 Generic graph generation

Dataset We evaluate on eight widely used graph generation benchmarks used in [3, 51, 6, 16]: Planar (synthetic planar graphs; 128/32/40), Lobster (tree graphs; 69/20/11), Enzymes (protein tertiary structures from the BRENDA database [52]; 410/59/118), SBM (stochastic block model graphs; 128/32/40), Ego (3-hop ego-networks from the Citeseer dataset [53]; 529/76/152), and Grid (2D lattice graphs; 70/10/20); community-small (70/10/20); Ego-small (140/20/40), where the numbers denote the train/validation/test splits. We follow the same splits as in Jang et al. [6].

Baseline We report baseline results obtained from Jang et al. [6], including GraphVAE [54], GraphRNN [3], GRAN [55], EDP-GNN [56], GraphGen [57], GraphGen-Redux [58], GraphAF [12], GraphDF [13], BiGG [59], GDSS [51], DiGress [60], GDSM [61], BwR [10], HGGT [62], and GEEL [6]. Training statistics were recomputed to derive the standard deviation. Due to the resulting changes in compatibility thresholds, the specific methods marked in bold differ from the original reports by Jang et al. [6].

Tokenization Given that generic graphs lack edge attributes, we omit edge-type tokens from the SNACK representation. To accommodate graphs with multiple components, we employ a disconnected BFS masking scheme (refer to Appendix A.2).

Training We train all models using random BFS ordering with a batch size of 16 for up to 1M iterations, evaluating every 10k steps. We use a learning rate of 1 × 10<sup>−4</sup> with weight decay of 0.01 and apply gradient clipping with a maximum norm of 5.0. Early stopping is employed with a patience of 20 based on validation performance, and in practice training typically converges and stops around 200k iterations. Generic graph experiments are repeated with five different random seeds, using a single GPU (NVIDIA GeForce RTX 3090) and 8 CPU cores.

## F.3 Molecular string generation

Tokenization For SMILES and SAFE, we adopt character-level tokenization. In the case of SAFE, molecules are first fragmented using the BRICS algorithm [63]; if fragmentation fails, we fall back to the SMILES representation. For SNACK, STGG, GEEL, and G2PT, we remove aromatic bonds and consider only single, double, and triple bond types. Node types include atom types and charges (e.g., [N+]), and hydrogens are added implicitly after generation.

All tokenizers include special tokens: [bos], [eos], [pad], and [unk]. The unknown token [unk] is introduced because the vocabulary is constructed from the training data using a single representation per molecule. The [bos], [pad], and [unk] tokens are used only during training and are masked out at inference time for all tokenizer types (including, e.g., SMILES).

For STGG, GEEL, G2PT and SNACK, we use the following node-type tokens:

• QM9: [C-], [C], [F], [N+], [N-], [N], [O-], [O]

• QM9 (with hydrogens): [C-], [CH-], [CH2], [CH3], [CH], [C], [F], [N+], [N-], [NH+], [NH2+], [NH2], [NH3+], [NH3], [NH], [N], [O-], [OH2], [OH], [O]

• MOSES: [Br], [C], [Cl], [F], [N], [O], [S]

• MOSES (with hydrogens): [Br], [CH2], [CH3], [CH], [C], [Cl], [F], [NH2], [NH], [N], [OH], [O], [S]

In accordance with previous graph generation literature [2], molecules were Kekulized to use explicit single, double, and triple bonds instead of aromatic bonds. Generated molecules were subsequently sanitized.

Baseline implementation For GEEL, G2PT, STGG, and SNACK, we use the same set of node-type tokens and add hydrogen atoms implicitly after generation. GEEL and STGG follow their original token masking strategies (grammar masking for GEEL and grammar+valence masking for STGG). For STGG, we use the code provided in their supplementary material, while the other methods are implemented based on the descriptions in their respective papers. For all methods, training sequences are randomly sampled from valid enumerations, as this randomization has been shown to enhance model performance [64].

Training We use a batch size of 128, apply no weight decay, optimize with the Adam optimizer [65], and clip gradients at 5.0 for all datasets. For the QM9 dataset, we train for 200 epochs with a learning rate of 0.0001 and select the best model based on validation loss. For the MOSES dataset, we train for up to 100 epochs with a learning rate of 0.001 and apply early stopping if no improvement in validation loss is observed for 5 consecutive epochs. From the selected model, 30k molecules are sampled for evaluation. Experiments are repeated with three different random seeds. Molecular experiments are conducted using a single GPU (NVIDIA GeForce RTX 3090) and 8 CPU cores. In one of the training runs, the SNACK-Transformer took approximately 10 hours and stopped after 67 epochs on the MOSES dataset. The validation loss is computed on 10k molecules, and evaluation metrics are computed on a test set of 30k molecules.

Conditioned generation We evaluated substructure-conditioned generation using 129 molecules with at most 13 heavy atoms from the standard MOSES test split. For each molecule, we sampled 100 randomized representations with replacement and generated one continuation per representation, resulting in 12,900 attempts per checkpoint. Each complete molecular representation was supplied as a prefix with only the [eos] token removed, and immediate [eos] generation was permitted. We evaluated three pretrained models per method without additional fine-tuning. An extension was successful if the output was valid, connected, contained the conditioning substructure, and had strictly more heavy atoms.

## F.4 Goal-directed learning

We evaluate SNACK on two complementary molecular generation tasks to assess its performance across different scales and reward landscapes.

Tokenization For the HOMO-LUMO gap task, we use node-type tokens obtained from QM9. For two similarity-based tasks, we use the following node-type tokens, derived from ZINC250k: [Br], [C-], [C], [Cl], [F], [I], [N+], [N-], [N], [O+], [O-], [O], [P+], [P], [S+], [S-], [S]. For the conditioned optimization task, we utilize the following node-type tokens, derived from the fragments used for conditioning: [Br], [C], [Cl], [F], [I], [N+], [N-], [N], [O+], [O-], [O], [S-], [S].

HOMO–LUMO Gap Optimization Following Bengio et al. [9] and Jain et al. [18], we use a small molecule generation environment based on the QM9 dataset. Molecules are generated atom-by-atom up to a maximum of 9 heavy atoms. The reward for a completed molecule is the HOMO–LUMO gap predicted by a pre-trained MXMNet model [66].

Similarity-based Optimization To evaluate the scalability of SNACK, we conduct similaritybased optimization for larger drug-like molecules. Specifically, we target the generation of molecules similar to Aripiprazole (30 heavy atoms) and Albuterol (17 heavy atoms). The reward is defined as the Tanimoto similarity between the generated molecule and the target structure, computed using RDKit fingerprints. These tasks are designed to probe the model’s capability as the action space and trajectory length increase significantly compared to the QM9 environment.

Implementation For the graph-based GFlowNet baseline, we use the standard open-source implementation.<sup>2</sup> We adopt its default hyperparameters, which are also used in MO-GFN [18]: learning rate $1 \times 1 0 ^ { - 4 }$ , gradient clipping 10, 4 layers, uniform policy mix 0.001, and reward exponent 32. With this configuration, the baseline collapses on the albuterol task: the unique fraction drops to 0.001 and the generator converges to repeated empty graphs rather than diverse molecules. We therefore performed an additional hyperparameter search on the two similarity tasks and adopted the best non-collapsing setting for the baseline, namely uniform policy mix 0.1, gradient clipping 1.0, and reward exponent 4; the HOMO–LUMO gap task uses the default values. For SNACK, we use a learning rate $1 \times 1 0 ^ { - 4 }$ , uniform policy mix 0.1, gradient clipping 1.0, and reward exponent 32 across all tasks. SNACK employs the same transformer architecture as in our distribution learning experiments, with the number of layers reduced from 6 to 4 to align with the graph-based baseline. To ensure a fair comparison, both models use the identical set of atoms.

Training We match the optimizer, batch size, reward setting, and remaining hyperparameters to the standard graph-based GFlowNet as closely as possible. Each training step uses 32 on-policy trajectories and 32 replay-buffer trajectories, and all models are trained with the trajectory balance objective [17]. When computing backward probabilities, we apply the automorphism correction of Kim et al. [8], so that the trajectory balance objective is unbiased with respect to the underlying graph symmetry. Each model is trained on a single GPU (NVIDIA GeForce RTX 3090) and a single CPU core until training convergence. We then sample 10,000 molecules from the final model and report the mean and standard deviation over three random seeds.

For SNACK and SMILES, we use the same respective tokenization schemes as in molecular distribution learning. Since multiple SMILES strings can decode to the same molecule, SMILES does not provide a tractable backward distribution over strings given a molecular graph; we therefore approximate the backward term by setting $q ( \mathbf { x } | G ) = 1 \ [ 2 0 ]$ . Despite hyperparameter tuning, the SMILES-based GFlowNet fails to learn from scratch.

Evaluation metrics Evaluation metrics we presented in Table 4 are defined as follows:

• Avg. Reward: The average reward over all generated samples.

• Diversity: The average pairwise Tanimoto distance among valid generated samples, using Morgan fingerprints (radius 2, 2048 bits).

• Top 100 reward: The average reward of the 100 highest-reward samples.

• Top 100 diversity: Diversity among the top 100 reward molecules.

• Diverse Top 100 reward: The average reward of the top 100 molecules, ensuring that each pair has a Tanimoto similarity below 0.7.

• Unique fraction: The fraction of unique molecules among the valid generated samples.

Conditioned optimization Following previous studies [40, 2], we define the initial states using a set of 800 ZINC molecules, specifically selected for their low penalized logP scores. We train a 4-layer transformer by optimizing the conditioned TB objective, as explained in Appendix E. The maximum number of nodes for generation is fixed at 40, accounting for the maximum size of initial fragments (37 nodes). The model is trained using the AdamW optimizer with a learning rate of 0.0005 and a batch size of 128. The reward exponent is set to 256. Given that GFlowNet is an off-policy algorithm, we maintain a replay buffer that is updated with 128 new samples every 100 iterations. The model was trained for 30,000 iterations, and we evaluated the checkpoint at 28,000 iterations, which exhibited the highest average reward. For each of the 800 fragments, we generated 10 samples and selected the one with the highest property score improvement. Similarity is measured using Morgan fingerprints (radius 2) with the Tanimoto similarity coefficient. Finally, the success rate is defined as the percentage of molecules that were successfully optimized across the 800 starting fragments. In Table 5, we compare our model with JT-VAE [40], GCPN [2], and GraphDF [13], where baseline results are obtained from their respective papers (corresponding to $\delta = 0 )$

## F.5 Bilinear probing

Training We use 160k training examples from MOSES to train the probe parameters $\mathbf { W } _ { Q } , \mathbf { W } _ { K }$ and $\{ a ^ { \ell } \} _ { \ell = 0 } ^ { 6 } .$ Hidden representations are extracted from a trained 6-layer SNACK (V+B) model and used to train the probe via binary cross-entropy for adjacency prediction and mean squared error for shortest path distance (SPD) prediction. We use the Adam optimizer [65] with a learning rate of 0.001 and gradient clipping with a maximum norm of 1. For each sample, the target matrix is of size $N \times N$ , and we train only on its lower triangular part. The batch size is 8. Training is conducted on a single GPU (NVIDIA GeForce RTX 3090). At inference time, the full matrix is reconstructed by symmetrizing the predictions.

## G Additional experimental results

## G.1 Full results on generic graphs

Full results are presented in Tables 8 and 9, which report MMD scores for degree, clustering, and orbit statistics; for these metrics, lower values indicate superior performance. In the table, N.A. denotes that all generated samples are invalid, and “–” denotes unavailable or unreported results. SNACK (BFS) and SNACK (CM) refer to SNACK trained using BFS and CM node orderings, respectively.

Baseline results are obtained from Jang et al. [6]. To ensure a consistent comparison, we recomputed the training statistics to derive the standard deviation; results within one standard deviation of the training data are highlighted in bold. Note that the specific methods marked in bold may differ from the original reports due to these updated compatibility thresholds.

Table 8: Results for Ego-small, Community-small, Planar, and Lobster datasets.
<table><tr><td rowspan="3">Method</td><td colspan="3">Ego-small</td><td colspan="2">Community-small</td><td colspan="3">Planar</td><td colspan="3">Lobster</td></tr><tr><td colspan="3"> $4 \leq | V | \leq 1 8$ </td><td colspan="2"> $1 2 \leq | V | \leq 2 0$ </td><td colspan="3">|V| = 64</td><td colspan="3"> $1 0 \leq | V | \leq 1 0 0$ </td></tr><tr><td>Deg.</td><td>Clus.</td><td>Orb.</td><td>Deg. Clus.</td><td>Orb.</td><td>Deg.</td><td>Clus.</td><td>Orb.</td><td>Deg.</td><td>Clus.</td><td>Orb.</td></tr><tr><td>Training</td><td>0.021</td><td>0.031</td><td>0.012</td><td>0.016 0.032</td><td>0.003</td><td>0.001</td><td>0.004</td><td>0.000</td><td>0.007</td><td>0.000</td><td>0.005</td></tr><tr><td>GraphVAE</td><td>0.130</td><td>0.170</td><td>0.050</td><td>0.350 0.980</td><td>0.540</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GraphRNN</td><td>0.090</td><td>0.220</td><td>0.003</td><td>0.080 0.120</td><td>0.040</td><td>0.005</td><td>0.278</td><td>1.254</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td>GRAN</td><td>0.009</td><td>0.038</td><td>0.009</td><td>0.005 0.142</td><td>0.090</td><td>0.001</td><td>0.043</td><td>0.001</td><td>0.038</td><td>0.000</td><td>0.001</td></tr><tr><td>EDP-GNN</td><td>0.052</td><td>0.093</td><td>0.007</td><td>0.053 0.144</td><td>0.026</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GraphGen</td><td>0.085</td><td>0.102</td><td>0.425</td><td>0.075 0.065</td><td>0.014</td><td>1.762</td><td>1.423</td><td>1.640</td><td>0.548</td><td>0.040</td><td>0.247</td></tr><tr><td>GraphGen-Redux</td><td></td><td></td><td></td><td></td><td></td><td>1.105</td><td>1.809</td><td>0.517</td><td>1.189</td><td>1.859</td><td>0.885</td></tr><tr><td>GraphAF</td><td>0.030</td><td>0.110</td><td>0.001</td><td>0.180</td><td>0.200 0.020</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GraphDF</td><td>0.040</td><td>0.130</td><td>0.010</td><td>0.060</td><td>0.120 0.030</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BiGG</td><td>0.013</td><td>0.030 0.005</td><td></td><td>0.004 0.005</td><td>0.000</td><td>0.002</td><td>0.004</td><td>0.000</td><td>0.000</td><td>0.000 0.000</td><td></td></tr><tr><td>GDSS</td><td></td><td>0.021 0.024 0.007</td><td></td><td></td><td>0.045 0.086 0.007</td><td>0.250</td><td>0.393</td><td>0.587</td><td>0.117</td><td>0.002</td><td>0.149</td></tr><tr><td>DiGress</td><td>0.021</td><td>0.026</td><td>0.024</td><td>0.012</td><td>0.025 0.002</td><td>0.000</td><td>0.002</td><td>0.008</td><td>0.021</td><td>0.000</td><td>0.004</td></tr><tr><td>GDSM</td><td></td><td></td><td></td><td>0.011</td><td>0.015 0.001</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BwR</td><td></td><td></td><td></td><td></td><td></td><td>0.609</td><td>0.542</td><td>0.097</td><td>0.316</td><td>0.000</td><td>0.247</td></tr><tr><td>HGGT</td><td></td><td></td><td></td><td></td><td></td><td>0.000</td><td>0.001</td><td>0.000</td><td>0.003</td><td>0.000</td><td>0.015</td></tr><tr><td>GEEL</td><td>0.020</td><td>0.035</td><td>0.012</td><td>0.020</td><td>0.022 0.006</td><td>0.001</td><td>0.010</td><td>0.001</td><td>0.002</td><td>0.000</td><td>0.001</td></tr><tr><td>SNACK (BFS)</td><td>0.025</td><td>0.063</td><td>0.010</td><td>0.038</td><td>0.063 0.006</td><td>0.001</td><td>0.005</td><td>0.003</td><td>0.009</td><td>0.000</td><td>0.013</td></tr><tr><td>SNACK (CM)</td><td>0.023</td><td>0.049</td><td>0.013</td><td>0.013</td><td>0.035 0.003</td><td>0.001</td><td>0.031</td><td>0.012</td><td>0.005</td><td>0.000</td><td>0.009</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 9: Results for Enzymes, SBM, Ego, and Grid datasets.
<table><tr><td rowspan="4">Method</td><td colspan="3">Enzymes</td><td colspan="3">SBM</td><td colspan="3">Ego</td><td colspan="3">Grid</td></tr><tr><td colspan="3">10 ≤ |V| ≤ 125</td><td colspan="3"> $3 1 \leq | V | \leq 1 8 7$ </td><td colspan="3"> $5 0 \leq | V | \leq 3 9 9$ </td><td colspan="3"> $1 0 0 \leq | V | \leq 4 0 0$ </td></tr><tr><td>Deg.</td><td>Clus.</td><td>Orb.</td><td>Deg.</td><td>Clus.</td><td>Orb.</td><td>Deg.</td><td>Clus.</td><td>Orb.</td><td>Deg.</td><td>Clus.</td><td>Orb.</td></tr><tr><td>Training</td><td>0.010</td><td>0.019</td><td>0.009</td><td>0.015</td><td>0.001</td><td>0.037</td><td>0.010</td><td>0.003</td><td>0.017</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td>GraphVAE</td><td>1.369</td><td>0.629</td><td>0.191</td><td></td><td></td><td></td><td></td><td></td><td></td><td>1.619</td><td>0.000</td><td>0.919</td></tr><tr><td>GraphRNN</td><td>0.017</td><td>0.062</td><td>0.046</td><td>0.006</td><td>0.058</td><td>0.079</td><td>0.117</td><td>0.615</td><td>0.043</td><td>0.011</td><td>0.000</td><td>0.021</td></tr><tr><td>GRAN</td><td>0.023</td><td>0.031</td><td>0.169</td><td>0.011</td><td>0.055</td><td>0.054</td><td>0.026</td><td>0.342</td><td>0.254</td><td>0.001</td><td>0.004</td><td>0.002</td></tr><tr><td>EDP-GNN</td><td>0.023</td><td>0.268</td><td>0.082</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.455</td><td>0.238</td><td>0.328</td></tr><tr><td>GraphGen</td><td>0.146</td><td>0.079</td><td>0.054</td><td>1.230</td><td>1.752</td><td>0.597</td><td>0.578</td><td>1.199</td><td>0.776</td><td>1.550</td><td>0.017</td><td>0.860</td></tr><tr><td>GraphGen-Redux</td><td>0.456</td><td>0.035</td><td>0.251</td><td></td><td></td><td></td><td>1.088</td><td>0.702</td><td>0.155</td><td></td><td></td><td></td></tr><tr><td>GraphAF</td><td>1.669</td><td>1.283</td><td>0.266</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GraphDF</td><td>1.503</td><td>1.283</td><td>0.266</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BiGG</td><td>0.010</td><td>0.018</td><td>0.011</td><td>0.029</td><td>0.003</td><td>0.036</td><td>0.010</td><td>0.017</td><td>0.012</td><td>0.000</td><td>0.000</td><td>0.001</td></tr><tr><td>GDSS</td><td>0.026</td><td>0.061</td><td>0.009</td><td>0.496</td><td>0.456</td><td>0.717</td><td>0.393</td><td>0.873</td><td>0.209</td><td>0.111</td><td>0.005</td><td>0.070</td></tr><tr><td>DiGress</td><td>0.011</td><td>0.039</td><td>0.010</td><td>0.006</td><td>0.051</td><td>0.058</td><td>0.063</td><td>0.031</td><td>0.024</td><td>0.016</td><td>0.000</td><td>0.004</td></tr><tr><td>GDSM</td><td>0.013</td><td>0.088</td><td>0.010</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.002</td><td>0.000</td><td>0.000</td></tr><tr><td>BwR</td><td>0.021</td><td>0.095</td><td>0.025</td><td></td><td>N.A.</td><td></td><td></td><td>N.A.</td><td></td><td>0.385</td><td>1.187</td><td>0.083</td></tr><tr><td>HGGT</td><td>0.005</td><td>0.017</td><td>0.000</td><td>0.017</td><td>0.014</td><td>0.076</td><td></td><td>N.A.</td><td></td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td>GEEL</td><td>0.005</td><td>0.018</td><td>0.006</td><td>0.025</td><td>0.003</td><td>0.026</td><td>0.053</td><td>0.017</td><td>0.016</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td>SNACK (BFS)</td><td>0.003</td><td>0.013</td><td>0.001</td><td>0.018</td><td>0.013</td><td>0.039</td><td>0.071</td><td>0.122</td><td>0.034</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td>SNACK (CM)</td><td>0.006</td><td>0.023</td><td>0.005</td><td>0.134</td><td>0.168</td><td>0.069</td><td>0.050</td><td>0.256</td><td>0.019</td><td>0.000</td><td>0.000</td><td>0.000</td></tr></table>

## G.2 Additional results on molecular string generation

## G.2.1 Peptide generation

We additionally evaluated our model on the Peptides-func dataset, following Diamant et al. [10]. Peptides-func presents a more challenging distribution-learning benchmark, as its molecules average 150 atoms (compared to 22 in MOSES). We removed chirality information (corresponding to @ symbols in SMILES) because SNACK does not currently support chirality. We used a 10,000 / 2,000 / 3,371 train/validation/test split and trained models for 100 epochs with an early stopping patience of 20, keeping all other hyperparameters identical to those used in the MOSES experiments.

On this benchmark, SNACK demonstrates a clear advantage in validity metrics and outperforms existing baselines on FCD, while remaining competitive on SNN. Compared to SELFIES, the only other method guaranteeing 100% validity, SNACK achieves superior performance across all metrics. This likely stems from SNACK’s compact, unambiguous tokens, which reduce syntactic redundancies and help the model focus on structural patterns.

Table 10: Performance on Peptides-func dataset.
<table><tr><td>Method</td><td>Valid ↑</td><td>FCD↓</td><td>SNN↑</td><td>Frag ↑</td><td>Scaf ↑</td></tr><tr><td>SMILES</td><td> $0 . 8 2 0 \pm 0 . 0 2 7$ </td><td> $0 . 7 5 5 \pm 0 . 0 7 1$ </td><td> $\mathbf { 0 . 7 6 2 \pm 0 . 0 0 3 }$ </td><td> $0 . 9 9 9 \pm 0 . 0 0 0$ </td><td> $0 . 8 8 7 \pm 0 . 0 0 6$ </td></tr><tr><td>SELFIES</td><td> $\mathbf { 1 . 0 0 0 \pm 0 . 0 0 0 }$ </td><td> $0 . 4 4 4 \pm 0 . 0 4 2$ </td><td> $0 . 7 1 5 \pm 0 . 0 0 2$ </td><td> $\mathbf { 1 . 0 0 0 \pm 0 . 0 0 0 }$ </td><td> $0 . 8 8 7 \pm 0 . 0 0 3$ </td></tr><tr><td>GEEL</td><td> $0 . 7 8 1 \pm 0 . 0 2 8$ </td><td> $1 . 7 0 1 \pm 0 . 3 3 1$ </td><td> $0 . 6 8 9 \pm 0 . 0 1 0$ </td><td> $0 . 9 9 8 \pm 0 . 0 0 0$ </td><td> $\mathbf { 0 . 8 9 6 \pm 0 . 0 0 4 }$ </td></tr><tr><td>SNACK (None)</td><td> $0 . 9 3 8 \pm 0 . 0 1 3$ </td><td> $\mathbf { 0 . 4 2 2 \pm 0 . 0 6 4 }$ </td><td> $0 . 7 5 5 \pm 0 . 0 0 2$ </td><td> $0 . 9 9 9 \pm 0 . 0 0 0$ </td><td> $0 . 8 8 9 \pm 0 . 0 0 1$ </td></tr><tr><td>SNACK (V+B)</td><td> $\mathbf { 1 . 0 0 0 \pm 0 . 0 0 0 }$ </td><td> $\mathbf { 0 . 3 7 6 \pm 0 . 0 7 0 }$ </td><td> $0 . 7 5 5 \pm 0 . 0 0 5$ </td><td> $\mathbf { 1 . 0 0 0 \pm 0 . 0 0 0 }$ </td><td> $0 . 8 9 1 \pm 0 . 0 0 3$ </td></tr></table>

## G.2.2 MOSES benchmark

Table 11: First set of experimental results for the MOSES benchmark.
<table><tr><td>METHOD</td><td>VALID</td><td>UNIQUE1</td><td>UNIQUE2</td><td>INTDIV</td><td>INTDIV2</td><td>FILTERS</td><td>NOVEL</td></tr><tr><td>TRAINING DATASET</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>0.8567</td><td>0.8508</td><td>1.0000</td><td>0.0000</td></tr><tr><td>HMM [37]</td><td>0.0760</td><td>0.6230</td><td>0.5671</td><td>0.8466</td><td>0.8104</td><td>0.9024</td><td>0.9994</td></tr><tr><td>NGRAM [37]</td><td>0.2376</td><td>0.9740</td><td>0.9217</td><td>0.8738</td><td>0.8644</td><td>0.9582</td><td>0.9694</td></tr><tr><td>COMBINATORIAL [67]</td><td>1.0000</td><td>0.9983</td><td>0.9909</td><td>0.8732</td><td>0.8666</td><td>0.9557</td><td>0.9878</td></tr><tr><td>CHARRNN [68]</td><td>0.9748</td><td>1.0000</td><td>0.9994</td><td>0.8562</td><td>0.8503</td><td>0.9943</td><td>0.8419</td></tr><tr><td>AAE [37]</td><td>0.9368</td><td>1.0000</td><td>0.9994</td><td>0.8557</td><td>0.8499</td><td>0.9960</td><td>0.7931</td></tr><tr><td>VAE [37]</td><td>0.9767</td><td>1.0000</td><td>0.9984</td><td>0.8558</td><td>0.8498</td><td>0.9970</td><td>0.6949</td></tr><tr><td>JT-VAE [40]</td><td>1.0000</td><td>1.0000</td><td>0.9996</td><td>0.8551</td><td>0.8493</td><td>0.9760</td><td>0.9143</td></tr><tr><td>LATENTGAN [69]</td><td>0.8966</td><td>1.0000</td><td>0.9968</td><td>0.8565</td><td>0.8505</td><td>0.9735</td><td>0.9498</td></tr><tr><td>STGG [5]</td><td>1.0000</td><td>1.0000</td><td>0.9987</td><td>0.8556</td><td>0.8496</td><td>0.9976</td><td>0.6727</td></tr><tr><td>GEEL [6]</td><td>0.9999</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\mathbf { G } 2 \mathbf { P } \mathrm { T _ { l a r g e } }$  [16]</td><td>0.9720</td><td>1.0000</td><td></td><td></td><td></td><td>0.9890</td><td>0.7940</td></tr><tr><td>SNACK (6-layer)</td><td>1.0000</td><td>0.9997</td><td>0.9996</td><td>0.8568</td><td>0.8508</td><td>0.9964</td><td>0.7660</td></tr><tr><td>SNACK (12-layer)</td><td>1.0000</td><td>1.0000</td><td>0.9993</td><td>0.8571</td><td>0.8511</td><td>0.9975</td><td>0.7002</td></tr></table>

The models presented in the main text were kept smaller (6-layer, 384-dimensional, comparable to $\mathrm { G 2 P T _ { s m a l l } ) }$ to facilitate efficient experimentation. To compare our method with previous work, we followed the evaluation protocol described in the MOSES benchmark [37] more thoroughly, including the dataset split, the number of generated samples (30,000), and the number of independent runs (3).

We evaluated two model configurations: (1) a 63M-parameter, 6-layer, 1024-dimensional transformer with 8 heads, matching the capacity of STGG [5]; and (2) a 71M-parameter, 12-layer, 768-dimensional transformer with 12 heads, comparable to $\mathbf { G } 2 \mathbf { P } \mathbf { T } _ { \mathrm { m e d i u m } }$ [16]. All models were trained using the AdamW optimizer [70] with a learning rate of $1 \times 1 0 ^ { - 4 }$ and a weight decay of 0.01 for a maximum of 100 epochs, with an early-stopping patience of 5 epochs. We refer readers to Polykovskiy et al. [37] for the definitions of each evaluation metric.

Table 12: Second set of experimental results for the MOSES benchmark.
<table><tr><td rowspan="2">METHOD</td><td colspan="2">FCD (↓)</td><td colspan="2">SNN (↑)</td><td colspan="2">FRAG (↑)</td><td colspan="2">SCAF (↑)</td></tr><tr><td>TEST</td><td>TESTSF</td><td>TEST</td><td>TESTSF</td><td>TEST</td><td>TESTSF</td><td>TEST</td><td>TESTSF</td></tr><tr><td>TRAINING DATASET</td><td>0.0080</td><td>0.4755</td><td>0.6419</td><td>0.5859</td><td>1.0000</td><td>0.9986</td><td>0.9907</td><td>0.0000</td></tr><tr><td>HMM [37]</td><td>24.466</td><td>25.431</td><td>0.3876</td><td>0.3795</td><td>0.5754</td><td>0.5681</td><td>0.2065</td><td>0.0490</td></tr><tr><td>NGRAM [37]</td><td>5.5069</td><td>6.2306</td><td>0.5209</td><td>0.4997</td><td>0.9846</td><td>0.9815</td><td>0.5302</td><td>0.0977</td></tr><tr><td>COMBINATORIAL [67]</td><td>4.2375</td><td>4.5113</td><td>0.4514</td><td>0.4388</td><td>0.9912</td><td>0.9904</td><td>0.4445</td><td>0.0865</td></tr><tr><td>CHARRNN [68]</td><td>0.0732</td><td>0.5204</td><td>0.6015</td><td>0.5649</td><td>0.9998</td><td>0.9983</td><td>0.9242</td><td>0.1101</td></tr><tr><td>AAE [37]</td><td>0.5555</td><td>1.0572</td><td>0.6081</td><td>0.5677</td><td>0.9910</td><td>0.9905</td><td>0.9022</td><td>0.0789</td></tr><tr><td>VAE [37]</td><td>0.0990</td><td>0.5670</td><td>0.6257</td><td>0.5783</td><td>0.9994</td><td>0.9984</td><td>0.9386</td><td>0.0588</td></tr><tr><td>JT-VAE [40]</td><td>0.3954</td><td>0.9382</td><td>0.5477</td><td>0.5194</td><td>0.9965</td><td>0.9947</td><td>0.8964</td><td>0.1009</td></tr><tr><td>LATENTGAN [69]</td><td>0.2968</td><td>0.8281</td><td>0.5371</td><td>0.5132</td><td>0.9986</td><td>0.9972</td><td>0.8867</td><td>0.1072</td></tr><tr><td>STGG [5]</td><td>0.0680</td><td>0.5032</td><td>0.6359</td><td>0.5851</td><td>0.9998</td><td>0.9984</td><td>0.9416</td><td>0.0389</td></tr><tr><td>GEEL [6]</td><td>0.1603</td><td></td><td>0.6310</td><td></td><td>0.9996</td><td></td><td>0.8622</td><td></td></tr><tr><td> $\mathbf { G } 2 \mathbf { P } \mathbf { T } _ { \mathrm { l a r g e } } \left[ 1 6 \right]$ </td><td>1.0200</td><td></td><td>0.5500</td><td>一</td><td></td><td></td><td>0.0290</td><td>一</td></tr><tr><td>SNACK (6-layer)</td><td>0.0619</td><td>0.5050</td><td>0.6169</td><td>0.5731</td><td>1.0000</td><td>1.0000</td><td>0.9388</td><td>0.0798</td></tr><tr><td>SNACK (12-layer)</td><td>0.0611</td><td>0.5257</td><td>0.6274</td><td>0.5775</td><td>1.0000</td><td>1.0000</td><td>0.9432</td><td>0.0590</td></tr></table>

The detailed results are provided in Tables 11 and 12, with baseline results retrieved from Ahn et al. [5], Jang et al. [6], Chen et al. [16]. While some of these comparisons involve diverse architectures making a strict head-to-head evaluation difficult, the SNACK-Transformer combination effectively matches or exceeds the performance reported by existing baselines.

## G.2.3 Explicit hydrogen modeling

Explicitly including hydrogen atoms in node types (e.g., [N] and [NH]) can violate sequence bijectivity if distinct tokens map to the same molecule after implicit hydrogens are added to satisfy valency. Therefore, we adopted a representation using only single-atom node types. To evaluate the impact of this design choice on distribution learning, we compared our approach with an explicit hydrogen configuration. For SNACK (V+B), the explicit model yielded FCD scores of $0 . 0 4 0 \pm 0 . 0 0 2$ on QM9 and $0 . 1 4 4 \pm 0 . 0 3 1$ on MOSES, whereas our implicit model achieved $0 . 0 3 7 \pm 0 . 0 0 3$ and $0 . 1 0 3 \pm 0 . 0 0 2$ , respectively. These results demonstrate that implicit hydrogen modeling does not compromise, and may even enhance, generative performance.

## G.3 Scalability of GFlowNet training

We profile both SNACK and the graph-based GFlowNet on synthetic molecule batches as the number of atoms increases. Training measurements use batch size 8, while inference measurements use batch size 100. For each node size (N) and method, we measure (1) wall-clock time per training step, (2) wall-clock time per inference step, (3) peak CUDA memory during training, and (4) peak CUDA memory during inference. To isolate framework cost, weights are randomly initialized and forward-only generation uses no checkpoints. We use torch.cuda.synchronize() to fence timing measurements and torch.cuda.max\_memory\_allocated() for peak memory, with an extra warm up pass per size. Following [18], the graph-based GFlowNet uses a 2.2M-parameter graph transformer with 4 layers, embedding dimension 128, and 2 heads. SNACK uses a 2.6M-parameter, 4-layer transformer (embedding dimension 256, 4 heads), trained with full-sequence causal attention and decoded with KV-cached autoregressive generation. All measurements are run on a single GPU (NVIDIA GeForce RTX 3090, 24 GB VRAM) and a single CPU core. When a configuration triggers a CUDA OOM, we record the failure and exclude all larger sizes for that method. The corresponding curves for graph-based GFlowNet and SNACK are therefore truncated at the last size that fit in memory. The full results are shown in Figure 5a, Figure 5b, and Figure 5c, together with the maintext training-time result in Figure 2. Across both training and inference protocols, SNACK is faster and more memory-efficient, with the gap increasing as N grows. This trend is consistent with our complexity analysis.

![](images/bc8a2f0b54458f7427c5bbeb34c782748ff33bf061c34967d2494cec2a776f06.jpg)  
(a) Peak training memory.

![](images/cf83764e8e815061f72fca99eca5de10858b42e4ab563d3c72dc12d864c9c9f1.jpg)  
(b) Time per inference step

![](images/42408d158194d897a265a2e1456527433f9226ff9ad24ff06e4a349c213f3491.jpg)  
(c) Peak inference memory.  
Figure 5: Scalability of SNACK and the graph GFlowNet versus the number of atoms N.

## G.4 Substructure conditioned generation

The SNACK framework is exceptionally well-suited for conditional generation, as evidenced by its superior performance. To understand how this works in practice, let us look at a simple example where we generate a complete molecule starting from a partial fragment:

• SMILES (C(CC)C): Standard SMILES representations restrict the model to appending new atoms only to the very end of the sequence. This makes it impossible to attach functional groups directly to internal atoms. Additionally, valid prefixes are extremely limited—in this example, only two prefixes (C and C(CC)) are valid—which severely restricts the usable training data for substructure-conditioned generation during pretraining.

• SNACK ([C]1-[C]1-[C]3-[C]): In contrast, SNACK gives the model the flexibility to attach new functional groups (such as 3-[N]) to any previously generated atom within its bandwidth—not just the last one (e.g., directly targeting the second atom). Furthermore, every subsequence that ends with a node-type token acts as a fully valid prefix. In this case, it yields three valid prefixes ([C], [C]1-[C], and [C]1-[C]1-[C]), thereby maximizing the utility of the training data during pretraining.

![](images/068ada29f52a7248fc14a058cbc89085fc5097adbddd3c12f3eeaffb2e1ab702.jpg)  
Figure 6: Generated samples conditioned on Maribavir core.

We show example molecules generated from Maribavir core fragment in Figure 6. This fragment is extracted from the drug Maribavir, and used in Noutahi et al. [38] (SMILES: c1ccc2c(c1)ncn2[C@@H]1CCCO1). Utilizing a random BFS-based search, we linearize the given fragments into SNACK representations and proceed with generation from a pretrained SNACK (V+B) checkpoint. This showcases the versatility of the SNACK representation in navigating constrained chemical spaces.

## G.5 Model internals

Table 13: Reconstruction performance of the bilinear probe.
<table><tr><td>Target</td><td>Metric</td><td>SNACK</td><td>SMILES</td><td>Random</td></tr><tr><td rowspan="2">Adjacency</td><td>Accuracy ↑</td><td>0.9976</td><td>0.9894</td><td>0.9191</td></tr><tr><td>AUROC↑</td><td>0.9998</td><td>0.9970</td><td>0.9362</td></tr><tr><td rowspan="2">SPD</td><td>MSE↓</td><td>1.6659</td><td>1.2722</td><td>3.0564</td></tr><tr><td>MAE↓</td><td>0.9250</td><td>0.8148</td><td>1.2897</td></tr></table>

![](images/55354d4be53a814c74cc84115bd4efd6398c88fb3ed343906cc41352f27a31b2.jpg)  
Figure 7: Layer-wise adjacency matrices reconstructed from the hidden representations of the SNACK-Transformer.

Table 13 presents the per-molecule reconstruction performance of the bilinear probe on the MOSES dataset. The probe achieves high fidelity in recovering both local connectivity and global geometry, indicating that the learned representations capture essential structural information.

We evaluated the bilinear probe on both the trained SMILES model and an untrained baseline. The trained SMILES model achieves high accuracy in reconstructing the adjacency matrix, indicating that the next-token prediction task generally induces representations that encode graph structural patterns despite the linearized input. However, SNACK achieves higher accuracy on adjacency (edge) prediction, suggesting that SNACK captures more localized information than SMILES.

In contrast, the randomly initialized model reconstructs adjacency matrices that capture only coarse structural trends, yielding blurred probability distributions along the main diagonal—a trivial artifact of BFS ordering, where non-zero entries inherently cluster near the diagonal. Similarly, the recovered SPD matrices merely exhibit averaged patterns rather than fine-grained structural representations.

To further investigate this encoding process, we perform an independent probing of each model layer and visualize the reconstruction of a sample molecule’s adjacency matrix in Figure 7. This layer-wise analysis illustrates how structural details are progressively integrated across the network, showing that structural recoverability is most distinct in the intermediate layers.

## H Analysis of different node ordering strategies

In this section, we provide a comprehensive examination of how different node ordering policies, denoted as $q ( \mathbf { x } | G )$ , influence the performance and behavior of graph generative models. While prior literature [10, 6] has advocated for the use of Cuthill-McKee (CM) ordering, our empirical results on molecular datasets lead to a different conclusion. Before addressing the underlying reasons for this discrepancy, we first outline the theoretical and practical rationale that favors CM ordering.

## H.1 Rationale for CM ordering

Diamant et al. [10] advocated for CM ordering primarily to address two technical challenges: high computational complexity and an expansive output space. In models like GraphRNN [3], reduced bandwidth decreases the number of generation steps required per graph, thereby accelerating both training and inference. In SNACK, however, bandwidth dictates the total vocabulary size, which only marginally increases the computational overhead during the transformer’s output projection stage. Regarding the output space, SNACK does not inherently hold an advantage over GraphRNN; while GraphRNN extends sequence length to accommodate structural complexity, SNACK expands its vocabulary size. Consequently, CM ordering remains advantageous, as it can reduce the output space from $\mathcal { O } ( \breve { N } ^ { 2 } )$ to $\mathcal { O } ( \hat { \varphi } \cdot \bar { N } )$ , where $\hat { \varphi }$ denotes the estimated bandwidth.

## H.2 Optimal ordering policy depends on dataset

Similar to SNACK, GEEL [6] is a string-based method where only the vocabulary size is affected by the graph bandwidth. In their work, experiments on the Grid dataset demonstrated that CM ordering leads to faster convergence. We also observe this trend; furthermore, our results show the superior performance of CM ordering on certain datasets, notably on Community-small and Lobster; see Table 8. These datasets are characterized by their low-degree peripheral structures, where peripheral nodes are clearly distinguished from others. Since CM ordering initiates the traversal from a peripheral node, it yields more consistent sequences for these types of structured graphs. Representative examples of these graphs are visualized in Figure 8. Furthermore, these datasets are characterized by a significant disparity in maximum bandwidth between ordering policies, as evidenced by the results in Table 14.

![](images/580f4a8e0f19bf9770dff9b64dd245bc273214d079b0ad97c8e22c67cecb9d94.jpg)  
Figure 8: Sample graphs of generic datasets.

Table 14: Maximum bandwidth statistics for generic datasets using BFS versus CM ordering.
<table><tr><td>Dataset</td><td>BFS</td><td>CM</td></tr><tr><td>Community-small</td><td>15</td><td>8</td></tr><tr><td>Ego-small</td><td>16</td><td>15</td></tr><tr><td>Planar</td><td>33</td><td>27</td></tr><tr><td>Lobster</td><td>60</td><td>49</td></tr><tr><td>Enzymes</td><td>26</td><td>24</td></tr><tr><td>SBM</td><td>120</td><td>121</td></tr><tr><td>Ego</td><td>284</td><td>261</td></tr><tr><td>Grid</td><td>39</td><td>19</td></tr></table>

![](images/5fb31e53fd26cebfa8aa3a34fe9a3527e5923faea8e026257650f58fb8841f83.jpg)  
Figure 9: Training loss on Grid dataset.

In short, our findings suggest that there is no universally superior ordering policy; rather, the optimal ordering is highly dependent on the specific structural characteristics of the dataset. This observation aligns with Chen et al. [7], where they demonstrated that jointly training the ordering policy alongside the graph generative model can yield significant performance gains. While CM ordering effectively reduces the output space, our results indicate that the learnability of the resulting sequence—how well the model can capture the structural dependencies within a given budget—is a more critical determinant of generative quality.

## H.3 Analysis of evaluation metrics

We argue that several evaluation metrics previously employed to compare BFS and CM orderings are inadequate for a rigorous assessment of generative quality.

Loss and log-likelihood The negative log-likelihood (NLL) as defined in Equation 1, or the sequence-level log-likelihood, should not be used as a primary metric for comparison. As discussed in Section 2.4, the sequence-level log-likelihood log p<sub>θ</sub>(x) does not directly equate to the graph-level log-likelihood log $p _ { \theta } ( G )$ . Even under the assumption of perfect convergence, these values differ by the entropy of $q ( \mathbf { x } | G )$ , as illustrated by the following formulation:

$$
\begin{array} { r } { \log p _ { \theta } ( G ) \geq \mathbb { E } _ { \mathbf { x } \sim q ( \mathbf { x } \mid G ) } \left[ \log p _ { \theta } ( \mathbf { x } ) \right] + \mathcal { H } ( q ( \mathbf { x } \vert G ) ) } \end{array}\tag{12}
$$

Since different ordering policies $( \mathrm { e . g . }$ ., BFS vs. CM) yield different entropy values $\mathcal { H } ( q )$ , a direct comparison of their raw loss values is fundamentally misleading. This discrepancy is empirically evidenced by the gap in training loss between BFS and CM orderings shown in Figure 9. Furthermore, we note that the negative log-likelihood (NLL) is frequently computed at the token-level rather than the sequence-level for numerical stability and convenience. In such cases, the inherent differences in loss scales may become less discernible, as the per-token normalization masks the cumulative effect of the ordering’s complexity over the entire sequence.

Precision and Recall While Improved Precision and Recall metrics are widely adopted in the generative modeling literature [71] to quantify sample quality and diversity by approximating the data manifold, the metrics used in Diamant et al. [10] follow a fundamentally different definition. Specifically, their precision and recall are defined based on the percentage of correct actions (i.e., token-level accuracy) during autoregressive generation. Consequently, these metrics evaluate the model’s ability to reconstruct specific training sequences rather than assessing the generated graph distribution. We argue that such action-level metrics are inherently biased toward CM ordering. This bias arises because, for any given intermediate state during generation, there typically exist multiple valid actions that lead to distinct but equally correct trajectories for the same target graph. Since BFS induces a more diverse set of valid trajectories compared to the rigid paths of CM, it may naturally suffer from lower action-level accuracy.

Consequently, we recommend using metrics that leverage graph-level statistics, such as FCD or MMD, which are invariant to node ordering. Indeed, a closer inspection of the results reported by Diamant et al. [10] reveals that BFS ordering performs competitively with CM when evaluated via the MMD metric.

## I Discussion

String-based methods for graph generation have evolved to incorporate the strengths of graph-based approaches. This evolution is primarily driven by the objective of combining the capabilities of Large Language Models (LLMs) with the structural flexibility of graphs. While early molecular graph generation approaches relying on SMILES [28] often struggled to maintain syntactical validity [68], recent frameworks leverage specialized decoding rules to achieve 100% validity [4, 5]. Expanding beyond the molecular domain, string-based methods have been generalized to generic graphs [6, 16] and further extended to facilitate conditioned generation adhering to specific graph substructures [38, 30]. As a result, string-based methods have enabled diverse graph generation tasks with superior scalability compared to graph-based approaches [6, 30].

However, these advances remain fragmented, as existing string-based methods typically address only a subset of the desired capabilities. Some works do not guarantee strict structural validity [38, 16, 30], others are restricted to molecular graphs and thus lack general applicability [4, 38], and some do not support flexible conditioned generation in their current form [5, 6, 16]; moreover, evaluations are often confined to distribution learning tasks [5, 6, 30], which do not fully capture the challenges of exploration. Although some argue that 100% validity is of secondary importance since rejection sampling can filter invalid samples [72], in goal-directed settings, such an approach can severely hinder exploration and undermine training stability due to the sparse reward signals from invalid sequences.

Our paper addresses these gaps by proposing SNACK, a general-purpose string representation that guarantees 100% structural validity. SNACK provides a flexible framework that supports distribution learning, conditioned generation, and string-based GFlowNet training.

Limitation The SNACK framework currently relies on a fixed node ordering policy and atom-level generation. Although learning the ordering policy and fragment-level generation were not empirically explored in this initial study, SNACK provides the necessary structural foundation to implement such enhancements without architectural modifications, promising further improvements in model expressivity and scalability. Another limitation is that our empirical evaluation relied on the trajectory balance (TB) objective within a specific GFlowNet implementation. While TB provides a robust starting point, its convergence efficiency can be sensitive to reward landscapes and exploration strategies. Future work will investigate alternative training objectives to further stabilize the learning dynamics and enhance the generative diversity of SNACK-based models. Finally, due to constraints in computational resources, we were unable to conduct extensive hyperparameter search. In particular, the high computational cost associated with the graph-based GFlowNet baseline limited the scope of our comparative analysis at larger scales.

Broader impact By improving graph representation and generative modeling, our work establishes a practical framework that can potentially accelerate molecular design. This approach is particularly valuable for drug discovery and materials science, where efficient exploration of vast chemical spaces is essential. While our model facilitates rapid molecular design, we acknowledge the risk of it being misused to generate harmful or toxic compounds, a challenge that necessitates responsible deployment and screening protocols.