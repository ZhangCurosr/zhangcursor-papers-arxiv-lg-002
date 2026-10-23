# xVGAE: A Hierarchical Variational Graph Autoencoder for Exchangeable Graphs

Daniele Micheletti University of California, Irvine dmichele@uci.edu

Federica Zoe Ricci Swarthmore College fricci1@swarthmore.edu

Erik B. Sudderth University of California, Irvine sudderth@uci.edu

## Abstract

Graphs play a crucial role in many applications, ranging from drug discovery to social network modeling and astronomy. Generative models for graphs have substantially advanced in recent years, but we identify relatively simple scenarios where state-of-the-art models struggle to converge within a standard compute budget, or exhibit prohibitively slow generation times. We propose a generative architecture that achieves quality competitive with state-of-the-art models and is orders of magnitude faster at generation. Our approach is inspired by the Aldous-Hoover theorem, a classic representation theorem that characterizes any probability distribution of graphs that is invariant to permutations of node indices or exchangeable. This theorem establishes that three ingredients suffice: a graphlevel latent variable, a set of exchangeable node-specific latent variables, and a so-called graphon function that maps graph- and node-level variables to the probability of an edge between any node pair. Given a training set of graphs, our exchangeable Variational Graph AutoEncoder (xVGAE) learns an approximation to their underlying graphons, as well as graph- and node-level latent variables. We show that our xVGAE encodes latent representations that correlate with graphand node-level properties, and can (quickly, in one shot) generate new graphs that closely match key training set statistics.

## 1 Introduction

Graphs provide mathematical descriptions of interconnected systems in fields ranging from biology [Boisset et al., 2018] to neuroscience [Bassett and Sporns, 2017] and behavioral science [Aral and Dhillon, 2018]. While generative models for graphs offer substantial promise, their development is hindered by two primary challenges: (i) respecting the permutation invariance of node indices (assigning equal probability to isomorphic graphs), and (ii) balancing computational efficiency with the expressivity required to model potentially complex joint dependencies between all pairs of nodes.

Prior to recent advances in deep learning, statistical frameworks such as stochastic block models (SBMs) [Holland et al., 1983, Nowicki and Snijders, 2001], mixed-membership extensions (MMS-BMs) [Airoldi et al., 2008], random dot product graphs (RDPGs) [Young and Scheinerman, 2007], and latent space models (LSMs) [Hoff et al., 2002] were the primary generative models for networks. These models structure the graph distribution via node-level and graph-level latent variables, a strategy that inherently addresses permutation invariance and maintains computational tractability. However, each of these models imposes restrictive assumptions on the structural graph characteristics captured by the latent variables, as well as the functional form governing edge probabilities.

![](images/b3a3908fe3de14f6fceed6bb4c8f58e30d3ae8cf329629194be93ecb938f123b.jpg)  
Figure 1: Overview of the xVGAE architecture (inference and generation). The encoder (left) maps a graph A to a global posterior $q _ { \phi } ( z _ { 0 } \mid A )$ and node-level posteriors $q _ { \phi } ( z _ { i } \mid z _ { 0 } , A )$ for the n nodes. The prior (middle) samples the global latent $z _ { 0 } \sim p _ { \theta } ( z _ { 0 } )$ and, conditionally, the independent node latents $z _ { i } \sim p _ { \theta } ( z _ { i } \mid z _ { 0 } )$ . The decoder (right) refines the node latents into features $h _ { i }$ via an MLP, z -conditioned FiLM [Perez et al., 2018], and optional attention. Finally, a bilinear head predicts pairwise edge probabilities $p _ { \theta } { \left( { A _ { i j } \mid h _ { i } , h _ { j } , z _ { 0 } } \right) }$ for all node pairs to generate A<sup>ˆ</sup>.

A seminal effort to enhance the expressivity of these traditional generative models was the variational graph auto-encoder (VGAE) [Kipf and Welling, 2016], which employed graph convolutional networks (GCNs) [Kipf and Welling, 2017] and inference methods for variational autoencoders (VAEs) [Kingma and Welling, 2014] to generalize LSMs. While the original VGAE assumed an inflexible generative process and was primarily designed for representation learning, subsequent extensions introduced learnable decoders [Simonovsky and Komodakis, 2018, Jin et al., 2018, Samanta et al., 2020], though their experiments were limited to very small graphs and some training algorithms incorporated graph-matching procedures that are not scalable to larger graphs.

Among models capable of generating larger graphs, autoregressive architectures such as GraphRNN [You et al., 2018] and GRAN [Liao et al., 2019] construct adjacency matrices sequentially, sacrificing permutation invariance. More recently, diffusion-based approaches — both continuous [Niu et al., 2020, Jo et al., 2022] and discrete [Austin et al., 2021, Hoogeboom et al., 2021, Vignac et al., 2023, Chen et al., 2023] — preserve permutation equivariance and have set new standards for sample quality by directly denoising graph structures. Despite rapid advances, the practical success of diffusion graph models remains largely confined to small graphs with rich metadata (e.g., molecules) or medium-sized but very sparse graphs (e.g., proteins, point clouds).

Current state-of-the-art models can become prohibitively slow during generation when applied to dense graphs of moderate size (hundreds of nodes). Methods prioritizing scalability often rely on sparsity assumptions that fail in denser settings [Bergmeister et al., 2024], while highly expressive models suffer from generation bottlenecks because they either construct graphs sequentially [You et al., 2018, Liao et al., 2019] or iteratively update explicit latent variables for every potential edge across many denoising steps [Niu et al., 2020, Chen et al., 2023, Vignac et al., 2023, Zhou et al., 2024]. Furthermore, the latent variables of existing diffusion and autoregressive models represent either noisy versions of the adjacency matrix or sequence-dependent connectivity profiles, failing to provide any semantic decomposition or interpretable compression of the graph.

In this work, we address these limitations by introducing the eXchangeable Variational Graph AutoEncoder (xVGAE), a VAE-based generative model that adopts the Aldous-Hoover representation theorem for exchangeable graphs as its foundational design principle. This theoretical framework provides a principled way to decompose the structure of any exchangeable graph into node-level and graph-level latents without requiring per-edge latent variables. By adopting this design principle, our model achieves a dual benefit: it remains computationally tractable on dense graphs with hundreds of nodes—where existing methods often incur heavy training and sampling overhead—and it enables one-shot generation with quality comparable to competitors at speeds orders of magnitude faster. An added advantage of our approach is the ability to explore its learned latent representations, which we show correlate with key properties of community, ego, and protein networks.

## 2 Background: Representations of Exchangeable Graphs

In this section, we review the theoretical foundations underlying the xVGAE architecture developed in Sec. 3. An undirected binary graph with n nodes is represented by its adjacency matrix $A \in \{ 0 , \bar  1 \} \} ^ { n \times n }$ While graph sizes n vary across the dataset, we denote the node count of any given graph simply by $n$ for clarity. We index nodes by $i , j \in \{ 1 , \ldots , n \}$ , where $A _ { i j } = A _ { j i } = 1$ if an edge links nodes i and $j ,$ and 0 otherwise.

Permutation symmetry of graphs. The adjacency representation $A \in \{ 0 , 1 \} ^ { n \times n }$ of a graph is not unique: permuting nodes’ labels with a permutation π results in a new adjacency matrix $A ^ { \prime }$ such that $A _ { \pi ( i ) , \pi ( j ) } ^ { \prime } = A _ { i j }$ , which corresponds to the same graph up to isomorphism. Any probabilistic model defined on adjacency matrices should respect this symmetry. Existing generative models of graphs have employed different approaches to account for it: permutation-equivariant graph neural networks build the symmetry into the architecture; equivariant diffusion models enforce it on each step of the noising and denoising processes. Autoregressive models are not invariant to permutations, but try to reduce artifacts by imposing (heuristically chosen) canonical node orderings, or by augmenting the training data with permutations of the graphs.

Joint exchangeability and the Aldous-Hoover theorem. The probabilistic analogue of permutation invariance is joint exchangeability. A random infinite 2d array $A = ( A _ { i j } ) _ { i , j \in \mathbb { N } }$ is jointly exchangeable [Orbanz and Roy, 2015] if and only if, for every permutation π of N,

$$
( A _ { i j } ) \overset { d } { = } ( A _ { \pi ( i ) , \pi ( j ) } ) ,\tag{1}
$$

where $\circeq$ stands for equality in distribution, meaning that all permutations of the adjacency result in equal-probability graphs. Aldous and Hoover [Aldous, 1981, Hoover, 1979, Kallenberg, 2005] generalized the classic de Finetti theorem for exchangeable sequences to characterize jointly exchangeable 2d arrays. Such an array of a simple undirected graph is jointly exchangeable if and only if it admits the representation:

$$
A _ { i j } \mid u _ { 0 } , u _ { i } , u _ { j } \sim \mathrm { B e r n o u l l i } ( \omega ( u _ { 0 } , u _ { i } , u _ { j } ) ) .\tag{2}
$$

Here, $u _ { 0 } \in \mathcal { U }$ is a global or graph-level latent variable; $u _ { i } , u _ { j } \in \mathcal { U }$ are local or node-level latent variables; and $\omega : \mathcal { U } ^ { 3 }  [ 0 , 1 ]$ is a measurable function called a graphon, which is symmetric in $u _ { i }$ and $u _ { j }$ . For each realization of $u _ { 0 }$ , the graphon $\omega ( u _ { 0 } , \cdot , \cdot )$ is deterministic, and the distribution of a set of exchangeable graphs is a mixture of graphons indexed by $u _ { 0 }$ . In classic presentations of the Aldous-Hoover representation theorem, $u _ { 0 }$ and $( u _ { i } ) _ { i \in \mathbb { N } }$ are defined to be independent uniform random variables on [0, 1]. However, any non-atomic distribution can take the place of the uniform, as ω can be adjusted via an appropriate inverse-CDF transform [Orbanz and Roy, 2015]. In Section 3, we use this representation to construct a model that learns Euclidean (Gaussian or mixture) latent variables in place of uniformly-distributed ones, while maintaining exchangeability.

Connection to classical statistical models for graphs. Classical latent variable models for graphs correspond to specific parametric restrictions for the graphon of Eq. (2): SBMs assign each node to one of finitely many communities and let edge probabilities depend on community memberships only; MMSBMs relax the hard assignment of SBMs by associating each node with a distribution over communities; RDPGs replace community labels with continuous node embeddings whose inner product determines the edge probability; LSMs [Hoff et al., 2002] extend this idea by passing the resulting score through a link function. In every case, the global variable $u _ { 0 }$ controls the structure shared across the graph — community interaction probabilities, distribution over embeddings — while the per-node variables $u _ { i }$ encode each node’s identity within that structure.

From representation to architecture. The Aldous–Hoover theorem is an existence result (for infinite exchangeable arrays) rather than a constructive prescription: it does not specify how to parameterize ω, nor how to learn it from finite data. Yet, in a field with as many architectural degrees of freedom as deep generative modeling, a representation theorem provides a valuable structural inductive bias. It motivates organizing the latent space of a graph generative model around the two-level decomposition identified by the theorem: a global variable shared across the graph, and exchangeable per-node variables conditioned on it (see Fig. 1). A variational autoencoder that respects this hierarchy — with a decoder playing the role of ω — inherits a clean correspondence with the theory: the global latent indexes a graphon and the node latents index positions within it. Critically, the theorem indicates that this two-level latent structure is, in principle, sufficient to express rich exchangeable graph distributions without requiring explicit per-edge latent variables, provided the decoder is sufficiently flexible. We adopt this structure as the guiding design principle for our xVGAE model, whose concrete generative (decoder) and inference (encoder) architectures we develop in the next section.

## 3 A Hierarchical VAE for Exchangeable Graphs

We now introduce xVGAE, a variational autoencoder whose latent structure is designed according to the Aldous–Hoover representation theorem for jointly exchangeable arrays (summarized in Fig. 1). In contrast to prior work drawing on this theorem for classical latent space models [Orbanz and Roy, 2015, Lloyd et al., 2012] or graphon estimation [Xia et al., 2023], xVGAE adopts the Aldous–Hoover representation as its foundational framework for deep graph generation, empirically demonstrating its effectiveness on graphs with hundreds of nodes.

## 3.1 Generative model

To take the roles of $u _ { 0 }$ and $u _ { 1 } , \ldots , u _ { n }$ from Eq. (2), we define learnable Euclidean latent variables: a graph-level vector $z _ { 0 } \in \mathbb { R } ^ { d _ { 0 } }$ (the global latent variable) and per-node vectors $z _ { 1 } , \ldots , z _ { n } \in \mathbb { R } ^ { d }$ (the local latent variables). A function $\stackrel { \circ } { \omega } _ { \theta } : \mathbb { R } ^ { d _ { 0 } } \times \mathbb { R } ^ { d } \times \mathbb { R } ^ { d }  \lceil 0 , \bar { 1 } \rceil$ , symmetric in its last two arguments, determines the probability of each edge. Letting $\theta$ denote the set of all learnable parameters in the generative model,

$$
z _ { 0 } \sim p _ { \theta } ( z _ { 0 } ) , \qquad z _ { i } \mid z _ { 0 } \stackrel { \mathrm { i i d } } { \sim } p _ { \theta } ( z _ { i } \mid z _ { 0 } ) , \qquad A _ { i j } \mid z _ { 0 } , z _ { i } , z _ { j } \sim \mathrm { B e r n o u l l i } ( \omega _ { \theta } ( z _ { 0 } , z _ { i } , z _ { j } ) ) .\tag{3}
$$

Notice that the xVGAE local latent variables are independent and identically distributed conditionally on $z _ { 0 }$ , a parameterization allowed by the Aldous-Hoover theorem. Intuitively, this model formulates the graphon as a combination of two learnable parts: priors on per-graph and per-node latent variables that return Euclidean counterparts to the uniform latent variables of Eq. (2), and a decoder that maps the latent variables to edge probabilities. Next, we provide more details on these two parts.

Priors. For the conditional per-node prior distribution, we assume $p _ { \theta } { \left( z _ { i } \mid z _ { 0 } \right) }$ is a Gaussian mixture with K components whose weights, means, and diagonal covariances are determined by applying a deep neural network (in our experiments, a multi-layer perceptron (MLP)) to $z _ { \mathrm { 0 } }$ . The mixture form accommodates multimodal node distributions, such as community structure, while preserving exchangeability. Note that for $K = 1$ we obtain a Normal distribution whose mean and variance are learnable functions of the global variable $z _ { \mathrm { 0 } }$ . Marginalizing over $z _ { \mathrm { 0 } }$ yields a flexible prior on each $z _ { i } .$ so the global latent variable is responsible for mediating any dependence across nodes. This structure reveals the pivotal role played by the global latent variable, which is also charged with guiding the decoder’s mapping of per-node latents into edge probabilities (as detailed below). This motivates us to employ a flexible formulation for the graph-level prior $p _ { \theta } ( z _ { 0 } )$ , for which we adopt a latent diffusion process [Rombach et al., 2022]. In practice, we find that a simple MLP score network and the DDPM framework [Ho et al., 2020] allow the model to learn a sufficiently expressive prior.

Decoder. The decoder parameterizes ω<sub>θ</sub> as a $z _ { 0 } \cdot$ conditioned bilinear form on per-node embeddings,

$$
\begin{array} { r } { \omega _ { \theta } ( z _ { 0 } , z _ { i } , z _ { j } ) = \sigma \big ( h _ { i } ^ { \top } W ( z _ { 0 } ) h _ { j } + b ( z _ { 0 } , h _ { i } , h _ { j } ) \big ) , \qquad h _ { i } = \mathrm { M L P } ( z _ { i } ; \mathrm { F i L M } ( z _ { 0 } ) ) , } \end{array}\tag{4}
$$

where $\sigma$ is the logistic sigmoid activation function, $h _ { i } \in \mathbb { R } ^ { d _ { h } }$ is a per-node embedding produced by a shared MLP on $z _ { i }$ whose layers are FiLM-modulated [Perez et al., 2018] by $z _ { \mathrm { 0 } }$ , and $b ( z _ { 0 } , h _ { i } , h _ { j } ) =$ $b _ { 0 } ( z _ { 0 } ) + c ( h _ { i } ) + c ( h _ { j } )$ are a global bias and a per-node bias c shared by all nodes. $W ( z _ { 0 } ) \in \mathbb { R } ^ { d _ { h } \times d _ { h } }$ is a symmetric interaction matrix parameterized by a MLP hypernetwork taking $z _ { \mathrm { 0 } }$ as input. The symmetry of $W ( z _ { 0 } )$ and $b ( z _ { 0 } , h _ { i } , \bar { h _ { j } } )$ enforces the symmetry condition on ω<sub>θ</sub> required by the Aldous-Hoover Theorem (Sec. 2), and the per-node application of the MLP keeps $h _ { i }$ a function of $z _ { i }$ and $z _ { \mathrm { 0 } }$ alone, so $\omega _ { \theta }$ remains a pointwise graphon in the sense of Eq. (2). More details on the technical parameterization of the bilinear decoder are discussed in Appendix D.2. The classical models from Sec. 2 are recovered as special cases: a one-hot $h _ { i }$ with $W ( z _ { 0 } )$ containing the log-odds of inter- and intra-community probabilities corresponds to a SBM, a simplex-valued $h _ { i }$ to an MMSBM, and an identity $W$ to the inner-product decoder of the VGAE [Kipf and Welling, 2016], a logistic version of the RDPG.

Optional extension: Set attention decoder. The form of Eq. (4) matches the graphon specified in Eq. (2), up to the inverse-CDF identification of Euclidean $z _ { i }$ with uniform variables noted above and exchangeable transformation. A natural extension that can increase the expressivity of the decoder is to replace the independent per-node MLPs with a series of $s e l f .$ -attention blocks (SAB, Lee et al. 2019) operating on $\mathbf { \bar { \{ z } } _ { 1 } , . . . , \bar { z } _ { n } \}$ with FiLM conditioning on $z _ { 0 } .$ , producing embeddings $\{ h _ { i } \}$ that may depend on the full set of node latents. SAB is permutation-equivariant, and applying a permutation-equivariant map to a conditionally i.i.d. collection of latent variables preserves their joint exchangeability, so the induced graph distribution remains exchangeable. This extension trades projectivity of the decoder for added flexibility.

## 3.2 Inference model

VAEs are trained via an inference model or encoder, which outputs an (amortized) variational approximation of the true posterior. We assume it factorizes to match the generative hierarchy,

$$
q _ { \phi } ( z _ { 0 } , z _ { 1 : n } \mid A ) = q _ { \phi } ( z _ { 0 } \mid A ) \prod _ { i = 1 } ^ { n } q _ { \phi } ( z _ { i } \mid A , z _ { 0 } ) ,\tag{5}
$$

with both factors (global and local encoder) parameterized as Normal distributions, whose means and (diagonal) covariances are produced by SABs acting on graph features. We use Laplacian eigenvectors and a small graph neural network to produce node features for our encoder input. The global encoder conditions the SAB layers on the normalized Laplacian eigenvalues through FiLM layers, and pools its output through a permutation-invariant pooling by multihead attention (PMA) module [Lee et al., 2019]. Similarly to the decoder, the local encoder is FiLM-conditioned on a sample of $z _ { 0 }$ and outputs per-node Gaussian parameters. By construction, $q _ { \phi } ( z _ { 0 } \mid A )$ is permutation-invariant in the node ordering of $A$ and $q _ { \phi } { \left( z _ { 1 : n } \mid A , z _ { 0 } \right) }$ is permutation-equivariant, mirroring the symmetries of the generative model.

## 3.3 Training

Similar to many latent diffusion models [Rombach et al., 2022], training occurs in two stages. First, we train the hierarchical VAE to maximize the following two-level variational lower bound:

$$
\begin{array} { r l } { \mathcal { L } ( \theta , \phi ) \ = \ \mathbb { E } _ { q _ { \phi } ( z _ { 0 } , z _ { 1 : n } | A ) } [ \log p _ { \theta } ( A \mid z _ { 0 } , z _ { 1 : n } ) ] \ ~ } & { } \\ { \quad \quad - \ \beta _ { 0 } \operatorname { K L } \bigl ( q _ { \phi } ( z _ { 0 } \mid A ) \left\| p ( z _ { 0 } ) \right) \ ~ } & { } \\ { \quad \quad - \ \beta \mathbb { E } _ { q _ { \phi } ( z _ { 0 } | A ) } \bigl [ \operatorname { K L } \bigl ( q _ { \phi } ( z _ { 1 : n } \mid A , z _ { 0 } ) \left\| p _ { \theta } ( z _ { 1 : n } \mid z _ { 0 } ) \right) \bigr ] . } \end{array}\tag{6}
$$

In this first stage $p ( z _ { 0 } ) = \mathcal { N } ( 0 , I )$ ; the diffusion prior $p _ { \theta } ( z _ { 0 } )$ is fitted afterwards. The first (reconstruction) term decomposes as a sum over the upper triangular entries of the symmetric adjacency matrix A, and the third (local KL) term decomposes as a sum over the n nodes. Notice that training does not require sampling from the local prior, which would mean sampling the discrete assignments of the Gaussian mixture. This allows us to use a Monte Carlo approximation to get an unbiased estimate of the local KL term. The hyperparameters $\beta _ { 0 }$ and $\beta$ may be tuned to improve the quality of the posterior approximation [Higgins et al., 2017]. In experiments, we typically fix $\beta _ { 0 } \leq 1$ , and initialize $\beta$ to a small value that is gradually annealed towards 1. Appendix D provides additional information on the annealing schedule used.

Second, a small latent diffusion score network is trained using the encoded global embeddings $z _ { 0 }$ from the training data as target values, by minimizing the diffusion loss:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { D M } } = \mathbb { E } _ { t , z _ { 0 } ^ { ( 0 ) } , \epsilon } \Big [ \big \| \epsilon - \epsilon _ { \theta } \big ( z _ { 0 } ^ { ( t ) } , t \big ) \big \| _ { 2 } ^ { 2 } \Big ] , } \end{array}
$$

where $\epsilon _ { \theta }$ predicts the noise in $z _ { 0 } ^ { ( t ) }$ . Similar to Ho et al. [2020], we optimize an unweighted diffusion loss rather than incorporating the weights required by the full ELBO [Kingma et al., 2021], which helps avoid training instabilities at low noise levels. Note that latent diffusion is not a computational bottleneck for our model; this process is fast because we only have a single low-dimensional global variable for each graph in the training set.

![](images/d538687e53d805ea486ce25465a5f810c7ed8f387df6a06f1def32bd28daca30.jpg)  
Figure 2: Two-dimensional PCA projections of the latent space for the global variable $z _ { \mathrm { 0 } }$ across all MMSBM, LFR, Ego, and Protein graphs in the test set. The latent space embeds structural information of the graphs: the plots show a clear association between the global variable and the graphs’ number of communities, mixing rate $\mu ,$ average degree, and number of nodes n.

## 3.4 Generation of New Graphs

A new graph with n nodes is easily sampled by following the generative process of Eq. (3): draw $z _ { 0 } \sim p _ { \theta } ( z _ { 0 } )$ from the latent diffusion process, draw $z _ { 1 } , . . . , z _ { n } \stackrel { \mathrm { i i d } } { \sim } p _ { \theta } ( z _ { i } \mid z _ { 0 } )$ from the $z _ { 0 } \cdot$ -conditioned Gaussian mixture, and pass global and local latents through the decoder in one forward pass, to obtain edge probabilities $\omega _ { \theta } \big ( z _ { 0 } , z _ { i } , z _ { j } \big )$ . Edges are then drawn independently, $A _ { i j } \mid z _ { 0 } , z _ { i } , z _ { j }$ ∼ Bernoull $\mathrm { i } ( \omega _ { \theta } \big ( z _ { 0 } , z _ { i } , z _ { j } \big ) \big )$ . The node count n can be set freely or drawn from the empirical distribution of node counts in the training set, as we do in our experiments.

## 3.5 Visualization of the Learned Graph and Node Embeddings

The two-level hierarchical xVGAE generative process incentivizes the latent variables to represent different graph characteristics: $z _ { 0 }$ summarizes graph-level properties shared across all nodes, while each $z _ { i }$ encodes the role of node i given this global context. Unlike diffusion-based generative models that operate directly on graphs, where intermediate states are noisy versions of the adjacency matrix and admit no direct interpretation, both levels of our latent variables are inspectable on a per-graph and per-node basis. Figure 2 shows the correlation between graph statistics and the global latent space for four datasets. Figure 3 shows that the local latent variables capture node-level details such as group membership for MMSBM networks, and hop distance for Ego networks.

## 4 Related Work

Implicit Graphons. Xia et al. [2023] parameterize graphons as coordinate-based MLPs to build an algorithm for graphon estimation, which recovers a graphon on uniform variables as in Eq. (2) given an observed graph. They learn no explicit generative model, and Gromov–Wasserstein alignment makes training slow. Azizpour et al. [2025] scale this approach to larger graphs, for representation learning and graph augmentation. We do not align nodes to a canonical uniform space; we instead use the Aldous–Hoover representation as an architectural guide for a hierarchical VAE aimed at fast, one-shot generation of new graphs.

VAEs for graph generation. The VGAE [Kipf and Welling, 2016] assumed a fixed decoder and was thus used for representation learning, but it inspired VAE-based generative models, first for small molecules [Simonovsky and Komodakis, 2018, Jin et al., 2018, Samanta et al., 2020] and later via diffusion over continuous graph latents [Xu et al., 2023]. LGD [Zhou et al., 2024] embeds graphs with a permutation-equivariant transformer and trains a latent diffusion as a second stage; its unconditional generation results are limited to small graphs, and without a structured prior it needs both per-node and per-edge latents, which is costly on our benchmarks (Sec. 5). xVGAE instead structures the latent space into a global $z _ { \mathrm { 0 } }$ and conditionally i.i.d. node latents $z _ { i } ,$ recovering the inductive biases of classical statistical models within the Aldous–Hoover representation.

Local latent space across datasets (test) each point is one node

![](images/efc2ddd2f0c85c88bc507109e5f569aef42d260564a6888e5e9e65912f7e433f.jpg)

Figure 3: Top: graphs from the test set of four different datasets (MMSBM, Ego, LFR, Protein) visualized using a spring layout. The nodes are colored by community membership for MMSBM and LFR, by distance from the ego node (represented by a star) for the Ego graph, and by Fiedler vector for Protein graphs. Bottom: Two-dimensional PCA projections of the encoded local variables, which are associated with these node characteristics.  
![](images/993f4cfe73e1fbc744cfae0927ce77316884b138024c749f4a87878a70567e6a.jpg)  
Figure 4: Benchmark overview across four datasets. Panels compare xVGAE (blue) against the strongest (red) and second-strongest (orange) baselines per dataset. Both the xVGAE variant (MLP on MMSBM; SAB on LFR, Ego, and Protein) and the two baselines are selected per dataset by total column wins across degree, clustering, spectral, and WL $\mathrm { M M D ^ { 2 } }$ ratios (baseline names shown under each dataset). $\mathrm { M M D ^ { 2 } }$ ratios and s/G use a log scale; dashed lines mark the train–test reference (1.0). Lower is better; whiskers show min–max ranges over three draws.

## 5 Experiments

We compare xVGAE against twelve graph generative baselines across four datasets spanning synthetic mixed-membership communities, power-law community networks, citation ego networks, and protein contact maps. We evaluate three variants of our model (hyperparameters and chunked bilinear decoder details in Appendix D): xVGAE-MLP, which uses the pointwise graphon head of Eq. (4); xVGAE-SAB, which precedes the bilinear head with permutation-equivariant set-attention refinement (Sec. 3.1); and xVGAE without diffusion, an ablation using a standard normal prior on z<sub>0</sub>.

Table 1: MMSBM (n ∈ [120, 400], density ∼27.5%).
<table><tr><td></td><td colspan="4">MMD² / train-test ↓</td><td colspan="3">Auxiliary</td></tr><tr><td>Model</td><td>deg</td><td>clu</td><td>spec</td><td>orb</td><td>WL↓</td><td> $\mathrm { C 2 S T _ { \Delta } \downarrow }$ </td><td>s/G↓</td></tr><tr><td>Train vs. test (raw)</td><td>0.00179</td><td>0.00990</td><td>0.00016</td><td>0.01000</td><td>0.00714</td><td>n/a</td><td>n/a</td></tr><tr><td>One-shot</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VGAE</td><td>85.7 [85.1–88.4]</td><td>31.2[29.6–31.6]</td><td>1,002[984–1,032]</td><td> $1 . 0 0 ^ { \dag } _ { [ 1 . 0 0 - 1 . 0 0 ] }$ </td><td> $2 5 . 4 _ { [ 2 5 . 3 - 2 6 . 4 ] }$ </td><td>0.470[0.460–0.475]</td><td>&lt;0.01</td></tr><tr><td>GraphVAE*</td><td>300</td><td>102</td><td>4,847</td><td>100†</td><td>166</td><td>0.500</td><td>&lt;0.01</td></tr><tr><td>SPECTRE</td><td>9.19[8.38–9.39]</td><td>2.81 [2.76–2.87]</td><td>6.00[5.61–6.63]</td><td> $1 . 0 0 ^ { \dagger } _ { . } 1 1 . 0 0 { - } 1 . 0 0 ]$ </td><td>2.28[2.05–2.47]</td><td>0.195 [0.180–0.228]</td><td>0.03</td></tr><tr><td>IGNR</td><td>22.2[19.2–23.8]</td><td>21.3[19.8–22.1]</td><td> $1 4 . 0 _ { [ 1 4 . 0 - 1 5 . 1 ] }$ </td><td> $1 . 0 0 ^ { \dagger } _ { [ 1 . 0 0 - 1 . 0 0 ] }$ </td><td>7.75 [6.64–8.45]</td><td>0.238 [0.222–0.255]</td><td>&lt;0.01</td></tr><tr><td>Autoregressive</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GraphRNN</td><td>33.5 [32.0–34.3]</td><td>1.40[1.39–1.48]</td><td>157 [157–168]</td><td> $1 . 0 0 ^ { \dagger } _ { [ 1 . 0 0 - 1 . 0 0 ] }$ </td><td>16.8[16.0-16.8]</td><td>0.347 [0.320–0.350]</td><td>1.25</td></tr><tr><td>GRAN</td><td>2.07 [0.91–2.24]</td><td>1.07[1.06–1.07]</td><td>1.24[1.04–2.63]</td><td> $1 . 0 0 ^ { \dagger } _ { [ 1 . 0 0 - 1 . 0 0 ] }$ </td><td>1.35 [0.97–1.35]</td><td>0.037 [0.022–0.052]</td><td>4.15</td></tr><tr><td>Diffusion</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>EDP-GNN</td><td>83.8[23.0–90.0]</td><td>6.93[2.60–8.24]</td><td>319[16.8–356]</td><td> $\begin{array} { r } { 1 . 0 0 ^ { \dagger } \stackrel { \lbrack 1 . 0 0 - 1 . 0 0 \rbrack } { 1 0 0 ^ { \dagger } } } \end{array}$ </td><td>26.0[9.93–28.1]</td><td>0.463 [0.120–0.463]</td><td>4.39</td></tr><tr><td>LGD*</td><td>300</td><td>102</td><td>4,847</td><td></td><td>166</td><td>0.500</td><td>26.6</td></tr><tr><td>DiGress</td><td>74.3[71.3–75.4]</td><td> $2 . 7 8 [ 2 . 5 6 - 3 . 9 2 ]$ </td><td>846[818–849]</td><td> $1 . 0 0 ^ { \dagger } _ { . } 1 1 . 0 0 { - 2 . } 0 6 ]$ </td><td>26.4[25.6–27.8]</td><td>0.356[0.350–0.372]</td><td>18.2</td></tr><tr><td>SparseDiff</td><td>24.1 [20.1–26.8]</td><td> $3 . 4 9 _ { [ 3 . 2 8 - 3 . 7 7 ] }$ </td><td> $1 8 6 _ { [ 1 8 2 - 2 0 7 ] }$ </td><td> $1 . 0 0 ^ { \dagger } _ { [ 1 . 0 0 - 1 . 0 0 ] }$ </td><td>11.3[10.1–13.1]</td><td>0.218 [0.195–0.225]</td><td>45.2</td></tr><tr><td>EDGE ILE</td><td> $1 . 8 6 _ { [ 1 . 6 0 - 1 . 9 0 ] }$ </td><td> $2 . 3 2 _ { [ 2 . 1 6 - 2 . 4 4 ] }$ </td><td> $6 4 . 3 \scriptstyle { [ 3 7 . 8 - 6 9 . 3 ] }$ </td><td> $1 . 0 0 ^ { \dagger } _ { [ 1 . 0 0 - 1 . 0 0 ] }$ </td><td> $1 . 2 7 _ { [ 1 . 2 4 - 1 . 3 1 ] }$ </td><td> $0 . 0 3 7 _ { [ 0 . 0 2 0 - 0 . 0 5 5 ] }$ </td><td>0.16</td></tr><tr><td></td><td></td><td></td><td></td><td>OOM</td><td></td><td></td><td></td></tr><tr><td>xVGAE-SAB</td><td>2.53[2.48–4.31]</td><td>1.01 [0.99–1.07]</td><td>1.86[1.14–5.03]</td><td>1.00† [1.00–1.00]</td><td>1.34[1.25–1.76]</td><td>0.062 [0.018–0.065]</td><td>&lt;0.02</td></tr><tr><td>xVGAE-MLP</td><td> $2 . 2 7 _ { [ 2 . 1 5 - 4 . 5 2 ] }$ </td><td> $I . O 5 _ { [ 1 . 0 5 - 1 . 0 9 ] }$ </td><td> $\mathbf { 0 . 4 4 } _ { [ 0 . 3 7 - 3 . 4 1 ] }$ </td><td> $1 . 0 0 ^ { \dagger } _ { [ 1 . 0 0 - 1 . 0 0 ] }$ </td><td> $\underline { { 1 . 2 8 } } _ { [ 1 . 2 1 - 1 . 9 3 ] }$ </td><td>0.032[0.018–0.035]</td><td>&lt;0.02</td></tr><tr><td>xVGAE without diffusion</td><td> $3 . 4 7 _ { [ 1 . 5 6 - 4 . 6 9 ] }$ </td><td> $\underline { { 1 . 0 3 } } _ { [ 1 . 0 0 - 1 . 0 3 ] }$ </td><td> $\underline { { 0 . 9 6 } } \lbrack 0 . 3 7 - 1 . 9 8 \rbrack$ </td><td> $1 . 0 0 ^ { \dagger } _ { [ 1 . 0 0 - 1 . 0 0 ] }$ </td><td> $1 . 4 7 _ { [ 0 . 9 6 - 1 . 8 7 ] }$ </td><td>0.040[0.017–0.050]</td><td>&lt;0.01</td></tr></table>

<sup>†</sup> The summed orbit kernel saturates on this corpus (Appendix C.3). <sup>∗</sup> Collapsed: degenerate samples (Table 7).

## 5.1 Setup

Datasets. We evaluate on two synthetic and two real-world benchmarks (full specifications in Appendix A): MMSBM (1,000 graphs from Airoldi et al. [2008], n∈[120, 400], mean density 27.5%, $K \in \{ 3 , 5 \} )$ ; LFR [Lancichinetti et al., 2008] (1,000 graphs, $n \in [ 2 0 0 , 5 0 0 ]$ , mean density 3.7%, per-graph mixing µ and power-law degree/community-size tails); CiteSeer-Ego [You et al., 2018, Martinkus et al., 2022, Vignac et al., 2023] (757 3-hop ego networks from Sen et al. [2008], n ∈ [50, 399], mean density 3.6%); and Protein/D&D [Dobson and Doig, 2003] (918 protein graphs with heterogeneous size $n \in [ 1 0 0 , 5 0 0 ]$ , mean density 2.3%).

Baselines and compute protocol. We compare against one-shot models (VGAE [Kipf and Welling, 2016], GraphVAE [Simonovsky and Komodakis, 2018], SPECTRE [Martinkus et al., 2022], IGNR [Xia et al., 2023]), autoregressive generators (GraphRNN [You et al., 2018], GRAN [Liao et al., 2019]), continuous and discrete diffusion models (EDP-GNN [Niu et al., 2020], LGD [Zhou et al., 2024], DiGress [Vignac et al., 2023], SparseDiff [Qin et al., 2024], EDGE [Chen et al., 2023]), and iterative local expansion (ILE [Bergmeister et al., 2024]). All models share a uniform budget (one 48 GB GPU, 12 h training, 10 h sampling) using published hyperparameters. Under this fixed budget, computationally heavier baselines do not always reach their full published epoch counts and are evaluated at their best validation checkpoint (see Appendix C.1 for training protocols and a verification against published results). Generation time $( \mathrm { s } / \bar { \mathrm { G } } )$ is the fastest per-graph runtime across batch sizes 1, 8, and the largest batch up to 256 that fits in GPU memory (Appendix C.2). Because xVGAE runs diffusion only on the low-dimensional global vector z —followed by parallel node and pairwise decoding with a small memory footprint— it supports large batch sizes without memory bottlenecks.

Metrics. Following You et al. [2018], Martinkus et al. [2022], Vignac et al. [2023], we compute biased Gaussian-TV $\mathrm { M M D ^ { 2 } }$ [Gretton et al., 2012] on degree (deg), clustering coefficient (clu), normalized Laplacian spectrum (spec), and 4-node orbit counts (orb), along with Weisfeiler–Leman (WL) subtree kernel $\mathrm { \bar { M M D } ^ { 2 } }$ [Shervashidze et al., 2011] (detailed in Appendix B). Each $\mathrm { M M D ^ { 2 } }$ is reported as a ratio to the empirical train–test $\mathrm { M M D ^ { 2 } }$ ; a ratio near 1.0 (or slightly below due to finite-sample variance in the subsampled train–test denominator) indicates that generated graphs match the test distribution within sampling noise. Because the summed orbit kernel saturates on dense or high-degree graphs (returning ≈ 1.0 for any graph distribution), saturated orbit columns are marked with <sup>†</sup> (Appendix C.3). We also report the classifier two-sample test [Lopez-Paz and Oquab, 2017, Thompson et al., 2022], which trains a discriminator to distinguish generated from real graphs; we report its accuracy deviation from random guessing, $\mathrm { C 2 S T } _ { \Delta } = | \mathrm { a c c } - 0 . 5 |$ , where 0 means the two sets are indistinguishable (best) and 0.5 means perfect separation (worst). Finally, we report generation time in seconds per graph (s/G) considering sampling graphs in batches. All metrics are scored on the largest connected component across three draws (median and range reported); collapsed rows (<sup>∗</sup>, Table 7) show a single value.

Table 2: CiteSeer-Ego (n ∈ [50, 399], density ∼3.6%).
<table><tr><td></td><td colspan="4">MMD² / train-test ↓</td><td colspan="3">Auxiliary</td></tr><tr><td>Model</td><td>deg</td><td>clu</td><td>spec</td><td>orb</td><td>WL↓</td><td>C2ST△↓</td><td>s/G↓</td></tr><tr><td>Train vs. test (raw)</td><td>0.00031</td><td>0.00993</td><td>0.00095</td><td>0.01299</td><td>0.00437</td><td>n/a</td><td>n/a</td></tr><tr><td>One-shot</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VGAE*</td><td>1,390</td><td>16.9</td><td>382</td><td>4.94</td><td>160</td><td>0.500</td><td>&lt;0.01</td></tr><tr><td>GraphVAE*</td><td>2,469</td><td>107</td><td>719</td><td>77.1</td><td>368</td><td>0.500</td><td>&lt;0.01</td></tr><tr><td>SPÉCTRE</td><td>42.1 [39.0–43.9]</td><td>5.62 [5.37–5.72]</td><td>11.1 [10.3–11.9]</td><td>2.48 [2.33–2.53]</td><td>11.5[11.4-12.0]</td><td>0.260[0.259-0.276]</td><td>&lt;0.01</td></tr><tr><td>IGNR</td><td>79.1 [74.2–101]</td><td>14.3[14.1–14.5]</td><td>47.1 [46.9–50.9]</td><td>2.97 [2.58–3.04]</td><td>23.7 [22.0-27.1]</td><td>0.359[0.355–0.372]</td><td>&lt;0.01</td></tr><tr><td>Autoregressive</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GraphRNN</td><td>3.74[2.90–6.60]</td><td>26.8 [26.3–30.4]</td><td>16.2[14.8-17.2]</td><td>4.35 [4.19–4.59]</td><td>5.11 [4.94–5.60]</td><td>0.464[0.454–0.470]</td><td>0.14</td></tr><tr><td>GRAN</td><td>27.2[25.0–31.0]</td><td>7.73 [6.79–8.40]</td><td>13.9[9.57–14.2]</td><td>3.50[2.10–4.60]</td><td>7.32[6.80–7.88]</td><td>0.270[0.253-0.273]</td><td>0.46</td></tr><tr><td>Diffusion</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>EDP-GNN*</td><td>512</td><td>12.5</td><td>88.6</td><td>7.08</td><td>79.0</td><td>0.441</td><td>54.0</td></tr><tr><td>LGD</td><td>18.3[17.2–18.7]</td><td>16.3[15.7-16.9]</td><td>10.3[9.01–10.4]</td><td>2.21 [2.10–2.26]</td><td>6.89 [6.78–7.00]</td><td>0.230[0.217–0.243]</td><td>17.6</td></tr><tr><td>DiGress</td><td>10.9[8.92–12.8]</td><td>4.13[4.12–4.23]</td><td>3.57 [3.00–4.21]</td><td>1.55[1.54–2.10]</td><td>4.41 [3.99–4.63]</td><td>0.217 [0.158–0.220]</td><td>7.05</td></tr><tr><td>SparseDiff</td><td>55.7 [53.1–59.1]</td><td>6.36[6.10–6.46]</td><td>20.8[20.1–21.0]</td><td>2.93 [2.71–3.04]</td><td>8.99[8.75-9.14]</td><td>0.312[0.303-0.329]</td><td>15.0</td></tr><tr><td>EDGE</td><td>18.0[16.9-26.9]</td><td>6.78 [6.33–7.25]</td><td>14.5[12.2-18.7]</td><td>2.30[2.22–2.86]</td><td>7.08 [6.98–9.40]</td><td>0.148[0.112–0.165]</td><td>0.011</td></tr><tr><td>ILE</td><td>21.7 [20.1–24.2]</td><td>5.85 [5.33–5.88]</td><td>5.90[5.60–6.35]</td><td>3.11 [3.02–3.63]</td><td>8.98 [8.50-9.02]</td><td>0.299[0.277–0.346]</td><td>55.7</td></tr><tr><td>xVGAE-SAB</td><td>8.30[7.79-12.2]</td><td>5.20[4.53–5.64]</td><td>4.97[4.87–7.72]</td><td>2.62[2.41–2.63]</td><td>3.85[3.81–4.80]</td><td>0.102[0.095-0.141]</td><td>&lt;0.01</td></tr><tr><td>xVGAE-MLP</td><td>7.01 [4.73–8.15]</td><td>7.01 [6.44–7.47]</td><td>12.0[11.1–13.4]</td><td>1.92[1.77–2.09]</td><td>3.95[3.49–4.08]</td><td>0.326[0.326-0.335]</td><td>&lt;0.01</td></tr><tr><td>xVGAE without diffusion</td><td>5.91 [5.79–8.59]</td><td>5.81 [5.24–6.23]</td><td>12.9[10.9–12.9]</td><td>2.06[1.71–2.31]</td><td>3.57 [3.52–3.92]</td><td>0.368 [0.355–0.382]</td><td>&lt;0.01</td></tr></table>

<sup>∗</sup> Collapsed: degenerate samples (Table 7).

## 5.2 Results

Figure 4 summarizes performance against the two top-ranked baselines per dataset, and Tables 1–5 report full results (best in bold, second underlined, third italicized; OOM denotes out-of-memory at batch size 1; <sup>∗</sup> marks collapsed models).

MMSBM. On MMSBM (Table 1), xVGAE achieves the best clustering (SAB, 1.01), spectral (MLP, 0.44), and C2ST (MLP, 0.032) scores, and places a close second on WL (MLP, 1.28 vs. 1.27 for EDGE). Ratios near 1.0 on spectrum and clustering indicate that xVGAE’s generated graphs are close in distribution to the test graphs on these summaries, with C2ST<sub>∆</sub> (0.032) similarly close to chance. Interestingly, xVGAE-MLP outperforms xVGAE-SAB on four of the five ranked metrics, suggesting that our pointwise Aldous–Hoover bilinear head is already well suited to dense mixedmembership community structures. While GRAN and EDGE also achieve strong quality, ILE runs out of GPU memory (OOM) due to dense edge-triple message passing (Appendix C.1), and several diffusion models (DiGress, SparseDiff, EDP-GNN, LGD) underperform within the 12 h training budget while requiring up to 45 s/G to sample. Exploiting parallel batched decoding, xVGAE samples in <0.02 s/G—roughly 8× faster than EDGE and over 200× faster than GRAN.

LFR. On LFR (Table 4), EDGE and DiGress are the strongest baselines overall, while ILE again exhausts GPU memory (OOM). xVGAE-SAB achieves the best clustering (2.50) and spectral (1.17) $\mathrm { M M D ^ { 2 } }$ ratios overall, outperforming both EDGE and DiGress, though it trails them on degree, WL, and C2ST (Fig. 4). Unlike on MMSBM, xVGAE-SAB improves noticeably over xVGAE-MLP on clustering, spectrum, and WL, which suggests that set-attention refinement may help to model power-law degree and community-size variations. At generation time, xVGAE (< 0.01 s/G) is roughly 5× faster than EDGE (0.047 s/G), 100× faster than GRAN (1.03 s/G), and over 3,000× faster than DiGress (33.0 s/G).

CiteSeer-Ego. On CiteSeer-Ego (Table 2), DiGress and GraphRNN are the two strongest baselines overall. Here, xVGAE-SAB achieves the best C2ST (0.102 vs. 0.148 for EDGE and 0.217 for DiGress) and places second on clustering (5.20), spectrum (4.97), and WL (3.85, with our nondiffusion ablation reaching 3.57), while outperforming DiGress on degree (5.91–8.30 vs. 10.9; Fig. 4). Furthermore, xVGAE generates graphs in < 0.01 s/G—nearly three orders of magnitude faster than DiGress (7.05 s/G) and over 5,000× faster than ILE (55.7 s/G).

Protein. On Protein/D&D (Table 5), GRAN and ILE achieve the strongest overall sample quality, with SparseDiff winning on orbits. While xVGAE-SAB places second overall on degree (17.9) and WL (7.46) behind GRAN, it trails GRAN and ILE on clustering, spectrum, and $\mathrm { C 2 S T _ { \Delta } }$ (Fig. 4). Other latent-space and one-shot methods (LGD, IGNR, SPECTRE) similarly struggle on D&D, suggesting that compressing highly irregular, chain-like protein geometries into fixed-dimensional global and local vectors limits fine-grained structural fidelity. Even so, xVGAE remains one to three orders of magnitude faster at sampling (< 0.01 s/G vs. 0.82 s/G for GRAN and 14.5 s/G for ILE; qualitative comparisons in Appendix F).

Meaningful latent variables. Figures 2 and 3 suggest that xVGAE’s two-level hierarchy captures interpretable graph- and node-level structure without auxiliary supervision. The global latent $z _ { 0 }$ correlates with key graph properties: community count K on MMSBM, mixing $\mu$ on LFR, average degree on CiteSeer-Ego, and node count n on Protein. Simultaneously, local latents $z _ { i }$ reflect community membership on MMSBM and LFR, hop distance from the ego node on CiteSeer-Ego, and the Fiedler vector on Protein. See Appendix E for extended visualizations.

## 6 Conclusion

We introduced an exchangeable variational graph autoencoder (xVGAE) whose latent structure is guided by the Aldous–Hoover representation theorem for jointly exchangeable arrays. To our knowledge, xVGAE is the first deep generative model built on the Aldous–Hoover representation to reach sample quality competitive with state-of-the-art baselines on graphs with hundreds of nodes, while sampling up to three orders of magnitude faster. The two-level latent decomposition (a graphlevel $z _ { 0 } .$ , per-node $z _ { i }$ conditionally i.i.d. given $z _ { 0 }$ , and edges produced by a graphon-like map) yields a compact latent representation, with z and $z _ { i }$ empirically capturing global graph structure and node-level variation, respectively. This latent representation also allows rich generative processes, including denoising diffusions and mixture models, to be efficiently applied to graphs with hundreds of nodes. While the underlying representation theorem suggests that an xVGAE-style architecture can express rich exchangeable graph distributions, theoretical derivations are still needed to characterize the conditions under which a neural decoder can approximate arbitrary measurable symmetric kernels on finite graphs.

We evaluated our xVGAE on two complex synthetic benchmarks (MMSBM, LFR) and two real-world benchmarks (CiteSeer-Ego, Protein/D&D) against twelve baselines spanning one-shot VAEs, GANs, graphon estimators, autoregressive models, continuous- and discrete-state diffusion, and iterative local expansion. xVGAE achieves sample quality competitive with or superior to the strongest baselines on MMSBM, LFR, and CiteSeer-Ego, and places second on degree and WL on Protein/D&D, while generating graphs faster than every autoregressive and diffusion competitor on every dataset, from about 2× (EDGE) to three orders of magnitude.

The Aldous–Hoover form admits extensions to separately exchangeable and partially exchangeable arrays [Orbanz and Roy, 2015], opening a path for future generalizations of our xVGAE to bipartite, directed, and other asymmetric relational structures. While we focused strictly on binary adjacency matrices to isolate pure topological modeling, our framework can also be naturally extended to incorporate weighted edges, edge attributes, and node features or labels. On the architectural side, replacing the set-attention block with a linear-attention variant would drop the encoder and SABdecoder cost from quadratic to linear in the number of nodes. However, even with the current asymptotically quadratic cost, the xVGAE’s practical time and memory footprint is substantially lower than that of graph diffusion baselines.

Finally, while graph generative models offer powerful tools for simulating complex relational systems in fields such as network science and biology, their deployment requires careful consideration of the structural biases and distributional limitations inherent in training data, ensuring that generated synthetic graphs are used responsibly and transparently.

## Acknowledgments and Disclosure of Funding

This research was supported in part by the HPI Research Center in Machine Learning and Data Science at UC Irvine.

## References

Edoardo M. Airoldi, David M. Blei, Stephen E. Fienberg, and Eric P. Xing. Mixed membership stochastic blockmodels. Journal of Machine Learning Research, 9:1981–2014, 2008. URL https://jmlr.org/papers/v9/airoldi08a.html.

David J. Aldous. Representations for partially exchangeable arrays of random variables. Journal of Multivariate Analysis, 11(4):581–598, 1981. doi: 10.1016/0047-259X(81)90099-3.

Sinan Aral and Paramveer S. Dhillon. Social influence maximization under empirical influence models. Nature Human Behaviour, 2(6):375–382, 2018.

Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured denoising diffusion models in discrete state-spaces. In Advances in Neural Information Processing Systems, volume 34, 2021. URL https://arxiv.org/abs/2107.03006.

Ali Azizpour, Nicolas Zilberstein, and Santiago Segarra. Scalable implicit graphon learning. In International Conference on Artificial Intelligence and Statistics, volume 258 of PMLR, pages 3952–3960, 2025. URL https://proceedings.mlr.press/v258/azizpour25a.html.

Danielle S. Bassett and Olaf Sporns. Network neuroscience. Nature Neuroscience, 20(3):353–364, 2017.

Andreas Bergmeister, Karolis Martinkus, Nathanaël Perraudin, and Roger Wattenhofer. Efficient and scalable graph generation through iterative local expansion. In International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/2312.11529.

Vincent D. Blondel, Jean-Loup Guillaume, Renaud Lambiotte, and Etienne Lefebvre. Fast unfolding of communities in large networks. Journal ofStatistical Mechanics: Theory and Experiment, 2008 (10):P10008, 2008.

Jean-Charles Boisset, Judith Vivié, Dominic Grün, Mauro J. Muraro, Anna Lyubimova, and Alexander van Oudenaarden. Mapping the physical network of cellular interactions. Nature Methods, 15(7): 547–553, 2018.

Xiaohui Chen, Jiaxing He, Xu Han, and Li-Ping Liu. Efficient and degree-guided graph generation via discrete diffusion modeling. In International Conference on Machine Learning, 2023. URL https://arxiv.org/abs/2305.04111.

Paul D. Dobson and Andrew J. Doig. Distinguishing enzyme structures from non-enzymes without alignments. Journal ofMolecular Biology, 330(4):771–783, 2003. doi: 10.1016/S0022-2836(03) 00628-4.

Matthias Fey and Jan Eric Lenssen. Fast graph representation learning with PyTorch Geometric. In ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019. URL https: //arxiv.org/abs/1903.02428.

Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Schölkopf, and Alexander Smola. A kernel two-sample test. Journal ofMachine Learning Research, 13:723–773, 2012.

Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner. β-VAE: Learning basic visual concepts with a constrained variational framework. In International Conference on Learning Representations, 2017.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33, 2020. URL https://arxiv.org/abs/ 2006.11239.

Tomaž Hocevar and Janez Demšar. A combinatorial approach to graphlet counting. ˇ Bioinformatics, 30(4):559–565, 2014.

Peter D. Hoff, Adrian E. Raftery, and Mark S. Handcock. Latent space approaches to social network analysis. Journal ofthe American Statistical Association, 97(460):1090–1098, 2002.

Paul W. Holland, Kathryn Blackmond Laskey, and Samuel Leinhardt. Stochastic blockmodels: First steps. Social Networks, 5(2):109–137, 1983. doi: 10.1016/0378-8733(83)90021-7.

Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forré, and Max Welling. Argmax flows and multinomial diffusion: Learning categorical distributions. In Advances in Neural Information Processing Systems, volume 34, 2021. URL https://arxiv.org/abs/2102.05379.

Douglas N. Hoover. Relations on probability spaces and arrays of random variables. Preprint, Institute for Advanced Study, Princeton, 1979.

Wengong Jin, Regina Barzilay, and Tommi Jaakkola. Junction tree variational autoencoder for molecular graph generation. In International Conference on Machine Learning, 2018. URL https://arxiv.org/abs/1802.04364.

Jaehyeong Jo, Seul Lee, and Sung Ju Hwang. Score-based generative modeling of graphs via the system of stochastic differential equations. In International Conference on Machine Learning, 2022. URL https://arxiv.org/abs/2202.02514.

Olav Kallenberg. Probabilistic Symmetries and Invariance Principles. Springer, 2005.

Diederik Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. Advances in Neural Information Processing Systems, 34:21696–21707, 2021.

Diederik P. Kingma and Max Welling. Auto-encoding variational Bayes. In International Conference on Learning Representations, 2014. URL https://arxiv.org/abs/1312.6114.

Thomas N. Kipf and Max Welling. Variational graph auto-encoders. In NIPS Workshop on Bayesian Deep Learning, 2016. URL https://arxiv.org/abs/1611.07308.

Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations, 2017. URL https://arxiv.org/ abs/1609.02907.

Andrea Lancichinetti, Santo Fortunato, and Filippo Radicchi. Benchmark graphs for testing community detection algorithms. Physical Review E, 78(4):046110, 2008.

Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi, and Yee Whye Teh. Set transformer: A framework for attention-based permutation-invariant neural networks. In International Conference on Machine Learning, 2019. URL https://arxiv.org/abs/1810.00825.

Renjie Liao, Yujia Li, Yang Song, Shenlong Wang, Charlie Nash, William L. Hamilton, David Duvenaud, Raquel Urtasun, and Richard Zemel. Efficient graph generation with graph recurrent attention networks. In Advances in Neural Information Processing Systems, volume 32, 2019. URL https://arxiv.org/abs/1910.00760.

James Lloyd, Peter Orbanz, Zoubin Ghahramani, and Daniel M. Roy. Random function priors for exchangeable arrays with applications to graphs and relational data. In Advances in Neural Information Processing Systems, volume 25, 2012.

David Lopez-Paz and Maxime Oquab. Revisiting classifier two-sample tests. In International Conference on Learning Representations, 2017.

Karolis Martinkus, Andreas Loukas, Nathanaël Perraudin, and Roger Wattenhofer. SPECTRE: Spectral conditioning helps to overcome the expressivity limits of one-shot graph generators. In International Conference on Machine Learning, volume 162 of PMLR, pages 15159–15179, 2022. URL https://proceedings.mlr.press/v162/martinkus22a.html.

Chenhao Niu, Yang Song, Jiaming Song, Shengjia Zhao, Aditya Grover, and Stefano Ermon. Permutation invariant graph generation via score-based generative modeling. In International Conference on Artificial Intelligence and Statistics, 2020. URL https://arxiv.org/abs/2003.00638.

Krzysztof Nowicki and Tom A. B. Snijders. Estimation and prediction for stochastic blockstructures. Journal of the American Statistical Association, 96(455):1077–1087, 2001. doi: 10.1198/016214501753208735.

Peter Orbanz and Daniel M. Roy. Bayesian models of graphs, arrays and other exchangeable random structures. IEEE Transactions on Pattern Analysis and Machine Intelligence, 37(2):437–461, 2015. doi: 10.1109/TPAMI.2014.2334607.

Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron Courville. FiLM: Visual reasoning with a general conditioning layer. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 32, 2018.

Yiming Qin, Clément Vignac, and Pascal Frossard. Sparse training of discrete diffusion models for graph generation. Transactions on Machine Learning Research, 2024. URL https://arxiv. org/abs/2311.02142.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.

Bidisha Samanta, Abir De, Gourhari Jana, Vicenç Gómez, Pratim Chattaraj, Niloy Ganguly, and Manuel Gomez-Rodriguez. NeVAE: A deep generative model for molecular graphs. Journal of Machine Learning Research, 21(114):1–33, 2020.

Prithviraj Sen, Galileo Namata, Mustafa Bilgic, Lise Getoor, Brian Galligher, and Tina Eliassi-Rad. Collective classification in network data. AI Magazine, 29(3):93–106, 2008.

Nino Shervashidze, Pascal Schweitzer, Erik Jan van Leeuwen, Kurt Mehlhorn, and Karsten M. Borgwardt. Weisfeiler-Lehman graph kernels. Journal ofMachine Learning Research, 12:2539– 2561, 2011. URL https://jmlr.org/papers/v12/shervashidze11a.html.

Martin Simonovsky and Nikos Komodakis. GraphVAE: Towards generation of small graphs using variational autoencoders. In International Conference on Artificial Neural Networks (ICANN), 2018. URL https://arxiv.org/abs/1802.03480.

Rylee Thompson, Boris Knyazev, Elahe Ghalebi, Jungtaek Kim, and Graham W. Taylor. On evaluation metrics for graph generative models. In International Conference on Learning Representations, 2022. URL https://arxiv.org/abs/2201.09871.

Vincent A. Traag, Ludo Waltman, and Nees Jan van Eck. From Louvain to Leiden: Guaranteeing well-connected communities. Scientific Reports, 9(1):5233, 2019.

Clément Vignac, Igor Krawczuk, Antoine Siraudin, Bohan Wang, Volkan Cevher, and Pascal Frossard. DiGress: Discrete denoising diffusion for graph generation. In International Conference on Learning Representations, 2023. URL https://arxiv.org/abs/2209.14734.

Xinyue Xia, Gal Mishne, and Yusu Wang. Implicit graphon neural representation. In International Conference on Artificial Intelligence and Statistics, pages 10619–10634. PMLR, 2023.

Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? In International Conference on Learning Representations, 2019. URL https:// arxiv.org/abs/1810.00826.

Minkai Xu, Alexander S. Powers, Ron O. Dror, Stefano Ermon, and Jure Leskovec. Geometric latent diffusion models for 3D molecule generation. In International Conference on Machine Learning, pages 38592–38610. PMLR, 2023.

Jiaxuan You, Rex Ying, Xiang Ren, William L. Hamilton, and Jure Leskovec. GraphRNN: Generating realistic graphs with deep auto-regressive models. In International Conference on Machine Learning, pages 5708–5717. PMLR, 2018.

Stephen J. Young and Edward R. Scheinerman. Random dot product graph models for social networks. In Algorithms and Models for the Web-Graph (WAW 2007), volume 4863 of Lecture Notes in Computer Science, pages 138–149. Springer, 2007. doi: 10.1007/978-3-540-77004-6\_11.

Cai Zhou, Xiyuan Wang, and Muhan Zhang. Unifying generation and prediction on graphs with latent graph diffusion. In Advances in Neural Information Processing Systems, 2024. URL https://arxiv.org/abs/2402.02518.

## A Datasets

We evaluate on four graph-generation benchmarks spanning real-world citation topology, synthetic community-structured graphs, synthetic power-law community graphs, and biological protein-contact networks. Table 3 summarises the key statistics; the following subsections describe each dataset’s origin, generation process, split, and prior uses in the literature.

Table 3: Summary statistics for the four benchmarks. Splits are train / val / test graph counts. $n ,$ density $2 | E | / ( n ( { \overset { - } { n } } - 1 ) )$ and mean degree $2 | E | / n$ are per-graph averages over all splits; E<sup>¯</sup> is the mean edge count of the scored test graphs (largest connected component).
<table><tr><td>Dataset</td><td>Split (tr/va/te)</td><td>n (min / mean / max)</td><td>Density</td><td>Mean degree</td><td>E</td><td>Type</td></tr><tr><td>CiteSeer-Ego</td><td>484 / 121 / 152</td><td>50 / 145 / 399</td><td>0.036</td><td>4.1</td><td>328</td><td>Real, citation</td></tr><tr><td>MMSBM</td><td>640 / 160 / 200</td><td>120 / 249 / 400</td><td>0.275</td><td>68.4</td><td>9144</td><td>Synthetic, blockmodel</td></tr><tr><td>LFR</td><td>640 / 160 / 200</td><td>200 / 350 / 500</td><td>0.037</td><td>12.2</td><td>2044</td><td>Synthetic, power-law</td></tr><tr><td>Protein (D&amp;D)</td><td>587 / 147 /  184</td><td>100 / 258 / 500</td><td>0.023</td><td>5.0</td><td>632</td><td>Real, biological</td></tr></table>

## A.1 CiteSeer-Ego

Origin and generation. Three-hop ego-networks extracted from the CiteSeer citation graph [Sen et al., 2008]. For each seed node (paper), the induced subgraph is formed by collecting all nodes within three BFS hops and retaining every edge whose both endpoints lie within this neighbourhood. The resulting graphs have a hub-and-spoke topology: the ego node is linked to its immediate neighbours, which connect to sparser outer layers. Self-loops are removed and the graph is treated as undirected.

Node features. Each node is decorated with the leading d=50 eigenvectors of the normalized graph Laplacian. A deterministic sign convention is applied (argmax-| · | column forced non-negative) for reproducibility across runs and machines. The eigenvalue vector $\lambda \in \mathbb { R } ^ { d }$ is stored alongside.

Split. 484 / 121 / 152 (train / val / test), seed 42123. Node-count filter: graphs with fewer than 50 or more than 399 nodes are discarded before splitting.

Uses in the literature. Standard small-graph benchmark for unconditional graph generation. Used by GraphRNN [You et al., 2018], SPECTRE [Martinkus et al., 2022], DiGress [Vignac et al., 2023], GRAN [Liao et al., 2019], and essentially every modern graph generation paper as the canonical “dense small graphs” benchmark. Our version follows the GraphRNN preprocessing (3-hop, symmetric, no self-loops) and the SPECTRE eigenvector-feature convention.

## A.2 MMSBM

Origin and generation. Synthetic graphs sampled from a hierarchical instantiation of the Mixed-Membership Stochastic Blockmodel (MMSB) of Airoldi et al. [2008]. Each graph is generated independently as follows:

1. Sample the community count K uniformly from {3, 5}.

2. Sample Dirichlet concentration α ∼ Gamma(2, 0.15) (per graph).

3. For each node i, sample membership vector $\theta _ { i } \sim \operatorname { D i r } ( \boldsymbol { \alpha } \cdot \mathbf { 1 } _ { K } )$

4. Sample block affinities with mean $\bar { p } _ { \mathrm { i n } } \approx 0 . 5$ and $\bar { p } _ { \mathrm { o u t } } \approx 0 . 0 2 5$ , scaled by $\kappa { = } 5 0$

5. For each node pair $( i , j ) \colon$ draw community indicators $z _ { i } \sim \mathrm { C a t } ( \theta _ { i } ) , z _ { j } \sim \mathrm { C a t } ( \theta _ { j } )$ , then $A _ { i j } \sim \mathrm { B e r n } ( B _ { z _ { i } z _ { j } } )$ with $B _ { k l } = p _ { \mathrm { i n } }$ if $k { = } l ,$ else $p _ { \mathrm { o u t } }$

6. Sample graph size n from a negative binomial with mean 250 and standard deviation 60, clipped to [120, 400].

Graphs are post-processed to retain only the largest connected component (LCC). Node features: leading d=100 normalised-Laplacian eigenvectors.

Split. 1 000 graphs total; 640 / 160 / 200, seed 4242. Average density $\approx 2 7 \%$ , driven by the high $p _ { \mathrm { i n } }$ and the moderate–large node counts.

Uses in the literature. Introduced in this work as a benchmark for graph generative models with explicit per-graph community structure. The MMSB generative process is well-studied in the community detection literature [Airoldi et al., 2008]; using it as a generation benchmark is novel to this paper. The key challenge it poses is the per-graph variation in K and in membership concentration α: models without a graph-level latent variable collapse to a dataset-mean community structure and cannot reproduce per-graph variation in modular topology. With $\bar { n } { = } 2 4 9$ and density 0.275 it is also the densest corpus of the suite; ILE, whose state grows with the edges and triangles of the graph, runs out of GPU memory on it (Appendix C.1).

## A.3 LFR

Origin and generation. Lancichinetti–Fortunato–Radicchi (LFR) benchmark graphs [Lancichinetti et al., 2008], drawn with our own LFR sampler. Each graph has power-law degree and communitysize distributions and a mixing parameter $\mu ,$ the fraction of each node’s edges that leave its community. All of these are drawn at random per graph:

$n \sim$ Uniform $\{ 2 0 0 , \ldots , 5 0 0 \}$

• minimum degree ∼ Uniform $\{ 3 , \ldots , 1 0 \}$ , maximum degree log-uniform on [20, 150] and at most $n / 2 ;$

• degree exponent $\tau _ { 1 } \sim$ Uniform[2, 3] and community-size exponent $\tau _ { 2 } \sim$ Uniform[1.1, 2], community sizes in $[ 2 0 , n / 2 ] $

• µ ∼ Uniform[0.05, 0.5].

Degrees and community sizes are drawn exactly from these power laws, stubs are matched inside and between communities, and degree-preserving swaps make the graph simple. Every graph is checked for its exact degree sequence, its internal degrees, its per-node mixing and its connectivity; a graph that fails is redrawn. Node features: leading d=20 normalised-Laplacian eigenvectors.

Split. 640 / 160 / 200 (train / val / test). Training and validation graphs are a seeded permutation of 800 graphs; the 200 test graphs are drawn from the same generating process with seeds disjoint from training and validation. Mean n is 350 and mean density 0.037 (Table 3).

Uses in the literature. LFR is the standard community-detection benchmark [Lancichinetti et al., 2008] used to evaluate algorithms such as Louvain [Blondel et al., 2008], Leiden [Traag et al., 2019], Infomap, and others; it is rarely used to evaluate generative models. We include it because it combines heavy-tailed degrees, planted communities, a per-graph mixing level and graphs of up to 500 nodes.

## A.4 Protein (D&D)

Origin. The D&D dataset of Dobson and Doig [2003]. Each graph represents a protein: nodes are amino acids, and edges connect amino acids within a threshold Euclidean distance in the folded threedimensional structure. The dataset is a standard graph classification benchmark, here repurposed as an unconditional generation benchmark.

Generation and preprocessing. We use the version distributed with PyTorch Geometric [Fey and Lenssen, 2019]. Node labels (amino-acid type) are not used: like every model in our tables, ours sees only the graph structure. Graphs are filtered to the range $n \in [ 1 0 0 , \dot { 5 } 0 0 ]$ , yielding 918 graphs; the LCC is retained for all evaluation. Node features: leading d=100 normalised-Laplacian eigenvectors.

Split. 587 / 147 / 184 (train / val / test), seed 42123.

Uses in the literature. D&D is used as a graph-classification benchmark in the Weisfeiler-Leman literature and in kernel-based methods. As a generation benchmark it appears in GRAN [Liao et al., 2019], SPECTRE [Martinkus et al., 2022] and ILE [Bergmeister et al., 2024]. It is the most heterogeneous dataset in our suite: node count ranges from 100 to 500 within a single split, and local density varies substantially between chain-dominated and core-dominated regions of the protein graph.

## A.5 Comparison: dataset structural regimes

The four datasets cover four structurally distinct regimes:

• Small real-world. CiteSeer-Ego: real topology, hub-and-spoke, high per-graph size variability, sparse. Widest baseline coverage in the literature.

• Synthetic community structure. MMSBM: planted mixed-membership block structure with known ground truth, moderate–large node count and high density. Tests per-graph K and α variability under significant scale.

• Power-law synthetic. LFR: power-law degree and community-size tails, $n \in [ 2 0 0 , 5 0 0 ]$ with the exponents and the mixing level varying from graph to graph. Stresses tail fidelity on the largest graphs of the suite on average.

• Real-world heterogeneous. Protein (D&D): real biological graphs, large size range, heterogeneous density. Tests joint size-topology calibration under real-world structural variability.

## B Metrics: Explanation and Discussion

We evaluate generated graphs against the test set with six metrics. Five are kernel-based two-sample tests (Maximum Mean Discrepancy, $\mathrm { M M D ^ { 2 } } )$ ; one is a classifier two-sample test (C2ST). All metrics are computed on the largest connected component (LCC) of each graph.

## B.1 MMD<sup>2</sup> framework

Given test set $\mathcal { X }$ and generated set $\mathcal { V }$ with graph-level feature maps $\phi ( G ) \in \mathbb { R } ^ { d }$ , the biased $\mathrm { M M D ^ { 2 } }$ estimator under kernel $k ( \cdot , \cdot )$ is

$$
\widehat { \mathrm { M M D } } ^ { 2 } = \frac { 1 } { | \mathcal { X } | ^ { 2 } } \sum _ { x , x ^ { \prime } \in \mathcal { X } } k ( x , x ^ { \prime } ) + \frac { 1 } { | \mathcal { Y } | ^ { 2 } } \sum _ { y , y ^ { \prime } \in \mathcal { Y } } k ( y , y ^ { \prime } ) - \frac { 2 } { | \mathcal { X } | | \mathcal { Y } | } \sum _ { x \in \mathcal { X } , y \in \mathcal { Y } } k ( x , y ) .\tag{7}
$$

We use the Gaussian total-variation (TV) kernel $k ( x , y ) = \mathrm { e x p } \big ( - \| x - y \| _ { \mathrm { T V } } ^ { 2 } / 2 \sigma ^ { 2 } \big )$ with $\| x - y \| _ { \mathrm { T V } } =$ ${ \frac { 1 } { 2 } } \sum _ { i } \left| x _ { i } - y _ { i } \right|$ , following the GraphRNN/SPECTRE protocol [You et al., 2018, Martinkus et al., 2022]. The biased estimator is reported, to match the literature.

For the histogram-based metrics (degree, clustering and spectral), the per-graph feature is a normalized histogram of a node-level statistic, on a grid fixed before any sample is scored. The kernel keeps the published bandwidths: $\sigma { = } 1$ for degree and spectrum, $\sigma { = } 0 . 1$ for clustering.

Ratios. Every table reports $\widehat { \mathrm { M M D } } ^ { 2 } ( \mathcal { V } , \mathcal { X } _ { \mathrm { t e s t } } ) / \widehat { \mathrm { M M D } } ^ { 2 } ( \mathcal { X } _ { \mathrm { t r a i n } } , \mathcal { X } _ { \mathrm { t e s t } } )$ , with the training set subsampled to the size of the test set and the denominator averaged over five subsamples; a ratio of 1 means the generated graphs are as close to the test graphs as held-out real graphs are [Martinkus et al., 2022]. We report the ratio for each statistic separately.

## B.2 Histogram metrics: degree, clustering, spectral

Degree MMD (deg). Per graph: empirical histogram of node degrees, one bin per integer degree.   
Captures the marginal degree distribution.

Clustering MMD (clu). Per graph: histogram of per-node local clustering coefficients (triangle ratio over neighbour pairs), 100 bins on $[ 0 , \breve { 1 } ]$ . Captures local triangle density.

Spectral MMD (spec). Per graph: histogram of normalized graph-Laplacian eigenvalues, 100 bins on [0, 2]. Captures global mixing properties and modular structure [Martinkus et al., 2022].

## B.3 Orbit MMD

Standard protocol (GraphRNN/SPECTRE). For each graph, count the number of times each node participates in each of the 15 node orbits of graphlets with up to four nodes using ORCA [Hocevar and Demšar, 2014], average across nodes to produce a single 15-dimensional graph-levelˇ feature vector, and apply Eq. 7 with a Gaussian-TV kernel $\left( \sigma { = } 3 0 \right)$ on these vectors. On MMSBM and LFR this kernel saturates (Appendix C.3). We report this standard protocol, so that the orbit column is comparable with prior work, and mark the corpora on which it saturates.

## B.4 Weisfeiler–Leman MMD (wl)

The graph-level feature is the Weisfeiler–Leman subtree feature [Shervashidze et al., 2011], computed as a hashed bag of WL labels after four refinement rounds (16,384 buckets, $\ell _ { 2 } { \mathrm { - n o r m a l i s e d ) } } . \ \mathrm { M } \mathrm { { M D } ^ { 2 } }$ is then computed in the WL feature space under a linear kernel. WL is the most stable ranker in our experience: it captures local structural motifs at multiple scales and is robust to scale and size differences across the four datasets. We always include a WL column in every $\mathrm { M M D ^ { 2 } }$ table.

## B.5 Classifier Two-Sample Test (C2ST)

A logistic-regression binary classifier is trained to distinguish real $( y { = } 1 )$ from generated $( y { = } 0 )$ graphs in the embedding of an untrained three-layer GIN [Xu et al., 2019, Thompson et al., 2022], under 5-fold cross-validation. The reported score is the mean held-out accuracy: $\mathrm { C 2 S T = 0 . 5 }$ indicates real and generated graphs are indistinguishable; $\mathrm { C 2 S T = 1 . 0 }$ indicates the classifier separates them perfectly. Unlike $\mathrm { \bar { M M D } ^ { 2 } }$ , C2ST is a single scalar that aggregates all features through a learned decision boundary. In tables we report the deviation from chance $\mathrm { C 2 S T } _ { \Delta } = | \mathrm { a c c } - \frac { 1 } { 2 } |$ , so that $\mathrm { C 2 S T } _ { \Delta } = 0$ corresponds to indistinguishable distributions (best) and $\mathrm { C 2 S T } _ { \Delta } = 0 . 5$ to perfect separability (worst).

## B.6 Reporting conventions

• All MMD<sup>2</sup> values are reported in the biased form for consistency with the published baseline literature.

• All metrics are computed on the LCC of each graph;

• Lower is better for all five $\mathrm { M M D ^ { 2 } }$ columns and for $\mathrm { C 2 S T } _ { \Delta } \ ( 0 = \mathrm { c h a n c e \ a c c u r a c y } )$

• Every $\mathrm { M M D ^ { 2 } }$ is reported as a ratio to the training-versus-test value of the same statistic.

• The orbit column uses the standard protocol; <sup>†</sup> marks the corpora on which it saturates (Appendix C.3).

• A model is collapsed on a dataset when its samples are degenerate: the largest connected component is empty or fragmented (mean size below a quarter of the test graphs’), or the graphs have at least ten times the test edge count (Table 7). Collapsed rows are marked <sup>∗</sup> and shown without ranges.

Table 4: LFR (n ∈ [200, 500], density ∼3.7%).
<table><tr><td></td><td colspan="4">MMD² / train-test ↓</td><td colspan="3">Auxiliary</td></tr><tr><td>Model</td><td>deg</td><td>clu</td><td>spec</td><td>orb†</td><td>WL↓</td><td>C2ST∆ ↓</td><td>s/G↓</td></tr><tr><td>Train vs. test (raw)</td><td>0.00075</td><td>0.00914</td><td>0.00013</td><td>0.00986</td><td>0.00405</td><td>n/a</td><td>n/a</td></tr><tr><td>One-shot</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VGAE*</td><td>458</td><td>40.1</td><td>3,439</td><td>1.05</td><td>138</td><td>0.500</td><td>&lt;0.01</td></tr><tr><td>GraphVAE*</td><td>862</td><td>110</td><td>5,744</td><td>102</td><td>351</td><td>0.500</td><td>&lt;0.01</td></tr><tr><td>SPECTRE*</td><td>247</td><td>9.61</td><td>3,986</td><td>1.28</td><td>69.4</td><td>0.500</td><td>0.03</td></tr><tr><td>IGNR</td><td>33.4[32.7–36.8]</td><td>25.0[24.0–26.0]</td><td>11.2[11.0-11.8]</td><td>1.31 [1.29–1.35]</td><td>24.2 [23.9–27.0]</td><td>0.380[0.360–0.380]</td><td>&lt;0.01</td></tr><tr><td>Autoregressive</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GraphRNN</td><td>70.1 [64.9–86.9]</td><td>12.4[11.3-13.6]</td><td>9.78 [8.53–19.2]</td><td>1.16[1.16-1.20]</td><td>40.8[37.3–51.3]</td><td>0.465 [0.457–0.468]</td><td>0.18</td></tr><tr><td>GRAN</td><td>20.7 [16.3–25.6]</td><td>3.69[3.35–4.00]</td><td>8.49[6.97-17.0]</td><td>1.04[1.04–1.05]</td><td>16.6[13.9–18.8]</td><td>0.150[0.097–0.157]</td><td>1.03</td></tr><tr><td>Diffusion</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>EDP-GNN*</td><td>423</td><td>19.8</td><td>3,475</td><td>1.15</td><td>121</td><td>0.500</td><td>68.2</td></tr><tr><td>LGD*</td><td>308</td><td>13.3</td><td>3,074</td><td>9.86</td><td>103</td><td>0.490</td><td>37.5</td></tr><tr><td>DiGress</td><td>10.8[10.4-15.3]</td><td>3.61 [3.28–3.86]</td><td>5.22[4.42–5.31]</td><td>1.04[1.03–1.04]</td><td>8.64[8.38–12.2]</td><td>0.103 [0.080-0.107]</td><td>33.0</td></tr><tr><td>SparseDiff</td><td>31.3[26.5–35.1]</td><td>3.60[3.22-4.16]</td><td>86.4[81.7-104]</td><td>1.05[1.05–1.05]</td><td>21.3[16.9–23.2]</td><td>0.325[0.323–0.330]</td><td>65.3</td></tr><tr><td>EDGE</td><td>3.88[1.79–5.55]</td><td>3.16[3.14–3.30]</td><td>3.78[3.57–5.55]</td><td>1.06[1.06–1.08]</td><td>3.65[1.84–4.45]</td><td>0.060[0.055–0.097]</td><td>0.05</td></tr><tr><td>ILE</td><td></td><td></td><td></td><td>OOM</td><td></td><td></td><td></td></tr><tr><td>xVGAE-SAB</td><td>19.5[19.2–21.0]</td><td>2.50[2.12–2.52]</td><td>1.17[1.13–1.74]</td><td>1.04[1.00–1.06]</td><td>8.38[7.71–8.95]</td><td>0.165 [0.140–0.175]</td><td>&lt;0.01</td></tr><tr><td>xVGAE-MLP</td><td>22.0[21.9–25.9]</td><td>3.55[3.13–3.82]</td><td>3.28[1.83–5.21]</td><td>1.02[1.00–1.04]</td><td>9.41 [8.55–10.5]</td><td>0.153[0.140–0.155]</td><td>&lt;0.01</td></tr><tr><td>xVGAE without diffusion</td><td>16.8[15.8–16.8]</td><td>5.39[4.60–8.04]</td><td>1.46[1.19–1.87]</td><td>1.02[1.01-1.07]</td><td>7.70[7.45-7.71]</td><td>0.190[0.162–0.200]</td><td>&lt;0.01</td></tr></table>

<sup>†</sup> The summed orbit kernel saturates on this corpus (Appendix C.3). ∗ Collapsed: degenerate samples (Table 7).

Table 5: Protein/D&D $( n \in [ 1 0 0 , 5 0 0 ]$ , density heterogeneous).
<table><tr><td></td><td colspan="4">MMD² / train-test ↓</td><td colspan="4">Auxiliary</td></tr><tr><td>Model</td><td>deg</td><td>clu</td><td>spec</td><td>orb</td><td>WL↓</td><td>C2ST△ ↓</td><td></td><td>s/G↓</td></tr><tr><td>Train vs. test (raw)</td><td>0.00022</td><td>0.01058</td><td>0.00031</td><td>0.00318</td><td>0.00174</td><td></td><td>n/a</td><td>n/a</td></tr><tr><td>One-shot</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VGAE</td><td>2,541 [2,456–2,544]</td><td>28.7 [28.5–29.0]</td><td>1,012[995-1,032]</td><td>238 [238–238]</td><td></td><td>623 [610–625]</td><td>0.500[0.500–0.500]</td><td>0.006</td></tr><tr><td>GraphVAE</td><td>2,956[2,955–2,961]</td><td>100[100-101]</td><td>536[536-537]</td><td>444[444-444]</td><td></td><td>970[970-971]</td><td>0.500[0.500–0.500]</td><td>&lt;0.01</td></tr><tr><td>SPÉCTRE*</td><td>1,330</td><td>88.5</td><td>1,790</td><td></td><td>238</td><td>395</td><td>0.500</td><td>0.02</td></tr><tr><td>IGNR</td><td>597 [562–597]</td><td>19.1 [18.5–19.2]</td><td>149[148-151]</td><td>218[209–218]</td><td></td><td>152[146-153]</td><td>0.497 [0.497–0.497]</td><td>&lt;0.01</td></tr><tr><td>Autoregressive</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GraphRNN</td><td>29.2[26.5–34.4]</td><td>11.1 [11.0-11.4]</td><td>54.4[54.0–57.2]</td><td>135[132-149]</td><td></td><td>13.5[12.2-16.1]</td><td>0.440 [0.435–0.446]</td><td>0.12</td></tr><tr><td>GRAN</td><td>9.67 [9.21–12.0]</td><td>7.93 [7.57–8.56]</td><td>9.72[9.22–9.96]</td><td>64.1 [58.3–76.1]</td><td></td><td>3.91[3.85–5.29]</td><td>0.239 [0.215–0.280]</td><td>0.82</td></tr><tr><td>Diffusion</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>EDP-GNN</td><td>177[170-197]</td><td>49.2[47.0-51.2]</td><td>207 [206-208]</td><td>332[324–333]</td><td></td><td>55.8[53.1–61.9]</td><td>0.489 [0.486–0.492]</td><td>68.2</td></tr><tr><td>LGD*</td><td>1,741</td><td>102</td><td>2,391</td><td></td><td>239</td><td>485</td><td>0.500</td><td>30.5</td></tr><tr><td>DiGress</td><td>196[185-224]</td><td>8.22[8.14–8.39]</td><td>14.2[13.3-15.4]</td><td>200[196-212]</td><td></td><td>60.8[58.6–70.0]</td><td>0.391 [0.389–0.399]</td><td>19.4</td></tr><tr><td>SparseDiff</td><td>85.2[77.8–91.7]</td><td>10.2[9.37–11.1]</td><td>16.8[14.6–18.9]</td><td>8.72[7.57–9.99]</td><td></td><td>36.5[33.1–38.6]</td><td>0.418[0.413–0.429]</td><td>36.8</td></tr><tr><td>EDGE</td><td>91.9[77.9–93.0]</td><td>18.9[17.5-19.0]</td><td>112[109–112]</td><td>350[339–351]</td><td></td><td>31.5[26.6–31.9]</td><td>0.435 [0.429–0.446]</td><td>0.022</td></tr><tr><td>ILE</td><td>88.3[82.2–117]</td><td>7.38[6.53-7.53]</td><td>11.1 [10.2–15.2]</td><td>48.3[44.8–60.0]</td><td></td><td>34.8[32.9-44.6]</td><td>0.323 [0.312-0.324]</td><td>14.5</td></tr><tr><td>xVGAE-SAB</td><td>17.9[15.5–20.5]</td><td>16.1 [14.0–16.1]</td><td>72.9[68.4-73.7]</td><td></td><td>191 [173–212]</td><td>7.46[6.77-8.46]</td><td>0.456[0.448–0.462]</td><td>&lt;0.01</td></tr><tr><td>xVGAE-MLP</td><td>113[111-115]</td><td>14.5[13.8–14.9]</td><td>74.8[74.3–75.0]</td><td></td><td>201 [192–202]</td><td>34.9[33.7–35.7]</td><td>0.484[0.478–0.484]</td><td>&lt;0.01</td></tr><tr><td>xVGAE without diffusion</td><td>19.4[18.1–20.2]</td><td>13.9[13.5–15.5]</td><td>72.0[70.4–72.5]</td><td>205[174–210]</td><td></td><td>7.63[7.57–8.22]</td><td>0.451 [0.446–0.459]</td><td>&lt;0.01</td></tr></table>

Collapsed: degenerate samples (Table 7).

## C Experiments: extended details

## C.1 Compute limits, protocol and baseline notes

Limits. Every model, ours included, gets one GPU, 64 GiB of host RAM, 12 h of training and 10 h of sampling, on a 48 GB NVIDIA RTX 6000 Ada (some training ran on an A6000 of the same memory). Out of memory (OOM) is reported only when measured at batch size 1 within these limits. This happens for ILE alone, on MMSBM and LFR: training fails with a CUDA out-of-memory error in the sparse PPGN message aggregation over edge triples, whose number is large on these dense graphs (MMSBM: 5.4 GiB requested with 45.0 GiB already allocated; LFR: 3.6 GiB with 43.2 GiB, after 54 minutes). DiGress had a bug that would put it OOM on larger datasets like MMSBM or LFR, more details below.

Undertrained within the limits. LGD on CiteSeer-Ego (373 of the authors’ 3000 diffusion epochs), SPECTRE (about 23 epochs), SparseDiff (about 125 epochs) and EDGE (about 156 epochs) on LFR, and EDGE on MMSBM reach the 12 h training limit before converging; they are reported at the checkpoint chosen on validation among those saved. GRAN completes the authors’ 3000 epochs on every dataset within the limits (MMSBM: about 7 h; LFR: about 5 h); its checkpoints are chosen on validation (MMSBM: epoch 2925; LFR: epoch 1725).

Fairness of the budget. Some baselines are undertrained relative to the settings of their own papers. The comparison is nevertheless fair under our protocol: every model gets the same limits (one GPU, 12 h of training, 10 h of sampling), our model and most baselines converge well within them, and we prefer comparing more models to giving some of them more time. To check the implementations, we re-scored each baseline under the protocol of a published result (raw $\mathrm { M M D ^ { 2 } }$ of degree, clustering and orbits with the authors’ evaluator, full graphs, one draw; Table 6). 7 out of 12 comparisons match; the mismatches are attributed mostly to the training budget (suspected on most of them by comparing the completed epochs to the ones suggested by the authors). One caveat: our CiteSeer-Ego split differs from the published ones.

Table 6: Baselines against their published numbers: raw $\mathrm { M M D ^ { 2 } }$ degree / clustering / orbit under the published protocol (EMD kernel where marked, TV otherwise). Sources in parentheses: the table of the paper reporting the number.
<table><tr><td>Baseline</td><td>Dataset</td><td>Published (source)</td><td>Ours, their protocol</td><td>Verdict</td><td>Likely cause</td></tr><tr><td>GRAN</td><td>Protein</td><td>2.0e-3 / 0.049 / 0.13 (GRAN T1)</td><td>2.5e-3 / 0.082 / 0.16</td><td>match</td><td></td></tr><tr><td>GRAN</td><td>Ego</td><td>7.7e-3 / 0.066 / 0.043 (HiGen T9)</td><td>0.017 / 0.11 / 0.061</td><td>match (≤ 2.2×)</td><td>split</td></tr><tr><td>GraphRNN</td><td>Protein</td><td>4.0e-3 / 0.15 / 0.59 (SPECTRE T2)</td><td>7.0e-3 / 0.12 / 0.40</td><td>match</td><td></td></tr><tr><td>GraphRNN</td><td>Ego (EMD)</td><td>0.077 / 1.15 / 0.11 (EDGE T2)</td><td>0.18 / 1.08 / 0.052</td><td>match</td><td></td></tr><tr><td>DiGress</td><td>Ego</td><td>8.9e-3 / 0.054 / 0.030 (SparseDiff T1)</td><td>3.9e-3 / 0.041 / 0.027</td><td>match</td><td></td></tr><tr><td>DiGress</td><td>Protein</td><td>4.1e-3 / 0.049 / 0.13 (ILE T2)</td><td>0.050 / 0.080 / 0.58</td><td>mismatch, 4–12×</td><td>undertrained</td></tr><tr><td>EDGE</td><td>Ego (EMD)</td><td>0.058 / 0.18 / 0.052 (EDGE T2)</td><td>0.024 / 0.14 / 0.026</td><td>match</td><td></td></tr><tr><td>EDGE</td><td>Protein</td><td>0.19 / 0.34 / 0.68 (ILE T2)</td><td>0.023 / 0.20 / 1.08</td><td>match</td><td></td></tr><tr><td>SparseDiff</td><td>Protein</td><td>1.5e-3 / 0.034 / 5.3e-3 (SparseDiff T7)</td><td>0.018 / 0.10 / 0.025</td><td>mismatch, 3-12×</td><td>undertrained</td></tr><tr><td>SparseDiff</td><td>Ego</td><td>3.7e-3 / 0.032 / 0.020 (SparseDiff T7)</td><td>0.018 / 0.065 / 0.039</td><td>mismatch, 2–5×</td><td>undertrained; split</td></tr><tr><td>ILE</td><td>Protein</td><td>3.0e-3 / 0.031 / 4.7e-3 (ILE T2)</td><td>0.017 / 0.080 / 0.13</td><td>mismatch, 3–28×</td><td>undertrained</td></tr><tr><td>SPECTRE</td><td>Protein</td><td>5.6e-3 / 0.084 / 0.027 (SPECTRE T2)</td><td>0.34 / 0.94 / 0.73</td><td>mismatch, 30–60×</td><td>undertrained</td></tr></table>

Protocol. Baselines run from their public code (VGAE and GraphVAE from our reimplementations) with upstream hyper-parameters, changed only where our graph sizes force it. All models share the scoring code, the splits and the selection rule: draws of one checkpoint are averaged, the checkpoint with the lowest validation score (geometric mean over the five statistics of $\mathrm { M M D ^ { 2 } }$ divided by its train-vs-validation value) wins, every candidate within 10% of it is tied, and the tie goes to the C2ST accuracy closest to 0.5; only the winner is scored on test. Each table row is the median of three sampling draws of that checkpoint, with their range. Collapsed rows (Appendix B) are drawn once. For xVGAE this rule picks both the autoencoder and the $z _ { 0 }$ diffusion. The xVGAE model without diffusion is the best on validation among the MLP and SAB versions used for the model with diffusion and new MLP and SAB models trained with both $\beta$ and $\beta _ { 0 }$ set to 1 for fairness. Every model draws n from the training sizes, except GraphRNN and GraphVAE, which choose it themselves. VGAE, GraphVAE and LGD threshold edge probabilities at 0.5; the others sample them. EDGE, as published, steers generation toward the degree sequence of a training graph, so it does not have to learn the degree distribution. EDP-GNN runs $6 \times 1 0 0 0$ annealed Langevin steps $( 6 \times 1 0 0$ on MMSBM).

Collapsed models. Table 7 lists the collapsed rows and the reason; all have C2ST accuracy close to 1 $( \mathrm { C 2 S T } _ { \Delta } \geq 0 . 4 4 )$  
Table 7: Collapsed rows: mean LCC size and mean edge count of the generated graphs against the test graphs (test: n¯ / E<sup>¯</sup>).
<table><tr><td>Dataset (test ñ / È)</td><td>Model</td><td>LCCñ</td><td>E</td><td>Reason</td></tr><tr><td>MMSBM (249 / 9,144)</td><td>GraphVAE</td><td>1</td><td>0</td><td>empty</td></tr><tr><td rowspan="2">LFR (350 / 2,044)</td><td>LGD</td><td>1</td><td>0</td><td>empty</td></tr><tr><td>GraphVAE</td><td>1</td><td>0</td><td>empty</td></tr><tr><td rowspan="7">CiteSeer-Ego (145 / 328)</td><td>SPECTRE</td><td>42</td><td>191</td><td>fragmented</td></tr><tr><td>VGAE</td><td>346</td><td>31,679</td><td>15× edges</td></tr><tr><td>EDP-GNN</td><td>360</td><td>43,119</td><td>21× edges</td></tr><tr><td>LGD</td><td>223</td><td>28,092</td><td>14× edges</td></tr><tr><td>GraphVAE</td><td>1</td><td>0</td><td>empty</td></tr><tr><td>VGAE</td><td>148</td><td>7,213</td><td>22× edges</td></tr><tr><td>EDP-GNN</td><td>178</td><td>20,888</td><td>64× edges</td></tr><tr><td rowspan="2">Protein (258 / 632)</td><td>LGD</td><td>254</td><td>37,573</td><td>59× edges</td></tr><tr><td>SPECTRE</td><td>4</td><td>4</td><td>fragmentedª</td></tr></table>

<sup>a</sup> Two fresh draws of the reported checkpoint; the reported draw has LCC $\bar { n } = 2 0 3$ and is equally separable (C2ST<sub>∆</sub> = 0.500).

DiGress. Running the official DiGress code on MMSBM and LFR initially exhausted host RAM (234 GiB in preprocessing) due to an implementation bug: the dataset-information object keeps a reference to the whole data module, every graph of every split, and PyTorch Lightning copies it into the saved hyperparameters before the first training step. Our "DiGress-lean" version drops this reference once it has been used and keeps the extra-feature buffers out of the saved hyperparameters; the model, loss, noise schedule and sampler are unchanged, and a seeded CPU check gives identical losses and weights with and without it. With it, Protein needs 0.6 GB of host memory after setup and 9.6 GB of GPU memory per 499-node graph at batch size 1. Every DiGress row uses DiGress-lean except CiteSeer-Ego, whose run used the original entry point, which fits on that dataset. DiGress’s cycle-count features are differences of traces of powers of the adjacency matrix; on dense graphs they exceed the exact-integer range of float32 (about 10<sup>8</sup> per node on MMSBM), so small counts came out negative and stopped training at DiGress’s own check. An opt-in switch computes them in float64, used on MMSBM and LFR.

## C.2 Sampling protocol

Sampled-graph counts equal the test-set size for each dataset. Node counts n are drawn from the empirical training-set distribution (exceptions above). All scoring is on the largest connected component (LCC) of each graph. Every model is timed on one NVIDIA RTX 6000 Ada from the checkpoint of its table row, at batch size 1, at batch size 8 and at the largest batch up to 256 that fits in GPU memory; the tables report the fastest, in seconds per graph (s/G). Times cover the whole generative pipeline at the settings of the reported samples (all denoising or Langevin steps, decoding and edge sampling), with n drawn from the training sizes, and exclude model loading. For EDP-GNN and LGD each batch size is timed on one batch after a warm-up batch. Table 8 gives the batch size of each reported s/G.

## C.3 Orbit metric on LFR and MMSBM

The standard orbit kernel compares two graphs by a Gaussian kernel of width 30 on the TV distance between their node-averaged ORCA count vectors. Per-node counts grow like the cube of the degree, so these vectors are far apart: between two MMSBM test graphs the median distance is about $2 \times 1 0 ^ { 5 }$ and the kernel is numerically 0. When every off-diagonal kernel value is 0, the biased $\mathrm { M M D ^ { 2 } }$ of any two sets of m graphs equals $2 / m ,$ , whatever the graphs. We measure this with 1 − κ¯, where κ¯ is the mean kernel value between two distinct real graphs (1 = saturated): it reads 0.95 on CiteSeer-Ego,

Table 8: Batch size of the fastest timing (the s/G reported in the result tables).
<table><tr><td>Model</td><td>MMSBM</td><td>LFR</td><td>CiteSeer-Ego</td><td>Protein</td></tr><tr><td>VGAE</td><td>64</td><td>8</td><td>1</td><td>1</td></tr><tr><td>GraphVAE</td><td>128</td><td>128</td><td>64</td><td>128</td></tr><tr><td>SPECTRE</td><td>8</td><td>8</td><td>8</td><td>8</td></tr><tr><td>IGNR</td><td>8</td><td>8</td><td>8</td><td>8</td></tr><tr><td>GraphRNN</td><td>256</td><td>256</td><td>256</td><td>256</td></tr><tr><td>GRAN</td><td>64</td><td>64</td><td>64</td><td>64</td></tr><tr><td>EDP-GNN</td><td>8</td><td>8</td><td>4</td><td>8</td></tr><tr><td>LGD</td><td>8</td><td>16</td><td>8</td><td>8</td></tr><tr><td>DiGress</td><td>8</td><td>8</td><td>8</td><td>1</td></tr><tr><td>SparseDiff</td><td>32</td><td>16</td><td>8</td><td>8</td></tr><tr><td>EDGE</td><td>128</td><td>256</td><td>128</td><td>256</td></tr><tr><td>ILE</td><td>OOM</td><td>OOM</td><td>8</td><td>64</td></tr><tr><td>xVGAE (with  $z _ { 0 }$  diffusion)</td><td>256</td><td>256</td><td>256</td><td>256</td></tr><tr><td>xVGAE without diffusion</td><td>64</td><td>64</td><td>8</td><td>128</td></tr></table>

1.00 on MMSBM and 1.00 on LFR, and 0.24 on Protein. On MMSBM the summed kernel returns exactly $2 / m$ for held-out real graphs, for the same graphs after degree-preserving rewiring and for Erdos–Rényi graphs with the same edge counts. Because a saturated orbit ratio is˝ $2 / m$ divided by $2 / m$ , it reads close to 1 for every model, including models whose graphs are far from the data. We therefore report the orbit column for comparability with prior work but flag it where it saturates (marked <sup>†</sup>), and rely on the degree, clustering, spectral and WL ratios and on C2ST there.

## D Hyperparameter Details

## D.1 Configurations per Dataset

Table 9 reports the autoencoder configurations for each xVGAE variant. All models share a 5-layer set-attention (SAB) encoder, hidden dimension $h = 6 4$ , latent dimensions $d _ { z } = 8$ and $d _ { z _ { 0 } } = 3 2$ $K _ { h } = 8$ chunked bilinear heads (rank 4), a Gaussian mixture node prior conditioned on z<sub>0</sub>, and a batch size of 8. Models are trained using AdamW with a learning rate of $5 \times 1 0 ^ { - 4 }$ and weight decay $1 0 ^ { - 5 }$ . Decoders use either 5 SAB layers or a pointwise MLP on $( z _ { 0 } , z _ { i } )$

Six of the seven diffusion-prior models follow our standard training schedule: $\beta _ { 0 } = 0 . 1$ , a cosine learning rate schedule, and β annealed from $1 0 ^ { - 4 }$ to 1 along a half-cosine after a warm-up phase (with $\bar { \beta _ { 0 } }$ scaled by β). On CiteSeer-Ego, $\mathrm { x V G A E - S A B }$ uses a fixed $\beta = \beta _ { 0 } = 1$ and a constant learning rate for 2000 epochs. The $z _ { \mathrm { 0 } }$ diffusion prior is subsequently fitted to the frozen encoder, with both autoencoder and diffusion checkpoints selected on validation (Appendix C.1). For variants without a diffusion prior $( z _ { 0 } \sim \mathcal { N } ( 0 , I ) )$ , checkpoints are selected on validation: CiteSeer-Ego uses the xVGAE-MLP autoencoder, Protein and LFR use the xVGAE-SAB autoencoders, and MMSBM uses the dedicated configuration in the final row of Table 9.

Table 9: xVGAE autoencoder configurations. All models use a 5-layer SAB encoder and either a 5-layer SAB or pointwise MLP decoder. Schedule triples denote warm-up / anneal / total epochs. <sup>∗</sup>Uses a 32-component Gaussian-mixture node prior, weight decay $1 0 ^ { - 8 }$ , and $z _ { 0 } \sim \mathcal { N } ( 0 , I )$
<table><tr><td>Dataset</td><td>Row</td><td>Encoder</td><td>Decoder</td><td> $\beta _ { 0 }$ </td><td> $\beta$  schedule; LR schedule</td></tr><tr><td>CiteSeer-Ego</td><td>xVGAE-SAB</td><td>SAB, 5</td><td>SAB, 5</td><td>1</td><td>β=1, 2000 ep.; constant</td></tr><tr><td>CiteSeer-Ego</td><td>xVGAE-MLP</td><td>SAB,5</td><td>MLP</td><td>0.1</td><td>16 / 46 / 1230; cosine</td></tr><tr><td>MMSBM</td><td>xVGAE-SAB</td><td>SAB,5</td><td>SAB,5</td><td>0.1</td><td>12 / 35 / 938; cosine</td></tr><tr><td>MMSBM</td><td>xVGAE-MLP</td><td>SAB,5</td><td>MLP</td><td>0.1</td><td>12 / 35 / 938; cosine</td></tr><tr><td>LFR</td><td>xVGAE-SAB</td><td>SAB,5</td><td>SAB,5</td><td>0.1</td><td>12 / 35 / 938; cosine</td></tr><tr><td>LFR</td><td>xVGAE-MLP</td><td>SAB,5</td><td>MLP</td><td>0.1</td><td>12 / 35 / 938; cosine</td></tr><tr><td>Protein (D&amp;D)</td><td>xVGAE-SAB</td><td>SAB,5</td><td>SAB, 5</td><td>0.1</td><td>13 / 38 / 1014; cosine</td></tr><tr><td>MMSBM</td><td>without diffusion*</td><td>SAB, 5</td><td>MLP</td><td>1</td><td> $\beta$  fixed, 5000 ep.; constant</td></tr></table>

## D.2 Chunked Bilinear Decoder and W Parameterisations

The logit of $\mathrm { E q . ~ } ( 4 )$ is a mixture of $K _ { h }$ bilinear forms, each acting on its own slice of the node embedding $h _ { i } \in \mathbb { R } ^ { d _ { h } }$ :

$$
s _ { i j } ~ = ~ \sum _ { k = 1 } ^ { K _ { h } } \alpha _ { k } ( z _ { 0 } ) { h } _ { i } ^ { ( k ) \top } W _ { k } ( z _ { 0 } ) { h } _ { j } ^ { ( k ) } ~ + ~ b _ { 0 } ( z _ { 0 } ) + c ( h _ { i } ) + c ( h _ { j } ) , \qquad \omega _ { \theta } ( z _ { 0 } , h _ { i } , h _ { j } ) = \sigma ( s _ { i j } ) ,\tag{8}
$$

where $h _ { i } ^ { ( k ) } \in \mathbb { R } ^ { d _ { h } / K _ { h } }$ is the k-th of $K _ { h }$ disjoint chunks of $h _ { i }$ (for example, $d _ { h } = 6 4 , K _ { h } = 8 \colon$ eight dimensions each), $\alpha ( z _ { 0 } )$ are softmax-normalised head weights, $b _ { 0 } \big ( z _ { 0 } \big )$ is a per-graph bias and c is a per-node bias shared by all nodes. The per-node bias and a symmetric pair term are used by the MLP decoder only.

Chunking. The chunks are what make the heads distinct: applied to the full embedding, $\begin{array} { r } { \sum _ { k } \alpha _ { k } \ : h _ { i } ^ { \top } W _ { k } h _ { j } = h _ { i } ^ { \top } \big ( \sum _ { k } \alpha _ { k } W _ { k } \big ) h _ { j } } \end{array}$ is a single bilinear form.

Interaction matrices. Each head uses $W _ { k } ( z _ { 0 } ) = W _ { k } ^ { \mathrm { b a s e } } + W _ { k } ^ { \mathrm { m o d } } ( z _ { 0 } )$ , where $W _ { k } ^ { \mathrm { b a s e } }$ is a symmetric matrix shared across the dataset (normalised by its spectral norm, detached from the gradient) and $W _ { k } ^ { \mathrm { m o d } }$ is predicted from $z _ { 0 } ,$ with $d _ { k } = d _ { h } / K _ { h }$ . Four forms of $W _ { k } ^ { \mathrm { m o d } }$ are available, nested as follows: full, a general symmetric $d _ { k } \times d _ { k }$ matrix; low-rank, $P _ { a } P _ { a } ^ { \top } - \tilde { P } _ { r } P _ { r } ^ { \top }$ with $P _ { a } , P _ { r } \in \mathbb { R } ^ { d _ { k } \times r }$ , which equals full when $r = d _ { k }$ (every symmetric matrix is a difference of two positive semi-definite ones) and allows both assortative and disassortative patterns; diag, the same difference with diagonal factors in learned bases, a cheaper special case of low-rank; and psd, $L L ^ { \top }$ , the special case $P _ { r } = 0$ which allows assortative patterns only. All reported models use low-rank with $r = 4$

## E Latent spaces

## E.1 Per-graph latent variables

![](images/c30d76be47ecbfdcb5244ebec5e4fbd2233a37789b9fce880189980aedc568ca.jpg)  
Figure 5: The latent space of the global variable $z _ { \mathrm { 0 } }$ embeds structural information of MMSBM graphs. The plots show a clear association between the global variables and the graphs’ average degree, clustering coefficient, size and number of communities.

![](images/9a13de7bbeb00640e7246a3fe474086c666426e3d3db49e89df5a9cfd97832db.jpg)  
Figure 6: The latent space of the global variable $z _ { \mathrm { 0 } }$ embeds structural information of Ego graphs. The plots show a clear association between the global variables and the graphs’ average degree, clustering coefficient, size and the degree of the Ego node.

![](images/9070c8d431b93ad582634290feec108274e08a1b293c07b059400d77d9a10fc2.jpg)  
Figure 7: The latent space of the global variable $z _ { 0 }$ embeds structural information of LFR graphs. The plots show a clear association between the global variables and the graphs’ average degree, clustering coefficient, number of communities and mixing parameter.

![](images/ed1ba0354a0bf63f42d06399a62f61fa0068feffcbfc18bbf78e87c974e2c979.jpg)  
Figure 8: The latent space of the global variable $z _ { \mathrm { 0 } }$ embeds structural information of Protein graphs. The plots show a clear association between the global variables and the graphs’ average degree, clustering coefficient, algebraic connectivity and graph size.

## E.2 Per-node latent variables

![](images/972fca7a8da5a80cbf85d569968433c8d7f8f4f54b620b04bdda95f96859cddb.jpg)  
Local latent space — MMSBM (test) each point is one node

![](images/e71e805fa6b504da4e529c8372e390bc752bce49a74986f34293f30317449074.jpg)

![](images/66c6da2aa7868d1b73542fbfd4dc725a5f0c3d3120da5e018c56463c621085ed.jpg)

![](images/6c18c9d8de1fa3451fd60eb099471cad704197d90d4eaef964bed2915bee7c25.jpg)

![](images/969d22aeafeb2948e100aceb8512093be5f1c77a97174596ac18725064655261.jpg)

![](images/7a1be3d960d3e583812262c1bb376bc522968e2e30b99e721ff724a2b5006e35.jpg)  
Figure 9: Top: Three graphs from the test set of MMSBM (spring layout). The nodes are colored by their strongest community membership. Bottom: PCA 2D projection of the latent space for the encoded local variables, demonstrating that the latent space of our xVGAE clusters together nodes with similar membership distribution even though it was only exposed to the adjacency matrix.

![](images/6bef10306caa989d81740137a3064739b870fd8914d255b3cedf62a0e8eec7e5.jpg)  
Local latent space — Ego (test) each point is one node  
Figure 10: Top: Three graphs from the test set of Ego (spring layout). The nodes are colored based on the number of hops separating them from the Ego node, represented with a star. Bottom: PCA 2D projection of the latent space for the encoded local variables, demonstrating that the latent space of our xVGAE maps nodes based on their distance from the Ego node, even though it does not know which node is the Ego.

![](images/5573ed05ccbe363991206dafa7ca7e978666bc5b83686bede6dd836050077334.jpg)

Figure 11: Top: Three graphs from the test set of LFR (spring layout). The nodes are colored by their community membership, showing that the latent space of our xVGAE clusters together nodes with the same community membership and orders them by relative degree, even though it was only exposed to the adjacency matrix. Bottom: PCA 2D projection of the latent space for the encoded local variables, demonstrating that the latent space of our xVGAE clusters together nodes with the same community membership even though it was only exposed to the adjacency matrix.  
![](images/863a061f9e8c3863fb36c7e986765940ea42493498f470ed69924a4a802d7d19.jpg)  
Figure 12: Top: Three graphs from the test set of Protein (spring layout). The nodes are colored by Fiedler vector. Bottom: PCA 2D projection of the latent space for the encoded local variables, demonstrating that the latent space of our xVGAE contains meaningful information given the clustering colors based on the Fiedler vector.

![](images/bf690a69dfde37a93404cb2cdfdf88bcc3c96d7e306e76083a23cf251580c1ad.jpg)  
Figure 13: Examples of generated Ego graphs (spring layout) from our model, main baselines and version of xVGAE without global diffusion.

![](images/1c39ffbb48f4ded2f039cd305e2086ac792eed216727abc465252d0c6736cb73.jpg)  
Figure 14: Examples of generated LFR graphs (spring layout) from our model, main baselines and version of xVGAE without global diffusion.

![](images/1c36cf4032cc739ae7877f9fc31378132af3737d5b8e0f6660d6b17d17c50e8a.jpg)  
Figure 15: Examples of generated MMSBM graphs (spring layout) from our model, main baselines and version of xVGAE without global diffusion.

![](images/643b7cf6f305649fd817cd846f7315b315a52517187ec9c0389750ee8e9302cf.jpg)  
Figure 16: Examples of generated Protein graphs (spring layout) from our model, main baselines and version of xVGAE without global diffusion.