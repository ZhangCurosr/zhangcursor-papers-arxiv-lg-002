# Leaner Transformers Can Easily Learn to Cluster

Charlotte Park†, Kenneth L. Clarkson†, Lior Horesh†, Takuya Ito†, Parikshit Ram†.

†IBM Research, MIT.

Email: parikshit.ram@ibm.com

## Abstract

Transformers have in-context learning capabilities, where some known learning algorithms can be executed in the forward pass through the model. Recent work shows that transformers can exactly perform Lloyd's algorithm for k-means clustering with n points in d dimensions with an embedding size $d _ { \mathsf { e m b } } = d + k$ (thus, requiring attention projection matrices of size $( d + k ) ^ { 2 } )$ . In this work, we build upon this result in the following ways: First, we present an equally expressive but smaller transformer that executes Lloyd's algorithm with embedding size $d _ { \mathsf { e m b } } = ( d + \lceil \log _ { 2 } k \rceil )$ . Next, we train these transformers to learn the clustering algorithms given a distribution of clustering tasks, and theoretically characterize and empirically validate the factors affecting the convergence and in-distribution generalization of learning algorithms based on stochastic gradients. Finally, we probe the general clustering abilities of these learned algorithms (in the form of transformers), and try to understand situations where they succeed and fail.

## 1 Transformers Can Execute Clustering Algorithms In-Context

Clustering is a well-studied problem in computer science, having various applications across many fields, and possessing a nontrivial level of hardness [Hartigan, 1975, Aggarwal and Reddy, 2013]. Within machine learning, clustering can help with tasks such as exploratory data analysis, unsupervised labeling, data compression, and quantization. Given a set of items, such as points in Euclidean space, the goal of k-clustering is to partition the items into k groups. Clustering is categorized through various axes such as discrete vs fuzzy or flat vs hierarchical, and have application-dependent clustering objectives such as k-means, discrete [MacQueen, 1967, Lloyd, 1982] or soft [Dunn, 1974, Bezdek, 2013]; k-medoids [Kaufman, 1990, Ng and Han, 2002]; spectral [Von Luxburg, 2007]; kernelized [Schölkopf et al., 1998, Girolami, 2002, Dhillon et al., 2004]; and variants.

Here we focus on discrete (flat) Euclidean k-means clustering. Given a set of points $S = \{ \pmb { x } _ { 1 } , \ldots , \pmb { x } _ { n } \}$ in a ddimensional Euclidean space, with $\pmb { x } _ { i } \in \mathbb { R } ^ { d }$ for $i \in [ n ] \triangleq \{ 1 , 2 , . . . , n \}$ , the goal is to find k centroids/prototypes $\pmb { c } _ { j } \in \mathbb { R } ^ { d } , j \in [ k ]$ by solving the following problem:

$$
\operatorname* { m i n } _ { \{ \substack { c _ { j } \in \mathbb { R } ^ { d } , j \in [ k ] \} } } \sum _ { i = 1 } ^ { n } \operatorname* { m i n } _ { j \in [ k ] } \left\| \pmb { x } _ { i } - \pmb { c } _ { j } \right\| ^ { 2 } .\tag{1}
$$

This has a discrete non-differentiable objective function, and so is inherently combinatorial [MacQueen, 1967, Lloyd, 1982]. This clustering problem is known to be NP-hard even with k = 2, which highlights its non-trivial hardness [Dasgupta, 2008], and prompts the development of various clustering algorithms [Hartigan, 1975, Aggarwal and Reddy, 2013].

Beyond learning predictive models from data (like a decision tree ensemble to predict credit default or a deep convolutional neural network to label an image), an area of interest in machine learning is the ability to learn algorithms from data. In this line of work, attention-based transformers [Vaswani et al., 2017, Phuong and Hutter, 2022] and recurrent neural networks [Graves et al., 2014, Andrychowicz et al., 2016] have garnered much interest. With transformers, the ability to execute an algorithm is often studied as in-context learning, where a transformer is shown to be able to execute a specific learning algorithm within its forward pass [Garg et al., 2022, Akyürek et al., 2023, Von Oswald et al., 2023, Li et al., 2023]. Given this ability to express a specific (learning) algorithm in-context — thus demonstrating algorithmic expressivity — an important question is how and when such algorithms can be learned [Von Oswald et al., 2023, Li et al., 2023, Ahn et al., 2023, Huang et al., 2023, Zhang et al., 2023, Cheng et al., 2024], and what kind of algorithmic generalization can be expected [Li et al., 2023, Ito et al., 2025]. Recently, Clarkson et al. [2026] established algorithmic expressivity of a transformer with a model embedding dimensionality $d _ { \mathsf { e m b } } = d + k$ for Euclidean clustering, demonstrating that this architecture can exactly execute Lloyd's algorithm for k-means [Lloyd, 1982], and some variants for fuzzy [Dunn, 1974, Bezdek, 2013] and robust clustering [Cuesta-Albertos et al. 1997].

Contributions. We build upon Clarkson et al. [2026] to make the following contributions:

• We present a novel transformer architecture that exactly executes Lloyd's algorithm for k-means with a reduced model embedding size $d _ { \mathsf { e m b } } = d + \lceil \log _ { 2 } k \rceil$ , an up to 75% reduction¹in the number of transformer parameters when $k \approx d ,$ and much larger if $k \gg d ,$

• We study a computationally cheap algorithm to train these transformers to learn to cluster in an end-to-end differentiable manner given a distribution of clustering tasks, and theoretically characterize the various factors affecting its convergence and in-distribution generalization. Specific factors of interest here are the token embedding scheme in the transformer, and the smoothing procedure to optimize the discrete objective in Equation (1).

One insight from this analysis is that, for a fixed number of clusters $k ,$ and a relatively small data dimensionality $d ,$ our proposed smaller transformer with $d _ { \mathsf { e m b } } = d + \lceil \log _ { 2 } k \rceil$ takes longer to converge and generalize compared to the previous transformer with $d _ { \mathsf { e m b } } = d + k$ . However, both eventually converge to the same performance, and this difference in convergence rate vanishes as d grows.

Notation. We denote the index set as $[ m ] \triangleq \{ 1 , 2 , \dots , m \}$ for a positive integer $m \in \mathbb { N }$ We use lowercase letters (such as $x )$ to denote scalars and uppercase letters (such as $L )$ to denote functions, boldface lowercase letters (x) to denote vectors, boldface uppercase (X) to denote matrices, and calligraphic letters (X) to denote sets. The $i ^ { \mathrm { t h } }$ row of a matrix X is $X _ { i , : }$ the $j ^ { \mathrm { t h } }$ column is $X _ { : , j : }$ the first i rows of X is $X _ { : i , : } , I _ { k }$ is the $( k \times k )$ identity matrix. · denotes the $\ell _ { 2 }$ norm for vectors and the spectral norm for matrices. For a set of matrices $\theta = \{ W _ { 1 } , W _ { 2 } , \ldots , W _ { m } \}$ (such as model parameter sets), $\| \theta \| \triangleq \operatorname* { m a x } _ { i \in [ m ] } \| \boldsymbol { W } _ { i } \|$ . For any function $F : \mathcal { X }  \mathcal { Y }$ with learnable parameters $\theta ,$ we write it as $F ( \mathbf { x } ; \theta )$ for any $\mathbf { x } \in \mathcal { X }$ to explicitly note the dependence on the parameters $\theta .$

## 2 Expressing a Clustering Algorithm with a Leaner Transformer

The existing algorithmic expressivity of transformers exactly emulating Lloyd's algorithm [Clarkson et al., 2026] uses three critical components: (i) a procedure to embed the points (to be clustered) and the initial centers (to be refined), (ii) a limiting version of softmax attention with the softmax temperature tending to zero, making the softmax behave like argmax, and (iii) an update based on cross-attention and self-attention, first for the token embeddings of the points, and then for the token embeddings for the cluster centers. For a set of points $S = \{ \pmb { x } _ { i } , i \in [ n ] \} \subset \mathbb { R } ^ { d }$ , represented as $\pmb { X } \in \mathbb { R } ^ { d \times n }$ , and k initial cluster centers $\{ c _ { j } , j \in [ k ] \}$ represented as $C \in \mathbb { R } ^ { d \times k } , X$ and C are embedded in $( d + k )$ dimensions with $E _ { X } ^ { 0 \mathrm { H } } : \mathbb { R } ^ { d \times n }  \mathbb { R } ^ { ( \bar { d + } k ) \times n }$ and $E _ { C } ^ { 0 \tt H } : \mathbb { R } ^ { d \times k }  \mathbb { R } ^ { ( d + k ) \times k }$ as follows:

$$
\bar { C } ^ { ( 0 ) } \gets E _ { C } ^ { 0 \mathrm { H } } ( C ) \triangleq \left[ \frac { C } { I _ { k } } \right] , \quad \bar { X } ^ { ( 0 ) } \gets E _ { X } ^ { 0 \mathrm { H } } ( X ) \triangleq \left[ \frac { X } { Y ^ { ( 0 ) } } \right] , \quad Y ^ { ( 0 ) } = \mathbf { 0 } _ { k \times n } \in \{ 0 , 1 \} ^ { k \times n } ,\tag{OH}
$$

where the OH denotes the one-hot encoding of the cluster indices with $\scriptstyle { I _ { k } }$ . This makes the transformer embedding dimension $d _ { \mathsf { e m b } } = d + k$ . Beginning with $\mathbf { { \cal Y } } ^ { ( 0 ) } = \mathbf { 0 } _ { k \times n } ,$ the all zero $( k \times n )$ matrix, the first set of cross-attention and self-attention updates $\bar { X } ^ { ( t ) } \to \bar { X } ^ { ( t + 1 ) }$ by modifying the $\mathbf { \nabla } \mathbf { \mathbf { { Y } } } ^ { ( t ) }$ part of the embeddings to match the current cluster assignments via Lloyd's algorithm, that ${ \mathrm { i } } \mathbf { s } ,$ each point's embedding is updated to contain the one-hot encoded index of the closest center. The next set of cross and self-attention updates $\bar { C } ^ { ( t ) } \to \bar { C } ^ { ( t + 1 ) }$ by modifying the $C ^ { ( t ) }$ part of the embeddings:

$$
\bar { X } ^ { ( t ) } = \left[ \frac { X } { Y ^ { ( t ) } } \right] \to \bar { X } ^ { ( t + 1 ) } = \left[ \begin{array} { c } { { X } } \\ { { Y ^ { ( t + 1 ) } } } \end{array} \right] , \quad \bar { C } ^ { ( t ) } = \left[ \begin{array} { c } { { C ^ { ( t ) } } } \\ { { I _ { k } } } \end{array} \right] \to \bar { C } ^ { ( t + 1 ) } = \left[ \begin{array} { c } { { C ^ { ( t + 1 ) } } } \\ { { I _ { k } } } \end{array} \right] .\tag{2}
$$

With the $d _ { \mathsf { e m b } } = d + k _ { \mathsf { m } }$ the query/key/value projection matrices for the cross and self-attention are of ${ \cal O } ( d _ { \mathsf { e m b } } { } ^ { 2 } ) \sim { \cal O } ( ( d + k ) ^ { 2 } )$ size. To mitigate the quadratic dependence on $k ,$ we propose the following:

Definition 2.1 (Token embeddings). For points $\pmb { X } \in \mathbb { R } ^ { d \times n }$ and initial centers $C \in \mathbb { R } ^ { d \times k }$ , we use the embedding scheme $E _ { X } ^ { \mathtt { B N } } : \mathbb { R } ^ { d \times n }  \mathbb { R } ^ { ( { \dot { d } } + \lceil \log _ { 2 } k \rceil ) \times n }$ and $E _ { C } ^ { \mathtt { B N } } : \mathbb { R } ^ { d \times k }  \mathbb { R } ^ { ( d + \lceil \log _ { 2 } k \rceil ) \times k }$

$$
\bar { C } ^ { ( 0 ) } \gets E _ { C } ^ { \mathtt { B N } } ( C ) \triangleq \left[ \begin{array} { l } { C } \\ { B _ { k } } \end{array} \right] , \quad \bar { X } ^ { ( 0 ) } \gets E _ { X } ^ { \mathtt { B N } } ( X ) \triangleq \left[ \begin{array} { l } { X } \\ { Y ^ { ( 0 ) } } \end{array} \right] , \quad Y ^ { ( 0 ) } = \mathbf { 0 } _ { \lceil \log _ { 2 } k \rceil \times n } ,\tag{BN}
$$

where $B _ { k } = [ b _ { 1 } ; b _ { 2 } ; . . . ; b _ { k } ] \in \{ 0 , 1 \} ^ { \lceil \log _ { 2 } k \rceil \times k }$ is a binary matrix with the $i ^ { \mathrm { t h } }$ column $b _ { i } \in \{ 0 , 1 \} ^ { \lceil \log _ { 2 } k \rceil }$ as the binary representation of the integer (i – 1) with $\lceil \log _ { 2 } k \rceil$ bits for each $i \in [ k ]$

The main intuition behind this BN (binary) embedding scheme is that, while the (0H) token embedding [Clarkson et al., 2026] uses one-hot vectors to represent a cluster index (thus requiring k dimensions), we represent that index more compactly with a binary representation using only $\lceil \log _ { 2 } k \rceil$ dimensions. As an example, when $k = 6 , B _ { 6 } = \left[ \begin{array} { l } { 0 , \overline { { 1 } } , 0 , 1 , \overline { { 0 } } , 1 } \\ { 0 , 0 , 1 , 1 , 0 , 0 } \\ { 0 , 0 , 0 , 0 , 1 , 1 } \end{array} \right] \in \{ 0 , 1 \} ^ { 3 \times 6 }$ . Then we use the following softmax attention with negative squared Euclidean distances:

Definition 2.2 (Attention). Given queries $\boldsymbol { Z } \in \mathbb { R } ^ { d _ { \mathrm { e m b } } \times q }$ and keys $Z ^ { \prime } \in \mathbb { R } ^ { d _ { \mathrm { e m b } } \times r }$ , we define attention as:

$$
\mathcal { A } _ { 2 } ^ { \gamma } ( Z , Z ^ { \prime } ; ( Q , K , V ) ) = V Z ^ { \prime } \mathrm { s o f t m a x } \left( - \frac { \gamma } { \sqrt { d _ { \mathrm { e m b } } } } \mathrm { s q d i s t } ( Q Z , K Z ^ { \prime } ) \right) ,\tag{3}
$$

where $Q , K , V \in \mathbb { R } ^ { d _ { \mathrm { e m b } } \times d _ { \mathrm { e m b } } }$ are the query, key and value projection matrices respectively, $\gamma$ is the inverse temperature of the columnwise softmax operation and sqdist : $\mathbb { R } ^ { d _ { \mathrm { e m b } } \times q } \times \mathbb { R } ^ { d _ { \mathrm { e m b } } \times r }  \mathbb { R } ^ { r \times q }$ computes the pairwise squared Euclidean distances with the $( i , j ) \ – \ t h$ entry of its output given by sqdist $( \mathcal { Z } , \mathcal { Z } ^ { \prime } ) _ { i j } = \left\| \mathcal { Z } _ { : , j } - \mathcal { Z } _ { : , i } ^ { \prime } \right\| ^ { 2 }$ for some matrices $\mathcal { Z } \in \mathbb { R } ^ { d _ { \mathsf { e m b } } \times q } , \mathcal { Z } ^ { \prime } \in \mathbb { R } ^ { d _ { \mathsf { e m b } } \times r }$

This distance-based attention [Tsai et al., 2019] was used to establish algorithmic expressivity for clustering, though this can be replaced with dot-product attention if we also utilize a single-hidden-layer feedforward network in the transformer [Clarkson et al., 2026]. Based on the above, we consider the following transformer updates:

Definition 2.3 (Per-layer updates). In the $( t { + } 1 ) ^ { \mathrm { t h } }$ transformer layer, with parameters $\Theta = \{ ( Q _ { \mu } , K _ { \mu } , V _ { \mu } ) , \mu =$ $1 , \ldots , 4 \}$ shared across all layers, the layer input token embeddings $( \bar { X } ^ { ( t ) } , \bar { C } ^ { ( t ) } )$ are updated with the attention operation in Definition 2.2 as follows:

$$
\bar { X } ^ { ( t + 1 ) } = \bar { X } ^ { ( t ) } + \ A _ { 2 } ^ { \gamma } ( \bar { X } ^ { ( t ) } , \bar { C } ^ { ( t ) } ; ( Q _ { 1 } , K _ { 1 } , V _ { 1 } ) ) + A _ { 2 } ^ { \gamma } ( \bar { X } ^ { ( t ) } , \bar { X } ^ { ( t ) } ; ( Q _ { 2 } , K _ { 2 } , V _ { 2 } ) ) ,
$$

$$
\bar { C } ^ { ( t + 1 ) } = \bar { C } ^ { ( t ) } + \ A _ { 2 } ^ { \gamma } ( \bar { C } ^ { ( t ) } , \bar { X } ^ { ( t + 1 ) } ; ( Q _ { 3 } , K _ { 3 } , V _ { 3 } ) ) \vert + A _ { 2 } ^ { \gamma } ( \bar { C } ^ { ( t ) } , \bar { C } ^ { ( t ) } ; ( Q _ { 4 } , K _ { 4 } , V _ { 4 } ) ) .\tag{4}
$$

where the first operation on the right-hand-side after the residual connectionin each of the update steps corresponds tocross-attention, while the second operation isself-attention

Given this architecture, we can show the following algorithmic expressivity result:

![](images/d42a60b4a9e3b4c9abe1cfefa12a2dd03d81b44fa437cfde5f5a45b77f7349b3.jpg)  
Figure 1: Comparison of the proposed (BN) embedding against the existing (0H) embedding with varying values of $d \in \{ 4 , 8 , 1 6 , 3 2 , 6 4 , 1 2 8 \}$ and $k \in \{ 1 0 , 1 6 , 2 5 , 4 0 , 6 4 , 1 0 0 \}$ . Each heatmap reports the relative improvement (in %) of (BN) over (0H) for each of following metrics: (i) Forward time (train): forward pass time during training, (ii) Backward time (train): backward pass time, (iv) Encoding specific GPU memory usage: peak GPU memory usage at training (less the usage from the self-attention matrices which does not depend on $d _ { \mathsf { e m b } } )$ , and (iv) Forward pass (eval): forward pass time with the computation graph disabled (via model.eval()). Each cell in the $( 6 \times 6 )$ grid in the heatmap lists the relative gain (or degradation), with actual metrics for (0H) also listed in the cell as (mean±std) for completeness; darker greens denote larger gain values. All runs use 32 clustering problems with $n = 1 0 2 4$ points, and numbers are generated with 30 repetitions (10 warm-up rounds). In terms of memory usage, (BN) shows gains across the board, ranging from +1-15%. For the computation times (forward (train), backward, and forward (eval)), (BN) does not always show a positive gain for smaller values of $k ,$ especially when the baseline (0H) runtimes is already quite small (< 10ms). However, once k is large enough, the gains can be quite significant, reaching over 50% in some cases.

Theorem 2.1. Given a set of points $\pmb { X } \in \mathbb { R } ^ { d \times n }$ to be clustered into k disjoint sets and initial cluster centers $C \in \mathbb { R } ^ { d \times k }$ , both embedded with $( E _ { C } ^ { \tt B N } , E _ { X } ^ { \tt B N } ) ( D e f n i t i o n \ \mathcal { Q . I } )$ to give us a latent dimension $d _ { \mathsf { e m b } } = d + \lceil \log _ { 2 } k \rceil$ , there exists parameters $\Theta = \{ ( Q _ { \mu } , K _ { \mu } , V _ { \mu } ) , \mu = 1 , \ldots , 4 \}$ for the transformer (Definition 2.3) with the inverse softmax temperature $\gamma = \infty$ such that the $t ^ { t h }$ transformer layer output exactly matches the output of Lloyd's algorithm after t iterations for any $t \geq 1$

The precise Lloyd's algorithm is presented in Algorithm 2. The proof follows a construction similar to that of Clarkson et al. [2026] with modifications to handle the binary version of the cluster indices instead of the one-hot version. See Appendix B for details and the proof. This architecture now requires $O ( ( d + \log _ { 2 } k ) ^ { 2 } )$ parameters reducing the dependence on k from $\Omega ( k ^ { 2 } )$ to $O ( ( \log _ { 2 } k ) ^ { 2 } )$ . While the number of parameters is not really a factor in the expressivity results, it can affect computational cost of the inference (which in turn affects the training time). Figure 1 compares the computational costs (in terms of forward pass time, backward pass time, GPU memory usage) of the models with $d _ { \mathsf { e m b } } = d + k$ for (oí) and our proposed $d _ { \mathsf { e m b } } = d + \lceil \log _ { 2 } k \rceil$ for (BN) for varying values of (d, k). While the advantage of the proposed model is clear across the board in terms of memory consumption, the gain in terms of forward/backward pass time is only significant when values of d and k are large enough. Table 1 profiles a different setup where we fix d and vary $k ,$ and see that, once k is sufficiently larger than $d ,$ the gains from the proposed (BN) can be very significant even in terms of the forward and backward pass runtimes.

Table 1: Comparison between proposed (BN) and existing (0H) under the same measurement setting as in Figure 1 but where k is varied in the range [6, 1000] (column 1) for a fixed data dimensionality $d = 3 2$ We note the actual forward (column 3) and backward pass (column 4) times (and the speedup of (BN) over (0H)), and the peak GPU memory usage (column 5) as described in Figure 1 (and the reduction from using (BN)). We also note the additional embedding dimensions $( d _ { \mathsf { e m b } } - d )$ for each value of k for the two embeddings (column 2), and the significant reduction in the additional embedding dimensions lead to the significant reductions in the runtimes once k is sufficiently larger than d with up to $1 2 \times / 9 \times$ speedup in forward/backward pass, and almost 70% reduction in memory usage.
<table><tr><td>k</td><td> $( d _ { \mathrm { e m b } } - d )$  OH/BN</td><td>Forward pass (ms) OH / BN [speedup]</td><td>Backward pass (ms) OH / BN [speedup]</td><td>GPU memory usage (MB) OH /BN [reduction]</td></tr><tr><td>6</td><td>6/3</td><td>4.25 / 4.83 [0.88×]</td><td>9.53 / 10.66 [0.89×]</td><td>419.93 / 416.29 [0.9%]</td></tr><tr><td>25</td><td>25/5</td><td>5.24 /4.99 [1.05×]</td><td>11.21 /11.01 [1.02×]</td><td>441.08 / 423.23 [4.0%]</td></tr><tr><td>100</td><td>100/7</td><td>9.76 /4.52 [2.16×]</td><td>20.21 / 9.93 [2.03×]</td><td>532.96 / 447.04 [16.1%]</td></tr><tr><td>400</td><td>400/9</td><td>38.68 / 7.89 [4.90×]</td><td>66.13 / 16.47 [4.02×]</td><td>1095.60 / 531.02 [51.5%]</td></tr><tr><td>1000</td><td>1000/10</td><td>193.07 /15.90 [12.14×]</td><td>308.28 / 32.33 [9.54×]</td><td>3930.34 / 1207.75 [69.3%]</td></tr></table>

## 3 Understanding Learning Convergence and Generalization

Given the above architecture, we will train the transformers to learn a clustering algorithm given a distribution of clustering tasks. In any learning setup, we need training data, a model defining the "forward pass" or inference, and a loss that quantifies the difference between the (current) model output and the desired one. As we aim to learn a clustering algorithm, the training data will be a set of k-clustering tasks in d dimensions, each with a pair $( X , C )$ of points to be clustered $\pmb { X } \in \mathbb { R } ^ { d \times n }$ and initial cluster centers $C \in \mathbb { R } ^ { d \times k }$ , generated from a distribution D of clustering tasks (for example, each X is randomly sampled from a different mixture of Gaussians, and C is random k-subset of X). The model $F _ { d : k } ^ { \gamma } : \mathbb { R } ^ { d \times n } \times \mathbb { R } ^ { d \times k }  \mathbb { R } ^ { d \times k }$ is a single-layer version of the one presented in Definition 2.1, 2.2 and 2.3 with a finite $\gamma$ (in all our experiments, we set $\gamma = 1 )$ , and parameterized with $\Theta = \{ ( Q _ { \mu } , K _ { \mu } , V _ { \mu } ) , \mu = 1 , \ldots , 4 \}$

For a clustering task $( X , C )$ , the initial token embeddings $( \bar { \pmb X } ^ { ( 0 ) } , \bar { \pmb C } ^ { ( 0 ) } )$ will be generated using embedding functions $\left( E _ { C } , E _ { X } \right)$ - either with $( E _ { C } ^ { 0 \mathrm { H } } , E _ { X } ^ { 0 \mathrm { H } } )$ from (0H) or $( E _ { C } ^ { { \tt B N } } , E _ { X } ^ { { \tt B N } } )$ from (BN) - and passed through a single transformer layer to get $( \bar { \pmb X } ^ { ( 1 ) } , \bar { \pmb C } ^ { ( 1 ) } )$ , and the model output $F _ { d : k } ^ { \gamma } ( X , C ; \Theta )$ is the first d rows $\bar { C } _ { : d , : } ^ { ( 1 ) }$ of $\bar { C } ^ { ( 1 ) }$ — this choice implies that we are training the model to perform

Algorithm 1 Learning to $k -$   
cluster in d dimensions using a   
transformer $F _ { d : k } ^ { \gamma }$ with learnable   
parameters Θ, given a distribu  
tion D of clustering tasks, an   
min-regularizer $\Omega : { \Delta } ^ { k }  \mathbb { R }$   
with penalty $\tau > 0 ,$ and op  
timization hyperparameters of   
learning rate $\eta ,$ task batch size   
B and M training steps.   
1 Initialize $F _ { d : k } ^ { \gamma }$ parameters Θ   
2 for $m = 1 , \ldots , M$ do   
// Compute batch loss   
3 Initialize batch loss $l ( \Theta ) ~ = ~ 0$   
4 for $b = 1 , \dots , B$ do   
5 Sample task (X, C) ∼ D   
6 $l ( \Theta ) \ + = \ \tilde { \mathcal { L } } _ { \Omega } ^ { \tau } ( X , F _ { d : k } ^ { \gamma } ( X , C ; \Theta ) )$   
// update model params   
$\Theta  \tt U p d a t e ( \Theta , \nabla _ { \Theta } \it l ( \Theta ) , \eta )$   
8 return Θ

single-step clustering. Note that we need to learn a separate model for each value of d and k (denoted by the d:k subscript in $F _ { d : k } ^ { \gamma } )$ ; the same model can be applied to clustering tasks with different values of $n .$

Given the model output $\bar { C } ( \Theta ) \triangleq \operatorname { F } _ { d : k } ^ { \gamma } ( X , C ; \Theta )$ , we can compute the per-task clustering objective $\mathcal { L } : \mathbb { R } ^ { d \times n } \times \mathbb { R } ^ { d \times k }  \mathbb { R }$ using the per-point clustering objective $L : \mathbb { R } ^ { d } \times \mathbb { R } ^ { d \times k }  \mathbb { R }$ as

$$
\mathcal { L } ( X , F _ { d : k } ^ { \gamma } ( X , C ; \Theta ) ) = \sum _ { i = 1 } ^ { n } L ( X _ { : , i } , F _ { d : k } ^ { \gamma } ( X , C ; \Theta ) ) .\tag{5}
$$

As a concrete example, we will use the discrete k-means objective $\begin{array} { r } { L ( \boldsymbol X _ { : , i } , \boldsymbol { \bar { C } } ( \boldsymbol \Theta ) ) = \operatorname* { m i n } _ { j \in [ k ] } \left. \boldsymbol X _ { : , i } - \boldsymbol { \bar { C } } ( \boldsymbol \Theta ) _ { : , j } \right. ^ { 2 } } \end{array}$ With in-context supervised learning, the learning loss, such as mean-squared-error or negative cross-entropy, is continuous and thus differentiable, while the k-means objective in Equation (5) is not. Hence, we will consider a smoothed surrogate objective. While min is not a differentiable function, various regularized forms of min, with a convex regularizer $\Omega : \Delta ^ { k } $ R and a penalty $\tau > 0$ , are differentiable, and give us an upper-bound for the per-point k-means objective $\tilde { L } _ { \Omega } ^ { \tau } : \mathbb { R } ^ { d } \times \mathbb { R } ^ { d \times k }  \mathbb { R }$ of the following form (see Appendix H) for any $\pmb { x } \in \mathbb { R } ^ { d }$ and cluster centers $C \in \mathbb { R } ^ { d \times k }$ with simplex weights $\{ \lambda _ { j } ^ { \Omega _ { \tau } } \geq 0 \} _ { j \in [ k ] }$ where $\begin{array} { r } { \dot { \sum _ { j = 1 } ^ { k } } \lambda _ { j } ^ { \top } = 1 \dag , } \end{array}$

$$
L ( { \pmb x } , C ) \triangleq \operatorname* { m i n } _ { j \in [ k ] } \| { \pmb x } - { \pmb C } _ { : , j } \| ^ { 2 } \leq \sum _ { j = 1 } ^ { k } \lambda _ { j } ^ { \Omega _ { \tau } } \| { \pmb x } - { \pmb C } _ { : , j } \| ^ { 2 } \triangleq \widetilde L _ { \Omega } ^ { \tau } ( { \pmb x } , C ) ,
$$

Here, $\tau = 0$ implies no regularization (and thus the exact minimum). While a common choice for the weights $\{ \lambda _ { j } ^ { \Omega _ { \tau } } \} _ { j \in [ k ] }$ is the softmax function (obtained by a negative Shannon entropy or NE-regularized min, with τ serving as the softmax temperature), a tighter bound can be obtained with the sparsemax function which uses the L2-norm regularization leading to sparse weights [Martins and Astudillo, 2016, Blondel et al., 2020]

With this smoothed loss surrogate $\tilde { L } _ { \Omega } ^ { \tau } ,$ we have $L ( X _ { : , i } , \bar { C } ( \Theta ) ) \leq \tilde { L } _ { \Omega } ^ { \tau } ( X _ { : , i } , \bar { C } ( \Theta ) )$ ; equality holds at $\tau = 0$ However, note that as $\tau \to 0 , \tilde { L } _ { \Omega } ^ { \tau }$ becomes progressively less smooth (Proposition H.2 in Appendix H). Given this continuous objective, we train the transformer by solving the following

$$
\operatorname* { m i n } _ { \Theta } \mathbb { E } _ { ( X , C ) \sim \mathcal { D } } \tilde { \mathcal { L } } _ { \Omega } ^ { \tau } ( X , F _ { d : k } ^ { \gamma } ( X , C ; \Theta ) ) , \quad \mathrm { w h e r e } \quad \tilde { \mathcal { L } } _ { \Omega } ^ { \tau } ( X , \bar { C } ( \Theta ) ) \triangleq \sum _ { i = 1 } ^ { n } \tilde { L } _ { \Omega } ^ { \tau } ( X _ { : i } , \bar { C } ( \Theta ) ) .\tag{6}
$$

In practice we will utilize a finite approximation of the above expectation $\mathbb { E } _ { ( X , C ) \sim \mathcal { D } }$ by sampling clustering tasks from $\mathcal { D } .$ The training procedure is detailed in Algorithm 1. Note that while we focus on the discrete k-means objective, this learning scheme is applicable to any discrete clustering objective L for which we can find a reasonably smoothed surrogate objective $\tilde { L }$ such that $L \leq \tilde { L }$ . In each training step, we sample a batch of clustering tasks, compute the surrogate loss $\tilde { \mathcal { L } } _ { \Omega } ^ { \tau }$ of the model outputs $F _ { d : k } ^ { \gamma } ( X , C ; \Theta )$ , and use the batch-loss-gradient to update the model.

We build upon existing analysis to discuss the effect of different choices in the learning procedure (namely the regularizer $\Omega ,$ the penalty $\tau ,$ and the token embedding) on the training convergence. For any α-Lipschitz and β-smooth learning loss, various results establish guarantees for SGD (stochastic gradient descent) based learning Bousquet and Elisseeff, 2000, Hardt et al., 2016]. For a finite-sum nonconvex objective, such as the one in Equation (6), M steps of SGD with a time-dependent learning rate $\eta _ { m } , m \in [ M ]$ converges to €-stationarity with

$$
\begin{array}{c} \epsilon \sim O \left( \beta \alpha ^ { 2 } \frac { \sum _ { m = 1 } ^ { M } \eta _ { m } ^ { 2 } } { \sum _ { m = 1 } ^ { M } \eta _ { m } } \right) , \quad \mathrm { w i t h } \ \eta _ { m } = \left\{ \eta / m \right. \quad \Rightarrow \epsilon \sim O ( \beta \alpha ^ { 2 } / \log M )  \\ { \eta / \sqrt { m } \quad \Rightarrow \epsilon \sim O ( \beta \alpha ^ { 2 } \log M / \sqrt { M } ) } \end{array} ,\tag{7}
$$

for some initial learning rate $\eta > 0$ . The $( \alpha , \beta )$ properties of the loss function, along with the learning rate scheduling, drive the convergence rate guarantees. However, $( \alpha , \beta )$ also affect generalization. Given a set $s$ of clustering tasks sampled from ${ \mathcal { D } } ,$ model parameters Θ learned with M steps of SGD using a time dependent learning rate $\eta _ { m } = \eta / m _ { ; }$ ,the generalization error (the gap between the true and estimated loss) can be bounded by combining Hardt et al. [2016, Theorems 2.2 and 3.12]:

$$
\left| \mathbb { E } _ { \mathcal { S } } \Big [ \mathbb { E } _ { ( \pmb { X } , C ) \sim \mathcal { D } } \tilde { \mathcal { L } } _ { \Omega } ^ { \tau } ( \pmb { X } , \bar { C } ( \Theta ) ) - \frac { 1 } { | \mathcal { S } | } \sum _ { ( \pmb { X } , C ) \in \mathcal { S } } \tilde { \mathcal { L } } _ { \Omega } ^ { \tau } ( \pmb { X } , \bar { C } ( \Theta ) ) \Big ] \right| \leq \underbrace { \frac { 1 + \frac { 1 } { \beta \eta } } { | \mathcal { S } | } ( 2 \eta \alpha ^ { 2 } ) ^ { \frac { 1 } { \beta \eta + 1 } } M ^ { \frac { \beta \eta } { \beta \eta + 1 } } } _ { : \lfloor \pmb { \mathscr { S } } \rfloor } .\tag{8}
$$

The above provides a bound on the smoothed objective $\tilde { \mathcal { L } } _ { \Omega } ^ { \tau }$ . With $\varepsilon$ denoting the above gap bound, we have the following in-distribution generalization bound for the non-differentiable k-means objective:

$$
\mathbb { E } _ { \mathcal { S } } \Big [ \mathbb { E } _ { ( \pmb { X } , \pmb { C } ) \sim D } \mathcal { L } \big ( \pmb { X } , \vec { F } _ { d : k } ^ { \gamma } ( \pmb { X } , \pmb { C } ; \Theta ) \big ) \Big ] \le \mathbb { E } _ { \mathcal { S } } \Big [ \frac { 1 } { | \mathcal { S } | } \sum _ { ( \pmb { X } , \pmb { C } ) \in \mathcal { S } } \mathcal { L } \big ( \pmb { X } , \vec { F } _ { d : k } ^ { \gamma } ( \pmb { X } , \pmb { C } ; \Theta ) \big ) \Big ] + C _ { \Omega } ^ { \tau } + \varepsilon ,\tag{9}
$$

where $0 \le C _ { \Omega } ^ { \tau } \triangleq \operatorname* { m a x } _ { ( X , \hat { C } ) } \tilde { \mathcal { L } } _ { \Omega } ^ { \tau } ( X , \hat { C } ) - \mathcal { L } ( X , \hat { C } )$ quantifies the gap between the true non-differentiable objective $\mathcal { L }$ and smoothed $\tilde { \mathcal { L } } _ { \Omega } ^ { \tau } .$ with smaller $C _ { \Omega } ^ { \tau }$ implying tighter generalization bounds, and can be controlled with $\tau .$ Note that, under certain conditions, some regularized forms of min can ensure that this gap $C _ { \Omega } ^ { \tau } = 0 ;$ as an example, if there is usually a sufficiently large margin between the inter-cluster distances and intra-cluster distances, the sparsemax based upperbound is tight. If this gap $C _ { \Omega } ^ { \tau }$ is large (relative to the other terms), the guarantee is weak. Thus, it is important to ensure that this gap $C _ { \Omega } ^ { \tau }$ is sufficiently small by using a tighter smoothed upper-bound of the clustering loss. We will evaluate its effect empirically. However, the penalty τ also plays a role in the Lipschitz constant $\alpha$ of the smoothed $\tilde { \mathcal { L } } _ { \Omega } ^ { \tau }$ as we show in the following:

Theorem 3.1 (informal, see Theorem I.2 in Appendix I for a detailed version). Assume bounded points $\pmb { X } \in [ 0 , 1 ] ^ { d \times n }$ and initial cluster centers $C \in [ 0 , 1 ] ^ { d \times k }$ for all clustering tasks $( \pmb { X } , \pmb { C } ) \sim \mathcal { D } _ { : }$ , and bounded learnable model parameters $\Theta , \Theta ^ { \prime }$ . Let $d _ { E }$ denote the maximum squared norm of the initial center embeddings, with $\left. \bar { C } _ { : , j } ^ { ( 0 ) } \right. ^ { 2 } \leq d _ { E }$ for all $j \in [ k ]$ for any $^ { C , }$ and let the Ω-regularized argmax with penalty $\tau > 0$ be $\left( \omega _ { \tau } / \tau \right)$ -input-stable. Then the smoothed clustering loss $\tilde { \mathcal { L } } _ { \Omega } ^ { \tau }$ is α-Lipschitz as follows:

$$
\begin{array} { r l } & { \left| \tilde { \mathcal { L } } _ { \Omega } ^ { \tau } ( \boldsymbol { X } , \boldsymbol { F } _ { d : k } ^ { \gamma } ( \boldsymbol { X } , \boldsymbol { C } ; \Theta ) ) - \tilde { \mathcal { L } } _ { \Omega } ^ { \tau } ( \boldsymbol { X } , \boldsymbol { F } _ { d : k } ^ { \gamma } ( \boldsymbol { X } , \boldsymbol { C } ; \Theta ^ { \prime } ) ) \right| \leq \ \alpha \ \left\| \Theta - \Theta ^ { \prime } \right\| , } \\ & { \ u h e r e \ \alpha \sim O \left( \frac { k n \omega _ { \tau } } { \tau } \cdot \operatorname* { m a x } \{ 1 , \gamma ^ { 2 } \} \cdot \sqrt { d _ { E } } \cdot \operatorname* { m a x } \left\{ 1 , \left( \frac { d _ { E } } { \sqrt { d _ { \mathrm { e m b } } } } \right) ^ { 2 } \right\} \cdot \exp \left( \gamma \frac { d _ { E } } { \sqrt { d _ { \mathrm { e m b } } } } \right) \right) } \end{array}\tag{10}
$$

This result highlights how the Lipschitz constant α of the smoothed loss $\tilde { \mathcal { L } } _ { \Omega } ^ { \tau }$ depends on (i) the regularized min penalty τ as $1 / \tau , ( \mathrm { i i } )$ the regularizer Ω through $\omega _ { \tau } ,$ (iii) the attention inverse temperature $\gamma ,$ and (iv) the token embedding scheme through the $d _ { E }$ and $d _ { \mathsf { e m b } }$ terms, where $d _ { E } = d + 1 , d _ { \mathrm { e m b } } = d + k$ with (OH), and $d _ { E } = d _ { \mathsf { e m b } } = d + \lceil \log _ { 2 } k \rceil$ with (BN). Thus, the (0H) scheme of Clarkson et al. [2026] has a better Lipschitz constant than the proposed (BN), thereby presenting a tradeoff between fewer parameters and improved convergence. However, for large d (relative to $k )$ , this difference is limited as $d _ { \mathsf { e m b } }$ and $d _ { E }$ scale as d for both token embeddings. This result in conjunction with those in Section 2 (Figure 1 and Table 1) highlight a tradeoff between the architecture size and the actual training times — the faster forward/backward passes of the leaner architecture with (BN) token embeddings do not necessarily translate to faster model training to convergence

To understand how this theoretical characterization aligns with empirical behaviour, we train this model

![](images/84ebd18e59b24f0ea769cdd105047fb17fdf121546289a6942cdfa3fd714adf0.jpg)

![](images/cea576bbafce2ab1bb270cf4049b2af6065e1fc4dfd14bad3135709fb0e3f28c.jpg)  
(a) Varying $\tau \in \{ 1 , 0 . 1 \}$ with d = 4 dimensions.

![](images/7b02f2b4dbc1fcc4815890346ee394ab8c168401645855fca9712a649b3d613f.jpg)

![](images/1e8b70baf43b18b29edadddb746e48cdca3b60b9397e517709e333f7169bb53c.jpg)  
(b) Varying $d \in \{ 4 , 1 6 \}$ with $\tau = 0 . 2 5$

Figure 2: Effect of data dimensionality $d ,$ min-regularizer Ω with penalty τ, token embedding $( E _ { C } , E _ { X } )$ on training convergence and generalization. We show the convergence of (i) the smoothed training loss $\tilde { \mathcal { L } } _ { \Omega } ^ { \tau }$ (top row in each figure) and (ii) the relative non-differentiable clustering loss L on held-out validation clustering tasks (bottom row) for multiple configurations; the dotted black line at $y = 1$ (bottom row) corresponds to the performance of a single Lloyd's iteration on the validation tasks. See Appendix F for details and ablations. In all cases, lower values on the vertical axes are better. Here, we use clustering tasks with $n = 5 1 2$ and $k = 6$ The initial learning rate for Adam is 0.01 (ablation in Figure 11) with no gradient clipping. In each plot, all 6 curves correspond to the same set of clustering tasks used for training, and we study the behaviour of the NE-regularized min (softmax) vs the L2-regularized min (sparsemax), jointly with the effect of $( E _ { C } , E _ { X } ) - ( 0 \mathrm { H } )$ vs (BN) vs no token embedding (NA). In Figure 2a, we fix the data dimensionality $d = 4$ and vary the min-regularization penalty τ (ablation in Figure 10), and in Figure 2b, we fix $\tau = 0 . 2 5$ and vary d (ablation in Figure 14).

![](images/17164e616d74385726816103b7d707d73fe02f95dcd5eade56d0ee90e6501b6a.jpg)  
(a) d = 4, 0H

![](images/3a65383d52777a1b771e4d53e58283317c3ba41abca213861c47d2d56a79cbf8.jpg)  
(b) d = 4, BN

![](images/d74d400c346e2449638d38852f26403bfc5f6e896b925981f704b3785f9f0ee0.jpg)  
(c) d = 32, OH

![](images/95dbbcace79886c271a6f6854fbe1b01f6486deffa805f85f95a64678594adaa.jpg)  
(d) d = 32, BN  
Figure 3: Loss surfaces for different dimensionalities $( d = 4 \mathrm { { \bf ~ v s ~ } } d = 3 2 )$ , token embeddings (0H) vs (BN) with the L2-regularized min. We fix $k = 6$ and vary d to visualize its effect. We use the procedure in Li et al. [2018] with filterwise normalization and 2 random vectors in the parameter space.

with various configurations. The training clustering tasks contains points sampled from a mixture of isotropic normal distributions (data generation detailed in Appendix D). All experiments are repeated with 10 different random seeds (which affect model initialization, and the tasks sampled), and we aggregate the results over these 10 trials, and present the median (the curve) and the inter-quartile ranges (the translucent ribbon around the curve) of the reported metrics. We use the Adam optimizer with a learning rate of $\eta = 0 . 0 1$ for $M = 1 0 0 0 0$ steps with a task batch size $B = 3 2$ . Note that we do not utilize the standard practice of gradient clipping in our experiments to highlight the ease of training the transformer for clustering. Extended ablation of experimental configurations are presented in Appendix F.

Figure 2 shows the training convergence and validation performance. In almost all cases, the training loss converges quite monotonically, and the validation performance usually tracks the training. In Figure 2a, we vary the temperature τ of the smoothed loss $\tilde { \mathcal { L } } _ { \Omega } ^ { \tau } { : }$ and observe that for any value of $\tau ,$ the L2-regularized min always produces lower training losses, and thus tighter upper-bounds (and lower values of C). The worst upper-bounds with the NE-regularizer are reflected in the validation performance at $\tau = 1$ , where the learned model does not perform well on the unseen validation clustering tasks for any of the token embedding schemes (the validation loss increases during training). In contrast, the L2-regularization is able to reduce the clustering loss on unseen tasks as training progresses, even though it is unable to match Lloyd's at $\tau = 1$ . This highlights the effect of the tightness of the smoothed upper-bound discussed in Equation (9) $\mathrm { A t } ~ \tau = 0 . 1$ , both forms of regularized min (NE and L2) produce tighter bounds, and the learned model is able to significantly outperform Lloyd's for single-step clustering on the unseen validation tasks except with the NA embedding scheme (which does show reduction in validation loss though is unable to match Lloyd's). However, training converges faster for higher values of $\tau$ as implied by Equation (10) though it appears to converge to a worse solution. Thus lowering $\tau$ to tighten the upper-bound can slow down training convergence. Finally, for all values of $\tau ,$ the training converges faster with (0H) compared to (BN), though both finally converge to a similar loss and are able to outperform single-step Lloyd's. This effect is visible for both L2-regularized and NE-regularized min for $\tau = 0 . 1$ . This aligns with Equation (10), where (OH) has a smaller Lipschitz constant than (BN). In Figure $^ { \mathrm { 2 b , } }$ we vary the data dimensionality $d ,$ and see that the results continue to highlight the expressivity of (OH) and (BN) (and the lack thereof with NA), and the tighter upper-bounds of the L2-regularized min as discussed for Figure 2a. We also see that, for a fixed $\tau = 0 . 2 5$ (and controlling for other factors), increasing d removes the difference between the convergence of (OH) vs (BN). We compare the convergence in terms of actual training times in Figure 17 in Appendix F to understand the interplay of the faster forward/backward pass and slower convergence of the proposed (BN).

![](images/cebcb268439ec1d846a81566aaa310a6ed5c5543d2b3ac19d8be3051d625e46c.jpg)  
(a) τ = 1.

![](images/357fce70978b6fd252802f226ac4e5b23e512390c520fec372588df1f21c6814.jpg)  
(b) $\tau = 0 . 1 .$  
Figure 4: Effect of data distribution. The above figures are as described in Figure 2. We train on tasks with $n = 5 1 2 , d = 3 2 , k = 1 0$ generated from the noted distributions. The validation loss is computed on clustering tasks generated from the training distribution (see ablation in Figure 16).

We visualize the loss surfaces [Li et al., 2018] for the L2- regularized smoothed loss in Figure 3, and see that in all cases, the loss surfaces appear fairly unimodal without any complicated landscapes, explaining why these models train so easily with apparent monotonic convergence. For the smaller $d = 4 ,$ , the loss surfaces of (OH) and (BN) are markedly different aligning with the significant difference in their $d _ { E } , d _ { \mathrm { e m b } }$ terms in Theorem 3.1. For the larger $d =$ 32, the loss surfaces of 0H and BN are quite similar, as Theorem 3.1 suggests given the smaller relative differences between their $d _ { E } , d _ { \mathrm { e m b } }$ terms.

In Figure 4, we use clustering tasks with the points X generated from a mixture of isotropic distributions, and con-

![](images/090f8e67835c70bfde955a82084c697c835db88a5a337d87ad1d45aa0f656bbf.jpg)  
(a) normal

![](images/fdd0335f34a4cab44c1ac329555b1a3e8bbc070c94dbb7b61dbc6f4be464bee3.jpg)  
(b) cauchy  
Figure 5: Training loss gap surfaces. We visualize the landscape of the loss gap $( \tilde { \mathcal { L } } _ { \Omega } ^ { \tau } - \mathcal { L } )$ between the smoothed $\tilde { \mathcal { L } } _ { \Omega } ^ { \tau }$ and the k-means loss $\mathcal { L }$ at $\tau = 0 . 1$

sider 5 different distribution families: cauchy, gumbel, laplace, lognormal, normal. We evaluate two values of τ. In all cases, the models are trained and evaluated on the same family. First, we see that our proposed training scheme is unable to generalize when trained with data from the cauchy distribution — the training seems to converge very quickly, but the performance on unseen clustering tasks are significantly worse than that of the single Lloyd's iteration. For the remaining families with the larger $\tau = 1$ , we see that the L2-regularized min with the (0H) and (BN) token embeddings generalize well, and significantly outperform a single Lloyd's iteration; the NE-regularization is unable to match a single Lloyd's iteration, highlighting the benefit of the tighter loss upper-bound. When $\tau = 0 . 1$ , both NE-regularized and L2-regularized min generalize equally well. Overall, we see that we learn generalizing transformers for all considered distributions except with cauchy; this failure remains an open question. We hypothesize that it is related to the larger gap between the smoothed upper-bound $\tilde { \mathcal { L } }$ and the discrete k-means objective L over the parameter space, which we visualize for cauchy and normal in Figure 5 — the gap is at most 0.5 for normal, but has an order-of-magnitude larger range with cauchy, with the largest gap being greater than 5.

## 4 Probing Algorithmic Generalization Beyond Training Distributions

In this section, we evaluate the trained transformers (which have implicitly learned a single-step clustering algorithm) on varied distributions of clustering tasks. First, in Figure $^ { 6 , }$ we present the performance of the models (trained for single-step clustering) for multiple clustering steps. We consider two versions of multi-step clustering, and compare them to Lloyd's iterations with the same initial centers. See Appendix C for details.

![](images/69df9953b86b3bb3594a4abc0eed1988c689177ca803e861a4de885ed9cadb65.jpg)

![](images/d081e0035d32e6d91a881aec876e43807d52ea724a21f138e1df0c33a6c3a075.jpg)  
Figure 6: Multi-step clustering performance. Models trained with $n = 5 1 2 , d = 3 2 , k = 1 0 , \tau = 0 . 1$ and used for $T = 2 0 \ { \mathsf { s t e p } }$ clustering. We consider all configurations (NE/L2 regularizers and $\tt O H / B N / N A$ embeddings). [Left] Multi-step clustering by applying Equation (4) iteratively for T steps. [Right] Multi step clustering where the model input is re-embedded at each of the T steps, treating T-steps as T single-steps of clustering. See Appendix C for further details.

The results clearly show that naively applying the attention updates in Equation (4) iteratively does not work beyond a single-step, as should be expected since the models are trained for single-step clustering. After the first-step, the k-means objective of the centers generated by the models diverge for all forms of token embeddings $\left( 0 \mathrm { H } / \mathrm { B N } / \mathrm { N A } \right)$ and min-regularizer (NE/L2). However, treating multi-step clustering as multiple single-steps of clustering (thus, re-embedding the inputs at each step) does show improved clustering performance with multiple steps, and models with OH/BN token embeddings significantly outperform Lloyd's iterations. Models with no token embeddings also show improvements with multiple steps but underperform Lloyd's. This highlights that even though we train these models for single-step clustering, allowing for efficient training, they are viable for multi-step clustering as the performance improves when used iteratively in a careful manner.

![](images/dfdf0b1c98d1460d6beb8278260fccb8756788869755f9c399f389a5d4c33b14.jpg)

![](images/4c1a42776f08af43681fe6a4ad95776e08ad13eadbc2bf49c072f22e17faa653.jpg)

![](images/abce42b40bdda3759d4e009f146fd64d7e5b6df23679fa1d38ddedadb0db62b7.jpg)

![](images/a4dfe5dc6961b280a812beef3619a070cb164702319d8f448b0f52247aaf220a.jpg)  
Figure 7: Generalization to varying distributional properties. Models are trained with $n = 5 1 2 , d =$ $3 2 , k = 1 0 , \tau = 0 . 1$ , L2-regularized min with OH/BN/NA token embeddings, and evaluated on $T = 2 0$ step clustering with tasks from mixtures of normal distributions. Training tasks are from equally weighted mixtures of isotropic normal distributions with a per-dimensional scale of 0.1. We evaluate generalization in 4 scenarios: [Left] unequally weighted mixture of isotropic distributions, [Left center] equally weighted mixture of anisotropic distributions, [Right center] equally weighted mixture of isotropic distributions with a per-dimension scale smaller (half) than that during training, and [Right| same as Right center but with a scale larger (double) than that during training.

Next, we evaluate the ability of the model (trained on clustering tasks from one distribution) to generalize to tasks from different distributions. In Figure 7, we consider models trained on tasks with data sampled from an equally-weighted mixtures of isotropic normal distributions, and study their generalization to various changes to the mixture distributions. The learned models can perform multi-step clustering quite well with OH and BN token embedding based models outperforming Lloyd's iterations significantly in all but one case — for data sampled from a mixture of normal distributions with a larger scale (than the one used during training), the models slightly under-perform Lloyd's initially, but eventually outperform Lloyd's with additional clustering steps.

As a further evaluation of algorithmic generalization, we study the effect of changing the distribution family in Figure 8 (larger version in Figure 18). We see that models trained on clustering tasks with data from the cauchy distributions (first column in Figure 8) always diverge for all distributions. This aligns with our previous results in Figure 4, where models trained on cauchy tasks are even unable to generalize well to single-step cauchy clustering problems. Models trained on other distributions are also unable to perform well with cauchy tasks (first row in Figure 8); the models with OH and BN embeddings do not diverge, but are not competitive to Lloyd's here. The remaining pairwise evaluations (the lower-right $4 \times 4$ in Figure 8) all show that the learned models can generalize well to evaluation tasks from other distributions, and continue to significantly outperform Lloyd's iterations (for OH/BN). While generalization to cauchy is an open question, and we leave it for future work, we provide a preliminary probe in Figure 9 (more tasks per family in Figure 19) where we show the pairwise distributional dissimilarities between the per-task initial-center to point squared Euclidean distances. The cauchy tasks are clearly only similar to other cauchy tasks, and are quite dissimilar to the rest; all other task families are quite similar to each

![](images/8f0c6ba589fab7ec63fd918bf68b9c18b179ef656036c2997db3c12be646e8e5.jpg)  
Figure 8: Generalization to different distribution families. The models are trained on one distribution family (each column maps to a training family), and tested for multi-step $( T = 2 0 )$ clustering from a different distribution family (each row maps to a test family).

other. These task (dis)similarities align quite well with the generalization (or lack thereof) we observe in Figure 8.

## 5 Discussion

Building upon the initial findings of Garg et al. [2022] and Akyürek et al. [2023], the area of in-context learning with transformers is very active, usually focusing on supervised learning tasks where some labeled examples $\{ ( { \pmb x } _ { i } , y _ { i } ) \} _ { i \in [ n ] }$ are provided in the context alongside a single unlabeled query $\scriptstyle { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } _ { \mathbf { } } _ { \mathbf { } } { \mathbf } _ { \mathbf { } } _ { \mathbf { } } _ { \mathbf { } \mathbf { } } _ { \mathbf { } \mathbf { } } _ { \mathbf { } } _ { \mathbf } { \mathbf } _ { } { \mathbf } _ { \mathbf { } } _ { \mathbf } { \mathbf } _ { } \mathbf { } \mathbf { } _ { \mathbf }  _ { \mathbf { } \mathbf } _ { \mathbf { } \mathbf } _ { \mathbf { } \mathbf } _ { \mathbf } { \mathbf } _ { \mathbf } _ { \mathbf } \mathbf { } _ { \mathbf } \mathbf { } _ \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \Sigma  _ _ { \mathbf \mathbf { } \mathbf } _ { \mathbf \mathbf } \mathbf \Sigma \Sigma _ { } _ \mathbf \mathbf { } \mathbf \Sigma \Sigma \mathbf \Sigma \Sigma  _ _ { \mathbf \Sigma \Sigma } _ \mathbf \Sigma \Sigma \mathbf \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma \Sigma$ and the predicted output $\hat { y } _ { q }$ for the query is generated in-context during the forward pass through the transformer. Von Oswald et al. [2023] demonstrate that transformers with linear attention can execute gradient descent for a linear model in-context, while subsequent works study learnability with simplified linear attention [Zhang et al. 2023] and the standard softmax attention [Huang et al., 2023]; note that Huang et al. [2023] study a very stylized transformer with structured weight matrices. These results have been extended to demonstrate preconditioned gradient descent for a linear model [Ahn et al., 2023] and functional gradient descent [Cheng et al., 2024]. Usually, a differentiable supervised learning objective (such as mean squared error for regression) is considered. Our focus is on discrete clustering problems, where the learning objective is not natively differentiable, and we carefully study the effect of the necessary smoothing of the clustering objective on

learnability.

While most existing literature focuses on linear attention transformers with (causal) self-attention, we build upon Clarkson et al. [2026] and study softmax attention with an encoderdecoder model that uses both self-attention and cross-attention, though the algorithmic expressivity results rely on a limiting softmax attention where the attention softmax temperature goes to zero. However, this form of softmax attention is practically infeasible, and we use standard softmax attention for our learnability analysis and experiments.

Another line of research connecting transformers and clustering is the study of the token dynamics of self-attention where the tokens are iteratively updated via self-attention, and the goal is to understand the conditions under which these dynamics stabilize and converge [Geshkovski et al., 2023a,b]. The results show that, under certain conditions, the token representations tend to self-cluster, first into a small number of clusters, and then eventually, to a single cluster. In contrast, we focus on learning clustering algorithms that are competitive to existing hand-crafted schemes, and train a single transformer layer instead of the problem setup of the aforementioned papers with multiple (possibly infinite) layers. We learn the transformer parameters from data (in this case, a collection of clustering tasks) and study their training convergence and algorithmic generalization.

![](images/5e774bbcaf509cc18013c76eb7397684fc3d2e50d79ef62b24f0e30ad42b29bc.jpg)  
Figure 9: Pairwise task dissimilarities. We use 4 tasks from each of the families and compute the pairwise Earth-mover distance (EMD) between their respective distribution of the distances between the points X and initial centers $^ { C , }$ giving us a $2 0 \times 2 0$ matrix. Each row and column corresponds to a single task, and the tasks are arranged into family-based blocks in the following order: normal, cauchy, gumbel, laplace, lognormal. Light shades show low EMD.

Conclusion. We study the ability of transformers to express and learn clustering algorithms. First, we establish expressivity of Lloyd's algorithm for k-means clustering with a leaner transformer (Theorem 2.1) than existing constructions [Clarkson et al., 2026]. Then, we study the factors affecting the ability of transformers to learn a clustering algorithm (Theorem 3.1), and empirically validate their importance (Figure 2, 3, 4, 5). We show that these learned algorithms - in the form of transformers – can progressively improve clustering with iterative application (Figure 6), though naive iteration fails, and can successfully execute clustering for tasks that are different than those used for training (Figure 7, 8), though exceptions exist. This work comes with various limitations, with the main ones being (i) the restriction that transformers learn an algorithm for a specific data dimensionality d and number of clusters $k ,$ while hand-crafted algorithms are more general, and (ii) the lack of understanding of the failure to train and generalize with cauchy tasks. We discuss these and various others in Appendix A.

## References

John A Hartigan. Clustering Algorithms. Wiley series in probability and mathematical statistics, 1975. 1

Charu C Aggarwal and Chandan K Reddy. Data Clustering: Algorithms and Applications. Chapman & Hall/CRC, 2013. 1

James MacQueen. Some methods for classification and analysis of multivariate observations. In Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, Volume 1: Statistics, volume 5, pages 281–298. University of California press, 1967. 1

Stuart Lloyd. Least squares quantization in PCM. IEEE transactions on information theory, 28(2): 129–137, 1982. 1, 2, 20

J. C. Dunn. A fuzzy relative of the isodata process and its use in detecting compact well-separated clusters. Journal of Cybernetics, 3:32–57, 1974. 1, 2

James C Bezdek. Pattern recognition with fuzzy objective function algorithms. Springer Science & Business Media, 2013. 1, 2

Leonard Kaufman. Partitioning around medoids (program pam). Finding groups in data, 344:68–125, 1990. 1,20

Raymond T. Ng and Jiawei Han. CLARANS: A method for clustering objects for spatial data mining. IEEE transactions on knowledge and data engineering, 14(5):1003–1016, 2002. 1

Ulrike Von Luxburg. A tutorial on spectral clustering. Statistics and computing, 17(4):395-416, 2007. 1

Bernhard Schölkopf, Alexander Smola, and Klaus-Robert Müller. Nonlinear component analysis as a kernel eigenvalue problem. Neural computation, 10(5):1299–1319, 1998. 1

Mark Girolami. Mercer kernel-based clustering in feature space. IEEE transactions on neural networks, 13 (3):780–784, 2002. 1

Inderjit S Dhillon, Yuqiang Guan, and Brian Kulis. Kernel k-means: spectral clustering and normalized cuts. In Proceedings of the tenth ACM SIGKDD international conference on Knowledge discovery and data mining, pages 551–556, 2004. 1

Sanjoy Dasgupta. The hardness of k-means clustering. UCSD Technical Report, 2008. URL https: //cseweb.ucsd.edu/\~dasgupta/papers/kmeans.pdf. 1

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.1

Mary Phuong and Marcus Hutter. Formal algorithms for transformers. arXiv preprint arXiv:2207.09238, 2022. URL https://arxiv.org/pdf/2207.09238.pdf. 1

Alex Graves, Greg Wayne, and Ivo Danihelka. Neural turing machines. arXiv preprint arXiv:1410.5401, 2014. URL https://arxiv.org/pdf/1410.5401. 1

Marcin Andrychowicz, Misha Denil, Sergio Gomez, Matthew W Hoffman, David Pfau, Tom Schaul, Brendan Shillingford, and Nando De Freitas. Learning to learn by gradient descent by gradient descent. Advances in neural information processing systems, 29, 2016. URL https://proceedings.neurips.cc/paper\_ files/paper/2016/file/fb87582825f9d28a8d42c5e5e5e8b23d-Paper.pdf. 1

Shivam Garg, Dimitris Tsipras, Percy S Liang, and Gregory Valiant. What can transformers learn incontext? a case study of simple function classes. Advances in neural information processing systems, 35:30583-30598, 2022. URL https://proceedings.neurips.cc/paper\_files/paper/2022/file/ c529dba08a146ea8d6cf715ae8930cbe-Paper-Conference.pdf. 2, 11

Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning algorithm is in-context learning? investigations with linear models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=0g0X4H8yN4I. 2, 11, 18

Johannes Von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by gradient descent. In International Conference on Machine Learning, pages 35151-35174. PMLR, 2023. URL https://proceedings.mlr. press/v202/von-oswald23a/von-oswald23a.pdf. 2, 11

Yingcong Li, Muhammed Emrullah Ildiz, Dimitris Papailiopoulos, and Samet Oymak. Transformers as algorithms: Generalization and stability in in-context learning. In International Conference on Machine Learning, pages 19565-19594. PMLR, 2023. URL https://proceedings.mlr.press/v202/1i231/1i231. pdf. 2, 33, 37

Kwangjun Ahn, Xiang Cheng, Hadi Daneshmand, and Suvrit Sra. Transformers learn to implement preconditioned gradient descent for in-context learning. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=LziniAXEI9. 2, 11

Yu Huang, Yuan Cheng, and Yingbin Liang. In-context convergence of transformers. arXiv preprint arXiv:2310.05249, 2023. URL https://arxiv.org/pdf/2310.05249.pdf. 2, 11

Ruiqi Zhang, Spencer Frei, and Peter L Bartlett. Trained transformers learn linear models in-context. arXiv preprint arXiv:2306.09927, 2023. URL https://arxiv.org/pdf/2306.09927.pdf. 2, 11

Xiang Cheng, Yuxin Chen, and Suvrit Sra. Transformers implement functional gradient descent to learn non-linear functions in context. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview.net/forum?id=ah1BlQcLv4. 2, 11

Takuya Ito, Murray Campbell, Lior Horesh, Tim Klinger, and Parikshit Ram. Quantifying artificial intelligence through algorithmic generalization. Nature Machine Intelligence, pages 1-11, 2025. URL https://www.nature.com/articles/s42256-025-01092-w. 2

Kenneth L. Clarkson, Lior Horesh, Takuya Ito, Charlotte Park, and Parikshit Ram. Transformer circuits can realize clustering algorithms. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id=2jw5U060C4. 2, 3, 4, 7, 12, 17, 18, 20, 47

Juan Antonio Cuesta-Albertos, Alfonso Gordaliza, and Carlos Matrán. Trimmed k-means: an attempt to robustify quantizers. The Annals of Statistics, 25(2):553–576, 1997. URL http://dx.doi.org/10.1214/ aos/1031833664. 2

Yao-Hung Hubert Tsai, Shaojie Bai, Makoto Yamada, Louis-Philippe Morency, and Ruslan Salakhutdinov. Transformer dissection: An unified understanding for transformer's attention via the lens of kernel. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 4344-4353. Association for Computational Linguistics, 2019. doi: 10.18653/v1/D19-1443. URL https://aclanthology.org/D19-1443/. 3

André Martins and Ramon Astudillo. From softmax to sparsemax: A sparse model of attention and multi-label classification. In International conference on machine learning, pages 1614-1623. PMLR, 2016. URL https://proceedings.mlr.press/v48/martins16.pdf. 6, 18, 32

Mathieu Blondel, André FT Martins, and Vlad Niculae. Learning with fenchel-young losses. Journal of Machine Learning Research, 21(35):1-69, 2020. URL https://jmlr.csail.mit.edu/papers/volume21/ 19-021/19-021.pdf. 6,33

Olivier Bousquet and André Elisseeff. Algorithmic stability and generalization performance. Advances in Neural Information Processing Systems, 13, 2000. URL https://proceedings.neurips.cc/paper files/paper/2000/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf. 6

Moritz Hardt, Ben Recht, and Yoram Singer. Train faster, generalize better: Stability of stochastic gradient descent. In International conference on machine learning, pages 1225-1234. PMLR, 2016. URL https://arxiv.org/pdf/1509.01240. 6

Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, and Tom Goldstein. Visualizing the loss landscape of neural nets. Advances in neural information processing systems, 31, 2018. URL https://proceedings. neurips.cc/paper\_files/paper/2018/file/a41b3bb3e6b050b6c9067c67f663b915-Paper.pdf. 8,9

Borjan Geshkovski, Cyril Letrouit, Yury Polyanskiy, and Philippe Rigollet. The emergence of clusters in self-attention dynamics. arXiv preprint arXiv:2305.05465, 2023a. URL https://arxiv.org/pdf/2305. 05465.pdf. 12

Borjan Geshkovski, Cyril Letrouit, Yury Polyanskiy, and Philippe Rigollet. A mathematical perspective on transformers. arXiv preprint arXiv:2312.10794, 2023b. URL https://arxiv.org/pdf/2312.10794.pdf. 12

Weiran Wang and Miguel A Carreira-Perpinán. Projection onto the probability simplex: An efficient algorithm with a simple proof, and an application. arXiv preprint arXiv:1309.1541, 2013. URL https://arxiv.org/pdf/1309.1541. 32

## Appendix

## Table of Contents

A Limitations 17   
B Leaner Transformers Can Emulate Lloyd's Iteration 18   
C Single-Step and Multi-Step Clustering 20   
D Clustering Task Generation 21   
E Computational Resources and Implementation 22   
F Ablative Experiments 22   
G Additional Figures 30   
H Smoothed Clustering Objective via Regularized min 32   
I Lipschitz-ness of the Smoothed Upperbound of the Clustering Objective 34

## A Limitations

As discussed in our concluding remarks, one major limitation of our work is that we currently lack a clear understanding as to why the learning and generalization fail in certain situations. For example, it is not clear why learning and generalization is hard with cauchy distributions. We hypothesize that this is related to the inability of the smoothed clustering objective (via a regularized min) to provide a tight upper-bound for the actual k-means clustering loss (as we show in Figure 5). However, it is not completely clear why, but we speculate that the following is a potential path to understanding: The cauchy distribution is fat-tailed, and hence, many points in X can be quite far from the true cluster, and act like outliers in a clustering problem. The presence of outlier values – the large pairwise squared Euclidean distances to the outliers compared to the relatively smaller pairwise squared distances to the inliers – in the regularized min can severely degrade the upper-bounds obtained via the smoothed objective. This still does not give us a clear way to mitigate the issues while training with cauchy tasks seen in Figure 4. Similarly, the presence of outliers in X also results in the significant pairwise distributional distances between the tasks shown in Figure 9 and Figure 19, leading to the failures of the trained transformers on cauchy validation tasks.

Beyond that, there are architectural limitations we inherit from Clarkson et al. [2026], where the transformer learns to cluster for a specific choice of data dimensionality d and number of clusters $k ,$ though it is able to perform clustering for varying number of points n in d dimensions into k clusters (as shown in Figure 12). It is important to extend this architecture to be able to generalize to different values of k or $d ,$ and be able to train on clustering tasks with a certain k value and generalize to larger k values. One possible way to generalize in terms of d is to train the transformer with a large $d ^ { \prime }$ Then any clustering task with $d < d ^ { \prime }$ can be input to the transformer by zero-padding the necessary extra $( d ^ { \prime } - d )$ dimensions to the original data in d dimensions. As Euclidean distances are unaffected by zero-padding, this idea should conceptually work. However, we pay the computational cost of training a model for a large $d ^ { \prime }$ as the model embedding dimensionality $d _ { \mathsf { e m b } }$ would scale linearly in $d ^ { \prime } ,$ while the number of parameters would scale quadratically in $d ^ { \prime } .$ Whether such a setup would show strong generalization performance across various values of $d$ is unclear, and would depend on the training strategies. A similar scheme could also be considered to generalize to different values of $k ,$ where we would train a model on a large value of $k ^ { \prime }$

Additionally, there are various open questions we do not address in this paper, such as

• It is not clear if the additional dimensions - k dimensions with (OH) and $\lceil \log _ { 2 } k \rceil$ with (BN) - are necessary to learn a clustering algorithm that is competitive to Lloyd's algorithm. Empirically, we see that these additional dimensions do allow the models to surpass Lloyd's algorithm, and the models without these additional dimensions (the NA models) are empirically unable to match Lloyd's. However, we do not have theory codifying this need for the additional dimensions.

• Our proposed token embedding in (BN) can express Lloyd's algorithm for k-means clustering with the limiting softmax attention (with $\gamma \to \infty )$ , but can introduce an arbitrary topological order among the cluster centers with a finite $\gamma$ . We already see that models with the BN token embedding have slower training convergence compared to the models with the OH token embeddings (especially for small $d )$ , but it is possible that this arbitrary order can introduce other unexpected behaviours.

• The standard transformer has an attention block and a feedforward block (which we do not consider in this work). It is not clear how the inclusion of the feedforward block would affect learning and generalization. The theoretical results in Clarkson et al. [2026] utilize the feedforward block to perform Euclidean clustering with dot-product attention (instead of the negative squared Euclidean distance based attention in Definition 2.2), but it is not clear what role the feedforward block will play in the practical learning setup.

• While we are able to train the model efficiently for single-step clustering tasks, and utilize it for multi-step clustering by treating multi-step clustering as multiple single-step clustering tasks, it is not clear if we can learn a better model by training directly for multi-step clustering with appropriately designed objectives, and do so efficiently.

• Our discussion in Section 3 around learning convergence and generalization consider training losses that are β-smooth. We see that the L2-regularized min (using sparsemax [Martins and Astudillo, 2016]) provides a tighter upper-bound and thus generalizes better in many cases over the NE-regularized min (using softmax). However, the sparsemax operation is only differentiable almost everywhere, and is not generally β-smooth, thus leading to a mismatch between our theoretical discussion and empirical results. One possible explanation for the strong empirical performance of the L2-regularized min is that the loss surface is actually more favourable than the usual nonconvex loss landscapes of neural networks, as can be seen in Figure 3, where the loss landscapes appear convex for the larger value of d.

```latex
Algorithm 2 Lloyd's algorithm to cluster n points into k clusters
1 Input: Instances $\{ \pmb { x } _ { i } \in \mathbb { R } ^ { d } , i \in [ n ] \}$
2 Input: Number of iterations T
3 Input: Initial centers $\{ c _ { j } \in \mathbb { R } ^ { d } , j \in [ k ] \}$
4 $\forall j \in [ k ] , c _ { j } ^ { ( 0 ) } \gets c _ { j }$
5 for $t = 1 , \dots , T$ do
6 $\begin{array} { r } { \forall i \in [ n ] , \ a _ { i } ^ { ( t ) }  \arg \operatorname* { m i n } _ { j \in [ k ] } \| { \pmb x } _ { i } - { \pmb x } _ { j } ^ { ( t - 1 ) } \| ^ { 2 } } \end{array}$
7 $\begin{array} { r } { \forall j \in [ k ] , \ c _ { j } ^ { ( t ) } \gets \arg \operatorname* { m i n } _ { \pmb { c } \in \mathbb { R } ^ { d } } \sum _ { i \in [ n ] : a _ { i } ^ { ( t ) } = j } \left\| \pmb { x } _ { i } - \pmb { c } \right\| ^ { 2 } } \end{array}$
8 return $\{ c _ { j } ^ { ( T ) } , j \in [ k ] \}$
```

• We treat the softmax attention inverse temperature γ and the regularized min penalty τ as fixed hyperparameters in our analysis in Theorem 3.1 and in our experiments. But in practice, these parameters can be adaptive or annealed through the training and inference, and this is not something we consider here.

• Finally, while we see that the clustering algorithm learned by the transformer is able to outperform Lloyd's algorithm for k-means clustering, it is not clear what the precise clustering algorithm is that the learned model is executing. While we can consider the tools used in Akyürek et al. 2023| such as the notion of "algorithmic distance", we leave this for future work.

## B Leaner Transformers Can Emulate Lloyd's Iteration

We use $\mathbf { \delta } _ { I _ { d } }$ to denote the $( d \times d )$ identity matrix. Further, we specify our matrices for attention in a compact form, using the following notation. For integers $1 \leq i \leq j \leq d ,$ let $\pmb { I } _ { d } ^ { i : j }$ be the matrix such that for any $\pmb { x } \in \mathbb { R } ^ { d }$ such that

$$
\begin{array} { r } { v = I _ { d } ^ { i : j } x , \quad \mathrm { ~ w h e r e ~ } v _ { m } = \left\{ \begin{array} { l l } { x _ { m } } & { i \leq m \leq j } \\ { 0 } & { o . w . } \end{array} \right. . } \end{array}\tag{11}
$$

The $v = I _ { d } ^ { i : j }$ x zeroes out the m-th entries of x with indices $m \in [ 1 , i ) \cup ( j , d ]$ , leaving the rest unchanged When $j = d ,$ we use $I _ { d } ^ { i : }$ , and when $i = 1$ , we use $\bar { \cal I } _ { d } ^ { : j }$

We prove Theorem 2.1 based on the construction in Clarkson et al. [2026, Theorem 2] with minor modification. We use the selective property of attention as described in Clarkson et al. [2026, Appendix D] in the form of averaging hard attention or AHAT as the inverse temperature of the softmax goes to infinity; this behaves as a hardmax when there is a unique maximum. We are restating Theorem 2.1 as follows:

Theorem B.1. Given as input a set of points $\pmb { X } \in \mathbb { R } ^ { d \times n }$ and initial centers $C \in \mathbb { R } ^ { d \times k }$ , assuming that no iterations of Lloyd's algorithm lead to empty clusters, there exists parameters $\{ ( Q _ { \mu } , K _ { \mu } , V _ { \mu } ) , \mu =$ $1 , \ldots , 4 \}$ for the transformer in Definition 2.3 with $\gamma = \infty$ and initial token embeddings $( \bar { \pmb X } ^ { ( 0 ) } , \bar { \pmb C } ^ { ( 0 ) } )$ as in Definition 2.1 then, for any $t \in \mathbb { N }$ , the output of the t-th transformer layer exactly matches the output of Lloyd's algorithm (Algorithm 2) with the same input after t iterations

Proof. With $d _ { \mathsf { e m b } } = d + \lceil \log _ { 2 } k \rceil$ , let $K _ { 1 } = Q _ { 1 } = K _ { 2 } = Q _ { 2 } = I _ { d _ { \mathrm { e m b } } } ^ { : d }$ , and $V _ { 1 } = - V _ { 2 } = I _ { d _ { \mathrm { e m b } } } ^ { d + 1 : }$ . Note that $\pmb { I } _ { d _ { \mathrm { e m b } } } ^ { : d }$ and $I _ { d _ { \mathrm { e m b } } } ^ { d + 1 : }$ are $( d + \lceil \log _ { 2 } k \rceil ) \times ( d + \lceil \log _ { 2 } k \rceil )$ matrices.

At any step $t \geq 0$ , let us assume that $\bar { \mathbf { X } } ^ { ( t ) } = \left\lceil \sum _ { Y ^ { ( t ) } } \right\rceil$ , and $\bar { C } ^ { ( t ) } = \left[ { C ^ { ( t ) } } \right]$ . This is true at $t = 0$ by the definition of the token embeddings in Definition 2.1. We will show that, at any $t \geq 0$ the update to $( \bar { X } ^ { ( t + 1 ) } , \bar { C } ^ { ( t + 1 ) } )$ maintains this form where $\bar { X } ^ { ( t + 1 ) } = \biggl [ \underset { } { X } ^ { \biggr ] }$ , and $\bar { C } ^ { ( t + 1 ) } = \left[ \begin{array} { c } { { C ^ { ( t + 1 ) } } } \\ { { B _ { k } } } \end{array} \right]$ , where

$\mathbf { \nabla } _ { \mathbf { Y } } ( t { + } 1 )$ is the matrix with the i-th column corresponding to the binary representation of $( j ^ { \star } ( i ) - 1 )$ in $\lceil \log _ { 2 } k \rceil$ bits, with $j ^ { \star } ( i ) \in [ k ]$ denoting the index of the cluster to which the i-th point is assigned to in the $( t + 1 ) – \mathrm { t h }$ Lloyd's iteration in line 6 of Algorithm 2.

$C ^ { ( t + 1 ) }$ corresponds to the updated cluster centers as in line 7 of Algorithm 2.

To update cluster assignments for Lloyd's algorithm, that is, to compute $\bar { X } ^ { ( t + 1 ) }$ from $\bar { X } ^ { ( t ) }$ and $\bar { C } ^ { ( t ) }$ , we compute

$$
\begin{array} { r l } & { \mathcal A _ { 2 } ^ { \gamma } ( \bar { X } ^ { ( t ) } , \bar { C } ^ { ( t ) } ; ( I _ { { \mathrm { e n b } } } ^ { : d } , I _ { { d _ { \mathrm { e n b } } } } ^ { : d } , I _ { { d _ { \mathrm { e n b } } } } ^ { d + 1 : } ) ) + \mathcal A _ { 2 } ^ { \gamma } ( \bar { X } ^ { ( t ) } , \bar { X } ^ { ( t ) } ; ( I _ { { d _ { \mathrm { e n b } } } } ^ { : d } , I _ { { d _ { \mathrm { e n b } } } } ^ { : d } , - I _ { { d _ { \mathrm { e n b } } } } ^ { d + 1 : } ) ) } \\ & { \quad \quad \quad = I _ { { d _ { \mathrm { e n b } } } } ^ { d + 1 : } \bar { C } ^ { ( t ) } \mathrm { s o f t m a x } ( - ( \gamma / \sqrt { d _ { \mathrm { e n b } } } ) \mathrm { s q d i s t } ( I _ { { d _ { \mathrm { e n b } } } } ^ { : d } \bar { X } ^ { ( t ) } , I _ { { d _ { \mathrm { e n b } } } } ^ { : d } \bar { C } ^ { ( t ) } ) ) } \\ & { \quad \quad \quad \quad - I _ { { d _ { \mathrm { e n b } } } } ^ { d + 1 : } \bar { X } ^ { ( t ) } \mathrm { s o f t m a x } ( - ( \gamma / \sqrt { d _ { \mathrm { e n b } } } ) \mathrm { s q d i s t } ( I _ { { d _ { \mathrm { e n b } } } } ^ { : d } \bar { X } ^ { ( t ) } , I _ { { d _ { \mathrm { e n b } } } } ^ { : d } \bar { X } ^ { ( t ) } ) ) } \end{array}\tag{12}
$$

$$
= \left[ \begin{array} { l } { \mathbf { 0 } _ { d \times k } } \\ { B _ { k } } \end{array} \right] \operatorname { s o f t m a x } \left( - ( \gamma / \sqrt { d _ { \operatorname { e m b } } } ) \operatorname { s q d i s t } \left( \left[ \begin{array} { c } { X } \\ { 0 _ { \lceil \log _ { 2 } k \rceil \times n } } \end{array} \right] , \left[ \begin{array} { c } { C ^ { ( t ) } } \\ { 0 _ { \lceil \log _ { 2 } k \rceil \times k } } \end{array} \right] \right) \right)
$$

$$
- \left[ \pmb { 0 } _ { d \times n } \right] \mathrm { s o f t m a x } \left( - ( \gamma / \sqrt { d _ { \mathrm { e m b } } } ) \mathrm { s q d i s t } \left( \left[ \pmb { 0 } _ { \lceil \log _ { 2 } k \rceil \times n } ^ { X } \right] , \left[ \pmb { 0 } _ { \lceil \log _ { 2 } k \rceil \times n } ^ { X } \right] \right) \right)\tag{13}
$$

$$
= \left[ \begin{array} { l } { \mathbf { 0 } _ { d \times k } } \\ { \mathbf { 0 } _ { k } } \end{array} \right] \mathrm { h a r d m a x } \left( - \mathrm { s q d i s t } \left( \mathbf { X } , C ^ { ( t ) } \right) \right) - \left[ \begin{array} { l } { \mathbf { 0 } _ { d \times n } } \\ { Y ^ { ( t ) } } \end{array} \right] \mathrm { h a r d m a x } \left( - \mathrm { s q d i s t } \left( \mathbf { X } , \mathbf { X } \right) \right) ,\tag{14}
$$

as softmax with inverse temperature $\gamma = \infty$ gives us hardmax, where for any matrix $\pmb { A } \in \mathbb { R } ^ { a \times b }$ , hardmax $( A ) \in$ $\{ 0 , 1 \} ^ { a \times b }$ is binary matrix of the same size as A with each column having exactly a single nonzero value corresponding to the unique maximum of the column, assuming each column has an unique maximum.

First, assuming $\mathbf { \boldsymbol { x } } _ { i } \neq \mathbf { \boldsymbol { x } } _ { i ^ { \prime } }$ for any $i \neq i ^ { \prime } ,$ hardmax $( - \mathrm { s q d i s t } ( \pmb { X } , \pmb { X } ) ) = \pmb { I _ { n } }$ . Next hardmax $\left( - \mathrm { s q d i s t } \left( \boldsymbol { X } , \boldsymbol { C } ^ { ( t ) } \right) \right)$ applies a columnwise argmin to the $\left( k \times n \right)$ matrix sqdist $( X , C ^ { ( t ) } )$ , thereby, each column having only the $j ^ { \star } ( i ) -$ th entry as 1 where $\begin{array} { r } { j ^ { \star } ( i ) = \arg \operatorname* { m i n } _ { j \in [ k ] } \left\| { \bf X } _ { : , i } - { \bf C } _ { : , j } ^ { ( t ) } \right\| ^ { 2 } } \end{array}$ . Thus $\pmb { Y } ^ { ( t + 1 ) } = \pmb { B } _ { k } \mathrm { h a r d m a x } \left( - \mathrm { s q d i s t } \left( \pmb { X } , \pmb { C } ^ { ( t ) } \right) \right)$ is a $\left( \lceil \log _ { 2 } k \rceil \times n \right)$ binary matrix, where the i-th column $Y _ { : , i } ^ { ( t ) }$ is the $j ^ { \star } ( i )  – \mathrm { t h }$ column of $\scriptstyle B _ { k }$ , thereby denoting the updated cluster assignments as per Lloyd's algorithm (Algorithm 2, line 5). All together, we have

$$
\begin{array} { r l } & { \mathcal { A } _ { 2 } ^ { \gamma } ( \bar { X } ^ { ( t ) } , \bar { C } ^ { ( t ) } ; ( I _ { d _ { \mathrm { e m b } } } ^ { : d } , I _ { d _ { \mathrm { e m b } } } ^ { : d } , I _ { d _ { \mathrm { e m b } } } ^ { d + 1 : } ) ) + \mathcal { A } _ { 2 } ^ { \gamma } ( \bar { X } ^ { ( t ) } , \bar { X } ^ { ( t ) } ; ( I _ { d _ { \mathrm { e m b } } } ^ { : d } , I _ { d _ { \mathrm { e m b } } } ^ { : d } , - I _ { d _ { \mathrm { e m b } } } ^ { d + 1 : } ) ) } \\ & { \quad = \bigg [ \frac { \mathbf { 0 } _ { d \times n } } { \big ( \mathbf { Y } ^ { ( t + 1 ) } - \mathbf { Y } ^ { ( t ) } \big ) } \bigg ] . } \end{array}\tag{15}
$$

Adding this to $\bar { X } ^ { ( t ) } = \biggl [ \frac { X } { Y ^ { ( t ) } } \biggr ]$ via the residual connection in Equation (4) gives us the updated $\bar { \pmb X } ^ { ( t + 1 ) } =$ $\left[ \begin{array} { c } { { X } } \\ { { Y ^ { ( t + 1 ) } } } \end{array} \right]$ with the updated cluster assignments $\mathbf { \nabla } _ { \mathbf { Y } } ( t { + } 1 )$ matching Lloyd's algorithm (Algorithm 2, line 5). Let $Q _ { 3 } = K _ { 3 } = Q _ { 4 } = K _ { 4 } = I _ { d _ { \mathrm { e m b } } } ^ { d + 1 : }$ , and $V _ { 3 } = - V _ { 4 } = I _ { d _ { \mathrm { e m b } } } ^ { : d }$ . To update cluster centers as per Lloyd's algorithm (Algorithm 2, line 6), we compute

$$
\begin{array} { r l r } & { } & { \mathcal { A } _ { 2 } ^ { \gamma } ( \bar { C } ^ { ( t ) } , \bar { X } ^ { ( t + 1 ) } ; ( I _ { d _ { \mathrm { e m b } } } ^ { d + 1 : } , I _ { d _ { \mathrm { e m b } } } ^ { d + 1 : } , I _ { d _ { \mathrm { e m b } } } ^ { \prime d } ) ) + \mathcal { A } _ { 2 } ^ { \gamma } ( \bar { C } ^ { ( t ) } , \bar { C } ^ { ( t ) } ; ( I _ { d _ { \mathrm { e m b } } } ^ { d + 1 : } , I _ { d _ { \mathrm { e m b } } } ^ { d + 1 : } , - I _ { d _ { \mathrm { e m b } } } ^ { \prime d } ) ) } \\ & { } & { = I _ { d _ { \mathrm { e m b } } } ^ { \gamma d } \bar { X } ^ { ( t + 1 ) } \mathrm { s o f t m a x } ( - ( \gamma / \sqrt { d _ { \mathrm { e m b } } } ) \mathrm { s q d i s t } ( I _ { d _ { \mathrm { e m b } } } ^ { d + 1 : } \bar { C } ^ { ( t ) } , I _ { d _ { \mathrm { e m b } } } ^ { d + 1 : } \bar { X } ^ { ( t + 1 ) } ) ) } \\ & { } & { - I _ { d _ { \mathrm { e m b } } } ^ { \gamma d } \bar { C } ^ { ( t ) } \mathrm { s o f t m a x } ( - ( \gamma / \sqrt { d _ { \mathrm { e m b } } } ) \mathrm { s q d i s t } ( I _ { d _ { \mathrm { e m b } } } ^ { d + 1 : } \bar { C } ^ { ( t ) } , I _ { d _ { \mathrm { e m b } } } ^ { d + 1 : } \bar { C } ^ { ( t ) } ) ) } \\ & { } & { = [ \mathbf { X }  } \\ & { } &   0 _ { \lceil \log _ { 2 } k \rceil \times n } ] \mathrm { s o f t m a x } ( - ( \gamma / \sqrt { d _ { \mathrm { e m b } } } ) \mathrm { s q d i s t } ( [ \begin{array} { l } { \mathbf { 0 } _ { d \times k } } \\ { B _ { k } } \end{array} ] \end{array}\tag{16}
$$

$$
\begin{array} { r l } & { \quad - \left[ \underset { \mathbf { 0 } _ { \lceil \log _ { 2 } k \rceil \times k } } { C ^ { ( t ) } } \right] \mathrm { s o f t m a x } \left( - \big ( \gamma / \sqrt { d _ { \mathrm { e m b } } } \big ) \mathrm { s q d i s t } \left( \left[ \begin{array} { l } { \mathbf { 0 } _ { d \times k } } \\ { \mathbf { B } _ { k } } \end{array} \right] , \left[ \begin{array} { l } { \mathbf { 0 } _ { d \times k } } \\ { B _ { k } } \end{array} \right] \right) \right) } \\ & { = \left[ \underset { \mathbf { 0 } _ { \lceil \log _ { 2 } k \rceil \times n } } { X } \right] \mathrm { a v g h a r d m a x } \left( - \mathrm { s q d i s t } \left( \pmb { B } _ { k } , \pmb { Y } ^ { ( t + 1 ) } \right) \right) } \\ & { \quad \quad - \left[ \underset { \mathbf { 0 } _ { \lceil \log _ { 2 } k \rceil \times k } } { C ^ { ( t ) } } \right] \mathrm { h a r d m a x } \left( - \mathrm { s q d i s t } \left( \pmb { B } _ { k } , \pmb { B } _ { k } \right) \right) } \end{array}\tag{17}
$$

(18)

where $\gamma = \infty$ gives us the columnwise hardmax and the columnwise averaging hardmax operation avghardmax. The reason for the averaging hardmax is because each column in sqdist $\left( B _ { k } , \pmb { Y } ^ { ( t + 1 ) } \right)$ can have a non-singleton set of minima (or multiple maxima). As before, hardmax(—sqdist $( B _ { k } , B _ { k } ) ) = I _ { k }$

Next, avghardmax (—sqdist $\left( B _ { k } , Y ^ { ( t + 1 ) } \right) \Big )$ will be ${ \sf a } \left( n \times k \right)$ matrix, where in the j-th column, the i-th entry will be zero if $X _ { : , i }$ is not assigned to the j-th cluster, that is the i-th column of $\mathbf { \nabla } _ { \mathbf { Y } } ( t { + } 1 )$ does not match the $j \mathrm { - t h }$ column of ${ \mathbf { } } B _ { k } ;$ otherwise, the averaging hardmax sets it to $1 / m _ { j }$ where $m _ { j }$ is the number of points assigned to the $j \mathrm { - t h }$ cluster, and thus, the squared Euclidean distance between the i-th column of $\mathbf { \nabla } _ { \mathbf { Y } } ( t { + } 1 )$ and the j-th column of $B _ { k }$ is zero. Note that $m _ { j } > 0 \forall j \in [ k ]$ as we assume that there are no empty clusters during Lloyd's iterations. Thus, Xavghardmax (—sqdist $\left( B _ { k } , Y ^ { ( t + 1 ) } \right) \Big )$ will be a $( d \times k )$ matrix corresponding to the centers $C ^ { ( t + 1 ) }$ computed with Lloyd's algorithm (Algorithm 2, line 6). Thus, we have

$$
\begin{array} { r l } & { \mathcal { A } _ { 2 } ^ { \gamma } ( \bar { C } ^ { ( t ) } , \bar { X } ^ { ( t + 1 ) } ; ( I _ { d _ { \mathrm { e m b } } } ^ { d + 1 ; } , I _ { d _ { \mathrm { e m b } } } ^ { d + 1 ; } , I _ { d _ { \mathrm { e m b } } } ^ { \prime d } ) ) + \mathcal { A } _ { 2 } ^ { \gamma } ( \bar { C } ^ { ( t ) } , \bar { C } ^ { ( t ) } ; ( I _ { d _ { \mathrm { e m b } } } ^ { d + 1 ; } , I _ { d _ { \mathrm { e m b } } } ^ { d + 1 ; } , - I _ { d _ { \mathrm { e m b } } } ^ { \prime d } ) ) } \\ & { \quad = \left[ \begin{array} { l } { C ^ { ( t + 1 ) } - C ^ { ( t ) } } \\ { \mathbf { 0 } _ { \lceil \log _ { 2 } k \rceil \times k } } \end{array} \right] , } \end{array}\tag{19}
$$

and adding this to ${ \bar { C } } ^ { ( t ) } = { \left[ \begin{array} { l } { C ^ { ( t ) } } \\ { B _ { k } } \end{array} \right] } { \mathrm { ~ g i v e s ~ u s ~ } } { \bar { C } } ^ { ( t + 1 ) } = { \left[ \begin{array} { l } { C ^ { ( t + 1 ) } } \\ { B _ { k } } \end{array} \right] }$ , thus tracking the same cluster centers as in Lloyd's algorithm.

Thus we have established that both the cluster assignments and the cluster center updates of Lloyd's algorithm can be maintained in $( \bar { \pmb X } ^ { ( t ) } , \bar { C } ^ { ( t ) } )$ for any t with the leaner token embedding with $d _ { \mathsf { e m b } } =$ $d + \lceil \log _ { 2 } k \rceil$ □

Note that this construction for the $\bar { X } ^ { ( t ) } \to \bar { X } ^ { ( t + 1 ) }$ is the same as in the proof of Clarkson et al. [2026, Theorem 2]. The construction for the $\bar { C } ^ { ( t ) } \to \bar { C } ^ { ( t + 1 ) }$ update is different from the one in Clarkson et al. [2026] as we make use of negative squared Euclidean distance based attention with $\gamma = \infty$ instead of a linear attention as in Clarkson et al. [2026].

## C Single-Step and Multi-Step Clustering

We set up the learning problem in Equation (6) as a single-step clustering task where the transformer model generates cluster prototypes in one-shot (with a single forward pass through the model) such that the k-means objective (actually a surrogate of it) is reduced. However, almost all clustering algorithms are iterative in nature, where the cluster prototypes are iteratively refined in a way that usually progressively improves the clustering objective; Lloyd's algorithm for k-means clustering [Lloyd, 1982| and partitioning around medoids or PAM [Kaufman, 1990] are examples of such algorithms. Thus, it is natural to explore whether we can utilize this trained transformer for multi-step clustering, with the cluster prototypes being iteratively refined. Given the theoretical results in Theorem 2.1 and Clarkson et al. [2026, Theorem 1], it would seem natural to perform T-step clustering refinement by applying the attention-based updates in Definition 2.3 iteratively for T steps with the learned model parameters, starting from $( \bar { X } ^ { ( 0 ) } , \bar { C } ^ { ( 0 ) } )  ( E _ { X } ( X ) , E _ { C } ( C ) )$ , and use the first d rows $\bar { C } _ { : d , : } ^ { ( T ) }$ of the T-step output $( \bar { \pmb X } ^ { ( T ) } , \bar { \pmb C } ^ { ( T ) } )$ as the final set of prototypes. Note that the token embedding scheme can be either the $( E _ { X } ^ { 0 \mathrm { H } } , E _ { C } ^ { 0 \mathrm { H } } )$ scheme in (0H) or the proposed $( E _ { X } ^ { \mathtt { B N } } , E _ { C } ^ { \mathtt { B N } } )$ scheme in (BN) in Definition 2.1.

However, it is important to note that this is valid only if the transformer model parameters are as in the constructive proofs of these theoretical results, and if we are using the softmax attention with the inverse temperature $\gamma \to \infty$ . We have trained the transformer model with finite $\gamma .$

Thus, in the following, we precisely explain what we mean by single-step clustering (used during training) vs multi-step clustering (used for evaluating the model as a clustering algorithm). First, we will demonstrate how standard algorithms such as Lloyd's perform multi-step clustering by applying the same operation iteratively on evolving inputs (data and current centers). Then we will use the same intuition to motivate how we can similarly use the model iteratively on evolving inputs.

For a clustering task with data X and initial centers ${ \cal C } ^ { ( 0 ) }$ , T iterations of Lloyd's algorithm (Algorithm 2) does the following:

$[ \mathrm { S t e p ~ 1 } ] X , C ^ { ( 0 ) } $ assign/center-update $ C ^ { ( 1 ) }$

$[ \mathrm { S t e p ~ } 2 ] X , C ^ { ( 1 ) } $ assign/center-update $ C ^ { ( 2 ) }$

● .

• [Step $T ] X , C ^ { ( T - 1 ) } \to \mathrm { a s s i g n } / \mathrm { c e n t e r } \mathrm { - } \mathrm { u p d a t e } \to C ^ { ( T ) }$

With a trained model $F _ { d : k } ^ { \gamma } ( \cdot , \cdot ; \Theta )$ which uses token embeddings $\left( 0 \mathrm { H } / \mathrm { B N } \right)$ and attention based updates in (4), we can perform T iterations of multi-step clustering in the same spirit as the above iterative Lloyd's algorithm by applying the same model $F _ { d : k } ^ { \gamma }$ on evolving inputs (data and current centers):

• [Step 1]X, C(0) → Ftk(X, C(0); Θ) → C(1),

$[ \mathrm { S t e p ~ } 2 ] X , C ^ { ( 1 ) }  F _ { d : k } ^ { \gamma } ( X , C ^ { ( 1 ) } ; \Theta )  C ^ { ( 2 ) } $

● .

• [ $\operatorname { S t e p } T ] X , C ^ { ( T - 1 ) } \to F _ { d : k } ^ { \gamma } ( X , C ^ { ( T - 1 ) } ; \Theta ) \to C ^ { ( T ) }$

This follows the iterative nature of Lloyd's algorithm, and this is what is evaluated in Figure 6 [right], Figure 7, Figure $^ { 8 , }$ and Figure 12. With single-step clustering, we refer to the fact that, during training, this model is executed on each training clustering task for only 1 step, and the training loss is computed with ${ \cal C } ^ { ( 1 ) }$ , the output of the [Step 1] update. This makes the training extremely efficient as we are training a single transformer layer without any recursion. However, we see that this model learns some clustering mechanisms as we can reapply it for $T$ and progressively improve clustering quality, much like applying the "assign/center-update" step in Lloyd's algorithm iteratively

Figure 6 [left] is ablating the role of the token embedding in $F _ { d : k } ^ { \gamma }$ by evaluating an alternate form of multi-step clustering that only applies the attention updates in (4) iteratively as follows:

• [Step 0] ${ X , C ^ { ( 0 ) }  ( \mathrm { { 0 H } } ) / ( \mathrm { { B N } } )  \bar { X } ^ { ( 0 ) } , \bar { C } ^ { ( 0 ) } }$

• [Step 1]X(0), Č(0) → eq. (4) → X(1), ¯(1),

• [Step 2]X(1), Č(1) → eq. (4) → X(2), Č(2),

• …,

$[ \mathrm { S t e p ~ T } ] \bar { X } ^ { ( T - 1 ) } , \bar { C } ^ { ( T - 1 ) }  \mathrm { e q . ~ } ( 4 )  \bar { X } ^ { ( T ) } , \bar { \mathbf { C } } ^ { ( T ) } ]$

• [final $\mathbf { s t e p } ] \bar { \mathbf { C } } ^ { ( T ) }  \mathbf { C } ^ { ( T ) }$

The results in Figure 6 [left] shows that just applying the attention updates recursively does not improve the clustering quality, and both the token embeddings $\mathrm { ( O H ) } / \mathrm { ( B N ) }$ and the attention updates (4) in the trained model $F _ { d : k } ^ { \gamma } ( \cdot , \cdot ; \Theta )$ are essential for the trained transformer to successfully cluster iteratively

## D Clustering Task Generation

To train the transformer, we generate clustering tasks $( \pmb { X } , \pmb { C } ) \sim \mathcal { D }$ from a distribution D where X denotes the points to be clustered, and C denotes the initial cluster centers. We consider mixture distributions (such as mixture of normal distributions or mixture of cauchy distributions, etc) to generate $\pmb { X } \in [ 0 , 1 ] ^ { d \times n }$ in the following manner:

• Randomly sample k centers $\{ l _ { j } \sim \mathcal { U } ( 0 , 1 ) ^ { d } , j \in [ k ] \}$ in d dimensions uniformly from the unit cube $[ 0 , 1 ] ^ { d }$

• Randomly sample k d-dimensional scales $\{ \pmb { s } _ { j } \sim \mathcal { U } ( 0 , 1 ) ^ { d } , j \in [ k ] \}$ if using anisotropic distributions else $\pmb { s } _ { j } = \mathbf { 1 } _ { d } \forall j \in [ k ]$

• Randomly sample k weights $\{ w _ { j } , j \in [ k ] \}$ from the k-dimensional simplex $\Delta ^ { k }$ if using unequally weighted mixtures else $w _ { j } = 1 / k \forall i \in [ k ]$

• Create a mixture distribution $\mathcal { T } \triangleq \{ ( l _ { j } , s _ { j } , w _ { j } ) _ { j \in [ k ] } \}$ and sample n samples $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } } \sim \mathcal { T }$ in d dimensions to get $\hat { \boldsymbol X } \in \mathbb R ^ { d \times n }$

$X \gets \mathsf { M i n M a x S c a l e r } ( \hat { X } )$

Note that, as the data dimensionality d increases, the diameter of the set of samples increases. Given the set of samples, we randomly sample k points from $\boldsymbol { X }$ to define the set of initial centers $C \in [ 0 , 1 ] ^ { d \times k }$

The motivation behind generating clustering tasks where $\pmb { X } \in [ 0 , 1 ] ^ { d \times n } , \pmb { C } \in [ 0 , 1 ] ^ { d \times k }$ is that this aligns with our theoretical analyses, and any general k-means clustering problem with $\hat { \boldsymbol X } \in \mathbb R ^ { d \times n } , \hat { \boldsymbol C } \in \mathbb R ^ { d \times k }$ can be mapped accordingly given the translation and scaling invariance of the k-means clustering problem. Let $v _ { | \mathrm { b } } \in \mathbb { R } ^ { d }$ be the vector of the per-dimension minimum values for all the points in $\hat { X }$ and $\hat { C }$ combined. Then setting $\bar { X }  \hat { X } - v _ { \vert \mathbf { b } } \mathbf { 1 } _ { n } ^ { \top } , \bar { C }  \hat { C } - v _ { \vert \mathbf { b } } \mathbf { 1 } _ { k } ^ { \top }$ moves all the points and centers to the nonnegative quadrant $\mathbb { R } _ { \geq 0 } ^ { d }$ . Let $v _ { \mathsf { u b } } \in \mathbb { R } _ { + }$ be the single largest scalar value in $\bar { X } , \bar { C }$ We assume that this $v _ { \mathrm { u b } } > 0$ , otherwise the problem is trivial with $\hat { \pmb { X } } = \mathbf { 0 } _ { d \times n } , \pmb { C } = \mathbf { 0 } _ { d \times k }$ . Then setting $X  \bar { X } / v _ { \mathsf { u b } } , C  \bar { C } / v _ { \mathsf { u b } }$ ensures that $\pmb { X } \in [ 0 , 1 ] ^ { d \times n } , \pmb { C } \in [ 0 , 1 ] ^ { d \times k }$ while the k-means problem remains the same due to the translation and scaling invariance.

## E Computational Resources and Implementation

All experiments were executed on a Intel i7 Core CPU (16 threads, 64GB memory), and a NVIDIA V100 GPU (32GB memory). Each experiment was executed with 10 random seeds and all results are aggregated across these 10 trials. The implementation is in Pytorch and available at this GitHub repository (https://github.com/rithram/kmeans-trf). We train models with 300 different configurations — 50 for each of the 6 different combinations of regularized min, softmax vs sparsemax, and initial token embedding, one-hot (OH) vs binary (BN) vs none (NA). With 10 repetitions for each, we train a total of 3000 models, each taking around 15-20 minutes to train, for a total training compute time of 1000 hours. The loss surface generation for Figure 3 took about 15 minutes per model for a total of 1 hour of compute. Each of the inference evaluations in Section 4 took around 5 minutes each for total of $( 2 5 + 2 + 4 ) \times 5 \approx 1 5 0$ minutes. The timed runs in Figure 1 and Table 1 in Section 2 and Figure 17 in Appendix F are performed on a NVIDIA RTX PRO 3000 Blackwell Generation Laptop $\mathrm { G P U }$

## F Ablative Experiments

In this section, we present ablations across various axes, presenting the training loss trajectory as the training progresses, and the relative validation loss. Given a transformer parameterized with $\Theta ,$ for an input $( \pmb { X } , \pmb { C } ) \sim \mathcal { D }$ sampled from the distribution D of clustering tasks, we will denote the updated centers after a single transformer layer as $\bar { C } ( \Theta )$ . Then, the training loss $\mathbb { E } _ { ( X , C ) \sim \mathcal { D } } \tilde { \mathcal { L } } _ { \Omega } ^ { \tau } ( X , \bar { C } ( \Theta ) )$ (defined in Equation (6)) is the smoothed upper-bound of the desired k-means loss $\mathbb { E } _ { ( X , C ) \sim \mathcal { D } } \mathcal { L } ( X , \bar { C } ( \Theta ) )$ (defined in Equation (1)) with $\mathcal { L } ( X , \bar { C } ( \Theta ) ) \leq \tilde { \mathcal { L } } _ { \Omega } ^ { \tau } ( X , \bar { C } ( \Theta ) )$ , and the inequality approaching equality as the temperature $\tau  0$ We define the relative validation loss as the ratio of the k-means objective with the centers $\bar { C } ( \Theta )$ obtained with a single transformer layer to the k-means objective with the centers $C ^ { \star }$ generated from a single iteration of Lloyd's algorithm (Algorithm 2) with the same set of points X and initial centers C. Thus the relative validation error is given by

$$
\mathbb { E } _ { ( X , C ) \sim \mathcal { D } } \frac { \mathcal { L } ( X , \bar { C } ( \Theta ) ) } { \mathcal { L } ( X , C ^ { \star } ) } .\tag{20}
$$

Note that, for estimating the training loss and the relative validation loss, we use different sets of clustering tasks, and the clustering tasks in the validation set are not used for training. A relative validation error of less than 1 denotes that the single transformer layer outperforms Lloyd's algorithm (on average) when considered for a single iteration.

Each figure in these ablations has 6 curves corresponding to (i) the min-regularizer Ω — the negative Shannon entropy or NE-regularizer (denoted by NE) vs the L2-regularizer (denoted by L2), and (ii) the use of the one-hot initial token embeddings $( E _ { X } ^ { 0 \mathrm { H } } , E _ { C } ^ { 0 \mathrm { H } } )$ defined in (0H) with $d _ { \mathsf { e m b } } = d + k$ and denoted by OH vs the binary initial token embeddings $( E _ { X } ^ { \tt B N } , E _ { C } ^ { \tt B N } )$ defined in (BN) with $d _ { \mathsf { e m b } } = d + \lceil \log _ { 2 } k \rceil$ and denoted by BN vs no initial token embedding with $d _ { \mathsf { e m b } } = d$ denoted by NA. Thus we have 6 lines corresponding to labels BN:NE, BN:L2, NA:NE, NA:L2, OH:NE, OH:L2. Each experiment is executed for 10 trials with different random seeds, and each curve reports the median performance metrics, with a ribbon around it denoting the inter-quartile range of the performance metrics.

![](images/d37f2c35b41f08716a53f14ec327437dbb9d7311e237ffd21c7c0e5aefcfed13.jpg)

![](images/a20ac0f352853f36990924556d53287366d65a5177de5227fc5913f8556180a3.jpg)  
(a) Clustering tasks in d = 4 dimensions.  
(b) Clustering tasks in d = 32 dimensions.

Figure 10: Training loss convergence (top row) and relative validation loss from Equation (20) (bottom row) for varying values of the min-regularization penalty τ in the training loss $\tilde { \mathcal { L } } _ { \Omega } ^ { \tau }$ (the smoothed upper-bound of the discrete non-differentiable k-means objective $\mathcal { L } )$ . This is an extended version of Figure 2a. We consider $\tau \in \{ 1 , 0 . 2 5 , 0 . 1 4 , 0 . 1 \}$ . These set of results correspond to a training with clustering tasks involving $n = 5 1 2$ points and $k = 6$ clusters with $d = 4$ (Figure 10a) and $d = 3 2$ (Figure 10b) dimensions. Here the initial learning rate $\eta = 0 . 0 1$ for the Adam optimizer with $M = 1 0 0 0 0$ training steps with a task batch-size $B = 3 2 ^ { \circ }$

Varying the min-regularization penalty $\tau$ in the smoothed loss $\tilde { \mathcal { L } } _ { \Omega } ^ { \tau } .$ In Figure 10, we present the effect of varying the min-regularization penalty τ in the smoothed upper-bound $\tilde { \mathcal { L } } _ { \Omega } ^ { \tau } { : }$ highlighting that, as the penalty τ decreases, the training loss (the upper-bound) becomes tighter (and thus has lower values), and the performance on the validation tasks actually starts matching or even outperforming 1-step Lloyd's algorithm. Note that, for the larger value of $d ,$ the number of dimensions, the min-regularization penalty τ does not need to be too low for strong performance. This result also highlights that the L2-regularized min provides tighter upper-bounds, which translate to better validation performance, with the difference most apparent at large values of $\tau$ and the smaller data dimensionality of $d = 4 ,$ when the penalization is stronger in general, and the upper-bounds are looser but smoother. The training loss converges faster at the higher values of $\tau ,$ though the validation performance highlights that the faster training loss convergence does not necessary guarantee strong validation performance because of the gap between the smoothed upper-bound and the actual discrete clustering objective we care about.

Varying the learning rate η in Algorithm 1. In Figure 11, we present the effect of varying the initial learning rate in the Adam optimizer. The results indicate that, for low and moderately high dimensions $d ,$ a low learning rate of 0.001 shows monotonic convergence but is slower than a learning rate of 0.01. On the other hand, larger values of learning rates such as 0.1 and 1.0 show that both the training and the validation losses do not decrease monotonically, especially at the higher d = 32 dimensions. The larger learning rates $( \eta = 0 . 1 , 1 . 0$ for $d = 3 2$ and $\eta = 1 . 0$ for d = 4) converge to a worse local minima, and thus do not generalize well. In all cases where the training converges favourably, the L2-regularized min continues to converge to a better minima (both in terms of training loss and validation) than the NE-regularized min (controlling for all other factors).

![](images/7bd0d7e3e06c61a8cbf40b7a19be5eb7a40b9981c6fc981a25e109f48edd82fd.jpg)  
(a) Clustering tasks in d = 4 dimensions.

![](images/eac42cd8dcc184d6bd6525f7be366dc4b4bfedb6fd4802d792abed62386a5be5.jpg)  
(b) Clustering tasks in d = 32 dimensions.  
Figure 11: Training loss convergence (top row) and relative validation loss from Equation (20) (bottom row) for varying values of the initial learning rate η for the Adam optimizer selected from {0.001, 0.01, 0.1, 1.0} with $M = 5 0 0 0$ training steps with a task batch-size $B = 3 2$ . These set of results correspond to a training with clustering tasks involving $n = 5 1 2$ points and k = 6 clusters with d = 4 (Figure 11a) and $d = 3 2$ (Figure 11b) dimensions with a min-regularization penalty $\tau = 0 . 1$ 1

![](images/efbe83c584182f8104c7001cf0d78168aa249feca8eb8e57fb36120382ccdc96.jpg)

![](images/3c1861dfc0b474f09688f97f8c2a8e4a88e6665d79b42e0171fa2e7883711fef.jpg)

![](images/f39a5a62ff60b384d3212bffab7dca398f4420c04844a886a9fa9cfc4f9a234a.jpg)

![](images/06ed1d25eb9f13968cb39fa78fd214dbf943dfa8dfe00614c225125876c60edc.jpg)  
Figure 12: “Length-generalization" of learned models. We consider models trained with $n = 5 1 2 , d =$ $3 2 , k = 1 0 , \tau = 0 . 1$ utilized for $T = 2 0$ step clustering on tasks with {128, 256, 1024, 2048} points. We only consider L2-regularized min and the three forms of token embeddings, OH, BN and NA as described in Figure 2. The performance of Lloyd's algorithm is plotted in black

Evaluating the “length generalization" of models, learning on one n, and generalizing to other values of n Here we evaluate the ability of these trained models to length generalize — the ability of models to be trained on problems of a certain size, but generalize to problems of different (often larger) sizes. In the context of clustering, we train the models on (single-step) clustering tasks with n = 512 samples (for a mixture of normal distributions), and then evaluate their (multi-step) clustering performance on problem with varying number of samples (again from the same distribution as it was trained on) in Figure 12. The results show that the learned model can generalize to larger clustering problems. The models using the (0H) and (BN) token embeddings outperform Lloyd's iterations, both for smaller problems (with 128 and 256

![](images/9559f34aac152fdc82019027d3a62c60871cff6ce8a49c09e24021fb6ef69d26.jpg)  
(a) Clustering tasks in d = 4 dimensions.

![](images/1594f32be7f44171a7dedf445fae8e3013bd1be2155cd3151302dac16a2f7725.jpg)  
(b) Clustering tasks in d = 32 dimensions.  
Figure 13: Training loss convergence (top row) and relative validation loss from Equation (20) (bottom row) for varying values of the number of points n in the clustering tasks used for training the transformers, with $n \in \{ 1 2 8 , 2 5 6 , 5 1 2 , 1 0 2 4 \}$ . These set of results correspond to a training with clustering tasks involving $k = 6$ clusters with $d = 4$ (Figure 13a) and d = 32 (Figure 13b) dimensions. Here we use the initial learning rate $\eta = 0 . 0 1$ for the Adam optimizer with M = 10000 training steps, a task batch-size $B = 3 2$ and the min-regularization penalty $\tau = 0 . 1$

Varying the number of points n to be clustered in the training tasks. In Figure 13, we study effect of varying the number of points n in the clustering tasks $( X , C )$ with $\pmb { X } \in \mathbb { R } ^ { d \times n }$ used during training, selecting $n \in \{ 1 2 8 , 2 5 6 , 5 1 2 , 1 0 2 4 \}$ for clustering tasks in $d \in \{ 4 , 3 2 \}$ dimensional Euclidean space. The results indicate that the number of points in the clustering tasks do not seem to have a significant effect on the training convergence or in-distribution generalization for any data dimensionality $d ,$ min-regularizer Ω or initial token embedding scheme (OH vs BN vs NA). This combined with our length-generalization results in Figure 12 indicate that a good strategy would be to train the transformer with clustering tasks with a small n as that would be computationally efficient, and still be able to use this transformer for larger clustering tasks with larger number of points. Note that, for a fixed d and k and initial token embedding choice (OH vs BN vs NA), the number of learnable parameters remains the same as we vary n in the training clustering tasks

Varying the data dimensionality d of the clustering tasks. In Figure 14, we study the effect of the data dimensionality d of the clustering tasks used to train the transformer. We consider two different values for the min-regularization penalty $\tau ,$ with smaller value of τ implying tighter upper-bounds to the discrete clustering loss. For the larger penalty value of $\tau = 0 . 2 5$ in Figure 14a, we can see that the L2-regularized min based smoothed loss converges and generalizes better than the NE-regularized min especially at a low $d = 4 ;$ also note that the OH token embedding converges faster than the BN token embeddings as previously discussed (though both generalize comparably). However, as the data dimensionality d increases, this difference between L2-regularized and NE-regularized min vanishes. This is because, as we consider all $\pmb { X } \in [ 0 , 1 ] ^ { d \times n }$ and $C \in [ 0 , 1 ] ^ { d \times k }$ , the pairwise squared Euclidean distances generally increases with $d ,$ and thus the regularized min produces tighter upper-bounds with increasing d for a fixed min-regularization penalty τ. We will discuss this further in Appendix H. Note that the number of parameters in the transformer model grows quadratically with $d ,$ but we do not see any clear adverse effect of this increase in the number of learnable parameters.

![](images/8502df28e01743ed520ce4f3a065c84c842fb00ca4c5976609ae0618bbacf03e.jpg)

![](images/b78d00f4dc614c153f7768d6241a9ed9cef61f7be3e89f9ee426de68c61f703b.jpg)  
(a) min-regularization penalty τ = 0.25.  
(b) min-regularization penalty $\tau = 0 . 1$  
Figure 14: Training loss convergence (top row) and relative validation loss from Equation (20) (bottom row) for varying values of the dimensionality d of the clustering tasks used for training the transformers, with $d \in \{ 4 , 8 , 1 6 , 3 2 \}$ . This is an extended version of Figure 2b. These set of results correspond to a training with clustering tasks involving $n = 5 1 2$ points, $k = 6$ clusters, trained with the smoothed loss with a loss min-regularizer penalty $\tau = 0 . 2 5$ (Figure 14a) and $\tau = 0 . 1$ (Figure 14b). Here the initial learning rate $\eta = 0 . 0 1$ for the Adam optimizer with $M = 1 0 0 0 0$ training steps, each with a task batch-size $B = 3 2$

Varying the number of clusters k in the training tasks. In Figure 15, we carefully study the effect of varying the number of clusters k in the training tasks to learning convergence and generalization. To understand the interplay of the number of clusters $k ,$ data dimensionality d and the min-regularization penalty τ, we consider a pair of values of d and for τ, and for each of the 4 combinations, we vary $k \in \{ 6 , 1 0 , 1 6 , 2 5 \}$ First, in Figure 15a, we consider a low $d = 4$ and high penalty $\tau = 1$ , implying a looser smoother upper-bound. In this case, the L2-regularized min is able to learn and somewhat generalize, but is unable to match the results of a single Lloyd's iteration (the black dotted line) for any value of $k ;$ the NE-regularized min is unable to learn at all as is evidenced by the high relative loss on the single-step validation clustering tasks. One observation is that the convergence (and generalization) of BN token embedding relatively deteriorates with increasing k as one would expect from our results in Theorem 3.1, where the Lipschitz constant depends on $\sqrt { d + \left\lceil \log _ { 2 } k \right\rceil }$ , and thus increases with k (thereby hurting convergence). In contrast, the OH token embedding does not seem to be affected by the increase in $k ,$ which corroborates our theoretical results in Theorem 3.1 where the Lipschitz constant would depend on $( d + 1 ) / \sqrt { d + k } ,$ and thus is not adversely affected by increasing k. Fixing the dimension $d = 4 ,$ and lowering the min-regularization $\tau = 0 . 1$ in Figure 15b, thereby using a tighter (less smooth) upper-bound for training, we see that the NE-regularized min is able to somewhat learn and generalize for smaller values of $k ,$ with its performance deteriorating with increasing $k ;$ L2-regularized min with OH token embedding does significantly better and outperforms Lloyd's for all values of k considered. For smaller values of $k ,$ the BN token embedding is able to outperform Lloyd's but converges and generalizes slower than OH as our theory suggests; for larger values of $k ,$ the convergence and generalization of BN token embedding slows down considerably, and even underperforms the model with no token embedding (NA). As we increase the data dimensionality to $d = 3 2$ and utilize a larger min-regularization penalty $\tau = 1$ in Figure 15c, we see that L2-regularized min with OH and BN token embeddings outperforms Lloyd's for all values of k. However, with NE-regularized min, the smoothed upper-bound $\tilde { \mathcal { L } } _ { \Omega } ^ { \tau }$ for a fixed τ gets progressively looser with larger values of $k ,$ and this is evident in the worsening validation losses. Finally, fixing the data dimensionality to $d = 3 2$ and lowering the min-regularization penalty to $\tau = 0 . 1$ for a tighter clustering loss upper-bound in Figure 15d, we see that both L2-regularized and NE-regularized min for both OH and BN token embeddings outperform Lloyd's on the validation tasks. The only thing of note here is that, at the largest value of $k = 2 5$ , the model with the OH token embeddings experiences intermediate numerical issues for $3 / 1 0$ random trials in Figure 15d which throws the model off, and the training does not recover in the allotted 10000 steps, thereby broadening the inter-quartile range unfavourably; note that the median performance continues to outperform Lloyd's. We would like to note a couple of points regarding this:

![](images/8e3ed117edd57b487227f4e77dc24b856b9def9846ca2670132eb1001691b10f.jpg)  
(a) d = 4, τ = 1.  
(b) d = 4, τ = 0.1.  
(c) d = 32, τ = 1.  
(d) d = 32, τ = 0.1.  
Figure 15: Training loss convergence (top row) and relative validation loss from Equation (20) (bottom row) for varying values of the number of clusters k in the clustering tasks used for training the transformers, with $k \in \{ 6 , 1 0 , 1 6 , 2 5 \}$ . These set of results correspond to a training with clustering tasks with $n = 5 1 2$ points. To highlight the interplay of the dimensionality d and the min-regularization penalty τ, we consider four cases: $d = 4 , \tau = 1$ (Figure 15a), $d = 4 , \tau = 0 . 1$ (Figure 15b), $d = 3 2 , \tau = 1$ (Figure 15c), and $d = 3 2 , \tau = 0 . 1$ (Figure 15d). In all cases, the initial learning rate $\eta = 0 . 0 1$ for the Adam optimizer with $M = 1 0 0 0 0$ training steps each with a task batch-size B = 32.

• First, we did not use the standard practice of gradient clipping in any of our results, and our results indicate that the learning problem is well-structured enough for the training to easily converge in almost all cases (also highlighted by the loss landscapes in Figure 3).

• Second, for the 3/10 random trials where the training gets stuck, adding gradient clipping mitigates the issue of the numerical instability, and these previously failing training runs converge as expected consistent with the other random trials. We presented the results in Figure 15d without gradient clipping to be consistent across all evaluations.

• Finally, this only occurs with the OH token embedding for both NE (softmax) and L2 (sparsemax) regularization, where $d _ { \mathbf { e m b } } = d + k = 5 7$ with $d = 3 2 , k = 2 5$ , and the training is not able to find the right basin of attraction in this relatively high $O ( d _ { \mathrm { e m b } } ^ { 2 } ) \sim$ 3600 dimensional loss landscape once the training is destabilized. In contrast, for the same setup (same random seed and clustering tasks for training), the training converges fine with the BN token embedding, where $d _ { \mathbf { e m b } } = d + \lceil \log _ { 2 } k \rceil = 3 7$ with $d = 3 2 , k = 2 5 , \lceil \log _ { 2 } k \rceil = 5$ , implying that the loss landscapes are easier to navigate in a relatively lower dimensionality $O ( d _ { \mathrm { e m b } } ^ { 2 } ) \sim 1 6 0 0$

![](images/b45996268199d7f48a36ff05fba7b7a6d14c36fb17885835d8df9edc6d9bf56c.jpg)

![](images/1d2bc9de2dc1b278bb65a5151c6e04583cd84a19d97044e47ec40d15aa679378.jpg)

(a) k = 6, τ = 1.  
![](images/464f1fab0ae8771764f30f0cd0462f00f916bee8078825774b1afe17a5f0a477.jpg)  
(c) k = 6, τ = 0.1.

(b) k = 10, τ = 1.  
![](images/384d728a9f08977b719e76555177ec610317b50202498548721470db5dec2618.jpg)  
(d) k = 10, τ = 0.1.  
Figure 16: Training loss convergence (top row) and relative validation loss from Equation (20) (bottom row) for varying the distribution of the points to the clustered. These set of results correspond to a training with clustering tasks with $n = 5 1 2$ points in $d = 3 2$ dimensions, and is an extended version of Figure 4. To highlight the interplay of the number of clusters k and the loss temperature τ, we consider four cases: $k = 6 , \tau = 1$ (Figure 16a), $k = 1 0 , \tau = 1$ (Figure 16b), $k = 6 , \tau = 0 . 1$ (Figure 16c), and $k = 1 0 , \tau = 0 . 1$ (Figure 16d). In all cases, the initial learning rate $\eta = 0 . 0 1$ for the Adam optimizer with $M = 1 0 0 0 0$ training steps and a per-step task batch-size $B = 3 2$

Varying the distribution of the data from which the points are generated for the training tasks. In Figure 16, we build upon Figure 4, and provide an extended evaluation of the effect of the data distribution from which the set of points $\pmb { X } \in [ 0 , 1 ] ^ { d \times n }$ are generated for any clustering tasks $( X , C )$ . As with Figure $^ { 4 , }$ we see that our proposed learning scheme is unable to learn from and generalize to clustering tasks generated from a mixture of cauchy distributions under any of the configuarations considered. Focusing on the remaining distributions, namely gumbel, laplace, lognormal, and normal, we see that, at a larger min-regularization of $\tau = 1$ in Figure 16a and Figure 16b, the L2-regularized min provides better generalization than the NE-regularized min, with NE-regularized min failing to generalize at all with the gumbel and laplace distributions; this behaviour is more evident with the larger $k = 1 0$ value in Figure 16b. With a smaller min-regularization penalty $\tau = 0 . 1$ providing tighter clustering loss upper-bounds for training in Figure 16c and Figure 16d, we see that both NE-regularized and L2-regularized min with OH and BN token embeddings can outperform Lloyd's across all data distributions.

![](images/497ba0d5a5bc84ad6e88325ace302d5511af5e2342d7aa5821124c94532a061c.jpg)

![](images/e496d747c2d47fb3109f60007c97ad76fac944912fc7b8d64d7769f659fe4547.jpg)

![](images/2ba6d7174fc1fa8b1ad3f0b73eec32b51634c003054e68b8cfb578a1475da1e5.jpg)  
(a) k = 16, d = 16, τ = 1.  
(b) k = 25, d = 32, τ = 1.  
Figure 17: Training loss convergence (top row) and relative validation loss from Equation (20) (bottom row) in terms of actual training time for varying values of $n \in \{ 2 5 6 , 5 1 2 , 1 0 2 4 \}$ . In one case, we use $k = 1 6 , d = 1 6$ with $\tau = 1 \ ( { \mathrm { F i g u r e \ 1 7 a } } )$ , while in another, we use $k = 2 5 , d = 3 2$ , again with $\tau = 1 \ ( { \mathrm { F i g u r e \ 1 7 b } } )$ . Here we limit ourselves to the L2-regularized min, and compare the OH and BN token embeddings, giving us only the BN:L2 and the OH:L2 curves.

Interplay of the lower computational overhead and slower convergence of the leaner (BN). The results in Figure 1 and Table 1 indicate that the proposed BN token embeddings can provide significant computational gains over the existing OH token embeddings primarily on account of having smaller embeddings sizes $d _ { \mathsf { e m b } }$ (OH with (d + k) vs BN with $\left( d + \lceil \log _ { 2 } k \rceil \right) )$ . However, the results in Theorem 3.1 indicate that the model with the BN token embeddigns will have slower convergence guarantees (in terms of the number of optimization steps). To understand their interplay, we consider two settings of $( d , k )$ , and compare the convergence (of training and validation loss) in terms of actual elapsed time in Figure 17. These training runs use gradient clipping to ensure that all the model trainings avoided any gradient explosions and numerical issues. With moderately small $d = k = 1 6$ in Figure 17a, we see that the model with the BN token embeddings converges signficantly slower than the model with the OH token embedding, though both appear to eventually converge close to each other, and better than single step Lloyd's update for all values of n. However, as the data dimensionality d and the number of clusters k grows to $d = 3 2 , k = 2 5$ in Figure 17b, we see that, initially, the model with BN token embeddings converges faster in terms of the runtime, with both methods eventually matching or outperforming single-step Lloyd's updates by the end of their 10000 training steps (which take different amounts of time for both).

## G Additional Figures

![](images/1309d8a296b149814ea1e7f424948973a17dd8ee3acb4fe4851652b88cf4af82.jpg)  
Figure 18: Generalization to different distribution families. Larger version of Figure 8. The models are trained on one distribution family (each column maps to a training family), and tested for multi-step (T = 20) clustering from a different distribution family (each row maps to a test family). We consider 5 distributions — cauchy, gumbel, laplace, lognormal and normal — giving us a 5 × 5 pairwise evaluations. Note that the pairwise evaluations on the "diagonal" correspond to training and evaluations tasks being sampled from the same distribution (though the tasks themselves are different, and the training only considers single-step clustering while we are evaluating multi-step clustering). We only consider models trained with the L2-regularized min and the three forms of token embeddings, OH, BN and NA as described in Figure 2. Each curve corresponds to the inter-quartile range aggregated over 10 trials. The performance of Lloyd's is in black.

![](images/0f24cfd4256c1fefb83e267934ed7549d01491b929720a37f7641979594d0aeb.jpg)  
Figure 19: Pairwise task dissimilarities. A version of Figure 9 with 16 tasks per distribution family. We use the 16 tasks from each of the families and compute the pairwise Earth-mover distance between their respective distribution of the squared Euclidean distances $\left\{ \left\| X _ { : , i } - C _ { : , j } \right\| ^ { 2 } , i \in [ n ] , j \in [ k ] \right\}$ between the points X and the initial centers $^ { C , }$ giving us a $8 0 \times 8 0$ matrix. Each row and column corresponds to a single task, and the tasks are arranged into family-based blocks in the following order: normal, cauchy, gumbel, laplace, lognormal. Light shades show low distances. The "diagonal" corresponds to tasks distances to themselves, and thus are zero (the lightest shade). cauchy tasks are only similar to other cauchy tasks, and significantly dissimilar to tasks from other distribution families, with the laplace distribution appearing the least dissimilar (but still quite dissimilar).

## H Smoothed Clustering Objective via Regularized min

For the purposes of clustering, our main target is the non-differentiable per-point k-means loss $L : \mathbb { R } ^ { d } \times \mathbb { R } ^ { d \times k } $ R, given a sample $\pmb { x } \in \mathbb { R } ^ { d }$ and cluster centers $C = [ c _ { 1 } , \dots , c _ { k } ] \in \mathbb { R } ^ { d \times k }$

$$
L ( \pmb { x } , \pmb { C } ) = \operatorname* { m i n } _ { j \in [ k ] } \left\| \pmb { x } - \pmb { c } _ { j } \right\| ^ { 2 } ,\tag{21}
$$

with the complete k-means objective in Equation (1) given by $\textstyle \sum _ { i \in [ n ] } { L ( { \pmb x } _ { i } , { \pmb C } ) }$ . We can obtain a differentiable trainable surrogate for $L ( { \pmb x } , C )$ in a few steps. First, by defining function $R : \mathbb { R } ^ { d } \times \mathbb { R } ^ { d \times k }  \mathbb { R } ^ { k }$ where $\pmb { r } = R ( \pmb { x } , \pmb { C } )$ has $r _ { j } \triangleq \Vert \pmb { x } - \pmb { C } _ { : , j } \Vert ^ { 2 }$ for each $j \in [ k ]$ , we have equivalently

$$
L ( \pmb { x } , \pmb { C } ) = \operatorname* { m i n } _ { j \in [ k ] } r _ { j } \quad \mathrm { w h e r e } \quad r = R ( \pmb { x } , \pmb { C } ) .\tag{22}
$$

Let $\Delta ^ { k }$ be the $( k - 1 )$ -dimensional probability simplex with $\Delta ^ { k } \ = \ \ \{ \pmb { p } \in \mathbb { R } _ { > 0 } ^ { k } \ : \ \lVert \pmb { p } \rVert _ { 1 } \ = \ 1 \}$ . Since mir $\begin{array} { r } { { ^ { 1 } \lambda } { \in \Delta ^ { k } \left. \lambda , r \right. } = \operatorname* { m i n } _ { j \in [ k ] } r _ { j } } \end{array}$ , we could also equivalently say, again with $\pmb { r } = R ( \pmb { x } , \pmb { C } )$

$$
L ( \pmb { x } , C ) \triangleq \mathtt { L } ( \pmb { r } ) = \left. P ( \pmb { r } ) , \pmb { r } \right. , \quad \mathrm { w h e r e } ~ P ( \pmb { r } ) \triangleq \underset { \pmb { x } \in \Delta ^ { k } } { \arg \operatorname* { m i n } } \left. \lambda , \pmb { r } \right.\tag{23}
$$

We haven't changed $L ( { \pmb x } , { \pmb C } ) _ { \dag }$ which remains non-differentiable. However, we can regularize the minimization of $\lambda ,$ obtaining a surrogate $\tilde { L } _ { \Omega } ^ { \tau } ( { \pmb x } , C )$ , defined as

$$
\tilde { L } _ { \Omega } ^ { \tau } ( \boldsymbol { x } , C ) \triangleq \tilde { \boldsymbol { \mathrm { L } } } _ { \Omega } ^ { \tau } ( \boldsymbol { r } ) = \langle P _ { \Omega } ^ { \tau } ( \boldsymbol { r } ) , \boldsymbol { r } \rangle , \quad \mathrm { w h e r e ~ } P _ { \Omega } ^ { \tau } ( \boldsymbol { r } ) \overset { \Delta } { = } \underset { \boldsymbol { \lambda } \in \Delta ^ { k } } { \mathrm { a r g } \mathrm { m i n } } \langle \boldsymbol { \lambda } , \boldsymbol { r } \rangle + \tau \Omega ( \boldsymbol { \lambda } ) ,\tag{24}
$$

where $\tau > 0$ is the penalty on the regularization $\Omega : \Delta ^ { k }  \mathbb { R } .$ with $\tau = 0$ yields the unregularized arg min. If Ω is strongly convex, $\tilde { \mathrm { L } } _ { \Omega } ^ { \tau } ( r )$ is differentiable with respect to r. Note that, for any $\tau > 0 , P _ { \Omega } ^ { \tau } ( \pmb { r } ) = P _ { \Omega } ^ { 1 } ( \pmb { r } / \tau )$ and thus we can just operate with $P _ { \Omega } ^ { 1 } ( \pmb { r } )$ . We will write $P _ { \Omega } ^ { 1 }$ as $P _ { \Omega }$ for the ease of exposition with:

$$
P _ { \Omega } ( r ) \triangleq \underset { \lambda \in \Delta ^ { k } } { \arg \operatorname* { m i n } } \left. \lambda , r \right. + \Omega ( \lambda )\tag{25}
$$

For $\begin{array} { r } { \lambda \in \Delta ^ { k } , \mathrm { i f } \Omega ( \lambda ) = \sum _ { j \in [ k ] } \lambda _ { j } \log \lambda _ { j } , } \end{array}$ the negative Shannon entropy, then the corresponding $P _ { \mathrm { N E } } ( { \pmb r } / \tau )$ are the softmax weights (actually "softmin" weights) with the temperature τ, giving us the following "soft-min" k-means objective $\tilde { L } _ { \mathrm { N E } } ^ { \tau }$

$$
\begin{array} { r l } & { P _ { \mathrm { { N B } } } ( { \boldsymbol { r } } / { \tau } ) _ { j } = \frac { \exp ( - r _ { j } / \tau ) } { \sum _ { j ^ { \prime } \in [ k ] } \exp ( - r _ { j ^ { \prime } } / \tau ) } , } \\ & { \quad \tilde { L } _ { \mathtt { N B } } ^ { \tau } ( \pmb { x } , C ) \triangleq \tilde { \mathrm { L } } _ { \mathtt { N B } } ^ { \tau } ( \pmb { r } ) = \langle P _ { \mathtt { N B } } ( \pmb { r } / \tau ) , \pmb { r } \rangle = \displaystyle \sum _ { j \in [ k ] } \frac { \exp ( - r _ { j } / \tau ) r _ { j } } { \sum _ { j ^ { \prime } \in [ k ] } \exp ( - r _ { j ^ { \prime } } / \tau ) } . } \end{array}\tag{26}
$$

Note that $L ( \pmb { x } , \pmb { C } ) \leq \tilde { L } _ { \mathrm { N E } } ^ { \tau } ( \pmb { x } , \pmb { C } )$ with the equality achieved when $\tau  0$ . In the learning procedure, we use Equation (26) instead of Equation (21) (or Equation (23)) to have a differentiable objective.

An alternative choice for the regularization for $\lambda \in \Delta ^ { k }$ is $\Omega ( \lambda ) = 1 / { 2 } \left\| \lambda \right\| _ { 2 } ^ { 2 }$ yielding

$$
P _ { \mathrm { { L } 2 } } ( r ) = \underset { \lambda \in \Delta ^ { k } } { \mathrm { a r g } \operatorname* { m i n } } \left. \lambda , r \right. + 1 / 2 \left. \lambda \right. _ { 2 } ^ { 2 } = \underset { \lambda \in \Delta ^ { k } } { \mathrm { a r g } \operatorname* { m i n } } 1 / 2 \left. r \right. ^ { 2 } + \left. \lambda , r \right. + 1 / 2 \left. \lambda \right. _ { 2 } ^ { 2 } = \underset { \lambda \in \Delta ^ { k } } { \mathrm { a r g } \operatorname* { m i n } } 1 / 2 \left. r + \lambda \right. ^ { 2 } ,
$$

the closest point to $\left( - r \right)$ in $\Delta ^ { k }$ . For $\tau > 0$ this yields the sparsemax weights $P _ { \mathrm { L } 2 } ( { \pmb r } / \tau )$ , which can be computed as:

$$
\begin{array} { r } { P _ { \mathrm { L } 2 } ( r / \tau ) _ { j } = [ - ( r _ { j } / \tau ) - \kappa ( - r / \tau ) ] _ { + } , \quad \mathrm { ~ w i t h ~ } [ a ] _ { + } = \operatorname* { m a x } \{ a , 0 \} , a \in \mathbb { R } , } \end{array}\tag{27}
$$

where $\kappa : \mathbb { R } ^ { k }  \mathbb { R }$ is the appropriately defined threshold function such that, for any $\begin{array} { r } { z \in \mathbb { R } ^ { k } , \sum _ { j \in [ k ] } [ z _ { j } - } \end{array}$ $\kappa ( z ) ] _ { + } = 1 ;$ see Wang and Carreira-Perpinán [2013] and Martins and Astudillo [2016, Proposition 1]. For any $z \in \mathbb { R } ^ { k }$ , let its entries be ordered as $z _ { ( 1 ) } \geq z _ { ( 2 ) } \geq \cdot \cdot \cdot \geq z _ { ( k ) }$ . Then $\kappa ( z )$ is computed as follows:

$$
N ( z ) = \operatorname* { m a x } \left\{ m \in [ k ] : 1 + m z _ { ( m ) } > \sum _ { j \leq m } z _ { ( j ) } \right\} , \quad \kappa ( z ) = \frac { \left( \sum _ { j \leq N ( z ) } z _ { ( j ) } \right) - 1 } { N ( z ) } .\tag{28}
$$

This results in sparser weights, where some of the $P _ { \mathrm { L } 2 } ( \pmb { r } / \tau ) _ { j }$ values can be exactly zero. This gives us the following differentiable upper-bound of the k-means objective:

$$
\tilde { L } _ { \mathrm { L 2 } } ^ { \tau } ( \boldsymbol { x } , C ) \triangleq \tilde { \mathrm { L } } _ { \mathrm { L 2 } } ^ { \tau } ( r ) = \langle P _ { \mathrm { L 2 } } ( r / \tau ) , r \rangle = \sum _ { j \in [ k ] } \left[ - ( r _ { j } / \tau ) - \kappa ( - r / \tau ) \right] _ { + } r _ { j } .\tag{29}
$$

Again $L ( \mathbf { \boldsymbol { x } } , C ) \le \tilde { L } _ { \mathrm { L } 2 } ^ { \tau } ( \mathbf { \boldsymbol { x } } , C )$ with equality now achieveable at $\tau > 0$ (as opposed to at $\tau  0$ for NEregularization) and depends on this notion of "margin" [Blondel et al., 2020]. We will also consider this as our learning loss. Using Blondel et al. [2020, Proposition 1, Approximation Error], we can show that, for the k-means objective, for the same regularization penalty $\tau > 0$ , the sparsemax-based upper-bound in Equation (29) provides a tighter upper than the softmax-based upper-bound in Equation (26).

Proposition H.1 (Blondel et al. [2020]). Suppose domain(Ω) contains all vectors in $\mathcal { Y } = \{ e _ { 1 } , \ldots , e _ { k } \}$ where $e _ { j }$ is the $j ^ { t h }$ characteristic vector. If such an Ω is γ-strongly convex and bounded, with $L \leq \Omega ( \lambda ) \leq U$ for all $\lambda \in d o m a i n ( \Omega )$ , then $\begin{array} { r } { 1 / { _ 2 } \left\| P ( z ) - P _ { \Omega } ( z ) \right\| ^ { 2 } \leq \frac { U - L } { \gamma } } \end{array}$ for any $z \in \mathbb { R } ^ { k }$

For our purposes domain $( \Omega ) = \Delta ^ { k }$ and thus, $\mathcal { V } \subset \Delta ^ { k }$ . Quantifying the tightness of the upper-bound, we have

$$
\begin{array} { r l } & { 0 < \tilde { L } _ { \Omega } ^ { \tau } ( { \pmb x } , { \pmb C } ) - L ( { \pmb x } , { \pmb C } ) = \tilde { \mathrm { L } } _ { \Omega } ^ { \tau } ( { \pmb r } ) - \mathrm { L } ( { \pmb r } ) } \\ & { \qquad = \langle P _ { \Omega } ( { \pmb r } / \tau ) - P ( { \pmb r } ) , { \pmb r } \rangle \leq \| P ( { \pmb r } ) - P _ { \Omega } ( { \pmb r } / \tau ) \| \| { \pmb r } \| . } \end{array}\tag{30}
$$

Thus, a smaller value of $\| P ( \pmb { r } ) - P _ { \Omega } ( \pmb { r } / \tau ) \|$ implies a tighter upper-bound. Note that for $\tau > 0 , P ( \boldsymbol { r } ) = P ( \boldsymbol { r } / \tau )$ and thus $\| P ( \pmb { r } ) - P _ { \Omega } ( \pmb { r } / \tau ) \| = \| P ( \pmb { r } / \tau ) - P _ { \Omega } ( \pmb { r } / \tau ) \|$ , so it suffices to bound $\| P ( z ) - P _ { \Omega } ( z ) \|$ for any $z \in \mathbb { R } ^ { k }$ For sparsemax with $\Omega ( \lambda ) = { 1 } / { 2 } \left\| \lambda \right\| _ { 2 } ^ { 2 } , \Omega$ is γ-strongly convex with $\gamma = 1$ , and $\Omega ( \lambda ) = 1 / { 2 } \left\| \lambda \right\| _ { 2 } ^ { 2 } \in [ 1 / 2 k , 1 / 2 ]$ for any $\lambda \in \Delta ^ { k }$ . Thus $1 / 2 \left\| P ( z ) - P _ { \Omega } ( z ) \right\| ^ { 2 } \leq 1 / 2 ( 1 - 1 / k ) < 1 / 2$ for any $z \in \mathbb { R } ^ { k }$ using Proposition H.1. Thus $\| P ( \pmb { r } / \tau ) - P _ { \Omega } ( \pmb { r } / \tau ) \| \le \sqrt { 1 - 1 / k } < 1$

For softmax with $\begin{array} { r } { \Omega ( \lambda ) = \sum _ { j \in [ k ] } \lambda _ { j } \log \lambda _ { j } } \end{array}$ with $0 \leq \lambda _ { j } \leq M \leq 1$ , Ω is γ-strongly convex with $\gamma = 1 / M$ Naively $M = 1$ and thus, Ω would be 1-strongly convex. When λ is one-hot, $\Omega ( \lambda ) = 0 = U$ , while $\lambda _ { j } = 1 / k \forall j \in [ k ]$ gives us $\Omega ( \lambda ) = - \log k = L$ . Thus, we have

$$
1 / 2 \left\| P ( z ) - P _ { \Omega } ( z ) \right\| ^ { 2 } \leq \frac { U - L } { \gamma } = \log k .\tag{31}
$$

Thus, for softmax, $\| P ( \pmb { r } / \tau ) - P _ { \Omega } ( \pmb { r } / \tau ) \| \le \sqrt { 2 \log k }$ giving a looser bound than with sparsemax of $\sqrt { 1 - { ^ { 1 } } / { k } }$ Next, we will establish the Lipschitz continuity of the smoothed objective $\tilde { \mathrm { L } } _ { \Omega } ^ { \tau }$ . First, let us assume that the regularized min operation is ω-input-stable. That is:

Assumption H.1. The regularized min operation $P _ { \Omega } : \mathbb { R } ^ { k }  \Delta ^ { k }$ in Equation (25) is ω-input-stable for a τ-dependent $\omega _ { \tau } > 0$ , such that, for any $\pmb { r } , \pmb { r } ^ { \prime } \in \mathbb { R } ^ { k }$

$$
\begin{array} { r } { \| P _ { \Omega } ( \pmb { r } / \tau ) - P _ { \Omega } ( \pmb { r } ^ { \prime } / \tau ) \| _ { 1 } \le ( \omega _ { \tau } / \tau ) \| \pmb { r } - \pmb { r } ^ { \prime } \| _ { 1 } . } \end{array}\tag{32}
$$

This assumption is already satisfied for softmax with bounded inputs $r , r ^ { \prime } \ [ \mathrm { L i }$ et al., 2023, Lemma B.1]:

$$
\| P _ { \mathrm { N E } } ( \boldsymbol { r } / \tau ) - P _ { \mathrm { N E } } ( \boldsymbol { r } ^ { \prime } / \tau ) \| _ { 1 } \le \frac { \exp ( \delta / \tau ) } { k \tau } \| \boldsymbol { r } - \boldsymbol { r } ^ { \prime } \| _ { 1 } ,\tag{33}
$$

$$
\mathrm { w h e r e ~ } \delta \geq ( \operatorname* { m a x } _ { j \in [ k ] } r _ { j } ) - ( \operatorname* { m i n } _ { j \in [ k ] } r _ { j } ) \forall r \qquad \mathrm { a n d ~ } \delta \geq ( \operatorname* { m a x } _ { j \in [ k ] } r _ { j } ^ { \prime } ) - ( \operatorname* { m i n } _ { j \in [ k ] } r _ { j } ^ { \prime } ) \forall r ^ { \prime } ,\tag{34}
$$

with the corresponding $\omega _ { \tau } = \exp ( \delta / \tau ) / k$ . Thus we can show the following result:

Proposition H.2. The smoothed k-means objective $\tilde { L } _ { \Omega } ^ { \tau } : \mathbb { R } ^ { k } \to \mathbb { R }$ in Equation (24) is Lipschitz continuous, such that, with any $r , r ^ { \prime } \in [ 0 , r ^ { \star } ] ^ { k }$ , we have

$$
\left| \tilde { L } _ { \Omega } ^ { \tau } ( \boldsymbol { r } ) - \tilde { L } _ { \Omega } ^ { \tau } ( \boldsymbol { r } ^ { \prime } ) \right| = | \langle P _ { \Omega } ( \boldsymbol { r } / \tau ) , \boldsymbol { r } \rangle - \langle P _ { \Omega } ( \boldsymbol { r } ^ { \prime } / \tau ) , \boldsymbol { r } ^ { \prime } \rangle | \le \left( 1 + \frac { r ^ { \star } \omega _ { \tau } } { \tau } \right) \| \boldsymbol { r } - \boldsymbol { r } ^ { \prime } \| _ { 1 } .\tag{35}
$$

Proof. First, we have the following inequalities by simply applying the triangle inequalities:

$$
\begin{array} { r l } & { | \langle P _ { \Omega } ( { \boldsymbol r } / \tau ) , { \boldsymbol r } \rangle - \langle P _ { \Omega } ( { \boldsymbol r } ^ { \prime } / \tau ) , { \boldsymbol r } ^ { \prime } \rangle | } \\ & { \quad \le | \langle P _ { \Omega } ( { \boldsymbol r } / \tau ) , ( { \boldsymbol r } - { \boldsymbol r } ^ { \prime } ) \rangle | + | \langle ( P _ { \Omega } ( { \boldsymbol r } / \tau ) - P _ { \Omega } ( { \boldsymbol r } ^ { \prime } / \tau ) ) , { \boldsymbol r } ^ { \prime } \rangle | } \\ & { \quad \le \displaystyle \operatorname* { m a x } _ { j \in [ k ] } | P _ { \Omega } ( { \boldsymbol r } / \tau ) _ { j } | \| { \boldsymbol r } - { \boldsymbol r } ^ { \prime } \| _ { 1 } + \displaystyle \operatorname* { m a x } _ { j \in [ k ] } | r _ { j } ^ { \prime } | \| P _ { \Omega } ( { \boldsymbol r } / \tau ) - P _ { \Omega } ( { \boldsymbol r } ^ { \prime } / \tau ) ) \| _ { 1 } . } \end{array}
$$

Noting that $\begin{array} { r } { \operatorname* { m a x } _ { j \in [ k ] } P _ { \Omega } ( \pmb { r } / \tau ) _ { j } \le 1 } \end{array}$ as $P _ { \Omega } : \mathbb { R } ^ { k }  \Delta ^ { k }$ , and $r _ { j } , r _ { j } ^ { \prime } \in [ 0 , r ^ { \star } ]$ , and applying Equation (32) gives us the result in Equation (35). □

This result indicates that the Lipschitz constant of the smoothed loss usually increases as the minregularization penalty $\tau$ decreases. In general, the Lipschitz constant depends on $( 1 / \tau )$ . Furthermore, if $\omega _ { \tau }$ increases as τ decreases, then the Lipschitz constant also increases. With softmax, $\omega _ { \tau } = \exp ( \delta / \tau ) / k$ also increases as τ decreases.

## I Lipschitz-ness of the Smoothed Upperbound of the Clustering Objective

In this section, we will establish the Lipschitz constant for the smoothed upper-bound $\tilde { \mathcal { L } } _ { \Omega } ^ { \tau }$ of the original discrete non-differentiable clustering objective L. We begin with some standard assumptions:

Assumption I.1 (bounded inputs). For any clustering task $( \pmb { X } , \pmb { C } ) \sim \mathcal { D }$ with $\pmb { X } \in \mathbb { R } ^ { d \times n }$ and $C \in \mathbb { R } ^ { d \times k }$ we assume that any point and initial cluster center is bounded to the unit hypercube, that is, $\boldsymbol { X } _ { : , i } \in [ 0 , 1 ] ^ { d }$ for all $i \in [ n ]$ and $C _ { : , j } \in [ 0 , 1 ] ^ { d }$ for all $j \in [ k ]$

We explicitly enforce this boundedness in all our experiments.

Assumption I.2 (bounded model parameters). We assume that the query, key and value projection matrices $( Q _ { \mu } , K _ { \mu } , V _ { \mu } ) , \mu = 1 , . . . , 4$ have a bounded spectral norm throughout the learning process, that is,

$$
\| Q _ { \mu } \| \leq \Gamma _ { 1 } , \quad \| K _ { \mu } \| \leq \Gamma _ { 1 } , \quad \| V _ { \mu } \| \leq \Gamma _ { 2 } , \quad \mu = 1 , \ldots , 4 .\tag{36}
$$

For the ease of exposition, we assume that the query and key projection matrices share the norm bound.

Lemma I.1 (bounded model outputs). Under Assumptions I.1 and $I . { \mathcal { Q } } ,$ and the attention based updates in Definition 2.3, the initial token embeddings, the intermediate token embeddings, and the cluster centers $C ( \Theta ) \triangleq F _ { d : k } ^ { \gamma } ( X , C ; \Theta )$ output by the model $F _ { d : k } ^ { \gamma }$ with initial token embedding scheme $\left( E _ { C } , E _ { X } \right)$ for any clustering task $( X , C )$ have bounded norms as follows:

$$
\forall i \in [ n ] , \left\| \bar { \mathbf { X } } _ { : , i } ^ { ( 0 ) } \right\| \leq \sqrt { d } , \qquad \left\| \bar { \mathbf { X } } _ { : , i } ^ { ( 1 ) } \right\| \leq ( 1 + \Gamma _ { 2 } ) \sqrt { d } + \Gamma _ { 2 } \sqrt { d _ { E } } ,
$$

$$
\forall j \in [ k ] , \left\| \bar { C } _ { : , j } ^ { ( 0 ) } \right\| \leq \sqrt { d _ { E } } , \qquad \| C ( \Theta ) _ { : , j } \| \leq ( 1 + 2 \Gamma _ { 2 } + 2 \Gamma _ { 2 } ^ { 2 } ) \sqrt { d _ { E } } ,\tag{37}
$$

$$
u h e r e \ d _ { E } = \left\{ \begin{array} { l l } { d + 1 } & { i f \ u s i n g \ ( \mathrm { { 0 H } } ) } \\ { d + \lceil { \log _ { 2 } k } \rceil } & { i f \ u s i n g \ ( \mathrm { { B N } } ) } \end{array} \right. .
$$

Proof. Denoting $\bar { X } ^ { ( 0 ) }  E _ { X } ( X )$ and $\bar { C } ^ { ( 0 ) }  E _ { C } ( C )$ , we can see that $\left\| \bar { X } _ { : , i } ^ { ( 0 ) } \right\| \leq \sqrt { d }$ for any $i \in [ n ]$ given Assumption I.1 and the fact that $Y ^ { ( 0 ) } = \mathbf { 0 } _ { ( d _ { \mathrm { e m b } } - d ) \times n }$ for any initial token embedding $E _ { X }$ for the points. For the embedded initial cluster centers, for any $j \in [ k ]$ , their norms $\left\| \bar { C } _ { : , j } ^ { ( 0 ) } \right\| \le \sqrt { d + 1 }$ with the OH token embedding scheme $E _ { C } ^ { 0 \mathrm { H } }$ and $\left\| \bar { C } _ { : , j } ^ { ( 0 ) } \right\| \le \sqrt { d + \lceil \log _ { 2 } k \rceil }$ with the binary token embedding scheme $E _ { C } ^ { \mathtt { B N } }$ . Thus we can see that $\left\| \bar { C } _ { : , j } ^ { ( 0 ) } \right\| \le \sqrt { d _ { E } }$ with $d _ { E }$ defined in Equation (37).

Since the output of $C ( \Theta ) \triangleq F _ { d : k } ^ { \gamma } ( X , C ; \Theta )$ is the first d rows $\bar { C } _ { : d , : } ^ { ( 1 ) }$ of the token embeddings for the cluster centers $\bar { C } ^ { ( 1 ) }$ after a single transformer layer with updates defined as in Definition 2.3, then we have

$$
\| C ( \Theta ) _ { : , j } \| = \Big \| \bar { C } _ { : d , j } ^ { ( 1 ) } \Big \| \le \Big \| \bar { C } _ { : , j } ^ { ( 1 ) } \Big \| = \left\| \bar { C } _ { : , j } ^ { ( 0 ) } + \left( V _ { 3 } \sum _ { i = 1 } ^ { n } a _ { j i } \bar { X } _ { : , i } ^ { ( 1 ) } \right) + \left( V _ { 4 } \sum _ { j ^ { \prime } = 1 } ^ { k } b _ { j j ^ { \prime } } \bar { C } _ { : , j ^ { \prime } } ^ { ( 0 ) } \right) \right\|\tag{38}
$$

$$
\leq \left\| \bar { C } _ { : , j } ^ { ( 0 ) } \right\| + \operatorname* { m a x } _ { i \in [ n ] } \left\| V _ { 3 } \bar { X } _ { : , i } ^ { ( 1 ) } \right\| + \operatorname* { m a x } _ { j ^ { \prime } \in [ k ] } \left\| V _ { 4 } \bar { C } _ { : , j ^ { \prime } } ^ { ( 0 ) } \right\| ,\tag{39}
$$

where $a _ { j i }$ are the cross-attention weights and $b _ { j j ^ { \prime } }$ are the self-attention weights in the cluster center updates in Equation (4). The first inequality holds above since the norm of the first d dimensions is smaller than the norm of all the $d _ { \mathsf { e m b } }$ dimensions. The last inequality applies triangle inequality, and the fact that the attention weights lie on a simplex (that is, $\begin{array} { r } { a _ { j i } \in [ 0 , 1 ] , \sum _ { i } a _ { j i } = 1 } \end{array}$ and $\begin{array} { r } { b _ { j j ^ { \prime } } \in [ 0 , 1 ] , \sum _ { j ^ { \prime } } b _ { j j ^ { \prime } } = 1 ) } \end{array}$ thus corresponding to a convex sum of the value vectors, and a convex sum is upper-bounded by the maximum. Now, we need to bound the norm of the updated token embeddings $\bar { X } ^ { ( 1 ) }$ of the points:

$$
\forall i \in [ n ] , \left\| \bar { X } _ { : , i } ^ { ( 1 ) } \right\| = \left\| \bar { X } _ { : , i } ^ { ( 0 ) } + \left( V _ { 1 } \sum _ { j ^ { \prime } = 1 } ^ { k } a _ { i j ^ { \prime } } ^ { \prime } \bar { C } _ { : , j ^ { \prime } } ^ { ( 0 ) } \right) + \left( V _ { 2 } \sum _ { i ^ { \prime } = 1 } ^ { n } b _ { i i ^ { \prime } } ^ { \prime } \bar { X } _ { : , i ^ { \prime } } ^ { ( 0 ) } \right) \right\|\tag{40}
$$

$$
\leq \left\| \bar { X } _ { : , i } ^ { ( 0 ) } \right\| + \operatorname* { m a x } _ { j ^ { \prime } \in [ k ] } \left\| V _ { 1 } \bar { C } _ { : , j ^ { \prime } } ^ { ( 0 ) } \right\| + \operatorname* { m a x } _ { i ^ { \prime } \in [ n ] } \left\| V _ { 2 } \bar { X } _ { : , i ^ { \prime } } ^ { ( 0 ) } \right\|\tag{41}
$$

$$
\leq \sqrt { d } + \Gamma _ { 2 } \sqrt { d _ { E } } + \Gamma _ { 2 } \sqrt { d } ,\tag{42}
$$

where $a _ { i j ^ { \prime } } ^ { \prime }$ and $b _ { i i ^ { \prime } } ^ { \prime }$ are the cross-attention and self-attention weights for the point embedding updates in Equation (4). This gives us the bound on $\left\| \bar { X } _ { : , i } ^ { ( 1 ) } \right\|$ in Equation (37) in the statement of the result. Plugging this in Equation (39), we see

$$
\left\| C ( \Theta ) _ { : , j } \right\| \leq \left\| \bar { C } _ { : , j } ^ { ( 0 ) } \right\| + \operatorname* { m a x } _ { i \in [ n ] } \left\| V _ { 3 } \bar { X } _ { : , i } ^ { ( 1 ) } \right\| + \operatorname* { m a x } _ { j ^ { \prime } \in [ k ] } \left\| V _ { 4 } \bar { C } _ { : , j ^ { \prime } } ^ { ( 0 ) } \right\| ,\tag{43}
$$

$$
\leq \sqrt { d _ { E } } + \Gamma _ { 2 } ( \sqrt { d } + \Gamma _ { 2 } \sqrt { d _ { E } } + \Gamma _ { 2 } \sqrt { d } ) + \Gamma _ { 2 } \sqrt { d _ { E } } \leq ( 1 + 2 \Gamma _ { 2 } + 2 \Gamma _ { 2 } ^ { 2 } ) \sqrt { d _ { E } } ,\tag{44}
$$

thus giving us Equation (37) in the statement of lemma.

Lemma I.2. Under Assumptions I.1, I.2, H.1, and the conditions of Lemma I.1, for any clustering task $( \pmb { X } , \pmb { C } ) \sim \mathcal { D }$ with n points in d dimensions and k clusters, the smoothed loss $\tilde { \mathcal { L } } _ { \Omega } ^ { \tau }$ is input-stable with respect to the output of the model $F _ { d : k } ^ { \gamma }$ upon changes to the model parameters Θ as follows for any pair of parameters Θ, Θ′:

$$
\begin{array} { r l } & { \left| \tilde { \mathcal { L } } _ { \Omega } ^ { \tau } ( \boldsymbol { X } , \boldsymbol { F } _ { d : k } ^ { \gamma } ( \boldsymbol { X } , \boldsymbol { C } ; \boldsymbol { \Theta } ) ) - \tilde { \mathcal { L } } _ { \Omega } ^ { \tau } ( \boldsymbol { X } , \boldsymbol { F } _ { d : k } ^ { \gamma } ( \boldsymbol { X } , \boldsymbol { C } ; \boldsymbol { \Theta } ^ { \prime } ) ) \right| } \\ & { \quad \le \kappa _ { 1 } ( n , \Gamma _ { 2 } , d _ { E } , \omega _ { \tau } , \tau ) \| F _ { d : k } ^ { \gamma } ( \boldsymbol { X } , \boldsymbol { C } ; \boldsymbol { \Theta } ) - F _ { d : k } ^ { \gamma } ( \boldsymbol { X } , \boldsymbol { C } ; \boldsymbol { \Theta } ^ { \prime } ) \| _ { 2 , 1 } } \\ & { \quad w h e r e \kappa _ { 1 } ( n , \Gamma _ { 2 } , d _ { E } , \omega _ { \tau } , \tau ) = 4 n ( 1 + \Gamma _ { 2 } ) ^ { 2 } \sqrt { d _ { E } } \left( 1 + \frac { 4 ( 1 + \Gamma _ { 2 } ) ^ { 4 } d _ { E } \omega _ { \tau } } { \tau } \right) \sim O \left( \frac { n \omega _ { \tau } d _ { E } ^ { 3 / 2 } } { \tau } \right) . } \end{array}\tag{45}
$$

Proof. Denoting the output cluster centers as $C ( \Theta ) \triangleq F _ { d : k } ^ { \gamma } ( X , C ; \Theta )$ and $C ( \Theta ^ { \prime } ) \triangleq F _ { d : k } ^ { \gamma } ( X , C ; \Theta ^ { \prime } )$ , first, we can see the expand the clustering loss $\tilde { \mathcal { L } } _ { \Omega } ^ { \tau }$ to a per-point loss $\tilde { L } _ { \Omega } ^ { \tau }$ as follows

$$
\left| \tilde { \mathcal { L } } _ { \Omega } ^ { \tau } ( \boldsymbol { X } , \boldsymbol { C } ( \boldsymbol { \Theta } ) ) - \tilde { \mathcal { L } } _ { \Omega } ^ { \tau } ( \boldsymbol { X } , \boldsymbol { C } ( \boldsymbol { \Theta } ^ { \prime } ) ) \right| = \left| \sum _ { i = 1 } ^ { n } \tilde { L } _ { \Omega } ^ { \tau } ( \boldsymbol { X } _ { : , i } , \boldsymbol { C } ( \boldsymbol { \Theta } ) ) - \sum _ { i = 1 } ^ { n } \tilde { L } _ { \Omega } ^ { \tau } ( \boldsymbol { X } _ { : , i } , \boldsymbol { C } ( \boldsymbol { \Theta } ^ { \prime } ) ) \right|\tag{46}
$$

$$
\leq \sum _ { i = 1 } ^ { n } \left| \tilde { L } _ { \Omega } ^ { \tau } ( X _ { : , i } , C ( \boldsymbol { \Theta } ) ) - \tilde { L } _ { \Omega } ^ { \tau } ( X _ { : , i } , C ( \boldsymbol { \Theta } ^ { \prime } ) ) \right|\tag{47}
$$

$$
\left. \leq n \operatorname* { m a x } _ { i \in [ n ] } \left| \tilde { L } _ { \Omega } ^ { \tau } ( X _ { : , i } , C ( \Theta ) ) - \tilde { L } _ { \Omega } ^ { \tau } ( X _ { : , i } , C ( \Theta ^ { \prime } ) ) \right| . \right.\tag{48}
$$

We will bound the above for a generic $\pmb { x } \in [ 0 , 1 ] ^ { d }$ . Using the notation that $r = R ( { \bf x } , C ( \Theta ) )$ and $\boldsymbol { r } ^ { \prime } =$ $R ( { \pmb x } , { \pmb C } ( \Theta ^ { \prime } ) )$ , where $R : \mathbb { R } ^ { d } \times \mathbb { R } ^ { d \times k }  \mathbb { R } ^ { k }$ is as previously defined where the j-th entry of its output $r _ { j } = \Vert \pmb { x } - \pmb { C } ( \Theta ) _ { : , j } \Vert ^ { 2 }$ (and correspondingly $r _ { j } ^ { \prime } = \Vert \pmb { x } - \pmb { C } ( \Theta ^ { \prime } ) _ { : , j } \Vert ^ { 2 } )$ , we have $r , r ^ { \prime } \in [ 0 , r ^ { \star } ] ^ { k }$ with

$$
r ^ { \star } \triangleq \operatorname* { m a x } _ { \substack { x \in [ 0 , 1 ] ^ { d } , j \in [ k ] , \Theta } } \| x - C ( \Theta ) _ { : , j } \| ^ { 2 } \leq \left( \operatorname* { m a x } _ { \substack { x \in [ 0 , 1 ] ^ { d } } } \| x \| + \operatorname* { m a x } _ { j \in [ k ] , \Theta } \| C ( \Theta ) _ { : , j } \| \right) ^ { 2 }\tag{49}
$$

$$
\leq \left( \sqrt { d } + ( 1 + 2 \Gamma _ { 2 } + 2 \Gamma _ { 2 } ^ { 2 } ) \sqrt { d _ { E } } \right) ^ { 2 }\tag{50}
$$

$$
\leq 4 d _ { E } ( 1 + \Gamma _ { 2 } + \Gamma _ { 2 } ^ { 2 } ) ^ { 2 } \leq 4 ( 1 + \Gamma _ { 2 } ) ^ { 4 } d _ { E } .\tag{51}
$$

where we bounded ma $\mathbf { \boldsymbol { x } } _ { j , \Theta } \left\| \boldsymbol { C } ( \Theta ) \right\|$ using Lemma I.1. Now we can utilize Proposition H.2 as follows:

$$
\Big \lvert \tilde { L } _ { \Omega } ^ { \tau } ( \pmb { x } , C ( \Theta ) ) - \tilde { L } _ { \Omega } ^ { \tau } ( \pmb { x } , C ( \Theta ^ { \prime } ) ) \Big \rvert = \big \lvert \tilde { \mathtt { L } } _ { \Omega } ^ { \tau } ( \pmb { r } ) - \tilde { \mathtt { L } } _ { \Omega } ^ { \tau } ( \pmb { r } ^ { \prime } ) \big \rvert\tag{52}
$$

$$
\leq \left( 1 + \frac { r ^ { \star } \omega _ { \tau } } { \tau } \right) \| r - r ^ { \prime } \| _ { 1 }\tag{53}
$$

$$
= \left( 1 + \frac { 4 ( 1 + \Gamma _ { 2 } ) ^ { 4 } d _ { E } \omega _ { \tau } } { \tau } \right) \| r - r ^ { \prime } \| _ { 1 } .\tag{54}
$$

Finally, we can bound $\| \pmb { r } - \pmb { r } ^ { \prime } \| _ { 1 }$ as follows:

$$
\| \boldsymbol { r } - \boldsymbol { r } ^ { \prime } \| _ { 1 } = \sum _ { j = 1 } ^ { k } | \boldsymbol { r } _ { j } - \boldsymbol { r } _ { j } ^ { \prime } |\tag{55}
$$

$$
= \sum _ { j = 1 } ^ { k } | \langle \left( C ( \Theta ) _ { : , j } - C ( \Theta ^ { \prime } ) _ { : , j } \right) , \left( C ( \Theta ) _ { : , j } + C ( \Theta ^ { \prime } ) _ { : , j } - 2 \pmb { x } \right) \rangle |\tag{56}
$$

$$
\leq \sum _ { j = 1 } ^ { k } \| C ( \Theta ) _ { : , j } - C ( \Theta ^ { \prime } ) _ { : , j } \| \| C ( \Theta ) _ { : , j } + C ( \Theta ^ { \prime } ) _ { : , j } - 2 x \|\tag{57}
$$

$$
\leq \sum _ { j = 1 } ^ { k } \| C ( \Theta ) _ { : , j } - C ( \Theta ^ { \prime } ) _ { : , j } \| \left( 2 \sqrt { d } + 2 ( 1 + 2 \Gamma _ { 2 } + 2 \Gamma _ { 2 } ^ { 2 } ) \sqrt { d _ { E } } \right)\tag{58}
$$

$$
\leq \left( 4 ( 1 + \Gamma _ { 2 } ) ^ { 2 } \sqrt { d _ { E } } \right) \sum _ { j = 1 } ^ { k } \| C ( \Theta ) _ { : , j } - C ( \Theta ^ { \prime } ) _ { : , j } \|\tag{59}
$$

$$
= 4 ( 1 + \Gamma _ { 2 } ) ^ { 2 } \sqrt { d _ { E } } \left\| C ( \Theta ) - C ( \Theta ^ { \prime } ) \right\| _ { 2 , 1 } .\tag{60}
$$

Finally, combining the above with Equation (54) and Equation (48) gives us the inequality in Equation (45) in the statement of the result. □

The above result ties the sensitivity of the loss to the model parameters $\left| \tilde { \mathcal { L } } _ { \Omega } ^ { \tau } ( X , C ( \Theta ) ) - \tilde { \mathcal { L } } _ { \Omega } ^ { \tau } ( X , C ( \Theta ^ { \prime } ) ) \right|$ to the sensitivity of the model output to the model parameters $\lVert C ( \Theta ) - C ( \Theta ^ { \prime } ) \rVert _ { 2 , 1 }$ To bound this $\| C ( \Theta ) - C ( \Theta ^ { \prime } ) \| _ { 2 , 1 }$ with perturbations to the model parameters, we have to consider the attention based updates in Definition 2.2. Since there are four different attention operations in Equation (3) in Definition 2.2, we will first present bounds on the sensitivity of the output of a single attention module $\mathcal { A } _ { 2 } ^ { \gamma } ( \cdots )$ to changes in the specific model parameters (the query/key/value projection matrices) for generic query and key tokens $\boldsymbol { Z } , \boldsymbol { Z } ^ { \prime }$ , and then apply them to each of the four attention modules with the appropriate keys and queries These results are presented in the following, and then summarized in Table 2.

Definition I.1. For any $\pmb { v } \in \mathbb { R } ^ { m }$ , let its dispersion be defined as $D ( v )$ with a function $D : \mathbb { R } ^ { m }  \mathbb { R } _ { \geq 0 }$ such that

$$
D ( \pmb { v } ) = \bigg ( \operatorname* { m a x } _ { i \in [ m ] } v _ { i } \bigg ) - \bigg ( \operatorname* { m i n } _ { i \in [ m ] } v _ { i } \bigg ) .\tag{61}
$$

Lemma I.3 (adapted from Li et al. [2023], Lemma B.3). For any $\pmb { v } , \pmb { v } ^ { \prime } \in R ^ { m }$ such that $D ( \pmb { v } ) , D ( \pmb { v } ^ { \prime } ) \leq \delta$ $f o r$ some $\delta > 0$ , the softmax operation is bounded and input-stable as follows:

$$
\lVert \mathrm { s o f t m a x } ( \pmb { v } ) \rVert _ { \infty } \leq \frac { \exp ( \delta ) } { m } , \quad \lVert \mathrm { s o f t m a x } ( \pmb { v } ) - \mathrm { s o f t m a x } ( \pmb { v } ^ { \prime } ) \rVert _ { 1 } \leq \frac { \exp ( \delta ) } { m } \lVert \pmb { v } - \pmb { v } ^ { \prime } \rVert _ { 1 } .\tag{62}
$$

Lemma I.4 (attention score dispersion). For queries $\boldsymbol { Z } \in \mathbb { R } ^ { d _ { \mathrm { e m b } } \times m }$ and keys $Z ^ { \prime } \in \mathbb { R } ^ { d _ { \mathrm { e m b } } \times m ^ { \prime } }$ such that $\| Z _ { : , i } \| \le \rho$ for all $i \in [ m ]$ and $\left\| Z _ { : , i } ^ { \prime } \right\| \le \rho ^ { \prime }$ for all $\textit { i } \in \ [ m ^ { \prime } ]$ , under the assumption $I . { \mathcal { Q } } ,$ for $\begin{array} { r l } { \mathbf { \nabla } v _ { i } } & { { } = } \end{array}$ $- ( \gamma / \sqrt { d _ { \mathrm { e m b } } } )$ sqdist $( Q _ { \mu } Z _ { : , i } , K _ { \mu } Z ^ { \prime } )$ , we have the dispersion bounded as

$$
\forall i \in [ m ] , D ( \pmb { v } _ { i } ) \leq \delta ( \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \gamma , d _ { \sf e m b } ) = ( \gamma / \sqrt { d _ { \sf e m b } } ) \Gamma _ { 1 } ^ { 2 } ( \rho + \rho ^ { \prime } ) ^ { 2 } .\tag{63}
$$

Proof. We have the following for any $i \in [ m ]$

$$
D ( v _ { i } ) = \operatorname* { m a x } _ { i ^ { \prime } \in [ m ^ { \prime } ] } \left\{ - ( \gamma / \sqrt { d _ { \mathrm { e m b } } } ) \left\| Q _ { \mu } Z _ { ; i } - K _ { \mu } Z _ { ; i ^ { \prime } } ^ { \prime } \right\| ^ { 2 } \right\} - \operatorname* { m i n } _ { i ^ { \prime } \in [ m ^ { \prime } ] } \left\{ - ( \gamma / \sqrt { d _ { \mathrm { e m b } } } ) \left\| Q _ { \mu } Z _ { ; i } - K _ { \mu } Z _ { ; i ^ { \prime } } ^ { \prime } \right\| ^ { 2 } \right\}\tag{64}
$$

$$
= \gamma / \sqrt { d _ { \mathrm { e n b } } } \left( \operatorname* { m a x } _ { i ^ { \prime } \in [ m ^ { \prime } ] } \left\{ \big \| Q _ { \mu } { Z } _ { ; i } - K _ { \mu } { Z } _ { ; i ^ { \prime } } ^ { \prime } \big \| ^ { 2 } \right\} - \operatorname* { m i n } _ { i ^ { \prime } \in [ m ^ { \prime } ] } \left\{ \big \| Q _ { \mu } { Z } _ { ; i } - K _ { \mu } { Z } _ { ; i ^ { \prime } } ^ { \prime } \big \| ^ { 2 } \right\} \right)\tag{65}
$$

$$
\leq \gamma / \sqrt { d _ { \mathsf { e m b } } } \operatorname* { m a x } _ { i ^ { \prime } \in [ m ^ { \prime } ] } \left\{ \left. Q _ { \mu } Z _ { : , i } - K _ { \mu } Z _ { : , i ^ { \prime } } ^ { \prime } \right. ^ { 2 } \right\}\tag{66}
$$

$$
\leq \gamma / \sqrt { d _ { \mathrm { e m b } } } \left( \| Q _ { \mu } Z _ { : , i } \| + \operatorname* { m a x } _ { i ^ { \prime } \in [ m ^ { \prime } ] } \left\| K _ { \mu } Z _ { : , i ^ { \prime } } ^ { \prime } \right\| \right) ^ { 2 } \leq \gamma / \sqrt { d _ { \mathrm { e m b } } } \left( \Gamma _ { 1 } ( \rho + \rho ^ { \prime } ) \right) ^ { 2 } ,\tag{67}
$$

giving us the definition of $\delta ( \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \gamma , d _ { \mathrm { e m b } } )$ in the statement of the result.

Lemma I.5. Under conditions of Lemma $I . 4 ,$ the attention operation is stable with respect to the value projection matrix as follows:

$$
\begin{array} { r } { \left\| \mathcal { A } _ { 2 } ^ { \gamma } ( Z , Z ^ { \prime } ; ( Q _ { \mu } , K _ { \mu } , V _ { \mu } ) ) - \mathcal { A } _ { 2 } ^ { \gamma } ( Z , Z ^ { \prime } ; ( Q _ { \mu } , K _ { \mu } , V _ { \mu } ^ { \prime } ) ) \right\| _ { 2 , 1 } \leq \kappa _ { 2 } ( m , \rho ^ { \prime } ) \left\| V _ { \mu } - V _ { \mu } ^ { \prime } \right\| , } \end{array}\tag{68}
$$

with $\kappa _ { 2 } ( m , \rho ^ { \prime } ) = m \rho ^ { \prime }$

Proof. Expanding out the attention operation, we can see that

$$
\begin{array} { r } { \left\| \mathcal { A } _ { 2 } ^ { \gamma } ( Z , Z ^ { \prime } ; ( Q _ { \mu } , K _ { \mu } , V _ { \mu } ) ) - \mathcal { A } _ { 2 } ^ { \gamma } ( Z , Z ^ { \prime } ; ( Q _ { \mu } , K _ { \mu } , V _ { \mu } ^ { \prime } ) ) \right\| _ { 2 , 1 } } \end{array}
$$

$$
= \sum _ { i = 1 } ^ { m } \left\| \sum _ { j = 1 } ^ { m ^ { \prime } } a _ { i j } V _ { \mu } \boldsymbol { Z } _ { ; , j } ^ { \prime } - \sum _ { j = 1 } ^ { m ^ { \prime } } a _ { i j } V _ { \mu } ^ { \prime } \boldsymbol { Z } _ { ; , j } ^ { \prime } \right\| = \sum _ { i = 1 } ^ { m } \left\| ( V _ { \mu } - V _ { \mu } ^ { \prime } ) \sum _ { j = 1 } ^ { m ^ { \prime } } a _ { i j } \boldsymbol { Z } _ { ; , j } ^ { \prime } \right\|\tag{69}
$$

$$
\leq \sum _ { i = 1 } ^ { m } \left\| ( V _ { \mu } - V _ { \mu } ^ { \prime } ) \right\| \left\| \sum _ { j = 1 } ^ { m ^ { \prime } } a _ { i j } Z _ { : , j } ^ { \prime } \right\| \leq \sum _ { i = 1 } ^ { m } \left\| V _ { \mu } - V _ { \mu } ^ { \prime } \right\| \operatorname* { m a x } _ { j \in [ m ^ { \prime } ] } \left\| Z _ { : , j } ^ { \prime } \right\| \leq m \rho ^ { \prime } \left\| V _ { \mu } - V _ { \mu } ^ { \prime } \right\| ,\tag{70}
$$

where $a _ { i j }$ are the softmax attention weights for the i-th query computed by softmax $\cdot ( - \gamma / \sqrt { d _ { \mathrm { e m b } } } \mathrm { s q d i s t } ( Q _ { \mu } Z _ { \cdot , i } , K _ { \mu } Z ^ { \prime } ) )$ thus giving us $\kappa _ { 2 } ( m , \rho ^ { \prime } )$ in the statement of the result. □

Lemma I.6. Under conditions of Lemma $I . 4 ,$ the attention operation is stable with respect to the query projection matrix as follows:

$$
\begin{array} { r } { \left\| \mathcal { A } _ { 2 } ^ { \gamma } ( Z , Z ^ { \prime } ; ( Q _ { \mu } , K _ { \mu } , V _ { \mu } ) ) - \mathcal { A } _ { 2 } ^ { \gamma } ( Z , Z ^ { \prime } ; ( Q _ { \mu } ^ { \prime } , K _ { \mu } , V _ { \mu } ) ) \right\| _ { 2 , 1 } } \end{array}
$$

$$
\leq \kappa _ { 3 } ( m , \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \left| \left| Q _ { \mu } - Q _ { \mu } ^ { \prime } \right| \right| ,\tag{71}
$$

$$
\begin{array} { r l } { u h e r e } & { \kappa _ { 3 } ( m , \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) = \frac { 2 m \gamma \Gamma _ { 1 } \Gamma _ { 2 } \rho \rho ^ { \prime } ( \rho + \rho ^ { \prime } ) } { \sqrt { d _ { \mathrm { e m b } } } } \exp ( \delta ( \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \gamma , d _ { \mathrm { e m b } } ) ) . } \end{array}
$$

Proof. First, focusing on just the i-th query corresponding to $Z _ { : , i }$ for some $i \in [ m ]$ , let ${ \pmb v } = - ( \gamma / \sqrt { d _ { \mathrm { e m b } } } ) \mathrm { s q d i s t } ( { \pmb Q } _ { \mu } { \pmb Z } _ { : , i } , { \pmb K } _ { \mu } { \pmb Z } ^ { \prime } )$ and ${ \pmb v } ^ { \prime } = - ( \gamma / \sqrt { d _ { \mathrm { e m b } } } ) \mathrm { s q d i s t } ( { \pmb Q } _ { \mu } ^ { \prime } { \pmb Z } _ { : , i } , { \pmb K } _ { \mu } { \pmb Z } ^ { \prime } )$ , Lemma I.4 tells us that their dispersions are bounded as $D ( \pmb { v } ) , D ( \pmb { v } ^ { \prime } ) \leq$ $\delta ( \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \gamma , d _ { \mathsf { e m b } } )$ . Now the $\| \pmb { v } - \pmb { v } ^ { \prime } \| _ { 1 }$ can be bounded as

$$
\| \pmb { v } - \pmb { v } ^ { \prime } \| _ { 1 } = \sum _ { j = 1 } ^ { m ^ { \prime } } | v _ { j } - v _ { j } ^ { \prime } | = \frac { \gamma } { \sqrt { d _ { \mathrm { e m b } } } } \sum _ { j = 1 } ^ { m ^ { \prime } } \left| \left\| Q _ { \mu } Z _ { ; i } - K _ { \mu } Z _ { ; j } ^ { \prime } \right\| ^ { 2 } - \left\| Q _ { \mu } ^ { \prime } Z _ { ; i } - K _ { \mu } Z _ { ; j } ^ { \prime } \right\| ^ { 2 } \right|\tag{72}
$$

$$
\leq \frac { \gamma } { \sqrt { d _ { \mathrm { e m b } } } } \sum _ { j = 1 } ^ { m ^ { \prime } } \left. ( Q _ { \mu } - Q _ { \mu } ^ { \prime } ) Z _ { : , i } \right. \left. ( Q _ { \mu } + Q _ { \mu } ^ { \prime } ) Z _ { : , i } - 2 K _ { \mu } Z _ { : , j } ^ { \prime } \right.\tag{73}
$$

$$
\leq \frac { \gamma } { \sqrt { d _ { \mathrm { e m b } } } } m ^ { \prime } \left. ( Q _ { \mu } - Q _ { \mu } ^ { \prime } ) \right. \left. Z _ { : , i } \right. \left( \left. Q _ { \mu } Z _ { : , i } \right. + \left. Q _ { \mu } ^ { \prime } Z _ { : , i } \right. + 2 \left. K _ { \mu } Z _ { : , j } ^ { \prime } \right. \right)\tag{74}
$$

$$
\leq { \frac { m ^ { \prime } \gamma } { \sqrt { d _ { \mathsf { e m b } } } } } \left\| ( Q _ { \mu } - Q _ { \mu } ^ { \prime } ) \right\| \rho ( 2 \Gamma _ { 1 } ( \rho + \rho ^ { \prime } ) ) .\tag{75}
$$

Combining this with Lemma I.3, we can see that

$$
\left\| \operatorname { s o f t m a x } ( \pmb { v } ) - \operatorname { s o f t m a x } ( \pmb { v } ^ { \prime } ) \right\| _ { 1 } \leq \frac { \exp ( \delta ( \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \gamma , d _ { \mathsf { e m b } } ) ) } { m ^ { \prime } } \left\| \pmb { v } - \pmb { v } ^ { \prime } \right\| _ { 1 }\tag{76}
$$

$$
\leq \frac { \exp ( \delta ( \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \gamma , d _ { \mathrm { e m b } } ) ) } { m ^ { \prime } } \frac { m ^ { \prime } \gamma } { \sqrt { d _ { \mathrm { e m b } } } } \rho \left( 2 \Gamma _ { 1 } ( \rho + \rho ^ { \prime } ) \right) \left. \left. Q _ { \mu } - Q _ { \mu } ^ { \prime } \right. \right.\tag{77}
$$

$$
= \frac { 2 \gamma \Gamma _ { 1 } \rho ( \rho + \rho ^ { \prime } ) \exp ( \delta ( \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \gamma , d _ { \mathrm { e m b } } ) ) } { \sqrt { d _ { \mathrm { e m b } } } } \left\| Q _ { \mu } - Q _ { \mu } ^ { \prime } \right\| .\tag{78}
$$

Denoting ${ \cal A } = \mathrm { s o f t m a x } ( - ( \gamma / \sqrt { d _ { \mathrm { e m b } } } ) \mathrm { s q d i s t } ( Q _ { \mu } Z , K _ { \mu } Z ^ { \prime } ) )$ and ${ \cal A } ^ { \prime } = \mathrm { s o f t m a x } ( - ( \gamma / \sqrt { d _ { \mathrm { e m b } } } ) \mathrm { s q d i s t } ( Q _ { \mu } ^ { \prime } Z , K _ { \mu } Z ^ { \prime } ) )$ with the softmax applied columnwise, we have $\pmb { A } , \pmb { A } ^ { \prime } \in \mathbb { R } ^ { m ^ { \prime } \times m }$ . Now expanding the left-hand-side of Equation (71), and we see that

$$
\begin{array} { r } { \left\| \mathcal { A } _ { 2 } ^ { \gamma } ( Z , Z ^ { \prime } ; ( Q _ { \mu } , K _ { \mu } , V _ { \mu } ) ) - \mathcal { A } _ { 2 } ^ { \gamma } ( Z , Z ^ { \prime } ; ( Q _ { \mu } ^ { \prime } , K _ { \mu } , V _ { \mu } ) ) \right\| _ { 2 , 1 } } \end{array}
$$

$$
= \sum _ { i = 1 } ^ { m } \left\| { \cal V } _ { \mu } \sum _ { j = 1 } ^ { m ^ { \prime } } ( A _ { j i } - A _ { j i } ^ { \prime } ) Z _ { : , j } ^ { \prime } \right\|\tag{79}
$$

$$
\leq \| \boldsymbol { V } _ { \boldsymbol { \mu } } \| \sum _ { i = 1 } ^ { m } \left\| \sum _ { j = 1 } ^ { m ^ { \prime } } ( A _ { j i } - A _ { j i } ^ { \prime } ) \boldsymbol { Z } _ { : , j } ^ { \prime } \right\| \leq \Gamma _ { 2 } \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { m ^ { \prime } } | A _ { j i } - A _ { j i } ^ { \prime } | \left\| \boldsymbol { Z } _ { : , j } ^ { \prime } \right\|\tag{80}
$$

$$
\leq \Gamma _ { 2 } \rho ^ { \prime } \sum _ { i = 1 } ^ { m } \left. \mathbf { A } _ { : , i } - \mathbf { A } _ { : , i } ^ { \prime } \right. _ { 1 }\tag{81}
$$

$$
\leq \Gamma _ { 2 } \rho ^ { \prime } m \frac { 2 \gamma \Gamma _ { 1 } \rho ( \rho + \rho ^ { \prime } ) \exp ( \delta ( \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \gamma , d _ { \mathrm { e m b } } ) ) } { \sqrt { d _ { \mathrm { e m b } } } } \left\| Q _ { \mu } - Q _ { \mu } ^ { \prime } \right\| ,\tag{82}
$$

where we invoked Equation $( 7 8 )$ , noting that $\pmb { A } _ { : , i } = \mathrm { s o f t m a x } ( \pmb { v } )$ and ${ A ^ { \prime } } _ { : , i } = \operatorname { s o f t m a x } ( { v ^ { \prime } } )$ and summed over $i \in [ m ]$ . This gives us the coefficient $\kappa _ { 3 } ( m , \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } )$ in Equation (71) in the statement of the result. □

Lemma I.7. Under conditions of Lemma $I . 4 ,$ the attention operation is stable with respect to the key projection matrix as follows:

$$
\begin{array} { r l } & { \big \| \mathcal { A } _ { 2 } ^ { \gamma } ( Z , Z ^ { \prime } ; ( Q _ { \mu } , K _ { \mu } , V _ { \mu } ) ) - \mathcal { A } _ { 2 } ^ { \gamma } ( Z , Z ^ { \prime } ; ( Q _ { \mu } , K _ { \mu } ^ { \prime } , V _ { \mu } ) ) \big \| _ { 2 , 1 } } \\ & { \quad \leq \kappa _ { 4 } ( m , \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \big \| K _ { \mu } - K _ { \mu } ^ { \prime } \big \| , } \\ & { \quad w h e r e \quad \kappa _ { 4 } ( m , \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) = \frac { 2 m \gamma \Gamma _ { 1 } \Gamma _ { 2 } \rho ^ { \prime 2 } ( \rho + \rho ^ { \prime } ) } { \sqrt { d _ { \mathrm { e m b } } } } \exp ( \delta ( \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \gamma , d _ { \mathrm { e m b } } ) ) . } \end{array}\tag{83}
$$

Proof. First, focusing on just the i-th query corresponding to $Z _ { : , i }$ for some $i \in [ m ]$ , let ${ \pmb v } = - ( \gamma / \sqrt { d _ { \mathrm { e m b } } } ) \mathrm { s q d i s t } ( { \pmb Q } _ { \mu } { \pmb Z } _ { : , i } , { \pmb K } _ { \mu } { \pmb Z } ^ { \prime } )$ and ${ \pmb v } ^ { \prime } = - ( \gamma / \sqrt { d _ { \mathrm { e m b } } } ) \mathrm { s q d i s t } ( { \pmb Q } _ { \mu } { \pmb Z } _ { : , i } , K _ { \mu } ^ { \prime } { \pmb Z } ^ { \prime } )$ , Lemma I.4 tells us that their dispersions are bounded as $D ( \pmb { v } ) , D ( \pmb { v } ^ { \prime } ) \leq$ $\delta ( \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \gamma , d _ { \mathsf { e m b } } )$ . Now the $\| \pmb { v } - \pmb { v } ^ { \prime } \| _ { 1 }$ can be bounded as

$$
\displaystyle | | \boldsymbol { v } - \boldsymbol { v ^ { \prime } } | | | _ { 1 } = \sum _ { j = 1 } ^ { m ^ { \prime } } | v _ { j } - v _ { j } ^ { \prime } | = \frac { \gamma } { \sqrt { d _ { \mathrm { e m b } } } } \sum _ { j = 1 } ^ { m ^ { \prime } } \left| \left| \left\| Q _ { \mu } Z _ { ; i } - K _ { \mu } Z _ { ; j } ^ { \prime } \right| \right| ^ { 2 } - \left| \left| Q _ { \mu } Z _ { ; i } - K _ { \mu } ^ { \prime } Z _ { ; j } ^ { \prime } \right| \right| ^ { 2 } \right|\tag{84}
$$

$$
\leq \frac { \gamma } { \sqrt { d _ { \mathrm { e m b } } } } \sum _ { j = 1 } ^ { m ^ { \prime } } \left. ( K _ { \mu } - K _ { \mu } ^ { \prime } ) Z _ { : , j } ^ { \prime } \right. \left. ( K _ { \mu } + K _ { \mu } ^ { \prime } ) Z _ { : , j } ^ { \prime } - 2 Q _ { \mu } Z _ { : , i } \right.\tag{85}
$$

$$
\begin{array} { r } { \leq \frac { \gamma } { \sqrt { d _ { \mathrm { e m b } } } } m ^ { \prime } \left\| ( K _ { \mu } - K _ { \mu } ^ { \prime } ) \right\| \left\| Z _ { : , j } ^ { \prime } \right\| \left( \left\| K _ { \mu } Z _ { : , j } ^ { \prime } \right\| + \left\| K _ { \mu } ^ { \prime } Z _ { : , j } ^ { \prime } \right\| + 2 \left\| Q _ { \mu } Z _ { : , i } \right\| \right) } \end{array}\tag{86}
$$

$$
\leq \frac { m ^ { \prime } \gamma } { \sqrt { d _ { \mathrm { e m b } } } } \left. ( K _ { \mu } - K _ { \mu } ^ { \prime } ) \right. \rho ^ { \prime } \left( 2 \Gamma _ { 1 } ( \rho ^ { \prime } + \rho ) \right) .\tag{87}
$$

Combining this with Lemma I.3, we can see that

$$
\Vert \operatorname { s o f t m a x } ( v ) - \operatorname { s o f t m a x } ( v ^ { \prime } ) \Vert _ { 1 } \leq \frac { \exp ( \delta ( \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \gamma , d _ { \operatorname { e m b } } ) ) } { m ^ { \prime } } \Vert v - v ^ { \prime } \Vert _ { 1 }\tag{88}
$$

$$
\leq \frac { \exp ( \delta ( \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \gamma , d _ { \mathrm { e m b } } ) ) } { m ^ { \prime } } \frac { m ^ { \prime } \gamma } { \sqrt { d _ { \mathrm { e m b } } } } \rho ^ { \prime } \left( 2 \Gamma _ { 1 } ( \rho + \rho ^ { \prime } ) \right) \left. \left. K _ { \mu } - K _ { \mu } ^ { \prime } \right. \right.\tag{89}
$$

$$
= \frac { 2 \gamma \Gamma _ { 1 } \rho ^ { \prime } ( \rho + \rho ^ { \prime } ) \exp ( \delta ( \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \gamma , d _ { \mathrm { e m b } } ) ) } { \sqrt { d _ { \mathrm { e m b } } } } \left\| K _ { \mu } - K _ { \mu } ^ { \prime } \right\| .\tag{90}
$$

Denoting ${ \cal A } = \mathrm { s o f t m a x } ( - ( \gamma / \sqrt { d _ { \mathrm { e m b } } } ) \mathrm { s q d i s t } ( Q _ { \mu } Z , K _ { \mu } Z ^ { \prime } ) )$ and ${ \cal A } ^ { \prime } = \mathrm { s o f t m a x } ( - ( \gamma / \sqrt { d _ { \mathrm { e m b } } } ) \mathrm { s q d i s t } ( Q _ { \mu } Z , K _ { \mu } ^ { \prime } Z ^ { \prime } ) )$ with the softmax applied columnwise, we have $\pmb { A } , \pmb { A } ^ { \prime } \in \mathbb { R } ^ { m ^ { \prime } \times m }$ . Now expanding the left-hand-side of Equation (83), and we see that

$$
\begin{array} { r } { \left\| \mathcal { A } _ { 2 } ^ { \gamma } ( Z , Z ^ { \prime } ; ( Q _ { \mu } , K _ { \mu } , V _ { \mu } ) ) - \mathcal { A } _ { 2 } ^ { \gamma } ( Z , Z ^ { \prime } ; ( Q _ { \mu } , K _ { \mu } ^ { \prime } , V _ { \mu } ) ) \right\| _ { 2 , 1 } } \end{array}
$$

$$
= \sum _ { i = 1 } ^ { m } \left\| { \cal V } _ { \mu } \sum _ { j = 1 } ^ { m ^ { \prime } } ( A _ { j i } - A _ { j i } ^ { \prime } ) Z _ { : , j } ^ { \prime } \right\|\tag{91}
$$

$$
\leq \| \boldsymbol { V } _ { \mu } \| \sum _ { i = 1 } ^ { m } \left\| \sum _ { j = 1 } ^ { m ^ { \prime } } ( A _ { j i } - A _ { j i } ^ { \prime } ) \boldsymbol { Z } _ { : , j } ^ { \prime } \right\| \leq \Gamma _ { 2 } \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { m ^ { \prime } } | A _ { j i } - A _ { j i } ^ { \prime } | \left\| \boldsymbol { Z } _ { : , j } ^ { \prime } \right\|\tag{92}
$$

$$
\leq \Gamma _ { 2 } \rho ^ { \prime } \sum _ { i = 1 } ^ { m } \left. \mathbf { A } _ { : , i } - { A } _ { : , i } ^ { \prime } \right. _ { 1 }\tag{93}
$$

$$
\leq \Gamma _ { 2 } \rho ^ { \prime } m \frac { 2 \gamma \Gamma _ { 1 } \rho ^ { \prime } ( \rho + \rho ^ { \prime } ) \exp ( \delta ( \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \gamma , d _ { \mathrm { e m b } } ) ) } { \sqrt { d _ { \mathrm { e m b } } } } \left\| K _ { \mu } - K _ { \mu } ^ { \prime } \right\| ,\tag{94}
$$

where we invoked Equation (90), noting that $\pmb { A } _ { : , i } = \mathrm { s o f t m a x } ( \pmb { v } )$ and ${ A ^ { \prime } } _ { : , i } = \operatorname { s o f t m a x } ( { v ^ { \prime } } )$ and summed over $i \in [ m ]$ . This gives us the coefficient $\kappa _ { 4 } ( m , \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } )$ in Equation (83) in the statement of the result. □

Lemma I.8. Under conditions of Lemma $I . 4 ,$ the attention operation is stable with respect to the keys as follows:

$$
\begin{array} { r l } & { \forall Z ^ { \prime } , Z ^ { \prime \prime } \in \mathbb { R } ^ { d _ { \mathsf { e n b } } \times m ^ { \prime } } , \forall j \in [ m ^ { \prime } ] , \| Z _ { : , j } ^ { \prime } \| \leq \rho ^ { \prime } , \| Z _ { : , j } ^ { \prime \prime } \| \leq \rho ^ { \prime } , } \\ & { \| A _ { 2 } ^ { \gamma } ( Z , Z ^ { \prime } ; ( Q _ { \mu } , K _ { \mu } , V _ { \mu } ) ) - A _ { 2 } ^ { \gamma } ( Z , Z ^ { \prime \prime } ; ( Q _ { \mu } , K _ { \mu } , V _ { \mu } ) ) \| _ { 2 , 1 } } \\ & { \quad \leq \kappa _ { 5 } ( m , m ^ { \prime } , \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathsf { e m b } } ) \| Z ^ { \prime } - Z ^ { \prime \prime } \| _ { 2 , 1 } , } \\ & { \quad \quad w h e r e \quad \kappa _ { 5 } ( m , m ^ { \prime } , \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathsf { e m b } } ) } \\ & { \quad \quad = \bigg ( 1 + 2 \bigg ( \frac { \gamma } { \sqrt { d _ { \mathsf { e m b } } } } \bigg ) \Gamma _ { 1 } ^ { 2 } \rho ^ { \prime } ( \rho + \rho ^ { \prime } ) \bigg ) \bigg ( \frac { m \Gamma _ { 2 } } { m ^ { \prime } } \bigg ) \exp ( \delta ( \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \gamma , d _ { \mathsf { e m b } } ) ) . } \end{array}\tag{95}
$$

Proof. We can denote with $A = \mathrm { s o f t m a x } ( - ( \gamma / \sqrt { d _ { \mathsf { e m b } } } )$ sqdist $( Q _ { \mu } Z , K _ { \mu } Z ^ { \prime } ) \rangle$ using the keys $\pmb { Z } ^ { \prime } ,$ and the corresponding $A ^ { \prime } = \mathrm { s o f t m a x } ( - ( \gamma / \sqrt { d _ { \mathrm { e m b } } } ) \mathrm { s q d i s t } ( Q _ { \mu } Z , K _ { \mu } Z ^ { \prime \prime } ) )$ with the keys $\pmb { Z } ^ { \prime \prime }$ (the softmax applied columnwise) we have $\pmb { A } , \pmb { A } ^ { \prime } \in \mathbb { R } ^ { m ^ { \prime } \times m }$ . Now expanding the left-hand-side of Equation (95), and we see that

$$
\begin{array} { r l } & { \| \mathcal { A } _ { 2 } ^ { \gamma } ( Z , Z ^ { \prime } ; ( Q _ { \mu } , K _ { \mu } , V _ { \mu } ) ) - \mathcal { A } _ { 2 } ^ { \gamma } ( Z , Z ^ { \prime \prime } ; ( Q _ { \mu } , K _ { \mu } , V _ { \mu } ) ) \| _ { 2 , 1 } } \\ & { \quad = \| V _ { \mu } Z ^ { \prime } A - V _ { \mu } Z ^ { \prime \prime } A ^ { \prime } \| _ { 2 , 1 } \leq \| V _ { \mu } ( Z ^ { \prime } - Z ^ { \prime \prime } ) A \| _ { 2 , 1 } + \| V _ { \mu } Z ^ { \prime \prime } ( A - A ^ { \prime } ) \| _ { 2 , 1 } . } \end{array}\tag{96}
$$

We can bound each of these terms separately. First, noting that $A _ { j i } \leq \exp ( \delta ( \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \gamma , d _ { \mathsf { e m b } } ) ) / m ^ { \prime }$ for any $( i , j )$ from Lemma I.3, we have

$$
\| V _ { \mu } ( Z ^ { \prime } - Z ^ { \prime \prime } ) A \| _ { 2 , 1 } = \sum _ { i = 1 } ^ { m } \| V _ { \mu } ( Z ^ { \prime } - Z ^ { \prime \prime } ) A _ { : , i } \| \leq \| V _ { \mu } \| \sum _ { i = 1 } ^ { m } \| ( Z ^ { \prime } - Z ^ { \prime \prime } ) A _ { : , i } \|\tag{97}
$$

$$
= \| \boldsymbol { V } _ { \mu } \| \sum _ { i = 1 } ^ { m } \left\| \sum _ { j = 1 } ^ { m ^ { \prime } } ( Z _ { : , j } ^ { \prime } - Z _ { : , j } ^ { \prime \prime } ) A _ { j i } \right\| \leq \| \boldsymbol { V } _ { \mu } \| \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { m ^ { \prime } } \left\| Z _ { : , j } ^ { \prime } - Z _ { : , j } ^ { \prime \prime } \right\| | A _ { j i } |\tag{98}
$$

$$
\leq \Gamma _ { 2 } \frac { \exp ( \delta ( \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \gamma , d _ { \mathsf { e m b } } ) ) } { m ^ { \prime } } \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { m ^ { \prime } } \| Z _ { : , j } ^ { \prime } - Z _ { : , j } ^ { \prime \prime } \|\tag{99}
$$

$$
= \Gamma _ { 2 } { \frac { \exp ( \delta ( \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \gamma , d _ { \sf e m b } ) ) } { m ^ { \prime } } } m \| Z ^ { \prime } - Z ^ { \prime \prime } \| _ { 2 , 1 } .\tag{100}
$$

Next, focusing on just the i-th query corresponding to $Z _ { : , i }$ for some $i \in [ m ] ,$ let $\pmb { v } = - ( \gamma / \sqrt { d _ { \mathrm { e m b } } } )$ sqdist $( Q _ { \mu } Z _ { : , i } , K _ { \mu } Z ^ { \prime } )$ and $v ^ { \prime } ~ = ~ - ( \gamma / \sqrt { d _ { \mathrm { e m b } } } ) \mathrm { s q d i s t } ( Q _ { \mu } Z _ { : , i } , K _ { \mu } Z ^ { \prime \prime } )$ , Lemma I.4 tells us that their dispersions are bounded as $D ( \pmb { v } ) , D ( \pmb { v } ^ { \prime } ) \leq \delta ( \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \gamma , d _ { \mathsf { e m b } } )$ . Note that $A _ { : , i } = \mathrm { s o f t m a x } ( \pmb { v } ) , A _ { : , i } ^ { \prime } = \mathrm { s o f t m a x } ( \pmb { v } ^ { \prime } )$ . Now the $\| v - v ^ { \prime } \|$ 1 can be bounded as

$$
\| \pmb { v } - \pmb { v } ^ { \prime } \| _ { 1 } = \sum _ { j = 1 } ^ { m ^ { \prime } } | v _ { j } - v _ { j } ^ { \prime } | = \frac { \gamma } { \sqrt { d _ { \mathrm { e m b } } } } \sum _ { j = 1 } ^ { m ^ { \prime } } \left| \left\| Q _ { \mu } Z _ { ; i } - K _ { \mu } Z _ { ; j } ^ { \prime } \right\| ^ { 2 } - \left\| Q _ { \mu } Z _ { ; i } - K _ { \mu } Z _ { ; j } ^ { \prime \prime } \right\| ^ { 2 } \right|\tag{101}
$$

$$
\leq \frac { \gamma } { \sqrt { d _ { \mathrm { e m b } } } } \sum _ { j = 1 } ^ { m ^ { \prime } } \Vert \boldsymbol { K } _ { \mu } ( Z _ { : , j } ^ { \prime } - Z _ { : , j } ^ { \prime \prime } ) \Vert \left. \boldsymbol { K } _ { \mu } ( Z _ { : , j } ^ { \prime } + Z _ { : , j } ^ { \prime \prime } ) - 2 \boldsymbol { Q } _ { \mu } \boldsymbol { Z } _ { : , i } \right.\tag{102}
$$

$$
\leq \frac { \gamma } { \sqrt { d _ { \mathrm { e m b } } } } \sum _ { j = 1 } ^ { m ^ { \prime } } \Vert \boldsymbol { K } _ { \mu } \Vert \left. Z _ { : , j } ^ { \prime } - Z _ { : , j } ^ { \prime \prime } \right. \left( \left. \boldsymbol { K } _ { \mu } Z _ { : , j } ^ { \prime } \right. + \left. \boldsymbol { K } _ { \mu } Z _ { : , j } ^ { \prime \prime } \right. + 2 \left. \boldsymbol { Q } _ { \mu } Z _ { : , i } \right. \right)\tag{103}
$$

$$
\leq \frac { \gamma } { \sqrt { d _ { \mathrm { e m b } } } } \sum _ { j = 1 } ^ { m ^ { \prime } } \Gamma _ { 1 } \left\| Z _ { : , j } ^ { \prime } - Z _ { : , j } ^ { \prime \prime } \right\| \left( 2 \Gamma _ { 1 } ( \rho ^ { \prime } + \rho ) \right)\tag{104}
$$

$$
= \frac { \gamma } { \sqrt { d _ { \mathrm { e m b } } } } \Gamma _ { 1 } \left( 2 \Gamma _ { 1 } ( \rho ^ { \prime } + \rho ) \right) \sum _ { j = 1 } ^ { m ^ { \prime } } \left. Z _ { : , j } ^ { \prime } - Z _ { : , j } ^ { \prime \prime } \right.\tag{105}
$$

$$
= \frac { \gamma } { \sqrt { d _ { \mathsf { e m b } } } } \Gamma _ { 1 } \left( 2 \Gamma _ { 1 } ( \rho ^ { \prime } + \rho ) \right) \| Z ^ { \prime } - Z ^ { \prime \prime } \| _ { 2 , 1 } .\tag{106}
$$

Combining this with Lemma I.3, we can see that

$$
\begin{array} { r } { \| \mathrm { s o f t m a x } ( v ) - \mathrm { s o f t m a x } ( v ^ { \prime } ) \| _ { 1 } \leq \frac { \exp ( \delta ( \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \gamma , d _ { \mathrm { e m b } } ) ) } { m ^ { \prime } } \| v - v ^ { \prime } \| _ { 1 } } \end{array}\tag{107}
$$

$$
\leq \frac { \exp ( \delta ( \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \gamma , d _ { \mathrm { e m b } } ) ) } { m ^ { \prime } } \frac { \gamma } { \sqrt { d _ { \mathrm { e m b } } } } 2 \Gamma _ { 1 } ^ { 2 } ( \rho ^ { \prime } + \rho ) \left. Z ^ { \prime } - Z ^ { \prime \prime } \right. _ { 2 , 1 }\tag{108}
$$

Considering the second term on the right-hand-side of Equation (96), we see that

$$
\| V _ { \mu } Z ^ { \prime \prime } ( A - A ^ { \prime } ) \| _ { 2 , 1 } = \sum _ { i = 1 } ^ { m } \left\| V _ { \mu } Z ^ { \prime \prime } ( A _ { : , i } - A _ { : , i } ^ { \prime } ) \right\| = \sum _ { i = 1 } ^ { m } \left\| \sum _ { j = 1 } ^ { m ^ { \prime } } V _ { \mu } Z _ { : , j } ^ { \prime \prime } ( A _ { j i } - A _ { j i } ^ { \prime } ) \right\|\tag{109}
$$

$$
\leq \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { m ^ { \prime } } \left\| V _ { \mu } \pmb { Z } _ { : , j } ^ { \prime \prime } \right\| \left\| A _ { j i } - A _ { j i } ^ { \prime } \right\| \leq \sum _ { i = 1 } ^ { m } \Gamma _ { 2 } \rho ^ { \prime } \sum _ { j = 1 } ^ { m ^ { \prime } } \left| A _ { j i } - A _ { j i } ^ { \prime } \right|\tag{110}
$$

$$
\mathbf { \tau } = \Gamma _ { 2 } \rho ^ { \prime } \sum _ { i = 1 } ^ { m } \left. \mathbf { A } _ { : , i } - \mathbf { A } _ { : , i } ^ { \prime } \right. _ { 1 }\tag{111}
$$

$$
\leq \Gamma _ { 2 } \rho ^ { \prime } \sum _ { i = 1 } ^ { m } \frac { \exp ( \delta ( \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \gamma , d _ { \mathrm { e m b } } ) ) } { m ^ { \prime } } \frac { \gamma } { \sqrt { d _ { \mathrm { e m b } } } } 2 \Gamma _ { 1 } ^ { 2 } ( \rho ^ { \prime } + \rho ) \left. Z ^ { \prime } - Z ^ { \prime \prime } \right. _ { 2 , 1 } ,\tag{112}
$$

where we applied Equation (108) noting that $\pmb { A } _ { : , i } = \mathrm { s o f t m a x } ( \pmb { v } )$ and ${ A _ { : , i } ^ { \prime } } = \operatorname { s o f t m a x } ( { v ^ { \prime } } )$ . Now combining Equation (100) and Equation (112) in Equation (96), we have

$$
\begin{array} { r } { \| \mathcal { A } _ { 2 } ^ { \gamma } ( Z , Z ^ { \prime } ; ( Q _ { \mu } , K _ { \mu } , V _ { \mu } ) ) - \mathcal { A } _ { 2 } ^ { \gamma } ( Z , Z ^ { \prime \prime } ; ( Q _ { \mu } , K _ { \mu } , V _ { \mu } ) ) \| _ { 2 , 1 } } \end{array}
$$

$$
\begin{array} { r } { \leq \Vert Z ^ { \prime } - Z ^ { \prime \prime } \Vert _ { 2 , 1 } \frac { m \Gamma _ { 2 } } { m ^ { \prime } } \exp ( \delta ( \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \gamma , d _ { \mathrm { e m b } } ) ) \left( 1 + 2 \frac { \gamma } { \sqrt { d _ { \mathrm { e m b } } } } \Gamma _ { 1 } ^ { 2 } \rho ^ { \prime } ( \rho ^ { \prime } + \rho ) \right) } \end{array}\tag{113}
$$

$$
\begin{array} { r l } { \allowdisplaybreaks } & { { } = \Vert Z ^ { \prime } - Z ^ { \prime \prime } \Vert _ { 2 , 1 } \kappa _ { 5 } ( m , m ^ { \prime } , \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathsf { e m b } } ) , } \end{array}\tag{114}
$$

as in Equation (95) in the statement of the result.

We summarize the above results — the sensitivity of the attention module output to perturbations to its parameters (value projection matrix in Lemma I.5, query projection matrix in Lemma I.6, key projection matrix in Lemma I.7) and its input (keys in Lemma I.8) — in the following Table 2 for convenience, noting the factors that affect the bounds in the first sub-table (such as the number of queries or key, their respective norm upper-bounds, parameter matrix upper-bounds and such), and how the factors affect the bounds in the second and third sub-tables (in terms of whether they worsen or improve with an increase in the factor).

We can now apply these results to the attention modules in Definition 2.2 to bound the stability of the model output $\| C ( \Theta ) - C ( \Theta ^ { \prime } ) \| _ { 2 , 1 }$ . In Table $^ { 3 , }$ we note the relevant quantities necessary to apply our general results in Table 2 to the specific attention operations in Definition 2.2. This presentation is for the ease of exposition, and we will formally present the bounds in the following.

Theorem I.1. Assuming bounded inputs (assumption I.1) and bounded model parameters (assumption I.2), we can bound the stability of the model output $C ( \Theta )$ with respect to perturbations to the model parameters Θ as follows for any $\Theta , \Theta ^ { \prime }$ given $( X , C )$ as input:

$$
\begin{array} { r l } & { \| C ( \Theta ) - C ( \Theta ^ { \prime } ) \| _ { 2 , 1 } \leq \phi \cdot \| \Theta - \Theta ^ { \prime } \| , } \\ & { \quad u h e r e \phi \sim \mathcal { O } \left( \operatorname* { m a x } \{ 1 , \gamma ^ { 2 } \} \cdot \Big ( k \sqrt { d _ { E } } \Big ) \cdot \operatorname* { m a x } \left\{ 1 , \left( \frac { d _ { E } } { \sqrt { d _ { \mathrm { e m b } } } } \right) ^ { 2 } \right\} \cdot p o l y \left( \exp \left( \gamma \frac { d _ { E } } { \sqrt { d _ { \mathrm { e m b } } } } \right) \right) \right) . } \end{array}\tag{115}
$$

Proof. Given that $\Theta = \{ ( Q _ { \mu } , K _ { \mu } , V _ { \mu } ) , \mu = 1 , . . . , 4 \}$ and $\Theta ^ { \prime } = \{ ( Q _ { \mu } ^ { \prime } , K _ { \mu } ^ { \prime } , V _ { \mu } ^ { \prime } ) , \mu = 1 , \ldots , 4 \}$ , we will study the effect of modifying a single weight matrix at a time for ease of exposition (for example, effect of modifying $Q _ { 1 } ~ \mathrm { t o } ~ Q _ { 1 } ^ { \prime } )$ . In terms of notation, when we modify a single weight matrix (say $Q _ { 1 } ~ \mathrm { t o } ~ Q _ { 1 } ^ { \prime } )$ , and compare the outputs, we denote the outputs of the respective models as $C ( Q _ { 1 } )$ and $C ( Q _ { 1 } ^ { \prime } )$ to make it explicit that we are studying the effect of just modifying $Q _ { 1 }$ to $Q _ { 1 } ^ { \prime }$ . To make explicit the effect of the model parameters, we will also write the final embeddings of points and clusters centers as $\bar { \pmb X } ^ { ( 1 ) } ( { \pmb Q } _ { \mu } )$ VS $\bar { X } ^ { ( 1 ) } ( Q _ { \mu } ^ { \prime } )$ and $\bar { C } ^ { ( 1 ) } ( Q _ { \mu } )$ VS $\bar { C } ^ { ( 1 ) } ( Q _ { \mu } ^ { \prime } )$ only if the change in the considered weight matrix affects the token embeddings. Note that modifying any of the $( Q _ { \mu } , K _ { \mu } , V _ { \mu } ) , \mu = 3$ 4 does not affect the final point embeddings $\bar { X } ^ { ( 1 ) }$ , and in those situations, we will just denote the point embeddings as $\bar { X } ^ { ( 1 ) }$

With that in mind, we first study the effect of modifying $( Q _ { 4 } , K _ { 4 } , V _ { 4 } )$ one at a time, applying Lemma ${ \mathrm { I . 6 } } ,$ Lemma I.7, Lemma I.5, respectively. We see that

$$
\begin{array} { r } { \| C ( Q _ { 4 } ) - C ( Q _ { 4 } ^ { \prime } ) \| _ { 2 , 1 } \leq \left\| \bar { C } ^ { ( 1 ) } ( Q _ { 4 } ) - \bar { C } ^ { ( 1 ) } ( Q _ { 4 } ^ { \prime } ) \right\| _ { 2 , 1 } } \end{array}\tag{116}
$$

Table 2: Different quantities affecting the various bounds in Lemma I.4, Lemma I.3, Lemma I.5, Lemma I.6, Lemma I.7 and Lemma I.8. First, we provide a description of the terms, and then we present the effect on the bound when the terms of interest are increased. In almost all cases, the bounds worsen with increase in the term. The only quantities that lead to a decrease in the terms are the embedding dimension $d _ { \mathsf { e m b } }$ and the number of key tokens $m ^ { \prime }$ We
<table><tr><td>Term</td><td>Description</td></tr><tr><td> $m$ </td><td>Number of query tokens</td></tr><tr><td> $m ^ { \prime }$ </td><td>Number of key tokens</td></tr><tr><td> $\rho$ </td><td>Upper-bound on the query token norms</td></tr><tr><td> $\rho ^ { \prime }$ </td><td>Upper-bound on the key token norms</td></tr><tr><td> $\Gamma _ { 1 }$ </td><td>Upper-bound on the spectral norm of the query/key projection matrix</td></tr><tr><td> $\Gamma _ { 2 }$ </td><td>Upper-bound on the spectral norm of the value projection matrix</td></tr><tr><td> $\gamma$ </td><td>Inverse temperature for the attention operation in Equation (3)</td></tr><tr><td> $d _ { \mathsf { e m b } }$ </td><td>Token embedding dimension for the transformer architecture</td></tr><tr><td> $\delta ( \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \gamma , d _ { \mathrm { e m b } } )$ </td><td>Maximum dispersion in the attention softmax (defined in Lemma I.4)</td></tr><tr><td> $\kappa _ { 2 } ( m , \rho ^ { \prime } )$ </td><td>Stability constant w.r.t. the value projection matrix in Lemma I.5</td></tr><tr><td> $\kappa _ { 3 } ( m , \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } )$ </td><td>Stability constant w.r.t. the query projection matrix in Lemma I.6</td></tr><tr><td> $\kappa _ { 4 } ( m , \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } )$ </td><td>Stability constant w.r.t. the key projection matrix in Lemma I.7</td></tr><tr><td> $\kappa _ { 5 } ( m , m ^ { \prime } , \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } )$ </td><td>Stability constant  $\mathtt { w . r . t . }$  the key tokens in Lemma I.8</td></tr></table>

<table><tr><td> $\delta ( \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \gamma , d _ { \mathrm { e m b } } )$  (Lemma I.4)</td><td> $\kappa _ { 2 } ( m , \rho ^ { \prime } )$  (Lemma I.5)</td><td> $\kappa _ { 3 } ( m , \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \ ( \mathtt { L e m m a \ } \mathrm { I . 6 } )$ </td></tr><tr><td> $( \gamma / \sqrt { d _ { \mathsf { e m b } } } ) \Gamma _ { 1 } ^ { 2 } ( \rho + \rho ^ { \prime } ) ^ { 2 }$ </td><td> $m \rho ^ { \prime }$ </td><td> $2 ( \gamma / \sqrt { d _ { \mathrm { e m b } } } ) m \rho \rho ^ { \prime } ( \rho + \rho ^ { \prime } ) \Gamma _ { 1 } \Gamma _ { 2 } \exp ( \delta ( \cdot \cdot \cdot ) )$ </td></tr><tr><td> $m$ </td><td></td><td>4</td></tr><tr><td> $m ^ { \prime }$ </td><td></td><td></td></tr><tr><td> $\rho$ </td><td></td><td>▲</td></tr><tr><td> $\rho ^ { \prime }$ </td><td>▲</td><td>4</td></tr><tr><td> $\Gamma _ { 1 }$ </td><td>4</td><td>▲</td></tr><tr><td> $\Gamma _ { 2 }$ </td><td></td><td>▲</td></tr><tr><td> $\gamma$   $d _ { \mathsf { e m b } }$ </td><td>▲</td><td>▲</td></tr><tr><td></td><td></td><td>▼</td></tr><tr><td> $\kappa _ { 4 } ( m , \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \ ( \mathrm { L e m m a \ [ . 7 ] }$ </td><td></td><td> $\kappa _ { 5 } ( m , m ^ { \prime } , \rho , \rho ^ { \prime } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \ ( \mathrm { L e m m a \ 1 . 8 ) }$ </td></tr><tr><td></td><td> $2 ( \gamma / \sqrt { d _ { \mathrm { e m b } } } ) m \rho ^ { \prime 2 } ( \rho + \rho ^ { \prime } ) \Gamma _ { 1 } \Gamma _ { 2 } \exp ( \delta ( \cdot \cdot \cdot ) )$ </td><td> $( m \Gamma _ { 2 } / m ^ { \prime } ) ( 1 + 2 ( \gamma / \sqrt { d _ { \mathrm { e m b } } } ) \Gamma _ { 1 } ^ { 2 } \rho ^ { \prime } ( \rho + \rho ^ { \prime } ) ) \exp ( \delta ( \cdot \cdot \cdot ) )$ </td></tr><tr><td> $m$ </td><td></td><td></td></tr><tr><td> $m ^ { \prime }$ </td><td></td><td></td></tr><tr><td> $\rho$ </td><td></td><td></td></tr><tr><td> $\rho ^ { \prime }$ </td><td></td><td></td></tr><tr><td> $\Gamma _ { 1 }$ </td><td></td><td></td></tr><tr><td> $\Gamma _ { 2 }$ </td><td></td><td></td></tr><tr><td> $\begin{array} { l } { \gamma } \\ { d _ { \mathrm { e m b } } } \end{array}$ </td><td></td><td>▲</td></tr><tr><td></td><td>▼</td><td>▼</td></tr></table>

$$
= \Big \| \mathcal { A } _ { 2 } ^ { \gamma } ( \bar { C } ^ { ( 0 ) } , \bar { C } ^ { ( 0 ) } ; ( Q _ { 4 } , K _ { 4 } , V _ { 4 } ) ) - \mathcal { A } _ { 2 } ^ { \gamma } ( \bar { C } ^ { ( 0 ) } , \bar { C } ^ { ( 0 ) } ; ( Q _ { 4 } ^ { \prime } , K _ { 4 } , V _ { 4 } ) ) \Big \| _ { 2 , 1 }\tag{117}
$$

$$
\leq \kappa _ { 3 } ( k , \sqrt { d _ { E } } , \sqrt { d _ { E } } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathsf { e m b } } ) \| Q _ { 4 } - Q _ { 4 } ^ { \prime } \|\tag{118}
$$

$$
\begin{array} { r } { \| C ( K _ { 4 } ) - C ( K _ { 4 } ^ { \prime } ) \| _ { 2 , 1 } \le \Big \| \bar { C } ^ { ( 1 ) } ( K _ { 4 } ) - \bar { C } ^ { ( 1 ) } ( K _ { 4 } ^ { \prime } ) \Big \| _ { 2 , 1 } } \end{array}\tag{119}
$$

$$
= \Big \| \mathcal { A } _ { 2 } ^ { \gamma } ( \bar { C } ^ { ( 0 ) } , \bar { C } ^ { ( 0 ) } ; ( Q _ { 4 } , K _ { 4 } , V _ { 4 } ) ) - \mathcal { A } _ { 2 } ^ { \gamma } ( \bar { C } ^ { ( 0 ) } , \bar { C } ^ { ( 0 ) } ; ( Q _ { 4 } , K _ { 4 } ^ { \prime } , V _ { 4 } ) ) \Big \| _ { 2 , 1 }\tag{120}
$$

$$
\begin{array} { r l r } {  { \le \ \kappa _ { 4 } ( k , \sqrt { d _ { E } } , \sqrt { d _ { E } } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \| K _ { 4 } - K _ { 4 } ^ { \prime } \| } } \quad  & { { } } & { } \end{array}\tag{121}
$$

$$
\begin{array} { r } { \| \boldsymbol { C } ( \boldsymbol { V } _ { 4 } ) - \boldsymbol { C } ( \boldsymbol { V } _ { 4 } ^ { \prime } ) \| _ { 2 , 1 } \ \le \left\| \boldsymbol { \bar { C } } ^ { ( 1 ) } ( \boldsymbol { V } _ { 4 } ) - \boldsymbol { \bar { C } } ^ { ( 1 ) } ( \boldsymbol { V } _ { 4 } ^ { \prime } ) \right\| _ { 2 , 1 } } \end{array}\tag{122}
$$

Table 3: The value of the relevant terms from Table 2 in the different attention operations in Definition 2.2. We write the attention operations compactly as $\mathcal { A } _ { 2 } ^ { \gamma } ( Z , Z ^ { \prime } , ( \cdot , \mu ) )$ for query tokens $\boldsymbol { Z } \in \mathbb { R } ^ { d _ { \mathrm { e m b } } \times m }$ , key tokens $\boldsymbol { Z ^ { \prime } } \in \mathbb { R } ^ { d _ { \mathrm { e m b } } \times m ^ { \prime } }$ , where (i) the query token norms are bounded as $\| Z _ { : , i } \| \leq \rho$ for all $i \in [ m ]$ , (ii) the key token norms are $\left\| Z _ { : , j } ^ { \prime } \right\| \le \rho ^ { \prime }$ for all $j \in [ m ^ { \prime } ]$ , and (iii) the $( \cdots , \mu )$ corresponds to the appropriate $( Q _ { \mu } , K _ { \mu } , V _ { \mu } )$ query/key/value projection matrices. The bounds on the norms of the key and query tokens are obtained from Lemma I.1
<table><tr><td></td><td></td><td></td><td> $\begin{array} { r l r l r l } & { A _ { 2 } ^ { \gamma } ( \bar { X } ^ { ( 0 ) } , \bar { C } ^ { ( 0 ) } ; ( \cdot , 1 ) ) } & & { A _ { 2 } ^ { \gamma } ( \bar { X } ^ { ( 0 ) } , \bar { X } ^ { ( 0 ) } ; ( \cdot , 2 ) ) } & & { A _ { 2 } ^ { \gamma } ( \bar { C } ^ { ( 0 ) } , \bar { X } ^ { ( 1 ) } ; ( \cdot , 3 ) ) } & & { A _ { 2 } ^ { \gamma } ( \bar { C } ^ { ( 0 ) } , \bar { C } ^ { ( 0 ) } ; ( \cdot , 4 ) ) } \end{array}$ </td><td></td></tr><tr><td>m</td><td>n</td><td>n</td><td>k</td><td>k</td></tr><tr><td> $m ^ { \prime }$ </td><td>k</td><td>n</td><td>n</td><td>k</td></tr><tr><td>ρ</td><td> $\sqrt { d }$ </td><td>√d</td><td>√dE</td><td> $\sqrt { d _ { E } }$ </td></tr><tr><td> $\rho ^ { \prime }$ </td><td> $\sqrt { d _ { E } }$ </td><td> $\sqrt { d }$ </td><td> $( 1 + \Gamma _ { 2 } ) \sqrt { d } + \Gamma _ { 2 } \sqrt { d _ { E } }$ </td><td> $\sqrt { d _ { E } }$ </td></tr></table>

$$
= \Big \| \mathcal { A } _ { 2 } ^ { \gamma } ( \bar { C } ^ { ( 0 ) } , \bar { C } ^ { ( 0 ) } ; ( Q _ { 4 } , K _ { 4 } , V _ { 4 } ) ) - \mathcal { A } _ { 2 } ^ { \gamma } ( \bar { C } ^ { ( 0 ) } , \bar { C } ^ { ( 0 ) } ; ( Q _ { 4 } , K _ { 4 } , V _ { 4 } ^ { \prime } ) ) \Big \| _ { 2 , 1 }\tag{123}
$$

$$
\leq \kappa _ { 2 } ( k , \sqrt { d _ { E } } ) \| V _ { 4 } - V _ { 4 } ^ { \prime } \| \ .\tag{124}
$$

Next, we consider the effect of modifying $( Q _ { 3 } , K _ { 3 } , V _ { 3 } )$ one at a time, applying Lemma I.6, Lemma I.7, Lemma I.5, respectively:

$$
\big \lVert C ( Q _ { 3 } ) - C ( Q _ { 3 } ^ { \prime } ) \big \rVert _ { 2 , 1 } = \Big \lVert A _ { 2 } ^ { \gamma } ( \bar { C } ^ { ( 0 ) } , \bar { X } ^ { ( 1 ) } ; ( Q _ { 3 } , K _ { 3 } , V _ { 3 } ) ) - A _ { 2 } ^ { \gamma } ( \bar { C } ^ { ( 0 ) } , \bar { X } ^ { ( 1 ) } ; ( Q _ { 3 } ^ { \prime } , K _ { 3 } , V _ { 3 } ) ) \Big \rVert _ { 2 , 1 }\tag{125}
$$

$$
\begin{array} { r l } { \leq } & { { } \kappa _ { 3 } ( k , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \left| \right| Q _ { 3 } - Q _ { 3 } ^ { \prime } \left| \right| } \end{array}\tag{126}
$$

$$
\begin{array} { r } { \| C ( K _ { 3 } ) - C ( K _ { 3 } ^ { \prime } ) \| _ { 2 , 1 } = \Big \| A _ { 2 } ^ { \gamma } ( \bar { C } ^ { ( 0 ) } , \bar { X } ^ { ( 1 ) } ; ( Q _ { 3 } , K _ { 3 } , V _ { 3 } ) ) - A _ { 2 } ^ { \gamma } ( \bar { C } ^ { ( 0 ) } , \bar { X } ^ { ( 1 ) } ; ( Q _ { 3 } , K _ { 3 } ^ { \prime } , V _ { 3 } ) ) \Big \| _ { 2 , 1 } } \end{array}\tag{127}
$$

$$
\begin{array} { r l } { \le } & { { } \kappa _ { 4 } ( k , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \left\| K _ { 3 } - K _ { 3 } ^ { \prime } \right\| } \end{array}\tag{128}
$$

$$
\begin{array} { r } { \| C ( V _ { 3 } ) - C ( V _ { 3 } ^ { \prime } ) \| _ { 2 , 1 } = \Big \| A _ { 2 } ^ { \gamma } ( \bar { C } ^ { ( 0 ) } , \bar { X } ^ { ( 1 ) } ; ( Q _ { 3 } , K _ { 3 } , V _ { 3 } ) ) - A _ { 2 } ^ { \gamma } ( \bar { C } ^ { ( 0 ) } , \bar { X } ^ { ( 1 ) } ; ( Q _ { 3 } , K _ { 3 } , V _ { 3 } ^ { \prime } ) ) \Big \| _ { 2 , 1 } } \end{array}\tag{129}
$$

$$
\leq \kappa _ { 2 } ( k , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) ) \| V _ { 3 } - V _ { 3 } ^ { \prime } \|\tag{130}
$$

Studying the effect of modifying $( Q _ { 1 } , K _ { 1 } , V _ { 1 } )$ one at a time, we need to incorporate the effect of the modified $\bar { X } ^ { ( 1 ) }$ using Lemma I.8 along with the use of Lemma I.6, Lemma I.7, Lemma I.5:

(131)

$$
= \Big \| \mathcal { A } _ { 2 } ^ { \gamma } ( \bar { C } ^ { ( 0 ) } , \bar { X } ^ { ( 1 ) } ( Q _ { 1 } ) ; ( Q _ { 3 } , K _ { 3 } , V _ { 3 } ) ) - \mathcal { A } _ { 2 } ^ { \gamma } ( \bar { C } ^ { ( 0 ) } , \bar { X } ^ { ( 1 ) } ( Q _ { 1 } ^ { \prime } ) ; ( Q _ { 3 } , K _ { 3 } , V _ { 3 } ) ) \Big \| _ { 2 , 1 }\tag{132}
$$

$$
\leq \kappa _ { 5 } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \left\| \bar { \pmb X } ^ { ( 1 ) } ( \pmb Q _ { 1 } ) - \bar { \pmb X } ^ { ( 1 ) } ( \pmb Q _ { 1 } ^ { \prime } ) \right\| _ { 2 , 1 }\tag{133}
$$

$$
\leq \kappa _ { 5 } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } )
$$

$$
\begin{array} { r } { \Big \| \mathcal { A } _ { 2 } ^ { \gamma } ( \bar { X } ^ { ( 0 ) } , \bar { C } ^ { ( 0 ) } ; ( Q _ { 1 } , K _ { 1 } , V _ { 1 } ) ) - \mathcal { A } _ { 2 } ^ { \gamma } ( \bar { X } ^ { ( 0 ) } , \bar { C } ^ { ( 0 ) } ; ( Q _ { 1 } ^ { \prime } , K _ { 1 } , V _ { 1 } ) ) \Big \| _ { 2 , 1 } } \end{array}\tag{134}
$$

$$
\leq | \kappa _ { 5 } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \cdot \kappa _ { 3 } ( n , \sqrt { d } , \sqrt { d _ { E } } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) | | Q _ { 1 } - Q _ { 1 } ^ { \prime } |\tag{135}
$$

(136)

$$
= \Big \| A _ { 2 } ^ { \gamma } ( \bar { C } ^ { ( 0 ) } , \bar { X } ^ { ( 1 ) } ( K _ { 1 } ) ; ( Q _ { 3 } , K _ { 3 } , V _ { 3 } ) ) - A _ { 2 } ^ { \gamma } ( \bar { C } ^ { ( 0 ) } , \bar { X } ^ { ( 1 ) } ( K _ { 1 } ^ { \prime } ) ; ( Q _ { 3 } , K _ { 3 } , V _ { 3 } ) ) \Big \| _ { 2 , 1 }\tag{137}
$$

$$
\leq \kappa _ { 5 } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \left\| \bar { X } ^ { ( 1 ) } ( K _ { 1 } ) - \bar { X } ^ { ( 1 ) } ( K _ { 1 } ^ { \prime } ) \right\| _ { 2 , 1 }\tag{138}
$$

$$
\leq \kappa _ { 5 } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } )
$$

$$
\begin{array} { r } { \Big \| \mathcal { A } _ { 2 } ^ { \gamma } ( \bar { X } ^ { ( 0 ) } , \bar { C } ^ { ( 0 ) } ; ( Q _ { 1 } , K _ { 1 } , V _ { 1 } ) ) - \mathcal { A } _ { 2 } ^ { \gamma } ( \bar { X } ^ { ( 0 ) } , \bar { C } ^ { ( 0 ) } ; ( Q _ { 1 } , K _ { 1 } ^ { \prime } , V _ { 1 } ) ) \Big \| _ { 2 , 1 } } \end{array}\tag{139}
$$

$$
\leq | \kappa _ { 5 } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \cdot \kappa _ { 4 } ( n , \sqrt { d } , \sqrt { d _ { E } } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) | | K _ { 1 } - K _ { 1 } ^ { \prime } |\tag{140}
$$

(141)

$$
= \Big \| \mathcal { A } _ { 2 } ^ { \gamma } ( \bar { C } ^ { ( 0 ) } , \bar { X } ^ { ( 1 ) } ( V _ { 1 } ) ; ( Q _ { 3 } , K _ { 3 } , V _ { 3 } ) ) - \mathcal { A } _ { 2 } ^ { \gamma } ( \bar { C } ^ { ( 0 ) } , \bar { X } ^ { ( 1 ) } ( V _ { 1 } ^ { \prime } ) ; ( Q _ { 3 } , K _ { 3 } , V _ { 3 } ) ) \Big \| _ { 2 , 1 }\tag{142}
$$

$$
\leq \kappa _ { 5 } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \left| \left| \bar { \mathbfcal X } ^ { ( 1 ) } ( V _ { 1 } ) - \bar { \mathbfcal X } ^ { ( 1 ) } ( V _ { 1 } ^ { \prime } ) \right| \right| _ { 2 , 1 }\tag{143}
$$

$$
\leq \kappa _ { 5 } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } )
$$

$$
\begin{array} { r } { \left\| \mathcal { A } _ { 2 } ^ { \gamma } ( \bar { X } ^ { ( 0 ) } , \bar { C } ^ { ( 0 ) } ; ( Q _ { 1 } , K _ { 1 } , V _ { 1 } ) ) - \mathcal { A } _ { 2 } ^ { \gamma } ( \bar { X } ^ { ( 0 ) } , \bar { C } ^ { ( 0 ) } ; ( Q _ { 1 } , K _ { 1 } , V _ { 1 } ^ { \prime } ) ) \right\| _ { 2 , 1 } } \end{array}\tag{144}
$$

$$
\begin{array} { r l } { } & { { } \leq \ \kappa _ { 5 } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \cdot \kappa _ { 2 } ( n , \sqrt { d _ { E } } ) \ \| V _ { 1 } - V _ { 1 } ^ { \prime } \| } \end{array}\tag{145}
$$

Finally, we consider $( Q _ { 2 } , K _ { 2 } , V _ { 2 } )$ employing the previously used results:

(146)

$$
= \Big \| A _ { 2 } ^ { \gamma } ( \bar { C } ^ { ( 0 ) } , \bar { X } ^ { ( 1 ) } ( Q _ { 2 } ) ; ( Q _ { 3 } , K _ { 3 } , V _ { 3 } ) ) - A _ { 2 } ^ { \gamma } ( \bar { C } ^ { ( 0 ) } , \bar { X } ^ { ( 1 ) } ( Q _ { 2 } ^ { \prime } ) ; ( Q _ { 3 } , K _ { 3 } , V _ { 3 } ) ) \Big \| _ { 2 , 1 }\tag{147}
$$

$$
\leq \kappa _ { 5 } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \left\| \bar { \pmb X } ^ { ( 1 ) } ( \pmb Q _ { 2 } ) - \bar { \pmb X } ^ { ( 1 ) } ( \pmb Q _ { 2 } ^ { \prime } ) \right\| _ { 2 , 1 }\tag{148}
$$

$$
\leq \kappa _ { 5 } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } )
$$

$$
\begin{array} { r } { \left\| \mathcal { A } _ { 2 } ^ { \gamma } ( \bar { X } ^ { ( 0 ) } , \bar { X } ^ { ( 0 ) } ; ( Q _ { 2 } , K _ { 2 } , V _ { 2 } ) ) - \mathcal { A } _ { 2 } ^ { \gamma } ( \bar { X } ^ { ( 0 ) } , \bar { X } ^ { ( 0 ) } ; ( Q _ { 2 } ^ { \prime } , K _ { 2 } , V _ { 2 } ) ) \right\| _ { 2 , 1 } } \end{array}\tag{149}
$$

$$
\leq | \kappa _ { 5 } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \kappa _ { 3 } ( n , \sqrt { d } , \sqrt { d } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) | | Q _ { 2 } - Q _ { 2 } ^ { \prime } |\tag{150}
$$

(151)

$$
= \Big \| \mathcal { A } _ { 2 } ^ { \gamma } ( \bar { C } ^ { ( 0 ) } , \bar { X } ^ { ( 1 ) } ( K _ { 2 } ) ; ( Q _ { 3 } , K _ { 3 } , V _ { 3 } ) ) - \mathcal { A } _ { 2 } ^ { \gamma } ( \bar { C } ^ { ( 0 ) } , \bar { X } ^ { ( 1 ) } ( K _ { 2 } ^ { \prime } ) ; ( Q _ { 3 } , K _ { 3 } , V _ { 3 } ) ) \Big \| _ { 2 , 1 }\tag{152}
$$

$$
\leq \kappa _ { 5 } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \left\| \bar { X } ^ { ( 1 ) } ( K _ { 2 } ) - \bar { X } ^ { ( 1 ) } ( K _ { 2 } ^ { \prime } ) \right\| _ { 2 , 1 }\tag{153}
$$

$$
\leq \kappa _ { 5 } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } )
$$

$$
\begin{array} { r } { \left\| \mathcal { A } _ { 2 } ^ { \gamma } ( \bar { X } ^ { ( 0 ) } , \bar { X } ^ { ( 0 ) } ; ( Q _ { 2 } , K _ { 2 } , V _ { 2 } ) ) - \mathcal { A } _ { 2 } ^ { \gamma } ( \bar { X } ^ { ( 0 ) } , \bar { X } ^ { ( 0 ) } ; ( Q _ { 2 } , K _ { 2 } ^ { \prime } , V _ { 2 } ) ) \right\| _ { 2 , 1 } } \end{array}\tag{154}
$$

$$
\leq _ { \mathrm { \ell } \kappa _ { 5 } } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \kappa _ { 4 } ( n , \sqrt { d } , \sqrt { d } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \| K _ { 2 } - K _ { 2 } ^ { \prime } \|\tag{155}
$$

$$
\| C ( V _ { 2 } ) - C ( V _ { 2 } ^ { \prime } ) \| _ { 2 , 1 }\tag{156}
$$

$$
= \Big \| \mathcal { A } _ { 2 } ^ { \gamma } ( \bar { C } ^ { ( 0 ) } , \bar { X } ^ { ( 1 ) } ( V _ { 2 } ) ; ( Q _ { 3 } , K _ { 3 } , V _ { 3 } ) ) - \mathcal { A } _ { 2 } ^ { \gamma } ( \bar { C } ^ { ( 0 ) } , \bar { X } ^ { ( 1 ) } ( V _ { 2 } ^ { \prime } ) ; ( Q _ { 3 } , K _ { 3 } , V _ { 3 } ) ) \Big \| _ { 2 , 1 }\tag{157}
$$

$$
\leq \kappa _ { 5 } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \left\| \bar { \mathbfcal X } ^ { ( 1 ) } ( V _ { 2 } ) - \bar { \mathbfcal X } ^ { ( 1 ) } ( V _ { 2 } ^ { \prime } ) \right\| _ { 2 , 1 }\tag{158}
$$

$$
\leq \kappa _ { 5 } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } )
$$

$$
\begin{array} { r } { \left\| \mathcal { A } _ { 2 } ^ { \gamma } ( \bar { X } ^ { ( 0 ) } , \bar { X } ^ { ( 0 ) } ; ( Q _ { 2 } , K _ { 2 } , V _ { 2 } ) ) - \mathcal { A } _ { 2 } ^ { \gamma } ( \bar { X } ^ { ( 0 ) } , \bar { X } ^ { ( 0 ) } ; ( Q _ { 2 } , K _ { 2 } , V _ { 2 } ^ { \prime } ) ) \right\| _ { 2 , 1 } } \end{array}\tag{159}
$$

$$
\begin{array} { r l r } & { { } } & { \leq \kappa _ { 5 } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \kappa _ { 2 } ( n , \sqrt { d } ) \| V _ { 2 } - V _ { 2 } ^ { \prime } \| } \end{array}\tag{160}
$$

Putting all these together, and noting the definition of $\| \Theta - \Theta ^ { \prime } \|$ as the maximum of the norms of the differences of the individual weight matrices, we can see that:

$$
\| C ( \Theta ) - C ( \Theta ^ { \prime } ) \| _ { 2 , 1 } \leq \phi \cdot \| \Theta - \Theta ^ { \prime } \| ,\tag{161}
$$

where the quantity $\phi$ is defined as (also noting the equations from which the quantity is derived):

$$
\kappa _ { 5 } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \cdot \kappa _ { 3 } ( n , \sqrt { d } , \sqrt { d _ { E } } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } )\tag{135}
$$

$$
\kappa _ { 5 } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \cdot \kappa _ { 4 } ( n , \sqrt { d } , \sqrt { d _ { E } } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } )\tag{140}
$$

$$
\kappa _ { 5 } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \cdot \kappa _ { 2 } ( n , \sqrt { d _ { E } } )\tag{145}
$$

$$
\kappa _ { 5 } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \cdot \kappa _ { 3 } ( n , \sqrt { d } , \sqrt { d } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } )\tag{150}
$$

$$
\kappa _ { 5 } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \cdot \kappa _ { 4 } ( n , \sqrt { d } , \sqrt { d } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } )\tag{155}
$$

$$
\phi = \operatorname { m a x } \quad \int \kappa _ { 5 } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \cdot \kappa _ { 2 } ( n , \sqrt { d } )\tag{160}
$$

$$
\kappa _ { 3 } ( k , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathsf { e m b } } )\tag{162}
$$

(126)

$$
\kappa _ { 4 } ( k , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } )\tag{128}
$$

$$
\kappa _ { 2 } ( k , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) )\tag{130}
$$

$$
\kappa _ { 3 } ( k , \sqrt { d _ { E } } , \sqrt { d _ { E } } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { \small e m b } } )\tag{118}
$$

$$
\kappa _ { 4 } ( k , \sqrt { d _ { E } } , \sqrt { d _ { E } } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { \small e m b } } )\tag{121}
$$

$$
\kappa _ { 2 } ( k , \sqrt { d _ { E } } )\tag{124}
$$

To simplify the above definition of $\phi ,$ we note that, for a fixed $( m , \rho , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } )$ , a larger $\rho ^ { \prime }$ implies larger constant. Thus, we can see that

$$
\kappa _ { 2 } ( k , \sqrt { d _ { E } } ) < \kappa _ { 2 } ( k , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) )\tag{163}
$$

$$
\kappa _ { 2 } ( k , \sqrt { d } ) < \kappa _ { 2 } ( k , \sqrt { d _ { E } } )\tag{164}
$$

$$
\kappa _ { 3 } ( k , \sqrt { d _ { E } } , \sqrt { d _ { E } } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) < \kappa _ { 3 } ( k , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } )\tag{165}
$$

$$
\kappa _ { 3 } ( n , \sqrt { d } , \sqrt { d } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) < \kappa _ { 3 } ( n , \sqrt { d } , \sqrt { d _ { E } } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } )\tag{166}
$$

$$
\kappa _ { 4 } ( k , \sqrt { d _ { E } } , \sqrt { d _ { E } } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) < \kappa _ { 4 } ( k , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } )\tag{167}
$$

$$
\kappa _ { 4 } ( n , \sqrt { d } , \sqrt { d } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) < \kappa _ { 4 } ( n , \sqrt { d } , \sqrt { d _ { E } } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) ,\tag{168}
$$

thus simplifying further to

$$
\phi = \operatorname* { m a x } \left\{ \begin{array} { l } { \kappa _ { 5 } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \cdot \kappa _ { 3 } ( n , \sqrt { d } , \sqrt { d _ { E } } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) } \\ { \kappa _ { 5 } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \cdot \kappa _ { 4 } ( n , \sqrt { d _ { E } } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) } \\ { \kappa _ { 5 } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \cdot \kappa _ { 2 } ( n , \sqrt { d _ { E } } ) } \\ { \kappa _ { 3 } ( k , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) } \\ { \kappa _ { 4 } ( k , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) } \\ { \kappa _ { 2 } ( k , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) ) } \end{array} \right\}\tag{169}
$$

Next given the definitions of $\kappa _ { 3 }$ and $\kappa _ { 4 }$ in Lemma I.6 and Lemma I.7 respectively, we can see that if $\rho < \rho ^ { \prime }$ $\kappa _ { 4 } ( \cdot )$ provides a larger constant, giving us:

$$
\phi = \operatorname* { m a x } \left\{ \begin{array} { l } { \kappa _ { 5 } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \cdot \kappa _ { 4 } ( n , \sqrt { d } , \sqrt { d _ { E } } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) } \\ { \kappa _ { 5 } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \cdot \kappa _ { 2 } ( n , \sqrt { d _ { E } } ) } \\ { \kappa _ { 4 } ( k , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) } \\ { \kappa _ { 2 } ( k , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) ) } \end{array} \right\}\tag{170}
$$

Writing things out explicitly while highlighting the dependencies on problem and architectural choices, $n , k , d _ { \mathrm { e m b } } , d _ { E }$ , we first apply the definition of $\kappa _ { 2 } ( \cdots )$ in Lemma I.5 to $\kappa _ { 2 } ( k , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) )$ to get:

$$
\kappa _ { 2 } ( k , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) ) = [ 1 + 2 \Gamma _ { 2 } ] \ k d _ { E } ^ { 1 / 2 }\tag{171}
$$

Next, we apply the definition of $\kappa _ { 4 } ( \cdot \cdot \cdot )$ in Lemma I.7 to $\kappa _ { 4 } ( k , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathsf { e m b } } )$ to get:

$$
\kappa _ { 4 } ( k , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathsf { e m b } } )\tag{172}
$$

$$
= 2 ( \gamma / \sqrt { d _ { \mathrm { e m b } } } ) k d _ { E } ( 1 + 2 \Gamma _ { 2 } ) ^ { 2 } \sqrt { d _ { E } } ( 2 + 2 \Gamma _ { 2 } ) \Gamma _ { 1 } \Gamma _ { 2 } \exp \left( ( \gamma / \sqrt { d _ { \mathrm { e m b } } } ) \Gamma _ { 1 } ^ { 2 } d _ { E } ( 2 + 2 \Gamma _ { 2 } ) ^ { 2 } \right)\tag{173}
$$

$$
\leq \frac { k d _ { E } ^ { 3 / 2 } } { \sqrt { d _ { \mathrm { e m b } } } } \left[ 4 \gamma ( 1 + 2 \Gamma _ { 2 } ) ^ { 2 } ( 1 + \Gamma _ { 2 } ) \Gamma _ { 1 } \Gamma _ { 2 } \right] \exp \left( \frac { d _ { E } } { \sqrt { d _ { \mathrm { e m b } } } } \left[ \gamma \Gamma _ { 1 } ^ { 2 } ( 2 + 2 \Gamma _ { 2 } ) ^ { 2 } \right] \right)\tag{174}
$$

$$
\leq \left[ 4 \gamma ( 1 + 2 \Gamma _ { 2 } ) ^ { 3 } \Gamma _ { 1 } \Gamma _ { 2 } \right] \cdot \frac { k d _ { E } ^ { 3 / 2 } } { \sqrt { d _ { \mathrm { e m b } } } } \cdot \exp \left( \frac { d _ { E } } { \sqrt { d _ { \mathrm { e m b } } } } \left[ \gamma \Gamma _ { 1 } ^ { 2 } ( 2 + 2 \Gamma _ { 2 } ) ^ { 2 } \right] \right)\tag{175}
$$

Applying definitions of $\kappa _ { 2 } ( \cdots )$ and $\kappa _ { 5 } ( \cdots )$ from Lemma I.5 and Lemma I.8 respectively to $\kappa _ { 5 } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 +$ $2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathsf { e m b } } ) \kappa _ { 2 } ( n , \sqrt { d _ { E } } )$ , we get:

$$
\kappa _ { 5 } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \kappa _ { 2 } ( n , \sqrt { d _ { E } } )\tag{176}
$$

$$
= \frac { k \Gamma _ { 2 } } { n } \left[ 1 + 2 ( \gamma / \sqrt { { d _ { \mathrm { e m b } } } } ) \Gamma _ { 1 } ^ { 2 } d _ { E } ( 1 + 2 \Gamma _ { 2 } ) ( 2 + 2 \Gamma _ { 2 } ) \right] \exp \left( ( \gamma / \sqrt { { d _ { \mathrm { e m b } } } } ) \Gamma _ { 1 } ^ { 2 } d _ { E } ( 2 + 2 \Gamma _ { 2 } ) ^ { 2 } \right) n \sqrt { { d _ { E } } }\tag{177}
$$

$$
\leq \Gamma _ { 2 } \left[ k d _ { E } ^ { 1 / 2 } ~ + ~ \frac { k d _ { E } ^ { 3 / 2 } } { \sqrt { d _ { \mathrm { e m b } } } } ~ \left[ 4 \gamma \Gamma _ { 1 } ^ { 2 } ( 1 + 2 \Gamma _ { 2 } ) ^ { 2 } \right] \right] \exp \left( ~ \frac { d _ { E } } { \sqrt { d _ { \mathrm { e m b } } } } ~ \left[ \gamma \Gamma _ { 1 } ^ { 2 } ( 2 + 2 \Gamma _ { 2 } ) ^ { 2 } \right] \right)\tag{178}
$$

Applying definitions of $\kappa _ { 4 } ( \cdot \cdot \cdot )$ and $\kappa _ { 5 } ( \cdots )$ from Lemma I.7 and Lemma I.8 respectively to $\kappa _ { 5 } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 +$ $2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \kappa _ { 4 } ( n , \sqrt { d } , \sqrt { d _ { E } } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } )$ , we get:

$$
\kappa _ { 5 } ( k , n , \sqrt { d _ { E } } , \sqrt { d _ { E } } ( 1 + 2 \Gamma _ { 2 } ) , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } ) \kappa _ { 4 } ( n , \sqrt { d } , \sqrt { d _ { E } } , \Gamma _ { 1 } , \Gamma _ { 2 } , \gamma , d _ { \mathrm { e m b } } )\tag{179}
$$

$$
= \frac { k \Gamma _ { 2 } } { n } \left[ 1 + 2 ( \gamma / \sqrt { d _ { \mathrm { e m b } } } ) \Gamma _ { 1 } ^ { 2 } d _ { E } ( 1 + 2 \Gamma _ { 2 } ) ( 2 + 2 \Gamma _ { 2 } ) \right] \exp \left( ( \gamma / \sqrt { d _ { \mathrm { e m b } } } ) \Gamma _ { 1 } ^ { 2 } d _ { E } ( 2 + 2 \Gamma _ { 2 } ) ^ { 2 } \right)
$$

$$
\cdot 2 ( \gamma / \sqrt { d _ { \mathrm { e m b } } } ) n d _ { E } ( \sqrt { d } + \sqrt { d _ { E } } ) \Gamma _ { 1 } \Gamma _ { 2 } \exp \Big ( ( \gamma / \sqrt { d _ { \mathrm { e m b } } } ) \Gamma _ { 1 } ^ { 2 } ( \sqrt { d } + \sqrt { d _ { E } } ) ^ { 2 } \Big )\tag{180}
$$

$$
\leq k \Gamma _ { 2 } \left[ 1 + \frac { d _ { E } } { \sqrt { d _ { \mathrm { e m b } } } } \left[ 4 \gamma \Gamma _ { 1 } ^ { 2 } ( 1 + 2 \Gamma _ { 2 } ) ^ { 2 } \right] \right] \cdot \frac { d _ { E } ^ { 3 / 2 } } { \sqrt { d _ { \mathrm { e m b } } } } \left[ 4 \gamma \Gamma _ { 1 } \Gamma _ { 2 } \right] \cdot \exp \left( ( \gamma / \sqrt { d _ { \mathrm { e m b } } } ) \Gamma _ { 1 } ^ { 2 } \left[ d _ { E } ( 2 + 2 \Gamma _ { 2 } ) ^ { 2 } + 4 d _ { E } \right] \right)\tag{181}
$$

$$
= \left[ 4 \gamma \Gamma _ { 1 } \Gamma _ { 2 } ^ { 2 } \right] \cdot \left[ \frac { k d _ { E } ^ { 3 / 2 } } { \sqrt { d _ { \mathrm { e m b } } } } + \frac { k d _ { E } ^ { 5 / 2 } } { d _ { \mathrm { e m b } } } \left[ 4 \gamma \Gamma _ { 1 } ^ { 2 } ( 1 + 2 \Gamma _ { 2 } ) ^ { 2 } \right] \right] \cdot \exp \left( \frac { d _ { E } } { \sqrt { d _ { \mathrm { e m b } } } } \left[ 4 \gamma \Gamma _ { 1 } ^ { 2 } \left[ ( 1 + \Gamma _ { 2 } ) ^ { 2 } + 1 \right] \right] \right)\tag{182}
$$

Note that in the above terms, there are two quantities that seem to repeatedly occur: (i) $k \sqrt { d _ { E } } ,$ and $( \mathrm { i i } ) ~ \frac { d _ { E } } { \sqrt { d _ { \mathrm { e m b } } } }$ Considering $\Gamma _ { 1 } , \Gamma _ { 2 }$ to be constants that do not depend on the architectural choices or problem setting (such as the number of clusters, points, data dimensionality or token embedding scheme), we can write $\phi$ as following:

$$
\phi = \operatorname* { m a x } \left\{ \begin{array} { l } { c _ { 1 } \left( k \sqrt { d _ { E } } \right) } \\ { c _ { 2 } \cdot \gamma \cdot \left( k \sqrt { d _ { E } } \right) \cdot \left( \frac { d _ { E } } { \sqrt { d _ { \mathrm { e m b } } } } \right) \cdot \exp \left( c _ { 3 } \cdot \gamma \cdot \left( \frac { d _ { E } } { \sqrt { d _ { \mathrm { e m b } } } } \right) \right) } \\ { \Gamma _ { 2 } \left[ \left( k \sqrt { d _ { E } } \right) + c _ { 4 } \cdot \gamma \cdot \left( k \sqrt { d _ { E } } \right) \cdot \left( \frac { d _ { E } } { \sqrt { d _ { \mathrm { e m b } } } } \right) \right] \cdot \exp \left( c _ { 3 } \cdot \gamma \cdot \left( \frac { d _ { E } } { \sqrt { d _ { \mathrm { e m b } } } } \right) \right) } \\ { \gamma \cdot \left( k \sqrt { d _ { E } } \right) \cdot \left( \frac { d _ { E } } { \sqrt { d _ { \mathrm { e m b } } } } \right) \left[ c _ { 5 } + c _ { 6 } \cdot \gamma \cdot \left( \frac { d _ { E } } { \sqrt { d _ { \mathrm { e m b } } } } \right) \right] \cdot \exp \left( c _ { 7 } \cdot \gamma \cdot \left( \frac { d _ { E } } { \sqrt { d _ { \mathrm { e m b } } } } \right) \right) } \end{array} \right\}\tag{183}
$$

where

$$
c _ { 1 } = \left( 1 + 2 \Gamma _ { 2 } \right)\tag{184}
$$

$$
c _ { 2 } = 4 \Gamma _ { 1 } \Gamma _ { 2 } c _ { 1 } ^ { 3 } = 4 \Gamma _ { 1 } \Gamma _ { 2 } ( 1 + 2 \Gamma _ { 2 } ) ^ { 3 }\tag{185}
$$

$$
c _ { 3 } = 4 \Gamma _ { 1 } ^ { 2 } ( 1 + \Gamma _ { 2 } ) ^ { 2 }\tag{186}
$$

$$
c _ { 4 } = 4 \Gamma _ { 1 } ^ { 2 } c _ { 1 } ^ { 2 } = 4 \Gamma _ { 1 } ^ { 2 } ( 1 + 2 \Gamma _ { 2 } ) ^ { 2 }\tag{187}
$$

$$
c _ { 5 } = 4 \Gamma _ { 1 } \Gamma _ { 2 } ^ { 2 }\tag{188}
$$

$$
c _ { 6 } = c _ { 4 } \cdot c _ { 5 } = 1 6 \Gamma _ { 1 } ^ { 3 } \Gamma _ { 2 } ^ { 2 } ( 1 + 2 \Gamma _ { 2 } ) ^ { 2 }\tag{189}
$$

$$
c _ { 7 } = c _ { 3 } + 4 \Gamma _ { 1 } ^ { 2 } = 4 \Gamma _ { 1 } ^ { 2 } ( ( 1 + \Gamma _ { 2 } ) ^ { 2 } + 1 ) .\tag{190}
$$

Thus, simplifying all, we can see that

$$
\phi \sim { \cal O } \left( \operatorname* { m a x } \{ 1 , \gamma ^ { 2 } \} \cdot \left( k \sqrt { d _ { E } } \right) \cdot \operatorname* { m a x } \left\{ 1 , \left( \frac { d _ { E } } { \sqrt { d _ { \mathrm { e m b } } } } \right) ^ { 2 } \right\} \cdot \mathsf { p o l y } \left( \exp \left( \gamma \frac { d _ { E } } { \sqrt { d _ { \mathrm { e m b } } } } \right) \right) \right) ,\tag{191}
$$

which gives us the result in Equation (115) in the statement of the theorem.

If $d _ { E } > \sqrt { d _ { \mathrm { e m b } } }$ which is the case with the proposed binary initial token embeddings, with $d _ { E } = d + \lceil \log _ { 2 } k \rceil$ and $d _ { \mathsf { e m b } } = d + \lceil \log _ { 2 } k \rceil$ , thus, $\begin{array} { r } { \frac { d _ { E } } { \sqrt { d _ { \mathsf { e m b } } } } = \sqrt { d + \lceil \log _ { 2 } k \rceil } > 1 } \end{array}$ , we can see that

$$
\phi \sim { \cal O } \left( \operatorname* { m a x } \{ 1 , \gamma ^ { 2 } \} \cdot \left( k \sqrt { d _ { E } } \right) \cdot \left( \frac { d _ { E } } { \sqrt { d _ { \mathrm { e m b } } } } \right) ^ { 2 } \cdot \mathsf { p o l y } \left( \exp \left( \gamma \frac { d _ { E } } { \sqrt { d _ { \mathrm { e m b } } } } \right) \right) \right) .\tag{192}
$$

However $d _ { E } < \sqrt { d _ { \mathrm { e m b } } }$ is possible with the OH initial token embedding of Clarkson et al. [2026] as $d _ { E } = d + 1$ and $d _ { \mathsf { e m b } } = d + k ,$ and $k \sim O ( ( d + 1 ) ^ { 2 } )$ implies that $d + 1 < { \sqrt { d + k } }$ In this case, we have

$$
\phi \sim O \left( \operatorname* { m a x } \{ 1 , \gamma ^ { 2 } \} \cdot \left( k \sqrt { d _ { E } } \right) \cdot \mathsf { p o l y } \left( \exp \left( \gamma \frac { d _ { E } } { \sqrt { d _ { \mathrm { e m b } } } } \right) \right) \right) .\tag{193}
$$

In all cases, for a fixed $k ,$ an architecture with a larger $d _ { \mathsf { e m b } }$ and smaller $d _ { E }$ results in a smaller $\phi .$

Combining the above Theorem I.1 with Lemma I.2, we can obtain the following main result regarding the Lipschitz-ness of the smoothed clustering loss function $\tilde { \mathcal { L } } _ { \Omega } ^ { \tau } ( X , F _ { d : k } ^ { \gamma } ( X , C ; \Theta ) )$ with respect to the learnable model parameters $\Theta$ (this is an extended version of Theorem 3.1):

Theorem I.2. Assuming bounded points and initial cluster centers $( \pmb { X } , \pmb { C } ) \sim \mathcal { D }$ (assumption $I . 1 )$ , and bounded learnable model parameters $\Theta , \Theta ^ { \prime }$ (assumption I.2), we can show that

$$
\begin{array} { r l } & { \left| \tilde { \mathcal { E } } _ { \Omega } ^ { \tau } ( \mathbf { X } , { H } _ { \mathrm { d } ; k } ^ { \gamma } ( \mathbf { X } , C ; \Theta ) ) - \tilde { \mathcal { L } } _ { \Omega } ^ { \tau } ( \mathbf { X } , { F } _ { d ; k } ^ { \gamma } ( \mathbf { X } , C ; \Theta ^ { \prime } ) ) \right| \leq ~ \varphi ~ \Vert \Theta - \Theta ^ { \prime } \Vert , } \\ & { ~ w h e r e ~ \varphi \sim O \left( \frac { k n \omega _ { \tau } } { \tau } \cdot \operatorname* { m a x } \{ 1 , \gamma ^ { 2 } \} \cdot \sqrt { d _ { E } } \cdot \operatorname* { m a x } \left\{ 1 , \left( \frac { d _ { E } } { \sqrt { d _ { \mathrm { e m b } } } } \right) ^ { 2 } \right\} \cdot p o l y \left( \exp \left( \gamma \frac { d _ { E } } { \sqrt { d _ { \mathrm { e m b } } } } \right) \right) \right) } \end{array}\tag{194}
$$

Proof. We can get the above result by combining Equation (115) in Theorem I.1 with Equation (45) in Lemma I.2:

$$
\begin{array} { r l } & { \left| \tilde { \mathcal { E } } _ { \Omega } ^ { \tau } ( X , F _ { { d } ; k } ^ { \gamma } ( X , C ; \Theta ) ) - \tilde { \mathcal { L } } _ { \Omega } ^ { \tau } ( X , F _ { { d } ; k } ^ { \gamma } ( X , C ; \Theta ^ { \prime } ) ) \right| \leq \varphi \cdot \| \Theta - \Theta ^ { \prime } \| } \\ & { \quad \mathrm { w h e r e ~ } \varphi \sim O \left( \frac { n \omega _ { \tau } } { \tau } \cdot \operatorname* { m a x } \{ 1 , \gamma ^ { 2 } \} \cdot \left( k \sqrt { d _ { E } } \right) \cdot \operatorname* { m a x } \left\{ 1 , \left( \frac { d _ { E } } { \sqrt { d _ { \mathrm { e m b } } } } \right) ^ { 2 } \right\} \cdot \mathsf { p o l y } \left( \exp \left( \gamma \frac { d _ { E } } { \sqrt { d _ { \mathrm { e m b } } } } \right) \right) \right) , } \end{array}\tag{195}
$$

thus giving us the main result.

This highlights the role of the different choices we make such as the choice of the min-regularizer $\Omega ,$ the corresponding regularization penalty $\tau ,$ the attention inverse temperature $\gamma ,$ the number of points n and the number of clusters $k ,$ the embedding dimension $d _ { \mathrm { e m b } }$ and the maximum initial cluster center embedding norm $d _ { E }$