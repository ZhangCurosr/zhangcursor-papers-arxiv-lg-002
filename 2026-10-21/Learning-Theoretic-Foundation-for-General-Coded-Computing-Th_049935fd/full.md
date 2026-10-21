# Learning-Theoretic Foundation for General Coded Computing: The Adversarial Setting

Parsa Moradi, Hanzaleh Akbarinodehi, and Mohammad Ali Maddah-Ali

## Abstract

Adversarial workers pose a severe challenge in distributed computing systems by returning corrupted outputs that can substantially distort the recovered computations. Coded computing addresses this threat by introducing redundancy across worker assignments. Existing exact and approximate coded-computing methods typically handle adversarial outputs by explicitly locating and correcting or removing erroneous evaluations, which adds significant computational overhead at the master node. Moreover, existing analyses do not characterize how the recovery error scales with the number of workers and the adversarial budget.

In this paper, we develop an adversarially robust extension of general coded computing (GCC), a learning-theoretic approximate coded-computing framework that designs the encoder and decoder by minimizing an upper bound on the end-to-end recovery error. Rather than identifying corrupted workers, the proposed method uses smoothness regularization to limit the influence of their outputs. The resulting encoder and decoder can be implemented using smoothing splines with linear master-side complexity. We derive a non-asymptotic bound on the worst-case end-to-end approximation error by decomposing it into three components: (1) the encoder approximation error, (2) the decoder reconstruction error under fully honest results, and (3) the adversarial perturbation error. For maximum of $q \geq 1$ adversarial workers among N workers results in a worst-case error of order $\mathcal { O } ( ( q / N ) ^ { 6 / 5 } )$ with appropriate selection of smoothness parameter, demonstrating that the approximation error vanishes asymptotically whenever the number of adversaries grows sublinearly with the network size. Experiments on scalar computation and neural-network inference are consistent with the predicted convergence behavior.

## Index Terms

Adversarial workers, approximate coded computing, distributed computing, general coded computing, reproducing kernel Hilbert space, smoothing spline.

## I. INTRODUCTION

The growing scale of data-intensive applications and computationally demanding machine-learning (ML) models has made distributed computation central to modern computing systems [1]. Such systems commonly operate under a master–worker architecture, in which a central master node assigns computational tasks to a collection of worker nodes. Each worker node performs its assigned computation and returns the result to the master node, which aggregates the worker outputs to recover the desired function evaluations. By executing these tasks in parallel, the system can substantially reduce the computation time incurred at the master node.

This benefit, however, comes with a reliability challenge: some workers may fail to return their results (known as straggler worker), while others may return incorrect values because of faults or malicious behavior (known as adversarial worker). These two forms of unreliability are fundamentally different. A straggler removes an observation, whereas an adversarial worker introduces a corrupted observation that may be coordinated with other corruptions and deliberately chosen to distort the final estimate. The master node must therefore reconstruct the desired function values without knowing which worker outputs, if any, are corrupted.

Coded computing addresses worker unreliability by introducing redundancy across the assigned computations. The master node encodes the original inputs into some coded inputs and send each coded inputs to one worker nodes. Next, each worker node apply the target function on its assigned coded input and return the results to the master node. Finally, the master node reconstructs the desired computations from the worker outputs [2]–[8] even though some of them are missing or incorrect.

For polynomial computations, Lagrange Coded Computing (LCC) achieves exact recovery while simultaneously providing resilience against stragglers and robustness against adversarial workers whenever the number of worker nodes exceeds a prescribed recovery threshold [4]. This threshold is determined by the degree of the target polynomial computation as well as the maximum numbers of straggling and adversarial nodes. However, LCC is originally formulated over finite fields and does not scale well with the input batch size due to numerical overflow and the severe ill-conditioning of high-degree polynomial interpolation [9], [10]. To address this limitation, Analog Lagrange Coded Computing (ALCC) extends LCC to infinite fields (the analog domain), preserving straggler resilience and dataset privacy [11].

Approximate coded computing extends coded computation to a broader class of functions by allowing approximate recovery. Berrut approximated coded computing (BACC) uses barycentric rational interpolation to approximate general nonlinear computations from the non-straggler worker outputs [9]. More recently, the learning-theoretic general coded computing (GCC) framework designs the encoder and decoder by deriving and minimizing an upper bound on the average end-to-end approximation error [12].

Although GCC provides an efficient scheme that outperforms BACC with a faster convergence rate and provably mitigates the effect of stragglers, its robustness against adversarial workers remains unexplored. Conversely, adversarial robustness has been studied within the Berrut framework. ApproxIFER combines rational interpolation with an algebraic error-location algorithm inspired by Berlekamp–Welch decoding to provide Byzantine-robust distributed inference [13]. However, these approaches address adversarial corruption through explicit error-location and correction procedures prior to reconstructing the desired outputs, which imposes substantial computational overhead at the master node. Furthermore, none of these works investigates how the approximation error behaves as the number of workers increases for a given adversarial budget, nor do they establish fundamental limits on the adversarial budget required for the approximation error to vanish asymptotically

To address these gaps, we investigate the adversarial robustness of the GCC framework from a complementary perspective: rather than explicitly locating and discarding adversarial workers, we mitigate their influence by inducing smoothness in the encoding and decoding procedures. Building upon the GCC architecture, the proposed method utilizes smooth functions to distribute computations and reconstruct the desired outputs. The decoder aggregates all returned outputs and bounds the impact of corrupted values via smoothness regularization. While the regularization parameters can be calibrated to a known adversarial budget, the framework operates without prior knowledge of either the adversaries’ identities or their specific corruptions.

Crucially, we establish that with an appropriately tuned smoothness regularization, the end-to-end approximation error converges to zero as the total number of workers increases, provided the number of adversarial workers grows sub-linearly with the network size.

Our main contributions are summarized as follows:

1) Adversarial GCC Framework: We propose an adversarially robust GCC scheme for general high-dimensional computations that operates reliably in the presence of adversarial workers. We derive an upper bound on the end-to-end approximation error and decompose it into three components: (i) the encoder approximation error, (ii) the decoder reconstruction error in a fully honest setting (referred to as the clean approximation error), and (iii) the perturbation caused by adversarial workers. This decomposition explicitly reveals how the smoothness of the encoder and decoder governs the robustness of the overall scheme.

2) Finite-Sample Guarantees and Convergence Rates: We derive a non-asymptotic upper bound on the worst-case mean-squared approximation error. This bound captures the fundamental trade-off introduced by decoder regularization: stronger smoothing reduces the influence of adversarial outputs but increases the clean approximation error. Balancing these two effects reveals that for a maximum of $q \geq 1$ adversarial workers out of N total workers, the decoder smoothing parameter should scale as $( q / N ) ^ { 8 / 5 }$ . With this tuning, the proposed scheme achieves an end-to-end error rate of $\mathcal { O } ( ( q / N ) ^ { 6 / 5 } )$ , demonstrating that the worst-case approximation error converges to zero whenever the number of adversarial workers grows sub-linearly with the total number of workers.

3) Efficient Implementation: We show that both the encoder and decoder can be implemented using vector-valued cubic smoothing splines. Because their coefficients can be computed by solving banded linear systems, the master-side encoding and decoding costs scale linearly with both the input batch size and the total number of workers. Notably, this computational efficiency is achieved without requiring a separate adversarial detection or error-location subroutine.

4) Empirical Evaluation: We evaluate the proposed scheme on both scalar nonlinear computations and highdimensional neural network inference. The experiments illustrate the predicted convergence behavior under adversarial attacks and confirm that the observed error decay aligns with our theoretical guarantees.

The remainder of the paper is organized as follows. Section II introduces the problem formulation. Section III reviews the necessary mathematical preliminaries regarding Sobolev spaces, reproducing kernels, and smoothing splines. Section IV develops the proposed encoding and decoding procedures, establishes their error and convergence guarantees, analyzes the computational complexity. The proofs of the main theorems and corollaries are collected in Section V, and Section VI presents the experiments result.

## A. Notation

For $n \in \mathbb { N } ,$ let $[ n ] : = \{ 1 , \dots , n \}$ . The cardinality of a set S is denoted by |S|. Scalars are denoted by lowercase letters, such as v, vectors by bold lowercase letters, such as v, and matrices by bold uppercase letters, such as V. The Euclidean norm is denoted by $\lVert \cdot \rVert _ { 2 }$ . For a vector $\mathbf { v } \in \mathbb { R } ^ { m }$ , its coordinatewise maximum norm is defined as $\| \mathbf { v } \| _ { \infty } : = \mathrm { m a x } _ { j \in [ m ] } | v _ { j } |$ . For a matrix A, its induced operator norm is defined as $\begin{array} { r } { \| \mathbf { A } \| _ { \mathrm { o p } } : = \operatorname* { s u p } _ { \| \mathbf { v } \| _ { 2 } = 1 } \| \mathbf { A } \mathbf { v } \| _ { 2 } } \end{array}$ . We write $a \asymp b$ to indicate that there exist positive constants $c _ { 1 } , c _ { 2 } > 0$ such that $c _ { 1 } a \leq b \leq c _ { 2 } a$

For a twice differentiable scalar-valued function $g : \mathbb { R } ^ { d }  \mathbb { R } , \nabla g ( \mathbf { x } )$ and $D ^ { 2 } g ( \mathbf { x } )$ denote its gradient and Hessian, respectively. For a twice differentiable vector-valued function $g = [ g _ { 1 } , \ldots , g _ { m } ] ^ { T } , \nabla g _ { j } ( \mathbf { x } )$ and $D ^ { 2 } g _ { j } ( \mathbf { x } )$ denote the gradient and Hessian of its jth component, respectively.

For a measurable function $u : \Omega \to \mathbb { R } ^ { M }$ , u<sup>′</sup> and $u ^ { \prime \prime }$ denote its first and second weak derivatives, taken componentwise with respect to its scalar argument. The formal definition of a weak derivative is provided in Definition 3 in Appendix A. The Lebesgue measure of a measurable set ${ \mathcal { E } } \subseteq \mathbb { R }$ is denoted by $\operatorname { L e b } ( \mathcal { E } )$ . For $1 \leq p \leq \infty , \| u \| _ { L ^ { p } ( \Omega ; \mathbb { R } ^ { M } ) }$ denotes its L<sup>p</sup>-norm over Ω, as formally defined in Definition 4 in Appendix A.

## II. PROBLEM FORMULATION

We consider a master-worker distributed computing system consisting of one master node and N worker nodes. The master node aims to evaluate a prescribed computing function $f : \mathbb { R } ^ { d }  \mathbb { R } ^ { m }$ on a collection of K input data points $\{ \mathbf { x } _ { k } \} _ { k = 1 } ^ { K }$ , using the worker nodes, where N, K, d, and m are positive integers.

The set of worker nodes is partitioned into two disjoint subsets: the set of honest workers, denoted by D, and the set of adversarial workers, denoted by B, where ${ \mathcal { D } } \cup B = [ N ] , { \mathcal { D } } \cap B = \emptyset$ . Each honest worker evaluates the prescribed function at its assigned input and returns the resulting output without modification. In contrast, an adversarial worker may return an arbitrary vector in the bounded adversarial-output region

$$
\begin{array} { r } { y _ { \mathrm { a d v } } : = [ - T _ { a } , T _ { a } ] ^ { m } , } \end{array}\tag{1}
$$

for some $M > 0 _ { ; }$ , regardless of the computation assigned to that worker. We assume that

$$
| B | \leq q ,\tag{2}
$$

where $q \in \{ 0 , \ldots , N \}$ is the maximum number of adversarial workers.

All adversarial workers are controlled by a single entity, referred to as the adversary, and may coordinate their returned outputs. An attack is specified by the set of corrupted workers and their returned vectors. Accordingly, define

$$
\begin{array} { r } { \mathcal { A } _ { q } : = \left\{ A = \left( \mathcal { B } , \{ \mathbf { a } _ { j } \} _ { j \in \mathcal { B } } \right) : \mathcal { B } \subseteq [ N ] , | \mathcal { B } | \leq q , \mathbf { a } _ { j } \in \mathcal { V } _ { \mathrm { a d v } } \right\} . } \end{array}\tag{3}
$$

The adversary has full knowledge of the input batch $\{ \mathbf { x } _ { k } \} _ { k = 1 } ^ { K }$ , the computing function $f ,$ and the complete encoding and decoding procedures. Thus, the adversary may choose the corrupted worker set and all corrupted outputs jointly.

## A. General Coded Computing in the Adversarial Setting

Similar to GCC in the straggler setting [12], the proposed framework consists of three stages all depicted in Figure 1: (i) encoding at the master node, (ii) computation at the worker nodes, and (iii) decoding at the master node.

(1) Encoding: The master node selects K fixed scalar points $\alpha _ { 1 } < \cdots < \alpha _ { K }$ with $\alpha _ { k } \in \Omega$ , for $k \in [ K ]$ referred to as the encoder design points, and constructs an encoder function $u _ { \mathrm { e n c } } : \Omega \to \mathbb { R } ^ { d }$ such that

$$
u _ { \mathrm { e n c } } ( \alpha _ { k } ) \approx { \bf x } _ { k } , \qquad k \in [ K ] .\tag{4}
$$

The master node also selects N fixed scalar points $\beta _ { 1 } < \cdots < \beta _ { N }$ , with $\beta _ { j } \in \Omega$ , for $j \in [ N ]$ referred to as the decoder design points. It forms the coded inputs by evaluating the encoder at these points:

$$
\begin{array} { r } { \widetilde { \mathbf { x } } _ { j } : = u _ { \mathrm { e n c } } ( \beta _ { j } ) , \qquad j \in [ N ] . } \end{array}\tag{5}
$$

The master node then assigns $\widetilde { \mathbf { x } } _ { j }$ to worker $j .$

(2) Computation: Every honest worker $j \in \mathcal { D }$ evaluates the prescribed computing function at its assigned coded input and returns $f ( \widetilde { \mathbf { x } } _ { j } ) = f ( u _ { \mathrm { e n c } } ( \beta _ { j } ) )$ . In contrast, an adversarial worker $j \in B$ may replace this value by an arbitrary vector $\mathbf { a } _ { j } \in \mathcal { V } _ { \mathrm { a d v } }$ . Therefore, under an attack $A \in { \mathcal { A } } _ { q } .$ , the output received by the master node from worker $j$ is

$$
\begin{array} { r } { \overline { { \mathbf { y } } } _ { j } ^ { A } = \left\{ \begin{array} { l l } { f ( u _ { \mathrm { e n c } } ( \beta _ { j } ) ) , } & { j \in \mathcal { D } , } \\ { \mathbf { a } _ { j } \in \mathcal { D } _ { \mathrm { a d v } } , } & { j \in \mathcal { B } . } \end{array} \right. } \end{array}\tag{6}
$$

(3) Decoding: Using the returned outputs $\{ \overline { { \mathbf { y } } } _ { j } ^ { A } \} _ { j = 1 } ^ { N }$ <sub>1</sub> and the corresponding decoder design points $\{ \beta _ { j } \} _ { j = 1 } ^ { N }$ , the master node constructs a decoder function $u _ { \mathrm { d e c } } : \Omega  \mathbb { R } ^ { m }$ . The decoder function is designed to suppress the effect of the adversarial outputs and accurately approximate the composed function $f \circ u _ { \mathrm { e n c } }$ in the presence of up to $q$ corrupted worker outputs. In particular, the objective is to have a decoder function $u _ { \mathrm { d e c } } ( \cdot )$ such that

$$
u _ { \mathrm { d e c } } ( z ) \approx f ( u _ { \mathrm { e n c } } ( z ) ) , \qquad z \in \Omega .\tag{7}
$$

The master node then estimates the desired computation at each original input point by evaluating the decoder at the corresponding encoder design point:

$$
\widehat f _ { q } ( \mathbf x _ { k } ) : = u _ { \sf d e c } ( \alpha _ { k } ) \overset { \mathrm { ( a ) } } \approx f ( u _ { \sf e n c } ( \alpha _ { k } ) ) \overset { \mathrm { ( b ) } } \approx f ( \mathbf x _ { k } ) , \qquad k \in [ K ] ,\tag{8}
$$

where approximation (a) follows from the decoder approximation in (7), while approximation (b) follows from the encoder approximation in (4) and the regularity of $f .$ These two sources of error will be formally quantified in the Section IV-A.

For a fixed computing function $f ,$ prescribed encoder design points $\pmb { \alpha } : = \{ \alpha _ { k } \} _ { k = 1 } ^ { K } ,$ prescribed decoder design points $\beta : = \{ \beta _ { j } \} _ { j = 1 } ^ { N }$ , and an adversarial budget $q ,$ we measure the performance of the GCC scheme by the worst-case mean squared error (MSE)

$$
\mathcal { L } _ { q } \big ( u _ { \mathrm { e n c } } , u _ { \mathrm { d e c } } ; f , \boldsymbol { \alpha } , \boldsymbol { \beta } \big ) : = \operatorname* { s u p } _ { A \in \mathcal { A } _ { q } } \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \lVert \widehat { f } _ { q } \big ( \mathbf { x } _ { k } \big ) - f \big ( \mathbf { x } _ { k } \big ) \rVert _ { 2 } ^ { 2 } .\tag{9}
$$

![](images/5c162461142404143b40d72235254d59a51359e2df5c81b6e8be90b7ba9915fb.jpg)  
Fig. 1: General coded computing in adversarial setting. Honest workers return $f ( \widetilde { \mathbf { x } } _ { j } )$ , whereas adversarial workers may return arbitrary vectors ${ \bf a } _ { j } \in \mathcal { V } _ { \mathrm { a d v } }$ . The adversary icons and dashed red arrows illustrate corrupted outputs. The adversarial workers’ identities are unknown to the master node.

Throughout our analysis, $\alpha ,$ and $\beta$ are fixed. We therefore suppress their dependence and write

$$
\mathcal { L } _ { q } ( u _ { \mathrm { e n c } } , u _ { \mathrm { d e c } } ; f ) : = \mathcal { L } _ { q } ( u _ { \mathrm { e n c } } , u _ { \mathrm { d e c } } ; f , \alpha , \beta ) .\tag{10}
$$

The supremum in (9) is taken over all admissible attacks that corrupt the outputs of at most $q$ workers. The encoder and decoder may be designed using the known budget $q ,$ but they do not know the realized set of adversarial workers or the corrupted values returned by them.

Accordingly, the objective of GCC in adversarial environment is to design the encoder and decoder functions to minimize the worst-case error in (9):

$$
\operatorname* { i n f } _ { u _ { \mathrm { e n c } } , u _ { \mathrm { d e c } } } \mathcal { L } _ { q } ( u _ { \mathrm { e n c } } , u _ { \mathrm { d e c } } ; f ) .\tag{11}
$$

## III. PRELIMINARIES

In this section, we introduce the notation and mathematical foundations required for the design and analysis of the proposed encoder and decoder.

Let $\Omega = [ a , b ] \subset \mathbb { R }$ denote a bounded closed interval. For $1 \leq p \leq \infty ,$ let $L ^ { p } \left( \Omega ; \mathbb { R } ^ { M } \right)$ denote the space of measurable functions on Ω with finite L<sup>p</sup>-norm.

## A. Sobolev Spaces

For an integer $r \geq 1$ and $1 \leq p \leq \infty$ , the Sobolev space $\mathbb { W } ^ { r , p } \left( \Omega ; \mathbb { R } ^ { M } \right)$ consists of functions $g \in L ^ { p } \left( \Omega ; \mathbb { R } ^ { M } \right)$ whose weak derivatives up to order r, denoted $g ^ { ( i ) }$ for $i = 1 , \ldots , r ,$ , also belong to $L ^ { p } \left( \Omega ; \mathbb { R } ^ { M } \right)$ . Setting $g ^ { ( 0 ) } : = g$ the norm on $\mathbb { W } ^ { r , p } \left( \Omega ; \mathbb { R } ^ { M } \right)$ is defined by

$$
\| g \| _ { \mathbb { W } ^ { r , p } ( \Omega ; \mathbb { R } ^ { M } ) } : = \bigg ( \sum _ { i = 0 } ^ { r } \| g ^ { ( i ) } \| _ { L ^ { p } ( \Omega ; \mathbb { R } ^ { M } ) } ^ { p } \bigg ) ^ { 1 / p } , \qquad 1 \le p < \infty ,\tag{12}
$$

and for $p = \infty$

$$
\| g \| _ { \mathbb { W } ^ { r , \infty } ( \Omega ; \mathbb { R } ^ { M } ) } : = \operatorname* { m a x } _ { 0 \leq i \leq r } \| g ^ { ( i ) } \| _ { L ^ { \infty } ( \Omega ; \mathbb { R } ^ { M } ) } .\tag{13}
$$

When $p = 2 , \mathbb { W } ^ { r , 2 } \left( \Omega ; \mathbb { R } ^ { M } \right)$ forms a Reproducing Kernel Hilbert Space (RKHS) [14], [15], which we denote by

$$
\mathcal { H } ^ { r } \left( \Omega ; \mathbb { R } ^ { M } \right) : = \mathbb { W } ^ { r , 2 } \left( \Omega ; \mathbb { R } ^ { M } \right) .\tag{14}
$$

Throughout the paper, $\mathcal { H } ^ { r } \left( \Omega ; \mathbb { R } ^ { M } \right)$ denote the r-th order Sobolev space over interval Ω.

1) Equivalent Norms: A Sobolev space can be equipped with several equivalent norms [14]–[16]. Two norms ∥·∥<sub>W</sub> and $\lVert \cdot \rVert _ { \mathbb { W } _ { 2 } }$ on the same space are equivalent if there exist constants $\theta _ { 1 } , \theta _ { 2 } > 0$ such that $\theta _ { 1 } \| g \| _ { \mathbb { W } _ { 1 } } \le \| g \| _ { \mathbb { W } _ { 2 } } \le \theta _ { 2 } \| g \| _ { \mathbb { W } _ { 1 } }$ for every function g in that space.

The equivalent norm used in this paper is the anchored Sobolev norm introduced in [17]. Let $\Omega = [ a , b ]$ and $1 \leq p < \infty .$ . For every $g \in \mathbb { W } ^ { r , p } \left( \Omega ; \mathbb { R } ^ { M } \right)$ , define

$$
\| g \| _ { \widetilde { \mathbb { W } } ^ { r , p } ( \Omega ; \mathbb { R } ^ { M } ) } : = \left( \sum _ { i = 0 } ^ { r - 1 } \| g ^ { ( i ) } ( a ) \| _ { 2 } ^ { p } + \| g ^ { ( r ) } \| _ { L ^ { p } ( \Omega ; \mathbb { R } ^ { M } ) } ^ { p } \right) ^ { 1 / p } .\tag{15}
$$

This norm is equivalent to the standard Sobolev norm in (12) [18]. In particular, since $\mathcal { H } ^ { r } \left( \Omega ; \mathbb { R } ^ { M } \right) = \mathbb { W } ^ { r , 2 } \left( \Omega ; \mathbb { R } ^ { M } \right)$ ， throughout the remainder of the paper we use the convention

$$
\Vert g \Vert _ { \mathcal { H } ^ { r } ( \Omega ; \mathbb { R } ^ { M } ) } : = \Vert g \Vert _ { \widetilde { \mathbb { W } } ^ { r , 2 } ( \Omega ; \mathbb { R } ^ { M } ) } , \qquad g \in \mathcal { H } ^ { r } \left( \Omega ; \mathbb { R } ^ { M } \right) .\tag{16}
$$

Remark 1. We define $\Omega = [ a , b ]$ as a closed interval; however, the underlying L<sup>p</sup> and Sobolev norms are identical whether defined over [a, b] or the open interval $( a , b )$ , as the boundary set {a, b} has Lebesgue measure zero. Thus, we have:

$$
\begin{array} { r l } & { \mathcal { H } ^ { r } ( [ a , b ] , \mathbb { R } ^ { M } ) = \mathcal { H } ^ { r } ( ( a , b ) , \mathbb { R } ^ { M } ) } \\ & { } \\ & { L ^ { r } ( [ a , b ] , \mathbb { R } ^ { M } ) = L ^ { r } ( ( a , b ) , \mathbb { R } ^ { M } ) } \end{array}\tag{17}
$$

## B. Representer Theorem

The representer theorem states that the solution to a broad class of regularized empirical risk minimization problems over an RKHS admits a finite-dimensional representation, expressed as a linear combination of kernel evaluations at the sampling points.

Theorem 1 (Representer Theorem [19], [20]). Let H be a vector-valued RKHS of functions from Ω to $\mathbb { R } ^ { d } ;$ , associated with a matrix-valued reproducing kernel Φ : $\Omega \times \Omega \to \mathbb { R } ^ { d \times d } .$ . Let $\{ t _ { i } \} _ { i = 1 } ^ { n } \subset \Omega , \lambda > 0 ,$ , and let $\Theta : [ 0 , \infty ) \to \mathbb { R }$ be a strictly increasing function. Consider the regularized optimization problem

$$
\operatorname* { m i n } _ { g \in \mathcal { H } } \Psi \big ( g ( t _ { 1 } ) , \ldots , g ( t _ { n } ) \big ) + \lambda \Theta \big ( \| g \| _ { \mathcal { H } } \big ) ,\tag{18}
$$

where $\Psi : ( \mathbb { R } ^ { d } ) ^ { n }  \mathbb { R } \cup \{ + \infty \}$ is an arbitrary loss function. If a minimizer of (18) exists, it admits the representation

$$
g ^ { * } ( t ) = \sum _ { i = 1 } ^ { n } \Phi ( t , t _ { i } ) \mathbf { w } _ { i } ,\tag{19}
$$

for some coefficient vectors $\mathbf { w } _ { 1 } , \ldots , \mathbf { w } _ { n } \in \mathbb { R } ^ { d } .$

When $\mathcal { H } = \mathcal { H } ^ { r } \left( \Omega ; \mathbb { R } ^ { d } \right)$ , the reproducing kernel decouples as $\Phi ( t , s ) = \phi ( t , s ) I _ { d }$ , where $\phi : \Omega \times \Omega  \mathbb { R }$ denotes the scalar reproducing kernel of $\mathcal { H } ^ { r } \left( \Omega ; \mathbb { R } \right)$ , and $I _ { d }$ is the $d \times d$ identity matrix [21], [22]. Under this setting, the representation (19) simplifies to

$$
g ^ { * } ( t ) = \sum _ { i = 1 } ^ { n } \phi ( t , t _ { i } ) \mathbf { w } _ { i } .\tag{20}
$$

## C. Second-Order Smoothing Splines

Let $a < t _ { 1 } < \cdot \cdot \cdot < t _ { n } < b$ , with $n \geq 2 .$ , denote an ordered set of sampling points, and let $\mathbf { y } : = ( \mathbf { y } _ { 1 } , \dots , \mathbf { y } _ { n } ) \in$ $( \mathbb { R } ^ { d } ) ^ { n }$ be the corresponding vector-valued observations. For a regularization parameter $\lambda > 0$ (known as smoothing parameter), consider the penalized least-squares problem over second-order Sobolev space:

$$
s _ { n , \lambda } \vert \mathbf { y } \vert : = \underset { g \in \mathcal { H } ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } { \mathrm { a r g m i n } } \left. \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Vert g ( t _ { i } ) - \mathbf { y } _ { i } \Vert _ { 2 } ^ { 2 } + \lambda \Vert g ^ { \prime \prime } \Vert _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 } \right. .\tag{21}
$$

Because both the quadratic empirical loss and the derivative penalty decouple across the $d$ coordinates, the minimization in (21) decomposes into d independent scalar problems. For each coordinate, the objective admits a unique minimizer given by a natural cubic smoothing spline (also referred to as a second-order smoothing spline) [15]. Consequently, (21) possesses a unique vector-valued solution $s _ { n , \lambda } [ g ]$

Moreover, the solution operator is linear in the observations and admits the equivalent kernel representation [23]

$$
s _ { n , \lambda } [ \mathbf { y } ] ( t ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } G _ { n , \lambda } ( t , t _ { i } ) \mathbf { y } _ { i } .\tag{22}
$$

The equivalent kernel (or weight function) $G _ { n , \lambda }$ depends solely on the design points $\{ t _ { i } \} _ { i = 1 } ^ { n }$ and the regularization parameter $\lambda ,$ and is identical across all coordinates.

When the observations are noiseless samples from a function $h : [ a , b ] \to \mathbb { R } ^ { d }$ , let h $\mathbf { \Psi } : = [ h ( t _ { 1 } ) , \ldots , h ( t _ { n } ) ]$ denote the corresponding observation vector. We then use the shorthand $s _ { n , \lambda } [ h ] : = s _ { n , \lambda } [ \mathbf { h } ]$

## IV. GCC IN ADVERSARIAL ENVIRONMENTS

In this section, we formulate the GCC framework under an adversarial setting in which, for a given $q \in \mathbb { N }$ , at most $q$ outputs returned by worker nodes are arbitrarily corrupted. We begin by characterizing the spacing of the design points, which plays a central role in our theoretical guarantees.

Definition 1 (Design Spacing Parameters). Let $\mathcal { T } : = \{ t _ { 1 } , \dots , t _ { n } \} \subset \Omega$ be an ordered set of design points satisfying $a \leq t _ { 1 } < \cdot \cdot \cdot < t _ { n } \leq b .$ The maximum and minimum consecutive spacing of $\tau$ are defined, respectively, by

$$
\bar { \delta } _ { \mathcal { T } } : = \operatorname* { m a x } _ { 1 \leq j < n } ( t _ { j + 1 } - t _ { j } ) ,\tag{23}
$$

$$
\underline { { \delta } } _ { T } : = \operatorname* { m i n } _ { 1 \leq j < n } ( t _ { j + 1 } - t _ { j } ) .\tag{24}
$$

For the remainder of this paper, we assume that both the encoder and decoder belong to the second-order vector-valued Sobolev space; that is,

$$
u _ { \mathrm { e n c } } \in \mathcal { H } ^ { 2 } \left( \Omega ; \mathbb { R } ^ { d } \right) , \qquad u _ { \mathrm { d e c } } \in \mathcal { H } ^ { 2 } \left( \Omega ; \mathbb { R } ^ { m } \right) .\tag{25}
$$

This regularity assumption imposes only mild smoothness constraints while providing the analytical tractability needed to leverage reproducing kernel techniques. In addition, without loss of generality, we set $\Omega = [ 0 , 1 ] . ^ { 1 }$

Next, we define the class of computing functions considered throughout this paper.

Definition 2 (Admissible Computing Function Class). We define $\mathcal { F } _ { \mathrm { s m } } ^ { d  m }$ as the class of twice continuously differentiable functions $f = [ f _ { 1 } , \ldots , f _ { m } ] ^ { \intercal } : \mathbb { R } ^ { d }  \mathbb { R } ^ { m }$ with bounded outputs, gradients, and Hessians:

$$
\begin{array} { r l } & { \mathcal { F } _ { \mathrm { s m } } ^ { d  m } : = \Big \{ f = [ f _ { 1 } , \dotsc , f _ { m } ] ^ { \top } \in C ^ { 2 } ( \mathbb { R } ^ { d } ; \mathbb { R } ^ { m } ) : \ \underset { \mathbf { x } \in \mathbb { R } ^ { d } } { \operatorname* { s u p } } \| f ( \mathbf { x } ) \| _ { \infty } < \infty , } \\ & { \qquad \quad \underset { \mathbf { x } \in \mathbb { R } ^ { d } } { \operatorname* { s u p } } \| \nabla f _ { j } ( \mathbf { x } ) \| _ { 2 } < \infty , \quad j \in [ m ] , } \\ & { \qquad \quad \underset { \mathbf { x } \in \mathbb { R } ^ { d } } { \operatorname* { s u p } } \| D ^ { 2 } f _ { j } ( \mathbf { x } ) \| _ { \mathrm { o p } } < \infty , \quad j \in [ m ] \Big \} . } \end{array}\tag{26}
$$

The function class $\mathcal { F } _ { \mathrm { s m } } ^ { d  m }$ is highly general and encompasses a wide variety of non-linear functions. For a given computing function $f \in \mathcal { F } _ { \mathrm { s m } } ^ { d  m }$ , our objective is to design the encoder and decoder functions to solve the robust risk minimization problem

$$
J _ { \mathrm { a d v } } : = \operatorname* { i n f } _ { \begin{array} { c } { u _ { \mathrm { e n c } } \in \mathcal { H } ^ { 2 } \left( \Omega ; \mathbb { R } ^ { d } \right) , } \\ { u _ { \mathrm { d e c } } \in \mathcal { H } ^ { 2 } \left( \Omega ; \mathbb { R } ^ { m } \right) } \end{array} } \mathcal { L } _ { q } ( u _ { \mathrm { e n c } } , u _ { \mathrm { d e c } } ; f ) .\tag{27}
$$

The remainder of this section is organized as follows. In Section IV-A, we present the encoder and decoder constructions and establish both non-asymptotic and asymptotic upper bounds on $\boldsymbol { J _ { \mathrm { { a d v } } } }$ . In Section IV-B, we analyze the computational complexity of the proposed scheme and compare it against existing baselines. Finally, in Section IV-C, we establish a complementary lower bound on the worst-case adversarial error of the proposed encoding and decoding design.

## A. Code Design

In this subsection, we develop the encoder and decoder constructions for the adversarial optimization problem in (27). To state the proposed design and the performance guarantees, we impose the following assumptions throughout this section:

(A1) Bounded computation outputs: The computing function $f : \mathbb { R } ^ { d }  \mathbb { R } ^ { m }$ satisfies

$$
\underset { \mathbf { x } \in \mathbb { R } ^ { d } } { \operatorname* { s u p } } \| f ( \mathbf { x } ) \| _ { \infty } \leq T _ { f }\tag{28}
$$

for some constant $T _ { f } > 0$

(A2) Regularity of the computation function: Let $f = [ f _ { 1 } , \dots , f _ { m } ] ^ { T }$ . For every $\mathbf { x } \in \mathbb { R } ^ { d }$ and $j \in [ m ]$ , assume that

$$
\| \nabla f _ { j } ( \mathbf x ) \| _ { 2 } \leq G _ { j } ,
$$

$$
\| D ^ { 2 } f _ { j } ( \mathbf x ) \| _ { \mathrm { o p } } \leq H _ { j } .\tag{29}
$$

We define the aggregate gradient and Hessian bounds as

$$
G _ { f } : = \left( \sum _ { j = 1 } ^ { m } G _ { j } ^ { 2 } \right) ^ { 1 / 2 } ,
$$

$$
H _ { f } : = \left( \sum _ { j = 1 } ^ { m } H _ { j } ^ { 2 } \right) ^ { 1 / 2 } ,\tag{30}
$$

and we assume that $G _ { f } , H _ { f } > 0 .$

(A3) Bounded adversarial outputs: Every corrupted output belongs to $[ - T _ { a } , T _ { a } ] ^ { m }$ for some $T _ { a } > 0$ . We define

$$
T : = \operatorname* { m a x } \{ T _ { a } , T _ { f } \} .\tag{31}
$$

Directly optimizing (27) is difficult because the encoder and decoder belong to infinite-dimensional function spaces and up to $q$ returned worker outputs may be arbitrarily corrupted. A central step in our approach is to construct an upper bound on $\mathcal { L } _ { q } ( u _ { \mathrm { e n c } } , u _ { \mathrm { d e c } } ; f )$ that decomposes the loss into three interconnected components.

The first component captures the encoder approximation error, which measures the discrepancy caused by approximating the original input data at the encoder design points, corresponding to (4). The remaining two components capture the decoder error, corresponding to (7), by separating the effect of adversarial corruptions from the clean generalization error of the decoder.

More formally, for any admissible attack $A \in { \mathcal { A } } _ { q }$ , we have

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \| \widehat { f } ( { \mathbf x } _ { k } ) - f ( { \mathbf x } _ { k } ) \| _ { 2 } ^ { 2 } } \\ & { = \displaystyle \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \| u _ { \mathrm { d e c } } ( \alpha _ { k } ) - f ( u _ { \mathrm { e n c } } ( \alpha _ { k } ) ) + f ( u _ { \mathrm { e n c } } ( \alpha _ { k } ) ) - f ( { \mathbf x } _ { k } ) \| _ { 2 } ^ { 2 } } \\ & { \displaystyle \le \displaystyle \frac { 2 } { K } \sum _ { k = 1 } ^ { K } \| f ( u _ { \mathrm { e n c } } ( \alpha _ { k } ) ) - f ( { \mathbf x } _ { k } ) \| _ { 2 } ^ { 2 } + \displaystyle \frac { 2 } { K } \sum _ { k = 1 } ^ { K } \| u _ { \mathrm { d e c } } ( \alpha _ { k } ) - f ( u _ { \mathrm { e n c } } ( \alpha _ { k } ) ) \| _ { 2 } ^ { 2 } , } \end{array}\tag{32}
$$

where the inequality follows from $\| \mathbf { u } + \mathbf { v } \| _ { 2 } ^ { 2 } \leq 2 \| \mathbf { u } \| _ { 2 } ^ { 2 } + 2 \| \mathbf { v } \| _ { 2 } ^ { 2 }$ . Taking the supremum over all admissible attacks yields

$$
\mathcal { L } _ { q } ( u _ { \mathrm { e n c } } , u _ { \mathrm { d e c } } ; f ) \leq \mathcal { L } _ { \mathrm { e n c } } + \mathcal { L } _ { \mathrm { d e c } } ,\tag{33}
$$

where

$$
\mathcal { L } _ { \mathrm { e n c } } : = \frac { 2 } { K } \sum _ { k = 1 } ^ { K } \lVert f ( u _ { \mathrm { e n c } } ( \alpha _ { k } ) ) - f ( \mathbf { x } _ { k } ) \rVert _ { 2 } ^ { 2 } ,\tag{34}
$$

$$
\mathcal { L } _ { \mathrm { d e c } } : = \operatorname* { s u p } _ { A \in \mathcal { A } _ { q } } \frac { 2 } { K } \sum _ { k = 1 } ^ { K } \lVert u _ { \mathrm { d e c } } ( \alpha _ { k } ) - f ( u _ { \mathrm { e n c } } ( \alpha _ { k } ) ) \rVert _ { 2 } ^ { 2 } .\tag{35}
$$

The encoder term ${ \mathcal { L } } _ { \mathrm { e n c } }$ is independent of the attack. In contrast, the value of ${ \mathcal { L } } _ { \mathrm { d e c } }$ is affected by the attack through the returned worker outputs used to construct $u _ { \mathrm { d e c } }$

To systematically manage the worst-case decoding error, we further decompose the decoder error. Let $u _ { \mathrm { d e c } } ^ { \circ }$ represent the clean decoder function that would be constructed if all worker nodes were honest, i.e., if the master node received $\overline { { \mathbf { y } } } _ { j } ^ { A } = f ( u _ { \mathrm { e n c } } ( \beta _ { j } ) )$ for all $j \in [ N ]$ . Thus, $u _ { \mathrm { d e c } } ^ { \circ }$ is a counterfactual function introduced only for the analysis. Adding and subtracting $u _ { \mathrm { d e c } } ^ { \circ } ( \alpha _ { k } )$ in (35) gives

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { d e c } } = \displaystyle \operatorname* { s u p } _ { A \in \mathcal { A } _ { q } } \frac { 2 } { K } \sum _ { k = 1 } ^ { K } \| u _ { \sf d e c } ( \alpha _ { k } ) - u _ { \sf d e c } ^ { \circ } ( \alpha _ { k } ) + u _ { \sf d e c } ^ { \circ } ( \alpha _ { k } ) - f ( u _ { \sf e n c } ( \alpha _ { k } ) ) \| _ { 2 } ^ { 2 } } \\ & { \qquad \leq 2 \displaystyle \operatorname* { s u p } _ { A \in \mathcal { A } _ { q } } \operatorname* { s u p } _ { z \in \Omega } \| u _ { \sf d e c } ( z ) - u _ { \sf d e c } ^ { \circ } ( z ) + u _ { \sf d e c } ^ { \circ } ( z ) - f ( u _ { \sf e n c } ( z ) ) \| _ { 2 } ^ { 2 } } \\ & { \qquad \leq 4 \displaystyle \operatorname* { s u p } _ { A \in \mathcal { A } _ { q } } \operatorname* { s u p } _ { z \in \Omega } \| u _ { \sf d e c } ( z ) - u _ { \sf d e c } ^ { \circ } ( z ) \| _ { 2 } ^ { 2 } + 4 \displaystyle \operatorname* { s u p } _ { z \in \Omega } \| u _ { \sf d e c } ^ { \circ } ( z ) - f ( u _ { \sf e n c } ( z ) ) \| _ { 2 } ^ { 2 } . } \\ & { \qquad \mathcal { L } _ { \sf d e c } ^ { \sf d i v } } \end{array}\tag{36}
$$

Consequently,

$$
\mathcal { L } _ { q } ( u _ { \mathrm { e n c } } , u _ { \mathrm { d e c } } ; f ) \leq \mathcal { L } _ { \mathrm { e n c } } + \mathcal { L } _ { \mathrm { d e c } } ^ { \mathrm { g e n } } + \mathcal { L } _ { \mathrm { d e c } } ^ { \mathrm { a d v } } .\tag{37}
$$

Remark 2 (Interpretation of the Error Terms). The term ${ \mathcal { L } } _ { \mathrm { e n c } }$ represents the encoder approximation error, measuring the average distortion at the encoder design points $\{ \alpha _ { i } \} _ { i = 1 } ^ { K }$ after applying $f .$ The term ${ \mathcal { L } } _ { \mathrm { d e c } } ^ { \mathrm { a d v } }$ captures the adversarial perturbation error, defined as the worst-case deviation in the decoder output induced by up to q corrupted worker transmissions. Finally, $\mathcal { L } _ { \mathrm { d e c } } ^ { \mathrm { g e n } }$ represents the clean decoder generalization error, which reflects the reconstruction error of the composite map $z \mapsto f ( u _ { \mathrm { e n c } } ( z ) )$ in the benign setting where all worker outputs are uncorrupted.

Remark 3 (Connection to Generalization in Learning Theory). The term generalization error is borrowed from learning theory, where it measures how well a model trained on one dataset performs on new, unseen test data [24]. In our setting, the decoder is constructed based on the decoder design points $\{ \beta _ { j } \} _ { j = 1 } ^ { N }$ , whereas $\mathcal { L } _ { \mathrm { d e c } } ^ { \mathrm { g e n } }$ measures its performance at the distinct encoder design points $\{ \alpha _ { i } \} _ { i = 1 } ^ { K }$

Remark 4 (Coupling through the encoder). Although the three terms in (37) have different interpretations, they are not independent. The encoder term depends directly on the values $u _ { \mathrm { e n c } } ( \alpha _ { k } )$ , while both decoder terms depend on the encoder through the composite function $f \circ u _ { \mathrm { e n c } }$

We address this coupling through a nested construction. For a fixed encoder, we first design the decoder and derive bounds on $\mathcal { L } _ { \mathrm { d e c } } ^ { \mathrm { a d v } } + \mathcal { L } _ { \mathrm { d e c } } ^ { \mathrm { g e n } }$ . Substituting these bounds into (37) yields an encoder-only objective, which is then minimized in an outer optimization step.

1) Decoder Design: As part of the code design, we select the decoder design points to be equidistant over $\Omega = [ 0 , 1 ]$

$$
\beta _ { j } : = \frac { j - 1 } { N - 1 } , \qquad j \in [ N ] .\tag{38}
$$

This choice will help us in our subsequent theoretical results.

Fix an encoder $u _ { \mathrm { e n c } } \in \mathcal { H } ^ { 2 } \left( \Omega ; \mathbb { R } ^ { d } \right)$ . The master node receives the outputs $\{ \overline { { \mathbf { y } } } _ { j } ^ { A } \} _ { j = 1 } ^ { N }$ defined in (6) without knowing which, if any, of the returned outputs are corrupted. We construct the decoder using a regularized smoothing-spline objective. Specifically, the decoder is defined as:

$$
u _ { \mathrm { d e c } } ^ { * } : = \operatorname * { a r g m i n } _ { u \in \mathcal { H } ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } \left\{ \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \lVert u ( \beta _ { j } ) - \overline { { \mathbf { y } } } _ { j } ^ { A } \rVert _ { 2 } ^ { 2 } + \lambda _ { \mathrm { d } } \lVert u ^ { \prime \prime } \rVert _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } ^ { 2 } \right\} ,\tag{39}
$$

where $\lambda _ { \mathrm { d } } > 0$ is the decoder smoothing parameter. The smoothing parameter $\lambda _ { \mathrm { d } }$ may depend on the total number of workers N, the adversarial budget $q ,$ and the decoder evaluation points $\{ \beta _ { j } \} _ { j = 1 } ^ { N }$ , and is selected prior to observing the worker transmissions.

The first term in (39) measures the empirical squared error over all returned worker outputs, while the second term penalizes the curvature of the decoder. The smoothing parameter $\lambda _ { \mathrm { d } }$ balances these two objectives: larger values of $\lambda _ { \mathrm { d } }$ yield a smoother decoder with lower curvature, whereas smaller values prioritize fitting the empirical data, pushing the decoder toward interpolating the received outputs $\{ \overline { { \mathbf { y } } } _ { j } ^ { A } \} _ { j = 1 } ^ { N }$ . By penalizing rapid variations, this roughness penalty reduces sensitivity to localized anomalies and limits the impact of adversarial corruptions.

Based on decoder function in (39), its corresponding clean decoder has the following form:

$$
u _ { \mathrm { d e c } } ^ { \circ } : = \underset { u \in \mathcal { H } ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } { \arg \operatorname* { m i n } } \left\{ \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \lVert u ( \beta _ { j } ) - f \big ( u _ { \mathrm { e n c } } ( \beta _ { j } ) \big ) \rVert _ { 2 } ^ { 2 } + \lambda _ { \mathrm { d } } \lVert u ^ { \prime \prime } \rVert _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } ^ { 2 } \right\} .\tag{40}
$$

The clean decoder which is purely an analytical tool used in our analysis, corresponds to the exact same optimization, assuming that all worker nodes are honest and compute and transmit uncorrupted evaluations.

As established in Section III-C, the separability of the squared Euclidean norm across coordinate dimensions decouples (39) and (40) into m independent scalar smoothing-spline variational problems. Consequently, both $u _ { \mathrm { d e c } }$ and $u _ { \mathrm { d e c } } ^ { \circ }$ are unique vector-valued natural cubic smoothing splines. Specifically, the decoder admits the representer decomposition

$$
u _ { \mathrm { d e c } } ^ { * } ( t ) = \mathbf { d } _ { 0 } + \mathbf { d } _ { 1 } t + \sum _ { j = 1 } ^ { N } \mathbf { c } _ { j } \phi _ { 0 } ( t , \beta _ { j } ) ,\tag{41}
$$

where ${ \bf d } _ { 0 } , { \bf d } _ { 1 } , { \bf c } _ { 1 } , \ldots , { \bf c } _ { N } \in \mathbb { R } ^ { m }$ , and $\phi _ { 0 }$ denotes the reproducing kernel associated with the space $\mathcal { H } _ { 0 } ^ { 2 } \left( \Omega ; \mathbb { R } \right)$ , which is the second-order Sobolev space with compact support (see Definition 6 and Remark 7 in Appendix A for its explicit formulation). The coefficients are obtained by solving the standard linear system for vector-valued cubic smoothing splines [15], [23].

Theorem 2 (Decoder error bound). Consider the GCC scheme with N worker nodes and the equidistant decoder design points in (38). Suppose that Assumptions A1 through A3 hold. Then there exist constants $C _ { 1 } , C _ { 2 } , c _ { 0 } > 0$ independent of $N , \ q , \lambda _ { d } ,$ and $u _ { e n c }$ , such that, whenever $0 < \lambda _ { d } \leq 1$ and $N \lambda _ { d } ^ { 1 / 4 } \geq c _ { 0 }$ , we have

$$
\operatorname* { i n f } _ { \substack { u _ { d e } \in \mathcal { H } ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } } \mathcal { L } _ { \mathrm { d e c } } ^ { \mathrm { g e n } } + \mathcal { L } _ { \mathrm { d e c } } ^ { \mathrm { a d v } } \leq C _ { 1 } \operatorname* { m a x } \{ H _ { f } ^ { 2 } , G _ { f } ^ { 2 } \} ( \lambda _ { d } ^ { 3 / 4 } + N ^ { - 3 } ) \psi ( \| u _ { e n c } \| _ { \mathcal { H } ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 } ) + C _ { 2 } m T ^ { 2 } \frac { q ^ { 2 } } { N ^ { 2 } } \lambda _ { d } ^ { - 1 / 2 } ,\tag{42}
$$

where $\psi ( t ) : = t + 4 t ^ { 2 }$

Remark 5 (Trade-off in the Decoder Error Bound). The first term in (42) upper-bounds the clean decoder generalization error. For a fixed number of workers $N ,$ this term decreases monotonically as $\lambda _ { \mathrm { d } }$ decreases. It also scales with the regularity of the computation pipeline, depending on the smoothness parameters $H _ { f }$ and $G _ { f }$ of the target function, as well as the encoder Sobolev norm $\| u _ { \mathrm { e n c } } \| _ { \mathcal { H } ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 }$ which means smoother functions and encoders yield a strictly smaller bound. The second term bounds the adversarial perturbation error. In contrast to the first term, it grows as $\lambda _ { \mathrm { d } }$ decreases or as the adversary’s leverage increases through the corruption ratio $T q / N$ . Consequently, the regularization parameter $\lambda _ { \mathrm { d } }$ controls a fundamental trade-off between clean reconstruction accuracy and robustness to adversarial corruptions.

Remark 6 (Necessity of Positive Decoder Smoothing). In the straggler setting, taking the interpolating limit $\lambda _ { \mathrm { d } } \to 0$ yields an effective parameter-free decoder [12]. In the adversarial regime, however, interpolation is catastrophic, as interpolating arbitrary corruptions severely distorts the reconstructed function. The factor $\lambda _ { \mathrm { d } } ^ { - 1 / 2 }$ in (42) captures this instability explicitly, suggesting why strictly positive regularization $( \lambda _ { \mathrm { d } } > 0 )$ is essential in the presence of malicious worker nodes.

Corollary 1 (Optimal scaling of the decoder smoothing parameter). Assume that the conditions of Theorem 2 hold. Suppose that $q \geq 1$ . Then, for all sufficiently large N, the value of $\lambda _ { d }$ that minimizes the decoder-error upper bound in (42) satisfies

$$
\lambda _ { d } ^ { \star } \asymp \left( \frac { q } { N } \right) ^ { 8 / 5 } ,\tag{43}
$$

where the implicit constants may depend on the fixed encoder and the constant parameters of the scheme, but are independent of N and $q .$

2) Encoder Design: We next substitute the decoder bound into (37) to obtain an encoder-only objective. Assumption A2 implies that $f$ is G<sub>f</sub>-Lipschitz:

$$
\| f ( \mathbf { x } ) - f ( { \widetilde { \mathbf { x } } } ) \| _ { 2 } \leq G _ { f } \| \mathbf { x } - { \widetilde { \mathbf { x } } } \| _ { 2 } .\tag{44}
$$

Therefore,

$$
\begin{array} { r l r } {  { \mathcal { L } _ { \mathrm { e n c } } = \frac { 2 } { K } \sum _ { k = 1 } ^ { K } \lVert f ( u _ { \mathrm { e n c } } ( \alpha _ { k } ) ) - f ( { \mathbf x } _ { k } ) \rVert _ { 2 } ^ { 2 } } } \\ & { } & { \leq \frac { 2 G _ { f } ^ { 2 } } { K } \sum _ { k = 1 } ^ { K } \lVert u _ { \mathrm { e n c } } ( \alpha _ { k } ) - { \mathbf x } _ { k } \rVert _ { 2 } ^ { 2 } . ~ } \end{array}\tag{45}
$$

Combining (37), (45), and Theorem 2 gives

$$
\begin{array} { r l } & { \displaystyle \mathcal { L } _ { q } ( u _ { \mathrm { e n c } } , u _ { \mathrm { d e c } } ^ { * } ; f ) \leq \frac { 2 G _ { f } ^ { 2 } } { K } \sum _ { k = 1 } ^ { K } \| u _ { \mathrm { e n c } } ( \alpha _ { k } ) - \mathbf { x } _ { k } \| _ { 2 } ^ { 2 } } \\ & { \qquad + C _ { 1 } \operatorname* { m a x } \{ H _ { f } ^ { 2 } , G _ { f } ^ { 2 } \} \left( \lambda _ { \mathrm { d } } ^ { 3 / 4 } + N ^ { - 3 } \right) \psi \left( \| u _ { \mathrm { e n c } } \| _ { \mathcal { H } ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 } \right) } \\ & { \qquad + C _ { 2 } m T ^ { 2 } \frac { q ^ { 2 } } { N ^ { 2 } } \lambda _ { \mathrm { d } } ^ { - 1 / 2 } . } \end{array}\tag{46}
$$

The last term in (46) is independent of the encoder. Assuming $G _ { f } > 0$ , define

$$
\Gamma _ { q , N } : = \frac { C _ { 1 } } { 2 G _ { f } ^ { 2 } } \operatorname* { m a x } \{ H _ { f } ^ { 2 } , G _ { f } ^ { 2 } \} \left( \lambda _ { \mathrm { d } } ^ { 3 / 4 } + N ^ { - 3 } \right) .\tag{47}
$$

The dependence of $\Gamma _ { q , N }$ on the adversarial budget $q$ is through the choice of the decoder smoothing parameter $\lambda _ { \mathrm { d } } .$ Ignoring the additive term that is independent of the encoder, we choose the encoder as

$$
u _ { \mathrm { e n c } } ^ { * } : = \operatorname * { a r g m i n } _ { u \in \mathcal { H } ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } \left\{ \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \lVert u ( \alpha _ { k } ) - \mathbf { x } _ { k } \rVert _ { 2 } ^ { 2 } + \Gamma _ { q , N } \psi \left( \lVert u \rVert _ { \mathcal { H } ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 } \right) \right\} .\tag{48}
$$

Since $\psi$ is strictly increasing, the representer theorem stated in Theorem 1 implies that every minimizer of (48) admits the finite-dimensional representation

$$
u _ { \mathrm { e n c } } ^ { * } ( t ) = \sum _ { k = 1 } ^ { K } \phi ( t , \alpha _ { k } ) { \bf z } _ { k } ,\tag{49}
$$

for some coefficient vectors $\mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { K } \in \mathbb { R } ^ { d }$ , where $\phi$ is the scalar reproducing kernel of $\mathcal { H } ^ { 2 } \left( \Omega ; \mathbb { R } \right)$ given by $\phi ( t , s ) = 1 + t s + \phi _ { 0 } ( t , s )$ , with $\phi _ { 0 }$ defined in Remark 7. Thus, the infinite-dimensional encoder design reduces to a finite-dimensional optimization over the coefficient vectors $\{ \mathbf { z } _ { k } \} _ { k = 1 } ^ { K }$

The encoder objective in (48) contains the nonlinear regularization term $\psi \big ( \lVert u \rVert _ { \mathcal { H } ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 } \big )$ , where $\psi ( t ) = t + 4 t ^ { 2 }$ Although (49) reduces the problem to a finite-dimensional optimization, directly computing its coefficient vectors can still be computationally demanding. As in the straggler setting studied in [12], the encoder can instead be constructed through a standard cubic smoothing-spline problem. The following theorem describes this reduction for the adversarial setting.

For convenience, define the encoder-independent adversarial term

$$
\Delta _ { q , N } : = C _ { 2 } m T ^ { 2 } \frac { q ^ { 2 } } { N ^ { 2 } } \lambda _ { \mathrm { d } } ^ { - 1 / 2 } .\tag{50}
$$

Theorem 3 (Efficient encoder construction). Suppose that Assumptions A1 through A3 and the conditions of Theorem 2 hold. There exist constants $m _ { 1 } , m _ { 2 } > 0$ and $R _ { \mathrm { e } } = R _ { \mathrm { e } } \left( \{ \alpha _ { k } \} _ { k = 1 } ^ { K } , \{ \mathbf { x } _ { k } \} _ { k = 1 } ^ { K } \right) > 0$ such that

$$
J _ { a d v } \leq \Delta _ { q , N } + 2 G _ { f } ^ { 2 } \Big [ \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \lVert u _ { e n c } ^ { \dagger } ( \alpha _ { k } ) - \mathbf { x } _ { k } \rVert _ { 2 } ^ { 2 } + \lambda _ { e } \left( R _ { \mathrm { e } } + \lVert ( u _ { e n c } ^ { \dagger } ) ^ { \prime \prime } \rVert _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 } \right) \Big ] ,\tag{51}
$$

where the encoder smoothing parameter is

$$
\lambda _ { e } : = \Gamma _ { q , N } \left( m _ { 1 } + m _ { 2 } R _ { \mathrm { e } } \right) ,\tag{52}
$$

and $u _ { e n c } ^ { \dagger }$ is a minimizer of

$$
u _ { e n c } ^ { \dagger } : = \underset { u \in \mathcal { H } ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } { \arg \operatorname* { m i n } } \left\{ \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \lVert u ( \alpha _ { k } ) - \mathbf { x } _ { k } \rVert _ { 2 } ^ { 2 } + \lambda _ { e } \lVert u ^ { \prime \prime } \rVert _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 } \right\} .\tag{53}
$$

The proof follows the same argument used for the encoder construction in the straggler setting [12]. The only changes are the regularization coefficient $\Gamma _ { q , N }$ and the additive term $\Delta _ { q , N }$ , which is independent of the encoder and therefore does not affect the encoder optimization.

The squared Euclidean loss and the second-derivative penalty in (53) decouple across the d input coordinates. Consequently, $u _ { \mathrm { e n c } } ^ { \dagger }$ is a vector-valued natural cubic smoothing spline and admits the representation

$$
u _ { \mathrm { e n c } } ^ { \dagger } ( t ) = \mathbf { a } _ { 0 } + \mathbf { a } _ { 1 } t + \sum _ { k = 1 } ^ { K } \mathbf { b } _ { k } \phi _ { 0 } ( t , \alpha _ { k } ) ,\tag{54}
$$

Algorithm 1 General Coded Computing in the Adversarial Setting   
Require: Input batch $\{ \mathbf { x } _ { k } \} _ { k = 1 } ^ { K }$ , target function $f ,$ number of workers $N _ { \ast }$ , adversarial budget $q ,$ encoder design   
points $\{ \alpha _ { k } \} _ { k = 1 } ^ { K }$ , and smoothing parameters $\lambda _ { \mathrm { e } }$ and $\lambda _ { \mathrm { d } } .$   
Ensure: Estimates $\{ \widehat { f } ( \mathbf { x } _ { k } ) \} _ { k = 1 } ^ { K }$   
1: The master node selects the equidistant decoder design points $\begin{array} { r } { \beta _ { j } = \frac { j - 1 } { N - 1 } } \end{array}$ for $j \in [ N ]$   
2: The master node constructs the encoder $u _ { \mathrm { e n c } } ^ { \dagger }$ by solving   
$u _ { \mathrm { e n c } } ^ { \dagger } = \operatorname * { a r g m i n } _ { u \in \mathcal { H } ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } \left\{ \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \lVert u ( \alpha _ { k } ) - \mathbf { x } _ { k } \rVert _ { 2 } ^ { 2 } + \lambda _ { \mathrm { e } } \lVert u ^ { \prime \prime } \rVert _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 } \right\} .$   
which admits the vector-valued cubic smoothing-spline representation in (54).   
3: The master node computes the coded inputs $\widetilde { \mathbf { x } } _ { j } = u _ { \mathrm { e n c } } ^ { \dagger } ( \beta _ { j } )$ for $j \in [ N ]$ , and sends $\widetilde { \mathbf { x } } _ { j }$ to worker $j .$   
4: Each honest worker $j$ computes and returns $f ( \widetilde { \mathbf { x } } _ { j } ) = f ( u _ { \mathrm { e n c } } ^ { \dagger } ( \beta _ { j } ) )$ . An adversarial worker may instead return an   
arbitrary vector $\mathbf { a } _ { j } \in [ - T _ { a } , T _ { a } ] ^ { m }$   
5: The master node receive $\{ \overline { { \mathbf { y } } } _ { j } ^ { A } \} _ { j = 1 } ^ { N }$ and constructs the decoder from all returned outputs by solving   
$u _ { \mathrm { d e c } } ^ { * } = \operatorname * { a r g m i n } _ { u \in \mathcal { H } ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } \left\{ \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \lVert u ( \beta _ { j } ) - \overline { { \mathbf { y } } } _ { j } ^ { A } \rVert _ { 2 } ^ { 2 } + \lambda _ { \mathrm { d } } \lVert u ^ { \prime \prime } \rVert _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } ^ { 2 } \right\} \mathrm { . ~ }$   
which admits the vector-valued cubic smoothing-spline representation in (41).   
6: The master node outputs ${ \widehat { f } } ( \mathbf { x } _ { k } ) = u _ { \mathrm { d e c } } ^ { * } ( \alpha _ { k } )$ for $k \in [ K ]$

where $\mathbf { a } _ { 0 } , \mathbf { a } _ { 1 } , \mathbf { b } _ { 1 } , \dots , \mathbf { b } _ { K } \in \mathbb { R } ^ { d }$ , and $\phi _ { 0 } ( \cdot )$ defined in Remark 7. Therefore, the encoder coefficients can be computed efficiently by solving the standard linear system for vector-valued cubic smoothing splines with smoothing parameter $\lambda _ { \mathrm { e } } .$

3) GCC Scheme: The preceding constructions define the GCC scheme in the adversarial setting. The master node first constructs the encoder using (48) (or using (53) for efficient encoder), evaluates it at the equidistant decoder design points in (38), and sends the resulting coded inputs to the workers. After receiving all outputs from worker nodes, the master node constructs the decoder in (39) using all returned outputs without attempting to identify the adversarial workers. using all returned outputs, without attempting to identify the adversarial workers. Finally, it computes the estimates ${ \widehat { f } } ( \mathbf { x } _ { k } ) = u _ { \mathrm { d e c } } ( \alpha _ { k } )$ for $k \in [ K ]$ . Algorithm 1 summarizes the efficient implementation of the proposed scheme.

In terms of asymptotic analysis, the following corollary establishes the asymptotic convergence rate of the robust end-to-end objective as the worker nodes increases, demonstrating that the adversarial risk in (27) asymptotically vanishes whenever the adversarial budget satisfies $q / N \to 0$ as $N \to \infty$

Corollary 2 (Asymptotic Convergence Rate). Under the assumptions of Theorem $^ { 2 , }$ suppose the adversarial budget satisfies $1 \leq q \leq C _ { q } N ^ { \theta }$ for constants $C _ { q } ~ > ~ 0$ and $\theta \in [ 0 , 1 )$ . If the decoder smoothing parameter is chosen according to Corollary 1 such that $\lambda _ { d } \asymp ( q / N ) ^ { 8 / 5 }$ , then for sufficiently large $N ,$ , the end-to-end objective satisfies

$$
J _ { a d \nu } = \mathcal { O } \left( N ^ { \frac { 6 } { 5 } ( \theta - 1 ) } \right)\tag{55}
$$

## B. Computational Complexity

The computational complexity of the encoder and decoder constructions closely follows the analysis developed for the straggler setting in [12]. We therefore summarize the main results and highlight the difference introduced by the adversarial model.

For the nonlinear encoder in (48), the finite-dimensional kernel coefficients can be computed using an iterative optimization method such as limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) [25]. For L-BFGC solver, If the solver performs I iterations, constructing the encoder and evaluating it at the N decoder design points require

$$
\mathcal { O } ( I K ^ { 2 } d + N K d )\tag{56}
$$

operations. In contrast, the efficient encoder in (53) is a vector-valued cubic smoothing spline. Using a B-spline implementation with a banded linear system, it can be constructed and evaluated at the decoder design points in

$$
\mathcal { O } \big ( ( K + N ) d \big )\tag{57}
$$

operations.

The decoder in (39) has the same smoothing-spline structure. Unlike the straggler setting, all N worker nodes return their results, and hence the decoder is constructed using all N received outputs. Constructing the decoder and evaluating it at the K encoder design points require

$$
\mathcal { O } \big ( ( N + K ) m \big )\tag{58}
$$

operations. Consequently, with the efficient encoder, the total master-side computational complexity of the adversarial GCC scheme is

$$
\mathcal O \big ( ( K + N ) d \big ) + \mathcal O \big ( ( N + K ) m \big ) .\tag{59}
$$

In the common regime $N \geq K$ , this reduces to $\mathcal { O } \left( N ( d + m ) \right)$

1) Comparison with Existing Schemes: We compare the computational complexity of different coded computing schemes at the master node, excluding worker-side evaluations of the target function.

Lagrange Coded Computing (LCC) relies on polynomial interpolation and fast multipoint evaluation [4], [26], yielding encoding and decoding complexities of $\mathcal { O } ( N \log ^ { 2 } K$ log log $K \cdot d )$ and $\mathcal { O } ( N \log ^ { 2 }$ N log log $N \cdot m )$ respectively. In the presence of adversarial workers, LCC requires an explicit error-localization step to identify corrupted outputs. By leveraging sub-quadratic decoding routines for Reed–Solomon codes [27], [28], locating these errors incurs an additional complexity of $\mathcal { O } ( q \log ^ { 2 } q \log \log q \cdot m )$ . Consequently, in addition to the overhead of error localization, LCC exhibits superlinear computational complexity with respect to the number of worker nodes N. Overall, for an input batch size K and at most $q \leq N$ adversaries, the total computational complexity at the master node is

$$
\begin{array} { r l } & { \mathcal { O } \big ( N \log ^ { 2 } K \log \log K \cdot d \big ) + \mathcal { O } \big ( N \log ^ { 2 } N \log \log N \cdot m \big ) + \mathcal { O } \big ( q \log ^ { 2 } q \log \log q \cdot m \big ) } \\ & { \quad = \mathcal { O } \big ( N \log ^ { 2 } K \log \log K \cdot d + N \log ^ { 2 } N \log \log N \cdot m \big ) . } \end{array}\tag{60}
$$

Berrut Approximated Coded Computing (BACC) relies on barycentric rational interpolation [9], where direct evaluation yields encoding and decoding complexities of $\mathcal { O } ( K N d )$ and $\mathcal { O } ( K N m )$ , respectively. Because BACC alone cannot identify corrupted worker outputs, ApproxIFER [13] extends it to the adversarial setting by adapting the Berlekamp–Welch approach [28] to rational interpolation. Specifically, before performing the final rational reconstruction, ApproxIFER constructs candidate error-locator functions by solving linear systems of size $\mathcal { O } ( N )$ across the m output coordinates and combining the coordinatewise estimates to filter out suspected adversarial reports. Although [13] does not explicitly analyze the asymptotic cost of this error-localization step, solving these $\mathcal { O } ( N )$ -dimensional linear systems requires $\mathcal { O } ( m N ^ { 3 } )$ operations under standard dense solvers.

In contrast, our proposed GCC framework requires only $\mathcal { O } ( ( K + N ) d )$ and $\mathcal { O } ( ( K + N ) m )$ arithmetic operations for encoding and decoding, respectively, yielding a total master-node complexity of

$$
\mathcal { O } ( ( K + N ) ( d + m ) ) .\tag{61}
$$

Compared with BACC-based schemes, GCC reduces the multiplicative $\mathcal { O } ( K N )$ dependence on the input batch size K and worker count N to a purely additive $\mathcal { O } ( K + N )$ scaling. Furthermore, unlike LCC, Robust ALCC, ApproxIFER, and RBACC, GCC mitigates corrupted worker reports inherently through spline regularization at the decoder, completely eliminating the need for dedicated error-detection or error-localization stages. Consequently, the computational complexity of GCC remains strictly linear in $K + N$ and is entirely independent of the adversarial budget q.

## C. Impossibility Result for Smoothing-Spline Decoding

Corollary 2 establishes vanishing worst-case error for the proposed scheme when the adversarial budget grows sublinearly with the number of workers $( q = o ( N ) )$ . In the following theorem, we show that this sublinear scaling is necessary for the smoothing-spline decoder in (39) to achieve vanishing adversarial loss.

Theorem 4 (Adversarial Error Lower Bound). Consider a GCC scheme with $N \geq 2 , K \geq 1$ , and an adversarial budget $1 \leq q < N .$ . For any encoder $u _ { e n c } \in \mathcal { H } ^ { 2 } \left( \Omega ; \mathbb { R } ^ { d } \right)$ , let $\boldsymbol { u } _ { d e c } ^ { * }$ denote the smoothing-spline decoder defined in (39) with smoothing parameter $\lambda _ { d } > 0 .$ . Under Assumption (A3) with $T _ { a } > 0 ,$ , for every computing function $f \in \mathcal { F } _ { \mathrm { s m } } ^ { d  m }$ we have

$$
\operatorname* { i n f } _ { \substack { u _ { e n c } \in \mathcal { H } ^ { 2 } \left( \Omega ; \mathbb { R } ^ { d } \right) , } } \mathcal { L } _ { q } ( u _ { e n c } , u _ { d e c } ^ { * } ; f ) \geq \frac { T _ { a } ^ { 2 } } { K } \left( \frac { q } { N } \right) ^ { 2 } .\tag{62}
$$

Corollary 3 (Positive Adversarial Fraction). Fix $K \geq 1 , \mu \in ( 0 , 1 )$ , and $T _ { a } > 0$ , and let $q = \lfloor \mu N \rfloor$ . Then, for every computing function $f \in \mathcal { F } _ { \mathrm { s m } } ^ { d  m }$

$$
\operatorname* { l i m i n f } _ { N \to \infty } \operatorname* { i n f } _ { \substack { u _ { e n c } \in \mathcal { H } ^ { 2 } \left( \Omega ; \mathbb { R } ^ { d } \right) , } } \mathcal { L } _ { q } ( u _ { e n c } , u _ { d e c } ^ { * } ; f ) \ge \frac { T _ { a } ^ { 2 } \mu ^ { 2 } } { K } > 0 .\tag{63}
$$

In particular, the worst-case approximation error of the smoothing-spline decoder cannot vanish as $N \to \infty$ when the adversarial budget constitutes a constant positive fraction of N.

## V. PROOFS OF THE MAIN RESULTS

In this section, we provide the proofs of the main theorems and corollaries, each presented in a separate subsection. Several proofs rely on auxiliary lemmas whose proofs are deferred to Appendix B.

## A. Proof of Theorem 2

Fix $u _ { \mathrm { e n c } } \in \mathcal { H } ^ { 2 } \left( \Omega ; \mathbb { R } ^ { d } \right)$ , and define

$$
g ( z ) : = f ( u _ { \mathrm { e n c } } ( z ) ) .\tag{64}
$$

Lemma 1 ( [12, Lemma 1]). Let $u _ { e n c } \in \mathcal { H } ^ { 2 } \left( \Omega ; \mathbb { R } ^ { d } \right)$ , and let $f = [ f _ { 1 } , \ldots , f _ { m } ] ^ { \intercal } : \mathbb { R } ^ { d }  \mathbb { R } ^ { m }$ satisfy Assumption (A2). Then, $f \circ u _ { e n c } \in \mathcal { H } ^ { 2 } \left( \Omega ; \mathbb { R } ^ { m } \right)$

Applying Lemma 1, we have $g \in \mathcal { H } ^ { 2 } \left( \Omega ; \mathbb { R } ^ { m } \right)$ . The clean decoder in (40) is a smoothing spline constructed over the design points $\{ \beta _ { j } \} _ { j = 1 } ^ { N }$ with respect to the target function g(·). Therefore, using the smoothing spline operator defined in (21), we obtain

$$
u _ { \mathrm { d e c } } ^ { \circ } = s _ { N , \lambda _ { \mathrm { d } } } [ g ] .\tag{65}
$$

1) Upper bound for clean generalization error: We first consider the clean generalization error. By definition in (36), we have

$$
\mathcal { L } _ { \mathrm { d e c } } ^ { \mathrm { g e n } } = 4 \operatorname* { s u p } _ { z \in \Omega } \| s _ { N , \lambda _ { \mathrm { d } } } [ g ] ( z ) - g ( z ) \| _ { 2 } ^ { 2 } .\tag{66}
$$

Thus, we need a uniform upper bound for the second order smoothing-spline approximation error. The following lemma provides the required estimate.

Lemma 2 (Second-order smoothing-spline error bound). Let $h \in \mathcal { H } ^ { 2 } \left( \Omega ; \mathbb { R } ^ { m } \right)$ and let $t _ { j } = ( j - 1 ) / ( n - 1 )$ for $j \in [ n ]$ . Let $s _ { n , \lambda } [ h ]$ be the second order smoothing spline constructed over design point set $\{ t _ { j } \} _ { j \in [ n ] }$ and target function $h ( \cdot )$ with smoothing parameter $0 < \lambda \leq 1$ . Then, there exists constant $C _ { \mathrm { s p } } ,$ , independent of n, λ, and h, such that

$$
\operatorname* { s u p } _ { z \in \Omega } \| s _ { N , \lambda } [ h ] ( z ) - h ( z ) \| _ { 2 } ^ { 2 } \leq C _ { \mathrm { s p } } \left( \lambda ^ { 3 / 4 } + N ^ { - 3 } \right) \| h ^ { \prime \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } ^ { 2 } .\tag{67}
$$

The proof is provided in Appendix B-A. Applying Lemma 2 to $h = g = f \circ u _ { \mathrm { e n c } }$ and $\lambda = \lambda _ { \mathrm { d } }$ gives

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { d e c } } ^ { \mathrm { g e n } } \leq 4 C _ { \mathrm { s p } } \left( \lambda _ { \mathrm { d } } ^ { 3 / 4 } + N ^ { - 3 } \right) \| g ^ { \prime \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } ^ { 2 } . } \end{array}\tag{68}
$$

It remains to derive an upper bound for the Sobolev regularity of the composite function $\| g ^ { \prime \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } ^ { 2 }$ . Following lemma provides an upper bound for $\| g ^ { \prime \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } ^ { 2 }$ in terms of Sobolev norm of encoder $u _ { \mathrm { e n c } }$ and target function regularity constants $H _ { f } , G _ { f }$

Lemma 3 (Regularity of the composite function). Under Assumption $( \mathsf { A } 2 ) ,$ , there exists a constant $C _ { \mathrm { c } } > 0$ such that, for every $u \in \mathcal { H } ^ { 2 } \left( \Omega ; \mathbb { R } ^ { d } \right)$ ,

$$
\begin{array} { r } { \| ( f \circ u ) ^ { \prime \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } ^ { 2 } \leq C _ { \mathrm { c } } \operatorname* { m a x } \{ H _ { f } ^ { 2 } , G _ { f } ^ { 2 } \} \psi \left( \| u \| _ { \mathcal { H } ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 } \right) , } \end{array}\tag{69}
$$

where $\psi ( t ) = t + 4 t ^ { 2 }$

The detailed proof can be found in Appendix B-B. Applying Lemma 3 to $u = u _ { \mathrm { e n c } }$ in (68) yields

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { d e c } } ^ { \mathrm { g e n } } \leq C _ { 1 } \operatorname* { m a x } \{ H _ { f } ^ { 2 } , G _ { f } ^ { 2 } \} \left( \lambda _ { \mathrm { d } } ^ { 3 / 4 } + N ^ { - 3 } \right) \psi \left( \left. u _ { \mathrm { e n c } } \right. _ { \mathcal { H } ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 } \right) , } \end{array}\tag{70}
$$

where $C _ { 1 } : = 4 C _ { \mathrm { s p } } C _ { \mathrm { c } }$

2) Upper bound for Adversarial perturbation error: We next bound the adversarial perturbation error. By linearity of the smoothing-spline operator discussed in Section III-C, the corrupted and clean decoders admit the representations

$$
u _ { \mathrm { d e c } } ^ { * } ( z ) = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } G _ { N , \lambda _ { \mathrm { d } } } ( z , \beta _ { j } ) \overline { { \mathbf { y } } } _ { j } ^ { A } ,\tag{71}
$$

$$
u _ { \mathrm { d e c } } ^ { \circ } ( z ) = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } G _ { N , \lambda _ { \mathrm { d } } } ( z , \beta _ { j } ) g ( \beta _ { j } ) .\tag{72}
$$

Let B denote the adversarial worker set. Since the outputs of all workers outside B are uncorrupted, subtracting (72) from (71) gives

$$
u _ { \mathrm { d e c } } ^ { * } ( z ) - u _ { \mathrm { d e c } } ^ { \circ } ( z ) = \frac { 1 } { N } \sum _ { j \in { \cal B } } G _ { N , \lambda _ { \mathrm { d } } } ( z , \beta _ { j } ) \left( { \bf a } _ { j } - g ( \beta _ { j } ) \right) .\tag{73}
$$

To bound (73), we require an upper bound on the exact discrete weight function ${ \cal G } _ { N , \lambda _ { \mathrm { d } } }$ , which does not follow directly from its definition. Instead, we derive it by comparing ${ \cal G } _ { N , \lambda _ { \mathrm { d } } }$ with the continuous Green’s function and its equivalent kernel. More specifically, for $h \in L ^ { 2 } \left( \Omega ; \mathbb { R } \right)$ ), consider the continuous smoothing problem

$$
v _ { \lambda _ { \mathrm { d } } } [ h ] : = \operatorname * { a r g m i n } _ { v \in \mathcal { H } ^ { 2 } ( \Omega ; \mathbb { R } ) } \left\{ \int _ { \Omega } \bigl ( h ( t ) - v ( t ) \bigr ) ^ { 2 } d t + \lambda _ { \mathrm { d } } \int _ { \Omega } \bigl ( v ^ { \prime \prime } ( t ) \bigr ) ^ { 2 } d t \right\} .\tag{74}
$$

It can be shown that the solution to (74) is unique and can be represented using the continuous Green’s function [29], [30] $G _ { \lambda _ { \mathrm { d } } }$ as

$$
v _ { \lambda _ { \mathrm { d } } } ( x ) = \int _ { \Omega } G _ { \lambda _ { \mathrm { d } } } ( x , t ) h ( t ) d t .\tag{75}
$$

Similar to smoothing spline weight function $G _ { N , \lambda _ { \mathrm { d } } } ,$ deriving explicit formula or upper bound for $G _ { \lambda _ { \mathrm { d } } }$ is challenging in general form. However, authors in [31] proposed a kernel that is a good approximation of $G _ { \lambda _ { \mathrm { d } } }$ for equidistant design points. Formally, define

$$
\begin{array} { r l } & { { \cal K } _ { \lambda _ { \mathrm { d } } } ( x , t ) = \frac { e ^ { - \frac { \vert x - t \vert } { \sqrt { 2 } \lambda _ { \mathrm { d } } ^ { 1 / 4 } } } } { 2 \sqrt { 2 } \lambda _ { \mathrm { d } } ^ { 1 / 4 } } \left( \sin \left( \frac { \vert x - t \vert } { \sqrt { 2 } \lambda _ { \mathrm { d } } ^ { 1 / 4 } } \right) + \cos \left( \frac { x - t } { \sqrt { 2 } \lambda _ { \mathrm { d } } ^ { 1 / 4 } } \right) \right) } \\ & { \quad \quad \quad \quad + \frac { 1 } { 2 \sqrt { 2 } \lambda _ { \mathrm { d } } ^ { 1 / 4 } } \left[ \Phi \left( \frac { x + t } { \sqrt { 2 } \lambda _ { \mathrm { d } } ^ { 1 / 4 } } , \frac { x - t } { \sqrt { 2 } \lambda _ { \mathrm { d } } ^ { 1 / 4 } } \right) + \Phi \left( \frac { 2 - x - t } { \sqrt { 2 } \lambda _ { \mathrm { d } } ^ { 1 / 4 } } , \frac { t - x } { \sqrt { 2 } \lambda _ { \mathrm { d } } ^ { 1 / 4 } } \right) \right] , } \end{array}\tag{76}
$$

where $\Phi ( u , v ) : = e ^ { - u } \left( \cos u - \sin u + 2 \cos v \right)$ . We first bound this explicit kernel in the following lemma.

Lemma 4 (Upper Bound on the Equivalent Kernel). There exists a constant $\tau > 0$ such that for every $\lambda \in ( 0 , 1 ]$

$$
\operatorname* { s u p } _ { x , t \in [ 0 , 1 ] } | K _ { \lambda } ( x , t ) | \leq \tau \lambda ^ { - 1 / 4 } .\tag{77}
$$

See Appendix B-C for the proof. Next, the following lemma relates the finite-sample spline weight function $G _ { N , \lambda }$ and the equivalent kernel $K _ { \lambda }$

Lemma 5 (Approximation of the Finite-Sample Spline Weights). Under equidistant design points, there exist constants $c _ { 0 } , C _ { W } , \kappa , C _ { K } > 0$ such that whenever $0 < \lambda \leq 1$ and $N \lambda ^ { 1 / 4 } > c _ { 0 } ,$ , we have for all $x , t \in [ 0 , 1 ]$

$$
| G _ { N , \lambda } ( x , t ) - K _ { \lambda } ( x , t ) | \leq C _ { W } \lambda ^ { - 1 / 4 } \exp \left( - \kappa | x - t | \lambda ^ { - 1 / 4 } \right) + C _ { M } \lambda ^ { - 1 / 4 } \exp \left( - \frac { 1 } { \sqrt { 2 } } \lambda ^ { - 1 / 4 } \right) .\tag{78}
$$

The detailed proof is provided in Appendix B-D. We can now derive an upper bound on the exact weight function used by the decoder. By the triangle inequality,

$$
\vert G _ { N , \lambda _ { \mathrm { d } } } ( x , t ) \vert \leq \vert K _ { \lambda _ { \mathrm { d } } } ( x , t ) \vert + \vert G _ { N , \lambda _ { \mathrm { d } } } ( x , t ) - K _ { \lambda _ { \mathrm { d } } } ( x , t ) \vert .\tag{79}
$$

Using Lemmas 4 and 5, we obtain

$$
\vert G _ { N , \lambda _ { \mathrm { d } } } ( x , t ) \vert \leq \tau \lambda _ { \mathrm { d } } ^ { - 1 / 4 } + C _ { W } \lambda _ { \mathrm { d } } ^ { - 1 / 4 } \exp \left( - \kappa \vert x - t \vert \lambda _ { \mathrm { d } } ^ { - 1 / 4 } \right) + C _ { M } \lambda _ { \mathrm { d } } ^ { - 1 / 4 } \exp \left( - \frac { 1 } { \sqrt { 2 } } \lambda _ { \mathrm { d } } ^ { - 1 / 4 } \right)
$$

$$
\begin{array} { r } { \stackrel { \mathrm { \scriptsize ( a ) } } { \leq } ( \tau + C _ { M } + C _ { W } ) \lambda _ { \mathrm { d } } ^ { - 1 / 4 } , } \end{array}\tag{80}
$$

where (a) follows from the fact that both exponential terms are bounded by one.

Therefore, defining $C _ { \mathrm { G } } : = \tau + C _ { M } + C _ { W }$ , we have

$$
| G _ { N , \lambda _ { \mathrm { d } } } ( x , t ) | \le C _ { G } \lambda _ { \mathrm { d } } ^ { - 1 / 4 }\tag{81}
$$

We now return to the decoder adversarial perturbation error in (73). Assumptions A1 and A3 imply

$$
\| \mathbf { a } _ { j } \| _ { 2 } \leq \sqrt { m } T _ { a } ,\tag{82}
$$

$$
\| g ( \beta _ { j } ) \| _ { 2 } = \| f ( u _ { \mathrm { e n c } } ( \beta _ { j } ) ) \| _ { 2 } \leq \sqrt { m } T _ { f } .\tag{83}
$$

Since $T = \operatorname* { m a x } \{ T _ { a } , T _ { f } \}$

$$
\| \mathbf { a } _ { j } - g ( \beta _ { j } ) \| _ { 2 } \leq 2 { \sqrt { m } } T .\tag{84}
$$

Using (81) and (84) in (73), and $| B | \leq q ,$ , gives

$$
\begin{array} { r l r } {  { \operatorname* { s u p } _ { z \in \Omega } \| u _ { \mathrm { d e c } } ^ { * } ( z ) - u _ { \mathrm { d e c } } ^ { \circ } ( z ) \| _ { 2 } \leq \frac { 2 \sqrt { m } T } { N } \sum _ { j \in \mathcal { B } } \operatorname* { s u p } _ { z \in \Omega } \lvert G _ { N , \lambda _ { \mathrm { d } } } ( z , \beta _ { j } ) \rvert } } \\ & { } & { \leq 2 C _ { \mathrm { G } } \sqrt { m } T \frac { q } { N } \lambda _ { \mathrm { d } } ^ { - 1 / 4 } . } \end{array}\tag{85}
$$

Consequently,

$$
\mathcal { L } _ { \mathrm { d e c } } ^ { \mathrm { a d v } } = 4 \operatorname* { s u p } _ { A \in A _ { q } } \operatorname* { s u p } _ { z \in \Omega } \lVert u _ { \mathrm { d e c } } ^ { * } ( z ) - u _ { \mathrm { d e c } } ^ { \circ } ( z ) \rVert _ { 2 } ^ { 2 } \leq 1 6 C _ { \mathrm { G } } ^ { 2 } m T ^ { 2 } \frac { q ^ { 2 } } { N ^ { 2 } } \lambda _ { \mathrm { d } } ^ { - 1 / 2 } .\tag{86}
$$

Thus, the adversarial perturbation term in (42) holds with

$$
C _ { 2 } : = 1 6 C _ { \mathrm { G } } ^ { 2 } .\tag{87}
$$

Adding (70) and (86) proves the claimed bound for the constructed decoder $u _ { \mathrm { d e c } } ^ { \ast }$ defined in (39). Since $u _ { \mathrm { d e c } } ^ { \ast }$ is an admissible decoder in $\mathcal { H } ^ { 2 } \left( \Omega ; \mathbb { R } ^ { m } \right)$ , we obtain

$$
\operatorname* { i n f } _ { \substack { u _ { \mathrm { d e c } } \in \mathcal { H } ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } } \left\{ \mathcal { L } _ { \mathrm { d e c } } ^ { \mathrm { g e n } } + \mathcal { L } _ { \mathrm { d e c } } ^ { \mathrm { a d v } } \right\} \leq C _ { 1 } \operatorname* { m a x } \{ H _ { f } ^ { 2 } , G _ { f } ^ { 2 } \} \left( \lambda _ { \mathrm { d } } ^ { 3 / 4 } + N ^ { - 3 } \right) \psi \left( \left. u _ { \mathrm { e n c } } \right. _ { \mathcal { H } ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 } \right) + C _ { 2 } m T ^ { 2 } \frac { q ^ { 2 } } { N ^ { 2 } } \lambda _ { \mathrm { d } } ^ { - 1 / 2 } .\tag{88}
$$

This proves the theorem.

## B. Proof of Corollary 1

Define

$$
\begin{array} { r } { A _ { u _ { \mathrm { e n c } } } : = C _ { 1 } \operatorname* { m a x } \{ H _ { f } ^ { 2 } , G _ { f } ^ { 2 } \} \psi \left( \lVert u _ { \mathrm { e n c } } \rVert _ { \mathcal { H } ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 } \right) , } \end{array}\tag{89}
$$

$$
B _ { N } : = C _ { 2 } m T ^ { 2 } \frac { q ^ { 2 } } { N ^ { 2 } } .\tag{90}
$$

In the nondegenerate case where $q \geq 1$ and $A _ { u _ { \mathrm { { e n c } } } } > 0$ , the part of the upper bound in (42) that depends on $\lambda _ { \mathrm { d } }$ is

$$
\begin{array} { r } { \Phi _ { N } ( \lambda _ { \mathrm { d } } ) : = A _ { u _ { \mathrm { e n c } } } \lambda _ { \mathrm { d } } ^ { 3 / 4 } + B _ { N } \lambda _ { \mathrm { d } } ^ { - 1 / 2 } . } \end{array}\tag{91}
$$

Its derivative is

$$
\Phi _ { N } ^ { \prime } ( \lambda _ { \mathrm { d } } ) = \frac { 3 } { 4 } { \cal A } _ { u _ { \mathrm { e n c } } } \lambda _ { \mathrm { d } } ^ { - 1 / 4 } - \frac { 1 } { 2 } B _ { N } \lambda _ { \mathrm { d } } ^ { - 3 / 2 } .\tag{92}
$$

The unique stationary point satisfies

$$
\lambda _ { \mathrm { d } } ^ { 5 / 4 } = \frac { 2 B _ { N } } { 3 A _ { u _ { \mathrm { e n c } } } } .\tag{93}
$$

Since $\Phi _ { N } ^ { \prime }$ is negative below this point and positive above it, the stationary point is the unique minimizer:

$$
\lambda _ { \mathrm { d } } ^ { * } = \left( \frac { 2 C _ { 2 } m T ^ { 2 } q ^ { 2 } } { 3 C _ { 1 } N ^ { 2 } \operatorname* { m a x } \{ H _ { f } ^ { 2 } , G _ { f } ^ { 2 } \} \psi \left( \lVert u _ { \mathrm { e n c } } \rVert _ { \mathcal { H } ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 } \right) } \right) ^ { 4 / 5 } .\tag{94}
$$

For fixed $q$ and fixed problem parameters, $\lambda _ { \mathrm { d } } ^ { * } \to 0$ as $N \to \infty$ . Moreover,

$$
N ( \lambda _ { \mathrm { d } } ^ { * } ) ^ { 1 / 4 } = \left( \frac { 2 C _ { 2 } m T ^ { 2 } } { 3 C _ { 1 } \operatorname* { m a x } \{ H _ { f } ^ { 2 } , G _ { f } ^ { 2 } \} \psi \left( \| u _ { \mathrm { e n c } } \| _ { \mathcal { H } ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 } \right) } \right) ^ { 1 / 5 } q ^ { 2 / 5 } N ^ { 3 / 5 }\tag{95}
$$

Thus, there exists $N _ { 0 }$ such that, for every $N \geq N _ { 0 }$ , we have $0 < \lambda _ { \mathrm { d } } ^ { * } \leq 1$ , and $N ( \lambda _ { \mathrm { d } } ^ { * } ) ^ { 1 / 4 } \geq c _ { 0 }$ . Finally,

$$
\lambda _ { \mathrm { d } } ^ { * } \asymp \left( \frac { q } { N } \right) ^ { 8 / 5 } ,\tag{96}
$$

as claimed.

## C. Proof of Theorem 3

The proof follows the same argument as that of Theorem 3 in [12]. Indeed, for a fixed decoder smoothing parameter $\lambda _ { \mathrm { d } } .$ , the encoder-dependent component of the loss upper bound has the exact same form as in the straggler setting. The only additional term is the adversarial perturbation $\Delta _ { q , N }$ , which is independent of the encoder and therefore does not affect the optimization or the comparison argument used in the original proof.

## D. Proof of Corollary 2

Proof. Let $s _ { \mathbf { X } } \in \mathcal { H } ^ { 2 } \left( \Omega ; \mathbb { R } ^ { d } \right)$ be the natural cubic spline interpolating the input batch in encoder design points:

$$
s _ { \mathbf { X } } ( \alpha _ { k } ) = { \mathbf { x } } _ { k } , \qquad k \in [ K ] .\tag{97}
$$

Because the input batch and encoder design points are fixed, $\| s \mathbf { x } \| _ { \mathcal { H } ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) }$ is independent of N. Moreover,

$$
\frac { 1 } { K } \sum _ { k = 1 } ^ { K } \lVert s _ { \mathbf { X } } ( \alpha _ { k } ) - \mathbf { x } _ { k } \rVert _ { 2 } ^ { 2 } = 0 .\tag{98}
$$

By Theorem 3, we have

$$
J _ { \mathrm { a d v } } \leq \Delta _ { q , N } + 2 G _ { f } ^ { 2 } \Big [ \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \lVert u _ { \mathrm { e n c } } ^ { \dagger } ( \alpha _ { k } ) - \mathbf { x } _ { k } \rVert _ { 2 } ^ { 2 } + \lambda _ { \mathrm { e } } \left( R _ { \mathrm { e } } + \lVert ( u _ { \mathrm { e n c } } ^ { \dagger } ) ^ { \prime \prime } \rVert _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 } \right) \Big ] .\tag{99}
$$

Since $u _ { \mathrm { e n c } } ^ { \dagger } ( \cdot )$ minimizes the regularized objective

$$
\frac { 1 } { K } \sum _ { k = 1 } ^ { K } \lVert u ( \alpha _ { k } ) - \mathbf { x } _ { k } \rVert _ { 2 } ^ { 2 } + \lambda _ { \mathrm { e } } \lVert u ^ { \prime \prime } \rVert _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 }
$$

over all $u \in \mathcal { H } ^ { 2 } \left( \Omega ; \mathbb { R } ^ { d } \right)$ , its objective value is upper bounded by that of the natural cubic spline interpolant $s _ { \mathbf { X } } \in \mathcal { H } ^ { 2 } \left( \Omega ; \mathbb { R } ^ { d } \right)$ . Consequently, we obtain

$$
\begin{array} { r l } & { J _ { \mathrm { a d v } } \leq \Delta _ { q , N } + 2 G _ { f } ^ { 2 } \Big [ \displaystyle \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \lVert u _ { \mathrm { e n c } } ^ { \dagger } ( \alpha _ { k } ) - \mathbf { x } _ { k } \rVert _ { 2 } ^ { 2 } + \lambda _ { \mathrm { e } } \left( R _ { \mathrm { e } } + \lVert ( u _ { \mathrm { e n c } } ^ { \dagger } ) ^ { \prime \prime } \rVert _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 } \right) \Big ] } \\ & { \quad \leq \Delta _ { q , N } + 2 G _ { f } ^ { 2 } \Big [ \displaystyle \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \lVert s _ { \mathbf { x } } ( \alpha _ { k } ) - \mathbf { x } _ { k } \rVert _ { 2 } ^ { 2 } + \lambda _ { \mathrm { e } } \left( R _ { \mathrm { e } } + \lVert ( s _ { \mathbf { X } } ) ^ { \prime \prime } \rVert _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 } \right) \Big ] } \\ & { \quad \stackrel { \mathrm { ( a ) } } { = } \Delta _ { q , N } + 2 G _ { f } ^ { 2 } \lambda _ { \mathrm { e } } \left( R _ { \mathrm { e } } + \lVert ( s _ { \mathbf { X } } ) ^ { \prime \prime } \rVert _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 } \right) , } \end{array}\tag{100}
$$

where (a) follows from the exact interpolation property $s _ { \mathbf { X } } ( \alpha _ { k } ) = \mathbf { x } _ { k }$ in (98). Substituting the definitions of $\Delta _ { q , N }$ and $\lambda _ { \mathrm { e } }$ from (50) and (47), respectively, yields

$$
J _ { \mathrm { a d v } } \leq A _ { 0 } \left( \lambda _ { \mathrm { d } } ^ { 3 / 4 } + N ^ { - 3 } \right) + B _ { 0 } \frac { q ^ { 2 } } { N ^ { 2 } } \lambda _ { \mathrm { d } } ^ { - 1 / 2 } ,\tag{101}
$$

where $A _ { 0 } : = 2 G _ { f } ^ { 2 } \big ( R _ { \mathrm { e } } + \| ( s _ { \mathbf { X } } ) ^ { \prime \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 } \big ) ( m _ { 1 } + m _ { 2 } R _ { \mathrm { e } } )$ and $B _ { 0 } : = C _ { 2 } m T ^ { 2 }$

For $q \geq 1$ , choose $\textstyle \lambda _ { \mathrm { d } } \asymp \left( \frac { q } { N } \right) ^ { 8 / 5 }$ . Then we have $\textstyle \lambda _ { \mathrm { d } } ^ { 3 / 4 } \asymp \left( \frac { q } { N } \right) ^ { 6 / 5 }$ and $\textstyle { \frac { q ^ { 2 } } { N ^ { 2 } } } \lambda _ { \mathrm { d } } ^ { - 1 / 2 } \asymp { \bigl ( } { \frac { q } { N } } { \bigr ) } ^ { 6 / 5 }$ . Consequently,

$$
J _ { \mathrm { a d v } } = \mathcal { O } \left( \left( \frac { q } { N } \right) ^ { 6 / 5 } + N ^ { - 3 } \right) .\tag{102}
$$

Since $q \leq C _ { q } N ^ { \theta }$ , we obtain

$$
\left( { \frac { q } { N } } \right) ^ { 6 / 5 } \leq C _ { q } ^ { 6 / 5 } N ^ { \frac 6 5 ( \theta - 1 ) } .\tag{103}
$$

For $\theta \in [ 0 , 1 ) , \frac { 6 } { 5 } ( \theta - 1 ) \geq - \frac { 6 } { 5 } > - 3$ . Thus, the $N ^ { - 3 }$ term is asymptotically no larger than the first term, and

$$
\begin{array} { r } { J _ { \mathrm { a d v } } = \mathcal { O } \left( N ^ { \frac { 6 } { 5 } ( \theta - 1 ) } \right) . } \end{array}\tag{104}
$$

It remains to verify the conditions of Theorem 2. We have

$$
N \lambda _ { \mathrm { d } } ^ { 1 / 4 } \asymp N \left( \frac { q } { N } \right) ^ { 2 / 5 } = N ^ { 3 / 5 } q ^ { 2 / 5 } \longrightarrow \infty .\tag{105}
$$

Therefore for sufficiently large N it satisfies $N \lambda _ { \mathrm { d } } ^ { 1 / 4 } \geq c _ { 0 }$ . Also, $q = o ( N )$ implies $\lambda _ { \mathrm { d } } \to 0$

When $q = 0 ,$ , the adversarial term vanishes. Choosing $\lambda _ { \mathrm { d } } \asymp N ^ { - 4 }$ gives an error of order $N ^ { - 3 }$ , which also satisfies the stated upper bound. □

## E. Proof of Theorem 4

Fix $f \in \mathcal { F } _ { \mathrm { s m } } ^ { d  m }$ , an encoder $u _ { \mathrm { e n c } } \in \mathcal { H } ^ { 2 } \left( \Omega ; \mathbb { R } ^ { d } \right)$ , and a decoder smoothing parameter $\lambda _ { \mathrm { d } } > 0$ chosen prior to observing the worker outputs. As established in Section III-C and (22), the smoothing-spline decoder is a linear operator with respect to the received worker outputs. Specifically, under any admissible adversarial attack A, the decoded function takes the form

$$
u _ { \mathrm { d e c } } ^ { * } ( z ) = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } G _ { N , \lambda _ { \mathrm { d } } } ( z , \beta _ { n } ) \overline { { \mathbf { y } } } _ { n } ^ { A } , \qquad z \in \Omega ,\tag{106}
$$

where the weight function $G _ { N , \lambda _ { \mathrm { d } } } ( z , \beta _ { n } )$ depends solely on the decoder design points $\{ \beta _ { n } \} _ { n = 1 } ^ { N }$ , the smoothing parameter $\lambda _ { \mathrm { d } }$ , and $N .$ . Evaluating $u _ { \mathrm { d e c } } ^ { * } ( \cdot )$ at the first target point $\alpha _ { 1 }$ yields

$$
u _ { \mathrm { d e c } } ^ { * } ( \alpha _ { 1 } ) = \sum _ { n = 1 } ^ { N } w _ { n } \overline { { \mathbf { y } } } _ { n } ^ { A } ,\tag{107}
$$

where $\begin{array} { r } { w _ { n } : = \frac { 1 } { N } G _ { N , \lambda _ { \mathrm { d } } } ( \alpha _ { 1 } , \beta _ { n } ) } \end{array}$ . Note that the same scalar weights $\{ w _ { n } \} _ { n = 1 } ^ { N }$ apply across all m output coordinates because the smoothing-spline objective in (39) decouples coordinatewise.

To determine the sum of these weights, suppose that $\overline { { \mathbf { y } } } _ { n } ^ { A } = \mathbf { c }$ for all $n \in [ N ]$ , where $\mathbf { c } \in \mathbb { R } ^ { m }$ is an arbitrary non-zero constant vector. In this case, the constant function $u _ { \mathrm { d e c } } ( z ) \equiv \mathbf { c }$ achieves an objective value of zero in (39) and is therefore the unique minimizer $u _ { \mathrm { d e c } } ^ { \ast }$ . Substituting $\bar { \mathbf { y } } _ { n } ^ { A } = \mathbf { c }$ and $u _ { \mathrm { d e c } } ^ { * } ( \alpha _ { 1 } ) = \mathbf { c }$ into (107) implies that the weights reproduce constants:

$$
\sum _ { n = 1 } ^ { N } w _ { n } = 1 .\tag{108}
$$

Now let $B \subset [ N ]$ denote the subset of $q$ worker indices corresponding to the q largest values of $\left| w _ { n } \right|$ (breaking ties arbitrarily). Since the average of the $q$ largest magnitudes is at least the overall average across all N indices, it follows from (108) and the triangle inequality that

$$
\sum _ { n \in \mathcal { B } } \lvert w _ { n } \rvert \ge \frac { q } { N } \sum _ { n = 1 } ^ { N } \lvert w _ { n } \rvert \ge \frac { q } { N } \lvert \sum _ { n = 1 } ^ { N } w _ { n } \rvert = \frac { q } { N } .\tag{109}
$$

For each sign $\sigma \in \{ - 1 , 1 \}$ , consider the adversarial attack $A _ { \sigma }$ that corrupts the workers in B by setting

$$
\begin{array} { r } { \overline { { \mathbf { y } } } _ { n } ^ { A _ { \sigma } } = \sigma T _ { a } \mathrm { s i g n } ( w _ { n } ) \mathbf { e } _ { 1 } , \qquad n \in \mathcal { B } , } \end{array}\tag{110}
$$

while leaving all remaining worker outputs $( n \notin B )$ uncorrupted, where $\mathbf { e } _ { 1 } \in \mathbb { R } ^ { m }$ denotes the first standard basis vector and sign $\left( 0 \right) : = 0$ . Both $A _ { 1 }$ and $A _ { - 1 }$ are admissible under Assumption (A3), as each modifies at most $| B | = q$ worker outputs and satisfies $\mathbf { \overline { { y } } } _ { n } ^ { A _ { \sigma } } \in [ - T _ { a } , T _ { a } ] ^ { m }$

Next, define the uncorrupted residual b and the adversarial perturbation magnitude a in the first output coordinate at $\alpha _ { 1 }$ by

$$
b : = \sum _ { n \not \in \mathcal { B } } w _ { n } f _ { 1 } \big ( u _ { \mathrm { e n c } } ( \beta _ { n } ) \big ) - f _ { 1 } ( \mathbf { x } _ { 1 } ) ,\tag{111}
$$

$$
a : = T _ { a } \sum _ { n \in B } | w _ { n } | .\tag{112}
$$

By (107), under attack $A _ { \sigma }$ , the estimation error in the first coordinate at $\alpha _ { 1 }$ is given by $\left[ u _ { \mathrm { d e c } } ^ { * } ( \alpha _ { 1 } ) - f ( \mathbf { x } _ { 1 } ) \right] _ { 1 } = b + \sigma a$ Using the identity

$$
\operatorname* { m a x } _ { \sigma \in \{ - 1 , 1 \} } ( b + \sigma a ) ^ { 2 } \geq \frac { ( b + a ) ^ { 2 } + ( b - a ) ^ { 2 } } { 2 } = b ^ { 2 } + a ^ { 2 } \geq a ^ { 2 } ,\tag{113}
$$

together with (109), we lower-bound the worst-case squared Euclidean error at $\alpha _ { 1 }$ as

$$
\operatorname* { m a x } _ { \sigma \in \{ - 1 , 1 \} } \| u _ { \mathrm { d e c } } ^ { * } ( \alpha _ { 1 } ) - f ( \mathbf { x } _ { 1 } ) \| _ { 2 } ^ { 2 } \geq \operatorname* { m a x } _ { \sigma \in \{ - 1 , 1 \} } ( b + \sigma a ) ^ { 2 } \geq a ^ { 2 } \geq T _ { a } ^ { 2 } \left( \frac { q } { N } \right) ^ { 2 } .\tag{114}
$$

Because the squared errors at the remaining $K - 1$ target points $\{ \alpha _ { k } \} _ { k = 2 } ^ { K }$ are non-negative, the worst-case risk over all admissible attacks A satisfies

$$
\mathcal { L } _ { q } ( u _ { \mathrm { e n c } } , u _ { \mathrm { d e c } } ^ { * } ; f ) \geq \frac { 1 } { K } \operatorname* { m a x } _ { \sigma \in \{ - 1 , 1 \} } \| u _ { \mathrm { d e c } } ^ { * } ( \alpha _ { 1 } ) - f ( \mathbf { x } _ { 1 } ) \| _ { 2 } ^ { 2 } \geq \frac { T _ { a } ^ { 2 } } { K } \left( \frac { q } { N } \right) ^ { 2 } .\tag{115}
$$

Since this lower bound holds for any $u _ { \mathrm { e n c } } \in \mathcal { H } ^ { 2 } \left( \Omega ; \mathbb { R } ^ { d } \right)$ and any $\lambda _ { \mathrm { d } } > 0$ , taking the infimum over both parameters yields (62) and completes the proof.

## F. Proof of Corollary 3

Since $\mu \in ( 0 , 1 )$ , for all $N \geq \lceil 1 / \mu \rceil$ , the adversarial budget $q = \lfloor \mu N \rfloor$ satisfies $1 \leq q < N$ . Applying Theorem 4 for every such N yields

$$
\operatorname* { i n f } _ { u _ { \mathrm { e n c } } \in \mathcal { H } ^ { 2 } \left( \Omega ; \mathbb { R } ^ { d } \right) , } \mathcal { L } _ { q } \big ( u _ { \mathrm { e n c } } , u _ { \mathrm { d e c } } ^ { * } ; f \big ) \geq \frac { T _ { a } ^ { 2 } } { K } \left( \frac { \lfloor \mu N \rfloor } { N } \right) ^ { 2 } .\tag{116}
$$

Because lim $N {  } { \infty } \lfloor \mu N \rfloor / N = \mu$ , taking the limit inferior as $N \to \infty$ on both sides establishes the corollary.

## VI. EXPERIMENTAL RESULTS

In this section, we empirically evaluate the proposed adversarial GCC framework. We consider two types of computing functions: (1) a scalar nonlinear function and (2) a high-dimensional neural-network inference task. The primary purpose of these experiments is to examine the effectiveness of GCC to mitigate the adversarial effect and convergence-rate prediction of Corollary 2 under a concrete adversarial strategy.

## A. Experimental Setup

Target Tasks: We evaluate our framework on two representative computing tasks:

• Scalar nonlinear function: The first target function is given by

$$
f _ { \mathrm { s c } } ( x ) : = x \sin ( x ) ,\tag{117}
$$

![](images/cf325bf27e6fed064e3d1ee4bdb5f0dfd2fb67ed0212071079cd880119f50400.jpg)  
(a) Scalar computation $f _ { \mathrm { s c } } ( x ) = x \sin ( x )$

![](images/006f91d9667a7189ffcd778c325409fc5d4863112afe47abbf126960d96fa8db.jpg)  
(b) LeNet-5 inference.  
Fig. 2: Approximation error as a function of the number of workers for adversarial budgets $q = \lfloor N ^ { \theta } \rfloor$ . The light solid curves show the empirical errors, the dark solid lines show the fitted empirical slopes, and the dashed lines show the convergence-rate predictions from Corollary 2.

evaluated over the compact interval [−12, 12] to ensure bounded inputs and outputs.

• Neural-network inference: The second task evaluates a LeNet-5 convolutional neural network [32] trained on the MNIST dataset [33]. Each input is preprocessed into a $3 2 \times 3 2$ grayscale image (1024-dimensional vector), and the network outputs a 10-dimensional vector representing the class probability distribution across digits 0 through 9:

$$
f _ { \mathrm { N N } } : \mathbb { R } ^ { 1 0 2 4 }  \mathbb { R } ^ { 1 0 } .\tag{118}
$$

In both settings, we fix the batch size to $K = 1 0$ . The encoder design points $\{ \alpha _ { k } \} _ { k = 1 } ^ { K }$ and worker evaluation points $\{ \beta _ { j } \} _ { j = 1 } ^ { N }$ are placed uniformly over [0, 1] according to (38):

$$
\alpha _ { k } = \frac { k - 1 } { K - 1 } , \qquad k \in [ K ] ,\tag{119}
$$

$$
\beta _ { j } = \frac { j - 1 } { N - 1 } , \qquad j \in [ N ] .\tag{120}
$$

To evaluate performance across different adversarial regimes, we scale the number of adversarial workers as

$$
q : = \left\lfloor N ^ { \theta } \right\rfloor , \qquad 0 \leq \theta < 1 .\tag{121}
$$

Evaluation Metric: We evaluate performance via the expected end-to-end mean squared error (MSE):

$$
\mathrm { M S E } : = \mathbb { E } _ { \mathbf { x } \sim \mathcal { X } } \left[ \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \lVert \widehat { f } ( \mathbf { x } _ { k } ) - f ( \mathbf { x } _ { k } ) \rVert _ { 2 } ^ { 2 } \right] ,\tag{122}
$$

where $\mathcal { X }$ denotes the input data distribution. In practice, we approximate (122) by computing the empirical average over 10 independent experimental runs.

## B. Adversarial Strategy

For each encoder design point $\alpha _ { k }$ , the adversary selects worker nodes whose decoder design points $\beta _ { j }$ are closest to $\alpha _ { k }$ . The total adversarial budget $q$ is distributed as evenly as possible among the K encoder design points. Consequently, approximately $q / K$ outputs near each $\alpha _ { k }$ are corrupted.

The corrupted values are selected differently for the two computing tasks.

• Scalar nonlinear function: For every selected worker $j ,$ the adversary replaces the honest output $f _ { \mathrm { s c } } ( u _ { \mathrm { e n c } } ( \beta _ { j } ) )$ by the largest admissible scalar value $\overline { { y } } _ { j } ^ { A } = T _ { a }$ . We set $T _ { a } = 1 0$ in this experiment.

• LeNet-5 inference: Let

$$
c _ { j } : = \underset { i \in [ m ] } { \arg \operatorname* { m a x } } \left[ f _ { \mathrm { N N } } \left( u _ { \mathrm { e n c } } ( \beta _ { j } ) \right) \right] _ { i }\tag{123}
$$

denote the class predicted by the honest worker output. The adversary selects a random different class $c _ { j } ^ { A }$ . It then replaces the honest output vector by $\overline { { \mathbf { y } } } _ { j } ^ { A } = \mathbf { e } _ { c _ { j } ^ { A } }$ , where $\mathbf { e } _ { c _ { j } ^ { A } } \in \mathbb { R } ^ { m }$ is the corresponding standard basis vector. Thus, each corrupted worker outputs a one-hot vector concentrated on a class different from the class predicted by its honest output.

Both attacks concentrate the corrupted outputs near the locations at which the final estimates are evaluated to inject maximum noise. These attacks represent particular admissible strategies and do not restrict the general adversarial model considered in the theoretical analysis.

## C. Convergence-Rate Evaluation

Under the scaling in (121), Corollary 2 gives $J _ { \mathrm { a d v } } = \mathcal { O } ( N ^ { - \frac { 6 } { 5 } ( 1 - \theta ) } )$ . Therefore, the predicted slope in a log-log plot of the approximation error versus N is $- { \frac { 6 } { 5 } } { \left( 1 - \theta \right) }$ . Figure 2 compares these predicted slopes with the slopes fitted to the empirical curves.

In every considered regime, the empirical slope is more negative than the corresponding theoretically guaranteed slope. Thus, over the investigated range of worker populations, the approximation error decays at least as rapidly as predicted by Corollary 2. As expected, increasing θ slows the convergence because the adversarial population grows more rapidly with N.

## VII. CONCLUSION AND FUTURE DIRECTIONS

In this paper, we introduced an adversarially robust framework for general coded computing (GCC) that mitigates the impact of malicious worker nodes through smoothness regularization rather than explicit error localization. By distributing computations via a smooth encoder and aggregating all returned outputs with a regularized decoder, we established a non-asymptotic worst-case error bound. This theoretical analysis proves that our proposed splinebased scheme achieves linear master-side complexity and guarantees a vanishing approximation error whenever the adversarial population grows sublinearly with the total number of workers.

Several open questions and directions remain for future investigation. First, our current design assumes the adversarial budget is known a priori in order to tune the optimal smoothing parameters. Developing adaptive procedures capable of estimating the effective corruption level directly from the received outputs would significantly enhance the framework’s practical applicability. Second, future work should address the simultaneous presence of stragglers and adversarial workers, as missing observations and corrupted outputs distort the regularized decoder via fundamentally different mechanisms. Finally, establishing a matching minimax lower bound for the $\mathcal { O } ( ( q / N ) ^ { 6 / 5 } )$ convergence rate would clarify whether our achievable exponent is strictly optimal, or if exploring higher-order regularization penalties and alternative reproducing kernels could yield sharper convergence rates.

## ACKNOWLEDGMENT

The work has been partially supported by the National Science Foundation under Grant CCF-2348638.

## REFERENCES

[1] J. Verbraeken, M. Wolting, J. Katzy, J. Kloppenburg, T. Verbelen, and J. S. Rellermeyer, “A survey on distributed machine learning,” Acm computing surveys (csur), vol. 53, no. 2, pp. 1–33, 2020.

[2] K.-H. Huang and J. A. Abraham, “Algorithm-based fault tolerance for matrix operations,” IEEE Transactions on Computers, vol. C-33, no. 6, pp. 518–528, 1984.

[3] K. Lee, M. Lam, R. Pedarsani, D. Papailiopoulos, and K. Ramchandran, “Speeding up distributed machine learning using codes,” IEEE Transactions on Information Theory, vol. 64, no. 3, pp. 1514–1529, 2017.

[4] Q. Yu, S. Li, N. Raviv, S. M. M. Kalan, M. Soltanolkotabi, and S. A. Avestimehr, “Lagrange coded computing: Optimal design for resiliency, security, and privacy,” in The 22nd International Conference on Artificial Intelligence and Statistics, pp. 1215–1225, PMLR, 2019.

[5] Q. Yu, M. A. Maddah-Ali, and A. S. Avestimehr, “Straggler mitigation in distributed matrix multiplication: Fundamental limits and optimal coding,” IEEE Transactions on Information Theory, vol. 66, no. 3, pp. 1920–1933, 2020.

[6] S. Dutta, V. Cadambe, and P. Grover, ““short-dot”: Computing large linear transforms distributedly using coded short dot products,” vol. 65, pp. 6171–6193, 2019.

[7] S. Dutta, M. Fahim, F. Haddadpour, H. Jeong, V. Cadambe, and P. Grover, “On the optimal recovery threshold of coded matrix multiplication,” IEEE Transactions on Information Theory, vol. 66, no. 1, pp. 278–301, 2020.

[8] Z. Jia and S. A. Jafar, “On the capacity of secure distributed batch matrix multiplication,” IEEE Transactions on Information Theory, vol. 67, no. 11, pp. 7420–7437, 2021.

[9] T. Jahani-Nezhad and M. A. Maddah-Ali, “Berrut approximated coded computing: Straggler resistance beyond polynomial computing,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 1, pp. 111–122, 2022.

[10] W. Gautschi and G. Inglese, “Lower bounds for the condition number of vandermonde matrices,” Numerische Mathematik, vol. 52, no. 3, pp. 241–250, 1987.

[11] M. Soleymani, H. Mahdavifar, and A. S. Avestimehr, “Analog Lagrange coded computing,” IEEE Journal on Selected Areas in Information Theory, vol. 2, no. 1, pp. 283–295, 2021.

[12] P. Moradi, B. Tahmasebi, and M. A. Maddah-Ali, “Learning-theoretic foundation for general coded computing: The straggler setting,” arXiv preprint arXiv:2608.28910, 2026.

[13] M. Soleymani, R. E. Ali, H. Mahdavifar, and A. S. Avestimehr, “ApproxIFER: A model-agnostic approach to resilient and robust predictio serving systems,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 36, pp. 8342–8350, 2022.

[14] A. Berlinet and C. Thomas-Agnan, Reproducing kernel Hilbert spaces in probability and statistics. Springer Science & Business Media, 2011.

[15] G. Wahba, Spline models for observational data. SIAM, 1990.

[16] R. A. Adams and J. J. Fournier, Sobolev spaces. Elsevier, 2003.

[17] G. Kimeldorf and G. Wahba, “Some results on tchebycheffian spline functions,” Journal of mathematical analysis and applications, vol. 33, no. 1, pp. 82–95, 1971.

[18] P. Moradi, B. Tahmasebi, and M. A. Maddah-Ali, “Coded computing for resilient distributed computing: A learning-theoretic framework,” in The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

[19] B. Schölkopf, R. Herbrich, and A. J. Smola, “A generalized representer theorem,” in International conference on computational learnin theory, pp. 416–426, Springer, 2001.

[20] C. A. Micchelli and M. Pontil, “On learning vector-valued functions,” Neural computation, vol. 17, no. 1, pp. 177–204, 2005.

[21] J.-P. Aubin, Applied functional analysis. John Wiley & Sons, 2000.

[22] Z. Li, D. Meunier, M. Mollenhauer, and A. Gretton, “Towards optimal sobolev norm rates for the vector-valued regularized least-square algorithm,” Journal of Machine Learning Research, vol. 25, no. 181, pp. 1–51, 2024.

[23] G. Wahba, “Smoothing noisy data with spline functions,” Numerische mathematik, vol. 24, no. 5, pp. 383–393, 1975.

[24] F. Bach, Learning theory from first principles. MIT press, 2024.

[25] J. Nocedal and S. J. Wright, Numerical optimization. Springer, 2006.

[26] A. Borodin and R. Moenck, “Fast polynomial algorithms,” in Proceedings of the 6th annual ACM symposium on Theory of computing, pp. 147–157, 1974.

[27] M. JL, “Shift-register synthesis and bch decoding,” IEEE Trans. Inform. Theory, vol. 13, pp. 21–27, 1967.

[28] E. R. Berlekamp, Nonbinary BCH decoding. University of North Carolina. Department of Statistics, 1966.

[29] G. F. Roach, Green’s functions. Cambridge University Press Cambridge, 1982.

[30] D. D. Cox, “Multivariate smoothing spline functions,” SIAM Journal on Numerical Analysis, vol. 21, no. 4, pp. 789–813, 1984.

[31] K. Messer and L. Goldstein, “A new class of kernels for nonparametric curve estimation,” The Annals of Statistics, pp. 179–195, 1993.

[32] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, 1998.

[33] Y. LeCun, C. Cortes, C. Burges, et al., “MNIST handwritten digit database,” 2010.

[34] D. L. Ragozin, “Error bounds for derivative estimates based on spline smoothing of exact or noisy data,” Journal of approximation theory, vol. 37, no. 4, pp. 335–355, 1983.

[35] G. Leoni, A first course in Sobolev spaces, vol. 181. American Mathematical Society, 2024.

[36] D. Nychka, “Splines as local smoothers,” The Annals of Statistics, pp. 1175–1197, 1995.

## VIII. BIOGRAPHIES

Parsa Moradi is a Ph.D. candidate in the Department of Electrical and Computer Engineering at the University of Minnesota Twin Cities. He received his B.Sc. and M.Sc. degrees from the Department of Electrical Engineering, Sharif University of Technology, Tehran, Iran, in 2016 and 2019, respectively. His research interests include deep learning, generative AI, distributed machine learning, and AI for healthcare.

Hanzaleh Akbarinodehi is a Ph.D. candidate in the Department of Electrical and Computer Engineering at the University of Minnesota Twin Cities. He received his B.Sc. and M.Sc. degrees from the Department of Electrical Engineering, Sharif University of Technology, Tehran, Iran, in 2016 and 2018, respectively. His research interests include information theory and coding, blockchain technology, game theory, and decentralized systems.

Mohammad Ali Maddah-Ali (IEEE Fellow, 2023) is an Associate Professor at the University of Minnesota Twin Cities. He received his B.Sc. degree in Electrical Engineering from Isfahan University of Technology, his M.A.Sc. degree from the University of Tehran, and his Ph.D. in Electrical and Computer Engineering from the University of Waterloo, Canada, in 2007.

From 2007 to 2008, he was with the Wireless Technology Laboratories at Nortel Networks, Ottawa, ON, Canada. He then held a Postdoctoral Fellowship at the Department of Electrical Engineering and Computer Sciences, University of California, Berkeley, from 2008 to 2010. From September 2010 to September 2020, he served as a Communication Research Scientist at Nokia Bell Labs, NJ, USA.

Dr. Maddah-Ali is the recipient of several honors, including the NSERC Postdoctoral Fellowship (2007), the Best Paper Award at the IEEE International Conference on Communications (ICC) in 2014, the IEEE Communications Society and IEEE Information Theory Society Joint Paper Award in 2015, and the IEEE Information Theory Society Paper Award in 2016. He served as an Associate Editor for the IEEE Transactions on Information Theory (2019–2022) and as Lead Editor for the IEEE Journal on Selected Areas in Information Theory. He is currently a distinguished lecturer of the IEEE Information Theory Society.

## APPENDIX A

## ADDITIONAL FUNCTIONAL-ANALYTIC DETAILS

Definition 3 (Weak Derivatives and Vector-Valued Norms). A locally integrable scalar function v has a weak derivative w on $\Omega \subset \mathbb { R } { \mathrm { ~ i f ~ } }$

$$
\int _ { \Omega } { v ( t ) \phi ^ { \prime } ( t ) d t } = - \int _ { \Omega } { w ( t ) \phi ( t ) d t }\tag{124}
$$

for every infinitely differentiable test function $\phi$ with compact support in Ω. Weak derivatives of vector-valued functions are defined coordinate-wise.

Definition 4 $( L ^ { p } { \mathrm { - n o r m } } )$ . For a vector-valued function $\boldsymbol { u } = [ u _ { 1 } , \ldots , u _ { M } ] ^ { T }$ , the $L ^ { p _ { . } }$ norm on a domain $\Omega \subset \mathbb { R }$ is defined as follows:

$$
\| u \| _ { L ^ { p } ( \Omega ; \mathbb { R } ^ { M } ) } : = \left\{ \int _ { \Omega } \| u ( t ) \| _ { 2 } ^ { p } d t \right\} ^ { 1 / p } ,\tag{125}
$$

for $1 \leq p < \infty$ , and

$$
\| u \| _ { L ^ { \infty } ( \Omega ; \mathbb { R } ^ { M } ) } : = \operatorname { e s s } _ { z \in \Omega } \| u _ { i } ( z ) \| _ { 2 } ,\tag{126}
$$

when $p = \infty$

Definition 5 (Local Sobolev spaces). A function $g : \Omega \to \mathbb { R } ^ { M }$ belongs to local Sobolev space $\mathbb { W } _ { \mathrm { l o c } } ^ { r , p } \left( \Omega ; \mathbb { R } ^ { M } \right)$ if $g \in \mathbb { W } _ { \mathrm { l o c } } ^ { r , p } \left( \Omega ; \mathbb { R } ^ { M } \right)$ for every compact set $V \subset \Omega$

Lemma 6 (Theorem 8 [12]). $\mathbb { W } ^ { 2 , 2 } \left( \Omega ; \mathbb { R } ^ { M } \right) \subseteq \mathbb { W } ^ { 1 , 1 } \left( \Omega ; \mathbb { R } ^ { M } \right) \subseteq \mathbb { W } _ { l o c } ^ { 1 , 1 } \left( \Omega ; \mathbb { R } ^ { M } \right) .$

Definition 6 (Sobolev Space with Compact Support). Given bounded interval $\Omega = [ a , b ]$ , Sobolev order $r \in \mathbb N ,$ , and exponent $p \in [ 1 , \infty )$ , we define $W _ { 0 } ^ { r , p } ( \Omega ; \mathbb { R } ^ { M } )$ as the subspace of functions $g \in W ^ { r , p } ( \Omega ; \mathbb { R } ^ { M } )$ satisfying vanishing boundary conditions

$$
g ^ { ( j ) } ( a ) = 0 , \qquad \mathrm { ~ f o r ~ a l l ~ } j = 0 , 1 , \ldots , m - 1 .\tag{127}
$$

This space is usually equipped with the norm based on the highest-order derivative:

$$
\| g \| _ { W _ { 0 } ^ { r , p } ( \Omega ; \mathbb { R } ^ { M } ) } : = \| g ^ { ( m ) } \| _ { L ^ { p } ( \Omega ; \mathbb { R } ^ { M } ) } , \qquad 1 \leq p < \infty ,\tag{128}
$$

and for $p = \infty .$

$$
\| g \| _ { W _ { 0 } ^ { r , \infty } ( \Omega ; \mathbb { R } ^ { M } ) } : = \| g ^ { ( m ) } \| _ { L ^ { \infty } ( \Omega ; \mathbb { R } ^ { M } ) } .\tag{129}
$$

When $p = 2 .$ , this space is an RKHS, and we denote it by $\mathcal { H } _ { 0 } ^ { m } \left( \Omega ; \mathbb { R } ^ { M } \right)$

Remark 7 (Explicit kernel form). The scalar reproducing kernel of $\mathcal { H } _ { 0 } ^ { 2 } \left( \Omega ; \mathbb { R } \right)$ for $\Omega = [ 0 , 1 ]$ is

$$
\phi _ { 0 } ( s , t ) = \int _ { 0 } ^ { 1 } ( s - u ) _ { + } ( t - u ) _ { + } d u = { \frac { \operatorname* { m i n } ( s , t ) ^ { 2 } \left( 3 \operatorname* { m a x } ( s , t ) - \operatorname* { m i n } ( s , t ) \right) } { 6 } } ,\tag{130}
$$

and the reproducing kernel of $\mathcal { H } ^ { 2 } \left( \Omega ; \mathbb { R } \right)$ under the norm above is $\phi ( s , t ) = 1 + s t + \phi _ { 0 } ( s , t )$

## APPENDIX B

## PROOF OF LEMMAS

## A. Proof of Lemma 2

Define the smoothing-spline approximation error

$$
\begin{array} { r } { e ( z ) : = s _ { n , \lambda } [ h ] ( z ) - h ( z ) . } \end{array}\tag{131}
$$

Since $s _ { n , \lambda } [ h ] , h \in \mathcal { H } ^ { 2 } \left( \Omega ; \mathbb { R } ^ { m } \right)$ , it follows immediately that

$$
e \in \mathcal { H } ^ { 2 } \left( \Omega ; \mathbb { R } ^ { m } \right) .\tag{132}
$$

We first bound the supremum norm of $e ( \cdot )$ using its $L ^ { 2 } \left( \Omega ; \mathbb { R } ^ { m } \right)$ norm and the $L ^ { 2 } \left( \Omega ; \mathbb { R } ^ { m } \right)$ norm of its first derivative.

Using the smoothing-spline operator defined in (21), we have

$$
s _ { n , \lambda } [ h ] = \operatorname * { a r g m i n } _ { u \in \mathcal H ^ { 2 } ( \Omega ; \mathbb R ^ { m } ) } \left\{ \frac 1 n \sum _ { i = 1 } ^ { n } \lVert u ( t _ { i } ) - h ( t _ { i } ) \rVert _ { 2 } ^ { 2 } + \lambda \lVert u ^ { \prime \prime } \rVert _ { L ^ { 2 } ( \Omega ; \mathbb R ^ { m } ) } ^ { 2 } \right\} .\tag{133}
$$

For simplicity, let

$$
s : = s _ { n , \lambda } [ h ] .
$$

We first derive a useful orthogonality property of the smoothing-spline error. For an arbitrary direction $v \in \mathcal { H } ^ { 2 } \left( \Omega ; \mathbb { R } ^ { m } \right)$ define the one-dimensional function

$$
\Gamma _ { v } ( \varepsilon ) : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \lVert s ( t _ { i } ) + \varepsilon v ( t _ { i } ) - h ( t _ { i } ) \rVert _ { 2 } ^ { 2 } + \lambda \lVert s ^ { \prime \prime } + \varepsilon v ^ { \prime \prime } \rVert _ { L ^ { 2 } ( \Omega ; \mathbb R ^ { m } ) } ^ { 2 } , \qquad \varepsilon \in \mathbb { R } .\tag{134}
$$

Because $s ( \cdot )$ minimizes the smoothing-spline objective, $\varepsilon = 0$ minimizes $\Gamma _ { v } ( \varepsilon )$ . Therefore,

$$
\Gamma _ { v } ^ { \prime } ( 0 ) = 0 .\tag{135}
$$

Differentiating the two quadratic terms gives

$$
\Gamma _ { v } ^ { \prime } ( 0 ) = \frac { 2 } { n } \sum _ { i = 1 } ^ { n } \langle s ( t _ { i } ) - h ( t _ { i } ) , v ( t _ { i } ) \rangle + 2 \lambda \langle s ^ { \prime \prime } , v ^ { \prime \prime } \rangle _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } ,\tag{136}
$$

where

$$
\langle s ^ { \prime \prime } , v ^ { \prime \prime } \rangle _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } : = \int _ { 0 } ^ { 1 } \langle s ^ { \prime \prime } ( t ) , v ^ { \prime \prime } ( t ) \rangle d t .\tag{137}
$$

Recalling that $e = s - h$ and dividing by two, we obtain the first-order optimality condition

$$
\frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left. { e ( t _ { i } ) , v ( t _ { i } ) } \right. + \lambda \left. { s ^ { \prime \prime } , v ^ { \prime \prime } } \right. _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } = 0\tag{138}
$$

for every $v \in \mathcal { H } ^ { 2 } \left( \Omega ; \mathbb { R } ^ { m } \right)$ . This is the standard variational optimality condition for smoothing splines [15], [23].

Now fix an arbitrary vector $\mathbf { v } _ { 0 } \in \mathbb { R } ^ { m }$ and choose the constant perturbation

$$
v ( t ) \equiv \mathbf { v } _ { 0 } .\tag{139}
$$

Since $v ^ { \prime \prime } ( t ) = \mathbf { 0 }$ , condition (138) becomes

$$
0 = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \left. e ( t _ { i } ) , \mathbf { v _ { 0 } } \right. = \left. { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } e ( t _ { i } ) , \mathbf { v _ { 0 } } \right. .\tag{140}
$$

Because this equality holds for every $\mathbf { v } _ { 0 } \in \mathbb { R } ^ { m }$ , we must have

$$
\sum _ { i = 1 } ^ { n } e ( t _ { i } ) = \mathbf { 0 } .\tag{141}
$$

Equivalently, for every coordinate $\ell \in [ m ]$

$$
\sum _ { i = 1 } ^ { n } e _ { \ell } ( t _ { i } ) = 0 .\tag{142}
$$

We next show that every coordinate error $e _ { \ell }$ vanishes somewhere on [0, 1]. If $e _ { \ell } ( t _ { i } ) = 0$ for some i, we simply take $z _ { \ell } = t _ { i } .$ . Otherwise, all the values $\{ e _ { \ell } ( t _ { i } ) \} _ { i = 1 } ^ { n }$ are nonzero. In this case, (142) implies that they cannot all have the same sign. Hence, there exist two consecutive design points $t _ { i }$ and $t _ { i + 1 }$ such that

$$
e _ { \ell } ( t _ { i } ) e _ { \ell } ( t _ { i + 1 } ) < 0 .\tag{143}
$$

Since $e _ { \ell } \in H ^ { 2 } ( 0 , 1 ) \subset H ^ { 1 } ( 0 , 1 )$ , it has a continuous, absolutely continuous representative on [0, 1]. The intermediate value theorem therefore ensures the existence of a point

$$
z _ { \ell } \in [ t _ { i } , t _ { i + 1 } ]\tag{144}
$$

such that

$$
\begin{array} { r } { e _ { \ell } ( z _ { \ell } ) = 0 . } \end{array}\tag{145}
$$

Because $e _ { \ell } \in H ^ { 1 } ( 0 , 1 )$ ), the function $e _ { \ell } ^ { 2 }$ is absolutely continuous and satisfies

$$
\frac { d } { d t } e _ { \ell } ( t ) ^ { 2 } = 2 e _ { \ell } ( t ) e _ { \ell } ^ { \prime } ( t )\tag{146}
$$

for almost every $t \in ( 0 , 1 )$ . Hence, for every $z \in [ 0 , 1 ]$

$$
\begin{array} { l } { | e _ { \ell } ( z ) | ^ { 2 } = \displaystyle \big | e _ { \ell } ( z ) ^ { 2 } - e _ { \ell } ( z _ { \ell } ) ^ { 2 } \big | } \\ { \displaystyle \quad = 2 \left| \int _ { z _ { \ell } } ^ { z } e _ { \ell } ( t ) e _ { \ell } ^ { \prime } ( t ) d t \right| } \\ { \displaystyle \quad \leq 2 \int _ { 0 } ^ { 1 } | e _ { \ell } ( t ) | | e _ { \ell } ^ { \prime } ( t ) | d t } \\ { \displaystyle \quad \leq 2 \| e _ { \ell } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ) } \| e _ { \ell } ^ { \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ) } , } \end{array}\tag{147}
$$

where the last step follows from the Cauchy–Schwarz inequality.

Taking the supremum over z in (147), summing over $\ell \in [ m ]$ , and applying the Cauchy–Schwarz inequality to the coordinate index give

$$
\begin{array} { r l r } {  { \| e \| _ { L ^ { \infty } ( \Omega ; \mathbb { R } ^ { m } ) } ^ { 2 } \le \displaystyle \sum _ { \ell = 1 } ^ { m } \| e _ { \ell } \| _ { L ^ { \infty } ( \Omega ; \mathbb { R } ) } ^ { 2 } } } \\ & { } & { \le 2 \displaystyle \sum _ { \ell = 1 } ^ { m } \| e _ { \ell } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ) } \| e _ { \ell } ^ { \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ) } } \\ & { } & { \le 2 \| e \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } \| e ^ { \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } . } \end{array}\tag{148}
$$

Taking square roots yields the slightly sharper estimate

$$
\left\| e \right\| _ { L ^ { \infty } ( \Omega ; \mathbb { R } ^ { m } ) } \leq \sqrt { 2 } \left\| e \right\| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } ^ { 1 / 2 } \left\| e ^ { \prime } \right\| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } ^ { 1 / 2 } .\tag{149}
$$

In particular, since ${ \sqrt { 2 } } \leq 2 .$

$$
\left\| e \right\| _ { L ^ { \infty } ( \Omega ; \mathbb { R } ^ { m } ) } \leq 2 \| e \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } ^ { 1 / 2 } \| e ^ { \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } ^ { 1 / 2 } .\tag{150}
$$

Next, we bound the two quantities on the right-hand side of (150). Recall from Definition 1 that for any ordered design set $\mathcal { T } = \{ t _ { 1 } , \ldots , t _ { n } \}$ where $t _ { 1 } < \ldots < t _ { n }$ , the terms $\bar { \delta } _ { \tau }$ and $\underline { { \delta } } _ { \tau }$ denote the maximum and minimum consecutive spacings in $\tau ,$ respectively.

Authors in [34] have shown that, for a scalar target function $g \in H ^ { 2 } ( \Omega ; \mathbb { R } )$ and its smoothing-spline approximation $s _ { n , \lambda } [ g ]$ over $\tau$ , there exist constants $G _ { 0 } , G _ { 1 } > 0$ such that

$$
\| g - s _ { n , \lambda } [ g ] \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ) } ^ { 2 } \leq G _ { 0 } L _ { \mathcal { T } } \| g ^ { \prime \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ) } ^ { 2 }\tag{151}
$$

and

$$
\lVert ( g - s _ { n , \lambda } [ g ] ) ^ { \prime } \rVert _ { L ^ { 2 } ( \Omega ; \mathbb { R } ) } ^ { 2 } \leq G _ { 1 } L _ { \mathcal { T } } ^ { 1 / 2 } \left( 1 + \frac { L _ { \mathcal { T } } } { 1 6 } \right) ^ { 1 / 2 } \lVert g ^ { \prime \prime } \rVert _ { L ^ { 2 } ( \Omega ; \mathbb { R } ) } ^ { 2 } ,\tag{152}
$$

where

$$
L \tau : = p _ { 2 } \left( \frac { \bar { \delta } _ { T } } { \underline { { { \delta } } } _ { T } } \right) \frac { n \bar { \delta } _ { T } } { 4 } \lambda + D \bar { \delta } _ { T } ^ { 4 } .\tag{153}
$$

Here, $p _ { 2 }$ is a degree-two polynomial with positive coefficients, and $D > 0$ is an absolute constant.

The vector-valued smoothing-spline objective separates across its m output coordinates. Applying (151) and (152) to each pair $\left( h _ { \ell } , s _ { \ell } \right)$ and summing over $\ell \in [ m ]$ gives

$$
\| e \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } ^ { 2 } \leq G _ { 0 } L _ { \mathcal { T } } \| h ^ { \prime \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } ^ { 2 } ,\tag{154}
$$

$$
\| e ^ { \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } ^ { 2 } \leq G _ { 1 } L _ { \mathcal { T } } ^ { 1 / 2 } \left( 1 + \frac { L _ { \mathcal { T } } } { 1 6 } \right) ^ { 1 / 2 } \| h ^ { \prime \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } ^ { 2 } ,\tag{155}
$$

where $L _ { T }$ is the design factor in (153) for

$$
\mathcal { T } = \{ t _ { 1 } , \ldots , t _ { n } \} .\tag{156}
$$

For the equidistant design points $t _ { i } = ( i - 1 ) / ( n - 1 )$ , we have

$$
t _ { i + 1 } - t _ { i } = { \frac { 1 } { n - 1 } } , \qquad i \in [ n - 1 ] .\tag{157}
$$

Hence,

$$
\bar { \delta } _ { T } = \underline { { { \delta } } } _ { T } = \frac { 1 } { n - 1 } ,\tag{158}
$$

and therefore

$$
\begin{array} { l } { { { \cal L } _ { 7 } = p _ { 2 } ( 1 ) \displaystyle \frac { n } { 4 ( n - 1 ) } \lambda + \displaystyle \frac { D } { ( n - 1 ) ^ { 4 } } } } \\ { { \mathrm { ~  ~ { ~ \stackrel { ~ \scriptscriptstyle ~ ( a ) ~ } { ~ \leq ~ } } ~ } } } \\ { { \mathrm { ~  ~ { ~ \sum ~ } ~ } } } \\ { { \mathrm { ~  ~ { ~ \sum ~ } ~ } } } \\ { { \mathrm { ~  ~ { ~ \leq ~ } ~ } C _ { L } \left( \lambda + n ^ { - 4 } \right) , } } \end{array}\tag{159}
$$

where $C _ { L } > 0$ is independent of n and λ and (a) follows from $\textstyle { \frac { n } { n - 1 } } \leq 2$ and $\frac { 1 } { ( n - 1 ) ^ { 4 } } \leq 1 6 n ^ { - 4 }$ , for $n \geq 2$ Combining (148), (154), and (155), we obtain

$$
\begin{array} { r l } & { \displaystyle \| e \| _ { L ^ { \infty } ( \Omega ; \mathbb { R } ^ { m } ) } ^ { 2 } \leq 2 \| e \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } \| e ^ { \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } } \\ & { \qquad \leq 2 \left( G _ { 0 } G _ { 1 } \right) ^ { 1 / 2 } L _ { \mathcal { T } } ^ { 3 / 4 } \left( 1 + \frac { L _ { \mathcal { T } } } { 1 6 } \right) ^ { 1 / 4 } \| h ^ { \prime \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } ^ { 2 } . } \end{array}\tag{160}
$$

Since $0 < \lambda \leq 1$ , it follows from (159) that

$$
L _ { T } \leq 2 C _ { L } .\tag{161}
$$

Consequently,

$$
\left( 1 + \frac { L \tau } { 1 6 } \right) ^ { 1 / 4 } \leq \left( 1 + \frac { 2 C _ { L } } { 1 6 } \right) ^ { 1 / 4 } ,\tag{162}
$$

which is independent of $n , \lambda ,$ and h. Absorbing this factor and the constants $G _ { 0 } , G _ { 1 }$ , and $C _ { L }$ into a single constant gives

$$
\begin{array} { r l r } & { } & { \| e \| _ { L ^ { \infty } ( \Omega ; \mathbb { R } ^ { m } ) } ^ { 2 } \leq C _ { \mathrm { s p } } \left( \lambda + n ^ { - 4 } \right) ^ { 3 / 4 } \| h ^ { \prime \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } ^ { 2 } } \\ & { } & { \qquad \leq C _ { \mathrm { s p } } \left( \lambda ^ { 3 / 4 } + n ^ { - 3 } \right) \| h ^ { \prime \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } ^ { 2 } , } \end{array}\tag{163}
$$

where the last inequality follows from $( a + b ) ^ { 3 / 4 } \leq a ^ { 3 / 4 } + b ^ { 3 / 4 }$ for $a , b \geq 0$

Finally, using $e = s _ { n , \lambda } [ h ] - h$ , we conclude that

$$
\operatorname* { s u p } _ { z \in \Omega } \| s _ { n , \lambda } [ h ] ( z ) - h ( z ) \| _ { 2 } ^ { 2 } \leq C _ { \mathrm { s p } } \left( \lambda ^ { 3 / 4 } + n ^ { - 3 } \right) \| h ^ { \prime \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } ^ { 2 } .\tag{164}
$$

This proves the lemma.

## B. Proof of Lemma 3

To bound the second derivative of the composite function, we make use of the following Gagliardo–Nirenberg-type interpolation inequality on bounded intervals.

Lemma 7 ( [35, Theorem 7.34]). Let $\Omega \subseteq \mathbb { R }$ be an open interval and let $g \in \mathbb { W } _ { l o c } ^ { 1 , 1 } \left( \Omega ; \mathbb { R } ^ { M } \right)$ . Assume $1 \leq p , v , r \leq \infty$ and $r \geq v .$ . Then, for every $0 < \ell < \mathrm { L e b } ( \Omega )$

$$
\| g \| _ { L ^ { r } ( \Omega ; \mathbb { R } ^ { M } ) } \leq \ell ^ { \frac { 1 } { r } - \frac { 1 } { v } } \| g \| _ { L ^ { v } ( \Omega ; \mathbb { R } ^ { M } ) } + \ell ^ { 1 - \frac { 1 } { p } + \frac { 1 } { r } } \| g ^ { \prime } \| _ { L ^ { p } ( \Omega ; \mathbb { R } ^ { M } ) } .\tag{165}
$$

We now proceed with the proof of Lemma 3. Let $g : = f \circ u = [ g _ { 1 } , \dotsc , g _ { m } ] ^ { \intercal }$ , where $u = [ u _ { 1 } , \ldots , u _ { d } ] ^ { \top } \in$ $\mathcal { H } ^ { 2 } \left( \Omega ; \mathbb { R } ^ { d } \right)$ . By the second-order chain rule for Sobolev functions, for each $j \in [ m ]$ and almost every $z \in \Omega$ , we have

$$
g _ { j } ^ { \prime \prime } ( z ) = u ^ { \prime } ( z ) ^ { \top } D ^ { 2 } f _ { j } ( u ( z ) ) u ^ { \prime } ( z ) + \nabla f _ { j } ( u ( z ) ) ^ { \top } u ^ { \prime \prime } ( z ) .\tag{166}
$$

Applying the triangle inequality, the Cauchy–Schwarz inequality, and the definition of the operator norm, together with the regularity bounds in Assumption (A2), yields

$$
\begin{array} { r l } & { | g _ { j } ^ { \prime \prime } ( z ) | \leq \| D ^ { 2 } f _ { j } ( u ( z ) ) \| _ { \mathrm { o p } } \| u ^ { \prime } ( z ) \| _ { 2 } ^ { 2 } + \| \nabla f _ { j } ( u ( z ) ) \| _ { 2 } \| u ^ { \prime \prime } ( z ) \| _ { 2 } } \\ & { \qquad \leq H _ { j } \| u ^ { \prime } ( z ) \| _ { 2 } ^ { 2 } + G _ { j } \| u ^ { \prime \prime } ( z ) \| _ { 2 } . } \end{array}\tag{167}
$$

Squaring both sides and using the elementary inequality $( a + b ) ^ { 2 } \leq 2 a ^ { 2 } + 2 b ^ { 2 }$ , we obtain

$$
| g _ { j } ^ { \prime \prime } ( z ) | ^ { 2 } \leq 2 H _ { j } ^ { 2 } \| u ^ { \prime } ( z ) \| _ { 2 } ^ { 4 } + 2 G _ { j } ^ { 2 } \| u ^ { \prime \prime } ( z ) \| _ { 2 } ^ { 2 } .\tag{168}
$$

Integrating over Ω and summing over $j \in [ m ]$ , while recalling that $\begin{array} { r } { H _ { f } ^ { 2 } = \sum _ { j = 1 } ^ { m } H _ { j } ^ { 2 } } \end{array}$ and $\begin{array} { r } { G _ { f } ^ { 2 } = \sum _ { j = 1 } ^ { m } G _ { j } ^ { 2 } } \end{array}$ , gives

$$
\| g ^ { \prime \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } ^ { 2 } = \sum _ { j = 1 } ^ { m } \int _ { \Omega } | g _ { j } ^ { \prime \prime } ( z ) | ^ { 2 } d z \leq 2 H _ { f } ^ { 2 } \int _ { \Omega } \| u ^ { \prime } ( z ) \| _ { 2 } ^ { 4 } d z + 2 G _ { f } ^ { 2 } \| u ^ { \prime \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 } .\tag{169}
$$

Next, we bound the quartic integral $\textstyle \int _ { \Omega } \lVert u ^ { \prime } ( z ) \rVert _ { 2 } ^ { 4 }$ dz using Lemma 7. First, observe that by Minkowski’s inequality in $L ^ { 2 } \left( \Omega ; \mathbb { R } \right)$ ,

$$
\begin{array} { r l r } {  { \int _ { \Omega } \| u ^ { \prime } ( z ) \| _ { 2 } ^ { 4 } d z = \int _ { \Omega } ( \sum _ { i = 1 } ^ { d } \lvert u _ { i } ^ { \prime } ( z ) \rvert ^ { 2 } ) ^ { 2 } d z = \| \sum _ { i = 1 } ^ { d } ( u _ { i } ^ { \prime } ) ^ { 2 } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ) } ^ { 2 } } } \\ & { } & { \leq ( \displaystyle \sum _ { i = 1 } ^ { d } \| ( u _ { i } ^ { \prime } ) ^ { 2 } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ) } ) ^ { 2 } = ( \displaystyle \sum _ { i = 1 } ^ { d } \| u _ { i } ^ { \prime } \| _ { L ^ { 4 } ( \Omega ; \mathbb { R } ) } ^ { 2 } ) ^ { 2 } . } \end{array}\tag{170}
$$

Since $u \in \mathcal { H } ^ { 2 } \left( \Omega ; \mathbb { R } ^ { d } \right)$ , each coordinate $u _ { i }$ satisfies

$$
u _ { i } ^ { \prime } \in \mathbb { W } ^ { 1 , 1 } \left( \Omega ; \mathbb { R } ^ { d } \right) \stackrel { \mathrm { ( a ) } } { \subseteq } \mathbb { W } _ { \mathrm { l o c } } ^ { 1 , 1 } \left( \Omega ; \mathbb { R } ^ { d } \right) ,\tag{171}
$$

where (a) follows from Lemma 6.

Fix any $\ell \in ( 0 , \mathrm { L e b } ( \Omega ) )$ and apply Lemma 7 to $u _ { i } ^ { \prime }$ with $r = 4$ and $p = v = 2 \ /$

$$
\begin{array} { r l } & { \| u _ { i } ^ { \prime } \| _ { L ^ { 4 } ( \Omega ; \mathbb { R } ) } \leq \ell ^ { - 1 / 4 } \| u _ { i } ^ { \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ) } + \ell ^ { 3 / 4 } \| u _ { i } ^ { \prime \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ) } } \\ & { \qquad \leq C _ { \ell } \left( \| u _ { i } ^ { \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ) } + \| u _ { i } ^ { \prime \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ) } \right) , } \end{array}\tag{172}
$$

where $C _ { \ell } : = \operatorname* { m a x } \{ \ell ^ { - 1 / 4 } , \ell ^ { 3 / 4 } \}$ (for instance, since $\mathrm { L e b } ( \Omega ) = 1$ , choosing $\ell = 1 / 2$ gives $C _ { \ell } = 2 ^ { 1 / 4 } )$ . Squaring both sides and applying $( a + b ) ^ { 2 } \leq 2 a ^ { 2 } + 2 b ^ { 2 }$ yields

$$
\| u _ { i } ^ { \prime } \| _ { L ^ { 4 } ( \Omega ; \mathbb { R } ) } ^ { 2 } \leq 2 C _ { \ell } ^ { 2 } \left( \| u _ { i } ^ { \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ) } ^ { 2 } + \| u _ { i } ^ { \prime \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ) } ^ { 2 } \right) .\tag{173}
$$

Summing over $i \in [ d ]$ gives

$$
\begin{array} { r l } { \displaystyle \sum _ { i = 1 } ^ { d } \| u _ { i } ^ { \prime } \| _ { L ^ { 4 } ( \Omega ; \mathbb { R } ) } ^ { 2 } \leq 2 C _ { \ell } ^ { 2 } \left( \| u ^ { \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 } + \| u ^ { \prime \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 } \right) } & { } \\ { \displaystyle } & { \overset { \mathrm { ( a ) } } { \leq } 2 C _ { \ell } ^ { 2 } \| u \| _ { \mathbb { W } ^ { 2 , 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 } } \\ { \displaystyle } & { \overset { \mathrm { ( b ) } } { \leq } 2 C _ { \ell } ^ { 2 } \theta _ { \mathrm { s o b } } ^ { 2 } \| u \| _ { \mathcal { H } ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 } , } \end{array}\tag{174}
$$

where (a) follows from $\| u ^ { \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 } + \| u ^ { \prime \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 } \leq \| u \| _ { \mathbb { W } ^ { 2 , 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 }$ and (b) comes from the equivalency of $\lVert \cdot \rVert _ { \mathbb { W } ^ { 2 , 2 } ( \Omega ; \mathbb { R } ^ { d } ) }$ and $\| \cdot \| _ { \mathcal { H } } ^ { 2 } \left( \Omega ; \mathbb { R } ^ { d } \right)$ with a factor $\theta _ { \mathrm { s o b } }$ (see Section III-A1 for more details).

Substituting this bound into (170), we obtain

$$
\int _ { \Omega } \lVert u ^ { \prime } ( z ) \rVert _ { 2 } ^ { 4 } d z \leq C _ { \ell } ^ { 4 } \theta _ { \mathrm { s o b } } ^ { 4 } \lVert u \rVert _ { \mathcal { H } ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 4 } .\tag{175}
$$

Finally, substituting (175) and the bound $\| u ^ { \prime \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 } \leq \| u \| _ { \mathcal { H } ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 }$ into (169) yields

$$
\begin{array} { r l } & { \| g ^ { \prime \prime } \| _ { L ^ { 2 } ( \Omega ; \mathbb { R } ^ { m } ) } ^ { 2 } \le 8 C _ { \ell } ^ { 4 } \theta _ { \mathrm { s o b } } ^ { 4 } H _ { f } ^ { 2 } \| u \| _ { \mathcal { H } ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 4 } + 2 G _ { f } ^ { 2 } \| u \| _ { \mathcal { H } ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 } } \\ & { \qquad \le 2 \operatorname* { m a x } \{ C _ { \ell } ^ { 4 } \theta _ { \mathrm { s o b } } ^ { 4 } , 1 \} \operatorname* { m a x } \{ H _ { f } ^ { 2 } , G _ { f } ^ { 2 } \} \left( \| u \| _ { \mathcal { H } ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 } + 4 \| u \| _ { \mathcal { H } ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 4 } \right) } \\ & { \qquad = C _ { \mathrm { c } } \operatorname* { m a x } \{ H _ { f } ^ { 2 } , G _ { f } ^ { 2 } \} \psi \left( \| u \| _ { \mathcal { H } ^ { 2 } ( \Omega ; \mathbb { R } ^ { d } ) } ^ { 2 } \right) , } \end{array}\tag{176}
$$

where $C _ { \mathrm { c } } : = 2 \operatorname* { m a x } \{ C _ { \ell } ^ { 4 } \theta _ { \mathrm { s o b } } ^ { 4 } , 1 \}$ . This completes the proof.

## C. Proof of Lemma 4

Using | sin v| ≤ 1 and $| \cos v | \le 1$ , we have

$$
| \Phi ( u , v ) | \leq 4 e ^ { - u } , \qquad u \geq 0 .\tag{177}
$$

Applying this bound to (76) gives

$$
\begin{array} { l } { | K _ { \lambda } ( x , t ) | \le \displaystyle \frac { 1 } { \sqrt { 2 } \lambda ^ { 1 / 4 } } e ^ { - \frac { | x - t | } { \sqrt { 2 } \lambda ^ { 1 / 4 } } } + \frac { 4 } { \sqrt { 2 } \lambda ^ { 1 / 4 } } \left( e ^ { - \frac { x + t } { \sqrt { 2 } \lambda ^ { 1 / 4 } } } + e ^ { - \frac { 2 - x - t } { \sqrt { 2 } \lambda ^ { 1 / 4 } } } \right) } \\ { \displaystyle \qquad \le \frac { 9 } { \sqrt { 2 } } \lambda ^ { - 1 / 4 } , } \end{array}\tag{178}
$$

which proves (77).

## D. Proof of Lemma 5

To establish Lemma 5, we first relate the equivalent kernel $K _ { \lambda }$ to the continuous Green’s function $G _ { \lambda }$ , and then bound the discrepancy between $G _ { \lambda }$ and its finite-sample counterpart $G _ { N , \lambda }$

Lemma 8 ( [31, Theorem 4.1]). There exists a constant $C _ { \mathrm { M } } > 0$ such that for every $\lambda \in ( 0 , 1 ]$

$$
\operatorname* { s u p } _ { x , t \in [ 0 , 1 ] } | K _ { \lambda } ( x , t ) - G _ { \lambda } ( x , t ) | \leq C _ { \mathrm M } \lambda ^ { - 1 / 4 } \exp \left( - \frac { 1 } { \sqrt { 2 } } \lambda ^ { - 1 / 4 } \right) .\tag{179}
$$

Next, we relate the continuous Green’s function $G _ { \lambda }$ to the finite-sample smoothing spline weight function $G _ { N , \lambda }$

Lemma 9 ( [36, Theorem 2.1]). Under equidistant design points, there exist constants $\mu _ { 0 } , \mu _ { 1 } , \mu _ { 2 } > 0$ such that whenever $N \lambda ^ { 1 / 4 } > \mu _ { 0 }$ , we have for all $x , t \in [ 0 , 1 ]$

$$
| G _ { N , \lambda } ( x , t ) - G _ { \lambda } ( x , t ) | \leq \frac { \mu _ { 0 } \mu _ { 1 } N ^ { - 1 } \lambda ^ { - 1 / 4 } } { 1 - \mu _ { 0 } N ^ { - 1 } \lambda ^ { - 1 / 4 } } \lambda ^ { - 1 / 4 } \exp \left( { - \mu _ { 2 } | x - t | \lambda ^ { - 1 / 4 } } \right) .\tag{180}
$$

We now combine Lemmas 8 and 9 to complete the proof. Set $c _ { 0 } : = 2 \mu _ { 0 } > 0 .$ . Since $\lambda _ { \mathrm { d } } \in ( 0 , 1 ]$ and $N \lambda _ { \mathrm { d } } ^ { 1 / 4 } > c _ { 0 }$ we have $N \lambda _ { \mathrm { d } } ^ { 1 / 4 } > 2 \mu _ { 0 } > \mu _ { 0 }$ . Therefore, the condition of Lemma 9 holds. Moreover, we have

$$
\mu _ { 0 } N ^ { - 1 } \lambda _ { \mathrm { d } } ^ { - 1 / 4 } < \frac { 1 } { 2 } ,\tag{181}
$$

which implies $1 - \mu _ { 0 } N ^ { - 1 } \lambda _ { \mathrm { d } } ^ { - 1 / 4 } > 1 / 2$ . Consequently, the prefactor in (180) is bounded above by

$$
\frac { \mu _ { 0 } \mu _ { 1 } N ^ { - 1 } \lambda _ { \mathrm { d } } ^ { - 1 / 4 } } { 1 - \mu _ { 0 } N ^ { - 1 } \lambda _ { \mathrm { d } } ^ { - 1 / 4 } } < \frac { \mu _ { 1 } / 2 } { 1 / 2 } = \mu _ { 1 } .\tag{182}
$$

By the triangle inequality, for all $x , t \in [ 0 , 1 ]$

$$
\begin{array} { r } { | G _ { N , \lambda _ { 4 } } ( x , t ) - K _ { \lambda _ { 4 } } ( x , t ) | \leq | G _ { N , \lambda _ { 4 } } ( x , t ) - G _ { \lambda _ { 4 } } ( x , t ) | + | G _ { \lambda _ { 4 } } ( x , t ) - K _ { \lambda _ { 4 } } ( x , t ) | . } \end{array}\tag{183}
$$

Applying Lemmas 8 and 9 together with (182) to (183) yields

$$
\begin{array} { r l r } {  { \vert G _ { N , \lambda _ { d } } ( x , t ) - K _ { \lambda _ { d } } ( x , t ) \vert \leq \frac { \mu _ { 0 } \mu _ { 1 } N ^ { - 1 } \lambda _ { \mathrm { d } } ^ { - 1 / 4 } } { 1 - \mu _ { 0 } \lambda _ { \mathrm { d } } ^ { - 1 } \lambda _ { \mathrm { d } } ^ { - 1 / 4 } } \lambda _ { \mathrm { d } } ^ { - 1 / 4 } \exp ( - \mu _ { 2 } \vert x - t \vert \lambda _ { \mathrm { d } } ^ { - 1 / 4 } ) + C _ { \mathsf { M } } \lambda _ { \mathrm { d } } ^ { - 1 / 4 } \exp ( - \frac { 1 } { \sqrt { 2 } } \lambda _ { \mathrm { d } } ^ { - 1 / 4 } ) } } \\ & { } & { \leq \mu _ { 1 } \lambda _ { \mathrm { d } } ^ { - 1 / 4 } \exp ( - \mu _ { 2 } \vert x - t \vert \lambda _ { \mathrm { d } } ^ { - 1 / 4 } ) + C _ { \mathsf { M } } \lambda _ { \mathrm { d } } ^ { - 1 / 4 } \exp ( - \frac { 1 } { \sqrt { 2 } } \lambda _ { \mathrm { d } } ^ { - 1 / 4 } ) . \qquad ( \mathsf { l } \otimes \lambda _ { \mathrm { d } } ^ { - 1 / 4 } ) } \end{array}\tag{4}
$$

Defining $C _ { W } : = \mu _ { 1 }$ and $\kappa : = \mu _ { 2 }$ gives (78) and completes the proof.