# Dynamics as Code: On Model Compression via Dynamic System

Fan Gao, Wei Su\*, Juntong Fan\*, Renfeng Peng, Hongyu Liu, Jinqiao Duan, Feng-Lei Fan

Abstract—The escalating size of pretrained neural networks has rendered model compression a prerequisite for deployment under stringent memory and compute constraints. With the irrational winding as an example, earlier work introduced a dynamic system (DS) paradigm that reconceptualizes compression as compact weight representation: high-dimensional parameters are encoded by the index of a trajectory produced by a dynamic system, from which the vector is recovered during decompression. This mechanism is fundamentally distinct from pruning, quantization, knowledge distillation, and low-rank decomposition. Along this direction, we prove that under a Diophantine condition, a finite trajectory of $M = { \cal O } ( \epsilon ^ { - ( d + \nu ) } )$ states in the irrational winding constitutes an ϵ-net over the d-dimensional weight space, thereby linking state resolution, decompression error, and compression ratio in a predictable manner. Furthermore, we propose a generalized DS-based model compression framework by unifying four DS families—spacefilling curves (Hilbert, Peano, Morton/Z-order, Snake), chaotic systems (Lorenz), congruential and pseudo-random generators (LCG, PCG), and low-discrepancy sequences (Halton). Also, we introduce the KD-tree and coordinate-template acceleration to scale to large models as well as outlier identification to control the error. Experiments on ResNet-18 and Qwen2.5-1.5B/Qwen1.5- 7B validate that DS-based compression achieves competitive compression ratios without post-hoc retraining, with controllable decompression error and flexible state-space design, establishing it as a principled and practical compression approach.

Index Terms—Model compression, dynamic systems, neural networks, parameter representation.

## I. INTRODUCTION

In the past decade, the emergence of large-scale pretrained models creates substantial deployment challenges, particularly in resource-constrained settings such as workstations, smartphones, and wearables [1]. Loading LLaMA-2 70B weights in FP16 precision requires approximately 140 GB, nearly six times the memory capacity of an NVIDIA GeForce RTX 4090 [2]. The development of multimodal world models [3], [4] is likely to further worsen this issue due to the introduction of vision and speech encoders alongside extended context lengths. Consequently, model compression has emerged as a critical strategy for serving large models, aiming to minimize storage requirements without compromising task performance [5]. This is quite different from the pre-large-model era, where model sizes remained modest, and the incentives for compression were comparatively weak.

Classical model compression methods fall into four primary categories: pruning, quantization, knowledge distillation, and low-rank decomposition [6]: i) Pruning removes unimportant parameters or structural components [7], which pursues the parameter sparsity within a model; ii) Quantization reduces the numerical precision of weights and activations (e.g., FP32 → INT8/INT4) to decrease storage and bandwidth requirements [8]; iii) Knowledge distillation transfers knowledge from a large teacher model to a compact student model [9], [10], which typically requires additional training, task data, and careful teacher–student design; iv) Low-rank factorization approximates weight matrices or tensors using the low-rank decomposition [11]. The achievable compression depends on the intrinsic rank of weights, and can vary substantially across layers, architectures, and tasks. In essence, classical methods exploit different forms of redundancy in the original parameter space such as sparsity, reduced numerical precision, or lowrank structure.

Recently, inspired by the biological genomic bottleneck [12], a novel basic mechanism for model compression was discovered and engineered in [13], which uses the ergodicity principle that a low-dimensional dynamic system can fill the high-dimensional space. Mathematically, given target weights $\textbf { w } = ~ [ w _ { 1 } , \dots , w _ { N } ] ^ { \top }$ , [13] employs irrational winding to construct a function $h ( \theta ; n )$ to approximate each weight such that $w _ { n } \approx h ( \theta ; n )$ , where the number of trajectory parameters θ is significantly smaller than the total number of weights $( \dim ( \theta ) \ll N )$ . This turns model compression into a problem of parsimonious parameter representation. Model compression via dynamic systems is not merely incremental. Instead, it shifts away from decades-old techniques by opening new research directions at the intersection of dynamic systems and efficient deep learning, while also being practically viable for billion-parameter models in resource-constrained settings. Introducing dynamic systems carves out a new research direction in model compression. Nevertheless, several critical theoretical and methodological challenges must be addressed to fully realize the potential of this approach:

Theoretically, the relationship between the compression ratio and the error remains unexplored. Due to the nonlocality of the irrational winding, to achieve the same level of error, how large θ needs to be at least is intractable. Accordingly, the compression ratio is unpredictable. This is a fundamental question in this domain, and solving it will transform the dynamic system from a heuristic into a principled, predictable tool. For example, one can predict to what extent increasing θ will surely reduce the approximation error, thereby providing the lower bound for the compression ratio and facilitating allocating bits across layers fairly. Methodologically, since model compression via dynamic systems is general, can we enrich the armory of this kind by effectively engineering other dynamic systems? The dynamic system (DS) theory provides a broader design space beyond the irrational winding. Instead of treating irrational winding as the default choice, a broader design space of transformation-induced compression mappings characterized by different coverage, locality, and indexability properties should be examined.

In this manuscript, we theoretically establish both the asymptotic feasibility and the practical optimizability of DSbased compression. First, the covering guarantee (Theorem 1) establishes that, under Diophantine conditions, M = $O ( \epsilon ^ { - ( d + \nu ) } )$ trajectory states form an ϵ-net over the weight space, directly linking the compression ratio to the reconstruction error through the covering radius. This justifies why a low-dimensional scalar can approximate high-dimensional weights with bounded distortion. Second, the optimization procedure addresses the discontinuity of the irrational winding by partitioning the domain at all breakpoints; within each continuous segment, the loss reduces to a convex quadratic with a closed-form minimizer, enabling exact global minimization. Together, these results transform the theoretical density guarantee into a practical, deterministic algorithm with a predictable error profile, ensuring that the mechanism is not merely an existence statement. Methodologically, we expand this novel framework from a single ergodic construction to a broader toolbox, i.e., space-filling curves, chaotic dynamics, pseudo-random sequences, and low-discrepancy sequences, which cover representative DS families:

• Space-filling and grid traversal mappings (Hilbert [14], Peano [15], Morton/Z-order [16], and Snake traversal): deterministic mappings from compact indices to high-dimensional grid points, offering different locality-preserving structures for grouped weight representation.

• Chaotic-system-based mappings (Lorenz system [17]): nonlinear trajectory-based mappings that provide aperiodic coverage of the weight representation space beyond regular grid traversal.

• Congruential and pseudo-random generator mappings (Linear Congruential Generator (LCG) [18] and Permuted Congruential Generator (PCG)): efficient deterministic point generators that produce reproducible pseudorandom sequences with simple indexable structures for scalable compression.

• Quasi-Monte Carlo sequences (Halton [19]): deterministic low-discrepancy sequences that provide uniform coverage for compact weight representation.

These families differ in coverage guarantees, computational cost, and approximation accuracy, thereby leading to different compression ratios, speed, and decompression fidelity. The central problem in developing DS-based compression is how to enable fast, accurate, and scalable compression. We introduce effective engineering techniques such as KD-tree to speed up the compression speed and outlier identification to control the error. Particularly, for the space-filling curves, we design the coordinate template method to further accelerate the compression relative to the KD-tree search.

In brief, our contributions are threefold:

• We lay a solid theoretical foundation for model compression based on dynamic systems by illustrating the asymptotic feasibility. To the best of our knowledge, it is the first theoretical result in the direction of DS-based model compression.

• We build a general framework for model compression via dynamic systems, engage mainstream dynamic systems as the components, and design engineering techniques to make this framework scalable to large models.

• Systematic and comprehensive experiments confirm that DS-based model compression can deliver competitive performance on ResNet and large models such as Qwen2.5-1.5B and Qwen1.5-7B.

## II. RELATED WORK

## A. Classical Model Compression Methods

Model compression aims to minimize storage requirements without incurring substantial performance degradation. Established techniques include pruning, quantization, knowledge distillation, and low-rank decomposition.

Pruning compresses neural networks by removing redundant weights, channels, or structural components. Early methods such as Optimal Brain Damage [20] and Optimal Brain Surgeon [21] estimate parameter saliency using secondorder information, while structured pruning improves hardware compatibility by uniformly removing filters or channels rather than isolated weights [22]. For large language models, SparseGPT [7] formulates pruning as a sparse regression problem for one-shot compression. Wanda [23] combines weight magnitude with activation statistics to guide pruning; it requires a small calibration set to estimate activation statistics. Recent work further studies structured and task-sensitive LLM pruning, including layer-level pruning [24] and reasoningaware pruning based on chain-of-thought reconstruction [25]. Despite these advances, practical speedups from sparsity often depend on specialized sparse kernels, storage formats, and hardware support [26].

Quantization reduces storage and memory bandwidth requirements by adopting lower numerical precision for weights and activations. For large models, post-training quantization (PTQ) has gained prominence due to its convenience. ZeroQuant [27] provides fine-grained weight and activation quantization for large Transformers, while SQuant [28] introduces a data-free PTQ through a diagonal-Hessian approximation. SmoothQuant [29] applies activation-to-weight smoothing transformations to mitigate activation outliers.

For weight-only compression, GPTQ [8] employs approximate Hessian-based layer-wise reconstruction to minimize the error; AWQ [30] protects salient weight channels through activation awareness. SpQR [31] retains quantization-sensitive outliers in higher precision, while QuIP [32] improves low-bit quantization through incoherence processing.

Knowledge Distillation transfers knowledge from a large teacher model to a compact student model [9], with extensions to intermediate feature matching [33] and relation transfer [34]. For large language models, MiniLLM [10] adopts reverse KL divergence to improve distribution matching, while recent work extends knowledge distillation to pre-training [35] and improves teacher–student alignment through speculative training or score-matching objectives [36], [37]. However, distillation typically requires additional training and task data, and its effectiveness depends on student capacity, teacher quality, and teacher–student distribution alignment [38].

Low-Rank Decomposition approximates weight matrices using products of lower-dimensional factors [11], while tensor decomposition extends this idea to higher-order parameter structures [39]. Parameter-efficient fine-tuning methods such as LoRA [40] further suggest that large-model adaptations often lie in low-dimensional subspaces. Recent LLM compression methods increasingly use low-rank components as residual or subspace corrections, including low-precision and lowrank decomposition [41], low-rank approximation combined with quantization and sparsity [42], and low-rank residuals for mixed-precision quantization [43]. However, the achievable compression depends on the intrinsic rank structure of the weights, and can vary substantially across layers, architectures, and tasks.

Here, Our work falls under the umbrella of model compression based on dynamic systems, which is orthogonal to these classical methods by representing model weights through a compact representation.

## B. Model Compression via dynamic systems

Motivated by ergodic theory [44], Fan et al. [13] introduce a distinct model compression paradigm by employing irrational winding to generate dense trajectories over the parameter domain and report approximately 2.6× compression on LLaMA2-7B. [45] applies the irrational winding-based model compression to SAM and its variants, demonstrating its applicability to large segmentation foundation models. The method introduces HyperLinear, a dedicated operator that fuses decompression with matrix multiplication to improve the inference efficiency of compressed SAMs. COLI [46] incorporates the irrational winding-based model compression into an INR-based large-image compression framework, where NeRV represents images as neural network parameters, and DS is used to further compress the resulting model weights. This shows how DS can serve as a plug-and-play post-training parameter-compression module beyond standard classification and language-model settings.

However, many dynamic systems conform to ergodicity. Irrational winding provides a mathematically tractable instantiation, and the existence of better dynamic systems is possible. This leaves a broad design space of DS-induced compression mappings largely unexplored. Here, we analyze and compare a broad family of dynamic systems under a unified theoretical and empirical framework as illustrated in Figure 1.

![](images/7bf2654366b17766c134b128026db64a7470209b2210701e0cc7ee6095c8fecd.jpg)  
Fig. 1. Representative geometries generated by eight evaluated mapping families. Hilbert, Peano, Morton/Z-order, and Snake produce structured lattice traversals with different locality characteristics and traversal continuity. Lorenz generates a non-uniform attractor-induced distribution concentrated on a low-dimensional manifold. LCG and PCG produce deterministic pseudorandom point sets with different correlation structures, while Halton generates low-discrepancy coverage with reduced large-scale gaps. All methods are visualized after normalization to the same local domain.

## C. Dynamic Systems and Deterministic Mappings

Dynamic systems describe how the state of a system evolves according to a fixed rule. In mathematical terms, an evolution map $\Phi ( t , \mathbf { x } ) : T \times \mathcal { X }  \mathcal { X }$ maps an initial state $\mathbf { x } \in \mathcal { X }$ in a phase space X to its future state under the evolution parameter t. This formulation provides a compact way to generate structured trajectories or point sets. Common dynamic systems include space-filling curves, chaotic systems, pseudo-random generators, and low-discrepancy sequences. As illustrated in Figure 1, i) space-filling curves, such as Hilbert, Peano, and Morton/Z-order mappings, define structured traversals between the one-dimensional index and multi-dimensional coordinates, often with locality-preserving behaviors. Snake ordering is a simpler grid traversal based on deterministic scanning. ii) Pseudo-random generators, including linear congruential generators (LCG) and permuted congruential generators (PCG), produce deterministic sequences with compact mathematical descriptions. iii) Chaotic systems such as the Lorenz system generate trajectory-based point sets with nontrivial phase-space structure. iv) Low-discrepancy sequences, such as Halton, provide more uniform domain coverage than random samples, and are widely used in quasi-Monte Carlo methods (QMC).

These dynamic systems induce a state set with different locality, coverage, regularity, and indexing properties, making them natural candidates for studying DS-based model compression. Our work marries the dynamic system with model compression by exploring the possibility of using these dynamic systems as a model compression tool, which extends the application boundary of dynamic systems into AI, and shall bring many new research opportunities.

## III. THEORY

Proposition 1 ( [47]). Suppose $a _ { 1 } , \cdots , a _ { N }$ are rationally independent, for any given set $\{ w _ { n } \} _ { n = 1 } ^ { N } \subseteq [ 0 , 1 ]$ and $\epsilon > 0 \AA$ there exists $\theta ^ { \ast } \in \mathbb { R } _ { + }$ such that

$$
| w _ { n } - \tau ( \theta ^ { * } a _ { n } ) | < \epsilon , \ : n = 1 , \cdot \cdot \cdot , N ,\tag{1}
$$

where $\tau ( z ) = z - \lfloor z \rfloor .$

Intrinsically, earlier work [13] just engineers Proposition 1 and shows the feasibility of model compression based on dynamic systems. Here, we elaborate on the relationship between θ and ϵ under mild conditions, which is a fundamental theoretical issue regarding using the irrational winding to do model compression.

We use the following standard definitions:

• For $\mathbf { x } = ( x _ { 1 } , \ldots , x _ { d } ) \in \mathbb { R } ^ { d }$ , the sup-norm is

$$
\| \mathbf { x } \| _ { \infty } = \operatorname* { m a x } _ { 1 \leq j \leq d } | x _ { j } | .
$$

• For $z \in \mathbb { R }$ , the torus distance to the nearest integer is

$$
\| z \| _ { \mathbb { T } } = \operatorname* { m i n } _ { m \in \mathbb { Z } } | z - m | .
$$

For vectors $\textbf { z } \in \mathbb { R } ^ { d }$ , we take the coordinate-wise torus distance:

$$
\| \mathbf { z } \| _ { \mathbb { T } ^ { d } } = \operatorname* { m a x } _ { 1 \leq j \leq d } \| z _ { j } \| _ { \mathbb { T } } ,
$$

where $\mathbb { T } ^ { d } = \mathbb { R } ^ { d } / \mathbb { Z } ^ { d }$ , which is a torus meaning that two vectors $\mathbf { x } , \mathbf { y } \ \in \ \mathbb { R } ^ { d }$ are identified if and only if their difference is an integer vector.

Lemma 1 (Density of the ϵ-net). Let $K > 0$ , and define

$$
\Delta ( K ) : = \operatorname* { m i n } _ { 0 < | \mathbf { k } | \leq K } \| \mathbf { k } \cdot \mathbf { a } \| _ { \mathbb { T } } ,\tag{2}
$$

where $\mathbf { a } = [ a _ { 1 } , a _ { 2 } , \cdots , a _ { d } ]$ and $\begin{array} { r } { \left. \mathbf { k } \right. = \sum _ { j = 1 } ^ { d } \left. k _ { j } \right. } \end{array}$ . There exists a constant $C _ { d }$ depending only on the dimension d such that if

$$
M \geq C _ { d } \epsilon ^ { - d } \Delta ( K ) ^ { - 1 } , \qquad K \asymp \epsilon ^ { - 1 } ,\tag{3}
$$

then the finite set $\lbrace m \mathbf { a } : 0 \leq m \leq M \rbrace$ is an ϵ-net in $\mathbb { T } ^ { d } ,$ $i . e .$ every $\beta \in \mathbb { T } ^ { \mathbf { d } }$ is within distance < ϵ of some point in the set.

Proof. We argue by contradiction. Suppose there exists a cube Q of radius ϵ centered at some $\beta \in \bar { \mathbb { T } } ^ { \bar { d } }$ that contains no point ma for $0 \leq m \leq M$ . Choose a non-negative smooth function $\phi :  { \mathbb { T } } ^ { d } \to  { \mathbb { R } } _ { \geq 0 }$ such that

1) $\phi ( \mathbf { x } ) \geq 1$ on the sub-cube $Q _ { \epsilon / 2 }$ (radius $\epsilon / 2 ) ;$

2) $\operatorname { s u p p } ( \phi ) \subset Q$ (so ϕ vanishes outside $Q ) ;$

3) The integral satisfies $\begin{array} { r } { \widehat { \phi } ( 0 ) : = \int _ { \mathbb { T } ^ { d } } \phi ( \mathbf { x } ) d \mathbf { x } \asymp \epsilon ^ { d } . } \end{array}$

Such a function exists (e.g., a standard smooth bump function supported in $Q )$

Since the orbit points $\{ m \mathbf { a } \} _ { m = 0 } ^ { M }$ avoid $Q ,$ and $\phi$ vanishes outside $Q ,$ we have the exact identity:

$$
\sum _ { m = 0 } ^ { M } \phi ( m \mathbf { a } ) = 0 .\tag{4}
$$

Using a Fejer kernel truncation [48], for any arbitrarily small´ $\delta > 0$ , there exists a trigonometric polynomial

$$
P ( \mathbf { x } ) = \sum _ { | \mathbf { k } | \leq K } \widehat { P } ( \mathbf { k } ) e ^ { 2 \pi i \mathbf { k } \cdot \mathbf { x } }\tag{5}
$$

with frequencies bounded by K (where $K \asymp \epsilon ^ { - 1 } )$ such that

$$
\underset { \mathbf { x } \in \mathbb { T } ^ { d } } { \operatorname* { s u p } } \left| P ( \mathbf { x } ) - \phi ( \mathbf { x } ) \right| \leq \delta .\tag{6}
$$

Moreover, we can ensure the following standard properties:

$$
\widehat { P } ( 0 ) \asymp \epsilon ^ { d } , \qquad | \widehat { P } ( k ) | \leq \widehat { P } ( 0 ) \quad \mathrm { f o r ~ a l l ~ } k .\tag{7}
$$

Combining Eqs. (4) and (6), we slightly abuse the notation by writing

$$
\sum _ { m = 0 } ^ { M } P ( m \mathbf { a } ) = 0 .\tag{8}
$$

On the other hand, expand the Fourier series:

$$
\sum _ { m = 0 } ^ { M } P ( m \mathbf { a } ) = \sum _ { \mathbf { k } \in \mathbb { Z } ^ { d } } \widehat { P } ( \mathbf { k } ) \sum _ { m = 0 } ^ { M } e ^ { 2 \pi i m ( \mathbf { k } \cdot \mathbf { a } ) } .\tag{9}
$$

Separating this term from the rest gives

$$
\sum _ { m = 0 } ^ { M } P ( m \mathbf { a } ) = \widehat { P } ( 0 ) ( M + 1 ) + \sum _ { 0 < | \mathbf { k } | \leq K } \widehat { P } ( \mathbf { k } ) \sum _ { m = 0 } ^ { M } e ^ { 2 \pi i m ( \mathbf { k } \cdot \mathbf { a } ) } .
$$

Since $\widehat { P } ( 0 ) M \leq \widehat { P } ( 0 ) ( M + 1 )$ , we have

$$
M \widehat { P } ( 0 ) \leq ( M + 1 ) \widehat { P } ( 0 ) = \left| \sum _ { 0 < | \mathbf { k } | \leq K } \widehat { P } ( \mathbf { k } ) \sum _ { m = 0 } ^ { M } e ^ { 2 \pi i m ( \mathbf { k } \cdot \mathbf { a } ) } \right| .
$$

Finally, applying the triangle inequality to the finite sum on the right, we arrive at the key inequality:

$$
M \widehat { P } ( 0 ) \leq \sum _ { 0 < | \mathbf { k } | \leq K } | \widehat { P } ( \mathbf { k } ) | \left| \sum _ { m = 0 } ^ { M } e ^ { 2 \pi i m ( \mathbf { k } \cdot \mathbf { a } ) } \right| .\tag{10}
$$

Eq. (10) is the rigorous starting point for the rest of the proof. The crucial step here is the use of the reverse triangle inequality, which correctly transfers the negligible approximation error δ from the left side to the right side as an additive error term, rather than incorrectly assuming $\textstyle \sum P = 0$

Using the geometric series estimate:

$$
\left| \sum _ { m = 0 } ^ { M } e ^ { 2 \pi i m \theta } \right| \leq \operatorname* { m i n } \left( M + 1 , \ { \frac { 1 } { 2 \left\| \theta \right\| _ { \mathbb { T } } } } \right) \leq { \frac { 1 } { \left\| \theta \right\| _ { \mathbb { T } } } } ,
$$

and $| \widehat { P } ( k ) | \le \widehat { P } ( 0 )$ , we obtain

$$
M { \widehat { P } } ( 0 ) \leq { \widehat { P } } ( 0 ) \sum _ { 0 < | \mathbf { k } | \leq K } { \frac { 1 } { \| \mathbf { k } \cdot \mathbf { a } \| _ { \mathbb { T } } } } .
$$

Cancel $\widehat { P } ( 0 )$ . By the definition of $\Delta ( K )$ , for all $0 < | { \bf k } | \le K$ we have $\| \mathbf { k } \cdot \mathbf { a } \| _ { \mathbb { T } } \geq \Delta ( K )$ . Therefore

$$
M \leq \sum _ { 0 < | \mathbf { k } | \leq K } \frac { 1 } { \| \mathbf { k } \cdot \mathbf { a } \| _ { \mathbb { T } } } \leq \frac { \# \{ k : 0 < | \mathbf { k } | \leq K \} } { \Delta ( K ) } \leq \frac { C _ { d } ^ { \prime } K ^ { d } } { \Delta ( K ) } .
$$

Since $K \asymp \epsilon ^ { - 1 }$ , we have $K ^ { d } \le C _ { d } ^ { \prime \prime } \epsilon ^ { - d }$ . Hence,

$$
M \leq C _ { d } ^ { \prime \prime \prime } \epsilon ^ { - d } \Delta ( K ) ^ { - 1 } .
$$

This contradicts the lemma’s assumption $\begin{array} { l l } { M } & { } \end{array} \geq$ $C _ { d } \epsilon ^ { - d } \Delta ( K ) ^ { - 1 }$ (choosing $C _ { d } > C _ { d } ^ { \prime \prime \prime } )$ . Thus the contradiction is established, and the lemma follows. □

Theorem 1. Let $\mathbf { a } \in \mathbb { R } ^ { d }$ satisfy the linear form Diophantine condition: there exist constants $c > 0 , \nu > 0$ such that for all nonzero integer vectors k $\cdot \in \mathbb { Z } ^ { d } \setminus \{ 0 \}$

$$
\| \mathbf { k } \cdot \mathbf { a } \| _ { \mathbb { T } } \geq c | \mathbf { k } | ^ { - \nu } .
$$

Then there exist constants $C > 0$ and $\epsilon _ { 0 } > 0$ such that for every $0 < \epsilon < \epsilon _ { 0 }$ and every $\beta \in  { \mathbb { T } } ^ { d }$ , there exists an integer m satisfying

$$
0 \leq m \leq C \epsilon ^ { - ( d + \nu ) } , \qquad \| m \mathbf { a } - \beta \| _ { \mathbb { T } ^ { d } } < \epsilon .
$$

Proof. By the Diophantine condition, taking $K \asymp \epsilon ^ { - 1 }$ , we have

$$
\Delta ( K ) = \operatorname* { m i n } _ { 0 < | \mathbf { k } | \leq K } \| \mathbf { k } \cdot \mathbf { a } \| _ { \mathbb { T } } \geq \operatorname* { m i n } _ { 0 < | \mathbf { k } | \leq K } c | \mathbf { k } | ^ { - \nu } \geq c K ^ { - \nu } .
$$

Since $K \asymp \epsilon ^ { - 1 }$ , there exists a constant $c ^ { \prime } > 0$ such that

$$
\Delta ( K ) \geq c ^ { \prime } \epsilon ^ { \nu } .
$$

Substituting this lower bound into Lemma 1, there exists a constant $C > 0$ such that whenever

$$
M \geq C \epsilon ^ { - d } \cdot ( \epsilon ^ { \nu } ) ^ { - 1 } = C \epsilon ^ { - ( d + \nu ) } ,
$$

the point set $\left\{ m \mathbf { a } : 0 \leq m \leq M \right\}$ forms an ϵ-net.

Hence, for any $\beta \in  { \mathbb { T } } ^ { d }$ , there must exist $0 \leq m \leq M$ satisfying:

$$
\| m \mathbf { a } - \beta \| _ { \mathbb { T } ^ { d } } < \epsilon .
$$

Choosing the constant C according to Lemma 1 concludes the proof.

□

Remark 1. We argue that Theorem 1 holds in a general sense. The linear form Diophantine condition is by no means an exotic or artificially restrictive assumption. Common irrational numbers can satisfy this condition. For instance, the classical two-dimensional vector $\mathbf { a } = ( { \sqrt { 2 } } , { \sqrt { 3 } } )$ satisfies the condition with the exponent $\nu = 2$ , as a direct consequence of the Schmidt Theorem [49]. Thus, rather than being a pathological rarity, the linear form Diophantine condition serves as a sharp but broadly applicable threshold.

## IV. A UNIFIED FRAMEWORK

## A. Dynamic System Families

Using a dynamic system converts a high-dimensional vector into low-dimensional trajectory parameters. Early work [13] has shown that the irrational winding can indeed compress large models well. In this work, we examine four common families of dynamic systems: space-filling/grid traversal mappings, chaotic trajectory-based mappings, congruential and pseudo-random generator mappings, and quasi-Monte Carlo sequence mappings. In doing so, we can enrich the armory of model compression via dynamic systems. Because there is no definitive theory on the superiority of dynamic systems, involving more tools can increase the likelihood of achieving better compression performance.

Space-filling curves (SFC) define deterministic mappings from a one-dimensional index to a D-dimensional lattice point:

$$
\Phi _ { \mathtt { S F C } } : \{ 0 , \dotsc , S - 1 \}  \{ 0 , \dotsc , m - 1 \} ^ { D } , \qquad S = m ^ { D } ,\tag{11}
$$

where m denotes the per-axis resolution, and SFC is the instantiated specific space-filling method.

Representative space-filling curves are Hilbert, Peano, Morton/Z-order, and Snake traversals, as shown in Figure 1(a)-(d). Each dynamic system Φ can generate a trajectory

$$
\Phi _ { \mathtt { S F C } } ( \xi ) , \quad \xi \in [ 0 , S - 1 ] .\tag{12}
$$

For Hilbert, Morton/Z-order, and Snake, we use $m \ = \ 2 ^ { k }$ , whereas Peano uses $m \ = \ 3 ^ { k }$ . Each mapping provides a deterministic bijection between the scalar indices and the corresponding lattice points. Detailed constructions of the four mappings are provided in the Supplementary Material.

Chaotic systems can generate deterministic trajectories with complex aperiodic behavior under chaotic parameter settings. As a representative case, we instantiate this family with the Lorenz system:

$$
\dot { x } = \sigma ( y - x ) , \quad \dot { y } = x ( \rho - z ) - y , \quad \dot { z } = x y - \gamma z .\tag{13}
$$

where $\sigma , \rho ,$ and $\gamma$ are system parameters. With system parameters, an initial condition $( x _ { 0 } , y _ { 0 } , z _ { 0 } )$ , and an integration step size $\Delta t ,$ we numerically integrate Eq. (13). After discarding the first $T _ { \mathrm { t r a n s } }$ transient states, we retain S consecutive posttransient states:

$$
\mathbf { h } _ { \xi } = \left( x _ { \xi } , y _ { \xi } , z _ { \xi } \right) \in \mathbb { R } ^ { 3 } , \qquad \xi = 0 , \dots , S - 1 .\tag{14}
$$

Here, $\xi$ denotes the retained trajectory index.

For the two-dimensional compression setting, the posttransient Lorenz trajectory is projected to a two-dimensional space by

$$
\Phi _ { \mathrm { L o r e n z } } ( \xi ) = P _ { \mathrm { 2 D } } \left( \mathbf { h } _ { \xi } \right) \in \mathbb { R } ^ { 2 } , \qquad \xi = 0 , \dots , S - 1 ,\tag{15}
$$

where $ { P _ { \mathrm { 2 D } } }$ denotes the projection operator from the threedimensional Lorenz states to the two-dimensional template space. In our implementation, $ { P _ { \mathrm { 2 D } } }$ is obtained by PCA on the post-transient Lorenz states.

As shown in Figure 1(e), unlike grid-based traversals, Lorenz-based systems do not lie on a regular lattice. Unlike QMC and congruential generators, it is not designed to uniformly cover the canonical domain. Although the states are uniform in integration time, their spatial distribution follows the geometry and visiting density of the Lorenz attractor. Under chaotic parameter settings, the Lorenz trajectory is confined to its attractor after the transient phase and repeatedly visits different regions of the attractor. Thus, the retained posttransient samples can provide a dense empirical coverage of the attractor support as the trajectory length increases.

Congruential and pseudo-random generators can be viewed as deterministic recursive systems that map a scalar index to a multi-dimensional system state. As shown in Figure $1 ( \mathrm { f } ) \substack { \mathrm { - } } ( \mathrm { g } )$ , unlike grid-based traversals, systems do not explicitly enumerate a dense grid. Instead, a D-dimensional system point is obtained by unfolding a one-dimensional recurrence for D consecutive steps.

For LCG, the internal state follows the recurrence

$$
z _ { t + 1 } = ( a z _ { t } + b ) \bmod M ,\tag{16}
$$

where $z _ { t } ~ \in ~ \{ 0 , \dots , M - 1 \}$ denotes the integer state at recursion step $t ,$ and $a , \ b ,$ and M denote the multiplier, increment, and modulus. Given the size S for the candidate

set, we generate $S + D - 1$ scalar states and form the ξ-th D-dimensional template by a sliding window:

$$
\Phi _ { \mathrm { L C G } } ( \xi ) = \left( z _ { \xi } , \dots , z _ { \xi + D - 1 } \right) \in \{ 0 , \dots , M - 1 \} ^ { D } ,\tag{17}
$$

where $\xi \in \{ 0 , 1 , \ldots , S - 1 \}$ denotes the index.

For LCG, we set $M = 2 ^ { k }$ and use full-period parameters satisfying the Hull–Dobell conditions [50]. This guarantees a full-period state transition for each modulus M.

PCG uses the same congruential state transition as Eq. (16), but does not directly induce the raw state $z _ { t }$ . Instead, it applies a deterministic output permutation:

$$
y _ { t } = \mathcal { P } ( z _ { t } ) ,\tag{18}
$$

where $y _ { t } \in \{ 0 , \ldots , M - 1 \}$ , and $\mathcal { P } ( \cdot )$ denotes a deterministic output transformation composed of bit mixing and statedependent rotation operations. In our implementation, $\mathcal { P } ( \cdot )$ is a PCG-style output permutation as

$$
\mathcal { P } ( z ) = \mathrm { x o r s h i f t } _ { 3 } ^ { R } \left( \mathrm { r o t r } _ { k } \left( \mu \cdot \mathrm { x o r s h i f t } _ { 5 } ^ { R } ( z ) , \rho \right) \right) ,\tag{19}
$$

where

$$
\mathrm { x o r s h i f t } _ { q } ^ { R } ( x ) = x \oplus ( x \gg q ) .\tag{20}
$$

Here, ⊕ denotes bitwise $\mathbf { X O R } , \mathbf { \Phi } \gg$ denotes right shift, ro $\operatorname { r } _ { k } ( \cdot , \rho )$ denotes circular right rotation over a k-bit word, $\mu$ is an odd multiplication constant, and $\rho$ is a fixed rotation offset. All operations are performed within the k-bit integer space. Since xorshift, circular rotation, and multiplication by an odd integer are invertible over k-bit words, $\mathcal { P }$ defines a deterministic bijection on $\{ 0 , \ldots , M - 1 \}$ , where $M = 2 ^ { k }$

Using the emitted sequence $\{ y _ { t } \} _ { t \ge 0 }$ , PCG adopts the same sliding-window template construction as LCG. The coordinate functions of the PCG system are then defined as

$$
\Phi _ { \mathrm { P C G } } ( \xi ) = \left( y _ { \xi } , \dots , y _ { \xi + D - 1 } \right) \in \{ 0 , \dots , M - 1 \} ^ { D } ,\tag{21}
$$

where $\xi \in \{ 0 , 1 , \ldots , S - 1 \}$ denotes the template index, and $S$ is the number of states.

Therefore, LCG and PCG share the same congruential state transition and sliding-window construction. The difference lies in the sequence: LCG directly uses the raw state $z _ { t } ,$ , whereas PCG uses the permuted output $y _ { t } = \mathcal { P } ( z _ { t } )$ . This output permutation changes the structure of the state set while preserving deterministic generation and efficient reconstruction.

Quasi-Monte Carlo (QMC) sequences are deterministic low-discrepancy point sets that provide structured coverage of the local representation domain, as shown in Figure 1(h). Compared with pseudo-random generators, QMC sequences are designed to reduce discrepancy, meaning that their empirical distribution better matches the target domain volume over axisaligned regions. This makes them suitable for constructing relatively uniform states under a finite sampling budget.

We instantiate QMC mappings with the D-dimensional Halton sequence. Let $j = 0 , \ldots , S - 1$ denote the candidate index and set $\xi = j + 1$ as the Halton sequence index. Let $\left( b _ { 1 } , \ldots , b _ { D } \right)$ be the coordinate bases, which are chosen as the first D prime numbers in our implementation. The Halton state is defined as

$$
\Phi _ { \mathrm { H a l t o n } } ( \xi ) = \big ( \varphi _ { b _ { 1 } } ( \xi ) , \varphi _ { b _ { 2 } } ( \xi ) , \dots , \varphi _ { b _ { D } } ( \xi ) \big ) \in [ 0 , 1 ) ^ { D } ,\tag{22}
$$

![](images/00a5dec557209631f8f9d30f08a517b9d021551eaf97bce69bff279ed3651af3.jpg)  
Fig. 2. An overview of the proposed DS-based compression framework. The parameter tensor is flattened and partitioned into grouped weight vectors, which are mapped into a tensor-wise local domain. A dynamic system generates a finite trajectory of states, and each grouped vector is assigned to its nearest state and the associated index. The model is compressed by storing the resulting index instead of the original floating-point vectors, while the states are recovered from the index for decompression.

where $\varphi _ { b } : \mathbb { N }  [ 0 , 1 )$ is the radical-inverse function in base b. Specifically, if the base-b expansion of $\xi$ is

$$
\xi = \sum _ { r = 0 } ^ { m } a _ { r } ^ { ( b ) } ( \xi ) b ^ { r } , \qquad a _ { r } ^ { ( b ) } ( \xi ) \in \{ 0 , \ldots , b - 1 \} ,\tag{23}
$$

then

$$
\varphi _ { b } ( \xi ) = \sum _ { r = 0 } ^ { \infty } \frac { a _ { r } ^ { ( b ) } ( \xi ) } { b ^ { r + 1 } } .\tag{24}
$$

Here, $a _ { r } ^ { ( b ) } ( \xi )$ denotes the r-th digit of ξ in its base-b expansion. For example, when $D = 2$ used in our implementation, the bases are $( b _ { 1 } , b _ { 2 } ) = ( 2 , 3 )$ , giving

$$
\begin{array} { r } { \Phi _ { \mathrm { H a l t o n } } ( \xi ) = ( \varphi _ { 2 } ( \xi ) , \varphi _ { 3 } ( \xi ) ) . } \end{array}\tag{25}
$$

B. The Framework of Model Compression via Dynamic $S y s \mathrm { - }$ tems

As Figure 2 shows, this section describes the unified model compression framework based on dynamic systems. In this compression framework, each dynamic system serves as a plug-and-play component.

Flatten: Let $\mathbf { \Theta } \Theta = \{ \mathbf { W } ^ { ( \ell ) } \} _ { \ell = 1 } ^ { L }$ denote the set of parameter tensors across L layers in a pretrained neural network. For a tensor $\mathbf { W } ^ { ( \ell ) }$ , let $N _ { \ell } = \mathtt { n u m e l } ( \mathbf { W } ^ { ( \ell ) } )$ denote the total number of weights. We flatten each tensor into a vector:

$$
\tilde { \mathbf { w } } ^ { ( \ell ) } = \mathtt { F l a t t e n } \left( \mathbf { W } ^ { ( \ell ) } \right) \in \mathbb { R } ^ { N _ { \ell } } .\tag{26}
$$

Partition: Since $N _ { l }$ is huge, it is impossible for the algorithm to deal with $\tilde { \mathbf { w } } ^ { ( l ) }$ directly. Therefore, we use the divide-and-conquer strategy by partitioning $\tilde { \mathbf { w } } ^ { ( l ) }$ into small Ddimensional vectors. $\tilde { \mathbf { w } } ^ { ( l ) }$ is padded if necessary, so that the number of scalar entries is divisible by D. The padded entries are discarded after decompression. Mathematically, suppose we have $\begin{array} { r } { M _ { \ell } = \left\lceil \frac { N _ { \ell } } { D } \right\rceil } \end{array}$ groups,

$$
\left\{ \bar { \mathbf { w } } _ { i } ^ { ( \ell ) } \right\} _ { i = 1 } ^ { M _ { \ell } } = \mathsf { P a r t i t i o n } ( \tilde { \mathbf { w } } ^ { ( l ) } ) , \qquad \bar { \mathbf { w } } _ { i } ^ { ( \ell ) } \in \mathbb { R } ^ { D } .\tag{27}
$$

Each group contains D consecutive scalar parameters from the flattened tensor. Padded entries are recorded in the metadata and discarded after decompression.

Shift and Scale: Since different weights may have different magnitudes, we shift and rescale the vector w¯ into a hypercube, followed by a unified operation. For each weight vector $\bar { \mathbf { w } } ^ { ( \ell ) }$ , we apply a shift and scale transformer ShiftScale : $\begin{array} { r } { \mathbb { R } ^ { D } \to [ c ^ { ( \ell ) } - \frac { B _ { \ell } } { 2 } , c ^ { ( \ell ) } + \frac { B _ { \ell } } { 2 } ] ^ { D } } \end{array}$

$$
\mathbf { w } _ { i } ^ { ( \ell ) } = \mathtt { S h i f t S c a l e } \left( \bar { \mathbf { w } } _ { i } ^ { ( \ell ) } \right) ,\tag{28}
$$

where $c ^ { ( l ) }$ is the geometric center $\mathbf { c } ^ { ( \ell ) }$ of all vectors $\{ \bar { \mathbf { w } } _ { i } ^ { ( \ell ) } \} _ { i = 1 } ^ { M _ { \ell } }$ and $B _ { l }$ is the size of the hypercube. Both $c ^ { ( l ) }$ and $B _ { l }$ can be set to different values with respect to different layers. Hereafter, we stop using the superscript or subscript l for convenience.

The trajectory $\Phi ( \xi )$ generated by a dynamic system is in $[ c - \textstyle { \frac { B } { 2 } } , c + \textstyle { \frac { B } { 2 } } ] ^ { \bar { D } }$ . Mathematically, we have for any ${ \bf w } \in [ c -$ $\begin{array} { r } { \mathbf { \bar { \rho } } _ { 2 } ^ { B } , c \mathbf { \bar { + } } \mathbf { \frac { \bar { B } } { 2 } } ] ^ { D } } \end{array}$ , there exists $\xi ^ { * }$ such that

$$
\| \Phi ( \xi ^ { * } ) - \mathbf { w } \| \leq \epsilon .\tag{29}
$$

According to this property, for each dynamic system, the compression is essentially realized by

$$
\operatorname* { m i n } _ { \boldsymbol { \xi } } \| \Phi ( \boldsymbol { \xi } ) - \mathbf { w } \| ,\tag{30}
$$

which turns a vector w into ξ. In real world, ξ itself is discrete like the space-filling curves; otherwise, we sample $\xi$ into a table and pre-store this table to avoid solving Eq. (30) analytically. Without doing so, the entire compression will be time-consuming and unable to scale to large models. We denote the discrete values of ξ as $\xi _ { j } , j = 0 , \dots , S - 1$ where S denotes the number of discrete values. For each w, we search along ξ for the state $\Phi ( \xi )$ that can best represent it. Then, we have

$$
\Phi _ { \mathcal { T } } ( \xi _ { j } ) = \mathbf { w } ^ { \prime } \in [ c - \frac { B } { 2 } , c + \frac { B } { 2 } ] ^ { D } .\tag{31}
$$

Now, we analyze how Eqs. (30) and (31) are implemented in a specific dynamic system:

• Space-filling curves: Including Hilbert, Peano, Morton $/ \operatorname { Z } - \operatorname { O } \mathtt { r } \mathrm { d } \mathsf { e } \mathtt { r } .$ and Snake, SFC first quantizes w onto a regular lattice. We set the per-axis grid resolution m and the grid-cell side length s as

$$
m = \sqrt [ D ] { S } , \qquad s = \frac { B } { m } .\tag{32}
$$

For w, we divide it by s to locate its grid cell. Then w is assigned to the geometrical nearest lattice point as

$$
\mathbf { w } ^ { \prime } = \operatorname* { d i p } \left( \left\lfloor { \frac { \mathbf { w } } { s } } \right\rfloor + \mathbb { I } \left[ \left( \mathbf { w } { \bmod { s } } \right) \geq { \frac { s } { 2 } } \right] , 0 , m - 1 \right) _ { \iota { \ ' } }\tag{33}
$$

where $\mathbb { I } [ \cdot ]$ is applied element-wise and $\mathrm { c l i p } ( \cdot , 0 , m - 1 )$ ensures that each coordinate remains within the valid latticeindex range. Then, we can use the explicit relationship according to Supplementary Eqs. S2, S5, S9, and S12 to obtain ξ from $\mathbf { w } ^ { \prime }$ . The generated mapping

$$
\Phi _ { \mathtt { S F C } } : \{ 0 , \dotsc , m ^ { D } - 1 \}  \{ 0 , \dotsc , m - 1 \} ^ { D }\tag{34}
$$

is a bijection over the finite lattice. Therefore, the corresponding ξ is computed by the inverse traversal rule:

$$
\begin{array} { r } { \xi = \Phi _ { \mathtt { S F C } } ^ { - 1 } ( \mathbf { w } ^ { \prime } ) , \mathtt { S F C } \in \{ \mathrm { H i l b e r t , P e a n o , M o r t o n , S n a k e } \} . } \end{array}\tag{35}
$$

For Morton/Z-order and Snake traversal, ξ can be computed explicitly from the coordinate rules, while for Hilbert and Peano traversals, it is obtained by reversing the corresponding recursive construction.

• Pseudo-random (PR): The scalar state $\xi$ is not obtained by an explicit inverse traversal rule. Instead, we apply Eqs. (17) and (21) to generate a state set

$$
\begin{array} { r } { \mathcal { C } _ { \mathtt { P R } } = \{ \Phi _ { \mathtt { P R } } ( \xi _ { i } ) \mid i \in ( 0 , \dots , S - 1 ) \} , } \end{array}\tag{36}
$$

where P $\mathtt { R } \in \{ \mathtt { L C G } , \mathtt { P C G } \}$ , and $\xi$ is selected by the nearestneighbor assignment:

$$
\boldsymbol { \xi } ^ { * } = \arg \operatorname* { m i n } _ { \mathcal { C } _ { \mathtt { P R } } } \left\| \Phi _ { \mathtt { P R } } ( \boldsymbol { \xi } ) - \mathbf { w } \right\| .\tag{37}
$$

Unlike SFC, this assignment is not an inverse of a lattice bijection, but a nearest-state search over a finite deterministic candidate set.

• QMC: ξ indexes a deterministic low-discrepancy point. Since Halton does not provide a closed-form inverse for w, we apply Eq. (22) to construct the finite state set:

$$
\mathcal { C } _ { \mathrm { Q M C } } = \left\{ \Phi _ { \mathrm { H a l t o n } } ( \xi _ { i } ) \ | \ i \in ( 0 , \dots , S - 1 ) \right\} ,\tag{38}
$$

and $\xi$ is then computed as

$$
\boldsymbol { \xi } ^ { * } = \arg \operatorname* { m i n } _ { \mathcal { C } _ { \mathrm { q u c } } } \left\| \Phi _ { \mathrm { H a l t o n } } ( \boldsymbol { \xi } ) - \mathbf { w } \right\| .\tag{39}
$$

This assigns w to the nearest index $\xi ^ { * }$

• Chaos: ξ is also sampled from a deterministic trajectory rather than from a regular lattice. For the Lorenz instantiation, since the trajectory winds around the attractor, we align a circular region instead of the default hypercube. We apply Eq. (15) to define the scalar-state set as

$$
\begin{array} { r } { \mathcal { C } _ { \mathtt { C h a o s } } = \left\{ \Phi _ { \mathtt { C h a o s } } ( \xi _ { i } ) \ \Big | \ \xi _ { i } \in \{ T _ { \mathrm { t r } } + n \Delta t \} _ { n = 0 } ^ { S - 1 } \right\} . } \end{array}\tag{40}
$$

Given w, we compute

$$
\xi ^ { * } = \arg \operatorname* { m i n } _ { \mathcal { C } _ { \mathrm { C h a o s } } } \left. \lvert \Phi _ { \mathrm { L o r e n z } } ( \xi ) - \mathbf { w } \right. \rvert .\tag{41}
$$

This assigns w to the nearest $\xi ^ { * }$

Remark 2. We can store a table or an explicit relationship between $j$ and $\xi _ { j }$ such that only j needs to be stored:

$$
j \to \xi _ { j } \to \Phi ( \xi _ { j } ) \to \mathbf { w } ,
$$

thereby w is compressed into the integer index, which is similar to the classical vector quantization. This can be viewed as a combination of dynamic system+quantization. For SFC systems, compression directly combines lattice quantization with a deterministic dynamical mapping: quantization defines a finite regular domain, while the dynamic system provides the ordering and indexing rule. For LCG, PCG, QMC sequences and chaotic systems, they directly produce a finite state set, and w is represented by the state which is nearest to it. Then, the associated index is stored.

Yet, quantization is not an essential step for realizing compression. Without quantization, we can find a point ξ such that $\Phi ( \xi )$ can represent w; therefore, compression can also be realized. In conclusion, the essential compression mechanism is the low-dimensional scalar parameters to high-dimensional states.

## C. Engineering Techniques

Here, we introduce our engineering techniques to make the proposed model compression via dynamic systems adapt well in real-world model compression. We mainly deal with two issues: i) how to compress fast so that our method can handle large models with billions of parameters; ii) how to control the representation error for weights to avoid the performance collapse of the compressed model.

Fast compression: The major computational cost in the proposed framework arises from assigning w to its nearest state Φ(ξ) (Eq. (30)), particularly for a large candidate set $\mathcal { C } .$ To accelerate the computation, we adopt different assignment strategies for different generator families. For the two-dimensional Lorenz system, we construct a KD-tree over all S states in C once and reuse it across layers. We perform nearest-neighbor queries with an average complexity of O(log(S)) per query.

For SFC methods, because their states in C are integers, Eq. (33) directly quantizes w to an integer lattice coordinate. If we still build a KD-tree, this integer structure will be ignored and introduce additional construction cost. Thus, we directly use the state set C as a template, where each state is associated with a scalar index $j .$ The template is generated once under a canonical local coordinate system and reused across different layers by translation and scaling, if necessary, to align with the hypercube $[ c - { \frac { B } { 2 } } , c + { \frac { B } { 2 } } ] ^ { D }$ . This reduces both assignment overhead during compression and lookup overhead during decompression, because the stored index can directly refer to the corresponding state.

For LCG, PCG and QMC, we perform exact Euclidean nearest-state assignment over S candidate states. The canonical candidate set is generated once and reused across layers through translation and scaling. To limit memory usage during nearest-state search, the D-dimensional grouped vectors are processed in batches, while the candidate states are further scanned in memory-bounded blocks when necessary. The minimum distance and corresponding index are updated across all blocks. Thus, the assignment remains exact, while the computational cost remains O(SD) per grouped vector.

Error Control: For decompression, we need to restore the weight vector w˜ from index $j$ to:

$$
j \to \xi _ { j } \to \Phi ( \xi _ { j } ) \to \mathbf { w } \to \bar { \mathbf { w } } \to \tilde { \mathbf { w } } ,
$$

The error mainly occurs at $\Phi ( \xi _ { j } ) \to \textbf { w }$ , and $\mathbf { w } \to \bar { \mathbf { w } }$ may amplify the error. Since increasing S can lower the error in $\Phi ( \xi _ { j } ) \to \mathbf { w }$ , we let S increase quadratically until the error is acceptable.

For $\mathbf { w } \to \bar { \mathbf { w } }$ , we need to scale it into $[ { \bf c } - \frac { B } { 2 } , { \bf c } + \frac { B } { 2 } ] ^ { D }$ Mathematically, let

$$
\rho _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { i } \left\| \bar { \mathbf { w } } _ { i } - \mathbf { c } \right\| _ { \infty } .\tag{42}
$$

Vectors inside $[ \mathbf { c } - \textstyle { \frac { B } { 2 } } , \mathbf { c } + \textstyle { \frac { B } { 2 } } ] ^ { D }$ are assigned to the inner class $k = 0 ,$ . For the outer vectors, the radial interval $\left( B / 2 , \rho _ { \mathrm { m a x } } \right]$ is divided into K different domains, and k is the class number:

$$
\rho _ { k } = \frac { B } { 2 } + k \frac { \rho _ { \operatorname* { m a x } } - \frac { B } { 2 } } { K } , \qquad k = 1 , \dots , K ,\tag{43}
$$

where K is selected dynamically according to an MAE-based loss threshold to control the error. Thus, the closer $\bar { \bf w } _ { i }$ require only mild scaling, while farther $\bar { \bf w } _ { i }$ are scaled more strongly. Given $\bar { \bf w } _ { i } ,$ it is assigned to the smallest class:

$$
\operatorname* { m i n } \left\{ k \in \left\{ 1 , \ldots , K \right\} : \left\| { \bar { \mathbf { w } } } _ { i } - \mathbf { c } \right\| _ { \infty } \leq \rho _ { k } \right\} .\tag{44}
$$

The corresponding scaling factor is

$$
a _ { k } = \frac { B / 2 } { \rho _ { k } } ,\tag{45}
$$

and the scaled vector is

$$
\mathbf { w } _ { i } = \mathbf { c } + a _ { k } \left( \bar { \mathbf { w } } _ { i } - \mathbf { c } \right) .\tag{46}
$$

Furthermore, we can combine the class information with j by updating j with the radial class label k as

$$
j ^ { \prime } = j + k S ,\tag{47}
$$

During decompression, $j ^ { \prime }$ is parsed into k and $j$ by the following:

$$
k = \left\lfloor j ^ { \prime } / S \right\rfloor , \qquad j = j ^ { \prime } \bmod S .\tag{48}
$$

Thus, the required bit width is $\left\lceil \log _ { 2 } \left( ( K + 1 ) S \right) \right\rceil$

Furthermore, we set a mean-MAE threshold η for a tensor to control the error. After encoding w, we decompress it and compute the MAE in place. If the mean MAE exceeds the threshold, the corresponding tensor $\mathbf { w } _ { i }$ is stored directly without compression.

## V. ANALYSIS EXPERIMENTS

Before formally comparing dynamic system-based compression methods with other state-of-the-arts, here we first conduct analytical experiments to investigate how different hyperparameters affect the compression–accuracy trade-off, with an emphasis on the grouping dimension and state scale. Then, we systematically compare the compression performance of different dynamic system-based compression methods using the Pareto curves. The analysis experiments can not only enrich the understanding of each method but also help choose representative configurations for them.

Experimental Setup. We evaluate Hilbert, Morton/Z-order, Peano, Snake, Lorenz, LCG, PCG, and QMC/Halton. The dimension and scale settings are summarized in Table I. Across the eight methods, we evaluate 254 configurations. We use the standard torchvision ResNet-18 pretrained on ImageNet-1K as the backbone.<sup>1</sup> The original checkpoint occupies 44.7 MB and achieves Top-1 accuracy 69.76% and Top-5 accuracy 89.07% on the 50,000-image ILSVRC 2012 validation set [51]. We follow the standard evaluation protocol by resizing each image to 256 pixels, applying a 224 × 224 center crop, and using the normalization recommended here [52]. The reconstructed models are evaluated without any test-time augmentation. All experiments are conducted on servers equipped with NVIDIA L40 and A800 GPUs.

TABLE I  
EVALUATED STATE-GENERATION SETTINGS. FOR GRID-BASED MAPPINGS, $2 ^ { k }$ (OR 3<sup>k</sup> FOR PEANO) DENOTES THE PER-AXIS RESOLUTION, YIELDING $\stackrel { \triangledown } { \boldsymbol { S } } = 2 ^ { k D } \ ( \mathrm { O R \ } S = 3 ^ { k D } )$ SYSTEM STATES. FOR TRAJECTORY- AND SEQUENCE-BASED MAPPINGS, S DENOTES THE SYSTEM STATES NUMBER.  
Method D Scale setting   
Space-filling curves and traversal mappings   
Hilbert 2–3 $S = 2 ^ { k \cdot D } , \ k \in \{ 4 , 5 , 6 , 7 , 8 , 9 , 1 0 \}$   
4–6 $S = 2 ^ { k \cdot D } , \ k \in \{ 3 , 4 , 5 , 6 , 7 , 8 , 9 , 1 0 \}$   
Snake 2–3 $S = 2 ^ { k \cdot D } , \ k \in \{ 4 , 5 , 6 , 7 , 8 , 9 , 1 0 \}$   
4–6 $S = 2 ^ { k \cdot D } , \ k \in \{ 3 , 4 , 5 , 6 , 7 , 8 , 9 , 1 0 \}$   
Z-order 2–3 $S = 2 ^ { k \cdot D } , \ k \in \{ 4 , 5 , 6 , 7 , 8 , 9 , 1 0 \}$   
4–6 $S = 2 ^ { k \cdot D } , \ k \in \{ 3 , 4 , 5 , 6 , 7 , 8 , 9 , 1 0 \}$   
Peano 2–6 $S = 3 ^ { k \cdot D } , k \in \{ 3 , 4 , 5 , 6 \}$   
Chaotic trajectory-based mapping   
Lorenz 2 $S = \bar { 2 } ^ { 1 8 } , \bar { 2 } ^ { 2 0 } , 2 ^ { 2 2 } , 2 ^ { 2 4 }$   
Pseudo-random sequence mappings   
LCG 2 $S = 2 ^ { k } , \ k \in \{ 6 , 8 , 1 0 , 1 2 , 1 4 , 1 6 , 1 8 \}$   
3 $S = 2 ^ { k } , \ k \in \{ 8 , 1 0 , 1 2 , 1 4 , 1 6 , 1 8 , 2 0 , 2 2 \}$   
4 S = 2<sup>k</sup>, k ∈ {12, 14, 16, 18, 20, 22, 24, 26}   
5, 6 $S = 2 ^ { k } , \ k \in \{ 1 2 , 1 4 , 1 6 , 1 8 , 2 0 , 2 2 , 2 4 , 2 6 , 2 8 \}$   
PCG 2 $S = 2 ^ { k } , \ k \in \{ 6 , 8 , 1 0 , 1 2 , 1 4 , 1 6 , 1 8 \}$   
3 $S = 2 ^ { k } , \ k \in \{ 8 , 1 0 , 1 2 , 1 4 , 1 6 , 1 8 , 2 0 , 2 2 \}$   
4 S = 2<sup>k</sup>, k ∈ {12, 14, 16, 18, 20, 22, 24, 26}   
5, 6 $S = 2 ^ { k } , \ k \in \{ 1 2 , 1 4 , 1 6 , 1 8 , 2 0 , 2 2 , 2 4 , 2 6 , 2 8 \}$   
Low-discrepancy sequence mapping   
QMC 2 $S = 2 ^ { k } , \ k \in \{ 6 , 8 , 1 0 , 1 2 , 1 4 , 1 6 , 1 8 , 2 0 \}$   
3 $S = 2 ^ { k } , \ k \in \{ 8 , 1 0 , 1 2 , 1 4 , 1 6 , 1 8 , 2 0 , 2 2 \}$   
4 $S = 2 ^ { k } , \ k \in \{ 1 2 , 1 4 , 1 6 , 1 8 , 2 0 , 2 2 , 2 4 , 2 6 \}$   
$5 , 6$ $S = 2 ^ { k } , k \in \{ 2 0 , 2 2 , 2 4 , 2 6 , 2 8 \}$

All methods are evaluated under the same tensor-level compression protocol. Each tensor is either stored directly or compressed by the corresponding dynamic system. As common practice [30], the buffers, biases, very small tensors, and tensors whose reconstruction error exceeds the error threshold will be considered as outliers and therefore stored directly. For tensors compressed, we save their state indices and the associated metadata for decompression.

Overall Compression–Accuracy Behavior. The measured compression ratio is governed by two coupled factors: the dimensionality (D) which is the number of weights to compress at once, and the scale (S). We visualize the joint effects of D and S on the measured compression ratio and Top-1 accuracy drop in Figures 3 and 4.

SFC. A notable observation from Figures 3 and 4 is that Hilbert, Morton/Z-order and Snake yield identical compression ratios and accuracies under all matched configurations. For a grouping dimension $D$ and a per-axis resolution $m = 2 ^ { k }$ , the three methods traverse the same state grid:

$$
\begin{array} { r } { \mathcal { C } _ { m , D } = \{ \Phi _ { T } ( \xi _ { i } ) \} _ { i = 0 } ^ { m ^ { D } - 1 } \quad T \in \{ \mathrm { H i l b e r t } , \mathrm { M o r t o n } , \mathrm { S n a k e } \} . } \end{array}\tag{49}
$$

They differ only in the permutation used to assign scalar indices to the grid points. Because exact assignment selects the same reconstruction point from $\mathcal { C } _ { m , D }$ , the reconstructed tensors and threshold-based fallback decisions remain unchanged. Although the stored index values differ across traversal orders, they are stored in the same bit width. The resulting compression–accuracy behavior is thus determined by the grid geometry and resolution rather than by traversal order.

At $\begin{array} { r l r } { D } & { { } = } & { 2 , 3 . } \end{array}$ , the compression performance shows a consistent and stable trend. As S increases, both the loss of Top-1 accuracy and the compression ratio decrease, the latter decrease resulting from the enlarged index width. At $D = 4 , 5 , 6$ , their compression ratios initially increase with per-axis resolution from $2 ^ { 3 } \ \mathrm { t o } \ 2 ^ { 4 }$ , while the loss of accuracy decreases. From the per-axis resolution of $2 ^ { 4 }$ onward, the compression performance becomes consistent and stable. The compression ratio and accuracy loss decrease with increasing S. $D = 4$ achieves the lowest Top-1 accuracy drop of 0.25% while keeping the compression ratio above 5×. $\textit { D } = \ 5$ achieves the highest compression ratio of 5.88× constrained by the Top-1 accuracy drop below 1%.

For a fixed $D ,$ increasing S may expand the compressed tensor set, making previously uncompressed tensors eligible for compression. This reduces fallback storage and thus can improve the compression ratio. Therefore, the compression ratio increases when the reduction in fallback storage outweighs the larger index cost; otherwise, the compression ratio decreases. Newly-compressed tensors may introduce additional error and thus can decrease the accuracy. Therefore, the accuracy decreases when the newly introduced error outweighs the overall error improvement gained from larger S; otherwise, the accuracy increases.

Furthermore, the Peano curve follows the same permutation invariance on an identical grid. In our experiments, however, the Peano curve uses ternary resolutions $m = 3 ^ { k }$ whereas the Hilbert, Morton/Z-order, and Snake curves use binary resolutions $m = 2 ^ { k }$ . Both their evaluated grid points and scale do not match exactly. The observed differences therefore arise from their different resolution schedules rather than the traversal itself.

When the per-axis resolution is $^ { 2 ^ { 6 } , }$ for the Hilbert, snake and Zorder curves, all dimensions $D = 2 \mathrm { - } 6$ achieve compression ratios between 4.84–5.13× with a Top-1 drop at most 0.40%. This range provides a favorable balance between compression efficiency and accuracy preservation. For Peano, per-axis resolution of $3 ^ { 4 }$ achieves compression between 4.52– 4.80× with Top-1 accuracy drops between 0.14–0.36%.

Quasi-Monte Carlo. This method exhibits more irregular and less monotonic compression–accuracy surfaces than the SFC-based methods.

At D = 2, 3, increasing S generally prevents the Top-1 accuracy from dropping too much and reduces the compression ratio, resulting in a stable accuracy–compression tradeoff. For example, at $D = 3 .$ , increasing S from $2 ^ { 8 }$ to 2<sup>22</sup> reduces the loss of Top-1 accuracy from 5.22% to 0.16%, with the compression ratio decreasing from 5.65× to 4.15×. $D = 3$ achieves the highest compression ratio of 5.50× at Top-1 accuracy drop below 1%.

At $D = 4 , 6 ,$ , the accuracy degradation exhibits larger fluctuations as S increases, indicating that performance depends not only on the state count, but also on the effective coverage of states and the tensors satisfying the compression error threshold. $D = 4$ achieves the lowest Top-1 accuracy drop of 0.28% with the compression ratio above 5×.

At D = 5, increasing S from $2 ^ { 2 0 } ~ \mathrm { t o } ~ 2 ^ { 2 8 }$ reduces the Top-1 accuracy drop from 8.26% to 0.38%, while the compression ratio decreases from 6.62× to 5.43× because of the larger bit width for indices.

![](images/dc5bae56104cbb3625519f0534a53850c740506f49200021ca1087d265974456.jpg)  
Fig. 3. Compression-ratio landscapes of the eight evaluated methods on ResNet-18. Each panel corresponds to one method, where the horizontal axis denotes the system scale, the depth axis denotes the grouping dimension D, and the vertical axis reports the measured compression ratio. For each fixed dimension, the curve and translucent curtain connect the evaluated scale configurations, while the numerical labels indicate the corresponding compression ratios. Higher values represent stronger compression. Lorenz is evaluated only in the two-dimensional setting because its trajectory states are projected onto a two-dimensional representation.

Congruential and pseudo-random generators. LCG exhibits a more irregular response than QMC. $\mathbf { A } \mathbf { t } \ { \boldsymbol { D } } = 2$ , increasing S produces a stable compression performance. At $S = 2 ^ { 6 } – \bar { 2 } ^ { 1 8 }$ the loss of Top-1 accuracy decreases further from 6.80% to 0.01%, while the compression ratio also gradually decreases from 6.33× to 3.36×.

At higher dimensions, compression performance fluctuations appear. Taking $D = 4$ as an example, as S increases from $2 ^ { \hat { 1 } \hat { 2 } }$ to $2 ^ { 1 8 }$ , the compression ratio changes from 5.83× to 7.02×, 6.36× and 5.78×, while the corresponding accuracy drops are 15.16%, 4.19%, 8.10% and 1.70%. From $2 ^ { 1 2 }$ to $2 ^ { 1 4 }$ , the reduction in fallback storage outweighs the increase in index cost, resulting in both a higher compression ratio and better accuracy. $\mathrm { { A t } } \ S = 2 ^ { 1 6 }$ , the accuracy drop temporarily increases, suggesting that newly-compressed tensors introduce perturbations. At $S \ = \ 2 ^ { 1 8 }$ , accuracy preservation improves substantially, although the wider indices continue to reduce the compression ratio.

LCG attains its smallest Top-1 drop of 0.41% at $D = 4$ while keeping the compression ratio above 5×. Its highest compression ratio is 5.47× obtained at D = 3 with the Top-1 accuracy drop below 1%.

PCG is more sensitive to the grouping dimension. At $D = 2 .$ increasing S from $2 ^ { 1 0 }$ to $2 ^ { \overset { \vartriangle } { 1 \ v { 2 } } }$ reduces the loss of Top-1 accuracy from 1.21% to 0.27%, while the compression ratio decreases slightly from 4.90× to 4.81×. For $\bar { S } = 2 ^ { 1 2 } – 2 ^ { 1 8 }$ the Top-1 drop remains within 0.22–0.32%. The compression ratio decreases from 4.81× to 3.34×. However, this favorable low-dimensional behavior does not extend to higher dimensions, where the PCG exhibits substantially larger accuracy degradation. PCG achieves the Top-1 accuracy drop under 1%

only at $D = 2 .$

Lorenz is evaluated only at $D = 2 .$ , because the trajectory states are projected onto a two-dimensional representation. Increasing S from $2 ^ { 1 8 }$ to $2 ^ { 2 4 }$ reduces the compression ratio from 3.36× to 2.56×, while the Top-1 drop remains within 0.03– 0.12%. Accuracy preservation is already close to saturation at $S \ = \ 2 ^ { 1 8 }$ , and further extending the trajectory provides little additional accuracy benefit while constantly increasing the index cost.

These results suggest that DS-based compression is fundamentally a state-space approximation problem: a dynamic system defines the geometry of a finite representation space, while the compression quality depends on how effectively this state space covers the structure of model weights. This explains why different systems with the same number of states can exhibit substantially different compression behaviors, while systems with different indexing rules may produce identical results, such as the Hilbert, Morton/Z-order, and Snake compression performance. Therefore, effective dynamical representations should prioritize coverage of groupedweight distributions under a constrained storage budget, rather than simply increasing trajectory complexity or optimizing index traversal.

We jointly analyze the performance of the compression ratio and the accuracy drop using global and method-wise Pareto frontiers, which are provided in Supplementary Figs. S1 and S2. Based on the method-wise frontiers, we select one representative configuration for each method and summarize the configurations in Table II. Excluding Lorenz, representative configurations achieve the compression ratio between 4.80– 5.77× with the accuracy drop between 0.14–0.59%. Lorenz

TABLE II  
![](images/04615402f90e8eee3170d5022e26714e24007f74345f8d7f48204f734cfa8020.jpg)  
Fig. 4. Top-1 accuracy-drop landscapes of the eight evaluated methods on ResNet-18. The vertical axis reports the Top-1 accuracy drop relative to the original model in percentage points. Lower values indicate better accuracy preservation.

REPRESENTATIVE CONFIGURATIONS SELECTED FROM THE METHOD-WISE PARETO FRONTIERS ON RESNET-18. CR DENOTES THE MEASURED COMPRESSION RATIO, AND ∆ DENOTES THE ACCURACY CHANGE RELATIVE TO THE ORIGINAL MODEL, WHICH ACHIEVES 69.76% TOP-1 AND 89.07% TOP-5 ACCURACY.
<table><tr><td>Method</td><td>D</td><td>S</td><td>CR</td><td>Top-1 Acc. (∆)%</td><td>Top-5 Acc. (∆)%</td></tr><tr><td>Hilbert</td><td>3</td><td>215</td><td>5.77×</td><td>69.17 (-0.59)</td><td>88.67 (-0.40)</td></tr><tr><td>Peano</td><td>5</td><td>320</td><td>4.80×</td><td>69.62 (-0.14)</td><td>88.91 (-0.16)</td></tr><tr><td>Lorenz</td><td>2</td><td>218</td><td>3.36×</td><td>69.64 (-0.12)</td><td>89.07 (0.00)</td></tr><tr><td>LCG</td><td>3</td><td>218</td><td>4.96×</td><td>69.52 (-0.24)</td><td>88.99 (-0.08)</td></tr><tr><td>PCG</td><td>2</td><td>212</td><td>4.81×</td><td>69.49 (-0.27)</td><td>88.86 (-0.21)</td></tr><tr><td>QMC</td><td>3</td><td>216</td><td>5.50×</td><td>69.18 (-0.58)</td><td>88.72 (-0.35)</td></tr></table>

achieves a 3.36× compression ratio with a 0.12% accuracy drop.

## VI. COMPARATIVE EXPERIMENTS

Through analysis experiments, the characteristics of different dynamic systems and the hyperparameter configurations are probed. For large language models, we compare the proposed dynamic system-based compression with other stateof-the-art compression methods. Experiments show that DS compression methods can deliver competitive performance.

Experimental Setup. All six representative configurations in Table II are cast for six dynamic systems. The benchmark models are the FP16 checkpoints of Qwen2.5-1.5B and Qwen1.5-7B [53], [54], which account for 3.09 and 15.44 GB in memory, respectively. We compress all eligible parameter tensors using the proposed methods, while tensors whose decompression errors exceed the prescribed threshold are directly retained in the original form.

Compression quality is evaluated from two complementary perspectives: language modeling quality and downstream-task performance. We evaluate perplexity (PPL) on all complete non-overlapping 2048-token blocks of the WikiText-2 raw test split [55]. Downstream performance is evaluated on 16 tasks covering knowledge and question answering (SciQ, BoolQ, and MMLU) [56]–[58], commonsense and causal reasoning (WinoGrande, HellaSwag, PIQA, and COPA) [59]–[62], scientific and mathematical reasoning (ARC-Easy, ARC-Challenge, and GSM8K) [63], [64], and natural language inference, reading comprehension, and linguistic analysis (CB, MultiRC, ReCoRD, RTE, WiC, and WSC) from SuperGLUE [65].

![](images/afaf5ca73626218680feb9464499597977a839c1b74d1dc7da07e86f32dad1e2.jpg)  
Fig. 5. Compression–PPL trade-offs on Qwen2.5-1.5B and Qwen1.5-7B. Dashed lines denote the PPL of the corresponding uncompressed models. Higher compression ratios and lower perplexity indicate better trade-offs.

Comparison with Existing Compression Methods. Table III summarizes the performance of six DS-based compression methods on Qwen2.5-1.5B and Qwen1.5-7B. Across both model scales, all configurations exhibit distinct compression– quality operating points while maintaining perplexity and downstream performance close to the corresponding uncompressed models.

Figure 5 further visualizes the compression-PPL trade-offs, showing distinct operating points across the six dynamic system configurations. Lorenz preserves near-baseline perplexity, with only 0.03 and 0.02 PPL increases on Qwen2.5-1.5B and

TABLE III  
COMPRESSION RESULTS ON QWEN2.5-1.5B AND QWEN1.5-7B. CR DENOTES THE COMPRESSION RATIO. PPL IS EVALUATED ON WIKITEXT-2, AND AVG. SCORE IS AVERAGED OVER 16 DOWNSTREAM TASKS. VALUES IN PARENTHESES DENOTE CHANGES RELATIVE TO THE CORRESPONDING UNCOMPRESSED MODEL.
<table><tr><td>Model</td><td>Method</td><td>CR</td><td>PPL(∆)</td><td>Avg. Score (∆)%</td></tr><tr><td rowspan="5">Owe-.5B</td><td>Original</td><td></td><td>9.26</td><td>67.31</td></tr><tr><td>Hilbert</td><td>2.67×</td><td>9.61 (+0.35)</td><td>66.52 (-0.79)</td></tr><tr><td>Peano</td><td>2.37×</td><td>9.55 (+0.29)</td><td>67.70 (+0.39)</td></tr><tr><td>Lorenz</td><td>1.68×</td><td>9.29 (+0.03)</td><td>67.31 (+0.00)</td></tr><tr><td>QMC</td><td>2.56×</td><td>9.75 (+0.49)</td><td>66.44 (-0.87)</td></tr><tr><td rowspan="4"></td><td>LCG</td><td>2.36×</td><td>9.62 (+0.36)</td><td>66.79 (-0.52)</td></tr><tr><td>PCG</td><td>2.33×</td><td>9.66 (+0.40)</td><td>66.24 (-1.07)</td></tr><tr><td>Original</td><td></td><td>11.30</td><td>65.49</td></tr><tr><td>Hilbert</td><td>3.00×</td><td>11.81 (+0.51)</td><td>66.12 (+0.63)</td></tr><tr><td rowspan="5">Owe.-7B</td><td>Peano</td><td>2.42×</td><td>11.30 (+0.00)</td><td>64.99 (-0.50)</td></tr><tr><td>Lorenz</td><td>1.68×</td><td>11.32 (+0.02)</td><td>65.32 (-0.17)</td></tr><tr><td>QMC</td><td>2.82×</td><td>11.96 (+0.66)</td><td>65.26 (-0.23)</td></tr><tr><td>LCG</td><td>2.52×</td><td>11.52 (+0.22)</td><td>65.00 (-0.49)</td></tr><tr><td>PCG</td><td>2.46×</td><td>11.63 (+0.33)</td><td>65.20 (-0.29)</td></tr></table>

Qwen1.5-7B, respectively, while maintaining a conservative compression ratio of 1.68×. In contrast, Hilbert achieves the highest compression ratios among the evaluated configurations, reaching 2.67× and 3.00× on the two models with PPL increases of 0.35 and 0.51. Other configurations occupy intermediate positions, providing different compression-quality operating points. In particular, QMC reaches relatively high compression ratios of 2.56× and 2.82×, but is accompanied by the largest PPL increases at 0.49 and 0.66, respectively. This consistent trend indicates that the proposed DS method scales beyond convolutional networks to billion-parameter language models. Furthermore, the average downstream scores also remain close to those of the corresponding uncompressed models, indicating that the obtained trade-offs are not subjected to systematic degradation in aggregate downstream tasks.

Among six configurations evaluated above, Hilbert achieves the highest compression ratio on two models. We therefore select it for comparison with existing LLM compression methods. We compare against the irrational-winding reference (IR) [13], weight-only configurations of conventional low-bit post-training quantization methods including Zero-Quant [27], SQuant [28], and vanilla PTQ, as well as representative weight-only methods including GPTQ [8], AWQ [30], SpQR [31], and QuIP [32]. All compared configurations compress model weights while keeping activations and KV caches at FP16 precision. For a consistent comparison, all methods are evaluated on the same model checkpoints and the same evaluation protocols. Compression ratios are computed using the same storage accounting rule.

Table IV, Table V and Figure 6 show that Hilbert occupies a competitive operating point in the high-compression region across both models. On Qwen2.5-1.5B, Hilbert achieves a compression ratio of 2.67× with a PPL increase of only 0.35. GPTQ and AWQ reach close matched compression ratios: 2.68× and 2.69×, respectively. However, their PPL increases are 0.98 and 0.72, compared with 0.35 for Hilbert. Likewise, ZeroQuant-INT4, SQuant-INT4, and PTQ-INT4 reach 2.74× compression but increase PPL by 4.24, 4.27, and 5.88, respectively. Compared with IR, Hilbert achieves the nearly identical compression ratio while reducing the PPL increase from 0.58 to 0.35. SpQR provides a lower PPL increase of 0.19 at a lower compression ratio of 2.50×, whereas QuIP reaches a slightly higher compression ratio of 2.75× with a similar PPL increase of 0.37. These results place Hilbert at a competitive operating point in the highcompression region of the weight-only comparison.

![](images/9d5f207ccc4a0420e4f4a6693ac750817292a9d6ee9627c19868f9a7f396a461.jpg)  
Fig. 6. Compression–perplexity trade-offs of weight-only compression methods on Qwen2.5-1.5B and Qwen1.5-7B. Higher CR and lower PPL indicate better trade-offs.

On Qwen1.5-7B, Hilbert reaches 3.00× compression with a PPL increase of 0.51. ZeroQuant-INT4, SQuant-INT4, and PTQ-INT4 achieve somewhat higher compression at 3.21× but incur larger PPL increases of 2.68, 2.67 and 3.27, respectively. Compared with IR, Hilbert increases the compression ratio from 2.91× to 3.00× while reducing the PPL increase from 1.17 to 0.51. Other methods provide different operating points: AWQ achieves a smaller PPL increase of 0.38 at 2.64×, while SpQR retains nearly unchanged perplexity at 2.47× compression. Relative to these lowercompression operating points, Hilbert achieves a higher compression ratio of 3.00× at the cost of a moderate increase in perplexity.

The average downstream results further provide a complementary view of compression quality. Hilbert obtains average scores of 66.52 and 66.12 on Qwen2.5-1.5B and Qwen1.5- 7B, corresponding to changes of −0.79 and +0.63 points, respectively. On Qwen1.5-7B, its average score remains higher than those of the three conventional INT4 PTQ baselines despite their somewhat higher compression ratios. Overall, these results indicate that Hilbert maintains competitive aggregate downstream performance while operating in the high-compression region.

Tables IV and V also provide a more detailed view of compression effects on downstream-task performance. For Hilbert, performance remains relatively stable on several broad-coverage benchmarks, including HellaSwag, BoolQ, MMLU, PIQA, and ReCoRD. On Qwen2.5-1.5B, the score changes on these tasks remain within approximately one percentage point, while on Qwen1.5-7B the same group of tasks exhibits similarly moderate variations.

Meanwhile, the task-wise results reveal substantially larger method-dependent variations on several benchmarks. GSM8K, CB, MultiRC, and WSC show noticeably larger fluctuations across compression approaches and model scales, with both positive and negative changes observed after compression.

TABLE IV  
COMPRESSION AND DOWNSTREAM-TASK COMPARISON ON QWEN2.5-1.5B. CR DENOTES THE COMPRESSION RATIO, PPL IS EVALUATED ON WIKITEXT-2. TASK SCORES ARE REPORTED IN PERCENTAGE (%). THE BEST DOWNSTREAM TASK SCORE IS HIGHLIGHTED IN BOLD. AVG. SCORE IS AVERAGED OVER THE 16 DOWNSTREAM TASKS. ZQ, SQ, AND PTQ DENOTE ZEROQUANT, SQUANT, AND VANILLA PTQ, RESPECTIVELY.
<table><tr><td></td><td>Reference</td><td>Dyn-Sys</td><td>Previous</td><td colspan="6">Conventional PTQ</td><td colspan="4">Advanced Weight-only</td></tr><tr><td>Metric / Task</td><td>Original</td><td>Hilbert</td><td>IR</td><td>ZQ4</td><td>ZQ8</td><td>SQ4</td><td>SQ8</td><td>PTQ4</td><td>PTQ8</td><td>GPTQ</td><td>AWQ</td><td>SpQR</td><td>QuIP</td></tr><tr><td>CR</td><td></td><td>2.67×</td><td>2.67×</td><td>2.74×</td><td>1.73×</td><td>2.74×</td><td>1.73×</td><td>2.74×</td><td>1.73×</td><td>2.68×</td><td>2.69×</td><td>2.50×</td><td>2.75×</td></tr><tr><td>PPL</td><td>9.26</td><td>9.61</td><td>9.84</td><td>13.50</td><td>9.27</td><td>13.53</td><td>9.27</td><td>15.14</td><td>9.28</td><td>10.24</td><td>9.98</td><td>9.45</td><td>9.63</td></tr><tr><td>(∆ PPL)</td><td></td><td>+0.35</td><td>+0.58</td><td>+4.24</td><td>+0.01</td><td>+4.27</td><td>+0.01</td><td>+5.88</td><td>+0.02</td><td>+0.98</td><td>+0.72</td><td>+0.19</td><td>+0.37</td></tr><tr><td>SciQ</td><td>93.30</td><td>94.20</td><td>93.70</td><td>89.60</td><td>93.40</td><td>89.60</td><td>93.40</td><td>86.30</td><td>93.30</td><td>93.70</td><td>93.80</td><td>93.40</td><td>93.40</td></tr><tr><td>WinoGrande</td><td>63.38</td><td>62.19</td><td>64.40</td><td>58.17</td><td>63.93</td><td>59.75</td><td>63.69</td><td>59.27</td><td>63.38</td><td>62.51</td><td>63.85</td><td>63.54</td><td>64.25</td></tr><tr><td>ARC-Easy</td><td>71.51</td><td>72.85</td><td>72.60</td><td>66.20</td><td>71.04</td><td>67.38</td><td>71.34</td><td>58.71</td><td>71.55</td><td>73.36</td><td>72.64</td><td>71.55</td><td>73.19</td></tr><tr><td>ARC-Challenge</td><td>54.78</td><td>52.39</td><td>53.33</td><td>45.14</td><td>54.86</td><td>45.05</td><td>54.61</td><td>43.52</td><td>54.61</td><td>51.96</td><td>51.96</td><td>54.18</td><td>51.88</td></tr><tr><td>HellaSwag</td><td>67.95</td><td>67.40</td><td>66.63</td><td>61.27</td><td>67.92</td><td>61.39</td><td>67.95</td><td>59.85</td><td>67.86</td><td>66.72</td><td>66.29</td><td>67.42</td><td>66.39</td></tr><tr><td>BoolQ</td><td>78.84</td><td>78.75</td><td>77.86</td><td>65.69</td><td>78.81</td><td>64.98</td><td>78.75</td><td>43.09</td><td>78.96</td><td>76.27</td><td>78.10</td><td>77.16</td><td>77.71</td></tr><tr><td>MMLU</td><td>60.95</td><td>60.06</td><td>59.74</td><td>50.81</td><td>61.00</td><td>50.22</td><td>60.85</td><td>49.91</td><td>60.90</td><td>57.39</td><td>58.28</td><td>60.20</td><td>59.07</td></tr><tr><td>GSM8K</td><td>62.62</td><td>59.14</td><td>54.21</td><td>16.38</td><td>62.02</td><td>15.69</td><td>61.56</td><td>12.59</td><td>62.93</td><td>46.55</td><td>53.37</td><td>59.06</td><td>51.71</td></tr><tr><td>PIQA</td><td>75.95</td><td>76.33</td><td>75.79</td><td>74.10</td><td>76.01</td><td>74.10</td><td>75.90</td><td>72.63</td><td>76.12</td><td>75.41</td><td>75.68</td><td>75.46</td><td>74.70</td></tr><tr><td>CB</td><td>71.43</td><td>64.29</td><td>67.86</td><td>50.00</td><td>73.21</td><td>46.43</td><td>73.21</td><td>41.07</td><td>67.86</td><td>64.29</td><td>60.71</td><td>58.93</td><td>46.43</td></tr><tr><td>COPA</td><td>83.00</td><td>85.00</td><td>84.00</td><td>81.00</td><td>83.00</td><td>79.00</td><td>83.00</td><td>77.00</td><td>83.00</td><td>83.00</td><td>83.00</td><td>82.00</td><td>82.00</td></tr><tr><td>MultiRC</td><td>28.55</td><td>30.38</td><td>42.06</td><td>40.18</td><td>28.24</td><td>41.30</td><td>28.28</td><td>44.66</td><td>29.08</td><td>48.41</td><td>40.45</td><td>41.63</td><td>35.25</td></tr><tr><td>ReCoRD</td><td>84.83</td><td>83.90</td><td>84.06</td><td>78.95</td><td>84.75</td><td>78.55</td><td>84.78</td><td>75.39</td><td>84.86</td><td>83.27</td><td>83.47</td><td>84.38</td><td>84.29</td></tr><tr><td>RTE</td><td>70.04</td><td>64.98</td><td>67.51</td><td>54.51</td><td>70.76</td><td>53.79</td><td>69.68</td><td>59.57</td><td>68.23</td><td>71.12</td><td>68.59</td><td>63.90</td><td>64.62</td></tr><tr><td>WiC</td><td>53.13</td><td>53.76</td><td>58.78</td><td>50.78</td><td>53.13</td><td>51.88</td><td>53.13</td><td>51.88</td><td>53.45</td><td>57.37</td><td>51.88</td><td>57.05</td><td>54.86</td></tr><tr><td>WSC</td><td>56.73</td><td>58.65</td><td>53.85</td><td>63.46</td><td>58.65</td><td>63.46</td><td>57.69</td><td>63.46</td><td>57.69</td><td>36.54</td><td>47.12</td><td>56.73</td><td>57.69</td></tr><tr><td>Avg. Score</td><td>67.31</td><td>66.52</td><td>67.27</td><td>59.14</td><td>67.55</td><td>58.91</td><td>67.36</td><td>56.18</td><td>67.11</td><td>65.49</td><td>65.57</td><td>66.66</td><td>64.84</td></tr><tr><td>(∆ Avg)</td><td></td><td>(-0.79)</td><td>(-0.04)</td><td>(-8.17)</td><td>(+0.24)</td><td>(-8.40)</td><td>(+0.05)</td><td>(-11.13)</td><td>(-0.20)</td><td>(-1.82)</td><td>(-1.74)</td><td>(-0.65)</td><td>(-2.47)</td></tr></table>

TABLE V

COMPRESSION AND DOWNSTREAM-TASK COMPARISON ON QWEN1.5-7B. METRICS, NOTATION, AND HIGHLIGHTING FOLLOW TABLE IV.
<table><tr><td></td><td>Reference</td><td>Dyn-Sys</td><td>Previous</td><td colspan="6">Conventional PTQ</td><td colspan="4">Advanced Weight-only</td></tr><tr><td>Metric / Task</td><td>Original</td><td>Hilbert</td><td>IR</td><td>ZQ4</td><td>ZQ8</td><td>SQ4</td><td>SQ8</td><td>PTQ4</td><td>PTQ8</td><td>GPTQ</td><td>AWQ</td><td>SpQR</td><td>QuIP</td></tr><tr><td>CR</td><td></td><td>3.00×</td><td>2.91×</td><td>3.21×</td><td>1.85×</td><td>3.21×</td><td>1.85×</td><td>3.21×</td><td>1.85×</td><td>2.63×</td><td>2.64×</td><td>2.47×</td><td>2.69×</td></tr><tr><td>PPL</td><td>11.30</td><td>11.81</td><td>12.47</td><td>13.98</td><td>11.30</td><td>13.97</td><td>11.30</td><td>14.57</td><td>11.31</td><td>11.92</td><td>11.68</td><td>11.29</td><td>11.99</td></tr><tr><td>(∆ PPL)</td><td></td><td>+0.51</td><td>+1.17</td><td>+2.68</td><td>+0.00</td><td>+2.67</td><td>+0.00</td><td>+3.27</td><td>+0.01</td><td>+0.62</td><td>+0.38</td><td>-0.01</td><td>+0.69</td></tr><tr><td>SciQ</td><td>83.40</td><td>85.30</td><td>82.70</td><td>79.50</td><td>83.80</td><td>80.20</td><td>83.80</td><td>80.80</td><td>83.90</td><td>79.00</td><td>84.80</td><td>83.10</td><td>83.90</td></tr><tr><td>WinoGrande</td><td>65.19</td><td>64.17</td><td>62.75</td><td>61.72</td><td>64.96</td><td>62.12</td><td>65.19</td><td>61.33</td><td>65.27</td><td>65.19</td><td>65.19</td><td>65.75</td><td>66.06</td></tr><tr><td>ARC-Easy</td><td>63.34</td><td>61.62</td><td>63.68</td><td>61.36</td><td>63.30</td><td>61.03</td><td>63.38</td><td>60.52</td><td>63.59</td><td>58.04</td><td>62.37</td><td>62.96</td><td>60.90</td></tr><tr><td>ARC-Challenge</td><td>56.57</td><td>53.50</td><td>55.63</td><td>48.72</td><td>55.80</td><td>48.21</td><td>55.55</td><td>51.88</td><td>56.06</td><td>55.12</td><td>55.46</td><td>55.97</td><td>54.69</td></tr><tr><td>HellaSwag</td><td>78.68</td><td>77.63</td><td>76.62</td><td>75.73</td><td>78.72</td><td>75.77</td><td>78.73</td><td>75.16</td><td>78.64</td><td>77.66</td><td>77.74</td><td>78.36</td><td>77.76</td></tr><tr><td>BoolQ</td><td>85.47</td><td>84.80</td><td>85.08</td><td>83.70</td><td>85.32</td><td>83.88</td><td>85.38</td><td>84.50</td><td>85.35</td><td>84.68</td><td>85.11</td><td>85.11</td><td>85.17</td></tr><tr><td>MMLU</td><td>60.45</td><td>59.69</td><td>58.87</td><td>57.56</td><td>60.62</td><td>57.62</td><td>60.58</td><td>56.24</td><td>60.53</td><td>59.72</td><td>59.97</td><td>60.13</td><td>59.68</td></tr><tr><td>GSM8K</td><td>20.92</td><td>22.14</td><td>19.26</td><td>23.43</td><td>23.35</td><td>25.17</td><td>23.05</td><td>22.59</td><td>19.48</td><td>41.62</td><td>23.43</td><td>18.50</td><td>25.93</td></tr><tr><td>PIQA</td><td>75.90</td><td>75.79</td><td>75.24</td><td>74.16</td><td>75.90</td><td>74.05</td><td>75.90</td><td>73.45</td><td>75.79</td><td>74.59</td><td>75.35</td><td>76.06</td><td>75.46</td></tr><tr><td>CB</td><td>60.71</td><td>73.21</td><td>35.71</td><td>83.93</td><td>60.71</td><td>83.93</td><td>60.71</td><td>50.00</td><td>60.71</td><td>64.29</td><td>57.14</td><td>58.93</td><td>57.14</td></tr><tr><td>COPA</td><td>86.00</td><td>86.00</td><td>86.00</td><td>78.00</td><td>86.00</td><td>81.00</td><td>84.00</td><td>81.00</td><td>87.00</td><td>87.00</td><td>85.00</td><td>85.00</td><td>86.00</td></tr><tr><td>MultiRC</td><td>25.68</td><td>26.88</td><td>23.10</td><td>35.11</td><td>24.96</td><td>34.36</td><td>25.00</td><td>31.83</td><td>25.45</td><td>28.44</td><td>29.19</td><td>25.74</td><td>23.41</td></tr><tr><td>ReCoRD</td><td>81.06</td><td>80.66</td><td>80.44</td><td>79.57</td><td>81.21</td><td>79.90</td><td>81.20</td><td>79.40</td><td>81.11</td><td>80.20</td><td>81.07</td><td>81.31</td><td>80.66</td></tr><tr><td>RTE</td><td>83.03</td><td>82.67</td><td>85.20</td><td>82.67</td><td>83.39</td><td>83.39</td><td>83.39</td><td>80.51</td><td>83.75</td><td>83.75</td><td>83.39</td><td>82.31</td><td>85.56</td></tr><tr><td>WiC</td><td>67.55</td><td>67.08</td><td>67.08</td><td>61.44</td><td>67.08</td><td>63.64</td><td>67.08</td><td>64.11</td><td>67.40</td><td>66.46</td><td>66.61</td><td>67.71</td><td>65.05</td></tr><tr><td>WSC</td><td>53.85</td><td>56.73</td><td>56.73</td><td>48.08</td><td>55.77</td><td>49.04</td><td>55.77</td><td>45.19</td><td>55.77</td><td>56.73</td><td>54.81</td><td>54.81</td><td>53.85</td></tr><tr><td>Avg. Score</td><td>65.49</td><td>66.12</td><td>63.38</td><td>64.67</td><td>65.68</td><td>65.21</td><td>65.54</td><td>62.41</td><td>65.61</td><td>66.41</td><td>65.41</td><td>65.11</td><td>65.08</td></tr><tr><td>(∆ Avg)</td><td></td><td>+0.63</td><td>-2.11</td><td>-0.82</td><td>+0.19</td><td>-0.28</td><td>+0.05</td><td>-3.08</td><td>+0.12</td><td>+0.92</td><td>-0.08</td><td>-0.38</td><td>-0.41</td></tr></table>

For example, GPTQ obtains a substantial improvement on GSM8K for Qwen1.5-7B, whereas its behaviors on several other tasks do not show the same trend. Similarly, CB and MultiRC exhibit large variations across multiple quantization families. These task-dependent fluctuations further show that changes in PPL are not necessarily aligned with changes in individual downstream-task scores. Consequently, isolated score improvements after compression should not be interpreted as a consistent gain in model capability. Overall, Hilbert maintains competitive task-level performance.

## VII. CONCLUSION

This paper has investigated model compression through the lens of dynamic systems. We first established a theoretical foundation for DS-based compression with the irrational winding, thereby transforming DS-based compression from a heuristic construction into a principled tool with a controllable error profile. We have then extended the paradigm from the single irrational-winding instantiation to a unified framework based on DS. To make the framework practical at scale, we have introduced KD-tree and coordinate-template acceleration for fast state assignment, together with outlier identification and threshold-based fallback for error control. Experiments confirm that DS-based compression is competitive. Several open questions remain for future work, including automatic selection of the dynamic system and its configuration and a more complete characterization of computational overhead during both compression and deployment. We hope this work stimulates further research at the intersection of dynamic systems and efficient deep learning.

## REFERENCES

[1] Z. Zhou, X. Ning, K. Hong, T. Fu, J. Xu, S. Li, Y. Lou, L. Wang, Z. Yuan, X. Li, S. Yan, G. Dai, X.-P. Zhang, Y. Dong, and Y. Wang, “A survey on efficient inference for large language models,” 2024. [Online]. Available: https://arxiv.org/abs/2404.14294

[2] NVIDIA, “GeForce RTX 4090 graphics card specifications,” https: //www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/, 2022, accessed: 2026-07-27.

[3] S. Yin, C. Fu, S. Zhao, K. Li, X. Sun, T. Xu, and E. Chen, “A survey on multimodal large language models,” National Science Review, vol. 11, no. 12, p. nwae403, 2024. [Online]. Available: https://pmc.ncbi.nlm.nih.gov/articles/PMC11645129/

[4] J. Zhang, M. Jiang, N. Dai, T. Lu, A. Uzunoglu, S. Zhang, Y. Wei, J. Wang, V. M. Patel, P. P. Liang, D. Khashabi, C. Peng, R. Chellappa, T. Shu, A. Yuille, Y. Du, and J. Chen, “World-in-world: World models in a closed-loop world,” in International Conference on Learning Representations, 2026. [Online]. Available: https://proceedings.iclr.cc/paper files/paper/2026/ hash/5b4263be85820683d78675cc18d2efc7-Abstract-Conference.html

[5] X. Zhu, J. Li, Y. Liu, C. Ma, and W. Wang, “A survey on model compression for large language models,” Transactions of the Association for Computational Linguistics, vol. 12, pp. 1556–1577, 2024. [Online]. Available: https://aclanthology.org/2024.tacl-1.85/

[6] Z. Wan, X. Wang, C. Liu, S. Alam, Y. Zheng, J. Liu, Z. Qu, S. Yan, Y. Zhu, Q. Zhang, M. Chowdhury, and M. Zhang, “Efficient large language models: A survey,” Transactions on Machine Learning Research, 2024. [Online]. Available: https://arxiv.org/abs/2312.03863

[7] E. Frantar and D. Alistarh, “SparseGPT: Massive language models can be accurately pruned in one-shot,” in Proceedings of the 40th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 202. PMLR, 2023, pp. 10 323–10 337. [Online]. Available: https://proceedings.mlr.press/v202/frantar23a.html

[8] E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh, “GPTQ: Accurate post-training quantization for generative pre-trained transformers,” in International Conference on Learning Representations, 2023. [Online]. Available: https://iclr.cc/virtual/2023/poster/10855

[9] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” in NIPS Deep Learning and Representation Learning Workshop, 2015. [Online]. Available: https://research.google/ pubs/distilling-the-knowledge-in-a-neural-network/

[10] Y. Gu, L. Dong, F. Wei, and M. Huang, “Minillm: Knowledge distillation of large language models,” in International Conference on Learning Representations, 2024. [Online]. Available: https://proceedings.iclr.cc/paper files/paper/2024/ hash/8ac015d409635f196f9e3e9dcfb9a94e-Abstract-Conference.html

[11] T. N. Sainath, B. Kingsbury, V. Sindhwani, E. Arisoy, and B. Ramabhadran, “Low-rank matrix factorization for deep neural network training with high-dimensional output targets,” in 2013 IEEE International Conference on Acoustics, Speech and Signal Processing, 2013, pp. 6655–6659. [Online]. Available: https://ieeexplore.ieee.org/ document/6638949

[12] S. Shuvaev, D. Lachi, A. Koulakov, and A. Zador, “Encoding innate ability through a genomic bottleneck,” Proceedings of the National Academy of Sciences, vol. 121, no. 38, p. e2409160121, 2024. [Online]. Available: https://www.pnas.org/doi/10.1073/pnas.2409160121

[13] F.-L. Fan, J. Fan, D. Wang, J. Zhang, Z. Dong, S. Zhang, G. Wang, and T. Zeng, “Hyper-Compression: Model compression via hyperfunction,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 8, pp. 8813–8830, Aug. 2026. [Online]. Available: https://doi.org/10.1109/TPAMI.2026.3673772

[14] D. Hilbert, “Uber die stetige abbildung einer linie auf ein<sup>¨</sup> flachenst ¨ uck,” ¨ Mathematische Annalen, vol. 38, no. 3, pp. 459– 460, 1891. [Online]. Available: https://link.springer.com/chapter/10. 1007/978-3-662-38452-7 1

[15] G. Peano, “Sur une courbe, qui remplit toute une aire plane,” Mathematische Annalen, vol. 36, no. 1, pp. 157–160, 1890. [Online]. Available: https://link.springer.com/article/10.1007/BF01199438

[16] G. M. Morton, “A computer oriented geodetic data base and a new technique in file sequencing,” IBM Ltd., Tech. Rep., 1966. [Online]. Available: https://dominoweb.draco.res.ibm.com/ 0dabf9473b9c86d48525779800566a39.html

[17] E. N. Lorenz, “Deterministic nonperiodic flow,” Journal of the Atmospheric Sciences, vol. 20, no. 2, pp. 130–141, 1963. [Online]. Available: https://www.lmd.polytechnique.fr/<sup>∼</sup>dubos/Cours/PRL/Lorenz63.pdf

[18] D. H. Lehmer, “Mathematical methods in large-scale computing units,” Annals of the Computation Laboratory of Harvard University, vol. 26, pp. 141–146, 1951. [Online]. Available: https: //bitsavers.org/pdf/harvard/Proceedings of a Second Symposium on Large-Scale Digital Calculating Machinery Sep49.pdf

[19] J. H. Halton, “On the efficiency of certain quasi-random sequences of points in evaluating multi-dimensional integrals,” Numerische Mathematik, vol. 2, no. 1, pp. 84–90, 1960. [Online]. Available: https://doi.org/10.1007/BF01386213

[20] Y. LeCun, J. S. Denker, and S. A. Solla, “Optimal brain damage,” in Advances in Neural Information Processing Systems, vol. 2, 1990, pp. 598–605. [Online]. Available: https://proceedings.neurips.cc/paper/ 1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html

[21] B. Hassibi and D. G. Stork, “Optimal brain surgeon and general network pruning,” in IEEE International Conference on Neural Networks, 1993, pp. 293–299. [Online]. Available: https://doi.org/10.1109/ICNN.1993. 298572

[22] H. Li, A. Kadav, I. Durdanovic, H. Samet, and H. P. Graf, “Pruning filters for efficient convnets,” in International Conference on Learning Representations, 2017. [Online]. Available: https: //mlanthology.org/iclr/2017/li2017iclr-pruning

[23] M. Sun, Z. Liu, A. Bair, and J. Z. Kolter, “A simple and effective pruning approach for large language models,” in International Conference on Learning Representations, 2024. [Online]. Available: https://proceedings.iclr.cc/paper files/paper/2024/ hash/14c856c7a41297804de4c4890e846b25-Abstract-Conference.html

[24] Y. Lu, H. Cheng, Y. Fang, Z. Wang, J. Wei, D. Xu, Q. Xuan, X. Yang, and Z. Zhu, “Reassessing layer pruning in llms: New insights and methods,” in International Conference on Learning Representations, 2026, poster. [Online]. Available: https://proceedings.iclr.cc/paper files/paper/2026/hash/ 40624bef9a7d22c0c780dfe9291f5843-Abstract-Conference.html

[25] R. Lucas, K. Behdin, Z. Wang, Q. Song, S. Tang, and R. Mazumder, “Reasoning models can be accurately pruned via chain-of-thought reconstruction,” in NeurIPS 2025 Workshop on Efficient Reasoning, 2025. [Online]. Available: https://arxiv.org/abs/2509.12464

[26] T. Hoefler, D. Alistarh, T. Ben-Nun, N. Dryden, and A. Peste, “Sparsity in deep learning: Pruning and growth for efficient inference and training in neural networks,” Journal of Machine Learning Research, vol. 22, no. 241, pp. 1–124, 2021. [Online]. Available: https://www.jmlr.org/papers/v22/21-0366.html

[27] Z. Yao, R. Y. Aminabadi, M. Zhang, X. Wu, C. Li, and Y. He, “ZeroQuant: Efficient and affordable post-training quantization for large-scale transformers,” in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 27 168–27 183. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/2022/hash/ adf7fa39d65e2983d724ff7da57f00ac-Abstract-Conference.html

[28] C. Guo, Y. Qiu, J. Leng, X. Gao, C. Zhang, Y. Liu, F. Yang, Y. Zhu, and M. Guo, “SQuant: On-the-fly data-free quantization via diagonal hessian approximation,” in International Conference on Learning Representations, 2022. [Online]. Available: https: //iclr.cc/virtual/2022/poster/7037

[29] G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, and S. Han, “SmoothQuant: Accurate and efficient post-training quantization for large language models,” in Proceedings of the 40th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 202. PMLR, 2023, pp. 38 087–38 099. [Online]. Available: https://proceedings.mlr.press/v202/xiao23c.html

[30] J. Lin, J. Tang, H. Tang, S. Yang, W.-M. Chen, W.-C. Wang, G. Xiao, X. Dang, C. Gan, and S. Han, “AWQ: Activation-aware weight quantization for on-device LLM compression and acceleration,” in Proceedings of Machine Learning and Systems, vol. 6, 2024. [Online]. Available: https://proceedings.mlsys.org/paper files/paper/2024/ hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html

[31] T. Dettmers, R. Svirschevski, E. Kurtic et al., “Spqr: A sparsequantized representation for near-lossless llm weight compression,” in

International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum?id=Q1u25ah2M7

[32] J. Chee, Y. Cai, V. Kuleshov, and C. De Sa, “QuIP: 2-bit quantization of large language models with guarantees,” in Advances in Neural Information Processing Systems, vol. 36, 2023, pp. 4396–4429. [Online]. Available: https://neurips.cc/virtual/2023/poster/69982

[33] A. Romero, N. Ballas, S. E. Kahou, A. Chassang, C. Gatta, and Y. Bengio, “FitNets: Hints for thin deep nets,” in International Conference on Learning Representations, 2015. [Online]. Available: https://mlanthology.org/iclr/2015/romero2015iclr-fitnets/

[34] W. Park, D. Kim, Y. Lu, and M. Cho, “Relational knowledge distillation,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 3967–3976. [Online]. Available: https://openaccess.thecvf.com/content CVPR 2019/ html/Park Relational Knowledge Distillation CVPR 2019 paper.html

[35] Y. Gu, H. Zhou, F. Meng, J. Zhou, and M. Huang, “Knowledge distillation for pre-training language models,” in International Conference on Learning Representations, 2025. [Online]. Available: https://proceedings.iclr.cc/paper files/paper/2025/ hash/ea05e4fc0299c27648c9985266abad47-Abstract-Conference.html

[36] W. Xu, R. Han, Z. Wang, L. T. Le, D. Madeka, L. Li, W. Y. Wang, R. Agarwal, C.-Y. Lee, and T. Pfister, “Speculative knowledge distillation: Bridging the teacher-student gap through interleaved sampling,” in International Conference on Learning Representations, 2025. [Online]. Available: https://proceedings.iclr.cc/paper files/paper/2025/ hash/a2747a3844ca1e4667fbff3f558eb39b-Abstract-Conference.html

[37] Y. Kim, D. Shin, M. Kang, B. Na, and I.-c. Moon, “Distillation of large language models via concrete score matching,” in International Conference on Learning Representations, 2026. [Online]. Available: https://proceedings.iclr.cc/paper files/paper/2026/ hash/1bfc9f74afa91b9b8add5a97a97001a1-Abstract-Conference.html

[38] S. Stanton, P. Izmailov, P. Kirichenko, A. A. Alemi, and A. G. Wilson, “Does knowledge distillation really work?” in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 6906– 6919. [Online]. Available: https://proceedings.neurips.cc/paper/2021/ hash/376c6b9ff3bedbbea56751a84fffc10c-Abstract.html

[39] A. Novikov, D. Podoprikhin, A. Osokin, and D. P. Vetrov, “Tensorizing neural networks,” in Advances in Neural Information Processing Systems, vol. 28, 2015, pp. 442– 450. [Online]. Available: https://proceedings.neurips.cc/paper/2015/ hash/6855456e2fe46a9d49d3d3af4f57443d-Abstract.html

[40] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “Lora: Low-rank adaptation of large language models,” in International Conference on Learning Representations, 2022. [Online]. Available: https://iclr.cc/virtual/2022/poster/6319

[41] R. Saha, N. Sagan, V. Srivastava, A. J. Goldsmith, and M. Pilanci, “Compressing large language models using low rank and low precision decomposition,” in Advances in Neural Information Processing Systems, vol. 37, 2024. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/2024/hash/ a20e8451ffb07ad25282c21945ad4f19-Abstract-Conference.html

[42] M. Mozaffari, A. Yazdanbakhsh, and M. Mehri Dehnavi, “SLiM: One-shot quantization and sparsity with low-rank approximation for LLM weight compression,” in Proceedings of the 42nd International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 267. PMLR, 2025, pp. 45 024–45 049. [Online]. Available: https://proceedings.mlr.press/v267/mozaffari25a.html

[43] U. Saxena, S. Sharify, K. Roy, and X. Wang, “ResQ: Mixedprecision quantization of large language models with low-rank residuals,” in Proceedings of the 42nd International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 267. PMLR, 2025, pp. 53 095–53 114. [Online]. Available: https://proceedings.mlr.press/v267/saxena25b.html

[44] P. Walters, An introduction to ergodic theory. Springer Science & Business Media, 1982, vol. 79. [Online]. Available: https: //link.springer.com/book/9780387951522

[45] J. Fan, Z. Hao, J. Shen, S.-L. Jui, Y. Zhang, J.-X. Liao, and F.-L. Fan, “Compress any segment anything model (sam),” arXiv preprint arXiv:2507.08765, 2025. [Online]. Available: https: //arxiv.org/abs/2507.08765

[46] H. Wang, H. Pei, Y. Lyu, K. Zhang, L. Li, and F.-L. Fan, “Coli: A hierarchical efficient compressor for large images,” arXiv preprint arXiv:2507.11443, 2025. [Online]. Available: https://arxiv.org/abs/2507. 11443

[47] A. Katok and B. Hasselblatt, Introduction to the modern theory of dynamical systems. Cambridge University Press, 1995,

vol. 54. [Online]. Available: https://assets.cambridge.org/97805215/ 75577/frontmatter/9780521575577 frontmatter.pdf

[48] L. Coroianu and S. G. Gal, “Lp-approximation by truncated max-product sampling operators of kantorovich-type based on fejer kernel,” The Journal of Integral Equations and Applications, vol. 29, no. 2, pp. 349– 364, 2017.

[49] W. M. Schmidt, Diophantine Approximation, ser. Lecture Notes in Mathematics. Springer, 1980, no. 785.

[50] T. E. Hull and A. R. Dobell, “Random number generators,” SIAM Review, vol. 4, no. 3, pp. 230–254, 1962. [Online]. Available: https://doi.org/10.1137/1004061

[51] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei, “ImageNet Large Scale Visual Recognition Challenge,” International Journal of Computer Vision, vol. 115, no. 3, pp. 211–252, 2015. [Online]. Available: https://doi.org/10.1007/s11263-015-0816-y

[52] PyTorch, “torchvision.models.resnet18,” https://docs.pytorch.org/ vision/stable/models/generated/torchvision.models.resnet18.html, 2026, accessed: 2026-07-27.

[53] Qwen Team, A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang et al., “Qwen2.5 technical report,” arXiv preprint arXiv:2412.15115, 2024. [Online]. Available: https://arxiv.org/abs/2412.15115

[54] J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang et al., “Qwen technical report,” arXiv preprint arXiv:2309.16609, 2023. [Online]. Available: https: //arxiv.org/abs/2309.16609

[55] S. Merity, C. Xiong, J. Bradbury, and R. Socher, “Pointer sentinel mixture models,” in International Conference on Learning Representations, 2017. [Online]. Available: https://mlanthology.org/iclr/ 2017/merity2017iclr-pointer/

[56] J. Welbl, N. F. Liu, and M. Gardner, “Crowdsourcing multiple choice science questions,” arXiv preprint arXiv:1707.06209, 2017. [Online]. Available: https://aclanthology.org/W17-4413/

[57] C. Clark, K. Lee, M.-W. Chang, T. Kwiatkowski, M. Collins, and K. Toutanova, “BoolQ: Exploring the surprising difficulty of natural yes/no questions,” in Proceedings of NAACL-HLT, 2019, pp. 2924–2936. [Online]. Available: https://aclanthology.org/N19-1300/

[58] D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt, “Measuring massive multitask language understanding,” in International Conference on Learning Representations, 2021. [Online]. Available: https://iclr.cc/virtual/2021/poster/2962

[59] K. Sakaguchi, R. Le Bras, C. Bhagavatula, and Y. Choi, “WinoGrande: An adversarial winograd schema challenge at scale,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 5, 2020, pp. 8732–8740. [Online]. Available: https://aaai.org/papers/ 08732-winogrande-an-adversarial-winograd-schema-challenge-at-scale/

[60] R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi, “HellaSwag: Can a machine really finish your sentence?” in Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 2019, pp. 4791–4800. [Online]. Available: https://aclanthology.org/P19-1472/

[61] Y. Bisk, R. Zellers, R. Le Bras, J. Gao, and Y. Choi, “PIQA: Reasoning about physical commonsense in natural language,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 5, 2020, pp. 7432–7439. [Online]. Available: https: //doi.org/10.1609/aaai.v34i05.6239

[62] A. Gordon, Z. Kozareva, and M. Roemmele, “SemEval-2012 task 7: Choice of plausible alternatives: An evaluation of commonsense causal reasoning,” in Proceedings of the First Joint Conference on Lexical and Computational Semantics. Association for Computational Linguistics, 2012, pp. 394–398. [Online]. Available: https://aclanthology.org/ S12-1052/

[63] P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord, “Think you have solved question answering? try ARC, the AI2 reasoning challenge,” arXiv preprint arXiv:1803.05457, 2018. [Online]. Available: https://arxiv.org/abs/1803.05457

[64] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, and J. Schulman, “Training verifiers to solve math word problems,” arXiv preprint arXiv:2110.14168, 2021. [Online]. Available: https: //arxiv.org/abs/2110.14168

[65] A. Wang, Y. Pruksachatkun, N. Nangia, A. Singh, J. Michael, F. Hill, O. Levy, and S. R. Bowman, “SuperGLUE: A stickier benchmark for general-purpose language understanding systems,” in Advances in Neural Information Processing Systems, vol. 32, 2019. [Online]. Available: https://papers.nips.cc/paper files/paper/2019/hash/ 4496bf24afe7fab6f046bf4923da8de6-Abstract.html

# Supplementary Materials of “Dynamics as Code: On Model Compression via Dynamic System”

Fan Gao, Wei Su\*, Juntong Fan\*, Renfeng Peng, Hongyu Liu, Jinqiao Duan, Feng-Lei Fan

Index Terms—Model compression, dynamic systems, neural networks, parameter representation.

## I. DETAILED SPACE-FILLING CURVE CONSTRUCTIONS

This section provides the detailed index-to-coordinate constructions of the four space-filling mappings used in the main manuscript. All four mappings define deterministic bijections between a scalar index and a finite D-dimensional lattice. The main manuscript retains their common formulation and compression role, while the explicit traversal rules are given below for completeness.

## A. Hilbert Traversal

For the Hilbert traversal, let $m = 2 ^ { k }$ . For a scalar index $\xi \in \{ 0 , \ldots , 2 ^ { k D } - 1 \}$ , we define

$$
h _ { 0 } = \left\lfloor { \frac { \xi } { 2 ^ { ( k - 1 ) D } } } \right\rfloor , \qquad \xi ^ { \prime } = \xi \bmod 2 ^ { ( k - 1 ) D } ,\tag{S1}
$$

where $h _ { 0 } \in \{ 0 , \ldots , 2 ^ { D } - 1 \}$ denotes the first-level subcube index, and $\xi ^ { \prime }$ is the remaining index at the next recursion level. The Hilbert lattice state can be written abstractly as

$$
\Phi _ { \mathrm { H i l b e r t } } ^ { ( D , k ) } ( \boldsymbol { \xi } ) = 2 ^ { k - 1 } \mathbf { v } _ { h _ { 0 } } + \mathcal { R } _ { h _ { 0 } } ^ { ( k - 1 ) } \left( \Phi _ { \mathrm { H i l b e r t } } ^ { ( D , k - 1 ) } ( \boldsymbol { \xi } ^ { \prime } ) \right) ,\tag{S2}
$$

with $\Phi _ { \mathrm { H i l b e r t } } ^ { ( D , 0 ) } ( 0 ) ~ = ~ { \bf 0 }$ . Here, ${ \bf v } _ { h _ { 0 } } \in \{ 0 , 1 \} ^ { D }$ denotes the subcube position, and $\mathcal { R } _ { h _ { 0 } } ^ { ( k - 1 ) }$ reorients the lower-order Hilbert pattern within the subcube via coordinate permutations and axis reflections to preserve continuity and locality. Thus,

$$
\Phi _ { \mathrm { H i l b e r t } } ^ { ( D , k ) } ( \xi ) \in \{ 0 , \dots , 2 ^ { k } - 1 \} ^ { D } .\tag{S3}
$$

## B. Peano Traversal

For the Peano traversal, let $m \ : = \ : 3 ^ { k }$ . The scalar index $\xi \in \{ 0 , \ldots , 3 ^ { k D } - 1 \}$ is defined as

$$
p _ { 0 } = \left\lfloor { \frac { \xi } { 3 ^ { ( k - 1 ) D } } } \right\rfloor , \quad \xi ^ { \prime } = \xi \bmod 3 ^ { ( k - 1 ) D } ,\tag{S4}
$$

where $p _ { 0 } \in \{ 0 , \ldots , 3 ^ { D } - 1 \}$ denotes the first-level subblock index.The corresponding lattice Peano state is written recursively as

$$
\Phi _ { \mathrm { P e a n o } } ^ { ( D , k ) } ( \xi ) = 3 ^ { k - 1 } \mathbf { v } _ { p _ { 0 } } + \mathcal { R } _ { p _ { 0 } } ^ { ( k - 1 ) } \left( \Phi _ { \mathrm { P e a n o } } ^ { ( D , k - 1 ) } ( \xi ^ { \prime } ) \right) ,\tag{S5}
$$

with $\Phi _ { \mathrm { P e a n o } } ^ { ( D , 0 ) } ( 0 ) = \mathbf { 0 }$ . Here, $\mathbf { v } _ { p _ { 0 } } \in \{ 0 , 1 , 2 \} ^ { D }$ denotes the geometric position of the $p _ { 0 }$ -th first-level subblock, and $\mathcal { R } _ { p _ { 0 } } ^ { ( k - 1 ) }$ denotes the corresponding orientation operation. Thus,

$$
\Phi _ { \mathrm { P e a n o } } ^ { ( D , k ) } ( \xi ) \in \{ 0 , \dots , 3 ^ { k } - 1 \} ^ { D } .\tag{S6}
$$

## C. Morton/Z-order Traversal

Morton/Z-order uses bit interleaving, providing a simpler indexing rule but weaker locality near block boundaries. Let $m = 2 ^ { k }$ be the grid resolution per-axis, where k is the number of bits used for each coordinate. For a scalar index $\xi \in \{ 0 , \ldots , m ^ { D } - 1 \}$ , we write its binary representation as

$$
\xi = \sum _ { r = 0 } ^ { k D - 1 } b _ { r } ( \xi ) 2 ^ { r } , \qquad b _ { r } ( \xi ) \in \{ 0 , 1 \} .\tag{S7}
$$

Under the Morton bit-interleaving convention, the d-th coordinate is obtained by collecting every D-th bit:

$$
q _ { d } ( \xi ) = \sum _ { r = 0 } ^ { k - 1 } b _ { r D + d } ( \xi ) 2 ^ { r } , \quad d = 0 , \ldots , D - 1 .\tag{S8}
$$

Accordingly, the state generated by the Morton/Z-order system is

$$
\Phi _ { \mathrm { M o r t o n } } ^ { ( D , k ) } ( \xi ) = \left( q _ { 0 } ( \xi ) , \dots , q _ { D - 1 } ( \xi ) \right) \in \{ 0 , \dots , m - 1 \} ^ { D } .\tag{S9}
$$

## D. Snake Traversal

Snake traversal follows a regular alternating scan pattern and provides a low-complexity grid-ordering rule. Let $m = 2 ^ { k }$ denote the per-axis grid resolution. For a scalar index $\xi \in \mathbf { \Xi }$ $\{ 0 , \ldots , m ^ { D } - 1 \}$ , we first obtain the index coordinates $\mathbf { c } ( \xi ) =$ ${ \big ( } c _ { 1 } ( \xi ) , \dots , c _ { D } ( \xi ) { \big ) }$ , where

$$
\xi = \sum _ { j = 1 } ^ { D } c _ { j } ( \xi ) m ^ { j - 1 } , \qquad c _ { j } ( \xi ) = \left\lfloor { \frac { \xi } { m ^ { j - 1 } } } \right\rfloor { \bmod { m } } .\tag{S10}
$$

For $j = 1 , \ldots , D - 1$ , the scan direction along the coordinate $j$ is determined by the parity of the next higher-order coordinate:

$$
c _ { j } ^ { \prime } ( \xi ) = { \left\{ \begin{array} { l l } { c _ { j } ( \xi ) , } & { c _ { j + 1 } ( \xi ) \equiv 0 { \pmod { 2 } } , } \\ { m - 1 - c _ { j } ( \xi ) , } & { c _ { j + 1 } ( \xi ) \equiv 1 { \pmod { 2 } } , } \end{array} \right. }\tag{S11}
$$

and $c _ { D } ^ { \prime } ( \xi ) = c _ { D } ( \xi )$ . Thus,

$$
\Phi _ { \mathrm { S n a k e } } ^ { ( D , k ) } ( \xi ) = \left( c _ { 1 } ^ { \prime } ( \xi ) , \ldots , c _ { D } ^ { \prime } ( \xi ) \right) \in \{ 0 , \ldots , m - 1 \} ^ { D } .\tag{S12}
$$

## II. ANALYSIS EXPERIMENTS

In this section, we supplement the global and method-wise Pareto analysis to support the representative configurations.

Pareto Analysis. We jointly analyze the performance of the compression ratio and the accuracy drop using global and method-wise Pareto frontiers in Figure S2 and Figure S1. A hyper-parameter configuration is Pareto-optimal only if no other configuration achieves an equal or higher compression ratio and an equal or lower accuracy drop.

![](images/8c6675dd88e3fae9c3238471028f0eefe489bd57bec5d31c838faa3805338115.jpg)

![](images/56d344df95bb3c2f9848af2fe559cdc7dae008d73125b2df0b35dc89516a89aa.jpg)

Fig. S1. Compression–accuracy method-wise Pareto frontiers on ResNet-18/ImageNet. The uncompressed model achieves 69.76% Top-1 accuracy. For clarity, only configurations with a Top-1 accuracy drop of at most 4.0 percentage points are shown in panel (a). Panel (b) enlarges the high-fidelity regio indicated by the dashed rectangle in panel (a).  
![](images/0f383324a7751e8bc2a64a4837a3ec7846ed61777a09cb84f2aa4cc0c4c57252.jpg)  
Fig. S2. Compression–accuracy global Pareto analysis on ResNet-18/ImageNet. The uncompressed model achieves 69.76% Top-1 accuracy. For clarity, configurations with a Top-1 accuracy drop greater than 4.0 percentage points are omitted.

Figure S2 presents the Pareto frontier of all methods with all hyper-parameter configurations. Because the Hilbert, Morton/Z-order, and Snake curves produce identical results under matched configurations, their Pareto frontiers coincide. We only annotate Hilbert to avoid visual clutter. No single method dominates the Pareto frontier in Figure S2, i.e., no single method is completely superior to the rest. However, in the high-fidelity and moderate-compression regions, the frontier is jointly supported by Peano, LCG, Hilbert, and QMC, and neither Lorenz nor PCG contributes a point to the global frontier. This means that in this region, Peano, LCG, Hilbert, and QMC are better.

Figure S1 presents the Pareto frontier of each method under different hyper-parameter configurations, which characterizes the performance of each method. Hilbert and Peano exhibit relatively stable frontiers but occupy different regions because of their different resolution schedules. QMC and LCG present broader but less regular frontiers, and this frontier spans a large compression range. This non-monotonic character means that the optimal configuration of D and S may appear in different places. PCG provides a limited effective frontier, only D = 2 achieves a competitive compression– accuracy trade-off. Lorenz forms a narrow frontier with stable accuracy preservation but limited compression capacity.

As Figure S1 shows, since a Pareto frontier does not define a unique optimum, we select one representative configuration for each method from its Pareto frontier near the knee region. Specifically, we prioritize an approximate compression ratio range of 5× with a drop in Top-1 accuracy below 0.6%. However, Lorenz does not reach this compression region, we select its highest-compression configuration that maintains near-lossless accuracy. The representative configurations are reported in the main manuscript. Excluding Lorenz, representative configurations achieve the compression ratio between 4.80–5.77×, with the Top-1 accuracy drop between 0.14– 0.59%. Lorenz achieves a 3.36× compression ratio with a 0.12% accuracy drop.