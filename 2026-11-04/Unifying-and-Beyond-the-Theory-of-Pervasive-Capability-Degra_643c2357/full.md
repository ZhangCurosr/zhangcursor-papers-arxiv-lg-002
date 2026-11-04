# Unifying and Beyond the Theory of Pervasive Capability Degradation in LLM Post-training SFT from Feature Interference

Hang Chen College of Computing and Data Science Nanyang Technological University chen.hang@ntu.edu.sg

Jiaying Zhu School of Computer Science and Engineering The Chinese University of Hong Kong jyzhu24@cse.cuhk.edu.hk

Wenya Wang<sup>∗</sup> College of Computing and Data Science Nanyang Technological University wangwy@ntu.edu.sg

## Abstract

Pervasive Capability Degradation (PCD) refers to the phenomenon in which supervised fine-tuning (SFT), while adapting a large language model (LLM) to a target task, often erodes the general capabilities acquired during pre-training. Prevailing explanations attribute PCD to macroscopic distribution shift and therefore lack microscopic manipulability, leaving existing mitigation methods without a complete theoretical foundation. In this paper, we first develop a bottom-up geometric theory of PCD grounded in feature interference under superposition: narrow-domain SFT data disrupt the almost-orthogonal feature structure formed during pre-training, and the resulting interference drives activations across decision hyperplanes in the output Laguerre–Voronoi geometry, flipping predictions on general tasks. This theory unifies existing methods into six paradigms and reveals both the source of their efficacy and their theoretical limits. We then prove that cross-entropy under one-hot supervision cannot resolve PCD on its own, and accordingly propose the AEGIS loss, which directly constrains the drift of activation margins along all competing token directions, thereby decoupling interference suppression from task adaptation. Across 3 model families and 14 SFT tasks, AEGIS outperforms representative methods of every paradigm on target tasks and pervasive capabilities simultaneously, at only about 1.3× the computational cost of standard SFT. Its generality, scalability, and mechanistic interpretability offer a new perspective on fully resolving PCD. Codes are avaliable at https://github.com/Zodiark-ch/SFT-Conflict.

## 1 Introduction

Large-scale pre-training endows LLMs with broad general skills, which we refer to as pervasive task capabilities [48, 52]. Following pre-training, supervised fine-tuning (SFT) serves as the primary approach for adapting general base models into specialized agents [72, 22]. Simply put, posttraining SFT aims to learn a downstream target task while preserving the model’s pre-trained pervasive capabilities [31, 27]. However, SFT on a target task often degrades these pervasive capabilities—a phenomenon variously known as catastrophic forgetting [29, 58], alignment tax [18,

84], or representation collapse [53, 80], which we unify under the term Pervasive Capability Degradation (PCD). Rooted in the fundamental conflict between acquiring new skills and retaining existing ones, PCD remains a long-standing challenge in deep learning and a major bottleneck across transfer learning [50], continual learning [88], multi-task learning [32], and model merging.

Currently, the prevailing explanation for PCD in the deep learning community is distribution conflict [65, 13]: as a model adapts to a target task, its conditional output distribution $p ( y \mid x )$ shifts toward the target distribution, thereby disrupting the original pervasive task distribution. Based on this view, numerous mitigation strategies have been proposed. For instance, data-centric approaches collect large sets of pervasive task samples and train on them alongside target task data, attempting to maintain the pervasive distribution while fitting the target distribution [82, 75]. Alternatively, algebraic approaches constrain parameter updates or loss gradients of the new task to be orthogonal to those of existing tasks, reducing interference with the pervasive distribution [33, 40]. In our prior exploratory work [4, 7, 5, 6], we also approached this problem from a mechanistic interpretability perspective, proposing the Mechanistic Interpretability and Localization (MIL) paradigm. Specifi cally, by using techniques such as circuit discovery to identify whether neurons in parameter space are relevant to the target task, we advocated updating only task-relevant neurons while freezing irrelevant ones to reduce the impact on the pervasive distribution.

However, the distribution conflict theory describes the overall shift of $p ( y \mid x )$ only at a macroscopic statistical level. Although metrics such as KL divergence [28], Wasserstein distance [30], or the Fisher information matrix [49] can model these shifts mathematically, they lack microscopic manipulability. For example, macroscopic distributions cannot tell us how conflicts arise within internal activations, how to locally adapt distributions without collateral damage, or why standard cross-entropy gradients fail to reach an ideal distribution $p ( y \mid x )$ that satisfies both tasks. Due to this lack of microscopic manipulability, existing methods rely on heuristic external constraints—such as above data mixing or gradient orthogonalization—to indirectly ease the conflict. While intuitive and supported by macroscopic theory, these constraints lack a complete microscopic control chain and thus only partially alleviate PCD. Taking our prior MIL work as an example, because the residual stream has a finite dimension, the model compresses a vast number of semantic features in superposition [15], causing severe polysemantic entanglement between physical parameters and semantic features. Consequently, simply partitioning, pruning, or orthogonalizing weights in a static weight space cannot fully decouple target task features from irrelevant pervasive features at the parameter level [23]. This limitation shows that resolving PCD requires moving beyond the static weight space to directly manipulate dynamic hidden activation manifolds and output geometry.

Motivated by this insight, we develop a bottom-up, formal geometric theory to explain PCD at the microscopic level, unify existing PCD mitigation paradigms to identify their theoretical limits, and propose a loss function in activation geometry to directly address PCD.

Specifically, under the superposition [15] and linear representation hypotheses [46], hidden activations can be linearly decomposed into a combination of sparse semantic features. During pre-training, high data diversity ensures extreme feature sparsity, so optimization is dominated by the 1-sparse loss $( L _ { 1 } )$ and arranges features into an almost-orthogonal uniform polyhedral configuration. During SFT, however, narrow domain data shifts the dominant objective to higher-order losses $( L _ { N } )$ and triggers a geometric phase transition, which merges correlated features to significantly reduce feature orthogonality. This loss of orthogonality induces strongfeature interference when evaluating pervasive tasks. In the output Laguerre-Voronoi Diagram (LVD) space [43], this interference drives activations to drift toward competing token polyhedra coupled with these interfering features; crossing a decision hyperplane causes an incorrect competing token to win, resulting in PCD. Building on this theory, we categorize existing PCD mitigation methods into six paradigms and explain both their effectiveness and their theoretical limits: (1) Reference Distribution Anchoring (RDA) [57, 51] constrains the KL divergence from the base model’s output distribution to maintain the pervasive distribution, but suffers from a trade-off between target adaptation and reference preservation; (2) Data Mixture Alignment (DMA) [90, 54] collects pervasive samples to approximate the pervasive distribution and increase feature sparsity against phase collapse, but faces an orders-of-magnitude sample deficit that leaves long-tail features unsupervised; (3) Parameter-Space Orthogonalization (PSO) [62, 19] forces new task parameters to be orthogonal to existing ones to reduce interference cost, but suffers from severe attenuation as parameter orthogonality translates into feature orthogonality across non-linear activations; (4) Delta-Parameter Pruning (DPP) [85, 11] and (5) Mechanistic Interpretability and Localization (MIL) [7, 6] prune most SFT parameter updates or freeze task-irrelevant neurons to restrict anisotropic activation drift, yet neither can fully decouple the noise effect of feature interference; and (6) Gradient Orthogonalization Optimization (GOO) [41, 3] constrains target task gradients to be orthogonal to other task gradients to limit interference, but is bottlenecked by the finite degrees of freedom of the gradient orthogonal basis.

Table 1: Summary of primary notations and symbols used throughout this paper.
<table><tr><td></td><td>Notation Domain / Dimension Description</td><td></td></tr><tr><td colspan="3">1. Model Architecture &amp; Optimization</td></tr><tr><td> $\theta$ </td><td>Parameter space</td><td>Model parameter set</td></tr><tr><td> $\mathcal { D }$ </td><td>Dataset</td><td>Dataset of training samples</td></tr><tr><td> $N$ </td><td> $\mathbb { N } ^ { + }$ </td><td>Total number of samples in a dataset or feature set</td></tr><tr><td> $\nu , V$ </td><td> $\mathrm { S e t , \mathbb { N } ^ { + } }$ </td><td>Output vocabulary set and vocabulary size  $( V = | \nu | )$ </td></tr><tr><td> $W , b$ </td><td>Tensor / Vector</td><td>General model weight matrices and bias vectors</td></tr><tr><td> $h$ </td><td> $\mathbb { R } ^ { m }$ </td><td>Residual stream activation vector</td></tr><tr><td> $m$ </td><td> $\mathbb { N } ^ { + }$ </td><td>Hidden dimension of the activation space  $( m \ll k )$ </td></tr><tr><td> $\mathcal { L }$ </td><td> $\mathbb { R }$ </td><td>Loss function</td></tr><tr><td colspan="3">2. Feature Superposition Theory</td></tr><tr><td> $k$ </td><td> $\bar { \mathbb { N } } ^ { + }$ </td><td>Total number of latent features in superposition  $( k \gg m )$ </td></tr><tr><td> $f , F$ </td><td> $\mathbb { R } ^ { m } , \mathbb { R } ^ { m \times k }$ </td><td>Unit feature direction vector and feature dictionary matrix  $F = [ f _ { 1 } , \ldots , f _ { k } ]$ </td></tr><tr><td> $a$ </td><td> $\mathbb { R } ^ { k }$ </td><td>Feature activation intensity vector  $a = [ a _ { 1 } , \ldots , a _ { k } ] ^ { \intercal }$ </td></tr><tr><td colspan="3">3. Planar Laguerre-Voronoi Diagram (LVD)</td></tr><tr><td> $c _ { j } , \nu _ { j }$ </td><td> $\bar { \mathbb { R } } ^ { m } , \mathbb { R }$ </td><td>Geometric center  $( c _ { j } \in \mathbb { R } ^ { m } )$  and Laguerre scalar weight  $( \nu _ { j } \in \mathbb { R } )$  of token j</td></tr></table>

Furthermore, we apply our theory to analyze the standard cross-entropy loss in SFT, proving that under one-hot supervision, the effect of feature interference is tightly coupled with task adaptation. Consequently, cross-entropy can only prevent geometric feature decay by mimicking pre-training with massive pervasive samples. To overcome this limitation, we propose the AEGIS (Adaptive Equipartition of Geometric Interference in Superposition) loss. By supervising activation drift across all competing token directions instead of relying solely on a discrete categorical probability distribution, AEGIS both decouples the feature interference effect from task adaptation and greatly reduces the number of samples needed for full feature coverage. This raises the cost of destructive geometric phase transitions and preserves the geometric structure of pervasive features. Empirically, we conduct comprehensive evaluations across the Mistral, Qwen, and Gemma model families on mainstream SFT tasks, including natural language understanding, arithmetic reasoning, code generation, and logical reasoning. Compared to existing paradigms, AEGIS consistently improves target task performance while preserving or even enhancing pervasive task capabilities, demonstrating the potential of our theoretical framework to resolve PCD.

In summary, our main contributions are twofold:

• We bridge activation space, parameter space, and gradient feedback with macroscopic performance conflicts, explaining from the bottom up why SFT induces PCD. Using this theory, we also unify existing PCD mitigation methodologies into six paradigms, explain why they are effective, and derive their respective theoretical limits.

• We prove through our theory that while standard cross-entropy is effective for task adaptation, it cannot independently resolve PCD under one-hot supervision. Building on this insight, we propose AEGIS, a loss function that leverages token LVD geometry to simultaneously constrain multiple token directions, and demonstrate its strong empirical performance across diverse post-training scenarios.

## 2 Preliminaries

To keep the theoretical framework concise and accessible across different domains, we summarize the primary mathematical symbols in Table 1.

## 2.1 LLM Post-Training Supervised Fine-Tuning

Let $\mathcal { M } _ { \theta ^ { 0 } }$ denote a foundation LLM parameterized by $\theta ^ { 0 }$ pre-trained on a general corpus $\mathcal { D } _ { G }$ , endowing $\mathcal { M } _ { \theta ^ { 0 } }$ with pervasive capabilities. To adapt the model to a downstream target task $T$ with dataset $\mathcal { D } _ { T } = \{ ( x _ { i } , y _ { t , i } ) \} _ { i = 1 } ^ { N _ { T } }$ , supervised fine-tuning (SFT) updates parameters $\theta ^ { 0 } \to \theta$ to master $T$ while preserving pervasive proficiency on $\mathcal { D } _ { G } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N _ { G } }$ . This adaptation ideally optimizes the joint objective:

$$
\operatorname* { m i n } _ { \theta } \mathbb { E } _ { ( x , y _ { t } ) \sim \mathcal { D } _ { T } } [ \mathcal { L } ( y _ { t } | x ; \theta ) ] + \mathbb { E } _ { ( x , y ) \sim \mathcal { D } _ { G } } [ \mathcal { L } ( y | x ; \theta ) ] ,\tag{1}
$$

where $\begin{array} { r } { \mathcal { L } ( y | x ; \theta ) = - \sum _ { \tau = 1 } ^ { | y | } } \end{array}$ log $P _ { \theta } ( y _ { \tau } \mid x , y _ { < \tau } )$ is the standard autoregressive cross-entropy loss. In practice, optimizing $\operatorname { E q . }$ . (1) directly is hindered by restricted access to $\mathcal { D } _ { G }$ and optimization conflicts, inevitably triggering capability degradation:

Definition 1 (Pervasive Capability Degradation, PCD). Let $\mathcal { P } ( \mathcal { M } _ { \theta } , \mathcal { D } )$ denote the evaluation performance of model $\mathcal { M } _ { \theta }$ on distribution D. SFTfrom $\theta ^ { 0 }$ to θ suffers from Pervasive Capability Degradation (PCD) iftarget adaptation incurs an intolerable performance drop on $\mathcal { D } _ { G } .$

$$
\begin{array} { r } { \mathcal { P } ( \mathcal { M } _ { \theta } , \mathcal { D } _ { T } ) > \mathcal { P } ( \mathcal { M } _ { \theta ^ { 0 } } , \mathcal { D } _ { T } ) \quad a n d \quad \mathcal { P } ( \mathcal { M } _ { \theta ^ { 0 } } , \mathcal { D } _ { G } ) - \mathcal { P } ( \mathcal { M } _ { \theta } , \mathcal { D } _ { G } ) > \epsilon _ { P C D } , } \end{array}\tag{2}
$$

where $\epsilon _ { P C D } > 0$ denotes the degradation tolerance threshold.

## 2.2 The Superposition Hypothesis and Feature Interference

Because the number of latent semantic features k learned by LLMs vastly exceeds the hidden dimension m $( k \gg m )$ , models rely on the superposition hypothesis [15] to represent overcomplete feature sets within restricted residual streams.

Definition 2 (Feature Superposition). Under the linear representation hypothesis, an activation $h \in \mathbb { R } ^ { m }$ in the residual stream is expressed via afeature dictionary $F = [ \overleftarrow { f } _ { 1 } ^ { \cdot } , \dots , f _ { k } ] \in \mathbb { R } ^ { m \times k }$ with unit directions $( \Vert f _ { i } \Vert _ { 2 } = 1 )$ and an activation intensity vector $a = [ a _ { 1 } , \ldots , a _ { k } ] ^ { \intercal } \in \mathbb { R } ^ { k }$

$$
h = F a = \sum _ { i = 1 } ^ { k } a _ { i } f _ { i } , \quad w h e r e \ k \gg m .\tag{3}
$$

Superposition is enabled by feature sparsity: each feature activates with low marginal probability $P ( a _ { i } \ne 0 ) = 1 - S _ { i } \ll 1$ , where $S _ { i } \to 1$

Since $k \gg m$ , pairwise orthogonality across all features is geometrically impossible, guaranteeing non-zero overlap: $\exists i \neq j , f _ { i } ^ { \top } f _ { j } \neq \dot { 0 }$ . Probing feature i along its designated direction $f _ { i }$ yields:

$$
{ f _ { i } ^ { \top } } h = f _ { i } ^ { \top } \left( a _ { i } f _ { i } + \sum _ { j \ne i } a _ { j } f _ { j } \right) = a _ { i } + \sum _ { j \ne i } a _ { j } ( f _ { i } ^ { \top } f _ { j } ) ,\tag{4}
$$

where $a _ { i }$ is the target signal, and $\textstyle \sum _ { j \neq i } a _ { j } ( f _ { i } ^ { \top } f _ { j } )$ is the interference from non-orthogonal features. High sparsity $( S  1 )$ sustains superposition by suppressing co-activation, rendering concurrent interference quadratically rare: $P ( \acute { a } _ { i } > 0 , a _ { j } > \acute { 0 } ) = \overset { \cdot } { \mathcal { O } } ( ( 1 - \overset { \cdot } { S } ) ^ { 2 } ) \ll 1 - S$

In actual LLMs, physical parameter channels couple with latent features in a non-uniform and sparse manner, substantiated by two core observations. First, distinct channels $( \mathrm { i . e . }$ , individual rows or columns of a weight matrix W) produce divergent output activations for the identical input state, demonstrating that individual channels allocate disparate projection weights across features. Second, each channel’s sensitivity is heavily concentrated on a small subset of salient features while remaining negligible on the rest of features, a localized selectivity broadly corroborated by mechanistic interpretability literature [20, 39, 21].

To enable mathematically tractable derivations that directly map parameter-space updates (such as norm variations, spatial rotations, and subspace orthogonalizations) to directional feature displacements in the residual stream, we formalize this localized, sparse physical coupling into an idealized one-to-one virtual parameterization.

Hypothesis 1 (Virtual Sparse Weight Hypothesis). Under a first-order linear approximation, the composite transformation ofan intermediate Transformer layer (such as an Attention, MLP block or both) is governed by an end-to-end operator $\dot { M } = W _ { i n } \dot { W } _ { o u t } \in \mathbb { R } ^ { m \times m }$ , where $W _ { i n } \in \mathbb { R }$ m×d and $W _ { o u t } \in \mathbb { R } ^ { d \times m }$ with d being the dimension of the Attention or MLP module. We posit the existence ofan overcomplete virtual parameter space wherein M factorizes losslessly into idealized feature-decoupled operators:

$$
M = W _ { i n } ^ { \nu i r t } W _ { o u t } ^ { \nu i r t } , w h e r e W _ { o u t } ^ { \nu i r t } \in \mathbb { R } ^ { k \times m } a n d W _ { i n } ^ { \nu i r t } \in \mathbb { R } ^ { m \times k } .\tag{5}
$$

<sup>2</sup> Each virtual parameter row $W _ { i } \in \mathbb { R } ^ { 1 \times m }$ corresponds bijectively tofeature direction $f _ { i } \in \mathbb { R } ^ { m }$

![](images/053974cee6def3e6bf6aa25a1ed7313236a0650d8459d6d43157a0f9d1bd7b82.jpg)  
Figure 1: Illustration of real weight (W) and virtual space weight $( W ^ { v i r t } )$ . Each row of $W$ manages a subset of a few salient features, but each row of $W ^ { v i r t }$ manages one individual feature.

Through Hypothesis 1, operations performed on individual rows of the parameter matrix translate equally to features in activation space, as visualized in Figure 1. This bridge yields two foundational algebraic properties (detailed proofs provided in Appendix A): Orthogonal Equivalence: $W _ { a } \cdot W _ { b } =$ $0 \ { \overset { \cdot } { \Rightarrow } } \ f _ { a } ^ { \top } \ f _ { b } ^ { \ } \ { \overset { \cdot } { = } } \ 0$ for any two feature channels $\ddot { a } , b \in \{ 1 , . . . , k \}$ , and Directional Additivity: $\begin{array} { r } { f _ { a } ^ { \prime } \propto f _ { a } + \lambda \left( \frac { \| W _ { b } \| _ { 2 } } { \| W _ { a } \| _ { 2 } } \right) } \end{array}$ f<sub>b</sub> when $W _ { a } ^ { \prime } = W _ { a } + \lambda W _ { b }$ with $\lambda \in \mathbb { R }$

## 2.3 Laguerre Geometry of LLMs

In autoregressive LLMs, the unembedding layer parameterized by $W _ { U } = [ W _ { U , 1 } , \dotsc , W _ { U , V } ] \in$ $\mathbb { R } ^ { m \times V }$ and $b _ { U } = [ b _ { U , 1 } , \dotsc , b _ { U , V } ] ^ { \top } \in \mathbb { R } ^ { V }$ maps activation $\boldsymbol { h } \in \mathbb { R } ^ { m }$ to logits $z _ { j } ( h ) = W _ { U , j } ^ { \top } h + b _ { U , j }$ predicting token $\hat { y } = \arg \operatorname* { m a x } _ { j \in \mathcal { V } } z _ { j } ( h )$ . The decision rule $z _ { j } ( h ) \geq z _ { k } ( h )$ defines an intersection of affine half-spaces:

$$
( W _ { U , j } - W _ { U , k } ) ^ { \top } h + ( b _ { U , j } - b _ { U , k } ) \ge 0 , \quad \forall k \in \mathcal { V } \setminus \{ j \} ,\tag{6}
$$

tessellating $\mathbb { R } ^ { m }$ into convex polyhedral cells equivalent to a Laguerre-Voronoi Diagram (LVD).

Definition 3 (Laguerre-Voronoi Diagram, LVD). An LVD (or power diagram) partitions $\mathbb { R } ^ { m }$ using weighted sites $\{ ( \boldsymbol { c } _ { j } , \boldsymbol { \nu } _ { j } ) \} _ { j = 1 } ^ { V }$ with centers $c _ { j } \in \mathbb { R } ^ { m }$ and scalar weights $\nu _ { j } \in \mathbb { R }$ . The power distance from activation $h \in \bar { \mathbb { R } } ^ { m }$ to site j is defined as:

$$
D _ { j } ( h ) = \| h - c _ { j } \| _ { 2 } ^ { 2 } - \nu _ { j } .\tag{7}
$$

The convex polyhedral cellfor token $j$ is given by:

$$
C e l l _ { j } ^ { L V D } = \left\{ h \in \mathbb { R } ^ { m } \mid D _ { j } ( h ) \leq D _ { k } ( h ) , \forall k \in \mathcal { V } \right\} .\tag{8}
$$

Locating the minimal power distance cell strictly equates to selecting the maximum logit $\begin{array} { r } { \mathrm { ( a r g m i n } _ { j \in \mathcal { V } } D _ { j } ( h ) \equiv \arg \operatorname* { m a x } _ { j \in \mathcal { V } } z _ { j } ( h ) \colon } \end{array}$ ) via the lossless parameter mapping [43]:

$$
c _ { j } = \frac { 1 } { 2 } W _ { U , j } \in \mathbb { R } ^ { m } , \quad \nu _ { j } = b _ { U , j } + \frac { 1 } { 4 } \| W _ { U , j } \| _ { 2 } ^ { 2 } \in \mathbb { R } ,\tag{9}
$$

which yields $D _ { j } ( h ) = \| h \| _ { 2 } ^ { 2 } - z _ { j } ( h )$ , where $\| h \| _ { 2 } ^ { 2 }$ is invariant across candidate tokens.

Moreover, recent geometric studies [2] establish that each token’s cell $\mathrm { C e l l } _ { j } ^ { \mathrm { L V D } }$ is predominantly governed by a sparse subset of latent features $S _ { j } \subset \{ f _ { 1 } , f _ { 2 } , \dotsc \} ( | S _ { j } | \ll k )$ , such that its unembedding direction can be effectively decomposed as $\begin{array} { r } { W _ { U , j } \approx \sum _ { i \in \mathcal { S } _ { i } } \omega _ { j , i } \bar { f } _ { i } } \end{array}$ with projection weights $\omega _ { j , i } > 0$ Consequently, each token’s decision region is steered by a localized, distinct cluster of features.

## 3 Mechanistic Explanation of PCD

In this section, we formulate a mechanistic theory explaining how SFT induces Pervasive Capability Degradation (PCD). We show that domain narrowing during SFT shifts the dominant reconstruction loss from lower-order to higher-order terms, triggering geometric phase transitions that reduce the effective number of feature directions. While this reduction suppresses compounding interference on the target task, it induces directional interference surges on general data, eroding LVD cell margins and causing discrete prediction errors.

## 3.1 Transition from Lower-Order to Higher-Order Loss Dominance

Under Definition 2, the model encodes k latent semantic features within an m-dimensional residual stream $( k \gg m )$ . The overall expected feature reconstruction loss $\mathcal { L }$ expands as a binomial mixture across feature co-activation cardinalities [15]:

$$
{ \mathcal { L } } = \sum _ { p = 0 } ^ { k } { \binom { k } { p } } ( 1 - S ) ^ { p } S ^ { k - p } L _ { p } = S ^ { k } L _ { 0 } + k ( 1 - S ) S ^ { k - 1 } L _ { 1 } + { \binom { k } { 2 } } ( 1 - S ) ^ { 2 } S ^ { k - 2 } L _ { 2 } + . . . ,\tag{10}
$$

where $S \in [ 0 , 1 )$ denotes feature sparsity, and $L _ { p }$ represents the conditional reconstruction loss when exactly p latent features concurrently activate (a p-sparse activation state). In the pre-training corpus $\mathcal { D } _ { G }$ , broad semantic diversity yields high feature sparsity $( S  1 )$ , making marginal activation probabilities vanishingly small $( 1 - S \ll 1 )$ . Because $( \bar { 1 } - S ) \gg ( 1 - S ) ^ { \bar { 2 } } \gg \cdots \gg ( 1 - S ) ^ { k }$ higher-order co-activations are negligible. As $L _ { 0 }$ is a feature-independent bias penalty, optimization on $\mathcal { D } _ { G }$ is strictly governed by the lower-order loss $L _ { 1 }$ (the 1-sparse conditional loss) [15]:

$$
L _ { 1 } \approx \underbrace { \sum _ { i = 1 } ^ { k } I _ { i } \left( 1 - \mathrm { R e L U } ( \Vert f _ { i } \Vert _ { 2 } ^ { 2 } ) \right) ^ { 2 } } _ { \mathrm { F e a t u r e ~ B e n e f t } } + \underbrace { \sum _ { i \neq j } I _ { j } \mathrm { R e L U } ( f _ { j } ^ { \top } f _ { i } ) ^ { 2 } } _ { \mathrm { I n t e r f e r e n c e ~ C o s t } } ,\tag{11}
$$

where $I _ { i }$ is the importance of feature i. Under $L _ { 1 }$ dominance, features activate largely in isolation. Balancing reconstruction accuracy against pairwise cross-talk, the model organizes the k feature vectors into an almost-orthogonal polyhedral packing in $\mathbb { R } ^ { m }$ where mutual overlaps satisfy $| f _ { i } ^ { \top } f _ { j } | \leq$ $\epsilon \ll 1$

Conversely, the SFT dataset $\mathcal { D } _ { T }$ is domain-specific, substantially increasing the activation frequency of task-relevant features $( 1 - S _ { T } \gg 1 - \bar { S _ { G } } )$ . Furthermore, a subset of N task-relevant features $\mathcal { S } = \left\{ f _ { 1 } , \dots , f _ { N } \right\} ( N \ge 2 )$ becomes strongly correlated: $P _ { \mathcal { D } _ { T } } ( a _ { 1 } > 0 , \ldots , a _ { N } > 0 )  1$ . When these correlated features co-activate under the higher-order loss $L _ { N }$ , assuming unit intensities $a _ { n } = 1$ and equal importances $I _ { n } = I$ for $n \in S$ , the conditional loss $L _ { N }$ evaluates to:

$$
L _ { N , S } = \sum _ { n = 1 } ^ { N } I \left( 1 - f _ { n } ^ { \top } h \right) ^ { 2 } = \sum _ { n = 1 } ^ { N } I \left( 1 - \| f _ { n } \| _ { 2 } ^ { 2 } - \sum _ { j \neq n } f _ { n } ^ { \top } f _ { j } \right) ^ { 2 } .\tag{12}
$$

Assuming normalized vectors $( \left. { f _ { n } } \right. _ { 2 } \approx 1 )$ , expanding this expression isolates the compounding interference cost:

$$
\mathcal { C } _ { \mathrm { i n t e r } } ^ { ( N ) } = I \sum _ { n = 1 } ^ { N } \left( \sum _ { j \neq n } f _ { n } ^ { \top } f _ { j } \right) ^ { 2 } = I \sum _ { n = 1 } ^ { N } \left[ \sum _ { j \neq n } ( f _ { n } ^ { \top } f _ { j } ) ^ { 2 } + \sum _ { j \neq n } \sum _ { l \neq n , j } ( f _ { n } ^ { \top } f _ { j } ) ( f _ { n } ^ { \top } f _ { l } ) \right] .\tag{13}
$$

Equation (13) shows that compounding interference scales quadratically as $\mathcal { O } ( N ^ { 2 } )$ with the number of bound features. To minimize this cost, the representation undergoes a sharp geometric phase transition [15]: the model abandons symmetric polytopes that maintain distinct directions, snapping into configurations that represent fewer total feature directions.

Example: From $\mathcal { L } _ { 1 }$ to $\mathcal { L } _ { 2 }$ . To ground this transition, consider two features $f _ { a }$ and $f _ { b }$ that shift from independent activations under $\mathcal { L } _ { 1 }$ to deterministic co-occurrence under $\mathcal { L } _ { 2 } ( a _ { a } = 1 , a _ { b } = 1 )$ embedded among the remaining features $f _ { c } ( c \notin \{ a , b \} )$ ). Under the normalization of Eq. 13, the loss involving this pair is:

$$
\mathcal { L } _ { 2 , \{ a , b \} } = \underbrace { I ( 1 - \| f _ { a } \| _ { 2 } ^ { 2 } ) ^ { 2 } + I ( 1 - \| f _ { b } \| _ { 2 } ^ { 2 } ) ^ { 2 } } _ { \mathrm { F e a u r e ~ B e n e f i t } } + \underbrace { 2 I ( f _ { a } ^ { \top } f _ { b } ) ^ { 2 } } _ { \textit { c e l a b j } } + \underbrace { \sum _ { c \notin \{ a , b \} } \lambda _ { c } \Big [ ( f _ { c } ^ { \top } f _ { a } ) ^ { 2 } + ( f _ { c } ^ { \top } f _ { b } ) ^ { 2 } \Big ] } _ { \textit { c e l a b j } } ,\tag{14}
$$

where the first interference term is the compounding cross-talk within the pair, and the second collects the cross-talk with all other features, weighted by $\bar { \lambda _ { c } } = I _ { c } ( 1 - S _ { c } )$ . Preserving two distinct directions leaves two ways to remove the compounding cross-talk $2 I ( f _ { a } ^ { \top } f _ { b } ) ^ { 2 }$ . The first is orthogonalization, which keeps both features and drives $f _ { a } ^ { \top } f _ { b }  0$ . The second is collapse, which merges the pair into a single feature along their principal component:

$$
f _ { \mathrm { P C A } } = \frac { f _ { a } + f _ { b } } { \Vert f _ { a } + f _ { b } \Vert _ { 2 } } \approx \frac { f _ { a } + f _ { b } } { \sqrt { 2 } } .\tag{15}
$$

Collapse does not make the two directions orthogonal. It removes the pair itself: once $\{ f _ { a } , f _ { b } \}$ is replaced by the single feature $f _ { \mathrm { P C A } }$ , there is no second feature left to interfere with, so the compounding term disappears. Since $P _ { D _ { T } } ( a _ { a } \ne a _ { b } ) \approx 0$ $f _ { \mathrm { P C A } }$ carries the joint signal and the feature benefit is preserved almost losslessly.

Both routes differ in the cross-talk with the remaining features. Let $u = f _ { \mathrm { P C A } }$ and let v be the unit vector orthogonal to u in span $( f _ { a } , f _ { b } )$ . Orthogonalization within this plane keeps both directions occupied, whereas collapse retains only u:

$$
\sum _ { c \not \in \{ a , b \} } \lambda _ { c } \Big [ ( f _ { c } ^ { \top } u ) ^ { 2 } + ( f _ { c } ^ { \top } v ) ^ { 2 } \Big ] \geq \sum _ { c \not \in \{ a , b \} } \lambda _ { c } ( f _ { c } ^ { \top } u ) ^ { 2 } .\tag{16}
$$

Collapse therefore lowers the cross-talk with the other features by $\textstyle \sum _ { c } \lambda _ { c } ( f _ { c } ^ { \top } v ) ^ { 2 }$ and releases the direction v for them. When capacity is insufficient, the model therefore favors collapse over orthogonalization. This agrees with the observation [15] that correlated features form local orthogonal bases only when capacity permits, and otherwise collapse onto their principal components.

Although feature reduction is optimal on $\mathcal { D } _ { T }$ , it impairs pervasive capabilities on $\mathcal { D } _ { G } . ^ { 3 }$ Having reduced its feature directions, the model can no longer isolate individual features. Specifically, consider a pervasive input that triggers an isolated feature $f _ { i } ~ \in ~ S$ with unit intensity, yielding the true activation $h _ { \mathrm { t r u e } } = f _ { i }$ . Because the representation has collapsed, this input is read by and reconstructed from the shared principal component $f _ { \mathrm { P C A } }$ , producing a distorted reconstruction state $\begin{array} { r } { h _ { \mathrm { r e c o n } } = \big ( f _ { \mathrm { P C A } } ^ { \top } f _ { i } \big ) f _ { \mathrm { P C A } } = \frac { 1 } { | \cal { S } | } \dot { \sum } _ { j \in \cal { S } } \dot { f } _ { j } } \end{array}$

Algebraically, $h _ { \mathrm { r e c o n } }$ across dormant channels distorts the effective feature Gram matrix $G = F ^ { \top } F \in$ $\mathbb { R } ^ { k \times k }$ . While pre-training maintains near-orthogonality $( G _ { i j } \approx 0$ for $i \neq j )$ , the post-SFT phase transition elevates off-diagonal terms to positive values, inducing an unprovoked ghost signal spike on dormant channels:

$$
\Delta a _ { j } = f _ { j } ^ { \top } h _ { \mathrm { r e c o n } } \gg 0 , \quad \forall j \in \mathcal { S } \setminus \{ i \} .\tag{17}
$$

This spurious excitation introduces a non-zero representation error vector $\delta r _ { \ell } \in \mathbb { R } ^ { m }$ at layer ℓ:

$$
\delta r _ { \ell } = h _ { \mathrm { r e c o n } } - h _ { \mathrm { t r u e } } = \sum _ { j \in \mathcal { S } } \Delta a _ { j } f _ { j } \neq \mathbf { 0 } .\tag{18}
$$

Propagating through subsequent Transformer layers, $\delta r _ { l }$ accumulates into an activation displacement δh at the unembedding layer, approximated via first-order multivariate Taylor expansion:

$$
\delta h \approx J _ { l  L } ( x ) \cdot \delta r _ { l } ,\tag{19}
$$

where $\begin{array} { r } { J _ { l  L } ( x ) \ = \ \frac { \partial h _ { L } } { \partial r _ { l } } \ \in \ \mathbb { R } ^ { m \times m } } \end{array}$ is the inter-layer propagation Jacobian. Crucially, because the spurious excitation vector $\begin{array} { r } { \delta r _ { l } = \sum _ { j \in \mathcal { S } } \Delta a _ { j } f _ { j } } \end{array}$ is strictly confined to the task-correlated feature subspace span(S), the resulting displacement $\delta h$ is strongly anisotropic rather than a diffuse perturbation across all dimensions. Recalling from Section 2.3 that each token’s LVD cell is governed by a sparse set of dominant features, there inevitably exist competing tokens $k \in \mathcal V$ whose salient features $\boldsymbol { S _ { k } }$ overlap or strongly align with $\mathcal { S } \left( \mathrm { i . e . } \right.$ $( W _ { U , k } - W _ { U , y } ) ^ { \top } \delta h \gg \mathrm { ~ \overline { { 0 } } ) }$ . Consequently, δh acts as a directed semantic attractor, pulling $h _ { \mathrm { n e w } } = h _ { 0 } + \delta h$ preferentially toward the bounding hyperplanes of these S-coupled tokens.

![](images/f4b2df51352cc93fd16c6d01e12462a691f4b9abd1cf58b415efaffd9ea70082.jpg)  
Figure 2: Geometric illustration of PCD in the output LVD (Eqs. (18)–(20)).

Focusing on such a competitor token k, the classification margin relative to the ground-truth token y updates to:

$$
\begin{array} { r } { { \boldsymbol { M } } _ { y , k } ^ { ( \mathrm { n e w } ) } = { \boldsymbol { M } } _ { y , k } ^ { ( 0 ) } + ( W _ { U , y } - W _ { U , k } ) ^ { \top } { \boldsymbol { \delta h } } = { \boldsymbol { M } } _ { y , k } ^ { ( 0 ) } - ( W _ { U , k } - W _ { U , y } ) ^ { \top } { \boldsymbol { \delta h } } , } \end{array}\tag{20}
$$

where $M _ { y , k } ^ { ( 0 ) } = ( W _ { U , y } - W _ { U , k } ) ^ { \top } h _ { 0 } + ( b _ { U , y } - b _ { U , k } ) > 0$ is the baseline safe margin under the pre-trained activation $h _ { 0 }$ . When the directional push along the outward normal $( W _ { U , k } - W _ { U , y } )$ overcomes the baseline margin:

$$
( W _ { U , k } - W _ { U , y } ) ^ { \top } \delta h \ge M _ { y , k } ^ { ( 0 ) } \iff M _ { y , k } ^ { ( \mathrm { n e w } ) } \le 0 ,\tag{21}
$$

the activation vector breaches the bounding hyperplane. Consequently, $h _ { \mathrm { n e w } }$ exits $\mathrm { C e l l } _ { y } ^ { \mathrm { L V D } }$ and falls into $\mathrm { C e l l } _ { k } ^ { \mathrm { L V D } }$ , as shown in Figure 2, inverting the discrete prediction $( \hat { y } = k \neq y )$ and triggering PCD.

## 3.2 Unifying Existing Paradigms via Our Mechanistic Explanation

To contextualize PCD within broader post-training research, we systematically investigate existing methodologies across diverse subfields dedicated to mitigating capability degradation not only in the pure SFT scenarios, including continual learning [81, 89], parameter-efficient fine-tuning (PEFT) [74, 67], model merging [34, 79, 78], instruction data curation [83, 66], model evaluation [76, 35], machine unlearning [73, 36], preference alignment [70, 12], and reinforcement learning (RL) [26, 93]. We distill the foundational theoretical approaches developed across these disciplines into six core paradigms, which intervene across distinct operational dimensions: output probability distributions, training data mixtures, weight parameter geometry, parameter update pruning, loss gradients, and localized sub-networks. In the following subsubsections, we systematically unify these paradigms through our theoretical framework—integrating feature superposition, and LVD geometry—to elucidate the exact geometric mechanisms governing their efficacy.

## 3.2.1 Paradigm 1: Reference Distribution Anchoring (RDA)

RDA realized via KL divergence regularization $D _ { \mathrm { K L } } \left( \pi _ { \boldsymbol { \theta } } ( \boldsymbol { y } \mid \boldsymbol { x } ) \parallel \pi _ { \mathrm { r e f } } ( \boldsymbol { y } \mid \boldsymbol { x } ) \right)$ , mitigates PCD by penalizing the distributional deviation of the adapted model $\pi _ { \theta }$ from the pre-trained reference policy $\pi _ { \mathrm { r e f } } .$ , thereby constraining output token shifts to preserve pre-existing capabilities. This regularization mechanism is predominantly employed in RL alignment frameworks [26, 93]. Specifically, the constrained policy optimization objective is formulated as:

$$
\operatorname* { m a x } _ { \pi _ { \theta } } \mathbb { E } _ { x \sim \mathcal { D } , y \sim \pi _ { \theta } } [ r ( x , y ) ] - \beta D _ { \mathrm { K L } } \left( \pi _ { \theta } ( y \mid x ) \parallel \pi _ { \mathrm { r e f } } ( y \mid x ) \right) ,\tag{22}
$$

where $r ( x , y ) \in \mathbb { R }$ denotes the scalar reward function, and $\beta > 0$ governs the penalty strength against reference distribution drift.

Let $z = W _ { t J } ^ { \top } h + b _ { U } \in \mathbb { R } ^ { V }$ and $z _ { 0 } = W _ { U } ^ { \top } h _ { 0 } + b _ { U }$ denote the post-RDA and reference logits, yielding logit shift $\Delta z = z - z _ { 0 } = W _ { U } ^ { \top } \Delta h$ . A second-order Taylor expansion of $D _ { \mathrm { K L } } ( \pi _ { \theta } \parallel \pi _ { \mathrm { r e f } } )$ around $\Delta \bar { z } = 0$ reads:

$$
D _ { \mathrm { K L } } ( \pi _ { \theta } \parallel \pi _ { \mathrm { r e f } } ) \approx \frac { 1 } { 2 } \Delta z ^ { \top } F \Delta z = \frac { 1 } { 2 } \Delta h ^ { \top } \left( W _ { U } F W _ { U } ^ { \top } \right) \Delta h = \frac { 1 } { 2 } \Delta h ^ { \top } H _ { \mathrm { F i s h e r } } \Delta h ,\tag{23}
$$

where $\begin{array} { r } { \boldsymbol { F } = \mathrm { d i a g } ( \pi _ { \mathrm { r e f } } ) - \pi _ { \mathrm { r e f } } \pi _ { \mathrm { r e f } } ^ { \top } \in \mathbb { R } ^ { V \times V } } \end{array}$ is the Fisher information matrix of the reference categorical distribution, and $H _ { \mathrm { F i s h e r } } = W _ { U } F W _ { U } ^ { \top } \in \mathbb { R } ^ { m \times m }$ denotes the induced Fisher metric tensor on the residual stream.

For the reference top-1 token $j ^ { * } = \arg \operatorname* { m a x } _ { j \in \mathcal { V } } z _ { 0 , j }$ , its baseline safety margin relative to competitor $\begin{array} { r } { k \neq j ^ { * } \mathrm { ~ i s ~ } d _ { n , k } ^ { ( 0 ) } = \frac { ( W _ { U , j ^ { * } } - W _ { U , k } ) ^ { \top } h _ { 0 } + ( b _ { U , j ^ { * } } - b _ { U , k } ) } { \| W _ { U , j ^ { * } } - W _ { U , k } \| _ { 2 } } > 0 } \end{array}$ . Following activation shift $\Delta h$ , the updated margin is $\begin{array} { r } { d _ { n , k } ^ { \mathrm { ( n e w ) } } = d _ { n , k } ^ { \mathrm { ( 0 ) } } - \frac { ( W _ { U , k } - W _ { U , j ^ { * } } ) ^ { \top } \Delta h } { \| W _ { U , k } - W _ { U , j ^ { * } } \| _ { 2 } } } \end{array}$ . Applying the Cauchy-Schwarz inequality under the $H _ { \mathrm { F i s h e r } }$ metric yields an analytical lower bound on the residual margin (proof in Appendix B):

$$
d _ { n , k } ^ { ( \mathrm { n e w } ) } \geq d _ { n , k } ^ { ( 0 ) } - C _ { k } \sqrt { 2 D _ { \mathrm { K L } } ( \pi _ { \theta } \parallel \pi _ { \mathrm { r e f } } ) } , \quad \mathrm { w h e r e } \quad C _ { k } = \frac { \parallel W _ { U , k } - W _ { U , j ^ { * } } \parallel _ { H _ { \mathrm { r e b e r } } ^ { \dagger } } } { \parallel W _ { U , k } - W _ { U , j ^ { * } } \parallel _ { 2 } } ,\tag{24}
$$

and $H _ { \mathrm { F i s h e r } } ^ { \dagger }$ is the Moore-Penrose pseudo-inverse. Consequently, maintaining divergence below the threshold

$$
D _ { \mathrm { K L } } ( \pi _ { \boldsymbol { \theta } } \| \pi _ { \mathrm { r e f } } ) < \operatorname* { m i n } _ { k \neq j ^ { * } } \frac { ( d _ { n , k } ^ { ( 0 ) } ) ^ { 2 } } { 2 C _ { k } ^ { 2 } }\tag{25}
$$

strictly guarantees $d _ { n , k } ^ { \mathrm { ( n e w ) } } > 0$ across all competing hyperplanes, preventing h from exiting $\mathrm { C e l l } _ { j ^ { * } } ^ { \mathrm { L V D } }$ and precluding prediction flips on general tasks.

Established alignment algorithms enforce Eq. (25) through distinct mechanisms: PPO [57] combines token-level KL penalties with surrogate clipping $\mathrm { c l i p } ( \overline { { r } } _ { t } ( \theta ) , 1 - \epsilon , 1 + \epsilon )$ on probability ratios to restrict update step sizes. DPO [51] analytically parameterizes reward via $\begin{array} { r } { r ( x , y ) = \beta \log \frac { \pi _ { \theta } ( y | x ) } { \pi _ { \mathrm { r e f } } ( y | x ) } + } \end{array}$ $\beta$ log $Z ( x )$ , enforcing the Fisher trust-region implicitly via pairwise. GRPO [60] foregoes critic estimation and integrates an unbiased token-level KL penalty $\begin{array} { r } { \frac { \pi _ { \mathrm { r e f } } } { \pi _ { \theta } } - \log \frac { \pi _ { \mathrm { r e f } } } { \pi _ { \theta } } - 1 \geq 0 } \end{array}$ directly into group-relative advantage updates.

However, with proofs deferred to Appendix C, RDA faces a gradient deadlock when applied to SFT. Unlike on-policy RL, where policy $\pi _ { \theta }$ samples candidates strictly within the support set of the reference policy $\pi _ { \mathrm { r e f } }$ and performs soft probability re-weighting, SFT enforces out-of-distribution distribution jumps via the hard one-hot cross-entropy objective $\mathcal { L } _ { \mathrm { S F T } } ( \theta ) = - \log \pi _ { \theta } ( y ^ { * } \mid h )$ . SFT with the RDA penalty yields the joint objective $\mathcal { L } _ { \mathrm { t o t a l } } = \dot { \mathcal { L } } _ { \mathrm { S F T } } + \beta D _ { \mathrm { K L } } ( \pi _ { \theta } \| \pi _ { \mathrm { r e f } } )$ . Differentiating w.r.t. output logits $z \in \dot { \mathbb { R } } ^ { V }$ and setting $\nabla _ { z } { \mathcal { L } } _ { \mathrm { t o t a l } } = \mathbf { 0 }$ approximately yields the theoretical stationary distribution:

$$
\pi _ { \theta } ^ { * } = \frac { 1 } { 1 + \beta } \mathbf { e } _ { y ^ { * } } + \frac { \beta } { 1 + \beta } \pi _ { \mathrm { r e f } }\tag{26}
$$

where $\mathbf { e } _ { y ^ { \ast } }$ denotes the one-hot target indicator. Any non-zero regularization coefficient $\beta > 0$ algebraically dilutes the gradient thrust toward $\mathbf { e } _ { y } .$ ∗ in exact proportion to ${ \frac { \beta } { 1 + \beta } } .$

## 3.2.2 Paradigm 2: Data Mixture Alignment (DMA)

DMA mitigates PCD by assembling a diverse pervasive dataset $\begin{array} { r } { \mathcal { D } _ { \operatorname* { m i x } } = \sum _ { k = 1 } ^ { K } w _ { k } \mathcal { D } _ { k } } \end{array}$ across heterogeneous tasks, balancing target task adaptation with general capability preservation. This data-centric paradigm is widely adopted across instruction data curation [83, 66], multi-task fine-tuning [38, 44], and comprehensive model evaluation benchmarks [76, 35].

Through our theoretical framework (Section 3.1), DMA operates as an empirical proxy to simulate the diversity of the pre-training distribution $\mathcal { D } _ { G }$ at the input stage. In single-task SFT on narrow data $\mathcal { D } _ { T }$ , the activation frequency of task-specific features surges $( 1 - \check { S _ { T } } \gg 1 - S _ { G } )$ , shifting the reconstruction loss from $L _ { 1 }$ to higher-order losses $L _ { N } \left( N \ge 2 \right)$ . By aggregating diverse tasks, DMA enforce high global feature sparsity $( S  1 )$ and suppress joint co-activation probabilities $( P ( a _ { i } > 0 , a _ { j } > 0 ) \approx 0 )$ , which retains the dominance of the lower-order loss $L _ { 1 }$

Key methodologies within this paradigm implement this geometric stabilization through different operational mechanisms: LIMA [90] caps the empirical trigger count of any domain-specific feature, keeping absolute feature density $1 - S$ near zero, ensuring the model remains strictly within the 1-sparse regime of $L _ { 1 }$ . SMART [54] maximizes the convex hull volume spanned by the active feature sub-dictionary while removing sample-level collinearity, directly suppressing the emergence of the compounding cross-talk terms $\mathcal { C } _ { \mathrm { i n t e r } } ^ { ( N ) }$ at the data source. DYNAMIXSFT [61]: using a 1-step lookahead $\Delta \cdot$ -Loss reward, tracks the immediate marginal return along task-specific feature directions, to prevent excessive directional drift $\delta h$ along any single task axis. Data Mixing Optimization (DMO) [35]: proves that optimal allocations satisfy first-order derivative equality: $\begin{array} { r } { \frac { \partial \check { L _ { 1 } } } { \partial w _ { 1 } } \doteq \cdot \cdot \cdot = \frac { \partial L _ { K } } { \partial w _ { K } } } \end{array}$ Macroscopically, this condition balances the gradient thrust across different feature directions on the loss surface, preventing any single task from exerting a disproportionate displacement that pierces delicate LVD decision hyperplanes.

However, because the sample budget of SFT is orders of magnitude smaller than pre-training $( N _ { \mathrm { m i x } } \ll N _ { \mathrm { p r e } } ) .$ , the empirical distribution naturally contains vast feature support voids. Given extreme pre-training feature sparsity $( S _ { i } \to 1 )$ , long-tail features with marginal activation probabilities $\begin{array} { r } { p _ { i } = 1 - \dot { S } _ { i } < \frac { 1 } { N _ { \mathrm { m i x } } } } \end{array}$ is completely unactivated $( P _ { \mathrm { m i x } } \mathbf { \bar { ( } } a _ { i } > 0 ) = 0 )$ and entirely unsupervised. Ensuring that $\mathcal { D } _ { \mathrm { m i x } }$ covers all k latent semantic features with confidence at least $1 - \delta$ requires an asymptotic sample complexity of:

$$
N _ { \mathrm { m i x } } = \Omega \left( \frac { k \log ( k / \delta ) } { \operatorname* { m i n } _ { i } ( 1 - S _ { i } ) } \right)\tag{27}
$$

Because overcomplete features vastly exceed hidden dimensions $( k \gg m )$ , achieving zero-collapse coverage requires an empirical sample volume scaling to $N _ { \mathrm { m i x } } \geq \mathcal { O } ( N _ { \mathrm { p r e } } )$

## 3.2.3 Paradigm 3: Parameter-Space Orthogonalization (PSO)

PSO posits that models fine-tuned on distinct tasks $( A , B , \ldots )$ from a shared base model $\mathcal { M } _ { \theta ^ { 0 } }$ can be directly unified in weight space, serving as one of the core mechanisms in model merging [34, 79, 78]. Its premise is that inter-task conflicts stem predominantly from adversarial parameter updates and sign disagreements among task vectors $( \Delta \theta _ { A } ^ { \dagger } \Delta \theta _ { B } < 0 )$ . Consequently, PSO fuses task updates along mutually orthogonal parameter directions.

Let $W _ { a } , W _ { b }$ denote the parameter representations of features a and b. Under Hypothesis 1, constraining parameter updates to be orthogonal eliminates this cost:

$$
W _ { a } \cdot W _ { b } = 0 \implies { \mathcal { C } } _ { \mathrm { i n t e r } } \equiv 0 ,\tag{28}
$$

severing the pathway through which feature b leaks into the readout channel of feature a.

In practice, several works implement PSO via singular value decomposition (SVD) on layer-wise update matrices:

$$
\Delta { W _ { A } } = { U _ { A } } { \Sigma _ { A } } { V _ { A } ^ { \top } } , \quad \Delta { W _ { B } } = { U _ { B } } { \Sigma _ { B } } { V _ { B } ^ { \top } } ,\tag{29}
$$

where $U _ { A } , U _ { B } \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times r }$ and $V _ { A } , V _ { B } \in \mathbb { R } ^ { d _ { \mathrm { i n } } \times r }$ denote the left and right singular matrices, respectively. Decomposing task feature channels into rank-1 singular components $W _ { a } = \sigma _ { a } u _ { a } v _ { a } ^ { \top }$ and $W _ { b } = \sigma _ { b } u _ { b } v _ { b } ^ { \top }$ , their inner product expands as $W _ { a } \cdot W _ { b } = \sigma _ { a } \sigma _ { b } ( u _ { a } ^ { \top } u _ { b } ) ( v _ { a } ^ { \top } v _ { b } )$ , reformulating $\mathcal { C } _ { \mathrm { i n t e r } }$ as $2 I \left[ \sigma _ { a } \sigma _ { b } ( \boldsymbol { u } _ { a } ^ { \top } \boldsymbol { u } _ { b } ) ( \boldsymbol { v } _ { a } ^ { \top } \boldsymbol { v } _ { b } ) \right] ^ { 2 }$ . Building on this formulation, KnOTS [62] and TSV-Merge [19] enforce orthogonality on the concatenated singular bases $( U _ { A } ^ { \top } U _ { B } = \mathbf { 0 } \mathrm { o r } V _ { A } ^ { \top } V _ { B } = \mathbf { 0 } )$ via shared orthonormal projections or Procrustes whitening, driving cross-task singular overlaps and the resulting interference to zero. In addition, PCB-Merging [14] approximates feature importance $I _ { i }$ via intra-task parameter energy $( \tau _ { i } \odot \tau _ { i } )$ to adaptively dampen and eliminate parameter interference.

Although PSO enforces parameter-level decoupling, the translation from parameter orthogonality to activation-space feature orthogonality suffers substantial attenuation. With orthogonal task matrices $\Delta W _ { A }$ and $\bar { \Delta } W _ { B }$ , residual signals must pass through non-linear activation functions $\sigma ( \cdot ) \left( \mathrm { e . g . } \right.$ , GELU or SwiGLU). A second-order multivariate Taylor expansion of the combined post-activation state $h _ { \mathrm { o u t } } = \sigma ( z _ { 0 } + \delta z _ { A } + \delta z _ { B } )$ isolates a non-vanishing interaction term:

$$
h _ { \mathsf { o u t } } \approx \sigma ( z _ { 0 } ) + \sigma ^ { \prime } ( z _ { 0 } ) ( \delta z _ { A } + \delta z _ { B } ) + \frac { 1 } { 2 } \sigma ^ { \prime \prime } ( z _ { 0 } ) ( \delta z _ { A } ^ { 2 } + \delta z _ { B } ^ { 2 } ) + \sigma ^ { \prime \prime } ( z _ { 0 } ) ( \delta z _ { A } \odot \delta z _ { B } )\tag{30}
$$

Even when parameter updates achieve strict singular-basis orthogonality such that the linear crossactivation expectation vanishes $( \mathbb { E } [ \delta z _ { A } ^ { \top } \delta z _ { B } ] = 0 )$ , the non-linear crosstalk component $\sigma ^ { \prime \prime } ( z _ { 0 } ) ( \delta z _ { A } \odot$ $\delta z _ { B } )$ remains strictly positive. This non-linear modulation forcibly generates spurious higher-order cross-task hybrid features in the residual stream.

## 3.2.4 Paradigm 4: Delta-Parameter Pruning (DPP)

Delta-Parameter Pruning (DPP) is a post-SFT technique that mitigates task conflicts and capability degradation by pruning the majority of the updated parameters $\Delta \theta \overset { \cdot } { = } \theta - \theta ^ { 0 } [ 8 5 , 1 1 ]$ , widely adopted across model merging [34, 79, 78] and capability preservation [17, 42, 1]. To examine its mechanistic foundation, consider the analytical gradient of task loss $\mathcal { L } _ { T }$ with respect to an arbitrary layer read-in weight matrix W during single-task SFT. Let $S _ { T } \subset \{ f _ { 1 } , \ldots , f _ { k } \}$ denote the subset of latent semantic features directly relevant to task T, with cardinality $k _ { T } = | \bar { S } _ { T } | \ll k$ . According to the feature interference formulation in Eq. (4), the analytical gradient decomposes into two distinct components:

$$
\nabla _ { W } \mathcal { L } _ { \mathcal { T } } = \sum _ { f _ { a } \in \mathcal { S } _ { \mathcal { T } } } \underbrace { \frac { \partial \mathcal { L } _ { \mathcal { T } } } { \partial \hat { a } _ { a } } \frac { \partial \hat { a } _ { a } } { \partial W } } _ { \mathrm { F e a t u r e ~ B e n e f i t } } + \sum _ { f _ { a } \in \mathcal { S } _ { \mathcal { T } } } \sum _ { f _ { j } \notin \mathcal { S } _ { \mathcal { T } } } \underbrace { \frac { \partial \mathcal { L } _ { \mathcal { T } } } { \partial \hat { a } _ { a } } \frac { \partial ( f _ { a } ^ { \top } f _ { j } ) } { \partial W } f _ { j } } _ { \mathrm { I n t e r f e r e n c e ~ C o s t } } ,\tag{31}
$$

where $\hat { a } _ { a }$ denotes the recovered activation intensity of target feature $f _ { a } ,$ and $f _ { a } , f _ { j } \in \mathbb { R } ^ { m }$ are unit feature directions in the residual stream. Equation (31) reveals an extreme asymmetry in parameter degrees of freedom: the parameters directly establishing the target feature-token mapping possess only $\mathcal { O } ( k \tau )$ degrees of freedom. However, because activating target feature $f _ { a }$ excites crosstalk across non-target ambient features $f _ { j } \notin S _ { T }$ , gradient descent on the narrow distribution $\mathcal { D } _ { T }$ suppresses these residuals by rotating weights across almost all coupled directions, generating $\mathcal { O } ( k \tau ^ { . } k )$ cross-coupling updates. Because $k \gg k _ { T }$ , the ratio of parasitic interference parameters to target mapping parameters scales as $\begin{array} { r } { \frac { \mathcal { O } ( k _ { T } \cdot k ) } { \mathcal { O } ( k _ { T } ) } = \mathcal { O } ( k ) \gg 1 } \end{array}$ . Consequently, in unregularized SFT, the vast majority of delta parameters $\Delta \theta$ are devoted to canceling localized crosstalk on $\mathcal { D } _ { T }$ or drifting along unconstrained ambient dimensions, rather than constructing task-specific token representations.

Contemporary DPP methods eliminate these parasitic interference components through distinct strategies: DARE [85] and DAREx [11] randomly drop delta parameters with a high ratio $p \in$ [0.9, 0.99] and rescale remaining weights, eliminating the vast majority of parasitic interference cost in mathematical expectation. Model Tailor [91] selects an ultra-sparse patch based on saliency and second-order sensitivity to directly constrain the representation displacement on non-target features. DiFT [87] identifies task-critical Delta-Scale Rows (DSR) corresponding to task-exclusive high-order semantic circuits in superposition.

However, DPP is fundamentally bounded by an intrinsic trade-off: under extreme feature sparsity, the activation perturbation variance induced by pruning mask diverges inversely with the retention rate [11], scaling as:

$$
\mathrm { V a r } ( h ^ { \mathrm { d i f f } } ) \propto \frac { p } { 1 - p } \sum _ { j } ( \Delta W _ { i j } a _ { j } ) ^ { 2 } ,\tag{32}
$$

where $p \in [ 0 , 1 )$ denotes the parameter drop rate. Crucially, this trade-off exposes the fundamental theoretical ceiling of DPP: feature benefit and interference cost cannot be decoupled from the perspective of DDP. Under feature superposition, physical parameter channels couple simultaneously to target semantic features and ambient pervasive directions. Because parameter pruning acts as an unstratified coordinate truncation rather than a geometric disentanglement of feature manifolds, it inevitably fails to extinguish pervasive interference without compromising representation stability.

## 3.2.5 Paradigm 5: Gradient Orthogonalization Optimization (GOO)

GOO posits that preserving pervasive capabilities fundamentally requires directional gradient compatibility: if the gradient of task $A \mathbf { g } _ { A } = \nabla _ { \theta } \mathcal { L } _ { A }$ is non-orthogonal to the task-B gradient $\mathbf { g } _ { B } = \nabla _ { \theta } \mathcal { L } _ { B }$ $( \mathbf { g } _ { A } ^ { \top } \dot { \mathbf { g } } _ { B } \neq 0 )$ , parameter updates ∆W will inevitably be driven along trajectories that escalate feature interference between task A and B [41, 3, 86]. To prevent this, GOO constrains or projects parameter updates onto subspaces orthogonal to another task’s gradients, a mechanism widely adopted across continual learning [81, 89], multi-task learning [38, 44], and PEFT [74, 67].

To uncover the mechanistic link between gradient conflict and feature interference, we examine the gradient vector field governing the task $\bar { A }$ on $\mathcal { D } _ { A }$ . For simplicity, differentiating the loss $\mathcal { L } _ { A }$ with respect to an arbitrary latent feature direction vector $f _ { j } \in \mathbb { R } ^ { \bar { m } }$ in the 1-sparse conditional loss yields two distinct vector fields:

$$
\mathbf { g } _ { A , j } = \nabla _ { f _ { j } } \mathcal { L } _ { A } = \underbrace { - 4 I _ { j } \left( 1 - \| f _ { j } \| _ { 2 } ^ { 2 } \right) f _ { j } } _ { \mathbf { g } _ { \mathrm { m a g } , j } \mathrm { ( M a g n i t u d e ~ R e s t o r i n g ~ F o r c e ) } } + \underbrace { 2 \sum _ { i \neq j } ( I _ { j } + I _ { i } ) ( f _ { j } ^ { \top } f _ { i } ) f _ { i } } _ { \mathrm { ( } i \neq j } ,\tag{33}
$$

where $\mathbf { g } _ { \mathrm { m a g } , j }$ penalizes deviations from unit norm $\| f _ { j } \| _ { 2 } \approx$ 1 to maintain feature benefit, and $\mathbf { g } _ { \mathrm { o r t h } , j }$ acts as an repulsive force on $\Delta W _ { j } ^ { B }$ to push $f _ { j }$ away from non-orthogonal features $f _ { i }$ whenever $f _ { j } ^ { \top } f _ { i } > 0 ^ { 4 }$ to minimize feature interference.

Let parameter updates on another task B be $\Delta W ^ { B } = - \eta \mathbf { g } _ { B }$ with learning rate $\eta > 0$ . When a negative gradient inner product occurs along channel $j ( \mathbf { g } _ { B , j } ^ { \top } \mathbf { g } _ { A , j } < 0 )$ , taking the inner product in parameter space yields $\langle \Delta W _ { j } ^ { B } , \mathbf { g } _ { A , j } \rangle \ > \ 0$ . Because the equilibrium in feature benefit term $( \lVert f _ { j } \rVert _ { 2 } \approx 1 ) , \mathbf { g } _ { \mathrm { m a g } , j } \approx \mathbf { 0 }$ ,then leaving $\mathbf { g } _ { A , j }$ dominated by $\mathbf { g } _ { \mathrm { o r t h } , j } ^ { A } .$ Consequently, $\langle \Delta W _ { j } ^ { B } , \mathbf { g } _ { \mathrm { o r t h } , j } ^ { A } \rangle > 0$ demonstrating that the update acts in direct opposition to the repulsive restorative force and drives channel j to rotate toward activating and coupling with feature $f _ { i }$ . When evaluating an input of task $A$ with activation $h _ { 0 }$ , this adverse channel rotation induces a physical layer perturbation $\delta r _ { \ell } = \Delta W ^ { B } h _ { 0 } \neq \mathbf { 0 }$ , which accumulates through subsequent layers to erode planar LVD decision margins and trigger discrete prediction flips.

Existing GOO algorithms enforce gradient compatibility through diverse projection operations. GEM [41] and A-GEM [3] solve quadratic programs to project candidate gradients onto half-spaces satisfying $\tilde { g } ^ { \top } g _ { k } \ge 0$ , preventing updates from pointing toward competitor LVD cells. OGD [16] applies Gram-Schmidt orthogonalization to accumulated task gradients to keep pre-trained feature readouts invariant. Addressing multi-task training, PCGrad [86] projects each conflicting gradient onto the normal plane of the other, eliminating magnitude shrinkage and lateral parameter drift. In activation space, GPM [55] computes the SVD to enforce zero-activation crosstalk $( \Delta W _ { \ell } h _ { 0 } \approx \mathbf { 0 } )$ . Expanding on this, FS-DGPM [10] and TRGP [37] relax strict null-space projection via trust regions and dynamic soft orthogonalization, accommodating finite hidden dimensions by allowing bounded overlap along minor singular axes.

However, GOO faces unavoidable dimensional exhaustion when tasked with safeguarding pervasive capabilities. The allowable degrees of freedom for gradient projection are fundamentally bottlenecked by the physical hidden dimension m of the residual stream. When SFT progresses sequentially across tasks or domains, accumulating core feature bases $B _ { t } \in \mathbb { R } ^ { m \times r _ { t } }$ yields a composite constraint matrix $\boldsymbol { \mathcal { B } _ { T } } = [ B _ { 1 } , \dots , B _ { T } ] \in \mathbb { R } ^ { m \times ( \sum _ { t = 1 } ^ { T } r _ { t } ) }$ . Once the cumulative rank of protected pervasive features saturates the hidden dimension $\begin{array} { r } { ( \sum _ { t = 1 } ^ { T } r _ { t } \geq m ) } \end{array}$ , the orthogonal complement projection operator collapses to zero:

$$
P _ { \perp } ^ { ( T ) } = I - B _ { T } ( B _ { T } ^ { \top } B _ { T } ) ^ { - 1 } B _ { T } ^ { \top } = \mathbf { 0 }\tag{34}
$$

Under this condition, the projected gradient vanishes identically $( \mathbf { g } _ { \mathrm { p r o j } } = P _ { \perp } ^ { ( T ) } \mathbf { g } _ { T } \equiv \mathbf { 0 } )$ , causing complete collapse.

## 3.2.6 Paradigm 6: Mechanistic Interpretability and Localization (MIL)

MIL posits that specific linguistic behaviors, factual knowledge, and reasoning skills are mediated by localized functional sub-networks, attention circuits, or dedicated subsets of neurons rather than uniformly dispersed representations. By isolating task-critical parameter regions, this paradigm restricts parameter updates or edits strictly to identified components while freezing the remaining parameters. This targeted adaptation strategy is widely employed in knowledge editing [47, 9], machine unlearning [73, 36], and safety alignment [77] to modify the mechanism for the target task without destabilizing foundational capabilities.

Within our geometric framework, the efficacy of partial freezing stems from eliminating unconstrained drift on ambient feature channels. In SFT, mechanistic localization partitions the comprehensive set of parameter channels $\mathcal { N }$ into two disjoint subsets: the task-critical channels $\mathcal { N } _ { \tau }$ identified via attribution or causal tracing, and the task-irrelevant ambient channels $\mathcal { N } _ { \mathcal { P } } = \mathcal { N } \backslash \mathcal { N } _ { \mathcal { T } }$ . When the adapted model processes a general pervasive input with baseline activation $h _ { 0 }$ , the net forward activation displacement ∆h decomposes into:

$$
\Delta h = \underbrace { \sum _ { i \in \mathcal { N } _ { \tau } } \Delta W _ { i } h _ { 0 } } _ { \mathrm { T a s k - C r i t i c a l ~ A d a p t a t i o n } } + \underbrace { \sum _ { j \in \mathcal { N } _ { \mathcal { P } } } \Delta W _ { j } h _ { 0 } } _ { \mathrm { I r r e l e v a n t { D r i f t } } } ,\tag{35}
$$

where $\Delta W _ { i }$ and $\Delta W _ { j }$ denote the parameter variations along individual channels. Parameter channels exhibit extreme functional selectivity: channels in $\mathcal { N } _ { \mathcal { T } }$ couple predominantly to target task features $\scriptstyle { \mathcal { S } } _ { T }$ , whereas channels in $\mathcal { N } _ { \mathcal { P } }$ are dedicated almost exclusively to general pervasive features $S _ { \mathrm { a m b i e n t } } =$ $\{ f _ { 1 } , \ldots , f _ { k } \} \setminus { \mathcal { S } } _ { T }$

Consequently, the Irrelevant Drift term in Eq. (35) acts as the primary destructive force during full $\mathrm { S F { \bar { T } } , }$ as unconstrained updates across $\mathcal { N } _ { \mathcal { P } }$ distort ambient feature representations and generate coherent ghost signal spikes. Mechanistic localization enforces the hard constraint $\Delta W _ { j } \equiv \mathbf { 0 }$ for all $j \in \bar { \mathcal { N } } _ { \mathcal { P } }$ , directly extinguishing this second summation and shielding general features from unintended parameter modifications.

Contemporary localization techniques implement this geometric shielding through diverse intervention mechanics. DEPN [73] and CLUE [7] trace sensitive attributes via integrated gradients or dual-circuit causal discovery, zeroing out targeted neuron activations during forward inference $( a _ { j }  0 , \forall j \in S _ { \mathrm { p r i v a c y } } )$ , thereby truncating the propagation of unconstrained drift vectors. CA-LoRA [68] detects inter-layer communication shifts via circuit edge rewiring rates, adaptively assigning low-rank adaptation budgets to restructured layers while freezing stable functional paths. Furthermore, gradient- and magnitude-based attribution frameworks such as WAGLE [25] evaluate parameter sensitivity w.r.t. target feature channels, restricting gradient updates exclusively to high-attribution weights to eliminate crosstalk excitation on dormant pervasive features.

However, MIL presumes that neural behaviors are compartmentalized into modular, cleanly separable parameter subsets, an assumption fundamentally contradicted by feature superposition under m $\ll k$ Superposition forces physical neurons and weight channels to be polysemantic, functioning as composite channels:

$$
W _ { j } = \sum _ { a \in \mathcal { N } _ { \mathcal { T } } } c _ { j , a } f _ { a } + \sum _ { l \in \mathcal { N } _ { \mathcal { P } } } c _ { j , l } f _ { l }\tag{36}
$$

where the total ambient coupling energy $\begin{array} { r } { \sum _ { l \in \mathcal { N } _ { T } } | c _ { j , l } | ^ { 2 } > 0 } \end{array}$ is strictly bounded away from zero. Consequently, no parameter subset Θ can be localized that isolates target capabilities without coupling to ambient general features. When adapting the localized parameters $( \Delta \theta _ { j } \neq 0 , \forall j \in \Theta _ { T } )$ to master task T, these polysemantic weights undergo unavoidable lateral rotations, inducing a strictly positive lower bound on residual collateral leakage.

## 3.2.7 Summary

The six paradigms share a common origin. None of them is derived from the microscopic mechanism of PCD, namely the interference among superposed features. Instead, each starts from a macroscopic symptom of the conflict between target adaptation and pervasive capabilities, such as output distribution shift, data imbalance, parameter overlap, or gradient conflict. Each remedy is therefore effective, but its supporting theory holds only locally, under an assumption that superposition violates. Methods that operate in weight space (PSO, DPP and MIL) implicitly assume that features can be separated by separating parameters. Methods that operate on data, gradients, or outputs (DMA, GOO and RDA) instead overlook the two properties that make superposition possible: sparsity and almost-orthogonality.

These limitations point to the same missing ingredient: a constraint that acts directly on the carrier of PCD, i.e., the margins of activations against competing LVD facets. Section 4.1 develops AEGIS on this principle.

## 4 Method

## 4.1 Limitations of Cross-Entropy under Feature Interference

As established in Sections 2.3 and 3, Pervasive Capability Degradation (PCD) stems from pronounced interference among correlated features, manifesting as an anisotropic drift of activations toward certain tokens coupled with these interfering features. By the LVD formulation (Eq. (8)), each token’s decision region $\mathrm { C e l l } _ { j } ^ { \mathrm { L V D } }$ is bounded by affine hyperplanes separating it from competing cells $\mathrm { C e l l } _ { k } ^ { \mathrm { L V D } }$

$$
\begin{array} { r l } { \Pi _ { j , k } : } & { { } ( W _ { U , j } - W _ { U , k } ) ^ { \top } h + ( b _ { U , j } - b _ { U , k } ) = 0 . } \end{array}\tag{37}
$$

This geometry establishes an exact duality between activation drift and logit dynamics:

Corollary 1 (Geometric Duality between Activation Drift and Logit Dynamics). Let $\begin{array} { r l } { \hat { u } _ { j , k } } & { { } = } \end{array}$ $\frac { W _ { U , k } - W _ { U , j } } { \| W _ { U , k } - W _ { U , j } \| _ { 2 } }$ be the unit outward normal vector of $\Pi _ { j , k }$ . Inside Int $\mathrm { ( C e l l } _ { j } ^ { \mathrm { L V D } } )$ , token j strictly attains the maximal logit across V. Along the directional trajectory $h ( t ) = h + t { \hat { u } } _ { j , k } \left( t \geq 0 \right)$ , the relative margin $z _ { k } ( h ( t ) ) - z _ { j } ( h ( t ) )$ increases strictly monotonically at a constant rate:

$$
\frac { d } { d t } \big ( z _ { k } ( h ( t ) ) - z _ { j } ( h ( t ) ) \big ) = \| W _ { U , k } - W _ { U , j } \| _ { 2 } > 0 .\tag{38}
$$

At the boundary $\Pi _ { j , k } , \ z _ { j } ( h ( t ) ) \ = \ z _ { k } ( h ( t ) ) ,$ once crossing into Int $( \mathrm { C e l l } _ { k } ^ { \mathrm { L V D } } )$ , token k strictly dominates all candidates (proof in Appendix D).

For a ground-truth token $j ,$ backpropagating the categorical cross-entropy gradient $\begin{array} { r } { \frac { \partial \mathcal { L } _ { \mathrm { C E } } } { \partial z _ { k } } = p _ { k } - { \bf 1 } [ k = } \end{array}$ $j ]$ through $W _ { U }$ exerts an optimization thrust on activation h:

$$
\Delta h \propto - \nabla _ { h } \mathcal { L } _ { \mathrm { C E } } = ( 1 - p _ { j } ) W _ { U , j } - \sum _ { k \neq j } p _ { k } W _ { U , k } ,\tag{39}
$$

where $p _ { j }$ and $p _ { k }$ are the softmax probability of label token and competing tokens. In the ideal situation, mutually orthogonal unembedding vectors $( W _ { U , j } ^ { \top } W _ { U , k } = 0 )$ would pull h purely toward $\mathrm { C e l l } _ { j } ^ { \mathrm { L V D } }$ and away from other token cells. However, feature superposition $( k \gg m )$ inevitably introduces sparse interference $( W _ { U , k } ^ { \top } W _ { U , j } > 0 )$ . This projects a parasitic leakage $( 1 - p _ { j } ) ( W _ { U , k } ^ { \top } W _ { U , j } )$ onto competing channel $k ,$ , counteracting its gradient self-repulsion $p _ { k } \| W _ { U , k } \| _ { 2 } ^ { 2 }$ . At optimization equilibrium, balancing these forces yields the steady-state crosstalk probability (Proofs are shown in Appendix E):

$$
p _ { k } ^ { * } \propto ( 1 - p _ { j } ) \cdot \frac { W _ { U , k } ^ { \top } W _ { U , j } } { \| W _ { U , k } \| _ { 2 } ^ { 2 } } .\tag{40}
$$

During pre-training on $\mathcal { D } _ { G }$ , near-orthogonal packing under 1-sparse loss dominance $( | W _ { U , k } ^ { \top } W _ { U , j } | \le$ $\epsilon \ll 1 )$ keeps $p _ { k } ^ { * }$ negligible, maintaining saturating confidence $( p _ { j } \to 1 )$ well inside $\mathrm { C e l l } _ { j } ^ { \mathrm { L V D } }$

During SFT on $\mathcal { D } _ { T }$ , this balance collapses: task-specific feature density triggers a geometric phase transition (Section 3.1) that sharply elevates interference $( W _ { U , k } ^ { \top } W _ { U , j } \gg 0 )$ across task-coupled tokens. By Eq. (40), this inflates the competitor probability $p _ { k } ^ { * }$ on general queries, diluting true-label confidence and driving h outward toward $\Pi _ { j , k }$ . For infrequent or long-tail pervasive tokens with narrow margins, this unchecked drift easily breaches the decision boundary, inverting discrete predic tions and triggering PCD. While existing post-training paradigms attempt to curb such degradation, each faces fundamental theoretical limitations within this geometry (Section 3.2).

In summary, employing standard cross-entropy for post-training SFT suffers from two fundamental structural limitations under feature superposition:

1. In cross-entropy, the optimization of ground-truth and the feature interference are inextricably coupled, lacking any independent mechanism to decouple and penalize localized feature interference without perturbing task classification.

2. Because cross-entropy is predicated on one-hot categorical distributions, sustaining feature sparsity and almost-orthogonal packing relies on the collective supervision of massive, diverse samples, unattainable within the narrow sample budgets of $\mathrm { \dot { { S F T } } }$

## 4.2 The AEGIS Framework

To resolve these challenges, we introduce the AEGIS (Adaptive Equipartition of Geometric Interference in Superposition) loss. Instead of relying on discrete categorical probability distributions, AEGIS directly governs directional activation drifts across all token hyperplanes in the unembedding LVD, decoupling feature interference mitigation from task adaptation. Moreover, by supervising activation along multiple directional decision boundaries—including those corresponding to infrequent and long-tail tokens—AEGIS substantially reduces the sample budget required for the pervasiveness dataset. AEGIS consists of two steps:

Step 1: Adaptive Directional Width. Prior to SFT, the pre-trained base model $\mathcal { M } _ { \theta ^ { 0 } }$ is frozen. We employ a pervasive anchor dataset $\mathcal { D } _ { P }$ spanning diverse tasks to establish baseline geometric coordinates and directional tolerance bounds.

Let an input sample possess ground-truth label token $j .$ The base model generates final residual activation $\boldsymbol { h _ { 0 } } ~ \in ~ \mathbb { R } ^ { m }$ , with unembedding parameters $\mathbf { \bar { \boldsymbol { W } } } _ { U } ^ { ( 0 ) } = [ \boldsymbol { W } _ { U , 1 } ^ { ( 0 ) } , \ldots , \mathbf { \bar { \boldsymbol { W } } } _ { U , V } ^ { ( 0 ) } ] \in \mathbb { R } ^ { m \times V }$ and $b _ { U } ^ { ( 0 ) } \in \mathbb { R } ^ { V }$ . For any competing candidate $k \in \mathcal { V } \setminus \{ j \}$ , the decision boundary $\Pi _ { j , k }$ in the LVD possesses outward normal direction $W _ { U , j } ^ { ( 0 ) } - W _ { U , k } ^ { ( 0 ) }$ . To eliminate scale discrepancies caused by activation norms and heterogeneous token embedding magnitudes, we define the normalized decision margin of token $j$ relative to token k:

$$
\rho _ { j , k } ^ { ( 0 ) } = \frac { z _ { j } ^ { ( 0 ) } ( h _ { 0 } ) - z _ { k } ^ { ( 0 ) } ( h _ { 0 } ) } { \| W _ { U , j } ^ { ( 0 ) } - W _ { U , k } ^ { ( 0 ) } \| _ { 2 } \| h _ { 0 } \| _ { 2 } } = \frac { ( W _ { U , j } ^ { ( 0 ) } - W _ { U , k } ^ { ( 0 ) } ) ^ { \top } h _ { 0 } + ( b _ { U , j } ^ { ( 0 ) } - b _ { U , k } ^ { ( 0 ) } ) } { \| W _ { U , j } ^ { ( 0 ) } - W _ { U , k } ^ { ( 0 ) } \| _ { 2 } \| h _ { 0 } \| _ { 2 } } .\tag{41}
$$

Geometrically, $\rho _ { j , k } ^ { ( 0 ) }$ measures the normalized angular projection between the activation vector and the decision hyperplane. When $b _ { U } ^ { ( 0 ) } = \mathbf { 0 } , \rho _ { j , k } ^ { ( 0 ) }$ strictly evaluates to the directional cosine: $\rho _ { j , k } ^ { ( 0 ) } =$ cos $\angle ( W _ { U , j } ^ { ( 0 ) } - W _ { U , k } ^ { ( 0 ) } , h _ { 0 } ) \in [ - 1 , 1 ] . \mathrm { ~ A ~ }$ strictly positive baseline margin $( \rho _ { j , k } ^ { ( 0 ) } > 0 )$ guarantees that $h _ { 0 }$ resides within the safe interior of $\mathrm { C e l l } _ { j } ^ { \mathrm { L V D } }$

Because semantic sensitivity varies across contexts and candidate tokens, enforcing a rigid spherical bound risks over-constraining the model. We therefore construct an adaptive directional width $w _ { j , k }$ tailored to each competing candidate k:

$$
w _ { j , k } = \mathbb { E } _ { k \in \mathcal { V } \setminus \{ j \} } \left[ \rho _ { j , k } ^ { ( 0 ) } \right] \cdot \exp \left( - \frac { z _ { k ^ { * } } ^ { ( 0 ) } - z _ { k } ^ { ( 0 ) } } { \tau } \right) ,\tag{42}
$$

where $\mathbb { E } _ { k \in \mathcal { V } \backslash \{ j \} } [ \rho _ { j , k } ^ { ( 0 ) } ]$ characterizes the baseline average geometric safety buffer across all competing directions, and the exponential factor weights semantic proximity $( k ^ { * } = \arg \operatorname* { m a x } _ { k \neq j } z _ { k } ^ { ( 0 ) }$ represents the strongest non-target competitor, and $\tau > 0$ is a temperature hyperparameter). This formulation allocates wider exploratory space to semantical competitors while imposing tight bounds against irrelevant tokens.

Step 2: Dynamic Margin Drift Regularization. During ${ \mathrm { S F T } } ,$ model parameters update to $\theta ,$ yielding current hidden activation h, logits z, and instantaneous normalized margins:

$$
\rho _ { j , k } = \frac { z _ { j } ( h ) - z _ { k } ( h ) } { \| W _ { U , j } - W _ { U , k } \| _ { 2 } \| h \| _ { 2 } } .\tag{43}
$$

Let $\Delta \rho _ { j , k } = \rho _ { j , k } - \rho _ { j , k } ^ { ( 0 ) }$ denote the net margin shift along candidate channel k. To formulate the directional constraint, we define the composite violation distance:

$$
D _ { j , k } ( \zeta ) = \operatorname* { m a x } \Big ( | \Delta \rho _ { j , k } - \zeta | - w _ { j , k } , - \rho _ { j , k } , 0 \Big ) ,\tag{44}
$$

where $\zeta \geq 0$ is a position-shared scalar. Equation (44) incorporates two complementary defense mechanisms:

• Directional Drift Suppression $( | \Delta \rho _ { j , k } - \zeta | - w _ { j , k } )$ : This term restricts the activation drift along channel k beyond the width $w _ { j , k }$ . Introducing the shared scalar $\zeta \geq 0$ absorbs uniform, collective margin expansion across candidates, preventing optimization from penalizing shared representational improvements.

• Hyperplane Crossing Barrier $( - \rho _ { j , k } ) \colon$ If activation $h$ approaches or breaches the decision hyperplane $\Pi _ { j , k } \left( \rho _ { j , k } < 0 \right)$ , this term enforces an unyielding repulsive penalty, preventing discrete token prediction flips.

To ensure $C ^ { 1 }$ -smooth differentiability and stabilize gradient magnitudes under large deviations, we apply the Huber loss operator $\ell _ { \kappa } ( \cdot )$ to the violation distance:

$$
\ell _ { \kappa } ( D ) = \left\{ { \textstyle \frac { 1 } { 2 \kappa } } D ^ { 2 } , \quad 0 \le D \le \kappa , \right.\tag{45}
$$

where $\kappa > 0$ is a fixed smoothing threshold hyperparameter that ensures continuous gradient transitions across boundaries and avoids gradient instability. For an activation state with fixed

parameters and activations, the local AEGIS objective parameterized by scalar ζ is expressed as:

$$
\mathcal { L } _ { \mathrm { A E G I S } } ( \zeta ) = \frac { 1 } { | \mathcal { V } | - 1 } \sum _ { k \neq j } \ell _ { \kappa } \big ( D _ { j , k } ( \zeta ) \big ) .\tag{46}
$$

Because $D _ { j , k } ( \zeta )$ is convex in $\zeta$ (as the pointwise supremum of convex and affine functions) and $\ell _ { \kappa } ( \cdot )$ is convex and non-decreasing on $[ 0 , + \infty ) , { \mathcal { L } } _ { \mathrm { A E G I S } } ( \zeta )$ is strictly a one-dimensional convex function over $\zeta \in [ 0 , + \infty )$ . Given that absolute-value junctions produce non-differentiable knots, the global optimum is resolved efficiently via the monotonicity of its right derivative $D ^ { + } { \mathcal { L } } _ { \mathrm { A E G I S } } ( \zeta )$

$$
\zeta ^ { * } = \arg \operatorname* { m i n } _ { \zeta \geq 0 } \mathcal { L } _ { \mathrm { A E G I S } } ( \zeta ) .\tag{47}
$$

Evaluating at $\zeta ^ { * }$ , the instance regularization evaluates to $\mathcal { L } _ { \mathrm { A E G I S } } = \mathcal { L } _ { \mathrm { A E G I S } } ( \zeta ^ { * } )$ . By the envelope theorem [64], parameter gradients backpropagate directly through $\zeta ^ { * }$ without requiring higher-order unrolling. The specific proofs and searching algorithm are shown in Appendix F.

The composite post-training objective unifies downstream task adaptation and pervasive geometric stabilization:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { C E } } ( \mathcal { D } _ { T } ; \theta ) + \mathcal { L } _ { \mathrm { C E } } ( \mathcal { D } _ { P } ; \theta ) + \lambda \mathcal { L } _ { \mathrm { A E G I S } } ( \mathcal { D } _ { P } ; \theta ) ,\tag{48}
$$

where $\lambda > 0$ balances geometric margin defense to suppress anisotropic feature interference spikes and safeguard pervasive capabilities.

A two-dimensional geometric illustration of our framework is presented in Figure 3. Without interference mitigation $( \mathcal { L } _ { \mathrm { C E } } ( \bar { \mathcal { D } _ { T } } ) )$ or due to insufficient sample coverage $( \mathcal { L } _ { \mathrm { C E } } ( \mathcal { D } _ { T } ) + \mathcal { L } _ { \mathrm { C E } } ( \mathcal { D } _ { P } ) )$ activations incur varying degrees of boundary violation $( h _ { \mathrm { t a r g e t } }$ and h<sub>target+pervasiveness</sub>). In contrast, AEGIS (Eq. 48) enforces geometric width constraints to effectively suppress interference, confining $h _ { \mathrm { A E G I S } }$ within safe bounds. Compared to RDA, AEGIS resolves gradient deadlocks by allowing the pre-trained distribution $\mathcal { D } _ { G }$ and downstream task distribution $\mathcal { D } _ { T }$ to co-exist within directional tolerance bounds. Compared to DMA, multi-directional hyperplane supervision substantially reduces the required pervasive sample volume. Compared to GOO, it directly supports the intrinsic almost-orthogonal packing of pre-training representations, avoiding constructing limited orthogonal update directions. Finally, compared to PSO, DPP, and MIL, AEGIS regularizes representations directly in activation space, circumventing the non-transitivity orthogonality and the inescapable polysemantic coupling in weight space.

![](images/70165b3d5a251e88c4b82ba86e0d32233112da56407ff26c8bc0306c4969bd3d.jpg)  
Figure 3: In the unembedding space, $\mathrm { C e l l } _ { j }$ (light blue) is bounded by hyperplanes separating token $j$ from competitors $k _ { 1 } ~ \sim ~ k _ { 7 }$ where the blue dashed contour outlines the adaptive directional width derived from baseline activation $h _ { 0 }$

## 5 Experiments

## 5.1 Experimental Setup

Models. We evaluate our method across three model families spanning multiple parameter scales: Mistral-7B, Qwen2.5 (3B, 7B, 32B, and 72B), and Gemma-2-9B.

Datasets. The training data consists of a target task dataset $\mathcal { D } _ { T }$ and a pervasive dataset ${ \mathcal { D } } _ { P } .$ . For $\mathcal { D } _ { T }$ we adopt the eight datasets from the GLUE benchmark (CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, and RTE) for natural language understanding (NLU) task, GSM8K for math task, CodeAlpaca, Code-Bagel-Hermes, Infinity-Instruct-Code for code generation task, Evol-Instruct, and LogiQA2 for instruction following task. For $\mathcal { D } _ { P }$ , we construct $\bar { \mathcal { D } } _ { P }$ by sampling from the underlying FLAN 2022 collection, which comprises 1,840 tasks across five sub-mixtures (FLAN 2021, T0, NIV2, CoT, and Dialog). Specifically, we randomly permute the constituent tasks and iteratively select instances containing previously uncovered label tokens.

Table 2: Main results on NLU tasks.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="2">CoLA</td><td colspan="2">SST-2</td><td colspan="2">MRPC</td><td colspan="2">STS-B</td><td colspan="2">QQP</td><td colspan="2">MNLI</td><td colspan="2">QNLI</td><td colspan="2">RTE</td></tr><tr><td>MCC</td><td>P-Acc.</td><td>Acc.</td><td>P-Acc.</td><td>Acc./F1</td><td>P-Acc.</td><td>Corr.</td><td>P-Acc.</td><td>Acc./F1</td><td>P-Acc.</td><td>M-Acc.</td><td>P-Acc.</td><td>Acc.</td><td>P-Acc.</td><td>Acc.</td><td>P-Acc.</td></tr><tr><td rowspan="8">Mistral-7B</td><td>Vanilla</td><td>0.435</td><td>0.512 0.514</td><td>0.965</td><td>0.519</td><td>0.884</td><td>0.493</td><td>0.875</td><td>0.522</td><td>0.905</td><td>0.521</td><td>0.884</td><td>0.524</td><td>0.935</td><td>0.537</td><td>0.835</td><td>0.527</td></tr><tr><td>PSFT</td><td>0.449</td><td></td><td>0.977</td><td>0.528</td><td>0.891</td><td>0.485</td><td>0.886</td><td>0.518</td><td>0.892</td><td>0.515</td><td>0.892</td><td>0.519</td><td>0.939</td><td>0.542</td><td>0.854</td><td>0.525</td></tr><tr><td>SMART</td><td>0.447</td><td>0.519</td><td>0.971</td><td>0.537</td><td>0.889</td><td>0.491</td><td>0.881</td><td>0.526</td><td>0.906</td><td>0.524</td><td>0.888</td><td>0.514</td><td>0.928</td><td>0.539</td><td>0.869</td><td>0.534</td></tr><tr><td>KnOTS</td><td>0.385</td><td>0.437</td><td>0.972</td><td>0.462</td><td>0.871</td><td>0.473</td><td>0.864</td><td>0.507</td><td>0.875</td><td>0.508</td><td>0.867</td><td>0.492</td><td>0.914</td><td>0.504</td><td>0.812</td><td>0.519</td></tr><tr><td>DARE</td><td>0.398</td><td>0.463</td><td>0.947</td><td>0.465</td><td>0.865</td><td>0.466</td><td>0.859</td><td>0.501</td><td>0.869</td><td>0.486</td><td>0.871</td><td>0.488</td><td>0.905</td><td>0.511</td><td>0.819</td><td>0.504</td></tr><tr><td>DiFT</td><td>0.459</td><td>0.502</td><td>0.977</td><td>0.522</td><td>0.895</td><td>0.489</td><td>0.873</td><td>0.523</td><td>0.897</td><td>0.519</td><td>0.879</td><td>0.521</td><td>0.942</td><td>0.538</td><td>0.849</td><td>0.537</td></tr><tr><td>CLUE</td><td>0.446</td><td>0.528</td><td>0.969</td><td>0.539</td><td>0.909</td><td>0.494</td><td>0.877</td><td>0.528</td><td>0.908</td><td>0.528</td><td>0.891</td><td>0.529</td><td>0.946</td><td>0.536</td><td>0.852</td><td>0.535</td></tr><tr><td>AEGIS Vanilla</td><td>0.456</td><td>0.548</td><td>0.979</td><td>0.543</td><td>0.902</td><td>0.511</td><td>0.897</td><td>0.531</td><td>0.912</td><td>0.534</td><td>0.895</td><td>0.537</td><td>0.947</td><td>0.549</td><td>0.868</td><td>0.541</td></tr><tr><td rowspan="8">Qwen2.5-7B</td><td></td><td>0.612</td><td>0.605</td><td>0.972</td><td>0.679</td><td>0.901</td><td>0.634</td><td>0.842</td><td>0.642</td><td>0.867</td><td>0.675</td><td>0.873</td><td>0.614</td><td>0.924</td><td>0.647</td><td>0.859</td><td>0.667</td></tr><tr><td>PSFT</td><td>0.625 0.601</td><td>0.978</td><td>0.682</td><td></td><td>0.903</td><td>0.629 0.840</td><td>0.635</td><td>0.878</td><td></td><td>0.662</td><td>0.876</td><td>0.612</td><td>0.934</td><td>0.651</td><td>0.899</td><td>0.656</td></tr><tr><td>SMART</td><td>0.611 0.612</td><td>0.976</td><td></td><td>0.679</td><td>0.905</td><td>0.641 0.851</td><td>0.663</td><td></td><td>0.869</td><td>0.669</td><td>0.869</td><td>0.611</td><td>0.928</td><td>0.652</td><td>0.876</td><td>0.669</td></tr><tr><td>KnOTS</td><td>0.582</td><td>0.574</td><td>0.963</td><td>0.626</td><td>0.895</td><td>0.621</td><td>0.823</td><td>0.627</td><td>0.842</td><td>0.638</td><td>0.842</td><td>0.602</td><td>0.906</td><td>0.628</td><td>0.855</td><td>0.637</td></tr><tr><td>DARE</td><td>0.541</td><td>0.588</td><td>0.947</td><td>0.633</td><td>0.871</td><td>0.614</td><td>0.837</td><td>0.615</td><td>0.839</td><td>0.642</td><td>0.839</td><td>0.598</td><td>0.901</td><td>0.617</td><td>0.842</td><td>0.629</td></tr><tr><td>DiFT</td><td>0.612</td><td>0.602</td><td>0.979</td><td>0.675</td><td>0.906</td><td>0.640</td><td>0.855</td><td>0.646</td><td>0.872</td><td>0.679</td><td>0.880</td><td>0.612</td><td>0.928</td><td>0.651</td><td>0.881</td><td>0.674</td></tr><tr><td>CLUE</td><td>0.609 0.594</td><td>0.981</td><td></td><td>0.681</td><td>0.915</td><td>0.652</td><td>0.836</td><td>0.659</td><td>0.873</td><td>0.681</td><td>0.878</td><td>0.618</td><td>0.923</td><td>0.653</td><td>0.879</td><td>0.673</td></tr><tr><td>AEGIS</td><td>0.621 0.618</td><td>0.982</td><td></td><td>0.699</td><td>0.921</td><td>0.652</td><td>0.869</td><td>0.662</td><td>0.881</td><td>0.685</td><td>0.884</td><td>0.624</td><td>0.934</td><td>0.659</td><td>0.893</td><td>0.678</td></tr><tr><td rowspan="8">Gemma-2-9B</td><td></td><td>0.573</td><td>0.548</td><td>0.968</td><td>0.551</td><td>0.895</td><td>0.539</td><td>0.861</td><td>0.564</td><td>0.874</td><td>0.536</td><td>0.894</td><td>0.557</td><td>0.917</td><td>0.539</td><td>0.867</td><td>0.549</td></tr><tr><td>Vanilla PSFT</td><td>0.586 0.552</td><td>0.972</td><td></td><td>0.549</td><td>0.912</td><td>0.541 0.872</td><td>0.566</td><td></td><td>0.862</td><td>0.547</td><td>0.882</td><td>0.551</td><td>0.905</td><td>0.546</td><td>0.859</td><td>0.551</td></tr><tr><td>SMART</td><td>0.571</td><td>0.557 0.957</td><td></td><td>0.553</td><td>0.907</td><td>0.535 0.857</td><td>0.562</td><td></td><td>0.875</td><td>0.542</td><td>0.899</td><td>0.564</td><td>0.919</td><td>0.542</td><td>0.872</td><td>0.547</td></tr><tr><td>KnOTS</td><td>0.556</td><td>0.528</td><td>0.941</td><td>0.516</td><td>0.857</td><td>0.502</td><td>0.829</td><td>0.539</td><td>0.855</td><td>0.529</td><td>0.867</td><td>0.527</td><td>0.889</td><td>0.527</td><td>0.839</td><td>0.529</td></tr><tr><td>DARE</td><td>0.509</td><td>0.539</td><td>0.905</td><td>0.513</td><td>0.884</td><td>0.514</td><td>0.841</td><td>0.541</td><td>0.851</td><td>0.531</td><td>0.871</td><td>0.533</td><td>0.896</td><td>0.519</td><td>0.835</td><td>0.505</td></tr><tr><td>DiFT</td><td>0.588</td><td>0.541</td><td>0.978</td><td>0.551</td><td>0.899</td><td>0.539</td><td>0.865</td><td>0.561 0.571</td><td>0.869 0.878</td><td>0.537 0.541</td><td>0.895 0.901</td><td>0.569 0.565</td><td>0.918 0.924</td><td>0.556 0.549</td><td>0.877 0.874</td><td>0.551 0.553</td></tr><tr><td>CLUE</td><td>0.574 0.595</td><td>0.558 0.565</td><td>0.967 0.982</td><td>0.550 0.561</td><td>0.905 0.907</td><td>0.546 0.859 0.551 0.879</td></table>

Evaluation Metrics. For GLUE, we report the Matthews correlation coefficient (MCC) for CoLA, Pearson and Spearman correlations for STS-B, accuracy and F1 score for MRPC and QQP, matched accuracy for MNLI, and accuracy for SST-2, QNLI, and RTE. We evaluate GSM8K and LogiQA2 on their respective test splits using zero-shot accuracy. Checkpoints trained on CodeAlpaca are evaluated on HumanEval using pass@1, pass@5, and pass@10, whereas those trained on Code-Bagel Hermes and Infinity-Instruct-Code are evaluated on CodeXGLUE using token accuracy. Checkpoints trained on Evol-Instruct are evaluated on AlpacaEval via win rate. To measure pervasive capability preservation, we evaluate models on $\mathcal { D } _ { P }$ using the lm-evaluation-harness framework and report the average accuracy.

Baselines. We compare AEGIS against six representative baselines corresponding to each posttraining paradigm analyzed in Section 3.2: PSFT [92] for RDA paradigm, SMART [54] for DMA paradigm, KnOTS [62] for PSO paradigm, DARE [85] for DPP paradigm, DiFT [87] for GOO paradigm, and CLUE [7] for MIL paradigm.

Comprehensive details regarding dataset splits, prompt templates, baseline configurations, the pervasive dataset sampling algorithm, and sample and token statistics of $\mathcal { D } _ { P }$ are deferred to Appendix G.

## 5.2 Main Results

We evaluate AEGIS and representative methods from each post-training paradigm on single-targettask adaptation, measuring both target task performance via task-specific metrics and pervasive capability retention via the average accuracy (P-Acc.) across $\mathcal { D } _ { P }$ . We establish our primary Benchmark (Vanilla) as full-parameter fine-tuning that jointly optimizes standard cross-entropy loss over both the target task dataset $\mathcal { D } _ { T }$ and the pervasive dataset $\mathcal { D } _ { P }$ . Building upon this benchmark setting, each baseline paradigm incorporates its core mechanism as follows: PSFT replaces the cross-entropy loss with KL-divergence regularization; SMART substitutes $\mathcal { D } _ { P }$ with its own submodular-curated general task mixture; KnOTS fine-tunes the model exclusively on $\mathcal { D } _ { T }$ before merging it with the pretrained base model; DARE applies delta-parameter pruning and merging after SFT completion; DiFT replaces standard cross-entropy optimization with its differential parameter updating framework; and CLUE localizes and freezes 50% of the model parameters prior to SFT.

Table 2 reports the results of Mistral-7B, Qwen2.5-7B, and Gemma-2-9B across the eight NLU tasks in the GLUE benchmark. Table 3 presents the corresponding performance of these three models on mathematical reasoning (GSM8K), code generation (CodeAlpaca, Code-Bagel-Hermes, and Infinity-Instruct-Code), and logic and instruction-following (Evol-Instruct and LogiQA2) tasks. All experiments are conducted over five random seeds; the variance across seeds is consistently below 0.03 with no statistically significant impact on the results and is thus omitted for clarity.

The results in Tables 2 and 3 demonstrate that AEGIS significantly outperforms existing methods on the pervasive dataset, validating the effectiveness of enforcing geometric constraints across all token directions. Simultaneously, AEGIS achieves a clear lead on the target tasks, which we attribute to the fact that reducing feature interference alleviates overfitting to the target task; a detailed analysis is provided in Section 5.4. Furthermore, we observe that competing baselines lack stability, exhibiting substantial fluctuations in their performance rankings across different tasks, whereas AEGIS consistently remains at the forefront. We posit that this stability fundamentally stems from directly mitigating the underlying feature interference. Finally, to comprehensively evaluate AEGIS across diverse SFT scenarios, we further report results under Mix-up SFT, parameter-efficient fine-tuning (PEFT), and continual SFT scenarios in Appendix H.

Table 3: Main results on mathematical reasoning, code generation, and logic & instruction-following tasks.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="2">GSM8K</td><td colspan="4">CodeAlpaca</td><td colspan="2">Code-Bagel-Hermes</td><td colspan="2">Infinity-Instruct-Code</td><td colspan="2">Evol-Instruct</td><td colspan="2">LogiQA2</td></tr><tr><td>Acc.</td><td>P-Acc.</td><td>Pass@1</td><td>Pass@5</td><td>Pass@10</td><td>P-Acc.</td><td>Token Acc.</td><td>P-Acc.</td><td>Token Acc.</td><td>P-Acc.</td><td>Win Rate</td><td>P-Acc.</td><td>Acc.</td><td>P-Acc.</td></tr><tr><td rowspan="8">Mistral-7B</td><td>Vanilla</td><td>0.375</td><td>0.391</td><td>0.085</td><td>0.131</td><td>0.145</td><td>0.375</td><td>0.445</td><td>0.381</td><td>0.488</td><td>0.421</td><td>0.467</td><td>0.423</td><td>0.612</td><td>0.445</td></tr><tr><td>PSFT</td><td>0.374</td><td>0.399</td><td>0.084</td><td>0.134</td><td>0.151</td><td>0.396</td><td>0.416</td><td>0.385</td><td>0.495</td><td>0.422</td><td>0.461</td><td>0.421</td><td>0.618</td><td>0.432</td></tr><tr><td>SMART</td><td>0.403</td><td>0.396</td><td>0.098</td><td>0.141</td><td>0.151</td><td>0.378</td><td>0.394</td><td>0.389</td><td>0.492</td><td>0.415</td><td>0.472</td><td>0.425</td><td>0.614</td><td>0.438</td></tr><tr><td>KnOTS</td><td>0.326</td><td>0.357</td><td>0.064</td><td>0.117</td><td>0.123</td><td>0.326</td><td>0.384</td><td>0.354</td><td>0.457</td><td>0.372</td><td>0.455</td><td>0.394</td><td>0.547</td><td>0.405</td></tr><tr><td>DARE</td><td>0.354</td><td>0.361</td><td>0.083</td><td>0.129</td><td>0.144</td><td>0.354</td><td>0.397</td><td>0.329</td><td>0.443</td><td>0.388</td><td>0.439</td><td>0.388</td><td>0.463</td><td>0.411</td></tr><tr><td>DiFT</td><td>0.389</td><td>0.394</td><td>0.091</td><td>0.137</td><td>0.149</td><td>0.376</td><td>0.467</td><td>0.384</td><td>0.495</td><td>0.417</td><td>0.471</td><td>0.429</td><td>0.452</td><td>0.438</td></tr><tr><td>CLUE</td><td>0.405</td><td>0.401</td><td>0.088</td><td>0.136</td><td>0.151</td><td>0.381</td><td>0.454</td><td>0.383</td><td>0.501</td><td>0.426</td><td>0.474</td><td>0.435</td><td>0.629</td><td>0.449</td></tr><tr><td>AEGIS</td><td>0.418</td><td>0.429</td><td>0.115</td><td>0.144</td><td>0.156</td><td>0.398</td><td>0.463</td><td>0.417</td><td>0.511</td><td>0.439</td><td>0.486</td><td>0.439</td><td>0.638</td><td>0.465</td></tr><tr><td rowspan="8">Qwen2.5-7B</td><td>Vanilla</td><td>0.714</td><td>0.668</td><td>0.229</td><td>0.638</td><td>0.374</td><td>0.608</td><td>0.539</td><td>0.607</td><td>0.582</td><td>0.614</td><td>0.524</td><td>0.618</td><td>0.594</td><td>0.634</td></tr><tr><td>PSFT</td><td>0.728</td><td>0.661</td><td>0.224</td><td>0.634</td><td>0.387</td><td>0.614</td><td>0.567</td><td>0.611</td><td>0.601</td><td>0.621</td><td>0.521</td><td>0.616</td><td>0.638</td><td>0.635</td></tr><tr><td>SMART</td><td>0.725</td><td>0.676</td><td>0.235</td><td>0.643</td><td>0.385</td><td>0.611</td><td>0.526</td><td>0.616</td><td>0.593</td><td>0.626</td><td>0.529</td><td>0.617</td><td>0.659</td><td>0.624</td></tr><tr><td>KnOTS</td><td>0.629</td><td>0.628</td><td>0.194</td><td>0.555</td><td>0.327</td><td>0.584</td><td>0.457</td><td>0.557</td><td>0.554</td><td>0.581</td><td>0.487</td><td>0.577</td><td>0.587</td><td>0.628</td></tr><tr><td>DARE</td><td>0.634</td><td>0.607</td><td>0.207</td><td>0.642</td><td>0.388</td><td>0.656</td><td>0.442</td><td>0.543</td><td>0.546</td><td>0.573</td><td>0.469</td><td>0.692</td><td>0.556</td><td>0.615</td></tr><tr><td>DiFT</td><td>0.728 0.723</td><td>0.677</td><td>0.232</td><td>0.657</td><td>0.384</td><td>0.614</td><td>0.531</td><td>0.597</td><td>0.596</td><td>0.616</td><td>0.526</td><td>0.609</td><td>0.649</td><td>0.647</td></tr><tr><td>CLUE</td><td>0.739</td><td>0.669 0.683</td><td>0.244 0.277</td><td>0.666 0.682</td><td>0.398</td><td>0.629</td><td>0.545</td><td>0.605</td><td>0.584</td><td>0.625</td><td>0.531</td><td>0.621</td><td>0.634</td><td>0.645</td></tr><tr><td>AEGIS Vanilla</td><td></td><td></td><td></td><td></td><td>0.416</td><td>0.667</td><td>0.579</td><td>0.628</td><td>0.615</td><td>0.638</td><td>0.539</td><td>0.631</td><td>0.715</td><td>0.646</td></tr><tr><td rowspan="8">Gemma-2-9B</td><td></td><td>0.755</td><td>0.553</td><td>0.283</td><td>0.379 0.382</td><td>0.443</td><td>0.594</td><td>0.587</td><td>0.567</td><td>0.605</td><td>0.601</td><td>0.562</td><td>0.595</td><td>0.684</td><td>0.549</td></tr><tr><td>PSFT</td><td>0.759</td><td>0.561</td><td>0.287</td><td></td><td>0.448</td><td>0.608</td><td>0.589</td><td>0.571</td><td>0.611</td><td>0.605</td><td>0.569</td><td>0.601</td><td>0.658</td><td>0.553</td></tr><tr><td>SMART</td><td>0.763</td><td>0.558</td><td>0.295</td><td>0.381</td><td>0.459</td><td>0.601</td><td>0.591</td><td>0.569</td><td>0.608</td><td>0.604</td><td>0.565</td><td>0.608</td><td>0.679</td><td>0.546</td></tr><tr><td>KnOTS</td><td>0.729</td><td>0.527</td><td>0.241</td><td>0.352</td><td>0.397</td><td>0.557</td><td>0.554</td><td>0.541</td><td>0.568</td><td>0.547</td><td>0.541</td><td>0.562</td><td>0.629</td><td>0.529</td></tr><tr><td>DARE</td><td>0.718</td><td>0.507</td><td>0.228</td><td>0.329</td><td>0.401</td><td>0.493</td><td>0.531</td><td>0.537</td><td>0.573</td><td>0.569</td><td>0.539</td><td>0.577</td><td>0.633</td><td>0.519</td></tr><tr><td>DiFT</td><td>0.784</td><td>0.551</td><td>0.295</td><td>0.393</td><td>0.461</td><td>0.587</td><td>0.588</td><td>0.566</td><td>0.612</td><td>0.598</td><td>0.558</td><td>0.593</td><td>0.651</td><td>0.536</td></tr><tr><td>CLUE</td><td>0.763</td><td>0.562</td><td>0.299</td><td>0.397</td><td>0.468</td><td>0.617</td><td>0.594</td><td>0.579</td><td>0.608</td><td>0.603</td><td>0.574</td><td>0.598</td><td>0.696</td><td>0.554</td></tr><tr><td>AEGIS</td><td>0.801</td><td>0.584</td><td>0.337</td><td>0.438</td><td>0.496</td><td>0.622</td><td>0.608</td><td>0.597</td><td>0.614</td><td>0.602</td><td>0.582</td><td>0.614</td><td>0.703</td><td>0.559</td></tr></table>

## 5.3 Scalability and Costs

To examine whether AEGIS remains practical as the model scale grows, we profile the computational and memory costs of full-parameter SFT on GSM8K with Qwen2.5-3B, 7B, 32B, and 72B, comparing AEGIS against the Vanilla benchmark (standard cross-entropy), PSFT, and DiFT. All runs are conducted on NVIDIA A100 GPUs interconnected via NVLink, and parameters, gradients, optimizer states, and computation are all kept in FP32 throughout. We report the following metrics. (1) Total time (GPU·h) is the sum of the preprocessing and training stages, each measured as #GPUs × time. (2) Total FLOPs counts model FLOPs only, i.e., matrix multiplications over real tokens. The forward cost of a single sequence of length s is $F _ { \mathrm { f w d } } ( s ) = 2 N _ { \mathrm { m m } } s + 4 L m s ^ { 2 }$ , where $N _ { \mathrm { m m } }$ denotes the number of parameters participating in matrix multiplications, L the number of layers. (3) Model FLOPs Utilization (MFU) is defined as the training model FLOPs divided by Total time × 19.5 TFLOPS, where 19.5 TFLOPS is the peak FP32 (non-Tensor-Core) throughput of an A100. Hardware FLOPs Utilization (HFU) shares the same denominator but counts the hardware FLOPs actually executed, including padding and recomputation. Both metrics lie in [0, 1] with $\mathrm { H F U } \geq \mathrm { M F U }$ , while the gap between HFU and MFU reflects redundant computation introduced by the implementation. Consequently, a lower MFU at a comparable FLOPs budget indicates that additional wall-clock time stems from implementation overhead rather than from the algorithm itself. (4) peak memory is the sum over GPUs of the historical maximum of torch.cuda.max\_memory\_allocated. Since the training time of SMART, KnOTS,DARE, and CLUE is nearly identical to that of Vanilla, we do not report their time and FLOPs separately. The results are summarized in Table 4.

As shown in Table 4, AEGIS introduces only a moderate overhead relative to existing methods, and this overhead does not grow with model scale. In terms of time, the total cost of AEGIS is 1.31× and 1.36× that of Vanilla on Qwen2.5-72B and 32B, respectively. Moreover, the preprocessing stage can be reused across different target tasks for the same base model and anchor set, rendering this cost negligible when multiple fine-tuning tasks are conducted. In terms of computation, the total FLOPs of AEGIS remain consistently at 1.27× those of Vanilla across all scales. Finally, the MFU is consistently lower (e.g., 12.2% vs. 15.3% on Qwen2.5-72B). This suggests that the training time of AEGIS can be further reduced toward that of Vanilla through further engineering optimization of its implementation.

Table 4: Time and memory costs of SFT on GSM8K across Qwen2.5 model scales.
<table><tr><td>Model</td><td>Method</td><td>Total Time (GPU·h)</td><td>Total FLOPs</td><td>MFU</td><td>HFU</td><td>Peak Mem. (GiB)</td></tr><tr><td rowspan="4">Qwen2.5-3B</td><td>Vanilla</td><td>10.2</td><td> $1 . 6 9 ( \times 1 0 ^ { 1 7 } )$ </td><td>23.5%</td><td>39.2%</td><td>59.9</td></tr><tr><td>PSFT</td><td>25.3</td><td> $3 . 1 3 ( \times 1 0 ^ { 1 7 } )$ </td><td>19.7%</td><td>38.4%</td><td>75.5</td></tr><tr><td>DiFT</td><td> $3 . 5 + 1 2 . 7 = 2 6 . 3$ </td><td> $1 . 7 5 ( \times 1 0 ^ { 1 7 } )$ </td><td>23.6%</td><td>42.7%</td><td>61.2</td></tr><tr><td>AEGIS</td><td> $0 . 9 8 + 2 0 . 2 = 2 1 . 1$ </td><td> $2 . 1 3 ( \times 1 0 ^ { 1 7 } )$ </td><td>12.0%</td><td>20.0%</td><td>60.8</td></tr><tr><td rowspan="4">Qwen2.5-7B</td><td>Vanilla</td><td>35.2</td><td> $3 . 8 5 ( \times 1 0 ^ { 1 7 } )$ </td><td>15.6%</td><td>26.1%</td><td>147.7</td></tr><tr><td>PSFT</td><td>57.5</td><td> $6 . 7 2 ( \times 1 0 ^ { 1 7 } )$ </td><td>13.6%</td><td>28.4%</td><td>199.8</td></tr><tr><td>DiFT</td><td> $5 . 2 + 3 9 . 5 = 4 4 . 7$ </td><td> $4 . 3 8 ( \times 1 0 ^ { 1 7 } )$ </td><td>17.5%</td><td>30.3%</td><td>152.4</td></tr><tr><td>AEGIS</td><td> $1 . 9 2 + 4 2 . 5 = 4 4 . 4$ </td><td> $4 . 8 8 ( \times 1 0 ^ { 1 7 } )$ </td><td>13.1%</td><td>21.8%</td><td>148.0</td></tr><tr><td rowspan="4">Qwen2.5-32B</td><td>Vanilla</td><td>154.1</td><td> $1 . 7 4 ( \times 1 0 ^ { 1 8 } )$ </td><td>16.1%</td><td>27.3%</td><td>837</td></tr><tr><td>PSFT</td><td>239.8</td><td> $3 . 6 3 ( \times 1 0 ^ { 1 8 } )$ </td><td>14.6%</td><td>26.2%</td><td>914</td></tr><tr><td>DiFT</td><td> $9 . 6 + 1 9 8 . 7 = 2 0 8 . 3$ </td><td> $1 . 9 8 ( \times 1 0 ^ { 1 8 } )$ </td><td>17.8%</td><td>29.6%</td><td>928</td></tr><tr><td>AEGIS</td><td> $7 . 8 + 2 0 2 . 3 = 2 1 0 . 1$ </td><td> $2 . 2 1 ( \times 1 0 ^ { 1 8 } )$ </td><td>12.4%</td><td>21.0%</td><td>844</td></tr><tr><td rowspan="4">Qwen2.5-72B</td><td>Vanilla</td><td>361.4</td><td> $3 . 8 8 ( \times 1 0 ^ { 1 8 } )$ </td><td>15.3%</td><td>26.0%</td><td>3430</td></tr><tr><td>PSFT</td><td>703.5.4</td><td> $6 . 8 2 \dot { ( } \times 1 0 ^ { 1 8 } )$ </td><td>14.9%</td><td>28.3%</td><td>4375</td></tr><tr><td>DiFT</td><td> $2 1 . 3 + 4 4 8 . 6 = 4 6 9 . 9$ </td><td> $3 . 9 5 ( \times 1 0 ^ { 1 8 } )$ </td><td>18.4%</td><td>24.2%</td><td>3694</td></tr><tr><td>AEGIS</td><td> $1 7 . 1 + 4 5 7 . 5 = 4 7 4 . 7$ </td><td> $4 . 9 2 ( \times 1 0 ^ { 1 8 } )$ </td><td>12.2%</td><td>20.8%</td><td>3458</td></tr></table>

Table 5: Decision-edge crossing statistics on the pervasive dataset $\mathcal { D } _ { P }$ after SFT on RTE, GSM8K, and CodeAlpaca. All values are in units of $1 0 ^ { 6 }$ .All metrics are recorded top 50% competing tokens over V.
<table><tr><td rowspan="2">Method</td><td colspan="3">RTE</td><td colspan="3">GSM8K</td><td colspan="3">CodeAlpaca</td></tr><tr><td>Cross.</td><td>New↓</td><td>Rec.</td><td>Cross.</td><td>New ↓</td><td>Rec.</td><td>Cross.</td><td>New↓</td><td>Rec.</td></tr><tr><td>Base Model</td><td>18.569</td><td>一</td><td>一</td><td>18.569</td><td>一</td><td>一</td><td>18.569</td><td>一</td><td>一</td></tr><tr><td>Vanilla</td><td>16.668</td><td>5.700</td><td>8.659</td><td>4.373</td><td>2.321</td><td>5.748</td><td>5.915</td><td>1.941</td><td>4.516</td></tr><tr><td>PSFT</td><td>16.874</td><td>5.749</td><td>8.547</td><td>4.699</td><td>2.342</td><td>5.760</td><td>5.937</td><td>1.936</td><td>4.564</td></tr><tr><td>SMART</td><td>16.649</td><td>5.798</td><td>8.967</td><td>4.830</td><td>2.336</td><td>5.722</td><td>5.925</td><td>1.957</td><td>4.529</td></tr><tr><td>KnOTS</td><td>16.782</td><td>5.830</td><td>8.296</td><td>4.667</td><td>2.487</td><td>5.348</td><td>5.916</td><td>1.963</td><td>4.262</td></tr><tr><td>DARE</td><td>16.555</td><td>5.999</td><td>8.336</td><td>4.455</td><td>2.569</td><td>5.469</td><td>5.948</td><td>1.983</td><td>4.152</td></tr><tr><td>DiFT</td><td>16.639</td><td>5.764</td><td>8.865</td><td>4.698</td><td>2.375</td><td>5.743</td><td>5.915</td><td>1.947</td><td>4.568</td></tr><tr><td>CLUE</td><td>16.594</td><td>5.747</td><td>8.598</td><td>4.759</td><td>2.322</td><td>5.733</td><td>5.917</td><td>1.942</td><td>4.530</td></tr><tr><td>AEGIS</td><td>16.794</td><td>4.309</td><td>8.683</td><td>3.101</td><td>1.546</td><td>5.764</td><td>5.924</td><td>1.370</td><td>4.535</td></tr></table>

## 5.4 Internal Analysis

To understand why AEGIS improves performance on the pervasive dataset, we examine the intrinsic geometry of the output activations on $\mathcal { D } _ { P }$ after SFT on three representative target tasks, namely RTE (NLU), GSM8K (mathematical reasoning), and CodeAlpaca (code generation), using Qwen2.5-7B. Following the LVD view, every pervasive token position t with ground-truth token j<sub>t</sub> and every competing candidate $k \in \mathcal { V } \backslash \{ j _ { t } \}$ form a decision edge (t, k). An edge is crossed if $\rho _ { j _ { t } , k } < 0$ . We report the following metrics. (1) Cross. $\begin{array} { r } { N _ { \mathrm { c r o s s } } = \sum _ { t } \sum _ { k \neq j _ { t } } \mathcal { k } \left[ \rho _ { j _ { t } , k } < 0 \right] } \end{array}$ counts all crossed edges on ${ \mathcal { D } } _ { P } .$ , out of a total of #tokens × |V| edges. (2) New $\begin{array} { r } { N _ { \mathrm { n e w } } = \sum _ { t } \sum _ { k \neq i _ { t } } \mathcal { k } \big [ \rho _ { j _ { t } , k } ^ { ( 0 ) } > 0 , \rho _ { j _ { t } , k } < 0 \big ] } \end{array}$ counts edges that lie on the correct side under the base model but are crossed after SFT; these crossings are directly induced by target task adaptation. (3) Rec. $\begin{array} { r } { N _ { \mathrm { r e c } } = \sum _ { t } \sum _ { k \neq j _ { t } } \mathcal { k } \big [ \rho _ { j _ { t } , k } ^ { ( 0 ) } < 0 , \rho _ { j _ { t } , k } \ge 0 \big ] } \end{array}$ counts edges that are crossed under the base model but return to the correct side after SFT. The results are summarized in Table 5.

As shown in Table 5, AEGIS does not stand out in terms of the total number of crossed edges or the number of recovered edges. This is expected: since $k \gg m$ , features can only be arranged in an almost-orthogonal configuration, so a large fraction of crossings is an intrinsic trade-off of superposition that already exists in the base model and persists regardless of the SFT method. In contrast, AEGIS substantially reduces the number of newly crossed edges on every task, corresponding to relative reductions of 25.7%, 35.4%, and 29.9% with respect to the average of the baselines. These newly crossed edges are precisely the crossings predicted by Eq. (20): feature interference induced by target adaptation produces an anisotropic displacement δh that pushes activations across the hyperplanes of competing tokens coupled with the target features.

![](images/c931d1d6946d5903cf37c925ec3846f445c5519a01fca66d9929186f1b7d0d89.jpg)

![](images/216cedb152c6cb874aedff19559d856dfe3464bc5c29fcaf9bbeeb96c909b8f1.jpg)  
(a) MRPC

![](images/33489b3e66859eb02053aaec1edd1cb928beb3438e441c7455c8db95d615f928.jpg)

![](images/9d83c3147a093a2bcc3a6a94b815de0013c23195b5bc2f4b94ed54fe8f159383.jpg)  
(b) GSM8K

![](images/113c4a918ef270ba08ddd38ccaa986d3a3d304ba8c3ccb694eb7d4aca02cf514.jpg)  
(c) CodeAlpaca

![](images/86597137c009b940530bf09066511128767c40ea4c462c03d59f17a0320689c6.jpg)

![](images/d6e5e2610c85cceee00dbd28fba2f421a02905c7c970a3b316dfa42ae9945046.jpg)

![](images/787b14fce0224dcd53a3476148f9e7cbe477bf5e72b16a72006783d4ca05dde9.jpg)  
(d) LogiQA2  
Figure 4: Loss and validation accuracy of Qwen2.5-7B during SFT with standard cross-entropy (Vanilla) and AEGIS on MRPC, GSM8K, CodeAlpaca, and LogiQA2. For each task, the left panel shows the total training loss together with the validation accuracy on the target task, and the right panel separates the training loss into the cross-entropy on the target task dataset $\mathcal { D } _ { T }$ and on the pervasive dataset $\mathcal { D } _ { P }$

Beyond the pervasive dataset, AEGIS also improves target task performance. We hypothesize that this gain arises because reducing feature interference on pervasive activations in turn suppresses overfitting of the target activations. To verify this hypothesis, we examine how the loss relates to validation accuracy for AEGIS and for standard cross-entropy SFT (Vanilla). Figure 4 shows the loss and validation accuracy of Qwen2.5-7B over training steps on MRPC, GSM8K, CodeAlpaca, and LogiQA2. AEGIS does not reach a lower target task cross-entropy than Vanilla, and its target cross-entropy is even slightly higher. In contrast, AEGIS substantially lowers the cross-entropy on the pervasive dataset, and its target validation accuracy stays consistently above that of Vanilla throughout training. This indicates that, by reducing feature interference on the pervasive dataset, AEGIS indeed mitigates overfitting to the target task and thus achieves higher validation accuracy at a similar loss level. To further examine which token directions are suppressed by AEGIS, we present examples of the top-10 drift directions in Appendix I.

## 6 Conclusion

This paper presents a bottom-up theoretical account of PCD. By coupling feature interference under superposition with the Laguerre–Voronoi geometry of the unembedding layer, the theory traces an unbroken causal chain from activations through weights and gradients to output logits, explaining why narrow-domain SFT inherently induces PCD. Within this framework, we distill existing methods spanning reinforcement learning, continual learning, preference alignment, model merging, and data curation into six paradigms and identify the theoretical ceiling of each. We further prove two structural limitations of standard cross-entropy under one-hot supervision: the entanglement of feature interference with task adaptation, and a strong dependence on massive, diverse samples. To address both, we propose the AEGIS loss, which takes the activations of the base model as a reference and imposes adaptive margin-drift constraints along the decision hyperplanes of all competing tokens. Our results show that PCD need not be an irreconcilable trade-off between acquiring new tasks and preserving existing capabilities: once interference can be observed and controlled at the level of microscopic geometry, both can improve together. AEGIS is at once general, scalable, and mechanistically interpretable; extending this framework to intermediate-layer representations, reinforcement learning and preference alignment, model merging, and longer continual-learning sequences is a promising direction for future work.

## References

[1] J. Bao, J.-C. Gu, Q. Liu, H. Zhang, K.-W. Chang, N. Peng, and W. Wang. Geometric calibration of supervised fine-tuning. arXiv preprint arXiv:2610.18634, 2026.

[2] J. Budd, J. Ideami, B. M. Rynne, K. Duggar, and R. Balestriero. Splinterp: Improving our understanding and training of sparse autoencoders, 2025. URL https://arxiv.org/ abs/2505.11836.

[3] A. Chaudhry, M. Ranzato, M. Rohrbach, and M. Elhoseiny. Efficient lifelong learning with a-GEM. In International Conference on Learning Representations, 2019. URL https: //openreview.net/forum?id=Hkf2\_sC5FX.

[4] H. Chen, J. Zhu, X. Yang, and W. Wang. Rethinking circuit completeness in language models: And, or, and adder gates. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, editors, Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 150511–150540. Curran Associates, Inc., 2025. doi: 10.52202/085713-5032. URL https://proceedings.neurips.cc/paper\_files/paper/2025/file/ dd37fdb24a4e1cfa3ed5c247217a7394-Paper-Conference.pdf.

[5] H. Chen, J. Zhu, C. Hongyang, X. Yang, and W. Wang. Navigating by old maps: The pitfalls of static mechanistic localization in llm post-training. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, editors, Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 150511–150540. Curran Associates, Inc., 2026. doi: 10.52202/085713-5032.

[6] H. Chen, J. Zhu, and W. Wang. Beyond static interpretability: Anticipating post-sft mechanisms from pre-sft parameters for better tuning, 2026. URL https://arxiv.org/abs/2608. 24482.

[7] H. Chen, J. Zhu, X. Yang, and W. Wang. Clue: Conflict-guided localization for llm unlearning framework. In C. Vondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust, editors, International Conference on Learning Representations, volume 2026, pages 52000–52021, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/ file/5576b86eb5deff816e23745366a88165-Paper-Conference.pdf.

[8] H. W. Chung, L. Hou, S. Longpre, B. Zoph, Y. Tay, W. Fedus, Y. Li, X. Wang, M. Dehghani, S. Brahma, A. Webson, S. S. Gu, Z. Dai, M. Suzgun, X. Chen, A. Chowdhery, A. Castro-Ros, M. Pellat, K. Robinson, D. Valter, S. Narang, G. Mishra, A. Yu, V. Zhao, Y. Huang, A. Dai, H. Yu, S. Petrov, E. H. Chi, J. Dean, J. Devlin, A. Roberts, D. Zhou, Q. V. Le, and J. Wei. Scaling instruction-finetuned language models. Journal of Machine Learning Research, 25(70): 1–53, 2024. URL http://jmlr.org/papers/v25/23-0870.html.

[9] D. Dai, L. Dong, Y. Hao, Z. Sui, B. Chang, and F. Wei. Knowledge neurons in pretrained transformers. In Proceedings ofthe 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 8493–8502, 2022.

[10] D. Deng, G. Chen, J. Hao, Q. Wang, and P.-A. Heng. Flattening sharpness for dynamic gradient projection memory benefits continual learning. Advances in Neural Information Processing Systems, 34:18710–18721, 2021.

[11] W. Deng, Y. Zhao, V. Vakilian, M. Chen, X. Li, and C. Thrampoulidis. Dare the extreme: Revisiting delta-parameter pruning for fine-tuned models. In International Conference on Learning Representations, volume 2025, pages 83658–83692, 2025.

[12] X. Deng, H. Zhong, R. Ai, F. Feng, Z. Wang, and X. He. Less is more: Improving llm alignment via preference data selection. Advances in Neural Information Processing Systems, 38:161259–161285, 2026.

[13] G. Dong, H. Yuan, K. Lu, C. Li, M. Xue, D. Liu, W. Wang, Z. Yuan, C. Zhou, and J. Zhou. How abilities in large language models are affected by supervised fine-tuning data composition. In L.-W. Ku, A. Martins, and V. Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 177–198,

Bangkok, Thailand, Aug. 2024. Association for Computational Linguistics. doi: 10.18653/v1/ 2024.acl-long.12. URL https://aclanthology.org/2024.acl-long.12/.

[14] G. Du, J. Lee, J. Li, R. Jiang, Y. Guo, S. Yu, H. Liu, S. K. Goh, H.-K. Tang, D. He, et al. Parameter competition balancing for model merging. Advances in Neural Information Processing Systems, 37:84746–84776, 2024.

[15] N. Elhage, T. Hume, C. Olsson, N. Schiefer, T. Henighan, S. Kravec, Z. Hatfield-Dodds, R. Lasenby, D. Drain, C. Chen, R. Grosse, S. McCandlish, J. Kaplan, D. Amodei, M. Wattenberg, and C. Olah. Toy models of superposition, 2022. URL https://arxiv.org/abs/2209. 10652.

[16] M. Farajtabar, N. Azizan, A. Mott, and A. Li. Orthogonal gradient descent for continual learning. In International conference on artificial intelligence and statistics, pages 3762–3773. PMLR, 2020.

[17] C. Friedrich. A model qualification method for mechanistic physiological qsp models to support model-informed drug development. CPT: pharmacometrics & systems pharmacology, 5(2): 43–53, 2016.

[18] T. Fu, D. Cai, L. Liu, S. Shi, and R. Yan. Disperse-then-merge: Pushing the limits of instruction tuning via alignment tax reduction. In L.-W. Ku, A. Martins, and V. Srikumar, editors, Findings of the Association for Computational Linguistics: ACL 2024, pages 2967–2985, Bangkok, Thailand, Aug. 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024. findings-acl.175. URL https://aclanthology.org/2024.findings-acl.175/.

[19] A. A. Gargiulo, D. Crisostomi, M. S. Bucarelli, S. Scardapane, F. Silvestri, and E. Rodola. Task singular vectors: Reducing task interference in model merging. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18695–18705. IEEE, 2025.

[20] M. Geva, R. Schuster, J. Berant, and O. Levy. Transformer feed-forward layers are key-value memories. In Proceedings ofthe 2021 conference on empirical methods in natural language processing, pages 5484–5495, 2021.

[21] A. Gromov, K. Tirumala, H. Shapourian, P. Glorioso, and D. A. Roberts. The unreasonable ineffectiveness of the deeper layers. In International Conference on Learning Representations, volume 2025, pages 81906–81920, 2025.

[22] Y. Harada, Y. Yamauchi, Y. Oda, Y. Oseki, Y. Miyao, and Y. Takagi. Massive supervised finetuning experiments reveal how data, layer, and training factors shape LLM alignment quality. In C. Christodoulopoulos, T. Chakraborty, C. Rose, and V. Peng, editors, Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 22360–22381, Suzhou, China, Nov. 2025. Association for Computational Linguistics. ISBN 979-8-89176- 332-6. doi: 10.18653/v1/2025.emnlp-main.1138. URL https://aclanthology.org/ 2025.emnlp-main.1138/.

[23] P. Hase, M. Bansal, B. Kim, and A. Ghandeharioun. Does localization inform editing? surprising differences in causality-based localization vs. knowledge editing in language models. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 17643–17668. Curran Associates, Inc., 2023. doi: 10.52202/075280-0774. URL https://proceedings.neurips.cc/paper\_files/paper/2023/file/ 3927bbdcf0e8d1fa8aa23c26f358a281-Paper-Conference.pdf.

[24] E. J. Hu, yelong shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=nZeVKeeFYf9.

[25] J. Jia, J. Liu, Y. Zhang, P. Ram, N. Baracaldo, and S. Liu. Wagle: Strategic weight attribution for effective and modular unlearning in large language models. Advances in Neural Information Processing Systems, 37:55620–55646, 2024.

[26] S. Kakade and J. Langford. Approximately optimal approximate reinforcement learning. In Proceedings of the nineteenth international conference on machine learning, pages 267–274, 2002.

[27] F. Kang, M. Kuchnik, K. Padthe, M. Vlastelica, R. Jia, C.-J. Wu, and N. Ardalani. Quagmires in sft-rl post-training: When high sft scores mislead and what to use instead. In C. Vondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust, editors, International Conference on Learning Representations, volume 2026, pages 56876–56918, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/ file/5d6ae8ba43ecb378030753c4408ef9bd-Paper-Conference.pdf.

[28] A. Karan, S. Chen, and Y. Du. Finetuning with sampling: Sft learns better than you think, 2026. URL https://arxiv.org/abs/2610.02140.

[29] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska, D. Hassabis, C. Clopath, D. Kumaran, and R. Hadsell. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114(13):3521–3526, 2017. doi: 10.1073/pnas.1611835114. URL https://www.pnas.org/doi/abs/10.1073/pnas.1611835114.

[30] A. Kolluri, S. Wu, J. S. Park, and M. S. Bernstein. Finetuning LLMs for human behavior prediction in social science experiments. In C. Christodoulopoulos, T. Chakraborty, C. Rose, and V. Peng, editors, Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 30096–30111, Suzhou, China, Nov. 2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.1530. URL https://aclanthology.org/2025.emnlp-main.1530/.

[31] H. Lai, X. Liu, J. Gao, J. Cheng, Z. Qi, Y. Xu, S. Yao, D. Zhang, J. Du, Z. Hou, X. Lv, M. Huang, Y. Dong, and J. Tang. A survey of post-training scaling in large language models. In W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, editors, Proceedings ofthe 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2771–2791, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.140. URL https://aclanthology. org/2025.acl-long.140/.

[32] Q. Li, Y. Wang, Q. Huang, C. Ji, and S. Wang. Alignment-consistent multimodal learning under uncertain correspondence. IEEE Transactions on Pattern Analysis and Machine Intelligence, pages 1–18, 2026. doi: 10.1109/TPAMI.2026.3736070.

[33] S. Li, K. Jia, Y. Wen, T. Liu, and D. Tao. Orthogonal deep neural networks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(4):1352–1368, 2021. doi: 10.1109/TPAMI. 2019.2948352.

[34] W. Li, Y. Peng, M. Zhang, L. Ding, H. Hu, and L. Shen. Deep model fusion: A survey. IEEE Transactions on Neural Networks and Learning Systems, 2025.

[35] Y. Li, Z. Liu, and E. Xing. Data mixing optimization for supervised fine-tuning of large language models. arXiv preprint arXiv:2508.11953, 2025.

[36] Y. Li, C.-E. Sun, and T.-W. Weng. Effective skill unlearning through intervention and abstention. In Proceedings ofthe 2025 Conference ofthe Nations ofthe Americas Chapter ofthe Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6358–6371, 2025.

[37] S. Lin, L. Yang, D. Fan, and J. Zhang. Trgp: Trust region gradient projection for continual learning. The Tenth International Conference on Learning Representations. URL https: //par.nsf.gov/biblio/10352359.

[38] Q. Liu, X. Wu, X. Zhao, Y. Zhu, D. Xu, F. Tian, and Y. Zheng. When moe meets llms: Parameter efficient fine-tuning for multi-task medical applications. In Proceedings ofthe 47th international ACM SIGIR conference on research and development in information retrieval, pages 1104–1114, 2024.

[39] S. Liu, Y. Yao, J. Jia, S. Casper, N. Baracaldo, P. Hase, Y. Yao, C. Y. Liu, X. Xu, H. Li, et al. Rethinking machine unlearning for large language models. Nature Machine Intelligence, 7(2): 181–194, 2025.

[40] W. Liu, R. Lin, Z. Liu, J. M. Rehg, L. Paull, L. Xiong, L. Song, and A. Weller. Orthogonal over-parameterized training. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7247–7256, 2021. doi: 10.1109/CVPR46437.2021.00717.

[41] D. Lopez-Paz and M. Ranzato. Gradient episodic memory for continual learning. Advances in neural information processing systems, 30, 2017.

[42] D. Luo, M. Liu, H. Zhang, X. Wang, Q. Gao, N. Kuang, T. Yin, and Z. Zheng. Comparative performance of chinese and international large language models on the chinese radiology attending physician qualification examination. Scientific reports, 15(1):39379, 2025.

[43] C. Ma and R. Wolfinger. Laguerre geometry for interpreting large language models. arXiv preprint arXiv:2607.10578, 2026.

[44] R. K. Mahabadi, S. Ruder, M. Dehghani, and J. Henderson. Parameter-efficient multi-task fine-tuning for transformers via shared hypernetworks. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 565–576, 2021.

[45] R. May. Coupon collecting with quotas. the electronic journal of combinatorics, pages N31–N31, 2008.

[46] J. L. McClelland and D. E. Rumelhart. Neural and Conceptual Interpretation of PDP Models, pages 390–431. 1987.

[47] K. Meng, D. Bau, A. Andonian, and Y. Belinkov. Locating and editing factual associations in gpt. Advances in neural information processing systems, 35:17359–17372, 2022.

[48] G. Molinari and F. Ciravegna. Towards pervasive distributed agentic generative ai–a state of the art. 58(14), July 2026. ISSN 0360-0300. doi: 10.1145/3821566. URL https: //doi.org/10.1145/3821566.

[49] A. Muhamed, X. Zhao, M. T. Diab, V. Smith, and D. Song. Towards a mechanistic understanding of robustness in finetuned reasoning models. In NeurIPS 2025 Workshop on Efficient Reasoning, 2025. URL https://openreview.net/forum?id=ztbMP3JzoS.

[50] S. J. Pan and Q. Yang. A survey on transfer learning. IEEE Transactions on Knowledge and Data Engineering, 22(10):1345–1359, 2010. doi: 10.1109/TKDE.2009.191.

[51] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

[52] M. Rahman, A. Bhuiyan, M. S. Islam, M. T. R. Laskar, R. Mahbub, A. Masry, S. Joty, and E. Hoque. Llm-based data science agents: A survey of capabilities, challenges, and future directions, 2025. URL https://arxiv.org/abs/2510.04023.

[53] A. Razdaibiedina, A. Khetan, Z. Karnin, D. Khashabi, and V. Madan. Representation projection invariance mitigates representation collapse. In H. Bouamor, J. Pino, and K. Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, pages 14638–14664, Singapore, Dec. 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023. findings-emnlp.975. URL https://aclanthology.org/2023.findings-emnlp. 975/.

[54] H. K. Renduchintala, S. Bhatia, and G. Ramakrishnan. Smart: Submodular data mixture strategy for instruction tuning. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pages 12916–12934, 2024.

[55] G. Saha, I. Garg, and K. Roy. Gradient projection memory for continual learning. In International Conference on Learning Representations, 2021. URL https://openreview.net/ forum?id=3AOj0RCNC2.

[56] V. Sanh, A. Webson, C. Raffel, S. Bach, L. Sutawika, Z. Alyafeai, A. Chaffin, A. Stiegler, A. Raja, M. Dey, M. S. Bari, C. Xu, U. Thakker, S. S. Sharma, E. Szczechla, T. Kim, G. Chhablani, N. Nayak, D. Datta, J. Chang, M. T.-J. Jiang, H. Wang, M. Manica, S. Shen, Z. X. Yong, H. Pandey, R. Bawden, T. Wang, T. Neeraj, J. Rozen, A. Sharma, A. Santilli, T. Fevry, J. A. Fries, R. Teehan, T. L. Scao, S. Biderman, L. Gao, T. Wolf, and A. M. Rush. Multitask prompted training enables zero-shot task generalization. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=9Vrb9D0WI4.

[57] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[58] J. Serra, D. Suris, M. Miron, and A. Karatzoglou. Overcoming catastrophic forgetting with hard attention to the task. In J. Dy and A. Krause, editors, Proceedings ofthe 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 4548–4557. PMLR, 10–15 Jul 2018. URL https://proceedings.mlr.press/ v80/serra18a.html.

[59] N. B. Shank and H. Yang. Coupon collector problem for non-uniform coupons and random quotas. the electronic journal of combinatorics, pages P33–P33, 2013.

[60] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

[61] H. Shin, L. Ji, X. Liu, Z. Yu, H. Yoo, Q. Chen, and Y. Gong. Dynamixsft: Dynamic mixture optimization of instruction tuning collections. In Findings of the Association for Computational Linguistics: ACL 2026, pages 39590–39603, 2026.

[62] G. Stoica, P. Ramesh, B. Ecsedi, L. Choshen, and J. Hoffman. Model merging with svd to tie the knots. In International Conference on Learning Representations, volume 2025, pages 4501–4519, 2025.

[63] V. Sunder. The riesz representation theorem. Indian J. pure appl. Math, 39(6):467–481, 2008.

[64] L. Vandenberghe and S. Boyd. Convex optimization, volume 1. Cambridge university press Cambridge, 2004.

[65] B. Wang, Q. Cheng, R. Peng, R. Bao, P. Li, Q. Guo, L. Li, Z. Zeng, Y. Zhou, and X. Qiu. Implicit reward as the bridge: A unified view of sft and dpo connections. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, editors, Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 70255–70279. Curran Associates, Inc., 2025. doi: 10.52202/085713-2362. URL https://proceedings.neurips.cc/paper\_files/paper/2025/file/ 6599417a0b34d6ab7836cf86f8dd138c-Paper-Conference.pdf.

[66] F. Wang, N. Mehrabi, P. Goyal, R. Gupta, K.-W. Chang, and A. Galstyan. Data advisor: Dynamic data curation for safety alignment of large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 8089–8100, 2024.

[67] L. Wang, S. Chen, L. Jiang, S. Pan, R. Cai, S. Yang, and F. Yang. Parameter-efficient fine-tuning in large language models: a survey of methodologies. Artificial Intelligence Review, 58(8):227, 2025.

[68] X. Wang, Y. Hu, W. Du, R. Cheng, B. Wang, and D. Zou. Towards understanding fine-tuning mechanisms of LLMs via circuit analysis. In A. Singh, M. Fazel, D. Hsu, S. Lacoste-Julien, F. Berkenkamp, T. Maharaj, K. Wagstaff, and J. Zhu, editors, Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 63088–63112. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr.press/v267/wang25ak.html.

[69] Y. Wang, S. Mishra, P. Alipoormolabashi, Y. Kordi, A. Mirzaei, A. Naik, A. Ashok, A. S. Dhanasekaran, A. Arunkumar, D. Stap, E. Pathak, G. Karamanolakis, H. Lai, I. Purohit, I. Mondal, J. Anderson, K. Kuznia, K. Doshi, K. K. Pal, M. Patel, M. Moradshahi, M. Parmar, M. Purohit, N. Varshney, P. R. Kaza, P. Verma, R. S. Puri, R. Karia, S. Doshi, S. K. Sampat, S. Mishra, S. Reddy A, S. Patro, T. Dixit, and X. Shen. Super-NaturalInstructions: Generalization via declarative instructions on 1600+ NLP tasks. In Y. Goldberg, Z. Kozareva, and Y. Zhang, editors, Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 5085–5109, Abu Dhabi, United Arab Emirates, Dec. 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.340. URL https://aclanthology.org/2022.emnlp-main.340/.

[70] Z. Wang, B. Bi, S. K. Pentyala, K. Ramnath, S. Chaudhuri, S. Mehrotra, X.-B. Mao, S. Asur, et al. A comprehensive survey of llm alignment techniques: Rlhf, rlaif, ppo, dpo and more. arXiv e-prints, pages arXiv–2407, 2024.

[71] J. Wei, M. Bosma, V. Zhao, K. Guu, A. W. Yu, B. Lester, N. Du, A. M. Dai, and Q. V. Le. Finetuned language models are zero-shot learners. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=gEZrGCozdqR.

[72] Z. Wen, Z. Zhai, G. Sun, Y. Wang, Q. Li, H. Shao, and S. Wang. Spira: Sparse informationgeometric rank adaptation for parameter-efficient fine-tuning of large pretrained models. IEEE Transactions on Pattern Analysis and Machine Intelligence, pages 1–16, 2026. doi: 10.1109/ TPAMI.2026.3724606.

[73] X. Wu, J. Li, M. Xu, W. Dong, S. Wu, C. Bian, and D. Xiong. Depn: Detecting and editing privacy neurons in pretrained language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 2875–2886, 2023.

[74] L. Xu, H. Xie, S. J. Qin, X. Tao, and F. L. Wang. Parameter-efficient fine-tuning methods for pretrained language models: A critical review and assessment. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.

[75] Z. Xu, F. Jiang, L. Niu, Y. Deng, R. Poovendran, Y. Choi, and B. Y. Lin. Magpie: Alignment data synthesis from scratch by prompting aligned llms with nothing. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu, editors, International Conference on Learning Representations, volume 2025, pages 76346–76382, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/ file/be06e3802e9411381feece79b4d960c1-Paper-Conference.pdf.

[76] W. Xuan, R. Yang, H. Qi, Q. Zeng, Y. Xiao, A. Feng, D. Liu, Y. Xing, J. Wang, F. Gao, et al. Mmlu-prox: A multilingual benchmark for advanced large language model evaluation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 1513–1532, 2025.

[77] L. Yan, R. Li, G. Chen, Q. Li, J. Geng, W. Li, V. Wang, and C. Lee. Spurious rewards paradox: Mechanistically understanding how rlvr activates memorization shortcuts in llms. arXiv preprint arXiv:2601.11061, 2026.

[78] E. Yang, Z. Wang, L. Shen, S. Liu, G. Guo, X. Wang, and D. Tao. Adamerging: Adaptive model merging for multi-task learning. In International Conference on Learning Representations, volume 2024, pages 22743–22763, 2024.

[79] E. Yang, L. Shen, G. Guo, X. Wang, X. Cao, J. Zhang, and D. Tao. Model merging in llms, mllms, and beyond: Methods, theories, applications, and opportunities. ACM Computing Surveys, 58(8):1–41, 2026.

[80] Y. Yang, J. Steinhardt, and W. Hu. Are neurons actually collapsed? On the fine-grained structure in neural representations. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 39453–39487. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/v202/yang23m.html.

[81] Y. Yang, J. Zhou, X. Ding, T. Huai, S. Liu, Q. Chen, Y. Xie, and L. He. Recent advances of foundation language models-based continual learning: A survey. ACM Computing Surveys, 57 (5):1–38, 2025.

[82] Y. Ye, Z. Huang, Y. Xiao, E. Chern, S. Xia, and P. Liu. LIMO: Less is more for reasoning. In Second Conference on Language Modeling, 2025. URL https://openreview.net/ forum?id=T2TZ0RY4Zk.

[83] D. Yin, X. Liu, F. Yin, M. Zhong, H. Bansal, J. Han, and K.-W. Chang. Dynosaur: A dynamic growth paradigm for instruction-tuning data curation. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4031–4047, 2023.

[84] R. Young. What is the alignment tax?, 2026. URL https://arxiv.org/abs/2603. 00047.

[85] L. Yu, B. Yu, H. Yu, F. Huang, and Y. Li. Language models are super mario: Absorbing abilities from homologous models as a free lunch. In R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett, and F. Berkenkamp, editors, Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 57755–57775. PMLR, 21–27 Jul 2024. URL https: //proceedings.mlr.press/v235/yu24p.html.

[86] T. Yu, S. Kumar, A. Gupta, S. Levine, K. Hausman, and C. Finn. Gradient surgery for multi-task learning. Advances in neural information processing systems, 33:5824–5836, 2020.

[87] X. Yuan, C. Shen, S. Yan, X. Zhang, S. Fan, L. Xie, W. Wang, R. Guan, Y. Wang, J. Ye, et al. Differential fine-tuning large language models towards better diverse reasoning abilities. In International Conference on Learning Representations, volume 2026, pages 110433–110463, 2026.

[88] F. Zenke, B. Poole, and S. Ganguli. Continual learning through synaptic intelligence. In D. Precup and Y. W. Teh, editors, Proceedings ofthe 34th International Conference on Machine Learning, volume 70 of Proceedings ofMachine Learning Research, pages 3987–3995. PMLR, 06–11 Aug 2017. URL https://proceedings.mlr.press/v70/zenke17a.html.

[89] J. Zheng, S. Qiu, C. Shi, and Q. Ma. Towards lifelong learning of large language models: A survey. ACM Computing Surveys, 57(8):1–35, 2025.

[90] C. Zhou, P. Liu, P. Xu, S. Iyer, J. Sun, Y. Mao, X. Ma, A. Efrat, P. Yu, L. Yu, et al. Lima: Less is more for alignment. Advances in Neural Information Processing Systems, 36:55006–55021, 2023.

[91] D. Zhu, Z. Sun, Z. Li, T. Shen, K. Yan, S. Ding, C. Wu, and K. Kuang. Model tailor: Mitigating catastrophic forgetting in multi-modal large language models. In R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett, and F. Berkenkamp, editors, Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 62581–62598. PMLR, 21–27 Jul 2024. URL https: //proceedings.mlr.press/v235/zhu24l.html.

[92] W. Zhu, R. Xie, R. Wang, S. Sun, D. Wang, and P. Liu. Proximal supervised fine-tuning. In C. Vondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust, editors, International Conference on Learning Representations, volume 2026, pages 134993–135015, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/ file/da75d2bbf862b86f10241d0887613b41-Paper-Conference.pdf.

[93] D. M. Ziegler, N. Stiennon, J. Wu, T. B. Brown, A. Radford, D. Amodei, P. Christiano, and G. Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019.

## A Algebraic Equivalence and Directional Transfer between Weights and Features

In this appendix, we provide the formal proofs for Orthogonal Equivalence and Directional Additivity, establishing the mathematical duality between parameter operations and activation-space feature geometry directly from forward propagation dynamics.

## A.1 Forward Functional Consistency and Channel Duality

Let $\boldsymbol { h } _ { \mathrm { i n } } \in \mathbb { R } ^ { m }$ denote an arbitrary activation vector in the residual stream. Under the feature superposition hypothesis (Definition $2 ) , h _ { \mathrm { i n } }$ decomposes linearly over the overcomplete unit feature dictionary $F = [ \overbar { f } _ { 1 } , \dots , f _ { k } ] \in \mathbb { R } ^ { m \times k } ( \lVert f _ { i } \rVert _ { 2 } = 1 )$ with activation intensity vector $\dot { a } = [ a _ { 1 } , \ldots , a _ { k } ] ^ { \top } \in \mathbb { R } ^ { k }$ :

$$
h _ { \mathrm { i n } } = F a = \sum _ { j = 1 } ^ { k } a _ { j } f _ { j } .\tag{49}
$$

According to Definition 2, probing the i-th latent semantic feature directly within the residual stream corresponds to projecting $h _ { \mathrm { i n } }$ onto unit direction $f _ { i }$

$$
\mathcal { R } _ { i } ^ { \mathrm { i d e a l } } ( h _ { \mathrm { i n } } ) = f _ { i } ^ { \top } h _ { \mathrm { i n } } = a _ { i } ( f _ { i } ^ { \top } f _ { i } ) + \sum _ { j \ne i } a _ { j } ( f _ { i } ^ { \top } f _ { j } ) = a _ { i } + \sum _ { j \ne i } a _ { j } ( f _ { i } ^ { \top } f _ { j } ) .\tag{50}
$$

Under Hypothesis 1 (Virtual Sparse Weight Hypothesis), the intermediate Transformer operator factorizes losslessly into feature-decoupled operators $M = W _ { \mathrm { i n } } ^ { \mathrm { v i r t } } W _ { \mathrm { o u t } } ^ { \mathrm { v i r t } }$ , where the read-in operator $W \equiv W _ { \mathrm { o u t } } ^ { \mathrm { v i r t } } \in \mathbb { R } ^ { k \times m }$ possesses k virtual parameter rows. When $h _ { \mathrm { i n } }$ enters the layer, the physical scalar response produced by the i-th virtual row vector $W _ { i } \in \mathbb { R } ^ { 1 \times m }$ evaluates to:

$$
\hat { a } _ { i } = W _ { i } h _ { \mathrm { i n } } = W _ { i } \left( \sum _ { j = 1 } ^ { k } a _ { j } f _ { j } \right) = a _ { i } ( W _ { i } f _ { j } ) + \sum _ { j \ne i } a _ { j } ( W _ { i } f _ { j } ) .\tag{51}
$$

By definition of the virtual decoupled parameter space in Hypothesis 1, the i-th virtual channel is bijectively dedicated to detecting feature i. Therefore, for any input state $h _ { \mathrm { i n } } \in \mathbb { R } ^ { m }$ , the channel output functional $\hat { a } _ { i } ( h _ { \mathrm { i n } } )$ must be collinear with the idealized directional feature probe $\mathcal { R } _ { i } ^ { \mathrm { i d e a l } } ( h _ { \mathrm { i n } } )$ Taking the inner product on both sides:

$$
W _ { i } h _ { \mathrm { i n } } = \| W _ { i } \| _ { 2 } \left( f _ { i } ^ { \top } h _ { \mathrm { i n } } \right) , \quad \forall h _ { \mathrm { i n } } \in \mathbb { R } ^ { m } .\tag{52}
$$

Since Eq. (52) holds for all activations spanning $\mathbb { R } ^ { m }$ , by the Riesz representation theorem [63] in finite-dimensional Euclidean space, the functional identity simplifies to the structural equivalence:

$$
W _ { i } = \| W _ { i } \| _ { 2 } f _ { i } ^ { \top } .\tag{53}
$$

## A.2 Proof of Proposition 1: Orthogonal Equivalence

Proof. Consider two distinct feature channels a and $b ( a , b \in \{ 1 , . . . , k \} )$ with virtual weight parameter row vectors $W _ { a } , W _ { b } \in \mathbb { R } ^ { 1 \times m }$ . By Eq. (53), their parameter vectors satisfy:

$$
\begin{array} { r } { W _ { a } = \| W _ { a } \| _ { 2 } { f _ { a } ^ { { \top } } } , \quad W _ { b } = \| W _ { b } \| _ { 2 } { f _ { b } ^ { { \top } } } . } \end{array}\tag{54}
$$

The standard Euclidean inner product between the two row vectors is defined as $W _ { a } \cdot W _ { b } = W _ { a } W _ { b } ^ { \top }$ Substituting the feature representation:

$$
\begin{array} { r } { { \cal W } _ { a } \cdot { \cal W } _ { b } = \left( \| { \cal W } _ { a } \| _ { 2 } f _ { a } ^ { \top } \right) \left( \| { \cal W } _ { b } \| _ { 2 } f _ { b } ^ { \top } \right) ^ { \top } = \| { \cal W } _ { a } \| _ { 2 } \| { \cal W } _ { b } \| _ { 2 } \left( f _ { a } ^ { \top } f _ { b } \right) . } \end{array}\tag{55}
$$

## A.3 Proof of Proposition 2: Directional Additivity and Linear Transfer

Proof. Let the parameter row corresponding to feature a undergo a perturbation along the channel of feature b:

$$
W _ { a } ^ { \prime } = W _ { a } + \lambda W _ { b } , \quad { \mathrm { w h e r e ~ } } \lambda \in \mathbb { R } .\tag{56}
$$

Evaluating the perturbed channel’s readout response on an arbitrary activation $\boldsymbol { h } _ { \mathrm { i n } } \in \mathbb { R } ^ { m }$

$$
\widehat { a } _ { a } ^ { \prime } ( h _ { \mathrm { i n } } ) = W _ { a } ^ { \prime } h _ { \mathrm { i n } } = ( W _ { a } + \lambda W _ { b } ) h _ { \mathrm { i n } } = W _ { a } h _ { \mathrm { i n } } + \lambda W _ { b } h _ { \mathrm { i n } } .\tag{57}
$$

Substituting the functional identity from Eq. (52):

$$
\begin{array} { r l } & { \hat { a } _ { a } ^ { \prime } ( h _ { \mathrm { i n } } ) = \| W _ { a } \| _ { 2 } ( f _ { a } ^ { \top } h _ { \mathrm { i n } } ) + \lambda \| W _ { b } \| _ { 2 } ( f _ { b } ^ { \top } h _ { \mathrm { i n } } ) } \\ & { \qquad = \left( \| W _ { a } \| _ { 2 } f _ { a } + \lambda \| W _ { b } \| _ { 2 } f _ { b } \right) ^ { \top } h _ { \mathrm { i n } } . } \end{array}\tag{58}
$$

The normalized effective probing direction $f _ { a } ^ { \prime }$ realized by the updated channel is obtained by normalizing the functional operator vector:

$$
f _ { a } ^ { \prime } = \frac { \| \boldsymbol { W _ { a } } \| _ { 2 } f _ { a } + \lambda \| \boldsymbol { W _ { b } } \| _ { 2 } f _ { b } } { \| \| \boldsymbol { W _ { a } } \| _ { 2 } f _ { a } + \lambda \| \boldsymbol { W _ { b } } \| _ { 2 } f _ { b } \| _ { 2 } } \propto f _ { a } + \lambda \left( \frac { \| \boldsymbol { W _ { b } } \| _ { 2 } } { \| \boldsymbol { W _ { a } } \| _ { 2 } } \right) f _ { b } .\tag{59}
$$

Now, consider the change in mutual crosstalk between feature channel a and feature b after this update. The new inner product evaluates to:

$$
\begin{array} { l } { ( f _ { a } ^ { \prime } ) ^ { \top } f _ { b } \propto \displaystyle \left( f _ { a } + \lambda \frac { \| W _ { b } \| _ { 2 } } { \| W _ { a } \| _ { 2 } } f _ { b } \right) ^ { \top } f _ { b } } \\ { = f _ { a } ^ { \top } f _ { b } + \lambda \frac { \| W _ { b } \| _ { 2 } } { \| W _ { a } \| _ { 2 } } \| f _ { b } \| _ { 2 } ^ { 2 } = f _ { a } ^ { \top } f _ { b } + \lambda \frac { \| W _ { b } \| _ { 2 } } { \| W _ { a } \| _ { 2 } } . } \end{array}\tag{60}
$$

## B Theoretical Proofs for Section 3.2.1

## B.1 Derivation of the Induced Fisher Metric in Activation Space

Let $\mathcal { V } = \{ 1 , \ldots , V \}$ denote the output vocabulary. For a given input sequence, the pre-trained model produces final residual activation $\boldsymbol { h _ { 0 } } \in \mathbb { R } ^ { m }$ , yielding reference logits $z _ { 0 } = W _ { U } ^ { \top } h _ { 0 } + b _ { U } \in \mathbb { R } ^ { V }$ , where $\mathbf { \bar { \boldsymbol { W } } } _ { U } \in \mathbb { R } ^ { m \times V }$ and $b _ { U } \in \mathbb { R } ^ { V }$ . The reference token distribution is:

$$
\pi _ { \mathrm { r e f } } ( y = j \mid h _ { 0 } ) = { \frac { \exp ( z _ { 0 , j } ) } { \sum _ { k = 1 } ^ { V } \exp ( z _ { 0 , k } ) } } .\tag{61}
$$

Following fine-tuning, the hidden activation shifts to $h = h _ { 0 } + \Delta h$ , inducing updated logits $z =$ $W _ { U } ^ { \top } ( h _ { 0 } + \Delta h ) + b _ { U } = z _ { 0 } + \Delta z$ , where $\Delta z = W _ { U } ^ { \top } \Delta h \in \bar { \mathbb { R } ^ { V } }$ . The updated policy is $\pi _ { \theta } ( y = j { \textbf { ) } }$ $\begin{array} { r } { h ) = \frac { \exp ( z _ { j } ) } { \sum _ { k = 1 } ^ { V } \exp ( z _ { k } ) } . } \end{array}$

We expand the conditional KL divergence $\begin{array} { r } { D _ { \mathrm { K L } } ( \pi _ { \theta } \parallel \pi _ { \mathrm { r e f } } ) = \sum _ { j = 1 } ^ { V } \pi _ { \theta } ( j ) \log \frac { \pi _ { \theta } ( j ) } { \pi _ { \mathrm { r e f } } ( j ) } } \end{array}$ as a multivariate Taylor series w.r.t. logit shift $\Delta z$ around $\Delta z = 0$

$$
D _ { \mathrm { K L } } ( \pi _ { \theta } \left\| \pi _ { \mathrm { r e f } } \right) = D _ { \mathrm { K L } } { \Big | } _ { \Delta z = 0 } + \left( \nabla _ { z } D _ { \mathrm { K L } } { \Big | } _ { z = z _ { 0 } } \right) ^ { \top } \Delta z + { \frac { 1 } { 2 } } \Delta z ^ { \top } \left( \nabla _ { z } ^ { 2 } D _ { \mathrm { K L } } { \Big | } _ { z = z _ { 0 } } \right) \Delta z + { \mathcal O } ( \| \Delta z \| _ { 2 } ^ { 3 } ) .\tag{62}
$$

When $\Delta z = \mathbf { 0 } , \pi _ { \theta } = \pi _ { \mathrm { r e f } } ,$ giving $D _ { \mathrm { K L } } = 0$

First-order derivative: Using the relation $\begin{array} { r } { \frac { \partial \pi _ { \theta } ( k ) } { \partial z _ { j } } = \pi _ { \theta } ( j ) ( \delta _ { j k } - \pi _ { \theta } ( k ) ) \colon } \end{array}$

$$
\frac { \partial D _ { \mathrm { K L } } } { \partial z _ { j } } = \sum _ { k = 1 } ^ { V } \frac { \partial \pi _ { \theta } ( k ) } { \partial z _ { j } } \left( \log \pi _ { \theta } ( k ) - \log \pi _ { \mathrm { r e f } } ( k ) + 1 \right) = \pi _ { \theta } ( j ) \left( \log \frac { \pi _ { \theta } ( j ) } { \pi _ { \mathrm { r e f } } ( j ) } - D _ { \mathrm { K L } } ( \pi _ { \theta } \parallel \pi _ { \mathrm { r e f } } ) \right) .\tag{63}
$$

Evaluating at $z = z _ { 0 }$ (where $\pi _ { \theta } ( j ) = \pi _ { \mathrm { r e f } } ( j ) )$ :

$$
\left. \frac { \partial D _ { \mathrm { K L } } } { \partial z _ { j } } \right| _ { z = z _ { 0 } } = \pi _ { \mathrm { r e f } } ( j ) \left( \log 1 - 0 \right) = 0 \left. \implies \nabla _ { z } D _ { \mathrm { K L } } \right| _ { z = z _ { 0 } } = \mathbf { 0 } .\tag{64}
$$

Second-order derivative (Hessian): Differentiating once more w.r.t. z<sub>l</sub>:

$$
\frac { \partial ^ { 2 } D _ { \mathrm { K L } } } { \partial z _ { l } \partial z _ { j } } = \frac { \partial \pi _ { \theta } ( j ) } { \partial z _ { l } } \left( \log \frac { \pi _ { \theta } ( j ) } { \pi _ { \mathrm { r e f } } ( j ) } - D _ { \mathrm { K L } } \right) + \pi _ { \theta } ( j ) \left( \frac { 1 } { \pi _ { \theta } ( j ) } \frac { \partial \pi _ { \theta } ( j ) } { \partial z _ { l } } - \frac { \partial D _ { \mathrm { K L } } } { \partial z _ { l } } \right) .\tag{65}
$$

Evaluating at $z = z _ { 0 }$ , the first term vanishes because log $\begin{array} { r } { \frac { \pi _ { \theta } ( j ) } { \pi _ { \mathrm { r e f } } ( j ) } = 0 } \end{array}$ and $D _ { \mathrm { K L } } = 0$ . The second term simplifies to:

$$
\frac { \partial ^ { 2 } D _ { \mathrm { K L } } } { \partial z _ { l } \partial z _ { j } } \bigg | _ { z = z _ { 0 } } = \left. \frac { \partial \pi _ { \theta } ( j ) } { \partial z _ { l } } \right| _ { z = z _ { 0 } } = \pi _ { \mathrm { r e f } } ( j ) \delta _ { j l } - \pi _ { \mathrm { r e f } } ( j ) \pi _ { \mathrm { r e f } } ( l ) .\tag{66}
$$

In matrix notation, the Hessian matches the Fisher information matrix $F \in \mathbb { R } ^ { V \times V }$ of the categorical distribution:

$$
\left. F = \nabla _ { z } ^ { 2 } D _ { \mathrm { K L } } \right| _ { z = z _ { 0 } } = \mathrm { d i a g } ( \pi _ { \mathrm { r e f } } ) - \pi _ { \mathrm { r e f } } \pi _ { \mathrm { r e f } } ^ { \top } .\tag{67}
$$

Substituting $\Delta z = W _ { U } ^ { \top } \Delta h$ into the quadratic approximation:

$$
D _ { \mathrm { K L } } ( \pi _ { \theta } \parallel \pi _ { \mathrm { r e f } } ) \approx \frac { 1 } { 2 } \Delta z ^ { \top } F \Delta z = \frac { 1 } { 2 } \Delta h ^ { \top } \left( W _ { U } F W _ { U } ^ { \top } \right) \Delta h = \frac { 1 } { 2 } \Delta h ^ { \top } H _ { \mathrm { F i s h e r } } \Delta h ,\tag{68}
$$

where $H _ { \mathrm { F i s h e r } } = W _ { U } \left( \mathrm { d i a g } ( \pi _ { \mathrm { r e f } } ) - \pi _ { \mathrm { r e f } } \pi _ { \mathrm { r e f } } ^ { \top } \right) W _ { U } ^ { \top } \in \mathbb { R } ^ { m \times m }$ is symmetric positive semi-definite. ■

## B.2 Proof of Theorem 1: Planar LVD Margin Protection

Theorem 1 (LVD Safe Margin Lower Bound under KL Regularization). Let $\boldsymbol { h _ { 0 } } ~ \in ~ \mathbb { R } ^ { m }$ be an activation classified into token $j ^ { * } = \arg \operatorname* { m a x } _ { j \in \mathcal { V } } z _ { 0 , j }$ with baseline margin $d _ { n , k } ^ { ( 0 ) } > 0$ relative to competing candidate $k \neq j ^ { * }$ . If the policy shift induces activation displacement $\Delta h ,$ , the updated margin $d _ { n , k } ^ { ( n e w ) }$ is lower-bounded by:

$$
d _ { n , k } ^ { ( n e w ) } \geq d _ { n , k } ^ { ( 0 ) } - C _ { k } \sqrt { 2 D _ { K L } ( \pi _ { \theta } \parallel \pi _ { r e f } ) } ,\tag{69}
$$

where $C _ { k } = \frac { \| W _ { U , k } - W _ { U , j ^ { * } } \| _ { H _ { F i s h e r } ^ { \dagger } } } { \| W _ { U , k } - W _ { U , j ^ { * } } \| _ { 2 } }$ . Consequently, $d _ { n , k } ^ { ( n e w ) } > 0$ for all $k \neq j ^ { * }$ provided $D _ { K L } ( \pi _ { \theta } \| \pi _ { r e f } ) <$ min $\imath _ { k \neq j ^ { * } } ~ \frac { ( d _ { n , k } ^ { ( 0 ) } ) ^ { 2 } } { 2 C _ { k } ^ { 2 } }$

Proof. By Definition 3, the updated margin between reference token $j ^ { * }$ and competitor k under $h _ { \mathrm { n e w } } = h _ { 0 } + \Delta h$ is:

$$
\begin{array} { r l } & { d _ { n , k } ^ { \mathrm { ( n e w ) } } = \frac { \left( W _ { U , j ^ { * } } - W _ { U , k } \right) ^ { \top } \left( h _ { 0 } + \Delta h \right) + \left( b _ { U , j ^ { * } } - b _ { U , k } \right) } { \left\| W _ { U , j ^ { * } } - W _ { U , k } \right\| _ { 2 } } } \\ & { ~ = d _ { n , k } ^ { \mathrm { ( 0 ) } } - \frac { \left( W _ { U , k } - W _ { U , j ^ { * } } \right) ^ { \top } \Delta h } { \left\| W _ { U , k } - W _ { U , j ^ { * } } \right\| _ { 2 } } . } \end{array}\tag{70}
$$

To bound the maximum erosion $\begin{array} { r } { \Delta d _ { n , k } \ = \ \frac { ( W _ { U , k } - W _ { U , j ^ { * } } ) ^ { \top } \Delta h } { \| W _ { U , k } - W _ { U , j ^ { * } } \| _ { 2 } } } \end{array}$ , consider the inner product space defined by the symmetric positive semi-definite matrix $H _ { \mathrm { F i s h e r } }$ . On $\mathrm { r a n g e } ( H _ { \mathrm { F i s h e r } } )$ , we apply the generalized Cauchy-Schwarz inequality:

$$
\left| { ( W _ { U , k } - W _ { U , j ^ { * } } ) } ^ { \top } \Delta h \right| \leq \sqrt { ( W _ { U , k } - W _ { U , j ^ { * } } ) ^ { \top } H _ { \mathrm { F i s h e r } } ^ { \dagger } ( W _ { U , k } - W _ { U , j ^ { * } } ) \cdot \sqrt { \Delta h ^ { \top } H _ { \mathrm { F i s h e r } } \Delta h } } ,\tag{71}
$$

where $H _ { \mathrm { F i s h e r } } ^ { \dagger }$ denotes the Moore-Penrose pseudo-inverse. Recalling from Eq. (23) that $\Delta h ^ { \top } H _ { \mathrm { F i s h e r } } \Delta h \approx 2 D _ { \mathrm { K L } } ( \pi _ { \theta } \left. \pi _ { \mathrm { r e f } } \right)$ , we substitute this into the inequality:

$$
\big | ( W _ { U , k } - W _ { U , j ^ { * } } ) ^ { \top } \Delta h \big | \le \| W _ { U , k } - W _ { U , j ^ { * } } \| _ { H _ { \mathrm { F i s h e r } } ^ { \bot } } \cdot \sqrt { 2 D _ { \mathrm { K L } } ( \pi _ { \theta } \| \pi _ { \mathrm { r e f } } ) } .\tag{72}
$$

Dividing both sides by the Euclidean norm $\| W _ { U , k } - W _ { U , j ^ { * } } \| _ { 2 } \colon$

$$
\frac { ( W _ { U , k } - W _ { U , j ^ { * } } ) ^ { \top } \Delta h } { \| W _ { U , k } - W _ { U , j ^ { * } } \| _ { 2 } } \le \frac { \| W _ { U , k } - W _ { U , j ^ { * } } \| _ { H _ { \mathrm { F i s b e t } } ^ { \dag } } } { \| W _ { U , k } - W _ { U , j ^ { * } } \| _ { 2 } } \sqrt { 2 D _ { \mathrm { K L } } ( \pi _ { \theta } \| \pi _ { \mathrm { r e f } } ) } = C _ { k } \sqrt { 2 D _ { \mathrm { K L } } ( \pi _ { \theta } \| \pi _ { \mathrm { r e f } } ) } .\tag{73}
$$

Substituting this upper bound into Eq. (70) yields the desired lower bound:

$$
d _ { n , k } ^ { ( \mathrm { n e w } ) } \geq d _ { n , k } ^ { ( 0 ) } - C _ { k } \sqrt { 2 D _ { \mathrm { K L } } ( \pi _ { \theta } \left. \pi _ { \mathrm { r e f } } \right) } .\tag{74}
$$

To ensure that $d _ { n , k } ^ { \mathrm { ( n e w ) } } > 0$ strictly across all competing tokens $k \in \mathcal { V } \setminus \{ j ^ { * } \}$ }:

$$
d _ { n , k } ^ { ( 0 ) } - C _ { k } \sqrt { 2 D _ { \mathrm { K L } } } > 0 \Longleftrightarrow \sqrt { 2 D _ { \mathrm { K L } } } < \frac { d _ { n , k } ^ { ( 0 ) } } { C _ { k } } \Longleftrightarrow D _ { \mathrm { K L } } ( \pi _ { \theta } \| \pi _ { \mathrm { r e f } } ) < \frac { ( d _ { n , k } ^ { ( 0 ) } ) ^ { 2 } } { 2 C _ { k } ^ { 2 } } .\tag{75}
$$

Taking the minimum over all competing candidates $k \neq j ^ { * }$ establishes the condition in Eq. (25).   
Under this threshold, $h _ { \mathrm { n e w } } \in \mathrm { C e l l } _ { j ^ { * } } ^ { \mathrm { L V D } }$ holds unconditionally, precluding token prediction flips.

## C Theoretical Limitations of Existing Post-Training Paradigms

## C.1 Proof of Gradient Deadlock in RDA

Proof. The composite optimization objective w.r.t. the output logit vector $z \in \mathbb { R } ^ { V }$ is formulated as:

$$
\mathcal { L } _ { \mathrm { r o t a l } } ( z ) = \mathcal { L } _ { \mathrm { S F T } } ( z ) + \beta D _ { \mathrm { K L } } ( \pi _ { \theta } \parallel \pi _ { \mathrm { r e f } } ) = - z _ { y ^ { * } } + \log \left( \sum _ { k = 1 } ^ { V } \exp ( z _ { k } ) \right) + \beta \sum _ { k = 1 } ^ { V } \pi _ { \theta } ( k ) \log \frac { \pi _ { \theta } ( k ) } { \pi _ { \mathrm { r e f } } ( k ) } .\tag{76}
$$

We evaluate the analytical gradients of each component w.r.t. an arbitrary logit coordinate $z _ { j }$ $( j \in \{ 1 , \ldots , V \} )$ .

Recall the standard derivative of the categorical softmax distribution:

$$
\frac { \partial \pi _ { \theta } ( k ) } { \partial z _ { j } } = \pi _ { \theta } ( j ) \left( \delta _ { j k } - \pi _ { \theta } ( k ) \right) ,\tag{77}
$$

where $\delta _ { j k }$ is the Kronecker delta. Differentiating the cross-entropy term yields:

$$
\frac { \partial \mathcal { L } _ { \mathrm { S F T } } } { \partial z _ { j } } = - \frac { \partial z _ { y ^ { * } } } { \partial z _ { j } } + \frac { 1 } { \sum _ { k = 1 } ^ { V } \exp ( z _ { k } ) } \frac { \partial } { \partial z _ { j } } \left( \sum _ { k = 1 } ^ { V } \exp ( z _ { k } ) \right) = - \delta _ { j , y ^ { * } } + \pi _ { \theta } ( j ) .\tag{78}
$$

In vector notation over the full vocabulary $\nu ,$ the gradient vector is given by:

$$
\nabla _ { z } \mathcal { L } _ { \mathrm { S F T } } = \pi _ { \theta } - \mathbf { e } _ { y ^ { * } } .\tag{79}
$$

Applying the chain rule to the KL divergence:

$$
\begin{array} { r l r } & { } & { \displaystyle { \frac { \partial } { \partial z _ { j } } } D _ { \mathrm { K L } } ( \pi _ { \theta } \| \pi _ { \mathrm { r e f } } ) = \sum _ { k = 1 } ^ { V } \frac { \partial \pi _ { \theta } ( k ) } { \partial z _ { j } } \log \frac { \pi _ { \theta } ( k ) } { \pi _ { \mathrm { r e f } } ( k ) } + \sum _ { k = 1 } ^ { V } \pi _ { \theta } ( k ) \frac { \partial } { \partial z _ { j } } \left( \log \pi _ { \theta } ( k ) - \log \pi _ { \mathrm { r e f } } ( k ) \right) } \\ & { } & { \displaystyle { = \sum _ { k = 1 } ^ { V } \pi _ { \theta } ( j ) ( \delta _ { j k } - \pi _ { \theta } ( k ) ) \log \frac { \pi _ { \theta } ( k ) } { \pi _ { \mathrm { r e f } } ( k ) } + \sum _ { k = 1 } ^ { V } \pi _ { \theta } ( k ) \left( \frac { 1 } { \pi _ { \theta } ( k ) } \frac { \partial \pi _ { \theta } ( k ) } { \partial z _ { j } } \right) } } \\ & { } & { \displaystyle { = \pi _ { \theta } ( j ) \left( \log \frac { \pi _ { \theta } ( j ) } { \pi _ { \mathrm { r e f } } ( j ) } - \sum _ { k = 1 } ^ { V } \pi _ { \theta } ( k ) \log \frac { \pi _ { \theta } ( k ) } { \pi _ { \mathrm { r e f } } ( k ) } \right) + \frac { \partial } { \partial z _ { j } } \left( \sum _ { k = 1 } ^ { V } \pi _ { \theta } ( k ) \right) } } \end{array}\tag{80}
$$

Because the policy distribution is normalized on the probability simplex $\begin{array} { r } { ( \sum _ { k = 1 } ^ { V } \pi _ { \theta } ( k ) \equiv 1 ) } \end{array}$ , the second term vanishes identically: $\begin{array} { r } { \frac { \partial } { \partial z _ { j } } ( 1 ) = 0 } \end{array}$ . Under the canonical Fisher metric geometry (or considering the first-order Taylor expansion around the reference manifold where log $\frac { \pi _ { \theta } ( k ) } { \pi _ { \mathrm { r e f } } ( k ) }$ ≈ $\frac { \pi _ { \theta } ( k ) - \pi _ { \mathrm { r e f } } ( k ) } { \pi _ { \mathrm { r e f } } ( k ) } )$ , the natural unconstrained gradient vector across logits evaluates directly to:

$$
\nabla _ { z } \left[ \beta D _ { \mathrm { K L } } ( \pi _ { \theta } \parallel \pi _ { \mathrm { r e f } } ) \right] = \beta ( \pi _ { \theta } - \pi _ { \mathrm { r e f } } ) .\tag{81}
$$

Combining Eq. (79) and Eq. (81), the total gradient acting on the logit vector z is:

$$
\begin{array} { r } { \nabla _ { z } \mathcal { L } _ { \mathrm { t o t a l } } = ( \pi _ { \theta } - \mathbf { e } _ { y ^ { * } } ) + \beta ( \pi _ { \theta } - \pi _ { \mathrm { r e f } } ) = ( 1 + \beta ) \pi _ { \theta } - \mathbf { e } _ { y ^ { * } } - \beta \pi _ { \mathrm { r e f } } . } \end{array}\tag{82}
$$

Setting the stationary condition $\nabla _ { z } { \mathcal { L } } _ { \mathrm { t o t a l } } = \mathbf { 0 }$ yields:

$$
( 1 + \beta ) \pi _ { \theta } ^ { * } = { \bf e } _ { y ^ { * } } + \beta \pi _ { \mathrm { r e f } } \implies \pi _ { \theta } ^ { * } = \frac { 1 } { 1 + \beta } { \bf e } _ { y ^ { * } } + \frac { \beta } { 1 + \beta } \pi _ { \mathrm { r e f } } ,\tag{83}
$$

which establishes Eq. (26).

## C.2 Proof of Sample Complexity Lower Bound in DMA

In this subsection, we establish the asymptotic sample complexity lower bound governing DMA, proving Eq. (27) in Section 3.2.

Proof. We structure the proof in four sequential stages: (1) formalizing the non-activation probability of individual sparse features; (2) mapping feature activation under the 1-sparse capacity constraint to a generalized coupon collector process; (3) deriving the extremal probability lower bound; and (4) analyzing the asymptotic sample complexity.

1. Single-Feature Non-Activation Probability. Let $X _ { i , t } = \mathbb { I } ( a _ { i , t } > 0 ) \in \{ 0 , 1 \}$ be the indicator variable denoting whether latent feature i is activated at token position $t \in \{ 1 , \dots , N _ { \operatorname* { m i x } } \}$ . Under independent sequence sampling, $X _ { i , t } \ \sim \ \mathrm { B e r n o u l l i } ( p _ { i } )$ , where $p _ { i } \ = \ 1 - \ S _ { i } .$ The cumulative activation count of feature i across the mixture dataset $\mathcal { D } _ { \mathrm { m i x } }$ is given by the binomial random variable $\begin{array} { r } { C _ { i } = \sum _ { t = 1 } ^ { N _ { \mathrm { m i x } } } X _ { i , t } . } \end{array}$

The probability that feature i is never activated throughout the entire fine-tuning duration is:

$$
P ( C _ { i } = 0 ) = ( 1 - p _ { i } ) ^ { N _ { \mathrm { m i x } } } .\tag{84}
$$

Taking the natural logarithm and applying the first-order Taylor expansion $\ln ( 1 - x ) = - x - \mathcal { O } ( x ^ { 2 } )$ for $x \in ( 0 , 1 )$

$$
\ln P ( C _ { i } = 0 ) = N _ { \mathrm { m i x } } \ln ( 1 - p _ { i } ) = - N _ { \mathrm { m i x } } p _ { i } - \mathcal { O } ( N _ { \mathrm { m i x } } p _ { i } ^ { 2 } ) .\tag{85}
$$

Because features in superposition are ultra-sparse $( S _ { i } \to 1 \implies p _ { i } \ll 1 )$ , higher-order terms vanish, yielding:

$$
P ( C _ { i } = 0 ) \approx \exp \left( - N _ { \mathrm { m i x } } p _ { i } \right) = \exp \left( - N _ { \mathrm { m i x } } ( 1 - S _ { i } ) \right) .\tag{86}
$$

When $N _ { \operatorname* { m i x } } \lvert 1 - S _ { i } \rangle < 1 , \exp ( - N _ { \operatorname* { m i x } } \lvert 1 - S _ { i } \rangle ) \approx 1 - N _ { \operatorname* { m i x } } ( 1 - S _ { i } )  1$ , indicating that long-tail features are omitted with near certainty in finite datasets. If $C _ { i } = 0$ , feature channel $\bar { W _ { i } }$ receives zero supervisory gradient during fine-tuning, leaving it entirely unprotected against lateral representation drift.

2. Mapping to the Non-Uniform Coupon Collector’s Problem. To prevent geometric phase collapse $( L _ { 1 }  L _ { 2 } )$ across all features and maintain the stability of the entire feature dictionary $\mathcal { F }$ every single feature $i \in \{ 1 , \ldots , k \}$ must be sampled at least once in $\mathcal { D } _ { \operatorname* { m i x } }$ . The successful coverage event is defined as:

$$
\mathcal { E } _ { \mathrm { c o v } } = \bigcap _ { i = 1 } ^ { k } \{ C _ { i } \geq 1 \} .\tag{87}
$$

Correspondingly, the failure event where at least one latent feature remains completely unactivated is:

$$
\mathcal { E } _ { \mathrm { f a i l } } = \bigcup _ { i = 1 } ^ { k } \{ C _ { i } = 0 \} .\tag{88}
$$

In a Transformer residual stream of dimension m, the model operates under feature superposition where the number of active features per token is strictly bounded to prevent destructive crosstalk. Under the pre-training 1-sparse regime $( L _ { 1 } )$ , the total expected active feature mass across all k features per token is bounded:

$$
\mathbb { E } \left[ \Vert a _ { t } \Vert _ { 0 } \right] = \sum _ { j = 1 } ^ { k } p _ { j } = \sum _ { j = 1 } ^ { k } ( 1 - S _ { j } ) = k ( 1 - \bar { S } ) \leq \kappa ,\tag{89}
$$

where $\kappa = \mathcal { O } ( 1 )$ represents the average concurrent activation budget per token position, and $1 - \bar { S } =$ $\textstyle { \frac { 1 } { k } } \sum _ { j = 1 } ^ { k } ( 1 - S _ { j } )$ denotes the mean feature density.

Conditioned on an activation occurring, the relative arrival probability $q _ { i }$ of selecting feature i as an active coupon is normalized by the total feature density:

$$
q _ { i } = \frac { p _ { i } } { \sum _ { j = 1 } ^ { k } p _ { j } } = \frac { 1 - S _ { i } } { k ( 1 - \bar { S } ) } .\tag{90}
$$

The minimal arrival probability corresponds to the most sparse tail feature:

$$
q _ { \operatorname* { m i n } } = \operatorname* { m i n } _ { 1 \leq i \leq k } q _ { i } = \frac { \operatorname* { m i n } _ { i } ( 1 - S _ { i } ) } { k ( 1 - \bar { S } ) } .\tag{91}
$$

3. Extremal Probability and Lower Bound Derivation. Let $T _ { \mathrm { c o v } }$ denote the random variable representing the total number of coupon draws required to observe all k distinct features at least once. By the classical theory of the Non-Uniform Coupon Collector’s Problem [59, 45], the probability that the most difficult (least frequent) coupon has not been collected after M effective draws is lower-bounded by:

$$
P \left( \bigcup _ { i = 1 } ^ { k } \{ C _ { i } = 0 \} \right) \geq P \left( C _ { i ^ { * } } = 0 \right) = ( 1 - q _ { \operatorname* { m i n } } ) ^ { M } ,\tag{92}
$$

where $i ^ { * } = \mathrm { a r g }$ min<sub>i</sub> $q _ { i }$ .

To obtain an exact bound across the ensemble of k features, we consider the random variable $\begin{array} { r } { Z = \sum _ { i = 1 } ^ { k } \mathbb { I } ( C _ { i } = 0 ) } \end{array}$ counting the total number of uncovered features. The expectation of Z is:

$$
\mathbb { E } [ Z ] = \sum _ { i = 1 } ^ { k } ( 1 - q _ { i } ) ^ { M } \geq k ( 1 - q _ { \operatorname* { m a x } } ) ^ { M } .\tag{93}
$$

By the Paley-Zygmund inequality and the asymptotic theory of extreme value statistics (where the waiting time for the last coupon converges weakly to a Gumbel distribution), guaranteeing $P ( Z = 0 ) \geq 1 - \delta$ (or $P ( \mathcal { E } _ { \mathrm { f a i l } } ) \leq \bar { \delta } )$ requires that the failure probability upper bound satisfies:

$$
P ( \mathcal { E } _ { \mathrm { f a i l } } ) \approx 1 - \exp { \left( - \sum _ { i = 1 } ^ { k } e ^ { - M q _ { i } } \right) } \leq \delta .\tag{94}
$$

For small failure tolerance $\delta \in ( 0 , 1 )$ , using $1 - e ^ { - x }$ ≈ x:

$$
\sum _ { i = 1 } ^ { k } e ^ { - M q _ { i } } \leq \delta .\tag{95}
$$

Since $\begin{array} { r } { \sum _ { i = 1 } ^ { k } e ^ { - M q _ { i } } \geq k e ^ { - M q _ { \operatorname* { m i n } } } } \end{array}$ , a necessary condition to ensure $P ( \mathcal { E } _ { \mathrm { f a i l } } ) \leq \delta$ is:

$$
k \exp \left( - M q _ { \mathrm { m i n } } \right) \leq \delta \iff \exp \left( - M q _ { \mathrm { m i n } } \right) \leq \frac { \delta } { k } .\tag{96}
$$

Taking the natural logarithm of both sides:

$$
- M q _ { \mathrm { m i n } } \le \ln \left( \frac { \delta } { k } \right) = - \ln \left( \frac { k } { \delta } \right) \iff M \ge \frac { 1 } { q _ { \mathrm { m i n } } } \ln \left( \frac { k } { \delta } \right) .\tag{97}
$$

4. Asymptotic Sample Complexity in Token Space. Substituting the analytical expression of $q _ { \mathrm { m i n } }$ from Eq. (91) into Eq. (97):

$$
M \geq { \frac { k ( 1 - { \bar { S } } ) } { \operatorname* { m i n } _ { i } ( 1 - S _ { i } ) } } \ln \left( { \frac { k } { \delta } } \right) .\tag{98}
$$

Because the average number of active feature slots per token is $\kappa = \mathcal { O } ( 1 )$ , the total number of tokens $N _ { \mathrm { m i x } }$ scales proportionally with the number of coupon draws M $( N _ { \mathrm { m i x } } = \Theta ( M ) )$ . Furthermore, since $( 1 - \bar { S } )$ represents the baseline non-zero mean feature activation rate $( 1 - { \bar { S } } = \mathsf { \tilde { \Theta } } ( 1 )$ relative to token sequence length under 1-sparse capacity), we obtain the formal asymptotic lower bound:

$$
N _ { \mathrm { m i x } } \geq c \cdot { \frac { k \log ( k / \delta ) } { \operatorname* { m i n } _ { i } ( 1 - S _ { i } ) } } \implies N _ { \mathrm { m i x } } = \Omega \left( { \frac { k \log ( k / \delta ) } { \operatorname* { m i n } _ { i } ( 1 - S _ { i } ) } } \right) ,\tag{99}
$$

for some universal positive constant $c > 0$ . This establishes Eq. (27).

## D Proof of Corollary 1: Duality between LVD Geometry and Logit Dynamics

In this section, we present the formal proof of Corollary 1, establishing the bijective correspondence between geometric activation trajectories in planar LVD and output logit dynamics.

Proof. Let $h \in \mathrm { I n t } ( \mathrm { C e l l } _ { j } ^ { \mathrm { L V D } } )$ . By Eq. (8), the power distance to site $j$ expands as:

$$
D _ { j } ( h ) = \| h - c _ { j } \| _ { 2 } ^ { 2 } - \nu _ { j } = \| h \| _ { 2 } ^ { 2 } - z _ { j } ( h ) .\tag{100}
$$

Substituting this relation into the cell membership condition yields:

$$
\| h \| _ { 2 } ^ { 2 } - z _ { j } ( h ) < \| h \| _ { 2 } ^ { 2 } - z _ { k ^ { \prime } } ( h ) , \quad \forall k ^ { \prime } \in \mathcal { V } \setminus \{ j \} .\tag{101}
$$

Since $\| h \| _ { 2 } ^ { 2 }$ is an identical scalar across all vocabulary items for a given activation vector $h ,$ canceling $\| h \| _ { 2 } ^ { 2 }$ from both sides directly establishes:

$$
- z _ { j } ( h ) < - z _ { k ^ { \prime } } ( h ) \iff z _ { j } ( h ) > z _ { k ^ { \prime } } ( h ) , \quad \forall k ^ { \prime } \in \mathcal { V } \setminus \{ j \} .\tag{102}
$$

Hence, within the interior of $\mathrm { C e l l } _ { j } ^ { \mathrm { L V D } }$ , the logit score of token $j$ is strictly greater than that of any other competing token in $\nu$

Consider the directional trajectory $h ( t ) = h + t { \hat { u } } _ { j , k }$ with $t \geq 0$ , where the unit outward normal vector is defined as $\begin{array} { r } { \hat { u } _ { j , k } = \frac { W _ { U , k } - W _ { U , j } } { \| W _ { U , k } - W _ { U , j } \| _ { 2 } } } \end{array}$

Evaluating the difference between the logits of competitor k and reference token j along this path:

$$
\begin{array} { r l } & { z _ { k } ( h ( t ) ) - z _ { j } ( h ( t ) ) = \big ( W _ { U , k } ^ { \top } h ( t ) + b _ { U , k } \big ) - \big ( W _ { U , j } ^ { \top } h ( t ) + b _ { U , j } \big ) } \\ & { \qquad = \big ( W _ { U , k } - W _ { U , j } \big ) ^ { \top } ( h + t \hat { u } _ { j , k } ) + \big ( b _ { U , k } - b _ { U , j } \big ) } \\ & { \qquad = \big [ z _ { k } ( h ) - z _ { j } ( h ) \big ] + t \big ( W _ { U , k } - W _ { U , j } \big ) ^ { \top } \hat { u } _ { j , k } . } \end{array}\tag{103}
$$

Substituting the analytical form of $\hat { u } _ { j , k } \mathrm { : }$

$$
( W _ { U , k } - W _ { U , j } ) ^ { \top } \hat { u } _ { j , k } = \frac { \| W _ { U , k } - W _ { U , j } \| _ { 2 } ^ { 2 } } { \| W _ { U , k } - W _ { U , j } \| _ { 2 } } = \| W _ { U , k } - W _ { U , j } \| _ { 2 } .\tag{104}
$$

Therefore, the relative logit margin evolves as a strictly affine function of the displacement parameter t:

$$
z _ { k } ( h ( t ) ) - z _ { j } ( h ( t ) ) = \big [ z _ { k } ( h ) - z _ { j } ( h ) \big ] + t \| W _ { U , k } - W _ { U , j } \| _ { 2 } .\tag{105}
$$

Differentiating with respect to t:

$$
\frac { d } { d t } \big ( z _ { k } ( h ( t ) ) - z _ { j } ( h ( t ) ) \big ) = \| { \cal W } _ { U , k } - { \cal W } _ { U , j } \| _ { 2 } > 0 ,\tag{106}
$$

Because the exponential map is strictly increasing and the derivative is strictly positive, the relative probability ratio $\frac { P ( k | h ( t ) ) } { P ( j | h ( t ) ) }$ increases strictly monotonically as $h ( t )$ moves along $\hat { u } _ { j , k }$ □

## E Derivation of Dynamic Equilibrium and Crosstalk Probability under Superposition

In this subsection, we formally derive the dynamic equilibrium between feature leakage and gradient self-repulsion, establishing Eq. (40).

Proof. Let $\boldsymbol { h } \in \mathbb { R } ^ { m }$ denote the residual stream activation, and let $W _ { U } = [ W _ { U , 1 } , \dotsc , W _ { U , V } ] \in$ $\mathbb { R } ^ { m \times V }$ and $b _ { U } = [ b _ { U , 1 } , \dotsc , b _ { U , V } ] ^ { \top } \in \mathbb { R } ^ { V }$ parameterize the unembedding layer. The logit vector is $z = W _ { \scriptscriptstyle { U } } ^ { \top } h + b _ { \scriptscriptstyle { U } }$ , and the categorical probability distribution is given by the softmax function $\begin{array} { r } { p _ { i } = P ( i \bar { \mid h ) } = \frac { \exp ( z _ { i } ) } { \sum _ { l = 1 } ^ { V } \exp ( z _ { l } ) } } \end{array}$ for all $i \in \nu$

For an instance with target label $j ,$ the cross-entropy loss is $\mathcal { L } _ { \mathrm { C E } } ( h ) = - \log p _ { j }$ . Differentiating with respect to h via the chain rule yields the net descent direction:

$$
- \nabla _ { h } \mathcal { L } _ { \mathrm { C E } } = ( 1 - p _ { j } ) W _ { U , j } - \sum _ { i \neq j } p _ { i } W _ { U , i } .\tag{107}
$$

Projecting this optimization thrust onto the unembedding vector $W _ { U , k }$ of an arbitrary competing candidate $k \neq j \colon$

$$
W _ { U , k } ^ { \top } ( - \nabla _ { h } \mathcal { L } _ { \mathrm { C E } } ) = ( 1 - p _ { j } ) ( W _ { U , k } ^ { \top } W _ { U , j } ) - p _ { k } \| W _ { U , k } \| _ { 2 } ^ { 2 } - \sum _ { l \neq j , k } p _ { l } ( W _ { U , k } ^ { \top } W _ { U , l } ) .\tag{108}
$$

Under the feature superposition hypothesis (Definition 2), features activate with extreme sparsity $( S  1 )$ . The probability mass is predominantly distributed between the target and immediate competitors, rendering the collective background crosstalk over ambient tokens negligible:

$$
\sum _ { l \neq j , k } p _ { l } ( W _ { U , k } ^ { \top } W _ { U , l } ) \approx 0 .\tag{109}
$$

Substituting this into Eq. (108), the net projection along channel k simplifies to:

$$
\begin{array} { r } { W _ { U , k } ^ { \top } ( - \nabla _ { h } \mathcal { L } _ { \mathrm { C E } } ) \approx ( 1 - p _ { j } ) ( W _ { U , k } ^ { \top } W _ { U , j } ) - p _ { k } \| W _ { U , k } \| _ { 2 } ^ { 2 } . } \end{array}\tag{110}
$$

At dynamic optimization stationarity along channel $k ,$ the lateral leakage is balanced by the negative self-repulsion $( W _ { U , k } ^ { \top } ( - \nabla _ { h } \mathcal { L } _ { \mathrm { C E } } ) \approx \mathbf { \bar { 0 } } )$ :

$$
( 1 - p _ { j } ) ( W _ { U , k } ^ { \top } W _ { U , j } ) - p _ { k } ^ { * } \| W _ { U , k } \| _ { 2 } ^ { 2 } = 0 .\tag{111}
$$

Solving for the steady-state probability $p _ { k } ^ { * }$ directly yields:

$$
p _ { k } ^ { * } = ( 1 - p _ { j } ) \cdot \frac { W _ { U , k } ^ { \top } W _ { U , j } } { \| W _ { U , k } \| _ { 2 } ^ { 2 } } ,\tag{112}
$$

which establishes Eq. (40).

## F Convexity Analysis and Right-Derivative Bisection for the Shared Center $\zeta ^ { * }$

In this appendix, we provide the theoretical convexity proof of the local AEGIS objective with respect to the shared non-negative improvement scalar $\zeta ,$ establish the exact analytical formulation of its right derivative, and outline the numerical bisection procedure used to resolve $\begin{array} { r } { \zeta ^ { * } = \arg \operatorname* { m i n } _ { \zeta \ge 0 } \mathcal { L } _ { \mathrm { A E G I S } } ( \zeta ) } \end{array}$

## F.1 Convexity and Monotonicity Proof

Proof. We structure the proof in three sequential steps.

Define the candidate-wise affine argument $\psi _ { k } ( \zeta ) = \zeta - \Delta \rho _ { j , k }$ . The absolute deviation $| \Delta \rho _ { j , k } - \zeta | =$ $| \psi _ { k } ( \zeta ) |$ is the composition of the convex absolute-value function with an affine mapping, and is therefore convex in $\zeta .$ Subtracting the constant bandwidth $w _ { j , k }$ yields the convex excess function $E _ { k } ( \zeta ) = | \zeta - \Delta \rho _ { j , k } | - w _ { j , k }$ . Next, let $F _ { k } = \operatorname* { m a x } ( - \rho _ { j , k } , 0 ) \overset { \cdot } { \geq } 0$ denote the non-negative crossing threshold, which is independent of $\zeta , D _ { j , k } ( \zeta )$ can be rewritten as:

$$
D _ { j , k } ( \zeta ) = \operatorname* { m a x } \big ( E _ { k } ( \zeta ) , F _ { k } \big ) .\tag{113}
$$

Since the pointwise maximum of convex functions is convex, $D _ { j , k } ( \zeta )$ is unconditionally convex on R, and consequently convex on the non-negative domain $[ 0 , + \infty )$

Evaluating the first and second derivatives of the Huber operator $\ell _ { \kappa } ( D )$ on $D \in [ 0 , + \infty )$

$$
\ell _ { \kappa } ^ { \prime } ( D ) = \operatorname* { m i n } \left( \frac { D } { \kappa } , 1 \right) \geq 0 , \qquad \ell _ { \kappa } ^ { \prime \prime } ( D ) = \left\{ \frac { 1 } { \kappa } > 0 , 0 \leq D < \kappa , \right.\tag{114}
$$

Hence, $\ell _ { \kappa } ( D )$ is non-decreasing $( \ell _ { \kappa } ^ { \prime } ( D ) \geq 0 )$ and convex $( \ell _ { \kappa } ^ { \prime \prime } ( D ) \geq 0 ) \mathrm { o n } [ 0 , + \infty )$ . By the standard scalar composition rule in convex analysis, if $g : \mathbb { R } $ R is convex and non-decreasing, and $f : \mathbb { R } \to \mathbb { R }$ is convex, then the composite function $g ( f ( \zeta ) )$ is convex. Since $D _ { j , k } ( \zeta ) \geq 0$ is convex and $\ell _ { \kappa } ( \cdot )$ is convex and non-decreasing on $[ 0 , + \infty )$ , each composite component $\ell _ { \kappa } ( D _ { j , k } ( \zeta ) )$ is convex with respect to $\zeta .$ Finally, because $\mathcal { L } _ { \mathrm { A E G I S } } ( \zeta )$ is a non-negative weighted average of convex functions, $\mathcal { L } _ { \mathrm { A E G I S } } ( \zeta )$ is convex on $[ 0 , + \infty )$

For a convex function on $\mathbb { R } ^ { + }$ , the one-sided right derivative $\begin{array} { r } { D ^ { + } f ( \zeta ) = \operatorname* { l i m } _ { \epsilon \to 0 ^ { + } } \frac { f ( \zeta + \epsilon ) - f ( \zeta ) } { \epsilon } } \end{array}$ exists everywhere. Applying the one-sided chain rule to $\ell _ { \kappa } ( D _ { j , k } ( \zeta ) )$ :

$$
D ^ { + } \left[ \ell _ { \kappa } ( D _ { j , k } ( \zeta ) ) \right] = \ell _ { \kappa } ^ { \prime } ( D _ { j , k } ( \zeta ) ) \cdot D ^ { + } D _ { j , k } ( \zeta ) = \operatorname* { m i n } \left( \frac { D _ { j , k } ( \zeta ) } { \kappa } , 1 \right) \cdot D ^ { + } D _ { j , k } ( \zeta ) .\tag{115}
$$

We evaluate the right derivative of $D _ { j , k } ( \zeta ) = \operatorname* { m a x } \left( | \zeta - \Delta \rho _ { j , k } | - w _ { j , k } , F _ { k } \right)$

• Case 1: Forward Outward Drift $( \zeta \geq \Delta \rho _ { j , k } )$ . Here $| \zeta - \Delta \rho _ { j , k } | = \zeta - \Delta \rho _ { j , k }$ , giving $D ^ { + } E _ { k } ( \zeta ) = + 1$ . If $E _ { k } ( \zeta ) \geq \dot { F } _ { k }$ , increasing ζ strictly drives $\bar { D } _ { j , k } ( \zeta )$ along the outward slope +1. Thus, $D ^ { + } D _ { j , k } ( \zeta ) = + 1$

• Case 2: Backward Inward Drift $( \zeta < \Delta \rho _ { j , k } ) .$ . Here $| \zeta - \Delta \rho _ { j , k } | = \Delta \rho _ { j , k } - \zeta ,$ giving $D ^ { + } E _ { k } ( \zeta ) = - 1$ . If $E _ { k } ( \zeta ) > ^ { \bar { F } _ { k } }$ , increasing ζ contracts the excess distance along the inward slope −1, giving $\dot { D } ^ { + } D _ { j , k } ( \zeta ) = - 1$

• Case 3: Invariant Region. In all other cases—including within the tolerance band $( E _ { k } ( \zeta ) < F _ { k } )$ or where the static crossing floor dominates $( E _ { k } \bar { ( \zeta ) } = F _ { k }$ with slope −1)—an infinitesimal step $\epsilon > 0$ leaves the distance clamped at $F _ { k } ,$ yielding $D ^ { + } D _ { j , k } ( \zeta ) = 0$

Combining these branches, the directional sensitivity of candidate k evaluates to:

$$
g _ { k } ( \zeta ) = D ^ { + } \big [ \ell _ { \kappa } ( D _ { j , k } ( \zeta ) ) \big ] = \left\{ \begin{array} { l l } { + \operatorname* { m i n } \Big ( \frac { D _ { j , k } ( \zeta ) } { \kappa } , 1 \Big ) , } & { \mathrm { i f ~ } \zeta \ge \Delta \rho _ { j , k } \mathrm { ~ a n d ~ } E _ { k } ( \zeta ) \ge F _ { k } , } \\ { - \operatorname* { m i n } \Big ( \frac { D _ { j , k } ( \zeta ) } { \kappa } , 1 \Big ) , } & { \mathrm { i f ~ } \zeta < \Delta \rho _ { j , k } \mathrm { ~ a n d ~ } E _ { k } ( \zeta ) > F _ { k } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{116}
$$

The total right derivative of the objective is the linear sum:

$$
D ^ { + } \mathcal { L } _ { \mathrm { A E G I S } } ( \zeta ) = \frac { 1 } { | \mathcal { V } | - 1 } \sum _ { k \neq j } g _ { k } ( \zeta ) .\tag{117}
$$

Because $\mathcal { L } _ { \mathrm { A E G I S } } ( \zeta )$ is convex on $\lbrack 0 , + \infty )$ , its right derivative $D ^ { + } { \mathcal { L } } _ { \mathrm { A E G I S } } ( \zeta )$ is monotonically nondecreasing: for any $0 \leq \zeta _ { 1 } < \zeta _ { 2 }$ , we have $D ^ { + } \mathcal { L } _ { \mathrm { A E G I S } } ^ { - } ( \zeta _ { 1 } ) \leq D ^ { + } \mathcal { L } _ { \mathrm { A E G I S } } ( \zeta _ { 2 } )$ □

## F.2 Exact Numerical Resolution via Bisection

By the first-order optimality condition for constrained convex optimization, the global minimum $\zeta ^ { * } \in [ 0 , + \infty )$ satisfies:

$$
\zeta ^ { * } = 0 \longleftrightarrow D ^ { + } \mathcal { L } _ { \mathrm { { A E G I S } } } ( 0 ) \geq 0 , \qquad \mathrm { a n d } \qquad \zeta ^ { * } > 0 \longleftrightarrow 0 \in \partial \mathcal { L } _ { \mathrm { { A E G I S } } } ( \zeta ^ { * } ) ,\tag{118}
$$

where ∂ denotes the subdifferential. Due to the strict monotonicity established in Theorem $\mathbf { ? ? } , \zeta ^ { * }$ can be located via numerical bisection on $D ^ { + } { \mathcal { L } } _ { \mathrm { A E G I S } } ( \zeta )$

We establish a tight analytical bound $\zeta _ { \mathrm { m a x } }$ . For any candidate k, setting $\zeta \geq \Delta \rho _ { j , k } + w _ { j , k } + F _ { k }$ ensures that:

$$
\zeta - \Delta \rho _ { j , k } \geq w _ { j , k } + F _ { k } \iff E _ { k } ( \zeta ) \geq F _ { k } .\tag{119}
$$

Under this condition, candidate k falls exclusively into Case $1 ( \zeta \geq \Delta \rho _ { j , k } )$ , producing a non-negative directional slope $( g _ { k } ( \zeta ) \geq 0 )$ . Therefore, defining the upper search boundary as:

$$
\zeta _ { \operatorname* { m a x } } = \operatorname* { m a x } \Big ( 0 , \operatorname* { m a x } _ { k \in \mathcal { V } \backslash \{ j \} } \big ( \Delta \rho _ { j , k } + w _ { j , k } + \operatorname* { m a x } ( - \rho _ { j , k } , 0 ) \big ) \Big ) ,\tag{120}
$$

unconditionally guarantees $D ^ { + } \mathcal { L } _ { \mathrm { A E G I S } } ( \zeta _ { \mathrm { m a x } } ) \geq 0$ . The global minimum $\zeta ^ { * }$ is thus guaranteed to reside in the compact interval $[ 0 , \zeta _ { \mathrm { m a x } } ]$

Algorithm 1 Exact Bisection for the Shared Improvement Scalar $\zeta ^ { * }$   
Require: Pre-computed baseline margins $\{ \rho _ { j , k } ^ { ( 0 ) } \}$ , current margins $\{ \rho _ { j , k } \}$ , tolerance widths $\{ w _ { j , k } \}$   
smoothing threshold $\kappa > 0 .$ , iteration budget $T _ { \mathrm { m a x } } = 4 8 .$   
Ensure: Optimal shared scalar $\zeta ^ { * } \in [ 0 , + \infty ) ^ { \overline { { } } }$   
1: Compute margin displacements: $\Delta \rho _ { j , k } \gets \rho _ { j , k } - \rho _ { j , k } ^ { ( 0 ) } ,$ ∀k $\in \mathcal { V } \setminus \{ j \}$   
2: Compute static crossing thresholds: $F _ { k } \gets$ max $\bar { \mathbf { \rho } } ( - \bar { \rho } _ { j , k } , 0 )$ $\forall k \in \mathcal { V } \setminus \{ j \}$   
3: Initialize search bracket: $\zeta _ { \mathrm { m i n } } \gets 0 ,$ and $\zeta _ { \mathrm { m a x } }$ via Eq. (120)   
4: if $D ^ { + } \mathcal { L } _ { \mathrm { A E G I S } } ( 0 ) \geq 0$ {Evaluated via Eq. (117)} then   
5: return $\zeta ^ { * } = 0$ {Zero improvement benchmark is globally optimal}   
6: end if   
7: for $t = 1$ to $T _ { \mathrm { m a x } }$ do   
8: $\zeta _ { \mathrm { m i d } } \gets \frac { 1 } { 2 } ( \zeta _ { \mathrm { m i n } } + \zeta _ { \mathrm { m a x } } )$   
9: Compute right derivative $g _ { \mathrm { m i d } }  D ^ { + } \mathcal { L } _ { \mathrm { A E G I S } } ( \zeta _ { \mathrm { m i d } } )$ via Eq. (117)   
10: if $g _ { \mathrm { m i d } } \geq 0$ then   
11: ζ<sub>max</sub> ← ζ<sub>mid</sub> {Target minimum lies to the left (preferring the leftmost knot)}   
12: else   
13: $\zeta _ { \mathrm { m i n } }  \zeta _ { \mathrm { m i d } }$ {Objective decreases to the right}   
14: end if   
15: end for   
16: return $\zeta ^ { * } = \zeta _ { \mathrm { m a x } }$

Convergence Rate and Plateau Disambiguation. After $T _ { \mathrm { m a x } } = 4 8$ iterations, the bracket width contracts geometrically by a factor of $2 ^ { - 4 8 } \approx 3 . 5 5 \times 1 0 ^ { - 1 5 }$ , achieving near-machine precision under 64-bit floating-point arithmetic. Crucially, when $\mathcal { L } _ { \mathrm { A E G I S } } ( \zeta )$ contains an optimal plateau interval $[ \zeta _ { A } ^ { * } , \zeta _ { B } ^ { * } ]$ where $\dot { D } ^ { + } \mathcal { L } _ { \mathrm { A E G I S } } ( \zeta ) = 0 .$ , branching left whenever $g _ { \mathrm { m i d } } \geq 0$ (Line 10 in Algorithm 1) deterministically contracts the search toward the leftmost boundary $\zeta _ { A } ^ { * }$ . This prevents numerical oscillations and provides a stable minimum for gradient backpropagation.

## G Detailed Experimental Settings

## G.1 Dataset Splits and Statistics

We summarize the statistics of all target task datasets $\mathcal { D } _ { T }$ and their evaluation benchmarks in Table G1. Since the GLUE test labels are not publicly available, for each GLUE task we use the official validation split as the evaluation set, and hold out 10% of the official training split (stratified by label, seed 0) as the validation set; for STS-B, the stratification bins the similarity score in steps of 0.5. LogiQA2 uses its official train, dev, and test splits, and GSM8K uses its official train and test splits. For Code-Bagel-Hermes and Infinity-Instruct-Code, we randomly sample 20,000 training instances with seed 0 (for Infinity-Instruct-Code, from the code subset obtained by filtering Infinity-Instruct 7M). CodeAlpaca, Code-Bagel-Hermes, Infinity-Instruct-Code, and Evol-Instruct are evaluated on external benchmarks, namely HumanEval, CodeXGLUE (Python), and AlpacaEval.

Table G2 presents the prompt template of each dataset together with an input–output case. For every task, the training label is the output token span followed by the EOS token (<|endoftext|> for Qwen and $< / \varsigma >$ for Mistral), and the prompt tokens are excluded from the loss (label set to −100). Each GLUE task uses two templates that alternate according to the parity of the sample idx; only the first one is listed.

Table G1: Statistics of the target task datasets and their evaluation benchmarks. #Samples is the total number of instances used from each dataset (training, validation, and evaluation splits combined); for datasets evaluated on an external benchmark, it equals the number of training instances. For GLUE, the evaluation set is the official validation split (validation\_matched for MNLI).
<table><tr><td>Dataset</td><td>#Samples</td><td>Evaluation</td><td>#Eval. Samples</td><td>Metric</td></tr><tr><td>CoLA</td><td>9,594</td><td>CoLA</td><td>1,043</td><td>MCC</td></tr><tr><td>SST-2</td><td>68,221</td><td>SST-2</td><td>872</td><td>Accuracy</td></tr><tr><td>MRPC</td><td>4,076</td><td>MRPC</td><td>408</td><td>Accuracy / F1</td></tr><tr><td>STS-B</td><td>7,249</td><td>STS-B</td><td>1,500</td><td>Pearson / Spearman</td></tr><tr><td>QQP</td><td>404,276</td><td>QQP</td><td>40,430</td><td>Accuracy / F1</td></tr><tr><td>MNLI</td><td>402,517</td><td>MNLI</td><td>9,815</td><td>Matched accuracy</td></tr><tr><td>QNLI</td><td>110,201</td><td>QNLI</td><td>5,463</td><td>Accuracy</td></tr><tr><td>RTE</td><td>2,767</td><td>RTE</td><td>277</td><td>Accuracy</td></tr><tr><td>GSM8K</td><td>8,792</td><td>GSM8K</td><td>1,319</td><td>Accuracy</td></tr><tr><td>LogiQA2</td><td>15,708</td><td>LogiQA2</td><td>1,572</td><td>Accuracy</td></tr><tr><td>CodeAlpaca</td><td>20,022</td><td>HumanEval</td><td>164</td><td>pass @ 1 / pass@5 / pass @ 10</td></tr><tr><td>Code-Bagel-Hermes</td><td>20,000</td><td>CodeXGLUE</td><td>50,000</td><td>Token accuracy</td></tr><tr><td>Infinity-Instruct-Code</td><td>20,000</td><td>CodeXGLUE</td><td>50,000</td><td>Token accuracy</td></tr><tr><td>Evol-Instruct</td><td>70,000</td><td>AlpacaEval</td><td>805</td><td>Win rate</td></tr></table>

Table G2: Prompt templates and input–output cases for each dataset. \n denotes a newline, a leading space that is part of the label, and {...} a field filled from the sample; bracketed parts of the Alpaca template appear only when the sample has a non-empty input. Long inputs are abridged with “. . . ”.
<table><tr><td>Dataset</td><td>Prompt Template</td><td>Input Case</td><td>Output Case (Label)</td></tr><tr><td>GSM8K</td><td>Question: {question}\nAnswer:</td><td>Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?</td><td>Natalia sold 48/2 = &lt;&lt;48/2=24&gt;&gt;24 clips in May.\nNatalia sold 48+24 = &lt;&lt;48+24=72&gt;&gt;72</td></tr><tr><td>CoLA</td><td>Sentence: "{sentence}"\nIs this sentence grammatically acceptable English? Answer acceptable or unacceptable.\nAnswer:</td><td>of the rocks.</td><td>The sailors rode the breeze clearacceptable</td></tr><tr><td></td><td>Sentence: "{sentence}"\nWhat is its sentiment?Answer positive or negative.\nSentiment:</td><td>it 's sweet and fluffy at the time, positive</td><td></td></tr><tr><td>MRPC</td><td>Sentence 1: "{s1} "\nSentence 2: "{s2}"\nDo these sentences have the same meaning?Answer yes or no.\nAnswer:</td><td>s1: Slow-moving , drenching Tropical Storm Henri doused an already soaked Florida Friday , s2: ... on Friday , pushing heavy rains into areas where lakes and rivers were full to</td><td>yes</td></tr><tr><td>STS-B</td><td>Sentence 1: "{s1}"\nSentence 2: "{s2}"\nRate their semantic similarity from 0.0 (unrelated) to 5.0</td><td>s1: A man with a hard hat is dancing. s2: A man wearing a hard hat is dancing.</td><td>5.0 (trained with the MSE of the expected value over number tokens instead of</td></tr><tr><td>QQP</td><td>Question 1: "{q1}"\nQuestion 2: "{q2}"\nAre these duplicate questions with the same intent?Answer yes or no.\nAnswer:</td><td>q1: Why are African-Americansno so beautiful? q2: Why are hispanics so beautiful?</td><td></td></tr><tr><td></td><td>Premise: "{premise}"\nHypothesis: "{hypothesis}"\nClassify the relationship as entailment, neutral, or contradiction.\nAnswer:</td><td>Premise: Paying lawyers for proneutral bono work is out of the question. Hypothesis: We have something we can take advantage of here.</td><td></td></tr><tr><td></td><td>Question: "{question} "\nSentence: "{sentence}"\nDoes the sentence contain an answer to the question? Answer yes or no.\nAnswer:</td><td>Question: What new technologies in particular were courted by Moltke? Sentence: Moltke embraced new technology, particularly the railroad and telegraph, . ..</td><td></td></tr><tr><td>RTE</td><td>Given the premise: "{s1}"\nDoes this imply that "{s2}"? Please answer yes or no.\nAnswer:</td><td>s1: Dana Reeve, the widow of the actor Christopher Reeve, has died of lung cancer at age 44, s2: Christopher Reeve had an accident.</td><td>no</td></tr><tr><td></td><td>Passage: {text}\nQuestion: {question}\nA. {o1}\nB. {o2}\nC. {o3}\nD. {o4}\nAnswer:</td><td>Passage: One seminar had 18 participants. It is known that (1) At least 5 young teachers are female; ... Question: ... which of the following can be concluded . .. ? (four options A-D)</td><td>_D</td></tr><tr><td>/ Code- Bagel- Hermes</td><td>CodeAlpaca Below is an instruction that describes a task[, paired with an input ...]. between 1 and 10. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n[### Input:\n{input}\n\n] ###</td><td>Create an array of length 5 which contains all even numbers 6, 8, 10]</td><td></td></tr><tr><td>Infinity- Instruct- Code</td><td>Query:\n{user}, Response:\n{assistant}, or System:\n{system}; turns are separated by a blank line, followed by Response:\n</td><td>and answer this: While Fargesia family is a genus of Asian bamboo . . . , what family is Diphylleia in?</td><td>Query: Question: Combine facts Berberidaceae</td></tr><tr><td>Evol- Instruct</td><td>{instruction}\n\n### Response:</td><td>Identify the most suitable adverb for the following sentence\nShe works</td><td>hard.</td></tr><tr><td></td><td>HumanEval Original problem prompt (function signature with docstring), with leading and trailing whitespace stripped</td><td>def has_close_elements (numkselution (used only List[float], threshold: float) -&gt; bool: """ Check if in given list ...""" cases)</td><td>Official canonical for loss computation; scoring is based on executing the test</td></tr><tr><td></td><td>CodeXGLUE Official code tokens joined by spaces</td><td>&lt;s&gt; from django . utils . translation import ugettext_lazy as _ &lt;EOL&gt; ...</td><td>Every official token is predicted; scored by token accuracy</td></tr><tr><td></td><td>AlpacaEval A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {instruction} ASSISTANT:</td><td>What are the names of some famous actors that started their careers on Broadway?</td><td>Official reference answer; the model output is compared against it by the judge</td></tr></table>

## G.2 Pervasive Dataset Sampling Algorithm

To construct the pervasive dataset $\mathcal { D } _ { P }$ , we design a multi-pass task-permuted sampling algorithm tailored to achieve two primary objectives: (1) preserving broad task diversity across heterogeneous domains, and (2) maximizing ground-truth label-token coverage across the output vocabulary V. Specifically, we first assemble a comprehensive multi-task candidate pool $\mathcal { D } _ { \mathrm { p o o l } } = \mathbf { \bar { \{ } } \mathcal { T } _ { 1 } ,  \mathcal { T } _ { 2 } , \dots , \mathcal { T } _ { M } \}$ from: FLAN 2021 [71], T0 [56], NIV2 [69], CoT [8]. Let $\mathcal { V } _ { \mathrm { c o v } } \subset \mathcal { V }$ track the set of vocabulary tokens covered by the ground-truth labels in $\mathcal { D } _ { P }$ (initialized as $\nu _ { \mathrm { c o v } } = \emptyset )$ . In each traversal pass, we randomly permute the order of all remaining non-empty tasks. Starting from the first task in the permuted sequence, we draw a random unexamined sample $( x , y )$ from the current task without replacement and inspect its ground-truth label token set $\mathcal { V } ( y )$ . If $\nu ( y )$ contains at least one nove token not yet present in $\mathcal { V } _ { \mathrm { c o v } } \ ( \mathrm { i . e . , } \ \mathcal { V } ( y ) \setminus \mathcal { V } _ { \mathrm { c o v } } \ \ne \ \emptyset )$ , we append $( x , y )$ to $\mathcal { D } _ { P }$ , update the covered vocabulary via $\mathcal { V } _ { \mathrm { c o v } }  \mathcal { V } _ { \mathrm { c o v } } \cup \mathcal { V } ( y )$ , and immediately advance to the next task. Conversely, if $\mathcal { V } ( y )$ contains no new label tokens, we repeatedly draw another random unexamined sample from the same task until either a sample introducing a novel label token is found or all samples within that task are exhausted, after which we proceed to the next task. Once a full pass over all active tasks is completed, we initiate a subsequent pass over the remaining unexamined samples across non-exhausted tasks, repeating this round-robin process until all candidate samples in $\mathcal { D } _ { \mathrm { p o o l } }$ have been searched (or full vocabulary coverage $\dot { \nu _ { \mathrm { c o v } } } = \nu$ is achieved). The formal procedure is summarized in Algorithm 2.

After applying Algorithm 2, the resulting pervasive dataset $\mathcal { D } _ { P }$ contains 12,110 samples with 551,944 label tokens for Mistral-7B, of which 27,943 are distinct, covering 85.3% of its 32,768-token vocabulary. Since Qwen2.5-3B and Qwen2.5-7B share the same tokenizer, they share an identical $\mathcal { D } _ { P }$ of 31,931 samples with 1,281,820 label tokens, of which 77,031 are distinct, covering 50.8% of the 151,665-token tokenizer vocabulary (50.7% relative to the embedding sizes of 151,936 and 152,064, respectively). On $\mathcal { D } _ { P } .$ , the base models $M _ { \theta _ { 0 } }$ achieve an average accuracy of 66.4% for Mistral-7B, 62.2% for Qwen2.5-3B, and 65.2% for Qwen2.5-7B (62.9%, 59.5%, and 62.4% when averaged over samples).

## G.3 Baseline Descriptions and Implementation Details

Baseline Descriptions. We compare AEGIS against six representative baselines corresponding to the six PCD-mitigation paradigms established in Section 3.2:

Algorithm 2 Task-Permuted Vocabulary-Covering Sampling for Pervasive Dataset $\mathcal { D } _ { P }$   
Require: Multi-task corpus pool $\mathcal { D } _ { \mathrm { p o o l } } = \{ \mathcal { T } _ { 1 } , \mathcal { T } _ { 2 } , \dotsc , \mathcal { T } _ { M } \}$ compiled from FLAN 2021, T0, NIV2,   
CoT, and Dialog; output vocabulary V.   
Ensure: Sampled pervasive dataset $\mathcal { D } _ { P } .$   
1: Initialize pervasive dataset $\mathcal { D } _ { P }  \emptyset$ and covered label-token set $\nu _ { \mathrm { c o v } }  \emptyset$   
2: Initialize active task index set $\mathcal { T } _ { \mathrm { a c t i v e } } \gets \{ 1 , 2 , \dots , M \}$   
3: while $\mathcal { T } _ { \mathrm { a c t i v e } } \neq \emptyset$ and $\mathcal { V } _ { \mathrm { c o v } } \neq \mathcal { V }$ do   
4: Generate a random permutation π $\cdot ( \mathcal { T } _ { \mathrm { a c t i v e } } )$ of the active tasks   
5: for each task index m $\in \pi ( \mathcal { T } _ { \mathrm { a c t i v e } } )$ do   
6: found\_new\_token ← false   
7: while $\mathcal { T } _ { m } \neq \emptyset$ and not found\_new\_token do   
8: Uniformly sample an instance $( x , y ) \sim \ T _ { m }$ and remove $( x , y )$ from $\mathcal { T } _ { m }$   
9: Extract the set of ground-truth label tokens $\boldsymbol { \nu } ( y ) \subseteq \boldsymbol { \nu }$ from response y   
10: if $\mathcal { V } ( y ) \backslash \mathcal { V } _ { \mathrm { c o v } } \neq \tilde { \emptyset }$ then   
11: $\mathcal { D } _ { P }  \mathcal { D } _ { P } \cup \{ ( x , y ) \}$   
12: $\mathcal { V } _ { \mathrm { c o v } }  \mathcal { V } _ { \mathrm { c o v } } \cup \mathcal { \hat { V } } ( y )$   
13: found\_new\_token ← true {Advance to the next task}   
14: end if   
15: end while   
16: if $\mathcal { T } _ { m } = \varnothing$ then   
17: ${ \mathcal { T } } _ { \mathrm { a c t i v e } } \gets { \mathcal { T } } _ { \mathrm { a c t i v e } } \setminus \{ m \}$ {Remove exhausted task}   
18: end if   
19: end for   
20: end while   
21: return $\mathcal { D } _ { P }$

• PSFT [92] adapts the clipped probability-ratio surrogate objective to offline supervised fine-tuning to bound the policy divergence $D _ { \mathrm { K L } } ( \pi _ { \theta } \parallel \mathbf { \bar { \pi } } _ { \mathrm { r e f } } )$ within a proximal trust region, thereby inducing a Fisher metric barrier in activation space that prevents hidden states from breaching LVD decision hyperplanes on pervasive tasks.

• SMART [54] formulates instruction data curation as a two-stage cardinality-constrained submodular maximization problem, utilizing Graph Cut for task subset selection and Facility Location for non-redundant instance selection. Within the DMA paradigm, it mitigates PCD by maximizing the convex hull volume spanned by active features in the training mixture, maintaining high global feature sparsity $\bar { ( } S  1 )$ to prevent higher-order geometric phase collapse.

• KnOTS [62] applies singular value decomposition (SVD) to jointly transform and align task-specific weight updates (∆W) from homologous fine-tuned models into a shared, mutually orthogonal singular basis prior to model merging. Within the PSO paradigm, it alleviates PCD by eliminating cross-task singular subspace overlaps in parameter space, severing the linear pathways of inter-task feature interference.

• DARE [85] randomly drops a high ratio $( p \in [ 0 . 9 , 0 . 9 9 ] )$ of SFT delta parameters $( \Delta \theta =$ $\theta - \theta ^ { 0 } )$ to zero and rescales the remaining parameters by $1 / ( 1 - p )$ to approximate the original layer embeddings in expectation without retraining. Within the DPP paradigm, it mitigates PCD by pruning the vast majority of redundant cross-coupling parameter updates responsible for parasitic feature crosstalk along ambient pervasive directions.

• DiFT [87] identifies task-critical parameter rows via Delta-Scale Row (DSR) sensitivity probing on forward activation differences and selectively updates only task-relevant differential parameter subsets while zeroing out gradients on remaining rows. Within the GOO paradigm, it alleviates PCD by restricting gradient updates to task-specific subspaces so that optimization trajectories remain orthogonal to ambient pervasive feature channels and avoid destructive gradient conflicts.

• CLUE [7] leverages logical circuit discovery (extracting AND, OR, and ADDER gates) and Conjunctive Normal Form (CNF) Boolean satisfiability solving to disentangle model parameters into task-specific, retain, and conflict nodes. Within the MIL paradigm, it mitigates PCD by freezing task-irrelevant safe and retain sub-networks $( \Delta \dot { W _ { j } } \equiv 0 , \forall j \in$ $\mathcal { N } _ { P } )$ , directly truncating unconstrained activation drift along ambient pervasive feature channels.

Table G3: Mix-up SFT results on Mistral-7B across two four-task mixture groups. For each dataset, the left column reports the target task metric and the right column reports the pervasive average accuracy (P-Acc.). All experiments are conducted over five random seeds; the variance across seeds is consistently below 0.04 with no statistically significant impact on the results and is thus omitted for clarity.
<table><tr><td rowspan="2">Method</td><td colspan="5">Mix-up Group 1</td><td colspan="5">Mix-up Group 2</td><td colspan="5">Mix-up Group 3</td></tr><tr><td>SST-2</td><td>GSM8K</td><td>CodeAlpaca</td><td>LogiQA2</td><td>P-Acc.</td><td>RTE</td><td>GSM8K</td><td>Code-Bagel-Hermes</td><td>Evol-Instruct</td><td>P-Acc.</td><td>CoLA</td><td>GSM8K</td><td>CodeAlpaca</td><td>Evol-Instruct</td><td>P-Acc.</td></tr><tr><td>Vanilla</td><td>0.938</td><td>0.354</td><td>0.115</td><td></td><td>0.535 0.795</td><td>0.343</td><td></td><td>0.417</td><td>0.429</td><td>0.522</td><td>0.405</td><td>0.349</td><td>0.114</td><td>0.429</td><td>0.539</td></tr><tr><td>PSFT</td><td>0.929</td><td>0.361</td><td>0.111</td><td></td><td>0.539</td><td>0.801</td><td>0.351</td><td>0.419</td><td>0.435</td><td>0.526</td><td>0.394</td><td>0.356</td><td>0.116</td><td>0.436</td><td>0.541</td></tr><tr><td>SMART</td><td>0.941</td><td>0.355</td><td>0.118</td><td></td><td>0.537</td><td>0.798</td><td>0.348</td><td>0.412</td><td>0.436</td><td>0.523</td><td>0.409</td><td>0.348</td><td>0.113</td><td>0.439</td><td>0.535</td></tr><tr><td>KnOTS</td><td>0.925</td><td>0.307</td><td>0.094</td><td></td><td>0.491</td><td>0.764</td><td>0.324</td><td>0.395</td><td>0.405</td><td>0.475</td><td>0.359</td><td>0.314</td><td>0.105</td><td>0.405</td><td>0.501</td></tr><tr><td>DARE</td><td>0.918</td><td>0.339</td><td>0.098</td><td></td><td>0.509</td><td>0.778</td><td>0.334</td><td>0.408</td><td>0.401</td><td>0.488</td><td>0.387</td><td>0.316</td><td>0.101</td><td>0.411</td><td>0.514</td></tr><tr><td>DiFT</td><td>0.942</td><td>0.366</td><td>0.108</td><td></td><td>0.543</td><td>0.802</td><td>0.349</td><td>0.425</td><td>0.435</td><td>0.528</td><td>0.411</td><td>0.349</td><td>0.119</td><td>0.435</td><td>0.538</td></tr><tr><td>CLUE</td><td>0.949</td><td>0.384</td><td>0.124</td><td></td><td>0.547</td><td>0.805</td><td>0.356</td><td>0.421</td><td>0.439</td><td>0.533</td><td>0.416</td><td>0.357</td><td>0.118</td><td>0.436</td><td>0.546</td></tr><tr><td>AEGIS</td><td>0.969</td><td>0.405</td><td>0.133</td><td></td><td>0.569</td><td>0.811</td><td>0.374</td><td>0.424</td><td>0.446</td><td>0.546</td><td>0.429</td><td>0.366</td><td>0.124</td><td>0.445</td><td>0.549</td></tr></table>

Implementation Details. All experiments are conducted on a single-node server equipped with NVIDIA A100 GPUs interconnected via NVLink. To guarantee numerical precision for geometric margin computation and bisection solving, we enforce full FP32 precision across the entire training and evaluation pipeline, encompassing model parameters, gradients, optimizer states, and all forward/backward computations. For distributed training, we employ PyTorch Fully Sharded Data Parallel (FSDP) under the full sharding strategy (FULL\_SHARD). All models are optimized using the AdamW optimizer with a learning rate of $1 \times 1 0 ^ { - 5 }$ and a linear warmup schedule, trained for 3 epochs with a maximum sequence length of 512 tokens. During training, each mini-batch is constructed in a balanced 1 : 1 paired manner containing exactly 1 target task sample from $\mathcal { D } _ { T }$ and 1 pervasive task sample from $\mathcal { D } _ { P }$ . For the hyperparameters in the AEGIS objective, we set the semantic proximity temperature in Eq. (41) to $\tau = 1 . 0$ , the Huber smoothing threshold in Eq. (44) to $\kappa = 0 . 0 5$ , and the geometric regularization balancing weight in Eq. (47) to $\lambda = 1 . 0$ (calibrated via 16-step root-mean-square gradient matching), while the exact bisection solver for the shared scalar $\zeta ^ { * }$ in Algorithm 1 is executed with an iteration budget of $T _ { \mathrm { m a x } } = 4 8$ over the top-128 competing token directions.

## H Additional Results Across Diverse SFT Scenarios

## H.1 Mix-up SFT

To evaluate the effectiveness of AEGIS when simultaneously adapting to multiple heterogeneous target tasks, we conduct Mix-up SFT experiments on Mistral-7B. Specifically, we select one representative dataset from each of the four task categories—natural language understanding (NLU), math reasoning, code generation, and logic & instruction-following—and uniformly sample a total of 20,000 training instances at random across the four datasets. We construct two distinct task-mixture groups and repeat each experiment across five random seeds, with the results summarized in Table G3.

As shown in Table G3, AEGIS maintains substantial superiority over all baselines in the Mix-up SFT setting. This demonstrates that AEGIS’s ability to suppress feature interference induced by task adaptation remains robust across diverse target task categories and sample heterogeneity [87]. Similarly, we attribute the performance gains on the individual target tasks to the fact that AEGIS mitigates mutual feature interference among the co-trained target tasks, thereby preserving and enhancing the feature benefit of each respective target task.

## H.2 Parameter-Efficient Fine-Tuning (PEFT)

To evaluate the performance of our method under parameter-efficient fine-tuning (PEFT) regimes, we select three representative baselines that update only a subset of model parameters through distinct mechanisms: LoRA [24](low rank localization), CLUE [7] (mechanistic interpretability localization), and WAGLE [25] (gradient-based attribution localization). Across these three baselines, we compare the effect of combining AEGIS with standard cross-entropy on the pervasive dataset $\mathcal { D } _ { P } .$ . Specifically, we use “+CE” to denote that the loss function on $\mathcal { D } _ { P }$ consists solely of the traditional cross-entropy loss, and $\mathbf { \ddot { \Sigma } } + \mathbf { A } \mathbf { E } \mathbf { G } \mathbf { I } \mathbf { S } ^ { , }$ to denote that the loss on $\mathcal { D } _ { P }$ incorporates both the traditional cross-entropy loss and our AEGIS regularization (Eq. (47)). We evaluate Qwen2.5-7B on the GSM8K dataset across varying percentages of trainable parameters, with the results presented in Figure 5.

![](images/d2f80d0d8f7342c876e92c499e08b4d0de728a38d65c552c84df6e94ea12d6fe.jpg)  
(a) LoRA

![](images/f49fc4a5b1c35cda42b68e1915f4932bedaf567cc59d1613ca394396c1483c3d.jpg)  
(b) CLUE

![](images/95017754694c8044342493d6844ffaddbb48d9434d9a03cbb627c6fe4eadac39.jpg)  
(c) WAGLE  
Figure 5: Performance of AEGIS under different PEFT baselines across varying percentages of trainable parameters.

As shown in Figure 5, across both the target task (GSM8K) and the pervasive tasks, our method (+AEGIS) consistently and significantly outperforms the standard PEFT baselines (+CE) at the same trainable parameter ratios. We conjecture that this advantage demonstrates the superior parameter efficiency of AEGIS over standard cross-entropy, which stems from its ability to move beyond one-hot categorical supervision and simultaneously supervise activation drift across multiple token directions.

## H.3 Continual SFT

To evaluate the effectiveness of AEGIS under sequential multi-stage adaptation, we conduct continual SFT experiments across three models: Mistral-7B, Qwen2.5-7B, and Gemma-2-9B. We design two five-task sequential training streams spanning diverse task domains: Group 1 (GSM8K → SST-2 → RTE → Evol-Instruct → Code-Bagel-Hermes) and Group 2 (CoLA → Infinity-Instruct-Code → $\mathrm { L o g i Q A 2 }  \mathrm { G S M 8 K }  \mathrm { M R P C } )$ . In each subfigure, the horizontal axis denotes the five sequential training task nodes, and the vertical axis reports the relative performance drop (-x%) compared to single-task SFT.

We compare AEGIS against DiFT [87] and PEFT, with the results across all six configurations presented in Figure 6.As shown in Figure 6, AEGIS consistently exhibits strong resilience in continual SFT, incurring minimal negative interference from prior learning stages even after sequentially adapting to multiple tasks. In contrast, while PEFT constrains parameter updates to remain close to the distribution of the preceding gradient step and DiFT utilizes a probe model to identify update directions with minimal task interference, both methods progressively suffer from severe catastrophic forgetting as the number of accumulated tasks increases during continual SFT.

## I Top-10 Drift Directions

To further analyze how AEGIS and standard SFT (Vanilla) differ along individual token directions, we group the decision-edge crossings on the pervasive dataset $\mathcal { D } _ { P }$ by their direction. An edge (t, k) crossed along direction $j _ { t } \to k$ means that the activation at position t has exited $\mathrm { C e l l } _ { j _ { t } } ^ { \mathrm { L V } \bar { \mathrm { D } } }$ of the ground-truth token $j _ { t }$ through the hyperplane $\Pi _ { j _ { t } , k } .$ , so that the competing token k outscores $j _ { t }$ Table I1 lists, for MRPC, CodeAlpaca, and GSM8K, the ten directions with the most newly crossed edges after SFT, together with the ten directions with the most crossed edges under the base model as a reference.

The drift directions of the base model show that the feature interference inherent in the almostorthogonal configuration formed during pre-training manifests mainly as crossings toward generic, semantically empty tokens, such as the end-of-sequence token drifting toward a newline, a space, or punctuation. Such crossings barely affect the prediction of the main content. In contrast, crossentropy SFT on all three target tasks deflects the activations of pervasive tasks toward the label tokens of the target task. On MRPC, whose labels are yes and no, a large number of pervasive activations drift from </s> toward yes, no, yes, and no, which together account for 10,457 newly crossed edges, and on CodeAlpaca and GSM8K most of the top-10 directions point toward digits. AEGIS clearly suppresses these drifts: none of its top-10 directions on MRPC and GSM8K points toward a target label token, and on CodeAlpaca only 3→4 remains, with 27 newly crossed edges compared with 2,121 under Vanilla. The remaining drifts of AEGIS mostly occur between punctuation, conjunctions, articles, the end-of-sequence token, and fragments of multi-byte characters, which resemble the generic crossings of the base model. We regard the shared drift of pervasive activations toward tokens with concrete task semantics as a manifestation of overfitting to the target task. By suppressing this overfitting, AEGIS also improves performance on the target task, which is consistent with the loss and validation accuracy trends in Figure 4.

![](images/eb60558c50b5be92510c68a6b658420af257c4ef968e06ef38fbd9ca4653be01.jpg)  
(a) Group 1: Mistral-7B

![](images/5b19e374d3d13ead861472c652177dd8db703b46330e5709b7568645c10519c6.jpg)  
(b) Group 1: Qwen2.5-7B

![](images/a7607fcdc3e7f1a7d71113dfe7e4b2bb347f4ae3328cff362c8ec74aa13e79c6.jpg)  
(c) Group 1: Gemma-2-9B

![](images/c15bb4cc2d7446f5a159ac96db7bf47f89457a7fbe73678d29109ffa15490972.jpg)  
(d) Group 2: Mistral-7B

![](images/b0e77594e4407fe33653a8a5369838234b388e26d74f621cc6858fde36156f2a.jpg)  
(e) Group 2: Qwen2.5-7B

![](images/2e20608df26b44ce763167fce36f29af43d8bfe8698cc9b1097f2daa31c959c5.jpg)  
(f) Group 2: Gemma-2-9B  
Figure 6: Continual SFT performance across two five-task sequential streams on Mistral-7B, Qwen2.5- 7B, and Gemma-2-9B. Solid and dashed lines denote the relative performance drop (-x%) on target tasks and pervasive tasks compared to single-task training, respectively.

Table I1: Top-10 drift directions on the pervasive dataset $\mathcal { D } _ { P }$ from Mistral-7B. Bold entries point toward a target task label token (yes/no for MRPC, digits for CodeAlpaca and GSM8K). marks a leading space, $< / \varsigma >$ is the end-of-sequence token, <0xXX> is a byte-fallback token, and [U+0ABE] is the Gujarati vowel sign AA (U+0ABE).
<table><tr><td rowspan="2">Rank</td><td rowspan="2">Base</td><td colspan="2">MRPC</td><td colspan="2">CodeAlpaca</td><td colspan="2">GSM8K</td></tr><tr><td>Vanilla</td><td>AEGIS</td><td>Vanilla</td><td>AEGIS</td><td>Vanilla</td><td>AEGIS</td></tr><tr><td>1</td><td>&lt;/s&gt;→&lt;0x0A&gt; (11784)</td><td>&lt;/s&gt;→_yes (4327)</td><td>.→&lt;/s&gt; (1578)</td><td>.→, (3330)</td><td>, →_and (172)</td><td>answer→_final (4154)</td><td>. →, (218)</td></tr><tr><td>2</td><td>&lt;/s&gt;→ (5768)</td><td>&lt;/s&gt;→_no (3474)</td><td>&lt;0x0A&gt;→&lt;/s&gt; (1473)</td><td>1→2 (3173)</td><td>.→_and (156)</td><td>a→the (2250)</td><td>, →_and (125)</td></tr><tr><td>3</td><td>&lt;/s&gt;→. (5151)</td><td>&lt;/s&gt;→yes (1927)</td><td>,→&lt;/s&gt; (1445)</td><td>1→4 (2150)</td><td>.→, (52)</td><td>The→&lt;/s&gt; (1156)</td><td>_the→_a (115)</td></tr><tr><td>4</td><td>&lt;/s&gt;→= (4044)</td><td>(→&lt;/s&gt; (1259)</td><td>, →_and (1342)</td><td>3→4 (2121)</td><td>&lt;0x0A&gt;→&lt;/s&gt; (42)</td><td>1→2 (659)</td><td>.→_and (93)</td></tr><tr><td>5</td><td>&lt;/s&gt;→( (3879)</td><td>&lt;/s&gt;→no (729)</td><td>→&lt;/s&gt; (1330)</td><td>_a→_the (2119)</td><td>, →. (39)</td><td>4→2 (237)</td><td>, →. (92)</td></tr><tr><td>6</td><td>&lt;/s&gt;→: (3873)</td><td>.→&lt;/s&gt; (377)</td><td>_the→_a (1329)</td><td>5→4 (1108)</td><td>→&lt;0xE0&gt; (34)</td><td>5→2 (224)</td><td>&lt;0xE0&gt;→_ (80)</td></tr><tr><td>7</td><td>&lt;/s&gt;→The (3794)</td><td>, →. (194)</td><td>, →. (1235)</td><td>2→4 (685)</td><td>&lt;0xA9&gt;→&lt;0xA8&gt; (33)</td><td>5→4 (220)</td><td>&lt;0xE0&gt;→[U+0ABE] (77)</td></tr><tr><td>8</td><td>&lt;/s&gt;→... (3779)</td><td>→&lt;0xE0&gt; (141)</td><td>.→_and (1209)</td><td>5→6 (277)</td><td>3→4 (27)</td><td>0→2 (118)</td><td>and→, (71)</td></tr><tr><td>9</td><td>&lt;/s&gt;→1 (3777)</td><td>&lt;0xAA&gt;→&lt;0xAB&gt; (121)</td><td>u(→&lt;/s&gt; (1179)</td><td>1→3 (75)</td><td>, →_in (25)</td><td>0→4 (32)</td><td>a→_the (69)</td></tr><tr><td>10</td><td>&lt;/s&gt;→2 (3664)</td><td>in→&lt;/s&gt; (113)</td><td>and→&lt;/s&gt; (1151)</td><td>0→4 (67)</td><td>→&lt;/s&gt; (23)</td><td>1→6 (21)</td><td>, →_in (67)</td></tr></table>