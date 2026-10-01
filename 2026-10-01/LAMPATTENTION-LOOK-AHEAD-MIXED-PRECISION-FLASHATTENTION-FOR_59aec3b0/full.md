# LAMPATTENTION: LOOK-AHEAD MIXED-PRECISION FLASHATTENTION FOR DEDICATED ACCELERATORS

Stanislav Budzinskiy, Marian Gloser, Tolunay Yilmaz   
Faculty of Mathematics   
University of Vienna, Austria

Ying Hong Tham Huawei Heisenberg Research Center, Munich, Germany

Yuanyi Lin, Wenyi Fang, Fan Wu Huawei Technologies Co. Ltd

Philipp Petersen Faculty of Mathematics University of Vienna, Austria

## ABSTRACT

While most attention logits can be computed in low precision without degrading numerical stability, current attention kernels fail to exploit this phenomenon. We introduce a novel hardware-algorithm co-design in the form of mixed-precision FlashAttention. Our method accumulates key-query products and evaluates their exponentials in 8-bit formats, then adaptively identifies sensitive sub-blocks and recomputes them in 16-bit formats. We propose the specifications for a dedicated accelerator capable of executing this pipeline efficiently. Simulated experiments with Qwen3 and Gemma 3 show that rerouting a selective minority of sub-blocks to high precision is sufficient to recover the baseline model performance.

## 1 INTRODUCTION

Mixed-precision floating-point arithmetic is a fundamental driver of scale in modern deep learning (Gupta et al., 2015; Micikevicius et al., 2018; Kalamkar et al., 2019; Micikevicius et al., 2022). In practice, this efficiency relies entirely on mixed-precision matrix products: the operands are stored in low precision, whereas the output is accumulated in high precision. As modern AI accelerators natively support these operations, they yield massive throughput gains across both training and inference. Furthermore, the numerical reliability of mixed-precision matrix products is guaranteed by well-established theoretical round-off error bounds (Blanchard et al., 2020; Abdelfattah et al., 2025).

Beyond isolated matrix products, the overall numerical stability of a neural network depends heavily on how local round-off errors propagate through the entire architecture and interact with its layers, especially the highly nonlinear attention layers of transformers (Budzinskiy et al., 2025). A standard, conservative strategy is to accumulate all matrix products and execute all intermediate computations in 32-bit arithmetic before passing the results down the computational pipeline. However, recent theoretical advances (El Arar et al., 2026; Budzinskiy et al., 2026) reveal that this approach is overly restrictive and introduce a dynamic alternative: a look-ahead mixed-precision (LAMP) framework. The core idea of LAMP is to maximize the use of low-precision arithmetic by default, using analytical error bounds to selectively trigger high-precision recomputation only where numerical stability is at risk. These works empirically validate LAMP on feedforward activations and transformer softmax functions, showing that large fractions of the directly preceding matrix products can be accumulated in low precision without degrading end-to-end inference accuracy.

A fundamental limitation of these prior theoretical works is that they are blind to physical hardware realities. Their LAMP algorithms fail to account for the structural constraints imposed by practical, hardware-aware kernels such as the industry-standard FlashAttention (Dao et al., 2022; Dao, 2024).

## 1.1 CONTRIBUTIONS

We address this gap by explicitly adapting the LAMP framework to FlashAttention. Specifically, we develop a hardware-algorithm co-design (Sections 2 and 3):

• LampAttention, a LAMP modification of FlashAttention that identifies numerically destabilizing logits in two stages, guided by running maxima and an analytical error bound;

• a specification for a hypothetical dedicated accelerator capable of executing LampAttention efficiently, featuring native 8-bit accumulation and LUT-based evaluation of exponentials.

LampAttention is distinctly not a quantization technique. Rather than compressing model weights or the input key-query-value operands, it exclusively minimizes the precision of intermediate calculations inside the attention kernel: the matrix accumulations and the softmax exponentials. Therefore, the algorithm operates independently of, and seamlessly alongside, external optimizations such as operand quantization or rotary positional encodings (Su et al., 2024).

Targeting the massive computational demands of LLM deployment, we evaluate LampAttention for inference across two modern LLM architectures, Qwen3 (Yang et al., 2025) and Gemma 3 (Kamath et al., 2025), measuring its impact on perplexity and downstream tasks (Section 4). Because commercially available hardware lacks native support for 8-bit accumulation, we simulate the arithmetic to guarantee numerical fidelity. Consequently, rather than reporting wall-clock timings, we quantify the volume of high-precision recomputations against task metrics, mapping the efficiency-accuracy trade-off attainable by future accelerators.

## 1.2 RELATED WORK

Transformers. The attention mechanism was originally introduced in Vaswani et al. (2017) in the context of machine translation and forms the basis of all modern transformers. These have achieved widespread success in a variety of natural language tasks, with prominent examples including BERT (Devlin et al., 2019), GPT (Brown et al., 2020), PaLM (Chowdhery et al., 2023), Llama (Touvron et al., 2023), Qwen (Yang et al., 2025), and DeepSeek (Liu et al., 2024). Transformers and related attention-based architectures have also enabled advances in scientific and mathematical applications, including accurate protein structure prediction with AlphaFold 2 (Jumper et al., 2021) and Olympiad-level geometry problem solving with AlphaGeometry (Trinh et al., 2024).

Efficient attention. Despite their success, transformers are bound by a fundamental bottleneck: the compute and memory costs of standard attention scale quadratically with sequence length. The family of FlashAttention algorithms (Dao et al., 2022; Dao, 2024; Shah et al., 2024; Zadouri et al., 2026) achieves linear memory cost by reorganizing the standard formulation in an I/O-aware manner. Sparse attention Beltagy et al. (2020); Kitaev et al. (2020) uses local sliding windows or fixed stride patterns to discard specific token interactions. Linear (Katharopoulos et al., 2020) and log-linear (Guo et al., 2026) variants of attention remove the softmax function. KV-caching (Dai et al., 2019) prevents redundant recomputation of past tokens during generation, while methods such as groupedquery attention (Ainslie et al., 2023) reduce the memory footprint of the KV-cache.

Quantization. To alleviate disk-storage and runtime memory requirements, quantization reduces the precision of model weights and activations. Weight-only quantization (Frantar et al., 2023; Lin et al., 2024) compresses static model parameters. Weight-activation quantization (Dettmers et al., 2022; Xiao et al., 2023) compresses both weights and activations, preparing them to become mixedprecision matrix-product operands. LampAttention performs neither. It exclusively controls the precision of calculations themselves.

Mixed-precision arithmetic. Frameworks for mixed-precision deep learning (Micikevicius et al., 2018; Kalamkar et al., 2019) accelerate training and inference by allocating different numerical formats to different components of the computational pipeline. This assignment is fundamentally interoperational and homogeneous: both operands of a mixed-precision matrix product are quantized to one precision¹ and its entire output is accumulated in another precision; all of the softmax exponentials are computed in the same precision. In contrast, LampAttention introduces intra-operational heterogeneous mixed precision.

## 2 LOOK-AHEAD MIXED-PRECISION COMPUTATION OF ONLINE SOFTMAX

The principal enabling feature of FlashAttention is the online computation of the softmax function

$$
\varphi ( \mathbf { x } ) = { \frac { 1 } { \sum _ { i = 1 } ^ { n } \exp ( x _ { i } ) } } \left[ \exp ( x _ { 1 } ) \quad \cdots \quad \exp ( x _ { n } ) \right] ^ { \intercal } , \quad \mathbf { x } \in \mathbb { R } ^ { n } ,
$$

whereby an intermediate result is updated with new logits $\mathbf { y } \in \mathbb { R } ^ { m }$ . Let $\varphi _ { \mathbf { x } } : \mathbb { R } ^ { m }  \mathbb { R } ^ { m + n }$ be the corresponding update function. It is computed as follows. The algorithm keeps track of

$$
\mu  \underset { 1 \leq i \leq n } { \operatorname* { m a x } } x _ { i } , \quad \omega  \sum _ { i = 1 } ^ { n } \exp ( x _ { i } - \mu ) , \quad \mathbf { s }  \varphi ( \mathbf { x } ) ,
$$

and updates their running values according to

$$
\begin{array} { r } { \hat { \mu }  \operatorname* { m a x } \Big \{ \mu , \underset { 1 \leq i \leq m } { \operatorname* { m a x } } y _ { i } \Big \} , \quad \hat { \alpha }  \exp ( \mu - \hat { \mu } ) \omega , \quad \hat { \mathbf { e } }  [ \exp ( y _ { 1 } - \hat { \mu } ) \quad \cdots \quad \exp ( y _ { m } - \hat { \mu } ) ] ^ { \intercal } , } \\ { \hat { \omega }  \hat { \alpha } + \| \hat { \mathbf { e } } \| _ { 1 } , \quad \hat { \mathbf { s } }  \displaystyle \frac { 1 } { \hat { \omega } } [ \hat { \alpha } { \mathbf { s } } ^ { \intercal } \quad \hat { \mathbf { e } } ^ { \intercal } ] ^ { \intercal } = \varphi _ { \mathbf { x } } ( \mathbf { y } ) . } \end{array}
$$

## 2.1 FLOATING-POINT NORMALIZATION

To analyze the effects of rounding errors on the computation of $\varphi _ { \mathbf { x } } .$ , we focus on its compositional structure and introduce the $\ell _ { 1 }$ normalization function $\eta ( \mathbf { z } ) = \mathbf { z } / \lVert \mathbf { z } \rVert _ { 1 }$ , so that

$$
\begin{array} { r } { \varphi _ { \mathbf { x } } ( \mathbf { y } ) = \eta ( \mathbf { z } ) , \quad \mathbf { z } = [ \hat { \alpha } \mathbf { s } ^ { \intercal } \quad \hat { \mathbf { e } } ^ { \intercal } ] ^ { \intercal } \in \mathbb { R } ^ { n + m } . } \end{array}
$$

Then the componentwise error of floating-point (FP) evaluation of $\varphi _ { \mathbf { x } }$ can be bounded $\mathsf { b y } ^ { 2 }$

$$
| \mathsf { f l } ( \eta ( \mathbf { z } ) ) - \eta ( \mathbf { z } ) | \leq | \mathsf { f l } ( \eta ( \mathbf { z } ) ) - \eta ( \mathsf { f l } ( \mathbf { z } ) ) | + | \eta ( \mathsf { f l } ( \mathbf { z } ) ) - \eta ( \mathbf { z } ) | ,
$$

where the first term is the rounding error of normalization at the evaluated value of z and the second term is the discrepancy between the exactly normalized exact and evaluated values of z. Denoting by $\mathrm { u } _ { \eta }$ the FP precision used to evaluate η, we get with standard techniques (Higham, 2002)

$$
\left| \mathsf { f l } \left( \eta ( \mathbf { z } ) \right) - \eta ( \mathbf { z } ) \right| \leq \gamma _ { \eta , m + 1 } \eta ( \mathbf { z } ) + ( 1 + \gamma _ { \eta , m + 1 } ) | \eta ( \mathsf { f l } ( \mathbf { z } ) ) - \eta ( \mathbf { z } ) | , \quad \gamma _ { \eta , m + 1 } = \frac { ( m + 1 ) \mathsf { u } _ { \eta } } { 1 - ( m + 1 ) \mathsf { u } _ { \eta } } .\tag{1}
$$

The bound is now determined by how the $\ell _ { 1 }$ normalization function η propagates the rounding errors in the computation of the exponentials and their arguments.

## 2.2 FLOATING-POINT INPUTS OF NORMALIZATION

Let us now consider the evaluation of $\mathbf { z } ,$ taking into account the rounding errors due to FP evaluation of logits y. Specifically, suppose that $\Delta y _ { j } \stackrel { \_ } { = } | \mathsf { f l } ( y _ { j } ) - y _ { j } | = \mathcal { O } ( \mathrm { u } _ { y _ { j } } )$ for $1 \leq j \leq m$ , and denote $\Delta \hat { \mu } = | { \sf f l } ( \hat { \mu } ) - \hat { \mu } |$ . Then standard worst-case rounding error analysis yields

$$
\begin{array} { r } { \left| \mathsf { H } ( \hat { \alpha } s _ { i } ) - \hat { \alpha } s _ { i } \right| \le \hat { \alpha } s _ { i } \left[ ( 1 + \gamma _ { \hat { \alpha } , 3 } ) \exp \left( \mathsf { u } _ { \hat { \alpha } } ( \hat { \mu } - \mu + \Delta \hat { \mu } ) + \Delta \hat { \mu } \right) - 1 \right] = \hat { \alpha } s _ { i } w _ { 0 } , } \end{array}
$$

$$
\begin{array} { r } { | \mathsf { H } ( \hat { e } _ { j } ) - \hat { e } _ { j } | \le \hat { e } _ { j } \Big [ \big ( 1 + \mathtt { u } _ { \hat { e } _ { j } } \big ) \exp \Big ( \mathtt { u } _ { \hat { e } _ { j } } \big ( \hat { \mu } - y _ { j } + \Delta \hat { \mu } + \Delta y _ { j } \big ) + \Delta \hat { \mu } + \Delta y _ { j } \Big ) - 1 \Big ] = \hat { e } _ { j } w _ { j } . } \end{array}
$$

It follows from Taylor's expansion that

$$
\left| \eta ( \mathsf { f l } ( \mathbf { z } ) ) - \eta ( \mathbf { z } ) \right| \leq | \mathbf { J } _ { \eta } ( \mathbf { z } ) \mathrm { d i a g } ( \mathbf { z } ) | \hat { \mathbf { w } } + \mathcal { O } ( \| \hat { \mathbf { w } } \| _ { \infty } ^ { 2 } ) , \quad \hat { \mathbf { w } } = [ w _ { 0 } \mathbf { 1 } _ { n } ^ { \mathsf { T } } \quad \mathbf { w } ^ { \mathsf { T } } ] ^ { \mathsf { T } } \in \mathbb { R } ^ { n + m } ,\tag{2}
$$

where $\mathbf { J } _ { \eta } ( \mathbf { z } ) \in \mathbb { R } ^ { ( n + m ) \times ( n + m ) }$ is the Jacobian and  describes the relative rounding errors of $\mathsf { f l } ( \mathbf { z } )$ where distinct precisions are used for its components.

It remains to bound $\Delta \hat { \mu } .$ for which we introduce a reasonable assumption that rounding errors in the evaluation of $\mathbf { y }$ do not switch its maximizer, i.e., $y _ { c } = \operatorname* { m a x } _ { j } y _ { j }$ and $\mathsf { f l } ( y _ { c } ) = \operatorname* { m a x } _ { j } \mathsf { f l } ( y _ { j } )$ . Then

$$
\Delta \hat { \mu } \leq \left\{ \begin{array} { l l } { 0 , } & { \mu \geq \operatorname* { m a x } \{ y _ { c } , \mathsf { f l } ( y _ { c } ) \} , } \\ { \Delta y _ { c } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

We make another structural decision and assume that  is computed in high precision. This choice imposes negligible computational overhead since a single exponential is evaluated, and in the limiting case of ${ \mathrm { u } } _ { \hat { \alpha } } = 0$ , which we shall adopt, it yields $w _ { 0 } = \exp ( \Delta \hat { \mu } ) - 1$

## 2.3 PROPAGATION OF MIXED-PRECISION ROUNDING ERRORS

To simplify presentation, and without loss of generality, we let $\mathrm { u } _ { j } = \mathrm { u } _ { y _ { j } } = \mathrm { u } _ { \hat { e } _ { j } }$ for each $1 \leq j \leq m$ Suppose that these precisions are divided into two categories:

$$
\mathfrak { u } _ { j } = \left\{ \mathfrak { u } , \quad j \notin \Omega , \quad \Omega \subseteq \{ 1 , \dots , m \} , \quad \epsilon \ll 1 . \right\} \in \Omega .
$$

That ${ \mathrm { i s } } ,$ the exponential $\boldsymbol { \hat { e } } _ { j }$ and the logit $y _ { j }$ are evaluated in high precision when $j \in \Omega$ and in lower precision when $j \not \in \Omega$ . Henceforth, we resort to the limiting case of $\epsilon = 0$ to streamline the analysis. Consider a binary vector $\mathbf { g } \in \{ 0 , 1 \} ^ { m }$ that vanishes exactly on Ω. We can represent  as

$$
\hat { \mathbf { w } } = \mathrm { d i a g } ( \mathbf { h } ) \hat { \mathbf { w } } , \quad \mathbf { h } = \left\{ \left[ \mathbf { 0 } _ { n } ^ { \top } \quad \mathbf { g } ^ { \top } \right] ^ { \top } , \quad \mu \geq \operatorname* { m a x } \{ y _ { c } , \mathsf { f l } ( y _ { c } ) \} ~ \mathrm { o r } ~ c \in \Omega , \right.\tag{3}
$$

The binary vector $\mathbf { h } \in \{ 0 , 1 \} ^ { n + m }$ specifies how the rounding errors propagate through η.

Proposition 1. Let $\mathbf { z } \in \mathbb { R } _ { + } ^ { n + m } , \hat { \mathbf { w } } \in \mathbb { R } ^ { n + m }$ , and $\mathbf { h } \in \{ 0 , 1 \} ^ { n + m }$ . For every $1 \leq l \leq n + m$

$$
| \mathbf { e } _ { l } ^ { \mathsf { T } } \mathbf { J } _ { \eta } ( \mathbf { z } ) \mathrm { d i a g } ( \mathbf { z } ) \mathrm { d i a g } ( \mathbf { h } ) | \hat { \mathbf { w } } \leq \eta ( \mathbf { z } ) _ { l } \Big [ \mathbf { h } ^ { \mathsf { T } } \eta ( \mathbf { z } ) + h _ { l } \Big ( 1 - 2 \eta ( \mathbf { z } ) _ { l } \Big ) \Big ] \| \hat { \mathbf { w } } \| _ { \infty } .
$$

Proof. See Appendix A.

Let us combine Proposition 1 with error bounds (1) and (2). Summing over $1 \leq l \leq n + m$

$$
\begin{array} { r } { \lVert \mathbf { f } \rVert ( \eta ( \mathbf { z } ) ) - \eta ( \mathbf { z } ) \rVert _ { 1 } \leq \gamma _ { \eta , m + 1 } + 2 ( 1 + \gamma _ { \eta , m + 1 } ) \left| \mathbf { h } ^ { \mathsf { T } } \mathrm { d i a g } \big ( \mathbf { 1 } _ { n + m } - \eta ( \mathbf { z } ) \big ) \eta ( \mathbf { z } ) \right| \lVert \hat { \mathbf { w } } \rVert _ { \infty } + \mathcal { O } ( \lVert \hat { \mathbf { w } } \rVert _ { \infty } ^ { 2 } ) . } \end{array}\tag{4}
$$

This bound controls the probability mass shift caused by the FP evaluation of softmax and is maximized when $\mathbf { h } = \mathbf { 1 } _ { n + m } , 1 . \mathsf { e } .$ , when the running maximum $\hat { \mu }$ is set to a low-precision logit.

## 2.4 LOOK-AHEAD DISTRIBUTION OF PRECISIONS

Our error bound (4) represents a trade-off between cheap, low-precision evaluation of softmax and its numerical stability: maximum efficiency is achieved with $\mathbf { g } = \mathbf { 1 } _ { m }$ , whereas $\mathbf { g } = \mathbf { 0 } _ { m }$ guarantees unconditional numerical stability. This setting naturally lends itself to LAMP analysis (El Arar et al., 2026; Budzinskiy et al., 2026). Specifically, we can optimize

$$
\mathbf { 1 } _ { m } ^ { \intercal } \mathbf { g }  \mathrm { m a x } \quad \mathrm { s . t . } \quad \mathbf { h } ^ { \intercal } \mathrm { d i a g } \big ( \mathbf { 1 } _ { n + m } - \eta ( \mathbf { z } ) \big ) \eta ( \mathbf { z } ) \leq \tau ,\tag{5}
$$

where $0 \leq \tau < 1$ is a small threshold and h is derived from g via (3). However, we do not have access to the exact values of $\eta ( \mathbf { z } )$ in practice. Additionally, the explicit storage of the first n softmax probabilities would render any method inefficient in long-context scenarios.

A two-step procedure can overcome these issues. First, we look at the logits $\mathsf { f l } ( \mathbf { y } )$ evaluated in low precision and identify their maximizer $\mathsf { f l } ( y _ { c } )$ . If $\mu > \mathsf { f l } ( y _ { c } ) + | \mu | \delta$ with a safety margin $\delta \geq 0$ then we proceed; otherwise, $\mathsf { f l } ( y _ { c } )$ is recomputed in high precision. In either case, we end up in the first branch of $( 3 ) , ^ { 3 }$ and therefore the optimization problem (5) relies entirely on the newly added softmax probabilities. Denote by $\mathbf { p } \in \mathbb { R } ^ { m }$ these last m components of $\mathfrak { f l } ( \eta ( \mathbf { z } ) )$ evaluated according to the precisions chosen for fl(y). Then we can reformulate (5) equivalently as

$$
\mathbf { 1 } _ { m } ^ { \intercal } \mathbf { g } \to \operatorname* { m a x } \quad \mathrm { s . t . } \quad \mathbf { g } ^ { \intercal } \mathrm { d i a g } \big ( \mathbf { 1 } _ { m } - \mathbf { p } \big ) \mathbf { p } \leq \tau , \quad g _ { c } = 0 \mathrm { ~ i f ~ } \mu \leq \mathsf { f l } ( y _ { c } ) + | \mu | \delta .\tag{6}
$$

This is a knapsack problem with uniform values, and hence its optimal solution can be obtained with a greedy algorithm: select the smallest entries of $\mathrm { d i a g } \big ( { \bf 1 } _ { m } - \bar { { \bf p } } \big ) { \bf p }$ until the threshold is surpassed, subject to the constraint on $g _ { c } .$ The solution g determines the distribution of precisions across logits, and we reevaluate them, together with their exponentials, in higher precision accordingly. Finally, we recompute the $\ell _ { 1 }$ normalization for its updated inputs.

By restricting the binary vector h in (3) strictly to the first branch, we ensure that the LAMP evaluation of online softmax can be seamlessly integrated into FlashAttention. Indeed, the method relies only on the running maximum and the running normalization constant, completely avoiding the need to materialize the historical probability distribution.

## 3 LOOK-AHEAD MIXED-PRECISION FLASHATTENTION

The framework developed in Section 2 deviates from the practice of FlashAttention in three aspects, two of which stem from the block-based nature of efficient matrix multipliers in modern AI accelerators. First, instead of looking at a single vector of logits $\mathbf { y } \in \mathbb { R } ^ { m }$ at a time, FlashAttention processes an entire block of queries simultaneously, leading to a matrix of logits $\mathbf { Y } \in \mathbb { R } ^ { m \times b _ { q } }$ . Second, assuming that the “atomic"matrix product is $\widehat { ( b _ { k } \times b ) ^ { - } } \times ( b \times b _ { q } ) \to ( \bar { b _ { k } } \times b _ { q } )$ , the number of new keys per query is $m =  { b _ { k } } N _ { k }$ and the (re)computation of logits is strictly blockwise. Third, the output of attention is scaled by the normalization constant only at the very end of the computational pipeline, i.e., FlashAttention keeps the intermediate exponentials unnormalized.

## 3.1 BLOCK MODIFICATION OF LAMP COMPUTATION OF ONLINE SOFTMAX

Let $\pmb { \mu } \in \mathbb { R } ^ { b _ { q } }$ be the vector of running maxima for each query and denote by $\mathbf { C } \in \{ 1 , \dots , b _ { k } \} ^ { N _ { k } \times b _ { q } }$ the matrix of maximum-logit indices for each sub-block of low-precision fl(Y). Consider

$$
T _ { \theta } = \operatorname* { m a x } _ { 1 \leq j \leq b _ { q } } \big \{ \mathfrak { f } | ( Y _ { i , j } ) - \mu _ { j } \big \} , \quad i = C _ { \theta , j } + ( \theta - 1 ) b _ { k } , \quad 1 \leq \theta \leq N _ { k } .
$$

This quantity represents the “threat" caused by the θth sub-block, and we assume that the sub-blocks are sorted so that $T _ { 1 } \ge \cdots \ge T _ { N _ { k } }$ . Let ${ \hat { \pmb { \mu } } } = { \pmb { \mu } }$ and process the sub-blocks sequentially. If

$$
\begin{array} { r } { \hat { \mu } _ { j } > \mathsf { f l } ( Y _ { i , j } ) + | \hat { \mu } _ { j } | \delta , \quad i = C _ { \theta , j } + ( \theta - 1 ) b _ { k } \quad \mathrm { f o r ~ a l l } \quad 1 \leq j \leq b _ { q } , } \end{array}\tag{7}
$$

the θth sub-block of $\mathsf { f l } ( \mathbf { Y } )$ remains in low precision; otherwise, it is recomputed in high precision and the running maxima $\hat { \pmb { \mu } }$ are updated. Denote by $\Theta \subseteq \{ 1 , \dots , N _ { k } \}$ the set of indices of recomputed sub-blocks. The role of sorting according to $\dot { T _ { \theta } }$ is to attempt to reduce the cardinality of Θ.

Next, we need to modify the LAMP problem (6). For every query $1 \le j \le b _ { q } ,$ denote by $\hat { \mathbf { e } } _ { j } \in \mathbb { R } ^ { b _ { k } N _ { k } }$ the vector of unnormalized shifted exponentials and by $\bar { \hat { \omega } } _ { j } \ \in \bar { \mathbb { R } }$ the updated running normalization constant. Let $\mathbf { g } \in \{ 0 , 1 \} ^ { N _ { k } }$ encode the precisions used for the sub-blocks, so that $g _ { \theta } = 0$ whenever the θth sub-block requires high precision. Then we propose to solve

$$
\begin{array} { r l } { { \bf 1 } _ { N _ { k } } ^ { \intercal } \mathbf { g }  \mathrm { m a x } \quad \mathrm { s . t . } \quad \xi _ { j } ( \mathbf { g } ) \leq \tau \hat { \omega } _ { j } ^ { 2 } \mathrm { ~ f o r ~ a l l ~ } 1 \leq j \leq b _ { q } , \quad g _ { \theta } = 0 \mathrm { ~ f o r ~ } \theta \in \Theta , } \\ & { \xi _ { j } ( \mathbf { g } ) = ( \mathbf { g } \otimes \mathbf { 1 } _ { b _ { k } } ) ^ { \intercal } \mathrm { d i a g } \big ( \hat { \omega } _ { j } \mathbf { 1 } _ { b _ { k } N _ { k } } - \hat { \mathbf { e } } _ { j } \big ) \hat { \bf e } _ { j } . } \end{array}\tag{8}
$$

Here, $\otimes$ is the Kronecker product. Unlike (6), this modified optimization problem is a multidimensional knapsack problem (Kellerer et al., 2004) and, in general, does not admit a greedy solution. To solve it, we can rely on the number of sub-blocks $N _ { k }$ being small and compare all feasible configurations of g in the descending order of $\mathbf { 1 } _ { N _ { k } } ^ { \intercal } \mathbf { g }$ . Upon finding an optimal precision distribution, we recompute the identified sub-blocks, their exponentials, and the running normalization constant.

## 3.2 LAMPATTENTION

We integrate the LAMP framework into the FlashAttention-2 kernel (Dao, 2024), omitting the warpgroup instructions and specialization of FlashAttention-3 (Shah et al., 2024) and the asynchronous computations with dedicated memory levels of FlashAttention-4 (Zadouri et al., 2026). We assume a standard accelerator architecture characterized by (i) a memory hierarchy that distinguishes between HBM, SRAM, and registers; (ii) a SIMT execution model partitioned into thread blocks and further subdivided into warps. The pseudocode of our LampAttention algorithm is listed in Algorithm 1.

The mixed-precision sections of Algorithm 1 are on lines 10–20 and 22–26, while the rest is vanilla FlashAttention-2. Note that (causal) masking and RoPE (Su et al., 2024) are seamlessly supported.

Algorithm 1: LampAttention   
Input: $\mathbf { Q } _ { f u l l } , \mathbf { K } _ { f u l l } , \mathbf { V } _ { f u l l } \in \mathbb { R } ^ { d \times L }$ stored in HBM for a single attention head and a single sequence of   
L tokens; query block size $b _ { q }$ and key/value block size $\bar { b } _ { k }$ such that $L = b _ { q } N _ { q } M _ { q } = b _ { k } \mathbf { \dot { N } } _ { k } M _ { k } .$   
safety margin $\dot { \delta } > 0$ of condition (7); threshold $0 \leq \tau < 1$ of block-LAMP problem (8)   
Output: attention $\mathbf { \bar { O } } _ { f u l l } \in \mathbb { R } ^ { d \times L }$   
1 foreach $v _ { q } = 1 , \ldots , M _ { q }$ in parallel do   
/\* single thread block with its SRAM section \*/   
2 load $v _ { q } \mathbf { t }$ h tile $[ \mathbf { Q } _ { 1 } \ \cdot \ \cdot \ \mathbf { Q } _ { N _ { q } } ] \in \mathbb { R } ^ { d \times b _ { q } N _ { q } }$ into SRAM   
3 initialize output ${ \mathbf O } = [ { \mathbf O } _ { 1 } \ \dot { \bf \Phi } \cdot \cdot \ { \mathbf O } _ { N _ { q } } ] \gets { \mathbf 0 } _ { d \times b _ { q } N _ { q } }$   
4 initialize running quantities ${ \pmb \mu } = [ { \pmb \mu } _ { 1 } ^ { \top } \mathrm { ~ } \cdots { \pmb \mu } _ { N _ { q } } ^ { \top } ] ^ { \top }  ( - \infty ) { \bf 1 } _ { b _ { q } N _ { q } }$ and $\boldsymbol { \omega } = [ \omega _ { 1 } ^ { \intercal } \cdot \cdot \cdot \omega _ { N _ { q } } ^ { \intercal } ] ^ { \intercal } \gets \mathbf { 0 } _ { b _ { q } N _ { q } }$   
5 for $v _ { k } = 1 , \ldots , M _ { k }$ do   
6 load vkth tiles $\mathbf { K } = [ \mathbf { K } _ { 1 } \ \cdot \cdot \cdot \ \mathbf { K } _ { N _ { k } } ] , \mathbf { V } = [ \mathbf { V } _ { 1 } \ \cdot \cdot \cdot \ \mathbf { V } _ { N _ { k } } ] \in \mathbb { R } ^ { d \times b _ { k } N _ { k } }$ into SRAM   
7 initialize $\hat { \pmb { \mu } }  \pmb { \mu }$   
8 foreach $w = 1 , \ldots , N _ { q }$ in parallel do   
/\* single warp with its SRAM subsection and registers \*/   
9 accumulate $\begin{array} { r } { \mathbf { Y }  \frac { 1 } { \sqrt { d } } \mathbf { K } ^ { \top } \mathbf { Q } _ { w } \in \mathbb { R } ^ { b _ { k } N _ { k } \times b _ { q } } } \end{array}$ in low precision   
10 find maximum-logit indices $\mathbf { C } \in \{ 1 , \dots , b _ { k } \} ^ { N _ { k } \times b _ { q } }$ for Y and compute $\mathbf { T } \in \mathbb { R } ^ { N _ { k } }$   
11 sort $( \theta _ { 1 } , \ldots , \theta _ { N _ { k } } )$ so that $T _ { \theta _ { 1 } } \geq \cdots \geq T _ { \theta _ { N _ { k } } }$ and initialize $\Theta  \emptyset$   
12 for $\theta = \theta _ { 1 } , \ldots , \theta _ { N _ { k } }$ do   
13 if safety-margin condition (7) based on ${ \hat { \mu } _ { w } }$ fails for θth sub-block of Y then   
14 accumulate $\begin{array} { r } { \mathbf { Y } _ { \theta } \gets \frac { 1 } { \sqrt { d } } \mathbf { K } _ { \theta } ^ { \intercal } \mathbf { Q } _ { w } \in \mathbb { R } ^ { \dot { b } _ { k } \times \ddot { b } _ { q } } } \end{array}$ in high precision and update $\Theta  \Theta \cup \{ \theta \}$   
15 update $\hat { \pmb { \mu } } _ { w } \gets \operatorname* { m a x } \{ \hat { \pmb { \mu } } _ { w } ,$ colmax(Yθ)}   
16 for $\theta = 1 , \ldots , N _ { k }$ do   
17 if $\theta \in \Theta$ then   
18 compute in-place $\hat { \mathbf { E } } _ { \theta } \gets \exp ( \mathbf { Y } _ { \theta } - \mathbf { 1 } _ { b _ { k } } \hat { \pmb { \mu } } _ { w } ^ { \top } )$ in high precision   
19 else   
20 compute in-place $\hat { \mathbf { E } } _ { \theta } \gets \exp ( \mathbf { Y } _ { \theta } - \mathbf { 1 } _ { b _ { k } } \hat { \pmb { \mu } } _ { w } ^ { \top } )$ in low precision   
21 compute $\hat { \lambda }  \exp ( \pmb { \mu } _ { w } - \hat { \pmb { \mu } } _ { w } )$ , and $ \mathrm { d i a g } ( \hat { \lambda } ) \omega _ { w } ,$ and $\hat { \omega }  \hat { \alpha } + [ \| \hat { \mathbf { e } } _ { 1 } \| _ { 1 } \cdot \cdot \cdot \| \hat { \mathbf { e } } _ { b _ { q } } \| _ { 1 } ] ^ { \intercal }$   
22 obtain solution $\mathbf { g } \in \{ 0 , 1 \} ^ { N _ { k } }$ of block-LAMP problem (8)   
23 for $\theta = 1 , \ldots , \bar { N } _ { k }$ do   
24 if $g _ { \boldsymbol { \theta } } = 0$ and $\theta \not \in \Theta$ then   
25 accumulate $\begin{array} { r } { \mathbf { \dot { Y } } _ { \theta } \gets \frac { 1 } { \sqrt { d } } \mathbf { K } _ { \theta } ^ { \intercal } \mathbf { Q } _ { w } \in \mathbb { R } ^ { b _ { k } \times b _ { q } } } \end{array}$ in high precision   
26 compute in-place $\hat { \mathbf { E } } _ { \theta } \gets \exp ( \mathbf { Y } _ { \theta } - \mathbf { 1 } _ { b _ { k } } \hat { \pmb { \mu } } _ { w } ^ { \top } )$ in high precision   
27 update $\pmb { \mu } _ { w }  \hat { \pmb { \mu } } _ { w }$ and $\omega _ { w } \gets \hat { \pmb { \alpha } } + [ \| \hat { \mathbf e } _ { 1 } \| _ { 1 } \cdot \cdot \cdot \| \hat { \pmb { \mathrm { e } } } _ { b _ { q } } \| _ { 1 } ] ^ { \intercal }$   
28 update $\mathbf { O } _ { w } \gets \mathbf { O } _ { w } \mathrm { d i a g } ( \hat { \mathbf { \xi } } ) + \mathbf { V } \hat { \mathbf { E } }$   
29 scale $\mathbf { O }  \mathbf { O } \mathrm { d i a g } ( \omega ) ^ { - 1 }$ and write into $\scriptstyle { v _ { q } }$ th submatrix of $\mathbf { O } _ { f u l l }$ in HBM

## 3.3 HARDWARE-ALGORITHM CO-DESIGN

In this section, we map the pseudocode of Algorithm 1 to the hardware of a hypothetical dedicated AI accelerator that would be able to execute it efficiently.

Floating-point formats. We propose the 8-bit e 4m3 format for the low-precision accumulation of key-query matrix products. However, this choice poses a strict upper limit of 448 on the maximum allowed absolute value of a pre-softmax logit to avoid overflow. To guarantee this bound, we require that a transformer employ headwise QK-Norm with moderate gain weights, e.g., as in Qwen3 (Yang et al., 2025) and Gemma 3 (Kamath et al., 2025) models. For a typical head dimension of $d = 1 2 8 ,$ the maximum attainable value of pre-softmax logit is $8 \sqrt { 2 } \times \mathrm { m a x g a i n ^ { 2 } }$ . Thus, as long as the $1 / \sqrt { d }$ scaling is applied before the matrix product is accumulated, the condition maxgain $\leq 6 . 2 9$ prevents overflow during e 4m3 accumulation. To execute LampAttention efficiently, a dedicated accelerator needs to support native packed e4m3 accumulation, whereby 4 accumulators occupy a single 32-bit register. The swamping due to e 4m3 accumulation is typically benign for Algorithm 1: it primarily affects “unimportant" sub-blocks where the resulting exponential probabilities are vanishingly small, thus contributing no meaningful error to the final attention output. To mitigate its potential adversary effects, stochastic rounding (Croci et al., 2022) could be used instead of round-to-nearest.

Next, the logits are shifted by the running maximum, guaranteeing that the shifted logits are strictly non-positive. To account for the predetermined sign and the doubled absolute-value limit of 896, we propose to compute the shifted logits in-place using an unsigned ue 5m3 format.

The exponentials of the shifted logits cannot exceed one. Therefore, we can use the ue 5m3 format again—but with an adapted dynamic range—to compute the exponentials in-place. Namely, we can choose the exponent bias to map the bit pattern $1 1 1 1 1 0 0 0$ to 1.0, which yields the smallest normalized number $2 ^ { - 3 0 }$ and the smallest subnormal number $2 ^ { - 3 3 }$ . Crucially, as both the exponentials and their operands are restricted to 8 bits, the evaluation can be physically implemented as a look-up table (LUT) and executed in one clock cycle, bypassing the multi-cycle special-function unit (SFU).

To carry out high-precision recomputations, we recall that the upper bound on the pre-softmax logits remains fixed. Therefore, we propose using a 16-bit e 4m11 format to accumulate matrix products, maximizing mantissa width, and an unsigned ue5m11 format to evaluate the shifted exponentials. In this case, the accelerator must support native packed e4m11 accumulation with two accumulators per register and must route the evaluation of exponentials through the SFU, since a 16-bit LUT would occupy a prohibitive amount of silicon space. In addition, while the running maxima of LampAttention require e4m11 storage, they can be seamlessly rounded to e4m3 for the low-precision shifts by truncating the lower mantissa bits.

Finally, to compute the attention output, the accelerator must support native matrix products where one operand is stored in the ue5m3 or ue5m11 format with the modified exponent bias.

Register layout. To execute the inner loop over w in Algorithm 1, each warp must have a sufficient number of registers for both the initial 8-bit results and their 16-bit refinements. A “safe" layout, which aligns perfectly with the pseudocode, is to reserve enough registers to store all $N _ { k }$ sub-blocks of Y in high precision. While this layout accommodates the worst-case scenario, it allows only two 8-bit accumulators to reside in a register, leaving space for potential recomputations, and therefore half of the register space is wasted in the best-case scenario without any recomputations.

In practice, softmax probability distributions tend to be highly concentrated for pre-trained models, and we thus expect the recomputations to be sparse. For a“compact"layout, we propose to separate 8-bit and 16-bit storage in the register space. Namely, sufficient registers are reserved for the entire low-precision Y with four 8-bit accumulators per register without padding. Furthermore, we require additional register space to store a single high-precision sub-block of Y. In those instances where LampAttention marks two or more sub-blocks for recomputation, the algorithm should branch to a sequential, fully 16-bit update of $\mathbf { O } _ { w } ,$ reusing the registers for $\lfloor N _ { k } / 2 \rfloor + 1$ high-precision sub-blocks.

Sources of speedup. The hardware-algorithm co-design of LampAttention serves to maximize the speedup relative to vanilla FlashAttention-2. Independent of the register layout, the primary latency reduction stems from the exponentials: evaluating an 8-bit exponential in one clock cycle via a LUT is drastically more efficient than routing a high-precision value through a multi-cycle $\mathrm { S F U . ^ { 4 } }$ This operation-level efficiency is present in both the “safe" and “compact" layouts. At the same time, the identification of salient sub-blocks and their recomputation introduces additional latency; while we purposefully omit warp-group specialization in the present paper, it could be used to mask this latency overhead (Shah et al., 2024).

In contrast, the “compact" layout improves computational throughput by reducing register pressure. Packing four 8-bit accumulators into a single 32-bit register reduces the accumulator footprint by 75%. This enables a two-step hardware optimization: first, tile dimensions can be scaled up to the physical SRAM limit to minimize HBM reads; second, the remaining register savings increase warp occupancy to better hide memory-transfer latency. Provided that “disruptive" recomputations of two or more sub-blocks remain infrequent, this dual scaling can potentially yield consistent speedups.

## 4 NUMERICAL EXPERIMENTS

## 4.1 SETUP AND IMPLEMENTATION

We implement LampAttention as a Triton kernel with simulated 8-bit and 16-bit arithmetic. Namely, the corresponding numbers are stored in FP32 with mantissas truncated to 3 and 11 bits, respectively. The low-precision accumulation of “atomic" matrix products is carried out via recursive summation with FMA, i.e., based on round(c + a · b), where the scalar multiplication and addition are in FP32. Low-precision exponentials are modeled as correctly rounded, round(exp a). The tile sizes are fixed as $b _ { q } N _ { q } = 1 2 8$ and $b _ { k } N _ { k } = 6 4$ with $N _ { k } = 4$ key sub-blocks.

To validate the performance of LampAttention, we evaluate it on models from the Qwen3 (8B, 30B-MoE, 32B) and Gemma 3 (12B, 27B) families. The LampAttention kernel is dynamically injected into the standard HuggingFace implementations at runtime. We compare this computation against two fixed-precision baselines: the vanilla 32-bit baseline and an 8-bit LampAttention baseline restricted from utilizing any 16-bit recomputations. We stress that these bit-widths refer strictly to the attention-logit accumulators and their exponentials, rather than the quantization of model weights. Task metrics are measured using the EleutherAI lm-evaluation-harness.

The code used for the experiments is publicly available.⁵ The experiments were performed on a cloud-based GPU equipped with 80 GB of VRAM and required approximately 540 GPU-hours.

## 4.2 NUMERICAL RESULTS

Tables 1 and 2 show that the adaptive 16-bit recomputations of LampAttention substantially improve upon the pure 8-bit baseline across both perplexity (C4) and downstream (MMLU) tasks; two more tasks are evaluated in Appendix B. Even with a high threshold of $\tau = 0 . 5$ in (8), i.e., with a small number of recomputations during the second stage of Algorithm 1, the recovered metrics broadly approach the 32-bit vanilla baselines (although the degree of closeness remains model- and task-dependent). Crucially, the recovery is achieved with only about 20-30% of key tiles being “disruptive"for the “compact"layout, requiring the recomputation of two or more sub-blocks.

Table 1: LampAttention with $\delta = 2 ^ { - 8 }$ and $\tau = 2 ^ { - 1 }$ on C4.
<table><tr><td rowspan="2">Model name</td><td colspan="3">Perplexity (↓)</td><td colspan="3">16-bit sub-blocks per tile</td></tr><tr><td>8-bit</td><td>8/16-bit LAMP</td><td>32-bit</td><td>0</td><td>1</td><td>2+</td></tr><tr><td>gemma-3-27b-pt</td><td>21.470</td><td>17.150</td><td>16.540</td><td>46.30%</td><td>20.52%</td><td>33.18%</td></tr><tr><td>gemma-3-12b-pt</td><td>36.740</td><td>21.830</td><td>18.680</td><td>50.94%</td><td>20.30%</td><td>28.76%</td></tr><tr><td>Qwen3-32B</td><td>28.180</td><td>23.740</td><td>23.800</td><td>65.09%</td><td>15.43%</td><td>19.48%</td></tr><tr><td>Qwen3-8B</td><td>44.030</td><td>33.780</td><td>33.340</td><td>66.19%</td><td>14.58%</td><td>19.23%</td></tr><tr><td>Qwen3-30B-A3B</td><td>43.900</td><td>28.270</td><td>27.510</td><td>69.75%</td><td>13.71%</td><td>16.54%</td></tr></table>

Furthermore, Figure 1 depicts the empirical efficiency-accuracy trade-off of LampAttention across varying thresholds. As τ decreases, task performance converges toward the 32-bit vanilla baseline, though non-monotonically for smaller values of τ. This observation indicates that enforcing higher precision on intermediate computations does not unconditionally guarantee superior metrics; in fact, Table 4 in Appendix B illustrates an instance where the 8-bit baseline accuracy exceeds the 32-bit baseline. In contrast, the rate of “disruptive" recomputations increases monotonically as τ tends to zero, reflecting the increasingly stringent constraints of the block-LAMP optimization problem (8).

Table 2: LampAttention with $\delta = 2 ^ { - 8 }$ and $\tau = 2 ^ { - 1 }$ on 5-shot MMLU.
<table><tr><td rowspan="2">Model name</td><td colspan="3">Accuracy (↑)</td><td colspan="3">16-bit sub-blocks per tile</td></tr><tr><td>8-bit</td><td>8/16-bit LAMP</td><td>32-bit</td><td>0</td><td>1</td><td>2+</td></tr><tr><td>gemma-3-27b-pt</td><td>0.6261</td><td>0.7851</td><td>0.7993</td><td>48.49%</td><td>21.55%</td><td>29.96%</td></tr><tr><td>gemma-3-12b-pt</td><td>0.3618</td><td>0.5899</td><td>0.7654</td><td>53.39%</td><td>22.19%</td><td>24.42%</td></tr><tr><td>Qwen3-32B</td><td>0.7555</td><td>0.8355</td><td>0.8344</td><td>54.52%</td><td>19.67%</td><td>25.81%</td></tr><tr><td>Qwen3-8B</td><td>0.5625</td><td>0.7412</td><td>0.7840</td><td>54.05%</td><td>19.11%</td><td>26.84%</td></tr><tr><td>Qwen3-30B-A3B</td><td>0.5998</td><td>0.7884</td><td>0.8169</td><td>56.97%</td><td>18.99%</td><td>24.04%</td></tr></table>

Figure 1 also provides a new perspective on the results in Tables 1- 2. We observe that the empirical difference between $\tau = 1$ and $\tau = 0 . { \stackrel { \ r } { . } }$ is marginal in terms of both recomputation rate and accuracy. Because the optimal solution to (8) at $\tau = 1$ strictly omits stage-two recomputations, we can deduce the structural division of labor within the algorithm:

• the first, running-maximum stage of LampAttention is responsible for the “heavy lifting" of the evaluation metrics from the 8-bit baseline toward the 32-bit baseline using a moderate budget of “disruptive" recomputations for the compact register layout;

• the second stage, solving (8), acts as a fine-grained correction mechanism that closes the final metric gap at the cost of additional recomputations.

Finally, Figure 2 visualizes the dynamics of the recomputation cascade as the threshold τ tightens: the mass of zero-recomputation cases systematically transfers into full-recomputation cases.

Additional experimental results are presented in the Appendix.

Our findings highlight the potential utility of incorporating the new mixed-precision paradigm, the LAMP, into FlashAttention. They expose a controllable trade-off between inference efficiency and downstream performance without resorting to the 32-bit arithmetic for the computationally intensive sections of the algorithm.

C4 | Perplexity (↓)  
5-shot MMLU | Accuracy (↑)  
![](images/1077786296bf8a6bb160c9770aaa566abb31b9f9d3b91e0b1608fb20dbdf3ae3.jpg)  
Figure 1: LampAttention with $\delta = 2 ^ { - 8 }$ and $\tau \in \{ 2 ^ { - t }$ $t = 0 , \ldots , 5 \}$

![](images/74656ab1c252a5a12e170a510d6c394b39a0415d517e4597cbec78604bb3d2e8.jpg)  
Figure 2: LampAttention with $\delta = 2 ^ { - 8 }$ and $\tau \in \{ 2 ^ { - t }$ $t = 0 , \ldots , 5 \}$

## AUTHOR CONTRIBUTIONS

SB conceived the approach, formulated the research problem, and carried out the formal analysis, experimentation, and implementation. SB wrote the original draft of the manuscript. MG and TY contributed to the analysis and implementation and assisted with proofreading. YHT contributed to project administration and to reviewing and editing the manuscript. YL, WF, and FW contributed to project administration. PP supervised the project as laboratory head and reviewed the manuscript

## ACKNOWLEDGMENTS

This work was carried out in the framework of a research project funded by Huawei Technologies Ltd.

## AI USE STATEMENT

In this work, we used generative AI tools to assist with software development and to refine the clarity and flow of the text. All core research ideation, theoretical development, experimental design, and data analysis were conducted entirely by the authors. We have reviewed all AI-assisted work and assume full responsibility for the accuracy, integrity, and originality of this work.

## REFERENCES

Ahmad Abdelfattah, Jack Dongarra, Massimiliano Fasi, Mantas Mikaitis, and Françoise Tisseur. Analysis of floating-point matrix multiplication computed via integer arithmetic. arXiv, art. 2506.11277, 2025. doi: 10.48550/arXiv.2506.11277.

Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit Sanghai. GQA: Training generalized multi-query transformer models from multi-head checkpoints. In EMNLP, pp. 4895–4901, 2023. doi: 10.18653/v1/2023.emnlp-main.298.

Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. arXiv, art. 2004.05150, 2020. doi: 10.48550/arXiv.2004.05150.

Pierre Blanchard, Nicholas J Higham, Florent Lopez, Theo Mary, and Srikara Pranesh. Mixed precision block fused multiply-add: Error analysis and application to gpu tensor cores. SIAM J Sci Comput, 42(3):C124–C141, 2020. doi: 10.1137/19M1289546.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In NeurIPS, pp. 1877–1901, 2020. doi: 10.48550/arXiv.2005.14165.

Stanislav Budzinskiy, Wenyi Fang, Longbin Zeng, and Philipp Petersen. Numerical stability analysis of large language models. arXiv, art. 2503.10251, 2025. doi: 10.48550/arXiv.2503.10251.

Stanislav Budzinskiy, Marian Gloser, Tolunay Yilmaz, Ying Hong Tham, Yuanyi Lin, Wenyi Fang, Fan Wu, and Philipp Petersen. LAMP: Look-ahead mixed-precision inference of large language models. arXiv, art. 2601.21623, 2026. doi: 10.48550/arXiv.2601.21623.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. J Mach Learn Res, 24(240):1–113, 2023. doi: 10.48550/arXiv.2204.02311.

Matteo Croci, Massimiliano Fasi, Nicholas J Higham, Theo Mary, and Mantas Mikaitis. Stochastic rounding: implementation, error analysis and applications. R Soc Open Sci, 9(3):211631, 2022. doi: 10.1098/rsos.211631.

Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G Carbonell, Quoc Le, and Ruslan Salakhutdinov. Transformer-XL: Attentive language models beyond a fixed-length context. In ACL, pp. 2978– 2988, 2019. doi: 10.18653/v1/P19-1285.

Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In ICLR, 2024. doi: 10.48550/arXiv.2307.08691.

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memoryefficient exact attention with io-awareness. In NeurIPS, pp. 16344–16359, 2022. doi: 10.5555/ 3600270.3601459.

Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. Gpt3.int8(): 8-bit matrix multiplication for transformers at scale. In NeurIPS, pp. 30318–30332, 2022. doi: 10.48550/ arXiv.2208.07339.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In NAACL-HLT, volume 1, pp. 4171– 4186, 2019. doi: 10.48550/arXiv.1810.04805.

El-Mehdi El Arar, Silviu-Ioan Filip, Theo Mary, and Elisa Riccietti. Mixed precision accumulation for neural network inference guided by componentwise forward error analysis. IMA Journal of Numerical Analysis, art. drag054, 2026. doi: 10.1093/imanum/drag054/8785932.

Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. GPTQ: Accurate post-training quantization for generative pre-trained transformers. In ICLR, 2023. doi: 10.48550/arXiv.2210. 17323.

Guo Guo, Songlin Yang, Tarushii Goel, Eric P Xing, Tri Dao, and Yoon Kim. Log-linear attention. In ICLR, pp. 91296–91317, 2026. doi: 10.48550/arXiv.2506.04761.

Suyog Gupta, Ankur Agrawal, Kailash Gopalakrishnan, and Pritish Narayanan. Deep learning with limited numerical precision. In ICML, volume 37, pp. 1737–1746, 2015. doi: 10.48550/arXiv. 1502.02551.

Nicholas J Higham. Accuracy and Stability of Numerical Algorithms. SIAM, 2 edition, 2002. doi: 10.1137/1.9780898718027.

John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Žídek, Anna Potapenko, Alex Bridgland, Clemens Meyer, Simon A. A. Kohl, Andrew J. Ballard, Andrew Cowie, Bernardino Romera-Paredes, Stanislav Nikolov, Rishub Jain, Jonas Adler, Trevor Back, Stig Petersen, David Reiman, Ellen Clancy, Michal Zielinski, Martin Steinegger, Michalina Pacholska, Tamas Berghammer, Sebastian Bodenstein, David Silver, Oriol Vinyals, Andrew W. Senior, Koray Kavukcuoglu, Pushmeet Kohli, and Demis Hassabis. Highly accurate protein structure prediction with AlphaFold. Nature, 596(7873):583–589, 2021. doi: 10.1038/s41586-021-03819-2.

Dhiraj Kalamkar, Dheevatsa Mudigere, Naveen Mellempudi, Dipankar Das, Kunal Banerjee, Sasikanth Avancha, Dharma Teja Vooturi, Nataraj Jammalamadaka, Jianyu Huang, Hector Yuen, et al. A study of BFLOAT16 for deep learning training. arXiv, art. 1905.12322, 2019. doi: 10.48550/arXiv.1905.12322.

Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, S Perrin, T Matejovicova, A Ramé, M Rivière, L Rouillard, et al. Gemma 3 technical report. arXiv, art. 2503.19786, 2025. doi: 10.48550/arXiv.2503.19786.

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In ICML, pp. 5156–5165, 2020. doi: 10.48550/arXiv.2006.16236.

Hans Kellerer, Ulrich Pferschy, and David Pisinger. Knapsack Problems. Springer, 2004. doi: 10.1007/978-3-540-24777-7.

Nikita Kitaev, Łukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. In ICLR, 2020. doi: 10.48550/arXiv.2001.04451.

Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Xingyu Dang, and Song Han. AWQ: Activationaware weight quantization for on-device LLM compression and acceleration. In MLSys, volume 6, pp. 87–100, 2024. doi: 10.48550/arXiv.2306.00978.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 Technical report. arXiv, art. 2412.19437, 2024. doi: 10.48550/arXiv.2412.19437.

Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, et al. Mixed precision training. In ICLR, 2018. doi: 10.48550/arXiv.1710.03740.

Paulius Micikevicius, Dusan Stosic, Neil Burgess, Marius Cornea, Pradeep Dubey, Richard Grisenthwaite, Sangwon Ha, Alexander Heinecke, Patrick Judd, John Kamalu, et al. FP8 formats for deep learning. arXiv, art. 2209.05433, 2022. doi: 10.48550/arXiv.2209.05433.

Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. FlashAttention-3: Fast and accurate attention with asynchrony and low-precision. In NeurIPS, pp. 68658–68685, 2024. doi: 10.5555/3737916.3740109.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. RoFormer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. doi: 10.1016/j.neucom.2023.127063.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv, art. 2302.13971, 2023. doi: 10.48550/arXiv.2302. 13971.

Trieu H Trinh, Yuhuai Wu, Quoc V Le, He He, and Thang Luong. Solving olympiad geometry without human demonstrations. Nature, 625(7995):476–482, 2024. doi: 10.1038/ s41586-023-06747-5.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NIPS, pp. 5998–6008, 2017. doi:10.48550/arXiv.1706.03762.

Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. SmoothQuant: Accurate and efficient post-training quantization for large language models. In ICML, pp. 38087– 38099, 2023. doi:10.48550/arXiv.2211.10438.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv, art. 2505.09388, 2025. doi: 10.48550/arXiv.2505.09388.

Ted Zadouri, Markus Hoehnerbach, Jay Shah, Timmy Liu, Vijay Thakkar, and Tri Dao. FlashAttention-4: Algorithm and kernel pipelining co-design for asymmetric hardware scaling. In MlSys, 2026. doi: 10.48550/arXiv.2603.05451.

## A PROOF OF PROPOSITION 1

The $\ell _ { 1 }$ normalization function η is differentiable when every entry of z is nonzero; in our case, they are positive. With $\pmb { \eta } = \pmb { \eta } ( \mathbf { z } )$ , direct calculation yields

$$
\mathbf { A } = \mathbf { J } _ { \eta } ( \mathbf { z } ) \mathrm { d i a g } ( \mathbf { z } ) \mathrm { d i a g } ( \mathbf { h } ) = ( \mathbf { I } - \eta \mathbf { 1 } ^ { \intercal } ) \mathrm { d i a g } ( \eta ) \mathrm { d i a g } ( \mathbf { h } ) . 
$$

Since $\eta _ { i } \in ( 0 , 1 )$ for each ¿, the diagonal entries of A are nonnegative, and hence

$$
\big | \mathbf { e } _ { l } ^ { \mathsf { T } } \mathbf { A } \big | \hat { \mathbf { w } } \leq \mathbf { e } _ { l } ^ { \mathsf { T } } \mathbf { A } \mathbf { 1 } \big \| \hat { \mathbf { w } } \big \| _ { \infty } = \Big [ \eta _ { l } \sum _ { i \neq l } \eta _ { i } h _ { i } + ( 1 - \eta _ { l } ) \eta _ { l } h _ { l } \Big ] \big \| \hat { \mathbf { w } } \big \| _ { \infty } = \eta _ { l } \Big [ \mathbf { h } ^ { \mathsf { T } } \eta + h _ { l } ( 1 - 2 \eta _ { l } ) \Big ] \big \| \hat { \mathbf { w } } \big \| _ { \infty } .
$$

## B ADDITIONAL BENCHMARKS

In addition to the C4 and MMLU tasks presented in the main text, we have evaluated the performance of LampAttention on Wikitext and ARC-Challenge; see Tables 3 and 4. Notably, the 8-bit accuracy baseline of Qwen3-32B on ARC-Challenge is higher than the corresponding 32-bit baseline, indicating that there is no strict causal relationship between higher-precision intermediate calculations and higher final accuracy. Figure 3 is an extension of Figure 2 that contains all models and tasks.

Table 3: LampAttention with $\delta = 2 ^ { - 8 }$ and $\tau = 2 ^ { - 1 }$ on Wikitext.
<table><tr><td rowspan="2">Model name</td><td colspan="3">Perplexity (↓)</td><td colspan="3">16-bit sub-blocks per tile</td></tr><tr><td>8-bit</td><td>8/16-bit LAMP</td><td>32-bit</td><td>0</td><td>1</td><td>2+</td></tr><tr><td>gemma-3-27b-pt</td><td>8.6290</td><td>6.2100</td><td>5.9000</td><td>47.21%</td><td>19.86%</td><td>32.93%</td></tr><tr><td>gemma-3-12b-pt</td><td>15.940</td><td>8.6370</td><td>7.3790</td><td>52.86%</td><td>19.10%</td><td>28.04%</td></tr><tr><td>Qwen3-32B</td><td>12.170</td><td>9.4290</td><td>9.3610</td><td>76.50%</td><td>11.55%</td><td>11.95%</td></tr><tr><td>Qwen3-8B</td><td>17.480</td><td>12.510</td><td>12.300</td><td>76.44%</td><td>11.01%</td><td>12.55%</td></tr><tr><td>Qwen3-30B-A3B</td><td>18.780</td><td>11.260</td><td>10.890</td><td>81.82%</td><td>9.02%</td><td>9.16%</td></tr></table>

Table 4: LampAttention with $\delta = 2 ^ { - 8 }$ and $\tau = 2 ^ { - 1 }$ on 0-shot ARC-Challenge.
<table><tr><td rowspan="2">Model name</td><td colspan="3">Accuracy (↑)</td><td colspan="3">16-bit sub-blocks per tile</td></tr><tr><td>8-bit</td><td>8/16-bit LAMP</td><td>32-bit</td><td>0</td><td>1</td><td>2+</td></tr><tr><td>gemma-3-27b-pt</td><td>0.5977</td><td>0.6484</td><td>0.6562</td><td>48.47%</td><td>30.41%</td><td>21.12%</td></tr><tr><td>gemma-3-12b-pt</td><td>0.4570</td><td>0.6191</td><td>0.6387</td><td>48.71%</td><td>32.21%</td><td>19.08%</td></tr><tr><td>Qwen3-32B</td><td>0.6191</td><td>0.6016</td><td>0.6094</td><td>48.54%</td><td>31.22%</td><td>20.24%</td></tr><tr><td>Qwen3-8B</td><td>0.4766</td><td>0.5352</td><td>0.5605</td><td>48.24%</td><td>28.71%</td><td>23.05%</td></tr><tr><td>Qwen3-30B-A3B</td><td>0.4512</td><td>0.5176</td><td>0.5762</td><td>48.28%</td><td>28.94%</td><td>22.78%</td></tr></table>

![](images/785d5f6545ed1e8f16dd553015776f3d9627c85b89e1a7358be0b0214ca8c117.jpg)  
Figure 3: LampAttention with $\delta = 2 ^ { - 8 }$ and $\tau \in \{ 2 ^ { - t }$ $t = 0 , \ldots , 5 \}$

## C IMPACT OF SAFETY MARGIN

Here, we compare the performance of LampAttention with two different values of the safety margin δ from the first stage of the algorithm. As Figures 4 and 5 show, larger δ results in more “disruptive" recomputations being initiated during the first stage (which τ = 1 isolates). At the same time, the two trade-off curves exhibit similar convergent behavior as the threshold τ tightens.

Wikitext | Perplexity (↓)  
C4 | Perplexity (↓)  
![](images/2e069cee18e3a9a9a6fb8d74800db9dcc4f1d9c55ebb1fea26aea99d42616c64.jpg)  
Figure 4: LampAttention with $\delta \in \{ 2 ^ { - 8 } , 2 ^ { - 4 } \}$ and $\tau \in \{ 2 ^ { - t }$ $t = 0 , \ldots , 5 \}$

![](images/a7e458109ac6be5db7d1483efea28670c59a17396a7f7e7cdecdadcae7f0b96e.jpg)  
Figure 5: LampAttention with $\delta \in \{ 2 ^ { - 8 } , 2 ^ { - 4 } \}$ and $\tau \in \{ 2 ^ { - t }$ $t = 0 , \ldots , 5 \}$

## D IMPACT OF FIRST-STAGE RECOMPUTATION

While setting $\tau = 1$ effectively disables the second recomputation stage in Algorithm 1, we can choose to disable the first stage instead (even though this violates the theoretical assumptions underlying the derivation of the block-LAMP problem (8)). Figures 6 and 7 show the corresponding trade-off curves. Taken together with the results in Appendix C, they suggest that the specific mechanism triggering a recomputation is secondary; whether driven by the first-stage running maximum or the second-stage threshold, both phases identify fundamentally “important" sub-blocks

Wikitext | Perplexity (↓)  
C4 | Perplexity (↓)  
![](images/534cc2fde12b1afb886d1e7b2987f7527770d15742dac847c890c939e07c2ea0.jpg)  
Figure 6: LampAttention with $\delta = 2 ^ { - 8 }$ or disabled first stage, and $\tau \in \{ 2 ^ { - t } : t = 0 , \ldots , 5 \}$

![](images/3b6cf999af639225aa341238fb5e7501885cbec40b1ea1cfca1fc97036c90cab.jpg)  
Figure 7: LampAttention with $\delta = 2 ^ { - 8 }$ or disabled first stage, and $\tau \in \{ 2 ^ { - t } : t = 0 , \ldots , 5 \}$

## E IMPACT OF IN-CONTEXT LEARNING

We assess how task familiarity affects the performance of LampAttention by comparing 0-shot and 25-shot evaluations on ARC-Challenge. Tables 5 and 6 (a reproduction of Table 4 for convenience) demonstrate a dual benefit: the few-shot setting naturally increases the accuracy while simultaneously expanding the share of tiles processed entirely in 8-bit. Interestingly, Figure 8 reveals that the distribution of recomputed sub-blocks per tile evolves distinctively between the two task instances.

Table 5: LampAttention with $\delta = 2 ^ { - 8 }$ and $\tau = 2 ^ { - 1 }$ on 25-shot ARC-Challenge.
<table><tr><td rowspan="2">Model name</td><td colspan="3">Accuracy (↑)</td><td colspan="3">16-bit sub-blocks per tile</td></tr><tr><td>8-bit</td><td>8/16-bit LAMP</td><td>32-bit</td><td>0</td><td>1</td><td>2+</td></tr><tr><td>gemma-3-27b-pt</td><td>0.6504</td><td>0.6953</td><td>0.7070</td><td>56.64%</td><td>19.37%</td><td>23.99%</td></tr><tr><td>gemma-3-12b-pt</td><td>0.4609</td><td>0.6289</td><td>0.6777</td><td>62.56%</td><td>19.31%</td><td>18.13%</td></tr><tr><td>Qwen3-32B</td><td>0.6758</td><td>0.7305</td><td>0.7324</td><td>55.23%</td><td>19.55%</td><td>25.22%</td></tr><tr><td>Qwen3-8B</td><td>0.5488</td><td>0.6660</td><td>0.6660</td><td>54.10%</td><td>18.53%</td><td>27.37%</td></tr><tr><td>Qwen3-30B-A3B</td><td>0.5879</td><td>0.7031</td><td>0.6953</td><td>57.72%</td><td>19.01%</td><td>23.27%</td></tr></table>

Table 6: LampAttention with $\delta = 2 ^ { - 8 }$ and $\tau = 2 ^ { - 1 }$ on 0-shot ARC-Challenge.
<table><tr><td rowspan="2">Model name</td><td colspan="3">Accuracy (↑)</td><td colspan="3">16-bit sub-blocks per tile</td></tr><tr><td>8-bit</td><td>8/16-bit LAMP</td><td>32-bit</td><td>0</td><td>1</td><td>2+</td></tr><tr><td>gemma-3-27b-pt</td><td>0.5977</td><td>0.6484</td><td>0.6562</td><td>48.47%</td><td>30.41%</td><td>21.12%</td></tr><tr><td>gemma-3-12b-pt</td><td>0.4570</td><td>0.6191</td><td>0.6387</td><td>48.71%</td><td>32.21%</td><td>19.08%</td></tr><tr><td>Qwen3-32B</td><td>0.6191</td><td>0.6016</td><td>0.6094</td><td>48.54%</td><td>31.22%</td><td>20.24%</td></tr><tr><td>Qwen3-8B</td><td>0.4766</td><td>0.5352</td><td>0.5605</td><td>48.24%</td><td>28.71%</td><td>23.05%</td></tr><tr><td>Qwen3-30B-A3B</td><td>0.4512</td><td>0.5176</td><td>0.5762</td><td>48.28%</td><td>28.94%</td><td>22.78%</td></tr></table>

![](images/455de3bff8a57ad30ee070995b9013b162ba6b4e48b8204be61b260b144b403f.jpg)  
Figure 8: LampAttention with $\delta = 2 ^ { - 8 }$ and $\tau \in \{ 2 ^ { - t }$ : t = 0, . . . , 5}.