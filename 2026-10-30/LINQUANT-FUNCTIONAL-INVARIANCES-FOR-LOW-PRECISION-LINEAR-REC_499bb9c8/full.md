# LINQUANT: FUNCTIONAL INVARIANCES FOR LOW-PRECISION LINEAR RECURRENCES

Neehal Tumma<sup>\*</sup> <sup>1</sup> Philipp Nazari<sup>\*</sup> <sup>2</sup> <sup>3</sup> <sup>4</sup> <sup>5</sup> Rohan Cherukuri<sup>1</sup>

Daniela Rus<sup>1</sup> T. Konstantin Rusch<sup>2</sup> <sup>4</sup> <sup>5</sup>

<sup>1</sup>MIT CSAIL <sup>2</sup>Max Planck Institute for Intelligent Systems <sup>3</sup>ETH Zurich¨ <sup>4</sup>Tubingen AI Center¨ <sup>5</sup>ELLIS Institute Tubingen¨

## ABSTRACT

Linear recurrences are emerging as practical alternatives to softmax attention in large language models, appearing in both standalone and hybrid architectures. However, despite their rapid emergence, low-precision inference remains much less developed for these models than for softmax attention. During prefill, chunkwiseparallel algorithms cast much of the recurrence as dense matrix multiplications, suggesting a natural path to low-precision execution. Simply quantizing these operations in their native form, however, can be fragile. In this work, we identify functional invariances that preserve the full-precision computation while making the operands more amenable to quantization. Building on this observation, we introduce LinQuant, a framework for low-precision recurrent prefill based on seven such invariances, including rotations, diagonal rescalings, and alternative placements of recurrent decay and write factors. We derive when these transformations remain valid across common recurrence families spanning linear, gated, and delta-rule updates, and show that they provide complementary degrees of freedom for improving low-precision accuracy. Finally, we translate this framework to production by augmenting SSDi8, an existing low-precision prefill pipeline for Mamba-2, with the applicable invariances. LinQuant improves perplexity at matched latency on Mamba-2 1.3B, 2.7B, and 8B and Nemotron-H 8B, demonstrating the practical utility of invariance-based quantization for recurrent models.

§ <sub>g</sub>ithub.com/ntumm120/LinQuant

## 1 INTRODUCTION

Linear recurrences offer an efficient, sub-quadratic alternative to softmax attention for sequence modeling (Katharopoulos et al., 2020; Dao & Gu, 2024; Yang et al., 2025; Team et al., 2025). They summarize past tokens in a fixed-size state, avoiding the growing key-value cache of softmax attention, and are increasingly used in both standalone and hybrid language models (NVIDIA, 2025; Amini et al., 2025; Peng et al., 2025; Beck et al., 2024). For prefill, these models are typically evaluated using chunkwise-parallel algorithms, which process blocks of tokens in parallel through dense matrix multiplications while propagating a recurrent state between them (Dao & Gu, 2024; Yang & Zhang, 2024). This mirrors the blockwise organization of efficient attention kernels such as FlashAttention and makes recurrent prefill naturally amenable to low-precision execution.

Low-precision attention has been studied extensively, with optimized integer and floating-point kernels now forming a mature part of the Transformer inference stack (Zhang et al., 2025; Shah et al., 2024). In contrast, low-precision kernels for linear recurrences remain largely unexplored. Prior work has studied quantization of recurrent models (Xu et al., 2025; Chiang et al., 2025), largely targeting the projections and activations surrounding the Mamba-2 recurrence. SSDi8, on the other hand, executes the recurrent computation in low precision, demonstrating an efficient INT8 implementation specific to Mamba-2 (Kim et al., 2026). We instead study the space more generally, investigating low-precision computation across the design space of modern linear recurrent architectures.

Table 1: Applicability of the seven invariances introduced in Section 3 to common linear recurrences. ×: not applicable. ✓: applicable. $( \checkmark )$ : applicable under constraints.
<table><tr><td>Nr. Transformation</td><td>LA</td><td>Mamba-2</td><td>GLA</td><td>DN</td><td>GDN</td><td>KDA</td></tr><tr><td>(1/7)  $\mathbf { R } _ { Q K }$ </td><td>V</td><td>√</td><td>(√)</td><td>√</td><td>√</td><td> $( \checkmark )$ </td></tr><tr><td>(2/7)  $ { \mathbf Ḋ Q Ḍ } _ { K }$ </td><td>V</td><td>V</td><td>V</td><td>X</td><td>X</td><td>X</td></tr><tr><td>(3/7)  $\mathbf { R } _ { V }$ </td><td>V</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>(4/7)  $ { \mathbf Ḋ \ Ḍ } _ { V }$ </td><td>V</td><td>V</td><td>V</td><td>V</td><td>V</td><td>V</td></tr><tr><td>(5/7)  $( ( \mathbf { \Gamma } _ { [ i ] } ^ {  } ) ^ { \eta } , \mathbf { \Gamma } ( \mathbf { \Gamma } _ { [ i ] } ^ {  } ) ^ { 1 - \eta } )$ </td><td>X</td><td>√</td><td>(√)</td><td>X</td><td>√</td><td>(√)</td></tr><tr><td>(6/7)  $\left( \mathbf { E } _ { [ i ] } ^ { \eta } , \mathbf { E } _ { [ i ] } ^ { 1 - \eta } \right)$ </td><td>×</td><td>√</td><td>√</td><td>X</td><td>√</td><td>√</td></tr><tr><td>(7/7)  $\bigl ( \mathbf { B } _ { [ i ] } ^ { \xi } , \ \mathbf { B } _ { [ i ] } ^ { 1 - \xi } \bigr )$ </td><td>(√)</td><td>(√)</td><td>(√)</td><td>√</td><td>√</td><td>V</td></tr></table>

Executing sequence mixers in low-precision can come at the cost of performance. One powerful lever to mitigate this isfunctional invariances. Such transformations preserve the full-precision operator while changing the numerics presented to a quantizer. Figure 12 in the appendix illustrates this principle through write-factor absorption. We distinguish two broad classes. Mixer-agnostic invariances act on the residual stream or individual linear layers and do not depend on the algebra of the sequence operator (Xiao et al., 2023; Ashkboos et al., 2024; Liu et al., 2025). Because these transformations are largely independent of the mixer, they transfer naturally to recurrent architectures and have already been adopted by methods such as MambaQuant and Quamba2 (Xu et al., 2025; Chiang et al., 2025). Mixer-specific invariances instead rely on the sequence operator itself, and can be classified as either mixer-external or mixer-internal. The former materialize as changes of basis and can be applied outside of the sequence-mixer, while the latter are functions of a recurrence’s algorithmic realization (Figure 1). Transformer PTQ has exploited such invariances within softmax attention, for example through transformations of its query, key, and value representations (Ashkboos et al., 2024; Liu et al., 2025). In contrast, the mixer-specific invariances of linear recurrences have yet to be characterized.

Linear recurrences present a qualitatively different invariance structure from softmax attention. Their transformations must remain consistent with a state that is updated and carried across chunks, while decay, gating, and delta-rule updates impose additional algebraic constraints. At the same time, these same mechanisms create new degrees of freedom that have no direct attention analogue. We therefore ask which attention-inspired invariances survive in recurrences, what new restrictions arise from the recurrent computation, and what additional invariances are unique to linear recurrences. Our main contributions are summarized as follows:

• We introduce LinQuant, a framework for computing low-precision linear recurrences based on both mixer-agnostic and mixer-specific functional invariances, and characterize their applicability across common recurrent architectures.

• Across a controlled set of linear recurrences, we show how recurrence structure, query-key normalization, quantization scale granularity, and state-storage basis determine which functionally equivalent realization is most robust to low precision.

• We implement LinQuant in INT8 chunkwise kernels and fuse query–key transformations into SSDi8 without increasing measured prefill latency.

• We apply LinQuant to production W4A8 models with INT8 recurrent states, improving over SSDi8 on Mamba-2 1.3B, 2.7B, and 8B and Nemotron-H 8B.

## 2 BACKGROUND AND PRELIMINARIES

## 2.1 LINEAR RECURRENCES

Causal softmax attention (Vaswani et al., 2017) computes

$$
\mathbf { o } _ { t } = \mathbf { V } _ { 1 : t } ^ { \top } \mathrm { s o f t m a x } \left( \mathbf { K } _ { 1 : t } \mathbf { q } _ { t } / \sqrt { D _ { k } } \right) ,
$$

from queries and keys $\mathbf { q } _ { t } , \mathbf { k } _ { t } \in \mathbb { R } ^ { D _ { k } }$ and values $\mathbf { v } _ { t } \in \mathbb { R } ^ { D _ { v } }$ , where $D _ { k }$ and $D _ { v }$ denote the state (key/query) and value dimensions, respectively. Linear attention replaces the nonlinear softmax

operator with an unnormalized linear kernel (Katharopoulos et al., 2020). In its simplest form, associativity allows past keys and values to be summarized in a fixed-size matrix-valued state $\mathbf { S } _ { t } \in \mathbb { R } ^ { D _ { v } \times \mathbf { \check { \mathit { D } } } _ { k } }$

$$
\mathbf { S } _ { t } = \mathbf { S } _ { t - 1 } + \mathbf { v } _ { t } \mathbf { k } _ { t } ^ { \top } , \qquad \mathbf { o } _ { t } = \mathbf { S } _ { t } \mathbf { q } _ { t } .
$$

Modern linear recurrences extend this basic update with decay, write gating, and delta-rule corrections (Dao & Gu, 2024; Yang et al., 2024b;a; Team et al., 2025). We capture these axes through the common recurrence

$$
\mathbf { S } _ { t } = \mathbf { S } _ { t - 1 } \mathbf { A } _ { t } \left( \mathbf { I } - \delta \beta _ { t } \mathbf { k } _ { t } \mathbf { k } _ { t } ^ { \top } \right) + \beta _ { t } \mathbf { v } _ { t } \mathbf { k } _ { t } ^ { \top } ,\tag{1}
$$

where $\beta _ { t } > 0 ( \beta _ { t } \in ( 0 , 2 )$ for the delta rule), $\| \mathbf { A } _ { t } \| \leq 1$ , and $\delta \in \{ 0 , 1 \}$ is fixed for a given recurrence.   
Equation 1 separates three structural choices.

Decay controls how the previous state is forgotten before new information is written. A scalar gate $\mathbf { \Lambda } \Lambda _ { t } = \alpha _ { t } \mathbf { I }$ uses one decay rate per recurrent head (Dao & Gu, 2024; Yang et al., 2025), whereas a diagonal gate $\Lambda _ { t } = \mathrm { d i a g } ( \alpha _ { t } )$ allows the decay to vary across state channels (Yang et al., 2024b; Team et al., 2025). Write gating scales the new association $\mathbf { v } _ { t } \mathbf { k } _ { t } ^ { \top }$ . Delta correction (Schlag et al., 2021; Yang et al., 2024c) removes the component of the existing state aligned with the current key before writing the new association. Together, these choices organize the space of modern recurrences as shown in Table 3 in Appendix B.1. We later instantiate this taxonomy as a controlled mixer zoo, varying the recurrent update while holding the surrounding model architecture and training setup fixed.

## 2.2 CHUNKWISE-PARALLEL COMPUTATION

Chunkwise implementations parallelize Equation (1) within chunks of C tokens (Dao & Gu, 2024; Yang & Zhang, 2024; Tumma et al., 2026), while processing chunks sequentially. Chunk [i] contains positions $i C \bar { + } 1 , \ldots , ( i + 1 ) C$ , with incoming state $\begin{array} { r } { \mathbf { S } _ { [ i ] } : = \mathbf { S } _ { i C } , } \end{array}$ , queries/keys $\mathbf { \dot { Q } } _ { [ i ] } , \mathbf { K } _ { [ i ] } \mathbf { \dot { \in } } \mathbb { R } ^ { C \times D _ { k } }$ and values/outputs $\mathbf { V } _ { [ i ] } , \mathbf { O } _ { [ i ] } \ \in \ \mathbb { R } ^ { C \times D _ { v } }$ . Indices $r , s$ denote positions within chunk [i], and n denotes a channel. For example, $\begin{array} { r } { \alpha _ { [ i ] } ^ { r } : = \alpha _ { i C + r } } \end{array}$ and $\beta _ { [ i ] } ^ { r } : = \beta _ { i C + r }$ . We collect the write gates in $\mathbf { B } _ { [ i ] } : = \mathrm { d i a g } ( \beta _ { [ i ] } ^ { 1 } , \dots , \beta _ { [ i ] } ^ { C } ) \in \mathbb { R } ^ { C \times C }$ . Throughout this work, we also refer to matrix multiplications as “contractions”. In the following, we lay out chunkwise parallel formulations for the recurrences covered by Equation (1).

Scalar decay. For no delta correction $( \delta = 0 )$ and scalar decay $\mathbf { \Lambda } \Lambda _ { t } = \alpha _ { t } \mathbf { I }$ , define the cumulative decay $\begin{array} { r } { \gamma _ { \lceil i \rceil } ^ { s \to r } : = \prod _ { u = s + 1 } ^ { r } \alpha _ { \lceil i \rceil } ^ { u } } \end{array}$ and $\gamma _ { \lceil i \rceil } ^ { r } : = \gamma _ { \lceil i \rceil } ^ { 0 \to r }$ . The prefix and tail vectors $\Gamma _ { [ i ] } ^ { \left. } , \Gamma _ { [ i ] } ^ { \right. } \in \mathbb { R } ^ { C }$ have entries $( \Gamma _ { [ i ] } ^ {  } ) _ { r } = \gamma _ { [ i ] } ^ { r }$ and $( \Gamma _ { [ i ] } ^ {  } ) _ { r } = \dot { \gamma } _ { [ i ] } ^ { r  C }$ . Define the causal decay matrix $\mathbf { \Gamma } \mathbf { \Gamma } _ { [ i ] } \in \mathbb { R } ^ { C \times C }$ , with entries $( \mathbf { { r } } _ { [ i ] } ) _ { r s } = \gamma _ { [ i ] } ^ { s  r }$ for $s \leq r$ and zero otherwise. The chunkwise parallel realization then reads (Dao & Gu, 2024)

$$
\begin{array} { r l } & { \mathbf { S } _ { [ i + 1 ] } = \gamma _ { [ i ] } ^ { C } \mathbf { S } _ { [ i ] } + \mathbf { V } _ { [ i ] } ^ { \top } \mathbb { B } _ { [ i ] } ( \mathbf { \Gamma } _ { [ i ] } ^ { \right. } \odot \mathbf { K } _ { [ i ] } ) , } \\ & { \mathbf { \Gamma } \mathbf { O } _ { [ i ] } = ( \mathbf { \Gamma } _ { [ i ] } ^ { \left. } \odot \mathbf { Q } _ { [ i ] } ) \mathbf { S } _ { [ i ] } ^ { \top } + ( ( \mathbf { Q } _ { [ i ] } \mathbf { K } _ { [ i ] } ^ { \top } ) \odot \mathbf { \Gamma } _ { [ i ] } ) \mathbb { B } _ { [ i ] } \mathbf { V } _ { [ i ] } . } \end{array}
$$

Diagonal decay. In the absence of delta correction $( \delta = 0 )$ but with diagonal decay $\Lambda _ { t } = \mathrm { d i a g } ( \alpha _ { t } )$ define a channel-wise diagonal decay $\begin{array} { r } { \gamma _ { [ i ] , n } ^ { s  r } : = \prod _ { u = s + 1 } ^ { r } \alpha _ { [ i ] , n } ^ { u } } \end{array}$ and $\gamma _ { [ i ] , n } ^ { r } : = \gamma _ { [ i ] , n } ^ { 0 \to r }$ . The stacked prefixand tail-decay now are $\Gamma _ { [ i ] } ^ { \left. } , \Gamma _ { [ i ] } ^ { \right. } \in \mathbb { R } ^ { C \times D _ { k } }$ , with entries $( \Gamma _ { [ i ] } ^ {  } ) _ { r n } = \gamma _ { [ i ] , n } ^ { r }$ and $( \Gamma _ { [ i ] } ^ {  } ) _ { r n } = \gamma _ { [ i ] , n } ^ { r  C }$ . For positive decay gates, the chunkwise parallel form then reads (Yang et al., 2024b; Team et al., 2025)

$$
\begin{array} { r l } & { \mathbf { S } _ { [ i + 1 ] } = \mathbf { S } _ { [ i ] } \mathrm { d i a g } ( \gamma _ { [ i ] } ^ { C } ) + ( \mathrm { B } _ { [ i ] } \mathbf { V } _ { [ i ] } ) ^ { \top } ( \mathbf { \Gamma } _ { [ i ] } ^ {  } \odot \mathbf { K } _ { [ i ] } ) , } \\ & { \mathbf { \Gamma } \mathbf { O } _ { [ i ] } = ( \Gamma _ { [ i ] } ^ {  } \odot \mathbf { Q } _ { [ i ] } ) \mathbf { S } _ { [ i ] } ^ { \top } + \mathrm { t r i l } ( ( \Gamma _ { [ i ] } ^ {  } \odot \mathbf { Q } _ { [ i ] } ) ( \mathbf { K } _ { [ i ] } \oslash \Gamma _ { [ i ] } ^ {  } ) ^ { \top } ) \mathbf { B } _ { [ i ] } \mathbf { V } _ { [ i ] } . } \end{array}
$$

Delta correction. For delta correction $( \delta = 1 )$ without decay, the chunkwise parallel form reads (Yang et al., 2024c)

$$
\begin{array} { r l } & { \mathbf { S } _ { [ i + 1 ] } = \mathbf { S } _ { [ i ] } + \mathbf { F } _ { [ i ] } ^ { \top } \mathbf { K } _ { [ i ] } , } \\ & { \mathbf { O } _ { [ i ] } = \mathbf { Q } _ { [ i ] } \mathbf { S } _ { [ i ] } ^ { \top } + \mathrm { t r i l } ( \mathbf { Q } _ { [ i ] } \mathbf { K } _ { [ i ] } ^ { \top } ) \mathbf { F } _ { [ i ] } , } \end{array}
$$

where $\mathbf { F } _ { [ i ] }$ are delta-corrected pseudo-values generated by

$$
\mathbf { F } _ { [ i ] } = ( \mathbf { I } + \operatorname { t r i l } ( \mathbb { B } _ { [ i ] } \mathbf { K } _ { [ i ] } \mathbf { K } _ { [ i ] } ^ { \top } , - 1 ) ) ^ { - 1 } \mathbb { B } _ { [ i ] } ( \mathbf { V } _ { [ i ] } - \mathbf { K } _ { [ i ] } \mathbf { S } _ { [ i ] } ^ { \top } ) .
$$

Appendix D.1 gives the decaying delta forms (Yang et al., 2025; Team et al., 2025).

![](images/261a3ec538fe3370f082c52fef5ac5cf2fe0c516faf12415096001eef35416ce.jpg)  
Figure 1: Architectural overview of a recurrent model at three resolutions: the model, recurrent layer, and chunkwise recurrence. Axis transformations and decay/write-factor allocations change the operands exposed to quantization. Applicability depends on the recurrence, as detailed in Section 3. Style inspired by (Liu et al., 2025, Figure 1).

## 3 LINQUANT: FUNCTIONAL INVARIANCES FOR LOW-PRECISION RECURRENCES

The chunkwise parallel forms in Section 2.2 expose a set of matrix multiplications and a state carried between chunks. Importantly, these chunkwise forms are not unique: the same full-precision recurrence can be expressed through functionally equivalent realizations that expose different quantities to low precision. LinQuant exploits this freedom. We first define the low-precision realization used throughout the paper, then derive seven functional invariances: four axis transformations and three absorptions. Table 1 summarizes their applicability; proofs are in Appendix C.

## 3.1 LOW-PRECISION RECURRENCE REALIZATION

Across the recurrences in Table 3, we find that the chunkwise computation can be organized into four matrix contractions: a state write, state read, local score, and intra-chunk output. Together with a recurrent state pass between chunks, this decomposition defines the low-precision boundaries we study. Consider for example the recurrence underlying Mamba-2. Using the notation of Section 2.2 and writing Q for quantization, with scale factors implicit, we consider the following low-precision chunkwise realization:

$$
\begin{array} { r l r } & { \quad \widehat { \mathbf { H } } _ { [ i ] } = { \cal Q } ( { \mathbf { B } } _ { [ i ] } \mathbf { V } _ { [ i ] } ) ^ { \top } { \cal Q } ( \mathbf { { \Gamma } } _ { [ i ] } ^ {  } \odot \mathbf { { K } } _ { [ i ] } ) , } & { \mathrm { ( s t a t e ~ w r i t e ) } , } \\ & { \widehat { \mathbf { S } } _ { [ i + 1 ] } = { \cal Q } ( \gamma _ { [ i ] } ^ { C } \mathbf { S } _ { [ i ] } + \widehat { \mathbf { H } } _ { [ i ] } ) , } & { \mathrm { ( s t a t e ~ p a s s ) } , } \\ & { \widehat { \mathbf { O } } _ { [ i ] } ^ { \mathrm { i n t e r } } = \mathbf { { \Gamma } } _ { [ i ] } ^ { + } \odot \big ( { \cal Q } ( \mathbf { Q } _ { [ i ] } ) \widehat { \mathbf { S } } _ { [ i ] } ^ { \top } \big ) , } & { \mathrm { ( s t a t e ~ r e a d ) } , } \\ & { \quad \widehat { \mathbf { A } } _ { [ i ] } = \big ( { \cal Q } ( \mathbf { Q } _ { [ i ] } ) { \cal Q } ( \mathbf { K } _ { [ i ] } ) ^ { \top } \big ) \odot { \boldsymbol \Gamma } _ { [ i ] } , } & { \mathrm { ( l o c a l ~ s c o r e ) } , } \\ & { \widehat { \mathbf { O } } _ { [ i ] } ^ { \mathrm { i n t r a } } = \widehat { \mathbf { A } } _ { [ i ] } { \cal Q } ( \mathrm { B } _ { [ i ] } \mathbf { V } _ { [ i ] } ) , } & { \mathrm { ( i n t r a - c h u n k ~ o u t p u t ) } . } \end{array}
$$

For models that employ the delta-rule, the Gram and WY solve are executed in full precision. The incoming-state correction used to form pseudo-values is an additional quantized contraction. Delta-rule writes and intra-chunk outputs replace the gated values $\mathbf { B } _ { [ i ] } \mathbf { V } _ { [ i ] }$ by the pseudo-values $\mathbf { F } _ { [ i ] }$ computed for the corresponding recurrence. Appendix D.1 provides more details.

LinQuant exploits the fact that functional invariances of these operators generally do not commute with quantization Q. These functional invariances fall into two classes: changes of basis, which change the representation along the key or value axes, and absorptions, which redistribute decay or write factors within an equivalent chunkwise computation. Appendix E.2 studies sensitivity to the stored-state and write-side quantization boundaries.

## 3.2 MIXER-EXTERNAL INVARIANCES: CHANGES OF BASIS

Both the key- and the value axis of the recurrent state $\mathbf { S } _ { t } \in \mathbb { R } ^ { D _ { v } \times D _ { k } }$ expose a choice of basis which leaves the full precision recurrence unchanged. We consider two such transformations along each axis: orthogonal rotations and positive diagonal rescalings. Notably, changes of basis along the key axis must remain compatible with the recurrent dynamics, whereas the value axis admits the same equivariance across all recurrences in Equation (1).

(1/7) Key-side rotation. A joint orthogonal rotation of queries and keys,

$$
\begin{array} { r } { ( { \pmb q } _ { t } , { \pmb k } _ { t } )  ( { \pmb { \mathrm { R } } } _ { Q K } { \pmb q } _ { t } , { \pmb { \mathrm { R } } } _ { Q K } { \pmb k } _ { t } ) , \qquad { \pmb { \mathrm { R } } } _ { Q K } \in \mathcal { O } ( D _ { k } ) , } \end{array}
$$

preserves query/key inner products and is a standard invariance of full precision softmax attention exploited by rotation-based PTQ methods such as QuaRot and SpinQuant (Ashkboos et al., 2024; Liu et al., 2025). In low precision, random rotations spread energy across channels, reducing outlier concentration and improving quantization robustness.

For a linear recurrence, the invariance holds if $\mathbf { R } _ { Q K } \pmb { \Lambda } _ { t } = \pmb { \Lambda } _ { t } \mathbf { R } _ { Q K }$ (Nazari & Rusch, 2026), and implicitly applies a change of basis to the state’s key axis, which transforms like $\mathbf { S } _ { t } \to \mathbf { S } _ { t } \mathbf { R } _ { Q K } ^ { \top }$ Invariance thus holds with no decay or scalar decay. With diagonal decay, a general rotation does not commute with $\Lambda _ { t } .$ , so the recurrent state cannot be carried in a globally rotated basis. However, we can still rotate queries and keys together after applying their decay factors when computing within-chunk scores. The recurrent state remains unrotated (Appendix D.1).

In this work, we use random signed Hadamard rotations. Recurrent models may apply per-channel causal convolutions and nonlinearities between their input projections and the recurrence, which generally do not commute with a dense rotation. We thus apply $\mathbf { R } _ { Q K }$ online after these operations.

(2/7) Key-side rescaling. The key axis also admits a reciprocal diagonal transformation,

$$
( { \bf q } _ { t } , { \pmb k } _ { t } )  ( { \bf D } _ { Q K } { \pmb q } _ { t } , { \bf D } _ { Q K } ^ { - 1 } { \pmb k } _ { t } ) , \qquad { \bf D } _ { Q K } \succ 0 \mathrm { { d i a g o n a l } , }
$$

which likewise preserves query-key inner products. Related rescalings underlie SmoothAttention in QServe and, more broadly, methods such as SmoothQuant (Lin et al., 2025; Xiao et al., 2023).

For a linear recurrence, the state implicitly transforms like $\mathbf { S } _ { t } \to \mathbf { S } _ { t } \mathbf { D } _ { Q K } ^ { - 1 }$ . Unlike a dense rotation, $\mathbf { D } _ { Q K }$ commutes with diagonal decay, making the transformation valid for linear writes under no, scalar, or diagonal decay. It does not, however, preserve delta corrected writes.

$\mathbf { D } _ { Q K }$ transports magnitude between queries and keys. In this work, we parameterize $\mathbf { D } _ { Q K } = \mathrm { d i a g } ( d )$ by $\begin{array} { r } { \dot { d } _ { n } = ( \operatorname* { m a x } _ { t } | k _ { t , n } | / \operatorname* { m a x } _ { t } | q _ { t , n } | ) ^ { \kappa } } \end{array}$ , where the maxima are computed over calibration tokens within each head. The exponent $\kappa \in [ 0 , 1 ]$ controls the rescaling strength. Specifically, $\begin{array} { r } { \kappa = \frac { 1 } { 2 } } \end{array}$ balances the magnitudes of queries and keys. We leave other parametrizations to future work.

(3/7) Value-side rotation. The value axis admits an analogous orthogonal transformation,

$$
\begin{array} { r } { \pmb { v } _ { t }  \mathbf { R } _ { V } \pmb { v } _ { t } , \qquad \mathbf { R } _ { V } \in \mathcal { O } ( D _ { v } ) . } \end{array}
$$

In softmax attention, value rotations are used for KV-cache quantization (Liu et al., 2025). The operator, however, is not invariant to value rotations, which have to be undone after the softmax. Linear recurrences exhibit the same equivariance. Intuitively, this is because the recurrence acts independently along the value axis, which is implicitly rotated like ${ \bf S } _ { t }  { \bf R } _ { V } { \bf S } _ { t }$

We note that, compared to softmax attention, the recurrent setting reveals another degree of freedom. The state can, in between chunks, either be stored in its canonical basis, $\mathscr { Q } ( \mathbf { S } _ { t } )$ , or in a rotated basis, ${ \mathcal { Q } } ( \mathbf { R } _ { V } \mathbf { S } _ { t } )$ . These realizations are equivalent in full precision but expose different representations to Q. Rotated storage is what the rotation produces by itself, whereas canonical storage returns each chunk’s write to the original basis before it joins the carried state. This comes at the cost of an inverse transform per chunk, and keeps the state in the basis whose per-channel scales fit it.

(4/7) Value-side rescaling. Finally, the value axis admits the positive diagonal analogue

$$
\begin{array} { r } { \pmb { v } _ { t }  \mathrm { D } _ { V } \pmb { v } _ { t } , \qquad \mathbf { D } _ { V } \succ 0 \mathrm { d i a g o n a l } . } \end{array}
$$

The recurrent state transforms accordingly as $\mathbf { S } _ { t } \to \mathbf { D } _ { V } \mathbf { S } _ { t } ,$ , with the inverse rescaling applied to the recurrence output before downstream normalization or other operations that do not commute with it.

If a linear output projection follows the recurrence directly, it can absorb the inverse rescaling. As with $\mathbf { R } _ { V }$ , this transformation is valid across all recurrence families considered here.

In this work, we parametrize $\mathbf { D } _ { V } = \mathrm { d i a g } ( d )$ by $\begin{array} { r } { d _ { p } = \big ( ( \prod _ { j = 1 } ^ { D _ { v } } \operatorname* { m a x } _ { t } | v _ { t , j } | ) ^ { 1 / D _ { v } } / \operatorname* { m a x } _ { t } | v _ { t , p } | \big ) ^ { \kappa } } \end{array}$ where the maxima are computed over calibration tokens and the product runs over value channels within each head. This parametrization moves range between value channels. $\mathrm { A t } \kappa = 1$ , the ranges of the value channels are equalized.

## 3.3 MIXER-INTERNAL INVARIANCES: ABSORPTIONS

The first four invariances discussed in the previous section change the basis along the two axes of the recurrent state. The remaining three redistribute multiplicative factors within a chunkwise parallel implementation of the recurrence. In particular, decay and write factors can often be divided between the two sides of a matrix product without changing the full-precision operator.

(5/7) Scalar-decay absorption. For scalar decay, the tail-decay $\Gamma _ { [ i ] } ^ {  }$ can be split between the value and key sides of the state write:

$$
\mathbf { H } _ { [ i ] } = ( ( \Gamma _ { [ i ] } ^ {  } ) ^ { \eta } \odot ( \mathbf { B } _ { [ i ] } \mathbf { V } _ { [ i ] } ) ) ^ { \top } ( ( \Gamma _ { [ i ] } ^ {  } ) ^ { 1 - \eta } \odot \mathbf { K } _ { [ i ] } ) , \qquad \eta \in [ 0 , 1 ] .
$$

Thus, η continuously interpolates between equivalent realizations of the same write. $\mathrm { A t } \eta = 0 .$ , all tail decay remains on the keys, which we call key-side decay absorption. $\mathbf { A } \mathbf { t } \ \eta = 1$ , it moves entirely to the values, giving value-side decay absorption, the endpoint used by SSDi8 (Kim et al., 2026). Intermediate values split the decay between the two sides. The same split applies to delta-rule writes, replacing $\mathbf { B } _ { [ i ] } \mathbf { V } _ { [ i ] }$ by the corresponding pseudo-values $\mathbf { F } _ { [ i ] }$

(6/7) Diagonal-decay absorption. Diagonal decay varies across state channels and therefore cannot, in general, be transferred entirely to the value side. Nevertheless, we can factor each token’s decay into a scalar magnitude and a relative channelwise pattern. This decomposition is not unique: any positive scalar factor yields an exact redistribution. Here, we use the channelwise maximum, and leave alternative choices to future work.

For diagonal tail decay $\Gamma _ { [ i ] } ^ {  }$ , define the per-token scalar magnitude and normalized decay

$$
e _ { [ i ] } ^ { r } : = \operatorname* { m a x } _ { n } ( \mathbf { \Gamma } _ { [ i ] } ^ {  } ) _ { r n } , \qquad \mathbf { E } _ { [ i ] } : = \operatorname { d i a g } ( e _ { [ i ] } ^ { 1 } , \dots , e _ { [ i ] } ^ { C } ) , \qquad \mathbf { \overline { { D } } } _ { [ i ] } ^ { \mathrm { t a i l } } : = \mathbf { E } _ { [ i ] } ^ { - 1 } \mathbf { \Gamma } _ { [ i ] } ^ {  } .
$$

Then the state write can be expressed as

$$
\mathbf { H } _ { [ i ] } = ( \mathbf { E } _ { [ i ] } ^ { \eta } \mathrm { B } _ { [ i ] } \mathbf { V } _ { [ i ] } ) ^ { \top } ( \mathbf { E } _ { [ i ] } ^ { 1 - \eta } \mathbf { \overline { { D } } } _ { [ i ] } ^ { \mathrm { t a i l } } \odot \mathbf { K } _ { [ i ] } ) , \qquad \eta \in [ 0 , 1 ] .
$$

Here, η moves the magnitude of the diagonal decay. $\mathbf { A t } ~ \eta ~ = ~ 0$ , the standard key-side form is recovered. $\mathbf { A } \mathbf { t } \eta = 1$ , each token’s scalar decay magnitude moves to the value side, while the relative channel-dependent decay necessarily remains on the key.

(7/7) Write-factor absorption. For a linear write, the write factor admits an analogous split:

$$
\mathbf { H } _ { [ i ] } = ( \mathbb { B } _ { [ i ] } ^ { \xi } \mathbf { V } _ { [ i ] } ) ^ { \top } ( \mathbf { { T } } _ { [ i ] } ^ {  } \odot \mathbf { B } _ { [ i ] } ^ { 1 - \xi } \mathbf { K } _ { [ i ] } ) , \qquad \xi \in [ 0 , 1 ] .
$$

The endpoints $\xi = 1$ and $\xi = 0$ place the write factor entirely on the value and key sides, respectively.

For delta-corrected writes, the write gate also enters the triangular system used to construct the pseudo-values. Without decay, split it between the two key factors of the Gram matrix:

$$
\mathbf L _ { [ i ] } ^ { ( \sigma ) } : = \mathrm { t r i l } ( ( \mathbb { B } _ { [ i ] } ^ { \sigma } \mathbf { K } _ { [ i ] } ) ( \mathbb { B } _ { [ i ] } ^ { 1 - \sigma } \mathbf { K } _ { [ i ] } ) ^ { \top } , - 1 ) , \quad \quad \sigma \in [ 0 , 1 ] .
$$

Using $\mathbf { B } _ { [ i ] } ^ { \sigma }$ on the corresponding right-hand side yields transformed pseudo-values $\mathbf { F } _ { [ i ] } ^ { ( \sigma ) }$ satisfying

$$
\mathbf { F } _ { [ i ] } ^ { ( \sigma ) } = \mathbf { B } _ { [ i ] } ^ { \sigma - 1 } \mathbf { F } _ { [ i ] } ,
$$

where $\mathbf { F } _ { [ i ] }$ denotes the pseudo-values produced by the standard formulation in Section 2.2. A second parameter τ splits the complementary factor $\mathbf { B } _ { [ i ] } ^ { 1 - \sigma }$ between the transformed write operands:

$$
\mathbf { V } _ { [ i ] } ^ { ( \sigma , \tau ) } = \mathbb { B } _ { [ i ] } ^ { ( 1 - \sigma ) ( 1 - \tau ) } \mathbf { F } _ { [ i ] } ^ { ( \sigma ) } , \qquad \mathbf { K } _ { [ i ] } ^ { ( \sigma , \tau ) } = \mathbb { B } _ { [ i ] } ^ { ( 1 - \sigma ) \tau } \mathbf { K } _ { [ i ] } ,
$$

$$
\mathbf { H } _ { [ i ] } = \big ( \mathbf { V } _ { [ i ] } ^ { ( \sigma , \tau ) } \big ) ^ { \top } \mathbf { K } _ { [ i ] } ^ { ( \sigma , \tau ) } = \mathbf { F } _ { [ i ] } ^ { \top } \mathbf { K } _ { [ i ] } , \qquad \tau \in [ 0 , 1 ] .
$$

![](images/fad6c7ece6f1e2f513ed4524dd34b9d2310f89737f7f7f627ace765fa2f457c7.jpg)

(a) Recurrence structure.  
![](images/595e9c37a56ce9345e5b43e558ebb295f3193b5729be0945dd3fcd814c815cd7.jpg)  
(b) QK norm.

![](images/4590901a57b8b4fa55e49dc037d338d3007df09ff29aa84300c57318a6ba7a5d.jpg)  
(c) Scale granularity.

![](images/e2e84ba63cd17bb4aa4ee14509d223838ebeff88b065fb21bd158f45feca9966.jpg)  
⟨H,C⟩ ⟨T,T⟩ ⟨T,C⟩  
(d) Rescaling exponent.  
Figure 2: Key-axis transformations at INT6. (a) Recovered excess NLL by decay and write rule. (b) Excess NLL with and without QK normalization (†). Panels (a,b) use ⟨T, C⟩ and $\kappa = 1 / 2$ for $\mathbf { D } _ { Q K }$ . (c) Recovery under ⟨H, C⟩ and its change under ⟨T, C⟩, with selected κ. (d) κ sweep, one curve per model.

Thus, σ determines how the write factor is distributed inside the Gram construction, while τ determines how the remaining compensation is distributed in the subsequent write. $\mathbf { A } \mathbf { t } { \boldsymbol { \tau } } = 0$ , the compensation remains on the pseudo-values. $\mathbf { A } \mathbf { t } \boldsymbol { \tau } = 1$ , it moves to the keys. Setting $\sigma = 1$ recovers the standard triangular formulation, while $\sigma = 1 / 2$ places square-root write factors on the two Gram operands. Appendix C proves these equivalences and extends the construction to decayed delta-rule recurrences.

Together, the four changes of basis and three absorptions define seven functional degrees of freedom for realizing the same recurrent computation. Their applicability depends on the recurrence structure, while their numerical behavior depends on the resulting low-precision representation, which we explore in the following section.

## 4 EXPERIMENTS

In this section, we first evaluate how the invariances discussed in Section 3 affect quantization error in a controlled model suite (Section 4.1). We then translate these insights to a production setting, evaluating quality on pretrained Mamba-2 and Nemotron-H models and latency on Mamba-2 (Section 4.2).

## 4.1 ANALYZING LINQUANT

Model suite. To isolate the impact of each component that comprises the linear recurrence, we set up a controlled mixer zoo. All models share the same backbone and training recipe. It consists of Linear Attention, Mamba-2, GLA, DeltaNet, Gated DeltaNet, and KDA at 340M and 1.3B parameters, trained with 15B and 100B tokens of SlimPajama, respectively. The 340M suite furthermore contains models that vary write gating $\beta _ { t }$ and QK normalization. Appendix B.1 gives the full setup.

Quantization settings. We evaluate how the identified invariances affect quantization error using WikiText-2 test NLL, comparing against full precision and a naive quantizer without invariance transforms. We use the signed Hadamard rotations and diagonal rescalings discussed in Section 3.2, calibrated using WikiText-2. Several normalized decaying and delta-rule models already incur little quantization error at INT8 with per-channel value scales (Table 7). We therefore use INT6 in the controlled study to amplify quantization effects and make the impact of the invariances easier to distinguish. INT6 uses a restricted integer range within our INT8 kernels; the production experiments use INT8 recurrent computation. The end-of-chunk states are passed in FP32, while quantizing a copy to an INT8 readout. Appendix B.1 gives calibration details, and Appendix E reports INT8, INT4, 1.3B, and zero-shot results.

Scale geometries. The queries/read-keys and values can be quantized with different granularities, denoted by $\left. b _ { Q K } , b _ { V } \right.$ . Here, $b _ { Q K }$ gives the scale-granularity for queries and read-keys, while $b _ { V }$ gives that of the values. Both can be C (per-channel), H (per head) or T (per-tensor). The scale groupings considered are directly influenced by the chunkwise kernels. In a matrix contraction, outputchannel scales can be applied after the product, whereas scales along the summed dimension must enter the contraction. The key axis is contracted in query/key products and state reads. In contrast, the value axis remains an output dimension and admits per-channel scales that can be incorporated in a kernel’s output scaling without modifying the integer contraction or requiring an additional pass. We compare the three scale regimes ⟨H, C⟩, ⟨T, C⟩, and ⟨T, T⟩. In the state write contraction, the key dimension stays an outer dimension. We exploit this by quantizing write keys per channel in ⟨H, C⟩, where each contraction quantizes its own copy of the keys, so the write can take the finest scale it can absorb. In the other two regimes, write keys are quantized per tensor and reused by each contraction.

![](images/ca45a5eb818142f24ab591ecea8c7603c86c925f4b52da6d0cdade3bcb22f2b6.jpg)

(a) Recurrence structure.  
![](images/e964b3ecef67eef8bcdca46f2e890297f0d36bae305e565aa670ca1d23259ca8.jpg)  
(b) QK norm.

![](images/c4bd329141320b6352326d867e1aa178c19505753eabeee7daf5614a6f2fa870.jpg)  
(c) Scale granularity.

![](images/f95df0b9df0117f9ec74358aef7be112fbd473169a92aa3f29e1ca8bd0c419cc.jpg)  
(d) Storage basis.  
Figure 3: Value-axis transformations at INT6. (a) Recovered excess NLL under ⟨T, T⟩. (b) Excess NLL under ⟨T, T⟩ with and without QK normalization (†). (c) Recovery under ⟨T, C⟩ and ⟨T, T⟩. (d) Recovery lost by storing the state rotated instead of canonically under $\mathbf { R } _ { V }$ , relative to naive.

We next examine key-axis transformations (1-2), value-axis transformations (3-4), and absorptions (5-7) on quantization error under different scale-granularities, as well as the choice of storage basis.

(1-2) Key-axis transformations. The benefit of $\mathbf { R } _ { Q K }$ depends on the recurrence structure. With no or scalar decay, it rotates the carried state along the key axis and reduces quantization error. With diagonal decay, it is restricted to local contractions and provides little benefit (Figure 2a). In contrast, $\mathbf { D } _ { Q K }$ commutes with diagonal decay and improves quantization error where applicable. QK normalization also reduces quantization error (Figure 2b). Coarsening key scales increases the benefit of $\mathbf { R } _ { Q K }$ , while $\mathbf { D } _ { Q K }$ remains useful with per-head scales (Figure 2c). The diagonal rescaling is generally best shared between queries and keys (Figure 2d).

(3-4) Value-axis transformations. Both $\mathbf { R } _ { V }$ and $\mathbf { D } _ { V }$ apply to every recurrence. They are most useful with per-tensor value scales, where they recover a large portion of the excess NLL (Figure 3a). The two transformations yield similar gains as value scales are coarsened (Figure 3c). Only $\mathbf { R } _ { V }$ introduces a choice of state basis: quantizing the state in the rotated basis incurs more error than quantizing it in the canonical basis (Figure 3d). While the two transformations give similar quality, the cost of canonical storage favors $ { \mathbf Ḋ \ Ḍ } _ { V }$ , which folds into the quantization scales.

(5-7) Absorptions. We study absorptions in the coarse-scale regime. Under ⟨T, T⟩, write-gate absorption recovers a significant portion of the excess NLL (Figure 4a). The preferred

![](images/1529ede56805f760e0c916aa52304a6d6686a0a1d4c07354c1cb075146fcb685.jpg)  
(a) Write gate.

![](images/bfd36dbcc8faece0ce484b0eddec8ec3e3ad0edaaf93273b55ea2cf42dd277c1.jpg)  
(b) Decay absorption.  
Figure 4: Absorptions at INT6 under ⟨T, T⟩. (a) Recovered excess NLL from the selected write-factor allocation (ξ or $( \sigma , \tau ) )$ (b) NLL increase from placing decay on keys $( \eta ~ = ~ 0 )$ rather than values $( \eta = 1 )$ , relative to $\eta = 1$

write-gate placement also depends on scale granularity. For gated linear writes, under ⟨T, C⟩ keeping the gate on values $( \xi = 1 )$ is preferred, while the other regimes can favor splitting it between keys and values (Figure 10). For delta writes under $\langle \mathrm { T } , \mathrm { T } \rangle$ , the selected settings use $\sigma \in \{ 0 , \frac { 1 } { 2 } \}$ and $\tau = 1 ( { \mathrm { F i g u r e ~ } } 1 1 )$ . Compared with the standard WY placement $( \sigma , \tau ) = ( 1 , \mathbf { \bar { 1 } } )$ used in Flash Linear equivalent placements of scalar decay. Table 2 equivalent placements of scalar decay. Table 2 reports quality. Setup and further results are in Appendices B.2 and F. Appendices B.2 and F.

![](images/d526939585079f1c0f6bd42a73d57d853554550ecceb2d2f9bb8137874043cfd.jpg)  
(a) Prefill time.

![](images/f3b33269fe6b0c1eef15d9815732bf3cff1ab48389ba2ad4e1bdc9869db47820.jpg)  
(b) Value rotation.

![](images/483e3cde771c4aacdb4fa1f15ae16ff8b80fffc4e4c2e85e5ad994180192cb3c.jpg)  
(c) Decay side.  
Figure 5: (a) End-to-end prefill time of LinQuant $( { \bf R } _ { Q K }$ , value-side decay) relative to the matched SSDi8 setting, production Mamba-2 (A5000, T = 2048). (b) INT8 zoo kernel with $\mathbf { R } _ { V }$ and canonical storage relative to naive INT8 (A5000, B = H = 8, T = 2048, $D _ { k } = D _ { v } = 1 2 8 )$ . (c) Production recurrence latency with decay on keys relative to values (static scales, 64 heads, batch 8).

Attention (Yang & Zhang, 2024), this moves more of the gate factor onto the transformed keys. The standard floating-point placement is therefore not always best after quantization. Decay placement matters most for scalar decay: with per-tensor value scales, moving scalar decay from values to keys increases excess NLL. By contrast, diagonal-decay absorption provides little gain (Figure 4b). We conjecture that variation across the diagonal decay, which remains on the keys, matters more than its overall magnitude and leave other ways of extracting a factor to future work.

Combined transformations. Combining transformations reduces quantization error more than applying them individually (Figure 13 and Tables 5 and 6). At INT6 under ⟨T, T⟩, the selected combination recovers roughly twice as much excess NLL as the best single transformation for both 340M Mamba-2 and LA (Table 5).

Downstream accuracy. The NLL improvements also translate to downstream accuracy for LA. At INT8 under ⟨T, C⟩, combining transformations largely closes the gap to full precision for 340M LA on both LAMBADA and the six-task average (Table 11). At 1.3B, LA+β recovers most of the LAMBADA accuracy lost to naive quantization, though a gap to full precision remains (Table 12).

Kernels. We implement each model’s applicable invariances in a single INT8 Triton chunk kernel (Appendix D.2). Rescalings fold into the quantization scales, and absorptions redistribute factors between existing operands. Canonical storage under $\mathbf { R } _ { V }$ requires inverse transforms of the state write and local output (Figure 5b). These kernel timings exclude forward rotations performed outside the kernel; the production timings below cover end-to-end prefill.

## 4.2 LINQUANT IN A PRODUCTION SETTING

Production setup. We next apply LinQuant to production-scale pretrained Mamba-2 models and Nemotron-H (Blakeman et al., 2025). Our starting point is SSDi8 (Kim et al., 2026), a low-precision recurrent prefill pipeline for Mamba-2, which we augment with the applicable LinQuant invariances. The production Mamba-2 recurrence is most analogous to the $\mathrm { n o } \prec \beta ,$ no-QK-normalization Mamba-2 model in our mixer zoo. The controlled study therefore points primarily to the key-side changes of basis, $\mathbf { R } _ { Q K }$ and $\mathbf { D } _ { Q K }$ while its fine-grained value scales suggest little need for a value-side change of basis. We add these key-side transformations to SSDi8’s W4A8 projection and INT8-recurrence pipeline and evaluate both

Table 2: Production WikiText-2 perplexity. SSDi8 uses value-side decay absorption. The combined transformation is evaluated with decay on either operand. More results in Table 17.
<table><tr><td rowspan="3">Model</td><td rowspan="3">FP16 SSDi8</td><td colspan="3"> $\mathbf { R } _ { Q K } \ \mathbf { R } _ { Q K } + \mathbf { D } _ { Q K }$ </td></tr><tr><td>value</td><td>value</td><td>key</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Mamba-2</td><td></td><td></td><td></td><td></td></tr><tr><td>1.3B</td><td>10.420 10.92810.860 10.84910.830</td><td>9.379</td><td></td><td></td></tr><tr><td>2.7B 8B</td><td>9.060 9.441</td><td>7.456</td><td>9.383 7.435</td><td>9.372</td></tr><tr><td></td><td>7.250 7.620</td><td></td><td></td><td>7.437</td></tr><tr><td>Nemotron-H</td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td>6.030</td><td>6.338</td><td>6.186 6.178</td><td>6.175</td></tr><tr><td>Reasoning</td><td>8.423</td><td>8.921</td><td>8.666 8.650</td><td>8.670</td></tr></table>

Key-side transformations transfer. Across every reported Mamba-2 and Nemotron-H model, $\mathbf { R } _ { Q K }$ improves perplexity over the matched SSDi8 setting, while $\mathbf { D } _ { Q K }$ provides an additional gain in most configurations (Table 2). Both $\mathbf { D } _ { Q K }$ and $\mathbf { R } _ { Q K }$ are fused into the operand-preparation pass. Query–key rotation adds no measurable end-to-end prefill latency on the evaluated Mamba-2 models (Figure 5a), while fused rescaling leaves the measured Mamba-2 1.3B layer latency unchanged (Table 14). Full results are reported in Appendix F. The SSDi8 pipeline already uses per-channel value scales, matching the controlled setting in which $\mathbf { R } _ { V }$ and $ { \mathbf Ḋ \ Ḍ } _ { V }$ provide little benefit. Rotated state storage performs worse, while canonical storage does not improve over the unrotated baseline (Appendix F.1). Since canonical $\mathbf { R } _ { V }$ adds overhead (Figure 5b), we apply neither value-side transformation in the production configuration.

Kernels. The optimized SSDi8 kernels quantize operands once and reuse their INT8 representations across multiple contractions. Kim et al. (2026) place scalar decay on the value side, motivated by concerns that key-side decay would require per-position scales and materializing per-head keys. Our evaluated key-side configurations achieve comparable perplexity to the value-side configurations (Appendix F), and per-head decay can be applied in GPU registers without materializing the expanded tensor (Appendix D.2). Their runtime also depends on operand reuse. Moving decay onto an operand after its INT8 representation has been formed requires rereading and requantizing the floating-point tensor. The cost therefore depends on which operand is affected, its dimension, and how often its quantized representation is reused. Figure 5c shows that value-side absorption is faster for the evaluated Mamba-2 shapes, although the gap narrows and can reverse as the value dimension $D _ { v }$ increases. We therefore retain value-side absorption for its lower runtime on the target Mamba-2 geometries.

## 5 CONCLUSION

In this work, we introduce LinQuant, a framework that exploits seven functional invariances for lowprecision linear recurrences, and characterize their applicability across a wide range of recurrent mixers. These consist of mixer-external changes of basis and mixer-internal absorptions, namely alternative allocations of decay and write factors between operands. Our controlled study shows their benefits depend on recurrence structure, QK normalization, scale granularity, and the basis used to quantize the state. Notably, changes of basis and write-gate allocation are particularly effective under coarse scales. These findings transfer to production Mamba-2 and Nemotron-H models, where LinQuant improves quantization quality. Moreover, on the evaluated Mamba-2 models, query–key rotation adds no measurable prefill latency. Although our implementation focuses on integer prefill, the introduced invariances provide a broader design space (e.g. floating-point) for future low-precision implementations.

## ACKNOWLEDGEMENTS

This work was supported by the Hector Foundation, Max Planck ETH Center for Learning Systems, and the Department of the Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Department of the Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.

## REFERENCES

Alexander Amini, Anna Banaszak, Harold Benoit, Arthur Bo¨ok, Tarek Dakhran, Song Duong, Alfred¨ Eng, Fernando Fernandes, Marc Hark¨ onen, Anne Harrington, Ramin Hasani, Saniya Karwa,¨ Yuri Khrustalev, Maxime Labonne, Mathias Lechner, Valentine Lechner, Simon Lee, Zetian Li, Noel Loo, Jacob Marks, Edoardo Mosca, Samuel J. Paech, Paul Pak, Rom N. Parnichkun, Alex Quach, Ryan Rogers, Daniela Rus, Nayan Saxena, Bettina Schlager, Tim Seyde, Jimmy T. H. Smith, Aditya Tadimeti, and Neehal Tumma. Lfm2 technical report, 2025. URL https: //arxiv.org/abs/2511.23404.

Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, and James Hensman. QuaRot: Outlier-free 4-bit inference in rotated llms. In Advances in Neural Information Processing Systems, volume 37, 2024. URL https://proceedings.neurips.cc/paper\_files/paper/2024/hash/b5b93 9436789f76f08b9d0da5e81af7c-Abstract-Conference.html.

Maximilian Beck, Korbinian Poppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova,¨ Michael Kopp, Gunter Klambauer, Johannes Brandstetter, and Sepp Hochreiter. xlstm: Extended¨ long short-term memory, 2024. URL https://arxiv.org/abs/2405.04517.

Aaron Blakeman, Aarti Basant, Abhinav Khattar, Adithya Renduchintala, Akhiad Bercovich, Aleksander Ficek, Alexis Bjorlin, Ali Taghibakhshi, Amala Sanjay Deshmukh, Ameya Sunil Mahabaleshwarkar, et al. Nemotron-h: A family of accurate and efficient hybrid mamba-transformer models. arXiv preprint arXiv:2504.03624, 2025.

Makram Chahine, Philipp Nazari, Daniela Rus, and T Konstantin Rusch. The curious case of in-training compression of state space models. arXiv preprint arXiv:2510.02823, 2025.

Hung-Yueh Chiang, Chi-Chih Chang, Natalia Frumkin, Kai-Chiang Wu, and Diana Marculescu. Quamba: A post-training quantization recipe for selective state space models. arXiv preprint arXiv:2410.13229, 2024.

Hung-Yueh Chiang, Chi-Chih Chang, Natalia Frumkin, Kai-Chiang Wu, Mohamed S. Abdelfattah, and Diana Marculescu. Quamba2: A robust and scalable post-training quantization framework for selective state space models, 2025. URL https://arxiv.org/abs/2503.22879.

Tri Dao and Albert Gu. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality. arXiv preprint arXiv:2405.21060, 2024.

Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. GPTQ: Accurate post-training quantization for generative pre-trained transformers. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=tcbBPnfwxS.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id= tEYskw1VY2.

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Franc¸ois Fleuret. Transformers are RNNs: Fast autoregressive transformers with linear attention. In Hal Daume III and Aarti Singh´ (eds.), Proceedings ofthe 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pp. 5156–5165. PMLR, 13–18 Jul 2020. URL https://proceedings.mlr.press/v119/katharopoulos20a.html.

Hyunwoo Kim, Byoungchan Ko, Minseok Kang, Minwoo Kim, Dongjin Lee, Jaehoon Lee, Sungroh Yoon, and Dahuin Jung. Ssdi8: Accurate and efficient 8-bit quantization for state space duality. In The Fourteenth International Conference on Learning Representations, 2026.

Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. AWQ: Activation-aware weight quantization for on-device llm compression and acceleration. In Proceedings ofMachine Learning and Systems, volume 6, 2024. URL https://proceedings.mlsys.org/paper\_files/paper/ 2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference. html.

Yujun Lin, Haotian Tang, Shang Yang, Zhekai Zhang, Guangxuan Xiao, Chuang Gan, and Song Han. QServe: W4A8KV4 quantization and system co-design for efficient llm serving. In Proceedings of Machine Learning and Systems, volume 7, 2025. URL https://proceedings.mlsys.or g/paper\_files/paper/2025/hash/fbe2b2f74a2ece8070d8fb073717bda6-A bstract-Conference.html.

Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, and Tijmen Blankevoort. SpinQuant: Llm quantization with learned rotations. In International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=ogO6DGE6FZ.

Philipp Nazari and T Konstantin Rusch. The key to state reduction in linear attention: A rank-based perspective. arXiv preprint arXiv:2602.04852, 2026.

NVIDIA. Nvidia nemotron 3: Efficient and open intelligence, 2025. URL https://arxiv.or g/abs/2512.20856.

Rom N. Parnichkun, Neehal Tumma, Armin W. Thomas, Alessandro Moro, Qi An, Taiji Suzuki, Atsushi Yamashita, Michael Poli, and Stefano Massaroli. Quantifying memory utilization with effective state-size, 2025. URL https://arxiv.org/abs/2504.19561.

Bo Peng, Ruichong Zhang, Daniel Goldstein, Eric Alcaide, Xingjian Du, Haowen Hou, Jiaju Lin, Jiaxing Liu, Janna Lu, William Merrill, Guangyu Song, Kaifeng Tan, Saiteja Utpala, Nathan Wilce, Johan S. Wind, Tianyi Wu, Daniel Wuttke, and Christian Zhou-Zheng. Rwkv-7 ”goose” with expressive dynamic state evolution, 2025. URL https://arxiv.org/abs/2503.14456.

Imanol Schlag, Kazuki Irie, and Jurgen Schmidhuber. Linear transformers are secretly fast weight¨ programmers, 2021. URL https://arxiv.org/abs/2102.11174.

Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. Flashattention-3: Fast and accurate attention with asynchrony and low-precision, 2024. URL https://arxiv.org/abs/2407.08608.

Julien Siems, Timur Carstensen, Arber Zela, Frank Hutter, Massimiliano Pontil, and Riccardo Grazzi. Deltaproduct: Improving state-tracking in linear rnns via householder products, 2025. URL https://arxiv.org/abs/2502.10297.

Lintang Sutawika, Hailey Schoelkopf, Leo Gao, Stella Biderman, Baber Abbasi, Jonathan Tow, et al. Eleutherai/lm-evaluation-harness: v0.4.1, January 2024. URL https://doi.org/10.528 1/zenodo.10600400. Software.

Kimi Team, Yu Zhang, Zongyu Lin, Xingcheng Yao, Jiaxi Hu, Fanqing Meng, Chengyin Liu, Xin Men, Songlin Yang, Zhiyuan Li, Wentao Li, Enzhe Lu, Weizhou Liu, Yanru Chen, Weixin Xu, Longhui Yu, Yejie Wang, Yu Fan, Longguang Zhong, Enming Yuan, Dehao Zhang, Yizhi Zhang, T. Y. Liu, Haiming Wang, Shengjun Fang, Weiran He, Shaowei Liu, Yiwei Li, Jianlin Su, Jiezhong Qiu, Bo Pang, Junjie Yan, Zhejun Jiang, Weixiao Huang, Bohong Yin, Jiacheng You, Chu Wei, Zhengtao Wang, Chao Hong, Yutian Chen, Guanduo Chen, Yucheng Wang, Huabin Zheng, Feng Wang, Yibo Liu, Mengnan Dong, Zheng Zhang, Siyuan Pan, Wenhao Wu, Yuhao Wu, Longyu Guan, Jiawen Tao, Guohong Fu, Xinran Xu, Yuzhi Wang, Guokun Lai, Yuxin Wu, Xinyu Zhou, Zhilin Yang, and Yulun Du. Kimi linear: An expressive, efficient attention architecture, 2025. URL https://arxiv.org/abs/2510.26692.

Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, and Christopher De Sa. Quip#: Even better llm quantization with hadamard incoherence and lattice codebooks, 2024. URL https://arxiv.org/abs/2402.04396.

Neehal Tumma, Mathias Lechner, Noel Loo, Ramin Hasani, and Daniela Rus. Leveraging low-rank and sparse recurrent connectivity for robust closed-loop control. In The Twelfth International Conference on Learning Representations, 2024.

Neehal Tumma, Noel Loo, and Daniela Rus. Preconditioned deltanet: Curvature-aware sequence modeling for linear recurrences, 2026. URL https://arxiv.org/abs/2604.21100.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. SmoothQuant: Accurate and efficient post-training quantization for large language models. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 38087–38099. PMLR, 2023. URL https://proceedings.mlr.pr ess/v202/xiao23c.html.

Zukang Xu, Yuxuan Yue, Xing Hu, Zhihang Yuan, Zixu Jiang, Zhixuan Chen, Jiangyong Yu, Chen Xu, Sifan Zhou, and Dawei Yang. Mambaquant: Quantizing the mamba family with variance aligned rotation methods, 2025. URL https://arxiv.org/abs/2501.13484.

Songlin Yang and Yu Zhang. FLA: A Triton-based library for hardware-efficient implementations of linear attention mechanism, January 2024. URL https://github.com/fla-org/flash -linear-attention. Software.

Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule. arXiv preprint arXiv:2412.06464, 2024a.

Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon Kim. Gated linear attention transformers with hardware-efficient training. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pp. 56501–56523. PMLR, 21–27 Jul 2024b. URL https://proceeding s.mlr.press/v235/yang24ab.html.

Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. Parallelizing linear transformers with the delta rule over sequence length. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024c.

Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule, 2025. URL https://arxiv.org/abs/2412.06464.

Jintao Zhang, Jia Wei, Haofeng Huang, Pengle Zhang, Jun Zhu, and Jianfei Chen. Sageattention: Accurate 8-bit attention for plug-and-play inference acceleration, 2025. URL https://arxiv. org/abs/2410.02367.

Tao Zhang, Jianchao Tan, Pingwei Sun, Yanqi Yu, Zunhai Su, Zixu Jiang, Yuchen Xie, Xunliang Cai, and Ziqian Zeng. Damp: Decay-aware mixed-precision recurrent-state quantization, 2026. URL https://arxiv.org/abs/2608.27513.

## A EXTENDED RELATED WORK

Linear recurrent and hybrid language models. A growing line of work replaces softmax attention with recurrent sequence mixers whose state size (Parnichkun et al., 2025) and per-token decode cost do not grow with context length. This is achieved by implicitly summarizing past tokens in a fixed-size hidden state, instead of explicitly in a KV-cache (Katharopoulos et al., 2020; Schlag et al., 2021; Yang et al., 2024b). Selective state-space models such as Mamba and Mamba-2 introduce input-dependent recurrent dynamics and hardware-efficient implementations (Gu & Dao, 2024; Dao & Gu, 2024).

More expressive recurrences modify how information is retained and written into the hidden state. DeltaNet corrects stored associations by removing existing associations from its state (Yang et al., 2024c), while Gated DeltaNet combines this correction mechanism with scalar forgetting (Yang et al., 2025), and KDA uses channelwise decay (Team et al., 2025). Preconditioned DeltaNet further incorporates curvature-aware updates (Tumma et al., 2026), while DeltaProduct enhances the state transition operator (Siems et al., 2025). In a recent development, large-scale hybrid models combine recurrent and softmax attention layers (Team et al., 2025; Blakeman et al., 2025).

Our work studies how the structure of these different recurrences enables and constrains functional invariances. We then leverage these invariances for post-training quantization (PTQ).

PTQ of softmax Transformers. Weight-only PTQ methods such as GPTQ and AWQ reduce quantization error through reconstruction-based compensation or channel protection (Frantar et al., 2023; Lin et al., 2024). For joint weight and activation quantization, SmoothQuant transfers activation range into weights (Xiao et al., 2023), while QServe combines quantization with efficient serving (Lin et al., 2025). Rotation-based methods redistribute concentrated energy across channels: QuIP# uses Hadamard incoherence processing for weight quantization (Tseng et al., 2024), QuaRot extends rotations to weights, activations, and the KV cache (Ashkboos et al., 2024), and SpinQuant learns rotations from calibration data (Liu et al., 2025). LinQuant extends the study of functional invariances to linear recurrences.

Quantization of recurrent language models. Quamba develops an 8-bit PTQ recipe around sensitive selective-scan features and activation outliers (Chiang et al., 2024). MambaQuant uses variancealigned rotations (Xu et al., 2025), while Quamba2 extends PTQ across Mamba and Mamba-2 at multiple precisions (Chiang et al., 2025).

SSDi8 targets low-precision execution inside the chunkwise Mamba-2 computation (Kim et al., 2026). It reuses quantized keys and queries, absorbs decay into values, and carries an INT8 state, all to maintain a persistent INT8 computational path. This decay placement is the value-side endpoint of our scalar-decay absorption family. Our production pipeline builds on SSDi8 by adding state-axis rotations and rescalings (Section 4.2).

Recurrent-state compression beyond quantization. Chahine et al. (2025) prune the hidden state during training by leveraging the free choice of key-axis basis, which we identify as invariance (1). Similarly, Nazari & Rusch (2026) leverage the free choice of basis to prune the state of pretrained linear recurrences. Low-rank and sparse recurrent connectivity has also been studied outside language modeling (Tumma et al., 2024). In LinQuant, we instead retain state dimensions and use equivalent representations to improve quantization, addressing a complementary source of recurrent efficiency.

## B EXPERIMENTAL SETUP

We give the setup and timing methodology for the controlled study and production setting. Result are reported in Appendices E and F.

## B.1 CONTROLLED STUDY

Models and training. The 340M zoo contains fifteen recurrence variants, or cells: LA, Mamba-2, and GLA with and without write gating and QK normalization, plus DN, GDN, and KDA with QK normalization. All models use the FLA implementation (Yang & Zhang, 2024) and share a 24-layer architecture with a SwiGLU MLP, an output gate on the mixer, and tied embeddings. At 340M parameters, the model width is 1024 with 8 heads and $D _ { k } = D _ { v } = 1 2 8$ . At 1.3B, the width is 2048 with 16 heads and the same head dimensions.

Table 3: State-update rules for recurrent models in the mixer-zoo design space. Colors identify decay and write gating.
<table><tr><td>model</td><td>update</td></tr><tr><td>Linear attention (Katharopoulos et al., 2020)</td><td> $\mathbf { S } _ { t } = \mathbf { S } _ { t - 1 } + v _ { t } \pmb { k } _ { t } ^ { \top }$ </td></tr><tr><td>Mamba-2 (Dao &amp; Gu, 2024)</td><td> $\mathbf { S } _ { t } = \alpha _ { t } \mathbf { S } _ { t - 1 } + v _ { t } \pmb { k } _ { t } ^ { \top }$ </td></tr><tr><td>Gated Linear Attention (GLA) (Yang et al., 2024b)</td><td> $\mathbf { S } _ { t } = \mathbf { S } _ { t - 1 } \mathrm { d i a g } ( \alpha _ { t } ) + v _ { t } \boldsymbol { k } _ { t } ^ { \top }$ </td></tr><tr><td>DeltaNet (Schlag et al., 2021; Yang et al., 2024c)</td><td> $\mathbf { S } _ { t } = \mathbf { S } _ { t - 1 } \left( \mathbf { I } - \beta _ { t } \pmb { k } _ { t } \pmb { k } _ { t } ^ { \top } \right) + \beta _ { t } \pmb { v } _ { t } \pmb { k } _ { t } ^ { \top }$ </td></tr><tr><td>Gated DeltaNet (Yang et al., 2025)</td><td> $\mathbf { S } _ { t } = \alpha _ { t } \mathbf { S } _ { t - 1 } \left( \mathbf { I } - \beta _ { t } \pmb { k } _ { t } \pmb { k } _ { t } ^ { \top } \right) + \beta _ { t } \pmb { v } _ { t } \pmb { k } _ { t } ^ { \top }$ </td></tr><tr><td>Kimi Delta Attention (KDA) (Team et al., 2025)</td><td> $\mathbf { S } _ { t } = \mathbf { S } _ { t - 1 } \operatorname { d i a g } ( \alpha _ { t } ) \left( \mathbf { I } - \beta _ { t } \pmb { k } _ { t } \pmb { k } _ { t } ^ { \top } \right) + \beta _ { t } \pmb { v } _ { t } \pmb { k } _ { t } ^ { \top }$ </td></tr></table>

Each cell is an instance of Equation (1) with the decay and write rule of Table 3. Appendix D.1 gives the quantized chunkwise forms. QK normalization applies $\ell _ { 2 }$ normalization to $\pmb q _ { t }$ and $\boldsymbol { k } _ { t }$ independently within each head.

Every model is trained from scratch on SlimPajama using FLA and a Llama tokenizer, with 15B tokens for the 340M models and 100B for the 1.3B models. All runs use sequence length 2048 and AdamW with peak learning rate $4 \times 1 0 ^ { - 4 }$ , 1024 warm-up steps, and cosine decay to one tenth of the peak. We clip gradients at 1.

Quantization. Quantization is symmetric, with static scales calibrated by the maximum absolute activation over eight WikiText-2 training windows of 2048 tokens. The state write, state read, and local-score contractions use INT8 operands with INT32 accumulation (except the 16-token diagonal blocks of the local score under diagonal decay, which compute the decay-weighted dot products in FP32 using dequantized queries and keys); the intra-chunk output uses FP16 arithmetic on INT8- quantized values. An FP32 state is carried between chunks, and reads consume an INT8 copy with one scale per head and value channel, calibrated on the carry. We call this copy the stored state, and vary operand precision and stored-state precision separately. INT6 and INT4 configurations restrict the integer range of the operands within the same boundaries and kernels. The stored state remains INT8 unless stated otherwise. Chunk size is 64, with 16-token sub-blocks for diagonal decay, as in FLA.

Evaluation. We evaluate perplexity on WikiText-2 test at sequence length 2048. Transformation parameters are selected using cross-entropy on separate WikiText-2 samples. Excess NLL is $\log ( \bar { \mathrm { P P L } } _ { \mathrm { q u a n t } } / \mathrm { P P L } _ { \mathrm { F P } } )$ , and NLL recovered is the difference between the naive quantizer’s NLL and that of the transformed configuration. We evaluate zero-shot accuracy with the Language Model Evaluation Harness (Sutawika et al., 2024) on LAMBADA, HellaSwag, PIQA, ARC-Easy, ARC-Challenge, and WinoGrande.

Transformation selection. We use the diagonal rescaling parameterizations of Section 3.2 and sweep $\kappa \in \{ \textstyle { \frac { 1 } { 4 } } , \frac { 1 } { 2 } , \frac { 3 } { 4 } , 1 \}$ , including both $\mathbf { D } _ { Q K }$ alone and its composition with $\mathbf { R } _ { Q K }$ on linear-write cells. The full $\bar { \mathbf { D } _ { V } }$ sweep is evaluated under shared value scales.

For decaying cells, we sweep $\eta \in \{ 0 , { \frac { 1 } { 4 } } , { \frac { 1 } { 2 } } , { \frac { 3 } { 4 } } , 1 \}$ . Linear write-factor absorption uses $\xi \in \{ 1 , \frac { 1 } { 2 } , 0 \}$ while delta cells use $( \sigma , \tau ) \in \{ 1 , { \frac { 1 } { 2 } } , 0 \} ^ { 2 }$

We form the stack by combining transformations whose selected settings improve held-out crossentropy by at least 0.003 nats over the naive quantizer. We use canonical state storage unless stated otherwise, and retain the naive quantizer if no transformation qualifies.

Timing. We compare the transformation stack with canonical $\mathbf { R } _ { V }$ against naive INT8 on the A5000, A100, and H100 across batch sizes, with $T = 2 0 4 8 , H = 8 .$ , and $\bar { D _ { k } } \bar { = } \bar { D _ { v } } = 1 2 8$ . We report ratios of median call times after autotuning and warm-up. Both calls include gate processing, operand quantization, and recurrent computation, including preparation for delta cells. Forward rotations performed before the call are excluded.

## B.2 PRODUCTION SETTING

We evaluate pretrained Mamba-2 and Nemotron-H models using SSDi8’s W4A8 projection and INT8- recurrence pipeline. Unless otherwise stated, the matched baseline uses value-side decay absorption, static per- $( H , D _ { v } )$ value scales, per-group INT8 keys and queries, and an INT8 fixed-point state pass.

Calibration. The controlled study calibrates activations on WikiText-2 (Appendix B.1). In production, static activation scales and $\mathbf { D } _ { Q K }$ are calibrated on 512 Pile sequences of 512 tokens for Mamba-2 and 2048 tokens for Nemotron-H. The activation scales are measured on the FP16 model with the projection-level Hadamard transformations applied; production weight quantization then uses 128 WikiText-2 training samples with GPTQ after fusing these transformations. Each layer uses maximum absolute activation measurements with the same scale axes as the quantized kernel (Algorithm 3). In the recurrent core, we observe the raw operands ${ \mathbf { V } } , { \mathbf { K } } , { \mathbf { Q } } ,$ , decay-weighted operands $\begin{array} { r } { \dot { \mathbf { {Gamma } } } _ { [ i ] } ^ {  } \odot \mathbf { { V } } _ { [ i ] } , \dot { \mathbf { {Gamma } } } _ { [ i ] } ^ {  } \odot } \end{array}$ $\mathbf { K } _ { [ i ] }$ , local state contribution H, and chunk product $\mathbf { Z } = \mathbf { Q } \mathbf { K } ^ { \top }$

For $\mathbf { D } _ { Q K }$ , we measure channel maxima of the Mamba-2 key and query streams, conventionally denoted B and $\mathbf { C } ,$ after convolution and SiLU. Rescaling is applied at this same point during inference. Moving it before the nonlinear preprocessing would not preserve the function.

Evaluation. We evaluate WikiText-2 test perplexity at sequence length 2048, using the standard tokenizer for each model: EleutherAI/gpt-neox-20b for Mamba-2 1.3B and 2.7B, and NVIDIA SentencePiece for Mamba-2 8B. Downstream evaluation uses the Language Model Evaluation Harness (Sutawika et al., 2024) on SSDi8’s six zero-shot tasks: LAMBADA, HellaSwag, PIQA, ARC-Easy, ARC-Challenge, and WinoGrande. For Mamba-2, rows with static- $( H , D _ { v } )$ valueside scales run without SSDi8’s output-projection mean correction, as do the tables marked “no compensation”; the remaining Mamba-2 rows apply it with step factor 0.15, and the Nemotron-H rows with 0.1.

Timing. Unless otherwise indicated, latency measurements use an NVIDIA RTX A5000 with batch size $B = 8$ and sequence length $T = 2 0 4 8$ . For component measurements, we instantiate a single Mamba-2 mixer layer with random weights and time each kernel or sub-operation using CUDA events. Each run uses 30 warm-up forward passes for Triton autotuning and allocation, followed by 200 measured forward passes. We synchronize the stream after each measured forward pass, sum calls belonging to the same stage, and report the median stage latency in microseconds.

## C THE SEVEN INVARIANCES: STATEMENTS AND PROOFS

We prove the invariances of Section 3 for Equation (1) in exact arithmetic, with positive decay and write gates and consistently transformed initial states. Table 1 summarizes their applicability.

(1/7) Key-axis rotation $\mathbf { R } _ { Q K }$ . Let $\mathbf { R } = \mathbf { R } _ { Q K }$ be orthogonal and commute with every $\mathbf { \Lambda } _ { \Lambda _ { t } }$ . With ${ \pmb q } _ { t } ^ { \prime } = \mathrm { R } { \pmb q } _ { t } , { \pmb k } _ { t } ^ { \prime } = \mathrm { R } { \pmb k } _ { t }$ , and $\mathbf { S } _ { t - 1 } ^ { \prime } = \mathbf { S } _ { t - 1 } \mathbf { R } ^ { \intercal }$ , the update and readout satisfy

$$
\begin{array} { r } { \begin{array} { r l } & { \quad \mathbf { S } _ { t } ^ { \prime } = \big ( \mathbf { S } _ { t - 1 } \mathbf { \Lambda } \Lambda _ { t } ( \mathbf { I } - \delta \beta _ { t } \pmb { k } _ { t } \pmb { k } _ { t } ^ { \top } ) + \beta _ { t } \pmb { v } _ { t } \pmb { k } _ { t } ^ { \top } \big ) \mathbf { R } ^ { \top } = \mathbf { S } _ { t } \mathbf { R } ^ { \top } , } \\ & { \quad \mathbf { S } _ { t } ^ { \prime } \pmb { q } _ { t } ^ { \prime } = \mathbf { S } _ { t } \pmb { q } _ { t } . } \end{array} } \end{array}
$$

Commutation is automatic for absent or scalar decay. For fixed diagonal decay, R must preserve the common decay eigenspaces. Transforming the decay itself would generally make it dense. Local rotations of both decay-weighted score operands remain valid and leave the carried state unchanged. A global key rotation preserves the state quantizer’s scale axes but can change its calibrated maxima.

(2/7) Key-axis rescaling $\mathbf { D } _ { Q K }$ . Let $\mathbf { D } = \mathbf { D } _ { Q K } \succ 0$ be diagonal, with $\pmb q _ { t } ^ { \prime } = \mathrm { D } \pmb q _ { t } , \pmb k _ { t } ^ { \prime } = \mathrm { D } ^ { - 1 } \pmb { k } _ { t }$ and $\mathbf { S } _ { t - 1 } ^ { \prime } = \mathbf { S } _ { t - 1 } \mathbf { D } ^ { - 1 }$ . Since D commutes with diagonal decay, a linear write gives

$$
\mathbf { S } _ { t } ^ { \prime } = \bigl ( \mathbf { S } _ { t - 1 } \mathbf { A } _ { t } + \beta _ { t } \pmb { v } _ { t } \pmb { k } _ { t } ^ { \mathsf { T } } \bigr ) \mathbf { D } ^ { - 1 } = \mathbf { S } _ { t } \mathbf { D } ^ { - 1 } , \qquad \mathbf { S } _ { t } ^ { \prime } \pmb { q } _ { t } ^ { \prime } = \mathbf { S } _ { t } \pmb { q } _ { t } .
$$

For a delta write, matching the erasure terms requires D $\mathbf { \Phi } ^ { - 2 } \mathbf { \mathbf { } } k _ { t } \mathbf { \mathbf { } } k _ { t } ^ { \mathsf { T } } \mathbf { D } ^ { - 1 } = \mathbf { \Phi } k _ { t } \mathbf { \mathbf { } } k _ { t } ^ { \mathsf { T } } \mathbf { D } ^ { - 1 }$ , or $\mathrm { { \bf D } } ^ { 2 } { \pmb k } _ { t } = { \pmb k } _ { t }$ excluding a general rescaling. The transformation acts after any QK normalization.

(3/7–4/7) Value-axis rotation and rescaling. For any invertible T, set $\pmb { v } _ { t } ^ { \prime } = \mathbf { T } \pmb { v } _ { t }$ and $\mathbf { S } _ { t - 1 } ^ { \prime } =$ $\mathbf { T S } _ { t - 1 }$ . For either write rule and any diagonal decay,

$$
\begin{array} { r } { \mathbf { S } _ { t } ^ { \prime } = \mathbf { T } \big ( \mathbf { S } _ { t - 1 } \boldsymbol { \Lambda } _ { t } ( \mathbf { I } - \delta \beta _ { t } \boldsymbol { k } _ { t } \boldsymbol { k } _ { t } ^ { \mathsf { T } } ) + \beta _ { t } \boldsymbol { v } _ { t } \boldsymbol { k } _ { t } ^ { \mathsf { T } } \big ) = \mathbf { T } \mathbf { S } _ { t } , \qquad \mathbf { T } ^ { - 1 } \boldsymbol { o } _ { t } ^ { \prime } = o _ { t } . } \end{array}
$$

Choose $\mathbf { T } = \mathbf { R } _ { V } \thinspace \mathbf { o r } \thinspace \mathbf { T } = \mathbf { D } _ { V }$ . Canonical storage undoes T on each chunk’s write and local output. Delta pseudo-values additionally require the incoming-state correction in the transformed value basis. With per-channel scales, diagonal rescaling cancels because $( d _ { p } v _ { p } ) / ( d _ { p } s _ { p } ) = v _ { p } / s _ { p }$ , where $s _ { p }$ is the channel scale. Consistently transformed scales therefore leave quantization unchanged, apart from finite-precision rounding.

(5/7–7/7) Decay and linear-write absorptions. All three follow from the token-wise identity

$$
c _ { [ i ] } ^ { r } v _ { [ i ] } ^ { r } ( k _ { [ i ] } ^ { r } ) ^ { \top } = \big ( ( c _ { [ i ] } ^ { r } ) ^ { \eta } v _ { [ i ] } ^ { r } \big ) \big ( ( c _ { [ i ] } ^ { r } ) ^ { 1 - \eta } k _ { [ i ] } ^ { r } \big ) ^ { \top } , \qquad c _ { [ i ] } ^ { r } > 0 .
$$

Apply it to each chunk-write term with $c _ { [ i ] } ^ { r }$ equal to the scalar tail decay, the diagonal-decay magnitude $e _ { [ i ] } ^ { r }$ , or the write gate $\beta _ { [ i ] } ^ { r }$ . Other factors retain their placement. In particular, the relative channeldependent decay remains on the key. For delta-rule decay absorption, use the corresponding pseudovalues in place of gated values.

(7/7) Delta write-factor absorption. Using the interval weights of Section 2.2, define the strictly lower-triangular Gram matrix

$$
C _ { [ i ] , r s } = \mathbf { 1 } _ { s < r } \sum _ { n } K _ { [ i ] , r n } K _ { [ i ] , s n } \gamma _ { [ i ] , n } ^ { s  r } .
$$

Scalar decay is broadcast across channels. Without decay, $\gamma \equiv 1$ . The gate allocation is a diagonal similarity transformation:

$$
\begin{array} { c } { \mathbf { L } _ { [ i ] } ^ { ( \sigma ) } = \mathbb { B } _ { [ i ] } ^ { \sigma } \mathbf { C } _ { [ i ] } \mathbb { B } _ { [ i ] } ^ { 1 - \sigma } = \mathbb { B } _ { [ i ] } ^ { \sigma - 1 } \mathbf { L } _ { [ i ] } ^ { ( 1 ) } \mathbb { B } _ { [ i ] } ^ { 1 - \sigma } , } \\ { ( \mathbf { I } + \mathbf { L } _ { [ i ] } ^ { ( \sigma ) } ) ^ { - 1 } \mathbb { B } _ { [ i ] } ^ { \sigma } = \mathbb { B } _ { [ i ] } ^ { \sigma - 1 } ( \mathbf { I } + \mathbf { L } _ { [ i ] } ^ { ( 1 ) } ) ^ { - 1 } \mathbb { B } _ { [ i ] } . } \end{array}
$$

Writing $\mathbf { F } _ { [ i ] } : = \mathbf { F } _ { [ i ] } ^ { ( 1 ) }$ , the complete pseudo-values satisfy

$$
\mathbf { F } _ { [ i ] } ^ { ( \sigma ) } = ( \mathbf { I } + \mathbf { L } _ { [ i ] } ^ { ( \sigma ) } ) ^ { - 1 } \mathrm { B } _ { [ i ] } ^ { \sigma } \big ( \mathbf { V } _ { [ i ] } - \big ( \mathbf { \Gamma } _ { [ i ] } ^ {  } \odot \mathbf { K } _ { [ i ] } \big ) \mathbf { S } _ { [ i ] } ^ { \top } \big ) = \mathrm { B } _ { [ i ] } ^ { \sigma - 1 } \mathbf { F } _ { [ i ] } .
$$

For the transformed operands of $( 7 / 7 )$ , the remaining gate powers sum to $1 - \sigma ,$ , giving

$$
\begin{array} { r l } & { \bigl ( \mathbf { V } _ { [ i ] } ^ { ( \sigma , \tau ) } \bigr ) ^ { \top } ( { \boldsymbol { \Gamma } } _ { [ i ] } ^ {  } \odot \mathbf { K } _ { [ i ] } ^ { ( \sigma , \tau ) } ) = \bigl ( \mathbf { F } _ { [ i ] } ^ { ( \sigma ) } \bigr ) ^ { \top } \mathrm { B } _ { [ i ] } ^ { 1 - \sigma } ( { \boldsymbol { \Gamma } } _ { [ i ] } ^ {  } \odot \mathbf { K } _ { [ i ] } ) } \\ & { \qquad = \mathbf { F } _ { [ i ] } ^ { \top } ( { \boldsymbol { \Gamma } } _ { [ i ] } ^ {  } \odot \mathbf { K } _ { [ i ] } ) . } \end{array}
$$

The same token-wise cancellation preserves the local output. The incoming-state read is unchanged. $\mathrm { A t } \left( \sigma , \tau \right) = \left( 1 , 1 \right)$ , this recovers the standard WY form (Yang et al., 2024c; 2025; Yang & Zhang, 2024).

Aside: channel permutations. As an aside, we note that channel permutations within a head also preserve the recurrence. On the key axis, the rotation argument applies with a permutation matrix P and consistently permuted decay $\mathbf { \Lambda } \mathbf { } \mathbf { } ^ { \prime } _ { t } = \mathbf { P } \pmb { \Lambda } _ { t } \mathbf { P } ^ { \mathsf { T } }$ , which remains diagonal. This holds for both linear and delta writes. Value-axis permutations follow from the equivariance above, with the inverse applied to the output. We do not explore permutations in our study: under our per-tensor, per-head, and per-channel quantizers, they only reorder quantized entries and their associated scales, leaving quantization error unchanged. They can matter under groupwise quantization, where channel order determines which entries share a scale (Zhang et al., 2026).

## D LOW-PRECISION REALIZATIONS AND KERNEL IMPLEMENTATION

This appendix describes the quantized chunkwise forms, the controlled and production kernels, and the placement and fusion of transformations. Figure 1 provides an architectural overview of these transformation locations.

## D.1 QUANTIZED CHUNKWISE FORMS

Section 3.1 gives the scalar-decay linear recurrence underlying Mamba-2. The other five families retain its state write, state pass, state read, local score, and intra-chunk output, but form their operands differently. Throughout this subsection, Q denotes quantization at the indicated boundary, with scale factors implicit. The write, read, and local-score contractions use INT8 operands with INT32 accumulation (except the 16-token diagonal blocks of the local score under diagonal decay, which compute the decay-weighted dot products in FP32 using dequantized queries and keys), with the corresponding scales applied to the accumulated products. The intra-chunk output uses FP16 arithmetic on INT8-quantized values.

Decay factors. We use the interval weights $\gamma _ { [ i ] } ^ { s  r }$ and stacked prefix and tail weights $\Gamma _ { [ i ] } ^ { \left. } , \Gamma _ { [ i ] } ^ { \right. }$ from Section 2.2.

No decay, linear write (LA). Setting $\gamma \equiv 1$ removes all decay factors. With an optional write gate, the chunk computation becomes

$$
\begin{array} { r l } & { \quad \hat { \bf H } _ { [ i ] } = { \mathcal Q } ( { \bf B } _ { [ i ] } { \bf V } _ { [ i ] } ) ^ { \top } { \mathcal Q } ( { \bf K } _ { [ i ] } ) , } \\ & { { \bf S } _ { [ i + 1 ] } = { \bf S } _ { [ i ] } + \hat { \bf H } _ { [ i ] } , \qquad \hat { \bf S } _ { [ i + 1 ] } = { \mathcal Q } ( { \bf S } _ { [ i + 1 ] } ) , } \\ & { \quad \hat { \bf O } _ { [ i ] } ^ { \mathrm { i n t e r } } = { \mathcal Q } ( { \bf Q } _ { [ i ] } ) \hat { \bf S } _ { [ i ] } ^ { \top } , } \\ & { \quad \hat { \bf A } _ { [ i ] } = \big ( { \mathcal Q } ( { \bf Q } _ { [ i ] } ) { \mathcal Q } ( { \bf K } _ { [ i ] } ) ^ { \top } \big ) \odot { \bf M } , } \\ & { \quad \hat { \bf O } _ { [ i ] } ^ { \mathrm { i n t r a } } = \hat { \bf A } _ { [ i ] } { \mathcal Q } ( { \bf B } _ { [ i ] } { \bf V } _ { [ i ] } ) . } \end{array}
$$

Diagonal decay, linear write (GLA). For positive decay gates, the masked product in Section 2.2 expands as

$$
[ \operatorname { t r i l } ( ( \mathbf { T } _ { [ i ] } ^ {  } \odot \mathbf { Q } _ { [ i ] } ) ( \frac { \mathbf { K } _ { [ i ] } } { \mathbf { T } _ { [ i ] } ^ {  } } ) ^ { \top } ) ] _ { r s } = M _ { r s } \sum _ { n } Q _ { [ i ] , r n } K _ { [ i ] , s n } \gamma _ { [ i ] , n } ^ { s  r } .\tag{2}
$$

The right-hand side also applies when decay gates are zero. Quantizing the contraction operands for positive gates gives

$$
\begin{array} { r l } & { \quad \widehat { \mathbf { H } } _ { [ i ] } = \mathcal { Q } ( \mathbf { B } _ { [ i ] } \mathbf { V } _ { [ i ] } ) ^ { \top } \mathcal { Q } ( \Gamma _ { [ i ] } ^ { \right. } \odot \mathbf { K } _ { [ i ] } ) , } \\ & { \mathbf { S } _ { [ i + 1 ] } = \mathbf { S } _ { [ i ] } \mathrm { d i a g } ( \gamma _ { [ i ] , 1 } ^ { C } , \ldots , \gamma _ { [ i ] , D _ { k } } ^ { C } ) + \widehat { \mathbf { H } } _ { [ i ] } , \qquad \widehat { \mathbf { S } } _ { [ i + 1 ] } = \mathcal { Q } ( \mathbf { S } _ { [ i + 1 ] } ) , } \\ & { \quad \widehat { \mathbf { O } } _ { [ i ] } ^ { \mathrm { i n t e r } } = \mathcal { Q } ( \mathbf { { \Gamma } } _ { [ i ] } ^ { \left. } \odot \mathbf { Q } _ { [ i ] } ) \widehat { \mathbf { S } } _ { [ i ] } ^ { \top } , } \\ & { \quad \widehat { A } _ { [ i ] , r s } = \displaystyle \sum _ { n } \mathcal { Q } \big ( Q _ { [ i ] , r n } \gamma _ { [ i ] , n } ^ { r } / \gamma _ { [ i ] , n } ^ { a } \big ) \mathcal { Q } \big ( K _ { [ i ] , s n } \gamma _ { [ i ] , n } ^ { a } / \gamma _ { [ i ] , n } ^ { s } \big ) , } \\ & { \quad \widehat { \mathbf { O } } _ { [ i ] } ^ { \mathrm { i n t r a } } = \widehat { \mathbf { A } } _ { [ i ] } \mathcal { Q } ( \mathbf { B } _ { [ i ] } \mathbf { V } _ { [ i ] } ) . } \end{array}\tag{3}
$$

where the displayed local score applies to causal pairs $s \leq r .$ . Entries above the causal diagonal are zero. The decay lies inside the reduction over channels, as in Equation (2). For cross-block scores, the kernel anchors the two decay factors at the first position a of the block containing r, so that both $\gamma _ { [ i ] , n } ^ { r } / \gamma _ { [ i ] , n } ^ { a }$ and $\gamma _ { [ i ] , n } ^ { a } / \gamma _ { [ i ] , n } ^ { s }$ are at most one, then quantizes and multiplies the anchored operands. Within a block, it uses quantized queries and keys and applies $\gamma _ { [ i ] , n } ^ { s  r }$ inside the FP32 sum. A query–key rotation can act on the anchored operands in registers, but cannot generally be applied once to the unweighted Q and K (Section 3).

Delta write (DeltaNet, GDN, KDA). Delta cells first construct pseudo-values that account for interactions among writes and with the incoming state. The Gram matrix, the triangular solve, and the WY intermediates are computed in FP32; only the incoming-state correction uses an INT8-quantized copy of $\mathbf { W } _ { [ i ] }$ against the INT8 copy of the incoming state. For absent or scalar decay, let $\Gamma _ { [ i ] }$ denote the causal decay mask. The compensated write-factor placement $( \sigma , \tau )$ gives

$$
\begin{array} { r l } & { { \bf G } _ { [ i ] } = \mathrm { t r i l } \Big ( \big ( ( \mathrm { B } _ { [ i ] } ^ { \sigma } { \bf K } _ { [ i ] } \big ) \big ( \mathrm { B } _ { [ i ] } ^ { 1 - \sigma } { \bf K } _ { [ i ] } \big ) ^ { \top } \big ) \odot \mathbf { \boldsymbol { \Gamma } } _ { [ i ] } , - 1 \Big ) , \quad { \bf \boldsymbol { \mathrm { T } } } _ { [ i ] } = \big ( { \bf \boldsymbol { \mathrm { \bf { I } } } } + { \bf { G } } _ { [ i ] } \big ) ^ { - 1 } , } \\ & { { \bf W } _ { [ i ] } = { \bf \boldsymbol { \mathrm { \bf { T } } } } _ { [ i ] } \mathrm { B } _ { [ i ] } ^ { \sigma } ( { \bf \boldsymbol { \mathrm { \bf { T } } } } _ { [ i ] } ^ {  } \odot { \bf { \boldsymbol { \mathrm { \bf { K } } } } } _ { [ i ] } ) , \quad \quad { \bf { U } } _ { [ i ] } = { \bf \boldsymbol { \mathrm { \bf { T } } } } _ { [ i ] } \mathrm { B } _ { [ i ] } ^ { \sigma } { \bf \boldsymbol { \mathrm { \bf { V } } } } _ { [ i ] } , } \\ & { \widehat { \bf F } _ { [ i ] } ^ { ( \sigma ) } = { \bf U } _ { [ i ] } - \mathcal { Q } ( { \bf W } _ { [ i ] } ) \widehat { \bf S } _ { [ i ] } ^ { \top } , } \\ & { { \bf V } _ { [ i ] } ^ { ( \sigma , \tau ) } = { \bf B } _ { [ i ] } ^ { ( 1 - \sigma ) ( 1 - \tau ) } \widehat { \bf F } _ { [ i ] } ^ { ( \sigma ) } , \quad \quad { \bf K } _ { [ i ] } ^ { ( \sigma , \tau ) } = { \bf B } _ { [ i ] } ^ { ( 1 - \sigma ) \tau } { \bf K } _ { [ i ] } , } \end{array}
$$

Here $\mathbf { U } _ { [ i ] } , \mathbf { W } _ { [ i ] }$ are WY intermediates, and $\widehat { \mathbf { F } } _ { [ i ] } ^ { ( \sigma ) }$ approximates the complete pseudo-values $\mathbf { F } _ { [ i ] } ^ { ( \sigma ) }$ using the quantized incoming-state correction. The transformed write operands ${ \bf V } _ { [ i ] } ^ { ( \sigma , \tau ) } , { \bf K } _ { [ i ] } ^ { ( \sigma , \tau ) }$ enter the chunk step:

$$
\begin{array} { r l } & { \quad \widehat { \mathbf { H } } _ { [ i ] } = \mathcal { Q } ( { \mathbf { V } } _ { [ i ] } ^ { ( \sigma , \tau ) } ) ^ { \top } \mathcal { Q } ( { \mathbf { \Gamma } } _ { [ i ] } ^ { \right. } \odot { \mathbf { K } } _ { [ i ] } ^ { ( \sigma , \tau ) } ) , } \\ & { \mathbf { S } _ { [ i + 1 ] } = \mathbf { S } _ { [ i ] } \mathbf { P } _ { [ i ] } + \widehat { \mathbf { H } } _ { [ i ] } , \qquad \widehat { \mathbf { S } } _ { [ i + 1 ] } = \mathcal { Q } ( \mathbf { S } _ { [ i + 1 ] } ) , } \\ & { \widehat { \mathbf { O } } _ { [ i ] } ^ { \mathrm { i n t e r } } = { \mathbf { \Gamma } } _ { [ i ] } ^ { \left. } \odot \big ( \mathcal { Q } ( \mathbf { Q } _ { [ i ] } ) \widehat { \mathbf { S } } _ { [ i ] } ^ { \top } \big ) , } \\ & { \quad \widehat { \mathbf { A } } _ { [ i ] } = \big ( \mathcal { Q } ( \mathbf { Q } _ { [ i ] } ) \mathcal { Q } ( { \mathbf { K } } _ { [ i ] } ^ { ( \sigma , \tau ) } ) ^ { \top } \big ) \odot \mathbf { \Gamma } _ { [ i ] } , } \\ & { \widehat { \mathbf { O } } _ { [ i ] } ^ { \mathrm { i n t r a } } = \widehat { \mathbf { A } } _ { [ i ] } \mathcal { Q } ( { \mathbf { V } } _ { [ i ] } ^ { ( \sigma , \tau ) } ) , } \end{array}
$$

where $\mathbf { P } _ { [ i ] }$ is I for DeltaNet, $\gamma _ { [ i ] } ^ { C } \mathbf { I }$ for GDN, and diag $( \gamma _ { [ i ] , 1 } ^ { C } , \dots , \gamma _ { [ i ] , D _ { k } } ^ { C } )$ for KDA. For KDA, both the Gram and local-score contractions place the channel-dependent decay inside the reduction, the local score using the anchored construction of Equation (3) in place of a scalar mask applied after the product, and the state read quantizes the decay-weighted query $\mathcal { Q } ( \mathbf { { T } } _ { [ i ] } ^ {  } \odot \mathbf { { Q } } _ { [ i ] } )$ . At (σ, τ) = (1, 1), with no decay and quantization removed, $\mathbf { V } _ { [ i ] } ^ { ( \sigma , \tau ) } = \mathbf { F } _ { [ i ] }$ recovers the pseudo-values in Section 2.2. Setting $\begin{array} { r } { \sigma = \frac { 1 } { 2 } } \end{array}$ instead assigns $\mathbf { B } _ { [ i ] } ^ { 1 / 2 }$ to each Gram operand.

Absorptions. For a linear write with scalar decay, the tail weights $\Gamma _ { [ i ] } ^ {  }$ and write gate $\mathbf { B } _ { [ i ] }$ can each be split between the two operands before quantization. With elementwise powers of the tail vector and broadcasting across channels, the write is

$$
\begin{array} { r } { \mathcal { Q } \Big ( ( \mathbf { \Gamma } _ { [ i ] } ^ {  } ) ^ { \eta } \odot ( \mathbf { B } _ { [ i ] } ^ { \xi } \mathbf { V } _ { [ i ] } ) \Big ) ^ { \top } \mathbf { \mathcal { Q } } \Big ( ( \mathbf { \Gamma } _ { [ i ] } ^ {  } ) ^ { 1 - \eta } \odot ( \mathbf { B } _ { [ i ] } ^ { 1 - \xi } \mathbf { K } _ { [ i ] } ) \Big ) \ . } \end{array}
$$

Here $\eta = 1$ and $\xi = 1$ put the respective factors on the value, while zero puts them on the key. Under diagonal decay, only the scalar maximum $e _ { [ i ] } ^ { r } = \operatorname* { m a x } _ { n } { ( \mathbf { \Gamma } _ { [ i ] } ^ {  } ) _ { r n } }$ moves. The relative channeldependent decay remains on the key. Delta cells use pseudo-values for decay absorption and the compensated $( \sigma , \tau )$ construction for write strength, as proved in Appendix C.

## D.2 KERNEL IMPLEMENTATION

Controlled study. Diagonal rescalings are applied to the operands before quantization, and absorptions redistribute token-wise factors already used by the write. Algorithm 1 shows one chunk step for scalar or no decay with canonical state storage. Unused axis transformations are set to the identity. Matrices stack tokens as rows, so the rotations of Section 3.2 act by right multiplication with $\mathbf { R } ^ { \top }$

Here, $\mathcal { Q } _ { b }$ quantizes operands to b-bit integers and deq restores the associated scales. Scale factors not shown explicitly are folded into the surrounding operations. The full-precision carry is $\mathbf { S } ,$ , and $\hat { \mathbf { S } } _ { [ i ] }$ is its INT8 read copy. The tail weights are $^ { g , }$ the prefix weights $^ { p , }$ the causal decay mask Γ, and the chunk decay $\gamma ^ { C }$

Algorithm 1 One INT8 chunk step with LinQuant transformations (per head).   
1: once: $\mathbf { Q }  \mathbf { Q } \mathrm { D } _ { Q K } , \mathbf { K }  \mathbf { K } \mathrm { D } _ { Q K } ^ { - 1 } , \mathbf { V }  \mathbf { V } \mathrm { D } _ { V }$ ▷ rescale operands   
2: once: $\mathbf { Q }  \mathbf { Q } \mathrm { R } _ { Q K } ^ { \top } , \mathbf { K }  \mathbf { K } \mathbf { R } _ { Q K } ^ { \top } , \mathbf { V }  \mathbf { V } \mathbf { R } _ { V } ^ { \top }$ ▷ rotate operands   
3: $\mathbf { i f } \delta = 0$ then   
4: $( \mathbf { K } , \mathbf { V } ) \gets ( \mathrm { B } _ { [ i ] } ^ { 1 - \xi } \mathbf { K } , \ \mathrm { B } _ { [ i ] } ^ { \xi } \mathbf { V } )$ ▷ write-gate absorption   
5: else   
6: $( \mathbf { K } , \mathbf { V } )  ( \mathbf { K } _ { [ i ] } ^ { ( \sigma , \tau ) } , \mathbf { V } _ { [ i ] } ^ { ( \sigma , \tau ) } )$ from Algorithm 2   
7: end if   
8: $\pmb { w } _ { k }  g ^ { 1 - \eta } , \pmb { w } _ { v }  g ^ { \eta }$ ▷ decay absorption   
9: $\mathbf { O } \gets p \odot \mathrm { d e q } \left( \mathscr { Q } _ { b } ( \mathbf { Q } ) \hat { \mathbf { S } } _ { [ i ] } ^ { \mathsf { T } } \right)$ ▷ state read   
10: $\mathbf { A } \gets \mathrm { d e q } \left( \mathcal { Q } _ { b } ( \mathbf { \dot { Q } } ) \mathcal { Q } _ { b } ( \mathbf { \dot { K } } ) ^ { \dagger } \right) \odot \boldsymbol { \Gamma }$ ▷ local score   
11: $\mathbf { O } _ { \mathrm { l o c } }  \mathbf { A } \stackrel { \cdot } { \mathcal { Q } } _ { b } ( \mathbf { V } )$ ▷ intra-chunk output   
12: $\mathbf { O } _ { \mathrm { l o c } }  \mathbf { O } _ { \mathrm { l o c } } \mathbf { \vec { R } } _ { V }$ ▷ restore value basis   
13: $\mathbf { O } \gets \mathbf { O } + \mathbf { O } _ { \mathrm { l o c } }$ ▷ combine outputs   
14: $\hat { \mathbf { K } } , \hat { \mathbf { V } } \gets \mathcal { Q } _ { b } \big ( \{ \pmb { w } _ { k } \odot \mathbf { K } \} , \mathcal { Q } _ { b } ( \pmb { w } _ { v } \odot \mathbf { V } )$ ▷ quantize write operands   
$1 5 \colon { \bf H }  \deg ( \hat { \mathbf { V } } ^ { 1 } \hat { \mathbf { K } } )$ ▷ state write   
16: $\mathbf { H }  \mathbf { R } _ { V } ^ { \intercal } \mathbf { H }$ ▷ restore value basis   
17: $\mathbf { S } \gets \mathbf { S } \boldsymbol { \gamma } ^ { C } + \mathbf { H } . \hat { \mathbf { S } } _ { [ i + 1 ] }  \mathcal { Q } _ { 8 } ( \mathbf { S } )$ ▷ carry and read copy

The initial rescalings and rotations are fused with quantization; the output is multiplied by $\mathbf { D } _ { V } ^ { - 1 }$ to undo value rescaling. The local path and state write quantize their operands separately. The chosen write-gate allocation is used in the local score, intra-chunk output, and state write; the delta-rule operands from Algorithm 2 already include it. The state read, local score, and state write use INT8 operands with INT32 accumulation, while the intra-chunk output uses FP16 arithmetic on quantized values. The state is carried in FP32, with an INT8 copy used by the next chunk.

Decay weights are unity for LA and DeltaNet, scalars for Mamba-2 and GDN, and channel-dependent vectors for GLA and KDA. Under diagonal decay, only the scalar magnitude $e _ { r } = \operatorname* { m a x } _ { n } g _ { r n }$ is redistributed: ${ \pmb w } _ { k } = { \pmb g } \odot { e } ^ { - \eta }$ and $\mathbf { \boldsymbol { w } } _ { v } = e ^ { \eta }$ . Prefix decay is applied to the query before quantization in the state read, and the chunk decay becomes $\mathrm { d i a g } ( \gamma _ { n } ^ { \bar { C } } )$ . The local score uses the construction in Appendix D.1, including its FP32 computation within each 16-token block. Query–key rotations act only on the decay-weighted operands of Equation (3), before quantization.

Canonical storage under $\mathbf { R } _ { V }$ adds matrix products inside the call. Each chunk’s rotated write is multiplied by ${ \bf R } _ { V } ^ { \top }$ before it joins the FP32 carry, and the local output is returned to the original basis. The state read stays canonical. For canonical storage, delta cells right-multiply the incomingstate correction $\boldsymbol { \mathcal { Q } } ( \mathbf { W } _ { [ i ] } ) \widehat { \mathbf { S } } _ { [ i ] } ^ { \top }$ by $\mathbf { R } _ { V } ^ { \top }$ before subtracting it from $\mathbf { U } _ { [ i ] }$ , so both terms use the same rotated value basis. With rotated storage, the inverse rotations of the local output and state write are replaced by one $\mathbf { O }  \mathbf { O } \mathbf { R } _ { V }$ at the end of the chunk. This incurs the accuracy penalty discussed in Appendix E.1. The canonical delta kernels hold the full state and the $D _ { v } \times D _ { v }$ rotation operand per program.

Algorithm 2 WY preparation with compensated write-factor placement $( \sigma , \tau )$ , per head and chunk.   
$\mathbf { M } _ { < }$ is the strictly lower causal mask, including scalar decay when present. Diagonal decay is applied   
inside the Gram reduction as described above. The Gram matrix, solve, and WY intermediates   
are $\mathrm { F P } 3 2 ; \widehat { \mathbf { S } } _ { \mathrm { i n } }$ is the INT8-quantized incoming state and $\mathbf { W } _ { [ i ] }$ is the only quantized operand of the   
preparation. The resulting transformed write operands are used in the chunk step.   
1: $\mathbf { K } _ { [ i ] } ^ { \sigma } \gets \mathbb { B } _ { [ i ] } ^ { \sigma } \mathbf { K } _ { [ i ] } , \mathbf { K } _ { [ i ] } ^ { 1 - \sigma } \gets \mathbb { B } _ { [ i ] } ^ { 1 - \sigma } \mathbf { K } _ { [ i ] }$ ▷ Gram factors   
2: $\mathbf { G } _ { [ i ] }  ( \mathbf { K } _ { [ i ] } ^ { \sigma } \ ( \mathbf { K } _ { [ i ] } ^ { 1 - \sigma } ) ^ { \mathsf { T } } ) \odot \mathbf { M } _ { < }$ ▷ Gram matrix, fp32   
3: $\mathbf { T } _ { [ i ] }  ( \mathbf { I } + \mathbf { G } _ { [ i ] } ) ^ { - 1 }$ ▷ unit lower-triangular solve, fp32   
4: $\mathbf { \dot { W } } _ { [ i ] } ^ {  }  \mathbf { T } _ { [ i ] } ( \mathbf { K } _ { [ i ] } ^ {  } \odot \underline { { \Gamma } } _ { [ i ] } ^ {  } ) , \mathbf { U } _ { [ i ] }  \mathbf { T } _ { [ i ] } \mathbb { B } _ { [ i ] } ^ { \sigma } \mathbf { V } _ { [ i ] }$ ▷ WY intermediates, fp32   
5: $\mathbf { C } _ { [ i ] }  \setminus ( \mathbf { W } _ { [ i ] } ) \widehat { \mathbf { S } } _ { \mathrm { i n } } ^ { \mathsf { T } }$ ▷ incoming-state correction   
6: Canonical $\mathbf { R } _ { V } \colon \mathbf { \dot { C } } _ { [ i ] }  \mathbf { C } _ { [ i ] } \mathbf { R } _ { V } ^ { \top }$ ▷ match the rotated value basis of $\mathbf { U } _ { [ i ] }$   
7: $\widehat { \mathbf { F } } _ { [ i ] } ^ { ( \sigma ) } \gets \mathbf { U } _ { [ i ] } - \mathbf { C } _ { [ i ] }$ ▷ complete pseudo-values   
8: $\mathbf { K } _ { [ i ] } ^ { \left( \sigma , \tau \right) } \gets \mathbb { B } _ { [ i ] } ^ { ( 1 - \sigma ) \tau } \mathbf { K } _ { [ i ] } , \mathbf { V } _ { [ i ] } ^ { \left( \sigma , \tau \right) } \gets \mathbb { B } _ { [ i ] } ^ { ( 1 - \sigma ) ( 1 - \tau ) } \widehat { \mathbf { F } } _ { [ i ] } ^ { \left( \sigma \right) }$ ▷ transformed write operands   
9: return ${ \bf K } _ { [ i ] } ^ { ( \sigma , \tau ) } , { \bf V } _ { [ i ] } ^ { ( \sigma , \tau ) }$ for Algorithm 1

Production Mamba-2. The production implementation adds state-axis transformations to SSDi8’s dataflow, as summarized in Algorithm 3. The added $\mathbf { D } _ { Q K }$ rescaling is applied as a per-channel multiply inside the fused split-and-quantize pass over the convolution output, followed by $\mathbf { R } _ { Q K }$ in the same pass. Neither transformation requires a separate kernel launch or memory pass in this implementation.

Algorithm 3 The Mamba-2 kernel of LinQuant: SSDi8’s dataflow with $\mathbf { R } _ { Q K }$ and $\mathbf { D } _ { Q K }$ . Keys and   
queries are shared by the G groups and read by the H heads. $g ( h )$ denotes the group of head $h .$ w   
denotes the within-chunk decay weights (tail factors for the write, prefix factors for the read), and $\gamma _ { [ i ] } ^ { C }$   
denotes the chunk decay.   
Once per layer (the fused pass over the convolution output)   
1: $\mathbf { K } \gets \mathbf { K } \mathbf { D } _ { Q K } ^ { - 1 } \mathbf { R } _ { Q K } ^ { \top } , \mathbf { Q } \gets \mathbf { Q } \mathbf { D } _ { Q K } \mathbf { R } _ { Q K } ^ { \top } \hat { \mathbf { K } } \gets \mathcal { Q } _ { 8 } ( \mathbf { K } , s _ { K } ) , \hat { \mathbf { Q } } \gets \mathcal { Q } _ { 8 } ( \mathbf { Q } , s _ { Q } )$ ▷ per group. The   
quantized queries and keys are reused by three stages   
Per chunk $[ i ]$ and head h   
2: $\hat { \mathbf { V } }  \mathcal { Q } _ { 8 } ( w _ { \mathrm { t a i l } } \odot \mathbf { V } , s _ { V } )$ , tile by tile in registers ▷ decay on the value. Static $\mathsf { p e r } \mathsf { - } ( H , D _ { v } )$ scale or   
dynamic per tile. Key side: $\hat { \mathbf { K } } \gets \mathcal { Q } _ { 8 } ( w _ { \mathrm { t a i l } } \odot \mathbf { K } )$ from an fp re-read instead   
3: $\mathbf { H }  \hat { \mathbf { V } } ^ { \top } \hat { \mathbf { K } } _ { g ( h ) }$ ▷ write: INT8×INT8, INT32 accumulation, one rescale   
4: $\hat { \mathbf { S } } _ { [ i + 1 ] }  \mathrm { f x } ( \gamma _ { [ i ] } ^ { C } \hat { \mathbf { S } } _ { [ i ] } ) + \mathcal { Q } _ { 8 } ( \mathbf { H } , s _ { S } )$ ▷ state pass in INT8 fixed point, $\mathsf { p e r } \mathsf { - } ( H , D _ { v } )$ scale   
5: $\hat { \mathbf { Z } }  \mathcal { Q } _ { 8 } ( \hat { \mathbf { Q } } _ { g ( h ) } \hat { \mathbf { K } } _ { g ( h ) } ^ { \mathsf { T } } )$ ▷ local score: INT32 accumulation, requantized per group   
6: $\mathbf { O } \gets w _ { \mathrm { p r e f } } \odot \mathrm { d e q } ( \hat { \mathbf { Q } } _ { g ( h ) } \hat { \mathbf { S } } _ { [ i ] } ^ { \mathsf { T } } ) + ( \mathrm { d e q } ( \hat { \mathbf { Z } } ) \odot \boldsymbol { \Gamma } ) \mathbf { V }$ ▷ read on INT8. Intra-chunk output fp16, the scale of $\hat { \mathbf { Z } }$   
folded into Γ

Transformation placement and rotation fusion. We implement the online rotations using structured signed Hadamard transforms. Forward rotations are fused into kernels that already materialize the activations. The query–key rotation joins the split-and-quantize kernel after convolution. The value rotation joins value quantization. The inverse value rotation must restore the original basis before an incompatible downstream operation. In our Mamba-2 implementation, the gated RMSNorm after the recurrence normalizes across the full hidden dimension. We therefore apply the per-head inverse rotation at the end of the recurrence-output kernel, before this normalization.

Applying decay in registers. Keys are shared by groups, while decay weights can differ across heads. A direct implementation of key-side absorption could therefore materialize a larger, headindexed tensor of weighted keys. Our write kernel avoids this intermediate tensor. Each program loads its group’s key tile and its head’s decay weights, then forms the weighted entries in registers:

$$
k _ { r , n } ^ { \prime } = w _ { h , r } k _ { g ( h ) , r , n } ,
$$

The weighted tile is quantized and clipped before the INT8 tensor-core contraction. Programs for heads in the same group read the same underlying key tile and apply their own weights, so head-specific values never need to be stored as an expanded floating-point tensor.

## E CONTROLLED-STUDY RESULTS

This appendix reports the controlled-study results for Section 4.1, using the setup in Appendix B.1. We organize them by key-axis transformations (1/7–2/7), value-axis transformations (3/7–4/7), and absorptions (5/7–7/7), followed by compact quality summaries, precision controls, and downstream accuracy. The complete numerical grids and per-task breakdowns are retained in the supplementary source.

## E.1 INDIVIDUAL TRANSFORMATIONS

Key-axis transformations (1/7)-(2/7). Figure 6 shows the response to query–key rescaling, with separate panels for cells and curves for scale regimes. The query-key rescaling sweep tests both range balancing and its composition with rotation.

![](images/ffca0cd1d3f239dbd12c991e1591ddd744c88e0ef35510eaa7b1c0f13e952839.jpg)  
Figure 6: Response to the exponent κ of $\mathbf { D } _ { Q K }$ , alone (solid) and composed with $\mathbf { R } _ { Q K }$ (dashed, hollow), per linear cell and regime, INT6-quantized operands, INT8 stored state: percent of the naive excess NLL recovered. At κ = 0 the dashed curve is $\mathbf { R } _ { Q K }$ alone.

Value-axis transformations (3/7)-(4/7). Figure 7 shows the value-rescaling sweep, with separate panels for cells and curves for scale regimes. The value-rescaling sweep isolates the role of scale granularity: consistent per-channel scales absorb the rescaling, while shared scales allow it to change quantization error.

![](images/3af5273077d254b8155d6850d6d4e6a5dd87ab74cc2dba28a3c4e70f956123bf.jpg)  
Figure 7: Response to the exponent κ of $\mathbf { D } _ { V }$ , percent of the naive excess NLL recovered: per-channel scales absorb a consistent diagonal rescaling, whereas shared scales allow the exponent to change the error. The full grid $\kappa \in \{ \frac { 1 } { 4 } , \frac { 1 } { 2 } , \frac { 3 } { 4 } , 1 \}$ was run only under coarse scales ⟨T, T⟩.

State-storage basis. Figure 8 compares canonical and rotated storage for every cell at INT6. The stored state uses one scale per head and value channel. A value rotation mixes these channels before quantization, changing the range covered by each scale. The largest benefit from canonical storage occurs in LA. Decaying and delta-rule cells are much less sensitive, although the sign and size of the difference depend on the configuration.

![](images/d3d79d1360b104168df496b93fb2626aa9ba7c4af7e7dbff9f1139642dd0c1bc.jpg)  
Figure 8: Full storage-basis control for the 340M zoo. Excess NLL of rotated minus canonical storage under $\mathbf { R } _ { V }$ , as a percentage of each model’s naive excess NLL, per scale regime, INT6-quantized operands. Positive values favor canonical storage. Values use rounded per-configuration perplexities.

Canonical value-rotation overhead. Table 4 compares the transformation stack with canonical R<sub>V</sub> against naive INT8 across GPUs at a fixed batch size, complementing Figure 5b. The timing methodology and boundaries are given in Appendix B.1.

Table 4: Canonical value-rotation overhead across GPUs: median INT8 kernel time for the transformation stack with $\mathbf { R } _ { V }$ relative to naive INT8 (1×: equal time). $B = H = 8 , T = 2 0 4 8$ , and $D _ { k } = D _ { v } = 1 2 8 ;$ ; methodology is in Appendix B.1.
<table><tr><td>Recurrence</td><td>RTX A5000</td><td>A100-SXM4</td><td>H100</td></tr><tr><td>LA</td><td>1.61</td><td>1.24</td><td>1.60</td></tr><tr><td>Mamba-2</td><td>1.42</td><td>1.42</td><td>1.60</td></tr><tr><td>GLA</td><td>1.36</td><td>1.72</td><td>1.59</td></tr><tr><td>DN</td><td>3.37</td><td>10.56</td><td>2.43</td></tr><tr><td>GDN</td><td>2.84</td><td>7.90</td><td>2.27</td></tr><tr><td>KDA</td><td>1.88</td><td>4.99</td><td>1.77</td></tr></table>

Absorptions (5/7)-(7/7). The absorption sweeps compare where token-wise factors enter the quantized computation. Scalar-decay placement has relatively small effects in the deploymentoriented regimes (Figure 9). For gated linear writes, the value-side endpoint $\xi = 1$ performs best under $\langle \mathrm { T } , \bar { \mathrm { C } } \rangle$ , while the other regimes can favor an interior split. For the delta cells under coarse scales, the selection procedure moves the compensation to the keys $( \tau = 1 )$ and chooses $\sigma \in \{ 0 , \frac { 1 } { 2 } \}$ with small differences between placements. The individual curves make clear how these conclusions depend on the cell and scale regime.

![](images/224f7eb877a64805a3f538612a84b2f02f804dd3cc02f6017775fdca305eba71.jpg)  
Figure 9: Response to the decay allocation η (decay on the key at 0, on the value at 1) for the decaying cells, INT6 operands and INT8 stored state: percent of the naive excess NLL recovered, one curve per scale regime. Placing scalar decay on the keys raises the excess by 21–27 % on M2+β and GDN under ⟨T, T⟩. Diagonal decay and the per-channel-value regimes stay within a few percent.

![](images/36656c64347eac6816e046c0cc92a8ac1c4d91b03ce3225a02aa062b8ff6f401.jpg)  
write-factor allocation ξ (1: all on the value) ⟨H,C⟩ ⟨T,C⟩ ⟨T,T⟩

Figure 10: Write-factor allocation on the six gated linear models, with INT6 operands and INT8 stored state: $\mathbf { B } _ { [ i ] } ^ { \xi }$ multiplies values and $\mathbf { B } _ { [ i ] } ^ { 1 - \xi }$ multiplies keys. Percent of the naive excess NLL recovered. Each curve shows one scale regime. † marks models without QK normalization.  
![](images/76e1c9f07b4e4628d4e0eab7ffc3b1fed289a8ef9e453bf7ca0702a2ee6c4791.jpg)  
Figure 11: Delta write-factor allocation with INT6 operands and INT8 stored state. Curves vary the Gram exponent σ at fixed τ, which allocates the remaining factor between the transformed write operands: solid $\tau = 1$ , dashed $\tau = 1 / 2$ , dotted $\tau = 0$ . Percent of the naive excess NLL recovered. Each colour denotes one scale regime.

## E.2 COMBINED TRANSFORMATIONS AND PRECISION CONTROLS

Figure 13 compares the two shared-key regimes at INT6. Key scales are held fixed and only value scales change, from per channel on the left to per tensor on the right. State-axis transformations can help in either panel, whereas the larger value-axis gains emerge when values share a scale.

Scale granularity and combined transformations. Figures 2c and 3c isolate the effects of coarsening key and value scales. Tables 5 and 6 summarize INT6 quality at both model sizes, retaining every recurrence reported in the corresponding full grids. Coarse value scales particularly affect the delta-rule models; value-axis transformations recover much of this loss. Combining transformations can improve further, although the selected stack does not outperform every individual setting in every configuration.

![](images/fa1bf71ce6ab77c579028348761b82a1aad7863ca7271eb693776d908eb255db.jpg)  
Figure 12: Write error of DeltaNet 340M against the gate allocation a $\begin{array} { r } { ( \sigma = 1 - a , \tau = 1 ; F _ { a } = B ^ { - a } F , } \end{array}$ $\check { K _ { a } } = B ^ { a } K$ via the matched solve), with INT6 static per-tensor write scales recalibrated per allocation on the calibration split. Relative Frobenius error of the quantized write against the full-precision $H = F ^ { \top } K$ , quantizing both operands (solid), the corrected values only, or the keys only (dashed). Per-head medians over 256 held-out WikiText-2 chunks at layers 3, 9, 15 and 21; the curve is the median over all heads and the band the interquartile range. Here, a = 0 is value-side absorption and a = 1 is key-side absorption.

![](images/13f998d76453b9d50432ffb39588d0d4aed2edfd869f7bdde4a6be336f480b23.jpg)  
Figure 13: Percentage of the naive quantizer’s excess NLL recovered per cell and invariance at its selected setting (negative: the invariance increases the loss), shared keys with per-channel values (left) and with per-tensor values (right).

Each individual invariance uses its held-out-selected parameters. “Best single” is the lowest reported test perplexity among these individual settings, including either storage basis for value rotation but excluding compositions. It is a descriptive oracle comparison, not a selection rule. The stack retains the original held-out selection and canonical storage.

Precision sensitivity. Table 7 contrasts INT8 and INT4. With per-channel value scales, several normalized decaying and delta-rule models are already close to full precision at INT8, leaving little room for a promoted transformation. At INT4, the stack can reduce perplexity substantially but does not consistently recover full-precision quality. The gap is particularly large with coarse value scales in the 1.3B zoo.

The write-scale control in Table 8 shows no improvement from dynamic scaling in the tested configurations. These experiments use 4-bit integer ranges within INT8 kernels; deployment below INT8 therefore requires both a more accurate quantization scheme and an efficient sub-byte implementation.

Table 5: INT6 WikiText-2 perplexity for the 340M zoo. Keys use per-tensor scales; value scales are per channel ⟨T, C⟩ or per tensor $\langle \mathrm { T } , \mathrm { T } \rangle$ . Best single is the descriptive comparison defined in the text; the stack is selected on held-out data and uses canonical storage. A dash denotes no reported promoted stack. †: no QK normalization.
<table><tr><td></td><td></td><td colspan="3">(T, C)</td><td colspan="3">(T, T)</td></tr><tr><td>Recurrence</td><td>FP</td><td>Naive</td><td>Best single</td><td>Stack</td><td>Naive</td><td>Best single</td><td>Stack</td></tr><tr><td>LA</td><td>17.117</td><td>25.706</td><td>19.428</td><td>18.097</td><td>30.694</td><td>23.151</td><td>18.138</td></tr><tr><td> $\mathrm { L A } { + } \beta$ </td><td>16.688</td><td>19.631</td><td>17.499</td><td>17.089</td><td>28.381</td><td>19.773</td><td>17.388</td></tr><tr><td>Mamba-2</td><td>15.553</td><td>16.029</td><td>15.722</td><td>15.605</td><td>16.446</td><td>16.036</td><td>15.633</td></tr><tr><td>Mamba-2+β</td><td>15.420</td><td>15.595</td><td>15.485</td><td>15.449</td><td>16.230</td><td>15.609</td><td>15.473</td></tr><tr><td>GLA</td><td>15.104</td><td>15.651</td><td>15.302</td><td>15.302</td><td>16.048</td><td>15.646</td><td>15.309</td></tr><tr><td>GLA+β</td><td>15.308</td><td>15.540</td><td>15.418</td><td>15.415</td><td>16.046</td><td>15.547</td><td>15.440</td></tr><tr><td>DeltaNet</td><td>15.678</td><td>15.925</td><td>15.778</td><td>15.769</td><td>20.323</td><td>16.138</td><td>15.871</td></tr><tr><td>GDN</td><td>15.004</td><td>15.168</td><td>15.055</td><td>15.042</td><td>17.064</td><td>15.236</td><td>15.103</td></tr><tr><td>KDA</td><td>14.489</td><td>14.846</td><td>14.843</td><td></td><td>15.837</td><td>14.885</td><td>14.906</td></tr><tr><td> $\mathrm { L A } ^ { \dagger }$ </td><td>17.230</td><td>24.916</td><td>18.079</td><td>18.071</td><td>31.582</td><td>22.486</td><td>18.336</td></tr><tr><td> $\mathrm { L A } { + } \beta ^ { \dagger }$ </td><td>16.980</td><td>30.719</td><td>21.042</td><td>20.738</td><td>100.247</td><td>35.441</td><td>20.429</td></tr><tr><td> $\mathbf { M a m b a - } 2 ^ { \dagger }$ </td><td>16.721</td><td>20.842</td><td>17.086</td><td>16.848</td><td>21.479</td><td>17.506</td><td>16.924</td></tr><tr><td> $\mathrm { M a m b a } { - 2 + \beta } ^ { \dagger }$ </td><td>16.189</td><td>17.665</td><td>16.306</td><td>16.264</td><td>18.844</td><td>17.213</td><td>16.326</td></tr><tr><td> $\mathrm { G L A ^ { \dagger } }$ </td><td>16.070</td><td>20.282</td><td>16.740</td><td>16.739</td><td>20.977</td><td>17.168</td><td>16.907</td></tr><tr><td> $\mathrm { G L A } { + } \beta ^ { \dagger }$ </td><td>15.891</td><td>19.222</td><td>16.834</td><td>16.838</td><td>20.523</td><td>17.873</td><td>16.647</td></tr></table>

Table 6: INT6 WikiText-2 perplexity for the 1.3B zoo. Keys use per-tensor scales; value scales are per channel ⟨T, C⟩ or per tensor $\langle \mathrm { T } , \mathrm { T } \rangle$ . Best single is the descriptive comparison defined in the text; the stack is selected on held-out data and uses canonical storage. A dash denotes no reported promoted stack.
<table><tr><td></td><td></td><td colspan="3">(T, C)</td><td colspan="3">(T, T)</td></tr><tr><td>Recurrence</td><td>FP</td><td>Naive</td><td>Best single</td><td>Stack</td><td>Naive</td><td>Best single</td><td>Stack</td></tr><tr><td> $\mathrm { L A } { + } \beta$ </td><td>10.276</td><td>12.101</td><td>10.768</td><td>10.625</td><td>148.173</td><td>12.588</td><td>10.842</td></tr><tr><td> $\mathbf { M a m b a } – 2 + \beta$ </td><td>9.755</td><td>9.839</td><td>9.805</td><td>9.789</td><td>12.687</td><td>9.895</td><td>9.802</td></tr><tr><td>GLA+β</td><td>9.189</td><td>9.307</td><td>9.267</td><td>9.260</td><td>10.115</td><td>9.316</td><td>9.263</td></tr><tr><td>DeltaNet</td><td>9.917</td><td>10.050</td><td>9.979</td><td>9.973</td><td>46.639</td><td>10.406</td><td>10.111</td></tr><tr><td>GDN</td><td>9.727</td><td>9.784</td><td>9.751</td><td>9.752</td><td>11.401</td><td>9.867</td><td>9.788</td></tr><tr><td>KDA</td><td>9.460</td><td>9.603</td><td>9.593</td><td></td><td>10.741</td><td>9.642</td><td>9.653</td></tr></table>

State precision and write-scale controls. Table 9 separates state-read error from error fed back through the carry. Keeping the read state in floating point reduces error most strongly in accumulating recurrences. Carrying the requantized state into subsequent updates adds error beyond that of an INT8 read copy. The INT6 comparison retains all 340M recurrences at a fixed scale policy.

Refining write-key scales alone does not resolve the error from coarse read keys (Table 10). At INT6 it recovers part of the loss in linear-write models, with much smaller effects in delta-rule models. INT32 local-score accumulation and the FP16 intra-chunk output product remain fixed in these controls.

## E.3 DOWNSTREAM ACCURACY

Tables 11 and 12 retain LAMBADA and the six-task average for every reported recurrence. The stack largely closes the gap to full precision for 340M LA and gated LA. At 1.3B, gated LA recovers most of the LAMBADA accuracy lost to naive quantization, although a gap remains. The other normalized recurrent models change little under this INT8 setting.

Table 7: Precision sensitivity on WikiText-2. INT8 uses per-channel value scales; INT4 includes both value-scale policies. Keys use per-tensor scales throughout. The stack is held-out-selected; dashes denote no reported promoted stack. Quantized settings retain the FP32 carry and INT8 read copy. †: no QK normalization.
<table><tr><td colspan="2"></td><td colspan="2">INT8,  $\langle \mathrm { T } , \mathrm { C } \rangle$ </td><td colspan="2">INT4,  $\langle \mathrm { T } , \mathrm { C } \rangle$ </td><td colspan="2">INT4,  $\langle \mathrm { T } , \mathrm { T } \rangle$ </td></tr><tr><td>Recurrence</td><td>FP</td><td>Naive</td><td>Stack</td><td>Naive</td><td>Stack</td><td>Naive</td><td>Stack</td></tr><tr><td>340M</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LA</td><td>17.117</td><td>19.482</td><td>17.265</td><td>73.226</td><td>24.736</td><td>498.768</td><td>32.750</td></tr><tr><td> $\mathrm { L A } { + } \beta$ </td><td>16.688</td><td>17.541</td><td>16.798</td><td>83.120</td><td>22.495</td><td>726.570</td><td>73.118</td></tr><tr><td> $\mathbf { M a m b a } { - 2 }$ </td><td>15.553</td><td>15.612</td><td>15.561</td><td>25.812</td><td>16.427</td><td>85.441</td><td>17.141</td></tr><tr><td> $\mathbf { M a m b a } – 2 + \beta$ </td><td>15.420</td><td>15.443</td><td></td><td>21.059</td><td>16.070</td><td>55.010</td><td>16.748</td></tr><tr><td> $\mathrm { G L A }$ </td><td>15.104</td><td>15.124</td><td></td><td>32.431</td><td>20.587</td><td>88.884</td><td>21.214</td></tr><tr><td> $\mathrm { G L A } { + } \beta$ </td><td>15.308</td><td>15.322</td><td></td><td>24.896</td><td>18.766</td><td>70.122</td><td>19.410</td></tr><tr><td> $_ \mathrm { D e l t a N e t }$ </td><td>15.678</td><td>15.698</td><td></td><td>24.523</td><td>17.685</td><td>168.077</td><td>20.072</td></tr><tr><td> $\mathrm { G D N }$ </td><td>15.004</td><td>15.019</td><td></td><td>20.540</td><td>16.098</td><td>82.814</td><td>17.421</td></tr><tr><td> $\mathrm { K D A }$ </td><td>14.489</td><td>14.511</td><td></td><td>28.732</td><td></td><td>111.449</td><td>30.205</td></tr><tr><td> $\mathrm { L A } ^ { \dagger }$ </td><td>17.230</td><td>17.797</td><td>17.388</td><td>314.039</td><td>37.950</td><td>436.488</td><td>66.355</td></tr><tr><td> $\mathrm { L A } { + } \beta ^ { \dagger }$ </td><td>16.980</td><td>18.744</td><td>17.902</td><td>515.217</td><td>69.456</td><td>663.576</td><td>94.058</td></tr><tr><td> $\mathrm { M a m b a } { - 2 } ^ { \dagger }$ </td><td>16.721</td><td>16.911</td><td>16.735</td><td>316.762</td><td>23.555</td><td>362.199</td><td>25.470</td></tr><tr><td> $\mathrm { M a m b a } { - 2 + \beta } ^ { \dagger }$ </td><td>16.189</td><td>16.266</td><td>16.194</td><td>202.732</td><td>18.344</td><td>285.469</td><td>22.110</td></tr><tr><td> $\mathrm { G L A } ^ { \dagger }$ </td><td>16.070</td><td>16.282</td><td>16.099</td><td>173.810</td><td>46.248</td><td>263.196</td><td>54.814</td></tr><tr><td> $\mathrm { G L A } { + } \beta ^ { \dagger }$ </td><td>15.891</td><td>16.009</td><td>15.935</td><td>172.781</td><td>42.062</td><td>244.264</td><td>45.162</td></tr><tr><td colspan="8">1.3B</td></tr><tr><td> $\mathrm { L A } { + } \beta$ </td><td>10.276</td><td>10.763</td><td>10.349</td><td>60.518</td><td>13.987</td><td>4185.731</td><td>44.904</td></tr><tr><td> $\mathbf { M a m b a } – 2 + \beta$ </td><td>9.755</td><td>9.774</td><td></td><td>11.767</td><td>10.185</td><td>295.661</td><td>10.971</td></tr><tr><td> $\mathrm { G L A } { + } \beta$ </td><td>9.189</td><td>9.205</td><td></td><td>12.509</td><td>10.786</td><td>160.470</td><td>11.423</td></tr><tr><td>DeltaNet</td><td>9.917</td><td>9.934</td><td></td><td>13.753</td><td>11.322</td><td>957.275</td><td>15.235</td></tr><tr><td>GDN</td><td>9.727</td><td>9.731</td><td></td><td>11.916</td><td>10.329</td><td>424.986</td><td>16.016</td></tr><tr><td>KDA</td><td>9.460</td><td>9.467</td><td></td><td>20.150</td><td>19.877</td><td>352.929</td><td>25.376</td></tr></table>

Table 8: WikiText-2 perplexity for the static and dynamic per-chunk write-scale controls at INT4, with per-head read-key scales and INT8 stored states (340M models).
<table><tr><td>Model</td><td>Static write scales</td><td>Dynamic write scales</td></tr><tr><td>LA</td><td>22.7</td><td>26.2</td></tr><tr><td>Mamba-2</td><td>16.1</td><td>16.4</td></tr><tr><td>GDN</td><td>18.4</td><td>18.8</td></tr></table>

Table 9: State-precision control at INT6 for all 340M recurrences, with per-head keys, per-channel values, and no invariance transforms. FP state leaves the state unquantized; INT8 read quantizes a read copy while retaining an FP32 carry; INT8 carry feeds the requantized state into subsequent updates. Entries are WikiText-2 perplexities. †: no QK normalization.
<table><tr><td>Recurrence</td><td>FP model</td><td>FP state</td><td>INT8 read</td><td>INT8 carry</td></tr><tr><td> $\mathrm { L A }$ </td><td>17.117</td><td>19.954</td><td>22.349</td><td>23.963</td></tr><tr><td> $\mathrm { L A } { + } \beta$ </td><td>16.688</td><td>17.373</td><td>18.209</td><td>19.118</td></tr><tr><td> $\mathbf { M a m b a } { - 2 }$ </td><td>15.553</td><td>15.763</td><td>15.795</td><td>15.914</td></tr><tr><td> $\mathbf { M a m b a } – 2 + \beta$ </td><td>15.420</td><td>15.518</td><td>15.524</td><td>15.545</td></tr><tr><td> $\mathrm { G L A }$ </td><td>15.104</td><td>15.307</td><td>15.312</td><td>15.360</td></tr><tr><td> $\mathrm { G L A } { + } \beta$ </td><td>15.308</td><td>15.457</td><td>15.456</td><td>15.566</td></tr><tr><td> $_ \mathrm { D e l t a N e t }$ </td><td>15.678</td><td>15.870</td><td>15.867</td><td>16.012</td></tr><tr><td> $\mathrm { G D N }$ </td><td>15.004</td><td>15.120</td><td>15.122</td><td>15.153</td></tr><tr><td> $\mathrm { K D A }$ </td><td>14.489</td><td>14.760</td><td>14.764</td><td>14.861</td></tr><tr><td> $\mathrm { L A } ^ { \dagger }$ </td><td>17.230</td><td>18.294</td><td>18.502</td><td>19.443</td></tr><tr><td> $\mathrm { L A } { + } \beta ^ { \dagger }$ </td><td>16.980</td><td>20.621</td><td>21.446</td><td>22.277</td></tr><tr><td> $\mathbf { M a m b a - } 2 ^ { \dagger }$ </td><td>16.721</td><td>17.106</td><td>17.115</td><td>17.328</td></tr><tr><td> $\mathrm { M a m b a } { - 2 + \beta ^ { \dagger } }$ </td><td>16.189</td><td>16.384</td><td>16.391</td><td>16.577</td></tr><tr><td> $\mathrm { G L A } ^ { \dagger }$ </td><td>16.070</td><td>18.087</td><td>18.091</td><td>18.326</td></tr><tr><td> $\mathrm { G L A } { + } \beta ^ { \dagger }$ </td><td>15.891</td><td>16.709</td><td>16.703</td><td>16.818</td></tr></table>

Table 10: Write-key scale control at INT6 for all 340M recurrences, with an FP32 carry and INT8 read copy. Read keys keep per-tensor scales. Shared uses the same per-tensor keys for reads and writes; fine write uses a separate per-channel write-key quantizer. No invariance transforms are applied. Entries are WikiText-2 perplexities; † marks no QK normalization.
<table><tr><td colspan="2"></td><td colspan="2">Per-channel values</td><td colspan="2">Per-tensor values</td></tr><tr><td>Recurrence</td><td>FP</td><td>Shared</td><td>Fine write</td><td>Shared</td><td>Fine write</td></tr><tr><td>LA</td><td>17.117</td><td>25.706</td><td>22.422</td><td>30.694</td><td>26.380</td></tr><tr><td> $\mathrm { L A } { + } \beta$ </td><td>16.688</td><td>19.631</td><td>18.312</td><td>28.381</td><td>26.888</td></tr><tr><td>Mamba-2</td><td>15.553</td><td>16.029</td><td>15.796</td><td>16.446</td><td>16.242</td></tr><tr><td> $\mathbf { M a m b a } – 2 + \beta$ </td><td>15.420</td><td>15.595</td><td>15.536</td><td>16.230</td><td>16.161</td></tr><tr><td>GLA</td><td>15.104</td><td>15.651</td><td>15.325</td><td>16.048</td><td>15.678</td></tr><tr><td> $\mathrm { G L A } { + } \beta$ </td><td>15.308</td><td>15.540</td><td>15.463</td><td>16.046</td><td>15.952</td></tr><tr><td> $_ \mathrm { D e l t a N e t }$ </td><td>15.678</td><td>15.925</td><td>15.916</td><td>20.323</td><td>20.261</td></tr><tr><td>GDN</td><td>15.004</td><td>15.168</td><td>15.145</td><td>17.064</td><td>17.059</td></tr><tr><td>KDA</td><td>14.489</td><td>14.846</td><td>14.787</td><td>15.837</td><td>15.766</td></tr><tr><td> $\mathrm { L A } ^ { \dagger }$ </td><td>17.230</td><td>24.916</td><td>20.597</td><td>31.582</td><td>25.976</td></tr><tr><td> $\mathrm { L A } { + } \beta ^ { \dagger }$ </td><td>16.980</td><td>30.719</td><td>29.583</td><td>100.247</td><td>92.325</td></tr><tr><td> $\mathbf { M a m b a - } 2 ^ { \dagger }$ </td><td>16.721</td><td>20.842</td><td>20.215</td><td>21.479</td><td>20.817</td></tr><tr><td> $\mathrm { M a m b a } { - 2 + \beta } ^ { \dagger }$ </td><td>16.189</td><td>17.665</td><td>17.380</td><td>18.844</td><td>18.460</td></tr><tr><td> $\mathrm { G L A ^ { \dagger } }$ </td><td>16.070</td><td>20.282</td><td>19.788</td><td>20.977</td><td>20.411</td></tr><tr><td> $\mathrm { G L A } { + } \beta ^ { \dagger }$ </td><td>15.891</td><td>19.222</td><td>18.182</td><td>20.523</td><td>19.410</td></tr></table>

Table 11: Zero-shot accuracy for the 340M zoo with INT8 operands and $\langle \mathrm { T } , \mathrm { C } \rangle$ scales. The six-task average includes LAMBADA, HellaSwag, ARC-e, ARC-c, PIQA, and WinoGrande, with normalized accuracy for HellaSwag and ARC-c. A dash denotes an unreported stack evaluation. †: no QK normalization.
<table><tr><td rowspan="2">Recurrence</td><td colspan="3">LAMBADA</td><td colspan="3">Six-task average</td></tr><tr><td>FP</td><td>Naive</td><td>Stack</td><td>FP</td><td>Naive</td><td>Stack</td></tr><tr><td>LA</td><td>0.267</td><td>0.042</td><td>0.258</td><td>0.398</td><td>0.356</td><td>0.395</td></tr><tr><td> $\mathrm { L A } { + } \beta$ </td><td>0.283</td><td>0.095</td><td>0.268</td><td>0.400</td><td>0.367</td><td>0.398</td></tr><tr><td>Mamba-2</td><td>0.324</td><td>0.310</td><td>0.316</td><td>0.410</td><td>0.409</td><td>0.409</td></tr><tr><td> $\mathbf { M a m b a } – 2 + \beta$ </td><td>0.316</td><td>0.307</td><td></td><td>0.413</td><td>0.412</td><td></td></tr><tr><td> $\mathrm { G L A }$ </td><td>0.335</td><td>0.331</td><td></td><td>0.419</td><td>0.417</td><td>一</td></tr><tr><td> $\mathrm { G L A } { + } \beta$ </td><td>0.332</td><td>0.328</td><td></td><td>0.416</td><td>0.415</td><td>一</td></tr><tr><td> $_ \mathrm { D e l t a N e t }$ </td><td>0.273</td><td>0.283</td><td></td><td>0.402</td><td>0.406</td><td>一</td></tr><tr><td>GDN</td><td>0.361</td><td>0.349</td><td></td><td>0.418</td><td>0.413</td><td>一</td></tr><tr><td> $\mathrm { K D A }$ </td><td>0.341</td><td>0.339</td><td></td><td>0.413</td><td>0.413</td><td></td></tr><tr><td> $\mathrm { L A } ^ { \dagger }$ </td><td>0.266</td><td>0.242</td><td>0.250</td><td>0.397</td><td>0.391</td><td>0.394</td></tr><tr><td> $\mathrm { L A } { + } \beta ^ { \dagger }$ </td><td>0.273</td><td>0.150</td><td>0.222</td><td>0.400</td><td>0.372</td><td>0.380</td></tr><tr><td> $\mathbf { M a m b a - } 2 ^ { \dagger }$ </td><td>0.290</td><td>0.283</td><td>0.286</td><td>0.402</td><td>0.401</td><td>0.401</td></tr><tr><td> $\mathrm { M a m b a } { - 2 + \beta } ^ { \dagger }$ </td><td>0.304</td><td>0.299</td><td>0.298</td><td>0.405</td><td>0.405</td><td>0.404</td></tr><tr><td> $\mathrm { G L A ^ { \dagger } }$ </td><td>0.302</td><td>0.301</td><td>0.302</td><td>0.406</td><td>0.406</td><td>0.405</td></tr><tr><td> $\mathrm { G L A } { + } \beta ^ { \dagger }$ </td><td>0.314</td><td>0.311</td><td>0.311</td><td>0.406</td><td>0.406</td><td>0.405</td></tr></table>

Table 12: Zero-shot accuracy for the 1.3B zoo with INT8 operands and $\langle \mathrm { T } , \mathrm { C } \rangle$ scales. The six-task average includes LAMBADA, HellaSwag, ARC-e, ARC-c, PIQA, and WinoGrande, with normalized accuracy for HellaSwag and ARC-c. A dash denotes an unreported stack evaluation.
<table><tr><td rowspan="2">Recurrence</td><td colspan="3">LAMBADA</td><td colspan="3">Six-task average</td></tr><tr><td>FP</td><td>Naive</td><td>Stack</td><td>FP</td><td>Naive</td><td>Stack</td></tr><tr><td> $\mathrm { L A } { + } \beta$ </td><td>0.465</td><td>0.169</td><td>0.408</td><td>0.507</td><td>0.447</td><td>0.493</td></tr><tr><td> $\mathbf { M a m b a } – 2 + \beta$ </td><td>0.480</td><td>0.469</td><td></td><td>0.522</td><td>0.521</td><td></td></tr><tr><td> $\mathrm { G L A } { + } \beta$ </td><td>0.531</td><td>0.518</td><td>一</td><td>0.544</td><td>0.545</td><td>一</td></tr><tr><td>DeltaNet</td><td>0.454</td><td>0.458</td><td>1</td><td>0.505</td><td>0.508</td><td></td></tr><tr><td>GDN</td><td>0.511</td><td>0.505</td><td>一</td><td>0.535</td><td>0.532</td><td>一</td></tr><tr><td>KDA</td><td>0.522</td><td>0.514</td><td>一</td><td>0.540</td><td>0.538</td><td>一</td></tr></table>

## F PRODUCTION RESULTS

This appendix reports the production results for Section 4.2. Setup and timing methodology are given in Appendix B.2. We compare with the official SSDi8 release, then examine query–key transformations, value rotation, and decay allocation. Compact quality and downstream summaries follow.

## F.1 ABLATIONS

Comparison with the official SSDi8 release. Our implementation closely matches the official SSDi8 release in perplexity and prefill latency across the three Mamba-2 sizes (Table 13). Both use value-side absorption, static per- $\bar { ( H , D _ { v } ) }$ scales, and no recurrent rotation.

Small numerical differences remain in the fixed-point decay gate and FP32 rounding of the fused quantizers. For Nemotron-H, where no official SSDi8 implementation is available, the matched baseline uses our implementation of this setting.

Table 13: Mamba-2 baseline validation on WikiText-2. Quantized rows use W4A8 projections, valueside $\mathsf { s t a t i c - } ( H , D _ { v } )$ absorption, no rotation, and no compensation. Full-model prefill is measured on the A5000 at $B = \stackrel { \cdot } { 8 } , T = 2 0 4 8$

<table><tr><td>Method</td><td>1.3B</td><td>2.7B</td><td>8B</td></tr><tr><td>WikiText-2 perplexity</td><td></td><td></td><td></td></tr><tr><td>FP16</td><td>10.42</td><td>9.06</td><td>7.25</td></tr><tr><td>SSDi8 W4A8 (official implementation)</td><td>10.923</td><td>9.430</td><td>7.629</td></tr><tr><td>Value-side baseline (our re-implementation)</td><td>10.928</td><td>9.441</td><td>7.620</td></tr><tr><td>Prefill latency (ms)</td><td></td><td></td><td></td></tr><tr><td>SSDi8 W4A8 (official implementation)</td><td>616</td><td>1184</td><td>2563</td></tr><tr><td>Value-side baseline (our re-implementation)</td><td>609</td><td>1163</td><td>2542</td></tr></table>

Query–key rotation and diagonal rescaling. Query–key rotation lowers perplexity at both decay endpoints and under both scale policies, with the largest gains at 8B (Table 16). The endpoints remain close in quality after rotation.

The exponent sweep favors $\begin{array} { r } { \kappa = \frac { 1 } { 2 } } \end{array}$ , with $\textstyle { \frac { 1 } { 4 } }$ close behind (Table 14). Incorporating rescaling into the existing kernel leaves measured layer latency unchanged, whereas a separate elementwise operation adds overhead.

Table 14: Production query–key rescaling controls. Top: WikiText-2 perplexity with static value-side decay absorption. Bottom: Mamba-2 1.3B layer latency at $B = 1$ , relative to the baseline without rescaling. A dash denotes an unreported absolute time.
<table><tr><td>Rescaling exponent κ</td><td>1.3B</td><td>8B</td></tr><tr><td>1/4</td><td>10.858</td><td>7.435</td></tr><tr><td>1/2</td><td>10.849</td><td>7.435</td></tr><tr><td>3/4</td><td>10.866</td><td>7.441</td></tr><tr><td>Implementation</td><td>Latency (ms)</td><td>Relative latency</td></tr><tr><td>Without rescaling</td><td>1.498</td><td>1.00</td></tr><tr><td>Fused rescaling</td><td>1.498</td><td>1.00</td></tr><tr><td>Separate elementwise rescaling</td><td></td><td>1.04</td></tr></table>

Value rotation. With full-precision projections and the production pipeline’s per-channel value scales, storing the state in a value-rotated basis increases perplexity from 10.496 to 10.533 (Table 15). Canonical storage gives 10.497, essentially matching the unrotated baseline.

Figure 14 shows that value-rotation error grows in later layers with small recurrence outputs. This suggests a role for the RMSNorm between the recurrence and output projection: normalizing a small output can amplify its quantization error before it enters the residual stream. Canonical storage avoids the measured perplexity penalty but still requires inverse transforms. Since it provides no quality gain under the production scale policy, LinQuant leaves the value axis unchanged.

Table 15: Value-rotation diagnostics on Mamba-2 1.3B: WikiText-2 perplexity with full-precision projections and an INT8 fixed-point state. Canonical storage is evaluated by fake quantization.
<table><tr><td>Transformation and state basis</td><td>FP projections</td></tr><tr><td>No rotation</td><td>10.496</td></tr><tr><td> $\mathbf { R } _ { V }$  , rotated storage</td><td>10.533</td></tr><tr><td> $\mathbf { R } _ { V } ,$  canonical storage</td><td>10.497</td></tr></table>

![](images/f25c80aa0e8ac4e50e8a509d3c13657f1d8606547c9a030f2183efc69740f48d.jpg)  
Figure 14: Per-layer relative error against the FP16 recurrence output, alongside its magnitude, for three rotation settings with rotated state storage. Value-rotation error grows in later layers with small recurrence outputs. Section F.1 discusses the interaction with RMSNorm.

Decay allocation. We compare decay placement along two axes: quantization quality and execution cost. Table 16 compares key- and value-side configurations under static and dynamic scales, with and without query–key rotation. Static value-side runs omit the output-projection mean correction used by the other configurations (Appendix B.2), so the static comparison does not isolate decay placement. Figure 15 isolates the affected kernel stage, and Figure 5c varies the operand dimensions.

Table 16: Mamba-2 W4A8 absorption endpoint and rotation ablation, WikiText-2 perplexity, mean ± std over four calibration sets, random Hadamard $\mathbf { R } _ { Q K }$ , without $\mathbf { D } _ { Q K }$ . Static value-side runs omit output-projection mean correction; the other configurations include it.
<table><tr><td rowspan="2"></td><td rowspan="2">Decay on Granularity</td><td colspan="2">1.3B</td><td colspan="2">2.7B</td><td colspan="2">8B</td></tr><tr><td>None</td><td> $\mathbf { R } _ { Q K }$ </td><td>None</td><td> $\mathbf { R } _ { Q K }$ </td><td>None</td><td> $\mathbf { R } _ { Q K }$ </td></tr><tr><td>Value</td><td> $\mathrm { s t a t i c } _ { - } ( H , D _ { v } )$ </td><td>10.928 ±0.004</td><td>10.860 ±0.007</td><td>9.441 ±0.017</td><td>9.379±0.011</td><td>7.620 ±0.009</td><td>7.456 ±0.009</td></tr><tr><td>Value</td><td>dynamic-(H, 1)</td><td>10.894 ±0.005</td><td>10.829±0.012</td><td>9.433 ±0.006</td><td>9.371 ±0.012</td><td> $7 . 6 4 2 \pm 0 . 0 1 3$ </td><td>7.447±0.008</td></tr><tr><td>Key</td><td>static-(H, Dk)</td><td> $1 0 . 9 2 5 \pm 0 . 0 0 3$ </td><td> $1 0 . 8 4 9 \pm 0 . 0 0 8$ </td><td> $9 . 4 2 4 \pm 0 . 0 1 6$ </td><td> $9 . 3 6 7 \pm 0 . 0 1 1$ </td><td> $7 . 6 0 1 \pm 0 . 0 0 9$ </td><td> $7 . 4 5 5 \pm 0 . 0 0 6$ </td></tr><tr><td>Key</td><td>dynamic-(H, i)</td><td> $1 0 . 8 8 4 \pm 0 . 0 0 8$ </td><td> $1 0 . 8 2 8 \pm 0 . 0 1 0$ </td><td>9.424±0.011</td><td> $9 . 3 6 0 \pm 0 . 0 0 4$ </td><td> $7 . 6 3 4 \pm 0 . 0 0 9$ </td><td> $7 . 4 4 4 \pm 0 . 0 0 9$ </td></tr></table>

The endpoints have similar perplexity under the tested scale policies (Table 16). Without rotation, key-side absorption is slightly better in each comparison. Static scales are per channel on the absorbed operand, while dynamic scales are per (chunk, H, 1) and computed inside the kernel. These results support either placement on quality grounds, consistent with the limited sensitivity to scalar-decay placement at INT8 in the zoo.

![](images/3dece0d2ef78b40aeb219e42b0a6229c7348fde085981aaa7466d8395718a9ed.jpg)  
Figure 16: Mamba-2 1.3B recurrence latency against batch size at $T = 2 0 4 8$ (left) and against sequence length at $B = 3 2$ (right) for ${ \mathrm { F P } } 1 6 ,$ the official SSDi8 implementation, and the value- and key-side configurations with static scales and $\mathbf { R } _ { Q K }$ . SSDi8 and the value-side configuration have similar latency once launch overhead no longer dominates.

![](images/be436bb2a64c68afe887299751195b3e2411300b6fb82b486e36e09c5e01f769.jpg)  
Figure 15: Per-stage Mamba-2 1.3B recurrence latency at $B = 8 , T = 2 0 4 8 \mathrm { : }$ : the SSDi8 setting (our matched value-side baseline) and key-/value-side configurations with $\mathbf { R } _ { Q K }$ and static or dynamic (<sup>′</sup>) scales.

Key-side absorption need not materialize a head-expanded tensor. Each thread block loads its group’s keys and applies head-specific decay in registers (Appendix D.2). The remaining difference is in the state write, which reloads the absorbed operand in floating point before weighting and quantizing it. Figure 15 shows that the other stages are unchanged.

Value-side absorption is faster for the tested Mamba-2 geometry, where $D _ { k } = 1 2 8$ and $D _ { v } = 6 4$ (Figure 15). Dynamic scales add cost on the key side because finding the maximum requires another pass over the weighted operand. Figure 16 extends the comparison across batch size and sequence length.

The geometry sweep in Figure 5c shows that wider values can reverse the preferred endpoint at $D _ { k } \ = \ 6 4$ , while the key-side penalty persists across the tested widths at $\dot { D } _ { k } \ = \ 1 2 8$ . Operand dimensions and the kernel’s accumulation and rescaling layout therefore both influence the cost of decay placement.

## F.2 QUALITY AND DOWNSTREAM SUMMARIES

Table 17 reports the static-scale configurations used in the main comparison, retaining calibration variability for Mamba-2 and both decay endpoints for the combined transformation. Dynamic-scale endpoint comparisons remain in Table 16. The complete configuration grid and per-task breakdowns are retained in the supplementary source.

Table 17: Production quality with static scales and W4A8 projections. Perplexity is mean ± standard deviation over four calibration sets for Mamba-2 (rows without a ± are single runs) and a single run for Nemotron-H. Avg. is the six-task average from the default calibration run. Value and Key denote the decay-absorption endpoint, with per-channel scales on that operand. The official SSDi8 implementation is available only for Mamba-2.
<table><tr><td>Configuration</td><td>WikiText-2 PPL</td><td> $\operatorname { A v g } .$ </td></tr><tr><td>Mamba-2 1.3B</td><td></td><td></td></tr><tr><td>FP16</td><td>10.420</td><td>0.595</td></tr><tr><td>SSDi8 (official)</td><td> $1 0 . 9 2 3 \pm 0 . 0 0 7$ </td><td>0.588</td></tr><tr><td>Value:  $R _ { Q K }$ </td><td> $1 0 . 8 6 0 \pm 0 . 0 0 7$ </td><td>0.590</td></tr><tr><td>Value:  $R _ { Q K } + D _ { Q K }$ </td><td> $1 0 . 8 4 9 \pm 0 . 0 1 1$ </td><td>0.588</td></tr><tr><td>Key:  $R _ { Q K } + D _ { Q K }$ </td><td> $1 0 . 8 3 0 \pm 0 . 0 0 7$ </td><td>0.589</td></tr><tr><td>Mamba-2 2.7B</td><td></td><td></td></tr><tr><td>FP16</td><td>9.060</td><td>0.638</td></tr><tr><td>SSDi8 (official)</td><td>9.430</td><td>0.626</td></tr><tr><td>Value:  $R _ { Q K }$ </td><td> $9 . 3 7 9 \pm 0 . 0 1 1$ </td><td>0.629</td></tr><tr><td>Value:  $R _ { Q K } + D _ { Q K }$ </td><td> $9 . 3 8 3 \pm 0 . 0 0 7$ </td><td>0.627</td></tr><tr><td>Key:  $R _ { Q K } + D _ { Q K }$ </td><td> $9 . 3 7 2 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td>0.630</td></tr><tr><td>Mamba-2 8B</td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td>FP16</td><td>7.250</td><td>0.707</td></tr><tr><td>SSDi8 (official)</td><td>7.629</td><td>0.696</td></tr><tr><td>Value:  $R _ { Q K }$ </td><td>7.456 ±0.009</td><td>0.703</td></tr><tr><td>Value:  $R _ { Q K } + D _ { Q K }$ </td><td> $7 . 4 3 5 \pm 0 . 0 0 8$ </td><td>0.699</td></tr><tr><td>Key:  $R _ { Q K } + D _ { Q K }$ </td><td> $7 . 4 3 7 \pm 0 . 0 0 4$ </td><td>0.697</td></tr><tr><td>Nemotron-H Base</td><td></td><td></td></tr><tr><td>FP16</td><td>6.030</td><td>0.758</td></tr><tr><td>Value:  $R _ { Q K }$ </td><td>6.186</td><td>0.751</td></tr><tr><td>Value:  $R _ { Q K } + D _ { Q K }$ </td><td>6.178</td><td>0.753</td></tr><tr><td>Key:  $R _ { Q K } + D _ { Q K }$ </td><td>6.175</td><td>0.753</td></tr><tr><td>Nemotron-H Reasoning</td><td></td><td></td></tr><tr><td>FP16</td><td>8.423</td><td>0.731</td></tr><tr><td>Value:  $R Q { \cal K }$ </td><td>8.666</td><td>0.727</td></tr><tr><td>Value:  $R _ { Q K } + D _ { Q K }$ </td><td>8.650</td><td>0.728</td></tr><tr><td>Key:  $R _ { Q K } + D _ { Q K }$ </td><td>8.670</td><td>0.728</td></tr></table>

Table 18 summarizes LAMBADA and the six-task average for the rescaled configurations. The two decay endpoints have similar downstream accuracy. Perplexity improvements do not imply uniform gains in task accuracy, as the comparison with the unrescaled configurations in Table 17 shows.

Table 18: Downstream accuracy for the static production configurations with $\mathbf { R } _ { Q K } + \mathbf { D } _ { Q K }$ . Both endpoints use per-channel scales on the absorbed operand. Mamba-2 uses the default calibration set. The six-task average follows Table 17.
<table><tr><td rowspan="2">Model</td><td colspan="2">LAMBADA</td><td colspan="2">Six-task average</td></tr><tr><td>Value</td><td>Key</td><td>Value</td><td>Key</td></tr><tr><td>Mamba-2 1.3B</td><td>0.644</td><td>0.645</td><td>0.588</td><td>0.589</td></tr><tr><td>Mamba-2 2.7B</td><td>0.675</td><td>0.684</td><td>0.627</td><td>0.630</td></tr><tr><td>Mamba-2 8B</td><td>0.706</td><td>0.704</td><td>0.699</td><td>0.697</td></tr><tr><td>Nemotron-H Base</td><td>0.729</td><td>0.727</td><td>0.753</td><td>0.753</td></tr><tr><td>Nemotron-H Reasoning</td><td>0.660</td><td>0.660</td><td>0.728</td><td>0.728</td></tr></table>

Hadamard signs. Table 19 isolates deterministic versus random Hadamard signs under static value-side absorption. Random signs improve five of the six query–key rotation comparisons shown; the exception is the small WikiText-2 difference at 8B. The main experiments use random signs. Additional endpoint and scale combinations are retained in the full source grids.

Table 19: Hadamard-sign control with W4A8 projections, static value-side decay absorption, and no diagonal rescaling. Entries are single-calibration-run perplexities, distinct from the multi-calibration means above. Det. and Rand. apply deterministic and random-signed query–key Hadamard rotations.
<table><tr><td>Model</td><td>Dataset</td><td>No rotation</td><td>Det.</td><td>Rand.</td></tr><tr><td>Mamba-2 1.3B</td><td>WikiText-2</td><td>10.930</td><td>10.882</td><td>10.857</td></tr><tr><td>Mamba-2 1.3B</td><td>Pile</td><td>7.415</td><td>7.397</td><td>7.382</td></tr><tr><td>Mamba-2 2.7B</td><td>WikiText-2</td><td>9.420</td><td>9.400</td><td>9.374</td></tr><tr><td>Mamba-2 2.7B</td><td>Pile</td><td>6.572</td><td>6.557</td><td>6.555</td></tr><tr><td>Mamba-2 8B</td><td>WikiText-2</td><td>7.619</td><td>7.463</td><td>7.467</td></tr><tr><td>Mamba-2 8B</td><td>Pile</td><td>6.293</td><td>6.213</td><td>6.206</td></tr></table>