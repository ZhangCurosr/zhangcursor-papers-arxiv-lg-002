# Low-Bit GDN Recurrent-State Quantization: Global Gradient-aware Allocation and Pure-Write Sliding Windows

Qinghang Huang Xuchu Huang Xunzhao Yin

Zhejiang University

22560023@zju.edu.cn xchuang\_@zju.edu.cn xzyin1@zju.edu.cn

<sup>\*</sup>Corresponding author.

October 18, 2026

## Abstract

Linear attention stores historical information in a recurrent state of fixed size, providing a compact memory representation for long-context inference. Reducing state precision further lowers per-request storage requirements, but quantized writeback errors enter subsequent recurrences, degrade prediction quality on long-reasoning tasks, and can be accompanied by repetitive generation and excessively long outputs. We propose Global Gradient-aware Allocation (GGA) and the Pure-Write Sliding Window (PW) to address precision allocation and recent-information maintenance. GGA combines candidate quantization residuals with gradient sensitivity to future prediction loss to allocate head bit widths under a model-wide mixed-precision budget. PW independently maintains recent pure writes as FP8 vector pairs, propagates them through the full GDN state transition, and merges them into the low-bit historical state only after they leave the window. Across seven evaluations on Qwen3.5-4B/9B, GGA allocation alone improves over STEPQuant@4 by 19.48/29.84 percentage points. Adding PW-S8 yields seven-task macro-averages of 69.83%/78.63%, improving over STEPQuant@4 by 38.39/41.21 percentage points, respectively. Both precision reallocation and window maintenance improve task scores while mitigating excessively long outputs: at @6, GGA reduces mean generation length, averaged equally across tasks, by 10.45%/14.65% relative to STEPQuant; adding PW-S8 to GGA@4 further reduces mean generation length by 32.32%/36.77%. Comparisons at similar storage budgets further show that using additional capacity to maintain recent writes yields higher task scores and shorter mean outputs than solely increasing the bit width of the accumulated state.

## 1 Introduction

Autoregressive inference requires the continuous retention of historical information. In standard attention, the KV cache grows with sequence length, whereas linear attention compresses historical interactions into a recurrent state of fixed size [1]. Architectures such as GLA, Gated DeltaNet (GDN), and Mamba-2 further improve the expressiveness and computational efficiency of recurrent memory [2, 3, 4]. Nevertheless, even a fixed-size state must be stored separately for each concurrent request and read and updated at every decoding step; its storage and memory-access costs therefore motivate compression in inference deployment [5, 6, 7].

Recurrent-state quantization faces both precision-allocation and error-feedback challenges. Perturbations to different heads have different effects on subsequent predictions, while writeback errors introduced by stepwise quantization enter future recurrences. DAMP protects high-risk channels using decay-weighted risk, STEPQuant jointly models error persistence and readout impact, and LeapQuant delays state quantization through window buffering [5, 6, 7]. These studies provide an effective foundation for state compression and motivate two further questions: how final prediction loss can guide head precision allocation, and how recent information can be maintained according to write age.

For precision allocation, local candidate distortion and error lifetime alone do not determine changes in final predictions. State residuals of equal magnitude can have different effects because of their directions, subsequent query readouts, and downstream network mappings. We retain STEPQuant’s candidate quantization formats and nominal budget, and weight candidate residuals by gradients of future language-model loss with respect to the writeback state to construct an additive risk proxy for Global Gradient-aware Allocation (GGA). For temporal maintenance, we separate pure writes in the GDN update into individual floating-point vector branches, apply the full state transition to them, and evict them individually into the low-bit historical state at a fixed age, yielding the Pure-Write Sliding Window (PW). This rolling maintenance differs from quantizing the entire state at the end of a window: the old state is still quantized at every step.

Our contributions are as follows:

• We propose Global Gradient-aware Allocation (GGA) guided by future prediction loss. Along calibration trajectories, we estimate the sensitivity of future loss to the state at each writeback position and construct comparable head risks by weighting squared candidate quantization residuals coordinatewise with squared gradients. Retaining STEPQuant’s candidate quantization formats and nominal budget, we combine FP16 pivot selection with discrete bit-width optimization to reallocate precision across layers and heads; gradient computation is used only for offline calibration.

• We propose the Pure-Write Sliding Window (PW). We separate recent pure writes from the GDN recurrence, represent each branch as a vector pair, apply the full state transition at subsequent steps, and merge branches into the low-bit historical state on a rolling basis according to write age. This design retains stepwise quantization of the old state while delaying the first incorporation of recent branches into the low-bit matrix. We provide a proof of recurrence equivalence under ideal conditions and an online FP8 maintenance algorithm with per-vector scales.

## 2 Background and Related Work

## 2.1 Linear Recurrent Models and State Compression

Linear attention accumulates key/value interactions in a fixed-size state, enabling autoregressive inference through recurrent updates [1]. GLA adds data-dependent forgetting gates [2]; Gated DeltaNet (GDN) combines gated forgetting with the delta rule to modify memory along the current key direction [3]; Mamba-2 connects selective state-space models and attention through structured state-space duality [4]. Although these architectures avoid a state that grows with sequence length, each concurrent request retains its own state, which is read and updated at every decoding step. Compressing that state therefore remains relevant to inference storage.

State compression must account for how a memory representation is used. KIVI quantizes attention keys per channel and values per token, while retaining a floating-point buffer of recent KV entries [8]. For selective state-space models, Quamba2 groups recurrent-operator inputs according to persistent channel magnitudes and applies the grouping to an 8-bit state cache [9]; Q-Mamba uses decoupled scaling along the state and channel axes and simulates quantized recurrences during blockwise reconstruction [10]. In contrast to KV entries appended to a cache, recurrent states repeatedly become inputs to later updates, allowing writeback errors to propagate through subsequent computation.

## 2.2 GDN Recurrence and Quantized Writeback

Consider one state/value head in GDN, omitting layer and head indices. At update $t , S _ { t - 1 } \in \mathbb { R } ^ { d _ { k } \times d _ { \tau } }$ is the incoming historical state and $S _ { t }$ is the updated state; $S _ { 0 }$ is the state at the end of prefill. The query and key are $q _ { t } , k _ { t } \in \mathbb { R } ^ { d _ { k } }$ , and the value is $v _ { t } \in \mathbb { R } ^ { d _ { v } }$ . These are the processed activations entering the recurrent operator, including the model’s normalization, scaling, and mapping from shared queries/keys to state heads. With retention coefficient $\gamma _ { t }$ and key-update strength $\beta _ { t }$ , GDN updates the state and then reads the current head output:

$$
\begin{array} { r l } & { \boldsymbol { S } _ { t } = \gamma _ { t } \boldsymbol { S } _ { t - 1 } + \beta _ { t } k _ { t } \big ( \boldsymbol { v } _ { t } - \gamma _ { t } \boldsymbol { S } _ { t - 1 } ^ { \top } k _ { t } \big ) ^ { \top } , } \\ & { \boldsymbol { o } _ { t } = \boldsymbol { S } _ { t } ^ { \top } \boldsymbol { q } _ { t } . } \end{array}\tag{1}
$$

The raw readout $o _ { t }$ is followed by normalization, output gating, and projection in the downstream network. We separate the update into a state transition and a pure write:

$$
\begin{array} { r l } & { S _ { t } = A _ { t } S _ { t - 1 } + B _ { t } , } \\ & { A _ { t } = \gamma _ { t } ( I - \beta _ { t } k _ { t } k _ { t } ^ { \top } ) , \qquad B _ { t } = \beta _ { t } k _ { t } v _ { t } ^ { \top } . } \end{array}\tag{2}
$$

$A _ { t }$ applies forgetting and key-selective correction to the historical state; $B _ { t }$ has rank at most one. Given the current activations, $B _ { t }$ contains no explicit dependence on this head’s historical state, although those activations depend on the model and generated text during free generation.

Quantized writeback. Let $\mathcal { Q } _ { b }$ denote quantization followed by reconstruction at bit width b, and let $\hat { S } _ { t - 1 }$ be the matrix decoded from the previous quantized state. Naive stepwise full-state quantization updates this matrix to the candidate state $S _ { t } ^ { \mathrm { c a n d } }$ , reads the current output, and then quantizes the state for the next step:

$$
S _ { t } ^ { \mathrm { c a n d } } = A _ { t } \hat { S } _ { t - 1 } + B _ { t } ,\tag{3}
$$

$$
o _ { t } = \left( S _ { t } ^ { \mathrm { c a n d } } \right) ^ { \top } q _ { t } , \qquad \hat { S } _ { t } = \mathcal { Q } _ { b } ( S _ { t } ^ { \mathrm { c a n d } } ) .\tag{4}
$$

$S _ { t } ^ { \mathrm { c a n d } }$ already contains historical quantization errors. The new writeback error is introduced after the current readout and affects subsequent states and predictions.

## 2.3 Recurrent-State Quantization

Precision allocation. DAMP combines local quantization error and future decay to rank key channels, retaining a fixed number of high-risk channels per head in FP16 and quantizing the rest to INT8 [5]. Channels with similar error magnitudes can receive different protection because they are forgotten at different rates. For delta-rule states, STEPQuant uses heads as allocation units and combines error-persistence weights with candidate reconstruction distortion under a model-wide bit budget [6]. Its distortion includes key-row readout weights estimated from propagated queries, while dual-axis scale fitting adapts to both axes of the state matrix. These approaches capture the magnitude, persistence, and local readout effects of state errors. Loss-based sensitivity has also been used for weight quantization: SqueezeLLM weights reconstruction errors with a diagonal Fisher approximation from gradient statistics and performs sensitivity-weighted nonuniform clustering [11]. Its targets are model weights that remain fixed after offline processing. Recurrent states instead change with each input, and each writeback perturbation undergoes further state transitions and readouts. GGA uses gradients of future language-model loss to weight candidate state residuals along this recurrence, retaining STEPQuant’s candidate quantizers and nominal budget.

Delayed quantization. LeapQuant maintains a low-bit boundary state and a high-precision update buffer within each block window, recovers intermediate readouts using state-dependent netcorrection terms, and reconstructs and requantizes the state at the boundary [7]. Its Compensator Tokens retain dominant state structure as high-precision low-rank components across windows while quantizing the residual. Unlike buffering independent KV entries, a recurrent update buffer must account for transformations of earlier write contributions. PW separates the pure writes $B _ { t }$ in Equation (2). Each recent write is stored as an FP8 vector pair whose left factor undergoes the full state transition; a rolling queue merges branches individually into the low-bit historical state according to write age. The old state remains updated and quantized at every step.

## 3 Motivation

## 3.1 Matching Head Precision to Prediction Sensitivity

Assigning different precisions to recurrent-state heads should direct limited storage toward heads whose perturbations have a greater effect on subsequent predictions. We examine this correspondence on fixed diagnostic samples from Qwen3.5-4B. For each head, we introduce one quantization perturbation to its state, hold subsequent token inputs fixed, and measure the KL divergence be tween subsequent output distributions and the unquantized reference. Averaging this effect over the candidate precisions $\{ 2 , 4 , 6 , 8 \}$ gives an empirical sensitivity $I _ { u } .$ . Figure 1 groups heads into equalwidth bins of $I _ { u }$ and compares the mean bit width assigned to each bin; the horizontal axis runs from lower to higher empirical impact. Measurement details and complete binned data are provided in Appendix C.

![](images/4c2ab3d9c0b8f4ed115f98cee396e93efdcb6d3e738f920fb144d2a789b3d375.jpg)  
Figure 1: Head precision versus empirical quantization impact. The horizontal axis shows centers of $0 . 3 0 \times$ $1 0 ^ { - 4 }$ -wide bins of $I _ { u } ,$ with head counts n. Both allocations average 4.33 bit/value over all heads; FP16 counts as 16 bits. The empty bin $[ 3 . 0 0 , 3 . 3 0 ) \times 1 0 ^ { - 4 }$ and the vertical range 8–14 bits are omitted.

The STEPQuant allocation does not increase consistently with this measured sensitivity. It assigns 5.28 bits on average in the $[ 1 . 5 0 , 1 . 8 0 ) \times 1 0 ^ { - 4 }$ bin, but only 3.73 bits in the higher-impact $[ 2 . 4 0 , 2 . 7 0 ) \times 1 0 ^ { - 4 }$ bin. STEPQuant combines error persistence, candidate reconstruction distortion, and key-row readout weights [6]; these describe local properties of state perturbations without directly measuring the response of the final language-model loss. This observation motivates GGA: we introduce downstream loss gradients, propagate writeback sensitivity along reference state transitions, and use squared gradients to weight squared candidate quantization residuals before recomputing the model-wide allocation.

## 3.2 Preserving Recent Conclusions during Reasoning

The left side of Figure 2 illustrates a failure to carry recently established conclusions forward under GGA@4. The problem asks for the sum of all positive integers with at least six positive divisors whose sixth-largest positive divisor is 6; the reference answer is 48. The model correctly lists all ten divisors of 48 and confirms their count on line 947, yet rejects the same complete list on line 948 and resumes searching for divisors that are not missing. The count just verified is no longer used consistently as the reasoning proceeds. Under stepwise full-state quantization, the state contribution supporting such a conclusion is incorporated into low-bit writeback immediately after the current readout, then repeatedly quantized with the accumulated history. Perturbing that recent contribution may affect the model’s subsequent use of the conclusion.

The same trajectory acknowledges 2 as a divisor on line 1933 but again treats it as missing on line 1940. Rechecking eventually gives way to 831 consecutive lines of “Wait.” Generation is stopped by the repetition detector after 81,921 tokens, without a final answer. These observations motivate preserving recent writes separately from the repeatedly quantized historical state. Writeback perturbations enter later readouts through the recurrence, and the text they influence becomes subsequent model input; this feedback offers a possible explanation for how stalled reasoning can persist. We therefore design PW to maintain recent pure writes separately and delay their incorporation into the low-bit accumulated state. Its representation and rolling update are developed in Section 4.3, followed by the corresponding reasoning on the right of Figure 2.

<table><tr><td colspan="4">Compute the sum of all positive integers n such that n has at least 6 positive integer divisors and the 6th largest divisor of n is 6. Reference answer: 48.</td></tr><tr><td colspan="3">GGA@4 (without PW)</td><td colspan="2">GGA@4 + PW-S8</td></tr><tr><td colspan="3"></td><td></td><td></td></tr><tr><td rowspan="3"></td><td>934Count: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.</td><td></td><td>758Divisors of 48: 1, 2, 3, 4, 6, 8, 12, 16, 24, 48.</td><td rowspan="3"></td></tr><tr><td>935 Okay, I miscounted when I said &quot;that is 9 num-</td><td></td><td>771 Total k = 10 divisors.</td></tr><tr><td>bers&quot;.</td><td></td><td></td></tr><tr><td></td><td>947Okay, there are 10 divisors.</td><td></td><td>779 So for n = 48, the 6th largest divisor is indeed 6. 780 So n = 48 is a solution.</td></tr><tr><td></td><td>948 The list is 1, 2, 3, 4, 6, 8, 12, 16, 24, 48 is wrong. 949 Missing divisor:</td><td></td><td></td></tr><tr><td></td><td>962 Wait, where is the 10th one?</td><td></td><td>912 Plus 6 itself. Total 10. Correct. 913 So condition is satisfied.</td></tr><tr><td></td><td></td><td></td><td>915 Now, let&#x27;s explore general n.</td></tr><tr><td></td><td>1933 Is there a factor 2? Yes.</td><td></td><td>2641 And we concluded n = 48 is the only solution.</td></tr><tr><td></td><td>1940 Wait, did I forget 21 . 30 = 2? Yes.</td><td></td><td>2643 Therefore the sum is 48.</td></tr><tr><td>5899 5900 Wait.</td><td>Wait.</td><td>2644 &lt;/think&gt;</td><td></td></tr><tr><td>6729</td><td></td><td></td><td></td></tr><tr><td>Wait.</td><td></td><td>2677 48</td><td></td></tr></table>

Figure 2: Paired reasoning excerpts from the same problem. Numbers identify original output lines; ellipses mark omitted intervening lines. Without PW, the model rejects a verified divisor count, rechecks a known factor, and eventually repeats “Wait.” With PW, it retains the count, continues the search, and returns 48.

## 4 Method

## 4.1 Overview

We address recurrent-state quantization through precision allocation across heads and maintenance of recent writes over time. GGA constructs a loss-sensitive cost for each head and candidate bit width, then optimizes a model-wide mixed-precision allocation. PW separates recent pure writes from the accumulated history and delays their incorporation into the low-bit old state using a fixed-length queue.

GGA determines head bit widths and FP16 pivots offline; PW maintains recent branches online as FP8 vector pairs and applies the full GDN state transition to them. Their combination leaves model weights unchanged and requires no backpropagation during deployment. We present GGA in Section 4.2 and PW in Section 4.3.

## 4.2 Global Gradient-aware Precision Allocation

## 4.2.1 Sensitivity of Future Loss to the Writeback State

To address the precision-allocation issue in Section 3.1, GGA uses gradients of the language-model loss to construct candidate quantization risks and allocates bit widths across the entire model. The candidate quantizers, bit-width menu, and nominal budget follow STEPQuant.

Let u ∈ U denote a layer/head allocation unit. On an unquantized calibration trajectory, we use the next-token language-model loss

$$
\mathcal { L } = \sum _ { t = 1 } ^ { T - 1 } - \log p _ { \theta } ( x _ { t + 1 } \mid x _ { \le t } ) .\tag{5}
$$

Let $g _ { u , t } = \partial \mathcal { L } / \partial o _ { u , t } \in \mathbb { R } ^ { d _ { v } }$ denote the downstream gradient of the loss with respect to the raw head readout. Since $o _ { u , t } = S _ { u , t } ^ { \top } q _ { u , t }$ , the current readout branch contributes $q _ { u , t } g _ { u , t } ^ { \top }$ to the state gradient. To match the writeback position in Equation (4), we distinguish the current readout contribution from the future recurrence contribution. Let $G _ { u , t } ^ { \mathrm { i v b } }$ denote the writeback-sensitivity adjoint and $J _ { u , t }$ the state adjoint including the current readout. We propagate them backward along the recorded reference state transitions:

$$
\begin{array} { r } { J _ { u , t } = q _ { u , t } g _ { u , t } ^ { \top } + G _ { u , t } ^ { \mathrm { w b } } , } \\ { G _ { u , t - 1 } ^ { \mathrm { w b } } = A _ { u , t } ^ { \top } J _ { u , t } . } \end{array}\tag{6}
$$

This backward recurrence is performed independently for each calibration segment. At the end of a segment, if the writeback state is no longer used by subsequent computation, then $G _ { u , T } ^ { \mathrm { w b } } = 0$ . This boundary condition is consistent with the actual range of supervised tokens.

Equation (6) describes adjoint propagation on the explicit state-recurrence graph, given the reference activations of this head; the downstream gradient $g _ { u , i }$ <sub>t</sub> supplies sensitivity information from the language-model loss. We use $G _ { u , t } ^ { \mathrm { w b } }$ as a sensitivity estimate for candidate quantization risk. Adjoint propagation includes the forgetting term and key-selective correction in $A _ { u , t }$ . Because the current writeback occurs after the current readout, we use $G _ { u , t } ^ { \mathrm { w b } }$ to construct the cost, without attributing the already completed readout contribution to the newly introduced error.

## 4.2.2 Squared-Gradient-Weighted Candidate Cost

For a high-precision snapshot $S _ { u , t } ^ { \mathrm { F P } }$ of head $u ,$ the candidate quantization residual is

$$
\Delta S _ { u , t } ^ { ( b ) } = \mathcal { Q } _ { b } ^ { \mathrm { S T E P } } ( S _ { u , t } ^ { \mathrm { F P } } ) - S _ { u , t } ^ { \mathrm { F P } } .\tag{7}
$$

Let $\mathcal { T } _ { u }$ be the set of valid snapshots across calibration segments, jointly indexed by segment and time, with the segment index omitted below; $N _ { u } = | T _ { u } | > \mathrm { ~ \bar { 0 } ~ }$ is the total number of snapshots. GGA defines the nonnegative cost

$$
C _ { u , b } ^ { \mathrm { G G A } } = \frac { 1 } { 2 N _ { u } } \sum _ { t \in \mathcal { T } _ { u } } \left\| \boldsymbol { G } _ { u , t } ^ { \mathrm { w b } } \odot \Delta S _ { u , t } ^ { ( b ) } \right\| _ { F } ^ { 2 } .\tag{8}
$$

Here, $\odot$ denotes elementwise multiplication, and $\| \cdot \| _ { F }$ is the Frobenius norm. This cost assigns greater weight to quantization residuals at loss-sensitive coordinates and averages over the calibration snapshots. By comparison, the STEP allocation cost combines a lifetime weight with candidate reconstruction distortion:

$$
C _ { u , b } ^ { \mathrm { S T E P } } = L _ { u } d _ { u } ( b ) .\tag{9}
$$

$d _ { u } ( b )$ follows the original STEP definitions of candidate quantization and distortion. GGA does not additionally multiply Equation (8) by $L _ { u }$ , nor does it normalize costs separately for different layers or heads, preserving their global relative scale.

Equation (8) uses diagonal sensitivity weighting on the reference trajectory to aggregate the singleperturbation quantization residuals at individual coordinates into an approximate head-level risk. The sum of these head costs then forms the global allocation objective.

## 4.2.3 Global Allocation under a Nominal Budget

Let B be the set of candidate integer bit widths and $\bar { b }$ the nominal average bit-width budget. GGA follows STEPQuant’s allocation framework, retaining a small number of heads in FP16, called pivots, and selecting an integer bit width from B for each remaining head. For state heads of identical shape, we first solve a preliminary allocation over all heads without pivots, under the budget $\bar { b } \vert \mathcal { U } \vert$ obtaining bit widths $b _ { u } ^ { ( 0 ) }$ and their corresponding residual risks $r _ { u }$ :

$$
\begin{array} { r l } & { \{ b _ { u } ^ { ( 0 ) } \} \in \arg \operatorname* { m i n } _ { \{ b _ { u } \} } \sum _ { u \in \mathcal { U } } C _ { u , b _ { u } } ^ { \mathrm { G G A } } , \qquad r _ { u } = C _ { u , b _ { u } ^ { ( 0 ) } } ^ { \mathrm { G G A } } , } \\ & { \mathrm { s . t . } \quad b _ { u } \in \mathcal { B } , \quad \displaystyle \sum _ { u \in \mathcal { U } } b _ { u } = \bar { b } | \mathcal { U } | . } \end{array}\tag{10}
$$

We select a prescribed number $P$ of heads as pivots in descending order of their residual risks $r _ { u }$ after the preliminary allocation, and denote their set by $\mathcal { P }$ . After fixing this set, let $B _ { \mathrm { i n t } }$ be the budget for

the sum of integer bit widths available to the remaining non-pivot heads. We reallocate these heads as follows:

$$
\begin{array} { l } { \displaystyle \operatorname* { m i n } _ { \left\{ b _ { u } \right\} } \quad \displaystyle \sum _ { u \notin \mathcal { P } } C _ { u , b _ { u } } ^ { \mathrm { G G A } } } \\ { \mathrm { s . t . } \quad b _ { u } \in \mathcal { B } , \quad \displaystyle \sum _ { u \notin \mathcal { P } } b _ { u } = B _ { \mathrm { i n t } } . } \end{array}\tag{11}
$$

Both stages are solved by global dynamic programming; the recurrence and backtracking procedure are given in Appendix B.2. Bit widths are allocated across the entire model, with each layer’s share determined jointly by the candidate costs and the total budget.

STEP and GGA use the same nominal constraint, while the specific pivot identities and integer bitwidth allocations may differ. Heterogeneous state shapes require corresponding budgets based on the number of elements. Gradients and candidate costs are used only for offline calibration; deployment directly uses the frozen pivot set and bit-width table, without backpropagation. Candidate bit widths and pivot counts are given in Section 5.1, and calibration and frozen configurations in Appendix B.

Resulting precision distribution. Figure 1 shows how the GGA allocation responds to the mismatch identified in Section 3.1. At the same all-head average of 4.33 bit/value, GGA’s mean bit width rises overall with empirical quantization impact. It assigns 4.56 versus STEPQuant’s 3.73 bits in the $[ 2 . 4 0 , 2 . 7 0 ) \times 1 0 ^ { - 4 } \mathrm { \dot { \ b i n } }$ , and 6.96 versus 5.02 bits in the $[ 2 . 7 0 , 3 . 0 0 ) \times 1 0 ^ { - 4 }$ bin. Lowerimpact bins receive fewer bits on average, directing more precision toward head groups with greater measured effects on subsequent predictions.

## 4.3 Pure-Write Sliding Window

## 4.3.1 Exact Expansion and Vector Representation of Recent Writes

To address the maintenance of recent information discussed in Section 3.2, PW separates the purewrite branches in Equation (2), maintains the most recent m writes as FP8 vector pairs, and propagates them through the full state transition. Earlier history is carried by the stepwise-quantized old state.

Figure 3 illustrates the state decomposition and the one-step shift of the recent-write window.

![](images/badead2556942b2aae2999c3c657aafbb4394d922f132b9edd5a726ad97420b9.jpg)  
Propagate $p \gets A _ { t + 1 } p ;$ keep v fixed. Read out before eviction and enqueue.  
Figure 3: PW state decomposition and rolling update, with $m = 3$ shown for illustration. Each FP8 vector pair contributes a rank-one write, maintained separately from the low-bit history. Existing left factors and the history propagate through $A _ { t + 1 } .$ After readout, the oldest branch is merged into history before requantization and the new write enters the window; retained right factors remain fixed.

Let $r = \operatorname* { m i n } ( m , t )$ and unroll Equation (2) over the most recent r steps:

$$
S _ { t } = ( A _ { t } \cdot \cdot \cdot A _ { t - r + 1 } ) S _ { t - r } + \sum _ { j = t - r + 1 } ^ { t } ( A _ { t } \cdot \cdot \cdot A _ { j + 1 } ) B _ { j } .\tag{12}
$$

An empty product is defined as the identity matrix. The first term is the earlier state propagated to the current time, while the second term is the propagated contribution of recent pure writes.

Since $B _ { j } = ( \beta _ { j } k _ { j } ) v _ { j } ^ { \top }$ , we define a left factor for each branch:

$$
p _ { j , j } = \beta _ { j } k _ { j } , \qquad p _ { j , t } = A _ { t } p _ { j , t - 1 } .\tag{13}
$$

Then

$$
\begin{array} { r l r } { ( A _ { t } \cdot \cdot \cdot A _ { j + 1 } ) B _ { j } = p _ { j , t } v _ { j } ^ { \top } , } & { { } } & { p _ { j , t } = \gamma _ { t } \big [ p _ { j , t - 1 } - \beta _ { t } k _ { t } ( k _ { t } ^ { \top } p _ { j , t - 1 } ) \big ] . } \end{array}\tag{14}
$$

Left multiplication by $A _ { t }$ preserves the rank-one structure, so each recent branch only needs to store $( p _ { j , t } , v _ { j } )$ rather than a full state matrix. $v _ { j }$ is the value stored at the birth of the branch and is not updated by subsequent transitions; their effects accumulate in $p _ { j , t }$

## 4.3.2 Online State Maintenance and Readout Timing

Let $\hat { O } _ { t }$ be the reconstructed matrix of the low-bit old state. After the update, the queue index set is

$$
\mathcal { I } _ { t } = \{ \operatorname* { m a x } ( 1 , t - m + 1 ) , \ldots , t \} , \qquad \hat { S } _ { t } = \hat { O } _ { t } + \sum _ { j \in \mathcal { I } _ { t } } p _ { j , t } v _ { j } ^ { \top } .\tag{15}
$$

The sum here defines the logical state; the implementation does not persistently materialize the entire recent matrix. The queue is initially empty at the decoding boundary, $\mathrm { i . e . , } \mathcal { I } _ { 0 } = \emptyset$ , and the old state is initialized from the persistent-state representation produced by the original prefill path.

At each decoding update, we first propagate the old state and all existing recent left factors:

$$
O _ { t } ^ { \mathrm { p r e } } = A _ { t } \hat { O } _ { t - 1 } , \qquad p _ { j , t } = A _ { t } p _ { j , t - 1 } , \quad j \in \mathcal { J } _ { t - 1 } .\tag{16}
$$

The current new write is $p _ { t , t } = \beta _ { t } k _ { t }$ . The current readout uses the old state, all propagated recent branches, and the current new write:

$$
o _ { t } = ( O _ { t } ^ { \mathrm { p r e } } ) ^ { \top } q _ { t } + \sum _ { j \in \mathcal { I } _ { t - 1 } } v _ { j } ( p _ { j , t } ^ { \top } q _ { t } ) + v _ { t } \big ( ( \beta _ { t } k _ { t } ) ^ { \top } q _ { t } \big ) .\tag{17}
$$

If the queue is full upon entering this step, the oldest branch $j = t - m$ is evicted after the current readout and merged into the old state. Define

$$
E _ { t } ^ { \mathrm { e v i c t } } = { \left\{ \begin{array} { l l } { p _ { t - m , t } v _ { t - m } ^ { \top } , } & { t > m , } \\ { 0 , } & { t \leq m . } \end{array} \right. }\tag{18}
$$

The merged old state is then quantized, and the current new write is added to the queue:

$$
O _ { t } ^ { \mathrm { m e r g e } } = O _ { t } ^ { \mathrm { p r e } } + E _ { t } ^ { \mathrm { e v i c t } } , \qquad \hat { O } _ { t } = \mathcal { Q } _ { b _ { u } } \mathopen { } \mathclose \bgroup \left( O _ { t } ^ { \mathrm { m e r g e } } \aftergroup \egroup \right) .\tag{19}
$$

When the queue is full, the current readout has access to the original m historical branches and the current write; the persistent queue after the update still contains at most m vector pairs. The old state is updated and quantized at every step. PW delays the first incorporation of a new write into the low-bit old state. Before the window fills, there is no eviction term, but the old state still undergoes its prescribed update and writeback.

The recent component has rank at most m and is stored separately as FP8 vector pairs. Recurrence equivalence under ideal precision is established in Appendix A, and the FP8 encoding and complete update algorithm are given in Appendix B.4; combined configurations and equivalent state bit widths are described in Appendix B.5.

Case-study response. The right side of Figure 2 presents the generation with PW-S8, using the same prompt, random seed, sampling parameters, and GGA@4 bit-width configuration as the trajectory in Section 3.2. After rechecking, it carries forward that there are ten divisors and the sixth-largest is six, examines other candidates, and terminates naturally with the answer 48. The complete output contains 57,682 tokens, 24,239 fewer than without PW, a reduction of 29.59%. Section 5 evaluate the effects on scores and generation length across tasks.

## 5 Experiments

This section reports task performance, component comparisons, and generation behavior, and compares persistent-state storage costs in equivalent bits.

## 5.1 Experimental Setup

Models and comparison methods. We evaluate post-trained Qwen3.5-4B and Qwen3.5-9B [12, 13]. Each evaluated model contains 24 GDN layers, with 32 state/value heads of size 128 × 128 per layer and 768 heads in total; weights and convolutions remain in BF16. The @4 group compares naive INT4, STEPQuant@4, GGA@4, and each allocation combined with PW-S8, alongside FP32 and FP16 recurrent-state references. The @6 group compares naive INT6, STEPQuant@6, GGA@6, and each allocation combined with PW-S8; we additionally compare STEP/GGA@5 against @4+PW-S8 to examine how similar storage budgets are used. Naive integer states use symmetric quantization with a group size of 128 along the value axis; STEP and GGA use the same candidate quantizers, with a value group size of 32. The candidate integer bit widths for both @4 and @5 are {2, 4, 6, 8} bits, and those for @6 are {4, 6, 8} bits; all budgets retain 32 FP16 heads. PW-S8 denotes a window length of m = 8, with the queue format defined in Section 4.3; each combination uses the frozen precision configuration at its corresponding nominal budget. Floating-point references accumulate in FP32 and persistently store the recurrent state in the target dtype. State storage is measured in equivalent bits/value, including state codes, scales, FP16 pivots, and the PW queue; the accounting method and complete storage-cost table are provided in Appendix B.5.

Calibration. We perform offline calibration on 32 segments from the WikiText-2 training set, each containing 2048 tokens, to estimate candidate quantization errors, readout effects, and loss sensitivity, and to determine head bit widths and FP16 pivots. Calibration details are provided in Appendix B.

Evaluation data. We evaluate on randomly sampled evaluation sets drawn from AIME 2026 (30 problems), HMMT February 2026 (33 problems) [14], GPQA Diamond (198 questions) [15], HumanEval+ (164 problems) [16], MATH500 (500 problems) [17], IFBench (300 prompts) [18], and LiveCodeBench v6 (200 problems) [19]. The problem lists are fixed before evaluation and shared across all comparison methods; we independently sample five responses per problem for AIME and HMMT, and two responses per problem for the remaining datasets.

Generation settings. All settings use the same frozen problem lists, prompts, and per-problem seeds, with temperature = 1, top-p = 0.95, top-k = 20, thinking enabled, an output limit of 100,000 tokens (including thinking), and a context limit of 131,072. FP32/FP16 serve as floating-point state references; gains between methods are calculated only among completed quantized settings. Responses end naturally upon EOS; a repetition detector using the same rules also stops clearly periodic outputs, which are counted separately from outputs that reach the token limit. The rules are given in Appendix D. Responses without a final answer remain in the evaluation denominator.

Inference uses an A100-SXM4-80GB GPU with SGLang 0.5.12.

## 5.2 Main Results

Table 1 compares task scores for floating-point state references and the @4 and @6 settings. Figures 4 and 5 summarize the @4 score–length tradeoff; task-level lengths are provided in Appendix D, Table 6. At both budgets, the benefits of GGA and PW appear in task performance and generation behavior: loss-aware allocation improves the use of limited bit widths, and recent-write maintenance further improves scores and mitigates excessively long outputs.

Effects of low-bit state quantization on long-reasoning tasks. At @4, naive INT4 achieves macro-averages of only 1.67%/2.10% on 4B/9B, indicating that directly applying uniform low-bit quantization to the full recurrent state severely degrades model capabilities. STEPQuant’s mixedprecision scheme achieves macro-averages of 31.44%/37.42%, but the extent of recovery varies substantially across tasks. For example, its HumanEval+ scores remain at 78.96%/83.84%, whereas AIME scores are only 0.67%/2.67%, and HMMT scores are only 0.00%/4.24%. Long-reasoning tasks are particularly affected, and average scores alone do not fully reflect this degradation.

![](images/7a4fbb059b2012805d6d64dfd2c14f0e1236c662a64d9e0ab58fa8308644f0ec.jpg)

![](images/ee7b321a140cc5058615a7617e0be8bb8cbaa9ec4034ef6e1b0101b8b94fd19d.jpg)  
Figure 4: Score–length tradeoff on Qwen3.5-4B at @4. Both axes weight tasks equally. Gray arrows trace STEP to GGA to GGA+PW; FP32 and FP16 show floating-point references. PW denotes PW-S8 and INT4 denotes naive INT4.  
Figure 5: Score–length tradeoff on Qwen3.5-9B at @4, with FP32/FP16 recurrent-state references and naive INT4. Markers, method arrows, and the better direction follow Figure 4.

Table 1: Measured scores (%) on seven tasks at @4 and @6. For each model, floating-point references are followed by the @4 and @6 groups; boldface compares scores among completed quantized settings within each model and nominal-budget group. Scores are averaged across independent samples, and the macro-average weights all tasks equally. HE+, IFB, and LCB denote HumanEval+ base+plus, IFBench prompt strict, and the LiveCodeBench v6 evaluation subset, respectively.
<table><tr><td>Model</td><td>Method</td><td>AIME</td><td>HMMT</td><td>GPQA</td><td>HE+</td><td>MATH</td><td>IFB</td><td>LCB</td><td>Macro-avg.</td></tr><tr><td rowspan="7">4B</td><td>FP32</td><td>86.67</td><td>63.64</td><td>70.96</td><td>86.89</td><td>94.90</td><td>62.00</td><td>63.75</td><td>75.54</td></tr><tr><td>FP16</td><td>84.00</td><td>61.82</td><td>71.97</td><td>86.28</td><td>95.60</td><td>58.00</td><td>62.25</td><td>74.27</td></tr><tr><td>Naive INT4</td><td>0.00</td><td>0.00</td><td>1.26</td><td>0.00</td><td>2.40</td><td>8.00</td><td>0.00</td><td>1.67</td></tr><tr><td>STEP@4</td><td>0.67</td><td>0.00</td><td>16.41</td><td>78.96</td><td>47.30</td><td>46.50</td><td>30.25</td><td>31.44</td></tr><tr><td>GGA@4</td><td>30.67</td><td>17.58</td><td>50.51</td><td>78.66</td><td>83.60</td><td>51.67</td><td>43.75</td><td>50.92</td></tr><tr><td>STEP@4+PW-S8</td><td>54.67</td><td>37.58</td><td>60.61</td><td>82.62</td><td>94.00</td><td>56.00</td><td>60.25</td><td>63.67</td></tr><tr><td>GGA@4+PW-S8</td><td>74.00</td><td>57.58</td><td>62.12</td><td>81.10</td><td>97.30</td><td>56.50</td><td>60.25</td><td>69.83</td></tr><tr><td>Naive INT6</td><td>0.00</td><td>0.00</td><td></td><td>1.52 13.11</td><td></td><td>13.80</td><td>17.83</td><td>2.25</td><td>6.93</td></tr><tr><td>STEP@6</td><td></td><td>74.67</td><td>51.52</td><td>65.66</td><td>83.54</td><td>96.00</td><td>61.17</td><td>61.25</td><td>70.54</td></tr><tr><td>GGA@6</td><td>78.00</td><td>57.58</td><td></td><td>64.65</td><td>85.37</td><td>97.30</td><td>59.17</td><td>60.75</td><td>71.83</td></tr><tr><td>STEP@6+PW-S8</td><td>83.33</td><td>63.03</td><td>67.17</td><td>83.54</td><td></td><td>98.20</td><td>59.00</td><td>61.50</td><td>73.68</td></tr><tr><td>GGA@6+PW-S8</td><td>82.00</td><td>65.45</td><td>69.70</td><td>82.93</td><td>97.90</td><td>58.50</td><td></td><td>61.25</td><td>73.96</td></tr><tr><td rowspan="10">9B</td><td>FP32</td><td>90.67</td><td>66.67</td><td>76.77</td><td>87.20</td><td>97.50</td><td>64.83</td><td>76.00</td><td></td><td>79.95</td></tr><tr><td>FP16</td><td>88.67</td><td>70.30</td><td>76.01</td><td>86.89</td><td>96.70</td><td></td><td>64.50</td><td>74.00</td><td>79.58</td></tr><tr><td>Naive INT4</td><td>0.00</td><td>0.00</td><td>1.52</td><td>0.61</td><td>1.60</td><td>11.00</td><td></td><td>0.00</td><td>2.10</td></tr><tr><td>STEP@4</td><td>2.67</td><td>4.24</td><td>25.25</td><td>83.84</td><td>59.10</td><td></td><td>48.83</td><td>38.00</td><td>37.42</td></tr><tr><td>GGA@4</td><td>66.00</td><td>45.45</td><td>66.41</td><td>82.62</td><td></td><td>94.90</td><td>58.17</td><td>57.25</td><td>67.26</td></tr><tr><td>STEP@4+PW-S8</td><td>67.33</td><td>46.67</td><td>66.41</td><td>87.50</td><td>96.50</td><td></td><td>60.00</td><td>65.75</td><td>70.02</td></tr><tr><td>GGA@4+PW-S8</td><td>84.00</td><td>72.12</td><td>69.70</td><td>86.89</td><td>99.30</td><td></td><td>62.67</td><td>75.75</td><td>78.63</td></tr><tr><td>Naive INT6</td><td>0.00</td><td>0.00</td><td>5.05</td><td>16.77</td><td>24.70</td><td></td><td>23.50</td><td></td><td></td></tr><tr><td>STEP@6</td><td>80.00</td><td>63.64</td><td>70.20</td><td>88.11</td><td>97.90</td><td></td><td>66.33</td><td>6.00 72.75</td><td>10.86 76.99</td></tr><tr><td>GGA@6</td><td>86.67</td><td>66.06</td><td></td><td>75.00</td><td>88.11</td><td>98.80</td><td>65.17</td><td>74.75</td><td>79.22</td></tr><tr><td>STEP@6+PW-S8</td><td></td><td>86.00</td><td>67.27</td><td>74.24</td><td>85.06</td><td>99.20</td><td>64.50</td><td>74.50</td><td>78.68</td></tr><tr><td>GGA@6+PW-S8</td><td></td><td>84.00</td><td>67.27</td><td>79.80</td><td>88.11</td><td>98.80</td><td>65.67</td><td>75.50</td><td>79.88</td></tr></table>

AIME and HMMT therefore provide important tasks for examining precision allocation and recentinformation maintenance.

GGA recovers task performance under the same budget. Keeping the candidate quantizers and nominal budget unchanged and replacing only STEP allocation with GGA improves the 4B/9B macro-averages by 19.48/29.84 percentage points, to 50.92%/67.26%. This improvement is not confined to a single problem set: on 9B, for example, GPQA increases from 25.25% to 66.41%, MATH500 from 59.10% to 94.90%, and AIME and HMMT recover to 66.00% and 45.45%, respectively. These improvements under the same bit-width budget indicate that precision configuration across heads itself has a substantial effect on task performance. Together with the allocation observations in Section 3.1, these results suggest that sensitivity to final prediction loss provides a more effective basis for bit-width allocation, directing limited precision toward states with greater effects on predictions.

PW further improves existing precision allocations. Adding PW-S8 to GGA@4 further raises the 4B/9B macro-averages to 69.83%/78.63%, improvements of 18.92/11.37 percentage points over GGA@4, respectively. On 9B, AIME increases from 66.00% to 84.00%, HMMT from 45.45% to 72.12%, and the macro-average approaches the FP16 state reference of 79.58%. PW is also effective with STEP allocation: STEP@4+PW-S8 achieves macro-averages of 63.67%/70.02%. With PW-S8 fixed, GGA still improves over STEP allocation by 6.16/8.61 percentage points. Together, these comparisons demonstrate the complementary roles of the two designs: GGA improves precision configuration across heads, while PW protects recent writes during state updates, enabling existing precision allocations to better support sustained reasoning. Section 5.3 compares the additional storage introduced by the window with directly increasing bit widths.

Allocation and maintenance remain effective at @6. With a larger state budget, STEPQuant@6 recovers 4B/9B macro-averages to 70.54%/76.99%, improvements of 39.10/39.57 percentage points over @4; AIME recovers to 74.67%/80.00% and HMMT to 51.52%/63.64%. This indicates that increasing bit widths substantially mitigates the degradation of long reasoning at low budgets, and provides a comparison for examining the methods after capabilities have recovered. GGA@6 achieves macro-averages of 71.83%/79.22%, still exceeding STEPQuant@6 by 1.29/2.23 percentage points. The score gap between the two allocations narrows relative to @4, but the benefit of loss-aware allocation remains on both models.

PW also remains effective at this higher budget. STEPQuant@6+PW-S8 reaches macro-averages of 73.68%/78.68%, and GGA@6+PW-S8 reaches 73.96%/79.88%. The GGA combination improves over GGA@6 by 2.13/0.66 percentage points, approaching the overall performance of the floating-point state references. Thus, the value of recent-write maintenance extends beyond recovering severely degraded capabilities at @4 to further improving task performance at higher precision. Meanwhile, differences in generation length at similar scores can further reveal how different state-maintenance strategies affect reasoning. Figures 6 and 7 show the corresponding score–length comparisons, with close-ups of the high-score configurations.

GGA shortens mean outputs at both budgets. Table 6 first computes the mean full output length for each task and then averages equally across the seven tasks, preventing tasks with more problems from dominating the overall result. At @4, GGA reduces the 4B/9B mean lengths from STEP’s 58.23/55.42 thousand tokens to 42.96/37.78 thousand tokens, reductions of 26.23%/31.83%. At @6, although STEPQuant has already substantially recovered task performance, GGA still reduces mean lengths from 32.64/29.35 thousand tokens to 29.23/25.05 thousand tokens, reductions of 10.45%/14.65%. In both allocation comparisons, all seven tasks have shorter mean outputs on both models. The length improvements are therefore not driven by changes in only a few tasks; loss-aware allocation consistently mitigates excessive generation after quantization while improving task macro-average scores.

PW further improves output lengths in long reasoning. Adding PW-S8 to GGA@4 further reduces the 4B/9B mean lengths to 29.07/23.89 thousand tokens, additional reductions of 32.32%/36.77% relative to GGA without a window. Adding PW-S8 to STEPQuant@4 also reduces mean lengths by 39.63%/41.56%, respectively. Both allocation methods gain further benefits from the window, indicating that recent-write maintenance still has room for improvement after precision is reallocated. On 9B AIME, for example, STEPQuant@4, GGA@4, and GGA@4+PW-S8 produce mean lengths of 93.51, 69.27, and 43.23 thousand tokens, respectively, while their scores increase from 2.67% to 66.00% and then to 84.00%. These changes show that higher accuracy accompanies shorter mean outputs, consistent with the case in Section 4.3, where recent conclusions are retained and reasoning continues to advance.

![](images/0f7d1862537bb5b4dca76f4055f722f5f555b38c3a2360a0c1d8a4a0802b59f4.jpg)  
Figure 6: Score–length tradeoff on Qwen3.5-4B at @6. The inset enlarges the high-score configurations, including the FP32/FP16 references and method arrows. PW denotes PW-S8 and INT6 denotes naive INT6.

![](images/41e51886b20ee9b089ca6de397cd6e3afc0a3b49aa25d936d9d0a9550546a676.jpg)  
Figure 7: Score–length tradeoff on Qwen3.5-9B at @6. The inset enlarges the high-score configurations; markers, method arrows, and the better direction use the same conventions as Figures 4–6.

PW’s length benefits also extend to @6. STEPQuant@6+PW-S8 has 4B/9B mean lengths of 22.01/22.07 thousand tokens, reductions of 32.58%/24.80% relative to the corresponding settings without a window; GGA@6+PW-S8 has mean lengths of 24.86/21.85 thousand tokens, reductions of 14.98%/12.76% relative to GGA@6. Comparing the two window combinations, the GGA combination on 4B achieves a slightly higher macro-average score, whereas the STEP combination has shorter mean outputs; on 9B, the GGA combination achieves both a higher macro-average score and slightly shorter mean outputs. These results show that PW improves scores and mean lengths with both allocations, while the advantages of specific combinations vary with the model and budget.

![](images/4ba2c403808c7f1bc03b4ca73ac9e5910dc1ac74d74718168563708209f4fd37.jpg)  
Figure 8: Stopping categories on Qwen3.5-4B at @4. Natural EOS, the 100K cap (Cap), and repetition stops partition the responses; PW denotes PW-S8.

![](images/1e9f1d4e9e0ed8f9609dd233590c9d9949384a809c78041fed0b02d884d79aca.jpg)  
Figure 9: Stopping categories on Qwen3.5-9B at @4, alongside floating-point state references.

Shorter outputs accompany fewer abnormal stops. Figures 8 and 9 show the stopping categories at @4, further explaining the generation behavior behind the length improvements; complete re sults are provided in Appendix D, Table 8. At @4, 36.66%/30.67% of STEPQuant responses on 4B/9B reach the 100K limit, and another 9.97%/9.71% are stopped for repetitive output. With GGA@4+PW-S8, these rates decrease to 4.61%/2.44% and 0.46%/0.10%, respectively, while natural EOS rates reach 94.93%/97.47%. The reduction in mean output length thus accompanies fewer responses that exhaust the generation budget or fall into repetition, demonstrating effective mitiga-

tion of excessive generation.

Naive INT4 also produces very short outputs that all end with EOS, but its task scores are near zero; GGA and PW simultaneously improve task scores, shorten mean outputs, and reduce token-limit and repetition stops. This indicates that shorter outputs accompany the recovery of problem-solving capabilities and a higher proportion of completed responses. At @6, adding PW to either allocation also exhibits the joint trend of higher scores, shorter mean outputs, and fewer abnormal stops, further supporting the benefits of recent-write maintenance for quantized reasoning.

## 5.3 Supplementary Experiments: Higher Precision versus Window Maintenance at Similar Budgets

The main results show that both increasing bit widths and introducing PW improve task performance. A further question is how to use similar amounts of additional state storage: to increase the average bit width of the accumulated state, or to maintain recent writes independently? To examine this question, we pair @5 with @4+PW-S8 under the same allocation method: the former increases state bit widths, whereas the latter retains the @4 setting and adds a window. Figures 10 and 11 compare seven-task macro-averages and task-equal mean output lengths, respectively. The window configurations require 0.10–0.11 more bits/value than the corresponding @5 configurations; complete numeric results are provided in Appendix D, Table 7.

![](images/8352b18f5dfe58acdc74d0e4d8390b0814db4a3d41673db619d17b2c0e3aa10a.jpg)

![](images/eb12c7bd8aa8cef88010e391192bd5b3cf43af4fe73c5b536dc2f8454dcbaeac.jpg)  
Figure 10: Seven-task macro-average scores at similar storage budgets: @5 versus @4+PW-S8 under each allocation method. The two settings have similar equivalent state-storage costs (Table 7).  
Figure 11: Task-equal mean output lengths at similar storage budgets. The paired bars compare @5 with @4+PW-S8.

Using the additional budget for window maintenance. Under STEP allocation, @4+PW-S8 improves 4B/9B macro-averages over @5 by 3.29/1.57 percentage points and reduces mean output lengths by 5.63%/7.54%, respectively. Under GGA allocation, the corresponding score improvements are 2.61/0.41 percentage points, with mean-length reductions of 9.91%/10.80%. Thus, in these comparisons at similar storage budgets, window maintenance achieves higher macro-average scores and shorter mean outputs with both allocations and both models, indicating that PW’s benefits persist when compared with directly increasing bit widths.

On 9B, GGA@5 and GGA@4+PW-S8 achieve very similar macro-averages of 78.22% and 78.63%, but the latter still reduces mean output length from 26.78 thousand tokens to 23.89 thousand tokens. This indicates that even when task scores approach the floating-point reference, recent-write maintenance provides additional benefits by mitigating excessive outputs. Appendix D provides the per-task lengths and termination statistics for @5 in Tables 10 and 9, respectively.

Increasing state bit widths reduces quantization distortion in the accumulated state, whereas PW devotes additional storage to recent writes, delaying their entry into the low-bit historical state. These results support allocating part of the storage resources according to write age: beyond deciding how many bits each head should use, the time at which information is first subjected to low-bit quantization after being written also affects task performance and generation length.

## 6 Discussion and Limitations

GGA and PW improve recurrent-state quantization through precision allocation and write maintenance, respectively. GGA uses prediction-loss sensitivity to guide bit-width allocation across heads, improving task scores and shortening mean outputs on the evaluated models. PW independently maintains recent writes and delays their incorporation into the low-bit historical state, further miti gating excessively long generation on top of existing precision allocations. In supplementary comparisons at similar storage budgets, window maintenance also achieves higher macro-averages and shorter mean outputs, demonstrating the value of devoting part of the storage capacity to recentinformation maintenance.

The mechanisms behind excessively long outputs and repetitive generation require further study. PW mitigates verbose outputs after quantization, but some responses still reach the token limit or terminate because of repetition. We have not yet fully characterized how information written by tokens is retained and read during state recurrences, or how quantization perturbations affect its subsequent use. Future work will further analyze the contribution of recent writes to predictions and investigate its relationship to generation feedback, stalled reasoning, and repetitive outputs.

PW characterization and algorithm design remain open to improvement. Our current evaluation mainly considers a fixed window configuration. Systematic experiments on properties such as window length, queue precision, and branch-merging strategies remain incomplete. We will extend the corresponding comparisons and ablations, improve the window-maintenance algorithm, and further investigate joint optimization of window maintenance and global precision allocation.

Dedicated efficient inference kernels remain to be developed. We have not yet developed dedicated efficient inference kernels for PW. Future work will develop and optimize kernels for propagating and reading recent branches and merging them into the historical state, reduce computation and memory-access overhead in queue maintenance, and evaluate end-to-end inference latency and throughput.

Validation covers a limited set of models and methods. The current experiments mainly use Qwen3.5-4B/9B, with limited coverage of other model sizes, model families, and recurrent-state quantization methods. We will expand the range of models and comparison methods and assess performance across architectures and deployment conditions using task accuracy, generation length, and inference-performance metrics.

## 7 Conclusion

We propose GGA and PW, which use writeback sensitivity to future prediction loss to guide precision allocation across heads and maintain independent floating-point branches according to write age, respectively. At @4 on Qwen3.5-4B/9B, GGA allocation alone improves scores by 19.48/29.84 percentage points; adding PW-S8 yields task macro-averages of 69.83%/78.63%, improving over STEPQuant@4 by 38.39/41.21 percentage points. At @6, STEPQuant substantially recovers task performance, while GGA still reduces mean generation length, averaged equally across tasks, by approximately 10.45%/14.65% with similar or higher accuracy. At similar storage budgets, GGA@4+PW-S8 further reduces mean output length by 9.91%/10.80% relative to GGA@5, while increasing the macro-average by 2.61/0.41 percentage points. Allocation diagnostics and equivalentbit accounting further characterize precision placement and window storage overhead, respectively. Future work will expand sampling and investigate additional model families and window lengths.

## References

[1] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. ICML, 2020. https://arxiv.org/abs/ 2006.16236.

[2] Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon Kim. Gated Linear Attention Transformers with Hardware-Efficient Training. ICML, 2024. https://arxiv.org/abs/2312.06635.

[3] Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated Delta Networks: Improving Mamba2 with Delta Rule. ICLR, 2025. arXiv:2412.06464. https://arxiv.org/abs/2412.06464.

[4] Tri Dao and Albert Gu. Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. ICML, 2024. https://arxiv.org/abs/2405.21060.

[5] Tao Zhang, Jianchao Tan, Pingwei Sun, Yanqi Yu, Zunhai Su, Zixu Jiang, Yuchen Xie, Xunliang Cai, and Ziqian Zeng. DAMP: Decay-Aware Mixed-Precision Recurrent-State Quantization. arXiv:2608.27513v2, 2026. https://arxiv.org/abs/2608.27513v2.

[6] Bingchen Yao, Haobo Xu, Haokun Lin, Yichen Wu, Ziyu Guo, Renrui Zhang, Zhichao Lu, Zhenan Sun, and Ying Wei. STEPQuant: When and Where Errors Matter in Delta-Rule Recurrent State Quantization. arXiv:2609.38169v1, 2026. https://arxiv.org/abs/2609.38169v1.

[7] Yi Pan, Haocheng Xi, Kan Zhu, Xingyang Li, Yibo Wu, Mayank Mishra, Hongtao Zhang, William X. Zheng, Baris Kasikci, Song Han, Kurt Keutzer, Rishabh Iyer, and Ion Stoica. LeapQuant: Efficient Linear Attention with Accurate Recurrent State Quantization. arXiv:2609.38166v1, 2026. https:// arxiv.org/abs/2609.38166v1.

[8] Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. ICML, 2024. https: //arxiv.org/abs/2402.02750.

[9] Hung-Yueh Chiang, Chi-Chih Chang, Natalia Frumkin, Kai-Chiang Wu, Mohamed S. Abdelfattah, and Diana Marculescu. Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models. arXiv:2503.22879, 2025. https://arxiv.org/abs/2503.22879.

[10] Chen Tianqi, Yuanteng Chen, Peisong Wang, Weixiang Xu, Zeyu Zhu, and Jian Cheng. Q-Mamba: Towards More Efficient Mamba Models via Post-training Quantization. Findings of ACL, 2025. https: //aclanthology.org/2025.findings-acl.551/.

[11] Sehoon Kim, Coleman Hooper, Amir Gholami, Zhen Dong, Xiuyu Li, Sheng Shen, Michael W. Mahoney, and Kurt Keutzer. SqueezeLLM: Dense-and-Sparse Quantization. arXiv:2306.07629, 2023. https: //arxiv.org/abs/2306.07629.

[12] Qwen Team. Qwen3.5-4B: Official Model Card. Hugging Face, 2026. https://huggingface.co/ Qwen/Qwen3.5-4B.

[13] Qwen Team. Qwen3.5-9B: Official Model Card. Hugging Face, 2026. https://huggingface.co/ Qwen/Qwen3.5-9B.

[14] Jasper Dekoninck et al. Beyond Benchmarks: MathArena as an Evaluation Platform for Mathematics with LLMs. arXiv:2605.00674, 2026. https://arxiv.org/abs/2605.00674.

[15] David Rein et al. GPQA: A Graduate-Level Google-Proof Q&A Benchmark. arXiv:2311.12022, 2023. https://arxiv.org/abs/2311.12022.

[16] Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation. NeurIPS, 2023. https://arxiv.org/abs/2305.01210.

[17] HuggingFaceH4. MATH-500. Hugging Face dataset repository, 2024. https://huggingface.co/ datasets/HuggingFaceH4/MATH-500.

[18] Valentina Pyatkin et al. Generalizing Verifiable Instruction Following. NeurIPS Datasets and Benchmarks, 2025. https://arxiv.org/abs/2507.02833.

[19] Naman Jain et al. LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code. arXiv:2403.07974, 2024. https://arxiv.org/abs/2403.07974.

## A Equivalence of PW under Exact Arithmetic

Assume that the initial old state equals the initial GDN state and that the recent queue is empty. The two recurrences use identical exogenous $q , k , v , \beta , \gamma$ , all operations use infinite precision, and the old state is not quantized. Denote the sum of the two components by

$$
S _ { t } ^ { \mathrm { P W } } = O _ { t } + \sum _ { j \in \mathcal { I } _ { t } } p _ { j , t } v _ { j } ^ { \top } .\tag{20}
$$

The initial conditions give $S _ { 0 } ^ { \mathrm { P W } } = S _ { 0 }$ . If this equality holds at the preceding step, applying $A _ { t }$ to the old state and all existing left factors and then appending $B _ { t }$ yields

$$
\begin{array} { l } { S _ { t } ^ { \mathrm { P W } } = A _ { t } O _ { t - 1 } + \displaystyle \sum _ { j \in \mathcal { T } _ { t - 1 } } A _ { t } p _ { j , t - 1 } v _ { j } ^ { \top } + B _ { t } } \\ { \quad = A _ { t } S _ { t - 1 } ^ { \mathrm { P W } } + B _ { t } = S _ { t } . } \end{array}\tag{21}
$$

When the window is full, eviction only moves the same term $p _ { t - m , t } v _ { t - m } ^ { \top }$ from recent to old, leaving their sum unchanged. Induction therefore establishes that the logical state and current readout are identical at every step. Finite-precision rounding, quantization feedback, and activation changes during free generation fall outside the assumptions of this equivalence.

With quantization, $\mathcal { Q } ( O _ { t } ) + H _ { t }$ generally differs from $\mathcal { Q } ( O _ { t } { + } H _ { t } )$ , where $H _ { t }$ is the sum of the recent branches. PW thus changes the state representation and the location at which errors are injected. The algebraic equivalence above does not guarantee that elementwise errors necessarily decrease in finite-precision deployment, nor that free-generation trajectories match those of the high-precision model.

## B Calibration, Deployment, and Storage Accounting

## B.1 Frozen Plans and Budget Settings

Each evaluated model contains $2 4 \times 3 2 = 7 6 8$ state/value heads of size $1 2 8 \times 1 2 8 .$ , and each plan selects 32 FP16 pivots. Under the @4, @5, and @6 settings, the integer bit-width sum budgets $B _ { \mathrm { i n t } }$ for the remaining 736 heads are 2816, 3584, and 4352, respectively. The $\textcircled { a } 5$ setting increases the budget using the same calibration costs and the same candidate formats as @4. Table 2 lists the bit-width distributions of the frozen plans; GGA and STEP may select different pivot identities.

Table 2: Head counts in the precision allocation plans. PW configurations not listed separately use the same bit-width tables as their corresponding configurations without a window.
<table><tr><td>Model</td><td>Allocation</td><td>INT2</td><td>INT4</td><td>INT6</td><td>INT8</td><td>FP16</td></tr><tr><td>4B</td><td>STEP@4</td><td>380</td><td>119</td><td>158</td><td>79</td><td>32</td></tr><tr><td>4B</td><td>GGA@4</td><td>369</td><td>139</td><td>151</td><td>77</td><td>32</td></tr><tr><td>4B</td><td>STEP@5</td><td>214</td><td>185</td><td>140</td><td>197</td><td>32</td></tr><tr><td>4B</td><td>GGA@5</td><td>215</td><td>171</td><td>165</td><td>185</td><td>32</td></tr><tr><td>4B</td><td>STEP@6</td><td>0</td><td>302</td><td>164</td><td>270</td><td>32</td></tr><tr><td>4B</td><td>GGA@6</td><td>0</td><td>293</td><td>182</td><td>261</td><td>32</td></tr><tr><td>4B</td><td>GGA@6+PW-S8</td><td>0</td><td>319</td><td>130</td><td>287</td><td>32</td></tr><tr><td>9B</td><td>STEP@4</td><td>372</td><td>132</td><td>156</td><td>76</td><td>32</td></tr><tr><td>9B</td><td>GGA@4</td><td>361</td><td>167</td><td>119</td><td>89</td><td>32</td></tr><tr><td>9B</td><td>STEP@5</td><td>213</td><td>185</td><td>143</td><td>195</td><td>32</td></tr><tr><td>9B</td><td>GGA@5</td><td>218</td><td>157</td><td>184</td><td>177</td><td>32</td></tr><tr><td>9B</td><td>STEP@6</td><td>0</td><td>299</td><td>170</td><td>267</td><td>32</td></tr><tr><td>9B</td><td>GGA@6</td><td>0</td><td>285</td><td>198</td><td>253</td><td>32</td></tr><tr><td>9B</td><td>GGA@6+PW-S8</td><td>0</td><td>315</td><td>138</td><td>283</td><td>32</td></tr></table>

## B.2 Solving the Global Allocation by Dynamic Programming

To solve Eq. (11), denote the non-pivot heads in a fixed order by $u _ { 1 } , \ldots , u _ { n }$ . Let $F ( j , s )$ be the minimum cost for the first j heads when their bit widths sum to exactly s. The recurrence is

$$
F ( j , s ) = \operatorname* { m i n } _ { b \in \mathcal { B } } \bigl [ F ( j - 1 , s - b ) + C _ { u _ { j } , b } ^ { \mathrm { G G A } } \bigr ] .\tag{22}
$$

The boundary condition is $F ( 0 , 0 ) = 0 { \mathrm { ; } }$ , with all other initial values set $\mathrm { \bf t o } + \infty$ . Backtracking from $F ( n , B _ { \mathrm { i n t } } )$ gives the integer bit-width table.

The preliminary allocation applies the same recurrence to all heads, with a target budget of $\bar { b } \vert \mathcal { U } \vert$

## B.3 Calibration and Deployment Settings

Calibration for standalone GGA allocation uses 32 frozen token sequences from the WikiText-2 training split, each containing 2048 tokens, with data sampling seed 0. Each sequence is processed in a separate forward pass, and the language-model loss is the sum of the 2047 next-token crossentropies within that sequence. Updated states are collected every 8 tokens, giving 8192 candidate positions per layer, from which reservoir sampling with seed 0 retains 64 snapshots. State residuals and writeback adjoints are paired at the same sequence and token positions, and Eq. (8) is averaged over these snapshots. Gradients are taken from the original GDN head readouts, and the adjoints are computed using reference activations that have been normalized, scaled, and mapped to the state heads.

The implementation divides all GGA costs across the model by the median positive cost. This common positive rescaling changes neither the optimal solution of the objective nor the pivot risk ranking; it is not normalization per layer or per head. Candidate costs and precision allocations for each model are determined offline on the calibration set. Inference uses the frozen bit widths and pivots for each configuration. The local diagnostics use the corresponding STEP/GGA@4 allocation configurations.

Calibration for GGA@6+PW-S8 uses a PW reference trajectory with window length $m = 8$ , in which the historical states of non-pivot heads are stored in FP32 and the recent queue uses a scaled FP8 representation. Candidate costs retain the squared-gradient weighting in Eq. (8), using snapshots of the historical state $O _ { u , t }$ and the gradient of the full-model cross-entropy loss with respect to that writeback state as the sensitivity. Backpropagation through FP8 and FP16 rounding uses an identity straight-through estimator (identity STE). The 32 FP16 pivots of the corresponding GGA@6 configuration remain fixed; global dynamic programming allocates the remaining heads over the candidate set {4, 6, 8} with an integer bit-width sum of 4352.

## B.4 FP8 Encoding and Online Updates for PW

The recent queue in PW uses E4M3FN FP8 encoding. Each $p / v$ vector of each branch in each head stores its own FP32 scale. Propagation, readout, and eviction use the reconstructed FP32 values. PW-S8 denotes a window length of $m = 8$ . For a nonzero vector $x ,$ define

$$
s ( x ) = \frac { \| x \| _ { \infty } } { 4 4 8 } , \qquad \mathcal { R } _ { \mathrm { F P 8 } } ( x ) = s ( x ) \mathrm { F P 3 2 } \biggl [ \mathrm { F P 8 } _ { \mathrm { E 4 M 3 F N } } \biggl ( \frac { x } { s ( x ) } \biggr ) \biggr ] ,\tag{23}
$$

with $\mathcal { R } _ { \mathrm { F P 8 } } ( 0 ) = 0$ . The map $\mathcal { R }$ <sub>FP8</sub> composes encoding and reconstruction; persistent storage retains only the FP8 codes and the scale. The finite-precision recent branches are

$$
\begin{array} { r l } & { \bar { p } _ { j , j } = \mathcal { R } _ { \mathrm { F P 8 } } ( \beta _ { j } k _ { j } ) , \qquad \bar { v } _ { j } = \mathcal { R } _ { \mathrm { F P 8 } } ( v _ { j } ) , } \\ & { \bar { p } _ { j , t } = \mathcal { R } _ { \mathrm { F P 8 } } ( A _ { t } \bar { p } _ { j , t - 1 } ) . } \end{array}\tag{24}
$$

After propagation at each step, existing left factors are re-encoded and their scales updated; their reconstructed values are then used for the current readout and eviction. Each $\bar { v } _ { j }$ remains fixed after enqueueing. The current new write contributes to the current readout using FP32 intermediate values, and its $p / v$ vectors are encoded and enqueued only afterward.

Algorithm 1 represents recent storage as a bounded queue R ordered by birth time. Queue elements are written as their FP32 reconstructed values ${ \bar { p } } , { \bar { v } }$ . Persistent matrices and queue vectors are decoded before computation. The map $\mathcal { R } _ { \mathrm { F P 8 } }$ implicitly includes FP8 encoding, scale writeback, and reconstruction, while $\mathcal { Q } _ { b }$ denotes the reconstructed value after old-state encoding. In the algorithm, A is implemented using the low-rank form in Eq. (14), without constructing a dense matrix. FP32 intermediate values are used only within the current step.

The oldest branch is evicted only after the current readout is complete. The current new write contributes to the readout as temporary vectors and is then enqueued, so no additional persistent queue slot numbered $m + 1$ is required.

FP16 pivots retain full-state updates and an FP16 persistent representation, with no PW queue enabled. Prefill follows the base execution path, and the last m writes of the prompt are not automatically added to the decode queue. Queue storage accounting is given in Appendix B.5.

Algorithm 1 PW decode update for one head   
Require: Reconstructed persistent matrix ${ \hat { O } } ,$ FP8 queue with scales, current $q , k , v , \gamma , \beta$   
Frozen format b or FP16 pivot, window m $\geq 1 , | \mathcal { R } | \leq m$   
Ensure: Current readout o, persistent representation and queue for the next step   
1: $\smash { A \gets \gamma ( I - \beta k k ^ { \top } ) , p _ { \mathrm { n e w } } \gets \beta k }$   
2: if the current head is an FP16 pivot then   
3: $S ^ { \mathrm { c a n d } } \gets A \hat { O } + p _ { \mathrm { n e w } } v ^ { \top }$   
4: $o  ( S ^ { \mathrm { c a n d } } ) ^ { \top } q$   
5: $\hat { O }  \mathrm { F P 1 6 } ( S ^ { \mathrm { c a n d } } )$   
6: return $o , \hat { O } , \varnothing$   
7: end if   
8: ${ O ^ { \mathrm { p r e } }  A \hat { O } }$   
9: for each existing vector pair (¯p, v¯) in the queue do   
10: $\bar { p }  \mathcal { R } _ { \mathrm { F P 8 } } ( A \bar { p } )$ ▷ Decode and propagate; re-encode and update the scale   
11: end for   
12: $o \gets ( O ^ { \mathrm { p r e } } ) ^ { \top } q + \sum \bar { v } ( \bar { p } ^ { \top } q ) + v ( p _ { \mathrm { n e w } } ^ { \top } q )$   
(¯p,v¯)   
13: $O ^ { \mathrm { m e r g e } }  O ^ { \mathrm { p r e } }$   
14: $\mathbf { i f } \left| \mathcal { R } \right| = m$ then   
15: (¯p<sub>old</sub>, v¯<sub>old</sub>)  PopOldest( )   
16: $\dot { O } ^ { \mathrm { m e r g e } }  O ^ { \mathrm { m e r g \bar { e } } } + \bar { p } _ { \mathrm { o l d } } \bar { v } _ { \mathrm { o l d } } ^ { \top }$   
17: end if   
18: Append $( \mathcal { R } _ { \mathrm { F P 8 } } ( p _ { \mathrm { n e w } } ) , \mathcal { R } _ { \mathrm { F P 8 } } ( v ) )$ to   
19: $\hat { O }  \mathcal { Q } _ { b } ( O ^ { \mathrm { m e r g e } } )$ ▷ Encode and write back old at every step   
20: return $o , \hat { O } , \mathcal { R }$

Propagation of the recent left factors costs $O ( m d _ { k } )$ , and recent readout costs $O ( m ( d _ { k } + d _ { v } ) )$ . The old-state update, the outer product of the evicted branch, and encoding also involve $O ( d _ { k } d _ { v } )$ operations and quantization encoding overhead. Combined configurations and equivalent state bit widths are given in Appendix B.5.

## B.5 State Storage Accounting and Queue Capacity

In the combined configuration, GGA determines the bit-width table and pivot set offline, and PW maintains the old state online at the corresponding bit widths. Deployment involves no backpropagation, and both the window length and the bit width of every head are frozen.

This section accounts for the persistent data arrays of a single request’s state slot, converting storage to equivalent bit/value using the total number of elements in all GDN state matrices, $\begin{array} { r } { N _ { S } = \sum _ { u \in \mathcal { U } } \bar { d } _ { k , u } d _ { v , u } , } \end{array}$ as the denominator. The accounting includes integer codes, quantization scales, FP16 pivots, and reserved PW queues. It excludes indices, counters, shared plan metadata, unused request slots in the state pool, and temporary computation workspace, as well as weights, full-attention KV caches, and convolution caches.

Let $\mathcal { U } _ { I } = \mathcal { U } \setminus \mathcal { P }$ be the set of non-pivot heads, and let $M _ { Q , u } ( b _ { u } )$ be the packed old-state storage in bytes for an integer head, including integer codes and the scales for that format. Let $\boldsymbol { \mathcal { U } } _ { R }$ be the set of heads for which PW queues are actually reserved. Each FP8 branch stores two encoded vectors and their respective FP32 scales, requiring $d _ { k , u } + d _ { v , u } + 8$ bytes. The total number of bytes in the persistent data arrays being accounted for is

$$
M _ { \mathrm { p e r s i s t } } = \sum _ { u \in \mathcal { U } _ { I } } M _ { Q , u } ( b _ { u } ) + \sum _ { u \in \mathcal { P } } 2 d _ { k , u } d _ { v , u } + m \sum _ { u \in \mathcal { U } _ { R } } ( d _ { k , u } + d _ { v , u } + 8 ) .\tag{25}
$$

The corresponding equivalent state bit width and additional queue bit width are

$$
b _ { \mathrm { e f f } } = \frac { 8 M _ { \mathrm { p e r s i s t } } } { N _ { S } } , \qquad \Delta b _ { \mathrm { q u e u e } } = \frac { 8 m } { N _ { S } } \sum _ { u \in \mathcal { U } _ { R } } ( d _ { k , u } + d _ { v , u } + 8 ) .\tag{26}
$$

When PW is disabled, $\boldsymbol { \mathcal { U } } _ { R } = \boldsymbol { \mathcal { O } }$ . With PW enabled, the current layout reserves queues for all heads, so $\mathcal { U } _ { R } = \mathcal { U }$

For $d _ { k } = d _ { v } = 1 2 8 , m = 8$ , each FP8 vector occupies $1 2 8 + 4 = 1 3 2$ bytes, including its codes and FP32 scale. Each head with a queue requires $8 \times 2 \times 1 3 2 = 2 1 1 2$ bytes, equivalent to 1.03125 bit/value of that head’s state matrix. Allocating queues only for the 736 non-pivot heads would add approximately 0.98828 bit/value when averaged over all 768 heads. The current storage layout reserves queues for every head, including pivots; Table 3 therefore accounts for an additional 1.03125 bit/value. Small metadata such as counters is accounted for separately.

Table 3 converts the nominal budgets to storage under the same accounting convention: @5 requires approximately 5.68 bit/value, and @4+PW-S8 approximately 5.78–5.79 bit/value; @6 and @6+PW-S8 require 6.573 and 7.604 bit/value, respectively.

Table 3: Equivalent state bit widths (bit/value) for different allocation and window settings. State encoding, scales, FP16 pivots, and PW queues are included using the fixed array layout. FP32/FP16 floating-point states require 32/16 bit/value, respectively.
<table><tr><td>Model</td><td>Allocation</td><td>@4</td><td>@4+PW</td><td>@5</td><td>@6</td><td>@6+PW</td></tr><tr><td>4B</td><td>STEP</td><td>4.758</td><td>5.790</td><td>5.677</td><td>6.573</td><td>7.604</td></tr><tr><td></td><td>GGA</td><td>4.753</td><td>5.784</td><td>5.678</td><td>6.573</td><td>7.604</td></tr><tr><td>9B</td><td>STEP</td><td>4.755</td><td>5.786</td><td>5.677</td><td>6.573</td><td>7.604</td></tr><tr><td></td><td>GGA</td><td>4.749</td><td>5.780</td><td>5.679</td><td>6.573</td><td>7.604</td></tr></table>

## C Protocol and Binned Data for Precision Allocation Diagnostics

This section supplements the precision configuration analysis in Section 3.1 with the experimental protocol, metric definitions, and binned data. The analysis uses all 768 state heads of Qwen3.5-4B.

## C.1 Reference Trajectories and Single Interventions

This analysis uses the H256 local diagnostic set. Questions are drawn from the evaluation tasks listed in the main text and selected according to the frozen source fingerprint and hash order, excluding calibration and previously used development questions. Selection does not depend on differences between methods or answer correctness. The set must satisfy the replay length requirement and is used for local intervention analysis; results are averaged equally over questions rather than datasets.

Given a generated sequence $g [ 0 : T ]$ , at the after-call-d point the model has consumed $g [ 0 : d ]$ and the logits that currently predict g[d] have already been determined. We then quantize the entire accumulated state of only the target head once, reconstruct it in FP32, and replay the fixed sequence $g [ d : d + 2 5 6 ]$ . The two checkpoints are determined by the trajectory length:

$$
D _ { \mathrm { m a x } } = T - 2 5 6 - 1 ,\tag{27}
$$

$$
d _ { 1 } = 1 + \lfloor 0 . 2 5 ( D _ { \mathrm { m a x } } - 1 ) \rfloor ,\tag{28}
$$

$$
d _ { 2 } = \operatorname* { m a x } \{ d _ { 1 } + 1 , 1 + \lfloor 0 . 6 5 ( D _ { \operatorname* { m a x } } - 1 ) \rfloor \} .\tag{29}
$$

The current logits, which have already been produced, are excluded from the 256 steps after the intervention. Teacher forcing fixes the token inputs but does not fix the activations or states of the whole model after the intervention. This therefore differs from the fixed exogenous activation condition used for the algebraic derivations in the Method section.

Every head is tested at INT2, INT4, INT6, and INT8, using the frozen candidate quantizer with dual-axis fitting, row-impact, INT2 group32, and FP16 scales.

## C.2 Empirical Quantization Impact and Binned Statistics

For question i, checkpoint $p ,$ head $u ,$ and integer format $b \in \{ 2 , 4 , 6 , 8 \}$ , define the mean predictive distribution divergence over the subsequent 256 steps as

$$
R _ { i , p , u , b } = \frac { 1 } { 2 5 6 } \sum _ { s = 1 } ^ { 2 5 6 } D _ { \mathrm { K L } } \Big ( P _ { i , p , s } ^ { \mathrm { F P } } \parallel P _ { i , p , u , b , s } ^ { \mathrm { i n j } } \Big ) ,\tag{30}
$$

where $P ^ { \mathrm { F P } }$ and $P ^ { \mathrm { i n j } }$ denote the model’s predictive probability distributions over the vocabulary along the reference trajectory and after a single intervention, respectively. Let $N _ { \mathrm { d i a g } }$ be the number of diagnostic questions. We first average the two checkpoints within each question and then average equally over questions, and define a composite quantization impact across the four precisions:

$$
\begin{array} { r } { R _ { u , b } = \frac { 1 } { N _ { \mathrm { d i a g } } } \displaystyle \sum _ { i = 1 } ^ { N _ { \mathrm { d i a g } } } \frac { 1 } { 2 } \displaystyle \sum _ { p = 1 } ^ { 2 } R _ { i , p , u , b } , } \\ { I _ { u } = \frac { 1 } { 4 } \displaystyle \sum _ { b \in \{ 2 , 4 , 6 , 8 \} } R _ { u , b } . } \end{array}\tag{31}
$$

The metric $I _ { u }$ summarizes quantization residuals and their subsequent propagation effects under common candidate precisions, providing the same observational coordinate for the two precision configurations. GGA solves for bit widths using the calibration costs in Eq. (8).

Heads are partitioned by $I _ { u }$ into 11 equal-width bins of width $0 . 3 0 \times 1 0 ^ { - 4 }$ . For each nonempty bin $\mathcal { T } _ { k } \left( n _ { k } > 0 \right)$ , both methods compute the mean state-value bit width over the same set of heads:

$$
\bar { b } _ { a , k } = \frac { 1 } { n _ { k } } \sum _ { u : I _ { u } \in \mathcal { Z } _ { k } } b _ { u } ^ { ( a ) } , \qquad n _ { k } = \big | \{ u : I _ { u } \in \mathcal { Z } _ { k } \} \big | ,\tag{32}
$$

where $a \in \{ \mathrm { S T E P } , \mathrm { G G A } \}$ . Integer formats count according to their bit widths, and FP16 counts as 16 bits. Both configurations contain 32 FP16 pivots, and the bit widths of the 736 integer heads sum to 2816 in each configuration, giving a weighted mean of 4.33 bit/value across all heads. This statistic accounts for state-value encoding; the storage overhead of scales and metadata is accounted for separately in Appendix B.5.

The bin boundaries are $\left[ 0 . 3 0 , 0 . 6 0 , \dots , 3 . 6 0 \right] \times 1 0 ^ { - 4 }$ . Bins are left-closed and right-open, except for the final bin, which is closed on the right. All 768 heads are included, and empty bins are not filled with zeros. Table 4 reports the final statistics for all 11 bins, with displayed values rounded to two decimal places. The composite metric and this binning are descriptive analyses conducted after the experiment; the source table has already been aggregated across questions and checkpoints.

Both configurations query the same table of measured impacts. For the 61 heads in the union of the FP16 pivot sets, we additionally measure FP32(FP16(S)). The same definition of R extends to these additionally tested FP16 formats, whereas the composite metric $I _ { u }$ still averages only the four integer formats. Querying $R _ { u , b _ { u } }$ at each head’s assigned bit width and averaging equally over heads gives mean single-head local KL values of $2 . 1 5 5 5 9 \times 1 0 ^ { - 4 }$ for STEPQuant and $2 . 1 6 0 1 3 \times 1 0 ^ { - 4 }$ for GGA, with GGA approximately 0.21% higher. This statistic corresponds to a single intervention on each head; the joint effects of persistent quantization are examined separately through task evaluation.

Table 4: Mean state-value bit width under the common composite impact metric. Bin units are $1 0 ^ { - 4 }$ , and FP16 counts as 16 bits.
<table><tr><td>Composite impact bin</td><td>Heads</td><td>STEPQuant</td><td>GGA</td></tr><tr><td>[0.30, 0.60)</td><td>10</td><td>2.00</td><td>2.00</td></tr><tr><td>[0.60, 0.90)</td><td>7</td><td>4.00</td><td>2.57</td></tr><tr><td>[0.90, 1.20)</td><td>25</td><td>2.96</td><td>2.40</td></tr><tr><td>[1.20, 1.50)</td><td>56</td><td>5.14</td><td>3.36</td></tr><tr><td>[1.50, 1.80)</td><td>81</td><td>5.28</td><td>3.09</td></tr><tr><td>[1.80, 2.10)</td><td>104</td><td>4.37</td><td>3.13</td></tr><tr><td>[2.10, 2.40)</td><td>127</td><td>4.19</td><td>4.02</td></tr><tr><td>[2.40, 2.70)</td><td>231</td><td>3.73</td><td>4.56</td></tr><tr><td>[2.70, 3.00)</td><td>125</td><td>5.02</td><td>6.96</td></tr><tr><td>[3.00, 3.30)</td><td>0</td><td></td><td></td></tr><tr><td>[3.30, 3.60]</td><td>2</td><td>7.00</td><td>16.00</td></tr></table>

## D Evaluation Protocol and Supplementary Statistics

The random evaluation subsets for all tasks are fixed before inference, and models and methods share the same question lists. LiveCodeBench v6 questions are selected using a predetermined hash ordering, with question identifiers, source file hashes, and the original ordering frozen. The frozen upstream evaluation code and grading package versions are listed in Table 5. For IFBench, this paper uses prompt strict rather than the prompt loose metric usually reported in its paper.

Table 5: Frozen upstream code and grading dependencies for the formal evaluation. Code repositories are identified by the first 12 characters of the commit hash, and grading packages by their version numbers.
<table><tr><td>Implementation</td><td>Task</td><td>Version</td></tr><tr><td>MathArena</td><td>AIME, HMMT</td><td>b89f2f0ad64c</td></tr><tr><td>math-verify</td><td>MATH500</td><td>0.9.0</td></tr><tr><td>GPQA</td><td>GPQA Diamond</td><td>56686c06f5e1</td></tr><tr><td>EvalPlus</td><td>HumanEval+</td><td>26d6d00bb1fd</td></tr><tr><td>IFBench</td><td>IFBench</td><td>1c40f0c10d9b</td></tr><tr><td>LiveCodeBench</td><td>LCB v6 subset</td><td>28fef95ea8c9</td></tr></table>

In addition to natural EOS and the 100K length cap, all settings use the same stopping rule for short-period repetition. Repetition stops and length-cap terminations are reported separately. The final answer is the content after the last </think>; it is counted as missing if empty after removing termination markers. Missing final answers can overlap with any stopping category, and all responses remain in the evaluation denominator.

Tables 6 and 7 provide the complete task-level lengths and numeric results for the similar-budget comparison in Section 5.3.

Table 6: Mean full output lengths (thousand tokens) by task for 4B/9B at @4 and @6, alongside floatingpoint state references. Lengths include thinking and final, exclude the input prompt, and retain all stopping types. Mean is the equally weighted average of the seven task means, computed from unrounded values. Task abbreviations follow Table 1.
<table><tr><td>Model</td><td>Method</td><td>AIME</td><td>HMMT</td><td>GPQA</td><td>HE+</td><td>MATH</td><td>IFB</td><td>LCB</td><td>Mean</td></tr><tr><td>4B</td><td>FP32 FP16</td><td>39.79 47.98</td><td>53.10 63.50</td><td>26.62 28.41</td><td>1.27 1.30</td><td>15.06</td><td>17.81 20.92</td><td>20.94</td><td>24.94</td></tr><tr><td></td><td>Naive INT4</td><td>3.74</td><td>3.33</td><td></td><td></td><td>16.46</td><td></td><td>22.48</td><td>28.72</td></tr><tr><td></td><td>STEP@4</td><td>94.21</td><td>97.13</td><td>3.03 84.17</td><td>2.18 2.06</td><td>3.18</td><td>2.87</td><td>2.64</td><td>2.99</td></tr><tr><td></td><td>GGA@4</td><td>81.36</td><td>86.87</td><td></td><td>1.41</td><td>56.67</td><td>27.30</td><td>46.06</td><td>58.23</td></tr><tr><td></td><td>STEP@4+PW-S8</td><td>70.31</td><td>81.62</td><td>48.60</td><td></td><td>30.33</td><td>16.66</td><td>35.46</td><td>42.96</td></tr><tr><td></td><td>GGA@4+PW-S8</td><td>54.13</td><td>67.48</td><td>34.76</td><td>1.44 1.26</td><td>19.68</td><td>12.85</td><td>25.38</td><td>35.15</td></tr><tr><td></td><td></td><td></td><td></td><td>30.22</td><td></td><td>15.06</td><td>11.25</td><td>24.10</td><td>29.07</td></tr><tr><td></td><td>Naive INT6</td><td>32.21</td><td>31.65</td><td>21.74</td><td>7.53</td><td>23.09</td><td>7.11</td><td>18.35</td><td>20.24</td></tr><tr><td></td><td>STEP@6</td><td>59.40</td><td>74.65</td><td>34.66</td><td>1.29</td><td>17.19</td><td>16.59</td><td>24.73</td><td>32.64</td></tr><tr><td></td><td>GGA@6</td><td>52.69</td><td>66.17</td><td>30.44</td><td>1.20</td><td>14.93</td><td>15.90</td><td>23.31</td><td>29.23</td></tr><tr><td></td><td>STEP@6+PW-S8</td><td>40.14</td><td>52.34</td><td>20.96</td><td>1.23</td><td>11.18</td><td>11.81</td><td>16.41</td><td>22.01</td></tr><tr><td>9B</td><td>GGA@6+PW-S8</td><td>44.63</td><td>59.15</td><td>25.25</td><td>1.30</td><td>12.66</td><td>11.64</td><td>19.36</td><td>24.86</td></tr><tr><td></td><td>FP32 FP16</td><td>36.05</td><td>48.40</td><td>24.72</td><td>1.01</td><td>12.67</td><td>14.82</td><td>21.06</td><td>22.68</td></tr><tr><td></td><td></td><td>40.29</td><td>54.15</td><td>26.16</td><td>1.03</td><td>14.13</td><td>15.68</td><td>22.04</td><td>24.78</td></tr><tr><td></td><td>Naive INT4</td><td>3.75</td><td>3.67</td><td>3.82</td><td>2.74</td><td>3.76</td><td>3.12</td><td>3.40</td><td>3.46</td></tr><tr><td></td><td>STEP@4</td><td>93.51</td><td>94.62</td><td>77.66</td><td>1.31</td><td>48.28</td><td>27.54</td><td>45.02</td><td>55.42</td></tr><tr><td></td><td>GGA@4</td><td>69.27</td><td>77.76</td><td>42.69</td><td>1.25</td><td>22.36</td><td>16.46</td><td>34.68</td><td>37.78</td></tr><tr><td></td><td>STEP@4+PW-S8</td><td>60.25</td><td>71.16</td><td>35.54</td><td>1.12</td><td>16.20</td><td>13.97</td><td>28.47</td><td>32.39</td></tr><tr><td></td><td>GGA@4+PW-S8</td><td>43.23</td><td>53.68</td><td>24.53</td><td>1.05</td><td>11.78</td><td>10.71</td><td>22.24</td><td>23.89</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Naive INT6</td><td>18.16</td><td>18.69</td><td>13.56</td><td>3.79</td><td>13.29</td><td>7.50</td><td>9.30</td><td>12.04</td></tr><tr><td>GGA@6</td><td>STEP@6</td><td>51.83</td><td>64.53</td><td>33.67</td><td>1.04</td><td>14.00</td><td>15.71</td><td>24.67</td><td>29.35</td></tr><tr><td></td><td></td><td>43.74</td><td>57.42</td><td>26.73</td><td>1.02</td><td>11.79</td><td>12.76</td><td>21.90</td><td>25.05</td></tr><tr><td></td><td>STEP@6+PW-S8 GGA@6+PW-S8</td><td>37.03 38.04</td><td>51.47 51.75</td><td>21.98 22.14</td><td>1.05 1.03</td><td>10.18 10.50</td><td>13.76 10.65</td><td>19.01 18.87</td><td>22.07 21.85</td></tr></table>

Table 7: Paired comparisons at similar storage budgets. Mean length weights tasks equally and is measured in thousand tokens; scores are seven-task macro-averages (%). Boldface indicates the higher macro-average and shorter mean output within each allocation method.
<table><tr><td>Model</td><td>Setting</td><td>bit/value</td><td>Macro-avg.</td><td>Mean length</td></tr><tr><td rowspan="4">4B</td><td>STEP@5</td><td>5.677</td><td>60.38</td><td>37.25</td></tr><tr><td>STEP@4+PW-S8</td><td>5.790</td><td>63.67</td><td>35.15</td></tr><tr><td>GGA@5</td><td>5.678</td><td>67.23</td><td>32.27</td></tr><tr><td>GGA@4+PW-S8</td><td>5.784</td><td>69.83</td><td>29.07</td></tr><tr><td rowspan="4">9B</td><td>STEP@5</td><td>5.677</td><td>68.46</td><td>35.03</td></tr><tr><td>STEP@4+PW-S8</td><td>5.786</td><td>70.02</td><td>32.39</td></tr><tr><td>GGA@5</td><td>5.679</td><td>78.22</td><td>26.78</td></tr><tr><td>GGA@4+PW-S8</td><td>5.780</td><td>78.63</td><td>23.89</td></tr></table>

Figures 12 and 13 supplement the main-text stopping-category plots at @6. Tables 8 and 9 report termination statistics for the floating-point references and the @4 and @6 settings, and for the supplementary @5 settings, respectively. Stopping-category percentages are aggregated over responses, while mean lengths retain the equal task weighting used in the main text. Table 10 additionally reports per-task mean lengths for @5 and @4+PW-S8, corresponding to the similar-budget comparison in Section 5.3.

Table 8: Generation lengths and termination statistics for the floating-point references and the @4 and @6 settings. Percentages are aggregated over responses. Mean lengths are in thousands of tokens and are the equally weighted mean of each task’s mean full output length, including thinking and excluding input prompts. EOS, length-cap termination, and repetition stops are mutually exclusive; missing final answers may overlap with them.
<table><tr><td>Model</td><td>Method</td><td>EOS</td><td>100K cap</td><td>Repetition</td><td>No final</td><td>Mean length</td></tr><tr><td>4B</td><td>FP32</td><td>91.81</td><td>4.54</td><td>3.65</td><td>7.54</td><td>24.94</td></tr><tr><td></td><td>FP16</td><td>90.10</td><td>6.58</td><td>3.32</td><td>9.38</td><td>28.72</td></tr><tr><td></td><td>Naive INT4</td><td>100.00</td><td>0.00</td><td>0.00</td><td>11.62</td><td>2.99</td></tr><tr><td></td><td>STEP@4</td><td>53.37</td><td>36.66</td><td>9.97</td><td>45.74</td><td>58.23</td></tr><tr><td></td><td>GGA@4</td><td>79.43</td><td>14.18</td><td>6.38</td><td>19.55</td><td>42.96</td></tr><tr><td></td><td>STEP@4+PW-S8</td><td>91.12</td><td>8.36</td><td>0.53</td><td>8.88</td><td>35.15</td></tr><tr><td></td><td>GGA@4+PW-S8</td><td>94.93</td><td>4.61</td><td>0.46</td><td>5.30</td><td>29.07</td></tr><tr><td></td><td>Naive INT6</td><td>92.27</td><td>0.03</td><td>7.70</td><td>13.62</td><td>20.24</td></tr><tr><td></td><td>STEP@6</td><td>91.05</td><td>6.58</td><td>2.37</td><td>8.75</td><td>32.64</td></tr><tr><td></td><td>GGA@6</td><td>92.66</td><td>4.74</td><td>2.60</td><td>7.17</td><td>29.23</td></tr><tr><td></td><td>STEP@6+PW-S8</td><td>96.97</td><td>1.51</td><td>1.51</td><td>3.29</td><td>22.01</td></tr><tr><td></td><td>GGA@6+PW-S8</td><td>96.25</td><td>2.30</td><td>1.45</td><td>3.82</td><td>24.86</td></tr><tr><td>9B</td><td>FP32</td><td>95.26</td><td>3.06</td><td>1.68</td><td>4.28</td><td>22.68</td></tr><tr><td></td><td>FP16</td><td>94.11</td><td>3.95</td><td>1.94</td><td>5.56</td><td>24.78</td></tr><tr><td></td><td>Naive INT4</td><td>100.00</td><td>0.00</td><td>0.00</td><td></td><td></td></tr><tr><td></td><td>STEP@4</td><td>59.62</td><td>30.67</td><td>9.71</td><td>15.76 39.59</td><td>3.46</td></tr><tr><td></td><td>GGA@4</td><td>90.69</td><td>8.23</td><td>1.09</td><td>8.49</td><td>55.42</td></tr><tr><td></td><td>STEP@4+PW-S8</td><td>91.67</td><td>7.01</td><td>1.32</td><td>8.46</td><td>37.78 32.39</td></tr><tr><td></td><td>GGA@4+PW-S8</td><td>97.47</td><td>2.44</td><td>0.10</td><td>2.83</td><td>23.89</td></tr><tr><td></td><td>Naive INT6</td><td>99.11</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>92.93</td><td>0.00 5.59</td><td>0.89</td><td>11.85</td><td>12.04</td></tr><tr><td></td><td>STEP@6 GGA@6</td><td>96.05</td><td>2.99</td><td>1.48 0.95</td><td>6.81 3.75</td><td>29.35</td></tr><tr><td></td><td>STEP@6+PW-S8</td><td>97.20</td><td>1.84</td><td>0.95</td><td>2.86</td><td>25.05 22.07</td></tr><tr><td></td><td></td><td></td><td>1.68</td><td>0.39</td><td>2.17</td><td>21.85</td></tr><tr><td></td><td>GGA@6+PW-S8</td><td>97.93</td><td></td><td></td><td></td><td></td></tr></table>

Table 9: Generation lengths and termination statistics for the supplementary @5 configurations. Percentages are in %, and mean lengths are in thousands of tokens; the accounting convention is the same as in Table 8.
<table><tr><td>Model</td><td>Method</td><td>EOS</td><td>100K cap</td><td>Repetition</td><td>No final</td><td>Mean length</td></tr><tr><td>4B</td><td>STEP@5</td><td>83.38</td><td>9.64</td><td>6.98</td><td>16.42</td><td>37.25</td></tr><tr><td></td><td>GGA@5</td><td>89.87</td><td>5.03</td><td>5.10</td><td>9.94</td><td>32.27</td></tr><tr><td>9B</td><td>STEP@5</td><td>89.40</td><td>8.03</td><td>2.57</td><td>10.37</td><td>35.03</td></tr><tr><td></td><td>GGA@5</td><td>95.56</td><td>3.29</td><td>1.15</td><td>4.18</td><td>26.78</td></tr></table>

Table 10: Mean full output lengths for each task in the supplementary similar-budget comparison (thousands of tokens). The @5 and @4+PW-S8 settings are paired within each allocation method; the accounting convention is the same as in Table 6.
<table><tr><td>Model</td><td>Method</td><td>AIME</td><td>HMMT</td><td>GPQA</td><td>HE+</td><td>MATH</td><td>IFB</td><td>LCB</td><td>Mean</td></tr><tr><td rowspan="4">4B</td><td>STEP@5</td><td>69.58</td><td>80.97</td><td>41.15</td><td>1.27</td><td>23.02</td><td>18.96</td><td>25.77</td><td>37.25</td></tr><tr><td>STEP@4+PW-S8</td><td>70.31</td><td>81.62</td><td>34.76</td><td>1.44</td><td>19.68</td><td>12.85</td><td>25.38</td><td>35.15</td></tr><tr><td>GGA@5</td><td>60.99</td><td>73.37</td><td>33.32</td><td>1.27</td><td>17.41</td><td>14.99</td><td>24.56</td><td>32.27</td></tr><tr><td>GGA@4+PW-S8</td><td>54.13</td><td>67.48</td><td>30.22</td><td>1.26</td><td>15.06</td><td>11.25</td><td>24.10</td><td>29.07</td></tr><tr><td rowspan="4">9B</td><td>STEP@5</td><td>62.23</td><td>76.95</td><td>40.44</td><td>1.05</td><td>19.44</td><td>14.31</td><td>30.79</td><td>35.03</td></tr><tr><td>STEP@4+PW-S8</td><td>60.25</td><td>71.16</td><td>35.54</td><td>1.12</td><td>16.20</td><td>13.97</td><td>28.47</td><td>32.39</td></tr><tr><td>GGA@5</td><td>46.49</td><td>60.65</td><td>29.06</td><td>1.03</td><td>13.18</td><td>13.14</td><td>23.91</td><td>26.78</td></tr><tr><td>GGA@4+PW-S8</td><td>43.23</td><td>53.68</td><td>24.53</td><td>1.05</td><td>11.78</td><td>10.71</td><td>22.24</td><td>23.89</td></tr></table>

![](images/049373bb471b6c3f2fcd1f77f7a60fde5ab3ce3fde6ee9d66ea9258ce3817614.jpg)  
Figure 12: Stopping categories on Qwen3.5-4B at @6, alongside floating-point state references.

![](images/aa58b33bfe4a4f6e8eb3c768c12983347746465ecd908a4fc244e06b3b0916d5.jpg)  
Figure 13: Stopping categories on Qwen3.5-9B at @6, alongside floating-point state references.