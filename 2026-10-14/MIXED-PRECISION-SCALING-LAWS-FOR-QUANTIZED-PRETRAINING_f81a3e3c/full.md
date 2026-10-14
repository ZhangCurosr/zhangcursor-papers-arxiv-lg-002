# MIXED PRECISION SCALING LAWS FOR QUANTIZED PRETRAINING

Jinuk Kim<sup>1†∗</sup>, Rajarshi Saha<sup>2</sup>, Muhammed Fatih Balın<sup>2</sup>, Youngsuk Park<sup>2†</sup> <sup>1</sup>Seoul National University <sup>2</sup>Amazon Annapurna Labs

## ABSTRACT

Since scaling laws were introduced for modern LLMs (Kaplan et al., 2020), several works have extended them to account for quantization, tracking the emerging low-precision data types supported on AI accelerators. We develop a more accurate scaling law for quantized pretraining with mixed precision across the forward and backward passes, grounded in optimization theory and validated empirically. Our formulation separates the effects of quantization on model capacity and on optimization efficiency, and explicitly models the persistent optimization error from forward and backward quantization identified through convergence analysis. To select the regimes for fitting and evaluating our law, we first decompose existing 4-bit training recipes into their forward and backward components, isolating those that most affect loss. Our Qwen3 experiments under selected 4-bit regimes, together with published Llama-style experiments, show that our law predicts loss with up to 11× lower MSE than existing laws as model size and token count scale. Finally, we extend our law to layer-wise mixed precision and identify the optimal quantized-pretraining recipe, i.e., model size, training tokens, and per-layer forward/backward precision, under a hardware-aware compute budget. Unlike prior precision scaling laws, our law predicts that the optimal precision increases with the compute budget.

## 1 INTRODUCTION

Scaling laws have played a central role in guiding large language model (LLM) pretraining by characterizing how training loss varies with model size, data, and compute (Kaplan et al., 2020; Hoffmann et al., 2022; Videau et al., 2026). By extrapolating from smaller scale experiments, they enable researchers to choose effective training configurations for expensive runs in advance. As pretraining costs continue to grow, accurate scaling laws are becoming increasingly valuable for allocating limited resources and maximizing model quality under a given compute budget.

Quantized training offers a promising path towards more efficient LLM pretraining by performing computations in the forward and backward passes at reduced numerical precision (Micikevicius et al., 2022; Fishman et al., 2025). Recent recipes push precision down to 4 bits using block-scaled formats such as NVFP4, while combining different techniques in the forward and backward passes to preserve training quality (Tseng et al., 2025; Abecassis et al., 2025). Existing methods differ both in how quantization scales are shared and selected during the forward pass and in how gradients are estimated during the backward pass. For example, Abecassis et al. (2025) and Panferov et al. (2026a) adopt different choices along both dimensions. Together, these techniques have substantially narrowed the gap between FP4 and BF16 training, making low-precision training an increasingly practical option for LLM pretraining (Chmiel et al., 2025; Abecassis et al., 2025).

We first revisit existing NVFP4 training recipes by decomposing them into forward and backward components and isolating the contribution of each to pretraining loss, since prior works typically proposed and evaluated these techniques only as complete configurations (Section 2). We find that forward quantization choices affect loss more than backward techniques, but even the best combination leaves a noticeable loss gap relative to BF16 training. Mixed-precision quantized training offers a promising way to narrow this gap, either by assigning different precisions to the forward and backward passes (Castro et al., 2026) or by varying precision across layers (Abecassis et al., 2025). Choosing among such configurations before training requires accurately predicting their loss at the target model size and token budget, which motivates scaling laws for mixed precision both across the forward and backward passes and across layers.

![](images/06524d3d103ee42aec66fff0cdd1cc28981b4227f37ebb89541b551fe27cdace.jpg)

![](images/864420c7d7322e921e4b3e86a7c827d19e178ef8145cf9a3398076c648fce129.jpg)

Optimal model size $N ^ { * }$  
![](images/032e4bb0cea65710bd84ea1135a88c36b1f5d81e568791152a6faae4b30bd632.jpg)  
Compute budget (BF16-equivalent FLOPs)

Tokens per parameter $D ^ { * } / N ^ { * }$  
![](images/e14610856e5f41dc85a15d7f3b046f78f20839b2b4c2563d2b2c454274efcca8.jpg)  
Compute budget (BF16-equivalent FLOPs)  
Figure 1: Compute-optimal precision allocation predicted by our scaling law. With each Transformer block using MXFP8 or NVFP4 in each pass, the panels show the optimal fraction of NVFP4 blocks in the forward pass $p _ { f } ^ { \ast }$ (top, left) and backward pass $p _ { b } ^ { * }$ (top, right), the optimal model size $N ^ { * }$ (bottom, left), and tokens per parameter $D ^ { * } / N ^ { * }$ (bottom, right), given the compute budget (x-axis) and the NVFP4 speedup over BF16 (y-axis), with the MXFP8 speedup fixed at 1.8× (Section 4.3).

For mixed precision across the forward and backward passes, existing scaling laws model forward quantization as a reduction in the effective parameter count (Kumar et al., 2025; Frantar et al., 2025; Panferov et al., 2026b), and the Quartet law additionally models backward quantization as a reduction in the effective training token count, whose penalty can be offset by training on more data (Castro et al., 2026) (Table 2). However, quantization also introduces optimization error that can persist with continued training. In the forward pass, evaluating gradients at quantized weights while updating full-precision weights can induce oscillations that prevent convergence (Bengio et al., 2013; Bai et al., 2019; Defossez et al., 2022; Nagel et al., 2022), and in the backward pass, quantization´ can bias gradient estimates (Ajalloeian & Stich, 2020). Accurate loss prediction therefore requires modeling this persistent error in both passes.

In this work, we develop a scaling law for forward–backward mixed-precision pretraining that accounts for this persistent optimization error (Section 3). Guided by a theoretical analysis of the optimization gap, the law combines effective parameter and data counts with separate additive terms for persistent optimization error from forward and backward quantization. The resulting law (Table 2) takes the form

$$
{ \cal L } = \left( A \left( \rho _ { N } ( \psi _ { f } ) N \right) ^ { - \alpha } + B \left( \rho _ { D } ( \psi _ { b } ) D \right) ^ { - \beta } \right) ^ { \gamma } + \epsilon _ { f } ( \psi _ { f } ) + \epsilon _ { b } ( \psi _ { b } ) + { \cal E } ,
$$

where $\rho _ { N }$ and $\rho _ { D }$ rescale the parameter count $N$ and the training token count $D ,$ , respectively, and $\epsilon _ { f }$ and $\epsilon _ { b }$ model persistent optimization error associated with the forward and backward quantization schemes $\psi _ { f }$ and $\psi _ { b } ,$ respectively. We evaluate the law on our Qwen3 (Yang et al., 2025) pretraining experiments, covering models with up to 0.9B parameters trained on up to 35.8B tokens with BF16,

MXFP8, and different NVFP4 schemes in both passes, and on the published Quartet results for Llama-style models trained with up to 800 tokens per parameter (Castro et al., 2026). Across both settings, the law predicts held-out loss more accurately than existing laws when extrapolating to larger model sizes and token budgets.

For mixed precision across layers, we extend the law to predict the loss when different quantization schemes are assigned to different Transformer blocks (Section 4). The extension interpolates between the predicted losses of the two schemes being mixed, with an exponent that captures the strategy used to select blocks for each scheme (Table 2). Accounting for the hardware speedup of each format, we then use the law to identify the compute-optimal model size, number of training tokens, and precision of the forward and backward passes under a given compute budget (Figure 1).

We summarize our main contributions as follows:

(a) We revisit existing NVFP4 recipes by decomposing them into forward and backward components, identifying which components matter for loss and measuring the remaining gap to BF16, which guides the design of our scaling law experiments (Section 2).

(b) We develop a scaling law that models persistent optimization error from forward and backward quantization under mixed precision across the two passes, backed by theoretical analysis (Table 2). It predicts held-out loss more accurately than existing laws on our Qwen3 experiments and the published Quartet results (Section 3).

(c) We extend the law to layer-wise mixed precision, and identify compute-optimal model sizes and precision assignments under a hardware-aware compute budget (Section 4).

## 2 REVISITING QUANTIZED PRETRAINING RECIPES

Quantized pretraining reduces training cost by quantizing computation in both the forward and backward passes, with recent recipes pushing precision down to FP4 using block-scaled formats such as NVFP4. Since these recipes are typically evaluated as complete configurations, it remains unclear which of their individual forward and backward components matter for pretraining loss. In this section, we decompose two state-of-the-art FP4 recipes, the NVFP4 recipe (Abecassis et al., 2025) and Quartet II (Panferov et al., 2026a), into their forward and backward components to isolate which matter most for pretraining loss, and select the most effective regimes to adopt in our scaling law experiments. In the forward pass, the NVFP4 recipe uses 2D scaling, sharing scales across 16 × 16 weight blocks so that quantized weights remain consistent when transposed in the backward pass,

Table 1: Validation loss of FP4 pretraining recipes decomposed into forward and backward components, for a 430M parameter Qwen3 model trained at 12.8B tokens. Loss increase is relative to the BF16 baseline. The last column indicates whether each combination is included in our scaling law grid.
<table><tr><td>Forward</td><td>Backward</td><td>Val. loss ↓</td><td>Increase ↓</td><td>In grid</td></tr><tr><td rowspan="4">BF16</td><td>BF16</td><td>3.245</td><td>一</td><td>√</td></tr><tr><td>MS-EDEN</td><td>3.257</td><td>+0.36%</td><td>√</td></tr><tr><td>RHT + SR</td><td>3.255</td><td>+0.32%</td><td>V</td></tr><tr><td>SR</td><td>3.268</td><td>+0.72%</td><td>一</td></tr><tr><td rowspan="4">4/6 + 1D scaling</td><td>BF16</td><td>3.275</td><td>+0.94%</td><td>√</td></tr><tr><td>MS-EDEN</td><td>3.280</td><td>+1.07%</td><td>√</td></tr><tr><td>RHT + SR</td><td>3.282</td><td>+1.14%</td><td>√</td></tr><tr><td>SR</td><td>3.291</td><td>+1.42%</td><td>√</td></tr><tr><td rowspan="3">2D scaling</td><td>BF16</td><td>3.284</td><td>+1.22%</td><td>X</td></tr><tr><td>RHT + SR</td><td>3.291</td><td>+1.41%</td><td>x</td></tr><tr><td>SR</td><td>3.300</td><td>+1.69%</td><td>X</td></tr></table>

whereas Quartet II uses 1D scaling with 1 × 16 blocks and 4/6 scaling, which selects the lowererror of two candidate scales for each block (Cook et al., 2025). In the backward pass, the NVFP4 recipe applies stochastic rounding (SR) to the output activation gradients and randomized Hadamard transforms (RHT) to the operands of the weight gradient computation, similar to the MXFP4 recipe (Tseng et al., 2025), whereas Quartet II uses MS-EDEN, which combines randomized rotations with EDEN-based rescaling to obtain unbiased matrix multiplication.

Results Table 1 reports the validation loss of a Qwen3 model (Yang et al., 2025) with 430M non-embedding parameters trained on 12.8B tokens for each combination of forward and backward components. The results show that forward pass quantization is generally more sensitive than backward pass quantization. From the BF16 baseline, quantizing only the backward pass to NVFP4 increases the loss by 0.32–0.72%, whereas quantizing only the forward pass increases it by 0.94–

Table 2: Comparison of scaling laws for quantization and compression. The last three columns indicate whether each law models different schemes in the forward and backward passes, different schemes across layers, and persistent optimization error. $\rho _ { N }$ and $\rho _ { D }$ rescale $N$ and $D$ according to the forward and backward schemes $\psi _ { f }$ and $\psi _ { b } .$ , and $\epsilon _ { f }$ and $\epsilon _ { b }$ model persistent optimization error independent of N and D. For layer-wise mixed precision, $p \in [ 0 , 1 ]$ is the fraction of layers assigned to the second of two schemes with losses $L _ { 1 }$ and $L _ { 2 } .$ , and $\gamma _ { S }$ depends on the layer selection strategy S. PTQ, QAT, and QPT denote post-training quantization, quantization-aware training, and quantized pretraining, respectively. Other notation is described in Appendix I.
<table><tr><td>Scaling law</td><td>Loss formulation</td><td>Fwd-bwd mixed</td><td>Layer mixed</td><td>Optim. error ∈(·)</td></tr><tr><td>Precision (PTQ) (Kumar et al., 2025)</td><td> $L = A N ^ { - \alpha } + B D ^ { - \beta } + E + C _ { T } D ^ { \gamma _ { D } } N ^ { - \gamma _ { N } } e ^ { - P _ { \mathrm { p o s t } } / \gamma _ { \mathrm { p o s t } } }$ </td><td>X</td><td>x</td><td>X</td></tr><tr><td>Mixed (PTQ) (Cao et al., 2024)</td><td> $L = A N ^ { - \alpha } + B D ^ { - \beta } + E + C e ^ { \kappa p } N ^ { - \gamma _ { N } } ( G + d ) ^ { \gamma _ { G } }$ </td><td>X</td><td>√</td><td>X</td></tr><tr><td>Precision (QPT) (Kumar et al., 2025)</td><td> $L = A \left( ( 1 - e ^ { - P / \bar { \gamma } } ) ^ { 3 } N \right) ^ { - \alpha } + B D ^ { - \beta } + E$ </td><td>X</td><td>X</td><td>X</td></tr><tr><td>Compression (Frantar et al., 2025)</td><td> $L = A \left( \rho _ { N } ( \psi _ { f } ) N \right) ^ { - \alpha } + B D ^ { - \beta } + E$ </td><td>X</td><td>X</td><td>X</td></tr><tr><td>QAT (Chen et al., 2025)</td><td> $L = A N ^ { - \alpha } + B D ^ { - \beta } + E + k D ^ { \gamma _ { D } } N ^ { - \gamma _ { N } } ( \log _ { 2 } G ) ^ { \gamma _ { G } }$ </td><td>x</td><td>x</td><td>x</td></tr><tr><td>Representations (Panferov et al., 2026b)</td><td> $L = A \left( N \widetilde { \rho } ( \mathrm { G M S E } ( \psi _ { f } ) ) \right) ^ { - \alpha } + B D ^ { - \beta } + E$ </td><td>x</td><td>x</td><td>x</td></tr><tr><td>Quartet (Castro et al., 2026)</td><td> $L = \left( A \left( \rho _ { N } ( \psi _ { f } ) N \right) ^ { - \alpha } + B \left( \rho _ { D } ( \psi _ { b } ) D \right) ^ { - \beta } \right) ^ { \gamma } + E$ </td><td>√</td><td>X</td><td>X</td></tr><tr><td>Mixed (QPT) (Ours)</td><td> $\begin{array} { r } { L = \left( A \left( \rho _ { N } ( \psi _ { f } ) N \right) ^ { - \alpha } + B \left( \rho _ { D } ( \psi _ { b } ) D \right) ^ { - \beta } \right) ^ { \gamma } } \\ { + \epsilon _ { f } ( \psi _ { f } ) + \epsilon _ { b } ( \psi _ { b } ) + E , \qquad } \end{array}$  (Section 3) (Section 4)  $L = \left( L _ { 2 } - L _ { 1 } \right) p ^ { \gamma _ { S } } + L _ { 1 }$ </td><td></td><td>1</td><td>1</td></tr></table>

1.22%. Similarly, the advantage of Quartet II (4/6 + 1D scaling, MS-EDEN) over the NVFP4 recipe (2D scaling, RHT + SR) comes largely from the forward pass. Starting from Quartet II, replacing MS-EDEN with RHT + SR increases the loss by only 0.07 percentage points, whereas subsequently replacing 4/6 + 1D scaling with 2D scaling while keeping RHT + SR in the backward pass increases it by a further 0.27 percentage points. Based on these results, we adopt 4/6 + 1D scaling for NVFP4 forward quantization rather than 2D scaling, which performs worse overall, and retain all three backward techniques in our scaling law experiments. These selected regimes, together with BF16 and MXFP8 in both passes, form our experimental grid (Table 3).

Notably, even with the best FP4 recipe (Quartet II), a noticeable loss gap remains relative to the BF16 baseline, suggesting that uniformly quantizing to 4-bit may not be preferable. Mixed precision offers a way to narrow this gap by assigning different precisions to the forward and backward passes (Castro et al., 2026) or by varying precision across layers (Abecassis et al., 2025; Li & humans& ai, 2026). Accurately predicting the quantized pretraining loss of these configurations can help practitioners choose ones that meet their quality and compute targets, motivating quantized pretraining scaling laws for mixed precision across the forward and backward passes and across layers.

To this end, we develop a scaling law for mixed precision across the forward and backward passes in Section 3, guided by theoretical analysis, that predicts quantized pretraining loss more accurately than existing laws. In Section 4, we extend this law to layer-wise mixed precision and use it to identify compute-optimal configurations. We evaluate these laws on the selected regimes, and additionally validate the law in Section 3 on the published Quartet grid (Castro et al., 2026).

## 3 FORWARD–BACKWARD MIXED-PRECISION SCALING LAW WITH OPTIMIZATION ERROR

We first consider mixed precision across the forward and backward passes, where the two passes use quantization schemes $\psi _ { f }$ and $\psi _ { b }$ that may differ, as in the Quartet law (Castro et al., 2026), with uniform precision as the special case of matching schemes. Existing scaling laws for quantized pretraining treat forward quantization as reducing the effective parameter count (Kumar et al., 2025; Frantar et al., 2025; Panferov et al., 2026b) and backward quantization as reducing the effective training token count (Castro et al., 2026), implying that quantization penalties can be offset by scaling model size or data (Table 2). However, quantized training can also introduce optimization error that does not decrease with more data or parameters, as weights can oscillate across rounding boundaries in the forward pass (Defossez et al., 2022; Nagel et al., 2022) and gradient estimates can´ be biased in the backward pass (Ajalloeian & Stich, 2020).

We develop a scaling law that accounts for this persistent optimization error,

$$
{ \cal L } = \left( A \left( \rho _ { N } ( \psi _ { f } ) N \right) ^ { - \alpha } + B \left( \rho _ { D } ( \psi _ { b } ) D \right) ^ { - \beta } \right) ^ { \gamma } + \epsilon _ { f } ( \psi _ { f } ) + \epsilon _ { b } ( \psi _ { b } ) + { \cal E } ,\tag{1}
$$

where $\rho _ { N }$ and $\rho _ { D }$ rescale N and D according to the forward and backward schemes $\psi _ { f }$ and $\psi _ { b } ,$ , and $\epsilon _ { f } , \epsilon _ { b } \ge 0$ capture persistent optimization error introduced by each pass. Section 3.1 formalizes this intuition using optimization theory, and Sections 3.2 and 3.3 empirically validate that the resulting law predicts held-out loss more accurately than existing laws.

## 3.1 SCALING LAW FORMULATION

Quantization during pretraining affects both the representational capacity of the model and the accuracy of its training updates. To separate these effects, we decompose the loss gap into a capacity gap and an optimization gap (Su, 2026), which guides the formulation of our law. For a language model with parameters $\boldsymbol { \theta } ^ { \setminus } \in \mathbb { R } ^ { N }$ , let ${ \mathcal { L } } _ { N } ( \theta )$ denote the population loss. For a forward quantization scheme $\psi _ { f }$ , let $Q ( : ; \psi _ { f } ) : \mathbb { R } ^ { N } \to \mathbb { R } ^ { N }$ denote the corresponding quantization map. We define the optimal loss of the quantized model and the irreducible loss over the model family, respectively, as

$$
\mathcal L _ { N , \psi _ { f } } ^ { * } : = \operatorname* { i n f } _ { \theta \in \mathbb { R } ^ { N } } \mathcal L _ { N } ( Q ( \theta ; \psi _ { f } ) ) , \qquad E : = \operatorname* { i n f } _ { N } \operatorname* { i n f } _ { \theta \in \mathbb { R } ^ { N } } \mathcal L _ { N } ( \theta ) \qquad ( \mathcal L _ { N , \psi _ { f } } ^ { * } \geq E ) .
$$

Consider training for $T$ steps, with weights $\theta _ { t }$ and quantized weights $\hat { \theta } _ { t } = Q ( \theta _ { t } ; \psi _ { f } )$ at step $t \in [ T ]$ The gap between the average expected loss of the quantized model and E decomposes as

$$
\begin{array} { r } { \mathbb { E } _ { J , \theta , \boldsymbol { J } } \Big [ \mathcal { L } _ { N } ( \hat { \theta } _ { J } ) \Big ] - E = \underbrace { \Big ( \mathcal { L } _ { N , \psi _ { f } } ^ { * } - E \Big ) } _ { \mathrm { c a p a c i t y ~ g a p } } + \underbrace { \Big ( \mathbb { E } _ { J , \theta _ { J } } \Big [ \mathcal { L } _ { N } ( \hat { \theta } _ { J } ) \Big ] - \mathcal { L } _ { N , \psi _ { f } } ^ { * } \Big ) } _ { \mathrm { o p i m i z a t i o n ~ g a p } } , } \end{array}
$$

where $J$ is sampled uniformly from $[ T ]$ . The first term represents the model capacity gap due to the limited number of parameters and the restricted numerical representations imposed by forward quantization. The second term represents the optimization gap, measuring how far the average expected loss during training remains above the best loss achievable by the quantized model.

Capacity gap. The capacity gap $\mathcal { L } _ { N , \psi _ { f } } ^ { * } - E$ depends only on model size N and the forward quantization $\psi _ { f }$ . The backward quantization ψ<sub>b</sub> only affects training updates without changing the set of functions the model can represent. We therefore model it through the effective parameter count $\rho _ { N } ( \psi _ { f } ) N$ , as in existing laws (Kumar et al., 2025; Castro et al., 2026; Panferov et al., 2026b).

Optimization gap. To examine how quantization affects optimization, Remark 3.1 below bounds the optimization gap for training with quantized weights and biased backward gradients under suitable regularity conditions. Similar analyses appear in Li et al. (2017); Panferov et al. (2026b) for quantized weights and Ajalloeian & Stich (2020) for biased gradients.

Remark 3.1. Consider updates $\theta _ { t + 1 } = \theta _ { t } - \eta \hat { g } _ { t }$ , where $\hat { \boldsymbol g } _ { t } \in \mathbb R ^ { N }$ is a minibatch gradient estimate at $\hat { \theta } _ { t } = Q ( \theta _ { t } ; \psi _ { f } )$ using backward quantization scheme $\psi _ { b }$ . Define $b _ { t } : = \mathbb { E } [ \hat { g } _ { t } \mid \theta _ { t } ] - \nabla \mathcal { L } _ { N } ( \hat { \theta } _ { t } )$ . For a minimizer $q ^ { * }$ of $\mathcal { L } _ { N }$ over $Q ( \mathbb { R } ^ { N } ; \psi _ { f } )$ , set $R _ { 1 } ( N , \psi _ { f } ) : = \mathbb { E } \| \theta _ { 1 } - q ^ { * } \| ^ { 2 }$ . Assume $\psi _ { f }$ is deterministic, $\mathcal { L } _ { N }$ is differentiable and convex, and, uniformly over $t , T , \mathbb { E } \| \nabla \mathcal { L } _ { N } ( \hat { \theta } _ { t } ) \| ^ { 2 } \leq G ^ { 2 }$ and $\hat { g } _ { t } , b _ { t } , \hat { \theta } _ { t } - \theta _ { t }$ ， and $\theta _ { t } - q ^ { * }$ have bounded second moments. For learning rate chosen as $\eta = c / \sqrt { T } \ ( c > 0 )$ ,

$$
\begin{array} { r l } { \mathbb { E } _ { J , \theta _ { J } } [ \mathcal { L } _ { N } ( \hat { \theta } _ { J } ) ] - \mathcal { L } _ { N , \psi _ { f } } ^ { * } \le \underbrace { \cfrac { 1 } { 2 \sqrt { T } } \left[ \cfrac { R _ { 1 } ( N , \psi _ { f } ) } { c } + \cfrac { c } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \| \hat { g } _ { t } \| ^ { 2 } \right] } _ { \mathrm { ( a ) } O ( T ^ { - 1 / 2 } ) } } & { } \\ { + \underbrace { G \left( \cfrac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \| \hat { \theta } _ { t } - \theta _ { t } \| ^ { 2 } \right) ^ { 1 / 2 } } _ { \mathrm { ( b ) } O ( 1 ) } + \underbrace { \cfrac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } [ \| b _ { t } \| \| \theta _ { t } - q ^ { * } \| ] } _ { \mathrm { ( c ) } O ( 1 ) } . } \end{array}
$$

Table 3: Overview of the experimental data grids used to evaluate scaling laws for quantized pretraining. Model sizes N are non-embedding parameter counts. TPP denotes tokens per parameter $( D / N )$ , where D is the training token count. Token budgets show the range covered by each grid, with detailed settings in Appendix B.2. Each grid covers all combinations of the listed forward and backward schemes at every evaluated model size and token budget, including the BF16 baseline.
<table><tr><td>Data grid (architecture)</td><td># points</td><td>Model sizes N</td><td>Token budgets</td><td>Forward schemes ψ f</td><td>Backward schemes ψb</td></tr><tr><td>Quartet grid (Llama)</td><td>84</td><td>30M, 50M, 100M,200M</td><td>25–800 TPP</td><td>BF16, MXFP4 (QuEST)</td><td>BF16, MXFP4 (SR)</td></tr><tr><td>Our grid (Qwen3)</td><td>375</td><td>280M, 430M, 640M, 910M</td><td>12.8B-35.8B</td><td>BF16, MXFP8,  $\mathrm { N V F P 4 } ~ ( 4 / 6 + \mathrm { 1 D } )$ </td><td>BF16, MXFP8, NVFP4 (MS-EDEN), NVFP4 (RHT + SR), NVFP4 (SR)</td></tr></table>

Each term in the bound corresponds to a term in the scaling law $( \mathrm { E q . 1 } )$

(a) The first term decreases as $O ( T ^ { - 1 / 2 } )$ with additional optimization steps, which correspond to training tokens at a fixed batch size. It motivates the effective data size $\rho _ { D } ( \psi _ { b } ) D$

(b) The second term depends on the forward quantization error $\lVert \hat { { \boldsymbol { \theta } } } _ { t } - { \boldsymbol { \theta } } _ { t } \rVert$ , which need not vanish with additional training. Example 3.1 shows that forward quantization can cause an Ω(1) optimization gap, motivating the persistent forward error term $\epsilon _ { f } ( \psi _ { f } )$ . Related oscillations and illustrative examples have been studied by Defossez et al. (2022); Nagel et al. (2022).´

(c) The third term reflects bias $b _ { t }$ in backward gradient estimates, which can persist despite additional training (Ajalloeian & Stich, 2020), motivating the backward error term $\epsilon _ { b } ( \psi _ { b } )$

Example 3.1 (Persistent optimization error from forward quantization). Consider $\begin{array} { r } { \mathcal { L } _ { 1 } ( \theta ) = ( \theta - \frac { 1 } { 4 } ) ^ { 2 } } \end{array}$ and $\begin{array} { r } { Q ( \theta ; \psi _ { f } ) = \lfloor \theta + \frac { 1 } { 2 } \rfloor } \end{array}$ . For T divisible by four, initialize $\textstyle \theta _ { 1 } = { \frac { 1 } { 2 } }$ and choose $\eta = c / \sqrt { T }$ with $\ 0 \ < \ c \ \leq \ \ \frac { 1 } { 4 }$ . Using exact gradients at the quantized weights $\hat { \theta } _ { t } = Q ( \theta _ { t } ; \psi _ { f } )$ gives the updates $\begin{array} { r } { \theta _ { t + 1 } = \theta _ { t } - 2 \eta ( \hat { \theta } _ { t } - \frac { 1 } { 4 } ) } \end{array}$ . The quantized iterates repeat the cycle $( 1 , 0 , 0 , 0 )$ . The optimal quantized weight is $q ^ { * } = 0$ , with $\begin{array} { r } { \mathcal { L } _ { 1 , \psi _ { f } } ^ { * } = \frac { 1 } { 1 6 } } \end{array}$ . The optimization gap is

$$
\mathbb { E } _ { J } [ \mathcal { L } _ { 1 } ( \hat { \theta } _ { J } ) ] - \mathcal { L } _ { 1 , \psi _ { f } } ^ { * } = \frac { 1 } { 4 } \cdot \frac { 9 } { 1 6 } + \frac { 3 } { 4 } \cdot \frac { 1 } { 1 6 } - \frac { 1 } { 1 6 } = \frac { 1 } { 8 } = \Omega ( 1 ) .
$$

Thus, the optimization gap does not vanish as $T \to \infty$

Proposed scaling law. Combining the capacity gap, modeled through $\rho _ { N } ( \psi _ { f } ) N$ , with terms (a), (b), and (c) yields our proposed scaling law (Eq. 1), where the persistent contributions are modeled as separate additive terms. While term (a) depends on both quantization schemes, we retain only the backward dependence in $\rho _ { D }$ because including $\psi _ { f }$ did not improve held-out loss prediction (Section 3.3). Table 2 highlights the differences between our law and existing formulations.

## 3.2 EXPERIMENTAL SETUP

We evaluate our scaling law on two pretraining grids: (1) the published Quartet experiments with Llama models (Castro et al., 2026) and (2) our Qwen3 experiments. For each grid, we fit the laws on calibration data and evaluate predictions on held-out runs with larger model sizes or token budgets.

Data grids. Table 3 details the two grids, the Quartet grid with 84 published runs of Llama-style models (Touvron et al., 2023) over two forward and two backward schemes (Castro et al., 2026; Panferov et al., 2025), and our grid with 375 runs of Qwen3-based models (Yang et al., 2025) over three forward and five backward schemes selected in Section 2, covering every forward–backward combination (see Appendix B.1 for details).

Evaluation protocol. We compare our law with the Quartet law (Castro et al., 2026) under model size extrapolation, fitting on the three smaller models and evaluating on the largest, and data extrapolation, fitting on smaller token budgets and evaluating at the largest (Appendix B.2). Following Castro et al. (2026), we fit each law in two stages, first to BF16 runs and then, with the baseline coefficients frozen, to quantized runs (Appendix B.3). We report $R ^ { 2 }$ , mean squared error (MSE), and mean absolute percentage error (MAPE) on held-out quantized runs.

Table 4: Held-out loss prediction under model size and data extrapolation on the Quartet grid (Llama) and our grid (Qwen3). Each law is fitted to the calibration split of each setting and evaluated on the quantized runs in the held-out split. The best results are shown in bold.
<table><tr><td rowspan="2">Evaluation</td><td rowspan="2">Scaling law</td><td colspan="3">Quartet grid (Llama)</td><td colspan="3">Our grid (Qwen3)</td></tr><tr><td> $R ^ { 2 } \uparrow$ </td><td>MSE↓</td><td>MAPE (%) ↓</td><td> $R ^ { 2 } \uparrow$ </td><td>MSE↓</td><td>MAPE (%) ↓</td></tr><tr><td rowspan="2">Model size extrapolation</td><td>Quartet</td><td>0.964</td><td>1.6e-4</td><td>0.386</td><td>0.990</td><td> ${ 8 . 4 \mathrm { e } { - 6 } }$ </td><td>0.076</td></tr><tr><td>Ours</td><td>0.997</td><td>1.4e-5</td><td>0.096</td><td>0.994</td><td>5.3e-6</td><td>0.059</td></tr><tr><td rowspan="2">Data extrapolation</td><td>Quartet</td><td>0.995</td><td>9.0e-5</td><td>0.234</td><td>0.992</td><td> $2 . 7 \mathrm { e } { \cdot } 5$ </td><td>0.121</td></tr><tr><td>Ours</td><td>0.997</td><td>5.1e-5</td><td>0.217</td><td>0.998</td><td>6.1e-6</td><td>0.064</td></tr></table>

![](images/4eee307ed70cbc913d316a65fd8b37d4592ec5f4abb77aeace89f18dd1df7cb8.jpg)  
Figure 2: Scaling law fits on 30M–100M models (Quartet grid, MXFP4 QuEST forward, MXFP4 stochastic rounding backward).

![](images/d8cd10ebffada298ab2fa50e043ce488404380f68e9b1f9683a3e959d988ffd2.jpg)  
Figure 3: Extrapolation to held-out 200M models (Quartet grid, MXFP4 QuEST forward, MXFP4 stochastic rounding backward).

## 3.3 EXPERIMENTAL RESULTS

Table 4 reports model size and data extrapolation results on both grids. Our law outperforms the Quartet law across all settings, reducing MSE by $1 . 6 \times \mathrm { ~ t o ~ } 1 1 \times , \mathrm { e . g . }$ , from $1 . 6 \times 1 0 ^ { - 4 } \mathrm { t o } 1 . 4 \times 1 0 ^ { - 5 }$ (11×) for model size extrapolation on the Quartet grid and from $2 . 7 \times 1 0 ^ { - 5 } ~ { \mathrm { t o } } 6 . 1 \times 1 0 ^ { - 6 } \left( 4 . 4 \times \right)$ for data extrapolation on our grid. Figures 2 and 3 illustrate this on the Quartet grid with MXFP4 QuEST in the forward pass and MXFP4 stochastic rounding in the backward pass, showing the fits on 30M– 100M models and the extrapolation to held-out 200M models. Our predictions match more closely the measured losses of the 200M models across token budgets, showing that modeling persistent optimization error improves extrapolation to larger models and token budgets. Appendix E further provides a comprehensive comparison of both laws across all quantization scheme combinations, grids, and extrapolation settings (Figures 5–10).

Ablation results. We further ablate the design choices in Eq. 1 by evaluating three variants, (1) allowing $\rho _ { D }$ to depend on both forward and backward quantization schemes, (2) removing the backward error term $\epsilon _ { b } ( \psi _ { b } )$ , and (3) removing the forward error term $\epsilon _ { f } ( \psi _ { f } )$ . Table 5 reports the geometric mean of held-out MSE across model size and data extrapolation on the Quartet grid and our grid, giving equal weight to relative changes across settings with different error scales. The proposed law achieves the lowest error among the evaluated variants, supporting the chosen formulation. Appendix C further evaluates all combinations of these choices, together with adding the error terms inside the outer exponent $\gamma$ instead of outside, and reports the held-out MSE in each setting (Table 9).

Table 5: Ablations of our scaling law, reporting geometric-mean held-out MSE across model size and data extrapolation on the Quartet and Qwen3 grids. Variants modify our law as indicated. Best result is bolded.
<table><tr><td>Scaling law form</td><td>Geo. mean MSE ↓</td></tr><tr><td>Quartet</td><td>4.27e-5</td></tr><tr><td>Ours</td><td>1.23e-5</td></tr><tr><td>- (1) With  $\rho _ { D } ( \psi _ { f } , \psi _ { b } )$ </td><td>1.29e-5</td></tr><tr><td>- (2) Without  $\epsilon _ { b } ( \psi _ { b } )$ </td><td>1.34e-5</td></tr><tr><td>- (3) Without  $\epsilon _ { f } ( \psi _ { f } )$ </td><td>3.76e-5</td></tr></table>

## 4 LAYER-WISE MIXED-PRECISION SCALING LAW AND COMPUTE-OPTIMAL ALLOCATION

Practical quantized pretraining recipes often retain higher precision in sensitive layers while using lower-precision computation elsewhere, enabling a flexible trade-off between training loss and cost (Abecassis et al., 2025; Li & humans& ai, 2026; Pan et al., 2026). Selecting such configurations before training requires predicting how loss depends not only on the fraction of high-precision blocks, but also on which blocks are assigned higher precision. In this section, we extend $\operatorname { E q . }$ 1 to model both the fraction of high-precision blocks and how they are selected (Table 2), and use the extended law to identify compute-optimal precision allocations. Section 4.1 formulates the extension, Section 4.2 evaluates its predictions, and Section 4.3 studies compute-optimal precision allocation under a fixed compute budget, showing that the optimal precision increases with the budget.

## 4.1 FORMULATION

Consider a model with K Transformer blocks and two quantization configurations: high precision $\psi _ { 1 }$ and low precision $\psi _ { 2 } ,$ , each comprising forward and backward schemes, $\psi _ { i } = ( \psi _ { f , i } , \psi _ { b , i } )$ for $i \in \{ 1 , 2 \}$ . We interpolate between the two configurations by assigning $\psi _ { 2 }$ to a subset of the blocks and $\psi _ { 1 } ~ \mathrm { t o }$ the rest. A selection strategy $S : [ \bar { 0 , } 1 ]  2 ^ { [ \bar { K } ] }$ maps a fraction p to a set of blocks $S ( p ) \subseteq [ K ]$ with $| S ( p ) | =$ round(pK), and the mixed configuration $\psi _ { S } ^ { \mathrm { m i x } } ( \bar { p } )$ assigns $\psi _ { 2 }$ to the blocks in $S ( p )$ and $\psi _ { 1 }$ to the remaining blocks. We model its loss as

$$
L \left( N , D , \psi _ { S } ^ { \operatorname * { m i x } } ( p ) \right) = \left( L _ { 2 } - L _ { 1 } \right) p ^ { \gamma _ { S } } + L _ { 1 } ,\tag{2}
$$

where $L _ { i } = L ( N , D , \psi _ { f , i } , \psi _ { b , i } )$ is predicted by Eq. 1, $L _ { 2 } \ge L _ { 1 }$ , and $\gamma _ { S } > 0$ depends on the strategy S. Mixing only the forward or only the backward pass corresponds to $\psi _ { b , 1 } = \psi _ { b , 2 } \mathrm { o r } \psi _ { f , 1 } = \psi _ { f , 2 }$

By construction, the loss equals $L _ { 1 }$ at $p = 0$ and $L _ { 2 } \operatorname { a t } p = 1$ , so the strategy only determines how the loss moves between these endpoints. A good strategy assigns $\psi _ { 2 }$ to the least sensitive blocks first, keeping the loss close to $L _ { 1 }$ until $p$ becomes large, whereas a poor strategy assigns it to sensitive blocks early and increases the loss quickly. The exponent $\gamma _ { S }$ captures this difference, where a larger $\gamma _ { S }$ indicates that the strategy keeps the sensitive blocks in $\psi _ { 1 }$ longer, a smaller $\gamma _ { S }$ indicates that it assigns them to ψ<sub>2</sub> earlier, and $\gamma _ { S } = 1$ corresponds to linear interpolation between $L _ { 1 }$ and $L _ { 2 }$

## 4.2 EXPERIMENTS

We evaluate the extended law on Qwen3 models with layer-wise MXFP8/NVFP4 mixed precision. We vary precision in either the forward or backward pass, keeping 25%, 50%, or 75% of the blocks in MXFP8 $( p \in \{ 0 . 7 5 , 0 . 5 , 0 . 2 5 \}$ up to rounding). We compare two strategies for selecting higherprecision blocks: (1) first/last, which keeps the first and last blocks in MXFP8, following practical recipes (Abecassis et al., 2025; Li & humans& ai, 2026), and (2) middle, which keeps the central blocks in MXFP8. We hold out the 910M model, predict endpoints $L _ { 1 }$ and $L _ { 2 }$ using laws fitted on smaller models, and fit $\gamma _ { S }$ for each pass and strategy using their mixed-precision runs (see $\mathsf { A p - }$ pendix B.4 for details).

Prediction accuracy. Figure 4 compares the predicted and observed losses of the mixed-precision runs on the held-out 910M model. With endpoints predicted by our law, the extended law closely matches the observed losses across both passes, both strategies, all fractions, and all token budgets $( R ^ { 2 } =$ 0.991, $\mathrm { M S E } = 7 . 1 \times 1 0 ^ { - 6 } )$ , indicating that $\gamma _ { S }$ fitted on smaller models transfers to a larger model. With endpoints predicted by the Quartet law, the predictions consistently underestimate the observed losses, increasing the MSE by nearly three times $( R ^ { 2 } = 0 . 9 7 4 , { \mathrm { M S E } } = 2 . 0 \times 1 0 ^ { - 5 } )$ . This shows that the improved accuracy of our proposed scaling law in Section 3 carries over to the layer-wise mixed-precision setting. Appendix F further visualizes the fits of Eq. 2 for each pass and strategy in terms of the normalized increase in loss (Figure 11).

Table 6: Fitted $\gamma _ { S }$ (higher means better strategy) with our law, and the resulting fraction of the loss gap incurred at $p = 0 . 5$
<table><tr><td>Pass</td><td>Strategy  $S$ </td><td> $\gamma _ { S } \uparrow$  Gap at  $p = 0 . 5 \downarrow$ </td></tr><tr><td rowspan="2">Forward</td><td>First/last</td><td>1.10 46.7%</td></tr><tr><td>Middle</td><td>1.00 50.1%</td></tr><tr><td rowspan="2">Backward</td><td>First/last</td><td>1.60 33.0%</td></tr><tr><td>Middle</td><td>0.50 70.7%</td></tr></table>

![](images/b524f75cd8896a218e437eb22a9c9b7829346a5e51f00a13f253601438cc83f6.jpg)  
Figure 4: Predicted versus observed loss of layer-wise mixed-precision runs on the held-out 910M model, with $\gamma _ { S }$ fitted on the three smaller models. The endpoints $L _ { 1 }$ and $L _ { 2 }$ in Eq. 2 are predicted by our law (left) or the Quartet law (right). Markers indicate the mixed pass and the blocks kept in higher precision, and each point corresponds to one fraction p and one token budget.

First and last blocks matter more in the backward pass. Table 6 reports the fitted $\gamma _ { S } ,$ , where larger values indicate that a strategy retains more sensitive blocks at higher precision and incurs a smaller fraction of the loss gap. Keeping the first and last blocks in higher precision, as proposed in practical NVFP4 training recipes (Abecassis et al., 2025; Li & humans& ai, 2026), is particularly effective in the backward pass, as reflected in its higher fitted $\gamma _ { S }$ . In the forward pass, both strategies incur about half of the gap at $p = 0 . 5$ (46.7% and 50.1%), so placement has little effect. In the backward pass, keeping the first and last blocks in MXFP8 incurs 33.0% of the gap, compared with 70.7% for the middle blocks.

## 4.3 COMPUTE-OPTIMAL PRECISION ALLOCATION

Using the scaling laws from Sections 3 and 4.1, we identify the compute-optimal forward, backward and layer-wise precision, along with model size and the number of training tokens, under a given compute budget (Figure 1). Unlike prior precision scaling laws (Kumar et al., 2025, Section 4.3.2), our law predicts that the optimal precision increases with the compute budget.

Setup. We consider the setting of mixing (1) MXFP8 and (2) NVFP4 with $4 / 6 + 1 \mathrm { D }$ scaling in the forward pass, and (1) MXFP8 and (2) NVFP4 with MS-EDEN in the backward pass, assigning NVFP4 to fractions $p _ { f } \in [ 0 , 1 ]$ and $p _ { b } \in [ 0 , 1 ]$ of the blocks, respectively, with the first/last strategy. Applying Eq. 2 to each pass in turn, we predict the loss as

$$
L = ( 1 - u ) ( 1 - v ) L _ { \mathrm { M X B , M X B } } + u ( 1 - v ) L _ { \mathrm { N V 4 , M X B } } + ( 1 - u ) v L _ { \mathrm { M X B , N V 4 } } + u v L _ { \mathrm { N V 4 , N V 4 } } ,\tag{3}
$$

where $u = p _ { f } ^ { \gamma _ { f } } , v = p _ { b } ^ { \gamma _ { b } } , L _ { f , b }$ is the loss predicted by Eq. 1 with forward format f and backward format $b ,$ and $\gamma _ { f }$ and $\gamma _ { b }$ are the first/last exponents in Table 6. We assume that each exponent is independent of the NVFP4 backward technique and of the format in the other pass, which makes Eq. 3 independent of the order in which the passes are mixed.

Cost model and optimal allocation. The forward and backward passes account for roughly one third and two thirds of the $6 N D$ training FLOPs, respectively (Kaplan et al., 2020). Let $\lambda _ { f , \mathrm { M X } 8 }$ $\lambda _ { f , \mathrm { N V 4 } } , \lambda _ { b , \mathrm { M X 8 } }$ , and $\lambda _ { b , \mathrm { N V 4 } }$ denote the hardware-dependent speedups of MXFP8 and NVFP4 quantization recipes over BF16 in each pass. Assuming equal cost across blocks and ignoring operations other than matrix multiplications, the training cost in BF16-equivalent FLOPs is

$$
C ( N , D , p _ { f } , p _ { b } ) = 2 N D \left( \frac { 1 - p _ { f } } { \lambda _ { f , \mathrm { M X B } } } + \frac { p _ { f } } { \lambda _ { f , \mathrm { N V 4 } } } \right) + 4 N D \left( \frac { 1 - p _ { b } } { \lambda _ { b , \mathrm { M X B } } } + \frac { p _ { b } } { \lambda _ { b , \mathrm { N V 4 } } } \right) .\tag{4}
$$

Given a compute budget $C _ { 0 } .$ , the compute-optimal allocation solves

$$
\operatorname* { m i n } _ { N , D , p _ { f } , p _ { b } } \quad L ( N , D , p _ { f } , p _ { b } ) \quad { \mathrm { s u b j e c t ~ t o } } \quad C ( N , D , p _ { f } , p _ { b } ) \leq C _ { 0 } , \quad p _ { f } , p _ { b } \in [ 0 , 1 ] ,\tag{5}
$$

where L is given by Eq. 3. We solve this problem via numerical optimization (Appendix G).

Figure 1 shows the solution of Eq. 5, specifically the optimal model size $N ^ { * }$ , the optimal number of training tokens $D ^ { * }$ , and the optimal fractions of blocks in NVFP4 in the forward and backward passes, $p _ { f } ^ { \ast }$ and $p _ { b } ^ { * }$ , as a function of the compute budget $C _ { 0 }$ (x-axis) and the NVFP4 speedup $\lambda _ { \mathrm { { N V 4 } } }$ $( y { \mathrm { - a x i s } } )$ . We assume that each format has the same speedup in the forward and backward passes, $\mathrm { i . e . , } \lambda _ { f , \mathrm { M X 8 } } = \lambda _ { b , \mathrm { M X 8 } } = \lambda _ { \mathrm { M X 8 } }$ and $\lambda _ { f , \mathrm { N V 4 } } = \lambda _ { b , \mathrm { N V 4 } } = \lambda _ { \mathrm { N V 4 } }$ , and fix $\lambda _ { \mathrm { M X 8 } } = 1 . 8$ , following the GEMM kernel speedup reported by NVIDIA Transformer Engine (NVIDIA, 2026). Since actual speedups differ across hardware types and generations, supported data types, and kernel implementations, we report results under other assumed MXFP8 speedups in Appendix H.

Optimal precision increases with the budget. Following a horizontal line in the top panels of Figure 1, i.e., fixing the NVFP4 speedup $\lambda _ { \mathrm { { N V 4 } } }$ while increasing the compute budget $C _ { 0 } ,$ the optimal NVFP4 fractions $p _ { f } ^ { \ast }$ (top left) and $p _ { b } ^ { * }$ (top right) decrease, leaving more blocks in MXFP8. For example, at $\lambda _ { \mathrm { N V 4 } } \doteq 3$ , the optimal configuration switches to MXFP8 in the forward pass around $C _ { 0 } = 1 0 ^ { 2 0 }$ FLOPs and in both passes at around $C _ { 0 } = 5 \times 1 0 ^ { 2 2 }$ FLOPs, with layer-wise mixing smoothing the backward switch into a gradual transition. This contrasts with prior scaling laws, which predict budget-independent optimal precision when jointly optimizing model size, data, and precision (Kumar et al., 2025, Section 4.3.2), and stems from the persistent optimization error in our law. As $C _ { 0 }$ grows, model- and data-dependent loss terms shrink while persistent error remains, so the lower error of higher precision eventually outweighs the compute savings of lower precision.

The backward pass should use low precision first. Comparing the top panels of Figure 1 at a fixed NVFP4 speedup, the backward pass keeps NVFP4 up to much larger compute budgets than the forward pass. Three factors favor NVFP4 in the backward pass. The backward pass accounts for two thirds of the training FLOPs, so each block in NVFP4 saves more compute. Its larger exponent γ<sub>b</sub> (Table 6) means that assigning blocks to NVFP4 incurs a smaller fraction of the loss gap. Finally, the persistent error of NVFP4 relative to MXFP8 is about half as large in the backward pass as in the forward pass in our fitted law (Appendix D). Practitioners with a limited budget for higher precision should therefore allocate it to the forward pass first.

## 5 CONCLUSION

In this work, we develop a scaling law guided by theoretical analysis to predict the loss of quantized pretraining with mixed precision across the forward and backward passes, modeling persistent optimization error from each pass, and show that it predicts held-out loss more accurately than existing laws. Extending the law to layer-wise mixed precision, we find that keeping the first and last blocks in higher precision is particularly effective in the backward pass. Finally, the law predicts that the compute-optimal precision increases with the compute budget, unlike previous precision scaling laws, since the persistent optimization error does not diminish with scale.

## AI USE STATEMENT

We used large language models and coding agents to assist with drafting and editing the manuscript, implementing plotting and optimization code, and providing assistance with mathematical derivations and proofs. We reviewed and verified the AI-assisted content and take full responsibility for the paper.

## ETHICS STATEMENT

This work studies scaling laws and optimization for quantized language model pretraining. It does not involve sensitive personal data or real-world deployment. While improved training efficiency may broaden access to capable models, it may also enable harmful downstream uses. We encourage responsible use of these methods.

## REPRODUCIBILITY STATEMENT

We provide the experimental settings, model configurations, quantization schemes, optimization details, and fitting procedures needed to reproduce our results. We also report the scaling-law experiments and ablations used in our analyses.

## ACKNOWLEDGMENTS

Jinuk Kim is supported by Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) [No. RS2026-25524173, Ultra-Long-Term Hierarchical Memory and Reasoning Architecture for Next-Generation Omnimodal Agents, 50%; and No. RS-2020-II200882, (SW STAR LAB) Development of deployable learning intelligence via self-sustainable and trustworthy machine learning, 20%], and National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. RS-2024-00354036, 30%).

## REFERENCES

Felix Abecassis, Anjulie Agrusa, Dong Ahn, Jonah Alben, Stefania Alborghetti, Michael Andersch, Sivakumar Arayandi, Alexis Bjorlin, Aaron Blakeman, Evan Briones, et al. Pretraining large language models with nvfp4. arXiv preprint arXiv:2509.25149, 2025.

Ahmad Ajalloeian and Sebastian U Stich. On the convergence of sgd with biased gradients. arXiv preprint arXiv:2008.00051, 2020.

Yu Bai, Yu-Xiang Wang, and Edo Liberty. Proxquant: Quantized neural networks via proximal operators. ICLR, 2019.

Yoshua Bengio, Nicholas Leonard, and Aaron Courville. Estimating or propagating gradients ´ through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013.

Johan Bjorck, Alon Benhaim, Vishrav Chaudhary, Furu Wei, and Xia Song. Scaling optimal lr across token horizons. In International Conference on Learning Representations, volume 2025, pp. 83640–83657, 2025.

Dan Busbridge, Amitis Shidani, Floris Weers, Jason Ramapuram, Etai Littwin, and Russell Webb. Distillation scaling laws. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 5977–6045. PMLR, 13–19 Jul 2025.

Richard H Byrd, Peihuang Lu, Jorge Nocedal, and Ciyou Zhu. A limited memory algorithm for bound constrained optimization. SIAM Journal on scientific computing, 16(5):1190–1208, 1995.

Zeyu Cao, Boyang Gu, Cheng Zhang, Pedro Gimenes, Jianqiao Lu, Jianyi Cheng, Xitong Gao, and Yiren Zhao. Scaling laws for mixed quantization. arXiv preprint arXiv:2410.06722, 2024.

Roberto Castro, Andrei Panferov, Rush Tabesh, Oliver Sieberling, Jiale Chen, Mahdi Nikdan, Saleh Ashkboos, and Dan Alistarh. Quartet: Native fp4 training can be optimal for large language models. Advances in Neural Information Processing Systems, 38:43552–43572, 2026.

Mengzhao Chen, Chaoyi Zhang, Jing Liu, Yutao Zeng, Zeyue Xue, Zhiheng Liu, Yunshui Li, Jin Ma, Jie Huang, Xun Zhou, et al. Scaling law for quantization-aware training. arXiv preprint arXiv:2505.14302, 2025.

Brian Chmiel, Maxim Fishman, Ron Banner, and Daniel Soudry. Fp4 all the way: Fully quantized training of llms. arXiv preprint arXiv:2505.19115, 2025.

Jack Cook, Junxian Guo, Guangxuan Xiao, Yujun Lin, Keith Wyss, Mahdi Nazemi, Asit Mishra, Carlo del Mundo, Tijmen Blankevoort, and Song Han. Four over six: More accurate nvfp4 quantization with adaptive block scaling. arXiv preprint arXiv:2512.02010, 2025.

Alexandre Defossez, Yossi Adi, and Gabriel Synnaeve. Differentiable model compression via´ pseudo quantization noise. TMLR, 2022.

Nolan Dey, Bin Zhang, Lorenzo Noci, Mufan Li, Blake Bordelon, Shane Bergsma, Cengiz Pehlevan, Boris Hanin, and Joel Hestness. Don’t be lazy: Completep enables compute-efficient deep transformers. Advances in Neural Information Processing Systems, 38:137707–137739, 2026.

Jesse Dodge, Maarten Sap, Ana Marasovic, William Agnew, Gabriel Ilharco, Dirk Groeneveld,´ Margaret Mitchell, and Matt Gardner. Documenting large webtext corpora: A case study on the colossal clean crawled corpus. In Proceedings of the 2021 conference on empirical methods in natural language processing, pp. 1286–1305, 2021.

Maxim Fishman, Brian Chmiel, Ron Banner, and Daniel Soudry. Scaling fp8 training to trilliontoken llms. In International Conference on Learning Representations, volume 2025, pp. 98631– 98644, 2025.

Elias Frantar, Utku Evci, Wonpyo Park, Neil Houlsby, and Dan Alistarh. Compression scaling laws: Unifying sparsity and quantization. arXiv preprint arXiv:2502.16440, 2025.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Train ing compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395, 2024.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

Tanishq Kumar, Zachary Ankner, Benjamin Spector, Blake Bordelon, Niklas Muennighoff, Mansheej Paul, Cengiz Pehlevan, Christopher Re, and Aditi Raghunathan. Scaling laws for precision.´ In International Conference on Learning Representations, volume 2025, pp. 71833–71864, 2025.

Hao Li, Soham De, Zheng Xu, Christoph Studer, Hanan Samet, and Tom Goldstein. Training quantized nets: A deeper understanding. Advances in neural information processing systems, 30, 2017.

Ziang Li and humans& ai. The 4-bitter lesson: Balancing stability and performance in nvfp4 rl. humans& ai Blog, July 2026. https://humansand.ai/blog/nvfp4-rl.html.

Ilya Loshchilov, Frank Hutter, et al. Fixing weight decay regularization in adam. arXiv preprint arXiv:1711.05101, 5(5):5, 2017.

Paulius Micikevicius, Dusan Stosic, Neil Burgess, Marius Cornea, Pradeep Dubey, Richard Grisenthwaite, Sangwon Ha, Alexander Heinecke, Patrick Judd, John Kamalu, et al. Fp8 formats for deep learning. arXiv preprint arXiv:2209.05433, 2022.

Markus Nagel, Marios Fournarakis, Yelysei Bondarenko, and Tijmen Blankevoort. Overcoming oscillations in quantization-aware training. In International Conference on Machine Learning, pp. 16318–16330. PMLR, 2022.

NVIDIA. Transformer engine, 2026. URL https://github.com/NVIDIA/ TransformerEngine/blob/9c33764bd9fd88cb4984ef2b60e9f0623055d3b2/ docs/examples/gemm\_profiling/gemm\_profiling.rst.

Yunjie Pan, Yongyi Yang, Hanmei Yang, and Scott Mahlke. Snip: An adaptive mixed precision framework for subbyte large language model training. In Proceedings of the 31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, pp. 1815–1831, 2026.

Andrei Panferov, Jiale Chen, Soroush Tabesh, Mahdi Nikdan, and Dan Alistarh. QuEST: Stable training of LLMs with 1-bit weights and activations. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pp. 47820–47836. PMLR, 13–19 Jul 2025.

Andrei Panferov, Erik Schultheis, Soroush Tabesh, and Dan Alistarh. Quartet ii: Accurate llm pretraining in nvfp4 by improved unbiased gradient estimation. arXiv preprint arXiv:2601.22813, 2026a.

Andrei Panferov, Alexandra Volkova, Ionut-Vlad Modoranu, Vage Egiazarian, Mher Safaryan, and Dan Alistarh. Unified scaling laws for compressed representations. Advances in Neural Information Processing Systems, 38:151086–151112, 2026b.

Jianlin Su. Deconstructing scaling laws: The interplay of optimization, architecture, and data, July 2026. URL https://kexue.fm/archives/11833.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Albert Tseng, Tao Yu, and Youngsuk Park. Training llms with mxfp4. In Yingzhen Li, Stephan Mandt, Shipra Agrawal, and Emtiyaz Khan (eds.), Proceedings of The 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings ofMachine Learning Research, pp. 1630–1638. PMLR, 03–05 May 2025.

Mathurin Videau, Badr Youbi-Idrissi, David Lopez-Paz, and Kartik Ahuja. Skaling: Chinchilla’s exponents meet kaplan’s coupling. arXiv preprint arXiv:2608.07222, 2026.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

## A PROOF OF REMARK 3.1

In this section, we provide the proof of Remark 3.1, restated here for convenience.

Remark 3.1. Consider updates $\theta _ { t + 1 } = \theta _ { t } - \eta \hat { g } _ { t }$ , where $\hat { \boldsymbol g } _ { t } \in \mathbb R ^ { N }$ is a minibatch gradient estimate at $\hat { \theta } _ { t } = Q ( \theta _ { t } ; \psi _ { f } )$ using backward quantization scheme $\psi _ { b }$ . Define $b _ { t } : = \mathbb { E } [ \hat { g } _ { t } \mid \theta _ { t } ] - \nabla \mathcal { L } _ { N } ( \hat { \theta } _ { t } )$ . For a minimizer $q ^ { * }$ of $\mathcal { L } _ { N }$ over $Q ( \mathbb { R } ^ { N } ; \psi _ { f } )$ , set $R _ { 1 } ( N , \psi _ { f } ) : = \mathbb { E } \| \theta _ { 1 } - q ^ { * } \| ^ { 2 }$ . Assume $\psi _ { f }$ is deterministic, $\mathcal { L } _ { N }$ is differentiable and convex, and, uniformly over $t , T , \mathbb { E } \| \nabla \mathcal { L } _ { N } ( \hat { \theta } _ { t } ) \| ^ { 2 } \leq G ^ { 2 }$ and $\hat { g } _ { t } , b _ { t } , \hat { \theta } _ { t } - \theta _ { t }$ and $\theta _ { t } - q ^ { * }$ have bounded second moments. For learning rate chosen as $\eta = c / \sqrt { T } \ ( c > 0 )$

$$
\begin{array} { r l } { \mathbb { E } _ { J , \theta _ { J } } [ \mathcal { L } _ { N } ( \hat { \theta } _ { J } ) ] - \mathcal { L } _ { N , \psi _ { f } } ^ { * } \le \underbrace { \cfrac { 1 } { 2 \sqrt { T } } \left[ \cfrac { R _ { 1 } ( N , \psi _ { f } ) } { c } + \cfrac { c } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \| \hat { g } _ { t } \| ^ { 2 } \right] } _ { \mathrm { ( a ) } O ( T ^ { - 1 / 2 } ) } } & { } \\ { + \underbrace { G \left( \cfrac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \| \hat { \theta } _ { t } - \theta _ { t } \| ^ { 2 } \right) ^ { 1 / 2 } } _ { \mathrm { ( b ) } O ( 1 ) } + \underbrace { \cfrac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } [ \| b _ { t } \| \| \theta _ { t } - q ^ { * } \| ] } _ { \mathrm { ( c ) } O ( 1 ) } . } \end{array}
$$

Proof. Write $g _ { t } = \nabla \mathcal { L } _ { N } ( \hat { \theta } _ { t } )$ and $e _ { t } = \hat { \theta } _ { t } - \theta _ { t }$ . By convexity and $\mathbb { E } [ \hat { g } _ { t } \mid \theta _ { t } ] = g _ { t } + b _ { t }$

$$
\begin{array} { r l } & { \mathbb { E } [ \mathcal { L } _ { N } ( \hat { \theta } _ { t } ) - \mathcal { L } _ { N } ( q ^ { * } ) ] \leq \mathbb { E } \langle g _ { t } , \theta _ { t } - q ^ { * } + e _ { t } \rangle } \\ & { \qquad = \mathbb { E } \langle \hat { g } _ { t } , \theta _ { t } - q ^ { * } \rangle - \mathbb { E } \langle b _ { t } , \theta _ { t } - q ^ { * } \rangle + \mathbb { E } \langle g _ { t } , e _ { t } \rangle . } \end{array}
$$

The update rule gives

$$
2 \eta \langle \hat { g } _ { t } , \theta _ { t } - q ^ { * } \rangle = \| \theta _ { t } - q ^ { * } \| ^ { 2 } - \| \theta _ { t + 1 } - q ^ { * } \| ^ { 2 } + \eta ^ { 2 } \| \hat { g } _ { t } \| ^ { 2 } .
$$

Summing over t and dropping the nonpositive terminal term yields

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } [ \mathcal { L } _ { N } ( \hat { \theta } _ { t } ) - \mathcal { L } _ { N } ( { q } ^ { * } ) ] \leq \frac { R _ { 1 } ( N , \psi _ { f } ) } { 2 \eta T } + \frac { \eta } { 2 T } \sum _ { t = 1 } ^ { T } \mathbb { E } \| \hat { g } _ { t } \| ^ { 2 } } \\ & { \displaystyle \quad \quad \quad \quad + \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } [ \| b _ { t } \| \| \theta _ { t } - { q } ^ { * } \| ] + \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \langle g _ { t } , e _ { t } \rangle . } \end{array}
$$

By Cauchy–Schwarz,

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \langle g _ { t } , e _ { t } \rangle \leq \left( \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \| g _ { t } \| ^ { 2 } \right) ^ { 1 / 2 } \left( \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \| e _ { t } \| ^ { 2 } \right) ^ { 1 / 2 } \leq G \left( \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \| e _ { t } \| ^ { 2 } \right) ^ { 1 / 2 } .
$$

Taking J uniformly from $\{ 1 , \ldots , T \}$ independently of the trajectory, and substituting $\eta = c / \sqrt { T }$ and $\check { \mathcal { L } _ { N } } ( q ^ { * } ) = \check { \mathcal { L } _ { N , \psi _ { f } } ^ { * } }$ , proves the claim. The stated orders follow from the uniform second-moment bounds. □

## B IMPLEMENTATION DETAILS

## B.1 MODEL ARCHITECTURE AND TRAINING

Our scaling law grid models are derived from the Qwen3-8B architecture (Yang et al., 2025) by varying the number of Transformer blocks K and setting the remaining dimensions from K following the proportions of Qwen3-8B, which has $K = 3 6$ blocks, width $d _ { \mathrm { m o d e l } } = 4 0 9 6$ , and 32 query heads. The width scales linearly with depth, $d _ { \mathrm { m o d e l } } = \mathrm { r o u n d } ( 4 0 9 6 K / 3 6 , 1 2 8 )$ ), and the number of query heads scales with width, $n _ { \mathrm { h e a d s } } = \mathrm { r o u n d } ( 3 2 d _ { \mathrm { m o d e l } } / 4 0 9 6 , 8 _ { \mathrm { } }$ ), where $\operatorname { r o u n d } ( x , m )$ rounds x to the nearest multiple of m. All other choices follow Qwen3-8B, including grouped-query attention with 8 key-value heads and a head dimension of 128, an FFN dimension of $3 d _ { \mathrm { m o d e l } }$ , QK normalization, and tied input and output embeddings over the Qwen3 vocabulary of 151,936 tokens. Table 7 lists the resulting configurations, where non-embedding parameter counts include normalization weights.

Table 7: Model configurations derived from Qwen3-8B. K denotes the number of Transformer blocks. Parameter counts are given without and with the tied embeddings.
<table><tr><td>Model</td><td>K</td><td> $d _ { \mathrm { m o d e l } }$ </td><td>Query heads</td><td>KV heads</td><td>FFN dim.</td><td>Non-emb. params</td><td>Total params</td></tr><tr><td>280M</td><td>12</td><td>1408</td><td>8</td><td>8</td><td>4224</td><td>283M</td><td>497M</td></tr><tr><td>430M</td><td>14</td><td>1536</td><td>16</td><td>8</td><td>4608</td><td>429M</td><td>663M</td></tr><tr><td>640M</td><td>16</td><td>1792</td><td>16</td><td>8</td><td>5376</td><td>639M</td><td>911M</td></tr><tr><td>910M</td><td>18</td><td>2048</td><td>16</td><td>8</td><td>6144</td><td>906M</td><td>1217M</td></tr><tr><td>Qwen3-8B</td><td>36</td><td>4096</td><td>32</td><td>8</td><td>12288</td><td>6946M</td><td>7568M</td></tr></table>

Table 8: Calibration and held-out splits for model size and data extrapolation. # points denotes the number of data points in each split, including all forward and backward scheme combinations. The Quartet grid uses tokens-per-parameter ratios $( D / N )$ , whereas our grid uses training token counts (D). N denotes the non-embedding parameter count.
<table><tr><td colspan="2"></td><td colspan="2">Calibration</td><td colspan="2">Held-out</td></tr><tr><td>Data grid</td><td>Evaluation</td><td># points</td><td>Model sizes and token budgets</td><td># points</td><td>Model sizes and token budgets</td></tr><tr><td rowspan="2">Quartet grid (Llama)</td><td>Model size extrapolation</td><td>72</td><td> $\overline { { N \in \{ 3 0 , 5 0 , 1 0 0 \} \mathrm { M } } }$   $D / \bar { N } \in \{ 2 5 , 5 0 , \dot { 1 } 0 0 , 2 0 0 , 4 0 0 , 8 0 0 \}$ </td><td>12</td><td> $\overline { { N = 2 0 0 \mathrm { M } } }$   $D / N \in \{ 2 5 , 5 0 , 1 0 0 \}$ </td></tr><tr><td>Data extrapolation</td><td>60</td><td>N ∈ {30, 50, 100}M D/N ^∈ {25, 50, 100, 200, 400}</td><td>12</td><td> $N \in \{ 3 0 , 5 0 , 1 0 0 \} \mathrm { M }$   $D / N \stackrel { \cdot } { = } 8 0 0$ </td></tr><tr><td rowspan="2">Our grid (Qwen3)</td><td>Model size extrapolation</td><td>315</td><td>N ∈ {280, 430, 640}M  $D \in \{ 1 2 . 8 , 1 5 . 3 , 1 7 . 9 , 2 1 . 1 , 2 5 . 5 , 3 0 . 0 , 3 5 . 8 \} \mathrm { B }$ </td><td>60</td><td>N = 910M  $D \in \{ 1 8 . 1 , 2 1 . 7 , 2 5 . 4 , 2 9 . 9 \} \mathrm { B }$ </td></tr><tr><td>Data extrapolation</td><td>270</td><td> $\overline { { N \in \{ 2 8 0 , 4 3 0 , 6 4 0 \} \mathrm { M } } }$   $D \in \{ 1 2 . 8 , 1 5 . 3 , 1 7 . 9 , 2 1 . 1 , 2 5 . 5 , 3 0 . 0 \} \mathrm { B }$ </td><td>45</td><td> $N \in \{ 2 8 0 , 4 3 0 , 6 4 0 \} \mathrm { M }$   $D = \mathrm { \dot { 3 } 5 . 8 \dot { B } }$ </td></tr></table>

Parameterization and initialization We use the CompleteP parameterization (Dey et al., 2026) with Qwen3-8B as the base shape, so that hyperparameters tuned on one model transfer across width and depth. For each model size, all runs start from the same initial checkpoint and see the same data order, so BF16 and all quantized recipes differ only in their quantization schemes.

Optimization and data We use AdamW (Loshchilov et al., 2017) $( \beta = ( 0 . 9 , 0 . 9 5 ) , \epsilon = 1 0 ^ { - 8 }$ weight decay 0.1, gradient clipping 1.0) with batches of 1024 sequences of length 4096 (about 4.2M tokens), and a warmup–stable–decay schedule (Hu et al., 2024) with 244 warmup steps and a linear decay to 1% of the peak over the final 10% of steps. A BF16 sweep on the 280M model at 5.7B tokens (20 tokens per parameter) gives an optimal peak learning rate of $1 . 2 \times 1 0 ^ { - 3 }$ , which is also optimal for the 170M and 430M models, confirming transfer across width and depth (Dey et al., 2026). Scaling it to the smallest token budget of each model (Table 8) with $\eta \propto \dot { D } ^ { - 0 . 3 2 }$ (Bjorck et al., 2025) gives $9 . 2 5 \times 1 0 ^ { - 4 }$ for the three smaller models and $8 . 2 7 \times 1 0 ^ { - 4 }$ for the 910M model, which we use for all budgets and quantization schemes. We report validation loss on C4 (Dodge et al., 2021).

Token budgets We train every model with at least 20 tokens per parameter (Hoffmann et al., 2022), using the token budgets listed in Table 8. The three smaller models share the budgets of the 640M model, ranging from 12.8B to 35.8B tokens, and the 910M model uses budgets from 18.1B to 29.9B tokens. For cost efficiency, we obtain each token budget by resuming from the run of a smaller budget before its decay phase and applying its own 10% decay.

## B.2 EVALUATION SPLITS

Table 8 lists the calibration and held-out splits for the two extrapolation settings in Section 3.2. In model size extrapolation, each law is fitted to runs with the three smaller models of each grid and evaluated on the largest model, which is 200M for the Quartet grid and 910M for our grid. In data extrapolation, each law is fitted to runs with the three smaller models at all but the largest token budget and evaluated at that budget, which is 800 tokens per parameter for the Quartet grid and 35.8B tokens for our grid. The largest model is therefore excluded from data extrapolation. In both settings, the laws are fitted only to the calibration split.

Table 9: Ablation of our scaling law over all combinations of its design choices. We report held-out MSE under model size and data extrapolation on both grids and their geometric mean. Variants (1)–(3) are those in Table 5, and – marks the remaining combinations. The best geometric mean is shown in bold.
<table><tr><td></td><td colspan="4">Form</td><td colspan="2">Quartet grid (Llama)</td><td colspan="2">Our grid (Qwen3)</td><td rowspan="2">Geo. mean ↓</td></tr><tr><td>Scaling law</td><td>∈f</td><td> $\epsilon _ { b }$ </td><td> $\rho _ { D }$ </td><td>€ placement</td><td>Model size</td><td>Data</td><td>Model size</td><td>Data</td></tr><tr><td></td><td>X</td><td>x</td><td> $\psi _ { f } , \psi _ { b }$ </td><td></td><td>1.27e-4</td><td>7.98e-5</td><td>7.28e-6</td><td>2.61e-5</td><td>3.72e-5</td></tr><tr><td>Quartet</td><td>X</td><td>x</td><td> $\psi _ { b }$ </td><td>一</td><td>1.65e-4</td><td>9.02e-5</td><td>8.39e-6</td><td>2.66e-5</td><td>4.27e-5</td></tr><tr><td></td><td></td><td>X</td><td> $\psi _ { f } , \psi _ { b }$ </td><td>Inside</td><td>2.74e-5</td><td>5.63e-5</td><td>4.47e-6</td><td>2.12e-5</td><td>1.96e-5</td></tr><tr><td></td><td></td><td>X</td><td> $\psi _ { b }$ </td><td>Inside</td><td>2.87e-5</td><td>7.10e-5</td><td>5.17e-6</td><td>2.31e-5</td><td>2.22e-5</td></tr><tr><td></td><td></td><td>X</td><td> $\psi _ { f } , \psi _ { b }$ </td><td>Outside</td><td>1.95e-5</td><td>2.02e-5</td><td>3.69e-6</td><td>2.20e-5</td><td>1.34e-5</td></tr><tr><td>Variant (2)</td><td></td><td>X</td><td> $\psi _ { b }$ </td><td>Outside</td><td>1.67e-5</td><td>1.86e-5</td><td>4.32e-6</td><td>2.38e-5</td><td>1.34e-5</td></tr><tr><td></td><td>X</td><td></td><td> $\psi _ { f } , \psi _ { b }$ </td><td>Inside</td><td>9.97e-5</td><td>1.02e-4</td><td>8.19e-6</td><td>4.98e-6</td><td>2.54e-5</td></tr><tr><td></td><td>X</td><td></td><td> $\psi _ { b }$ </td><td>Inside</td><td>1.76e-4</td><td>1.37e-4</td><td>1.06e-5</td><td>8.08e-6</td><td>3.79e-5</td></tr><tr><td></td><td>X</td><td></td><td> $\psi _ { f } , \psi _ { b }$ </td><td>Outside</td><td>1.15e-4</td><td>1.05e-4</td><td>6.07e-6</td><td>6.14e-6</td><td>2.59e-5</td></tr><tr><td>Variant (3)</td><td>X</td><td></td><td> $\psi _ { b }$ </td><td>Outside</td><td>1.84e-4</td><td>1.35e-4</td><td>8.82e-6</td><td>9.12e-6</td><td>3.76e-5</td></tr><tr><td></td><td></td><td></td><td> $\psi _ { f } , \psi _ { b }$ </td><td>Inside</td><td>3.32e-5</td><td>7.35e-5</td><td>7.74e-6</td><td>3.98e-6</td><td>1.66e-5</td></tr><tr><td></td><td></td><td></td><td> $\psi _ { b }$ </td><td>Inside</td><td>3.86e-5</td><td>9.75e-5</td><td>8.17e-6</td><td>5.43e-6</td><td>2.02e-5</td></tr><tr><td>Variant (1)</td><td></td><td></td><td> $\psi _ { f } , \psi _ { b }$ </td><td>Outside</td><td>2.14e-5</td><td>5.24e-5</td><td>4.81e-6</td><td>5.12e-6</td><td>1.29e-5</td></tr><tr><td>Ours</td><td></td><td></td><td> $\psi _ { b }$ </td><td>Outside</td><td>1.43e-5</td><td>5.06e-5</td><td>5.29e-6</td><td>6.07e-6</td><td>1.23e-5</td></tr></table>

## B.3 FITTING THE SCALING LAWS

We fit both our law and the Quartet law in two stages, following Castro et al. (2026), and both stages minimize a Huber loss with threshold $1 0 ^ { - 3 }$ on log-loss residuals log $\hat { L } - \log L$ . We use Levenberg– Marquardt with 200 initializations for the baseline fit and a bounded trust-region reflective solver with 20 initializations for the quantization parameters. We repeat both stages independently for each data grid and extrapolation setting using only calibration data.

## B.4 LAYER-WISE MIXED-PRECISION EXPERIMENTS

We use four Qwen3 model sizes with 280M, 430M, 640M, and 910M non-embedding parameters, each trained with four token budgets, yielding 16 combinations of N and D. The three smaller models are trained on 12.8B, 15.3B, 17.9B, and 21.1B tokens, and the largest model on 18.1B, 21.7B, 25.4B, and 29.9B tokens. We mix precision in one pass at a time. For each pass and strategy, we keep 25%, 50%, or 75% of the blocks in MXFP8, with each selection containing the smaller ones, so that $p = 1 - k / K$ for k MXFP8 blocks out of K equals 0.75, 0.5, or 0.25 up to rounding.

• Forward mixing. $\psi _ { 1 }$ and $\psi _ { 2 }$ use MXFP8 and NVFP4 with $4 / 6 + 1 \mathrm { D }$ scaling in the forward pass, respectively, and both use MS-EDEN in the backward pass.

• Backward mixing. $\psi _ { 1 }$ and $\psi _ { 2 }$ use MXFP8 with round-to-nearest and NVFP4 with SR in the backward pass, respectively, and both use 4/6 + 1D scaling in the forward pass.

We evaluate the extended law under model size extrapolation, holding out the 910M model. To predict the endpoints $L _ { 1 }$ and $L _ { 2 } ,$ we use each scaling law fitted to the calibration split for model size extrapolation on our grid (Appendix B.2), which excludes the 910M model. We then fit $\gamma _ { S }$ in Eq. 2 for each pass and strategy to all mixed-precision runs on the three smaller models, using predicted endpoints, so that the 910M model is excluded from both fits.

## C ABLATION OVER ALL COMBINATIONS OF DESIGN CHOICES

While Table 5 varies one design choice of our law at a time, Table 9 evaluates all combinations of three choices, (i) which persistent error terms are included among $\epsilon _ { f } ( \psi _ { f } )$ and $\epsilon _ { b } ( \psi _ { b } )$ , (ii) whether $\rho _ { D }$ depends on the backward scheme ψ<sub>b</sub> only or on both schemes ψ<sub>f</sub> and $\psi _ { b }$ , and (iii) whether the error terms are added outside $\gamma _ { \mathrm { : } }$ , as in Eq. 1, or inside it, i.e., $L = \bar { ( } A ( \rho _ { N } N ) ^ { - \alpha } + B ( \rho _ { D } D ) ^ { - \beta } +$ $\epsilon _ { f } + \epsilon _ { b } ) ^ { \gamma } + E .$ . Each of the resulting 14 forms is fitted and evaluated with the same protocol as our law (Section 3.2). Overall, our law achieves the lowest geometric-mean MSE. For (i) the persistent error terms, removing $\epsilon _ { f }$ increases the geometric-mean MSE by 1.53× to 3.04× (rows 7–10 vs. rows 11–14), and removing $\epsilon _ { b }$ increases it by 1.04× to 1.18× (rows 3–6 vs. rows 11–14). For (ii) the dependence of $\rho _ { D } .$ , the effect of a joint $\rho _ { D } ( \psi _ { f } , \psi _ { b } )$ is mixed across the other choices, and it does not improve on our law (row 13 vs. row 14). For (iii) the placement of the error terms, placing them inside $\gamma$ increases the geometric-mean MSE by 1.31× on average.

Table 10: Fitted parameters of our scaling law on the Quartet grid (Llama) and our grid (Qwen3), for each extrapolation setting (model size and data). The shared coefficients are fitted to the BF16 runs, and the scheme-dependent parameters to the quantized runs, with $\rho _ { N } = \rho _ { D } = 1$ and $\epsilon _ { f } = \epsilon _ { b } = 0$ for BF16 by construction. N and $D$ are measured in non-embedding parameters and tokens.
<table><tr><td rowspan="2">Parameter</td><td rowspan="2">Scheme</td><td colspan="2">Quartet grid (Llama)</td><td colspan="2">Our grid (Qwen3)</td></tr><tr><td>Model size</td><td>Data</td><td>Model size</td><td>Data</td></tr><tr><td>E</td><td></td><td>1.650</td><td>1.763</td><td>2.546</td><td>2.574</td></tr><tr><td>A</td><td></td><td> $1 . 1 0 \times 1 0 ^ { 5 }$ </td><td> $3 . 5 0 \times 1 0 ^ { 5 }$ </td><td> $7 . 6 2 \times 1 0 ^ { 4 }$ </td><td> $9 . 3 8 \times 1 0 ^ { 4 }$ </td></tr><tr><td>B</td><td></td><td> $2 . 5 0 \times 1 0 ^ { 5 }$ </td><td> $6 . 7 0 \times 1 0 ^ { 5 }$ </td><td> $8 . 2 1 \times 1 0 ^ { 5 }$ </td><td> $1 . 0 4 \times 1 0 ^ { 6 }$ </td></tr><tr><td>α</td><td></td><td>0.622</td><td>0.701</td><td>0.629</td><td>0.642</td></tr><tr><td>β</td><td></td><td>0.550</td><td>0.605</td><td>0.631</td><td>0.642</td></tr><tr><td>γ</td><td></td><td>0.322</td><td>0.315</td><td>0.767</td><td>0.787</td></tr><tr><td rowspan="3"> $\rho _ { N }$ </td><td>MXFP4 (QuEST)</td><td>0.770</td><td>0.798</td><td></td><td></td></tr><tr><td>MXFP8</td><td></td><td></td><td>0.987</td><td>0.984</td></tr><tr><td>NVFP4 (4/6 + 1D)</td><td></td><td></td><td>0.922</td><td>0.923</td></tr><tr><td rowspan="5"> $\rho _ { D }$ </td><td>MXFP4 (SR)</td><td>0.951</td><td>0.973</td><td></td><td></td></tr><tr><td>MXFP8</td><td></td><td></td><td>0.998</td><td>1.000</td></tr><tr><td>NVFP4 (MS-EDEN)</td><td></td><td></td><td>1.000</td><td>1.000</td></tr><tr><td>NVFP4 (RHT + SR)</td><td></td><td></td><td>1.000</td><td>1.000</td></tr><tr><td>NVFP4 (SR)</td><td></td><td></td><td>1.000</td><td>1.000</td></tr><tr><td rowspan="3"> $\epsilon _ { f }$ </td><td>MXFP4 (QuEST)</td><td>0.0340</td><td>0.0397</td><td></td><td></td></tr><tr><td>MXFP8</td><td></td><td></td><td>0.0021</td><td>0.0016</td></tr><tr><td>NVFP4 (4/6 + 1D)</td><td></td><td></td><td>0.0151</td><td>0.0152</td></tr><tr><td rowspan="5"> $\epsilon _ { b }$ </td><td>MXFP4 (SR)</td><td>0.0076</td><td>0.0100</td><td></td><td></td></tr><tr><td>MXFP8</td><td></td><td></td><td>0.0004</td><td>0.0008</td></tr><tr><td>NVFP4 (MS-EDEN)</td><td></td><td></td><td>0.0076</td><td>0.0076</td></tr><tr><td>NVFP4 (RHT + SR)</td><td></td><td></td><td>0.0076</td><td>0.0076</td></tr><tr><td>NVFP4 (SR)</td><td></td><td></td><td>0.0215</td><td>0.0211</td></tr></table>

## D FITTED SCALING LAW PARAMETERS

Table 10 lists the parameters of our scaling law (Eq. 1) fitted to the calibration split of each grid and extrapolation setting. The shared coefficients $E , \bar { A , } B , \alpha , \beta ,$ , and $\gamma$ are fitted in the first stage to the BF16 runs, and the scheme-dependent parameters $\rho _ { N } , \rho _ { D } , \epsilon _ { f } ,$ and $\epsilon _ { b }$ are fitted in the second stage to the quantized runs, with $\rho _ { N } = \rho _ { D } = 1$ and $\epsilon _ { f } = \epsilon _ { b } = 0$ for BF16 by construction. N and $D$ are measured in non-embedding parameters and tokens, respectively.

## E SCALING LAW FITS FOR ALL SCHEMES

Figures 5–10 show the fits of the Quartet law and our law for all forward and backward scheme combinations on both grids, under model size and data extrapolation. Blue points and curves show the calibration runs and the fitted law, and red points and curves show the held-out runs and the extrapolated predictions. Each legend reports the held-out $R ^ { 2 }$ and MSE in Table 4. These figures compare the two laws for each scheme combination on each grid and extrapolation setting. For

![](images/c1cd8dded29f1516e064b50f3b7fd7e7e13751e83b13ca82697e5ea750f52a8d.jpg)  
Figure 5: Quartet grid (Llama), model size extrapolation, for (a) the Quartet law and (b) our law. Each law is fitted to the 30M–100M models and evaluated on the held-out 200M models.

(b) Our law

instance, on the Quartet grid, the Quartet law underestimates the loss of the held-out 200M models with MXFP4 (QuEST) in the forward pass (Figure 5a, middle and right), whereas our law matches it closely (Figure 5b). Similarly, on our grid, the Quartet law underestimates the held-out loss at 35.8B tokens for the runs with NVFP4 (SR) in the backward pass and deviates from their calibration points (Figure 9, first row, fourth column, and third row, first column), whereas our law matches both closely (Figure 10).

![](images/edfb9c182bf40ee19cb56be6fe0bb7a7763814051e2466dfa2fe999bf1808520.jpg)  
(a) Quartet law

![](images/56d2745e0169afff90554979cfa93c33e007e8bbae0d04b4d6c1071528f12495.jpg)

![](images/c225fa614f4cab3db4d3eda82f0e43627a790f7ba894e344bba3bd3984d93f3d.jpg)

![](images/fe219c2cf40485a8fbfe02988806481a77c5a2d9511d63ceb11e7e04557a5956.jpg)

![](images/c95871660a5c1005a62d98706bb951d0639f1e99f9e3130f89af5ca31eeef81c.jpg)  
(b) Our law

![](images/8c8952045d74de72c636a6a9662951e63a0b7e25ea0936b509aa53a3ff0a6a37.jpg)  
Figure 6: Quartet grid (Llama), data extrapolation, for (a) the Quartet law and (b) our law. Each law is fitted to 25–400 tokens per parameter and evaluated at the held-out 800 tokens per parameter.

![](images/0ae32dbadd076cd8aad76950c286c17aa91b8ad3cb988d6a1fda1875dcc8fa38.jpg)  
Figure 7: Our grid (Qwen3), model size extrapolation, for the Quartet law. The law is fitted to the 280M–640M models and evaluated on the held-out 910M models.

![](images/226f9e79ba3897bac262d3d558f4f4e32f44f2335c8a168f44d474132ff776d2.jpg)  
Figure 8: Our grid (Qwen3), model size extrapolation, for our law. The law is fitted to the 280M– 640M models and evaluated on the held-out 910M models.

![](images/7092dd0f143aba4a0b5d356f25f832c45d010f57037a82442d570cd4458bf32d.jpg)  
Figure 9: Our grid (Qwen3), data extrapolation, for the Quartet law. The law is fitted to all but the largest token budget and evaluated at the held-out 35.8B tokens.

![](images/6196055ef51a4410e20c166afdc4b3144c1948525def0959700b224dab3ec347.jpg)  
Figure 10: Our grid (Qwen3), data extrapolation, for our law. The law is fitted to all but the largest token budget and evaluated at the held-out 35.8B tokens.

![](images/0c7aa83519da0014674190e0aab7401f563f5fe83ec02ff93041cc8917f90452.jpg)  
(a) Quartet law endpoints

![](images/26c059c8df97fe5253ffe89d6a7d0a26e5f0c162f20498fae05a79fe7389cc03.jpg)  
(b) Our law endpoints  
Figure 11: Normalized increase in loss of layer-wise mixed-precision runs versus the fraction p of blocks in NVFP4, with endpoints $L _ { 1 }$ and $L _ { 2 }$ predicted by (a) the Quartet law and (b) our law. Solid curves show $p ^ { \gamma _ { S } }$ with $\gamma _ { S }$ fitted on the calibration runs, and dotted lines show linear interpolation $( \gamma _ { S } = 1 )$ .

## F LAYER-WISE MIXED-PRECISION FITS

Figure 11 plots the normalized increase in loss $( L - L _ { 1 } ) / ( L _ { 2 } - L _ { 1 } )$ of each mixed-precision run against the fraction p of blocks in NVFP4, along with the fitted prediction $p ^ { \gamma _ { S } }$ of Eq. 2, using the endpoints $L _ { 1 }$ and $L _ { 2 }$ predicted by the Quartet law and by our law. Blue points show the calibration runs used to fit $\gamma _ { S } .$ , and red points show the runs on the held-out 910M model. Each legend reports the held-out $R ^ { 2 }$ and MSE in Figure 4, and each panel title reports the fitted $\gamma _ { S }$ and the held-out $R ^ { 2 }$ for that pass and strategy. With endpoints predicted by our law, the fitted predictions match the normalized losses more closely than with endpoints predicted by the Quartet law, with held-out $R ^ { 2 }$ of 0.979–0.997 versus 0.962–0.984 across panels.

## G SOLVING THE COMPUTE-OPTIMAL ALLOCATION

The mixed loss in Eq. 3 combines the four uniform endpoint losses $L _ { f , b } ( N , D )$ from Eq. 1, all evaluated at the same N and D, since a mixed configuration trains a single model on a single

dataset. We write the cost in Eq. 4 as $C = N D c ( p _ { f } , p _ { b } )$ , where

$$
c ( p _ { f } , p _ { b } ) = 2 \left( \frac { 1 - p _ { f } } { \lambda _ { f , \mathrm { M X } 8 } } + \frac { p _ { f } } { \lambda _ { f , \mathrm { N V 4 } } } \right) + 4 \left( \frac { 1 - p _ { b } } { \lambda _ { b , \mathrm { M X } 8 } } + \frac { p _ { b } } { \lambda _ { b , \mathrm { N V 4 } } } \right)\tag{6}
$$

is the training cost per parameter per token, which equals 6 for BF16. Since the loss decreases in $D ,$ the budget constraint is tight at the optimum, so $D \doteq C _ { 0 } / ( N c ( p _ { f } , p _ { b } ) )$ ), and we solve

$$
\operatorname* { m i n } _ { \substack { p _ { f } , p _ { b } \in [ 0 , 1 ] , N > 0 } } L \bigg ( N , \frac { C _ { 0 } } { N c ( p _ { f } , p _ { b } ) } , p _ { f } , p _ { b } \bigg ) .\tag{7}
$$

This problem does not admit a closed-form solution, since the endpoint losses have different opti mal splits between parameters and data. We therefore solve it numerically, similar to Kumar et al. (2025, Appendix E.3), who solve a related compute-optimal problem via grid search. Specifically, we perform a grid search over $( p _ { f } , p _ { b } , N )$ and refine the best candidates with L-BFGS-B (Byrd et al., 1995), running it from multiple starting points and keeping the best solution, as in the fitting procedure of Hoffmann et al. (2022, Section 3.3).

## H COMPUTE-OPTIMAL ALLOCATION UNDER OTHER MXFP8 SPEEDUPS

Figure 1 fixes the MXFP8 speedup over BF16 at 1.8×, following the GEMM kernel speedup reported by NVIDIA Transformer Engine (NVIDIA, 2026), but end-to-end speedups can be lower in practice. Figure 12 shows the results with MXFP8 speedups of 1.6× and 1.4×. A lower MXFP8 speedup makes higher precision more expensive, so NVFP4 remains optimal up to larger compute budgets. The qualitative conclusions of Section 4.3 hold in all cases, as the optimal precision still increases with the compute budget and the backward pass retains NVFP4 up to larger budgets than the forward pass, while the optimal model size and tokens per parameter remain nearly unchanged.

## I RELATED WORK

Kaplan et al. (2020) propose a coupled power law of the form $L = ( A N ^ { - \alpha } + B D ^ { - 1 } ) ^ { \gamma }$ , where the outer exponent γ couples the model and data terms and the inner data exponent is fixed to one. The Chinchilla scaling law (Hoffmann et al., 2022) takes the additive form $\dot { L ^ { - } } = A N ^ { - \alpha } + B D ^ { - \beta } + E$ allowing separate exponents for model size and data but no interaction between their contributions. More recent work combines independent inner exponents with an outer coupling exponent in the form $L = ( A N ^ { - \alpha } + B D ^ { - \beta } ) ^ { \gamma } \stackrel { . } { + } E$ , empirically improving loss prediction across data regimes (Busbridge et al., 2025; Videau et al., 2026).

Table 2 summarizes existing laws that incorporate quantization or compression. For post-training quantization (PTQ), Kumar et al. (2025) add a degradation term $C _ { T } D ^ { \gamma _ { D } } N ^ { - \gamma _ { N } } e ^ { - P _ { \mathrm { p o s t } } / \gamma _ { \mathrm { p o s t } } }$ to the Chinchilla law, where $P _ { \mathrm { p o s t } }$ denotes the post-training precision. The degradation grows with the training token count and decreases with model size and precision. Cao et al. (2024) model mixedprecision PTQ by adding a degradation term $C e ^ { \kappa p } N ^ { - \bar { \gamma } _ { N } } ( G + d ) ^ { \gamma _ { G } }$ , where $p$ is the fraction of parameters in low precision, chosen by random search, and $G$ is the block size. For quantized pretraining (QPT), Kumar et al. (2025) replace N with the effective parameter count $( 1 - e ^ { - \bar { P } / \bar { \gamma } } ) ^ { 3 } N .$ where $P$ denotes the training precision and γ¯ is a fitted constant. Frantar et al. (2025) generalize this rescaling to an effective parameter count $\rho _ { N } ( \psi _ { f } ) N$ that depends on the compression scheme. For quantization-aware training (QAT), Chen et al. (2025) add an error term $k D ^ { \gamma _ { D } } \bar { N } ^ { - \gamma _ { N } } ( \log _ { 2 } G ) ^ { \gamma _ { G } }$ that also depends on the quantization group size $G .$ Panferov et al. (2026b) predict parameter efficiency from the representation error of the compression scheme, where ${ \mathrm { G M S E } } ( \psi _ { f } )$ measures the error of fitting Gaussian data using the forward scheme $\psi _ { f }$ and $\widetilde { \rho }$ maps this error to parameter efficiency. Finally, the Quartet law (Castro et al., 2026) rescales both the parameter count and the training token count, with $\rho _ { N } ( \psi _ { f } )$ depending on the forward scheme and $\rho _ { D } ( \psi _ { b } )$ on the backward scheme.

![](images/597faa89f69b47cffdd82338ce025ba30d6d73baa245b347f3208b796475b663.jpg)

Backward NVFP4 fraction p  
![](images/5011b5345a11aa0e220e7d931176b57e845efe946fed218aa09e539907a3cf0b.jpg)

![](images/5431efe8a9a17f019f032ff38382de599ed5117208ab9f1a23532e1864650abb.jpg)  
Compute budget (BF16-equivalent FLOPs)

![](images/b63696cf288ef0489bbd76a24ca1aa0de294c88dfaf19522598e86426a011e9e.jpg)  
Compute budget (BF16-equivalent FLOPs)

(a) MXFP8 speedup 1.6×  
![](images/f34fa1d1189fac8256cdf1e1e03efbe320fa1a0af374f37b439ca7efb70919d7.jpg)

Backward NVFP4 fraction p  
![](images/ec92a82a10f9dc599a23a9f4383923ba0c60fa7e412eaf631073f78cff079475.jpg)

![](images/809dc071c1069394ccab64b18cc963cbaa46d9f79fd6bc88d5293498cca8a09d.jpg)

![](images/16117b6483205132f0d812b3690e3a490fd71782938c1361b253bb046f7adbd7.jpg)  
(b) MXFP8 speedup 1.4×  
Figure 12: Compute-optimal precision allocation predicted by our scaling law, as in Figure 1, with the MXFP8 speedup over BF16 set to (a) 1.6× and (b) 1.4× instead of 1.8×.