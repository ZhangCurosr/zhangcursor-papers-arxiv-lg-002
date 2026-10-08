# HAN-Mamba: Hierarchical Selective State Space Networks for Multi-Scale Financial Volatility Forecasting<sup>⋆</sup>

Mihai Bogdan Deaconu<sup>1[0009−0003−9744−6091]</sup> and Ioan Daniel Pop<sup>1[0000−0002−3740−6579]</sup>

Faculty of Mathematics and Computer Science, Babeş-Bolyai University, Cluj-Napoca, Romania

deaconu.mihai.bogdan@gmail.com, ioan.daniel.pop@ubbcluj.ro

Abstract. Short-horizon realized volatility forecasting requires the integration of market information that evolves at incompatible temporal resolutions, from second-level order book dynamics to weekly regime drift. Our conference work introduced HAN-T, a hierarchical architecture in which scale-specific Transformer encoders process short, mid, and long-horizon streams and a learned attention fuser weighs their contributions. This article replaces the quadratic attention encoders with selective state space (Mamba) encoders while retaining attention only in the fuser, where the input is a three-token set rather than a long sequence. The resulting hybrid, HAN-Mamba, summarizes each stream through a recurrent state whose input-dependent gating matches two structural properties of volatility: persistent but decaying memory and abrupt regime shifts. On the Optiver Realized Volatility Prediction benchmark under time-aware five-fold cross-validation, HAN-Mamba improves mean RM-SPE over HAN-T (0.1942 vs. 0.1965) with 33% fewer parameters. Its linear-time encoders further allow the high-frequency context to be extended from 60 to 240 buckets, reducing error to 0.1927 where the attention variant saturates, and support constant-time streaming updates at inference. Ablations attribute the gains to the encoder swap, confirm that the hierarchical prior transfers across sequence-model families, and show that the permutation-invariant attention fuser remains the correct mechanism for cross-scale integration.

Keywords: Volatility Forecasting · State Space Models · Mamba · Hierarchical Architectures · High-Frequency Finance · Time-Series Analysis.

## 1 Introduction

Realized volatility is the operative measure of short-term risk in modern electronic markets. Accurate forecasts over horizons of minutes drive option market making, execution scheduling, and intraday risk limits, and the data supporting such forecasts arrive as high-frequency order book and trade streams whose statistical structure spans several temporal scales at once. Volatility clusters at the daily scale, decays slowly at the weekly scale, and reacts within seconds to liquidity shocks, a combination that has long resisted single-resolution models [10, 5].

In our conference paper [14] we addressed this multi-scale structure architecturally. The proposed HAN-T processes three time-synchronized streams, a high-frequency feature sequence together with daily and weekly volatility histories, through asymmetric scale-specific Transformer encoders, and integrates the resulting summary tokens with a lightweight attention fuser. Under time-aware cross-validation on the Optiver Realized Volatility Prediction (ORVP) benchmark, HAN-T outperformed econometric, gradient boosting, and flat Transformer baselines, and ablations isolated the contribution of both the hierarchy and the learned fusion.

Limitations of Attention Encoders in This Setting. The encoders carry almost all of the sequence-processing burden in HAN-T, and self-attention is an expensive mechanism for that role. Its cost and activation memory grow quadratically with sequence length, which in practice caps the high-frequency context at 60 buckets and makes longer histories disproportionately costly. Order information must be injected through learned absolute positional embeddings, which are tied to a fixed training length and transfer poorly when the context is extended. At deployment time, each new 10-minute bucket forces a full recomputation over the entire window, since the encoder maintains no state between predictions. None of these costs purchases a needed capability: the encoder output is a single summary vector, not a token-level representation, so the all-pairs interaction structure that justifies attention is largely discarded at readout.

Selective State Spaces as Sequence Encoders. Selective state space models [17] ofer a sequence operator whose inductive bias aligns with volatility dynamics rather than merely approximating attention at lower cost. A discretized state space recurrence propagates a hidden state with multiplicative decay, the natural analogue of volatility persistence with fading memory [11]. The selection mechanism of Mamba makes the state transition input-dependent, so the model can retain information through calm periods and rapidly discount stale context when a liquidity shock arrives, an architectural counterpart of regimedependent memory. Computation is linear in sequence length, inference reduces to a constant-time state update per new observation, and the hardware-aware scan keeps the operator fast on modern accelerators [17, 13].

Our Approach. We propose HAN-Mamba, a hybrid revision of HAN-T that allocates each mechanism where its inductive bias is warranted. The three scalespecific encoders become stacks of Mamba blocks read out through their final recurrent state, which removes positional embeddings and the classification token entirely. The fuser remains a small Transformer: its input is a set of three scale summaries with no canonical ordering, so a causal scan would impose an arbitrary bias, while at length three the quadratic cost objection is vacuous and the attention weights stay directly interpretable. This selective placement follows the design logic of recent hybrid attention-SSM stacks [20], applied here at the granularity of architectural roles rather than interleaved layers. Our contributions are summarized as follows:

1. We introduce HAN-Mamba, a hierarchical hybrid in which selective state space encoders with final-state readout replace the Transformer encoders of the conference architecture, and we motivate the swap through the correspondence between selective forgetting and regime-dependent volatility memory (Sect. 4.2).

2. We give a set-versus-sequence analysis of cross-scale fusion, arguing that attention is the correct operator for the three-token fusion stage, and we validate the analysis with an ablation over four fusion designs, including order-sensitivity of recurrent fusers (Sect. 5.4).

3. We exploit the linear-time encoders to extend the high-frequency context from 60 to 240 buckets, a regime the conference architecture could not usefully reach, and we characterize accuracy and cost along this axis (Sects. 5.3 and 5.5).

4. We strengthen the econometric grounding of the framework by connecting the three-stream hierarchy to the heterogeneous-horizon cascade of the HAR model and to the stylized facts of volatility, positioning the architecture within both literatures (Sects. 2 and 6).

Relation to the Conference Version. This article extends [14]. The feature engineering pipeline, the multi-scale stream construction, the evaluation protocol, and the baseline suite are inherited from the conference version and are summarized here in condensed and rewritten form, with the original paper as the reference for full detail. New material comprises the selective state space background (Sect. 3.2), the HAN-Mamba architecture and its design analysis (Sect. 4.2), the treatment of padding under selective gating (Sect. 4.3), the Flat Mamba baseline, the context-length scaling study (Sect. 5.3), the fusion and encoder-placement ablations (Sect. 5.4), the computational eficiency analysis (Sect. 5.5), the extended related work on state space models, and the discussion of Sect. 6. All experimental results for HAN-Mamba and its variants are new.

## 2 Related Work

Econometric Volatility Modeling. The ARCH and GARCH families model conditional variance as a function of past squared innovations [16, 7] and remain the reference point for parametric volatility analysis [15]. The availability of intraday data shifted attention to realized measures, which estimate latent volatility by summing squared high-frequency returns [5, 9]. Within this line, the HAR model of Corsi [11] regresses future realized volatility on daily, weekly, and monthly averages, encoding the heterogeneous-horizon structure of market participants in a fixed linear cascade. The three-stream hierarchy studied here can be read as a learned, nonlinear counterpart of that cascade, a connection the conference version left implicit. Stylized facts such as volatility clustering and slowly decaying autocorrelations [10] motivate sequence operators with persistent but fading memory.

Deep Learning for Volatility Forecasting. DeepVol applies dilated causal convolutions directly to high-frequency returns and reports substantial gains over GARCH, HAR, and recurrent baselines on NASDAQ-100 data [23]. Transformerbased models incorporating on-chain metrics improve short-term volatility prediction in cryptocurrency markets [26], and hybrid pipelines that inject textderived sentiment into recurrent forecasters show market-dependent gains [25]. Systematic comparisons find that deep architectures outperform GARCH variants across realized measures and windowing choices, with the margin depending on the variance estimator [3]. These works operate at a single temporal resolution or fuse heterogeneous inputs by concatenation, whereas our framework makes the resolution structure explicit in the architecture.

Attention Architectures for Time Series. Self-attention [29] underlies most recent forecasting architectures. Informer sparsifies attention to push horizon length [32], iTransformer inverts the token dimension to attend across variates [21], and hierarchical auxiliary objectives improve long-horizon stability across backbones [28]. Linear baselines remain competitive on several benchmarks [31], and IO-aware kernels reduce but do not remove the quadratic footprint [12]. Our conference model belongs to this family, and this article targets precisely the component where quadratic attention is least justified, the long single-stream encoders.

State Space Sequence Models. Structured state space models parameterize a linear time-invariant recurrence whose convolution form supports parallel training [18], with subsequent simplifications improving stability and throughput [27]. Mamba makes the state transition input-dependent through a selection mechanism and pairs it with a hardware-aware scan, matching Transformer quality on language at linear cost [17]. The state space duality framework tightens the theoretical and practical connection between the two operator families [13], and hybrid stacks such as Jamba interleave attention layers sparsely within an SSM backbone [20]. We adopt the complementary allocation strategy at the level of architectural roles: recurrence for long homogeneous streams, attention for small heterogeneous sets.

State Space Models for Forecasting. Mamba-based forecasters are an active line of work. TimeMachine composes four Mamba modules over coarse and fine views for long-term multivariate forecasting [2], and a systematic study finds

Mamba variants competitive with strong Transformer forecasters across standard benchmarks [30]. These studies target generic long-horizon benchmarks with homogeneous inputs. Selective state space encoders have not been examined for high-frequency realized volatility prediction, nor within a hierarchical multistream design whose fusion stage is itself a learned component. This article provides that study.

## 3 Preliminaries

## 3.1 Problem Formulation and Data

The task is to predict the realized volatility of a stock over the next 10-minute interval from market data available up to the current time t. For a bucket containing M one-second log-returns $r _ { t , j } = \log P _ { t , j } - \log P _ { t , j - 1 }$ , realized volatility is the square root of the realized variance [5, 9]:

$$
\mathrm { R V } _ { t } = \Big ( \sum _ { j = 1 } ^ { M } r _ { t , j } ^ { 2 } \Big ) ^ { 1 / 2 } .\tag{1}
$$

Compared with squared daily returns, this estimator exploits the intraday information set and yields a far less noisy ex-post target.

We use the ORVP dataset, a public high-frequency benchmark containing order book snapshots and trade records for a cross-section of equities, organized in 10-minute windows indexed by a stock identifier and a time identifier. Each sample requires forecasting the realized volatility of the following window. Dataset composition, preprocessing, and the target construction follow the conference version [14].

## 3.2 Selective State Space Models

State space models describe a sequence-to-sequence map through a latent state $h ( \tau ) \in \overline { { \mathbb { R } } } ^ { N }$ obeying a linear ordinary diferential equation,

$$
h ^ { \prime } ( \tau ) = A h ( \tau ) + B x ( \tau ) , \qquad y ( \tau ) = C h ( \tau ) ,\tag{2}
$$

with $A \in \mathbb { R } ^ { N \times N } , B \in \mathbb { R } ^ { N \times 1 } , C \in \mathbb { R } ^ { 1 \times N }$ applied channelwise. Discretizing with step ∆ under a zero-order hold gives the recurrence

$$
h _ { k } = \bar { A } h _ { k - 1 } + \bar { B } x _ { k } , \qquad y _ { k } = C h _ { k } ,\tag{3}
$$

with $\bar { A } = \exp ( \varDelta A )$ and $\bar { B } = ( \varDelta A ) ^ { - 1 } ( \exp ( \varDelta A ) - I ) \varDelta B$ . With fixed $( A , B , C , \varDelta )$ the system is linear time-invariant and can be trained as a long convolution, the route taken by S4 and its successors [18, 27]. Mamba [17] instead makes $B , C ,$ and ∆ functions of the current input $x _ { k }$ . This selection mechanism breaks time invariance, so the convolutional form is lost, but a hardware-aware parallel scan retains $O ( L )$ training cost, and inference reduces to one update of Eq. (3) per incoming token. A Mamba block wraps the selective SSM in a gated expansion with a short causal convolution, in the style of gated linear units.

Two properties of this operator matter for volatility. First, the spectrum of $\bar { A } = \exp ( \varDelta A )$ lies inside the unit circle for stable parameterizations, so past information decays multiplicatively, mirroring the persistent but fading autocorrelation of realized volatility [10, 11]. Second, the input dependence of $\varDelta _ { k }$ modulates the efective decay per token: a large step contracts exp( $\varDelta _ { k } A )$ and resets the state on informative inputs, while a step near zero passes the state through unchanged. The operator can therefore hold memory through quiet regimes and discount it sharply at shocks, a behavior attention can only emulate through position-by-position score patterns recomputed at every step.

## 4 Methodology

## 4.1 Feature Pipeline and Multi-Scale Streams

The input representation is inherited unchanged from the conference version, which documents it in full [14]. We summarize its three stages. Microstructural extraction computes, per 10-minute bucket, liquidity and price-pressure descriptors from the top two order book levels, including the weighted average price

$$
\mathrm { W A P } = \frac { P _ { 1 } ^ { \mathrm { b i d } } S _ { 1 } ^ { \mathrm { a s k } } + P _ { 1 } ^ { \mathrm { a s k } } S _ { 1 } ^ { \mathrm { b i d } } } { S _ { 1 } ^ { \mathrm { b i d } } + S _ { 1 } ^ { \mathrm { a s k } } } ,\tag{4}
$$

log-returns on WAP and quote prices [8], spread and imbalance measures, and trade activity statistics, aggregated over the full bucket and over receding subwindows. Relational construction identifies market analogues along two axes, cross-sectional neighbors of a stock at the same time identifier under Canberra and Mahalanobis distances and longitudinal neighbors of the same stock through its own history under the Manhattan distance [1], and appends aggregate and rank-normalized statistics of these neighbor sets. Final transformations apply cross-sectional ranking, logarithmic compression of heavy-tailed counts, and a three-dimensional stock embedding obtained from latent Dirichlet allocation [6] over volatility profiles. The result is a 604-dimensional feature vector per sample.

Each sample is then expanded into three time-synchronized streams: $X _ { \mathrm { s h o r t } } \in$ $\mathbb { R } ^ { L _ { s } \times 6 0 4 }$ , the most recent $L _ { s }$ feature vectors at the 10-minute resolution, $X _ { \mathrm { m i d } } \in$ $\mathbb { R } ^ { 2 0 \times 1 }$ , the last 20 daily aggregated realized volatilities, and $X _ { \mathrm { l o n g } } \in \mathbb { R } ^ { 1 2 \times 1 }$ , the last 12 weekly aggregates. The conference version fixed $L _ { s } = 6 0$ , roughly ten trading hours, a value set by the memory footprint of the attention encoder rather than by the data. In this article $L _ { s }$ becomes an experimental axis, $L _ { s } \in$ {60, 120, 240}, with the largest setting covering approximately one trading week of high-frequency context (Sect. 5.3).

## 4.2 The HAN-Mamba Architecture

HAN-Mamba retains the macro-structure of the conference architecture, three scale-specific encoders, a common projection, a learned fuser, and a regression head, and re-implements the encoders as selective state space stacks. Figure 1 gives the overview.

![](images/8e29da7cdaa33b3051192dc7452b91c65bf57b51827f2e62b2083b6b546f6457.jpg)  
Fig. 1. The HAN-Mamba architecture. Adapted from the conference version of this work [14]: the scale-specific Transformer encoders are replaced by asymmetric selective state space encoders read out through their final recurrent state $h ^ { ( \check { L } ) }$ , while the threetoken attention fuser, mean pooling, and regression head are retained.

Selective Scale Encoders. Each stream passes through an independent encoder consisting of a linear embedding into the scale-specific width d followed by a stack of pre-normalized Mamba blocks with residual connections. The asymmetric capacity allocation of the conference version is preserved as a form of architectural regularization: the short encoder uses 4 blocks at d=128 with state size N=16, the mid encoder 2 blocks at d=64 with N=16, and the long encoder a single block at d=32 with N=8, all with expansion factor 2 and causal convolution width 4. Encoders do not share weights, so each specializes to the statistics of its resolution.

Two Transformer-specific components disappear in this swap. Learned absolute positional embeddings are unnecessary because the recurrence processes tokens in order, and order awareness is intrinsic to the operator rather than learned per position. The classification token is replaced by final-state readout: the encoder output is the representation at the last position, which under a causal recurrence is a function of the entire stream and plays exactly the summary role the classification token was trained to approximate. Readout at the final position also aligns the representation with the most recent market state, the natural anchor for a one-step forecast.

Why the Fuser Remains Attention. The fusion stage receives the three encoder summaries, projected to a common width of 128 and stacked as a length-3 sequence, and must model their context-dependent interaction. We deliberately do not extend the Mamba swap to this stage, for three reasons. First, the input is a set, not a sequence: $\{ h _ { \mathrm { s h o r t } } , h _ { \mathrm { m i d } } , h _ { \mathrm { l o n g } } \}$ has no canonical causal order, and a recurrent scan would impose one arbitrarily. Attention without positional encodings is permutation-equivariant and respects the set structure. Second, the quadratic cost argument that motivates the encoder swap is vacuous at length three, where all-pairs interaction costs nine score evaluations. Third, the fuser attention weights over three tokens are directly interpretable as scale importances, a property we exploit in Sect. 6. The empirical ablation in Sect. 5.4 supports this analysis: recurrent fusers are both worse and sensitive to the arbitrary scale ordering. The fused representation is the mean of the three output tokens of a 2-layer, 4-head Transformer encoder, as in the conference version.

Regression Head. The 128-dimensional fused vector passes through layer normalization and a two-hidden-layer perceptron with GELU activations and dropout, contracting through widths 64 and 32 to a scalar prediction of log realized volatility. The head is unchanged from the conference version, which keeps the comparison between encoder families clean.

## 4.3 Padding Under Selective Gating

Histories shorter than the nominal stream length are left-padded with zero vectors. The conference version reported that leaving these positions unmasked outperformed explicit attention masking, and attributed the efect to a learned null-token behavior and to sequence-length information carried by positional embeddings [14]. Under selective state space encoders the same design choice acquires a more direct mechanism. Because the step parameter $\varDelta _ { k }$ is a function of the input token, the encoder can drive $\varDelta _ { k }$ toward zero on all-zero inputs, in which case $\exp ( \varDelta _ { k } A ) \approx I$ and the state passes through the padded prefix unchanged. Selectivity thus implements null-token skipping natively, without a mask and without spending positional capacity on it. We therefore keep unmasked padding, which also preserves protocol comparability with the conference results.

## 4.4 Training Protocol

Training follows the conference protocol exactly [14], so that every performance diference is attributable to the architecture. The model predicts log-transformed realized volatility and is trained to minimize the root mean squared percentage error after exponentiation,

$$
\mathcal { L } _ { \mathrm { R M S P E } } = \Big ( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \big ( ( y _ { i } - e ^ { \hat { y } _ { i } ^ { \mathrm { l o g } } } ) / y _ { i } \big ) ^ { 2 } \Big ) ^ { 1 / 2 } ,\tag{5}
$$

the oficial metric of the benchmark and an appropriate loss for a positive, rightskewed target [24]. Optimization uses AdamW [22] at learning rate $1 0 ^ { - 4 }$ with weight decay $1 0 ^ { - 5 }$ , a one-epoch linear warmup followed by cosine annealing, gradient norm clipping at 1.0, mixed precision, and batch size 4096. Features are standardized with statistics fitted on each training fold.

Table 1. Forecasting performance, per-fold RMSPE under time-aware five-fold crossvalidation, lower is better. Rows marked [1a] are reproduced from the conference version of this work [14]. Best per column in bold
<table><tr><td>Model</td><td>Fold 1</td><td>Fold 2</td><td>Fold 3</td><td>Fold 4</td><td>Fold 5</td><td>Mean</td><td>Std</td></tr><tr><td>GARCH(1,1) [1a]</td><td>0.2753</td><td>0.2911</td><td>0.2805</td><td>0.3051</td><td>0.2745</td><td>0.2853</td><td>0.0118</td></tr><tr><td>LightGBM [1a]</td><td>0.2150</td><td>0.2112</td><td>0.2046</td><td>0.2144</td><td>0.2206</td><td>0.2132</td><td>0.0053</td></tr><tr><td>LightGBM + Optuna [1a]</td><td>0.2083</td><td>0.2049</td><td>0.2020</td><td>0.2097</td><td>0.2130</td><td>0.2076</td><td>0.0044</td></tr><tr><td>Flat Transformer [1a]</td><td>0.2039</td><td>0.1972</td><td>0.1925</td><td>0.2027</td><td>0.1981</td><td>0.1989</td><td>0.0041</td></tr><tr><td>HAN-T [1a]</td><td>0.1971</td><td>0.1939</td><td>0.1948</td><td>0.2009</td><td>0.1958</td><td>0.1965</td><td>0.0025</td></tr><tr><td>Flat Mamba</td><td>0.2012</td><td>0.1955</td><td>0.1938</td><td>0.2001</td><td>0.1949</td><td>0.1971</td><td>0.0033</td></tr><tr><td>HAN-Mamba (ours)</td><td>0.1949</td><td>0.1921</td><td>0.1928</td><td>0.1978</td><td>0.1936</td><td>0.1942</td><td>0.0022</td></tr><tr><td>HAN-Mamba240 (ours)</td><td>0.1934</td><td>0.1907</td><td>0.1911</td><td>0.1962</td><td>0.1921</td><td>0.1927</td><td>0.0022</td></tr></table>

Generalization is estimated with the time-aware five-fold scheme of the conference version. Unique time identifiers are sorted chronologically and split into five contiguous blocks, each block serving once as the validation period, which prevents lookahead leakage at the block level and keeps every time identifier on a single side of the split. Early stopping selects the best checkpoint per fold. All reported figures are means and standard deviations across the five folds.

## 5 Experiments

## 5.1 Baselines

We compare against the full baseline suite of the conference version, whose results are reproduced for reference, plus one new architectural control. GARCH(1,1) [7] is fitted per stock on daily WAP log-returns and scaled to the 10-minute horizon, using price history only. LightGBM [19] consumes the flattened 604-dimensional feature vector and represents the strongest tabular paradigm on this benchmark, reported both with fixed hyperparameters and with per-fold Optuna tuning [4]. The Flat Transformer concatenates the three streams into a single sequence processed by an encoder matched in capacity to the short-scale encoder of HAN-T, isolating the value of the hierarchy under attention. The new Flat Mamba baseline applies the same flattening but processes the concatenated sequence with a Mamba stack matched to the short-scale encoder of HAN-Mamba, isolating the value of the hierarchy under selective state spaces. HAN-T is the conference architecture [14].

## 5.2 Main Results

Table 1 reports per-fold RMSPE for all models at the matched context length $L _ { s } = 6 0$ , together with the extended-context configuration HAN-Mamba discussed in Sect. 5.3. We draw four conclusions.

Table 2. Efect of the high-frequency context length $L _ { s } ,$ mean ± std RMSPE over five folds. The $L _ { s } { = } 6 0$ entry for HAN-T is from the conference version [14]
<table><tr><td>Model</td><td> $L _ { s } = 6 0$ </td><td> $L _ { s } = 1 2 0$ </td><td> $L _ { s } = 2 4 0$ </td></tr><tr><td>HAN-T</td><td> $0 . 1 9 6 5 \pm 0 . 0 0 2 5$ </td><td> $0 . 1 9 5 9 \pm 0 . 0 0 2 7$ </td><td> $0 . 1 9 6 3 \pm 0 . 0 0 3 1$ </td></tr><tr><td>HAN-Mamba</td><td> $0 . 1 9 4 2 \pm 0 . 0 0 2 2$ </td><td> $0 . 1 9 3 1 \pm 0 . 0 0 2 1$ </td><td> $\mathbf { 0 . 1 9 2 7 \ : \pm { \ : 0 . 0 0 2 2 } }$ </td></tr></table>

1. The encoder swap improves accuracy at matched context. HAN-Mamba reduces mean RMSPE over HAN-T (0.1942 vs. 0.1965, a relative gain of 1.2%) under an identical feature set, training protocol, and validation scheme, and does so with 33% fewer parameters (Sect. 5.5). The gain therefore reflects a better-matched inductive bias rather than added capacity.

2. The hierarchical prior transfers across operator families. Flat Mamba improves on the Flat Transformer (0.1971 vs. 0.1989) but remains behind both hierarchical models. The gap between HAN-Mamba and Flat Mamba (0.1942 vs. 0.1971) closely mirrors the conference-version gap between HAN-T and the Flat Transformer (0.1965 vs. 0.1989), indicating that the multi-scale decomposition contributes value independently of the sequence operator that consumes it.

3. Stability improves alongside accuracy. HAN-Mamba attains the lowest cross-fold standard deviation of all $L _ { s } ~ = ~ 6 0$ models (0.0022), continuing the pattern observed in the conference version, where the hierarchical prior narrowed dispersion across market periods.

4. Linear-time encoders convert directly into accuracy headroom. Extending the high-frequency context to 240 buckets yields a further reduction to 0.1927, a cumulative 1.9% relative improvement over HAN-T. This configuration is analyzed next.

## 5.3 Context-Length Scaling

The conference architecture fixed $L _ { s } = 6 0$ because the attention encoder made longer contexts costly, not because the data ceased to inform. Volatility autocorrelations decay slowly [10, 11], so history beyond ten trading hours plausibly carries signal. Table 2 and Fig. 2 report both architectures trained at $L _ { s } \in \{ 6 0 , 1 2 0 , 2 4 0 \}$ under the standard protocol.

HAN-T is non-monotone: a mild gain at $L _ { s } = 1 2 0 \ ( 0 . 1 9 5 9 )$ is followed by regression at $L _ { s } = 2 4 0 \ ( 0 . 1 9 6 3 )$ , alongside sharply growing cost (Sect. 5.5). Two mechanisms plausibly compound here. Learned absolute positional embeddings must be estimated afresh for every length, and at 240 positions the embedding table for the padded early history dilutes the signal available per position. Attention over longer windows also spreads probability mass across more mostly redundant keys, and the classification token must compress an increasingly heterogeneous window through a single set of scores. HAN-Mamba improves monotonically (0.1942, 0.1931, 0.1927), with diminishing returns consistent with the decaying information content of older buckets. The recurrent parameterization is length-agnostic, and its multiplicative state decay handles stale context by construction. We adopt $\mathrm { H A N  – M a m b a _ { 2 4 0 } }$ as the headline configuration and retain HAN-Mamba at $L _ { s } = 6 0$ for matched-budget comparisons.

![](images/02d67f9a42088f21a4e79251f53c884c894e3f7e4531a9bd3e31fd2ad329d690.jpg)  
High-frequency context length $L _ { s }$ (10-minute buckets)  
Fig. 2. Context-length scaling. The attention encoders saturate and then regress as $L _ { s }$ grows, while the selective state space encoders improve monotonically with diminishing returns

Table 3. Ablations, mean RMSPE over five folds at $L _ { s } { = } 6 0$ . The HAN-T reference and its average-fuser variant are from the conference version [14]
<table><tr><td>Group</td><td>Variant</td><td>Mean RMSPE</td></tr><tr><td rowspan="4">Fusion design</td><td>Attention fuser (HAN-Mamba default)</td><td>0.1942</td></tr><tr><td>Mamba fuser, order long→mid→short</td><td>0.1958</td></tr><tr><td>Mamba fuser, order short→mid→long</td><td>0.1963</td></tr><tr><td>Non-learnable average fuser</td><td>0.1966</td></tr><tr><td rowspan="4">Encoder placement</td><td>Mamba on all three streams (HAN-Mamba default)</td><td>0.1942</td></tr><tr><td>Mamba on short stream only</td><td>0.1949</td></tr><tr><td>Mamba on mid and long streams only</td><td>0.1961</td></tr><tr><td>No swap, HAN-T reference [14]</td><td>0.1965</td></tr></table>

## 5.4 Ablation Studies

Table 3 quantifies the two design decisions that define HAN-Mamba, what operator fuses the scales and where the Mamba swap is applied. All variants use $L _ { s } = 6 0$ and the standard protocol.

Fusion Design. Replacing the attention fuser with a Mamba scan degrades performance under both scale orderings (0.1958 and 0.1963 vs. 0.1942), and the gap between the two orderings confirms the set-versus-sequence analysis of Sect. 4.2: the recurrent fuser is sensitive to an ordering that carries no meaning. The non-learnable average is worst (0.1966), reproducing the conference-version finding that learned, context-dependent weighting of the scales matters (0.1983 vs. 0.1965 there). The margin between the attention fuser and the average fuser under HAN-Mamba (0.0024) is comparable to the corresponding margin under HAN-T (0.0018), so the value of learned fusion is preserved across encoder families.

Table 4. Eficiency at $L _ { s } { = } 6 0$ and $L _ { s } \mathrm { = } 2 4 0$ . FLOPs are per forward sample. Peak memory is per GPU during training. Streaming latency is the time to update one prediction when a new 10-minute bucket arrives at inference
<table><tr><td></td><td></td><td colspan="2">GFLOPs</td><td colspan="2">Epoch time (s)1</td><td colspan="2">Peak mem. (GB)</td></tr><tr><td>Model</td><td>Params</td><td> $L _ { s } { = } 6 0$ </td><td> $L _ { s } \mathrm { = } 2 4 0$ </td><td> $L _ { s } { = } 6 0$ </td><td> $L _ { s } \mathrm { = } 2 4 0$ </td><td> $L _ { s } { = } 6 0$ </td><td> $L _ { s } \mathrm { = } 2 4 0$ </td></tr><tr><td>HAN-T</td><td>1.43M</td><td>0.16</td><td>0.67</td><td>41</td><td>152</td><td>5.8</td><td>18.9</td></tr><tr><td>HAN-Mamba</td><td>0.96M</td><td>0.06</td><td>0.23</td><td>33</td><td>78</td><td>4.1</td><td>8.7</td></tr></table>

Encoder Placement. Applying the swap only to the short stream captures most of the benefit (0.1949 vs. 0.1965), which is expected, as the short stream is both the longest sequence and the dominant information source in the conferenceversion ablations. Swapping only the two context streams yields a small gain (0.1961), and the full swap is best (0.1942). The mid and long streams are short enough that the operator choice matters little in isolation, but consistent state space processing across all three streams appears to produce summaries that are easier for the fuser to integrate.

## 5.5 Computational Eficiency

Table 4 characterizes cost. Training measurements use eight NVIDIA A100 40 GB GPUs with the batch partitioned across devices, matching the conference setup, and streaming latency is measured on a single A100 at batch size one.

Three observations follow. First, HAN-Mamba is smaller (0.96M vs. 1.43M parameters, 33% fewer) because a Mamba block at width d spends roughly $6 d ^ { 2 }$ parameters against roughly $4 d ^ { 2 } + 2 d d _ { \mathrm { f f } }$ for an attention layer with the feedforward expansion used here, and because positional embedding tables disappear. Second, the cost gap widens with context, as expected from the linear against quadratic scaling: at $L _ { s } ~ = ~ 2 4 0$ the epoch time ratio grows to 1.9× and the peak memory ratio to 2.2×, and it is the memory column that renders longcontext HAN-T training impractical at the batch sizes this benchmark requires. Third, deployment difers qualitatively. HAN-T must re-encode the full window for every new bucket, at 3.8 ms per prediction, while HAN-Mamba advances its recurrent state in 0.31 ms, a 12× reduction, and the state itself is a few kilobytes per instrument, so thousands of names can be tracked concurrently on one device. We also observe faster convergence, with the early-stopping optimum reached after 24 epochs on average against 31 for HAN-T under the identical schedule.

## 6 Discussion

Selective Memory and the Structure of Volatility. The HAR model [11] explains realized volatility through fixed-weight averages at daily, weekly, and monthly horizons, a parsimonious encoding of heterogeneous participant horizons. The hierarchy inherited from the conference version already generalized the cascade structure by learning nonlinear per-scale summaries. The selective encoders generalize the second ingredient, the memory weights themselves: $\exp ( \varDelta _ { k } A )$ implements a data-dependent forgetting schedule per channel, so the efective lookback contracts in turbulent stretches and dilates in calm ones. The monotone context-scaling curve of Fig. 2 is consistent with this reading, as additional history helps but with returns that decay the way volatility autocorrelations do [5, 10].

What the Fuser Attends To. Averaged over validation folds, the fuser assigns 0.68 of its attention mass to the short-scale token, 0.21 to the mid, and 0.11 to the long. Restricted to the top decile of realized target volatility, the distribution shifts to 0.54, 0.29, and 0.17: under stress the model leans harder on daily and weekly context. This pattern agrees with the conference-version ablation hierarchy, where removing the mid-scale stream cost more than removing the long-scale one, and it gives the practitioner a per-prediction diagnostic of which horizon is driving a forecast.

Limitations. The evidence is confined to a single benchmark, one asset class, and a fixed 10-minute horizon, and the relational features are computed over the full cross-section, which requires care in a live setting where neighbor statistics must be maintained incrementally. Mamba hyperparameters (state size, convolution width, expansion) were adopted from language modeling defaults with light tuning, and horizon-specific tuning may hold further gains. Finally, the streaming latency advantage assumes feature vectors are available at bucket close, so the end-to-end latency budget is dominated by the feature pipeline rather than the model.

## 7 Conclusions

This article revisited the sequence-processing core of our hierarchical volatility forecasting framework and replaced its attention encoders with selective state space encoders, keeping attention only where its inductive bias is warranted, the permutation-structured fusion of three scale summaries. The resulting HAN-Mamba matches the conference architecture on protocol and features, exceeds it on accuracy at matched context (0.1942 vs. 0.1965 mean RMSPE), and converts its linear-time scaling into two capabilities the attention design could not ofer, a usefully longer high-frequency context (0.1927 at 240 buckets) and constanttime streaming updates with a compact per-instrument state. Ablations showed that the hierarchical prior is operator-agnostic, that recurrent fusion of an unordered scale set is both weaker and order-sensitive, and that the full-stack swap outperforms partial ones.

Beyond the specific numbers, the study supports a design principle for financial sequence models: attention and selective recurrence are not competitors but complements, and the allocation should follow the structure of the data, recurrence along long homogeneous time axes, attention across small heterogeneous sets. Future work includes multi-asset joint forecasting with cross-sectional state sharing, distributional and quantile extensions of the regression head, incremental maintenance of the relational features for live deployment, and a bidirectional or state-space-duality treatment of the encoders [13] where strict causality is not required.

Acknowledgments. The authors thank the ICAART 2026 reviewers, whose feedback on the conference version shaped the extensions presented here.

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Aggarwal, C.C.: Data Mining: The Textbook. Springer, Cham (2015). https://doi.org/10.1007/978-3-319-14142-8

2. Ahamed, M.A., Cheng, Q.: TimeMachine: A time series is worth 4 Mambas for long-term forecasting. In: ECAI 2024 – 27th European Conference on Artificial Intelligence. Frontiers in Artificial Intelligence and Applications, vol. 392, pp. 1688– 1695. IOS Press (2024). https://doi.org/10.3233/FAIA240677

3. Akgun, O.B., Gulay, E.: Dynamics in realized volatility forecasting: Evaluating GARCH models and deep learning algorithms across parameter variations. Computational Economics 65(6), 3971–4013 (2025). https://doi.org/10.1007/s10614- 024-10694-2

4. Akiba, T., Sano, S., Yanase, T., Ohta, T., Koyama, M.: Optuna: A nextgeneration hyperparameter optimization framework. In: Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. pp. 2623–2631. ACM (2019). https://doi.org/10.1145/3292500.3330701

5. Andersen, T.G., Bollerslev, T., Diebold, F.X., Labys, P.: Modeling and forecasting realized volatility. Econometrica 71(2), 579–625 (2003). https://doi.org/10.1111/1468-0262.00418

6. Blei, D.M., Ng, A.Y., Jordan, M.I.: Latent Dirichlet allocation. Journal of Machine Learning Research 3, 993–1022 (2003)

7. Bollerslev, T.: Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics 31(3), 307–327 (1986). https://doi.org/10.1016/0304- 4076(86)90063-1

8. Bouchaud, J.P., Bonart, J., Donier, J., Gould, M.: Trades, Quotes and Prices: Financial Markets Under the Microscope. Cambridge University Press, Cambridge (2018). https://doi.org/10.1017/9781316659335

9. Bucci, A.: Forecasting realized volatility: A review. Journal of Advanced Studies in Finance 8(2), 94–138 (2017)

10. Cont, R.: Empirical properties of asset returns: Stylized facts and statistical issues. Quantitative Finance 1(2), 223–236 (2001). https://doi.org/10.1080/713665670

11. Corsi, F.: A simple approximate long-memory model of realized volatility. Journal of Financial Econometrics 7(2), 174–196 (2009). https://doi.org/10.1093/jjfinec/nbp001

12. Dao, T., Fu, D.Y., Ermon, S., Rudra, A., Ré, C.: FlashAttention: Fast and memoryeficient exact attention with IO-awareness. In: Advances in Neural Information Processing Systems. vol. 35, pp. 16344–16359 (2022)

13. Dao, T., Gu, A.: Transformers are SSMs: Generalized models and eficient algorithms through structured state space duality. In: Proceedings of the 41st International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 235, pp. 10041–10071. PMLR (2024)

14. Deaconu, M.B., Pop, I.D.: Hierarchical attention networks for multi-scale financial volatility forecasting. In: Proceedings of the 18th International Conference on Agents and Artificial Intelligence (ICAART 2026), Volume 2. pp. 1297–1305. SCITEPRESS (2026). https://doi.org/10.5220/0014264900004052

15. Dhingra, B., Batra, S., Aggarwal, V., Yadav, M., Kumar, P.: Stock market volatility: A systematic review. Journal of Modelling in Management 19(3), 925–952 (2024). https://doi.org/10.1108/JM2-04-2023-0080

16. Engle, R.F.: Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. Econometrica 50(4), 987–1007 (1982). https://doi.org/10.2307/1912773

17. Gu, A., Dao, T.: Mamba: Linear-time sequence modeling with selective state spaces. In: First Conference on Language Modeling (COLM) (2024)

18. Gu, A., Goel, K., Ré, C.: Eficiently modeling long sequences with structured state spaces. In: International Conference on Learning Representations (2022)

19. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., Liu, T.Y.: Light-GBM: A highly eficient gradient boosting decision tree. In: Advances in Neural Information Processing Systems. vol. 30, pp. 3146–3154 (2017)

20. Lieber, O., Lenz, B., Bata, H., Cohen, G., Osin, J., Dalmedigos, I., Safahi, E., Meirom, S., Belinkov, Y., Shalev-Shwartz, S., Abend, O., Alon, R., Asida, T., Bergman, A., Glozman, R., Gokhman, M., Manevich, A., Ratner, N., Rozen, N., Shwartz, E., Zusman, M., Shoham, Y.: Jamba: A hybrid Transformer-Mamba language model. arXiv preprint arXiv:2403.19887 (2024)

21. Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., Long, M.: iTransformer: Inverted Transformers are efective for time series forecasting. In: International Conference on Learning Representations (2024)

22. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: International Conference on Learning Representations (2019)

23. Moreno-Pino, F., Zohren, S.: DeepVol: Volatility forecasting from high-frequency data with dilated causal convolutions. Quantitative Finance 24(8), 1105–1127 (2024). https://doi.org/10.1080/14697688.2024.2387222

24. Murphy, K.P.: Probabilistic Machine Learning: Advanced Topics. MIT Press, Cambridge, MA (2023)

25. Ncume, V., van Zyl, T.L., Paskaramoorthy, A.: Volatility forecasting using deep learning and sentiment analysis. arXiv preprint arXiv:2210.12464 (2022)

26. Rafi, M.A., Shaboj, S.M.I., Rasul, I., Miah, M.K., Islam, M.R., Ahmed, A.: Cryptocurrency volatility forecasting using transformer-based deep learning models and on-chain metrics. Journal of Economics, Finance and Accounting Studies 6(1), 119– 130 (2024). https://doi.org/10.32996/jefas.2024.6.1.12

27. Smith, J.T.H., Warrington, A., Linderman, S.W.: Simplified state space layers for sequence modeling. In: International Conference on Learning Representations (2023)

28. Sun, Y., Xie, Z., Chen, D., Eldele, E., Hu, Q.: Hierarchical classification auxiliary network for time series forecasting. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 39, pp. 20743–20751 (2025). https://doi.org/10.1609/aaai.v39i19.34286

29. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. In: Advances in Neural Information Processing Systems. vol. 30, pp. 5998–6008 (2017)

30. Wang, Z., Kong, F., Feng, S., Wang, M., Yang, X., Zhao, H., Wang, D., Zhang, Y.: Is Mamba efective for time series forecasting? Neurocomputing 619, 129178 (2025). https://doi.org/10.1016/j.neucom.2024.129178

31. Zeng, A., Chen, M., Zhang, L., Xu, Q.: Are Transformers efective for time series forecasting? In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 37, pp. 11121–11128 (2023). https://doi.org/10.1609/aaai.v37i9.26317

32. Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., Zhang, W.: Informer: Beyond eficient Transformer for long sequence time-series forecasting. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 35, pp. 11106–11115 (2021). https://doi.org/10.1609/aaai.v35i12.17325