# PATHFM: FAST GENERATIVE FORECASTING OF MULTIVARIATE TRAJECTORIES

Jiachen Yao<sup>1,∗</sup> Xiyuan Zhang<sup>2,†</sup> Annan Yu<sup>2</sup> Boran Han<sup>2</sup> Abdul Fatir Ansari<sup>2</sup> Shuai Zhang<sup>2</sup> Michael Bohlke-Schneider<sup>2</sup> Bernie Wang<sup>2</sup>

<sup>1</sup>California Institute of Technology <sup>2</sup>Amazon

## ABSTRACT

Time series foundation models are becoming increasingly popular for generalpurpose forecasting. However, most leading models predict marginal quantiles, leaving the dependence structure of future trajectories unspecified. This distinction matters for trajectory-dependent quantities such as cumulative demand, peak load, and event timing, yet point and marginal scores do not reveal it. To model this dependence, we introduce PathFM, a trajectory-based forecasting model that uses flow matching to generate complete future paths. We further distill the model to one or two steps and share history attention cache across draws. Together, these enable PathFM to generate multiple paths for a single series in roughly the time needed for one query to a quantile model. The distilled sampler also makes it practical to optimize a calibration loss directly. PathFM achieves competitive marginal accuracy on the established forecasting benchmarks fev-bench and TIME. To complement these evaluations, we introduce Forecast Path Score Bench (fpsbench), which augments fev-bench tasks with energy and variogram scores and downstream metrics. PathFM achieves a higher aggregate path score on fps-bench than all evaluated external baselines. Experiments on chaotic systems further show that our flow model sustains long-run variability, while strong baselines rapidly collapse. PathFM makes generative time series forecasting both competitive and practical, enabling diverse downstream applications.

## 1 INTRODUCTION

Time series foundation models (TSFMs) are a hot topic (Ansari et al., 2024; Das et al., 2023; Ansari et al., 2025; Jain & Sen, 2026; Cohen et al., 2025). The field is growing fast: Every year, new models are released that refresh the leaderboard. The models are commonly evaluated through point accuracy (e.g., MASE) and marginal probabilistic scores (e.g., CRPS, scaled quantile loss) (Aksu et al., 2024; Shchur et al., 2025). Consequently, most of the recent TSFMs are designed and trained to produce quantile levels, which are optimized for these benchmarks.

Point accuracy and marginal probabilistic scores are widely adopted for good reason: they provide an interpretable, per-step summary of forecast quality that directly supports common operational decisions such as inventory planning, resource allocation, and risk management. However, they do not capture all aspects of forecast quality. Other properties, particularly trajectory-dependent ones, are important but overlooked by these metrics. Among many, we illustrate two important aspects:

• Learning distributions with multiple modes: Quantiles are defined through the inverse of the cumulative distribution function (CDF) of the target distribution. These CDFs, however, are not always nice and smooth, for instance, when the target distribution has multiple modes or sharp peaks. With a few or a few dozens of quantiles, the resulting approximation of the CDF can be quite rough and suffer from a regression-to-the-mean issue (Yu et al., 2026). See the Lorenz oscillator example in Figure 1.

• Learning autocorrelation: Many quantile regression models predict marginal quantiles at individual time steps across a forecast window. These marginal outputs do not specify temporal dependence among future values within that window. For instance, many forecasting problems also require learning trajectory-dependent quantities, such as cumulative demand, peak load, and threshold-crossing times (Perez-Diaz et al., 2025; Cai et al., 2026). See the solar-output example in Figure 1. A marginal quantile loss does not identify these joint properties.

Lorenz: long-term dynamics

![](images/4d311e399a8ec384ef344f85946ea2c20d94f555415b8e9b7c22f8cb8ba3534b.jpg)

![](images/c0b37d3212ae29510443de025ea8d998d82ec2d3c1788640ed5cbd673a34c9ae.jpg)

![](images/2992308336c2ccb726c5b94c476237d50f597d6536242c443bb47c2e98e19a59.jpg)  
Solar: daily-peak uncertainty

![](images/ab8edb36801ff58f58a2fca5d236cc341a0dcfd983cb1991514c035abc45c484.jpg)

![](images/702b7f6263db44f1912b5b1cec0aad735098f8cb4ef06945250d4a32d2a13aae.jpg)

![](images/5825676b4b1e94dbad68f7f3c71d699af7665b01a37db626e1020aaa93669a32.jpg)

![](images/55fdd4db971f2a62755179f7adcfe90c4481ec5361709eaafd5165b954ba0a3d.jpg)  
Figure 1: Two advantages of trajectory generation. Left: a sampled Lorenz rollout sustains variability and a two-lobe structure, while the fine-tuned Chronos-2 trace contracts. Both use the dynamical models in Section 6. Gray Lorenz curves show the observed continuation. Right: solar-output paths from PathFM and independent (ind.) and comonotone (com.) completions of the Chronos-2 (C2) quantiles. Horizontal dashed lines marks each solar path’s daily maximum; black denotes the observed future. In (g), boxes show medians and interquartile ranges across 32 paths, whiskers show the 10–90% range, and the dashed line marks the observed maximum.

Ad hoc methods have been proposed to mitigate these weaknesses, such as completing marginal quantiles under independence or comonotone assumptions (Perez-Diaz et al., 2025), or learning copula structures on top of marginal forecasts (Drouin et al., 2022; Ashok et al., 2024). On the other hand, diffusion- or flow-matching-based generative time-series models are naturally well-suited for capturing joint trajectory distributions even without these ad hoc methods, as they can generate complete future paths while maintaining temporal and cross-variable dependencies. Unfortunately, existing generative models (Kollovieh et al., 2025; Liu et al., 2025) achieve neither high marginal accuracy nor efficient sampling, limiting their practical utility in real-world forecasting scenarios. This leads us to ask:

## Can we construct a generative time-series model that achieves both high marginal accuracy and efficient sampling while also preserving trajectory dependence?

We introduce PathFM to combine joint and multi-mode trajectory modeling with accurate marginal forecasting and efficient sampling. The model learns temporal and cross-variable dependence through conditional flow matching (Lipman et al., 2022), generating complete future paths conditioned on history and known future covariates. Parallel decoding distillation (Shaul et al., 2026) and history attention caching enable faster sampling with one or two model evaluations. To narrow the marginal accuracy gap, we add a calibration loss that directly optimizes fair CRPS on generated forecasts. The distilled sampler makes this optimization practical by allowing calibration gradients to pass through its short rollout and refine the forecast distribution.

With these components, PathFM-300M ranks second on the widely used fev-bench for marginal scores (Section 5).<sup>1</sup> On Forecast Path Score Bench (fps-bench), which we curate to evaluate trajectory quality, PathFM ranks first for path scores (Section 5). In a separate evaluation on chaotic systems, our sampling approach improves point forecast accuracy upon the state-of-the-art PANDA model, captures useful temporal dependence, and preserves long-run variability more faithfully than baselines (Section 6). These results demonstrate that PathFM generates high-fidelity trajectory forecasts without sacrificing marginal accuracy, making it a strong candidate for forecasting applications.

Our contributions are as follows:

• Fast and calibrated trajectory generation. We introduce PathFM, a joint flow foundation model for fast, calibrated trajectory generation with accurate small-ensemble marginals (Figure 2). Distillation enables one- or two-step generation, marginal-CRPS refinement improves empirical coverage, and stratified sampling reduces finite-ensemble marginal error.

• A benchmark for trajectory distributions. We introduce fps-bench, to our knowledge the first dedicated benchmark for evaluating path distributions from time series foundation models on real-world data. It reveals how strong marginal accuracy in existing models can mask errors in temporal dependence and uncertainty over trajectory-dependent quantities.

• State-of-the-art performance. PathFM outperforms leading foundation models on downstream probabilistic forecasting tasks and stays among the top on marginal benchmarks. It also achieves the highest forecasting accuracy across strong baselines on chaotic systems.

## 2 PROBLEM SETUP AND PRELIMINARIES

Forecast object and downstream query. Let $H \in \mathbb { R } ^ { T \times C }$ be the observed history and $Y \in \mathbb { R } ^ { L \times C }$ the future. The available information I includes H, observation masks, and known future covariates. A predictive law $P _ { \theta } ( Y \mid { \mathcal { T } } )$ supports queries about individual coordinates or a trajectory functional $g ( \bar { Y } )$ , such as cumulative sum or autocorrelation. A trajectory generator represents this law through $\dot { X } = \dot { G } _ { \theta } (  { \mathcal { T } } , Z )$ , where each noise draw Z produces one complete future path.

Perez-Diaz et al. (2025) distinguish point, quantile, parametric-distribution, and trajectory forecast products by the decisions they support. Coordinatewise quantiles $Q _ { \tau } ( Y _ { \ell , c } \mid \mathcal { T } )$ and marginal distribution parameters leave temporal and cross-variable dependence unspecified. Consequently, they do not determine predictive autocorrelation or distributions of sums, maxima, and path-dependent events without an additional dependence assumption. Joint trajectory samples support direct estimation of those quantities by applying g to each generated path. Table 1 compares forecast objects, computational costs, and task compatibility; Section A discusses related work.

Table 1: Forecast objects, inference cost, and task compatibility of representative models.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Forecast object</td><td rowspan="2">NFE</td><td colspan="4">Task compatibility</td></tr><tr><td>Pretrained Marginal</td><td></td><td>Path</td><td>Cross-variable</td></tr><tr><td>Quantile models</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Chronos-2 (Ansari et al., 2025) Marginal quantiles</td><td></td><td>1</td><td></td><td></td><td>X</td><td>×</td></tr><tr><td>TimesFM-3 (Jain &amp; Sen, 2026)</td><td>Marginal quantiles</td><td>1</td><td></td><td></td><td>X</td><td>X</td></tr><tr><td>Toto 2.0 (Khwaja et al., 2026)</td><td>Marginal quantiles</td><td>1</td><td></td><td></td><td>X</td><td>X</td></tr><tr><td>Flow models</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TSFlow (Kollovieh et al., 2025) Joint paths</td><td></td><td>32</td><td>X</td><td></td><td></td><td></td></tr><tr><td>Sundial (Liu et al., 2025)</td><td>Univariate paths</td><td>50</td><td></td><td></td><td>‡</td><td>X</td></tr><tr><td>PathFM (Ours)</td><td>Joint paths</td><td>1-2</td><td></td><td></td><td></td><td></td></tr></table>

Path and Cross-variable refer to native joint forecast laws. NFE counts evaluations per output block (per sampled path when applicable). <sup>†</sup>TSFlow’s multivariate extension adds attention across features. <sup>‡</sup>Temporal coupling here refers to the block generated by the conditional flow.

Existing benchmarks and their scope. GIFT-Eval reports point-forecast metrics and approximates univariate CRPS with weighted quantile loss (Aksu et al., 2024); fev-bench uses scaled quantile loss across heterogeneous forecasting tasks (Shchur et al., 2025). TIME evaluates zero-shot forecasting on freshly curated datasets, reporting MASE and quantile-based CRPS (Qiao et al., 2026). These summaries assess central predictions and marginal uncertainty, including on multivariate tasks, but cannot distinguish joint laws with identical coordinatewise marginals. We introduce fps-bench to address this gap by evaluating temporal dependence and trajectory-dependent quantities (Section 4). Section C.1 analyzes finite-ensemble approximation, and Sections D.1 and F define the evaluation metrics for general-corpus and chaotic-system prediction, respectively.

![](images/eb4db537d70cc1e56c8bcc7a16bc058c42e0e5fcceaf40cc7185677c7c8f0a9e.jpg)  
Figure 2: Training, post-training, and sampling with PathFM. Flow matching learns a joint velocity field; distillation matches frozen-teacher velocities at student-proposed states. Calibration maps eight iid draws through one fused step of trainable $G _ { \theta }$ to samples scored by fair CRPS, together with PDD supervision. Sampling uses stratified initial noise to improve coverage of the noise distribution with a small ensemble. Below: Flow enters after the flow-independent prefix, and history attention values are computed once and reused by separate packed future states.

## 3 FAST AND CALIBRATED TRAJECTORY FORECASTING

PathFM learns a flow over complete future trajectories, then distills it for fast sampling and refines its marginal scores. Figure 2 summarizes the training and sampling stages.

## 3.1 PRETRAINING BY FLOW MATCHING

Conditional flow matching (Lipman et al., 2022) learns a velocity field that transports source noise toward the conditional law $P _ { \theta } \overset { \cdot } { ( } Y \mid \mathcal { T } )$ over complete $L \times C$ futures. Here I contains the observed history, masks, and known future covariates. We transform targets to Y<sup>¯</sup> using context-derived location and scale followed by the inverse hyperbolic sine (Ansari et al., 2025). We draw $Z$ independently of the future, with the same shape as $\dot { \bar { Y } }$ and independent coordinates $Z _ { \ell c } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ . The mask M selects valid unknown coordinates, $\Pi _ { M }$ restores known values and zeros invalid coordinates, and $\| \cdot \| _ { M } ^ { 2 }$ averages squared error over transported coordinates.

At flow time $t \in [ 0 , 1 ]$ , we train $v _ { \theta }$ on the straight path from noise to data:

$$
\begin{array} { r } { X _ { t } = \Pi _ { M } \big ( ( 1 - t ) Z + t \bar { Y } \big ) , \quad \mathcal { L } _ { \mathrm { F M } } = \mathbb { E } \left[ \big \| v _ { \theta } ( X _ { t } , t ; \mathcal { T } ) - ( \bar { Y } - Z ) \big \| _ { M } ^ { 2 } \right] . } \end{array}\tag{1}
$$

The expectation is over training pairs $( \mathcal { T } , \bar { Y } )$ , source noise, and flow time. Training uses one scalar flow time t per row, shared across all channels and forecast steps (Section B.1).

Inference integrates the ordinary differential equation, yielding $\widehat { X } _ { 1 } .$ , one complete trajectory:

$$
\widehat { X } _ { 0 } = \Pi _ { M } ( Z ) , \qquad \frac { \mathrm { d } \widehat { X } _ { t } } { \mathrm { d } t } = M \odot v _ { \theta } ( \widehat { X } _ { t } , t ; \mathcal { Z } ) , \quad t \in [ 0 , 1 ] .\tag{2}
$$

## 3.2 EFFICIENT BACKBONE FOR JOINT GENERATION

The velocity field uses a transformer based on T5 (Raffel et al., 2020), with separate time and channel axes. Its inputs are patches of 16 values with masks and time indices, a register token, and future patches populated by available covariates. Each block combines temporal self-attention, attention across channels at aligned patches, and a pointwise feed-forward network. We condition flow time via adaptive RMS normalization (Chen et al., 2024) and apply QK normalization and weight decay.

For reuse across paths, late entry separates conditioning from flow-dependent computation. The first e blocks use unrestricted attention over history, the register, and future placeholders, without source noise or flow time. These representations incorporate the requested horizon and known future covariates. Restricting history attention before this stage would exclude that future information. Future positions then receive a gated difference between flow-state and placeholder tokens. In subsequent blocks, history-only self-attention keeps history independent of the sampled path: history tokens attend only to history, while future tokens attend to both history and future. Flow-time modulation acts only on register and future tokens. Section B.1 specifies the configuration.

Packing and caching. Packed execution stores history once, with a separate register/future block per draw and masks preventing communication between paths. Training averages four independent source/time draws per history, giving 4× flow-matching supervision at 1.5× the cost. This sharing across paths is distinct from reusing history states across successive flow evaluations. At inference, the first e blocks run once per history and horizon; their outputs are reused across paths and flow evaluations. The flow model can reuse deeper history keys and values across evaluations. Section 5.3 compares accuracy and inference cost.

## 3.3 POST-TRAINING: DISTILLATION WITH CALIBRATION

Sampling a vanilla flow requires tens of sequential velocity evaluations, each depending on the state produced by the preceding step. Distillation learns a shorter rollout, i.e., one or two evaluations. This makes both ensemble generation and differentiation through generated forecasts practical.

Distillation. Parallel decoding distillation (PDD) (Shaul et al., 2026) groups consecutive flow-time intervals into blocks, predicting all interval velocities in a block from one backbone evaluation. We initialize the student from a flow teacher that remains frozen throughout post-training. We divide [0, 1] into $J = 2 0$ intervals with knots $\tau _ { j } = j / J$ and widths $\Delta _ { j } = \tau _ { j + 1 } - \tau _ { j }$ . Each interval has an affine output head $h _ { \theta , j } = W _ { j } \phi _ { \theta } + b _ { j }$ . All heads share features ϕ<sub>θ</sub> computed at the block’s starting state and time, including nonlinear outputs and residual inputs.

For a block of b intervals starting at s, the mean velocity advances X to $X ^ { + }$ :

$$
\bar { h } _ { \theta , s , b } = \sum _ { j = s } ^ { s + b - 1 } \frac { \Delta _ { j } } { \tau _ { s + b } - \tau _ { s } } h _ { \theta , j } , \quad X ^ { + } = \Pi _ { M } \big ( X + ( \tau _ { s + b } - \tau _ { s } ) \bar { h } _ { \theta , s , b } ( X ) \big ) .\tag{3}
$$

Because the heads are affine in shared features, their interval-weighted parameter sums give a single readout per block. One block of 20 intervals requires one evaluation; two blocks of 10 requires two.

For a selected interval k, summing preceding predicted displacements gives $\smash { \widetilde { X } } _ { k }$ , where the teacher supplies head k’s target:

$$
\widetilde { X } _ { k } = \Pi _ { M } \left( X _ { \tau _ { s } } + \sum _ { j = s } ^ { k - 1 } \Delta _ { j } \operatorname { s g } ( h _ { \theta , j } ) \right) , \quad \mathcal { L } _ { \mathrm { P D D } } = \mathbb { E } \left[ \left. h _ { \theta , k } - \operatorname { s g } ( v _ { \psi } ( \widetilde { X } _ { k } , \tau _ { k } ; Z ) ) \right. _ { M } ^ { 2 } \right] .\tag{4}
$$

Here ψ denotes the frozen teacher and sg stops gradients. Gradients update the student through the selected head, with the proposed state and teacher target detached. Targets are Euler velocities at student-proposed states, supplied without a complete teacher rollout. Four packed draws each supervise one interval, with the flow-training coordinate normalization (Section B.1).

Calibration. We refine the student’s marginals by backpropagating the continuous ranked probability score (CRPS) through one fused sampling step. Using few paths keeps this step affordable, but empirical CRPS has a bias that can favor underdispersion (Roberts, 2026). Fair CRPS (Ferro, 2014) is unbiased for the marginal population score under iid sampling and finite first moments. Let $G _ { \theta } ( Z ; \mathcal { T } )$ denote the normalized endpoint of a fused block spanning [0, 1]. For m iid source draws $Z ^ { \bar { j } } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I )$ , set $X ^ { j } = G _ { \theta } ( Z ^ { j } ; \dot { \bar { \mathcal { Z } } } )$ . At coordinate $( \ell , c )$ , the fair estimator is

$$
\widehat C _ { \ell c } = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \left| X _ { \ell c } ^ { j } - \bar { Y } _ { \ell c } \right| - \frac { 1 } { 2 m ( m - 1 ) } \sum _ { \stackrel { i , j = 1 } { i \neq j } } ^ { m } \left| X _ { \ell c } ^ { i } - X _ { \ell c } ^ { j } \right| .\tag{5}
$$

Retaining PDD provides teacher supervision while the generated endpoints are optimized against observations. With ${ \widehat { C } } _ { M }$ the mean of $\widehat { C } _ { \ell c }$ over transported coordinates, the objective becomes $\mathcal { L } _ { \mathrm { c a l } } = \mathcal { L } _ { \mathrm { P D D } } + \mathbb { E } [ \widehat { C } _ { M } ]$ . We evaluate trajectory quality using fps-bench (Section 4) and report direct calibration diagnostics in Section C.2.

## 3.4 INFERENCE: ENSEMBLE SAMPLING

At inference, PathFM generates m complete future trajectories, from which we estimate quantiles and trajectory-dependent quantities. We stratify the initial Gaussian noise to improve coverage at a fixed ensemble size, then propagate each draw through the distilled sampler. Section C.1.2 constructs the ensembles.

## 4 EVALUATING TRAJECTORY FORECASTS

Trajectory forecasts should capture temporal dependence and accurately quantify uncertainty. Marginal accuracy alone cannot establish this. We introduce Forecast Path Score Bench (fps-bench) to evaluate these properties alongside marginal scores. The benchmark combines established scoring rules into a common protocol for constructing and comparing finite trajectory ensembles.

Six complementary path scores. Different downstream queries depend on different aspects of a path, motivating the complementary diagnostics in Table 7. Following the aggregation-andtransformation framework of Pic et al. (2025), we score both complete paths and transformations that expose these aspects. Energy score assesses the distribution of level paths, while differenced energy score emphasizes local changes by scoring the initial level and subsequent increments. Variogram score compares pairwise temporal contrasts to probe how values vary together across time (Scheuerer & Hamill, 2015). We then apply scalar CRPS to the horizon total, maximum, and first-passage time to assess accumulated uncertainty, extremes, and event timing. Specifically, first-passage time records the first crossing of a history-derived threshold. fps-bench evaluates 94 fev-bench tasks (Shchur et al., 2025), excluding those with NaN values that prevent computing some path scores.

Quantile-model completions. Applying these scores requires complete paths, which marginal quantile forecasts do not specify. We retain native samples when available. To include quantile models, we construct paths from their marginal quantiles under two explicit dependence assumptions. Sorted quantiles define a piecewise-linear inverse CDF with constant extension beyond the outer quantiles. An independent completion draws a separate quantile rank at each forecast time and channel, whereas a comonotone completion shares one rank across an item’s times and channels. Both laws preserve the same underlying interpolated marginals, allowing us to examine how the dependence assumption affects trajectory scores. These two choices serve as baselines, as different dependence assumptions may yield different scores for the same marginal forecasts. Section D.1 specifies the score definitions, estimators, and aggregation.

## 5 FORECASTING ACCURACY AND EFFICIENCY

We evaluate forecasts through their marginal accuracy, the path quality scores, and the cost of generating an ensemble. The experiments establish where trajectory forecasts help, then examine how distillation and score refinement make those forecasts practical. Section 6 complements these experiments with a study of chaotic systems, with a particular focus on long-run dynamics.

(a) FEV SQL skill ↑
<table><tr><td rowspan=1 colspan=2>TimesFM-3             0.4900PathFM-300M         0.4735</td></tr><tr><td rowspan=1 colspan=2>Chronos-2              0.4728</td></tr><tr><td rowspan=1 colspan=2>PathFM-126M         0.4640</td></tr><tr><td rowspan=1 colspan=2>TiRex-2                0.4550</td></tr><tr><td rowspan=1 colspan=2>Toto-2.0-2.5B           0.4442</td></tr><tr><td rowspan=1 colspan=1>TiRex</td><td rowspan=1 colspan=1>0.4258</td></tr><tr><td rowspan=1 colspan=1>TimesFM-2.5</td><td rowspan=1 colspan=1>0.4220</td></tr><tr><td rowspan=1 colspan=1>Sundial</td><td rowspan=1 colspan=1>0.3342</td></tr></table>

(b) TIME Norm. CRPS ↓
<table><tr><td>TimesFM-3</td><td>0.5363</td></tr><tr><td>Toto-2.0-2.5B</td><td>0.5394</td></tr><tr><td>PathFM-300M</td><td>0.5523</td></tr><tr><td>PathFM-126M</td><td>0.5531</td></tr><tr><td>Chronos-2</td><td>0.5563</td></tr><tr><td>PatchTST-FM (Ext.)</td><td>0.5568</td></tr><tr><td>TimesFM-2.5</td><td>0.5674</td></tr><tr><td>TiRex</td><td>0.5731</td></tr><tr><td>Sundial</td><td>0.6634</td></tr></table>

(c) FPS Path skill ↑
<table><tr><td rowspan=1 colspan=2>PathFM-300M         0.5458PathFM-126M         0.5386</td></tr><tr><td rowspan=1 colspan=2>TimesFM-3             0.4871</td></tr><tr><td rowspan=1 colspan=2>0.4699</td></tr><tr><td></td><td rowspan=1 colspan=1>0.4632</td></tr><tr><td></td><td rowspan=1 colspan=1>0.4586</td></tr><tr><td></td><td rowspan=1 colspan=1>0.4261</td></tr><tr><td></td><td rowspan=1 colspan=1>0.4220</td></tr><tr><td></td><td rowspan=1 colspan=1>0.4014</td></tr></table>

Figure 3: Marginal accuracy and trajectory quality for PathFM-126M and PathFM-300M. Native samplers PathFM and Sundial are teal and black. FPS selects each quantile model’s better independent or comonotone completion.

## 5.1 EXPERIMENTAL SETUP

Unless stated otherwise, PathFM refers to the 126M model; we use PathFM-126M and PathFM-300M when comparing sizes. We evaluate marginal accuracy on fev-bench (Shchur et al., 2025) and TIME (Qiao et al., 2026). Baselines include TimesFM-3 (Jain & Sen, 2026), Chronos-2 (Ansari et al., 2025), Toto-1.0/2.0 (Cohen et al., 2025; Khwaja et al., 2026), and TiRex/TiRex-2 (Auer et al., 2025; Podest et al., 2026). Quantile models use explicit independent or comonotone completions, whereas Toto-1.0 and Sundial (Liu et al., 2025) retain their native sampled paths. GIFT-Eval results appear in Section B.5.1, and the full component comparisons in Section E.1.

PathFM and all fps-bench forecasts use 32 paths per forecast. For PathFM, we use stratified Gaussian sources and one model evaluation per output block. For each quantile model, we evaluate independent and comonotone completions and report the higher aggregate skill.

## 5.2 RESULTS

PathFM-300M and PathFM-126M achieve the two highest FPS scores among the compared models while remaining competitive on marginal accuracy (Figure 3). The 300M model ranks second on fev-bench and third on TIME; both sizes have lower TIME CRPS than Chronos-2. On fps-bench, both outperform TimesFM-3 and Chronos-2 even when each quantile baseline uses its better independent or comonotone completion. Both also improve on Sundial’s native sampled forecasts across all three benchmarks. TimesFM-3 leads both marginal benchmarks, but its lower FPS ranking illustrates why marginal accuracy alone is insufficient to assess forecast paths.

Path-dependent scores. Figure 13 compares the six FPS components for PathFM and the three strongest external baselines. The clearest gains over TimesFM-3 (the strongest baseline) occur in scores for temporal changes, horizon totals, and maxima. These gains highlight the value of complete paths for downstream tasks that depend on temporal structure beyond marginal forecasts.

Ablation studies. We study the cumulative effects of architecture and learning choices in Section B.4. The teacher–student evaluation-budget comparison (Section B.3) and inference speedup breakdown (Figure 12) examine how distillation and shared history computation reduce sampling cost. Section B.5 provides additional results on GIFT-Eval. We also evaluate cross-variable dependence (Section E.2) and test fitted temporal copulas for quantile baselines (Section E.3).

Latency (ms)  
(a) Marginal accuracy  
![](images/381ccb0206e16b290447e4fe339d9cf4db0a0e755a082492b7b6e2fd23205be7.jpg)

(b) Path accuracy  
![](images/60a4a0e7d2f4fef817e5d3afba203442c9c7e79720fa860fa765de0da510d3b6.jpg)

Figure 4: Accuracy, latency, and model size across six forecasting models. Area denotes parameter count. Latency is the median time to return nine quantiles (a) or forecast paths (b) per query on an NVIDIA L4, with history and forecast lengths fixed across models. Quantile models use their better aggregate completion from Section 4.  
(a) Ensemble size  
![](images/0c1a0f3d496ff1b0f27c2ab91fba9ba0db1a25f7f517d153685cbb5634fa7178.jpg)

(b) Distillation  
![](images/b93121c448c350ec7066fcac83286db43aa57f03d1af672a471228ed83d62d57.jpg)  
Figure 5: Ensemble size and distillation on fev-bench tasks. (a) Calibrated 126M model with different sampling strategies at NFE=1 (Section C.1.2). (b) Its matched teacher and PDD student. Thin bars in (b) show pointwise 95% task-bootstrap intervals (Section B.3).

## 5.3 ACCURACY, MODEL SIZE, AND INFERENCE COST

PathFM combines the highest FPS skill among the compared models with a compact 126M-parameter network (Figure 4). With one evaluation per output block, it generates 32 paths in 51.6 ms on an NVIDIA L4, placing it on the observed accuracy–latency frontier for path forecasts. We compare the number of function evaluations in Section B.3 and sampling controls in Section C.3.

Ensemble size and distillation. Figure 5 examines two ways to reduce the sampling budget in separate controlled studies. On all 100 fev-bench tasks, stratification improves marginal accuracy most at small ensemble sizes: 32-path stratified sampling achieves almost the same FEV skill as 256 iid paths. With one evaluation, the student approaches the ten-evaluation teacher; two evaluations slightly exceed its FEV skill. This shows the power of distillation for time series flow matching models, and the effectiveness of the parallel decoding distillation.

Scaling to 300M parameters. We investigate scalability by comparing PathFM-126M with PathFM-300M. Figure 6 tracks flow pretraining; the larger model reaches lower training and corpus-evaluation losses at the same number of updates. The larger model improves both marginal accuracy and trajectory quality (Figure 3 and section B.5). These gains come at roughly twice the inference latency for generating 32 paths on an NVIDIA L4 (Figure 10). The scaling curves suggest that larger models may continue to improve accuracy. This shows the potential for larger flow models.

![](images/e6e3a866c8add771c8e39089565410042152659d998410cb055e2fcd4e3f7ccf.jpg)

![](images/821cfa6cd8d139363ec4eb4da70f0b4120b510302e684ac83361913d8ed80fcf.jpg)

![](images/5db7db1544588a5ab5158b894fce4df88b9a20601244d59b34e0e23f7b392e55.jpg)  
Figure 6: Pretraining progress on general and chaotic time series. (a) Training velocity MSE; (b) EMA velocity MSE on fixed corpus windows for PathFM-126M and PathFM-300M; (c) EMA mean validation sMAPE for the 75M and 126M chaotic models at horizon 256, zoomed to 1M–2M updates.

## 6 FORECASTING CHAOTIC DYNAMICS

Chaotic systems pose a sharp test of forecasting: can a model preserve statistical structure after losing alignment with a particular future? We compare fine-tuned Chronos-2, PANDA (Lai et al., 2026), and PathFM on point forecasting and long continuations of 135 DYSTS systems (Gilpin, 2021). We evaluate PathFM with ten-step Euler sampling per 256-step block. Section F gives its training recipe, and evaluation details.

## 6.1 FORECAST ACCURACY AND DYNAMICAL FIDELITY

Using sMAPE as in PANDA (Lai et al., 2026), we evaluate forecasts at horizons 64, 128, 256, and 512. Four complementary diagnostics assess continued paths: Wasserstein-1 distance to an empirical approximation of the invariant distribution, correlation-dimension error, squared Hellinger distance between normalized power spectral densities (PSD), and segment analog score. Together, they assess state occupancy, geometric dimension, frequency content, and similarity of short multivariate segments to reference dynamics. Held-out truth provides a finite-trajectory reference.

Table 2: Point accuracy and dynamical fidelity (lower is better; best in bold).
<table><tr><td></td><td colspan="4">sMAPE</td><td colspan="4">Dynamics</td></tr><tr><td>Model</td><td>H = 64</td><td>H = 128</td><td>H = 256</td><td>H = 512</td><td> $W _ { 1 }$ </td><td>PSD  $H ^ { 2 }$ </td><td>CD err.</td><td>Seg. analog</td></tr><tr><td>Chronos-1 FT</td><td>24.80</td><td>31.86</td><td>39.69</td><td>47.24</td><td>0.411</td><td>0.148</td><td>0.750</td><td>16.40</td></tr><tr><td>Chronos-2 FT</td><td>21.07</td><td>26.78</td><td>33.28</td><td>40.07</td><td>9.686</td><td>0.269</td><td>0.569</td><td>11.12</td></tr><tr><td>PANDA</td><td>19.09</td><td>24.42</td><td>31.94</td><td>41.22</td><td>0.588</td><td>0.195</td><td>0.537</td><td>10.03</td></tr><tr><td>PathFM (Ours)</td><td>16.45</td><td>21.52</td><td>27.43</td><td>34.50</td><td>0.251</td><td>0.061</td><td>0.273</td><td>4.80</td></tr><tr><td>Held-out truth</td><td></td><td></td><td></td><td></td><td>0.047</td><td>0.014</td><td>0.055</td><td>1.00</td></tr></table>

It achieves substantially better dynamical fidelity over 4,096-step rollouts than all evaluated baselines (Table 2). Relative to PANDA, it reduces the four long-term diagnostic scores by 49–69%. These gains accompany the lowest pointwise error at every reported horizon and persist when PathFM is scaled down to approximately PANDA’s 72M parameter count (Section F.2).

Chronos-1 (Ansari et al., 2024), an autoregressive forecasting model, has been shown to mimic local chaotic dynamics (Yu et al., 2026). Chronos-1 achieves lower $W _ { 1 }$ and PSD $H ^ { 2 }$ than the quantile model Chronos-2, with higher pointwise, correlation-dimension, and segment analog errors. These results highlight the distinct strengths and weaknesses of traditional autoregressive and quantile forecasting models. PathFM improves across every reported metric, combining stronger point accuracy with greater dynamical fidelity.

Distillation increases point-forecast error relative to the ten-step sampler. We hypothesize that sensitivity to perturbations amplifies small distillation errors during autoregressive forecasts. With one or two distilled steps, PathFM still outperforms PANDA on point accuracy and all four dynamical diagnostics. Section F.2 examines this trade-off across model sizes and sampling budgets.

## 6.2 WHAT SAMPLING PRESERVES

![](images/03a3acca0a0ada36454cdff170c645934486d31884d00f5ea32bff99042a1bba.jpg)  
Figure 7: Lorenz and Halvorsen continuations; PathFM uses one sampled path. Each column shows a full 4,096-step portrait above its first 512 x-coordinate steps. Gray curves show the observed continuation; open and filled circles mark the first and last forecast states.

Figure 7 pairs long continuations with their early time traces for Lorenz and Halvorsen systems. The sampled Lorenz path retains a two-lobe portrait and sustained x variability, whereas the plotted baselines collapse within 256 steps. For Halvorsen, PathFM’s sample retains the broad three-loop form, while PANDA follows a smaller recurrent orbit. Section F provides further diagnostics.

## 7 DISCUSSION AND CONCLUSION

PathFM combines flow matching, distillation, shared computation, and direct score refinement to generate complete future trajectories efficiently. Its strongest gains concern the quality of complete trajectories and the downstream predictions they support. The dynamical study shows a unique benefit: independently continued paths preserve variability that point forecasts can lose.

Several limitations remain. PathFM does not yet support conditioning on modalities beyond time series, such as text. For chaotic systems, all four dynamical diagnostics remain well above the held-out-truth reference (Table 2). Accurately recovering the invariant distribution remains an open challenge. A key direction for future work is to continue scaling the backbone and training data, pushing the frontier of generative time series forecasting.

## ACKNOWLEDGMENTS

We thank William Gilpin, Jeffrey Lai, and Anthony Bao for their insightful discussions and generous help. This work was conducted at Amazon AWS.

## AI USE STATEMENT

We used generative AI tools to support research execution, including clean and reformat dataset, implementation and analysis of experimental results. We also used these tools for literature retrieval and discovery, drafting sections of the paper, and polishing the writing for clarity and readability. We have reviewed all AI-assisted work, including code, experimental analyses, literature references, and manuscript text. We take responsibility for the final content of this work, including text, claims, and artifacts produced with the aid of generative AI.

## REPRODUCIBILITY STATEMENT

Section 3 describes the model and learning objectives, and Section B provides the architecture and training recipe. Sections 4 and D specify the trajectory benchmark, score estimators, and aggregation rules. Section F describes the chaotic-system experiments and their evaluation metrics. We will release the code and model weights to support reproducibility.

## REFERENCES

Taha Aksu, Gerald Woo, Juncheng Liu, Xu Liu, Chenghao Liu, Silvio Savarese, Caiming Xiong, and Doyen Sahoo. GIFT-Eval: A benchmark for general time series forecasting model evaluation. arXiv preprint arXiv:2410.10393, 2024. URL https://arxiv.org/abs/2410.10393.

Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, et al. Chronos: Learning the language of time series. arXiv preprint arXiv:2403.07815, 2024.

Abdul Fatir Ansari, Oleksandr Shchur, Jaris Kuken, Andreas Auer, Boran Han, Pedro Mercado,¨ Syama Sundar Rangapuram, Huibin Shen, Lorenzo Stella, Xiyuan Zhang, et al. Chronos-2: From univariate to universal forecasting. arXiv preprint arXiv:2510.15821, 2025.

Arjun Ashok, Etienne Marcotte, Valentina Zantedeschi, Nicolas Chapados, and Alexandre Drouin.<sup>´</sup> Tactis-2: Better, faster, simpler attentional copulas for multivariate time series. In International conference on learning representations, volume 2024, pp. 22897–22924, 2024.

Andreas Auer, Patrick Podest, Daniel Klotz, Sebastian Bock, G¨ unter Klambauer, and Sepp Hochreiter.¨ Tirex: Zero-shot forecasting across long and short horizons with enhanced in-context learning. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, Main Conference, pp. 57529–57580. Curran Associates, Inc., 2025. doi: 10.52202/085713-1929. URL https://proceedings.neurip s.cc/paper\_files/paper/2025/file/5356603f9c47399adfd372f77a67705 7-Paper-Conference.pdf.

Nicholas Boffi, Michael Albergo, and Eric Vanden-Eijnden. How to build a consistency model: Learning flow maps via self-distillation. Advances in Neural Information Processing Systems, 38: 33346–33382, 2026.

Nicholas M Boffi, Michael S Albergo, and Eric Vanden-Eijnden. Flow map matching with stochastic interpolants: A mathematical framework for consistency models. arXiv preprint arXiv:2406.07507, 2024.

Rongyao Cai, Yuxi Wan, Kexin Zhang, Ming Jin, Zhiqiang Ge, Daoyi Dong, Hang Yu, Yong Liu, and Qingsong Wen. The procrustean bed of time series: The optimization bias in point-wise loss functions, 2026. URL https://arxiv.org/abs/2512.18610.

Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. PixArt-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In International conference on learning representations, volume 2024, pp. 57611–57640, 2024.

Hanyin Cheng, Linfeng Wang, Zhengbo Qu, Yang Shu, Zhongwen Rao, Meng Wang, Yijie Li, Xin Jiang, Bin Yang, and Chenjuan Guo. Qiyao-m: Multimodal time series foundation model with role-aware modeling of endogenous and exogenous modalities. arXiv preprint arXiv:2609.34842, 2026.

Kevin Clark, Paul Vicol, Kevin Swersky, and David Fleet. Directly fine-tuning diffusion models on differentiable rewards. In International Conference on Learning Representations, volume 2024, pp. 4793–4822, 2024.

Ben Cohen, Emaad Khwaja, Youssef Doubli, Salahidine Lemaachi, Chris Lettieri, Charles Masson, Hugo Miccinilli, Elise Rame, Qiqi Ren, Afshin Rostamizadeh, Jean Ogier du Terrail, Anna-Monica´ Toon, Kan Wang, Stephan Xie, Zongzhe Xu, Viktoriya Zhukova, David Asker, Ameet Talwalkar,

and Othmane Abou-Amal. This time is different: An observability perspective on time series foundation models. arXiv preprint arXiv:2505.14766, 2025. URL https://arxiv.org/ab s/2505.14766.

Abhimanyu Das, Weihao Kong, Rajat Sen, and Yichen Zhou. A decoder-only foundation model for time-series forecasting. arXiv preprint arXiv:2310.10688, 2023.

Alexandre Drouin, Etienne Marcotte, and Nicolas Chapados. Tactis: Transformer-attentional copulas<sup>´</sup> for time series. In International Conference on Machine Learning, pp. 5447–5493. PMLR, 2022.

Christopher AT Ferro. Fair scores for ensemble forecasts. Quarterly Journal ofthe Royal Meteorological Society, 140(683):1917–1923, 2014.

William Gilpin. Chaos as an interpretable benchmark for forecasting and data-driven modelling. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021. URL https://openreview.net/forum?id=enYjtbjYJrf.

Tilmann Gneiting and Adrian E Raftery. Strictly proper scoring rules, prediction, and estimation. Journal ofthe American Statistical Association, 102(477):359–378, 2007.

Jiahe Huang, Sihan Xu, Sharvaree Vadgama, and Rose Yu. Recursive flow matching, 2026. URL https://arxiv.org/abs/2605.26535.

Alexis Jacq, Guillaume Couairon, Valentin De Bortoli, Quentin Berthet, Arnaud Doucet, and Romuald Elie. Diffusion fine-tuning with rewarded moment matching distillation. arXiv preprint arXiv:2606.30414, 2026.

Ayush Jain and Rajat Sen. TimesFM-3: A zero-shot foundation model for multivariate forecasting. Google Research Blog, August 2026. URL https://research.google/blog/times fm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/. Published August 31, 2026; accessed September 12, 2026.

Emaad Khwaja, Chris Lettieri, Gerald Woo, Eden Belouadah, Marc Cenac, Guillaume Jarry, Enguerrand Paquin, Xunyi Zhao, Viktoriya Zhukova, Othmane Abou-Amal, Chenghao Liu, Ameet Talwalkar, and David Asker. Toto 2.0: Time series forecasting enters the scaling era. arXiv preprint arXiv:2605.20119, 2026. URL https://arxiv.org/abs/2605.20119.

Marcel Kollovieh, Marten Lienen, David Ludke, Leo Schwinn, and Stephan G¨ unnemann. Flow¨ matching with gaussian process priors for probabilistic time series forecasting. In International Conference on Learning Representations, volume 2025, pp. 95435–95457, 2025.

Jeffrey Lai, Anthony Bao, and William Gilpin. Panda: A pretrained forecast model for chaotic dynamics. In International Conference on Learning Representations, volume 2026, pp. 104651– 104695, 2026.

Seunghan Lee, Jaehoon Lee, Jun Seo, Tae Yoon Lim, Dongwan Kang, Hwanil Choi, Minjae Kim, Sungdong Yoo, Junhyeok Kang, Sangjun Han, et al. Exaone finance 1.0: An attention-free time series foundation model for financial time series. arXiv e-prints, pp. arXiv–2609, 2026.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Yong Liu, Guo Qin, Zhiyuan Shi, Zhi Chen, Caiyin Yang, Xiangdong Huang, Jianmin Wang, and Mingsheng Long. Sundial: A family of highly capable time series foundation models. arXiv preprint arXiv:2502.00816, 2025.

Lucas Meyer, Claudio Sole, Huikan Xiang, Nicolas Li, Lucas Franceschino, Arnau Quera-Bofarull, Maarten P Scholl, Joachim Fainberg, and Geoffrey Negiar.´ t 0: A time-series foundation model for forecasting with context. arXiv preprint arXiv:2609.24559, 2026.

Alvaro Perez-Diaz, James C Loach, Danielle E Toutoungi, and Lee Middleton. Foundation model forecasts: Form and function. arXiv preprint arXiv:2510.19345, 2025.

Romain Pic, Clement Dombry, Philippe Naveau, and Maxime Taillardat. Proper scoring rules´ for multivariate probabilistic forecasts based on aggregation and transformation. Advances in Statistical Climatology, Meteorology and Oceanography, 11(1):23–58, 2025.

Patrick Podest, Marco Pichler, Elias Burger, Levente Z¨ olyomi, Bernhard Voggenberger, Wilhelm´ Berghammer, Daniel Klotz, Sebastian Bock, G¨ unter Klambauer, and Sepp Hochreiter. TiRex-2:¨ Generalizing TiRex to multivariate data and streaming, 2026. URL https://arxiv.org/ab s/2607.01204.

Zhongzheng Qiao, Sheng Pan, Anni Wang, Viktoriya Zhukova, Yong Liu, Xudong Jiang, Qingsong Wen, Mingsheng Long, Ming Jin, and Chenghao Liu. It’s TIME: Towards the next generation of time series forecasting benchmarks. In Proceedings ofthe 43rd International Conference on Machine Learning, 2026. URL https://arxiv.org/abs/2602.12147.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67, 2020. URL https://jmlr.org/papers/v21/20-074.html.

Kashif Rasul, Calvin Seward, Ingmar Schuster, and Roland Vollgraf. Autoregressive denoising diffusion models for multivariate probabilistic time series forecasting. In International conference on machine learning, pp. 8857–8868. PMLR, 2021.

Christopher David Roberts. Ensemble-size-dependence of deep-learning post-processing methods that minimize an (un)fair score: motivating examples and a proof-of-concept solution. arXiv preprint arXiv:2602.15830, 2026.

David Salinas, Valentin Flunkert, Jan Gasthaus, and Tim Januschowski. Deepar: Probabilistic forecasting with autoregressive recurrent networks. International journal offorecasting, 36(3): 1181–1191, 2020.

Michael Scheuerer and Thomas M Hamill. Variogram-based proper scoring rules for probabilistic forecasts of multivariate quantities. Monthly Weather Review, 143(4):1321–1334, 2015.

Neta Shaul, Chao Liu, Arash Vahdat, and Julius Berner. Parallel decoding distillation for fast image and video generation. arXiv preprint arXiv:2607.26004, 2026.

Oleksandr Shchur, Abdul Fatir Ansari, Caner Turkmen, Lorenzo Stella, Nick Erickson, Pablo Guerron, Michael Bohlke-Schneider, and Yuyang Wang. fev-bench: A realistic benchmark for time series forecasting. arXiv preprint arXiv:2509.26468, 2025.

The Forecasting Company. t0-gamma: Model card. Hugging Face model card, 2026. URL https://huggingface.co/theforecastingcompany/t0-gamma.

Annan Yu, Danielle Maddix, Boran Han, Xiyuan Zhang, Abdul Fatir Ansari, Oleksandr Shchur, Christos Faloutsos, Andrew Gordon Wilson, Michael W Mahoney, and Bernie Wang. Understanding the implicit biases of design choices for time series foundation models. In International Conference on Learning Representations, volume 2026, pp. 71998–72049, 2026.

## A RELATED WORK

Marginal forecasting. Chronos-2, TimesFM-3, and Toto 2.0 provide direct marginal quantiles while sharing information across variables (Ansari et al., 2025; Jain & Sen, 2026; Khwaja et al., 2026). Toto 1.0 instead predicts marginal Student-t mixtures and samples forecasts autoregressively across output patches (Cohen et al., 2025). Perez-Diaz et al. (2025) distinguish forecast products by their operational uses and show that marginals do not identify path-dependent event probabilities. PathFM represents the joint forecast through complete multivariate paths, so the same samples support marginal summaries and distributions of trajectory-dependent quantities.

Generative trajectory forecasting. DeepAR samples temporally dependent futures autoregressively (Salinas et al., 2020), and TACTiS learns multivariate predictive distributions with attentional copulas (Drouin et al., 2022). TimeGrad introduces autoregressive diffusion for multivariate forecasting (Rasul et al., 2021). Flow matching learns generative vector fields through conditional probability paths (Lipman et al., 2022; Huang et al., 2026); TSFlow incorporates Gaussian-process priors (Kollovieh et al., 2025), and Sundial uses flow matching for foundation-model pretraining (Liu et al., 2025). PathFM shares history computation across paths and uses distillation to support fast ensembles and direct forecast-score training.

Distillation and direct score optimization. Flow map methods learn finite-time transport and support both teacher distillation and self-distillation (Boffi et al., 2024; 2026). Parallel Decoding Distillation (PDD) predicts multiple integration updates per network evaluation (Shaul et al., 2026). DRaFT optimizes differentiable rewards by backpropagating through diffusion sampling (Clark et al., 2024). Rewarded Moment Matching Distillation (RMMD) combines reward optimization with moment-matching supervision and applies unbiased sample CRPS to GenCast (Jacq et al., 2026). Its reward gradient passes through a denoising step applied to a re-noised student sample.

PathFM backpropagates fair CRPS through the complete source-to-forecast map of one distilled evaluation. The same optimization updates continue matching the frozen teacher’s joint velocity field while fitting the student’s marginals to observations. Distillation thus supplies both a short path for score gradients and continued supervision from a joint generative teacher. Section 3.3 and table 4 give the objective and post-training comparisons.

Forecast evaluation. Proper scoring rules underpin forecast comparison (Gneiting & Raftery, 2007). Fair scores address finite-ensemble effects (Ferro, 2014); variogram and transformation-based scores expose complementary multivariate properties (Scheuerer & Hamill, 2015; Pic et al., 2025). fps-bench combines these principles on fev-bench tasks (Shchur et al., 2025) to assess temporal dependence and trajectory-dependent quantities alongside marginal accuracy. DYSTS supplies interpretable chaotic systems (Gilpin, 2021), while PANDA studies pretraining across chaotic dynamics (Lai et al., 2026). Together, these evaluations test the value of complete forecast paths through downstream uncertainty, short-term tracking, and long-run dynamical fidelity.

## A.1 CONCURRENT WORK

Time series foundation models are advancing rapidly. t0-gamma alternates temporal and variate attention to incorporate historical and known-future covariates (Meyer et al., 2026). It averages weights from five fine-tuned checkpoints into one 256M-parameter model (The Forecasting Company, 2026). QiYao-M fuses numerical patches with text and image embeddings (Cheng et al., 2026), but treats each target channel independently. EXAONE 1.0 specializes at financial forecasting and mixes time with causal convolutions and variates with groupwise pooling MLPs (Lee et al., 2026).

All three models predict marginal quantiles, leaving dependence across future times and variables unspecified. PathFM retains its advantage in directly generating joint multivariate trajectories, combining learned temporal and cross-variable dependence with fast sampling and marginal calibration.

## B MODEL AND TRAINING

This section specifies the backbone and the training recipe for 126M and 300M models, then examines distillation, calibration, and the architecture controls that support the design. It closes with model-size comparisons of forecast accuracy and inference cost, including results on FEV, FPS, TIME, and GIFT-Eval. The flow model has 125.1M parameters, and its distilled version has 126.3M including the PDD heads.

## B.1 BACKBONE AND INPUT

Backbone configuration. The settings below describe the general-domain forecasting model; Section F gives the separate dynamical configuration. The backbone uses velocity prediction, ReLU activation, rotary temporal positions, and dynamic adaptive RMS normalization. The latter applies $( 1 + \alpha ) \odot \mathrm { R M S N o r m } ( h ) \hat { + } \beta ,$ with scale and shift obtained from a shared time projection plus block-specific offsets. Modulation and velocity readouts start at zero; the learned state-entry gate starts at 0.1. Temporal and channel attention use query–key normalization. The Gaussian source scale is fixed at $\sigma = 0 . 7 5$ . Both model sizes use width 768, 12 attention heads, and feed-forward width 3,072. The 126M and 300M backbones have 12 and 30 blocks, with two and ten flow-free entry blocks, respectively.

Input representation. Following the input representation of Chronos-2 (Ansari et al., 2025), each channel uses context-derived location $\mu _ { c }$ and positive scale $s _ { c } \colon$

$$
\bar { Y } _ { \ell c } = \mathrm { a s i n h } \bigg ( { \frac { Y _ { \ell c } - \mu _ { c } } { s _ { c } } } \bigg ) , \qquad Y _ { \ell c } = \mu _ { c } + s _ { c } \mathrm { s i n h } ( \bar { Y } _ { \ell c } ) .\tag{6}
$$

The scale is the context root-mean-square deviation, floored at $1 0 ^ { - 4 }$ . Let binary masks V and K mark valid requested coordinates and known future coordinates, respectively, and let A contain normalized known values, zero elsewhere. Only $M = V \odot ( 1 - K )$ is transported. We write

$$
\Pi _ { M } ( X ) = M \odot X + K \odot A , \qquad \| B \| _ { M } ^ { 2 } : = \frac { \sum _ { \ell , c } M _ { \ell c } B _ { \ell c } ^ { 2 } } { \operatorname* { m a x } ( 1 , \sum _ { \ell , c } M _ { \ell c } ) } ,\tag{7}
$$

Here ⊙ denotes elementwise multiplication; batch axes are included when present. The supervision mask excludes missing targets, known covariates, and final-patch padding.

Flow-time sampling. Each training row uses one scalar flow time, shared across all channels and forecast steps. For each row and draw, we sample $t = \mathrm { s i g m o i d } ( - 0 . 8 + 0 . 8 \epsilon ) , \epsilon \sim \mathcal { N } ( 0 , 1 )$ . The interpolated state and velocity target follow Equation (1), with this same t applied to the entire future. Four independent source/time draws share each history, with their losses averaged and normalized by the batch’s transported-coordinate count. At inference, all channels advance at a common flow time.

## B.2 TRAINING RECIPE

Both 126M and 300M models follow the four-stage schedule in Table 3. Transformer weights are initialized afresh while retaining the released Chronos-2 input embeddings, which remain trainable.

Table 3: General-corpus training schedule for both model sizes. Warmup is linear and included in the update budget. Flow matching and PDD use constant rates after warmup; WSD decays linearly to zero. The final stage decays linearly to zero for 126M and uses a constant rate for 300M. EMA starts at the end of warmup and is preserved across continuation stages. Each complete recipe contains 2.3M optimizer updates.
<table><tr><td>Stage</td><td>Updates</td><td>Peak learning rate</td><td>Warmup</td><td>EMA decay</td></tr><tr><td>Flow matching</td><td>2,000,000</td><td> $5 \times 1 0 ^ { - 5 }$ </td><td>2,000</td><td>0.9999</td></tr><tr><td>Terminal decay (WSD)</td><td>100,000</td><td> $5 \times 1 0 ^ { - 5 }$ </td><td></td><td>0.9999</td></tr><tr><td>PDD</td><td>100,000</td><td> $3 \times 1 0 ^ { - 5 }$ </td><td>500</td><td>0.999</td></tr><tr><td>PDD + fair CRPS</td><td>100,000</td><td> $1 0 ^ { - 5 }$ </td><td></td><td>0.999</td></tr></table>

Optimization and stage transitions. The recipe uses AdamW with $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 9 )$ and global gradient-norm clipping at 1. Both models use weight decay 0.01 during flow matching and WSD. An optimizer update contains 1,024 channel rows. Terminal decay continues the flow model with its optimizer and EMA state. PDD initializes the student from the terminal teacher’s EMA weights, freezes that teacher, and starts a fresh optimizer. Calibration resumes the PDD student, optimizer, and EMA.

Data and task sampling. Training uses a 83M-row dataset of real-world and synthetic time series, including both univariate and multivariate data. Training samples contexts up to 8,192 time steps and forecast horizons up to 480 time steps. During training, we randomly sample context and forecast lengths within these limits.

Post-training objectives. PDD trains the backbone and twenty interval heads on a uniform time grid, using the block-size menu {1, 2, 5, 10, 20}, on-policy states, Euler teacher labels, and tiled residual heads. Calibration retains path-wise PDD and adds fair CRPS with coefficient one on each loss, globally normalized over scored coordinates. The CRPS term uses eight iid paths, one parallel evaluation, and the same Gaussian source scale as flow training and inference.

Main benchmark results use the final EMA checkpoint after all four stages. The architecture, distillation, and refinement controls below retain their stated training budgets.

## B.3 DISTILLATION AND EVALUATION BUDGET

Figure 8 reports a teacher–student inference budget comparison in both model sizes. Each pair is evaluated at 1, 2, 4, and 10 sequential network evaluations (NFE).

![](images/b2eb30d305abcd8d2c101bf7ee657a3bf84c99d46d256ebb33fb262f6c95003c.jpg)

![](images/422a706a38ea05e1d12c1a3ff48e1f892ba45fb6e36bdb4be494aaf00c4b92ba.jpg)

![](images/b8201512755f51cce9970a23724f8db75480105f7dc110c67b33b6ab55e0b5aa.jpg)

![](images/8491cf0c9673106aba956f402ab9a6e6d8da1486bb31d7c94f226fb926fe4ff9.jpg)  
Figure 8: Teacher–student accuracy at different evaluation budgets. Rows show teacher–student pairs from the 126M and 295M families; columns show FEV and six-component FPS skill, with higher values preferred. NFE uses a logarithmic horizontal axis. Thin vertical bars show pointwise 95% task-bootstrap intervals from 2,000 resamples.

At one evaluation, PDD avoids the teacher’s sharp loss of accuracy and has higher skill scores than the ten-evaluation teacher in both pairs. Additional evaluations give no consistent improvement. These results show that the student preserves the teacher’s forecasting accuracy at a much lower budget.

## B.4 ARCHITECTURE AND TRAINING ABLATIONS

Table 4 follows the cumulative design at the controlled development budget, from flow matching (R0– R5) through distillation, fair-CRPS refinement, and stratified sampling (R6–R8). The modulation, initialization, normalization, and regularization bundle in R2 improves both FEV and FPS. Restricting history attention in R3 enables shared history computation but reduces FEV skill; late entry and four training draws partially recover it. Distillation then improves both scores while reducing the number of evaluations from ten to one. R7 adds fair CRPS to the PDD objective, and R8 changes source sampling at the same weights. Stratification improves empirical FPS from 0.5248 to 0.5329, with a paired 95% task-bootstrap interval of [0.0042, 0.0131] for the skill gain.

Table 4: Cumulative architecture and post-training study. Rows R0–R7 use iid sources.
<table><tr><td>Cumulative change</td><td></td><td>NFE</td><td>FEV↑ FPS ↑</td></tr><tr><td>R0</td><td>Flow matching with clean target</td><td>10</td><td>0.4314 0.4379</td></tr><tr><td>R1</td><td>+ Velocity prediction</td><td>10</td><td>0.4311 0.4216</td></tr><tr><td>R2</td><td>+ Time modulation, zero-init head, QK norm, weight decay</td><td>10 0.4468</td><td>0.4960</td></tr><tr><td>R3</td><td>+ History-only attention</td><td>10</td><td>0.4356 0.5008</td></tr><tr><td>R4</td><td>+ Two flow-free entry blocks</td><td>10</td><td>0.4381 0.5088</td></tr><tr><td>R5</td><td>+ Four packed training draws</td><td>10 0.4402</td><td>0.5076</td></tr><tr><td>R6</td><td>+ Parallel Decoding Distillation</td><td>1</td><td>0.4471 0.5221</td></tr><tr><td>R7</td><td>+ PDD with fair CRPS</td><td>1 0.4488</td><td>0.5248</td></tr><tr><td>R8</td><td>+ Stratified sampling</td><td>1</td><td>0.4575 0.5329</td></tr></table>

## B.5 SCALING AND ADDITIONAL BENCHMARKS

We compare the 126M and 300M models on FEV, FPS, GIFT-Eval, and TIME, as well as their cost.   
Their training curves are in Figure 6.

## B.5.1 MODEL SIZE AND FORECAST ACCURACY

The 126M and 300M families contain 126.3M and 296.3M learned parameters, respectively, including their distillation heads. Their training recipes are given in Section B.2.

(a) FEV SQL skill ↑
<table><tr><td rowspan=1 colspan=2>TimesFM-3             0.4900</td></tr><tr><td rowspan=1 colspan=2>PathFM-300M         0.4735</td></tr><tr><td rowspan=1 colspan=2>Chronos-2              0.4728</td></tr><tr><td rowspan=1 colspan=2>PathFM-126M         0.4640</td></tr><tr><td rowspan=1 colspan=2>TiRex-2                0.4550</td></tr><tr><td rowspan=1 colspan=2>Toto-2.0-2.5B           0.4442</td></tr><tr><td rowspan=1 colspan=1>TiRex</td><td rowspan=1 colspan=1>0.4258</td></tr><tr><td rowspan=1 colspan=1>TimesFM-2.5</td><td rowspan=1 colspan=1>0.4220</td></tr><tr><td rowspan=1 colspan=1>Sundial</td><td rowspan=1 colspan=1>0.3342</td></tr></table>

(b) TIME Norm. CRPS ↓
<table><tr><td rowspan=1 colspan=4>TimesFM-3</td><td rowspan=1 colspan=1>0.5363</td></tr><tr><td rowspan=1 colspan=4>Toto-2.0-2.5B</td><td rowspan=1 colspan=1>0.5394</td></tr><tr><td rowspan=1 colspan=4>PathFM-300M</td><td rowspan=1 colspan=1>0.5523</td></tr><tr><td rowspan=1 colspan=4>PathFM-126M</td><td rowspan=1 colspan=1>0.5531</td></tr><tr><td rowspan=1 colspan=4>Chronos-2</td><td rowspan=1 colspan=1>0.5563</td></tr><tr><td rowspan=1 colspan=4>PatchTST-FM (Ext.)    0</td><td rowspan=1 colspan=1>.5568</td></tr><tr><td rowspan=1 colspan=4>TimesFM-2.5           0</td><td rowspan=1 colspan=1>.5674</td></tr><tr><td rowspan=1 colspan=5>TiRex                  0.5731</td></tr><tr><td rowspan=2 colspan=5>0.6634</td></tr><tr><td rowspan=1 colspan=2></td></tr></table>

(c) FPS Path skill ↑
<table><tr><td rowspan=1 colspan=2>PathFM-300M          0.5458PathFM-126M         0.5386</td></tr><tr><td rowspan=1 colspan=2>TimesFM-3             0.4871</td></tr><tr><td rowspan=1 colspan=2>0.4699</td></tr><tr><td></td><td rowspan=1 colspan=1>0.4632</td></tr><tr><td></td><td rowspan=1 colspan=1>0.4586</td></tr><tr><td></td><td rowspan=1 colspan=1>0.4261</td></tr><tr><td></td><td rowspan=1 colspan=1>0.4220</td></tr><tr><td></td><td rowspan=1 colspan=1>0.4014</td></tr></table>

Figure 9: Scaling PathFM from the 126M to the 300M family. Panels rank nine models on FEV SQL skill (100 tasks), TIME normalized CRPS (98 tasks), and FPS path skill (94 tasks). Native samplers PathFM and Sundial are teal and black. FPS selects each quantile model’s better independent or comonotone completion.

Latency (ms)

The 300M PathFM variant has higher FEV and FPS skill and lower GIFT-Eval and TIME CRPS than the 126M model (Figure 9 and table 5). PathFM-300M ranks second on FEV, third on TIME, and first on FPS among the baselines.

Table 5 adds GIFT-Eval (Aksu et al., 2024) to the benchmark suite. GIFT uses all 97 configurations without series or window caps and with cross-learning disabled. Against the external baselines, our 126M model ranks fourth on GIFT-Eval CRPS and the 300M model ranks third.

Table 5: Forecasting accuracy across four benchmarks. Darker blue indicates better performance on a linear scale within each column; bold marks the best value. Dashes indicate unavailable results.
<table><tr><td rowspan="2">Model</td><td>FEV</td><td>FPS</td><td colspan="2">GIFT-Eval</td><td colspan="2">TIME</td></tr><tr><td>Skill ↑</td><td>Skill ↑</td><td>MASE↓</td><td>CRPS↓</td><td>MASE ↓</td><td>CRPS↓</td></tr><tr><td>TimesFM-3</td><td>0.4900</td><td>0.4871</td><td></td><td></td><td>0.6398</td><td>0.5363</td></tr><tr><td>Chronos-2</td><td>0.4728</td><td>0.4699</td><td>0.6978</td><td>0.4854</td><td>0.6620</td><td>0.5563</td></tr><tr><td>TiRex-2</td><td>0.4550</td><td>0.4586</td><td>0.6973</td><td>0.4781</td><td></td><td></td></tr><tr><td>Toto-2.0-2.5B</td><td>0.4442</td><td>0.4261</td><td>0.6956</td><td>0.4759</td><td>0.6419</td><td>0.5394</td></tr><tr><td>TiRex</td><td>0.4258</td><td>0.4220</td><td>0.7158</td><td>0.4885</td><td>0.6831</td><td>0.5731</td></tr><tr><td>TimesFM-2.5</td><td>0.4220</td><td>0.4632</td><td>0.7050</td><td>0.4903</td><td>0.6686</td><td>0.5674</td></tr><tr><td>Sundial</td><td>0.3342</td><td>0.4014</td><td>0.7502</td><td>0.5590</td><td>0.7576</td><td>0.6634</td></tr><tr><td>PathFM-126M</td><td>0.4640</td><td>0.5386</td><td>0.7073</td><td>0.4878</td><td>0.6560</td><td>0.5531</td></tr><tr><td>PathFM-300M</td><td>0.4735</td><td>0.5458</td><td>0.7008</td><td>0.4818</td><td>0.6538</td><td>0.5523</td></tr></table>

## B.5.2 ACCURACY AND INFERENCE COST

Figure 10 adds the 300M model to the earlier 126M timing comparison in Figure 4. Both variants lie on the observed accuracy–latency frontier for path forecasts. Section C.3 specifies the matched timing protocol and its limitations.

(a) Marginal accuracy

![](images/92185fade3d2c2a033ec7d5001c7bfafb15d796d4c692bb24e2a8da2ad96d089.jpg)  
(b) Path accuracy

![](images/66b81beff62ce57f921bf62c6dbb509da42aebced9de51734828e9229339360b.jpg)  
Figure 10: Accuracy, latency, and model size with the 300M PathFM variant. Circle area denotes parameter count. Latency is the median time to return nine quantiles (a) or forecast paths (b) for one query on an NVIDIA L4 at context 2,048 and horizon 288. Accuracy uses the native benchmark settings.

## C SAMPLING AND COMPUTATIONAL EFFICIENCY

Ensemble size determines the precision of forecast summaries, while the sampler and its execution determine their cost. We first describe sampling error and ensemble construction, then report controlled latency and memory comparisons.

## C.1 FINITE-ENSEMBLE APPROXIMATION

A finite ensemble approximates a model’s predictive distribution through sampled paths. Forecast summaries and scores can vary with the returned ensemble due to Monte Carlo approximation. We first separate approximation error from model error, then describe ensemble construction; Section D.1.1 defines the scores of the empirical forecast.

## C.1.1 FORECAST QUANTITIES AND MODEL ERROR

Condition on the available forecasting information I and on the fitted model. For $m \geq 1$ iid paths from $P _ { \theta } ( \cdot \mid \mathcal { T } )$ and a fixed functional g of finite variance, write $\mu _ { \theta } = \mathbb { E } _ { P _ { \theta } } [ g ( X ) \mid { \mathcal { T } } ]$ . The sample average is unbiased for $\mu _ { \theta } .$ , and independence gives

$$
{ \widehat { \mu } } _ { g } = { \frac { 1 } { m } } \sum _ { i = 1 } ^ { m } g ( X ^ { ( i ) } ) , \qquad \mathbb { E } _ { Z } [ ( { \widehat { \mu } } _ { g } - \mu _ { \theta } ) ^ { 2 } \mid { \mathcal { Z } } ] = { \frac { \operatorname { V a r } _ { P _ { \theta } } ( g ( X ) \mid { \mathcal { Z } } ) } { m } } .\tag{8}
$$

Thus ordinary iid Monte Carlo standard error scales as $m ^ { - 1 / 2 }$ . This error remains for nondegenerate queries even when $P _ { \theta }$ is the true conditional law. Direct quantile outputs avoid estimating their reported quantiles from finitely many paths, but retain model error. Efficient ensemble generation therefore matters for both inference cost and the accuracy of downstream estimates.

Let the finite value $\mu _ { \star }$ <sub>⋆</sub> be the corresponding expectation under the true conditional law. The sample average’s squared error relative to this target decomposes as

$$
\operatorname { \mathbb { E } } _ { Z } [ ( \widehat { \mu } _ { g } - \mu _ { \star } ) ^ { 2 } \mid { \mathcal { Z } } ] = ( \mu _ { \theta } - \mu _ { \star } ) ^ { 2 } + \frac { \operatorname { V a r } _ { P _ { \theta } } ( g ( X ) \mid { \mathcal { Z } } ) } { m } .\tag{9}
$$

The cross term vanishes because $\mathbb { E } _ { Z } [ \widehat { \mu } _ { g } - \mu _ { \theta } \ | \ Z ] = 0$ . The first term is error in the model’s conditional functional expectation; the second is Monte Carlo approximation error. This expression concerns estimation of a conditional expectation, not prediction error against one random future outcome, which also contains outcome variability.

For an event A with model probability $p _ { \theta } = P _ { \theta } ( A \mid \mathcal { T } )$ , taking $g ( X ) = \mathbf { 1 } \{ X \in A \}$ gives sampling variance $p _ { \theta } ( 1 - p _ { \theta } ) / m$ . For $0 < p _ { \theta } < 1$ , the relative standard error is $\sqrt { ( 1 - p _ { \theta } ) / ( m p _ { \theta } ) }$ , making rare-event probability estimates demanding at small ensemble sizes. Increasing m reduces this sampling term but does not remove model error. Using conditional marginals or analytical integration, when available for a particular query, need not incur the same sampling error. Dependent or stratified draws require a different variance analysis. The $m ^ { - 1 / 2 }$ rate describes the approximation error of the iid sample average at a finite sample budget.

## C.1.2 ENSEMBLE CONSTRUCTION AND STRATIFICATION

At inference, we approximate the forecast distribution with an ensemble of m complete future trajectories, from which we estimate quantiles and other forecast summaries. Ensemble size controls the number of paths, independently of the one- or two-evaluation budget per path. With few draws, these summaries can be sensitive to sampling variability. Section C.2 measures the effect of sample count on marginal interval coverage across all 100 fev-bench test tasks. For more even source coverage at fixed $m .$ , each coordinate uses the midpoints of m equiprobable Gaussian bins, independently permuted across paths:

$$
u _ { r } = \frac { r - \frac { 1 } { 2 } } { m } , \qquad Z _ { \ell c } ^ { j } = \sigma \Phi ^ { - 1 } ( u _ { \pi _ { \ell c } ( j ) } ) , \quad r , j \in \{ 1 , \ldots , m \} ,\tag{10}
$$

Here Φ is the standard-normal CDF and each $\pi _ { \ell c }$ is an independent permutation. This assigns every rank once per coordinate without tying a path’s ranks across times or channels.

## C.2 MARGINAL COVERAGE AND ENSEMBLE SIZE

Larger ensembles improve marginal interval coverage at fixed model weights (Figure 11). Across all 100 fev-bench tasks, increasing the ensemble from 32 to 1,024 iid samples raises 80% interval coverage by 4 percentage points. The gains diminish as ensemble size increases.

![](images/9af4b14867a60eeaa93f51f66793667ffc3c2666883d46c57fe651a300ca3b7e.jpg)  
Figure 11: Marginal 80% interval coverage versus the number of forecast samples on fev-bench. Both model sizes use iid sampling. Horizontal references use the native 10th and 90th percentile outputs of Chronos-2 and TimesFM-3.

## C.3 INFERENCE LATENCY

Table 6 compares complete serving calls on an NVIDIA L4 with context 2,048, horizon 288, and batches of one or sixteen histories. We measure end-to-end time, with CUDA synchronization around each call. We report medians over 35 measured calls in seven blocks, following warmup and with randomized execution order. Models retain their native behavior. The marginal interface returns nine quantiles. The path interface returns forecast trajectories, using independent completion for quantile outputs.

Table 6: End-to-end latency on an NVIDIA L4 at context 2,048 and horizon 288.
<table><tr><td rowspan="2">Model</td><td rowspan="2"></td><td colspan="2">Nine quantiles (ms)</td><td colspan="2">Paths (ms)</td></tr><tr><td>Batch 1</td><td>Batch 16</td><td>Batch 1</td><td>Batch 16</td></tr><tr><td>PathFM, one evaluation</td><td>126.3</td><td>51.7</td><td>215.3</td><td>51.6</td><td>215.2</td></tr><tr><td>PathFM, two evaluations</td><td>126.3</td><td>96.4</td><td>403.2</td><td>96.4</td><td>403.3</td></tr><tr><td>Chronos-2</td><td>119.5</td><td>21.8</td><td>66.6</td><td>22.5</td><td>72.9</td></tr><tr><td>TimesFM-3</td><td>330.7</td><td>160.0</td><td>508.2</td><td>160.6</td><td>515.1</td></tr><tr><td>Toto-2.0-2.5B</td><td>2,454.3</td><td>198.3</td><td>1,482.6</td><td>199.0</td><td>1,489.5</td></tr><tr><td>TiRex</td><td>35.3</td><td>170.4</td><td>180.9</td><td>170.9</td><td>187.1</td></tr><tr><td>TiRex-2</td><td>82.5</td><td>27.0</td><td>44.0</td><td>27.6</td><td>50.5</td></tr><tr><td>Sundial</td><td>128.3</td><td>71.2</td><td>1,118.3</td><td>69.6</td><td>1,085.1</td></tr></table>

PathFM returns an ensemble in 51.6 ms for one history and 215.2 ms for sixteen. Batching sixteen histories reduces the average time per history to 13.5 ms. At both batch sizes, PathFM with one evaluation is faster than Sundial and TimesFM-3. Chronos-2 and TiRex-2 offer lower latency, while PathFM achieves higher FPS skill (Figure 4).

## C.3.1 INFERENCE SPEEDUP BREAKDOWN

Figure 12 isolates execution choices in a separate study on A100. Packing paths reduces teacher latency from 2,663.8 to 827.1 ms per batch; the packed two-evaluation student takes 201.9 ms. Adding deeper history K/V caching changes teacher latency only to 801.4 ms and is disabled for the student. The combined student speedup therefore comes from packing and distillation. Packing also reduces peak PyTorch allocated memory from 3.22 to 1.58 GiB.

![](images/5917a21b691846ab9a5e397324cc31b37470ed27951faf828b8acfe919ddeb43.jpg)  
Figure 12: Inference speedup breakdown on an A100 at batch 16, context 2,048, and horizon 480. The teacher uses ten Euler evaluations; the student uses two. All configurations use float32 weights with bfloat16 autocast.

## D EVALUATION METRICS

We specify the empirical forecast scores and aggregation rules used in the forecasting results. The chaotic-system protocols and diagnostics are given separately in Section F.

## D.1 SCORE CONSTRUCTION

## D.1.1 EMPIRICAL FORECAST SCORES

For a fixed outcome y and predictive law P, CRPS and energy score have the common form

$$
\begin{array} { r } { S _ { \rho } ( P , y ) = \mathbb { E } \rho ( X , y ) - \frac { 1 } { 2 } \mathbb { E } \rho ( X , X ^ { \prime } ) , \qquad X , X ^ { \prime } \overset { \mathrm { i i d } } { \sim } P , } \end{array}\tag{11}
$$

using absolute and Euclidean distance, respectively (Gneiting & Raftery, 2007). We evaluate the returned empirical forecast $\begin{array} { r } { \widehat { P } _ { m } = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \delta _ { x ^ { ( i ) } } } \end{array}$ . Substituting this distribution gives

$$
S _ { \rho } ( \widehat { P } _ { m } , y ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \rho ( x ^ { ( i ) } , y ) - \frac { 1 } { 2 m ^ { 2 } } \sum _ { i , j = 1 } ^ { m } \rho ( x ^ { ( i ) } , x ^ { ( j ) } ) .\tag{12}
$$

This is the exact score of the finite forecast distribution and requires no independence assumption on its members. We use Equation (12) for every CRPS and energy component. The variogram component evaluates the moments of the same empirical forecast (Equation (14)).

## D.1.2 TEMPORAL PATH SCORES

For item n and channel c, define the seasonal-error scale from history h and seasonal period $p \mathrm { : }$

$$
a _ { n c } = \frac { 1 } { | \mathcal { H } _ { n c } | } \sum _ { t \in \mathcal { H } _ { n c } } | h _ { n t c } - h _ { n , t - p , c } | , \mathcal { H } _ { n c } = \{ t : h _ { n t c } , h _ { n , t - p , c } \mathrm { ~ o b s e r v e d } \} .\tag{13}
$$

Except for first passage, the temporal scores use paths divided by finite positive $a _ { n c }$ . Table 7 specifies their distances in Equation (12); the variogram uses its separate formula below.

Level, sum, and maximum scores use the same observed-coordinate mask for every forecast path. DES retains the initial value when observed and increments only when both adjacent observations are available.

Variogram. For the nonempty set of observed time pairs $\mathcal { P } = \{ ( s , t ) : s < t , \ y _ { s } , y _ { t }$ observed}:

$$
\widehat { \mathrm { V S } } _ { 1 / 2 } = \frac { 1 } { | \mathcal { P } | } \sum _ { ( s , t ) \in \mathcal { P } } \left( | y _ { s } - y _ { t } | ^ { 1 / 2 } - \frac { 1 } { m } \sum _ { i = 1 } ^ { m } | x _ { s } ^ { ( i ) } - x _ { t } ^ { ( i ) } | ^ { 1 / 2 } \right) ^ { 2 } .\tag{14}
$$

All pairs receive equal weight (Scheuerer & Hamill, 2015).

Table 7: The definitions of the six temporal FPS components.  
Score Distance or transformation   
Energy (ES) $\rho ( x , y ) = \| x - y \| _ { 2 }$   
Differenced ES (DES) $\rho ( x , y ) = \| D ( x ) - D ( y ) \| _ { 2 } ,$ where $D ( x ) = ( x _ { 1 } , x _ { 2 } - x _ { 1 } , \dots , x _ { L } - x _ { L - 1 } )$   
Variogram (VS) Squared errors in expected temporal contrasts, Equation (14)   
Sum CRPS $\begin{array} { r } { \rho ( x , y ) = | \sum _ { t } x _ { t } - \sum _ { t } y _ { t } | } \end{array}$   
Max CRPS $\rho ( x , y ) = | \operatorname* { m a x } _ { t } x _ { t } - \operatorname* { m a x } _ { t } y _ { t } |$   
First-passage CRPS $\rho ( u , v ) = | T _ { \tau } ( u ) - T _ { \tau } ( v ) |$ in original units, Equation (15)

First passage. The threshold $\tau$ is the maximum of the last available context segment of length at most ${ \bar { L } } ,$ in original units. For a path $u ,$

$$
T _ { \tau } ( u ) = \operatorname* { m i n } \bigl ( \{ t \in \{ 1 , \ldots , L \} : u _ { t } > \tau \} \cup \{ L + 1 \} \bigr ) .\tag{15}
$$

This score requires a fully observed future and finite threshold, uses forecast steps as units, and treats no crossing as $L + { \dot { 1 } }$ . It therefore evaluates a censored event time. The sum, maximum, and first-passage scores identify their respective functional distributions rather than the full path law.

## D.1.3 CROSS-VARIABLE VARIOGRAM SCORE

The cross-variable variogram compares channel pairs at the same forecast time. Channels are standardized by their mean and standard deviation over at most 8,192 context observations; insufficient history or negligible variance excludes a channel. Using shared observed-coordinate masks, the estimator in Equation (14) averages over valid times and channel pairs, items, origins, and finally tasks with an arithmetic mean. We report this absolute score.

## D.2 AGGREGATION RULES

Marginal benchmarks. We retain the official within-task reductions for FEV, GIFT-Eval, and TIME (Shchur et al., 2025; Aksu et al., 2024; Qiao et al., 2026). FEV uses nine quantiles at $\{ 0 . 1 , 0 . 2 , \ldots , 0 . 9 \}$ . SQL scales errors by $a _ { n c }$ before averaging. GIFT-Eval and TIME report equally weighted geometric means of task or configuration losses relative to Seasonal Naive.

fps-bench. FPS compares each model with Seasonal Naive across six path metrics on the same 94 tasks with valid scores for every model and metric. For each metric, we average valid items within each channel, then channels and forecast origins, to obtain one loss per task. Let $r _ { d , k }$ be the model’s loss divided by Seasonal Naive’s loss for task d and metric k, clipped to [0.01, 100]. The aggregate score is

$$
\mathrm { F P S } = 1 - \mathrm { G M } _ { d , k } ( r _ { d , k } ) ,\tag{16}
$$

where GM denotes the geometric mean with equal weight for each task and metric. A score of zero matches Seasonal Naive; higher is better. Individual component skills use the same rule for one metric at a time.

## E FORECAST ACCURACY AND DEPENDENCE

These experiments examine which aspects of a forecast improve when the model generates complete paths. We report temporal component scores, evaluate cross-variable relationships, and test explicit dependence completions of quantile forecasts. Metric definitions and aggregation rules are collected in Section D.

## E.1 TEMPORAL FORECAST ACCURACY

Figure 13 compares the six FPS components for PathFM and the three strongest baselines (selecting the best-performing completion for each). Table 8 reports the full comparison. For quantile models, independent completion usually gives better temporal scores than comonotone completion, but the latter can improve sum skill. PathFM achieves the highest skill scores on all six components, with a 10.6% average improvement against the strongest baseline, TimesFM-3.

![](images/6fbb410547b3282af90864874131e12733ac9fd01474faca0636d7529e416730.jpg)  
Figure 13: Component skills for PathFM and the three strongest baselines on fps-bench. Bars show skill relative to Seasonal Naive; higher is better.

Table 8: Temporal-path skill on the 94 tasks in fps-bench; higher is better.
<table><tr><td>Forecast</td><td>Completion</td><td>ES</td><td>DES</td><td>VS</td><td>Sum</td><td>Max</td><td>FPT</td><td>FPS skill</td></tr><tr><td>TimesFM-3</td><td>Independent</td><td>0.4913</td><td>0.3437</td><td>0.5998</td><td>0.5032</td><td>0.3966</td><td>0.5455</td><td>0.4871</td></tr><tr><td>TimesFM-3</td><td>Comonotone</td><td>0.4757</td><td>0.3650</td><td>0.4894</td><td>0.5049</td><td>0.3435</td><td>0.5462</td><td>0.4590</td></tr><tr><td>Chronos-2</td><td>Independent</td><td>0.4783</td><td>0.3255</td><td>0.5901</td><td>0.4797</td><td>0.3805</td><td>0.5224</td><td>0.4699</td></tr><tr><td>Chronos-2</td><td>Comonotone</td><td>0.4616</td><td>0.3590</td><td>0.4818</td><td>0.4862</td><td>0.3331</td><td>0.5148</td><td>0.4434</td></tr><tr><td>TimesFM-2.5</td><td>Independent</td><td>0.4645</td><td>0.3164</td><td>0.5893</td><td>0.4818</td><td>0.3687</td><td>0.5135</td><td>0.4632</td></tr><tr><td>TiRex-2</td><td>Independent</td><td>0.4608</td><td>0.3201</td><td>0.5790</td><td>0.4663</td><td>0.3678</td><td>0.5165</td><td>0.4586</td></tr><tr><td>Toto-2.0-2.5B</td><td>Independent</td><td>0.4355</td><td>0.2707</td><td>0.5333</td><td>0.4339</td><td>0.3125</td><td>0.5225</td><td>0.4261</td></tr><tr><td>TiRex</td><td>Independent</td><td>0.4322</td><td>0.2783</td><td>0.5419</td><td>0.4323</td><td>0.3236</td><td>0.4828</td><td>0.4220</td></tr><tr><td>Sundial</td><td>Native</td><td>0.3769</td><td>0.3615</td><td>0.5072</td><td>0.4280</td><td>0.2999</td><td>0.4141</td><td>0.4014</td></tr><tr><td>Toto-1.0</td><td>Native</td><td>0.2346</td><td>-0.2072</td><td>0.3583</td><td>0.3985</td><td>-0.1313</td><td>0.4833</td><td>0.2300</td></tr><tr><td>PathFM</td><td>Native</td><td>0.4917</td><td>0.4557</td><td>0.6244</td><td>0.5383</td><td>0.5103</td><td>0.5896</td><td>0.5386</td></tr></table>

## E.2 CROSS-VARIABLE VARIOGRAM SCORES

Table 9 compares absolute cross-variable variogram scores on all 35 multi-target fev-bench tasks, retaining all 625 forecast origins. The score compares pairs of target variables at the same forecast time, using the standardization and reduction in Section D.1.3.

The Gaussian copula uses shrinkage estimates of normal-score residual-rank correlation and a shared AR(1) temporal factor, fitted on four forecast windows ending before the first test origin. Six shorthistory tasks use shorter calibration horizons. The nine marginal quantiles are interpolated linearly, with constant tails outside probabilities 0.1–0.9.

Task-specific adaptation. The fitted Gaussian copula estimates dependence parameters from historical forecast–outcome pairs on each benchmark task. The forecasting weights remain frozen, but the copula introduces task-specific parameter fitting. We report these adapted references in a separate table group from PathFM and the fixed independent and comonotone completions, which require no additional task-specific parameter fitting at evaluation time.

Table 9: Contemporaneous cross-variable variogram scores on 35 multi-target tasks. Entries are arithmetic task means; lower is better, with the lowest value in each protocol group bold. Fitted Gaussian references use task-specific calibration.
<table><tr><td>Forecast</td><td>Copula</td><td>VS↓</td></tr><tr><td colspan="3">No task-specific fitting</td></tr><tr><td>Chronos-2</td><td>Independent</td><td>0.1491</td></tr><tr><td>Chronos-2</td><td>Comonotonic</td><td>0.1648</td></tr><tr><td>TimesFM-3</td><td>Independent</td><td>0.1446</td></tr><tr><td>TimesFM-3</td><td>Comonotonic</td><td>0.1609</td></tr><tr><td>Toto-2.0-2.5B</td><td>Independent</td><td>0.1560</td></tr><tr><td>Toto-2.0-2.5B</td><td>Comonotonic</td><td>0.1741</td></tr><tr><td>PathFM</td><td>Native</td><td>0.1419</td></tr><tr><td colspan="3">Task-calibrated references</td></tr><tr><td>Chronos-2</td><td>Fitted Gaussian</td><td>0.1469</td></tr><tr><td>TimesFM-3</td><td>Fitted Gaussian</td><td>0.1420</td></tr><tr><td>Toto-2.0-2.5B</td><td>Fitted Gaussian</td><td>0.1533</td></tr></table>

## E.3 FITTED GAUSSIAN TEMPORAL COPULAS

We test whether fitting a Gaussian temporal copula to historical forecast errors improves the path forecasts of frozen quantile models. Building on the independent and comonotonic completions in Section E.1, we estimate temporal dependence from target-task history while retaining each baseline’s marginal quantile forecasts.

Gaussian copula fitting. For each baseline and item, four forecast windows ending before the first test origin supply standardized residuals, which we rank-transform to normal scores. We estimate a lag-one coefficient within these windows, share it across target channels, clip it to [−0.95, 0.95], and fix it before testing. Items with insufficient information use a task-pooled estimate. All eligible items are retained; 27 tasks use shorter calibration horizons without changing their test windows. The fitted coefficient defines a stationary latent AR(1) process used to generate temporally dependent ranks for the baseline’s interpolated marginal quantiles, with constant tails outside probabilities 0.1–0.9. This parameterization restricts the fitted temporal dependence to AR(1).

Table 10: Effect of fitting a Gaussian temporal copula on FPS over 94 common fev-bench tasks. Each baseline uses the same marginal forecasts across copula completions. Fitted Gaussian rows use task-specific calibration. Bold marks the best value within each protocol group.
<table><tr><td>Forecast</td><td>Copula</td><td>FPS ↑</td></tr><tr><td colspan="3">No task-specific fitting</td></tr><tr><td>Chronos-2</td><td>Independent</td><td>0.4699 0.4434</td></tr><tr><td>Chronos-2 TimesFM-3</td><td>Comonotonic Independent</td><td>0.4871</td></tr><tr><td>TimesFM-3</td><td>Comonotonic</td><td>0.4590</td></tr><tr><td>Toto-2.0-2.5B</td><td>Independent</td><td>0.4261</td></tr><tr><td>Toto-2.0-2.5B</td><td>Comonotonic</td><td>0.4147</td></tr><tr><td>PathFM</td><td>Native</td><td>0.5386</td></tr><tr><td colspan="3">Task-calibrated references</td></tr><tr><td>Chronos-2</td><td>Fitted Gaussian AR(1)</td><td>0.5077</td></tr><tr><td>TimesFM-3</td><td>Fitted Gaussian AR(1)</td><td>0.5243</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Toto-2.0-2.5B</td><td>Fitted Gaussian AR(1)</td><td>0.4765</td></tr></table>

Fitting the Gaussian copula improves FPS from 0.4699 to 0.5077 for Chronos-2, from 0.4871 to 0.5243 for TimesFM-3, and from 0.4261 to 0.4765 for Toto-2.0-2.5B, relative to independent completion. However, these completions impose a stationary Gaussian AR(1) dependence structure, and PathFM has a higher FPS score without task-specific copula fitting.

## F CHAOTIC SYSTEMS

We examine how model size, sampling, and fine-tuning affect point accuracy and long-run dynamics.

## F.1 SETUP AND FORECASTING PROTOCOLS

Dataset. We train PathFM on the PANDA-72M reproduction corpus (Lai et al., 2026), GilpinLab/dataset-panda72M-repro, containing 190,006 trajectories of 4,096 steps with 3–10 state coordinates from individual DYSTS systems and skew-product combinations. Reserving trajectories involving ten parent systems leaves 156,784 training trajectories and 33,222 validation trajectories. Point evaluation uses the separate base40 and skew40 test releases.

Models and training. The quantitative comparisons use the 75M and 126M PathFM families with native 256-step horizons, trained for 2M flow-matching updates followed by 100k terminal-decay updates. We train the 75M variant to approximately match PANDA’s parameter count (74.2M versus 71.6M). The 126M backbone has 12 transformer blocks and feedforward width 3,072; the 75M model uses 11 blocks and width 768. Both initialize their backbones afresh while retaining trainable input-patch embeddings. The global batch budget is 384 channel rows, with four flow draws per window. The learning rate is $1 0 ^ { - 4 }$ after 500 warmup updates, followed by the 100k linear terminal decay to zero; weight decay is zero and EMA decay is 0.999. Main results use ten-step Euler sampling, source standard deviation 0.86, and seed 42.

Training-progress evaluation. Figure 6c uses the same 2,048 fixed three-channel validation windows for both model sizes, selected randomly from the validation shard. Each window supplies 512 context steps and 256 forecast steps from a distinct held-out trajectory. We evaluate EMA weights with 16 stratified samples and ten Euler steps. Context-standardized sMAPE is averaged over channels and forecast steps, then across the 2,048 trajectories.

Point forecasts. We evaluate six 512-context windows from each of 9,347 trajectories using sMAPE, as in PANDA (Lai et al., 2026). We average sMAPE within each of the 480 source groups and then report the arithmetic mean across groups. PathFM feeds back the coordinatewise median of 16 stratified samples per 256-step block, averaging the middle two order statistics. Chronos-1 uses 16

samples and median feedback every 128 steps, chaining two native 64-step calls within each sample;   
Chronos-2 and PANDA continue point forecasts in 128-step blocks.

Long continuations. For each of the 135 systems, we generate 4,096-step continuations from origins 512 and 4,096 on an unseen trajectory. PathFM and Chronos-1 independently continue 16 iid paths per origin; Chronos-2 and PANDA continue point forecasts. For each diagnostic, we take the arithmetic mean over all realizations in a system. Across the 135 systems, we use a geometric mean for segment analog and arithmetic means for the other diagnostics. Segment-analog ratios differ by orders of magnitude across systems because their calibration distances can have very different scales; geometric aggregation gives each system equal weight on a log scale.

## F.2 MODEL SIZE AND SAMPLER CHOICE

Table 11 compares PathFM with 72M-parameter PANDA. The smaller PathFM retains lower errors at all four forecasting horizons and on all four dynamical diagnostics, showing that the gains persist at similar model size. We obtain these gains with our general forecasting architecture and flow-matching objective, without adding architectural components or losses tailored to chaotic systems. PANDA uses specialized dynamics embeddings with random polynomial and Fourier features motivated by Koopman-operator approximations (Lai et al., 2026).

Table 11: Model size at ten Euler evaluations per block. Both PathFM sizes follow the protocol in Section F.1; lower is better, with the best value in bold.
<table><tr><td></td><td colspan="4">sMAPE</td><td colspan="4">Dynamics</td></tr><tr><td>Model</td><td> $H = 6 4$ </td><td> $H = 1 2 8$ </td><td> $H = 2 5 6$ </td><td> $H = 5 1 2$ </td><td> $W _ { 1 }$ </td><td> $\mathrm { P S D } \ H ^ { 2 }$ </td><td>CD err.</td><td>Seg. analog</td></tr><tr><td>PANDA</td><td>19.09</td><td>24.42</td><td>31.94</td><td>41.22</td><td>0.588</td><td>0.195</td><td>0.537</td><td>10.03</td></tr><tr><td>PathFM-75M</td><td>16.94</td><td>22.16</td><td>28.22</td><td>35.37</td><td>0.263</td><td>0.063</td><td>0.286</td><td>5.07</td></tr><tr><td>PathFM-126M</td><td>16.45</td><td>21.52</td><td>27.43</td><td>34.50</td><td>0.251</td><td>0.061</td><td>0.273</td><td>4.80</td></tr></table>

(a) Point error, H= 128  
![](images/4561334810ccc307dd1e62fac8ce715474b4f8bf6e3689bd5b31f6b69df6bbb8.jpg)

(b) Point error, H= 512  
![](images/4bc79dcea6a6c32c65e6f7bd65d311da2760a39625478aba9c0e65c13e967c83.jpg)

(c) Long-run occupancy  
![](images/d008212bd3948ee6b1f0d8ab0ae6ffe3c005098e4c5e0ef6cfef62ddfe7b4228.jpg)  
75M Euler 75M PDD100k 126M Euler 126M PDD100k PANDA

Figure 14: Accuracy versus sampler budget for both model sizes. Point errors average all 480 source groups; $W _ { 1 }$ averages all 135 systems and uses 4,096-step sampled continuations. The dotted horizontal line marks PANDA; lower is better.

The sampler comparison evaluates each teacher and its student after 100k PDD updates (Figure 14 and table 12). At one or two evaluations, PDD improves point accuracy over Euler; ten-step Euler remains best within each size at every horizon. For 126M long continuations, two-evaluation PDD reduces mean $W _ { 1 }$ from 0.522 to 0.316 relative to two-step Euler, approaching ten-step Euler’s 0.251.

Table 12: Mean source-group sMAPE for the Euler teacher and its student after 100k PDD updates. NFE counts evaluations per 256-step block and sample; bold marks the lowest error within each size.
<table><tr><td>Model</td><td>Sampler</td><td>NFE</td><td> $H = 6 4$ </td><td> $H = 1 2 8$ </td><td> $H = 2 5 6$ </td><td> $H = 5 1 2$ </td></tr><tr><td rowspan="5">PathFM-75M</td><td>Euler</td><td>1</td><td>20.99</td><td>26.44</td><td>32.97</td><td>41.27</td></tr><tr><td>Euler</td><td>2</td><td>17.81</td><td>23.56</td><td>30.23</td><td>37.34</td></tr><tr><td>Euler</td><td>10</td><td>16.94</td><td>22.16</td><td>28.22</td><td>35.37</td></tr><tr><td>PDD100k</td><td>1</td><td>17.62</td><td>22.89</td><td>29.02</td><td>36.31</td></tr><tr><td>PDD100k</td><td>2</td><td>17.17</td><td>22.44</td><td>28.55</td><td>35.69</td></tr><tr><td rowspan="5">PathFM-126M</td><td>Euler</td><td>1</td><td>20.14</td><td>25.43</td><td>31.88</td><td>40.14</td></tr><tr><td>Euler</td><td>2</td><td>17.24</td><td>22.76</td><td>29.25</td><td>36.31</td></tr><tr><td>Euler</td><td>10</td><td>16.45</td><td>21.52</td><td>27.43</td><td>34.50</td></tr><tr><td>PDD100k</td><td>1</td><td>17.08</td><td>22.24</td><td>28.36</td><td>35.74</td></tr><tr><td>PDD100k</td><td>2</td><td>16.69</td><td>21.83</td><td>27.86</td><td>34.97</td></tr></table>

## F.3 FULL CONTINUATIONS

Figure 15 extends the qualitative comparison in Figure 7 to all 4,096 steps and adds Rossler. The¨ displayed PathFM-126M paths use the same checkpoint and ten Euler evaluations per block as the quantitative comparison. PathFM sustains Lorenz’s two-lobe motion and Rossler’s varying oscillation¨ amplitudes throughout the continuation. In comparison, Chronos-2 and PANDA both collapse to the center of the Lorenz example. In the Rossler example, Chronos-2’s amplitude decays and PANDA¨ settles into more regular oscillations.

(a) Lorenz  
![](images/9086852b6029c443f5bce5eb3c800af1beea4fdcce6c0e1054e9227819bd4e95.jpg)

![](images/a7a4f7e27f01416ef2ce97315cc938cf5dd8ddfa592b1c360dd182f5bb832173.jpg)

![](images/da837e9f498bb798f9b4f345e44933869d85e1914fd93a96537bf49daf231ab7.jpg)

![](images/ff87dceb440bcc7f2620930ecfe4518aaa213b8f7c761eed82174f7213202f33.jpg)

![](images/b36711fecdd3e65c69b873f98f605963b5ec00ba4ad23e5d8924da7b0b8b9a09.jpg)

![](images/ee1ac9a5ed7c5006a2e412dfca6bda972c08dba17558e18ab96201c3aba8ae39.jpg)

![](images/6925f74e777c15e0cf4926cb8aa4e745e61de70f33b2d4373bb35fd441190c1a.jpg)

![](images/e61d09afa6d521d8e426eb8b92df376e04ef73c26cf133fd68f4f6c9d2216882.jpg)

![](images/911ad962ee4a3d0887f16264718291fff7a6d627c7c2bdd8ad11ffc91aa1e2e8.jpg)

![](images/ad36359ea23f26e7b8d8f7caf9530c7f000f5a7afa46032b2af11b7fc457553c.jpg)  
Figure 15: Full 4,096-step Lorenz and Rossler continuations. All panels use the same initial¨ conditions. Gray curves show the observed continuation; circles mark the first and last forecast states, and shading marks the first 512 steps.

## F.4 METRIC DEFINITIONS

Point error. With context channel mean $\mu _ { c }$ and population standard deviation $\sigma _ { c } ,$ let $\begin{array} { r } { \widetilde { x } _ { t c } = } \end{array}$ $( x _ { t c } - \mu _ { c } ) / \operatorname* { m a x } ( \sigma _ { c } , 1 0 ^ { - 1 0 } )$ and define $\widetilde { y }$ identically. We use PANDA’s factor-100 convention (Lai et al., 2026):

$$
\mathrm { s M A P E } _ { h } = \frac { 1 0 0 } { h C } \sum _ { t = 1 } ^ { h } \sum _ { c = 1 } ^ { C } \frac { \left| \widetilde { x } _ { t c } - \widetilde { y } _ { t c } \right| } { \left| \widetilde { x } _ { t c } \right| + \left| \widetilde { y } _ { t c } \right| + 1 0 ^ { - 1 0 } } .\tag{17}
$$

This scores the blockwise median forecast; individual sampled-path errors are not averaged into the point score.

Wasserstein-1 distance $( W _ { 1 } )$ . Let R be the reference state cloud and $s$ the post-burn-in rollout states, standardized using reference channel means and standard deviations. For $\dot { K } = 6 4$ seeded unit directions, $\widehat { F } _ { \mathcal { R } , k }$ and $\widehat { F } _ { S , k }$ are the empirical CDFs of the projected states:

$$
\widehat { W } _ { 1 } ( \mathcal { R } , \mathcal { S } ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \int _ { 0 } ^ { 1 } \left| \widehat { F } _ { \mathcal { R } , k } ^ { - 1 } ( u ) - \widehat { F } _ { \mathcal { S } , k } ^ { - 1 } ( u ) \right| d u .\tag{18}
$$

The reference cloud retains at most 20,000 states; this calculation subsamples each cloud to at most 10,000. It measures distance to an empirical state distribution, without an exact invariant measure.

Spectral distance. Spectral and segment diagnostics standardize channels using trajectories 0–7. Welch spectra use 512-point periodic Hann windows, 50% overlap, unit sampling rate, and constant detrending. Each spectrum is normalized to unit mass; $p _ { x , c }$ is the forecast spectrum and $q _ { c }$ averages the six normalized reference spectra. The channel-mean squared Hellinger distance is

$$
H _ { \mathrm { s p e c } } ^ { 2 } = \frac { 1 } { 2 C } \sum _ { c = 1 } ^ { C } \sum _ { f } \left[ \sqrt { p _ { x , c } ( f ) } - \sqrt { q _ { c } ( f ) } \right] ^ { 2 } .\tag{19}
$$

It lies in [0, 1] and compares spectral shape independently of total power.

Correlation dimension. We estimate $\widehat { d } _ { \mathrm { c o r r } }$ from log–log slopes of empirical pair-distance probabilities, excluding pairs within a 30-step Theiler window. The estimator uses at most 4,096 states and 200,000 positive pair distances, taking the median slope over probability ranges [0.005, 0.05], [0.01, 0.10], and [0.02, 0.20], with 20 geometrically spaced probabilities per range. Against the dataset’s stored pooled estimate $\widehat { d } _ { \mathrm { p o o l } }$ , we report

$$
E _ { \mathrm { C D } } = \left| \widehat { d } _ { \mathrm { c o r r } } - \widehat { d } _ { \mathrm { p o o l } } \right| .\tag{20}
$$

The pooled target uses all ten original trajectories, including the test trajectory, and its own standardization; it is a descriptive geometric reference rather than an independent held-out estimate.

Segment analog. Flatten 64-step multivariate segments into vectors in the shared standardized coordinates, without recentering each segment. Let $\bar { \mathcal { U } _ { x } } , \bar { \mathcal { U } } _ { \mathrm { r e f } }$ , and $\mathcal { U } _ { \mathrm { c a l } }$ denote generated, reference, and calibration segments, and $\begin{array} { r } { d ( u , \mathcal { U } ) = \operatorname* { m i n } _ { v \in \mathcal { U } } \| u - v \| _ { 2 } } \end{array}$ . The score compares their nearest-reference distances:

$$
A _ { \mathrm { s e g } } = \frac { \mathrm { m e d i a n } _ { u \in \mathcal { U } _ { x } } d ( u , \mathcal { U } _ { \mathrm { r e f } } ) } { \mathrm { m a x } \{ \mathrm { m e d i a n } _ { u \in \mathcal { U } _ { \mathrm { c a l } } } d ( u , \mathcal { U } _ { \mathrm { r e f } } ) , 1 0 ^ { - 1 2 } \} } .\tag{21}
$$

Generated and calibration segments use stride 32; reference segments use stride eight, retaining at most 20,000. The calibration normalization gives a reference level of one; larger values indicate less similar local dynamics.