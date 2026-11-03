# ST-GFM: Spectral-Temporal Graph Forecasting with Measured Perturbation Bounds and Masked Spectral Reconstruction

Aashish Bohra<sup>a,∗</sup>, Vivek Vijay<sup>b</sup>

<sup>a</sup>Department of Computer Science and Engineering, Indian Institute of Technology Jodhpur, Jodhpur, Rajasthan, India <sup>b</sup>Department of Mathematics, Indian Institute of Technology Jodhpur, Jodhpur, Rajasthan, India

## A R T I C L E I N F O

Keywords: Graph neural networks perturbation stability power spectral density self-supervised learning spatio-temporal forecasting spectral graph theory trafic forecasting

## A BS T RA C T

Graph forecasters for sensor networks are trained in one regime and deployed in others, where sensor coupling and signal frequency content both change. We split this spectral shift into an operator channel and a frequency channel, each with a dimensionless measure. ST-GFM (Spectral-Temporal Graph Forecasting Model) filters every input step with Chebyshev and bidirectional difusion filters on one learned, non-symmetric operator and alternates them with per-node temporal attention. For the operator channel we prove Frobenius-norm perturbation bounds for both graph branches, show that the Chebyshev bound is attained to first order, and compute its constant from the trained weights and the measured operator norm: 17.9 on METR-LA and 21.5 on PEMS-BAY. Under random edge dropout the trained layer moves by 2.0% to 3.1% of this bound. Masked Spectral Reconstruction (MSR), a self-supervised loss active in the first 40 epochs, matches the log power spectrum of the encoder output to that of the input. For small log-spectral residuals on unclamped bins it is proportional to an Itakura–Saito divergence to leading order. Removing it raises the 60-minute MAE by 0.30 on METR-LA, close to the 0.33 lost when all three self-supervised losses go. Under one shared evaluation pipeline, the 60-minute MAE of ST-GFM is 3.5% below a retrained STAEformer on METR-LA and 13.0% below it on PEMS-BAY. Its 60-minute MAE is 3.098 ± 0.017 on METR-LA (five seeds) and 1.576 ± 0.006 on PEMS-BAY (three seeds), the lowest PEMS-BAY MAE among twelve published baselines.

## 1. Introduction

A loop detector network on a metropolitan freeway looks diferent at 8 AM and at 10 PM. In the morning, sensors along a few commuter corridors move together, crosscorridor coupling is weak, and speed series contain sharp transitions as queues form and clear. Late at night those clusters dissolve, coupling becomes difuse, and the series are smooth. A graph neural forecaster fitted across both regimes uses one graph operator and one set of filters for both [36, 56]. Agencies use its output for ramp metering, incident response and traveller information, so a forecaster that holds up when the regime changes is worth more than one that only wins on a fixed split. Energy, climate and financial networks show the same pattern at daily, weekly and seasonal scales.

We call the joint change in sensor coupling and signal frequency content spectral shift. It reaches a deployed model through two channels (Figure 1). In the operator channel, the graph operator the model uses no longer describes the coupling it faces. This happens when a distance graph is reestimated, when failed sensors take their edges with them, or when a graph fitted on one period is reused on another. In the frequency channel, the operator is unchanged but the inputs carry energy at temporal frequencies that difer from training. The two channels call for diferent tools. The first asks how a filter responds when its operator moves, which spectral graph theory can answer. The second asks what the encoder has learned to represent, which the training objective can influence.

Spatio-temporal forecasters are judged almost entirely on in-distribution accuracy. DCRNN [36], Graph WaveNet [56], MTGNN [55], D<sup>2</sup>STGNN [47], STAEformer [37] and others report strong numbers on METR-LA and PEMS-BAY, and none of them states how its predictions respond to a change of the operator. Perturbation bounds for graph filters exist [16, 18, 29, 35], and CITRUS [12] carries such an analysis into spatio-temporal forecasting. These bounds are written for generic filter banks or for other architectures, and none of these works computes its constants from the trained weights of an attention-based trafic forecaster. Existing self-supervised and frequency-aware objectives [17, 27, 46, 52, 54, 62] supervise embeddings, reconstructed values, spectrogram values or forecast coeficients. Under a squared error on values, the weak, fast bins of a speed window carry little weight next to the slow trend (Section 5.4). We have not found a pretext task that compares the log power spectrum of a graph encoder’s output with that of its input, which expresses every spectral discrepancy as a log-power ratio.

ST-GFM (Spectral-Temporal Graph Forecasting Model) is built around the two channels. Its encoder alternates Chebyshev-plus-difusion graph filtering on a learned, nonsymmetric operator with per-node temporal attention, and it keeps all twelve input steps through every graph layer, so each filter acts on the trafic state at one step. For the operator channel we bound how far each graph branch moves under an operator perturbation, compute the constants from the trained weights and the measured operator, and compare the Chebyshev bound with the measured response of the layer it describes. The frequency channel motivates the training objective. Masked Spectral Reconstruction (MSR) matches the log power spectral density (PSD) of the encoder output to that of the input during the first 40 epochs. Under one shared evaluation pipeline, the 60-minute MAE of ST-GFM is 3.5% below a retrained STAEformer on METR-LA and 13.0% below it on PEMS-BAY. The evaluation follows four research questions (RQs).

• RQ1. Does ST-GFM improve multi-horizon accuracy over strong published baselines on speed and flow data, and is the improvement stable across seeds and across evaluation pipelines?

• RQ2. Which components account for the accuracy?

• RQ3. How sensitive is the forecast to perturbations of the graph operator, and how does the measured response compare with the bound?

• RQ4. Where does frequency shift occur in trafic, and how does ST-GFM’s accuracy on METR-LA weekends compare with its full-test accuracy?

## The paper makes four contributions.

1. Perturbation bounds whose constants are measured on the trained model. We bound both graph branches and the encoder stack in the Frobenius norm without assuming a symmetric operator, so the bounds cover the learned operator ST-GFM uses. The Chebyshev bound is attained to first order on a constructed example. Its constant, computed from the trained weights and the measured norm of the trained operator, is 17.9 on METR-LA and 21.5 on PEMS-BAY. Under edge dropout the observed change of the first DCSE layer is 2.0% to 3.1% of the bound and grows close to linearly in the size of the edit.

2. A two-channel account of spectral shift with a protocol to match. Definition 1 gives the operator channel and the frequency channel each a dimensionless measure. The protocol perturbs the operator with the inputs held fixed (RQ3) and measures frequency shift on real data with the operator held fixed (RQ4). On both test sets the frequency term concentrates in weekday peak hours and is lowest on weekends.

3. MSR, a pretext loss on the log spectrum of the encoder output. MSR aligns the log-PSD of the encoder output with that of its input. For small log-spectral residuals on unclamped bins, neglecting numerical stabilisation, the objective is proportional to the Itakura–Saito divergence to leading order. Among the objectives reviewed in Section 2, it is the only one that compares log spectra. Removing MSR raises the 60-minute MAE by 0.30 on METR-LA and 0.22 on PEMS-BAY, against 0.33 and 0.24 when all three self-supervised losses are removed.

4. A spectral encoder on a learned, non-symmetric operator. DCSE runs an isotropic Chebyshev filter and a two-direction difusion filter on one learned anchor operator at every input step and alternates with pernode temporal attention. Replacing DCSE with a standard GCN costs 0.32 MAE at 60 minutes on METR-LA and 0.21 on PEMS-BAY. Collapsing the time axis before the graph layers costs 0.12 and 0.08.

Section 2 reviews related work. Section 3 defines the task and spectral shift, Section 4 the model, and Section 5 the analysis. Sections 6 and 7 report the experiments, including an exploratory transfer to industry portfolio returns (Section 7.5). Section 8 interprets them and Section 9 states their scope.

## 2. Related Work

The related work is categorized into four categories which are spatio-temporal forecasting on sensor graphs, distribution shift in temporal and graph data, spectral graph filters and their stability, and self-supervised and frequencyaware pre-training. We examine how existing methods capture spatial and temporal dependencies, adapt to changing data distributions, characterize the sensitivity of graph filters to structural perturbations, and learn representations from temporal and spectral information.

## 2.1. Spatio-Temporal Forecasting on Sensor Graphs

Most models pair a graph module for spatial coupling with a sequence module for time. DCRNN [36] places difusion convolution inside a recurrent encoderdecoder. STGCN [59] and ASTGCN [19] apply Chebyshev graph convolution to each time slice of a fixed distance graph, with temporal convolution in STGCN and spatio-temporal attention in ASTGCN. Graph WaveNet [56] learns an adaptive adjacency softmax $\mathrm { R e L U } ( E _ { 1 } E _ { 2 } ^ { \top } ) )$ from two node-embedding tables, the form ST-GFM also uses. AGCRN [3] builds its graph from one embedding table and adds node-specific parameters, MTGNN [55] learns the graph jointly with dilated temporal convolutions, and Adp-STGCN [8] combines several feature graphs with the road graph through adaptive graph convolution. GMAN [64], D<sup>2</sup>STGNN [47], DSTAGNN [34] and MegaCRN [25] model time-varying or decoupled spatial structure. Attention-based models now hold most of the leading numbers. STAEformer [37] shows that adaptive spatio-temporal embeddings make a plain Transformer competitive, PDFormer [24] and Traformer [26] model propagation delay and joint spacetime attention, T-Graphormer [2] applies Graphormer-style structural encodings to space-time tokens, and TITAN [51] routes between sequence-centric and variable-centric experts. STID [45] is a useful corrective: identity embeddings and an MLP recover much of the accuracy on these benchmarks. ST-GFM uses identity embeddings for that reason, and Section 7.2 reads its ablation with this in mind. Graph WaveNet and MTGNN shorten the window layer by layer with dilated temporal convolutions, so later graph convolutions see fewer and coarser steps. STGCN shortens it between graph layers with unpadded convolutions. DCRNN applies its difusion filter to a recurrent state that summarises all earlier steps. GMAN and STAEformer keep every step but use attention in place of a spectral filter. Keeping the time axis is therefore not new by itself. ST-GFM combines it with a spectral graph path on a learned, non-symmetric operator and states how far that path moves when the operator moves.

![](images/81f8a431d7d2848d04975f549348e9183ec04c885b73be46641d488514badbd5.jpg)  
Figure 1: Spectral shift between two regimes of the same freeway sensor network. The left and centre panels are schematic: sensor coupling above, the power spectrum of a typical speed series below. Spectral shift reaches a deployed model through an operator channel and a frequency channel, which the paper analyses and tests separately.

## 2.2. Distribution Shift in Temporal and Graph Data

RevIN [32] and Non-stationary Transformers [39] handle drift in per-series statistics through normalisation. DIDA [63] learns invariant patterns in dynamic graphs under spatio-temporal distribution shift, and the survey of Kazemi et al. [28] covers models that track evolving structure. These methods target the input distribution or the graph sequence. They do not state how a fixed spectral filter responds when its operator changes, which is the question behind our operator channel.

## 2.3. Spectral Graph Filters and Their Stability

Spectral graph convolution was introduced through the graph Fourier transform [6]. Chebyshev polynomial filters [10, 20] avoid eigendecomposition and are localised to � − 1 hops, and GCN [33] is their first-order simplification. BernNet [22], ChebNetII [21] and the analysis of Wang and Zhang [53] study what polynomial filters can represent and how their coeficients should be constrained. ChebNetII shows that unconstrained Chebyshev coeficients behave poorly at high order, which bears on the coeficient norms we report. On stability, Gama et al. [16] bound filter response under relative perturbations, Levie et al. [35] prove transferability of spectral filters across graphs, Kenlay et al. [29] give bounds in terms of edge edits, Gao et al. [18] treat stochastic perturbations, and Keriven et al. [30] study large random graphs. Ruiz et al. [44] survey the area. CITRUS [12] brings a perturbation analysis to spatio-temporal forecasting with continuous product-graph heat kernels. Our analysis uses standard tools. Its addition is a statement for the encoder we deploy, with a Chebyshev branch, a degree-normalised difusion branch and interleaved attention, a tightness result, constants computed from trained weights, and a comparison of the bound with the measured response of the trained layer.

## 2.4. Self-Supervised and Frequency-Aware Pre-Training

GraphCL [58] and GRACE [66] learn graph representations from augmented views. For time series, TNC [48], TS2Vec [61] and CoST [54] use temporal contrast, and SimMTM [11] reconstructs masked series in the time domain. STEP [46] and STD-MAE [17] pre-train masked autoencoders on long trafic histories and pass the result to a downstream graph network. TF-C [62] enforces consistency between time-based and frequency-based embeddings from two contrastive encoders. FreqMAE [27] masks patches of a short-time Fourier spectrogram of IoT sensor input and reconstructs them with a frequency-weighted squared error on the raw values. FEDformer [65] and FreTS [57] forecast in the frequency domain, wavelet-based financial forecasters [4, 5] build features from wavelet and filter-bank representations, and FreDF [52] adds an $\ell _ { 1 }$ loss between the Fourier coeficients of the forecast and those of the label. None of these compares log spectra. MSR compares the log-PSD of the graph encoder’s own output with the log-PSD of its input, which turns the error in every bin into a relative one. Log-spectral matching connects to the Itakura– Saito divergence [15], made precise in Section 5.4. Table 1 summarises the comparison.

## 3. Problem Formulation and Spectral Shift

## 3.1. Graph Setting and Task

A sensor network with � nodes is described by a symmetric nonnegative adjacency $A \in \mathbb { R } _ { > 0 } ^ { N \times N }$ with positive degrees $D = \mathrm { d i a g } ( A \mathbf { 1 } )$ . Road-distance kernels are asymmetric in general, and we symmetrise them as $A  ( A + A ^ { \top } ) / 2$ The symmetric normalised Laplacian

$$
L = I - D ^ { - 1 / 2 } A D ^ { - 1 / 2 }\tag{1}
$$

Table 1 Positioning of ST-GFM. “Cheb. Filter” Means a Chebyshev Polynomial Graph Filter. CITRUS Uses Continuous Heat-Kernel Filters on a Product Graph. “Bound” Means a Stated Bound on How the Output Moves When the Graph Operator Is Perturbed.
<table><tr><td>Method</td><td>Cheb. filter</td><td>Learned graph</td><td>SSL</td><td>Freq. pretext</td><td>Bound</td></tr><tr><td>DCRNN [36]</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>STGCN [59]</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>ASTGCN [19]</td><td>√</td><td>x</td><td>x</td><td>X</td><td>x</td></tr><tr><td>Graph WaveNet [56]</td><td>x</td><td>√</td><td>x</td><td>x</td><td>x</td></tr><tr><td>MTGNN [55]</td><td>x</td><td>√</td><td>x</td><td>X</td><td>x</td></tr><tr><td>D2STGNN [47]</td><td>x</td><td>√</td><td>x</td><td>x</td><td>x</td></tr><tr><td>STEP [46]</td><td>x</td><td>√</td><td>√</td><td>x</td><td>x</td></tr><tr><td>STD-MAÊ [17]</td><td>x</td><td>√</td><td>√</td><td>x</td><td>x</td></tr><tr><td>STAEformer [37]</td><td>x</td><td>x</td><td>x</td><td>X</td><td>x</td></tr><tr><td>CITRUS [12]</td><td>x</td><td>X</td><td>x</td><td>x</td><td>√</td></tr><tr><td>ST-GFM (ours)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

is real symmetric with $\mathrm { s p e c } ( L ) \subset [ 0 , 2 ]$ . Chebyshev filters act on a rescaled operator ${ \ll } = ( 2 / { \lambda _ { \operatorname* { m a x } } } ) L - I$ with spectrum in [−1, 1]. The learned operator of Section 4.2 need not be symmetric, and the perturbation analysis of Section 5 covers that case.

At each forecast origin we observe $X ~ \in ~ \mathbb { R } ^ { N \times W }$ , the last � readings of every sensor, and predict $Y \in \mathbb { R } ^ { N \times H }$ the next � readings. We write �<sup>̃</sup> for the window with each sensor standardised by its training mean and standard deviation. A model is a map $f _ { \theta } ( X , L )$ . In all experiments $W = H = 1 2$ at five-minute resolution, so horizons $H \in$ {3, 6, 12} are 15, 30 and 60 minutes ahead.

## 3.2. Spectral Shift

Definition 1 (Spectral shift). Let $L _ { \mathrm { r e f } }$ be a reference operator. For an input window � with standardised form ${ \tilde { X } } _ { t } .$ let $S ( X _ { t } ) = | \hat { F ( X _ { t } ) } | ^ { 2 } \in \mathbb { R } ^ { N \times } \dot { | W / 2 | }$ be its per-node temporal PSD from the real FFT, restricted to the non-DC bins. Let �<sup>̄</sup> be the mean of � over training windows. The spectral shift of a pair $( L _ { t } , X _ { t } )$ has an operator component and a frequency component,

$$
\delta _ { t } ^ { \mathrm { t o p } } = \frac { \| L _ { t } - L _ { \mathrm { r e f } } \| _ { 2 } } { \| L _ { \mathrm { r e f } } \| _ { 2 } } , \qquad \delta _ { t } ^ { \mathrm { f r e q } } = \frac { \| S ( X _ { t } ) - \bar { S } \| _ { \mathrm { F } } } { \| \bar { S } \| _ { \mathrm { F } } } .\tag{2}
$$

We drop the DC bin because for a 12-step window it equals 144 times the squared window mean, which would turn $\delta ^ { \mathrm { f r e q } }$ into a measure of speed level rather than of frequency content. Sensors with a missing reading in the window are left out of both norms. A window with almost no non-DC energy, such as free-flowing trafic at night, has $\delta ^ { \mathrm { f r e q } } \approx 1$ . The measure detects departures in non-DC spectral power, including changes in fluctuation amplitude and in the distribution of energy across frequencies. It does not isolate spectral shape, because multiplying a standardised window by � multiplies its PSD by $c ^ { 2 }$

Both components are relative and dimensionless. The normalisation expresses both changes relative to their reference magnitudes. The symmetric normalised Laplacian of (1) has spectral norm at most 2, but that bound does not extend to the learned, non-symmetric operator. The operator component is measured on the operator rather than its spectrum, so it registers changes that leave the eigenvalues in place. For symmetric operators it also controls the spectrum. By Weyl’s inequality every eigenvalue moves by at most $\| L _ { t } - L _ { \mathrm { r e f } } \| _ { 2 }$ . By the Davis–Kahan theorem [9] in the form of Yu et al. [60], an �-dimensional eigenspace with eigengap � rotates by at most $\Vert \sin \Theta \Vert _ { \mathrm { F } } \leq 2 \operatorname* { m i n } ( \sqrt { r } \Vert L _ { t } -$ $L _ { \mathrm { r e f } } \| _ { 2 } , \ \| L _ { t } - L _ { \mathrm { r e f } } \| _ { \mathrm { F } } ) / g$ . Neither result applies to the nonsymmetric learned operator, and the bounds of Section 5 do not rely on them.

The two components difer in what can be observed. $\delta ^ { \mathrm { f r e q } }$ is computable from data, and we measure it on every test window. $\delta ^ { \mathrm { t o p } }$ needs the true sensor coupling of a window, which is not observed, so we use it to parameterise controlled perturbations and not to describe the data.

## 3.3. How Each Channel Reaches the Model

ST-GFM uses a learned anchor operator $L _ { \theta } \left( \mathrm { S e c t i o n } 4 . 2 \right)$ fixed once training ends. The operator channel acts when the encoder receives $L _ { \theta } + \Delta L$ for any of the reasons listed in Section 1. The frequency channel acts through � alone. RQ3 perturbs the operator with the test inputs fixed, so only the operator component varies, and Section 5 bounds the response. RQ4 holds the operator at $L _ { \theta } ,$ measures $\delta ^ { \mathrm { f r e q } }$ on the test data, and compares forecasters on the low-shift weekend windows with the full test set. RQ2 measures what MSR, the objective motivated by the frequency channel, contributes to accuracy.

## 4. The ST-GFM Model

The full model of ST-GFM is shown in Figure 2. As shown in this figure, inputs pass through a multi-scale decomposition that keeps all � steps, are projected and given node, step and time embeddings, and enter an encoder of four layers that alternate DCSE graph filtering with per-node temporal attention. An aggregation layer and an MLP decoder produce the forecast, to which a learnable persistence term is added. During the first 40 epochs three self-supervised branches also train the encoder. The model has 5.22 million parameters. Only the two node embeddings (the additive one and the one in the decoder) and the adaptive-graph vectors depend on �, and they change the count by less than one percent between the two speed networks.

## 4.1. Input Embedding and Multi-Scale Decomposition

The reading is fused with two scalar time features, time of day on a 288-bin grid and day of week on a 7-bin grid, both scaled to [0, 1], through a learned map Linear(3, 1). The result is decomposed by $K _ { s } = 3$ stacks in the style of N-HiTS [7]. Stack � applies max pooling with rate $r _ { k } ~ \in$ {1, 2, 4}, a two-layer MLP with GELU [23], and a linear backcast to � steps. The input to stack $k + 1$ is the residual after the backcast of stack � is subtracted, so successive stacks capture progressively slower content.

![](images/7a83228d37c72fc6486e1d09c990101fb3af1ba12f097e09949ef78ad0661ef7.jpg)  
Figure 2: Overview of ST-GFM. Row 1: time-preserving decomposition, embeddings and the learned anchor operator. Row 2: DCSE layers, which share one operator, alternate with per-node temporal attention. Row 3: the bounds of Section 5 and the forecasting head with its persistence term. Row 4: self-supervised branches, active for the first 40 epochs.

Standard N-HiTS collapses the time axis. Each stack ap plies a one-dimensional convolution with kernel size 3 to its input and modulates the per-step features by the pooled summary through FiLM [43]. Stack � returns $D _ { \mathrm { d e c } } ^ { ( k ) } \in \mathbb { R } ^ { N \times W \times F } ,$ and

$$
D _ { \mathrm { d e c } } = \left[ D _ { \mathrm { d e c } } ^ { ( 1 ) } \left. \right. D _ { \mathrm { d e c } } ^ { ( 2 ) } \left. \right. D _ { \mathrm { d e c } } ^ { ( 3 ) } \right. \in \mathbb { R } ^ { N \times W \times F K _ { s } } ,\tag{3}
$$

with $F = 6 4$ , so each step carries 192 features. Keeping the time axis lets the graph layers that follow act on the trafic state at each step rather than on a shortened or summarised window (Section 2). If the steps were collapsed first, every graph filter would act on a temporal summary, and a filter applied to a summary cannot tell the onset of a queue at minute 50 from the state at minute 5 unless the summary has kept that diference. Collapsing the axis before the graph layers costs 0.12 MAE at 60 minutes on METR-LA (Section 7.2).

The decomposition is concatenated with a 48-dimensional learned time embedding (24 dimensions each for time of day and day of week) and projected to width $d = 3 2 0$ with Linear( $F K _ { s } + 4 8 , d )$ and GELU. Two learnable additive embeddings follow: a step embedding $P ^ { \mathrm { t m p } } \in \mathbb { R } ^ { W \times d }$ shared by all nodes and a node embedding $P ^ { \mathrm { n o d } } \in \mathbb { R } ^ { N \times d }$ , both initialised from $\mathcal { N } ( 0 , 0 . 0 2 ^ { 2 } )$ . These identity embeddings follow STAEformer [37] and STID [45] and reach every later layer through the residual connections.

## 4.2. Learned Anchor Graph

The static adjacency $A _ { \mathrm { s t a t i c } }$ comes from road distances through a thresholded Gaussian kernel (Appendix B). It is combined with a learned component [3, 56],

$$
\begin{array} { r l } & { A _ { \theta } = \left( 1 - \alpha \right) A _ { \mathrm { s t a t i c } } + \alpha \operatorname { s o f t m a x } \big ( \mathrm { R e L U } ( E _ { 1 } E _ { 2 } ^ { \top } ) \big ) , } \\ & { \quad \alpha = \sigma ( \alpha _ { 0 } ) , } \end{array}\tag{4}
$$

where $E _ { 1 } , E _ { 2 } \in \mathbb { R } ^ { N \times 1 0 }$ are learnable node embeddings and $\alpha _ { 0 }$ is a learnable scalar. After training � is 0.45 on both speed networks (Table 3), so the static and learned parts carry similar weight. From $A _ { \theta }$ we form the renormalised Laplacian with self-loops, $L _ { \theta } = I - \hat { D } ^ { - 1 / 2 } ( A _ { \theta } + I ) \hat { D } ^ { - 1 / 2 }$ with $\hat { D } = \mathrm { d i a g } ( ( A _ { \theta } { + } I ) \mathbf { 1 } )$ , and rescale it as $\boldsymbol { \mathfrak { s } } _ { \theta } = ( 2 / \hat { \lambda } ) L _ { \theta } - \boldsymbol { I } ,$ where �<sup>̂</sup> is the largest eigenvalue magnitude estimated by 20 power iterations. The softmax term is normalised by rows and $A _ { \theta }$ is not symmetrised, so ${ \mathfrak { s } } _ { \theta }$ is in general not symmetric. This rescaling is used as an operator-scaling convention. For a non-symmetric matrix, even an exact estimate of the largest eigenvalue magnitude does not guarantee that the eigenvalues of $\lessgtr _ { \theta }$ lie in the unit disk or that ‖ $\mathfrak { s } _ { \theta } \parallel _ { 2 } \leq 1$ . The perturbation analysis therefore uses the measured spectral norm directly. On the trained models it is 1.30 on METR-LA and 1.35 on PEMS-BAY, and Section 5.1 keeps this norm explicit. The operator is recomputed at each forward pass from the current parameters and is a fixed matrix after training. We write $L _ { \theta }$ to separate it from any externally supplied operator in the perturbation experiments.

## 4.3. Decoupled Chebyshev Spectral Encoder

The encoder treats time as a batch axis. The tensor $Z \in \mathbb { R } ^ { B \times N \times W \times d }$ is reshaped to $( B \cdot W ) \times N \times d .$ , so the graph filter at step � acts on the state at step �. A DCSE layer (shown in Figure 3) has two branches. The Chebyshev branch computes

$$
Z _ { \mathrm { c h e b } } = \sum _ { k = 0 } ^ { K - 1 } T _ { k } ( \Lt _ { \theta } ) Z \Theta _ { k } ,\tag{5}
$$

with $K = 3 ,$ , Chebyshev polynomials $T _ { 0 } ( x ) = 1 , T _ { 1 } ( x ) = x ,$ $T _ { k } ( x ) = 2 x T _ { k - 1 } ( x ) - T _ { k - 2 } ( x )$ evaluated on $\ll _ { \theta } ,$ and trainable $\Theta _ { k } ~ \in ~ \mathbb { R } ^ { d \times d }$ [10]. The difusion branch runs $K _ { \mathrm { d i f f } } \ = \ 2$ propagation steps in two directions,

$$
\begin{array} { r l r } {  { Z _ { \mathrm { d i f f } } = \sum _ { s = 1 } ^ { K _ { \mathrm { d i f f } } } \big ( P _ { f } S _ { s } W _ { f , s } + P _ { b } S _ { s } W _ { b , s } \big ) , } } \\ & { } & { S _ { 1 } = Z , \qquad S _ { s + 1 } = ( P _ { f } + P _ { b } ) S _ { s } , } \\ & { } & { P _ { f } = \hat { A } \mathrm { d i a g } ( \hat { A } \mathbf { 1 } ) ^ { - 1 } , \quad P _ { b } = P _ { f } ^ { \top } , \quad \hat { A } = I - \frac { 1 } { 2 } \preccurlyeq _ { \theta } , } \end{array}\tag{6}
$$

with separate weights per step and direction. The state carried to the next step is the sum of the forward and backward propagations. $P _ { f }$ divides each column of �<sup>̂</sup> by the matching row sum of ${ \hat { A } } ,$ and $P _ { b }$ is its transpose. The static distance kernel is symmetric, so neither direction encodes the direction of travel. The two difer through the learned, nonsymmetric part of $A _ { \theta }$ and through degree normalisation. The Chebyshev branch applies one isotropic polynomial filter of reach two hops, while the difusion branch can weight the two directions and the two hops separately. The branches are merged in a residual block,

$$
Z ^ { \prime } = \mathrm { L N } \Big ( \mathrm { G E L U } ( W _ { \mathrm { m e r g e } } [ Z _ { \mathrm { c h e b } } \ : \lVert Z _ { \mathrm { d i f f } } ] ) + Z \Big ) ,\tag{7}
$$

where $W _ { \mathrm { { m e r g e } } }$ maps 2� to � and LN is layer normalisation [1]. The learned part of $A _ { \theta }$ is dense, so one multiplication by ${ \mathfrak { s } } _ { \theta }$ or $P _ { f }$ costs $O ( N ^ { 2 } d )$ and not the $O ( | \mathcal { E } | d )$ of a sparse graph. $\mathrm { A t } \ \dot { N } \leq 3 2 5$ this is small on one GPU. Larger networks would need top-� sparsification of the learned term.

## 4.4. Per-Node Temporal Attention, Aggregation and Decoder

Between DCSE layers the tensor is reshaped to $\left( B \right) .$ $N ) \times W \times d$ , so each sensor attends only over its own $W = 1 2$ steps. The block uses pre-layer-norm multi-head self-attention [49] with four heads and dropout 0.1, a residual connection, and a two-layer feed-forward network of width 2� with GELU. No positional encoding is added inside the block, because $P ^ { \mathrm { t m p } }$ already reaches it through the residual stream. The encoder is

$$
\mathrm { D C S E } _ { 1 } \to \mathrm { T e m p A t t n } _ { 1 } \to \mathrm { D C S E } _ { 2 } \to \mathrm { T e m p A t t n } _ { 2 } .\tag{8}
$$

The encoder output is reshaped to $B \times N \times W \times d .$ At each node the $W \cdot d$ features are flattened and mapped to � by one linear layer, so every step contributes to every horizon. A gated skip from the encoder input $Z _ { 0 }$ is added, $Z _ { \mathrm { o u t } } ~ = ~ Z _ { \mathrm { a g g } } + \sigma ( g ) W _ { \mathrm { s k i p } } \mathrm { m e a n } _ { W } ( Z _ { 0 } )$ , with the scalar � initialised at 0. Following T-Graphormer [2], a degree embedding indexed by the number of nonzero entries in each row of $A _ { \mathrm { s t a t i c } } ,$ clamped at 60, is added and initialised at zero. A 32-dimensional node embedding is concatenated and a two-layer MLP decoder produces � outputs per node. The forecast adds a persistence term $W _ { \mathrm { b a s e } } X$ with $W _ { \mathrm { b a s e } } \in$ $\mathbb { R } ^ { W \times H }$ initialised to copy the last observed reading to every horizon, so the decoder learns the departure from persistence.

## 4.5. Self-Supervised Objectives and Masked Spectral Reconstruction

Three self-supervised branches act on the encoder during the first 40 epochs.

## 4.5.1. Masked Node Reconstruction

A fraction $q _ { \mathrm { m } } = 0 . 3$ of nodes is zeroed at the encoder input. A two-layer MLP decoder reconstructs, for each masked node �, the decomposition features $\bar { D } _ { \mathrm { d e c } , i } \in \mathbb { R } ^ { F K _ { : } }$ averaged over three randomly drawn steps:

$$
\mathcal { L } _ { \mathrm { s p a t i a l } } = \frac { 1 } { | \mathcal { M } | } \sum _ { i \in \mathcal { M } } \Big \| \mathrm { D e c } ( z _ { i } ) - \bar { D } _ { \mathrm { d e c } , i } \Big \| _ { 2 } ^ { 2 } ,\tag{9}
$$

where  is the masked set and $z _ { i } ~ \in ~ \mathbb { R } ^ { d }$ is the encoder summary of node �.

## 4.5.2. Masked Spectral Reconstruction

MSR uses the same masked input. Let $\bar { Z } \in \mathbb { R } ^ { N \times W }$ be the encoder output averaged over the feature axis, and let $\ell _ { Z } = \log ( | \mathcal { F } ( \bar { Z } ) | ^ { 2 } + \epsilon )$ and $\ell _ { X } = \log ( | \mathcal { F } ( \tilde { X } ) | ^ { 2 } + \epsilon )$ be the log-PSDs of the encoder output and of the standardised input window �<sup>̃</sup> before decomposition, both along time with the real FFT and $\epsilon = 1 0 ^ { - 8 }$ . The loss is

$$
\mathcal { L } _ { \mathrm { f r e q } } = \frac { 1 } { N W ^ { \prime } } \left\| \frac { \ell _ { Z } - \ell _ { X } } { s _ { X } } \right\| _ { \mathrm { F } } ^ { 2 } , \qquad s _ { X } = \mathrm { s g } \big ( \langle | \ell _ { X } | \rangle \big ) ,\tag{10}
$$

where $W ^ { \prime } = \lfloor W / 2 \rfloor + 1 = 7$ is the number of frequency bins, ⟨⋅⟩ is the mean over nodes and bins and sg stops the gradient. The loss averages over all � nodes, so it covers the 30% of sensors hidden at the input as well as the visible ones. The scale $s _ { X }$ keeps gradients stable when the input spectrum spans several orders of magnitude. For half-precision safety, $\ell _ { Z }$ is clamped to [−8, 8] before division and both normalised log-PSDs are clamped to [−10, 10]. The target is the input �<sup>̃</sup> and not the decomposition $D _ { \mathrm { d e c } }$ , so MSR asks the encoder to keep the spectral profile of the observed series while the masked branch asks it to fill in hidden nodes.

![](images/b564c4535fb7eea6f75d3d06129a85ada43df515c26e81f644243a2d895ea167.jpg)  
Figure 3: One DCSE layer, applied to the $N \times d$ state at one time step. Both graph branches read the anchor operator ${ \tilde { L } } _ { \theta } .$ The Chebyshev branch applies a polynomial filter of orders 0 to 2, and the difusion branch takes two propagation steps in each direction. Their outputs are concatenated, mixed by a linear map and added to the input before layer normalisation. The red tags mark the result in Section 5 that bounds each branch.

Three properties bound what MSR can enforce. The logarithm weights relative power error in every bin (Section 5.4). The PSD discards phase, so MSR constrains how energy is spread over frequency but not when events happen. And the loss acts on one readout of the encoder output, its feature mean. This readout is taken after the final temporalattention block, whose residual updates can produce feature means that vary across nodes and time steps. Using a parameter-free mean readout applies spectral supervision directly to the encoder representation without introducing an additional learned projection head. Unlike $\delta ^ { \mathrm { f r e q } }$ , MSR keeps the DC bin, so it also asks the readout to track the window mean.

## 4.5.3. Temporal Contrast

Two views are made by random amplitude scaling within $\pm 1 0 \%$ , a temporal shift of up to two steps and Gaussian noise with standard deviation 0.02. A symmetric InfoNCE loss with cosine similarity and temperature $\tau = 0 . 0 7$ is computed on up to 4096 sampled node embeddings [48, 61].

## 4.5.4. Training Schedule

The self-supervised objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { S S L } } = \mathcal { L } _ { \mathrm { s p a t i a l } } + \lambda _ { 1 } \mathcal { L } _ { \mathrm { f r e q } } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { t e m p } } , \quad \lambda _ { 1 } = 0 . 5 , \lambda _ { 2 } = 0 . 4 . } \end{array}\tag{11}
$$

The first $E _ { w } ~ = ~ 2 0$ epochs use $ { \mathcal { L } } _ { \mathrm { S S I } }$ alone. From epoch $e > E _ { w }$ the objective is $w ( e ) \mathcal { L } _ { \mathrm { S S L } } + \mathcal { L } _ { \mathrm { s u p } }$ with

$$
w ( e ) = \operatorname* { m a x } \biggl ( 0 , \ : 1 - \frac { e - E _ { w } - 1 } { 1 9 } \biggr ) ,\tag{12}
$$

which reaches zero at epoch 40. Training is purely supervised afterwards. $\mathcal { L } _ { \mathrm { s u p } }$ is the MAE over valid readings (Section 6). Supplementary Algorithm S1 lists the procedure.

## 5. Stability Analysis

This section bounds how far the encoder output moves when the operator moves from � to $L ^ { \prime } \ = \ L + \Delta L$ with the input fixed. Section 5.1 treats the Chebyshev branch, Section 5.2 the difusion branch, and Section 5.3 the stack. All bounds use the Frobenius norm, which matches the perturbation budgets of Section 7.3. Proofs not in the text are in Appendix A.

## 5.1. The Chebyshev Branch

The bound needs only the recurrence of the Chebyshev polynomials and the inequality $\left\| A B \right\| _ { \mathrm { F } } \leq \left\| A \right\| _ { 2 } \| B \| _ { \mathrm { F } } ,$ so it holds for any square operator, symmetric or not. This matters because the learned operator ${ \mathfrak { s } } _ { \theta }$ is not symmetric. The argument is elementary. What it adds is a constant that can be evaluated on the trained weights and checked against the measured response of the layer, which Section 7.3 does.

Theorem 1 (Chebyshev branch, $K = 3 )$ . Let $\Lt \Lt \Lt \Lt \Lt \Lt \Lt \Lt \Lt \Lt \Lt \Lt \Lt \Lt \Lt \Lt \Lt \Lt \Lt \Lt \Lt \Lt \Lt \Lt \Lt \Lt \ L \Lt \ L \Lt \ L \Lt \ L \ L _ { \ L } \ L \Lt \ L \ L \ L _ { \ L } \ L \ L \ L _ { \ L \ L } \ L \ L \ L \ L \ L _ { \ L \ L \ L } \ L \ L \Lambda \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L \ L$ any matrix with $\rho = \| \ll \| _ { 2 }$ , let Δ� be a perturbation with $\begin{array} { r } { \eta = \| \Delta L \| _ { 2 } , } \end{array}$ and let $\begin{array} { r } { g ( \ll ) = \sum _ { k = 0 } ^ { 2 } T _ { k } ( \ll ) \frac { } { } Z \Theta _ { k } . } \end{array}$ . Then

$$
\begin{array} { r } { \| g ( \llangle \Delta L ) - g ( \llangle ) \| _ { \mathrm { F } } \leq \| Z \| _ { 2 } \| \Delta L \| _ { \mathrm { F } } \qquad } \\ { \times \big ( \| \Theta _ { 1 } \| _ { 2 } + 2 ( 2 \rho + \eta ) \| \Theta _ { 2 } \| _ { 2 } \big ) . } \end{array}\tag{13}
$$

Proof. The $k = 0$ term cancels. $T _ { 1 } ( \ast + \Delta L ) - T _ { 1 } ( \ast ) = \Delta L ,$ and $T _ { 2 } ( \ll \ + \Delta L ) \ - \ T _ { 2 } ( \ll ) \ = \ 2 ( \ll \ \Delta L + \Delta L \ \ll \ + \Delta L ^ { 2 } )$ whose Frobenius norm is at most $2 ( 2 \rho + \eta ) \Vert \Delta L \Vert _ { \mathrm { F } }$ . Since $\| ( T _ { k } ( \ast \ + \Delta L ) - T _ { k } ( \ast ) ) Z \Theta _ { k } \| _ { \mathrm { F } } \ \leq \ \| T _ { k } ( \ast \ + \Delta L ) - T _ { k } ( \ast$ $) \| _ { \mathrm { F } } \| Z \| _ { 2 } \| \Theta _ { k } \| _ { 2 }$ , summing over $k = 1 , 2$ gives (13). □

Proposition 1 (Tightness). Let � be a symmetric normalised Laplacian with eigenvalue 2, as when the graph has a bipartite component, with unit eigenvector �, and let $\leqslant = L - I ,$ , so that $\rho = 1$ . Take $Z = u v ^ { \mathsf { T } }$ for a unit vector �, $\Theta _ { k } ~ = ~ M I _ { d }$ for $k \ = \ 0 , 1 , 2 ,$ , and $\Delta L \ = \ - \varepsilon u u ^ { \top }$ with $0 < \varepsilon < 2 .$ . Then

$$
\| g ( \ll + \Delta L ) - g ( \ll ) \| _ { \mathrm { F } } = M ( 5 \varepsilon - 2 \varepsilon ^ { 2 } ) ,
$$

while the right-hand side of (13) equals $M ( 5 \varepsilon + 2 \varepsilon ^ { 2 } )$ . The bound is attained tofirst order in �.

Proof. � is an eigenvector of ⪡ with eigenvalue 1 and of $\leqslant + \Delta L$ with eigenvalue $1 - \varepsilon$ . Hence $\begin{array} { r } { g ( \leqslant + \Delta L ) - g ( \leqslant ) = } \end{array}$ $M \left[ ( T _ { 1 } ( 1 - \varepsilon ) - \bar { 1 } ) + ( T _ { 2 } ( 1 - \varepsilon ) - 1 ) \right] \boldsymbol { u v } ^ { \top } = M ( 2 \varepsilon ^ { 2 } - 5 \varepsilon ) \boldsymbol { u v } ^ { \top }$ and $\Vert u v ^ { \top } \Vert _ { \mathrm { F } } = 1$ . On the right-hand side $\| Z \| _ { 2 } = 1 , \rho = 1$ and $\eta = \Vert \Delta L \Vert _ { \mathrm { F } } = \varepsilon$ □

For general �, symmetric operators and perturbations that keep the spectrum in [−1, 1], Markov’s inequality $| T _ { k } ^ { \prime } | \le k ^ { 2 }$ on [−1, 1] [41] gives a constant that grows as $\begin{array} { r } { \sum _ { k < K } \boldsymbol { k } ^ { 2 } \sim K ^ { 3 } / 3 } \end{array}$ , which argues for a small polynomial order. In relative form, Theorem 1 reads

$$
\begin{array} { r l } & { \frac { \| g ( \leqslant + \Delta L ) - g ( \leqslant ) \| _ { \mathrm { F } } } { \| Z \| _ { 2 } } \le c _ { \mathrm { c h e b } } ( \rho , \eta ) \| \Delta L \| _ { \mathrm { F } } , } \\ & { \qquad c _ { \mathrm { c h e b } } ( \rho , \eta ) = \| \Theta _ { 1 } \| _ { 2 } + 2 ( 2 \rho + \eta ) \| \Theta _ { 2 } \| _ { 2 } . } \end{array}\tag{14}
$$

Table 2 evaluates the coeficient norms in the first DCSE layer of every trained model and reports the unit-norm constant $c _ { \mathrm { c h e b } } ( 1 , 0 ) = \| \Theta _ { 1 } \| _ { 2 } + 4 \| \Theta _ { 2 } \| _ { 2 }$ . Across runs it is $1 4 . 4 9 \pm 0 . 5 5$ on METR-LA and $1 6 . 8 0 \pm 1 . 2 0$ on PEMS-BAY, 22% and 24% below the value it would take if $\Theta _ { 1 }$ and $\Theta _ { 2 }$ had the norm of $\Theta _ { 0 }$ . In every run $\Theta _ { 0 }$ has the largest norm. The order of the other two is not fixed: in PEMS-BAY seed 101, $\| \Theta _ { 2 } \| _ { 2 } ~ = ~ 3 . 5 9$ exceeds $\| \Theta _ { 1 } \| _ { 2 } ~ = ~ 3 . 4 3$ . Three coeficients cannot support a decay law, so we report the norms as measured.

The trained operator is not of unit norm. Each unit of $\rho$ above one adds $4 \| \Theta _ { 2 } \| _ { 2 }$ to the constant, about 11 on METR-LA and 13 on PEMS-BAY. Measured on the trained operators, $\| \ll _ { \theta } \| _ { 2 } \mathrm { i s } 1 . 3 0$ on METR-LA and 1.35 on PEMS-BAY (Table 3), which gives $c _ { \mathrm { c h e b } } ( \rho , 0 ) = 1 7 . 9 3$ and 21.48, about 24% and 28% above the unit-norm values. These are the constants of the deployed layer, and Section 7.3 compares the bound they give with the measured layer response. The constant grows linearly in � with slope $2 \Vert \Theta _ { 2 } \Vert _ { 2 } .$ , about 5.7 on METR-LA and 6.7 on PEMS-BAY. Supplementary Figure S1 plots it against � at both operator norms.

## 5.2. The Difusion Branch

The difusion branch depends on the operator through �<sup>̂</sup> and through the degree normalisation, which couples all entries of a row.

Lemma 1 (Difusion branch). Let ⪡ and $\leqslant ^ { \prime } = \leqslant + \Delta L$ be two operators such that $\begin{array} { r } { \hat { A } = I - \frac { 1 } { 2 } \ll } \end{array}$ and $\hat { A } ^ { \prime } = I - { \textstyle \frac { 1 } { 7 } } \ll ^ { \prime }$ have row sums of at least � > 0 and spectral norms of at most �. Let $s _ { \Delta }$ be the largest number of nonzero entries in a row of $\Delta L .$ . For the branch in (6),

$$
\| g _ { \mathrm { d i f f } } ( \llangle ^ { \prime } ) - g _ { \mathrm { d i f f } } ( \ll ) \| _ { \mathrm { F } } \leq \| Z \| _ { 2 } c _ { \mathrm { d i f f } } \frac { 1 } { 2 m } \Bigl ( 1 + \frac { a \sqrt { s _ { \Delta } } } { m } \Bigr ) \| \Delta L \| _ { \mathrm { F } } ,
$$

Spectral Norms of the Trained Chebyshev Coeficients in the First DCSE Layer of Every Run, and the Unit-Norm Constant $c _ { \mathrm { c h e b } } ( 1 , 0 ) = \lVert \Theta _ { 1 } \rVert _ { 2 } + 4 \lVert \Theta _ { 2 } \rVert _ { 2 }$ of (14), Computed From Unrounded Norms.
<table><tr><td></td><td> $\lVert \Theta _ { 0 } \rVert _ { 2 }$ </td><td> $\lVert \Theta _ { 1 } \rVert _ { 2 }$ </td><td> $\lVert \Theta _ { 2 } \rVert _ { 2 }$ </td><td> $c _ { \mathrm { c h e b } } ( 1 , 0 )$ </td></tr><tr><td>METR-LA</td><td></td><td></td><td></td><td></td></tr><tr><td>Seed 13</td><td>4.133</td><td>3.005</td><td>2.958</td><td>14.84</td></tr><tr><td>Seed 42</td><td>4.004</td><td>2.866</td><td>2.730</td><td>13.79</td></tr><tr><td>Seed 101</td><td>3.535</td><td>3.153</td><td>2.900</td><td>14.75</td></tr><tr><td>Seed 202</td><td>3.344</td><td>3.170</td><td>2.971</td><td>15.05</td></tr><tr><td>Seed 303</td><td>3.608</td><td>2.977</td><td>2.766</td><td>14.04</td></tr><tr><td>Mean</td><td>3.725</td><td>3.034</td><td>2.865</td><td>14.49</td></tr><tr><td>Std. deviation</td><td>0.331</td><td>0.127</td><td>0.111</td><td>0.55</td></tr><tr><td>PEMS-BAY</td><td></td><td></td><td></td><td></td></tr><tr><td>Seed 42</td><td>4.026</td><td>3.267</td><td>3.052</td><td>15.47</td></tr><tr><td>Seed 101</td><td>4.869</td><td>3.429</td><td>3.592</td><td>17.80</td></tr><tr><td>Seed 303</td><td>4.405</td><td>3.589</td><td>3.385</td><td>17.13</td></tr><tr><td>Mean</td><td>4.433</td><td>3.428</td><td>3.343</td><td>16.80</td></tr><tr><td>Std. deviation</td><td>0.422</td><td>0.161</td><td>0.272</td><td>1.20</td></tr></table>

(15)

where $\begin{array} { r } { c _ { \tiny { \mathrm { d i f f } } } = \sum _ { s ^ { \prime } = 1 } ^ { K _ { \tiny { \mathrm { d i f f } } } } s ^ { \prime } ( 2 a / m ) ^ { s ^ { \prime } - 1 } \big ( \| W _ { f , s ^ { \prime } } \| _ { 2 } + \| W _ { b , s ^ { \prime } } \| _ { 2 } \big ) . } \end{array}$

The factor $\sqrt { s _ { \Delta } }$ comes from the degree normalisation. It counts the nonzero entries in a row of the operator perturbation, not of the adjacency edit that produced it. Removing a few edges changes the degrees at their endpoints, and the normalisation and the rescaling by $\hat { \lambda }$ then spread the change over whole rows of $\ll _ { \theta } .$ . The learned part of $A _ { \theta }$ is dense, so for most edits to the graph of ST-GFM $s _ { \Delta }$ is close to � and the bound is looser by about $\sqrt { N }$ than for a perturbation with few entries per row.

For $K _ { \mathrm { d i f f } } = 2$ the constant is $c _ { \mathrm { d i f f } } = w _ { 1 } + ( 4 a / m ) w _ { 2 }$ with $w _ { s } = \| W _ { f , s } \| _ { 2 } + \| W _ { b , s } \| _ { 2 }$ . For $\llangle = \boldsymbol { L } - \boldsymbol { I }$ , the rescaling of a symmetric normalised Laplacian with $\lambda _ { \operatorname* { m a x } } = 2$ , the matrix $\begin{array} { r } { \hat { A } \ = \ \frac { 3 } { 2 } I \ - \ \frac { 1 } { 2 } L } \end{array}$ has spectral norm at most $\frac { 3 } { 2 }$ and row sums $\begin{array} { r } { 1 + \frac { 1 } { 2 } \sum _ { j } A _ { i j } / \sqrt { d _ { i } d _ { j } } \geq 1 } \end{array}$ , with $d _ { i }$ the degrees of �, so $4 a / m \leq 6$ . The trained operator is not of this form, and we measure � and � on it directly. Table 3 lists the result. The row sums of $\hat { A }$ are at least 1.38 on METR-LA and 1.33 on PEMS-BAY, and its spectral norm is 1.54 and 1.55, slightly above the symmetric value of $\frac { 3 } { 2 }$ . This gives $4 a / m = 4 . 4 6$ and 4.66, below the symmetric worst case, and $c _ { \mathrm { d i f f } } = 2 7 . 4$ on METR-LA and 32.0 on PEMS-BAY. These values hold at ${ \mathfrak { s } } _ { \theta }$ . Lemma 1 needs � and � at both operators, so a finite edit uses min(�, �<sup>′</sup>) and $\operatorname* { m a x } ( a , a ^ { \prime } )$ , and we do not instantiate the difusion bound under edge dropout. The second propagation step accounts for about four fifths of the constant on both networks. For a perturbation that fills whole rows, $s _ { \Delta } = N$ , the full normalisation prefactor $\frac { 1 } { \gamma _ { m } } ( 1 + \frac { a \sqrt { s _ { \Delta } } } { m } )$ in (15) is about 6.2 on METR-LA and 8.3 on PEMS-BAY, so the difusion bound per unit $\Vert \Delta L \Vert _ { \mathrm { F } }$ is about 170 and 265. That is roughly ten times the Chebyshev constant, and the $\sqrt { s _ { \Delta } }$ factor accounts for most of the gap.

Table 3 Operator and Branch Constants in the First DCSE Layer, Mean Over Runs. $c _ { \mathrm { c h e b } } ( \rho , 0 ) = \lVert \Theta _ { 1 } \rVert _ { 2 } + 4 \rho \lVert \Theta _ { 2 } \rVert _ { 2 }$ and $c _ { \mathrm { d i f f } } =$ $w _ { 1 } + ( 4 a / m ) w _ { 2 }$ With $w _ { s } = \| W _ { f , s } \| _ { 2 } + \| W _ { b , s } \| _ { 2 } , a = \| \hat { A } \| _ { 2 }$ and $m = \operatorname* { m i n } _ { i } ( { \hat { A } } \mathbf { 1 } ) _ { i }$ , as in Lemma 1 for $K _ { \mathrm { d i f f } } = 2 .$
<table><tr><td></td><td>METR-LA</td><td>PEMS-BAY</td></tr><tr><td>一  $| \ll _ { \theta } \| _ { 2 }$ </td><td>1.30</td><td>1.35</td></tr><tr><td> $c _ { \mathrm { c h e b } } ( \rho , 0 )$ </td><td>17.93</td><td>21.48</td></tr><tr><td>Learned-graph gate α</td><td>0.45</td><td>0.45</td></tr><tr><td> $a = \| \hat { A } \| _ { 2 }$ </td><td>1.54</td><td>1.55</td></tr><tr><td> $m = \operatorname* { m i n } _ { i } ( { \hat { A } } \mathbf { 1 } ) _ { i }$ </td><td>1.38</td><td>1.33</td></tr><tr><td> $4 a / m$ </td><td>4.46</td><td>4.66</td></tr><tr><td> $\dot { w _ { 1 } } = \| W _ { f , 1 } \| _ { 2 } + \| W _ { b , 1 } \| _ { 2 }$ </td><td>6.0</td><td>6.8</td></tr><tr><td> $w _ { 2 } = \| W _ { f , 2 } \| _ { 2 } + \| W _ { b , 2 } \| _ { 2 }$ </td><td>4.8</td><td>5.4</td></tr><tr><td>Cdiff</td><td>27.4</td><td>32.0</td></tr></table>

## 5.3. The Encoder Stack

A DCSE layer $F _ { \ell } ( Z , L )$ depends on its input and the operator. A temporal block $T _ { \ell } ( Z )$ depends only on its input.

Proposition 2 (Encoder stack). Let  be a compact set of rescaled operators such that every row sum of $I - { \textstyle { \frac { 1 } { 7 } } } L$ is at least $m _ { \star } > 0$ for every $L \in \mathcal { V }$ . Assume the layer inputs reachable under these operators belong to a bounded set. Suppose that uniformly on these sets each DCSE layer satisfies $\begin{array} { r } { \| F _ { \ell } ( Z ^ { \prime } , L ^ { \prime } ) - F _ { \ell } ( Z , L ) \| _ { \mathrm { F } } \ \le \ \kappa _ { Z } \| Z ^ { \prime } - Z \| _ { \mathrm { F } } + } \end{array}$ $\kappa _ { L } \| L ^ { \prime } - L \| _ { \mathrm { F } }$ and each temporal block satisfies $\| T _ { \ell } ( Z ^ { \prime } ) -$ $T _ { \ell } ( Z ) \| _ { \mathrm { F } } \le \kappa _ { T } \| Z ^ { \prime } - Z \| _ { \mathrm { F } }$ . Then, for a fixed input and any $L , L ^ { \prime } \in \mathcal { V } ,$ , the encoder (8) satisfies

$$
\begin{array} { r } { \| \mathrm { E n c } ( L ^ { \prime } ) - \mathrm { E n c } ( L ) \| _ { \mathrm { F } } \leq \kappa _ { T } \kappa _ { L } ( 1 + \kappa _ { Z } \kappa _ { T } ) \| \Delta L \| _ { \mathrm { F } } . } \end{array}\tag{16}
$$

These assumptions give finite uniform constants. Compactness of $\boldsymbol { \nu }$ bounds the operator norms and perturbation sizes, while $m _ { \star }$ keeps the difusion normalisation away from zero. For $\kappa _ { L }$ , Theorem 1 and Lemma 1 bound the two branches, the merge contributes $\| \boldsymbol { W } _ { \mathrm { m e r g e } } \| _ { 2 }$ , GELU is 1.13- Lipschitz [23], and layer normalisation with gain $\gamma$ and stabiliser $\epsilon _ { \mathrm { L N } }$ is $\| \gamma \| _ { \infty } / \sqrt { \epsilon _ { \mathrm { L N } } } \mathrm { - L i p s c h i t z }$ . The second DCSE layer receives inputs built from layer-normalised rows plus bounded attention and feed-forward updates, so $\| Z \| _ { 2 }$ in its branch bounds is bounded too. For $\kappa _ { T }$ , dot-product attention is not globally Lipschitz [31], but in a pre-layer-norm block the attention only sees rows of norm at most ${ \sqrt { d } } \| { \gamma } \| _ { \infty }$ + $\| \beta \| _ { 2 }$ , where it is Lipschitz. The encoder output is therefore Lipschitz in the operator on $\boldsymbol { \nu }$ for the bounded reachable input set, with a bound linear in $\Vert \Delta L \Vert _ { \mathrm { F } }$

We do not instantiate $\kappa _ { T } \kappa _ { L } ( 1 + \kappa _ { Z } \kappa _ { T } )$ . With $\epsilon _ { \mathrm { L N } } ~ =$ $1 0 ^ { - 5 }$ the layer-norm bound contains the factor 316.2 ‖�‖ , and products of worst-case layer constants overstate the true Lipschitz constant of deep networks by many orders of magnitude [14, 50]. The quantitative content sits in the branch constants, above all $c _ { \mathrm { c h e b } }$ . Proposition 2 gives a bound linear in $\Vert \Delta L \Vert _ { \mathrm { F } }$ on the stated admissible operator set. It does not give a uniform guarantee over arbitrary operator edits or edits that make a difusion row sum approach zero. Section 7.3 compares the bound with a measured response at the first DCSE layer, where both live in the same units.

## 5.4. MSR and the Itakura–Saito Divergence

$\begin{array} { r } { D _ { \mathrm { I S } } ( P _ { X } \Vert P _ { Z } ) = \sum _ { n , \omega } [ P _ { X } / P _ { Z } - \log ( P _ { X } / P _ { Z } ) - 1 ] } \end{array}$ , is the Bregman divergence of − log and the maximum-likelihood criterion for matching Gaussian-process spectra [15]. It is invariant to a common rescaling of both spectra.

Proposition 3 (MSR and the Itakura–Saito divergence). Let $u ( n , \omega ) ~ = ~ \log P _ { Z } ( n , \omega ) - \log P _ { X } ( n , \omega )$ on bins where no clamp in (10) is active, and ignore �. Then

$$
\mathcal { L } _ { \mathrm { f r e q } } = \frac { 2 D _ { \mathrm { I S } } ( P _ { X } \| P _ { Z } ) } { N W ^ { \prime } s _ { X } ^ { 2 } } + \frac { \sum u ^ { 3 } } { 3 N W ^ { \prime } s _ { X } ^ { 2 } } + O \biggl ( \frac { \sum u ^ { 4 } } { N W ^ { \prime } s _ { X } ^ { 2 } } \biggr ) .
$$

To leading order in the log-spectral residual, minimising MSR minimises the Itakura–Saito divergence between the encoder and input spectra. This is a local property that any loss on log-power ratios shares, and we state it to make the weighting of MSR explicit. Two consequences matter for forecasting. First, the residual � is a log-ratio, so an error of 10% in a bin holding little power costs as much as an error of 10% in the dominant bin. Under a squared-error loss on values, which by Parseval’s theorem is a squared-error loss on Fourier coeficients, the same relative error in the weak bin costs far less. In a one-hour speed window the weak bins are mostly the fast ones, which carry short transients such as a queue forming. Second, ignoring � and the clamps, the residual � does not change when both spectra of a sensor are multiplied by the same factor, so up to the stop-gradient scale $s _ { X }$ the loss weights relative spectral ratios and not raw power levels. The scale $s _ { X }$ is the mean absolute log-PSD of the input and does change with the input level. It multiplies every bin of a window by the same factor, so it sets the overall size of the loss and leaves the relative weight of the bins unchanged. Far from spectral agreement the cubic and higher terms matter, and on clamped bins the loss is no longer tied to $D _ { \mathrm { I S } }$

## 6. Experimental Setup

## Datasets and Preprocessing

Table 4 lists the four benchmark datasets. METR-LA and PEMS-BAY [36] record speed, and PEMS04 and PEMS08 [19] record flow. All are aggregated to five-minute intervals and cut into windows with stride one. The speed datasets are split chronologically $7 0 / 1 0 / 2 0$ and the flow datasets $6 0 / 2 0 / 2 0 $ the splits used by the baselines we compare against [24, 36, 37]. Inputs and targets are 12 steps long, and we report horizons 3, 6 and 12 on the speed datasets and the average over all 12 horizons on the flow datasets. Readings are standardised per node with training-set statistics. Missing readings, which the datasets record as zero, are excluded from the loss and from every reported metric by a mask that keeps readings above 1 in the original units. For the exploratory transfer study of Section 7.5 we use monthly value-weighted returns of the Fama–French 49 industry portfolios [13] from the Kenneth French data library, July 1926 to March 2026. Six industries whose histories start

Table 4 Datasets
<table><tr><td>Dataset</td><td>Signal</td><td>Sensors</td><td>Steps</td><td>Period</td><td>Split (%)</td></tr><tr><td>METR-LA</td><td>speed</td><td>207</td><td>34,272</td><td>03/2012-06/2012</td><td>70/10/20</td></tr><tr><td>PEMS-BAY</td><td>speed</td><td>325</td><td>52,116</td><td>01/2017-06/2017</td><td>70/10/20</td></tr><tr><td>PEMS04</td><td>flow</td><td>307</td><td>16,992</td><td>01/2018-02/2018</td><td>60/20/20</td></tr><tr><td>PEMS08</td><td>flow</td><td>170</td><td>17,856</td><td>07/2016-08/2016</td><td>60/20/20</td></tr></table>

in the 1960s are excluded, which leaves 43 portfolios with complete histories. After the 20-month volatility warm-up (Appendix B) the first usable month is March 1928, so 982 months before 2010 are available for fine-tuning. Months from 2010 onward are held out.

## Baselines

On the speed datasets we compare with twelve models that report results at horizons 3, 6 and 12 under the chronological split above: DCRNN [36], Graph WaveNet [56], MTGNN [55], STEP [46], D<sup>2</sup>STGNN [47], MegaCRN [25], Traformer [26], STAEformer [37], STD-MAE [17], Adp-STGCN [8], TITAN [51] and T-Graphormer [2]. They cover recurrent, convolutional, adaptive-graph, pre-training and attention-based designs. All use a 12-step input window, except that STEP and STD-MAE also encode long-term history through pre-trained encoders. On the flow datasets we compare against PDFormer [24] and STAEformer, which use the same 12-step input and report the average over all 12 horizons. STD-MAE [17], which pre-trains on 864 (PEMS04) and 2016 (PEMS08) steps of history, is listed as a long-history reference. The baseline numbers in Tables 5, 6 and 7 come from the original papers. General time-series Transformers such as FEDformer [65] and PatchTST [42] have no oficial numbers under this protocol. To separate the model from our pipeline, we retrain a reduced Graph WaveNet (hidden width 32, four layers, 50 epochs) and STAEformer from its oficial code, and score two naive forecasters, all under our own preprocessing and evaluation code. Each retrained baseline is a single run with its own training recipe.

## Protocol and Implementation

One model is trained with $H _ { \operatorname* { m a x } } = 1 2$ and evaluated at every horizon by slicing its output, as in STAEformer and T-Graphormer. We use AdamW [40] with peak learning rate $2 \times 1 0 ^ { - 4 }$ , weight decay $3 \times 1 0 ^ { - 4 }$ , cosine annealing with a 5% linear warm-up, gradient clipping at 2.0, dropout 0.1 and mixed precision. Batch size is 64 on METR-LA, 32 on PEMS-BAY and 16 on PEMS04 and PEMS08. Training runs for at most 180 epochs and stops once validation MAE has not improved for six consecutive checks, that is, 30 epochs. After the self-supervised warm-up, validation MAE is checked every five epochs, and the five best checkpoints are averaged at the end. Every speed-dataset run stopped between epochs 65 and 75. One hyperparameter configuration (Supplementary Table S1) is used apart from batch size. Hyperparameters were tuned on the validation split only, and early stopping, checkpoint selection and checkpoint averaging also use the validation split. The test split was never used to select anything. The model has 5.22 million parameters. All runs used one workstation with an AMD Ryzen 9 9900 processor, 32 GB of RAM and one NVIDIA GeForce RTX 5060 Ti with 16 GB of memory, running PyTorch 2.12.0 with CUDA 13.0 and cuDNN 9.20.0.

## Seeds and Uncertainty

METR-LA results are means over five seeds (13, 42, 101, 202, 303) and PEMS-BAY results over three (42, 101, 303). PEMS-BAY used fewer seeds because each run takes eight to nine hours and the seed spread was already an order of magnitude below the 60-minute margin to the nextbest published model. Intervals are 95% Student-� intervals over seeds. They describe training variability, not uncertainty over test windows. PEMS04 and PEMS08 results are single runs with seed 42 and carry no interval. Baseline papers report single numbers, so comparisons with them are numerical and carry no significance claim. Supplementary Table S2 lists every run.

## 7. Results

This section answers the four research questions. RQ1 compares ST-GFM with twelve published baselines on the speed datasets METR-LA and PEMS-BAY, with published baselines on the flow datasets PEMS04 and PEMS08, and with models retrained under our own evaluation pipeline. RQ2 removes one component at a time. RQ3 perturbs the graph operator and compares the measured response of the first DCSE layer with the bound of Section 5. RQ4 locates frequency shift in the test data and compares accuracy on weekend windows with accuracy on the full test set. The section ends with an exploratory transfer to industry portfolio returns and the computational cost of the model.

## 7.1. RQ1: Forecasting Accuracy 7.1.1. Speed Datasets

Tables 5 and 6 compare ST-GFM with published numbers. Figure 4 places ST-GFM against every baseline that reports both speed benchmarks, and Figure 5 plots MAE against horizon. On PEMS-BAY, ST-GFM has the lowest MAE and MAPE at every horizon among all thirteen methods. Its 60-minute MAE of $1 . 5 7 6 \pm 0 . 0 0 6$ is 3.3% below T-Graphormer, the next best, 6.7% below TITAN and 11.0% below STD-MAE.

METR-LA behaves diferently. TITAN reports lower values than ST-GFM in every cell, with a margin that narrows with horizon from 0.144 MAE at 15 minutes to 0.018 at 60, about one seed interval. ST-GFM has the secondlowest MAE and MAPE at 30 and 60 minutes, and its 60- minute MAE is 2.9% below T-Graphormer and 7.2% below STAEformer. At 15 minutes MegaCRN is also lower, 2.52 against $2 . 5 5 4 { \scriptstyle \pm 0 . 0 1 1 }$ . T-Graphormer has the lowest RMSE at every horizon on PEMS-BAY and at 60 minutes on METR-LA. RMSE weights the largest errors, which on freeway speed data mostly come from rare incident-driven drops. ST-GFM is trained on MAE and anchored to a persistence term, and it lowers typical errors more than the largest ones.

Table 5  
Forecasting Results on METR-LA. ST-GFM Entries Are Five-Seed Means With 95% Seed Intervals. Bold Marks the Best and Underline the Second Best in Each Column. Baseline Numbers Are From the Original Papers.
<table><tr><td rowspan="2">Method</td><td colspan="3">H = 3 (15 min)</td><td colspan="3">H = 6 (30 min)</td><td colspan="3">H = 12 (60 min)</td></tr><tr><td>MAE</td><td>RMSE</td><td>MAPE (%)</td><td>MAE</td><td>RMSE</td><td>MAPE (%)</td><td>MAE</td><td>RMSE</td><td>MAPE (%)</td></tr><tr><td>DCRNN [36]</td><td>2.77</td><td>5.38</td><td>7.30</td><td>3.15</td><td>6.45</td><td>8.80</td><td>3.60</td><td>7.59</td><td>10.50</td></tr><tr><td>Graph WaveNet [56]</td><td>2.69</td><td>5.15</td><td>6.90</td><td>3.07</td><td>6.22</td><td>8.37</td><td>3.53</td><td>7.37</td><td>10.01</td></tr><tr><td>MTGNN [55]</td><td>2.69</td><td>5.18</td><td>6.86</td><td>3.05</td><td>6.17</td><td>8.19</td><td>3.49</td><td>7.23</td><td>9.87</td></tr><tr><td>STEP [46]</td><td>2.61</td><td>4.98</td><td>6.60</td><td>2.96</td><td>5.97</td><td>7.96</td><td>3.37</td><td>6.99</td><td>9.61</td></tr><tr><td>D²STGNN [47]</td><td>2.56</td><td>4.88</td><td>6.48</td><td>2.90</td><td>5.89</td><td>7.78</td><td>3.35</td><td>7.03</td><td>9.40</td></tr><tr><td>MegaCRN [25]</td><td>2.52</td><td>4.94</td><td>6.44</td><td>2.93</td><td>6.06</td><td>7.96</td><td>3.38</td><td>7.23</td><td>9.72</td></tr><tr><td>Trafformer [26]</td><td>2.78</td><td>5.35</td><td>7.32</td><td>3.05</td><td>6.18</td><td>8.67</td><td>3.41</td><td>7.17</td><td>9.96</td></tr><tr><td>STAEformer [37]</td><td>2.65</td><td>5.11</td><td>6.85</td><td>2.97</td><td>6.00</td><td>8.13</td><td>3.34</td><td>7.02</td><td>9.70</td></tr><tr><td>STD-MAE [17]</td><td>2.62</td><td>5.02</td><td>6.70</td><td>2.99</td><td>6.07</td><td>8.04</td><td>3.40</td><td>7.07</td><td>9.59</td></tr><tr><td>AdpSTGCN [8]</td><td>2.59</td><td>5.11</td><td>6.68</td><td>2.96</td><td>6.08</td><td>8.02</td><td>3.40</td><td>7.21</td><td>9.45</td></tr><tr><td>TITAN [51]</td><td>2.41</td><td>4.43</td><td>5.85</td><td>2.72</td><td>5.33</td><td>7.07</td><td>3.08</td><td>6.21</td><td>8.43</td></tr><tr><td>T-Graphormer [2]</td><td>2.62</td><td>4.93</td><td>6.61</td><td>2.90</td><td>5.53</td><td>7.56</td><td>3.19</td><td>6.12</td><td>8.62</td></tr><tr><td>ST-GFM (ours)</td><td>2.554±.011</td><td>4.768±.023</td><td>6.39±.07</td><td>2.800±.012</td><td>5.480±.039</td><td>7.30±.11</td><td>3.098±.017</td><td>6.300±.052</td><td>8.47±.13</td></tr></table>

![](images/ad4a440e53c25789c83ae44040b2e7eb936e75ad8dc102cb8581159468778170.jpg)  
Figure 4: 60-minute MAE on METR-LA and PEMS-BAY for every compared model that reports both datasets. Lower left is better. The shaded region holds the models with a higher MAE than ST-GFM on both datasets. Gray dots are published values from the original papers. The open pentagon is STAEformer retrained under our pipeline (Table 8).

On METR-LA the relative MAE reduction grows with horizon against Graph WaveNet, STEP, D<sup>2</sup>STGNN, STAEformer and MegaCRN in Figure 5. It rises from 5.0% to 12.2% against Graph WaveNet and from 3.6% to 7.2% against STAEformer. Against TITAN it remains negative, changing from −6.0% at 15 minutes to −0.6% at 60 minutes. The reduction against T-Graphormer is 2.5%, 3.4% and 2.9% at the three horizons, so it is not monotonic. On PEMS-BAY there is no consistent trend. The margin over Graph WaveNet stays near 19% from 30 minutes on, and the margin over STEP moves between 12% and 14.5%. Supplementary Table S2 lists every run. On METR-LA the seed standard deviation never exceeds 0.014 MAE, 0.042 RMSE and 0.11 percentage points of MAPE. On PEMS-BAY it never exceeds 0.002 MAE, 0.021 RMSE and 0.03 points ofMAPE. On PEMS-BAY the 60-minute interval, ±0.006, is about one ninth of the gap to T-Graphormer. On METR-LA the interval, ±0.017, is about the size of the gap to TITAN.

## 7.1.2. Flow Datasets

Table 7 compares ST-GFM with the two baselines that use a 12-step input and report the average over all 12 horizons on PEMS04 and PEMS08. ST-GFM is lower than both on all six values, by 1.0% to 2.1% against STAEformer and 0.4% to 3.5% against PDFormer. ST-GFM is a single run here with no seed interval, so we cannot say whether the smallest margins, 0.12 RMSE against PDFormer and 0.19 MAE against STAEformer on PEMS04, exceed run-to-run variation. The model was designed and tuned on speed data, and the flow result tests the architecture on a signal with a diferent range and diferent noise. STD-MAE [17], which pre-trains on several days of history while ST-GFM sees 12 steps, has a lower PEMS04 MAE and RMSE and a lower PEMS08 RMSE. STEP reports a PEMS04 RMSE of 29.71, also below ST-GFM. ST-GFM is lower than STD-MAE on PEMS08 MAE and on MAPE for both datasets, by 0.03 points of MAPE on PEMS08.

## 7.1.3. Same-Pipeline Comparison

Published numbers come from diferent code bases, so a margin over them mixes architecture with pipeline. Table 8 scores reference models with our own preprocessing, windows, mask and metric code. The comparison shares data handling and evaluation. Training follows each model’s own recipe. Persistence repeats the last observed reading at every horizon. The historical average predicts, for each sensor, the mean of its non-zero training readings at the same fiveminute slot of the week. Our historical average gives 4.19 on METR-LA, close to the 4.16 reported with DCRNN [36], which checks the pipeline.

At 60 minutes ST-GFM is 39.8% below persistence and 26.1% below the historical average on METR-LA, and 27.5% and 40.1% below them on PEMS-BAY. Both retrained models score better under our pipeline than in their papers. The reduced Graph WaveNet reaches 3.258 against the 3.53 published for the full model, and STAEformer reaches 3.21 on METR-LA and 1.81 on PEMS-BAY against the published 3.34 and 1.88. Part of the margin over published numbers therefore need not come from the architecture. The gap between published and retrained values could come from our preprocessing, which standardises each sensor with its own training statistics and masks readings at or below 1 (Section 6), from software versions, from run-to-run variation in single runs, or, for Graph WaveNet, from the reduced configuration. The gap, 3.7% to 7.7% on the two retrained models, is as large as the margins of ST-GFM over the strongest published baselines. We therefore treat the samepipeline numbers as the primary comparison of forecasting performance under shared data handling and evaluation, and the published-number comparisons as context. Under shared evaluation code, ST-GFM is 4.9% below the reduced Graph WaveNet and 3.5% below STAEformer at 60 minutes on

Table 6  
Forecasting Results on PEMS-BAY. ST-GFM Entries Are Three-Seed Means With 95% Seed Intervals. Formatting as in Table 5.
<table><tr><td></td><td colspan="3">H = 3 (15 min)</td><td colspan="3">H = 6 (30 min)</td><td colspan="3">H = 12 (60 min)</td></tr><tr><td>Method</td><td>MAE</td><td>RMSE</td><td>MAPE (%)</td><td>MAE</td><td>RMSE</td><td>MAPE (%)</td><td>MAE</td><td>RMSE</td><td>MAPE (%)</td></tr><tr><td>DCRNN [36]</td><td>1.38</td><td>2.95</td><td>2.90</td><td>1.74</td><td>3.97</td><td>3.90</td><td>2.07</td><td>4.74</td><td>4.90</td></tr><tr><td>Graph WaveNet [56]</td><td>1.30</td><td>2.74</td><td>2.73</td><td>1.63</td><td>3.70</td><td>3.67</td><td>1.95</td><td>4.52</td><td>4.63</td></tr><tr><td>MTGNN [55]</td><td>1.32</td><td>2.79</td><td>2.77</td><td>1.65</td><td>3.74</td><td>3.69</td><td>1.94</td><td>4.49</td><td>4.53</td></tr><tr><td>STEP [46]</td><td>1.26</td><td>2.73</td><td>2.59</td><td>1.55</td><td>3.58</td><td>3.43</td><td>1.79</td><td>4.20</td><td>4.18</td></tr><tr><td>D2STGNN [47]</td><td>1.24</td><td>2.60</td><td>2.58</td><td>1.55</td><td>3.52</td><td>3.49</td><td>1.85</td><td>4.30</td><td>4.37</td></tr><tr><td>MegaCRN [25]</td><td>1.28</td><td>2.72</td><td>2.67</td><td>1.60</td><td>3.68</td><td>3.57</td><td>1.88</td><td>4.42</td><td>4.41</td></tr><tr><td>Trafformer [26]</td><td>1.31</td><td>2.83</td><td>2.92</td><td>1.61</td><td>3.74</td><td>3.82</td><td>1.88</td><td>4.38</td><td>4.59</td></tr><tr><td>STAEformer [37]</td><td>1.31</td><td>2.78</td><td>2.76</td><td>1.62</td><td>3.68</td><td>3.62</td><td>1.88</td><td>4.34</td><td>4.41</td></tr><tr><td>STD-MAE [17]</td><td>1.23</td><td>2.62</td><td>2.56</td><td>1.53</td><td>3.53</td><td>3.42</td><td>1.77</td><td>4.20</td><td>4.17</td></tr><tr><td>AdpSTGCN [8]</td><td>1.29</td><td>2.75</td><td>2.69</td><td>1.64</td><td>3.69</td><td>3.67</td><td>1.92</td><td>4.49</td><td>4.62</td></tr><tr><td>TITAN [51]</td><td>1.12</td><td>2.21</td><td>2.29</td><td>1.46</td><td>3.24</td><td>3.15</td><td>1.69</td><td>3.79</td><td>3.85</td></tr><tr><td>T-Graphormer [2]</td><td>1.16</td><td>2.10</td><td>2.38</td><td>1.38</td><td>2.64</td><td>2.97</td><td>1.63</td><td>3.20</td><td>3.65</td></tr></table>

![](images/be424d728e8b6c378a768370d4fd9d0b2fc6746d7d799285f2dc85806d48b8ba.jpg)

![](images/b7d6da4d6a9a5aa8d1ce7903ff8aad237417097663a6a6b78f5b1d2a6ae18ebc.jpg)

(c) METR-LA, MAE reduction of ST-GFM  
![](images/e001cf76f31a3cda664571c88cf00484ab98c29acc84c9bfff6321ee739339ca.jpg)

(d) PEMS-BAY, MAE reduction of ST-GFM  
![](images/7c9725c8b27ea4d8a2f53a931f8768b143c83c8359235a791c087a08a56c89c3.jpg)  
Figure 5: Accuracy by horizon against seven baselines, including TITAN and T-Graphormer. (a) and (b) MAE of ST-GFM and the baselines, with 95% seed intervals on ST-GFM (five seeds on METR-LA, three on PEMS-BAY, smaller than the marker on PEMS-BAY). (c) and (d) Relative MAE reduction of ST-GFM over each baseline, computed from unrounded seed means and the values in Tables 5 and 6. Baselines are single published numbers.

Table 8  
Table 9  
Table 7  
Average Over All 12 Horizons on the Flow Datasets PEMS04 and PEMS08. Bold Marks the Best and Underline the Second Best Among Models With a 12-Step Input. STD-MAE Pre-Trains on 864 to 2016 Steps and Is Listed Below the Rule as a Long-History Reference. Baseline Numbers Are From the Original Papers. ST-GFM Is a Single Run (Seed 42).
<table><tr><td></td><td colspan="3">PEMS04</td><td colspan="3">PEMS08</td></tr><tr><td>Method</td><td>MAE</td><td>RMSE</td><td>MAPE (%)</td><td>MAE</td><td>RMSE</td><td>MAPE (%)</td></tr><tr><td>PDFormer [24]</td><td>18.32</td><td>29.97</td><td>12.10</td><td>13.58</td><td>23.51</td><td>9.05</td></tr><tr><td>STAEformer [37]</td><td>18.22</td><td>30.18</td><td>11.98</td><td>13.46</td><td>23.25</td><td>8.88</td></tr><tr><td>ST-GFM (ours)</td><td>18.03</td><td>29.85</td><td>11.79</td><td>13.18</td><td>22.91</td><td>8.73</td></tr><tr><td>STD-MAE [17]</td><td>17.80</td><td>29.25</td><td>11.97</td><td>13.44</td><td>22.47</td><td>8.76</td></tr></table>

MAE Under Our Pipeline, With the Same Test Windows, Code and Mask for Every Model. Graph WaveNet Is the Reduced Model of Section 6. ST-GFM Is the Mean Over Seeds. Bold Marks the Lowest MAE per Column, Including Ties at the Reported Precision.
<table><tr><td></td><td colspan="3">METR-LA</td><td colspan="3">PEMS-BAY</td></tr><tr><td>Method</td><td>15 min</td><td>30 min</td><td>60 min</td><td>15 min</td><td>30 min</td><td>60 min</td></tr><tr><td>Persistence</td><td>3.557</td><td>4.153</td><td>5.145</td><td>1.301</td><td>1.649</td><td>2.173</td></tr><tr><td>Historical average</td><td>4.193</td><td>4.193</td><td>4.193</td><td>2.631</td><td>2.631</td><td>2.630</td></tr><tr><td>Graph WaveNet</td><td></td><td></td><td>3.258</td><td></td><td></td><td></td></tr><tr><td>STAEformer</td><td>2.55</td><td>2.86</td><td>3.21</td><td>1.26</td><td>1.56</td><td>1.81</td></tr><tr><td>ST-GFM (ours)</td><td>2.554</td><td>2.800</td><td>3.098</td><td>1.098</td><td>1.325</td><td>1.576</td></tr></table>

METR-LA, against 12.2% and 7.2% below the published values. On METR-LA the margin over STAEformer narrows at shorter horizons. It is 2.1% at 30 minutes, and at 15 minutes the two are level at the precision of the retrained value, 2.55 against 2.554. On PEMS-BAY the gap stays large under the shared pipeline, between 12.9% and 15.0% at every horizon and 13.0% at 60 minutes.

## 7.2. RQ2: Component Contributions

Table 9 removes one component at a time at � = 12, and Supplementary Figure S2 plots the increases. Each variant is a single run trained with the early-stopping rule, validation schedule and five-checkpoint averaging of the full model, and every MAE increase in the table is at least five times the seed standard deviation of the full model on its dataset, which we use as a proxy for the run-to-run spread of the variants. Replacing the DCSE layers with a standard GCN and removing MSR give the two largest singlecomponent increases on both datasets: +0.32 and +0.30 MAE on METR-LA, +0.21 and +0.22 on PEMS-BAY. Their order swaps between datasets and the gaps between them are 0.02 and 0.01, so we treat the spectral encoder and MSR as jointly dominant.

Removing all three self-supervised branches at once costs 0.33 on METR-LA and 0.24 on PEMS-BAY, only 0.03 and 0.02 more than removing MSR alone. The three singlebranch increases sum to 0.47 and 0.30, well above the joint figure, so the branch efects are not additive, and removing

Ablation at � = 12. Δ Is the MAE Increase Over the Full Model. The Full-Model Reference Is the Multi-Seed Mean of Tables 5 and 6, and Each Variant Is a Single Run.
<table><tr><td rowspan="2">Variant</td><td colspan="2">METR-LA</td><td colspan="2">PEMS-BAY</td></tr><tr><td>MAE</td><td>Δ</td><td>MAE</td><td>Δ</td></tr><tr><td>Full ST-GFM</td><td>3.098</td><td></td><td>1.576</td><td>一</td></tr><tr><td>w/o decomposition</td><td>3.29</td><td>+0.19</td><td>1.71</td><td>+0.13</td></tr><tr><td>w/o DCSE (GCN instead)</td><td>3.42</td><td>+0.32</td><td>1.79</td><td>+0.21</td></tr><tr><td>w/o MSR</td><td>3.40</td><td>+0.30</td><td>1.80</td><td>+0.22</td></tr><tr><td>w/o masked nodes</td><td>3.20</td><td>+0.10</td><td>1.63</td><td>+0.05</td></tr><tr><td>w/o temporal contrast</td><td>3.17</td><td>+0.07</td><td>1.61</td><td>+0.03</td></tr><tr><td>w/o all three SSL branches</td><td>3.43</td><td>+0.33</td><td>1.82</td><td>+0.24</td></tr><tr><td>time axis collapsed in decomposition</td><td>3.22</td><td>+0.12</td><td>1.66</td><td>+0.08</td></tr></table>

MSR alone accounts for most of the error increase seen when all three branches are dropped. These are ratios between ablations and not a split of the gain among branches. The masked-node and contrastive branches add 0.10 and 0.07 when removed on their own from METR-LA and less on PEMS-BAY, and their efects overlap with that of MSR.

The decomposition comes third (+0.19 and +0.13). In this ablation, collapsing the time axis alone while keeping the three stacks produces more than 60% of the error increase caused by removing the entire decomposition: 0.12 against 0.19 on METR-LA and 0.08 against 0.13 on PEMS-BAY. Keeping all 12 steps for the graph layers, the design choice of Section 4.1, matters on its own.

The variant without DCSE reaches 3.42 on METR-LA. This is below the 3.53 published for Graph WaveNet but above the 3.258 of the reduced Graph WaveNet under our pipeline (Table 8), so the comparison does not isolate how much the components shared by all variants contribute. STID [45] has shown how much identity embeddings alone deliver on these datasets.

MSR is active only in the first 40 epochs, and Figure 6 shows why that timing matters. In all eight runs the MSR loss is between 0.035 and 0.046 at epoch 20 and still below 0.35 at epoch 40, when the self-supervised weight reaches zero. By epoch 60 it has risen to between 2.65 and 6.21, the contrastive loss has followed the same path, and the masked-node loss lies between 0.05 and 0.08. Validation MAE is lowest between epochs 35 and 45 in every run and climbs once the self-supervised terms are switched of, although the supervised training loss keeps falling, from about 0.15 to 0.13 on METR-LA and from 0.17 to 0.15 on PEMS-BAY between epochs 40 and 60. All 40 averaged checkpoints come from epochs 30 to 60, and 35 of them from epochs 30 to 50, so the deployed model sits close to the end of the self-supervised phase. This fits MSR and the contrastive term acting as regularisers of the encoder while they are active. Once the self-supervised weight reaches zero at epoch 40, the monitored MSR loss rises in every run while the supervised training loss keeps falling, and the averaged checkpoints come from around this transition. We therefore read the MSR ablation gain as an efect of the objective and of the training path it induces. It does not show that the deployed encoder keeps the spectral alignment. The variant without MSR keeps the masked-node and contrastive losses on the same 40-epoch schedule and still loses 0.30 MAE, so the auxiliary schedule alone does not explain the gain.

(a) METR-LA: self-supervised losses  
![](images/be0640b641bfc1cfc71559f120f0353d986e3e5a0205882cb7f530fdd60bd67f.jpg)  
(c) METR-LA: validation MAE

(b) PEMS-BAY: self-supervised losses  
![](images/025c191d4b4323f926749d990fec88d1ddb725bc078a5257ca96792770cd5fa6.jpg)

![](images/efc41b7c0386d5fd6afc76f344c438320192ca8c7267f4c20ae026b5f53597fa.jpg)

(d) PEMS-BAY: validation MAE  
![](images/d7c4f586606a1e32077131e2bd0c3d0bdeafffd730c7fc98c8502dfe932964e9.jpg)  
Figure 6: Self-supervised losses (a), (b) and validation MAE (c), (d) over training, for all eight runs. Thick lines are means, bands span the runs, thin lines are single runs and dots mark each run’s best epoch. Shading marks the SSL-only phase and the ramp-down of its weight. After epoch 40 the SSL losses are monitored, not optimised. Brackets count the averaged checkpoints that fall in epochs 30 to 50.

## 7.3. RQ3: Sensitivity to Operator Perturbations 7.3.1. Random Operator Perturbations

We add a dense random symmetric Δ� with $\| \Delta L \| _ { \mathrm { F } } = \varepsilon .$ for $\varepsilon \in \{ 0 , 0 . 0 5 , 0 . 1 , 0 . 2 , 0 . 3 , 0 . 5 , 0 . 7 5 , 1 . 0 \}$ , to the trained operator ${ \mathfrak { s } } _ { \theta }$ and evaluate the unchanged test inputs. The perturbation is symmetric and the trained operator is not, which Theorem 1 allows. For a dense Gaussian perturbation $\eta \ = \ \| \Delta L \| _ { 2 } \ \approx \ 2 \varepsilon / \sqrt { N }$ , about 0.14� at $N \ = \ 2 0 7$ . The sweep therefore reaches � of at most about 0.14 on METR-LA and 0.11 on PEMS-BAY, which is $\delta ^ { \mathrm { t o p } } \approx 0 . 1 1$ and 0.08. At the measured operator norm $c _ { \mathrm { c h e b } } ( \rho , \eta )$ stays below 18.8 on METR-LA (Supplementary Figure S1).

Figure 7 shows every run on both datasets. On METR-LA the mean MAE over five runs moves from 3.102 at � = 0 to 3.107 at $\varepsilon \ = \ 1$ , a change of 0.19%, and no single run changes by more than 0.25%. On PEMS-BAY the mean over three runs moves from 1.575 to 1.579, a change of 0.23%, and no run changes by more than 0.35%. Percentages are computed from unrounded values. All levels of � are scored in one evaluation pass, and each level is compared with the $\varepsilon ~ = ~ 0$ value of that pass. That baseline difers from the clean-test means of Tables 5 and 6 by 0.004 on METR-LA and 0.001 on PEMS-BAY, which is of the same order as the largest perturbation efect. The sweep therefore supports a flat response and does not resolve its shape. On METR-LA the spread between seeds is also larger than the efect of the largest perturbation.

Theorem 1 and Proposition 2 bound the encoder representation, while Figure 7 measures forecast error after the aggregation layer, the decoder and the persistence term. The two quantities live in diferent units, so this sweep does not test the constant of (14). The edge-dropout study below compares the Chebyshev-branch response with its corresponding bound at the realised perturbation sizes.

## 7.3.2. Matched Adjacency Perturbations

To compare sensitivity with a baseline under identical conditions, we apply the same random symmetric perturbations with $\| \Delta A \| _ { \mathrm { F } } ~ = ~ \varepsilon$ to the static adjacency of ST-GFM and of the reduced Graph WaveNet, using the seed-101 checkpoint of each and averaging two perturbation draws per level (Table 10). In ST-GFM the static adjacency enters $A _ { \theta }$ with weight $1 - \alpha \ : = \ : 0 . 5 5$ (4) before degree normalisation and spectral rescaling, so the change that reaches the operator is not a fixed multiple of �. Both models are almost flat. ST-GFM moves from 3.076 to 3.087 (+0.36%) and Graph WaveNet from 3.258 to 3.264 (+0.18%). The relative MAE increase is about twice as large for ST-GFM. The models transform the perturbed adjacency diferently, so these percentages do not isolate the sensitivity of the operators themselves. The accuracy margin between the two holds across the whole sweep, from 0.182 at � = 0 to 0.177 at � = 1, so the gain over Graph WaveNet does not depend on a clean graph.

![](images/84201211fb17cf9e26334cbfe7303200fc949c9e3e33b25adc8cf6bdec8eab47.jpg)

![](images/51e33d00f2ed047993ee3cab3cb56608883c9d54bab16046f3365496ab9171d4.jpg)  
Figure 7: Test MAE of ST-GFM under random symmetric perturbations of the operator with Frobenius norm �. Thick lines: mean over runs, bands: one standard deviation, thin lines: individual runs. One perturbation draw per run and level.

Matched Adjacency Perturbations on METR-LA at $H = 1 2$ (Seed-101 Checkpoints, Mean of Two Draws). The Graph WaveNet Entries Are From the Reduced Model Retrained Under Our Preprocessing and Evaluation.
<table><tr><td>ε</td><td>Graph WaveNet</td><td>ST-GFM</td><td>Gap</td></tr><tr><td>0.00</td><td>3.258</td><td>3.076</td><td>0.182</td></tr><tr><td>0.05</td><td>3.259</td><td>3.076</td><td>0.183</td></tr><tr><td>0.10</td><td>3.259</td><td>3.076</td><td>0.183</td></tr><tr><td>0.20</td><td>3.259</td><td>3.077</td><td>0.182</td></tr><tr><td>0.30</td><td>3.260</td><td>3.079</td><td>0.181</td></tr><tr><td>0.50</td><td>3.261</td><td>3.080</td><td>0.181</td></tr><tr><td>0.75</td><td>3.262</td><td>3.084</td><td>0.178</td></tr><tr><td>1.00</td><td>3.264</td><td>3.087</td><td>0.177</td></tr></table>

## 7.3.3. Edge Dropout Scored at the First DCSE Layer

To test the constant of (14) directly, we apply a structured perturbation and read its efect at the layer the bound describes. We remove a fraction $p \in \{ 0 . 0 5 , 0 . 1 , 0 . 2 , 0 . 3 , 0 . 5 \}$ of the edges of $A _ { \mathrm { s t a t i c } }$ at random, rebuild the anchor operator, and compare the Chebyshev-branch output of the first DCSE layer with and without the edit on the unchanged test inputs. For each � we record the realised ‖ $| \Delta L | | _ { \mathrm { F } }$ and �, the observed change $\| g ( \ll \ + \Delta L ) - g ( \ll ) \| _ { \mathrm { F } } / \| Z \| _ { 2 }$ averaged over time slices, the right-hand side of (14) at the measured $\| \ll _ { \theta } \| _ { 2 } ,$ and the forecast MAE. Figure 8 reports the result, averaged over all runs and three dropout draws, and Supplementary Table S3 lists every value.

Edge dropout moves the operator much further than the random sweep. At $p = 0 . 0 5$ the realised $| \Delta L | | _ { \mathrm { F } }$ is already 1.40 on METR-LA and 2.70 on PEMS-BAY, above the largest random budget, because the degree normalisation and the rescaling by �<sup>̂</sup> spread each removed edge over whole rows of the operator (Section 5.2). At $p \ = \ 0 . 5$ it reaches 5.83 and 9.64. The edit is also more concentrated in spectral norm: $\eta / \Vert \Delta L \Vert _ { \mathrm { F } }$ lies between 0.13 and 0.33, against about 0.14 and 0.11 for the dense random draws on the two networks.

The observed change grows slightly less than linearly in $\Vert \Delta L \Vert _ { \mathrm { F } } .$ . Its ratio to $\Vert \Delta L \Vert _ { \mathrm { F } }$ falls from 0.65 to 0.57 on METR-$\mathrm { L A }$ and from 0.82 to 0.61 on PEMS-BAY as $p$ rises from 0.05 to 0.5. The bound grows slightly faster than linearly, because $c _ { \mathrm { c h e b } }$ increases with �. The observed change is therefore between 2.4% and 3.1% ofthe bound on METR-LA and between 2.0% and 3.1% on PEMS-BAY, and the ratio falls as the edit grows. A ratio near 3% indicates that random edge edits lie far from the worst case of the bound, which Proposition 1 attains only on a constructed operator and input. Across a sevenfold range of $\Vert \Delta L \Vert _ { \mathrm { F } }$ the ratio moves by a factor of about 1.6, far less than the size of the edit.

At the forecast, MAE rises by 1.0% at $p = 0 . 3$ and 1.4% at $p = 0 . 5$ on METR-LA, and by 2.5% and 3.1% on PEMS-BAY. These changes are larger than in the random sweep, in line with the larger operator change, and PEMS-BAY is the more sensitive network at every level. With half of the static edges removed, the 60-minute MAE is still 3.144 on METR-LA and 1.624 on PEMS-BAY, below every baseline in Table 6 and every baseline in Table 5 except TITAN. The edit leaves the learned part of $A _ { \theta } ,$ , which carries weight $\alpha = 0 . 4 5$ , unchanged.

## 7.4. RQ4: Frequency Shift Across Regimes 7.4.1. Where Frequency Shift Occurs

For every test window of both datasets we computed $\delta ^ { \mathrm { f r e q } }$ of Definition 1, with �<sup>̄</sup> estimated from 20,000 training windows drawn at random. Figure 9 groups the windows by the hour of the forecast origin and by weekday or weekend. Spectral shift concentrates in weekday peak hours, when queues form and clear inside the 12-step window. On METR-LA the median $\delta ^ { \mathrm { f r e q } }$ is 2.43 on weekdays and 1.53 on weekends, and on PEMS-BAY it is 3.41 and 1.25. A twosided Mann–Whitney � test gives $\textit { p } < 1 0 ^ { - 2 0 }$ on both datasets. Consecutive windows overlap in eleven of twelve steps and are not independent, so this �-value is nominal, and the evidence is the separation of the medians and interquartile ranges in Figure 9. Weekday peaks form the high-shift regime and weekends the low-shift one.

![](images/445f197e6a415c1a2d558e18d61f5f5c71abe9e07a576d85e7de2b75d823575f.jpg)

![](images/f6c24b78ce36a510b558c22caac16f5b340b95eef91f418cc15c17f0b3fac7fd.jpg)

![](images/6addcf6bb474704e56acff396daeadbd693baba9e216510430e508f0583d3728.jpg)  
Figure 8: Edge dropout on the static adjacency, mean over runs and three draws (values in Supplementary Table S3). (a) Observed relative change of the Chebyshev-branch output in the first DCSE layer and the bound of (14) at the realised perturbation and the measured operator norm, on a log scale. Shading spans the gap on METR-LA. (b) Observed change as a percentage of the bound. (c) Relative increase of the 60-minute forecast MAE, with the absolute MAE at $p = 0$ and $p = 0 . 5$ annotated.

![](images/6584d58850fb9910c7fc89b315dc30e9b66e2a323dadf1c2a6234e26ce780788.jpg)

![](images/8a7f7e7a35cd98f47992b41b2e7637872986d77390f5cdc27f32f10b5a9898b7.jpg)  
Figure 9: Spectral shift $\delta ^ { \mathrm { f r e q } }$ of the test windows by hour of the forecast origin. Lines are medians and bands interquartile ranges, which show spread, not confidence intervals. A weekend hour holds about 72 windows on METR-LA and 120 on PEMS-BAY, a weekday hour about 216 and 312.

## 7.4.2. Accuracy Across Regimes

Table 11 scores two naive forecasters on the same windows with the same evaluation code and mask as ST-GFM. Weekend windows are the easy regime as well as the lowshift one. The historical average improves by 16.7% on METR-LA weekends and by 35.5% on PEMS-BAY weekends, and persistence improves by 32 to 46% on PEMS-BAY. On METR-LA, ST-GFM (seed 42, 60 minutes) moves from 3.112 on the full test set to 2.581 on weekends, an improvement of 17.1% and almost exactly the gain of the historical average. Its relative margin over the historical average is therefore almost unchanged, 25.8% on the full test set and 26.1% on weekends, and its margin over persistence grows from 39.5% to 49.1% because persistence gains little from the weekend on this network. ST-GFM tracks the easier regime. The full test set is dominated by weekday windows, which contain the high-shift peak hours, and ST-GFM keeps the same relative lead over the historical average on the full test set as on weekends. We did not score weekday peak windows separately. RQ4 characterises where frequency shift occurs and how the full model fares on weekends relative to the full test set.

## 7.5. Exploratory Transfer to Industry Portfolio Returns

This exploratory study asks whether an encoder pretrained only on trafic can support forecasting on a graph with diferent structure and dynamics. It sits outside the four research questions, and we do not claim that ST-GFM is a strong return model. The graph links the 43 portfolios within nine broad sector groups formed from the Fama– French industry definitions, which gives 218 edges, and enters the model as $A _ { \mathrm { s t a t i c } } ,$ . Weights are transferred by name and shape. Fine-tuning trains the second DCSE layer, the second temporal-attention block, the aggregation layer, the decoder with its node embedding and the persistence term, 3.1 million parameters in all. The decomposition, the input projection, the time and step embeddings, the gated skip and the first DCSE and attention pair stay frozen at their trafic values. Parameters whose shape depends on the number of nodes are re-initialised, and of these only the decoder node embedding is trained. The additive node embedding, the degree embedding and the adaptive-graph vectors $E _ { 1 } , E _ { 2 }$ stay frozen at their new random values, and $\alpha _ { 0 }$ is set to 0. The operator the DCSE layers see is therefore half sector graph and half softmax(ReLU $\mathbb { E } _ { 1 } E _ { 2 } ^ { \top } ) \mathbb { \ Z }$ ) built from random vectors, and the transferred encoder is not a fully pretrained backbone. Transferred weights train at a learning rate of $3 \times 1 0 ^ { - 5 }$ and the decoder node embedding at $3 \times 1 0 ^ { - 4 }$ . The loss combines Pearson IC, rank IC and cross-sectional MSE, and early stopping uses validation IC. Each window holds 12 months of input and 12 months of targets, and we score the one-month-ahead forecast. Test windows are cut from the post-2010 data alone, so the scored targets are the 172 months from January 2011 to April 2025.

Table 11  
MAE of Two Naive Forecasters on the Full Test Sets and on Weekend Windows (First Forecast Step on Saturday or Sunday), Scored With the Same Windows, Code and Mask as ST-GFM. Δ Is the Change From Full to Weekend.
<table><tr><td></td><td colspan="3"> $H = 3$ </td><td colspan="3"> $H = 6$ </td><td colspan="3"> $H = 1 2$ </td></tr><tr><td>Method</td><td>Full</td><td>Weekend</td><td>∆(%)</td><td>Full</td><td>Weekend</td><td>∆(%)</td><td>Full</td><td>Weekend</td><td>∆(%)</td></tr><tr><td>METR-LA</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Persistence (last reading)</td><td>3.557</td><td>3.337</td><td>-6.2</td><td>4.153</td><td>3.971</td><td>-4.4</td><td>5.145</td><td>5.073</td><td>-1.4</td></tr><tr><td>Historical average (time of week)</td><td>4.193</td><td>3.496</td><td>-16.6</td><td>4.193</td><td>3.493</td><td>-16.7</td><td>4.193</td><td>3.494</td><td>-16.7</td></tr><tr><td>PEMS-BAY</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Persistence (last reading)</td><td>1.301</td><td>0.887</td><td>-31.8</td><td>1.649</td><td>1.016</td><td>-38.4</td><td>2.173</td><td>1.169</td><td>-46.2</td></tr><tr><td>Historical average (time of week)</td><td>2.631</td><td>1.698</td><td>-35.5</td><td>2.631</td><td>1.697</td><td>-35.5</td><td>2.630</td><td>1.697</td><td>-35.5</td></tr></table>

Table 12 reports, for two encoders per trafic dataset, the mean monthly cross-sectional information coeficient (IC), its rank version, the information ratio of the IC series, and the annualised Sharpe ratio of a long-short decile portfolio net of 1 basis point of cost. A control repeats the fine-tuning from a random initialisation of the same architecture, with two runs. Figure 10 plots IC and Sharpe ratio for all six runs.

All four transferred runs give a mean IC significantly above zero, with IC between 0.043 and 0.059 and Sharpe ratios between 0.54 and 0.71. The result survives a Holm correction across the four runs (largest adjusted � = 0.019). The signal is noisy: an information ratio near 0.2 means the monthly IC has a standard deviation about five times its mean.

The random control sets the scale of the transfer efect, and it is modest. The random-initialisation control reaches an IC of 0.042 and 0.036, both significant at the 5% level,

## Table 12

Transfer to 43 Fama–French Industry Portfolios (172 Scored Months, January 2011 to April 2025). Runs #1 and #2 Use METR-LA Seeds 202 and 303 and PEMS-BAY Seeds 101 and 303. R-IC Is Rank IC, IR the IC Information Ratio, SR the Annualised Sharpe Ratio Net of 1 bp and � the One-Sided �-Test of Mean IC.

<table><tr><td>Encoder</td><td>IC</td><td>R-IC</td><td>IR</td><td>SR</td><td>p</td></tr><tr><td>ST-GFM (METR-LA → FF49) #1</td><td>0.055</td><td>0.049</td><td>0.21</td><td>0.57</td><td>0.004</td></tr><tr><td>ST-GFM (METR-LA → FF49) #2</td><td>0.043</td><td>0.041</td><td>0.16</td><td></td><td>0.540.019</td></tr><tr><td>ST-GFM (PEMS-BAY → FF49) #1</td><td>0.052</td><td></td><td></td><td></td><td>0.051 0.20 0.60 0.006</td></tr><tr><td>ST-GFM (PEMS-BAY → FF49) #2</td><td>0.059</td><td>0.057</td><td>0.22</td><td>0.71</td><td>0.003</td></tr><tr><td>Random frozen backbone, #1</td><td></td><td></td><td></td><td>0.042 0.0400.16 0.49</td><td>0.020</td></tr><tr><td>Random frozen backbone, #2</td><td></td><td></td><td></td><td>0.036 0.034 0.14 0.42 0.038</td><td></td></tr></table>

(a) Mean IC  
![](images/0d8e7410f40f5a3fd44332844c19d56f71541a344ae8839b10f5d0af7b163bac.jpg)

(b) Sharpe ratio, net  
![](images/a4ce2de47a6352dfcf717392ffee1d41cf814c9a7ac7c884059d297745d61d5e.jpg)  
Figure 10: Transfer to the 43 Fama–French industry portfolios (Table 12). Blue bars use METR-LA encoders, green bars PEMS-BAY encoders and gray bars the randomly initialised frozen backbone. Dashed lines are group means.

and Sharpe ratios of 0.49 and 0.42. Averaged over runs, trafic pre-training raises the IC from 0.039 to 0.052 and the Sharpe ratio from 0.46 to 0.61. The random controls reach about three quarters of the mean IC of the transferred runs. Both arms use the same design for the re-initialised nodedependent modules. Every transferred run is at or above every control run on both measures, but the weakest transferred run (IC 0.043) and the strongest control (IC 0.042) are close. The small number of trained runs limits assessment of training variability, and we do not report a direct statistical comparison between the transfer and random-initialisation conditions. Under this adaptation protocol, trafic initialisation gives a consistent numerical advantage over random initialisation.

![](images/0704674be60981051f21aab1c1e9eadff261e67fa09c604c4972aa077080d865.jpg)  
Figure 11: Accuracy against model size on METR-LA at 60 minutes. Each label gives the width, the 60-minute MAE and the training time on one RTX 5060 Ti. Circles are single runs with seed 42, and Supplementary Table S4 lists their values. The star is the five-seed mean of the full model. The dashed line is the published STAEformer MAE and the dotted line STAEformer retrained under our pipeline.

## 7.6. Computational Cost

ST-GFM has 5.22 million parameters. A training run takes 3 to 4 hours on METR-LA and 8 to 9 hours on PEMS-BAY on one RTX 5060 Ti, and inference on a batch of 64 windows takes about 15 ms. The model is larger than some baselines, and two observations argue against size as the main source of its accuracy. T-Graphormer reports that its 4.15-million-parameter variant does worse on METR-LA than its 1.57-million one because it overfits [2], and ST-GFM loses 0.30 MAE when MSR is removed with the parameter count unchanged.

Figure 11 varies the hidden width with everything else fixed, using seed 42 at every width, and Supplementary Table S4 lists the values. At � = 128 the model has 0.93 million parameters, trains in 1.2 hours and reaches a 60- minute MAE of 3.15 on METR-LA. That is 0.04 above the full model with the same seed, below every baseline in Table 5 except TITAN, and below STAEformer retrained under our pipeline (3.21). Accuracy stops improving by $d \ : = \ : 2 5 6$ . The single runs give 3.10 at � = 256 and 3.11 at � = 320, a diference smaller than the seed standard deviation of 0.014, so going from � = 256 to � = 320 adds 1.8 million parameters and 1.8 hours of training without lowering the error. At � = 192 the model comes within 0.01 of the full model in less than half its training time.

## 8. Discussion

• Where the accuracy comes from: The same-pipeline comparison separates two cases. On PEMS-BAY the margin over a retrained STAEformer stays between 12.9% and 15.0% at every horizon, so most of the gain survives a shared pipeline. On METR-LA the 60- minute margin falls from 7.2% against the published value to 3.5% against the retrained baseline. At 15 minutes the models are level at the precision reported for the retrained baseline. The spectral encoder and MSR have the largest ablation efects, and collapsing the time axis also increases error. The width study retains competitive accuracy at much smaller sizes, which suggests that parameter count alone does not explain the performance.

• What the operator bounds establish: The numerical content of the analysis is the pair of branch constants computed from the trained model. Proposition 1 shows that the Chebyshev bound is first-order sharp when nothing further is assumed about $\Theta _ { k }$ and $Z .$ The low observed-to-bound ratios show that the bound is conservative for the sampled edge-dropout perturbations. They do not determine the smallest uniform constant for the trained weights. The approximately linear observed response is consistent with the dependence of the bound on perturbation size. The difusion bound is about ten times looser for row-filling edits, mostly through the $\sqrt { s _ { \Delta } }$ factor.

• What the MSR evidence shows: Every run reaches its best validation MAE near the end of the selfsupervised phase, and removing MSR while the other two losses keep the same schedule still costs 0.30 MAE. Both observations are consistent with MSR acting as a regulariser during the early training phases, although they do not establish its mechanism. Its link to the frequency channel rests on the form of the loss (Section 5.4), which keeps the fast, low-power bins of weekday peak windows, the high-shift regime of RQ4, in the objective. Section 9 lists what this evidence does not cover.

## 9. Scope and Limitations

All four datasets come from Californian freeways with at most 325 sensors. The dense learned adjacency costs $O ( N ^ { 2 } )$ per graph multiplication, so networks of the size of LargeST [38] would need a sparse learned operator. Most baseline numbers are published values, and checkpoint averaging, which ST-GFM uses, is not part of the published baseline training procedures. The two retrained baselines score better under our pipeline than in their papers, and we have not isolated the cause of that gain. They are single runs that follow their own training recipes, so the same-pipeline comparison shares data handling and evaluation and not the full training protocol. The ablated variants, the width study and the flow-dataset results are single runs. The random perturbation sweeps use one draw per run and level, and the structured study covers one perturbation family, edge dropout of the static graph, with three draws per level. Edge dropout removes edges and keeps every input reading, so it does not model a failed sensor, which loses both. $\delta ^ { \mathrm { t o p } }$ cannot be measured on real data because the true coupling of a window is not observed, so the operator channel is studied through controlled perturbations. The evidence on the frequency channel is weaker. RQ4 locates frequency shift but scores ST-GFM only on the full test set and on METR-LA weekends, not on weekday peak windows. It does not compare variants with and without MSR across shift levels, and MSR has not been compared with a time-domain reconstruction loss on the same target. MSR also works on seven frequency bins of one readout of the encoder output, which limits the spectral detail it can enforce on a 12-step window. The exploratory transfer study uses 43 portfolios, four transferred runs and two random controls. In both arms the re-initialised node-dependent modules stay frozen at random values, except the decoder node embedding, which is trained.

## 10. Conclusion

ST-GFM splits spectral shift into an operator channel and a frequency channel. For the operator channel, both graph branches admit Frobenius-norm bounds. The Chebyshev constant computed from the trained model is 17.9 on METR-LA and 21.5 on PEMS-BAY, and under the tested edge-dropout perturbations the observed Chebyshev-branch response is at most 3.1% of the corresponding computed bound. The frequency channel motivates MSR, which aligns the log spectra of the encoder output and its input and has one of the two largest ablation efects. With 5.22 million parameters on one consumer GPU, ST-GFM has a lower MAE and MAPE than all twelve published baselines at every evaluated horizon on PEMS-BAY, and a lower MAE than a retrained STAEformer at 30 and 60 minutes on both speed datasets.

The next step is the sparse learned operator. Replacing the dense softmax term with a top-� neighbourhood per node would bring the cost of a graph multiplication from $O ( N ^ { 2 } d )$ to �(���) and would let the bounds, which depend on the number of nonzero entries per row of the perturbation, be stated for networks with thousands of sensors.

## Supplementary Material

Supplementary material for this article is provided as a separate file. It contains the full training configuration (Table S1), the training procedure (Algorithm S1), the results of every individual run (Table S2), the Chebyshev constant against perturbation size (Figure S1), the ablation chart (Figure S2), the edge-dropout values (Table S3) and the width-study values (Table S4).

## References

[1] Ba, J.L., Kiros, J.R., Hinton, G.E., 2016. Layer normalization. arXiv preprint arXiv:1607.06450 .

[2] Bai, H.Y., Liu, X., 2025. T-graphormer: using transformers for spatiotemporal forecasting. arXiv preprint arXiv:2501.13274 .

[3] Bai, L., Yao, L., Li, C., Wang, X., Wang, C., 2020. Adaptive graph convolutional recurrent network for trafic forecasting. Advances in neural information processing systems 33, 17804–17815.

[4] Bohra, A., Vijay, V., 2026. Wavefuse: Regime-adaptive equity index forecasting via channel-wise wavelet denoising and vertical attention fusion. arXiv preprint arXiv:2609.14733 .

[5] Bohra, A., Vishwakarm, L., 2026. Dstnet: Dynamic spectral trajectory network for causal multi-horizon financial forecasting. arXiv preprint arXiv:2610.09654 .

[6] Bruna, J., Zaremba, W., Szlam, A., LeCun, Y., 2013. Spectral networks and locally connected networks on graphs. arXiv preprint arXiv:1312.6203 .

[7] Challu, C., Olivares, K.G., Oreshkin, B.N., Ramirez, F.G., Canseco, M.M., Dubrawski, A., 2023. Nhits: Neural hierarchical interpolation for time series forecasting, in: Proceedings of the AAAI conference on artificial intelligence, pp. 6989–6997.

[8] Chen, X., Tang, H., Wu, Y., Shen, H., Li, J., et al., 2024. Adpstgcn: Adaptive spatial–temporal graph convolutional network for trafic forecasting. Knowledge-based systems 301, 112295.

[9] Davis, C., Kahan, W.M., 1970. The rotation of eigenvectors by a perturbation. iii. SIAM Journal on Numerical Analysis 7, 1–46.

[10] Deferrard, M., Bresson, X., Vandergheynst, P., 2016. Convolutional neural networks on graphs with fast localized spectral filtering. Advances in neural information processing systems 29.

[11] Dong, J., Wu, H., Zhang, H., Zhang, L., Wang, J., Long, M., 2023. Simmtm: A simple pre-training framework for masked time-series modeling. Advances in Neural Information Processing Systems 36, 29996–30025.

[12] Einizade, A., Malliaros, F., Giraldo, J.H., 2024. Continuous product graph neural networks. Advances in Neural Information Processing Systems 37, 90226–90252.

[13] Fama, E.F., French, K.R., 1997. Industry costs of equity. Journal of Financial Economics 43, 153–193. doi:10.1016/S0304-405X(96) 00896-3.

[14] Fazlyab, M., Robey, A., Hassani, H., Morari, M., Pappas, G., 2019. Eficient and accurate estimation of lipschitz constants for deep neural networks. Advances in neural information processing systems 32.

[15] Févotte, C., Bertin, N., Durrieu, J.L., 2009. Nonnegative matrix factorization with the itakura-saito divergence: With application to music analysis. Neural computation 21, 793–830.

[16] Gama, F., Bruna, J., Ribeiro, A., 2020. Stability properties of graph neural networks. IEEE Transactions on Signal Processing 68, 5680– 5695.

[17] Gao, H., Jiang, R., Dong, Z., Deng, J., Ma, Y., Song, X., 2023. Spatial-temporal-decoupled masked pre-training for spatiotemporal forecasting. arXiv preprint arXiv:2312.00516 .

[18] Gao, Z., Isufi, E., Ribeiro, A., 2021. Stability of graph convolutional neural networks to stochastic perturbations. Signal Processing 188, 108216.

[19] Guo, S., Lin, Y., Feng, N., Song, C., Wan, H., 2019. Attention based spatial-temporal graph convolutional networks for trafic flow forecasting, in: Proceedings of the AAAI conference on artificial intelligence, pp. 922–929.

[20] Hammond, D.K., Vandergheynst, P., Gribonval, R., 2011. Wavelets on graphs via spectral graph theory. Applied and computational harmonic analysis 30, 129–150.

[21] He, M., Wei, Z., Wen, J.R., 2022. Convolutional neural networks on graphs with chebyshev approximation, revisited. Advances in neural information processing systems 35, 7264–7276.

[22] He, M., Wei, Z., Xu, H., et al., 2021. Bernnet: Learning arbitrary graph spectral filters via bernstein approximation. Advances in neural information processing systems 34, 14239–14251.

[23] Hendrycks, D., Gimpel, K., 2016. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415 .

[24] Jiang, J., Han, C., Zhao, W.X., Wang, J., 2023a. Pdformer: Propagation delay-aware dynamic long-range transformer for trafic flow prediction, in: Proceedings of the AAAI conference on artificial intelligence, pp. 4365–4373.

[25] Jiang, R., Wang, Z., Yong, J., Jeph, P., Chen, Q., Kobayashi, Y., Song, X., Fukushima, S., Suzumura, T., 2023b. Spatio-temporal metagraph learning for trafic forecasting, in: Proceedings of the AAAI conference on artificial intelligence, pp. 8078–8086.

[26] Jin, D., Shi, J., Wang, R., Li, Y., Huang, Y., Yang, Y.B., 2023. Traformer: Unify time and space in trafic prediction, in: Proceedings

of the AAAI conference on artificial intelligence, pp. 8114–8122.

[27] Kara, D., Kimura, T., Liu, S., Li, J., Liu, D., Wang, T., Wang, R., Chen, Y., Hu, Y., Abdelzaher, T., 2024. Freqmae: Frequency-aware masked autoencoder for multi-modal iot sensing, in: Proceedings of the ACM Web Conference 2024, pp. 2795–2806.

[28] Kazemi, S.M., Goel, R., Jain, K., Kobyzev, I., Sethi, A., Forsyth, P., Poupart, P., 2020. Representation learning for dynamic graphs: A survey. Journal of Machine Learning Research 21, 1–73.

[29] Kenlay, H., Thanou, D., Dong, X., 2021. Interpretable stability bounds for spectral graph filters, in: International conference on machine learning, PMLR. pp. 5388–5397.

[30] Keriven, N., Bietti, A., Vaiter, S., 2020. Convergence and stability of graph convolutional networks on large random graphs. Advances in Neural Information Processing Systems 33, 21512–21523.

[31] Kim, H., Papamakarios, G., Mnih, A., 2021a. The lipschitz constant of self-attention, in: International Conference on Machine Learning, PMLR. pp. 5562–5571.

[32] Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.H., Choo, J., 2021b. Reversible instance normalization for accurate time-series forecasting against distribution shift, in: International conference on learning representations.

[33] Kipf, T.N., Welling, M., 2016. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907 .

[34] Lan, S., Ma, Y., Huang, W., Wang, W., Yang, H., Li, P., 2022. Dstagnn: Dynamic spatial-temporal aware graph neural network for trafic flow forecasting, in: International conference on machine learning, PMLR. pp. 11906–11917.

[35] Levie, R., Huang, W., Bucci, L., Bronstein, M., Kutyniok, G., 2021. Transferability of spectral graph convolutional neural networks. Journal of Machine Learning Research 22, 1–59.

[36] Li, Y., Yu, R., Shahabi, C., Liu, Y., 2017. Difusion convolutional recurrent neural network: Data-driven trafic forecasting. arXiv preprint arXiv:1707.01926 .

[37] Liu, H., Dong, Z., Jiang, R., Deng, J., Deng, J., Chen, Q., Song, X., 2023a. Spatio-temporal adaptive embedding makes vanilla transformer sota for trafic forecasting, in: Proceedings of the 32nd ACM international conference on information and knowledge management, pp. 4125–4129.

[38] Liu, X., Xia, Y., Liang, Y., Hu, J., Wang, Y., Bai, L., Huang, C., Liu, Z., Hooi, B., Zimmermann, R., 2023b. Largest: A benchmark dataset for large-scale trafic forecasting. Advances in neural information processing systems 36, 75354–75371.

[39] Liu, Y., Wu, H., Wang, J., Long, M., 2022. Non-stationary transformers: Exploring the stationarity in time series forecasting. Advances in neural information processing systems 35, 9881–9893.

[40] Loshchilov, I., Hutter, F., 2019. Decoupled weight decay regularization. URL: https://arxiv.org/abs/1711.05101, arXiv:1711.05101.

[41] Mason, J.C., Handscomb, D.C., 2003. Chebyshev Polynomials. Chapman and Hall/CRC, Boca Raton, FL.

[42] Nie, Y., Nguyen, N.H., Sinthong, P., Kalagnanam, J., 2022. A time series is worth 64 words: Long-term forecasting with transformers. arXiv preprint arXiv:2211.14730 .

[43] Perez, E., Strub, F., De Vries, H., Dumoulin, V., Courville, A., 2018. Film: Visual reasoning with a general conditioning layer, in: Proceedings of the AAAI conference on artificial intelligence.

[44] Ruiz, L., Gama, F., Ribeiro, A., 2021. Graph neural networks: Architectures, stability, and transferability. Proceedings of the IEEE 109, 660–682.

[45] Shao, Z., Zhang, Z., Wang, F., Wei, W., Xu, Y., 2022a. Spatialtemporal identity: A simple yet efective baseline for multivariate time series forecasting, in: Proceedings of the 31st ACM international conference on information & knowledge management, pp. 4454– 4458.

[46] Shao, Z., Zhang, Z., Wang, F., Xu, Y., 2022b. Pre-training enhanced spatial-temporal graph neural network for multivariate time series forecasting, in: Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining, pp. 1567–1577.

[47] Shao, Z., Zhang, Z., Wei, W., Wang, F., Xu, Y., Cao, X., Jensen, C.S., 2022c. Decoupled dynamic spatial-temporal graph neural network for trafic forecasting. arXiv preprint arXiv:2206.09112 .

[48] Tonekaboni, S., Eytan, D., Goldenberg, A., 2021. Unsupervised representation learning for time series with temporal neighborhood coding. arXiv preprint arXiv:2106.00750 .

[49] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I., 2017. Attention is all you need. Advances in neural information processing systems 30.

[50] Virmaux, A., Scaman, K., 2018. Lipschitz regularity of deep neural networks: analysis and eficient estimation. Advances in neural information processing systems 31.

[51] Wang, G., Chen, Y., Gao, M., Wu, Z., Tang, J., Zhao, J., 2024. A time series is worth five experts: Heterogeneous mixture of experts for trafic flow prediction. arXiv preprint arXiv:2409.17440 .

[52] Wang, H., Pan, L., Shen, Y., Chen, Z., Yang, D., Yang, Y., Zhang, S., Liu, X., Li, H., Tao, D., 2025. Fredf: Learning to forecast in the frequency domain, in: International Conference on Learning Representations, pp. 6893–6922.

[53] Wang, X., Zhang, M., 2022. How powerful are spectral graph neural networks, in: International conference on machine learning, PMLR. pp. 23341–23362.

[54] Woo, G., Liu, C., Sahoo, D., Kumar, A., Hoi, S., 2022. Cost: Contrastive learning of disentangled seasonal-trend representations for time series forecasting. arXiv preprint arXiv:2202.01575 .

[55] Wu, Z., Pan, S., Long, G., Jiang, J., Chang, X., Zhang, C., 2020. Connecting the dots: Multivariate time series forecasting with graph neural networks, in: Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 753– 763.

[56] Wu, Z., Pan, S., Long, G., Jiang, J., Zhang, C., 2019. Graph wavenet for deep spatial-temporal graph modeling. arXiv preprint arXiv:1906.00121 .

[57] Yi, K., Zhang, Q., Fan, W., Wang, S., Wang, P., He, H., An, N., Lian, D., Cao, L., Niu, Z., 2023. Frequency-domain mlps are more efective learners in time series forecasting. Advances in neural information processing systems 36, 76656–76679.

[58] You, Y., Chen, T., Sui, Y., Chen, T., Wang, Z., Shen, Y., 2020. Graph contrastive learning with augmentations. Advances in neural information processing systems 33, 5812–5823.

[59] Yu, B., Yin, H., Zhu, Z., 2017. Spatio-temporal graph convolutional networks: A deep learning framework for trafic forecasting. arXiv preprint arXiv:1709.04875 .

[60] Yu, Y., Wang, T., Samworth, R.J., 2015. A useful variant of the davis– kahan theorem for statisticians. Biometrika 102, 315–323.

[61] Yue, Z., Wang, Y., Duan, J., Yang, T., Huang, C., Tong, Y., Xu, B., 2022. Ts2vec: Towards universal representation of time series, in: Proceedings of the AAAI conference on artificial intelligence, pp. 8980–8987.

[62] Zhang, X., Zhao, Z., Tsiligkaridis, T., Zitnik, M., 2022a. Selfsupervised contrastive pre-training for time series via time-frequency consistency. Advances in neural information processing systems 35, 3988–4003.

[63] Zhang, Z., Wang, X., Zhang, Z., Li, H., Qin, Z., Zhu, W., 2022b. Dynamic graph neural networks under spatio-temporal distribution shift. Advances in neural information processing systems 35, 6074– 6089.

[64] Zheng, C., Fan, X., Wang, C., Qi, J., 2020. Gman: A graph multiattention network for trafic prediction, in: Proceedings of the AAAI conference on artificial intelligence, pp. 1234–1241.

[65] Zhou, T., Ma, Z., Wen, Q., Wang, X., Sun, L., Jin, R., 2022. Fedformer: Frequency enhanced decomposed transformer for long-term series forecasting, in: International conference on machine learning, PMLR. pp. 27268–27286.

[66] Zhu, Y., Xu, Y., Yu, F., Liu, Q., Wu, S., Wang, L., 2020. Deep graph contrastive representation learning. arXiv preprint arXiv:2006.04131

## A. Mathematical Proofs

## Proof of Lemma 1

Write ${ \mathcal { D } } = \mathrm { d i a g } ( { \hat { A } } \mathbf { 1 } )$ and $D ^ { \prime } = \mathrm { d i a g } ( \hat { A } ^ { \prime } { \bf 1 } )$ , so that $P _ { f } =$ $\hat { A } D ^ { - 1 }$ and $\| D ^ { - 1 } \| _ { 2 } , \| D ^ { \prime - 1 } \| _ { 2 } \leq 1 / m$ . Hence $\| P _ { f } \| _ { 2 } , \| P _ { b } \| _ { 2 } \leq$ $p : = a / m$ , and the same holds under $\ll ^ { \prime }$ . Since $\Delta \hat { A } \ =$ $- { \scriptstyle { \frac { 1 } { 2 } } } \Delta L .$

$$
P _ { f } ^ { \prime } - P _ { f } = \Delta \hat { A } \mathcal { D } ^ { \prime - 1 } + \hat { A } ( \mathcal { D } ^ { \prime - 1 } - \mathcal { D } ^ { - 1 } ) .
$$

The first term has Frobenius norm at most $\Vert \Delta L \Vert _ { \mathrm { F } } / ( 2 m )$ For the second, $| 1 / d _ { i } ^ { \prime } - 1 / d _ { i } | \ \leq \ | d _ { i } ^ { \prime } - d _ { i } | / m ^ { 2 }$ , and the vector of row-sum changes is $\Delta \hat { A } \mathbf { 1 } = - \textstyle \frac { 1 } { 2 } \Delta L \mathbf { 1 }$ . Row � of $\Delta L$ has at most $s _ { \Delta }$ nonzero entries, so by Cauchy–Schwarz $\| \Delta L \mathbf { 1 } \| _ { 2 } \leq \sqrt { s _ { \Delta } } \| \Delta L \| _ { \mathrm { F } }$ . The second term is therefore at most $a \sqrt { s _ { \Delta } } \| \Delta L \| _ { \mathrm { F } } / ( 2 m ^ { 2 } )$ , and

$$
\| P _ { f } ^ { \prime } - P _ { f } \| _ { \mathrm { F } } \leq \delta : = \frac { 1 } { 2 m } \Big ( 1 + \frac { a \sqrt { s _ { \Delta } } } { m } \Big ) \| \Delta L \| _ { \mathrm { F } } .
$$

The same bound holds for $P _ { b } ^ { \prime } - P _ { b } = ( P _ { f } ^ { \prime } - P _ { f } ) ^ { \top }$ . Let $\boldsymbol { Q } = \boldsymbol { P } _ { f } + \boldsymbol { P } _ { b } ,$ , so that $\| Q \| _ { 2 } \leq 2 p$ and $\Vert Q ^ { \prime } - \dot { Q } \Vert _ { \mathrm { F } } \leq 2 \delta$ . Step $s ^ { \prime }$ applies $P _ { f } Q ^ { s ^ { \prime } - 1 }$ or $P _ { b } Q ^ { s ^ { \prime } - 1 }$ to �. Changing one factor of the product at a time,

$$
\begin{array} { r l } & { \| P _ { f } ^ { \prime } Q ^ { \prime s ^ { \prime } - 1 } - P _ { f } Q ^ { s ^ { \prime } - 1 } \| _ { \mathrm { F } } } \\ & { \quad \leq \delta ( 2 p ) ^ { s ^ { \prime } - 1 } + ( s ^ { \prime } - 1 ) p ( 2 p ) ^ { s ^ { \prime } - 2 } 2 \delta = s ^ { \prime } ( 2 p ) ^ { s ^ { \prime } - 1 } \delta , } \end{array}
$$

and the same holds for $P _ { b }$ . Multiplying by $\| Z \| _ { 2 } \| W \| _ { 2 }$ and summing over steps and both directions gives (15). □

## Proof of Proposition 2

Fix the input and let $e _ { 1 } , \ldots , e _ { 4 }$ be the Frobenius distances between outputs under $L ^ { \prime }$ and � after DCSE , TempAttn , $\mathrm { D C S E } _ { 2 }$ and TempAttn . Both runs feed DCSE the same input, so $e _ { 1 } \le \kappa _ { L } \vert \vert \Delta L \vert \vert _ { \mathrm { F } }$ . The temporal block does not depend on the operator, so $e _ { 2 } \leq \kappa _ { T } e _ { 1 }$ . Then $e _ { 3 } \leq \kappa _ { Z } e _ { 2 } + \kappa _ { L } \| \Delta L \| _ { \mathrm { F } }$ and $\begin{array} { l l l } { e _ { 4 } } & { \leq } & { \kappa _ { T } e _ { 3 } } \end{array}$ . Substituting gives $\begin{array} { r } { e _ { 4 } \leq \kappa _ { T } \kappa _ { L } ( 1 + } \end{array}$ $\kappa _ { Z } \kappa _ { T } ) \| \Delta L \| _ { \mathrm { F } }$ □

## Proof of Proposition 3

With $P _ { X } / P _ { Z } ~ = ~ e ^ { - u }$ , each term of $D _ { \mathrm { I S } }$ equals $e ^ { - u } + $ $u - 1 = { \textstyle \frac { 1 } { 7 } } u ^ { 2 } - { \textstyle \frac { 1 } { 6 } } u ^ { 3 } + { \cal O } ( u ^ { 4 } )$ . Summing over bins, $\textstyle \sum u ^ { 2 } =$ $\begin{array} { r } { 2 D _ { \mathrm { I S } } + \frac { 1 } { 3 } \sum u ^ { 3 } + O ( \sum u ^ { 4 } ) } \end{array}$ . On unclamped bins $\begin{array} { r l } { \mathcal { L } _ { \mathrm { f r e q } } } & { { } = } \end{array}$ $\sum u ^ { 2 } / ( N W ^ { \prime } s _ { X } ^ { 2 } )$ , which gives the result. □

## B. Implementation Details

## Preprocessing

Readings are standardised per node with training-set statistics. The static adjacency is $A _ { i j } = \mathrm { e x p } ( - \mathrm { d i s t } _ { i j } ^ { 2 } / 2 \sigma ^ { 2 } )$ when this value is at least 0.1 and zero otherwise, with � the standard deviation of the finite pairwise road distances. Self-loops are removed and the matrix is symmetrised. Time of day uses 288 five-minute bins and day of week 7 bins, each mapped to a 24-dimensional learned embedding. Supplementary Table S1 lists the full configuration and Supplementary Algorithm S1 the training procedure.

## Fama–French Transfer

Monthly value-weighted returns of the 43 industry portfolios with complete histories are each divided by a trailing standard deviation over the previous 60 months, with at least 20 months and a one-month lag, so month � is scaled by months $t - 6 0$ to � − 1. The scaled returns are winsorised at $\pm 5$ and standardised cross-sectionally each month. The first 20 months have no volatility estimate and are dropped. Windows hold 12 months of input and 12 months of targets. The graph links portfolios in the same one of nine broad sector groups (218 edges) and is fixed during finetuning. Section 7.5 lists the trained and frozen modules. Data before 2010 are used for fine-tuning (717 training and 162 validation windows), and months from 2010 onward for evaluation. The long-short portfolio buys the top decile of predicted returns and sells the bottom decile each month. The random control uses the same architecture, trainable modules, loss, learning rates and early stopping, with every module initialised at random instead of from trafic.