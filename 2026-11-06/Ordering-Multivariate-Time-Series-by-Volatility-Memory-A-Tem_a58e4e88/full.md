Ordering Multivariate Time Series by Volatility Memory: A Temporal Convolutional Approach to Volatility Heredity

# Ordering Multivariate Time Series by Volatility Memory: A Temporal Convolutional Approach to Volatility Heredity

Kaoutar El Azaar<sup>1,2,3</sup>, Mohamed Hamlich<sup>1</sup>, Moncef Garouani<sup>2</sup>, Franck Ravat<sup>2</sup>, Asmae Chakir<sup>3</sup> and Mohamed Tabaa<sup>3</sup>

<sup>1</sup> Casablanca (LCCPS Laboratory), Hassan II University <sup>2</sup> IRIT (UMR 5505 CNRS), Université Toulouse Capitole, Toulouse, France <sup>3</sup> LPRI Laboratory, EMSI, Casablanca, Morocco

## Abstract

Volatility dynamics can carry information about temporal relationships that is not apparent in the conditional mean. Extracting this information is dificult because the persistence of each variable can obscure its dependence on other variables. This paper presents QS-TCN, a method for ordering multivariate time series that combines statistical estimates of volatility memory with temporal convolutional learning. Persistence estimates obtained by quasi-maximum likelihood and temporal modes extracted by the Matrix Pencil method define structured filters within the network. Together with learned filters, they represent the history of the target and support comparisons of directional evidence between variables. These comparisons are combined into a global order of precedence, which does not identify a causal graph. The method is evaluated on synthetic systems with linear and nonlinear volatility interactions and on the Finance and fMRI benchmarks, whose reference relations are known. Across the four synthetic volatility regimes studied, QS-TCN reaches a higher mean ordering quality than an external lead–lag ordering and its squared-input adaptation, although the gain over the adaptation is uncertain in two regimes. Applications to currency, gold and cryptocurrency markets are reported descriptively.

Keywords: Stochastic volatility, Temporal ordering, Temporal convolutional networks, Functional magnetic resonance imaging (fMRI), Financial time series

## 1 Introduction

Understanding the directional relationships between the variables of a multivariate time series is important in finance and neuroscience. In financial markets, the fluctuations of one asset can precede those of another. In neuroscience, temporal relationships between recorded signals help characterize interactions between brain regions. In both cases, a useful goal is to organize the variables in a global order that summarizes the directional evidence present in the data.

Several approaches address related questions from diferent perspectives. Granger [1] defines direction through the additional predictive information provided by a variable’s past. Graphical methods rely on conditional independence relations to study dependencies among several variables, including nonlinear relationships in time series [2–4]. Other approaches estimate structural equation models through continuous optimization [5] or exploit assumptions on the noise distribution to identify causal structure. LiNGAM, for example, relies on non-Gaussian disturbances [6]. The method has been extended to time series [7], with further developments addressing functional MRI [8] and functional data [9]. These approaches ofer diferent ways of extracting directional information from observational data.

Volatility provides another source of temporal information. Econometric research has developed causality-in-variance tests [10–13], models of interdependent conditional variances [14], and measures of volatility transmission between markets [15]. In causal discovery, location-scale noise models provide conditions under which causal direction is identifiable from heteroscedastic observational data [16]. More recent work distinguishes causal efects on the mean from those on the variance by learning separate causal graphs from heteroscedastic observational data [17]. Building on this literature, the present study examines how estimates of volatility memory can contribute to learning an order among variables.

The persistence of fluctuations alone does not establish a direction. A variable can display several temporal scales through its own dynamics, while its dependencies on other variables can add further structure to the observed series. Results on Prony systems and related super-resolution problems show how closely spaced components can make parameter recovery sensitive to measurement errors [18, 19]. These dificulties motivate an approach that first represents the target variable’s history, then examines the relationship between the source’s past and the remaining fluctuations. The target’s history provides a predictive representation, without being interpreted as an isolated structural mechanism.

This article proposes QS-TCN, a two-channel method that connects statistical estimates of volatility memory to learning with temporal convolutions. The first channel estimates persistence by quasi-maximum likelihood under a working stochastic-volatility model [20, 21] and extracts temporal modes using the Matrix Pencil method [22, 23]. In the second channel, the persistence estimates and accepted poles define structured temporal filters, which are combined with learned filters. The network represents the target’s past and evaluates directional evidence associated with its residual dynamics. A shared scoring function evaluates the two orientations of each pair, and the resulting contrast is exactly antisymmetric. Synchronized block-bootstrap samples allow the stability of the scores to be assessed, and a decoder combines them into a global order. This order summarizes directional evidence and does not identify direct causal links.

The evaluation covers controlled synthetic systems with diferent volatility mechanisms and the Finance and fMRI datasets, for which reference directional constraints are available [24–26]. In the synthetic experiments, QS-TCN is compared with an ordering constructed from the lead–lag scores of Bennett et al. [27]. On the Finance and fMRI datasets, it is compared with the contemporaneous order returned by VAR-LiNGAM [7]. These comparisons concern agreement between the recovered orders and the reference directional constraints. The methods are not treated as equivalent procedures for graph reconstruction. Applications to currency, gold and cryptocurrency markets extend the study to observed financial data.

The evaluation reports ordering quality for each system and for the datasets as a whole, together with its sensitivity to resampling and the agreement expected from a random order. Simple marginal properties can also align with reference directional constraints [28, 29]. The evaluation of the full model is therefore distinguished from the attribution of its performance to individual components.

This work makes three contributions. The first is a framework for ordering the variables of multivariate time series using temporal information in volatility dynamics. The second connects QML persistence estimates and Matrix Pencil modes to learned temporal filters, with a shared directional scoring function and a global ordering procedure. The third is an evaluation across controlled volatility mechanisms, the Finance and fMRI datasets, and observed financial markets, including external comparisons and a bootstrap sensitivity analysis.

Section 2 presents the main technical dificulties and evaluation issues. Section 3 describes QS-TCN, Section 4 reports the experiments and results, and Section E concludes.

## 2 Background

Volatility-based ordering examines temporal information in the amplitude of fluctuations. Interpreting this information requires distinguishing conditional variance from the conditional second moment. For a variable $X _ { t }$ with finite second moment and an available history $\mathcal { H } _ { t - 1 }$

$$
\operatorname { \mathbb { E } } [ X _ { t } ^ { 2 } \mid { \mathcal { H } } _ { t - 1 } ] = \operatorname { V a r } ( X _ { t } \mid { \mathcal { H } } _ { t - 1 } ) + { \Big ( \operatorname { \mathbb { E } } [ X _ { t } \mid { \mathcal { H } } _ { t - 1 } ] \Big ) } ^ { 2 } .\tag{1}
$$

Dependencies between squared observations can therefore reflect both mean and variance dynamics. This distinction matters for directional analysis, since neglecting dependence in the conditional mean can distort tests for causality in variance [30].

A second dificulty concerns the temporal structure generated by volatility persistence. In the stationary stochastic-volatility reference model of Section 3.1, Gaussian logvolatility follows a first-order autoregression with a single persistence parameter $\phi .$ Nevertheless, the positive-lag autocovariance of the squared observations admits an expansion in geometric components with rates $\phi , \phi ^ { 2 } , \phi ^ { 3 } , . . . ,$ as derived in Appendix B. Multiple temporal modes can thus arise within a univariate model, without cross-variable coupling. An additional estimated mode does not, by itself, identify an external influence.

Over a finite lag range, this memory profile can be approximated by a short sum of exponentials whose parameters are estimated using the Matrix Pencil method [22, 23]. Separating nearby components remains sensitive to estimation error, a dificulty studied in Prony systems and related super-resolution problems [18, 19]. Persistence estimates and accepted poles therefore provide summaries of marginal temporal structure. In QS TCN, these summaries define structured filters that connect Channel I to the temporal representations learned in Channel II, without attributing individual poles to specific causal mechanisms.

Pairwise directional scores do not necessarily define a consistent global order. Antisymmetry ensures that reversing a pair reverses its score, but does not ensure transitivity. For example, $S _ { i j } > 0 , S _ { j k } > 0$ and $S _ { k i } > 0$ form a directed cycle. Ranking methods must reconcile such inconsistencies when aggregating pairwise comparisons [31]. Global decoding therefore seeks an order that reflects the combined evidence, which can require placing some pairs against their individually preferred orientation.

Evaluation must also distinguish agreement with a reference structure from the information used to obtain that agreement. Marginal properties, including variance, can align with reference structures and account for strong performance on some synthetic benchmarks [28]. Related sortability efects occur in time-series settings [29]. These findings motivate comparisons that assess whether pairwise temporal representations contribute information beyond node-specific summaries.

The fMRI setting introduces an additional observation layer. Neural activity is measured indirectly through the haemodynamic response, with a temporal resolution that can be coarser than the neural interactions of interest. Results for Granger-type analysis identify conditions under which haemodynamic filtering preserves directional relationships, while also examining the efects of downsampling and measurement noise [32, 33]. The NetSim benchmark provides simulated BOLD data for evaluating network methods under dif ferent observation conditions [26]. It therefore ofers a controlled setting for examining how informative volatility-based temporal representations remain after haemodynamic transformation and limited temporal sampling.

## 3 Methodology

Given a multivariate time series $\mathbf { X } _ { t } ~ \in ~ \mathbb { R } ^ { d }$ , QS-TCN estimates a permutation $\widehat { \pi } =$ $\big ( \widehat { \pi } _ { 1 } , \dots , \widehat { \pi } _ { d } \big )$ , in which $\widehat { \pi } _ { a }$ is the variable placed in position $^ { a , }$ bdescribing directional preceb b bdence among its variables. It combines two channels: a statistical description of marginal

volatility memory and a temporal network that converts this information into pairwise scores. These scores are subsequently aggregated into a global order. The output specifies precedence, not the existence of a direct causal edge.

## 3.1 Channel I: QML persistence and spectral memory

For each variable, we use the stationary stochastic-volatility reference

$$
X _ { i , t } = \exp ( h _ { i , t } / 2 ) \varepsilon _ { i , t } , \qquad h _ { i , t } = \mu _ { i } + \phi _ { i } ( h _ { i , t - 1 } - \mu _ { i } ) + \eta _ { i , t } ,\tag{2}
$$

where $| \phi _ { i } | < 1$ and the innovation sequences are independent and Gaussian. Following [20, 21], the persistence $\phi _ { i }$ is estimated by QML: squared observations are logtransformed, and a Kalman filter evaluates a Gaussian approximation to their likelihood. The estimate characterizes marginal persistence rather than an identified structural noise process.

To represent additional temporal scales, we form the centered-square signal $q _ { i , t } = X _ { i , t } ^ { 2 } -$ $\overline { { { X } _ { i } ^ { 2 } } }$ and approximate its positive-lag autocovariance by

$$
\widehat { \mathcal { C } } _ { i } ( k ) \approx \sum _ { r = 1 } ^ { R _ { i } } A _ { i r } \lambda _ { i r } ^ { k - 1 } , \qquad k = 1 , \dots , K _ { \mathrm { s p e c } } .\tag{3}
$$

The Matrix Pencil method of [23] extracts the poles from a reduced singular subspace of the profile’s Hankel matrix. A block-bootstrap threshold determines the supported rank, and amplitudes are fitted by least squares. Channel II receives the numerical persistence $\widetilde { \phi } _ { i }$ , the accepted real poles, and a descriptor $\Theta _ { i }$ summarizing estimation ediagnostics. Unsupported modes are masked. The decomposition concerns the total squared-memory profile; individual poles are not assigned a causal origin.

## 3.2 Channel II: memory–heredity temporal network

Channel II uses shared causal temporal convolutions to represent lagged dependencies, drawing on their application to multivariate time-series forecasting [34]. The persistence estimates $\widetilde { \phi } _ { i }$ and accepted poles $\widehat { \Lambda } _ { i }$ from Channel I enter the TCN through structured e bconvolution kernels, which are combined with learned temporal filters. These estimated rates remain fixed during neural training.

Let $\mathbf { q } _ { j , t } ^ { ( L ) } = ( q _ { j , t - 1 } , \ldots , q _ { j , t - L } ) ^ { \top }$ denote a lagged input window.

The Memory Cell represents the target’s own history through free filters and geometric filters proportional to $\widetilde { \phi } _ { j } ^ { \ell - 1 }$ and $\widehat { \lambda } _ { j r } ^ { \ell - 1 }$ , using the target’s persistence and accepted poles. eA nonlinear readout produces

$$
\widehat { q } _ { j , t } = \operatorname { M e m } _ { \theta _ { m } } \Big ( \mathbf { q } _ { j , t } ^ { ( L _ { m } ) } ; \widetilde { \phi } _ { j } , \widehat { \Lambda } _ { j } , \Theta _ { j } \Big ) , \qquad r _ { j , t } = q _ { j , t } - \widehat { q } _ { j , t } .\tag{4}
$$

The residual captures fluctuations left unexplained by this fitted memory representation.

For the direction $i  j .$ , the Heredity Cell filters the source history using the source persistence $\widetilde { \phi } _ { i }$ and the target poles $\widehat { \lambda } _ { j r }$ . Its structured kernels are

$$
K _ { i j r } ( \ell ) = \sum _ { s = 0 } ^ { \ell - 1 } \widetilde { \phi } _ { i } ^ { \ell - 1 - s } \widehat { \lambda } _ { j r } ^ { s } , \qquad \ell = 1 , \dots , L _ { h } .\tag{5}
$$

These kernels are normalized and combined with free source filters through learned weights. The representation retains the temporal means of both branches and their mean products with $r _ { j , t } ,$ , together with $\Theta _ { i }$ and $\Theta _ { j }$ . The same scoring network evaluates this representation in both orientations:

$$
D _ { i j } = f _ { \theta } ( i , j ) - f _ { \theta } ( j , i ) , \qquad S _ { i j } = S _ { 0 , i j } + g _ { i j } D _ { i j } , \quad g _ { i j } = g _ { j i } .\tag{6}
$$

Here $S _ { 0 , i j }$ is an analytical anchor comparing lagged covariances between the source’s squared signal and autoregressive residuals of the target. A symmetric learned gate $g _ { i j }$ controls the correction, giving $S _ { j i } = - S _ { i j }$ exactly. The kernel itself is symmetric in its rates; directional evidence arises from the source–target assignment and the residualbased comparison.

## 3.3 Learning and global ordering

The network is trained on synthetic systems with acyclic coupling in latent log-volatility. The Memory Cell first minimizes squared prediction error and is then frozen. The directional module minimizes a logistic precedence loss, with a penalty on deviations from the anchor. Its checkpoint is selected on independent synthetic validation systems; benchmark graph labels are excluded from optimization and checkpoint selection.

Benchmark inference uses a fixed chronological prefix, also used for centering, scaling, and Channel-I estimation. The market applications use the full series or windows specified below. Neural parameters remain fixed. Synchronized block-bootstrap views preserve alignment across variables, and the anchor is refitted within each view. Pairwise scores are aggregated by their median, weighted by sign stability and penalized for dispersion, yielding an antisymmetric matrix C. These weights describe resampling stability rather than causal probabilities.

The decoder seeks an order with high agreement with this matrix:

$$
J ( \pi ; C ) = \sum _ { 1 \le a < b \le d } C _ { \pi _ { a } \pi _ { b } } .\tag{7}
$$

Starting from decreasing row sums, it accepts improving single-node insertions until no improvement is found or the limit of 25 moves is reached. The returned order is therefore a heuristic estimate. Appendix D specifies the structured filters, statistical score $S _ { 0 }$ , bootstrap aggregation and training settings used in the controlled synthetic evaluation.

## 4 Results

This section asks how well QS-TCN orders variables when it is applied exactly as frozen. Two questions guide the analysis. How closely does the returned order agree with the structure known for each dataset, and how stable is the resulting OQ under resampling? The analysis starts with two simulated benchmarks, Finance and fMRI, whose reference structure allows a direct measurement of agreement. A controlled synthetic evaluation follows (Section 4.4). The last part turns to real markets, where no complete reference order exists. It covers a panel of six currency pairs and gold, then several cryptocurrency systems whose recovered orders are compared with relations published by other authors.

The Finance, fMRI, Forex/Gold and cryptocurrency experiments all use the frozen implementation without modification. The controlled synthetic evaluation difers, since it uses frozen checkpoints of a reconstructed implementation (Section 4.4). On Finance and fMRI, inference is repeated under ten bootstrap seed schemes labelled A to J, and scheme A is the historical frozen run. All ten recorded schemes are reported, and the neural parameters stay fixed during evaluation. One point needs attention from the start. The variation across seed schemes measures the sensitivity of the frozen inference to the bootstrap construction. This variation is neither an estimate of training uncertainty nor an estimate of sampling uncertainty.

## 4.1 Datasets and evaluation setting

Table 1 lists the four benchmark and market settings. The controlled synthetic systems are described separately in Section 4.4.

The Finance data are the preprocessed financial benchmark distributed with TCDF [24], which comes from the simulated financial systems of Kleinberg [25]. The fMRI data are the preprocessed NetSim benchmark, also distributed with TCDF [24] and first introduced by Smith et al. [26]. In NetSim, the directed neural activity is not observed directly. This activity passes through a haemodynamic transformation, so the analysed BOLD signals combine temporal smoothing, nonlinear observation efects and coloured noise [35].

The original NetSim benchmark contains 28 systems. Dataset 21 is missing from the retained frozen evaluation output, and the retained files do not explain why. Every fMRI aggregate below therefore covers the 27 available systems. A later byte-level check showed that the observations and the reference-edge file of dataset 21 are identical to those of dataset 1. The original reason for the omission was not recorded.

The two other settings use real financial data. The Forex/Gold panel has seven assets and shows what QS-TCN returns when no complete reference graph exists. The cryptocurrency experiments serve a diferent purpose. Their recovered orders are compared with directional structures published by other authors: the six-cryptocurrency connectedness analysis of Ji et al. [36] and the regime-specific spillover analysis of Bitcoin, Ethereum and Litecoin by Yousaf and Ali [38]. These published structures serve as points of comparison and are not treated as causal ground truth.

Table 1: Datasets used in the empirical evaluation. A known reference structure allows direct computation of ordering quality. Published market relations are used only for cross-study comparison and are not interpreted as complete causal ground truth.
<table><tr><td>Data setting</td><td>Domain</td><td>Systems / assets</td><td>Length</td><td>Reference structure</td></tr><tr><td>Finance [24, 25]</td><td>Simulated financial systems</td><td>9 datasets, 25 variables</td><td>T = 4000</td><td>Known</td></tr><tr><td>fMRI [24, 26]</td><td>Simulated BOLD systems</td><td>27 evaluated datasets, 5–50 nodes</td><td>T = 50–5000</td><td>Known</td></tr><tr><td>Forex/Gold</td><td>Observed financial markets</td><td>7 assets</td><td>T = 2384</td><td>No complete GT</td></tr><tr><td>Cryptocurrencies [36, 38]</td><td>Observed crypto markets</td><td>3 or 6 assets</td><td>Study-dependent</td><td>Published directional relations</td></tr></table>

The benchmark analyses have a clear scope. They measure how well the recovered order agrees with the reference constraints and how stable the OQ is under the bootstrap. They do not show that the recovered signal comes only from volatility. A descriptive VAR-LiNGAM comparison is therefore included below, whereas marginal-ordering baselines and ablations on the benchmark data are absent. The synthetic evaluation compares QS-TCN with an external lead-lag ordering method (Section 4.4).

Ordering quality. A benchmark states, for some pairs of variables, which one must come first. The score is the share of these requirements which the returned order respects. The precise definition follows. Fix a dataset m and call its variables nodes. Let $A _ { m }$ be the set of directed reference constraints of dataset m. A pair $( i , j ) \in \mathcal { A } _ { m }$ , with $i \neq j$ means that the reference structure requires node i to precede node j for the purpose of the score. This definition does not require a single reference permutation. The recovered order is a permutation $\widehat { \pi } _ { m }$ of the nodes, and $p _ { \widehat { \pi } _ { m } } ( i )$ denotes the position of node i in this b bpermutation. For a nonempty reference set, the ordering quality of dataset m is

$$
\mathrm { O Q } _ { m } = \frac { 1 } { | \mathcal { A } _ { m } | } \sum _ { ( i , j ) \in \mathcal { A } _ { m } } \mathbb { I } \left[ p _ { \widehat { \pi } _ { m } } ( i ) < p _ { \widehat { \pi } _ { m } } ( j ) \right] .\tag{8}
$$

In this formula, $| { \mathcal { A } } _ { m } |$ is the number of scored constraints, and the indicator $\mathbb { I } [ \cdot ]$ equals 1 when the statement inside the brackets is true and 0 otherwise. Thus $\mathrm { O Q } _ { m }$ lies between 0 and 1. The value 1 means that every constraint is respected, and the value 0 means that none is.

Each benchmark supplies $A _ { m }$ as the list of directed edges of its ground-truth graph. Self-loops are removed, since a node does not precede itself, and no transitive closure is applied, so only direct edges are scored. Under this construction, the benchmarks contain 197 constraints over the nine Finance datasets and 239 over the 27 evaluated fMRI datasets, and these are the totals used below. Some reference graphs contain a bidirectional pair, meaning that both $( i , j )$ and $( j , i )$ belong to $A _ { m }$ , or a longer directed cycle. No linear order respects both constraints of such a pair, or all constraints of a directed cycle, so $\mathrm { O Q } _ { m }$ stays below 1 for every method. Enumerating the orders inside each strongly connected component and arranging the components in a topological order gives the exact maximum attainable value. Rounded to three decimal places, this maximum is 0.625 for fMRI dataset 13, 0.800 for fMRI dataset 14, 0.947 for Finance 20\_1D, 0.950 for 20\_1\_3 and for 40\_1\_3, and 0.974 for 40\_1. All other datasets have an acyclic reference graph, and their maximum is 1. The value 0.625 reported for dataset 13 under every seed scheme therefore equals its maximum.

A small example, which is not data, makes the computation concrete. Take three nodes and the constraints (1, 2) and (2, 3). The order (2, 1, 3) respects (2, 3) and violates (1, 2), so $\mathrm { O Q } = 1 / 2$

What a random order obtains. A random order gives the reference point for these scores. Suppose the order is drawn uniformly among all permutations of the nodes. For a fixed pair $i \neq j ,$ , exchanging the roles of i and $j$ maps the permutations with $p ( i ) < p ( j )$ one-to-one onto those with $p ( j ) < p ( i )$ , so each event has probability $1 / 2$ . By linearity of expectation, $\mathbb { E } [ \mathrm { O Q } _ { m } ] = 1 / 2$ for every nonempty set $A _ { m } .$ whatever the dependence between constraints. Dependence matters for the distribution of $\mathrm { O Q } _ { m }$ and not for its mean. In the example above, the six permutations give the values 1, $1 / 2 , 1 / 2 , 1 / 2$ $1 / 2$ and 0. Both constraints are respected with probability $1 / 6$ , whereas independence would give $1 / 4$ . Dependence also works in the other direction. If node $i$ must precede both $j$ and $k ,$ , both constraints are respected exactly when i comes first among the three nodes, which has probability $1 / 3$ . For this reason, a binomial calculation which treats all scored pairs as independent is not a valid significance test, and none is used here.

Two ways of summarizing a benchmark. Finance and fMRI are evaluated under $S ~ = ~ 1 0$ seed schemes. Write $\mathrm { O Q } _ { m } ^ { ( s ) }$ for the value of dataset m under scheme $s ,$ and $\begin{array} { r } { \overline { { \mathrm { O Q } } } _ { m } = S ^ { - 1 } \sum _ { s } \mathrm { O Q } _ { m } ^ { ( s ) } } \end{array}$ for its average over schemes, which is the “Mean $\mathrm { O Q } ^ { \because }$ column of the tables. With M datasets, the macro-average gives the same weight to every dataset,

$$
\mathrm { O Q } _ { \mathrm { m a c r o } } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \overline { { \mathrm { O Q } } } _ { m } ,\tag{9}
$$

and the pair-weighted score gives the same weight to every scored constraint, so datasets with more constraints count more,

$$
\mathrm { O Q } _ { \mathrm { w e i g h t e d } } = \frac { \sum _ { m = 1 } ^ { M } | \mathcal { A } _ { m } | \overline { { \mathrm { O Q } } } _ { m } } { \sum _ { m = 1 } ^ { M } | \mathcal { A } _ { m } | } .\tag{10}
$$

Two standard deviations appear below, and they answer diferent questions. The first is computed across schemes. If ${ \mathrm { O Q } } _ { \mathrm { m a c r o } } ^ { ( s ) }$ is the macro-average of scheme s, then

$$
\mathrm { S D } _ { \mathrm { s e e d } } = \left[ \frac { 1 } { S - 1 } \sum _ { s = 1 } ^ { S } \left( \mathrm { O Q } _ { \mathrm { m a c r o } } ^ { ( s ) } - \mathrm { O Q } _ { \mathrm { m a c r o } } \right) ^ { 2 } \right] ^ { 1 / 2 } .\tag{11}
$$

This quantity describes how much the benchmark average moves when the bootstrap seed scheme changes. The second is computed across datasets,

$$
\mathrm { S D } _ { \mathrm { d a t a s e t s } } = \left[ \frac { 1 } { M - 1 } \sum _ { m = 1 } ^ { M } \left( \overline { { \mathrm { O Q } } } _ { m } - \mathrm { O Q } _ { \mathrm { m a c r o } } \right) ^ { 2 } \right] ^ { 1 / 2 } ,\tag{12}
$$

and describes how diferent the datasets are from one another. Every standard deviation in this section uses the denominator $n - 1$ . Neither quantity is a confidence interval for performance on new systems.

For the real markets, no benchmark OQ is reported, since no complete reference structure exists. The recovered order is reported instead. When a published directional structure maps onto the same assets, concordance is measured separately from benchmark ordering quality.

VAR-LiNGAM comparison. VAR-LiNGAM serves as a competing temporalordering method. The upstream implementation is fitted to the same chronological estimation prefix used by QS-TCN, with $T _ { 0 } = \mathrm { m i n } \{ T , \mathrm { m a x } \{ 4 0 , \lfloor 0 . 6 5 T \rfloor \} \}$ , one VAR lag, and pruning enabled [7]. With the maximum lag fixed at one, the BIC setting has no alternative lag order to select. No reference graph is supplied during fitting. The resulting contemporaneous order is scored against the same direct reference constraints used for QS-TCN, after omitting their delay labels. The comparison is therefore descriptive and is not a like-for-like comparison of lag-specific graph recovery. Tables 2 and 3 give the per-system values, and the full seed-wise QS-TCN scores appear in Appendix A.

## 4.2 Finance benchmark

The Finance benchmark distributed with TCDF [24] contains nine evaluated systems derived from the financial simulations of Kleinberg [25]. Averaging the nine dataset-level values over the ten seed schemes gives

$$
\mathrm { O Q } _ { \mathrm { F i n a n c e , m a c r o } } = 0 . 7 1 5 5 .\tag{13}
$$

Across the ten schemes, the macro-average has a standard deviation $\mathrm { S D } _ { \mathrm { s e e d } } = 0 . 0 1 5 6$ and the scheme-level macro-averages range from

$$
0 . 6 9 6 1 \leq \mathrm { O Q } _ { \mathrm { F i n a n c e , m a c r o } } ^ { ( s ) } \leq 0 . 7 4 6 5 .\tag{14}
$$

The value 0.0156 describes only the sensitivity of the benchmark mean to the seed scheme. It is not the variation between Finance systems, which is much larger. From the dataset means of Table 2, the calculation gives $\mathrm { S D _ { d a t a s e t s } \approx 0 . 1 3 8 }$ . Giving the same weight to every scored reference pair instead of every dataset yields

$$
\mathrm { O Q } _ { \mathrm { F i n a n c e , w e i g h t e d } } = 0 . 6 9 1 4\tag{15}
$$

Table 2: Finance ordering quality by dataset. QS-TCN A is the historical frozen order; VAR-LiNGAM is one deterministic order fitted on the same estimation prefix. Mean, standard deviation and range for QS-TCN are computed across the ten bootstrap seed schemes A–J.
<table><tr><td>Dataset</td><td>Pairs</td><td>QS-TCN A</td><td>VAR-LiNGAM</td><td>Mean OQ</td><td>SD</td><td>Min-Max</td></tr><tr><td>manyinputs</td><td>6</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.000</td><td>1.000-1.000</td></tr><tr><td>20_1A</td><td>18</td><td>0.778</td><td>0.500</td><td>0.806</td><td>0.054</td><td>0.722-0.889</td></tr><tr><td>20_1B</td><td>20</td><td>0.650</td><td>0.650</td><td>0.590</td><td>0.052</td><td>0.500-0.650</td></tr><tr><td>20_1C</td><td>19</td><td>0.526</td><td>0.421</td><td>0.532</td><td>0.017</td><td>0.526–0.579</td></tr><tr><td>20_1D</td><td>19</td><td>0.684</td><td>0.474</td><td>0.705</td><td>0.037</td><td>0.632-0.737</td></tr><tr><td>20_1E</td><td>17</td><td>0.824</td><td>0.765</td><td>0.782</td><td>0.088</td><td>0.647–0.882</td></tr><tr><td>40_1</td><td>38</td><td>0.737</td><td>0.605</td><td>0.700</td><td>0.051</td><td>0.605-0.763</td></tr><tr><td>40_1_3</td><td>40</td><td>0.725</td><td>0.475</td><td>0.700</td><td>0.051</td><td>0.625-0.775</td></tr><tr><td>20_1_3</td><td>20</td><td>0.550</td><td>0.500</td><td>0.625</td><td>0.054</td><td>0.550-0.750</td></tr><tr><td>Mean over datasets</td><td>一</td><td>0.7193</td><td>0.5989</td><td>0.7155</td><td></td><td> $\mathrm { S D } _ { \mathrm { s e e d } } = 0 . 0 1 5 6$ </td></tr></table>

over 197 scored reference pairs. The historical frozen scheme gives a macro-OQ of 0.7193, close to the ten-scheme macro-average. On the same nine systems and prefixes, VAR-LiNGAM obtains macro and pair-weighted OQ values of 0.5989 and 0.5584. Its macro score is below the historical QS-TCN score. The comparison remains descriptive, because VAR-LiNGAM supplies one deterministic contemporaneous order whereas the scoring constraints include lagged relations.

The highest dataset-level mean is on manyinputs, where all six scored reference constraints are oriented correctly under every seed scheme,

$$
\mathrm { O Q } _ { \mathrm { m a n y i n p u t s } } = 1 .\tag{16}
$$

Only six directed constraints are scored for this system. The value therefore means perfect agreement on these six admissible pairs and does not mean that every possible pairwise relation among the 25 variables is recovered. The systems 20\_1A (0.806) and 20\_1E (0.782) also show high mean OQ. These values are descriptive and are not presented as dataset-wise significance claims, since no graph-specific random-order distribution has been evaluated.

The weaker systems show a diferent pattern. Take 20\_1C. Its mean OQ is 0.532 and its seed-scheme SD is 0.017, the smallest nonzero value among the Finance systems. Its agreement with the reference constraints is therefore stable across the ten seed schemes, at a level close to the random-order expectation of 0.5. A larger seed-scheme SD would instead show that OQ is sensitive to the bootstrap construction. These two properties are separate. Seed-scheme stability describes the reproducibility of the ordering score, and OQ measures agreement with the reference. Equal OQ values do not imply identical recovered orders. On Finance, QS-TCN reaches a macro-OQ of 0.7155 and a pair weighted OQ of 0.6914, both above the random-order expectation of 0.5, together with large diferences between individual systems. The per-system VAR-LiNGAM comparison appears in Table 2, and no marginal-statistic baseline is included.

## 4.3 fMRI benchmark

The fMRI experiments use the NetSim systems of Smith et al. [26], distributed in preprocessed form with TCDF [24]. These simulations generate directed neural activity and then observe it through BOLD dynamics, which adds temporal smoothing and nonlinear haemodynamic efects [35]. Twenty-seven datasets are present in the frozen QS-TCN evaluation, and all fMRI aggregates below are restricted to these systems, as stated in the dataset description above.

Averaging the 27 dataset-level values over the ten seed schemes gives

$$
\mathrm { O Q } _ { \mathrm { f M R I , m a c r o } } = 0 . 5 4 0 2 .\tag{17}
$$

Its seed-scheme standard deviation is $\mathrm { S D } _ { \mathrm { s e e d } } = 0 . 0 1 8 7$ , and the scheme-level values range from

$$
0 . 5 1 4 2 \leq \mathrm { O Q } _ { \mathrm { f M R I , m a c r o } } ^ { ( s ) } \leq 0 . 5 6 2 6 .\tag{18}
$$

As for Finance, 0.0187 describes the sensitivity of the benchmark mean to the seed scheme and is not a confidence interval. From the dataset means of Table 3, the calculation gives $\mathrm { S D _ { d a t a s e t s } \approx 0 . 1 4 1 }$ . Giving the same weight to each of the 239 scored reference constraints yields

$$
\mathrm { O Q } _ { \mathrm { f M R I , w e i g h t e d } } = 0 . 5 2 5 1 .\tag{19}
$$

Equivalently, 1255 of the 2390 constraint evaluations (239 constraints in each of the ten schemes) are correctly oriented. A uniformly random permutation has an expected OQ of 0.5, so the pair-weighted fMRI result lies only slightly above this expectation. Without a graph-level permutation analysis, the aggregate fMRI score is not interpreted as a statistically established departure from random ordering. The historical frozen scheme gives a macro-OQ of 0.5236. On the same 27 systems and estimation prefixes, VAR-LiNGAM obtains macro and pair-weighted OQ values of 0.3764 and 0.4100. These scores are lower than the historical QS-TCN aggregate. The contemporaneous-versuslagged diference limits the interpretation, and the fMRI benchmark remains highly heterogeneous across datasets.

Complete seed-scheme values are reported in Appendix A.

The dataset-level results difer widely. Dataset 24 has the highest mean OQ (0.840), followed by datasets 8 and 9 (0.780). These maxima are reported descriptively, for two reasons. Eighteen of the 27 fMRI systems contain five scored reference pairs, so one correctly or incorrectly ordered pair changes OQ by 0.2. In addition, the largest values observed among 27 inspected datasets are weak evidence of general fMRI recovery. At the other end, dataset 6 has a mean OQ of 0.318 with relatively low seed-scheme dispersion. Dataset 20 has a mean of the same order but the widest seed-scheme range in the benchmark,

$$
0 \leq \mathrm { O Q } _ { 2 0 } ^ { ( s ) } \leq 0 . 8 .\tag{20}
$$

Dataset 26 also ranges from 0.2 to 0.8, whereas the OQ values of datasets 7, 13, 23, 25 and 28 stay identical across the ten seed schemes.

Table 3: fMRI ordering quality by dataset. QS-TCN A is the historical frozen order; VAR-LiNGAM is one deterministic order fitted on the same estimation prefix. Mean, standard deviation and range for QS-TCN are computed across the ten bootstrap seed schemes A–J.
<table><tr><td>Dataset</td><td>Pairs</td><td>QS-TCN A</td><td>VAR-LiNGAM</td><td>Mean OQ</td><td>SD</td><td>Min-Max</td></tr><tr><td>1</td><td>5</td><td>0.400</td><td>0.400</td><td>0.460</td><td>0.165</td><td>0.200–0.600</td></tr><tr><td>2</td><td>11</td><td>0.545</td><td>0.727</td><td>0.618</td><td>0.120</td><td>0.455–0.818</td></tr><tr><td>3</td><td>18</td><td>0.444</td><td>0.444</td><td>0.439</td><td>0.032</td><td>0.389–0.500</td></tr><tr><td>4</td><td>61</td><td>0.475</td><td>0.475</td><td>0.503</td><td>0.021</td><td>0.475-0.525</td></tr><tr><td>5</td><td>5</td><td>0.400</td><td>0.200</td><td>0.380</td><td>0.063</td><td>0.200-0.400</td></tr><tr><td>6</td><td>11</td><td>0.273</td><td>0.273</td><td>0.318</td><td>0.048</td><td>0.273-0.364</td></tr><tr><td>7</td><td>5</td><td>0.400</td><td>0.000</td><td>0.400</td><td>0.000</td><td>0.400–0.400</td></tr><tr><td>8</td><td>5</td><td>0.800</td><td>0.800</td><td>0.780</td><td>0.063</td><td>0.600–0.800</td></tr><tr><td>9</td><td>5</td><td>0.800</td><td>0.400</td><td>0.780</td><td>0.063</td><td>0.600–0.800</td></tr><tr><td>10</td><td>5</td><td>0.600</td><td>0.400</td><td>0.600</td><td>0.163</td><td>0.200–0.800</td></tr><tr><td>11</td><td>11</td><td>0.545</td><td>0.455</td><td>0.636</td><td>0.096</td><td>0.455-0.727</td></tr><tr><td>12</td><td>11</td><td>0.727</td><td>0.455</td><td>0.645</td><td>0.125</td><td>0.455-0.818</td></tr><tr><td>13</td><td>8</td><td>0.625</td><td>0.375</td><td>0.625</td><td>0.000</td><td>0.625–0.625</td></tr><tr><td>14</td><td>5</td><td>0.400</td><td>0.200</td><td>0.640</td><td>0.158</td><td>0.400–0.800</td></tr><tr><td>15</td><td>5</td><td>0.600</td><td>0.000</td><td>0.560</td><td>0.084</td><td>0.400–0.600</td></tr><tr><td>16</td><td>7</td><td>0.429</td><td>0.286</td><td>0.443</td><td>0.171</td><td>0.143-0.714</td></tr><tr><td>17</td><td>11</td><td>0.273</td><td>0.273</td><td>0.436</td><td>0.103</td><td>0.273-0.545</td></tr><tr><td>18</td><td>5</td><td>0.600</td><td>0.600</td><td>0.560</td><td>0.126</td><td>0.200–0.600</td></tr><tr><td>19</td><td>5</td><td>0.600</td><td>0.400</td><td>0.400</td><td>0.189</td><td>0.200-0.800</td></tr><tr><td>20</td><td>5</td><td>0.000</td><td>0.400</td><td>0.320</td><td>0.253</td><td>0.000–0.800</td></tr><tr><td>22</td><td>5</td><td>0.200</td><td>0.600</td><td>0.320</td><td>0.103</td><td>0.200–0.400</td></tr><tr><td>23</td><td>5</td><td>0.600</td><td>0.000</td><td>0.600</td><td>0.000</td><td>0.600–0.600</td></tr><tr><td>24</td><td>5</td><td>0.800</td><td>0.400</td><td>0.840</td><td>0.084</td><td>0.800-1.000</td></tr><tr><td>25</td><td>5</td><td>0.600</td><td>0.200</td><td>0.600</td><td>0.000</td><td>0.600–0.600</td></tr><tr><td>26</td><td>5</td><td>0.800</td><td>0.200</td><td>0.560</td><td>0.227</td><td>0.200-0.800</td></tr><tr><td>27</td><td>5</td><td>0.600</td><td>0.600</td><td>0.520</td><td>0.169</td><td>0.200-0.600</td></tr><tr><td>28</td><td>5</td><td>0.600</td><td>0.600</td><td>0.600</td><td>0.000</td><td>0.600–0.600</td></tr><tr><td>Mean over datasets</td><td></td><td>0.5236</td><td>0.3764</td><td>0.5402</td><td></td><td> $\mathrm { S D } _ { \mathrm { s e e d } } = 0 . 0 1 8 7$ </td></tr></table>

The aggregate fMRI evidence therefore does not support a claim of uniform ordering recovery. The evidence shows a near-chance pair-weighted score together with strong heterogeneity across datasets. The macro-average moves little across seed schemes (0.0187), while the variation between datasets (≈ 0.141) is much larger. Reporting every seedlevel value separates the benchmark-level average from the stability of each dataset-level OQ.

## 4.4 Comparison with external lead-lag ordering

The frozen full QS-TCN model is compared with the CCF-AUC ordering of Bennett et al. [27]. The comparator is run with the authors’ code, distance correlation, a maximum lag of five, and decreasing row means. Each regime contains 100 systems with five variables, and both methods use identical 780-observation prefixes. QS-TCN retains its ten-model ensemble, whereas the comparator requires no supervised training. A volatility-oriented adaptation applied to squared standardized observations is evaluated as well.

Table 4: Mean ordering quality on synthetic systems. The linear lead-lag control is scored against factor-delay precedence.
<table><tr><td>Regime</td><td>QS-TCN</td><td>Native dCor</td><td>Squared-input dCor</td></tr><tr><td>Positive linear volatility</td><td>0.8595</td><td>0.5740</td><td>0.7726</td></tr><tr><td>Signed linear volatility</td><td>0.7428</td><td>0.5475</td><td>0.7202</td></tr><tr><td>Original SV mechanism</td><td>0.5729</td><td>0.4643</td><td>0.5414</td></tr><tr><td>Target-state nonlinear volatility</td><td>0.7772</td><td>0.5633</td><td>0.6991</td></tr><tr><td>Linear lead-lag control</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr></table>

Table 5: Paired OQ diferences (QS-TCN minus comparator) over the same 100 systems, with descriptive 95% system-bootstrap intervals. W/T/L counts the systems in which QS-TCN scores higher, equal or lower.
<table><tr><td>Regime</td><td>Comparator</td><td>Difference</td><td>95% interval</td><td>W/T/L</td></tr><tr><td rowspan="2">Positive linear</td><td>Native dCor</td><td>+0.2855</td><td>[+0.2351, +0.3351]</td><td>78/16/6</td></tr><tr><td>Squared-input dCor</td><td>+0.0869</td><td>[+0.0535, +0.1206]</td><td>45/42/13</td></tr><tr><td rowspan="2">Signed linear</td><td>Native dCor</td><td>+0.1953</td><td>[+0.1441, +0.2462]</td><td>67/14/19</td></tr><tr><td>Squared-input dCor</td><td>+0.0226</td><td>[-0.0291, +0.0758]</td><td>35/33/32</td></tr><tr><td rowspan="2">Original SV</td><td>Native dCor</td><td>+0.1086</td><td>[+0.0457, +0.1714]</td><td>53/16/31</td></tr><tr><td>Squared-input dCor</td><td>+0.0314</td><td>[-0.0229, +0.0857]</td><td>44/23/33</td></tr><tr><td rowspan="2">Target-state nonlinear</td><td>Native dCor</td><td>+0.2139</td><td>[+0.1636, +0.2637]</td><td>69/21/10</td></tr><tr><td>Squared-input dCor</td><td>+0.0781</td><td>[+0.0359, +0.1210]</td><td>46/36/18</td></tr></table>

QS-TCN achieves a higher mean OQ than both tabulated comparators in all four volatility regimes. The original SV mechanism is the hardest regime for all methods, with mean OQ between 0.4643 and 0.5729. Table 5 gives the paired diferences, together with the number of systems in which QS-TCN scores higher, equal or lower than the comparator.

Against the native ordering, all four intervals are positive. Against squared-input dCor, the paired gains are 0.0869 for positive linear coupling and 0.0781 for target-state nonlinear coupling, and the other two intervals include zero. The native Pearson, signature and lag-eight distance-correlation variants of the comparator also give positive intervals in all four volatility regimes. On the control, QS-TCN and the distancecorrelation and Pearson variants reach 1.0000, and the signature variant reaches 0.9460. The estimates for these additional variants are provided in results/summary.csv and results/paired\_differences.csv of the accompanying comparison archive. The intervals are the 2.5th and 97.5th percentiles of the mean paired OQ diferences obtained from 20,000 bootstrap resamples of systems. The paired intervals are descriptive and are not adjusted for multiple comparisons. The comparison remains exploratory, because the volatility benchmarks had been examined before.

Table 6: Selected directional relations from the frozen Forex/Gold QS-TCN analysis. $C _ { i j }$ denotes the aggregated antisymmetric pairwise evidence and $R _ { i j }$ its associated reported reliability.
<table><tr><td>Directional relation</td><td> $C _ { i j }$ </td><td> $R _ { i j }$ </td></tr><tr><td>GBPUSD → USDCAD</td><td>0.0892</td><td>0.847</td></tr><tr><td>GOLD → GBPUSD</td><td>0.0691</td><td>0.775</td></tr><tr><td>GOLD → USDCAD</td><td>0.0958</td><td>0.759</td></tr><tr><td>USDCHF → USDCAD</td><td>1.2944</td><td>0.621</td></tr><tr><td>USDCHF → EURUSD</td><td>0.3230</td><td>0.589</td></tr><tr><td>USDJPY → GOLD</td><td>0.1116</td><td>0.450</td></tr></table>

## 4.5 Forex and Gold markets

Data. Daily closing prices are taken from the public GitHub repository ejtraderLabs/ historical-data. The repository provides one daily file for each of the six currency pairs and for gold quoted against the dollar (XAUUSD, labelled GOLD here). Each file holds 2,400 daily records ending on 4 March 2022. Only dates for which all seven series have a price are retained, which leaves 2,385 common dates from 4 December 2012 to 4 March 2022. The returns are log-returns, $r _ { t } = \log P _ { t } - \log P _ { t - 1 }$ , computed between consecutive common dates, so a return over a weekend or a holiday covers more than one calendar day. This gives $T = 2 3 8 4$ returns, from 5 December 2012 to 4 March 2022. Zero returns occur in the panel, with 3 for EURUSD, 4 for GBPUSD, 5 for USDJPY, 3 for USDCHF, none for AUDUSD, 4 for USDCAD and 1 for GOLD. These make 20 zeros among 16,688 entries, each on a diferent date. These dates are retained, because deleting every date with a zero would reduce the sample to 2364 returns.

The QS-TCN pipeline is applied to an observed market panel with T = 2384 observations of seven assets, namely EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD and GOLD. Unlike Finance and fMRI, this experiment has no complete causal-order ground truth, so no reference-based accuracy score is reported. The analysis reports the global order returned by the same inference mechanism as in the benchmarks and examines the strongest pairwise relations behind this order.

The full-series QS-TCN analysis returns

$$
\mathrm { U S D C H F \prec U S D J P Y \prec G O L D \prec E U R U S D \prec G B P U S D \prec A U D U S D \prec U S D C A D } .\tag{21}
$$

The pairwise field behind this order is uneven. After bootstrap aggregation, the reported bootstrap weights difer across source-target relations. Table 6 lists the strongest retained examples.

This sequence is interpreted only as the order induced by the QS-TCN directional score and not as a validated causal graph. The $R _ { i j }$ values are the bootstrap weights reported with the archived pairwise scores, and they are not probabilities that a direction is correct. The magnitude of $C _ { i j }$ difers widely across relations, and this diference is not interpreted, because the sensitivity of the lag-asymmetry anchor to node scale is not assessed in the experiments reported in this section. Without a complete market reference structure and without marginal-ordering controls, the Forex/Gold experiment is treated as an application of the frozen ordering mechanism. The experiment should therefore not be read as validating either causal direction or independence from marginal scale.

Table 7: Comparison of QS-TCN cryptocurrency orderings with directional structures reported in independent studies. Concordance is computed only for relations that map unambiguously onto the corresponding asset set.
<table><tr><td>Study</td><td>Published comparison object</td><td>QS-TCN result</td><td>Agreement</td><td></td></tr><tr><td>Ji et al. [36]</td><td>directions and a separate net- spillover ranking</td><td>15 pairwise volatility-spillover DASH &lt; BTC &lt; LTC &lt; XRP &lt; XLM &lt; ETH Pairwise: 10/15 ≈ 0.667,</td><td>ing: τ = 0.60</td><td>Pperm = 0.2347; net rank-</td></tr><tr><td>Yousaf–Ali [38]</td><td>transmission relations</td><td>Regime-specific cryptocurrency Pre-COVID: ETH &lt; LTC &lt; BTC; COVID: 2/5 pre-COVID; 2/3 BTC &lt; ETH &lt; LTC</td><td>COVID; combined agree- ment = 4/8</td><td></td></tr></table>

## 4.6 Cryptocurrency market comparisons

Cryptocurrency markets provide a second real-data setting, in which the QS-TCN output is confronted with evidence produced independently of QS-TCN. Ji et al. [36] analyze return and volatility connectedness across six major cryptocurrencies. Yousaf and Ali [38] study return and volatility transmission among Bitcoin, Ethereum and Litecoin before and during the COVID-19 period. These studies use diferent statistical objects, samples and modelling assumptions, so their reported directions are not treated as causal ground truth. The comparison asks a narrower question, namely how much of the independently reported directional structure the QS-TCN order reproduces. A published relation $^ { 6 6 } a$ before $b ^ { \prime \prime }$ counts as reproduced when a precedes b in the QS-TCN order. All permutation probabilities in this subsection are one-sided, and each gives the probability, under uniformly random orders of the assets, of at least the observed number of reproduced relations.

## 4.6.1 Comparison with Ji et al.

Ji et al. [36] examine connectedness through return and volatility spillovers across six large cryptocurrencies. For the corresponding six-asset system, the frozen full-series QS-TCN ordering is

$$
\mathrm { D A S H \prec B T C \prec L T C \prec X R P \prec X L M \prec E T H } .\tag{22}
$$

Among the 15 published pairwise directional relations used in the comparison, QS-TCN satisfies

$$
{ \frac { 1 0 } { 1 5 } } \approx 0 . 6 6 6 7 .\tag{23}
$$

The archived exact one-sided enumeration over the $6 ! \ : = \ : 7 2 0$ possible orders, which counts the orders satisfying at least 10 of the 15 published relations, gives

$$
p _ { \mathrm { p e r m } } = 0 . 2 3 4 7 .\tag{24}
$$

The archived Kendall agreement with the separately reported net-spillover ranking is

$$
\tau = 0 . 6 0 .\tag{25}
$$

For two strict rankings of six assets, $\tau = ( n _ { c } - n _ { d } ) / 1 5$ , where $n _ { c }$ and $n _ { d }$ count the concordant and discordant pairs. The value $\tau = 0 . 6 0$ thus corresponds to $n _ { c } ~ = ~ 1 2$ and $n _ { d } = 3$ The two agreement numbers describe diferent comparison objects and must not be mixed. The value 10/15 compares QS-TCN with the 15 published pairwise spillover directions, whereas $\tau = 0 . 6 0$ compares the QS-TCN order with the separately reported net-spillover ranking. The pairwise concordance is not unusual under exact random-order enumeration $( p _ { \mathrm { p e r m } } = 0 . 2 3 4 7 )$ . The experiment is therefore interpreted as partial concordance between studies and not as independent validation of the QS-TCN ordering.

## 4.6.2 Comparison with Yousaf and Ali

Yousaf and Ali [38] analyze high-frequency return and volatility transmission among Bitcoin, Ethereum and Litecoin, and they separate the pre-COVID-19 and COVID-19 periods. This separation is retained and the complete sample is not pooled before inference, because the published study reports that transmission changes between the two regimes.

The directional constraints come from the cross-coeficients $b _ { i j }$ of Table 3 of Yousaf and Ali. A relation is retained only when its coeficient is significant at the 5% level. With the source-target convention stated in the table note, this rule gives five pre-COVID constraints, BTC→ETH, BTC→LTC, ETH→BTC, ETH→LTC and LTC→ETH, and three COVID constraints, BTC→ETH, BTC→LTC and LTC→ETH. Negative coeficients keep their source-target orientation (Appendix C). The input is the series of hourly log-returns. Within each regime, the returns are split into consecutive non-overlapping windows of 1000 returns, starting at the beginning of the regime, and the incomplete tail is dropped. QS-TCN is run on every window under the ten frozen seeds. The consensus ordering of a regime is the rank consensus over all window-seed pairs of that regime, so the assets are sorted by their mean position across these orders. The two regimes are scored separately, and the final score is the sum of the two hit counts, $H = \mathrm { \ h i t s { _ { p r e - C O V I D } } + \ h i t s { _ { C O V I D } } }$ . The random-order law uses one independent uniform order per regime, which gives $6 \times 6 = 3 6$ equally likely joint outcomes.

For the pre-COVID period, ten complete 1,000-return windows are available, and the QS-TCN consensus ordering is

$$
{ \mathrm { E T H } } \prec { \mathrm { L T C } } \prec { \mathrm { B T C } } .\tag{26}
$$

This order satisfies

$$
2 / 5\tag{27}
$$

of the pre-specified directional constraints retained from the published comparison object. For the COVID period, two complete 1,000-return windows are available, and the recovered order changes to

$$
\mathrm { B T C } \prec \mathrm { E T H } \prec \mathrm { L T C } ,\tag{28}
$$

which satisfies

$$
2 / 3\tag{29}
$$

of the corresponding COVID-period constraints. Combining the two regimes gives an overall agreement of

$$
{ \frac { 4 } { 8 } } = 0 . 5 0 .\tag{30}
$$

The exact random-order distribution has the same median value, and the one-sided probability of at least four agreements under independent random orders in the two regimes is

$$
p _ { \mathrm { p e r m } } = 0 . 6 6 6 7 .\tag{31}
$$

The QS-TCN order changes between the pre-COVID and COVID windows. This change is qualitatively compatible with regime-dependent transmission, but it does not validate the specific directional structure reported by Yousaf and Ali. Across the eight retained directional constraints, the observed agreement is $4 / 8 _ { ; }$ , and the exact random-order comparison gives $p _ { \mathrm { p e r m } } = 0 . 6 6 6 7$ . This result is therefore retained as a negative comparison and not as supporting evidence.

## A Supplementary Material: Seed-wise Benchmark Ordering Quality

The following tables report the ordering quality for each of the ten frozen bootstrap seed schemes (A–J). They provide the per-scheme values underlying the benchmark summaries in Section 4.

Table 8: Finance ordering quality for all ten frozen bootstrap seed schemes.
<table><tr><td>Dataset</td><td>A</td><td>B</td><td>C</td><td>D</td><td>E</td><td>F</td><td>G</td><td>H</td><td>I</td><td>J</td><td>Mean</td></tr><tr><td>manyinputs</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td></tr><tr><td>20_iA</td><td>.778</td><td>.889</td><td>.833</td><td>.833</td><td>.778</td><td>.889</td><td>.722</td><td>.778</td><td>.778</td><td>.778</td><td>.806</td></tr><tr><td>20 1B</td><td>.650</td><td>.550</td><td>.650</td><td>.600</td><td>.550</td><td>.650</td><td>.550</td><td>.500</td><td>.600</td><td>.600</td><td>.590</td></tr><tr><td>20 1C</td><td>.526</td><td>.526</td><td>.526</td><td>.526</td><td>.579</td><td>.526</td><td>.526</td><td>.526</td><td>.526</td><td>.526</td><td>.532</td></tr><tr><td>20 _1D</td><td>.684</td><td>.737</td><td>.737</td><td>.684</td><td>.737</td><td>.684</td><td>.737</td><td>.632</td><td>.684</td><td>.737</td><td>.705</td></tr><tr><td>20 _1E</td><td>.824</td><td>.824</td><td>.647</td><td>.765</td><td>.824</td><td>.882</td><td>.647</td><td>.706</td><td>.882</td><td>.824</td><td>.782</td></tr><tr><td>40 1</td><td>.737</td><td>.711</td><td>.658</td><td>.763</td><td>.737</td><td>.737</td><td>.658</td><td>.737</td><td>.658</td><td>.605</td><td>.700</td></tr><tr><td>40 1_3</td><td>.725</td><td>.650</td><td>.650</td><td>.700</td><td>.725</td><td>.700</td><td>.675</td><td>.775</td><td>.625</td><td>.775</td><td>.700</td></tr><tr><td>201_3</td><td>.550</td><td>.600</td><td>.600</td><td>.600</td><td>.650</td><td>.650</td><td>.750</td><td>.650</td><td>.600</td><td>.600</td><td>.625</td></tr></table>

T<sub>a</sub>bl<sub>e</sub> 9 <sub>:</sub> fMRI <sub>or</sub>d<sub>er</sub>i<sub>ng qua</sub>lit<sub>y</sub> f<sub>or every</sub> f<sub>rozen</sub> b<sub>oo</sub>t<sub>s</sub>t<sub>rap see</sub>d <sub>sc</sub>h<sub>eme .</sub> Th<sub>e</sub> t<sub>a</sub>bl<sub>e g</sub>i<sub>ves</sub> th<sub>e</sub> t<sub>en recor</sub>d<sub>e</sub>d <sub>see</sub>d-<sub>sc</sub>h<sub>eme scores</sub> <sub>un</sub>d<sub>er</sub>l<sub>y</sub>i<sub>ng</sub> th<sub>e</sub> fMRI <sub>summar</sub>i<sub>es</sub> i<sub>n</sub> T<sub>a</sub>bl<sub>e</sub> 3
<table><tr><td>DS</td><td>A</td><td>B</td><td>C</td><td>D</td><td>E</td><td>F</td><td>G</td><td>H</td><td>I</td><td></td><td>J Mean</td><td>SD</td></tr><tr><td>1</td><td>.400</td><td>.200</td><td>.600</td><td>.400</td><td>.200</td><td>.600</td><td>.600</td><td>.600</td><td>.600</td><td>.400</td><td>.460.165</td><td></td></tr><tr><td>2</td><td>.545</td><td>.636</td><td>.455</td><td>.545</td><td>.636</td><td>.727</td><td>.818</td><td>.727</td><td>.636</td><td>.455</td><td>.618 .120</td><td></td></tr><tr><td>3</td><td>.444</td><td>.389</td><td>.444</td><td>.444</td><td>.444</td><td>.444</td><td>.444</td><td>.444</td><td>.389</td><td>.500</td><td></td><td>.439.032</td></tr><tr><td>4</td><td>.475</td><td>.492</td><td>.525</td><td>.525</td><td>.475</td><td>.525</td><td>5.492</td><td>.492</td><td>.525</td><td>.508</td><td></td><td>.503 .021</td></tr><tr><td>5</td><td>.400</td><td>.400</td><td>.400</td><td>.400</td><td>.400</td><td>.200</td><td>.400</td><td>.400</td><td>.400</td><td>.400</td><td></td><td>.380.063</td></tr><tr><td>6</td><td>.273</td><td>.273</td><td>.273</td><td>.364</td><td>1.273</td><td></td><td>.364.364</td><td>.364</td><td>.364</td><td>4.273</td><td></td><td>.318.048</td></tr><tr><td>7</td><td>.400</td><td>.400</td><td>.400</td><td>.400</td><td>.400</td><td></td><td>.400 .400</td><td>.400</td><td>.400</td><td>.400</td><td></td><td>.400.000</td></tr><tr><td>8</td><td>.800</td><td>.800</td><td>.800</td><td>.800</td><td>.800</td><td></td><td>.800.800</td><td></td><td>.800.800</td><td>.600</td><td></td><td>.780.063</td></tr><tr><td>9</td><td>.800</td><td>.800</td><td>.800</td><td>0.600.800</td><td></td><td></td><td>.800 .800</td><td></td><td>.800 .800 .800</td><td></td><td></td><td>.780 .063</td></tr><tr><td>10</td><td>.600</td><td>.800</td><td>.600</td><td>.600</td><td>.800</td><td></td><td>.200 .600</td><td>.600</td><td>.600</td><td>.600</td><td></td><td>.600.163</td></tr><tr><td>11</td><td>.545</td><td>.636</td><td>.545</td><td>.727</td><td>.727</td><td>.727</td><td>7.455</td><td>.636</td><td>.727</td><td>.636</td><td></td><td>.636 .096</td></tr><tr><td>12</td><td>.727</td><td>.545</td><td>.455</td><td>.818</td><td>.818</td><td>.545</td><td>.636</td><td>.636</td><td>.727</td><td>.545</td><td></td><td>.645 .125</td></tr><tr><td>13</td><td>.625</td><td>.625</td><td>.625</td><td>.625</td><td>.625</td><td></td><td>.625 .625</td><td>.625</td><td>.625</td><td>.625</td><td></td><td>.625 .000</td></tr><tr><td>14</td><td>.400</td><td>.800</td><td>.600</td><td>.800</td><td>.600</td><td></td><td>.800 .600</td><td>.400</td><td>.800</td><td>.600</td><td></td><td>.640.158</td></tr><tr><td>15</td><td>.600</td><td>.600</td><td>.600</td><td>0.400.600</td><td></td><td></td><td>.600.600</td><td></td><td>.600 .400</td><td>.600</td><td></td><td>.560.084</td></tr><tr><td>16</td><td>.429</td><td>.429</td><td>.429</td><td>.429</td><td>.429</td><td>.714 .429</td><td></td><td>.286</td><td>.143</td><td>.714</td><td></td><td>.443 .171</td></tr><tr><td>17</td><td>.273</td><td>.455</td><td>.545</td><td>.455</td><td>.455</td><td>.364 .273</td><td></td><td>.545</td><td>.455</td><td>.545</td><td></td><td>.436 .103</td></tr><tr><td>18</td><td>.600</td><td>.600</td><td>.600</td><td>.600</td><td>.200</td><td>.600.600</td><td></td><td>.600</td><td>.600</td><td>.600</td><td>.560.126</td><td></td></tr><tr><td>19</td><td>.600</td><td>.200</td><td>.400</td><td>.400.400</td><td></td><td>.400.200</td><td></td><td></td><td>.200 .400 .800</td><td></td><td>.400.189</td><td></td></tr><tr><td>20</td><td>.000</td><td>.400</td><td>.200</td><td>.200</td><td>.000</td><td>.400</td><td>.800</td><td>.400</td><td>.600</td><td>.200</td><td></td><td>.320.253</td></tr><tr><td>22</td><td>.200</td><td>.400</td><td>.400</td><td>.400</td><td>.400</td><td>.200</td><td>.400</td><td>.200</td><td>.200</td><td>.400</td><td>.320.103</td><td></td></tr><tr><td>23</td><td>.600</td><td>.600</td><td>.600</td><td>.600.600</td><td></td><td></td><td>.600.600</td><td></td><td>.600 .600 .600</td><td></td><td></td><td>.600.000</td></tr><tr><td>24</td><td>.800</td><td>.800</td><td>.800</td><td>.800.800</td><td></td><td>1.000.800</td><td></td><td>1.000</td><td>0.800.800</td><td></td><td></td><td>.840.084</td></tr><tr><td>25</td><td>.600</td><td>.600</td><td>.600</td><td>.600</td><td>.600</td><td>.600.600</td><td></td><td>.600</td><td>.600</td><td>.600</td><td>.600.000</td><td></td></tr><tr><td>26</td><td>.800</td><td>.800</td><td>.400</td><td>.600.200</td><td></td><td></td><td>.600.600</td><td>.200</td><td>.800.600</td><td></td><td>.560.227</td><td></td></tr><tr><td>27</td><td>.600</td><td>.200</td><td>.200</td><td>.600</td><td>.600</td><td></td><td>.600.600</td><td>.600</td><td>.600</td><td>.600</td><td>.520.169</td><td></td></tr><tr><td>28</td><td>.600</td><td>.600</td><td>.600 .600 .600</td><td></td><td></td><td></td><td>.600.600</td><td></td><td>.600 .600 .600</td><td></td><td>.600 .000</td><td></td></tr></table>

## B Squared-memory modes

Under the stationary Gaussian log-volatility model in Eq. (2), let $v = \sigma _ { \eta } ^ { 2 } / ( 1 - \phi ^ { 2 } )$ and omit the node index. Independent standard Gaussian measurement innovations, also independent of the log-volatility process, give $\mathbb { E } [ X _ { t } ^ { 2 } ] = \exp ( \mu + v / 2 )$ . Since $\mathrm { C o v } ( h _ { t } , h _ { t - k } ) =$ $v \phi ^ { k }$ , the Gaussian exponential-moment identity yields, for $k \geq 1$

$$
\begin{array} { r l r } & { } & { \mathrm { C o v } ( X _ { t } ^ { 2 } , X _ { t - k } ^ { 2 } ) = \exp ( 2 \mu + v ) \left[ \exp ( v \phi ^ { k } ) - 1 \right] } \\ & { } & { = \exp ( 2 \mu + v ) \displaystyle \sum _ { m = 1 } ^ { \infty } \frac { v ^ { m } } { m ! } ( \phi ^ { m } ) ^ { k } . } \end{array}\tag{32}
$$

Thus several geometric modes can arise from a single persistence parameter. Channel I characterizes these observable modes without assigning them a causal origin. The restriction to positive lags is necessary, because the lag-zero expression also involves the fourth moment of the measurement innovation.

## C Published Yousaf–Ali comparison set

The comparison below measures agreement with published spillover estimates. These estimates serve as comparison objects and not as causal ground truth.

The comparison retains all of-diagonal $b _ { i j }$ coeficients in Table 3 of Yousaf and Ali [38] with reported $p \ < \ 0 . 0 5$ . Directions follow the table note, which interprets $b _ { 1 2 }$ as BTC→ETH, irrespective of coeficient sign. The indexing in their variance equation suggests the opposite convention, so the reversal of all directions is also checked. Printed p-values of 0.000 are rounded values.

Table 10: Retained Yousaf–Ali coeficients under the direction convention stated in their Table 3 note.
<table><tr><td>Period</td><td>Direction</td><td>Coefficient</td><td>p-value</td></tr><tr><td>Pre-COVID</td><td>BTC → ETH</td><td>-0.144</td><td>0.000</td></tr><tr><td>Pre-COVID</td><td>BTC → LTC</td><td>-0.129</td><td>0.000</td></tr><tr><td>Pre-COVID</td><td>ETH → BTC</td><td>-0.058</td><td>0.032</td></tr><tr><td>Pre-COVID</td><td>ETH → LTC</td><td>-0.269</td><td>0.000</td></tr><tr><td>Pre-COVID</td><td>LTC → ETH</td><td>1.028</td><td>0.000</td></tr><tr><td>COVID</td><td>BTC → ETH</td><td>0.037</td><td>0.000</td></tr><tr><td>COVID</td><td>BTC → LTC</td><td>0.025</td><td>0.040</td></tr><tr><td>COVID</td><td>LTC → ETH</td><td>0.069</td><td>0.000</td></tr></table>

The reported orders satisfy $2 / 5$ pre-COVID constraints and $2 / 3$ COVID constraints, giving $4 / 8$ overall. Of the $6 \times 6 = 3 6$ pairs of independent uniform regime-specific orders, 24 achieve at least four agreements, which gives $p _ { \mathrm { p e r m } } = 0 . 6 6 6 7$ . Reversing all directions gives $3 / 5$ and $1 / 3$ , and the overall agreement is preserved. Reciprocal relations limit the maximum combined agreement to $6 / 8$

## D Implementation details

The settings below describe the reconstructed implementation used for the synthetic comparison. They do not establish numerical equivalence with the historical benchmark implementation.

## D.1 Structured temporal filters

Channel I supplies the rates used by the structured filters in Channel II. The Memory Cell applies the normalized geometric filter

$$
p _ { a } ( \ell ) = \frac { a ^ { \ell - 1 } } { \sqrt { \sum _ { s = 1 } ^ { L _ { m } } a ^ { 2 ( s - 1 ) } } + 1 0 ^ { - 8 } } , \qquad \ell = 1 , \dots , L _ { m } .\tag{33}
$$

This filter is applied at the target persistence $a ~ = ~ { \tilde { \phi } } _ { j }$ and at each accepted target pole $a = \widehat { \lambda } _ { j r }$ e<sub>.</sub> <sub>The</sub> <sub>readout</sub> <sub>combines</sub> <sub>these</sub> <sub>responses</sub> <sub>with</sub> <sub>free-filter</sub> <sub>responses</sub> <sub>and</sub> <sub>the</sub> descriptor $\Theta _ { j }$

For $i  j$ , the Heredity Cell filters the source history with Eq. (5), which combines $\widetilde { \phi } _ { i }$ and $\widehat { \lambda } _ { j r }$ . Each kernel is normalized by its Euclidean norm, and its unnormalized first bentry is one, including at coincident rates. Unsupported pole slots are masked. The estimated rates stay fixed, while the neural weights learn to combine the responses.

Both modules use eight-lag causal filters with unit dilation. With one-based indexing, memory predictions use $t = 9 , \ldots , n$ . A second eight-sample truncation leaves $t \ =$ $1 7 , \ldots , n$ for directional pooling.

## D.2 Statistical pairwise score

For each centered-square series $q _ { j }$ , autoregressions with an intercept are fitted. Candidate orders range from zero to four and share the fitting window $t = 5 , \ldots , n$ . With $N = n - 4$ , the selected order minimizes

$$
\mathrm { B I C } _ { j , p } = { N } \log \left( \frac { \mathrm { R S S } _ { j , p } } { N } \right) + ( p + 2 ) \log { N } .\tag{34}
$$

The selected coeficients define the residuals

$$
\zeta _ { j , t } = q _ { j , t } - \widehat { c } _ { j } - \sum _ { \ell = 1 } ^ { \widehat { p } _ { j } } \widehat { a } _ { j , \ell } q _ { j , t - \ell } .\tag{35}
$$

For $\mathcal { T } _ { k } = \{ \operatorname* { m a x } ( k , 4 ) + 1 , \dots , n \}$ , the overlap-centered covariance is

$$
\widehat { \Gamma } _ { i j } ( k ) = \frac { 1 } { | \mathcal { T } _ { k } | } \sum _ { t \in \mathcal { T } _ { k } } \Big ( q _ { i , t - k } - \overline { { q } } _ { i , k } \Big ) \Big ( \zeta _ { j , t } - \overline { { \zeta } } _ { j , k } \Big ) ,\tag{36}
$$

where both means are computed on the same overlap. The statistical score is

$$
S _ { 0 , i j } = \frac { 1 } { K _ { 0 } } \sum _ { k = 1 } ^ { K _ { 0 } } \left[ \widehat { \Gamma } _ { i j } ( k ) - \widehat { \Gamma } _ { j i } ( k ) \right] , \qquad K _ { 0 } = 8 .\tag{37}
$$

These autoregressive residuals difer from the Memory Cell residuals in Eq. (4). Autoregressions and covariances are refitted in every inference view. Candidates with non-finite RSS or $\mathrm { R S S } \leq 1 0 ^ { - 1 2 }$ are excluded, and fitting stops if none remains.

## D.3 Bootstrap aggregation and decoding

Inference uses $B = 1 9$ synchronized circular-block views, each of length n, with block length $b _ { n } = \operatorname* { m a x } \{ 1 2 , 3 \operatorname { r o u n d } ( n ^ { 1 / 3 } ) \}$ . Uniformly sampled blocks are concatenated and truncated to n indices, and the indices are shared across variables. Centered squares are computed before resampling. The Channel-I estimates and the neural parameters remain fixed, while $S _ { 0 }$ is refitted.

The pairwise scores are aggregated as

$$
\begin{array} { l } { { \displaystyle M _ { i j } = \mathrm { m e d i a n } _ { b = 1 , \ldots , B } S _ { i j } ^ { ( b ) } } , } \\ { { \displaystyle V _ { i j } = \mathrm { m e d i a n } _ { b = 1 , \ldots , B } \left. S _ { i j } ^ { ( b ) } - M _ { i j } \right. , } } \\ { { \displaystyle U _ { i j } = \left. \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \mathrm { s i g n } \big ( S _ { i j } ^ { ( b ) } \big ) \right. , } } \\ { { \displaystyle C _ { i j } = U _ { i j } \exp \left[ - \frac { V _ { i j } } { 2 ( \vert M _ { i j } \vert + 1 0 ^ { - 6 } ) } \right] M _ { i j } } , } \end{array}\tag{38}
$$

with $\mathrm { s i g n } ( 0 ) = 0$ . The weight combines sign stability $U _ { i j }$ and dispersion $V _ { i j }$ , and is not a probability of causal correctness. Antisymmetry is enforced through $C \stackrel { \cdot } {  } ( C - C ^ { \top } ) / 2$ with a zero diagonal.

For the synthetic comparison, the ten checkpoint-specific matrices are averaged before decoding. Content hashes define a canonical column order for tie-breaking, and the labels are restored afterwards. Starting from decreasing row sums, the decoder accepts single-node insertions with gain above $1 0 ^ { - 1 2 }$ , and stops when none improves the objective or after 25 moves.

## D.4 Training for the synthetic comparison

The checkpoints used in Section 4.4 are trained on separate banks of 160 systems for memory and 160 for directional learning, with 80 independent validation systems. Each bank covers the four volatility regimes equally. Every system has five variables and 1,200 observations, and only the first 780 are used. Each variable is centered and scaled by its empirical standard deviation on this prefix, with divisor $n ,$ before the centered squares are formed. The Channel-I estimates are fixed during training.

The Memory Cell minimizes squared prediction error over valid times and variables and is frozen after six epochs. For each synthetic graph, let P contain the ordered ancestordescendant pairs. The directional loss is

$$
\mathcal { L } _ { \mathrm { d i r } } = \frac { 1 } { | \mathcal { P } | } \sum _ { ( i , j ) \in \mathcal { P } } \log ( 1 + \exp ( - S _ { i j } / 0 . 2 5 ) ) + \frac { 0 . 0 2 } { d ^ { 2 } } \| S - S _ { 0 } \| _ { F } ^ { 2 } .\tag{39}
$$

Each optimizer step uses one system, and the systems are reshufled at each epoch. Training labels include ancestor pairs, whereas validation and test OQ use direct reference edges.

Table 11: Architecture and optimization settings for the synthetic-comparison checkpoints. Memory parameters are frozen before directional training.
<table><tr><td>Setting</td><td>Memory Cell</td><td>Directional module</td></tr><tr><td>Lag horizon</td><td>8</td><td>8</td></tr><tr><td>Free filters</td><td>12</td><td>12</td></tr><tr><td>Parameters</td><td>601</td><td>4,959</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $2 \times 1 0 ^ { - 3 }$ </td><td> $1 . 5 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Weight decay</td><td> $1 0 ^ { - 4 }$ </td><td> $2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Epochs</td><td>6</td><td>8</td></tr><tr><td>Gradient-norm clipping</td><td>5</td><td>5</td></tr></table>

The 5,560 parameters include 601 memory parameters, which are frozen during directional training. Readout dimensions are $1 9  2 4  1$ for memory, $9  2 4  1 2$ for modal weights, $8  1 6  1$ for branch mixing, $5 6 \to 4 8 \to 2 4 \to 1$ for directional scoring, and $1 2  1 6  1$ for the symmetric gate. Hidden layers use tanh, and the mixing and gate outputs use sigmoid. Free filters have independent Gaussian initialization with standard deviation 0.03. Linear layers use PyTorch initialization, and AdamW uses its default moment parameters.

Training seeds run from 97101 to 97110. For each run, the checkpoint with the highest mean validation OQ is selected among epochs 1 to 8 after 19-view bootstrap aggregation and decoding, with the 80 systems weighted equally. Exact ties retain the earliest epoch.

All ten checkpoints enter the ensemble. Generator settings, data seeds and checkpoint hashes are recorded in the accompanying implementation.

## E Conclusion

This paper introduced QS-TCN, a framework that orders the variables of a multivariate time series from volatility memory. The method links statistical estimates of persistence and temporal modes to structured filters within a shared temporal network, then combines pairwise directional scores into a global order. The historical benchmark evaluation yielded a mean OQ of 0.7155 on Finance and 0.5402 on fMRI. In a separate synthetic evaluation, the reconstructed implementation achieved higher mean OQ than the native lead–lag comparator in all four volatility regimes. Against the squared-input adaptation, descriptive bootstrap intervals for paired OQ diferences excluded zero in two regimes. These comparisons assess the complete framework and do not isolate the contribution of individual components. The recovered order describes precedence rather than direct causal links. Future work will examine the contribution of the structured filters and neural components, and evaluate QS-TCN as an ordering component for structural equation modelling.

## References

[1] C. W. J. Granger, “Investigating causal relations by econometric models and cross-spectral methods,” Econometrica, vol. 37, no. 3, pp. 424–438, 1969. doi: 10.2307/1912791.

[2] P. Spirtes, C. Glymour, and R. Scheines, Causation, Prediction, and Search, 2nd ed. Cambridge, MA: MIT Press, 2001. doi: 10.7551/mitpress/1754.001.0001.

[3] M. Eichler, “Graphical modelling of multivariate time series,” Probability Theory and Related Fields, vol. 153, pp. 233–268, 2012. doi: 10.1007/s00440-011-0345-8.

[4] J. Runge, P. Nowack, M. Kretschmer, S. Flaxman, and D. Sejdinovic, “Detecting and quantifying causal associations in large nonlinear time series datasets,” Science Advances, vol. 5, no. 11, Art. no. eaau4996, 2019. doi: 10.1126/sciadv.aau4996.

[5] R. Pamfil, N. Sriwattanaworachai, S. Desai, P. Pilgerstorfer, K. Georgatzis, P. Beaumont, and B. Aragam, “DYNOTEARS: Structure learning from time-series data,” in Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics, Proceedings of Machine Learning Research, vol. 108, pp. 1595–1605. PMLR, 2020.

[6] S. Shimizu, P. O. Hoyer, A. Hyvärinen, and A. Kerminen, “A linear non-Gaussian acyclic model for causal discovery,” Journal of Machine Learning Research, vol. 7, no. 72, pp. 2003–2030, 2006.

[7] A. Hyvärinen, K. Zhang, S. Shimizu, and P. O. Hoyer, “Estimation of a structural vector autoregression model using non-Gaussianity,” Journal of Machine Learning Research, vol. 11, no. 56, pp. 1709–1731, 2010.

[8] L. Xu, T. Fan, X. Wu, K. Chen, X. Guo, J. Zhang, and L. Yao, “A pooling-LiNGAM algorithm for efective connectivity analysis of fMRI data,” Frontiers in Computational Neuroscience, vol. 8, Art. no. 125, 2014. doi: 10.3389/fncom.2014.00125.

[9] T. Yang and J. Suzuki, “The functional LiNGAM,” in Proceedings of the 11th International Conference on Probabilistic Graphical Models, Proceedings of Machine Learning Research, vol. 186, pp. 25–36. PMLR, 2022.

[10] Y.-W. Cheung and L. K. Ng, “A causality-in-variance test and its application to financial market prices,” Journal of Econometrics, vol. 72, nos. 1–2, pp. 33–48, 1996. doi: 10.1016/0304-4076(94)01714-X.

[11] Y. Hong, “A test for volatility spillover with application to exchange rates,” Journal of Econometrics, vol. 103, nos. 1–2, pp. 183–224, 2001. doi: 10.1016/S0304-4076(01)00043-4.

[12] C. M. Hafner and H. Herwartz, “A Lagrange multiplier test for causality in variance,” Economics Letters, vol. 93, no. 1, pp. 137–141, 2006. doi: 10.1016/j.econlet.2006.04.008.

[13] C. M. Hafner and H. Herwartz, “Testing for causality in variance using multivariate GARCH models,” Annales d’Économie et de Statistique, no. 89, pp. 215–241, 2008. doi: 10.2307/27715168.

[14] R. F. Engle and K. F. Kroner, “Multivariate simultaneous generalized ARCH,” Econometric Theory, vol. 11, no. 1, pp. 122–150, 1995. doi: 10.1017/S0266466600009063.

[15] F. X. Diebold and K. Yilmaz, “Better to give than to receive: Predictive directional measurement of volatility spillovers,” International Journal of Forecasting, vol. 28, no. 1, pp. 57–66, 2012. doi: 10.1016/j.ijforecast.2011.02.006.

[16] A. Immer, C. Schultheiss, J. E. Vogt, B. Schölkopf, P. Bühlmann, and A. Marx, “On the identifiability and estimation of causal location-scale noise models,” in Proceedings of the 40th International Conference on Machine Learning, Proceedings of Machine Learning Research, vol. 202, pp. 14316–14332. PMLR, 2023.

[17] Y. Chikahara, “Moment matters: Mean and variance causal graph discovery from heteroscedastic observational data,” in Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, pp. 640–651. ACM, 2026. doi: 10.1145/3770855.3817708.

[18] D. Batenkov and Y. Yomdin, “On the accuracy of solving confluent Prony systems,” SIAM Journal on Applied Mathematics, vol. 73, no. 1, pp. 134–154, 2013. doi: 10.1137/110836584.

[19] D. Batenkov, G. Goldman, and Y. Yomdin, “Super-resolution of near-colliding point sources,” Information and Inference: A Journal of the IMA, vol. 10, no. 2, pp. 515–572, 2021. doi: 10.1093/imaiai/iaaa005. Erratum: vol. 10, no. 2, p. 721, 2021; doi: 10.1093/imaiai/iaaa015.

[20] A. C. Harvey, E. Ruiz, and N. Shephard, “Multivariate stochastic variance models,” The Review of Economic Studies, vol. 61, no. 2, pp. 247–264, 1994. doi: 10.2307/2297980.

[21] E. Ruiz, “Quasi-maximum likelihood estimation of stochastic volatility models,” Journal of Econometrics, vol. 63, no. 1, pp. 289–306, 1994. doi: 10.1016/0304-4076(93)01569-8.

[22] Y. Hua and T. K. Sarkar, “Matrix pencil method for estimating parameters of exponentially damped/undamped sinusoids in noise,” IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 38, no. 5, pp. 814–824, 1990. doi: 10.1109/29.56027.

[23] T. K. Sarkar and O. Pereira, “Using the matrix pencil method to estimate the parameters of a sum of complex exponentials,” IEEE Antennas and Propagation Magazine, vol. 37, no. 1, pp. 48–55, 1995. doi: 10.1109/74.370583.

[24] M. Nauta, D. Bucur, and C. Seifert, “Causal discovery with attention-based convolutional neural networks,” Machine Learning and Knowledge Extraction, vol. 1, no. 1, pp. 312–340, 2019. doi: 10.3390/make1010019.

[25] S. Kleinberg, Causality, Probability, and Time. Cambridge: Cambridge University Press, 2012. doi: 10.1017/CBO9781139207799.

[26] S. M. Smith, K. L. Miller, G. Salimi-Khorshidi, M. Webster, C. F. Beckmann, T. E. Nichols, J. D. Ramsey, and M. W. Woolrich, “Network modelling methods for fMRI,” NeuroImage, vol. 54, no. 2, pp. 875–891, 2011. doi: 10.1016/j.neuroimage.2010.08.063.

[27] S. Bennett, M. Cucuringu, and G. Reinert, “Lead–lag detection and network clustering for multivariate time

series with an application to the US equity market,” Machine Learning, vol. 111, pp. 4497–4538, 2022. https://doi.org/10.1007/s10994-022-06250-4.

[28] A. Reisach, C. Seiler, and S. Weichwald, “Beware of the simulated DAG! Causal discovery benchmarks may be easy to game,” in Advances in Neural Information Processing Systems, vol. 34, pp. 27772–27784, 2021.

[29] C. Lohse and J. Wahl, “Sortability of time series data,” Transactions on Machine Learning Research, 2025. https://openreview.net/forum?id=OGvmCpcHdV.

[30] T. Pantelidis and N. Pittis, “Testing for Granger causality in variance in the presence of causality in mean,” Economics Letters, vol. 85, no. 2, pp. 201–207, 2004. doi: 10.1016/j.econlet.2004.04.006.

[31] X. Jiang, L.-H. Lim, Y. Yao, and Y. Ye, “Statistical ranking and combinatorial Hodge theory,” Mathematical Programming, vol. 127, no. 1, pp. 203–244, 2011. doi: 10.1007/s10107-010-0419-x.

[32] A. K. Seth, P. Chorley, and L. C. Barnett, “Granger causality analysis of fMRI BOLD signals is invariant to hemodynamic convolution but not downsampling,” NeuroImage, vol. 65, pp. 540–555, 2013. doi: 10.1016/j.neuroimage.2012.09.049.

[33] L. Novelli, L. Barnett, A. K. Seth, and A. Razi, “Minimum-phase property of the hemodynamic response function, and implications for Granger causality in fMRI,” Human Brain Mapping, vol. 46, no. 10, Art. no. e70285, 2025. doi: 10.1002/hbm.70285.

[34] A. Borovykh, S. Bohte, and C. W. Oosterlee, “Dilated convolutional neural networks for time series forecasting,” Journal of Computational Finance, vol. 22, no. 4, pp. 73–101, 2019. doi: 10.21314/JCF.2019.358.

[35] K. J. Friston, A. Mechelli, R. Turner, and C. J. Price, “Nonlinear responses in fMRI: The Balloon model, Volterra kernels, and other hemodynamics,” NeuroImage, vol. 12, no. 4, pp. 466–477, 2000. doi: 10.1006/nimg.2000.0630.

[36] Q. Ji, E. Bouri, C. K. M. Lau, and D. Roubaud, “Dynamic connectedness and integration in cryptocurrency markets,” International Review of Financial Analysis, vol. 63, pp. 257–272, 2019. doi: 10.1016/j.irfa.2018.12.002.

[37] Z. Li, Y. Wang, and Z. Huang, “Risk connectedness heterogeneity in the cryptocurrency markets,” Frontiers in Physics, vol. 8, Art. no. 243, 2020. doi: 10.3389/fphy.2020.00243.

[38] I. Yousaf and S. Ali, “Discovering interlinkages between major cryptocurrencies using high-frequency data: New evidence from COVID-19 pandemic,” Financial Innovation, vol. 6, Art. no. 45, 2020. doi: 10.1186/s40854-020-00213-1.