# Q-PhotoMarket: A Design Space Exploration Framework for Photonic Hybrid Quantum Neural Networks in Financial Market Prediction

Alberto Marchisio<sup>§†‡∗</sup>, Hanzalah Mohamed Siraj<sup>§†</sup>, Muhammad Kashif<sup>§†</sup>, Nouhaila Innan<sup>§†</sup>, Muhammad Shafique<sup>§†</sup>

eBrain Lab, Division of Engineering, New York University Abu Dhabi, PO Box 129188, Abu Dhabi, UAE

<sup>†</sup> Center for Quantum and Topological Systems, NYUAD Research Institute, New York University Abu Dhabi, UAE <sup>‡</sup> SDU Microelectronics, Institute of Mechanical and Electrical Engineering, University of Southern Denmark, Odense, Denmark Emails: chisio@sdu.dk, {sm12152, muhammadkashif, nouhaila.innan, muhammad.shafique}@nyu.edu

Abstract—Photonic quantum computing has recently emerged as a promising platform for hybrid quantum machine learning due to its native realization of linear-optical circuits and the computational complexity of boson sampling. However, despite growing interest in quantum methods for finance, the influence of photonic circuit design choices on predictive performance remains largely unexplored. Existing studies typically evaluate a single architecture, leaving the broader photonic design space unexamined. In this work, we present Q-PhotoMarket, a systematic design space exploration (DSE) framework for photonic hybrid quantum neural networks (HQNNs) applied to financial market prediction. We explore over 5,000 valid photonic configurations spanning input photon states, circuit architectures, entangling models, and measurement strategies across their compatible computation spaces, for U.S., Indian, and cryptocurrency markets. To improve search efficiency, the exhaustive exploration is complemented with Bayesian optimization. We further incorporate threshold calibration and prediction-collapse diagnostics to enable reliable evaluation under increasingly imbalanced return thresholds. Experimental results show that systematic exploration of more than 5,000 photonic HQNN configurations reveals consistent architectural patterns across financial markets, identifies robust high-performing designs, and demonstrates competitive performance relative to classical machine learning baselines.

Index Terms—Quantum Machine Learning, Photonic Quantum Computing, Design Space Exploration, Financial Market Prediction

## I. INTRODUCTION

Quantum Machine Learning (QML) has emerged as a promising direction for extending machine learning with quantum computational primitives [1], [2]. Among the various QML paradigms, Hybrid Quantum Neural Networks (HQNNs) have attracted considerable attention due to their compatibility with near-term quantum hardware and their ability to combine quantum representations with classical optimization [3]. While HQNNs have demonstrated encouraging results across a growing range of applications, understanding how quantum architectural choices influence learning performance remains an important open problem.

Recent advances in photonic quantum computing have established linear-optical processors as a promising alternative to conventional gate-based quantum hardware. Compared with many qubit-based platforms, photonic systems offer practical advantages such as room-temperature operation, relatively low susceptibility to decoherence, and compatibility with existing optical communication technologies [4], [5]. Coupled with increasingly mature software ecosystems, these advances have made photonic HQNNs more accessible for developing and evaluating practical quantum machine learning models [6], [7]. Unlike gate-based implementations, however, photonic HQNNs expose a rich architectural design space involving multiple interacting design decisions [8], [9]. Despite this flexibility, existing studies typically adopt a single manually designed architecture, leaving the impact of these design choices largely unexplored. This gap is particularly important given the demonstrated sensitivity of HQNN performance to circuit design in gate-based models [10] and the computational potential of linear-optical circuits based on boson sampling [11].

![](images/906e7587d12d78af81c1da7bc2f0ea98806e37308add04aec8cec3b76cf6b8c6.jpg)  
Fig. 1: Distribution of AUC across valid photonic HQNN configurations for the S&P 500, NIFTY 50, and cryptocurrency markets.

To illustrate the practical importance of this design problem, Figure 1 shows the distribution of AUC across the explored photonic HQNN configurations for the three considered markets. Although the median performance remains close to the randomranking reference, the observed distributions are broad, and the best configurations achieve notably higher AUC. This spread indicates that predictive performance is highly sensitive to photonic design choices and that manually selecting a single architecture may lead to misleading conclusions about the capability of a photonic HQNN.

Financial market prediction provides a realistic and challenging benchmark for studying this problem. The task is characterized by weak predictive signals, non-stationary dynamics, and severe class imbalance, while state-of-the-art classical approaches generally achieve only modest improvements under rigorous walk-forward evaluation [12]. Existing QML studies for finance typically employ fixed quantum circuit architectures with limited architectural analysis [13], [14], making it difficult to understand how photonic design choices influence predictive performance.

To address this gap, we propose Q-PhotoMarket, a systematic design space exploration (DSE) framework for photonic HQNNs. Q-PhotoMarket explores multiple photonic design dimensions, including input photon states, circuit architectures, entangling models, measurement strategies, and computation spaces, while combining exhaustive exploration with Bayesian optimization to efficiently identify promising regions of the design space.

The main contributions of this work are:

• We propose Q-PhotoMarket, a systematic DSE framework for photonic HQNNs, enabling the analysis of key photonic design dimensions for real-world QML applications.

• We perform a large-scale empirical evaluation of the photonic design space by exploring thousands of valid HQNN configurations across multiple financial markets and return prediction thresholds.

• We integrate exhaustive exploration with Bayesian optimization to efficiently identify promising photonic architectures while reducing the search cost of high-dimensional design spaces.

• We provide empirical insights into the impact of photonic architectural choices on HQNN performance and establish a reproducible benchmark to support future research in photonic QML.

## II. BACKGROUND AND RELATED WORK

## A. Photonic HQNNs

HQNNs integrate parameterized quantum circuits with classical neural networks, combining quantum feature transformations with classical optimization. In photonic HQNNs, information is encoded into bosonic modes and processed through trainable linear-optical circuits composed of beam splitters, phase shifters, and interferometers. Universal interferometer decompositions, such as the Reck and Clements architectures, enable the realization of arbitrary unitary transformations and constitute the foundation of many photonic quantum learning models [15]. Recent software frameworks, including Perceval and Merlin, have further facilitated the development of photonic QML models by providing differentiable simulation environments for training and evaluation [6], [7].

## B. Design Space Exploration in HQNNs

The predictive performance of HQNNs is strongly influenced by architectural design choices, including data encoding, circuit topology, variational ansatze, measurement strategies,¨ and optimization protocols. Consequently, recent studies have explored systematic design space exploration (DSE) for gatebased HQNNs to identify architectures that balance expressivity, trainability, and hardware efficiency [10], [16], [17]. In contrast, comparable studies for photonic HQNNs remain scarce, despite the considerably richer design space introduced by photon-state preparation, linear-optical circuit topologies, and measurement schemes [8]. Existing photonic QML studies typically evaluate a single manually designed architecture, providing limited understanding of how individual design choices influence learning performance. This gap motivates the systematic photonic DSE framework proposed in this work.

## III. Q-PHOTOMARKET FRAMEWORK

Q-PhotoMarket systematically explores the design space of photonic hybrid quantum neural networks (HQNNs) for financial market prediction across three asset classes: U.S. equities (S&P 500), Indian equities (NIFTY 50), and cryptocurrencies. The design space comprises four independent dimensions: input photon state, photonic circuit architecture, entangling model, and measurement strategy. The corresponding computation space is determined by the selected measurement strategy. As illustrated in Figure 2, the framework consists of four main stages: (i) leakage-safe dataset construction, (ii) photonic HQNN design space exploration, (iii) unified experimental evaluation, and (iv) Bayesian optimization for refining promising configurations.

## A. Problem Formulation

We formulate financial market prediction as a binary classification task that predicts the direction of the next-day asset return. Let $r _ { t  t + h }$ denote the forward return of an asset over prediction horizon h from time t. The binary label is defined as

$$
y _ { t } = 1 ( r _ { t  t + h } > \tau ) ,\tag{1}
$$

where 1(·) is the indicator function and τ is a user-defined return threshold. Setting $\tau = 0$ corresponds to conventional up/down direction prediction, whereas larger thresholds $( \tau \in$ {0.5%, 1.0%}) require economically more significant price movements, resulting in progressively imbalanced classification tasks. Throughout this work, we evaluate three thresholds, $\tau \in \{ 0 , 0 . 5 \% , 1 . 0 \% \}$ , with a one-day prediction horizon $( h = 1 )$ To ensure realistic evaluation on non-stationary financial time series, all experiments employ an expanding-window walkforward protocol where each validation window strictly follows its corresponding training window to prevent look-ahead bias.

## B. Dataset Construction

## 1) Data Acquisition and Feature Engineering

Daily OHLCV (Open, High, Low, Close, Volume) data for each asset class are collected from Yahoo Finance for the period 2018-2025. For each asset, we construct a leakage-safe feature set using only information available at time t. The resulting dataset comprises 23 engineered features covering return and momentum, volatility and risk, technical indicators, liquidity and microstructure, and price structure:

• Return and momentum: ret\_1, ret\_5, ret\_10, ret\_20, cross\_momentum

• Volatility and risk: vol\_5, vol\_10, vol\_20, vol\_ratio\_5\_20, ret\_autocorr\_20

• Technical indicators: rsi\_14, sma20\_dist, bb\_width\_20, vwap\_deviation

• Liquidity and microstructure: volume\_ratio\_20, amihud\_illiquidity, obv\_slope\_10

• Price structure and event: high\_low\_ratio, overnight\_gap, day\_of\_week, vix\_bucket, month\_end\_flag, earnings\_event\_flag

Macro-contextual features are incorporated through a 5-day rolling index return (index\_ret\_5d), a discretized VIX volatility regime proxy (vix\_bucket), and a binary earningsevent indicator (earnings\_event\_flag). Samples with missing values are removed before label construction.

To satisfy the photonic mode constraints imposed by the Merlin simulation framework, the HQNN operates on a fixed subset of 16 input features throughout all experiments. The selected features correspond exclusively to price-derived market descriptors, including return, momentum, volatility, technical, liquidity, and price-structure indicators, while external contextual signals (e.g., market-index and event-related features) are excluded.

## 2) Label Construction

Future return labels are generated only after feature engineering to eliminate look-ahead leakage. For each threshold τ, labels are computed independently using the forward return $r _ { t  t + h } =$ $P _ { t + h } / P _ { t } ~ - ~ 1$ . As τ increases, the positive class prevalence decreases from approximately 50% $( \tau = 0 ) \ t o 2 0 { - } 3 4 \% ( \tau = 1 \% ) .$ inducing increasing label imbalance across all three markets.

## 3) Walk-Forward Cross-Validation

To respect the temporal ordering of financial data and prevent look-ahead bias, we construct time-consistent evaluation folds using an expanding-window walk-forward scheme. Training windows grow monotonically, while each validation window is strictly subsequent to its corresponding training window $( n _ { \mathrm { f o l d s } } = 5 ,$ minimum training fraction = 0.5). This protocol provides reproducible evaluation across all dataset-threshold combinations.

![](images/78f8a1f36cf857d8be22de3c26a40946c27c885b470dc4203f5ef7c20b6384ca.jpg)  
Fig. 2: Overview of the proposed Q-PhotoMarket framework.

![](images/5c2cd868f6d32ba5d0b1819196f41f1cb30dd94b66dff30fa9dbe719b5c1a0a2.jpg)  
Fig. 3: Two-dimensional t-SNE visualization of the engineered feature spaces for the S&P 500, NIFTY 50, and cryptocurrency datasets.

## 4) Dataset and Cross-Market Feature-Space Analysis

The proposed framework is evaluated on three market domains: U.S. equities, Indian equities, and cryptocurrencies. Daily OHLCV data from 2018-2025 are transformed into a common leakage-safe feature representation and subsequently analyzed across markets. Since these datasets exhibit different volatility characteristics, trend persistence, and class distributions, we assess the similarity of their engineered feature spaces before performing cross-market architectural comparisons.

To this end, we employ t-distributed Stochastic Neighbor Embedding (t-SNE) [18] as a qualitative visualization technique. Feature vectors are first normalized to [0, 1], pre-reduced using PCA for embedding stability, and projected into two dimensions using a perplexity of min(30, N/4), 500 optimization iterations, and a fixed random seed.

As shown in Figure 3, the three markets occupy overlapping yet distinguishable regions in feature space. This qualitative analysis suggests that the datasets share sufficient statistical structure to support meaningful cross-market comparisons while preserving market-specific characteristics, allowing architectural differences to be interpreted with reduced influence from datasetlevel distributional effects.

TABLE I: Photonic HQNN design space explored by Q-PhotoMarket. Computation space is determined by the selected measurement strategy. Total valid configurations per (market, τ) are approximately 600 after invalid combination pruning.

<table><tr><td>Dimension</td><td>Options</td><td>Count</td></tr><tr><td>Input photon state</td><td>Alternating, Bunched, Pair-bunched, Thermal-like</td><td>4</td></tr><tr><td>Circuit architecture</td><td>Simple, Ring NN, Clements Mesh, Phase-only, Permuted Ring</td><td>5</td></tr><tr><td>Entangling model</td><td>MZI, Bell</td><td>2</td></tr><tr><td>Measurement strategy</td><td>Probs-Fock, Probs-Unbunched, Probs-Dual-Rail, Mode expectations, Mode correlations</td><td>5</td></tr><tr><td colspan="2">Valid configs per (market, τ), after pruning</td><td>≈600</td></tr><tr><td colspan="2">Total runs (×3 markets, ×3 thresholds)</td><td>≈5,400</td></tr></table>

## C. Photonic HQNN Architecture and Design Space

The proposed HQNN is implemented using the Merlin photonic simulation framework and trained end-to-end in PyTorch. Since Merlin provides a differentiable simulation backend, gradients are computed through standard backpropagation without requiring parameter-shift evaluation. The hybrid architecture couples a trainable photonic QuantumLayer with a classical MLP prediction head (LazyLinear→ReLU→Linear(1)), which maps measurement-derived photonic features to the final prediction logit. Throughout the design space exploration, the number of photonic modes is fixed as $n _ { \mathrm { m o d e s } } = n _ { \mathrm { f e a t u r e s } } + 1$ with $n _ { \mathrm { f e a t u r e s } } = 1 6 .$ . Q-PhotoMarket explores four configurable dimensions of the photonic HQNN architecture: input photon state, circuit architecture, entangling model, and measurement strategy. The resulting design space is summarized in Table I.

## 1) Input Photon State

The input state determines the initial photon occupation used to encode classical features into the photonic Hilbert space. We consider four configurations:

• Alternating: photons distributed across alternating modes.

• Bunched: photons initialized in adjacent low-index modes.

• Pair-bunched: photons injected in paired modes.

• Thermal-like: photon occupations sampled from a thermal distribution.

## 2) Photonic Circuit Architecture

Five linear-optical circuit topologies are investigated:

• Simple: single-layer beamsplitter mesh.

• Ring NN: circular beamsplitter topology.

• Clements Mesh: universal rectangular interferometer [15], providing maximal expressivity.

• Phase-only: phase-shifter network without beamsplitter entanglement.

• Permuted Ring: ring topology with randomized mode permutations.

## 3) Entangling Model

Two entangling primitives are evaluated:

• MZI: Mach-Zehnder Interferometer, the canonical two-mode entangling unit.

• Bell: Bell-state-inspired entangling structure, generating stronger inter-mode correlations.

## 4) Measurement Strategy

Five measurement strategies are explored to extract classical representations from the output photonic state. Invalid combinations (e.g., non-binary input states paired with probs\_unbunched or probs\_dual\_rail) are discarded during the search. The associated computation space is determined by the selected measurement strategy.

• Fock probabilities (probs\_fock): output occupation probabilities in the Fock basis; restricted to $n _ { \mathrm { m o d e s } } ~ \leq ~ 4$ because of memory requirements.

• Unbunched probabilities (probs\_unbunched): distinguishable-photon probabilities in the unbunched computation space.

• Dual-rail probabilities (probs\_dual\_rail): probabilities obtained under dual-rail logical encoding.

• Mode expectations (mode\_expectations): first-order expectations ⟨nˆ<sub>i</sub>⟩.

• Mode correlations (mode\_correlations): second-order correlations $\langle \hat { n } _ { i } \hat { n } _ { j } \rangle$

## D. Experimental Protocol

## 1) HQNN Model Training

All HQNN configurations are trained under a unified protocol. We employ the Adam optimizer with learning rate $\eta = 3 \times 1 0 ^ { - 4 }$ and binary cross-entropy loss (BCEWithLogitsLoss). Each model is trained for 8 epochs during Stage 1 and for 20 epochs during Stage 2, using a batch size of 128 and 100 measurement shots per circuit execution with a random seed fixed to 42. An optional cosine annealing scheduler and early stopping based on validation AUC (patience = 5 epochs) are supported. Gradients are propagated through Merlin’s differentiable simulation backend using automatic differentiation [7].

## 2) Class Imbalance Handling and Threshold Calibration

Increasing the return threshold τ substantially reduces the positive-class prevalence, producing increasingly imbalanced classification tasks. For $\tau > 0 .$ , class imbalance is mitigated through inverse-frequency class weighting,

$$
w _ { + } = \frac { N _ { - } } { \operatorname* { m a x } ( N _ { + } , 1 ) } ,\tag{2}
$$

where $N _ { + }$ and N<sub>−</sub> denote the numbers of positive and negative training samples, respectively. Empirically, S&P 500 and NIFTY 50 achieve the best performance using class-weighting=balanced, whereas cryptocurrency datasets benefit from class-weighting=none.

To avoid suboptimal fixed decision boundaries, the classification threshold is calibrated on the validation fold by maximizing a selected objective using Eq. 3, where Obj ∈ {MCC, F1, BalAcc}.

$$
\hat { \tau } ^ { * } = \arg \operatorname* { m a x } _ { \tau \in [ 0 , 1 ] } \mathrm { O b j } ( y _ { \mathrm { v a l } } , \mathbf { 1 } ( \hat { p } _ { \mathrm { v a l } } \geq \tau ) ) ,\tag{3}
$$

## 3) Prediction Collapse Diagnostics

A common failure mode was identified at $\tau > 0 .$ , where positive class prevalence drops to 20-34% and predicted probabilities remain below the conventional decision threshold of 0.5, causing all samples to be assigned to the majority class 0. Although ranking performance (AUC) may remain meaningful, threshold-dependent metrics such as F1-score and MCC collapse to zero,

![](images/04acca773d6f0de4af33e31e00b6ea8d739e3187a784e27abeb99c1c34f38e2e.jpg)

![](images/c3a454eaa47281c02fa35b369f12359ccb1166e7446bf3d45f54de4bd18b1abb.jpg)  
Fig. 4: Prediction-collapse diagnostics. Left: calibrated decision thresholds versus positive prediction rate (PPR). Right: fraction of configurations exhibiting prediction collapse under different return thresholds.

$$
\mathrm { F 1 } = 0 , \quad \mathrm { M C C } = 0 , \quad \mathrm { B a l A c c } = 0 . 5 ,\tag{4}
$$

To identify this behavior, each experiment records the calibrated decision threshold, its optimization objective, the positive prediction rate (PPR), and a prediction-collapse warning when PPR ∈/ [0.02, 0.98].

Figure 4 illustrates the effectiveness of this diagnostic procedure. The left panel shows that calibrated thresholds successfully recover meaningful positive prediction rates across a wide range of operating points, while the right panel quantifies how the frequency of prediction collapse increases as the return threshold becomes more stringent. These diagnostics provide an additional validation of the proposed calibration strategy before comparing HQNN architectures.

## E. Bayesian Optimization

To complement the exhaustive grid search, we employ Bayesian optimization (BO) using the Tree-structured Parzen Estimator (TPE) sampler implemented in Optuna [19]. BO explores the same photonic design space while efficiently concentrating the search on promising regions identified during earlier trials.

## 1) Stage 1: TPE-Guided Search

The TPE sampler models the distribution of high-performing configurations and proposes new candidates by jointly optimizing the photonic design parameters, the number of input features, and the circuit shot count. Candidate configurations are ranked using the composite objective

$$
\mathcal { L } = w _ { \mathrm { A U C } } \cdot \mathrm { A U C } + w _ { \mathrm { F 1 } } \cdot \mathrm { F 1 } + w _ { \mathrm { G P S } } \cdot \mathrm { G P S } - \delta _ { \mathrm { c o l l a p s e } } ,\tag{5}
$$

where $w _ { \mathrm { A U C } } = 0 . 6 , w _ { \mathrm { F 1 } } = 0 . 3 , w _ { \mathrm { G P S } } = 0 . 1$ , and $\delta _ { \mathrm { c o l l a p s e } } =$ 0.15 is a penalty applied when PPR ∈/ [0.05, 0.95]. A resource guardrail skips configurations whose heuristic cost index, defined as $( n _ { \mathrm { m o d e s } } ^ { 3 } \times m _ { \mathrm { f a c t o r } } \times a _ { \mathrm { f a c t o r } } \times \mathrm { s h o t s } / 1 0 0 )$ , exceeds a threshold, preventing memory or runtime instability for high-mode, highshot combinations. A median pruner with $ { t _ { \mathrm { w a r m u p } } } = 5$ epochs additionally terminates unpromising trials early.

## 2) Stage 2: Local Refinement and Warm-Starting

After Stage 1, the top-k discovered configurations are used to seed a local dense refinement pass. In subsequent studies, prior sweep results are enqueued directly into the optimization process, allowing the sampler to incorporate all prior grid and confirm-run evidence before proposing new trials. Warm-start rows are ranked by AUC and filtered to respect mode and shot constraints.

## 3) Optimization Protocol

The BO phase is performed independently for each market with separate Optuna studies, using $n _ { \mathrm { t r i a l s } } = 1 5 0$ per shard, $n _ { \mathrm { s t a r t u p } } =$

30 random trials before activating the TPE sampler, and a pertrial subprocess timeout of 300 seconds. The resulting studies are merged for downstream analysis, enabling parameter-importance estimation and convergence analysis across the explored photonic design space.

## F. Performance Evaluation

## 1) Primary Metrics

Increasing return thresholds produce progressively imbalanced classification tasks, making raw accuracy alone insufficient for performance evaluation. Accordingly, we emphasize thresholdrobust metrics including the Area Under the ROC Curve (AUC), Matthews Correlation Coefficient (MCC), and balanced accuracy, while reporting conventional accuracy only as a complementary measure. All configurations are evaluated using accuracy, F1- score, MCC, Area Under the ROC Curve (AUC), and Average Precision (AP). MCC is defined as:

$$
{ \bf M C C } = \frac { T P \cdot T N - F P \cdot F N } { \sqrt { ( T P + F P ) ( T P + F N ) ( T N + F P ) ( T N + F N ) } } ,\tag{6}
$$

and is particularly suitable for imbalanced financial classification because it accounts for all elements of the confusion matrix. Balanced accuracy is defined as $( T P R + T N R ) / 2$

## 2) Composite Metrics

Following [20], we report four General Performance Score (GPS) variants, each defined as the harmonic mean of a complementary subset of metrics:

$$
\mathrm { G P S _ { 1 } = \frac { 3 } { \frac { 1 } { B a l A c c } + \frac { 1 } { A U C } + \frac { 1 } { F 1 } } }\tag{7}
$$

$$
\mathrm { G P S } _ { 2 } = \frac { 4 } { \frac { 1 } { \mathrm { A c c } } + \frac { 1 } { \mathrm { P r e c } } + \frac { 1 } { \mathrm { R e c a l l } } + \frac { 1 } { \mathrm { F 1 } } }\tag{8}
$$

$$
\mathrm { G P S _ { 3 } = \frac { 3 } { \frac { 1 } { B a l A c c } + \frac { 1 } { A U C } + \frac { 1 } { A c c } _ { C V } } }\tag{9}
$$

$$
\mathrm { G P S _ { 4 } = \frac { 4 } { \frac { 1 } { B a l A c c } + \frac { 1 } { A U C } + \frac { 1 } { F 1 } + \frac { 1 } { A c c } { c v } } }\tag{10}
$$

where $\overline { { \mathrm { A c c } } } _ { \mathrm { C V } }$ denotes the mean walk-forward cross-validation accuracy. These composite metrics provide a unified ranking while emphasizing different performance trade-offs.

## 3) Finance-Specific Metrics

To complement conventional classification metrics, we report two finance-oriented measures. Pseudo-Sharpe approximates the risk-adjusted return of a strategy that enters a long position whenever $\hat { y } _ { t } ~ = ~ 1$ , computed as the ratio between the mean predicted-positive return and its standard deviation. Magnitudeweighted accuracy weights each correct prediction by $| r _ { t  t + h } |$ assigning greater importance to larger market movements.

## 4) Baselines and Diagnostic Outputs

For each dataset-threshold combination, we compare the proposed photonic HQNNs against three classical baselines: a majority-class classifier, logistic regression, and a shallow multilayer perceptron. These baselines provide reference points for assessing whether photonic quantum models offer meaningful improvements over classical machine learning approaches [10].

Each experiment records calibrated decision thresholds, positive prediction rates, prediction probabilities, and collapse diagnostics. Bayesian optimization studies additionally store the composite objective value, parameter importance estimates, and convergence history for subsequent analysis.

![](images/38a2aedf73464f9b7e121fade0f821fe0e335daeeb18fec92af4f32ae25b4c24.jpg)  
Fig. 5: Mean AUC obtained across the explored photonic HQNN design space for different combinations of circuit architectures and measurement strategies $\mathrm { a t } \tau = 1 \%$

## IV. RESULTS AND DISCUSSION

## A. Overall Design Space Exploration

We first evaluate the complete photonic design space using the exhaustive screening protocol comprising approximately 1,080 configurations per market. For each market, approximately 600 candidate architectures are subsequently re-evaluated using multifold validation to confirm robustness. Throughout this section, we emphasize AUC, MCC, and balanced accuracy because increasing return thresholds introduce progressively stronger class imbalance.

Figure 5 summarizes the mean AUC obtained for different combinations of photonic circuit architectures and measurement strategies under the most challenging return threshold $( \tau = 1 \% )$ Several consistent trends emerge across all three markets. Architectures based on the Clements Mesh and Simple interferometer generally achieve higher mean AUC than the Phase-only and Permuted Ring variants, while probabilitybased measurements in the Fock basis frequently outperform expectation- and correlation-based alternatives. Although the optimal configuration remains market-dependent, these results demonstrate that architectural decisions have a measurable and systematic impact on HQNN performance.

## B. Influence of Photonic Design Choices

Figures 6-8 illustrate the impact of the explored photonic design dimensions on HQNN performance across the three financial markets. The results reveal consistent performance differences arising from input photon states, circuit architectures, entangling models, and measurement strategies.

Input photon state. The choice of input photon state has a relatively modest influence on predictive performance compared with the remaining architectural dimensions. Across all three markets, the four investigated input states exhibit similar GPS and AUC distributions, suggesting that photonic circuit design and measurement strategy play a more dominant role than the initial photon occupation.

Circuit architecture. Among the investigated architectures, the Clements Mesh consistently achieves the strongest average performance across markets, particularly for the S&P 500 and cryptocurrency datasets. Ring-based architectures also demonstrate competitive behavior, whereas the Phase-only architecture generally produces the weakest results. These observations indicate that expressive interferometer topologies provide a measurable advantage for financial prediction tasks.

Entangling model. The choice of entangling primitive has a comparatively smaller impact than circuit topology. Across all three markets, the performance distributions of the Bellinspired and Mach-Zehnder Interferometer (MZI) entangling models overlap substantially, although MZI exhibits a slight advantage on the NIFTY 50 dataset. Overall, these results suggest that circuit architecture contributes more strongly to predictive performance than the specific entangling primitive.

![](images/df6f3464d3ffa6e24c3c23b78b74a7e68e1cd55a65b6c20307074d2b58a8bf25.jpg)  
Fig. 7: NIFTY 50: 8-panel DSE diagnostics from the screening grid.

Measurement strategy. Measurement strategy emerges as one of the most influential design dimensions. While probabilitybased Fock measurements provide strong performance on the S&P 500 and NIFTY 50 datasets, correlation-based measurements become increasingly competitive on the crypto dataset. This behavior indicates that the optimal measurement representation is application-dependent and should be treated as an important hyperparameter during photonic HQNN design.

## C. Cross-Market and Threshold Sensitivity

Having established the influence of individual photonic design dimensions, we next examine whether these trends remain consistent across financial markets and prediction settings.

Figure 9 shows the mean and best AUC obtained across all evaluated configurations under increasing return thresholds. Across the three markets, both the average and best-performing photonic HQNNs exhibit improved ranking performance as the return threshold increases. The largest improvement is observed for the S&P 500, where the mean AUC increases from 0.500 at τ = 0 to 0.528 at τ = 1%, followed by NIFTY 50 (0.499→0.517) and cryptocurrencies (0.500→0.510). Although higher thresholds produce increasingly imbalanced datasets, they also define more discriminative prediction tasks by focusing on economically significant price movements, enabling improved class separability.

## D. Configuration Selection and Bayesian Refinement

Following the exhaustive screening stage, the highestperforming configurations were re-evaluated using the staged multi-fold protocol described in Section III. Figure 10 compares the fold-wise AUC distributions obtained during Phase 1 and Phase 2. While the variance across validation folds remains comparable, the longer Phase 2 training consistently improves the median AUC, indicating that the shortlisted architectures maintain their performance under more rigorous evaluation.

The highest-performing configurations obtained after Phase 2 are summarized for each market. For the S&P 500, the bestperforming architecture combines the Simple circuit with Bell entangling and Fock probability measurements, achieving a mean AUC of 0.528 (±0.019). For NIFTY 50, the strongest configuration employs a Clements Mesh architecture with MZI entangling and Fock measurements, reaching a mean AUC of 0.514 (±0.028).

Tables II-IV summarize representative high-performing configurations (top-10 per market) identified during the exhaustive search, revealing recurring architectural patterns across multiple evaluation metrics. In particular, Clements Mesh, Ring NN, and Simple architectures appear repeatedly among the highest-ranked configurations, confirming the trends observed during the design space exploration.

![](images/3ca9ef1c825563a48cf867577aa2cb3422582d97596460082a01c48b2ee3e47f.jpg)  
Fig. 8: Crypto: 8-panel DSE diagnostics from the screening grid.

TABLE II: Top-10 overlap table for S&P 500. Top10Count indicates in how many of 8 metrics a configuration appears in the top-10.
<table><tr><td>R T</td><td></td><td>Arch</td><td>Ent</td><td>Meas</td><td>State</td><td>Top10Count MetricsHit</td><td></td><td>BestAUC</td><td>BestMCC</td><td>BestBalAcc</td></tr><tr><td>1</td><td>0.0100</td><td>simple</td><td>bell</td><td>probs_fock</td><td>alternating</td><td>4/8</td><td>AUC, MCC, BalAcc, GPS</td><td>0.573</td><td>0.000</td><td>0.500</td></tr><tr><td>2</td><td>0.0100</td><td>clements_mesh</td><td>mzi</td><td>mode_correlations</td><td>pair_bunched</td><td>4/8</td><td>MCC, BalAcc, MWA, PSharpe</td><td>0.557</td><td>0.008</td><td>0.501</td></tr><tr><td>3</td><td>0.0000</td><td>simple</td><td>mzi</td><td>mode_expectations</td><td>thermal_like</td><td>4/8</td><td>MCC, F1, BalAcc, AP</td><td>0.519</td><td>0.022</td><td>0.500</td></tr><tr><td>4</td><td>0.0000</td><td>simple</td><td>mzi</td><td>mode_expectations</td><td>bunched</td><td>3/8</td><td>MCC, F1, BalAcc</td><td>0.483</td><td>0.022</td><td>0.500</td></tr><tr><td>5</td><td>0.0000</td><td>ring_nn</td><td>mzi</td><td>mode_correlations</td><td>pair_bunched</td><td>3/8</td><td>MCC, F1, BalAcc</td><td>0.479</td><td>0.022</td><td>0.501</td></tr><tr><td>6</td><td>0.0100</td><td>clements_mesh</td><td>mzi</td><td>mode_expectations</td><td>alternating</td><td>2/8</td><td>AUC, GPS</td><td>0.576</td><td>0.000</td><td>0.500</td></tr><tr><td>7</td><td>0.0100</td><td>ring_nn</td><td>mzi</td><td>probs_fock</td><td>pair_bunched</td><td>2/8</td><td>AUC, GPS</td><td>0.573</td><td>0.000</td><td>0.500</td></tr><tr><td>8</td><td>0.0100 simple</td><td></td><td>bell</td><td>probs_fock</td><td>pair_bunched</td><td>2/8</td><td>AUC, GPS</td><td>0.572</td><td>0.000</td><td>0.500</td></tr><tr><td>9</td><td></td><td>0.0100 clements_mesh</td><td>mzi</td><td>mode_expectations</td><td>bunched</td><td>2/8</td><td>AUC, GPS</td><td>0.571</td><td>0.000</td><td>0.500</td></tr><tr><td></td><td></td><td>10 0.0100 clements_mesh</td><td>mzi</td><td>mode_correlations</td><td>bunched</td><td>2/8</td><td>AUC, GPS</td><td>0.570</td><td>0.000</td><td>0.500</td></tr></table>

TABLE III: Top-10 overlap table for NIFTY 50. Top10Count indicates in how many of 8 metrics a configuration appears in the top-10.
<table><tr><td rowspan="2">R τ</td><td rowspan="2"></td><td rowspan="2">Arch</td><td rowspan="2">Ent Meas</td><td rowspan="2"></td><td rowspan="2">State</td><td rowspan="2">Top10Count MetricsHit</td><td rowspan="2"></td><td rowspan="2">BestAUC</td><td rowspan="2">BestMCC</td><td rowspan="2">BestBalAcc</td></tr><tr><td></td></tr><tr><td>1</td><td>0.0000</td><td>ring_nn</td><td></td><td>bell mode_expectations pair_bunched</td><td></td><td>5/8</td><td>MCC, BalAcc, AP, GPS, PSharpe</td><td>0.511</td><td>0.021</td><td>0.507</td></tr><tr><td>2</td><td>0.0000</td><td>ring_nn</td><td></td><td>mzi mode_expectations</td><td>pair_bunched</td><td>4/8</td><td>MCC, BalAcc, GPS, PSharpe</td><td>0.516</td><td>0.048</td><td>0.522</td></tr><tr><td>3</td><td>0.0000</td><td> ring_nn</td><td></td><td>bell mode_expectations</td><td>bunched</td><td>4/8</td><td>MCC, BalAcc, GPS, PSharpe</td><td>0.512</td><td>0.027</td><td>0.511</td></tr><tr><td>4</td><td>0.0000</td><td>clements_mesh</td><td>bell</td><td>mode_correlations</td><td>thermal_like</td><td>3/8</td><td>AP, GPS, PSharpe</td><td>0.520</td><td>0.003</td><td>0.500</td></tr><tr><td>5</td><td></td><td>0.0000 clements_mesh</td><td>bell</td><td>mode_correlations</td><td>pair_bunched</td><td>3/8</td><td>AP, GPS, PSharpe</td><td>0.516</td><td>0.002</td><td>0.500</td></tr><tr><td>6</td><td>0.0000 ring_nn</td><td></td><td>mzi probs_fock</td><td></td><td>pair_bunched</td><td>3/8</td><td>F1, AP, GPS</td><td>0.514</td><td>0.000</td><td>0.500</td></tr><tr><td>7</td><td>0.0000</td><td>) ring_nn</td><td></td><td>mzi mode_correlations</td><td>bunched</td><td>3/8</td><td>MCC, BalAcc, AP</td><td>0.512</td><td>0.051</td><td>0.511</td></tr><tr><td>8</td><td>0.0000</td><td>permuted_ring</td><td></td><td>bell mode_correlations</td><td>bunched</td><td>3/8</td><td>MCC, F1, PSharpe</td><td>0.500</td><td>0.024</td><td>0.501</td></tr><tr><td>9</td><td>0.0000</td><td>permuted_ring</td><td></td><td>bell mode_expectations alternating</td><td></td><td>2/8</td><td>F1, GPS</td><td>0.521</td><td>0.000</td><td>0.500</td></tr><tr><td></td><td>10 0.0000 ring_nn</td><td></td><td></td><td>bell probs_fock</td><td>pair_bunched</td><td>2/8</td><td>AP, GPS</td><td>0.514</td><td>0.000</td><td>0.500</td></tr></table>

![](images/5ab803c96f3cff8af56aff73cd020fbb81a5e9a071bec38e624c591c1a6b3e50.jpg)  
Fig. 9: Threshold robustness of the explored photonic HQNN design space. Mean and best AUC are reported across all evaluated configurations, across τ ∈ {0, 0.5%, 1%}.

## E. Comparison with Classical Baselines

Figure 11 compares the best photonic HQNN identified by Q-PhotoMarket against three classical baselines: a majorityclass classifier, logistic regression, and a shallow multilayer perceptron. Across the evaluated markets and return thresholds, the photonic HQNN consistently outperforms the majority baseline and achieves competitive performance relative to conventional machine learning models.

![](images/fb2711763d63be5d7feec73a6b746f9a564ba0857336853bf0e98091eadf9f47.jpg)  
Fig. 10: Fold-variance analysis for shortlisted configurations (Phases 1 vs 2).

The performance gap becomes more apparent at higher return thresholds, where class imbalance increases and simple classifiers struggle to identify minority positive samples. Although the improvements over classical neural networks remain modest, these results demonstrate that systematic photonic architecture exploration enables HQNNs to remain competitive on realistic financial prediction tasks while providing valuable insights into photonic circuit design.

TABLE IV: Top-10 overlap table for Crypto. Top10Count indicates in how many of 8 metrics a configuration appears in the top-10.
<table><tr><td>R T</td><td></td><td>Arch</td><td>Ent Meas</td><td></td><td>State</td><td>Top10Count</td><td>MetricsHit</td><td>BestAUC</td><td>BestMCC</td><td>BestBalAcc</td></tr><tr><td>1</td><td>0.0000</td><td>clements_mesh</td><td>mzi</td><td>mode_expectations</td><td>bunched</td><td>3/8</td><td>MCC, BalAcc, AP</td><td>0.536</td><td>0.020</td><td>0.501</td></tr><tr><td>2</td><td>0.0000</td><td>clements_mesh</td><td></td><td>mzi mode_expectations</td><td>alternating</td><td>3/8</td><td>MCC, BalAcc, AP</td><td>0.536</td><td>0.031</td><td>0.503</td></tr><tr><td>3</td><td>0.0000</td><td>permuted_ring</td><td></td><td>bell mode_expectations</td><td>thermal_like</td><td>3/8</td><td>MCC, F1, BalAcc</td><td>0.505</td><td>0.035</td><td>0.503</td></tr><tr><td>4</td><td>0.0050</td><td>clements_mesh</td><td>bell</td><td>mode_correlations</td><td>thermal_like</td><td>3/8</td><td>MCC, BalAcc, PSharpe</td><td>0.505</td><td>0.063</td><td>0.514</td></tr><tr><td>5</td><td>0.0050</td><td>clements_mesh</td><td>mzi</td><td>mode_correlations</td><td>thermal_like</td><td>3/8</td><td>MCC, BalAcc, PSharpe</td><td>0.503</td><td>0.042</td><td>0.509</td></tr><tr><td>6</td><td>0.0000</td><td>ring_nn</td><td></td><td>bell mode_correlations</td><td>pair_bunched</td><td>3/8</td><td>MCC, F1, BalAcc</td><td>0.500</td><td>0.031</td><td>0.503</td></tr><tr><td>7</td><td>0.0050</td><td></td><td></td><td>clements_mesh bell mode_correlations</td><td>pair_bunched</td><td>3/8</td><td>MCC, BalAcc, PSharpe</td><td>0.493</td><td>0.063</td><td>0.505</td></tr><tr><td>8</td><td>0.0100</td><td>ring_nn</td><td></td><td>mzi mode_correlations</td><td>thermal_like</td><td>2/8</td><td>AUC, GPS</td><td>0.552</td><td>0.000</td><td>0.500</td></tr><tr><td>9</td><td>0.0100</td><td>clements_mesh bell probs_fock</td><td></td><td></td><td>pair_bunched</td><td>2/8</td><td>AUC, GPS</td><td>0.550</td><td>0.000</td><td>0.500</td></tr><tr><td>10</td><td>0.0100</td><td>ring_nn</td><td></td><td>mzi probs_fock</td><td>pair_bunched</td><td>2/8</td><td>AUC, GPS</td><td>0.546</td><td>0.000</td><td>0.500</td></tr></table>

![](images/8fd5d6cd497a75f42deb44f7906ead9bc5a61ac747c2f3b81264ad098e009170.jpg)  
Fig. 11: Accuracy comparison against majority, logistic, and MLP baselines.

## F. Discussion

The primary contribution of this work is not the identification of a single best-performing photonic HQNN, but the systematic characterization of the photonic design space. Across more than 5,000 evaluated configurations, the results consistently demonstrate the influence of architectural decisions on predictive performance. In particular, circuit topology and measurement strategy contribute substantially larger performance variations than input photon states or entangling primitives, suggesting that future photonic HQNN development should prioritize architectural exploration over isolated circuit optimization.

Among the investigated architectures, the Clements Mesh and, in several cases, the Simple and Ring NN topologies repeatedly appear among the highest-performing configurations. Their recurring presence across multiple evaluation metrics and financial markets indicates that expressive interferometer structures provide a favorable balance between representational capacity and trainability. Likewise, the influence of the measurement strategy highlights that the classical representation extracted from the photonic circuit is an integral component of the learning pipeline.

An interesting observation is that increasing the return threshold generally improves AUC despite introducing more severe class imbalance. By filtering out small price fluctuations, larger thresholds produce prediction tasks with clearer decision boundaries, enabling photonic HQNNs to better separate positive and negative samples. At the same time, this setting increases the risk of prediction collapse, emphasizing the importance of calibration-aware evaluation and imbalance-sensitive metrics.

This work has several limitations. All experiments are performed using differentiable photonic simulation, and the study focuses on daily market prediction using three representative financial datasets. In addition, the explored design space is limited to the investigated circuit topologies, measurement strategies, and encoding schemes. Future work will extend Q-PhotoMarket to larger photonic architectures, hardware-aware optimization, execution on real quantum devices, and additional real-world QML applications, while investigating automated architecture search techniques that further reduce the cost of exploring highdimensional photonic design spaces.

## V. CONCLUSION

This work presents Q-PhotoMarket, a systematic design space exploration framework for photonic hybrid quantum neural networks applied to financial market prediction. By jointly exploring multiple photonic design dimensions under a unified evaluation protocol, the proposed framework provides a methodology for understanding how architectural decisions influence learning performance. Q-PhotoMarket serves not only as a benchmark for photonic QML in finance but also as a foundation for future research on automated photonic architecture optimization and application-driven design space exploration across various QML domains.

## ACKNOWLEDGMENT

This work was supported in part by the NYUAD Center for Quantum and Topological Systems (CQTS), funded by Tamkeen under the NYUAD Research Institute grant CG008.

## REFERENCES

[1] M. Cerezo et al., “Variational quantum algorithms,” Nature Reviews Physics, vol. 3, no. 9, pp. 625–644, 2021.

[2] K. Zaman et al., “A survey on quantum machine learning: Current trends, challenges, opportunities, and the road ahead,” arXiv:2310.10315, 2023.

[3] M. Cerezo et al., “Challenges and opportunities in quantum machine learning,” Nature computational science, 2022.

[4] H. Zhu et al., “Quantum computing and machine learning on an integrated photonics platform,” Information, vol. 15, no. 2, 2024.

[5] N. Maring et al., “A general-purpose single-photon-based quantum computing platform,” arXiv preprint arXiv:2306.00874, 2023.

[6] N. Heurtel et al., “Perceval: A software platform for discrete variable photonic quantum computing,” Quantum, vol. 7, p. 931, 2023.

[7] C. Notton et al., “Merlin: A discovery engine for photonic and hybrid quantum machine learning,” arXiv preprint arXiv:2602.11092, 2026.

[8] N. Maring et al., “A versatile single-photon-based quantum computing platform,” Nature Photonics, vol. 18, no. 6, pp. 603–609, 2024.

[9] F. Elnakhal et al., “Q-photonas: Hybrid quantum neural architecture search framework on photonic devices,” arXiv:2605.22097, 2026.

[10] M. Kashif and S. Al-Kuwari, “Design space exploration of hybrid quantum– classical neural networks,” Electronics, 2021.

[11] S. Aaronson and A. Arkhipov, “The computational complexity of linear optics,” in STOC, 2011, pp. 333–342.

[12] N. Ayyildiz and O. Iskenderoglu, “How effective is machine learning in stock market predictions?” Heliyon, vol. 10, no. 2, p. e24123, 2024.

[13] N. Innan et al., “Financial fraud detection using quantum graph neural networks,” Quantum Machine Intelligence, vol. 6, no. 1, p. 7, 2024.

[14] P. K. Choudhary et al., “Hqnn-fsp: A hybrid classical-quantum neural network for regression-based financial stock market prediction,” Quantum Machine Intelligence, vol. 8, no. 1, p. 55, 2026.

[15] W. R. Clements et al., “Optimal design for universal multiport interferometers,” Optica, vol. 3, no. 12, pp. 1460–1465, 2016.

[16] Y. Du et al., “Quantum circuit architecture search for variational quantum algorithms,” npj Quantum Information, 2022.

[17] Z. He, M. Deng, S. Zheng, L. Li, and H. Situ, “Training-free quantum architecture search,” in AAAI, 2024.

[18] L. Van der Maaten and G. Hinton, “Visualizing data using t-sne.” Journal of machine learning research, vol. 9, no. 11, 2008.

[19] T. Akiba et al., “Optuna: A next-generation hyperparameter optimization framework,” in KDD, 2019.

[20] I. M. de Diego et al., “General performance score for classification problems,” Applied Intelligence, 2022.