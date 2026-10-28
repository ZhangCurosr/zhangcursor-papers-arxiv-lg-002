Please cite this paper as:   
M. Wojnar, "Sample-and-Score: Surrogate-Guided Coordinated Spatial Reuse in IEEE 802.11 Networks," in IEEE Communications Letters, doi: 10.1109/LCOMM.2026.3746794.

```bib
@ARTICLE{wojnar2026sample,
author={Wojnar, Maksymilian},
journal={IEEE Communications Letters},
title={{Sample-and-Score: Surrogate-Guided Coordinated
Spatial Reuse in IEEE 802.11 Networks}},
year={2026},
doi={10.1109/LCOMM.2026.3746794}
```

# Sample-and-Score: Surrogate-Guided Coordinated Spatial Reuse in IEEE 802.11 Networks

Maksymilian Wojnar

Abstract—Coordinated spatial reuse (Co-SR) in IEEE 802.11bn increases area throughput through simultaneous transmissions, yet the combinatorial configuration space makes realtime optimization challenging. We propose a sample-and-score approach: draw random Co-SR configurations, score each with a surrogate model, and select the best. The surrogate is trained offline and deployed without online fine-tuning or sequential exploration, yet performs close to multi-armed bandit and heuristic baselines that need multiple interaction steps or extensive measurements. On unseen topologies with 2–16 access points, it recovers 52–77% of a perfect surrogate’s throughput. A perfect sample-and-score approach reaches 95% of the upper bound throughput with 8192 candidates.

Index Terms—Machine learning, multi-AP coordination, optimization, surrogate, Wi-Fi.

## I. INTRODUCTION

Wi-Fi networks are shifting from competitive channel access to collaborative coordination among access points (APs). With multi-AP coordination (MAPC), a defining feature of IEEE 802.11bn (Wi-Fi 8), a pair of APs coordinate their transmissions for improved spectral efficiency [1]. Coordinated spatial reuse (Co-SR), a particularly promising MAPC scheme, allows multiple APs to transmit simultaneously on the same channel by jointly tuning transmit power levels. This can significantly increase aggregate throughput in dense deployments where overlapping basic service sets are inevitable. However, this comes at a cost: each simultaneous transmission adds interference at other receivers, so transmit power, the served station (STA), and modulation and coding scheme (MCS) must be optimized jointly to maintain a sufficiently high signal-tointerference-and-noise ratio (SINR) at every receiver. Striking this balance is a delicate optimization problem.

Orchestrating Co-SR transmissions in real time is challenging due to the combinatorial explosion in the number of possible configurations. Centralized heuristics driven by periodic received signal strength indicator (RSSI) reports [2], [3] and convex-optimization-based methods [4] work in small topologies but become intractable as the network grows. Online learning methods such as multi-armed bandits (MABs) adapt without channel knowledge [5], [6], yet their convergence time increases with topology size. Deep reinforcement learning approaches [7], [8] select from a fixed catalog of pre-formed Co-SR groups, requiring action-space regeneration and retraining for unseen deployments. A common limitation across these methods is that they decouple or fix MCS selection; although MCS negotiation is not formally part of the Co-SR specification, it critically determines achievable rates.

![](images/b622a276ce4651976ea57c97150d448f270c023635b1a6fd39b3a59e659a0ebc.jpg)  
Fig. 1. Overview of the proposed surrogate-based Co-SR optimization pipeline: random configurations are encoded as graphs, scored by the surrogate, and the best ones are selected for execution.

In this letter, we propose a simpler alternative: draw N random Co-SR configurations, score each with a surrogate model, and select the best ones. Fig. 1 illustrates the proposed pipeline. Surrogate-based optimization is a well-established paradigm [9]. Our contribution lies in applying it to the Co-SR configuration problem with a graph-structured representation that generalizes across topologies and scales to groups of up to 16 APs, extending beyond the pair-wise scope of the current Wi-Fi 8 specification. The surrogate is trained offline and replaces simulation at test time, requiring only a single forward pass and no convergence period, unlike online methods that need hundreds of interactions to reach the target performance. Controller-side planners increasingly rely on digital-twin models to evaluate configurations before deployment [10]. Our surrogate plays a similar role as a fast, learned twin, replacing analytical planning with sampling-and scoring. The source code is publicly available.<sup>1</sup>

## II. PROBLEM FORMULATION

A Co-SR configuration specifies: (i) which APs transmit simultaneously, (ii) which STA each active AP serves, (iii) the transmission power level, and (iv) the MCS index for each active AP. The configuration space grows combinatorially with the number of APs and STAs. For a network with |A| APs, each serving up to $| S _ { a } |$ STAs associated with it, the number of possible configurations is

TABLE I  
SIMULATION PARAMETERS.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Band</td><td>5 GHz</td></tr><tr><td>PHY</td><td>IEEE 802.11be</td></tr><tr><td>Channel width</td><td>80 MHz</td></tr><tr><td>Spatial streams</td><td>1, SISO</td></tr><tr><td>Path loss model</td><td>TGax w/ Nakagami fading</td></tr><tr><td>Nakagami-m parameter</td><td>1.5</td></tr><tr><td>CCA threshold</td><td>-82 dBm</td></tr><tr><td>Frame aggregation</td><td>Enabled (A-MPDU)</td></tr><tr><td>A-MPDU sub-frame size</td><td>1500 B</td></tr><tr><td>TXOP duration</td><td>5.484 ms</td></tr><tr><td>Traffic</td><td>Downlink, full buffer</td></tr></table>

$$
\prod _ { a \in \mathcal { A } } \big ( 1 + | S _ { a } | \cdot N _ { \mathrm { M C S } } \cdot N _ { \mathrm { T P } } \big ) ,\tag{1}
$$

where $N _ { \mathrm { M C S } }$ and $N _ { \mathrm { T P } }$ are the number of MCS indices and transmission power levels, respectively, and +1 accounts for the option of an AP not transmitting. With $N _ { \mathrm { { M C S } } } ~ = ~ 1 4$ $N _ { \mathrm { T P } } = 4$ , and $| S _ { a } | = 4 ,$ , a network of just 5 APs already faces over $1 0 ^ { 1 1 }$ possible configurations, making exhaustive search infeasible at per-transmission opportunity (TXOP) timescales.

We evaluate our approach using mapc-sim<sup>2</sup> [5], a custom open-source Monte Carlo MAPC simulator, as currently no full-stack simulator (e.g., ns-3) supports Co-SR. Our simulator computes path loss according to the TGax enterprise model [11], derives the SINR from the set of concurrent transmissions, and estimates the probability of successful aggregate MAC protocol data unit (A-MPDU) sub-frame reception using a binomial model based on the selected MCS. Nakagami-m fading is introduced to model small-scale channel variation. The optimization objective is the network throughput. In line with the Wi-Fi 8 Co-SR design, we consider downlink (APto-STA) transmissions only; uplink transmissions follow DCF and are excluded from evaluation. Every Co-SR transmission is preceded by our proposed MAPC signaling: the sharing AP sends a Co-SR invite, the invited APs reply simultaneously over OFDMA, and a trigger frame confirms the agreed parameters, with the acknowledgments likewise sent over OFDMA. We account for the backoff procedure, the interframe spaces, and the signaling overhead using the frame lengths and timings of [12]. We evaluate under full-buffer traffic, the worst-case load for throughput maximization. Table I summarizes the simulation parameters.

## III. PROPOSED METHOD

We model the Wi-Fi network as an undirected graph $\mathcal { G } = ( \nu , \mathcal { E } )$ , where $\nu = \mathcal { A } \cup \mathcal { S }$ comprises APs and STAs. The edge set E contains AP–STA transmission links (one per association) and AP–AP interference links (between APs whose mutual RSSI exceeds the clear channel assessment (CCA) threshold). Each edge carries a four-attribute feature vector:

• rssi $\in \mathbb { R } .$ : RSSI, standardized as $\tilde { r } _ { s } = ( r _ { s } - \mu _ { \mathrm { R S S I } } ) / \sigma$ RSSI with µ<sub>RSSI</sub> and σ<sub>RSSI</sub> computed from the training set,

• selected ∈ {True, False, N/A}: whether the link is chosen for transmission,

$m c s \in \{ 0 , \ldots , 1 3 , \mathrm { N / A } \}$ : MCS index of the selected link,

• tx\_power ∈ {16, 13, 10, 7, N/A} dBm: transmission power level.

Categorical attributes are one-hot encoded while RSSI remains a continuous scalar, yielding a 24-dimensional edge feature vector. Edges are bidirectional. The N/A class handles inapplicable attributes (e.g., MCS on interference links). A central controller gathers these measurements, scores the candidate configurations with the surrogate, and sends the selected ones back to the APs. This centralized architecture targets managed Wi-Fi deployments, where coordination among multiple APs is typically available through a network controller.

During a Co-SR transmission, the interference at a given STA depends on its channel to every other active AP. Capturing all of these links directly would require $| { \mathcal { A } } | \times | { \mathcal { S } } |$ RSSI measurements, far more than the network gathers in normal operation. We instead retain only the AP–AP and associated AP–STA links, which are available at little cost: inter-AP RSSI is obtained by listening to beacons, and AP-to-STA RSSI is derived from acknowledgment frames through channel reciprocity. The surrogate then infers the residual crossinterference through message passing, a deliberate approximation that trades exactness for a lower signaling overhead.

The surrogate model takes a graph-encoded Co-SR configuration and predicts the data rate. The model is trained offline on simulated data and replaces costly network simulation during candidate evaluation. The neural network architecture follows the graph network framework [13] with transformer-based [14] update functions (3 GNN layers with residual connections, a 3-layer transformer, a hidden dimension of 64, 4 attention heads). Attention is masked, so that graphs within a batch are scored independently.

The output layer follows the mixture density network (MDN) framework [15] with 4 Gaussian components. MDN is preferred over a standard mean squared error (MSE) regression as the stochastic data rate is better captured by a mixture and in our experiments MSE overfits where MDN does not. Each component is parametrized by a weight $\hat { w } _ { k } = \mathrm { s o f t m a x } ( \pmb { \alpha } ) _ { k }$ where α is the vector of output logits, a mean $\hat { \mu } _ { k } .$ , and a scale $\hat { \sigma } _ { k }$ . The model is trained by minimizing the negative log-likelihood of the standardized data rate $\tilde { y } = ( y - \mu _ { y } ) / \sigma _ { y }$ under the predicted Gaussian mixture:

$$
\mathcal { L } = - \log \sum _ { k = 1 } ^ { m } w _ { k } \cdot \mathcal { N } ( \tilde { y } \mid \hat { \mu } _ { k } , \hat { \sigma } _ { k } ^ { 2 } ) .\tag{2}
$$

The model is trained with AdamW [16] (learning rate 1.5 × $1 0 ^ { - 3 }$ , weight decay 0.01, gradient clipping) under a cosine one-cycle schedule for 20,000 steps with batch size 16. On a hold-out validation set, the surrogate achieves a Spearman correlation of $\rho = 0 . 8 7$ and $R ^ { 2 } = 0 . 7 0$

The training set is generated from 11 parameterized scenario types with 2–10 APs and up to 50 STAs. For each of the 1,000 randomly sampled realizations, 50 Co-SR configurations are drawn: 40 with fully random parameters and 10 with simulator-computed ideal MCS to increase the high-rate region coverage, yielding 50,000 training examples.

![](images/4e703a39baf0838d4f3cc08c4635bb9f75739d5d558df37cb1c8f6b85f6b226e.jpg)  
Fig. 2. Throughput vs. number of candidates N in the residential scenario (2 × 2 apartments) with top-1 selection. Error bars denote 95% confidence intervals of the mean.

Given a network topology, its associations, and the collected RSSI values, the inference pipeline proceeds as follows:

1) Random sampling. Draw N random Co-SR configurations by selecting $k \sim \mathrm { U n i f o r m } ( 1 , | \mathcal { A } | )$ active APs, one STA per active AP, and random MCS and transmission power settings. Each configuration is encoded as a graph G .

2) Surrogate scoring. Batch the N graphs and compute the expected data rate for each candidate as $\begin{array} { r } { \hat { y } _ { i } = \sum _ { k } w _ { k } \hat { \mu } _ { k } } \end{array}$

3) Selection. Select configurations to apply in the network according to the chosen strategy.

The selection step introduces an additional degree of freedom that can be tailored to the optimization objective. The proposed method is therefore best understood as a scheduling aid: it supplies a pre-computed catalog of high-quality Co-SR groups that a network controller can apply according to its own policy (e.g., factoring in latency or fairness), rather than acting as a standalone scheduler. The simplest strategy, adopted in our experiments, is top-k: retain the k highestscoring configurations and apply them in subsequent TXOPs in a round-robin fashion.

By design, the method operates on a snapshot of the current network state and is therefore directly applicable to static topologies. For dynamic environments, the catalog is regenerated periodically or adaptively when a throughput drop or association change is detected.

## IV. PERFORMANCE EVALUATION

The residential scenario follows the TGax enterprise model [11]: a 2 × 2 grid of 10 m-wide apartments, each containing one AP and four randomly placed STAs, with walls providing additional attenuation. The open-space scenario places 2–16 APs at random within a 75 m×75 m area without walls, with four STAs per AP normally distributed around their associated AP (standard deviation of 5 m).

We compare the following methods:

• Surrogate (ours): draws N candidates, scores each with the trained surrogate, and selects the best ones.

![](images/86098b90a3aa59bcb81ff9d13238d1a2a9de9b8a77e5443a7f88c2c29a472574.jpg)  
Fig. 3. Co-SR throughput across open-space scenarios with 2–16 APs (top-4 selection, number of candidates N ranging from 64 to 512).

• Perfect surrogate (ours): deterministically evaluates the expected data rate of each of the N candidates and selects the best one; this is the upper bound for the sample-andscore approach.

• Random: picks one random Co-SR configuration.

• DCF: standard 802.11 distributed coordination function.<sup>3</sup>

• SR groups: the at-most-K SR-compatible group-formation heuristic [3]: APs are greedily grouped so that the estimated SINR stays above a threshold $( \gamma = 2 0 \mathrm { d B } , K = 3 ,$ fixed transmission power).

• H-MAB: hierarchical MAB [5] with three levels of agents that sequentially select active APs, recipient STAs, and transmission power levels.

• F-optimal: the fair upper bound [6], maximizing the minimum per-STA throughput, enforcing perfectly equal shares. A globally optimal reference, not a runtime scheduler.

• T-optimal: the throughput upper bound [6], maximizing total network throughput. A globally optimal reference, not a runtime scheduler.

All methods are averaged over 64 independent runs. For H-MAB, only the last 20% of the 2000-step runs are used. In all experiments, the surrogate methods jointly sample MCS with all configuration parameters, whereas DCF and H-MAB use ideal MCS. SR groups uses its own MCS heuristic, while F/T-optimal jointly optimize MCS. We retain the default MCS selection strategy for each baseline. This choice favors the baselines, as our sample-and-score approach must instead identify a suitable MCS solely from randomly sampled candidates.

Fig. 2 evaluates the residential scenario where we assess how performance scales with the number of sampled candidates N. The throughput of the perfect surrogate increases from 156 Mbit/s at N = 1 to 973 Mbit/s at $N \ = \ 8 1 9 2 .$ exceeding all baselines and reaching 95% of T-optimal. More candidates cover the Co-SR configuration space more densely, making a near-optimal choice available. The learned surrogate improves to 656 Mbit/s at N = 1024 and plateaus beyond $N \ = \ 2 0 4 8 .$ while the upper bound continues to improve. Finding increasingly rare good configurations is thus the limiting factor. Nevertheless, it exceeds DCF by 16% without any online interaction. H-MAB reaches higher average throughput but needs convergence over many steps. SR groups achieves a similar average, but its runs span 473 Mbit/s–1205 Mbit/s, which limits scheduling predictability. It also requires crosslink RSSI measurements for every AP–STA pair, which are both unreliable and expensive to collect and exchange. In contrast, our method relies only on readily available measurements and incorporates signal strength as graph features, rather than making threshold-based decisions from individual RSSI values, which are inherently noisy and error-prone.

TABLE II  
SURROGATE COMPUTATION TIME VS. H-MAB CONVERGENCE TIME [ms]. DASHES INDICATE OUT-OF-MEMORY CONDITIONS UNDER FULL-BATCH INFERENCE.
<table><tr><td rowspan="2">N</td><td colspan="2">Small (4 APs)</td><td colspan="2">Large (16 APs)</td></tr><tr><td>Full</td><td>Split</td><td>Full</td><td>Split</td></tr><tr><td>4</td><td>1.5</td><td>1.5</td><td>1.9</td><td>1.9</td></tr><tr><td>8</td><td>1.7</td><td>1.7</td><td>2.7</td><td>2.7</td></tr><tr><td>16</td><td>2.0</td><td>2.0</td><td>6.0</td><td>6.0</td></tr><tr><td>32</td><td>2.8</td><td>3.9</td><td>18.5</td><td>12.1</td></tr><tr><td>64</td><td>6.2</td><td>7.8</td><td>66.2</td><td>24.1</td></tr><tr><td>128</td><td>18.8</td><td>15.4</td><td>313.8</td><td>47.8</td></tr><tr><td>256</td><td>66.6</td><td>30.2</td><td></td><td>95.2</td></tr><tr><td>512</td><td>314.5</td><td>60.4</td><td>1</td><td>190.7</td></tr><tr><td>1024</td><td></td><td>120.7</td><td></td><td>381.1</td></tr><tr><td>H-MAB</td><td colspan="2">3780 一</td><td colspan="2">inf</td></tr></table>

We next examine how well the surrogate generalizes to new topologies of varying size. Fig. 3 evaluates unseen open-space topologies with 2–16 APs, where each point aggregates 10 independently drawn layouts, using top-4 selection and an APdependent sample budget (since the configuration space grows with network size, we scale N with the number of APs, from N = 64 candidates at 2–4 APs and doubling every two APs, up to N = 512 at 14–16 APs). The top-4 configurations form a pre-computed Co-SR catalog: when a sharing AP wins DCF contention and belongs to a stored group, the corresponding shared APs join the transmission. The reported metric covers Co-SR transmissions only. The surrogate recovers 52–77% of the perfect surrogate median rate across all topology sizes, matching most closely at 2 APs (77%) and dropping to 52–54% for 14–16 APs, while maintaining consistently low variability across scenarios. The largest topologies are out-ofdistribution, as training covers 2–10 APs.

Table II reports wall-clock inference times on a server with NVIDIA GH200 and 118 GB of RAM, for a surrogate of ∼1M parameters. We compare two batching strategies: full batch (all N graphs in one pass) and split batch (chunks of 16), where full-batch attention matrix scales quadratically with N, while split batching restores linear scaling in time. In the small scenario (4 APs), split batching scores 1024 candidates in 121 ms; in the large scenario (16 APs), the same workload takes 381 ms. Crucially, full-batch inference runs out of memory for N > 128. H-MAB requires 3780 ms to converge (i.e., reach its performance plateau) in the small scenario and fails entirely in the large one [5]. Our surrogate is over 30× faster with no online interaction. Additionally, as the surrogate produces a catalog that is reused across many subsequent TXOPs and refreshed periodically or upon a significant topology change, its inference cost is amortized. This makes it feasible to apply the method even in CPU-based deployments.

## V. CONCLUSION

We have presented a surrogate-based approach to Co-SR optimization: draw random configurations, score them, and select the best. The perfect surrogate shows that this strategy reaches 95% of T-optimal throughput by increasing catalog coverage. The surrogate model already exceeds DCF without online interaction and evaluates 1024 candidates over 30× faster than H-MAB convergence, while requiring only sparse, readily available RSSI links. Its plateau beyond 2048 candidates identifies ranking many configurations as the main limitation and the clearest direction for future work.

Several simplifications constrain the current evaluation: we do not use a full-stack simulator (e.g., ns-3), and only static topologies with full-buffer downlink traffic are considered. In addition, extending the method to other MAPC schemes (e.g., coordinated beamforming) and analyzing alternative scheduling strategies beyond top-k are important directions for future work. Finally, a natural next step is to validate our approach using both a packet-level simulator and a hardware testbed.

## REFERENCES

[1] W. Ning, F. Tang, M. Zhao, and N. Kato, “A Survey on IEEE 802.11bn Wi-Fi 8: Advantages of Ultra High Reliability for Next-Generation Wireless LANs,” IEEE Comm. Surveys & Tutorials, vol. 28, 2025.

[2] J. Haxhibeqiri et al., “Coordinated Spatial Reuse for WiFi Networks: A Centralized Approach,” in Proc. of WFCS, 2024.

[3] D. Nunez, M. Smith, and B. Bellalta, “Multi-AP Coordinated Spatial Reuse for Wi-Fi 8: Group Creation and Scheduling,” in Proc. of MedComNet, 2023.

[4] D. Zhu, L. Wang, G. Pan, and S. Luan, “Two enhanced schemes for coordinated spatial reuse in IEEE 802.11be: Adaptive and distributed approaches,” Computer Networks, vol. 258, p. 111060, 2025.

[5] M. Wojnar et al., “IEEE 802.11bn Multi-AP Coordinated Spatial Reuse With Hierarchical Multi-Armed Bandits,” IEEE Communications Letters, vol. 29, no. 3, pp. 428–432, 2025.

[6] M. Wojnar et al., “Coordinated Spatial Reuse Scheduling With Machine Learning in IEEE 802.11 MAPC Networks,” IEEE Journal on Selected Areas in Communications, vol. 43, no. 11, pp. 3666–3682, 2025.

[7] M. Du et al., “Deep Reinforcement Learning Based Spatial Reuse for IEEE 802.11bn,” in Proc. of WCNC 2025, 2025.

[8] J. Jung, G. Lee, and J.-M. Chung, “Coordinated Spatial Reuse With Shared AP Index Optimization for IEEE 802.11be WLAN Enhancement,” IEEE Tr. on Network Science and Engineering, vol. 13, 2026.

[9] A. I. J. Forrester, A. Sóbester, and A. J. Keane, Engineering Design via Surrogate Modelling: A Practical Guide. John Wiley & Sons, 2008.

[10] T. T. Sari, S. Mohanti, and G. Secinti, “UWB Assisted Digital Twin for Enhanced Spatial Reuse in Wi-Fi 7,” IEEE Communications Magazine, vol. 64, no. 1, pp. 102–107, 2026.

[11] S. Merlin et al., “TGax Simulation Scenarios,” Nov. 2015, doc.: IEEE 802.11-14/0980r16.

[12] F. Wilhelmi et al., “A tutorial on ieee 802.11bn multi-ap coordination for wi-fi 8: From standardization to performance evaluation,” 2026. [Online]. Available: https://arxiv.org/abs/2606.13759

[13] P. W. Battaglia et al., “Relational inductive biases, deep learning, and graph networks,” CoRR, vol. abs/1806.01261, 2018.

[14] A. Vaswani et al., “Attention is All you Need,” in Advances in Neural Information Processing Systems, vol. 30. Curran Associates, Inc., 2017.

[15] C. M. Bishop. (1994) Mixture Density Networks. [Online]. Available: https://research.aston.ac.uk/en/publications/mixture-density-networks/

[16] I. Loshchilov and F. Hutter, “Fixing Weight Decay Regularization in Adam,” CoRR, vol. abs/1711.05101, 2017.