# ScaleTraj: Scale-Decomposed Human Trajectory Generation

Yulin Zeng   
Santa Clara University   
Santa Clara, CA, USA   
yzeng@scu.edu   
Jun Yan   
Concordia University   
Montreal, Canada   
jun.yan@concordia.ca

Yiheng Zhao Concordia University Montreal, Canada yiheng.zhao@mail.concordia.ca

Yuhong Liu<sup>∗</sup>   
Santa Clara University   
Santa Clara, CA, USA   
yhliu@scu.edu

## Abstract

Individual-level trajectories are essential for applications that rely on user-level mobility behavior. Existing mobility-generation approaches face a fundamental tension between individual-level and population-level statistical fidelity: top-down models reproduce ag gregate mobility statistics but struggle to guarantee realistic individ ual trajectories, while bottom-up generators synthesize individual trajectories yet often fail to preserve key population-level mobility regularities after aggregation. In particular, many bottom-up gen erated populations fail to reproduce the displacement distribution, whose heavy tail is well summarized by a truncated power law. We formalize this phenomenon as the Population-Level Displacement Gap. To investigate its prevalence, we conduct a systematic evaluation of representative model-driven and data-driven trajec tory generators. Our results provide evidence that the Population-Level Displacement Gap is prevalent across diverse generation paradigms. To diagnose this gap, we introduce the Scale Fidelity Profile (SFP), a scale-decomposed metric that quantifies discrep ancies in trip allocations across spatial scales. SFP indicates that population-level displacement mismatches co-occur with systematic errors in how trips are allocated across spatial scales. Motivated by this observation, we develop ScaleTraj, an autoregressive Transformer that decomposes location prediction into a Scale Selection Head that decides the movement scale and a Scale-Aware Cell Selection Head that predicts the destination conditioned on the selected scale. Experiments on two datasets show that ScaleTraj reduces the Population-Level Displacement Gap versus model-driven and datadriven baselines, while preserving fidelity. Our code is available at https://github.com/SCU-TrustworthyAI/ScaleTraj.

## CCS Concepts

• Information systems → Geographic information systems; • Computing methodologies → Modeling and simulation.

## Keywords

Human Mobility, Trajectory Generation, Spatial Scale, Power Law

ACM Reference Format: Yulin Zeng, Yiheng Zhao, Jun Yan, and Yuhong Liu. 2026. ScaleTraj: Scale-Decomposed Human Trajectory Generation. In The 34th ACM International Conference on Advances in Geographic Information Systems (SIGSPATIAL ’26), November 03–06, 2026, Riverside, CA, USA. ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3841645.3842996

## 1 Introduction

Granular, individual-level trajectory generation is essential for critical downstream applications, including electric-vehicle charging load prediction and epidemic simulation, which heavily rely on userlevel behavioral analytics. While macro-level, top-down generative approaches excel at capturing aggregate population statistics (i.e., often reproducing the characteristic power-law tail with high accuracy), they inherently lack the capacity to satisfy these fine-grained individual analytic tasks. Consequently, researchers rely on bottomup trajectory generators to synthesize individual traces, bypassing the privacy constraints and prohibitive collection costs.

However, bottom-up models often struggle to sustain populationlevel statistical properties upon aggregation. Specifically, the aggregated synthetic population can deviate significantly from the corresponding real-data displacement distribution. This paper formalizes this distributional mismatch as the Population-Level Displacement Gap, defined as the statistical discrepancy between the dataset’s own observed displacement distribution and the distribution obtained after aggregating generated individual trajectories. Literature-consensus values are reported only as external context. Such deviations are not merely statistical artifacts. The displacement distribution governs the frequency of long-range movements, which directly influences downstream applications such as epidemic spread, transportation demand, and infrastructure planning. Although recent work [20] has reported this phenomenon in specific leading models, its prevalence across diferent trajectory-generation paradigms remains unknown. We therefore pose RQ1: Does this displacement gap persist across diverse trajectory generation mechanisms, or is it restricted to specific model architectures?

Existing studies address this population-level mismatch through distinct analytical lenses. One line of work attributes populationlevel discrepancies in bottom-up generation to insuficient coordination among independently generated agents, which is associated with the synthetic population missing collective structures and emergent scaling regularities [9, 14, 20]. An alternative account traces the gap to missing activity-purpose semantics: trajectories are driven by underlying intents (home–work commuting, leisure, errands) that collectively shape decisions, and generators that ignore purpose tend to produce locally plausible but globally unstructured trips [2, 4, 15]. Human mobility studies demonstrate that aggregate displacement statistics are strongly shaped by heterogeneous individual behaviors and characteristic movement scales [1, 6, 21], while trajectory-generation surveys consistently identify characteristic distances as vital spatial patterns for evaluating generative fidelity [10]. Grounded in these insights, and rather than attempting to evaluate all potential mechanisms, we introduce and test a targeted, scale-based hypothesis: that the Population-Level Displacement Gap is strongly associated with systemic errors in how trips are allocated across distance scales. Under this scalebased lens, we anticipate diferent underlying failure modes across generation paradigms: for model-driven generators, rigid distancekernel parameters may fail to capture multi-scale mixtures, leading to severe trip misallocation; conversely, for data-driven generators, scale-mixed objectives may allow dominant scales to monopolize the learning signal, smoothing away rare scales under finite model capacity. Consequently, we seek to answer RQ2: To what extent are population-level displacement errors associated with scale-level trip misallocation, and how does this misallocation difer between model-driven and data-driven generators?

This leads to RQ3: Can scale-decomposed generation mitigate the Population-Level Displacement Gap while preserving individuallevel trajectory realism? To address this inquiry, we propose Scale-Traj, an autoregressive Transformer architecture for bottom-up trajectory generation that explicitly models movement scales. Existing autoregressive mobility generators often treat location prediction as sampling from a flattened spatial vocabulary, leaving scale structure implicit. Rather than predicting the next location from a monolithic spatial distribution, ScaleTraj decouples the process: it first predicts the target movement scale and subsequently models the destination cell conditioned on that scale. This factorization treats scale as an explicit routing variable while maintaining an individual-level generation paradigm.

In summary, we make three contributions. Cross-Paradigm Diagnosis: we evaluate representative model-driven and data-driven trajectory generators, spanning Markov, EPR, autoregressive recurrent, autoregressive Transformer, variational autoencoder (VAE), and generative adversarial network (GAN) paradigms, and show that the Population-Level Displacement Gap is prevalent across them. Scale Fidelity Profile (SFP): a scale-decomposed diagnostic metric that localizes displacement errors across spatial scales, exposing structural anomalies invisible to aggregate power-law parameters alone. ScaleTraj: an autoregressive Transformer with a Scale Selection Head and a Scale-Aware Cell Selection Head that mitigates the Population-Level Displacement Gap while preserving fidelity.

## 2 Related Work

We position our contribution against two threads: the literature on multi-scale structure in human mobility, and the design space of bottom-up trajectory generators.

## 2.1 Multi-Scale Structure in Human Mobility

A central spatial regularity in human mobility is the distribution of travel distances � (Δ�), characterized by a heavy-tailed, truncated power law [3, 5]. Historically, this scale-free property suggested that mobility lacked statistically distinct levels, such as “neighborhood trips” versus “city trips.” However, this invariance stands in tension with the hierarchical nature of urban structure and cognitive spatial representation [13]. Subsequent research resolved this tension by demonstrating that individual mobility does not necessarily follow a power law [6]. Instead, aggregate scaling emerges as a result of mixing heterogeneous individual behaviors [21]. Alessandretti et al. [1] formalized this via container theory, proposing that individual mobility operates within nested spatial containers of characteristic sizes. The observed population-level power law is thus a superposition of these bounded, multi-level distributions.

This hierarchical perspective dictates a crucial constraint for the bottom-up trajectory generation explored in this paper: if the underlying data-generating process is multi-scale, a generator using a monolithic spatial distribution or a scale-agnostic objective may fail to replicate the population-level tail. Specifically, such models risk systemic scale-level trip misallocation, in which movements are incorrectly distributed across spatial containers—leading to the Population-Level Displacement Gap. While top-down models can bypass this by directly sampling from an aggregate power law, bottom-up generators should explicitly reconcile individual-level choices with these emergent scales to preserve aggregate-level realism after population aggregation.

## 2.2 Trajectory Generation

Existing trajectory generation approaches can be broadly categorized into model-driven and data-driven methods. Model-driven generators encode mechanistic mobility rules, and two sub-families dominate. Markov-chain models represent trajectories as transitions over a discrete set of locations. EPR models build on the Exploration and Preferential Return (EPR) framework [17], mixing uniform exploration with rank-weighted return. TimeGeo [8] ex tends EPR with circadian patterns, activity-based scheduling, and rank-based destination selection, and is widely used for whole-day trip-chain generation. Both sub-families ofer interpretability, but their hand-crafted rules hard-code the scaling regularities they aim to describe [16]: the scaling exponent is either an explicit input or an implicit consequence of the EPR kernel, and population-level accuracy can mask substantial individual-level deviations [12].

Data-driven generators learn trajectory distributions directly from data without explicit mobility rules. Three generative families have been widely applied [10]. Autoregressive models generate next locations conditioned on visit history. Long short-term memory [? ] (LSTM)-based generators factorize the trajectory likelihood as a product of conditional distributions over discrete locations and continuous sojourn times, autoregressively generating wholeday visit sequences. TrajGPT [7] recasts trajectory generation as a text-infilling problem: visit sequences are rearranged with special delimiters so that a Transformer can autoregressively fill variablelength gaps. Recent work has also explored large language model (LLM)-based mobility generation. Rather than directly learning trajectory dynamics from mobility records alone, these approaches leverage LLMs to simulate mobility decisions, activity planning, or behavioral reasoning. For example, Chain-of-Planned-Behavior (CoPB) [15] prompts LLMs with structured behavioral workflows to generate daily mobility trajectories, while subsequent studies have shown that LLM-generated populations may still exhibit discrepancies in aggregate mobility statistics after aggregation [20]. These findings suggest that improvements in semantic reasoning alone do not necessarily ensure fidelity to population-level mobility regularities. Variational autoencoders compress trajectory into a continuous latent space. TrajSynVAE [18] combines a variational autoencoder with a temporal point process: per-step latent codes capture user decision tendencies, and a deep network parameterizes the continuous-time probabilistic distribution over each visit’s (location, timestamp, waiting-time) tuple. Adversarial methods train a generator–discriminator pair to match the distribution. MoveSim [4] introduces a mobility-specific discriminator incorporating spatial and temporal regularizers. STAR [19] augments a sequential generator with graph-structured branches encoding spatial adjacency and temporal periodicity.

Across those approaches, displacement is typically modeled through a single spatial distribution or generation objective. While these methods difer in architecture and inductive bias, explicit representations of the multi-scale hierarchical structure of human mobility remain largely unexplored. Moreover, existing evaluations primarily rely on aggregate mobility metrics, such as displacement distributions, radius of gyration, visitation frequency, and temporal regularity. Although informative, these metrics provide limited insight into how generation errors are distributed across spatial scales. Consequently, it remains unclear whether population-level displacement mismatches arise from broad distributional deviations or from systematic misallocation of trips across movement scales. This unresolved diagnostic gap motivates the central question of this paper: whether population-level displacement errors can be understood and mitigated through an explicit scale-based representation of human mobility.

## 3 Preliminaries

## 3.1 Definition

A trajectory $\mathbf { X } = \{ ( t _ { i } , c _ { i } ) \} _ { i = 1 } ^ { L }$ is a time-ordered sequence of visits, where each � indexes a tile in a uniform square tessellation over the study area [10]. A globally defined �-scale partition $\big \{ S _ { 1 } , \dots , S _ { K } \big \}$ segments displacement space by distance thresholds into neighborhood / city / urban / region bands, operationalizing the hierarchical spatial structure described in container theory [1]. Whereas container theory allows the number and size of containers to vary across individuals, we use a dataset-level set of ordered scale levels: evaluation assigns realized displacements to scale bands, while ScaleTraj uses the corresponding GMM components as routing variables for generation. Container theory posits that the populationlevel truncated power law emerges as a mixture of bounded-scale components:

$$
P ( \Delta \boldsymbol { r } ) = \sum _ { k = 1 } ^ { K } \pi _ { k } \cdot P ( \Delta \boldsymbol { r } \mid k ) ,\tag{1}
$$

where each $P ( \Delta r \mid k )$ has bounded support within scale band $S _ { k } ,$ and $\pi _ { k }$ is the population-level mixing weight for scale �. In the

generative formulation of §6, we denote the visit timestamp by $a _ { i } \equiv t _ { i }$ and additionally model the stay duration $\tau _ { i } .$

## 3.2 Datasets

Our evaluations are conducted on two real-world mobility datasets that difer in user population, sensing modality, and spatial resolution. GeoLife [22] is a small-scale public GPS trajectory corpus, and Tencent Beijing [15] is a large-scale mobility dataset. The two datasets were selected to represent diferent mobility-observation regimes. GeoLife contains high-resolution GPS trajectories from a relatively small user population, whereas Tencent Beijing is a large-scale released human-mobility corpus with coarser spatial resolution. Both datasets are snapped to a cell grid and split chronologically within each user into train and test partitions.

## 3.3 Evaluation

Population-level displacement fidelity is assessed using two fittedparameter diagnostics, |Δ�| and |Δ�| (km), obtained from the truncated power law [5], where $\beta$ is the scaling exponent and � the cutof distance controlling tail truncation. These are reported together with the SFP, a scale-decomposed diagnostic metric defined in §5. At the individual level, we report eight JSD-based metrics (all lower-is-better): step distance (SD) and radius of gyration (RG) for spatial fidelity; step interval (SI) and stay duration (STAY) for temporal fidelity; and daily travel distance (DTD), daily unique locations (DUL), visitation frequency (Visitation), and individual-rank correlation (IRank) for behavioral fidelity.

## 4 RQ1: Evaluating the Displacement Gap Across Generative Paradigms

This section examines the extent to which individual-level trajectory generators replicate the emergent scaling properties of mobility data. We first verify that the displacement distributions of both datasets can be summarized by a truncated power law, then quantify the distribution-level discrepancies across all representative mechanisms under a unified evaluation framework.

## 4.1 Cross-Paradigm Analysis

Prior work has noted discrepancies in emergent scaling for individual paradigms: Alessandretti et al. [1] for EPR-family models, Yan et al. [20] for LLM-based generators. This analysis is extended systematically to representative major paradigms under a unified evaluation protocol.

As established in Table 1, the displacement distributions on both datasets are well described by a truncated power law. The exponent $\beta$ is consistent across train and test $( | \Delta \beta | < . 0 3 )$ on both datasets, while � shows greater sampling variation on the shorter test window, particularly on Tencent Beijing. For reference, prior work reports approximately $\beta \approx 1 . 7 5 \pm 0 . 1 \updownarrow$ and � ≈ 400 km [5]; we use these literature-consensus values only as external context rather than as the evaluation target. Both datasets have displacement mass in all four Gaussian Mixture Model (GMM)-identified scale bands (Figure 2), and all reported |Δ�| and |Δ�| values are computed against the corresponding test-split fits.

To answer RQ1, we evaluate whether the Population-Level Displacement Gap persists across generation paradigms rather than being tied to a specific architecture. We assess seven representative generators: two model-driven (SemiMarkov, TimeGeo) and five data-driven (LSTM, TrajSynVAE, MoveSim, STAR, TrajGPT). Table 2 summarizes the corresponding fitted-parameter deviations for the seven controlled baselines. The SemiMarkov baseline uses the public implementation released with TrajSynVAE [18]. We additionally examine the LLM-based generator CoPB [15] as a separate case. Rather than retraining it, we diagnose the synthetic trajectory dataset the authors released alongside the method, which already exhibits a qualitatively diferent failure mode. Because this analysis uses the authors’ released trajectories without retraining under the unified protocol, CoPB is treated as a qualitative external case and is not included in the controlled baseline comparison. As Figure 1(a,b) shows, every baseline deviates from the test distribution, but the deviations fall into two opposite failure modes.

![](images/b6757a1c8fc6e45b522efda0852e700cc1e71e21975f072461acda785d69e870.jpg)  
Figure 1: Power-law diagnostic of trajectory generators. The vertical axis $P ( \Delta r \geq x )$ gives the fraction of trips with displacement at least �. (a, b) Complementary cumulative distribution functions (CCDFs) of displacement distances on the GeoLife and Tencent Beijing test splits. The collected real-world data are shown in black, with the literature-consensus truncated-power-law family [5] shown as external reference: the GeoLife curve lies largely within the band, whereas the Tencent Beijing curve approaches and falls below its lower edge in the tail. (c) Same diagnostic applied to CoPB’s oficially released dataset [15]; its maximum observed displacement is 38.3 km, indicating a substantially shorter tail than the Tencent Beijing test distribution (� =129 km).

Table 1: Truncated power-law fit (�, �) for train and test splits. The exponent � is consistent across splits $( | \Delta \beta | < . 0 3$ on both datasets); � shows expected sampling variation on the shorter test window, especially on Tencent Beijing (213 → 129 km), but remains within the same order of magnitude.
<table><tr><td>Dataset</td><td>Split</td><td>Users</td><td>Days</td><td>β</td><td>κ (km)</td></tr><tr><td>GeoLife</td><td>Train</td><td>121</td><td>4,457</td><td>1.707</td><td>524</td></tr><tr><td></td><td>Test</td><td>121</td><td>1,171</td><td>1.730</td><td>356</td></tr><tr><td>Tencent Beijing</td><td>Train</td><td>500</td><td>28,957</td><td>1.674</td><td>213</td></tr><tr><td></td><td>Test</td><td>500</td><td>7,507</td><td>1.686</td><td>129</td></tr></table>

The predominant mode is tail inflation: most methods produce an inflated �, over-representing long-distance trips. The magnitude varies widely across methods, and a close exponent match does not guarantee truncation fidelity: TimeGeo achieves the lowest $| \Delta \beta | = . 0 7 7$ on GeoLife yet still inflates �.

The opposite mode is tail collapse, which appears in the examined LLM-based generator. Figure 1(c) applies the same diagnostic to CoPB’s oficially released synthetic dataset [15], a release accompanying the original method. Its maximum observed displacement is 38.3 km, indicating a substantially shorter tail than the Tencent Beijing test distribution, whose fitted cutof is � =129 km. This sharply truncated support represents a qualitatively diferent failure mode from the tail inflation observed for most other generators.

Despite pointing in opposite directions, both modes share the same underlying failure: no evaluated baseline simultaneously reproduces the test-set scaling exponent and its truncation structure. The Population-Level Displacement Gap is therefore consistently observed across all evaluated paradigms and is not restricted to any single architecture. These aggregate diagnostics, however, reveal only that the distribution deviates, not where in the displacement range the deviation concentrates.

## 5 RQ2: Scale-Decomposed Diagnosis

RQ1 establishes the Population-Level Displacement Gap using aggregate displacement diagnostics. RQ2 asks where in the displacement range this discrepancy concentrates, and why. We proceed in four steps: (i) a training-set GMM defines statistically distinct displacement bands (§5.1); (ii) the Scale Fidelity Profile (SFP) summarizes the discrepancy in how trips are allocated across these bands (§5.2); (iii) applying SFP to the baselines localizes where each one misallocates trips (§5.3); and (iv) we explain why these misallocations arise, with distinct mechanisms for model-driven and data-driven generators (§5.4).

Table 2: Power-law diagnostics on the test set. $| \Delta \beta | , | \Delta \kappa |$ (km): deviation of fitted parameters from real data; lower is better.
<table><tr><td rowspan="2">Method</td><td colspan="2">GeoLife</td><td colspan="2">Tencent Beijing</td></tr><tr><td>|∆β|↓</td><td>|∆κ|↓</td><td>|∆β|↓</td><td> $| \Delta \kappa | \downarrow$ </td></tr><tr><td>SemiMarkov</td><td>.433</td><td>7,937</td><td>.304</td><td>6,629</td></tr><tr><td>TimeGeo</td><td>.077</td><td>4,703</td><td>.052</td><td>3,960</td></tr><tr><td>LSTM</td><td>.373</td><td>7,898</td><td>.260</td><td>5,922</td></tr><tr><td>TrajSynVAE</td><td>.382</td><td>6,566</td><td>.296</td><td>5,967</td></tr><tr><td>MoveSim</td><td>.343</td><td>10,046</td><td>.337</td><td>5,752</td></tr><tr><td>STAR</td><td>.191</td><td>5,267</td><td>.221</td><td>3,366</td></tr><tr><td>TrajGPT</td><td>.536</td><td>13,816</td><td>.280</td><td>2,422</td></tr></table>

## 5.1 Dataset Scale Bands

We partition displacement distances into �=4 bands by fitting a Gaussian Mixture Model to log-displacement on each dataset’s training split. We adopt four ordered scale levels (neighborhood, city, urban, region), sorting the fitted components by increasing center. The default scale count is evaluated against alternative settings in Appendix B.

The resulting bands are strongly imbalanced: the region band carries only about 0.8% of trips on Tencent Beijing and 3.8% on GeoLife. This imbalance is observed in both datasets, while the fitted components remain distinct in log-distance space (Table 5). Preserving this skewed allocation is exactly what SFP is designed to evaluate.

## 5.2 Scale Fidelity Profile

Scale proportions are the quantity that matters for the Population Level Displacement Gap. By the superposition structure in Equation (1), the population-level displacement distribution is a mixture ofper-scale components weighted by �; if� is wrong, the aggregate distribution is wrong, regardless of how accurate each per-scale component is. For example, a generator that over-allocates trips to the region band will inflate the aggregate tail even if its intra-scale shapes are perfect; conversely, perfect intra-scale shapes cannot fix a misallocated �. A useful diagnostic should therefore target the mixing weights directly, rather than the full displacement distribu tion, which also entangles grid resolution and intra-scale details that are less informative for our purposes.

This motivates comparing the real and generated scale-proportion vectors directly, where a lower discrepancy indicates closer scalelevel fidelity.

Definition 5.1 (Scale Fidelity Profile). The Scale Fidelity Profile is defined as

$$
\mathrm { S F P } = \mathrm { J S D } ( \pi ^ { \mathrm { r e a l } } \| \pi ^ { \mathrm { s i m } } ) ,\tag{2}
$$

where $\pmb { \pi } ^ { \mathrm { r e a l } }$ and $\pmb { \pi } ^ { \mathrm { s i m } }$ are the scale-proportion vectors ofthe real and generated populations. Here $\pi _ { k }$ is the fraction of step displacements assigned to band $S _ { k } , \mathrm { i . e . }$ the count of displacements in $S _ { k }$ over the total, where a step displacement is the haversine distance between two consecutive stay events in a trajectory.

## 5.3 SFP Analysis of Baselines

Figure 3 visualizes the per-scale proportion ratios (sim/real) for all baselines, with the corresponding SFP values reported in Table 3.

Model-driven generators exhibit distinct patterns of scale misallocation. TimeGeo over-produces neighborhood-scale trips while under-producing urban-scale trips, whereas SemiMarkov underproduces neighborhood-scale trips and strongly over-produces region-scale trips. These contrasting profiles yield diferent SFP values on GeoLife (.057 for TimeGeo versus .182 for SemiMarkov), showing that scale-level errors can difer substantially even within the same model family.

Data-driven generators share a more consistent direction of misallocation. LSTM, TrajSynVAE, MoveSim, and STAR all overproduce region-scale trips while under-producing neighborhoodscale trips, and TrajGPT shows a more extreme variant of the same, reaching the highest SFP among all baselines (.406 on GeoLife). Moreover, similar aggregate scaling-exponent errors can mask substantially diferent scale-level errors: on Tencent Beijing, TrajGPT and TrajSynVAE have similar |Δ�| values (.280 versus .296), but markedly diferent SFP scores (.228 versus .080).

Across both families, the dominant tendency is therefore to under-produce neighborhood-scale trips and over-produce regionscale trips, with TimeGeo as the notable exception. SFP spans a wide range (from .038 for STAR to .406 for TrajGPT on GeoLife), showing that the Population-Level Displacement Gap coincides with systematic scale-level misallocation. Although this dominant pattern crosses generation paradigms, the mechanisms underlying these scale-level errors may difer across model families, which we examine next.

## 5.4 Interpreting Scale Misallocation

The SFP diagnostic suggests that the Population-Level Displacement Gap is closely related to how generators allocate trips across spatial scales. The mechanisms difer between model-driven and data-driven generators.

For model-driven generators, the main limitation is that a small number of distance parameters is often used to represent a displacement distribution that is multi-scale in practice. For example, EPR-style models typically rely on a compact power-law jump kernel, while TimeGeo and related models add temporal structure but still use a limited distance-generation mechanism. Such models can fit an aggregate slope reasonably well, but they have limited control over how probability mass is distributed across neighborhood, city, urban, and region scales. As a result, a model may match a global power-law parameter while still over-producing trips at one scale and under-producing trips at another. This is illustrated by TimeGeo: it achieves the lowest $| \Delta \beta |$ among baselines on GeoLife, yet its SFP remains .057, showing that a close aggregate fit can coexist with substantial per-scale misallocation.

For data-driven generators, the issue is less about the number of hand-designed parameters and more about how the training signal is distributed. Standard autoregressive, VAE, and adversarial objectives are usually optimized over scale-mixed data, in which region-scale trips account for under 1% of tokens on Tencent Beijing and under 4% on GeoLife. Under unweighted token-level objectives, region-scale examples occur roughly one to two orders of magni tude less frequently than neighborhood-scale examples, depending on the dataset. This imbalance can weaken the learning signal for rare scales. With finite model capacity, the network may preferen tially fit the much more frequent non-region scales, while sparse region-tail structure may be smoothed or absorbed into adjacent scales. Notably, this scarcity in training does not translate into scarcity at generation: because the rare scale is poorly modeled rather than simply suppressed, its probability mass is misplaced rather than removed, which is consistent with the inflated aggregate � observed for these generators in RQ1.

![](images/fbacd8508f4c1346c33e2b2a05fdd2a09a710e6c615af5a6f976deea9bc66bb1.jpg)

![](images/4a7a469dcde7a5b0a1480c9fc10e7d8fe5316a11408af0ce973dffee948fcb3b.jpg)

Figure 2: Gaussian Mixture Model (� = 4) fitted to log-displacement distances on the training set. Both datasets exhibit multi-modal structure in log-space, with four displacement bands from neighborhood to region. Vertical dotted lines indicate the dataset-specific scale boundaries used for the SFP diagnostic; the fitted GMM components provide the scale-conditioned distance structure used by ScaleTraj.  
![](images/2edeb9a89a1ac7359b466841cb852345aceefa9e3e4c2b317195540a2d0d261e.jpg)  
Figure 3: Per-scale proportion ratio (sim/real) on log scale for seven baselines on the test set. The black line marks ratio = 1, and the gray band spans ±20%. Many methods show substantial scale-level misallocation, with short-range and long-range movements often deviating in diferent directions. These patterns expose compensating errors that are not visible from aggregate power-law parameters alone.

These observations do not prove that scale misallocation is the only source of the Population-Level Displacement Gap. Rather, they suggest why scale-level errors are plausible and why they appear across diferent generator families. Model-driven methods lack enough explicit degrees of freedom to control scale proportions, while data-driven methods receive weaker training signals for rare scales because of imbalanced token frequencies. That these distinct mechanisms can produce the same symptom points to a shared omission rather than a paradigm-specific flaw: neither family represents movement scale as an explicit, separately controllable quantity. This shared omission directly motivates the design of ScaleTraj, which promotes scale from an implicit by-product of the spatial softmax to an explicit routing variable, giving each scale band dedicated capacity and a controllable mixing weight.

## 6 RQ3: ScaleTraj Framework

RQ2 localized the Population-Level Displacement Gap to systematic misallocation of trips across scale bands. ScaleTraj addresses this by treating scale as an explicit routing variable rather than an emergent property of a single spatial softmax. Figure 4 summarizes the overall architecture. We first present the scale-decomposed factorization (§6.1), then the Transformer backbone producing the hidden state $h _ { i }$ (§6.2), and finally the prediction heads that implement the factorization (§6.3).

## 6.1 Scale-Decomposed Factorization

Standard autoregressive trajectory generators predict cell $c _ { i }$ from a single softmax over the full cell vocabulary, implicitly marginalizing over scales. As described in the previous section, this formulation allocates capacity in proportion to token frequency, starving rare region-scale trips. Rather than letting rare region-scale trips com pete with dominant neighborhood-scale trips in a single softmax, ScaleTraj first decides the scale � and then predicts the cell � conditioned on �.

Concretely, ScaleTraj decomposes the prediction into a cascade of conditionals. At each step �, conditioned on the Transformer hidden state $h _ { i } { \mathrm { : } }$

$$
P ( \operatorname { t r a j } ) = { \Big \vert } \ \prod _ { i } P ( k _ { i } \mid h _ { i } ) P ( c _ { i } \mid k _ { i } , h _ { i } ) P ( \tau _ { i } \mid k _ { i } , c _ { i } , h _ { i } ) P ( a _ { i } \mid k _ { i } , c _ { i } , h _ { i } ) .\tag{3}
$$

where $k _ { i } \in \{ 1 , . . . , K \}$ is the spatial scale, $c _ { i }$ is the discrete grid cell, $\tau _ { i }$ is the stay duration, and $a _ { i }$ is the arrival time. Scale is predicted first by the Scale Selection Head; cell prediction is then explicitly conditioned on scale by the Scale-Aware Cell Selection Head, so each scale band receives dedicated model capacity regardless of its population frequency.

ScaleTraj uses the training-set GMM introduced in §5.1 to define its ordered scale-conditioned components. The selected $k _ { i }$ acts as a routing variable for destination prediction rather than as a hard constraint on the realized displacement. Because the GMM is refit from each dataset’s own displacement distribution rather than using fixed scale components across datasets, the same architecture can adapt its scale-conditioned components to dataset-specific displacement structure.

## 6.2 Transformer Backbone

A causal Transformer [? ] serves as the autoregressive backbone because daily mobility decisions depend on non-local visit history: a return trip is conditioned on where the day began, and an evening destination on the accumulated trajectory rather than only the previous stop. Within the supplied context window, self-attention allows each prediction to condition directly on any earlier visit through explicit pairwise attention. This is useful for scale decomposition in particular, since the appropriate movement scale $( \mathrm { e . g . , a }$ long commute home versus a short errand) is often determined by visits several steps earlier.

Each visit token concatenates a cell embedding, a scale embedding, Fourier-encoded temporal features, an anchor-conditioned embedding carrying home/work cells and a user-latent code, and persona and time-budget features; Rotary Position Embedding (RoPE) [? ] encodes visit order and a causal mask restricts $h _ { i }$ to visits $1 , \ldots , i - 1$ . We prepend up to $C { = } 3$ previous days of the same user’s stays as history context. The four prediction heads branch from $h _ { i }$ and are trained jointly with teacher forcing plus scheduled sampling on the scale step. More details are in Appendix A.

## 6.3 Prediction Heads

The two scale-aware heads are the primary architectural contribution of this paper. They realize the factorization of $\operatorname { E q . } \left( 3 \right)$ : the Scale Selection Head produces $P ( k _ { i } \mid h _ { i } )$ and the Scale-Aware Cell Selection Head produces $P ( c _ { i } \mid k _ { i } , h _ { i } )$ .

Scale Selection Head. The Scale Selection Head produces a $K \cdot$ dimensional logit vector:

$$
\begin{array} { r } { \mathrm { l o g i t } _ { i } ( k ) = \alpha _ { m } \log \hat { \pi } _ { k } + \alpha _ { h } \big [ \mathrm { M L P } _ { \mathrm { s c a l e } } ( h _ { i } ) \big ] _ { k } . } \end{array}\tag{4}
$$

where $\hat { \pi } _ { k }$ is the marginal scale proportion estimated from the training set, $\alpha _ { m }$ and $\alpha _ { h }$ are fixed hyperparameters, and ${ \mathrm { M L P } } _ { \mathrm { s c a l e } }$ is a two-layer MLP whose output provides a context-dependent residual adjustment to the marginal-prior logits. Keeping $\alpha _ { m }$ and $\alpha _ { h }$ fixed retains an explicit prior contribution and is intended to discourage collapse toward a frequency-dominated free softmax.

Scale-Aware Cell Selection Head. Once $k _ { i }$ is determined, cell logits combine two additive terms:

$$
\ell _ { i } ( c ) = \alpha _ { d } { \big [ } g _ { k _ { i } } ( c ) - J _ { k _ { i } } ( c ) { \big ] } + \alpha _ { r } { \big [ } r ^ { ( k _ { i } ) } ( h _ { i } ) { \big ] } _ { c } .\tag{5}
$$

where $g _ { k _ { i } } ( c )$ is a GMM-based Gaussian distance prior, given by the Gaussian log-density of log $d ( c _ { i - 1 } , c )$ under scale $\boldsymbol { k _ { i } } ^ { \prime } \mathrm { : }$ s fitted component, $r ^ { ( k _ { i } ) }$ is the scale-specific residual ${ \mathrm { M L P } } , J _ { k _ { i } } ( c )$ is a correction term for the non-uniform density of candidate cells, and $\alpha _ { d } , \alpha _ { r }$ are fixed coeficients. $P ( c _ { i } \mid k _ { i } , h _ { i } ) = \operatorname { s o f t m a x } ( \ell )$ . The prior $g _ { k _ { i } }$ adds a quadratic log-penalty growing with distance from the scale’s characteristic range, rather than relying on the model to learn spatial boundaries from data alone.

The residual term $r ^ { ( k _ { i } ) }$ is realized as � independent two-layer MLPs, one per scale level. For compactness, the notation $r ^ { ( k _ { i } ) } ( { \ r _ { h _ { i } } } )$ suppresses two auxiliary inputs used in the implementation: the selected-scale feature and the previous-cell embedding. We retain the name “residual $\mathrm { M L P ^ { * } }$ because $r ^ { ( k _ { i } ) }$ acts as a learnable residual correction added to the geometric prior $g _ { k _ { i } }$ , not because the MLP itself contains a skip connection. At each training step, only the routed scale-specific expert receives gradients; routing uses the ground-truth scale under teacher forcing and the predicted scale under scheduled sampling. This gives each scale dedicated capacity rather than forcing all movements through a single shared residual predictor. The ablation verifies that replacing a single shared MLP with these per-scale MLPs is associated with improved heavy-tail fidelity. Additional details on the correction term and personaconditioned user latents are provided in Appendix C.

![](images/5a63e47cd78aca29b69f102749e67dcce1f7df952431a4e0ddb9979a7a215be3.jpg)  
Figure 4: ScaleTraj architecture. Past visits are embedded and encoded by a causal Transformer $( L { = } 4 , H { = } 4 , d { = } 1 2 8 )$ into $h _ { i } .$ The Scale Selection Head combines a frozen training-set prior $\hat { \pi } _ { k }$ with a learned context MLP to sample $k _ { i }$ (here “urban”). The Scale-Aware Cell Selection Head routes the current model state through a scale-specific MLP and combines it with a scale-conditioned Gaussian distance prior; an optional composite inference-time reranking stage Reranker(�) is applied before sampling $c _ { i } .$ . Duration and arrival heads complete the visit, and $( k _ { i } , c _ { i } , \tau _ { i } , a _ { i } )$ is fed back autoregressively. Colors: orange = frozen; blue = learned.

Duration and arrival heads. Stay duration �<sub>�</sub> is predicted by a twolayer MLP from $h _ { i } , e _ { k _ { i } }$ , and $e _ { c _ { i } { : } }$ , output over non-uniform duration bins (denser at short durations to match empirical heavy-tailed stays); an auxiliary Gaussian regression head (�, log �) is combined at weight $\lambda _ { g } = 0 . 5$ . Arrival time $a _ { i }$ is predicted from the same inputs, augmented with the duration head’s predicted Gaussian parameters (�<sub>�</sub>, log �<sub>�</sub> ).

## 6.4 Training and Generation

Training minimizes categorical cross-entropy losses on the cascade outputs, together with a Gaussian duration negative log-likelihood and auxiliary end-of-sequence (EOS), activity-type, day-level statistics, and persona-conditioned user-latent regularization terms.

Generation samples autoregressively at frozen parameters. The cell head is sharpened by a fixed temperature and augmented with two inference-time rerankers motivated by well-documented mobility regularities: a global-frequency term biasing generation toward popular places, and a revisit reward encouraging the routine returntrips that dominate individual mobility [17]. Causal masks enforce temporal validity, and generation terminates at the EOS threshold or end-of-day. More details are in Appendix A.

## 7 Experiments

This section evaluates ScaleTraj against the baselines under the unified protocol of §3.3. Synthetic generation is aligned to the heldout test user-day records. ScaleTraj is trained from scratch on a single A6000 GPU. Beyond the main comparison, an ablation over the two cell-head knobs (§7.2) and a sweep over the number of scale bands � is reported in Appendix B. Key configuration details are reported in Appendix A.

## 7.1 ScaleTraj Results

The primary evaluation considers the three diagnostics directly tied to RQ1 and RQ2: SFP, |Δ�|, and |Δ�|. As shown in Table 3, ScaleTraj achieves the best performance on all three diagnostics on both datasets. This indicates that the improvement is observed at both levels targeted by the proposed framework: the allocation of trips across spatial scales and the resulting aggregate displacement distribution.

ScaleTraj models displacement through scale-specific components rather than relying on a single scale-agnostic spatial prediction process. The Scale Selection Head explicitly controls the allocation of movements across scale bands, while the Scale-Aware Cell Selection Head models destination choice conditioned on the selected scale. The improvements in both SFP and the fitted powerlaw diagnostics in Table 3 are consistent with this decomposition: better preservation of scale proportions is accompanied by a closer match to the aggregate test-set displacement distribution.

Figure 5 provides the corresponding distributional comparison. ScaleTraj closely follows the test-set CCDF across the displacement range, while TimeGeo, the strongest baseline in terms of $| \Delta \beta |$ , still exhibits a substantially diferent tail. This comparison illustrates why matching the aggregate scaling exponent alone is insuficient for displacement fidelity.

Table 3: Full evaluation on both datasets. All metrics ↓ (lower is better); per-dataset best in bold.
<table><tr><td></td><td colspan="10">Tencent Beijing</td><td colspan="10">GeoLife</td></tr><tr><td>Method</td><td>SD</td><td>RG</td><td>SI</td><td>STAY</td><td>DTD</td><td>DUL</td><td>Visit</td><td>IRank</td><td>SFP</td><td>|∆β|</td><td>|∆κ|</td><td>SD</td><td>RG</td><td>SI</td><td>STAY</td><td>DTD</td><td>DUL Visit</td><td>IRank</td><td>SFP</td><td>|∆β|</td><td>|∆κ|</td></tr><tr><td>SemiMarkov</td><td>.089</td><td>.199</td><td>.282</td><td>.101</td><td>.405</td><td>.487</td><td>.319</td><td>.440</td><td>.067</td><td>.304</td><td>6,629 .234</td><td>.351</td><td>.294</td><td>.142</td><td>.575</td><td>.572</td><td>.387</td><td>.586</td><td>.182</td><td>.433</td><td>7,937</td></tr><tr><td>TimeGeo</td><td>.056</td><td>.143</td><td>.032</td><td>.110</td><td>.127</td><td>.037</td><td>.269</td><td>.039</td><td>.040</td><td>.052</td><td>3,960</td><td>.158</td><td>.219</td><td>.111 .174</td><td></td><td>.207 .284</td><td>.435</td><td>.136</td><td>.057</td><td>.077</td><td>4,703</td></tr><tr><td>LSTM</td><td>.081</td><td>.197</td><td>.182</td><td>.106</td><td>.467</td><td>.548</td><td>.318</td><td>.502</td><td>.057</td><td>.260</td><td>5,922</td><td>.174 .202</td><td>.014</td><td>.030</td><td>.327</td><td>.227</td><td>.118</td><td>.264</td><td>.121</td><td>.373</td><td>7,898</td></tr><tr><td>TrajSynVAE</td><td>.106</td><td>.169</td><td>.088</td><td>.021</td><td>.284</td><td>.256</td><td>.112</td><td>.234</td><td>.080</td><td>.296</td><td>5,967</td><td>.204 .251</td><td>.020</td><td>.038</td><td>.345</td><td>.160</td><td>.070</td><td>.207</td><td>.155</td><td>.382</td><td>6,566</td></tr><tr><td>MoveSim</td><td>.231</td><td>.262</td><td>.178</td><td>.091</td><td>.318</td><td>.210</td><td>.158</td><td>.258</td><td>.186</td><td>.337</td><td>5,752 .197</td><td>.129</td><td>.122</td><td>.123</td><td>.140</td><td>.057</td><td>.143</td><td>.061</td><td>.117</td><td>.343</td><td>10,046</td></tr><tr><td>STAR</td><td>.093</td><td>.147</td><td>.297</td><td>.274</td><td>.186</td><td>.294</td><td>.175</td><td>.161</td><td>.064</td><td>.221</td><td>3,366</td><td>.099 .075</td><td>.353</td><td>.363</td><td>.096</td><td>.087</td><td>.087</td><td>.086</td><td>.038</td><td>.191</td><td>5,267</td></tr><tr><td>TrajGPT</td><td>.272</td><td>.432</td><td>.212</td><td>.267</td><td>.570</td><td>.575</td><td>.212</td><td>.587</td><td>.228</td><td>.280</td><td>2,422 .459</td><td>.672</td><td>.200</td><td>.097</td><td>.693</td><td>.693</td><td>.399</td><td>.693</td><td>.406</td><td>.536</td><td>13,816</td></tr><tr><td>ScaleTraj</td><td>.039</td><td>.023</td><td>.007</td><td>.055</td><td>.016</td><td>.018</td><td>.018</td><td>.019</td><td>.003</td><td>.021</td><td>24 .123</td><td>.080</td><td>.047</td><td>.032</td><td>.061</td><td>.054</td><td>.038</td><td>.021</td><td>.003</td><td>.026</td><td>131</td></tr></table>

![](images/2683185f7fe140f4a673d7911ed1b840c9090c5326dcf68a995c906ecf4dbe27.jpg)  
Figure 5: Displacement CCDF comparison of ScaleTraj (red), the real test-set distribution (black), and TimeGeo (blue dashed), the strongest baseline in terms of |Δ�|. The literature-consensus band is shown as external reference.

Individual-level metrics provide an additional preservation check. As reported in Table 3, ScaleTraj remains competitive across the spatial, temporal, and behavioral metrics while substantially improving the scale-level and aggregate-displacement diagnostics. Its strongest individual-level results occur on the behavioral metrics, while several baselines remain better on selected spatial or temporal measures. Overall, the results indicate that improved populationlevel displacement fidelity does not require a broad sacrifice in individual-level trajectory fidelity.

## 7.2 Ablation

The cell head has two architectural knobs: the number of scale bands � and whether the residual MLP is shared across scales or split into � per-scale MLPs (Table 4).

� = 1, shared MLP (no scale decomposition). With � = 1 the scale-conditioned Gaussian collapses to a single global Gaussian fit on the full displacement distribution, and the per-scale MLP set collapses to one shared MLP. This single-scale baseline leaves substantial residual error on every diagnostic $( | \Delta \beta | = . 1 6 9 / | \Delta \kappa | = 7 4 $ on Tencent Beijing and $| \Delta \beta | = . 8 6 9 / | \Delta \kappa | = 3 4 3$ on GeoLife). A single Gaussian cannot represent the multi-scale dataset’s displacement distribution, so it concentrates displacement mass on short-range cells; the small Tencent Beijing $\lvert \Delta \kappa \rvert = 7 4$ is a maximum likelihood estimation (MLE) artifact of that collapsed support, not evidence of good tail fidelity. |Δ�| becomes informative only once a generator has the right qualitative shape.

Table 4: Ablation on the cell head’s two knobs: � and residual-MLP sharing. Bottom row is the full ScaleTraj system.
<table><tr><td rowspan="2">Variant</td><td colspan="3">Tencent Beijing</td><td colspan="3">GeoLife</td></tr><tr><td></td><td>SFP↓ |∆κ|↓ |∆β|↓</td><td></td><td>SFP↓</td><td>|∆κ|↓</td><td>|∆β|↓</td></tr><tr><td>K=1, shared MLP</td><td>.038</td><td>74</td><td>.169</td><td>.043</td><td>343</td><td>.869</td></tr><tr><td>K=4, shared MLP</td><td>.009</td><td>58</td><td>.050</td><td>.002</td><td>19,644</td><td>.121</td></tr><tr><td>ScaleTraj</td><td>.003</td><td>24</td><td>.021</td><td>.003</td><td>131</td><td>.026</td></tr></table>

� = 1 → � = 4: scale decomposition improves scale fidelity. Expanding to four scale bands (one global Gaussian becomes four scale-conditioned Gaussians, and the Scale Selection Head chooses among them), with the shared MLP held fixed, drops SFP from .038 to .009 on Tencent Beijing and from .043 to .002 on GeoLife; aggregate |Δ�| drops from .169 to .050 on Tencent Beijing and from .869 to .121 on GeoLife. Scale decomposition alone reduces the population-level scale misallocation. On GeoLife |Δ�| temporarily inflates to 19,644 before per-scale MLPs resolve it, suggesting that the shared MLP remains a bottleneck.

Shared MLP → � =4 per-scale MLPs: per-scale capacity improves heavy-tail fidelity. A shared MLP must model all scales jointly despite their highly imbalanced token frequencies; on GeoLife, this configuration exhibits $| \Delta \kappa | = 1 9 , 6 4 4$ . Splitting the residual into � =4 per-scale MLPs (with only the routed scale-specific MLP active at each training step) drops |Δ�| to 131 on GeoLife and to 24 on Tencent Beijing; |Δ�| also drops to .021/.026. Per-scale MLPs strictly dominate the shared MLP on both power-law diagnostics, with only a marginal SFP cost on GeoLife. We adopt the � =4 per-scale configuration as the released ScaleTraj.

## 8 Discussion and Limitations

This work’s evaluation spans representative model-driven and datadriven generators, providing evidence that the Population-Level Displacement Gap is not confined to a single model. The evaluated set, nevertheless, is not exhaustive; difusion-based, hybrid-based, and other emerging mobility generators are not included here and remain to be systematically examined. Whether the Population-Level Displacement Gap extends to these paradigms, and whether scale decomposition ofers similar benefits there, is an open question.

Generalization across datasets also warrants further study. Both datasets are collected in China, and GeoLife contains a relatively small number of users, which limits the diversity of geographic settings and population structures represented in the current evaluation. Mobility patterns may difer across cities and regions because ofvariations in urban form, transportation infrastructure, and observation scale, potentially afecting both the learned scale boundaries and the resulting displacement distributions. In addition, displace ment statistics can be sensitive to stay-point extraction, spatial discretization, and other trajectory preprocessing choices, particularly at neighborhood and city scales. Future work should therefore examine whether the observed scale-level misallocation patterns remain stable across additional generators, geographic regions, population sizes, and preprocessing configurations. Such evaluations would help determine which aspects of the observed scale structure are broadly transferable and which are dataset-specific.

The truncated power law and SFP are diagnostics of distributional fidelity, not complete definitions of mobility realism. Real displacement patterns are shaped by transportation infrastructure, land use, culture, and observation timescale, and deviations in �, �, or scale proportions do not by themselves imply behaviorally unrealistic trajectories. SFP should therefore be viewed as comple mentary to conventional spatial, temporal, and behavioral metrics rather than as their replacement. Moreover, the absolute SFP values depend on implementation choices such as the smoothing constant and logarithm base, so SFP is best interpreted as a relative ranking across methods under a fixed evaluation protocol rather than as an absolute measure.

ScaleTraj explicitly incorporates training-set scale statistics via GMM-derived bands and priors. Its gains therefore show the bene fit of preserving scale structure rather than discovering it without prior information. A natural concern is that ScaleTraj encodes the same scale statistics on which it is evaluated. However, the reported individual-level JSD metrics are not directly optimized, yet ScaleTraj improves on most of them; the gains therefore extend beyond the scale statistics it explicitly incorporates. The individual-level metrics indicate no broad loss of trajectory realism, but do not establish improvements in all behavioral dimensions. Our use of learned user latent codes also limits the current evaluation to known-user generation; unseen-user generalization remains future work. Finally, our results come from a single generation run per configuration; reporting variance across repeated generation runs is left to future work.

## 9 Conclusion

The Population-Level Displacement Gap, a systematic failure of bottom-up trajectory generators to reproduce the displacement distribution after aggregation, is shown to be prevalent across representative model-driven and data-driven paradigms. To diagnose the gap, the Scale Fidelity Profile reveals systematic misallocation of trips across spatial scales. Building on this diagnosis, ScaleTraj treats scale as an explicit routing variable and substantially reduces the gap while preserving individual-level fidelity. These results indicate that the Population-Level Displacement Gap is a diagnosable, scale-localized phenomenon, and that scale-aware evaluation ofers a useful lens for future mobility generators.

## Acknowledgments

This research was funded by the U.S. National Science Foundation under Grant 2330504. The views and conclusions presented in this study are those of the authors and do not necessarily reflect the oficial policy or position of NSF.

The authors disclose use of generative AI tools (Claude and Chat-GPT) to assist with manuscript editing, code review, experiments, and analyses; all research design and interpretation of results were performed by the authors.

## Ethics and Privacy Statement

Both datasets used in this work are publicly released research corpora used under their intended terms. Synthetic trajectory generation carries dual-use considerations. On the benefit side, it can reduce the need to share raw mobility traces while supporting downstream research such as epidemic simulation and EV charging planning. On the risk side, mobility generators could in principle be misused to profile movement patterns or to fabricate plausible traces for surveillance. We note that ScaleTraj is currently evaluated in a known-user setting, replaying trained user codes so that generated and real populations share the same anchors; it is therefore a tool for augmenting and studying existing populations rather than for reconstructing traces of unseen individuals. All experiments use only de-identified data, and we release our code to support reproducibility and to encourage downstream use under appropriate consent and governance.

## References

[1] Laura Alessandretti, Ulf Aslak, and Sune Lehmann. 2020. The scales of human mobility. Nature 587, 7834 (2020), 402–407.

[2] John L Bowman and Moshe E Ben-Akiva. 2001. Activity-based disaggregate travel demand model system with activity schedules. Transportation research part a: policy and practice 35, 1 (2001), 1–28.

[3] Dirk Brockmann, Lars Hufnagel, and Theo Geisel. 2006. The scaling laws of human travel. Nature 439, 7075 (2006), 462–465.

[4] Jie Feng, Zeyu Yang, Fengli Xu, Haisu Yu, Mudan Wang, and Yong Li. 2020. Learning to simulate human mobility. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining. ACM, New York, NY, USA, 3426–3433.

[5] Marta C González, César A Hidalgo, and Albert-László Barabási. 2008. Understanding individual human mobility patterns. nature 453, 7196 (2008), 779–782.

[6] Xiao-Pu Han, Qiang Hao, Bing-Hong Wang, and Tao Zhou. 2011. Origin of the scaling law in human mobility: Hierarchy of trafic systems. Physical Review E—Statistical, Nonlinear, and Soft Matter Physics 83, 3 (2011), 036117.

[7] Shang-Ling Hsu, Emmanuel Tung, John Krumm, Cyrus Shahabi, and Khurram Shafique. 2024. Trajgpt: Controlled synthetic trajectory generation using a multitask transformer-based spatiotemporal model. In Proceedings ofthe 32nd

ACM International Conference on Advances in Geographic Information Systems. ACM, New York, NY, USA, 362–371.

[8] Shan Jiang, Yingxiang Yang, Siddharth Gupta, Daniele Veneziano, Shounak Athavale, and Marta C González. 2016. The TimeGeo modeling framework for urban mobility without travel surveys. Proceedings of the National Academy ofSciences 113, 37 (2016), E5370–E5378.

[9] Yifan Liu, Xishun Liao, Haoxuan Ma, Brian Yueshuai He, Chris Stanford, and Jiaqi Ma. 2025. Human mobility modeling with household coordination activities under limited information via retrieval-augmented LLMs. In 2025 IEEE 28th International Conference on Intelligent Transportation Systems (ITSC). IEEE, IEEE, New York, NY, USA, 951–958.

[10] Massimiliano Luca, Gianni Barlacchi, Bruno Lepri, and Luca Pappalardo. 2021. A survey on deep learning for human mobility. ACM Computing Surveys (CSUR) 55, 1 (2021), 1–44.

[11] Leandros A Maglaras and Dimitrios Katsaros. 2015. Social clustering of vehicles based on semi-Markov processes. IEEE Transactions on Vehicular Technology 65, 1 (2015), 318–332.

[12] Ludovico Napoli, Márton Karsai, and Esteban Moro. 2024. One rule does not fit all: deviations from universality in human mobility modeling. arXiv preprint arXiv:2410.23502 (2024).

[13] Anssi Paasi. 2004. Place and region: looking through the prism of scale. Progress in human geography 28, 4 (2004), 536–546.

[14] Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. 2023. Generative agents: Interactive simulacra of human behavior. In Proceedings ofthe 36th annual acm symposium on user interface software and technology. ACM, New York, NY, USA, 1–22.

[15] Chenyang Shao, Fengli Xu, Bingbing Fan, Jingtao Ding, Yuan Yuan, Meng Wang, and Yong Li. 2024. Chain-of-planned-behaviour workflow elicits few-shot mobility generation in llms. arXiv preprint arXiv:2402.09836 (2024)

[16] Shuji Shinohara, Daiki Morita, Hayato Hirai, Ryosuke Kuribayashi, Nobuhito Manome, Toru Moriyama, Yoshihiro Nakajima, Yukio-Pegio Gunji, and Ung-il Chung. 2025. A Simple Walk Model for Reproducing Power Laws in Human mobility. arXiv preprint arXiv:2502.17885 (2025).

[17] Chaoming Song, Tal Koren, Pu Wang, and Albert-László Barabási. 2010. Modelling the scaling properties of human mobility. Nature physics 6, 10 (2010), 818–823.

[18] Huandong Wang, Qizhong Zhang, Yuchen Wu, Depeng Jin, Xing Wang, Lin Zhu, and Li Yu. 2023. Synthesizing human trajectories based on variational point processes. IEEE Transactions on Knowledge and Data Engineering 36, 4 (2023), 1785–1799.

[19] Yu Wang, Tongya Zheng, Shunyu Liu, Zunlei Feng, Kaixuan Chen, Yunzhi Hao, and Mingli Song. 2024. Spatiotemporal-augmented graph neural networks for human mobility simulation. IEEE Transactions on Knowledge and Data Engineering 36, 11 (2024), 7074–7086.

[20] Hua Yan, Heng Tan, and Yu Yang. 2026. Bridging Individual and Collective Realism in LLM-Based Human Mobility Simulation via Mobility Scaling-Law Guidance. arXiv preprint arXiv:2602.16726 (2026).

[21] Xiao-Yong Yan, Xiao-Pu Han, Bing-Hong Wang, and Tao Zhou. 2013. Diversity of individual mobility patterns and emergence of aggregated scaling laws. Scientific reports 3, 1 (2013), 2678.

[22] Yu Zheng, Xing Xie, Wei-Ying Ma, et al. 2010. GeoLife: A collaborative social networking service among user, location and trajectory. IEEE Data Eng. Bull. 33, 2 (2010), 32–39.

## A Hyperparameters

Unless noted otherwise, all values below are shared across both datasets and across every reported configuration (the three ablation rows of §7.2 and the �-sweep of Appendix B); only the number of scale bands � and the residual-MLP sharing flag vary between configurations. The Transformer backbone uses hidden size $d = 1 2 8 , L = 4$ layers, and �=4 attention heads. Optimization uses AdamW with learning rate $3 \times 1 0 ^ { - 4 }$ and batch size 64, with validation-based early stopping (no curriculum or pre-training). The cell head combines the GMM-based distance prior with the per-scale residual MLP via fixed coeficients $\alpha _ { d } { = } 2 . ($ 0 and $\alpha _ { r } { = } 0 . 5$ (Eq. (5)); the scale head’s two fixed coeficients are $\alpha _ { m } { = } \alpha _ { h } { = } 1 . 0$ . Inference applies a cell-head temperature $T _ { c } { = } 0 . 3$ together with the two additive rerankers of §6.4 at $\alpha _ { \mathrm { f r e q } } { = } \alpha _ { \mathrm { r e t } } { = } 0 . 3$ . Two inference-time termination parameters are dataset-specific: the EOS threshold $\theta _ { \mathrm { E O S } }$ is 0.5 on Tencent Beijing and 0.7 on GeoLife, and the minimum-stops floor is 1 on Tencent

Table 5: Fitted �=4 GMM components on log-displacement: per-component center (km), log-space standard deviation $\sigma _ { \mathrm { l o g } } ,$ mixture weight, and per-band token share after boundary assignment.
<table><tr><td>Dataset</td><td>Band</td><td> $e ^ { \mu _ { \mathrm { l o g } } } ~ ( \mathrm { k m } )$ </td><td> $\sigma _ { \mathrm { l o g } }$ </td><td>weight</td><td>tokens</td></tr><tr><td rowspan="4">GeoLife</td><td>neighborhood</td><td>0.37</td><td>0.41</td><td>0.247</td><td>45.6%</td></tr><tr><td>city</td><td>1.26</td><td>0.55</td><td>0.368</td><td>24.3%</td></tr><tr><td>urban</td><td>8.23</td><td>0.70</td><td>0.360</td><td>26.3%</td></tr><tr><td>region</td><td>165.27</td><td>0.70</td><td>0.025</td><td>3.8%</td></tr><tr><td rowspan="4">Tencent Beijing</td><td>neighborhood</td><td>0.51</td><td>0.19</td><td>0.362</td><td>53.3%</td></tr><tr><td>city</td><td>1.46</td><td>0.59</td><td>0.261</td><td>20.5%</td></tr><tr><td>urban</td><td>9.97</td><td>0.70</td><td>0.365</td><td>25.4%</td></tr><tr><td>region</td><td>207.75</td><td>0.70</td><td>0.012</td><td>0.8%</td></tr></table>

Beijing and 2 on GeoLife (reflecting each corpus’s empirical daylength distribution). The remaining auxiliary-loss weights and tokenization settings are fixed across datasets and configurations; their exact values are specified in the released training and reproduction code.

## B Sensitivity to the number of scale bands �

## B.1 Fitted �=4 GMM components

For reference, Table 5 reports the full �=4 GMM fit used throughout the paper. The fitted components capture diferent regions of the log-displacement distribution, with the region component carrying relatively small mixture weight at this observation timescale.

## B.2 �-sweep on the ScaleTraj configuration

To verify that the default � =4 choice is not narrowly tuned, we additionally train the ScaleTraj configuration (per-scale MLPs of Table 4) at $K \in \{ 5 , 6 \}$ on both datasets; only � varies, with the GMM refit at the new number of components. Table 6 reports the three scale-emergence metrics.

Table 6: �-sweep on the ScaleTraj configuration; lower is better. �=4 is the default.
<table><tr><td></td><td colspan="3">Tencent Beijing</td><td colspan="3">GeoLife</td></tr><tr><td>K</td><td>SFP↓</td><td>|∆β|↓</td><td>|∆κ|↓</td><td>SFP↓</td><td>|∆β|↓</td><td>|∆κ|↓</td></tr><tr><td>4 (default)</td><td>.003</td><td>.021</td><td>24</td><td>.003</td><td>.026</td><td>131</td></tr><tr><td>5</td><td>.007</td><td>.078</td><td>146</td><td>.002</td><td>.033</td><td>237</td></tr><tr><td>6</td><td>.004</td><td>.059</td><td>407</td><td>.004</td><td>.058</td><td>102</td></tr></table>

On Tencent Beijing �=4 wins SFP, $| \Delta \beta |$ , and |Δ�| simultaneously. On GeoLife the three configurations are clustered: � =4 wins $| \Delta \beta |$ � =5 marginally improves SFP, and � =6 marginally improves |Δ�|. We retain � =4 for its strict dominance on the larger-pool Tencent Beijing corpus.

## C Architectural Components

This appendix specifies two architectural components mentioned in §6 as implementation details rather than ablated contributions.

Jacobian correction. The term $J _ { k } ( c ) = \log \hat { \rho } _ { k } bigl ( \log d ( c _ { i - 1 } , c ) \bigr )$ in Eq. (5) is an empirical correction that accounts for the non-uniform density of candidate grid cells in log-distance space, including effects from the distance transformation and grid geometry. Here, $\hat { \rho } _ { k }$ is a Gaussian kernel density estimate (KDE) of the empirical log-distance distribution between grid cells, with bandwidth derived from $\sigma _ { k }$ and a small lower-bound safeguard. It is precomputed once at model construction and serves as a numerical safeguard against cell-density bias; we did not isolate its contribution in the experiments.

Persona-conditioned generation via user latents. For generation we attach a learnable latent code $z _ { u } \in \mathbb { R } ^ { 1 6 }$ to each training user, mixed into the anchor stream of every input token. Users are clustered into $P$ personas during preprocessing, and a learnable Gaussian prior ${ N } \big ( \mu _ { p } , \mathrm { d i a g } ( \sigma _ { p } ^ { 2 } ) \big )$ is maintained for each persona. A personaconditioned regularization term encourages each per-user code to remain close to its persona-specific prior. At generation time, all experiments replay the trained user codes so that synthetic and real populations share the same anchor distributions.