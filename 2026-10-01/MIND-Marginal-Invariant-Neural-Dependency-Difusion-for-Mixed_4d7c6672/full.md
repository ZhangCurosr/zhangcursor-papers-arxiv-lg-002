# MIND: Marginal-Invariant Neural Dependency Difusion for Mixed-Type Tabular Generation

Pengfei Li, Mohammad Khalil

Centre for the Science of Learning & Technology (SLATE), University of Bergen, Bergen, Norway

## Abstract

This paper proposes MIND, a marginal-invariant neural dependency difusion model for mixed-type tabular data. MIND does not directly learn the joint distribution in the original heterogeneous feature space. Instead, it first maps different variable types into a unified latent dependency space via column-wise marginal transport. A conditional difusion model then learns cross-column relationships. Copulatangent denoising separates known marginal components from learnable dependency residuals. Rank projection during the sampling phase further mitigates marginal shift in reverse difusion. Experiments across nine diverse tabular benchmarks show that MIND consistently improves marginal fidelity and dependency preservation over existing unified approaches. By explicitly isolating marginal modelling from dependency learning, MIND achieves a strong and stable balance among marginal fidelity, joint dependency preservation, and downstream prediction utility. This work supports sep arating marginal and dependency modelling as a principled and highly efective paradigm for complex mixed-type tabular generation.

## Introduction

Driven by the surging demand for data augmentation and privacy protection, synthetic tabular generation has become a critical research direction in machine learning (Borisov et al. 2024). The current evaluation paradigm has evolved from singular macroscopic similarity to multidimensional metrics like downstream utility, statistical fidelity, and privacy risks (Stoian, Giunchiglia, and Lukasiewicz 2026). Unlike the uniform metric space of images or the natural sequential dependencies of text, tabular data is a complex combination of multivariate heterogeneous random variables (Grinsztajn, Oyallon, and Varoquaux 2022; Borisov et al. 2024). Individual variables exhibit entirely distinct statistical behaviours (e.g., skewed continuous values, high-cardinality discrete values, or non-random missingness), and they intertwine with strong domain-specific nonlinear dependencies and explicit constraints (Xu et al. 2019; Zhao et al. 2021). Therefore, the key challenge in high-quality tabular generation lies in simultaneously achieving the statistical fidelity of single-column marginal distributions and the semantic validity of multi-column joint distributions.

Existing deep tabular generators do address feature heterogeneity, but mainly as a representation or optimisation problem rather than by separating marginal modelling from dependency learning. CTGAN and TVAE use conditional sampling, mode-specific normalisation, and reconstruction objectives to accommodate mixed-type columns (Xu et al. 2019). TabDDPM applies type-specific difusion processes to numerical and categorical variables (Kotelnikov et al. 2023), while STaSy improves score-based generation through dedicated training strategies (Kim, Lee, and Park 2023). TabSyn moves difusion into a continuous latent space learned by a VAE (Zhang et al. 2024), and TabDif introduces a joint mixed-type difusion process with feature-wise noise schedules (Shi et al. 2025). These designs substantially improve tabular synthesis, yet column marginals and cross-column dependencies are still learned through shared representations and coupled objectives.

This unified modelling paradigm requires models to balance column-level marginal distributions and cross-column joint structures under an end-to-end objective. Existing research alleviates this dificulty from various angles. For instance, CTGAN designs mode-specific normalisation and conditional sampling for multimodal continuous columns and imbalanced discrete categories. TabDDPM extends diffusion models to heterogeneous tabular data composed of continuous and discrete features. TabDif further emphasises the challenges ofcomplex inter-column correlations and fine-grained column-level distributions in mixed-type tabular generation (Xu et al. 2019; Kotelnikov et al. 2023; Shi et al. 2025). However, the fidelity of column-level marginal distributions cannot substitute for structural consistency at the pairwise, conditional, or full-joint levels (Yang et al. 2024). When models fail to capture highly skewed continuous columns, long-tail categories, or missing patterns, the generated data often weakens the coverage of low-frequency subgroups and introduces biases in marginal distributions or inter-column dependencies (Grinsztajn, Oyallon, and Varoquaux 2022; Xu et al. 2019; Yang et al. 2024; Shi et al. 2025). Therefore, we argue that the dificulty in tabular generation is not solely insuficient generator capacity. It also involves the representation-and-optimisation coupling dilemma when aligning marginal distributions and joint structures within a unified representation space.

To address these challenges, we propose a novel modelling perspective. Mixed-type tabular generation should explicitly decouple column marginal distributions and inter-column dependencies rather than fitting them simultaneously in the original heterogeneous feature space or an entangled continuous latent space. Based on this idea, we introduce MIND, the Marginal-Invariant Neural Dependency difusion model. MIND first maps continuous variables, categorical variables, and missing patterns into a unified normalised dependency space via column-wise marginal transformations. A neural difusion model then learns the complex nonlinear and highorder dependencies. Inverse transformations finally generate valid mixed-type tabular data. Isolating heterogeneous marginal processing allows MIND to focus its model capacity entirely on cross-column dependency modelling.

MIND is inspired by classic copula theory. Sklar’s theorem decomposes any multivariate joint distribution into univariate marginal distributions and a copula that describes variable dependencies (Sklar 1959; Nelsen 2006). Traditional copula models rely on predefined parametric families or complex structural selections. They struggle to capture nonlinear, high-order, and mixed-type dependencies in high-dimensional tabular data. MIND combines this classic decomposition with neural generative models. Column-wise transformations model heterogeneous marginals, and a diffusion model learns flexible dependency structures within a unified copula-normalised space. Crucially, rather than treating this mapping as a mere preprocessing step, MIND deeply customises the generative process to preserve this decoupled geometry rigorously. MIND is therefore not a simple modification of existing models. It is a marginal-invariant generation framework centred on dependency modelling. Figure 1 illustrates the MIND architecture.

The main contributions of this paper are as follows:

• We propose a marginal-invariant dependency modelling paradigm for mixed-type tabular generation, where column marginals are handled separately from cross-column dependence. MIND maps heterogeneous columns into a unified latent space through column-wise marginal transforms and learns the remaining dependency structure with a neural generative model. This decoupling allows the generation backbone to focus on statistical relationships rather than repeatedly fitting heterogeneous marginal shapes.

• We introduce a dependency difusion model incorporating copula-aware denoising in this latent space, latent alignment, and progressive marginal projection. These mechanisms capture high-order relationships while suppressing marginal shift to align dependency learning with marginal calibration.

• We systematically compare MIND against classic statistical and deep tabular generative models on multiple public mixed-type datasets. We validate the framework’s efectiveness through marginal fidelity, dependency structure fidelity, and downstream task utility.

## Related Work

## Synthetic Tabular Data Generation

Synthetic tabular data generation methods broadly cover traditional probabilistic models, deep generative models, and sequence-based modelling approaches (Stoian, Giunchiglia, and Lukasiewicz 2026). Probabilistic models like Bayesian networks and copulas characterise variable relationships via explicit structural assumptions, ofering some interpretability. However, their reliance on strong distributional or dependency assumptions often hinders them from adequately capturing complex joint distributions in high-dimensional, nonlinear, and mixed-type scenarios (Zhang et al. 2017; Patki, Wedge, and Veeramachaneni 2016; Xu et al. 2019). Subsequently, GAN and VAE-based methods are widely used for tabular synthesis. To address multimodal continuous columns and imbalanced discrete columns, CTGAN proposes designs like mode-specific normalisation and trainingby-sampling (Xu et al. 2019). Other works introduce structural or sequential inductive biases. For instance, GOGGLE explicitly learns variable relationship graphs and models inter-column dependencies via message passing (Liu et al. 2023). GReaT and REaLTabFormer serialise tabular rows into text and generate records using autoregressive language models or GPT-style Transformers (Borisov et al. 2023; Solatorio and Dupriez 2023), while TabMT models tabular feature generation with a masked Transformer (Gulati and Roysdon 2023). These methods enhance tabular generation from statistical structure, neural generation, and sequence modelling perspectives. Yet, most directly model the joint distribution in the original feature or encoding space, leaving marginal distribution fitting and inter-column dependency learning to the same generative process.

## Difusion and Score-Based Generative Models for Tabular Data

Recent tabular generators increasingly adopt difusion or score-based models. TabDDPM directly models heterogeneous numerical and categorical features (Kotelnikov et al. 2023), while STaSy improves score-based tabular generation through dedicated stabilisation strategies (Kim, Lee, and Park 2023). TabSyn performs difusion in a VAE latent space (Zhang et al. 2024), and TabDif introduces feature-wise diffusion processes for mixed-type variables (Shi et al. 2025). These methods improve generation quality but still learn marginal variation and cross-column dependence within a shared feature or latent representation. MIND instead performs difusion in a copula-normalised space, separating marginal calibration from dependency learning.

## Copula Dependency Modelling

Copula theory provides a classical statistical foundation for separating marginal distributions and dependency structures. By Sklar’s theorem, a multivariate joint distribution can be decomposed into univariate marginal distributions and a copula function (Sklar 1959; Nelsen 2006). This perspective highly aligns with the variable-level structure of tabular data, where each column usually possesses independent semantics and distribution shapes. Gaussian and vine copulas are representative methods in synthetic tabular data (Tagasovska, Ackerer, and Vatter 2019; Sun, Cuesta-Infante, and Veeramachaneni 2019). Such methods naturally distinguish column-level marginal distributions from cross-column dependencies but usually rely on predefined copula families or structure selection. Thus, they require strong modelling assumptions when facing complex nonlinear dependencies, high-dimensional relations, and mixed-type data (Sun, Cuesta-Infante, and Veeramachaneni 2019; Fan et al. 2017; Genest and Nešlehová 2007). We inherit this marginal-dependency decomposition perspective but adopt more flexible neural generative modelling within a normalised dependence space.

![](images/891dd5230074205873e35392b41f3d30c790ddef09dbb0cda561a415cba1d56c.jpg)  
Figure 1: Overview of MIND: type-aware marginal mapping, conditional dependency difusion, and calibrated inverse generation.

## Methodology

## Overview

Let $\mathcal { D } = \{ ( \mathbf { x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ be a mixed-type tabular dataset. The vector x contains continuous, categorical, ordinal, and potentially missing feature variables, while $y _ { i }$ is a designated fully observed target column. MIND addresses target-aware synthesis by first encoding each record through column-wise marginal transports:

$$
( \mathbf { z } _ { 0 } , z _ { 0 } ^ { y } ) = \mathcal { T } ( \mathbf { x } , y , \pmb { \xi } ) ,\tag{1}
$$

where ξ denotes dequantization randomness for discrete variables. The latent joint distribution is then factorised as

$$
p ( \mathbf { z } _ { 0 } , z _ { 0 } ^ { y } ) = \widehat { q } _ { y } ( z _ { 0 } ^ { y } ) p _ { \theta } ( \mathbf { z } _ { 0 } \mid z _ { 0 } ^ { y } ) ,\tag{2}
$$

where $\widehat { q } _ { y }$ is the empirical target marginal prior after transport. The transports calibrate column marginals. The conditional difusion model learns the feature dependency distribution given the sampled target.

## Marginal-Invariant Representation

For variable $j ,$ MIND constructs

$$
z _ { j } = T _ { j } ( x _ { j } , \xi _ { j } ) = \Phi ^ { - 1 } ( U _ { j } ( x _ { j } , \xi _ { j } ) ) ,\tag{3}
$$

where Φ is the standard normal cumulative distribution function and $U _ { j } \in \mathsf { \Gamma } ( 0 , 1 )$ is a type-specific empirical quantile coordinate. Continuous variables use empirical mid-ranks and monotone interpolation. Categorical, binary, and ordinal variables are assigned disjoint quantile intervals according to smoothed empirical frequencies and are dequantized within the corresponding interval. This maps heterogeneous columns to approximately Gaussian-calibrated marginals, while inverse transports return valid values in the original domain (Sklar 1959; Nelsen 2006; Dunn and Smyth 1996).

For each variable with missing values, MIND additionally introduces a binary missingness coordinate. Numerical values at missing positions are replaced by auxiliary Gaussian values and excluded from the denoising loss. Categorical missing values are represented by a dedicated token. Full transport and missingness constructions are provided in the appendix.

## Copula-Tangent Dependency Difusion

MIND difuses only non-target coordinates and keeps the target coordinate clean:

$$
\begin{array} { r } { \mathbf { z } _ { t } = a _ { t } \mathbf { z } _ { 0 } + b _ { t } \boldsymbol { \epsilon } , \qquad z _ { t } ^ { y } = z _ { 0 } ^ { y } , \qquad \boldsymbol { \epsilon } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) , } \end{array}\tag{4}
$$

where $t \in \{ 1 , \ldots , T \} , a _ { t } = \sqrt { \bar { \alpha } _ { t } } .$ , and $b _ { t } = \sqrt { 1 - \bar { \alpha } _ { t } }$ (Ho, Jain, and Abbeel 2020; Nichol and Dhariwal 2021). In the normalised latent space, the noise admits the decomposition

$$
\epsilon = b _ { t } { \bf z } _ { t } + a _ { t } { \bf r } _ { t } , \qquad { \bf r } _ { t } = \frac { \epsilon - b _ { t } { \bf z } _ { t } } { a _ { t } } .\tag{5}
$$

Under ideal standard-normal marginals, $\boldsymbol { r } _ { t , j }$ is uncorrelated with $z _ { t , j } .$ . Thus, $b _ { t } \mathbf { z } _ { t }$ is an analytically determined marginal component, and the predictable part of $\mathbf { r } _ { t }$ carries targetconditional dependency information.

Let $\mathcal { P } _ { \eta } ( \cdot , { \bf z } _ { t } , { \bf M } )$ denote the type-aware copula-tangent operator defined in the appendix. It removes masked batch means and centred latent scale directions, with a relaxed projection for dequantized discrete coordinates. Given

$$
\begin{array} { r } { { \bf u } _ { \theta } = f _ { \theta } ( { \bf z } _ { t } , t , z _ { 0 } ^ { y } ) , } \end{array}
$$

MIND uses the hybrid parameterisation

$$
\widehat { \mathbf { \epsilon } } _ { \theta } = \left\{ \begin{array} { l l } { \mathbf { u } _ { \theta } , } & { t \leq \tau _ { \mathrm { l o w } } , } \\ { b _ { t } \mathbf { z } _ { t } + a _ { t } \mathcal { P } _ { \eta } ( \mathbf { u } _ { \theta } , \mathbf { z } _ { t } , \mathbf { M } ) , } & { t > \tau _ { \mathrm { l o w } } . } \end{array} \right.\tag{6}
$$

The low-noise branch directly predicts total noise to preserve local details and discrete interval boundaries. The remaining steps predict a tangent-constrained dependency residual.

Define

$$
\mathbf { D } _ { t } = \left\{ \begin{array} { l l } { \epsilon - \mathbf { u } _ { \theta } , } & { t \leq \tau _ { \mathrm { l o w } } , } \\ { \mathcal { P } _ { \eta } ( \mathbf { r } _ { t } , \mathbf { z } _ { t } , \mathbf { M } ) - \mathcal { P } _ { \eta } ( \mathbf { u } _ { \theta } , \mathbf { z } _ { t } , \mathbf { M } ) , } & { t > \tau _ { \mathrm { l o w } } . } \end{array} \right.\tag{7}
$$

The denoising objective is

$$
\mathcal { L } _ { \mathrm { C T D } } = \mathbb { E } \left[ \frac { \lVert \mathbf { M } \odot \mathbf { D } _ { t } \rVert _ { F } ^ { 2 } } { \lVert \mathbf { M } \rVert _ { 1 } } \right] ,\tag{8}
$$

where $\mathbf { M } \in \{ 0 , 1 \} ^ { B \times d }$ masks numerical coordinates that were originally missing.

## Noise-Adaptive Dependency Network

MIND represents each latent coordinate as a column token:

$$
\mathbf { h } _ { j } ^ { ( 0 ) } = E _ { \mathrm { v a l } } ( z _ { t , j } ) + \mathbf { e } _ { j } ^ { \mathrm { c o l } } + \mathbf { e } _ { j } ^ { \mathrm { t y p e } } + \mathbf { e } _ { j } ^ { \mathrm { r o l e } } + \mathbf { e } _ { t } ^ { \mathrm { t i m e } } .\tag{9}
$$

A column-wise Transformer models interactions among these tokens (Vaswani et al. 2017).

To stabilise attention under heavy corruption, MIND interpolates between topology-based and value-based queries and keys:

$$
\begin{array} { r } { \mathbf { Q } _ { t } = \omega _ { t } \mathbf { Q } ^ { \mathrm { t o p } } + ( 1 - \omega _ { t } ) \mathbf { Q } ^ { \mathrm { v a l } } , } \\ { \mathbf { K } _ { t } = \omega _ { t } \mathbf { K } ^ { \mathrm { t o p } } + ( 1 - \omega _ { t } ) \mathbf { K } ^ { \mathrm { v a l } } . } \end{array}\tag{10}
$$

where

$$
\omega _ { t } = \frac { \sqrt { 1 - \bar { \alpha } _ { t } } } { \sqrt { 1 - \bar { \alpha } _ { T } } } .\tag{11}
$$

High-noise attention relies primarily on stable column identities. Low-noise attention increasingly uses sample-specific values. A time-dependent target-attention bias, an asymmetric condition mask, target-anchor reinjection, and a bounded column-tied readout further preserve the conditioning signal. Architectural details are given in the appendix.

## Objective and Copula-Projected Sampling

MIND reconstructs the clean latent variables as

$$
\widehat { \mathbf { z } } _ { 0 } = \frac { \mathbf { z } _ { t } - b _ { t } \widehat { \mathbf { \epsilon } } _ { \theta } } { a _ { t } } .\tag{12}
$$

It then matches normal-score correlations among generated coordinates and between generated coordinates and the target. These terms act as a marginal-invariant second-order dependency anchor. Nonlinear and higher-order structure is learned by the difusion Transformer (Liu, Laferty, and Wasserman 2009). Denoting the two terms by $\mathcal { L } _ { \mathrm { G G } }$ and $\mathcal { L } _ { \mathrm { G Y } }$ the total objective is

$$
\mathcal { L } _ { \mathrm { M I N D } } = \mathcal { L } _ { \mathrm { C T D } } + \lambda _ { \mathrm { d e p } } \left( \mathcal { L } _ { \mathrm { G G } } + \lambda _ { y } \mathcal { L } _ { \mathrm { G Y } } \right) .\tag{13}
$$

Their exact forms are provided in the appendix.

For generation, MIND samples $\begin{array} { r } { \widehat { z } ^ { y } \sim \widehat { q } _ { y } , } \end{array}$ initializes $\widetilde { \mathbf z } _ { T } \sim \breve { \mathcal { N } } ( \mathbf { 0 } , \mathbf { I } )$ , and performs target-conditional DDPM sampling. To correct accumulated marginal drift, it intermittently applies the column-wise rank projection

$$
[ \Pi _ { B } ( { \bf Z } ) ] _ { i j } = \Phi ^ { - 1 } \left( \frac { r _ { i j } } { B + 1 } \right) ,\tag{14}
$$

Algorithm 1 Training and Sampling with MIND   
Require: Training data D and difusion horizon $T$   
Ensure: Synthetic table $\widetilde { \mathcal { D } }$   
1: Fit transports $\tau$ and encode $( \mathbf { Z } _ { 0 } , \mathbf { z } _ { 0 } ^ { y } )$   
2: while not converged do   
3: Sample $( \mathbf { z } _ { 0 } , z _ { 0 } ^ { y } ) , t \in \{ 1 , . . . , T \}$ , and ϵ   
4: Form $\mathbf { z } _ { t } = a _ { t } \mathbf { z } _ { 0 } + b _ { t } \mathbf { \epsilon } ,$ and keep $z _ { t } ^ { y } = z _ { 0 } ^ { y }$   
5: Compute $\mathbf { u } _ { \theta } = f _ { \theta } ( \mathbf { z } _ { t } , t , z _ { 0 } ^ { y } )$   
6: Form $\widehat { \epsilon } _ { \theta }$ using Eq. (6)   
7: Update θ by minimizing Eq. (13)   
8: end while   
9: Sample $\widetilde { z } ^ { y } \sim \widehat { q } _ { y }$ and initialize $\widetilde { \mathbf z } _ { T } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$   
10: for $t = T , \dots , 1$ do   
11: Perform one target-conditional reverse-difusion up  
date   
12: Apply the scheduled hard or soft rank projection   
13: end for   
14: Decode $\widetilde { \mathcal { D } } = \mathcal { T } ^ { - 1 } ( \widetilde { \mathbf { z } } _ { 0 } , \widetilde { z } ^ { y } )$   
15: return $\widetilde { \mathcal { D } }$

where $r _ { i j }$ is the within-column rank of $Z _ { i j }$ . This monotone projection preserves empirical ranks while recalibrating each latent marginal. The projection schedule is detailed in the appendix.

## Experiment Setup

We evaluate MIND on six public classification benchmarks: Adult (Becker and Kohavi 1996), Default Credit Card (Yeh 2009), FICO HELOC (Fair Isaac Corporation 2018), Covertype (Blackard 1998), Online Shoppers (Sakar and Kastro 2018), and Telco Churn (IBM 2019). We further include two public regression benchmarks: Beijing PM2.5 (Chen 2015), which predicts hourly PM2.5 concentration from temporal and meteorological variables, and Online News Popularity (Fernandes et al. 2015), which predicts the number of social-media shares from article-level features. These datasets extend the evaluation to continuous targets from environmental and media domains. We also use a controlled HeavyTail stress test containing heavy-tailed numerical variables, a long-tail categorical attribute, and structured missingness. Covertype is evaluated on a class-stratified 50,000- row subset for computational tractability (Blackard 1998).

To rigorously assess generation quality, we select baselines that represent a comprehensive spectrum of direct statistical and deep generative modelling paradigms. While recent autoregressive models sequence tables into text, we focus on continuous latent and feature-space generators: an independent empirical-marginal sampler (Indep.) and Gaussian Copula (Patki, Wedge, and Veeramachaneni 2016) as classical approximations, CTGAN and TVAE (Xu et al. 2019) as standard deep tabular generators, and Tab-Syn (Zhang et al. 2024) and TabDif (Shi et al. 2025) as the recent state-of-the-art difusion-based methods. For each seed in {42, 43, 44}, all methods use the same stratified $6 4 \% / 1 6 \% / 2 0 \%$ train/validation/test split and generate $| \mathcal { D } _ { \mathrm { t r a i n } } |$ synthetic records.

We assess marginal fidelity using the Kolmogorov–

Smirnov statistic (Massey 1951), Wasserstein distance (Villani 2009), total variation, and Jensen–Shannon divergence (Lin 1991); dependency preservation using Pearson/Spearman correlation-matrix and mutual-information errors (Cover and Thomas 2006); and predictive utility using train-synthetic-test-real (TSTR) AUC, accuracy, and macro-F1 (Esteban, Hyland, and Rätsch 2017). Distributional fidelity and coverage are further evaluated by α-Precision and β-Recall (Alaa et al. 2022), while real–synthetic distinguishability is measured by classifier two-sample-test (C2ST) AUC (Lopez-Paz and Oquab 2017) and propensityscore mean squared error (pMSE) (Snoke et al. 2018). Results are reported as mean ± standard deviation over three seeds. Complete implementation and evaluation details are provided in the supplementary.

## Results

## Overall Generation Quality

Table 1 summarises marginal fidelity, dependency preservation, distinguishability, and support coverage over nine datasets. MIND records the best mean on six of the ten metrics. Its KS, JS, and Column JS errors are 0.003, 0.005, and 0.009. Relative to the next lowest means, these errors fall by 62.5%, 58.3%, and 40.0%.

MIND also obtains the lowest Pairwise MI error at 0.008 and the lowest C2ST gap at 0.167, compared with 0.214 for TabDif. Its Pearson error is 0.016, behind TabDif at 0.014 but slightly ahead of TabSyn at 0.017. Support coverage remains high, with both α-Precision and β-Recall at 0.981. TVAE is best on precision, while TabDif is best on recall. The independent model gives the lowest pMSE, yet its TSTR score and dependency errors are much worse. This contrast shows that pMSE should be read together with utility and dependency measures.

## Downstream Utility

Table 2 reports TSTR utility on seven classification datasets and two regression datasets. MIND achieves the highest average score of 0.693, exceeding TVAE, TabDif, and TabSyn by 0.046, 0.059, and 0.117. On Beijing, TabDif leads with 0.560, while MIND reaches 0.525 and remains close to Tab-Syn at 0.529. News is more dificult. Every method has a negative mean R<sup>2</sup>, with CTGAN best at −0.038 and MIND at −0.137.

On the classification datasets, MIND ranks first on Covertype, Default, and FICO, and second on Telco. Its largest gain appears on Covertype, where it reaches 0.940 compared with 0.882 for TVAE. On Adult, HeavyTail, and Shoppers, the gaps to the best method are 0.011, 0.014, and 0.021. The gap on Telco is 0.003. Additionally, MIND shows better cross-seed stability than Tabdif and Tabsyn.

## Dependency Preservation

Figure 2 further illustrates error distributions across variable pairs. The independent marginal model forms large high-error areas across most datasets. This shows that accurate univariate recovery cannot reconstruct joint structures.

![](images/48cecc44556b0781370fecc0ab470322adf86d8f4e1a932dda855f2e30bc2661.jpg)  
Figure 2: Pairwise Pearson correlation errors across datasets. Lighter colours indicate more accurate dependency preservation.

Gaussian Copula improves significantly but still leaves concentrated error blocks in FICO, Covertype, and Shoppers. This reflects the limits of fixed dependency families on complex mixed-type relationships. CTGAN and TVAE errors show strong dataset dependence with noticeable deviations in certain pairs.

TabSyn, TabDif, and MIND exhibit lighter overall error distributions. MIND specifically reduces locally concentrated high errors in Default, FICO, and Telco. It achieves balanced dependency recovery across diferent pairs. Meanwhile, TabDif retains a slightly lower overall Pearson error. This aligns with the aggregated results in Table 1. Thus, the heatmap illustrates that MIND avoids severe distortion in specific local dependencies rather than strictly outperforming TabDif on all pairs.

## Marginal Distribution Analysis

Figure 3 compares the density estimates of six representative continuous variables. MIND successfully recovers the sharp main peak of Hours/week in Adult, the broad peak and right shoulder of Install burden in FICO, and the asymmetric peak of Elevation in Covertype. In contrast, CTGAN exhibits varying degrees of peak shift on these variables. TabSyn and TabDif fit well overall but still show deviations on certain narrow peaks or multi-scale structures. TabSyn, TabDif, and CTGAN all hallucinate a sharp spurious mode around 65–75 that doesn’t exist in the real density, while MIND tracks the true flat/bimodal shape.

<table><tr><td colspan="10"></td></tr><tr><td>Metric</td><td>Indep.</td><td>G-Copula</td><td>CTGAN</td><td>TVAE</td><td>TabSyn</td><td>TabDiff</td><td>MIND</td><td>Sig. VS.</td></tr><tr><td>TSTR score ↑</td><td> $0 . 3 4 4 { \pm } 0 . 2 9 3$ </td><td> $0 . 6 2 0 { \scriptstyle \pm 0 . 3 2 2 }$ </td><td> $0 . 5 8 6 { \scriptstyle \pm 0 . 3 1 7 }$ </td><td> $0 . 6 4 7 { \scriptstyle \pm 0 . 3 4 5 }$ </td><td> $0 . 5 7 6 { \pm } 0 . 5 4 5$ </td><td> $0 . 6 3 4 { \scriptstyle \pm 0 . 4 0 6 }$ </td><td> $\mathbf { 0 . 6 9 3 { \scriptstyle \pm 0 . 3 3 5 } }$ </td><td>I</td></tr><tr><td>KS↓</td><td> $0 . 0 0 8 { \pm } 0 . 0 0 5$ </td><td> $0 . 0 0 8 { \pm } 0 . 0 0 5$ </td><td> $0 . 1 4 7 { \scriptstyle \pm 0 . 0 5 9 }$ </td><td> $0 . 1 1 8 { \pm } 0 . 0 3 9$ </td><td> $0 . 0 3 3 { \scriptstyle \pm 0 . 0 2 2 }$ </td><td> $0 . 0 2 8 { \pm } 0 . 0 2 8$ </td><td> $\mathbf { 0 . 0 0 3 \pm 0 . 0 0 5 }$ </td><td>C,V,S,D</td></tr><tr><td>JS↓</td><td> $0 . 0 1 2 { \scriptstyle \pm 0 . 0 1 4 }$ </td><td> $0 . 0 1 2 { \pm } 0 . 0 1 4$ </td><td> $0 . 0 8 0 { \pm } 0 . 0 2 9$ </td><td> $0 . 1 1 9 { \pm } 0 . 0 8 0$ </td><td> $0 . 0 2 7 { \scriptstyle \pm 0 . 0 1 7 }$ </td><td> $0 . 0 1 9 { \pm } 0 . 0 1 3$ </td><td> $\mathbf { 0 . 0 0 5 } \pm \mathbf { 0 . 0 1 } 2$ </td><td>C,V,S,D</td></tr><tr><td>Column JS ↓</td><td> $0 . 0 1 5 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $0 . 0 1 7 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $0 . 1 0 1 { \pm } 0 . 0 3 5$ </td><td> $0 . 1 1 8 { \pm } 0 . 0 5 0$ </td><td> $0 . 0 5 0 { \scriptstyle \pm 0 . 0 3 3 }$ </td><td> $0 . 0 4 3 { \pm } 0 . 0 3 9$ </td><td> $\mathbf { 0 . 0 0 9 } \pm \mathbf { 0 . 0 0 9 }$ </td><td>C,V,S,D</td></tr><tr><td>Pearson err. ↓</td><td> $0 . 1 0 4 { \pm } 0 . 0 4 6$ </td><td> $0 . 0 3 7 { \pm } 0 . 0 1 3$ </td><td> $0 . 0 6 4 { \pm } 0 . 0 2 3$ </td><td> $0 . 0 5 9 { \pm } 0 . 0 2 9$ </td><td> $0 . 0 1 7 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $\mathbf { 0 . 0 1 4 } \pm \mathbf { 0 . 0 0 6 }$ </td><td> $0 . 0 1 6 { \pm } 0 . 0 0 5$ </td><td>I,G,C,V</td></tr><tr><td>Pairwise MI err. ↓</td><td> $0 . 0 6 7 { \scriptstyle \pm 0 . 0 5 8 }$ </td><td> $0 . 0 4 5 { \pm } 0 . 0 4 0$ </td><td> $0 . 0 3 3 { \pm } 0 . 0 1 8$ </td><td> $0 . 0 3 3 { \scriptstyle \pm 0 . 0 2 2 }$ </td><td> $0 . 0 1 0 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 0 0 9 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $\mathbf { 0 . 0 0 8 { \scriptstyle \pm 0 . 0 0 5 } }$ </td><td>I,G,C,V</td></tr><tr><td>C2ST gap ↓</td><td> $0 . 4 5 0 { \scriptstyle \pm 0 . 1 1 2 }$ </td><td> $0 . 4 1 2 { \scriptstyle \pm 0 . 1 4 9 }$ </td><td> $0 . 4 6 2 { \scriptstyle \pm 0 . 0 5 4 }$ </td><td> $0 . 4 6 6 { \pm } 0 . 0 2 9$ </td><td> $0 . 2 3 8 { \pm } 0 . 1 7 6$ </td><td> $0 . 2 1 4 { \pm } 0 . 1 9 2$ </td><td> $\mathbf { 0 . 1 6 7 { \pm 0 . 1 2 0 } }$ </td><td>I,G,C,V</td></tr><tr><td>pMSE↓</td><td> $\mathbf { 0 . 0 0 3 \pm 0 . 0 0 4 }$   $0 . 7 2 7 { \scriptstyle \pm 0 . 2 6 2 }$ </td><td> $0 . 0 0 4 { \scriptstyle \pm 0 . 0 0 4 }$   $0 . 8 6 2 { \scriptstyle \pm 0 . 1 6 7 }$ </td><td> $0 . 0 5 2 { \pm } 0 . 0 2 8$   $0 . 8 8 1 { \scriptstyle \pm 0 . 1 6 3 }$ </td><td> $0 . 0 8 8 { \pm } 0 . 0 7 0$   $\mathbf { 0 . 9 8 8 { \scriptstyle \pm 0 . 0 1 4 } }$ </td><td> $0 . 0 1 9 { \pm } 0 . 0 3 0$   $0 . 9 8 1 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $0 . 0 2 0 { \scriptstyle \pm 0 . 0 3 2 }$   $0 . 9 8 3 { \pm } 0 . 0 1 0$ </td><td> $0 . 0 0 3 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td>C,V,S,D</td></tr><tr><td>α-Precision ↑</td><td> $0 . 8 3 9 { \pm } 0 . 1 6 5$ </td><td> $0 . 9 2 6 { \pm } 0 . 0 7 9$ </td><td> $0 . 9 2 7 { \scriptstyle \pm 0 . 0 7 4 }$ </td><td> $0 . 9 4 4 { \pm } 0 . 0 1 8$ </td><td> $0 . 9 7 8 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $\mathbf { 0 . 9 8 3 } \pm \mathbf { 0 . 0 0 8 }$ </td><td> $0 . 9 8 1 { \scriptstyle \pm 0 . 0 0 6 }$   $0 . 9 8 1 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td>I,G,C</td></tr><tr><td>β-Recall ↑</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>I,G,C,V</td></tr></table>

Table 1: Aggregate generation quality across nine datasets. Bold denotes the best mean in each row. The “Sig. vs.” column lists baselines significantly outperformed by MIND under two-sided paired Wilcoxon signed-rank tests on nine dataset-level seed means, with Holm correction over six comparisons within each metric $( p _ { \mathrm { a d j } } < 0 . 0 5 )$ . I denotes Indep., G denotes G-Copula, C denotes CTGAN, V denotes TVAE, S denotes TabSyn, and D denotes TabDif.
<table><tr><td>Dataset</td><td>Indep.</td><td>G-Copula</td><td>CTGAN</td><td>TVAE</td><td> $\operatorname { T a b } S \mathbf { y } \mathbf { n }$ </td><td>TabDiff</td><td>MIND</td></tr><tr><td>Adult</td><td> $0 . 5 0 0 \pm 0 . 0 3 0$ </td><td> $0 . 7 9 2 \pm 0 . 0 1 2$ </td><td> $0 . 8 8 6 \pm 0 . 0 0 1$ </td><td> $0 . 8 8 4 \pm 0 . 0 0 5$ </td><td> $0 . 9 0 5 \pm 0 . 0 0 4$ </td><td> $\mathbf { 0 . 9 1 0 \pm 0 . 0 0 4 }$ </td><td> $0 . 8 9 9 \pm 0 . 0 0 3$ </td></tr><tr><td> $\mathrm { B e i j i n g ^ { * } }$ </td><td> $- 0 . 0 0 1 \pm 0 . 0 0 3$ </td><td> $0 . 2 3 3 \pm 0 . 0 3 1$ </td><td> $0 . 1 6 6 \pm 0 . 0 1 0$ </td><td> $0 . 1 8 6 \pm 0 . 1 0 8$ </td><td> $0 . 5 2 9 \pm 0 . 0 2 2$ </td><td> $\mathbf { 0 . 5 6 0 \pm 0 . 0 3 3 }$ </td><td> $0 . 5 2 5 \pm 0 . 0 2 8$ </td></tr><tr><td>Covertype</td><td> $0 . 5 0 2 \pm 0 . 0 4 7$ </td><td> $0 . 7 2 1 \pm 0 . 0 1 3$ </td><td> $0 . 6 2 5 \pm 0 . 0 6 9$ </td><td> $0 . 8 8 2 \pm 0 . 0 1 0$ </td><td> $0 . 5 6 5 \pm 0 . 0 1 0$ </td><td> $0 . 5 9 4 \pm 0 . 0 0 4$ </td><td> ${ \bf 0 . 9 4 0 \pm 0 . 0 0 1 }$ </td></tr><tr><td>Default</td><td> $0 . 4 7 1 \pm 0 . 0 0 8$ </td><td> $0 . 6 9 3 \pm 0 . 0 0 7$ </td><td> $0 . 7 2 1 \pm 0 . 0 1 6$ </td><td> $0 . 7 3 4 \pm 0 . 0 1 1$ </td><td> $0 . 7 5 3 \pm 0 . 0 1 0$ </td><td> $0 . 7 5 0 \pm 0 . 0 0 9$ </td><td> $\mathbf { 0 . 7 5 6 \pm 0 . 0 0 8 }$ </td></tr><tr><td>FICO</td><td> $0 . 5 2 9 \pm 0 . 0 1 6$ </td><td> $0 . 7 7 9 \pm 0 . 0 0 4$ </td><td> $0 . 6 3 8 \pm 0 . 0 5 6$ </td><td> $0 . 7 8 5 \pm 0 . 0 0 6$ </td><td> $0 . 7 8 4 \pm 0 . 0 0 7$ </td><td> $0 . 7 8 5 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 7 8 8 \pm 0 . 0 0 4 }$ </td></tr><tr><td>HeavyTail</td><td> $0 . 4 5 2 \pm 0 . 0 6 4$ </td><td> $0 . 7 1 8 \pm 0 . 0 1 5$ </td><td> $0 . 6 0 7 \pm 0 . 0 5 1$ </td><td> $0 . 7 2 6 \pm 0 . 0 1 5$ </td><td> $0 . 7 2 7 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 7 3 7 \pm 0 . 0 0 5 }$ </td><td> $0 . 7 2 3 \pm 0 . 0 2 1$ </td></tr><tr><td> ${ \mathrm { N e w s } } ^ { * }$ </td><td> $- 0 . 3 0 7 \pm 0 . 4 6 4$ </td><td> $- 0 . 0 7 6 \pm 0 . 0 9 5$ </td><td> ${ \bf - 0 . 0 3 8 \pm 0 . 0 8 2 }$ </td><td> $- 0 . 0 7 0 \pm 0 . 0 7 7$ </td><td> $- 0 . 8 3 4 \pm 0 . 7 3 0$ </td><td> $- 0 . 3 9 6 \pm 0 . 4 4 5$ </td><td> $- 0 . 1 3 7 \pm 0 . 2 1 4$ </td></tr><tr><td>Shoppers</td><td> $0 . 4 6 7 \pm 0 . 1 2 7$ </td><td> $0 . 8 8 1 \pm 0 . 0 1 4$ </td><td> $0 . 8 4 6 \pm 0 . 0 0 7$ </td><td> $0 . 8 7 4 \pm 0 . 0 1 6$ </td><td> $0 . 9 1 5 \pm 0 . 0 0 6$ </td><td> $\mathbf { 0 . 9 2 1 } \pm \mathbf { 0 . 0 0 8 }$ </td><td> $0 . 9 0 0 \pm 0 . 0 0 8$ </td></tr><tr><td>Telco</td><td> $0 . 4 8 3 \pm 0 . 0 7 8$ </td><td> $0 . 8 3 5 \pm 0 . 0 1 1$ </td><td> $0 . 8 1 9 \pm 0 . 0 1 0$ </td><td> $0 . 8 2 1 \pm 0 . 0 2 6$ </td><td> $0 . 8 4 1 \pm 0 . 0 1 2$ </td><td> $\mathbf { 0 . 8 4 6 \pm 0 . 0 1 6 }$ </td><td> $0 . 8 4 3 \pm 0 . 0 1 9$ </td></tr><tr><td>Avg.</td><td> $0 . 3 4 4 \pm 0 . 2 9 3$ </td><td> $0 . 6 2 0 \pm 0 . 3 2 2$ </td><td> $0 . 5 8 6 \pm 0 . 3 1 7$ </td><td> $0 . 6 4 7 \pm 0 . 3 4 5$ </td><td> $0 . 5 7 6 \pm 0 . 5 4 5$ </td><td> $0 . 6 3 4 \pm 0 . 4 0 6$ </td><td> $\mathbf { 0 . 6 9 3 \pm 0 . 3 3 5 }$ </td></tr></table>

Table 2: Per-dataset utility under the train-on-synthetic, test-on-real protocol. Scores are averaged over XGBoost, LightGBM, and MLP, using AUC for classification datasets and $R ^ { 2 }$ for regression datasets (marked with <sup>∗</sup>). Entries report mean ± standard deviation over three seeds. The Avg. row averages dataset-level means. Higher is better, and bold denotes the best mean in each row.

This diference becomes more pronounced on challenging distributions. For HeavyTail, MIND accurately recovers the centre, peak width, and right-tail decay of the true distribution. Conversely, CTGAN produces a significantly rightshifted and over-dispersed density. For the highly skewed Shoppers PageValues and the long-tailed Telco Total charges, MIND preserves the high-density region near zero and the tail decay as values increase.

Pairwise MI error from 0.0082 to 0.0089 and slightly reduces the TSTR score. This confirms that explicitly regularising normal-score correlations not only stabilises dependency learning but also yields better multivariate mutual information and downstream utility. Overall, the full MIND achieves the best performance in marginal fidelity, pairwise dependency preservation, and distinguishability, while maintaining highly competitive downstream utility and coverage.

## Discussion

## Ablation Study

Table 3 shows that copula projection is the key component for maintaining marginal fidelity. Removing this module increases Column JS from 0.0135 to 0.0432, an approximate 3.2-fold increase. Removing CTD, dependency regularisation, or attention enhancements has a minor impact on coverage, but each removal raises the C2ST gap from 0.140 to about 0.144. This indicates that these components jointly improve the realism of the overall joint distribution. Notably, removing the correlation regularisation increases the

Compared with the strongest baselines in our experiments, MIND does not dominate every metric or dataset. MIND leads on Covertype, Default, and FICO, and obtains lower average marginal, mutual-information, and C2ST errors. Although the aggregate TSTR mean favours MIND, this diference is influenced substantially by Covertype. We therefore interpret the results as showing that MIND is competitive with difusion-based SOTA. Separating marginals from dependence also has a clear precedent in Gaussian and vine copula synthesis (Patki, Wedge, and Veeramachaneni 2016; Sun, Cuesta-Infante, and Veeramachaneni 2019). MIND differs from these methods by replacing a fixed copula family with target-conditional neural difusion in a normalised dependence space, while correcting marginal drift during sampling.

<table><tr><td>Variant</td><td>TSTR ↑</td><td>Column JS↓</td><td>Pairwise MI error ↓</td><td>C2ST gap ↓</td><td>β-Recall ↑</td></tr><tr><td>Full MIND</td><td> $0 . 8 4 4 \pm 0 . 0 8 5$ </td><td> $\mathbf { 0 . 0 1 3 5 \pm 0 . 0 1 0 9 }$ </td><td> $\mathbf { 0 . 0 0 8 2 \pm 0 . 0 0 4 1 }$ </td><td> ${ \bf 0 . 1 4 0 \pm 0 . 1 0 4 }$ </td><td> $0 . 9 8 0 1 \pm 0 . 0 0 2 4$ </td></tr><tr><td>w/o CTD</td><td> $0 . 8 4 3 \pm 0 . 0 8 7$ </td><td> $0 . 0 1 3 6 \pm 0 . 0 1 0 9$ </td><td> $0 . 0 0 9 0 \pm 0 . 0 0 3 8$ </td><td> $0 . 1 4 4 \pm 0 . 1 0 3$ </td><td> $0 . 9 7 9 1 \pm 0 . 0 0 2 3$ </td></tr><tr><td>w/o Corr.</td><td> $0 . 8 4 1 \pm 0 . 0 8 9$ </td><td> $0 . 0 1 3 6 \pm 0 . 0 1 0 9$ </td><td> $0 . 0 0 8 9 \pm 0 . 0 0 3 8$ </td><td> $0 . 1 4 4 \pm 0 . 1 0 4$ </td><td> $0 . 9 7 9 7 \pm 0 . 0 0 2 3$ </td></tr><tr><td>w/o Attn. Extras</td><td> $\mathbf { 0 . 8 4 5 \pm 0 . 0 8 2 }$ </td><td> $0 . 0 1 3 6 \pm 0 . 0 1 0 9$ </td><td> $0 . 0 0 8 9 \pm 0 . 0 0 3 8$ </td><td> $0 . 1 4 4 \pm 0 . 1 0 0$ </td><td> $0 . 9 7 9 5 \pm 0 . 0 0 3 2$ </td></tr><tr><td>w/o Projection</td><td> $0 . 8 4 4 \pm 0 . 0 8 9$ </td><td> $0 . 0 4 3 2 \pm 0 . 0 0 8 4$ </td><td> $0 . 0 0 9 1 \pm 0 . 0 0 3 1$ </td><td> $0 . 1 4 6 \pm 0 . 1 1 5$ </td><td> $\mathbf { 0 . 9 8 1 3 \pm 0 . 0 0 2 0 }$ </td></tr></table>

Table 3: Component ablation of MIND over three datasets and three seeds. Entries report mean ± standard deviation over nine runs. Higher is better for TSTR and β-Recall, while lower is better for the remaining metrics. Bold denotes the best mean in each column.

![](images/bfdb2badc15fdc456fcd2ead8c6e10e6a7bfa9ed87ede0c26a7ba003949844cc.jpg)  
Figure 3: Single-column density overlays for representative continuous variables (seed 42). The real distribution is shown in black. Selected neural baselines are compared with MIND, and each x-axis is clipped to the range between the real-data 1st and 99th percentiles for readability.

Covertype illustrates the practical efect of this design. TabSyn and TabDif obtain TSTR scores of 0.565 and 0.594, compared with 0.940 for MIND. Although TabDif achieves a lower pairwise Pearson error, the Covertype stress-case analysis shows that it has a substantially larger C2ST gap and a pronounced shift in target-class mass (Supplementary Table 6). Class conditioning is also considered by CTGAN and TabDDPM, while TabSyn models the targetjointly with other columns in a learned latent space (Xu et al. 2019; Kotelnikov et al. 2023; Zhang et al. 2024). MIND instead samples the target from its empirical marginal prior and learns $p _ { \boldsymbol { \theta } } ( \mathbf { x } \mid \boldsymbol { y } )$ The difusion model therefore does not need to reconstruct class proportions, which helps account for its lower targetmarginal error on Covertype.

The independent marginal model obtains low pMSE and small marginal errors, yet performs poorly in TSTR and dependency preservation. On Covertype, TabDif achieves a lower Pearson error than MIND but substantially worse TSTR and C2ST results. Matching linear normal-score dependence therefore does not by itself ensure downstream utility or overall distributional similarity. MIND consequently uses normal-score correlation only as an auxiliary regularizer, while the difusion model learns broader dependency structures. This evaluation across complementary fidelity and utility criteria follows recent systematic frameworks for synthetic tabular data assessment (Du and Li 2025; Yang et al. 2024).

The ablation results provide the clearest evidence for the sampling projection. Removing it increases Column JS from 0.0135 to 0.0432, corresponding to an approximately threefold degradation in marginal fidelity. Removing CTD, correlation regularisation, or the attention additions changes the C2ST gap from 0.140 to between 0.144 and 0.146. These diferences are small relative to the reported variation and do not establish interaction efects among the components. Removing correlation regularisation also changes Pairwise MI error from 0.0082 to 0.0089 and TSTR from 0.844 to 0.841. The ablation therefore strongly supports the role of projection in marginal calibration, while the aggregate evidence for the remaining components is more modest.

MIND comes with limitations. Currently, we assume a designated target column and perform batch-level generation, which limits direct use in target-free or multi-target settings. Its rank-based calibration also depends on suficiently large generation batches. Future work will extend the framework to more flexible conditioning schemes, batch-independent sampling, and privacy-aware training, while preserving the separation between marginal modelling and dependency learning.

## Conclusion

This paper proposes MIND, a marginal-invariant neural dependency difusion model for mixed-type tabular data. MIND maps heterogeneous variables into a unified latent space through column-wise marginal transport, allowing conditional difusion to focus on cross-column dependencies. Copula-tangent denoising and rank projection further reduce marginal drift during generation.

Across diverse datasets, MIND achieves a strong balance among marginal fidelity, dependency preservation, and downstream utility. These results support explicit marginaldependency decoupling as a practical design principle for mixed-type tabular generation.

## References

Alaa, A.; van Breugel, B.; Saveliev, E. S.; and van der Schaar, M. 2022. How Faithful Is Your Synthetic Data? Sample-Level Metrics for Evaluating and Auditing Generative Models. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, 290–306. PMLR.

Becker, B.; and Kohavi, R. 1996. Adult. UCI Machine Learning Repository. [Dataset].

Blackard, J. 1998. Covertype. UCI Machine Learning Repository. [Dataset].

Borisov, V.; Leemann, T.; Seßler, K.; Haug, J.; Pawelczyk, M.; and Kasneci, G. 2024. Deep Neural Networks and Tabular Data: A Survey. IEEE Transactions on Neural Networks and Learning Systems, 35(6): 7499–7519.

Borisov, V.; Seßler, K.; Leemann, T.; Pawelczyk, M.; and Kasneci, G. 2023. Language Models Are Realistic Tabular Data Generators. In The Eleventh International Conference on Learning Representations.

Chen, S. 2015. Beijing PM2.5.

Cover, T. M.; and Thomas, J. A. 2006. Elements ofInformation Theory. John Wiley & Sons, 2 edition.

Du, Y.; and Li, N. 2025. Systematic assessment of tabular data synthesis. In Proceedings of the 2025 ACM SIGSAC Conference on Computer and Communications Security, 2414–2428.

Dunn, P. K.; and Smyth, G. K. 1996. Randomized Quantile Residuals. Journal of Computational and Graphical Statistics, 5(3): 236–244.

Esteban, C.; Hyland, S. L.; and Rätsch, G. 2017. Real-Valued (Medical) Time Series Generation with Recurrent Conditional GANs. arXiv preprint arXiv:1706.02633.

Fair Isaac Corporation. 2018. FICO Explainable Machine Learning Challenge: HELOC Dataset. FICO Explainable Machine Learning Challenge. [Dataset]. A commonly used mirror is OpenML dataset 46932.

Fan, J.; Liu, H.; Ning, Y.; and Zou, H. 2017. High-Dimensional Semiparametric Latent Graphical Model for Mixed Data. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 79(2): 405–421.

Fernandes, K.; Vinagre, P.; Cortez, P.; and Sernadela, P. 2015. Online News Popularity.

Genest, C.; and Nešlehová, J. 2007. A Primer on Copulas for Count Data. ASTIN Bulletin, 37(2): 475–515.

Grinsztajn, L.; Oyallon, E.; and Varoquaux, G. 2022. Why Do Tree-Based Models Still Outperform Deep Learning on Typical Tabular Data? In Advances in Neural Information Processing Systems, volume 35, 507–520.

Gulati, M. S.; and Roysdon, P. F. 2023. TabMT: Generating Tabular Data with Masked Transformers. In Advances in Neural Information Processing Systems, volume 36, 46245– 46254.

Ho, J.; Jain, A. N.; and Abbeel, P. 2020. Denoising Difusion Probabilistic Models. In Advances in Neural Information Processing Systems, volume 33, 6840–6851.

IBM. 2019. Telco Customer Churn. IBM Cognos Analytics Sample Data. [Dataset].

Kim, J.; Lee, C.; and Park, N. 2023. STaSy: Score-Based Tabular Data Synthesis. In The Eleventh International Conference on Learning Representations.

Kotelnikov, A.; Baranchuk, D.; Rubachev, I.; and Babenko, A. 2023. TabDDPM: Modelling Tabular Data with Difusion Models. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, 17564–17579. PMLR.

Lin, J. 1991. Divergence Measures Based on the Shannon Entropy. IEEE Transactions on Information Theory, 37(1): 145–151.

Liu, H.; Laferty, J.; and Wasserman, L. 2009. The Nonparanormal: Semiparametric Estimation of High-Dimensional Undirected Graphs. Journal of Machine Learning Research, 10: 2295–2328.

Liu, T.; Qian, Z.; Berrevoets, J.; and van der Schaar, M. 2023. GOGGLE: Generative Modelling for Tabular Data by Learning Relational Structure. In The Eleventh International Conference on Learning Representations.

Lopez-Paz, D.; and Oquab, M. 2017. Revisiting Classifier Two-Sample Tests. In International Conference on Learning Representations.

Massey, F. J., Jr. 1951. The Kolmogorov–Smirnov Test for Goodness of Fit. Journal ofthe American Statistical Association, 46(253): 68–78.

Nelsen, R. B. 2006. An Introduction to Copulas. Springer Series in Statistics. Springer New York, 2 edition.

Nichol, A. Q.; and Dhariwal, P. 2021. Improved Denoising Difusion Probabilistic Models. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, 8162–8171. PMLR.

Patki, N.; Wedge, R.; and Veeramachaneni, K. 2016. The Synthetic Data Vault. In 2016 IEEE International Conference on Data Science and Advanced Analytics, 399–410. IEEE.

Sakar, C.; and Kastro, Y. 2018. Online Shoppers Purchasing Intention Dataset. UCI Machine Learning Repository. [Dataset].

Shi, J.; Xu, M.; Hua, H.; Zhang, H.; Ermon, S.; and Leskovec, J. 2025. TabDif: A Mixed-Type Difusion Model for Tabular Data Generation. In The Thirteenth International Conference on Learning Representations.

Sklar, A. 1959. Fonctions de répartition à n dimensions et leurs marges. Publications de l’Institut de Statistique de l’Université de Paris, 8(3): 229–231.

Snoke, J.; Raab, G. M.; Nowok, B.; Dibben, C.; and Slavković, A. 2018. General and Specific Utility Measures for Synthetic Data. Journal of the Royal Statistical Society: Series A (Statistics in Society), 181(3): 663–688.

Solatorio, A. V.; and Dupriez, O. 2023. REaLTabFormer: Generating Realistic Relational and Tabular Data Using Transformers. arXiv preprint arXiv:2302.02041.

Stoian, M. C.; Giunchiglia, E.; and Lukasiewicz, T. 2026. A Survey on Deep Learning Approaches for Tabular Data Generation: Utility, Alignment, Fidelity, Privacy, Diversity, and Beyond. Transactions on Machine Learning Research.

Sun, Y.; Cuesta-Infante, A.; and Veeramachaneni, K. 2019. Learning Vine Copula Models for Synthetic Data Generation. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01): 5049–5057.

Tagasovska, N.; Ackerer, D.; and Vatter, T. 2019. Copulas as High-Dimensional Generative Models: Vine Copula Autoencoders. In Advances in Neural Information Processing Systems, volume 32, 6525–6537.

Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017. Attention Is All You Need. In Advances in Neural Information Processing Systems, volume 30, 5998–6008.

Villani, C. 2009. Optimal Transport: Old and New, volume 338 of Grundlehren der mathematischen Wissenschaften. Springer Berlin, Heidelberg.

Xu, L.; Skoularidou, M.; Cuesta-Infante, A.; and Veeramachaneni, K. 2019. Modeling Tabular Data Using Conditional GAN. In Advances in Neural Information Processing Systems, volume 32, 7333–7343.

Yang, S. C.-H.; Eaves, B.; Schmidt, M.; Swanson, K.; and Shafto, P. 2024. Structured Evaluation of Synthetic Tabular Data. arXiv preprint arXiv:2403.10424.

Yeh, I.-C. 2009. Default ofCredit Card Clients. UCI Machine Learning Repository. [Dataset].

Zhang, H.; Zhang, J.; Shen, Z.; Srinivasan, B.; Qin, X.; Faloutsos, C.; Rangwala, H.; and Karypis, G. 2024. Mixed-Type Tabular Data Synthesis with Score-Based Difusion in Latent Space. In The Twelfth International Conference on Learning Representations.

Zhang, J.; Cormode, G.; Procopiuc, C. M.; Srivastava, D.; and Xiao, X. 2017. PrivBayes: Private Data Release via Bayesian Networks. ACM Transactions on Database Systems, 42(4): 25:1–25:41.

Zhao, Z.; Kunar, A.; Birke, R.; and Chen, L. Y. 2021. CTAB-GAN: Efective Table Data Synthesizing. In Proceedings of the 13th Asian Conference on Machine Learning, volume 157 of Proceedings of Machine Learning Research, 97–112. PMLR.