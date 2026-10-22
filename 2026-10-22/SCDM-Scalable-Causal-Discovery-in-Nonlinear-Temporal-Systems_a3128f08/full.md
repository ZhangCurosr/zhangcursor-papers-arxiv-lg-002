# SCDM: Scalable Causal Discovery in Nonlinear Temporal Systems with Meta-Learning

Jingbo Wang Kegeng Tang Zihao Wang Shaogang Ren<sup>∗</sup> University of Tennessee at Chattanooga {ydk297,hcw575,zihao.wang,sren9}@tennessee.edu

## Abstract

Causal discovery from nonlinear multivariate time series is challenging in highdimensional systems, where the number of candidate directed relations grows quadratically with the number of variables. Existing methods are often limited by target-wise model fitting, repeated conditional testing, or expensive graph-search procedures. We propose SCDM, a shared-parameter meta-learning framework for scalable temporal causal discovery. SCDM treats each target variable as a task while learning a shared temporal predictor and a shared causal-strength matrix across tasks. Each meta-episode updates only a subset of target-variable tasks, allowing the model to accumulate structural evidence across the full system without exhaustive target-wise optimization. After training, a post-training graph readout converts the learned predictor into continuous directed causal scores for thresholdindependent evaluation. We also provide a finite-error recovery principle showing that, under fully observed delayed-system assumptions and a positive risk-gap condition, thresholding SCDM scores recovers the graph when statistical, metaoptimization, and readout errors are below the population margin. Experiments on controlled synthetic, neuroimaging-inspired, high-dimensional, and real-world temporal benchmarks demonstrate competitive causal recovery and improved scalability in large temporal systems.

Project page: https://shaogangren.github.io/SCDM/ Code: https://github.com/ShaogangRen/SCDM

## 1 Introduction

Causal discovery from multivariate time series seeks directed dependencies in systems such as brain activity, financial markets, and traffic networks. Temporal ordering supports predictive causal analysis through the Granger framework [Granger, 1969, Lütkepohl, 2005]. For d observed variables, however, the candidate cross-variable edge space contains d(d − 1) ordered pairs, so nonlinear prediction, target-wise fitting, and repeated conditional tests become costly in high-dimensional systems [Runge et al., 2019, Runge, 2020].

Existing methods address parts of this challenge through structured sparsity, temporal convolutions, self-explaining predictors, perturbation-based scores, or coarse-to-fine discovery [Tank et al., 2021, Nauta et al., 2019, Marcinkevics and Vogt, 2021, Bi et al., 2025, Cheng et al., 2024]. Many of themˇ still require target-specific optimization or repeated graph evaluation, which becomes limiting when the edge space contains millions of directed pairs. The question is therefore how to share nonlinear prediction across targets while keeping target-specific structural evidence.

We propose Scalable Causal Discovery with Meta-Learning (SCDM), a shared-parameter framework for nonlinear temporal causal discovery. Inspired by gradient-based meta-learning [Finn et al., 2017,

Nichol et al., 2018, Yang et al., 2022], SCDM treats the prediction of each target variable as a related task in the same dynamical system. It learns a shared nonlinear temporal predictor and a shared causal-strength matrix. Each meta-episode samples a small batch of target-variable tasks, performs temporary support adaptation, and updates the shared parameters through query losses. Repeated over episodes, this procedure accumulates structural evidence across variables while keeping each update tied to the sampled task batch rather than the full set of targets. This target-task subsampled meta-training is the main scalability mechanism of SCDM.

SCDM separates training from graph extraction. After meta-training, a post-training perturbationstyle readout, related to recent perturbation-based dynamic causal scoring [Bi et al., 2025], converts the trained predictor into continuous directed source–target scores. This readout is a graph-extraction interface rather than the main training contribution. We also provide a finite-error recovery principle for the fully observed delayed temporal setting: if statistical, meta-optimization, and readout errors are smaller than the population risk-gap margin, thresholding the learned scores separates causal parents from non-parents.

We evaluate SCDM on controlled nonlinear benchmarks with known ground-truth graphs, The Virtual Brain (TVB)-inspired neuroimaging simulations [Sanz Leon et al., 2013], a high-dimensional voxel-level functional magnetic resonance imaging (fMRI) benchmark, and the real-world LargeST traffic benchmark [Liu et al., 2023]. On controlled benchmarks, SCDM is competitive with strong baselines rather than uniformly best. Its main advantage appears at large scale: it gives a better runtime–accuracy tradeoff on 1,000 variables and finishes on 3,000 variables with far less time and memory. Two ablations separate the roles of the two training components: target-task subsampling lowers training cost without hurting recovery, and the support/query meta-objective yields shared parameters that adapt better to held-out targets than pooled multi-task training.

The main contributions are:

• We formulate nonlinear temporal causal discovery as a shared-parameter episodic metalearning problem. Each target variable defines one prediction task, and all tasks share one temporal predictor and one causal-strength matrix, so no separate model is fitted for each target.

• We separate two training components with different roles. Target-task subsampling sets how many target tasks each update touches (B instead of d), which controls training cost. The support/query objective trains the shared parameters to perform well after a few targetspecific steps, which pooled multi-task training does not do. We test each component with its own ablation.

• We give a finite-error recovery result for fully observed delayed temporal systems, and evaluate SCDM on controlled, neuroimaging-inspired, voxel-level (d = 1000), and realworld (d = 3000) benchmarks, together with stress tests for collinear sources and withheld variables.

## 2 Related Work

Temporal causal discovery. Temporal causal discovery exploits lagged dependencies to infer directed relations in multivariate time series. Classical Granger causality defines directed influence by whether one variable’s history improves another variable’s prediction [Granger, 1969], with VAR models providing a standard linear implementation [Lütkepohl, 2005]. VARLiNGAM further combines vector autoregression with non-Gaussianity assumptions for structural identification [Hyvärinen et al., 2010]. Constraint-based methods such as PCMCI and PCMCI+ use conditional independence and momentary conditional independence tests for autocorrelated time series [Runge et al., 2019, Runge, 2020], with related work studying time-series-specific assumptions and conditional independence structures [Entner and Hoyer, 2010, Peters et al., 2014, Malinsky and Spirtes, 2018]. These methods provide important foundations, but their linear, component-wise, or repeated-testing structures can become costly in nonlinear high-dimensional systems where the candidate edge space scales as d(d − 1). SCDM keeps the predictive interpretation of temporal causality while replacing repeated target-wise fitting or testing with a shared meta-learned temporal predictor and causal-strength matrix.

Neural temporal causal discovery. Neural methods improve nonlinear modeling capacity by replacing linear predictors with expressive sequence models. Neural Granger causality uses componentwise predictors with structured sparsity [Tank et al., 2021]; GVAR uses self-explaining neural networks for nonlinear and time-varying Granger effects [Marcinkevics and Vogt, 2021]; and TCDFˇ combines temporal convolutions, attention-based interpretation, and causal validation [Nauta et al., 2019]. More general neural causal discovery methods use differentiable graph learning, continuous optimization, and score-based discovery for nonlinear systems [Zheng et al., 2018, Kalainathan et al., 2018, Ng et al., 2022, Zhang et al., 2024, 2025]. Recent perturbation-based methods such as UnCLe estimate dynamic causal effects through uncoupled temporal modeling and perturbationbased causal scoring [Bi et al., 2025], related to occlusion-style attribution under input removal or perturbation [Zeiler and Fergus, 2014, Ribeiro et al., 2016, Shrikumar et al., 2017]. SCDM is related to these methods through nonlinear prediction and continuous graph scoring, but uses perturbation-style scoring only as a post-training readout; its main scalability mechanism is shared episodic meta-training across target-variable tasks.

Causal inference scalability and meta-learning. Existing scalable strategies include sparsity regularization and fast sparse solvers [Ren et al., 2020, Ren and Qian, 2024], variable-wise decomposition, screening [Ren et al., 2017, Ren, 2017], and coarse-to-fine search; for example, neural Granger methods use structured sparsity [Tank et al., 2021], while CUTS+ targets high-dimensional irregular time series through coarse-to-fine discovery and message-passing prediction modules [Cheng et al., 2024]. SCDM instead formulates target-variable prediction as a family of related tasks. Inspired by MAML and first-order meta-learning [Finn et al., 2017, Nichol et al., 2018], and related meta-learning applications in few-shot learning, optimization, and structured prediction [Vinyals et al., 2016, Ravi and Larochelle, 2017, Yao et al., 2019], SCDM samples a subset of target-variable tasks per episode and updates a shared temporal predictor together with a shared causal-strength matrix through query losses. This support/query formulation is used not for few-shot classification, but as a scalable mechanism for accumulating structural evidence across variables without updating all target tasks at every iteration.

## 3 Problem Formulation

Problem setup and target-variable tasks. Let $X = \{ x _ { t } \} _ { t = 1 } ^ { T }$ be a d-dimensional time series, with $\boldsymbol { x } _ { t } \in \mathbb { R } ^ { d }$ . For observed-history order $p ,$ define $u _ { t } = [ x _ { t - 1 } , \ldots , x _ { t - p } ] \in \mathbb { R } ^ { d p }$ for $t = p + 1 , . . . , T$ Task i predicts the target $y _ { t } ^ { ( i ) } = x _ { t } ^ { ( i ) }$ from this history. Following support/query meta-learning [Finn et al., 2017], the d target variables define d related tasks within the same dynamical process.

All tasks share the parameters $\Theta = \{ \theta , W \}$ , where θ parameterizes a nonlinear temporal predictor and $W \in \mathbb { R } ^ { d \times d }$ is a causal-strength matrix with $W _ { j , i }$ the strength from source j to target i. Each task starts from this shared initialization and adapts only a temporary local copy; no private target-specific head, embedding, or persistent adapted model is introduced. Causal structure is therefore represented once, globally, by $\bar { W } _ { } ^ { \bar { } }$ , and scalability comes from updating only sampled target tasks.

Figure 1 summarizes the shared-parameter episodic pipeline, and Table 5 at the start of the appendix lists all notation.

Structured predictor over observed variables. For the fully observed temporal system considered in the main method, the lagged history of each source variable j is modulated by a shared variablelevel structural weight $W _ { j , i }$ when predicting target variable i. Specifically, the weighted lagged history $\widetilde { H } _ { t } ^ { ( i ) } \in \mathbb { R } ^ { d \times p }$ has entries $\widetilde { H } _ { t , j , \ell } ^ { ( i ) } = W _ { j , i } x _ { t - \ell } ^ { ( j ) }$ for $j = 1 , \ldots , d$ and $\ell = 1 , \ldots , p .$ . The weighted history is flattened and passed to the shared predictor as $\widehat { x } _ { t } ^ { ( i ) } = f _ { \theta } ( \mathrm { v e c } ( \widetilde { H } _ { t } ^ { ( i ) } ) )$ . Since $W _ { j , i }$ gates all lagged coordinates of source $j$ when predicting target i, the learned matrix W provides a shared variable-level directed causal-strength representation, which is later converted into continuous graph scores by the post-training readout.

## 4 Methodology

## 4.1 Meta-Learning for Causal Discovery in Nonlinear Temporal Systems

SCDM alternates temporary target-task adaptation with persistent shared updates. Each episode samples a target-task batch B and support/query windows for its tasks. Starting from the shared

![](images/4db2c28ffbc1ad1678cd56ac11fb3c2fd8ea341b0e4ec5304cf52d591c05be42.jpg)  
Figure 1: Overview of SCDM. Each target variable is treated as a task; sampled tasks undergo temporary support adaptation, and their query losses update the shared temporal predictor and causalstrength matrix.

initialization $\Theta ,$ each task performs support updates and evaluates the adapted parameters on its query window. The averaged query loss then updates the persistent shared parameters, and the adapted task copies are discarded. The objectives below specify this procedure.

For each target variable i, let $\mathcal { D } _ { i } ^ { \mathrm { s u p } }$ and $\mathcal { D } _ { i } ^ { \mathrm { q r y } }$ denote support and query temporal windows, and let $\Theta = \{ \theta , W \}$ be the shared parameters. The support objective is

$$
\mathcal { L } _ { i } ^ { \mathrm { s u p } } ( \boldsymbol { \Theta } ) = \frac { 1 } { | \mathcal { D } _ { i } ^ { \mathrm { s u p } } | } \sum _ { ( \boldsymbol { u } _ { t } , \boldsymbol { y } _ { t } ^ { ( i ) } ) \in \mathcal { D } _ { i } ^ { \mathrm { s u p } } } \log \left( f _ { \boldsymbol { \theta } } \left( \mathrm { v e c } \left( \widetilde { \boldsymbol { H } } _ { t } ^ { ( i ) } \right) \right) , \boldsymbol { y } _ { t } ^ { ( i ) } \right) + \lambda _ { W } \| \boldsymbol { W } [ : , i ] \| _ { 1 } .\tag{1}
$$

Here $\widetilde { H } _ { t } ^ { ( i ) }$ is the weighted lagged history for target $i ,$ constructed using the source-to-target weights $W _ { j , i }$ . Starting from the shared initialization, task i performs K temporary support updates:

$$
\Theta _ { i } ^ { ( 0 ) } \gets \Theta , \qquad \Theta _ { i } ^ { ( k + 1 ) } = \Theta _ { i } ^ { ( k ) } - \alpha \nabla _ { \Theta _ { i } ^ { ( k ) } } \mathcal { L } _ { i } ^ { \mathrm { s u p } } ( \Theta _ { i } ^ { ( k ) } ) , \quad k = 0 , \dots , K - 1 .\tag{2}
$$

The adapted parameters $\Theta _ { i } ^ { \prime } = \Theta _ { i } ^ { ( K ) }$ are used only within the current episode; they are not stored as target-specific models. The query loss is

$$
\mathcal { L } _ { i } ^ { \mathrm { q r y } } ( \Theta _ { i } ^ { \prime } ) = \frac { 1 } { | \mathcal { D } _ { i } ^ { \mathrm { q r y } } | } \sum _ { \left( u _ { t } , y _ { t } ^ { ( i ) } \right) \in \mathcal { D } _ { i } ^ { \mathrm { q r y } } } \log \left( f _ { \theta _ { i } ^ { \prime } } \left( \mathrm { v e c } \left( \widetilde { H } _ { t } ^ { ( i ) } ( \Theta _ { i } ^ { \prime } ) \right) \right) , y _ { t } ^ { ( i ) } \right) + \lambda _ { W } \| W _ { i } ^ { \prime } [ : , i ] \| _ { 1 } .\tag{3}
$$

The $\ell _ { 1 }$ terms in (1) and (3) push the weights of uninformative sources toward zero [Tibshirani, 1996], which matches the expectation that each target has only a few true parents. For a sampled target-task mini-batch $B { \mathrm { , } }$ we define the meta-objective as:

$$
\mathcal { L } _ { \mathrm { m e t a } } = \frac { 1 } { \vert \boldsymbol { B } \vert } \sum _ { i \in \boldsymbol { B } } \mathcal { L } _ { i } ^ { \mathrm { q r y } } ( \Theta _ { i } ^ { \prime } ) .\tag{4}
$$

Algorithm 1 SCDM meta-training for scalable temporal causal discovery   
Require: Time series $\overline { { X _ { 1 : T } } }$ , lag order p, task batch size B, inner steps K   
1: Initialize shared parameters $\Theta = \{ \theta , W \}$   
2: repeat   
3: Sample target-task batch $B \subset \{ 1 , \ldots , d \}$   
4: for $i \in \boldsymbol { B }$ do   
5: Sample support/query windows $\mathcal { D } _ { i } ^ { \mathrm { s u p } } , \mathcal { D } _ { i } ^ { \mathrm { q r y } }$   
6: Adapt temporarily: $\varTheta _ { i } ^ { \prime } \gets \mathrm { A d a p t } _ { K } ^ { \prime } ( \not \Theta , \mathring { D } _ { i } ^ { \mathrm { s u p } } )$   
7: Compute query loss $\dot { \mathcal { L } } _ { i } ^ { \mathrm { q r y } } ( \Theta _ { i } ^ { \prime } )$   
8: end for   
9: Meta-update shared Θ using averaged query loss over $\boldsymbol { B }$   
10: until convergence   
11: return learned shared predictor $f _ { \theta }$ and causal-strength matrix $W$

Only the query objective updates the persistent shared parameters; the task-local adapted parameters are discarded after the episode.

We use a first-order MAML approximation [Finn et al., 2017, Nichol et al., 2018],

$$
\nabla _ { \Theta } \mathcal { L } _ { i } ^ { \mathrm { q r y } } ( \Theta _ { i } ^ { \prime } ) \approx \nabla _ { \Theta _ { i } ^ { \prime } } \mathcal { L } _ { i } ^ { \mathrm { q r y } } ( \Theta _ { i } ^ { \prime } ) , \qquad \Theta \gets \Theta - \beta \nabla _ { \Theta } \mathcal { L } _ { \mathrm { m e t a } } .\tag{5}
$$

This first-order update avoids differentiation through the inner optimization trajectory; its cost still grows with the predictor input dimension dp and with the number of sampled targets $B .$ . The two parts of this procedure play different roles. Target-task subsampling sets how many tasks each update touches, which controls training cost. The support/query split makes each query loss measure how well the shared parameters perform after K target-specific steps, so the outer update favors parameters that adapt well to every target rather than parameters that only fit the average target. Section 5.5 tests the two roles separately. Both the support and query losses use the same sparsity coefficient $\lambda _ { W }$

Algorithm 1 summarizes the training procedure. Task sampling reduces the number of target losses processed in each episode, while the cost of each predictor evaluation and the storage of $W$ still grow with d; Appendix B.1 separates training, storage, and readout costs.

## 4.2 Post-training Graph Score Readout

After meta-training, SCDM converts the learned shared predictor and causal-strength parameterization into directed source–target scores through a post-training graph readout. This readout is not part of the optimization objective; it is the graph-extraction interface for the trained meta-predictor. We use two readouts. The source-isolated readout below measures how much each candidate source alone helps the trained predictor. The structural readout uses $| W _ { j , i } |$ directly; it needs no extra model evaluation, which makes it the only affordable choice when d $\geq 1 0 0 0$ . Each main-text recovery table states its readout, and Appendix B.3 compares both on the same trained models.

The source-isolated readout is defined as follows. For a candidate source variable j and target variable i, we retain only the lagged history of source $j$ and set all other source histories to zero. Let $L _ { \mathrm { z e r } } ^ { ( i ) }$ o denote the prediction loss under a zero-input baseline, and let $L _ { j } ^ { ( i ) }$ denote the loss when only source j is retained. The directed score is

$$
s _ { j  i } = \frac { L _ { \mathrm { z e r o } } ^ { ( i ) } - L _ { j } ^ { ( i ) } } { L _ { \mathrm { z e r o } } ^ { ( i ) } + \epsilon } ,\tag{6}
$$

where $\epsilon > 0$ is a small numerical constant. Larger values indicate that source $j$ yields a larger reduction in prediction loss for target i, and are therefore interpreted as stronger directed temporal influence. This readout is related to perturbation-based temporal causal scoring [Bi et al., 2025], but the main scalability mechanism of SCDM is the shared episodic meta-training procedure. Full implementation details are provided in Appendix B.1. Both readouts measure predictive contribution rather than the effect of an intervention. When two sources carry nearly the same information, the predictor can rely on either one, so the score of an individual edge may move to a correlated non-parent; Appendix B.3 measures this effect with a controlled collinearity test.

## 4.3 Finite-Error Graph Recovery

We analyze the fully observed non-latent setting to characterize when the scores produced by SCDM recover the delayed causal graph. The analysis follows the standard risk-gap view of temporal causal discovery and is stated under delayed causal sufficiency, correct lag specification, and sufficient regularity of the time-series process. Under these conditions, thresholding the learned scores recovers the graph whenever the statistical, meta-optimization, and readout errors are smalle than the population separation margin.

For each target variable i, SCDM treats one-step prediction of $X _ { t } ^ { ( i ) }$ as a task. Let $H _ { t } \in \mathbb { R } ^ { d \times p }$ be the full lagged history and $H _ { t , - j }$ the history with all lags of source $j$ removed. Define the population full risk $\begin{array} { r } { R _ { i } ^ { \star } = \operatorname* { i n f } _ { f \in \mathcal { F } _ { i } } \mathbb { E } [ \mathrm { l o s s } ( f ( H _ { t } ) , X _ { t } ^ { ( i ) } ) ] } \end{array}$ and the source-restricted risk $R _ { i , - j } ^ { \star } ~ =$ $\begin{array} { r } { \operatorname* { i n f } _ { f \in \mathscr { F } _ { i , - j } } \mathbb { E } [ \log \mathscr { s } ( f ( H _ { t , - j } ) , X _ { t } ^ { ( i ) } ) ] } \end{array}$ ], where $\mathcal { F } _ { i , - j }$ excludes predictors using source $j ^ { \circ } \mathrm { s }$ lagged history. The population score is $S _ { i i } ^ { \star } = \bar { R } _ { i , - i } ^ { \star } - R _ { i } ^ { \star }$ , measuring the optimal risk increase for target task i when source j is removed. We assume a positive delayed-system margin

$$
\Delta _ { \operatorname* { m i n } } = \operatorname* { m i n } _ { ( j , i ) : j \in \mathrm { P a } ( i ) } S _ { j i } ^ { \star } > 0 , \qquad S _ { j i } ^ { \star } = 0 \mathrm { f o r } j \notin \mathrm { P a } ( i ) .\tag{7}
$$

In SCDM, the practical score is obtained after a finite meta-training procedure and a post-training readout. Let $\widetilde { S } _ { j i , T , N }$ denote this score after observing $T$ time points and running $N$ outer metaupdates with task batch size B and K inner steps. Its deviation from the population risk-gap score can be decomposed as

$$
\operatorname* { m a x } _ { i , j } \left. \widetilde { S } _ { j i , T , N } - S _ { j i } ^ { \star } \right. \le \epsilon _ { \mathrm { s t a t } } ( T , d , \delta ) + \epsilon _ { \mathrm { m e t a } } ( N , B , K ) + \epsilon _ { \mathrm { r e a d o u t } } ( T , N ) .\tag{8}
$$

Here $\epsilon _ { \mathrm { s t a t } }$ is the finite-sample error under temporal dependence; $\epsilon _ { \mathrm { m e t a } }$ is the SCDM-specific residual from finite episodic meta-training, task subsampling, support adaptation, first-order meta-updates, and shared query-loss optimization; and $\epsilon _ { \mathrm { r e a d o u t } }$ is the gap between the implemented readout and the ideal empirical source-ablation score.

Theorem 4.1 (Finite-error recovery of SCDM scores). Consider afully observed delayed temporal system satisfying causal sufficiency, correct lag specification, stationarity with sufficient mixing, and the population margin condition above. Suppose that, with probability at least $1 - \delta ,$ , the SCDM score satisfies Eq. (8). If

$$
\epsilon _ { \mathrm { s t a t } } ( T , d , \delta ) + \epsilon _ { \mathrm { m e t a } } ( N , B , K ) + \epsilon _ { \mathrm { r e a d o u t } } ( T , N ) < \frac { \Delta _ { \operatorname* { m i n } } } { 2 } ,\tag{9}
$$

then thresholding $\widetilde { S } _ { j i , T , N }$ at any $\begin{array} { r c l } { \gamma } & { \in } & { \left( \epsilon _ { \mathrm { t o t } } , \Delta _ { \mathrm { m i n } } - \epsilon _ { \mathrm { t o t } } \right) } \end{array}$ , where $\begin{array} { l c l } { { \epsilon _ { \mathrm { t o t } } } } & { { = } } & { { \epsilon _ { \mathrm { s t a t } } ( T , d , \delta ) + } } \end{array}$ $\epsilon _ { \mathrm { m e t a } } ( N , B , K ) + \epsilon _ { \mathrm { r e a d o u t } } ( T , N )$ , recovers the true delayed graph with probability at least $1 - \delta .$ . In particular, $\gamma = \Delta _ { \mathrm { m i n } } / 2$ is valid.

Theorem 4.1 separates identifiability from finite-error estimation. The margin $\Delta _ { \mathrm { m i n } }$ belongs to the delayed data-generating process, while SCDM enters through the meta-optimization and readout residuals. Task-subsampled meta-training is the estimation procedure, and correct recovery still requires these residuals to be small. The theorem gives no comparative guarantee between metatraining and other training objectives; Section 5.5 compares them empirically. The proof and statistical-rate instantiation are given in Appendix A.

Causal sufficiency is the assumption most likely to fail in practice. If an unobserved variable drives both j and i, the history of j carries information about that hidden driver, so removing j raises the risk of target i even when $j \not \in \mathrm { P a } ( i )$ . Then $S _ { j i } ^ { \star } > 0$ for a non-edge, the margin condition fails, and the theorem gives no guarantee for that pair. When the structural readout $| W _ { j , i } |$ is used, $\epsilon _ { \mathrm { r e a d o u t } }$ is the gap between $\vert W _ { j , i } \vert$ and the ablation score, which we do not bound; for that readout the theorem states a condition rather than a guarantee.

## 5 Experiments

## 5.1 Experimental Setup

Datasets. Our benchmarks serve three evaluation roles. NC8, simulated EEG, and ROI-level simulated fMRI provide known graphs for controlled recovery evaluation. The voxel-level fMRI simulation provides a larger known graph for runtime and recovery studies; it also contains latent regional drivers. LargeST provides real traffic data for computational feasibility, forecasting, and spatial-plausibility diagnostics, without complete directed causal ground truth. Dataset statistics and split protocols appear in Appendix B.2.

Table 1: Causal discovery performance on low- and medium-dimensional temporal benchmarks with known ground-truth graphs. We report AUROC and AUPRC as threshold-independent metrics. Results are shown as mean ± 95% confidence interval. Best results are shown in bold and second-best results are underlined.
<table><tr><td rowspan="2">Method</td><td colspan="2">NC8</td><td colspan="2">Simulated EEG</td><td colspan="2">Simulated fMRI</td></tr><tr><td>AUROC↑</td><td>AUPRC ↑</td><td> $\mathbf { A U R O C } \uparrow$ </td><td>AUPRC ↑</td><td>AUROC↑</td><td>AUPRC↑</td></tr><tr><td>UnCLe</td><td> $\mathbf { 0 . 9 4 7 \pm 0 . 0 1 4 }$ </td><td> $\mathbf { 0 . 7 5 6 \pm 0 . 0 8 0 }$ </td><td> $0 . 5 7 1 \pm 0 . 0 0 8$ </td><td> $0 . 1 5 5 \pm 0 . 0 0 8$ </td><td> $\underline { { 0 . 6 6 1 \pm 0 . 0 1 2 } }$ </td><td> $0 . 2 6 8 \pm 0 . 0 0 6$ </td></tr><tr><td>PCMCI</td><td> $0 . 9 0 9 \pm 0 . 0 0 3$ </td><td> $0 . 5 9 4 \pm 0 . 0 2 2$ </td><td> $\underline { { 0 . 7 4 3 \pm 0 . 0 1 2 } }$ </td><td> $0 . 3 9 0 \pm 0 . 0 1 8$ </td><td> $\overline { { 0 . 5 6 9 \pm 0 . 0 1 2 } }$ </td><td> $0 . 1 6 0 \pm 0 . 0 1 0$ </td></tr><tr><td>TCDF</td><td> $0 . 7 4 0 \pm 0 . 0 1 9$ </td><td> $0 . 4 1 5 \pm 0 . 0 2 0$ </td><td> $\overline { { 0 . 6 8 5 \pm 0 . 0 0 5 } }$ </td><td> $\overline { { 0 . 2 2 4 \pm 0 . 0 0 7 } }$ </td><td> $0 . 5 8 5 \pm 0 . 0 1 9$ </td><td> $0 . 1 7 8 \pm 0 . 0 2 0$ </td></tr><tr><td>GVAR</td><td> $0 . 7 5 0 \pm 0 . 0 1 3$ </td><td> $0 . 4 1 3 \pm 0 . 0 4 1$ </td><td> $\mathbf { 0 . 7 5 0 \pm 0 . 0 0 4 }$ </td><td> $0 . 3 4 4 \pm 0 . 0 1 9$ </td><td> $0 . 6 1 9 \pm 0 . 0 1 7$ </td><td> $0 . 2 1 9 \pm 0 . 0 1 9$ </td></tr><tr><td>CUTS+</td><td> $0 . 7 9 5 \pm 0 . 0 2 5$ </td><td> $0 . 3 5 4 \pm 0 . 0 3 5$ </td><td> $0 . 5 8 9 \pm 0 . 0 1 3$ </td><td> $0 . 1 6 0 \pm 0 . 0 0 7$ </td><td> $0 . 5 3 2 \pm 0 . 0 0 9$ </td><td> $0 . 1 4 1 \pm 0 . 0 0 9$ </td></tr><tr><td>cMLP</td><td> $0 . 7 5 3 \pm 0 . 0 0 2$ </td><td> $0 . 5 2 4 \pm 0 . 0 0 6$ </td><td> $0 . 6 5 6 \pm 0 . 0 1 4$ </td><td> $0 . 2 5 8 \pm 0 . 0 2 4$ </td><td> $0 . 6 2 4 \pm 0 . 0 1 3$ </td><td> $0 . 2 4 0 \pm 0 . 0 2 0$ </td></tr><tr><td>Ours</td><td> $\underline { { 0 . 9 1 6 \pm 0 . 0 1 5 } }$ </td><td> $\underline { { 0 . 7 0 4 \pm 0 . 0 3 5 } }$ </td><td> $0 . 7 1 7 \pm 0 . 0 3 7$ </td><td> $\mathbf { 0 . 4 7 6 \pm 0 . 0 6 7 }$ </td><td> $\mathbf { 0 . 8 6 7 \pm 0 . 0 0 6 }$ </td><td> ${ \bf 0 . 7 2 8 \pm 0 . 0 1 9 }$ </td></tr></table>

Baselines. We compare six temporal discovery methods. UnCLe uses uncoupled temporal modeling and perturbation-based scoring [Bi et al., 2025]; PCMCI uses conditional-independence testing [Runge et al., 2019]; TCDF combines temporal convolution, attention, and validation [Nauta et al., 2019]. GVAR uses self-explaining autoregressive coefficients [Marcinkevics and Vogt, 2021],ˇ cMLP uses structured sparsity in component-wise predictors [Tank et al., 2021], and CUTS+ uses coarse-to-fine graph discovery for high-dimensional series [Cheng et al., 2024]. Methods define temporal context differently (an observed-history order, a latent autoregressive order, or a convolution kernel); Table 6 lists each setting and how it was chosen, and Appendix B.1 gives all configurations. The 1,000-variable study compares against UnCLe and CUTS+, which output continuous scores and save intermediate checkpoints and can therefore be evaluated at matched wall-clock budgets (Appendix B.3).

Evaluation metrics. For datasets with known ground-truth graphs, we evaluate causal discovery using AUROC and AUPRC as threshold-independent metrics. We additionally report F1 and SHD for thresholded graphs in the appendix. Unless otherwise stated, self-edges are excluded from graph-level evaluation because they correspond to autoregressive self-dependencies rather than cross-variable causal discovery. For real-world LargeST experiments, where a definitive causal graph is unavailable, we report runtime, peak memory, forecasting error, and qualitative learned-graph visualizations rather than causal AUROC or AUPRC.

Runtime and implementation protocol. Runtime is measured as wall-clock time under a fixed hardware and software environment. Unless otherwise stated, runtime includes model training and causal score extraction, but excludes dataset preprocessing and figure generation. For highdimensional experiments, we additionally record peak GPU memory when available. A method is marked as OOM if it fails under the specified memory budget, and as timeout if it does not complete within the fixed wall-clock limit. Detailed hyperparameters, compute resources, thresholding rules, and runtime measurement details are provided in Appendix B.1 and Appendix B.3.

## 5.2 Causal Recovery on Controlled Benchmarks

Table 1 shows competitive but not uniformly superior recovery. On NC8, SCDM ranks second to UnCLe in both AUROC and AUPRC. On simulated EEG, it achieves the highest AUPRC, whereas GVAR and PCMCI achieve higher AUROC. On simulated fMRI, SCDM achieves the highest AUROC and AUPRC among the compared methods. The comparison shows that sharing one predictor and one causal-strength matrix across targets does not cost recovery accuracy at moderate scale; the gains of SCDM appear at larger scale (Sections 5.3 and 5.4).

(b) Ground-truth causal graph  
(a) Atlas-based ROI layout  
![](images/2d900ea2ccac4b2de5f59b6353779e64d143aeb2043a64523c21edcefa2f2da7.jpg)

![](images/31365e7463d27528a1e336cf699c3eb65516e0569f09b3c61d2985be6211950f.jpg)

![](images/bcb1fe733c0b9883008f3b2e7f08791fc804e28b6d67f082a44c7dddee23771c.jpg)

(d) Representative recovered interactions  
![](images/2bb47a5f9170175889c3439e55ce41f71630ca6c7adb08537381fb7d89f2f964.jpg)  
Figure 2: Qualitative causal recovery on the atlas-grounded simulated fMRI benchmark. (a) Schaefer-100 ROI layout with Yeo-network labels. (b) Network-sorted ground-truth directed graph. (c) Continuous causal scores recovered by SCDM, shown in the same source–target ordering as (b). (d) Representative high-confidence recovered interactions mapped back to ROI space.

Table 2: Fixed-budget results on voxel-level fMRI (d = 1000; mean over three seeds). Each entry is AUROC / AUPRC enrichment over random (random AUPRC = 0.0071). At each budget we evaluate the latest checkpoint finished within that budget, so equal values at consecutive budgets mean that no new checkpoint finished in between. SCDM uses the structural readout $| W _ { j , i } |$ . Best value per budget in bold.
<table><tr><td>Method</td><td>200 s</td><td>400 s</td><td>600 s</td><td>800 s</td></tr><tr><td>UnCLe (lag 1)</td><td>0.573 / 2.05×</td><td>0.556 /1.85×</td><td>0.556 /1.85×</td><td>0.541/ 1.53×</td></tr><tr><td>CUTS+ (L = 3)</td><td>0.571/1.31×</td><td>0.563 / 1.27×</td><td>0.577 / 1.33×</td><td>0.586 /1.36×</td></tr><tr><td>SCDM</td><td>0.566 / 1.96×</td><td>0.566 /1.96×</td><td>0.605 / 2.47×</td><td>0.645 / 4.01×</td></tr></table>

Figure 2 shows the ground truth and the recovered scores in the same network-sorted source–target order, so one can check whether the recovered scores keep the network-organized structure of the simulation; panel (d) maps representative recovered interactions back to ROI space.

## 5.3 High-Dimensional Scalability on Synthetic Voxel-level fMRI

To test scalability beyond ROI-level benchmarks, we evaluate SCDM on a synthetic voxel-level fMRI benchmark with d = 1000 observed variables, 999,000 off-diagonal candidate directed relations, and 7,142 true edges (density 0.71%). Its known observed graph allows recovery evaluation at this scale. The simulator also includes latent regional drivers, so this experiment is an empirical scalability test rather than a direct test of the fully observed recovery theorem. Because a random ranking has an AUPRC equal to the edge density, we report AUPRC as enrichment over random. We compare against UnCLe and CUTS+, which output continuous scores and save intermediate checkpoints, at matched wall-clock budgets.

Table 2 shows that SCDM improves steadily with the budget and reaches 0.645 AUROC and 4.01× enrichment at 800 seconds, compared with 0.586 and 1.36× for CUTS+, which is designed for high-dimensional causal discovery, and 0.541 and 1.53× for UnCLe. At 200 seconds UnCLe is slightly ahead on both metrics and CUTS+ on AUROC; from 400 seconds on, SCDM leads on both metrics, and its margin grows with the budget. At this scale we use the structural readout, because the source-isolated readout requires $O ( d ^ { 2 } )$ evaluations of the trained predictor (Appendix B.1). Appendix B.3 plots the same results and explains why the other baselines do not fit this fixed-budget protocol.

![](images/7b3e1a658ab421d307c88ee6f368ec76fdb246f50269ad10737d1974d8975778.jpg)

![](images/f111c220997bd994b2c4783d3ce08018fdedf645116fdef2c47a101e5274e374.jpg)

![](images/5150d11ebf3ab2bfd7e1dd676d8c291093d640a55509a6e93bead8e4e18c5975.jpg)  
Figure 3: Spatial-plausibility diagnostic on LargeST-GLA. (a) Locations of the sensors used in this experiment; this panel shows geography only. (b) Representative high-scoring directed relations from SCDM. Each arrow points from the inferred source sensor to the inferred target sensor; only relations shorter than 10 km are drawn to reduce clutter, and arrow curvature is for readability only. (c) Distribution of the geographic distance between the two sensors of each top-scoring relation. The median distance falls from about 26 km for random sensor pairs to 9 km after 200 training steps and 2.9 km after 1,000 steps, so high-scoring relations become more local as training proceeds.

Table 3: Feasibility on LargeST-GLA-3000 (3,000 sensors, 8,997,000 candidate pairs). One A100 80 GB GPU per run, 80-minute limit, three seeds. A run counts as completed only if it writes directed scores for all candidate pairs. Time to outcome is reported as mean ± standard deviation (SD) across three seeds.
<table><tr><td>Method</td><td>Outcome</td><td>Time to outcome (s)</td><td>Peak GPU (GB)</td></tr><tr><td>SCDM</td><td>Completed</td><td> $5 6 . 9 6 \pm 4 . 2 3$ </td><td>6.9</td></tr><tr><td>UnCLe</td><td>Out of memory</td><td> $3 4 9 . 2 5 \pm 1 3 . 3 0$ </td><td>15.6</td></tr><tr><td>CUTS+</td><td>Time limit</td><td> $> 4 8 0 0$ </td><td>72.7</td></tr><tr><td>JRNGC</td><td>Time limit</td><td> $> 4 8 0 0$ </td><td>14.3</td></tr><tr><td>cMLP</td><td>Time limit</td><td> $> 4 8 0 0$ </td><td>38.1</td></tr></table>

## 5.4 Real-World LargeST Scalability

We evaluate SCDM on LargeST as a real-world traffic benchmark for scalability, forecasting consistency, and qualitative structural plausibility. Since LargeST lacks a complete directed causal ground-truth graph, we do not claim exact causal recovery or report causal AUROC/AUPRC.

Figure 3 shows that the high-scoring directed relations are spatially coherent: as training proceeds, they concentrate on nearby freeway sensors instead of arbitrary long-range pairs. This is a spatialplausibility check, not evidence that individual edges are causal.

Table 3 tests whether each method can run on the 3,000-sensor subset, which has nearly nine million candidate pairs. SCDM finishes training and graph extraction in 56.96 s with 6.9 GB peak GPU memory. UnCLe writes an intermediate checkpoint after its first training phase and then runs out of memory in its joint phase; CUTS+, JRNGC [Zhou et al., 2024], and cMLP start normally but do not write a completed score matrix within 80 minutes. This test measures whether a method can run at this scale, not causal accuracy. Appendix B.3 adds a graph-conditioned forecasting diagnostic on LargeST-GLA-1500 (Table 11).

Table 4: Task-batch-size ablation on the 100-variable simulated fMRI benchmark. All variants use the full method with the structural readout and differ only in the number of target tasks per meta-episode. Results are mean ± 95% CI over five seeds; runtime reports training time only.
<table><tr><td>Task batch B</td><td>AUROC ↑</td><td>AUPRC ↑</td><td>F1↑</td><td>SHD ↓</td><td>Train (s) ↓</td><td>Speedup ↑</td></tr><tr><td>10</td><td> $0 . 8 1 9 \pm 0 . 0 0 7$ </td><td> $0 . 6 5 6 \pm 0 . 0 1 7$ </td><td> $0 . 5 4 1 \pm 0 . 0 3 0$ </td><td> $8 4 1 . 2 \pm 3 0 . 9$ </td><td> ${ \bf 2 2 6 . 9 \pm 2 . 5 }$ </td><td>9.0×</td></tr><tr><td>32</td><td> $\mathbf { 0 . 8 6 7 \pm 0 . 0 0 6 }$ </td><td> $\mathbf { 0 . 7 2 8 \pm 0 . 0 1 9 }$ </td><td> $\mathbf { 0 . 6 7 5 \pm 0 . 0 2 5 }$ </td><td> ${ \bf 7 7 0 . 6 \pm 2 2 . 2 }$ </td><td> $6 6 8 . 4 \pm 7 . 2$ </td><td>3.1×</td></tr><tr><td>64</td><td> $0 . 8 2 2 \pm 0 . 0 0 7$ </td><td> $0 . 6 8 0 \pm 0 . 0 1 2$ </td><td> $0 . 5 8 8 \pm 0 . 0 1 5$ </td><td> $7 8 5 . 2 \pm 1 4 . 9$ </td><td> $1 3 0 1 . 0 \pm 2 4 . 8$ </td><td>1.6×</td></tr><tr><td>100</td><td> $0 . 7 4 7 \pm 0 . 0 0 8$ </td><td> $0 . 5 5 2 \pm 0 . 0 1 5$ </td><td> $0 . 3 9 2 \pm 0 . 0 3 3$ </td><td> $1 0 2 0 . 8 \pm 2 9 . 9$ </td><td> $2 0 4 9 . 9 \pm 1 3 . 3$ </td><td>1.0×</td></tr></table>

## 5.5 Ablation Study

We run two ablations, one for each training component described in Section 4.1.

Target-task subsampling. All variants use the same full method, dataset, optimizer, number of outer updates, and graph-score evaluation; only the number of target tasks B sampled per metaepisode changes. We report training time only, because the post-training readout is identical across variants and does not measure the cost of meta-training. A smaller per-update cost could in principle be offset by needing more updates, so we fix the number of outer updates and compare total training time. Table 4 shows that $B = 3 2$ cuts total training time by 3.1× relative to all-target updates $( B = 1 0 0 )$ and also gives the best recovery.

Support/query meta-objective. Subsampling could also be combined with ordinary multi-task training, so the ablation above does not show what the support/query objective adds. We hold out a subset of target variables from training and train three variants with the same architecture, data, initialization, and budget: full SCDM, an episodic variant without inner adaptation $( K = 0 )$ , and pooled multi-task training on sampled targets (Joint MTL). All three then adapt to each held-out target with the same test-time protocol (Appendix B.3). SCDM reaches a post-adaptation query MSE of 3.782, which is 12.1% lower than Joint MTL (4.305) and 31.1% lower than $\bar { K = } 0 ( 5 . \bar { 4 } 8 7 )$ . Because sampling, data, and test-time adaptation are identical, the gap comes from the training objective. The shared matrix W also matters for recovery: removing it lowers AUPRC from 0.728 to 0.390 (Appendix B.3).

Readout choice. Applied to the same trained models (Table 12), the structural readout ranks edges better on simulated fMRI (AUPRC 0.728 vs. 0.693), so most of the ranking information is already stored in W, while the source-isolated readout is better on simulated EEG (0.476 vs. 0.426).

Stress tests. In NC8-Collinear (Appendix B.3), graph-level AUPRC stays between 0.741 and 0.965 as a non-parent becomes correlated with a true parent (correlation up to 0.89), but the non-edge outranks the true edge in 5 of 30 runs. With four of the 100 variables in simulated fMRI and EEG withheld, a latent-augmented variant keeps observed-block AUPRC at 0.692 and 0.470 (Appendix C).

## 6 Conclusion

We introduced SCDM, a shared-parameter meta-learning framework for scalable causal discovery in nonlinear temporal systems. By treating target-variable prediction as a collection of related tasks, SCDM learns a shared temporal predictor and causal-strength matrix while updating only a subset of target tasks per meta-episode. Ablations show that target-task subsampling lowers training cost without hurting recovery and that the support/query objective gives shared parameters that adapt better than pooled multi-task training. SCDM is competitive on controlled benchmarks and gives a better runtime–accuracy tradeoff at 1,000 and 3,000 variables. SCDM remains an observational method with clear limits. Its recovery result assumes no hidden common causes, a correct lag order, and no instantaneous effects; a hidden driver of several observed variables can create spurious edges, and our latent-variable experiments (Appendix C) are a controlled diagnostic rather than a general solution. Individual edge scores can also move to correlated non-parents when sources are nearly redundant (Appendix B.3). Future work will study adaptive lag selection, adaptive task sampling, and explicit modeling of hidden confounders.

## Acknowledgments and Disclosure of Funding

We thank the anonymous reviewers and the area chair for their constructive comments and suggestions, which improved the clarity of the method description, the design of the ablation studies, and the discussion of assumptions and limitations. Computing resources were provided by the University of Tennessee at Chattanooga. The authors declare no competing interests.

## References

Tingzhu Bi, Yicheng Pan, Xinrui Jiang, Huize Sun, Meng Ma, and Ping Wang. Uncle: Towards scalable dynamic causal discovery in non-linear temporal systems. arXiv preprint arXiv:2511.03168, 2025.

El Mahdi Chayti and Martin Jaggi. A new first-order meta-learning algorithm with convergence guarantees, 2024. URL https://arxiv.org/abs/2409.03682.

Yuxiao Cheng, Lianglong Li, Tingxiong Xiao, Zongren Li, Jinli Suo, Kunlun He, and Qionghai Dai. Cuts+: High-dimensional causal discovery from irregular time-series. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 11525–11533, 2024.

Doris Entner and Patrik O Hoyer. On causal discovery from time series data using fci. Probabilistic graphical models, 16, 2010.

Alireza Fallah, Aryan Mokhtari, and Asuman Ozdaglar. On the convergence theory of gradient-based model-agnostic meta-learning algorithms, 2020. URL https://arxiv.org/abs/1908.10400.

Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In International conference on machine learning, pages 1126–1135. PMLR, 2017.

Clive WJ Granger. Investigating causal relations by econometric models and cross-spectral methods. Econometrica: journal of the Econometric Society, pages 424–438, 1969.

Aapo Hyvärinen, Kun Zhang, Shohei Shimizu, and Patrik O Hoyer. Estimation of a structural vector autoregression model using non-gaussianity. Journal of Machine Learning Research, 11(5), 2010.

Diviyan Kalainathan, Olivier Goudet, Isabelle Guyon, David Lopez-Paz, and Michele Sebag. Structural agnostic model, causal discovery and penalized adversarial learning. In Causal Learning and Reasoning Workshop, 2018.

Xu Liu, Yutong Xia, Yuxuan Liang, Junfeng Hu, Yiwei Wang, Lei Bai, Chao Huang, Zhenguang Liu, Bryan Hooi, and Roger Zimmermann. LargeST: A benchmark dataset for large-scale traffic forecasting. In Advances in Neural Information Processing Systems (Datasets and Benchmarks Track), 2023.

Helmut Lütkepohl. New Introduction to Multiple Time Series Analysis. Springer, 2005. URL https://link.springer.com/book/10.1007/978-3-540-27752-1.

Daniel Malinsky and Peter Spirtes. Causal structure learning from multivariate time series in settings with unmeasured confounding. In Proceedings of 2018 ACM SIGKDD workshop on causal discovery, pages 23–47. PMLR, 2018.

Ricards Marcinkevi ˇ cs and Julia E Vogt. Interpretable models for granger causality using self-ˇ explaining neural networks. arXiv preprint arXiv:2101.07600, 2021.

Meike Nauta, Doina Bucur, and Christin Seifert. Causal discovery with attention-based convolutional neural networks. Machine Learning and Knowledge Extraction, 1(1):19, 2019.

Ignavier Ng, Shengyu Zhu, Zhuangyan Fang, Haoyang Li, Zhitang Chen, and Jun Wang. Masked gradient-based causal structure learning. In Proceedings of the 2022 SIAM International Conference on Data Mining (SDM), pages 424–432. SIAM, 2022.

Alex Nichol, Joshua Achiam, and John Schulman. On first-order meta-learning algorithms. arXiv preprint arXiv:1803.02999, 2018.

Jonas Peters, Joris M. Mooij, Dominik Janzing, and Bernhard Schölkopf. Causal discovery with continuous additive noise models. Journal ofMachine Learning Research, 15:2009–2053, 2014.

Sachin Ravi and Hugo Larochelle. Optimization as a model for few-shot learning. In International conference on learning representations, 2017.

Shaogang Ren. Scalable Algorithms for High Dimensional Structured Data. PhD thesis, TAMU, 2017.

Shaogang Ren and Xiaoning Qian. Dynamic incremental optimization for best subset selection. arXiv preprint arXiv:2402.02322, 2024.

Shaogang Ren, Shuai Huang, Jieping Ye, and Xiaoning Qian. Safe feature screening for generalized lasso. IEEE transactions on pattern analysis and machine intelligence, 40(12):2992–3006, 2017.

Shaogang Ren, Weijie Zhao, and Ping Li. Thunder: A fast coordinate selection solver for sparse learning. Advances in Neural Information Processing Systems, 33:1571–1582, 2020.

Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. " why should i trust you?" explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining, pages 1135–1144, 2016.

Jakob Runge. Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series datasets. In Conference on uncertainty in artificial intelligence, pages 1388–1397. Pmlr, 2020.

Jakob Runge, Peer Nowack, Marlene Kretschmer, Seth Flaxman, and Dino Sejdinovic. Detecting and quantifying causal associations in large nonlinear time series datasets. Science advances, 5 (11):eaau4996, 2019.

Paula Sanz Leon, Stuart A. Knock, Marmaduke M. Woodman, Lia Domide, Jochen Mersmann, Anthony R. McIntosh, and Viktor Jirsa. The virtual brain: a simulator of primate brain network dynamics. Frontiers in Neuroinformatics, 7:10, 2013. doi: 10.3389/fninf.2013.00010.

Alexander Schaefer, Ru Kong, Evan M. Gordon, Timothy O. Laumann, Xi-Nian Zuo, Avram J. Holmes, Simon B. Eickhoff, and B. T. Thomas Yeo. Local-global parcellation of the human cerebral cortex from intrinsic functional connectivity MRI. Cerebral Cortex, 28(9):3095–3114, 2018.

Avanti Shrikumar, Peyton Greenside, and Anshul Kundaje. Learning important features through propagating activation differences. In International conference on machine learning, pages 3145– 3153. PMlR, 2017.

Alex Tank, Ian Covert, Nicholas Foti, Ali Shojaie, and Emily B Fox. Neural granger causality. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(8):4267–4279, 2021.

Robert Tibshirani. Regression shrinkage and selection via the lasso. Journal ofthe Royal Statistical Society Series B: Statistical Methodology, 58(1):267–288, 1996.

Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Daan Wierstra, et al. Matching networks for one shot learning. Advances in neural information processing systems, 29, 2016.

Peng Yang, Shaogang Ren, Yang Zhao, and Ping Li. Calibrating cnns for few-shot meta learning. In 2022 IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV), pages 408–417. IEEE, 2022.

Huaxiu Yao, Ying Wei, Junzhou Huang, and Zhenhui Li. Hierarchically structured meta-learning. In International conference on machine learning, pages 7045–7054. PMLR, 2019.

B. T. Thomas Yeo, Fenna M. Krienen, Jorge Sepulcre, Mert R. Sabuncu, Danial Lashkari, Marisa Hollinshead, Joshua L. Roffman, Jordan W. Smoller, Lilla Zöllei, Jonathan R. Polimeni, Bruce Fischl, Hesheng Liu, and Randy L. Buckner. The organization of the human cerebral cortex estimated by intrinsic functional connectivity. Journal of Neurophysiology, 106(3):1125–1165, 2011.

Matthew D Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In European conference on computer vision, pages 818–833. Springer, 2014.

Ziyi Zhang, Shaogang Ren, Xiaoning Qian, and Nick Duffield. Learning flexible time-windowed granger causality integrating heterogeneous interventional time series data. In Proceedings ofthe 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 4408–4418, 2024.

Ziyi Zhang, Shaogang Ren, Xiaoning Qian, and Nick Duffield. Invargc: Invariant granger causality for heterogeneous interventional time series under latent confounding. arXiv preprint arXiv:2510.19138, 2025.

Xun Zheng, Bryon Aragam, Pradeep K Ravikumar, and Eric P Xing. Dags with no tears: Continuous optimization for structure learning. Advances in neural information processing systems, 31, 2018.

Wanqi Zhou, Shuanghao Bai, Shujian Yu, Qibin Zhao, and Badong Chen. Jacobian regularizer-based neural granger causality. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of PMLR, pages 61763–61782, 2024.

## Notation

Table 5: Notation used throughout the paper.
<table><tr><td>Symbol</td><td>Definition</td></tr><tr><td> $T$ </td><td>Number of time points</td></tr><tr><td> $d$ </td><td>Number of observed variables, which is also the number of target tasks</td></tr><tr><td> $p$ </td><td>Observed-history order: the input at time t is  $x _ { t - 1 } , \ldots , x _ { t - p }$ </td></tr><tr><td> $\ell$ </td><td>Lag index,  $\ell = 1 , \ldots , p$ </td></tr><tr><td> $\mathrm { l o s s } ( \cdot , \cdot )$ </td><td>Pointwise prediction loss (squared error in all experiments)</td></tr><tr><td> $\theta$ </td><td>Shared temporal predictor parameters</td></tr><tr><td> $W \in \mathbb { R } ^ { d \times d }$ </td><td>Shared causal-strength matrix;  $W _ { j , i }$  is the strength from source  $j$  to target i, and column  $W [ : , i ]$ </td></tr><tr><td>r</td><td>gates all sources of target i Number of components per edge in W  $( r = 1$  unless stated)</td></tr><tr><td> $\widetilde { H } _ { t } ^ { ( i ) }$ </td><td>Weighted lagged history of target i, with entries  $W _ { j , i } x _ { t - \ell } ^ { ( j ) }$ </td></tr><tr><td> $B$ </td><td>Number of target tasks sampled per meta-episode</td></tr><tr><td> $K$ </td><td>Number of inner (support) adaptation steps</td></tr><tr><td> $N$ </td><td>Number of outer meta-updates</td></tr><tr><td> $L _ { \mathrm { s u p } } , L _ { \mathrm { q r y } }$ </td><td>Lengths of the support and query windows</td></tr><tr><td> $\alpha , \beta$ </td><td>Inner and outer learning rates</td></tr><tr><td> $\lambda _ { W }$ </td><td>Sparsity coefficient on  $W [ : , i ]$  , shared by the support and query losses</td></tr><tr><td> $\mathrm { P a } ( i )$ </td><td>Set of true lagged parents of target i</td></tr><tr><td> $S _ { j i } ^ { \star }$ </td><td>Population ablation score: increase in the optimal risk of target i when source removed</td></tr><tr><td> $\Delta _ { \mathrm { m i n } }$ </td><td>Smallest  $S _ { j i } ^ { \star }$  over true edges</td></tr><tr><td> $s _ { j  i }$ </td><td>Implemented edge score from source  $; j$  to target i</td></tr><tr><td> $\epsilon _ { \mathrm { s t a t } } , \epsilon _ { \mathrm { m e t a } } , \epsilon _ { \mathrm { r e a d o u t } }$ </td><td>Statistical, meta-optimization, and readout errors in Theorem 4.1</td></tr></table>

## A Theoretical Analysis

This appendix provides the proof of Theorem 4.1. The analysis is intentionally restricted to the fully observed non-latent setting. It does not claim recovery under hidden confounding, instantaneous effects, incorrect lag specification, or misspecified predictors. The purpose of the result is to formalize a finite-error recovery principle: if the learned score is uniformly close to a population risk-gap score, then thresholding recovers the delayed causal graph.

## A.1 Population Target and Assumptions

Let $\begin{array} { r c l } { X _ { t } } & { = } & { ( X _ { t } ^ { ( 1 ) } , \ldots , X _ { t } ^ { ( d ) } ) } \end{array}$ be a stationary multivariate temporal process and let $\begin{array} { r l } { H _ { t } } & { { } = } \end{array}$ $[ X _ { t - 1 } , \ldots , X _ { t - p } ] \in \mathbb { R } ^ { d \times p }$ denote the lagged history. For each target $i ,$ the parent set $\mathrm { P a } ( i ) \subseteq$ $\{ 1 , \ldots , d \} \setminus \{ i \}$ contains source variables whose lagged histories directly influence $X _ { t } ^ { ( i ) }$ . For the pointwise loss loss $( \cdot , \cdot )$ , define

$$
R _ { i } ( f ) = \mathbb { E } [ \mathrm { l o s s } ( f ( H _ { t } ) , X _ { t } ^ { ( i ) } ) ] , \qquad R _ { i } ^ { \star } = \operatorname* { i n f } _ { f \in \mathcal { F } _ { i } } R _ { i } ( f ) .
$$

For a candidate source $j ,$ let $H _ { t , - j }$ be the lagged history with all lags of source j removed, and define the restricted optimal risk

$$
R _ { i , - j } ^ { \star } = \operatorname* { i n f } _ { f \in \mathscr { F } _ { i , - j } } \mathbb { E } [ \log ( f ( H _ { t , - j } ) , X _ { t } ^ { ( i ) } ) ] .
$$

The population ablation score is

$$
S _ { j i } ^ { \star } = R _ { i , - j } ^ { \star } - R _ { i } ^ { \star } .
$$

We group the assumptions into four categories: identifiability, statistical concentration, metaoptimization, and readout fidelity.

Assumption A.1 (Delayed-system identifiability). The process is fully observed and follows a delayed temporal causal graph with no instantaneous effects. The learner’s lag order covers the true maximum causal lag. Conditional on the lagged histories of its parents, each target is independent of non-parent lagged histories. The predictor class is rich enough to represent the population-optimal delayed conditional predictor. Finally, the population ablation score has a positive margin: there exists $\Delta _ { \operatorname* { m i n } } > 0$ such that $S _ { j i } ^ { \star } \geq \Delta _ { \operatorname* { m i n } } f o r$ every $j \in \mathrm { P a } ( i )$ , and $S _ { j i } ^ { \star } = 0 f o r$ every ${ \mathrm { ~ \# ~ P a } ( i ) }$

Assumption $\mathbf { A } . 2$ (Uniform concentration under temporal dependence). The temporal process is stationary and sufficiently mixing. The loss-composedfull and source-restricted predictor classes havefinite sequential complexity $\mathcal { C } ( \mathcal { F } )$ , yielding a uniform concentration bound with effective sample size $\dot { T } _ { \mathrm { e f f } }$

Assumption A.3 (Meta-optimization residual). After N outer iterations, task batch size $B ,$ and K inner adaptation steps, the SCDM meta-learned empirical score $\widehat { S } _ { j i , T , N } ^ { \mathrm { m e t a } }$ satisfies

$$
\operatorname* { m a x } _ { i , j } \left| \widehat { S } _ { j i , T , N } ^ { \mathrm { m e t a } } - \widehat { S } _ { j i , T } ^ { \mathrm { a b l } } \right| \leq \epsilon _ { \mathrm { m e t a } } ( N , B , K ) ,
$$

where $\widehat { S } _ { j i , T } ^ { \mathrm { a b l } }$ is the ideal empirical ablation score. This term capturesfinite episodic meta-training, target-task subsampling, finite support adaptation, and the first-order meta-update. We use existing MAML andfirst-order MAML convergence analyses only to motivate this residualform; we do not claim a new meta-optimization rate.

Assumption A.4 (Readout residual). Let $\widetilde { S } _ { j i , T , N }$ be the implemented post-training source-isolated score. Its discrepancyfrom the meta-learned empirical ablation score is bounded by

$$
\operatorname* { m a x } _ { i , j } \left| \widetilde { S } _ { j i , T , N } - \widehat { S } _ { j i , T , N } ^ { \mathrm { m e t a } } \right| \leq \epsilon _ { \mathrm { r e a d o u t } } ( T , N ) .
$$

This residual records the approximation introduced by using a scalable post-training readout instead ofretraining a restricted predictorfor every ordered pair.

## A.2 Statistical, Meta-optimization, and Readout Errors

We first control the gap between the ideal empirical ablation score and the population ablation score. Let $\begin{array} { r } { \widehat { R } _ { i , T } ( f ) = T ^ { \prime - 1 } \sum _ { t = p + 1 } ^ { T } } \end{array}$ loss $( f ( H _ { t } ) , X _ { t } ^ { ( i ) } )$ ), where $T ^ { \prime } = T - p \mathrm { . }$ . Similarly define $\widehat { R } _ { i , - j , T } ( f ) =$ $\begin{array} { r } { T ^ { \prime - 1 } \sum _ { t = p + 1 } ^ { T } \log ( f ( H _ { t , - j } ) , X _ { t } ^ { ( i ) } ) } \end{array}$ . The ideal empirical risks are $\widehat { R } _ { i , T } ^ { \star } = \operatorname* { i n f } _ { f \in \mathcal { F } _ { i } } \widehat { R } _ { i , T } ( f )$ and $\begin{array} { r } { \widehat { R } _ { i , - j , T } ^ { \star } = \operatorname* { i n f } _ { f \in \mathcal { F } _ { i , - j } } \widehat { R } _ { i , - j , T } ( f ) } \end{array}$ , and the ideal empirical ablation score is $\widehat { S } _ { j i , T } ^ { \mathrm { a b l } } = \widehat { R } _ { i , - j , T } ^ { \star } - \widehat { R } _ { i , T } ^ { \star }$ Lemma A.1 (Uniform convergence of ablation scores). Under Assumption A.2, with probability at least $1 - \delta ,$

$$
\operatorname* { m a x } _ { i , j } \left. \widehat { S } _ { j i , T } ^ { \mathrm { a b l } } - S _ { j i } ^ { \star } \right. \leq \epsilon _ { \mathrm { s t a t } } ( T , d , \delta ) , \qquad \epsilon _ { \mathrm { s t a t } } ( T , d , \delta ) = C \sqrt { \frac { \mathcal { C } ( \mathcal { F } ) + \log ( 2 d ^ { 2 } / \delta ) } { T _ { \mathrm { e f f } } } } .
$$

Here C is a constant depending on the loss scale and mixing constants. In the simplified boundedcomplexity and geometrically mixing case, this reduces to $\epsilon _ { \mathrm { s t a t } } ( T , d , \delta ) = O ( \sqrt { \log ( d ^ { 2 } / \delta ) / T } )$ .

Proof. For any fixed pair $( j , i )$

$$
\left| \widehat { S } _ { j i , T } ^ { \mathrm { a b l } } - S _ { j i } ^ { \star } \right| \leq \left| \widehat { R } _ { i , - j , T } ^ { \star } - R _ { i , - j } ^ { \star } \right| + \left| \widehat { R } _ { i , T } ^ { \star } - R _ { i } ^ { \star } \right| .
$$

The first term is controlled by the uniform deviation over $\mathcal { F } _ { i , - j }$

$$
\left| \widehat { R } _ { i , - j , T } ^ { \star } - R _ { i , - j } ^ { \star } \right| \leq \operatorname* { s u p } _ { f \in \mathcal { F } _ { i , - j } } \left| \widehat { R } _ { i , - j , T } ( f ) - R _ { i , - j } ( f ) \right| .
$$

The same argument gives $\begin{array} { r } { \left| \widehat { R } _ { i , T } ^ { \star } - R _ { i } ^ { \star } \right| \leq \operatorname* { s u p } _ { f \in \mathcal { F } _ { i } } \left| \widehat { R } _ { i , T } ( f ) - R _ { i } ( f ) \right| } \end{array}$ . By the mixing concentration assumption and the finite complexity of the loss-composed classes, each uniform deviation is bounded by a term of order $\sqrt { ( \mathcal { C } ( \mathcal { F } ) + \log ( 1 / \delta ^ { \prime } ) ) / T _ { \mathrm { e f f } } }$ . Applying a union bound over at most $d ( d - 1 ) \leq d ^ { 2 }$ ordered pairs and absorbing constants yields the stated bound. The simplified rate follows under bounded complexity and geometric mixing. □

The statistical bound above concerns the ideal empirical ablation score. The implemented method differs from this ideal object in two ways. First, the temporal predictor and causal-strength matrix are obtained by a finite number of meta-training iterations. Since each target variable is treated as a task, the shared initialization and shared temporal representation are optimized through support/query episodes. Finite N, finite task batch size B, finite inner adaptation K, and the first-order approximation all contribute to the residual

$$
\epsilon _ { \mathrm { m e t a } } ( N , B , K ) = \operatorname* { m a x } _ { i , j } \left| \widehat { S } _ { j i , T , N } ^ { \mathrm { m e t a } } - \widehat { S } _ { j i , T } ^ { \mathrm { a b l } } \right| .
$$

This term captures how closely the learned meta-predictor approximates the empirical full and restricted risk minimizers. Existing MAML convergence results support the view that such optimization residuals can decrease under standard smoothness, bounded-gradient, and step-size conditions [Fallah et al., 2020, Chayti and Jaggi, 2024]. We use this literature only to motivate the residual form; our graph-recovery theorem does not require claiming a new rate for the particular first-order implementation.

Second, the implemented graph extraction uses a post-training perturbation readout rather than explicitly retraining a restricted model for every ordered pair. We therefore define

$$
\epsilon _ { \mathrm { r e a d o u t } } ( T , N ) = \operatorname* { m a x } _ { i , j } \left| \widetilde { S } _ { j i , T , N } - \widehat { S } _ { j i , T , N } ^ { \mathrm { m e t a } } \right| .
$$

This term measures how faithfully the perturbation procedure approximates the ideal source-ablation score. When the trained predictor is well specified and the perturbation operation faithfully removes source-specific historical information, this discrepancy vanishes in the infinite-data and fully optimized limit. In finite samples, it is a practical residual controlled jointly by the quality of statistical estimation, meta-optimization, and score normalization.

Lemma A.2 (Error decomposition). Under Assumptions A.2, A.3, and A.4, with probability at least $1 - \delta$

$$
\operatorname* { m a x } _ { i , j } \left. \widetilde { S } _ { j i , T , N } - S _ { j i } ^ { \star } \right. \le \epsilon _ { \mathrm { s t a t } } ( T , d , \delta ) + \epsilon _ { \mathrm { m e t a } } ( N , B , K ) + \epsilon _ { \mathrm { r e a d o u t } } ( T , N ) .
$$

Proof. For each ordered pair (j, i), add and subtract $\widehat { S } _ { j i , T , N } ^ { \mathrm { m e t a } }$ and $\widehat { S } _ { j i , T } ^ { \mathrm { a b l } }$ :

$$
\left| \widetilde { S } _ { j i , T , N } - S _ { j i } ^ { \star } \right| \leq \left| \widetilde { S } _ { j i , T , N } - \widehat { S } _ { j i , T , N } ^ { \mathrm { m e t a } } \right| + \left| \widehat { S } _ { j i , T , N } ^ { \mathrm { m e t a } } - \widehat { S } _ { j i , T } ^ { \mathrm { a b l } } \right| + \left| \widehat { S } _ { j i , T } ^ { \mathrm { a b l } } - S _ { j i } ^ { \star } \right| .
$$

Taking the maximum over all $( j , i )$ and applying Lemma A.1, Assumption A.3, and Assumption A.4 gives the result. □

## A.3 Proof of Finite-Error Graph Recovery

Theorem A.3 (Finite-error graph recovery). Suppose Assumptions A.1, A.2, A.3, and A.4 hold. Let $\epsilon _ { \mathrm { t o t } } = \epsilon _ { \mathrm { s t a t } } ( T , d , \delta ) + \epsilon _ { \mathrm { m e t a } } ( \bar { N } , B , K ) + \epsilon _ { \mathrm { r e a d o u t } } ( T , N ) . \bar { H ^ { \epsilon } } _ { \mathrm { t o t } } < \Delta _ { \mathrm { m i n } } / 2 ,$ , then thresholding the practical score at any

$$
\gamma \in ( \epsilon _ { \mathrm { t o t } } , \Delta _ { \mathrm { m i n } } - \epsilon _ { \mathrm { t o t } } )
$$

recovers the true graph with probability at least $1 - \delta \colon \operatorname* { P r } ( { \widehat { A } } _ { T , N } = A ^ { \star } ) \geq 1 - \delta .$ . In particular, $\gamma = \Delta _ { \mathrm { m i n } } / 2$ is valid.

Proof. By Lemma A.2, with probability at least 1 − δ, m $\mathfrak { w } _ { i , j } | \widetilde { S } _ { j i , T , N } - S _ { j i } ^ { \star } | \le \epsilon _ { \mathrm { t o t } }$ . Consider a true parent $j \in \mathrm { P a } ( i )$ . By Assumption $\mathrm { A } . 1 , S _ { j i } ^ { \star } \geq \Delta _ { \operatorname* { m i n } }$ , and therefore $\widetilde { S } _ { j i , T , N } \geq S _ { j i } ^ { \star } - \epsilon _ { \mathrm { t o t } } \geq$ $\Delta _ { \mathrm { m i n } } - \epsilon _ { \mathrm { t o t } } > \gamma$ , so the edge $j  i$ is correctly included. Now consider a non-parent j $\mathrm { ~ \# ~ P a } ( i )$ By Assumption ${ \bf A } . 1 , S _ { i i } ^ { \star } = 0 $ , and therefore $\widetilde { S } _ { j i , T , N } \leq S _ { j i } ^ { \star } + \epsilon _ { \mathrm { t o t } } = \epsilon _ { \mathrm { t o t } } < \gamma$ , so the edge $j  i$ is correctly excluded. Since every ordered pair is classified correctly on the same high-probability event, the full graph is recovered with probability at least $1 - \delta$ □

## A.4 Asymptotic Recovery and Interpretation

The finite-error theorem immediately yields an asymptotic consistency statement.

Corollary A.4 (Asymptotic full-graph recovery). Under the assumptions of Theorem A.3, if $\epsilon _ { \mathrm { s t a t } } ( T , \dot { d } , \delta _ { T } )  0 , \dot { \epsilon } _ { \mathrm { m e t a } } ( N , B , \bar { K } )  0 ,$ , and ϵ<sub>readout</sub>(T, N) → 0 as T, N → ∞ with $\delta _ { T }  0 ,$ then $\operatorname* { P r } ( \widehat { A } _ { T , N } = A ^ { \star } )  1$

Proof. For sufficiently large T and N $\begin{array} { r l } & { \mathrm { , ~ } \epsilon _ { \mathrm { s t a t } } ( T , d , \delta _ { T } ) + \epsilon _ { \mathrm { m e t a } } ( N , B , K ) + \epsilon _ { \mathrm { r e a d o u t } } ( T , N ) < \Delta _ { \mathrm { m i n } } / 2 . } \\ & { \mathrm { ~ } \mathrm { ) r e m ~ A . 3 . } } \end{array}$ The result follows directly from The

A concrete sufficient condition can be obtained from the statistical bound above. For example, it is enough that

$$
T _ { \mathrm { e f f } } \ge C \frac { \mathcal { C } ( \mathcal { F } ) + \log ( 2 d ^ { 2 } / \delta ) } { \Delta _ { \operatorname* { m i n } } ^ { 2 } } , \qquad \epsilon _ { \mathrm { m e t a } } ( N , B , K ) + \epsilon _ { \mathrm { r e a d o u t } } ( T , N ) \le \frac { \Delta _ { \operatorname* { m i n } } } { 4 } .
$$

This condition should be interpreted as a sufficient recovery regime rather than a sharp samplecomplexity characterization. It shows that graph recovery is controlled jointly by temporal sample size, function-class complexity, graph dimension through a union bound over ${ \check { d } } ^ { 2 }$ candidate edges, and the quality of meta-optimization and perturbation readout.

The theorem is stated in terms of an ideal ablation risk-gap target because this quantity has a direct population interpretation: it measures how much optimal prediction risk increases when the source history of variable $j$ is removed. The implemented method avoids retraining a separate restricted predictor for every ordered pair $( j , i )$ . Instead, after training a shared temporal predictor, it computes a post-training perturbation score that measures how the target prediction changes when source-specific historical information is removed or neutralized.

This distinction is important in high dimensions. Explicitly fitting all restricted models would require $O ( d ^ { 2 } )$ model refits, which is incompatible with voxel-level or traffic-scale systems. The perturbation readout is therefore used as a scalable approximation to the same ablation principle. The residual ϵ<sub>readout</sub> explicitly records this approximation gap. In the infinite-data, well-specified, and fully optimized limit, if the perturbation operation faithfully removes the same source-specific information as the ideal ablation operator, then $\epsilon _ { \mathrm { r e a d o u t } }  0$ . In finite samples, the residual depends on predictor quality, optimization quality, score normalization, and the fidelity of the perturbation operator.

Finally, the identifiability component of the theorem comes from the delayed causal system and the population risk margin, not from meta-learning itself. Meta-learning enters through the finite-error term $\epsilon _ { \mathrm { m e t a } } ( N , B , \bar { K } )$ . In the proposed framework, each target variable defines a task, and all tasks share the same temporal predictor and causal-strength parameterization. The support/query split encourages the shared parameters to adapt efficiently to target-specific prediction tasks, while the outer update aggregates evidence across a mini-batch of target variables. Compared with fitting independent target-specific models, this shared episodic training can reduce adaptation error and optimization error in finite samples. The theory captures this effect through the condition that $\epsilon _ { \mathrm { m e t a } }$ becomes small enough relative to the population margin $\Delta _ { \mathrm { m i n } }$

Thus, the theorem does not claim that meta-learning alone identifies causal structure. Rather, under standard delayed-system identifiability conditions, meta-learning provides a scalable optimization mechanism for approaching the population risk-gap score in high-dimensional temporal systems.

## B Experimental Details and Supplementary Results

This appendix consolidates the implementation details and supplementary results supporting the empirical claims in the main text. We place baseline configurations first, followed by SCDM training and readout details, and then the supplementary dataset protocols and experimental results.

## B.1 Baseline and Implementation Details

This subsection reports the baseline configurations, SCDM hyperparameters, source-isolated readout protocol, thresholding rules, and compute setup used in the experiments. All methods are evaluated on the same train/validation/test split whenever applicable, and self-edges are excluded from graph-level evaluation unless otherwise stated.

Baseline methods. We report the baseline hyperparameter settings used in the controlled benchmark comparison. For NC8, we follow the established benchmark configuration used in prior work whenever available. For the simulated fMRI and simulated EEG benchmarks, lag-based baselines are configured to cover the corresponding data-generating temporal dependencies whenever supported by the public implementation. Unless otherwise stated, remaining hyperparameters follow the default public implementations.

UnCLe. UnCLe [Bi et al., 2025] is available at https://github.com/etigerstudi ${ \circ / }$ uncle-causal-discovery. On NC8, we follow the standard benchmark configuration with lag 1, kernel size 8, 6 TCN blocks, 20 filters, a 1000 + 2000 epoch training schedule, and learning rate $3 \times 1 0 ^ { - 4 }$ . For simulated fMRI and simulated EEG, we set lag 3 and kernel size 5 to match the data-generating temporal dependency. Both neuroimaging-inspired benchmarks use one hidden layer with hidden size 50, encoder and decoder hidden layers [50], prediction length 1, a 10 + 10 epoch training schedule, and learning rate $1 0 ^ { - 4 }$

PCMCI. For PCMCI [Runge et al., 2019], we use partial correlation as the conditional independence test unless otherwise specified. The maximum lag is set to the benchmark lag for NC8 and to the known simulation lag for simulated fMRI and simulated EEG. Since PCMCI is a constraintbased method and does not require neural-network training, its main settings are the conditional independence test, maximum lag, and significance threshold rather than epochs or learning rates.

cMLP. cMLP [Tank et al., 2021] is available at https://github.com/iancovert/Neural-GC. On NC8, we use lag 16, one hidden layer, 1000 epochs, and learning rate $5 \times 1 0 ^ { - 3 }$ . For simulated fMRI and simulated EEG, we use lag 3, hidden size [20], 3000 maximum iterations, and learning rate $5 \times 1 0 ^ { - 2 }$ . The target-specific cMLP architecture uses Conv $1 \mathrm { d } ( p \to 2 0 , k = 5 )$ followed by Conv1d $\lvert ( 2 0  1 , k = 1 )$ , where $p$ is the input lag order.

TCDF. TCDF [Nauta et al., 2019] is available at https://github.com/M-Nauta/TCDF. On NC8, we use kernel size 16, one hidden layer, 1000 epochs, and learning rate $5 \times 1 0 ^ { - 3 }$ . For simulated fMRI and simulated EEG, we use kernel size 4 and dilation coefficient 4 so that the effective temporal range covers lag-3 dependencies. Both neuroimaging-inspired benchmarks use one effective hidden layer, 1000 epochs, and learning rate $1 0 ^ { - 2 }$ . In the implementation, we use the ADDSTCN architecture and set the maximum explicit delay to 5 for the neuroimaging-inspired benchmarks.

GVAR. GVAR [Marcinkevics and Vogt, 2021] is available at ˇ https://github.com/i6092467/ GVAR. On NC8, we use lag order 16, one hidden layer, 1000 epochs, and learning rate $1 0 ^ { - 4 }$ . For simulated fMRI and simulated EEG, we use lag order 3, one hidden layer with width 50, 1000 epochs, and learning rate $1 0 ^ { - 4 }$ . The model uses one SENNGC coefficient network per lag.

CUTS+. CUTS+ [Cheng et al., 2024] is available at https://github.com/jarrycyx/UNN/ tree/main/CUTS\_Plus. We use learning rate $1 0 ^ { - 3 }$ , 64 training epochs, and maximum number of groups 32. For NC8, the temporal range is aligned with the benchmark lag setting $L = 1 6 .$ . For simulated fMRI and simulated EEG, the temporal range is set to cover lag-3 dependencies. We tune the regularization coefficient over $\lambda \in \{ 0 . 1 , \dot { 0 } . 0 5 , 0 . 0 \dot { 1 } , 0 . 0 0 5 \}$

Evaluation consistency. All methods are evaluated on the same train/validation/test split whenever applicable. Continuous directed scores are evaluated using AUROC and AUPRC. For thresholddependent metrics such as F1 and SHD, the threshold is selected on the validation split and applied once to the test split. Unless otherwise stated, self-edges are excluded from graph-level evaluation because they represent autoregressive self-dependencies rather than cross-variable causal discovery.

SCDM. We report all hyperparameter choices for SCDM. Unless noted otherwise, the settings in this section were used for every dataset.

Model architecture. The prediction network $f _ { \theta }$ is a two-hidden-layer MLP with widths dpr → 128 → 64 → 1 and ReLU activations, where d is the number of variables, $p$ the observed-history order, and r the number of components per edge in W. We use $r = 1$ (scalar edge strength) for all dataset except LargeST-GLA-1500 $( r = 1 6 )$ and LargeST-GLA-3000 $( r = 8 )$ ).

Table 6: Temporal settings of each method. These quantities are not directly comparable: SCDM’s p and the observed lag orders set how many past observations enter the model, UnCLe’s lag acts on latent representations after a convolutional encoder, and convolution kernels set a receptive field. The true maximum lag is 16 for NC8, 5 for simulated EEG, and 3 for simulated fMRI.
<table><tr><td>Method</td><td>Temporal setting</td><td>NC8</td><td>Sim. EEG</td><td>Sim. fMRI</td><td>How chosen</td></tr><tr><td>SCDM</td><td>observed-history order p</td><td>16</td><td>5</td><td>3</td><td>True maximum lag</td></tr><tr><td>UnCLe</td><td>latent AR lag / kernel</td><td>1/8</td><td>3/5</td><td>3/5</td><td>NC8: UnCLe benchmark configuration; others: cover</td></tr><tr><td>cMLP</td><td>observed lag order</td><td>16</td><td>3</td><td>3</td><td>the true lag NC8: UnCLe benchmark</td></tr><tr><td>GVAR</td><td>observed lag order</td><td>16</td><td>3</td><td>3</td><td>configuration; others: lag 3 As for cMLP</td></tr><tr><td>TCDF</td><td>kernel / dilation / max delay</td><td> $1 6 / - / -$ </td><td> $4 / 4 / 5$ </td><td> $4 / 4 / 5$ </td><td>Receptive field covers the true lag</td></tr><tr><td>PCMCI</td><td>maximum lag  $\tau _ { \mathrm { m a x } }$ </td><td>16</td><td>5</td><td>3</td><td>True maximum lag</td></tr><tr><td>CUTS+</td><td>temporal range L</td><td>16</td><td>3</td><td>3</td><td>NC8: 16; others: 3</td></tr></table>

Table 7: Dataset-specific hyperparameters for SCDM. d: number of observed variables; p: observedhistory order; $B \colon$ target tasks per outer update; $L _ { \mathrm { { s u p } } } / L _ { \mathrm { { q r y } } } \mathrm { { : } }$ support/query window length; λ<sub>W</sub>: sparsity coefficient (same in support and query losses); N: number of outer updates; Split: train/validation/test fraction.
<table><tr><td>Dataset</td><td>d</td><td>p</td><td>B</td><td> $L _ { \mathrm { s u p } }$ </td><td> $L _ { \mathrm { q r y } }$ </td><td> $\lambda _ { W }$ </td><td>N</td><td>Split</td></tr><tr><td>NC8</td><td>8</td><td>16</td><td>8</td><td>64</td><td>64</td><td>0.05</td><td>1,000</td><td>70/15/15</td></tr><tr><td>Simulated fMRI</td><td>100</td><td>3</td><td>32</td><td>64</td><td>64</td><td>0.05</td><td>5,000</td><td>70/15/15</td></tr><tr><td>Simulated EEG</td><td>100</td><td>5</td><td>32</td><td>64</td><td>64</td><td>0.05</td><td>3,000</td><td>70/15/15</td></tr><tr><td>Voxel-level fMRI</td><td>1,000</td><td>3</td><td>32</td><td>32</td><td>32</td><td>0.03</td><td>500</td><td>70/15/15</td></tr><tr><td>LargeST-GLA-1500</td><td>1,500</td><td>3</td><td>64</td><td>64</td><td>64</td><td> $1 0 ^ { - 4 }$ </td><td>200</td><td>60/20/20</td></tr><tr><td>LargeST-GLA-3000</td><td>3,000</td><td>3</td><td>64</td><td>64</td><td>64</td><td> $1 0 ^ { - 4 }$ </td><td>1,000</td><td>60/20/20</td></tr></table>

Causal-strength matrix initialization. Entries of W are drawn independently from Uniform(0.0, 0.3) for all datasets except LargeST-GLA-1500 and LargeST-GLA-3000, where the range is Uniform(0.0, 0.05). Self-edges are permanently masked to zero before and during training (allow\_self\_loops=false).

Regularization. For a sampled target i, both the support and the query loss penalize only that target’s column $W [ : , i ]$ , so each task regularizes only the incoming edges it can inform. For $r = 1$ the penalty is the element-wise $\ell _ { 1 }$ norm; for $r > 1$ it is a group lasso, the sum over sources of the $\ell _ { 2 }$ norm of each edge’s r components. The coefficient $\lambda _ { W }$ is set per dataset (Table 7).

Input normalization. Time series are z-score normalized using statistics estimated on the training split, with the same transform applied to validation and test. The simulated fMRI dataset is provided in pre-normalized form and is used without further scaling.

Meta-learning. SCDM uses first-order MAML with $K = 3$ inner gradient steps per task. The inner learning rate is $\alpha _ { \theta } = 0 . 0 1$ for the predictor and $\alpha _ { W } = 0 . 0 0 1$ for W; the smaller rate keeps a single task from moving the shared graph too far during adaptation. The outer optimizer is Adam with learning rate $\beta = 0 . 0 0 1$ and weight decay $1 0 ^ { - 5 }$ , and gradients are clipped to unit $\ell _ { 2 }$ norm before each outer update.

Episode structure. Each outer update samples B target tasks uniformly at random. For each task, a contiguous support window of length $L _ { \mathrm { s u p } }$ and a contiguous query window of length $L _ { \mathrm { q r y } }$ are drawn from the training split. Table 7 lists B, $L _ { \mathrm { s u p } } , L _ { \mathrm { q r y } } , p ,$ λ<sub>W</sub>, and the number of outer updates N for each dataset.

Simulated EEG note. The main simulated EEG benchmark uses the full 100 source-space ROIs for non-latent causal discovery. For the supplementary latent-view experiment, 4 ROIs are withheld during training and 96 ROIs are provided as observed variables; this setting is reported separately in Appendix C. The 64 EEG sensor channels are generated by a synthetic forward projection for qualitative visualization and robustness analysis, and are not the graph dimension used for the main causal discovery evaluation.

Simulated fMRI vs. voxel-level fMRI. The simulated fMRI benchmark in the main results table operates at the ROI level $( d = 1 0 0$ , Schaefer-100 parcellation) and is used to evaluate graph-recovery accuracy. The voxel-level fMRI experiment $( d = 1 0 0 0 )$ is a scalability benchmark; at this scale we use the structural readout $| W _ { j , i } |$ described below.

Threshold selection and evaluation. The binary adjacency matrix is obtained by sweeping 200 evenly-spaced candidate thresholds between the minimum and maximum score on the validation split and selecting the threshold that maximizes the F1 score on the validation set. The chosen threshold is then applied to the test-set scores without further adjustment. For voxel-level fMRI we additionally apply a top-k sparsity constraint $( k = 7 1 4 2$ , matching the ground-truth sparsity) before thresholding. All metrics (AUROC, AUPRC, F1, SHD, precision, recall) are computed on offdiagonal entries only; self-edges $( i = j )$ are excluded throughout. AUROC and AUPRC are computed with sklearn.metrics.roc\_auc\_score and average\_precision\_score, respectively.

Compute. All SCDM runs used one NVIDIA A100 80 GB PCIe GPU on the epyc-gpu SLURM partition. NC8 jobs used 32 GB CPU memory and a 3-hour limit; simulated fMRI multi-seed runs used a 1.5-hour limit; voxel-level fMRI runs used 64 GB CPU memory and a 4-hour limit. The five-method LargeST-GLA-3000 feasibility comparison in Table 3 requested 128 GB of host memory and an 80-minute limit per run. For this comparison, peak GPU memory denotes the maximum usage sampled by nvidia-smi during each run; for failed runs, it is the maximum recorded before failure. Such polling may miss short-lived allocation peaks. GPU memory was not recorded for the controlled benchmarks.

Computational cost. Let $C _ { f } ( d , p )$ be the cost of one forward and backward pass of the predictor on one sample; for the MLP above with hidden widths $h _ { 1 } = 1 2 8$ and $h _ { 2 } = 6 4 , C _ { f } \bar { ( d , p ) } = \bar { O ( } d p r h _ { 1 } + h _ { 1 } h _ { 2 } )$ One meta-episode adapts B tasks for K steps on $L _ { \mathrm { s u p } }$ samples and evaluates $L _ { \mathrm { q r y } }$ query samples, so it costs $\bar { O } ( B ( K L _ { \mathrm { s u p } } + L _ { \mathrm { q r y } } ) C _ { f } ( d , p ) )$ ), and N episodes cost $O ( N B ( K L _ { \mathrm { s u p } } \overset { \cdot } { + } \bar { L } _ { \mathrm { q r y } } ) \overset { \cdot } { C } _ { f } ( \overset { \cdot } { d , p } ) )$ First-order MAML avoids second-order terms through the inner loop. With fixed window lengths, T affects only the number of available windows and data storage, not the cost of an episode. Memory is dominated by $W ~ ( O ( d ^ { 2 } r ) )$ , the first predictor layer $( O ( d p r h _ { 1 } ) )$ , and the temporary adapted copies for the B sampled tasks. The structural readout costs $O ( d ^ { 2 } )$ ; the source-isolated readout costs $O ( d ^ { 2 } M L _ { \mathrm { q r y } } C _ { f } ( \dot { d } , p ) )$ ) plus dM adaptations. These expressions count work per update or per readout; they do not bound the number of updates needed to converge.

Post-training graph readout. The readout is applied only after training and is not part of the optimization objective. We use the main-text convention: $W _ { j , i }$ gates source $j$ for target i, and $s _ { j  i }$ is the score of edge $j \to i$

Source-isolated readout. For each target $i ,$ we first adapt a temporary copy of the trained parameters on a fresh support window with the same $K = 3$ inner steps used in training, because training also measures query losses after adaptation. On the paired query window we then compute two losses. The baseline loss $L _ { \mathrm { z e r o } } ^ { ( i ) }$ sets the whole column $W [ : , i ]$ to zero, which removes the history of every source. The source-isolated loss $L _ { j } ^ { ( i ) }$ keeps only $W _ { j , i }$ and sets $W _ { k , i } = 0$ for all $k \neq j$ , which keeps only the history of source j. Because each $W _ { k , i }$ multiplies all lags of source $k ,$ zeroing $W _ { k , i }$ is the same as replacing the history of source k with zeros. The score is

$$
s _ { j  i } = \frac { L _ { \mathrm { z e r o } } ^ { ( i ) } - L _ { j } ^ { ( i ) } } { L _ { \mathrm { z e r o } } ^ { ( i ) } + \varepsilon } , \qquad \varepsilon = 1 0 ^ { - 8 } ,\tag{10}
$$

so a larger value means that source $j$ alone removes more of the prediction error of target $i .$ Scores are averaged over M independent support/query draws $( M = 1 0 0$ for NC8 and $M = 5 0$ for simulated fMRI and EEG). We use this readout only for $d \leq 1 0 0$ because its cost grows with $d ^ { 2 }$ (see the computational cost paragraph above).

Structural readout. For voxel-level fMRI and LargeST-GLA we use $s _ { j  i } = | W _ { j , i } |$ (the $\ell _ { 2 }$ norm of the r components when $r > 1 )$ , which costs $O ( d ^ { 2 } )$ and needs no model evaluation.

Table 8: Summary of datasets used in the experiments. The symbol d denotes the number of observed variables provided to the learner, and q denotes withheld or latent variables when applicable. For LargeST, chronological train/validation/test splits are used and the complete directed causal ground truth is unavailable.
<table><tr><td>Dataset</td><td>d</td><td>q</td><td>GT graph</td><td>Primary role</td></tr><tr><td>NC8</td><td>8</td><td>0</td><td>Yes</td><td>Controlled causal recovery</td></tr><tr><td>Masked NC8</td><td>6</td><td>2</td><td>Yes</td><td>Controlled latent diagnostic</td></tr><tr><td>Simulated fMRI</td><td>100</td><td>0</td><td>Yes</td><td>Neuroimaging causal recovery</td></tr><tr><td>Latent-view simulated fMRI</td><td>96</td><td>4</td><td>Yes</td><td>Latent-view observed-block recovery</td></tr><tr><td>Simulated EEG</td><td>100</td><td>0</td><td>Yes</td><td>Source-space causal recovery</td></tr><tr><td>Latent-view simulated EEG</td><td>96</td><td>4</td><td>Yes</td><td>Latent-view observed-block recovery</td></tr><tr><td>Voxel-level fMRI</td><td>1000</td><td>4</td><td>Yes</td><td>High-dimensional scalability</td></tr><tr><td>LargeST-GLA-1500</td><td>1500</td><td>0</td><td>No</td><td>Real-world scalability and forecasting</td></tr><tr><td>LargeST-GLA-3000</td><td>3000</td><td>0</td><td>No</td><td>Real-world scalability and forecasting</td></tr></table>

Table 9: Dataset split protocol. Synthetic datasets use train, validation, and test partitions for model fitting, threshold selection, and final reporting. LargeST uses chronological splits to avoid leakage from future intervals.
<table><tr><td>Dataset group</td><td>Split protocol</td></tr><tr><td>NC8 and masked NC8</td><td>Training, validation, and test sequences or replicas are separated before model selection and final reporting.</td></tr><tr><td>Simulated fMRI / EEG and latent- view variants</td><td>200/50/50 train/validation/test samples are used for controlled recovery experiments.</td></tr><tr><td>Voxel-level fMRI</td><td>Train, validation, and test intervals are separated; ground-truth graph evaluation is performed on held-out test scores.</td></tr><tr><td>LargeST-GLA</td><td>Chronological train/validation/test intervals are used to avoid future-to-past leakage.</td></tr></table>

## B.2 Dataset Overview and Evaluation Protocols

This subsection gives a compact overview of the benchmark suite and evaluation protocol. Detailed simulation construction choices, including graph-generation rules, dynamics, observation models, and recoverability diagnostics, are provided separately in Appendix D.

Benchmark overview. NC8 is a low-dimensional nonlinear temporal benchmark with eight observed variables and a known lagged directed graph. It is used to verify causal recovery in a small controlled setting and to support the masked-NC8 latent diagnostic. The simulated fMRI benchmark is a TVB-inspired, atlas-grounded BOLD-like simulation over 100 Schaefer cortical ROIs. It combines Yeo-network organization, delayed nonlinear ROI interactions, and HRF-based observation effects while retaining a known directed ground-truth graph. The simulated EEG benchmark is a source-space EEG-like simulation over the same Schaefer-100 ROI scale. It uses network-modular directed source interactions, EEG-like oscillatory components, and a simplified 64-channel source-to sensor projection for qualitative visualization. The voxel-level fMRI benchmark scales the controlled setting to 1000 observed variables and nearly one million candidate directed pairs, preserving synthetic ground truth for runtime–accuracy evaluation. LargeST-GLA is a real-world traffic benchmark without complete directed causal ground truth; it is therefore used for scalability, forecasting, and qualitative structural analysis rather than causal AUROC/AUPRC.

Evaluation protocol. For datasets with known ground-truth graphs, we report AUROC and AUPRC as threshold-independent causal discovery metrics. When threshold-dependent metrics such as F1 and SHD are reported, the threshold is selected on the validation split and applied once to the test split. Unless otherwise stated, self-edges are excluded from graph-level evaluation because they correspond to autoregressive self-dependencies rather than cross-variable causal discovery. For LargeST, we do not report causal AUROC or AUPRC because a complete directed causal graph is not available.

(b) Network-sorted ground-truth graph

Table 10: Evaluation role of each dataset group. Controlled synthetic datasets support causal recovery metrics because directed ground-truth graphs are available. LargeST is used for real-world scalability, forecasting, and qualitative structural analysis because complete directed causal ground truth is unavailable.
<table><tr><td>Dataset group</td><td>Causal recovery metrics</td><td>Auxiliary use</td></tr><tr><td>NC8, simulated fMRI, simu- lated EEG</td><td>AUROC, AUPRC, F1, SHD against known directed graphs</td><td>Runtime and controlled ablations</td></tr><tr><td>Masked NC8 and latent-view neuroimaging benchmarks</td><td>Block-level recovery diagnostics under controlled masking</td><td>Latent-extension analysis</td></tr><tr><td>Voxel-level fMRI</td><td>AUROC/AUPRC against known high-dimensional synthetic graph</td><td>Runtime-performance tradeoff and scalability analysis</td></tr><tr><td>LargeST-GLA</td><td>No causal AUROC/AUPRC be- cause complete directed ground truth is unavailable</td><td>Real-world scalability, forecasting, and qualitative graph visualization</td></tr></table>

Tables 8–10 summarize the benchmark suite, split protocol, and evaluation role of each dataset. This organization separates controlled causal recovery experiments from high-dimensional scalability and real-world forecasting experiments.

![](images/c2e90770c6707c857be0bfb778a827f33c7aad37761dbe619456f7334f8e6134.jpg)

![](images/ba0ae9cb072ca8965f3efcb838942e310e15eaa78932d6a5f4edd09000413629.jpg)

![](images/bb89cccfddb0cc8703ddad5cd602831da8ad18313c9c8572bcf05e117222b00c.jpg)

![](images/7a1c06b4bc6802cbefc15f42e247c044a13cc136493e9ce86279c9a75759bd0a.jpg)  
Figure 4: Overview of the simulated fMRI benchmark. The benchmark uses Schaefer-100 cortical ROIs grouped by Yeo-7 functional networks. The visualization summarizes the atlas-based ROI layout, the network-sorted directed ground-truth graph, representative BOLD-like time series, and the structured lagged effective-connectivity pattern used for evaluation. This figure illustrates the construction of the controlled TVB-inspired, atlas-grounded fMRI simulation rather than empirical subject-level fMRI connectivity.

Neuroimaging simulation overviews. Figures 4 and 5 provide compact visual summaries of the two atlas-grounded neuroimaging-inspired benchmarks. The detailed fMRI and EEG construction procedures are intentionally kept in Appendix D.2 and Appendix D.3, respectively, to avoid duplicating the dataset specification here.

![](images/329d00b6ed0225066bfe65d65446744cacecc231c57fc85ef88f42e49c4b0a95.jpg)

(b) Network-sorted source graph  
![](images/1981e3672f7786cc14d1c603f293445edc838cca4147b68f54951636b6afc225.jpg)

(c) EEG-like dynamics and spectrum  
![](images/a6ec718d43b3cec5a9650d235b070a4ae3b675538b4942cf8a5c8b3f62768112.jpg)

(d) Synthetic sensor projection  
![](images/da4b42d594614301fdb2887193cc8e9e08840cc10c29f59220d62f113770a9df.jpg)

Figure 5: Overview of the simulated EEG benchmark. The benchmark is constructed as a Schaefer-100 source-space EEG-like simulation with Yeo-network-modular directed source interactions. The visualization summarizes the source ROI organization, the network-sorted source-level groundtruth graph, representative oscillatory source signals, and the synthetic 64-channel source-to-sensor projection used for qualitative visualization. The projection is a simplified source-mixing model and is not intended to represent a subject-specific BEM/FEM EEG forward solution.

## B.3 Additional Experimental Results

This subsection reports the high-dimensional voxel-level fMRI runtime–accuracy tradeoff, supplementary LargeST forecasting results, and additional ablations.

High-dimensional voxel-level fMRI results. The voxel-level fMRI benchmark is a highdimensional synthetic benchmark with known ground-truth causal structure. It contains d = 1000 observed variables and T = 800 time points, yielding 999,000 off-diagonal candidate directed relations after excluding self-edges. The ground-truth graph contains 7,142 positive directed off-diagonal edges, corresponding to a positive edge density of 0.7149%. Since the expected AUPRC of a random ranking equals the positive edge density, raw AUPRC values are numerically small in this setting. We therefore additionally report AUPRC enrichment over random:

$$
\mathrm { A U P R C \ e n r i c h m e n t } = \frac { \mathrm { A U P R C } } { \mathrm { p o s i t i v e \ e d g e \ d e n s i t y } } .
$$

All AUROC and AUPRC values are computed from continuous causal scores with self-edges excluded. For the fixed-budget runtime experiment, each value is averaged over three random seeds at aligned wall-clock budgets of 200, 400, 600, and 800 seconds.

LargeST-GLA-3000 feasibility. Table 3 in Section 5.4 reports the feasibility test on the LargeST-GLA-3000 subset, which contains 3000 traffic sensors and 8,997,000 off-diagonal candidate directed pairs. All five methods use the same split, z-scored preprocessing, one NVIDIA A100 80 GB GPU per run, three seeds, and an 80-minute limit. A run counts as completed only if it writes a valid directed-score checkpoint for all off-diagonal pairs. SCDM completes training and graph extraction in 56.96 s on average with 6.9 GB peak GPU memory. UnCLe writes an intermediate checkpoint after its first training phase and then runs out of memory in its joint phase; for this failed run, the reported peak GPU memory is the peak recorded before the failure. CUTS+, JRNGC, and cMLP start normally but do not write a completed checkpoint before the limit. Because LargeST has no complete causal ground truth, this test measures whether a method can run at this scale, not causal accuracy.

![](images/c905e2b5a53b49d72f3405bf11f1fd1425c51a5094d5a5936c72a37469619ee2.jpg)

![](images/abaea2af7caf7fcba9c2fa0635e20012590ae7ad81d98dc0b375cfe7a41a7a80.jpg)  
Figure 6: Runtime–accuracy tradeoff on the high-dimensional voxel-level fMRI benchmark under fixed wall-clock budgets. Each point reports the mean over three random seeds, without confidence intervals. Left: AUROC computed from continuous causal scores. Right: AUPRC enrichment over random, defined as AUPRC normalized by the random-ranking baseline. UnCLe is slightly ahead at 200 seconds; from 400 seconds on, SCDM has higher AUROC and AUPRC enrichment.

Figure 6 plots the results of Table 2 (Section 5.3). CUTS+ uses its official configuration with the temporal range set to $L = 3$ to match the generating process. Because the voxel-level graph is highly sparse, the enrichment-normalized AUPRC is a more interpretable sparse-edge ranking measure than raw AUPRC. We include UnCLe and CUTS+ in this study because they output continuous directed scores and save intermediate checkpoints, which allows evaluation at matched wall-clock budgets. cMLP and TCDF fit one model per target and PCMCI runs conditional independence tests over variable pairs, lags, and conditioning sets, so they do not fit this fixed-budget protocol, and we did not obtain a completed GVAR run under it; cMLP is instead included in the 3,000-variable feasibility test (Appendix B.3).

Supplementary LargeST forecasting diagnostic. Because LargeST does not provide a complete directed ground-truth causal graph, we additionally evaluate the learned structures through a one-step graph-conditioned forecasting diagnostic. We use the same graph-masked RidgeVAR forecaster for all graph priors and vary only the graph used to select candidate parent sensors. Active sensors are selected using train-only standard deviation $\sigma _ { \mathrm { t r a i n } } \geq 1 0$ , and the same mask is applied to all methods. This experiment is intended to assess whether the learned graph provides useful predictive structure, rather than to benchmark SCDM as a forecasting-specific architecture.

Table 11: Supplementary one-step graph-conditioned forecasting on LargeST-GLA-1500 active sensors. All variants use the same graph-masked RidgeVAR forecaster with $k = 5$ , lag $p = 3 .$ , and fixed ridge regularization α = 1.0. Active sensors are selected using train-only standard deviation $\sigma _ { \mathrm { t r a i n } } \geq 1 0 ( d = 1 4 8 9 )$
<table><tr><td>Graph prior</td><td>MAE↓</td><td>RMSE↓</td><td>∆MAE vs. Self-only ↑</td></tr><tr><td>Self-only AR</td><td>20.60</td><td>32.35</td><td></td></tr><tr><td>Random</td><td>19.77 ± 0.06</td><td>31.11 ± 0.16</td><td>+0.83</td></tr><tr><td>Distance-kNN</td><td>20.37</td><td>31.98</td><td>+0.23</td></tr><tr><td>Road-network</td><td>20.68</td><td>32.91</td><td>-0.08</td></tr><tr><td>UnCLe</td><td>19.90</td><td>31.38</td><td>+0.70</td></tr><tr><td>Permuted SCDM</td><td> $1 9 . 9 2 \pm 0 . 3 3$ </td><td> $3 1 . 4 4 \pm 0 . 7 7$ </td><td>+0.68</td></tr><tr><td>Distance-matched ctrl</td><td> $1 9 . 7 2 \pm 0 . 0 5$ </td><td> $3 1 . 0 1 \pm 0 . 1 3$ </td><td>+0.88</td></tr><tr><td>SCDM</td><td> ${ \bf 1 9 . 6 8 \pm 0 . 1 0 }$ </td><td> ${ \bf 3 0 . 9 3 \pm 0 . 1 9 }$ </td><td>+0.92</td></tr></table>

Table 12: Readout ablation on the simulated fMRI and simulated EEG benchmarks. Both readouts are applied to SCDM outputs. The learned $| W |$ score evaluates whether the trained causal-strength matrix itself contains useful structural information, while the source-isolated perturbation readout evaluates an alternative post-training graph-extraction procedure. Results are reported as mean ± 95% confidence interval.
<table><tr><td>Dataset</td><td>Readout</td><td>AUROC ↑</td><td>AUPRC ↑</td></tr><tr><td>Simulated fMRI</td><td>Learned |W| score</td><td> $0 . 8 6 7 \pm 0 . 0 0 6$ </td><td> $0 . 7 2 8 \pm 0 . 0 1 9$ </td></tr><tr><td>Simulated fMRI</td><td>Source-isolated perturbation readout</td><td> $0 . 8 3 2 \pm 0 . 0 0 3$ </td><td> $0 . 6 9 3 \pm 0 . 0 1 0$ </td></tr><tr><td>Simulated EEG</td><td>Learned |W| score</td><td> $0 . 7 1 3 \pm 0 . 0 5 3$ </td><td> $0 . 4 2 6 \pm 0 . 0 8 6$ </td></tr><tr><td>Simulated EEG</td><td>Source-isolated perturbation readout</td><td> $0 . 7 1 7 \pm 0 . 0 3 7$ </td><td> $0 . 4 7 6 \pm 0 . 0 6 7$ </td></tr></table>

Table 11 shows that the SCDM-derived graph gives the lowest one-step MAE and RMSE among the tested graph priors. The improvement over Self-only AR is modest but consistent, reducing MAE by 0.92. SCDM also improves over the UnCLe-derived graph and the Permuted-SCDM control under the same forecasting backbone. The improvement over Permuted SCDM suggests that the target-specific parent assignment contains predictive signal.

At the same time, the margins over Random and Distance-matched controls are small. We therefore interpret this result as supplementary evidence that the learned graph is forecast-compatible, rather than as a standalone claim of forecasting superiority or causal correctness. Recursive multi-horizon rollouts produced mixed results and are not used as the primary forecasting diagnostic.

Supplementary ablation results. We provide additional ablations to clarify which components of SCDM contribute to the reported causal recovery performance. The main text reports the taskbatch-size ablation, which isolates the role of target-task subsampling in the shared meta-learning framework. Here, we further analyze the graph readout step by comparing two scoring rules applied to SCDM outputs: the learned causal-strength score $| W _ { i j } |$ and the post-training source-isolated perturbation readout.

Readout ablation. This ablation is designed to test whether the reported recovery performance is driven only by the post-training perturbation readout or whether the learned causal-strength parameters already contain meaningful structural information. For each dataset, we compare the same SCDM training pipeline under two edge-scoring rules. The first uses the learned absolute causalstrength score $| W _ { i j } |$ directly. The second uses the source-isolated perturbation readout described in Appendix B.1. We do not change the training data, model architecture, optimizer, or evaluation protocol.

Table 12 compares the structural readout and the source-isolated readout on the same trained models. On simulated fMRI, the structural score |W| has higher AUROC and AUPRC, so useful ranking information is already present in the learned matrix. On simulated EEG, the source-isolated readout has similar AUROC and higher AUPRC. The better readout therefore depends on the dataset, and neither readout implements a causal intervention.

Collinearity stress test. The source-isolated score measures predictive contribution, so it can fail when two sources carry nearly the same information. We build NC8-Collinear, which adds a common driver $X _ { 0 }$ with edges $X _ { 0 }  X _ { 1 }$ and $X _ { 0 }  X _ { 2 } .$ , a true edge $X _ { 1 }  X _ { 3 } ,$ and no edge $X _ { 2 }  X _ { 3 }$ $X _ { 2 }$ is therefore a correlated non-parent that can stand in for $X _ { 1 }$ . We vary the driver strength a over six levels with five seeds each and record the mean absolute correlation between $X _ { 1 }$ and $X _ { 2 } ,$ the condition number of the lagged input, the graph AUPRC, and the number of runs in which the non-edge $X _ { 2 }  X _ { 3 }$ scores above the true edge ${ \bar { X _ { 1 } } }  X _ { 3 }$ (a proxy reversal). Graph-level ranking stays useful (AUPRC between 0.741 and 0.965), but individual attribution weakens as collinearity grows: the first reversal appears at a mean absolute correlation of 0.572, and reversals occur in 5 of 30 runs overall and in 2 of 5 runs at the highest level. At $a = 0$ the edges from $X _ { 0 }$ carry no signal but remain in the ground truth, which explains the lower AUPRC at that level. Conclusions about individual edges should therefore be treated with care when observed sources are nearly redundant.

Task-batch-size ablation. The task-batch-size ablation in the main text isolates the role of targettask subsampling. All variants use the same full SCDM model, dataset, optimizer, training budget, and post-training score readout; only the number of target-variable tasks sampled per meta-episode is changed. Using all target variables at every update removes target-task subsampling, but substantially increases the per-update computational cost and yields weaker recovery under the same training budget. Moderate task batches provide a better balance between target coverage and optimization effi ciency, supporting target-task subsampling as a practical scalability mechanism for high-dimensional temporal causal discovery.

Table 13: NC8-Collinear stress test (five seeds per level). Reversals count runs in which the non-edge $X _ { 2 }  X _ { 3 }$ scores above the true edge $X _ { 1 }  X _ { 3 }$
<table><tr><td>Driver a</td><td>Mean  $| \mathrm { c o r r } ( X _ { 1 } , X _ { 2 } ) |$ </td><td>Condition number</td><td>AUPRC ↑</td><td>Reversals ↓</td></tr><tr><td>0.0</td><td>0.037</td><td>8.45</td><td>0.741</td><td>0/5</td></tr><tr><td>0.2</td><td>0.252</td><td>9.91</td><td>0.960</td><td>0/5</td></tr><tr><td>0.4</td><td>0.572</td><td>12.76</td><td>0.955</td><td>1/5</td></tr><tr><td>0.6</td><td>0.750</td><td>17.09</td><td>0.947</td><td>1/5</td></tr><tr><td>0.8</td><td>0.841</td><td>21.81</td><td>0.965</td><td>1/5</td></tr><tr><td>1.0</td><td>0.892</td><td>26.69</td><td>0.915</td><td>2/5</td></tr></table>

Table 14: Component ablation on the Schaefer100 TVB-inspired simulated fMRI benchmark. All variants use the same training budget and evaluation protocol. AUROC and AUPRC are reported as mean ± 95% confidence interval; F1 and SHD are reported as means.
<table><tr><td>Variant</td><td>Task subsample</td><td>Shared W</td><td>AUROC↑</td><td>AUPRC ↑</td><td>F1↑</td><td>SHD ↓</td></tr><tr><td>Full SCDM</td><td>√</td><td>√</td><td> $\mathbf { 0 . 8 6 7 \pm 0 . 0 0 6 }$ </td><td> $\mathbf { 0 . 7 2 8 \pm 0 . 0 1 9 }$ </td><td>0.675</td><td>770.6</td></tr><tr><td>w/o task subsampling</td><td>-√</td><td>√</td><td> $0 . 7 4 7 \pm 0 . 0 0 8$ </td><td> $0 . 5 5 2 \pm 0 . 0 1 5$ </td><td>0.392</td><td>1020.8</td></tr><tr><td>w/o shared W (MLP-only)</td><td></td><td></td><td> $0 . 7 3 2 \pm 0 . 0 1 2$ </td><td> $0 . 3 9 0 \pm 0 . 0 2 2$ </td><td>0.401</td><td>1486</td></tr></table>

Component ablation on simulated fMRI. We further isolate two central design choices of SCDM on the simulated fMRI benchmark: target-task subsampling and the shared structural matrix. All variants use the same dataset, training budget, optimizer, thresholding rule, and evaluation protocol. The full model uses target-task subsampling with $B = 3 2$ and a trainable shared causal-strength matrix W. The no-subsampling variant uses all 100 target variables at each update. The MLP-only variant removes the trainable shared structural matrix and uses the same evaluation protocol for a fair comparison.

Table 14 shows that removing target-task subsampling substantially degrades graph recovery, reducing AUROC from 0.867 to 0.747 and AUPRC from 0.728 to 0.552. Removing the shared structural matrix also substantially weakens recovery, especially in AUPRC, which drops from 0.728 to 0.390. Under the same training and evaluation protocol, a shared MLP without the structural matrix therefore recovers the graph much less accurately.

Held-out-target protocol. For the support/query diagnostic, the 100 targets of the simulated fMRI benchmark were partitioned into 70 meta-training targets, 10 validation targets, and 20 final-test targets. The reported diagnostic evaluates the 10 validation targets using seed 0; the final-test targets remain untouched. Joint MTL updates the shared parameters directly on the pooled support and query losses of the sampled targets, and the $K = 0$ variant keeps the episodic query update but skips inner adaptation. At test time each variant adapts to each held-out target on $S = 2 5 6$ support samples with $K _ { \mathrm { t e s t } } = 5 0$ updates of θ and $W [ : , i ]$ , and we report the mean squared error on a disjoint query window that is never used for adaptation. This measures how well the learned parameters adapt to unseen targets, not graph recovery.

Table 15: Held-out-target adaptation. All variants share architecture, data, initialization, training budget, and test-time adaptation; only the training objective differs.
<table><tr><td>Training variant</td><td>Inner steps in training</td><td>Post-adaptation query MSE ↓</td></tr><tr><td>Joint MTL (sampled targets)</td><td></td><td>4.305</td></tr><tr><td>Episodic, no inner adaptation</td><td> $K = 0$ </td><td>5.487</td></tr><tr><td>Full SCDM</td><td> $K = 3$ </td><td>3.782</td></tr></table>

## C Latent Recovery and Results

This section reports supplementary experiments for the latent-augmented extension of SCDM. The main paper focuses on fully observed temporal causal discovery, where all variables involved in the target graph are available to the learner. Here, we consider a controlled partially observed setting in which a subset of variables is withheld from the learner input and treated as latent. The purpose of these experiments is not to solve unrestricted latent causal discovery, but to evaluate whether the proposed shared-parameter meta-learning framework can be extended to represent hidden-to-observed structural influence under controlled masking.

## C.1 Latent-Augmented Setting and Evaluation Protocol

We construct the latent benchmark from a fully observed temporal system by deliberately withholding a subset of variables from the learner input. Let

$$
s _ { t } = \left[ \boldsymbol { x } _ { t } \right] \in \mathbb { R } ^ { d _ { \mathrm { o b s } } + q } ,
$$

where $x _ { t } \in \mathbb { R } ^ { d _ { \mathrm { o b s } } }$ denotes the observed variables provided to the learner and $z _ { t } \in \mathbb { R } ^ { q }$ denotes the withheld variables. During training and inference, the model receives only the observed trajectory $x _ { 1 : T }$ . The withheld trajectories and the full ground-truth graph are used only for evaluation.

The full structural matrix is partitioned as

$$
A = \left[ \begin{array} { l l } { A _ { X X } } & { A _ { X Z } } \\ { A _ { Z X } } & { A _ { Z Z } } \end{array} \right] ,
$$

where $A _ { X X }$ denotes observed-to-observed edges, $A _ { Z X }$ denotes latent-to-observed edges, $A _ { X Z }$ denotes observed-to-latent edges, and $A _ { Z Z }$ denotes latent-to-latent edges. In our controlled masking protocol, the withheld variables are selected so that they have no observed parents, yielding

$$
A _ { X Z } = 0 .
$$

This design avoids a setting in which the withheld variables are merely downstream effects of observed variables, and instead focuses on whether the model can represent hidden drivers of observed dynamics.

For the latent-augmented model, we extend the structural score matrix to the augmented state space:

$$
W = \left[ W _ { X X } \quad \begin{array} { c } { { 0 } } \\ { { W _ { Z X } \quad W _ { Z Z } } } \end{array} \right] .
$$

Here, $W _ { X X }$ captures observed-to-observed temporal influence, $W _ { Z X }$ captures latent-to-observed influence, and $W _ { Z Z }$ captures latent temporal dependency. The zero block follows the controlled benchmark construction, where withheld variables are selected to have no observed parents.

The latent variables are learnable internal states $z _ { t } \in \mathbb { R } ^ { q }$ , not supervised targets: the withheld trajectories are never used for training or scoring. For an observed target i, the latent-aware predictor is

$$
\widehat { x } _ { t } ^ { ( i ) } = f _ { \theta } \Big ( \mathrm { v e c } \big ( \widetilde { H } _ { t } ^ { X , ( i ) } \big ) , W _ { : , i } ^ { Z X } \odot z _ { t - 1 } \Big ) , \qquad \widetilde { H } _ { t , j , \ell } ^ { X , ( i ) } = W _ { j , i } ^ { X X } x _ { t - \ell } ^ { ( j ) } ,\tag{11}
$$

so latent states enter each target only through its column of $W ^ { Z X }$ , gated in the same way as observed sources. For a support or query window $b ,$ the task loss is

$$
\begin{array} { r } { \mathcal { L } _ { i } ^ { b } = \mathcal { L } _ { \mathrm { p r e d } , i } ^ { b } + \lambda _ { W } \Vert W _ { : , i } ^ { X X } \Vert _ { 1 } + \lambda _ { z } R _ { z } ^ { b } , } \end{array}\tag{12}
$$

where $\mathcal { L } _ { \mathrm { p r e d } , i } ^ { b }$ is the one-step squared error of observed target i and $R _ { z } ^ { b }$ penalizes the magnitude of the latent states in the window. Without $R _ { z } ^ { b }$ , the model could make the latent states arbitrarily large and the weights in $W ^ { Z X }$ arbitrarily small without changing any prediction.

After training, we extract scores over the augmented state space and evaluate the corresponding graph blocks:

$$
S = \left[ { \cal S } _ { X X } \ : \ : \ : \ : \ : \ : 0 \ : \right] .
$$

Table 16: Controlled latent recovery on masked NC8. Upper block: full augmented graph and latent-to-observed (ZX) recovery with structural scores. Lower block: observed-to-observed (XX) recovery with the structural score and with the source-isolated readout. Mean $\pm 9 5 \%$ CI over five graph replicas and three seeds.
<table><tr><td colspan="5">Aggregate and latent-to-observed recovery using structural scores</td></tr><tr><td>Evaluation target</td><td>AUROC ↑</td><td>AUPRC ↑</td><td>F1↑</td><td>SHD↓</td></tr><tr><td>Full augmented graph</td><td> $0 . 9 2 6 \pm 0 . 0 0 8$ </td><td> $0 . 6 3 4 \pm 0 . 0 2 6$ </td><td> $0 . 5 9 8 \pm 0 . 0 5 7$ </td><td> $4 . 8 7 \pm 0 . 4 9$ </td></tr><tr><td>Latent → observed (ZX)</td><td> $0 . 6 0 6 \pm 0 . 0 4 0$ </td><td> $0 . 6 8 0 \pm 0 . 0 2 8$ </td><td> $0 . 6 4 8 \pm 0 . 0 5 9$ </td><td> $3 . 8 0 \pm 0 . 4 4$ </td></tr><tr><td colspan="5">Observed-block score diagnostic</td></tr><tr><td>Evaluation target</td><td>Score readout</td><td>AUROC ↑</td><td></td><td></td></tr><tr><td>Observed → observed (XX)</td><td>Structural score |W|</td><td> $0 . 9 4 3 \pm 0 . 0 0 7$ </td><td></td><td></td></tr><tr><td>Observed → observed (XX)</td><td>Source-isolated readout</td><td> $0 . 7 6 1 \pm 0 . 0 7 1$ </td><td></td><td></td></tr></table>

The XX block is used to evaluate observed-to-observed recovery, while the ZX block is used to evaluate hidden-to-observed structural influence. For AUROC and AUPRC, continuous scores are compared directly with the corresponding ground-truth graph blocks. For threshold-dependent metrics such as F1 and SHD, the threshold is selected on the validation split and applied once to the test split. Because learned latent coordinates are identifiable only up to permutation, metrics involving latent variables are computed after matching learned latent coordinates to the withheld variables.

## C.2 Masked NC8 Latent Recovery

We first use masked NC8 as a controlled diagnostic for the latent-augmented model. Two variables are withheld from the learner input and treated as latent variables, while the remaining six variables are used as observed inputs. Since the full eight-variable ground-truth graph is available, this setting allows direct evaluation of both the augmented full graph and the hidden-to-observed block.

This experiment is not intended as a baseline comparison on the small observed block. Instead, it tests whether the latent-augmented model can assign meaningful structural scores to withheld variables under a controlled masking protocol. For observed-to-observed recovery, we report both the learned structural score and the perturbation-based readout. For hidden-to-observed recovery, we use the learned latent-to-observed structural score because the withheld variables are not observed input channels.

Table 16 shows that the latent-augmented model achieves strong aggregate recovery on the controlled masked NC8 benchmark. The structural score recovers the augmented full graph with high AUROC. For the hidden-to-observed block, the model obtains non-trivial recovery after latent-coordinate matching. Although ZX AUROC remains modest, the high ZX AUPRC and thresholded metrics indicate that the model can rank sparse hidden-to-observed influences above many non-edges under the controlled masking protocol. We therefore treat this experiment as a proof-of-concept diagnostic for latent recovery, while the main scalability claims remain focused on the fully observed setting.

## C.3 Observed-block Recovery under Controlled Masking

We next evaluate the latent-augmented model on the controlled latent-view simulated EEG and simulated fMRI benchmarks. Unlike the masked NC8 diagnostic in Appendix C.2, which directly examines aggregate and latent-to-observed recovery in a small system, this experiment focuses on whether the latent-augmented model preserves strong observed-to-observed recovery in higherdimensional neuroimaging-inspired systems.

For both datasets, four variables are withheld from the learner input and treated as latent variables, leaving 96 observed variables for training and observed-block graph recovery. The full 100-node ground-truth graph is retained only for evaluation. For baselines and our latent-view model, we report recovery on the observed-to-observed XX block. We do not use these experiments as the main evidence for unrestricted hidden causal recovery; instead, they test whether controlled latent augmentation remains compatible with accurate observed-block causal discovery.

Table 17: Observed-block causal recovery under controlled latent-view masking on simulated EEG and fMRI. Results are reported as mean ± 95% confidence interval. Best results are shown in bold and second-best results are underlined.
<table><tr><td rowspan="2">Method</td><td colspan="2">Simulated EEG</td><td colspan="2">Simulated fMRI</td></tr><tr><td></td><td>XX AUROC ↑ XX AUPRC ↑</td><td></td><td>XX AUROC ↑ XX AUPRC ↑</td></tr><tr><td>UnCLe</td><td> $0 . 5 6 6 \pm 0 . 0 0 5$ </td><td> $0 . 1 4 6 \pm 0 . 0 0 2$ </td><td> $0 . 6 2 6 \pm 0 . 0 3 3$ </td><td> $0 . 2 2 8 \pm 0 . 0 3 3$ </td></tr><tr><td>PCMCI</td><td> $0 . 7 4 5 \pm 0 . 0 1 1$ </td><td> $0 . 3 8 8 \pm 0 . 0 1 9$ </td><td> $0 . 5 6 6 \pm 0 . 0 1 5$ </td><td> $0 . 1 5 6 \pm 0 . 0 1 2$ </td></tr><tr><td>TCDF</td><td> $\overline { { 0 . 6 8 1 \pm 0 . 0 0 6 } }$ </td><td> $\overline { { 0 . 2 1 3 \pm 0 . 0 1 2 } }$ </td><td> $0 . 5 9 0 \pm 0 . 0 0 9$ </td><td> $0 . 1 7 9 \pm 0 . 0 1 1$ </td></tr><tr><td>GVAR</td><td> $\mathbf { 0 . 7 4 7 \pm 0 . 0 0 3 }$ </td><td> $0 . 3 3 9 \pm 0 . 0 1 6$ </td><td> $0 . 6 1 9 \pm 0 . 0 1 4$ </td><td> $0 . 2 1 4 \pm 0 . 0 1 4$ </td></tr><tr><td>CUTS+</td><td> $0 . 5 8 9 \pm 0 . 0 0 1$ </td><td> $0 . 1 5 4 \pm 0 . 0 0 0$ </td><td> $0 . 5 2 7 \pm 0 . 0 1 2$ </td><td> $0 . 1 3 7 \pm 0 . 0 0 6$ </td></tr><tr><td>cMLP</td><td> $0 . 6 5 9 \pm 0 . 0 0 7$ </td><td> $0 . 2 5 6 \pm 0 . 0 2 0$ </td><td> $0 . 6 2 3 \pm 0 . 0 1 0$ </td><td> $0 . 2 3 5 \pm 0 . 0 1 6$ </td></tr><tr><td>Ours-Latent</td><td> $0 . 7 0 9 \pm 0 . 0 4 2$ </td><td> $\mathbf { 0 . 4 7 0 \pm 0 . 0 7 7 }$ </td><td> ${ \bf 0 . 8 4 9 \pm 0 . 0 1 1 }$ </td><td> $\mathbf { 0 . 6 9 2 \pm 0 . 0 1 3 }$ </td></tr></table>

Table 17 shows that the latent-augmented model remains competitive on observed-block recovery under controlled masking. On simulated EEG, the method achieves the best XX AUPRC, suggesting improved sparse-edge ranking even when its AUROC is not the highest. On simulated fMRI, the latent-augmented model obtains the best XX AUROC and AUPRC, indicating that the latent extension preserves strong observed-block recovery in this structured neuroimaging-inspired setting.

## C.4 Interpretation and Limitations

The latent experiments should be interpreted as a controlled extension study rather than the central contribution of the paper. The main SCDM framework targets fully observed scalable temporal causal discovery, and the main empirical claims are based on non-latent causal recovery, high-dimensional scalability, and real-world LargeST evaluation.

The latent-augmented setting introduces explicit hidden nodes, but the learned latent coordinates are not identifiable in an absolute sense. They may be permuted, rescaled, or transformed without changing the observed prediction objective. Therefore, we do not claim recovery of the exact latent trajectories. Instead, we evaluate whether the model can assign useful structural scores to latent-toobserved influence under a controlled masking protocol where the withheld variables and full graph are known for evaluation.

These results provide supplementary evidence that the shared-parameter meta-learning framework can be extended beyond the fully observed setting. The masked NC8 experiment provides an interpretable proof of concept for hidden-to-observed recovery, while the masked simulated EEG and fMRI experiments evaluate observed-block recovery in higher-dimensional temporal systems. However, unrestricted latent causal discovery under arbitrary hidden confounding, instantaneous effects, or misspecified latent dimension remains outside the scope of this work.

## D Details of Simulation Datasets

This appendix describes the construction of the simulation datasets used in the experiments. The purpose of these datasets is to provide controlled temporal systems with known directed ground-truth graphs, while retaining structured domain-inspired properties beyond purely random synthetic benchmarks. We include the low-dimensional NC8 benchmark, two neuroimaging-inspired source-level simulations, and a high-dimensional voxel-level fMRI benchmark. Dataset usage, train/validation/test splits, evaluation protocols, and compact overview figures are summarized separately in Appendix B.2. The present appendix focuses on construction details and omits repeated overview material.

## D.1 NC8 Benchmark

NC8 is used as a low-dimensional controlled benchmark for temporal causal discovery. It contains eight variables generated from a nonlinear temporal system with a known directed lagged causa graph. The benchmark is useful for verifying whether a method can recover directed temporal dependencies in a setting where the graph is small enough for direct inspection and thresholded graph comparison.

Table 18: Statistics of the NC8 benchmark and its controlled latent-view variant.
<table><tr><td>Statistic</td><td>NC8</td><td>Masked NC8</td></tr><tr><td>Total variables in full system</td><td>8</td><td>8</td></tr><tr><td>Observed variables provided to learner</td><td>8</td><td>6</td></tr><tr><td>Withheld / latent variables</td><td>0</td><td>2</td></tr><tr><td>Time points</td><td>2000</td><td>2000</td></tr><tr><td>Ground-truth graph available</td><td>Yes</td><td>Yes</td></tr><tr><td>Maximum lag</td><td>16</td><td>16</td></tr><tr><td>Primary metrics</td><td>AUROC, AUPRC, F1, SHD</td><td>XX / Z→X AUROC, AUPRC</td></tr><tr><td>Primary use</td><td>Low-dimensional recovery</td><td>Latent-extension evaluation</td></tr></table>

Table 19: Atlas and anatomical metadata used in the simulated fMRI benchmark.
<table><tr><td>Component</td><td>Setting</td></tr><tr><td>Parcellation</td><td>Schaefer-100 cortical atlas</td></tr><tr><td>Number of ROIs</td><td>100</td></tr><tr><td>Coordinate space</td><td>MNI152</td></tr><tr><td>Functional labels</td><td>Yeo-7 canonical networks</td></tr><tr><td>Hemisphere labels ROI metadata</td><td>Left/right hemisphere labels retained</td></tr><tr><td>Use in simulation</td><td>ROI name, hemisphere, Yeo network, and MNI centroid Distance-aware and network-aware graph construction</td></tr><tr><td>Use in visualization</td><td>Brain-template and ROI-coordinate visualization</td></tr></table>

For the fully observed setting, all eight variables are provided to the learner and the learned directed score matrix is evaluated against the known eight-node ground-truth graph. We report AUROC and AUPRC as the primary threshold-independent metrics. When binary graph metrics are reported, the threshold is selected on the validation split and applied once to the test split.

For the controlled latent-view setting, two variables are withheld from the learner input and treated as latent variables. The learner receives only the remaining six observed variables, while the full eight-variable graph is retained for evaluation. This setting is used only for the latent-augmented extension in Appendix C, where we evaluate observed-to-observed and latent-to-observed structural recovery.

## D.2 Simulated fMRI Benchmark

The simulated fMRI benchmark is a TVB-inspired, atlas-grounded, BOLD-like temporal simulation with known directed effective connectivity. It is designed to evaluate causal discovery under neuroimaging-inspired spatial organization, delayed neural interactions, and hemodynamic measurement distortion, while preserving full access to the ground-truth graph.

Each variable corresponds to a cortical region of interest (ROI) from the Schaefer-100 parcellation. ROIs are associated with Yeo-7 functional-network labels, hemisphere labels, and MNI-space centroids. These anatomical and functional metadata are used to construct a structured directed graph and to visualize recovered interactions in brain space.

The ground-truth graph is a sparse directed lagged effective-connectivity graph over the 100 ROIs. Edge probabilities are informed by inter-ROI distance, hemisphere membership, and functionalnetwork organization. Nearby ROIs are assigned higher prior probability, but long-range, crossnetwork, and cross-hemisphere interactions are retained to avoid a purely local graph. This construction should be interpreted as an atlas-informed structural prior rather than empirical diffusion-MRI, DTI, or ENIGMA structural connectivity.

Latent neural source activity is generated using stable nonlinear lagged dynamics. Coupling weights are rescaled to satisfy a spectral-radius stability target, preventing unstable trajectories while preserving directed delayed interactions. The resulting latent source signals provide the pre-hemodynamic neural activity from which BOLD-like observations are generated.

To model fMRI measurement distortion, latent neural source activity is transformed into BOLD-like observations through ROI-wise hemodynamic response function (HRF) convolution. The HRF parameters vary across ROIs, producing temporal smoothing and low-frequency dynamics typical of fMRI measurements. We use a double-gamma HRF and add observation noise before per-ROI normalization.

Table 20: Ground-truth graph construction in the simulated fMRI benchmark.
<table><tr><td>Component</td><td>Implementation</td></tr><tr><td>Node definition</td><td>Cortical ROIs from the Schaefer-100 atlas</td></tr><tr><td>Topology type</td><td>Sparse directed lagged effective-connectivity graph</td></tr><tr><td>Structural prior</td><td>Distance-, hemisphere-, and network-informed edge sampling</td></tr><tr><td>Spatial constraint</td><td>Nearby ROIs have higher prior probability while long-range edges are retained</td></tr><tr><td>Network constraint</td><td>Within-network and selected cross-network interactions are allowed</td></tr><tr><td>Hemisphere constraint</td><td>Both intra-hemispheric and inter-hemispheric edges are retained</td></tr><tr><td>Lag structure</td><td>Directed effects occur over lags 1–3</td></tr><tr><td>Binary adjacency</td><td>A.npy, shape  $1 0 0 \times 1 0 0$ </td></tr><tr><td>Lag-resolved adjacency</td><td>A_1ags.npy, shape  $3 \times 1 0 0 \times 1 0 0$ </td></tr><tr><td>Adjacency convention</td><td> $A _ { i j } = 1$  denotes a directed edge from source i to target j</td></tr><tr><td>Graph density</td><td> $1 2 . 3 2 \%$  off-diagonal density</td></tr><tr><td>Important distinction</td><td>Atlas-informed prior, not empirical ENIGMA/DTI structural connectivity</td></tr></table>

Table 21: Neural source dynamics used in the simulated fMRI benchmark.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Dynamics type</td><td>Stable nonlinear lagged ROI dynamics</td></tr><tr><td>Self autoregression</td><td>0.50</td></tr><tr><td>Coupling scale</td><td>0.25 before stability rescaling</td></tr><tr><td>Spectral-radius target</td><td>0.87</td></tr><tr><td>Process noise standard deviation</td><td>0.045</td></tr><tr><td>Slow drive standard deviation</td><td>0.015</td></tr><tr><td>Pink-noise strength</td><td>0.015</td></tr><tr><td>Lag order</td><td>1-3</td></tr><tr><td>Time points per sample</td><td>1000</td></tr><tr><td>Final generator seed</td><td>20260503</td></tr></table>

The same generated system supports two evaluation settings. In the full-view setting, all 100 ROIs are observed and the dataset is used for the main non-latent causal discovery experiments. In the latentview setting, four anatomically defined ROIs are withheld from the learner input and treated as hidden variables, leaving 96 observed variables and 4 latent variables under the same full ground-truth graph. This controlled masking protocol allows us to evaluate observed-to-observed and hidden-to-observed structural recovery in the latent-augmented extension.

For the latent-view setting, the full 100-node graph is decomposed into observed and hidden blocks. Let X denote the 96 observed ROIs and Z denote the 4 masked ROIs. We store the corresponding ground-truth blocks to support block-wise evaluation of the latent-augmented model.

Before model evaluation, we checked simple diagnostic baselines to ensure that the benchmark is neither trivially easy nor unrecoverable. Random prediction matches the graph-density baseline, lagged correlation detects partial temporal signal, and Ridge-VAR remains close to random under nonlinear dynamics and hemodynamic filtering. Our method substantially improves over these diagnostics.

Scope and limitations. The simulated fMRI benchmark is TVB-inspired and atlas-grounded, but it is not a full biophysical TVB simulator. In particular, we do not claim to use empirical ENIGMA/DTI structural connectivity, Wilson–Cowan E/I neural mass dynamics, or millisecond-level axonal conduction delays. The purpose of the benchmark is to provide a controlled BOLD-like causal discovery setting with known directed ground truth, interpretable ROI variables, hemodynamic temporal distortion, and a controlled latent-view variant.

Table 22: Hemodynamic observation model for the simulated fMRI benchmark.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Observation type</td><td>BOLD-like ROI time series</td></tr><tr><td>TR</td><td>2.0 s</td></tr><tr><td>HRF type</td><td>Double-gamma</td></tr><tr><td>HRF peak delay</td><td>Uniform(4.5, 6.5) s</td></tr><tr><td>HRF undershoot delay</td><td>Uniform(12.0, 18.0) s</td></tr><tr><td>HRF undershoot scale</td><td>Uniform(0.15, 0.30)</td></tr><tr><td>BOLD noise standard deviation</td><td>0.020</td></tr><tr><td>Output normalization</td><td>Per-ROI z-scored BOLD-like time series</td></tr></table>

Table 23: Full-view and latent-view versions of the simulated fMRI benchmark.
<table><tr><td>Statistic</td><td>Full-view</td><td>Latent-view</td></tr><tr><td>Total variables in full system</td><td>100</td><td>100</td></tr><tr><td>Observed variables provided to learner</td><td>100</td><td>96</td></tr><tr><td>Withheld / latent variables</td><td>0</td><td>4</td></tr><tr><td>Time points per sample</td><td>1000</td><td>1000</td></tr><tr><td>TR</td><td>2.0 s</td><td>2.0 s</td></tr><tr><td>Ground-truth graph available</td><td>Yes</td><td>Yes</td></tr><tr><td>Graph type</td><td>Directed lagged EC</td><td>Directed lagged EC</td></tr><tr><td>Maximum lag</td><td>3</td><td>3</td></tr><tr><td>Graph density</td><td>12.32%</td><td>12.32%</td></tr><tr><td>Train / validation / test split</td><td>200 / 50 / 50</td><td>200 / 50 / 50</td></tr><tr><td>Primary metrics</td><td>AUROC, AUPRC, F1, SHD</td><td>XX / Z→X AUROC, AUPRC</td></tr></table>

## D.3 Simulated EEG Benchmark

The simulated EEG benchmark is an atlas-grounded source-space EEG-like simulation with known directed ground truth. Each variable corresponds to a cortical source ROI from the Schaefer-100 parcellation, and the directed graph follows a network-modular prior based on the Yeo-7 functional systems. Source activity is generated with lagged temporal interactions, broadband 1/f-like background activity, and theta-, alpha-, and beta-band oscillatory components. We additionally provide a simplified 64-channel synthetic source-to-sensor projection to mimic spatial mixing and volume-conduction-like leakage for visualization and robustness analysis.

The benchmark should be interpreted as a controlled source-space EEG-like simulation, not as a subject-specific biophysical EEG forward model. In particular, the synthetic projection is not intended to represent a realistic head model, BEM/FEM conductivity model, or empirical leadfield. Its purpose is to introduce interpretable source-to-sensor mixing while preserving full access to the source-level ground-truth causal graph for quantitative evaluation.

The ground-truth structure is a sparse directed lagged graph over the 100 source ROIs. Edge probabilities are network-modular: within-network interactions are sampled more densely, while cross-network interactions are retained more sparsely to preserve long-range inter-system coupling. This produces a graph that is structured but nontrivial, with dense local blocks and sparse off-block directed edges.

The binary adjacency matrix follows a source-to-target convention: $A [ i , j ] = 1$ denotes a directed edge from source ROI i to target ROI j. The lag-resolved adjacency $A _ { \mathrm { l a g s } } [ \ell , i , j ]$ follows the same source-to-target convention at lag ℓ. The stored lagged weight tensor (file W\_lags\_target\_source.npy) is indexed as $[ \ell , j , i ]$ , that is, lag, target, and source, whereas the adjacency files use the source-to-target order $[ i , j ]$

Source activity is generated with lagged directed interactions over the ground-truth source graph. To make the temporal signals more EEG-like, the generator includes broadband $1 / f$ -like background activity, mild nonstationary amplitude variation, and network-dependent oscillatory components in theta, alpha, and beta frequency ranges. The alpha band is dominant but not degenerate, while theta and beta activity remain present. This design avoids a purely periodic oscillator while preserving recoverable lagged temporal dependencies.

Table 24: Recoverability diagnostics for the simulated fMRI benchmark.
<table><tr><td>Method</td><td>AUROC / AUPRC</td><td>Interpretation</td></tr><tr><td>Random baseline</td><td>0.500 / 0.123</td><td>Expected from graph density</td></tr><tr><td>Lagged correlation max-lag</td><td>0.673 / 0.285</td><td>Simple temporal baseline has signal but remains limited</td></tr><tr><td>Ridge-VAR</td><td>0.526 / 0.133</td><td>Linear VAR remains near random</td></tr><tr><td>Ours, 5000 steps</td><td>0.823 / 0.669</td><td>Strong improvement over simple baselines</td></tr></table>

Table 25: Atlas and source-space metadata used in the simulated EEG benchmark.
<table><tr><td>Component</td><td>Setting</td></tr><tr><td>Parcellation</td><td>Schaefer-100 cortical atlas</td></tr><tr><td>Number of source ROIs</td><td>100</td></tr><tr><td>Functional labels</td><td>Yeo-7 canonical networks</td></tr><tr><td>Hemisphere labels</td><td>Left/right hemisphere labels retained</td></tr><tr><td>ROI metadata</td><td>ROI name, hemisphere, Yeo network, and atlas coordinate information</td></tr><tr><td>Use in simulation</td><td>Network-aware source graph construction</td></tr><tr><td>Use in visualization</td><td>Atlas slice visualization and network-sorted adjacency display</td></tr><tr><td>Important distinction</td><td>Source-space ROI simulation, not sensor-space empirical EEG recording</td></tr></table>

Table 26: Ground-truth source graph construction in the simulated EEG benchmark.  
Component Implementation   
Node definition Cortical source ROIs from the Schaefer-100 atlas   
Topology type Sparse directed lagged source effective-connectivity graph   
Structural prior Yeo-network-modular edge sampling   
Within-network structure Denser directed interactions within the same functional network   
Cross-network structure Sparser directed interactions across functional networks   
Lag structure Directed effects occur over lags 1–5   
Binary adjacency A.npy, shape 100 × 100   
Lag-resolved adjacency A\_lags.npy, shape $5 \times 1 0 0 \times 1 0 0$   
Weight tensor W\_lags\_target\_source.npy, shape $5 \times 1 0 0 \times 1 0 0$   
Adjacency convention A[i, j] = 1 denotes a directed edge i → j   
Weight convention W\_lags\_target\_source[ℓ, j, i] stores target-by-source weights   
Number of directed edges 1138   
Graph density 11.495%   
Important distinction Network-informed synthetic source graph, not empirical anatomical connectivity

Table 27: Source dynamics and spectral diagnostics for the simulated EEG benchmark.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Dynamics type</td><td>Stable lagged source dynamics with oscillatory components</td></tr><tr><td>Lag order</td><td>1-5</td></tr><tr><td>Time points per sample</td><td>2000</td></tr><tr><td>Frequency structure</td><td>Theta, alpha, and beta-band source activity</td></tr><tr><td>Broadband component</td><td>1/ f-like background activity</td></tr><tr><td>Nonstationarity</td><td>Slow amplitude modulation and ROI-level variability</td></tr><tr><td>Theta power</td><td>0.0887</td></tr><tr><td>Alpha power</td><td>0.3197</td></tr><tr><td>Beta power</td><td>0.0689</td></tr><tr><td>Alpha dominance ratio</td><td>2.0287</td></tr><tr><td>Estimated 1/ f spectral slope</td><td>-1.4742</td></tr><tr><td>Output normalization</td><td>Source ROI time series are normalized before model evaluation</td></tr></table>

In addition to the source-level time series used for causal recovery, we generate a 64-channel synthetic sensor projection. The projection uses a simplified source-to-sensor mixing matrix to mimic spatial leakage and volume-conduction-like mixing. This sensor projection is included for visualization and robustness analysis, but the primary causal discovery benchmark remains the source-level graph over atlas-defined ROIs.

The full sensor signal EEG.npy is generated from all 100 source ROIs. Therefore, in the latentview setting, the withheld source activity can still contribute indirectly to the full synthetic sensor measurements. To separate this from the controlled source-level latent experiment, we also store EEG\_obs96.npy, which is generated using only the 96 observed source ROIs.

Table 28: Synthetic source-to-sensor projection used in the simulated EEG benchmark.
<table><tr><td>Component</td><td>Setting</td></tr><tr><td>Number of synthetic sensors</td><td>64</td></tr><tr><td>Sensor layout</td><td>Synthetic 10–10-like layout</td></tr><tr><td>Leadfield / mixing matrix</td><td>leadfield_synthetic.npy, shape 64 × 100</td></tr><tr><td>Sensor positions</td><td>sensor_positions.npy, shape 64 × 3</td></tr><tr><td>Full sensor signal</td><td>EEG. npy, generated from all 100 source ROIs</td></tr><tr><td>Observed-only sensor signal</td><td>EEG_obs96. npy, generated from 96 observed source ROIs</td></tr><tr><td>Raw sensor signal</td><td>EEG_raw. npy, before final normalization or postprocessing</td></tr><tr><td>Visualization quantity</td><td>Relative alpha-band power over synthetic sensors</td></tr><tr><td>Primary purpose</td><td>Spatial mixing visualization and sensor-level qualitative analysis</td></tr><tr><td>Important distinction</td><td>Simplified synthetic projection, not a subject-specific BEM/FEM EEG forward model</td></tr></table>

The same generated system supports two evaluation settings. In the full-view setting, all 100 source ROIs are observed and the dataset is used for the main non-latent causal discovery experiments. In the latent-view setting, four source ROIs are withheld from the learner input and treated as hidden variables, leaving 96 observed source ROIs and 4 latent source variables under the same full groundtruth graph. The withheld ROIs are selected from different functional systems and hemispheric locations to create a controlled partially observed source-space setting. The full graph remains available only for evaluation, while the learner receives only the observed source trajectories.

Table 29: Full-view and latent-view versions of the simulated EEG benchmark.
<table><tr><td>Statistic</td><td>Full-view</td><td>Latent-view</td></tr><tr><td>Total source ROIs in full system</td><td>100</td><td>100</td></tr><tr><td>Observed variables provided to learner</td><td>100</td><td>96</td></tr><tr><td>Withheld / latent variables</td><td>0</td><td>4</td></tr><tr><td>Time points per sample</td><td>2000</td><td>2000</td></tr><tr><td>Ground-truth graph available</td><td>Yes</td><td>Yes</td></tr><tr><td>Graph type</td><td>Directed lagged source EC</td><td>Directed lagged source EC</td></tr><tr><td>Maximum lag</td><td>5</td><td>5</td></tr><tr><td>Graph density</td><td>11.495%</td><td>11.495%</td></tr><tr><td>Train / validation / test split</td><td>200 / 50 / 50</td><td>200 / 50 / 50</td></tr><tr><td>Primary input file</td><td>x.npy</td><td>x_obs.npy</td></tr><tr><td>Hidden source file</td><td>N/A</td><td>Z.npy</td></tr><tr><td>Primary metrics</td><td>AUROC, AUPRC, F1, SHD</td><td>XX / Z→X AUROC, AUPRC</td></tr></table>

For the latent-view setting, the full 100-node graph is decomposed into observed and hidden blocks. Let X denote the 96 observed source ROIs and Z denote the 4 withheld source ROIs. We store the corresponding graph blocks to support block-wise evaluation of the latent-augmented model.

Table 30: Withheld ROIs in the latent-view simulated EEG setting.
<table><tr><td>Index</td><td>ROI name</td></tr><tr><td>5</td><td>7Networks_LH_Vis_6</td></tr><tr><td>58</td><td>7Networks_RH_SomMot_1</td></tr><tr><td>86</td><td>7Networks_RH_Cont_Cing_1</td></tr><tr><td>41</td><td>7Networks_LH_Default_PFC_1</td></tr></table>

Table 31: Ground-truth graph blocks in the latent-view simulated EEG setting.
<table><tr><td>Block</td><td>Description</td></tr><tr><td>A_xx.npy</td><td>Directed edges among observed source ROIs,  $X  X$ </td></tr><tr><td>A_xz.npy</td><td>Directed edges from observed source ROIs to hidden source ROIs,  $X  Z$ </td></tr><tr><td>A_zx.npy</td><td>Directed edges from hidden source ROIs to observed source ROIs,  $Z \to X$ </td></tr><tr><td> $\mathtt { A \_ z z . n p y }$ </td><td>Directed edges among hidden source ROIs,  $Z \to Z$ </td></tr></table>

Table 32 presents preliminary recoverability diagnostics for the simulated EEG benchmark. The main multi-seed causal-recovery results and graph-score readout comparisons are reported in Table 1 and Table 12, respectively.

Table 32: Recoverability diagnostics for the simulated EEG benchmark. The “Ours” row is a one-seed smoke diagnostic on the full 100-source non-latent setting and is not a replacement for the final multi-seed experimental table.
<table><tr><td>Method</td><td>AUROC / AUPRC</td><td>Interpretation</td></tr><tr><td>Random baseline</td><td>0.500 / 0.1149</td><td>Expected from graph density</td></tr><tr><td>Lagged correlation</td><td>0.6292 / 0.1826</td><td>Simple temporal association has partial signal</td></tr><tr><td>Ridge-VAR</td><td>0.6983 / 0.2668</td><td>Linear VAR recovers part of the lagged structure</td></tr><tr><td> $\mathrm { O u r s } , 3 0 0 0 \mathrm { s t e p s }$ </td><td>0.6933 / 0.4598</td><td>Stronger sparse-edge ranking in AUPRC</td></tr></table>

Scope and limitations. The simulated EEG benchmark is atlas-grounded and EEG-like, but it is not intended to be a fully realistic biophysical EEG simulator. We do not claim to model individual head geometry, tissue conductivity, skull anisotropy, BEM/FEM forward solutions, or empirical electrode recordings. The simplified source-to-sensor projection is used only to mimic source mixing and to provide sensor-space visualizations. The purpose of the benchmark is to provide a controlled source-level temporal causal discovery setting with known directed ground truth, interpretable cortical ROI variables, oscillatory EEG-like spectral properties, and a controlled latent-view variant.

## D.4 High-dimensional Voxel-level fMRI Benchmark

The voxel-level fMRI benchmark is designed to evaluate causal discovery in a substantially larger controlled system than the ROI-level simulated fMRI benchmark. While the ROI-level benchmark contains 100 Schaefer cortical ROIs, the voxel-level benchmark contains $d = 1 0 0 0$ observed variables arranged on a $1 0 \times 1 0 \times 1 0$ spatial grid. This yields $d ( d - 1 ) = 9 9 9$ ,000 candidate directed offdiagonal relations, allowing us to stress-test scalability in a regime where exhaustive target-wise modeling becomes expensive. Unlike LargeST, the voxel-level fMRI benchmark includes a complete synthetic directed ground-truth graph, so it supports AUROC and AUPRC evaluation in addition to runtime and enrichment analysis.

The benchmark should be interpreted as a controlled high-dimensional BOLD-like temporal simulation rather than a full biophysical simulator of empirical voxel-level fMRI acquisition. The generated variables are organized into spatial regions, with local lagged dependencies between neighboring voxels and additional long-range region-to-region influences. The simulation also includes latent regional drivers and nonlinear lagged dynamics, followed by fMRI-like observation effects such as hemodynamic response kernels, downsampling, and observation noise. Stability is controlled through global inflow scaling rather than an explicitly named spectral-radius rescaling step. The purpose of this benchmark is therefore to preserve a known directed temporal graph while increasing the dimensionality to a voxel-scale setting.

For the high-dimensional results reported in this paper, we use the voxel1000\_v2 version of the benchmark. This version contains $\bar { T } = 8 0 0$ time points and a sparse directed observed graph with 7142 true off-diagonal edges out of 999,000 possible directed pairs, corresponding to a density of approximately 0.715%. The lag-resolved graph has maximum lag $p = 3$ and is collapsed into a directed observed adjacency matrix for graph-level evaluation. Self-edges are excluded throughout. Because applying source-isolated perturbation readout to all ordered pairs is computationally expensive at this scale, the voxel-level fMRI experiments use the learned edge-strength score $| W _ { i j } |$ directly for ranking candidate edges. Table 33 summarizes the benchmark configuration and evaluation settings.

Table 33: At-a-glance summary of the high-dimensional voxel-level fMRI benchmark used in the scalability experiments.
<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Dataset version</td><td>voxel1000_v2</td></tr><tr><td>Dataset type</td><td>Controlled high-dimensional BOLD-like temporal simulation</td></tr><tr><td>Observed variables</td><td>1000</td></tr><tr><td>Latent drivers</td><td>4 800</td></tr><tr><td>Time points</td><td></td></tr><tr><td>Spatial layout</td><td> $1 0 \times 1 0 \times 1 0$  voxel grid</td></tr><tr><td>Lag order</td><td>3</td></tr><tr><td>Candidate directed off-diagonal pairs</td><td>999,000</td></tr><tr><td>True directed observed edges</td><td>7142</td></tr><tr><td>Graph density</td><td>0.007149</td></tr><tr><td>Evaluation graph</td><td>Lagged edges collapsed into directed observed adjacency</td></tr><tr><td>Self-edges</td><td>Excluded</td></tr><tr><td>Score used for scalability evaluation</td><td>Learned  $| W _ { i j } |$  score</td></tr><tr><td>Top-k sparsity matching</td><td> $k = 7 1 4 \dot { 2 }$ </td></tr><tr><td>Primary metrics</td><td>AUROC, AUPRC, AUPRC enrichment, runtime</td></tr><tr><td>Primary use</td><td>High-dimensional scalability and runtime-performance evaluation</td></tr></table>

This benchmark complements both the ROI-level simulated fMRI benchmark and the real-world LargeST experiments. Compared with the ROI-level benchmark, it increases the number of variables by an order of magnitude while retaining a known synthetic directed graph for quantitative recovery evaluation. Compared with LargeST, it is not intended to model an empirical traffic or neuroimaging system, but it provides complete ground truth and therefore allows AUROC, AUPRC, and enrichment based analysis under controlled high-dimensional conditions.

## D.5 Recoverability and Evaluation Scope

The simulation datasets are intended to support controlled causal recovery evaluation with known directed ground truth. They should not be interpreted as fully realistic biomedical simulators. Instead, they provide structured temporal systems that combine recoverable directed dependencies, domaininspired organization, and controlled access to ground-truth graphs.

The low-dimensional NC8 benchmark validates whether a method can recover directed temporal dependencies in a small nonlinear system. The simulated fMRI and simulated EEG benchmarks evaluate recovery in medium-dimensional neuroimaging-inspired systems with atlas-based organization and known directed graphs. The voxel-level fMRI benchmark increases the number of variables to stress-test runtime, memory, and high-dimensional score extraction. Together, these simulations separate causal recovery evaluation, scalability analysis, and real-world qualitative assessment, while avoiding unsupported claims about exact biological realism.

## E Broader Impacts

SCDM is a scalable causal discovery framework for nonlinear temporal systems. By reducing the cost of learning directed dependency scores in high-dimensional time series, the method may support more efficient scientific analysis and decision-making in domains such as neuroscience, traffic systems, finance, climate, and engineering monitoring. In particular, scalable temporal causal discovery can help researchers explore structured dynamic relations in settings where exhaustive target-wise modeling or repeated conditional testing is computationally expensive.

However, the outputs of SCDM should be interpreted with caution. The method is based on observational temporal data and relies on assumptions such as delayed causal influence, sufficient observation of relevant variables, correct lag specification, and informative predictive risk gaps. If these assumptions are violated, learned directed scores may reflect predictive association rather than true causal mechanisms. Misuse of such scores in high-stakes settings, such as clinical decision-making, public policy, infrastructure control, or financial intervention, could lead to misleading conclusions or harmful actions. Therefore, SCDM should be used as a tool for hypothesis generation and structural analysis, and its findings should be validated with domain knowledge, controlled interventions, or external evidence whenever possible.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction state the main scope and contributions of SCDM, including shared-parameter meta-learning, target-task subsampling, post-training graph readout, and the fully observed delayed-system recovery setting. The corresponding theoretical and experimental support is provided in Sections 4.3–5 and Appendices A–B.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Section 4.3 and Section 6 state the assumptions of the recovery result (causal sufficiency, correct lag order, no instantaneous effects, positive risk gap) and explain how a hidden common cause breaks the margin condition. Section 5.2 states that SCDM is not uniformly best on small benchmarks. Appendix B.3 reports a collinearity stress test in which individual edge scores can move to correlated non-parents, and Appendix C reports controlled latent-variable diagnostics. LargeST results are presented as spatial plausibility rather than causal recovery (Section 5.4).

Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: The main finite-error recovery result is stated in Theorem 4.1, with assumptions and error terms described in Section 4.3. The full assumptions, supporting lemmas, proof, and asymptotic interpretation are provided in Appendices A.1–A.4.

Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: The paper describes the benchmark groups, baselines, metrics, thresholding protocol, runtime measurement protocol, and evaluation settings in Section 5 and Appendix B. Additional dataset details, splits, hyperparameters, and simulation setups are provided in Appendices B.1, B.2, and D.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: Code, configuration files, and data-generation scripts for all synthetic benchmarks are available at https://github.com/ShaogangRen/SCDM, with commands to reproduce each table. LargeST is publicly available from its original authors [Liu et al., 2023].

Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: Section 5.1 describes the main experimental setup, including baselines, metrics, seeds, thresholding, runtime measurement, and high-dimensional evaluation protocol. Detailed hyperparameters, dataset splits, dataset roles, and benchmark-specific settings are provided in Appendix B.1, Table 7, Table 9, Table 10, and Appendix D.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Uncertainty estimates, where reported, are defined in the corresponding table and figure captions. Table 3 reports mean ± SD across three seeds; Table 2 and Figure 6 report means across three seeds without intervals.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: Appendix B.1 lists the GPU type (NVIDIA A100 80 GB), CPU memory, and per-job time limits for each experiment. Section 5.5 reports training time for the task-batch ablation, and Appendix B.3 reports runtime and peak GPU memory for the large-scale comparisons.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The work is a methodological study of temporal causal discovery and does not involve human subjects, private personal data, or deployment of a decision-making system. The experiments use controlled synthetic, neuroimaging-inspired, high-dimensional simulated, and public traffic benchmark settings as described in Section 5 and Appendix B.2.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: Appendix E discusses both potential positive impacts, such as more efficient scientific analysis of high-dimensional temporal systems, and potential negative impacts, such as misleading conclusions or harmful decisions if observational causal scores are used in high-stakes domains without validation. The paper emphasizes that SCDM should be used for hypothesis generation and structural analysis rather than direct deployment without domain knowledge or external evidence.

## Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: The paper does not release high-risk models, scraped personal datasets, pretrained generative models, or deployed systems with substantial misuse potential. The contribution is an algorithmic framework evaluated on synthetic, neuroimaging-inspired, and traffic benchmark datasets.

Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We cite the original papers and public repositories of all baseline methods (URLs in Appendix B.1) and use them under their stated licenses. LargeST [Liu et al., 2023] is used under its CC BY-NC 4.0 license. The Schaefer-100 parcellation [Schaefer et al., 2018] and the Yeo-7 network labels [Yeo et al., 2011] are cited where used.

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [N/A]

Justification: The paper does not release a new public dataset or pretrained model as a primary asset. The synthetic and neuroimaging-inspired benchmark construction details are documented in Appendices D, D.2, D.3, and D.4.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

## Answer: [N/A]

Justification: The work does not involve crowdsourcing, user studies, or experiments with human subjects. All experiments are conducted on synthetic, neuroimaging-inspired, highdimensional simulated, or public traffic benchmark data.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: The work does not involve human-subject research, personal data collection, or crowdsourcing. Therefore, IRB approval or an equivalent human-subject review is not applicable.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [N/A]

Justification: The core methodology, theoretical analysis, experiments, and evaluation pipeline do not use LLMs as an important, original, or non-standard component. Any use of general writing or formatting assistance does not affect the scientific method or experimental results.

## Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.