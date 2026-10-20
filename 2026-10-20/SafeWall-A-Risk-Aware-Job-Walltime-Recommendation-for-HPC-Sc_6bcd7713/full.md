# SafeWall: A Risk-Aware Job Walltime Recommendation for HPC Scheduling<sup>⋆</sup>

Kanglin Xu<sup>1</sup>, Michael Papka<sup>1,2</sup>, and Zhiling Lan<sup>1,2</sup>

<sup>1</sup> University of Illinois Chicago, Chicago, IL, USA

{kxu27,papka,zlan}@uic.edu

<sup>2</sup> Argonne National Laboratory, Lemont, IL, USA

Abstract. In high-performance computing (HPC), users must submit resource requirements for their jobs, with user-supplied walltime being a critical parameter. However, these estimates are notoriously inaccurate, as users tend to overestimate walltime to prevent the scheduler from terminating their jobs prematurely. While existing work on learning-based walltime prediction primarily focuses on maximizing prediction accuracy, such predictions are inherently risky for walltime recommendations. Specifically, underestimation leads to job termination before completion, resulting in lost progress and wasted resources. In this study, we present SafeWall, a novel recommendation method that adjusts walltime predic tions by incorporating an explicit, nonnegative safety bufer to mitigate underestimation risk. The core of our approach is a risk-aware bufer that dynamically calibrates safety margins based on individual user’s histori cal behavior. When user data is limited, the system employs statistical shrinkage to align these margins with a global baseline, ensuring robust ness. This yields a lightweight mechanism that optimizes the trade-of between prediction accuracy and termination risk. SafeWall is modular and applicable to various learning-based prediction models. We evaluate SafeWall using five public workload traces, including the full-lifetime job log from the 4,392-node Theta machine at the Argonne Leadership Computing Facility. Across all traces, SafeWall substantially reduces underestimation risk compared to the unbufered base predictors, while preserving competitive accuracy. The source code is publicly available at https://github.com/SPEAR-UIC/JSSPP26-SafeWall.

Keywords: High performance computing · Batch scheduling · Walltime predic tion · Recommendation · XGBoost

## 1 Introduction

Batch schedulers in high-performance computing (HPC) systems rely on runtime information for key scheduling decisions, especially backfilling, where the scheduler opportunistically fits shorter jobs into reservation gaps. However, schedulers typically observe only the user-supplied walltime at submission; this walltime is often a conservative upper bound rather than a calibrated estimate. This mismatch creates a dilemma: an overly large walltime reduces backfilling opportunities, while a small walltime may cause premature job termination, wasting all previously performed computation. Prior work has shown that replacing raw walltime with system-generated runtime information can improve scheduler behavior compared to directly using user-supplied walltime in backfilling settings [20,16,21].

Various runtime prediction techniques have been presented for refining walltime recommendations, including both classical statistical methods and machine learning-based methods. Recent work explores a range of predictors, including ensemble learning [5] and transformer-style architectures [4], and shows that richer application-level signals can improve runtime prediction when available [14]. However, high prediction accuracy does not translate into safe walltime recommendations. Batch scheduling imposes a strict penalty for underestimation: if a recommendation is smaller than the actual runtime, the job is terminated; in contrast, a similarly sized overestimate typically only increases reserved slack. Fan et al. [8] formalize this asymmetry by emphasizing the trade-of between accuracy and underestimation rate (UR), arguing for evaluation that measures underesti mation risk rather than only prediction error. This highlights a deployment-facing gap: a state-of-the-art predictor optimized for symmetric loss functions (penalizing under- and over-prediction equally) can yield high prediction accuracy yet fail as walltime recommendations by producing a large fraction of underestimates.

In this work, we treat walltime recommendation as a risk-control problem. Specifically, we introduce SafeWall, a two-component design that separates estimation from safety. SafeWall first trains a runtime estimator using scheduler-visible features (base predictor ); it then applies a model-agnostic risk-aware adapter that produces a nonnegative safety margin (risk-aware bufer) and outputs a conservative walltime recommendation. This separation is advantageous because it decouples predictive accuracy from risk management: improvements to the base predictor directly translate to better recommendations without redesigning the safety mechanism, while the adapter can enforce explicit underestimation controls independently of the underlying model. As a result, SafeWall is modular, portable across predictors, and robust to changes in workload characteristics. Moreover, the separation makes SafeWall complementary to prior eforts on runtime prediction: any predictor that outputs a runtime estimate can be wrapped with the same bufering logic, allowing existing models to be reused without modification.

A key insight that inspired SafeWall is that underestimation risk is heterogeneous across users. Users exhibit distinct submission patterns and habits; hence, a single global safety margin can be unnecessarily conservative for most jobs while still unsafe for specific users. To address this, SafeWall introduces a user-specific safety bufer: it learns each user’s degree of conservatism from historical data and uses this information to adjust the risk-aware bufer. In cases of cold start and low activity, SafeWall shrinks the risk-aware bufer toward a global baseline, avoiding unstable or overly aggressive personalization. This yields a continuum between global and personalized risk control, providing stability for cold-start and low-activity users while exploiting user-level structure when suficient evidence exists.

We first evaluate SafeWall using the full-lifetime job log from the 4,392- node Theta machine at the Argonne Leadership Computing Facility [1]. In our experiments, we use XGBoost, a gradient-boosted decision-tree regressor for tabular scheduler logs, as the base predictor due to its strong empirical performance in prior runtime prediction studies [3,15]. We observe that high predictive accuracy does not necessarily translate to a “safe” recommendation; even with a strong regression fit, the base predictor may sufer from a high underestimation rate. By incorporating an explicit nonnegative bufer, SafeWall significantly reduces underestimation risk. As expected, more conservative bufers reduce the underestimation rate at the cost of increased prediction slack.

We further evaluate the robustness of SafeWall using multiple public workload traces collected from established repositories, such as the Parallel Workloads Archive [9,13,6,17]. Our results demonstrate that the risk-control layer introduced by SafeWall consistently reduces underestimation risk while maintaining prediction accuracy comparable to existing learning-based methods.

Overall, this work ofers the following key contributions:

– We introduce SafeWall, a practical and transparent walltime recommendation design that separates runtime estimation from risk control via an explicit nonnegative bufering layer that can wrap any base predictor (§3).

– We evaluate SafeWall using production scheduler logs from multiple systems to assess its robustness across diverse architectures and workloads (§4).

– We discuss the strengths and limitations of SafeWall (§5).

The source code is publicly available at https://github.com/SPEAR-UIC/J SSPP26-SafeWall.

## 2 Related Work

Runtime estimates are a core scheduling primitive in HPC batch systems, informing decisions such as prioritization, queue management, and backfilling. In practice, however, schedulers typically rely on user-provided walltime limits, which are often discretized and conservative upper bounds rather than accurate runtime predictions. Using these values as if they were true processing times can degrade system performance; yet, replacing them with learned predictions raises an important operational concern: underestimation can be catastrophic because jobs may be terminated at the walltime limit. Fan et al. [8] emphasize that evaluation must therefore go beyond average error and explicitly quantify one-sided risk (e.g., the underestimation rate), since improving standard accuracy measures can inadvertently increase termination risk. Related work on underestimation-aware analysis includes an erratum notice associated with a prior study line on underestimation detection [12], underscoring that measurement and evaluation details matter.

A large body of work focuses on improving the base predictor yˆ by exploring diferent model families and feature sets. Ensemble learning has been shown to improve predictive accuracy over single models in HPC runtime prediction [5], and transformer-style architectures have been proposed to capture complex nonlinear dependencies in scheduler logs [4]. Another recurring theme is feature availability: although richer application-level signals can improve prediction, they are often dificult to collect and standardize across workloads and systems. Lamar et al. [14] quantify the value of application input parameters, highlighting the potential gains from domain-specific features but also the practical barriers to deploying such approaches broadly in production settings. These results motivate methods that achieve strong performance using scheduler-visible features and lightweight historical summaries.

Beyond base prediction, some approaches integrate runtime inference into the scheduling loop. Zrigui et al. [23] propose online job runtime classification to continuously refine processing-time information used by the scheduler, demonstrating that classification-oriented runtime information can improve scheduling outcomes compared to naively using loose walltime requests. In parallel, several systems incorporate conservative margins or post-processing rules into walltime recommendations to mitigate harmful underprediction. Oztop et al. [18] propose a job grouping framework that clusters similar jobs before prediction and discuss conservative strategies to reduce underestimation risk, reinforcing the view that post-processing policies can remain valuable even when base predictors are strong. Similarly, Cui et al. [7] target backfilling with a classification-oriented pipeline and conservative recommendations to reduce operational underprediction.

Finally, the realism of evaluation depends on trace fidelity and label quality: “runtime” recorded in logs may include artifacts (e.g., measurement granularity, truncation, or missing/censored outcomes). Klus´aˇcek and Chlumsk´y [13] introduce a real-life workload trace with refined runtime estimates and show that improving labels can materially change conclusions about prediction and scheduling policies. This highlights an important practical consideration for many studies: even strong models can be shaped by the quality and semantics of the available runtime labels.

The novelty of SafeWall lies in its separation of prediction from recommendation. While prior work focuses primarily on maximizing accuracy through advanced models or richer feature sets, SafeWall introduces an explicit riskcontrol layer atop any base predictor. By decoupling runtime estimation from recommendation via a model-agnostic, nonnegative bufer that adapts to individual historical behaviors, SafeWall remains orthogonal to existing research. It serves as a complementary framework that enhances, rather than replaces, traditional walltime prediction studies.

## 3 Methodology

HPC clusters typically use batch schedulers in which users submit jobs and the scheduler decides when and where they run; each job is submitted with a walltime limit specifying its maximum allowed runtime. The objective of SafeWall is twofold: (i) to accurately estimate a job’s actual execution time, and (ii) to recommend a conservatively bufered walltime that minimizes the risk of premature termination.

As shown in Figure 1, SafeWall adopts a two-component design. First, a learning-based runtime estimator produces a runtime estimate from schedulervisible features. Second, a model-agnostic risk-aware adapter computes a nonnegative adjustment value (i.e., bufer) from historical prediction outcomes and outputs the recommended walltime. SafeWall first trains a base predictor on scheduler-visible features (and lightweight user-history summaries when available) to produce ${ \hat { y } } .$ It then calibrates a bufer from training residuals $e = y - \hat { y }$ and applies a nonnegative adjustment b—globally or in a risk-aware, shrinkage-stabilized manner—to form the final recommendation $\tilde { y } = \hat { y } + b$

Because this bufering step is applied after prediction and is calibrated using past prediction errors, it is compatible with the various runtime predictors described in Section $2 \ { \mathrm { ( e . g . } }$ , ensembles or transformer-based models). This makes SafeWall complementary to existing eforts that focus solely on improving the base predictor accuracy.

Formally, let $i \in \{ 1 , \ldots , n \}$ index the jobs. Each job is associated with a feature vector $\mathbf { x } _ { i } \in \mathbb { R } ^ { d }$ available at submission time (where d is the number of features), an observed runtime $y _ { i } \in \mathbb { R } _ { \geq 0 }$ (seconds), and a user identifier $u _ { i }$ used for optional risk-aware bufering. SafeWall learns a prediction function $f ( \cdot )$ that produces a runtime estimate

$$
{ \hat { y } } _ { i } = f ( \mathbf { x } _ { i } ) ,\tag{1}
$$

and then applies a nonnegative safety bufer to convert this estimate into a conservative walltime recommendation:

$$
\tilde { y } _ { i } = \hat { y } _ { i } + b _ { i } , \qquad b _ { i } \geq 0 .\tag{2}
$$

## 3.1 Data Transformation

HPC schedulers typically archive workload trace data in disparate formats and with inconsistent field definitions. A rigorous preprocessing pipeline is therefore essential to transform raw logs into a unified structure suitable for training walltime recommendation models.

In this study, we utilize a production workload trace from the Theta supercomputer at the Argonne Leadership Computing Facility (ALCF) [1]. We perform several preprocessing steps to transform the raw log into a supervised learning dataset, which includes standardizing fields and adjusting right-censored records (i.e., jobs that did not complete normally).

![](images/589f1e62502391d8322bec13e290d83437af747466085a01c1f5cb9e5d0cf805.jpg)  
Fig. 1: SafeWall overview. The pipeline consists of a base predictor that estimates job runtime and a risk-aware bufer adapter that adjusts runtime estimation based on user-specific historic information.

Table 1: Mapping from Theta log fields to Standard Workload Format (SWF) fields. SWF fields not available in the log are set to −1.
<table><tr><td>SWF Field</td><td>Theta Log Field</td></tr><tr><td>Submit time</td><td>queued timestamp normalized to seconds since trace start</td></tr><tr><td>Wait time</td><td>ELIGIBLE_WAIT_SECONDS</td></tr><tr><td>Run time</td><td>RUNTIME_SECONDS</td></tr><tr><td>Requested time (walltime)</td><td>WALLTIME_SECONDS</td></tr><tr><td>Allocated nodes</td><td>NODES_USED</td></tr><tr><td>Exit status</td><td>EXIT_STATUS</td></tr><tr><td>User identifier</td><td>user</td></tr><tr><td>Project identifier</td><td>group_id</td></tr></table>

– Workload log → unified trace format. We map core workload log fields (submit time, wait time, runtime, requested walltime, nodes, exit status, and user/project identifiers) into Standard Workload Format (SWF) to ensure consistent definitions and reproducibility [2]. Submit times are normalized to seconds since the first job in the trace, and jobs are ordered chronologically (Table 1).

Handle right-censored jobs via runtime capping. Jobs whose recorded runtime reaches or exceeds the requested walltime are efectively rightcensored: the true runtime may be longer, but the log provides only the walltime limit. Because SafeWall calibrates bufers from prediction residuals, treating these entries as fully observed would inject unknown truncation into the error distribution. Following common trace-processing practice, we therefore cap the runtime label at the requested walltime for such jobs. This preserves a consistent, scheduler-visible definition of “observed runtime” and avoids mixing unbounded overruns with fully observed completion times during model training and bufer estimation.

## 3.2 Feature Engineering

Next, prior to applying any prediction method, we must develop a robust feature representation to capture the essential temporal and structural characteristics of the workload.

We compute per-user lag features from prior completed jobs (previous runtime, second previous runtime, and their mean), along with summary statistics capturing request accuracy and runtime scale. For each prior job $j ,$ we define request accuracy as

$$
a _ { j } = \operatorname* { m i n } ( { \mathrm { w a l l } } _ { j } / y _ { j } , y _ { j } / { \mathrm { w a l l } } _ { j } ) \in ( 0 , 1 ] ,
$$

and include per-user summaries (mean/max) and recent-history variants over the last 10 completed jobs when available. We also include longer-run runtime summaries (mean/max, recent mean/max, and the 25th percentile of prior successful runtime) [8,22]. All history features are computed strictly from jobs submitted earlier than the current job, preventing information leakage.

Missing values and cold-start users. User-history features are undefined for users with insuficient prior completed jobs. Rather than dropping these jobs or imputing missing history, we treat them as cold-start cases and fall back to the user-provided walltime request as the recommendation. This conservative fallback avoids unsafe recommendations when there is not enough evidence to support history-based personalization. After filtering, the dataset contains 430,223 jobs; 367,477 jobs have suficient history for the full feature set and are used for model training and evaluation, while 62,746 jobs use the walltime fallback. Because batch schedulers enforce a submission-time walltime limit and may terminate jobs when that limit is reached, user-submitted walltime provides a safe default when history features are unavailable [19].

## 3.3 Base Predictors

In this study, we consider two gradient-boosted tree models as base predictors: (i) standard XGBoost trained with squared-error loss for point prediction, and (ii) a quantile gradient boosting (Quantile-GBR) configuration that directly estimates upper conditional quantiles of runtime. Tree boosting is a strong baseline for heterogeneous tabular data and efectively captures nonlinear efects and feature interactions [3,11,10].

It is important to note that SafeWall is model-agnostic; while we utilize boosted tree models in this study, the bufering mechanism can be seamlessly integrated with other learning-based predictors.

XGBoost (Squared Loss). The standard predictor is an additive ensemble of regression trees,

$$
\hat { y } _ { i } = \sum _ { k = 1 } ^ { K } f _ { k } ( \mathbf { x } _ { i } ) , \qquad f _ { k } \in \mathcal { T } ,\tag{3}
$$

where $\tau$ denotes CART-style regression trees. The model is trained to minimize squared error with regularization, as implemented in XGBoost. Because runtime is nonnegative, we enforce feasibility by clipping predictions:

$$
\hat { y } _ { i } \gets \operatorname* { m a x } ( \hat { y } _ { i } , 0 ) .\tag{4}
$$

Quantile Gradient Boosting. To directly model conservative runtime estimates, we also train a quantile boosting model that minimizes the pinball loss at quantile level τ:

$$
\mathcal { L } _ { \tau } ( y _ { i } , \hat { y } _ { i } ) = \left\{ \begin{array} { l l } { \tau ( y _ { i } - \hat { y } _ { i } ) , } & { y _ { i } \geq \hat { y } _ { i } , } \\ { ( 1 - \tau ) ( \hat { y } _ { i } - y _ { i } ) , } & { y _ { i } < \hat { y } _ { i } . } \end{array} \right.\tag{5}
$$

For $\tau > 0 . 5$ , the model emphasizes underprediction penalties and produces upperquantile runtime estimates. This configuration provides a direct alternative to post-hoc bufering.

For both models, we tune hyperparameters using randomized search with five-fold cross-validation on the training split. We select the configuration that maximizes mean cross-validated $R ^ { 2 }$ (for squared-loss models) or minimizes quantile loss (for quantile models), and then retrain the final model on the full training set. The resulting predictions on the training and test splits are subsequently passed to the SafeWall bufering policies described in Section 3.4.

## 3.4 Risk-Aware Bufer Adapter

A base prediction $\hat { y } _ { i }$ may achieve excellent accuracy while still being operationally risky for batch scheduling due to asymmetric costs: an underestimate can terminate a job at the walltime limit, whereas an overestimate typically results only in reserved slack.

![](images/9e210ceb6164bceb6510fc680deabd68d2392c958bc04dfae8cbae8707243962.jpg)  
Fig. 2: Illustrative comparison of predicted versus actual job runtimes on a log scale.

Figure 2 presents an illustrative example. Because the predictor (XGBoost) is trained with a symmetric loss, residuals are approximately balanced around the ideal line $y = \hat { y }$ , so points are distributed on both sides of the diagonal. Jobs lying above the diagonal $( y > \hat { y } )$ are underestimated and therefore at risk of termination; when errors are roughly symmetric, this can yield underestimation rates approaching 50% even when global fit is strong. Accordingly, improving base prediction accuracy alone does not guarantee scheduler safety. We therefore treat walltime recommendation as a risk-control layer applied atop the base prediction, introducing an explicit upward adjustment to systematically reduce the mass of underestimates.

Let $y _ { i }$ denote the true runtime (seconds) and ${ \hat { y } } _ { i } = f ( \mathbf { x } _ { i } )$ the base estimate from the trained regressor. Define the residual

$$
e _ { i } = y _ { i } - { \hat { y } } _ { i } ,\tag{6}
$$

so $e _ { i } > 0$ indicates underprediction. The bufer generates a nonnegative adjustment value $b _ { i } \geq 0$ and produces a conservative walltime recommendation

$$
\tilde { y } _ { i } = \hat { y } _ { i } + b _ { i } .\tag{7}
$$

To evaluate and tune bufering policies, we focus on the scheduler-facing trade-of between termination risk and conservatism.

Termination risk is captured by the underestimation rate (UR). After bufering, define the post-bufer residual as $r _ { i } = y _ { i } - \tilde { y } _ { i }$ . A job is underestimated when $r _ { i } > 0$ , meaning the recommended walltime is shorter than the actual runtime and the job would be at risk of hitting the walltime limit under hard termination. We therefore compute UR using Eq. (11) as the fraction of jobs with $r _ { i } > 0$

SafeWall derives an additive safety margin from residuals and optionally personalizes it when there is suficient user history. The design begins with a globally calibrated error bufer and then applies a user-adjusted tweak when historical evidence supports personalization. This keeps the mechanism simple and auditable while explicitly targeting the one-sided underprediction tail that drives termination risk. Specifically, SafeWall proceeds in three steps as detailed below.

Step 1: global error bufer from training residuals. We begin by calibrating a single global adjustment value using the model’s historical underprediction errors. After training the point predictor, we run it on the training set and record the residual for each job,

$$
e _ { i } = y _ { i } - { \hat { y } } _ { i } ,
$$

which measures how far the prediction is below (or above) the realized runtime. Positive residuals $( e _ { i } > 0 )$ are precisely the cases that matter for termination risk: they correspond to jobs whose true runtime exceeded the point estimate.

To translate these historical underprediction errors into a conservative shift, we choose a target underestimation rate $\tau \in ( 0 , 1 )$ (e.g., $\tau = 0 . 1 0 )$ . Intuitively, τ specifies the fraction of future jobs we are willing to allow to still exceed the recommended walltime after bufering. We then set the global adjustment value $b _ { \mathrm { g } } ( \tau )$ to a high quantile of the positive residual distribution:

$$
b _ { \mathrm { g } } ( \tau ) = Q _ { 1 - \tau } \big ( \{ e _ { i } : e _ { i } > 0 , \ i \in \mathcal { T } _ { \mathrm { t r a i n } } \ \} \big ) .\tag{8}
$$

This choice has a direct interpretation. Consider all training jobs for which the model underpredicted. By construction, $b _ { \mathrm { g } } ( \tau )$ is large enough to cover approximately a $( 1 - \tau )$ fraction of those historical underpredictions; that is, for most past underpredicted jobs, ${ \hat { y } } _ { i } + b _ { \mathrm { g } } ( \tau )$ would have met or exceeded $y _ { i }$ . When training and test conditions are similar, applying the same adjustment on new jobs aims to keep the empirical underestimation rate near τ, while exposing a clear knob that trades lower UR for higher reserved slack.

Step 2: user-adjusted tweak with shrinkage. A single global adjustment value is an average-case policy. In practice, residual behavior can be heterogeneous across users: some users routinely run longer than the model predicts (large positive residuals), while others are typically well covered by the point predictor or even overpredicted. To reflect this heterogeneity, we estimate a risk-aware adjustment value from each user’s historical underprediction residuals, and then stabilize it using shrinkage.

For each user $u ,$ let $\mathcal { E } _ { u } = \{ e _ { i } : e _ { i } > 0 , ~ u _ { i } = u , ~ i \in \mathcal { T } _ { \mathrm { t r a i n } } \}$ denote the set of that user’s positive training residuals, and let $n _ { u } = | \mathcal { E } _ { u } |$ be the number of usable underprediction cases. Using the same target underestimation rate $\tau ,$ we define a risk-aware residual quantile

$$
\begin{array} { r } { b _ { u } ( \tau ) = Q _ { 1 - \tau } ( \mathcal { E } _ { u } ) , } \end{array}
$$

so users with larger or more frequent underpredictions naturally receive a larger adjustment.

However, $b _ { u } ( \tau )$ can be noisy when $n _ { u }$ is small $( \mathrm { e . g . }$ , new or infrequent users). To avoid unstable jumps caused by limited evidence, we shrink the risk-aware estimate toward the global baseline $b _ { \mathrm { g } } ( \tau )$ using a convex combination:

$$
\tilde { b } _ { u } ( \tau ) = \frac { n _ { u } } { n _ { u } + \kappa } b _ { u } ( \tau ) + \frac { \kappa } { n _ { u } + \kappa } b _ { \mathrm { g } } ( \tau ) ,\tag{9}
$$

where $\kappa > 0$ is a smoothing parameter that determines how much history is needed before personalization dominates. This weighting has a simple interpretation: when $n _ { u }$ is large, the first term approaches $b _ { u } ( \tau )$ and the adjustment becomes strongly risk-aware; when $n _ { u }$ is small, the second term dominates and the method falls back toward the global adjustment. In the extreme cold-start case $n _ { u } = 0$ the rule reduces exactly to $\tilde { b } _ { u } ( \tau ) = b _ { \mathrm { g } } ( \tau )$

Step 3: apply the adjustment value. For a job from user $u _ { i }$ , the recommended walltime is

$$
\tilde { y } _ { i } = \hat { y } _ { i } + \operatorname* { m a x } \{ 0 , \tilde { b } _ { u _ { i } } ( \tau ) \} .\tag{10}
$$

Rolling window option. To accommodate drift, we also consider computing $b _ { u } ( \tau )$ from only the most recent W completed jobs, while keeping the same shrinkage rule toward $b _ { \mathrm { g } } ( \tau )$

## 4 Evaluation

## 4.1 Experimental Configuration

Workload Traces. We evaluate SafeWall using production scheduler logs from multiple systems to assess its robustness across diverse architectures and work-

Distribution of Jobs by Runtime and Size

![](images/1992b53c184c22c1dec962999c669ed3ceed252d7d638184ece3bb5fa7fac44c.jpg)  
Fig. 3: Joint distribution of job runtime and resource scale for the 6.5-year Theta workload. Each cell represents the frequency of completed jobs within specific runtime (rows) and node-count (columns) bins; color intensity is capped at 1,000 jobs for visual clarity.

loads. These include the Theta system at ALCF (§4.2), as well as several supercomputers from publicly available traces (§4.3). In the remainder of the paper, we use the first 80% of each workload trace for model training and report results based on the remaining 20%.

The Theta log spans approximately 6.5 years, from June 2017 to December 2023, and comprises 540,503 batch jobs submitted by 1,827 unique users across 628 projects, accounting for a total of 13.42 billion core hours [1]. For evaluation, we partition the trace into an 80% training split and a 20% held-out test split, ensuring that the predictor and bufer are calibrated using only past jobs relative to the test period.

As shown in Figure 3, the Theta workload is strongly skewed toward smallscale, short-duration jobs: roughly two-thirds of all submissions use 1–127 nodes (about 65%), and a majority finish within 30 minutes (about 58%, with a median runtime of ∼18 minutes). Multiple bins in this low-node/short-runtime region exceed the heatmap’s capped color scale (1,000 jobs per bin), consistent with a workload mix dominated by debugging, testing, and other short iterative runs. The allocation portfolio is largely Discretionary (about 72%), and users submit conservative walltime requests (median utilization/runtime-to-walltime ratio of about 36.5%), yet the scheduler still keeps the dominant small-job regime responsive (median queue wait of ∼1.4 minutes). In contrast, mid-to-large scale jobs (129–1,024 nodes) exhibit a clearer diagonal density pattern with concentrations in multi-hour runtime bands (roughly 120–500 minutes), characteristic of production-grade simulations that scale both runtime and resource use.

A notable feature is a pronounced mass around 128 nodes (about 20% of jobs), indicating a commonly chosen request size rather than a sparse boundary. Finally, the largest capability-scale jobs (2,049–4,360 nodes) are rare (on the order of ∼0.5% of submissions) but span long runtimes (up to ∼31 hours) and account for a disproportionate share of total core-hours (on the order of ∼15%), forming the resource-intensive tail of the workload.

Evaluation Metrics. Our evaluation focuses on the primary objective of Safe Wall: (i) lowering termination risk from underestimated walltimes and (ii) maintaining tight recommendations $( { \mathrm { i . e . } }$ , avoiding unnecessary over-reservation).

Let $y _ { i }$ denote the true runtime and $\tilde { y } _ { i }$ the recommended walltime produced by a method (for the no-bufer baseline, $\tilde { y } _ { i } = \hat { y } _ { i } )$ . We define the post-recommendation residual $r _ { i } = y _ { i } - \tilde { y } _ { i }$ . When $r _ { i } > 0$ , the recommendation is an underestimate and would be at risk of hitting a hard walltime limit. We quantify this risk using the underestimation rate (UR) in Eq. (11). To quantify how tight the recommendation is around the true runtime (regardless of sign), we use the symmetric ratio accuracy in Eq. (12). Finally, to capture the magnitude of these errors, we also report standard regression metrics (RMSE, MAE, and $R ^ { 2 } )$ in Eqs. (13)–(15); unless otherwise noted, these are computed between $\tilde { y } _ { i }$ and $y _ { i }$ to reflect the quality of the final recommendation.

$$
\mathrm { U R } = \frac { \mathrm { N u m b e r ~ o f ~ U n d e r p r e d i c t e d ~ J o b s } } { \mathrm { N u m b e r ~ o f ~ T o t a l ~ J o b s } } \times 1 0 0 \%\tag{11}
$$

$$
\mathrm { A c c u r a c y } = \frac { \operatorname* { m i n } \{ \tilde { y } _ { i } , y _ { i } \} } { \operatorname* { m a x } \{ \tilde { y } _ { i } , y _ { i } \} } .\tag{12}
$$

$$
\mathrm { R M S E } \ : = \ : \sqrt { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( \tilde { y } _ { i } - y _ { i } \right) ^ { 2 } } .\tag{13}
$$

$$
\mathrm { M A E } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left| \tilde { y } _ { i } - y _ { i } \right| .\tag{14}
$$

$$
R ^ { 2 } = 1 - \frac { \sum _ { i = 1 } ^ { N } \left( \tilde { y } _ { i } - y _ { i } \right) ^ { 2 } } { \sum _ { i = 1 } ^ { N } \left( y _ { i } - \bar { y } \right) ^ { 2 } } , \qquad \bar { y } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } y _ { i } .\tag{15}
$$

Comparison Methods. We evaluate SafeWall as a post-processing layer for each base predictor. In this study, we choose two base predictors: XGBoost and Quantile Gradient Boosting Regressor (Quantile GBR) [3], where $\tilde { y } _ { i } = \hat { y } _ { i }$ (no post-processing). For SafeWall, it adjusts predicted output by applying a risk-aware, user-specific bufer given by $\tilde { y } _ { i } = \hat { y } _ { i } + b _ { i }$ . We compare SafeWall with the corresponding methods:

– Baselines. We consider two base predictors:

(a) XGBoost, a gradient-boosted decision tree regressor trained with squarederror loss to produce point runtime predictions from submission-time features.

(b) Quantile GBR, a Gradient Boosting Regression model trained with quantile loss at $q = 0 . 9 5$ , predicts a conservative runtime at the 95th percentile conditional on features (aiming for ${ \approx } 5 \%$ underestimation under good calibration).

– Naive prediction methods. We consider three naive bufering methods:

(a) Percentage bufer , which inflates each prediction by a fixed factor, $\tilde { y } _ { i } =$ $( 1 + 0 . 2 ) \hat { y } _ { i }$

(b) Fixed-value $b u f f e r .$ , which adds a constant margin of 30 seconds to every prediction, $\tilde { y } _ { i } = \hat { y } _ { i } + 3 0 \mathrm { s }$

(c) Piecewise adaptive $b u f f e r$ , which applies a piecewise percentage rule:

$$
\begin{array} { r } { \tilde { y } _ { i } = \left\{ \begin{array} { l l } { ( 1 + 0 . 1 0 ) \hat { y } _ { i } , } & { \hat { y } _ { i } \leq T _ { 1 } , } \\ { ( 1 + 0 . 1 5 ) \hat { y } _ { i } , } & { \hat { y } _ { i } > T _ { 1 } , } \end{array} \right. } \end{array}\tag{16}
$$

where the threshold $T _ { 1 }$ is fixed in advance.

For the risk-aware bufer, we use target underestimation rate $\tau = 0 . 1 0$ in Eq. (8) and the shrinkage parameter $\kappa = 1 0$ in Eq. (9). This configuration provides a stable transition from global to personalized bufering while limiting sensitivity to older user behavior. This combination allows us to quantify the benefit of SafeWall’s risk-aware, user-adaptive bufering relative to both straightforward heuristics and an established conservative pipeline.

Our evaluation is organized around three questions. First, §4.2 quantifies the performance of SafeWall on the Theta log across prediction strategies. Second, §4.3 evaluates SafeWall on additional workload traces to test SafeWall’s robustness across diferent systems and workloads. Finally, §4.4 examines the variance in prediction errors across individual users, demonstrating why personalized calibration improves the risk–tightness trade-of.

## 4.2 Results on Theta

Table 2 highlights the operational gap between predictive fit and safe walltime recommendations. Despite excellent test fit, using the base predictor directly is unsafe (UR ≈ 47.85%). We also include the job-grouping based resource prediction framework of Oztop et al. [18] by running their released code on the same Theta split. Under the “resampled $x \mathrm { f } ^ { \prime \prime }$ configuration with a two-sigma bias setting, the reproduced baseline achieves $\mathrm { U R } = 3 . 0 6 \%$ but with much larger error (RMSE = 18,505.64s, MAE = 14,733.11s). In contrast, SafeWall attains a comparable UR in the 3–4% range (e.g., 4.31% for XGBoost+SafeWall and 3.46% for Quantile+SafeWall) while maintaining substantially lower error (RMSE ≈ 960–970s, MAE ≈ 90–150s), indicating that SafeWall controls termination risk at a similar service level while preserving base-estimation fidelity.

Table 2: Comprehensive comparison of bufering strategies on the Theta log.
<table><tr><td>Method</td><td></td><td>UR (%) Accuracy (%)</td><td>RMSE(s)</td><td>MAE(s)</td><td> $R ^ { 2 }$ </td></tr><tr><td>XGBoost</td><td>47.8489</td><td>97.20</td><td>961.44</td><td>57.34</td><td>0.9835</td></tr><tr><td>XGBoost + SafeWall</td><td>4.3051</td><td>90.27</td><td>963.89</td><td>91.81</td><td>0.9834</td></tr><tr><td>Quantile GBR Quantile GBR + SafeWall</td><td>5.5568</td><td>85.44</td><td>685.90</td><td>159.20</td><td>0.9916</td></tr><tr><td></td><td>3.4614</td><td>87.70</td><td>770.25</td><td>147.55</td><td>0.9895</td></tr><tr><td>Percentage buffer</td><td>1.0613</td><td>82.83</td><td>1808.66</td><td>547.43</td><td>0.9418</td></tr><tr><td>Fixed-value buffer</td><td>7.0521</td><td>89.91</td><td>961.85</td><td>77.54</td><td>0.9835</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Piecewise buffer</td><td>2.5893</td><td>82.49</td><td>1484.90</td><td>384.39</td><td>0.9608</td></tr></table>

Table 3: Workload traces used for cross-system evaluation. “Cens.” denotes the fraction of censored jobs whose true runtimes are not observed in the trace (e.g., terminated at the walltime limit or otherwise missing a completion time).
<table><tr><td>Trace</td><td>Period</td><td>Jobs</td><td>Users</td><td>Nodes</td><td>Cens. (%)</td></tr><tr><td>Theta</td><td>Jun 2017–Dec 2023</td><td>540,503</td><td>1,827</td><td>4,392</td><td>0.1</td></tr><tr><td>Cori</td><td>Jan-Dec 2022</td><td>2,462,279</td><td>1,318</td><td>10,000</td><td>0.0</td></tr><tr><td>KIT-FH2</td><td>Jun 2016–Jan 2018</td><td>114,024</td><td>164</td><td>1,173</td><td>6.6</td></tr><tr><td>MetaCentrum</td><td>Jan-Dec 2023</td><td>10,139,198</td><td>1,618</td><td>47 clust.</td><td>4.9</td></tr><tr><td>CERIT-SC-K8s</td><td>Jul 2021–Jun 2022</td><td>64,093</td><td>103</td><td>20</td><td>0.0</td></tr></table>

We also compare against a quantile regression baseline implemented as a quantile Gradient Boosting Regressor, which directly predicts an upper conditional runtime quantile as the recommendation. As shown in Table 2, SafeWall achieves lower UR at comparable or better accuracy than the quantile models at similar operating points, while keeping the underlying point predictor unchanged. In addition, SafeWall is significantly lighter-weight operationally: changing the service level only requires recomputing a residual quantile from the already-trained model, whereas the quantile approach typically requires training (and validating) a separate model for each target quantile.

Overall, SafeWall should be viewed as a risk-control layer applied on top of an existing runtime predictor, rather than a replacement for the predictor itself, decoupling safety guarantees from predictive modeling.

## 4.3 Cross-System Robustness

To demonstrate that SafeWall generalizes beyond a single production system, we evaluate it on four additional public workload traces spanning a wide range of system scales, architectures, and scheduling paradigms (Table 3). All traces are obtained from the Parallel Workloads Archive [9], NERSC [17], JSSPP Workloads [6,13], and difer substantially in workload composition and user behavior.

Distribution of Jobs by Runtime and Size (CERIT-SC-K8s)  
Table 4: Runtime prediction across traces: base model and SafeWall.
<table><tr><td colspan="2"></td><td colspan="2">XGBoost</td><td colspan="2">SafeWall</td></tr><tr><td>Trace</td><td>#Jobs</td><td>UR (%)</td><td>Acc (%)</td><td>UR (%)</td><td>Acc (%)</td></tr><tr><td>Cori</td><td>2,465,286</td><td>47.92</td><td>93.47</td><td>5.06</td><td>83.21</td></tr><tr><td>KIT</td><td>114,355</td><td>49.23</td><td>90.13</td><td>6.79</td><td>74.76</td></tr><tr><td>CERIT</td><td>64,268</td><td>49.11</td><td>92.14</td><td>5.00</td><td>83.02</td></tr><tr><td>MetaCentrum</td><td>10,145,231</td><td>48.16</td><td>84.03</td><td>4.92</td><td>68.64</td></tr></table>

![](images/c1b55a9c3bf304e6b5e594e5f5c76e10135f78700362a8c6d9507b410cacf1b1.jpg)  
(a) KIT-FH2

![](images/1a50d0bda411b8b4bc3d06aa8641e4020da0d19d78f7549ad0b1b6ed21f3da35.jpg)  
(b) Cori

![](images/279494e7e0ca6e5cdbea570d9323ab5513451f2b6fd4db72e80d06d54745cd7a.jpg)  
(c) MetaCentrum

![](images/58706a5439701419f4fbfb13a395790ebcc86ac53e9db6fb64402206c992c794.jpg)  
(d) CERIT-SC-K8s  
Fig. 4: Joint distribution of job runtime and job size across cross-system traces. Each cell shows the number of completed jobs in a runtime bin (rows) and size bin (columns); color intensity is capped for visual clarity.

Cori [17] is a Cray XC40 system at the National Energy Research Scientific Computing Center (NERSC) comprising both Intel Knights Landing and Haswell partitions. Its 2022 trace contains approximately 2.46 million jobs submitted by 1,318 users, representing a large-scale leadership-class HPC workload.

KIT-FH2 [9] corresponds to the ForHLR II cluster at the Karlsruhe Institute of Technology (1,173 nodes, 24,048 cores). Its 19-month trace includes about 114,000 jobs from 164 users and exhibits a notable 6.6% censoring rate, indicating frequent walltime exhaustion.

MetaCentrum [13] is the Czech National Grid Infrastructure, a federated environment spanning 47 clusters and 34,400 cores across 42 queues. Its 2023 annual log is the largest dataset in our study, with over 10.1 million jobs from 1,618 users and a 4.9% censoring rate.

CERIT-SC-K8s [6] is a 20-node Kubernetes cluster at CERIT-SC (Czech Republic), where each workload unit corresponds to a single-node container pod. Its 11-month trace comprises roughly 64,000 pods from 103 namespaces and represents a containerized scheduling paradigm distinct from traditional batch systems.

Despite their architectural and operational diversity, all four traces exhibit the heavily right-skewed runtime distributions characteristic of HPC and cloud workloads. Median runtimes are short—approximately 17 minutes for Cori, 10 minutes for KIT-FH2, 22 minutes for MetaCentrum, and roughly 3 minutes for CERIT-SC-K8s—yet mean runtimes are 3–67× larger than the respective medians, reflecting long tails of resource-intensive jobs that dominate total consumption. Skewness ranges from 4.1 on KIT-FH2 to 31.6 on CERIT-SC-K8s, underscoring the prevalence of extreme runtimes across systems. Figure 4 further visualizes this pattern via the joint runtime–size distributions, where the mass concentrates in short, small jobs with a sparse but important long-duration/large-scale tail.

User walltime behavior also varies substantially across traces. MetaCentrum users are particularly conservative, with a median runtime-to-walltime ratio of only about 3.5%, compared with roughly 36.5% observed on Theta; KIT-FH2 falls between these extremes. These diferences in utilization patterns, censoring rates, and scheduling models (batch, grid, and Kubernetes) create a challenging generalization setting for any walltime recommendation method.

We apply the same two-component pipeline on all four traces: first training a point runtime predictor using submission-time features, then calibrating a nonnegative SafeWall bufer using training residuals only. Across all datasets, the base predictor achieves strong predictive fit (Table 4), yet the unbufered underestimation rate remains close to 50%, consistent with the behavior of symmetric-loss regression under skewed error distributions. SafeWall consistently reduces underestimation risk across systems while preserving predictive fidelity, demonstrating robustness to diferences in scale, workload mix, and scheduling paradigm.

## 4.4 Analyzing the Impact of Risk-Aware Bufering

Prediction errors are not uniformly distributed across users. Among “stable” test users (those with at least 10 jobs), 35.8% exhibit a positive mean residual (systematic under-prediction) and 64.2% exhibit a negative mean residual (systematic over-prediction). Moreover, user mean-residual bias shows persistence from train to test (Pearson correlation 0.167 among stable users). This empirical heterogeneity motivates risk-aware bufering: a single global margin is an average-case policy that can remain unsafe for some users while being unnecessarily conservative for others.

## 5 Discussion

SafeWall is motivated by a practical mismatch between how runtime predictors are typically trained and how their outputs are used as walltime limits in batch scheduling. Our results suggest that explicitly separating estimation (predict yˆ) from risk control (add a nonnegative margin b) is an efective way to convert strong base predictions into scheduler-safe recommendations. Below, we summarize the main strengths, limitations, and implications for deployment.

## 5.1 Strengths

SafeWall’s main strength is its modular two-component design. The bufer adapter relies only on historical residuals and can wrap any base predictor, including future models with richer features, without altering the safety logic. Improvements in prediction quality directly benefit, while the risk-aware bufering layer provides a separate, auditable mechanism for controlling underestimation risk.

From an operational standpoint, SafeWall is simple and tunable. A single parameter (e.g., a target underestimation rate or residual percentile) directly controls the trade-of between recommendation tightness and slack. Because bufering is applied post-training, SafeWall remains model-agnostic, allowing for straightforward validation and deployment. Moreover, risk-aware bufers ofer a practical balance between global and personalized policies: it adapts to individual user behaviors when historical data are abundant, while employing shrinkage toward a global baseline to mitigate instability in data-sparse or cold-start scenarios.

## 5.2 Limitations

SafeWall’s calibration depends on the semantics and quality of runtime labels. In real scheduler logs, jobs that reach their requested walltime are right-censored: the recorded runtime reflects the walltime limit rather than the job’s true completion time. If such walltime hits are common, then residuals computed from observed runtimes can understate the positive tail (i.e., the most harmful underpredictions), which may lead to bufers that are not conservative enough. In these settings, censored records should be handled explicitly during calibration. Two practical options are: (i) treat walltime-hit jobs as incomplete observations and incorporate them as conservative lower bounds when estimating tail residuals (so the bufer does not rely on artificially capped runtimes), or (ii) leverage refined-runtime or soft-walltime traces that reconstruct more accurate runtimes for censored jobs and thus provide cleaner residual signals for bufer estimation [13].

SafeWall assumes residual behavior is reasonably stable between calibration and use. Significant shifts in workload composition, software environments, or scheduling policies can alter residual distributions and degrade calibration accuracy. While rolling-window updates can mitigate such drift, they introduce additional hyperparameter tuning requirements.

Finally, personalization may introduce unequal service levels, as harder-topredict users receive larger bufers. Depending on site policy, group-based bufering policies (per partition, project, or job class) may be appropriate to ensure equitable resource access.

## 6 Conclusion

This work addressed the practical gap between accurate runtime prediction and safe walltime recommendation in batch-scheduled HPC systems. Using the 6.5-year Theta log, we showed that even strong symmetric predictors can yield unacceptably high underestimation rates under hard walltime limits. To address this, we introduced SafeWall, a two-component approach that separates estimation from risk control by augmenting a base predictor with a model-agnostic, risk-aware bufering layer that explicitly manages one-sided termination risk. Experiments on the Theta trace and multiple public datasets demonstrate that SafeWall outperforms existing methods by ofering a well-defined accuracy-risk trade-of; specifically, its residual-calibrated bufers provide predictable operating points, while the risk-aware bufering achieves lower underestimation rates than global or prior conservative approaches without introducing unnecessary slack. Overall, the results support treating walltime recommendation as an explicit risk-control problem.

SafeWall provides a modular framework that complements existing runtime prediction models. As a lightweight solution relying solely on submission-time features and historical data, it can be deployed as either a standalone recommendation service or a submission-time wrapper that automatically adjusts walltime requests. The source code is publicly available at https://github.com/SPEAR -UIC/JSSPP26-SafeWall.

Future work includes making calibration conditional on job attributes (e.g., queue/partition, node count, or project) so the bufer adapts to systematic heterogeneity beyond users, and integrating SafeWall into a scheduler workflow to evaluate system-level impacts such as backfilling eficiency, utilization, and wait time. We also plan to handle right-censored jobs explicitly in environments where walltime hits are frequent, and to pair SafeWall with uncertainty-aware base predictors (e.g., quantile or conformal calibration) to strengthen reliability while preserving the same modular two-component design.

## References

1. Argonne Leadership Computing Facility: Theta/ThetaGPU. https://www.alcf.a nl.gov/alcf-resources/theta, accessed 2026-02-07

2. Chapin, S.J., Cirne, W., Feitelson, D.G., Jones, J.P., Leutenegger, S.T., Schwiegelshohn, U., Smith, W., Talby, D.: Benchmarks and standards for the evaluation of parallel job schedulers. In: Feitelson, D.G., Rudolph, L. (eds.) Job Scheduling Strategies for Parallel Processing, IPPS/SPDP’99 Workshop, JSSPP’99, San Juan, Puerto Rico, April 16, 1999, Proceedings. Lecture Notes in Computer Science, vol. 1659, pp. 67–90. Springer (1999). https://doi.org/10.1007/3-540 -47954-6\_4

3. Chen, T., Guestrin, C.: XGBoost: A scalable tree boosting system. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. pp. 785–794. KDD ’16, Association for Computing Machinery, New York, NY, USA (2016). https://doi.org/10.1145/2939672.2939785

4. Chen, X.: Job runtime prediction of HPC cluster based on PC-Transformer. The Journal of Supercomputing (2023). https://doi.org/10.1007/s11227-023-054 70-2

5. Chen, X., Zhang, H., Bai, H., Yang, C., Zhao, X., Li, B.: Runtime prediction of high-performance computing jobs based on ensemble learning. In: Proceedings of the 2020 4th International Conference on High Performance Compilation, Computing and Communications (HP3C 2020). pp. 56–62 (2020). https://doi.org/10.1145/ 3407947.3407968

6. Chlumsk´y, V., Klus´aˇcek, D.: Improving accuracy of walltime estimates in PBS professional using soft walltimes. In: Job Scheduling Strategies for Parallel Processing (JSSPP 2022), Lecture Notes in Computer Science, vol. 13592, pp. 192–210. Springer (2023). https://doi.org/10.1007/978-3-031-22698-4\_10

7. Cui, H., Takahashi, K., Shimomura, Y., Takizawa, H.: Clustering based job runtime prediction for backfilling using classification. In: Klus´aˇcek, D., Corbal´an, J., Rodrigo, G.P. (eds.) Job Scheduling Strategies for Parallel Processing. pp. 40–59. Springer Nature Switzerland, Cham (2025)

8. Fan, Y., Lan, Z., Rich, P., Allcock, W.E., Papka, M.E.: Trade-of between prediction accuracy and underestimation rate in job runtime estimates. In: 2017 IEEE International Conference on Cluster Computing (CLUSTER) (2017). https: //doi.org/10.1109/CLUSTER.2017.53

9. Feitelson, D.G.: Parallel workloads archive. https://www.cs.huji.ac.il/labs/p arallel/workload/, accessed 2026-02-07

10. Friedman, J.H.: Greedy function approximation: A gradient boosting machine. The Annals of Statistics 29(5), 1189–1232 (2001). https://doi.org/10.1214/aos/10 13203451

11. Grinsztajn, L., Oyallon, E., Varoquaux, G.: Why do tree-based models still outperform deep learning on tabular data? (2022), https://arxiv.org/abs/2207.08815

12. Guo, X., et al.: Erratum to: Machine learning predictions for underestimation of job runtime on HPC system (2018), erratum notice (afiliation correction)

13. Klus´aˇcek, D., Chlumsk´y, V.: Real-life HPC workload trace featuring refined job runtime estimates. In: Job Scheduling Strategies for Parallel Processing (JSSPP

2024), Lecture Notes in Computer Science, vol. 14591, pp. 1–19. Springer (2024). https://doi.org/10.1007/978-3-031-74430-3\_1, first Online: 21 Dec 2024

14. Lamar, K., Goponenko, A.V., Aaziz, O., Allan, B.A., Brandt, J.M., Dechev, D.: Evaluating HPC job run time predictions using application input parameters. In: Proceedings of the 17th ACM International Conference on Distributed and Event-Based Systems (DEBS ’23). pp. 127–138 (2023). https://doi.org/10.1145/3583 678.3596893

15. Menear, K., Nag, A., Perr-Sauer, J., Lunacek, M., Potter, K., Duplyakin, D.: Mastering HPC runtime prediction: From observing patterns to a methodological approach. In: Practice and Experience in Advanced Research Computing (PEARC ’23) (2023). https://doi.org/10.1145/3569951.3593598, https://www.nrel.g ov/docs/fy23osti/86526.pdf, preprint: NREL/CP-2C00-86526

16. Mu’alem, A.W., Feitelson, D.G.: Utilization, predictability, workloads, and user runtime estimates in scheduling the IBM SP2 with backfilling. IEEE Transactions on Parallel and Distributed Systems 12(6), 529–543 (2001). https://doi.org/10 .1109/71.932708

17. NERSC: Cori – NERSC documentation. Online documentation (2023). https: //doi.org/10.25344/S4859K, accessed 2026-02-07

18. Oztop, B., Schwaller, B., Leung, V.J., Brandt, J., Kulis, B., Egele, M., Coskun, A.K.: Job grouping based intelligent resource prediction framework. In: Klus´aˇcek, D., Corbal´an, J., Rodrigo, G.P. (eds.) Job Scheduling Strategies for Parallel Processing. pp. 21–40. Springer Nature Switzerland, Cham (2026)

19. SchedMD: sbatch – submit a batch script. https://slurm.schedmd.com/sbatch.h tml (2025), accessed 2025-01-01

20. Srinivasan, S., Kettimuthu, R., Subramani, V., Sadayappan, P.: Characterization of backfilling strategies for parallel job scheduling. In: Proceedings of the International Conference on Parallel Processing Workshops (ICPPW). pp. 514–519. IEEE (2002). https://doi.org/10.1109/ICPPW.2002.1039773

21. Tsafrir, D., Etsion, Y., Feitelson, D.G.: Backfilling using system-generated predictions rather than user runtime estimates. IEEE Transactions on Parallel and Distributed Systems 18(6), 789–803 (2007). https://doi.org/10.1109/TPDS.200 7.70606

22. Zhang, H., You, H., Hadri, B., Fahey, M.R.: HPC usage behavior analysis and performance estimation with machine learning techniques. In: Proceedings of the 2012 International Conference on High Performance Computing and Simulation (HPCS) (2012), https://api.semanticscholar.org/CorpusID:2098182

23. Zrigui, S., de Camargo, R.Y., Legrand, A., Trystram, D.: Improving the performance of batch schedulers using online job runtime classification. Journal of Parallel and Distributed Computing (2022). https://doi.org/10.1016/j.jpdc.2022.01.003