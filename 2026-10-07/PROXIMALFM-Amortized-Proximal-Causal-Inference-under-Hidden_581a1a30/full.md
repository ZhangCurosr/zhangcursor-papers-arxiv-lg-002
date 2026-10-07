# PROXIMALFM: Amortized Proximal Causal Inference under Hidden Confounding

Christophe Muller<sup>1\*</sup>, Ayub Kharel<sup>1</sup>, Alex Luedtke<sup>2</sup>, Chan Park<sup>3</sup>, Eric Tchetgen Tchetgen<sup>4</sup>, Juan L. Gamella<sup>5</sup>, Rahul Krishnan<sup>6</sup>, Ricardo Silva<sup>7</sup>, Jakob Zeitler<sup>1,8</sup>

<sup>1</sup> University of Oxford <sup>2</sup> Harvard University <sup>3</sup> University of Illinois Urbana-Champaign

<sup>4</sup> Perelman School of Medicine, University of Pennsylvania <sup>5</sup> Causal Chamber

<sup>6</sup> U. of Toronto & Vector Institute <sup>7</sup> University College London <sup>8</sup> SMARTbiomed, University of Oxford

## Abstract

Standard causal identification methods often assume no unmeasured confounding and can fail when relevant confounders are unobserved. Proximal causal inference instead uses proxy variables to identify effects under hidden confounding and additional assumptions. However, nonparametric proximal estimation can be challenging in practice: recovering causal estimands such as the conditional average treatment effect (CATE) requires solving an ill-posed integral equation that is data-hungry, hyperparameter-sensitive, and optimization-unstable. Bayesian inference for such models provides a desirable alternative, mitigating these difficulties by regularizing through the prior. However, computing a posterior is itself challenging, as a typical likelihood function will include latent variables. Following the recent success of tabular foundation models in backdoor, instrumenta variable, and frontdoor settings, we propose that prior-data fitted networks (PFNs) are uniquely suited to resolve this bottleneck. Indeed, by training on synthetic data sampled from compliant structural causal models with access to oracle counterfactuals, we simplify the task substantially, amortizing the implied Bayesian operator inversion into a single transformer forward pass. Compared to prior literature that focuses primarily on point estimation, our model, PROXIMALFM, explicitly targets the Bayesian posterior distribution of the CATE. One unique aspect of this problem is that we need to provide Monte Carlo estimates of the oracle CATEs, leading to a novel variation of PFNs that accounts for the added stochastic error. Across a diverse suite of proximal regimes, PROXIMALFM achieves consistently strong CATE-estimation performance without dataset-specific tuning, with its largest advantage when latent confounding is substantial and the proxies are weakly informative; it also provides fast inference through a single amortized forward pass.

## Code and reproducibility:

https://github.com/ChristopheMuller/ProximalFM

## 1 Introduction

Causal inference from observational data is a fundamental task across disciplines, yet its validity typically hinges on the no unmeasured confounding assumption. This assumption requires that investigators have measured a sufficiently rich set of covariates to ensure that supervised regression and classification functions can, in theory, also provide how an outcome varies under intervention on a possible cause of interest (Hernan & Robins, 2020),´ with specialized estimation methods improving on off-the-shelf uses of supervised learning. In practice, however, there are often unmeasured variables that are plausible confounders (e.g., unrecorded disease severity).

One of the most common tasks in causal inference is estimating the conditional average treatment effect (CATE) function: how an outcome variable Y potentially varies in expectation, as we vary a treatment variable A by intervening on the latter. This is a function of an observed realization of a vector of covariates X. Using the notation $Y ( a )$ to denote the potential outcome under a hypothetical intervention on A at level a (Rubin, 1972), the CATE function for a binary A is given by

$$
Q ( x ) = \mathbb { E } [ Y ( 1 ) - Y ( 0 ) \mid X = x ] .\tag{1}
$$

This is identifiable from observational data if unconfoundedness holds. That failing, other assumptions are needed. Proximal causal learning has emerged as a principled framework to identify and estimate causal effects when hidden confounders cannot be ruled out (Tchetgen Tchetgen et al., 2020). The main idea is to exploit auxiliary variables, proxies, which are associated with the hidden confounders without fully removing their influence. However, using proxy information while avoiding parametric models requires solving ill-posed integral equations, leading to a difficult estimation problem. As a result, current nonparametric (Ghassami et al., 2022; Sverdrup & Cui, 2023; Yang et al., 2023) and neural estimation methods (NMMR, Kompa et al., 2022) are computationally demanding, requiring intensive, dataset-specific optimization.

Simultaneously, the machine learning community is shifting toward amortized inference via foundation models. Prior-data fitted networks (PFNs, Muller et al., 2022) have demonstrated that transformers can learn to perform¨ Bayesian inference through in-context learning. Models like TabPFN and its successors have shown state-of-theart performance on small tabular datasets without requiring retraining (Grinsztajn et al., 2026). As we review in the next section, recent work has extended PFN-style models to causal inference settings. However, they generally do not make use of proximal assumptions, limiting their utility in the presence of hidden confounders. Our contributions are:

1. PROXIMALFM: a foundation model for proximal causal learning. We introduce PROXIMALFM, the first tabular foundation model designed for proximal causal inference. It leverages proxy variables without requiring dataset-specific training at deployment. Causal effects are obtained in a single forward pass through a target dataset, providing a Bayesian solution to an ill-posed inverse problem without hyperparameter tuning.

2. A structural prior and Monte Carlo CATE targets for proximal causal learning. We introduce a diverse prior over structural causal models that enforces the proximal assumptions for identification. We use Monte Carlo integration to approximate the CATE. These approximations are then used as targets to train PROXI-MALFM. This prior determines the inductive bias regularizing the proximal estimation problem amortized by PROXIMALFM.

3. Comprehensive evaluation, from semi-synthetic data to physical systems. We evaluate PROXIMALFM on diverse semi-synthetic proximal benchmarks that combine real-world covariate distributions with known, controlled causal mechanisms. Beyond semi-synthetic benchmarks, we take a first step towards evaluation on real physical systems using Causal Chambers (Gamella et al., 2025), a controlled physical testbed for causal inference.

## 1.1 Related work

Tabular foundation models Recent advances have demonstrated the potential of transformer-based foundation models for tabular prediction. TabPFN pioneered this approach for supervised learning tasks with tabular data (Grinsztajn et al., 2026; Hollmann et al., 2023, 2025). This success prompted a rapidly expanding literature. Examples include work on more efficient tabular in-context learning, such as TabICL (Qu et al., 2025, 2026), and tuning on real-world data, as in RealTabPFN (Garg et al., 2025). This progress has also motivated standardized evaluation. TabArena provides a living benchmark for IID tabular classification and regression (Erickson et al., 2026). Extending this evaluation beyond conventional IID settings, Purucker et al. (2026) find that current tabular foundation models are particularly strong on small- to medium-scale IID datasets, but do not yet consistently outperform conventional methods on non-IID, large-scale, or high-dimensional data.

Amortized causal learning The success of tabular models has motivated their application to causal inference. Causal foundation models are pretrained on a prior over causal data-generating processes and use in-context learning to estimate causal quantities on new datasets in a single forward pass. Stith, Rahmani, and Cresswell (2026) provide a recent overview of this emerging literature.

Robertson et al. (2026) introduced Do-PFN, demonstrating that transformer-based foundation models can learn causal estimation procedures directly from synthetic data and perform zero-shot inference on unseen datasets. Balazadeh Meresht et al. (2026) introduced CausalPFN, a framework for doing amortized learning of causal effects identified through unconfoundedness. This setting was also considered in Luedtke and Chung (2024), under an adversarially chosen prior. Ma et al. (2026) proposed CausalFM, extending causal foundation models beyond unconfoundedness, such as the case of identifiable additive-error instrumental-variable (IV) settings. Rather than learning a single causal estimator, CausalFM trains separate models for distinct causal identification strategies, thereby preserving the classical separation between identification and estimation. Balazadeh et al. (2026) relaxes the assumption of additive-error in the IV setting to build in-context learners for bounding causal effects for instrumental variables. Different from proximal learning, such partial identification IV methods do not typically exploit multiple proxies, nor exploit completeness assumptions (as defined in the next section) that could otherwise allow for identification.

![](images/0d929fccaba15f66238e4b547181c05a9881c32cf144c0f9802cac16410c1c34.jpg)  
Figure 1: Examples of DAGs compatible with proximal causal identification assumptions. In every panel, Z and $W$ are informative proxies for the unmeasured confounder U. Direct Z–A edges (in either direction) and a direct $W  Y$ edge are optional.

Despite these advances, a foundation model designed for proximal causal inference remains absent from the literature.

## 2 Proximal causal inference

We consider an observational causal inference setting where the standard assumption of no unmeasured fails. Our objective is to estimate the CATE, as in Eq. (1), for binary treatments. Let the observed variables be $O =$ $( L , A , Y )$ , where $A \in \{ 0 , 1 \}$ denotes a treatment of interest, Y is the outcome, and L is a set of measured covariates. The unmeasured confounders are denoted by U.

In the presence of $U ,$ the standard exchangeability condition $Y ( a ) \perp \perp A \mid L$ is violated (Hernan & Robins,´ 2020). To achieve identification, we adopt the proximal causal learning framework (see, e.g., Miao et al., 2018; Tchetgen Tchetgen et al., 2020). This framework partitions L into three subsets, each with a distinct role. Let $\boldsymbol { L } = ( X , Z , W )$ where X contains baseline covariates, while Z and W are complementary proxies for $U , Z$ is a treatment-confounding proxy: while it carries information about the latent U, it may otherwise be related directly to treatment assignment. Conversely, W is an outcome-confounding proxy which carries information about U and may otherwise directly shape the outcome. Figure 1 illustrates some admissible proximal directed acyclic graphs (DAGs), with formal restrictions stated in the following section.

## 2.1 Identifying assumptions

To identify the causal effect $Q ( x )$ , it suffices for the joint distribution of $( \boldsymbol { X } , \boldsymbol { U } , \boldsymbol { Z } , \boldsymbol { W } , \boldsymbol { A } , \boldsymbol { Y } )$ to satisfy the following conditions:

Assumption 1 (Latent exchangeability). The potential outcomes are independent of the treatment assignment conditional on the observed covariates X and the unmeasured confounders U:

$$
Y ( a ) \downarrow \downarrow A \mid ( X , U ) f o r e a c h a \in \{ 0 , 1 \} .
$$

Assumption 2 (Proxy exclusion restrictions). The proxies must satisfy conditional independencies relative to the treatment and outcome:

• Treatment-inducing proxy: Y ⊥⊥ Z | (A, X, U) • Outcome-inducing proxy: $W \perp \perp ( A , Z ) \mid ( X , U )$

Assumption 3 (Latent positivity). Every observational unit has a non-zero probability of receiving any given treatment assignment, even when conditioning on the observed covariates and the unmeasured confounder:

$$
P ( A = a \mid X , U ) > 0 ~ f o r ~ a \in \{ 0 , 1 \}
$$

Assumption 4 (Completeness). The proxies must capture enough variation in U to identify the causal effect. For any square-integrable measurablefunctions v ofU and g ofZ,

$$
\mathbb { E } [ v ( U ) \mid Z = z , A = a , X = x ] = 0 f o r a l m o s t a l l ( z , a , x ) \implies v ( U ) = 0 a . s . ,
$$

$$
\begin{array} { r } { \mathbb { E } [ g ( Z ) \mid W = w , A = a , X = x ] = 0 f o r a l m o s t a l l \left( w , a , x \right) \implies g ( Z ) = 0 a . s . } \end{array}
$$

## 2.2 Classical and modern estimation via bridge functions

In addition to Assumptions 1–4, we assume the regularity conditions ensuring the existence of an outcome confounding bridgefunction $h _ { 0 } ( W , A , X )$ . This function satisfies the Fredholm integral equation (Miao et al., 2018; Tchetgen Tchetgen et al., 2020)

$$
\mathbb { E } [ Y \mid Z , A , X ] = \mathbb { E } [ h _ { 0 } ( W , A , X ) \mid Z , A , X ] .\tag{2}
$$

The bridge function enables identification of the conditional average treatment effect through the proximal $g -$ formula

$$
Q ( x ) = \mathbb { E } [ h _ { 0 } ( W , 1 , X ) - h _ { 0 } ( W , 0 , X ) \mid X = x ] .\tag{3}
$$

Equation (3) identifies the effect conditional on X alone by averaging over the proxies.

Early approaches to recovering this bridge function focused on parametric proximal g-computation. Under linear specifications, proximal g-computation reduces to a two-stage estimation procedure closely related to instrumental variable methods. While computationally efficient, proximal two-stage least squares (P2SLS, Tchetgen Tchetgen et al., 2020) relies heavily on strict parametric assumptions. To capture more complex confounding mechanisms, recent two-stage methods have replaced linear regressions with highly flexible nonparametric approximators, such as kernel proxy variable (KPV, Mastouri et al., 2021) and deep feature proxy variable (Xu et al., 2021).

An alternative is to estimate the outcome bridge function by directly enforcing the conditional moment restriction equivalent to Equation (2). Proxy maximum moment restriction (PMMR, Mastouri et al., 2021) leverages a reproducing kernel Hilbert space, while neural maximum moment restriction (NMMR, Kompa et al., 2022) extends this approach by using neural networks.

A separate strand of methods are doubly-robust meta-learners for heterogeneous treatment effect estimation, e.g. P-learner from Sverdrup and Cui (2023) and the Forster-Warmuth learner of (Yang et al., 2023). They estimate an outcome bridge and a treatment bridge functions, then constructs a pseudo-outcome from them. These outcomes are used to fit a standard conditional mean regression mapping the domain of covariates X to treatment effect.

While these modern approaches are highly adaptive, they introduce their own practical hurdles. They demand complex, dataset-specific hyperparameter tuning and extensive model retraining from scratch for every new causal task. The kernel methods also require the computation of kernel matrices, which can impose computational and memory burden in large sample sizes. This motivates the search for an amortized alternative capable of directly estimating $Q ( x )$ from observational context.

## 3 ProximalFM

We propose PROXIMALFM, a PFN for proximal causal inference. Pre-trained on synthetic proximal models, PROXIMALFM predicts a posterior distribution over $Q ( x )$ from an observational context and query covariates x. Unlike existing approaches, it does this without explicitly solving an ill-posed inverse problem as an intermediate step. Instead, it directly outputs a CATE posterior from a single forward pass of a neural network.

## 3.1 A synthetic prior for proximal identification

To train PROXIMALFM, we construct a prior Π over structural causal models (SCMs, Pearl, 2009). An SCM specifies a causal graph, structural functions mapping parent variables and exogenous noise to child variables, and distributions over the exogenous noise. Each SCM defines an episode: its structure stays identical throughout, while each of its units is generated by drawing fresh noise and propagating it through its structural equations:

$$
\begin{array} { l r } { { X = f _ { X } ( \epsilon _ { X } ) , } } & { { U = f _ { U } ( X , \epsilon _ { U } ) , } } \\ { { Z = f _ { Z } ( X , U , \epsilon _ { Z } ) , } } & { { W = f _ { W } ( X , U , \epsilon _ { W } ) , } } \\ { { A = f _ { A } ( X , U , Z , \epsilon _ { A } ) , } } & { { \left( Y ( 0 ) , Y ( 1 ) \right) = f _ { Y } ( X , U , W , \epsilon _ { Y } ) . } } \end{array}\tag{4}
$$

The exogenous-noise vectors $\epsilon _ { ( . ) }$ are drawn independently for every unit from distributions specified by the episode’s SCM. X, U, Z, and W may contain continuous and categorical columns. $A \in \{ 0 , 1 \}$ is drawn from a

![](images/57bbb949188aa0351f23e0c8e183f1f70cb10f7df645bf81fe982c41c27e6cef.jpg)  
Figure 2: Synthetic-prior episode: each covariate $x _ { i }$ is fixed across K downstream SCM draws. One draw forms the observational context; replicate potential outcomes yield Monte Carlo CATE targets and their sampling variance. var<sub>K</sub> is the unbiased sample variance of the K paired treatment effects.

Bernoulli propensity. $Y ( 0 )$ and $Y ( 1 )$ are continuous. A generated unit is the tuple $( X , U , Z , W , A , Y ( 0 ) , Y ( 1 ) )$ both potential outcomes are generated, but the observational dataset keeps only $Y = Y ( A )$

Π satisfies two complementary objectives. First, each SCM must be valid and satisfy the proximal assumptions. Second, the prior must be diverse: we build on the synthetic prior of TabICLv1 (Qu et al., 2025), in which each causal mechanism is a randomly initialized multi-layer perceptron. The prior Π induces variation across SCMs in their dimensions, nonlinear functional forms, feature types and noise distributions, as well as in confounding strength, treatment overlap, and treatment-effect heterogeneity. An overview of the synthetic prior is available in Appendix B.1.

Enforcing proximal assumptions. By construction, every SCM drawn from Π is designed to satisfy the identification conditions of Section 2.1 (see Appendix B.3 for empirical diagnostics):

• Structural constraints (1 & 2). By construction in (4), mutual independence of the exogenous noises $\epsilon _ { ( . ) }$ and the specified mechanism inputs encode the required exclusions. The corresponding conditional independencies follow immediately for every sampled SCM.

• Positivity (3). Guaranteed by clamping treatment assignment probabilities to $[ \delta , 1 - \delta ]$ with $\delta > 0$ before sampling A.

• Completeness (4). Unlike structural graph constraints, completeness is a more subtle, distribution-level condition: it is untestable in observational data and cannot be strictly guaranteed by construction for arbitrary non-linear mechanisms (Tchetgen Tchetgen et al., 2020). The prior therefore relies on a heuristic requiring the dimensions of W and Z to be greater than the one of U. Their cardinalities also dominate the one of U (see Appendix B.2.4).

Contexts & Monte Carlo targets. A naive training framework partitions each dataset of T units generated from the prior, $\{ ( x _ { i } , u _ { i } , z _ { i } , w _ { i } , a _ { i } , y _ { i } ( 0 ) , y _ { i } ( 1 ) ) \} _ { i = 1 } ^ { T }$ <sub>1</sub>, into $\bar { N }$ context units and $T - N$ query units. The context becomes $\mathcal { D } _ { N } = \{ ( x _ { i } , z _ { i } , w _ { i } , a _ { i } , y _ { i } \} _ { i = 1 } ^ { N }$ while the query inputs and targets are $\{ ( x _ { i } , y _ { i } ( 1 ) \stackrel { - } { - } y _ { i } ( 0 ) ) \} _ { i = N + 1 } ^ { T }$ . The PFN then uses the query individual treatment effects (ITE, $y _ { i } ( 1 ) - y _ { i } ( 0 ) )$ as targets to be predicted from $( \mathcal { D } _ { N } , x _ { i } )$ . While simple, this framework is flawed. First, training on ITEs would target their conditional distribution, whereas our estimand is the CATE $Q ( x _ { i } )$ . Second, the ITE distribution depends on the joint law of $( Y ( 1 ) , Y ( 0 ) )$ ). Our prior defines this law, but it is typically unknown and not identified from observational data, so the learned ITE uncertainty would only reflect our choice of prior.

We instead approximate the CATE of the query units. As $Q ( x )$ is an average over the heterogeneity in U, W and noise, it cannot be generated in a single draw. Each query $x _ { i }$ is kept fixed and we run the downstream SCM K times with independent noise. We obtain the individual effects $\tau _ { i , k } = y _ { i , k } ( 1 ) - y _ { i , k } ( 0 )$ whose mean is $\widehat { Q } _ { i , K } .$ with estimated sampling variance $S _ { i , K } ^ { 2 }$ (Figure 2). The estimator $\widehat { Q } _ { i , K }$ is unbiased for $Q ( x _ { i } )$ at every K and $S _ { i , K } ^ { 2 }  0$ as K grows, so K trades compute against target precision. The transformer receives exclusively the observational context $\mathcal { D } _ { N }$ and the query covariates $x _ { i } ;$ the Monte Carlo estimates $\widehat { Q } _ { i , K }$ and $S _ { i , K } ^ { 2 }$ are reserved for training supervision.

![](images/668745620065bfb53a7d3ead6a36db4fd9403dbb6306674ba7147a7943851ff5.jpg)  
Figure 3: PROXIMALFM architecture. $e _ { X } , e _ { Z }$ , and $e _ { W }$ are learned structural-role embeddings; $e _ { y \times a }$ encodes the factual outcome according to treatment.

## 3.2 Model & objective

PROXIMALFM is a PFN trained to map an observational dataset directly to a Bayesian posterior distribution over the CATE estimand. Given an observational context set $\mathcal { D } _ { N } = \big \{ ( x _ { i } , z _ { i } , w _ { i } , a _ { i } , y _ { i } ) \big \} _ { i = 1 } ^ { N }$ of observed units and a query covariate vector $x ^ { \star }$ , the model outputs a predictive distribution over the CATE $\bar { Q ( x ^ { \star } ) }$ in a single forward pass. As it is trained on datasets from the proximal prior Π (Section 3.1), its output approximates the CATE posterior predictive distribution under Π (Muller et al., 2022): dataset-specific Bayesian inference is amortized¨ once at pre-training time.

Encoding the proximal setting. The backbone of PROXIMALFM follows the TabICLv2 architecture (Qu et al., 2026); we defer architectural details to Appendix C.1. Our modifications concern the input processing which must carry the causal role of each column as the roles of (X, Z, W) are not identifiable from their position or values. PROXIMALFM differs from regular tabular models such as TabICLv2 in two respects. First, its target is not observed in the context: the ITE is never observed, only the factual outcome $Y = Y ( A )$ is. These observations only provide evidence about the underlying SCM and its CATE function $Q ( x )$ . Second, PROXIMALFM targets the posterior distribution over the CATE, $p { \big ( } Q ( x ^ { \star } ) \mid { \mathcal { D } } _ { N } { \big ) }$ , rather than a distribution over an individual-level label such as the ITE.

Each query is represented by $x ^ { \star }$ only: including its proxy values $( w ^ { \star } , z ^ { \star } )$ would condition the prediction on the proxies and change the target away from the CATE $Q ( x ^ { \star } )$ . We add structural role embeddings, a learnable token per feature group $( X , Z , W )$ , added to its column embeddings. A treatment-aware target encoder is also implemented. It consists of two linear projections of the outcome selected by the treatment indicator $A ,$ , so that outcomes observed under treatment and control are embedded separately (Figure 3).

Posterior output & noise-aware loss. PROXIMALFM uses a single distributional head to predict the posterior distribution $q _ { \theta } ( Q \mid x ^ { \star } , \mathcal { D } _ { N } )$ directly, rather than separate heads for the two potential outcomes. We use regressionas-classification: $q _ { \theta }$ is predicted as a histogram over 2 000 equal-width bins spanning [−5, 5], in units of the standard deviation of the observed outcomes in context.

The synthetic targets $\widehat { Q } _ { i , K }$ are Monte Carlo averages over K replicates. For sufficiently large $K ,$ their Monte Carlo error is approximately Gaussian by the central limit theorem, with variance estimated by $S _ { i , K } ^ { 2 }$ . Scoring them directly under a distributional loss would inflate the predicted variance by approximately $S _ { i , K } ^ { 2 }$ . Because $S _ { i , K } ^ { 2 }$ is estimated, we correct for it by blurring the model rather than the target: we maximize the likelihood of $\widehat { Q } _ { i , K }$ under the noise-convolved predictive distribution

$$
\tilde { q } _ { \theta } ( \cdot  { \mid } x _ { i } , \mathcal { D } _ { N } ) : = q _ { \theta } ( \cdot  { \mid } x _ { i } , \mathcal { D } _ { N } ) * \mathcal { N } \big ( 0 , S _ { i } ^ { 2 } \big ) .\tag{5}
$$

giving, over the query points $i \in \{ 1 , \ldots , n _ { q } \}$

$$
\mathcal { L } _ { \mathrm { n o i s e - a w a r e } } = - \frac { 1 } { n _ { q } } \sum _ { i = 1 } ^ { n _ { q } } \log \tilde { q } _ { \theta } \bigl ( \widehat { Q } _ { i , K } \mid x _ { i } , \mathcal { D } _ { N } \bigr ) .\tag{6}
$$

The Monte Carlo inflation is removed: the minimizer of (6) approximates the posterior over $Q ,$ , up to finite histogram resolution and the Gaussian approximation to the K Monte Carlo noise. The closed form of (5) for a histogram head is given in Appendix ${ \bf C . 2 }$ . We add a squared-error regularizer on the posterior mean, $\mathcal { L } _ { \mathrm { m e a n } } =$ $\begin{array} { r } { \frac { 1 } { n _ { q } } \sum _ { i = 1 } ^ { n _ { q } } \left( \mathbb { E } _ { q _ { \theta } } [ Q ~ \vert ~ x _ { i } , \mathcal { D } _ { N } ] - \hat { Q } _ { K } ( x _ { i } ) \right) ^ { 2 } } \end{array}$ . The total training objective is $\mathcal { L } = \mathcal { L } _ { \mathrm { n o i s e \_ a w a r e } } + \lambda _ { \mathrm { m e a n } } \mathcal { L } _ { \mathrm { m e a n } \cdot } \mathrm { ~ A t ~ }$ inference time, PROXIMALFM outputs the full CATE posterior distribution $q _ { \theta } ( Q \mid x ^ { \star } , \mathcal { D } _ { N } )$ in a single forward pass, with point estimate $\hat { Q } _ { \theta } ( x ^ { \star } ) = \mathbb { E } _ { q _ { \theta } } [ Q \mid x ^ { \star } , { \mathcal { D } } _ { N } ]$ . Details on the training are available in Appendix C.3.

## 3.3 From proximal identification to Bayesian consistency

Let ψ parametrize an SCM drawn from Π. We adapt the Bayesian consistency argument of Balazadeh Meresht et al. (2026) to the CATE $Q ( x ; \psi ) = \mu _ { 1 } ( x ; \psi ) - \mu _ { 0 } ( x ; \psi )$ , where $\mu _ { a } ( x ; \psi ) = \mathbb { E } _ { \psi } [ Y ( a ) \mid X = x ]$ , the conditional expected potential outcome (CEPO). Under the proximal identifying conditions, including completeness and existence of an outcome bridge, each CEPO is an observational functional $\mu _ { a } ( x ; \psi ) = \mathbb { E } _ { \psi } [ h _ { 0 } ( W , a , X ) \mid X = x ]$ Their difference therefore identifies $Q ( x )$ and directly targeting CATE requires no change to this identification argument.

Suppose these conditions hold almost surely under the SCM prior Π. Under the measurability and integrability assumptions in Appendix $\mathbf { A } ,$ for each admissible fixed query x and Π-almost every true SCM $\psi ^ { \star }$ , the oracle Bayesian posterior mean satisfies

$$
\overline { { { Q } } } _ { N } ( x ) : = \int Q ( x ; \psi ) \Pi ( d \psi \mid \mathcal { D } _ { N } ) \xrightarrow [ N  \infty ] { \mathrm { a . s . } } Q ( x ; \psi ^ { \star } ) .\tag{7}
$$

Theorem A.4 establishes asymptotic consistency of the Bayesian CATE target underlying PROXIMALFM: as the observational context grows, the oracle posterior concentrates on the true CATE and its mean converges to that value. This provides a theoretical foundation for the inference target that PROXIMALFM learns to approximate through Monte Carlo supervision and amortized inference. Our trained model’s posterior-mean predictions inherit consistency when their approximation error relative to the oracle also vanishes.

## 4 Evaluation

## 4.1 Evaluation setting

Methods evaluated. We compare PROXIMALFM against three families of methods. First, proximal estimators recover causal effects through bridge functions or conditional moment restrictions. These are the only baselines encoding the proximal identification assumptions. Second, we evaluate existing causal foundation models that assume no unmeasured confounding. Third, conventional S-, T-, and X-meta-learners provide other non-proximal backdoor baselines (Kunzel et al., 2019). The backdoor methods are evaluated by either fitting¨ X alone or fitting $( X , W , Z )$ followed by marginalizing the proxy-conditional contrast over the proxies. Both variants return a function of $x ,$ but neither is generally identified as $Q ( x )$ under the proximal assumptions. Appendix D.1 lists all methods and refers to their configuration and implementation details.

Datasets. Appendix D.2 details our semi-synthetic datasets generation. The covariates $X$ and unobserved U are features from real-world data. The proxies W and Z are then generated from $( U , X )$ . Similarly, the treatment assignment A and outcomes $Y ( 1 ) , Y ( 0 )$ are generated from $( U , X )$ . Note that we don’t use $Z$ and $W$ as direct inputs when generating the treatment and outcomes.

This semi-synthetic approach grounds the simulation on real-world features, while having control over dataset properties such as confounding strength of $U$ and proxy informativeness of $( W , Z )$ about $U .$ . We consider a factorial design crossing linear versus nonlinear mechanisms, low versus high latent-confounding strength, and low versus high proxy informativeness, yielding eight configurations. For each of the eight settings, we evaluate 12 datasets with three independent replicates each, giving a total of 288 evaluation instances. We set $d _ { X } = d _ { U }$ = $d _ { W } = d _ { Z } = 5$

![](images/49aa72fec5aff1e4eb77a3e5e74974dd508741af11de478cd6712694ba1c8ab5.jpg)  
Figure 4: Median CATE nPEHE across benchmark episodes versus context size. Rows show linear and nonlinear mechanisms; columns vary latent confounding and proxy reliability. The dotted line marks the constant-ATE baseline (nPEHE = 1); lower is better.

## 4.2 Empirical results

PROXIMALFM is competitive across settings. Figure 4 summarizes the performance of selected methods, with detailed results available in Tables D.7–D.10. Across all eight settings, PROXIMALFM is competitive, with its clearest advantage at larger context sizes under high confounding and low proxy reliability. Among the proximal methods, P-learner (TabICL) is also consistently competitive, but typically requires larger context sizes to obtain good performance. KPV and PMMR generally underperform, possibly because of the use of kernel-based bridge functions in this moderate dimension regime $( d _ { X } = d _ { U } = d _ { W } = d _ { Z } = 5 )$ . Flexible backdoor methods such as S-learner (TabICL) and CausalPFN perform surprisingly well when combined with simple proxy marginalization. Their performance deteriorates, however, when strong confounding is combined with low proxy information.

PROXIMALFM plateaus beyond its training range. We next consider the nonlinear regime with high latent confounding and low proxy reliability (Figure D.2, Table D.11). This is a particularly relevant proximal setting: the confounding cannot be ignored, the proxy signal is weak and thus conditioning on proxies is insufficient. As the context grows beyond $n _ { \mathrm { c t x } } = 2 0 4 8 .$ the P-learner (TabICL) continues to improve and becomes the strongest method, whereas PROXIMALFM improves little further and remains close to the constant-ATE baseline. A plausible explanation is a mismatch between evaluation and pre-training context sizes: PROXIMALFM was trained on contexts of at most 2 048 observations. To assess this, we continue training the model for limited time on episodes with contexts of up to 5 000 observations (Appendix C.3.1, ProximalFM-FT). The adapted model shows small improvements at large context sizes, consistent with limited large-context training support contributing to the observed plateau.

PROXIMALFM is fast and requires no task-specific tuning. PROXIMALFM lies on a favourable accuracy– runtime frontier (Figure 5). It achieves among the strongest CATE ranks while requiring little end-to-end runtime compared to several competing proximal methods. Moving inference from CPU to MPS further reduces its runtime without changing its predictions. PROXIMALFM does not require hyperparameter selection, simplifying deployment and making its reported runtime unaffected by the choice of a selection grid.

The posterior is overconfident on semi-synthetic data. Figure D.3 compares nominal and empirical credibleinterval coverage. On datasets sampled from the pre-training prior, coverage is close to nominal across all context sizes, and posterior uncertainty contracts steadily as the context grows. On the other hand, on the semi-synthetic datasets, the posterior is increasingly overconfident as the context size grows. This undercoverage under distribution shift is consistent with the out-of-distribution experiments from Balazadeh Meresht et al. (2026). Notably, the posterior standard deviation continues to decrease beyond the maximum pre-training context size.

![](images/f1b03e10b2f2213d6922023588f52e865dab79e385e6c29a96feff8afae4da5c.jpg)

![](images/47c55997f668676e8dbc57175fb239c802cd5bb0f280ac86d12f6688f3247570.jpg)  
End-to-end runtime Post-selection runtime ProximalFM: MPS CPU  
Figure 5: Accuracy and speed at context size $n = 2 0 4 8 .$ , shown separately for linear and nonlinear mechanisms. Points show median runtime per episode against median paired CATE nPEHE rank (IQR whiskers). Filled and hollow points respectively include and exclude hyperparameter selection. The orange segment compares PROXI-MALFM on CPU and MPS. More details in Appendix D.3.1.

Proxy informativeness matters. The good performance of TabICL and CausalPFN in the semi-synthetic benchmark raises the question of whether flexible backdoor methods can substitute for proximal estimation. We investigate this in a setting with linear outcome and Gaussian proxy mechanisms, where we control the information that (W, Z) carries about U (Appendix D.4). At $n _ { \mathrm { c t x } } = 2 0 4 8$ , PROXIMALFM outperforms proxy-marginalized CausalPFN throughout the sweep and proxy-adjusted linear regression (OLS) in most settings, although OLS catches up when proxies nearly reveal U. Well specified proximal 2SLS (Tchetgen Tchetgen et al., 2020) provides a reference for what matching the estimator to this mechanism can achieve.

Towards a real physical evaluation. We compare ATE estimations on Causal Chambers, a physical system with a known causal graph (Gamella et al., 2025) (Appendix E). Backdoor meta-learners outperform proximal learners, including PROXIMALFM. We interpret this cautiously: PROXIMALFM is designed for CATE, this benchmark has only twelve configurations of one one-dimensional system and moderate sample size (1 000 observations each).

## 5 Conclusion

We introduced PROXIMALFM, an amortized estimator for CATE under the proximal learning assumptions. Existing causal foundation models address settings such as backdoor, instrumental-variable, and frontdoor identification, but are not designed to exploit proxies for proximal identification. Rather than solving an ill-posed bridge equation separately for each new dataset, PROXIMALFM learns from synthetic episodes to estimate CATE from observational data, with regularization implied by the prior. Its CATE posterior also provides uncertainty estimates, though credible intervals can be overconfident beyond the training prior. Together with models for other identification settings, it gives practitioners a fast way to compare estimates under different plausible identifying assumptions.

## Acknowledgements

AK acknowledges funding support from His Majesty’s Government in the development of this research. JZ is funded by the Pioneer Center for Statistical and computational Methods for Advanced Research to Transform Biomedicine (SMARTbiomed), DNRF grant number P4 and a 2026 Jesus College, University of Oxford, SCR Major Research Grant. RS is partially funded by the EPSRC Open Fellowship EP/W024330/1 and, by the Causality in Healthcare AI (CHAI) Hub, EPSRC grant number EP/Y028856/1. AL was funded through a Patient-Centered Outcomes Research Institute (PCORI) Award (ME-2024C2- 39990).

## References

Balazadeh, V., Kamkari, H., Barath, M., Silva, R., & Krishnan, R. G. (2026). Iv-icl: Bounding causal effects with instrumental variables via in-context learning. arXiv preprint arXiv:2605.12924.

Balazadeh Meresht, V., Kamkari, H., Thomas, V., Ma, J., Li, B., Cresswell, J., & Krishnan, R. (2026). Causalpfn: Amortized causal effect estimation via in-context learning. Advances in Neural Information Processing Systems, 38, 154945–154984.

Dikkala, N., Lewis, G., Mackey, L., & Syrgkanis, V. (2020). Minimax Estimation of Conditional Moment Models. Advances in Neural Information Processing Systems, 33, 12248–12262.

Erickson, N., Purucker, L., Tschalzev, A., Holzmuller, D., Desai, P., Salinas, D., & Hutter, F. (2026). Tabarena:¨ A living benchmark for machine learning on tabular data. Advances in Neural Information Processing Systems, 38.

Gamella, J. L., Peters, J., & Buhlmann, P. (2025). Causal chambers as a real-world physical testbed for ai method-¨ ology. Nature Machine Intelligence, 7(1), 107–118. https://doi.org/10.1038/s42256-024-00964-x

Garg, A., Ali, M., Hollmann, N., Purucker, L., Muller, S., & Hutter, F. (2025). Real-tabPFN: Improving tabular¨ foundation models via continued pre-training with real-world data. 1st ICML Workshop on Foundation Modelsfor Structured Data. https://openreview.net/forum?id=BtEiqKsIMw

Ghassami, A., Ying, A., Shpitser, I., & Tchetgen Tchetgen, E. (2022). Minimax kernel machine learning for a class of doubly robust functionals with application to proximal causal inference. International Conference on Artificial Intelligence and Statistics, 7210–7239.

Grinsztajn, L., Floge, K., Key, O., Birkel, F., Jund, P., Roof, B., Manium, M., Bin, S., B¨ uhler, M., Garg, A., et al.¨ (2026). Tabpfn-3: Technical report. arXiv preprint arXiv:2605.13986.

Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? Advances in Neural Information Processing Systems, 35.

Hernan, M., & Robins, J. (2020).´ Causal inference: What if. Chapman & Hall/CRC.

Hollmann, N., Muller, S., Eggensperger, K., & Hutter, F. (2023, September). TabPFN: A Transformer That Solves¨ Small Tabular Classification Problems in a Second [arXiv:2207.01848 [cs]]. https://doi.org/10.48550/ arXiv.2207.01848

Hollmann, N., Muller, S., Purucker, L., Krishnakumar, A., K¨ orfer, M., Hoo, S. B., Schirrmeister, R. T., & Hutter,¨ F. (2025). Accurate predictions on small data with a tabular foundation model. Nature, 637(8045), 319– 326. https://doi.org/10.1038/s41586-024-08328-6

Kompa, B., Bellamy, D., Kolokotrones, T., Beam, A., et al. (2022). Deep learning methods for proximal inference via maximum moment restriction. Advances in Neural Information Processing Systems, 35, 11189– 11201.

Kunzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Meta-learners for Estimating Heterogeneous Treatment¨ Effects using Machine Learning [arXiv:1706.03461 [math.ST]]. Proceedings of the National Academy ofSciences, 116(10), 4156–4165. https://doi.org/10.1073/pnas.1804597116

Luedtke, A., & Chung, I. (2024). Adversarial monte carlo meta-learning of conditional average treatment effects. In Handbook ofstatistical methodsfor precision medicine (pp. 237–248). Chapman; Hall/CRC.

Ma, Y., Frauen, D., Javurek, E., & Feuerriegel, S. (2026). Foundation models for causal inference via prior-data fitted networks. International Conference on Learning Representations, 2026, 79065–79098.

Mastouri, A., Zhu, Y., Gultchin, L., Korba, A., Silva, R., Kusner, M., Gretton, A., & Muandet, K. (2021). Proximal Causal Learning with Kernels: Two-Stage Estimation and Moment Restriction. Proceedings of the 38th International Conference on Machine Learning, 7512–7523. Retrieved April 13, 2026, from https : // proceedings.mlr.press/v139/mastouri21a.html

Miao, W., Geng, Z., & Tchetgen Tchetgen, E. J. (2018). Identifying causal effects with proxy variables of an unmeasured confounder. Biometrika, 105(4), 987–993. https://doi.org/10.1093/biomet/asy038

Miller, J. W. (2018). A detailed treatment of doob’s theorem. https://arxiv.org/abs/1801.03122

Muller, S., Hollmann, N., Arango, S. P., Grabocka, J., & Hutter, F. (2022). Transformers can do bayesian inference.¨ International Conference on Learning Representations. https://openreview.net/forum?id=KSugKcbNf9

Neal, B., Huang, C.-W., & Raghupathi, S. (2021, March). RealCause: Realistic Causal Inference Benchmarking [arXiv:2011.15007 [cs]]. https://doi.org/10.48550/arXiv.2011.15007

Pearl, J. (2009). Causality: Models, Reasoning and Inference, 2nd edition. Cambridge University Press.

Purucker, L., Tschalzev, A., Erickson, N., Blayer, G., Holzmuller, D., Arazi, A., Pfefferle, A., Tajjar, M., Varo-¨ quaux, G., & Hutter, F. (2026). Beyond iid: How general are tabular foundation models, really? https: //arxiv.org/abs/2606.30410

Qu, J., Holzmuller, D., Varoquaux, G., & Le Morvan, M. (2025). TabICL: A tabular foundation model for in-¨ context learning on large data. In A. Singh, M. Fazel, D. Hsu, S. Lacoste-Julien, F. Berkenkamp, T.

Maharaj, K. Wagstaff, & J. Zhu (Eds.), Proceedings of the 42nd international conference on machine learning (pp. 50817–50847, Vol. 267). PMLR. https://proceedings.mlr.press/v267/qu25d.html

Qu, J., Holzmuller, D., Varoquaux, G., & Morvan, M. L. (2026). TabICLv2: A better, faster, scalable, and open tab-¨ ular foundation model. Forty-third International Conference on Machine Learning. https://openreview. net/forum?id=SxsyLjIfWB

Robertson, J., Reuter, A., Guo, S., Hollmann, N., Hutter, F., & Scholkopf, B. (2026). Do-pfn: In-context learning¨ for causal effect estimation. Advances in Neural Information Processing Systems, 38, 174811–174848.

Rubin, D. (1972). Estimating Causal Effects of Treatments in Experimental and Observational Studies [ eprint: https://onlinelibrary.wiley.com/doi/pdf/10.1002/j.2333-8504.1972.tb00631.x]. ETS Research Bulletin Series, 1972(2), i–31. https://doi.org/10.1002/j.2333-8504.1972.tb00631.x

Stith, C., Barath, M., Balazadeh, V., Cresswell, J. C., & Krishnan, R. G. (2026). Causal foundation models with continuous treatments. 2nd ICML Workshop on Foundation Models for Structured Data. https : //openreview.net/forum?id=DzcWAYcR2n

Stith, C., Rahmani, H., & Cresswell, J. C. (2026). Causal foundation models. https://arxiv.org/abs/2609.03003

Sverdrup, E., & Cui, Y. (2023). Proximal causal learning of conditional average treatment effects. International Conference on Machine Learning, 33285–33298.

Tchetgen Tchetgen, E. J., Ying, A., Cui, Y., Shi, X., & Miao, W. (2020, September). An Introduction to Proximal Causal Learning [arXiv:2009.10982 [stat]]. https://doi.org/10.48550/arXiv.2009.10982

Xu, L., Kanagawa, H., & Gretton, A. (2021). Deep proxy causal learning and its application to confounded bandit policy evaluation. Advances in Neural Information Processing Systems, 34, 26264–26275.

Yang, Y., Kuchibhotla, A., & Tchetgen Tchetgen, E. (2023). Forster-Warmuth counterfactual regression: A unified learning approach. https://doi.org/10.48550/arXiv.2307.16798

Zeitler, J. (2025). Physical benchmarks for testing algorithms. Nature Machine Intelligence, 7(2), 166–167. https: //doi.org/10.1038/s42256-025-00999-8

## Appendix

A Consistency result 15   
A.1 Setup and identification . 15   
A.2 Consistency of the oracle CATE posterior 15   
A.3 Proximal identification 16   
A.4 Application to PROXIMALFM 16   
B Details on the proximal prior 18   
B.1 Step-by-step procedure 18   
B.2 Generative model components 21   
B.2.1 Parameters of the prior . 21   
B.2.2 Dimensions & dimension heuristic . 24   
B.2.3 MLP generation, noise & edge sampling . 24   
B.2.4 Post-process: categorisation, outlier and scaling . 26   
B.2.5 Treatment assignment 30   
B.2.6 Potential outcomes & heterogeneity remix . . 31   
B.2.7 Monte Carlo CATE supervision 32   
B.2.8 Visualize the prior 32   
B.3 Proximal assumption audit 36   
B.3.1 Overlap . 36   
B.3.2 Exclusion restrictions . 36   
B.3.3 Latent exchangeability and confounding strength . 38   
B.3.4 Proxy relevance . 40   
C Model architecture, objective, training, and validation 42   
C.1 Architecture . 42   
C.2 The noise-aware histogram loss . 44   
C.3 Training the model 45   
C.3.1 Post-training on larger context sizes 47   
C.4 Validation and posterior diagnostics 49   
C.4.1 Validation datasets 49   
C.4.2 Point-estimate validation . 49   
C.4.3 Distributional validation 49   
C.4.4 Posterior representation & smoothing 51   
C.4.5 Posterior dynamics 52   
D Details on results 54   
D.1 Baseline methods 54   
D.1.1 Backdoor meta-learners, in-context learners and proxy marginalization 55   
D.1.2 P-learner 57   
D.1.3 Neural moment matching regression 57   
D.1.4 Proxy maximum moment restriction 59   
D.1.5 Kernel proxy variable . 60   
D.2 Semi-synthetic benchmarks from real covariates . 61   
D.3 Results on semi-synthetic datasets 63   
D.3.1 Speed . 71   
D.4 Synthetic study of proxy informativeness 71   
Towards a real physical system: Causal Chambers 74

## A Consistency result

We adapt the observational-equivalence argument of Balazadeh Meresht et al. (2026) to the CATE functional. The result characterizes the oracle Bayesian learner under the SCM prior Π used in the main text. Its application to PROXIMALFM is elaborated in Section A.4.

## A.1 Setup and identification

Let ψ range over a standard Borel parameter space Ψ. Each SCM induces a joint law $P ^ { \psi }$ of the observed variables and both potential outcomes, with $A \in \{ 0 , 1 \}$ and $Y = Y ( A )$ . Write $O = ( X , Z , W , A , Y )$ for the observed variables, taking values in a standard Borel space O, and $R ( \psi ) : = P _ { \mathrm { o b s } } ^ { \psi }$ for their law. Under the joint Bayesian model, first draw $\psi \sim \Pi$ and then draw $O _ { 1 } , O _ { 2 } , \ldots \mid \psi$ independently from $R ( \psi )$ . The observational context is $\mathcal { D } _ { N } = ( O _ { 1 } , \ldots , O _ { N } )$

Assumption 5 (Measurability). The map $\psi \mapsto P ^ { \psi }$ is measurable, and the image $\mathcal { R } : = \{ R ( \psi ) : \psi \in \Psi \}$ is a Borel subset of ${ \bf \dot { \mathcal { P } } } ( \mathcal { O } )$ , the standard Borel space ofprobability laws on O. We choose jointly measurable versions of

$$
\mu _ { a } ( x ; \psi ) : = \mathbb { E } _ { \psi } [ Y ( a ) \mid X = x ] , \qquad Q ( x ; \psi ) : = \mu _ { 1 } ( x ; \psi ) - \mu _ { 0 } ( x ; \psi ) .\tag{8}
$$

Throughout what follows, fix an admissible query x at which these conditional means are defined and the stated assumptions apply. Our results apply pointwise in x and we treat the query as fixed (if instead a query is sampled from an SCM-dependent covariate distribution and provides additional evidence, the same argument uses the conditional prior $\Pi ( d \psi \mid X _ { \mathrm { q u e r y } } = x )$ , assuming the context and query are conditionally independent given ψ and the assumptions below hold for that prior.)

Assumption 6 (Integrability). At the query under consideration,

$$
\int _ { \Psi } | Q ( x ; \psi ) | \Pi ( d \psi ) < \infty .\tag{9}
$$

Integrability of both CEPOs is sufficient, since $| Q ( x ; \psi ) | \leq | \mu _ { 1 } ( x ; \psi ) | + | \mu _ { 0 } ( x ; \psi ) |$

Definition A.1 (Observational equivalence and CATE-identifiability). Two SCMs are observationally equivalent when $R ( \psi ) = R ( \psi ^ { \prime } )$ . The prior is CATE-identifiable at x if there is a measurable map $F _ { x } : \mathcal { R }  \mathbb { R }$ such that $Q ( x ; \psi ) = F _ { x } ( R ( \psi ) )$ for Π-almost every ψ.

Definition A.2 (CEPO-identifiability). The prior is CEPO-identifiable at x if, for each $a \in \{ 0 , 1 \}$ , there is a measurable $F _ { a , x }$ with $\mu _ { a } ( x ; \psi ) = F _ { a , x } ( R ( \psi ) )$ for Π-almost every ψ.

CEPO-identifiability implies CATE-identifiability by taking $F _ { x } = F _ { 1 , x } - F _ { 0 , x }$ . The converse is unnecessary and need not hold, since a common unidentified component of the two means can cancel in their difference.

The oracle CATE posterior and its mean are

$$
\begin{array} { r l } { \Pi ^ { Q } ( B \mid x , \mathcal { D } _ { N } ) : = \displaystyle \int _ { \Psi } \mathbf { 1 } \{ Q ( x ; \psi ) \in B \} \Pi ( d \psi \mid \mathcal { D } _ { N } ) , } & { \quad B \in \mathcal { B } ( \mathbb { R } ) , } \\ { \overline { { Q } } _ { N } ( x ) : = \displaystyle \int _ { \Psi } Q ( x ; \psi ) \Pi ( d \psi \mid \mathcal { D } _ { N } ) . } \end{array}\tag{10}
$$

## A.2 Consistency of the oracle CATE posterior

Working directly with R represents observational equivalence classes by their common law. Let ν be the distribution of $R ( \psi )$ under Π, and choose a measurable version of

$$
g _ { x } ( r ) : = \mathbb { E } _ { \Pi } [ Q ( x ; \psi ) \mid R ( \psi ) = r ] .\tag{11}
$$

Lemma A.3 (Limit within observational equivalence classes). Under Assumptions 5–6, for Π-almost every $\psi ^ { \star }$ i.i.d. observations from $R ( \psi ^ { \star } )$ satisfy

$$
\begin{array} { r } { \overline { { Q } } _ { N } ( x ) \xrightarrow [ N  \infty ] { \mathrm { a . s . } } g _ { x } ( R ( \psi ^ { \star } ) ) . } \end{array}\tag{12}
$$

Proof. Conditional on $R ,$ the law of the entire observation sequence is $R ^ { \otimes \infty }$ and does not depend further on $\psi$ Thus

$$
\begin{array} { r } { \overline { { Q } } _ { N } ( x ) = \mathbb { E } [ Q ( x ; \psi ) \mid \mathcal { D } _ { N } ] = \mathbb { E } [ g _ { x } ( R ) \mid \mathcal { D } _ { N } ] . } \end{array}\tag{13}
$$

Moreover, conditional Jensen’s inequality gives $\begin{array} { r } { \int | g _ { x } | d \nu \leq \int | Q ( x ; \psi ) | d \Pi < \infty } \end{array}$ . The model indexed by $r \in \mathcal { R }$ with sampling law $r ,$ is measurable and identifiable by construction. Theorem 2.2 of Miller (2018) therefore gives $\mathbb { E } [ g _ { x } ( R ) \mid { \mathcal { D } } _ { N } ] \to g _ { x } ( R )$ almost surely under the joint Bayesian law. Intersecting the probability-one events on which (13) holds for each N and disintegrating with respect to ψ proves the stated parameterwise conclusion, as in Corollary 2.3 of Miller (2018). □

Theorem A.4 (CATE identification and oracle consistency). Under Assumptions ${ 5 - } 6 ,$ the following are equivalent at the fixed query x:

1. The prior is CATE-identifiable at x.

2. For Π-almost every $\psi ^ { \star }$ , i.i.d. observations from $R ( \psi ^ { \star } )$ satisfy

$$
\begin{array} { r } { \overline { { Q } } _ { N } ( x ) \xrightarrow [ N  \infty ] { \mathrm { a . s . } } Q ( x ; \psi ^ { \star } ) . } \end{array}\tag{14}
$$

Under these conditions the exact CATE posterior also concentrates: for Π-almost every $\psi ^ { \star }$ and every $\epsilon > 0$

$$
\begin{array} { r } { \Pi ^ { Q } ( \{ q : | q - Q ( x ; \psi ^ { \star } ) | > \epsilon \} \mid x , \mathcal { D } _ { N } ) \xrightarrow [ N  \infty ] { \mathrm { a . s . } } 0 . } \end{array}\tag{15}
$$

Proof. I $\mathbf { \dot { Q } } ( x ; \psi ) = F _ { x } ( R ( \psi ) )$ ) almost surely, then $g _ { x } = F _ { x }$ ν-almost everywhere. Intersecting the pullback of this set with the full-prior-measure set in Lemma A.3 proves consistency. Conversely, consistency and Lemma A.3 imply $Q ( x ; \psi ) = g _ { x } ( R ( \psi ) )$ for Π-almost every ψ. The measurable function $g _ { x }$ therefore witnesses CATEidentifiability.

For posterior concentration, apply the same Doob argument to the bounded functions $r \mapsto \mathbf { 1 } \{ F _ { x } ( r ) \leq t \}$ for all rational t. Intersecting the resulting countably many full-measure sets gives convergence of the posterior probabilities of these half-lines to their values at $Q ( x ; \psi ^ { \star } )$ . For any $\epsilon > 0 .$ choose rational thresholds on either side of $Q ( x ; \psi ^ { \star } )$ , each within ϵ of it. The posterior mass outside these thresholds tends to zero, proving (15). □

## A.3 Proximal identification

Suppose the proximal assumptions hold Π-almost surely, including the completeness and bridge-existence conditions needed for the conditional proximal g-formula at the fixed query x. For an appropriate outcome bridge $h _ { 0 }$ and the chosen identifying versions of the conditional means, the formula yields

$$
\begin{array} { r l } & { \mu _ { a } ( x ; \psi ) = \operatorname { \mathbb { E } } _ { \psi } [ h _ { 0 } ( W , a , x ) \mid X = x ] , } \\ & { \ Q ( x ; \psi ) = \operatorname { \mathbb { E } } _ { \psi } [ h _ { 0 } ( W , 1 , x ) - h _ { 0 } ( W , 0 , x ) \mid X = x ] . } \end{array}\tag{16}
$$

The right-hand sides are identified by the observational law as their measurable representations give CEPO- and hence CATE-identifiability. Theorem A.4 consequently applies. Equivalently, when both CEPO posterior means are consistent and integrable, linearity gives

$$
\overline { { { Q } } } _ { N } ( x ) = \mathbb { E } [ \mu _ { 1 } ( x ; \psi ) \mid \mathcal { D } _ { N } ] - \mathbb { E } [ \mu _ { 0 } ( x ; \psi ) \mid \mathcal { D } _ { N } ] \xrightarrow [ N  \infty ] { \mathrm { \Large ~ a . s . } } Q ( x ; \psi ^ { \star } ) .\tag{17}
$$

Note that this identity does not require the trained model to estimate the CEPO.

## A.4 Application to PROXIMALFM

For independent downstream simulations from the SCM’s conditional potential-outcome law given $X = x .$ , independent of the context conditional on ψ, the target $\begin{array} { r } { \widehat { Q } _ { K } ( x ) = K ^ { - 1 } \sum _ { k = 1 } ^ { K } ( Y _ { k } ( 1 ) - Y _ { k } ( 0 ) ) } \end{array}$ satisfies

$$
\mathbb { E } [ \widehat { Q } _ { K } ( x ) \mid \psi , \mathcal { D } _ { N } ] = Q ( x ; \psi ) , \qquad \mathbb { E } [ \widehat { Q } _ { K } ( x ) \mid \mathcal { D } _ { N } ] = \overline { { Q } } _ { N } ( x ) .\tag{18}
$$

With finite second moments, population squared-error regression on these labels therefore targets the oracle posterior mean even at finite K. Recovering the full posterior from noisy labels additionally requires a correctly

specified and identifiable measurement-error model, or a limit in which Monte Carlo error vanishes. Our theoretical result does not establish exact recovery by the finite-K Gaussian noise-aware objective or its combination with a mean penalty but is instead concerned with an oracle learner.

To make this explicit, let $\widehat { Q } _ { \theta , N } ( x )$ be the mean predicted by PROXIMALFM. Then

$$
\begin{array} { r l } & { | \widehat { Q } _ { \theta , N } ( x ) - Q ( x ; \psi ^ { \star } ) | \leq | \widehat { Q } _ { \theta , N } ( x ) - \overline { { Q } } _ { N } ( x ) | } \\ & { \qquad + | \overline { { Q } } _ { N } ( x ) - Q ( x ; \psi ^ { \star } ) | . } \end{array}\tag{19}
$$

Theorem A.4 controls the second term. Consistency of the learned estimator also requires the first term (comprising supervision, representation, and training errors) to vanish as context size grows, which extends beyond the scope of our theoretical contributions and would require substantially more developments in the theory of in-context learning.

## B Details on the proximal prior

## B.1 Step-by-step procedure

We generate each synthetic dataset from an independently sampled structural causal model (SCM) compatible with the proximal DAG. The full routine is sketched in Figure B.1 and specified in Algorithm 1; this section walks through it and fixes notation used across Appendix B.

Throughout, a dataset consists of $T$ units (the sequence length) and the target estimand is the conditional average treatment effect $Q ( x ) = \mathbb { E } [ Y ( 1 ) - Y ( 0 ) \mid X = x ]$ . Because $Q ( x )$ marginalises over all unit-level heterogeneity beyond X, we approximate the conditional expected potential outcomes (CEPOs) $\mu _ { a } ( x ) = \mathbb { E } [ Y ( a ) \mid X = x ]$ by Monte Carlo integration: every unit is replicated K times, sharing its covariate value x while drawing fresh exogenous noise—and therefore fresh latent-confounder and proxy realizations—and the potential outcomes are averaged over the K replicates.

Once per dataset: structure. We first sample the latent and observed dimensionalities, drawing the confounder dimension $d _ { U }$ from a law skewed toward small values and then the proxy dimensions $d _ { W } , d _ { Z }$ conditionally on $d _ { U }$ so as to respect the completeness heuristic $d _ { W } , d _ { Z } \geq d _ { U }$ , with the covariate dimension $d _ { X }$ drawn independently and uniformly; the sampling laws and the resulting induced marginals are described below in Appendix B.2.2. We then instantiate the causal mechanisms $f _ { X } , f _ { U } , f _ { W } , f _ { Z } , f _ { A } , f _ { Y }$ as independently initialised random multilayer perceptrons in the style of TabICL (Qu et al., 2025), together with an auxiliary confounding-duplication network $f _ { \mathrm { a u g } }$ (described below), and sample the per-dataset hyperparameters (network depth, width, activations, noise scales; see Appendix B.2.1). Mechanisms, dimensions and hyperparameters are held fixed across all T units and their K replicates, so the entire dataset is drawn from the same SCM.

Per replicate: the topological pass. The baseline covariates are generated once,

$$
X = { \mathrm { C o r r u p t } } { \big ( } f _ { X } ( \varepsilon _ { X } ) { \big ) } \in \mathbb { R } ^ { T \times d _ { X } } ,\tag{20}
$$

and each row is tiled K times. The remaining variables are sampled in topological order over all $T \cdot K$ rows in a single vectorised pass, with mutually independent exogenous noise $\varepsilon _ { ( \cdot ) }$ drawn afresh for every row:

$$
U = \mathrm { C o r r u p t } \big ( f _ { U } ( \tilde { X } _ { U } , \varepsilon _ { U } ) \big ) , \quad W = \mathrm { C o r r u p t } \big ( f _ { W } ( \tilde { U } , \tilde { X } _ { W } , \varepsilon _ { W } ) \big ) , \quad Z = \mathrm { C o r r u p t } \big ( f _ { Z } ( \tilde { U } , \tilde { X } _ { Z } , \varepsilon _ { Z } ) \big ) ,\tag{21}
$$

where $\tilde { \left( \cdot \right) }$ denotes standardisation to zero mean and unit variance and Corrupt(·) applies the feature postprocessing of Appendix B.2.4 (optional categorisation, outlier clamping, and scaling). The exogenous-noise blocks $\varepsilon _ { \left( \cdot \right) }$ are sized proportionally to each mechanism’s non-noise input width via a per-dataset share parameter (see Table B.3), so the noise level does not shrink as signal width grows. Crucially, $Z$ is excluded from the inputs of $f _ { W }$ and $f _ { Y }$ , and $W$ is excluded from those of $f _ { Z }$ and $f _ { A } ,$ , so the proxy exclusion restrictions of Assumption 2 hold by construction for any realisation of the random mechanisms.

Notation: $\tilde { X } _ { U } , \tilde { X } _ { W } , \tilde { X } _ { Z } , \tilde { X } _ { A } , \tilde { X } _ { Y }$ denote per-mechanism random subsets of $\tilde { X } \mathrm { ' s }$ columns, each drawn independently with its own per-dataset keep probability (see Appendix B.2.3); every mechanism that reads X, including $f _ { U }$ , sees one of these subsets rather than the full block. Likewise, we’ll use the notation ${ \tilde { W } } _ { Y } \subseteq { \tilde { W } }$ and $\tilde { Z } _ { A } \subseteq \bar { \tilde { Z } }$ to denote random column subsets for the direct proxy edges.

Mechanism wiring: confounding control and edge subsampling. Three per-dataset knobs then set which columns actually feed each mechanism. First, to control how strongly the confounder influences each of $A$ and $Y ~ .$ – independently of $d _ { U }$ , which in a random MLP would otherwise pin the confounding to $U ^ { \star } { } _ { \mathbf { s } }$ accidental share of the input width – we target a share $s _ { A } , s _ { Y } \sim$ Beta(1.2, 2.8) of each mechanism’s inputs to be U-derived and convert it to a column count $\begin{array} { r } { k = \operatorname* { m a x } ( 2 , \operatorname { r o u n d } ( \frac { s _ { c } } { 1 - s _ { c } } n ) ) } \end{array}$ with $s _ { c } = \operatorname* { m i n } ( s , 0 . 7 5 )$ and $n$ the width of the non-U inputs. The mechanism is then fed either a subsample of $U ^ { \star } { } _ { \mathbf { s } }$ columns (when $k \leq d \sigma )$ or all of $U$ plus $k - d _ { U }$ columns of a shared nonlinear re-encoding $U ^ { \mathrm { a u g } } = f _ { \mathrm { a u g } } ( \tilde { U } )$ (when $k > d _ { U } ) ;$ we write $U ^ { [ A ] } , U ^ { [ Y ] }$ for these confounding-controlled inputs. We detail this knob in Appendix B.2.3.

Second and third, the direct covariate and proxy edges are made per-column optional, mirroring each other. Every mechanism downstream of X – i.e. $f _ { U } , f _ { W } , f _ { Z } , f _ { A } , f _ { Y } -$ receives an independent random subset $\tilde { X } _ { U } , \tilde { X } _ { W } , \tilde { X } _ { Z } , \tilde { X } _ { A } , \tilde { X } _ { Y } \subseteq$ ${ \tilde { X } } ,$ , each column entering with its own per-dataset keep probability $\sim \mathrm { U n i f } [ 0 , 1 ]$ . Similarly, each column of W (resp. Z) enters $f _ { Y } \ ( \mathrm { r e s p . } \ f _ { A } )$ with its own per-dataset keep probability $\sim \mathrm { U n i f } [ 0 , 1 ]$ , giving subsets ${ \tilde { W } } _ { Y } \subseteq { \tilde { W } }$ $\tilde { Z } _ { A } \subseteq \tilde { Z }$ . These masks alter the random-function prior without adding graph edges: they only remove direct $X$ $W , \mathrm { o r } Z$ inputs, while the $U \to W$ and $U \to Z$ inputs and the completeness heuristics remain unchanged.

Algorithm 1 Generation of one proximal dataset.   
Require: sequence length $T ,$ Monte Carlo count $K ,$ , hyperparameters.   
1: sample $d _ { U }$ from (22); sample $d _ { W } , d _ { Z } , d _ { X }$ from (23)   
2: instantiate random MLP mechanisms $f _ { X } , f _ { U } , f _ { W } , f _ { Z } , f _ { A } , f _ { Y } , f _ { \mathrm { a u g } }$   
3: sample per-dataset hyperparameters: noise shares, confounding shares $s _ { A } , s _ { Y }$ , edge keep-probabilities, over  
lap knobs $s , b , \gamma \sim \mathrm { U n i f } ( 0 , 1 )$   
4: repeat   
5: $X \gets \mathrm { C o r r u p t } ( f _ { X } ( \varepsilon _ { X } ) ) \in \mathbb { R } ^ { T \times d _ { X } }$ , tiled $K$ times and standardised to $\tilde { X }$   
6: subsample covariate inputs $\tilde { X } _ { U } , \tilde { X } _ { W } , \tilde { X } _ { Z } , \tilde { X } _ { A } , \tilde { X } _ { Y } \subseteq \tilde { X }$   
7: draw independent noise $\varepsilon _ { U } , \varepsilon _ { W } , \varepsilon _ { Z } , \varepsilon _ { A } , \varepsilon _ { Y }$ over all $T K$ rows   
8: $U \gets \mathrm { C o r r u p t } ( f _ { U } ( \tilde { X } _ { U } , \varepsilon _ { U } ) )$ , standardised to $\tilde { U }$   
9: $W \gets \mathrm { C o r r u p t } ( f _ { W } ( \tilde { U } , \tilde { X } _ { W } , \varepsilon _ { W } ) )$ , standardised to $\tilde { W }$ $\triangleright Z \not \to f _ { W }$   
10: $Z \gets \mathrm { C o r r u p t } ( f _ { Z } ( \tilde { U } , \tilde { X } _ { Z } , \varepsilon _ { Z } ) ) _ { j }$ , standardised to $\tilde { Z }$ ▷ $W \not \to f _ { Z }$   
11: subsample direct-proxy inputs $\tilde { Z } _ { A } \subseteq \tilde { Z }$ and ${ \tilde { W } } _ { Y } \subseteq { \tilde { W } }$   
12: form $U ^ { [ A ] } , U ^ { [ Y ] }$ from $\tilde { U }$ and $U ^ { \mathrm { a u g } } = f _ { \mathrm { a u g } } ( \tilde { U } )$ via confounding shares $s _ { A } , s _ { Y }$   
13: $\ell  f _ { A } ( U ^ { [ A ] } , \tilde { \chi } _ { A } , \tilde { Z } _ { A } , \varepsilon _ { A } ) ; \tilde { \ell }  ( \ell - \bar { \ell } ) / ( \mathrm { s d } ( \ell ) + 1 0 ^ { - 8 } )$   
14: $\pi  \mathrm { c l i p } ( \sigma ( s \tilde { \ell } + b ) , \epsilon , 1 - \epsilon )$   
15: A ∼ Bernoulli(π)   
16: $( Y _ { 0 } ^ { \mathrm { r a w } } , Y _ { 1 } ^ { \mathrm { r a w } } ) \dot {  } \tilde { f } _ { Y } ( U ^ { [ Y ] } , \tilde { X } _ { Y } , \tilde { W } _ { Y } , \varepsilon _ { Y } )$   
17: draw $\alpha \stackrel { \cdot } { \sim } \mathrm { U n i f } ( 0 , 1 ) ^ { \cdot _ { T K } }$ ; remix to $( Y _ { 0 } , Y _ { 1 } )$ via (30)   
18: $\begin{array} { r } { \hat { \mu } _ { a } ( x _ { i } ) \gets \frac { 1 } { K } \sum _ { k = 1 } ^ { K } y _ { i , k } ( a ) \mathrm { f o r } a \gets \{ 0 , 1 \} } \end{array}$   
19: select $k _ { i } \sim \mathrm { U }$ nif $\{ 1 , \ldots , K \}$ per unit; keep observed $\{ x _ { i } , w _ { i } , z _ { i } , a _ { i } , y _ { i } \}$   
20: remove constant columns from $X , U , W , Z ;$ update active dims $d _ { X } , d _ { U } , d _ { W } , d _ { Z }$   
21: until $d _ { X } , d _ { W } , d _ { Z } > 0 ,$ min $( d _ { W } , d _ { Z } ) \geq d _ { U } > 0$ and categorical dominance of $W , Z$ over $U .$   
22: return observed $\{ x _ { i } , w _ { i } , z _ { i } , a _ { i } , y _ { i } \} _ { i = 1 } ^ { T }$ and targets $\{ \hat { \mu } _ { 0 } ( x _ { i } ) , \hat { \mu } _ { 1 } ( x _ { i } ) \} _ { i = 1 } ^ { T }$

Potential outcomes, heterogeneity remix, and the observed dataset. The outcome mechanism returns both potential outcomes $Y _ { 0 } ^ { \mathrm { r a w } } , Y _ { 1 } ^ { \mathrm { r a w } }$ from a shared forward pass, which are remixed via a per-dataset heterogeneity dial $\gamma \sim \mathrm { U n i f } [ 0 , 1 ]$ into $Y _ { 0 } , Y _ { 1 }$ , ranging continuously from the raw, fully heterogeneous effect $( \gamma = 1 )$ to a single constant effect $( \gamma = 0 ) ;$ we detail and analyse this construction in Appendix ${ \bf { B } } . 2 . 6 .$ Averaging the remixed potential outcomes over the K replicates of unit i gives the CEPO estimates $\begin{array} { r } { \hat { \mu } _ { a } ( x _ { i } ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } y _ { i , k } ( a ) } \end{array}$ , and hence the CATE target $\hat { Q } ( x _ { i } ) = \hat { \mu } _ { 1 } ( x _ { i } ) - \hat { \mu } _ { 0 } ( x _ { i } )$ . To form the observed dataset, one replicate $k _ { i } \sim \mathrm { U n i f } \{ 1 , \ldots , K \}$ is retained per unit, yielding the tuple $\left( { { x } _ { i } } , { { w } _ { i , { { k } _ { i } } } } , { { z } _ { i , { { k } _ { i } } } } , { { a } _ { i , { { k } _ { i } } } } , { { y } _ { i , { { k } _ { i } } } } \right)$ with $y _ { i , k _ { i } } = y _ { i , k _ { i } } ( a _ { i , k _ { i } } )$ . The transformer sees only these observational components; the CEPO targets $\hat { \mu } _ { 0 } , \hat { \mu } _ { 1 }$ are used purely as training targets and are never observed.

Proximal conditions. The construction enforces latent ignorability and the proxy exclusion restrictions of ${ \mathrm { A s } } -$ sumptions 1 and 2 through the mechanism input restrictions and independent exogenous noises. Positivity (Assumption 3) is enforced by the clipped treatment propensity (Appendix B.2.5). Completeness (Assumption 4) cannot be guaranteed for arbitrary nonlinear mechanisms; instead, the prior imposes numeric and categorical proxy-profile heuristics (Appendices B.2.2 and B.2.4). Empirical diagnostics of the resulting overlap, exclusion restrictions, confounding strength, and proxy relevance are collected in Appendix B.3.

![](images/22f511be2f0d7633f3e2e678d74bf8179161ddd92d356d32e69602bebfeee18e.jpg)  
Figure B.1: Detailed diagram of the proximal prior’s per-dataset generation pipeline.

## B.2 Generative model components

## B.2.1 Parameters of the prior

Three sets of parameters govern the prior. Structural parameters (Table B.1) fix the shape of a dataset – the sampling laws of the four block dimensions, the number of rows, and the number of Monte Carlo replicates behind the $\mu _ { 0 } / \mu _ { 1 }$ targets – and are the reference point for the dimension analysis of Appendix B.2.2. Fixed hyperparameters (Table B.2) are shared constants, identical across the whole prior. Sampled hyperparameters (Table B.3) are instead drawn once per dataset from a meta-distribution – typically a log-scaled truncated normal whose own centre is itself uniform on a log scale – so that mechanism depth, width, activation, and noise scale, as well as the causal knobs governing overlap and confounding strength, vary across the prior rather than being pinned to a single value.

The fixed table lists only the constants that pin down behaviour described qualitatively elsewhere in this appendix; the generator additionally exposes on/off switches for the mechanisms of the following subsections – the dimension heuristic (Appendix B.2.2), categorical completeness (Appendix B.2.4), the confounding-share subsampling of $U ^ { \star } { } _ { \mathbf { s } }$ inputs to $f _ { A }$ and $f _ { Y }$ , and the per-column subsampling of the direct and covariate edges (Appendix B.2.3). All are enabled throughout the prior.

Table B.1: Structural parameters of the prior: the block dimensions and the dataset shape. Each dimension is drawn per dataset from the law shown, stated in terms of the ceilings listed below it. The proxy dimensions are drawn conditionally on d so that the completeness heuristic of Appendix B.2.2.
<table><tr><td>hyperparameter</td><td>value</td><td>description</td></tr><tr><td>Block dimensions</td><td></td><td></td></tr><tr><td>dim_X</td><td> $\mathrm { U n i f } \{ 0 , \dots , \mathrm { m a x \_ d i m \_ X } \}$ </td><td>observed baseline covariate dimension</td></tr><tr><td>dim_U</td><td> $\mathrm { t r i a n g u l a r ~ o n } \left\{ 1 , \dots , \mathrm { m a x \_ d i m \_ U } \right\}$ </td><td>latent confounder dimension</td></tr><tr><td>dim_W</td><td> $\operatorname { U n i f } \{ d _ { U } , \dots , \operatorname { m a x \_ d i m \_ W } \}$ </td><td>outcome-inducing proxy dimension</td></tr><tr><td>dim_Z</td><td> $\mathrm { U n i f } \{ d _ { U } , \dots , \mathrm { m a x \_ d i m \_ Z } \}$ </td><td>treatment-inducing proxy dimension</td></tr><tr><td>Dimension bounds</td><td></td><td></td></tr><tr><td>max_dim_X</td><td>10</td><td>covariate-dimension ceiling</td></tr><tr><td>max_dim_U</td><td>10</td><td>confounder-dimension ceiling (dmax of Appendix B.2.2)</td></tr><tr><td>max_dim_W</td><td>10</td><td>outcome-confounding proxy</td></tr><tr><td>max_dim_Z</td><td>10</td><td>dimension ceiling treatment-confounding proxy</td></tr><tr><td>Dataset</td><td></td><td>dimension ceiling</td></tr><tr><td>context_rows</td><td> $\mathrm { U n i f } \{ 6 4 , \dots , 2 0 4 7 \}$ </td><td>context rows per episode</td></tr><tr><td>n-query</td><td>128</td><td>fixed query rows per episode</td></tr><tr><td>mc_cate</td><td>250</td><td>MC replicates K per row behind the</td></tr><tr><td>mc_target_rows</td><td>query rows only</td><td>µo/µ1 targets rows carrying MC CEPO targets</td></tr></table>

Table B.2: Fixed hyperparameters of the prior, held constant across every dataset. Listed are the constants that pin down behaviour the surrounding text describes only qualitatively; switches whose shipped value is that behaviour itself are stated in the text instead.
<table><tr><td>hyperparameter</td><td>value</td><td>description</td></tr><tr><td>Mechanism</td><td></td><td></td></tr><tr><td>noise_std</td><td>0.001</td><td>tiny fixed per-layer dither so continuous columns realise distinct values</td></tr><tr><td>Feature corruption</td><td></td><td></td></tr><tr><td>max_categories</td><td>50</td><td>cap on categories per feature</td></tr><tr><td>proxy-cat_cover-prob</td><td>0.5</td><td>chance a proxy mirrors a U category (vs. a continuous wildcard)</td></tr><tr><td>Positivity</td><td></td><td></td></tr><tr><td>propensity-epsilon</td><td>0.001</td><td>positivity clamp on the propensity</td></tr><tr><td>Completeness</td><td></td><td></td></tr><tr><td>completeness_max_attempts</td><td>50</td><td>resample budget before giving up</td></tr></table>

Table B.3: Per-dataset sampled hyperparameters of the prior. Each is drawn once per dataset from the listed metadistribution; log-scaled truncated normals draw their centre log-uniformly over the stated mean range and add the lower bound as an offset.
<table><tr><td>hyperparameter</td><td>sampling law</td><td>description</td></tr><tr><td colspan="3">Mechanism</td></tr><tr><td>num_layers</td><td>log-scaled trunc. normal, mean ∼ log Unif[1, 6], int ≥ 2</td><td>mechanism MLP depth</td></tr><tr><td>hidden_dim</td><td>log-scaled trunc. normal, mean ~ log Unif[5, 130], int ≥ 4</td><td>mechanism MLP width</td></tr><tr><td>mlp_activations</td><td>categorical over 54 random activation functions</td><td>activation, drawn from a random library</td></tr><tr><td>init_std</td><td>log-scaled trunc. normal, mean ~ log Unif[0.01, 10], real ≥ 0</td><td>weight-initialisation scale</td></tr><tr><td>block_wise_dropout</td><td>categorical {True, False} (random weights)</td><td>block-sparse vs. dense weight init</td></tr><tr><td>mlp_dropout-prob</td><td>scaled Beta on [0, 0.9], shapes ~ Unif[0.1, 5]</td><td>dropout rate (dense init only)</td></tr><tr><td colspan="3">Noise</td></tr><tr><td>sampling</td><td>categorical {normal,mixed,uniform}</td><td>family of the root cause variables</td></tr><tr><td>pre_sample_cause_stats</td><td>(random weights) categorical {True, False} (random</td><td>pre-sample per-cause mean/std</td></tr><tr><td>noise_share_of_input_in_U</td><td>weights) Beta(1.2, 2.8)</td><td>(normal sampling) target noise share of fu inputs (εU width)</td></tr><tr><td>noise_share_of_input_in_W</td><td>Beta(1.2, 2.8)</td><td>target noise share of fw inputs (εw</td></tr><tr><td>noise_share_of_input_in_Z</td><td>Beta(1.2, 2.8)</td><td>width) target noise share of fz inputs (εz</td></tr><tr><td>noise_share_of_input_in_A</td><td>Beta(1.2, 2.8)</td><td>width) target noise share of fA inputs (εA</td></tr><tr><td>noise_share_of_input_in_Y</td><td>Beta(1.2, 2.8)</td><td>width) target noise share of  $f _ { Y }$  inputs (εY width)</td></tr><tr><td colspan="3">Causal</td></tr><tr><td>propensity-shift</td><td>Unif[−2.5, 2.5]</td><td>treatment-logit bias (marginal treated fraction)</td></tr><tr><td>propensity-scale</td><td>log-scaled trunc. normal, mean ∼ log Unif[0.5, 3], real ≥ 0.1</td><td>treatment-logit gain (overlap vs. confounding)</td></tr><tr><td>confounding_share_of_U_in_Y</td><td>Beta(1.2, 2.8)</td><td>target share of  $f _ { Y }$  inputs drawn from</td></tr><tr><td>confounding_share_of_U_in_A</td><td>Beta(1.2, 2.8)</td><td>U target share of  $f _ { A }$  inputs drawn from</td></tr><tr><td>direct_edge_share_of_W_in_Y</td><td>Beta(1,1)</td><td>U per-column keep probability of the direct W → Y edge</td></tr><tr><td>direct_edge_share_of_Z_in_A</td><td>Beta(1,1)</td><td>per-column keep probability of the direct Z → A edge</td></tr><tr><td>covariate_share_of_X_in_U</td><td>Beta(1,1)</td><td>per-column keep probability of the direct X → U edge</td></tr><tr><td>covariate_share_of_X_in_W</td><td>Beta(1,1)</td><td>per-column keep probability of the direct X → W edge</td></tr><tr><td>covariate_share_of_X_in_Z</td><td>Beta(1,1)</td><td>per-column keep probability of the</td></tr><tr><td>covariate_share_of_X_in_A</td><td>Beta(1,1)</td><td>direct X → Z edge per-column keep probability of the</td></tr><tr><td>covariate_share_of_X_in_Y</td><td>Beta(1,1)</td><td>direct X → A edge per-column keep probability of the</td></tr><tr><td colspan="3">Feature corruption</td></tr><tr><td>cat_col-prob</td><td>Beta(1, 3)</td><td>per-column categorisation probability</td></tr></table>

## B.2.2 Dimensions & dimension heuristic

The block dimensions are sampled according to the completeness heuristic introduced in Appendix B.1. The confounder dimension is drawn from a triangular (linearly decreasing) law on $\{ 1 , \dots , d _ { U } ^ { \mathrm { m a x } } \}$ ,

$$
\mathrm { P r } [ d _ { U } = d ] \ \propto \ ( d _ { U } ^ { \operatorname* { m a x } } + 1 - d ) ,\tag{22}
$$

which skews $d _ { U }$ toward small values. Conditional on $d _ { U }$ , the two proxy dimensions are sampled uniformly subject to $d _ { W } , d _ { Z } \geq d _ { U }$ , while the covariate dimension $d _ { X }$ is sampled independently and uniformly:

$$
d _ { W } \sim \operatorname { U n i f } \{ d _ { U } , \dots , d _ { W } ^ { \operatorname* { m a x } } \} , \quad d _ { Z } \sim \operatorname { U n i f } \{ d _ { U } , \dots , d _ { Z } ^ { \operatorname* { m a x } } \} , \quad d _ { X } \sim \operatorname { U n i f } \{ d _ { X } ^ { \operatorname* { m i n } } , \dots , d _ { X } ^ { \operatorname* { m a x } } \} .\tag{23}
$$

This conditional-uniform sampling has a convenient consequence in the common-bound configuration used here, where $d _ { U } , d _ { W }$ , and $d _ { Z }$ all range from 1 to a shared maximum m. When $d _ { U }$ follows the triangular law, the weight $( m + 1 - d )$ placed on $d _ { U } = d$ exactly cancels the normaliser $1 / ( m - d + 1 )$ of the conditional uniform, so the induced marginal of each proxy dimension is the linear ramp

$$
\operatorname* { P r } [ d _ { W } = w ] ~ = ~ \frac { w } { \sum _ { j = 1 } ^ { m } j } ~ = ~ \frac { 2 w } { m ( m + 1 ) } , \qquad w \in \{ 1 , \ldots , m \} ,\tag{24}
$$

and likewise for $d _ { Z }$

The triangular choice for d is what buys this. Had d been uniform, the same conditional sampling would induce a harmonic proxy marginal, $\mathrm { P r } [ d _ { W } = w ] = ( H _ { m } - H _ { m - w } ) / m$ with $H _ { n }$ the n-th harmonic number, which piles proxy mass against the ceiling m: the value $w = m$ is reachable from every $d _ { U }$ , whereas small w requires a small $d _ { U }$ , so probability accumulates at the top (for $m = 1 0 , \mathrm { P r } [ d _ { W } = 1 0 ] = 0 . 2 9 )$ . The triangular law flattens this to the linear ramp of (24) $( \mathrm { P r } [ d _ { W } = 1 0 ] = 0 . 1 8 )$ , and lowers the mean confounder and proxy dimensions from 5.5 and 7.75 to 4.0 and 7.0 respectively. Figure B.2 shows the delivered marginals of both schemes against these analytic laws.

![](images/66c56c0a5add2d3c5199c1e7ee30491e762a36f1d301a8b13072f8469a96a1ee.jpg)  
Figure B.2: Delivered marginal distributions of the block dimensions $d _ { U } , d _ { W } , d _ { Z }$ across the prior, for the shipped (triangular $d _ { U } )$ and the alternative (uniform $d _ { U } )$ sampling, overlaid with their analytic sampling laws (dashed: triangular; dotted: uniform). Bars are empirical frequencies over 1024 datasets. Uniform d pushes proxy mass against the ceiling $d ^ { \operatorname* { m a x } } = 1 0 \colon$ ; the triangular law spreads it into the linear ramp of (24). Delivered bars track the theoretical curves, confirming the generator samples the intended laws.

## B.2.3 MLP generation, noise & edge sampling

TabICL builds a single random MLP that turns a block of root “causes” into one feature table, reading its columns from the pooled activations of the MLP’s intermediate layers. We reuse that construction unchanged as the mechanism for each individual node – the layer stack, the random activation library, the block-sparse or dropout-masked Gaussian initialisation, and the pooled-output selection, all sampled once per dataset (Table B.3); we refer to TabICL for these details (Qu et al., 2025). In PROXIMALFM, the innovative step is that a proximal dataset is not one block but a chain of them, $f _ { X } , f _ { U } , f _ { W } , f _ { Z } , f _ { A }$ , f along a proximal learning DAG. Wiring one block into the next raises two questions a single-MLP generator never faces:

![](images/abc58d9f6569c70c870d5be78d6426b4c706fd75107480fe6309f7af06909d13.jpg)  
Figure B.3: Every dial: each input channel’s share of $f _ { A } \mathrm { ' s } \left( \mathrm { t o p } \right)$ and $f _ { Y } \mathbf { \ ' } _ { \mathbf { S } }$ (bottom) total input variance, against its own sampled knob (s for the $U .$ -channel, the covariate and proxy keep-probabilities for $\tilde { X }$ and $\tilde { Z } _ { A } / \tilde { W } _ { Y }$ , the noise share for ε). Solid line: moving average. Each knob steers its channel’s delivered variance share.

• How should a parent block enter the child’s mechanism?

• How should we inject fresh per-node randomness while keeping it proportional to the signal?

Parents and noise. The root mechanism has no parents and maps a sampled noise input block straight to the covariates, $X ~ = ~ f _ { X } ( \varepsilon _ { X } )$ . Every downstream mechanism instead concatenates its standardised parent blocks with a fresh input block of its own, e.g. $U = f _ { U } ( \tilde { X } _ { U } , \varepsilon _ { U } )$ as in (21). These per-node blocks $\varepsilon _ { ( \cdot ) }$ play the role of exogenous noise: we draw them from the same sampler TabICL uses for its root causes – each column independently normal, uniform, multinomial, or clamped Zipf under per-dataset mixing weights – and size each to a per-dataset share of its mechanism’s signal width (Table B.3), so the noise-to-signal ratio stays controlled rather than shrinking as the parent blocks grow. The MLP’s own within-layer Gaussian perturbation is left small $( \mathtt { n o i s e \_ s t d } = 1 0 ^ { - 3 } )$ , so in practice the input blocks carry essentially all of a node’s exogenous randomness. Postprocessing Corrupt(·) (Appendix B.2.4) then reshapes each block before it is consumed.

Edge sampling. At the MLP level, the parent blocks above are passed through dataset-specific column masks. For each mechanism that depends on X, we draw a keep probability from Unif[0, 1] and retain the columns of X<sup>˜</sup> independently according to that probability. The same procedure is applied to the $W  Y$ and $Z  A$ links, producing the masked proxy inputs ${ \tilde { W } } _ { Y }$ and $\tilde { Z } _ { A }$ . Thus edge sampling can only remove potential parents from the fixed proximal DAG: the completeness links $Z \left. U \right. W$ remain intact, and no new dependencies are introduced. The resulting subgraphs therefore retain the dimensionality heuristic and the structural proximal restrictions. The separate control of how strongly $U$ enters $f _ { A }$ and $f _ { Y }$ is described next.

Confounding control. In a random MLP the confounder’s influence on A and Y would be pinned by $U ^ { \star } { } _ { \mathbf { s } }$ share of each mechanism’s input width, hence by $d _ { U } ;$ to decouple confounding strength from $d _ { U }$ we target a share $s _ { A } , s _ { Y } \sim \mathrm { B e t a } ( 1 . 2 , 2 . 8 )$ of the inputs to $f _ { A } , f _ { Y }$ to be U-derived and translate it into a column count k. When k does not exceed $d _ { U }$ the mechanism sees a k-column subsample of $U ;$ when it does, it sees all of U plus $( k - d _ { U } )$ extra columns drawn from a shared nonlinear re-encoding $U ^ { \mathrm { a u g } } = f _ { \mathrm { a u g } } ( \tilde { U } )$ , itself a random MLP, giving the confounding-controlled inputs $U ^ { [ A ] } , U ^ { [ Y ] }$ . Since $f _ { \mathrm { a u g } }$ is a deterministic function of U, these extra columns carry no information beyond U – they only let the prior push confounding above what $d _ { U }$ alone would permit, without altering the DAG or the completeness edges.

![](images/6e9f7a7e430285962a0d520a09ee1a2b0b0397333ff2ed7238ab76e85dbd0a70.jpg)  
Figure B.4: Output side of the composition: what each mechanism’s output depends on, measured by a crossfitted TabICL regression on the delivered data. Columns 1–3 plot each parent channel’s unique leave-one-out $\Delta R ^ { 2 }$ (drop in $\mathrm { o u t - o f - f o l d } \ R ^ { 2 }$ when that block is removed from the predictors) against its own knob, for $f _ { A } \ ( \mathrm { t o p } .$ target $\log \mathrm { i t } ( \pi ) )$ and $f _ { Y }$ (bottom, target $Y _ { 0 } ) ;$ column 4 plots the unexplained share $1 - R ^ { 2 }$ against the noise knob. Solid line: moving average. Each channel’s dependence rises with its knob, as at the input level. The U-channel reads modestly because leave-one-out credits only signal unique to $U ,$ not the part shared with its child proxy.

Figure B.3 evaluates the effect on the mechanisms’ input variance of the four signal knobs, one per channel<sup>1</sup> governing the $U , X -$ , proxy- and noise-channels respectively. We observe clearly the expected behaviour, as higher values for a knob correspond to a higher proportion of input variance carried by its own channel, on average. Then, Figure B.4 considers the outputs of each mechanism, and evaluates their dependence on the input. On the delivered data we predict each mechanism’s output from its parent blocks with a cross-fitted in-context regressor (TabICL, out-of-fold so every row is predicted exactly once), and read each block’s unique contribution as the leave-one-out drop $R ^ { 2 } ( { \mathrm { a l l } } ) - R ^ { 2 } ( { \mathrm { a l l } } \setminus b )$ . For $f _ { A }$ we predict the raw treatment logit logit(π) from $\{ U , X , Z \}$ ; for $f _ { Y }$ the potential outcome $Y _ { 0 }$ from $\{ U , X , W \}$ ; the noise channel, having no block to drop, is read as the unexplained share $1 - R ^ { 2 }$

Figure B.4 shows each channel’s contribution rising with its own knob, mirroring Figure B.3 at the output level: the mechanisms are predicted with median $R ^ { 2 } \approx 0 . 5$ , and the noise knob visibly inflates the unexplained share. Two caveats with this procedure. First, the blocks are supersets of the subsampled columns each mechanism actually consumed, so a drop is a lower bound on the true dependence. Second, because leave-one-out credits only unique signal, the confounder channel reads low wherever $U$ is partly recoverable from its child proxy. On the Y side, $Y _ { 0 }$ is post heterogeneity-remix, so its $1 - R ^ { 2 }$ also absorbs the exogenous mixing (which vanishes as $\gamma  1 )$ .

## B.2.4 Post-process: categorisation, outlier and scaling

Each block passes through the Corrupt(·) pipeline of TabICL (Qu et al., 2025) before being consumed in (21): a random subset of columns is discretised into categories, outliers are clamped at 4 standard deviations, and the result is standardised. Like TabICL, we apply this to each mechanism’s output rather than within its MLP layers, as Stith, Barath, et al. (2026) instead suggest.

Categorisation interacts with the completeness assumptions of Appendix B, which we preserve through two heuristics on the delivered blocks. Numerically, we keep $d _ { W } , d _ { Z } \geq d _ { U }$ . Categorically, every column of U must be dominated by a distinct proxy column: a numerical column of U by a numerical proxy column, and a categorical column of U by either a numerical proxy column or a categorical one with at least as many categories; as exemplified in Table B.4.

Table B.4: Worked examples of the categorical-completeness heuristic for the fixed confounder profile $U =$ [ num, 5-cat, 3-cat ] (one numerical column and two categorical columns, with 5 and 3 categories). A proxy block P satisfies the heuristic when its columns can be matched one-to-one to those of U so that each is at least as informative: a numerical column dominates anything, and a categorical column dominates a categorical requirement only if it has at least as many categories. A numerical proxy column may thus cover a categorical requirement, but a categorical proxy column can never cover a numerical one.
<table><tr><td>Proxy profile P</td><td>Heuristic?</td><td>Reason</td></tr><tr><td>[num, 5-cat, 3-cat ]</td><td>ok</td><td>exact match, column for column</td></tr><tr><td>num, 8-cat, 4-cat]</td><td>ok</td><td>categorical columns carry more categories than required</td></tr><tr><td>num, num, 3-cat ]</td><td>ok</td><td>a numerical column covers  $U ^ { \star } { } _ { \mathbf { s } }$  5-cat column</td></tr><tr><td>num, num, num]</td><td>ok</td><td>numerical columns dominate every requirement</td></tr><tr><td>[num, num ]</td><td>no</td><td>too few columns  $( d _ { P } = 3 < d _ { U } = 4 )$ </td></tr><tr><td>num, 4-cat, 3-cat ]</td><td>no</td><td>the 4-cat column cannot cover  $U ^ { \star } { } _ { \mathbf { s } }$  5-cat column</td></tr><tr><td>5-cat, 5-cat, 3-cat ]</td><td>no</td><td>only one numerical column for  $U ^ { \star } { } _ { \mathbf { s } }$  two numerical columns</td></tr><tr><td>[num, num, 2-cat ]</td><td>no</td><td>a numerical wildcard covers the 5-cat column, leaving the 2-cat column to cover  $U \mathrm { { ^ { \bullet } s } }$  3-cat column, but  $2 < 3$ </td></tr></table>

Numeric completeness. The dimension heuristic is imposed at sampling time, but the constant-column filtering that follows (Algorithm 1, line 20) can shrink a block afterwards, so we re-check it on the delivered dimensions. Figure B.5 illustrates this heuristic enforcement.

Categorical completeness. We call P categorically complete w.r.t. U when every column of U matches a distinct, at-least-as-informative column of P (“dominates”). Whenever U is categorised, the prior builds each proxy to satisfy this by construction: it reserves one numerical column per numerical column of U, covers each categorical column of U with either a numerical column or – with probability proxy cat cover prob = 50% – a categorical column drawing strictly more categories, and fills any surplus proxy columns freely. Figure B.6 contrasts the block compositions this yields with and without this construction: default (enforcement on) and no enforcement (U categorised, proxies left unconstrained) start from the same categorised U, but under no enforcement the proxies end up no more numerical than U itself, so nothing forces domination.

Enforcement scales with U’s demand. Figure B.7 reads the covering rule off the delivered data from two angles. Conditioning on a single U column’s cardinality (panels a–b), the proxy block’s numerical fraction is roughly flat in that cardinality and sits above the unconstrained arm (≈ 0.70 vs. ≈ 0.60), while the mean category count of the proxy’s categorical columns grows with the U column’s cardinality under default (≈ 12 at 2 categories to ≈ 19 in the 30+ bin), exactly as the “strictly more categories” covering rule dictates, and stays flat at ≈ 10 under no enforcement. Enforcement therefore acts on the depth of the proxy’s categorical columns, not on how many of them there are.

Panel (c) makes the same point negatively, at the dataset level. The categorical share of the proxies rises with U’s categorical demand |cats(U)| in both arms, because the per-dataset categorisation propensity is shared across blocks: a dataset that discretises many columns of U also discretises many columns of its proxies, enforcement or not. Enforcement in fact leaves the proxies less categorical than the free no enforcement arm at every demand level $( 0 . 0 8  0 . 4 3 \mathrm { v s . } 0 . 1 4  0 . 5 8 )$ , consistent with the reserved numerical columns of panel (a). So the categorical share is not the signature of enforcement; the per-column depth of panel (b), and the survival rate below, are.

Joint distribution of $d _ { U }$ and the other blocks, per variant

![](images/46f9a42823b34beb63a56b477311342f130e14073763354499e37c4ec1f4575f.jpg)

triangular: $d _ { U }$ vs d<sub>W</sub>  
![](images/c869f5d067b7c8c6df9b45101d3ee0626ba2d822c47f90ea5c31c620bc04b9e4.jpg)

triangular: $d _ { U }$ vs $d _ { Z }$  
![](images/6d9b50927b39bfeb369017e0fc516e7447288e890b982e35366243dd914b78ed.jpg)

uniform: d<sub>U</sub> vs d<sub>X</sub>  
![](images/6fd5fec61406556092af0fa6a8cf7e2e43f8b658f6a4574fbc2b3264bc991066.jpg)

![](images/dfad6ff1dfd13f9bb69169f10230fec1614bbe5208976d5e5d04a340272fbb05.jpg)

uniform: d<sub>U</sub> vs d<sub>Z</sub>  
![](images/1929a219522419ae6cefb2826c38917048d8cbc857762937dcf9f59e8fc88183.jpg)  
Figure B.5: Joint distribution of the latent dimension $d _ { U }$ against each feature block $( d _ { X } , d _ { W } , d _ { Z } ) .$ , one row per prior variant (triangular, uniform). Colour encodes the number of delivered datasets falling in each $( d _ { U } , \cdot )$ cell. The diagonal line on the $d _ { W } / d _ { Z }$ panels marks the completeness heuristic $d _ { W } , d _ { Z } \geq d _ { U }$ ; both variants place their mass on or below it, i.e. constant-column filtering does not measurably break the heuristic imposed at sampling time. $d _ { X }$ carries no such constraint and is shown for reference.

U col.: categories  
![](images/7e8e908f833a8ee8f2476373cf1cc14071403fe8b5cbb2dca982a2216906df02.jpg)  
Figure B.6: Column-type composition of each block $( U , W , Z )$ , one 100%-stacked bar per variant (default: categorical completeness enforced; no enforcement: proxies left unconstrained). Each bar partitions a block’s columns into continuous (grey, “cont.”) and categorical, the latter binned by category count (2–5 singletons, then coarser $6 { - } 1 0 , 1 1 { - } 2 0 , 2 1 { - } d _ { \mathrm { c a t } } ^ { \mathrm { m a x } }$ tail bins). Continuous columns dominate every block; contrasting W/Z across the two variants shows how much enforcement pushes proxy columns categorical relative to the unconstrained no enforcement arm.

(a)  
![](images/c8f90a1cd6bb6f1a831e70ed4deffa1f63c0e48637c7f1e0797f40c4b21dfa05.jpg)

![](images/c90590931a20d4e579d8ad633c6af9c2937d053e576b0b8d063a35051c5e4c3b.jpg)  
U col.: categories

![](images/a32132f16f75d445b0b2a2f83a8eb50a6e562be3d0cac4ce2e76083d46f43e25.jpg)  
Figure B.7: How enforcement makes proxy structure track $U ^ { \star } { } _ { \mathbf { s } }$ categorical demand, pooling W and $Z$ across the default and no enforcement variants; shaded bands are ± one SEM. (a, b) condition on a single U column’s own cardinality: the x-axis bins a $U$ column by its category count (continuous $\cdot \circ _ { \mathrm { { n u m } } } , \cdot $ , then $2 , \ldots , 9 ,$ , and coarser tail bins up $\mathrm { t o } \ ^ { \ast } 3 0 \mathrm { + } ^ { \prime \prime } ) ,$ , and each point averages a statistic of the co-occurring proxy block over every dataset containing a U column of that kind – (a) the fraction of proxy columns that are continuous, essentially flat in $U ^ { \star } { } _ { \mathbf { s } }$ cardinality but higher under default, and (b) the mean category count of the proxy’s categorical columns, which rises with $U ^ { \star } { } _ { \mathbf { s } }$ cardinality under default but stays low under no enforcement. (c) moves to the dataset level: the mean fraction of categorical proxy columns against $U ^ { \star } { } _ { \mathbf { s } }$ total categorical demand $\left| \mathrm { c a t s } ( U ) \right|$ , with bins of fewer than 5 datasets dropped. Both arms rise (the per-dataset categorisation propensity is shared across blocks) and default sits below no enforcement: enforcement deepens the proxy’s categorical columns rather than multiplying them. Because proxy statistics in (a, b) are block-level rather than tied to a matched column, those trends reflect association with “the dataset contains a $U$ column of this kind,” not a one-to-one mapping.

## B.2.5 Treatment assignment

The treatment mechanism $f _ { A }$ maps the confounding-controlled confounder input $U ^ { [ A ] }$ , the covariate subset ${ \tilde { X } } _ { A }$ the direct proxy subset $\tilde { Z } _ { A }$ and fresh noise $\varepsilon _ { A }$ to a single scalar logit per row,

$$
\ell = f _ { A } \big ( U ^ { [ A ] } , \tilde { X } _ { A } , \tilde { Z } _ { A } , \varepsilon _ { A } \big ) \in \mathbb { R } ^ { T K } ,\tag{25}
$$

with $W$ excluded from the inputs so that the proxy exclusion restriction of Assumption 2 holds by construction (Equation 21). The treatment A is binary; the steps below turn ℓ into a propensity score $\pi = \operatorname* { P r } [ A = 1 \mid \cdot ]$ and draw A.

Logit normalisation. A random MLP fixes neither the location nor the spread of its output, both of which drift with depth, width and input scale; left raw, the induced treatment prevalence and the amount of overlap would be an uncontrolled by-product of the sampled architecture. We therefore standardise the logits across the $T \cdot K$ rows of the dataset to zero mean and unit variance,

$$
\tilde { \ell } = \frac { \ell - \bar { \ell } } { \mathrm { s d } ( \ell ) + 1 0 ^ { - 8 } } ,\tag{26}
$$

so that the assignment distribution is decoupled from the accidental scale of $f _ { A }$ and every dataset starts from a common footing before the overlap knobs act.

Shift and scale. Two per-dataset propensity knobs then set the assignment regime through an affine map of the standardised logits,

$$
\ell ^ { \prime } = s \tilde { \ell } + b , \quad \quad b \sim \mathrm { U n i f } ( - 2 . 5 , 2 . 5 ) , \quad s \sim \mathrm { l o g } { \mathrm { - s c a l e d } } \mathrm { t r u n c a t e d n o r m a l } ,\tag{27}
$$

both drawn once per dataset (see Table B.3). The shift b moves the marginal treatment prevalence away from the balanced split – negative b makes treatment rare, positive b makes it common – while the scale s controls how sharply assignment depends on the covariates and confounder: small s flattens the logits toward a coin flip (strong overlap, weak confounding channelled through A), whereas large s sharpens them toward near-deterministic assignment (weaker overlap).

Propensity and sampling. A sigmoid turns the shifted logits into a propensity score, from which the treatment is drawn Bernoulli,

$$
\pi = \sigma ( \ell ^ { \prime } ) , \qquad A \sim \mathrm { B e r n o u l l i } ( \pi ) .\tag{28}
$$

Overlap clamp. Because a large s can push π arbitrarily close to 0 or 1, we clamp the propensity into $[ \epsilon , 1 - \epsilon ]$ with $\epsilon = 1 0 ^ { - 3 }$ before sampling,

$$
\pi  \mathrm { c l i p } ( \pi , \epsilon , 1 - \epsilon ) ,\tag{29}
$$

guaranteeing a floor on the probability of each treatment arm and hence enforcing positivity/overlap $( 0 < \operatorname* { P r } [ A =$ $1 \ | \ \cdot ] < 1 )$ for every row by construction (Assumption 3). We audit the delivered overlap across the prior in Appendix B.3.1.

Figure B.8 shows this five-step pipeline on a single representative dataset, following one unit (in red) from its raw logit through to its treatment draw.

![](images/951851009ac15f28df9354925a37bac8149ded50c976060c4615ff6512e46d34.jpg)  
= <sub>fA</sub>( )

![](images/55cfa1ae9994380a48c8f1e5a89b7426163aa648d02051a5a5b05c4e8493c7ce.jpg)

![](images/79128816f3e59daa3ff913b7656c53c2c43794227a79d48bf88644808a355be5.jpg)

![](images/4732d5c7ad8aec572e257b120c99bd3ca44aa4679fe97bb7bb3f5055da5ec9aa.jpg)  
= ( 0)

![](images/1d2bb634aa08b79b87c78f4a19d97f72f208fd7264dbef78a829d21be9acfa92.jpg)  
<sub>A</sub> Bern( )  
Figure B.8: The treatment-assignment pipeline of (25)–(29) on one representative dataset, tracking a single unit (red line) across all five stages. (a) The raw logit $\ell = f _ { A } ( \cdot )$ , whose scale and offset are an uncontrolled by-product of the sampled architecture. (b) The standardised $\ell \mathrm { o f } \left( 2 6 \right)$ , pinned to zero mean and unit variance. (c) The affine ${ \ell ^ { \prime } = s \tilde { \ell } + b }$ of (27), shifted and sharpened by the two sampled overlap knobs. (d) The propensity $\pi = \sigma ( \ell ^ { \prime } )$ of (28), confined to $[ \epsilon , 1 - \epsilon ]$ by the clamp of (29) (dotted bounds, drawn at an enlarged ϵ for visibility; production $\epsilon = 1 0 ^ { - 3 } )$ . (e) The realised treatment $A \sim \mathrm { B e r n o u l l i } ( \pi )$ across the batch (fraction in each arm), with the tracked unit’s realised arm marked in red.

## B.2.6 Potential outcomes & heterogeneity remix

The outcome mechanism $f _ { Y }$ emits both potential outcomes from a single forward pass, returning the pair $\left( Y _ { 0 } ^ { \mathrm { r a w } } , Y _ { 1 } ^ { \mathrm { r a w } } \right)$ from a common input and exogenous draw $\varepsilon _ { Y } ;$ the two arms differ only in their read-out, so the raw effect $\tau ^ { \mathrm { r a w } } = Y _ { 1 } ^ { \mathrm { r a w } } - Y _ { 0 } ^ { \mathrm { r a w } }$ stays coupled through the shared latent state.

Left as is, a dataset’s effect heterogeneity would be an uncontrolled by-product of the sampled MLP. To let the prior span the full range from strongly modulated effects to a single constant effect, we remix the raw outcomes before forming the targets. With a per-dataset dial $\gamma \sim \mathrm { U n i f } ( 0 , 1 )$ , a per-row weight $\alpha \sim \mathrm { U n i f } ( 0 , 1 )$ , and the raw average effect $\begin{array} { r } { \bar { \tau } = \frac { 1 ^ { - } } { T \cdot K } \sum ( Y _ { 1 } ^ { \mathrm { r a w } } - Y _ { 0 } ^ { \mathrm { r a w } } ) } \end{array}$

$$
\begin{array} { r l } & { Y _ { 1 } = \big ( \alpha + ( 1 - \alpha ) \gamma \big ) Y _ { 1 } ^ { \mathrm { r a w } } + ( 1 - \gamma ) ( 1 - \alpha ) \big ( Y _ { 0 } ^ { \mathrm { r a w } } + \bar { \tau } \big ) , } \\ & { Y _ { 0 } = \big ( ( 1 - \alpha ) + \alpha \gamma \big ) Y _ { 0 } ^ { \mathrm { r a w } } + ( 1 - \gamma ) \alpha \big ( Y _ { 1 } ^ { \mathrm { r a w } } - \bar { \tau } \big ) . } \end{array}\tag{30}
$$

This construction follows the heterogeneity knob of RealCause (Neal et al., 2021), later re-used in the causalfoundation-model prior of CausalPFN (Balazadeh Meresht et al., 2026).

Substituting (30) collapses the per-row weights to $\begin{array} { r } { Y _ { 1 } - Y _ { 0 } = \gamma \tau ^ { \mathrm { r a w } } + ( 1 - \gamma ) \bar { \tau } } \end{array}$ , so γ linearly interpolates each unit’s effect between its raw value $( \gamma = 1$ , fully heterogeneous) and the dataset mean $\bar { \tau } \left( \gamma = 0 \right.$ , fully homogeneous). Anchoring at $\bar { \tau }$ rather than zero preserves the ATE at every $\gamma ;$ the per-row α simply shares the required shift between $Y _ { 0 }$ and $Y _ { 1 }$ for each unit; averaged across rows, this allocation and the compensating ±τ¯ terms leave the marginal means of $Y _ { 0 } , Y _ { 1 }$ (hence the outcome levels and the ATE) unchanged in expectation, so the dial reshapes only the spread of the effect.

Because the collapse also gives sd $( Y _ { 1 } - Y _ { 0 } ) = \gamma { \mathrm { s d } } ( \tau ^ { \mathrm { r a w } } )$ , the dial should act on the delivered data as a pure, proportional spread control. Figure B.9 checks this across the prior: $\gamma$ is drawn independently per dataset and recorded, so the shipped prior is itself a γ sweep, and the (outcome-scaled) effect dispersion tracks it closely, while the outcome coupling corr $( Y _ { 0 } , Y _ { 1 } )$ approaches its exact $\gamma  0$ value of 1.

![](images/1caef188725c2794ff9f342bd031d2d6b9a0b54a3a6d80a562dec19048d70734.jpg)

![](images/b7ccf4c49d94b085301ce947a30a8c56b01492b02ed13496489a8b016953f2e5.jpg)

![](images/b3d0a3ce5c4a53159dc568f454b0671643b34d810e327e68ac96cd780c05dc48.jpg)  
Figure B.9: The heterogeneity dial on the delivered prior, over 1024 datasets. All statistics are ratios, since each SCM’s outcome carries its own arbitrary scale. (a) Two datasets’ effect densities on a shared outcome-scaled axis, each centred on its own $\bar { \tau }$ (dashed line), one gathered $( \gamma = 0 . 0 9 )$ and one spread $( \gamma = 0 . 9 1 )$ ; the pair is chosen with matched raw dispersion $c = \mathrm { s d } ( \tau ^ { \mathrm { r a w } } ) / \mathrm { s d } ( Y )$ (1.127 vs. 1.125), so the contrast is attributable $\mathbf { t o } \gamma$ rather than to SCM variation. (b) The outcome coupling corr $( Y _ { 0 } , Y _ { 1 } )$ against $\gamma$ (solid line: moving average). $\mathrm { A t } \gamma = 0$ the effect is the constant τ¯, so $Y _ { 1 } = Y _ { 0 } + \bar { \tau }$ and the coupling is exactly 1 (dashed reference). (c) Distribution of the delivered dispersion s $\mathrm { d } ( \tau ) / \mathrm { s d } ( Y )$ per $\gamma$ bin (Spearman 0.78 overall), against the proportionality $\gamma$ · median(c) implied by the collapse; within-bin spread is dominated by variation in $c .$ Because the reference line is anchored at zero and uses the overall median c for its slope, the panel verifies strict proportionality to γ—linear scaling starting from the origin—rather than predicting the absolute magnitude of dispersion.

## B.2.7 Monte Carlo CATE supervision

For a query covariate row $x _ { i } ,$ mc cate specifies the number K of independent downstream SCM draws used to approximate its CATE. The covariates X remain fixed, while the latent confounder $U ,$ proxies, and exogenous outcome noise are redrawn. Within draw $k ,$ , both potential outcomes are evaluated on the same latent state, yielding the paired effect $\tau _ { i , k } = Y _ { i , k } ( 1 ) - Y _ { i , k } ( 0 )$ . We supervise on $\begin{array} { r } { \widehat { Q } _ { K } ( x _ { i } ) = K ^ { - 1 } \sum _ { k } \tau _ { i , k } . } \end{array}$ an unbiased estimate of $Q ( x _ { i } )$ , with estimated sampling variance $S _ { i } ^ { 2 } = \widehat { \mathrm { V a r } _ { k } } ( \tau _ { i , k } ) / K$ . Thus $K$ controls target precision, not the SCM or estimand: its standard error decreases as $K ^ { - 1 / 2 }$ , at generation cost proportional to $K$

## B.2.8 Visualize the prior

We close Appendix B.2 with three galleries of the delivered data, each a fixed-seed cross-section of the prior’s diversity: Figure B.10 shows individual feature marginals, one column per block and one sampled dataset per row (the propensity π for $A ,$ , both potential outcomes $Y _ { 0 } , Y _ { 1 }$ for $Y ) ;$ Figure B.11 shows within-block dependence, scattering two features of the same block (π against the draw $A ,$ and $Y _ { 0 }$ against $Y _ { 1 }$ for the scalar blocks); and Figure B.12 shows cross-block dependence for a chosen few block pairs – the parent→child edges $U { - } W , U { - } Z$ the proxy pair $W { - } Z$ (associated only through the shared latent $U )$ , and the confounder against the propensity $U { - } \pi$

![](images/8b1a85fdd6c227e3a2dad4694e9fae6cefc291edb1bd030bfb6d17061a436602.jpg)  
Figure B.10: Gallery of feature marginals: one column per block $( X , U , W , Z ,$ the propensity $\pi = P ( A { = } 1 \mid \cdot )$ for $A ,$ both potential outcomes for $Y ) _ { : }$ , one row per randomly sampled dataset. Each panel is one feature column’s histogram, solid for continuous columns and dashed for categorical ones; the $Y$ column overlays $Y _ { 0 }$ (blue) and $Y _ { 1 }$ (red). Feature blocks share the standardised axis Corrupt leaves; $\pi \in [ 0 , 1 ]$ and $Y$ keep their natural scale.

![](images/1cee16ab7a545d192a354c1db9892dbd0dbc06ac15575b6c4414f1de9e8b9c93.jpg)

![](images/2e7704bf7605cf18d626ad4574c2a89970956be5e1fa7e7ddeafc6bdd8bef0c5.jpg)  
U

X  
![](images/ec9278b8adf9e10816237b2b7a915930260c25bfd95d8f8da5f2ae98715bf91c.jpg)

![](images/25a1293aad2551e114d14fd2e2439fb5da309dcc67cfec2c14dcb711d47b1c8e.jpg)

![](images/964387c7e229aecb8bc094b8ed0bef14e324d6bc772f2918da4916be197c374a.jpg)

![](images/a2b53f2444280e622d9317846a3235ed0a5ef2e121b62465d66fc980d1205609.jpg)

![](images/8f8bbe3c844d0fc329d2724c436708ee7004684df30f8ada672d5ca0c1de43b8.jpg)

![](images/221ba88a69cae893ac662098975b30426988eb172775b7beae8d78a9fce0b6be.jpg)

![](images/31b94d7631ce6c4a626e96e25d82225db5ba858848c119ec30ee2f87b9826847.jpg)

![](images/843257f6dafb6380e6f315ff918db1399523d574dac50dafee56c600a5c3cd65.jpg)

![](images/d4a0bbc8cf0181052327aa5618fc4f5771dd701ce431d7b838301a58e77b15a0.jpg)

![](images/48226a2f04a07fd09f0d123352b95f25640087a61b6c1a27211c778e0abd93a0.jpg)

![](images/1a697753f2dc43b3a259d47682270c28f90b1073e691596ebc7aca551a178c77.jpg)

![](images/4bf03d52fc2f19040364d7851f2d42c8de3fe456f8a9fc77e8459def0b7bfd66.jpg)

![](images/d8c558fd482c0160e103d7acbf7ddc335bc1bf11e3ff3d48b4fd4cd31cff6822.jpg)

![](images/4a1b0eb8b2cd8a7c35503b6d080279b8f74050828feee30955b7f37d2214afa2.jpg)

![](images/a7f8e4646a4f411e3541ab78d1007aaa60155ed7ed0df8997841bd6562287606.jpg)

![](images/6e31e99351f7af68242696a8fb40efee68b0464b5bff47080bdac7a77a4b530c.jpg)

![](images/98ddea92416c97fd985c26372d77f5a16095e88c9cc52dce0ec2ed7d4ce238ab.jpg)

![](images/6b2bc4d1ea1667f9c54dfafe10766081eb8c40722d03529b969f98421e88bc6f.jpg)

![](images/a46c6b21b2abf17df18a816b08ad2157fede9db12a60239cf1a6e5f00a022cb2.jpg)

![](images/2c2a3eed1941ee88a541565b5b22275751c5b2ca10476d2d34b712f92c2ae9dc.jpg)

![](images/626e48278d9133e22f12882895026bafc6236521762238f78beb846c1658fd91.jpg)

![](images/c1f57ad0fca00b8454ee70673abc276ac64eb3cdfda2e690fda4a1c81a75ad6d.jpg)

![](images/3a22995658951d5fe7a92936b61194be0aa8a37eda92d246d715037af6cbf145.jpg)

![](images/2ef7d2bcac022c023bbe495ce434992007d9117e893e0b365ac5ac23e63d3794.jpg)

![](images/297e7de4f195dde195416ab390d64d5edc232b1030557b450ee000be40335a1e.jpg)

![](images/2fd0125ceab4b3a0c728f93e5eabb8dae95ffa32424e7cafcbbd3f601bf3df9d.jpg)

![](images/a105e7b172593e79621331b426aa6997338b9a3ce6c8af03a2dfc0e3fdd64ce9.jpg)

![](images/d3b1f805d373f7d79a647158cc690668ee927ab1f2352c44f3737a004efa3a40.jpg)  
Figure B.11: Within-block dependence: one column per block, one randomly sampled dataset per row, each panel a scatter of two features from that block. For X, U, W, Z the two features are random columns of the block (sampled from datasets of width ≥ 2); for A the panel plots the propensity π against the treatment draw A; for Y the two potential outcomes $Y _ { 0 } \ \mathrm { v s } \ Y _ { 1 }$ . Rows are subsampled for legibility; feature axes inherit the standardised scale.

![](images/7e2c1efb6b10136a9ad939f2dc03bcce71e76e7d391d7cd08bf5700514c34c85.jpg)  
Figure B.12: Cross-block dependence: one column per chosen block pair, one sampled dataset per row, each panel a scatter of one feature from each block. $U { - } W$ and $U { - } Z$ are the parent→child edges; $W { - } Z$ is the proxy pair, associated only through the shared latent U (no direct edge); $U { - } \pi$ is the confounder against the propensity $\pi = P ( A { = } 1 \mid \cdot )$ . Rows are subsampled for legibility; feature axes inherit the standardised scale.

## B.3 Proximal assumption audit

## B.3.1 Overlap

Positivity (Assumption 3) cannot fail in this prior: the clamp of (29) confines π to [ε, 1 − ε] with $\varepsilon = 1 0 ^ { - 3 }$ chosen deliberately small (see Table B.2), so π is never exactly 0 or 1 and every row carries nonzero mass in both treatment arms by construction. Strictly, the stored π is realised conditional on the unit’s full set of mechanism inputs $( X , U , Z , \epsilon _ { A } )$ rather than the marginal $\operatorname* { P r } [ A = 1 \mid X , U ]$ that Assumption 3 is stated over, since A’s logit also depends on the proxy Z and the mechanism’s own exogenous noise; but the marginal is the expectation of this finer-grained π over $Z , \epsilon _ { A }$ , so keeping every realised π away from 0, 1 is a sufficient, if anything conservative, condition for the marginal to be as well, and auditing the stored values directly is valid. The question worth auditing is therefore not whether positivity holds but how close the sampled overlap knobs push π toward that clamp in practice, since a propensity pinned near 0 or 1 leaves little residual randomness in treatment assignment for any estimator to exploit even though positivity technically holds. The audit needs no additional estimation: Figure B.13 is a direct summary of these stored propensities.

Because ε is small, the clamp only bites at the very extremes and the vast majority of rows sit well within it: across the prior, 84.0% of all rows fall inside the conventional band [0.05, 0.95], and only 42.6% of datasets contain even a single row on the clamp, with the median dataset containing none. Overlap is nonetheless deliberately stressed away from that safe interior: the median dataset still places 13.6% of its rows outside [0.05, 0.95], since the sharpness s of (27) is sampled precisely to span easy and hard assignment regimes.

![](images/02afa50820c381239ba3457c26a49719021758a797daddb88eca79e02cf16cce.jpg)

![](images/3212a3717df86d0c49e35e5661d8cea8a5b6f0ab90a6246a2bfb021cbee8491d.jpg)  
Figure B.13: Delivered overlap across the prior, from the stored true propensity $\pi = \operatorname* { P r } [ A = 1 \mid X , U ]$ (no estimation involved). (a) Pooled distribution of the overlap margin min $( \pi , 1 - \pi )$ over all rows of 1024 datasets, on a logarithmic axis: the margin measures each unit’s distance from a degenerate assignment, folding both tails into one axis, and the log scale is what makes the mass piled on the clamp $\varepsilon = 1 0 ^ { - 3 }$ (dotted) visible at all. Bars are per-bin shares of all rows on a logarithmic grid, so bin width grows to the right. (b) Per-dataset quantile function of $\pi ,$ , summarised across datasets by the median curve with the 25–75% and 10–90% inter-dataset bands. Dashed lines mark the practical-overlap band [0.05, 0.95] in both panels.

## B.3.2 Exclusion restrictions

The proxy exclusion restriction of Assumption 2, $Y \perp \perp Z \mid ( A , X , U )$ and $W \perp \perp \left( A , Z \right) \mid \left( X , U \right)$ , holds by construction: (21) never wires Z into $f _ { W }$ or $f _ { Y }$ , nor W into $f _ { Z }$ or $f _ { A }$ . We audit this with a cross-fitted regression: on the Y side, the incremental $\Delta R _ { Z } ^ { 2 } = R ^ { 2 } ( \dot { Y } _ { \mathrm { o b s } } \mid A , X , U , \dot { Z } ) - R ^ { \dot { 2 } } ( Y _ { \mathrm { o b s } } \mid A , X , U )$ of adding Z to a conditioning set that already contains everything the clause names; on the A side, the analogous $\Delta R _ { W } ^ { 2 }$ for W against the raw treatment logit logit(π) given $( X , U , W )$ . Both are expected to be ≈ 0.

A near-zero $\Delta R ^ { 2 }$ is only informative if the estimator could have detected a leak had one been present. We therefore pair each check with a positive control that repeats the same regression with U dropped from the conditioning set (Y-side: $\{ A , X , Z \} ; \mathbf { A }$ -side: $\{ X , W \} )$ : since U is the only path linking Z to Y and W to A, removing it makes the excluded block genuinely predictive, and the control must light up clearly nonzero. Because the control’s power scales with how much confounding a given draw actually carries, the figure also colours and ranks each point by u dependenc $\acute { \iota } = R ^ { 2 } ( \mathrm { w i t h } \ U ) - R ^ { 2 } \nonumber$ (without $U )$ , the unique contribution of U to the target: the control should fan upward with warmer colour, while the check stays flat and near zero regardless of colour.

(a) <sub>Z</sub> excluded from <sub>f</sub> : <sub>R</sub><sup>2</sup>  
![](images/906ec6f3d5d983ba159d3333048f11762198874e654adb1c311c538dbb4b013a.jpg)

(b) <sub>W</sub> excluded from $f _ { A } \colon \Delta R _ { W } ^ { 2 }$  
![](images/fa255148de2cf74db45c22217b01ba648c0a7d2e894816b5b7b25e9e2afb10ac.jpg)  
Figure B.14: Exclusion-restriction audit (Assumption 2) over 1024 datasets. Each panel is one clause – (a) Z excluded from $f _ { Y }$ , (b) W excluded from $f _ { A } - \mathrm { a n d }$ each column a cross-fitted incremental $\Delta R ^ { 2 }$ of the excluded block: conditioning on U is the check (expected ≈ 0), control: U dropped the positive control (expected clearly positive). Colour and horizontal position both encode u dependence $= \ R ^ { 2 } ( \mathrm { w i t h \ } U ) - R ^ { 2 }$ (without U), i.e. how much confounding that draw carries (linear ramp, floored at $0 ,$ capped at the 95th percentile). Both checks stay on zero regardless of colour; both controls fan upward with warmer colour, most visibly in the top tercile of u dependence (Table B.5).

Figure B.14 confirms this reading. Both checks sit on zero (median $\Delta R _ { Z } ^ { 2 } = - 0 . 0 0 1 9 , \Delta R _ { W } ^ { 2 } = - 0 . 0 0 3 2 )$ , while both controls are slightly positive and rise further on the third of datasets with the largest U-dependence, with a positive Spearman correlation between u dependence and the control (0.375 Y-side, 0.415 A-side) visible as the colour gradient climbing alongside the control column. The zero-leak reading is therefore not an artefact of a blind test: the same regression detects a real, confounder-mediated dependence whenever U is removed, and detects none when the exclusion restriction is left intact.

Table B.5: Exclusion-restriction audit (Assumption 2): cross-fitted incremental $\Delta R ^ { 2 }$ of the excluded block over the stated conditioning set, for the check (conditions on $U )$ and the positive control (U dropped). Top confounded is the third of datasets with the largest u dependence $( R ^ { 2 }$ with U minus $R ^ { 2 }$ without it).
<table><tr><td>conditioning set role</td><td></td><td>datasets</td><td>median  $\Delta R ^ { 2 }$ </td><td></td><td>90th pct share &gt; 0.02</td></tr><tr><td colspan="6"> $Z$  excluded from  $f _ { Y }$ </td></tr><tr><td> $\{ A , X , U , Z \}$ </td><td>conditions on U</td><td>all</td><td>-0.002</td><td>0.007</td><td>0.021</td></tr><tr><td>{A, X, U, Z}</td><td>conditions on U</td><td>top confounded</td><td>-0.005</td><td>0.004</td><td>0.012</td></tr><tr><td> $\{ A , X , Z \}$ </td><td>positive control</td><td>all</td><td>0.004</td><td>0.099</td><td>0.313</td></tr><tr><td>{A, X, Z}</td><td>positive control</td><td>top confounded</td><td>0.029</td><td>0.222</td><td>0.598</td></tr><tr><td colspan="6">W excluded from  $f _ { A }$ </td></tr><tr><td> $\{ X , U , W \}$ </td><td>conditions on U</td><td>all</td><td>-0.003</td><td>0.009</td><td>0.020</td></tr><tr><td> $\{ X , U , W \}$ </td><td>conditions on U</td><td>top confounded</td><td>-0.004</td><td>0.004</td><td>0.021</td></tr><tr><td> $\{ X , W \}$ </td><td>positive control</td><td>all</td><td>0.012</td><td>0.218</td><td>0.415</td></tr><tr><td> $\{ X , W \}$ </td><td>positive control</td><td>top confounded</td><td>0.060</td><td>0.395</td><td>0.677</td></tr></table>

## B.3.3 Latent exchangeability and confounding strength

Latent exchangeability (Assumption 1) holds by construction. Treatment is drawn from $f _ { A } ( \boldsymbol { X } , \boldsymbol { U } , \boldsymbol { Z } , \varepsilon _ { A } )$ , while both potential outcomes are generated without treatment as an input and depend on $X , U , W$ and independent outcome noise (Equation (21)). Conditional on $( X , U )$ , the exogenous draws generating Z and treatment are independent of the ones that generate W and the potential outcomes. So, Y(a) ⊥⊥ $A \mid ( X , U )$ for each a.This establishes the assumption; we investigate the strength of the confounding of U below, but this does not test the conditional independence or prove finite-sample unbiasedness.

Figure B.15 compares five estimators (unadjusted; T-learners adjusting for X or $( X , W , Z )$ ; and oracle T-learners adjusting for $( X , U )$ or $( X , U , W ) )$ ) against the generated oracle ATE mean $( Y _ { 1 } - Y _ { 0 } )$ , with absolute errors divided by s $\mathrm { d } ( Y _ { \mathrm { o b s } } )$ (Table B.6). Median errors are close (unadjusted 0.041; oracle- $( X , U , W ) ~ 0 . 0 2 5 )$ . Errors above 0.1 sd $( Y _ { \mathrm { o b s } } )$ occur in 24.1% of datasets without adjustment but only 10.7–10.8% with either oracle. This reduction is consistent with residual confounding in some draws, although finite-sample estimation error also contributes to each rate. Figure B.16 groups datasets by the delivered U share of $f _ { A }$ and $f _ { Y } :$ : the backdoor-X error is higher mainly when the treatment mechanism receives a larger share of U-derived inputs. The pattern is modest at this sample size; adjusting for U removes this confounding pathway, explaining why the oracle error does not rise in the same way.

Table B.6: ATE error across adjustment sets. Errors are absolute deviations from the in-sample mean $( Y _ { 1 } - Y _ { 0 } )$ divided by sd $\lfloor \left( Y _ { \mathrm { o b s } } \right)$ . All adjusted estimates use cross-fitted T-learners; the oracle rows include the latent U.
<table><tr><td>rung</td><td>adjustment set</td><td>role</td><td>median error</td><td>90th pct</td><td>share &gt; 0.1</td></tr><tr><td>naive</td><td>none</td><td>unadjusted</td><td>0.041</td><td>0.195</td><td>0.241</td></tr><tr><td>backdoor X</td><td>X</td><td>correct if no latent confounder</td><td>0.037</td><td>0.163</td><td>0.211</td></tr><tr><td>backdoor XWZ</td><td>(X, W, Z)</td><td>DAG-blind: everything measured</td><td>0.030</td><td>0.121</td><td>0.148</td></tr><tr><td>oracle XU</td><td>(X, U)</td><td>identification: minimal sufficient set</td><td>0.028</td><td>0.104</td><td>0.108</td></tr><tr><td>oracle XUW</td><td>(X, U, W)</td><td>efficiency: adds W, a parent of  $f _ { Y }$ </td><td>0.025</td><td>0.105</td><td>0.107</td></tr></table>

![](images/5f18538e5ef1215d6b64d7f5fb03275964910a811b5047b3caff5fbaa0aa4dfa.jpg)

![](images/c25f891a89b75e59534af1b6e272a55c0fca6d2aa3ac895ee739953c8ebabeb5.jpg)  
Figure B.15: Confounding-strength comparison (Table B.6) over 1024 datasets. Each violin shows normalised absolute ATE error $| \widehat { \mathrm { A T E } } - \mathrm { A T E } | / \mathrm { s d } ( Y _ { \mathrm { o b s } } )$ on a log axis, scored against mean $( Y _ { 1 } - Y _ { 0 } )$ : naive (no adjustment), backdoor X and $( X , W , Z )$ (T-learners on measured covariates), and oracle (X, U) and (X, U, W) (T-learners also given the latent confounder).  
Figure B.16: Backdoor-X and oracle-(X, U) error over the delivered confounding-share grid: 1024 datasets binned $3 \times 3$ on the quantiles of the delivered U share of $f _ { A }$ (rows) and $f _ { Y }$ (columns) (Figure B.3). Cells report the median normalised error on a shared colour scale. The backdoor panel’s top row (delivered U share of $f _ { A }$ above 0.495) is elevated at 0.045–0.046 against 0.028–0.041 in the two rows below, and is nearly flat across $f _ { Y } { \mathrm { ' s } }$ share. The oracle panel shows no comparable pattern.

## B.3.4 Proxy relevance

Completeness (Assumption 4) asks that W and Z carry enough variation in $U$ for the bridge functions to be identified, not merely that they are wide enough. Appendices B.2.2 and B.2.4 enforce only a counting surrogate $- d _ { W } , d _ { Z } \geq d _ { U }$ and its categorical analogue dominates $( P , U ) -$ which holds on 100% of delivered datasets by construction and is silent on whether satisfying it buys functional recoverability of $U$ . We audit that directly with a cross-fitted regression: for each column $u _ { j }$ of $U$ and each proxy $P \in \{ W , Z \}$ , the incremental $\Delta R _ { P } ^ { 2 } ( u _ { j } ) =$ $R ^ { 2 } ( u _ { j } \mid X , P ) - R ^ { 2 } ( u _ { j } \mid X )$ , incremental over $X$ because $X$ is itself a parent of U (Equation 21) and a proxy that merely re-encoded it would be useless for identification however high its raw $R ^ { 2 }$ . We aggregate each proxy’s $\Delta R ^ { 2 }$ over $U ^ { \star } { } _ { \mathbf { s } }$ columns two ways: the mean (treating every column as equally important) and the minimum, since completeness is a claim about all of U and one unrecoverable direction is enough to break it.

Figure B.17 reads the same quantity three ways. Panel (a) shows both proxies are relevant on the typical draw (median $\Delta R ^ { 2 } \approx 0 . 2 8$ , over 93% of datasets positive) but the weakest $U$ column is a materially harder story (median $\Delta R ^ { 2 } \approx 0 . 0 5$ , roughly 60% of datasets below the 0.1 weak line). Panel (b) conditions on each proxy’s own dimension slack $d _ { P } - d _ { U } \mathbf { : }$ relevance rises with slack, validating the heuristic’s spirit. Panel (c) conditions on the proxy mechanism’s sampled noise share, and relevance falls as that share grows, as the dilution story predicts. Table B.7 separates the two components of slack. Mean relevance has a modest positive correlation with proxy width $( \rho = 0$ .166 for both proxies). Weakest-column relevance rises with slack $( \rho \approx 0 . 2 8 )$ and falls with $d _ { U }$ $( \rho \approx - 0 . 2 6 )$ , but has almost no association with proxy width alone $( \rho = 0 . 0 3 4$ for $W , 0 . 0 1 7$ for $Z ) .$ . Because the prior samples proxy width conditional on $d _ { U }$ , wider proxies tend to accompany higher-dimensional $U ;$ recovering the weakest column can therefore remain difficult as proxy width increases, especially since taking a minimum over more $U$ columns tends to lower that statistic.

Table B.7: Proxy-relevance audit (Assumption 4): cross-fitted incrementa $\Delta R ^ { 2 } = R ^ { 2 } ( u _ { j } \mid X , P ) - R ^ { 2 } ( u _ { j } \mid X )$ of each proxy $P$ for the latent columns, aggregated over U by the mean or by the weakest column. The counting heuristic (Appendix B.2.4) passes for every dataset, so the share column is the rate at which it and functional recoverability disagree. $\rho$ is Spearman’s correlation with each candidate driver: the proxy’s own width $d _ { P } , d _ { U }$ the slack $d _ { P } - d _ { U }$ that conflates the two, and the sampled noise share of the $f _ { P }$ input.
<table><tr><td>aggregation over U</td><td>median  $\Delta R ^ { 2 }$ </td><td>share  $< 0 . 1$ </td><td> $\rho d _ { P }$ </td><td> $\rho d _ { U }$ </td><td> $\rho d _ { P } - d _ { U }$ </td><td> $\rho$  noise</td></tr><tr><td>W</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>mean</td><td>0.273</td><td>0.226</td><td>0.166</td><td>-0.002</td><td>0.208</td><td>-0.202</td></tr><tr><td>weakest</td><td>0.053</td><td>0.593</td><td>0.034</td><td>-0.249</td><td>0.277</td><td>-0.140</td></tr><tr><td>Z</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>mean</td><td>0.271</td><td>0.235</td><td>0.166</td><td>-0.023</td><td>0.226</td><td>-0.208</td></tr><tr><td>weakest</td><td>0.052</td><td>0.588</td><td>0.017</td><td>-0.272</td><td>0.281</td><td>-0.095</td></tr></table>

(a) pooled relevance  
![](images/26f78fa6bf19ad2f559f3aff30f714304803f2996e63b02ba56a0eadaeae8ddc.jpg)

(b) by dimension slack  
![](images/0694b72a96f4e6bc1cb73fb284082799dd687e131d8d68940bfc38c389d7722e.jpg)  
dimension slack $d _ { P } - d _ { U }$

(c) by mechanism noise  
![](images/95aa177d06a9a8605a2ea23a401ae07d0509ba19c570bf460734f2f26937ec81.jpg)  
noise share of the f<sub>P</sub> input  
Figure B.17: Proxy relevance measured by the cross-fitted increment $\Delta R _ { P } ^ { 2 } ( u _ { j } ) = R ^ { 2 } ( u _ { j } \mid X , P ) - R ^ { 2 } ( u _ { j } \mid X )$ shown separately for W and Z. (a) Distribution of the mean and weakest-column $\Delta R ^ { 2 }$ over U; boxes show the IQR and whiskers the 10th–90th percentiles. (b) Median $\Delta R ^ { 2 }$ by proxy dimension slack $d _ { P } - d _ { U }$ , with slack $\geq 7$ pooled. (c) Median $\Delta R ^ { 2 }$ by sampled noise share of $f _ { P } ,$ in six equal-frequency bins per proxy. Bands in (b) and (c) show the IQR.

## C Model architecture, objective, training, and validation

## C.1 Architecture

PROXIMALFM encodes a dataset in the three sequential stages of TabICLv2 (Qu et al., 2026), which we use unchanged. A column embedder maps each scalar cell to an embedding through a shared inducing-point set transformer applied along the unit axis, so that a cell is represented relative to the empirical distribution of its own column. A row interactor then treats the embedded cells of a unit as a sequence of tokens along the column axis, applies a transformer with rotary positional encoding, and concatenates the outputs of $n _ { \mathrm { c l s } }$ prepended learnable CLS tokens into a single vector per unit. Finally, an in-context learner runs a transformer over the sequence of unit vectors in which query units attend to context units but not to one another; no positional encoding is applied along the unit axis, so the model is equivariant to the ordering of the dataset. All blocks are pre-norm and use the query-aware scalable softmax of Qu et al. (2026). Our only removals are the classification-specific components, which a continuous causal estimand does not use; the remaining departures are described below.

A target that is never observed. Every departure below follows from one change of task. In tabular in-context learning the query is a context row with a missing label: the model predicts $y ^ { \star }$ , the same variable it has just seen realized in the context, and the query differs from a context unit only in that one entry is withheld. PROXIMALFM predicts $Q ( x ^ { \star } ) = \mathbb { E } [ Y ( 1 ) - Y ( 0 ) \mid X = x ^ { \star } ]$ , which is realized for no unit in the table. The observed columns are evidence about the mechanism that generates it, and supervision comes from the prior’s oracle targets rather than from held-out labels.

Two context columns consequently have no query counterpart. The outcome y is context-only in TabICL as well, but there it is withheld because it is the answer; here it is withheld because a query unit is a covariate profile, not a realized unit, and the same holds for the treatment $a \mathrm { : }$ the estimand averages over both arms, so no value of a can sensibly be attached to the query vector $x ^ { \star }$ . The proxies are a different case. A query unit can have a $w ^ { \star }$ and a $z ^ { \star }$ , and the prior generates them, so withholding them is a design choice rather than a structural necessity: conditioning on them would target $\mathbb { E } [ Y ( 1 ) - Y ( 0 ) \ | \ X , W , Z ]$ instead. We follow the proximal literature in conditioning on the baseline covariates alone (Section 2.2) and marginalizing over the proxies. At inference the model therefore sees the context table in full and a query reduced to $x ^ { \star }$ alone. The modifications below adapt the backbone to exactly this interface.

Structural role embeddings. The backbone from TabICL consumes a single undifferentiated feature matrix, whereas proximal identification treats the three observed blocks asymmetrically: the roles of $Z , W$ and X are necessary to be known in order to infer the causal estimand of interest. Nothing in the values themselves reveals that role. Column order alone cannot supply it either: the block sizes $( d _ { X } , d _ { Z } , d _ { W } )$ are resampled for every dataset, so no fixed column index marks where one block ends and the next begins, and the model would have to infer the two boundaries from the data before it could use them. We therefore tag the role explicitly. Writing lin(·) for the shared cell projection into $\mathbb { R } ^ { E }$ , a cell v of block $g \in \{ X , Z , W \}$ enters the column embedder as

$$
\begin{array} { r } { \operatorname* { l i n } ( v ) + r _ { g } , \qquad r _ { X } , r _ { Z } , r _ { W } \in \mathbb { R } ^ { E } \mathrm { ~ l e a r n a b l e } , } \end{array}\tag{31}
$$

after which the three blocks are concatenated along the column axis into a single sequence of $d _ { X } + d _ { Z } + d _ { W }$ columns. Role identity is thus carried by the embedding rather than by position. The treatment A is not passed as a column at all: it enters only through the target encoder described next.

Treatment-aware target encoder. The remaining two observed variables, $y$ and $^ { a , }$ are not handled the way $X .$ $Z$ and W are. They are present for context units and absent for queries, so instead of being concatenated to the input row as columns, they are added to the tokens of context units. This also settles how a is encoded: it qualifies y rather than standing beside it, since an outcome means one thing under treatment and another under control. We therefore hold two projections and select between them with the observed treatment indicator,

$$
\begin{array} { r } { \mathrm { e n c } ( y _ { i } , a _ { i } ) = ( 1 - a _ { i } ) \operatorname* { l i n } _ { y _ { 0 } } ( y _ { i } ) + a _ { i } \operatorname* { l i n } _ { y _ { 1 } } ( y _ { i } ) , \qquad \operatorname* { l i n } _ { y _ { 0 } } , \operatorname* { l i n } _ { y _ { 1 } } : \mathbb { R } \to \mathbb { R } ^ { E } . } \end{array}\tag{32}
$$

Outcome and arm are thus encoded jointly, and the treated and control response surfaces may occupy different directions in embedding space from the first layer onwards. The encoder is applied at two points, the column embedder and the in-context learner, with untied parameters.

Standardization and query-row proxy masking. Each of X, Z, W and $y$ is standardized column-wise using the empirical mean and standard deviation of the context units only, so that no query information reaches the model through the scale and predictions are invariant to affine rescaling of the inputs.

A query unit is then presented without its proxies. The $Z$ and W positions of query rows are filled with a fixed value of 0 and then masked out of the attention in the row interactor, so those tokens do not merely carry a neutral value but are invisible to the query representation altogether. The same protocol applies during training and at inference time.

Output head. For each query unit the in-context learner emits B logits, passed through a softmax to give the predictive distribution $q _ { \theta } [ \cdot \ | \ x ^ { \star } , \mathcal { D } _ { N } ]$ over a fixed grid of B equal-width bins spanning $[ v _ { \mathrm { m i n } } , v _ { \mathrm { m a x } } ]$ , expressed in units of the context standard deviation of $y .$ Discretizing the predictive distribution in this way is standard practice for PFNs (Hollmann et al., 2023) and is what allows a single forward pass to return a full distribution without committing to a parametric family; amortized causal models adopt it as well (Balazadeh Meresht et al., 2026). A single head is placed directly over the CATE rather than one head per arm. Although $Q ( x )$ = $\mu _ { 1 } ( x ) - \mu _ { 0 } ( x )$ , the two CEPOs are functionals of the same unknown SCM and are therefore generally dependent under its posterior given $\mathcal { D } _ { N }$ . Their marginal posterior distributions alone do not determine the posterior of their difference: convolving them would implicitly impose posterior independence and omit their covariance, which can misstate the CATE uncertainty. Directly learning the distribution of $Q ( x )$ avoids this unsupported joint-posterior assumption.

Two properties of the discretization matter for what follows. First, we read the bin probabilities as a piecewiseuniform density rather than as point masses at the bin centers. This is what makes the noise convolution of (34) exact, and it puts a floor of $w / \sqrt { 1 2 } -$ — the standard deviation of a single bin, of width $w = ( v _ { \operatorname* { m a x } } - v _ { \operatorname* { m i n } } ) / B - \mathrm { o n }$ any posterior standard deviation the head can report. Second, the grid is a hard support constraint: a target outside $[ v _ { \mathrm { m i n } } , v _ { \mathrm { m a x } } ]$ cannot be represented at all, so the bounds must be wide enough to cover the CATE distribution the prior produces. Chosen $v _ { \operatorname* { m a x } } , v _ { \operatorname* { m i n } }$ and $B$ are reported in Table C.1.

Hyperparameters. Table C.1 lists the configuration of the three stages. Two entries are worth reading together. The column embedder and the row interactor operate at the embedding dimension $E ,$ , but the in-context learner receives the concatenation of the $n _ { \mathrm { c l s } }$ CLS tokens and therefore runs at $E \cdot n _ { \mathrm { c l s } } = 5 1 2 $ ; since it is also the deepest stage, it accounts for 28.3M of the 29.6M parameters, the column embedder and row interactor contributing 0.9M and 0.4M respectively. These parameters are kept identical to those of TabICLv2, so that its pre-trained weights can be used to initialize every component we did not modify.

Table C.1: Architectural hyperparameters of PROXIMALFM. The three stages are those of TabICLv2, kept at its values so that its pre-trained weights initialize every component we did not modify; the output head is ours. Parameter counts are per stage, the in-context learner’s including the head.
<table><tr><td>Stage</td><td>Hyperparameter</td><td>Value</td></tr><tr><td rowspan="5">Shared</td><td>Embedding dimension E</td><td>128</td></tr><tr><td>Feedforward expansion factor</td><td>2</td></tr><tr><td>Activation</td><td>GELU</td></tr><tr><td>Normalization</td><td>pre-norm</td></tr><tr><td>Dropout</td><td>0.0</td></tr><tr><td rowspan="4">Column embedder</td><td>Blocks</td><td>3</td></tr><tr><td>Attention heads</td><td>8</td></tr><tr><td>Inducing points</td><td>128</td></tr><tr><td>Scalable softmax</td><td>query-aware, elementwise</td></tr><tr><td rowspan="4">Row interactor</td><td>Blocks</td><td>3</td></tr><tr><td>Attention heads</td><td>8</td></tr><tr><td>CLS tokens  $n _ { \mathrm { c l s } }$ </td><td>4</td></tr><tr><td>RoPE base</td><td> $1 0 ^ { 5 }$ </td></tr><tr><td rowspan="2">In-context learner</td><td>Blocks</td><td>12</td></tr><tr><td>Attention heads</td><td>8</td></tr><tr><td rowspan="2">Output head</td><td>Model dimension  $E \cdot n _ { \mathrm { c l s } }$ </td><td>512</td></tr><tr><td>Bins B</td><td>2000</td></tr><tr><td rowspan="4">Parameters</td><td>Grid  $[ v _ { \mathrm { m i n } } , v _ { \mathrm { m a x } } ]$ </td><td>[−5,5]</td></tr><tr><td>Column embedder</td><td>0.9M</td></tr><tr><td>Row interactor</td><td>0.4M</td></tr><tr><td>In-context learner</td><td>28.3M</td></tr><tr><td></td><td>Total</td><td>29.6M</td></tr></table>

## C.2 The noise-aware histogram loss

This section gives the closed form of the training objective of Section 3.2, together with the motivation for scoring a noise-convolved predictive distribution rather than fitting the Monte Carlo targets directly.

Notation and standardization. All quantities below live on the context-standardized scale. For a dataset with N context rows, let $m _ { N }$ and $s N$ denote the empirical mean and standard deviation of the observed outcomes $\{ y _ { i } \} _ { i = 1 } ^ { N }$ . Outcomes are standardized as $( y - m _ { N } ) / s _ { N }$ , so the CATE target and its Monte Carlo variance enter as

$$
t _ { i } : = \frac { \hat { Q } _ { K } ( x _ { i } ) } { s _ { N } } , \qquad \varsigma _ { i } ^ { 2 } : = \frac { S _ { i } ^ { 2 } } { s _ { N } ^ { 2 } } ,\tag{33}
$$

the location shift $m _ { N }$ cancelling in the difference $\hat { \mu } _ { 1 } - \hat { \mu } _ { 0 }$ . Because $m _ { N }$ and $s _ { N }$ are computed on context rows only, no query information leaks into the scale.

The head emits B logits over a fixed uniform grid $e _ { 1 } < e _ { 2 } < \dots < e _ { B + 1 }$ spanning $[ v _ { \mathrm { m i n } } , v _ { \mathrm { m a x } } ]$ with bin width $w = ( v _ { \operatorname* { m a x } } - v _ { \operatorname* { m i n } } ) / B$ , and $q _ { \theta } [ b \mid x _ { i } , \mathcal { D } _ { N } ]$ denotes the predicted probability of bin $\boldsymbol { b } = [ e _ { b } , e _ { b + 1 } )$ . We write Φ for the standard normal CDF. The quantity $\sigma ^ { 2 }$ denotes the variance of an auxiliary Gaussian smoothing kernel, expressed on this standardized scale; it is distinct from the row-specific Monte Carlo variance $\varsigma _ { i } ^ { 2 }$ and is annealed during training according to the schedule described in Appendix C.3.

Closed form. Whereas $q _ { \theta }$ is the model’s predictive distribution over the estimand $Q ( x _ { i } )$ , the quantity actually observed during training is the noisy measurement $t _ { i } ; \tilde { q } _ { \theta }$ is the predictive density that $q _ { \theta }$ induces over that measurement. Write the per-row variance as $\kappa _ { i } ^ { 2 } : = \varsigma _ { i } ^ { 2 } + \sigma ^ { 2 }$ , combining the known target noise with the annealed smoothing scale (also known). Conditioning on which bin contains the estimand, the law of total probability gives

$$
\begin{array} { r } { \displaystyle \tilde { q } _ { \theta } \big ( t \mid x _ { i } , \mathcal { D } _ { N } \big ) = \sum _ { b = 1 } ^ { B } \mathbb { P } \big ( Q \in [ e _ { b } , e _ { b + 1 } ) \mid x _ { i } , \mathcal { D } _ { N } \big ) \cdot p \big ( t \mid Q \in [ e _ { b } , e _ { b + 1 } ) \big ) } \\ { = \displaystyle \sum _ { b = 1 } ^ { B } q _ { \theta } [ b \mid x _ { i } , \mathcal { D } _ { N } ] \ \frac { 1 } { w } \left[ \Phi \left( \frac { e _ { b + 1 } - t } { \kappa _ { i } } \right) - \Phi \left( \frac { e _ { b } - t } { \kappa _ { i } } \right) \right] . } \end{array}\tag{34}
$$

The first factor is read straight off the head. The second follows from the piecewise-uniform continuous reading of the histogram: given that Q lies in bin $b ,$ the head places it uniformly within that bin, so

$$
p \big ( t \mid Q \in [ e _ { b } , e _ { b + 1 } ) \big ) = \int _ { e _ { b } } ^ { e _ { b + 1 } } \frac { 1 } { w } \mathcal { N } \big ( t ; u , \kappa _ { i } ^ { 2 } \big ) \mathrm { d } u = \frac { 1 } { w } \left[ \Phi \bigg ( \frac { e _ { b + 1 } - t } { \kappa _ { i } } \bigg ) - \Phi \bigg ( \frac { e _ { b } - t } { \kappa _ { i } } \bigg ) \right] ,\tag{35}
$$

where the integral runs over the mean of the Gaussian. The training objective is the negative log-likelihood of the observed target under (34), evaluated at $t = t _ { i } { : }$

$$
\mathcal { L } _ { \mathrm { n o i s e , a w a r e } } = - \frac { 1 } { n _ { q } } \sum _ { i = 1 } ^ { n _ { q } } \log \left( \sum _ { b = 1 } ^ { B } q _ { \theta } [ b \mid x _ { i } , \mathcal { D } _ { N } ] \frac { 1 } { w } \left[ \Phi \left( \frac { e _ { b + 1 } - t _ { i } } { \kappa _ { i } } \right) - \Phi \left( \frac { e _ { b } - t _ { i } } { \kappa _ { i } } \right) \right] \right) .\tag{36}
$$

Why convolve the model rather than fit the target. Consider the population risk of (36) at fixed context. The target is distributed as $t \sim p ( { \bf \cdot } \mid \mathcal { D } _ { N } ) * \mathcal { N } ( 0 , \varsigma ^ { 2 } )$ , the posterior over the estimand blurred by Monte Carlo error. A negative log-likelihood is minimized when the scored density matches the density of the data, i.e. when

$$
q _ { \theta } * \mathcal { N } ( 0 , \varsigma ^ { 2 } + \sigma ^ { 2 } ) = p ( \cdot \mid \mathcal { D } _ { N } ) * \mathcal { N } ( 0 , \varsigma ^ { 2 } ) \quad \Longleftrightarrow \quad q _ { \theta } * \mathcal { N } ( 0 , \sigma ^ { 2 } ) = p ( \cdot \mid \mathcal { D } _ { N } ) .\tag{37}
$$

The Monte Carlo variance cancels on both sides: the correction is exact for any value of $\sigma .$ , and the recovered q converges to the posterior as $\sigma$ is annealed. Fitting the same head directly to $t _ { i }$ instead—the noise-unaware objective—has population optimum $q _ { \theta } = p ( { \bf \cdot } { \bf \nabla } | \mathcal { D } _ { N } ) * \mathcal { N } ( 0 , \varsigma ^ { 2 } + \sigma ^ { 2 } )$ , over-dispersed by exactly the target noise. The distortion is not a constant offset that could be calibrated away post hoc: $\dot { \varsigma _ { i } } ^ { 2 } = \dot { \mathrm { V a r } } ( \tau \mid X = x _ { i } ) \dot { / } K$ varies across query points, and is largest precisely where the individual treatment effect is most heterogeneous, so the inflation is heaviest in the regions where posterior width is most informative.

Full objective. To align the point prediction with the estimand, a squared-error penalty on the posterior mean is added,

$$
\mathcal { L } _ { \mathrm { m e a n } } = \frac { 1 } { n _ { q } } \sum _ { i = 1 } ^ { n _ { q } } \biggl ( \mathbb { E } _ { q _ { \theta } } \left[ \boldsymbol { Q } \mid \boldsymbol { x } _ { i } , \mathcal { D } _ { N } \right] - t _ { i } \biggr ) ^ { 2 } , \qquad \mathbb { E } _ { q _ { \theta } } \left[ \boldsymbol { Q } \mid \boldsymbol { x } _ { i } , \mathcal { D } _ { N } \right] = \sum _ { b = 1 } ^ { B } q _ { \theta } [ b \mid \boldsymbol { x } _ { i } , \mathcal { D } _ { N } ] c _ { b } ,\tag{38}
$$

with $c _ { b } = ( e _ { b } + e _ { b + 1 } ) / 2$ the bin centers, giving $\mathcal { L } = \mathcal { L } _ { \mathrm { n o i s e . a w a r e } } + \lambda _ { \mathrm { m e a n } } \mathcal { L } _ { \mathrm { m e a n } }$ . The penalty does not compete with the distributional objective: because the Monte Carlo error is zero-mean, the minimizer of (36) already carries the posterior mean and hence minimizes (38) as well, leaving the population optimum of $\mathcal { L }$ unchanged for any $\lambda _ { \mathrm { { m e a n } } } \geq 0$ . At inference time the predictive distribution is reported un-blurred, as $q _ { \theta } ( \cdot \mid x ^ { \star } , \mathcal { D } _ { N } )$ .

## C.3 Training the model

All results in this paper come from a single training run, and every number below is a property of that one run.

Configuration. Table C.2 reports the configuration in full: the prior the datasets are drawn from, the objective and its smoothing schedule, the optimizer, the pre-trained backbone the run is initialized from, and the validation sets used for checkpoint selection. Training runs for 75 000 steps at an effective batch of 64 datasets, accumulated in micro-batches of 16, for 4.8M datasets consumed in total — each drawn fresh from the prior and seen exactly once, so there is no notion of an epoch and no train/validation split on the prior itself.

![](images/5110550a8769fd9c45ce50fa1477ba3eef89d9259bd4560418b0e079a27fde52.jpg)

(b) past the anneal (step 3.5k on)  
![](images/ec8d3b225a570876463c802e40a635d936370571d797174a6bd442a4dd45158e.jpg)

![](images/347de1dfce977e0b39d908ce890e9d909cc33addd838f654df84c0118a53cbe9.jpg)

(d) mean term in detail  
![](images/a4645f132125e300fe348d14703ab36e94efa6c212c9114320a46f9033669a4d.jpg)  
Figure C.1: Training curves for PROXIMALFM. (a) The objective and its two terms against log step, with the σ-annealing window shaded; the dashed line is the loss of a uniform density over the CATE grid, matching the value at initialisation. (b) The same curves on a linear step axis after the anneal. (c) The σ and learning-rate schedules. (d) A magnified view of the weighted mean term, $\lambda _ { \mathrm { m e a n } } \mathcal { L } _ { \mathrm { m s e } }$ , with the σ-annealing window again shaded. Stochastic curves show the median and interquartile band within log-spaced step bins in panels (a) and (d), and linearly spaced bins in panel (b); the deterministic schedules in panel (c) are shown without binning.

Schedules and training curves. Figure C.1 shows the objective and the two schedules. The learning rate follows a linear warmup over 2 500 steps into a cosine decay; the smoothing scale σ of (36) is held at its initial value for 500 steps and then annealed linearly over 3 000 steps to two bin widths, its final value for the remaining of the training (panel c). The anneal is the reason the loss level is not comparable across the shaded window: changing σ changes the training objective, so movement during the anneal cannot be attributed solely to learning.

Panels (a) and (b) show that, after the initial phase, the total objective becomes dominated by the distributional term, the mean term staying a small regularizer term. Once the σ anneal is complete, both the distributional term and the total objective continue to decrease through step 75 000, but at a diminishing rate. Panel (d) zooms on the mean term of the loss. We observe that it follows the behaviour of the distributional term closely.

Validity of the objective. Three logged diagnostics establish that the loss the curves above report was well posed throughout. First, the grid of the output head is a hard support constraint: a target outside $[ v _ { \mathrm { m i n } } , v _ { \mathrm { m a x } } ]$ cannot be represented by any forecast on the grid, so a non-negligible out-of-grid fraction would invalidate every number scored on it. Averaged over the CATE targets in the on-the-fly synthetic training prior, the share of targets falling outside the grid is $2 . 0 \times 1 0 ^ { - 4 }$ , i.e. 0.02%, exceeding 1% on 275 of 75 000 steps.

Second, the Monte Carlo targets carried their full replicate budget. Each query row’s CATE target is an average over K replicate draws, and the realized K can be capped below the requested value by the generator’s memory budget. The mean realized K was 250 — the requested value — at every logged step, so the cap never bound.

Third, sampling noise remained material at K = 250 Monte Carlo replicates. The ratio of the mean target-noise standard deviation $\varsigma _ { i }$ to the mean predicted posterior standard deviation averaged 0.20, rising from 0.15 shortly after the anneal to 0.22 over the final third as the posterior sharpened while $\varsigma _ { i } \propto K ^ { - 1 / 2 }$ remained fixed. Ignoring this noise would therefore add the heterogeneous per-row variance $\varsigma _ { i } ^ { 2 }$ to the learned posterior variance, motivating the noise-aware correction.

Cost. Table C.3 summarizes the computational cost of the run. Interestingly, model optimization was not the dominant expense: generating the synthetic prior consumed 43.46 GPU-hours (65.7%), approximately $2 . 4 \times$ the cost of the forward and backward passes (27.4%). Validation accounted for a further 5.4%, while checkpointing, logging, and startup contributed only 1.5%. In total, training consumed 66.15 H100 GPU-hours over 75 000 steps and 4.8 million synthetic datasets, averaging 3.17 seconds per step.

## C.3.1 Post-training on larger context sizes

The main training run represents contexts of at most 2 048 observations. We therefore perform a post-training experiment to test whether the limited support at larger context sizes contributes to the plateau observed in Section 4.2. Starting from the completed main-training checkpoint, we optimize for a total of 30 000 steps on the same causal prior and noise-aware objective, but draw context sizes uniformly from 64 to 5 000 observations (with 128 query rows). Uniform, rather than log-uniform, sampling ensures that the adaptation gives substantial weight to the newly introduced large contexts while retaining smaller contexts for rehearsal. We retain the effective metabatch size of 64 datasets, use a lower learning rate of $2 \times 1 0 ^ { - 5 }$ with a 500-step warmup, and enable activation recomputation to fit the longer sequences.

To make this adaptation inexpensive and limit changes to the learned representation, we load the complete maintraining checkpoint and freeze all parameters except the in-context predictor. Within that predictor, its first six transformer blocks are frozen as well, so only its later layers and unfrozen predictor parameters are optimized. We evaluate the resulting checkpoints on separate frozen prior and realistic validation suites constructed at large context sizes.

Table C.2: Training configuration of PROXIMALFM: the prior the model is trained on, the objective and its smoothing schedule, the optimizer, the pre-trained backbone it is initialized from, and the validation sets used for checkpoint selection.
<table><tr><td>Stage</td><td>Hyperparameter</td><td>Value</td></tr><tr><td rowspan="7">Prior</td><td>Confounder dimension  $d _ { U }$ </td><td>1-10</td></tr><tr><td>Treatment proxy dimension  $d _ { Z }$ </td><td>1-10</td></tr><tr><td>Outcome proxy dimension dw</td><td>1-10</td></tr><tr><td>Covariate dimension  $d _ { X }$ </td><td>0-10</td></tr><tr><td>Context units N</td><td>64–2 048 (uniform)</td></tr><tr><td>Query units  $n _ { q }$ </td><td>128</td></tr><tr><td>Monte Carlo worlds K</td><td>250</td></tr><tr><td rowspan="10">Objective</td><td>MC targets on context units</td><td>no</td></tr><tr><td>Loss</td><td>noise_aware</td></tr><tr><td>Mean penalty  $\lambda _ { \mathrm { m e a n } }$ </td><td>0.5</td></tr><tr><td>Bins B</td><td>2000</td></tr><tr><td>Grid  $[ v _ { \mathrm { m i n } } , v _ { \mathrm { m a x } } ]$  Bin width w</td><td>[−5,5]</td></tr><tr><td>Initial smoothing</td><td>0.005</td></tr><tr><td> $\sigma _ { 0 }$  σ held constant for</td><td>0.5 500 steps</td></tr><tr><td>σ linear decay over</td><td></td></tr><tr><td>Final smoothing σ</td><td>3 000 steps</td></tr><tr><td>Optimizer</td><td>0.01 (2w)</td></tr><tr><td rowspan="10">Optimization</td><td></td><td>AdamW 1 × 10−4</td></tr><tr><td>Learning rate</td><td></td></tr><tr><td>Schedule</td><td>cosine_warmup</td></tr><tr><td>Warmup</td><td>2 500 steps</td></tr><tr><td>Weight decay</td><td>0</td></tr><tr><td>Gradient clipping</td><td>1</td></tr><tr><td>Training steps</td><td>75000</td></tr><tr><td>Datasets per step</td><td>64 (16 × 4 accumulated)</td></tr><tr><td>Precision</td><td>bfloat16(AMP)</td></tr><tr><td>Seed (numpy / torch)</td><td>1234 / 1234</td></tr><tr><td rowspan="3">Initialization</td><td>Init. checkpoint</td><td>ckpt_tabicl/tabicl-regressor-v2-20260212.ckpt</td></tr><tr><td>Transfer rules</td><td>ckpt_tabicl/transfer_rules.yaml</td></tr><tr><td>Weights only (fresh optimizer)</td><td>yes</td></tr><tr><td rowspan="5">Validation</td><td>Prior set</td><td>prior_val_v1</td></tr><tr><td>Prior set: episodes / interval</td><td>500 / 200 steps</td></tr><tr><td>Realistic set</td><td>synt_val_v1</td></tr><tr><td>Realistic set: episodes / interval</td><td>480 / 100 steps</td></tr><tr><td>Selection metric</td><td>synt_dist/crps (min)</td></tr><tr><td></td><td></td><td></td></tr></table>

Table C.3: Cost of the PROXIMALFM training run, as measured GPU-hours. Validation is the two frozen evaluation datasets, scored at their logging intervals throughout training.
<table><tr><td colspan="2"></td><td>Hours</td><td> $\%$ </td></tr><tr><td rowspan="5">Compute</td><td>Prior generation</td><td>43.46</td><td>65.7</td></tr><tr><td>Forward / backward</td><td>18.12</td><td>27.4</td></tr><tr><td>Validation, prior set (376 passes)</td><td>1.27</td><td>1.9</td></tr><tr><td>Validation, realistic set (751 passes)</td><td>2.34</td><td>3.5</td></tr><tr><td>Checkpointing, logging, startup</td><td>0.96</td><td>1.5</td></tr><tr><td>Total</td><td>GPU-hours (1× H100 NVL)</td><td>66.15</td><td>100.0</td></tr><tr><td rowspan="3">Scale</td><td>Training steps</td><td>75000</td><td></td></tr><tr><td>Synthetic datasets consumed</td><td>4 800 000</td><td></td></tr><tr><td>Seconds per step</td><td>3.17</td><td></td></tr></table>

## C.4 Validation and posterior diagnostics

## C.4.1 Validation datasets

We monitor the training of PROXIMALFM on a fixed suite of 480 datasets, balanced across 12 hand-written realistic proximal mechanisms. These mechanisms cover clinical severity and frailty, latent patient subtypes, thresholdbased treatment assignment, utilization counts, censored survival, marketing uplift, educational selection, wage returns, hierarchical site effects, nonlinear ordinal proxies, interaction-driven confounding, and near-null effects under strong confounding. These settings span a diverse range of datasets: continuous, binary, ordinal and categorical variables, the treatment is binary, the outcomes may be continuous, skewed, count-valued, zero-inflated, or censored.

For each dataset, the dimensions of U, X, W and $Z$ are sampled with the same procedure as in the prior $( \mathsf { A p - }$ pendix B.2.2). Each episode contains exactly 2048 context rows and 128 query points. The query target CATEs are computed by averaging 10 000 Monte Carlo replicates of ITE draws. At this replicate count, the Monte Carlo sampling noise is approximately 0.025 times the predictive posterior standard deviation (of the last model checkpoint). The same fixed validation set is evaluated at every training checkpoint.

## C.4.2 Point-estimate validation

We measure the point-estimate performance at both ATE and CATE levels. The ATE prediction is estimated as the average of the CATE predictions from the query set,

$$
\widehat { \mathrm { A T E } } = \frac { 1 } { n _ { q } } \sum _ { i = 1 } ^ { n _ { q } } \hat { \tau } _ { i } ,
$$

and so is the target ATE also computed from the references CATEs. To evaluate the ATE estimation, we use the root mean squared error (RMSE) and the bias of the ATE normalized by the standard deviation of the episode’s CATEs.

Regarding CATE evaluation, we use normalized precision in estimation of heterogeneous effect (nPEHE),

$$
\mathrm { n P E H E } = \sqrt { \frac { \frac { 1 } { n _ { q } } \sum _ { i = 1 } ^ { n _ { q } } ( \hat { \tau } _ { i } - \tau _ { i } ) ^ { 2 } } { \mathrm { V a r } _ { i } ( \tau _ { i } ) } } .\tag{39}
$$

We also measure the shape- $R ^ { 2 }$ , which isolates the recovery of treatment-effect heterogeneity from constant level error,

$$
R _ { \mathrm { s h a p e } } ^ { 2 } = 1 - \frac { \mathrm { V a r } _ { i } ( \hat { \tau } _ { i } - \tau _ { i } ) } { \mathrm { V a r } _ { i } ( \tau _ { i } ) } .\tag{40}
$$

A constant-ATE prediction gives nPEHE = 1 and $R _ { \mathrm { s h a p e } } ^ { 2 } = 0$

Figure C.2 shows the evolution of point-estimate performance on the fixed realistic-mechanism validation suite. In the initial stage of training, the model performs worse than the constant-ATE baseline (CATE nPEHE above 1 and the shape $R ^ { 2 }$ is close to or below zero). After approximately 3 500 training steps, the median nPEHE falls below the constant-ATE baseline and the shape $R ^ { 2 }$ becomes positive. Performance then continues to improve throughout training: ATE RMSE decreases, normalized ATE bias moves towards zero, nPEHE falls further, and the shape $R ^ { 2 }$ increases. The narrowing interquartile bands indicate that performance also becomes more consistent across validation episodes.

## C.4.3 Distributional validation

We evaluate the predictive posterior using the continuous ranked probability score (CRPS), the predictive standard deviation relative to the prediction error, and empirical coverage of central credible intervals. For query row $i ,$ let $q _ { i }$ denote the predicted distribution over $Q ( x _ { i } )$ , with CDF $F _ { i }$ mean $\mu _ { i } ,$ and variance $v _ { i } ^ { 2 }$ . Let $Q _ { K , i }$ denote the Monte Carlo CATE target and let $\varsigma _ { i } ^ { 2 }$ denote its estimated sampling variance.

The CRPS evaluates the full predictive distribution against the target:

$$
\mathrm { C R P S } ( q _ { i } , Q _ { K , i } ) = \int { [ F _ { i } ( t ) - { \bf 1 } \{ t \ge Q _ { K , i } \} ] ^ { 2 } ~ \mathrm { d } t } .\tag{41}
$$

![](images/33f90dc57497d8a9527a4e5eee145d531fc3adfe171469b3ecbe8cfe9b230f9f.jpg)

![](images/fdfbfa532b077c59b33eaa1ca10022921dd189d5a00f0c6651394e87267dc798.jpg)

![](images/84c196e17f5901164b6957fdcce5885bcaa7427e6ebf2ab0284e8e0047adf0f5.jpg)

(d) CATE shape $R ^ { 2 }$  
![](images/cf93f0c10c12345982bc6cab39abf2c4e2c53c29e0d70a704f0b840344964095.jpg)  
Figure C.2: Evolution of point-estimate performance during training on the fixed realistic-mechanism validation suite (Appendix C.4.1). Curves show the validation metric at each evaluation step. The shaded regions in panels (b)–(d) indicate the interquartile range across validation episodes. (a) ATE RMSE, a pooled metric and therefore shown without an episode-level band. (b) normalized ATE bias, where zero denotes no bias. (c) CATE nPEHE, with the dashed line marking the constant-ATE baseline (nPEHE = 1). (d) CATE shape $R ^ { 2 }$ , with the dashed line marking the corresponding constant-ATE baseline $( R _ { \mathrm { s h a p e } } ^ { 2 } = 0 )$ . Lower values are preferred in panels (a) and (c), while panel (b) is centered at zero and higher values are preferred in panel (d). For readability, curves and shaded bands are lightly smoothed by taking medians within log-spaced training-step bins.

![](images/587e306f32cd9e8f634287328f3a8f03ee19ee365d24770d6480ed6cc45761cc.jpg)

![](images/b0d1e6409225dbf029482f8993f228e6cfe38d82f2404110d32475b7d38f5749.jpg)  
(d) credible-interval width

(c) coverage  
![](images/68e6efb508b2681dc7a24acab28391f06a5f180e29bd315f31bd9d3444d8f131.jpg)

![](images/1d96a98c37b9c5fd890fe62c00e9ab38ec7f39dc644d868f26b2a167a5cd76bb.jpg)  
Figure C.3: Evolution of distributional predictive performance during training on the fixed realistic-mechanism validation suite (Appendix C.4.1). (a) CRPS, for which lower values indicate better probabilistic predictions. (b) predictive standard deviation divided by RMSE, with the dashed line at one denoting calibrated predictive spread. (c) empirical 90% and 50% coverages, with the dashed lines marking the nominal coverage levels and the shaded regions showing the interquartile ranges across validation episodes. (d) widths of the central 90% and 50% credible intervals, pooled over validation rows. Panels (b) and (d) use logarithmic y-axes. For readability, curves are lightly smoothed by taking medians within log-spaced training-step bins.

We also report the ratio of predictive standard deviation to prediction error,

$$
\mathrm { S D / R M S E } = \sqrt { \frac { \frac { 1 } { M } \sum _ { i = 1 } ^ { n _ { q } } v _ { i } ^ { 2 } } { \frac { 1 } { n _ { q } } \sum _ { i = 1 } ^ { n _ { q } } \left( \mu _ { i } - Q _ { K , i } \right) ^ { 2 } - \frac { 1 } { M } \sum _ { i = 1 } ^ { n _ { q } } \varsigma _ { i } ^ { 2 } } } ,\tag{42}
$$

where $n _ { q }$ is the total number of validation query rows. The subtraction of $\varsigma _ { i } ^ { 2 }$ removes the Monte Carlo noise from the observed prediction error. This denominator equals the average predictive variance under calibration, in expectation. Values below one indicate overconfident predictions, whereas values above one indicate underconfidence.

Figure C.3 displays the evolution of these distributional metrics during training. CRPS decreases as the model learns. Both the ratio SD/RMSE and the empirical coverages converge to overconfident states.

## C.4.4 Posterior representation & smoothing

The histogram representation of the predictive posterior can produce visually spiky densities. Small changes in the allocation of probability across neighbouring bins may have little effect on the training objective while producing noticeably different density shapes. We therefore use the CDF as an intermediate representation for visualization. A Savitzky–Golay filter is applied to the raw CDF, and the corresponding density is obtained by numerical differentiation. As illustrated in Figure C.4, the smoothed CDF stays very close to the original CDF, but its implied density is easier to interpret. This postprocessing is applied only for visual inspection and does not alter the predictive distribution used to compute validation metrics. Future work may investigate regularization techniques during training to encourage smoother posterior estimates.

![](images/9e2220c5e2ceaa817b4a19679a53147649c86dcb543743c8d75ad74687c26e37.jpg)

![](images/a1e1d1c1db54af27d74b854a667edeb33e502320140247b9deb80c86c8bd787c.jpg)  
(d) PDF implied by smoothed CDF

![](images/09a2dce325d4d4d78a6be4b69dad33a97083f8ab3118e20a267806f9be518d7b.jpg)

![](images/ded3631599a141a16c5f86d7d33c81e45b8ad974d105cb46758ff59cb40c0eab.jpg)  
Figure C.4: Illustration of the spiky posterior-density representation for one selected query point from the validation suite. (a) Raw posterior density induced by the model’s histogram output. (b) Corresponding raw CDF. (c) CDF after Savitzky–Golay smoothing with a 50-point window and polynomial degree three. (d) Density obtained by differentiating the smoothed CDF. The smoothed CDF closely follows the raw CDF while yielding a more visually interpretable density. The dashed line marks the true CATE. Smoothing is used for visualization only; all validation metrics are computed from the original posterior representation.

## C.4.5 Posterior dynamics

Figure C.5 shows that posterior width is initially independent of context size: at the first checkpoints, all context sizes produce similarly wide predictions. Over the first thousands of training steps, the width stays independent of the context size, but becomes narrower. Later in training, context-dependent contraction becomes evident, with larger context sizes producing narrower posteriors while smaller contexts remain wider. At the final checkpoint, the mean predictive standard deviation decreases from approximately 0.54 at $n _ { \mathrm { c t x } } = 1 6 \mathrm { t o } 0 . 1 0 \mathrm { a t } n _ { \mathrm { c t x } } = 2 0 4 8 .$

Figure C.6 illustrates the evolution of the predictive posterior for a fixed query point. Early in training, the posterior is diffuse and weakly dependent on context size. As training progresses, it concentrates around the true CATE, with the larger context producing a noticeably sharper final posterior.

![](images/40604acf84c6dbab598c506fd0c50f69dbed598a86069474409f8850555424f4.jpg)  
Figure C.5: Posterior-width contraction during training on the fixed realistic-mechanism validation suite (Appendix C.4.1). Each cell reports the mean predictive posterior standard deviation averaged over all validation query rows, for a given training checkpoint and context size. Widths are expressed in standardized units and use the piecewise-uniform posterior variance. The shared colour scale is logarithmic.

![](images/c9dc63996f1b8058176dace13df18b347690767559208d73a9ba49785e5793ab.jpg)  
Figure C.6: Evolution of the predictive CATE posterior for one fixed query point from the realistic-mechanism validation suite (Appendix C.4.1). Curves show the density obtained from the posterior CDF at four training checkpoints, with training steps indicated by colour. The same query point is evaluated with (a) $n _ { \mathrm { c t x } } = 6 4$ and (b) $n _ { \mathrm { c t x } } = 1 0 2 4$ . The dashed line marks the true CATE. Densities are shown in standardized units. The CDFs are smoothed for visualization, following Appendix C.4.4.

## D Details on results

## D.1 Baseline methods

The baseline methods are either newly implemented or adjusted from existing codebases. All relevant code and configurations are available in our codebase.

Table D.1: Catalogue of CATE predictors selected in the main benchmark.
<table><tr><td>Predictor</td><td>Family</td><td></td><td>Fitted estimator and prediction rule</td><td>Details</td></tr><tr><td>S-learner (ridge)</td><td></td><td></td><td>Backdoor meta- S-learner with ridge regression.</td><td>Appendix D.1.1</td></tr><tr><td>S-learner (RF)</td><td>learner learner</td><td></td><td>Backdoor meta- S-learner with random forest.</td><td>Appendix D.1.1</td></tr><tr><td>S-learner (TabICL)</td><td>learner</td><td></td><td>Backdoor meta- S-learner with zero-shot TabICL.</td><td>Appendix D.1.1</td></tr><tr><td>S-learner (TabICL) + kNN marg.</td><td>learner</td><td></td><td>Backdoor meta- S-TabICL fit on (X, W, Z), then 16- Appendix D.1.1 neighbour proxy marginalisation.</td><td></td></tr><tr><td>S-learner (TabICL) + kernel marg.</td><td>Backdoor meta- learner</td><td></td><td>S-TabICL fit on (X, W, Z), then 16- Appendix D.1.1 neighbour kernel proxy marginalisation</td><td></td></tr><tr><td>T-learner (ridge)</td><td></td><td></td><td>(scale 1). Backdoor meta- T-learner with ridge regression.</td><td>Appendix D.1.1</td></tr><tr><td>T-learner (RF)</td><td>learner</td><td></td><td>Backdoor meta- T-learner with random forest.</td><td>Appendix D.1.1</td></tr><tr><td>T-learner (TabICL)</td><td>learner</td><td></td><td>Backdoor meta- T-learner with zero-shot TabICL.</td><td>Appendix D.1.1</td></tr><tr><td>T-learner (ridge) + kNN marg.</td><td>learner learner</td><td></td><td>Backdoor meta- T-OLS fit on (X, W, Z), then 16- Appendix D.1.1 neighbour proxy marginalisation.</td><td></td></tr><tr><td>T-learner (ridge) + kernel Backdoor meta- marg.</td><td>learner</td><td></td><td>T-OLS fit on (X, W, Z), then 16- Appendix D.1.1 neighbour kernel proxy marginalisation</td><td></td></tr><tr><td>X-learner (RF)</td><td></td><td></td><td>(scale 1). Backdoor meta- X-learner with random forest.</td><td>Appendix D.1.1</td></tr><tr><td>X-learner (TabICL)</td><td>learner</td><td></td><td>Backdoor meta- X-learner with zero-shot TabICL.</td><td>Appendix D.1.1</td></tr><tr><td>P-learner (ridge)</td><td>learner Proximal</td><td></td><td>Kernel-bridge P-learner with ridge Appendix D.1.2</td><td></td></tr><tr><td>P-learner (RF)</td><td>Proximal</td><td></td><td>final-stage CATE regression. Kernel-bridge P-learner with random- Appendix D.1.2</td><td></td></tr><tr><td>P-learner (TabICL)</td><td>Proximal</td><td></td><td>forest final-stage CATE regression. Kernel-bridge P-learner with TabICL Appendix D.1.2</td><td></td></tr><tr><td>KPV (ind. W)</td><td>Proximal</td><td></td><td>final-stage CATE regression. KPV bridge with W integrated over the Appendix D.1.5</td><td></td></tr><tr><td>KPV (cond. W)</td><td>Proximal</td><td></td><td>context empirical marginal. KPV bridge with locally conditional Appendix D.1.5</td><td></td></tr><tr><td>PMMR (ind. W)</td><td>Proximal</td><td></td><td>W | X = x integration. PMMR bridge with W integrated over Appendix D.1.4</td><td></td></tr><tr><td>PMMR (cond. W)</td><td>Proximal</td><td></td><td>the context empirical marginal. PMMR bridge with locally conditional Appendix D.1.4</td><td></td></tr><tr><td>NMMR-V (fixed; ind. W) Proximal</td><td></td><td></td><td>W | X = x integration. V-statistic NMMR, fixed RBF width, Appendix D.1.3</td><td></td></tr><tr><td>NMMR-V (fixed; cond. W) Proximal</td><td></td><td></td><td>with marginal W integration. V-statistic NMMR, fixed RBF width, Appendix D.1.3 with conditional W | X = x integra-</td><td></td></tr></table>

Table D.1: Catalogue of CATE predictors selected in the main benchmark (continued).
<table><tr><td>Predictor</td><td>Family</td><td>Fitted estimator and prediction rule</td><td>Details</td></tr><tr><td>NMMR-U (fixed; ind. W)</td><td>Proximal</td><td>U-statistic NMMR, fixed RBF width, with marginal W integration.</td><td>Appendix D.1.3</td></tr><tr><td>NMMR-U (fixed; cond. W)</td><td>Proximal</td><td>U-statistic NMMR, fixed RBF width, with conditional W | X = x integra-</td><td>Appendix D.1.3</td></tr><tr><td>NMMR-V (median; ind. Proximal W)</td><td></td><td>tion. V-statistic NMMR, median-heuristic RBF width, with marginal W integra-</td><td>Appendix D.1.3</td></tr><tr><td>NMMR-V (median; cond. Proximal W)</td><td></td><td>tion. V-statistic NMMR, median-heuristic Appendix D.1.3 RBF width, with conditional W | X =</td><td></td></tr><tr><td>NMMR-U (median; ind. Proximal W)</td><td></td><td>x integration. U-statistic NMMR, median-heuristic RBF width, with marginal W integra-</td><td>Appendix D.1.3</td></tr><tr><td>NMMR-U (median; cond. Proximal W)</td><td></td><td>tion. U-statistic NMMR, median-heuristic Appendix D.1.3 RBF width, with conditional W | X =</td><td></td></tr><tr><td>CausalFM</td><td>Causal ICL</td><td>x integration. Pre-trained causal foundation model, X Appendix D.1.1</td><td></td></tr><tr><td>Do-PFN</td><td>Causal ICL</td><td>only. Pre-trained causal foundation model, X Appendix D.1.1</td><td></td></tr><tr><td>CausalPFN</td><td>Causal ICL</td><td>only. Pre-trained causal foundation model, X Appendix D.1.1</td><td></td></tr><tr><td>CausalPFN + kNN marg.</td><td>Causal ICL</td><td>only. CausalPFN fit on (X, W, Z), then 16- Appendix D.1.1</td><td></td></tr><tr><td>CausalPFN + kernel marg.</td><td>Causal ICL</td><td>neighbour proxy marginalisation. CausalPFN fit on (X, W, Z), then 16- Appendix D.1.1 neighbour kernel proxy marginalisation</td><td></td></tr><tr><td>PROXIMALFM</td><td>Proximal founda- tion model</td><td>(scale 1). Amortized proximal transformer; query proxies are masked.</td><td>Section 3</td></tr></table>

## D.1.1 Backdoor meta-learners, in-context learners and proxy marginalization

We evaluate some methods estimating the CATE under the backdoor adjustment assumptions. First, we evaluat standard causal meta-learners from the backdoor adjustment framework (Kunzel et al., 2019), specifically the S-¨ learner, T-learner, and X-learner. We also implement some in-context learners: CausalPFN (Balazadeh Meresht et al., 2026), CausalFM (Ma et al., 2026) and Do-PFN (Robertson et al., 2026). In our proximal causal setting, the presence of the unobserved confounder U directly violates the conditional ignorability assumption these methods rely on $( Y ( a ) ~ \bot \bot ~ A ~ | ~ X )$ ). Consequently, all backdoor methods are misspecified. They serve as empirical baselines.

Because the causal estimand of interest is the covariate-conditional effect $Q ( x ) = \mathbb { E } [ Y ( 1 ) - Y ( 0 ) \mid X = x ]$ , yet observational data contain proxy variables (W, Z) that carry information about U, we evaluate backdoor methods under two distinct options:

1. Discard proxies (X-only): The learners are trained exclusively on baseline covariates X, discarding $( W , Z )$ entirely. This reflects the standard practice of ignoring unobserved confounding and assuming $Y ( a ) \perp \perp A \mid X$

2. Proxy conditioning with marginalization $( ( X , W , Z ) \to X ) { \mathrm { : } }$ To avoid disadvantaging backdoor methods by withholding proxy features, we also fit learners on the full feature set $( X , W , Z )$ . Because this yields a proxyconditional effect ${ \hat { \tau } } ( x , w , z )$ rather than Q(x), we post-process the predictions by integrating out $( W , Z )$ with respect to a local empirical estimate of $P ( W , Z \mid X = x )$

Note that even with proxy features and marginalization, Option 2 does not identify $Q ( x )$ : naive regression is insufficient under the proximal assumptions (Section 2).

Meta-learner formulations. We evaluate three meta-learning architectures (Kunzel et al., 2019):¨

• S-Learner: Fits a single regressor $\hat { \mu }$ on the pooled data predicting Y from features and treatment A. The effect is estimated as $\hat { \tau } _ { \mathrm { S } } ( x ) = \hat { \mu } ( x , 1 ) - \hat { \mu } ( x , 0 )$ . When proxies are included, it estimates $\hat { \tau } _ { \mathrm { S } } ( x , w , z ) =$ $\hat { \mu } ( x , 1 , w , z ) - \hat { \mu } ( x , 0 , w , z )$

• T-Learner: Fits separate regressors $\hat { \mu } _ { 0 }$ on the control cohort $( A = 0 )$ and $\hat { \mu } _ { 1 }$ on the treated cohort $( A = 1 )$ predicting $\hat { \tau } _ { \mathrm { T } } = \hat { \mu } _ { 1 } - \hat { \mu } _ { 0 }$ . This allows separate response surfaces for each arm, at the cost of not sharing information across arms.

• X-Learner: A four-stage crossover architecture designed for imbalanced treatments. After fitting base models $\hat { \mu } _ { 0 } , \hat { \mu } _ { 1 }$ , it imputes counterfactual pseudo-effects $D _ { 1 } = Y _ { 1 } { - } \hat { \mu } _ { 0 } ( X _ { 1 } )$ and $D _ { 0 } = \hat { \mu } _ { 1 } ( X _ { 0 } ) – Y _ { 0 }$ , trains second-stage models $\hat { \tau } _ { 1 }$ and $\hat { \tau } _ { 0 }$ to predict them, and combines them using an estimated propensity score eˆ:

$$
\hat { \tau } _ { \mathrm { X } } = \hat { e } \hat { \tau } _ { 0 } + \left( 1 - \hat { e } \right) \hat { \tau } _ { 1 } .\tag{43}
$$

In our implementation, eˆ is estimated via logistic regression.

Each meta-learner accepts three distinct base model classes:

• Ridge regression: Standardized linear model with penalty $\alpha \in \{ 0 . 1 , 1 , 1 0 , 1 0 0 , 1 0 0 0 \}$ selected via 4-fold cross-validation.

• Random forest: 100 trees with maximum depth $\in \{ 1 , 3 , 5 \}$ and minimum leaf size $\in \{ 1 , 5 , 1 0 \}$ chosen via 4-fold cross-validation.

• TabICLv2 (Qu et al., 2026): In-context tabular foundation model evaluated zero-shot.

In-context learners. We evaluate CausalPFN, CausalFM and Do-PFN in their pretrained, zero-shot form: each model uses the episode’s observed data as context, with no episode-specific parameter updates. CausalPFN and CausalFM directly return a CATE prediction for each query. Do-PFN instead predicts outcomes under each treatment arm, and we take their difference to form the CATE.

Proxy marginalization. When trained on (X, W, Z), a meta-learner predicts a proxy-conditional contrast ${ \hat { \tau } } ( x , w , z )$ Because the evaluation target is the covariate-conditional effect $Q ( x ) = \mathbb { E } [ Y ( 1 ) - Y ( 0 ) \mid X = x ]$ , we recover a function of x alone by integrating out the proxies against a local empirical estimate of $P ( W , Z \mid X = x )$ :

$$
\hat { \tau } ( x ) = \sum _ { i \in \mathcal { N } _ { k } ( x ) } q _ { i } ( x ) \hat { \tau } ( x , W _ { i } , Z _ { i } ) .\tag{44}
$$

Crucially, the integration uses the aligned context pairs $( W _ { i } , Z _ { i } )$ rather than a Cartesian product of marginals, preserving the empirical joint dependence between the two proxies. The neighborhood $\mathcal { N } _ { k } ( x )$ comprises the k context points closest to x under a feature-standardized Euclidean distance:

$$
d ( x , X _ { i } ) = \sqrt { \sum _ { d = 1 } ^ { d _ { X } } \frac { \left( x _ { d } - X _ { i , d } \right) ^ { 2 } } { \sigma _ { d } ^ { 2 } } } , \qquad \sigma _ { d } = \sqrt { 2 } \operatorname * { m e d i a n } _ { j , l } \left( \vert X _ { j , d } - X _ { l , d } \vert \right) .\tag{45}
$$

We evaluate two local weighting schemes for $q _ { i } ( x )$

$k { \mathbf { - N N } }$ weighting: Uniform weights over the k nearest context atoms, $q _ { i } ( x ) = 1 / k$

• Gaussian kernel weighting: Nadaraya-Watson exponential weights with bandwidth scale $s > 0 { : }$

$$
q _ { i } ( x ) = \frac { \exp \big ( - \big ( d ( x , X _ { i } ) ^ { 2 } - \operatorname* { m i n } _ { j \in \mathcal { N } _ { k } ( x ) } d ( x , X _ { j } ) ^ { 2 } \big ) / s \big ) } { \sum _ { l \in \mathcal { N } _ { k } ( x ) } \exp \big ( - \big ( d ( x , X _ { l } ) ^ { 2 } - \operatorname* { m i n } _ { j \in \mathcal { N } _ { k } ( x ) } d ( x , X _ { j } ) ^ { 2 } \big ) / s \big ) } .\tag{46}
$$

In our benchmarks, we use $k = 1 6$ and fixed kernel scale $s = 1 . 0$

## D.1.2 P-learner

We implement the P-learner of Sverdrup and Cui (2023). The cross-fitted score and final-stage regression follow their procedure. For bridge estimation, we use a Nystrom-approximation of the minimax RKHS formulation,¨ described below.

For each data point, we compute the doubly robust proximal score:

$$
\Gamma _ { i } = ( - 1 ) ^ { 1 - A _ { i } } q ( Z _ { i } , A _ { i } , X _ { i } ) \big ( Y _ { i } - h ( W _ { i } , A _ { i } , X _ { i } ) \big ) + h ( W _ { i } , 1 , X _ { i } ) - h ( W _ { i } , 0 , X _ { i } ) ,\tag{47}
$$

The outcome and treatment bridges are characterized, for $a \in \{ 0 , 1 \}$ , by the conditional moments

$$
\begin{array} { r } { { \mathbb E } [ { \bf 1 } \{ A = a \} \{ Y - h ( W , a , X ) \} \mid Z , X ] = 0 , } \\ { { \mathbb E } [ { \bf 1 } \{ A = a \} q ( Z , a , X ) - 1 \mid W , X ] = 0 . } \end{array}\tag{48}
$$

We obtain the CATE estimate by regressing the cross-fitted scores on X, $\hat { Q } ( x ) = \hat { \mathbb { E } } [ \hat { \Gamma } \mid X = x ]$ . The mean of the same scores gives the corresponding proximal ATE estimate.

Kernel bridge estimation. We estimate both bridges using the regularised minimax RKHS formulation of Dikkala et al. (2020), following Sverdrup and Cui (2023). The outcome bridge h is represented by RBF features $\Phi _ { H }$ of $( W , X )$ , and the treatment bridge q by $\Phi _ { Q }$ of $( Z , X )$ . Because of computational and memory constraints, we use the Nystrom method with¨ r landmark points sampled from the training fold. The kernel bridge estimators have closed-form solutions with penalties $\lambda _ { h }$ and $\lambda _ { q }$ on the two bridges.

Hyperparameter tuning. Several components of the method is subject to a choice of parameter. First of all, the CATE predictor ${ \hat { Q } } ( . )$ can be made using any general predictor. We use a ridge regularized linear regression, a random forest or TabICL (Qu et al., 2026). The parameters of the random forest and ridge regressor are chosen by cross-validation, TabICL does not require parameter tuning.

Then, regarding the bridge functions h and $q ,$ the regularizer $\lambda _ { h }$ and $\lambda _ { q }$ are chosen by cross-validation to minimize their moment loss. The Nystrom rank¨ r is fixed. The kernel bandwidth is chosen by the median heuristic, and not searched. Table D.2 summarizes the hyperparameters.

Score stabilisation. The estimated treatment-bridge weight can have large magnitude: we cap it at $q _ { \mathrm { m a x } }$ before forming (47).

## D.1.3 Neural moment matching regression

We adapted the implementation of the neural moment matching regression (NMMR) method of Kompa et al. $( 2 0 2 2 ) ^ { 2 }$ to our setting. Their code estimates one population-level quantity per dataset, ${ \mathbb E } [ Y \mid \operatorname { d o } ( A = a ) ]$ , evaluated on a grid of treatment values, which for a binary treatment is simply the two arms whose difference is the ATE. Our benchmark instead asks for the CATE $\hat { Q } ( x )$ at each query row of every episode.

The method. NMMR replaces the outcome bridge $h _ { 0 }$ of $( 2 )$ by a neural network $g _ { \phi } ( x , w , a ) \in \mathbb { R }$ and fits it directly, with no two-stage regression. This neural network is trained on an observational dataset, with simple gradient descent. It uses the following loss function, determined by an RBF kernel matrix on the conditioning variables $A , Z$ and X:

$$
\begin{array} { r } { r _ { i } = Y _ { i } - g _ { \phi } ( X _ { i } , W _ { i } , A _ { i } ) , \qquad K _ { i j } = \exp \Big ( - \frac { \| \kappa _ { i } - \kappa _ { j } \| ^ { 2 } } { 2 \ell ^ { 2 } } \Big ) , \quad \kappa _ { i } = \big ( A _ { i } , \ Z _ { i } / \hat { \sigma } _ { Z } , \ X _ { i } / \hat { \sigma } _ { X } \big ) , } \end{array}\tag{49}
$$

which determines the quadratic forms to minimize

$$
\hat { R } _ { V } = \frac { r ^ { \top } K r } { n ^ { 2 } } \qquad \mathrm { o r } \qquad \hat { R } _ { U } = \frac { r ^ { \top } \big ( K - \mathrm { d i a g } ( K ) \big ) r } { n ( n - 1 ) } ,\tag{50}
$$

Table D.2: P-learner settings. Bridge regularization parameters are selected separately within each cross-fitting fold using held-out moment loss. Final-stage hyperparameters are selected per episode using four-fold crossvalidated prediction MSE.
<table><tr><td>Component</td><td>Setting</td><td>Value</td></tr><tr><td>Cross-fitting</td><td>Folds (stratified on A)</td><td>4</td></tr><tr><td>Bridge class</td><td>Kernel</td><td>RBF, blockwise median heuristic</td></tr><tr><td rowspan="4">Bridge regularization</td><td>Nyström rank r</td><td>256</td></tr><tr><td> $\lambda _ { h } , \lambda _ { q }$ </td><td> $\{ 1 0 ^ { - 6 } , 1 0 ^ { - 5 } , 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 1 0 ^ { - 1 } \}$  sweep:</td></tr><tr><td>Selection criterion</td><td>Held-out proximal moment loss</td></tr><tr><td>Validation fraction (critic loss)</td><td>0.25</td></tr><tr><td>Proximal score</td><td> $\lambda _ { g }$  Weight clipping qmax</td><td> $1 0 ^ { - 3 }$  , fixed 20</td></tr><tr><td>Final-stage CV</td><td>Folds</td><td>4</td></tr><tr><td rowspan="2">Ridge final stage</td><td>Selection criterion</td><td>Mean squared error</td></tr><tr><td>Model</td><td>Standardized ridge regression</td></tr><tr><td rowspan="4">Random-forest final stage</td><td>Ridge penalty α</td><td>searched: {0.1, 1, 10, 100, 1000}</td></tr><tr><td>Number of trees</td><td>100, fixed</td></tr><tr><td>Maximum depth</td><td>searched: {1, 3, 5}</td></tr><tr><td>Minimum leaf size</td><td>searched: {1, 5, 10}</td></tr><tr><td rowspan="2">TabICL final stage</td><td>Maximum feature fraction</td><td>1, fixed</td></tr><tr><td>Model</td><td>Zero-shot prediction, no tuning</td></tr></table>

the V- and U-statistics. The U-statistic drops the diagonal: it is unbiased but noisier, while the V-statistic is biased but lower variance. We report both as separate entries, as in the original paper.

Deviations at prediction time. The reference code never produces a conditional ATE, even though the bridge it fits supports one. It returns a single scalar per arm, $\mathbb { E } _ { W , X } [ g _ { \phi } ( a , W , X ) ]$ , whose difference is an ATE. Recovering Q(x) of (3) instead means marginalising W at each query point; we consider two possibilities.

1. We assume $W \perp \perp X ( ^ { \circ } \mathrm { i n d . } \mathrm { W } ^ { \flat } )$ . The direct transposition of the reference code: average the fitted bridge over the empirical marginal of W in the context, $\begin{array} { r } { \hat { Q } ( x ) = \frac { 1 } { n } \sum _ { i } g _ { \phi } ( x , W _ { i } , 1 ) - g _ { \phi } ( x , W _ { i } , 0 ) } \end{array}$ . This is a valid strategy only if one assumes that W ⊥⊥ X.

2. We do not assume W ⊥⊥ X (“cond. W”). We approximate ${ \mathbb E } [ g _ { \phi } ( x , W , a ) \mid X = x ]$ by averaging over the marginal of W implied by the context, reweighted by a Gaussian kernel on X. This computes the proximal g-formula, without independence assumption, but at a cost of an extra smoothing bandwidth.

Implementation corrections & computational notes. We do the following changes in the code compared to original code. First, in the MLP used to model the bridge function, the original implementation applied a ReLU transformation on the last layer. This made the prediction of negative outcomes impossible, and was removed in our implementation. Second, for computational optimization, we avoid recomputing the kernel matrix at each iteration mini-batch, but compute the complete matrix once and sub-sample from it.

Kernel bandwidth. The original implementation uses a fixed bandwidth of l = 1.0 in the Equation (49). This has obvious drawbacks as the average distance between data points grow with the dimensionality of these points. In non-trivial dimensions, the kernel becomes the identity matrix, and all information from the proxy Z is lost: the bridge function becomes a direct estimator of Y in the V-statistic case, and very unstable in the U-statistic one. Therefore, we implement an alternative bandwidth which is defined by the median heuristic and does not suffer from this degeneration in higher dimension. Both alternatives are evaluated.

Hyperparameter selection. We follow the tuning from Kompa et al. (2022) and choose hyperparameters of the MLP based on the V- and U-statistics on a held-out validation split. The model is then refitted on the whole context using the final parameters. The parameter space searched is described in Table D.3. Note that we do not evaluate every configuration, but sample without repetitions 10 of them.

Table D.3: NMMR settings. Values marked searched are selected per episode by the procedure described below; the remainder are fixed across all episodes. The U- and V-statistic variants are reported as separate methods, so the loss is fixed within a run rather than searched.
<table><tr><td>Component</td><td>Setting</td><td>Value</td></tr><tr><td>Bridge class</td><td>MLP, ReLU, linear output</td><td>searched: {16, 32, 64}</td></tr><tr><td rowspan="3">Moment kernel</td><td>Hidden width Depth (hidden layers)</td><td>searched: {2, 4, 6}</td></tr><tr><td>RBF over  $( A , Z / \hat { \sigma } _ { Z } , X / \hat { \sigma } _ { X } )$ </td><td></td></tr><tr><td>Length scale l</td><td>fixed at 1.0 or median heuristic</td></tr><tr><td>Loss</td><td>U- or V-statistic (50)</td><td>one method each</td></tr><tr><td>Optimisation</td><td>Adam, learning rate</td><td>searched:  $\{ 1 0 ^ { - 3 } , 5 \cdot 1 0 ^ { - 3 } \}$ </td></tr><tr><td rowspan="4">Prediction</td><td>Weight decay (L2)</td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>Epochs</td><td>sweep: {25, 50, 75, 100}</td></tr><tr><td>Minibatch size</td><td>256</td></tr><tr><td>W strategy</td><td>ind. W / cond. W</td></tr><tr><td rowspan="3"></td><td>X smoothing length scale (Wcond)</td><td>median heuristic</td></tr><tr><td>Neighbours k (Wcond)</td><td>512</td></tr><tr><td>Validation fraction</td><td>0.2</td></tr><tr><td>Tuning</td><td>Search budget</td><td>10 of 18 configurations</td></tr></table>

## D.1.4 Proxy maximum moment restriction

We reimplemented the proxy maximum moment restriction (PMMR) method of Mastouri et al. (2021), instead of reusing the authors’ experimental code<sup>3</sup>. This allowed us to adjust the code for our usecases (prediction of CATE, with binary treatments). We preserved the PMMR’s RKHS bridge class and objective, but adapted treatment kernel to binary setting, and changed the prediction heads to the same two cases as described for NMMR (Appendix D.1.3).

The method. PMMR fits the same outcome bridge $h _ { 0 }$ as NMMR, but takes it in an RKHS rather than in a neural network class, which makes the minimiser available in closed form. We use column-wise Gaussian kernels for W, Z and X, and a binary indicator kernel for A. With their products kernels over $( A , X , W )$ and $( A , Z , X )$ , respectively, write the two matrices:

$$
{ \cal L } = K _ { A } \odot K _ { W } \odot K _ { X } , \qquad \Phi = K _ { A } \odot K _ { Z } \odot K _ { X } ,\tag{51}
$$

so that $\begin{array} { r } { h ( \cdot ) = \sum _ { i } \alpha _ { i } k _ { L } ( \cdot , ( A _ { i } , W _ { i } , X _ { i } ) ) } \end{array}$ ) and the V-statistic (50) of the residuals $Y - L \alpha$ under Φ is minimised by

$$
\alpha = \big ( L \Phi L + \lambda n ^ { 2 } L + \epsilon n ^ { 2 } I \big ) ^ { - 1 } L \Phi Y .\tag{52}
$$

Here λ penalises $\| h \| _ { \mathcal { H } _ { I } }$ and is the regulariser of the original method; ϵ is added for numerical stability.

Deviations at prediction time. As for NMMR, the reference code returns only ${ \mathbb E } [ Y \mid \operatorname { d o } ( A = a ) ]$ , which here is $\hat { y } ( a ) = \alpha ^ { \top } [ k _ { A } ( A , a ) \odot \overline { { w x } } ]$ with wx $\begin{array} { r } { \mathrm { ~  ~ \psi ~ } _ { \mathrm { ~ i ~ } } = \frac { 1 } { n } \sum _ { j } k _ { W } ( W _ { i } , W _ { j } ) k _ { X } ( X _ { i } , X _ { j } ) } \end{array}$ . We obtain $\hat { Q } ( x )$ through the same two strategies as NMMR (“ind. W”, “cond. $W ^ { \prime \prime } ) .$ , with the same caveats. Unlike NMMR, Wcond does not need extra hyperparameters: the weights used to approximate $W \mid X = x$ are determined by the $k _ { X } ( X _ { j } , x )$ from the model.

Hyperparameter selection. We select per episode on the V-statistic of a held-out fraction of the context, with the scoring kernel fixed at the median heuristic of the held-out rows, so that all candidates are compared on one metric rather than each on its own. The parameter space is described in Table D.4. It splits in two: the bandwidth multipliers enter the matrices L and Φ, while λ enters only the matrix assembled in (52). We therefore sample the former and enumerate the latter exhaustively for each draw, re-solving (52) at every λ off a single pair of L, Φ.

Computational notes. For binary A the kernel $k _ { A }$ is an indicator, so (52) is block diagonal by arm and we assemble and solve it one arm at a time; this is exact and reduces computational and memory costs.

Table D.4: PMMR settings. Bandwidth multipliers are sampled; the regularisers are swept, i.e. every combination is evaluated for each bandwidth draw. Both are selected per episode on the held-out V-statistic.
<table><tr><td>Component</td><td>Setting</td><td>Value</td></tr><tr><td rowspan="3">Kernels</td><td>Bandwidths baseline</td><td>per-column median heuristic</td></tr><tr><td>Kernel bandwidth multiplier (W,X)</td><td>searched: {0.8, 1, 1.2}</td></tr><tr><td>Kernel bandwidth multiplier (Z)</td><td>searched: {0.8, 1, 1.2}</td></tr><tr><td rowspan="2">Regularisation</td><td>λ (RKHS norm)</td><td> $s w e p t \colon \lbrace \bar { 1 0 ^ { - 5 } } , \bar { 1 0 ^ { - 4 } } , \bar { 1 0 ^ { - 3 } } , 1 0 ^ { - 2 } , 1 0 ^ { - 1 } \rbrace$ </td></tr><tr><td>Cholesky jitter</td><td> $1 0 ^ { - 8 }$ </td></tr><tr><td>Prediction</td><td>W strategy</td><td>“ind. W”, “cond.  $\mathbf { W } ^ { \prime \prime }$ </td></tr><tr><td rowspan="2">Tuning</td><td>Validation fraction</td><td>0.2</td></tr><tr><td>Search budget</td><td>all 9 bandwidth multipliers  $\times 5$  regularisers</td></tr></table>

## D.1.5 Kernel proxy variable

We reimplemented the kernel proxy variable (KPV) method of Mastouri et al. (2021), presented in the same paper as PMMR and sharing its released code. Again, the reimplementation allowed us to adjust the code for our usage: binary kernel and CATE predictions.

The method. KPV fits the same outcome bridge $h _ { 0 }$ as PMMR and in the same RKHS, but reaches it through two ridge regressions rather than one moment restriction. The theoretical formulation uses two samples, $D _ { 1 }$ of size $m _ { 1 }$ and $D _ { 2 }$ of size $m _ { 2 }$ . Following the released implementation, we reuse the complete context in both stages, so $D _ { 1 } = D _ { 2 }$ and $m _ { 1 } = m _ { 2 } = n _ { \mathrm { c o n t e x t } } .$ . The first stage regresses the feature map of $W$ on $( A , X , Z )$ , giving the conditional mean embedding $\hat { \mu } _ { W \mid a , x , z } ;$ writing $K = K _ { A } \odot K _ { X } \odot K _ { Z }$ for the product kernel over the conditioning set, its solution is

$$
\Gamma = \left( \mathcal { K } _ { D _ { 1 } D _ { 1 } } + \lambda _ { 1 } m _ { 1 } I \right) ^ { - 1 } \mathcal { K } _ { D _ { 1 } D _ { 2 } } .\tag{53}
$$

The second stage regresses Y on that embedding, which yields the bridge coefficients in closed form,

$$
\Sigma = \left( \Gamma ^ { \top } K _ { W } \Gamma \right) \odot K _ { A } \odot K _ { X } , \qquad \alpha _ { i j } = \Gamma _ { i j } \left[ ( \Sigma + \lambda _ { 2 } m _ { 2 } I ) ^ { - 1 } Y _ { D _ { 2 } } \right] _ { j } ,\tag{54}
$$

with $K _ { W }$ taken on $D _ { 1 }$ and $K _ { A } , K _ { X }$ on $D _ { 2 } .$ As done in PMMR, we use columnwise Gaussian kernels for $K _ { X } , K _ { W }$ and $K _ { Z }$ , and a binary indicator one for $K _ { A }$ . Unlike PMMR, the conditional distribution of W given the conditioning set is estimated explicitly, in the first stage, rather than left implicit in a moment condition.

Hyperparameter selection. We use the authors’ closed-form leave-one-out criterion for both regularisers, run inside each fit. On the other hand, we modify the hyperparameter tuning of the kernel bandwidth. As suggested by the authors, we initialize the search by the median heuristic. We then search around this initialization by minimizing the stage-2 leave-one-out criterion (instead of the stage-1 one, used in their implementation). We prefer this as it compares bandwidths using prediction error against the same outcome $Y ,$ , whereas stage-1 LOOCV changes both the input kernel and the W-kernel defining the loss, making scores across bandwidths less directly comparable. Table D.5 summarises the tuning parameters.

Deviations at prediction time. As for NMMR and PMMR, the reference returns only ${ \mathbb E } [ Y \mid \operatorname { d o } ( A = a ) ]$ , here $\begin{array} { r } { \hat { \beta } ( a ) = \frac { 1 } { n } \sum _ { i , j , k } \hat { \alpha } _ { i j } k _ { A } ( \tilde { a } _ { j } , a ) k _ { X } ( \tilde { x } _ { j } , x _ { k } ) k _ { W } ( w _ { i } , w _ { k } ) } \end{array}$ . We obtain $\hat { Q } ( x )$ through the same two strategies (“ind. $\mathrm { W } ^ { \prime \prime } , \mathrm {  } \mathrm { c o n d . } \mathrm { \ W ^ { \prime \prime } } )$ , with the same caveats. As in PMMR, the marginalisation is over the joint (X, W), whereas the released code averages the two grams separately, which agrees only if X ⊥⊥ W in sample. The reference estimates the population effect by averaging the fitted bridge over a separate test sample; we instead use the context rows, which also serve as the fitting sample.

Implementation corrections & computational notes. In the original implementation, the median bandwidth is recomputed at every kernel call, and prediction uses the default scale rather than using the training one. Therefore, training and prediction do not use the same fitted kernel. We instead estimate each bandwidth once from the context and reuse it consistently. The released driver also fits a separate input scaler to the test data; we reuse the transformation from the context. Finally, we leverage the binary aspect of the treatment kernel to solve the closed-form formulas arm by arm, reducing computational and memory costs without changing the estimator.

Table D.5: KPV settings. Both stages use all context rows. The regularizers $\lambda _ { 1 }$ and $\lambda _ { 2 }$ are selected within each fit by their closed-form leave-one-out criteria. The shared kernel-scale multiplier is selected per episode using the stage-2 LOOCV loss.
<table><tr><td>Component</td><td>Setting</td><td>Value</td></tr><tr><td>Data split</td><td>m1 (stage-1 rows)</td><td>Ncontext</td></tr><tr><td rowspan="3">Kernels</td><td>m2 (stage-2 rows)</td><td>Ncontext</td></tr><tr><td>Kernel family</td><td>Column-wise Gaussian RBF for X, W, Z; binary for A</td></tr><tr><td>Base bandwidths Shared multiplier c on  $\sigma _ { d } ^ { 2 }$ </td><td>Per-column median heuristic, estimated on the context Shared across X, W, Z; searched: {0.8, 1.0, 1.2}</td></tr><tr><td rowspan="4">Regularization Prediction Tuning</td><td> $\lambda _ { 1 } , \lambda _ { 2 }$ </td><td>searched by LOOCV; 10-point log grid in  $[ 1 0 ^ { - 5 } , 1 0 ]$ </td></tr><tr><td>W strategy</td><td>“ind.  $W ^ { \prime \prime }$  , &quot;cond.  $W ^ { \prime \prime }$ </td></tr><tr><td>Selection score</td><td>Stage-2 leave-one-out loss</td></tr><tr><td>Bandwidth candidates</td><td>3 per episode; no separate validation split</td></tr></table>

## D.2 Semi-synthetic benchmarks from real covariates

We construct proximal benchmarks from twelve numerical datasets in the INRIA-SODA tabular benchmark collection (Grinsztajn et al., 2022).<sup>4</sup> Real source rows supply the joint distribution of observed and hidden variables. We generate treatment, proxies, and outcomes with known causal mechanisms. Crossing linear/nonlinear mechanisms, weak/strong confounding, and weak/strong proxy information gives $1 2 \times 2 \times 2 \times 2 = 9 6$ configurations. Three seeded realizations per configuration give 288 realizations. Figure D.1(a) shows how the real covariate distribution enters the construction.

Sources and calibration. We select twelve datasets from the INRIA-SODA tabular benchmark collection with at least 20 features and 10 000 rows. For each source, we reserve a fixed set of rows for calibration and use the remaining rows to construct evaluation episodes. Calibration rows determine missing-value handling, clipping, and feature standardisation; the resulting transformations remain fixed across episodes. We exclude features that cannot be reliably calibrated.

We assign the retained source columns, which may include the dataset’s original prediction target, to observed covariates X and hidden source variables $U _ { \mathrm { s r c } }$ using a deterministic ordering. The two groups contain the same number of features; we use $d = 5$ for our experiments. Each realization samples context and query rows from the evaluation pool without replacement. Table D.6 reports the original sizes of the selected datasets.

Latent and observed mechanisms. For each row, we draw $\eta _ { U } \sim \mathcal { N } ( 0 , I _ { d _ { U } } )$ independently and set

$$
U = \sqrt { 0 . 7 5 } U _ { \mathrm { s r c } } + 0 . 5 \eta _ { U } .
$$

Because X and $U _ { \mathrm { s r c } }$ come from the same source row, U retains some of their dependence. The independent noise term adds variation in U even when X is fixed. Let $g$ be identity in the linear family and componentwise asinh in the nonlinear family. A separate calibration innovation stream determines

$$
m = \operatorname { m e a n } _ { \mathrm { c a l } } g ( U ) , \qquad C = \operatorname { C o v } _ { \mathrm { c a l } } g ( U ) , \qquad V = ( C + 0 . 0 5 I ) ^ { - 1 / 2 } \{ g ( U ) - m \} .
$$

The covariance uses the population denominator and the inverse square root is symmetric. We draw $\widetilde { c } \sim \mathcal { N } ( 0 , I _ { d _ { U } } )$ and rescale it to c so that $L = c ^ { \top } V$ has unit calibration standard deviation. Treatment and baseline outcome share this latent score.

![](images/7dc2f71191d901d8a1f59039df551a7e855d3d9397c758069778cc855f7bdfc7.jpg)  
Figure D.1: Construction of the generated proximal benchmarks. (a) A shared real source row supplies dependent observed covariates and hidden source features. Independent Gaussian innovation augments the hidden features. Both X and U enter the proxy measurements and the treatment and outcome mechanisms. (b) Independent Gaussian X and U define the controlled synthetic study. The joint proxy information parameter changes the measurement channels while preserving the causal variables within each dimension. The diagrams display the generated causal structure, with independent disturbances suppressed where indicated.

The linear observed basis contains the coordinates of X. The nonlinear basis applies one uniformly selected transformation from t, asinh(t), sin(t), and $t ^ { 2 }$ to each coordinate, then adds products for up to eight distinct coordinate pairs selected uniformly without replacement. After standardising these basis columns on calibration rows, let $K$ denote their number. For each role $r ~ \in ~ \{ a , y , \tau , w , z \}$ , we draw independent coefficients from $\mathcal { N } ( 0 , 1 / K )$ and standardise the resulting linear combinations on calibration rows to obtain $b _ { r } ( X )$ . The functions $b _ { a } , b _ { y } , b _ { \tau }$ are scalar and $b _ { w } , b _ { z }$ each have $d _ { U }$ coordinates. Standard deviations used to normalise the score, basis, and functions have a floor of 10<sup>−8</sup>. We set

$$
f _ { A } ( \boldsymbol { X } ) = a _ { 0 } + 0 . 5 b _ { a } ( \boldsymbol { X } ) , \quad f _ { Y } ( \boldsymbol { X } ) = b _ { y } ( \boldsymbol { X } ) , \quad f _ { W } ( \boldsymbol { X } ) = 0 . 5 b _ { w } ( \boldsymbol { X } ) , \quad f _ { Z } ( \boldsymbol { X } ) = 0 . 5 b _ { z } ( \boldsymbol { X } ) ,
$$

where $a _ { 0 } \sim \mathrm { U n i f } [ - 0 . 2 5 , 0 . 2 5 ]$ , and define $Q ( X ) = 0 . 5 + b _ { \tau } ( X )$ . The CATE function therefore has calibration mean 0.5 and standard deviation one.

Proxy channels. We obtain orthogonal $B _ { W } , B _ { Z }$ from independent Gaussian matrices using QR decomposition. We choose signs for a nonnegative diagonal in the triangular factor. For $J \in \{ W , Z \}$ , define

$$
\sigma _ { J , j } = \operatorname* { m a x } \{ \mathrm { s d } _ { \mathrm { c a l } } [ ( B _ { J } V ) _ { j } ] , 1 0 ^ { - 8 } \} , \qquad D _ { J } ( r ) = \sqrt { \frac { 1 - r } { r } } \dim \mathrm { g } ( \sigma _ { J , 1 } , \ldots , \sigma _ { J , d _ { U } } ) .
$$

With independent standard-normal errors, the proxy measurements are

$$
{ \cal W } = \phi \{ B _ { W } V + f _ { W } ( X ) + D _ { W } ( r ) \epsilon _ { W } \} , \qquad Z = \phi \{ B _ { Z } V + f _ { Z } ( X ) + D _ { Z } ( r ) \epsilon _ { Z } \} .
$$

The map ϕ is identity in the linear family and componentwise asinh in the nonlinear family. Weak and strong proxy settings use $r = 0 . 2$ and $r = 0 . 8$ . This parameter is the calibration latent-signal fraction in each measurement coordinate before the observed offset and observation map. Joint predictive information about U also depends on the source distribution and both proxy channels.

Table D.6: Original sizes of the twelve selected datasets. Feature counts exclude the original prediction target, although that column remains eligible for selection as a source variable in our construction.
<table><tr><td>Source</td><td>Rows</td><td>Features</td></tr><tr><td>pol</td><td>10 082</td><td>26</td></tr><tr><td>MiniBooNE</td><td>72 998</td><td>50</td></tr><tr><td>de  $\mathtt { f a u l t - o f - c r e d i t - c a r d - c l i e n t s }$ </td><td>13 272</td><td>20</td></tr><tr><td>Higgs</td><td>940160</td><td>24</td></tr><tr><td>jannis</td><td>57580</td><td>54</td></tr><tr><td>heloc</td><td>10 000</td><td>22</td></tr><tr><td>covertype</td><td>423 680</td><td>54</td></tr><tr><td>albert</td><td>58 252</td><td>31</td></tr><tr><td>road-safety</td><td>111 762</td><td>32</td></tr><tr><td>Ailerons</td><td>13 750</td><td>33</td></tr><tr><td>superconduct</td><td>21 263</td><td>79</td></tr><tr><td>Allstate_Claims_Severity</td><td>188 318</td><td>124</td></tr></table>

Treatment, outcomes, and bridge. Using independent treatment and outcome disturbances, we generate

$$
e ( X , U ) = \mathrm { c l i p } \{ \mathrm { s i g m o i d } ( f _ { A } ( X ) + \alpha L ) , 0 . 0 2 , 0 . 9 8 \} ,\tag{55}
$$

$$
A \sim { \mathrm { B e r n o u l l i } } \{ e ( X , U ) \} ,\tag{56}
$$

$$
\boldsymbol { Y } ( 0 ) = \boldsymbol { f } _ { \ u { Y } } ( \boldsymbol { X } ) + \boldsymbol { \gamma } \boldsymbol { L } + \boldsymbol { \epsilon } _ { \ u { Y } } , \qquad \boldsymbol { \epsilon } _ { \ u { Y } } \sim \mathcal { N } ( 0 , 1 ) ,\tag{57}
$$

$$
Y ( 1 ) = Y ( 0 ) + Q ( X ) , \qquad Y = Y ( 0 ) + A Q ( X ) .\tag{58}
$$

Weak confounding uses $( \alpha , \gamma ) = ( 0 . 3 5 , 0 . 5 )$ and strong confounding uses (1.25, 1.5). The potential outcomes share their disturbance, giving the exact heterogeneous effect

$$
Y ( 1 ) - Y ( 0 ) = Q ( X ) = \mathbb { E } [ Y ( 1 ) - Y ( 0 ) \mid X ] .
$$

An explicit outcome bridge is

$$
h ( w , a , x ) = f _ { Y } ( x ) + \gamma c ^ { \top } B _ { W } ^ { - 1 } \{ \phi ^ { - 1 } ( w ) - f _ { W } ( x ) \} + a Q ( x ) .
$$

Here $\phi ^ { - 1 }$ is identity or componentwise sinh. Independent, zero-mean measurement error gives $\mathbb { E } [ h ( W , a , X )$ | $X , U ] = \mathbb { E } [ Y ( a ) \mid X , U ]$ . The graph in Figure D.1(a) also gives $W \perp ( A , Z ) \mid ( X , U )$ and $Y \perp Z \mid ( A , X , U )$ establishing

$$
\operatorname { \mathbb { E } } [ Y \mid A , Z , X ] = \operatorname { \mathbb { E } } [ h ( W , A , X ) \mid A , Z , X ] .
$$

The nonlinear family consequently retains an explicit bridge through invertible measurement transformations.

Paired configurations. For each source and dimension, feature assignments are fixed across all configurations and realizations. Within a realization, all configurations use the same sampled rows; configurations in the same mechanism family also share fitted functions and random disturbances. Different realizations independently sample rows from the same evaluation pool and redraw the mechanisms and disturbances. Their sampled rows may overlap.

## D.3 Results on semi-synthetic datasets

![](images/3e863e8f1cd6b928eb303fe9edbec8fc2966fb36876bed47f95cb7becaf92118.jpg)  
Figure D.2: Learning curves and error decomposition for the nonlinear, high-confounding, low-proxy-reliability setting (36 episodes: 12 sources with three realizations each). (a) Median CATE nPEHE versus context size; the dotted line marks the constant-ATE benchmark (nPEHE = 1). ProximalFM-FT continues training the main checkpoint on the same prior and objective with contexts up to 5 000 rows, updating only part of the incontext predictor (Appendix C.3.1). (b) For query-row errors $e _ { i } = { \hat { \tau } } _ { i } - \tau _ { i }$ their mean e¯, and true CATE variance $s _ { \tau } ^ { 2 } = \mathrm { V a r } _ { i } ( \tau _ { i } )$ ), the episode-level decomposition is $\mathrm { n P E H E ^ { 2 } } = \bar { e } ^ { 2 } / s _ { \tau } ^ { 2 } + \mathrm { V a r } _ { i } ( e _ { i } ) / s _ { \tau } ^ { 2 }$ . Panel (b) plots the first term horizontally (squared bias in the query-average treatment effect) and the second vertically (heterogeneity error after removing that bias). Dotted contours show constant ${ \sqrt { x + y } } ;$ plotted coordinates are separate component medians, so they need not reproduce panel (a)’s median nPEHE. Arrows and larger markers indicate increasing context size (128–8 192). ProximalFM-FT moves mainly leftward: post-training improves the average effect estimate, with little improvement in CATE shape.

![](images/b707b5846868d1fefdbd9a2ebd83233cf5bd52ba8fdcf91a09b9dedacb65a8ca.jpg)

![](images/6ba0ca045a41921da1d09fd0de603e01f7a42c00a6a2da5bd20ba644c35879f5.jpg)

![](images/ac5c6bc869c03c3bddda9c5cac6c4fa9c46c1f3c6bbf8970d211c6dc6eca1769.jpg)

![](images/8a00d15c7216dc4660dc0df13c9fd920f58229bb3c324cb9679ea857178e58e8.jpg)

![](images/6f39689c2eed9f8527ae3f696b6b9a88f6a9806995a99cd8653c7fae663119c6.jpg)

![](images/16c094f0eba8411cbd834cefc8b61eb44e801d06ef7d8ace8e3cba09b87682b3.jpg)  
Figure D.3: Posterior calibration and contraction. The top row evaluates 400 datasets drawn from the pre-training prior; the bottom row evaluates 120 semi-synthetic datasets with nonlinear mechanisms, high confounding, and low proxy reliability. In every episode, the same 256 query points are evaluated as the context grows through nested prefixes from 128 to 8 192 observations. $^ { ( \mathbf { a } , \mathbf { d } ) }$ Coverage by context: each solid curve gives empirical coverage of a central CATE credible interval at one nominal level (1%, 5%, 10%, . . . , 95%, 99%); the faint dotted line of the same colour marks that level’s nominal coverage. $^ { ( \mathbf { b } , \mathbf { e } ) }$ Calibration: each curve fixes a context size, coloured from purple (128) to yellow (8 192), and plots empirical against nominal coverage. The dotted diagonal denotes perfect calibration; curves below it indicate undercoverage. $\mathbf { \Gamma } ( \mathbf { c , f } )$ Posterior contraction: the black curve is the mean predicted CATE posterior standard deviation across query rows. The grey band is the interquartile range of episode-level mean standard deviations, showing variation between datasets. These panels use logarithmic vertical axes. Coverage stays close to nominal on the prior, whereas the semi-synthetic intervals increasingly undercover as context grows, even though their predicted uncertainty contracts.

Table D.7: Exhaustive semisynthetic CATE nPEHE results at $n _ { \mathrm { c t x } } = 1 2 8$ . Each entry first averages the three replicates of each source dataset, then averages equally over the 12 sources; lower is better. The lowest mean in each column is bold. Other underlined methods are not significantly worse than that empirical best under onesided exact paired sign-flip tests across source-level means, with Holm correction within the column $( \alpha = 0 . 0 5 )$
<table><tr><td rowspan="3">Method</td><td colspan="4">Linear</td><td colspan="4">Nonlinear</td></tr><tr><td colspan="2">Low confounding</td><td colspan="2">High confounding</td><td colspan="2">Low confounding</td><td colspan="2">High confounding</td></tr><tr><td>Low proxy</td><td>High proxy</td><td>Low proxy</td><td>High proxy</td><td>Low proxy</td><td>High proxy</td><td>Low proxy</td><td>High proxy</td></tr><tr><td>S-learner (ridge)</td><td>1.027</td><td>1.027</td><td>1.458</td><td>1.458</td><td>1.059</td><td>1.059</td><td>1.459</td><td>1.459</td></tr><tr><td>S-learner (RF)</td><td>0.868</td><td>0.868</td><td>1.229</td><td>1.229</td><td>0.986</td><td>0.986</td><td>1.542</td><td>1.542</td></tr><tr><td>S-learner (TabICL)</td><td>0.782</td><td>0.782</td><td>1.336</td><td>1.336</td><td>0.937</td><td>0.937</td><td>1.457</td><td>1.457</td></tr><tr><td>S-learner (TabICL) + kNN marg.</td><td>0.885</td><td>0.895</td><td>1.070</td><td>0.921</td><td>0.973</td><td>0.982</td><td>1.259</td><td>0.986</td></tr><tr><td>S-learner (TabICL) + kernel marg.</td><td>0.888</td><td>0.888</td><td>1.064</td><td>0.914</td><td>0.966</td><td>0.966</td><td>1.245</td><td>0.966</td></tr><tr><td>T-learner (ridge)</td><td>0.603</td><td>0.603</td><td>1.466</td><td>1.466</td><td>1.122</td><td>1.122</td><td>1.837</td><td>1.837</td></tr><tr><td>T-learner (RF)</td><td>0.846</td><td>0.846</td><td>1.580</td><td>1.580</td><td>1.157</td><td>1.157</td><td>1.751</td><td>1.751</td></tr><tr><td>T-learner (TabICL)</td><td>0.865</td><td>0.865</td><td>1.637</td><td>1.637</td><td>1.232</td><td>1.232</td><td>1.810</td><td>1.810</td></tr><tr><td>T-learner (ridge) + kNN marg.</td><td>0.639</td><td>0.603</td><td>1.255</td><td>0.694</td><td>1.063</td><td>1.103</td><td>1.631</td><td>1.398</td></tr><tr><td>T-learner (ridge) + kernel marg.</td><td>0.665</td><td>0.610</td><td>1.276</td><td>0.704</td><td>1.064</td><td>1.074</td><td>1.634</td><td>1.395</td></tr><tr><td>X-learner (RF)</td><td>0.718</td><td>0.718</td><td>1.436</td><td>1.436</td><td>1.025</td><td>1.025</td><td>1.605</td><td>1.605</td></tr><tr><td>X-learner (TabICL)</td><td>0.737</td><td>0.737</td><td>1.509</td><td>1.509</td><td>1.065</td><td>1.065</td><td>1.669</td><td>1.669</td></tr><tr><td>CausalFM</td><td>0.738</td><td>0.738</td><td>1.306</td><td>1.306</td><td>1.048</td><td>1.048</td><td>1.566</td><td>1.566</td></tr><tr><td>Do-PFN</td><td>0.806</td><td>0.806</td><td>1.294</td><td>1.294</td><td>1.606</td><td>1.606</td><td>1.604</td><td>1.604</td></tr><tr><td>CausalPFN</td><td>0.641</td><td>0.641</td><td>1.408</td><td>1.408</td><td>1.055</td><td>1.055</td><td>1.687</td><td>1.687</td></tr><tr><td>CausalPFN + kNN marg</td><td>0.577</td><td>0.565</td><td>1.112</td><td>0.654</td><td>0.971</td><td>0.945</td><td>1.460</td><td>1.048</td></tr><tr><td>CausalPFN + kernel marg.</td><td>0.597</td><td>0.574</td><td>1.133</td><td>0.668</td><td>0.979</td><td>0.956</td><td>1.466</td><td>1.072</td></tr><tr><td>P-learner (RF)</td><td>0.731</td><td>0.708</td><td>1.267</td><td>0.916</td><td>0.993</td><td>0.980</td><td>1.514</td><td>1.222</td></tr><tr><td>P-learner (ridge)</td><td>0.571</td><td>0.581</td><td>1.181</td><td>0.771</td><td>1.001</td><td>0.995</td><td>1.511</td><td>1.221</td></tr><tr><td>P-learner (TabICL)</td><td>0.652</td><td>0.664</td><td>1.289</td><td>0.891</td><td>0.984</td><td>0.972</td><td>1.535</td><td>1.210</td></tr><tr><td>KPV (cond. W)</td><td>1.325</td><td>1.021</td><td>1.905</td><td>1.192</td><td>1.300</td><td>1.014</td><td>2.054</td><td>1.204</td></tr><tr><td>KPV (independent W)</td><td>0.955</td><td>0.953</td><td>1.395</td><td>0.970</td><td>1.040</td><td>1.003</td><td>1.551</td><td>1.028</td></tr><tr><td>PMMR (cond. W)</td><td>0.957</td><td>0.922</td><td>1.011</td><td>0.953</td><td>1.006</td><td>1.002</td><td>1.105</td><td>1.072</td></tr><tr><td>PMMR (independent W)</td><td>0.993</td><td>0.989</td><td>0.982</td><td>0.953</td><td>1.017</td><td>1.021</td><td>1.039</td><td>1.006</td></tr><tr><td>NMMR-V (fixed; cond. W)</td><td>0.954</td><td>0.910</td><td>1.090</td><td>0.917</td><td>1.012</td><td>1.013</td><td>1.290</td><td>1.006</td></tr><tr><td>NMMR-V (fixed; ind. W)</td><td>0.957</td><td>0.915</td><td>1.094</td><td>0.926</td><td>1.016</td><td>1.017</td><td>1.292</td><td>1.006</td></tr><tr><td>NMMR-U (fixed; cond. W)</td><td>4.98×103</td><td>430.450</td><td>1.13×10⁵</td><td>155.343</td><td>6.16×103</td><td>55.108</td><td>6.11×105</td><td>937.488</td></tr><tr><td>NMMR-U (fixed; ind. W)</td><td>4.90×103</td><td>408.021</td><td>1.14×105</td><td>148.985</td><td>5.94×103</td><td>55.162</td><td>5.97×105</td><td>1.36 ×10³</td></tr><tr><td>NMMR-V (median; cond. W)</td><td>1.060</td><td>1.051</td><td>1.045</td><td>0.983</td><td>1.077</td><td>1.072</td><td>1.115</td><td>1.028</td></tr><tr><td>NMMR-V (median; ind. W)</td><td>1.061</td><td>1.051</td><td>1.044</td><td>0.983</td><td>1.077</td><td>1.073</td><td>1.115</td><td>1.029</td></tr><tr><td>NMMR-U (median; cond. W)</td><td>6.500</td><td>1.479</td><td>5.725</td><td>1.768</td><td>51.236</td><td>1.439</td><td>6.794</td><td>1.341</td></tr><tr><td>NMMR-U (median; ind. W)</td><td>6.940</td><td>1.460</td><td>5.725</td><td>1.763</td><td>49.715</td><td>1.420</td><td>7.024</td><td>1.324</td></tr><tr><td>ProximalFM (ours)</td><td>0.621</td><td>0.608</td><td>1.058</td><td>0.733</td><td>0.953</td><td>0.947</td><td>1.329</td><td>0.968</td></tr></table>

Table D.8: Exhaustive semisynthetic CATE nPEHE results at $n _ { \mathrm { c t x } } = 5 1 2 .$ . Each entry first averages the three replicates of each source dataset, then averages equally over the 12 sources; lower is better. The lowest mean in each column is bold. Other underlined methods are not significantly worse than that empirical best under onesided exact paired sign-flip tests across source-level means, with Holm correction within the column (α = 0.05).
<table><tr><td rowspan="3">Method</td><td colspan="4">Linear</td><td colspan="4">Nonlinear</td></tr><tr><td colspan="2">Low confounding</td><td colspan="2">High confounding</td><td colspan="2">Low confounding</td><td colspan="2">High confounding</td></tr><tr><td>Low proxy</td><td>High proxy</td><td>Low proxy</td><td>High proxy</td><td>Low proxy</td><td>High proxy</td><td>Low proxy</td><td>High proxy</td></tr><tr><td>S-learner (ridge)</td><td>1.013</td><td>1.013</td><td>1.564</td><td>1.564</td><td>1.023</td><td>1.023</td><td>1.580</td><td>1.580</td></tr><tr><td>S-learner (RF)</td><td>0.669</td><td>0.669</td><td>1.423</td><td>1.423</td><td>0.849</td><td>0.849</td><td>1.546</td><td>1.546</td></tr><tr><td>S-learner (TabICL)</td><td>0.499</td><td>0.499</td><td>1.294</td><td>1.294</td><td>0.762</td><td>0.762</td><td>1.432</td><td>1.432</td></tr><tr><td>S-learner (TabICL) + kNN marg.</td><td>0.527</td><td>0.581</td><td>1.053</td><td>0.683</td><td>0.793</td><td>0.822</td><td>1.206</td><td>0.865</td></tr><tr><td>S-learner (TabICL) + kernel marg.</td><td>0.526</td><td>0.574</td><td>1.053</td><td>0.678</td><td>0.787</td><td>0.811</td><td>1.196</td><td>0.846</td></tr><tr><td>T-learner (ridge)</td><td>0.315</td><td>0.315</td><td>1.338</td><td>1.338</td><td>1.015</td><td>1.015</td><td>1.665</td><td>1.665</td></tr><tr><td>T-learner (RF)</td><td>0.690</td><td>0.690</td><td>1.506</td><td>1.506</td><td>0.921</td><td>0.921</td><td>1.605</td><td>1.605</td></tr><tr><td>T-learner (TabICL)</td><td>0.498</td><td>0.498</td><td>1.451</td><td>1.451</td><td>0.785</td><td>0.785</td><td>1.531</td><td>1.531</td></tr><tr><td>T-learner (ridge) + kNN marg.</td><td>0.302</td><td>0.282</td><td>1.041</td><td>0.408</td><td>0.978</td><td>0.926</td><td>1.441</td><td>1.008</td></tr><tr><td>T-learner (ridge) + kernel marg.</td><td>0.307</td><td>0.279</td><td>1.051</td><td>0.413</td><td>0.975</td><td>0.897</td><td>1.437</td><td>0.981</td></tr><tr><td>X-learner (RF)</td><td>0.530</td><td>0.530</td><td>1.381</td><td>1.381</td><td>0.814</td><td>0.814</td><td>1.508</td><td>1.508</td></tr><tr><td>X-learner (TabICL)</td><td>0.424</td><td>0.424</td><td>1.379</td><td>1.379</td><td>0.726</td><td>0.726</td><td>1.462</td><td>1.462</td></tr><tr><td>CausalFM</td><td>0.638</td><td>0.638</td><td>1.288</td><td>1.288</td><td>0.945</td><td>0.945</td><td>1.467</td><td>1.467</td></tr><tr><td>Do-PFN</td><td>0.609</td><td>0.609</td><td>1.362</td><td>1.362</td><td>1.043</td><td>1.043</td><td>1.516</td><td>1.516</td></tr><tr><td>CausalPFN</td><td>0.372</td><td>0.372</td><td>1.330</td><td>1.330</td><td>0.730</td><td>0.730</td><td>1.488</td><td>1.488</td></tr><tr><td>CausalPFN + kNN marg.</td><td>0.356</td><td>0.310</td><td>1.035</td><td>0.424</td><td>0.760</td><td>0.740</td><td>1.264</td><td>0.826</td></tr><tr><td>CausalPFN + kernel marg.</td><td>0.360</td><td>0.314</td><td>1.043</td><td>0.436</td><td>0.752</td><td>0.713</td><td>1.251</td><td>0.783</td></tr><tr><td>P-learner (RF)</td><td>0.606</td><td>0.569</td><td>1.273</td><td>0.741</td><td>0.830</td><td>0.808</td><td>1.406</td><td>0.896</td></tr><tr><td>P-learner (ridge)</td><td>0.410</td><td>0.379</td><td>1.164</td><td>0.567</td><td>0.922</td><td>0.910</td><td>1.443</td><td>0.991</td></tr><tr><td>P-learner (TabICL)</td><td>0.498</td><td>0.465</td><td>1.244</td><td>0.661</td><td>0.773</td><td>0.747</td><td>1.368</td><td>0.838</td></tr><tr><td>KPV (cond. W)</td><td>1.936</td><td>0.909</td><td>3.028</td><td>1.202</td><td>2.039</td><td>0.996</td><td>2.573</td><td>1.162</td></tr><tr><td>KPV (independent W)</td><td>1.222</td><td>0.844</td><td>2.172</td><td>0.898</td><td>1.228</td><td>0.903</td><td>2.013</td><td>0.956</td></tr><tr><td>PMMR (cond. W)</td><td>0.878</td><td>0.856</td><td>1.034</td><td>0.910</td><td>0.914</td><td>0.898</td><td>1.155</td><td>1.009</td></tr><tr><td>PMMR (independent W)</td><td>0.911</td><td>0.924</td><td>1.009</td><td>0.912</td><td>0.962</td><td>0.966</td><td>1.121</td><td>0.993</td></tr><tr><td>NMMR-V (fixed; cond. W)</td><td>0.727</td><td>0.695</td><td>1.187</td><td>0.781</td><td>0.883</td><td>0.892</td><td>1.368</td><td>0.973</td></tr><tr><td>NMMR-V (fixed; ind. W)</td><td>0.731</td><td>0.706</td><td>1.190</td><td>0.793</td><td>0.887</td><td>0.901</td><td>1.371</td><td>0.982</td></tr><tr><td>NMMR-U (fixed; cond. W)</td><td>14.122</td><td>0.929</td><td>3.879</td><td>0.955</td><td>2.371</td><td>0.998</td><td>33.761</td><td>1.009</td></tr><tr><td>NMMR-U (fixed; ind. W)</td><td>15.140</td><td>0.935</td><td>3.891</td><td>0.959</td><td>2.385</td><td>1.002</td><td>33.990</td><td>1.011</td></tr><tr><td>NMMR-V (median; cond. W)</td><td>1.044</td><td>1.036</td><td>1.015</td><td>0.977</td><td>1.040</td><td>1.045</td><td>1.109</td><td>0.997</td></tr><tr><td>NMMR-V (median; ind. W)</td><td>1.044</td><td>1.037</td><td>1.019</td><td>0.980</td><td>1.042</td><td>1.047</td><td>1.111</td><td>1.001</td></tr><tr><td>NMMR-U (median; cond. W)</td><td>1.079</td><td>1.081</td><td>1.100</td><td>1.023</td><td>1.072</td><td>1.067</td><td>1.121</td><td>1.014</td></tr><tr><td>NMMR-U (median; ind. W)</td><td>1.079</td><td>1.081</td><td>1.101</td><td>1.023</td><td>1.073</td><td>1.068</td><td>1.121</td><td>1.014</td></tr><tr><td>ProximalFM (ours)</td><td>0.392</td><td>0.370</td><td>0.953</td><td>0.468</td><td>0.780</td><td>0.758</td><td>1.151</td><td>0.789</td></tr></table>

Table D.9: Exhaustive semisynthetic CATE nPEHE results at $n _ { \mathrm { c t x } } = 1 , 0 2 4$ . Each entry first averages the three replicates of each source dataset, then averages equally over the 12 sources; lower is better. The lowest mean in each column is bold. Other underlined methods are not significantly worse than that empirical best under onesided exact paired sign-flip tests across source-level means, with Holm correction within the column (α = 0.05).
<table><tr><td rowspan="3">Method</td><td colspan="4">Linear</td><td colspan="4">Nonlinear</td></tr><tr><td colspan="2">Low confounding</td><td colspan="2">High confounding</td><td colspan="2">Low confounding</td><td colspan="2">High confounding</td></tr><tr><td>Low proxy</td><td>High proxy</td><td>Low proxy</td><td>High proxy</td><td>Low proxy</td><td>High proxy</td><td>Low proxy</td><td>High proxy</td></tr><tr><td>S-learner (ridge)</td><td>1.011</td><td>1.011</td><td>1.567</td><td>1.567</td><td>1.024</td><td>1.024</td><td>1.593</td><td>1.593</td></tr><tr><td>S-learner (RF)</td><td>0.568</td><td>0.568</td><td>1.384</td><td>1.384</td><td>0.793</td><td>0.793</td><td>1.497</td><td>1.497</td></tr><tr><td>S-learner (TabICL)</td><td>0.379</td><td>0.379</td><td>1.256</td><td>1.256</td><td>0.653</td><td>0.653</td><td>1.393</td><td>1.393</td></tr><tr><td>S-learner (TabICL) + kNN marg.</td><td>0.393</td><td>0.426</td><td>0.994</td><td>0.552</td><td>0.694</td><td>0.732</td><td>1.169</td><td>0.783</td></tr><tr><td>S-learner (TabICL) + kernel marg.</td><td>0.396</td><td>0.425</td><td>0.995</td><td>0.550</td><td>0.686</td><td>0.715</td><td>1.159</td><td>0.765</td></tr><tr><td>T-learner (ridge)</td><td>0.242</td><td>0.242</td><td>1.284</td><td>1.284</td><td>0.998</td><td>0.998</td><td>1.627</td><td>1.627</td></tr><tr><td>T-learner (RF)</td><td>0.561</td><td>0.561</td><td>1.413</td><td>1.413</td><td>0.786</td><td>0.786</td><td>1.502</td><td>1.502</td></tr><tr><td>T-learner (TabICL)</td><td>0.382</td><td>0.382</td><td>1.357</td><td>1.357</td><td>0.670</td><td>0.670</td><td>1.425</td><td>1.425</td></tr><tr><td>T-learner (ridge) + kNN marg.</td><td>0.220</td><td>0.194</td><td>0.990</td><td>0.312</td><td>0.967</td><td>0.874</td><td>1.390</td><td>0.934</td></tr><tr><td>T-learner (ridge) + kernel marg.</td><td>0.225</td><td>0.197</td><td>0.992</td><td>0.318</td><td>0.962</td><td>0.855</td><td>1.386</td><td>0.913</td></tr><tr><td>X-learner (RF)</td><td>0.453</td><td>0.453</td><td>1.326</td><td>1.326</td><td>0.734</td><td>0.734</td><td>1.459</td><td>1.459</td></tr><tr><td>X-learner (TabICL)</td><td>0.348</td><td>0.348</td><td>1.303</td><td>1.303</td><td>0.609</td><td>0.609</td><td>1.396</td><td>1.396</td></tr><tr><td>CausalFM</td><td>0.616</td><td>0.616</td><td>1.272</td><td>1.272</td><td>0.927</td><td>0.927</td><td>1.453</td><td>1.453</td></tr><tr><td>Do-PFN</td><td>0.544</td><td>0.544</td><td>1.314</td><td>1.314</td><td>0.993</td><td>0.993</td><td>1.537</td><td>1.537</td></tr><tr><td>CausalPFN</td><td>0.297</td><td>0.297</td><td>1.277</td><td>1.277</td><td>0.629</td><td>0.629</td><td>1.412</td><td>1.412</td></tr><tr><td>CausalPFN + kNN marg.</td><td>0.273</td><td>0.224</td><td>0.977</td><td>0.333</td><td>0.646</td><td>0.621</td><td>1.175</td><td>0.675</td></tr><tr><td>CausalPFN + kernel marg.</td><td>0.275</td><td>0.225</td><td>0.979</td><td>0.338</td><td>0.637</td><td>0.599</td><td>1.168</td><td>0.643</td></tr><tr><td>P-learner (RF)</td><td>0.515</td><td>0.482</td><td>1.147</td><td>0.572</td><td>0.769</td><td>0.727</td><td>1.280</td><td>0.761</td></tr><tr><td>P-learner (ridge)</td><td>0.322</td><td>0.290</td><td>1.033</td><td>0.372</td><td>0.891</td><td>0.879</td><td>1.348</td><td>0.893</td></tr><tr><td>P-learner (TabICL)</td><td>0.416</td><td>0.387</td><td>1.094</td><td>0.481</td><td>0.664</td><td>0.636</td><td>1.218</td><td>0.669</td></tr><tr><td>KPV (cond. W)</td><td>1.499</td><td>0.926</td><td>2.598</td><td>1.208</td><td>1.433</td><td>0.825</td><td>2.208</td><td>0.960</td></tr><tr><td>KPV (independent W)</td><td>0.993</td><td>0.775</td><td>1.944</td><td>0.840</td><td>1.006</td><td>0.858</td><td>1.829</td><td>0.896</td></tr><tr><td>PMMR (cond. W)</td><td>0.878</td><td>0.846</td><td>1.062</td><td>0.903</td><td>0.918</td><td>0.897</td><td>1.188</td><td>0.995</td></tr><tr><td>PMMR (independent W)</td><td>0.903</td><td>0.910</td><td>1.033</td><td>0.905</td><td>0.956</td><td>0.961</td><td>1.156</td><td>0.991</td></tr><tr><td>NMMR-V (fixed; cond. W)</td><td>0.604</td><td>0.580</td><td>1.132</td><td>0.720</td><td>0.834</td><td>0.841</td><td>1.370</td><td>0.925</td></tr><tr><td>NMMR-V (fixed; ind. W)</td><td>0.609</td><td>0.602</td><td>1.135</td><td>0.745</td><td>0.840</td><td>0.859</td><td>1.375</td><td>0.937</td></tr><tr><td>NMMR-U (fixed; cond. W)</td><td>0.919</td><td>0.894</td><td>1.071</td><td>0.905</td><td>0.992</td><td>0.984</td><td>1.178</td><td>0.990</td></tr><tr><td>NMMR-U (fixed; ind. W)</td><td>0.920</td><td>0.903</td><td>1.076</td><td>0.920</td><td>0.994</td><td>0.989</td><td>1.180</td><td>0.995</td></tr><tr><td>NMMR-V (median; cond. W)</td><td>1.006</td><td>1.005</td><td>1.055</td><td>0.951</td><td>1.038</td><td>1.022</td><td>1.175</td><td>1.003</td></tr><tr><td>NMMR-V (median; ind. W)</td><td>1.006</td><td>1.009</td><td>1.058</td><td>0.956</td><td>1.041</td><td>1.027</td><td>1.178</td><td>1.006</td></tr><tr><td>NMMR-U (median; cond. W)</td><td>1.063</td><td>1.049</td><td>1.124</td><td>1.002</td><td>1.060</td><td>1.050</td><td>1.129</td><td>1.011</td></tr><tr><td>NMMR-U (median; ind. W)</td><td>1.063</td><td>1.050</td><td>1.122</td><td>1.007</td><td>1.060</td><td>1.051</td><td>1.132</td><td>1.010</td></tr><tr><td>ProximalFM (ours)</td><td>0.300</td><td>0.279</td><td>0.884</td><td>0.365</td><td>0.656</td><td>0.636</td><td>1.061</td><td>0.688</td></tr></table>

Table D.10: Exhaustive semisynthetic CATE nPEHE results at $n _ { \mathrm { c t x } } = 2 , 0 4 8$ . Each entry first averages the three replicates of each source dataset, then averages equally over the 12 sources; lower is better. The lowest mean in each column is bold. Other underlined methods are not significantly worse than that empirical best under onesided exact paired sign-flip tests across source-level means, with Holm correction within the column (α = 0.05).
<table><tr><td rowspan="3">Method</td><td colspan="4">Linear</td><td colspan="4">Nonlinear</td></tr><tr><td colspan="2">Low confounding</td><td colspan="2">High confounding</td><td colspan="2">Low confounding</td><td colspan="2">High confounding</td></tr><tr><td>Low proxy</td><td>High proxy</td><td>Low proxy</td><td>High proxy</td><td>Low proxy</td><td>High proxy</td><td>Low proxy</td><td>High proxy</td></tr><tr><td>S-learner (ridge)</td><td>1.011</td><td>1.011</td><td>1.560</td><td>1.560</td><td>1.022</td><td>1.022</td><td>1.606</td><td>1.606</td></tr><tr><td>S-learner (RF)</td><td>0.513</td><td>0.513</td><td>1.356</td><td>1.356</td><td>0.740</td><td>0.740</td><td>1.476</td><td>1.476</td></tr><tr><td>S-learner (TabICL)</td><td>0.310</td><td>0.310</td><td>1.239</td><td>1.239</td><td>0.564</td><td>0.564</td><td>1.371</td><td>1.371</td></tr><tr><td>S-learner (TabICL) + kNN marg.</td><td>0.291</td><td>0.298</td><td>0.953</td><td>0.441</td><td>0.580</td><td>0.624</td><td>1.111</td><td>0.695</td></tr><tr><td>S-learner (TabICL) + kernel marg.</td><td>0.293</td><td>0.297</td><td>0.953</td><td>0.437</td><td>0.585</td><td>0.608</td><td>1.111</td><td>0.679</td></tr><tr><td>T-learner (ridge)</td><td>0.205</td><td>0.205</td><td>1.264</td><td>1.264</td><td>1.112</td><td>1.112</td><td>1.754</td><td>1.754</td></tr><tr><td>T-learner (RF)</td><td>0.482</td><td>0.482</td><td>1.363</td><td>1.363</td><td>0.751</td><td>0.751</td><td>1.482</td><td>1.482</td></tr><tr><td>T-learner (TabICL)</td><td>0.306</td><td>0.306</td><td>1.303</td><td>1.303</td><td>0.604</td><td>0.604</td><td>1.420</td><td>1.420</td></tr><tr><td>T-learner (ridge) + kNN marg.</td><td>0.179</td><td>0.132</td><td>0.960</td><td>0.271</td><td>1.052</td><td>0.936</td><td>1.491</td><td>0.984</td></tr><tr><td>T-learner (ridge) + kernel marg.</td><td>0.181</td><td>0.134</td><td>0.962</td><td>0.272</td><td>1.046</td><td>0.913</td><td>1.486</td><td>0.961</td></tr><tr><td>X-learner (RF)</td><td>0.382</td><td>0.382</td><td>1.286</td><td>1.286</td><td>0.678</td><td>0.678</td><td>1.429</td><td>1.429</td></tr><tr><td>X-learner (TabICL)</td><td>0.299</td><td>0.299</td><td>1.262</td><td>1.262</td><td>0.504</td><td>0.504</td><td>1.352</td><td>1.352</td></tr><tr><td>CausalFM</td><td>0.600</td><td>0.600</td><td>1.262</td><td>1.262</td><td>0.917</td><td>0.917</td><td>1.460</td><td>1.460</td></tr><tr><td>Do-PFN</td><td>0.454</td><td>0.454</td><td>1.279</td><td>1.279</td><td>0.961</td><td>0.961</td><td>1.547</td><td>1.547</td></tr><tr><td>CausalPFN</td><td>0.246</td><td>0.246</td><td>1.255</td><td>1.255</td><td>0.545</td><td>0.545</td><td>1.372</td><td>1.372</td></tr><tr><td>CausalPFN + kNN marg.</td><td>0.217</td><td>0.167</td><td>0.946</td><td>0.285</td><td>0.571</td><td>0.540</td><td>1.127</td><td>0.595</td></tr><tr><td>CausalPFN + kernel marg.</td><td>0.218</td><td>0.168</td><td>0.948</td><td>0.288</td><td>0.567</td><td>0.521</td><td>1.121</td><td>0.571</td></tr><tr><td>P-learner (RF)</td><td>0.420</td><td>0.401</td><td>0.985</td><td>0.493</td><td>0.675</td><td>0.642</td><td>1.112</td><td>0.675</td></tr><tr><td>P-learner (ridge)</td><td>0.237</td><td>0.203</td><td>0.882</td><td>0.311</td><td>0.870</td><td>0.859</td><td>1.219</td><td>0.873</td></tr><tr><td>P-learner (TabICL)</td><td>0.334</td><td>0.308</td><td>0.939</td><td>0.420</td><td>0.575</td><td>0.553</td><td>1.037</td><td>0.589</td></tr><tr><td>KPV (cond. W)</td><td>1.745</td><td>0.715</td><td>2.778</td><td>0.934</td><td>1.752</td><td>0.781</td><td>2.591</td><td>0.948</td></tr><tr><td>KPV (independent W)</td><td>1.163</td><td>0.707</td><td>2.081</td><td>0.764</td><td>1.110</td><td>0.797</td><td>1.982</td><td>0.855</td></tr><tr><td>PMMR (cond. W)</td><td>0.881</td><td>0.843</td><td>1.084</td><td>0.905</td><td>0.924</td><td>0.892</td><td>1.217</td><td>1.005</td></tr><tr><td>PMMR (independent W)</td><td>0.897</td><td>0.905</td><td>1.050</td><td>0.902</td><td>0.954</td><td>0.957</td><td>1.182</td><td>0.994</td></tr><tr><td>NMMR-V (fixed; cond. W)</td><td>0.527</td><td>0.535</td><td>1.104</td><td>0.663</td><td>0.803</td><td>0.805</td><td>1.369</td><td>0.901</td></tr><tr><td>NMMR-V (fixed; ind. W)</td><td>0.531</td><td>0.557</td><td>1.109</td><td>0.694</td><td>0.807</td><td>0.818</td><td>1.372</td><td>0.910</td></tr><tr><td>NMMR-U (fixed; cond. W)</td><td>0.893</td><td>0.843</td><td>1.061</td><td>0.881</td><td>0.991</td><td>0.970</td><td>1.173</td><td>0.978</td></tr><tr><td>NMMR-U (fixed; ind. W)</td><td>0.897</td><td>0.858</td><td>1.063</td><td>0.902</td><td>0.992</td><td>0.979</td><td>1.175</td><td>0.984</td></tr><tr><td>NMMR-V (median; cond. W)</td><td>0.964</td><td>0.966</td><td>1.123</td><td>0.939</td><td>1.008</td><td>1.007</td><td>1.268</td><td>0.990</td></tr><tr><td>NMMR-V (median; ind. W)</td><td>0.969</td><td>0.972</td><td>1.127</td><td>0.950</td><td>1.008</td><td>1.009</td><td>1.271</td><td>0.997</td></tr><tr><td>NMMR-U (median; cond. W)</td><td>1.035</td><td>1.032</td><td>1.127</td><td>1.002</td><td>1.054</td><td>1.047</td><td>1.232</td><td>1.005</td></tr><tr><td>NMMR-U (median; ind. W)</td><td>1.035</td><td>1.032</td><td>1.134</td><td>1.003</td><td>1.054</td><td>1.048</td><td>1.228</td><td>1.006</td></tr><tr><td>ProximalFM (ours)</td><td>0.241</td><td>0.217</td><td>0.834</td><td>0.302</td><td>0.572</td><td>0.546</td><td>1.029</td><td>0.620</td></tr></table>

Table D.11: Extended-context semisynthetic CATE nPEHE results for nonlinear mechanisms with high confounding and low proxy reliability. Each entry first averages the 3 replicates of each source dataset, then averages equally over the 12 sources; lower is better. The lowest mean in each column is bold. Other underlined methods are not significantly worse than that empirical best under one-sided exact paired sign-flip tests across source-level means, with Holm correction within the column $( \alpha = 0 . 0 5 )$
<table><tr><td rowspan="2">Method</td><td colspan="7">Context size  $n _ { \mathrm { c t x } }$ </td></tr><tr><td>128</td><td>256</td><td>512</td><td>1,024</td><td>2,048</td><td>4,096</td><td>8,192</td></tr><tr><td>S-learner (ridge)</td><td>1.459</td><td>1.499</td><td>1.580</td><td>1.593</td><td>1.606</td><td>1.597</td><td>1.608</td></tr><tr><td>S-learner (RF)</td><td>1.542</td><td>1.536</td><td>1.546</td><td>1.497</td><td>1.476</td><td>1.602</td><td>1.473</td></tr><tr><td>S-learner (TabICL)</td><td>1.457</td><td>1.449</td><td>1.432</td><td>1.393</td><td>1.371</td><td>1.340</td><td>1.317</td></tr><tr><td>S-learner (TabICL) + kNN marg.</td><td>1.259</td><td>1.235</td><td>1.206</td><td>1.169</td><td>1.111</td><td>1.065</td><td>1.020</td></tr><tr><td>S-learner (TabICL) + kernel marg.</td><td>1.245</td><td>1.231</td><td>1.196</td><td>1.159</td><td>1.111</td><td>1.064</td><td>1.016</td></tr><tr><td>T-learner (ridge)</td><td>1.837</td><td>1.699</td><td>1.665</td><td>1.627</td><td>1.754</td><td>1.711</td><td>1.647</td></tr><tr><td>X-learner (RF)</td><td>1.605</td><td>1.559</td><td>1.508</td><td>1.459</td><td>1.429</td><td>1.504</td><td>1.363</td></tr><tr><td>X-learner (TabICL)</td><td>1.669</td><td>1.493</td><td>1.462</td><td>1.396</td><td>1.352</td><td>1.330</td><td>1.307</td></tr><tr><td>CausalPFN</td><td>1.687</td><td>1.541</td><td>1.488</td><td>1.412</td><td>1.372</td><td>1.371</td><td>1.376</td></tr><tr><td>CausalPFN + kNN marg.</td><td>1.460</td><td>1.338</td><td>1.264</td><td>1.175</td><td>1.127</td><td>1.121</td><td>1.117</td></tr><tr><td>CausalPFN + kernel marg.</td><td>1.466</td><td>1.349</td><td>1.251</td><td>1.168</td><td>1.121</td><td>1.140</td><td>1.127</td></tr><tr><td>P-learner (RF)</td><td>1.514</td><td>1.485</td><td>1.406</td><td>1.280</td><td>1.112</td><td>0.964</td><td>0.732</td></tr><tr><td>P-learner (ridge)</td><td>1.511</td><td>1.489</td><td>1.443</td><td>1.348</td><td>1.219</td><td>1.110</td><td>0.952</td></tr><tr><td>P-learner (TabICL)</td><td>1.535</td><td>1.467</td><td>1.368</td><td>1.218</td><td>1.037</td><td>0.847</td><td>0.660</td></tr><tr><td>NMMR-V (fixed; cond. W)</td><td>1.290</td><td>1.363</td><td>1.368</td><td>1.370</td><td>1.369</td><td>1.336</td><td>1.337</td></tr><tr><td>NMMR-V (fixed; ind. W)</td><td>1.292</td><td>1.366</td><td>1.371</td><td>1.375</td><td>1.372</td><td>1.338</td><td>1.339</td></tr><tr><td>NMMR-U (fixed; cond. W)</td><td> $6 . 1 1 \times 1 0 ^ { 5 }$ </td><td> $4 . 2 5 \times 1 0 ^ { 3 }$ </td><td>33.761</td><td>1.178</td><td>1.173</td><td>1.144</td><td>1.141</td></tr><tr><td>NMMR-U (fixed; ind. W)</td><td> $5 . 9 7 \times 1 0 ^ { 5 }$ </td><td> $4 . 2 9 \times 1 0 ^ { 3 }$ </td><td>33.990</td><td>1.180</td><td>1.175</td><td>1.149</td><td>1.144</td></tr><tr><td>NMMR-V (median; cond. W)</td><td>1.115</td><td>1.191</td><td>1.109</td><td>1.175</td><td>1.268</td><td>1.334</td><td>1.391</td></tr><tr><td>NMMR-V (median; ind. W)</td><td>1.115</td><td>1.188</td><td>1.111</td><td>1.178</td><td>1.271</td><td>1.335</td><td>1.396</td></tr><tr><td>NMMR-U (median; cond. W)</td><td>6.794</td><td>1.318</td><td>1.121</td><td>1.129</td><td>1.232</td><td>1.247</td><td>1.231</td></tr><tr><td>NMMR-U (median; ind. W)</td><td>7.024</td><td>1.311</td><td>1.121</td><td>1.132</td><td>1.228</td><td>1.242</td><td>1.230</td></tr><tr><td>ProximalFM (ours)</td><td>1.329</td><td>1.210</td><td>1.151</td><td>1.061</td><td>1.029</td><td>0.988</td><td>0.964</td></tr><tr><td>ProximalFM-FT (ours)</td><td>1.333</td><td>1.216</td><td>1.145</td><td>1.049</td><td>1.007</td><td>0.960</td><td>0.930</td></tr></table>

## D.3.1 Speed

We measured runtime in a separate run of the 36 semi-synthetic episodes with nonlinear mechanisms, strong confounding, and weak proxy information. Episodes were executed sequentially in one process to avoid contention between simultaneous benchmark jobs; methods retained the parallelism specified for their own hyperparameter searches. At each context size, the recorded wall-clock time covers estimator construction, fitting, and prediction on the 256 query rows. The filled points in Figure 5 show the median of these per-episode times at $n _ { \mathrm { c t x } } = 2 0 4 8 ;$ hollow points subtract the measured hyperparameter-search time. Accuracy ranks come independently from the complete semi-synthetic evaluation matrix, using paired episode-level CATE nPEHE ranks.

## D.4 Synthetic study of proxy informativeness

We vary proxy information within a fixed causal mechanism to examine when proximal estimation offers an advantage over regression adjustment using proxies. For each $d \in \{ 1 , 3 , 5 \}$ , we generate 20 information levels equally spaced from 0.01 to 0.99.

For each of the resulting 60 datasets, we evaluate methods using 128, 512, or 2 048 context rows and the same 256 held-out query rows. Each block $X , U , W , Z$ has dimension d. Figure D.1(b) shows the causal structure and proxy-information control.

Causal mechanism and heterogeneity. For every observation, X, U, ϵ , ϵ are mutually independent $\mathcal { N } ( 0 , I _ { d } )$ vectors. The independent outcome disturbance is $\epsilon _ { Y } \sim \mathcal { N } ( 0 , 1 )$ . Independent uniform draws determine the Bernoulli treatments. Define

$$
S _ { X } = d ^ { - 1 / 2 } \sum _ { j = 1 } ^ { d } X _ { j } , \qquad S _ { U } = d ^ { - 1 / 2 } \sum _ { j = 1 } ^ { d } U _ { j } .
$$

We generate

$$
p ( X , U ) = \mathrm { c l i p } \{ \mathrm { s i g m o i d } ( 0 . 5 S _ { X } + 1 . 2 5 S _ { U } ) , 0 . 0 2 , 0 . 9 8 \} ,\tag{59}
$$

$$
A \sim { \mathrm { B e r n o u l l i } } \{ p ( X , U ) \} ,\tag{60}
$$

$$
Y ( 0 ) = 0 . 5 S _ { X } + 1 . 5 S _ { U } + \epsilon _ { Y } ,\tag{61}
$$

$$
Y ( 1 ) = Y ( 0 ) + 0 . 5 + S _ { X } + 0 . 5 S _ { U } ,\tag{62}
$$

$$
Y = ( 1 - A ) Y ( 0 ) + A Y ( 1 ) .\tag{63}
$$

The individual and conditional average effects satisfy

$$
\mathrm { I T E } = 0 . 5 + S _ { X } + 0 . 5 S _ { U } , \qquad Q ( X ) = \mathbb { E } [ \mathrm { I T E } \mid X ] = 0 . 5 + S _ { X } .
$$

Thus the CATE varies with X, while latent variation also modifies individual effects. In the population, $\operatorname { V a r } ( Q ( X ) ) =$ 1 and $\operatorname { V a r } ( \operatorname { I T E } - Q ( X ) ) = 0 . 2 5$ . Normalising the scores by $\sqrt { d }$ holds the population scales of confounding and heterogeneity fixed across dimensions.

Joint proxy information. For each dimension, we draw and fix independent orthogonal matrices $R _ { W } , R _ { Z }$ using Gaussian matrices and QR decomposition with a nonnegative triangular diagonal. At target information $s ,$ define $q = s / ( 2 - s )$ and generate

$$
W = \sqrt { q } R _ { W } U + \sqrt { 1 - q } \epsilon _ { W } , \qquad Z = \sqrt { q } R _ { Z } U + \sqrt { 1 - q } \epsilon _ { Z } .
$$

Each proxy coordinate has unit variance. Each complete proxy block alone explains a fraction $q$ of each latent coordinate’s variance. The joint Gaussian distribution gives

$$
\operatorname { \mathbb { E } } [ U \mid W , Z ] = { \frac { \sqrt { q } } { 1 + q } } { \bigl ( } R _ { W } ^ { \top } W + R _ { Z } ^ { \top } Z { \bigr ) } ,\tag{64}
$$

$$
\mathrm { C o v } ( U \mid W , Z ) = \frac { 1 - q } { 1 + q } I _ { d } = ( 1 - s ) I _ { d } .\tag{65}
$$

Consequently, the population $R ^ { 2 }$ for predicting any coordinate of $U$ from $( W , Z )$ jointly is exactly s. Independence of X preserves this value when conditioning additionally on $X$ . The rotations mix latent coordinates without changing information.

Bridge and pairing. Define the unbiased proxy score $\widetilde { S } _ { U } ( W ) = \mathbf { 1 } ^ { \top } R _ { W } ^ { \top } W / \sqrt { q d }$ . An explicit bridge is

$$
h _ { 0 } ( W , a , X ) = 0 . 5 S _ { X } + ( 1 . 5 + 0 . 5 a ) \widetilde { S } _ { U } ( W ) + a ( 0 . 5 + S _ { X } ) .
$$

Independent proxy errors give $\operatorname { \mathbb { E } } [ Y - h _ { 0 } ( W , A , X ) \mid A , Z , X ] = 0$ , and averaging the bridge contrast over $W \mid X$ gives $Q ( X ) = 0 . 5 + S _ { X }$ . The proxy-score error has conditional variance $( 1 - q ) / q$ . This quantifies the noise amplification as proxy information decreases, even though the bridge equation remains valid.

Estimator comparison. We fit an interacted proximal two-stage least-squares (2SLS) estimator (Tchetgen Tchetgen et al., 2020) with bridge and instrument designs

$$
\begin{array} { r l } & { D _ { i } = ( 1 , { X } _ { i } ^ { \top } , { A } _ { i } , { A } _ { i } { X } _ { i } ^ { \top } , { W } _ { i } ^ { \top } , { A } _ { i } { W } _ { i } ^ { \top } ) , } \\ & { T _ { i } = ( 1 , { X } _ { i } ^ { \top } , { A } _ { i } , { A } _ { i } { X } _ { i } ^ { \top } , { Z } _ { i } ^ { \top } , { A } _ { i } { Z } _ { i } ^ { \top } ) . } \end{array}
$$

Writing $P _ { T }$ for projection onto the instrument column space, 2SLS minimises $\| P _ { T } ( Y - D \beta ) \| _ { 2 } ^ { 2 }$ . The treatment interactions allow the bridge basis to contain $h _ { 0 }$ and to represent effect modification by both X and the latent confounder. We compute the projection using economic QR followed by SVD, retaining singular values above $\epsilon _ { \mathrm { m a c h } } \operatorname* { m a x } ( n _ { \mathrm { c t x } } , p _ { T } ) \sigma _ { \mathrm { m a x } }$ , where $p _ { T } = 2 + 4 d$ . The subsequent least-squares fit uses NumPy’s rcond=None rank threshold.

We compare 2SLS with unpenalised OLS using the design $( 1 , X ^ { \top } , A , A X ^ { \top } , C ^ { \top } , A C ^ { \top } )$ , with either no additional covariates (C empty) or proxy adjustment $( C = ( W , Z ) )$ . For these linear estimators, we average the fitted treatment contrast over the context distribution of the additional covariates:

$$
\begin{array} { r } { \widehat { Q } ( x ) = \widehat { \beta } _ { A } + x ^ { \top } \widehat { \beta } _ { A X } + \overline { { C } } _ { \mathrm { c t x } } ^ { \top } \widehat { \beta } _ { A C } . } \end{array}
$$

For 2SLS, $C = W ;$ for X-only OLS, the final term is omitted. This marginalisation targets $Q ( x )$ because X is independent of the additional covariates in the data-generating mechanism.

We also evaluate ProximalFM and proxy-adjusted CausalPFN. ProximalFM conditions on the observational context (X, W, Z, A, Y) and predicts CATE from query X. CausalPFN treats $( W , Z )$ as adjustment covariates; we marginalise its treatment contrasts using 16 seeded, aligned proxy pairs sampled from the context. At each context size, all methods use the same training rows and are scored against the known $Q ( X )$ on the same 256 held-out query rows. We report ${ \mathrm { P E H E } } = \{ 2 5 6 ^ { - 1 } \sum _ { i \in { \mathrm { q u e r y } } } [ { \widehat { Q } } ( X _ { i } ) - Q ( X _ { i } ) ] ^ { 2 } \} ^ { 1 / 2 }$

Discussion. Figure D.4 shows the advantage of a well-specified proximal estimator: the 2SLS bridge basis contains the true outcome bridge. Its large fluctuations at low proxy information reflect the difficulty of estimating that bridge with weak instruments, rather than model misspecification. ProximalFM improves more smoothly as information increases and, at $n _ { \mathrm { c t x } } = 2 0 4 8$ , outperforms proxy-marginalized CausalPFN throughout the sweep and proxy-adjusted OLS in most settings. OLS is a favourable comparator here: the treatment effect is linear, and independence of X and (W, Z) makes proxy marginalisation a simple average over context rows, without conditional-neighbour estimation. As the proxies nearly reveal U, OLS catches up with and surpasses PROXI-MALFM. These curves come from one paired simulation per setting.

Proxy informativeness across context sizes  
![](images/f8c9d3fd75a72a811edefe09406850d10b0ab3c751bf9abb0aebc778026c9771.jpg)  
Figure D.4: CATE RMSE against joint proxy information, measured by the population $R ^ { 2 }$ for predicting U from (W, Z). Columns vary proxy dimension d; rows vary context size. All methods are evaluated on the same 256 held-out query rows at each setting. The vertical axis is logarithmic. Within each dimension, the underlying draws are shared across information levels; results come from one realization per setting.

## E Towards a real physical system: Causal Chambers

We use the Light Tunnel Mk2 on the Causal Chamber platform<sup>5</sup> (Gamella et al., 2025; Zeitler, 2025) to study average treatment effects in a controllable but real (non-simulated) optical system. The experiment combines physical responses with software control to create dependence between a hidden confounder, proxy variables, treatment, and outcome. This provides an exploratory step towards proximal estimation in a real-world systems, complementing the synthetic CATE benchmarks.

![](images/369600677f7aa373645bb6fdd3a9c973867a8c97de302163a47e7c7d002d053e.jpg)  
Figure E.1: The Light Tunnel Mk2 and a simplified diagram with the variables relevant to our setup.

Physical setup and causal structure. Figure E.1 shows the physical setup with the variables we consider, and Figure E.2 shows the intended pathways between them. The confounder U is an LED drive drawn uniformly from the integers $0 , \ldots , 2 5 5$ , controlling its brightness. The observed covariate X controls a red LED, Z controls the ultraviolet LED 2, and W controls the position (in degrees) of the first polariser. Treatment $A \in \{ 0 , 1 \}$ sets ultraviolet LED 3 to drive $a _ { 1 } A .$ , and the outcome Y is the infrared sensor 3 reading. Auxiliary measurements comprise the electrical current C drawn by the confounder LED, infrared readings $I _ { 1 } , I _ { 2 }$ , and visible-light reading $V _ { 1 }$

The pathways $C  X , I _ { 1 }  Z , V _ { 1 }  W$ , and $I _ { 2 }  A$ —resulting from the software control functions—link successive physical measurements to subsequent interventions. In the working proximal structure, Z influences treatment assignment through $I _ { 2 } ,$ , while $W$ influences the light reaching the outcome sensor. The green LED and crossed polarisers define the nominal weak-confounding setting. The blue LED and a second polariser fixed at zero degrees define the nominal strong-confounding setting.

Control functions and acquisition. Define

$$
C _ { [ l , h ] } ( v ) = \operatorname* { m a x } \{ l , \operatorname* { m i n } ( h , v ) \} , \qquad S _ { L , k , m , b } ( v ) = \frac { L } { 1 + \exp \{ - k ( v - m ) \} } + b .
$$

For each observation, we set the selected confounder LED to $U$ and measure $C .$ We then set

$$
X = C _ { [ 0 , 2 5 5 ] } \{ { \mathrm { r o u n d } } ( f _ { 1 } ( C ) ) \}
$$

and measure $I _ { 1 } , V _ { 1 }$ . Independent standard-normal draws $\epsilon _ { Z } , \epsilon _ { W }$ determine

$$
\begin{array} { r l } & { Z = C _ { \left[ 0 , 4 0 9 5 \right] } \{ \mathrm { r o u n d } ( f _ { 2 } ( I _ { 1 } ) + k _ { Z } \sigma _ { Z } \epsilon _ { Z } ) \} , } \\ & { W = C _ { \left[ - 1 8 0 , 1 8 0 \right] } \{ f _ { 4 } ( V _ { 1 } ) + k _ { W } \sigma _ { W } \epsilon _ { W } \} . } \end{array}\tag{66}
$$

(67)

Polariser angles use degrees. In the weak setting, the second polariser is crossed with the first at $W + 9 0 .$ , blocking most of the light reaching the third relative angle is zero. After setting these inputs, we measure $I _ { 2 } ,$ assign treatment, and measure $Y$

The linear family uses

$$
f _ { 1 } ( c ) = C _ { [ 0 , 2 5 5 ] } ( - 0 . 0 8 5 c - 1 4 4 5 ) , \qquad f _ { 2 } ( i ) = C _ { [ 0 , 4 0 9 5 ] } ( 0 . 0 6 2 4 i ) .
$$

The nonlinear family uses

$$
f _ { 1 } ( c ) = S _ { 2 5 5 , 0 . 0 0 2 , - 1 9 0 0 0 , 0 } ( c ) , \qquad f _ { 2 } ( i ) = S _ { L , 0 . 0 0 1 , 3 5 0 0 0 , 0 } ( i ) ,
$$

with $L = 4 0 9 5$ in the weak setting and $L = 1 0 0 0$ in the strong setting. Every family uses $f _ { 4 } ( v ) = S _ { 1 8 0 , 0 . 0 0 2 , 4 0 0 0 , - 9 0 } ( v )$ Thus the family label specifies the form of $f _ { 1 }$ and $f _ { 2 }$ . Figure E.3 shows their saturation, the common sigmoid polariser response, and the treatment-assignment rules. These mappings change which parts of the optical response the experiment explores. The reduced nonlinear amplitude in the strong setting limits the contribution of $Z$ to the treatment-assignment sensor, preserving variation associated with the confounder.

Treatment and proxy-information settings. We collect 1 000 under the following experimental design. We have a Bernoulli assignment as follows

$$
p ( I _ { 2 } ) = \mathrm { s i g m o i d } \{ ( c - I _ { 2 } ) / s _ { A } \} , \qquad A = { \bf 1 } \{ V _ { A } < p ( I _ { 2 } ) \} , \qquad V _ { A } \sim \mathrm { U n i f } ( 0 , 1 ) ,
$$

where $( c , s _ { A } ) = ( 5 2 5 0 , 8 0 0 )$ for both linear settings, (8789, 2000) for weak nonlinear, and (3930, 800) for strong nonlinear. The Bernoulli rule supplies assignment variability at a given $I _ { 2 }$ . Its probabilities lie between 0.05 and 0.95 for 82.4%–99.7% of observations across the corresponding experimental conditions.

The noise multipliers $( k _ { Z } , k _ { W } )$ equal $( 0 , 0 )$ at high nominal informativeness, (0.75, 0.75) at intermediate informativeness, and $( 2 , 2 )$ at low informativeness. The strong-linear low-information condition uses (0.6, 2.5) to retain a measurable confounding signal while reducing proxy information. The fixed noise scales are

<table><tr><td>Setting</td><td> $\sigma _ { Z }$   $\sigma _ { W }$ </td><td>(degrees)</td></tr><tr><td>Weak linear</td><td>325</td><td>62.2</td></tr><tr><td>Weak nonlinear</td><td>1789</td><td>64.5</td></tr><tr><td>Strong linear</td><td>383</td><td>66.2</td></tr><tr><td>Strong nonlinear</td><td>412</td><td>62.6</td></tr></table>

We quantify realised information by the $R ^ { 2 }$ of a linear regression of $U$ on $( Z , W )$ with an intercept. Confounding is summarised by $\mathbb { E } _ { n } [ R ^ { ( 0 ) } \ | \ A = 1 ] - \mathbb { E } _ { n } [ R ^ { ( 0 ) } \ | \ A = 0 ]$ . Because the proxies are physical actuators, their noise also changes the treatment and outcome distributions. Each noise level therefore defines a distinct operating distribution. NumPy random streams use seeds 981264976135, 981264976142, and 981264976146 for $U .$ , proxy noises, and treatment uniforms, respectively. Restarting these streams for each condition pairs the commanded inputs and software randomness, while sensor values come from fresh physical measurements.

ATE reference and interpretation. After measuring $Y ,$ we apply drive zero and then drive $a _ { 1 }$ , retaining the other commanded inputs, to obtain $R ^ { ( 0 ) }$ and $R ^ { ( 1 ) }$ . The empirical average intervention reference is

$$
\widehat { \theta } _ { \mathrm { r e f } } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \{ R _ { i } ^ { ( 1 ) } - R _ { i } ^ { ( 0 ) } \} .
$$

The sequence in Figure E.2 makes this average directly measurable over the sampled operating states. Its causal interpretation assumes stability during each sequence and negligible systematic effects of measurement order or carryover.

The mean contrast ranges from approximately 886 to 893 sensor counts at drive 500, and from 621 to 624 at drive 350. Across conditions, the standard deviation of individual contrasts is 3.55–65.64 counts. The root mean squared difference between $Y$ and the separately remeasured factual arm is 3.10–65.00 counts. The overlap in these scales supports evaluating the average contrast, with individual contrasts interpreted in the presence of measurement variability. The physical estimand we obtain here is the ATE over the acquired states. The synthetic benchmarks in the other sections provide complementary evaluations against analytic CATE functions.

Adaptation of methods for ATE estimation. For each context prefix ${ \mathcal { D } } _ { n } = \{ ( X _ { i } , W _ { i } , Z _ { i } , A _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n }$ , we estimate the empirical ATE over the acquired operating states,

$$
\theta _ { n } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \{ Y _ { i } ( 1 ) - Y _ { i } ( 0 ) \} .
$$

All methods are fitted using the factual observations in $\mathcal { D } _ { n }$ only. The paired intervention measurements $Y _ { i } ( 0 )$ and $Y _ { i } ( 1 )$ are reserved exclusively for the reference ATE. When a method produces a conditional effect estimate, we average its predictions over the same context prefix.

We consider a selection of methods from Appendix D.1. X-learner (TabICL) fits arm-specific outcome and effect models using $( X , W , Z )$ and returns an estimated contrast at each observed $( X _ { i } , W _ { i } , Z _ { i } )$ ; its ATE estimate is the mean of these contrasts. CausalPFN directly predicts a conditional treatment effect from the factual context, which we likewise average over the observed covariate and proxy values.

The proximal methods use the proxy roles explicitly. P-learner estimates outcome and treatment bridge functions and forms cross-fitted doubly robust proximal scores; its ATE estimate is the mean of these scores. KPV and PMMR estimate an outcome bridge $h ( a , w , x )$ and obtain a marginal ATE by integrating the contrast $h ( 1 , W , X ) - h ( 0 , W , X )$ over the empirical joint distribution of (X, W) in the fitted context. NMMR uses the same empirical target, averaging its learned bridge contrast at the observed $( X _ { i } , W _ { i } )$ pairs. P2SLS uses a linear two-stage specification: it first predicts W from $( A , X , Z )$ and then regresses Y on $( A , X , { \widehat { W } } )$ ; the fitted coefficient of A is its scalar ATE estimate.

Finally, PROXIMALFM treats the factual rows as its in-context dataset and predicts $\widehat { \tau } ( X _ { i } )$ for every context covariate value, with query-time proxies hidden. Its ATE estimate is the empirical mean $n ^ { - 1 } \sum _ { i } { \widehat { \tau } } ( X _ { i } )$ . This use of PROXIMALFM is not its primary design target: it uses $( W , Z )$ to interpret the factual context but predicts effects from X alone, deliberately discarding query-time proxy information. It can therefore provide a useful ATE estimate by averaging its CATE predictions, but it is not designed to estimate the ATE directly or to integrate an outcome bridge over the empirical proxy distribution.

![](images/65d36912fccc50ad001d9132fa1031199715804fd58f08fea5c79f02e4c58a24.jpg)  
Other commanded inputs remain fixed during the three outcome measurements

Figure E.2: Physical and software pathways in the optical experiment. Solid arrows represent the predetermined physical pathways and dashed green arrows represent effects resulting from the software control functions, i.e., software mappings. Independent Gaussian noise enters the two proxy mappings. The lower sequence shows acquisition of the factual outcome and the two intervention measurements. The graph describes the intended within-observation structure. Its causal interpretation requires stable device state and the proxy restrictions discussed in the text.

![](images/8bb4fbf9d5af768e1d4d002b35d34cd02f1e23e8ad06417243da95396a8cbc61.jpg)  
(c) Outcome-proxy control

(b) Treatment-proxy control  
![](images/83b361268029b57750b55b8172780026378d54fb94cb9d6e734f9e304bed4485.jpg)  
(d) Treatment assignment

![](images/af9ade4d3b1062228cba0be0cb95c5dc0425eb6b21ffbb199332fa8b612bf37f.jpg)

![](images/ac63d06ad9ebc4f680e293b8d67ec81014f1a8befd9715e215152ab4ac4380b5.jpg)  
Figure E.3: Software control functions for the physical experiment. Panels (a)–(c) show the deterministic mappings before addition of proxy noise and final actuator clipping. Panel (d) shows the treatment probabilities for Bernoulli assignment and the corresponding hard-threshold protocols. The plots display the specified control functions over illustrative sensor ranges.

Results Figure E.4 shows no single method that performs best in every physical setting. The X-learner (TabICL) is the most consistently accurate method, especially when confounding is weak. CausalPFN is also competitive, particularly for nonlinear mechanisms and weak confounding. In contrast, the proximal methods, including PROXIMALFM, perform well only in some settings, and larger contexts do not lead to systematic improvements. Overall, these results illustrate the difficulty and variability of causal estimation in real physical systems.

![](images/ce90d4a50c98c1238f89fa3e65e766a598a6cd140a2427d66c5eb810a008437e.jpg)  
Figure E.4: Absolute error of estimated average treatment effects in the Causal Chambers Light Tunnel experiment as the factual context grows. Each panel is one physical operating condition, crossing control-function family (linear or non-linear), confounding strength, and nominal proxy informativeness. Curves show $| { \widehat { \theta } } _ { n } - { \widehat { \theta } } _ { \mathrm { r e f } , n } |$ in raw infrared-sensor counts, where both the estimate and the paired-intervention reference are evaluated on the same nested context prefix of size n. Both axes use logarithmic scales; lower is better.