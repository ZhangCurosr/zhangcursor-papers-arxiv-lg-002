# When Interference Graphs Evolve: Doubly Robust Estimation of Dynamic Peer Efects

Xiaojing Du

Adelaide University

## Abstract

Peer efects are dificult to estimate when interaction graphs evolve because pre-assignment network history, dynamic peer exposure, and post-assignment network change have distinct causal roles. We introduce a controlled contrast framework that indexes potential outcomes by own treatment, temporally aggregated peer exposure, and a post-assignment evolution summary. Diferences between the resulting means define own-treatment, peer-exposure, controlled network-evolution, and joint controlled contrasts rather than a mediation decomposition. We develop the Dynamic Network Doubly Robust estimator, DynaNet-DR, which combines a temporally factorized propensity with normalized augmentation. Under consistency, summary suficiency, sequential exchangeability, positivity, nuisance convergence, and weak dependence, its canonical estimator is consistent when either the outcome regression or the propensity estimator is consistent. The reported implementation adds representative-score prediction, fixed clipping, and finite-sample stabilization. Semi-synthetic benchmarks on fixed real temporal graph sequences show favorable estimation accuracy among methods targeting the full profile. These benchmarks assess summary-indexed contrasts rather than counterfactual edge generation, and the MathOverflow study is an observational illustration under the stated assumptions.

## 1 Introduction

Interventions on social platforms, recommender systems, educational platforms, and digital marketplaces can afect treated individuals and their peers. When a user receives a promotion, recommendation, incentive, or collaboration tool, outcomes may also change among friends, teammates, classmates, or trading partners. Such peer efects violate the nointerference assumption of classical causal inference (Manski 1993; Hudgens and Halloran 2008; Aronow and Samii 2017). Because experimental and observational evaluations inform rollout and targeting decisions, analyses that ignore interference can conflate responses to a unit’s own assignment with responses to peer assignments (Eckles, Karrer, and Ugander 2017; Forastiere, Airoldi, and Mealli 2021). The central question is how to define and estimate these responses when the interaction graph changes over time.

Platform networks evolve as users follow, comment, form teams, and trade. A user’s neighborhood can change across periods, and brief interactions can afect later peer exposure. Static summaries based only on the contemporaneous fraction of treated neighbors may omit relevant temporal structure (Aronow and Samii 2017; Sävje 2024; Adhikari, Medya, and Zheleva 2025; Lin et al. 2026). Dynamic exposure can depend on current and historical neighbors, edge strength, and temporal persistence. We therefore construct a structured exposure mapping with interpretable intervention levels and diagnose their empirical support.

Post-assignment network evolution introduces a separate causal role. Invitation and recommendation policies may alter subsequent interactions or local connectivity, while other evolution may occur independently of treatment. Preassignment graph history informs confounding adjustment and exposure construction; a post-assignment evolution summary defines an additional coordinate of the controlled intervention. Varying this summary while fixing own treatment and peer exposure yields a controlled network-evolution contrast, not a natural indirect efect or a component of a totalefect decomposition. The framework permits externally occurring or treatment-responsive evolution, but the reported benchmarks keep each observed graph sequence fixed and generate the summary as a post-assignment label; they do not generate counterfactual edges. Figure 1 previews the four controlled comparisons in natural language. Within each panel, the target difers from the reference only in the highlighted feature, and the comparisons are not additive pathways.

Prior work addresses related components of this problem. Exposure mappings and generalized propensity scores formalize interference for a specified graph (Hudgens and Halloran 2008; Aronow and Samii 2017; Forastiere, Airoldi, and Mealli 2021). Graph-learning methods estimate highdimensional exposure representations (Ma and Tresp 2021; Adhikari, Medya, and Zheleva 2025), and recent studies develop doubly robust estimators for network interference (Chen et al. 2024; Khatami et al. 2025). Dynamicinterference models incorporate evolving graphs into prediction (Lin et al. 2026); treatment-responsive-network and edge-intervention studies target related but distinct changes to graph structure (Comola and Prina 2021; Chen, Hu, and Jiang 2026). Our framework uses pre-assignment graph history for adjustment and exposure construction, then includes a specified post-assignment evolution summary in the controlled mean profile.

![](images/1c119f409b11e43c7ed26b2dfaa1c9464a0d9ea461cd07b92c512b868734b5b5.jpg)  
Figure 1: Controlled comparisons for focal unit i. Each panel varies one highlighted feature while holding the others fixed.

This paper studies causal efect estimation under evolving network interference. We distinguish pre-assignment network history, used for adjustment and exposure construction, from a post-assignment network evolution summary, treated as a coordinate of the controlled intervention. We develop the Dynamic Network Doubly Robust estimator, DynaNet-DR, for the resulting controlled contrast profile. It combines temporal graph features, a temporally ordered propensity factorization, and normalized augmentation. We make four contributions.

• We formalize evolving-network interference through a controlled contrast profile that separates pre-assignment network history, dynamic peer exposure, and a postassignment evolution summary.

• We construct a dynamic exposure mapping from recent peer assignments, pre-assignment edge weights, and temporal decay, with interpretable intervention levels and explicit overlap diagnostics.

• We develop DynaNet-DR by combining a temporally ordered propensity factorization with normalized augmentation. We establish double robustness for its canonical form and distinguish that result from the reported finitesample stabilization.

• We evaluate same-estimand accuracy and restricteddesign scope on four fixed real temporal graph sequences with generated assignments, evolution-summary labels, and outcomes; isolate the adaptive gate in an ablation; report overlap and hidden-homophily diagnostics; and present MathOverflow as an observational illustration.

## 2 Problem Definition

We define the dynamic networked observation model, the exposure and evolution summaries, the target mean, three coordinate-wise controlled contrasts, and a joint controlled contrast. Appendix A summarizes the notation.

Dynamic networked observations. Consider n units observed over T time periods. At assignment time t, we observe a pre-assignment interaction graph $G _ { t } ^ { - }$ , a binary treatment vector $\pmb { A } _ { t }$ with $A _ { i t } ~ \in ~ \{ 0 , 1 \}$ , a subsequent graph $G _ { t + 1 } ^ { + } ,$ and an outcome vector $\mathbf { { \cal Y } } _ { t + 1 }$ measured after the evolution summary. For each unit-time pair (i, t), let $H _ { i t }$ contain observed covariates, treatments, outcomes, and network structure available before assignment. The index $t + 1$ denotes the post-assignment target horizon rather than requiring evolution and outcome to be measured simultaneously; Section 5 and Appendix H give the raw-bin mappings for the case study and benchmarks.

Peer exposure. The treatment vector of all other units is high-dimensional, so a dynamic exposure variable $Z _ { i t }$ aggregates treatments of relevant peers using recent network history, edge strength, and temporal decay. We use discrete levels $Z _ { i t } \in \{ 0 , 1 , \overline { { \mathbf { \Omega } } } . . . , K \}$ to define the target interventions and assess their empirical support.

Network evolution summary. Let $M _ { i , t + 1 }$ summarize local graph evolution after assignment, such as tie formation, dissolution, or intensity change. Unlike pre-assignment graph history, $M _ { i , t + 1 }$ is not included as an ordinary confounder; it is a coordinate of the controlled intervention. We take $M _ { i , t + 1 } \in \{ 0 , 1 \}$ . The summary may describe externally occurring or treatment-responsive evolution, but the reported benchmark keeps the observed graph sequence fixed and generates M as a label, as detailed in Section 5.

Potential outcomes. Let $Y _ { i , t + 1 } ( a , z , m )$ denote the potential outcome of unit i when its own treatment, peer exposure, and network evolution summary are set to $( a , z , m )$ . We target the marginal mean potential outcome

$$
\begin{array} { r } { \mu ( a , z , m ) = \operatorname { \mathbb { E } } \left[ Y _ { i , t + 1 } ( a , z , m ) \right] , } \end{array}\tag{1}
$$

where the expectation is over the target distribution of unittimes and pre-assignment histories. Equation (1) defines a single-world controlled mean. Natural, interventional, and path-specific mediation efects require additional counterfactual structure.

Causal contrasts. The target mean $\mu ( a , z , m )$ supports three coordinate-wise controlled contrasts. The owntreatment contrast compares own-treatment levels while holding peer exposure and network evolution fixed:

$$
\tau _ { D } ( z , m ) = \mu ( 1 , z , m ) - \mu ( 0 , z , m ) .\tag{2}
$$

The peer-exposure contrast compares peer-exposure levels while holding own treatment and network evolution fixed:

$$
\tau _ { S } ( a ; z , z ^ { \prime } , m ) = \mu ( a , z , m ) - \mu ( a , z ^ { \prime } , m ) .\tag{3}
$$

The controlled network-evolution contrast compares postassignment evolution-summary levels while holding own treatment and peer exposure fixed:

$$
\tau _ { M } ( a , z ; m , m ^ { \prime } ) = \mu ( a , z , m ) - \mu ( a , z , m ^ { \prime } ) .\tag{4}
$$

For two joint intervention levels $\theta _ { 1 } ~ = ~ ( a _ { 1 } , z _ { 1 } , m _ { 1 } )$ and $\theta _ { 0 } = ( a _ { 0 } , z _ { 0 } , m _ { 0 } )$ , write $\mu ( \theta _ { j } ) = \mu ( a _ { j } , z _ { j } , m _ { j } )$ . We define the joint controlled contrast $ { \tau } _ { J } ( \theta _ { 1 } , \theta _ { 0 } ) = \mu ( \theta _ { 1 } ) - \mu ( \theta _ { 0 } )$ Together, the four contrast families form a profile of distinct state comparisons without imposing an additive decomposition. The estimation task is to recover these means and contrasts under observed time-varying confounding and network dependence, with explicit diagnostics for limited overlap. Homophily left unmeasured in $H _ { i t }$ can violate sequential exchangeability.

## 3 Identification

We state the identification assumptions and derive the observed-data functional for the controlled means. Section 4 develops the estimator and states its theoretical scope.

## 3.1 Assumptions

Assumption 1 (Consistency). For each unit-time pair, the observed outcome equals the potential outcome indexed by the observed own treatment, peer exposure, and network evolution summary.

Assumption 2 (Exposure and evolution-summary suficiency). Conditional on $H _ { i t }$ , the potential outcome at the target horizon depends on other units’ assignments and $d y -$ namic network paths only through $A _ { i t } , \ Z _ { i t } ,$ , and $M _ { i , t + 1 }$ Graph configurations mapped to the same $( z , m )$ are treated as outcome-equivalent versions.

Assumption 3 (Sequential exchangeability). Conditional on $H _ { i t }$ , the joint assignment $( A _ { i t } , Z _ { i t } )$ is independent of the potential outcomes $\backslash \{ Y _ { i , t + 1 } ( a , \stackrel {  } { z } , { m } ) \}$ . In addition, conditional on $\left( H _ { i t } , A _ { i t } , Z _ { i t } \right)$ , the observed network evolution summary $M _ { i , t + 1 }$ is independent of the controlled potential outcomes $\{ Y _ { i , t + 1 } ( a , z , \bar { m } ) \}$

Assumption 4 (Positivity). Every treatment, peer exposure, and network evolution level used in the target estimand has positive probability conditional on histories in the target population: thejoint treatment-exposure probability and the conditional network-evolutionprobability are bounded away from zerofor target levels.

Assumptions 1 through 4 follow sequential identification arguments (Robins 1986; Hernán and Robins 2020) and their network counterparts under exposure mappings (Hudgens and Halloran 2008; Forastiere, Airoldi, and Mealli 2021). Assumption 2 specifies the summaries and their outcomeequivalent versions, which is substantive when exposure mappings may be misspecified (Sävje 2024). Stochastic interventions over graph configurations provide a possible extension. The second part of Assumption 3 addresses confounding of the post-assignment summary.

Figure 2 summarizes the assumed structure. Under $\mathbf { A } \mathbf { s } \mathbf { . }$ sumption 3, $H _ { i t }$ is suficient for the two stated sequential exchangeability conditions; in the default benchmark, its nuisance representation includes $U _ { i }$ . Summary suficiency and exchangeability are substantive assumptions, not consequences of the representation.

Where homophily enters. In the benchmark, homophilyrelated confounding is represented by a generated community-correlated unit trait $U _ { i }$ . The trait enters the treatment, evolution-label, and outcome models, while its community correlation associates it with peer exposure and the fixed graph structure. The default benchmark includes $U _ { i }$ in the nuisance features; Appendix E reports a sensitivity analysis that withholds it. As in observational network studies more generally, peer correlation is not separable from contagion without additional assumptions (Shalizi and Thomas 2011).

![](images/df2039255b946e5a49cccd6189ca7af13b48876c7d5243a88bd0808b373cd197.jpg)  
Figure 2: Time ordering. Peer treatments are aggregated over pre-assignment graphs into $Z _ { i t }$ . The post-assignment summary $M _ { i , t + 1 }$ precedes the outcome. The generated trait $U _ { i }$ is included by default and withheld in the sensitivity analysis.

## 3.2 Identification

Theorem 1 (Identification of controlled means). Under Assumptions 1 through $^ { 4 , }$ the mean potential outcome $\mu ( a , z , m )$ is identified by

$$
\begin{array} { r } { \mu ( a , z , m ) = \operatorname { \mathbb { E } } \bigl [ \operatorname { \mathbb { E } } [ Y _ { i , t + 1 } \mid A _ { i t } = a , Z _ { i t } = z , } \\ { M _ { i , t + 1 } = m , H _ { i t } ] \bigr ] . } \end{array}\tag{5}
$$

Consequently, all contrasts of $\dot { \mathbf { \mu } } ( a , z , m )$ , including Eqs. (2) to (4), are identifiable.

Proof. Consistency connects observed and potential outcomes, sequential exchangeability removes confounding for $( A _ { i t } , Z _ { i t } )$ and then $M _ { i , t + 1 }$ , and positivity makes the regression well defined at every target level. Averaging over histories yields Eq. (5), as derived in full in Appendix C. □

## 4 Method

The Dynamic Network Doubly Robust estimator, DynaNet-DR, combines a structured dynamic exposure, a postassignment evolution summary, temporally held-out nuisance estimation, and normalized augmentation. Algorithm 1 in Appendix H summarizes the reported pipeline.

## 4.1 Dynamic Exposure Representation

The reported exposure module uses fixed temporal decay and edge weights to aggregate recent peer treatments into a continuous score $S _ { i t }$ , then discretizes it into low, medium, and high levels $Z _ { i t }$ . The canonical estimator below treats Z as the intervention and uses $Q ( a , z , m , h )$ . In the reported implementation, $S _ { i t }$ is also supplied to the outcome learner: observed-outcome residuals use the observed score, whereas a target prediction at level z sets the score to its training conditional mean $s _ { z }$ . We refer to this as a representativescore plug-in approximation. It is not generally equivalent to integrating a nonlinear outcome regression over the withinlevel score distribution. The benchmark outcome is generated from discrete $Z$ rather than S, but this score-irrelevance property is not assumed for the observational application.

## 4.2 Network Evolution Representation

In the general framework, $M _ { i , t + 1 }$ may summarize a diference between post- and pre-assignment local graphs. The semi-synthetic benchmark instead generates a binary postassignment label conditional on the simulated variables while keeping the real graph sequence fixed; the MathOverflow application uses an indicator for gaining a previously unseen neighbor. In each case, M enters the target and nuisance models as a controlled level, not as a pre-assignment adjustment variable. The estimand does not by itself identify the efect of a particular edge-editing policy.

## 4.3 Nuisance Models with Temporal Graph Features

We fit gradient-boosted outcome and propensity models to handcrafted node-time features $\phi _ { i t }$ from $H _ { i t }$ . The outcome model conditions on $\phi _ { i t }$ and $( a , z , m )$ ; the propensity models estimate the joint assignment $g _ { A Z , 0 } ( a , z \mid H _ { i t } )$ and the subsequent summary mechanism $g _ { M , 0 } ( m \mid a , z , H _ { i t } )$ . The training window is divided into three contiguous blocks. Each nuisance model is fit on the complement of one block, and the three predictions are averaged on a later estimation window not used for fitting. This is a temporally held-out nuisance ensemble rather than observation-level cross-fitting of the estimation sample; theoretical consistency therefore requires nuisance convergence on the estimation-window distribution.

## 4.4 Canonical Estimator and Reported Stabilization

Let

$$
\begin{array} { r l } & { Q _ { 0 } ( a , z , m , h ) =  { \mathbb { E } } [ Y _ { i , t + 1 } \mid A _ { i t } = a , Z _ { i t } = z , } \\ & { ~ M _ { i , t + 1 } = m , H _ { i t } = h ] . } \end{array}\tag{6}
$$

Equations (6) through (10) use the canonical discrete-$Z$ regression ${ \widehat { Q } } .$ . For the reported implementation, let $\widehat { Q } ^ { S } ( a , z , m , h , s )$ denote the score-augmented learner. Its target prediction substitutes $\widehat { Q } ^ { S } ( a , z , m , H _ { i t } , s _ { z } )$ for $\widehat { Q } ( a , z , m , H _ { i t } )$ , whereas the observed residual uses $\widehat { Q } ^ { S } ( A _ { i t } , Z _ { i t } , M _ { i , t + 1 } , H _ { i t } , S _ { i t } )$ . These substitutions define the representative-score statistic and coincide with the canonical estimator only under the compatibility condition stated below. For $\theta = ( a , z , m )$ , write ${ \widehat Q } ( \theta , h ) = { \widehat Q } ( a , z , m , h )$ We factor the generalized propensity in temporal order,

$$
g _ { 0 } ( a , z , m \mid h ) = g _ { A Z , 0 } ( a , z \mid h ) g _ { M , 0 } ( m \mid a , z , h ) .\tag{7}
$$

Define

$$
\widehat { W } _ { \theta } ( i , t ) = \frac { \mathbb { I } \{ ( A _ { i t } , Z _ { i t } , M _ { i , t + 1 } ) = \theta \} } { \widehat { g } ( \theta \mid H _ { i t } ) } .\tag{8}
$$

For the canonical estimator, define

$$
\widehat { Q } _ { i t } ^ { \mathrm { o b s } } = \widehat { Q } ( A _ { i t } , Z _ { i t } , M _ { i , t + 1 } , H _ { i t } ) .
$$

For the representative-score statistic, instead define

$$
\widehat { Q } _ { i t } ^ { \mathrm { o b s } } = \widehat { Q } ^ { S } ( A _ { i t } , Z _ { i t } , M _ { i , t + 1 } , H _ { i t } , S _ { i t } ) .
$$

In either case, let $R _ { i t } \ = \ Y _ { i , t + 1 } - { \widehat Q } _ { i t } ^ { \mathrm { o b s } }$ . The normalized correction is

$$
\widehat { c } ( \theta ) = \frac { \sum _ { ( i , t ) } \widehat { W } _ { \theta } ( i , t ) R _ { i t } } { \sum _ { ( i , t ) } \widehat { W } _ { \theta } ( i , t ) } .\tag{9}
$$

For target levels $\theta _ { 1 } , \theta _ { 0 }$ and $N$ estimation-window unit-time records, we write

$$
\begin{array} { l } { { \displaystyle { \widehat \tau } ( \theta _ { 1 } , \theta _ { 0 } ) = \frac { 1 } { N } \sum _ { ( i , t ) } \{ { \widehat Q } ( \theta _ { 1 } , H _ { i t } ) - { \widehat Q } ( \theta _ { 0 } , H _ { i t } ) \} } } \\ { { \displaystyle ~ + ~ { \widehat \lambda } \{ { \widehat { c } } ( \theta _ { 1 } ) - { \widehat c } ( \theta _ { 0 } ) \} . } } \end{array}\tag{10}
$$

The canonical normalized augmented estimator sets $\widehat { \lambda } = 1$ and uses the unclipped estimated propensity. It is the estimator covered by Theorem 2.

The reported implementation first clips the estimated joint propensity to [0.02, 0.98]. For each target cell, the inverse-weight efective sample size is $\begin{array} { r } { ( \sum _ { i , t } \widehat { W } _ { \theta } ( i , t ) ) ^ { 2 } / \sum _ { i , t } \widehat { W } _ { \theta } ( i , t ) ^ { 2 } ; \widehat { n } _ { \mathrm { e f f } } } \end{array}$ is the smaller of the two cell-specific values. This diagnostic measures weight concentration, not the number of independent observations under network dependence. Let $\widehat { d } = \widehat { c } ( \theta _ { 1 } ) - \widehat { c } ( \theta _ { 0 } )$ and let $\widehat { V } ( \widehat { d } )$ be its node-clustered variance estimate. With $n _ { \mathrm { m i n } } ^ { * } =$ 60, the implementation sets

$$
\widehat { \lambda } = \frac { \widehat { d } ^ { 2 } } { \widehat { d } ^ { 2 } + \kappa \widehat { V } ( \widehat { d } ) } , \qquad \kappa = \operatorname * { m a x } \{ 4 , 1 6 n _ { \operatorname * { m i n } } ^ { * } / \widehat { n } _ { \mathrm { e f f } } \} ,\tag{11}
$$

then sets $\widehat { \lambda } = 0$ when $\widehat { n } _ { \mathrm { e f f } } < n _ { \mathrm { m i n } } ^ { * }$ or the displayed value is below 0.25. These operations are finite-sample stabilization choices.

Theorem 2 (Double robustness of the canonical estimator). Consider Eq. (10) with $\widehat { \lambda } = 1$ and an unclipped propensity estimate, or with clipping that is asymptotically inactive. Suppose Assumptions 1 through 4 hold. Both temporally held-out nuisances converge in $L _ { 2 }$ on the estimationwindow distribution to finite limits $\overline { { Q } }$ and ${ \overline { { g } } } ,$ with either $\overline { { Q } } = Q _ { 0 } o r \overline { { g } } = g _ { 0 }$ . For each target cell, the estimated inverse weights and the terms in Eqs. (9) through (10) are uniformly square-integrable, and the normalized denominators converge to positive limits. Assume further that, conditional on the nuisance-training data, estimation-window moments converge to their target-population counterparts and that a weak-dependence law of large numbers holds for the dataadaptive node-time terms. Then ${ \widehat { \tau } } ( \theta _ { 1 } , \theta _ { 0 } )$ is consistent for $\mu ( \theta _ { 1 } ) - \mu ( \theta _ { 0 } )$

Inverse-weight square integrability follows, for example, if the estimated target-cell propensity is bounded away from zero with probability tending to one. A suficient within-window dependence condition is that, conditional on nuisance-training data, the N node-time records admit a dependency graph with maximum degree $\Delta _ { N } \ = \ o ( N )$ and uniformly bounded second moments. The empirical weightbased efective sample size summarizes weight concentration; it does not establish temporal stability or this dependence condition.

Proof. $\mathrm { I f } \stackrel { \widehat { Q } } { \cal Q }  { \cal Q } _ { 0 }$ , the plug-in term converges to the target contrast and each normalized residual correction converges to zero. If ${ \widehat { g } } \to g _ { 0 }$ , the normalized denominator converges to one and ${ \widehat { c } } ( \theta )$ converges to the target-cell bias correction ${ \mathbb E } \{ Q _ { 0 } ( \theta , H ) - { \overline { { Q } } } ( \theta , H ) \}$ , so the diference of corrections removes the plug-in contrast bias when $\widehat { \lambda } = 1$ (Bang and Robins 2005; van der Laan and Rubin 2006). □

For the adaptive estimator, suppose both target-cell efective sample sizes diverge, $\widehat { V } ( \widehat { d } ) \to 0$ , and clipping becomes asymptotically inactive. When the limiting correction difference is nonzero, the hard gates are eventually inactive and $\widehat { \lambda }  1 $ ; when it is zero, the plug-in contrast has the correct limit under either nuisance-correct branch. Thus the adaptive estimator shares the canonical limit under these conditions. The theorem does not cover the reported active fixed clip or the representative-score approximation without an additional within-level score-irrelevance or correctmarginalization condition.

Inference. For DynaNet-DR, we report standard errors from the empirical variance of the unshrunk influence scores with node-level clustering. The resulting intervals are nominal: node clustering does not establish a central limit theorem for cross-node dependence induced by an evolving graph (Ogburn et al. 2024; Leung 2020). We therefore report their empirical coverage in the semi-synthetic benchmarks without claiming general coverage validity.

Before interpreting a target contrast, we inspect cell counts, estimated propensities, and efective sample sizes. These diagnostics indicate observed support; they do not verify conditional positivity.

## 5 Experiments

We study same-estimand accuracy (RQ1), the estimand scope of restricted designs and ablations (RQ2), nuisancemodel stress tests (RQ3), and empirical coverage of the reported nominal intervals (RQ4).

## 5.1 Semi-Synthetic Benchmarks on Real Dynamic Networks

We construct semi-synthetic benchmarks on CollegeMsg (Panzarasa, Opsahl, and Carley 2009), email-Eu (Paranjape, Benson, and Leskovec 2017), and the SocioPatterns Primary School (Stehlé et al. 2011) and High School 2012 (Fournet and Barrat 2014) networks. Their fixed graph sequences retain temporal structure, activity heterogeneity, and communities. On each sequence, the generator assigns treatment, derives exposure from assigned peer treatments and observed history, draws a post-assignment evolution-summary label, and generates outcomes with known contrasts; it does not regenerate edges or simulate an alternative graph sequence.

Treatment is logistic in observed history and a generated community-correlated trait $U _ { i } ; Z _ { i t } \in \{ 0 , 1 , 2 \}$ discretizes a decayed, edge-weighted treated-neighbor share. Binary $M _ { i , t + 1 }$ is logistic in (A, Z), their interaction, history, and $U _ { i } { \mathrm { : } }$ ; the continuous outcome follows the reported model with main efects and interactions. The default nuisance features include $U _ { i } ,$ whereas the hidden-homophily analysis withholds it. Appendix H gives the generator and dataset statistics.

## 5.2 Baselines and Protocol

Adapted targeted learning (TL; Chen et al. 2024), the adapted Graph Machine Learning Based Doubly Robust estimator (GML-DR; Khatami et al. 2025), and DynaNet-DR target all reported (A, Z, M) contrasts. The restricted designs are Naive-A (Imbens and Rubin 2015), static exposure outcome regression and doubly robust estimators (Static-Z OR/DR; Robins 1986; Bang and Robins 2005), an adapted generalized propensity score estimator (GPS; Forastiere, Airoldi, and Mealli 2021), and an adapted dynamic interference model (DynInt; Lin et al. 2026). Static rows replace dynamic exposure, DynInt uses a representative continuous score, and designs without M return CNE as zero and omit M elsewhere. These rows diagnose target scope rather than same-estimand accuracy. No-M IPW removes both M and the outcome regression; the other ablations remove augmentation, $M .$ , or dynamic exposure. The λ = 1 row retains the full target and normalized correction while removing only the adaptive gate. All rows use the same analysis population, ten replications, shared available features, and time-ordered 60/20/20 splits. GML-DR also uses embeddings. The joint propensity clip is [0.02, 0.98]. Appendix F details the adaptations.

## 5.3 Evaluation Metrics

We report seven contrasts in four labeled families: owntreatment (DE), peer-exposure (PE), controlled networkevolution (CNE), and joint controlled (JCC). Here and in the figures, DE denotes the own-treatment family, not a mediation or path-specific direct efect. Table 5 lists the target levels and exact truths. For each method and replication $r ,$ the error on family $\mathcal { F }$ is the root mean squared error (RMSE) against these truths,

$$
\begin{array} { r } { \mathrm { R M S E } _ { \mathcal { F } } ^ { ( r ) } = \Big ( \frac { 1 } { \vert \mathcal { F } \vert } \sum _ { k \in \mathcal { F } } \big ( \widehat { \tau } _ { k } ^ { ( r ) } - \tau _ { k } \big ) ^ { 2 } \Big ) ^ { 1 / 2 } , } \end{array}\tag{12}
$$

where k indexes contrasts and $r$ indexes replications. We average family RMSE over ten replications. Tables 2 and 3 instead pool all seven contrasts. Empirical coverage is the fraction of 70 node-clustered nominal 95 percent intervals containing the truth. Because standard-error constructions difer, including adapted TL’s node-clustered dispersion of updated predictions, coverage is descriptive rather than a like-for-like inferential comparison or evidence of validity under evolving-graph dependence. Appendix D reports overlap diagnostics.

## 5.4 Main Results (RQ1, RQ2)

Table 1 separates same-estimand accuracy from restricteddesign scope. Among adapted TL, adapted GML-DR, and DynaNet-DR, which all target (A, Z, M), DynaNet-DR has the lowest unrounded mean RMSE in all 16 dataset-by-family cells; CollegeMsg PE is tied with adapted GML-DR at threedecimal precision. Its unweighted mean cellwise reduction relative to the better of TL and GML-DR is 25.9%, with turns PE as zero. DynInt is lowest in two displayed cells, but its representative continuous score and omission of M preclude full-profile comparison. DynaNet-DR’s nominal intervals attain empirical coverage from 0.91 to 1.00.

<table><tr><td>Dataset</td><td>Method</td><td>DE</td><td>PE</td><td>CNE</td><td>JCC</td><td>Coverage</td></tr><tr><td rowspan="7">CollegeMsg</td><td>Naive-A</td><td> $0 . 7 2 2 { \scriptstyle \pm 0 . 0 4 2 }$ </td><td>0.552±0.000</td><td>0.326±0.000</td><td> $0 . 1 5 7 { \pm } 0 . 0 4 3$ </td><td>0.00</td></tr><tr><td>Static-Z OR</td><td> $0 . 0 9 9 { \pm } 0 . 0 4 3$ </td><td> $0 . 3 5 4 { \pm } 0 . 0 2 9$ </td><td>0.326±0.000</td><td> $0 . 5 9 1 { \scriptstyle \pm 0 . 0 5 5 }$ </td><td>0.01</td></tr><tr><td> $_ { \mathrm { S t a t i c - Z D R } }$ </td><td> $0 . 1 0 3 { \pm } 0 . 0 2 4$ </td><td> $0 . 2 7 9 { \scriptstyle \pm 0 . 0 5 1 }$ </td><td> $0 . 3 2 6 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 5 0 3 { \scriptstyle \pm 0 . 0 5 4 }$ </td><td>0.10</td></tr><tr><td> $\mathrm { G P S ^ { \dagger } }$ </td><td> $0 . 1 4 7 { \scriptstyle \pm 0 . 0 2 5 }$ </td><td> $0 . 4 1 7 { \scriptstyle \pm 0 . 0 4 4 }$ </td><td>0.326±0.000</td><td> $0 . 8 7 8 { \scriptstyle \pm 0 . 0 5 6 }$ </td><td>0.00</td></tr><tr><td> $\mathrm { { D y n I n t } ^ { \dag } }$ </td><td> $0 . 0 6 9 { \pm } 0 . 0 3 2$ </td><td> $0 . 0 6 2 { \scriptstyle \pm 0 . 0 2 6 }$ </td><td> $0 . 3 2 6 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 2 5 4 { \pm } 0 . 0 5 3$ </td><td>0.04</td></tr><tr><td> $\mathrm { T L } ^ { \dagger }$ </td><td> $0 . 0 5 6 { \pm } 0 . 0 2 1$ </td><td> $0 . 0 5 3 { \scriptstyle \pm 0 . 0 3 4 }$ </td><td> $0 . 0 4 9 { \pm } 0 . 0 2 8$ </td><td> $0 . 0 4 4 { \pm } 0 . 0 2 9$ </td><td>0.17</td></tr><tr><td> ${ \bf G M L - D R } ^ { \dagger }$ </td><td> $0 . 0 5 2 { \pm } 0 . 0 2 9$ </td><td> $\mathbf { 0 . 0 5 1 } { \pm } \mathbf { 0 . 0 2 8 }$ </td><td> $0 . 0 4 4 { \pm } 0 . 0 3 2$ </td><td> $0 . 0 5 5 { \pm } 0 . 0 3 3$ </td><td>0.87</td></tr><tr><td rowspan="8">email-Eu</td><td> $\mathrm { D y n a N e t – D R }$ </td><td> $\mathbf { 0 . 0 4 9 } \pm \mathbf { 0 . 0 1 7 }$ </td><td> $\mathbf { 0 . 0 5 1 } \pm \mathbf { 0 . 0 2 7 }$ </td><td> $\mathbf { 0 . 0 3 8 { \pm 0 . 0 1 6 } }$ </td><td>0.028±0.020</td><td>0.91</td></tr><tr><td>Naive-A</td><td> $0 . 8 1 2 { \scriptstyle \pm 0 . 0 7 7 }$ </td><td>0.552±0.000</td><td>0.326±0.000</td><td>0.083±0.056</td><td>0.06</td></tr><tr><td>Static-Z OR</td><td> $0 . 0 8 9 { \pm } 0 . 0 4 0$ </td><td> $0 . 3 1 0 { \pm } 0 . 0 2 8$ </td><td> $0 . 3 2 6 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td>0.558±0.044</td><td>0.00</td></tr><tr><td> $_ { \mathrm { S t a t i c - Z D R } }$ </td><td> $0 . 0 9 7 { \scriptstyle \pm 0 . 0 3 3 }$ </td><td> $0 . 3 0 0 { \scriptstyle \pm 0 . 0 2 3 }$ </td><td> $0 . 3 2 6 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td>0.540±0.043</td><td>0.23</td></tr><tr><td> $\mathrm { G P S ^ { \dagger } }$ </td><td> $0 . 0 6 1 { \scriptstyle \pm 0 . 0 1 3 }$ </td><td>0.351±0.036</td><td>0.326±0.000</td><td>0.689±0.045</td><td>0.01</td></tr><tr><td> $\mathrm { { D y n I n t } ^ { \dag } }$ </td><td> $0 . 0 6 8 { \pm } 0 . 0 3 9$ </td><td>0.039±0.027</td><td> $0 . 3 2 6 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td>0.250±0.030</td><td>0.11</td></tr><tr><td> $\mathrm { T L } ^ { \dagger }$ </td><td> $0 . 0 7 9 { \scriptstyle \pm 0 . 0 4 1 }$ </td><td>0.092±0.050</td><td> $0 . 0 8 9 { \pm } 0 . 0 4 3$ </td><td> $0 . 0 8 2 { \scriptstyle \pm 0 . 0 4 1 }$ </td><td>0.07</td></tr><tr><td> ${ \bf G M L - D R } ^ { \dagger }$ </td><td> $0 . 0 8 3 { \scriptstyle \pm 0 . 0 4 3 }$ </td><td>0.076±0.028</td><td> $0 . 0 7 9 { \scriptstyle \pm 0 . 0 3 4 }$ </td><td> $0 . 0 5 9 { \pm } 0 . 0 3 3$ </td><td>0.99</td></tr><tr><td rowspan="8">PrimarySchool</td><td>DynaNet-DR</td><td> $\mathbf { 0 . 0 2 6 { \overset { . } { = } } 0 . 0 1 9 }$ </td><td>0.064±0.038</td><td> $\mathbf { 0 . 0 4 2 { \scriptstyle \pm 0 . 0 1 7 } }$ </td><td> $\mathbf { 0 . 0 5 4 } \pm \mathbf { 0 . 0 4 7 }$ </td><td>1.00</td></tr><tr><td>Naive-A</td><td>0.633±0.120</td><td>0.552±0.000</td><td> $0 . 3 2 6 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 2 4 7 { \scriptstyle \pm 0 . 1 2 1 }$ </td><td>0.03</td></tr><tr><td> $_ { \mathrm { S t a t i c - Z O R } }$ </td><td> $0 . 0 8 8 { \pm } 0 . 0 4 7$ </td><td>0.276±0.076</td><td> $0 . 3 2 6 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 5 2 5 { \pm } 0 . 1 0 8$ </td><td>0.04</td></tr><tr><td> $_ { \mathrm { S t a t i c - Z D R } }$ </td><td> $0 . 0 8 9 { \pm } 0 . 0 4 2$ </td><td> $0 . 2 6 7 { \scriptstyle \pm 0 . 0 6 7 }$ </td><td> $0 . 3 2 6 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 5 3 6 { \pm } 0 . 1 0 9$ </td><td>0.41</td></tr><tr><td> $\mathrm { G P S ^ { \dagger } }$ </td><td> $0 . 2 6 7 { \scriptstyle \pm 0 . 0 7 4 }$ </td><td> $0 . 5 1 4 { \scriptstyle \pm 0 . 0 4 2 }$ </td><td> $0 . 3 2 6 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $1 . 0 9 2 { \scriptstyle \pm 0 . 0 8 5 }$ </td><td>0.00</td></tr><tr><td> $\mathrm { { D y n I n t } ^ { \dag } }$ </td><td> $0 . 0 8 0 { \scriptstyle \pm 0 . 0 4 1 }$ </td><td> $0 . 1 5 6 { \pm } 0 . 0 8 4$ </td><td> $0 . 3 2 6 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 2 6 3 { \scriptstyle \pm 0 . 1 7 6 }$ </td><td>0.04</td></tr><tr><td> $\mathrm { T L } ^ { \dagger }$ </td><td> $0 . 1 6 5 { \scriptstyle \pm 0 . 0 8 7 }$ </td><td> $0 . 2 0 3 { \pm } 0 . 1 4 8$ </td><td> $0 . 2 0 8 { \pm } 0 . 1 5 3$ </td><td> $0 . 2 1 7 { \scriptstyle \pm 0 . 2 2 8 }$ </td><td>0.09</td></tr><tr><td> ${ \bf G M L - D R } ^ { \dagger }$ </td><td> $0 . 1 4 6 { \pm } 0 . 0 6 3$ </td><td> $0 . 1 7 6 { \scriptstyle \pm 0 . 0 9 4 }$ </td><td> $0 . 1 9 2 { \pm } 0 . 1 3 1$ </td><td> $0 . 1 8 0 { \scriptstyle \pm 0 . 1 4 1 }$ </td><td>0.93</td></tr><tr><td rowspan="8">HighSchool</td><td> $\mathrm { D y n a N e t – D R }$ </td><td> $\mathbf { 0 . 0 6 4 } \pm \mathbf { 0 . 0 3 5 }$ </td><td> $\mathbf { 0 . 1 4 7 \pm 0 . 0 7 4 }$ </td><td> $\mathbf { 0 . 0 9 6 { \scriptstyle \pm 0 . 0 4 4 } }$ </td><td> $\mathbf { 0 . 1 5 4 } \pm \mathbf { 0 . 1 } 2 5$ </td><td>0.99</td></tr><tr><td>Naive-A</td><td> $0 . 7 8 4 { \pm } 0 . 1 0 1$ </td><td>0.552±0.000</td><td> $0 . 3 2 6 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 1 2 3 { \scriptstyle \pm 0 . 0 6 0 }$ </td><td>0.09</td></tr><tr><td> $_ { \mathrm { S t a t i c - Z O R } }$ </td><td> $0 . 0 9 9 { \pm } 0 . 0 4 9$ </td><td> $0 . 3 0 9 { \scriptstyle \pm 0 . 0 7 5 }$ </td><td> $0 . 3 2 6 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 5 7 3 { \scriptstyle \pm 0 . 1 0 5 }$ </td><td>0.01</td></tr><tr><td> $_ { \mathrm { S t a t i c - Z D R } }$ </td><td> $0 . 1 1 8 { \pm } 0 . 0 6 6$ </td><td> $0 . 2 6 7 { \scriptstyle \pm 0 . 0 4 2 }$ </td><td> $0 . 3 2 6 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 4 7 4 { \scriptstyle \pm 0 . 1 1 6 }$ </td><td>0.31</td></tr><tr><td> $\mathrm { G P S ^ { \dagger } }$ </td><td> $0 . 3 4 1 { \pm } 0 . 0 5 6$ </td><td> $0 . 4 9 2 { \scriptstyle \pm 0 . 0 5 7 }$ </td><td> $0 . 3 2 6 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $1 . 1 4 7 { \scriptstyle \pm 0 . 0 6 7 }$ </td><td>0.00</td></tr><tr><td> $\mathrm { { D y n I n t } ^ { \dag } }$ </td><td> $0 . 0 6 2 { \pm } 0 . 0 3 8$ </td><td> $0 . 1 1 5 { \pm } 0 . 0 9 4$ </td><td> $0 . 3 2 6 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 2 4 4 { \pm } 0 . 1 1 8$ </td><td>0.10</td></tr><tr><td> $\mathrm { T L } ^ { \dagger }$ </td><td> $0 . 1 1 3 { \pm } 0 . 0 5 4$ </td><td> $0 . 1 0 2 { \scriptstyle \pm 0 . 0 5 2 }$ </td><td> $0 . 1 0 6 { \pm } 0 . 0 6 3$ </td><td> $0 . 0 9 6 { \scriptstyle \pm 0 . 0 5 7 }$ </td><td>0.20</td></tr><tr><td> ${ \bf G M L - D R } ^ { \dagger }$ </td><td> $0 . 1 0 6 { \pm } 0 . 0 7 0$ </td><td> $0 . 0 9 1 { \scriptstyle \pm 0 . 0 5 9 }$ </td><td> $0 . 1 3 7 { \pm } 0 . 0 8 1$ </td><td> $0 . 0 8 5 { \pm } 0 . 0 5 4$ </td><td>0.94</td></tr><tr><td></td><td> $\mathrm { D y n a N e t – D R }$ </td><td> $\mathbf { 0 . 0 6 7 { \scriptstyle \pm 0 . 0 2 4 } }$ </td><td> $\mathbf { 0 . 0 8 7 } \pm \mathbf { 0 . 0 5 8 }$ </td><td> $\mathbf { 0 . 0 7 1 { \scriptstyle \pm 0 . 0 3 9 } }$ </td><td> $\mathbf { 0 . 0 7 7 { \scriptstyle \pm 0 . 0 5 1 } }$ </td><td>0.99</td></tr></table>

Table 1: RMSE by contrast family and empirical coverage of node-clustered nominal intervals. Bold marks the lowest RMSE among full-target implementations, and † denotes adaptations. Restricted designs have diferent targets, and coverage is descrip tive.

<table><tr><td>Variant</td><td>College</td><td>Email</td><td>Primary</td><td>High</td></tr><tr><td>Full  $\lambda = 1$ </td><td></td><td>.047±.011 .052±.021</td><td> $. 1 2 6 { \pm } . 0 4 5$ </td><td> $\mathbf { 0 8 3 } \pm \mathbf { . 0 2 7 }$ </td></tr><tr><td></td><td></td><td>.052±.022 .077±.021</td><td> $. 2 0 6 \pm . 1 2 3$ </td><td> $. 1 1 0 { \pm } . 0 4 0$ </td></tr><tr><td>Plug-in</td><td> $. 0 6 4 \pm . 0 1 9$ </td><td> $. 0 5 6 { \pm } . 0 2 0$ </td><td> $. 1 2 5 \pm . 0 4 5$ </td><td> $. 1 1 1 \pm . 0 5 0$ </td></tr><tr><td>No M</td><td> $. 1 9 9 { \pm } . 0 0 5$ </td><td> $. 2 0 3 { \pm } . 0 0 6$ </td><td> $. 2 3 1 { \pm } . 0 4 1$ </td><td> $. 2 0 2 { \pm } . 0 1 0$ </td></tr><tr><td>No  $M + \mathrm { p l u g - i n }$ </td><td> $. 2 0 6 { \pm } . 0 0 7$ </td><td> $. 2 0 4 { \pm } . 0 0 7$ </td><td> $. 2 3 1 { \pm } . 0 4 0$ </td><td> $. 2 1 9 { \pm } . 0 1 7$ </td></tr><tr><td> $\mathrm { N o - } M \mathrm { I P W }$ </td><td> $. 2 8 8 { \pm } . 0 5 8$ </td><td> $. 3 1 8 \pm . 0 4 4$ </td><td> $. 3 4 1 \pm . 1 1 1$ </td><td> $. 3 7 1 { \pm } . 0 8 8$ </td></tr><tr><td>Static exposure</td><td> $. 3 0 4 \pm . 0 2 5$ </td><td> $. 3 1 8 { \pm } . 0 1 5$ </td><td> $. 3 0 9 { \pm } . 0 4 0 $ </td><td> $. 2 9 9 { \pm } . 0 2 6$ </td></tr></table>

Table 2: Mean ± SD overall RMSE across seven contrasts. Full denotes DynaNet-DR with its adaptive gate. Bold marks the dataset minimum. The λ = 1 and plug-in rows share the full target; the other variants have target mismatch.

larger gains on email-Eu DE (0.026 versus 0.079) and PrimarySchool CNE (0.096 versus 0.192). These results concern the stated adaptations, not the original procedures.

The fixed λ = 1 ablation holds the target, nuisance learners, clipping, splits, and interval construction fixed, so its comparison with Full isolates the adaptive gate within the reported pipeline. Full has lower mean overall RMSE on all four datasets, although the CollegeMsg diference is small and mixed across replications. Relative to plug-in prediction, the implemented correction lowers RMSE on three datasets and is nearly unchanged on PrimarySchool. This benchmark comparison does not establish that the gate is optimal beyond the evaluated settings. Appendices D and H report the remaining profiles and paired Wilcoxon tests.

Five restricted rows return CNE as zero, and Naive-A re-

## 5.5 Nuisance-Model Stress Tests (RQ3)

Table 3 reports one-at-a-time nuisance-model restrictions for CollegeMsg and email-Eu. DynaNet-DR improves on the plug-in estimator under the outcome restriction but does not uniformly outperform TL or GML-DR; it performs best under the propensity restriction. These finite-sample stress tests do not verify Theorem 2; Appendix H specifies the restrictions.

![](images/c51f59ea6da7ab5632fd8e0de34193d5a8c334c9a74f7131ed87b0b763629362.jpg)  
Figure 3: Mean estimates and benchmark truths for seven contrasts on CollegeMsg and email-Eu. Error bars are one standard deviation across replications, not confidence intervals. Appendix B defines the targets; Appendix D reports the other networks.

<table><tr><td rowspan="2">Method</td><td colspan="3">CollegeMsg</td><td colspan="3">email-Eu</td></tr><tr><td>Default</td><td>Outcome</td><td>Prop.</td><td>Default</td><td>Outcome</td><td>Prop.</td></tr><tr><td>OR</td><td>.064 (.019) .288</td><td>.421 (.047) .288</td><td>.064 (.019) .305</td><td>.056 (.020) .318</td><td>.474 (.094) .318</td><td>.056 (.020)</td></tr><tr><td>No-M IPW</td><td>(.058) .054</td><td>(.058)</td><td>(.050)</td><td>(.044)</td><td>(.044)</td><td>.403 (.042)</td></tr><tr><td>TL</td><td>(.022) .053</td><td>.205 (.069)</td><td>.052 (.023)</td><td>.093 (.026)</td><td>.170 (.050)</td><td>.101 (.030)</td></tr><tr><td>GML-DR</td><td>(.023)</td><td>.242 (.060)</td><td>.051 (.020)</td><td>.081 (.023)</td><td>.222 (.052)</td><td>.088 (.027)</td></tr><tr><td>DynaNet</td><td>.047 (.011)</td><td>.213 (.069)</td><td>.046 (.014)</td><td>.052 (.021)</td><td>.277 (.047)</td><td>.053 (.018)</td></tr></table>

Table 3: Mean overall RMSE, with SD in parentheses. OR is the full-target plug-in outcome regression, and DynaNet denotes DynaNet-DR. Bold marks each minimum. Outcome and Prop. restrict one nuisance model; only No-M IPW has target mismatch.

As shown in Figure 4, DynaNet-DR changes little on three datasets across the clip range. Appendix G reports greater sensitivity for the restricted No-M IPW reference. This does not validate the propensity-correct branch under active clipping; Appendix E reports hidden-homophily sensitivity.

## 5.6 Illustrative MathOverflow Application

We use MathOverflow (Paranjape, Benson, and Leskovec 2017) as an observational illustration. In 30-day bins, A indicates whether a user answers at least one question, Z is the discretized decayed share of answering neighbors, M indicates whether the user interacts with at least one previously unseen neighbor at t + 1, and Y = log(1 + weighted degree) at t + 2. Adjustment uses prior network and activity summaries, but some variables share a bin with A and Z, leaving within-bin order unresolved; Appendix H gives details.

Averaged across nuisance-model replications, the two estimates in each coordinate-wise family are DE 0.00 and

![](images/3c1772afd948a688c39f5aea578a7e41e651c576822bfe457db52a04634752d9.jpg)  
Figure 4: DynaNet-DR overall RMSE and empirical coverage by clip bound. Bands show one standard deviation; dashed line marks c = 0.02.

0.14, PE −0.04 and −0.04, and CNE 0.37 and 0.46; JCC is 0.44, and the separate naive diference is 0.95. Without ground truth or a hidden-confounding sensitivity analysis, we interpret them as descriptive, representative-score, modeladjusted contrasts. They are not additive components, and the application is not causal validation.

## 6 Related Work

Network causal inference. Exposure mappings formalize interference, while network experiments address it by design (Hudgens and Halloran 2008; Aronow and Samii 2017; Eckles, Karrer, and Ugander 2017). Observational identification relies on strong adjustment assumptions (Shalizi and Thomas 2011; Forastiere, Airoldi, and Mealli 2021). Graphlearning and dynamic methods learn exposure mappings or use evolving graphs as predictors (Ma and Tresp 2021; Adhikari, Medya, and Zheleva 2025; Lin et al. 2026); recent work extends doubly robust estimation to networks (Chen et al. 2024; Khatami et al. 2025). We add a post-assignment summary while retaining pre-assignment history; Appendix I gives details.

## 7 Conclusion

DynaNet-DR estimates a controlled contrast profile using a temporally factorized propensity and normalized augmentation. Its canonical estimator is doubly robust under the stated conditions; stochastic graph interventions and cross-node inference remain future work.

## References

Adhikari, S.; Medya, S.; and Zheleva, E. 2025. Learning exposure mapping functions for inferring heterogeneous peer efects. arXiv:2503.01722.

Aronow, P. M.; and Samii, C. 2017. Estimating average causal efects under general interference, with application to a social network experiment. The Annals of Applied Statistics, 11(4): 1912–1947.

Bang, H.; and Robins, J. M. 2005. Doubly robust estimation in missing data and causal inference models. Biometrics, 61(4): 962–973.

Chen, S.; Hu, J.; and Jiang, Z. 2026. Connections as treatment: causal inference with edge interventions in networks. arXiv:2601.07267.

Chen, W.; Cai, R.; Yang, Z.; Qiao, J.; Yan, Y.; Li, Z.; and Hao, Z. 2024. Doubly robust causal efect estimation under networked interference via targeted learning. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, 6457–6485. PMLR.

Comola, M.; and Prina, S. 2021. Treatment efect accounting for network changes. The Review ofEconomics and Statistics, 103(3): 597–604.

Eckles, D.; Karrer, B.; and Ugander, J. 2017. Design and analysis of experiments in networks: reducing bias from interference. Journal ofCausal Inference, 5(1): 20150021.

Forastiere, L.; Airoldi, E. M.; and Mealli, F. 2021. Identification and estimation of treatment and interference efects in observational studies on networks. Journal ofthe American Statistical Association, 116(534): 901–918.

Fournet, J.; and Barrat, A. 2014. Contact patterns among high school students. PLoS ONE, 9(9): e107878.

Hernán, M. A.; and Robins, J. M. 2020. Causal inference: what if. Boca Raton: Chapman & Hall/CRC.

Hudgens, M. G.; and Halloran, M. E. 2008. Toward causal inference with interference. Journal of the American Statistical Association, 103(482): 832–842.

Imbens, G. W.; and Rubin, D. B. 2015. Causal inferencefor statistics, social, and biomedical sciences: an introduction. Cambridge: Cambridge University Press.

Khatami, S. B.; Parikh, H.; Chen, H.; Roy, S.; and Salimi, B. 2025. Graph machine learning based doubly robust estimator for network causal efects. In Proceedings of the 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings ofMachine Learning Research, 4366–4374. PMLR.

Leung, M. P. 2020. Treatment and spillover efects under network interference. The Review of Economics and Statistics, 102(2): 368–380.

Lin, X.; Bao, H.; Takeuchi, K.; Cui, Y.; and Kashima, H. 2026. Modeling dynamic interference for treatment efect estimation from dynamic graphs. ACM Transactions on Knowledge Discovery from Data, 20(3): 45:1–45:28. DOI: 10.1145/3796240.

Ma, Y.; and Tresp, V. 2021. Causal inference under networked interference and intervention policy enhancement. In Proceedings ofthe 24th International Conference on Artificial Intelligence and Statistics, volume 130 of Proceedings ofMachine Learning Research, 3700–3708. PMLR.

Manski, C. F. 1993. Identification of endogenous social efects: the reflection problem. The Review of Economic Studies, 60(3): 531–542.

Ogburn, E. L.; Sofrygin, O.; Díaz, I.; and van der Laan, M. J. 2024. Causal inference for social network data. Journal of the American Statistical Association, 119(545): 597–611.

Panzarasa, P.; Opsahl, T.; and Carley, K. M. 2009. Patterns and dynamics of users’ behavior and interaction: network analysis of an online community. Journal of the American Society for Information Science and Technology, 60(5): 911– 932.

Paranjape, A.; Benson, A. R.; and Leskovec, J. 2017. Motifs in temporal networks. In Proceedings of the Tenth ACM International Conference on Web Search and Data Mining, 601–610.

Robins, J. 1986. A new approach to causal inference in mortality studies with a sustained exposure period: application to control of the healthy worker survivor efect. Mathematical Modelling, 7(9-12): 1393–1512.

Sävje, F. 2024. Causal inference with misspecified exposure mappings: separating definitions and assumptions. Biometrika, 111(1): 1–15.

Shalizi, C. R.; and Thomas, A. C. 2011. Homophily and contagion are generically confounded in observational social network studies. Sociological Methods & Research, 40(2): 211–239.

Stehlé, J.; Voirin, N.; Barrat, A.; Cattuto, C.; Isella, L.; Pinton, J.-F.; Quaggiotto, M.; Van den Broeck, W.; Régis, C.; Lina, B.; and Vanhems, P. 2011. High-resolution measurements of face-to-face contact patterns in a primary school. PLoS ONE, 6(8): e23176.

van der Laan, M. J.; and Rubin, D. 2006. Targeted maximum likelihood learning. The International Journal of Biostatistics, 2(1): Article 11.