# Variational Approach to Optimal IPS Estimator for Multi-logger Off-Policy Evaluation

Joon Suk Huh<sup>∗</sup> Department of Computer Science University of Wisconsin–Madison, USA joon@cs.wisc.edu

Junghoon Seo<sup>∗</sup> PIT IN Corp., Republic of Korea sjh@pitin-ev.com

## Abstract

We study off-policy evaluation (OPE) in contextual bandits with data collected from multiple logging policies. Inverse propensity scoring (IPS) is a standard approach to OPE and extends naturally to the multi-logger setting. However, as highlighted by Agarwal et al. [2017], neither balanced nor weighted IPS consistently outperforms the other in this setting. We resolve this dilemma by deriving an optimal IPS estimator with sample-dependent weights that minimize variance subject to unbiasedness. Using a variational calculus approach, we obtain closed-form optimal weights, yielding an estimator that is unbiased and achieves asymptotically optimal variance within a generalized weighted-IPS estimator class. Instead of a reward model, the estimator needs only a K × K matrix and a K-vector of moments, where K is the number of logging policies, and we quantify how much efficiency this costs relative to the semiparametric oracle. Experiments on benchmark datasets show that our estimator generally improves on existing multi-logger IPS methods. We also extend our estimator to a doubly robust form by incorporating a conditional reward estimator. We compare the resulting DR extension with the semiparametrically efficient DR estimator of Kallus et al. [2021], both theoretically and empirically. We show that the two estimators achieve the same asymptotic variance when the conditional reward estimator is consistent, while our estimator can retain semiparametric efficiency under certain forms of reward misspecification. Empirically, the proposed DR estimator often achieves lower estimation error than the existing DR baseline.

## 1 Introduction

Off-policy evaluation (OPE) is a core tool for data-driven decision making, with applications in recommender systems, healthcare, and advertising. The goal is to estimate the value of an evaluation policy using data collected by one or more different logging policies. While the single-logger setting is well understood, many practical datasets are collected from multiple logging policies, for example through heterogeneous systems, parallel deployments, or sequential changes to a production policy [Agarwal et al., 2017, Saito et al., 2021, Jung and Bellot, 2024]. This multi-logger structure creates a basic statistical challenge: different strata can have very different overlaps with the evaluation policy, so combining them poorly can lead to large variance.

Inverse propensity scoring (IPS) is a standard approach to OPE [Rosenbaum and Rubin, 1983, Precup et al., 2000, Bottou et al., 2013]. In the single-logger case, IPS reweights each observed reward by the propensity ratio between the evaluation and logging policies. In the multi-logger case, however, there is an additional design choice: one must decide how much each logger should contribute to the final estimate. Existing approaches answer this question in different ways. Naive IPS treats all logged samples with their own propensity ratios, balanced IPS uses the sample-size-weighted average logging policy, and weighted IPS chooses logger-level weights based on estimated variances [Agarwal et al., 2017]. As observed by Agarwal et al. [2017], the relative performance of balanced and weighted IPS is highly instance dependent: neither consistently dominates the other across problem instances, and naive IPS is dominated in their analysis. Kallus et al. [2021] later characterized the semiparametrically efficient estimator over a broader class, which serves as our benchmark.

We revisit this problem from a variational perspective. Instead of choosing among a fixed set of IPS estimators, we consider a generalized weighted-IPS class $\Gamma _ { \mathrm { w I P S } }$ in which each sample may receive a weight depending on the logged tuple and its logger identity. That is, the weights are functions of the state, action, reward, and logger. We impose constraints that ensure unbiasedness and then directly minimize the variance functional over all feasible weight functions. This yields a convex variance minimization problem whose solution admits a closed-form characterization. The resulting estimator, denoted $\hat { J } _ { \mathrm { o I P S } }$ , can be viewed as a balanced-IPS estimator augmented with an optimal control-variate term built from the logging propensities. This construction does not require the conditional mean reward. Instead, it estimates a $\dot { K } \times K$ matrix T and a vector c $\in \mathbb { R } ^ { K }$ of moments (§3).

This perspective clarifies the apparent instance-dependence of existing IPS estimators. The optimal weights depend on the interaction among the evaluation policy, the logging policies, the reward distribution, and the stratum sizes. Thus, fixed IPS rules such as naive IPS, balanced IPS, and weighted IPS can each perform well on some instances and poorly on others. By optimizing over a class containing these IPS baselines, $\hat { J } _ { \mathrm { o I P S } }$ adapts to the instance and achieves the smallest asymptotic variance within $\Gamma _ { \mathrm { w I P S } }$ . We prove that the corresponding feasible cross-fitted implementation is unbiased when logging policies are known and is asymptotically normal with the optimal variance.

We also relate our estimator to the semiparametric efficiency theory of Kallus et al. [2021]. Their framework shows that the most efficient estimator uses the conditional mean reward as a control variate. Our IPS estimator can be viewed as a restricted, reward model-free version of this idea: instead of estimating the conditional mean reward directly, it constructs the best control variate available from the logging policies. When this logger-induced control variate is rich enough to represent the relevant reward structure, our estimator asymptotically recovers the efficient oracle. Otherwise, its excess variance is determined by the part of the reward structure that cannot be represented by the logging policies.

Finally, we extend our variational approach to a doubly robust estimator, denoted $\hat { J } _ { \mathrm { D R - o I P S } }$ , by incorporating a reward model. This extension keeps the same idea as our IPS estimator: it uses the logging policies to choose an additional variance-reducing correction. This makes it directly comparable to the doubly robust balanced IPS estimator $\hat { J } _ { \mathrm { D R - b I P S } }$ of Kallus et al. [2021]. When the reward model is correctly specified, both estimators attain the same asymptotic variance. When the reward model is misspecified, however, $\hat { J } _ { \mathrm { D R - o I P S } }$ can remain efficient if the misspecification is representable by the logging policies, whereas $\hat { J } _ { \mathrm { D R - b I P S } }$ need not. This gives our DR extension an additional robustness advantage.

## Summary of Contributions. We make the following contributions:

1. A variationally optimal IPS estimator for multi-logger OPE. We introduce a generalized weighted-IPS class $\Gamma _ { \mathrm { w I P S } }$ and derive the variance-minimizing unbiased estimator within this class. The resulting estimator $\hat { J } _ { \mathrm { o I P S } }$ has closed-form sample-dependent weights and estimates T and c in place of a reward model.

2. Closed-form characterization, feasible estimation, and projection interpretation. We show that the optimal weights are characterized by a linear system $\mathbf { T } \alpha = \mathbf { c } ,$ whose components can be estimated from data, leading to a practical cross-fitted implementation. We further show that this system has a geometric interpretation: it projects the reward-weighted evaluation policy onto the span of the logging policies, and the resulting projection residual exactly determines the variance gap to the semiparametrically efficient oracle.

3. Doubly robust extension and comparison with efficient DR estimators. We extend $\hat { J } _ { \mathrm { o I P S } }$ to a doubly robust estimator $\scriptstyle { \hat { J } } _ { \mathrm { D R - o I P S } }$ by applying the same variational construction to residuals based on ${ \hat { q } } .$ We show that $\hat { J } _ { \mathrm { D R - o I P S } }$ matches the asymptotic variance of $\hat { J } _ { \mathrm { D R - b I P S } }$ when $\hat { q }$ is consistent, and can retain the oracle variance under certain structured misspecifications of $\hat { q } .$

4. Empirical validation. Across benchmark datasets and a broad range of logger mixtures, our IPS estimator generally improves over existing multi-logger IPS baselines. The doubly robust extension also often achieves lower estimation error than the semiparametrically efficient baseline.

Related Work. Appendix G discusses related work on OPE with multiple loggers [He et al., 2019, Chen et al., 2020], connections to multiple importance sampling [Veach and Guibas, 1995, Elvira et al., 2019, Kondapaneni et al., 2019], and broader variance-reduction methods for IPS.

## 2 Preliminaries

This section introduces the problem setup and reviews baseline estimators used in our experiments.

Notation. For $n \in \mathbb N$ , let $[ n ] : = \{ 1 , \dots , n \}$ . We use boldface for finite-dimensional vectors and matrices, $\mathbf { e } . \mathbf { g } . , \mathbf { v } \in \mathbb { R } ^ { n }$ and $\mathbf { \bar { M } } \in \mathbf { \mathbb { R } } ^ { n \times n }$ . Their entries are written in plain font $\left( \mathrm { e . g . , } v _ { i }\right.$ and $M _ { i j } )$ and we likewise use plain symbols for scalar components such as $\alpha _ { i } , c _ { i }$ , and $w _ { i }$ . The symbol $\mathbb { 1 } [ \cdot ]$ denotes the indicator function.

Let $( \Omega , { \mathcal { F } } , \mu )$ be a σ-finite measure space and let $f , g \in L ^ { 2 } ( \mu )$ . We write $\langle \cdot , \cdot \rangle _ { \mu }$ for the $L ^ { 2 } ( \mu )$ inner product:

$$
\langle f , g \rangle _ { \mu } : = \int _ { \Omega } f ( \omega ) g ( \omega ) d \mu ( \omega ) .
$$

The associated $L ^ { 2 } ( \mu )$ -norm is $\| f \| _ { L ^ { 2 } ( \mu ) } : = \langle f , f \rangle _ { \mu } ^ { 1 / 2 }$ . We use the same notation $\left. f , g \right. _ { \mu }$ whenever $f g$ is µ-integrable, even if f or g is not in $L ^ { 2 } ( \mu )$

Throughout the paper, we fix a σ-finite measure space $( \Omega , { \mathcal { F } } , \mu )$ , where $\Omega : = \mathcal { S } \times \mathcal { A } \times \mathcal { R }$ , with $s , A .$ , and R denoting the sets of states, actions, and rewards, respectively, as defined in §2.1. For $\omega = ( s , a , r ) \in \Omega , \operatorname { l e t } \rho _ { r } ( \omega ) : = r$ denote the projection onto the reward coordinate; R denotes the corresponding random variable. Here, F is a suitable σ-algebra and $\mu$ is a fixed reference measure. For brevity, we write $L ^ { 2 } : = L ^ { 2 } ( \mu , \Omega )$ . For any measure $\nu \ll \mu$ , we write $L ^ { 2 } ( \nu )$ for the space of square-integrable functions with respect to ν. When Ω is a topological space, we denote by supp(µ) the support of a measure $\mu ,$ , and by $\operatorname { s u p p } ( f )$ the support of a measurable real-valued function f. For a policy π, we write supp $( \pi ( \cdot | s ) ) : = \operatorname { c l } \left( { \big \{ } a \in { \mathcal { A } } : \pi ( a \mid s ) > 0 { \big \} } \right)$ . For measurable functions f and g with $g \neq 0$ wherever $f \neq 0$ , we define the ratio $f / g$ pointwise, with the convention that $0 / 0 = 0$

## 2.1 Problem Setting

We study OPE in contextual bandits, following the multi-logger setup in [Agarwal et al., 2017, He et al., 2019, Kallus et al., 2021]. A policy π maps a state (i.e., context) $s \in S$ to a distribution over actions in ${ \mathcal { A } } ,$ written as $\pi ( a \mid s )$

An environment is defined by the following unknown distributions: a context distribution $p _ { s } ( s )$ and reward distributions $p _ { r } ( r \mid s , a )$ over $\mathcal { R } \subseteq \mathbb { R }$ for each context-action pair $( s , a ) \in \mathcal { S } \times$ ${ \mathcal { A } } .$ Interaction proceeds as follows: the environment samples $S \sim p _ { s } ,$ , the actor samples $A \sim$ $\pi ( \cdot \mid S )$ , and the environment then samples the reward $R \sim p _ { r } ( \cdot \mid S , A )$ . This induces a joint distribution $\mu _ { \pi }$ on $\mathcal { S } \times \mathcal { A } \times \mathcal { R }$ , whose density is $p _ { s } ( s ) \pi ( a | s ) p _ { r } ( r | s , a )$ with respect to the dominating measure $\mu ( d s , d a , d r ) = \mu _ { s } ( d s ) \mu _ { a } ( d a ) \mu _ { r } ( d r \mid s , a )$ , where $\mu _ { r } ( \cdot \mid s , a )$ is a reference kernel on R. Equivalently, $\mu _ { \pi } ( d s , d a , d r ) = P _ { S } ( d s ) \pi ( d a \mid s ) P _ { R } ( d r \mid s , a )$ , where $P _ { S } ( d s ) = p _ { s } ( s ) \mu _ { s } ( d s )$ $\pi ( d a \mid s ) = \pi ( a \mid s ) \bar { \mu } _ { a } ( d a )$ , and $P _ { R } ( d r \mid s , a ) = p _ { r } ( r \mid s , a ) \mu _ { r } ( d r \mid s , a )$ . The expected reward of the policy π, denoted $J ( \pi )$ , is

$$
J ( \pi ) : = \mathbb { E } _ { \mu _ { \pi } } [ R ] ,
$$

where $\mathbb { E } _ { \mu _ { \pi } }$ denotes expectation with respect to $\mu _ { \pi }$

The goal of OPE is to estimate the policy value $J ( \pi _ { e } )$ of an evaluation policy $\pi _ { e }$ from logged data. In the single-logger case, we observe i.i.d. samples $\mathcal { D } : = \left\{ ( S _ { j } , A _ { j } , R _ { j } ) \right\} _ { j = 1 } ^ { n }$ drawn from $\mu _ { \pi }$ , where π is the logging policy and typically $\pi \neq \pi _ { e }$ . In this work, we consider stratified OPE with multiple logging policies $( \pi _ { i } ) _ { i \in [ K ] }$ . The full dataset is $\mathcal { D } : = \{ \mathcal { D } _ { i } \} _ { i \in [ K ] }$ , where the strata are independent and each stratum $\mathcal { D } _ { i }$ consists of $n _ { i } \thinspace \mathrm { i . i . d . }$ . samples from $\mu _ { \pi _ { i } }$ , namely $\mathcal { D } _ { i } : = \big \{ ( S _ { i j } , A _ { i j } , R _ { i j } ) \big \} _ { j = 1 } ^ { n _ { i } }$ , where $S _ { i j } \sim p _ { s } , A _ { i j } \sim \pi _ { i } ( { \bf \cdot } \mid S _ { i j } )$ , and $R _ { i j } \sim p _ { r } ( { \bf \cdot } \mid S _ { i j } , A _ { i j } )$

## 2.2 Existing Inverse Propensity Scoring (IPS)-based Approaches

Naive IPS. Naive IPS reweights each observed reward by the propensity ratio between the evaluation and logging policies:

$$
\hat { J } _ { \mathrm { I P S } } ( \pi _ { e } ; \mathcal { D } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { K } \sum _ { ( S , A , R ) \in \mathcal { D } _ { i } } R \frac { \pi _ { e } ( A \mid S ) } { \pi _ { i } ( A \mid S ) } ,\tag{2.1}
$$

where $\textstyle n : = \sum _ { i = 1 } ^ { K } n _ { i }$ is the total number of samples.

Balanced IPS. Agarwal et al. [2017] proposed a multiple importance sampling (MIS)-inspired estimator that uses the averaged logging policy as inverse propensity score:

$$
\hat { J } _ { \mathrm { b I P S } } ( \pi _ { e } ; \mathcal { D } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { K } \sum _ { ( S , A , R ) \in \mathcal { D } _ { i } } R \frac { \pi _ { e } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ( A \mid S ) } ,\tag{2.2}
$$

where $\begin{array} { r } { \pi _ { \mathrm { a v g } } : = \frac { 1 } { n } \sum _ { i = 1 } ^ { K } n _ { i } \pi _ { i } } \end{array}$ and $n _ { i } : = | \mathcal { D } _ { i } |$

Weighted IPS. Agarwal et al. [2017] also introduced weighted IPS, which combines logger-specific IPS estimates using variance-based weights:

$$
\hat { J } _ { \mathrm { w I P S } } ( \pi _ { e } ; \mathcal { D } ) = \sum _ { i = 1 } ^ { K } \hat { \gamma } _ { i } \sum _ { ( S , A , R ) \in \mathcal { D } _ { i } } R \frac { \pi _ { e } ( A \mid S ) } { \pi _ { i } ( A \mid S ) } ,\tag{2.3}
$$

where $\begin{array} { r } { \hat { \gamma } _ { i } : = \left( \widehat { \sigma } _ { r } ^ { 2 } ( \pi _ { e } | | \pi _ { i } ) \sum _ { j = 1 } ^ { K } \frac { n _ { j } } { \widehat { \sigma } _ { r } ^ { 2 } ( \pi _ { e } | | \pi _ { j } ) } \right) ^ { - 1 } } \end{array}$ and $\hat { \sigma } _ { r } ^ { 2 } ( \pi _ { e } | | \pi _ { i } )$ is a positive consistent estimator of $\begin{array} { r } { \operatorname { V a r } _ { \mu _ { \pi _ { i } } } \left[ R \frac { \pi _ { e } ( A | S ) } { \pi _ { i } ( A | S ) } \right] } \end{array}$ , which quantifies the divergence between the target and logging policies.

Doubly Robust Balanced IPS. Kallus et al. [2021] proposed a doubly robust version of $\hat { J } _ { \mathrm { b I P S } }$ using a conditional mean reward estimator $\hat { q } \colon$

$$
\hat { J } _ { \mathrm { D R - b I P S } } ( \pi _ { e } ; \mathcal { D } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { K } \sum _ { ( S , A , R ) \in \mathcal { D } _ { i } } \left[ \frac { \pi _ { e } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ( A \mid S ) } \big ( R - \hat { q } ( S , A ) \big ) + \hat { q } ( S , \pi _ { e } ) \right] ,\tag{2.4}
$$

where $\begin{array} { r } { \hat { q } ( S , \pi _ { e } ) : = \int _ { A } \hat { q } ( S , a ) \pi _ { e } ( a \mid S ) d \mu _ { a } ( a ) } \end{array}$ . When the logging policies are known and $\hat { q }$ consistently estimates the conditional mean reward $q ( s , a ) : = \mathbb { E } [ R \mid S = s , A = a ] , \hat { J } _ { \mathrm { D R \mathrm { - } b I P S } }$ attains the semiparametric efficiency bound within the estimator class studied by Kallus et al. [2021], under their regularity conditions. In practice, $\hat { q }$ is estimated by cross-fitting: the reward model is trained on one part of the data and evaluated on held-out samples. The same cross-fitting strategy for nuisance quantities is also used in $\hat { J } _ { \mathrm { w I P S } }$

## 3 Variational Optimization Approach

In this section, we introduce our main estimator, which is derived by optimizing a convex variance functional and is asymptotically optimal within the generalized weighted-IPS class denoted by $\Gamma _ { \mathrm { w I P S } }$

## 3.1 Generalized Weighted-IPS Estimator Class

Let $\mathbf { w } : = ( w _ { i } ) _ { i \in [ K ] }$ , where each $w _ { i } : S \times \mathcal { A } \times \mathcal { R }  \mathbb { R }$ is a sample-dependent weight function. The generalized weighted-IPS estimator class $\Gamma _ { \mathrm { w I P S } }$ is

$$
\hat { J } ( \pi _ { e } , \mathbf { w } ; \mathcal { D } ) : = \sum _ { i = 1 } ^ { K } \frac { 1 } { n _ { i } } \sum _ { ( S , A , R ) \in \mathcal { D } _ { i } } w _ { i } ( S , A , R ) \frac { \pi _ { e } ( A \mid S ) } { \pi _ { i } ( A \mid S ) }\tag{3.1}
$$

Estimators in $\Gamma _ { \mathrm { w I P S } }$ are unbiased when the weights satisfy the following conditions for every $\boldsymbol { \omega } = ( s , a , r )$ :

Normalization: $\sum _ { i = 1 } ^ { K } w _ { i } ( \omega ) = \rho _ { r } ( \omega )$ Dominance: $\pi _ { i } ( a | s ) = 0 \implies w _ { i } ( s , a , r ) = 0 .$

Theorem 3.1. [Unbiasedness] For any weights $\mathbf { w } \in ( L ^ { 2 } ) ^ { K }$ that satisfy the normalization and dominance conditions and for which each weighted score $w _ { i } ( S , A , R ) { \stackrel {  } { \pi } } { } _ { e } ( A \mid S ) / \pi _ { i } ( A \mid S )$ is µ<sub>π</sub> -integrable, $\hat { J } ( \pi _ { e } , \mathbf { w } ; \mathcal { D } )$ is unbiased, that is, $J ( \pi _ { e } ) = \mathbb { E } [ \hat { J } ( \pi _ { e } , \mathbf { w } ; \mathcal { D } ) ]$ , where the expectation is over the sampling randomness of logged data $\mathcal { D } : = ( \mathcal { D } _ { i } ) _ { i \in [ K ] } .$

In $\ S 4 . 3 .$ , we compare $\Gamma _ { \mathrm { w I P S } }$ with the broader class of Kallus et al. [2021], whose efficient estimator requires an estimate of the conditional mean reward.

Variationally Optimized Weights. Since $\hat { J } ( \pi _ { e } , \mathbf { w } ; \mathcal { D } )$ is a linear functional of $\mathbf { w } .$ its variance is convex. We optimize this variance functional directly over feasible w satisfying the normalization and dominance constraints. The resulting optimal weight vector $\mathbf { w } ^ { \star } : = ( w _ { i } ^ { \star } ) _ { i \in [ K ] }$ , whose optimality is established in Theorem 4.1, takes the form

$$
w _ { i } ^ { \star } ( s , a , r ) = \frac { \alpha _ { i } ^ { \star } \pi _ { i } ( a | s ) } { \pi _ { e } ( a | s ) } + \frac { n _ { i } \pi _ { i } ( a | s ) } { n \pi _ { \mathrm { a v g } } ( a | s ) } \left( r - \frac { \sum _ { j = 1 } ^ { K } \alpha _ { j } ^ { \star } \pi _ { j } ( a | s ) } { \pi _ { e } ( a | s ) } \right)\tag{3.2}
$$

where ${ \pmb { \alpha } } ^ { \star } : = ( { \pmb { \alpha } } _ { i } ^ { \star } ) _ { i \in [ K ] }$ is a solution of $\mathbf { T } \alpha = \mathbf { c } .$ , with $\mathbf { T } \in \mathbb { R } ^ { K \times K }$ and $\mathbf { c } \in \mathbb { R } ^ { K }$ whose components are given by

$$
T _ { i j } : = \mathbb { E } _ { \mu _ { \tau _ { i } } } \left[ \frac { \pi _ { j } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ( A \mid S ) } \right] , c _ { i } : = \mathbb { E } _ { \mu _ { \tau _ { i } } } \left[ R \frac { \pi _ { e } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ( A \mid S ) } \right]\tag{3.3}
$$

We call T the technique matrix and c the contribution vector. Estimating T and c thus replaces estimating the conditional mean reward function. We later show in this section that the equation ${ \bf T } \alpha = { \bf c }$ can be interpreted as the normal equation for an orthogonal projection.

Control-Variate Form. Plugging the expression for $\mathbf { w } ^ { \star }$ in $( 3 . 2 )$ into $\hat { J } ( \pi _ { e } , \mathbf { w } ^ { \star } ; \mathcal { D } )$ in (3.1) yields the following form of the variationally optimal IPS estimator <sup>2</sup>:

$$
\boxed { \hat { J } ( \pi _ { e } , \alpha ^ { \star } ; \mathcal { D } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { K } \sum _ { ( S , A , R ) \in \mathcal { D } _ { i } } \left[ \frac { R \pi _ { e } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ( A \mid S ) } + \sum _ { k = 1 } ^ { K } \alpha _ { k } ^ { \star } - \frac { g _ { \alpha ^ { \star } } ( S , A ) } { \pi _ { \mathrm { a v g } } ( A \mid S ) } \right] }\tag{3.4}
$$

where $\begin{array} { r } { g _ { \alpha ^ { \star } } ( s , a ) : = \sum _ { k \in [ K ] } \alpha _ { k } ^ { \star } \pi _ { k } ( a | s ) } \end{array}$ . We refer to this as the control-variate form: the term involving $\begin{array} { r } { \sum _ { k } \alpha _ { k } ^ { \star } - g _ { \alpha ^ { \star } } / \bar { \pi } _ { \mathrm { a v g } } } \end{array}$ is mean-zero and therefore acts as a control variate.

Geometric View. Let $q ( s , a ) : = \mathbb { E } [ R \mid S = s , A = a ]$ denote the conditional mean reward. Then, we can view $g _ { \alpha ^ { \star } }$ as the orthogonal projection of $q \pi _ { e }$ onto the finite-dimensional space spanned by $\{ \pi _ { i } \} _ { i \in [ K ] }$ . Specifically, consider $L ^ { \hat { 2 } } ( \mu _ { \mathrm { l o g } } , S \times \mathcal { A } )$ , where $\mu _ { \mathrm { l o g } }$ is given by

$$
d \mu _ { \mathrm { l o g } } ( s , a ) : = \frac { p _ { s } ( s ) } { \pi _ { \mathrm { a v g } } ( a | s ) } d \mu _ { s , a } ( s , a ) ,
$$

with $\mu _ { s , a } = \mu _ { s } \otimes \mu _ { a }$ denoting the base measure over $\boldsymbol { s } \times \boldsymbol { A }$ . Equip $L ^ { 2 } ( S \times \mathcal { A } , \mu _ { \mathrm { l o g } } )$ with the inner product $\langle \cdot , \cdot \rangle _ { \mu _ { \mathrm { l o g } } }$ . Then, one can rewrite (3.3) in terms of $\langle \cdot , \cdot \rangle _ { \mu _ { \mathrm { l o g } } } \mathrm { a s }$

$$
T _ { i j } = \big \langle \pi _ { i } , \pi _ { j } \big \rangle _ { \mu _ { \log } } , \quad c _ { i } = \big \langle q \pi _ { e } , \pi _ { i } \big \rangle _ { \mu _ { \log } } .
$$

Thus, $\mathbf { T } \alpha ^ { \star } = \mathbf { c }$ if and only if $\begin{array} { r } { \left. q \pi _ { e } - \sum _ { j } \alpha _ { j } ^ { \star } \pi _ { j } , \pi _ { i } \right. _ { \mu _ { \mathrm { l o g } } } = 0 } \end{array}$ for all $i \in [ K ]$ . This shows that $g _ { \alpha ^ { \star } }$ is the projection of $q \pi _ { e }$ onto span $\left( \{ \pi _ { i } \} _ { i \in [ K ] } \right)$ , with respect to the norm induced by $\langle \cdot , \cdot \rangle _ { \mu _ { \mathrm { l o g } } }$

We formally state this observation in Lemma E.2. Later, in $\ S 4 . 3 .$ , we relate the squared $\| \cdot \| _ { \mu _ { \log } }$ norm of the residual component of $q \pi _ { e }$ orthogonal to span $\left( \{ \pi _ { i } \} _ { i \in [ K ] } \right)$ to the excess variance of our estimator relative to the semiparametrically efficient oracle of Kallus et al. [2021].

## 3.2 Feasible Variationally Optimal Estimator $\hat { J } _ { \mathrm { o I P S } }$

In this section, we derive a feasible variationally optimal estimator $\hat { J } _ { \mathrm { o I P S } }$ . Specifically, we employ the logger-wise cross-fitting strategy [Zheng and van der Laan, 2011, Chernozhukov et al., 2018, Kallus et al., 2021]: one part of the data is used to estimate the nuisance parameter α by estimating T and c, while the other part is used to compute the estimator (3.4).

Computing $\hat { J } _ { \mathrm { o I P S } } .$ . For each $i \in [ K ]$ , let $\big \{ \mathcal { D } _ { i } ^ { ( z ) } \big \} _ { z \in [ Z ] }$ be a Z-fold partition $\left( Z \ \ge \ 2 \right)$ of $\mathcal { D } _ { i }$ into nonempty folds chosen independently of the observed values, with $\mathcal { D } _ { i } ^ { ( - z ) } : = \mathcal { D } _ { i } \setminus \mathcal { D } _ { i } ^ { ( z ) }$ and $n _ { i } ^ { ( - z ) } : = \left| \mathcal { D } _ { i } ^ { ( - z ) } \right|$ . Then, for each fold $z \in [ Z ]$ , estimate T, c by

$$
\hat { T } _ { i j } ^ { ( z ) } : = \frac { 1 } { n _ { i } ^ { ( - z ) } } \sum _ { ( S , A , R ) \in \mathcal { D } _ { i } ^ { ( - z ) } } \frac { \pi _ { j } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ^ { ( - z ) } ( A \mid S ) } , \quad \hat { c } _ { i } ^ { ( z ) } : = \frac { 1 } { n _ { i } ^ { ( - z ) } } \sum _ { ( S , A , R ) \in \mathcal { D } _ { i } ^ { ( - z ) } } \frac { R \pi _ { e } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ^ { ( - z ) } ( A \mid S ) }\tag{3.5}
$$

where $\begin{array} { r } { \pi _ { \mathrm { a v g } } ^ { ( - z ) } : = \frac { 1 } { n ^ { ( - z ) } } \sum _ { i \in [ K ] } n _ { i } ^ { ( - z ) } \pi _ { i } } \end{array}$ with $\begin{array} { r } { n ^ { ( - z ) } : = \sum _ { i \in [ K ] } n _ { i } ^ { ( - z ) } } \end{array}$ . Then, we estimate $\alpha ^ { \star }$ with $\hat { \pmb { \alpha } } ^ { ( z ) } : = \hat { \mathbf { T } } ^ { ( z ) } ) ^ { \dagger } \hat { \mathbf { c } } ^ { ( z ) }$ , where † denotes the Moore–Penrose pseudoinverse.

Let $n _ { i } ^ { ( z ) } : = | \mathcal { D } _ { i } ^ { ( z ) } |$ and $\begin{array} { r } { n ^ { ( z ) } : = \sum _ { i \in [ K ] } n _ { i } ^ { ( z ) } } \end{array}$ . Then, the feasible variationally optimal estimator is

$$
\boxed { \hat { J } _ { \mathrm { o I P S } } ( \pi _ { e } ; \mathcal { D } ) : = \frac { 1 } { n } \sum _ { z = 1 } ^ { Z } n ^ { ( z ) } \hat { J } \big ( \pi _ { e } , \hat { \alpha } ^ { ( z ) } ; \mathcal { D } ^ { ( z ) } \big ) }\tag{3.6}
$$

where $\hat { J } \left( \pi _ { e } , \hat { \pmb { \alpha } } ^ { ( z ) } ; { \cal D } ^ { ( z ) } \right)$ is (3.4) evaluated with $\hat { \pmb { \alpha } } ^ { ( z ) }$ and $\mathcal { D } ^ { ( z ) } : = \{ \mathcal { D } _ { i } ^ { ( z ) } \} _ { i \in [ K ] }$ in place of $\alpha ^ { \star }$ and D. Note that the quantity $\pi _ { \mathrm { a v g } }$ in (3.4) depends on D through $n _ { i } / n$ , so it is also replaced with $\begin{array} { r } { \pi _ { \mathrm { a v g } } ^ { ( z ) } : = \frac { 1 } { n ^ { ( z ) } } \sum _ { i \in [ K ] } n _ { i } ^ { ( z ) } \pi _ { i } } \end{array}$

When Logging Policies are Unknown. When the logging policies are unknown, we plug consistent estimators of $\pi _ { i } ( a | s )$ and $\pi _ { \mathrm { a v g } } ( a | s )$ into the above procedure. The estimation procedures are given in Appendix A.

In Theorem 4.2, we show that $\hat { J } _ { \mathrm { o I P S } }$ is unbiased and asymptotically normal with the optimal asymptotic variance n $\operatorname { V a r } [ \hat { J } ( \pi _ { e } , \mathbf { w } ^ { \star } ; \mathcal { D } ) ]$ when the logging policies are known; Remark 2 discusses estimated logging policies.

## 4 Theoretical Analysis

In this section, we analyze $\hat { J } _ { \mathrm { o I P S } }$ introduced in §3 and prove its asymptotic variance optimality within the generalized weighted-IPS class (3.1). We then compare $\hat { J } _ { \mathrm { o I P S } }$ with other estimators, including the estimator $\hat { J } _ { \mathrm { D R - b I P S } } ~ ( 2 . 4 )$ , which is asymptotically optimal within a class containing the three IPS baselines [Kallus et al., 2021] but requires an estimator of the conditional mean reward. We characterize the variance gap between $\hat { J } _ { \mathrm { o I P S } }$ and $\hat { J } _ { \mathrm { D R - b I P S } }$ in terms of the residual from projecting the evaluation policy $\pi _ { e }$ , weighted by the conditional mean reward $\boldsymbol q ( s , a )$ , namely $q \pi _ { e } .$ , onto the logger basis space.

For simplicity of the analysis, we make the following assumptions.

Assumption 1 (Regularity).

(i) All logging and evaluation policies have strictly positive densities over the action space at every state in the support $o f p _ { s } . { } ^ { 3 }$

(ii) The weighted scores $w _ { i } ( S , A , R ) \pi _ { e } ( A \mid S ) / \pi _ { i } ( A \mid S )$ havefinite second moments under $\mu _ { \pi _ { i } }$ and the optimal weights in (3.2) are square-integrable.

(iii) For thefeasible estimator, K and Z are fixed and thefolds ofeach stratum have equal sizes up to rounding. For every $i \in [ K ]$ and $\dot { z } \in [ Z ]$ , the summand of (3.4) computed with $\hat { \pmb { \alpha } } ^ { ( z ) }$ converges in probability to the summand computed with $\alpha ^ { \star }$ in $L ^ { \mathrm { { \dot { 2 } } } } ( \mu _ { \pi _ { i } } )$ , and both summands havefinite second moments, including over the randomness ofthe trainingfolds.

Remark 1. $H \pi _ { \epsilon }$ has zeros, asfor the deterministic evaluation policy in §5, Assumption 1(i)fails: the weights $( 3 . 2 )$ are undefined where $\pi _ { e } ( a \mid s ) = 0$ , and the control-variateform (3.4) is no longer a member of $\mathrm { \Delta T _ { w I P S } }$ . The control-variate form remains unbiasedfor every α, and any solution of ${ \bf T } \alpha = { \bf c }$ still minimizes its variance over α $\in \mathbb { R } ^ { K }$ . The comparisons in $\vartheta 4 .$ .3 extend to such $\pi _ { e }$ by continuity, by applying them to $\pi _ { e } ^ { \epsilon } : = ( 1 - \epsilon ) \pi _ { e } + \epsilon \pi _ { u }$ , where $\pi _ { u }$ is the uniform policy, and letting $\epsilon  0$

## 4.1 Derivation of Optimal Weights

Let $\hat { J } ( \pi _ { e } , { \bf w } ; \mathcal { D } )$ be a generalized weighted-IPS estimator (3.1). Then, Var $\lceil \hat { J } ( \pi _ { e } , \mathbf { w } ; \mathcal { D } ) \rceil$ is a convex functional of w, so the variance can be optimized directly over the feasible weights. The main convex program of interest is

$$
\begin{array} { l l l } { \displaystyle \operatorname* { m i n i m i z e } \quad } & { \qquad } & { \mathrm { V a r } \big [ \hat { J } ( \pi _ { e } , \mathbf { w } ; \mathcal { D } ) \big ] } \\ { \displaystyle \quad \mathrm { s u b j e c t ~ t o } \quad } & { \qquad } & { \displaystyle \sum _ { i = 1 } ^ { K } w _ { i } ( \omega ) = \rho _ { r } ( \omega ) , \quad \forall \omega \in \mathrm { s u p p } ( \mu _ { \pi _ { e } } ) , } \end{array}\tag{4.1}
$$

Theorem 4.1. The weight vector $\mathbf { w } ^ { \star }$ given in (3.2) solves the variance minimization problem (4.1).

The proof of this theorem is given in Appendix E.2.

By Theorem 4.1, computing $\hat { J } _ { \mathrm { o I P S } }$ only requires estimating T and c, whose i-th rows are averages over stratum i, and then solving the $K \times K$ system ${ \bf T } \alpha = { \bf c }$ for each fold. Assuming each propensity evaluation takes constant time, accumulating the moments takes $O ( n K )$ time and each solve takes $O ( K ^ { 3 } )$ , so $Z { \mathrm { - } } \mathrm { f o l d }$ cross-fitting costs $O ( Z ( \overset { \smile } { n } K + K ^ { 3 } ) )$ , which is $\dot { O } ( n \dot { K } + K ^ { 3 } )$ for fixed $Z .$ If the per-fold moments are accumulated in one pass over the data, the extra memory is $O ( Z K ^ { 2 } )$ , and a second pass evaluates the estimator; storing every cross-logger propensity instead takes ${ \dot { O ( n K ) } }$ . These counts exclude fitting nuisance models such as propensity models. Appendix F reports the corresponding runtimes.

## 4.2 Unbiasedness and Asymptotic Variance Optimality of $\hat { J } _ { \mathrm { o I P S } }$ within $\Gamma _ { \mathrm { w I P S } }$

We next analyze the feasible cross-fitted estimator $\hat { J } _ { \mathrm { o I P S } }$ . Specifically, we prove its unbiasedness and asymptotic variance optimality within the generalized weighted-IPS estimator class $\Gamma _ { \mathrm { w I P S } }$ . We follow the multi-logger asymptotic regime of Kallus et al. [2021]: for each logger, the number of observations is $n _ { i } ^ { \prime } = m n _ { i } .$ , and we consider the limit $m  \infty$

Theorem 4.2. Let $n ^ { \prime } : = \textstyle \sum _ { i = 1 } ^ { K } n _ { i } ^ { \prime }$ with $n _ { i } ^ { \prime } : = m n _ { i }$ . Under Assumption 1, when the logging policies are known, $\hat { J } _ { \mathrm { o I P S } }$ is unbiased, and

$$
\sqrt { n ^ { \prime } } \left( \hat { J } _ { \mathrm { o I P S } } ( \pi _ { e } ; \mathcal { D } ) - J ( \pi _ { e } ) \right) \overset { d } { \to } \mathcal { N } \left( 0 , n \mathrm { V a r } [ \hat { J } ( \pi _ { e } , \mathbf { w } ^ { \star } ; \mathcal { D } ) ] \right)
$$

as $m  \infty ,$ , where $n : = \textstyle \sum _ { i = 1 } ^ { K } n _ { i }$ and $\operatorname { V a r } [ \hat { J } ( \pi _ { e } , \mathbf { w } ^ { \star } ; \mathcal { D } ) ]$ is computed at the base stratum sizes $( n _ { i } ) _ { i = 1 } ^ { K } .$

That $\mathrm { i s } , \ \hat { J } _ { \mathrm { o I P S } }$ is consistent and asymptotically normal, with the optimal asymptotic variance $n \operatorname { V a r } [ \hat { J } ( \pi _ { e } , \mathbf { w } ^ { \star } ; \mathcal { D } ) ]$ achievable within the generalized weighted-IPS estimator class $\Gamma _ { \mathrm { w I P S } }$ . The proof of Theorem 4.2 is given in Appendix E.3.

Remark 2 (Estimated logging policies). When the logging policies are unknown, we plug in estimates fitted on the training folds. Under mild regularity conditions, $\hat { J } _ { \mathrm { o I P S } }$ remains consistent if these estimates are consistent, but its limiting distribution generally differsfrom that in Theorem 4.2: the score of $\hat { J } _ { \mathrm { o I P S } }$ is not orthogonal to the propensities, so propensity estimation error enters at first order, as for IPS with estimated propensities [Hirano et al., 2003]. The limit in Theorem 4.2 still holds ifthe resultingfold-wise bias is $o _ { p } ( n ^ { - 1 / 2 } )$ , for example when $q \pi _ { e } \in \mathrm { s p a n } \bigl ( \{ \pi _ { i } \} _ { i \in [ K ] } \bigr )$

## 4.3 Theoretical Comparison with Other Estimators

We provide a theoretical comparison of the estimators in terms of their asymptotic variance.

Comparison with $\hat { J } _ { \mathrm { I P S } } , \hat { J } _ { \mathrm { b I P S } }$ and $\begin{array} { r l } { \hat { J } _ { \mathrm { w I P S } } . } & { { } \Gamma _ { \mathrm { w I P S } } } \end{array}$ contains the naive IPS estimator $( \hat { J } _ { \mathrm { I P S } } )$ , the balanced IPS estimator $( \hat { J } _ { \mathrm { b I P S } } )$ , and the weighted IPS estimator $( \hat { J } _ { \mathrm { w I P S } } )$ . Setting the weights in (3.1) to

$$
w _ { i } ^ { \mathrm { I P S } } = \frac { n _ { i } } { n } r , \qquad w _ { i } ^ { \mathrm { b I P S } } = \frac { n _ { i } } { n } \frac { \pi _ { i } } { \pi _ { \mathrm { a v g } } } r , \qquad w _ { i } ^ { \mathrm { w I P S } } = n _ { i } \hat { \gamma } _ { i } r
$$

recovers $( 2 . 1 ) , ( 2 . 2 )$ , and $( 2 . 3 ) $ , respectively. Each choice satisfies the normalization constraint because $\begin{array} { r } { \sum _ { i } n _ { i } / n = 1 , \sum _ { i } ( n _ { i } / n ) \pi _ { i } \overline { { / } } \pi _ { \mathrm { a v g } } = 1 } \end{array}$ , and $\textstyle \sum _ { i } n _ { i } { \hat { \gamma } } _ { i } = 1$ . The same identities hold for the cross-fitted version of $\hat { J } _ { \mathrm { w I P S } }$ , conditionally on the training folds, with the held-out stratum sizes in place of $n _ { i } .$ . Hence, with known propensities, our asymptotic efficiency result in Theorem 4.2 implies that, for every problem instance,

$$
\mathrm { A V a r } \big [ \hat { J } _ { \mathrm { o I P S } } ( \pi _ { e } ; \mathcal { D } ) \big ] \leq \mathrm { A V a r } \big [ \hat { J } _ { \mathrm { x } } ( \pi _ { e } ; \mathcal { D } ) \big ] , \quad \forall \mathrm { x } \in \{ \mathrm { I P S } , \mathrm { b I P S } , \mathrm { w I P S } \}
$$

where $\mathrm { A V a r } [ \hat { J } ]$ denotes the variance of the limiting distribution of $\sqrt { n ^ { \prime } } \left( \hat { J } - J ( \pi _ { e } ) \right)$ ). This resolves the dilemma between balanced and weighted IPS noted by Agarwal et al. [2017]: asymptotically, $\hat { J } _ { \mathrm { o I P S } }$ is at least as efficient as both.

Comparison with $\hat { J } _ { \mathrm { D R \cdot b I P S } }$ . Rewriting our infeasible optimal estimator $\hat { J } ( \pi _ { e } , \alpha ^ { \star } ; \mathcal { D } ) \ ( 3 . 4 )$ , we have

$$
\hat { J } ( \pi _ { e } , \alpha ^ { \star } ; \mathcal { D } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { K } \sum _ { ( S , A , R ) \in \mathcal { D } _ { i } } \left[ \frac { \pi _ { e } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ( A \mid S ) } \left( R - \frac { g _ { \alpha ^ { \star } } } { \pi _ { e } } ( S , A ) \right) + \frac { g _ { \alpha ^ { \star } } } { \pi _ { e } } ( S , \pi _ { e } ) \right] ,\tag{4.2}
$$

where $\begin{array} { r } { g _ { \alpha ^ { \star } } : = \sum _ { i \in [ K ] } \alpha _ { i } ^ { \star } \pi _ { i } } \end{array}$ and $\begin{array} { r }  \frac { g _ { \alpha ^ { \star } } } { \pi _ { e } } ( s , \pi _ { e } ) : = \int _ { \boldsymbol { \mathcal { A } } } \frac { g _ { \alpha ^ { \star } } ( s , a ) } { \pi _ { e } ( a | s ) } \pi _ { e } ( a | s ) d \mu _ { a } ( a ) = \sum _ { i \in [ K ] } \alpha _ { i } ^ { \star } \{ \ \end{array}$

Thus, our estimator belongs to the class of estimators studied by Kallus et al. [2021]:

$$
\hat { J } _ { \Gamma } ( \pi _ { e } , h , f ; \mathcal { D } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { K } \sum _ { ( S , A , R ) \in \mathcal { D } _ { i } } \big [ h ( i , S , A ) \pi _ { e } ( A \mid S ) \big ( R - f ( S , A ) \big ) + f ( S , \pi _ { e } ) \big ] ,
$$

where h satisfies $\textstyle \sum _ { i } ( n _ { i } / n ) \pi _ { i } ( a \mid s ) h ( i , s , a ) = 1$ , which makes $\hat { J } _ { \Gamma }$ unbiased. The optimal variance is achieved by $h ( \overline { { i , s } } , a ) = 1 / \pi _ { \mathrm { a v g } } ( a \mid s )$ for all $i \in [ K ]$ and $f ( s , a ) = q ( s , a )$ , with $q ( s , a )$ denoting the conditional mean reward. This holds even though our weights w may depend on $( s , a , r )$ while $h$ depends only on $( i , s , a )$ : the representation (4.2) corresponds to $h = 1 / \pi _ { \mathrm { a v g } }$ and $f = g _ { \alpha ^ { \star } } / \pi _ { e }$

The variance gap between our infeasible estimator $\hat { J } ( \pi _ { e } , \mathbf { w } ^ { \star } ; \mathcal { D } )$ , which is optimal within $\Gamma _ { \mathrm { w I P S } }$ and the infeasible optimal estimator $\hat { J } _ { \Gamma } ^ { \star } ( \pi _ { e } ; \mathcal { D } ) : = \hat { J } _ { \Gamma } ( \pi _ { e } , 1 / \pi _ { \mathrm { a v g } } , q ; \mathcal { D } )$ in this class is characterized in terms of the residual from the orthogonal projection of $q \pi _ { e }$ onto the finite-dimensional space spanned by the logging policies, span $\left( \breve { \left\{ \pi _ { i } \right\} } _ { i \in [ K ] } \right)$ . The precise statement is given below, with the proof deferred to Appendix E.4.

Theorem 4.3. The variance gap between $\hat { J } ( \pi _ { e } , \alpha ^ { \star } ; \mathcal { D } )$ and the semiparametrically efficient oracle estimator $\hat { J } _ { \Gamma } ^ { \star } ( \pi _ { e } ; \mathcal { D } ) : = \hat { J } _ { \Gamma } ( \pi _ { e } , 1 / \pi _ { \mathrm { a v g } } , q ; \mathcal { D } )$ is given by

$$
\mathrm { V a r } \big [ \hat { J } ( \pi _ { e } , \alpha ^ { \star } ; \mathcal { D } ) \big ] - \mathrm { V a r } \big [ \hat { J } _ { \Gamma } ^ { \star } ( \pi _ { e } ; \mathcal { D } ) \big ] = \frac { 1 } { n } \left( \operatorname* { i n f } _ { \alpha \in \mathbb { R } ^ { K } } \big \| q \pi _ { e } - g _ { \alpha } \big \| _ { \mu _ { \mathrm { l o g } } } ^ { 2 } - \mathrm { V a r } _ { p _ { s } } \big [ q ( S , \pi _ { e } ) \big ] \right) \geq 0 ,
$$

where $\| \cdot \| _ { \mu _ { \log } }$ is the norm induced by the inner product $\langle \cdot , \cdot \rangle _ { \mu _ { \mathrm { l o g } } }$ , with $d \mu _ { \mathrm { l o g } } : = p _ { s } / \pi _ { \mathrm { a v g } } d \mu _ { s , a } $ Moreover, equality holds whenever $q \pi _ { e } \in \mathrm { s p a n } \bigl ( \{ \pi _ { i } \} _ { i \in [ K ] } \bigr )$

Kallus et al. [2021] showed that, for any consistent estimator $\hat { q }$ of the conditional mean reward $q ,$ under their regularity conditions, J<sup>ˆ</sup><sub>DR-bIPS</sub> (2.4) achieves the optimal asymptotic variance $n \mathrm { V a r } \big [ \hat { J } _ { \Gamma } ^ { \star } ( \pi _ { e } ; \mathcal { D } ) \big ]$

Combining this result with Theorem 4.2, we obtain

$$
\mathrm { A V a r } \big [ \hat { J } _ { \mathrm { D R - b I P S } } ( \pi _ { e } ; \mathcal { D } ) \big ] \leq \mathrm { A V a r } \big [ \hat { J } _ { \mathrm { o I P S } } ( \pi _ { e } ; \mathcal { D } ) \big ] .
$$

We close this gap in Appendix B with a doubly robust extension that uses the same reward model: it applies the control-variate form (3.4) to the residual $R - \hat { q } ( S , A )$ and adds the direct-method term $\hat { q } ( S , \pi _ { e } )$ . We show that, with known logging policies, $\hat { J } _ { \mathrm { D R - o I P S } }$ has asymptotic variance no larger than that of $\hat { J } _ { \mathrm { D R - b I P S } }$ . Moreover, suppose that both estimators use the same misspecified reward model, $\hat { q }  q ^ { \prime } \neq q .$ . If $( q - q ^ { \prime } ) \pi _ { e } \in \mathrm { s p a n } \{ \pi _ { i } \} _ { i = 1 } ^ { K }$ , then $\hat { J } _ { \mathrm { D R - o I P S } }$ still attains the oracle asymptotic variance $n \mathrm { V a r } \big [ \hat { J } _ { \Gamma } ^ { \star } ( \pi _ { e } ; \mathcal { D } ) \big ]$ , whereas $\hat { J } _ { \mathrm { D F } }$ <sub>-bIPS</sub> in general does not. The reason is that $\scriptstyle { \hat { J } } _ { \mathrm { D R - o I P S } }$ projects $( q - q ^ { \prime } ) \pi _ { \epsilon }$ onto span $\{ \pi _ { i } \} _ { i = 1 } ^ { K }$ in $L ^ { 2 } ( \mu _ { \mathrm { l o g } } )$ and effectively uses the corrected reward model $q ^ { \prime } + g ^ { \Delta } / \pi _ { e }$ , where $g ^ { \Delta }$ denotes this projection; only the component of $( q - q ^ { \prime } ) \pi _ { e }$ orthogonal to the span contributes excess variance.

## 5 Numerical Experiments

## 5.1 Benchmark Datasets and Experiment Setup

$$
O p t D i g i t s [ \mathrm { A l - }
$$

Table 1: Dataset Statistics
<table><tr><td>Dataset Name</td><td>OptDigits</td><td>SatImage</td><td>PenDigits</td><td>Letter</td></tr><tr><td>#Classes (l)</td><td>10</td><td>6</td><td>10</td><td>26</td></tr><tr><td>#Samples</td><td>5620</td><td>6435</td><td>10992</td><td>20000</td></tr></table>

age [Srinivasan, 1993], PenDigits [Alpaydin and Alimoglu, 1996], and Letter [Slate, 1991]. Each classification dataset is converted into a contextual bandit dataset by treating labels as actions and assigning reward 1 for a correct action and 0 otherwise.

Mapping Classification Problems to Contextual Bandits. In deterministic-label multi-class classification, samples take the form $( S , y ( S ) )$ , where $S \sim p _ { \ast }$ <sub>s</sub> and $y ( \cdot )$ is the ground-truth labeling function. We view this as a contextual bandit problem by treating the feature distribution $p _ { s }$ as the context distribution and defining the deterministic reward as $r ( s , a ) : = \mathbb { 1 } [ y ( s ) = a ]$

To construct the evaluation policy, we train a multinomial logistic classifier on 30% of each dataset and define the deterministic policy $\pi _ { \operatorname* { d e t } } ( a \mid s ) : = \mathbb { 1 } [ a = { \hat { y } } ( s ) ]$ . We set the evaluation policy to $\pi _ { e } = \pi _ { \mathrm { d e t } } .$ Because $\pi _ { e }$ is deterministic, the weights (3.2) are undefined where $\pi _ { e } ( a \mid s ) = 0 .$ , so we compute $\hat { J } _ { \mathrm { o I P S } }$ through the control-variate form (3.4), which does not divide by $\pi _ { e }$ (Remark 1). The remaining 70% of the data is used as an evaluation pool. On this pool, the empirical accuracy of $\pi _ { \mathrm { d e t } }$ is treated as the ground-truth policy value ${ \cal J } ( \pi _ { e } )$

We generate logged bandit data from the evaluation pool using two logging policies,

$$
\pi _ { 1 } = 0 . 9 5 \cdot \pi _ { \mathrm { d e t } } + 0 . 0 5 \cdot \pi _ { u } , ~ \pi _ { 2 } = 0 . 0 5 \cdot \pi _ { \mathrm { d e t } } + 0 . 9 5 \cdot \pi _ { u } ,
$$

where $\pi _ { u }$ is the policy that chooses an action uniformly at random for any context. Thus, $\pi _ { 1 }$ is well aligned with $\pi _ { e } ,$ , whereas $\pi _ { 2 }$ is much less aligned with $\pi _ { e } .$ . For each context in the evaluation pool, we assign it to either $\mathcal { D } _ { 1 }$ or $\mathcal { D } _ { 2 } .$ , sample an action from the corresponding logging policy, and observe the deterministic reward $r ( s , a )$ . We vary the stratum-size ratio over $n _ { 1 } / \bar { n } _ { 2 } \in \{ 0 . 1 , 0 . 2 5 , 0 . 5 , 1 , 2 , 4 , 1 0 \}$ Smaller ratios make the logged dataset more dominated by $\pi _ { 2 }$ . Here the two loggers explore uniformly with probabilities 0.05 and 0.95 (the weights on $\pi _ { u }$ above). Appendix D uses (0.20, 0.80) and (0.35, 0.65), and Appendix C uses (0.05, 0.50, 0.95) for $K = 3$ and (0.05, 0.15, 0.30, 0.70, 0.95) for $\dot { K } = 5$

We benchmark each estimator $\hat { J }$ against the ground-truth value ${ \cal J } ( \pi _ { e } )$ using relative root mean squared error. For M independent trials with ground-truth values $J _ { m }$ and $\begin{array} { r } { \bar { J } \stackrel { \smile } { = } M ^ { - 1 } \sum _ { m } J _ { m } . } \end{array}$ , this metric is

$$
\mathrm { R e l a t i v e - R M S E } ( \hat { J } ) : = \frac { 1 } { \bar { J } } \sqrt { \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \left( J _ { m } - \hat { J } _ { m } \right) ^ { 2 } } ,\tag{5.1}
$$

where ${ \hat { J } } _ { m } $ denotes the estimated policy value in the m-th trial. To separate bias from variance, we also report the squared bias $\begin{array} { r } { \big ( \ b { \hat { M } ^ { - 1 } } \ b { \sum _ { m = 1 } ^ { M } } ( \ b { \hat { J } _ { m } } - \ b { J _ { m } } ) \big ) ^ { 2 } } \end{array}$ and the empirical standard deviation of $\hat { J } _ { 1 } , \dots , \hat { J } _ { M }$ (Appendix A). We use $M = 2 0 0$ trials and report $9 5 \%$ paired percentile-bootstrap intervals. Each of the $B = 2 { , } 0 0 0$ bootstrap resamples draws M trial indices with replacement and recomputes Relative-RMSE for every estimator from the same indices and the matching ground-truth values $J _ { m } ;$ the interval endpoints are the 2.5th and 97.5th percentiles. Because each trial redraws the data split, the logged actions, and the cross-fitting folds, the intervals reflect all three sources of variation. To compare two estimators, we apply the same procedure to the difference of their Relative-RMSEs computed on the same resampled trials.

Estimators Considered. We compare $\hat { J } _ { \mathrm { I P S } } ( \mathrm { E q . ~ } 2 . 1 ) , \hat { J } _ { \mathrm { b I P S } } ( \mathrm { E q . ~ } 2 . 2 ) , \hat { J } _ { \mathrm { w I P S } } ( \mathrm { E q . ~ } 2 . 3 )$ , our $\hat { J } _ { \mathrm { o I P S } }$ (§3), $\hat { J } _ { \mathrm { D R - b I P S } }$ [Kallus et al., 2021], and our $\scriptstyle { \dot { J } } _ { \mathrm { D R - o I P S } }$ (Appendix B). The two DR estimators use the same reward model qˆ, a logistic regression fitted with two-fold cross-fitting, and the same folds: for both, every reward, propensity, and weight quantity is fitted on the training fold and evaluated on the same held-out observations. In all experiments, we compute $\hat { \pmb { \alpha } } ^ { ( z ) }$ with NumPy’s Moore–Penrose pseudoinverse (numpy.linalg.pinv, default relative cutoff $1 0 ^ { - 1 5 } )$ ). In the designs with $K \geq 3$ loggers (Appendices C and F), every logger is a mixture of the same two policies, so T has rank two and the pseudoinverse returns the minimum-norm least-squares solution whenever $\hat { \mathbf { T } } ^ { ( z ) }$ is singular. Appendix A details estimation of unknown logging policies.

The experimental results are reported and discussed in Appendix A. In summary, $\hat { J } _ { \mathrm { o I P S } }$ generally improves over existing multi-logger IPS baselines across benchmark datasets and logger mixtures, and the DR extension $\hat { J } _ { \mathrm { D R - o I P S } }$ achieves lower Relative-RMSE than $\hat { J } _ { \mathrm { D R - b I P S } }$ in many finite-sample settings. In a synthetic experiment with $n = 1 0 ^ { 6 }$ samples (Appendix F.3), $\hat { J } _ { \mathrm { o I P S } }$ attains a Relative-RMSE of $1 . 9 9 \times 1 0 ^ { - 4 }$ , compared with $5 . 4 4 \times 1 0 ^ { - 4 }$ for $\hat { J } _ { \mathrm { w I P S } }$ . Code to reproduce the experiments is included in the supplementary material and will be publicly released.

## 6 Conclusion

We studied off-policy evaluation with multiple logging policies, where existing IPS estimators can have highly instance-dependent performance. We addressed this by deriving $\hat { J } _ { \mathrm { o I P S } }$ , an asymptotically variance-minimizing estimator within the generalized weighted-IPS class $\Gamma _ { \mathrm { w I P S } }$ , which contains previously considered IPS estimators. The optimal weights have a closed-form characterization through the linear system ${ \bf T } \alpha = { \bf c }$ , whose entries can be estimated from logged data, leading to a practical cross-fitted estimator. This gives a principled way to choose sample-dependent IPS weights rather than selecting among fixed rules such as naive IPS, balanced IPS, and weighted IPS.

We also related $\hat { J } _ { \mathrm { o I P S } }$ to the semiparametric efficiency framework of Kallus et al. [2021]. The linear system ${ \bf T } \alpha = { \bf c }$ has a projection interpretation: it chooses the best control variate representable by the logging policies. When this space is rich enough, $\hat { J } _ { \mathrm { o I P S } }$ asymptotically recovers the efficient oracle. Otherwise, the projection residual characterizes the variance gap. We further extended the estimator to a doubly robust version, $\hat { J } _ { \mathrm { D R - o I P S } }$ , which matches the asymptotic variance of the efficient DR baseline when the reward model is correctly specified and can be more robust under certain structured misspecifications.

Empirically, $\hat { J } _ { \mathrm { o I P S } }$ improves over existing multi-logger IPS baselines across benchmark datasets and logger mixtures, and $\scriptstyle { \hat { J } } _ { \mathrm { D R - o I P S } }$ often improves over the DR balanced IPS baseline in finite samples. The main limitation is that our optimality result is restricted to $\Gamma _ { \mathrm { w I P S } }$ and our asymptotic guarantees assume known logging policies; with estimated propensities, the limiting variance generally changes (Remark 2). We also do not analyze finite-sample behavior or the stability of the estimated weights, the method relies on sufficient overlap, and our experiments cover UCI benchmarks and one synthetic environment. Future work includes finite-sample analysis, richer basis classes beyond logging policies, and extensions to sequential settings.

## Acknowledgments and Disclosure of Funding

This research was supported by the Deep-Tech TIPS Program (Grant No. RS-2024-0046898) funded by the Ministry of SMEs and Startups, Republic of Korea.

## References

A. Agarwal, S. Basu, T. Schnabel, and T. Joachims. Effective evaluation using logged bandit feedback from multiple loggers. In Proceedings of the ACM SIGKDD international conference on knowledge discovery and data mining, pages 687–696, 2017.

E. Alpaydin and F. Alimoglu. Pen-based recognition of handwritten digits. UCI Machine Learning Repository, 1996. DOI: https://doi.org/10.24432/C5MG6K.

E. Alpaydin and C. Kaynak. Optical recognition of handwritten digits. UCI Machine Learning Repository, 1998. DOI: https://doi.org/10.24432/C50P49.

I. Aouali, V.-E. Brunel, D. Rohde, and A. Korba. Unified PAC-Bayesian study of pessimism for offline policy learning with regularized importance sampling. In Conference on Uncertainty in Artificial Intelligence (UAI), 2024.

L. Bottou, J. Peters, J. Quiñonero-Candela, D. X. Charles, D. M. Chickering, E. Portugaly, D. Ray, P. Simard, and E. Snelson. Counterfactual reasoning and learning systems: The example of computational advertising. Journal ofMachine Learning Research, 14(11), 2013.

X. Chen, L. Wang, Y. Hang, H. Ge, and H. Zha. Infinite-horizon off-policy policy evaluation with multiple behavior policies. In International Conference on Learning Representations, 2020.

V. Chernozhukov, D. Chetverikov, M. Demirer, E. Duflo, C. Hansen, W. Newey, and J. Robins. Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal, 21(1), 2018.

J.-M. Cornuet, J.-M. Marin, A. Mira, and C. P. Robert. Adaptive multiple importance sampling. Scandinavian Journal ofStatistics, 39(4):798–812, 2012.

M. Dudík, D. Erhan, J. Langford, and L. Li. Doubly robust policy evaluation and optimization. Statistical Science, 29:485––511, 2014.

V. Elvira, L. Martino, D. Luengo, and M. F. Bugallo. Generalized multiple importance sampling. Statistical Science, 34(1):129–155, 2019.

T. Hatt, D. Tschernutter, and S. Feuerriegel. Generalizing off-policy learning under sample selection bias. In Uncertainty in Artificial Intelligence, pages 769–779. PMLR, 2022.

L. He, L. Xia, W. Zeng, Z.-M. Ma, Y. Zhao, and D. Yin. Off-policy learning for multiple loggers. In Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 1184–1193, 2019.

K. Hirano, G. W. Imbens, and G. Ridder. Efficient estimation of average treatment effects using the estimated propensity score. Econometrica, 71(4):1161–1189, 2003.

D. G. Horvitz and D. J. Thompson. A generalization of sampling without replacement from a finite universe. Journal ofthe American Statistical Association, 47(260):663–685, 1952.

E. L. Ionides. Truncated importance sampling. Journal of Computational and Graphical Statistics, 17(2):295–311, 2008.

Y. Jung and A. Bellot. Efficient policy evaluation across multiple different experimental datasets. Advances in Neural Information Processing Systems, 37:136361–136392, 2024.

N. Kallus, Y. Saito, and M. Uehara. Optimal off-policy evaluation from multiple logging policies. In International Conference on Machine Learning, pages 5247–5256. PMLR, 2021.

I. Kondapaneni, P. Vévoda, P. Grittmann, T. Skˇrivan, P. Slusallek, and J. Kˇrivánek. Optimal multiple importance sampling. ACM Transactions on Graphics (TOG), 38(4):1–14, 2019.

I. Kuzborskij, C. Vernade, A. Gyorgy, and C. Szepesvári. Confident off-policy evaluation and selection through self-normalized importance weighting. In International Conference on Artificial Intelligence and Statistics (AISTATS), pages 640–648. PMLR, 2021.

S. D. Liu, C. Chen, and S. Zhang. Efficient multi-policy evaluation for reinforcement learning. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 39, pages 18951–18959, 2025.

A. M. Metelli, A. Russo, and M. Restelli. Subgaussian and differentiable importance sampling for off-policy evaluation and learning. Advances in neural information processing systems, 34: 8119–8132, 2021.

A. Owen and Y. Zhou. Safe and effective importance sampling. Journal ofthe American Statistical Association, 95(449):135–143, 2000.

D. Precup, R. S. Sutton, and S. P. Singh. Eligibility traces for off-policy policy evaluation. In Proceedings of the International Conference on Machine Learning, pages 759–766, 2000.

P. R. Rosenbaum and D. B. Rubin. The central role of the propensity score in observational studies for causal effects. Biometrika, 70(1):41–55, 1983.

Y. Saito, S. Aihara, M. Matsutani, and Y. Narita. Open bandit dataset and pipeline: Towards realistic and reproducible off-policy evaluation. In Neural Information Processing Systems Datasets and Benchmarks Track, 2021.

D. Slate. Letter recognition. UCI Machine Learning Repository, 1991. DOI: https://doi.org/10.24432/C5ZP40.

A. Srinivasan. Statlog (Landsat satellite). UCI Machine Learning Repository, 1993. DOI: https://doi.org/10.24432/C55887.

A. Swaminathan and T. Joachims. The self-normalized estimator for counterfactual learning. advances in neural information processing systems, 28, 2015.

E. Veach and L. J. Guibas. Optimally combining sampling techniques for monte carlo rendering. In Proceedings of the annual conference on Computer graphics and interactive techniques, pages 419–428, 1995.

J. M. Wooldridge. Asymptotic properties of weighted M-estimators for standard stratified samples. Econometric theory, 17(2):451–470, 2001.

W. Zheng and M. J. van der Laan. Cross-Validated Targeted Minimum-Loss-Based Estimation, pages 459–474. Springer New York, 2011.

## A Main Experimental Results and Discussion

This appendix reports the full experimental results for the setup described in §5. Figures 1 and 2 show Relative-RMSE on Letter, OptDigits, PenDigits, and SatImage across the stratum-size ratios $n _ { 1 } / n _ { 2 } \in \{ 0 . 1 , 0 . 2 5 , 0 . 5 , 1 , 2 , 4 , 1 0 \}$ . Each point is computed from $\bar { M } = 2 0 0$ independent trials, and our estimator $\hat { J } _ { \mathrm { o I P S } }$ uses Z = 5 folds.

When Logging Policies are Unknown. For non-DR estimators with unknown logging policies, we estimate each logging policy $\hat { \pi } _ { i }$ separately using 2-fold cross-fitted multinomial logistic regression within each logger. The out-of-fold models are used to estimate the observed-action propensities ${ \hat { \pi } } _ { i } ( A _ { i j } \mid S _ { i j } )$ for samples in stratum i. We also fit a model on all samples from stratum i and use it to evaluate $\hat { \pi } _ { i } \mathbf { \bar { ( } } A _ { k j } \ | \ S _ { k j } )$ for samples from all strata, which is needed when an estimator requires the propensity of a sample under a different logging policy.

For numerical stability, we blend each predicted action distribution with a 1% uniform distribution and clip all propensities below $1 0 ^ { - 3 }$ . For degenerate strata, such as strata with too few samples or only one observed action class, we instead use the uniform policy. The resulting estimated logging policies are then plugged into IPS-style estimators.

![](images/d174b6d3b06aedde4658ccc5fe11140bae885593ae087e0554a73c4f402755a0.jpg)  
Figure 1: Relative-RMSEs (5.1), computed over $M = 2 0 0$ independent experiments, across datasets and estimators when the logging policies are known. Our proposed estimator $\hat { J } _ { \mathrm { o I P S } }$ is implemented with $Z = \mathrm { 5 - f o l d }$ cross-fitting. The x-axis denotes the stratum-size ratio $( n _ { 1 } / n _ { 2 } )$ on a logarithmic scale, while the y-axis shows Relative-RMSE, also on a logarithmic scale. Lower values indicate better performance. Shaded bands show 95% paired percentile-bootstrap intervals (§5).

Known Logging Policies. When logging policies are known (Figure 1), $\hat { J } _ { \mathrm { o I P S } }$ consistently improves over non-DR IPS baselines across datasets and stratum-size ratios. In several regimes, it is also competitive with or better than $\hat { J } _ { \mathrm { D R - b I P S } }$ . Adding a reward model $( \hat { J } _ { \mathrm { D R - o I P S } } )$ improves only slightly on $\hat { J } _ { \mathrm { o I P S } } ^ { \mathrm { } } ,$ , which suggests that little residual structure remains after the control-variate correction. These results support the practical value of the control-variate form (3.4), which uses the logger-basis structure to reduce variance without requiring a conditional reward estimator. By contrast, $\hat { J } _ { \mathrm { I P S } }$ and $\hat { J } _ { \mathrm { b I P S } }$ often show larger errors, especially at small $n _ { \mathrm { 1 } } / n _ { \mathrm { 2 } }$ , where the logged data are dominated by the less aligned logger $\pi _ { 2 }$

Estimated Logging Policies. When logging policies are estimated (Figure 2), the same qualitative trend persists: $\hat { J } _ { \mathrm { o I P S } }$ remains strong and generally outperforms non-DR baselines $( \hat { J } _ { \mathrm { I P S } } , \ \hat { J } _ { \mathrm { b I P S } } .$ $\hat { J } _ { \mathrm { w I P S } } )$ across datasets and most stratum-size ratios. This is consistent with sample-dependent, crossfitted weighting through T<sup>ˆ</sup> and $\hat { \mathbf { c } } ,$ which stabilizes stratum balancing under propensity-estimation noise. In contrast, $\hat { J } _ { \mathrm { w I P S } }$ can underperform $\hat { J } _ { \mathrm { b I P S } }$ because its adaptive mixture weights depend on plug-in divergence estimates that are themselves sensitive to estimation noise.

Relative to the DR baseline $\hat { J } _ { \mathrm { D R - b I P S } } , ~ \hat { J } _ { \mathrm { o I P S } }$ is not uniformly better when logging policies are estimated; this regime is not covered by Theorem 4.2 (Remark 2). However, the natural DR extension $\scriptstyle { \hat { J } } _ { \mathrm { D R - o I P S } }$ achieves lower Relative-RMSE than $\hat { J } _ { \mathrm { D R - b I P S } }$ in many settings, particularly at low stratumsize ratios where the data are dominated by the less aligned logger. Subtracting the reward model first shrinks the inverse-propensity-weighted term $\pi _ { e } ( R - \hat { q } ) / \pi _ { \mathrm { a v g } } .$ , and $\hat { J } _ { \mathrm { D R - o I P S } }$ then applies the control-variate correction to this smaller residual, which can improve accuracy in finite samples.

![](images/6f2043d941b954d1cb0ebf7dec2fcf77917e5902852e3291e26f277bfedfdf24.jpg)  
Figure 2: Relative-RMSEs (5.1), computed over $M = 2 0 0$ independent experiments, across datasets and estimators when the logging policies are estimated. Shaded bands show 95% paired percentilebootstrap intervals (§5). Where bands are not visible, they are narrower than the markers.

Summary. Overall, $\hat { J } _ { \mathrm { o I P S } }$ is a strong practical estimator for multi-logger OPE, especially against IPS baselines, and its DR extension $\scriptstyle { \hat { J } } _ { \mathrm { D R - o I P S } }$ is often strongest when propensities are estimated. Appendices C and D show that these trends persist with more than two loggers and under alternative mixture settings. Across all settings, $\hat { J } _ { \mathrm { o I P S } }$ has a lower Relative-RMSE than each non-DR baseline in 276 of 276 configuration–baseline comparisons with known propensities and in 262 of 276 with estimated propensities. $\scriptstyle { \hat { J } } _ { \mathrm { D R - o I P S } }$ has a lower Relative-RMSE than ${ \hat { J } } _ { \mathrm { D R - b I P S } }$ in 92 of 92 and 66 of 92 configurations, respectively, and the paired 95% interval for the difference lies below zero in 89 and 64 of them.

Table 2 separates the error of each estimator into squared bias and empirical standard deviation at three representative stratum-size ratios. The squared bias is at most 1.5% of the variance in every cell, so the differences in Relative-RMSE come mainly from variance, and the smallest standard deviation is attained by $\hat { J } _ { \mathrm { o I P S } }$ or $\scriptstyle { \hat { J } } _ { \mathrm { D R - o I P S } }$ in all twelve dataset–ratio pairs.

## B Doubly Robust Extension of $\hat { J } _ { \mathrm { o I P S } }$

In this section, we present a doubly robust (DR) extension of our variationally optimal estimator $\hat { J } _ { \mathrm { o I P S } }$ . Given any conditional reward estimator $\hat { q } : \mathcal { S } \times \mathcal { A }  \mathbb { R }$ , we show how to incorporate $\hat { q }$ to obtain a DR version of $\hat { J } _ { \mathrm { o I P S } }$ , denoted by $\hat { J } _ { \mathrm { D R - o I P S } }$ . We then compare $\hat { J } _ { \mathrm { D R - o I P S } }$ both theoretically and empirically with the semiparametrically efficient estimator $\hat { J } _ { \mathrm { D R - b I P S } }$

## B.1 Generalized Weighted DR-IPS Estimator Class $\Gamma _ { \mathrm { D R w I P S } }$

We define the generalized weighted DR-IPS estimator class $\Gamma _ { \mathrm { D R w I P S } }$ as follows. Let $\hat { q } : \mathcal { S } \times \mathcal { A } $ R be an estimator of the conditional mean reward $q ( s , a ) : = \mathbb { E } [ R | S = s , A = a ]$ , and define

$$
\hat { J } _ { \mathrm { D M } } ( \pi _ { e } , \hat { q } ; \mathcal { D } ) : = \frac { 1 } { n } \sum _ { i = 1 } ^ { K } \sum _ { ( S , A , R ) \in \mathcal { D } _ { i } } \int _ { \mathcal { A } } \hat { q } ( S , a ) \pi _ { e } ( a \mid S ) d \mu _ { a } ( a )
$$

as the direct value estimator induced by ${ \hat { q } } .$

Then, the estimator class $\Gamma _ { \mathrm { D R w I P S } }$ consists of estimators of the following form:

$$
\hat { J } ( \pi _ { e } , { \bf w } , \hat { q } ; \mathcal { D } ) : = \hat { J } _ { \mathrm { D M } } ( \pi _ { e } , \hat { q } ; \mathcal { D } ) + \sum _ { i = 1 } ^ { K } \frac { 1 } { n _ { i } } \sum _ { ( S , A , R ) \in \mathcal { D } _ { i } } w _ { i } ( S , A , R ) \cdot \frac { \pi _ { e } ( A \mid S ) } { \pi _ { i } ( A \mid S ) } .\tag{B.1}
$$

where $\Delta ( s , a , r ) : = r - \hat { q } ( s , a )$ is the residual. The estimator is unbiased when $\hat { q }$ is a fixed function, or conditionally on the training data when $\hat { q }$ is fitted on data independent of ${ \mathcal { D } } ,$ , provided the weights satisfy the following constraints:

Table 2: Squared bias (top) and empirical standard deviation (bottom) of each estimator over $M = 2 0 0$ trials at three representative stratum-size ratios, with known logging policies $( \eta = 0 . 9 5 , K = 2 )$
<table><tr><td> $n _ { \mathrm { 1 } } / n _ { \mathrm { 2 } }$ </td><td>Dataset</td><td>IPS</td><td>bIPS</td><td>wIPS</td><td>oIPS</td><td>DR-bIPS</td><td>DR-oIPS</td></tr><tr><td rowspan="5"> $1 0 ^ { - 1 }$ </td><td>Letter</td><td>1.365e-6 0.021194</td><td>2.426e-7 0.013643</td><td>4.199e-9 0.012154</td><td>2.064e-7 0.009162</td><td>9.958e-8 0.012274</td><td>2.174e-7 0.008961</td></tr><tr><td>OptDigits</td><td>9.845e-6 0.036553</td><td>6.086e-6 0.025019</td><td>1.955e-6 0.015421</td><td>6.030e-8 0.007236</td><td>3.771e-6 0.020381</td><td>1.661e-8 0.008195</td></tr><tr><td>PenDigits</td><td>3.306e-6 0.024435</td><td>1.867e-6 0.016799</td><td>4.806e-7 0.011213</td><td>1.850e-7 0.006118</td><td>1.739e-6 0.013490</td><td>3.690e-7 0.006194</td></tr><tr><td>SatImage</td><td>1.315e-7 0.023837</td><td>1.240e-7 0.018533</td><td>1.246e-7 0.015773</td><td>9.588e-8 0.010424</td><td>2.831e-7 0.014409</td><td>1.638e-7 0.010491</td></tr><tr><td></td><td>2.891e-9</td><td>1.825e-8</td><td>2.700e-8</td><td>6.420e-9</td><td></td><td>1.414e-8 0.004680</td></tr><tr><td rowspan="5">1</td><td>Letter</td><td>0.016254 3.120e-6</td><td>0.005653</td><td>0.005392</td><td>0.004901</td><td>2.165e-8 0.004846</td><td>2.516e-8</td></tr><tr><td>OptDigits</td><td>0.026398</td><td>5.313e-8 0.009119</td><td>7.139e-8 0.007522</td><td>5.952e-9 0.004654</td><td>1.070e-9 0.005835</td><td>0.004749</td></tr><tr><td>PenDigits</td><td>6.577e-7 0.020048</td><td>2.949e-8 0.006382</td><td>5.040e-10 0.004676</td><td>1.553e-8 0.003435</td><td>2.238e-10 0.004076</td><td>2.223e-8 0.003374</td></tr><tr><td>SatImage</td><td>8.208e-8 0.018483</td><td>9.300e-8 0.009005</td><td>9.909e-8 0.008035</td><td>4.448e-10 0.006581</td><td>7.469e-9 0.007054</td><td>9.303e-10 0.006691</td></tr><tr><td></td><td>8.981e-7</td><td>1.555e-10</td><td>7.371e-9</td><td>2.392e-10</td><td>8.685e-10</td><td>6.415e-10</td></tr><tr><td>10</td><td>Letter</td><td>0.007988</td><td>0.004009</td><td>0.004053</td><td>0.003561</td><td>0.003556</td><td>0.003532</td></tr><tr><td rowspan="4"></td><td>OptDigits</td><td>1.723e-9 0.011590</td><td>8.947e-8 0.005328</td><td>1.261e-7 0.005066</td><td>1.026e-9 0.003627</td><td>3.153e-9 0.003636</td><td>7.132e-10</td></tr><tr><td></td><td>5.843e-7</td><td>1.267e-8</td><td></td><td></td><td></td><td>0.003605</td></tr><tr><td>PenDigits</td><td>0.009378</td><td>0.003771</td><td>9.083e-13 0.003493</td><td>5.156e-10 0.002592</td><td>1.545e-9 0.002580</td><td>4.439e-10 0.002573</td></tr><tr><td>SatImage</td><td>3.424e-7 0.008730</td><td>5.540e-8 0.005562</td><td>2.232e-8 0.005446</td><td>1.047e-8 0.005163</td><td>8.144e-9 0.005223</td><td>5.826e-9 0.005246</td></tr></table>

(i) Normalization: $\begin{array} { r } { \sum _ { i = 1 } ^ { K } w _ { i } ( s , a , r ) = \Delta ( s , a , r ) } \end{array}$

(ii) Dominance: $\pi _ { i } ( a | s ) = 0 \implies w _ { i } ( s , a , r ) = 0 .$

Intuitively, this form follows from the decomposition

$$
\mathbb { E } _ { \mu _ { \pi _ { e } } } [ R ] = \mathbb { E } _ { \mu _ { \pi _ { e } } } [ R - \hat { q } ] + \mathbb { E } _ { \mu _ { \pi _ { e } } } [ \hat { q } ] .
$$

That is, we apply weighted IPS to the residual term and use the direct method for the second term.

## B.2 Feasible Extension: $\hat { J } _ { \mathrm { D R - o I P S } }$

We seek to minimize the variance of (B.1). The variational approach used for $\hat { J } _ { \mathrm { o I P S } }$ applies directly here, with the target variable r replaced by the residual $\Delta ( s , a , r ) : = r - \hat { q } ( s , a )$

Following the same cross-fitting procedure described in §3 for constructing the feasible estimator $\hat { J } _ { \mathrm { o I P S } }$ , we obtain a feasible doubly robust extension, denoted $\hat { J } _ { \mathrm { D R - o I P S } }$ . For each fold z, let $\hat { q } ^ { ( z ) }$ denote the conditional reward estimator trained on $\mathcal { D } ^ { ( - z ) }$ . On the held-out fold $\mathcal { D } ^ { ( z ) }$ , define the fold-specific direct-method term by

$$
\hat { J } _ { \mathrm { D M } } \left( \pi _ { e } , \hat { q } ^ { ( z ) } ; \mathcal { D } ^ { ( z ) } \right) : = \frac { 1 } { n ^ { ( z ) } } \sum _ { i = 1 } ^ { K } \sum _ { ( S , A , R ) \in \mathcal { D } _ { i } ^ { ( z ) } } \int _ { A } \hat { q } ^ { ( z ) } ( S , a ) \pi _ { e } ( a \mid S ) d \mu _ { a } ( a ) .
$$

The residual IPS component is obtained by applying the control-variate form (3.4) to the residuals $R - \hat { q } ^ { ( z ) } ( S , A )$ . Specifically, let

$$
\begin{array} { r l } & { \hat { J } _ { \Delta } \Big ( \pi _ { e } , \hat { \alpha } ^ { ( z ) } ; \mathcal { D } ^ { ( z ) } \Big ) } \\ & { : = \frac { 1 } { n ^ { ( z ) } } \displaystyle \sum _ { i \in [ K ] } \sum _ { ( S , A , R ) \in \mathcal { D } _ { i } ^ { ( z ) } } \left[ \frac { \big ( R - \hat { q } ^ { ( z ) } ( S , A ) \big ) \pi _ { e } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ^ { ( z ) } ( A \mid S ) } - \frac { \sum _ { k \in [ K ] } \hat { \alpha } _ { k } ^ { ( z ) } \pi _ { k } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ^ { ( z ) } ( A \mid S ) } + \sum _ { k \in [ K ] } \hat { \alpha } _ { k } ^ { ( z ) } \right] , } \end{array}
$$

where

$$
\pi _ { \mathrm { a v g } } ^ { ( z ) } : = \frac { 1 } { n ^ { ( z ) } } \sum _ { i \in [ K ] } n _ { i } ^ { ( z ) } \pi _ { i } .
$$

The feasible DR-oIPS estimator is then

$$
\boxed { \hat { J } _ { \mathrm { D R - o I P S } } ( \pi _ { e } ; \mathcal { D } ) : = \cfrac { 1 } { n } \sum _ { z = 1 } ^ { Z } n ^ { ( z ) } \left[ \hat { J } _ { \mathrm { D M } } \left( \pi _ { e } , \hat { q } ^ { ( z ) } ; \mathcal { D } ^ { ( z ) } \right) + \hat { J } _ { \Delta } \left( \pi _ { e } , \hat { \alpha } ^ { ( z ) } ; \mathcal { D } ^ { ( z ) } \right) \right] }\tag{B.2}
$$

Shared outer-fold procedure. Both DR estimators split each stratum into the same two folds $\left( Z = 2 \right)$ . For each fold z:

1. On the training data $\mathcal { D } ^ { ( - z ) }$ , fit the reward model $\hat { q } ^ { ( z ) }$ , the logging-policy models when the propensities are unknown, and the moments $\hat { \mathbf { T } } ^ { ( z ) }$ and $\hat { \mathbf { c } } ^ { ( z ) }$ defined below, and set $\hat { \pmb { \alpha } } ^ { ( z ) } = ( \hat { \mathbf { T } } ^ { ( z ) } ) ^ { \dagger } \hat { \mathbf { c } } ^ { ( z ) }$

2. Evaluate both estimators on the held-out data $\mathcal { D } ^ { ( z ) }$ , using $\hat { \pmb { \alpha } } ^ { ( z ) }$ for $\scriptstyle { \hat { J } } _ { \mathrm { D R - o I P S } }$ and $\hat { \pmb { \alpha } } ^ { ( z ) } = 0$ for $\hat { J } _ { \mathrm { D R - b I P S } }$

The fold estimates are combined as in (B.2). With $\hat { \pmb { \alpha } } ^ { ( z ) } = 0 ;$ , this yields the cross-fitted form of (2.4) with $\hat { q } ^ { ( z ) }$ and $\pi _ { \mathrm { a v g } } ^ { ( z ) }$

Estimation of Coefficients. As in $\hat { J } _ { \mathrm { o I P S } }$ , the parameters $\hat { \pmb { \alpha } } ^ { ( z ) }$ in (B.2) are obtained from ${ \bf T } \alpha = { \bf c }$ via cross-fitting. Specifically, $\hat { \pmb { \alpha } } ^ { ( z ) } = ( \hat { \mathbf { T } } ^ { ( z ) } ) ^ { \dagger } \hat { \mathbf { c } } ^ { ( z ) }$ , where $\hat { \mathbf { T } } ^ { ( z ) }$ and $\hat { \mathbf { c } } ^ { ( z ) }$ are estimated on $\mathcal { D } ^ { ( - z ) }$ as follows.

(i) Technique Matrix. T depends only on propensities and sample sizes, so it is identical to the IPS case:

$$
\hat { T } _ { i j } ^ { ( z ) } : = \frac { 1 } { \left| \mathcal { D } _ { i } ^ { ( - z ) } \right| } \sum _ { ( S , A , R ) \in \mathcal { D } _ { i } ^ { ( - z ) } } \frac { \pi _ { j } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ^ { ( - z ) } ( A \mid S ) } .
$$

(ii) Contribution Vector. The vector c depends on the target signal. For DR, we use residuals $R - \hat { q } ^ { ( z ) } ( S , A )$ in place of R:

$$
\hat { c } _ { i } ^ { ( z ) } : = \frac { 1 } { \left| \mathscr { D } _ { i } ^ { ( - z ) } \right| } \sum _ { ( S , A , R ) \in \mathscr { D } _ { i } ^ { ( - z ) } } \frac { \left( R - \hat { q } ^ { ( z ) } ( S , A ) \right) \pi _ { e } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ^ { ( - z ) } ( A \mid S ) } .
$$

Here, $\begin{array} { r } { \pi _ { \mathrm { a v g } } ^ { ( - z ) } : = \frac { 1 } { n ^ { ( - z ) } } \sum _ { i \in [ K ] } n _ { i } ^ { ( - z ) } \pi _ { i } } \end{array}$ , with $\begin{array} { r } { n ^ { ( - z ) } : = \sum _ { i \in [ K ] } n _ { i } ^ { ( - z ) } } \end{array}$ , as in the main text, §3.2. When the logging policies are unknown, we replace the propensities in the above expressions by their cross-fitted estimates.

## B.3 Theoretical Comparison with the Optimal Estimator of Kallus et al. [2021]

In this section, we provide a theoretical comparison between $\scriptstyle { \hat { J } } _ { \mathrm { D R - o I P S } }$ and $\hat { J } _ { \mathrm { D R - b I P S } }$ . The purpose is to explain how the residual version of our variational construction relates to the semiparametrically efficient estimator studied in Kallus et al. [2021].

The comparison has two implications.

(i) When $\hat { q }$ is a consistent estimator of the conditional mean reward $\boldsymbol { q } ( s , a )$ , both $\hat { J } _ { \mathrm { D R - o I P S } }$ and J<sup>ˆ</sup><sub>DR-bIPS</sub> attain the same semiparametrically efficient oracle variance asymptotically.

(ii) When qˆ is misspecified, $\scriptstyle { \hat { J } } _ { \mathrm { D R - o I P S } }$ can remain variance-optimal in cases where the misspecification lies in the logger-basis space, whereas $\hat { J } _ { \mathrm { D R - b I P S } }$ need not.

With known logging policies and under the conditions of Theorem $4 . 2 ,$ these conclusions follow from the following lemma, proved in Appendix E.

Lemma E.3 (Variance Gap). Let f be measurable and such that the summands of the estimator below havefinite second moments under each $\mu _ { \pi _ { i } }$ , and let $\hat { J } _ { \Gamma } ( \pi _ { e } , f ; \mathcal { D } )$ be the estimator defined by

$$
\hat { J } _ { \Gamma } ( \pi _ { e } , f ; \mathcal { D } ) : = \frac { 1 } { n } \sum _ { i = 1 } ^ { K } \sum _ { ( S , A , R ) \in \mathcal { D } _ { i } } \left[ \frac { \pi _ { e } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ( A \mid S ) } \big ( R - f ( S , A ) \big ) + f ( S , \pi _ { e } ) \right] ,\tag{E.5}
$$

where $\begin{array} { r } { f ( s , \pi _ { e } ) : = \int _ { \cal A } f ( s , a ) \pi _ { e } ( a | s ) d \mu _ { a } ( a ) . } \end{array}$

Let $q ( s , a ) : = \mathbb { E } [ R | S = s , A = a ]$ denote the conditional mean reward, and define $m _ { i } ( f ) : =$ $\mathbb { E } _ { \mu _ { \pi _ { i } } } [ ( q - f ) \pi _ { e } / \pi _ { \mathrm { a v g } } ] - \mathbb { E } _ { p _ { s } } [ ( q - f ) ( S , \pi _ { e } ) ]$ . Then, the variance gap between $\hat { J } _ { \Gamma } ( \pi _ { e } , f ; \mathcal { D } )$ and $\hat { J } _ { \Gamma } ^ { \star } ( \pi _ { e } ; \mathcal { D } ) : = \hat { J } _ { \Gamma } ( \pi _ { e } , q ; \mathcal { D } )$ satisfies

$$
\begin{array} { r l } & { \mathrm { V a r } \big [ \hat { J } _ { \Gamma } ( \pi _ { e } , f ; \mathcal { D } ) \big ] - \mathrm { V a r } \big [ \hat { J } _ { \Gamma } ^ { \star } ( \pi _ { e } ; \mathcal { D } ) \big ] } \\ & { \qquad = \displaystyle \frac { 1 } { n } \left( \mathrm { V a r } _ { \mu _ { \pi _ { \mathrm { a v g } } } } \Bigg [ \frac { \pi _ { e } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ( A \mid S ) } \big ( q ( S , A ) - f ( S , A ) \big ) \Bigg ] - \mathrm { V a r } _ { p _ { s } } \big [ q ( S , \pi _ { e } ) - f ( S , \pi _ { e } ) \big ] \right) } \\ & { \qquad - \displaystyle \frac { 1 } { n } \sum _ { i } \frac { n _ { i } } { n } m _ { i } ( f ) ^ { 2 } \geq 0 . } \end{array}
$$

Below, the asymptotic excess variance of an estimator is its AVar minus $n \mathrm { V a r } [ \hat { J } _ { \Gamma } ^ { \star } ( \pi _ { e } ; \mathcal { D } ) ]$ , which equals n times the gap in Lemma E.3 evaluated at the limiting $f .$

Consistent reward estimation. We first explain how $\hat { J } _ { \mathrm { D R - o I P S } }$ fits into the $\hat { J } _ { \Gamma }$ class. For fixed $\hat { q }$ and $\alpha ,$ , the infeasible version of $\hat { J } _ { \mathrm { D R - o I P S } }$ has summand

$$
\hat { q } ( S , \pi _ { e } ) + \frac { \pi _ { e } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ( A \mid S ) } \bigl ( R - \hat { q } ( S , A ) \bigr ) - \frac { g _ { \alpha } ( S , A ) } { \pi _ { \mathrm { a v g } } ( A \mid S ) } + \sum _ { i \in [ K ] } \alpha _ { i } .
$$

Since

$$
\sum _ { i \in [ K ] } \alpha _ { i } = \frac { g _ { \alpha } } { \pi _ { e } } ( S , \pi _ { e } ) ,
$$

the last two terms simply replace $\hat { q }$ by $\hat { q } + g _ { \alpha } / \pi _ { e }$ in the DR regression function. Indeed, the above summand is equal to

$$
{ \frac { \pi _ { e } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ( A \mid S ) } } \left( R - \left( { \hat { q } } ( S , A ) + { \frac { g _ { \alpha } ( S , A ) } { \pi _ { e } ( A \mid S ) } } \right) \right) + \left( { \hat { q } } + { \frac { g _ { \alpha } } { \pi _ { e } } } \right) ( S , \pi _ { e } ) .
$$

Thus, the infeasible DR-oIPS estimator corresponds to

$$
\hat { J } _ { \Gamma } \Bigg ( \pi _ { e } , \hat { q } + \frac { g _ { \alpha } } { \pi _ { e } } ; \mathcal { D } \Bigg ) .
$$

The role of $_ { \pmb { \alpha } }$ is to choose $g _ { \alpha }$ as the projection of $( q - \hat { q } ) \pi _ { \epsilon }$ onto span $\left( \{ \pi _ { i } \} _ { i \in [ K ] } \right)$ . In other words, $\scriptstyle { \hat { J } } _ { \mathrm { D R - o I P S } }$ corrects $\hat { q }$ only in directions that can be represented by the logging policies.

If qˆ is consistent for $q ,$ then $( q - \hat { q } ) \pi _ { e }$ converges to zero, so its projection $g _ { \alpha }$ also converges to zero. Therefore,

$$
\hat { q } + \frac { g _ { \alpha } } { \pi _ { e } } \to q .
$$

Applying Lemma E.3 to the limiting regression function $f = q$ shows that the asymptotic excess variance of $\hat { J } _ { \mathrm { D R - o I P S } }$ over the oracle estimator $\hat { J } _ { \Gamma } ^ { \star } ( \pi _ { e } ; D ) : = \hat { J } _ { \Gamma } ( \pi _ { e } , q ; D )$ vanishes. By Kallus et al. [2021], the same is true for $\hat { J } _ { \mathrm { D R - b I P S } }$ when $\hat { q }$ is consistent for $q .$ Hence, in the well-specified case, $\scriptstyle { \hat { J } } _ { \mathrm { D R - o I P S } }$ and $\hat { J } _ { \mathrm { D R - b I P S } }$ have the same asymptotic variance, namely $n \mathrm { V a r } \big [ \hat { J } _ { \Gamma } ^ { \star } ( \pi _ { e } ; \mathcal { D } ) \big ]$

Misspecified reward estimation. Now suppose that $\hat { q }$ converges to a misspecified limit $q ^ { \prime } .$ Then J<sup>ˆ</sup><sub>DR-bIPS</sub> corresponds asymptotically to

$$
{ \hat { J } } _ { \Gamma } ( \pi _ { e } , q ^ { \prime } ; { \mathcal { D } } ) ,
$$

so Lemma $\mathrm { E } . 3$ characterizes its asymptotic excess variance over the oracle by taking $f = q ^ { \prime }$

By contrast, $\hat { J } _ { \mathrm { D R - o I P S } }$ corresponds asymptotically to

$$
\hat { J } _ { \Gamma } \bigg ( \pi _ { e } , q ^ { \prime } + \frac { g _ { \alpha ^ { \star } } } { \pi _ { e } } ; \mathcal { D } \bigg ) ,
$$

where $g _ { \alpha ^ { \star } }$ ⋆ is the projection of $\left( q - q ^ { \prime } \right) \pi _ { e } { \mathrm { ~ o n t o ~ s p a n } } \bigl ( \{ \pi _ { i } \} _ { i \in [ K ] } \bigr )$ . Therefore, compared with $\hat { J } _ { \mathrm { D R - b I P S } }$ $\scriptstyle { \hat { J } } _ { \mathrm { D R - o I P S } }$ replaces the misspecified regression function $q ^ { \prime }$ by the corrected regression function

$$
q ^ { \prime } + { \frac { g _ { \alpha ^ { \star } } } { \pi _ { e } } } .
$$

This removes the component of the misspecification that lies in the span of the logging policies. Lemma E.3 then implies that the asymptotic excess variance of $\hat { J } _ { \mathrm { D R - o I P S } }$ is no larger than that of $\hat { J } _ { \mathrm { D R - b I P S } }$ when both use the same limiting reward model $q ^ { \prime } .$ . Indeed, let $\begin{array} { r l } { Q _ { f } } & { { } : = } \end{array}$ $\pi _ { e } ( A ~ \mid ~ S ) { \big ( } R - f ( S , A ) { \big ) } / \pi _ { \mathrm { a v g } } ( A ~ \mid ~ S ) + f ( S , \pi _ { e } )$ denote the summand of (E.5) and $g ^ { \star } : =$ $g _ { \alpha ^ { \star } }$ . Then $\begin{array} { r } { Q _ { q ^ { \prime } } = Q _ { q ^ { \prime } + g ^ { \star } / \pi _ { e } } + g ^ { \star } / \pi _ { \mathrm { a v g } } - \sum _ { k } \alpha _ { k } ^ { \star } , \ m _ { i } ( q ^ { \prime } + g ^ { \star } / \pi _ { e } ) \ = \ 0 } \end{array}$ for every i, and $\mathrm { C o v } _ { \mu _ { \pi \mathrm { a v g } } } \left[ Q _ { q ^ { \prime } + g ^ { \star } / \pi _ { e } } , g ^ { \star } / \pi _ { \mathrm { a v g } } \right] = 0 ,$ , so $\imath \big ( \mathrm { V a r } [ \hat { J } _ { \Gamma } ( \pi _ { e } , q ^ { \prime } ; \mathcal { D } ) ] - \mathrm { V a r } [ \hat { J } _ { \Gamma } ( \pi _ { e } , q ^ { \prime } + { g ^ { \star } } / { \pi _ { e } } ; \mathcal { D } ) ] \big ) =$ $\begin{array} { r } { \sum _ { i } \frac { n _ { i } } { n } \mathrm { V a r } _ { \mu _ { \pi _ { i } } } [ g ^ { \star } / \pi _ { \mathrm { a v g } } ] \geq 0 . } \end{array}$

In particular, if

$$
( q - q ^ { \prime } ) \pi _ { e } \in \mathrm { s p a n } \bigl ( \{ \pi _ { i } \} _ { i \in [ K ] } \bigr ) ,
$$

then $g _ { \alpha ^ { \star } } = ( q - q ^ { \prime } ) \pi _ { e }$ , and hence

$$
q ^ { \prime } + \frac { g _ { \alpha ^ { \star } } } { \pi _ { e } } = q .
$$

Thus, $\scriptstyle { \hat { J } } _ { \mathrm { D R - o I P S } }$ attains the oracle asymptotic variance even though $\hat { q }$ is misspecified. In contrast, $\hat { J } _ { \mathrm { D R - b I P S } }$ remains tied to $q ^ { \prime }$ and may have nonzero asymptotic excess variance. This is the sense in which $\hat { J } _ { \mathrm { D R - o I P S } }$ is more robust to misspecification in directions spanned by the logging policies.

## B.4 Empirical Comparison with the Optimal Estimator of Kallus et al. [2021]

We provide an alternative, side-by-side presentation of the experimental results from $\ S 5 ,$ , in Figure 3 to more directly compare $\scriptstyle { \hat { J } } _ { \mathrm { D R - o I P S } }$ with $\hat { J } _ { \mathrm { D R - b I P S } }$ , which is known to be asymptotically optimal within the class studied by Kallus et al. [2021]. We also summarize the nuisance estimation procedures used in these experiments.

Nuisance Estimation. We summarize how nuisance components (logging policies and conditional rewards) are estimated.

• Logging policies $( \hat { \pi } _ { i } ) { : }$ We estimate logging policies with multinomial logistic regression and 2-fold cross-fitting. To maintain numerical stability in both the technique-matrix estimate $\hat { \mathbf { T } }$ and the denominator of Eq. (B.2), we clip the estimated propensities to the range $[ 1 0 ^ { - 3 } , 1 ]$ after 1% uniform smoothing.

• Conditional rewards (qˆ): We estimate $q ( s , a ) : = \mathbb { E } [ r | s , a ]$ via logistic regression with 2-fold cross-fitting, using the same protocol as for logging policies. Actions are represented with one-hot encodings.

Findings. In the setting of Figure $3 , \hat { J } _ { \mathrm { D R - o I P S } }$ has a lower Relative-RMSE than $\hat { J } _ { \mathrm { D R - b I P S } }$ in 16 of 28 configurations. Across all 184 configurations (92 with known and 92 with estimated propensities), it has a lower Relative-RMSE in 158, and the paired 95% interval for the difference lies below zero in 153.

![](images/a66936af9b6298bd7c193f733361178c33fc28fcce3ca66cf35299fd75b51b0f.jpg)  
Figure 3: Relative-RMSEs, computed over $M = 2 0 0$ independent experiments, are reported across different datasets and estimators under the setting where the logging policies are estimated. Shaded bands show 95% paired percentile-bootstrap intervals (§5). Where bands are not visible, they are narrower than the markers.

## C Empirical Results with $K > 2$ Logging Policies

We extend the two-logger experiment to heterogeneous mixtures with $K \in \{ 3 , 5 \}$ logging policies. As before, we train a multinomial logistic regression model on a training split to obtain a deterministic base policy $\pi _ { \mathrm { d e t } }$ and set the evaluation policy to $\pi _ { e } = \pi _ { \mathrm { d e t } }$ . The reward is deterministic: $r ( s , a ) : =$ $\mathbb { 1 } \left[ y ( s ) = a \right]$ , where $y ( s )$ is the label of context $s \in \mathbb { R } ^ { d }$ . The ground-truth policy value is therefore the classification accuracy of $\pi _ { e }$ on the OPE sample.

$$
J ( \pi _ { e } ) = \mathbb { E } _ { A \sim \pi _ { e } ( \cdot | S ) , S \sim p _ { s } } \big [ \mathbb { 1 } \left[ y ( S ) = A \right] \big ] .
$$

We construct $K \in \{ 3 , 5 \}$ logging policies by mixing $\pi _ { \mathrm { d e t } }$ with the uniform policy $\pi _ { u }$ (which ignores context and selects actions uniformly). For $K = 5$ , let $\beta _ { i } \in \{ 0 . 0 5 , 0 . 1 5 , 0 . 3 0 , 0 . 7 0 , 0 . 9 5 \}$ and define, for $i = 1 , \dots , K$

$$
\pi _ { i } ( a | s ) : = ( 1 - \beta _ { i } ) \cdot \pi _ { \mathrm { d e t } } ( a | s ) + \frac { \beta _ { i } } { | \mathcal { A } | } .
$$

The OPE sample is generated from an equal-weight logger mixture, with stratum weights $\tau _ { i } = 0 . 2$ for $K = 5$ and $\begin{array} { r } { \tau _ { i } = \frac { 1 } { 3 } } \end{array}$ for $K = 3 ( \ s o \bar { n _ { i } } / n \approx \bar { \tau _ { i } }$ , up to integer rounding). We partition the OPE sample into these strata, sample $a \sim \pi _ { i } ( \cdot \mid s )$ in stratum i, and assign reward $r ( s , a ) : = \mathbb { 1 } [ y ( s ) = a ]$ For $K = 3$ , we use $\beta _ { i } \in \{ 0 . { \bar { 0 } } 5 , 0 . 5 , 0 . 9 { \bar { 5 } } \}$

Results. Across settings with up to five loggers (Figure 4 and Figure 5), our logger-basis IPS estimator $\hat { J } _ { \mathrm { o I P S } }$ generally achieves strong accuracy when the logging policies are known, while its doubly robust extension $\hat { J } _ { \mathrm { D R - o I P S } }$ tends to perform best when the logging policies are estimated.

When logging policies are estimated, $\hat { J } _ { \mathrm { D R - o I P S } }$ outperforms $\hat { J } _ { \mathrm { D R - b I P S } }$ on all four datasets for $K = 5$ and three of four for $K = 3$ . Among the non-DR estimators, $\hat { J } _ { \mathrm { I P S } }$ and $\hat { J } _ { \mathrm { w I P S } }$ can become unstable (particularly on OptDigits). This instability appears when estimated logging propensities are near zero or noisy in specific strata, yielding extreme $1 / \hat { \pi } _ { i } ( a | s )$ weights and heavy-tailed variance. By comparison, the sample-dependent weighting in $\hat { J } _ { \mathrm { o I P S } }$ and $\hat { J } _ { \mathrm { D R - o I P S } }$ better controls variance and produces substantially more stable estimates. The same qualitative pattern holds for three loggers (Figures 6 and 7).

![](images/42596271254c46dd643a686c098fae45b05adf702123f8beb99d5906de419904.jpg)

![](images/fc065598c5f9fbb8a91aac61ca5188109b1959f1521057cecd0c4bdae7a5e25d.jpg)

![](images/78eba3dd31b6d1608a5c3891ef8f9644a35d03887d0ad0bc436799c874a1f112.jpg)

![](images/21815adde176fdd1608eaeedf171d27fd5096ccb308f0770ddb4ca2a2a189f9a.jpg)  
Figure 4: Rel-RMSE across five heterogeneous logging policies where they are known. Lower is better. Error bars show 95% paired percentile-bootstrap intervals (§5).

![](images/4ca4824c1b295851fc1bc03430040d06dad02475f5f93204b3a4fcf329d371db.jpg)

![](images/973a91f6f0171d24e0949c5687f2bdbeeb8f917e7a57922d60f7be2e3738edb8.jpg)

![](images/ce88e2a69ceebbb0110bfb112966c4e2523d40e1b915b6a7c18f930c497e9008.jpg)

![](images/e7f7f68be647e7ef63d01a4cff37e7cc12553d52c8d224e22db8f79ed6bed1af.jpg)  
Figure 5: Rel-RMSE across five heterogeneous logging policies where they are estimated. Lower is better. Error bars show 95% paired percentile-bootstrap intervals (§5). Where error bars are not visible, they are shorter than the markers.

![](images/a3c08a0f4ec2626519f161194d2a7d6a57938fb529e58235265aef9e0bf30c33.jpg)

![](images/40aad37a22f2523a379fb989b560e0379f6fb969581048ca5dd8bd74c8e22096.jpg)

![](images/5f7d7abdcbaa20453601d657d0317836281ad12f57ccc8b449cca8c9ea87c589.jpg)

![](images/d1a8dc934b972850041caf61c2ecabb86f566c2f3d8ac87552829383c083a5a2.jpg)

Figure 6: Rel-RMSE across three heterogeneous logging policies where they are known. Lower is better. Error bars show 95% paired percentile-bootstrap intervals (§5).  
![](images/4e9215f21a6612b20bc19f09183c5e8f2fd0202f020daff6a7cb19358107e75d.jpg)

![](images/0b9f34c4f1a017015e46ee36f557212ab38bcee727fe5b664bc1693e3cc7a4cf.jpg)

![](images/0708defde43eff37348e2009cadf66d46f9e8967f88d65fe3d7f4ea83d0c8f50.jpg)

![](images/7fe94a32566e240f5c834121fd15d00d7c9859d9cd6448e181e10cc5e07a38bf.jpg)  
Figure 7: Rel-RMSE across three heterogeneous logging policies where they are estimated. Lower is better. Error bars show 95% paired percentile-bootstrap intervals (§5). Where error bars are not visible, they are shorter than the markers.

## D Sensitivity to the Additive Mixture of Logging Policies

We repeat the main experiments with alternative mixture ratios between the deterministic policy $\pi _ { \mathrm { d e t } }$ and the uniform policy $\pi _ { u } .$ . For $\eta \in \{ 0 . 8 0 , 0 . 6 5 \}$ , define

$$
\pi _ { 1 } ^ { ( \eta ) } = \eta \pi _ { \mathrm { d e t } } + ( 1 - \eta ) \pi _ { u } , \quad \pi _ { 2 } ^ { ( \eta ) } = ( 1 - \eta ) \pi _ { \mathrm { d e t } } + \eta \pi _ { u } .
$$

All other settings, including the evaluation policy $\pi _ { e } = \pi _ { \mathrm { d e t } } .$ , datasets, train/evaluation split ratio, stratum-size ratios $n _ { 1 } / n _ { 2 }$ , number of repeats $M = 2 0 0$ , and non-DR cross-fitting folds $Z = 5 ,$ , match the main-text protocol. Both DR estimators use the shared two-fold procedure of Appendix B.

Results. Across both additive mixtures $( \eta = 0 . 8 0$ and $\eta = 0 . 6 5 )$ , the qualitative trends in Figures 8, 9, 10, and 11 remain consistent with the main-text results. Our logger-basis IPS estimator and its DR extension $( \hat { J } _ { \mathrm { o I P S } }$ and $\hat { J } _ { \mathrm { D R - o I P S } } )$ remain among the strongest methods in known- and estimated-logging settings, respectively. Improvements are most pronounced in low-overlap regimes (small $n _ { 1 } / n _ { 2 } )$ , while gaps narrow as overlap increases. Estimated propensities increase absolute errors and can change which estimator ranks first.

![](images/77bbdd767cc6fb278438f9c1c8fc2da4df10e7b3b19aea02d6d0c2ca3f836158.jpg)  
Figure 8: Mixture $\eta = 0 . 8 0 .$ . Relative-RMSEs, computed over $M = 2 0 0$ independent experiments, are reported across different datasets and estimators under the setting where the logging policies are known. Shaded bands show 95% paired percentile-bootstrap intervals (§5).

![](images/8c4ba1e22a9d75ea3c9cdb48afca037ae84926306bb3cc47b4447bc9fdd91462.jpg)  
Figure 9: Mixture $\eta = 0 . 8 0 .$ . Relative-RMSEs, computed over $M = 2 0 0$ independent experiments, are reported across different datasets and estimators under the setting where the logging policies are estimated. Shaded bands show 95% paired percentile-bootstrap intervals (§5). Where bands are not visible, they are narrower than the markers.

![](images/6922724b930a28f07b091484f73bd46efed99fca00f8af0660927315984f8188.jpg)  
Figure 10: Mixture η = 0.65. Relative-RMSEs, computed over M = 200 independent experiments, are reported across different datasets and estimators under the setting where the logging policies are known. Shaded bands show 95% paired percentile-bootstrap intervals (§5).

![](images/2611aec66f0fc878d927dd59ddfa34ade712ea1b393489e09fff8ad9b92983b0.jpg)  
Figure 11: Mixture $\eta = 0 . 6 5 .$ . Relative-RMSEs, computed over $M = 2 0 0$ independent experiments, are reported across different datasets and estimators under the setting where the logging policies are estimated. Shaded bands show 95% paired percentile-bootstrap intervals (§5). Where bands are not visible, they are narrower than the markers.

## E Deferred Proofs

In this section, we present detailed proofs of the main theorems/lemmas.

## E.1 Proof of Unbiasedness of Generalized Weighted-IPS Estimators

Theorem 3.1. [Unbiasedness] For any weights $\mathbf { w } \in ( L ^ { 2 } ) ^ { K }$ that satisfy the normalization and dominance conditions and for which each weighted score $w _ { i } ( S , A , R ) \pi _ { e } ( A \mid S ) / \pi _ { i } ( A \mid S )$ is $\mu _ { \pi _ { i } }$ -integrable, $\hat { J } ( \pi _ { e } , \mathbf { w } ; \mathcal { D } )$ is unbiased, that is, $J ( \pi _ { e } ) = \mathbb { E } [ \hat { J } ( \pi _ { e } , \mathbf { w } ; \mathcal { D } ) ]$ , where the expectation is over the sampling randomness oflogged data $\mathcal { D } : = ( \mathcal { D } _ { i } ) _ { i \in [ K ] }$

Proof of Theorem 3.1. Let $\omega : = ( s , a , r ) \in \Omega$ and $\mu _ { \pi _ { i } }$ be the probability distribution with density $p _ { r } ( r | s , a ) \pi _ { i } ( a | s ) p _ { s } ( s )$ with respect to $\mu .$ . Recall the general weighted IPS estimator,

$$
\widehat { J } ( \pi _ { e } , \mathbf { w } ; \mathcal { D } ) : = \sum _ { i = 1 } ^ { K } \frac { 1 } { n _ { i } } \sum _ { j = 1 } ^ { n _ { i } } \frac { w _ { i } ( S _ { i j } , A _ { i j } , R _ { i j } ) \pi _ { e } ( A _ { i j } \mid S _ { i j } ) } { \pi _ { i } ( A _ { i j } \mid S _ { i j } ) } : = \sum _ { i = 1 } ^ { K } \frac { 1 } { n _ { i } } \sum _ { j = 1 } ^ { n _ { i } } \widehat { J } _ { i j } ( w _ { i } ; \mathcal { D } _ { i } ) .
$$

Taking the expectation over data,

$$
\begin{array} { r l } { \displaystyle { d \big [ \widetilde { B } ( z , \mathbf { u } , \mathbf { u } ; \mathbf { p } , \mathbf { J } ) \big ] - \sum _ { i = 1 } ^ { N } \frac { 1 } { \mu } \sum _ { i = 1 } ^ { N } \frac { \sqrt { N } } { 2 } d \widetilde { \mu } \big ( \hat { \mu } _ { i } ( \mathbf { p } _ { i } ; \mathbf { x } , \mathbf { u } ) \big ) } } \\ { \displaystyle } & { = \sum _ { i = 1 } ^ { N } \frac { 1 } { \mu } \sum _ { i = 1 } ^ { N } \int _ { \mathbf { p } \cap \mathcal { S } _ { i } \cap \mathcal { S } _ { i } \cap \mathcal { S } _ { i } } \mathrm { d } \boldsymbol { \mu } \mathrm { e } ^ { - \frac { \mathbf { E } _ { \mathbf { p } } \cdot \mathbf { E } _ { \mathbf { p } } \cdot \mathbf { E } _ { \mathbf { p } } \cdot \mathbf { E } _ { \mathbf { p } } \cdot \mathbf { E } _ { \mathbf { p } } \cdot \mathbf { E } _ { \mathbf { p } } \cdot ( \mathbf { c } ) } { 2 \nu } } } \\ & { \quad - \sum _ { i = 1 } ^ { N } \frac { 1 } { \mu } \sum _ { i = 1 } ^ { N } \int _ { \mathbf { p } \cap \mathcal { S } _ { i } \cap \mathcal { S } _ { i } } \mathrm { d } \boldsymbol { \mu } \mathrm { e } ^ { - \frac { \mathbf { E } _ { \mathbf { p } } \cdot \mathbf { E } _ { \mathbf { p } } \cdot \mathbf { E } _ { \mathbf { p } } \cdot ( \mathbf { c } ) } { 2 \nu } } \mathrm { d } \boldsymbol { \mu } \mathrm { e } ^ { \mathrm { i } \cdot \mathbf { c } \cdot \mathbf { p } } \mathrm { d } \boldsymbol { \mu } \mathrm { e } ^ { \mathrm { i } \cdot \mathbf { c } \cdot \mathbf { p } } } \\ &  \quad - \sum _ { i = 1 } ^ { N } \frac { 1 } { \mu } \sum _ { i = 1 } ^ { N } \int _ { \mathbf { p } \cap \mathcal { S } _ { i } \cap \mathcal { S } _ { i } } \mathrm { d } \boldsymbol { \mu } \mathrm { e } ^  \mathrm { i } \cdot  \end{array}
$$

## E.2 Proof of Theorem 4.1.

In this section, we prove Theorem 4.1. Let $\mathbf { w } ^ { \star } : = ( w _ { i } ^ { \star } ) _ { i \in [ K ] }$ be defined by

$$
w _ { i } ^ { \star } ( s , a , r ) = \frac { \alpha _ { i } ^ { \star } \pi _ { i } ( a | s ) } { \pi _ { e } ( a | s ) } + \frac { n _ { i } \pi _ { i } ( a | s ) } { n \pi _ { \mathrm { a v g } } ( a | s ) } \left( r - \frac { \sum _ { j = 1 } ^ { K } \alpha _ { j } ^ { \star } \pi _ { j } ( a | s ) } { \pi _ { e } ( a | s ) } \right)
$$

where ${ \pmb { \alpha } } ^ { \star } : = ( { \pmb { \alpha } } _ { i } ^ { \star } ) _ { i \in [ K ] }$ is a solution of $\mathbf { T } \alpha = \mathbf { c } ,$ . Here, $\mathbf { T } \in \mathbb { R } ^ { K \times K }$ and $\mathbf { c } \in \mathbb { R } ^ { K }$ have components

$$
T _ { i j } : = \mathbb { E } _ { \mu _ { \tau _ { i } } } \left[ \frac { \pi _ { j } ( \boldsymbol { A } \mid \boldsymbol { S } ) } { \pi _ { \mathrm { a v g } } ( \boldsymbol { A } \mid \boldsymbol { S } ) } \right] , \quad c _ { i } : = \mathbb { E } _ { \mu _ { \tau _ { i } } } \left[ \boldsymbol { R } \frac { \pi _ { e } ( \boldsymbol { A } \mid \boldsymbol { S } ) } { \pi _ { \mathrm { a v g } } ( \boldsymbol { A } \mid \boldsymbol { S } ) } \right] .
$$

The theorem states that $\mathbf { w } ^ { \star }$ is an optimal solution to the variance minimization problem (4.1):

$$
\begin{array} { r l r l } & { \underset { \mathbf { w } \in ( L ^ { 2 } ) ^ { K } } { \mathrm { m i n i m i z e } } } & & { \mathrm { V a r } \big [ \hat { J } ( \pi _ { e } , \mathbf { w } ; \mathcal { D } ) \big ] } \\ & { \mathrm { ~ } } & & { } \\ & { \mathrm { s u b j e c t ~ t o } } & { \displaystyle } & { \displaystyle } & { \sum _ { i = 1 } ^ { K } w _ { i } ( \omega ) = r , \quad \forall \omega \in \mathrm { s u p p } ( \mu _ { \pi _ { e } } ) } \end{array}
$$

Variance Characterization. We begin with a variance characterization of the generalized weighted-IPS estimator $\hat { J } ( \pi _ { e } , \mathbf { w } ; \mathcal { D } )$ in (3.1). This characterization enables functional differentiation with respect to w.

Lemma E.1. For any weights ${ \bf w } \in ( L ^ { 2 } ) ^ { K }$ whose weighted scores $w _ { i } ( S , A , R ) \pi _ { e } ( A \mid S ) / \pi _ { i } ( A \mid S )$ havefinite second moments under $\mu _ { \pi _ { i } }$ ，

$$
\mathrm { V a r } \big [ \hat { J } ( \pi _ { e } , { \bf w } ; { \mathcal D } ) \big ] = \sum _ { i = 1 } ^ { K } \frac { 1 } { n _ { i } } \left( \Big \langle w _ { i } ^ { 2 } , \frac { \pi _ { e } } { \pi _ { i } } \Big \rangle _ { \mu _ { \pi _ { e } } } - \langle w _ { i } , 1 \rangle _ { \mu _ { \pi _ { e } } } ^ { 2 } \right) .
$$

Proof of Lemma E.1. As $\hat { J } _ { i j } ( w _ { i } ; \mathcal { D } _ { i } )$ are independent to each other,

$$
\begin{array} { l } { { \displaystyle \mathrm { V a r } [ \hat { J } ( \pi _ { e } , { \bf w } ; \mathcal { D } ) ] = \mathrm { V a r } \left[ \sum _ { i = 1 } ^ { K } \frac { 1 } { n _ { i } } \sum _ { j = 1 } ^ { n _ { i } } \hat { J } _ { i j } ( w _ { i } ; \mathcal { D } _ { i } ) \right] = \sum _ { i = 1 } ^ { K } \frac { 1 } { n _ { i } ^ { 2 } } \sum _ { j = 1 } ^ { n _ { i } } \mathrm { V a r } \left[ \hat { J } _ { i j } ( w _ { i } ; \mathcal { D } _ { i } ) \right] } } \\ { { \displaystyle \qquad = \sum _ { i = 1 } ^ { K } \frac { 1 } { n _ { i } ^ { 2 } } \sum _ { j = 1 } ^ { n _ { i } } \mathbb { E } \left[ \hat { J } _ { i j } ^ { 2 } ( w _ { i } ; \mathcal { D } _ { i } ) \right] - \sum _ { i = 1 } ^ { K } \frac { 1 } { n _ { i } ^ { 2 } } \sum _ { j = 1 } ^ { n _ { i } } \mathbb { E } \left[ \hat { J } _ { i j } ( w _ { i } ; \mathcal { D } _ { i } ) \right] ^ { 2 } . } } \end{array}
$$

The first term is

$$
\begin{array} { l } { \displaystyle \sum _ { i = 1 } ^ { K } \displaystyle \frac { 1 } { n _ { i } ^ { 2 } } \sum _ { j = 1 } ^ { n _ { i } } \mathbb { E } \left[ \hat { J } _ { i j } ^ { 2 } ( w _ { i } ; \mathcal { D } _ { i } ) \right] = \sum _ { i = 1 } ^ { K } \displaystyle \frac { 1 } { n _ { i } ^ { 2 } } \sum _ { j = 1 } ^ { n _ { i } } \int _ { \Omega } \displaystyle \frac { w _ { i } ^ { 2 } ( \omega ) \pi _ { e } ^ { 2 } ( a | s ) } { \pi _ { i } ^ { 2 } ( a | s ) } p _ { r } ( r | s , a ) \pi _ { i } ( a | s ) p _ { s } ( s ) d \mu ( \omega ) } \\ { = \displaystyle \sum _ { i = 1 } ^ { K } \displaystyle \frac { 1 } { n _ { i } } \int _ { \Omega } \displaystyle \frac { w _ { i } ^ { 2 } ( \omega ) \pi _ { e } ( a | s ) } { \pi _ { i } ( a | s ) } p _ { r } ( r | s , a ) \pi _ { e } ( a | s ) p _ { s } ( s ) d \mu ( \omega ) } \\ { = \displaystyle \sum _ { i = 1 } ^ { K } \displaystyle \frac { 1 } { n _ { i } } \int _ { \Omega } w _ { i } ^ { 2 } ( \omega ) \displaystyle \frac { \pi _ { e } ( a | s ) } { \pi _ { i } ( a | s ) } d \mu _ { \pi _ { \kappa } ( \omega ) } } \\ { = \displaystyle \sum _ { i = 1 } ^ { K } \displaystyle \frac { 1 } { n _ { i } } \left. w _ { i } ^ { 2 } , \displaystyle \frac { \pi _ { e } } { \pi _ { i } } \right. _ { \mu _ { \tau _ { \tau } } } . } \end{array}
$$

The second term is

$$
\begin{array} { l } { \displaystyle \sum _ { i = 1 } ^ { K } \frac { 1 } { n _ { i } ^ { 2 } } \sum _ { j = 1 } ^ { n _ { i } } \mathbb { E } \big [ \hat { J } _ { i j } ( w _ { i } ; \mathcal { D } _ { i } ) \big ] ^ { 2 } = \displaystyle \sum _ { i = 1 } ^ { K } \frac { 1 } { n _ { i } ^ { 2 } } \sum _ { j = 1 } ^ { n _ { i } } \bigg ( \int _ { \Omega } \frac { w _ { i } ( \omega ) \pi _ { c } ( a | s ) } { \pi _ { i } ( a | s ) } p _ { r } ( r | s , a ) \pi _ { i } ( a | s ) p _ { s } ( s ) d \mu ( \omega ) \bigg ) ^ { 2 } } \\ { = \displaystyle \sum _ { i = 1 } ^ { K } \frac { 1 } { n _ { i } } \left( \int _ { \Omega } w _ { i } ( \omega ) p _ { r } ( r | s , a ) \pi _ { \epsilon } ( a | s ) p _ { s } ( s ) d \mu ( \omega ) \right) ^ { 2 } } \\ { = \displaystyle \sum _ { i = 1 } ^ { K } \frac { 1 } { n _ { i } } \left( \int _ { \Omega } w _ { i } ( \omega ) d \mu _ { \pi _ { c } } ( \omega ) \right) ^ { 2 } } \\ { = \displaystyle \sum _ { i = 1 } ^ { K } \frac { 1 } { n _ { i } } \left. w _ { i } , 1 \right. _ { \mu _ { \pi _ { i } } } ^ { 2 } . } \end{array}
$$

Proof of Optimality of $\mathbf { w } ^ { \star }$ . We now prove Theorem 4.1, which is restated below.

Theorem 4.1. The weight vector $\mathbf { w } ^ { \star }$ given in (3.2) solves the variance minimization problem (4.1).

ProofofTheorem 4.1. Since the objective in (4.1) is the variance of a linear functional of $\mathbf { w } ,$ the variance minimization problem subject to the normalization constraint is convex. We therefore derive the first-order stationarity condition for the Lagrange-augmented functional $I [ \mathbf { w } , \lambda ]$ defined below.

Recall $\rho _ { r } ( \omega ) = r \mathrm { f o r } \omega = ( s , a , r )$ . Define the functional $I [ \mathbf { w } , \lambda ]$ as

$$
\begin{array} { l } { { \displaystyle { \cal I } [ { \bf w } , \lambda ] : = \mathrm { V a r } \big [ \hat { J } ( \pi _ { e } , { \bf w } ; \mathcal { D } ) \big ] - \left. \lambda , \displaystyle \sum _ { i = 1 } ^ { K } w _ { i } - \rho _ { r } \right. _ { \mu _ { \pi _ { e } } } } } \\ { { \displaystyle ~ = \mathrm { V a r } \big [ \hat { J } ( \pi _ { e } , { \bf w } ; \mathcal { D } ) \big ] - \sum _ { i = 1 } ^ { K } \big \langle \lambda , w _ { i } \big \rangle _ { \mu _ { \pi _ { e } } } + \big \langle \lambda , \rho _ { r } \big \rangle _ { \mu _ { \pi _ { e } } } } , } \end{array}
$$

where $\lambda : \Omega \to$ R is the Lagrange multiplier enforcing the normalization constraint.

Next, for a perturbation direction $\pmb { \delta } ( \omega ) : = ( \delta _ { i } ( \omega ) ) _ { i \in [ K ] }$ and $\epsilon \in \mathbb { R } ^ { K }$ , define $I ( \epsilon ; \delta , \mathbf { w } , \lambda )$ as

$$
I ( \epsilon ; \delta , \mathbf { w } , \lambda ) : = I [ w _ { 1 } + \epsilon _ { 1 } \delta _ { 1 } , \dots , w _ { K } + \epsilon _ { K } \delta _ { K } , \lambda ] ,
$$

which can be interpreted as the variation of $I [ \mathbf { w } , \lambda ]$ under perturbations along the direction $\delta .$

To verify optimality, it suffices to exhibit feasible $\mathbf { w } ^ { \star }$ and $\lambda ^ { \star }$ such that, for every perturbation $\delta$ with finite weighted second moments and each $i \in [ K ]$

$$
\left. \frac { \partial I ( \epsilon ; \delta , { \bf w } ^ { \star } , \lambda ^ { \star } ) } { \partial \epsilon _ { i } } \right| _ { \epsilon = 0 } = 0 .
$$

By Lemma E.1, this is equivalent to

$$
\left. \delta _ { i } , \frac { 2 w _ { i } ^ { \star } \pi _ { e } } { n _ { i } \pi _ { i } } - \frac { 2 \left. w _ { i } ^ { \star } , 1 \right. _ { \mu _ { \pi _ { e } } } } { n _ { i } } - \lambda ^ { \star } \right. _ { \mu _ { \pi _ { e } } } = 0 .\tag{E.1}
$$

Next, we deduce $\mathbf { w } ^ { \star }$ and $\lambda ^ { \star }$ from (E.1) and verify that the deduced $\mathbf { w } ^ { \star }$ is indeed a minimizer.

To proceed, let us define $\alpha _ { i } : = \left. w _ { i } ^ { \star } , 1 \right. _ { \mu _ { \pi _ { e } } }$ . Then, (E.1) implies that, for almost every $\omega : = ( s , a , r ) \in$ $\operatorname { s u p p } ( \mu _ { \pi _ { e } } )$

$$
\frac { 2 w _ { i } ^ { \star } ( \omega ) \pi _ { e } ( a | s ) } { n _ { i } \pi _ { i } ( a | s ) } - \frac { 2 \alpha _ { i } } { n _ { i } } - \lambda ^ { \star } ( \omega ) = 0 .
$$

Rearranging the above and summing over $i \in [ K ]$ , we obtain

$$
\begin{array} { c l } { \displaystyle w _ { i } ^ { \star } ( \omega ) = \frac { \alpha _ { i } \pi _ { i } ( a | s ) } { \pi _ { e } ( a | s ) } + \frac { n _ { i } \pi _ { i } ( a | s ) } { 2 \pi _ { e } ( a | s ) } \lambda ^ { \star } ( \omega ) } \\ { \displaystyle \implies r = \sum _ { i = 1 } ^ { K } w _ { i } ^ { \star } ( \omega ) = \frac { \sum _ { i = 1 } ^ { K } \alpha _ { i } \pi _ { i } ( a | s ) } { \pi _ { e } ( a | s ) } + \frac { \sum _ { i = 1 } ^ { K } n _ { i } \pi _ { i } ( a | s ) } { 2 \pi _ { e } ( a | s ) } \lambda ^ { \star } ( \omega ) , } \end{array}\tag{E.2}
$$

where we used the fact that ${ \begin{array} { r } { \sum _ { i = 1 } ^ { K } w _ { i } ^ { \star } ( \omega ) = r } \end{array} }$ for all $\omega \in \mathrm { s u p p } \left( \mu _ { \pi _ { e } } \right)$

Solving the above for $\lambda ^ { \star } ( \omega )$

$$
\lambda ^ { \star } ( \omega ) = \frac { 2 r \pi _ { e } ( a | s ) - 2 \sum _ { i = 1 } ^ { K } \alpha _ { i } \pi _ { i } ( a | s ) } { \sum _ { i = 1 } ^ { K } n _ { i } \pi _ { i } ( a | s ) } .
$$

Plugging $\lambda ^ { \star } ( \omega )$ back into (E.2) results in

$$
\begin{array} { c } { \displaystyle w _ { i } ^ { \star } ( \omega ) = \frac { \alpha _ { i } \pi _ { i } ( a | s ) } { \pi _ { e } ( a | s ) } + \frac { n _ { i } \pi _ { i } ( a | s ) } { \sum _ { j = 1 } ^ { K } n _ { j } \pi _ { j } ( a | s ) } \left( r - \frac { \sum _ { j = 1 } ^ { K } \alpha _ { j } \pi _ { j } ( a | s ) } { \pi _ { e } ( a | s ) } \right) , } \\ { \displaystyle = \frac { \alpha _ { i } \pi _ { i } ( a | s ) } { \pi _ { e } ( a | s ) } + \frac { n _ { i } } { n } \frac { \pi _ { i } ( a | s ) } { \pi _ { \mathrm { a v g } } ( a | s ) } \left( r - \frac { \sum _ { j = 1 } ^ { K } \alpha _ { j } \pi _ { j } ( a | s ) } { \pi _ { e } ( a | s ) } \right) } \end{array}\tag{E.3}
$$

where one can easily verify that the omitted constraint ${ } ^ { \mathfrak { c } \mathfrak { c } } \pi _ { i } ( a | s ) = 0 \implies \ w _ { i } ( \omega ) = 0$ for all $i \in [ K ] ^ { \ ' }$ is satisfied for any α.

Next, we derive a self-consistency equation for α by plugging the above expression for $w _ { i } ^ { \star } ( \omega )$ into $\alpha _ { i } : = \langle w _ { i } ^ { \star } , 1 \rangle _ { \mu _ { \pi _ { e } } }$

$$
\begin{array} { l } { \displaystyle \alpha _ { i } = \alpha _ { i } \mathbb { E } _ { \mu _ { \pi _ { e } } } \left[ \frac { \pi _ { i } ( \boldsymbol { A } \mid \boldsymbol { S } ) } { \pi _ { e } ( \boldsymbol { A } \mid \boldsymbol { S } ) } \right] + \frac { n _ { i } } { n } \mathbb { E } _ { \mu _ { \pi _ { e } } } \left[ \frac { R \pi _ { i } ( \boldsymbol { A } \mid \boldsymbol { S } ) } { \pi _ { \mathrm { a v g } } ( \boldsymbol { A } \mid \boldsymbol { S } ) } \right] } \\ { \displaystyle \qquad - \frac { n _ { i } } { n } \sum _ { j = 1 } ^ { K } \alpha _ { j } \mathbb { E } _ { \mu _ { \pi _ { e } } } \left[ \frac { \pi _ { i } ( \boldsymbol { A } \mid \boldsymbol { S } ) } { \pi _ { \mathrm { a v g } } ( \boldsymbol { A } \mid \boldsymbol { S } ) } \frac { \pi _ { j } ( \boldsymbol { A } \mid \boldsymbol { S } ) } { \pi _ { e } ( \boldsymbol { A } \mid \boldsymbol { S } ) } \right] . } \end{array}\tag{E.4}
$$

Since

$$
\begin{array} { r l } { \mathbb { E } _ { \mu _ { \pi _ { e } } } \left[ \frac { \pi _ { i } \left( A \mid S \right) } { \pi _ { e } \left( A \mid S \right) } \right] = \displaystyle \int _ { \Omega } \frac { \pi _ { i } ( a \mid s ) } { \pi _ { e } ( a \mid s ) } p _ { r } ( r \mid s , a ) \pi _ { e } ( a \mid s ) p _ { s } ( s ) d \mu ( \omega ) } & { } \\ & { = \displaystyle \int _ { \Omega } p _ { r } ( r \mid s , a ) \pi _ { i } ( a \mid s ) p _ { s } ( s ) d \mu ( \omega ) } \\ & { = \displaystyle \int _ { \Omega } \mu _ { \pi _ { i } } ( \omega ) = 1 , } \end{array}
$$

and

$$
\begin{array} { r l } & { T _ { i j } : = \mathbb { E } _ { \mu _ { \pi _ { i } } } \left[ \frac { \pi _ { j } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ( A \mid S ) } \right] = \mathbb { E } _ { \mu _ { \pi _ { e } } } \left[ \frac { \pi _ { i } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ( A \mid S ) } \frac { \pi _ { j } ( A \mid S ) } { \pi _ { e } ( A \mid S ) } \right] } \\ & { c _ { i } : = \mathbb { E } _ { \mu _ { \pi _ { i } } } \left[ \frac { R \pi _ { e } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ( A \mid S ) } \right] = \mathbb { E } _ { \mu _ { \pi _ { e } } } \left[ \frac { R \pi _ { i } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ( A \mid S ) } \right] } \end{array}
$$

The self-consistency equation (E.4) can be written as ${ \bf T } \alpha = { \bf c }$ with T and c given in (3.3).

Finally, by Lemma E.2, $\mathbf { T } \alpha = \mathbf { c }$ admits a solution. Hence, the corresponding weights w satisfy the first-order optimality condition (E.1) and therefore solve the variance minimization convex program (4.1). □

## E.3 Proof of Unbiasedness and Asymptotic Variance Optimality of $\hat { J } _ { \mathrm { o I P S } }$

In this section, we prove that $\hat { J } _ { \mathrm { o I P S } }$ is unbiased and asymptotically normal, with asymptotic variance $n \mathrm { V a r } \big [ \hat { J } ( \pi _ { e } , \mathbf { w } ^ { \star } ; \mathcal { D } ) \big ]$ , which is optimal within $\Gamma _ { \mathrm { w I P S } }$

Theorem 4.2. Let $n ^ { \prime } : = \textstyle \sum _ { i = 1 } ^ { K } n _ { i } ^ { \prime }$ with $n _ { i } ^ { \prime } : = m n _ { i }$ . Under Assumption 1, when the logging policies are known, $\hat { J } _ { \mathrm { o I P S } }$ is unbiased, and

$$
\sqrt { n ^ { \prime } } \left( \hat { J } _ { \mathrm { o I P S } } ( \pi _ { e } ; \mathcal { D } ) - J ( \pi _ { e } ) \right) \overset { d } { \to } \mathcal { N } \left( 0 , n \mathrm { V a r } [ \hat { J } ( \pi _ { e } , \mathbf { w } ^ { \star } ; \mathcal { D } ) ] \right)
$$

as $m  \infty ,$ , where $n : = \textstyle \sum _ { i = 1 } ^ { K } n _ { i }$ and $\operatorname { V a r } [ \hat { J } ( \pi _ { e } , \mathbf { w } ^ { \star } ; \mathcal { D } ) ]$ is computed at the base stratum sizes $( n _ { i } ) _ { i = 1 } ^ { K }$

Proof of Theorem 4.2. We first prove the unbiasedness of $\hat { J } _ { \mathrm { o I P S } } ( \pi _ { e } ; \mathcal { D } )$ when the logging policies are known.

Unbiasedness. We begin by rewriting the expectation of $\hat { J } _ { \mathrm { o I P S } } ( \pi _ { e } ; \mathcal { D } )$ as an iterated expectation, conditioning on the data $\mathcal { D } ^ { ( - z ) }$ used to estimate the nuisance parameter α:

$$
\begin{array} { r l } { \displaystyle \mathbb { E } \left[ \hat { J } _ { \mathrm { o I P S } } ( \pi _ { e } ; \mathcal { D } ) \right] = \frac { 1 } { n } \sum _ { z = 1 } ^ { Z } n ^ { ( z ) } \mathbb { E } \left[ \hat { J } ( \pi _ { e } , \hat { \pmb { \alpha } } ^ { ( z ) } ; \mathcal { D } ^ { ( z ) } ) \right] } & { } \\ { \displaystyle } & { = \frac { 1 } { n } \sum _ { z = 1 } ^ { Z } n ^ { ( z ) } \mathbb { E } \left[ \mathbb { E } \left[ \hat { J } ( \pi _ { e } , \hat { \pmb { \alpha } } ^ { ( z ) } ; \mathcal { D } ^ { ( z ) } ) \left| \mathcal { D } ^ { ( - z ) } \right. \right] \right] } \end{array}
$$

It therefore suffices to show that

$$
\mathbb { E } \big [ \hat { J } ( \pi _ { e } , \hat { \pmb { \alpha } } ^ { ( z ) } ; \mathcal { D } ^ { ( z ) } ) \big | \mathcal { D } ^ { ( - z ) } \big ] = J ( \pi _ { e } ) ,
$$

for each fold z. To see this, let $\hat { \mathbf { w } } ^ { ( z ) }$ be defined by

$$
\hat { w } _ { i } ^ { ( z ) } ( s , a , r ) : = \frac { \hat { \alpha } _ { i } ^ { ( z ) } \pi _ { i } ( a | s ) } { \pi _ { e } ( a | s ) } + \frac { n _ { i } ^ { ( z ) } \pi _ { i } ( a | s ) } { \sum _ { j = 1 } ^ { K } n _ { j } ^ { ( z ) } \pi _ { j } ( a | s ) } \left( r - \frac { \sum _ { k = 1 } ^ { K } \hat { \alpha } _ { k } ^ { ( z ) } \pi _ { k } ( a | s ) } { \pi _ { e } ( a | s ) } \right)
$$

Then, conditional on $\hat { \pmb { \alpha } } ^ { ( z ) }$ , these weights satisfy the normalization condition $\textstyle \sum _ { i = 1 } ^ { K } \hat { w } _ { i } ^ { ( z ) } ( s , a , r ) = r$ and the dominance condition $\pi _ { i } ( a | s ) = 0 \implies \hat { w } _ { i } ^ { ( z ) } ( s , a , r ) = 0$

Moreover, $\hat { J } ( \pi _ { e } , \hat { \pmb { \alpha } } ^ { ( z ) } ; \mathcal { D } ^ { ( z ) } )$ in (3.4) can be rewritten in the canonical generalized weighted-IPS form (3.1), with $\hat { \mathbf { w } } ^ { ( z ) }$ as its weights:

$$
\hat { J } \big ( \pi _ { e } , \hat { \pmb { \alpha } } ^ { ( z ) } ; \mathcal { D } ^ { ( z ) } \big ) = \sum _ { i = 1 } ^ { K } \frac { 1 } { n _ { i } ^ { ( z ) } } \sum _ { ( S , A , R ) \in \mathcal { D } _ { i } ^ { ( z ) } } \hat { w } _ { i } ^ { ( z ) } \big ( S , A , R \big ) \frac { \pi _ { e } \big ( A \mid S \big ) } { \pi _ { i } \big ( A \mid S \big ) } .
$$

Since $\hat { \mathbf { w } } ^ { ( z ) }$ satisfies the normalization and dominance conditions conditional on $\hat { \pmb { \alpha } } ^ { ( z ) }$ , Theorem 3.1 implies that $\hat { J } ( \pi _ { e } , \hat { \pmb { \alpha } } ^ { ( z ) } ; \mathcal { D } ^ { ( z ) } )$ is unbiased conditional on $\hat { \pmb { \alpha } } ^ { ( z ) }$ , equivalently conditional on $\mathcal { D } ^ { ( - z ) }$

Asymptotic Normality. Next, we show that

$$
\sqrt { n ^ { \prime } } \left( \hat { J } _ { \mathrm { o I P S } } ( \pi _ { e } ; \mathcal { D } ) - J ( \pi _ { e } ) \right) \overset { d } { \to } \mathcal { N } \left( 0 , n \mathrm { V a r } [ \hat { J } ( \pi _ { e } , \mathbf { w } ^ { \star } ; \mathcal { D } ) ] \right) .
$$

We work with known logging policies under Assumption 1: K and $Z$ are fixed, the logger-wise folds are balanced up to rounding, the fitted scores converge in probability to the oracle score in each $L ^ { 2 } ( \mu _ { \pi _ { i } } )$ , and all fitted and oracle scores have finite second moments, including over training randomness.

We rewrite our infeasible optimal estimator $\hat { J } ( \pi _ { e } , \alpha ^ { \star } ; \mathcal { D } )$ in (3.4) in the form considered by Kallus et al. [2021]:

$$
\hat { J } ( \pi _ { e } , \alpha ^ { \star } ; \mathcal { D } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { K } \sum _ { ( S , A , R ) \in \mathcal { D } _ { i } } \left[ \frac { \pi _ { e } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ( A \mid S ) } \left( R - \frac { g _ { \alpha ^ { \star } } } { \pi _ { e } } ( S , A ) \right) + \frac { g _ { \alpha ^ { \star } } } { \pi _ { e } } ( S , \pi _ { e } ) \right] .
$$

The normal equation gives the oracle summand mean ${ \cal J } ( \pi _ { e } )$ under every logger. Grouping the oracle observations into m independent blocks, each containing $n _ { i }$ observations from logger i, the i.i.d. central limit theorem gives the displayed limit.

Conditional independence and the assumed $L ^ { 2 }$ score convergence make each fold’s contribution to the centered error relative to the oracle, scaled by $\sqrt { n ^ { \prime } } , o _ { p } ( 1 )$ by conditional Chebyshev. With known policies, the conditional bias is zero. Since $Z$ is fixed and fold weights are sample proportions, summing the contributions and applying Slutsky’s theorem gives the desired asymptotic normality result. If the logging policies are instead estimated by normalized policies fitted on the training folds, with positive mixtures, the same argument applies whenever each fold mean has conditional bias $o _ { p } \big ( n ^ { \prime - 1 / 2 } \big )$ <sub>; this is the condition in Remark 2, and it is not implied by</sub> √<sub>n-consistency of the</sub> estimated propensities. The same argument applies to DR scores under the corresponding moment, score-convergence, and bias conditions. □

## E.4 Additional Proofs

Lemma E.2 (Normal-Equation Interpretation of $\mathbf T \pmb { \alpha } = \mathbf c )$ . For T and c defined in (3.3), their entries can be written as

$$
T _ { i j } = \big \langle \pi _ { i } , \pi _ { j } \big \rangle _ { \mu _ { \mathrm { l o g } } } , \quad c _ { i } = \big \langle q \pi _ { e } , \pi _ { i } \big \rangle _ { \mu _ { \mathrm { l o g } } } ,
$$

where $\mu _ { \mathrm { l o g } }$ is given by

$$
\frac { d \mu _ { \mathrm { l o g } } } { d \mu _ { s , a } } ( s , a ) : = \frac { p _ { s } ( s ) } { \pi _ { \mathrm { a v g } } ( s , a ) } ,
$$

with $\mu _ { s , a }$ denoting the base measure over ${ \mathcal { S } } \times { \mathcal { A } }$

Consequently, $\mathbf { T } \pmb { \alpha } = \mathbf { c }$ is the normal equation for the orthogonal projection of $q \pi _ { e }$ onto $\nu : =$ span $( { \bar { \{ } }  \pi _ { i } \} _ { i \in [ K ] } )$ . In particular, Tα = c admits a solution, and any solution $\alpha ^ { \star }$ defines the orthogonal projection of $q \pi _ { e }$ onto V through $\textstyle \sum _ { i \in [ K ] } { \alpha _ { i } ^ { \star } \pi _ { i } }$

ProofofLemma E.2. Recall the definitions of T and c in (3.3):

$$
T _ { i j } : = \mathbb { E } _ { \mu _ { \pi _ { i } } } \left[ \frac { \pi _ { j } ( \boldsymbol { A } \mid \boldsymbol { S } ) } { \pi _ { \mathrm { a v g } } ( \boldsymbol { A } \mid \boldsymbol { S } ) } \right] , c _ { i } : = \mathbb { E } _ { \mu _ { \pi _ { i } } } \left[ R \frac { \pi _ { e } ( \boldsymbol { A } \mid \boldsymbol { S } ) } { \pi _ { \mathrm { a v g } } ( \boldsymbol { A } \mid \boldsymbol { S } ) } \right] .
$$

The first identity follows by direct computation:

$$
\begin{array} { l } { { T _ { i j } = \mathbb E _ { \mu _ { \pi _ { i } } } \left[ \frac { \pi _ { j } ( A \mid S ) } { \pi _ { \mathrm { s g } } ( A \mid S ) } \right] } } \\ { { \ } } \\ { { \ } } \\ { { \displaystyle \quad = \int _ { \Omega } \frac { \pi _ { i } ( a | s ) \pi _ { j } ( a | s ) } { \pi _ { \mathrm { a v g } } ( a | s ) } p _ { r } ( r | s , a ) p _ { s } ( s ) d \mu ( \omega ) } } \\ { { \ } } \\ { { \displaystyle \quad = \int _ { S \times A } \pi _ { i } ( a | s ) \pi _ { j } ( a | s ) \frac { p _ { s } ( s ) } { \pi _ { \mathrm { a v g } } ( a | s ) } d \mu _ { s , a } ( s , a ) } } \\ { { \ } } \\ { { \displaystyle = \left. \pi _ { i } , \pi _ { j } \right. _ { \mu _ { \mathrm { l o g } } } . } } \end{array}
$$

Similarly, for $c _ { i }$ , we have

$$
\begin{array} { r l } & { { { { \kappa } _ { i } } } = \mathbb { E } _ { \mu _ { \nu _ { i } } } [ \ K \frac { \pi _ { \epsilon } ( A ) \ ( S ) } { \pi _ { \epsilon \kappa \alpha } ( A ) \ \sqrt { S } } ] } \\ & { \quad = \mathbb { E } _ { \mu _ { \nu _ { i } \times \epsilon } } [ \mathbb { E } _ { \mu _ { \nu \alpha } } [ \ K \frac { \pi _ { \epsilon } ( A ) \ ( S ) } { \pi _ { \epsilon \kappa \alpha } ( A ) \ \sqrt { S } } \ ] S , A ] ] } \\ & { \quad = \mathbb { E } _ { \mu _ { \nu _ { i } } } [ \mathbb { E } _ { \mu _ { \kappa \alpha } } [  \nabla R | \ \ S , A ] \frac { \pi _ { \epsilon } ( A ) \ \ \sqrt { S } } { \pi _ { \epsilon \kappa \alpha } ( A ) \ \sqrt { S } } ] } \\ & { \quad = \mathbb { E } _ { \mu _ { \nu _ { i } } } [ (  \nabla , A ) \frac { \pi _ { \epsilon } ( A ) \ ( S ) } { \pi _ { \epsilon \kappa \alpha } ( A ) \sqrt { S } } ] } \\ & { \quad = \int _ { \Omega } ( | \omega , \omega | \frac { \pi _ { \epsilon } ( A ) \ \cdot | \mathbf { v } | \cdot | \mathbf { v } | \cdot | \mathbf { v } | \cdot | \mathbf { v } | \cdot | \mathbf { v } | \cdot | \mathbf { \partial } | | \rho _ { \nu \kappa } ( \cdot | \mathbf { v } ) \cdot | \partial | \rho _ { \nu \kappa } ( \cdot | ) } { \pi _ { \epsilon \kappa \alpha } ( \omega ) \sqrt { S } } d \mu ( \omega ) \ } \\ &  \quad = \int _ { \partial \times \mathcal { A } } (  \partial \mathbf { s } ) \pi _ { \epsilon \kappa } ( \omega | S ) \pi _ { \epsilon } ( a | \mathcal { S } \ \end{array}
$$

It remains to show that ${ \bf T } \alpha = { \bf c }$ is the normal equation for the orthogonal projection. Let

$$
\gamma : = \mathrm { s p a n } \big ( \{ \pi _ { i } \} _ { i \in [ K ] } \big ) \subset L ^ { 2 } ( \mu _ { \log } ) .
$$

Since V is finite-dimensional, it is closed. Hence, the orthogonal projection of $q \pi _ { e }$ onto V exists. Write this projection as

$$
g _ { \alpha ^ { \star } } : = \sum _ { k \in [ K ] } \alpha _ { k } ^ { \star } \pi _ { k } .
$$

By the characterization of orthogonal projections, $g _ { \alpha ^ { \star } }$ satisfies

$$
\left. q \pi _ { e } - g _ { \alpha ^ { \star } } , \pi _ { i } \right. _ { \mu _ { \log } } = 0 , \quad \forall i \in [ K ] .
$$

Expanding this condition gives

$$
\big \langle q \pi _ { e } , \pi _ { i } \big \rangle _ { \mu _ { \mathrm { l o g } } } = \sum _ { k \in [ K ] } \alpha _ { k } ^ { \star } \big \langle \pi _ { k } , \pi _ { i } \big \rangle _ { \mu _ { \mathrm { l o g } } } , \quad \forall i \in [ K ] .
$$

By the identities proved above, this system is exactly $\mathbf { T } \alpha ^ { \star } = \mathbf { c }$ . Therefore, $\mathbf { T } \pmb { \alpha } = \mathbf { c }$ admits a solution, and any solution $\alpha ^ { \star }$ defines the orthogonal projection of $q \pi _ { e }$ onto V through $\begin{array} { r l } { g _ { \alpha ^ { \star } } } & { { } = } \end{array}$ $\textstyle \sum _ { i \in [ K ] } { \alpha _ { i } ^ { \star } \pi _ { i } }$ □

Lemma E.3 (Variance Gap). Let f be measurable and such that the summands of the estimator below havefinite second moments under each $\mu _ { \pi _ { i } }$ , and let $\hat { J } _ { \Gamma } ( \pi _ { e } , f ; \mathcal { D } )$ be the estimator defined by

$$
\hat { J } _ { \Gamma } ( \pi _ { e } , f ; \mathcal { D } ) : = \frac { 1 } { n } \sum _ { i = 1 } ^ { K } \sum _ { ( S , A , R ) \in \mathcal { D } _ { i } } \left[ \frac { \pi _ { e } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ( A \mid S ) } \big ( R - f ( S , A ) \big ) + f ( S , \pi _ { e } ) \right] ,\tag{E.5}
$$

where $\begin{array} { r } { f ( s , \pi _ { e } ) : = \int _ { \cal A } f ( s , a ) \pi _ { e } ( a | s ) d \mu _ { a } ( a ) . } \end{array}$

Let $q ( s , a ) : = \mathbb { E } [ R | S = s , A = a ]$ denote the conditional mean reward, and define $m _ { i } ( f ) : =$ $\mathbb { E } _ { \mu _ { \pi _ { i } } } [ ( q - f ) \pi _ { e } / \pi _ { \mathrm { a v g } } ] - \mathbb { E } _ { p _ { s } } [ ( q - f ) ( S , \pi _ { e } ) ]$ . Then, the variance gap between $\hat { J } _ { \Gamma } ( \pi _ { e } , f ; \mathcal { D } )$ and $\hat { J } _ { \Gamma } ^ { \star } ( \pi _ { e } ; \mathcal { D } ) : = \hat { J } _ { \Gamma } ( \pi _ { e } , q ; \mathcal { D } )$ satisfies

$$
\begin{array} { r l } & { \mathrm { V a r } \big [ \hat { J } _ { \Gamma } ( \pi _ { e } , f ; \mathcal { D } ) \big ] - \mathrm { V a r } \big [ \hat { J } _ { \Gamma } ^ { \star } ( \pi _ { e } ; \mathcal { D } ) \big ] } \\ & { \qquad = \displaystyle \frac { 1 } { n } \left( \mathrm { V a r } _ { \mu _ { \mathrm { r a v g } } } \left[ \frac { \pi _ { e } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ( A \mid S ) } ( q ( S , A ) - f ( S , A ) ) \right] - \mathrm { V a r } _ { p _ { s } } \big [ q ( S , \pi _ { e } ) - f ( S , \pi _ { e } ) \big ] \right) } \\ & { \qquad - \displaystyle \frac { 1 } { n } \sum _ { i } \frac { n _ { i } } { n } m _ { i } ( f ) ^ { 2 } \geq 0 . } \end{array}
$$

ProofofLemma E.3. Let $Q _ { f } = \pi _ { e } ( R - f ) / \pi _ { \mathrm { a v g } } + f ( S , \pi _ { e } )$ and $V ( f ) = \operatorname { V a r } _ { \mu _ { \pi _ { \mathrm { a v g } } } } \left[ Q _ { f } \right]$ . Independence across samples and the variance decomposition over loggers give

$$
\mathrm { V a r } [ \hat { J } _ { \Gamma } ( \pi _ { e } , f ; \mathcal { D } ) ] = \frac { 1 } { n ^ { 2 } } \sum _ { i } n _ { i } \mathrm { V a r } _ { \mu _ { \pi _ { i } } } [ Q _ { f } ] = \frac { 1 } { n } \left( V ( f ) - \sum _ { i } \frac { n _ { i } } { n } m _ { i } ( f ) ^ { 2 } \right) .
$$

Here $m _ { i } ( f ) = \mathbb { E } _ { \mu _ { \pi _ { i } } } [ Q _ { f } ] - J ( \pi _ { e } )$ , which agrees with the definition in the lemma, the second equality uses $\mathbb { E } _ { \mu _ { \pi _ { \mathrm { a v g } } } } [ Q _ { f } ] = J ( \pi _ { e } )$ , and $m _ { i } ( q ) = 0$ . For the remainder of the proof, all variances, covariances, and expectations are taken with respect to $\mu _ { \pi _ { \mathrm { a v g } } } ,$ , so we omit the corresponding subscript.

Using Var $[ X ] - \operatorname { V a r } [ Y ] = \operatorname { V a r } [ X - Y ] + 2 \mathrm { C o v } [ X - Y , Y ] ,$ , we obtain

$$
\begin{array} { r l } & { \frac { V ( f ) - V ( q ) } { n } } \\ & { = \frac { 1 } { n } \mathrm { V a r } \Bigg [ \frac { \pi _ { e } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ( A \mid S ) } \big ( q ( S , A ) - f ( S , A ) \big ) - \big ( q ( S , \pi _ { e } ) - f ( S , \pi _ { e } ) \big ) \Bigg ] } \\ & { \quad + \frac { 2 } { n } \mathrm { C o v } \Bigg [ \frac { \pi _ { e } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ( A \mid S ) } \big ( q ( S , A ) - f ( S , A ) \big ) - \big ( q ( S , \pi _ { e } ) - f ( S , \pi _ { e } ) \big ) , \Bigg ] } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \pi _ { e } ( A \mid S ) } \\ & { \quad \quad \quad \quad \quad \pi _ { \mathrm { a v g } } ( A \mid S ) \big ( R - q ( S , A ) \big ) + q ( S , \pi _ { e } ) } \end{array}
$$

We now show that the covariance term is zero. It decomposes into

$$
\begin{array} { r l } & { \mathrm { C o v } \left[ \frac { \pi _ { e } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ( A \mid S ) } \big ( q ( S , A ) - f ( S , A ) \big ) - \big ( q ( S , \pi _ { e } ) - f ( S , \pi _ { e } ) \big ) , \frac { \pi _ { e } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ( A \mid S ) } \big ( R - q ( S , A ) \big ) \right] } \\ & { \phantom { \frac { 1 } { 1 } } + \mathrm { C o v } \left[ \frac { \pi _ { e } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ( A \mid S ) } \big ( q ( S , A ) - f ( S , A ) \big ) - \big ( q ( S , \pi _ { e } ) - f ( S , \pi _ { e } ) \big ) , q ( S , \pi _ { e } ) \right] . } \end{array}
$$

The first covariance vanishes because the first argument is measurable with respect to $( S , A )$ , while

$$
\operatorname { \mathbb { E } } [ R - q ( S , A ) \mid S , A ] = 0 .
$$

Thus, by iterated expectation, its covariance with the centered residual term is zero.

For the second covariance, note that $q ( S , \pi _ { e } )$ depends only on $S .$ . Moreover, conditional on $S ,$

$$
\mathbb { E } \bigg [ \frac { \pi _ { e } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ( A \mid S ) } \big ( q ( S , A ) - f ( S , A ) \big ) \bigg | S \bigg ] = q ( S , \pi _ { e } ) - f ( S , \pi _ { e } ) .
$$

Hence,

$$
\begin{array} { r } { \mathbb { E } \bigg [ \frac { \pi _ { e } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ( A \mid S ) } \big ( q ( S , A ) - f ( S , A ) \big ) - \big ( q ( S , \pi _ { e } ) - f ( S , \pi _ { e } ) \big ) \bigg | S \bigg ] = 0 . } \end{array}
$$

Since the second argument $q ( S , \pi _ { e } )$ is a function only of $S ,$ the second covariance also vanishes. Therefore, the covariance term is zero, and the difference of mixture variances reduces to

$$
\begin{array} { r l } & { \frac { V ( f ) - V ( q ) } { n } } \\ & { = \frac { 1 } { n } \mathrm { V a r } \Big [ \frac { \pi _ { e } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ( A \mid S ) } \big ( q ( S , A ) - f ( S , A ) \big ) - \big ( q ( S , \pi _ { e } ) - f ( S , \pi _ { e } ) \big ) \Big ] \geq 0 . } \end{array}
$$

We proceed further to put the above expression into the claimed form. Define

$$
X : = { \frac { \pi _ { e } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ( A \mid S ) } } { \big ( } q ( S , A ) - f ( S , A ) { \big ) } , \qquad Y : = q ( S , \pi _ { e } ) - f ( S , \pi _ { e } ) .
$$

Under $\mu _ { \pi _ { \mathrm { a v g } } }$ , the conditional distribution of A given $S$ is $\pi _ { \mathrm { a v g } } ( \cdot \mid S )$ . Hence,

$$
\begin{array} { l } { \displaystyle \mathbb { E } \left[ X \mid S \right] = \int _ { \mathcal { A } } \frac { \pi _ { e } ( a \mid S ) } { \pi _ { \mathrm { a v g } } ( a \mid S ) } \big ( q ( S , a ) - f ( S , a ) \big ) \pi _ { \mathrm { a v g } } ( a \mid S ) d \mu _ { a } ( a ) } \\ { \displaystyle = \int _ { \mathcal { A } } \pi _ { e } ( a \mid S ) \big ( q ( S , a ) - f ( S , a ) \big ) d \mu _ { a } ( a ) } \\ { \displaystyle = q ( S , \pi _ { e } ) - f ( S , \pi _ { e } ) = Y . } \end{array}
$$

Therefore, the last display can be written as

$$
{ \frac { V ( f ) - V ( q ) } { n } } = { \frac { 1 } { n } } \mathrm { V a r } [ X - \mathbb { E } [ X \mid S ] ] .
$$

By the law of total variance,

$$
\operatorname { V a r } \left[ X \right] = \mathbb { E } \left[ \operatorname { V a r } \left[ X \mid S \right] \right] + \operatorname { V a r } \left[ \mathbb { E } \left[ X \mid S \right] \right] .
$$

Since $\mathbb { E } [ X \mid S ] = Y$ , this gives

$$
\operatorname { V a r } \left[ X - \mathbb { E } [ X \mid S ] \right] = \operatorname { V a r } \left[ X \right] - \operatorname { V a r } \left[ Y \right] \geq 0 .
$$

Substituting the definitions of X and $Y ,$ , and noting that the marginal distribution of $S$ under $\mu _ { \pi _ { \mathrm { a v g } } }$ is $p _ { s }$ , we obtain

$$
\begin{array} { r l } & { \frac { V ( f ) - V ( q ) } { n } } \\ & { = \frac { 1 } { n } \left( \mathrm { V a r } \left[ \frac { \pi _ { e } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ( A \mid S ) } \big ( q ( S , A ) - f ( S , A ) \big ) \right] - \mathrm { V a r } _ { p _ { s } } \big [ q ( S , \pi _ { e } ) - f ( S , \pi _ { e } ) \big ] \right) \geq 0 . } \end{array}
$$

Subtracting $\begin{array} { r } { n ^ { - 1 } \sum _ { i } ( n _ { i } / n ) m _ { i } ( f ) ^ { 2 } } \end{array}$ proves the claimed identity. Nonnegativity follows since the result equals $\begin{array} { r } { n ^ { - 1 } \sum _ { i } ( n _ { i } / n ) \mathrm { V a r } _ { \mu _ { \pi _ { i } } } [ X - Y ] } \end{array}$ . Here we use $\begin{array} { r } { \sum _ { i } ( n _ { i } / n ) \mathrm { C o v } _ { \mu _ { \pi _ { i } } } [ X - Y , Q _ { q } ] = \mathrm { C o v } _ { \mu _ { \pi _ { \mathrm { a v g } } } } [ X - } \end{array}$ $Y , Q _ { q } ] = 0$ , where the first equality holds because $\mathbb { E } _ { \mu _ { \pi _ { i } } } [ Q _ { q } ] = J ( \pi _ { e } )$ for every i. □

Theorem 4.3. The variance gap between $\hat { J } ( \pi _ { e } , \alpha ^ { \star } ; \mathcal { D } )$ and the semiparametrically efficient oracle estimator $\hat { J } _ { \Gamma } ^ { \star } ( \pi _ { e } ; \mathcal { D } ) : = \hat { J } _ { \Gamma } ( \pi _ { e } , 1 / \pi _ { \mathrm { a v g } } , q ; \mathcal { D } )$ is given by

$$
\mathrm { V a r } \big [ \hat { J } ( \pi _ { e } , \alpha ^ { \star } ; \mathcal { D } ) \big ] - \mathrm { V a r } \big [ \hat { J } _ { \Gamma } ^ { \star } ( \pi _ { e } ; \mathcal { D } ) \big ] = \frac { 1 } { n } \left( \operatorname* { i n f } _ { \alpha \in \mathbb { R } ^ { K } } \big \| q \pi _ { e } - g _ { \alpha } \big \| _ { \mu _ { \mathrm { l o g } } } ^ { 2 } - \mathrm { V a r } _ { p _ { s } } \big [ q ( S , \pi _ { e } ) \big ] \right) \geq 0 ,
$$

where $\| \cdot \| _ { \mu _ { \log } }$ is the norm induced by the inner product $\langle \cdot , \cdot \rangle _ { \mu _ { \mathrm { l o g } } }$ , with $d \mu _ { \mathrm { l o g } } : = p _ { s } / \pi _ { \mathrm { a v g } } d \mu _ { s , a } .$ Moreover, equality holds whenever $q \pi _ { e } \in \mathrm { s p a n } \bigl ( \{ \pi _ { i } \} _ { i \in [ K ] } \bigr )$

Proof of Theorem 4.3. First, rewrite $\hat { J } ( \pi _ { e } , \alpha ^ { \star } ; \mathcal { D } )$ in (3.4) as

$$
\hat { J } ( \pi _ { e } , \alpha ^ { \star } ; \mathcal { D } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { K } \sum _ { ( S , A , R ) \in \mathcal { D } _ { i } } \left[ \frac { \pi _ { e } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ( A \mid S ) } \left( R - f _ { \alpha ^ { \star } } ( S , A ) \right) + f _ { \alpha ^ { \star } } ( S , \pi _ { e } ) \right] ,
$$

where $\begin{array} { r } { f _ { \alpha ^ { \star } } : = g _ { \alpha ^ { \star } } / \pi _ { e } = \sum _ { i = 1 } ^ { K } \alpha _ { i } ^ { \star } \pi _ { i } / \pi _ { e } } \end{array}$ . Hence, $\hat { J } ( \pi _ { e } , \pmb { \alpha } ^ { \star } ; \mathcal { D } ) = \hat { J } _ { \Gamma } ( \pi _ { e } , f _ { \alpha ^ { \star } } ; \mathcal { D } )$

The normal equation gives $\mathbb { E } _ { \mu _ { \pi _ { i } } } [ \pi _ { e } ( q - f _ { \alpha ^ { \star } } ) / \pi _ { \mathrm { a v g } } ] = 0$ for every i. Also, weighting the i-th normal equation by $n _ { i } / n$ and using $\begin{array} { r } { \sum _ { i } \frac { n _ { i } } { n } T _ { i j } = 1 } \end{array}$ and $\begin{array} { r } { \sum _ { i } \frac { n _ { i } } { n } c _ { i } = J ( \pi _ { e } ) \operatorname { g i v e s } \sum _ { j } \alpha _ { j } ^ { \star } = J ( \pi _ { e } ) } \end{array}$ , so $m _ { i } ( f _ { \alpha ^ { \star } } ) = 0$ . Applying Lemma $\mathrm { E } . 3$ therefore gives

$$
\begin{array} { l } { { \mathrm { V a r } \big [ \hat { J } ( \pi _ { e } , \alpha ^ { \star } ; \mathcal { D } ) \big ] - \mathrm { V a r } \big [ \hat { J } _ { \Gamma } ^ { \star } ( \pi _ { e } ; \mathcal { D } ) \big ] } \ ~ } \\ { { \displaystyle ~ \quad = \frac { 1 } { n } \left( \mathrm { V a r } _ { \mu _ { \pi _ { \mathrm { a v g } } } } \left[ \frac { \pi _ { e } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ( A \mid S ) } \big ( q ( S , A ) - f _ { \alpha ^ { \star } } ( S , A ) \big ) \right] - \mathrm { V a r } _ { p _ { s } } \big [ q ( S , \pi _ { e } ) - f _ { \alpha ^ { \star } } ( S , \pi _ { e } ) \big ] \right) } \ ~ } \\ { { \displaystyle ~ \quad = \frac { 1 } { n } \left( \mathrm { V a r } _ { \mu _ { \pi _ { \mathrm { a v g } } } } \left[ \frac { \pi _ { e } ( A \mid S ) } { \pi _ { \mathrm { a v g } } ( A \mid S ) } \big ( q ( S , A ) - f _ { \alpha ^ { \star } } ( S , A ) \big ) \right] - \mathrm { V a r } _ { p _ { s } } \big [ q ( S , \pi _ { e } ) \big ] \right) . } } \end{array}
$$

The last equality follows because

$$
f _ { \alpha ^ { \star } } ( S , \pi _ { e } ) = \int _ { \varLambda } \sum _ { i = 1 } ^ { K } \alpha _ { i } ^ { \star } \pi _ { i } ( a | s ) d \mu _ { a } ( a ) = \sum _ { i = 1 } ^ { K } \alpha _ { i } ^ { \star } ,
$$

which is deterministic and therefore does not contribute to the variance.

It remains to rewrite the first variance term in terms of the norm $\| \cdot \| _ { \mu _ { \log } }$ . First, observe that the term inside $\mathrm { V a r } _ { \mu _ { \pi _ { \mathrm { a v g } } } }$ has mean zero:

$$
\begin{array} { r l } & { \mathbb { E } _ { \mu _ { \pi _ { \mathrm { s y g } } } } \left[ \frac { \pi _ { e } \left( A \mid S \right) } { \pi _ { \mathrm { a v g } } \left( A \mid S \right) } \big ( q ( S , A ) - f _ { \alpha ^ { \star } } ( S , A ) \big ) \right] = \mathbb { E } _ { \mu _ { \tau _ { e } } } \big [ q ( S , A ) - f _ { \alpha ^ { \star } } ( S , A ) \big ] } \\ & { \quad \quad \quad \quad = \mathbb { E } _ { \mu _ { \tau _ { e } } } \big [ q ( S , A ) \big ] - \mathbb { E } _ { \mu _ { \tau _ { e } } } \left[ \displaystyle \sum _ { i = 1 } ^ { K } \alpha _ { i } ^ { \star } \frac { \pi _ { i } \left( A | S \right) } { \pi _ { e } \left( A | S \right) } \right] } \\ & { \quad \quad \quad = J ( \pi _ { e } ) - \displaystyle \sum _ { i = 1 } ^ { K } \alpha _ { i } ^ { \star } = 0 . } \end{array}
$$

Here, the last equality follows from the definition $\alpha _ { i } ^ { \star } = \left. w _ { i } , 1 \right. _ { \mu _ { \pi _ { e } } }$ in the proof of Theorem 4.1 and the normalization condition $\begin{array} { r } { \sum _ { i \in [ K ] } w _ { i } ( s , a , r ) = r , } \end{array}$ which imply

$$
\sum _ { i \in [ K ] } { \alpha _ { i } ^ { \star } } = \Big \langle \sum _ { i \in [ K ] } { w _ { i } } , 1 \Big \rangle _ { \mu _ { \pi _ { e } } } = \mathbb { E } _ { \mu _ { \pi _ { e } } } \left[ R \right] = J ( \pi _ { e } ) .
$$

Therefore,

$$
\begin{array} { r l } { \operatorname { V a r } _ { \mu _ { \mathtt { r a v g } } } \left[ \frac { \pi _ { e } ( A \mid S ) } { \pi _ { \mathtt { a v g } } ( A \mid S ) } \big ( q ( S , A ) - f _ { \alpha ^ { \star } } ( S , A ) \big ) \right] = \mathbb { E } _ { \mu _ { \mathtt { r a v g } } } \left[ \frac { \pi _ { e } ^ { 2 } ( A \mid S ) } { \pi _ { \mathtt { a v g } } ^ { 2 } ( A \mid S ) } \big ( q ( S , A ) - f _ { \alpha ^ { \star } } ( S , A ) \big ) ^ { 2 } \right] } & { } \\ { = \mathbb { E } _ { \mu _ { \mathtt { r a v g } } } \left[ \frac { \big ( q ( S , A ) \pi _ { e } ( A \mid S ) - g _ { \alpha ^ { \star } } ( S , A ) \big ) ^ { 2 } } { \pi _ { \mathtt { a v g } } ^ { 2 } ( A \mid S ) } \right] } & { } \\ { = \left\| q \pi _ { e } - g _ { \alpha ^ { \star } } \right\| _ { \mu _ { \mathtt { l o g } } } ^ { 2 } } & { } \\ { = \displaystyle \operatorname* { i n f } _ { \alpha \in \mathbb { R } ^ { K } } \left\| q \pi _ { e } - g _ { \alpha } \right\| _ { \mu _ { \mathtt { l o g } } } ^ { 2 } , } \end{array}
$$

where the last equality follows from the fact that $g _ { \alpha ^ { \star } }$ is the orthogonal projection of $q \pi _ { e }$ with respect to the norm $\| \cdot \| _ { \mu _ { \log } } .$ . This proves the theorem. □

## F Computational Complexity and Scalability

In this section, we analyze the computational complexity of $\hat { J } _ { \mathrm { o I P S } }$ , especially when the number of loggers is large $( K \approx 1 0 0 )$ ). Although restricted-optimal coefficients require solving a linear system with technique matrix $\mathbf { T } \in \mathbb { R } ^ { K \times K }$ , we show both theoretically and empirically that the added cost is modest for practical K. In this appendix, N denotes the total sample size, written n in the main text.

## F.1 Theoretical Complexity Analysis

Computing $\hat { J } _ { \mathrm { o I P S } }$ within cross-fitting has two main steps: (i) estimating the technique matrix T and contribution vector c, and (ii) solving for ${ \pmb { \alpha } } = { \bf T } ^ { - 1 } { \bf c }$ (or its pseudoinverse analogue).

(i) Matrix construction: Constructing T aggregates propensity terms over N samples and K loggers, yielding $O ( N K )$ complexity. This matches the dominant complexity class of balanced IPS $( \hat { J } _ { \mathrm { b I P S } } )$ , which also computes aggregate propensity statistics $\begin{array} { r } { ( { \bf e . g . } , \sum _ { k } n _ { k } \pi _ { k } ) } \end{array}$ per sample.

(ii) Linear solve / pseudoinversion: Solving Tα = c has worst-case complexity $O ( K ^ { 3 } )$ with standard dense linear algebra (e.g., LU or SVD).

The total time is $O ( Z ( N K + K ^ { 3 } ) )$ , which is $O ( N K + K ^ { 3 } )$ for fixed Z. Accumulating the per-fold moments in one pass uses $O ( Z K ^ { 2 } )$ additional memory, while storing all cross-logger propensities uses $O ( N K )$ . These counts exclude fitting nuisance models such as propensity models. We compute α with the SVD-based Moore–Penrose pseudoinverse, which also handles a rank-deficient T; this occurs in the runtime study below, where all loggers are mixtures of the deterministic and uniform policies.

## F.2 Empirical Runtime Analysis

To validate practicality, we measured runtime on the Letter dataset $( N = 2 0 , 0 0 0 , 2 6$ actions) with $K = 1 0$ and $K = 1 0 0$ loggers. Experiments ran on an Apple MacBook Air M1 (base model, 8GB RAM). Logging policies were generated by mixing the deterministic policy with a uniform policy. We compared Naive IPS, Balanced IPS, Weighted IPS [Agarwal et al., 2017], and our Optimal IPS. Methods requiring sample-dependent weights (Weighted IPS and Optimal IPS) used $\bar { Z } = 5$ cross-fitting folds.

Table 3 reports mean inference time per trial (milliseconds) over 200 independent runs. Each ± value is the half-width of a 95% confidence interval based on the normal approximation.

Table 3: Comparison of Relative RMSE and wall-clock Time (in milliseconds) on the Letter dataset with $N = 2 0 { , } 0 0 0$ . We compare settings with $K = 1 0$ and $K = 1 0 0$ logging policies. Our method achieves the lowest error while maintaining a runtime comparable to the existing Weighted IPS baseline.
<table><tr><td>Estimator</td><td>K = 10 Loggers Rel-RMSE</td><td>Time (ms)</td><td>K = 100 Loggers Rel-RMSE</td><td>Time (ms)</td></tr><tr><td>Naive IPS  $( \hat { J } _ { \mathrm { I P S } } )$ </td><td>0.0113</td><td>0.16 ± 0.02</td><td>0.0104</td><td> $0 . 1 8 \pm 0 . 0 2$ </td></tr><tr><td>Balanced IPS  $( \hat { J } _ { \mathrm { b I P S } } )$ </td><td>0.0133</td><td>0.13 ± 0.02</td><td>0.0087</td><td> $0 . 5 1 \pm 0 . 0 1$ </td></tr><tr><td>Weighted IPS  $( \hat { J } _ { \mathrm { w I P S } } )$ </td><td>0.0122</td><td>2.16 ± 0.03</td><td>0.0075</td><td> $1 5 . 4 9 \pm 0 . 4 3$ </td></tr><tr><td>Optimal IPS  $( \hat { J } _ { \mathrm { o I P S } } )$ </td><td>0.0045</td><td>2.52 ± 0.05</td><td>0.0045</td><td> $2 0 . 0 6 \pm 0 . 1 3$ </td></tr></table>

## Observations.

• Scalability: Even at $K = 1 0 0$ , Optimal IPS takes about 20ms per trial. This is slower than heuristic balanced IPS but still inexpensive for offline evaluation.

• Comparison with Weighted IPS: Runtime is comparable to Weighted IPS. This is expected because both methods use cross-fitting and dataset passes to estimate nuisance quantities (variance terms for Weighted IPS, technique matrix for Optimal IPS). The linear solve adds limited overhead relative to those passes.

• Efficiency–Compute Trade-off: For a small runtime increase (milliseconds), Optimal IPS reduces Relative-RMSE by about 40% versus Weighted IPS and about 48% versus Balanced IPS in the K = 100 setting.

Practical Takeaway. These numbers should be interpreted in the context of the full OPE pipeline. In our runs, end-to-end processing, including data loading, preprocessing, and fitting logger models, dominates total cost. Against that baseline, the additional time for $\hat { J } _ { \mathrm { o I P S } }$ remains a moderate fraction of wall-clock time rather than a bottleneck, while providing substantial accuracy gains.

## F.3 Large-N Synthetic Experiment

We use the synthetic data generator of the Open Bandit Pipeline (OBP, version 0.5.7) [Saito et al., 2021] with 10 actions, five-dimensional Gaussian contexts, and binary rewards from a logistic model with fixed coefficients. The evaluation policy chooses the action with the highest conditional mean reward. Each of the five known logging policies mixes the evaluation policy with the uniform policy, with uniform-exploration probabilities 0.05, 0.25, 0.50, 0.75, and 0.95, and all strata have the same size. For each $\dot { N } \in \{ 1 0 ^ { 4 } , 1 0 ^ { 5 } , 1 0 ^ { 6 } \}$ , we draw contexts, actions, and rewards with 20 independent random seeds. The ground-truth value is the evaluation policy’s conditional mean reward averaged over the sampled contexts. Weighted IPS and $\hat { J } _ { \mathrm { o I P S } }$ use five-fold cross-fitting. Confidence intervals are computed as in $\ S 5 ,$ with the 20 seeds as the resampled units.

Table 4: Relative-RMSE $( \times 1 0 ^ { - 3 } )$ in the large-N synthetic experiment. Parentheses show 95% bootstrap CIs.
<table><tr><td>Estimator</td><td> $N = 1 0 ^ { 4 }$ </td><td> $N = 1 0 ^ { 5 }$ </td><td> $N = 1 0 ^ { 6 }$ </td></tr><tr><td>IPS</td><td>13.0 (9.71, 16.2)</td><td>3.92 (2.88, 4.84)</td><td>1.40 (0.941, 1.79)</td></tr><tr><td>bIPS</td><td>6.14 (4.32, 7.94)</td><td>2.15 (1.53, 2.72)</td><td>0.729 (0.555, 0.904)</td></tr><tr><td>wIPS</td><td>3.83 (2.77, 4.82)</td><td>1.85 (1.23, 2.43)</td><td>0.544 (0.370, 0.718)</td></tr><tr><td>oIPS</td><td>2.70 (1.89, 3.43)</td><td>0.775 (0.570, 0.948)</td><td>0.199 (0.144, 0.251)</td></tr></table>

Table 5: Mean runtime of each estimator in milliseconds, and the largest peak resident memory over 20 seeds. Memory is measured for a single process that generates the data and runs all four estimators.
<table><tr><td> $N$ </td><td>IPS</td><td>bIPS</td><td>wIPS</td><td>oIPS</td><td>Peak memory (MiB)</td></tr><tr><td> $1 0 ^ { 4 }$ </td><td>0.178</td><td>0.161</td><td>4.626</td><td>6.000</td><td>124.652</td></tr><tr><td> $\mathrm { 1 0 ^ { 5 } }$ </td><td>0.994</td><td>1.220</td><td>15.372</td><td>25.435</td><td>164.668</td></tr><tr><td> ${ 1 0 } ^ { 6 }$ </td><td>18.053</td><td>23.462</td><td>162.890</td><td>346.201</td><td>584.516</td></tr></table>

We ran these measurements on an x86-64 Linux machine with eight allocated CPU cores and 12 GiB of RAM. Runtimes exclude data generation and the computation of propensities; peak memory includes the stored cross-logger propensities.

## G Related Work

OPE evaluates counterfactual policies from logged data [Precup et al., 2000, Bottou et al., 2013]. Although classical treatments focus on a single logger, many real applications aggregate data from multiple behavior policies, such as parallel A/B systems, leading to a stratified design in which logger identity defines strata [Agarwal et al., 2017, He et al., 2019, Kallus et al., 2021, Chen et al., 2020, Wooldridge, 2001].

OPE with Multiple Loggers. Agarwal et al. [2017] initiated the multi-logger OPE literature with MIS-inspired IPS estimators [Veach and Guibas, 1995, Cornuet et al., 2012, Elvira et al., 2019] and showed that no single estimator uniformly dominates across instances. Kallus et al. [2021] formalized this setting, derived the semiparametric efficiency bound, and proposed doubly robust estimators that asymptotically attain it. Related directions include off-policy learning with multiple loggers [He et al., 2019], infinite-horizon analysis [Chen et al., 2020], and generalization under selection bias [Hatt et al., 2022]. Liu et al. [2025] studied multi-policy evaluation in RL via behavior-policy design.

Multiple Importance Sampling. MIS was originally developed to combine multiple proposal distributions in Monte Carlo rendering [Veach and Guibas, 1995, Owen and Zhou, 2000, Elvira et al., 2019, Kondapaneni et al., 2019]. The balance and power heuristics [Veach and Guibas, 1995] provide robust performance under nonnegative weights, and later work studied broader weighting families [Elvira et al., 2019]. Kondapaneni et al. [2019] derived variance-optimal MIS weights that may be negative, yielding further variance reduction in some regimes. Our method adapts this variance-minimization perspective to multi-logger OPE.

Broader Context. Beyond the multi-logger setting, there is extensive work on variance control for IPS in standard OPE. Classical IPS traces back to Horvitz–Thompson [Horvitz and Thompson, 1952]. Subsequent work studied weight truncation [Ionides, 2008] and doubly robust estimation in contextual bandits [Dudík et al., 2014]. More recent lines include self-normalization and implicitexploration objectives [Swaminathan and Joachims, 2015, Kuzborskij et al., 2021], transformations yielding sub-Gaussian concentration [Metelli et al., 2021], and pessimistic regularization through PAC-Bayesian analysis [Aouali et al., 2024]. These methods are complementary to, rather than replacements for, our multi-logger focus.

## H Broader Impacts

This work is methodological research on off-policy evaluation with multiple logging policies. Its main positive impact is to make retrospective policy evaluation more reliable in domains where direct online testing may be costly or risky, such as recommender systems, healthcare decision support, and advertising. The main potential negative impacts arise from downstream use: more accurate OPE could still be used to justify the deployment of policies trained on biased, privacy-sensitive, or otherwise unrepresentative logs, and an incorrectly specified evaluation pipeline could give false confidence in a harmful policy. These risks are not unique to our estimator, but they motivate using it together with domain review, data-governance checks, privacy controls, fairness and subgroup analyses, and prospective validation before deployment in high-stakes settings.