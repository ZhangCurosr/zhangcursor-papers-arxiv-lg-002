# Residual-Based Causal Discovery via Pretrained In-Context Quantile Regression

Daniel Klippert <sup>1</sup> <sup>2</sup> Quang Nhat Huynh <sup>3</sup> Alexander Marx <sup>1</sup> <sup>2</sup>

<sup>§</sup> https://github.com/causality-lab/loscar

## Abstract

Residual-based causal discovery assumes that each variable is generated from its parents and an independent noise source. It estimates this noise and tests its independence from candidate causes. For identifiable functions such as additivenoise models (ANMs) and the broader class of location-scale noise models (LSNMs), this principle identifies the causal direction, as only the causal model admits this independence. Existing LSNM methods rely on parametric assumptions to obtain standardized residuals, risking model misspecification. In contrast, the RESIT-framework allows for nonparametric estimation of residuals to iteratively identify sink nodes, but was proposed for ANMs. To close this gap, we propose LOSCAR, an extension of RESIT to LSNMs that estimates the conditional mean and scale via quantile regression, enabling pretrained tabular foundation models as plug-in estimators. We compare it with a feasible generalized least squares (FGLS) variant, which is consistent given suitable basis functions. Across bivariate and multivariate causal-discovery benchmarks, in-context quantile regression substantially outperforms FGLS with flexible function classes, particularly for multivariate causes.

## 1. Introduction

Causal discovery provides a rigorous framework to determine under which conditions causal relations can be derived from observational data (Spirtes et al., 2000; Peters et al., 2017). Provided that all relevant sources of variation are observed, i.e., it can be assumed that confounding and selection bias are absent, and no cyclic relations exist, the causal graph can be described as a directed acyclic graph (DAG).

Constraint-based approaches rely on conditional independence testing to recover the Markov equivalence class (Spirtes et al., 2000), whereas methods restricting the function class of the structural causal model identify the unique DAG. Early approaches showed identifiability of the DAG under linear non-Gaussian additive models (Shimizu et al., 2006, LiNGAM). Both the linearity assumption (Hoyer et al., 2008) and the additivity assumption have been relaxed by subsequent works that prove identification results for post-nonlinear (Zhang & Hyvarinen¨ , 2009, PNL) and location-scale noise models (Immer et al., 2023; Strobl & Lasko, 2023, LSNMs).

Alongside this line of research, another promising direction involves amortized approaches to causal discovery (Lorch et al., 2022; Petersen et al., 2023), which treat causal discovery as a supervised machine learning problem by training on a large corpus of synthetically generated data. Benefiting from the advantages of transformer models (Thompson et al., 2026) or the success of tabular foundation models (Xu et al., 2026), these approaches are scalable and show strong empirical results. These works aim to encode known identification results through drawing the training data from such data generative models, however, they lack formal guarantees and require extensive pretraining.

Our Contributions Taking a different approach, we integrate pretrained tabular foundation models (Hollmann et al., 2023) as plug-in estimators. Our method, LOSCAR, extends the well-studied RESIT algorithm (Peters et al., 2014) to LSNMs. In contrast to previous work on LSNMs that rely on distributional assumptions on the noise variable (Immer et al., 2023; Duong & Nguyen, 2023; Tran et al., 2024; Klippert & Marx, 2026) or symmetric noise variables (Lin et al., 2024) we show that nonparametric estimation is feasible by estimating the conditional standard deviation via quantile regression. This result enables us to harness the predictive capabilities of pretrained tabular foundation models such as TabPFN (Hollmann et al., 2023) through in-context quantile regression without the need of retraining or fine-tuning the model. We propose two strategies through quantile regression and compare them to a variant using feasible GLS (Greene & Zhang, 2000) which given appropriate basis functions consistently estimates the conditional mean and standard deviation without distributional assumptions. Through empirical evaluation on 19 bivariate benchmarks we show that LOSCAR using in-context quantile regression achieves state-of-the-art performance and outperforms the FGLS variant on flexible function classes (Figure 1). Further, we evaluate LOSCAR on multivariate simulated and real-world benchmarks obtaining competitive results while being scalable in low and medium dimensional settings.

![](images/b71446ce79e51a69b632c23aee334eebe02834562a76a994ac367e1e6e45ccbd.jpg)  
Figure 1. Accuracy in % over 19 bivariate benchmarks. LOSCAR and LOSCAR-GL use TabPFN for in-context quantile regression, computing the conditional scale using different variants (Section 3.1). LOSCAR-FGLS (Section 3.2) employs spline-based FGLS. LOSCAR has an average accuracy of 90.04%. While LOSCAR-GL performs on par, the FGLS variant struggles to model complex location-scale relationships.

We first review LSNMs and their estimation through conditional mean and variance in Section 2, followed by a description of LOSCAR. In Section 3.1, we derive two approaches that enable in-context quantile regression and describe the FGLS variant in Section 3.2. We discuss related work in Section 4, present experiments in Section 5, and conclude thereafter.

## 2. Residual-Based Causal Discovery

Below we describe the framework for residual-based causal discovery. First we review LSNMs, which include ANMs as a special case, and their identification results. Then, we discuss the basis of bivariate causal discovery via residualbased independence testing and how these ideas extend to the multivariate setting.

## 2.1. Location-Scale Noise Models

When latent confounding, cycles, and selection are absent, we can model the causal network as a directed acyclic graph (DAG). A causal DAG model is a tuple $( { \mathcal { G } } , P _ { X } )$ , where defines a DAG and $P _ { X }$ a probability distribution associated with the nodes of ${ \mathcal { G } } ,$ , that has been generated through a structural causal model (SCM). In particular, an SCM ${ \mathcal { C } } =$ $( S , P _ { N } )$ is defined as a collection of assignments S:

$$
X ^ { ( j ) } : = f ^ { ( j ) } ( \mathrm { P A } _ { X ^ { ( j ) } } ^ { \mathcal { G } } , N _ { X ^ { ( j ) } } ) , \qquad j = 1 , \dotsc , d ,
$$

where all $N _ { X ^ { ( j ) } }$ (exogenous or noise variables) are jointly independent so that $P _ { N }$ is a product distribution. The DAG $\mathcal { G } = ( V , \mathcal { E } )$ of an SCM is obtained by creating a vertex $j \in V$ , with $| V | = d ,$ , for each $X ^ { ( j ) }$ and drawing directed edges $( i , j ) \in \mathcal { E }$ from each parent $i \in \mathrm { P A } _ { X ( j ) } ^ { \mathcal { G } }$ to j. Without restricting the function class, recovery of the DAG  is only feasible up to its Markov equivalence class (Peters et al., 2017), for example, through assuming faithfulness.

In this paper, we consider the class of location-scale noise models (Immer et al., 2023). That is, an SCM is called a location-scale noise model (LSNM) if, for each variable $X ^ { ( j ) }$ with non-empty parent set $\mathrm { P A } _ { X ^ { ( j ) } } ^ { \mathcal { G } } \neq \emptyset$ , the structural equation of $X ^ { ( j ) }$ is given by

$$
X ^ { ( j ) } : = f ^ { ( j ) } ( \mathrm { P A } _ { X ^ { ( j ) } } ^ { \mathcal { G } } ) + g ^ { ( j ) } ( \mathrm { P A } _ { X ^ { ( j ) } } ^ { \mathcal { G } } ) N _ { X ^ { ( j ) } } ,\tag{1}
$$

where $f ^ { ( j ) }$ and $g ^ { ( j ) }$ are deterministic real-valued functions with $g ^ { ( j ) } ( \cdot ) > 0$ . If $\mathrm { P A } _ { X ( j ) } ^ { \mathcal { G } } = \varnothing$ , then $X ^ { ( j ) } = N _ { X ^ { ( j ) } }$ LSNMs naturally extend the class of additive noise models (ANMs) (Kano et al., 2003; Hoyer et al., 2008) and reduce to an ANM when $g ^ { ( j ) } ( \cdot ) \equiv 1$

Identifiability of the causal graph under LNSMs has been studied by Immer et al. (2023); Strobl & Lasko (2023) for bivariate models. Strobl & Lasko (2023) extend these results to the multivariate setting. They conclude that identifiablitly holds up to pathological cases. Related results cover the exact identifiability conditions for Gaussian noise (Khemakhem et al., 2021), uniform and continuous Bernoulli (Sun & Schulte, 2023), and exponential families (Bodik & Chavez-Demoulin, 2025). Below, we discuss how these results can be exploited by practical estimators.

## 2.2. Cause-Effect Inference by Residual-Based Independence Testing

We start by describing the bivariate setting in which we call the two involved random variables X and Y. If the data has been generated according to an LSNM encoding the graph $X  Y$ , i.e., $Y : = f ( X ) + g ( X ) N _ { Y }$ , it holds that, by assumption, $( Y - f ( X ) ) / g ( X ) = N _ { Y }$ is independent of X. If the LSNM is identifiable, such an independence relation does not exist for the anti-causal direction, i.e., we cannot find functions h and k such that Y is independent of $N _ { X } =$ $( X - h ( Y ) ) / k ( Y )$ . This approach has its roots in the RESIT (short for regression with subsequent independence testing) framework (Peters et al., 2014) that has been developed for ANMs. As in ANMs $g ( \cdot ) \equiv 1$ , RESIT approximates the noise terms by the residuals obtained through standard regression for both hypothetical causal models and decides for the model which does not reject the null hypothesis of independence between noise and hypothesized cause. If independence is rejected in both causal directions, the direction with the higher p-value is selected in cases where a decision is forced.<sup>1</sup>

In LSNMs, the noise is typically extracted through estimating f and $g .$ When the noise follows a standard Gaussian distribution, Immer et al. (2023) provide a weakly consistent spline-based estimator. To relax the Gaussianity assumptions, subsequent work has developed estimators that are more robust to heavy-tailed (Tran et al., 2024) or skewed noise (Klippert & Marx, 2026), which led to performance improvements on standard benchmarks. In this paper, we build upon the result of Sun & Schulte (2023), who show that consistently estimating mean and standard deviation is sufficient. We provide a derivation of this result below and propose appropriate estimators in Section 3.

Standardized Residuals Given an LSNM for $X  Y$ as defined in Eq. (1), then (Sun & Schulte, 2023)

$$
\frac { Y - \mathbb { E } [ Y | X ] } { \operatorname { S d } ( Y | X ) } = : \hat { N } _ { Y } \bot \bot X .\tag{2}
$$

In other words, if we had a consistent estimator for $E [ Y | X ]$ and $\operatorname { S d } ( Y | X )$ the estimated residuals $\hat { N } _ { Y }$ are independent of cause X. The implication of this result is that such an estimator would be applicable to most continuous noise distributions, since we only require that the first two moments of $N _ { Y }$ exist. In their paper, Sun & Schulte (2023) discuss the advantages of independence-based approaches in this context, but do not derive a general estimator for the above quantities.

Before introducing our proposed estimators, we briefly sketch the proof idea to provide more intuition to the result in Eq. (2). We first derive that

$$
\begin{array} { r } { \mathbb { E } [ Y | X ] = f ( X ) + g ( X ) \mathbb { E } [ N _ { Y } ] , } \\ { \mathrm { S d } ( Y | X ) = g ( X ) \mathrm { S d } ( N _ { Y } ) . \qquad } \end{array}
$$

Note that an estimator that consistently learns $\mathbb { E } [ Y | X ]$ will not coincide with $f ( X ) \ { \mathrm { i f } } \ \mathbb { E } [ N _ { Y } ] \neq 0 .$ . Similarly, $\operatorname { S d } ( Y | X )$ may differ from $g ( X )$ whenever $\mathrm { S d } ( N _ { Y } ) \neq 1$ . Importantly, by estimating the noise variable through Eq. (2), all terms that depend on X cancel and we are guaranteed to learn the standardized version of $N _ { Y }$ in the large sample limit (Sun & Schulte, 2023), i.e.,

$$
{ \frac { Y - \mathbb { E } [ Y | X ] } { \operatorname { S d } ( Y | X ) } } = { \frac { g ( X ) ( N _ { Y } - \mathbb { E } [ N _ { Y } ] ) } { g ( X ) \operatorname { S d } ( N _ { Y } ) } } = { \frac { N _ { Y } - \mathbb { E } [ N _ { Y } ] ) } { \operatorname { S d } ( N _ { Y } ) } }
$$

which we will exploit in our approach. Alternatively, Lemma 4.1 by Sun & Schulte (2023) clarifies this result by showing that for any LSNM, as defined in Eq. (1), there exists a likelihood-equivalent LSNM

$$
Y = \mu ( X ) + \sigma ( X ) \tilde { N } _ { Y } , \quad \tilde { N } _ { Y } = \frac { N _ { Y } - \mathbb { E } [ N _ { Y } ] } { \mathrm { S d } ( N _ { Y } ) } \amalg X ,
$$

under which $\mu ( X ) = \mathbb { E } [ Y | X ]$ and $\sigma ( X ) = \operatorname { S d } ( Y | X )$ . We further note, that these results directly translate to the multivariate setting, where X is replaced by X.

## 2.3. Multivariate Causal Discovery

To infer DAGs from an iid sample of a multivariate random vector X, we adapt the RESIT (Peters et al., 2014) algorithm to LSNMs as shown in Algorithm 1. We call the resulting algorithm LOSCAR (short for Location-Scale RESIT). Similar adaptations to LSNMs have been explored by Bodik & Chavez-Demoulin (2025); Khan et al. (2026).

In summary, akin to the original RESIT method, Algorithm 1 consists of two phases. Phase 1 learns a causal ordering π in which the parent set for each node k consists of all nodes that appear prior to k in the causal ordering. The causal ordering is defined as a bijective mapping $\pi : \{ 1 , \ldots , d \} $ $\{ 1 , \ldots , d \}$ satisfying

$$
\pi ( i ) < \pi ( j ) \mathrm { i f } j \in \mathrm { D e } _ { X ^ { ( i ) } } ^ { \mathcal { G } } ,
$$

where $\mathrm { D e } _ { X ^ { ( i ) } } ^ { \mathcal { G } }$ refers to the descendants of a node $i .$ We start with an empty ordering π and a candidate set ${ \boldsymbol { s } } =$ $\{ 1 , \ldots , d \}$ . Then, we iteratively identify sink nodes, which we add to π and remove from  until  is empty. At each step, we estimate the standardized residuals for each node $k \in \mathcal { S }$ by obtaining the conditional mean and standard deviation of $X ^ { ( k ) } \ \mid \ \{ X ^ { ( i ) } \} _ { i \in S \backslash \{ k \} }$ based on one of the estimators described in Section 3. Note that both the mean and the standard deviation can be non-constant and are modeled through $\{ X ^ { ( i ) } \} _ { i \in S \backslash \{ k \} }$ . We omit this dependence in the notation of Algorithm 1 to avoid clutter. The sink node is then determined as the node for which the estimated p-value between standardized residuals and corresponding parent set $\{ X ^ { ( i ) } \} _ { i \in S \backslash \{ k \} }$ is largest.

Phase 2 of the algorithm prunes superfluous edges for each node k by iteratively checking if removing a member of the current parent set will induce a dependence between the estimated standardized residuals and the nodes appearing before k in the ordering π.

Consistency The original RESIT algorithm is guaranteed to find the correct causal graph  for identifiable ANMs under causal minimality, given an independence oracle and assuming that the noise can be consistently estimated. In Algorithm 1, we only modified the procedure for estimating the noise terms and use the HSIC test. Thus, we need to assume that the data has been generated by an identifiable LSNM (cf. Section 2.1) and we exploit the result that it is sufficient to recover the standardized noise variables, as can be achieved by the multivariate extension to Eq. (2). Further, note that consistent estimation with HSIC can also be achieved via sample splitting if the estimator to extract the noise term is weakly consistent, or without sample splitting if a strongly consistent estimator is available (Mooij et al., 2016, Thm. 20).

Algorithm 1 LOSCAR   
1: Input: iid sample of d-dimensional X,   
significance level α   
2: # Phase 1: Causal order search   
3: $S = \{ 1 , \dots , d \} , \pi = [ ]$   
4: while $\vert { \cal S } \vert > 0$ do   
5: for $k \in { S }$ do   
6: ${ \hat { \mu } } _ { k } , { \hat { \sigma } } _ { k } = \operatorname { E s t i m a t o r } ( X ^ { ( k ) } \mid \{ X ^ { ( i ) } \} _ { i \in { \mathcal { S } } \backslash \{ k \} } )$   
7: $\begin{array} { r } { p _ { k } = \mathrm { H S I C } \left( \{ X ^ { ( i ) } \} _ { i \in \mathcal { S } \backslash \{ k \} } , \frac { X ^ { ( k ) } - \hat { \mu } _ { k } } { \hat { \sigma } _ { k } } \right) } \end{array}$   
8: end for   
9: $\mathrm { L e t } p ^ { * }$ be the largest p-value.   
10: $\boldsymbol { S } = \boldsymbol { S } \backslash \{ \boldsymbol { k } ^ { * } \}$   
11: $\mathrm { P A } ( k ^ { \ast } ) \stackrel { } { = } \dot { S }$   
12: $\pi = \left[ k ^ { * } , \pi \right]$   
13: end while   
14: # Phase 2: Pruning   
15: for $k = 2 , \ldots , d$ do   
16: for $\ell \in \mathrm { P A } ( \pi ( k ) )$ do   
17: $R = \mathrm { P A } ( \pi ( k ) ) \backslash \{ \ell \}$   
18: $\hat { \mu } , \hat { \sigma } = \dot { \mathrm { E s t i m a t o r } } ( X ^ { ( \pi ( k ) ) } \mid \{ X ^ { ( i ) } \} _ { i \in R } )$   
19: $\begin{array} { r } { p = \mathrm { H S I C } \left( \{ X ^ { ( i ) } \} _ { i \in \{ \pi ( 1 ) , \dots , \pi ( k - 1 ) \} } , \frac { X ^ { ( \pi ( k ) ) } - \hat { \mu } } { \hat { \sigma } } \right) } \end{array}$   
20: $\mathbf { i f } \ p > \alpha$ then   
21: Remove ℓ from $\mathrm { P A } ( \pi ( k ) )$   
22: end if   
23: end for   
24: end for   
25: Output: $[ \mathrm { P A } ( \pi ( 1 ) ) , \ldots , \mathrm { P A } ( \pi ( d ) ) ]$

## 3. Estimating Methods

In this section, we describe three approaches for estimating the conditional mean and scale that we employ within LOSCAR to extract residuals in LSNMs. We propose two methods that estimate the scale from a pretrained in-context quantile regression model (Section 3.1). Specifically, we use TabPFN (Hollmann et al., 2023), a tabular foundation model, to obtain estimates of the conditional mean and conditional quantile function. The first two approaches then recover the scale from the estimated conditional quantile function in different ways. The third approach (Section 3.2) jointly estimates the mean and variance functions via feasible generalized least squares, which is consistent under suitable modeling assumptions.

## 3.1. Pretrained In-Context Quantile Regression

Following the argumentation in Section 2.2, we assume without loss of generality that the LSNM is given by

$$
Y = \mu ( X ) + \sigma ( X ) N _ { Y } , \quad N _ { Y } \perp \perp X ,\tag{3}
$$

where $\mu ( X ) = \operatorname { \mathbb { E } } [ Y | X ]$ and $\sigma ( X ) = { \sqrt { \operatorname { V a r } ( Y | X ) } }$ . We obtain an estimate of µ directly from TabPFN and recover $\sigma$ from the quantile function of $Y | X \ = \ x .$ . For a random variable Z with cumulative distribution function (CDF) $F _ { Z } ( z ) = \mathbf { P } ( Z \leq z )$ , the quantile function is given by

$$
\begin{array} { r } { Q _ { Z } ( \tau ) = \operatorname* { i n f } \{ z : F _ { Z } ( z ) \ge \tau \} , \quad \tau \in ( 0 , 1 ) . } \end{array}
$$

Under the assumption of symmetric noise, mean and median of $Y | X \ = \ x$ coincide, in which case the residuals can be extracted solely based on the quantile function when estimating the scale based on $Q _ { Y | { \pmb X } = { \pmb x } }$

## 3.1.1. QUANTILE-DIFFERENCE SCALE ESTIMATION

As shown in Proposition 3.1, knowledge of two quantiles of $N _ { Y }$ and the quantile function $Q _ { Y | { \pmb X } = { \pmb x } }$ allows us to recover $\sigma ( { \pmb x } )$ exactly. More importantly, $\sigma ( { \pmb x } )$ can be recovered up to scale without knowledge of the quantiles of $N _ { Y }$ . This is sufficient to recover the residuals up to scale within LOSCAR under the LSNM assumption, as for any constant $c \neq 0$ we have that

$$
( Y - \mu ( X ) ) / ( c \sigma ( { \pmb x } ) ) \sqcup X .
$$

In the implementation, we calculate the quantiles of $N _ { Y }$ under the assumption that $N _ { Y } \sim \mathcal { N } ( 0 , 1 )$ . This choice, however, does not require the true distribution of $N _ { Y }$ to be Gaussian, as any choice of $Q _ { N _ { Y } } ( \tau _ { 1 } ) - Q _ { N _ { Y } } ( \tau _ { 2 } )$ only fixes the otherwise irrelevant multiplicative constant.

Proposition 3.1. Let $\tau _ { 1 } , \tau _ { 2 } \in ( 0 , 1 )$ . Under the LSNM in Equation 3, we have that

$$
\begin{array} { c } { { \sigma ( { \pmb x } ) = \displaystyle \frac { Q _ { Y | { \pmb X } = { \pmb x } } ( \tau _ { 1 } ) - Q _ { Y | { \pmb X } = { \pmb x } } ( \tau _ { 2 } ) } { Q _ { N _ { Y } } ( \tau _ { 1 } ) - Q _ { N _ { Y } } ( \tau _ { 2 } ) } } } \\ { { \propto Q _ { Y | { \pmb X } = { \pmb x } } ( \tau _ { 1 } ) - Q _ { Y | { \pmb X } = { \pmb x } } ( \tau _ { 2 } ) . } } \end{array}
$$

Proof. Let $\tau \in ( 0 , 1 )$ . We begin by showing that

<sup>Q</sup>Y|X=x<sup>(τ</sup> <sup>)</sup> <sup>=</sup> <sup>inf</sup>{<sup>y</sup> <sup>:</sup> <sup>F</sup>Y|X=x<sup>(y)</sup> ≥ <sup>τ</sup>}   
<sup>=</sup> <sup>inf</sup>{<sup>y</sup> <sup>:</sup> <sup>P(Y</sup> ≤ <sup>y</sup>|<sup>X</sup> <sup>=</sup> <sup>x)</sup> ≥ <sup>τ</sup>}   
$= \operatorname* { i n f } \{ y : \mathrm { P } ( \mu ( { \pmb x } ) + \sigma ( { \pmb x } ) N _ { Y } \leq y ) \geq \tau \}$   
$= \operatorname* { i n f } \left\{ y : \mathrm { P } \left( N _ { Y } \leq { \frac { y - \mu ( { \pmb x } ) } { \sigma ( { \pmb x } ) } } \right) \geq \tau \right\}$   
$ = \mu ( \pmb { x } ) + \sigma ( \pmb { x } ) \operatorname* { i n f } \left\{ n _ { y } : \mathbf { P } ( N _ { Y } \leq n _ { y } ) \geq \tau \right\}$   
$\mathbf { \xi } = \mu ( \pmb { x } ) + \sigma ( \pmb { x } ) Q _ { N _ { Y } } ( \tau ) .$

It follows that for any $\tau _ { 1 } \neq \tau _ { 2 } \in ( 0 , 1 )$

$$
\begin{array} { r l } & { Q _ { Y | \pmb { X } = \pmb { x } } ( \tau _ { 1 } ) - Q _ { Y | \pmb { X } = \pmb { x } } ( \tau _ { 2 } ) } \\ { = } & { \sigma ( \pmb { x } ) Q _ { N _ { Y } } ( \tau _ { 1 } ) - \sigma ( \pmb { x } ) Q _ { N _ { Y } } ( \tau _ { 2 } ) } \\ { } & { \Longleftrightarrow \sigma ( \pmb { x } ) = \displaystyle \frac { Q _ { Y | \pmb { X } = \pmb { x } } ( \tau _ { 1 } ) - Q _ { Y | \pmb { X } = \pmb { x } } ( \tau _ { 2 } ) } { Q _ { N _ { Y } } ( \tau _ { 1 } ) - Q _ { N _ { Y } } ( \tau _ { 2 } ) } . } \end{array}
$$

For numerical stability, we use well-separated quantiles, $( \tau _ { 1 } , \tau _ { 2 } ) = ( 0 . 0 5 , 0 . 9 5 )$ , to avoid a conditional quantile difference close to zero. In Appendix B.1.2, we compare the default setting with $( \tau _ { 1 } , \tau _ { 2 } ) = ( 0 . 2 5 , 0 . 7 5 )$ in the bivariate setting and obtain comparable performance in the downstream causal discovery task.

## 3.1.2. GAUSS-LEGENDRE SCALE ESTIMATION

Unlike the previous approach, this method uses both the estimated conditional mean and the conditional quantile function to recover $\sigma .$ Hence, a drawback of this method is that errors in the estimated conditional mean directly propagate to the variance estimate.

Given a fixed $\mathbf { \nabla } X = x$ we have that

$$
\begin{array} { l } { \displaystyle \mathrm { V a r } ( Y | X = x ) = \mathbb { E } \Big [ \big ( Y - \mathbb { E } [ Y | X = x ] \big ) ^ { 2 } | X = x \Big ] } \\ { \displaystyle \qquad = \int _ { \mathcal { V } } \big ( y - \mathbb { E } [ Y | X = x ] \big ) ^ { 2 } d F _ { Y | X = x } \big ( y \big ) . } \end{array}
$$

Using the well-known quantile representation where for $U \sim \mathrm { U n i f } ( 0 , 1 )$ we get $Y \overset { d } { = } Q _ { Y | X = x } ( U )$ , the conditional variance can also be expressed as

$$
\operatorname { V a r } ( Y | X = x ) = \int _ { 0 } ^ { 1 } \bigl ( Q _ { Y | X = x } ( u ) - \mathbb { E } [ Y | X = x ] \bigr ) ^ { 2 } d u .
$$

Thus, $\sigma ( { \bf x } ) = \sqrt { \mathrm { V a r } ( { \bf Y } | { \bf X } = { \bf x } ) }$ can be recovered from the conditional quantile function and conditional mean. We approximate the integral using the Gauss-Legendre quadrature (Golub & Welsch, 1969), evaluating the estimated quantile function at m nodes. The approximation is exact for polynomials of degree at most $2 m - 1$ given that the quantiles are correctly estimated. Per default, we set $m = 1 0$ We further evaluate setting $m = 1 0 0$ in Appendix B.1.2 to assess the sensitivity with respect to the number of knots.

## 3.1.3. SUITABILITY OF TABPFN

To estimate the quantile levels in practice, we perform incontext quantile regression with TabPFN (Hollmann et al., 2023). It is a tabular foundation model pretrained on a large corpus of tabular prediction tasks including classification, regression, and time-series forecasting. Further, it has been adapted to a range of tasks including causal inference (Robertson et al., 2026). We use the publicly available

TabPFN-3 model (Grinsztajn et al., 2026) which outperforms classical approaches to quantile regression such as quantile XGBoost (Chen & Guestrin, 2016) and quantile regression forests (Meinshausen & Ridgeway, 2006) on a tailored quantile regression variant of TabArena (Erickson et al., 2026). TabPFN-3 decodes quantiles at inference time by inverting the predicted CDF. We use the model as a plug-in estimator without further fine-tuning to obtain the conditional mean and a prespecified set of conditional quantile levels.

In particular, let $f _ { \theta }$ denote the pretrained TabPFN model and let the context dataset consist of the given iid sample of size n, i.e., $\mathbf { \mathcal { D } } = \{ ( \boldsymbol { { \bf x } } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ . Then we estimate the conditional τ -quantile for the i-th datapoint as the in-context prediction at the corresponding training point

$$
\hat { y } _ { i } ^ { ( \tau ) } = f _ { \boldsymbol { \theta } } ( \mathcal { D } , \mathbf { x } _ { i } , \tau ) .
$$

Note that we are only interested in estimating the quantiles within the provided context and thus get the quantiles of the training (context) points. For future work, one could also consider the effect of sample splitting to estimate these quantiles.

While our experiments are based on TabPFN-3, future or commercial models could be used, as well. We also evaluated TabPFN-3.5 (Jager et al.¨ , 2026) which performed, however, subpar to version 3 (cf. Appendix B.1.3). Further details regarding inference and used model checkpoints are provided in Appendix A.4.

## 3.2. Consistent Estimation via Feasible GLS

In this section, we discuss the estimation of mean and variance using a spline-based feasible generalized least squares (FGLS) procedure. We detail estimation based on n iid samples $\{ ( \pmb { x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ that are assumed to be generated by the LSNM

$$
Y _ { i } = B _ { 1 } ( \pmb { X } _ { i } ) ^ { \top } \pmb { \beta } + \exp \left\{ \frac { \left( \gamma _ { 0 } + B _ { 2 } ( \pmb { X } _ { i } ) ^ { \top } \pmb { \gamma } \right) } { 2 } \right\} N _ { Y _ { i } } ,\tag{4}
$$

where $N _ { Y _ { i } } \bot \bot X _ { i } , \mathbb { E } [ N _ { Y _ { i } } ] \ : = \ : 0$ and $\mathbb { E } [ N _ { Y _ { i } } ^ { 2 } ] = 1$ . Here, $B _ { 1 } ( \pmb { X } _ { i } )$ and $B _ { 2 } ( \pmb { X } _ { i } )$ are spline basis functions with fixed dimensions $k _ { 1 } , k _ { 2 }$ , respectively, that increase model flexibility compared to when using X in their place. Unlike $B _ { 1 }$ , we take $B _ { 2 }$ to exclude an intercept to denote $\gamma _ { 0 } \in \mathbb { R }$ separately. The conditional mean and variance are given by

$$
\mu ( \pmb { X } _ { i } ) = \mathbb { E } [ Y _ { i } | \pmb { X } _ { i } ] = B _ { 1 } ( \pmb { X } _ { i } ) ^ { \top } \pmb { \beta } ,\tag{5}
$$

$$
\sigma ^ { 2 } ( { \pmb X } _ { i } ) = { \mathrm { V a r } } ( Y _ { i } | { \pmb X } _ { i } ) = \exp \left( \gamma _ { 0 } + B _ { 2 } ( { \pmb X } _ { i } ) ^ { \top } \gamma \right)\tag{6}
$$

Let $U _ { i } : = \sigma ( \mathbf { X } _ { i } ) N _ { Y _ { i } } , \pmb { \Omega } ( \gamma ) : = \mathrm { d i a g } ( \mathrm { e x p } \{ B _ { 2 } ( \pmb { X } _ { i } ) ^ { \top } \pmb { \gamma } \} )$ The FGLS procedure which produces consistent estimators (Wooldridge, 2013, Ch. 8.4) $\widehat { \beta } _ { \mathrm { F G L S } }$ and $\widehat { \gamma }$ under the model in Eq. 4 is given as follows.

1. Obtain the ordinary least squares (OLS) estimator $\widehat { \beta } _ { \mathrm { { o L s } } }$ from regressing $\begin{array} { r } { \dot { Y _ { i } } = B _ { 1 } ( \mathbf { \bar { X } } _ { i } ) ^ { \top } \beta + U _ { i } } \end{array}$

2. Set $\widehat { U } _ { i } = Y _ { i } - B _ { 1 } ( X _ { i } ) ^ { \top } \widehat { \beta } _ { \mathrm { O L S } }$

3. Let $g _ { 0 } \in \mathbb { R }$ and $E [ \nu _ { i } | X _ { i } ] = 0$ and obtain $\widehat { \gamma }$ via OLS from $\log ( \widehat { U } _ { i } ^ { 2 } ) = g _ { 0 } + B _ { 2 } ( X _ { i } ) ^ { \top } \gamma + \nu _ { i } ,$

4. Let $\scriptstyle { B _ { 1 } }$ be the matrix with i-th row $B _ { 1 } ( X _ { i } ) ^ { \top } , Y =$ $( Y _ { 1 } , \ldots , Y _ { n } ) ^ { \top }$ and obtain

$$
\widehat { \beta } _ { \mathrm { F G L S } } = \left( \boldsymbol { B } _ { 1 } ^ { \intercal } \widehat { \boldsymbol { \Omega } } ( \widehat { \boldsymbol { \gamma } } ) ^ { - 1 } \boldsymbol { B } _ { 1 } \right) ^ { - 1 } \boldsymbol { B } _ { 1 } ^ { \intercal } \widehat { \boldsymbol { \Omega } } ( \widehat { \boldsymbol { \gamma } } ) ^ { - 1 } \boldsymbol { Y } .
$$

The third step is motivated by the fact that

$$
\log ( U _ { i } ^ { 2 } ) = \gamma _ { 0 } + B _ { 2 } ( X _ { i } ) ^ { \top } \gamma + \log ( N _ { Y _ { i } } ^ { 2 } ) .
$$

Let $r : = \mathbb { E } [ \log ( N _ { Y _ { i } } ^ { 2 } ) ] , \nu _ { i } : = \log ( N _ { Y _ { i } } ^ { 2 } ) - r , g _ { 0 } : = r + \gamma _ { 0 }$ Then the exogeneity assumption $\dot { E } [ \nu _ { i } | X _ { i } ] ~ = ~ 0$ holds ensuring consistency of the corresponding OLS regression (Greene & Zhang, 2000, Ch. 11). By plugging in $\widehat { \beta } _ { \mathrm { F G L S } }$ and $\widehat { \gamma }$ into Eq. 5 and Eq. 6, mean and variance are consistently estimated, with the latter only up to scale, since $\gamma _ { 0 }$ is not recovered. It follows that the extracted residuals are consistent up to scale which is sufficient for LOSCAR (cf. Section 2.2).

We note that for numerical stability, we use a small ridge penalty in the empirical implementation of each regression step to mitigate ill-conditioning of the empirical design matrices (Hastie, 2020).

## 4. Related Work

DAG Identifiability Identifiability conditions are well established for linear non-Gaussian models (Shimizu et al., 2006, LiNGAM), causal additive models (Buhlmann et al.¨ , 2014, CAM), (nonlinear) ANMs (Hoyer et al., 2008), post nonlinear additive noise models (Zhang & Hyvarinen¨ , 2009, PNL), and LSNMs (Immer et al., 2023; Strobl & Lasko, 2022). Both PNL models and LSNMs include ANMs as a special case. We review LSNMs in Section 2.1.

Quantile-Based Causal Discovery Methods that approach causal discovery through quantile regression are score-based. QCCD (Tagasovska et al., 2020) employs quantile regression to derive a minimum description length (MDL) based score in line with other approaches building upon MDL-based approximations (Budhathoki & Vreeken, 2016; Marx & Vreeken, 2017) of the algorithmic Markov condition (Janzing & Scholkopf ¨ , 2010). Recent work derives a score based on quantile partial effects (QPE) to learn the causal order (Chen et al., 2026), but their estimation method relies on a range of sensitive hyperparameters such as the assumed QPE model or transformation functions. Here, we use in-context quantile regression for estimating conditional mean and standard deviation based on pretrained tabular foundation models which does not require further tuning.

Causal Discovery with LSNMs Besides the work discussing identifiability of LSNMs, Lin et al. (2024) derive a method for symmetric noise leveraging skewness for identification. The code is, however, not publicly available. Further, HOST (Duong & Nguyen, 2023) iteratively identifies source nodes assuming Gaussianity. ROCHE (Tran et al., 2024) accounts for heavy-tailed noise by modeling a t-distribution, and SKEWD (Klippert & Marx, 2026) implicitly models skewness by assuming a skew-normal. We include the latter three as baselines. ROCHE, however, can only be applied to the bivariate setting.

Amortized Causal Discovery Through simulating data from causal networks during training, amortized causal discovery (Lorch et al., 2022) frames causal discovery as a supervised learning task (Petersen et al., 2023). By embedding causal assumptions into data generation and building upon the predictive capabilities of transformer models, more recent approaches show promising results for DAG learning and data imputation (Thompson et al., 2026; Li et al., 2026; Xu et al., 2026) and learning causal graphs in the presence of confounding and non-additive models such as CDFM (Qiao et al., 2026) and FOUNDCAUSE (Blobaum¨ et al., 2026). Unlike these models, our method does not require extensive training but can leverage any pretrained tabular foundation model that can perform quantile regression. Further, LOSCAR directly exploits identification results for LSNMs.

## 5. Experiments

We evaluate LOSCAR, using quantile-difference scale estimation via TabPFN by default, alongside the Gauss-Legendre variant LOSCAR-GL and the FGLS variant LOSCAR-FGLS against state-of-the-art (SOTA) baselines in bivariate and multivariate settings. To avoid any biases due to scaling, we standardize all datasets. We compare accuracy and area under the decision rate curve (AUDRC) in the bivariate setting and F1 score, order errors, structural Hamming distance (SHD) and structural intervention distance (SID) in the multivariate setting. Further metric details and implementation details of the baselines including a runtime analysis are provided in Appendices A.1 and A.5, respectively. Hyperparameters of LOSCAR are specified in Appendix A.4.

## 5.1. Bivariate Experiments

We compare against LOCI-IT, an independence-based method assuming Gaussianity, ROCHE, QCCD and the likelihood-based SKEWD-LL (consistent with multivariate SKEWD), as heteroscedastic baselines. We further include CAM (Buhlmann et al.¨ , 2014) as an ANM baseline and FOUNDCAUSE and CDFM as amortized causal discovery baselines.

Table 1. Averaged Results on Bivariate Benchmarks. Top two are in bold and the best result is underlined.
<table><tr><td>Method</td><td>Accuracy (%) ↑</td><td> $\mathrm { A U D R C } \left( \% \right) \uparrow$ </td></tr><tr><td>LOSCAR</td><td> $\mathbf { 9 0 . 0 4 \pm 1 0 . 8 4 }$ </td><td> ${ \bf 9 4 . 5 5 \pm 9 . 1 7 }$ </td></tr><tr><td>LOSCAR-GL LOSCAR-FGLS</td><td> ${ \bf 8 9 . 8 1 \pm 1 1 . 2 6 }$   $7 4 . 9 7 \pm 1 9 . 4 9$ </td><td> $9 4 . 3 6 \pm 9 . 8 1$   $7 4 . 5 7 \pm 2 6 . 4 4$ </td></tr><tr><td>SKEWD-LL</td><td> $8 4 . 5 6 \pm 1 9 . 9 5$ </td><td> $8 9 . 5 8 \pm 1 6 . 3 0$ </td></tr><tr><td>CAM</td><td> $7 5 . 1 0 \pm 2 3 . 1 7$ </td><td> $7 3 . 9 0 \pm 2 6 . 4 4$ </td></tr><tr><td>QCCD</td><td> $6 7 . 6 4 \pm 2 6 . 4 4$ </td><td> $6 9 . 9 2 \pm 3 3 . 2 3$ </td></tr><tr><td>ROCHE</td><td> $8 9 . 5 2 \pm 9 . 9 2$ </td><td> $\underline { { \mathbf { 9 4 . 9 9 } } } \pm 7 . 5 6 $ </td></tr><tr><td>LOCI-IT</td><td> $8 4 . 5 4 \pm 1 2 . 7 1$ </td><td> $8 9 . 4 3 \pm 1 3 . 6 3$ </td></tr><tr><td>FOUNDCAUSE CDFM</td><td> $8 6 . 5 5 \pm 1 7 . 9 6$ </td><td> $9 0 . 9 7 \pm 1 4 . 7 6$ </td></tr></table>

Our evaluation comprises 19 established benchmark datasets: ANM, LSNM, and multiplicative noise datasets from Tagasovska et al. (2020); the SIM datasets and Tubingen pairs (¨ Mooij et al., 2016); the Dataverse datasets (Guyon et al., 2019); and skewed ANM and LSNM datasets from Klippert & Marx (2026). Table 1 reports average accuracy and AUDRC. Detailed dataset descriptions and benchmark-wise results are provided in Appendix A.2 and B.1.1.

In terms of accuracy, LOSCAR achieves the best performance (90.04%), closely followed by LOSCAR-GL (89.81%). LOSCAR-FGLS trails substantially (74.97%), outperforming only QCCD. This gap between the quantile and FGLS variants of LOSCAR is further illustrated in Figure 1 showing the benchmark-wise accuracies. When investigating the individual benchmark results, the FGLS variant has a slight edge to the other variants on near Gaussian and low-noise setting. Among the baselines, ROCHE is most competitive (89.52%), while the remaining methods reach at most 86.55% (FOUNDCAUSE). AUDRC yields a similar ranking, except that ROCHE achieves the highest score, surpassing LOSCAR by 0.44%.

We report amortized approaches separately, as both were trained on ANM and LSNM models. FOUNDCAUSE also includes skewed noise and performs particularly well on such datasets (see Appendix B.1.1).

## 5.2. Multivariate Experiments

We evaluate the LOSCAR variants against SOTA baselines on synthetic and real-world multivariate datasets. We include SKEWD and HOST as LSNM-specific baselines, CAM, LiNGAM (Shimizu et al., 2006) and NOTEARS-MLP (Zheng et al., 2020) as ANM baselines. We additionally consider RESIT with TabPFN as regressor (RESIT-TABPFN). In Appendix B, we also provide results for RE-SIT with OLS which is outperformed by RESIT-TABPFN. FOUNDCAUSE and CDFM are included as amortized baselines.

Table 2. Results for Light Tunnel. Best amortized and nonamortized models marked (OE=Order Errors).
<table><tr><td>Method</td><td colspan="2">F1↑ OE↓ SHD↓ SID↓</td></tr><tr><td></td><td>Light Tunnel (n = 10 000, d = 20, #edges = 39)</td><td></td></tr><tr><td>LOSCAR LOSCAR-GL</td><td>0.430 12</td><td>45 49 45 49</td></tr><tr><td>LOSCAR-FGLS</td><td>0.430 13 0.382 16</td><td>68 52</td></tr><tr><td>HOST</td><td>0.000 39</td><td>89 181</td></tr><tr><td>SKEWD</td><td>NA NA</td><td>NA NA</td></tr><tr><td>CAM</td><td>0.213 17</td><td>59 158</td></tr><tr><td>LiNGAM</td><td>0.068 25</td><td>82 142</td></tr><tr><td>NOTEARS-MLP</td><td>0.092 20</td><td>59 138</td></tr><tr><td>RESIT-TABPFN</td><td></td><td></td></tr><tr><td></td><td>0.015 38</td><td>128 170</td></tr><tr><td>FOUNDCAUSE</td><td>0.687 0</td><td>21 23</td></tr><tr><td>CDFM</td><td>0.727 1</td><td>18 26</td></tr></table>

Synthetic Datasets We generate Erdos-R ˝ enyi´ graphs (Erdos & R ˝ enyi ´ , 1960) with $\begin{array} { r l r } { d } & { { } \in } & { \{ 6 , 1 2 \} } \end{array}$ nodes and b = d edges. For each $d ,$ we generate 100 ANM and LSNM datasets each of size n = 1 000, with $f ^ { ( j ) }$ and $g ^ { ( j ) }$ given by sums of random sigmoidal functions over the parents of each variable $X ^ { ( j ) }$ . The noise distribution of each $N _ { X ^ { ( j ) } }$ is sampled uniformly from Gaussian, exponential, and skew-t. Further sampling details and results on homogeneous noise distributions are provided in Appendices A.2 and B.2.5, respectively.

We focus on the LSNM results in Figure 2, which show the mean metrics as bar plots including standard errors of the mean. Exact values and corresponding ANM results including precision and recall are reported in Appendices B.2.2 and B.2.3. For d = 6, LOSCAR and LOSCAR-GL achieve the strongest performance among classical baselines in F1, order errors, and SID, while SKEWD performs best in SHD. For d = 12, SKEWD is generally comparable or slightly better, most notably in F1. Among all methods, FOUND-CAUSE performs best overall (particularly for $d { = } 1 2 )$ , while LOSCAR and LOSCAR-GL remain competitive. They outperform FOUNDCAUSE in F1 for d = 6 and, together with SKEWD, in SHD for d = 12. LOSCAR-FGLS is not competitive and falls short to RESIT-TABPFN. Overall, the quantile-based variants outperform the classical baselines except SKEWD, with FOUNDCAUSE achieving the strongest overall performance. Notably, in the corresponding ANM experiments (cf. Appendix B.2.3), LOSCAR-FGLS outperforms the quantile-based variants, whereas SKEWD and CAM are the strongest classical baselines.

![](images/9aaa7b9806d8d04ec255ca60d656c03f4289448ea9af4dd203b6e1464c22a521.jpg)

![](images/cc9a3a2faeb072484a473b0457e40bff8cd6d6d798c4b3881b6e26a9a1a4d273.jpg)

![](images/0a00a94bc3dd63234761e6c13b6311a2a016c63a9126b5de3415aa3d0648f47d.jpg)

![](images/39fb16f0aaead8af6bc6b2be1483f41c7409d5daf96c11b65eb4327ebdcfbc26.jpg)  
Figure 2. Comparison of LOSCAR and baselines in terms of F1, order errors, SHD and SID on LSNM data.

Real-World Data For real-world evaluation, we consider the Sachs (Sachs et al., 2005) dataset and Light Tunnel dataset from the Causal Chambers repository (Gamella et al., 2025), with all details in Appendix A.2. Results for Light Tunnel are shown in Table 2. LOSCAR and LOSCAR-GL are the strongest classical baselines, while the amortized methods substantially outperform all classical approaches across metrics. In particular, FOUNDCAUSE and CDFM achieve 0 and 1 order errors, respectively, compared with 12 for the second-best being LOSCAR. HOST performs poorly, potentially due to the source nodes being approximately uniform, and SKEWD did not terminate after more than 72 hours. Again, note that it is difficult to compare to amortized baselines on synthetic data or Light Tunnel, since the DGPs for these graphs are known. On Sachs most models perform comparably with the amortized baselines outperformed by the classical approaches (cf. Appendix B.2.1). We treat this result with caution as the ground truth of this dataset has been challenged (Mooij et al., 2020).

## 6. Conclusion

We introduced LOSCAR, a residual-based causal discovery method for location-scale noise models with nonparametric in-context quantile regression using tabular foundation models. We proposed two variants that differ in how the conditional scale is estimated for residual extraction and compared them with a consistent FGLS-based variant. Across our experiments, the quantile-based variants consistently outperform the FGLS variant and achieve state-of-the-art (SOTA) performance on established bivariate benchmarks, while remaining on par with SOTA baselines in low- to medium-dimensional settings.

Limitations and Future Work The scalability of LOSCAR is limited by the cost of repeated model fitting and the difficulty of conditional independence testing with large conditioning sets. Moreover, its greedy search may propagate errors across successive steps. To improve robustness to such errors, it would be interesting to extend LOSCAR with beam-search, as explored for RESIT by Ong et al. (2026), in future work.

## Acknowledgements

This work was partially supported by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – 598263919. We acknowledge that Quang Nhat Huynh contributed to this work while employed at TU Dortmund University and the Research Center Trustworthy Data Science and Security.

In this work, we used generative AI tools for code prototyping of parts of the code (Claude Sonnet 5 High from Anthropic). Further, we used the ChatGPT 5.6 model from OpenAI for minor language edits, aiming to enhance readability. LLM-generated code and text was verified, modified and tested for correctness by the authors. We take responsibility for the final content of this work, including text, claims, or artifacts produced with the aid of generative AI.

## References

Azzalini, A. The Skew-Normal and Related Families, volume 3. Cambridge University Press, 2013.

Blobaum, P., Balasubramanian, K., and Kasiviswanathan,¨ S. P. Foundcause: Causal discovery with latent confounders from observational data. arXiv preprint arXiv:2606.17516, 2026.

Bodik, J. and Chavez-Demoulin, V. Identifiability of causal graphs under non-additive conditionally parametric causal models. Journal of Machine Learning Research, 26(264):1–55, 2025.

Budhathoki, K. and Vreeken, J. Causal inference by compression. pp. 41–50. IEEE, 2016.

Buhlmann, P., Peters, J., and Ernest, J. CAM: Causal addi-¨ tive models, high-dimensional order search and penalized regression. The Annals of Statistics, 42(6):2526–2556, 2014.

Chen, T. and Guestrin, C. Xgboost: A scalable tree boosting system. In Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining, pp. 785–794, 2016.

Chen, Y., Sun, X., et al. Causal discovery via quantile partial effect. In International Conference on Learning Representations, volume 2026, pp. 148015–148046, 2026.

Cundy, C., Grover, A., and Ermon, S. Bcd nets: Scalable variational approaches for bayesian causal discovery. Advances in Neural Information Processing Systems, 34: 7095–7110, 2021.

Duong, B. and Nguyen, T. Heteroscedastic causal structure learning. In Proceedings of the 26th European Conference on Artificial Intelligence (ECAI 2023), pp. 598–605. IOS Press, 2023.

Erdos, P. and R ˝ enyi, A. On the evolution of random graphs.´ Publ. Math. Inst. Hungary. Acad. Sci., 5:17–61, 1960.

Erickson, N., Purucker, L., Tschalzev, A., Holzmuller, D.,¨ Desai, P., Salinas, D., and Hutter, F. Tabarena: A living benchmark for machine learning on tabular data. Advances in Neural Information Processing Systems, 38, 2026.

Gamella, J. L., Peters, J., and Buhlmann, P. Causal cham-¨ bers as a real-world physical testbed for ai methodology. Nature Machine Intelligence, 7(1):107–118, 2025.

Golub, G. H. and Welsch, J. H. Calculation of Gauss quadrature rules. Mathematics of computation, 23(106):221– 230, 1969.

Goudet, O., Kalainathan, D., Caillou, P., Guyon, I., Lopez-Paz, D., and Sebag, M. Learning functional causal models with generative neural networks. Explainable and Interpretable Models in Computer Vision and Machine Learning, pp. 39, 2018.

Greene, W. H. and Zhang, C. Econometric analysis, volume 5. Prentice hall Upper Saddle River, NJ, 2000.

Grinsztajn, L., Floge, K., Key, O., Birkel, F., Jund, P.,¨ Roof, B., Manium, M., Hoo, S. B., Buhler, M., Garg,¨ A., et al. Tabpfn-3: Technical report. arXiv preprint arXiv:2605.13986, 2026.

Guyon, I., Statnikov, A., and Batu, B. B. Cause effect pairs in machine learning. Springer, 2019.

Hagberg, A. A., Schult, D. A., and Swart, P. J. Exploring network structure, dynamics, and function using networkx. In Proceedings of the 7th Python in Science Conference, pp. 11 – 15, Pasadena, CA USA, 2008.

Hastie, T. Ridge regularization: An essential concept in data science. Technometrics, 62(4):426–433, 2020.

Hollmann, N., Muller, S., Eggensperger, K., and Hutter, F.¨ TabPFN: A transformer that solves small tabular classification problems in a second. In The Eleventh International Conference on Learning Representations, 2023.

Hoyer, P. O., Janzing, D., Mooij, J. M., Peters, J., and Scholkopf, B. Nonlinear causal discovery with additive¨ noise models. In Advances in Neural Information Processing Systems, volume 21, 2008.

Ikeuchi, T., Ide, M., Zeng, Y., Maeda, T. N., and Shimizu, S. Python package for causal discovery based on lingam. Journal of Machine Learning Research, 24(14):1–8, 2023.

Immer, A., Schultheiss, C., Vogt, J. E., Scholkopf, B.,¨ Buhlmann, P., and Marx, A. On the Identifiability and ¨ Estimation of Causal Location-Scale Noise Models. In International Conference on Machine Learning, pp. 14316– 14332. PMLR, 2023.

Jager, B., Erickson, N., Grinsztajn, L., Birkel, F., Fl ¨ oge,¨ K., Key, O., Kaya, K., Kubler, J., Frankel, A., Schr¨ oder,¨ T., et al. Tabpfn-3.5: Technical report. arXiv preprint arXiv:2609.17895, 2026.

Janzing, D. and Scholkopf, B. Causal inference using the¨ algorithmic markov condition. IEEE Transactions on Information Theory, 56(10):5168–5194, 2010.

Kalainathan, D. and Goudet, O. Causal discovery toolbox: Uncover causal relationships in python. arXiv preprint arXiv:1903.02278, 2019.

Kano, Y., Shimizu, S., et al. Causal inference using nonnormality. In Proceedings of the international symposium on science ofmodeling, the 30th anniversary ofthe information criterion, pp. 261–270, 2003.

Khan, M., Shimizu, S., and Pham, T. Beyond additivity: Causal discovery in location-scale noise models with hidden variables. arXiv preprint arXiv:2606.08196, 2026.

Khemakhem, I., Monti, R., Leech, R., and Hyvarinen, A. Causal autoregressive flows. In Banerjee, A. and Fukumizu, K. (eds.), Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, volume 130 of Proceedings of Machine Learning Research, pp. 3520–3528. PMLR, 13–15 Apr 2021.

Klippert, D. and Marx, A. Skewness-robust causal discovery in location-scale noise models. In Forty-third International Conference on Machine Learning, 2026.

Li, Z.-R., Liu, S.-Y., Wang, T.-Z., and Ye, H.-J. Tabcausal: Pretraining across causal environments for tabular causal discovery. arXiv preprint arXiv:2605.31156, 2026.

Lin, Y., Huang, Y., Liu, W., Deng, H., Ng, I., Zhang, K., Gong, M., Ma, Y.-A., and Huang, B. A skewness-based criterion for addressing heteroscedastic noise in causal discovery. In International Conference on Learning Representations, 2024.

Lorch, L., Sussex, S., Rothfuss, J., Krause, A., and Scholkopf, B. Amortized inference for causal structure¨ learning. Advances in Neural Information Processing Systems, 35:13104–13118, 2022.

Marx, A. and Vreeken, J. Telling cause from effect using mdl-based local and global regression. In 2017 IEEE international conference on data mining (ICDM), pp. 307– 316. IEEE, 2017.

Marx, A. and Vreeken, J. Telling cause from effect by local and global regression. Knowledge and Information Systems, 60(3):1277–1305, 2019.

Meinshausen, N. and Ridgeway, G. Quantile regression forests. Journal ofmachine learning research, 7(6), 2006.

Meinshausen, N., Hauser, A., Mooij, J. M., Peters, J., Versteeg, P., and Buhlmann, P. Methods for causal inference¨ from gene perturbation experiments and validation. Proceedings ofthe National Academy ofSciences, 113(27): 7361–7368, 2016.

Mooij, J. M., Peters, J., Janzing, D., Zscheischler, J., and Scholkopf, B. Distinguishing cause from effect using ob-¨ servational data: Methods and benchmarks. The Journal ofMachine Learning Research, 17(1):1103–1204, 2016.

Mooij, J. M., Magliacane, S., and Claassen, T. Joint causal inference from multiple contexts. Journal of machine learning research, 21(99):1–108, 2020.

Ong, H. J. J., Lim, B. G. S., Tan, R. R. P., and Ikeda, K. Causal discovery in additive noise models using beam search. Artificial Life and Robotics, 31(1):317–326, 2026.

Peters, J. and Buhlmann, P. Structural intervention distance¨ for evaluating causal graphs. Neural computation, 27(3): 771–799, 2015.

Peters, J., Mooij, J. M., Janzing, D., and Scholkopf, B.¨ Causal discovery with continuous additive noise models. Journal ofMachine Learning Research, 15(1):2009–2053, 2014.

Peters, J., Janzing, D., and Scholkopf, B.¨ Elements ofcausal inference: foundations and learning algorithms. The MIT Press, 2017.

Petersen, A. H., Ramsey, J., Ekstrøm, C. T., and Spirtes, P. Causal discovery for observational sciences using supervised machine learning. Journal of Data Science, 21(2), 2023.

Qiao, J., Cai, R., Li, Z., Chen, W., Hua, P., Xu, B., Chen, Z., Hao, Z., and Cui, P. Cdfm: Towards a general-purpose causal discovery foundation model. arXiv preprint arXiv:2607.11508, 2026.

Robertson, J., Reuter, A., Guo, S., Hollmann, N., Hutter, F., and Scholkopf, B. Do-pfn: In-context learning for¨ causal effect estimation. Advances in Neural Information Processing Systems, 38, 2026.

Rolland, P., Cevher, V., Kleindessner, M., Russell, C., Janzing, D., Scholkopf, B., and Locatello, F. Score matching¨ enables causal discovery of nonlinear additive noise models. In Proceedings ofthe 39th International Conference on Machine Learning, volume 162, pp. 18741–18753. PMLR, 17–23 Jul 2022.

Sachs, K., Perez, O., Pe’er, D., Lauffenburger, D. A., and Nolan, G. P. Causal protein-signaling networks derived from multiparameter single-cell data. Science, 308(5721): 523–529, 2005.

Schultheiss, C. and Buhlmann, P. Assessing the overall¨ and partial causal well-specification of nonlinear additive noise models. Journal ofMachine Learning Research, 25 (159):1–41, 2024.

Shimizu, S., Hoyer, P. O., Hyvarinen, A., Kerminen, A.,¨ and Jordan, M. A linear non-Gaussian acyclic model for causal discovery. Journal ofMachine Learning Research, 7(10), 2006.

Spirtes, P., Glymour, C. N., and Scheines, R. Causation, Prediction, and Search. MIT press, 2000.

Strobl, E. V. and Lasko, T. A. Identifying patient-specific root causes with the heteroscedastic noise model. arXiv preprint arXiv:2205.13085, 2022.

Strobl, E. V. and Lasko, T. A. Identifying patient-specific root causes with the heteroscedastic noise model. Journal ofComputational Science, 72:102099, 2023.

Sun, X. and Schulte, O. Cause-effect inference in locationscale noise models: Maximum likelihood vs. independence testing. In Conference on Neural Information Processing Systems, 2023.

Tagasovska, N., Chavez-Demoulin, V., and Vatter, T. Distinguishing cause from effect using quantiles: Bivariate quantile causal discovery. In International Conference on Machine Learning, pp. 9311–9323. PMLR, 2020.

Thompson, R., Zhao, H., Steinberg, D. M., and Bonilla, E. V. Arrow: A foundation model for causal discovery. arXiv preprint arXiv:2605.07204, 2026.

Tran, Q.-D., Duong, B., Nguyen, P., and Nguyen, T. Robust estimation of causal heteroscedastic noise models. In Proceedings ofthe 2024 SIAM International Conference on Data Mining (SDM), pp. 788–796. SIAM, 2024.

Wooldridge, J. M. Introductory Econometrics: A Modern Approach. South-Western Cengage Learning, Boston, MA, 5th edition, 2013.

Xu, S., Mameche, S., and Vreeken, J. Learning causal orderings for in-context tabular prediction. arXiv preprint arXiv:2605.22335, 2026.

Zhang, K. and Hyvarinen, A. On the identifiability of the¨ post-nonlinear causal model. In Proc. 25th Conference on Uncertainty in Artificial Intelligence (UAI2009), pp. 647–655. AUAI Press, 2009.

Zheng, X., Dan, C., Aragam, B., Ravikumar, P., and Xing, E. P. Learning sparse nonparametric DAGs. In International Conference on Artificial Intelligence and Statistics, 2020.

The Appendix is structured as follows. Appendix A contains all information regarding performance metrics, generated and used datasets and hyperparameter configurations of LOSCAR and the baselines we compare to. Then, Appendix B contains additional details to the experiments, as well as further ablation studies.

## A. Implementation Details

## A.1. Metrics

We use the following metrics to compare our method to other baselines.

Bivariate Experiments For the bivariate experiments, we use accuracy, i.e., the number of correct decisions divided by the total number of pairs and the area under the decision rate curve (AUDRC) which measures how well the confidence of a method is aligned with its correct output (Marx & Vreeken, 2019). For LOSCAR, we compute its confidence as the absolute difference between the HSIC test statistics. To compute the confidence of CDFM and FOUNDCAUSE we take the absolute difference in prediction probabilities between both directions; all remaining baselines come with their own confidence score. Given accuracy and confidence for each pair, the AUDRC is computed by first sorting all predictions by their confidence and then averaging the accuracies over the top m predictions for $m = 1 , \ldots , M$ , where M is the number of pairs. Denote ξ(i) as the index of the i-th most certain prediction, $f ( \xi ( i ) )$ the predicted direction and $t ( \xi ( i ) )$ the correct direction, then

$$
\mathrm { A U D R C } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \mathbb { 1 } [ f ( \xi ( i ) ) = t ( \xi ( i ) ) ] .
$$

The AUDRC is bounded by 0 and 1, with larger values indicating better performance. Averages over multiple benchmarks are weighted equally, independent of the number of pairs contributing to a benchmark.

For the Tubingen pairs ( ¨ Mooij et al., 2016), we use the recommended weighting to report accuracy and AUDRC.

Multivariate Experiments To benchmark our approach in the multivariate setting, we compute the following statistics with respect to the ground truth DAG:

• F1, Precision, Recall: All metrics compare predicted to ground truth directed edges.

• Order Errors: Order errors are computed as suggested by Rolland et al. (2022), where given a causal ordering π and a target adjacency matrix A (representing a DAG) they define the topological order divergence, here order errors (OE), as

$$
\mathrm { O E } ( \pi , A ) = \sum _ { i = 1 } ^ { d } \sum _ { j : \pi ( i ) > \pi ( j ) } A _ { i j } \ .
$$

Here, we have $A _ { i j } \in \{ 0 , 1 \}$ with $A _ { i j } = 1$ denoting an existing edge from i to j. If π is a correct ordering for A, $\mathrm { O E } ( \pi , A ) = 0$ . Otherwise, it counts the number of edges that cannot be recovered due to the choice of the causal ordering. Since all methods under comparison learn a DAG, we obtain a causal ordering π from their predicted graphs via the function topological sort of the package NetworkX (Hagberg et al., 2008).

• SHD: The structural Hamming distance (SHD) measures the number of edges in the predicted adjacency matrix that are either missing or not present in the target graph. Reversed edges are counted as two errors. We use the implementation from the Causal Discovery Toolbox (Kalainathan & Goudet, 2019, CDT).

• SID: The structural intervention distance (Peters & Buhlmann¨ , 2015, SID) measures the closeness of two graphs in terms of their causal inference statements. We use the Python wrapper provided by CDT (Kalainathan & Goudet, 2019).

## A.2. Datasets

## A.2.1. GENERATED SYNTHETIC DATASETS

We generate Erdos-R ˝ enyi graphs ( ´ Erdos & R ˝ enyi ´ , 1960) with $d \in \{ 6 , 1 2 \}$ nodes and $b = d$ edges to simulate synthetic data. For each configuration, we construct 100 ANM and LSNM datasets of $n = 1 0 0 0$ samples. The functions $\dot { \boldsymbol f } ^ { ( j ) }$ and $g ^ { ( j ) }$ are generated as sums of random sigmoidal functions of each parent variable of $X ^ { ( j ) }$ . Following the construction of these sigmoidal functions from Tagasovska et al. (2020), we refer to the two structural equation model classes as ANs and LSs, respectively.

For the noise variables $N _ { X ^ { ( j ) } }$ , we consider three distributions with distinct characteristics: symmetric Gaussian noise, positive-valued exponential noise, and unbounded skew-t noise with heavy tails (Azzalini, 2013). In the main experiments, the noise distribution is sampled uniformly from these three choices for each node $N _ { X ^ { ( j ) } }$ . We additionally evaluate each distribution separately and report the corresponding results in Appendix B.2.5.

For each sample i and node $X ^ { ( j ) }$ , the noise is generated according to one of the following two-step procedures:

$$
\bullet \sigma _ { i j } \sim \mathrm { U n i f } ( 1 , \sqrt { 2 } ) , \quad N _ { X _ { i } ^ { ( j ) } } \sim \mathcal { N } ( 0 , \sigma _ { i j } ^ { 2 } ) ,
$$

$$
\bullet \lambda _ { i j } \sim \operatorname { U n i f } ( 1 / \sqrt { 2 } , 1 ) , \quad N _ { X _ { i } ^ { ( j ) } } \sim \operatorname { E x p } ( \lambda _ { i j } ) ,
$$

$$
\bullet \ \omega _ { i j } \sim { \mathrm { U n i f } } ( 0 . 9 1 , 1 . 2 9 ) , \quad N _ { X _ { i } ^ { ( j ) } } \sim { \mathrm { S T } } ( 0 , \omega ^ { 2 } , - 2 , 4 ) ,
$$

where $\operatorname { S T } ( \xi , \omega ^ { 2 } , \alpha , \nu )$ denotes the skew-t distribution as defined in Azzalini (2013). These parameter ranges ensure that the variance of $N _ { X _ { i } ^ { ( j ) } }$ lies within the interval (1, 2) across all three distributions. For non-source nodes $j ,$ the noise is additionally scaled by 0.2 in the ANs setting and by 0.35 in the LSs setting.

## A.2.2. BIVARIATE BENCHMARK DATASETS

To assess our methods performance on the bivariate cause-effect discovery task, we evaluate it on benchmarks drawn from four data repositories:

• The SIM and Tubingen¨ datasets were introduced by Mooij et al. (2016). The SIM datasets each consist of 100 pairs with n = 1 000 each, with Gaussian-process based functions (SIM), a dataset with low-noise (SIM-ln), near Gaussian distributions (SIM-G) and additional confounding (SIM-c). From the Tubingen¨ dataset, we extracted all univariate pairs and excluded pairs with binary variables, resulting in 99 pairs. We use the recommended weighting of the pairs to compute the accuracy and AUDRC on this dataset. Note that n varies for this dataset between a few 100 datapoints and more than 30 000.

• (Goudet et al., 2018) provide the Cha dataset (used in the cause-effect pair challenge), the Net dataset, which is generated from random neural networks, and the Multi dataset generated from polynomial mechanisms. Each datase consists of 300 cause-effect pairs with n = 1 000.

• Tagasovska et al. (2020) introduce five synthetic benchmark datasets: two nonlinear additive noise datasets AN and ANs, where the latter uses sigmoidal functions to ensure that the function is invertible and noise distributions drawn from a Gaussian. Accordingly, LS and LSs provide location-scale models, and last MNU models multiplicative noise with uniform noise distributions. All datasets consist of 100 pairs each with n = 1 000.

• Klippert & Marx (2026) extend the more challenging ANs and LSs datasets from Tagasovska et al. (2020) to include skewed noise variables. The “-med” and $\mathbf { \omega ^ { 6 6 } } \mathbf { - s t r } ^ { \prime \prime }$ variants use noise variables drawn from the skew-normal and a generalized normal distribution with skewness 0.455 and 0.985, respectively. Each of the datasets consists of 200 cause-effect pairs, equally split between skew-normal and generalized normal. The “-ext” datasets consists of 100 pairs with the noise variable drawn from a generalized normal with skewness 1.750. All pairs consist of n = 1 000 samples.

To avoid any biases due to scaling, we standardize cause and effect by default.

## A.3. Real-World Multivariate

Our empirical evaluation uses two real-world datasets that we describe below.

Sachs The Sachs dataset consists of single-cell flow-cytometry measurements of $d = 1 1$ phosphoproteins and phospholipids in human immune cells under several perturbation conditions. The dataset is routinely used to benchmark causal discovery algorithms, whereas its ground truth network has been challenged by multiple authors (Meinshausen et al., 2016; Mooij et al., 2020). In our comparison, we included the observational-only part of the dataset consisting of $n = 8 5 3$ data points and 17 edges commonly used for benchmarking (Cundy et al., 2021).<sup>2</sup>

Light Tunnel The Light Tunnel dataset is part of the Causal Chambers repository (Gamella et al., 2025). The datase consists of measurements collected from a controlled physical light-tunnel system, where light-source and polarizer settings affect multiple sensor measurements. We use all $n = 1 0 0 0 0$ observational samples from the uniform-reference condition, providing measurements of the system variables without targeted interventions.<sup>3</sup> The ground-truth network consists of d = 20 nodes and 39 edges.

## A.4. Hyperparameter Configuration LOSCAR

We implemented our method in Python, extending the implementation of RESIT provided through the lingam pack age (Ikeuchi et al., 2023)<sup>4</sup> which also provides an efficient implementation of the HSIC independence criterion. For all experiments, we fix the significance threshold α used in Phase 2 of LOSCAR to 0.01.

We use TabPFN-3 (Grinsztajn et al., 2026) with model checkpoint tabpfn-v3-regressor-v3 default.ckpt as default in our experiments. Conditional mean and quantile levels are obtained through a single fit and predict sequence where the fitting step provides the data as context to the model. Quantile levels are obtained dependent on the instantiation of LOSCAR. LOSCAR estimates the scale based on the quantile difference (cf. Section 3.1.1) with $\tau _ { 1 } = 0 . 0 5$ and $\tau _ { 2 } = 1 - \tau _ { 1 }$ using the quantiles of the standard normal distribution for the normalization term. LOSCAR-GL selects quantiles to compute the variance integral stated in Section 3.1.2 using the Gauss-Legendre quadrature (Golub & Welsch, 1969) adjusted to the interval (0,1). This procedure deterministically selects m quantiles at which the quantile function is evaluated to compute the integral. By default we set $m = 1 0$ . We ablate both parameter choices in Appendix B.1.2. Further, for RESIT-TABPFN we only compute the conditional mean based on TabPFN in the RESIT algorithm. For both LOSCAR-FGLS and RESIT-OLS, the mean parameters were estimated using ridge regression with a penalty of λ = 0.0001. For each covariate, we used an independent cubic spline basis with 8 features and linear extrapolation beyond the boundary knots. For LOSCAR-FGLS, the variance parameters were estimated using ridge regression with $\lambda = 0 . 0 1$ , with an independent cubic spline basis of 6 features per covariate and the same linear extrapolation scheme.

Compute and Runtime: All experiments were performed single-threaded on an NVIDIA RTX PRO 6000 Blackwell Server Edition GPU node. Variants that did not use TabPFN were deployed on CPU (single-threaded, AMD EPYC 9655 Processor). Running a single pair of the bivariate benchmark took approximately 1 to 3 seconds. For the multivariate synthetic experiments, the runtime for d = 6 and d = 12 for a single DAG were 17 seconds and 2 minutes wall-clock time, respectively, for LOSCAR, whereas LOSCAR-GL was faster for d = 12, with LOSCAR-FGLS being fastest, taking 3 seconds for $d = 1 2 .$ . Details for the multivariate synthetic datasets are provided in Appendix B.2.4. As the real-world datasets contained more edges (especially Light Tunnel, which contained 20 nodes, 39 edges, and 10 000 data points) LOSCAR took longest to compute on these, taking 14 minutes on Sachs and 85 minutes on Light Tunnel, respectively. LOSCAR-GL terminated after 110 minutes on the Light Tunnel dataset.

## A.5. Baselines

We used QCCD (Tagasovska et al., 2020), ROCHE (Tran et al., 2024), and LOCI (Immer et al., 2023) with their default parameters as bivariate-only baselines, where for LOCI we selected the better-performing independence based variant. The default methods as published in the corresponding papers were used. Although QCCD is also implemented for the multivariate case, we conducted the multivariate analysis with the stronger-performing baselines. CAM (Buhlmann et al.¨ , 2014) was run with default parameters form the CDT wrapper (Kalainathan & Goudet, 2019) and pruning enabled since we found that activating the pruning step led to significantly stronger performance. LiNGAM was applied with its default parameters as implemented in CDT. SKEWD was applied with its default parameters, where we chose the likelihood-based variant for the bivariate comparison to be consistent with the multivariate extension that is also based on this implementation.

For the multivariate version we omitted the optimization step involving the expectation conditional maximization algorithm as advised by the authors to improve scalability. NOTEARS-MLP (Zheng et al., 2020) is applied with its recommended parameters $\lambda _ { 1 } = \lambda _ { 2 } = 0 . 0 1$

FOUNDCAUSE (Blobaum et al.¨ , 2026) we use the latest version, v1.0 of FOUNDCAUSE<sup>5</sup> with its default parameters. To force decisions for pairwise causal discovery, we directly compare prediction probabilities for both possible edges $X  Y$ and $Y  X$ and use their absolute difference as a score. For CDFM (Qiao et al., 2026) we apply the same rule to force decisions between pairs and otherwise use the default parameters.<sup>6</sup>

Compute and Runtime: FOUNDCAUSE, CDFM, and HOST have been evaluated on an NVIDIA RTX PRO 6000 Blackwell Server Edition GPU node. All remaining baselines were executed single-threaded on CPU (AMD EPYC 9655 Processor). We note that there was no computational bottleneck, except for SKEWD, for which we terminated the experiment on the Light Tunnel dataset after more than 3 days (2 repeated attempts), however, no GPU compliant version of the method is available. Wall-clock runtimes for each method on the synthetic multivariate experiments are reported in Table 10 and further visualized in Figure 6.

## B. Additional Experiments

## B.1. Additional Bivariate Experiments and Results

## B.1.1. DETAILED BIVARIATE EXPERIMENTAL RESULTS

We report the benchmark-wise accuracy and AUDRC obtained by each method in the bivariate setting in Tables 3 and 4. The quantile-based variants of LOSCAR perform well across all datasets, whereas LOSCAR-FGLS performs well for additive noise but has substantial performance drops in the LSNM case, indicating inappropriate variance estimates. Interestingly, increasing the skewness in the LSNM datasets (LSs-med, LSs-str, LSs-ext) leads to a strong increase of performance for LOSCAR-FGLS (37.5%, 58.0%, 72.0%). This behavior is in contrast to what is observed in most other methods. For CAM, QCCD and CDFM performance substantially degrades with increasing skewness, whereas other methods remain stable.

Among the amortized baselines, there are substantial performance differences between FOUNDCAUSE and CDFM in certain settings. While FOUNDCAUSE achieves almost perfect accuracy for multiplicative uniform noise (MNU), it seems that CDFM resorts to random guessing in that case. In contrast, when there is integrated confounding (SIMc), CDFM achieves 68% accuracy, whereas FOUNDCAUSE achieves 47%. This suggests certain differences in the training datasets used to obtain these causal foundation models.

Table 3. Accuracy in % over All Bivariate Benchmark Datasets. Baselines are split into conventional and amortized. The best result per benchmark is bolded.
<table><tr><td></td><td colspan="5">Tagasovska et al. (2020)</td><td colspan="5">Mooij et al. (2016)</td><td colspan="3">Guyon et al. (2019)</td><td colspan="5">Klippert &amp; Marx (2026)</td></tr><tr><td>Method</td><td>AN</td><td>ANs</td><td>LS</td><td>LSs</td><td>MNU</td><td>SIM</td><td>SIMc</td><td>SIMln</td><td>SIMG</td><td>Tübingen</td><td>Cha</td><td>Net</td><td>Multi</td><td>ANs-med</td><td>ANs-str</td><td>ANs-ext</td><td>LSs-med</td><td>LSs-str</td><td>LSs-ext</td></tr><tr><td>LOSCAR</td><td>100</td><td>100</td><td>100</td><td>99</td><td>100</td><td>79</td><td>82</td><td>72</td><td>76</td><td>71.69</td><td>75.67</td><td>85.67</td><td>85.67</td><td>99.0</td><td>98.0</td><td>100</td><td>96.5</td><td>97.5</td><td>93</td></tr><tr><td>LOSCAR-GL</td><td>100</td><td>100</td><td>99</td><td>100</td><td>100</td><td>77</td><td>81</td><td>73</td><td>76</td><td>69.34</td><td>74.67</td><td>85.33</td><td>86.00</td><td>99.0</td><td>98.0</td><td>100</td><td>98.0</td><td>97.0</td><td>93</td></tr><tr><td>LOSCAR-FGLS</td><td>95</td><td>100</td><td>54</td><td>38</td><td>50</td><td>79</td><td>82</td><td>81</td><td>82</td><td>69.78</td><td>76.67</td><td>81.67</td><td>77.33</td><td>97.5</td><td>94.0</td><td>99</td><td>37.5</td><td>58.0</td><td>72</td></tr><tr><td>SKEWD-LL</td><td>100</td><td>100</td><td>100</td><td>100</td><td>87</td><td>54</td><td>51</td><td>81</td><td>88</td><td>45.97</td><td>54.33</td><td>73.33</td><td>72.00</td><td>100.0</td><td>100.0</td><td>100</td><td>100.0</td><td>100.0</td><td>100</td></tr><tr><td>CAM</td><td>100</td><td>100</td><td>100</td><td>53</td><td>86</td><td>57</td><td>60</td><td>87</td><td>81</td><td>57.77</td><td>46.67</td><td>78.33</td><td>34.67</td><td>100.0</td><td>100.0</td><td>100</td><td>82.5</td><td>70.0</td><td>33</td></tr><tr><td>ROCHE</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>78</td><td>82</td><td>77</td><td>81</td><td>77.27</td><td>74.00</td><td>87.67</td><td>83.00</td><td>99.5</td><td>97.5</td><td>95</td><td>99.5</td><td>88.5</td><td>81</td></tr><tr><td>QCCD</td><td>100</td><td>82</td><td>100</td><td>96</td><td>99</td><td>62</td><td>72</td><td>80</td><td>64</td><td>77.09</td><td>53.67</td><td>80.33</td><td>50.67</td><td>56.0</td><td>43.5</td><td>41</td><td>95.5</td><td>21.5</td><td>11</td></tr><tr><td>LOCI-IT</td><td>100</td><td>100</td><td>94</td><td>89</td><td>100</td><td>78</td><td>82</td><td>73</td><td>78</td><td>60.84</td><td>72.67</td><td>86.67</td><td>79.00</td><td>99.5</td><td>99.5</td><td>98</td><td>72.0</td><td>69.0</td><td>75</td></tr><tr><td>FOUNDCAUSE</td><td>100</td><td>100</td><td>98</td><td>94</td><td>99</td><td>54</td><td>47</td><td>71</td><td>93</td><td>60.95</td><td>67.00</td><td>73.67</td><td>94.33</td><td>100.0</td><td>99.5</td><td>93</td><td>100.0</td><td>100.0</td><td>100</td></tr><tr><td>CDFM</td><td>92</td><td>86</td><td>92</td><td>88</td><td>46</td><td>52</td><td>68</td><td>71</td><td>88</td><td>62.30</td><td>60.33</td><td>76.67</td><td>75.00</td><td>96.5</td><td>96.5</td><td>85</td><td>96.0</td><td>84.0</td><td>81</td></tr></table>

## B.1.2. ABLATION STUDIES BIVARIATE BENCHMARK

In the following ablation study, we varied several aspects of LOSCAR. For the default variant which uses TabPFN to estimate the conditional mean and estimates the conditional scale based on the quantile difference (Section. 3.1.1) we ablate the parameter $\tau _ { 1 } \in \{ 0 . 0 5 , 0 . 2 5 \}$ with $\tau _ { 2 } = 1 - \tau _ { 1 }$ , where the default is 0.05. Similarly, for the Gauss-Legendre variant, we ablate the number of quantiles m that are used for computing the integral. We use m 10, 100 . Further, we added two ANM-based baselines RESIT-TABPFN which implements RESIT using TabPFN as in-context regression method and RESIT-OLS which instead uses spline-based regression.

Table 4. AUDRC in % over All Bivariate Benchmark Datasets. Baselines are split into conventional and amortized. The best result per benchmark is bolded.
<table><tr><td></td><td colspan="5">Tagasovska et al. (2020)</td><td colspan="5">Mooij et al. (2016)</td><td colspan="3">Guyon et al. (2019)</td><td colspan="5">Klippert &amp; Marx (2026)</td></tr><tr><td>Method</td><td>AN</td><td>ANs</td><td>LS</td><td>LSs</td><td>MNU</td><td>SIM</td><td>SIMc</td><td>SIMln</td><td>SIMG</td><td>Tübingen</td><td>Cha</td><td>Net</td><td>Multi</td><td>ANs-med</td><td>ANs-str</td><td>ANs-ext</td><td>LSs-med</td><td>LSs-str</td><td>LSs-ext</td></tr><tr><td>LOSCAR</td><td>100.00</td><td>100.00</td><td>100.00</td><td>99.96</td><td>100.00</td><td>87.81</td><td>93.72</td><td>87.60</td><td>94.09</td><td>70.36</td><td>71.83</td><td>96.79</td><td>95.69</td><td>99.93</td><td>99.94</td><td>100.00</td><td>99.79</td><td>99.85</td><td>99.09</td></tr><tr><td>LOSCAR-GL</td><td>100.00</td><td>100.00</td><td>99.98</td><td>100.00</td><td>100.00</td><td>88.27</td><td>92.78</td><td>87.87</td><td>94.08</td><td>66.40</td><td>71.49</td><td>96.90</td><td>96.41</td><td>99.92</td><td>99.95</td><td>100.00</td><td>99.93</td><td>99.86</td><td>98.92</td></tr><tr><td>LOSCAR-FGLS</td><td>98.32</td><td>100.00</td><td>52.71</td><td>22.87</td><td>66.92</td><td>88.77</td><td>92.71</td><td>84.55</td><td>93.25</td><td>62.29</td><td>76.12</td><td>92.16</td><td>77.50</td><td>99.09</td><td>97.94</td><td>99.96</td><td>16.80</td><td>40.00</td><td>54.93</td></tr><tr><td>SKEWD-LL</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>97.58</td><td>64.58</td><td>64.33</td><td>94.57</td><td>94.95</td><td>54.57</td><td>57.50</td><td>81.93</td><td>92.01</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td></tr><tr><td>CAM</td><td>100.00</td><td>100.00</td><td>100.00</td><td>36.29</td><td>76.45</td><td>67.76</td><td>68.82</td><td>87.20</td><td>88.24</td><td>69.37</td><td>41.05</td><td>84.88</td><td>40.02</td><td>100.00</td><td>100.00</td><td>100.00</td><td>74.08</td><td>59.61</td><td>10.24</td></tr><tr><td>ROCHE</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>86.61</td><td>93.85</td><td>88.96</td><td>92.43</td><td>75.46</td><td>78.42</td><td>97.62</td><td>94.40</td><td>100.00</td><td>99.96</td><td>99.85</td><td>100.00</td><td>99.28</td><td>97.97</td></tr><tr><td>QCCD</td><td>100.00</td><td>91.23</td><td>100.00</td><td>99.87</td><td>99.99</td><td>70.69</td><td>83.31</td><td>92.19</td><td>76.30</td><td>84.38</td><td>60.52</td><td>93.89</td><td>62.68</td><td>57.15</td><td>26.68</td><td>27.93</td><td>99.68</td><td>0.62</td><td>1.41</td></tr><tr><td>LOCI-IT</td><td>100.00</td><td>100.00</td><td>99.41</td><td>97.42</td><td>100.00</td><td>88.74</td><td>93.16</td><td>85.97</td><td>93.24</td><td>59.62</td><td>71.33</td><td>96.97</td><td>77.76</td><td>99.99</td><td>99.99</td><td>99.89</td><td>87.28</td><td>56.98</td><td>91.39</td></tr><tr><td>FOUNDCAUSE CDFM</td><td>100.00</td><td>100.00</td><td>99.95</td><td>99.28</td><td>99.99</td><td>63.93</td><td>56.97</td><td>80.26</td><td>99.42</td><td>66.81</td><td>73.33</td><td>89.23</td><td>99.55</td><td>100.00</td><td>100.00</td><td>99.68</td><td>100.00</td><td>100.00</td><td>100.00</td></tr><tr><td></td><td>95.62</td><td>97.20</td><td>98.74</td><td>98.40</td><td>38.75</td><td>64.26</td><td>76.47</td><td>86.58</td><td>95.86</td><td>75.10</td><td>71.94</td><td>90.08</td><td>89.16</td><td>99.85</td><td>99.59</td><td>97.42</td><td>99.60</td><td>95.26</td><td>93.09</td></tr></table>

In Tables 5 and 6 we show the accuracy and the AUDRC for each variant on all bivariate benchmarks. We can see that LOSCAR is relatively robust to the parameter choice staying within a 2.5% difference to the main variant for all benchmarks. For LOSCAR-GL, the results are similar, although performance drops of approximately 5% can be observed for Tubingen¨ and LSs-ext.

To further investigate this behavior, we plot the estimated means and standard deviations for m = 10 and m = 100 for Pair 1 of the LSs benchmark as an example in Figure 3. In general, more nodes should improve the integral approximation via the Gauss-Legendre quadrature, as it is exact for polynomials of degree up to 2m 1. However, note that we do not have access to the ground-truth quantile function, but only an estimate of it. Inaccurate estimates of the quantile function may impair the integral approximation. Figure 3 suggests that the quantile function near the quantile levels of 0 and 1 becomes more noisier, as indicated by the more rougher surface of the quantile function when m = 100, in which case generally more knots are being placed some of which more closely to the limits. In contrast, the most outer quantile function used in the case m = 10 is much smoother hinting that it might be well-recovered. Overall, the results indicate that a higher m does not result in better performance given that there is no ground-truth quantile function available. Nonetheless, the sensitivity with respect to m is rather small as indicated by the results in Tables 5 and 6.

Overall, LOSCAR-FGLS is clearly outperformed by its TabPFN supported counterparts on data which contains flexible location-scale models, whereas its performance under high skewness improves. Therefore, its mean and variance estimates must be worse than in the TabPFN variants, possibly due to a lack of regularization resulting in too flexible function fits. The ANM baselines are comparable, however, as expected they are outperformed by the location-scale variants of LOSCAR.

Table 5. Accuracy in % over All Bivariate Benchmark Datasets. The best result per benchmark is bolded. We include variations of our main approach, ablating τ<sub>1</sub> from its default 0.05 and m from its default 10. Further, RESIT-TABPFN is an ANM variant of LOSCAR performing in-context regression with TabPFN and last, RESIT-OLS is another ANM variant of our method leveraging standard spline-based regression.
<table><tr><td></td><td colspan="5">Tagasovska et al. (2020)</td><td colspan="5">Mooij et al. (2016)</td><td colspan="4">Guyon et al. (2019)</td><td colspan="5">Klippert &amp; Marx (2026)</td></tr><tr><td>Method</td><td>AN</td><td>ANs</td><td>LS</td><td>LSs</td><td>MNU</td><td>SIM</td><td>SIMc</td><td>SIMln</td><td>SIMG</td><td>Tübingen</td><td>Cha</td><td>Net</td><td>Multi</td><td>ANs-med</td><td>ANs-str</td><td>ANs-ext</td><td>LSs-med</td><td>LSs-str</td><td>LSs-ext</td></tr><tr><td>LOSCAR (τ = 0.05)</td><td>100</td><td>100</td><td>100</td><td>99</td><td>100</td><td>79</td><td>82</td><td>72</td><td>76</td><td>71.69</td><td>75.67</td><td>85.67</td><td>85.67</td><td>99.0</td><td>98.0</td><td>100</td><td>96.5</td><td>97.5</td><td>93</td></tr><tr><td>LOSCAR (τ = 0.25)</td><td>100</td><td>100</td><td>100</td><td>97</td><td>100</td><td>77</td><td>82</td><td>71</td><td>76</td><td>69.75</td><td>72.33</td><td>85.33</td><td>85.67</td><td>98.5</td><td>98.0</td><td>100</td><td>96.5</td><td>96.0</td><td>92</td></tr><tr><td>LOSCAR-GL (m = 10)</td><td>100</td><td>100</td><td>99</td><td>100</td><td>100</td><td>77</td><td>81</td><td>73</td><td>76</td><td>69.34</td><td>74.67</td><td>85.33</td><td>86.00</td><td>99.0</td><td>98.0</td><td>100</td><td>98.0</td><td>97.0</td><td>93</td></tr><tr><td>LOSCAR-GL (m = 100)</td><td>100</td><td>100</td><td>99</td><td>100</td><td>100</td><td>80</td><td>82</td><td>73</td><td>77</td><td>64.41</td><td>75.00</td><td>86.67</td><td>84.67</td><td>99.0</td><td>97.5</td><td>100</td><td>99.5</td><td>94.5</td><td>87</td></tr><tr><td>LOSCAR-FGLS</td><td>95</td><td>100</td><td>54</td><td>38</td><td>50</td><td>79</td><td>82</td><td>81</td><td>82</td><td>69.78</td><td>76.67</td><td>81.67</td><td>77.33</td><td>97.5</td><td>94.0</td><td>99</td><td>37.5</td><td>58.0</td><td>72</td></tr><tr><td>RESIT-TABPFN</td><td>100</td><td>100</td><td>63</td><td>12</td><td></td><td>75</td><td>80</td><td>74</td><td>76</td><td>60.48</td><td>73.67</td><td>78.00</td><td>38.00</td><td>100</td><td>100</td><td>100</td><td>28.0</td><td>49.5</td><td>44</td></tr><tr><td>RESIT-OLS</td><td>98</td><td>100</td><td>65</td><td>12</td><td></td><td>74</td><td>82</td><td>78</td><td>77</td><td>60.01</td><td>73.33</td><td>78.00</td><td>37.00</td><td>100</td><td>100</td><td>99</td><td>31.5</td><td>50.5</td><td>43</td></tr></table>

## B.1.3. REMARKS TO TABPFN VERSION 3.5

In this article, we use TabPFN-3 (Grinsztajn et al., 2026), while in theory, our approach is agnostic to the model version of TabPFN as long as this version supports mean and quantile regression. During the time of writing, version 3.5 of TabPFN has been released (Jager et al.¨ , 2026). Therefore, we ran a preliminary ablation study to compare both versions, where for v3.5, we use the model checkpoint tabpfn-v3.5-20260909.safetensors, Release v9.0.0. We found that in our framework, TabPFN-3 performed superior to TabPFN-3.5 and traced the performance difference to the fact that TabPFN-3 returned smoother estimates of conditional mean and estimated quantiles. We illustrate this observation Figure 4, where we plot the estimated conditional mean, the default quantiles for LOSCAR as well as additional quantile ranges for Pair 1 of the LSs dataset (Tagasovska et al., 2020).

Table 6. AUDRC in % over All Bivariate Benchmark Datasets. The best result per benchmark is bolded. We include variations of our main approach, ablating τ<sub>1</sub> from its default 0.05 and m from its default 10. Further, RESIT-TABPFN is an ANM variant of LOSCAR performing in-context regression with TabPFN and last, RESIT-OLS is another ANM variant of our method leveraging standard spline-based regression.
<table><tr><td></td><td colspan="5">Tagasovska et al. (2020)</td><td colspan="4">Mooij et al. (2016)</td><td colspan="4">Guyon et al. (2019)</td><td colspan="6">Klippert &amp; Marx (2026)</td></tr><tr><td>Method</td><td>AN</td><td>ANs</td><td>LS</td><td>LSs</td><td>MNU</td><td>SIM</td><td>SIMc</td><td>SIMln</td><td>SIMG</td><td>Tübingen</td><td>Cha</td><td>Net</td><td>Multi</td><td>ANs-med</td><td>ANs-str</td><td>ANs-ext</td><td>LSs-med</td><td>LSs-str LSs-ext</td></tr><tr><td>LOSCAR (q = 0.05)</td><td>100.00</td><td>100.00</td><td>100.00</td><td>99.96</td><td>100.00</td><td>87.81</td><td>93.72</td><td>87.60 94.09</td><td></td><td>70.36 71.83</td><td>96.79</td><td>95.69</td><td></td><td>99.93</td><td>99.94 100.00</td><td></td><td>99.79 99.85</td><td>99.09</td></tr><tr><td>LOSCAR (q = 0.25)</td><td>100.00</td><td>100.00</td><td>100.00</td><td>99.94</td><td>100.00</td><td>86.84</td><td>92.59</td><td>89.43 93.15</td><td></td><td>65.75 69.67</td><td>96.17</td><td>95.90</td><td>99.93</td><td>99.92</td><td>100.00</td><td>99.78</td><td>99.82</td><td>98.91</td></tr><tr><td>LOSCAR-GL (m = 10)</td><td>100.00</td><td>100.00</td><td>99.98</td><td>100.00</td><td>100.00</td><td>88.27</td><td>92.78</td><td>87.87 94.08</td><td></td><td>66.40 71.49</td><td>96.90</td><td>96.41</td><td>99.92</td><td>99.95</td><td>100.00</td><td>99.93</td><td>99.86</td><td>98.92</td></tr><tr><td>LOSCAR-GL (m = 100)</td><td>100.00</td><td>100.00</td><td>99.98</td><td>100.00</td><td>100.00</td><td>89.13</td><td>93.28 89.07</td><td>93.89</td><td>56.81</td><td>71.56</td><td>96.99</td><td>94.76</td><td>99.98</td><td>99.95</td><td>100.00</td><td>99.99</td><td>99.21</td><td>96.55</td></tr><tr><td>LOSCAR-FGLS</td><td>98.32</td><td>100.00</td><td>52.71</td><td>22.87</td><td>66.92</td><td>88.77</td><td>92.71 84.55</td><td>93.25</td><td>62.29</td><td>76.12</td><td>92.16</td><td>77.50</td><td>99.09</td><td>97.94</td><td>99.96</td><td>16.80</td><td>40.00</td><td>54.93</td></tr><tr><td>RESIT-TABPFN</td><td>100.00</td><td>100.00</td><td>70.87</td><td>2.63</td><td>0.00</td><td>84.47</td><td>90.25</td><td>89.74 90.15</td><td></td><td>53.74 70.58</td><td>91.72</td><td>20.88</td><td>100.00</td><td>100.00</td><td>100.00</td><td>10.94</td><td>23.20</td><td>16.54</td></tr><tr><td>RESIT-OLS</td><td>99.93</td><td>100.00</td><td>77.16</td><td>2.78</td><td>0.11</td><td>87.56</td><td>91.50</td><td>87.29 90.59</td><td></td><td>54.00 72.90</td><td>90.87</td><td>22.54</td><td>100.00</td><td>100.00</td><td>99.99</td><td>13.28</td><td>29.47</td><td>18.01</td></tr></table>

![](images/a9badc9cda0544737f6539b2eb03b63ac1768f31d89783e5eef313eb97867572.jpg)  
Figure 3. Shown are the estimates of the conditional mean and quantiles obtained from TabPFN-3 with m 10, 100 and the corresponding scale computed as described in Section 3.1.2 on Pair 1 of the LSs dataset (Tagasovska et al., 2020). The outer contours of the shaded area correspond to the highest and lowest quantile considered by the corresponding instantiation. The estimates are used for LOSCAR-GL, where m = 10 is the default used in our experiments.

## B.2. Additional Multivariate Experiments and Results

## B.2.1. RESULTS ON THE SACHS DATASET

We show the results in terms of F1, order errors, SHD and SID for the Sachs dataset in Table 7. We see that CAM and HOST perform best, despite their weak performance on Light Tunnel. LOSCAR and LOSCAR-GL perform equally with CAM in terms of order errors and fall short by 2 and 4, respectively, in terms of SHD to HOST. The amortized methods perform comparably to LOSCAR and LOSCAR-GL.

## B.2.2. TABULAR COMPARISON UNDER MIXED LOCATION-SCALE NOISE

We complement the results on synthetic multivariate data with heterogeneous noise distributions shown in Figure 2 in Section 5 by reporting the exact mean performances achieved by the baselines including RESIT-OLS in Table 8. Precision and recall are additionally included, with each variant of LOSCAR generally obtaining higher precision than recall.

![](images/348b57173567ccf014c813b64111b027a05f97b2b67d073a3fee2917db6a4f9a.jpg)  
Figure 4. Shown are the estimates of the conditional mean and quantiles obtained from TabPFN-3 (left) and TabPFN-3.5 (right) on Pair 1 of the LSs dataset (Tagasovska et al., 2020). It can be observed that the estimates for TabPFN-3 exhibit greater smoothness.

Table 7. Results on F1, Order Errors (OE), SHD and SID for the Sachs Dataset. The best result for amortized and non-amortized models each is bolded.
<table><tr><td colspan="3">Method F1 ↑ OE↓ SHD↓ SID↓</td></tr><tr><td>Sachs (n = 853, d = 11, #edges = 17)</td><td></td><td></td></tr><tr><td>LOSCAR 0.296</td><td>7 19</td><td>55</td></tr><tr><td>LOSCAR-GL</td><td>0.320 7 17</td><td>55</td></tr><tr><td>LOSCAR-FGLS</td><td>0.150 9</td><td>34 55</td></tr><tr><td>HOST</td><td>0.400 3</td><td>15 47</td></tr><tr><td>SKEWD</td><td>0.296 7</td><td>19 55</td></tr><tr><td>CAM</td><td>0.444 7</td><td>15 55</td></tr><tr><td>LiNGAM</td><td>7</td><td>17 55</td></tr><tr><td>NOTEARS-MLP</td><td>0.320</td><td></td></tr><tr><td>RESIT-TABPFN</td><td>0.333 6</td><td>16 48 25</td></tr><tr><td></td><td>0.286 9</td><td>52</td></tr><tr><td>FOUNDCAUSE</td><td>0.296 8</td><td>19 50</td></tr><tr><td>CDFM</td><td>0.307 11</td><td>18 50</td></tr></table>

## B.2.3. COMPARISON UNDER MIXED ADDITIVE NOISE

We visualize the results achieved by all baselines across the main metrics for the multivariate synthetic ANM datasets with heterogeneous noise via the bar plot given by Figure 5 showcasing both the mean and its standard error. Exact performances are supplied in Table 9 which also displays precision and recall. The strongest conventional baselines across all metrics are SKEWD and CAM with the latter achieving better results. As an example, in terms of F1 CAM achieves 0.839 when d = 6, whereas LOSCAR, LOSCAR-GL and LOSCAR-FGLS achieve 0.612, 0.632 and 0.718, respectively. Furthermore, SKEWD and CAM outperform both amortized baselines in terms of F1 and SHD, whereas in terms of order errors and SID either FOUNDCAUSE or CDFM yields the best result.

Unlike in the LSNM case, LOSCAR-FGLS outperforms its quantile-based counterparts. Moreover, as in the LSNM case, the performances of all variants of LOSCAR drop notably when the dimensionality increases from d = 6 to d = 12. While this behavior is observed for most methods, there are some exceptions. Most notably, CAM is barely affected in terms of F1 and FOUNDCAUSE improves when increasing d in terms of Recall (0.797 to 0.903) and order errors (0.83 to 0.74) showcasing that amortized baselines can behave unexpectedly.

Table 8. Results on Mean F1, Precision, Recall, Order Errors, SHD and SID on Synthetic Multivariate LSNM Data with Heterogeneous Noise. Baselines are split into conventional and amortized. The best result per benchmark is bolded.
<table><tr><td rowspan="2">Method</td><td colspan="2">F1↑</td><td colspan="2">Precision ↑</td><td colspan="2">Recall ↑</td><td colspan="2">Order errors ↓</td><td colspan="2">SHD↓</td><td colspan="2">SID ↓</td></tr><tr><td>d=6</td><td>d=12</td><td>d=6</td><td>d=12</td><td>d=6</td><td>d=12</td><td>d=6</td><td>d=12</td><td>d=6</td><td>d=12</td><td>d=6</td><td>d=12</td></tr><tr><td>LOSCAR</td><td>0.740</td><td>0.616</td><td>0.749</td><td>0.605</td><td>0.742</td><td>0.638</td><td>1.13</td><td>3.17</td><td>3.40</td><td>9.96</td><td>5.43</td><td>23.46</td></tr><tr><td>LOSCAR-GL</td><td>0.745</td><td>0.620</td><td>0.753</td><td>0.608</td><td>0.747</td><td>0.643</td><td>1.08</td><td>3.11</td><td>3.33</td><td>9.88</td><td>5.31</td><td>22.40</td></tr><tr><td>LOSCAR-FGLS</td><td>0.537</td><td>0.377</td><td>0.504</td><td>0.328</td><td>0.595</td><td>0.463</td><td>2.10</td><td>5.24</td><td>6.82</td><td>19.57</td><td>9.73</td><td>33.32</td></tr><tr><td>HOST</td><td>0.394</td><td>0.343</td><td>0.353</td><td>0.282</td><td>0.458</td><td>0.457</td><td>3.15</td><td>6.27</td><td>9.07</td><td>22.46</td><td>12.33</td><td>38.27</td></tr><tr><td>SKEWD</td><td>0.715</td><td>0.714</td><td>0.748</td><td>0.762</td><td>0.690</td><td>0.675</td><td>1.53</td><td>2.84</td><td>3.29</td><td>6.37</td><td>7.36</td><td>21.61</td></tr><tr><td>CAM</td><td>0.553</td><td>0.521</td><td>0.543</td><td>0.504</td><td>0.572</td><td>0.544</td><td>2.36</td><td>4.91</td><td>5.71</td><td>12.19</td><td>10.48</td><td>29.82</td></tr><tr><td>NOTEARS-MLP</td><td>0.404</td><td>0.384</td><td>0.400</td><td>0.376</td><td>0.420</td><td>0.399</td><td>2.99</td><td>6.14</td><td>7.39</td><td>15.24</td><td>13.78</td><td>40.41</td></tr><tr><td>LiNGAM</td><td>0.289</td><td>0.347</td><td>0.264</td><td>0.306</td><td>0.332</td><td>0.415</td><td>3.77</td><td>6.10</td><td>9.56</td><td>18.74</td><td>17.24</td><td>42.25</td></tr><tr><td>RESIT-TABPFN</td><td>0.701</td><td>0.535</td><td>0.703</td><td>0.510</td><td>0.710</td><td>0.573</td><td>1.27</td><td>3.78</td><td>3.88</td><td>12.52</td><td>6.11</td><td>27.34</td></tr><tr><td>RESIT-OLS</td><td>0.286</td><td>0.204</td><td>0.218</td><td>0.135</td><td>0.432</td><td>0.443</td><td>3.28</td><td>6.31</td><td>13.22</td><td>43.53</td><td>12.68</td><td>34.46</td></tr><tr><td>FOUNDCAUSE</td><td>0.728</td><td>0.732</td><td>0.739</td><td>0.681</td><td>0.752</td><td>0.829</td><td>0.91</td><td>1.27</td><td>3.40</td><td>7.89</td><td>4.70</td><td>10.01</td></tr><tr><td>CDFM</td><td>0.629</td><td>0.600</td><td>0.587</td><td>0.554</td><td>0.703</td><td>0.673</td><td>1.45</td><td>2.77</td><td>5.20</td><td>10.82</td><td>6.56</td><td>20.24</td></tr></table>

![](images/56a7fbe57dcb0fd7f7d594759c6a2eca8777503051eaa1dec6324dc377d0911d.jpg)  
Figure 5. Comparison of LOSCAR and baselines in terms of F1, order errors, SHD and SID on ANM data.

## B.2.4. RUNTIME COMPARISON UNDER MIXED NOISE DISTRIBUTION

In Table 10 and Figure 6, we show the average runtime for our synthetic data experiments with mixed noise distribution per mechanism type and number of nodes d. We can see that all approaches terminate within a reasonable time. While SKEWD

Table 9. Results on Mean F1, Precision, Recall, Order Errors, SHD and SID on Synthetic Multivariate ANM Data with Heterogeneous Noise.
<table><tr><td rowspan="2">Method</td><td colspan="2">F1↑</td><td colspan="2">Precision ↑</td><td colspan="2">Recall ↑</td><td colspan="2">Order errors ↓</td><td colspan="2">SHD↓</td><td colspan="2">SID ↓</td></tr><tr><td>d=6</td><td>d=12</td><td>d=6</td><td>d=12</td><td>d=6</td><td>d=12</td><td>d=6</td><td>d=12</td><td>d=6</td><td>d=12</td><td>d=6</td><td>d=12</td></tr><tr><td>LOSCAR</td><td>0.612</td><td>0.501</td><td>0.614</td><td>0.460</td><td>0.625</td><td>0.560</td><td>1.45</td><td>3.67</td><td>5.02</td><td>14.19</td><td>6.38</td><td>24.90</td></tr><tr><td>LOSCAR-GL</td><td>0.632</td><td>0.509</td><td>0.632</td><td>0.467</td><td>0.647</td><td>0.570</td><td>1.44</td><td>3.51</td><td>4.75</td><td>13.78</td><td>6.04</td><td>23.43</td></tr><tr><td>LOSCAR-FGLS</td><td>0.718</td><td>0.589</td><td>0.716</td><td>0.539</td><td>0.730</td><td>0.667</td><td>1.08</td><td>2.68</td><td>3.56</td><td>11.89</td><td>4.93</td><td>18.84</td></tr><tr><td>HOST</td><td>0.354</td><td>0.271</td><td>0.303</td><td>0.196</td><td>0.442</td><td>0.454</td><td>3.18</td><td>6.31</td><td>10.10</td><td>30.09</td><td>12.44</td><td>39.67</td></tr><tr><td>SKEWD</td><td>0.802</td><td>0.777</td><td>0.840</td><td>0.826</td><td>0.773</td><td>0.738</td><td>0.98</td><td>2.06</td><td>2.23</td><td>5.00</td><td>5.06</td><td>17.30</td></tr><tr><td>CAM</td><td>0.839</td><td>0.852</td><td>0.838</td><td>0.840</td><td>0.847</td><td>0.869</td><td>0.76</td><td>1.28</td><td>2.05</td><td>3.72</td><td>3.45</td><td>9.30</td></tr><tr><td>NOTEARS-MLP</td><td>0.363</td><td>0.365</td><td>0.336</td><td>0.312</td><td>0.407</td><td>0.448</td><td>3.15</td><td>5.98</td><td>8.73</td><td>18.80</td><td>12.99</td><td>38.01</td></tr><tr><td>LiNGAM</td><td>0.426</td><td>0.307</td><td>0.353</td><td>0.229</td><td>0.560</td><td>0.482</td><td>2.25</td><td>5.59</td><td>8.93</td><td>26.66</td><td>10.14</td><td>37.27</td></tr><tr><td>RESIT-TABPFN</td><td>0.648</td><td>0.542</td><td>0.646</td><td>0.504</td><td>0.662</td><td>0.598</td><td>1.29</td><td>3.18</td><td>4.51</td><td>12.83</td><td>5.18</td><td>21.76</td></tr><tr><td>RESIT-OLS</td><td>0.648</td><td>0.527</td><td>0.597</td><td>0.435</td><td>0.738</td><td>0.711</td><td>1.29</td><td>2.87</td><td>5.38</td><td>16.94</td><td>5.79</td><td>19.39</td></tr><tr><td>FOUNDCAUSE</td><td>0.758</td><td>0.695</td><td>0.751</td><td>0.595</td><td>0.797</td><td>0.903</td><td>0.83</td><td>0.74</td><td>3.05</td><td>10.71</td><td>3.39</td><td>5.24</td></tr><tr><td>CDFM</td><td>0.742</td><td>0.727</td><td>0.667</td><td>0.657</td><td>0.862</td><td>0.838</td><td>0.61</td><td>1.06</td><td>3.84</td><td>7.83</td><td>2.68</td><td>8.12</td></tr></table>

has the longest runtime (on CPU), taking on average less than 4 minutes for ANs with $d = 1 2 ,$ , LOSCAR has a comparable runtime with 2 minutes for the d = 12 datasets, whereas LOSCAR-GL is faster and takes at most 75 seconds for ANs with $d = 1 2 .$ . We note, however, that our method scales better than SKEWD for larger datasets. SKEWD did not terminate on Light Tunnel $( d = 2 0$ with 39 edges) within 72 hours, whereas LOSCAR and LOSCAR-GL took around 85 and 110 minutes, respectively. All amortized methods finish below one second and the remaining conventional baselines, which we outperform in terms of performance, are also faster. HOST, the other heteroscedastic baseline, however, also takes 51 seconds on LSs (d = 12).

Table 10. Mean Standard Deviation Runtime (Seconds) based on Synthetic Multivariate Data with Heterogeneous Noise. Baselines are split into conventional and amortized. The best result per benchmark is bolded.
<table><tr><td rowspan="3">Method</td><td colspan="2">LSs</td><td colspan="2">ANs</td></tr><tr><td>d=6</td><td>d=12</td><td>d=6</td><td>d=12</td></tr><tr><td>LOSCAR</td><td> $1 7 . 9 0 \pm 1 . 9 7$ </td><td> $1 2 7 . 0 4 \pm 2 1 . 3 0$ </td><td> $1 7 . 6 1 \pm 1 . 7 3$ </td><td> $1 2 5 . 5 9 \pm 2 2 . 0 4$ </td></tr><tr><td>LOSCAR-GL</td><td> $3 2 . 4 2 \pm 1 . 4 6$ </td><td> $7 2 . 9 9 \pm 4 . 4 1$ </td><td> $3 1 . 4 1 \pm 1 . 5 9$ </td><td> $7 5 . 5 5 \pm 3 . 4 8$ </td></tr><tr><td>LOSCAR-FGLS</td><td> $0 . 5 8 \pm 0 . 0 1$ </td><td> $2 . 8 3 \pm 0 . 0 3$ </td><td> $0 . 5 8 \pm 0 . 0 1$ </td><td> $2 . 8 2 \pm 0 . 0 3$ </td></tr><tr><td>HOST</td><td> $1 7 . 7 0 \pm 7 . 2 8$ </td><td> $5 1 . 5 7 \pm 3 5 . 4 8$ </td><td> $1 3 . 0 4 \pm 8 . 2 8$ </td><td> $2 2 . 5 0 \pm 1 . 1 3$ </td></tr><tr><td>SKEWD</td><td> $5 0 . 3 8 \pm 2 5 . 9 1$ </td><td> $1 6 5 . 8 4 \pm 6 0 . 7 4$ </td><td> $4 9 . 3 6 \pm 2 3 . 3 7$ </td><td> $2 1 5 . 8 7 \pm 8 8 . 0 2$ </td></tr><tr><td>CAM</td><td> $1 . 2 7 \pm 0 . 1 3$ </td><td> $2 . 9 9 \pm 0 . 5 9$ </td><td> $1 . 2 4 \pm 0 . 1 2$ </td><td> $2 . 6 1 \pm 0 . 4 9$ </td></tr><tr><td>NOTEARS-MLP</td><td> $2 . 7 8 \pm 0 . 8 1$ </td><td> $6 . 4 0 \pm 1 . 3 2$ </td><td> $3 . 6 8 \pm 1 . 2 8$ </td><td> $1 2 . 3 3 \pm 7 . 0 7$ </td></tr><tr><td>LiNGAM</td><td> $0 . 7 2 \pm 0 . 1 0$ </td><td> $0 . 8 3 \pm 0 . 1 4$ </td><td> $0 . 6 1 \pm 0 . 1 2$ </td><td> $0 . 6 8 \pm 0 . 0 9$ </td></tr><tr><td>RESIT-TABPFN</td><td> $1 8 . 8 7 \pm 0 . 6 4$ </td><td> $6 4 . 2 6 \pm 2 . 7 2$ </td><td> $1 9 . 0 2 \pm 0 . 6 7$ </td><td> $6 9 . 6 6 \pm 8 . 4 4$ </td></tr><tr><td>RESIT-OLS</td><td> $0 . 5 2 \pm 0 . 0 1$ </td><td> $2 . 3 2 \pm 0 . 0 3$ </td><td> $0 . 5 1 \pm 0 . 0 0$ </td><td> $2 . 3 0 \pm 0 . 0 2$ </td></tr><tr><td>FOUNDCAUSE</td><td> ${ \bf 0 . 3 6 \pm 0 . 0 2 }$ </td><td> ${ \bf 0 . 5 3 \pm 0 . 0 4 }$ </td><td> ${ \bf 0 . 3 7 \pm 0 . 0 4 }$ </td><td> ${ \bf 0 . 5 3 \pm 0 . 0 4 }$ </td></tr><tr><td>CDFM</td><td> ${ \bf 0 . 0 2 \pm 0 . 0 1 }$ </td><td> ${ \bf 0 . 0 3 \pm 0 . 0 1 }$ </td><td> ${ \bf 0 . 0 2 \pm 0 . 0 1 }$ </td><td> ${ \bf 0 . 0 3 \pm 0 . 0 1 }$ </td></tr></table>

## B.2.5. COMPARISON UNDER HOMOGENEOUS NOISE DISTRIBUTIONS

In this section, we display and summarize the results that arise when the noise distribution in the generation of the synthetic multivariate data is held fixed at either exponential, Gaussian or skew-t distribution. Apart from this nuance, the datasets were generated as in the heterogeneous noise case, where 100 datasets for both ANMs and LSNMs for $d = 6 ,$ 12 were created. The parameterizations used are detailed in Section A.2.

![](images/0516f7714d37f0b1ef264c5424975c0d4af6bb4695a0b2d7ec4d2b3c1082ec7d.jpg)  
Figure 6. Average runtimes in seconds (s).

We visualize the mean performances with respect to F1, order errors, SHD and SID as well as precision and recall for completeness using bar plots as in the previous sections. For LSNMs, the results are shown in Figures 7, 8 and 9 for the exponential, Gaussian, and skew-t distribution, respectively. Equivalently, the ANM results are provided by Figures 10, 11 and 12.

LSNM Setting Under exponential noise, LOSCAR and LOSCAR-GL constitute the strongest conventional baselines but are slightly outperformed by FOUNDCAUSE. Similarly under skew-t noise, LOSCAR and LOSCAR-GL perform best with regard to all metrics when d = 6 with SKEWD and FOUNDCAUSE being competitive. However, when d = 12, SKEWD and FOUNDCAUSE slightly outperform LOSCAR and LOSCAR-GL. Under Gaussianity, the assumptions of SKEWD and HOST are fulfilled. This leads to notable performance gains in HOST compared to the other noise settings. HOST outperforms al variants of LOSCAR with respect to all metrics and dimensions and, together with SKEWD, yields the best performances out of the conventional baselines. FOUNDCAUSE, however, performs even better making it the most reliable choice here.

In summary, the results signalize that the quantile-based variants of LOSCAR are well-suited to handle asymmetric noise in LSNMs, as they achieve stable and strong performances across the different noise distributions. This is not surprising given the nonparametric nature of the method. In comparison, while also making no distributional assumptions, LOSCAR-FGLS performs poorly in the LSNM setting.

ANM Setting As in the mixed distribution setting, LOSCAR-FGLS outperforms the quantile variants of LOSCAR under ANMs across exponential and Gaussian noise. Interestingly, under the skew-t distribution this is reversed. Moreover, the variants of LOSCAR remain competitive in the exponential settings, but are comparatively worse under Gaussian and skew-t noise.

Under the exponential distribution, the best performances are achieved by F C , CDFM, CAM and RESIT-OLS (d = 6). All variants of LOSCAR are slightly worse but much more reliable than the other baselines. Compared to the LSNM case, HOST does not become a valid option in the Gaussian noise case, where SKEWD, CAM as well as the amortized methods display the strongest results. This also applies to the skew-t settings, where CAM performs best regarding every metric.

![](images/e90b6874170d4b468be8ddbfac25a62c8c5d9d2be46bb73d1207f408d8f0332b.jpg)

![](images/d58dce9c098854eb22facab5b5d5525a52844320bf6f13f6b4d15ef77a57162c.jpg)

![](images/301b95ef710c3007badd134e7e0a31876ce38a19673f87e57ec1757bfe48bea9.jpg)

![](images/cac1c782e71edb05fedd91a991aa212d0d0fc6fc45988f600201a11f1c61949d.jpg)

![](images/963f394c83079f085963dab480f17c2994ce58ce19ca24ab57d56375ec089949.jpg)

![](images/18abe5e6541d50d0956c18506682a11b00cb3d9b857a464366ad14ba1c3e8d0f.jpg)  
Figure 7. Comparison of LOSCAR and baselines in terms of F1, order errors, SHD and SID on LSNM data with exponential noise.

![](images/9f7cb217a7c839e0b94b719eae94cbe5f353afe2c724ab59befb1072ca51d880.jpg)

![](images/24c7fb993b1b4f02e1c646f54b47977e2b65b8b648b1cb19f4ab92f2a9d04b5e.jpg)

![](images/03dacd60a277e33a856477c0e91d22a17e6b1c4f57da22acb6108bb56b412179.jpg)

![](images/d3630d20b1c23615be0ed25d53f77ac47966852715c281619cfad7f5ec010a57.jpg)

![](images/eee59b97486255a142f129da7be3b3831044bb349516e5a35526e99624dd33c7.jpg)

![](images/a9718573d1fb31ae4c6f9f115a5e833c2229ad01c603cd0eb160450ab8add81d.jpg)  
Figure 8. Comparison of LOSCAR and baselines in terms of F1, order errors, SHD and SID on LSNM data with Gaussian noise.

![](images/8d59e977a46db00f95bab729012dd0c99128b5846fb8bf5b61f9cd6b64ff8293.jpg)  
<sup>SHD</sup> ↓

![](images/35c2cefb78e8a572c3cf254da251657e5e5a66491b92a527f4e07181d282a829.jpg)

![](images/b6d73a5eaf6a27af3b06cf9b116d49c864321ac1343bfff5c9841ffabc0015ee.jpg)

![](images/b66f7163021f3930a7fc660210c035cd23689e184608658128fcc00613887996.jpg)

![](images/8dc620141e5b42fbce4314b9cecb5b43d3c48819e945b6a1936bddf102203076.jpg)

![](images/6e2e859c7a72bc7ff4c08c750b50595af2d68895c42c7e49dbc8f861a0f0d101.jpg)  
Figure 9. Comparison of LOSCAR and baselines in terms of F1, order errors, SHD and SID on LSNM data with skew-t noise.

![](images/b787129815ea597fc171153b40ac6603766a6e257f95e6e246d5d37991d3af0b.jpg)

![](images/e34f8c857a7d8ad155956a9221a1b3848d273414c8a3c27af6b26bf43b7a493f.jpg)

![](images/b7abbe4c665bb3a18967021c969305f0b49ca323f25139523ed31b39fef95a64.jpg)

![](images/bcbc1057ec55fbc32e3195b99b3e741f43b627eccd3a8211d031f4a9a84976bd.jpg)

![](images/780f0d3747980bc66adaab8248eeab7bbea7893cc25f264f4fe51a3a80a289c9.jpg)

![](images/b95218bc27d1913dfa69c6e99dfd2cb06c0c41bcd60be5323b9c4af3294fb6cd.jpg)  
Figure 10. Comparison of LOSCAR and baselines in terms of F1, order errors, SHD and SID on ANM data with exponential noise.

<sup>Recall</sup> ↑  
![](images/31aa57accce9203d9cbf2b5cd5b85f39b5f964f2ee6488c882d7f2350dd5bf5a.jpg)  
<sup>SHD</sup> ↓

![](images/267d904d45b07cbea5d958de15fd9b16ee3cf8e54b54d311b75b7b488b3f16c4.jpg)

![](images/f6fd353ee15fee7b208d742aa8feb2cfebffad6963d7078cb2bed9856b3ff818.jpg)

![](images/3aaa122d24f97deeca0bd1cf86e18d7f19d35e9e4e04d090215e8a1316a125ae.jpg)

![](images/6452ce1cfb0d401528e26e683f0c59c6ed4afcb175f0d6af20d5be8ab35fcf9c.jpg)

![](images/759441f8d563d7e607e64e845dae006f5bec9108a2d8f2d136bacdab41dfa024.jpg)  
Figure 11. Comparison of LOSCAR and baselines in terms of F1, order errors, SHD and SID on ANM data with Gaussian noise.

![](images/f53d4bb85dc8ecaeed302900823ec47f62ceedb04f6cc460feb98b2bda3ed8d4.jpg)

![](images/7501a1fce8508b5df6e9d5c72718b72fb8707808bb9e302b9edf1c829aa03501.jpg)

![](images/152106e342d22d62de3b0f95cd0cc9c3cd171b60acd2ba4ef5f46a57ec63a531.jpg)

![](images/9396dd3b179078a4e0dfc38d2092dd61fe0d5160b8277a63a70ab848eff8de64.jpg)

![](images/8dfef73a12ad0b768be05dd3ad84f0d041b7cc7e2518b84e94c0283c64a46316.jpg)

![](images/28c91c0d513059a36a50124f76ceed0b67f842739a9ae196edf3918496498e4a.jpg)  
Figure 12. Comparison of LOSCAR and baselines in terms of F1, order errors, SHD and SID on ANM data with skew-t noise.