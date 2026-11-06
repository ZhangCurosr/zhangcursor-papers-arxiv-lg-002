# Non-Parametric Mutual Information Estimation with Vine Copulas

Sigurd Holmsen<sup>1,3∗</sup> sigurho@ifi.uio.no

Berit Omli Øksnes<sup>2,3∗</sup> beritoo@math.uio.no

Ingrid Hobæk Haff<sup>2,3,5</sup> ingrihaf@math.uio.no

Sylvia Richardson<sup>3,4</sup> sylvia.richardson@ mrc-bsu.cam.ac.uk

Ali Ramezani-Kebrya<sup>1,3,5</sup> ali@ifi.uio.no

<sup>1</sup>Department of Informatics, University of Oslo <sup>2</sup>Department of Mathematics, University of Oslo <sup>3</sup>Integreat – Norwegian Centre for Knowledge-driven Machine Learning <sup>4</sup>MRC Biostatistics Unit, University of Cambridge <sup>5</sup>TRUST – The Norwegian Centre for Trustworthy AI

## Abstract

We introduce VINE: a non-parametric mutual information (MI) estimator based on simplified D-vine copulas. Approximating the joint density with a simplified Dvine allows the density ratio to simplify to a product of pair copulas, which we can estimate non-parametrically. This structural simplification yields VINE, a flexible and computationally efficient MI estimator. We compare VINE against SotA neural MI estimators on a range of datasets, both under settings where the simplified Dvine assumption holds and where it is violated, demonstrating competitive accuracy at a lower computational cost for low- to medium-dimensional datasets.

## 1 Introduction

Mutual information (MI) is an information-theoretic measure that quantifies statistical dependence between two random vectors by measuring the reduction in uncertainty about one given the other [1]. MI estimation is the key challenge in important applications, e.g., channel capacity in digital communication [1], gene regulatory networks in biology [2], and feature selection in data science [3]. In machine learning with neural networks, efficient MI estimation paves the way for achieving representations of an input that are sufficient for prediction of the ground-truth output while carrying minimal redundancies, i.e., noise in the input, which is irrelevant for output prediction [4].

In practice, accurate MI estimation is known to be notoriously challenging, particularly for random variables of high dimension. Standard non-parametric estimators with binning, kernel density estimation (KDE), and k-nearest neighbor (k-NN) [5, 6], are limited by the curse of dimensionality. These estimators rely on local density estimation, which becomes unreliable in high dimensions since the data points become exponentially sparse [7, 8]. To alleviate this challenge, the ML community has turned to neural MI estimators [9–17]. Despite improving estimation error in high dimensions compared to standard non-parametric models, neural MI estimators are resource-intensive, requiring substantial training epochs and hyperparameter tuning to achieve stable results. Furthermore, they are fundamentally “black-box” models, offering little to no insight into the structural dependencies inherent within data.

Recently, Chen et al. [16] explored a middle ground between the extreme flexibility of neural estimators and the intuitiveness of the Gaussian copula based on vector copula theory. Their method learns “vector ranks” using a neural network and then uses vector copulas to learn their joint distribution. While this method offers a robust trade-off between model capacity and data efficiency, it remains susceptible to the challenges related to neural network training outlined above. In contrast, vine copulas [18–21] provide an alternative framework for learning high-dimensional distributions that does not rely on globalfunction approximation but rather on dimensional decomposition. Vine copulas break down a d-dimensional density into a hierarchy of $d ( d - 1 ) / 2$ bivariate components, allowing for the construction of highly flexible models. Nagler and Czado [22] have proposed a non-parametric implementation of the vine copula and shown that, under specific model assumptions<sup>2</sup>, the density estimator converges at a rate that is independent ofthe dimensions, circumventing the curse of dimensionality. This non-parametric implementation does not suffer from the costly training cycle and hyperparameter tuning of neural MI estimators.

## Summary of contributions.

• We propose an efficient and interpretable MI estimator “VINE” built on non-parametric vine copulas with a closed-form expression for the density ratio under a simplified D-vine copula model.

• Our extensive experiments on popular datasets with known ground-truth MI values and challenging (input, output) dependence structures show that VINE is computationally efficient and improves estimation error compared to SotA neural MI estimator baselines.

## 2 Preliminaries

## 2.1 Vine copulas

A copula C with associated copula density c is a d-dimensional distribution function with Uniform(0, 1) margins. By Sklar’s theorem [23], any joint PDF can be characterized in terms of its marginal densities and a dependence structure, i.e., a copula:<sup>3</sup>

Theorem 2.1 (Sklar’s theorem). Suppose $Z _ { 1 } , \ldots , Z _ { d }$ are absolutely continuous withjoint PDF $f ,$ marginal PDFs $f _ { 1 } , \ldots , f _ { d }$ and $C D F s F _ { 1 } , \ldots , F _ { d } .$ . Then, there exists a copula C with corresponding copula density c such that $\begin{array} { r } { f ( z _ { 1 } , \dots , z _ { d } ) = \prod _ { i = 1 } ^ { d } f _ { j } ( z _ { j } ) \times c ( F _ { 1 } ( z _ { 1 } ) , \dots , F _ { d } ( z _ { d } ) ) } \end{array}$

Sklar’s theorem implies that the marginal distributions and the joint dependence structure can be completely disentangled. Vine copulas [18–21] are constructed by decomposing the copula density into a product o $f d ( \bar { d } - 1 ) / 2$ pair copulas. Specifically, the model can be characterized by a sequence of $d - 1$ trees, each consisting of a node set $N _ { t }$ and an edge set $E _ { t } . \mathrm { ~  ~ \cal ~ A ~ }$ specific edge $e \in E _ { t }$ represents a pair copula between two variables conditional on $t - 1$ other variables. Let us denote an edge $\boldsymbol { e } = \left( j _ { e } , k _ { e } ; D _ { e } \right)$ where $j _ { e } , k _ { e }$ denote the conditioned variables and $D _ { e }$ the set of conditioning variables of edge e. The product of all pair copula densities constitutes the full copula density with the joint PDF of $\boldsymbol { Z } _ { 1 } , \ldots , \boldsymbol { \overline { { Z } } } _ { d }$ as follows:

$$
f ( z _ { 1 } , \dots , z _ { d } ) = \prod _ { l = 1 } ^ { d } f _ { l } ( z _ { l } ) \times \prod _ { t = 1 } ^ { d - 1 } \prod _ { e \in E _ { t } } c _ { j _ { e } , k _ { e } ; D _ { e } } { ( F _ { j _ { e } | D _ { e } } ( z _ { j _ { e } } | z _ { D _ { e } } ) , F _ { k _ { e } | D _ { e } } ( z _ { k _ { e } } | z _ { D _ { e } } ) ) } , { \mathrm { ~ } } { \mathrm { ~ } } { \mathrm { ~ } } { \mathrm { ~ } } { \mathrm { ~ } } { \mathrm { ~ } } { \mathrm { ~ } } { \mathrm { ~ } } { \mathrm { ~ } } { \mathrm { ~ } } { \mathrm { ~ V i n e P D F } } { \mathrm { ~ } } { \mathrm { ~ } } \quad \forall _ { l = 1 } { \mathrm { ~ } } z _ { l } { \mathrm { ~ } }
$$

where $F _ { j | D }$ denotes a conditional CDF of variable j given variables with indices in the set $D .$ . There are several possible structures for the vine tree sequence. An example of a tree sequence in the class of D-vines is shown in Figure 6a, where each node is connected to at most two other nodes.

Vine copula’s simplifying assumption. We adopt the standard simplifying assumption [24–26] for computational tractability. Briefly explained, under the simplifying assumption one assumes that the dependence between $Z _ { j _ { \epsilon } }$ and $Z _ { k _ { \epsilon } }$ conditional of $Z _ { D _ { e } } = z _ { D _ { i } }$ is invariant to the conditioning value $z _ { D _ { e } } , \mathrm { i } . \mathbf { e } . , c _ { j _ { e } , k _ { e } ; D _ { e } }$ depends $\mathrm { o n } z _ { D _ { e } }$ only through its two arguments, for all edges $\boldsymbol { e } = \left( j _ { e } , k _ { e } ; D _ { e } \right)$ that are explicit in the vine tree sequence<sup>4</sup>. While there are distributions where the simplifying assumption holds, such as the multivariate Gaussian distribution [25], it will generally be an approximation. Several works illustrate that simplified vines may nevertheless serve as good approximations [22, 24, 25, 28, 29]. For further evidence of the practical usefulness of simplified vine models, see, e.g., Huk et al. [30], Xiao and Su [31], Tagasovska et al. [32]. More details about the simplifying assumption are provided in Appendix A.1.

## 2.2 Mutual information

Consider two random vectors $\pmb { X } \in \mathbb { R } ^ { d _ { X } }$ and $\pmb { Y } \in \mathbb { R } ^ { d _ { Y } }$ with joint PDF f and marginal PDFs $f _ { X }$ and $f _ { Y }$ . The MI $I ( X ; Y )$ is the Kullback-Leibler (KL) divergence between the joint distribution and the product of the marginals:

$$
I ( X ; Y ) = D _ { K L } \left( f \parallel f _ { X } f _ { Y } \right) = \mathbb { E } _ { f } \left[ \log { \frac { f ( X , Y ) } { f _ { X } ( X ) f _ { Y } ( Y ) } } \right] .\tag{MI}
$$

Since $I ( X ; Y )$ depends on a typically unknown joint density function, several methods have been developed to estimate MI using finite data samples. Traditional non-parametric methods including binning, kernel density estimation (KDE) [5], and k-NN estimators [6] to estimate $f$ directly, are fundamentally limited by the curse ofdimensionality. To address this, the research community has developed neural MI estimators, which can be broadly categorized into deep generative models that learn explicit density parameterizations [8, 10, 13, 17], and density ratio estimation methods that utilize neural discriminators to distinguish joint samples $( X , Y \sim f ( X , Y ) )$ from marginal products $( X , Y \sim f _ { X } ( X ) f _ { Y } ( Y ) )$ [12, 15, 33, 34]. An alternative strategy involves isolating the dependency structure from the marginal distribution using copula theory. While the non-parametric method of Safaai et al. [35] was restricted to low-dimensional settings, the recent Vector Copula Estimator (VCE) framework [16] extended this concept to high-dimensional vectors by adopting a vector copula structure. A more detailed explanation of MI estimators, including MINE [15], InfoNCE [33], SMILE [13], MIENF [14], and VCE [16], is given in Appendix A.2.

## 3 VINE mutual information estimator

We introduce VINE, a computationally efficient simplified D-vine copula-based MI estimator. The MI of X and Y is a function of their joint PDF $f ,$ and by assuming that $f$ is a simplified D-vine, the density ratio $f / ( f _ { X } f _ { Y } )$ simplifies to a product of pair copulas linking together elements of X and elements ofY. Thus, the logarithm of the density ratio becomes a sum over logarithms of pair copulas. R-vine tree structures can in principle be selected from data via Dißmann’s algorithm [36] and similar methods in the non-parametric case [22]. As there are $\begin{array} { r l } { { \frac { d ! } { 2 } } \cdot 2 ^ { ( d - 2 ) ( d - 3 ) / 2 } } \end{array}$ possible R-vine structures for d variables [37], restricting to D-vine structures may appear to sacrifice flexibility. However, we show that it leads to an intuitive expressionfor the MI with improved estimation error in high dimensions since we do not estimate $f _ { X }$ and $f _ { Y }$ separatelyfrom the joint density $f .$

Proposition 3.1 (MI for a simplified D-vine). Let $\pmb { X } \in \mathbb { R } ^ { d _ { X } } , \pmb { Y } \in \mathbb { R } ^ { d _ { Y } }$ , and $d = d _ { X } + d _ { Y }$ , and define $Z = ( X , Y ) \in \mathbb { R } ^ { d }$ , such that

$$
Z _ { l } = \left\{ \begin{array} { l l } { { X _ { l } , } } & { { i f 1 \leq l \leq d _ { X } } } \\ { { Y _ { l - d _ { X } } , } } & { { i f d _ { X } < l \leq d } } \end{array} \right. .
$$

Suppose the joint PDF f is a simplified D-vine. Then the MI reduces to

$$
I ( X ; Y ) = \mathbb { E } _ { f } \Big [ \sum _ { e \in E _ { X Y } } \log c _ { j _ { e } , k _ { e } ; D _ { e } } \left( F _ { j _ { e } | D _ { e } } \big ( Z _ { j _ { e } } | Z _ { D _ { e } } \big ) , F _ { k _ { e } | D _ { e } } \big ( Z _ { k _ { e } } | Z _ { D _ { e } } \big ) \right) \Big ] , \quad \mathrm { ( S u m L o g M I ) }
$$

where $E _ { X Y } = \cup _ { t = 1 } ^ { d - 1 } \{ e \in E _ { t } \colon 1 \leq j _ { e } \leq d _ { X } < k _ { e } \leq d \}$ . In other words, $E _ { X Y }$ is the set ofedges in the D-vine linking elements of X and elements of Y through a pair copula.

## Proof. See Appendix B.

To illustrate this construction, we show a simple example. Let $d = 6$ and $d _ { X } = d _ { Y } = 3$ , such that $( Z _ { 1 } , Z _ { 2 } , Z _ { 3 } ) = ( X _ { 1 } , X _ { 2 } , X _ { 3 } )$ and $( Z _ { 4 } , Z _ { 5 } , Z _ { 6 } ^ { \bullet } ) = ( Y _ { 1 } , \mathbf { \dot { \bar { Y } } } _ { 2 } , Y _ { 3 } )$ . The full joint PDF is visualized in

![](images/2478e37098dbf2dd26dc7c540c28059a6305125aaf0a0f767b6277efffe4de8c.jpg)  
Figure 1: Identifying the density ratio from a D-vine density. Functional arguments are omitted to simplify notation.

Figure 1, where we have highlighted $f _ { X }$ in orange and $f _ { Y }$ in blue. From this figure, we see how the density ratio $f / ( f _ { X } f _ { Y } )$ , being the remaining terms that are highlighted in green, becomes the product of the pair copulas linking elements of X and elements of Y.

VINE leverages on Proposition 3.1 to estimate the full joint PDF (including the margins and the copula) and extracts the relevant pair copulas to compute a Monte Carlo (MC)-type estimate. Given n i.i.d. samples from the true joint density $f , \{ \boldsymbol { Z } ^ { ( i ) } \} _ { i = 1 } ^ { \bar { n } } = \{ ( \boldsymbol { X } ^ { ( i ) } , \boldsymbol { Y } ^ { ( i ) } ) \} _ { i = 1 } ^ { n }$ , we estimate:

$$
\hat { I } ( \pmb { X } , \pmb { Y } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sum _ { e \in E _ { X Y } } \log \hat { c } _ { j _ { e } , k _ { e } ; D _ { e } } \Big ( \hat { F } _ { j _ { e } | D _ { e } } ( Z _ { j _ { e } } ^ { ( i ) } | \pmb { Z } _ { D _ { e } } ^ { ( i ) } ) , \hat { F } _ { k _ { e } | D _ { e } } ( Z _ { k _ { e } } ^ { ( i ) } | \pmb { Z } _ { D _ { e } } ^ { ( i ) } ) \Big ) .\tag{VINE}
$$

The general procedure is summarized in Algorithm 1. In principle, the different components of VINE can be estimated both parametrically and non-parametrically. However, we choose a fully non-parametric approach, as proposed by Nagler and Czado [22]. In Section 4, we argue theoretically why a fully non-parametric implementation of our proposed MI estimator is ideal.

```latex
Algorithm 1: VINE: a simplified D-vine copula based MI estimator.
Input: Data $\{ Z ^ { ( i ) } \} _ { i = 1 } ^ { n }$ with $\pmb { Z } ^ { ( i ) } = ( \mathbf { X } ^ { ( i ) } , \mathbf { Y } ^ { ( i ) } )$ and $\pmb { X } ^ { ( i ) } \in \mathbb { R } ^ { d _ { X } } , \pmb { Y } ^ { ( i ) } \in \mathbb { R } ^ { d _ { Y } }$
Output: MI estimate ${ \hat { I } } ( X ; Y )$
1: Estimate marginal CDFs ${ \hat { F } } _ { 1 } , \ldots , { \hat { F } } _ { d } .$
2: Transform data to pseudo-observations $\hat { F } _ { l } ( Z _ { l } ^ { ( i ) } )$ , as described in Czado [38].
3: Estimate a D-vine copula using the pseudo-observations to obtain pair copula estimates
$\{ \hat { c } _ { j _ { e } , k _ { e } ; D _ { e } } : e \in E _ { t } , t = 1 , \ldots , d - 1 \} ,$
4: Compute ${ \hat { I } } ( X ; Y )$ using Equation (VINE).
```

The flexibility of VINE can be controlled using tuning parameters. A common strategy for pruning the model is to set selected pair copulas to independence copulas. We employ thresholding [39], which means setting pair copulas that exhibit weak empirical dependencies to independence copulas. Specifically, an edge is pruned if its absolute empirical Kendall’s τ falls below a prespecified threshold. This approach allows the estimator to effectively resolve sparse dependence structures while reducing the risk of overfitting. Additionally, to improve the computational speed of the estimator, we parallelize the estimation of individual pair copulas within each tree level. VINE is implemented using the Python package pyvinecopulib [40].

## 4 Theoretical motivation

Suppose we have i.i.d. data $\{ \mathbf { Z } ^ { ( i ) } \} _ { i = 1 } ^ { n } = \{ ( \mathbf { X } ^ { ( i ) } , \mathbf { Y } ^ { ( i ) } ) \} _ { i = 1 } ^ { n }$ sampled from a simplified D-vine density $f$ as defined in Section 2. As shown in Section 3, the true MI simplifies to the expectation of the sum of logarithms of pair copulas linking together the elements of X and Y. Even in an ideal setting

where all components of $f$ are known, we might still have to approximate the expectation by MC, especially when d is large. An ideal MI estimator would therefore be

$$
I ^ { * } ( \mathbf { X } , \mathbf { Y } ) = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \sum _ { e \in E _ { X _ { Y } } } \log c _ { j _ { e } , k _ { e } ; D _ { e } } \left( F _ { j _ { e } | D _ { e } } ( Z _ { j _ { e } } ^ { ( i ) } | \mathbf { Z } _ { D _ { e } } ^ { ( i ) } ) , F _ { k _ { e } | D _ { e } } ( Z _ { k _ { e } } ^ { ( i ) } | \mathbf { Z } _ { D _ { e } } ^ { ( i ) } ) \right) . \quad { \mathrm { ( I d e a l M I E s t ) } }
$$

Under regularity conditions<sup>5</sup>, Equation (IdealMIEst) converges in probability to the ground-truth $I ( \mathbf { X } , \mathbf { Y } )$ by the Weak Law of Large Numbers. Realistically, all components of $f$ are unknown and must be estimated from the data. If $\hat { f }$ is the D-vine density estimator, then plugging in its components into Equation (IdealMIEst) yields the proposed estimator in VINE. Therefore, $\operatorname { i f } { \hat { f } }$ is a consistent approximation of $f ,$ , VINE becomes a good approximation of Equation (IdealMIEst), which is a good approximation of the true MI with a large enough amount of data.

There are several motivations behind estimating the components of the simplified vine model nonparametrically. A fully non-parametric implementation of VINE avoids model misspecification and offers higher flexibility than a parametric implementation. Moreover, Nagler and Czado [22] have shown that under certain assumptions, non-parametric vine density estimators achieve convergence rates that are independent $o f$ the ambient dimension. Specifically, if the pair copulas are $K$ times differentiable, and $\hat { f }$ satisfies assumptions A1 to A4 in Nagler and Czado [22], then for a given z, $| { \hat { f } } ( z ) - f ( z ) |$ is bounded in probability at rate $n ^ { - r }$ , where $r = K / ( 2 K + 2 )$ . In other words, this non-parametric density estimator overcomes the “curse of dimensionality” faced by essentially all consistent d-dimensional non-parametric density estimators, where the error rate increases exponentially with the ambient dimension [41]. Thus, under the simplifying and D-vine assumptions, we expect VINE to outperform MI estimators based on full-dimensional non-parametric density estimation (e.g. binning, KSG [6], KDE [5]).

Even in settings where the simplifying assumption and D-vine assumption do not hold, it may still be the case that $f$ can be well approximated by a simplified D-vine density ${ \hat { f } } .$ Furthermore, although the estimation of $\hat { f }$ involves $d + d ( d - 1 ) / 2$ estimation tasks, training VINE may still be less computationally demanding than training a full neural network.

## 5 Numerical experiments

In this section, we evaluate VINE’s performance on a range of benchmark datasets with known ground-truth MI. We examine how dimensionality, dependency structures, and data structures affect VINE’s performance, and consider settings where the assumptions underlying VINE hold and settings where they are known to be violated. In all experiments, we use a fully non-parametric D-vine estimator and fix the threshold parameter $\tau = 0 . 1$ We benchmark VINE against SotA neural and vector copula based MI estimators including MINE, InfoNCE, MIENF, SMI $\scriptstyle { \mathrm { . E , } }$ and VCE.

## 5.1 Evaluating the simplifying assumption

Experimental setup. We evaluate the estimators under four data-generating distributions spanning settings in which the simplifying assumption holds and settings in which it is violated. In the former, we generate $( \mathbf { X } , \mathbf { Y } ) \sim \tilde { \mathbf { N _ { d } } } ( \bar { \mathbf { 0 } } , \bar { \mathbf { \Sigma } } )$ ), with two different covariance structures: pairwise dependence only between $X _ { i }$ and $Y _ { i } ,$ and spatial dependence, defined by corr $( X _ { i } , Y _ { j } ) = \bar { \mathrm { e x p } } ( - \mathrm { d i s t } ( i \bar { , } j ) / \rho )$ , where dist $( i , j )$ denotes the distance between the coordinates of the features on a 2D grid. For the latter, we use a Mixture of Gaussians (MoG) distribution configured with the same covariance structures described above. We expect the mixture with sparse, pairwise-only dependence to exhibit a milder violation, and the spatial-dependence mixture to produce a stronger violation of the simplifying assumption.

Results. The results are summarized in Figure 2. In the Gaussian task, where the simplifying assumption holds, VINE demonstrates high accuracy and remains robust to high dimensionality and ground-truth MI compared to most neural estimators. We especially observe that MINE, InfoNCE, and SMILE suffer substantial error degradation for $d = 1 2 { \bar { 8 } }$ and $\dot { d } = 2 5 6$ , while VINE performs similarly to or better than the best neural estimators. Crucially, in the MoG dataset where the simplifying assumption is violated, VINE remains remarkably accurate and competitive against state-of-the-art neural estimators. This indicates that while the simplifying assumption is necessary for theoretical bounds, its violation in practice does not critically break the D-vine’s ability to approximate the global dependency structure. Furthermore, VINE achieves this with significant computational efficiency, computing estimates in a fraction of the time required for neural training (see Figure 7). VINE also remains accurate in high dimensions (up to d = 256). For a more thorough investigation of VINE’s limitations in terms of dimensionality and computational time and accuracy, see Appendix C.

![](images/db5af571ba42830796c8cfc7fd8de614f1d9cedf7c3376059a76f8e9e398d5ef.jpg)  
(a) Gaussian Pairs

![](images/9552e28acb0242b02ccf91819439e1f318cc477f741ec3190db3c13210a5c031.jpg)  
(b) Gaussian Spatial

![](images/294ed255b6b62036825407c181d89a95a0ac9636272c336ccfceaf7313c05f8c.jpg)  
(c) MoG Pairs

![](images/6e5ba05d0f986b530f1f3b44469d75adcd64b3cb9297797a3e9d31cc21f7d71e.jpg)  
(d) MoG Spatial  
Figure 2: Performance of VINE and neural estimators across data distributions where the simplifying assumption holds (Gaussian) versus where it is violated (Mixture of Gaussians).

## 5.2 Non-Gaussian datasets

Experimental setup. Following Czyz et al.˙ [42], we extend MI evaluation beyond the simple multivariate Gaussian case. We consider cases where both model assumptions underlying VINE hold true and are violated:

• Margin-wise maps of elliptical distributions: Elliptical distributions, such as the multivariate Gaussian and the multivariate Student’s t, have copulas satisfying the simplified D-vine assumption [25, Theorem 4.1]. According to the invariance property of copulas [38, p. 18], applying strictly increasing transformations to the margins of these distributions preserve the underlying copula, and thus keep model assumptions intact. We consider multivariate Gaussian data with marginal transformations $g ( x ) = \bar { \mathrm { s i g n } } x \cdot x ^ { 2 } , g ( x ) = x ^ { 3 } , g ( x ) = e ^ { x }$ , and multivariate Student’s t data.

• Cross-dimensional maps of Gaussian distributions: Czyz et al.˙ [42] consider cross-dimensional maps of Gaussian variables, such as the Spiral diffeomorphism and the Swiss Roll data. Although these transformations preserve the true MI, they do not preserve the underlying copula, so the simplified D-vine assumption may no longer hold.

• Mixture of Gaussian distributions: Mixture of Gaussian (MoG) distributions can be expressed as $( \boldsymbol { X } , \boldsymbol { Y } ) | \boldsymbol { C } = \boldsymbol { k } \sim \mathrm { N } _ { d } ( \mu _ { k } , \Sigma _ { k } )$ , for $k = 1 , \ldots , K$ , and where C is a latent variable. MoG data do not satisfy the simplified D-vine assumption.

Each experiment is run using 10, 000 data points. See Appendix E.2 for further details of the data generating distributions and comments on the simplifying assumption.

Results. Across all benchmarks, VINE consistently demonstrates high accuracy and stability compared to the neural estimators, especially as dimensionality and MI increase, as seen in Figure 3. For marginal transformations such as cube and sign-square mappings, the performance gap is especially pronounced: the vine’s rank-based mapping inherently exploits copulas’ invariance property and recovers the underlying dependency structure. The neural critics, e.g., MINE and InfoNCE struggle to recover such structure. We note that the simplifying assumption holds in these cases. In more geometrically complex tasks, such as the Spiral and Swiss Roll data, where the simplifying assumption may not theoretically hold, the results show that VINE still captures the vast majority of the information signal. This suggests that VINE can be effective even when the model assumptions are violated. The neural estimators exhibit poor performance in high dimensions and high dependency settings, as demonstrated in Section 5.1. In contrast, VINE remains significantly more robust across these regimes. We observe that VINE performs less optimally on the heavy-tailed Student’s-t data, as it overestimates ground-truth MI, likely by fitting noise within the distribution’s tails. However, this sensitivity to tail noise could be mitigated by adjusting the threshold parameter, τ , upward to force sparser dependencies. While the VCE remains exceptionally accurate on the MoG dataset, likely aided by its architectural assumption of a Gaussian mixture structure within the vector copula, VINE performs well on datasets requiring higher structural flexibility, such as Swiss Roll and the cubed Gaussian.

![](images/79a2ac57bdb5561199930dd72dbd09e9ddbd7d8ffba05f60d045c0bf64528b1d.jpg)  
(a) sign(X) · X<sup>2</sup>, e<sup>Y</sup>

![](images/a2e0a102723af9208847551d4a76171bd755e25623ce9c01c3e3d572f644e5b3.jpg)  
(b) Spiral

![](images/ad5bd4c5a028458bf3b570c0cb9d30b16e6cdcc31e59548d6975521a2e72d214.jpg)  
(c) MoG

![](images/52ceb5701b9731eaa0013643fec76fc7d7f2b08f05b3d90c3ec92611adb1a28c.jpg)  
(d) Student’s-t, ν = 2

![](images/7c908b002af61ea2e75c66a1abffa421c7adbcb950a00e5768f40039422fc0a4.jpg)  
(e) $X ^ { 3 } , Y ^ { 3 }$

![](images/725248940644b8bd8efd1556080d77ecf18246ccbca895e61e4a59ab678be847.jpg)  
(f) Swiss Roll  
Figure 3: MI Estimation on transformed and non-Gaussian data. Row 1 shows the absolute estimation error $| I - { \hat { I } } |$ as dimension d increases (fixed $\rho = 0 . 5 )$ . Row 2 displays error across varying correlation strengths ρ for complex non-linear dependencies (fixed d = 32).

## 5.3 Image dataset with known MI

Experimental setup. To evaluate VINE’s ability to model high-dimensional data, we replicate the image-based benchmarks proposed by Butakov et al. [43]. We consider two distinct synthetic image datasets: Gaussian images and rectangle images with known ground-truth MI, as visualized in Figure 4. MI estimation is known to be difficult in high-dimensional data: for any distribution-free MI estimator, the required sample complexity is proven to depend exponentially on the dimension of the data [7, 8]. However, according to the Manifold Hypothesis, high-dimensional data, such as large images, often lie on a significantly lower-dimensional manifold. In our experiments, we consider $\pmb { X } \in \mathbb { R } ^ { 1 6 \times 1 6 }$ and $\pmb { Y } \in \mathbb { R } ^ { 1 6 \times 1 6 }$ images where the ground-truth $I ( X ; Y )$ is known. Following Butakov et al. [43], we preprocess the raw image data using Principal Component Analysis $( \mathrm { P C A } ) \ e : \mathbb { R } ^ { 1 6 \times 1 6 } \to \mathbb { R } ^ { d }$ We show in Appendix E.3 that PCA successfully reduces the data dimensionality while preserving information. We set d to 12 and 40 for the Gaussian and rectangle images, respectively. We estimate MI on the latent variables with VINE, and compare against neural estimators MINE, MINDE, and VCE. We used a training sample size of 10,000.

Results and analysis. Table 1 compares the performance of the MI estimators across four different levels of MI for the Gaussian and rectangle image datasets, respectively. VINE shows high accuracy across most MI levels. Notably, in the “high” and “very high” MI regimes, standard neural estimators like MINDE and MINE exhibit significant downward bias, while VINE remains more stable. Although the methodology proposed by Butakov et al. [43] successfully utilized non-parametric estimators like KDE and WKL, their efficacy was strictly limited to extremely low-dimensional latent spaces $( d = 4 )$ . In our investigation, KDE and WKL at $d = 1 2$ and $d = 4 0$ are highly inaccurate due to the curse of dimensionality. VINE, however, remains robust to these dimensions. This allows us to utilize robust non-parametric estimates on relatively high-dimensional latent spaces. Despite the quadratic scaling of the number of pairs, VINE remains computationally competitive. In the Gaussian experiment, it outperforms all neural methods in training efficiency. Furthermore, in the rectangle experiment, VINE achieves comparable speed to the fastest neural models.

Table 1: MI Estimation Performance: Gaussian Images (d = 12) vs. Rectangles $( d = 4 0 )$ . Results reported as Mean (± Std). Time indicates the average duration per estimation.
<table><tr><td>Dataset</td><td>Estimator</td><td> $I ( \pmb { X } ; \pmb { Y } ) = 1$ </td><td> $I ( X ; Y ) = 3$ </td><td> $I ( \pmb { X } ; \pmb { Y } ) = 7$ </td><td>Avg. Time</td></tr><tr><td></td><td>MINE</td><td>0.992 (±0.017)</td><td>2.848 (±0.028)</td><td>5.802 (±0.039)</td><td>16.49s</td></tr><tr><td></td><td>InfoNCE</td><td>0.950 (±0.031)</td><td>2.604 (±0.014)</td><td>3.823 (±0.010)</td><td>465.8s</td></tr><tr><td>Gaussian</td><td>VCE</td><td>0.027 (±0.051)</td><td>3.140 (±0.096)</td><td>7.021 (±0.021)</td><td>124.60s</td></tr><tr><td></td><td>VINE (Ours)</td><td>1.089 (±0.055)</td><td>3.540 (±0.217)</td><td>6.325 (±0.412)</td><td>8.82s</td></tr><tr><td></td><td>MINE</td><td>0.874 (±0.043)</td><td>2.601 (±0.066)</td><td>5.727 (±0.163)</td><td>86.16s</td></tr><tr><td></td><td>InfoNCE</td><td>0.860 (±0.040)</td><td>2.427 (±0.057)</td><td>3.576 (±0.122)</td><td>633.21s</td></tr><tr><td></td><td>VCE</td><td>0.100 (±0.171)</td><td>1.282 (±0.107)</td><td>3.747 (±0.169)</td><td>418.35s</td></tr><tr><td>Recles</td><td>VINE (Ours)</td><td>1.100 (±0.199)</td><td>3.390 (±0.254)</td><td>7.883 (±0.340)</td><td>50.82s</td></tr></table>

![](images/11ed232435f696ed646833b5c2818127d5a4ad5a9698e111f5c490850ec0b027.jpg)  
(a)

![](images/334112d7f473cfb5150a1cb57440f6f5edee6995891ea12539352e9ea394f0c9.jpg)  
(b)

![](images/9a0dac1e7cba9456405a9de427af2eefb6cd794bb5504a7ac97bed1e37a83bcc.jpg)  
(c)

![](images/7a43e388c33a906e2d711958abd5e6114c6ee7a1a5d97210ce4e110dda1a4831.jpg)  
(d)  
Figure 4: Visual representation of the generated image datasets: (a-b) Gaussian plates for X and Y , and (c-d) Rectangles for X and Y.

## 5.4 Post-hoc inspection

Experimental setup. We demonstrate the structural transparency of VINE using contour plots of the learned pair copulas. Unlike neural MI estimators, which output a single scalar from a black-box neural net, VINE explicitly decomposes the total MI into a tractable sum of bivariate pair copula terms, directly reflecting the tree-level decomposition of the joint density (Figure 1). This sum allows us to identify pair copulas that contribute zero (independence copulas) and those that highly influence the final MI estimate. The highly influential pair copulas can further be visually inspected by contour plots. To demonstrate this transparency, we simulate Gaussian data $( d _ { X } = d _ { Y } = 8 )$ with spatial dependence between dimensions (as explained in Section 5.1), estimate a simplified D-vine copula, and compare the estimated bivariate copulas to the ground-truth. We also perform an experiment with the marginal transformations $X _ { l } ^ { 3 } , \mathrm { e x p } ( Y _ { l } )$ . Crucially, while these transformations warp the raw scatter plots, the underlying copula structure, and thus the true MI, remains invariant. Visualizations of copulas are projected from $[ 0 , 1 ] ^ { 2 }$ to Gaussian space using the inverse CDF mapping $( \Phi ^ { - 1 } )$ ).

Results. Figure 5 visualizes estimated bivariate copulas next to the ground truths, along with a scatter plot of the raw data. We observe that the contour plots reveal which bivariate copulas contribute to the MI estimate, and which do not (independence copulas). For instance, the contour plot for $\hat { c } ( X _ { 1 } , Y _ { 1 } | X _ { 2 : 8 } )$ in Figure 5a (top right) shows that the bivariate copula between $X _ { 1 }$ and $Y _ { 1 }$ given ${ \dot { X _ { 2 } } } , . . . , { \dot { X _ { d } } }$ has a strong positive correlation, indicating a significant contribution to the total MI. In contrast, the contour plots of ${ \hat { c } } ( X _ { 1 } , Y _ { 2 } | X _ { 2 : 8 } , Y _ { 1 } )$ and ${ \tilde { c } } ( X _ { 1 } , \mathbf { \bar { Y } } _ { 8 } | X _ { 2 : 8 } , Y _ { 1 : 7 } )$ (middle right and bottom right), takes the form of an independence copula with spherical contours, signaling that it contributes zero to the MI estimate. Note that although the scatter plot of $( X _ { 1 } , Y _ { 2 } )$ shows a positive correlation, ${ \hat { c } } ( X _ { 1 } , Y _ { 2 } | X _ { 2 : 8 } , Y _ { 1 } )$ is an independence copula because $X _ { 1 }$ and $Y _ { 2 }$ are conditionally independent given $X _ { 2 } , . . . , X _ { 8 } , Y _ { 1 }$ . Furthermore, since the ground-truth bivariate copulas are known for simulated Gaussian data, we can visually verify that the estimated copulas closely match the ground truth. Crucially, even when non-linear transformations obscure the dependencies in the raw scatter plots (Figure 5b), the estimated bivariate copulas continue to closely match the ground truth. Finally, we note that when the simplifying assumption is violated, the learned pair-copulas represent a structured approximation of the true conditional dependencies rather than an exact recovery [22, 29].

![](images/113769c40b91a100b77b1bf3755172ccd4a233a1e8f16bfe7339132e807f85fd.jpg)  
(a) Gaussian Baseline  
(b) Transformed Data: (X<sup>3</sup>, exp(Y ))  
Figure 5: Visualization of a selection of VINE’s learned pair copulas. VINE trained on Gaussian data of $d _ { X } = d _ { Y } = 8$ (Figure 5a) and transformed Gaussian data X<sup>3</sup>, exp(Y) (Figure 5b). Each subfigure displays a grid comparing the ground-truth pair copula (left column), scatter plots of the raw data (center column), and the estimated pair copula (right column).

## 6 Limitations and future work

While VINE scales well in terms of estimation error as the dimensions grow, it still requires estimating $O ( d ^ { 2 } )$ bivariate copulas sequentially. Addressing this quadratic complexity is a remaining computational challenge. Moreover, the convergence results for the density estimator underlying VINE depend on the well-known and common simplifying assumption. Further work should investigate how severely this assumption can be violated before performance degrades noticeably. Additional directions for future work include investigation of the theoretical properties of VINE.

## 7 Conclusion

In this work, we have presented VINE, a non-parametric D-vine copula-based MI estimator. After deriving an analytical expression for the true MI under the simplified D-vine assumption, we motivate a fully non-parametric approach, arguing that it provides high flexibility to capture diverse dependency structures. Through experiments on a variety of benchmarking datasets, we demonstrated that VINE achieves competitive accuracy to SotA neural methods at a considerably lower computational cost. We also demonstrate that the decompositional structure of VINE allows for inherent transparency through visualizations of dependencies, which is not offered by black-box neural MI estimators.

## Acknowledgments and Disclosure of Funding

The authors would like to thank Arnoldo Frigessi and Nils Lid Hjort for helpful discussions and valuable feedback on this work.

This work was supported by the Research Council of Norway through its Centres of Excellence scheme, Integreat - Norwegian Centre for knowledge-driven machine learning under project number 332645 and FRIPRO Grant under project number 356103.

The computations were performed on resources provided by the Educloud Research infrastructure at the University of Oslo.

## References

[1] Claude Elwood Shannon. A mathematical theory of communication. The Bell system technical journal, 27(3):379–423, 1948.

[2] Adam A Margolin, Ilya Nemenman, Katia Basso, Chris Wiggins, Gustavo Stolovitzky, Riccardo Dalla Favera, and Andrea Califano. ARACNE: an algorithm for the reconstruction of gene regulatory networks in a mammalian cellular context. BMC bioinformatics, 7(Suppl 1):S7, 2006.

[3] Hanchuan Peng, Fuhui Long, and Chris Ding. Feature selection based on mutual information criteria of max-dependency, max-relevance, and min-redundancy. IEEE Transactions on pattern analysis and machine intelligence, 27(8):1226–1238, 2005.

[4] Naftali Tishby, Fernando C Pereira, and William Bialek. The information bottleneck method. arXiv preprint physics/0004057, 2000.

[5] Young-Il Moon, Balaji Rajagopalan, and Upmanu Lall. Estimation of mutual information using kernel density estimators. Physical Review E, 52(3):2318, 1995.

[6] Alexander Kraskov, Harald Stögbauer, and Peter Grassberger. Estimating mutual information. Physical Review E—Statistical, Nonlinear, and Soft Matter Physics, 69(6):066138, 2004.

[7] Ziv Goldfeld, Kristjan Greenewald, Jonathan Niles-Weed, and Yury Polyanskiy. Convergence of smoothed empirical measures with applications to entropy estimation. IEEE Transactions on Information Theory, 66(7):4368–4391, 2020.

[8] David McAllester and Karl Stratos. Formal limitations on the measurement of mutual information. In International Conference on Artificial Intelligence and Statistics, pages 875–884. PMLR, 2020.

[9] Bao Duong and Thin Nguyen. Diffeomorphic information neural estimation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 7468–7475, 2023.

[10] Pengyu Cheng, Weituo Hao, Shuyang Dai, Jiachang Liu, Zhe Gan, and Lawrence Carin. Club: A contrastive log-ratio upper bound of mutual information. In International conference on machine learning, pages 1779–1788. PMLR, 2020.

[11] Qing Guo, Junya Chen, Dong Wang, Yuewei Yang, Xinwei Deng, Jing Huang, Larry Carin, Fan Li, and Chenyang Tao. Tight mutual information estimation with contrastive fenchel-legendre optimization. Advances in Neural Information Processing Systems, 35:28319–28334, 2022.

[12] Nunzio A Letizia, Nicola Novello, and Andrea M Tonello. Mutual information estimation via f-divergence and data derangements. Advances in Neural Information Processing Systems, 37: 105114–105150, 2024.

[13] Jiaming Song and Stefano Ermon. Understanding the limitations of variational mutual information estimators. arXiv preprint arXiv:1910.06222, 2019.

[14] Ivan Butakov, Alexander Tolmachev, Sofia Malanchuk, Anna Neopryatnaya, and Alexey Frolov. Mutual information estimation via normalizing flows. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

[15] Mohamed Ishmael Belghazi, Aristide Baratin, Sai Rajeshwar, Sherjil Ozair, Yoshua Bengio, Aaron Courville, and Devon Hjelm. Mutual information neural estimation. In International conference on machine learning, pages 531–540. PMLR, 2018.

[16] Yanzhi Chen, Zijing Ou, Adrian Weller, and Michael Gutmann. Neural mutual information estimation with vector copulas. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, editors, Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 44803–44823. Curran Associates, Inc., 2025. doi: 10. 52202/085713-1492.

[17] Giulio Franzese, Mustapha BOUNOUA, and Pietro Michiardi. MINDE: Mutual information neural diffusion estimation. In B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun, editors, International Conference on Learning Representations, volume 2024, pages 16685–16716, 2024.

[18] Harry Joe. Families of m-variate distributions with given margins and m(m − 1)/2 bivariate dependence parameters. In Institute of Mathematical Statistics Lecture Notes - Monograph Series, pages 120–141. Institute of Mathematical Statistics, 1996. ISBN 978-0-940600-40-9.

[19] Tim Bedford and Roger M. Cooke. Probability density decomposition for conditionally dependent random variables modeled by vines. Annals ofMathematics and Artificial Intelligence, 32: 245–268, 2001. doi: 10.1023/A:1016725902970.

[20] Tim Bedford and Roger M. Cooke. Vines — a new graphical model for dependent random variables. The Annals of Statistics, 30(4), 2002. doi: 10.1214/aos/1031689016.

[21] Kjersti Aas, Claudia Czado, Arnoldo Frigessi, and Henrik Bakken. Pair-copula constructions of multiple dependence. Insurance: Mathematics and Economics, 44(2):182–198, 2009. doi: 10.1016/j.insmatheco.2007.02.001.

[22] Thomas Nagler and Claudia Czado. Evading the curse of dimensionality in nonparametric density estimation with simplified vine copulas. Journal of Multivariate Analysis, 151:69–89, 2016. doi: 10.1016/j.jmva.2016.07.003.

[23] Abe Sklar. Fonctions de répartition à n dimensions et leurs marges. In Publications de l’Institut de Statistique de L’Université de Paris, volume 8, pages 229–231, 1959.

[24] Ingrid Hobæk Haff, Kjersti Aas, and Arnoldo Frigessi. On the simplified pair-copula construction — simply useful or too simplistic? Journal ofMultivariate Analysis, 101(5):1296–1310, 2010. doi: 10.1016/j.jmva.2009.12.001.

[25] Jakob Stöber, Harry Joe, and Claudia Czado. Simplified pair copula constructions — limitations and extensions. Journal ofMultivariate Analysis, 119:101–118, 2013. doi: 10.1016/j.jmva. 2013.04.014.

[26] Thomas Nagler. Simplified vine copula models: State of science and affairs. Risk Sciences, 1: 100022, 2025. doi: 10.1016/j.risk.2025.100022.

[27] Claudia Czado and Thomas Nagler. Vine copula based modeling. Annual Review of Statistics and Its Application, 9(1):453–477, March 2022. ISSN 2326-831X. doi: 10.1146/ annurev-statistics-040220-101153.

[28] Matthias Killiches, Daniel Kraus, and Claudia Czado. Examination and visualisation of the simplifying assumption for vine copulas in three dimensions. Australian & New Zealand Journal ofStatistics, 59(1):95–117, 2017. doi: 10.1111/anzs.12182.

[29] Fabian Spanhel and Malte S. Kurz. Simplified vine copula models: Approximations based on the simplifying assumption. Electronic Journal ofStatistics, 13(1), 2019. doi: 10.1214/19-EJS1547.

[30] David Huk, Yuanhe Zhang, Mark Steel, and Ritabrata Dutta. Quasi-bayes meets vines. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 40359–40392. Curran Associates, Inc., 2024. doi: 10.52202/079017-1277.

[31] Junxi Xiao and Qinliang Su. Copula-SVI: Vine-copula variational inference with stein refining for instance-level correlation capturing. In Forty-third International Conference on Machine Learning, 2026.

[32] Natasa Tagasovska, Damien Ackerer, and Thibault Vatter. Copulas as high-dimensional generative models: Vine copula autoencoders. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.

[33] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.

[34] Akash Srivastava, Seungwook Han, Kai Xu, Benjamin Rhodes, and Michael U Gutmann. Estimating the density ratio between distributions with high discrepancy using multinomial logistic regression. arXiv preprint arXiv:2305.00869, 2023.

[35] Houman Safaai, Arno Onken, Christopher D Harvey, and Stefano Panzeri. Information estimation using nonparametric copulas. Physical Review E, 98(5):053302, 2018.

[36] Jeffrey F Dißmann. Statistical inference for regular vines and application. Master’s thesis, Technische Universität München, 2010.

[37] Oswaldo Morales-Nápoles. Counting Vines. In Dorota Kurowicka and Harry Joe, editors, Dependence Modeling: Vine Copula Handbook, pages 189–218. World Scientific Publishing Co. Pte. Ltd., 2011. ISBN 978-981-4299-87-9.

[38] Claudia Czado. Analyzing Dependent Data with Vine Copulas: A Practical Guide With R. Number Volume 222 in Lecture Notes in Statistics. Springer Cham, 1 edition, 2019. ISBN 978-3-030-13784-7.

[39] Thomas Nagler, Christian Bumann, and Claudia Czado. Model selection in sparse highdimensional vine copula models with an application to portfolio risk. Journal ofMultivariate Analysis, 172:180–192, 2019. doi: 10.1016/j.jmva.2019.03.004.

[40] Thomas Nagler and Thibault Vatter. pyvinecopulib: A Python library for vine copula models, 2025.

[41] Charles J Stone. Optimal rates of convergence for nonparametric estimators. The annals of Statistics, pages 1348–1360, 1980.

[42] Paweł Czyz, Frederic Grabowski, Julia Vogt, Niko Beerenwinkel, and Alexander Marx. Beyond˙ normal: On the evaluation of mutual information estimators. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 16957–16990. Curran Associates, Inc., 2023.

[43] Ivan Butakov, Alexander Tolmachev, Sofia Malanchuk, Anna Neopryatnaya, Alexey Frolov, and Kirill Andreev. Information bottleneck analysis of deep neural networks via lossy compression. In International Conference on Learning Representations (ICLR), 2024.

[44] Gery Geenens, Arthur Charpentier, and Davy Paindaveine. Probit transformation for nonparametric kernel estimation of the copula density. Bernoulli, 23(3):1848–1873, 2017. doi: 10.3150/15-BEJ798.

1 Introduction 1   
2 Preliminaries 2   
2.1 Vine copulas . 2   
2.2 Mutual information 3   
3 VINE mutual information estimator 3   
4 Theoretical motivation 4   
5 Numerical experiments 5   
5.1 Evaluating the simplifying assumption . 5   
5.2 Non-Gaussian datasets 6   
5.3 Image dataset with known MI 7   
5.4 Post-hoc inspection . 8   
6 Limitations and future work 9   
7 Conclusion 9   
A Notation and full preliminaries 15   
A.1 Additional details on vine copulas 15   
A.2 MI estimation 17   
A.3 MI estimation with copulas . 17   
B Proof of Proposition 3.1 18   
C Computational and Scaling Limitations of VINE 20   
D Regularization via the threshold parameter 21   
E Experiment details 21   
E.1 Implementation setup . 21   
E.2 Transformed Gaussian datasets . 22   
E.3 Image dataset 23

## A Notation and full preliminaries

Notation. Throughout this work, we use the lowercase letter $f$ to denote probability density functions (PDFs) and the uppercase letter F to denote cumulative distribution functions (CDFs). Subscripts specify associated variables the functions refer to. Capital letters, such as $X _ { l } , Y _ { k }$ , denote univariate random variables. Bold capital letters, such as X, Y denote random vectors, while bold and lowercase letters, such as x, y, denote deterministic vectors.

## A.1 Additional details on vine copulas

This summary is based on [38]. Let $U _ { 1 } , \dots , U _ { d } \sim \mathrm { U n i f o r m } ( 0 , 1 )$ . We define the copula $C$ as their joint CDF,

$$
C ( u _ { 1 } , \dots , u _ { d } ) = \operatorname* { P r } ( U _ { 1 } \leq u _ { 1 } , \dots , U _ { d } \leq u _ { d } ) .
$$

For absolutely continuous variables, we obtain the copula density by partial differentiation, $c ( u _ { 1 } , \dots , u _ { d } ) = \partial ^ { d } C ( u _ { 1 } , \dots , u _ { d } ) / ( \partial u _ { 1 } \cdot \cdot \cdot \partial u _ { d } )$ . For bivariate random variables, there are many such copula functions [38, Chapter 3]. The simplest being the independence copula, which corresponds to $c ( F _ { 1 } ( Z _ { 1 } ) , F _ { 2 } ( Z _ { 2 } ) ) = \mathrm { \bar { 1 } } , \mathrm { i . e . , } Z _ { 1 } \perp Z _ { 2 }$ . Other popular choices include the Gaussian copula, which models the dependence between $\Phi ^ { - 1 } ( F _ { 1 } ( Z _ { 1 } ) )$ and $\Phi ^ { - 1 } ( F _ { 2 } ( Z _ { 2 } ) )$ as linear, where Φ is the standard Gaussian CDF. We also have Archimedean copulas, among those the Clayton copula, which can capture tail dependence. In higher dimensions, one can construct highly flexible dependence structures by combining such bivariate building blocks.

Vine copulas are a class of pair copula constructions, which are constructed by first decomposing the joint PDF into a product of univariate conditional PDFs,

$$
f ( z _ { 1 } , \dots , z _ { d } ) = f _ { 1 } ( z _ { 1 } ) \prod _ { j = 2 } ^ { d } f _ { j | 1 , \dots , j - 1 } ( z _ { j } | z _ { 1 } , \dots , z _ { j - 1 } ) .
$$

The idea is that we can express each of these univariate conditional PDFs in terms of bivariate copulas, i.e., pair copulas. We demonstrate this with a 3-dimensional example. The joint PDF f of $( Z _ { 1 } , Z _ { 2 } , Z _ { 3 } )$ can be decomposed as follows<sup>6</sup>

$$
f ( z _ { 1 } , z _ { 2 } , z _ { 3 } ) = f _ { 1 } ( z _ { 1 } ) f _ { 2 | 1 } ( z _ { 2 } | z _ { 1 } ) f _ { 3 | 1 , 2 } ( z _ { 3 } | z _ { 1 } , z _ { 2 } ) .
$$

For the first term, we get

$$
\begin{array} { l } { { f _ { 2 | 1 } ( z _ { 2 } | z _ { 1 } ) = \displaystyle \frac { f _ { 1 2 } ( z _ { 1 } , z _ { 2 } ) } { f _ { 1 } ( z _ { 1 } ) } \qquad } } \\ { { \qquad = \displaystyle \frac { f _ { 1 } ( z _ { 1 } ) f _ { 2 } ( z _ { 2 } ) c _ { 1 2 } ( F _ { 1 } ( z _ { 1 } ) , F _ { 2 } ( z _ { 2 } ) } { f _ { 1 } ( z _ { 1 } ) } \qquad } } \\ { { \qquad = f _ { 2 } ( z _ { 2 } ) c _ { 1 2 } ( F _ { 1 } ( z _ { 1 } ) , F _ { 2 } ( z _ { 2 } ) . } } \end{array}
$$

Further, for the second term, we get

$$
\begin{array} { r l } & { f _ { 3 | 1 , 2 } ( z _ { 3 } | z _ { 1 } , z _ { 2 } ) = \frac { f _ { 1 , 3 | 2 } ( z _ { 2 } , z _ { 3 } | z _ { 1 } ) } { f _ { 1 | 2 } ( z _ { 2 } | z _ { 1 } ) } } \\ & { \phantom { f _ { 3 | 1 , 2 } } = \frac { f _ { 1 | 2 } ( z _ { 1 } | z _ { 2 } ) f _ { 3 | 2 } ( z _ { 2 } | z _ { 2 } ) c _ { 1 , 3 ; 2 } ( F _ { 1 | 2 } ( z _ { 1 } | z _ { 2 } ) , F _ { 3 | 2 } ( z _ { 3 } | z _ { 2 } ) ) } { f _ { 1 | 2 } ( z _ { 1 } | z _ { 2 } ) } } \\ & { \phantom { f _ { 3 | 1 , 2 } } = f _ { 3 | 2 } ( z _ { 2 } | z _ { 2 } ) c _ { 1 , 3 ; 2 } ( F _ { 1 | 2 } ( z _ { 1 } | z _ { 2 } ) , F _ { 3 | 2 } ( z _ { 3 } | z _ { 2 } ) ) } \\ & { \phantom { f _ { 3 | 2 } ( z _ { 3 } | z _ { 2 } ) } = f _ { 3 } ( z _ { 2 } , 0 , F _ { 2 } ( z _ { 2 } ) , F _ { 3 } ( z _ { 3 } ) ) c _ { 1 , 3 ; 2 } ( F _ { 1 | 2 } ( z _ { 1 } | z _ { 2 } ) , F _ { 3 | 2 } ( z _ { 3 } | z _ { 2 } ) ) . } \end{array}
$$

Combining all terms gives us the pair copula decomposition

$$
\begin{array} { r l } & { f ( z _ { 1 } , z _ { 2 } , z _ { 3 } ) = f _ { 1 } ( z _ { 1 } ) \times f _ { 2 } ( z _ { 2 } ) \times f _ { 3 } ( z _ { 3 } ) } \\ & { \phantom { f _ { 1 } ( z _ { 1 } , z _ { 2 } , z _ { 3 } ) = } \times c _ { 1 , 2 } ( F _ { 1 } ( z _ { 1 } ) , F _ { 2 } ( z _ { 2 } ) ) \times c _ { 2 , 3 } ( F _ { 2 } ( z _ { 2 } ) , F _ { 3 } ( z _ { 3 } ) ) } \\ & { \phantom { f _ { 1 } ( z _ { 1 } , z _ { 2 } , z _ { 3 } ) = } \times c _ { 1 , 3 ; 2 } ( F _ { 1 | 2 } ( z _ { 1 } | z _ { 2 } ) , F _ { 3 | 2 } ( z _ { 3 } | z _ { 2 } ) ) . } \end{array}
$$

A vine copula on d variables is constructed in a similar manner, such that the pair copulas can be organized into a hierarchy of trees $\mathcal { T } = ( T _ { 1 } , \dots , T _ { d - 1 } )$ , called an R-vine tree sequence:

Definition A.1 (R-vine tree sequence). Let $\mathcal { T } = ( T _ { 1 } , \dots , T _ { d - 1 } )$ be a sequence of trees. Then $\tau$ is said to be an R-vine tree sequence if the following holds:

1. Each tree $T _ { t }$ consists of a node set $N _ { t }$ and an edge set $E _ { t } , t = 1 , \dots , d - 1$

2. The edges in tree t becomes nodes in tree t + 1, i.e., $N _ { t } = E _ { t - 1 }$ for $t = 2 , \ldots , d - 1$

3. An edge in tree $t \geq 2$ must satisfy the proximity condition: two edges in $T _ { t }$ can only form an edge in tree $T _ { t + 1 }$ if they share all variables except for one each.

In tree $T _ { t }$ , an edge $e \in E _ { t }$ represents a pair copula, and in the first trees these model the dependencies between unconditional pairs of variables, while in trees after the first they model dependencies between pairs of variables conditional on others. The proximity condition of Definition A.1 ensures that all conditional CDFs, being arguments to pair copulas in trees after the first, can be obtained by partially differentiating pair copulas in the previous trees. In particular, we can use the following result from Joe [18]:

Theorem A.2 (Recursive definition of conditional CDFs). Suppose $( Z _ { 1 } , \ldots , Z _ { d } )$ is a random vector, and consider any pair $j , k \in \{ 1 , \ldots , d \}$ with $j \neq k ,$ and subset $D \subseteq \{ 1 , \dots , d \} \setminus \{ j , k \}$ . Then, the following holds:

$$
F _ { j | k , D } ( z _ { j } | \boldsymbol { z } _ { k } , \boldsymbol { z } _ { D } ) = \frac { \partial C _ { j , k ; D } ( F _ { j | D } ( z _ { j | \boldsymbol { z } _ { D } } ) , F _ { k | D } ( z _ { k | \boldsymbol { z } _ { D } } ) } { \partial F _ { k | D } ( z _ { k } | \boldsymbol { z } _ { D } ) } .
$$

This property is utilized in the sequential estimation and selection algorithm for parametric vines by Dißmann [36]. That is, the trees are estimated top-down, and estimates from earlier trees are used to generate pseudo-observations for the estimation of the next tree. The same principle also applies to the non-parametric estimator proposed by Nagler and Czado [22], except that the estimated conditional CDFs are computed from the estimated non-parametric pair copulas.

There are three classes of vine copulas that satisfy Definition A.1. We have visualized two of these classes for $d = 4$ variables in Figure 6. Figure 6a shows an example of a D-vine tree sequence. These are characterized by trees where each node is connected to at most two other nodes. This is also the vine tree sequence used in our proposed method. Figure 6b shows an example of a C-vine. In a C-vine, each tree has a root note, i.e., a node that is fully connected. The last class of vine copulas is the general R-vines, encompassing all vine copulas, including D-vines and C-vines.

![](images/2bf8c753758ad84f934326bb27a46bbe31de96f9a9675cf07f92b0d7926ee6ed.jpg)  
Figure 6: Examples of R-vine tree sequences on $d = 4$ variables.

For a D-vine on d variables, the full joint PDF can be written as follows:

$$
f = \prod _ { l = 1 } ^ { d - 1 } f _ { l } \times \prod _ { j = 1 } ^ { d - 1 } \prod _ { k = 1 } ^ { d - j } c _ { k , j + k ; k + 1 , \ldots , j + k - 1 } ( F _ { k | k + 1 , \ldots , j + k - 1 } , F _ { j + k | k + 1 , \ldots , j + k - 1 } ) ,
$$

where we have omitted the function arguments of the marginal PDFs and the CDFs for notational convenience.

The simplifying assumption In Equation (VinePDF), we see that the pair copula $c _ { j _ { e } , k _ { e } ; D _ { e } }$ depends on the conditioning value $z _ { D _ { e } }$ through its arguments, but we have omitted the direct dependence on the function itself. Generally, it holds that $F _ { j _ { e } | D _ { e } } ^ { - } ( Z _ { j _ { e } } | Z _ { D _ { e } } ) \perp Z _ { D _ { e } }$ and $F _ { k _ { e } | D _ { e } } ( Z _ { k _ { e } } | \bar { Z _ { D _ { e } } } ) \perp Z _ { D _ { e } }$ This means that all marginal effects of $Z _ { j _ { e } } | Z _ { D _ { e } } = z _ { D _ { e } }$ and $Z _ { k _ { e } } | Z _ { D _ { e } } = z _ { D _ { e } }$ can be captured through the conditional CDFs $F _ { j _ { e } | D _ { \epsilon } }$ and $F _ { k _ { e } | D _ { e } }$ , respectively, for all values of $z _ { D _ { e } }$ . On the other hand, it does not generally hold that $( F _ { j _ { e } | D _ { e } } ( \bar { Z } _ { j _ { e } } | Z _ { D _ { e } } ) , F _ { k _ { e } | D _ { e } } ( Z _ { k _ { e } } | Z _ { D _ { e } } ) ) \perp Z _ { D _ { e } }$ , meaning that the dependence between $Z _ { j _ { e } }$ and $Z _ { k _ { e } }$ given $Z _ { D _ { e } } ~ = ~ z _ { D _ { i } }$ can vary with the conditioning value $z _ { D _ { e } }$ , in which case $c _ { j _ { e } , k _ { e } ; D _ { e } }$ would also depend on $z _ { D _ { e } }$ . An important assumption for making the estimation of vine copulas tractable in higher dimensions is that $c _ { j _ { e } , j k _ { e } ; D _ { e } } ( \cdot , \cdot ; z _ { D _ { e } } ) = c _ { j _ { e } , k _ { e } ; D _ { e } } ( \cdot , \cdot )$ for all possible values of $z _ { D _ { \epsilon } }$ and all $e \in E _ { k } , k = 2 , \ldots , d - 1$ This assumption is known in the vine copula literature as the simplifying assumption. Discussions of this assumption can be found in [24–26]. While there are distributions that can be represented by simplified vines, such as the multivariate Gaussian and t distributions [25], the simplifying assumption is generally an approximation, a limitation one must be aware of when using such models.

## A.2 MI estimation

MI is a dependency measure able to capture non-linear relationships between random variables. Since it is a function of the joint distribution, it can be estimated from a finite dataset $\{ ( \pmb { x } ^ { ( i ) } , \pmb { y } ^ { ( i ) } ) \} _ { i = 1 } ^ { n }$

The earliest approaches to estimating MI were through direct estimation of the joint distribution. These approaches include histogram-based binning, KDE [5], and k-NN [6], all of which are non-parametric methods. Although flexible, all these approaches suffer from the curse of dimensionality, where the sample size required to maintain estimation accuracy grows exponentially with the dimension [8].

To beat the curse of dimensionality faced by non-parametric estimators, recent research has seen the rise of neural MI estimators. Two main strategies are density estimation via deep generative models and density ratio estimation. Deep generative models, such as normalizing flows and variational autoencoders, attempt to learn an explicit parameterization of the underlying densities [8, 10, 13, 17]. The density ratio estimation approach trains a neural net to separate samples from the joint distribution $f$ against samples from the product of marginals $f _ { X } f _ { Y }$ . This framework includes the Mutual Information Neural Estimator (MINE), which optimizes the Donsker-Varadhan lower bound, and InfoNCE, which utilizes a contrastive objective where the estimate is notably bounded by the logarithm of the batch size [15, 33]. The Smoothing Mutual Information Lower-bound Estimator (SMILE) was proposed to mitigate the high variance often encountered in these bounds by clipping the density ratio [13]. Approaches such as MIENF simplify high-dimensional distributions before applying classical estimators such as KDE [14].

An alternative perspective involves isolating the dependency structure from the marginals using copula theory. Safaai et al. [35] utilized this for a non-parametric copula-based estimator using direct kernel estimation, but only considered the 2-dimensional setting. More recently, the Vine Copula Estimator (VCE) framework [16] extended this concept to high-dimensional vectors by assuming a simplified vector copula structure, where the marginals are learned using normalizing flows and the vector copula assumes a mixture of Gaussian structure.

## A.3 MI estimation with copulas

Generally, the MI of random vectors $\pmb { X } \in \mathbb { R } ^ { d _ { X } }$ and $\pmb { Y } \in \mathbb { R } ^ { d _ { Y } }$ can be expressed in terms of an interdependence copula density c and intradependence copula densities $c _ { X }$ and $c _ { Y }$ . Let f be the joint PDF of $( { \dot { \boldsymbol { X } } } , { \boldsymbol { Y } } )$ , and let $f _ { 1 } , \dots , f _ { d _ { X } }$ denote the marginal PDFs of $X _ { 1 } , \dots , X _ { d _ { X } }$ and $f _ { d _ { X } + 1 } , \ldots , f _ { d }$ the marginal PDFs of $Y _ { 1 } , \dots , Y _ { d _ { Y } }$ Then, by Theorem 2.1 we know that $\begin{array} { r } { f ( { \pmb x } , { \pmb y } ) = \prod _ { l = 1 } ^ { d _ { { \pmb x } } } f _ { l } ( { \pmb x } _ { l } ) \times \prod _ { l = 1 } ^ { d _ { { \pmb x } } } f _ { d _ { { \pmb x } + l } ( { \pmb y } _ { l } ) \times { \pmb c } ( { \pmb u } _ { X } , { \pmb u } _ { Y } ) , ~ f ( { \pmb x } ) } = \prod _ { l = 1 } ^ { d _ { { \pmb x } } } f _ { l } ( { \pmb x } _ { l } ) \times { \pmb c } _ { X } ( { \pmb u } _ { x } ) } \end{array}$ , and $\begin{array} { r } { f _ { Y } ( \pmb { y } ) = \prod _ { l = 1 } ^ { d _ { Y } } f _ { d _ { X } + l } ( y _ { l } ) \times c _ { Y } ( \pmb { u } _ { Y } ) } \end{array}$ , where we have let ${ \pmb u } _ { X } ~ = ~ ( F _ { 1 } ( x _ { 1 } ) , \dots , F _ { d _ { X } } ( x _ { d _ { X } } ) )$ and ${ \pmb u } _ { Y } = ( F _ { d _ { X } + 1 } \bar { ( } y _ { 1 } ) , \dots , F _ { d } ( y _ { d _ { Y } } ) )$ . Thus, the density ratio can be expressed as

$$
\frac { f ( { \pmb x } , { \pmb y } ) } { f _ { X } ( { \pmb x } ) f _ { Y } ( { \pmb y } ) } = \frac { c ( { \pmb u } _ { X } , { \pmb u } _ { Y } ) } { c _ { X } ( { \pmb u } _ { X } ) c _ { Y } ( { \pmb u } _ { Y } ) } .
$$

In Appendix B we prove Proposition 3.1, which states how this expression simplifies under a simplified D-vine copula assumption.

## B Proof of Proposition 3.1

Proof. To derive the MI of a D-vine, we need the joint copula density c of $( X , Y )$ , as well as the marginal copula densities $c _ { X }$ and $c _ { Y }$ of X and Y , respectively. We have

$$
\begin{array} { r l } &  \begin{array} { r l } & { \mathcal { E } _ { \xi } ( t ) \cdot \xi _ { 1 } = \frac { \mathcal { E } _ { \xi } ( t ) } { \mathcal { E } _ { \xi } ( t ) } \{ \mathbf { E } _ { \xi } ( t ) - \mathbf { E } _ { \xi } ( t ) \{ \mathbf { E } _ { \xi } ( t ) - \mathbf { E } _ { \xi } ( t ) \{ \mathbf { E } _ { \xi } ( t ) \} \} } \\ & { - \mathrm { e } _ { \xi } ( t ) \cdot \xi _ { 2 } = \frac { \mathcal { E } _ { \xi } ( t ) } { \mathcal { E } _ { \xi } ( t ) } \{ \mathbf { E } _ { \xi } ( t ) - \mathbf { E } _ { \xi } ( t ) - \mathbf { E } _ { \xi } ( t ) \{ \mathbf { E } _ { \xi } ( t ) - \mathbf { E } _ { \xi } ( t ) \} \} } \\ & { - \mathrm { e } _ { \xi } ( t ) \cdot \xi _ { 1 } = \mathrm { e } _ { \xi } \{ \mathbf { E } _ { \xi } ( t ) - \mathbf { E } _ { \xi } ( t ) \{ \mathbf { E } _ { \xi } ( t ) - \mathbf { E } _ { \xi } ( t ) \{ \mathbf { E } _ { \xi } ( t ) - \mathbf { E } _ { \xi } ( t ) \{ \mathbf { E } _ { \xi } ( t ) - \mathbf { E } _ { \xi } ( t ) \} \} } \\ & { - \mathrm { e } _ { \xi } ( t ) \cdot \mathrm { e } _ { \xi } \{ \mathbf { E } _ { \xi } ( t ) - \mathbf { E } _ { \xi } ( t ) \{ \mathbf { E } _ { \xi } ( t ) - \mathbf { E } _ { \xi } ( t ) \{ \mathbf { E } _ { \xi } ( t ) - \mathbf { E } _ { \xi } ( t ) \{ \mathbf { E } _ { \xi } ( t ) - \mathbf { E } _ { \xi } ( t ) \{ \mathbf { E } _ { \xi } ( t ) - \mathbf { E } _ { \xi } ( t ) \} \} } \\ &  \mathrm { e } _ { \xi } \{ \mathbf { E } _ { \xi } ( t ) -  \end{array} \end{array}
$$

where the first two double products correspond to the orange and blue area, respectively, of Figure 1, and the last three double products correspond to the green area, given by the edges in $\mathbb { E } _ { X Y }$ . Grouping the last four double products, we may re-write the joint copula density as

$$
\begin{array} { r l } & { \quad c ( F _ { 1 } ( z _ { 1 } ) , \ldots , F _ { d _ { X } } ( z _ { d _ { X } } ) , F _ { d _ { X } + 1 } ( z _ { d _ { X } + 1 } ) , \ldots , F _ { d } ( z _ { d } ) ) } \\ & { = \displaystyle \prod _ { t = 1 } \ \prod _ { i = 1 } ^ { d _ { X } - t } c _ { i , i + t ; i + 1 , \ldots , i + t - 1 } ( F _ { i | i + 1 , \ldots , i + t - 1 } ( z _ { i } | z _ { i + 1 } , \ldots , z _ { i + t - 1 } ) , } \\ & { \qquad F _ { i + t | i + 1 , \ldots , i + t - 1 } ( z _ { i + t } | z _ { i + 1 } , \ldots , z _ { i + t - 1 } ) ) } \\ & { \cdot \displaystyle \prod _ { j = d _ { X } + 1 } ^ { d } \prod _ { l = 1 } ^ { j - 2 } c _ { l , j ; l + 1 , \ldots , j - 1 } ( F _ { l | l + 1 , \ldots , j - 1 } ( z _ { l } | z _ { l + 1 } , \ldots , z _ { j - 1 } ) , } \\ & { \qquad F _ { j | l + 1 , \ldots , j - 1 } ( z _ { j } | z _ { l + 1 } , \ldots , z _ { j - 1 } ) ) \cdot c _ { j - 1 , j } ( F _ { j - 1 } ( z _ { j - 1 } , F _ { j } ( z _ { j } ) ) ) . } \end{array}
$$

Using Sklar’s theorem, we have

$$
\begin{array} { r l } & { \quad c _ { j - 1 , j } ( F _ { j - 1 } ( z _ { j - 1 } ) , F _ { j } ( z _ { j } ) ) ) = \frac { f _ { j - 1 , j } ( z _ { j - 1 } , z _ { j } ) } { f _ { j - 1 } ( z _ { j - 1 } ) f _ { j } ( z _ { j } ) } = \frac { f _ { j | j - 1 } ( z _ { j } | z _ { j - 1 } ) } { f _ { j } ( z _ { j } ) } } \\ & { c _ { l , j ; l + 1 , \dots , j - 1 } ( F _ { l | l + 1 , \dots , j - 1 } ( z _ { l } | z _ { l + 1 } , \dots , z _ { j - 1 } ) , F _ { j | l + 1 , \dots , j - 1 } ( z _ { j } | z _ { l + 1 } , \dots , z _ { j - 1 } ) ) } \\ & { = \frac { f _ { l , j | l + 1 , \dots , j - 1 } ( z _ { l } , z _ { j } | z _ { l + 1 } , \dots , z _ { j - 1 } ) } { f _ { l | l + 1 , \dots , j - 1 } ( z _ { l } | z _ { l + 1 } , \dots , z _ { j - 1 } ) f _ { j | l + 1 , \dots , j - 1 } ( z _ { j } | z _ { l + 1 } , \dots , z _ { j - 1 } ) } } \\ & { = \frac { f _ { j | l , \dots , j - 1 } ( z _ { j } | z _ { l } , \dots , z _ { j - 1 } ) } { f _ { j | l + 1 , \dots , j - 1 } ( z _ { j } | z _ { l + 1 } , \dots , z _ { j - 1 } ) } } \end{array}
$$

Omitting arguments for simplicity, we obtain

$$
\prod _ { l = 1 } ^ { j - 2 } c _ { l , j ; l + 1 , \dots , j - 1 } \cdot c _ { j - 1 , j } = { \frac { f _ { j | 1 , \dots , j - 1 } } { f _ { j | 2 , \dots , j - 1 } } } \cdot { \frac { f _ { j | 2 , \dots , j - 1 } } { f _ { j | 3 , \dots , j - 1 } } } \cdot \dots \cdot { \frac { f _ { j | j - 2 , j - 1 } } { f _ { j | j - 1 } } } \cdot { \frac { f _ { j | j - 1 } } { f _ { j } } } = { \frac { f _ { j | 1 , \dots , j - 1 } } { f _ { j } } } .
$$

Plugging this into the expression for the joint D-vine copula density, we get

$$
\begin{array} { l } { { { \scriptstyle c \displaystyle \sum _ { i = 1 } ^ { d } ( { \scriptstyle { \cal Z } _ { i } ( { \scriptstyle { \bf 2 } _ { 1 } } ) , \ldots , { \scriptstyle { \cal F } _ { d } ( { \scriptstyle { \bf 2 } _ { d } } ) , \ldots , { \scriptstyle { \cal F } _ { d } ( { \scriptstyle { \cal Z } _ { d + 1 } } ) , \ldots , { \scriptstyle { \cal F } _ { d } ( { \scriptstyle { \cal Z } _ { d + 1 } } ) } ) , } } } \ } } } \\  { { \scriptstyle d = \displaystyle \prod _ { i = 1 } ^ { d - 1 } \ \ \prod _ { i = 1 } ^ { d - 1 } \ \ ( { \scriptstyle { \cal Z } _ { i , i + 1 + 1 + 1 , \ldots , i + 1 } ( { \scriptstyle { \cal T } _ { i }  { \scriptstyle { \bf 2 } _ { 1 } }  + 1 , \ldots , i + 1 } ( { \scriptstyle { \cal Z } _ { i }  { \scriptstyle { \bf 2 } _ { 1 } } ) , \ldots , { \scriptstyle { \cal Z } _ { d + 1 } } ) , } } \ ) } } \\ { { \scriptstyle \qquad \ } } \\ { { \scriptstyle \qquad \ \sum _ { i \neq 1 } ^ { d } \ \ ( { \scriptstyle { \frac { d } { d } } } ) ( { \scriptstyle { \bf 2 } _ { 1 } } ) ( { \scriptstyle { \cal T } _ { i + 1 + 1 , \ldots , i + 1 } ^ { d } ( { \scriptstyle { \cal Z } _ { i }  { \scriptstyle { \bf 2 } _ { 1 } } ) , \ldots , { \scriptstyle { \cal T } _ { i }  { \scriptstyle { \cal Z } _ { i } }  } } } ) } } \\   \scriptstyle \qquad \sum _ { i = 2 + 3 + 1 } ^ { d } \ \ \frac { \displaystyle f _ { i } ( { \scriptstyle { \bf 2 } _ { 1 } } ) \ ( { \scriptstyle { \bf 2 } _ { 2 } } ) ( { \scriptstyle { \bf 2 } _ { 1 } } ) \ , \ldots , { \scriptstyle { \cal Z } _ { i - 1 } } ) }  \displaystyle f _ { j } (  \scriptstyle { \cal Z } _ { j } \end{array}
$$

The marginal copula density of X is then given by

$$
\begin{array} { r l } & { v _ { \alpha \times } ( u ) } \\ & { = \int \epsilon ( u _ { \alpha } , u _ { \beta } ) \mathrm { d } u _ { \mathcal { X } } } \\ & { = \int \epsilon ( \kappa ( \boldsymbol { \mathrm { I } } _ { \beta } \boldsymbol { \mathrm { I } } _ { \beta } ) ) \mathrm { d } u _ { \mathcal { X } } } \\ & { = \int \epsilon ( \kappa ( \boldsymbol { \mathrm { I } } _ { \beta } ( \boldsymbol { \mathrm { I } } _ { \beta } ) ) , \dots , K _ { \mathrm { P r e } } ( \boldsymbol { \mathrm { Z } } _ { \alpha \times 1 } ) , E _ { \beta \times 1 } ( \boldsymbol { \mathrm { Z } } _ { \alpha \times 1 } ) ) , \dots , E _ { \beta } ( \boldsymbol { \mathrm { Z } } _ { \alpha \times 1 } ) ) , \quad \underset { j = \delta \times + 1 } { \prod } \delta ( z _ { j } ) d z _ { \beta , j + 1 } , \dots , d z _ { \nu } } \\ & { = \underset { \mathrm { I } = 1 } { \prod } \underset { \mathrm { I } = 1 } { \prod } \underset { \mathrm { I } \neq j \neq i \neq i \neq i \neq j \neq 1 } { \prod } \epsilon _ { i \neq 1 , \dots , \delta \neq i - 1 } \big ( E _ { i } \big ( | \boldsymbol { \mathrm { I } } _ { \beta + 1 , \dots , \delta \times \delta } \big ( \boldsymbol { \mathrm { I } } _ { \alpha } \big | + | \boldsymbol { \mathrm { Z } } _ { \alpha \times 1 } , \dots , z _ { i + 1 } , \dots \big ) \big ) , } \\ & { = \underset { \mathrm { I } \neq j \neq i } { \prod } \epsilon _ { i \neq 1 , \dots , \delta \neq j \neq 1 } , \qquad \quad \underset { \boldsymbol { X } _ { i + 1 } \neq j \neq i \neq i \neq j \neq 1 } { \prod } \epsilon _ { i \neq 1 , \dots , \delta \neq j \neq i } , \qquad \big ( \epsilon _ { i \neq 1 } \big | \epsilon _ { 1 } \big | , \big | \boldsymbol { \mathrm { I } } _ { \beta } \big | 1 , \dots , z _ { i + 1 } , \dots , z _ { i + 1 } , \big ) \big ) } \\ &  \quad \cdot \int \delta _  \end{array}
$$

Similarly, we get the marginal copula density of Y

$$
c _ { Y } ( { \pmb u } _ { Y } ) = \prod _ { t = d _ { X } + 1 } ^ { d - 1 } \prod _ { i = 1 } ^ { d - t } c _ { i , i + t ; i + 1 , \dots , i + t - 1 } ( F _ { i | i + 1 , \dots , i + t - 1 } ( z _ { i } | z _ { i + 1 } , \dots , z _ { i + t - 1 } ) ,  \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ &  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad F _ { i + t | i + 1 , \dots , i + t - 1 } ( z _ { i + t } | z _ { i + 1 } , \dots , z _ { i + t - 1 } ) ) .
$$

$$
\begin{array} { r l } & { \quad :  \mathbf { X } ^ { p }  = [ \psi \frac { \psi _ { 1 } } { \psi _ { 2 } } \frac { \psi _ { 2 } } { \psi _ { 3 } \psi _ { 4 } } \frac { \psi _ { 1 } } { \psi _ { 4 } } \frac { \psi _ { 2 } } { \psi _ { 3 } \psi _ { 4 } } ] \psi \psi _ { 4 } , } \\ & { \quad : =  \psi \frac { \psi _ { 1 } } { \psi _ { 4 } \psi _ { 4 } } \frac { \psi _ { 2 } } { \psi _ { 3 } \psi _ { 4 } } \frac { \psi _ { 1 } } { \psi _ { 4 } } \frac { \psi _ { 2 } } { \psi _ { 3 } \psi _ { 4 } } \frac { \psi _ { 1 } } { \psi _ { 4 } } \frac { \psi _ { 1 } } { \psi _ { 4 } }  + \frac { \psi _ { 2 } } { \psi _ { 3 } \psi _ { 4 } } \frac { \psi _ { 2 } } { \psi _ { 4 } } \frac { \psi _ { 3 } } { \psi _ { 4 } } \frac { \psi _ { 4 } } { \psi _ { 4 } } , } \\ & { \quad +  \psi \frac { \psi _ { 2 } } { \psi _ { 4 } \psi _ { 4 } } \frac { \psi _ { 3 } } { \psi _ { 4 } } \frac { \psi _ { 2 } } { \psi _ { 4 } } \frac { \psi _ { 3 } } { \psi _ { 4 } } \frac { \psi _ { 4 } } { \psi _ { 4 } } \frac { \psi _ { 4 } } { \psi _ { 4 } }  + \frac { \psi _ { 3 } } { \psi _ { 4 } } \frac { \psi _ { 4 } } { \psi _ { 4 } } \frac { \psi _ { 4 } } { \psi _ { 5 } \psi _ { 4 } } \frac { \psi _ { 5 } } { \psi _ { 5 } \psi _ { 4 } } } \\ &  \quad -  \psi \frac { \psi _ { 2 } } { \psi _ { 5 } } \frac { \psi _ { 3 } } { \psi _ { 5 } \psi _ { 4 } } \frac { \psi _ { 5 } }  \psi _   \end{array}
$$

## C Computational and Scaling Limitations of VINE

While VINE scales to much higher dimensions than traditional non-parametric MI estimators, its practical implementation is subject to computational bottlenecks and numerical instabilities as d grows exceptionally large. Because a D-vine construction requires estimating up to $d ( d - 1 ) / 2$ non-parametric pair-copulas which cannot be estimated all in parallel, the computational complexity increases greatly with d. For standard tabular tasks or moderately-sized latent spaces $( d \leq 6 4 )$ , this overhead is well-managed and often computes faster than the training loops required for neural estimators. However, for heavily uncompressed spaces where $d > 1 0 0$ , the sheer volume of sequential kernel density estimations becomes a primary computational bottleneck.

Furthermore, while the theoretical error bound isolating $d ^ { 2 }$ is a polynomial scaling factor rather than an exponential one, numerical error accumulation remains a practical limitation. Because pseudo-observations are generated sequentially through the vine trees, small estimation errors in the conditional CDFs of the initial trees propagate into the inputs of deeper trees. Even with the threshold parameter τ actively pruning weak dependencies to combat this accumulation, the compounded numerical drift can eventually degrade the accuracy of the estimator when d scales into the hundreds. We illustrate this breakdown point in Figure 7, which plots both the average computational time and the corresponding loss in estimation accuracy as a function of the dimension. In our empirical evaluations, VINE successfully maintained stable estimates for dimensions up to $d = 2 5 6 ;$ however, when evaluated at $d = 5 1 2$ , the accumulated approximation errors and numerical instabilities cause the estimator to break down significantly.

![](images/03a3104a2715255ac9537506f35e6a6b78fe0dc4d6fb530eaa10079c3c063018.jpg)  
Figure 7: Empirical scaling limitations of the VINE estimator on a multivariate Gaussian task. As the d increases, the computational time scales quadratically (left axis). For d $> 6 4$ , error accumulation from deep conditional evaluations begins to notably degrade the overall MI estimation accuracy (right axis), illustrating the practical limits of the D-vine decomposition in extremely high-dimensional spaces.

## D Regularization via the threshold parameter

To evaluate the sensitivity of our estimator to the crucial threshold hyperparameter, we investigate four different datasets with different dimensions. As described in Section 3, the threshold is a pruning mechanism that replaces any bivariate pair-copula with an independence copula if its empirical Kendall’s τ falls below the specified value. This effectively removes weak dependencies in the higher-order trees of the vine, reducing the risk of error accumulation and overfitting. We evaluated the threshold’s impact across four diverse tasks: Gaussian, cube-transformed, spiral-mapped, and Student-t distribution. As illustrated in Figure 7, the estimator is fairly stable and aligned with the ground truth across all values between $1 0 ^ { - 4 }$ and $1 0 ^ { - 1 }$ . For values larger than $1 0 ^ { - 1 }$ , the estimator begins to prune legitimate dependencies, leading to a significant underestimation of the ground truth MI.

For every experiment, setting $\tau = 0 . 1$ 1 leads to the most accurate MI estimate. The selection of τ is particularly critical for the Student-t data. Due to the heavy-tailed nature of the distribution and the high dimensionality of the latent space, non-parametric estimators are susceptible to fitting spurious correlations in the deeper layers of the vine construction. In these regimes, the threshold acts as a vital structural regularizer. By setting a default value of $\tau = 0 . 1$ , we effectively filter out sampling noise without requiring ad hoc tuning for specific datasets, something that would be impossible in most realistic scenarios.

## E Experiment details

## E.1 Implementation setup

VINE is implemented using the Python package pyvinecopulib [40]. In all experiments, the pair copulas are estimated non-parametrically using Transformation Local-Likelihood (TLL) kernels [44] with quadratic estimation method. According to Appendix D, we set the threshold parameter $\tau = 0 . 1$

The neural benchmarks include MINE, InfoNCE, SMILE, MIENF, VCE, and MINDE. These estimators utilize a standardized neural critic consisting of a three-layer Multi-Layer Perceptron (MLP) architecture with 500 hidden units per layer and ReLU activation functions. All neural models are trained using the Adam optimizer with a learning rate of $5 \times 1 0 ^ { - 4 }$ , a weight decay of $1 0 ^ { - 5 }$ , and a batch size of 500. To balance computational overhead with convergence stability, we set the maximum training duration to 500 iterations with an early stopping patience of 200 iterations. For synthetic and transformed Gaussian tasks, the models are trained on 5,000 samples, while the image-based benchmarks utilize 10,000 samples to accommodate the higher-dimensional latent spaces. The vine method models were run on an Apple M4 Max (16-core CPU, 36GB RAM), whereas the neural estimators were run on NVIDIA A1000 GPUs.

![](images/e503559266a249895556120f5adb55808716a0eb6a2ca4dace2b599553397c95.jpg)  
(a) Gaussian Dependency (d = 16)

![](images/bacda3d11687e8353dc1153a8a1b6d6dccfd4f61c1ca1146b4bc9b5580f77ff0.jpg)  
(b) Cube Transformation (d = 8)

![](images/e171ae01c8c110fb82179ec4547eef130426effb75d3c0dbd7d4752dc0409ce4.jpg)  
(c) Spiral Mapping (d = 64)

![](images/0e00c3e7acde640fcda13333ea6cc1b6f5b9dbb05c77ae215d4e977887c2efcf.jpg)  
(d) Student-t Distribution $( d = 3 2 , \nu = 3 )$  
Figure 8: Sensitivity of the Vine Copula Estimator to the sparsity threshold τ across datasets. The estimator remains mostly stable across three orders of magnitude, with significant information loss occurring only after $\tau > 0 . 1$ . This robustness allows for a universal default threshold without per-task tuning.

## E.2 Transformed Gaussian datasets

This appendix provides the formal definitions for the non-Gaussian data-generating distributions used in our benchmarking suite. Mutual information is invariant under strictly increasing bijective transformations of the margins, but the underlying copula is not generally preserved under such transformations. As a result, applying these transformations offers no guarantee that the simplifying assumption still holds. The only case we consider in which this preservation is guaranteed is when we apply strictly increasing transformations to the margins of multivariate Gaussian distributions. Using these facts, we construct benchmarks with known ground-truth MI, allowing us to investigate VINE’s robustness to violations of model assumptions, particularly the simplifying assumption.

We consider the following mappings applied to multivariate Gaussian distributions:

• Strictly increasing transformation of the margins: simplifying assumption is preserved

– Cube Transformation: $f ( x ) = x ^ { 3 }$

– Hyperbolic Tangent (Tanh): $f ( x ) = \operatorname { t a n h } ( x )$

– Sign-Square: $f ( x ) = \mathrm { s i g n } ( x ) \cdot x ^ { 2 }$

• Joint transformation: simplifying assumption is not preserved

– Spiral: This task defines a dependency where variables are distributed along a 2D spiral arm. The coordinates are generated via a polar mapping $( x , y ) = ( \theta \cos \theta , \theta \sin \theta )$ , where the angle θ is coupled between X and Y. This creates a relationship that is locally structured but globally complex, requiring the estimator to capture a "winding" dependency that spans multiple rotations of the support.

– Swiss roll: A classic manifold learning benchmark where a 2D surface is "rolled" into a 3D volume. The dependency is defined on the surface of the roll, which is then embedded into a d-dimensional space. This tests whether an estimator can successfully isolate the underlying low-dimensional manifold structure from the high-dimensional ambient noise and redundant features.

In addition to transformations of Gaussian distributions, we include the following data-generating distributions in our experiments:

• Multivariate t-distribution: simplifying assumption holds. The true MI can be determined analytically from the distribution parameters.

• Mixture of Gaussian (MoG) distributions: simplifying assumption does not hold. The true MI can be estimated with high precision via Monte Carlo integration.

## E.3 Image dataset

Experiment details: Neural estimators were trained with a three-layer MLP architecture (500 units per layer) for 150 iterations per estimation. Our Vine estimator utilizes non-parametric Transformation Local Likelihood (TLL) kernels for bivariate pair-copula estimation to capture complex, non-linear dependencies without making restrictive parametric assumptions.

Table 2: Cumulative Explained Variance Captured by PCA for Different Latent Dimensions (d). Values represent the percentage of total variance preserved for the input (X) and target (Y ) image sets.
<table><tr><td>Dataset</td><td>Dim (d)</td><td>Var (X)</td><td>Var (Y)</td><td>Chosen?</td></tr><tr><td rowspan="3">Gaussian</td><td>4</td><td>83.04%</td><td>82.95%</td><td>No</td></tr><tr><td>8 12</td><td>92.59% 97.27%</td><td>93.58% 97.26%</td><td>No Yes</td></tr><tr><td>16</td><td>99.86%</td><td>99.86%</td><td>一</td></tr><tr><td rowspan="3">Rectangle</td><td>20</td><td>88.67%</td><td>88.58%</td><td>No</td></tr><tr><td>30</td><td>92.75%</td><td>92.74%</td><td>No</td></tr><tr><td>40</td><td>95.07%</td><td>95.09%</td><td>Yes</td></tr><tr><td></td><td>50</td><td>96.63%</td><td>96.65%</td><td></td></tr></table>

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction clearly outline the paper’s primary contributions: an efficient, interpretable MI estimator (VINE) that leverages vine copula theory to bypass the curse of dimensionality. These claims are supported by theoretical motivations in Section 4 and empirical results across synthetic and image datasets in Section 5.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Limitations are explicitly discussed in Section 6.

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

Justification: The paper states the assumptions for its theoretical contribution, Proposition 3.1, and provides a proof in Appendix B. Furthermore, it discusses the necessary conditions for the

"simplifying assumption" in Section 2.1 and references the specific regularity conditions (A1–A4) required for the convergence rate claims in Section 4.

Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: All experimental setups are explained, and all hyperparameters are given in detail in Appendix E.

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

## Answer: [Yes]

Justification: We have provided implementation details and hyperparameter settings in the supplemental material to ensure reproducibility.

Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/public/ guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: All hyperparameters and experimental details are given in Appendix E.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: For the experiment in Section 5.3, we report standard error for every estimator.   
However, we do not report standard error for every experiment due to the computational time.

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

Justification: This is provided in Appendix E.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The research is a foundational study focusing on a new methodology for mutual information estimation using synthetic and publicly available benchmark datasets. It does not involve human subjects, sensitive personal data, or any data collection practices that conflict with the NeurIPS Code of Ethics.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [N/A]

Justification: This work by its nature is focused on mutual information estimation, and has no direct societal impact.

Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: This work only uses synthetic data and standard synthetic image benchmarks.The project does not involve the release of pre-trained generative models, large-scale language models, or sensitive scraped datasets that carry a high risk of misuse or require specific safety safeguards.

Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We properly credit the original creators of all datasets and baseline mutual information estimators (e.g., VCE, MINE, InfoNCE, and other MI estimators) through formal citations.

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

Justification: No new assets are released in this work.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: No experiments are performed using human subjects.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: No experiments are performed using human subjects.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or nonstandard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [N/A]

Justification: Large Language Models were not a component of the core methodology, scientific rigor, or originality of the research.

## Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.