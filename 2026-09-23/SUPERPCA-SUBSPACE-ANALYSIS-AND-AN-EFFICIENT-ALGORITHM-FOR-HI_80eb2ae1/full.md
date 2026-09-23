# SUPERPCA: SUBSPACE ANALYSIS AND AN EFFICIENT ALGORITHM FOR HIGH-DIMENSIONAL PCA

IRINA-BEATRICE HAAS<sup>∗</sup>, MAIKE MEIER<sup>†</sup>, YUJI NAKATSUKASA<sup>∗</sup>, AND TAEJUN PARK<sup>‡</sup>

Abstract. Principal component analysis (PCA) is a fundamental tool to reduce the dimensionality of the data in many applications. PCA finds a few signal directions that contain most of the variability of the data by computing the eigenvectors of the sample covariance matrix. In this work, we focus on the spiked covariance model, in which the data vectors are defined by a few orthogonal signals plus an isotropic Gaussian noise, and our goal is to estimate one or more of the leading signals. Our main theoretical finding is that the subspace spanned by several leading eigenvectors of the sample covariance matrix contains significant information about the desired signals long before the individual eigenvectors converge to the population principal components. To prove this, we derive a posteriori bounds for the angle between the subspace spanned by the desired population signals and the subspace obtained from the sample using perturbation theory for singular vectors. This leads to a new algorithm, SuperPCA (SUbsPace subsamplER PCA), which capitalizes on an approximate eigenspace of the sample covariance matrix to find the leading signals far more eficiently and accurately than classical PCA in the high-dimensional, multi-signal setting. SuperPCA exploits only a small number of subsampled coordinates of the data, which can lead to tremendous savings in data acquisition cost, especially when the signals are approximately sparse. For the same number of measurements, SuperPCA can ofer a factor 10 improvement in accuracy compared to the classical PCA method.

Key words. principal component analysis, spiked covariance model, singular vector perturbation, randomized numerical linear algebra, subsampled least squares

AMS subject classifications. 65F15, 62H25, 65F20, 65F55

1. Introduction. Principal Component Analysis (PCA) is an important tool in data analysis and dimension reduction. Given a dataset, PCA seeks to find (orthogonal) directions that best explain the variance in the data. We consider the popular spiked covariance model introduced in [23].

In this model, we observe n p-dimensional measurements of the form

$$
x _ { i } = \sum _ { j = 1 } ^ { k } \sqrt { \beta _ { j } } g _ { j } ^ { i } u _ { j } + \sigma \eta _ { i } , \quad i = 1 , \ldots , n\tag{1.1}
$$

where $\sqrt { \beta _ { j } } g _ { j } ^ { i } u _ { j }$ are signals and $\sigma \eta _ { i }$ is noise. We have $k ( \geq 1 )$ p-dimensional signals with orthonormal directions $u _ { 1 } , \ldots , u _ { k }$ and respective signal strengths $\beta _ { 1 } \geq \cdot \cdot \cdot \geq \beta _ { k } > 0$ . The noise is assumed random normal with intensity $\sigma > 0$ . The $g _ { j } ^ { i } \sim \mathcal { N } ( 0 , 1 )$ and $\eta _ { i } \sim \mathcal { N } ( 0 , I _ { p } )$ are scalar and vector standard normal random variables, respectively. Each measurement $x _ { i }$ is then independently and identically distributed like $x \sim \mathcal { N } ( 0 , K _ { \mathrm { p o p } } )$ with population covariance matrix $K _ { \mathrm { p o p } }$ given by

$$
K _ { \mathrm { p o p } } = \sum _ { j = 1 } ^ { k } \beta _ { j } u _ { j } u _ { j } ^ { T } + \sigma ^ { 2 } I _ { p } .\tag{1.2}
$$

The population covariance matrix has dominant eigenvectors $u _ { 1 } , \ldots , u _ { k }$ with corresponding eigenvalues $\lambda _ { 1 } =$ $\beta _ { 1 } + \sigma ^ { 2 } , \ldots , \lambda _ { k } = \beta _ { k } + \sigma ^ { 2 }$ . We call the vectors $u _ { j }$ the population principal components. The aim of PCA is to approximate the principal components based on samples drawn from $\mathcal { N } ( 0 , K _ { \mathrm { p o p } } )$ . Classically, this is done through the eigendecomposition of the sample covariance matrix. Namely, collect the samples in a data matrix $X \in \mathbb { R } ^ { p \times n }$ of the form $X = [ x _ { 1 } , \ldots , x _ { n } ]$ and define the sample covariance matrix $S _ { n } = X X ^ { T } / n$ . The main topic of analysis in PCA is how well the dominant eigenvectors of $S _ { n }$ , say $\hat { u } _ { 1 } , \dotsc , \hat { u } _ { k }$ , called the sample principal components, approximate the population principal components, i.e., $u _ { 1 } , \ldots , u _ { k }$

Instead of the eigenvalue decomposition, the analysis in this paper is based on singular vector perturbation theory, and we will mostly focus on the singular value decomposition (SVD) of the scaled data matrix

$X _ { n } = X / { \sqrt { n } } = { \hat { U } } { \hat { \Sigma } } { \hat { V } }$ . The left singular vectors $\hat { U }$ are the same as the eigenvectors of $S _ { n }$ . The singular values $\hat { \sigma } _ { 1 } \geq \cdot \cdot \cdot \geq \hat { \sigma } _ { \operatorname* { m i n } ( p , n ) } \geq 0$ are the square roots of the eigenvalues of $S _ { n }$ . Ideally, the dominant singular values of $X _ { n }$ approximate the singular values $\sigma _ { 1 } , \ldots , \sigma _ { p }$ of the square root of $K _ { \mathrm { p o p } }$ . That is, $\sigma _ { j } = \sqrt { \lambda _ { j } } = \sqrt { \beta _ { j } + \sigma ^ { 2 } }$ for $j = 1 , \dots , k$ and $\sigma _ { j } = \sigma$ for $j > k$

PCA can be considered in diferent (asymptotic) regimes. In the classical context the number of samples n tends to infinity and the dimension of the problem $p$ is fixed. It is well known that the sample covariance matrix then converges to the population covariance matrix almost surely as $n \to \infty$ . As a result, the sample principal components consistently estimate the population principal components [1, 35].

However, in modern applications, the dimension of the problem $p$ is often of comparable size to the number of samples n, sometimes even larger. In this regime, the sample principal components are generally inconsistent and exhibit the so-called Ben Arous-Baik-P´ech´e (BBP) phase transition which describes suficient conditions on the signal to noise ratio and dimensions $n , p$ to detect the signal [4, 5, 23, 41, 49]. Recent work has therefore focused on finite-sample bounds that quantify the estimation error in the principal components for large but finite p and n [25, 28, 37, 42, 46]. These bounds typically depend on spectral gaps between consecutive eigenvalues of the population covariance matrix.

Our work builds on this finite-sample perspective. Rather than estimating each signal from a sample principal component of the same dimension, we show that a larger sample left singular-subspace of $S _ { n }$ can provide substantially more accurate information about the desired population subspace, especially when several signal strengths are close together.

The theoretical findings have computational and practical implications. Once we find a subspace in which the desired signal lies with suficient accuracy, a natural goal is to find the signal from the subspace. We introduce an algorithm, SuperPCA, that attempts to do so.

1.1. Motivating examples. The reasoning behind considering subspaces of difering dimensions can be illustrated with examples. First, consider a model with three (orthonormal) signals $u _ { 1 } , u _ { 2 }$ , and $u _ { 3 }$ with respective signal strengths $\beta _ { 1 } \geq \beta _ { 2 } \geq \beta _ { 3 }$ . Let $\hat { u } _ { 1 } , \hat { u } _ { 2 }$ , and $\hat { u } _ { 3 }$ denote the three dominant left singular vectors of the corresponding data matrix $X _ { n }$ . We are interested in how well $\hat { u } _ { 1 }$ , span $( \hat { u } _ { 1 } , \hat { u } _ { 2 } )$ , and span $( \hat { u } _ { 1 } , \hat { u } _ { 2 } , \hat { u } _ { 3 } )$ can approximate the principal component of interest: $u _ { 1 }$ . The relevant quantity, we argue, is the angle between $u _ { 1 }$ and the above mentioned vector and subspaces. If the sine of this angle is close to 0, then the principal component is well-contained. $\mathrm { A s }$ such, the sine of the angle can be considered as a measure of distance between $u _ { 1 }$ and the respective subspace.

In Figure 1, we fix $\beta _ { 1 } = 7 5$ and $\beta _ { 3 } = 2 5$ , and we vary the strength of the second signal $\beta _ { 2 } \in [ \beta _ { 3 } , \beta _ { 1 } ]$ We plot the distance between $u _ { 1 }$ and the subspaces span $( \hat { u } _ { 1 } )$ , span $( \hat { u } _ { 1 } , \hat { u } _ { 2 } )$ , and span $( \hat { u } _ { 1 } , \hat { u } _ { 2 } , \hat { u } _ { 3 } )$ . This is repeated for (approximations) to diferent asymptotic and finite-sample regimes. In the second frame, $p$ and n tend to infinity together, and theoretically the limit of the angle $\theta ( u _ { 1 } , \hat { u } _ { 1 } )$ is only dependent on $n , p , \sigma .$ , and $\beta _ { 1 }$ and independent from $\beta _ { 2 }$ and $\beta _ { 3 }$ , provided $\beta _ { 1 } \neq \beta _ { 2 }$ (see [41, Thm. 4]). In the same asymptotic regime, yet under the assumption $\beta _ { 1 } = \beta _ { 2 } > \beta _ { 3 }$ , it is impossible to distinguish between $u _ { 1 }$ and $u _ { 2 }$ . We can only use span $( \hat { u } _ { 1 } , \hat { u } _ { 2 } )$ to approximate $\operatorname { s p a n } ( u _ { 1 } , u _ { 2 } )$ . This phenomenon is displayed in the figure as sin $\theta ( u _ { 1 } , \hat { u } _ { 1 } )$ is independent from $\beta _ { 2 }$ for a large portion of the graph. The inconsistency for values of $\beta _ { 2 }$ close to $\beta _ { 1 }$ can be explained by the finiteness of p and $n .$ . The third panel aims to approximate the classical asymptotic regime. As $n  \infty$ , sin $\theta ( u _ { 1 } , \hat { u } _ { 1 } )  0$ . In the figure, the angle is dependent on $\beta _ { 2 }$ yet consistently small (note the log-scale).

Our main regime of interest is displayed in the first panel of Figure 1. The distance between $u _ { 1 }$ and $\hat { u } _ { 1 }$ deteriorates rapidly as $\beta _ { 2 }$ increases. Indeed, it is intuitive that, if $\beta _ { 1 } / \beta _ { 2 }$ is close to $1 , \hat { u } _ { 2 }$ would contain information on $u _ { 1 }$ and $\hat { u } _ { 1 }$ on $u _ { 2 }$ . Visually, this can be seen in the red, dotted curve, which contains the distance between $u _ { 1 }$ and span $( \hat { u } _ { 1 } , \hat { u } _ { 2 } )$ . This distance is consistently small. This experiment shows that when $\beta _ { 1 } \approx \beta _ { 2 }$ the error in the leading sample principal component $\hat { u } _ { 1 }$ can be large but a larger subspace such as span $( \hat { u } _ { 1 } , \hat { u } _ { 2 } )$ still contains the leading signal $u _ { 1 }$ with high accuracy.

Figure 2 visualizes the same phenomenon with a slightly diferent set-up. Here, there are $k = 9$ signals with strengths $\beta _ { 1 } = 1 0 , \beta _ { 2 } = 9 , \ldots , \beta _ { 9 } = 2$ . We plot the cosine of the angle between $u _ { 1 }$ and the jth sample principal component $\hat { u } _ { j }$ , which is close to 1 when the distance between $u _ { 1 }$ and $\hat { u } _ { j }$ is small. In either asymptotic regime (bottom two frames), we see that $\hat { u } _ { 1 }$ is very close to $u _ { 1 }$ and that all further sample principal components $\hat { u } _ { 2 } , \hat { u } _ { 3 } , . . .$ . contain hardly any components in the direction of $u _ { 1 }$ . However, in the finite sample regime in the top panel, trailing sample components contain large components in the direction of $u _ { 1 }$ It is immediately clear that, in this context, span $( \hat { u } _ { 1 } , \hat { u } _ { 2 } )$ contains much more information on $u _ { 1 }$ than just $\hat { u } _ { 1 }$

![](images/681fa00ecdc53b1d4670a746914b8998040fe53cd7ab9a2334ffb6bcb7826848.jpg)

![](images/eb6c57b237e41003dfd003809f56365dbc6c5f3f59894e0794564265d0fdbda9.jpg)

![](images/cd1eba90681ed3d503b35d7045d873a8c59f1487bb3f84d7e4161746404e27d9.jpg)  
Fig. 1. The distance between the dominant signal $u _ { 1 }$ and spaces spanned by the dominant principal components $\hat { u } _ { 1 } , ~ \hat { u } _ { 2 } ,$ and uˆ<sub>3</sub> $f o r$ various finite sample and asymptotic regimes. The plotted quantity is the sine of the angle(s) between the dominant population component $u 1$ and a subspace spanned by diferent sample components $\hat { u } _ { j } . ~ A$ small quantity indicates $u _ { 1 }$ is well-contained in the corresponding subspace. In each of the plots, $\beta _ { 1 } = 7 5$ and $\beta _ { 3 } = 2 5$ are kept constant and only $\beta _ { 2 }$ is changed. The lines show the mean of 20 iterations.

![](images/41f48fcb3eb3e084be8abce17d21b837d3984030ad8806296f268ce6ee13b008.jpg)  
Fig. 2. How much do sample components point in the direction of a specific principal component? Here, $\beta _ { 1 } = 1 0$ $\beta _ { 2 } = 9 , \ldots , \beta _ { 9 } = 2 ,$ , and $\sigma ^ { 2 } = \mathrm { i }$

1.2. Contributions. In this paper we investigate the accuracy of the sample principal components in the finite sample setting where the data contains multiple signals $( k > 1 )$ . More precisely, we introduce

new bounds on the angle between the population principal components and a subspace spanned by sample principal components, where the latter can be of larger dimension than the population subspace that is estimated (Section 3), and

a new algorithm SuperPCA, which finds an improved estimate of the leading signal directions in a (potentially larger) candidate subspace span $( \hat { U } _ { \mathrm { c a n d } } )$ . SuperPCA hinges on the idea of the classical Rayleigh-Ritz (RR) process [40, Sec. 11.3], but crucially, subsamples only a carefully chosen set of coordinates of the measurements in the refinement step, which reduces the cost of data acquisition, potentially dramatically (Section 4).

As our numerical experiments (Section 5) show, SuperPCA is particularly efective for highly coherent signal directions and for certain distributions of the signal. We compare our algorithm to standard PCA and to Johnstone and Lu’s SparsePCA algorithm [24].

1.3. Notation. Throughout the paper, we assume model (1.1). The signal directions are denoted as $u _ { 1 } , \ldots , u _ { k }$ and collected in a matrix $U \in \mathbb { R } ^ { p \times k }$ . We let $X \in \mathbb { R } ^ { p \times n }$ denote the data matrix whose columns are samples, and $X _ { n } = X / { \sqrt { n } }$ as the scaled data matrix. The singular value decomposition of $\begin{array} { r } { X _ { n } = \hat { U } \hat { \Sigma } \hat { V } ^ { T } } \end{array}$ is denoted with hats. We denote $\sigma _ { i }$ the singular values of $K _ { \mathrm { p o p } }$ and $\sigma _ { i } ( M )$ the singular values of any matrix $M .$ . The notation $\Theta ( V , W )$ , for $V \in \mathbb { R } ^ { m \times n }$ $W \in \mathbb { R } ^ { m \times l }$ , and $m \ge \operatorname* { m a x } ( n , l )$ , denotes the diagonal matrix of dimension $\operatorname* { m i n } ( n , l )$ of canonical angles between the subspaces spanned by V and W. The norms $\| \cdot \| _ { 2 }$ and $\| \cdot \| _ { F }$ denote the spectral and Frobenius norm, respectively. We use MATLAB notation to indicate submatrices, $\mathrm { e . g . } A ( i , : )$ is the ith row of A.

2. Review of related work. In this section we summarize previous work on the estimation of principal components. The existing results are mostly stated in terms of the eigendecompositions; we translate them to an SVD context. Finally we also discuss classical perturbation results for singular vectors.

2.1. Estimation in the limit $p / n  \gamma > 0$ . The high-dimensional asymptotic regime, in which $p , n  \infty$ with $p / n  \gamma > 0$ , has been extensively studied using random matrix theory. A cornerstone result is the Marchenko-Pastur law [34], which describes the limiting spectrum of the sample covariance matrix in the absence of signals. The convergence of the largest and smallest sample eigenvalues to the upper [16, 23] and lower [2, 3, 44] edges were previously studied, showing that the sample eigenvalues are more dispersed than the population eigenvalues [25]. For the spiked covariance model (1.1) introduced by Johnstone [23], Baik, Ben Arous and P´ech´e [4] and Baik and Silverstein [5] established the BBP phase transition: only spikes exceeding a critical threshold, namely $\sigma _ { i } ^ { 2 } / \sigma ^ { 2 } > 1 + \sqrt { \gamma } ,$ separate from the bulk spectrum (predicted by the Marchenko-Pastur law) and can be consistently detected.

A similar phase transition holds for the sample principal components. Paul [41] and subsequent work [49] showed that, above the BBP threshold, the sample eigenvectors retain nontrivial alignment with the population eigenvectors but remain asymptotically biased whenever $\gamma > 0$ . Below the threshold (e.g. when $\sigma _ { 1 } ^ { 2 } / \sigma ^ { 2 } \le 1 + \sqrt { \gamma } )$ , the sample and population eigenvectors become asymptotically orthogonal. These results motivate the search for improved estimators in the finite-sample regime considered in this work.

2.2. Finite sample estimation. The context where p and n are both finite and of the same order is of growing relevance and attention in light of the rapid growth of the size of datasets available today, however theoretical results are still significantly less developed for this context. Paul’s work [41] proves that, for finite $p , n ,$ sample principal components will contain a non-informative noise part that obstruct good estimation of the signals as in the asymptotic case.

Nadler (2008) [37] was among the first to investigate the finite sample case in more detail. He specifically considered the single signal case $( k = 1 )$ using eigenvalue and -vector perturbation theory and derives upper and lower bounds on the relevant quantities in terms of random variables. A simplified version of the main result — neglecting small terms and assuming the signal is suficiently strong — is sin $\theta ( u _ { 1 } , \hat { u } _ { 1 } ) \lesssim$ $\textstyle { \frac { \sigma } { \sqrt { \beta _ { 1 } } } } { \sqrt { { \frac { p } { n } } } } + O ( \sigma ^ { 2 } )$

There have been a few other results in the finite sample setting in the last decade. Koltchinskii and Lounici [28] and Reiss and Wahl [42] derive error bounds on sin $\theta ( u _ { j } , \hat { u } _ { j } )$ and the general $k > 1$ case sin $\Theta ( U _ { 1 } , \hat { U } _ { 1 } )$ , where $U _ { 1 }$ and $\widehat { U } _ { 1 }$ contain $r ~ \leq ~ k$ principal components respectively, which depend on the trace of $K _ { \mathrm { p o p } }$ band involve a spectral gap of the form $\sigma _ { j } ^ { 2 } - \sigma _ { j + 1 } ^ { 2 }$ (respectively $\sigma _ { r } ^ { 2 } - \sigma _ { r + 1 } ^ { 2 } )$ in their denominator. To be informative, their bounds require roughly that $n \gg \mathsf { \bar { p } } \sigma ^ { 2 } \mathrm { ~ [ 2 8 ] ~ }$ or $n \sim p ^ { 2 } ~ [ 4 2 ]$ , and that the spectral gap $\sigma _ { j } ^ { 2 } - \sigma _ { j + 1 } ^ { 2 }$ is suficiently large. Vaswani and Narayanamurthy [46] obtain error bounds on sin $\Theta ( U _ { 1 } , \hat { U } _ { 1 } ) \lVert _ { 2 }$ that behave as $\sim \sqrt { ( \beta _ { 1 } / \beta _ { r } ) ( \sigma ^ { 2 } / \beta _ { r } ) ( p \log p / n ) }$ , which again includes spectral gaps as $\sigma ^ { 2 } / \beta _ { r }$ and a $p / n$ term. In all these references, the dimensions of $U _ { 1 }$ is the same as that of $\hat { U } _ { 1 }$ . Hence a common feature of these results is that the error bounds depend on spectral gaps between adjacent singular values, making them less informative when several leading signals have similar strengths. Our analysis addresses this limitation by estimating signals using a subspace of larger dimension than the target population subspace.

2.3. Matrix perturbation theory for singular vectors. Our analysis of the error in the sample principal components hinges on classical perturbation theory for invariant subspaces. Indeed we can view the sample matrix $S _ { n }$ as a perturbed version of $K _ { \mathrm { p o p } } \mathrm { o r }$ , in the SVD formulation, the data matrix $X _ { n }$ is a noisy version of the signal part of (1.1). For this task, Davis and Kahan [11] established perturbation bounds for eigenspaces of symmetric matrices, while Wedin [48] obtained analogous results for singular vectors.

A key feature of Wedin’s theorem is that the perturbation bound depends on a spectral gap separating the desired singular subspace from the remaining spectrum. However, when several signal strengths are close together such that this gap becomes small, Wedin’s theorem becomes less informative. The main idea of this paper is to enlarge the approximation subspace, replacing the adjacent spectral gap by a larger separation that yields substantially sharper finite-sample bounds.

3. Main theoretical results: estimation of principal components using a larger subspace. We now quantify the accuracy with which the leading population signal subspace can be approximated by a larger sample singular subspace. Throughout, let $U _ { 1 } \in \mathbb { R } ^ { p \times r _ { 1 } }$ denote the leading population signal directions and let $[ \widehat { U } _ { 1 } , \stackrel { \times } { U } _ { 2 } ] \in \mathbb { R } ^ { p \times r _ { 2 } }$ , with $\widehat { U } _ { 1 } \in \mathbb { R } ^ { p \times r _ { 1 } }$ and $r _ { 2 } > r _ { 1 }$ , denote the leading left singular vectors of the data matrix $X _ { n }$ b b b. The approximation error is measured using canonical angles between these subspaces, which measure how much of the smaller subspace is contained in the larger one. We use the standard definition for subspaces of unequal dimensions [45, Section I.5]: the canonical angles between $U _ { 1 } \in \mathbb { R } ^ { p \times r _ { 1 } }$ and $[ \widehat { U } _ { 1 } , \widehat { U } _ { 2 } ] \in \mathbb { R } ^ { n \times r _ { 2 } }$ are defined as

$$
\Theta ( U _ { 1 } , [ \widehat { U } _ { 1 } , \widehat { U } _ { 2 } ] ) = \mathrm { d i a g } ( \theta _ { 1 } , \theta _ { 2 } , . . . , \theta _ { r _ { 1 } } ) , \mathrm { w h e r e } \theta _ { i } = \operatorname { a r c c o s } ( \sigma _ { i } ( U _ { 1 } ^ { T } [ \widehat { U } _ { 1 } , \widehat { U } _ { 2 } ] ) ) .\tag{3.1}
$$

We further denote $\widehat { U } _ { 3 }$ an orthogonal complement of $[ \widehat { U } _ { 1 } , \widehat { U } _ { 2 } ] ;$ that is, $\widehat { U } _ { 3 } ^ { T } [ \widehat { U } _ { 1 } , \widehat { U } _ { 2 } ] = 0$ . Noting that $\sin ^ { 2 } \theta +$ $\cos ^ { 2 } \theta = 1$ and that $\left\| U _ { 1 } ^ { T } [ \widehat { U } _ { 1 } , \widehat { U } _ { 2 } ] \right\| _ { 2 } ^ { 2 } + \left\| U _ { 1 } ^ { T } \widehat { U } _ { 3 } \right\| _ { 2 } ^ { 2 } = 1$ , a direct consequence of the definition above is that,

$$
\begin{array} { r } { \left\| \sin \Theta ( U _ { 1 } , [ \widehat { U } _ { 1 } , \widehat { U } _ { 2 } ] ) \right\| _ { 2 } = \left\| U _ { 1 } ^ { T } \widehat { U } _ { 3 } \right\| _ { 2 } , } \end{array}\tag{3.2}
$$

which is the quantity that we will use to measure the error made when we approximate the principal components $U _ { 1 }$ by span $( [ \widehat { U } _ { 1 } , \widehat { U } _ { 2 } ] )$ .

b bWe are now ready to state our theoretical results. First, we establish an upper bound for the case when there is only one signal of interest $( r _ { 1 } = 1 )$ by considering a subspace spanned by the leading left singular vectors of the data matrix; see Theorem 3.1. Then, we extend our result to the case of multiple signals in Theorem 3.2.

3.1. Estimating one signal. We derive a deterministic bound for the canonical angle between the dominant signal and the subspace spanned by the leading left singular vectors of the data matrix $X _ { n }$ . The proof of the theorem below can be found in Section 7.

Theorem 3.1 (Single signal case). Let $\begin{array} { r } { X _ { n } \ = \ \frac { 1 } { \sqrt { n } } X \ = \ \frac { 1 } { \sqrt { n } } [ x _ { 1 } , x _ { 2 } , . . . , x _ { n } ] \ \in \ \mathbb { R } ^ { p \times n } } \end{array}$ be the scaled data matrix where the data matrix X is as defined in (1.1). Then assuming $\sigma _ { 1 } ( X _ { n } ) > \sigma _ { r _ { 1 } + 1 } ( X _ { n } )$

$$
\sin \theta ( u _ { 1 } , \widehat { U } _ { 1 } ) \leq \sqrt { \frac { \sigma _ { 1 } ( X _ { n } ) ^ { 2 } - \left. u _ { 1 } ^ { T } X _ { n } \right. _ { 2 } ^ { 2 } } { \sigma _ { 1 } ( X _ { n } ) ^ { 2 } - \sigma _ { r _ { 1 } + 1 } ( X _ { n } ) ^ { 2 } } }\tag{3.3}
$$

where $u _ { 1 }$ is the dominant signal direction and $\widehat { U } _ { 1 } \in \mathbb { R } ^ { p \times r _ { 1 } }$ is the matrix containing the $r _ { 1 }$ leading left singular vectors $o f X _ { n }$

The bound for $r _ { 1 } > 1$ is useful in the multiple signals case $\left( k > 1 \right)$ when the dominant signal $u _ { 1 }$ is (almost) indistinguishable from the non-dominant signal(s), that is, the largest signal strength and the second largest signal strength are similar. By taking a larger subspace of dimension $r _ { 1 } > 1 , \sigma _ { 1 } ( X _ { n } ) ^ { 2 } - \sigma _ { r _ { 1 } + 1 } ( X _ { n } ) ^ { 2 }$ can be substantially larger than $\sigma _ { 1 } ( \bar { X } _ { n } ) ^ { 2 } - \bar { \sigma } _ { 2 } ( X _ { n } ) ^ { 2 }$ , making the bound (3.3) more informative. Furthermore,

$$
\sigma _ { 1 } ( X _ { n } ) = \left\| \widetilde { u } _ { 1 } ^ { T } X _ { n } \right\| _ { 2 } \approx \left\| u _ { 1 } ^ { T } X _ { n } \right\| _ { 2 } ,
$$

where $\tilde { u } _ { 1 }$ is the leading singular vector of the data matrix, which implies a small numerator.

3.2. Estimating multiple signals. We now extend Theorem 3.1 to estimating multiple population principal components. In particular, we consider the quantity $\begin{array} { r } { \left\| \sin \Theta ( U _ { 1 } , [ \widehat { U } _ { 1 } , \widehat { U } _ { 2 } ] ) \right\| _ { 2 } } \end{array}$ where $U _ { 1 } , \widehat { U } _ { 1 } \in \mathbb R ^ { p \times r _ { 1 } }$ and $\widehat { U } _ { 2 } \in \mathbb { R } ^ { p \times ( r _ { 2 } - r _ { 1 } ) }$ . The orthonormal matrix $U _ { 1 }$ is the dominant $r _ { 1 }$ signal directions and $[ \widehat { U } _ { 1 } , \widehat { U } _ { 2 } ]$ is the bleading $r _ { 2 }$ left singular vectors of the data matrix $X _ { n }$ b b. The result is presented below in Theorem 3.2 with its proof in Section 7.

Theorem 3.2. Let $\begin{array} { r } { X _ { n } = \frac { 1 } { \sqrt { n } } X = \frac { 1 } { \sqrt { n } } [ x _ { 1 } , x _ { 2 } , . . . , x _ { n } ] \in \mathbb { R } ^ { p \times n } } \end{array}$ be the scaled data matrix where the data matrix X is as defined in (1.1). Let $U _ { 1 } \in \mathbb { R } ^ { p \times r _ { 1 } }$ be the $r _ { 1 }$ leading signal directions with $U _ { \perp } \in \mathbb { R } ^ { p \times ( p - r _ { 1 } ) }$ as its orthogonal complement, and let $[ \widehat { U } _ { 1 } , \widehat { U } _ { 2 } ]$ be the leading $r _ { 2 } ( \geq r _ { 1 } )$ left singular vectors of the data matrix. Then assuming $\sigma _ { \operatorname* { m i n } } ( U _ { 1 } ^ { T } X _ { n } ) > \sigma _ { r _ { 2 } + 1 } ( X _ { n } )$

$$
\Bigl \| \sin \Theta ( U _ { 1 } , [ \widehat { U } _ { 1 } , \widehat { U } _ { 2 } ] ) \Bigr \| _ { 2 } \leq \frac { \bigl \| U _ { \bot } ^ { T } X _ { n } \check { V } \bigr \| _ { 2 } \sigma _ { \operatorname* { m i n } } ( U _ { 1 } ^ { T } X _ { n } ) } { \sigma _ { \operatorname* { m i n } } ( U _ { 1 } ^ { T } X _ { n } ) ^ { 2 } - \sigma _ { r _ { 2 } + 1 } ( X _ { n } ) ^ { 2 } } ,\tag{3.4}
$$

where $\check { V }$ is the leading $r _ { 1 }$ right singular vectors of $U _ { 1 } ^ { T } X _ { n . }$ , which is independent of $U _ { \mathrm { ~ l ~ } } ^ { T } X _ { n }$

The bound above is useful when the largest $r _ { 1 }$ signal strength is similar in magnitude to the $( r _ { 1 } + 1 )$ th strongest signal strength. Then by taking a larger subspace of dimension $r _ { 2 }$ such that $\sigma _ { r _ { 2 } + 1 } = \sqrt { \beta _ { r _ { 2 } + 1 } + \sigma ^ { 2 } }$ ≈ $\sigma _ { r _ { 2 } + 1 } ( X _ { n } )$ is suficiently smaller than $\sigma _ { r _ { 1 } } = \sqrt { \beta _ { r _ { 1 } } + \sigma ^ { 2 } } \approx \sigma _ { \operatorname* { m i n } } ( U _ { 1 } ^ { T } X _ { n } )$ , the estimate $[ \widehat { U } _ { 1 } , \widehat { U } _ { 2 } ]$ is likely to contain important information about the dominant signal directions $U _ { 1 }$

Remark 3.1. Since $\check { V }$ is independent of $U _ { \mathrm { ~ l ~ } } ^ { T } X _ { n }$ ,

$$
U _ { \perp } ^ { T } X _ { n } \check { V } \stackrel { d } { = } \mathrm { d i a g } ( \sigma _ { r _ { 1 } + 1 } , . . . , \sigma _ { p } ) G _ { ( p - r _ { 1 } ) \times r _ { 1 } }
$$

where $\sigma _ { j } = \sqrt { \beta _ { j } + \sigma ^ { 2 } }$ and $G _ { ( p - r _ { 1 } ) \times r _ { 1 } } \ i s \ a \ ( p - r _ { 1 } ) \times r _ { 1 }$ standard Gaussian matrix. We also have $U _ { 1 } ^ { T } X _ { n } \stackrel { d } { = }$ dia $\mathsf { \imath g } \bigl ( \sigma _ { 1 } , . . . , \sigma _ { r _ { 1 } } \bigr ) G _ { r _ { 1 } \times n }$ and $U _ { \perp } ^ { T } X _ { n } \stackrel { d } { = } \mathrm { d i a g } ( \sigma _ { r _ { 1 } + 1 } , . . . , \sigma _ { p } ) G _ { ( p - r _ { 1 } ) \times n }$ . Therefore, roughly,

$$
\frac { \left\| \boldsymbol { U } _ { \perp } ^ { T } \boldsymbol { X } _ { n } \boldsymbol { \check { V } } \right\| _ { 2 } \sigma _ { \operatorname* { m i n } } ( \boldsymbol { U } _ { 1 } ^ { T } \boldsymbol { X } _ { n } ) } { \sigma _ { \operatorname* { m i n } } ( \boldsymbol { U } _ { 1 } ^ { T } \boldsymbol { X } _ { n } ) ^ { 2 } - \sigma _ { r 2 + 1 } ( \boldsymbol { X } _ { n } ) ^ { 2 } } \approx \frac { \sigma _ { r _ { 1 } } \sigma _ { r _ { 1 } + 1 } ( \sqrt { p - r _ { 1 } } + \sqrt { r _ { 1 } } ) ( \sqrt { n } + \sqrt { r _ { 1 } } ) } { ( \sqrt { n } + \sqrt { r _ { 1 } } ) ^ { 2 } \sigma _ { r _ { 1 } } ^ { 2 } - ( \sqrt { n } + \sqrt { p } ) ^ { 2 } \sigma _ { r _ { 2 } + 1 } ^ { 2 } } ,\tag{3.5}
$$

which can be made more precise by a result by Davidson and Szarek $[ 1 0 ] ,$ given that the denominator is positive. When $n  \infty$ with p fixed, the right-hand side of (3.5) goes to zero and when $p , n  \infty$ with $p / n \to c ,$ , the right-hand side of (3.5) tends to

$$
\frac { \sigma _ { r _ { 1 } } \sigma _ { r _ { 1 } + 1 } \sqrt { c } } { \sigma _ { r _ { 1 } } ^ { 2 } - ( 1 + \sqrt { c } ) ^ { 2 } \sigma _ { r _ { 2 } + 1 } ^ { 2 } }
$$

assuming that the denominator remains positive, that is, $\sigma _ { r _ { 1 } } > ( 1 + \sqrt { c } ) \sigma _ { r _ { 2 } + 1 }$

To illustrate the accuracy of our bounds, in Figure 3 we plot the error and the bounds from Theorems 3.1 and 3.2 over the data sample size n. The experiment shows that both theorems explain well the trend of the error. We note that the bounds are only informative when smaller than 1.

4. SuperPCA: subsampling to extract high-accuracy principal components from a subspace. The results of Section 3 suggest that the leading signal can often be recovered more accurately from a larger candidate subspace than from the leading sample principal component alone. In this section, we propose an algorithm to refine a large singular subspace down to the desired dimension using very limited additional data acquisition. In this way, we exploit the information in the large subspace twofold: firstly, by searching only within that subspace for good principal components and secondly, by using it to decide what coordinates of the measurements are most rich in information.

Let $\widehat { U } _ { \mathrm { c a n d } } \ \in \ \mathbb { R } ^ { p \times r _ { 2 } }$ be a candidate subspace. Our goal is to compute $r _ { 1 }$ vectors $\tilde { u } _ { 1 } , \tilde { u } _ { 2 } , \ldots , \tilde { u } _ { r _ { 1 } } \in$ $\mathrm { s p a n } ( \widehat { U } _ { \mathrm { c a n d } } )$ that approximate the leading population principal components $U _ { 1 } \in \mathbb { R } ^ { p \times r _ { 1 } }$ more accurately bthan the initial guess $\widehat { U } _ { \mathrm { c a n d } } ( : , 1 : r _ { 1 } )$ while acquiring only a small number of additional measurements.

bThe candidate subspace $\widehat { U } _ { \mathrm { c a n d } }$ may be obtained from an initial PCA computation (since, as seen in the bprevious sections, this gives a good candidate) or from prior information. For example, $\widehat { U } _ { \mathrm { c a n d } }$ may come from bprior knowledge, such as information that has been acquired from related systems or previous experiments.

![](images/5ad2af25bef904084764dcc9bd56bca3ffc1931cc79bc042c02aff7274b73839.jpg)  
Fig. 3. Error and bounds from Theorems 3.1 and 3.2. Here the data is of dimension $p = 1 0 0 0 ,$ we take $k = 1 0$ signals of strengths $\beta _ { i } = ( 1 0 0 0 - i ^ { 3 } ) / 2 0 0 + 1 ,$ and set dim $( { \hat { U } } _ { 1 } ) = 5 ,$ , dim $( [ \hat { U } _ { 1 } , \hat { U } _ { 2 } ] ) = 1 0$ . The noise level is $\sigma = 1$ . The bounds and the error were averaged over 10 runs.

Throughout this section we only assume that it contains useful information about the desired signal. The quality may be high or low; that is, sin $\Theta ( U _ { 1 } , \widehat { U } _ { \mathrm { c a n d } } ) \vert \vert$ may range from very small, say $1 0 ^ { - 1 0 }$ , to relatively blarge, for example 0.1, the latter being perhaps the more realistic situation where the signal-to-noise ratio is low.

It is worth mentioning that the most basic approach to improve the estimate for the principal component would be to take additional samples $X _ { 2 }$ and take the SVD of $[ X _ { 1 } , X _ { 2 } ]$ . Clearly this is the classical PCA strategy. Note that this approach does not output a solution in the span of $\widehat { U } _ { \mathrm { c a n d } }$ , and therefore its output can be better than sin $\dot { \Theta } ( U _ { 1 } , \widehat { U } _ { \mathrm { c a n d } } ) | |$ b. We shall see however that in many cases SuperPCA outperforms bPCA in terms of the number of measurements needed, especially when $p \gg r _ { 2 }$

4.1. Rayleigh-Ritz PCA. Assume we take N additional measurements from model (1.1) and collect them in a data matrix $X _ { 2 }$ . To estimate the leading principal components, we propose to project the newly acquired data $X _ { 2 }$ onto the candidate subspace and perform PCA in that subspace, namely it sufices to perform PCA on $\widehat { U } _ { \mathrm { c a n d } } ^ { T } X _ { 2 }$ . This leads us to introducing Algorithm 4.1, which we call Rayleigh-Ritz PCA b(RR-PCA) due to its link to the Rayleigh-Ritz process [40, Ch. 11] as we explain below. To the best of our knowledge, RR-PCA is novel and may merit further investigation in its own right.

Algorithm 4.1 RR-PCA. Inputs: initial candidate subspace $\widehat { U } _ { \mathrm { c a n d } }$ (obtained from n samples or prior knowledge of   
the model), additional sample size N, and number of desired principal components r   
1: Take N additional measurements in all coordinates $X _ { 2 } \in \mathbb { R } ^ { p \times N }$   
2: Compute $\widehat { U } _ { \mathrm { c a n d } } ^ { T } X _ { 2 }$ and its SVD $\widehat { U } _ { \mathrm { c a n d } } ^ { T } X _ { 2 } = U _ { N } \Sigma _ { N } V _ { N } ^ { T }$   
3: Output the $r _ { 1 }$ leading vector(s) of $\hat { U } _ { \mathrm { c a n d } } U _ { N } .$

Applying RR-PCA to the data $X _ { 2 }$ corresponds mathematically to finding the leading left singular vectors of $\widehat { U } _ { \mathrm { c a n d } } \widehat { U } _ { \mathrm { c a n d } } ^ { T } X _ { 2 }$ , which is an orthogonal projection of $X _ { 2 }$ onto $\widehat { U } _ { \mathrm { c a n d } } .$ . Projection is the key idea of b b bsubspace methods for eigenvalue problems [43], and related randomized algorithms such as the popular Randomized SVD [18]. Equivalently, RR-PCA finds the (leading) eigenvalues and eigenvectors of the matrix $( \widehat { U } _ { \mathrm { c a n d } } \widehat { U } _ { \mathrm { c a n d } } ^ { T } X _ { 2 } ) ( \widehat { U } _ { \mathrm { c a n d } } \widehat { U } _ { \mathrm { c a n d } } ^ { T } X _ { 2 } ) ^ { T } = \widehat { U } _ { \mathrm { c a n d } } \widehat { U } _ { \mathrm { c a n d } } ^ { T } ( X _ { 2 } X _ { 2 } ^ { T } ) \widehat { U } _ { \mathrm { c a n d } } \widehat { U } _ { \mathrm { c a n d } } ^ { T } ( X _ { 2 } X _ { 2 } ^ { T } )$ , which is the projection of the covariance bmatrix $X _ { 2 } X _ { 2 } ^ { T }$ bonto $\widehat { U } _ { \mathrm { c a n d } } .$ b b b b. Therefore RR-PCA applies the classical Rayleigh-Ritz (RR) process [40, Ch. 11] bto the covariance matrix $X _ { 2 } X _ { 2 } ^ { T }$ . RR is known to be the optimal strategy for extracting approximate eigenvectors from a candidate subspace in a number of senses [40, Ch. 11]. As $N \to \infty$ , the leading eigenvectors of the sample covariance matrix $X _ { 2 } X _ { 2 } ^ { T }$ converge to the population principal components, and RR-PCA

(4.2)

converges to the best approximation of the population principal components contained in span $( \widehat { U } _ { \mathrm { c a n d } } )$

bFrom a practical viewpoint, however, RR-PCA is not very attractive because it still requires us to sample the whole dataset $X _ { 2 }$ and therefore ofers no reduction in measurement cost compared to classical PCA. <sup>1</sup> That is why we next introduce an alternative strategy that exploits coordinate sampling.

4.2. SuperPCA. Note that we can reinterpret RR-PCA as a least-squares problem. When we perform the orthogonal projection $\widehat { U } _ { \mathrm { c a n d } } \widehat { U } _ { \mathrm { c a n d } } ^ { T } X _ { 2 }$ as described above, we are efectively solving the least-squares b bproblem (with many right-hand sides)

$$
\operatorname* { m i n } _ { M \in \mathbb { R } ^ { r _ { 2 } \times N } } \| \widehat { U } _ { \mathrm { c a n d } } M - X _ { 2 } \| _ { F } .\tag{4.1}
$$

The solution to this problem is $M _ { * } = ( \widehat { U } _ { \mathrm { c a n d } } ) ^ { \dagger } X _ { 2 } = \widehat { U } _ { \mathrm { c a n d } } ^ { T } X _ { 2 }$ , where $( \cdot ) ^ { \dagger }$ denotes the Moore-Penrose pseudoinverse. Then $X _ { 2 }$ bcan be approximated by $\widehat { U } _ { \mathrm { c a n d } } M _ { * }$ b, and by taking the dominant subspace of $\widehat { U } _ { \mathrm { c a n d } } M _ { * }$ we recover the RR solution.

Note that (4.1) is a $p \times r _ { 2 }$ , highly overdetermined least-squares problem $( p \gg r _ { 2 } )$ , therefore instead of solving (4.1) exactly, we can approximate its solution with the subsampled least squares problem

$$
\operatorname* { m i n } _ { \tilde { M } \in \mathbb { R } ^ { r _ { 2 } \times N } } \| S ( \widehat { U } _ { \mathrm { c a n d } } \tilde { M } - X _ { 2 } ) \| _ { F } ,
$$

where $S \in \mathbb { R } ^ { s \times p }$ is a subsampling matrix, i.e. whose rows have only one entry equal to 1, and 0 elsewhere. Let ${ \mathcal { T } } \subseteq \{ 1 , 2 , \dotsc , p \}$ denote the indices chosen by $S .$ We require $| \mathcal { T } | = : s > r _ { 2 }$ , and usually $s \ll p .$ . This yields the SuperPCA algorithm, presented in Algorithm 4.2.

Algorithm 4.2 SuperPCA. Inputs: initial sample size $n ,$ additional sample size N, and subspace dimension $r _ { 2 }$ or   
a threshold $\tau \in ( 0 , 1 )$ . Optionally: approximate subspace $\widehat { U } _ { \mathrm { c a n d } } \in \mathbb { R } ^ { p \times r _ { 2 } } { \bf \Pi } ( \mathrm { i } \widehat { \mathrm { f } _ { \mathrm { \small ~ \widehat { U } _ \mathrm { c a n d } } } }$ is given, start from step 3).   
1: Take n measurements to obtain $X _ { 1 } \in \mathbb { R } ^ { p \times n }$   
2: Compute the SVD $X _ { 1 } = \widehat { U } \widehat { \Sigma } \hat { V } ^ { T }$ , and let $\widehat { U } _ { \mathrm { c a n d } } = \widehat { U } ( : , 1 : r _ { 2 } )$ . If $r _ { 2 }$ is not given, let $r _ { 2 }$ be smallest such that   
$\sigma _ { r _ { 2 } + 1 } ( X ) / \sigma _ { 1 } ( X ) \leq \tau .$   
3: Find “important” row indices ${ \mathcal { T } } \subseteq \{ 1 , 2 , \ldots , p \}$ of $\widehat { U } _ { \mathrm { c a n d } }$ via repeated and reweighted QRCP; see Section 4.3.   
4: Take N more measurements $\tilde { X } _ { 2 } \in \mathring { \mathbb { R } } ^ { s \times N } ,$ , only in the indices $\mathcal { Z } .$   
5: Solve the least-squares problem with Tikhonov regularization min<sub>M</sub> $\lVert \widehat { U } _ { \mathrm { c a n d } } ( \mathbb { Z } , : ) M - \tilde { X } _ { 2 } \rVert _ { F } ^ { 2 } + \lambda \lVert M \rVert _ { F } ^ { 2 }$ via the QR   
factorization of $[ \widehat { U } _ { \mathrm { c a n d } } ( \mathbb { Z } , : ) ; \sqrt { \lambda } I _ { r } ]$ (see Section 4.4).   
6: Find the economical SVD $M = \bar { U _ { M } } \Sigma _ { M } V _ { M } ^ { T }$   
7: Output the $r _ { 2 }$ leading vectors of $\widehat { U } _ { \mathrm { c a n d } } U _ { M } \in \mathbb { R } ^ { p \times r _ { 2 } }$

Solving (4.2) has two important advantages. First, it is much faster than solving (4.1), while giving solutions that are almost as good as (4.1). In fact, there is now a rich body of theory that justifies solving (4.2) often yields solutions of comparable quality to (4.1). For example, with leverage-score sampling (where S includes an importance-sampling weighting) one has $\lVert \widehat { U } _ { \mathrm { c a n d } } \tilde { M } _ { * } - X _ { 2 } \rVert _ { F } \leq ( 1 + \epsilon ) \lVert \widehat { U } _ { \mathrm { c a n d } } M _ { * } - X _ { 2 } \rVert _ { F }$ , where ǫ is bthe so-called subspace embedding constant [13], for which a typical value is say $1 / 2$ . Thus if $\widehat { U } _ { \mathrm { c a n d } }$ captures the signal well such that $\lVert \widehat { U } _ { \mathrm { c a n d } } M _ { * } - X _ { 2 } \rVert _ { F }$ bis small, the subsampled solution has a good fit with small $\lVert \widehat { U } _ { \mathrm { c a n d } } \tilde { M } _ { * } - X _ { 2 } \rVert _ { F }$ b. Solving heavily overdetermined least-squares problems via subsampling is a fundamental bidea that has been successfully used in $\mathrm { e . g . }$ . model order reduction [8] and numerical linear algebra [26].

The second advantage of SuperPCA is much more fundamental. To get the solution of (4.2), SuperPCA only needs new measurements on the sampled coordinates $\mathcal { T } .$ . Consequently, the measurement cost is reduced by approximately a factor $p / s \gg 1$ . Indeed, we are assuming that taking measurements at a subset $| \mathcal { T } | = s$ of indices can be done with cost proportional to $s .$ Such situation is common in $\mathrm { e . g . }$ model order reduction [8], nonetheless it is important to acknowledge that this is an assumption rather than a fact, and in some applications it may be equally expensive to obtain $S X _ { 2 }$ as it is to obtain $X _ { 2 }$

The remainder of this section discusses two practical ingredients of SuperPCA: the choice of sampled coordinates $\mathcal { T } ,$ and the regularization used in solving the sketched least-squares problem.

## 4.3. Selecting subsampling indices.

4.3.1. QRCP with iterative reweighting. A number of algorithms are available for choosing a subset of columns (or rows) that are “important” from a given matrix. These include classical algorithms in numerical linear algebra such as QR with column pivoting (QRCP) or Gaussian elimination with row pivoting [17, Ch. 3,5], [12]. Simpler methods include uniform random sampling or sampling based on the column norms; however they are known to fail for dificult problems. Another option is the use of leverage score sampling [33] for which extensive research has been performed. Other choices include the randomly pivoted Cholesky algorithm [9] and determinantal point process [30].

The method we advocate here is motivated by the observation that QRCP combines excellent performance and speed in practice [12, 15], despite the fact that in the worst case it can be exponentially far from the optimal choice of indices. We adopt an iterative and reweighting process introduced in [39], which selects the rows as follows: first perform QRCP as usual to obtain the $r _ { 2 }$ indices $\mathcal { T } ,$ , then compute the SVD of the submatrix $\widehat { U } _ { \mathrm { c a n d } } ( { \mathcal { T } } , : ) = U _ { \mathcal { T } } S _ { \mathcal { T } } V _ { \mathcal { T } } ^ { T }$ . We then compute the matrix $\widehat { U } _ { \mathrm { c a n d } } ^ { ( 2 ) } : = \widehat { U } _ { \mathrm { c a n d } } ( \mathbb { Z } ^ { C } , : ) \overline { { V _ { \mathbb { Z } } S _ { \mathbb { Z } } ^ { - 1 } } }$ , where $\mathcal { T } ^ { C }$ bdenotes the complement indices of in $\{ 1 , 2 , \ldots , p \}$ b. We then run QRCP on $\widehat { U } _ { \mathrm { c a n d } } ^ { ( 2 ) }$ to get $r _ { 2 }$ additional bindices. This process can be repeated. The idea is that we wish to emphasize the directions that we have failed to capture so far. The $V _ { \mathcal { T } }$ -multiplication gets the matrix in the right coordinates, and the further $S _ { \mathcal { T } } ^ { - 1 }$ -multiplication promotes the directions not yet captured, as described in [39].

As shown in [39], this method leads to a relatively large value of $\sigma _ { \operatorname* { m i n } } ( \widehat { U } _ { 1 } ( \mathcal { T } , : ) )$ , which is the quantity that bcontrols the suboptimality factor of the least-squares fit [15, 26]. We observe that this method performs well, in particular often better than leverage score sampling in that it results in a larger value of $\sigma _ { \operatorname* { m i n } } ( \widehat { U } _ { 1 } ( \mathcal { T } , : ) )$ bfor the same cardinality . Comparing the accuracy achieved by SuperPCA with leverage score sampling and with the sampling method described above we observed that the above method is more robust. With leverage scores sampling, if the subsampled least squares problem is weighted by the inverse square roots of the leverage scores as is usually done, then SuperPCA often did not improve the estimate of the leading signal, however if we do not use weights then the accuracy is comparable to the subsampling method above with the condition that the candidate subspace $\widehat { U } _ { 1 }$ is accurate enough.

4.3.2. Coherence of $\widehat { U } _ { \mathrm { c a n d } }$ and number of selected coordinates. The coherence of a matrix with borthonormal columns measures how strongly its column space is aligned with the coordinate axes [22]. In particular, for $\widehat { U } _ { \mathrm { c a n d } } \in \mathbb { R } ^ { p \times r _ { 2 } }$ , we define the coherence as

$$
\mu ( \widehat { U } _ { \mathrm { c a n d } } ) = \frac { p } { r _ { 2 } } \operatorname* { m a x } _ { i = 1 , \ldots , p } \| \widehat { U } _ { \mathrm { c a n d } } ( i , : ) \| _ { 2 } ^ { 2 } .
$$

The coherence satisfies $\begin{array} { r } { 1 \leq \mu ( \widehat { U } _ { \mathrm { c a n d } } ) \leq \frac { p } { r _ { 2 } } } \end{array}$ . When $\mu ( \widehat { U } _ { \mathrm { c a n d } } )$ is close to 1, the matrix is said to be incoherent, b bmeaning that its row norms are relatively uniform and its column space is not strongly concentrated along any individual coordinate direction. In contrast, high coherence indicates that the column space is strongly aligned with a small number of coordinate directions. Coherence therefore plays an important role in determining the accuracy of a subsampled least-squares solution as an approximation to the full least squares solution [14]. In SuperPCA, the coherence of $\widehat { U } _ { \mathrm { c a n d } }$ consequently influences the number of rows s bthat must be selected to obtain an accurate approximation, as we demonstrate in Section 5.

## 4.4. Regularization.

4.4.1. Tikhonov regularization. We suggest to include Tikhonov regularization in solving the least squares problem (4.2). The regularized solution is then obtained by solving the least squares problem

$$
\operatorname* { m i n } _ { M \in \mathbb { R } ^ { r _ { 2 } \times N } } \| ( S \widehat { U } _ { \mathrm { c a n d } } ) M - S X _ { 2 } \| _ { F } ^ { 2 } + \lambda \| M \| _ { F } ^ { 2 } .\tag{4.3}
$$

We found experimentally that regularization substantially improves the robustness of SuperPCA when the signal-to-noise ratio is moderate or when the candidate subspace dimension is chosen larger than the number of dominant signals. The regularization parameter λ is selected automatically using the classical L-curve criterion [6, sec. 3.6.4][20], which balances the residual norm against the solution norm. In practice, we maximize the curvature of the L-curve (whose analytic expression can be computed as shown in [19]) using MATLAB’s function fminbnd. Other automatic regression parameter rules (for example GCV) are possible.

4.4.2. Regularization and choice of the dimension of the candidate subspace. In real appli cations, the number of signals k is unknown, therefore to define the size $r _ { 2 }$ of the candidate subspace in SuperPCA the number of signals k needs to be estimated. Classical approaches to estimate the number of principal components include heuristic methods such as the scree plot [7] and explained-variance thresholds, statistical approaches such as Horn’s parallel analysis [21], and information-theoretic criteria such as AIC and MDL [47, 36]. In high-dimensional statistics, the data is often modeled by (1.1) and several practical methods to estimate the number of components were derived using random matrix theory [36, 29]. These methods are fundamentally linked to the BBP phase transition [4].

In our experiments we have set dim $( \widehat { U } _ { \mathrm { c a n d } } ) = r _ { 2 } = k$ . Assuming the data follows the spiked covariance bmodel, if the signal strengths are well above the noise $\beta _ { i } > \sigma ^ { 2 }$ then the singular values of the initial sample $X _ { 1 }$ that we use to obtain $\hat { U } _ { \mathrm { c a n d } }$ have an important spectral gap $\sigma _ { 1 } - \sigma _ { k + 1 }$ . Note that using $r _ { 2 }$ large may be bunnecessary if only the first principal component is desired, however it improves the guarantees provided in our theoretical results and experiments confirm that it is not harmful to take $r _ { 2 }$ k for example.

On the other hand, if one wishes to approximate many principal components then SuperPCA may benefit from taking $r _ { 2 } > k$ . Indeed, similarly to the situation in Figure 2, the sample components $i > k$ may contain information about some desired population principal components. In this situation, regularization plays an important role: without Tikhonov regularization if $r _ { 2 } > k$ , if the signal is weak $( \mathrm { e . g . ~ } \ \beta _ { 1 } \approx \sigma ^ { 2 } )$ or n is small $( \mathrm { e . g . ~ } n = 3 0 0$ for $p = 1 0 0 0 , k = 1 0 , \beta _ { i } = ( 1 0 0 0 - i ^ { 3 } ) / 2 0 0 + 1 )$ then SuperPCA gives an inaccurate principal component. Indeed, when $r _ { 2 } > k ,$ , we observe that the subsampled $S \widehat { U } _ { \mathrm { c a n d } }$ has additional singular values that are considerably smaller than 1, so taking the pseudo-inverse of $S \hat { U } _ { \mathrm { c a n d } }$ may amplify the noise in the data $X _ { 2 }$ b. In this situation Tikhonov regularization with the L-curve method for the choice of regularization parameter ensures that SuperPCA is robust and improves the estimation of the principal component.

4.5. SuperPCA vs. Sparse PCA. SuperPCA is related to sparse PCA [51, 24, 50, 27, 31], but the underlying assumptions are diferent. Sparse PCA assumes that the population principal components are sparse in the given coordinate system and seeks to recover their support together with the principal components themselves. By contrast, SuperPCA makes no sparsity assumption on the columns of $U _ { 1 }$ Instead, it assumes the availability of a candidate subspace containing the desired signals and uses this subspace to select informative coordinates for subsequent measurements.

When the population principal components are sparse or approximately sparse, the candidate subspace is naturally concentrated on a small number of coordinates. In this case, SuperPCA selects essentially the same coordinates as sparse PCA [24] and therefore benefits from similar reductions in measurement cost. However, SuperPCA remains applicable even when the signals are not sparse, provided the candidate subspace is suficiently informative.

Unlike iterative sparse PCA methods such as the truncated power method [50], GPower [27], or sparse Oja’s algorithm [31], SuperPCA selects the measurement coordinates only once and then refines the principal components by solving a sketched least-squares problem. Nevertheless, the power iteration algorithm is tightly connected to RR [40, Ch. 11], therefore we acknowledge the similarity between these methods and SuperPCA.

5. Numerical experiments. In this section we illustrate the behavior of the SuperPCA algorithm through numerical experiments, first on synthetic data then on a real data set. Using synthetic data, we test the SuperPCA algorithm for diferent distributions of the signal strength $\beta _ { i }$ and diferent structures of the population principal components $U _ { 1 }$ . Further, we illustrate that splitting optimally the budget in terms of the number of measurements between the initial data $X _ { 1 }$ and the subsampled data $S X _ { 2 }$ is not straightforward and depends notably on the distribution of the signal.

Synthetic experiments set-up. In the experiments with synthetic data, we define the matrix $U _ { 1 }$ containing the population principal components as the orthogonal factor of the thin QR decomposition of the matrix $[ C _ { \mathrm { b i g } } \times G _ { 1 } ; G _ { 2 } ]$ , where $G _ { 1 } \in \mathbb { R } ^ { n _ { \mathrm { b i g } } \times k }$ and $G _ { 2 } \in \mathbb { R } ^ { ( p - n _ { \mathrm { b i g } } ) \times k }$ are Gaussian matrices, $n _ { \mathrm { b i g } } \leq p$ and $C _ { \mathrm { b i g } }$ is a real number larger than 1 (whose value is fixed in the experiments below). We generate the sample $X _ { 1 }$ as in the model (1.1), with dimension $p = 1 0 0 0$ and $k = 1 0$ signals, and we subsample the follow up data X $X _ { 2 }$ only at s rows. Unless stated otherwise, we set the noise level to $\sigma = 1$

5.1. Refinement of the leading principal component starting from a candidate subspace $\widehat { U } _ { 1 }$ . First, we investigate the efect of additional measurements on the accuracy of the approximate leading bprincipal component computed with SuperPCA. We consider a fixed candidate subspace $\hat { U } _ { 1 } \in \mathbb { R } ^ { p \times r }$ obtained from applying PCA to a synthetic sample $X _ { 1 }$ b(following (1.1) with the parameters defined above) of size $n = 1 0 p$ and retaining the $r = 1 0$ leading principal components of $X _ { 1 }$ . We look at the error sin $\theta ( u _ { 1 } , u ^ { * } )$ as the number of columns N of $S X _ { 2 }$ increases. We denote $\hat { u } _ { 1 } = \widehat { U } _ { 1 } ( : , 1 )$ the initial guess. In the figures, bwe compare the error from SuperPCA with that of RR-PCA and classical PCA for the same number of measurements. Therefore, since classical PCA and RR-PCA use the entire data vectors, they are applied to data matrices that have $s N / p$ columns instead of N columns, which is indicated in the legends by “small sample”. The data $X _ { 1 }$ used to compute $\widehat { U } _ { 1 }$ is also included in the subsampled data $S X _ { 2 }$

In practice the subspace $\widehat { U } _ { 1 }$ bcould be obtained from prior knowledge of the system or a previous combputation of the principal components. For example, in the context of model order reduction [8], PCA is performed on a snapshot of states of the system to find a small dimensional subspace that capture well the dynamics of the system. If the system evolves, then this subspace may be updated using SuperPCA with the new snapshot in place of $X _ { 2 }$

5.1.1. Efect of the signal distribution. We first show how SuperPCA performs for diferent distributions of the signal strengths on three examples. We define the matrix of population principal components $U _ { 1 }$ as set up above, with $C _ { \mathrm { b i g } } ~ { = } ~ 1 0 0 0$ and $n _ { \mathrm { b i g } } = 2 0$ . In this way the signals are rather close to being sparse, in the sense that most components in $U _ { 1 }$ are very small compared to the components of the 20 most important rows. We find that since 20 rows of $\hat { U } _ { 1 }$ are much larger than the rest, subsampling $s = 2 0$ rows of $X _ { 2 }$ bis suficient and gives the best accuracy in SuperPCA for a given value of $N .$ . Therefore we choose to subsample $s = 2 r = 2 0$ rows in SuperPCA.

We experiment with three signal distributions:

Slowly decaying leading signal strengths (Figure 4a): we define $\beta _ { i } = ( 1 0 0 0 - i ^ { 3 } ) / 2 0 0 + 1$ , such that the three leading signals have nearly the same strength.

Uniform signal strengths (Figure 4b): we define the signal strengths uniformly in the interval $[ 2 , 5 ]$

Exponentially decaying signal strengths (Figure 4c): we define $\beta _ { i } = 1 0 e ^ { - i } + 1$

In Figure 4 we can see the gap between the error in the initial guess $\hat { u } _ { 1 }$ (blue dotted line) and the best candidate in span $( \widehat { U } _ { 1 } )$ (orange line). In the first two cases the gap is considerable and SuperPCA improves bthe accuracy significantly as $N  \infty$ . Remarkably, for the same number of measurements, SuperPCA is more accurate than RR-PCA and than classical PCA. A consequence is for example that in Figure 4b for the same accuracy, SuperPCA requires almost 100 less measurements than $\mathrm { R R - P C A }$ . Finally, in the example with the exponentially decaying signal strengths, the best candidate in span $( \widehat { U } _ { 1 } )$ is not much more accurate bthan the initial guess and the SuperPCA and RR-PCA algorithms do not improve the estimation faster than the classical PCA algorithm. Since the accuracy that SuperPCA and RR-PCA can achieve is limited by the choice of the search space $\widehat { U } _ { 1 }$ , in Figure 4b and Figure 4c the classical PCA algorithm eventually reaches a bhigher accuracy than the former. In all the cases above, SuperPCA finds the best candidate for $u _ { 1 }$ in the subspace span $( \widehat { U } _ { 1 } )$ after about $N = 1 0 ^ { 6 }$ samples.

bHence, comparing the three examples, we see that SuperPCA is particularly efective when the leading signals have about the same strength and the strength of the following components have a fast decay. We can explain this with the same intuition as in the introduction.

In the first case, since $\beta _ { 1 }$ and $\beta _ { 2 }$ are close to each other (both approximately equal to 5.9), we expect that the initial $\hat { u } _ { 1 }$ and $\hat { u } _ { 2 }$ both contain large components of the exact $u _ { 1 }$ and $u _ { 2 }$ (as explained in the introduction), therefore, for n large enough we expect to have a very good best guess of $u _ { 1 }$ in the subspace span $( \widehat { U } _ { 1 } )$ although the initial $\hat { u } _ { 1 }$ may be quite inaccurate.

bConversely, for exponentially decaying signal strengths, the first and second signal have very diferent strengths $\left( \beta _ { 1 } > \beta _ { 2 } \right)$ , so the classical PCA algorithm already approximates the leading principal component very well, and SuperPCA does not significantly improve on $\hat { u } _ { 1 }$ when we approximate $u _ { 1 }$ alone. However if we wished to approximate signals that are closer to the point where the singular values of the covariance matrix flatten out, for example if we estimate span $( [ u _ { 1 } , \ldots , u _ { 5 } ] )$ , then SuperPCA with a candidate subspace of dimension $r > 5$ could lead to larger improvements in the accuracy, similarly to the case with slowly decaying signal strengths.

![](images/8e2a1955e09469573ca211fbbebb38c72c26c8d1013db0cf75682efa3a9d9ffc.jpg)  
(a)

![](images/8512d3129688c985347abaf795c6b96ab6e6c1f2d523b74df6bf5312f38bc641.jpg)  
(b)

![](images/d96ca6c6b5ed5cd1a9a1dd20237049665ff3a34cbf34f51aadaa72d825c69f1f.jpg)  
(c)  
Fig. 4. Error in the approximation of the leading principal component u<sub>1</sub> for diferent signal distributions. We average the error over 20 runs and also show the standard deviation for each method.

It is also worth noting that the RR-PCA method with few data samples is as accurate as classical PCA based on the same $X _ { 2 }$ . Their cost is respectively $\mathcal { O } ( n r ^ { 2 } + r n p )$ and $\mathcal { O } ( \operatorname* { m i n } ( n p ^ { 2 } , n ^ { 2 } p ) )$ , so RR-PCA is cheaper and might be a relevant method for subspace tracking in itself.

5.1.2. Efect of the coherence of $\widehat { U } _ { 1 }$ . We now discuss the influence of the coherence of $U _ { 1 }$ on the baccuracy of the SuperPCA algorithm. In these experiments, we fix the signal strengths to $\beta _ { i } = ( 1 0 0 0 -$ ${ i ^ { 3 } } ) / { 2 0 0 + 1 } , \forall i = 1 , . . . , 1 0$ , and we define $U _ { 1 }$ as previously, with $n _ { \mathrm { b i g } } = 2 0$ rows that are weighted by $C _ { \mathrm { b i g } }$ . We consider three cases for the value of $C _ { \mathrm { b i g } } \mathrm { : }$ 1000, 100 and 10, which lead to diferent values of the coherence of $U _ { 1 }$ . For $C _ { \mathrm { b i g } } = 1 0 0 0$ and $C _ { \mathrm { b i g } } = 1 0 0$ we sample $s = 2 0$ rows of $X _ { 2 }$ in SuperPCA, while for $C _ { \mathrm { b i g } } = 1 0$ we choose to sample $s = 4 0 0$ rows. The results are presented in Figures 5a, 5b and 5c. In the following discussion, when $C _ { \mathrm { b i g } } \gg 1$ we call the signal nearly sparse, in the sense that there are a few dominant coeficients in $U _ { 1 }$ (and in $\widehat { U } _ { 1 } )$ and the others are near zero.

![](images/d2d32289201e4c2059c5d5132a14271db537d1d244454f6a5ff4e20364797450.jpg)  
(a) C<sub>big</sub> = 1000, s = 20

![](images/c411a0da6f09d221f5454b61f989f64507c6feac65b850fb122717aa8bc484fc.jpg)  
(b) $C _ { \mathrm { b i g } } = 1 0 0 , s = 2 0$

![](images/02320d39d13cfe1bdf231a0744042dce7828e7b010b3cfdec45e2d2552898c96.jpg)  
(c) C<sub>big</sub> = 10, s = 400  
Fig. 5. Error in the approximation of the leading principal component u<sub>1</sub> for diferent structures of the signal. We average the error over 20 runs and also show the standard deviation for each method.

In every figure we see a considerable gap between the accuracy of the initial candidate and that of the best estimate of $u _ { 1 }$ in $\operatorname { s p a n } ( \widehat { U } _ { 1 } )$ . In the cases where $C _ { \mathrm { b i g } } = 1 0 0 0$ and $C _ { \mathrm { b i g } } = 1 0 0$ , i.e. in Figures 5a and $5 \mathrm { b } ,$ bthe accuracy of SuperPCA improves as $N \to \infty$ , coming close to the best accuracy sin $\theta ( u _ { 1 } , \widehat { U } _ { 1 } )$ , however we see that when $C _ { \mathrm { b i g } } = 1 0 0$ the accuracy stagnates before reaching sin $\theta ( u _ { 1 } , \widehat { U } _ { 1 } )$ b. In these cases, SuperPCA is bmuch more accurate than RR-PCA and PCA for the same number of measurements. On the other hand, when $C _ { \mathrm { b i g } } = 1 0$ , the error in SuperPCA stagnates without improving much compared to the initial guess, while RR-PCA and classical PCA improve the accuracy as $N \to \infty$

These experiments show that for nearly sparse signal directions SuperPCA improves the accuracy considerably compared to PCA and RR-PCA for the same number of measurements. Figures 5a and 5b illustrate that the signal is well captured even when subsampling only $s = 2 0$ rows of $X _ { 2 }$ . Remarkably, we observe that for the same accuracy SuperPCA requires ten times fewer measurements than $\mathrm { R R - P C A }$

![](images/54d9bed59e44c5eb8e01f6137abd1413d6886fca94760a9bbbafab1e8c0d1f84.jpg)  
Fig. 6. Error for small noise $\sigma = 1 0 ^ { - 2 }$ . We chose uniform signal strengths in the interval [2, 5] and the signal directions $U _ { 1 }$ are Haar distributed (i.e. $C _ { \mathrm { b i g } } = 1 )$

On the other hand, when $C _ { \mathrm { b i g } } = 1 0$ , the matrix $U _ { 1 }$ is less coherent, so the error in SuperPCA stagnates and is eventually overtaken by RR-PCA and classical PCA, even if we take $s = 4 0 0$ in Figure 5c. It is relevant to note that RR-PCA still performs well, which means that $\widehat { U } _ { 1 }$ is a good candidate subspace that contains a good approximation $u ^ { * } \in \mathrm { s p a n } ( \widehat { U } _ { 1 } )$ b but this approximation cannot be attained by subsampling only $s = 4 0 0$ brows. Additional experiments indicate that even when only a few rows of $U _ { 1 }$ are very small, we need to sample all the other rows for SuperPCA to be accurate. A possible explanation for this limitation of the algorithm is that $\sigma _ { \mathrm { m i n } } ( S \widehat { U } _ { 1 } )$ is small if we don’t sample all the important rows. For example in the case $C _ { \mathrm { b i g } } = 1 0$ , for $s = 2 0$ b we get $\sigma _ { \mathrm { m i n } } ( S \widehat { U } _ { 1 } ) = 0 . 3$ , and for $s = 4 0 0$ we get approximately $\sigma _ { \operatorname* { m i n } } ( S \widehat { U } _ { 1 } ) = 0 . 8$

b bHowever, if the noise level is very small (or equivalently, the signal is very strong) then SuperPCA improves the accuracy of the sample principal component even if the signal directions $U _ { 1 }$ are incoherent. We show this in Figure 6, where we defined $U _ { 1 }$ as Haar distributed and the noise level is set to $\sigma =$ $1 0 ^ { - 2 }$ . It is worth mentioning that for this situation the SuperPCA algorithm performs better without regularization. The L-curve method used in the other experiments over-regularizes the solution, leading to a poor approximation of $u _ { 1 }$

Therefore we conclude that, for moderate to weak signals, SuperPCA is very eficient for nearly sparse signal directions (i.e. when $U _ { 1 }$ has only a few large components and the other components are close to zero), while for very strong signals SuperPCA without regularization is eficient even when the signal directions are dense and incoherent.

5.2. Allocation of the available budget in terms of the number of measurements. The main strength of our method is that once a candidate singular subspace $\widehat { U } _ { 1 }$ is identified, we do not need to sample the entire data vectors in the follow $\mathrm { u p }$ data $X _ { 2 } .$ b, which reduces considerably the cost of the measurements. Therefore one may wonder what portion of the data acquisition budget $B = n p + s N$ should be allocated to acquiring $X _ { 1 }$ and $S X _ { 2 }$ (the subsampled version of $X _ { 2 } )$ to achieve optimal accuracy.

In Figure 7, we looked at the error in the approximation of $u _ { 1 }$ over the portion $\alpha : = n p / B$ of the budget that is allocated to the initial PCA sample $X _ { 1 }$ for two values of the fixed total budget $B = 1 0 ^ { 6 } , 1 0 ^ { 8 }$ , for $\beta _ { i }$ uniformly distributed between 2 and $5 ,$ then we also included the results obtained with $B = 1 0 ^ { 8 }$ and $\beta _ { i } = ( 1 0 0 0 - i ^ { 3 } ) / 2 0 0 + 1$ . We set dim $( \widehat { U } _ { 1 } ) = 1 0$ and subsample $s = 4 0$ rows of $X _ { 2 }$ in SuperPCA. The exact $U _ { 1 }$ is defined as set up above with $n _ { \mathrm { b i g } } = 2 0$ rows weighted by $C _ { \mathrm { b i g } } = 1 0 0 0$ . For each method for approximating $u _ { 1 }$ , we averaged the error over 50 runs and included the standard deviation.

In Figure 7 we see that for a fixed total budget, the accuracy of the SuperPCA output first improves as α increases, until it reaches a minimum point, then it worsens as α gets closer to 1.

Figures 7a and 7b show that for uniformly distributed signal strengths the optimal split is to allocate

![](images/11279da5e8cd98c4ed87f35b926d3e53c7118315db2fe8caa55289b97fb21321.jpg)  
(a) $B = 1 0 ^ { 6 }$ , uniform $\beta _ { i }$

![](images/c13069c829c823db7288c5796a4522bd73d95d6c0b693453f288354b7aef036f.jpg)  
(b) $B = 1 0 ^ { 8 }$ , uniform $\beta _ { i }$

![](images/15de992929202e41a9bcc72745d2c70a585e97b1b039ce92306a5236d38cc378.jpg)  
(c) $B = 1 0 ^ { 8 }$ , β<sub>i</sub> = (1000 − i<sup>3</sup>)/200 + 1

Fig. 7. Error sin $\theta ( u _ { 1 } , \hat { u } _ { 1 } )$ over the portion α of the total budget B allocated to acquiring the data $X _ { 1 }$ that is used to obtain $\widehat { U } _ { 1 }$

about 70 to 80% of the measurements to the initial sample $X _ { 1 }$ and the remainder to acquiring the subsampled $S X _ { 2 }$ for SuperPCA. This still means that the matrix $S X _ { 2 }$ is very wide.

However, we observed that the optimal α depends on the distribution of the signal strengths, in particular on the spectral gap $\beta _ { 1 } - \beta _ { 2 }$ . If the gap is larger the optimal α is closer to 1, i.e. a larger portion of the budget should be invested in the candidate subspace $\widehat { U } _ { 1 }$ , while if the gap is smaller the optimum α is closer to zero. For example in Figure $\mathrm { 7 c . }$ , the gap $\beta _ { 1 } - \beta _ { 2 }$ bis smaller, which gives a smaller value of the optimal α for a total budget $B = 1 0 ^ { 8 }$

5.3. Application of SuperPCA to MNIST dataset. Finally, we applied the SuperPCA algorithm to the MNIST digits dataset [32]. This dataset contains grayscale images of handwritten numbers from 0 to 9 of size $2 8 \times 2 8$ pixels. We used the numbers 0, 1 and 2 and defined the data samples by mixing an image of each of these numbers weighted by the strengths 10, 6 and 4 respectively. We generated in this way a matrix $X _ { \mathrm { t o t } }$ of 1 million samples of dimension $p = 2 8 ^ { 2 } = 7 8 4$ and used the leading left singular vectors of this matrix as the population principal components. In this experiment the spectrum of $X _ { \mathrm { t o t } }$ has a fast decay therefore, as commented in the example with exponentially decaying signal strengths in Section 5.1, it is a case where it is hard for SuperPCA to do better than classical PCA.

We define the search space $\widehat { U } _ { 1 }$ by applying classical PCA to another set of samples of size $n = 5 0$ and taking its $r = 3 0$ bfirst leading singular vectors. Then we use SuperPCA to approximate the leading 3 principal components, with a follow up data of size N = 233 generated in the same way as the data samples from $X _ { \mathrm { t o t } }$ . The signals $\widehat { U } _ { 1 }$ are not very sparse, therefore we choose to subsample $s = 1 0 r = 3 0 0$ rows, bwhich corresponds roughly to the number of rows that have norm above 0.05. We also use standard PCA with $N _ { 0 } = 8 9$ samples so that the number of measurements used in SuperPCA and PCA are the same. In addition both in SuperPCA and classical PCA we also used the initial sample of size n in the approximation. Finally, for comparison we also implemented Johnstone and Lu’s Sparse PCA algorithm [24]. This algorithm selects the rows of largest variance and applies PCA to the subsampled data then pads the resulting principal components with zeros in the other coordinates. We used the same number of subsampled rows in SuperPCA and in Sparse PCA and the same number of data samples. We selected the rows of largest variance based on the (small) full data sample generated for standard PCA.

This gives the approximate principal components displayed in Figure 8. Comparing them with the population principal components we see that the third component is slightly better approximated by SuperPCA than classical PCA. The error sin $\Theta ( [ u _ { 1 } , u _ { 2 } , u _ { 3 } ] , [ \hat { u } _ { 1 } , \hat { u } _ { 2 } , \hat { u } _ { 3 } ] )$ in SuperPCA is 0.35 whereas the error in PCA is 0.58 and the error in Sparse PCA is 0.55. We can see in Figure 8 that SuperPCA gives a significantly better approximation of the third principal component.

6. Conclusion. We demonstrated that in the spiked covariance model, one can find an estimate of the leading principal components far more accurately than the initial estimate obtained from PCA by exploiting a larger subspace of eigenvectors of the sample covariance matrix. Assuming the leading signals have few dominant components or that the signal is strong, our proposed algorithm, SuperPCA, eficiently finds the best estimate in that subspace. An open question is the derivation of bounds on the sine of the angle between the desired principal components and the candidate subspace in terms of $\beta _ { i } , n ,$ , and $p .$

![](images/a7e1f8112681750e3832934feb62d3418863764641f135fc23268f913d931071.jpg)

![](images/79169fb09885f0cbb879464ba1f6deab7c8bbc8bd70afb436c976ba6113e530f.jpg)  
(a) Population principal components

![](images/68b0cc17ef19f81d5dbc954d9306bc954e8aefc162e0891710e19111a909ab2c.jpg)

![](images/f329f3948016ee70e02d7ea3a89cce6770ba7014937b7a538b0757fb2f74e89b.jpg)

![](images/5860b22eb12f9721340e74f0a0f0bead2fc0512ec28237a09c3bf5d317f4c23c.jpg)  
(b) SuperPCA approximation (N = 233 samples)

![](images/246d0686d4c6bb0b00975c88700cb2be7b23cff92046db3b3cacfcdadeaa6d39.jpg)

![](images/1f83a10d5527d8f3791f09e179b12b78864e860c1e3fcc0ec08f146b54264ac8.jpg)

![](images/61d2d09b4a021be9a49664b77772ae907ff2b102f070f970d08aecc4658ebf4c.jpg)  
(c) PCA approximation (N<sub>0</sub> = 89 samples)

![](images/de50f246568ffffc4e32113099c34f8b5724de4f732255b15e250164f79e32e3.jpg)

![](images/5fc2bc31d7ec24f39dd793707a515677ed3b9416fd11f68e269d874207cd67e4.jpg)

![](images/fbb259c3a7565ad7f0161c1a5f8f16c4bcf5c8d73739a7d0872d0f2b23172c27.jpg)

![](images/c82757594f50758bf67587acbbb315b0f15b2db7df49507259d6b9d8cdecef37.jpg)  
(d) Sparse PCA [24] approximation (N = 233 samples)  
Fig. 8. Leading 3 principal components and their estimates obtained with SuperPCA, PCA and SparsePCA.

Another question is the study of the convergence of the SuperPCA algorithm, or the complexity of the method for a desired final accuracy. For a fixed candidate subspace we observe a $\mathcal { O } ( 1 / \sqrt { N } )$ decrease of the sine of the angle between the desired population principal components and the estimated principal components (where N is the number of subsampled data vectors), with a constant factor that probably depends on some spectral gap, but we leave the theoretical proof for future research.

Additionally, we think that solving a total least square problem that accounts for noise in the matrice $S \widehat { U } _ { 1 }$ and $S X _ { 2 }$ could lead to an alternative version of SuperPCA with potential benefits in terms of robustness. bWe leave this algorithmic idea for future work.

7. Proofs of Theorems 3.1 and 3.2. In this section we give proofs of Theorems 3.1 and 3.2. Theorem 3.1 is specialized to the case when we estimate only the dominant signal, whereas Theorem 3.2 considers a subspace spanned by multiple signals. The proof of Theorem 3.1 is not a specialization of Theorem 3.2 as we show below by proving these results using diferent techniques. In the special case when $r _ { 1 } = 1$ , Theorem $3 . 2 \mathrm { : s }$ bound is weaker than Theorem 3.1’s bound when the noise, σ is large compared to the signal strengths.

The statement for Theorem 3.2 in this section is a generalization of the statement in Section 3. The generalization accommodates cases where the model has significantly smaller noise than the signal strengths.

7.1. Proof of Theorem 3.1. We first transform the scaled data matrix into a block lower triangular form using right orthogonal transformations. Note that right orthogonal transformations do not change the left singular vectors nor the singular values of the matrix.

First, since Gaussian matrices are invariant under orthogonal transformations, the scaled data matrix can be given by

$$
X _ { n } = \frac { 1 } { \sqrt { n } } \left( \sum _ { j = 1 } ^ { k } \sqrt { \beta _ { j } } g _ { j } ^ { i } u _ { j } + \sigma \eta _ { i } \right) \stackrel { d } { = } \frac { 1 } { \sqrt { n } } [ U , U _ { \perp } ] \underset { j \times p } { \underbrace { \mathrm { d i a g } } } ( \sqrt { \beta _ { 1 } + \sigma ^ { 2 } } , . . . , \sqrt { \beta _ { \mathrm { k } } + \sigma ^ { 2 } } , \sigma , . . . , \sigma ) G _ { p \times n } ,\tag{7.1}
$$

where $U = [ u _ { 1 } , . . . , u _ { k } ] \in \mathbb { R } ^ { p \times k }$ is the matrix of signal directions with $U _ { \perp }$ being any orthogonal complement of $U$ , and $G _ { a \times b }$ is an $a \times b$ Gaussian matrix with entries i.i.d. $\mathcal { N } ( 0 , 1 )$ . Here, $\circeq$ is used to denote equality in distribution. Now define

$$
\begin{array} { l } { { \widetilde { X } _ { n } : = [ U , U _ { \bot } ] ^ { T } X _ { n } \stackrel { d } { = } \frac { 1 } { \sqrt { n } } \underset { { \scriptstyle \sqrt { n } \leq n } } { \underline { { \mathrm { d i a g } } } ( \sqrt { \beta _ { 1 } + { \sigma ^ { 2 } } } , . . . , \sqrt { \beta _ { \mathrm { k } } + { \sigma ^ { 2 } } } , \sigma , . . . , \sigma ) } G _ { p \times n } } } \\ { { \mathrm { } } } \\ { { \mathrm { } = \frac { 1 } { \sqrt { n } } \underset { { \scriptstyle \sqrt { n } \times p } } { \underline { { \mathrm { d i a g } } } ( \sigma _ { 1 } , . . . , \sigma _ { \mathrm { k } } , \sigma , . . . , \sigma ) } G _ { p \times n } . } } \end{array}\tag{7.2}
$$

For shorthand let the rows of ${ \widetilde { X } } _ { n }$ be $b _ { i } = u _ { i } ^ { T } X _ { n } \in \mathbb { R } ^ { 1 \times n }$ for each $1 \leq i \leq k ,$ i.e., the indices corresponding to the signals. Then

$$
\widetilde X _ { n } = \left[ \begin{array} { c } { { b _ { 1 } } } \\ { { \vdots } } \\ { { b _ { k } } } \\ { { \frac { \sigma } { \sqrt { n } } G _ { \left( p - k \right) \times n } } } \end{array} \right] ,
$$

where $b _ { i } \sim { \mathcal { N } } ( 0 , ( \beta _ { i } + \sigma ^ { 2 } ) I _ { n } )$ . Let $Q _ { B _ { 1 } }$ be an $n \times n$ orthogonal matrix with $b _ { 1 } / \Vert b _ { 1 } \Vert _ { 2 }$ as the first column. Then right multiplication with $Q _ { B _ { 1 } }$ sets the first row of ${ \widetilde { X } } _ { n }$ to zero, except for the (1,1) element which is set to $\| b _ { 1 } \| _ { 2 }$ e. The rotation does not change the distribution of any of the other rows. Now, we use the notation $b _ { j } ^ { ( - k ) }$ to indicate the $( n - k )$ -dimensional vector with distribution $\mathcal { N } ( 0 , ( \beta _ { j } + \sigma ^ { 2 } ) I _ { n - k } )$ . The second row of $\tilde { \tilde { X } } _ { n } Q _ { B _ { 1 } } \mathrm { ~ i s ~ } [ \sqrt { \beta _ { 2 } + \sigma ^ { 2 } } \eta _ { 2 1 } ( b _ { 2 } ^ { ( - 1 ) } ) ^ { T } ] ,$ , for $\eta _ { 2 1 } \sim \mathcal { N } ( 0 , 1 )$ . We can repeat the same procedure as before, defining ean orthogonal matrix $Q _ { B _ { 2 } } = \Big [ _ { 0 } ^ { 1 } \hat { Q } _ { B _ { 2 } } \Big ]$ with the first column equal to $e _ { 1 }$ (first canonical basis vector) and the second column equal to $[ 0 \ ( b _ { 2 } ^ { ( - 1 ) } ) ^ { T } ] ^ { T }$ . Continuing this process, we obtain

$$
\begin{array} { r l } & { \tilde { X } _ { n } Q _ { B _ { 1 } } \ldots Q _ { B _ { k } } \stackrel { d } { = } \left[ \begin{array} { l l l l } { \frac { \| b _ { 1 } \| _ { 2 } } { \sqrt { \beta _ { 2 } + \sigma ^ { 2 } } \eta _ { 2 1 } } } & { \| b _ { 2 } ^ { ( - 1 ) } \| _ { 2 } } & & \\ { \vdots } & { \ddots } & \\ { \sqrt { \beta _ { k } + \sigma ^ { 2 } } \eta _ { k 1 } } & { \sqrt { \beta _ { k } + \sigma ^ { 2 } } \eta _ { k 2 } } & { \cdots } & { \| b _ { k } ^ { ( - ( k - 1 ) ) } \| _ { 2 } } \\ { \frac { \sigma } { \sqrt { n } } G _ { p - k , k } } & & & { \frac { \sigma } { \sqrt { n } } G _ { p - k , n - k } } \end{array} \right] : = \tilde { X } _ { n } ^ { ( T ) } , } \end{array}\tag{7.3}
$$

where each $\eta _ { i j } \sim \mathcal { N } ( 0 , 1 )$ is i.i.d., $G _ { i , j }$ is $i \times j$ Gaussian, and $b _ { j } ^ { - ( j - 1 ) } \in \mathbb { R } ^ { n - j + 1 }$ is a Gaussian vector with distribution $\sim \mathcal { N } ( 0 , ( \beta _ { j } + \sigma ^ { 2 } ) I _ { n - j + 1 } )$

We now use the block lower triangular form, ${ \widetilde { X } } _ { n } ^ { ( T ) }$ to prove Theorem 3.1. Note that ${ \widetilde { X } } _ { n } ^ { ( T ) }$ and ${ \widetilde { X } } _ { n }$ have the same left singular vectors and $X _ { n } , { \widetilde { X } } _ { n }$ and ${ \widetilde { X } } _ { n } ^ { ( T ) }$ all have the same singular values.

Theorem 7.1 (Theorem 3.1). Let $\begin{array} { r } { X _ { n } = \frac { 1 } { \sqrt { n } } X = \frac { 1 } { \sqrt { n } } [ x _ { 1 } , x _ { 2 } , . . . , x _ { n } ] \in \mathbb { R } ^ { p \times n } } \end{array}$ be the scaled data matrix where the data matrix X is as defined in (1.1). Then

$$
\sin \theta ( u _ { 1 } , \widehat { U } _ { 1 } ) \leq \sqrt { \frac { \sigma _ { 1 } ( X _ { n } ) ^ { 2 } - \left. u _ { 1 } ^ { T } X _ { n } \right. _ { 2 } ^ { 2 } } { \sigma _ { 1 } ( X _ { n } ) ^ { 2 } - \sigma _ { r _ { 1 } + 1 } ( X _ { n } ) ^ { 2 } } }\tag{3.3}
$$

where $u _ { 1 }$ is the dominant signal direction and $\widehat { U } _ { 1 } \in \mathbb { R } ^ { p \times r _ { 1 } }$ is the matrix containing the $r _ { 1 }$ leading left singular vectors of $X _ { n }$

Proof. First, let the SVD of ${ \widetilde { X } } _ { n } ^ { ( T ) }$ be

$$
\widetilde { X } _ { n } ^ { ( T ) } = \widetilde { U } \widehat { \Sigma } \widetilde { V } ^ { T } = \left[ \widetilde { U } _ { 1 } , \widetilde { U } _ { 2 } \right] \left[ \begin{array} { l l } { \widehat { \Sigma } _ { 1 } } & \\ & { \widehat { \Sigma } _ { 2 } } \end{array} \right] \left[ \widetilde { V } _ { 1 } , \widetilde { V } _ { 2 } \right] ^ { T }\tag{7.4}
$$

where $\begin{array} { r } { \widetilde { U } _ { 1 } \in \mathbb R ^ { p \times r _ { 1 } } , \widetilde { U } _ { 2 } \in \mathbb R ^ { p \times ( p - r _ { 1 } ) } , \widetilde { V } _ { 1 } \in \mathbb R ^ { n \times r _ { 1 } } , \widetilde { V } _ { 2 } \in \mathbb R ^ { n \times ( n - r _ { 1 } ) } , \widehat { \Sigma } _ { 1 } \in \mathbb R ^ { r _ { 1 } \times r _ { 1 } } } \end{array}$ and $\widehat { \Sigma } _ { 2 } \ \in \ \mathbb { R } ^ { ( p - r _ { 1 } ) \times ( n - r _ { 1 } ) }$ Here, $\widehat { \Sigma } _ { 1 }$ and $\widehat { \Sigma } _ { 2 }$ e eare the singular values of both ${ \widetilde { X } } _ { n } ^ { ( T ) }$ and $X _ { n }$ bwhere $\widehat { \Sigma } _ { 1 } = \mathrm { d i a g } ( \widehat { \sigma } _ { 1 } , \widehat { \sigma } _ { 2 } , . . . , \widehat { \sigma } _ { r _ { 1 } } )$ and $\widehat { \Sigma } _ { 2 } =$ $\left[ \begin{array} { c } { \mathrm { d i a g } ( \widehat { \sigma } _ { r _ { 1 } + 1 } , \widehat { \sigma } _ { r _ { 1 } + 2 } , . . . , \widehat { \sigma } _ { n } ) } \\ { 0 _ { \left( p - n \right) \times \left( n - r _ { 1 } \right) } } \end{array} \right]$ if $p > n$ and $\widehat { \Sigma } _ { 2 } \ = \ \left[ \mathrm { d i a g } ( \widehat \sigma _ { r _ { 1 } + 1 } , \widehat \sigma _ { r _ { 1 } + 2 } , . . . , \widehat \sigma _ { p } ) 0 _ { ( p - r _ { 1 } ) \times ( n - p ) } \right] \ \mathrm { i f } \ p \ \leq \ n ,$ i.e., ${ \widehat { \sigma } } _ { j } = \sigma _ { j } ( X _ { n } )$ . Furthermore let $\widehat { U }$ be the left singular vectors of $X _ { n }$ and $\widehat { U } _ { \perp }$ a complement of $\widehat { U }$ . Note that bby letting $\widehat { U } _ { \perp }$ bbe any orthogonal complement of ${ \widehat { U } } ,$ we have $[ \widetilde { U } _ { 1 } , \widetilde { U } _ { 2 } ] = [ U , U _ { \perp } ] ^ { T } [ \widehat { U } , \widehat { U } _ { \perp } ]$ b where $U _ { \perp }$ is any borthogonal complement of $U = [ u _ { 1 } , . . . , u _ { k } ]$ b e e. Therefore the left singular vectors of ${ \widetilde X } _ { n } ^ { ( n ) }$ are the left singular vectors of $X _ { n }$ up to left orthogonal transformation by $[ U , U _ { \bot } ] ^ { T }$

We now have

$$
\sin \theta ( u _ { 1 } , \widehat { U } ) = \left\| u _ { 1 } ^ { T } \widehat { U } _ { \bot } \right\| _ { 2 } = \left\| e _ { 1 } ^ { T } [ u _ { 1 } , . . . , u _ { k } , U _ { \bot } ] ^ { T } \widehat { U } _ { \bot } \right\| _ { 2 } = \sin \theta ( e _ { 1 } , [ U , U _ { \bot } ] ^ { T } \widehat { U } ) = \sin \theta ( e _ { 1 } , \widetilde { U } _ { 1 } )
$$

where $e _ { 1 }$ is the first canonical basis vector. Therefore we bound sin $\theta ( e _ { 1 } , \widetilde { U } _ { 1 } )$ instead.

We first bound cos $\theta ( e _ { 1 } , \widetilde { U } _ { 1 } ) \ = \ \left\| e _ { 1 } ^ { T } \widetilde { U } _ { 1 } \right\| _ { 2 }$ using the quantities $e _ { 1 } ^ { T } \widetilde { X } _ { n } ^ { ( T ) } \widetilde { V } _ { 1 }$ and $e _ { 1 } ^ { T } \widetilde { X } _ { n } ^ { ( T ) } \widetilde { V } _ { 2 }$ . First, for $e _ { 1 } ^ { T } \widetilde { X } _ { n } ^ { ( T ) } \widetilde { V } _ { 1 }$ we have

$$
e _ { 1 } ^ { T } \widetilde { X } _ { n } ^ { ( T ) } \widetilde { V } _ { 1 } = \| b _ { 1 } \| _ { 2 } e _ { 1 } ^ { T } \widetilde { V } _ { 1 } = \| b _ { 1 } \| _ { 2 } \widetilde { v } _ { 1 }\tag{7.5}
$$

where $\widetilde { \boldsymbol { v } } _ { 1 } \in \mathbb { R } ^ { 1 \times { r _ { 1 } } }$ is the first row of $\widetilde { V } _ { 1 }$ . On the other hand, we also have

$$
e _ { 1 } ^ { T } \widetilde { X } _ { n } ^ { ( T ) } \widetilde { V } _ { 1 } = e _ { 1 } ^ { T } \widetilde { U } _ { 1 } \widehat { \Sigma } _ { 1 } = \widetilde { u } _ { 1 } \widehat { \Sigma } _ { 1 }\tag{7.6}
$$

where $\boldsymbol { \widetilde { u } } _ { 1 } \in \mathbb { R } ^ { 1 \times r _ { 1 } }$ is the first row of $\widetilde { U } _ { 1 }$ . Therefore $\| b _ { 1 } \| _ { 2 } \widetilde { v } _ { 1 } = \widetilde { u } _ { 1 } \widehat { \Sigma } _ { 1 }$ and we get

$$
\begin{array} { r } { \left\| b _ { 1 } \right\| _ { 2 } \left\| \widetilde v _ { 1 } \right\| _ { 2 } = \left\| \widetilde u _ { 1 } \widehat \Sigma _ { 1 } \right\| _ { 2 } \leq \left\| \widetilde u _ { 1 } \right\| _ { 2 } \widehat \sigma _ { 1 } . } \end{array}\tag{7.7}
$$

Similarly for $e _ { 1 } ^ { T } \widetilde { X } _ { n } ^ { ( T ) } \widetilde { V } _ { 2 }$ we get $\Vert b _ { 1 } \Vert _ { 2 } \widetilde { v } _ { 2 } = \widetilde { u } _ { 2 } \widehat { \Sigma } _ { 2 }$ , where $\widetilde { u } _ { 2 }$ is the first row of $\widetilde { U } _ { 2 }$ and $\widetilde { v } _ { 2 }$ is the first row of $\widetilde { V } _ { 2 }$ which gives us

$$
\begin{array} { r } { \| b _ { 1 } \| _ { 2 } \| \widetilde v _ { 2 } \| _ { 2 } \le \| \widetilde u _ { 2 } \| _ { 2 } \widehat \sigma _ { r _ { 1 } + 1 } . } \end{array}\tag{7.8}
$$

Now since $\widetilde { U }$ and $\widetilde { V }$ are orthogonal matrices, we have $\left\| \widetilde { u } _ { 1 } \right\| _ { 2 } ^ { 2 } + \left\| \widetilde { u } _ { 2 } \right\| _ { 2 } ^ { 2 } = 1$ and $\| \widetilde { v } _ { 1 } \| _ { 2 } ^ { 2 } + \| \widetilde { v } _ { 2 } \| _ { 2 } ^ { 2 } = 1$ . Therefore e ewe can rewrite (7.7) as

$$
\left\| b _ { 1 } \right\| _ { 2 } ^ { 2 } \left( 1 - \left\| \widetilde v _ { 2 } \right\| _ { 2 } ^ { 2 } \right) \leq \left( 1 - \left\| \widetilde u _ { 2 } \right\| _ { 2 } ^ { 2 } \right) \widehat \sigma _ { 1 } ^ { 2 } ,\tag{7.9}
$$

which gives us

$$
\| \widetilde { u } _ { 2 } \| _ { 2 } ^ { 2 } \leq 1 - \frac { \| b _ { 1 } \| _ { 2 } ^ { 2 } } { \widehat { \sigma } _ { 1 } ^ { 2 } } \left( 1 - \| \widetilde { v } _ { 2 } \| _ { 2 } ^ { 2 } \right) \leq 1 - \frac { \| b _ { 1 } \| _ { 2 } ^ { 2 } } { \widehat { \sigma } _ { 1 } ^ { 2 } } + \frac { \| \widetilde { u } _ { 2 } \| _ { 2 } ^ { 2 } \widehat { \sigma } _ { r _ { 1 } + 1 } ^ { 2 } } { \widehat { \sigma } _ { 1 } ^ { 2 } }
$$

using (7.8) and (7.9). Finally, rearranging gives us the desired inequality,

$$
\begin{array} { r } { \sin \theta ( u _ { 1 } , \widehat { U } ) = \sin \theta ( \epsilon _ { 1 } , \widetilde { U } _ { 1 } ) = \left\| \epsilon _ { 1 } ^ { T } \widetilde { U } _ { 2 } \right\| _ { 2 } = \left\| \widetilde { u } _ { 2 } \right\| _ { 2 } \leq \sqrt { \frac { \widehat \sigma _ { 1 } ^ { 2 } - \left\| b _ { 1 } \right\| _ { 2 } ^ { 2 } } { \widehat \sigma _ { 1 } ^ { 2 } - \widehat \sigma _ { r _ { 1 } + 1 } ^ { 2 } } } = \sqrt { \frac { \sigma _ { 1 } ( X _ { n } ) ^ { 2 } - \left\| u _ { 1 } ^ { T } X _ { n } \right\| _ { 2 } ^ { 2 } } { \sigma _ { 1 } ( X _ { n } ) ^ { 2 } - \sigma _ { r _ { 1 } + 1 } ( X _ { n } ) ^ { 2 } } } . } \end{array}
$$

Extending this proof to the case where multiple signals are approximated simultaneously gives a slightly diferent bound involving the smallest singular value $\tilde { \sigma }$ of the top left $r _ { 1 } ~ \times ~ r _ { 1 }$ lower triangular block of ${ \widetilde X } _ { n } ^ { ( T ) }$ . Since this makes the numerator substantially larger, the bound does not scale well, according to our eexperiments.

7.2. Proof of Theorem 3.2. In this section, we prove Theorem 3.2 in a more general form, which scales better for small noise. We work with the matrix $\widetilde { X } _ { n } = [ U , U _ { \bot } ] ^ { T } X _ { n }$ in this section as in (7.3). Let $r _ { * }$ be a parameter defined for analysis and satisfies $r _ { 1 } \le r _ { * } \le r _ { 2 }$ (Theorem 3.2 corresponds to the case when $r _ { * } = r _ { 1 } )$ . Note that the dimensions satisfy the following relation,

$$
\begin{array} { r } { 1 \leq r _ { 1 } \leq r _ { * } \leq r _ { 2 } \leq k \leq \operatorname* { m i n } \{ p , n \} . } \end{array}
$$

Now define $\widetilde { X } _ { 1 } \in \mathbb { R } ^ { r _ { * } \times n }$ and $\widetilde { X } _ { 2 } \in \mathbb { R } ^ { ( p - r _ { * } ) \times n }$ as

$$
\widetilde { X } _ { n } = \left[ U \quad U _ { \bot } \right] ^ { T } X _ { n } = \frac { 1 } { \sqrt { n } } \left[ \begin{array} { c } { \medskip \displaystyle b _ { 1 } } \\ { \vdots } \\ { \displaystyle b _ { k } } \\ { \sigma G _ { ( p - k ) \times n } } \end{array} \right] = \left[ \widetilde { X } _ { 1 } \right] ,\tag{7.10}
$$

where $b _ { i } \sim \mathcal { N } ( 0 , ( \beta _ { i } + \sigma ^ { 2 } ) I _ { n } )$ . We use this division of the transformed data matrix $\widetilde { X } _ { n }$ in the proof.

eFor subspaces of equal dimension, an important property of the canonical angles is that sin $\Theta ( \cdot , \cdot ) \|$ is a metric on the space of k-dimensional subspaces [45]. We extend the corresponding triangle inequality to subspaces of diferent dimensions in the Lemma 7.2 below.

Lemma 7.2 (Triangle inequality for subspaces of diferent dimensions). Let $U = [ U _ { 1 } , U _ { 2 } , U _ { 3 } ] , \hat { U } =$ $[ \hat { U } _ { 1 } , \hat { U } _ { 2 } , \hat { U } _ { 3 } ]$ and $\tilde { U } = [ \tilde { U _ { 1 } } , \tilde { U } _ { 2 } , \tilde { U _ { 3 } } ]$ be n $, \times n$ orthogonal matrices where index 1 corresponds to $j$ columns, index 2 corresponds to $\ell - j > 0$ columns and index 3 corresponds to the other $n - \ell$ columns. Then

$$
\left\| \sin \Theta ( U _ { 1 } , [ \tilde { U } _ { 1 } , \tilde { U } _ { 2 } ] ) \right\| \leq \left\| \sin \Theta ( U _ { 1 } , [ \hat { U } _ { 1 } , \hat { U } _ { 2 } ] ) \right\| + \left\| \sin \Theta ( [ \hat { U } _ { 1 } , \hat { U } _ { 2 } ] , [ \tilde { U } _ { 1 } , \tilde { U } _ { 2 } ] ) \right\|\tag{7.11}
$$

and

$$
\left\| \sin \Theta ( U _ { 1 } , [ \tilde { U } _ { 1 } , \tilde { U } _ { 2 } ] ) \right\| \leq \left\| \sin \Theta ( U _ { 1 } , \hat { U } _ { 1 } ) \right\| + \left\| \sin \Theta ( \hat { U } _ { 1 } , [ \tilde { U } _ { 1 } , \tilde { U } _ { 2 } ] ) \right\|\tag{7.12}
$$

for any unitarily invariant norms.

Proof.

$$
\begin{array} { r l } { \left\| \sin \Theta ( U _ { 1 } , [ \tilde { U } _ { 1 } , \tilde { U } _ { 2 } ] ) \right\| = \left\| U _ { 1 } ^ { T } \tilde { U } _ { 3 } \right\| = \left\| U _ { 1 } ^ { T } [ \hat { U } _ { 1 } , \hat { U } _ { 2 } , \hat { U } _ { 3 } ] [ \hat { U } _ { 1 } , \hat { U } _ { 2 } , \hat { U } _ { 3 } ] ^ { T } \tilde { U } _ { 3 } \right\| } & { } \\ & { = \left\| U _ { 1 } ^ { T } [ \hat { U } _ { 1 } , \hat { U } _ { 2 } ] [ \hat { U } _ { 1 } , \hat { U } _ { 2 } ] ^ { T } \tilde { U } _ { 3 } \right\| + \left\| U _ { 1 } ^ { T } \hat { U } _ { 3 } \hat { U } _ { 3 } ^ { T } \tilde { U } _ { 3 } \right\| } \\ & { \leq \left\| [ \hat { U } _ { 1 } , \hat { U } _ { 2 } ] ^ { T } \tilde { U } _ { 3 } \right\| + \left\| U _ { 1 } ^ { T } \hat { U } _ { 3 } \right\| } \\ & { \leq \left\| \sin \Theta ( [ \hat { U } _ { 1 } , \hat { U } _ { 2 } ] , [ \tilde { U } _ { 1 } , \hat { U } _ { 2 } ] ) \right\| + \left\| \sin \Theta ( U _ { 1 } , [ \hat { U } _ { 1 } , \hat { U } _ { 2 } ] ) \right\| . } \end{array}
$$

The second result can be proved similarly.

Lemma 7.2 will be useful for proving a generalization of Theorem 3.2 below as we compare subspaces of diferent dimensions.

Theorem 7.3 (Generalization of Theorem 3.2). Let $\begin{array} { r } { X _ { n } = \frac { 1 } { \sqrt { n } } X = \frac { 1 } { \sqrt { n } } [ x _ { 1 } , x _ { 2 } , . . . , x _ { n } ] \in \mathbb { R } ^ { p \times n } } \end{array}$ be the scaled data matrix where the data matrix X is as defined in (1.1). Then

$$
\begin{array} { r } { \left\| \sin \Theta ( U _ { 1 } , [ \widehat { U } _ { 1 } , \widehat { U } _ { 2 } ] ) \right\| _ { 2 } \leq \operatorname* { m i n } _ { r _ { * } \in [ r _ { 1 } , r _ { 2 } ] } \frac { \left\| U _ { * , \bot } ^ { T } X _ { n } \check { V } \right\| _ { 2 } \sigma _ { \operatorname* { m i n } } ( U _ { * } ^ { T } X _ { n } ) } { \sigma _ { \operatorname* { m i n } } ( U _ { * } ^ { T } X _ { n } ) ^ { 2 } - \sigma _ { r _ { 2 } + 1 } ( X _ { n } ) ^ { 2 } } , } \end{array}\tag{7.13}
$$

where $U _ { * }$ is the $r _ { * }$ leading signal directions with $U _ { * , \bot }$ as its orthogonal complement, $[ \widehat { U } _ { 1 } , \widehat { U } _ { 2 } ]$ is the leading $r _ { 2 }$ left singular values of the data matrix and $\check { V }$ is the leading $r _ { 1 }$ bright singular vectors of $U _ { 1 } ^ { T } X _ { n } ,$ which is independent of $U _ { * , \bot } ^ { T } X _ { n }$

Proof. Let $\widehat { U } _ { \perp }$ be an orthogonal complement of $[ \widehat { U } _ { 1 } , \widehat { U } _ { 2 } ]$ and $[ \widetilde { U } _ { 1 } , \widetilde { U } _ { 2 } , \widetilde { U } _ { \perp } ] \ \in \ \mathbb { R } ^ { p \times p }$ be the orthogonal bmatrix of left singular vectors of ${ \widetilde { X } } _ { n }$ where $\widetilde { U } _ { 1 } \in \mathbb { R } ^ { p \times \bar { r } _ { 1 } } , \ \widetilde { U } _ { 2 } \in \mathbb { R } ^ { p \times \bar { ( } r _ { 2 } - r _ { 1 } ) }$ eand $\widetilde { U } _ { \perp } \in \mathbb { R } ^ { p \times ( p - r _ { 2 } ) }$ . Note that $[ \widetilde { U } _ { 1 } , \widetilde { U } _ { 2 } , \widetilde { U } _ { \perp } ] = [ U _ { 1 } , U _ { \perp } ] ^ { T } [ \widehat { U } _ { 1 } , \widehat { U } _ { 2 } , \widehat { U } _ { \perp } ]$ . Then

$$
\begin{array} { r l } & { \left\| \sin \Theta ( U _ { 1 } , [ \widehat { U } _ { 1 } , \widehat { U } _ { 2 } ] ) \right\| _ { 2 } = \left\| U _ { 1 } ^ { T } \widehat { U } _ { \perp } \right\| _ { 2 } = \left\| [ I _ { r _ { 1 } } , 0 ] [ U _ { 1 } , U _ { \perp } ] ^ { T } \widehat { U } _ { \perp } \right\| _ { 2 } } \\ & { \qquad = \left\| [ I _ { r _ { 1 } } , 0 ] \widetilde { U } _ { \perp } \right\| _ { 2 } = \left\| \sin \Theta \left( \left[ I _ { 0 } \right] , [ \widetilde { U } _ { 1 } , \widetilde { U } _ { 2 } ] \right) \right\| . } \end{array}
$$

Let the SVD of $\widetilde { X } _ { 1 } \in \mathbb { R } ^ { r _ { * } \times n }$ be $\tilde { X } _ { 1 } = \check { U } \check { \Sigma } \check { V } ^ { T }$ and let $\check { V } _ { \perp } \in \mathbb { R } ^ { n \times ( n - r _ { * } ) }$ be an orthogonal complement of $\check { V }$ eNow, form the following matrix,

$$
\check { X } _ { n } = \left[ \begin{array} { c c } { \check { U } } & { 0 } \\ { 0 } & { I _ { ( p - r _ { * } ) } } \end{array} \right] ^ { T } \left[ \begin{array} { c c } { \widetilde { X } _ { 1 } } \\ { \widetilde { X } _ { 2 } } \end{array} \right] \left[ \check { V }  &  \check { V } _ { \perp } \right] = \left[ \begin{array} { c c } { \check { \Sigma } } & { 0 } \\ { \widetilde { X } _ { 2 } \check { V } } & { \widetilde { X } _ { 2 } \check { V } _ { \perp } } \end{array} \right] \in \mathbb { R } ^ { p \times n } .\tag{7.14}
$$

Note that $\check { V }$ and $\check { V } _ { \perp }$ are only dependent on $\widetilde { X } _ { 1 }$ and therefore $\check { V }$ and $\check { V } _ { \perp }$ are independent of ${ \widetilde { X } } _ { 2 }$ as $\widetilde { X } _ { 1 }$ and $\widetilde { X } _ { 2 }$ eare independent. We also note that the singular values of $\check { X } _ { n } , \widetilde { X } _ { n }$ and $X _ { n }$ eare the same because ${ \check { X } } _ { n }$ and $ { \widetilde { X } } _ { n }$ are orthogonal transformations of $X _ { n }$

Now let us note that

$$
\begin{array}{c} \begin{array} { l } { \mathrm { s i n } \Theta \left( \left[ \overset { \sim } { \underset { 0 } { U } } \right] , \left[ \widetilde { U } _ { 1 } , \widetilde { U } _ { 2 } \right] \right) = \mathrm { s i n } \Theta \left( \left[ \overset { \sim } { U } \begin{array} { c c } { \textnormal { \ i } } { 0 } \\ { 0 } & { I _ { ( p - r _ { * } ) } } \end{array} \right] ^ { T } \left[ \overset { \sim } { U } \right] , \left[ \overset { \sim } { U } \begin{ c c } { 0 } & { 0 } \\ { 0 } & { I _ { ( p - r _ { * } ) } } \end{array} \right] ^ { T } \left[ \widetilde { U } _ { 1 } , \widetilde { U } _ { 2 } \right] \right) } \\ { = \mathrm { s i n } \Theta \left( \left[ I _ { r _ { * } } \right] , \left[ \check { U } _ { 1 } , \check { U } _ { 2 } \right] \right) } \end{array}
$$

where $\check { U } _ { 1 }$ is the leading $r _ { 1 }$ left singular vectors of ${ \check { X } } _ { n }$ and $\check { U } _ { 2 }$ is the next $\left( r _ { 2 } - r _ { 1 } \right)$ leading left singular vectors. Now, using the triangle inequality, Lemma 7.2, we obtain

$$
\begin{array} { r l } & { \left\| \sin \theta \left( \left[ I _ { r _ { 1 } } \right] , [ \widetilde { U } _ { 1 } , \widetilde { U } _ { 2 } ] \right) \right\| _ { 2 } \leq \left\| \sin \theta \left( \left[ I _ { r _ { 1 } } \right] , \left[ \widetilde { U } \right] \right) \right\| _ { 2 } + \left\| \sin \theta \left( \left[ \widetilde { U } \right] , \left[ \widetilde { U } _ { 1 } , \widetilde { U } _ { 2 } \right] \right) \right\| _ { 2 } } \\ & { \qquad = \left\| \sin \Theta \left( \left[ I _ { r _ { * } } \right] , [ \widetilde { U } _ { 1 } , \widetilde { U } _ { 2 } ] \right) \right\| _ { 2 } } \end{array}
$$

since

$$
\begin{array} { r } { \left\| \sin \theta \left( \left[ I _ { r _ { 1 } } \right] , \left[ \overset { \vartriangle } { 0 } \right] \right) \right\| _ { 2 } = \left\| \left[ I _ { 0 } \right] ^ { T } \left[ I _ { p - r _ { * } } \right] \right\| _ { 2 } = 0 } \end{array}\tag{7.15}
$$

because $\check { U } \in \mathbb { R } ^ { r _ { * } \times r _ { * } }$ is an orthogonal matrix and $r _ { 1 } \leq r _ { * }$ . Therefore, it sufices to bound

$$
\left\| \sin \theta \left( \left[ I _ { r _ { * } } \right] , \left[ \check { U } _ { 1 } , \check { U } _ { 2 } \right] \right) \right\| _ { 2 } = \left\| \left[ I _ { r _ { * } } \right] ^ { T } \check { U } _ { 3 } \right\| _ { 2 } = \left\| \check { U } _ { 3 1 } \right\| _ { 2 }\tag{7.16}
$$

instead, where $\check { U } _ { 3 } \in \mathbb { R } ^ { p \times ( p - r _ { 2 } ) }$ is the trailing $\left( p - r _ { 2 } \right)$ left singular vectors of ${ \check { X } } _ { n }$ and $\check { U } _ { 3 1 }$ is the first $r _ { * }$ rows of ${ \check { U } } _ { 3 }$

We now analyze $\left\| \check { U } _ { 3 1 } \right\| _ { 2 }$ using a similar technique to one used in [38]. Let $\check { V } _ { 3 } \in \mathbb { R } ^ { n \times ( n - r _ { 2 } ) }$ be the trailing right singular vectors of $\check { X }$ and $\widehat { \Sigma } _ { 3 } = \left[ \begin{array} { c c } { \mathrm { d i a g } ( \widehat { \sigma } _ { r _ { 2 } + 1 } , \widehat { \sigma } _ { r _ { 2 } + 2 } , . . . , \widehat { \sigma } _ { n } ) } \\ { 0 _ { ( p - n ) \times ( n - r _ { 2 } ) } } \end{array} \right] \mathrm { i f } p > n \mathrm { o r } \widehat { \Sigma } _ { 3 } = \left[ \mathrm { d i a g } ( \widehat { \sigma } _ { r _ { 2 } + 1 } , \widehat { \sigma } _ { r _ { 2 } + 2 } , . . . , \widehat { \sigma } _ { p } ) \quad 0 _ { ( p - r _ { 2 } ) \times ( n - p ) } \right]$ if $p \leq n$ where $\widehat { \sigma } _ { i } \mathrm { : s }$ are the singular values of $X _ { n } ,$ since $X _ { n }$ and ${ \check { X } } _ { n }$ have the same singular values. Then we have $\check { X } _ { n } \check { V } _ { 3 } = \check { U } _ { 3 } \widehat { \Sigma } _ { 3 }$ and $\check { X } _ { n } ^ { T } \check { U } _ { 3 } = \check { V } _ { 3 } \widehat { \Sigma } _ { 3 }$ . Now divide ${ \check { U } } _ { 3 }$ and ${ \check { V } } _ { 3 }$ into blocks as

$$
\check { U } _ { 3 } = \left[ \begin{array} { l } { \check { U } _ { 3 1 } } \\ { \check { U } _ { 3 2 } } \end{array} \right] , \qquad \check { V } _ { 3 } = \left[ \begin{array} { l } { \check { V } _ { 3 1 } } \\ { \check { V } _ { 3 2 } } \end{array} \right]\tag{7.17}
$$

where $\begin{array} { r } { \check { U } _ { 3 1 } \ \in \ \mathbb { R } ^ { r _ { * } \times ( p - r _ { 2 } ) } , \check { U } _ { 3 2 } \ \in \ \mathbb { R } ^ { ( p - r _ { * } ) \times ( p - r _ { 2 } ) } , \check { V } _ { 3 1 } \ \in \ \mathbb { R } ^ { r _ { * } \times ( n - r _ { 2 } ) } } \end{array}$ and $\check { V } _ { 3 2 } \ \in \ \mathbb { R } ^ { ( n - r _ { * } ) \times ( n - r _ { 2 } ) }$ . Then using $\check { X } _ { n } \check { V } _ { 3 } = \check { U } _ { 3 } \widehat { \Sigma } _ { 3 }$ , we get

$$
\left[ \begin{array} { c c } { \check { \Sigma } } & { 0 } \\ { \tilde { X } _ { 2 } \check { V } } & { \tilde { X } _ { 2 } \check { V } _ { \perp } } \end{array} \right] \left[ \check { V } _ { 3 1 } \right] = \left[ \check { U } _ { 3 1 } \right] \widehat { \Sigma } _ { 3 }\tag{7.18}
$$

and using $\check { X } _ { n } ^ { T } \check { U } _ { 3 } = \check { V } _ { 3 } \widehat { \Sigma } _ { 3 }$ , we get

$$
\begin{array} { r l } { \Big [ \breve { \Sigma } } & { { } ( \widetilde { X } _ { 2 } \breve { V } ) ^ { T } } \\ { 0 } & { { } ( \widetilde { X } _ { 2 } \breve { V } _ { \perp } ) ^ { T } \Big ] \left[ \breve { U } _ { 3 1 } \right] = \left[ \breve { V } _ { 3 1 } \right] \widehat { \Sigma } _ { 3 } . } \end{array}\tag{7.19}
$$

The first block of (7.18) and (7.19) give

$$
\begin{array} { r } { \check { \Sigma } \check { V } _ { 3 1 } = \check { U } _ { 3 1 } \hat { \Sigma } _ { 3 } , \qquad \check { \Sigma } \check { U } _ { 3 1 } + ( \check { X } _ { 2 } \check { V } ) ^ { T } \check { U } _ { 3 2 } = \check { V } _ { 3 1 } \hat { \Sigma } _ { 3 } } \end{array}\tag{7.20}
$$

from which we get the following inequalities

$$
\sigma _ { \operatorname* { m i n } } ( \check { \Sigma } ) \left\| \check { V } _ { 3 1 } \right\| _ { 2 } \leq \left\| \check { \Sigma } \check { V } _ { 3 1 } \right\| _ { 2 } = \left\| \check { U } _ { 3 1 } \widehat { \Sigma } _ { 3 } \right\| _ { 2 } \leq \left\| \check { U } _ { 3 1 } \right\| _ { 2 } \left\| \widehat { \Sigma } _ { 3 } \right\| _ { 2 }\tag{7.21}
$$

and

$$
\begin{array} { r } { \sigma _ { \operatorname* { m i n } } ( \check { \Sigma } ) \left\| \check { U } _ { 3 1 } \right\| _ { 2 } \leq \left\| \check { \Sigma } \check { U } _ { 3 1 } \right\| _ { 2 } = \left\| \check { V } _ { 3 1 } \widehat { \Sigma } _ { 3 } - ( \check { X } _ { 2 } \check { V } ) ^ { T } \check { U } _ { 3 2 } \right\| _ { 2 } \leq \left\| \check { V } _ { 3 1 } \right\| _ { 2 } \left\| \widehat { \Sigma } _ { 3 } \right\| _ { 2 } + \left\| ( \check { X } _ { 2 } \check { V } ) ^ { T } \right\| _ { 2 } . } \end{array}\tag{7.22}
$$

Now multiply (7.21) by $\left\| \widehat { \Sigma } _ { 3 } \right\| _ { 2 }$ and multiply (7.22) by $\sigma _ { \mathrm { m i n } } ( \check { \Sigma } )$ and then add them together to obtain

$$
\left\| \check { U } _ { 3 1 } \right\| _ { 2 } \leq \frac { \left\| \widetilde { X } _ { 2 } \check { V } \right\| _ { 2 } \sigma _ { \operatorname* { m i n } } ( \check { \Sigma } ) } { ( \sigma _ { \operatorname* { m i n } } ( \check { \Sigma } ) ) ^ { 2 } - \left\| \widehat { \Sigma } _ { 3 } \right\| _ { 2 } ^ { 2 } } = \frac { \left\| \widetilde { X } _ { 2 } \check { V } \right\| _ { 2 } \sigma _ { \operatorname* { m i n } } ( \widetilde { X } _ { 1 } ) } { \sigma _ { \operatorname* { m i n } } ( \widetilde { X } _ { 1 } ) ^ { 2 } - \sigma _ { r _ { 2 } + 1 } ( X _ { n } ) ^ { 2 } }
$$

Therefore

$$
\left\| \sin \Theta ( U _ { 1 } , [ \widehat { U } _ { 1 } , \widehat { U } _ { 2 } ] ) \right\| _ { 2 } \leq \frac { \left\| \widetilde { X } _ { 2 } \widetilde { V } \right\| _ { 2 } \sigma _ { \operatorname* { m i n } } ( \widetilde { X } _ { 1 } ) } { \sigma _ { \operatorname* { m i n } } ( \widetilde { X } _ { 1 } ) ^ { 2 } - \sigma _ { r _ { 2 } + 1 } ( X _ { n } ) ^ { 2 } } = \frac { \left\| U _ { * , \bot } ^ { T } X _ { n } \widetilde { V } \right\| _ { 2 } \sigma _ { \operatorname* { m i n } } ( U _ { * } ^ { T } X _ { n } ) } { \sigma _ { \operatorname* { m i n } } ( U _ { * } ^ { T } X _ { n } ) ^ { 2 } - \sigma _ { r _ { 2 } + 1 } ( X _ { n } ) ^ { 2 } }\tag{7.23}
$$

for any $r _ { * } \in [ r _ { 1 } , r _ { 2 } ]$ . Minimizing with respect to $r _ { * } \in [ r _ { 1 } , r _ { 2 } ]$ , we get the desired result.

Theorem 7.3 is a generalization of Theorem 3.2. Theorem 3.2 corresponds to the case when $r _ { * } = r _ { 1 }$ ， which scales well when the noise level is high compared to the signal strengths. When the signal strengths are much larger than the noise level, $r _ { * } = r _ { 1 }$ , hence Theorem 3.2 scales badly. Generally, taking $r _ { * } = r _ { 2 }$ scales much better in this case and Theorem 7.1 accommodates both cases.

## REFERENCES

[1] T. W. Anderson, Asymptotic theory for Principal Component Analysis, Ann. Math. Statist., 34 (1963), pp. 122 – 148, https://doi.org/10.1214/aoms/1177704248.

[2] Z. D. Bai, Methodologies in spectral analysis of large dimensional random matrices, a review, Statist. Sinica, 9 (1999), pp. 611–677.

[3] Z. D. Bai and Y. Q. Yin, Limit of the smallest eigenvalue of a large dimensional sample covariance matrix, Ann. Probab., 21 (1993), pp. 1275 – 1294, https://doi.org/10.1214/aop/1176989118.

[4] J. Baik, G. Ben Arous, and S. P´ech´e, Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices, Ann. Probab., 33 (2005), pp. 1643–1697, https://doi.org/10.1214/009117905000000233.

[5] J. Baik and J. W. Silverstein, Eigenvalues of large sample covariance matrices of spiked population models, 97 (2006), pp. 1382–1408, https://doi.org/10.1016/j.jmva.2005.08.003.

[6] <sup>˚</sup>A. Bjorck ¨ , Numerical Methods for Least Squares Problems, SIAM, Philadelphia, 2nd ed., 2024.

[7] R. B. Cattell, The scree test for the number of factors, Multivariate Behav. Res., 1 (1966), pp. 245–276, https://doi. org/10.1207/s15327906mbr0102 10.

[8] S. Chaturantabut and D. C. Sorensen, Nonlinear model reduction via discrete empirical interpolation, SIAM J. Sci. Comput., 32 (2010), pp. 2737–2764, https://doi.org/10.1137/090766498.

[9] Y. Chen, E. N. Epperly, J. A. Tropp, and R. J. Webber, Randomly pivoted Cholesky: Practical approximation of a kernel matrix with few entry evaluations, Comm. Pure Appl. Math., 78 (2025), pp. 995–1041, https://doi.org/10. 1002/cpa.22234.

[10] K. R. Davidson and S. J. Szarek, Local operator theory, random matrices and Banach spaces, in Handbook of the Geometry of Banach Spaces, W. Johnson and J. Lindenstrauss, eds., vol. 1, Elsevier, 2001, pp. 317–366, https://doi. org/https://doi.org/10.1016/S1874-5849(01)80010-3.

[11] C. Davis and W. M. Kahan, The rotation of eigenvectors by a perturbation. III, SIAM J. Numer. Anal., 7 (1970), pp. 1–46, https://doi.org/10.1137/0707001.

[12] Y. Dong and P.-G. Martinsson, Simpler is better: a comparative study of randomized pivoting algorithms for CUR and interpolative decompositions, Adv. Comput. Math., 49 (2023), p. 66, https://doi.org/10.1007/s10444-023-10061-z.

[13] P. Drineas, M. Magdon-Ismail, M. W. Mahoney, and D. P. Woodruff, Fast approximation of matrix coherence and statistical leverage, J. Mach. Learn. Res., 13 (2012), pp. 3475–3506.

[14] P. Drineas, M. W. Mahoney, and S. Muthukrishnan, Sampling algorithms for l2 regression and applications, in Proceedings of the seventeenth annual ACM-SIAM symposium on Discrete algorithm, 2006, pp. 1127–1136, https:// doi.org/10.1145/1109557.1109682.

[15] Z. Drmac and S. Gugercinˇ , A new selection operator for the discrete empirical interpolation method—improved a priori error bound and extensions, SIAM J. Sci. Comput., 38 (2016), pp. A631–A648, https://doi.org/10.1137/15M1019271.

[16] S. Geman, A limit theorem for the norm of random matrices, Ann. Probab., 8 (1980), pp. 252–261, https://doi.org/10. 1214/aop/1176994775.

[17] G. H. Golub and C. F. Van Loan, Matrix Computations, The Johns Hopkins University Press, Baltimore, 4th ed., 2013.

[18] N. Halko, P.-G. Martinsson, and J. A. Tropp, Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions, SIAM Rev., 53 (2011), pp. 217–288, https://doi.org/10.1137/ 090771806.

[19] P. C. Hansen, The L-Curve and its use in the numerical treatment of inverse problems, in Computational Inverse Problems in Electrocardiology, WIT Press, 2001, p. 119.

[20] P. C. Hansen and D. P. O’Leary, The use of the L-Curve in the regularization of discrete ill-posed problems, SIAM J. Sci. Comput., 14 (1993), pp. 1487–1503, https://doi.org/10.1137/0914086.

[21] J. L. Horn, A rationale and test for the number of factors in factor analysis, Psychometrika, 30 (1965), pp. 179–185, https://doi.org/10.1007/BF02289447.

[22] I. C. F. Ipsen and T. Wentworth, The efect of coherence on sampling from matrices with orthonormal columns, and preconditioned least squares problems, SIAM J. Matrix Anal. Appl., 35 (2014), pp. 1490–1520, https://doi.org/10. 1137/120870748.

[23] I. M. Johnstone, On the distribution of the largest eigenvalue in principal components analysis, Ann. Statist., 29 (2001), pp. 295–327, https://doi.org/10.1214/aos/1009210544.

[24] I. M. Johnstone and A. Y. Lu, On consistency and sparsity for principal components analysis in high dimensions, 104 (2009), pp. 682–693, https://doi.org/10.1198/jasa.2009.0121.

[25] I. M. Johnstone and D. Paul, PCA in high dimensions: An orientation, Proc. IEEE Inst. Electr. Electron. Eng., 106 (2018), pp. 1277–1292.

[26] E. Jones and Y. Nakatsukasa, SubApSnap: Solving parameter-dependent linear systems with a snapshot and subsampling, arXiv preprint arXiv:2510.04825, (2025).

[27] M. Journ´ee, Y. Nesterov, P. Richtarik, and R. Sepulchre ´ , Generalized power method for sparse principal component analysis, J. Mach. Learn. Res., 11 (2010), pp. 517–553.

[28] V. Koltchinskii and K. Lounici, Normal approximation and concentration of spectral projectors of sample covariance, Ann. Statist., 45 (2017), pp. 121 – 157, https://doi.org/10.1214/16-AOS1437.

[29] S. Kritchman and B. Nadler, Non-parametric detection of the number of signals: Hypothesis testing and random matrix theory, IEEE Trans. Signal Process., 57 (2009), pp. 3930–3941, https://doi.org/10.1109/TSP.2009.2022897.

[30] A. Kulesza, B. Taskar, et al., Determinantal point processes for machine learning, Found. Trends Mach. Learn., 5 (2012), pp. 123–286, https://doi.org/10.1561/2200000044.

[31] S. Kumar and P. Sarkar, Oja’s algorithm for streaming sparse PCA, Advances in Neural Information Processing Systems, 37 (2024), pp. 74528–74578.

[32] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, Gradient-based learning applied to document recognition, Proc.

IEEE, 86 (1998), pp. 2278–2324, https://doi.org/10.1109/5.726791.

[33] M. W. Mahoney, Randomized algorithms for matrices and data, Foundations and Trends® in Machine Learning, 3 (2011), pp. 123–224, https://doi.org/10.1561/2200000035.

[34] V. A. Marcenko and L. A. Pastur ˇ , Distribution Of Eigenvalues For Some Sets Of Random Matrices, Mathematics of the USSR-Sbornik, 1 (1967), pp. 457–483, https://doi.org/10.1070/SM1967v001n04ABEH001994.

[35] R. J. Muirhead, Aspects of multivariate statistical theory, John Wiley & Sons, Hoboken, 2009.

[36] R. R. Nadakuditi and A. Edelman, Sample eigenvalue based detection of high-dimensional signals in white noise using relatively few samples, IEEE Trans. Signal Process., 56 (2008), pp. 2625–2638, https://doi.org/10.1109/TSP.2008. 917356.

[37] B. Nadler, Finite sample approximation results for principal component analysis: A matrix perturbation approach, Ann. Statist., 36 (2008), pp. 2791–2817, https://doi.org/10.1214/08-AOS618.

[38] Y. Nakatsukasa, Sharp error bounds for Ritz vectors and approximate singular vectors, 89 (2020), pp. 1843–1866, https://doi.org/10.1090/mcom/3519.

[39] T. Park and Y. Nakatsukasa, Accuracy and stability of CUR decompositions with oversampling, SIAM J. Matrix Anal. Appl., 46 (2025), pp. 780–810, https://doi.org/10.1137/24M1660346.

[40] B. N. Parlett, The Symmetric Eigenvalue Problem, SIAM, Philadelphia, 1998.

[41] D. Paul, Asymptotics of sample eigenstructure for a large dimensional spiked covariance model, Stat. Sinica, 17 (2007), pp. 1617–1642.

[42] M. Reiss and M. Wahl, Nonasymptotic upper bounds for the reconstruction error of PCA, Ann. Statist., 48 (2020), pp. 1098–1123, https://doi.org/10.1214/19-AOS1839.

[43] Y. Saad, Numerical Methods for Large Eigenvalue Problems, SIAM, Philadelphia, 2nd ed., 2011.

[44] J. W. Silverstein, The smallest eigenvalue of a large dimensional wishart matrix, Ann. Probab., (1985), pp. 1364–1368, https://doi.org/10.1214/aop/1176992819.

[45] G. W. Stewart and J.-g. Sun, Matrix Perturbation Theory, Academic Press, Boston, 1990.

[46] N. Vaswani and P. Narayanamurthy, Finite sample guarantees for PCA in non-isotropic and data-dependent noise, in 55th Annu. Allert. Conf. Commun. Control Comput., IEEE, 2017, pp. 783–789, https://doi.org/10.1109/ALLERTON. 2017.8262819.

[47] M. Wax and T. Kailath, Detection of signals by information theoretic criteria, IEEE Trans. Acoust. Speech Signal Process., 33 (1985), pp. 387–392, https://doi.org/10.1109/TASSP.1985.1164557.

[48] P.-r. Wedin, Perturbation bounds in connection with singular value decomposition, BIT, 12 (1972), pp. 99–111, https:// doi.org/10.1007/BF01932678.

[49] J. Yao, S. Zheng, and Z. Bai, Large Sample Covariance Matrices and High-Dimensional Data Analysis, Cambridge University Press, New York, 2015, https://doi.org/https://doi.org/10.1017/CBO9781107588080.

[50] X.-T. Yuan and T. Zhang, Truncated power method for sparse eigenvalue problems, J. Mach. Learn. Res., 14 (2013), pp. 899–925.

[51] H. Zou, T. Hastie, and R. Tibshirani, Sparse principal component analysis, J. Comput. Graph. Statist., 15 (2006), pp. 265–286, https://doi.org/10.1198/106186006X113430.