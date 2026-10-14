# On the Optimality of Random Features under General Conditions: Misspecified Cases and Robust Learning

Caixing Wang

School of Statistics and Data Science Southeast University Nanjing, China caixingwang@seu.edu.cn

## Abstract

Random features provide a standard approach to scaling kernel methods to large datasets by approximating the kernel with a finite-dimensional feature map. Existing works of random feature methods primarily focus on well-specified models and light-tailed noise conditions, leaving a gap in understanding their performance under more general conditions, such as misspecified models and heavy-tailed noise. In this paper, we provide a comprehensive and improved analysis of kernel ridge regression with random features (KRR-RF) under general conditions. In the light-tailed noise setting, we improve upon the worst-case uniform bounds of the feature map used in the previous works and lead to the minimax optimal learning rates with $2 r + \gamma > \alpha _ { 0 }$ where $( r , \gamma , \alpha _ { 0 } )$ denote the regularity of the target function, the efective dimension of the kernel, and the embedding index of the RKHS, respectively. In the heavy-tailed noise setting, we leverage a recently developed Hilbert-valued Fuk–Nagaev inequality and extend the same optimal results under finite weak moment conditions. The additional cost to achieve robustness is a mildly stronger explicit sample-size requirement than in the light-tailed case. For both settings, our results recover the conditions random features budgets (up to log factors) required in the previous works. In the most benign case when $\alpha _ { 0 } = \gamma$ , KRR-RF are minimax optimal for all $r \in ( 0 , 1 ]$ . To the best of our knowledge, this is the first work to establish optimal learning rates for KRR-RF under weak moment conditions in the misspecified setting. Numerical experiments and a real data analysis validate the derived learning rates and demonstrate their practical relevance in the presence of mis-specification and heavy-tailed noise.

Keywords: Random Features; Misspecified Models; Robust Learning; Weak Moments; Generalization.

## 1 Introduction

In a standard supervised learning setting, we are given a dataset $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ consisting of n independent and identically distributed samples drawn from an unknown distribution $\rho$ on $\mathcal { X } \times \mathbb { R }$ The goal is to learn a function $f : \mathcal { X } \to \mathbb { R }$ that minimizes the expected risk

$$
\mathcal { E } ( f ) = \mathbb { E } _ { ( \pmb { x } , y ) \sim \rho } [ ( f ( \pmb { x } ) - y ) ^ { 2 } ] .\tag{1}
$$

The minimizer of (1), called the regression function, is given by the conditional expectation $f _ { \rho } ( { \pmb x } ) =$ $\textstyle \int _ { \mathbb { R } } y d \rho ( y | \pmb { x } )$ , where $\rho ( y | \pmb { x } )$ denotes the conditional distribution of y given x. Since $\rho$ is unknown, $f _ { \rho }$ cannot be computed directly, it must be estimated from the training samples within a pre-specified hypothesis space.

A natural and widely adopted choice of the hypothesis space is a reproducing kernel Hilbert space (RKHS) H associated with a continuous, symmetric, positive semi-definite kernel $K : \mathcal { X } \times$ $\mathcal X  \mathbb R$ (Aronszajn, 1950; Steinwart and Christmann, 2008). Kernel methods estimate the regression function within H, typically by solving a regularized empirical risk minimization problem. Among various kernel methods, kernel ridge regression (KRR) is one of the most popular approaches due to its simplicity and strong theoretical guarantees (Smale and Zhou, 2007; Caponnetto and De Vito, 2007). Despite its empirical success, KRR sufers from high computational complexity: it requires storing and inverting an $n \times n$ kernel matrix, which becomes prohibitive for large-scale datasets. This limitation has motivated the development of scalable techniques, which aim to reduce the computational burden while retaining the theoretical properties of kernel methods, such as distributed learning (Zhang et al., 2015; Lin et al., 2017, 2020b), Nystr¨om subsampling (Williams and Seeger, 2001; Rudi et al., 2015; Li et al., 2023a), random features (Rahimi and Recht, 2007, 2008; Rudin, 2017; Rudi and Rosasco, 2017), stochastic gradient methods (Lin and Rosasco, 2016; Lin and Cevher, 2020), Falkon (Rudi et al., 2017) and EigenPro 3.0 (Abedsoltan et al., 2023).

Among these scalable techniques, random features have become one of the most widely used approaches, owing to their conceptual simplicity and ease of implementation. The key idea is to approximate the kernel by the inner product of finite-dimensional randomized feature maps, so that KRR reduces to a linear ridge regression problem in lower dimensions (Rahimi and Recht, 2007, 2008). This brings the training cost down from $O ( n ^ { 3 } )$ to $O ( n M ^ { 2 } )$ and the memory cost from $O ( n ^ { 2 } )$ to $O ( M ^ { 2 } )$ , which is a substantial saving when $M \ll n$ . A significant body of research has focused on understanding the generalization properties of random feature approximations (Rudi and Rosasco, 2017; Avron et al., 2017; Bach, 2017; Li et al., 2021; Lanthaler and Nelsen, 2023). In the milestone work by Rudi and Rosasco (2017), the learning rates of $\mathrm { K R R - R F }$ align with those of the exact KRR up to a logarithmic factor, provided that the number of random features M is suficiently large. Their analysis leveraged the integral operator tool and concentration inequalities to establish these results. Crucially, these rates are governed by two fundamental quantities: the efective dimension of the kernel characterized by parameter γ (capacity condition) and the regularity of the target function characterized by parameter r (source condition). However, these results are essentially confined to the well-specified case, i.e., $f _ { \rho } \in \mathcal { H }$ when $r \geq 1 / 2$ , recent research has started to explore the generalization properties of KRR-RF under this setting. However, these studies are either capacity-independent (Lanthaler and Nelsen, 2023) or can only cover part of the misspecified regime when $2 r + \gamma \geq 1$ (Liu and Lian, 2023; Li and Liu, 2023; Li et al., 2023b; Wang et al., 2026). Whether the full misspecified regime $r \in ( 0 , 1 ]$ can be efectively addressed remains an open question and needs new analytical techniques.

Another important condition in the analysis of KRR-RF is the noise condition. The abovementioned results typically assume that the noise or response is bounded or drawn from a light-tailed distribution, such as Gaussian and sub-Gaussian distributions. This Bernstein-type assumption facilitates the derivation of Bernstein concentration inequalities and generalization bounds. However, the light-tailed assumption on the noise may fail in practice. Financial time series containing stock returns during market crashes, sensor networks subject to occasional malfunctions, and medical records with rare but extreme patient measurements all exhibit heavy-tailed behavior, which is typically modeled by distributions with polynomially decaying tails, such as the Student-t, Fr´echet, and Pareto distributions (Sigman, 1999; Rachev, 2003). In such cases, the noise possesses only a finite number of moments, and classical Bernstein concentration inequalities underlying the existing analyses of random features no longer apply. Despite its practical importance, the study of KRR-RF under heavy-tailed noise and misspecified settings remains relatively limited, necessitating the development of new analytical tools and techniques to handle such challenging scenarios. In this paper, we aim to fill this gap by providing a comprehensive and improved analysis of KRR-RF under general conditions including both misspecified settings and heavy-tailed noise. We focus on the following key questions:

First, in the light-tailed noise setting, what is the precise range of the source and capacity parameters $( r , \gamma )$ for which KRR-RF achieves optimal generalization performance and what are the corresponding requirements on the number of random features M and the sample size n? Second, in the heavy-tailed noise setting, does the optimal rate persist under weak moment conditions? If so, what are the additional costs to achieving the robust generalization performance?

We answer these questions by establishing new theoretical results that characterize the generalization performance of KRR-RF under both light-tailed and heavy-tailed noise conditions, and provide precise conditions on the source and capacity parameters, the number of random features, and the sample size required to achieve optimal or robust generalization performance. Our contributions can be summarized as follows:

• Comprehensive and Improved Theory. To derive a refined understanding of KRR-RF in the misspecified setting, we use an additional embedding condition characterized by index $\alpha _ { 0 } \in [ \gamma , 1 ]$ , which quantifies the $L ^ { \infty }$ embedding to interpolation space of the RKHS. In the light-tailed setting, we establish the minimax optimal learning rates of KRR-RF whenever $2 r + \gamma > \alpha _ { 0 }$ This extends the existing theory $2 r + \gamma > 1$ to a more general setting and covers the full range of source conditions $r \in ( 0 , 1 ]$ when $\alpha _ { 0 } = \gamma$ This is non-trivial and needs new analytical techniques to give a refined bound on the feature map. In the heavytailed noise setting, we further show that the same optimal rates persist when the noise only has a finite number of moments. The price of robustness is a mildly stronger and explicit requirement on the sample size n than in the light-tailed case. In both settings, our results recover the random-feature budgets required by previous works up to logarithmic factors for both uniformly sampling and data-dependent sampling schemes. Thus the extension to the full misspecified setting and heavy-tailed noise incurs essentially no additional cost in the number of features. To the best of our knowledge, this is the first work to establish optimal learning rates for KRR-RF under weak moment conditions in the mis-specified setting.

• Novel Analytical Techniques. We first decompose the generalization error into a population RF error, an empirical RF fluctuation error and a noise error. For the latter two stochastic error terms, they are both determined by a leverage-type quantity associated with the feature map $\| C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( \pmb { x } ) \| _ { 2 } ^ { 2 }$ . We develop new analytical techniques to bound this leverage-type quantity by using a variational leverage identity and the embedding condition. Instead of the uniform worst-case bound $\kappa ^ { 2 } / \lambda$ used in previous work, we establish tighter pointwise control of order $O ( \lambda ^ { - \alpha } )$ for $\alpha > \alpha _ { 0 }$ in high probability. This refinement together with Chebyshev’s and Bernstein’s inequalities enable us to obtain sharp bounds on the empirical RF fluctuation error and the noise error, leading to improved generalization guarantees for KRR-RF under both light-tailed noise conditions. In the heavy-tailed noise setting, we further combine these techniques with a recently developed Hilbert-valued Fuk-Nagaev inequality (Fuk, 1973; Nagaev, 1979; Yurinsky, 2006; Mollenhauer et al., 2026) to bound the noise error under weak moment conditions. We prove that once the sample size is suficiently large (larger than that for the light-tailed case), the additional polynomial factor can be dominated by the traditiona logarithmic factor, resulting in the same optimal learning rates even under heavy-tailed noise.

• Practical Experimental Validation. We conduct extensive experiments on synthetic data to validate our theoretical findings. The experiments demonstrate the optimal learning rates can be achieved in both light-tailed and heavy-tailed noise settings, even when the regression function lies outside the RKHS. We also apply our methods to real-world datasets to further illustrate the practical efectiveness and robustness of KRR-RF in practical scenarios.

Organization. The rest of this paper is organized as follows. Section 2 introduces the regression model, the RKHS, and its interpolation spaces. Section 3 reviews kernel ridge regression and its random feature approximation. Section 4 presents the main results: the assumptions, the optimal learning rates of KRR-RF under light-tailed and heavy-tailed noise, and a matching minimax lower bound. Section 5 compares our results with related work on KRR, random features, and robust kernel methods. Sections 6 and 7 report simulated experiments and a real data application. The proofs of the main results are provided in the appendices.

## 2 Preliminaries

## 2.1 Model Setup

Let X be a compact subset of $\mathbb { R } ^ { d }$ and $\mathcal { V } \subset \mathbb { R }$ be the output space. The joint distribution $\rho$ on $\mathcal { X } \times \mathcal { V }$ can be decomposed as $\rho ( \pmb { x } , y ) = \rho _ { X } ( \pmb { x } ) \rho ( y | \pmb { x } )$ , where $\rho _ { X }$ is the marginal distribution of the covariates and $\rho ( y | \pmb { x } )$ is the conditional distribution of y given x. Under the standard supervised framework of (1) and the definition the regression function $f _ { \rho }$ , we can express the output y as

$$
y = f _ { \rho } ( { \pmb x } ) + \varepsilon ,\tag{2}
$$

where $\varepsilon$ is the noise term with zero mean conditioned on x. In this paper, we do not impose assumption that the noise $\varepsilon$ is bounded (Smale and Zhou, 2007; Li et al., 2021) or drawn from a light-tailed distribution (Caponnetto and De Vito, 2007; Rudi and Rosasco, 2017).

## 2.2 RKHS and its Interpolation Space

We take the hypothesis space to be a reproducing kernel Hilbert space (RKHS). Let $K : \mathcal { X } \times \mathcal { X } $ R be a continuous, symmetric, and positive semi-definite Mercer kernel. The associated RKHS H is defined as the completion of span $\{ K ( \cdot , \pmb { x } ) : \pmb { x } \in \pmb { \mathcal { X } } \}$ with respect to the inner product satisfying $\langle K ( \cdot , { \pmb x } ) , K ( \cdot , { \pmb x } ^ { \prime } ) \rangle _ { \mathscr { H } } = K ( { \pmb x } , { \pmb x } ^ { \prime } )$ , and it enjoys the reproducing property

$$
f ( \pmb { x } ) = \langle f , K ( \cdot , \pmb { x } ) \rangle _ { \mathcal { H } }
$$

for all $f \in \mathcal H$ . Since K is a Mercer kernel and X is compact, there exists a constant $\kappa > 0$ such that

$$
\operatorname* { s u p } _ { \pmb { x } \in \mathcal { X } } K ( \pmb { x } , \pmb { x } ) \le \kappa ^ { 2 } .
$$

We also define the space of square-integrable functions with respect to the marginal distribution $\rho _ { X }$ as $\begin{array} { r } { L ^ { 2 } ( \rho _ { X } ) = \{ f : \mathcal { X } \to \mathbb { R } \mid \| f \| _ { L ^ { 2 } ( \rho _ { \mathcal { X } } ) } ^ { 2 } = \int _ { \mathcal { X } } | f ( \pmb { x } ) | ^ { 2 } d \rho _ { X } ( \pmb { x } ) < \infty \} } \end{array}$

The spectral properties of the RKHS are characterized through the integral operator $L _ { K }$ : $L ^ { 2 } ( \rho _ { X } ) \to L ^ { 2 } ( \rho _ { X } )$ , defined by

$$
( L _ { K } f ) ( { \pmb x } ) = \int _ { \pmb { \chi } } K ( { \pmb x } , { \pmb x } ^ { \prime } ) f ( { \pmb x } ^ { \prime } ) d \rho _ { X } ( { \pmb x } ^ { \prime } ) .\tag{3}
$$

The operator $L _ { K }$ is positive, self-adjoint, and trace-class. By Mercer’s theorem, it admits the spectral decomposition

$$
{ \cal L } _ { K } = \sum _ { i = 1 } ^ { \infty } \mu _ { i } \langle \cdot , e _ { i } \rangle _ { L ^ { 2 } ( \rho _ { X } ) } e _ { i } ,\tag{4}
$$

where $\{ \mu _ { i } \} _ { i \geq 1 }$ are the positive eigenvalues in non-increasing order and $\left\{ \boldsymbol { e } _ { i } \right\}$ are the corresponding eigenfunctions, which form an orthonormal basis of $L ^ { 2 } ( \rho _ { X } )$ . Throughout this paper, we assume that $L _ { K }$ is injective on $L ^ { 2 } ( \rho _ { X } )$ , so that the eigenfunctions $\{ e _ { j } \} _ { j \ge 1 }$ associated with the positive eigenvalues form a complete orthonormal basis of $L ^ { 2 } ( \rho _ { X } )$

We then introduce the interpolation space (or fractional power space) associated with the RKHS $\mathcal { H } .$ . For any $s \geq 0$ , we define the fractional integral operator $L _ { K } ^ { s }$ via the spectral decomposition (4) as $\begin{array} { r } { L _ { K } ^ { s } = \sum _ { i = 1 } ^ { \infty } \mu _ { i } ^ { s } \langle \cdot , e _ { i } \rangle _ { L ^ { 2 } ( \rho _ { X } ) } e _ { i } } \end{array}$

Definition 1. Given the RKHS H induced by the kernel K and its associated integral operator $L _ { K }$ for any $s \geq 0 ,$ , the interpolation space $[ { \mathcal { H } } ] ^ { s }$ is defined as the range of the operator $\overline { { L _ { K } ^ { s / 2 } } }$

$$
[ \mathcal { H } ] ^ { s } : = \mathrm { R a n } L _ { K } ^ { s / 2 } = \left\{ \sum _ { i = 1 } ^ { \infty } a _ { i } \mu _ { i } ^ { s / 2 } e _ { i } : \sum _ { i = 1 } ^ { \infty } a _ { i } ^ { 2 } < \infty \right\} \subset L ^ { 2 } ( \rho _ { X } ) .\tag{5}
$$

For any $f , g \in [ \mathcal { H } ] ^ { s }$ admitting the expansions $\begin{array} { r } { f = \sum _ { i = 1 } ^ { \infty } a _ { i } \mu _ { i } ^ { s / 2 } e _ { i } } \end{array}$ and $\begin{array} { r } { g = \sum _ { i = 1 } ^ { \infty } b _ { i } \mu _ { i } ^ { s / 2 } e _ { i } } \end{array}$ , the inner product and the induced norm on $[ { \mathcal { H } } ] ^ { s }$ are defined as $\begin{array} { r } { \langle f , g \rangle _ { [ \mathcal { H } ] ^ { s } } = \langle L _ { K } ^ { - s / 2 } f , L _ { K } ^ { - s / 2 } g \rangle _ { \rho } = \sum _ { i = 1 } ^ { \infty } a _ { i } b _ { i } } \end{array}$ and $\begin{array} { r } { \| f \| _ { [ \mathcal { H } ] ^ { s } } = \langle f , f \rangle _ { [ \mathcal { H } ] ^ { s } } ^ { 1 / 2 } = \bigl ( \sum _ { i = 1 } ^ { \infty } a _ { i } ^ { 2 } \bigr ) ^ { 1 / 2 } } \end{array}$

Equipped with this inner product, $[ { \mathcal { H } } ] ^ { s }$ is a separable Hilbert space with orthonormal basis $\{ \mu _ { i } ^ { s / 2 } e _ { i } \} _ { i \geq 1 }$ In particular, $[ { \mathcal { H } } ] ^ { 1 } = { \mathcal { H } }$ and $[ \mathcal { H } ] ^ { 0 } ~ = ~ L ^ { 2 } ( \rho _ { X } )$ . Moreover, for any $0 ~ \leq ~ s _ { 1 } ~ \leq ~ s _ { 2 }$ there are continuous embeddings $[ { \mathcal { H } } ] ^ { s _ { 2 } } \hookrightarrow [ { \mathcal { H } } ] ^ { s _ { 1 } } \hookrightarrow L ^ { 2 } ( \rho _ { X } )$ (Fischer and Steinwart, 2020). The interpolation spaces provide a natural framework for characterizing the regularity of the regression function $f _ { \rho } \colon$ the condition $f _ { \rho } \in [ \mathcal { H } ] ^ { s }$ with $s > 1$ corresponds to functions smoother than those in $\mathcal { H }$ , whereas $0 < s < 1$ corresponds to a larger space of rougher functions. In the latter regime, $[ { \mathcal { H } } ] ^ { s }$ can be equivalently characterized by the real interpolation method (Steinwart and Scovel, 2012): $[ \mathcal { H } ] ^ { s } \cong [ L ^ { 2 } ( \rho _ { X } ) , \mathcal { H } ] _ { s , 2 } ,$ , where <sup>∼</sup>= denotes isomorphism with equivalent norms and $[ \cdot , \cdot ] _ { s , 2 }$ is the real interpolation functor.

## 3 Kernel Ridge Regression with Random Features (KRR-RF)

## 3.1 Kernel Ridge Regression (KRR)

We estimate the regression function $f _ { \rho }$ by kernel ridge regression over the RKHS H. Given the training samples $\{ ( \pmb { x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ , the KRR estimator is defined as the solution of the regularized empirical risk minimization problem

$$
\widehat { f } _ { \lambda } = \arg \operatorname* { m i n } _ { f \in \mathcal { H } } \left\{ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( f ( \pmb { x } _ { i } ) - y _ { i } \right) ^ { 2 } + \lambda \| f \| _ { \mathcal { H } } ^ { 2 } \right\} ,\tag{6}
$$

where $\lambda > 0$ is the regularization parameter. By the representer theorem (Wahba, 1990), the solution takes the form $\begin{array} { r } { \widehat { f } _ { \lambda } = \sum _ { i = 1 } ^ { n } \alpha _ { i } K ( \cdot , \pmb { x } _ { i } ) } \end{array}$ , with coeficient vector α given by

$$
\begin{array} { r } { { \pmb \alpha } = ( { \bf K } + \lambda n { \bf I } ) ^ { - 1 } { \bf y } , } \end{array}\tag{7}
$$

where $\mathbf { K } = [ K ( \pmb { x } _ { i } , \pmb { x } _ { j } ) ] _ { i , j = 1 } ^ { n }$ is the Gram matrix and $\mathbf { y } = ( y _ { 1 } , \dots , y _ { n } ) ^ { \top }$

Despite its appealing theoretical properties (Smale and Zhou, 2007; Caponnetto and De Vito, 2007), the computational cost of KRR can be prohibitive for large-scale datasets due to the need to invert the $n \times n$ Gram matrix. Typically, this results in a computational complexity of $\mathcal { O } ( n ^ { 3 } )$ and a memory requirement of $O ( n ^ { 2 } )$ , which can be infeasible for large n. This scalability issue has motivated a line of scalable approximations, among which random features, introduced in the next section, are one of the most widely used approaches.

## 3.2 Kernel Ridge Regression with Random Features

Random features provide a scalable approximation of the kernel by randomized finite-dimensional feature maps (Rahimi and Recht, 2007). We assume that the kernel admits an integral representation

$$
K ( { \pmb x } , { \pmb x } ^ { \prime } ) = \int _ { \Omega } \psi ( { \pmb x } , { \pmb w } ) \psi ( { \pmb x } ^ { \prime } , { \pmb w } ) d \pi ( { \pmb w } ) ,\tag{8}
$$

where $( \Omega , \pi )$ is a probability space and $\psi : \mathcal { X } \times \Omega  \mathbb { R }$ is a feature map and π is a probability measure on Ω. This covers the standard Fourier features for shift-invariant kernels (Rahimi and Recht, 2007) (see Example 1) as well as more general constructions. Drawing $\pmb { w } _ { 1 } , \dots , \pmb { w } _ { M }$ independently from π, we define the random feature map

$$
\phi _ { M } ( \pmb { x } ) = \frac { 1 } { \sqrt { M } } \big ( \psi ( \pmb { x } , \pmb { w } _ { 1 } ) , \dots , \psi ( \pmb { x } , \pmb { w } _ { M } ) \big ) ^ { \top } \in \mathbb { R } ^ { M } ,\tag{9}
$$

which satisfies $\begin{array} { r } { \mathbb { E } _ { \pmb { w } \sim \pi } [ \phi _ { M } ( \pmb { x } ) ^ { \top } \phi _ { M } ( \pmb { x } ^ { \prime } ) ] = K ( \pmb { x } , \pmb { x } ^ { \prime } ) } \end{array}$

Example 1. For the Gaussian kernel, $K ( \pmb { x } , \pmb { x } ^ { \prime } ) = e ^ { - \frac { \| \pmb { x } - \pmb { x } ^ { \prime } \| _ { 2 } ^ { 2 } } { 2 \sigma ^ { 2 } } }$ with $\sigma > 0$ , it is continuous, positive definite, and scaled shift-invariant. Since the inverse Fourier transform of $e ^ { - \frac { \| z \| _ { 2 } ^ { 2 } } { 2 \sigma ^ { 2 } } }$ is a Gaussian, and using a basic symmetry argument, we have

$$
K ( \pmb { x } , \pmb { x } ^ { \prime } ) = \frac { 1 } { 2 \pi C } \int \int _ { 0 } ^ { 2 \pi } \sqrt { 2 } \cos ( \pmb { w } ^ { T } \pmb { x } + b ) \sqrt { 2 } \cos ( \pmb { w } ^ { T } \pmb { x } ^ { \prime } + b ) e ^ { - \frac { \sigma ^ { 2 } \| \pmb { x } - \pmb { x } ^ { \prime } \| _ { 2 } ^ { 2 } } { 2 } } d \pmb { w } d b ,
$$

where C is a normalizing factor. It can be approximated with $\psi ( { \pmb x } , { \pmb w } _ { i } ) = \sqrt { 2 } \cos ( { \pmb w } _ { i } ^ { T } { \pmb x } + b _ { i } )$ where $\{ w _ { i } \} _ { i = 1 } ^ { M }$ and $\{ b _ { i } \} _ { i = 1 } ^ { M }$ are independently drawn from distribution with density $\frac { 1 } { C } e ^ { - \frac { \sigma ^ { 2 } \| w \| _ { 2 } ^ { 2 } } { 2 } }$ and Unif[0, 2π], respectively.

Replacing the kernel in (6) with the finite-dimensional approximation $\begin{array} { r l } { K _ { M } ( { \pmb x } , { \pmb x } ^ { \prime } ) } & { { } = } \end{array}$ $\langle \phi _ { M } ( { \pmb x } ) , \phi _ { M } ( { \pmb x } ^ { \prime } ) \rangle$ leads to the KRR-RF estimator $\widehat { f } _ { \lambda , M } ( { \pmb x } ) = \widehat { \pmb \theta ^ { \top } } \phi _ { M } ( \overline { { \pmb x } } )$ , where

$$
\widehat { \pmb { \theta } } = ( \widehat { C } _ { M } + \lambda I ) ^ { - 1 } \widehat { S } _ { M } ^ { * } \widehat { \mathbf { y } } ,\tag{10}
$$

where we define operator $\begin{array} { r } { \widehat { S } _ { M } ^ { * } \ = \ \frac { 1 } { \sqrt { n } } \big ( \phi _ { M } ( \pmb { x } _ { 1 } ) , \dots , \phi _ { M } ( \pmb { x } _ { n } ) \big ) , \ \widehat { \mathbf { y } } \ = \ \frac { 1 } { \sqrt { n } } \big ( y _ { 1 } , \dots , y _ { n } \big ) ^ { \top } } \end{array}$ , and ${ \widehat { C } } _ { M } =$ $\begin{array} { r } { \widehat { S } _ { M } ^ { * } \widehat { S } _ { M } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \phi _ { M } ( \pmb { x } _ { i } ) \phi _ { M } ( \pmb { x } _ { i } ) ^ { \top } } \end{array}$ . Furthermore, we denote $S _ { M } : \mathbb { R } ^ { M } \to L ^ { 2 } ( \rho _ { X } ) , ( S _ { M } \beta ) ( \cdot ) =$ $\phi _ { M } ( \cdot ) ^ { \top } \beta$ for any $\beta \in \mathbb { R } ^ { M }$ , and $S _ { M } ^ { * } : L ^ { 2 } ( \rho _ { X } ) \to \mathbb { R } ^ { M }$ as its adjoint operator. Then the final estimator can be written as

$$
\widehat { f } _ { \lambda , M } = S _ { M } \widehat { C } _ { M , \lambda } ^ { - 1 } \widehat { S } _ { M } ^ { * } \widehat { \mathbf { y } } , \quad \widehat { C } _ { M , \lambda } = \widehat { C } _ { M } + \lambda I .\tag{11}
$$

The population counterpart of ${ \widehat { C } } _ { M }$ is $C _ { M } = \mathbb { E } _ { { \pmb x } \sim \rho _ { X } } [ \phi _ { M } ( { \pmb x } ) \phi _ { M } ( { \pmb x } ) ^ { \top } ]$ and $C _ { M , \lambda } = C _ { M } + \lambda I$ . Solving (10) typically needs the computation cost of $O ( M ^ { 2 } n )$ and the memory cost of $O ( M ^ { 2 } )$ , which is significantly lower than the $O ( n ^ { 3 } )$ computation and $O ( n ^ { 2 } )$ memory required for the exact kernel method when $M \ll n$

Remark 2 (Limitations of existing KRR-RF theory). Despite the approximation introduced by using a finite number of random features, the KRR-RF estimator can still achieve optimal learning rates under suitable conditions on M and λ. However, its existing theory is limited in two fundamental aspects. First, most analyses are restricted to the well-specified case, where the regression function belongs to the RKHS, i.e., $f _ { \rho } \in \mathcal { H }$ , corresponding to $r \geq 1 / 2$ under the source condition. Analyses that allow the misspecified case $r < 1 / 2$ typically impose the additional restriction $2 r + \gamma \geq 1$ , which excludes a substantial part of the source range. Second, existing results rely heavily on light-tailed noise assumptions, such as bounded or sub-Gaussian noise, which fail in many applications where the noise exhibits heavy tails. In the following, we aim to address these limitations by considering the full misspecified case and heavy-tailed noise scenarios.

## 4 Main Results

In this section, we present the main theoretical results regarding the learning rates of the KRR-RF estimator under light-tailed and heavy-tailed noise conditions. For an estimator ${ \widehat { f } } ,$ we are interested in bounding the excess risk, i.e., $\mathcal { E } ( \widehat { f } ) - \mathcal { E } ( f _ { \rho } ) = \| \widehat { f } - f _ { \rho } \| _ { L _ { \rho _ { X } } ^ { 2 } } ^ { 2 }$ . We first introduce the common assumptions that will be used in both the light-tailed and heavy-tailed noise scenarios.

## 4.1 Assumptions

Assumption 1 (Bounded random feature condition). Assume the kernel K admits the integral representation in (8) with $\psi$ being bounded and continuous in both variables. Specifically, there exists a constant $\kappa \geq 1$ such that $| \psi ( \pmb { x } , \pmb { w } ) | \le \kappa$ for all $\mathbf { \boldsymbol { x } } \in \mathcal { X }$ and $\pmb { w } \in \Omega$ . The associated RKHS H is separable.

Assumption 1 requires the feature function to be bounded and continuous, which implies that the kernel is bounded, $\begin{array} { r } { \mathrm { i . e . , s u p } _ { { \pmb x } \in \mathcal { X } } K ( { \pmb x } , { \pmb x } ) \le \kappa ^ { 2 } } \end{array}$ by (8). This assumption is standard in the analysis of random feature-based kernel methods (Rudi and Rosasco, 2017; Li et al., 2021).

Assumption 2 (Source condition). Suppose the regression function $f _ { \rho } \in [ \mathcal { H } ] ^ { 2 r }$ for some $r \in ( 0 , 1 ]$ and $\| f _ { \rho } \| _ { [ \mathcal { H } ] ^ { 2 r } } \leq R$ for some constant $R > 0$ , which is also equivalent to assuming that there exists $h _ { \rho } \in L _ { \rho _ { X } } ^ { 2 }$ such that

$$
f _ { \rho } = L _ { K } ^ { r } h _ { \rho } , \| h _ { \rho } \| _ { L _ { \rho _ { X } } ^ { 2 } } \leq R
$$

and $L _ { K } ^ { r }$ denotes the r-th power of the integral operator $L _ { K }$

Assumption 2 is commonly referred to as the source condition (Smale and Zhou, 2007; Caponnetto and De Vito, 2007; Rudi and Rosasco, 2017). It characterizes the regularity of the target function $f _ { \rho }$ with respect to the RKHS H, where a larger value of r indicates a smoother target function. When $r \in [ 1 / 2 , 1 ]$ ], the target function $f _ { \rho }$ belongs to the RKHS H itself. When $r \in ( 0 , 1 / 2 )$ the target function $f _ { \rho }$ may lie outside the RKHS ${ \mathcal { H } } ,$ capturing the misspecified case. Existing analyses typically focus on the well-specified case where $r \geq 1 / 2$ (Rudi and Rosasco, 2017; Avron et al., 2017; Li et al., 2021) or part of the misspecified case where $2 r + \gamma > 1$ . Our work aims to address the full misspecified regime where $r \in ( 0 , 1 / 2 )$

![](images/950487898e8b42d3071dcb73f95cac72feba9a05cdf382810c7477a17122a57a.jpg)  
Figure 1: Illustration of the well-specified cases and misspecified case.

Assumption 3 (Capacity Condition). For any $\lambda > 0$ , let the efective dimension be defined as $\mathcal { N } ( \lambda ) = \mathrm { T r } ( ( L _ { K } + \lambda I ) ^ { - 1 } L _ { K } )$ . Assume that there exist constants $Q > 0$ and $\gamma \in [ 0 , 1 ]$ such that

$$
{ \mathcal { N } } ( \lambda ) \leq Q ^ { 2 } \lambda ^ { - \gamma } .
$$

The capacity of the RKHS is measured by the efective dimension, defined as

$$
\mathcal { N } ( \lambda ) = \operatorname { T r } \bigl [ ( L _ { K } + \lambda I ) ^ { - 1 } L _ { K } \bigr ] = \sum _ { i = 1 } ^ { \infty } \frac { \mu _ { i } } { \mu _ { i } + \lambda } , \qquad \lambda > 0 ,\tag{12}
$$

where $\{ \mu _ { i } \} _ { i \geq 1 }$ are the eigenvalues of the integral operator $L _ { K }$ in (4). The efective dimension counts the number of eigenvalues that are significantly larger than λ, and it determines the complexity of the hypothesis space at the regularization scale λ: a slower eigenvalue decay leads to a larger $\mathcal { N } ( \lambda )$ and hence a more complex model. Assumption 3 always holds when $\gamma = 1$ by taking $Q = \mathrm { T r } ( L _ { K } ) \leq \kappa ^ { 2 }$ . It is satisfied when the eigenvalues $\{ \mu \} _ { i \geq 1 }$ $L _ { K }$ have a polynomial decay, i.e., $\mu _ { i } \leq C i ^ { - 1 / \gamma }$

$$
\mathcal { N } ( \lambda ) = \sum _ { i = 1 } ^ { \infty } \frac { \mu _ { i } } { \mu _ { i } + \lambda } \leq \sum _ { i = 1 } ^ { \infty } \frac { C } { C + i ^ { 1 / \gamma } \lambda } \leq \int _ { 0 } ^ { \infty } \frac { C } { C + t ^ { 1 / \gamma } \lambda } d t \leq \lambda ^ { - \gamma } \int _ { 0 } ^ { \infty } \frac { C } { C + t ^ { 1 / \gamma } } d t \leq C _ { u } \lambda ^ { - \gamma } ,
$$

for some constant $C _ { u }$

The following assumption is a key condition to establish the learning rates of KRR-RF under misspecification. Typically, an RKHS H has the embedding property into $L ^ { \infty } ( X )$ of order $\alpha \in [ \gamma , 1 ]$ if there exists a constant $M _ { \alpha } > 0$ such that

$$
\| { \mathcal { H } } | ^ { \alpha } \to L ^ { \infty } ( X ) \| \leq \sqrt { M _ { \alpha } }\tag{13}
$$

Fischer and Steinwart (2020) proved that (13) holds if and only if $\textstyle \sum _ { k = 1 } ^ { \infty } \mu _ { k } ^ { \alpha } e _ { k } ^ { 2 } ( { \pmb x } ) \le M _ { \alpha }$ for $\rho _ { \mathcal { X } ^ { - } }$ almost all $\mathbf { \boldsymbol { x } } \in \mathcal { X }$ . If there is no such $M _ { \alpha }$ satisfying this condition, we set $M _ { \alpha } = \infty$ . Note that $\begin{array} { r } { \operatorname* { s u p } _ { { \pmb x } \in \mathcal { X } } K ( { \pmb x } , { \pmb x } ) \le \kappa ^ { 2 } } \end{array}$ , it is always true that $M _ { \alpha } \ge \kappa$ for $\alpha \geq 1$ . By the inclusion relationship of interpolation spaces, we have H has the embedding property of order $\alpha ^ { \prime }$ whenever it has the embedding property of order α for any $\alpha ^ { \prime } \geq \alpha$ . Then we introduce the embedding index $\alpha _ { 0 }$ to characterize the smallest order of the embedding property, which first appears in Zhang et al. (2023, 2024).

Assumption 4 (Embedding condition). Suppose that there exists $\alpha _ { 0 } > 0$ , such that

$$
\alpha _ { 0 } = \operatorname* { i n f } \left\{ \alpha \in [ \gamma , 1 ] : \| { \mathcal { H } } | ^ { \alpha } \to L ^ { \infty } ( X ) \| < \infty \right\} ,
$$

where we call $\alpha _ { 0 }$ the embedding index of an RKHS H.

This assumption implies that for any $\alpha > \alpha _ { 0 }$ , the RKHS H has the embedding property of order $\alpha , \ \mathrm { i . e . }$ , there exists a constant $M _ { \alpha } \ > \ 0$ such that (13) holds. However, the embedding property of order $\alpha _ { 0 }$ itself may not hold, meaning that $M _ { \alpha _ { 0 } }$ could be infinite. The case $\alpha = \gamma$ corresponds to the benign RKHSs, which implies the RKHS feature space is highly homogeneous and free of extreme leverage points or spatial spikes. Benign RKHSs including RKHS with uniformly bounded eigenfunctions, Sobolev RKHS, RKHS with shift-invariant periodic kernels and so on. More examples and detailed discussions can be found in Section 4 of Zhang et al. (2024).

Remark 3. Assumption 4 implies the maximal capacity condition used in our analysis, $i . e . ,$ $\begin{array} { r } { \mathcal { N } _ { \infty } ( \lambda ) = \mathrm { e s s s u p } _ { { \pmb x } \in X } \sum _ { j } \frac { \mu _ { j } } { \mu _ { j } + \lambda } e _ { j } ^ { 2 } ( { \pmb x } ) \leq M _ { \alpha } \lambda ^ { - \alpha } } \end{array}$ for any $\alpha > \alpha _ { 0 }$ , which is fundamental to our analysis in the misspecified setting. We provide its proof in Lemma 30 of Appendix G. Although this condition may not be new and appears in the literature of KRR (Fischer and Steinwart, 2020; Zhang et al., 2023) and general spectral algorithms (Zhang et al., 2024), it is nontrivial to involve in the analysis of $K R R – R F$ under misspecified settings. This is because the error decomposition and operator representation in KRR-RF is totally diferent from that in standard KRR. It plays a crucial role in deriving a high-probability bound on the leverage-type quantity, i.e., $\| C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( \pmb { x } ) \| _ { 2 } ^ { 2 } \leq 8 M _ { \alpha } \lambda$ −α for any $\alpha > \alpha _ { 0 }$ outside an exceptional set of measure at most $O ( 1 / n )$ . Its proof is highly nontrivial and relies on some technical lemmas presented in Appendix C. In this way, the embedding index α determines the pointwise behavior of the leverage-type quantity, which replaces the uniform worst-case bound $\kappa ^ { 2 } / \lambda$ used in all of the previous analyses (Rudi and Rosasco, 2017; Liu et al., 2021; Li and Liu, 2023; Wang et al., 2026). This refinement significantly improves the tightness of the analysis in the misspecified setting, leading to the optimal learning rates of KRR-RF when $2 r + \gamma > \alpha _ { 0 }$

## 4.2 Learning Rates of KRR-RF in the Light-tailed Noise Settings

We first focus on the learning rates of KRR-RF under the light-tailed noise condition specified in the following assumption.

Assumption 5 (Light-tailed noise condition). Suppose the noise satisfies that $\mathbb { E } ( \varepsilon | \mathbf { x } ) = 0$ . There exist $\sigma \geq 0$ and $B \geq 0$ such that for all $p \geq 2$

$$
\mathbb { E } ( | \varepsilon | ^ { p } | \pmb { x } ) \le \frac { 1 } { 2 } p ! \sigma ^ { 2 } B ^ { p - 2 } .
$$

Assumption 5 is commonly referred to as the light-tailed noise condition, which ensures that the noise has sub-exponential tails. It is satisfied by many common noise distributions, including bounded, Gaussian and sub-Gaussian noise.

Theorem 4. Under Assumptions 1-5 with $0 < r \leq 1 , \gamma \leq \alpha _ { 0 } \leq 1$ , assume $2 r + \gamma > \alpha _ { 0 }$ , and let $\lambda = n ^ { - \frac { 1 } { 2 r + \gamma } }$ . Suppose the number of random features satisfies

$$
M \stackrel { > } { \sim } \left\{ \begin{array} { l l } { n ^ { \frac { 1 } { 2 r + \gamma } } \log \frac { c n } { \delta } , } & { 0 < r < \frac { 1 } { 2 } , } \\ { n ^ { \frac { 1 + \gamma ( 2 r - 1 ) } { 2 r + \gamma } } \log \frac { c n } { \delta } , } & { \frac { 1 } { 2 } \leq r \leq 1 , } \end{array} \right.\tag{14}
$$

for any $\delta \in ( 0 , 1 )$ , if the sample size $n \geq n _ { 0 }$ is large enough, then with probability at least $1 - \delta$ over the features and the samples,

$$
\begin{array} { r } { \left\| \widehat { f } _ { \lambda , M } - f _ { \rho } \right\| _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } \leq C _ { 1 } n ^ { - \frac { 2 r } { 2 r + \gamma } } \log ^ { 2 } \frac { c _ { 1 } } { \delta } , } \end{array}\tag{15}
$$

where $c , c _ { 1 } , C _ { 1 }$ are positive constants independent of $n _ { ; }$ , M and $\delta , n _ { 0 }$ is a suficiently large constant independent of n, M.

With the specified choice of the regularization parameter $\lambda = n ^ { - \frac { 1 } { 2 r + \gamma } }$ and the number of random features M satisfying (14), Theorem 4 guarantees that KRR-RF achieves the optimal learning rates of order $O ( n ^ { - \frac { 2 r } { 2 r + \gamma } } )$ with high probability. The derived bound aligns with those established in the literature (Rudi and Rosasco, 2017; Liu et al., 2021; Li and Liu, 2023; Wang et al., 2026). However, our results pertain to a broader range of source conditions satisfying $2 r + \gamma > \alpha _ { 0 }$ , extending beyond the well-specified case $r \geq 1 / 2$ (Rudi and Rosasco, 2017) and the misspecified case $2 r + \gamma \geq 1$ (Liu et al., 2021; Li and Liu, 2023; Wang et al., 2026). This extension comes from our tight bounds on the leverage-type quantity, i.e., $\| C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( \pmb { x } ) \| _ { 2 } ^ { 2 } \leq 8 M _ { \alpha } \lambda ^ { - \alpha }$ for any $\alpha > \alpha _ { 0 }$ outside an exceptional set of measure at most $O ( 1 / n )$ . It allows us to obtain improved bounds of (i) the similarity between operators $C _ { M , \lambda }$ and $\dot { C } _ { M , \lambda }$ , leading to the condition on the sample size from $n \gtrsim \lambda ^ { - 1 }$ to $n \gtrsim \lambda ^ { - \alpha }$ , and (ii) the noise error term, improving it roughly from $\mathcal { O } ( n ^ { - 1 } \lambda ^ { - 1 / 2 } + n ^ { - 1 / 2 } \lambda ^ { - \gamma / 2 } ) )$ ) to $\mathcal { O } ( n ^ { - 1 } \lambda ^ { - \alpha / 2 } + n ^ { - 1 / 2 } \lambda ^ { - \gamma / 2 } ) )$ . In the benign case when $\alpha _ { 0 } = \gamma$ , the optimal learning rates can be achieved for the full source range $r \in ( 0 , 1 ]$ We also note that the required number of random features M also recover the optimal scaling with respect to the sample size n as established in previous works $\Omega ( n ^ { \frac { 1 + \gamma ( 2 r - 1 ) } { 2 r + \gamma } } )$ for $r \geq 1 / 2$ (Rudi and Rosasco, 2017) and $\Omega ( n ^ { \frac { 1 } { 2 r + \gamma } } )$ for $r \le 1 / 2$ and $2 r + \gamma > 1$ (Li and Liu, 2023; Wang et al., 2026). This demonstrates the eficiency of the random feature approximation in maintaining the optimal learning rates while potentially reducing computational complexity, even for the tough cases when $2 r + \gamma \in ( \alpha _ { 0 } , 1 )$

Remark 5. The condition on M in the feature budget (14) suppresses an additional logarithmic factor log(cn) when $\alpha + \zeta = 1$ . In our proof, the spectral estimate in Lemma 21 produces an extra $\log ( 1 / \lambda )$ term, which propagates into Lemmas 22, 23, and Proposition 25 and adds a $\log ( 1 / \lambda )$ factor to the feature budget. However, this additional logarithmic factor is independent of δ and is negligible compared to main terms. Thus we omit it from the statement for clarity in this theorem and subsequent theorems. The full feature budget including the logarithmic factor can be found in the proofs.

In practice, random features are drawn directly from the spectral measure π in the integral representation (8). This uniform sampling scheme is easy to implement, but it ignores the data distribution and therefore typically demands a large number of features to reach the desired approximation accuracy. In the regime $2 r + \gamma < 1$ , the problem is more serious: our results show that uniform sampling requires $\begin{array} { r } { M \gtrsim n ^ { 1 / ( 2 r + \gamma ) } } \end{array}$ , which is larger than n. This defeats the original purpose of random features, which is to reduce the computational cost of KRR by using $M \ll n$ features. Consequently, to obtain the optimal learning rates with fewer random features in the regime $2 r + \gamma > \alpha _ { 0 }$ , we introduce the following compatibility condition, which is commonly used in the literature (Rudi et al., 2015; Rudi and Rosasco, 2017; Liu et al., 2020).

Assumption 6 (Compatibility condition). For any $\lambda > 0$ , define the maximum dimension of random features as $\begin{array} { r } { \mathcal { F } _ { \infty } ( \lambda ) = \operatorname* { s u p } _ { { \pmb w } \in \Omega } \left. ( L _ { K } + \lambda I ) ^ { - 1 / 2 } \boldsymbol \psi ( \cdot , { \pmb w } ) \right. _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } } \end{array}$ . Assume there exist constants $\zeta \in [ 0 , 1 ]$ and $F > 0$ , such that $\mathcal { F } _ { \infty } ( \lambda ) \leq F \lambda ^ { - \zeta }$

Assumption 6 quantifies the compatibility between the random feature representation and the data distribution through the maximum dimension ${ \mathcal { F } } _ { \infty } ( \lambda )$ . It always holds for $\zeta = 1$ and $F =$ $\kappa ^ { 2 }$ under Assumption 1, which corresponds to the standard uniform sampling scheme of random features. We also define the favorable case when $\zeta = \gamma _ { ; }$ , as it indicates a better alignment between the random feature representation and the data distribution. Theoretical example illustrating this scenario can be referred to in Rudi and Rosasco (2017).

Remark 6 (Diference between α and ζ). Both parameters impose polynomial decay on a leveragetype quantity, but they concern diferent objects. The embedding index α is intrinsic to the RKHS: it takes the supremum over the input space through the eigenfunctions of $L _ { K ; }$ , is independent of the feature construction. It determines the statistical dificulty of the problem and thereby controls the leverage-type quantity, which determines the regime boundary $2 r + \gamma > \alpha _ { 0 }$ in our main theorems. In contrast, $\zeta$ is a property of the feature representation: it takes the supremum over the feature parameter space Ω, depends on the sampling scheme $( e . g .$ , compatible sampling achieves the optimal value $\zeta = \gamma )$ . It enters our results only through the feature budget $M \gtrsim \lambda ^ { - \zeta }$ . Finally, we can not identify the leverage-type quantity $\| C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( \pmb { x } ) \| _ { 2 } ^ { 2 }$ with the maximum dimension $\mathcal { F } _ { \infty } ( \lambda ) = O ( \lambda ^ { - \zeta } )$ and derive the restriction $2 r + \gamma \geq \zeta$ from it. The two quantities are not equal because the supremum and the $L ^ { 2 }$ norm are taken over diferent variables: $\| C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( \pmb { x } ) \| ^ { 2 }$ is a pointwise quantity in $\pmb { x } \in X$ , whereas ${ \mathcal { F } } _ { \infty } ( \lambda )$ averages over x in $L _ { \rho _ { X } } ^ { 2 }$ and then takes the supremum over $\pmb { w } \in \Omega$

Theorem 7. Under Assumptions 1-6 with $0 < r \leq 1 , \gamma \leq \alpha _ { 0 } \leq 1 .$ , assume $2 r + \gamma > \alpha _ { 0 }$ , and let $\lambda = n ^ { - \frac { 1 } { 2 r + \gamma } }$ . Suppose the number of random features satisfies

$$
\begin{array} { r } { M \gtrsim \left\{ \begin{array} { l l } { n ^ { \frac { \zeta } { 2 r + \gamma } } \log \frac { c n } { \delta } , } & { 0 < r < \frac { 1 } { 2 } , } \\ { n ^ { \frac { \zeta + ( 1 + \gamma - \zeta ) ( 2 r - 1 ) } { 2 r + \gamma } } \log \frac { c n } { \delta } , } & { \frac 1 2 \leq r \leq 1 , } \end{array} \right. } \end{array}\tag{16}
$$

for any $\delta \in ( 0 , 1 )$ , if the sample size $n \geq n _ { 0 }$ is large enough, then with probability at least $1 - \delta$ over the features and the samples,

$$
\begin{array} { r } { \left\| \widehat { f } _ { \lambda , M } - f _ { \rho } \right\| _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } \leq C _ { 2 } n ^ { - \frac { 2 r } { 2 r + \gamma } } \log ^ { 2 } \frac { c _ { 2 } } { \delta } , } \end{array}\tag{17}
$$

where $c , c _ { 2 } , C _ { 2 }$ are positive constants independent of $n ,$ , M and $\delta , n _ { 0 }$ is a suficiently large constant independent of n, M.

The learning rate established in Theorem 7 is the same as those in Theorem 4, but the required number of random features is lower. Specifically, M reduces from $\Omega \big ( n ^ { \frac { 1 } { 2 r + \gamma } } \big )$ to $\Omega ( n ^ { \frac { \zeta } { 2 r + \gamma } } )$ when $\textstyle 0 < r < { \frac { 1 } { 2 } }$ and from $\Omega ( n ^ { \frac { 1 + \gamma ( 2 r - 1 ) } { 2 r + \gamma } } )$ to $\Omega ( n ^ { \overbar { \underbrace { \zeta + ( 1 + \gamma - \zeta ) } } ( 2 r - 1 ) } )$ when $\textstyle { \frac { 1 } { 2 } } \leq r \leq 1$ . These theoretical gains highlight the advantage of considering the compatibility parameter $\zeta$ in reducing the feature budget. We also note that condition (16) is the same as that in Li and Liu (2023)and Wang et al. (2026), but our analysis provides a more refined source condition $2 r + \gamma > \alpha _ { 0 }$ rather than $2 r + \gamma > 1$

![](images/cc86b249efc5e05e049a393825a19992925d86a60bf93099c939cd0eaaa2cb2e.jpg)

![](images/0ef8b654c4fab30f50aec393ab5ba5678d87cd97581ea06f9a1356ee38e009b7.jpg)

![](images/831f2576258687c92106f14d06640bfa03075d62bfa4eee3846a900a8969bfe5.jpg)

![](images/9de8c9e76f51a07d280f1cc1c2ba0814cb89f80337df3c8452fc77ceac9838d7.jpg)  
Figure 2: Exponent of n in the random-feature budget M in the benign RKHSs $\alpha _ { 0 } = \gamma$

From Remark 6, there is no general ordering between $\zeta$ and $\alpha _ { 0 }$ . Nevertheless, there exist benign cases in which two parameters coincide: for kernels whose eigenfunctions are well behaved $( \mathrm { e . g . }$ uniformly bounded), the embedding index attains its best value $\alpha _ { 0 } ~ = ~ \gamma$ , and with compatible sampling the compatibility parameter attains its optimal value $\zeta = \gamma _ { ; }$ , so that $\alpha _ { 0 } = \zeta = \gamma$ . In this case, our results show that KRR-RF achieves the minimax optimal learning rate for all $r \in ( 0 , 1 ]$ Figure 2 illustrates the exponent of n in the random-feature budget M for the benign RKHSs with $\alpha _ { 0 } = \gamma$ . Panels (a) and (b) correspond to the misspecified case with uniform and compatible sampling, respectively. Panels (c) and (d) correspond to the well-specified case with uniform and compatible sampling, respectively.

Corollary 8. Under the conditions in Theorem $^ { 7 , }$ if we consider the benign RKHSs and compatible random features with $\alpha _ { 0 } = \zeta = \gamma$ , then for all $0 < r \leq 1$ , suppose the number of random features satisfies

$$
M \stackrel { > } { \sim } \left\{ \begin{array} { l l } { n ^ { \frac { \gamma } { 2 r + \gamma } } \log \frac { c n } { \delta } , } & { 0 < r < \frac { 1 } { 2 } , } \\ { n ^ { \frac { 2 r + \gamma - 1 } { 2 r + \gamma } } \log \frac { c n } { \delta } , } & { \frac { 1 } { 2 } \le r \le 1 , } \end{array} \right.\tag{18}
$$

for any $\delta \in ( 0 , 1 )$ , if the sample size $n \geq n _ { 0 }$ is large enough, then with probability at least $1 - \delta$ over the features and the samples,

$$
\begin{array} { r } { \left\| \widehat { f } _ { \lambda , M } - f _ { \rho } \right\| _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } \leq C _ { 3 } n ^ { - \frac { 2 r } { 2 r + \gamma } } \log ^ { 2 } \frac { c _ { 3 } } { \delta } , } \end{array}\tag{19}
$$

where $c , c _ { 3 } , C _ { 3 }$ are positive constants independent of $n ,$ M and $\delta ,$ $n _ { 0 }$ is a suficiently large constant independent of $n , M$

For the compatible random features, the required number of random features $M$ is always sublinear in the sample size $n ,$ which ensures computational eficiency while achieving the minimax optimal learning rate for the whole source range of $0 < r \leq 1$

## 4.3 Learning Rates of KRR-RF in the Heavy-tailed Noise Settings

We next establish the learning rates of KRR-RF under the heavy-tailed noise condition specified in Assumption 7.

Assumption 7 (Heavy-tailed noise condition). For all $\mathbf { \boldsymbol { x } } \in \mathcal { X }$ , assume there exist some constants $\sigma , B > 0$ , and $p \in \mathbb N$ with $p \geq 3$ such that

$$
\begin{array} { r } { \mathbb { E } ( \varepsilon | \pmb { x } ) = 0 , \quad \mathbb { E } ( \varepsilon ^ { 2 } | \pmb { x } ) \leq \sigma ^ { 2 } , \quad a n d \quad \mathbb { E } ( | \varepsilon | ^ { p } | \pmb { x } ) \leq B . } \end{array}
$$

Compared to Assumption 5, Assumption 7 is a weak moment condition on the noise and only requires a higher conditional absolute moment $p \geq 3$ to be bounded. It allows the noise to have polynomially decaying tails, such as the Student-t distribution with more than three degrees of freedom, or centered Pareto and Fr´echet distributions with shape parameter larger than three. Under this condition, classical exponential Bernstein concentration inequalities no longer apply, and our analysis instead relies on a Hilbert-valued Fuk–Nagaev inequality (Lemma 29) to control the noise term.

Proposition 9. Under Assumptions 1, 2, 4, 6, 7 with $0 < r \leq 1 , \gamma \leq \alpha _ { 0 } < 1$ and $\alpha > \alpha _ { 0 }$ , let $\lambda \in ( 0 , \operatorname* { m i n } \{ 1 , \| L _ { K } \| \} )$ , suppose the number of random features satisfies

$$
\begin{array} { r } { M \gtrsim \left\{ \begin{array} { l l } { \lambda ^ { - \zeta } \log \frac { c n } { \lambda \delta } , } & { 0 < r < \frac { 1 } { 2 } , } \\ { \lambda ^ { - \{ \zeta + ( 1 + \gamma - \zeta ) ( 2 r - 1 ) \} } \log \frac { c n } { \lambda \delta } , } & { \frac { 1 } { 2 } \leq r \leq 1 , } \end{array} \right. } \end{array}\tag{20}
$$

for any $\delta \in ( 0 , 1 )$ , and the sample size n satisfies

$$
n \geq \frac { 8 3 2 } { 3 } M _ { \alpha } \lambda ^ { - \alpha } \log \frac { 2 0 M } { \delta } ,\tag{21}
$$

then with probability at least $1 - \delta$ over the features and the samples,

$$
\begin{array} { r } { \| \widehat { f } _ { \lambda , M } - f _ { \rho } \| _ { L ^ { 2 } ( \rho _ { X } ) } \leq C R \left( \sqrt { \frac { M _ { \alpha } \lambda ^ { 2 r - \alpha } } { n \delta } } + \lambda ^ { r } \right) + C \left( \frac { B ^ { \frac { 1 } { r } } \left( M _ { \alpha } \lambda ^ { - \alpha } \right) ^ { \frac { 1 } { 2 } - \frac { 1 } { p } } \mathscr { N } ( \lambda ) ^ { \frac { 1 } { p } } } { \delta ^ { \frac { 1 } { p } } n ^ { 1 - \frac { 1 } { p } } } + \sigma \sqrt { \frac { { \mathscr { N } } ( \lambda ) \log \frac { c } { \delta } } { n } } \right) , } \end{array}
$$

where $c , C$ are positive constants independent of n, M and δ

Proposition 9 provides a error decomposition for the estimator $\widehat { f } _ { \lambda , M }$ under the weak moment condition on the noise. For the light-tailed noise, the bound can be further simplified using classical Bernstein concentration inequalities, i.e.,

$$
\begin{array} { r } { \left\| \widehat { f } _ { \lambda , M } - f _ { \rho } \right\| _ { L ^ { 2 } ( \rho _ { X } ) } \leq C R \left( \sqrt { \frac { M _ { \alpha } \lambda ^ { 2 r - \alpha } } { n \delta } } + \lambda ^ { r } \right) + C \left( \frac { B \sqrt { M _ { \alpha } \lambda ^ { - \alpha } } \log \frac 2 \delta } { n } + \sigma \sqrt { \frac { N ( \lambda ) \log \frac 2 \delta } { n } } \right) . } \end{array}
$$

The key diference is the additional term $O \big ( \lambda ^ { \frac { ( 2 - p ) \alpha } { 2 p } } , N ( \lambda ) ^ { \frac { 1 } { p } } \delta ^ { - \frac { 1 } { p } } n ^ { - ( 1 - \frac { 1 } { p } ) } \big )$ in the bound. This term arises due to the heavy-tailed nature of the noise, which we use a recently developed Hilbert-valued Fuk–Nagaev inequality (Fuk, 1973; Nagaev, 1979; Yurinsky, 2006; Mollenhauer et al., 2026) instead of classical Bernstein concentration inequalities. It yields polynomial (rather than logarithmic) dependence on the confidence level δ. Using Assumption 3 that $\mathcal { N } ( \lambda ) \lesssim \lambda ^ { - \gamma }$ and $\lambda = n ^ { - \frac { 1 } { 2 r + \gamma } }$ , one verifies that polynomial term will be dominated by the logarithmic term whenever n satisfies the explicit requirement

$$
\begin{array} { r } { n \gtrsim \delta ^ { - \frac { 2 ( 2 r + \gamma ) } { ( p - 2 ) ( 2 r + 2 \gamma - \alpha ) } } \Rightarrow \lambda ^ { \frac { ( 2 - p ) \alpha } { 2 p } } \mathscr { N } ( \lambda ) ^ { \frac { 1 } { p } } \delta ^ { - \frac { 1 } { p } } n ^ { - ( 1 - \frac { 1 } { p } ) } \lesssim \lambda ^ { r } , \quad \mathscr { N } ( \lambda ) ^ { 1 / 2 } n ^ { - 1 / 2 } \lesssim \lambda ^ { r } . } \end{array}\tag{22}
$$

This leads to the conclusion that for suficiently large $n ,$ the heavy-tailed noise term does not afect the overall convergence rate, and the estimator achieves the same rate as in the light-tailed case.

Theorem 10. Under Assumptions ${ 1 \mathrm { - } } 4 , \ 6 ,$ 7 with $0 < r \leq 1 , \gamma \leq \alpha _ { 0 } \leq 1$ and $\alpha > \alpha _ { 0 }$ , assume $2 r + \gamma > \alpha _ { 0 }$ , and let $\lambda = n ^ { - \frac { 1 } { 2 r + \gamma } }$ . Suppose the number of random features satisfies (16) any $\delta \in ( 0 , 1 )$ , and if the sample size satisfies

$$
\begin{array} { r } { n \ge \operatorname* { m a x } \{ n _ { 0 } , c \delta ^ { - \frac { 2 ( 2 r + \gamma ) } { ( p - 2 ) ( 2 r + 2 \gamma - \alpha ) } } \} } \end{array}\tag{23}
$$

then with probability at least 1 − δ over the features and the samples,

$$
\left\| \widehat { f } _ { \lambda , M } - f _ { \rho } \right\| _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } \leq C _ { 4 } n ^ { - \frac { 2 r } { 2 r + \gamma } } \log ^ { 2 } \frac { c _ { 4 } } { \delta } ,\tag{24}
$$

where $c , c _ { 4 } , C _ { 4 }$ are positive constants independent of $n ,$ M and $\delta , n _ { 0 }$ is a suficiently large constant independent of n, M defined in Theorem 7.

It is worth emphasizing that under heavy-tailed noise, both the learning rates and the implementation of the algorithm remain the same as in the light-tailed case. Specifically, the estimator uses the same regularization parameter $\lambda = n ^ { - 1 / ( 2 r + \gamma ) }$ and the same random-feature budget M, neither of which depends on the moment parameter $p .$ . The only diference lies in the requirement of the sample size: a slightly larger n is needed to make the polynomial tail term produced by the Fuk–Nagaev inequality negligible. Hence, robustness to heavy-tailed noise is achieved at no cost in either the statistical rate or the computational complexity, only the sample size requirement is afected.

For better illustration, Figure 3 gives a simple example showing the minimum sample size required for the polynomial term to be dominated by the exponential term at a given confidence level $1 - \delta .$ , for diferent values of $p .$ We fix $\lambda = n ^ { - \frac { 1 } { 2 r + \gamma } } , r = 1 / 2$ , and $\alpha \approx \gamma$ , in which case the requirement (106) reduces to $n \gtrsim ( 1 / \delta ) ^ { \frac { 2 } { p - 2 } }$ . As shown in Figure 3, the minimum sample size increases as $p$ decreases or as $\delta$ decreases, which is consistent with the intuition that heavier-tailed noise (smaller p) or higher confidence (smaller δ) requires more samples to achieve the optimal learning rate.

The following two corollaries provide the results for the uniform and compatible random features under heavy-tailed noise, respectively.

Corollary 11. Under the conditions in Theorem 10, if we consider the benign RKHSs and uniformly random features with $\alpha _ { 0 } = \gamma$ and $\zeta = 1$ , then for all $0 < r \leq 1$ , suppose the number of random features satisfies (14) and the sample size satisfies (23) , then with probability at least $1 - \delta$ over the features and the samples,

$$
\begin{array} { r } { \left. \widehat { f } _ { \lambda , M } - f _ { \rho } \right. _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } \leq C _ { 5 } n ^ { - \frac { 2 r } { 2 r + \gamma } } \log ^ { 2 } \frac { c _ { 5 } } { \delta } , } \end{array}\tag{25}
$$

where $c , c _ { 5 } , C _ { 5 }$ are positive constants independent of $n ,$ , M and $\delta .$

Corollary 12. Under the conditions in Theorem 10, if we consider the benign RKHSs and compatible random features with $\alpha _ { 0 } = \zeta = \gamma$ , then for all $0 < r \leq 1$ , suppose the number of random features satisfies (18) and the sample size satisfies (23) , then with probability at least $1 - \delta$ over the features and the samples,

$$
\left\| \widehat { f } _ { \lambda , M } - f _ { \rho } \right\| _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } \leq C _ { 6 } n ^ { - \frac { 2 r } { 2 r + \gamma } } \log ^ { 2 } \frac { c _ { 6 } } { \delta } ,\tag{26}
$$

where $c , c _ { 6 } , C _ { 6 }$ are positive constants independent of n, M and $\delta .$

![](images/c318cfea21ede5954a905b8e52b4a8cfdc17706cb389bfbac59d4a69b15ca4da.jpg)  
Figure 3: Minimum sample size required for the polynomial term to be dominated by the exponential term at a given confidence level $1 - \delta ,$ , for diferent values of $p .$

## 4.4 Minimax Lower Bound

To demonstrate the optimality of the derived rates for KRR-RF, we recall the information-theoretic lower bound for KRR (Caponnetto and De Vito, 2007; Zhang et al., 2023). Since the KRR-RF estimator is itself a particular learning algorithm, any minimax lower bound established over the class of all estimators automatically applies to it. We impose the following standard two-sided eigenvalue decay condition.

Assumption 8. There exist constants $c _ { - } , c _ { + } > 0$ such that the eigenvalues $\{ \mu _ { i } \} _ { i \geq 1 }$ of $L _ { K }$ satisfy $c _ { - } j ^ { - 1 / \gamma } \leq \mu _ { j } \leq c _ { + } j ^ { - 1 / \gamma }$ for all $j \geq 1$ and $\gamma \in ( 0 , 1 )$

Theorem 13. Let $\rho _ { X }$ be a probability distribution on X such that the marginal $\rho _ { X }$ satisfies Assumption 8, and P be the class of all probability distributions on $X \times \mathbb { R }$ with marginal $\rho _ { X }$ satisfying Assumption 2 and Assumption 5. Then there exists a positive constant C and a distribution $\rho \in \mathcal P$ ，

$$
\operatorname* { i n f } _ { \widehat { f } } \operatorname* { s u p } _ { \rho \in \mathcal { P } } \mathbb { E } \big [ \| \widehat { f } - f _ { \rho } \| _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } \big ] \geq C n ^ { - \frac { 2 r } { 2 r + \gamma } } ,
$$

where the infimum is taken over all possible estimators $\widehat { f }$ based on n i.i.d. samples from $\rho .$

The proof of Theorem 13 follows from standard information-theoretic arguments for minimax lower bounds in KRR, as detailed in Caponnetto and De Vito (2007); Zhang et al. (2023). Note that the proof use a Gaussian noise in the regression model, which satisfies both the light-tailed condition (Assumption 5) and the weak moment condition (Assumption 7 with any $p \geq 3 )$ , this lower bound applies to the distribution classes considered in both Theorem 4 and Theorem 7. Consequently, the rates $n ^ { - \frac { 2 r } { 2 r + \gamma } }$ established in both theorems are minimax optimal up to logarithmic factors once n is large enough. Our contribution thus lies in providing matching upper bounds under the misspecified setting when $2 r + \gamma > \alpha _ { 0 }$ and the heavy-tailed noise condition.

## 5 Comparison with the Related Work

We compare our results with existing work on kernel ridge regression (KRR) and random features (RF) methods, and robust kernel methods.

## 5.1 Comparison with Kernel Ridge Regression (KRR)

Kernel ridge regression (KRR) is a classical method for nonparametric regression in reproducing kernel Hilbert spaces (RKHS). In the context of learning theory (Vapnik, 1999a,b), it has been extensively studied for its statistical properties and optimality under various assumptions. Early works (Cucker and Smale, 2002; De Vito et al., 2005; Wu et al., 2006; Smale and Zhou, 2007) established the capacity-independent learning rates for KRR, deriving bounds $O ( n ^ { - \frac { 2 r } { 2 r + 1 } } )$ solely from the source condition. Mendelson and Neeman (Mendelson and Neeman, 2010) further obtained capacity-independent minimax rates that cover targets outside the RKHS. Caponnetto and De Vito (2007) considered the polynomial decay of the eigenvalues of the RKHS and derived the capacitydependent learning rates $O ( n ^ { - \frac { 2 r } { 2 r + \gamma } } )$ of KRR with source condition. However, these results primarily focus on the well-specified setting where the target function lies within the RKHS.

Research on the misspecified settings, where $f _ { \rho }$ lies outside H, has developed along two lines. The first line relies on the source and capacity conditions alone: optimal rates were established for KRR with specific kernels (Wang and Jing, 2022) and for more general spectral algorithms (Lin et al., 2020a; Lin and Cevher, 2020), but these analyses require $2 r + \gamma \geq 1$ , which excludes the harder part of the misspecified settings. The second line combines the integral operator approach with the embedding property of the RKHS, which controls the sup-norm of functions in the interpolation spaces through an index $\alpha _ { 0 } \in [ \gamma , 1 ]$ . Initiated by Steinwart et al. (2009) and further developed by Fischer and Steinwart (2020), this line established optimal rates of KRR under the boundedness of $f _ { \rho } ,$ which implicitly requires $2 r > \alpha _ { 0 }$ . Based on this approach, the analysis in Zhang et al. (2023) extended the optimality results to Sobolev-type RKHSs for the full range of $r \in \mathsf { ( 0 , 1 ] }$ Zhang et al. (2024) further considered the general spectral algorithms beyond KRR and established their optimality under similar conditions. Our work difers from the above works in two aspects. First, they all study the exact kernel estimator, while we study KRR with random features, for which the proof techniques are completely diferent. For example, one of our contribution is a tight upper bound of the leverage-type quantity, which is not involved in the analysis of the exact kernel estimator. Second, they assume the noise is either bounded or light-tailed, whereas our results remain valid under heavy-tailed noise with only a finite $p \textmd { - }$ th moment.

Another line of research keeps the least squares loss and studies its behavior when the noise possesses finite moments. Mendelson (2016) developed sharp bounds for multiplier empirical processes, which provide the main technical tool for analyzing least squares without exponential concentration. Building on this, Han and Wellner (2019) derived convergence rates for the nonparametric least squares estimator under a finite p-th moment condition, and Mendelson (2015); Kuchibhotla and Patra (2022); Tong (2025a) obtained similar results in related settings. However, the learning rates in these works depend explicitly on the moment parameter p and are slower than the optimal rate under light-tailed noise. Recently, by leveraging a Hilbert-valued Fuk–Nagaev inequality,

Mollenhauer et al. (2026) derived optimal learning rates for regularized least squares over RKHS under heavy-tailed noise, which recovers the existing results for light-tailed noise. Their analysis, however, is restricted to the exact kernel estimator in the well-specified case $r \geq 1 / 2$ . Our work only extends the analysis to random features and the misspecified setting when $2 r + \gamma > \alpha _ { 0 }$ . This is nontrivial due to the additional approximation error introduced by random features and the challenges posed by heavy-tailed noise.

## 5.2 Comparison with Random Feature (RF) Methods

Random features were first introduced by Rahimi and Recht (2007, 2008) as a scalable approximation of kernel methods, where the kernel is approximated by the inner product of randomized finite-dimensional feature maps. Early works focused on the approximation quality and computational benefits of random features (Rahimi and Recht, 2007; Sriperumbudur and Szab´o, 2015). For the generalization performance, the seminal work by Rudi and Rosasco (2017) established the capacity-dependent learning rates $O ( n ^ { - \frac { 2 r } { 2 r + \gamma } } )$ for KRR with random features (KRR-RF), showing that with a suficient number of random features, the generalization performance can match that of the exact kernel method. The required number of features was further reduced by sampling strategies that exploit the data distribution, such as leverage score sampling (Bach, 2017; Avron et al., 2017). Random features have also been combined with distributed learning and general Lipschitz loss functions (Liu et al., 2021; Li et al., 2021; Wang et al., 2024; Wang and Feng, 2024). However, most of these results are restricted to the well-specified settings where $r \geq 1 / 2$ . Research on random features in the misspecified setting is relatively limited. Liu et al. (2021) studied KRR-RF in the misspecified setting and provided error analysis under certain conditions. Li and Liu (2023) further investigated this problem under distributed environments. Recently, Wang et al. (2026) proposed an adaptive parameter selection method for KRR-RF in the misspecified setting. However, all of these works require $2 r + \gamma > 1$ , which excludes the harder part of the misspecified regime. Thus none of the existing works fully address the misspecified regime when $2 r + \gamma < 1$

The restriction $2 r + \gamma \geq 1$ in these works stems largely from the use of the worst-case uniform bound $\| C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( \pmb { x } ) \| _ { 2 } ^ { 2 } \le \kappa ^ { 2 } / \lambda$ . This bound corresponds to the trivial case $\alpha = 1$ and ignores the embedding property of the RKHS. Our work leverages a more refined analysis that takes into account the embedding property of the RKHS, allowing us to handle the misspecified settings when $2 r + \gamma > \alpha _ { 0 }$ . For the benign RKHSs with $\alpha _ { 0 } = \gamma$ , our results can cover the full source range $r \in ( 0 , 1 ]$ . It is also worth noting the above works primarily focus on the light-tailed noise setting, whereas our analysis accommodates heavy-tailed noise with only a finite p-th moment.

Recently, Wang (2026) studied the generalization performance of robust learning with random features under the $( 1 + \delta ) – \mathrm { t h }$ moment condition. Their idea is to truncate the response variable to control the influence of heavy-tailed noise. Our analysis difers from this approach for three reasons. First, they impose a $( 1 + \delta )$ -th moment condition on the response for some $\delta > 0$ , which is diferent from our finite $p { \mathrm { - t h } }$ moment assumption with $p \geq 3$ . Second, their analysis relies on truncation techniques, whereas our approach directly handles heavy-tailed noise without truncation, leading to a more general and flexible framework. Third, the learning rates derived in their work depend explicitly on the moment parameter $\delta ,$ whereas our results achieve optimality independent of the moment parameter $p .$ The proof techniques and theoretical tools used in our work are also fundamentally diferent from those in Wang (2026), since their work only need to control the truncation error compared to that in Rudi and Rosasco (2017).

## 5.3 Comparison with Robust Kernel Methods

Robust kernel methods have been extensively studied in the literature, particularly in the context of heavy-tailed noise and outlier-robust learning. These methods typically involve modifying the loss function such as the Huber loss, the Tukey’s biweight loss or quantile loss to mitigate the influence of extreme observations. Christmann and Steinwart (2007) studied the relationship between the loss function and the tail behavior of the response. Feng and Wu (2022) and Liu and Lian (2026) analyzed kernel Huber regression and adaptive kernel Huber regression from a statistical learning viewpoint. Feng and Wu (2020) established generalization bounds for robust empirical risk minimization under $( 1 + \delta )$ -th moment conditions. Similar works include Tong and Ng (2025) on Ivanov regularized Huber regression and Tong (2025b) on robust regression with modified Huber’s loss. Quantile regression (Koenker and Bassett Jr, 1978; Li et al., 2007) estimates conditional quantiles instead of the conditional mean and is insensitive to heavy-tailed errors. Kernel quantile regression with random features was also studied in Wang and Feng (2024); Wang et al. (2024) without the light-tailed noise assumption. Nevertheless, quantile regression is not well-suited for conditional mean regression, which is the focus of this paper. There are also some works keep the squared loss but replace empirical risk minimization by a robust aggregation of the data. Representative examples include median-of-means estimators Hsu and Sabato (2014) and minimization of robust empirical risk Brownlees et al. (2015), which achieve sub-Gaussian-type deviations under weak moment assumptions.

We note that robust procedures for general losses and robust aggregation schemes are important topics in their own right, but they are beyond the scope of this paper. First, the learning rates in the aforementioned works depend explicitly on the moment parameter and deteriorate as the moment condition weakens. Moreover, no matching lower bounds were established, so it is unclear whether these rates are optimal. Second, methods based on losses other than the squared loss, and quantile loss even no longer target the conditional mean. The resulting estimators also have no closed-form solution, which leads to high computational cost in the kernel setting. In contrast, our KRR-RF estimator retains the squared loss and hence admits an explicit solution with low computational cost. Theoretically, it also achieves the minimax optimal rate $n ^ { - \frac { 2 r } { 2 r + \gamma } }$ uniformly over the misspecified range $2 r + \gamma > \alpha _ { 0 }$ , which is independent of the moment parameter $p .$

## 6 Simulated Experiments

In this section, we conduct numerical experiments to evaluate the proposed KRR-RF method, including (i) validating the learning rates under both light-tailed and heavy-tailed noise, (ii) examining the efect of the number of random features M, and (iii) comparing with robust kernel methods.

## 6.1 Simulation Setup

Suppose that $\mathcal { X } = [ 0 , 1 ]$ and the marginal distribution of x is uniform on [0, 1]. We consider the RKHS induced by the periodic Sobolev kernel

$$
K _ { \ell } ( x , x ^ { \prime } ) = \sum _ { k \in \mathbb { Z } } \mu _ { k , \ell } e ^ { 2 \pi \mathrm { i } k ( x - x ^ { \prime } ) } , \qquad \mu _ { k , \ell } = \left[ 1 + ( 2 \pi \ell k ) ^ { 2 } \right] ^ { - 2 } .
$$

For experiments (i), we set $\ell = 1$ , for experiments (ii) and (iii), we set $\ell = 1 / 2 \pi$ . For both choices of the kernel scale ℓ, the eigenfunctions of the integral operator $L _ { K }$ are the Fourier basis functions

$e _ { k } ( x ) = e ^ { 2 \pi \mathrm { i } k x }$ , with corresponding eigenvalues

$$
\mu _ { k , \ell } = \left[ 1 + ( 2 \pi \ell k ) ^ { 2 } \right] ^ { - 2 } \asymp ( 1 + | k | ) ^ { - 4 } , \qquad k \in \mathbb { Z } .
$$

Consequently, the efective dimension satisfies $\begin{array} { r } { \mathcal { N } ( \lambda ) = \sum _ { k \in \mathbb { Z } } \frac { \mu _ { k , \ell } } { \mu _ { k . \ell } + \lambda } \asymp \lambda ^ { - 1 / 4 } } \end{array}$ and capacity condition holds for $\gamma = 1 / 4$ . According to Zhang et al. (2024), the periodic Sobolev RKHS is benign RKHS with $\alpha _ { 0 } = \gamma = 1 / 4$ . The embedding condition holds for every $\alpha \in ( 1 / 4 , 1 )$ . We construct the true regression function as

$$
f _ { \rho , \ell } ( x ) = \frac { 1 } { c _ { 0 } } \sum _ { k = 1 } ^ { \infty } \frac { \mu _ { k , \ell } ^ { r } \sqrt { 2 } \cos ( 2 \pi k ( x - 0 . 3 7 ) ) } { \sqrt { k } \log ( e k ) } , \quad c _ { 0 } ^ { 2 } = \sum _ { k = 1 } ^ { \infty } \frac { 1 } { k \log ^ { 2 } ( e k ) } .
$$

By orthonormality of the Fourier basis, the source condition holds for $f _ { \rho , \ell } = L _ { K } ^ { r } g _ { \rho , \ell }$ with ${ g _ { \rho , \ell } } =$ $\begin{array} { r } { \frac { 1 } { c _ { 0 } } \sum _ { k = 1 } ^ { \infty } \frac { \sqrt { 2 } \cos \left( 2 \pi k \left( x - 0 . 3 7 \right) \right) } { \sqrt { k } \log \left( e k \right) } } \end{array}$ and $\| g _ { \rho , \ell } \| _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } = 1$ . Moreover, there holds that

$$
\left. f _ { \rho , \ell } \right. _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } = \frac { 1 } { c _ { 0 } ^ { 2 } } \sum _ { k = 1 } ^ { \infty } \frac { \mu _ { k , \ell } ^ { 2 r } } { k \log ^ { 2 } ( e k ) } \le 1 , \quad \left. f _ { \rho , \ell } \right. _ { \mathcal { H } _ { K _ { \ell } } } ^ { 2 } = \frac { 1 } { c _ { 0 } ^ { 2 } } \sum _ { k = 1 } ^ { \infty } \frac { \mu _ { k , \ell } ^ { 2 r - 1 } } { k \log ^ { 2 } ( e k ) } \asymp \frac { 1 } { c _ { 0 } ^ { 2 } } \sum _ { k = 1 } ^ { \infty } \frac { k ^ { 3 - 8 r } } { \log ^ { 2 } ( e k ) } ,
$$

where we use $\mu _ { k , \ell } \asymp ( 1 + | k | ) ^ { - 4 } \in ( 0 , 1 ]$ . Thus we can obtain that $f _ { \rho , \ell } \in \mathcal { H } _ { K _ { \ell } }$ when $r \geq 1 / 2$ and $f _ { \rho , \ell } \notin \mathcal { H } _ { K _ { \ell } }$ when $r < 1 / 2$

We consider the following data generating process for our simulations:

$$
y = f _ { \rho , \ell } ( x ) + \sigma \varepsilon ,
$$

where three noise error are considered: (i) Gaussian noise $\varepsilon \sim N ( 0 , 1 ) ; \ ( \mathrm { i i } )$ Student’s t noise $\varepsilon \sim t _ { 4 } / \sqrt { 2 }$ with 4 degrees of freedom; (iii) Pareto noise $\varepsilon \sim \sqrt { 4 8 / 5 } ( \mathrm { P a r e t o } ( 5 , 1 ) - 5 / 4 )$ . All three noise distributions have zero mean and unit variance. Here σ is the noise level, where the first two experiments use $\sigma = 1$ . Case [i] is the standard light-tailed distribution and satisfies assumption 5. Case [ii] and Case [iii] are heavy-tailed distributions with finite variance and satisfies Assumption 7 with $p = 3$ and $p = 4$

Random Features Construction. We consider two random feature representations of the periodic Sobolev kernel: the uniform random features and the compatible random features. We first consider the real orthonormal Fourier basis

$$
e _ { 0 } ( x ) = 1 , \quad e _ { k , c } ( x ) = \sqrt { 2 } \cos ( 2 \pi k x ) , \quad e _ { k , s } ( x ) = \sqrt { 2 } \sin ( 2 \pi k x ) , \quad k \geq 1 .
$$

The corresponding eigenvalues are 1 for $e _ { 0 } ( x )$ and $\mu _ { k , \ell }$ for both $e _ { k , c } ( x )$ and $e _ { k , s } ( x )$ with $k \geq 1$ . Let $\begin{array} { r } { S _ { \ell } = 1 + 2 \sum _ { k = 1 } ^ { \infty } \mu _ { k , \ell } . } \end{array}$ we draw a random index $\omega$ from $\Omega _ { \mathrm { U } } = \{ 0 \} \cup \{ ( k , c ) , ( k , s ) : k \geq 1 \}$ with probabilities

$$
\mathbb { P } ( \omega = 0 ) = \frac { 1 } { S _ { \ell } } , \qquad \mathbb { P } ( \omega = ( k , c ) ) = \mathbb { P } ( \omega = ( k , s ) ) = \frac { \mu _ { k , \ell } } { S _ { \ell } } .
$$

The corresponding feature is

$$
\psi _ { \mathrm { U } , \ell } ( x , \omega ) = \sqrt { S _ { \ell } } e _ { \omega } ( x ) .\tag{27}
$$

The representation is unbiased because

$$
\mathbb { E } _ { \omega } \big [ \psi _ { \mathrm { U } , \ell } ( x , \omega ) \psi _ { \mathrm { U } , \ell } ( x ^ { \prime } , \omega ) \big ] = e _ { 0 } ( x ) e _ { 0 } ( x ^ { \prime } ) + \sum _ { k = 1 } ^ { \infty } \mu _ { k , \ell } \big [ e _ { k , c } ( x ) e _ { k , c } ( x ^ { \prime } ) + e _ { k , s } ( x ) e _ { k , s } ( x ^ { \prime } ) \big ] = K _ { \ell } ( x , x ^ { \prime } ) .
$$

Since $\begin{array} { r } { \operatorname* { s u p } _ { x , \omega } | \psi _ { \mathrm { U } , \ell } ( x , \omega ) | ^ { 2 } \leq 2 S _ { \ell } } \end{array}$ , this is a bounded random-feature representation. Moreover, its compatibility quantity satisfies

$$
\mathcal { F } _ { \infty , \mathrm { U } } ( \lambda ) = \operatorname* { s u p } _ { \omega \in \Omega _ { \mathrm { U } } } \left. ( L _ { K _ { \ell } } + \lambda I ) ^ { - 1 / 2 } \psi _ { \mathrm { U } , \ell } ( \cdot , \omega ) \right. _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } = \operatorname* { s u p } _ { \omega \in \Omega _ { \mathrm { U } } } \frac { S _ { \ell } } { \mu _ { \omega } + \lambda } = \frac { S _ { \ell } } { \lambda } ,
$$

where $\mu _ { 0 } = 1$ and $\mu _ { ( k , c ) } = \mu _ { ( k , s ) } = \mu _ { k , \ell } .$ . Thus, Uniform RF satisfies Assumption 1 and 6 with $\zeta = 1$ . For the compatible random features, we draw $u \sim \mathrm { U n i f } [ 0 , 1 ]$ and set

$$
\psi _ { \mathrm { C } , \ell } ( x , u ) = \sum _ { k \in \mathbb { Z } } \sqrt { \mu _ { k , \ell } } e ^ { 2 \pi \mathrm { i } k ( x - u ) } = \frac { \cosh \left( \frac { 1 / 2 - d _ { \mathbb { T } } ( x , u ) } { \ell } \right) } { 2 \ell \sinh \left( \frac { 1 } { 2 \ell } \right) } ,\tag{28}
$$

where $d _ { \mathbb { T } } ( x , u ) = \operatorname* { m i n } \{ | x - u | , 1 - | x - u | \}$ . The representation is also unbiased by orthonormality of the Fourier basis

$$
\mathbb { E } _ { u } \big [ \psi _ { \mathrm { C } , \ell } ( x , u ) \psi _ { \mathrm { C } , \ell } ( x ^ { \prime } , u ) \big ] = \sum _ { k \in \mathbb { Z } } \mu _ { k , \ell } e ^ { 2 \pi \mathrm { i } k ( x - x ^ { \prime } ) } = K _ { \ell } ( x , x ^ { \prime } ) .
$$

Unlike the first representation, this construction has the favorable compatibility quantity

$$
\mathcal { F } _ { \infty , \mathbb { C } } ( \lambda ) = \operatorname* { s u p } _ { u \in [ 0 , 1 ] } \Big \| ( L _ { K _ { \ell } } + \lambda I ) ^ { - 1 / 2 } \psi _ { \mathbb { C } , \ell } ( \cdot , u ) \Big \| _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } = \sum _ { k \in \mathbb { Z } } \frac { \mu _ { k , \ell } } { \mu _ { k , \ell } + \lambda } = \mathcal { N } ( \lambda ) \asymp \lambda ^ { - 1 / 4 } .
$$

Thus, Compatible RF satisfies Assumption 1 and 6 with $\zeta = \gamma = 1 / 4$

Error Metric and Numerical Evaluation. Since the regression function is known in the simulation, we evaluate the squared population prediction error of an estimator ${ \widehat { f } } { \mathrm { : } }$

$$
\mathrm { E r r o r } = \int _ { 0 } ^ { 1 } ( \widehat { f } ( \pmb { x } ) - f _ { \rho , \ell } ( \pmb { x } ) ) ^ { 2 } d \boldsymbol { x } = \Vert \widehat { f } \Vert _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } - 2 \langle \widehat { f } , f _ { \rho , \ell } \rangle _ { L ^ { 2 } ( \rho _ { X } ) } + \Vert f _ { \rho , \ell } \Vert _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } .
$$

Unlike the empirical test MSE computed from noisy responses, this criterion directly measures the estimation error and excludes the irreducible observation noise. For uniform RF, the prediction error is evaluated through the Fourier coeficients using Parseval’s identity (Stein and Shakarchi, 2003). For KRR and compatible RF, the cross terms are computed using the spectral representation of the target function, while the squared norms of the fitted predictors are evaluated by piecewise Gauss–Legendre quadrature (Golub and Welsch, 1969; Davis and Rabinowitz, 1984). The regression function $f _ { \rho , \ell }$ is approximated by its first $J = 1 3 1 0 7 2$ Fourier coeficients and accounts for the remaining infinite series tail through numerical bounds.

## 6.2 Validation of the Learning Rates

We first validate the theoretical learning rates of KRR-RF under the light-tailed and heavy-tailed noise settings. We consider four diferent regression functions $f _ { \rho , \ell }$ with varying smoothness levels $r \in$ $\{ 0 . 1 , 0 . 2 , 0 . 4 , 0 . 7 5 \}$ , where the first three are less smooth and do not belong to the RKHS and the last one is suficiently smooth and belongs to the RKHS. In particular, $r = 0 . 1$ and $r = 0 . 2$ satisfy $2 r +$ $\gamma < 1$ , a regime not covered by existing theory, and $r = 0 . 1$ corresponds to an unbounded regression function. We vary the sample size $n \in \{ 1 2 8 , 1 8 1 , 2 5 6 , 3 6 2 , 5 1 2 , 7 2 4 , 1 0 2 4 , 1 4 4 8 , 2 0 4 8 , 2 8 9 6 , 4 0 9 6 \}$

![](images/e4a45bdbb37991167ab8fbaf10a0287dc30cfcc7af425e93ee7646431a03bb77.jpg)  
Figure 4: Learning rates under Gaussian (top row), standardized Student- $\cdot t _ { 4 }$ (middle row), and cen tered standardized Pareto(5) noise (bottom row). Columns correspond to $r \in \{ 0 . 1 , 0 . 2 , 0 . 4 , 0 . 7 5 \}$ Each panel shows log errors against log sample sizes for full KRR and Compatible RF. The dashed line marks the theoretical slope $- 2 r / ( 2 r + \gamma )$ .

For each $( r , n )$ setting, we compare the full KRR with KRR-RF methods using compatible random features. Following the theoretical results, we fix the number of random features by

$$
M _ { n } ( r ) = \left\lceil n ^ { \frac { 1 / 4 + ( 2 r - 1 ) _ { + } } { 2 r + 1 / 4 } } \log ( e n ) \right\rceil .
$$

The regularization parameter is set according to the theoretical guidance as well, i.e.,

$$
\lambda _ { n } ( r ) = C n ^ { - \frac { 1 } { 2 r + 1 / 4 } } ,
$$

where C is a constant chosen from $\{ 1 / 1 6 , 1 / 4 , 1 , 4 , 1 6 \}$ by an independent validation set. Each experiment is repeated 100 times and the error is averaged over all repetitions.

Figure 4 plots the logarithm of the error against logarithm of n for diferent settings. The black dashed lines have the theoretical slopes $\begin{array} { r } { - \frac { 2 { \bar { r } } } { 2 r + \gamma } = - \frac { 2 r } { 2 r + 1 / 4 } } \end{array}$ , namely −0.444, −0.615, −0.762, and −0.857 for increasing $r \in \{ 0 . 1 , 0 . 2 , 0 . 4 , 0 . 7 5 \}$ . Their slopes are fixed in advance, only their intercepts are fitted to the eleven full-KRR error values. In all the panels, the log-error decreases almost linearly in log n, and the slopes are close to the theoretical slope under all three noise distributions. For the less smooth functions with $r = 0 . 1$ and $r = 0 . 2$ where $2 r + \gamma < 1$ , the empirical slopes still match the theoretical predictions well, demonstrating the validity of our theory even in this previously unexplored regime. We also note that the slopes under $\mathrm { S t u d e n t } { - } t _ { 4 }$ and $\mathrm { P a r e t o } ( 5 )$ noise are almost the same as under Gaussian noise, which shows that heavy tails do not slow down the rate once the sample size is suficiently large. We further report the fitted slopes of compatible RF by regressing the log errors on log sample sizes at the five largest sample sizes. There is no systematic diference between the three noise distributions, which is consistent with our theory that the rate does not depend on the tail behavior.

Table 1: Learning-rate slopes for compatible RF, fitted by least squares to the $\log _ { 1 0 }$ mean errors at the final five sample sizes.
<table><tr><td>r</td><td>Theory</td><td>Gaussian</td><td>Student-t4</td><td>Pareto(5)</td></tr><tr><td>0.10</td><td>-0.4444</td><td>-0.4408</td><td>-0.4748</td><td>-0.4815</td></tr><tr><td>0.20</td><td>-0.6154</td><td>-0.5750</td><td>-0.6509</td><td>-0.6131</td></tr><tr><td>0.40</td><td>-0.7619</td><td>-0.7238</td><td>-0.7869</td><td>-0.7530</td></tr><tr><td>0.75</td><td>-0.8571</td><td>-0.8078</td><td>-0.8188</td><td>-0.9028</td></tr></table>

![](images/42f7847ebf2b4bfe8c8dea145d3f7cb1e573a5e2f9de42f8d284a681cf79cfe9.jpg)  
Figure 5: Efect of the number of random features M under Gaussian (top row), standardized $\mathrm { S t u d e n t } { - } t _ { 4 }$ (middle row), and centered standardized Pareto(5) noise (bottom row). Columns correspond to $r \in \{ 0 . 1 , 0 . 2 , 0 . 7 5 \}$ }. Each panel shows errors against M for full KRR, uniform RF, and compatible RF.

## 6.3 Efect of the Number of Random Features

We next investigate the efect of the number of random features on the learning performance of $\mathrm { K R R - R F }$ . We use the periodic Sobolev kernel with length scale $\ell ~ = ~ 1 / ( 2 \pi )$ , so that its eigenvalues are $\mu _ { k , \ell } ~ = ~ ( 1 + k ^ { 2 } ) ^ { - 2 }$ for $k \in \ \mathbb { Z } .$ We fix the training sample size at $n \ = \ 2 0 4 8$ and consider three source parameters $r ~ \in ~ \{ 0 . 1 , 0 . 2 , 0 . 7 5 \}$ We compare full KRR, uniform $\begin{array} { l c l } { \operatorname { R F } } & { ( \zeta } & { = } & { 1 ) } \end{array}$ , and compatible RF $\ l ( \zeta ~ = ~ 1 / 4 )$ by varying the number of random features $M \in \{ 1 6 , 3 2 , 6 4 , 9 6 , 1 2 8 , 1 9 2 , 2 5 6 , 3 8 4 , 5 1 2 , 7 6 8 , 1 0 2 4 \}$ . For a given r, the regularization parameter is selected in an independent pilot experiment using full KRR under Gaussian noise. Specifically, we minimize the mean squared population prediction error over $\lambda \in \{ 1 0 ^ { - 5 } , 1 0 ^ { - 4 . 5 } , \dots , 1 0 ^ { 0 } \}$ , using 20 pilot repetitions per candidate. This yields $\lambda = 1 0 ^ { - 4 }$ for $r = 0 . 1$ ， $\lambda = 1 0 ^ { - 3 }$ for $r = 0 . 2$ , and $\lambda = 1 0 ^ { - 2 }$ for $r = 0 . 7 5$ . For each r, the chosen λ is fixed across all three noise distributions, both random feature constructions, and all values of M.

Figure 5 shows how the prediction error changes with the number of random features M under the three noise distributions, where the red line represents the full KRR. Compatible RF $( \zeta = 1 / 4 )$ reaches the error level of full KRR with very few features in all nine panels. Its error is already close to the full KRR line at $M \approx 3 2$ and stays flat afterwards, so increasing M further brings almost no gain. However, uniform RF requires a much larger number of features to reach a comparable error level. The gap between the two sampling schemes shrinks as r grows: for $r = 0 . 7 5$ both methods converge to KRR within a few number of features and their curves almost coincide. These results are consistent with our theory: the feature budget required by KRR-RF is governed by the compatibility exponent $\zeta ,$ so a compatible sampling scheme $( \zeta = \gamma )$ attains the optimal rate with far fewer features than uniform sampling $( \zeta = 1 )$ , and the saving is most pronounced in the misspecified settings. Finally, the same pattern holds under all three noise distributions, which demonstrates the robustness of the KRR-RF approach once the sample size is suficiently large.

## 6.4 Comparison with Robust Kernel Methods

To further evaluate the competitive advantage of KRR-RF, specifically the trade-of between statistical eficiency and computational cost, we compare it with other four robust kernel methods with random features:

• TKRR-RF: Truncated KRR with random features (Wang, 2026). This method applies a truncation step to the heavy-tailed responses before performing KRR with random features. However, the truncation level needs to be carefully chosen to balance bias and robustness. We use the theoretically recommended truncation level from (Wang, 2026) in our experiments.

• KHR-RF: Kernel Huber Regression (Feng and Wu, 2022; Liu and Lian, 2026) with random features. This method employs the Huber loss (Huber and Ronchetti, 2011) to achieve robustness against heavy-tailed noise, with a tuning parameter controlling the trade-of between eficiency and robustness. We use parameters $a = c { \widehat { s } } .$ where $\widehat { s }$ is computed from three-fold out-of-fold KRR residuals on the training sample and equals their median absolute deviation divided by 0.675. We select $c \in \{ 0 . 5 , 1 , 2 , 4 , 8 , 1 6 , 3 2 , \infty \}$ from pilot experiments on the training sample and evaluate the performance on an independent validation set. The choice $c = \infty$ reduces KHR-RF to KRR-RF. We use the iteratively reweighted least squares algorithm for the convex Huber objective.

• KQR-RF: Kernel Quantile Regression (Li et al., 2007; Wang and Feng, 2024) with random features. This method focuses on estimating conditional quantiles of the response variable, providing robustness against heavy-tailed noise by targeting specific quantiles rather than the mean. We use the quantile level τ = 0.5 and ADMM algorithm (Boyd et al., 2011) with exact soft-thresholding of the median loss in our experiments.

![](images/5385b41b823d64c974ea497616730d8577a10c444110d7014fa6ee0b94e3b1af.jpg)  
Figure 6: Comparison of KRR-RF with robust kernel methods under Gaussian (left column), standardized Student-t (middle column), and centered standardized Pareto(5) noise (right column). The top row shows log errors against the noise level σ, while the bottom row shows training times.

• MoM-RF: Median-of-Means KRR-RF (Hsu and Sabato, 2014). This method divides the data into several blocks, computes the KRR-RF estimator within each block, and then takes the geometric median of these estimators to achieve robustness against heavy-tailed noise. The block size $B \in \{ 3 , 5 , 9 \}$ is selected from pilot experiments on the training sample and evaluated on an independent validation set. We use blockwise KRR-RF solutions followed by a modified Weiszfeld iteration (Weiszfeld and Plastria, 2009).

The experimental setup follows the same protocol as in the previous subsection. We fix the regression function with source parameter $r ~ = ~ 0 . 4$ and the training sample size at $n \ = \ 2 0 4 8$ We use the compatible sampling scheme with $\zeta = 1 / 4$ for these RF-based methods, and fix the number of random features $M = 5 1 2$ . The regularization parameter for all methods is selected from $1 0 ^ { - 3 } \cdot \{ 1 / 6 4 , 1 / 1 6 , 1 / 4 , 1 , 4 , 1 6 , 6 4 \}$ via an independent validation set. We consider three noise distributions with varying levels $\sigma \in \{ 0 . 2 5 , 0 . 5 , 1 , 2 , 4 \}$ . Figure 6 summarizes the prediction accuracy and computational cost of the six methods under the three noise distributions. The top row plots log errors against the noise level $\sigma ,$ while the bottom row reports the corresponding training time for each method.

Overall, the prediction errors increase as the noise level σ becomes larger, as expected. Under Gaussian noise, KRR, KRR-RF, TKRR-RF, KHR-RF, and MoM-RF exhibit very similar prediction errors over the entire range of noise levels, while KQR-RF is slightly less accurate. Under heavy-tailed noise, KHR-RF achieves the lowest prediction error, especially at moderate and large noise levels. However, KRR-RF shows competitive performance compared with KHR-RF, and outperforms TKRR-RF and MoM-RF in most cases. In contrast, KQR-RF shows a substantially larger squared error as $\sigma$ increases, especially in the Pareto noise. This is because for Gaussian and $\mathrm { S t u d e n t } { - } t _ { 4 }$ noise, the conditional mean and median coincide due to symmetry. However, the centered Pareto(5) noise is asymmetric, so its conditional median difers from the conditional mean.

The lower panels of Figure 6 demonstrate a clear computational advantage of random-feature methods. KRR-RF and TKRR-RF have the smallest training times among the six procedures, while KQR-RF is by far the most computationally expensive. KHR-RF and MoM-RF require substantially more time than KRR-RF because they involve iterative reweighting or repeated blockwise fitting and aggregation. Taken together, these experiments illustrate a trade-of between robustness and computational eficiency. Our KRR-RF can achieve competitive prediction performance while maintaining a low computation cost. More importantly, as described in Section 5, these robust methods are highly dependent on the moment parameters of the noise distribution in theory, whereas KRR-RF does not require such prior knowledge and remains efective across diferent noise settings.

## 7 Real Data Application

To further evaluate the practical performance of kernel ridge regression with random features, we conduct experiments on the cpusmall dataset from the LIBSVM Data Repository (https://www.csie.ntu.edu.tw/<sub>\~</sub>cjlin/libsvmtools/datasets/regression.html). The cpusmall dataset contains 8192 samples, each with 12 numerical features describing CPU characteristics. For example, Iread and Iwrite measure the numbers of reads and writes per second between system memory and user memory. The target variable usr is the percentage of time that the CPU runs in user mode, with larger values indicating a better performance. The details of all variables are summarized in Table 2.

We plot the histogram of the target variable usr in the left panel of Figure 7. Most values concentrate near the upper end of the range, with a small proportion of low values, so the distribution is moderately left-skewed. To construct a dataset suitable for studying heavy-tailed behavior, we first remove the samples with extremely low values, i.e., $y \leq 1 0$ . This excludes 294 samples, approximately 3.59% of the original dataset, and leaves 7,898 samples. We then transform the retained responses according to $y \mapsto 1 0 0 - y$ . After this transformation, the low-end extreme values in the original data become large values, and the distribution becomes right-skewed. Although the transformed target variable is bounded, but what matters for finite-sample behavior is the size of the moments rather than their existence. In practice, a bounded but strongly right-skewed distribution behaves like a heavy-tailed one when sample size is large. The histogram of the transformed target variable is shown in the right panel of Figure 7. We use these 7,898 samples and their transformed responses in the subsequent regression experiments.

We first investigate the sensitivity of KRR and KRR-RF to the regularization parameter λ and the number of random features M. We randomly select 3000 samples for training, 1000 samples for validation, and the remaining 3898 samples are used for testing. All input variables are standardized using the training data. We employ a Gaussian RBF kernel with bandwidth selected using the validation data. Specifically, let dmed denote the median pairwise Euclidean distance among the first 800 standardized training observations. We select the bandwidth from $( 0 . 5 d _ { m e d } , d _ { m e d } , 2 d _ { m e d } )$ by minimizing the validation RMSE. The corresponding random features are given in Example 1. Predictive performance is evaluated using the root mean squared error (RMSE) on the test data. In the first experiment, we fix M = 1024 for KRR-RF and vary λ from $1 0 ^ { - 6 } \ \mathrm { t o } \ 1 0 ^ { - 2 }$ . The left panel of Figure 8 presents the corresponding test RMSE. Both methods exhibit sensitivity to the regularization parameter. The test error initially decreases and then increases as λ grows. This behavior reflects the trade-of between insuficient and excessive regularization. The best parameters are $\lambda = 1 0 ^ { - 5 }$ for KRR and approximately $\lambda = 3 . 1 6 \times 1 0 ^ { - 6 }$ for KRR-RF. In the second experiment, we vary M over {64, 128, 256, 512, 1024, 2048, 4096}, fix bandwidths $d _ { m e d }$ for KRR and KRR-RF, and select λ using the validation set for each M. Since standard KRR does not depend on M, its test RMSE provides a horizontal reference line. As M increases, the test RMSE of KRR-RF decreases and eventually stabilizes near the test RMSE of standard KRR, indicating that a suficient number of random features can approximate the full kernel efectively.

![](images/a62c9bf6ab2b53836da5e2b1c536d4a605a2e8aeae7fb908ead27acf14ec8baf.jpg)

![](images/75dfd43b8766e9335a6753d07b1a511e279c0ead1103b29109f61f36e1bc54d6.jpg)  
Figure 7: Histogram of the target variable usr before (left) and after (right) transformation.

Table 2: Details of variables in the cpusmall dataset.
<table><tr><td>Variables Name</td><td></td><td>Description</td></tr><tr><td>x1</td><td>Iread</td><td>Reads (transfers per second) between system memory and user memory.</td></tr><tr><td>x2</td><td>lwrite</td><td>Writes (transfers per second) between system memory and user memory.</td></tr><tr><td>x3</td><td>scall</td><td>Number of system calls of all types per second.</td></tr><tr><td>x4</td><td>sread</td><td>Number of system read calls per second.</td></tr><tr><td>x5</td><td>swrite</td><td>Number of system write calls per second.</td></tr><tr><td>x6</td><td>fork</td><td>Number of system fork calls per second.</td></tr><tr><td>x7</td><td>exec</td><td>Number of system exec calls per second.</td></tr><tr><td>x8</td><td>rchar</td><td>Number of characters transferred per second by system read calls.</td></tr><tr><td>x9</td><td>wchar</td><td>Number of characters transferred per second by system write calls.</td></tr><tr><td> $x _ { 1 0 }$ </td><td>runqsz</td><td>Process run queue size.</td></tr><tr><td>x11</td><td></td><td>freemem Number of memory pages available to user processes.</td></tr><tr><td>x12</td><td></td><td>freeswap Number of disk blocks available for page swapping.</td></tr><tr><td>y</td><td>usr</td><td>Portion of time (%) that CPUs run in user mode.</td></tr></table>

We also compare the predictive performance of KRR and KRR-RF with diferent other four robust methods considered in the simulation study: TKRR-RF, KHR-RF, KQR-RF, MoM-RF. All methods use a Gaussian RBF kernel, and employ random Fourier features with M = 512. The regularization parameter λ is selected using the validation set for each method. For TKRR-RF, the training responses are truncated at their empirical 97.5-th percentile before fitting the model. The validation and test responses remain unchanged. For KHR-RF, KQR-RF, and MoM-RF, we follow the same algorithms as described in the simulation study. Figure 9 presents the test RMSE and training time across 20 repeated data splits. The left panel shows that KRR achieves the lowest average RMSE. The performance of KRR-RF, KHR-RF, MoM-RF is comparable, while TKRR-RF and KQR-RF exhibit slightly higher test errors. However, as shown in the right panel,

![](images/8072a9d3f2015c563710bc056a76852bf80b22bfa4c748f3b96038a104d22e50.jpg)

![](images/7c3868611597922160ae57c464c1bcde488b7740c3ea0e80829f1985204c15de.jpg)

Figure 8: Sensitivity of KRR and KRR-RF to the regularization parameter λ and the number of random features M.  
![](images/b29a1ab077babc903563a3d79fd2a425e274f85feba2ad288d0ccca804434ed6.jpg)  
Figure 9: Predictive performance and time comparison of KRR, KRR-RF, and other robust methods on the CPU dataset.

KRR-RF generally require less training time compared to standard KRR and other robust methods, highlighting the computational advantage of using random features and robust techniques. This is consistent with our conclusion in the simulation study.

At last, to check the noise characteristics of the cpusmall dataset, we analyze out-of-fold residuals obtained from KRR-RF estimator. We perform five-fold cross-fitting with five repeated random partitions. In each fold, KRR-RF is fitted using four subsets and evaluated on the held-out subset. Thus, every observation receives a prediction from a model that excludes that observation during training. We define the out-of-fold residuals as

$$
\hat { e } _ { i } = y _ { i } - \widehat { f } ^ { ( - k ( i ) ) } ( { \pmb x } _ { i } ) .
$$

We center the residuals by their median and standardize them using the MAD-based scale estimator. We then examine their empirical tail behavior using normal Q-Q plots, Hill’s tail-index estimator, and tail exceedance probabilities.

![](images/2076c3078fda0012bb3179d02ed821e14e86feead78122502fa5611d68f825b6.jpg)

![](images/92e8dc523304a333cd493d79ed98d6233c2c97c717fa29d3b1f6a89f3bbffb7c.jpg)

![](images/d9ba574cbbe8879f2e0c73d1be05063d62af1a4acf1d41cf584aa212fdc789f4.jpg)  
Figure 10: Normal Q-Q plot, Hill’s tail-index estimator, and tail exceedance probabilities of the standardized out-of-fold residuals for the cpusmall dataset.

The median out-of-fold RMSE over the five repetitions is 3.165, and the median MAD-based scale is 2.368. The standardized residuals are clearly heavier-tailed than Gaussian: 4.23% of them exceed three robust scale units, while the Gaussian value is only 0.27%, about 15.7 times smaller. The normal Q–Q plot shows a long upper tail, and the excess kurtosis is about 4.22. The Hill estimates decrease from about 4.00 to 3.08 as the tail fraction increases from 2% to 10%, so the estimated tail index depends on the threshold. We further split the observations into four groups by the CPU run-queue size runqsz, and compute the MAD-based scale and the three-MAD exceedance probability within each group. The residual scale grows from 1.67 in the lowest group to 2.88 in the highest group, and the exceedance probabilities range from 2.56% to 6.83%, all well above the Gaussian benchmark 0.27%. These results indicate that the noise in the cpusmall dataset is heteroscedastic and have much heavier tails than a Gaussian distribution.

## A Random Features Operators and Error Decomposition

Notation. We denote $\| \cdot \| _ { 2 } , \| \cdot \|$ and $\| \cdot \| _ { H S }$ as the Euclidean norm, the operatorial norm and Hilbert-Schmidt norm. Let L be a Hilbert space, we use $\langle \cdot , \cdot \rangle _ { \mathcal { L } }$ to denote its inner product, $\| \cdot \| _ { \mathcal { L } }$ to denote its norm, and $\operatorname { T r } ( \cdot )$ to denote the trace of an operator. Let $Q$ be a bounded self-adjoint linear operator on a separable Hilbert space ${ \mathcal { L } } ,$ we denote $Q _ { \lambda }$ as $Q _ { \lambda } = Q + \lambda I$ , where $\lambda > 0$ and I is the identity operator on ${ \mathcal { L } } .$ We denote $c , c _ { 0 } , c _ { 1 } , \ldots , C , C _ { 0 } , C _ { 1 } , \ldots$ . as a universal constant whose value may difer from line to line.

Definition 14. For all $g \in L ^ { 2 } ( \rho _ { X } ) , \beta \in \mathbb { R } ^ { M } , \alpha \in \mathbb { R } ^ { n }$ and $i \in \{ 1 , \ldots , M \}$ , we have

$$
\bullet s _ { M } : \mathbb { R } ^ { M }  L ^ { 2 } ( \rho _ { X } ) , \quad ( S _ { M } \beta ) ( \cdot ) = \phi _ { M } ( \cdot ) ^ { \top } \beta ,
$$

$$
\begin{array} { r } { \bullet S _ { M } ^ { \ast } : L ^ { 2 } ( \rho _ { X } )  \mathbb { R } ^ { M } , \quad ( S _ { M } ^ { \ast } g ) _ { i } = \frac { 1 } { \sqrt { M } } \int _ { X } \psi _ { \omega _ { i } } ( \pmb { x } ) g ( \pmb { x } ) d \rho _ { X } ( \pmb { x } ) , } \end{array}
$$

$$
\begin{array} { r } { \bullet \overrightarrow { S } _ { M } ^ { * } : L ^ { 2 } ( \rho _ { X } )  \mathbb { R } ^ { M } , \quad ( \overrightarrow { S } _ { M } ^ { * } g ) _ { i } = \frac { 1 } { n } \sum _ { j = 1 } ^ { n } g ( \pmb { x } _ { j } ) \phi _ { M } ( \pmb { x } _ { j } ) _ { i } , } \end{array}
$$

$$
\begin{array} { r } { \bullet L _ { M } : L ^ { 2 } ( \rho _ { X } )  L ^ { 2 } ( \rho _ { X } ) , \quad ( L _ { M } g ) ( \cdot ) = \int _ { X } K _ { M } ( \cdot , z ) g ( z ) d \rho _ { X } ( z ) , } \end{array}
$$

$$
\begin{array} { r } { \bullet \ C _ { M } : \mathbb { R } ^ { M }  \mathbb { R } ^ { M } , \quad C _ { M } = \int _ { X } \phi _ { M } ( \pmb { x } ) \phi _ { M } ( \pmb { x } ) ^ { \top } d \rho _ { X } ( \pmb { x } ) , } \end{array}
$$

$$
\begin{array} { r } { \bullet \widehat { C } _ { M } : \mathbb { R } ^ { M }  \mathbb { R } ^ { M } , \quad \widehat { C } _ { M } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \phi _ { M } ( \pmb { x } _ { i } ) \phi _ { M } ( \pmb { x } _ { i } ) ^ { \top } . } \end{array}
$$

From Rudi and Rosasco (2017) and Assumption 1, the linear operator L is trace class and the linear operators $L _ { M } , C _ { M } , S _ { M } , \widehat { C } _ { M } , \widehat { S } _ { M }$ are finite dimensional. It is easy to verify that $L _ { M } =$ $S _ { M } S _ { M } ^ { * } , C _ { M } = S _ { M } ^ { * } S _ { M }$ and $\widehat { C } _ { M } = \widehat { S } _ { M } ^ { * } \widehat { S } _ { M } = \overline { { S } } _ { M } ^ { * } S _ { M }$ . And $L _ { M } , C _ { M } , \widehat { C } _ { M }$ are self-adjoint and positive operators, with spectrum in $[ 0 , \kappa ^ { 2 } ]$ . From these operators, we can write the estimator as

$$
\widehat { f } _ { \lambda , M } = S _ { M } \widehat { C } _ { M , \lambda } ^ { - 1 } \widehat { S } _ { M } ^ { * } \widehat { \mathbf { y } } .
$$

If $\begin{array} { r } { \widehat { \mathbf g } _ { n } = \frac { 1 } { \sqrt { n } } ( g ( \pmb { x } _ { 1 } ) , \dots , g ( \pmb { x } _ { n } ) ) ^ { \top } } \end{array}$ , then $\overline { { { S } } } _ { M } ^ { * } g = \widehat { S } _ { M } ^ { * } \widehat { \mathbf { g } } _ { n }$

The following Proposition provides an error decomposition for the estimator $\widehat { f } _ { \lambda , M }$ in terms of noise error, the population RF error, and the empirical fluctuation of the population RF error.

Proposition 15 (Error decomposition). Let the feature realization $\omega _ { 1 } , \ldots , \omega _ { M }$ be fixed, then we can decompose the error as

$$
\begin{array} { r l } & { \left\| \widehat { f } _ { \lambda , M } - f _ { \rho } \right\| _ { L ^ { 2 } ( \rho _ { X } ) } \leq \underbrace { \left\| \widehat { C } _ { M , \lambda } ^ { - 1 / 2 } C _ { M , \lambda } ^ { 1 / 2 } \right\| ^ { 2 } \left\| C _ { M , \lambda } ^ { - 1 / 2 } \widehat { S } _ { M } ^ { * } \widehat { \varepsilon } \right\| _ { 2 } } _ { N o i s e \ r r o r } + \underbrace { \left\| f _ { \rho } - L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } \right\| _ { L ^ { 2 } ( \rho _ { X } ) } } _ { P o p u l a t i o n \ R F \ e r r o r } } \\ & { + \left\| \widehat { C } _ { M , \lambda } ^ { - 1 / 2 } C _ { M , \lambda } ^ { 1 / 2 } \right\| ^ { 2 } \left\| C _ { M , \lambda } ^ { - 1 / 2 } \left( \overline { { S } } _ { M } ^ { * } \left( f _ { \rho } - L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } \right) - S _ { M } ^ { * } \left( f _ { \rho } - L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } \right) \right) \right\| _ { 2 } , } \end{array}\tag{29}
$$

where $\begin{array} { r } { \widehat { \pmb { \varepsilon } } = \frac { 1 } { \sqrt { n } } ( \varepsilon _ { 1 } , \ldots , \varepsilon _ { n } ) ^ { \top } } \end{array}$

Proof. Since $\begin{array} { r } { y _ { i } = f _ { \rho } ( \pmb { x } _ { i } ) + \varepsilon _ { i } , \widehat { \mathbf { y } } = \frac { 1 } { \sqrt { n } } ( y _ { 1 } , \dots , y _ { n } ) ^ { \top } } \end{array}$ , and $\begin{array} { r } { \widehat { \pmb { \varepsilon } } = \frac { 1 } { \sqrt { n } } ( \varepsilon _ { 1 } , \ldots , \varepsilon _ { n } ) ^ { \top } } \end{array}$ , we have

$$
\widehat { f } _ { \lambda , M } - f _ { \rho } = S _ { M } \widehat { C } _ { M , \lambda } ^ { - 1 } \widehat { S } _ { M } ^ { * } \widehat { \pmb { \mathscr { G } } } - f _ { \rho } = \left( S _ { M } \widehat { C } _ { M , \lambda } ^ { - 1 } \overline { { S } } _ { M } ^ { * } f _ { \rho } - f _ { \rho } \right) + S _ { M } \widehat { C } _ { M , \lambda } ^ { - 1 } \widehat { S } _ { M } ^ { * } \widehat { \pmb { \mathscr { G } } } ,
$$

where we used $\widehat { S } _ { M } ^ { * } \widehat { \mathbf { y } } = \overline { { S } } _ { M } ^ { * } f _ { \rho } + \widehat { S } _ { M } ^ { * } \widehat { \pmb { \varepsilon } } .$ . By the triangle inequality,

$$
\begin{array} { r } { \| \widehat { f } _ { \lambda , M } - f _ { \rho } \| _ { L ^ { 2 } ( \rho _ { X } ) } \leq \big \| S _ { M } \widehat { C } _ { M , \lambda } ^ { - 1 } \overline { { S } } _ { M } ^ { * } f _ { \rho } - f _ { \rho } \big \| _ { L ^ { 2 } ( \rho _ { X } ) } + \big \| S _ { M } \widehat { C } _ { M , \lambda } ^ { - 1 } \widehat { S } _ { M } ^ { * } \widehat { \varepsilon } \big \| _ { L ^ { 2 } ( \rho _ { X } ) } . } \end{array}\tag{30}
$$

We bound the two terms separately.

Noise error. Since $\widehat { S } _ { M } ^ { \ast } \widehat { \varepsilon } \in \mathbb { R } ^ { M }$ and $\widehat { S } _ { M } ^ { * } \widehat { \pmb { \varepsilon } } = C _ { M , \lambda } ^ { 1 / 2 } C _ { M , \lambda } ^ { - 1 / 2 } \widehat { S } _ { M } ^ { * } \widehat { \pmb { \varepsilon } } ,$

$$
\begin{array} { r l } & { \left\| S _ { M } \widehat C _ { M , \lambda } ^ { - 1 } \widehat S _ { M } ^ { * } \widehat \varepsilon \right\| _ { L ^ { 2 } ( \rho _ { X } ) } = \left\| S _ { M } \widehat C _ { M , \lambda } ^ { - 1 } C _ { M , \lambda } ^ { 1 / 2 } C _ { M , \lambda } ^ { - 1 / 2 } \widehat S _ { M } ^ { * } \widehat \varepsilon \right\| _ { L ^ { 2 } ( \rho _ { X } ) } } \\ & { \qquad \leq \left\| S _ { M } \widehat C _ { M , \lambda } ^ { - 1 } C _ { M , \lambda } ^ { 1 / 2 } \right\| \| C _ { M , \lambda } ^ { - 1 / 2 } \widehat S _ { M } ^ { * } \widehat \varepsilon \| _ { 2 } . } \end{array}\tag{31}
$$

Population RF error and its empirical fluctuation. Recall that $Z ^ { * } f ( Z Z ^ { * } ) = f ( Z ^ { * } Z ) Z ^ { * }$ for any continuous spectral function f and any compact operator Z. By Definition 14, we have

$$
L _ { M } L _ { M , \lambda } ^ { - 1 } = S _ { M } S _ { M } ^ { * } \left( S _ { M } S _ { M } ^ { * } + \lambda I \right) ^ { - 1 } = S _ { M } \left( S _ { M } ^ { * } S _ { M } + \lambda I \right) ^ { - 1 } S _ { M } ^ { * } = S _ { M } C _ { M , \lambda } ^ { - 1 } S _ { M } ^ { * } ,\tag{32}
$$

and

$$
\begin{array} { r } { S _ { M } ^ { * } ( f _ { \rho } - L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } ) = S _ { M } ^ { * } f _ { \rho } - S _ { M } ^ { * } S _ { M } S _ { M } ^ { * } L _ { M , \lambda } ^ { - 1 } f _ { \rho } = S _ { M } ^ { * } f _ { \rho } - C _ { M } S _ { M } ^ { * } L _ { M , \lambda } ^ { - 1 } f _ { \rho } } \\ { = S _ { M } ^ { * } f _ { \rho } - C _ { M } C _ { M , \lambda } ^ { - 1 } S _ { M } ^ { * } f _ { \rho } = \lambda C _ { M , \lambda } ^ { - 1 } S _ { M } ^ { * } f _ { \rho } , \quad \quad } \end{array}\tag{33}
$$

where the last step uses $C _ { M , \lambda } - C _ { M } = \lambda I$ . Now we can write

$$
\begin{array} { r l } & { \quad S _ { M } \widehat { C } _ { M , \lambda } ^ { - 1 } \overline { { S } } _ { M } ^ { * } f _ { \rho } - f _ { \rho } = S _ { M } \widehat { C } _ { M , \lambda } ^ { - 1 } \big ( \overline { { S } } _ { M } ^ { * } f _ { \rho } - \widehat { C } _ { M , \lambda } C _ { M , \lambda } ^ { - 1 } S _ { M } ^ { * } f _ { \rho } \big ) + \big ( S _ { M } C _ { M , \lambda } ^ { - 1 } S _ { M } ^ { * } f _ { \rho } - f _ { \rho } \big ) } \\ & { = S _ { M } \widehat { C } _ { M , \lambda } ^ { - 1 } \big ( \overline { { S } } _ { M } ^ { * } f _ { \rho } - \widehat { C } _ { M } C _ { M , \lambda } ^ { - 1 } S _ { M } ^ { * } f _ { \rho } - \lambda C _ { M , \lambda } ^ { - 1 } S _ { M } ^ { * } f _ { \rho } \big ) + \big ( L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } - f _ { \rho } \big ) } \\ & { = S _ { M } \widehat { C } _ { M , \lambda } ^ { - 1 } \big ( \overline { { S } } _ { M } ^ { * } f _ { \rho } - \widehat { C } _ { M } C _ { M , \lambda } ^ { - 1 } S _ { M } ^ { * } f _ { \rho } - S _ { M } ^ { * } ( f _ { \rho } - L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } ) \big ) + \big ( L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } - f _ { \rho } \big ) } \\ &  = S _ { M } \widehat { C } _ { M , \lambda } ^ { - 1 } \big ( \overline { { S } } _ { M } ^ { * } \big ( f _ { \rho } - L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } \big ) - S _ { M } ^ { * } ( f _ { \rho } - L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } ) \big ) +  \end{array}\tag{34}
$$

where the second equality uses $\widehat { C } _ { M , \lambda } = \widehat { C } _ { M } + \lambda I$ and (32), the third equality uses (33). The last equality uses $\widehat { C } _ { M } C _ { M , \lambda } ^ { - 1 } S _ { M } ^ { * } f _ { \rho } = \overline { { S } } _ { M } ^ { * } L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho }$ , since every $w \in \mathbb { R } ^ { M }$ , there holds that

$$
\begin{array} { r } { \left. w , \widehat { C } _ { M } C _ { M , \lambda } ^ { - 1 } S _ { M } ^ { * } f _ { \rho } \right. _ { 2 } = \left. \widehat { C } _ { M } w , C _ { M , \lambda } ^ { - 1 } S _ { M } ^ { * } f _ { \rho } \right. _ { 2 } = \left. \overline { { S } } _ { M } ^ { * } S _ { M } w , C _ { M , \lambda } ^ { - 1 } S _ { M } ^ { * } f _ { \rho } \right. _ { 2 } } \\ { = \left. S _ { M } w , S _ { M } C _ { M , \lambda } ^ { - 1 } S _ { M } ^ { * } f _ { \rho } \right. _ { L ^ { 2 } ( \rho _ { X } ) } = \left. S _ { M } w , L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } \right. _ { L ^ { 2 } ( \rho _ { X } ) } } \\ { = \left. w , \overline { { S } } _ { M } ^ { * } ( L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } ) \right. _ { 2 } , } \end{array}
$$

where we use $\widehat { C } _ { M } = \overline { { S } } _ { M } ^ { * } S _ { M }$ , and (32).

Similar with (31), we can obtain that

$$
\begin{array} { r l } & { \big \| S _ { M } \widehat C _ { M , \lambda } ^ { - 1 } \overline { S } _ { M } ^ { * } f _ { \rho } - f _ { \rho } \big \| _ { L ^ { 2 } ( \rho _ { \mathcal { X } } ) } \leq \big \| f _ { \rho } - L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } \big \| _ { L ^ { 2 } ( \rho _ { \mathcal { X } } ) } } \\ & { \qquad + \left\| S _ { M } \widehat C _ { M , \lambda } ^ { - 1 } C _ { M , \lambda } ^ { 1 / 2 } \right\| \Big \| C _ { M , \lambda } ^ { - 1 / 2 } \left( \overline { S } _ { M } ^ { * } \big ( f _ { \rho } - L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } \big ) - S _ { M } ^ { * } \big ( f _ { \rho } - L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } \big ) \right) \Big \| _ { 2 } . } \end{array}\tag{35}
$$

Finally, we show that $\begin{array} { r } { \left\| S _ { M } \widehat { C } _ { M , \lambda } ^ { - 1 } C _ { M , \lambda } ^ { 1 / 2 } \right\| \leq \left\| \widehat { C } _ { M , \lambda } ^ { - 1 / 2 } C _ { M , \lambda } ^ { 1 / 2 } \right\| ^ { 2 } . } \end{array}$

$$
\left\| S _ { M } \widehat { C } _ { M , \lambda } ^ { - 1 } C _ { M , \lambda } ^ { 1 / 2 } \right\| \leq \left\| S _ { M } C _ { M , \lambda } ^ { - 1 / 2 } \right\| \left\| C _ { M , \lambda } ^ { 1 / 2 } \widehat { C } _ { M , \lambda } ^ { - 1 / 2 } \right\| \left\| \widehat { C } _ { M , \lambda } ^ { - 1 / 2 } C _ { M , \lambda } ^ { 1 / 2 } \right\| \leq \left\| \widehat { C } _ { M , \lambda } ^ { - 1 / 2 } C _ { M , \lambda } ^ { 1 / 2 } \right\| ^ { 2 } ,\tag{36}
$$

where we use $\left\| S _ { M } C _ { M , \lambda } ^ { - 1 / 2 } \right\| = \| C _ { M , \lambda } ^ { - 1 / 2 } S _ { M } ^ { * } S _ { M } C _ { M , \lambda } ^ { - 1 / 2 } \| ^ { 1 / 2 } = \| C _ { M } C _ { M , \lambda } ^ { - 1 } \| ^ { 1 / 2 } \leq 1 .$

Combining (30), (31), (35), and (36) gives (29). Thus we complete the proof.

## B Population RF Error

Lemma 16. Under Assumptions 1 and 2, the following holds for any $\lambda > 0$ and $M \in \mathbb { N }$

(i) (Misspecified case) if $\begin{array} { r } { 0 < r < \frac { 1 } { 2 } } \end{array}$ , then

$$
\left\| f _ { \rho } - L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } \right\| _ { L ^ { 2 } ( \rho _ { X } ) } \leq R \lambda ^ { r } \| L _ { M , \lambda } ^ { - 1 / 2 } L _ { K , \lambda } ^ { 1 / 2 } \| ^ { 2 r } ;
$$

(ii) (Well-specified case) $i f { \frac { 1 } { 2 } } \leq r \leq 1$ , then

$$
\begin{array} { r l } & { \quad \left\| f _ { \rho } - L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } \right\| _ { L ^ { 2 } ( \rho _ { X } ) } } \\ & { \leq R \lambda ^ { r } + R \sqrt { \lambda } \| L _ { M , \lambda } ^ { - 1 / 2 } L _ { K , \lambda } ^ { 1 / 2 } \| \| L _ { K , \lambda } ^ { - 1 / 2 } ( L _ { K } - L _ { M } ) \| ^ { 2 r - 1 } \| L _ { K , \lambda } ^ { - 1 / 2 } ( L _ { K } - L _ { M } ) L _ { K , \lambda } ^ { - 1 / 2 } \| ^ { 2 - 2 r } . } \end{array}
$$

Proof. For the misspecified case when $\ 0 \ < \ r \ \leq \ { \frac { 1 } { 2 } }$ , from Assumption 2, $f _ { \rho } ~ = ~ L _ { K } ^ { r } h _ { \rho }$ for some $h _ { \rho } \in L ^ { 2 } ( \rho _ { X } )$ , and $I - L _ { M } L _ { M , \lambda } ^ { - 1 } = \lambda L _ { M , \lambda } ^ { - 1 }$ , then we have

$$
f _ { \rho } - L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } = \lambda L _ { M , \lambda } ^ { - 1 } L _ { K } ^ { r } h _ { \rho } = \lambda L _ { M , \lambda } ^ { - ( 1 - r ) } \big ( L _ { M , \lambda } ^ { - r } L _ { K , \lambda } ^ { r } \big ) \big ( L _ { K , \lambda } ^ { - r } L _ { K } ^ { r } \big ) h _ { \rho } .
$$

It then follows that

$$
\begin{array} { r l } & { \| f _ { \rho } - L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } \| _ { L ^ { 2 } ( \rho x ) } \leq \lambda \| L _ { M , \lambda } ^ { - ( 1 - r ) } \| \| L _ { M , \lambda } ^ { - r } L _ { K , \lambda } ^ { r } \| \| L _ { K , \lambda } ^ { - r } L _ { K } ^ { r } \| \| h _ { \rho } \| _ { L ^ { 2 } ( \rho x ) } } \\ & { \qquad \leq \lambda \cdot \lambda ^ { r - 1 } \cdot \| L _ { M , \lambda } ^ { - 1 / 2 } L _ { K , \lambda } ^ { 1 / 2 } \| ^ { 2 r } \| L _ { K , \lambda } ^ { - 1 / 2 } L _ { K } ^ { 1 / 2 } \| ^ { 2 r } \cdot R \leq R \lambda ^ { r } \| L _ { M , \lambda } ^ { - 1 / 2 } L _ { K , \lambda } ^ { 1 / 2 } \| ^ { 2 r } . } \end{array}
$$

where we use $\| L _ { M , \lambda } ^ { - ( 1 - r ) } \| \le \lambda ^ { r - 1 }$ for $r \leq 1$ and the Cordes inequality $\| A ^ { \theta } B ^ { \theta } \| \leq \| A B \| ^ { \theta }$ for positive bounded operators $A , B$ and $0 \leq \theta \leq 1$ , applied with $A = L _ { M , \lambda } ^ { - 1 / 2 } , B = L _ { K , \lambda } ^ { 1 / 2 }$ and $\theta = 2 r < 1$ and $A = L _ { K , \lambda } ^ { - 1 } , B = L _ { K } ^ { 1 }$ and $\theta = r \leq 1$ , and $\| L _ { K , \lambda } ^ { - 1 } L _ { K } \| \leq 1$

For the well-specified case when $r \geq { \frac { 1 } { 2 } }$ , note that

$$
f _ { \rho } - L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } = \left( f _ { \rho } - L L _ { K , \lambda } ^ { - 1 } f _ { \rho } \right) + \left( L L _ { K , \lambda } ^ { - 1 } - L _ { M } L _ { M , \lambda } ^ { - 1 } \right) f _ { \rho } ,
$$

we use Rudi and Rosasco (2017, Lemmas 4 and 5) to bound the two terms separately. Consequently, by the triangle inequality, we obtain that

$$
\begin{array} { r l } & { \quad \| f _ { \rho } - L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } \| _ { L ^ { 2 } ( \rho _ { X } ) } \leq \| f _ { \rho } - L L _ { K , \lambda } ^ { - 1 } f _ { \rho } \| _ { L ^ { 2 } ( \rho _ { X } ) } + \| ( L L _ { K , \lambda } ^ { - 1 } - L _ { M } L _ { M , \lambda } ^ { - 1 } ) f _ { \rho } \| _ { L ^ { 2 } ( \rho _ { X } ) } } \\ & { \leq R \lambda ^ { r } + R \sqrt { \lambda } \| L _ { M , \lambda } ^ { - 1 / 2 } L _ { K , \lambda } ^ { 1 / 2 } \| \| L _ { K , \lambda } ^ { - 1 / 2 } ( L _ { K } - L _ { M } ) \| ^ { 2 r - 1 } \| L _ { K , \lambda } ^ { - 1 / 2 } ( L _ { K } - L _ { M } ) L _ { K , \lambda } ^ { - 1 / 2 } \| ^ { 2 - 2 r } . } \end{array}
$$

Combining the bounds for the well-specified and misspecified cases completes the proof.

We introduce the efective dimension induced by kernel $K _ { M }$ . For any $\lambda > 0$ , we define $\mathcal { N } _ { M } ( \lambda )$ as

$$
\mathcal { N } _ { M } ( \lambda ) = \mathrm { T r } ( L _ { M , \lambda } ^ { - 1 } L _ { M } ) = \mathrm { T r } ( C _ { M , \lambda } ^ { - 1 } C _ { M } ) ,
$$

where the second equality uses the cyclic property of the trace and (32),

$$
\mathrm { T r } ( L _ { M , \lambda } ^ { - 1 } L _ { M } ) = \mathrm { T r } ( L _ { M , \lambda } ^ { - 1 } S _ { M } S _ { M } ^ { * } ) = \mathrm { T r } ( S _ { M } ^ { * } L _ { M , \lambda } ^ { - 1 } S _ { M } ) = \mathrm { T r } ( C _ { M , \lambda } ^ { - 1 } S _ { M } ^ { * } S _ { M } ) = \mathrm { T r } ( C _ { M , \lambda } ^ { - 1 } C _ { M } ) .
$$

Lemma 17. Under Assumption 1, $i f 0 < \lambda \leq \| L _ { K } \|$ and $0 < \eta < 1 / 4$ , we set $\begin{array} { r } { t _ { \eta } = \log \frac { 4 \kappa ^ { 2 } } { \lambda \eta } } \end{array}$ , and suppose the number of random features satisfies

$$
M \geq c _ { 0 } \big ( 1 + \mathcal { F } _ { \infty } ( \lambda ) \big ) t _ { \eta } ,\tag{37}
$$

with $c _ { 0 } \geq 3 6$ . Then, with probability at least $1 - 4 \eta$ over the random feature realization $\omega _ { 1 } , \ldots , \omega _ { M }$ the following hold simultaneously:

(a) $\begin{array} { r l } { \big \| L _ { K , \lambda } ^ { - 1 / 2 } ( L _ { K } - L _ { M } ) L _ { K , \lambda } ^ { - 1 / 2 } \big \| } & { \leq \operatorname* { m i n } \big \{ \frac 1 2 , C \sqrt { \frac { \mathcal { F } _ { \infty } ( \lambda ) t _ { \eta } } { M } } \big \} } \end{array}$ , hence $\frac 1 2 L _ { K , \lambda } \preceq L _ { M , \lambda } \preceq \frac 3 2 L _ { K , \lambda }$ and $\| L _ { M , \lambda } ^ { - 1 / 2 } L _ { K , \lambda } ^ { 1 / 2 } \| \le \sqrt { 2 } ;$

(b) $\begin{array} { r } { \left\| \boldsymbol L _ { K , \lambda } ^ { - 1 / 2 } ( \boldsymbol L _ { K } - \boldsymbol L _ { M } ) \right\| \le C \kappa \Big ( \frac { \sqrt { \mathcal F _ { \infty } ( \lambda ) } t _ { \eta } } { M } + \sqrt { \frac { \mathcal N ( \lambda ) t _ { \eta } } { M } } \Big ) \kappa } \end{array}$

(c) $\mathcal { N } _ { M } ( \lambda ) \leq 4 \mathcal { N } ( \lambda )$

Proof. We bound the three events separately and then take the union bound.

Event for (a). Apply Rudi and Rosasco (2017, Proposition 6) with $Q = L _ { K } , U _ { i } = \psi _ { \omega _ { i } } , n = M$ and failure level $\eta ,$ with probability at least $1 - 2 \eta$

$$
\bigl \| L _ { K , \lambda } ^ { - 1 / 2 } ( L _ { K } - L _ { M } ) L _ { K , \lambda } ^ { - 1 / 2 } \bigr \| \leq \frac { 2 \beta \bigl ( 1 + \mathcal { F } _ { \infty } ( \lambda ) \bigr ) } { 3 M } + \sqrt { \frac { 2 \beta \mathcal { F } _ { \infty } ( \lambda ) } { M } } , \qquad \beta = \log \frac { 4 \mathrm { T r } ( L _ { K } ) } { \lambda \eta } .\tag{38}
$$

Since $\mathrm { T r } ( L _ { K } ) \ \leq \ \kappa ^ { 2 }$ and $\lambda \le \| L _ { K } \| \le \kappa ^ { 2 }$ , we have $\begin{array} { r } { \beta \le \log \frac { 4 \kappa ^ { 2 } } { \lambda \eta } = t _ { \eta } . } \end{array}$ and in particular $t _ { \eta } ~ \geq$ log $\begin{array} { r } { \frac { 4 \kappa ^ { 2 } } { \| L _ { K } \| \eta } \geq \log \frac { 4 } { \eta } \geq \log 1 6 > 2 } \end{array}$ for $\eta < 1 / 4$ . Under the budget (37) with $c _ { 0 } \geq 3 6$

$$
\begin{array} { r } { \sqrt { \frac { 2 \beta \mathcal { F } _ { \infty } ( \lambda ) } { M } } \leq \sqrt { \frac { 2 t _ { \eta } \mathcal { F } _ { \infty } ( \lambda ) } { M } } \leq \sqrt { \frac { 2 \mathcal { F } _ { \infty } ( \lambda ) t _ { \eta } } { 3 6 ( 1 + \mathcal { F } _ { \infty } ( \lambda ) ) t _ { \eta } } } \leq \sqrt { \frac { 1 } { 1 8 } } , } \\ { \frac { 2 \beta ( 1 + \mathcal { F } _ { \infty } ( \lambda ) ) } { 3 M } \leq \frac { 2 ( 1 + \mathcal { F } _ { \infty } ( \lambda ) ) t _ { \eta } } { 3 \cdot 3 6 ( 1 + \mathcal { F } _ { \infty } ( \lambda ) ) t _ { \eta } } = \frac { 1 } { 5 4 } . } \end{array}
$$

Moreover, note that $1 + { \mathcal { F } } _ { \infty } ( \lambda ) \leq 3 { \mathcal { F } } _ { \infty } ( \lambda )$ , where we use $\mathcal { F } _ { \infty } ( \lambda ) \ge \mathcal { N } ( \lambda ) \ge 1 / 2$ , since $\lambda \leq \| L _ { K } \|$ implies $\mathcal { N } ( \lambda ) \ge \sigma _ { 1 } / ( \sigma _ { 1 } + \lambda ) \ge 1 / 2$ with $\sigma _ { 1 } = \| L _ { K } \|$ . Hence we have

$$
\frac { 2 \beta ( 1 + \mathcal { F } _ { \infty } ( \lambda ) ) } { 3 M } \leq \frac { 2 \mathcal { F } _ { \infty } ( \lambda ) t _ { \eta } } { M } . \leq \frac { 1 } { 3 } \sqrt { \frac { \mathcal { F } _ { \infty } ( \lambda ) t _ { \eta } } { M } } ,
$$

where the last inequality uses the fact that $\begin{array} { r } { \frac { \mathcal { F } _ { \infty } ( \lambda ) t _ { \eta } } { M } \le \frac { 1 } { 3 6 } } \end{array}$ under the budget (37) with $c _ { 0 } \geq 3 6$ Combining the above we obtain that $\begin{array} { r } { \big \| L _ { K , \lambda } ^ { - 1 / 2 } ( L _ { K } - L _ { M } ) L _ { K , \lambda } ^ { - 1 / 2 } \big \| \leq \operatorname* { m i n } \big \{ \frac { 1 } { 2 } , c _ { 1 } \sqrt { \frac { \mathcal { F } _ { \infty } ( \lambda ) t _ { \eta } } { M } } \big \} } \end{array}$

To derive the order bounds in (a), note that

$$
\begin{array} { r } { L _ { M } = L _ { K , \lambda } ^ { 1 / 2 } ( I - L _ { K , \lambda } ^ { - 1 / 2 } ( L _ { K } - L _ { M } ) L _ { K , \lambda } ^ { - 1 / 2 } ) L _ { K , \lambda } ^ { 1 / 2 } - \lambda I \succeq L _ { K } - \frac 1 2 L _ { K , \lambda } , } \end{array}
$$

where we use $\begin{array} { r } { \big \| L _ { K , \lambda } ^ { - 1 / 2 } ( L _ { K } - L _ { M } ) L _ { K , \lambda } ^ { - 1 / 2 } \big \| \leq \frac { 1 } { 2 } } \end{array}$ . Hence $L _ { M , \lambda } \ \succeq \ \frac { 1 } { 2 } L _ { K , \lambda }$ , and the upper bound is analogous. Finally, taking inverses reverses the order, $L _ { M , \lambda } ^ { - 1 } \preceq 2 L _ { K , \lambda } ^ { - 1 }$ , so

$$
\begin{array} { r } { \mathopen { } \mathclose \bgroup \left\| L _ { M , \lambda } ^ { - 1 / 2 } L _ { K , \lambda } ^ { 1 / 2 } \aftergroup \egroup \right\| ^ { 2 } = \mathopen { } \mathclose \bgroup \left\| L _ { K , \lambda } ^ { 1 / 2 } L _ { M , \lambda } ^ { - 1 } L _ { K , \lambda } ^ { 1 / 2 } \aftergroup \egroup \right\| \leq 2 . } \end{array}
$$

Thus we have $\begin{array} { r } { \| L _ { M , \lambda } ^ { - 1 / 2 } L _ { K , \lambda } ^ { 1 / 2 } \| \le \sqrt { 2 } } \end{array}$

Event for (b). Apply Rudi and Rosasco (2017, Proposition 5) with $U _ { i } = L _ { K , \lambda } ^ { - 1 / 2 } \psi _ { \omega _ { i } } , z _ { i } = \psi _ { \omega _ { i } }$ (so that $\| z _ { i } \| _ { \infty } \leq \kappa ) , Q = L _ { K } , T = L _ { K } , T _ { M } = L _ { M }$ and failure level η, with probability at least $1 - \eta$

$$
\big \| \boldsymbol L _ { K , \lambda } ^ { - 1 / 2 } ( \boldsymbol L _ { K } - \boldsymbol L _ { M } ) \big \| _ { H S } \le \frac { 4 \sqrt { \mathcal F _ { \infty } ( \lambda ) } \kappa \log \frac { 2 } \eta } { M } + \sqrt { \frac { 4 \kappa ^ { 2 } \mathcal N ( \lambda ) \log \frac { 2 } \eta } { M } } .
$$

Since $\| \cdot \| \leq \| \cdot \| _ { H S }$ and log $\begin{array} { r } { \frac { 2 } { \eta } < \log \frac { 4 \kappa ^ { 2 } } { \lambda \eta } = t _ { \eta } \mathrm { ~ w i t h ~ } \lambda \leq \kappa ^ { 2 } } \end{array}$ , both terms are bounded by $4 \kappa \Big ( \frac { \sqrt { \mathcal { F } _ { \infty } ( \lambda ) } t _ { \eta } } { M } +$ $\sqrt { \frac { \mathcal { N } ( \lambda ) t _ { \eta } } { M } } \biggr )$ , which gives (b).

Event for (c). This is Rudi and Rosasco (2017, Proposition 10), which states that under the budget (37) (their condition $M \ge ( 4 + 1 8 { \mathcal F } _ { \infty } ( \lambda ) )$ log $\frac { 1 2 \kappa ^ { 2 } } { \lambda \eta }$ is implied by (37) with $c _ { 0 } \geq 3 6$ and $t _ { \eta } \geq \log 1 6 > \log 3$ , since $1 2 \leq 1 6 \leq 4 \kappa ^ { 2 } / ( \lambda \eta ) )$ , with probability at least $1 - \eta$

$$
\left| \mathcal { N } _ { M } ( \lambda ) - \mathcal { N } ( \lambda ) \right| \leq 1 . 5 5 \mathcal { N } ( \lambda ) ,
$$

hence $\mathcal { N } _ { M } ( \lambda ) \leq 2 . 5 5 \mathcal { N } ( \lambda ) \leq 4 \mathcal { N } ( \lambda )$

Taking the union of the three events, all statements hold simultaneously with probability at least $1 - 4 \eta$ . Thus we conclude the proof. □

Proposition 18. Under Assumptions 1, 2, 3 and 6, $i f 0 < \lambda \leq \| L _ { K } \|$ and $0 < \eta < 1 / 4$ , we set $\begin{array} { r } { t _ { \eta } = \log \frac { 4 \kappa ^ { 2 } } { \lambda \eta } } \end{array}$

(i) (Misspecified case) $\begin{array} { r } { I f 0 < r < \frac { 1 } { 2 } } \end{array}$ and the number of random features satisfies

$$
M \geq C \lambda ^ { - \zeta } t _ { \eta } ,\tag{39}
$$

then with probability at least $1 - 4 \eta$ over the random feature realization $\omega _ { 1 } , \ldots , \omega _ { M }$

$$
\left\| f _ { \rho } - L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } \right\| _ { L ^ { 2 } ( \rho _ { X } ) } \leq 2 ^ { r } R \lambda ^ { r } .
$$

(ii) (Well-specified case) $I f { \overset { 1 } { 2 } } \leq r \leq 1$ and the number of random features satisfies

$$
M \geq C \lambda ^ { - \{ \zeta + ( 1 + \gamma - \zeta ) ( 2 r - 1 ) \} } t _ { \eta } ,\tag{40}
$$

then with probability at least 1 − 4η over the random feature realization $\omega _ { 1 } , \ldots , \omega _ { M }$

$$
\begin{array} { r } { \left\| f _ { \rho } - L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } \right\| _ { L ^ { 2 } ( \rho _ { X } ) } \leq C R \lambda ^ { r } . } \end{array}
$$

Proof. Throughout the proof we work on the event of Lemma 17, which has probability at least 1 − 4η over the random feature realization $\omega _ { 1 } , \ldots , \omega _ { M }$ . On this event, the bounds (a), (b), and (c) from Lemma 17 hold.

For the misspecified case when $\textstyle 0 < r < { \frac { 1 } { 2 } }$ , from Lemmas 16 and 17, when the number of random features M satisfies $M \ge C \lambda ^ { - \zeta } t _ { \eta } \ge c _ { 0 } \bigl ( 1 \bar { + } \mathcal { F } _ { \infty } ( \lambda ) \bigr ) t _ { \eta }$ from Assumption 6 and C is large enough, we have

$$
\left\| f _ { \rho } - L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } \right\| _ { L ^ { 2 } ( \rho _ { X } ) } \leq R \lambda ^ { r } \| L _ { M , \lambda } ^ { - 1 / 2 } L _ { K , \lambda } ^ { 1 / 2 } \| ^ { 2 r } \leq 2 ^ { r } R \lambda ^ { r } .
$$

For the well-specified case when $r \ \geq \ { \frac { 1 } { 2 } } .$ , from Lemmas 16 and 17, when $M \ge C \lambda ^ { - \zeta } t _ { \eta } \ge$ $c _ { 0 } \big ( 1 + \mathcal { F } _ { \infty } ( \lambda ) \big ) t _ { \eta }$ from Assumption 6, we have

$$
\begin{array} { r l } & { \left\| f _ { \rho } - L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } \right\| _ { L ^ { 2 } ( \rho _ { X } ) } } \\ & { \leq R \lambda ^ { r } + R \sqrt { \lambda } \| L _ { M , \lambda } ^ { - 1 / 2 } L _ { K , \lambda } ^ { 1 / 2 } \| \| L _ { K , \lambda } ^ { - 1 / 2 } ( L _ { K } - L _ { M } ) \| ^ { 2 r - 1 } \| L _ { K , \lambda } ^ { - 1 / 2 } ( L _ { K } - L _ { M } ) L _ { K , \lambda } ^ { - 1 / 2 } \| ^ { 2 - 2 r } } \\ & { \leq R \lambda ^ { r } + C R \sqrt { 2 \lambda } \kappa ^ { 2 r - 1 } \left( \frac { \sqrt { \mathcal { F } _ { \infty } ( \lambda ) } t _ { \eta } } { M } + \sqrt { \frac { { \mathcal { N } ( \lambda ) } t _ { \eta } } { M } } \right) ^ { 2 r - 1 } \sqrt { \frac { \mathcal { F } _ { \infty } ( \lambda ) t _ { \eta } } { M } } ^ { 2 - 2 r } } \\ & { \leq R \lambda ^ { r } + C R \sqrt { 2 \lambda } \kappa ^ { 2 r - 1 } \left( \sqrt { \mathcal { F } _ { \infty } ( \lambda ) } \left( \frac { t _ { \eta } } { M } \right) ^ { r } + \sqrt { \frac { { \mathcal { N } ( \lambda ) } ^ { 2 r - 1 } \mathcal { F } _ { \infty } ( \lambda ) ^ { 2 - 2 r } t _ { \eta } } { M } } \right) } \end{array}\tag{41}
$$

where we use the inequality $( x + y ) ^ { 2 r - 1 } \leq x ^ { 2 r - 1 } + y ^ { 2 r - 1 }$ for $x , y \geq 0$ to split the bound into two branches when $0 \leq 2 r - 1 \leq 1$

On the one hand, when the number of random features M satisfies

$$
M \ge C \lambda ^ { - \{ \zeta + ( 1 + \gamma - \zeta ) ( 2 r - 1 ) \} } t _ { \eta } \ge C \mathcal { N } ( \lambda ) ^ { 2 r - 1 } F _ { \infty } ( \lambda ) ^ { 2 - 2 r } \lambda ^ { 1 - 2 r } t _ { \eta } ,\tag{42}
$$

where the last inequality follows from the bounds $\mathcal { N } ( \lambda ) \leq C \lambda ^ { - \gamma }$ and $F _ { \infty } ( \lambda ) \leq F \lambda ^ { - \zeta }$ from Assumptions 3 and 6. Thus we have

$$
C R \sqrt { 2 \lambda } \kappa ^ { 2 r - 1 } \sqrt { \frac { N ( \lambda ) ^ { 2 r - 1 } \mathcal { F } _ { \infty } ( \lambda ) ^ { 2 - 2 r } t _ { \eta } } { M } } \le C R \lambda ^ { r } .\tag{43}
$$

On the other hand, when the number of random features M satisfies

$$
M \geq C \lambda ^ { - \frac { 2 r - 1 + \zeta } { 2 r } } t _ { \eta } \geq C F _ { \infty } ( \lambda ) ^ { 1 / 2 r } \lambda ^ { \frac { 1 - 2 r } { 2 r } } t _ { \eta } ,\tag{44}
$$

where the last inequality follows from the bound $F _ { \infty } ( \lambda ) \leq F \lambda ^ { - \zeta }$ from Assumptions 6. Thus we have

$$
C R \sqrt { 2 \lambda } \kappa ^ { 2 r - 1 } \sqrt { \mathcal { F } _ { \infty } ( \lambda ) } \left( \frac { t _ { \eta } } { M } \right) ^ { r } \leq C R \lambda ^ { r } .\tag{45}
$$

Note that $\begin{array} { r } { \zeta + ( 1 + \gamma - \zeta ) ( 2 r - 1 ) \geq \frac { 2 r - 1 + \zeta } { 2 r } } \end{array}$ for $\zeta \leq 1 , \gamma \geq 0$ , and $r \geq 1 / 2$ , so (42) is the more stringent condition compared to (44). For $\zeta \leq 1$ and $r \geq 1 / 2 , \zeta + ( 1 + \gamma - \zeta ) ( 2 r - 1 ) \geq \zeta$ also implies that (42) is the more stringent condition compared to (39).

Plugging (43) and (45) into (41), we obtain the desired result

$$
\begin{array} { r } { \left\| f _ { \rho } - L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } \right\| _ { L ^ { 2 } ( \rho _ { X } ) } \leq C R \lambda ^ { r } . } \end{array}
$$

Combining the bounds for the well-specified and misspecified cases completes the proof.

## C Empirical RF Fluctuation Error

Lemma 19 (Variational leverage identity). For a fixed random feature realization $\omega _ { 1 } , \ldots , \omega _ { M }$ and ρ<sub>X</sub>-almost every $\mathbf { \boldsymbol { x } } \in \mathcal { X }$ 2

$$
\left\| C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( \pmb { x } ) \right\| _ { 2 } ^ { 2 } = \operatorname* { i n f } _ { \substack { h \in L ^ { 2 } ( \rho _ { X } ) } } \left\{ \| h \| _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } + \frac { 1 } { \lambda } \big \| \phi _ { M } ( \pmb { x } ) - S _ { M } ^ { * } h \big \| _ { 2 } ^ { 2 } \right\} .\tag{46}
$$

The minimum is attained at $h _ { 0 } = L _ { M , \lambda } ^ { - 1 } S _ { M } \phi _ { M } ( { \pmb x } )$

Proof. Fix $\mathbf { \boldsymbol { x } } \in \mathcal { X }$ and set $h _ { 0 } = L _ { M , \lambda } ^ { - 1 } S _ { M } \phi _ { M } ( { \pmb x } ) \in L ^ { 2 } ( \rho _ { X } )$ , which is well defined since $S _ { M } \phi _ { M } ( { \pmb x } )$ belongs to the range of $L _ { M }$ and $L _ { M , \lambda } ^ { - 1 }$ is bounded. We show that $h _ { 0 }$ attains the infimum and that the optimal value equals $\big \| \boldsymbol C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( \pmb x ) \big \| _ { 2 } ^ { 2 }$

Recall that $Z ^ { * } f ( Z Z ^ { * } ) = { \ddot { f } } ( Z ^ { * } Z ) Z ^ { * }$ for any continuous spectral function f and any compact operator Z. Let $f ( \cdot ) = ( \cdot + \lambda I ) ^ { - 1 }$ , then we have $S _ { M } ^ { * } L _ { M , \lambda } ^ { - 1 } = C _ { M , \lambda } ^ { - 1 } S _ { M } ^ { * }$ and ${ \cal L } _ { M , \lambda } ^ { - 1 } S _ { M } = S _ { M } { \cal C } _ { M , \lambda } ^ { - 1 }$ . It follows that

$$
\begin{array} { r l } & { \phi _ { M } ( \pmb { x } ) - S _ { M } ^ { * } h _ { 0 } = \phi _ { M } ( \pmb { x } ) - S _ { M } ^ { * } L _ { M , \lambda } ^ { - 1 } S _ { M } \phi _ { M } ( \pmb { x } ) = \phi _ { M } ( \pmb { x } ) - C _ { M , \lambda } ^ { - 1 } S _ { M } ^ { * } S _ { M } \phi _ { M } ( \pmb { x } ) } \\ & { \qquad = \phi _ { M } ( \pmb { x } ) - C _ { M , \lambda } ^ { - 1 } C _ { M } \phi _ { M } ( \pmb { x } ) = \lambda C _ { M , \lambda } ^ { - 1 } \phi _ { M } ( \pmb { x } ) , } \end{array}\tag{47}
$$

Let $h \in L ^ { 2 } ( \rho _ { X } )$ be arbitrary and write $h = h _ { 0 } + \Delta$ with $\Delta \in L ^ { 2 } ( \rho _ { X } )$ . By (47),

$$
\phi _ { M } ( \pmb { x } ) - S _ { M } ^ { * } h = \lambda C _ { M , \lambda } ^ { - 1 } \phi _ { M } ( \pmb { x } ) - S _ { M } ^ { * } \Delta ,
$$

and therefore

$$
\begin{array} { l } { \displaystyle | | h | | _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } + \frac { 1 } { \lambda } \big | | \phi _ { M } ( \boldsymbol { x } ) - S _ { M } ^ { * } h | \big | _ { 2 } ^ { 2 } = \| h _ { 0 } \| _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } + 2 \langle h _ { 0 } , \boldsymbol { \Delta } \rangle _ { L ^ { 2 } ( \rho _ { X } ) } + \| \boldsymbol { \Delta } \| _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } } \\ { \displaystyle \quad + \frac { 1 } { \lambda } \left( \lambda ^ { 2 } \big \| C _ { M , \lambda } ^ { - 1 } \phi _ { M } ( \boldsymbol { x } ) \big \| _ { 2 } ^ { 2 } - 2 \lambda \big \langle C _ { M , \lambda } ^ { - 1 } \phi _ { M } ( \boldsymbol { x } ) , S _ { M } ^ { * } \boldsymbol { \Delta } \big \rangle _ { 2 } + \big \| S _ { M } ^ { * } \boldsymbol { \Delta } \big \| _ { 2 } ^ { 2 } \right) } \\ { \displaystyle = \| h _ { 0 } \| _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } + \lambda \big \| C _ { M , \lambda } ^ { - 1 } \phi _ { M } ( \boldsymbol { x } ) \big \| _ { 2 } ^ { 2 } + 2 \Big \langle h _ { 0 } - S _ { M } C _ { M , \lambda } ^ { - 1 } \phi _ { M } ( \boldsymbol { x } ) , \boldsymbol { \Delta } \Big \rangle _ { L ^ { 2 } ( \rho _ { X } ) } + \| \boldsymbol { \Delta } \| _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } + \frac { 1 } { \lambda } \big \| S _ { M } ^ { * } \boldsymbol { \Delta } \big \| _ { 2 } ^ { 2 } } \\  \displaystyle = \| h _ { 0 } \| _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } + \lambda \big \| C _ { M , \lambda } ^ { - 1 } \phi _ { M } ( \boldsymbol { x } ) \big \| _ { 2 } ^ { 2 } + \big \| \boldsymbol { \Delta } \big \| \end{array}
$$

The last two terms are nonnegative and vanish for $\Delta = 0$ , so the infimum is attained at $h = h _ { 0 }$

Setting $\Delta = 0$ in (48) and using $S _ { M } ^ { * } S _ { M } = C _ { M }$ , we have

$$
\begin{array} { r l } & { \left\| h _ { 0 } \right\| _ { L ^ { 2 } ( \rho _ { \mathcal X } ) } ^ { 2 } + \lambda \left\| C _ { M , \lambda } ^ { - 1 } \phi _ { M } ( x ) \right\| _ { 2 } ^ { 2 } = \left. C _ { M , \lambda } ^ { - 1 } \phi _ { M } ( x ) , S _ { M } ^ { * } S _ { M } C _ { M , \lambda } ^ { - 1 } \phi _ { M } ( x ) \right. _ { 2 } + \lambda \big \langle C _ { M , \lambda } ^ { - 1 } \phi _ { M } ( x ) , C _ { M , \lambda } ^ { - 1 } \phi _ { M } ( x ) \big \rangle _ { 2 } } \\ & { \quad \quad \quad \quad \quad \quad \quad = \left. C _ { M , \lambda } ^ { - 1 } \phi _ { M } ( x ) , ( C _ { M } + \lambda I ) C _ { M , \lambda } ^ { - 1 } \phi _ { M } ( x ) \right. _ { 2 } } \\ & { \quad \quad \quad \quad \quad \quad \quad = \left. C _ { M , \lambda } ^ { - 1 } \phi _ { M } ( x ) , \phi _ { M } ( x ) \right. _ { 2 } = \left\| C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( x ) \right\| _ { 2 } ^ { 2 } , } \end{array}
$$

which completes the proof.

Lemma 20. Under Assumption $4 , \ i f \ 0 < \lambda \leq 1$ and $\alpha _ { 0 } < \alpha \le 1$ , then for a fixed random feature realization $\omega _ { 1 } , \ldots , \omega _ { M }$ and ρ<sub>X</sub>-almost every $\mathbf { \boldsymbol { x } } \in \mathcal { X }$ 2

$$
\begin{array} { r l } & { \displaystyle \big \| L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } \big \| _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } + \frac { 1 } { \lambda } \mathbb { E } _ { \omega } \Big | \psi _ { \omega } ( \pmb { x } ) - \big \langle L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } , \psi _ { \omega } \big \rangle _ { L ^ { 2 } ( \rho _ { X } ) } \Big | ^ { 2 } } \\ & { \displaystyle = \frac { K ( \pmb { x } , \pmb { x } ) - \big \langle L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } , K _ { \pmb { x } } \big \rangle _ { L ^ { 2 } ( \rho _ { X } ) } } { \lambda } = \sum _ { j } \frac { \mu _ { j } } { \mu _ { j } + \lambda } e _ { j } ( \pmb { x } ) ^ { 2 } \leq M _ { \alpha } \lambda ^ { - \alpha } . } \end{array}\tag{48}
$$

Proof. Expanding the second expected term in the left side of (48), we have

$$
\begin{array} { r l } & { \quad \displaystyle \frac { 1 } { \lambda } \mathbb { E } _ { \omega } \Big \lvert \psi _ { \omega } ( \pmb { x } ) - \langle L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } , \psi _ { \omega } \rangle _ { L ^ { 2 } ( \rho _ { \mathcal { X } } ) } \Big \rvert ^ { 2 } } \\ & { = \displaystyle \frac { 1 } { \lambda } \mathbb { E } _ { \omega } \psi _ { \omega } ( \pmb { x } ) ^ { 2 } - \frac { 2 } { \lambda } \mathbb { E } _ { \omega } \Big [ \psi _ { \omega } ( \pmb { x } ) \big \langle L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } , \psi _ { \omega } \big \rangle _ { L ^ { 2 } ( \rho _ { \mathcal { X } } ) } \Big ] + \frac { 1 } { \lambda } \mathbb { E } _ { \omega } \Big \lvert \big \langle L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } , \psi _ { \omega } \big \rangle _ { L ^ { 2 } ( \rho _ { \mathcal { X } } ) } \Big \rvert ^ { 2 } . } \end{array}\tag{49}
$$

We evaluate the three terms on the right side of (49) separately. For the first term, the integral representation (8) gives

$$
\begin{array} { r } { \mathbb { E } _ { \omega } \psi _ { \omega } ( { \pmb x } ) ^ { 2 } = K ( { \pmb x } , { \pmb x } ) . } \end{array}\tag{50}
$$

For the second term, linearity of the inner product and (8) give

$$
\begin{array} { r } { \mathbb { E } _ { \omega } \Big [ \psi _ { \omega } ( \pmb { x } ) \big \langle L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } , \psi _ { \omega } \big \rangle _ { L ^ { 2 } ( \rho _ { X } ) } \Big ] = \big \langle L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } , \mathbb { E } _ { \omega } [ \psi _ { \omega } ( \pmb { x } ) \psi _ { \omega } ] \big \rangle _ { L ^ { 2 } ( \rho _ { X } ) } = \big \langle L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } , K _ { \pmb { x } } \big \rangle _ { L ^ { 2 } ( \rho _ { X } ) } , } \end{array}\tag{51}
$$

For the third term, we also have

$$
\begin{array} { r l } & { \mathbb { E } _ { \omega } \Big | \big \langle L _ { K , \lambda } ^ { - 1 } K _ { \mathbf { x } } , \psi _ { \omega } \big \rangle _ { L ^ { 2 } ( \rho _ { X } ) } \Big | ^ { 2 } = \big \langle L _ { K , \lambda } ^ { - 1 } K _ { \mathbf { x } } , \mathbb { E } _ { \omega } \big [ \psi _ { \omega } \otimes \psi _ { \omega } \big ] L _ { K , \lambda } ^ { - 1 } K _ { \mathbf { x } } \big \rangle _ { L ^ { 2 } ( \rho _ { X } ) } } \\ & { \qquad = \big \langle L _ { K , \lambda } ^ { - 1 } K _ { \mathbf { x } } , L _ { K } L _ { K , \lambda } ^ { - 1 } K _ { \mathbf { x } } \big \rangle _ { L ^ { 2 } ( \rho _ { X } ) } , } \end{array}\tag{52}
$$

Substituting (50), (51), and (52) into (49), we obtain

$$
\begin{array} { l } { \displaystyle \frac { 1 } { \lambda } \mathbb { E } _ { \omega } \Big | \psi _ { \omega } ( \pmb { x } ) - \big \langle L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } , \psi _ { \omega } \big \rangle _ { L ^ { 2 } ( \rho _ { X } ) } \Big | ^ { 2 } = \frac { 1 } { \lambda } K ( \pmb { x } , \pmb { x } ) - \frac { 2 } { \lambda } \big \langle L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } , K _ { \pmb { x } } \big \rangle _ { L ^ { 2 } ( \rho _ { X } ) } } \\ { + \frac { 1 } { \lambda } \big \langle L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } , L L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } \big \rangle _ { L ^ { 2 } ( \rho _ { X } ) } . } \end{array}\tag{53}
$$

Note that

$$
\begin{array} { r l } & { \quad \big \langle L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } , L _ { K } L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } \big \rangle _ { L ^ { 2 } ( \rho _ { X } ) } + \lambda \big \| L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } \big \| _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } } \\ & { = \big \langle L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } , L _ { K } L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } \big \rangle _ { L ^ { 2 } ( \rho _ { X } ) } + \lambda \big \langle L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } , L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } \big \rangle _ { L ^ { 2 } ( \rho _ { X } ) } } \\ & { = \big \langle L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } , ( L _ { K } + \lambda I ) L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } \big \rangle _ { L ^ { 2 } ( \rho _ { X } ) } } \\ & { = \big \langle L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } , L _ { K , \lambda } L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } \big \rangle _ { L ^ { 2 } ( \rho _ { X } ) } = \big \langle L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } , K _ { \pmb { x } } \big \rangle _ { L ^ { 2 } ( \rho _ { X } ) } . } \end{array}\tag{54}
$$

Combining (53) and (54), we have

$$
\big \| \boldsymbol { L } _ { K , \lambda } ^ { - 1 } K _ { x } \big \| _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } + \frac { 1 } { \lambda } \mathbb { E } _ { \omega } \Big | \psi _ { \omega } ( \boldsymbol { x } ) - \big \langle \boldsymbol { L } _ { K , \lambda } ^ { - 1 } K _ { x } , \psi _ { \omega } \big \rangle _ { L ^ { 2 } ( \rho _ { X } ) } \Big | ^ { 2 } = \frac { K ( \boldsymbol { x } , \boldsymbol { x } ) - \big \langle \boldsymbol { L } _ { K , \lambda } ^ { - 1 } K _ { x } , K _ { x } \big \rangle _ { L ^ { 2 } ( \rho _ { X } ) } } { \lambda } ,
$$

which proves the first equality in (48).

For the spectral equality, the Mercer expansion $\begin{array} { r } { K _ { \pmb { x } } = \sum _ { i } \mu _ { j } e _ { j } ( \pmb { x } ) e _ { j } } \end{array}$ converges in $L ^ { 2 } ( \rho _ { X } )$ for ρ<sub>X</sub>-almost every x, and $L _ { K , \lambda } ^ { - 1 }$ acts diagonally, $L _ { K , \lambda } ^ { - 1 } e _ { j } = ( \mu _ { j } + \lambda ) ^ { - 1 } e _ { j }$ . Hence, by orthonormality of $\{ e _ { j } \} _ { j \ge 1 }$ 2

$$
\big < L _ { K , \lambda } ^ { - 1 } K _ { \pmb x } , K _ { \pmb x } \big > _ { L ^ { 2 } ( \rho _ { X } ) } = \sum _ { j } \frac { \mu _ { j } ^ { 2 } } { \mu _ { j } + \lambda } e _ { j } ( \pmb x ) ^ { 2 } , \qquad K ( \pmb x , \pmb x ) = \sum _ { j } \mu _ { j } e _ { j } ( \pmb x ) ^ { 2 } ,
$$

and subtracting the two gives $\begin{array} { r } { \lambda ^ { - 1 } \sum _ { j } \big ( \mu _ { j } - \mu _ { j } ^ { 2 } / ( \mu _ { j } + \lambda ) \big ) e _ { j } ( { \pmb x } ) ^ { 2 } = \sum _ { j } \frac { \mu _ { j } } { \mu _ { j } + \lambda } e _ { j } ( { \pmb x } ) ^ { 2 } } \end{array}$ . The last inequality follows from Lemma 30. Thus we complete the proof.

Since both terms on the left side of (48) are nonnegative. Hence under Assumption 4, we have

$$
\begin{array} { r } { \big \| L _ { K , \lambda } ^ { - 1 } K _ { \mathbf { x } } \big \| _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } \leq M _ { \alpha } \lambda ^ { - \alpha } , \qquad \mathbb { E } _ { \omega } \Big | \psi _ { \omega } ( \mathbf { x } ) - \big \langle L _ { K , \lambda } ^ { - 1 } K _ { \mathbf { x } } , \psi _ { \omega } \big \rangle _ { L ^ { 2 } ( \rho _ { X } ) } \Big | ^ { 2 } \leq M _ { \alpha } \lambda ^ { 1 - \alpha } . } \end{array}\tag{55}
$$

Under Assumption 1 that $\| \psi _ { \omega } \| _ { L ^ { 2 } ( \rho _ { X } ) } \leq \kappa .$ the triangle inequality gives

$$
\begin{array} { r } { \Big | \psi _ { \omega } ( \pmb { x } ) - \big \langle L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } , \psi _ { \omega } \big \rangle _ { L ^ { 2 } ( \rho _ { X } ) } \Big | ^ { 2 } \leq \kappa ^ { 2 } \big ( 1 + \sqrt { M _ { \alpha } \lambda ^ { - \alpha } } \big ) ^ { 2 } \leq 4 \kappa ^ { 2 } M _ { \alpha } \lambda ^ { - \alpha } , } \end{array}\tag{56}
$$

where the last inequality follows from the fact that $1 + \sqrt { M _ { \alpha } \lambda ^ { - \alpha } } \leq 2 \sqrt { M _ { \alpha } \lambda ^ { - \alpha } }$ for $M _ { \alpha } \lambda ^ { - \alpha } \ge 1$ with $M _ { \alpha } \geq 1$ and $0 < \lambda \leq 1$

Note that (56) already gives a uniform bound for the scalar residual under uniformly sampled features. With the compatibility condition of Assumption 6, we now refine this bound.

Lemma 21. Under Assumptions $1 , \ 4$ and 6, let $0 < \lambda \leq \operatorname* { m i n } \{ 1 , \| L _ { K } \| \}$ , then for any $\alpha _ { 0 } < \alpha \le 1$ a fixed random feature realization $\omega _ { 1 } , \ldots , \omega _ { M }$ and ρ<sub>X</sub>-almost every $\mathbf { \boldsymbol { x } } \in \mathcal { X } ,$

$$
\begin{array} { r } { \left| \psi _ { \omega } ( \pmb { x } ) - \langle L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } , \psi _ { \omega } \rangle _ { L ^ { 2 } ( \rho _ { X } ) } \right| ^ { 2 } \leq \left\{ \begin{array} { l l } { C M _ { \alpha } \lambda ^ { 1 - \zeta - \alpha } , } & { \zeta + \alpha \neq 1 , } \\ { C M _ { \alpha } \left[ 1 + \log \displaystyle \frac { 1 + \| L _ { K } \| } { \lambda } \right] , } & { \zeta + \alpha = 1 . } \end{array} \right. } \end{array}\tag{57}
$$

Proof. For a fixed $\omega ,$ we expand $\psi _ { \omega } = \textstyle \sum _ { j } a _ { j } e _ { j }$ in $L ^ { 2 } ( \rho _ { X } )$ , where $( \mu _ { j } , e _ { j } )$ are the nonzero eigenpairs of $L _ { K }$ and $a _ { j } = \langle \psi _ { \omega } , e _ { j } \rangle _ { L ^ { 2 } ( \rho _ { X } ) }$ . By self-adjointness of $L _ { K }$ and $I - L _ { K } L _ { K , \lambda } ^ { - 1 } = \lambda L _ { K , \lambda } ^ { - 1 }$

$$
\begin{array} { r l } & { \psi _ { \omega } ( \pmb { x } ) - \langle L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } , \psi _ { \omega } \rangle _ { L ^ { 2 } ( \rho _ { X } ) } = \psi _ { \omega } ( \pmb { x } ) - \langle K _ { \pmb { x } } , L _ { K , \lambda } ^ { - 1 } \psi _ { \omega } \rangle _ { L ^ { 2 } ( \rho _ { X } ) } } \\ & { \quad \quad \quad \quad = \psi _ { \omega } ( \pmb { x } ) - \left( L _ { K } L _ { K , \lambda } ^ { - 1 } \psi _ { \omega } \right) ( \pmb { x } ) = \left( \lambda L _ { K , \lambda } ^ { - 1 } \psi _ { \omega } \right) ( \pmb { x } ) , } \end{array}\tag{58}
$$

so the scalar we must bound is the pointwise value of the $L ^ { 2 } ( \rho _ { X } )$ function $\lambda L _ { K , \lambda } ^ { - 1 } \psi _ { \omega }$

We first control the coeficients $a _ { j }$ . For $0 < t \leq 1$ , Assumption 6 gives

$$
\sum _ { \mu _ { j } \leq t } \vert a _ { j } \vert ^ { 2 } \leq 2 t \sum _ { j } \frac { \vert a _ { j } \vert ^ { 2 } } { \mu _ { j } + t } = 2 t \Vert L _ { K , t } ^ { - 1 / 2 } \psi _ { \omega } \Vert _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } \leq 2 F t ^ { 1 - \zeta } .
$$

For $t > 1$ , Assumption 1 gives

$$
\sum _ { \mu _ { j } \le t } | a _ { j } | ^ { 2 } \le \sum _ { j } | a _ { j } | ^ { 2 } = \| \psi _ { \omega } \| _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } \le \kappa ^ { 2 } \le \kappa ^ { 2 } t ^ { 1 - \zeta } ,
$$

where the last inequality uses $\zeta \leq 1$ . Combining the two cases $0 < t \leq 1$ and $t > 1$ gives the following bound:

$$
\sum _ { \mu _ { j } \le t } | a _ { j } | ^ { 2 } \le ( 2 F + \kappa ^ { 2 } ) t ^ { 1 - \zeta } .\tag{59}
$$

We split into three cases according to the sign of $1 - \zeta - \alpha$ Case 1: $\zeta + \alpha < 1$ . By (58) and the spectral decomposition in $L ^ { 2 } ( \rho _ { X } )$ , we have

$$
\big \| L _ { K } ^ { - \alpha / 2 } \big ( \psi _ { \omega } - L _ { K } L _ { K , \lambda } ^ { - 1 } \psi _ { \omega } \big ) \big \| _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } = \sum _ { j } \frac { \lambda ^ { 2 } \mu _ { j } ^ { - \alpha } } { ( \mu _ { j } + \lambda ) ^ { 2 } } | a _ { j } | ^ { 2 } .\tag{60}
$$

We bound the sum in two parts. For $\mu _ { j } \leq \lambda _ { \ast }$ , we write $( 0 , \lambda ]$ as the disjoint union of the intervals $( 2 ^ { - k - 1 } \lambda , 2 ^ { - k } \lambda ] , k \geq 0$ , and bound each part separately. Take $t = 2 ^ { - k } \lambda$ in (59). Since $\zeta + \alpha < 1$ the sum over $k \geq 0$ is

$$
\begin{array} { r l } { \displaystyle \sum _ { u _ { j } \leq \lambda } \frac { \lambda ^ { 2 } \mu _ { j } ^ { - \alpha } } { ( \mu _ { j } + \lambda ) ^ { 2 } } | a _ { j } | ^ { 2 } - \displaystyle \sum _ { k \geq 0 } \sum _ { \lambda ^ { - 1 } \lambda < \mu _ { j } } \sum _ { \lambda ^ { - 2 } } \frac { \lambda ^ { 2 } \mu _ { j } ^ { - \alpha } } { \lambda \left( \mu _ { j } + \lambda \right) ^ { 2 } } | a _ { j } | ^ { 2 } } & \\ { \leq \displaystyle \sum _ { k \geq 0 } ( 2 ^ { - k - 1 } \lambda ) ^ { - \alpha } } & { \| a _ { j } | ^ { 2 } } \\ { \leq ( 2 F + \kappa ^ { 2 } ) \displaystyle \sum _ { k \geq 0 } ( 2 ^ { - k - 1 } \lambda ) ^ { - \alpha } ( 2 ^ { - k } \lambda ) ^ { 1 - k } } \\ { } & { = ( 2 F + \kappa ^ { 2 } ) \lambda ^ { 1 - \zeta - \alpha } 2 ^ { \alpha } \sum _ { k \geq 0 } ^ { \lambda } 2 ^ { - k ( 1 - \zeta - \alpha ) } } \\ { } & { = ( 2 F + \kappa ^ { 2 } ) \lambda ^ { 1 - \zeta - \alpha } 2 ^ { \alpha } \displaystyle \sum _ { k \geq 0 } 2 ^ { - k ( 1 - \zeta - \alpha ) } } \\ { } & { \leq \frac { 2 ^ { \alpha } ( 2 F + \kappa ^ { 2 } ) } { 1 - 2 ^ { - ( 1 - \zeta - \alpha ) } \lambda } 1 ^ { 1 - \zeta - \alpha } . } \end{array}\tag{61}
$$

Here the first inequality uses $\lambda ^ { 2 } \mu _ { j } ^ { - \alpha } / ( \mu _ { j } + \lambda ) ^ { 2 } \leq \mu _ { j } ^ { - \alpha }$ and $\mu _ { j } > 2 ^ { - k - 1 } \lambda$ for the shell, the second inequality uses (59) with $t = 2 ^ { - k } \lambda$ , and the last inequality uses $\begin{array} { r } { \sum _ { k > 0 } 2 ^ { - k ( 1 - \zeta - \alpha ) } } \end{array}$ converges to $1 / ( 1 - 2 ^ { - ( 1 - \zeta - \alpha ) } )$ since $1 - \zeta - \alpha > 0$

For $\mu _ { j } > \lambda$ , we split the sum over the scales $2 ^ { k } \lambda < \mu _ { j } \leq 2 ^ { k + 1 } \lambda , k \geq 0$ . Since $\mu _ { j } \leq \| L _ { K } \|$ , only finitely many scales are nonempty, and the empty scales contribute zero. Take $t = 2 ^ { k + 1 } \lambda$ in (59), the sum over $k \geq 0$ is

$$
\begin{array} { r l } { \displaystyle \sum _ { \mu _ { j } > \lambda } \frac { \lambda ^ { 2 } \mu _ { j } ^ { - \alpha } } { ( \mu _ { j } + \lambda ) ^ { 2 } } | a _ { j } | ^ { 2 } = \displaystyle \sum _ { k \geq 0 } \displaystyle \sum _ { \lambda < \mu _ { j } \leq 2 ^ { k + 1 } \lambda ^ { 2 } } \frac { \lambda ^ { 2 } \mu _ { j } ^ { - \alpha } } { ( \mu _ { j } + \lambda ) ^ { 2 } } | a _ { j } | ^ { 2 } } & \\ { \displaystyle \leq \displaystyle \sum _ { k \geq 0 } \lambda ^ { 2 } ( 2 ^ { k } \lambda ) ^ { - 2 - \alpha } } & { \displaystyle \sum _ { 2 ^ { k } < \mu _ { j } \leq 2 ^ { k + 1 } \lambda } | a _ { j } | ^ { 2 } } \\ { \displaystyle \leq ( 2 F + \kappa ^ { \prime } ) \displaystyle \sum _ { k \geq 0 } \lambda ^ { 2 } ( 2 ^ { k } \lambda ) ^ { - 2 - \alpha } ( 2 ^ { k + 1 } \lambda ) ^ { 1 - \zeta } } & \\ { = ( 2 F + \kappa ^ { \prime } ) \lambda ^ { 1 - \zeta - \alpha } 2 ^ { 1 - \zeta } \displaystyle \sum _ { k \geq 0 } 2 ^ { - k ( 1 + \zeta + \alpha ) } } & \\ { \displaystyle \leq \frac { 2 ^ { 1 - \zeta } ( 2 F + \kappa ^ { \prime } ) } { 1 - 2 ^ { - ( 1 + \zeta + \alpha ) } } \lambda ^ { 1 - \zeta - \alpha } . } \end{array}\tag{62}
$$

Here the first inequality uses $\lambda ^ { 2 } \mu _ { j } ^ { - \alpha } / ( \mu _ { j } + \lambda ) ^ { 2 } \le \lambda ^ { 2 } \mu _ { j } ^ { - 2 - \alpha }$ because $\mu _ { j } + \lambda > \mu _ { j }$ , and $\mu _ { j } > 2 ^ { k } \lambda$ on each scale, the second inequality uses (59) with $t \ = \ 2 ^ { k + 1 } \lambda .$ , and the last inequality uses $\textstyle \sum _ { k > 0 } 2 ^ { - k ( 1 + \zeta + \alpha ) } = 1 / ( 1 - 2 ^ { - ( 1 + \zeta + \alpha ) } )$ , which converges for every $\zeta , \alpha > 0$

Combining (59), (61), and (62), we have

$$
\begin{array} { r } { \big \| L _ { K } ^ { - \alpha / 2 } \big ( \psi _ { \omega } - L _ { K } L _ { K , \lambda } ^ { - 1 } \psi _ { \omega } \big ) \big \| _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } \leq C \lambda ^ { 1 - \zeta - \alpha } , } \end{array}
$$

where C is a constant depending on $F , \kappa , \zeta .$ , and α.

Applying the embedding condition of Assumption 4,

$$
\begin{array} { r } { \left| \psi _ { \omega } ( \pmb { x } ) - \langle L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } , \psi _ { \omega } \rangle _ { L ^ { 2 } ( \rho _ { X } ) } \right| ^ { 2 } \leq M _ { \alpha } \big \| L _ { K } ^ { - \alpha / 2 } \big ( \psi _ { \omega } - L _ { K } L _ { K , \lambda } ^ { - 1 } \psi _ { \omega } \big ) \big \| _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } \leq C M _ { \alpha } \lambda ^ { 1 - \zeta - \alpha } . } \end{array}
$$

Case 2: $\zeta + \alpha > 1$ . By (58) and the triangle inequality,

$$
\begin{array} { r } { \left| \psi _ { \omega } ( \pmb { x } ) - \langle L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } , \psi _ { \omega } \rangle _ { L ^ { 2 } ( \rho _ { X } ) } \right| ^ { 2 } = \left| \psi _ { \omega } ( \pmb { x } ) - L _ { K } L _ { K , \lambda } ^ { - 1 } \psi _ { \omega } ( \pmb { x } ) \right| ^ { 2 } \leq 2 \kappa ^ { 2 } + 2 \big | L _ { K } L _ { K , \lambda } ^ { - 1 } \psi _ { \omega } ( \pmb { x } ) \big | ^ { 2 } , } \end{array}\tag{63}
$$

where the inequality uses $| \psi _ { \omega } ( { \pmb x } ) | \le \kappa$ from Assumption 1. It remains to bound the second part pointwise, and its spectral decomposition is

$$
\left. L _ { K } ^ { - \alpha / 2 } L _ { K } L _ { K , \lambda } ^ { - 1 } \psi _ { \omega } \right. _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } = \sum _ { j } \frac { \mu _ { j } ^ { 2 - \alpha } } { ( \mu _ { j } + \lambda ) ^ { 2 } } | a _ { j } | ^ { 2 } .\tag{64}
$$

For $\mu _ { j } \leq \lambda$ , the multiplier satisfies $\begin{array} { r } { \frac { \mu _ { j } ^ { 2 - \alpha } } { ( \mu _ { j } + \lambda ) ^ { 2 } } \leq \frac { \mu _ { j } ^ { 2 - \alpha } } { \lambda ^ { 2 } } \leq \lambda ^ { - \alpha } } \end{array}$ . By (59) with $t = \lambda$

$$
\sum _ { \mu _ { j } \leq \lambda } \frac { \mu _ { j } ^ { 2 - \alpha } } { ( \mu _ { j } + \lambda ) ^ { 2 } } | a _ { j } | ^ { 2 } \leq \lambda ^ { - \alpha } \sum _ { \mu _ { j } \leq \lambda } | a _ { j } | ^ { 2 } \leq ( 2 F + \kappa ^ { 2 } ) \lambda ^ { 1 - \zeta - \alpha } .\tag{65}
$$

For $\mu _ { j } > \lambda$ , similarly with Case 1, we split the sum over the scales $2 ^ { k } \lambda < \mu _ { j } \leq 2 ^ { k + 1 } \lambda , k \geq 0$ and take $t = 2 ^ { k + 1 } \lambda$ in (59). The sum over $k \geq 0$ is

$$
\begin{array} { r l } { \displaystyle \sum _ { \mu _ { j } > \lambda } \frac { \mu _ { j } ^ { 2 } - \alpha } { ( \mu _ { j } + \lambda ) ^ { 2 } } | a _ { j } | ^ { 2 } = \displaystyle \sum _ { k \geq 0 } \displaystyle \sum _ { ( 2 k ) < \mu _ { j } \leq 2 ^ { k + 1 } \lambda } \frac { \mu _ { j } ^ { 2 } } { ( \mu _ { j } + \lambda ) ^ { 2 } } | a _ { j } | ^ { 2 } } & { } \\ { \displaystyle } & { \leq \displaystyle \sum _ { k \geq 0 } ( 2 ^ { k } \lambda ) ^ { - \alpha } \sum _ { 2 ^ { k } \lambda < \mu _ { j } \leq 2 ^ { k + 1 } \lambda } \sum _ { a _ { j } } | a _ { j } | ^ { 2 } } \\ { \displaystyle } & { \leq ( 2 F + \kappa ^ { 2 } ) \displaystyle \sum _ { k \geq 0 } ( 2 ^ { k } \lambda ) ^ { - \alpha } ( 2 ^ { k + 1 } \lambda ) ^ { 1 } } \\ { \displaystyle } & { = ( 2 F + \kappa ^ { 2 } ) \lambda ^ { 1 - \zeta - \alpha } 2 ^ { 1 - \zeta } \displaystyle \sum _ { k \geq 0 } 2 ^ { k } \lambda ^ { ( \zeta + ( \alpha - 1 ) ) } } \\ { \displaystyle } & { = \frac { 2 ^ { 1 - \zeta } ( 2 F + \kappa ^ { 2 } ) } { 1 - 2 ^ { - ( \zeta + \alpha - 1 ) } \lambda ^ { 1 - \zeta - \alpha } } , } \end{array}\tag{66}
$$

Here the first inequality uses $\begin{array} { r } { \frac { \mu _ { j } ^ { 2 - \alpha } } { ( \mu _ { j } + \lambda ) ^ { 2 } } \leq \frac { \mu _ { j } ^ { 2 - \alpha } } { \mu _ { i } ^ { 2 } } = \mu _ { j } ^ { - \alpha } \leq ( 2 ^ { k } \lambda ) ^ { - \alpha } } \end{array}$ , because $\mu _ { j } + \lambda > \mu _ { j }$ and $\mu _ { j } >$ $2 ^ { k } \lambda$ on each scale, the second inequality uses (59) with $t = 2 ^ { k + 1 } \lambda$ , and the last equality uses $\begin{array} { r } { \sum _ { k > 0 } 2 ^ { - k ( \zeta + \alpha - 1 ) } = 1 / ( 1 - 2 ^ { - ( \zeta + \alpha - 1 ) } ) } \end{array}$ , which converges because $\zeta + \alpha - 1 > 0$

Combining (63)-(66), and the embedding condition of Assumption 4, we have

$$
\left| \psi _ { \omega } ( \pmb { x } ) - \langle L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } , \psi _ { \omega } \rangle _ { L ^ { 2 } ( \rho _ { X } ) } \right| ^ { 2 } \leq 2 \kappa ^ { 2 } + C M _ { \alpha } \lambda ^ { 1 - \zeta - \alpha } .
$$

Since $1 - \zeta - \alpha < 0$ and $\lambda \leq 1$ , we have $\lambda ^ { 1 - \zeta - \alpha } \geq 1$ , so the constant $2 \kappa ^ { 2 }$ can be absorbed into $C M _ { \alpha } \lambda ^ { 1 - \zeta - \alpha }$ after enlarging C. This proves $\left| \psi _ { \omega } ( \pmb { x } ) - \langle L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } , \psi _ { \omega } \rangle _ { L ^ { 2 } ( \rho _ { X } ) } \right| ^ { 2 } \leq C M _ { \alpha } \lambda ^ { 1 - \zeta - \alpha }$ Case 3: $\zeta + \alpha = 1$ . Repeat the argument of Case 2 with $\zeta + \alpha = 1$ . For $\mu _ { j } \leq \lambda$ , the same bound as in Case 2 gives

$$
\sum _ { \mu _ { j } \le \lambda } \frac { \mu _ { j } ^ { 2 - \alpha } } { ( \mu _ { j } + \lambda ) ^ { 2 } } | a _ { j } | ^ { 2 } \le ( 2 F + \kappa ^ { 2 } ) \lambda ^ { 1 - \zeta - \alpha } = 2 F + \kappa ^ { 2 } .\tag{67}
$$

For $\mu _ { j } ~ > ~ \lambda$ , similarly with Case 1, we split the sum over the scales $2 ^ { k } \lambda < \mu _ { j } \leq 2 ^ { k + 1 } \lambda .$ $k \geq 0$ , and take $t = 2 ^ { k + 1 } \lambda$ in (59). Since $\mu _ { j } \leq \| L _ { K } \|$ , the number of nonempty scales is at most $1 + \lfloor \log _ { 2 } ( \| L _ { K } \| / \lambda ) \rfloor$ , so we have

$$
\begin{array} { l l l } { \displaystyle \sum _ { \mu _ { j } > \lambda } \frac { \mu _ { j } ^ { 2 - \alpha } } { ( \mu _ { j } + \lambda ) ^ { 2 } } | a _ { j } | ^ { 2 } = } & { \displaystyle \sum _ { k = 0 } ^ { \lfloor \log _ { 2 } ( \| L _ { k } \| / \lambda ) \rfloor } \sum _ { 2 ^ { k } \lambda < \mu _ { j } \le 2 ^ { k + 1 } \lambda } \frac { \mu _ { j } ^ { 2 - \alpha } } { ( \mu _ { j } + \lambda ) ^ { 2 } } | a _ { j } | ^ { 2 } } \\ { \displaystyle } & { \le } & { \displaystyle \sum _ { k = 0 } ^ { \lfloor \log _ { 2 } ( \| L _ { k } \| / \lambda ) \rfloor } ( 2 ^ { k } \lambda ) ^ { - \alpha } \sum _ { 2 ^ { k } \lambda < \mu _ { j } \le 2 ^ { k + 1 } \lambda } | a _ { j } | ^ { 2 } } \\ { \displaystyle } & { \le ( 2 F + \kappa ^ { 2 } ) \sum _ { k = 0 } ^ { \lfloor \log _ { 2 } ( \| L _ { k } \| / \lambda ) \rfloor } ( 2 ^ { k } \lambda ) ^ { - \alpha } ( 2 ^ { k + 1 } \lambda ) ^ { 1 - \zeta } } \\ { \displaystyle } & { = ( 2 F + \kappa ^ { 2 } ) 2 ^ { 1 - \zeta } \Big [ 1 + \log _ { 2 } ( \| L _ { K } \| / \lambda ) \rfloor \Big ] , } \end{array}\tag{68}
$$

where the last equality uses $\zeta + \alpha = 1$

Similarly with Case 2, combining (63), (64), (67), (68), and the embedding condition of $\mathrm { A s } _ { - }$ sumption 4, we have

$$
\left| \psi _ { \omega } ( \pmb { x } ) - \langle L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } , \psi _ { \omega } \rangle _ { L ^ { 2 } ( \rho _ { X } ) } \right| ^ { 2 } \leq C M _ { \alpha } \left[ 1 + \log \frac { 1 + \| L _ { K } \| } { \lambda } \right] .
$$

Combining the results of all three cases completes the proof.

Lemma 22. Under Assumptions 1, 4 and 6, let $0 < \lambda \leq \operatorname* { m i n } \{ 1 , \| L _ { K } \| \}$ , then for any α<sub>0</sub> $< \alpha \leq 1$ every $t \geq 4$ and $\rho _ { X }$ -almost every $\mathbf { \boldsymbol { x } } \in \mathcal { X }$

$$
\mathbb { P } _ { \Omega } \Big \{ \| C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( \pmb x ) \| _ { 2 } ^ { 2 } > t M _ { \alpha } \lambda ^ { - \alpha } \Big \} \leq \left\{ \begin{array} { l l } { \displaystyle \exp ( - c M \lambda ^ { \zeta } t ) , } & { \zeta + \alpha \neq 1 , } \\ { \displaystyle \exp \Big ( - \frac { c M \lambda ^ { \zeta } t } { 1 + \log ( ( 1 + \| L _ { K } \| ) / \lambda ) } \Big ) , } & { \zeta + \alpha = 1 , } \end{array} \right.\tag{69}
$$

Proof. For $\rho _ { X } .$ -almost every $\mathbf { \boldsymbol { x } } \in \mathcal { X } .$ , we define i.i.d. random variables

$$
\begin{array} { r } { \boldsymbol { U } _ { i } = \Big | \psi _ { \omega _ { i } } ( \pmb { x } ) - \big \langle L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } , \psi _ { \omega _ { i } } \big \rangle _ { L ^ { 2 } ( \rho _ { X } ) } \Big | ^ { 2 } , \quad i = 1 , \ldots , M . } \end{array}
$$

and $b _ { \lambda } = \left\{ { \begin{array} { l l } { C M _ { \alpha } \lambda ^ { 1 - \zeta - \alpha } , } & { \zeta + \alpha \neq 1 , } \\ { C M _ { \alpha } \left[ 1 + \log \frac { 1 + \| L _ { K } \| } { \lambda } \right] , } & { \zeta + \alpha = 1 . } \end{array} } \right.$ From Lemma 21, (55) and (56), we have

$$
0 \leq U _ { i } \leq b _ { \lambda } , \quad \mathbb { E } U _ { i } \leq M _ { \alpha } \lambda ^ { 1 - \alpha } , \quad \mathrm { V a r } ( U _ { i } ) \leq b _ { \lambda } M _ { \alpha } \lambda ^ { 1 - \alpha } ,\tag{70}
$$

where the last inequality follows from the fact that Var $\left( U _ { i } \right) \leq \mathbb { E } U _ { i } ^ { 2 } \leq \operatorname* { s u p } U _ { i } \cdot \mathbb { E } U _ { i }$

By Lemma 19, the definition of the infimum gives, for the particular choice $h = L _ { K , \lambda } ^ { - 1 } K _ { x }$

$$
\left\| C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( \pmb x ) \right\| _ { 2 } ^ { 2 } \leq \left\| L _ { K , \lambda } ^ { - 1 } K _ { \pmb x } \right\| _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } + \frac { 1 } { \lambda } \cdot \frac { 1 } { M } \sum _ { i = 1 } ^ { M } U _ { i } ,\tag{71}
$$

where $\begin{array} { r } { \frac { 1 } { M } \sum _ { i = 1 } ^ { M } U _ { i } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \bigl | \psi _ { \omega _ { i } } ( \pmb { x } ) - \bigl \langle L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } , \psi _ { \omega _ { i } } \bigr \rangle _ { L ^ { 2 } ( \rho _ { X } ) } \bigr | ^ { 2 } } \end{array}$ is exactly the second term of the variational objective evaluated at this h. On the event $\left\{ \left\| C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( \pmb x ) \right\| _ { 2 } ^ { 2 } > t M _ { \alpha } \lambda ^ { - \alpha } \right\}$ , rearranging (71) yields

$$
\frac { 1 } { M } \sum _ { i = 1 } ^ { M } U _ { i } > \lambda \left( t M _ { \alpha } \lambda ^ { - \alpha } - \left. L _ { K , \lambda } ^ { - 1 } K _ { \pm } \right. _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } \right) > ( t - 1 ) M _ { \alpha } \lambda ^ { 1 - \alpha } .
$$

where the second inequality uses $\left. L _ { K , \lambda } ^ { - 1 } K _ { \pmb { x } } \right. _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } \leq M _ { \alpha } \lambda ^ { - \alpha }$ from (55). Note that $\mathbb { E } U _ { i } \le M _ { \alpha } \lambda ^ { 1 - \alpha }$ from (70), we have

$$
\frac { 1 } { M } \sum _ { i = 1 } ^ { M } ( U _ { i } - \mathbb { E } U _ { i } ) > ( t - 1 ) M _ { \alpha } \lambda ^ { 1 - \alpha } - M _ { \alpha } \lambda ^ { 1 - \alpha } = ( t - 2 ) M _ { \alpha } \lambda ^ { 1 - \alpha } .\tag{72}
$$

Now we apply the scalar Bernstein inequality in Lemma 31 with $u = ( t - 2 ) M _ { \alpha } \lambda ^ { 1 - \alpha } , b = b \lambda$ and $v = b _ { \lambda } M _ { \alpha } \lambda ^ { 1 - \alpha }$

$$
\mathbb { P } _ { \Omega } \left\{ \frac { 1 } { M } \sum _ { i = 1 } ^ { M } ( U _ { i } - \mathbb { E } U _ { i } ) > ( t - 2 ) M _ { \alpha } \lambda ^ { 1 - \alpha } \right\} \leq \exp \left( - \frac { 3 M ( t - 2 ) ^ { 2 } M _ { \alpha } \lambda ^ { 1 - \alpha } } { 2 b _ { \lambda } ( t + 1 ) } \right)
$$

$$
\le \exp \left( - \frac { M \lambda ^ { 1 - \alpha } M _ { \alpha } t } { 8 b _ { \lambda } } \right) \le \left\{ \begin{array} { l l } { \exp \left( - c M \lambda ^ { \zeta } t \right) , } & { \zeta + \alpha \neq 1 , } \\ { \exp \left( - \frac { c M \lambda ^ { \zeta } t } { 1 + \log ( ( 1 + \Vert L _ { K } \Vert ) / \lambda ) } \right) , } & { \zeta + \alpha = 1 , } \end{array} \right.
$$

where the first inequality holds for $t \geq 4$ since $( t - 2 ) ^ { 2 } / ( t + 1 ) \geq t / 1 2$ , the last inequality uses $c = 1 / ( 8 C )$ with C being the constant in the definition of $b _ { \lambda }$ . Since the event $\big \{ \big \| C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( \pmb x ) \big \| _ { 2 } ^ { 2 } >$ $t M _ { \alpha } \lambda ^ { - \alpha } \}$ implies the event $\left\{ { \frac { 1 } { M } } \sum _ { i = 1 } ^ { M } ( U _ { i } - \mathbb { E } U _ { i } ) > ( t - 2 ) M _ { \alpha } \lambda ^ { 1 - \alpha } \right\}$ , we have

$$
\begin{array} { r } { \mathbb { P } _ { \Omega } \left\{ \left\| C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( \pmb { x } ) \right\| _ { 2 } ^ { 2 } > t M _ { \alpha } \lambda ^ { - \alpha } \right\} \leq \left\{ \begin{array} { l l } { \displaystyle \exp \left( - c M \lambda ^ { \zeta } t \right) , } & { \zeta + \alpha \neq 1 , } \\ { \displaystyle \exp \left( - \frac { c M \lambda ^ { \zeta } t } { 1 + \log ( ( 1 + \| L _ { K } \| ) / \lambda ) } \right) , } & { \zeta + \alpha = 1 . } \end{array} \right. } \end{array}
$$

Thus we complete the proof.

For a fixed random feature realization $\omega _ { 1 } , \ldots , \omega _ { M }$ , define two sets

$$
\mathcal { A } _ { 1 } = \big \{ \pmb { x } : \big \| C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( \pmb { x } ) \big \| _ { 2 } ^ { 2 } \leq 8 M _ { \alpha } \lambda ^ { - \alpha } \big \} , \qquad \mathcal { A } _ { 2 } = \mathcal { X } \setminus \mathcal { A } _ { 1 } .\tag{73}
$$

Lemma 23. Under Assumptions $1 , \ 4$ and $\delta ,$ let $0 < \lambda \leq \operatorname* { m i n } \{ 1 , \| L _ { K } \| \}$ and $0 < \eta < 1 / 4$ , for any $\alpha _ { 0 } < \alpha \le 1$ , if the number of random features M satisfies

$$
M \geq C _ { 0 } \lambda ^ { - \zeta } \left\{ \begin{array} { l l } { \log \displaystyle \frac { c ( n + M _ { \alpha } \lambda ^ { - \alpha } ) } { \eta ^ { 2 } } , } & { \zeta + \alpha \neq 1 , } \\ { \Big [ 1 + \log \displaystyle \frac { 1 + \| L _ { K } \| } { \lambda } \Big ] \log \displaystyle \frac { c ( n + M _ { \alpha } \lambda ^ { - \alpha } ) } { \eta ^ { 2 } } , } & { \zeta + \alpha = 1 , } \end{array} \right.\tag{74}
$$

then with probability at least $1 - 2 \eta$ over the random features,

$$
\rho _ { X } ( { \mathcal { A } } _ { 2 } ) \leq \frac { \eta } { n } , \qquad \int _ { { \mathcal { A } } _ { 2 } } \bigl \| C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( \pmb { x } ) \bigr \| _ { 2 } ^ { 2 } d \rho _ { X } ( \pmb { x } ) \leq \frac { 1 } { 8 } .\tag{75}
$$

Proof. First, the map $( \omega , \pmb { x } ) \mapsto \left\| C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( \pmb { x } ) \right\| _ { 2 } ^ { 2 }$ is jointly measurable, since each coordinate $\psi _ { \omega _ { i } } ( \pmb { x } )$ is measurable in $( \omega , \pmb { x } )$ by assumption, and $C _ { M , \lambda } ^ { - 1 / 2 }$ depends measurably on $\omega$ only. Hence Tonelli’s theorem applies to every nonnegative function of this map.

For the mass of $\boldsymbol { A } _ { 2 } .$ , Tonelli’s theorem and (69) with $t = 8 ~ \mathrm { g i v }$ ve

$$
\begin{array} { r l } {  { \mathbb { E } _ { \Omega } \bigl [ \rho _ { X } ( \mathcal { A } _ { 2 } ) \bigr ] = \mathbb { E } _ { \Omega } [ \int _ { \mathcal { X } } \mathbf { 1 } \bigl \{ \| C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( \pmb { x } ) \| _ { 2 } ^ { 2 } > 8 M _ { \alpha } \lambda ^ { - \alpha } \bigr \} d \rho _ { X } ( \pmb { x } ) ] } \quad } & { } \\ & { = \int _ { \mathcal { X } } \mathbb { P } _ { \Omega } \Big \{ \| C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( \pmb { x } ) \| _ { 2 } ^ { 2 } > 8 M _ { \alpha } \lambda ^ { - \alpha } \Big \} d \rho _ { X } ( \pmb { x } ) } \\ & { \leq \int _ { \mathcal { X } } \exp ( - \frac { M \lambda ^ { 1 - \alpha } M _ { \alpha } } { b _ { \lambda } } ) d \rho _ { X } ( \pmb { x } ) = \exp ( - \frac { M \lambda ^ { 1 - \alpha } M _ { \alpha } } { b _ { \lambda } } ) , } \end{array}\tag{76}
$$

where $b _ { \lambda }$ is defined in Lemma 22.

Recall the tail identity $\begin{array} { r } { \mathbb { E } \big [ W \mathbf { 1 } \{ W > u \} \big ] = u \mathbb { P } ( W > u ) + \int _ { u } ^ { \infty } \mathbb { P } ( W > v ) d v } \end{array}$ , applied to $W =$ $\big \| C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( \pmb x ) \big \| _ { 2 } ^ { 2 }$ with $u = 8 M _ { \alpha } \lambda ^ { - \alpha }$ , and integrated against $\rho _ { X }$ , together with Tonelli’s theorem gives

$$
\begin{array} { r } { \displaystyle \mathbb { E } _ { \Omega } \left[ \int _ { A _ { 2 } } \left\| C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( \pmb { x } ) \right\| _ { 2 } ^ { 2 } d \rho _ { X } ( \pmb { x } ) \right] = \int _ { \pmb { \chi } } \Big ( 8 M _ { \alpha } \lambda ^ { - \alpha } \mathbb { P } _ { \Omega } \Big \{ \left\| C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( \pmb { x } ) \right\| _ { 2 } ^ { 2 } > 8 M _ { \alpha } \lambda ^ { - \alpha } \Big \} } \\ { + \int _ { 8 M _ { \alpha } \lambda ^ { - \alpha } } \mathbb { P } _ { \Omega } \Big \{ \left\| C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( \pmb { x } ) \right\| _ { 2 } ^ { 2 } > v \Big \} d v \Big ) d \rho _ { X } ( \pmb { x } ) . } \end{array}\tag{77}
$$

For the inner probabilities, fix $v \geq 8 M _ { \alpha } \lambda ^ { - \alpha }$ and apply (69) at threshold $t = v / ( M _ { \alpha } \lambda ^ { - \alpha } ) \geq 8 \geq 4$

$$
\mathbb { P } _ { \Omega } \Big \{ \| C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( \pmb x ) \| _ { 2 } ^ { 2 } > v \Big \} \leq \exp \left( - \frac { M \lambda v } { 8 b _ { \lambda } } \right) .\tag{78}
$$

Substituting (78) and the bound from (76) into (77),

$$
\begin{array} { r l } & { \quad \mathbb { E } _ { \Omega } \left[ \displaystyle \int _ { A _ { 2 } } \lVert C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( { \pmb x } ) \rVert _ { 2 } ^ { 2 } d \rho _ { X } ( { \pmb x } ) \right] } \\ & { \le 8 M _ { \alpha } \lambda ^ { - \alpha } \exp \left( - \frac { M \lambda ^ { 1 - \alpha } M _ { \alpha } } { b _ { \lambda } } \right) + \displaystyle \int _ { 8 M _ { \alpha } \lambda ^ { - \alpha } } ^ { \infty } \exp \left( - \frac { M \lambda v } { 8 b _ { \lambda } } \right) d v } \\ & { = 8 M _ { \alpha } \lambda ^ { - \alpha } \exp \left( - \frac { M \lambda ^ { 1 - \alpha } M _ { \alpha } } { b _ { \lambda } } \right) + \frac { 8 b _ { \lambda } } { M \lambda } \exp \left( - \frac { M \lambda ^ { 1 - \alpha } M _ { \alpha } } { b _ { \lambda } } \right) } \\ & { \le 9 M _ { \alpha } \lambda ^ { - \alpha } \exp \left( - \frac { M \lambda ^ { 1 - \alpha } M _ { \alpha } } { b _ { \lambda } } \right) , } \end{array}\tag{79}
$$

where the last inequality uses the definition of $b _ { \lambda }$ which gives $b _ { \lambda } \ \leq \ C M _ { \alpha } \lambda ^ { 1 - \zeta - \alpha } [ 1 + \log ( ( 1 +$ $\| L _ { K } \| ) / \lambda ) ]$ in both cases, so that $8 b _ { \lambda } / ( M \lambda ) \leq M _ { \alpha } \lambda ^ { - \alpha }$ whenever

$$
M \geq 8 C \lambda ^ { - \zeta } \Big [ 1 + \log \frac { 1 + \| L _ { K } \| } { \lambda } \Big ] .\tag{80}
$$

Now we define two events

$$
G _ { 1 } : = \Big \{ \rho _ { X } ( \mathcal { A } _ { 2 } ) > \frac { \eta } { n } \Big \} , \qquad G _ { 2 } : = \Big \{ \int _ { \mathcal { A } _ { 2 } } \big \| C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( \pmb x ) \big \| ^ { 2 } d \rho _ { X } ( \pmb x ) > \frac { 1 } { 8 } \Big \} .
$$

and apply Markov’s inequality to (76) and (79),

$$
\mathbb { P } _ { \Omega } \left( G _ { 1 } \right) \le \frac { n } { \eta } \exp \left( - \frac { M \lambda ^ { 1 - \alpha } M _ { \alpha } } { b _ { \lambda } } \right) , \qquad \mathbb { P } _ { \Omega } \left( G _ { 2 } \right) \le 7 2 M _ { \alpha } \lambda ^ { - \alpha } \exp \left( - \frac { M \lambda ^ { 1 - \alpha } M _ { \alpha } } { b _ { \lambda } } \right) .
$$

Each of the above bounds is at most $\eta$ if

$$
\frac { M \lambda ^ { 1 - \alpha } M _ { \alpha } } { b _ { \lambda } } \geq \operatorname* { m a x } \left\{ \log \frac { n } { \eta ^ { 2 } } , \log \frac { 7 2 M _ { \alpha } \lambda ^ { - \alpha } } { \eta } \right\} .
$$

Substituting the definition of $b _ { \lambda }$ , this is implied by

$$
M \geq C _ { 0 } \lambda ^ { - \zeta } \left\{ \begin{array} { l l } { \log \displaystyle \frac { c ( n + M _ { \alpha } \lambda ^ { - \alpha } ) } { \eta ^ { 2 } } , } & { \zeta + \alpha \neq 1 , } \\ { \Big [ 1 + \log \displaystyle \frac { 1 + \| L _ { K } \| } { \lambda } \Big ] \log \displaystyle \frac { c ( n + M _ { \alpha } \lambda ^ { - \alpha } ) } { \eta ^ { 2 } } , } & { \zeta + \alpha = 1 , } \end{array} \right.
$$

with $C _ { 0 }$ large enough to cover (80) and $c \ge 7 2$ . This is possible because log $\frac { c ( n + M _ { \alpha } \lambda ^ { - \alpha } ) } { \eta ^ { 2 } } \ge \ge$ $\alpha \log ( 1 / \lambda ) + \log ( c M _ { \alpha } / \eta ^ { 2 } )$ , so in the case $\zeta ~ + ~ \alpha ~ \neq ~ 1$ the condition $C _ { 0 } ~ \ge ~ 8 C / \alpha$ gives $\begin{array} { r l r } { C _ { 0 } \log \frac { c ( n + M _ { \alpha } \lambda ^ { - \alpha } ) } { n ^ { 2 } } } & { { } \ge } & { 8 C [ 1 + \log ( ( 1 + \| L _ { K } \| ) / \lambda ) ] } \end{array}$ for all $0 ~ < ~ \lambda ~ \leq ~ 1$ , provided $c$ is chosen with log $( c \dot { M } _ { \alpha } / \eta ^ { 2 } ) \ge 1 + \log ( 1 + \| L _ { K } \| )$ . On the complement of $G _ { 1 } \cup G _ { 2 }$ , whose probability is at least $1 - \mathbb { P } _ { \Omega } ( G _ { 1 } ) - \mathbb { P } _ { \Omega } ( G _ { 2 } ) \geq 1 - 2 \eta$ , both statements of (75) hold. Thus we complete the proof.

Lemma 24. Under Assumptions $1 , \ 4$ and 6, let $0 < \lambda \leq \operatorname* { m i n } \{ 1 , \| L _ { K } \| \}$ and $0 < \eta < 1 / 4$ . For any $\alpha _ { 0 } < \alpha \le 1$ , suppose that the number of random features M satisfies (74), and the sample size n satisfies

$$
n \geq { \frac { 8 3 2 } { 3 } } M _ { \alpha } \lambda ^ { - \alpha } \log { \frac { 2 M } { \eta } } ,\tag{81}
$$

then with probability at least 1 − 4η over the random features and the samples,

$$
\left\| \widehat { C } _ { M , \lambda } ^ { - 1 / 2 } C _ { M , \lambda } ^ { 1 / 2 } \right\| \leq \sqrt { \frac { 8 } { 5 } } ,\tag{82}
$$

and consequently

$$
\left\| S _ { M } \widehat { C } _ { M , \lambda } ^ { - 1 } C _ { M , \lambda } ^ { 1 / 2 } \right\| \leq \frac { 8 } { 5 } ,\tag{83}
$$

Proof. Work on the event of Lemma 23, which has probability at least $1 - 2 \eta$ over the random features. By the definition of $\mathcal { A } _ { 1 }$ in (73), every $\pmb { x } \in \mathcal { A } _ { 1 }$ satisfies

$$
\left\| C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( \pmb { x } ) \right\| _ { 2 } ^ { 2 } \leq 8 M _ { \alpha } \lambda ^ { - \alpha } ,\tag{84}
$$

and Lemma 23 also gives $\textstyle \rho _ { X } ( { \mathcal { A } } _ { 2 } ) \leq { \frac { \eta } { n } }$ and $\begin{array} { r } { \int _ { \mathcal { A } _ { 2 } } \big \| C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( \pmb { x } ) \big \| _ { 2 } ^ { 2 } d \rho _ { X } ( \pmb { x } ) \leq \frac { 1 } { 8 } . } \end{array}$

For this fixed random features realization, we define random variables

$$
z _ { i } = C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( \pmb { x } _ { i } ) , \qquad Z _ { i } = z _ { i } z _ { i } ^ { \top } \mathbf { 1 } \{ \pmb { x } _ { i } \in A _ { 1 } \} , \qquad i = 1 , \dots , n .
$$

The matrices $Z _ { 1 } , \ldots , Z _ { n }$ are i.i.d., since the samples $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { n }$ are i.i.d. By (84), on $\mathcal { A } _ { 1 }$ , it holds that

$$
0 \preceq Z _ { i } \preceq \| z _ { i } \| _ { 2 } ^ { 2 } I \preceq 8 M _ { \alpha } \lambda ^ { - \alpha } I .\tag{85}
$$

Since $Z _ { i } ^ { 2 } = z _ { i } ( z _ { i } ^ { \top } z _ { i } ) z _ { i } ^ { \top } \mathbf { 1 } ^ { 2 } \{ \mathbf { x } _ { i } \in A _ { 1 } \} = \| z _ { i } \| _ { 2 } ^ { 2 } \mathbf { 1 } \{ \mathbf { x } _ { i } \in A _ { 1 } \} Z _ { i }$ , taking expectations and using (85) gives

$$
\mathbb { E } \left[ Z _ { i } ^ { 2 } \right] \preceq 8 M _ { \alpha } \lambda ^ { - \alpha } \mathbb { E } \left[ Z _ { i } \right] \preceq 8 M _ { \alpha } \lambda ^ { - \alpha } I ,\tag{86}
$$

where the last inequality uses E $\begin{array} { r } { \colon [ Z _ { i } ] \preceq C _ { M , \lambda } ^ { - 1 / 2 } \mathbb { E } \bigl [ \phi _ { M } ( \pmb { x } ) \phi _ { M } ( \pmb { x } ) ^ { \top } \bigr ] C _ { M , \lambda } ^ { - 1 / 2 } \preceq C _ { M , \lambda } ^ { - 1 / 2 } C _ { M } C _ { M , \lambda } ^ { - 1 / 2 } \preceq I } \end{array}$ since $C _ { M } \preceq C _ { M , \lambda }$ . From (85) and (86), we can further get

$$
\Vert Z _ { i } - \mathbb { E } Z _ { i } \Vert \leq 8 M _ { \alpha } \lambda ^ { - \alpha }
$$

since $Z _ { i } - \mathbb { E } Z _ { i } \preceq Z _ { i } \preceq 8 M _ { \alpha } \lambda ^ { - \alpha } I$ and $Z _ { i } - \mathbb { E } Z _ { i } \succeq - \mathbb { E } Z _ { i } \succeq - 8 M _ { \alpha } \lambda ^ { - \alpha } I$ . And

$$
\left\| { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \mathbb { E } { \big [ } ( Z _ { i } - \mathbb { E } Z _ { i } ) ^ { 2 } { \big ] } \right\| \leq \left\| { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \mathbb { E } { \big [ } Z _ { i } ^ { 2 } { \big ] } \right\| \leq 8 M _ { \alpha } \lambda ^ { - \alpha }
$$

Now we apply the matrix Bernstein inequality in Lemma 32 to $Z _ { i } - \mathbb { E } Z _ { i }$ with $B = 8 M _ { \alpha } \lambda ^ { - \alpha }$ and $V = 8 M _ { \alpha } \lambda ^ { - \alpha }$ , and we take $U = 1 / 4$

$$
\begin{array} { r } { \mathbb { P } \Big \{ \Big \| \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \big ( Z _ { i } - \mathbb { E } Z _ { i } \big ) \Big \| > \displaystyle \frac { 1 } { 4 } \Big \} \leq 2 M \exp \left( - \displaystyle \frac { n / 1 6 } { ( 1 6 + \frac { 4 } { 3 } ) M _ { \alpha } \lambda ^ { - \alpha } } \right) } \\ { = 2 M \exp \left( - \displaystyle \frac { 3 n } { 8 3 2 M _ { \alpha } \lambda ^ { - \alpha } } \right) \leq \eta , } \end{array}
$$

where the last inequality uses (81) with $\begin{array} { r } { C = \frac { 8 3 2 } { 3 } } \end{array}$ . Thus with probability at least $1 - \eta ,$ we have

$$
\left. { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } { \big ( } Z _ { i } - \mathbb { E } Z _ { i } { \big ) } \right. \leq { \frac { 1 } { 4 } } .\tag{87}
$$

On the event of Lemma 23, since $\rho _ { X } ( \mathcal { A } _ { 2 } ) \leq \eta / n$ , we have

$$
\mathbb { P } \Big \{ \bigcup _ { i = 1 } ^ { n } \{ { \pmb x } _ { i } \in \mathcal { A } _ { 2 } \} \Big \} \leq n \rho _ { X } ( \mathcal { A } _ { 2 } ) \leq \eta .
$$

Thus the complementary event, where none of the samples lie in $\mathcal { A } _ { 2 }$ , has probability at least $1 - \eta$ On this event, we have $\mathbf { 1 } \{ x _ { i } \in A _ { 1 } \} = 1$ for all $i = 1 , \ldots , n$ , so

$$
\frac { 1 } { n } \sum _ { i = 1 } ^ { n } Z _ { i } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } z _ { i } z _ { i } ^ { \top } = C _ { M , \lambda } ^ { - 1 / 2 } \widehat { C } _ { M } C _ { M , \lambda } ^ { - 1 / 2 } .\tag{88}
$$

On the other hand, since $\mathbf { 1 } \{ x \in { \mathcal { A } } _ { 1 } \} = 1 - \mathbf { 1 } \{ x \in { \mathcal { A } } _ { 2 } \}$ 2

$$
\mathbb { E } \big [ Z _ { i } \big ] = C _ { M , \lambda } ^ { - 1 / 2 } C _ { M } C _ { M , \lambda } ^ { - 1 / 2 } - \mathbb { E } \Big [ z _ { i } z _ { i } ^ { \top } \mathbf { 1 } \{ x _ { i } \in \mathcal { A } _ { 2 } \} \Big ] ,\tag{89}
$$

where we used $\mathbb { E } \big [ \phi _ { M } ( \pmb { x } ) \phi _ { M } ( \pmb { x } ) ^ { \top } \big ] = C _ { M }$ . The subtracted part is a positive matrix, and its norm is at most

$$
\left\| \mathbb { E } \Big [ z _ { i } z _ { i } ^ { \top } \mathbf { 1 } \{ x _ { i } \in \mathcal { A } _ { 2 } \} \Big ] \right\| \leq \mathbb { E } \Big [ \| z _ { i } \| _ { 2 } ^ { 2 } \mathbf { 1 } \{ x _ { i } \in \mathcal { A } _ { 2 } \} \Big ] = \int _ { \mathcal { A } _ { 9 } } \big \| C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( \pmb { x } ) \big \| _ { 2 } ^ { 2 } d \rho _ { X } ( \pmb { x } ) \leq \frac { 1 } { 8 } ,\tag{90}
$$

where the last inequality follows from Lemma 23.

Combining (87)-(90), we obtain by the triangle inequality that

$$
\left\| { C _ { M , \lambda } ^ { - 1 / 2 } } { \left( { \widehat { C } _ { M } - C _ { M } } \right) C _ { M , \lambda } ^ { - 1 / 2 } } \right\| \leq \left\| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } { \left( { Z _ { i } - \mathbb { E } Z _ { i } } \right) } \right\| + \left\| { \mathbb { E } { \left[ { z _ { i } z _ { i } ^ { \top } \mathbf { 1 } \{ x _ { i } \in \mathcal { A } _ { 2 } \} } \right] } } \right\| \leq \frac { 1 } { 4 } + \frac { 1 } { 8 } = \frac { 3 } { 8 }
$$

with probability at least $1 - 2 \eta$ over the samples for the fixed random features. By Lemma E.2 of Blanchard and Kr¨amer (2010), with probability at least $1 - 2 \eta$ over the samples for the fixed random features, we have

$$
\| \widehat { C } _ { M , \lambda } ^ { - 1 / 2 } C _ { M , \lambda } ^ { 1 / 2 } \| \leq \sqrt { \frac { 8 } { 5 } } .\tag{91}
$$

For (83), note that

$$
\big \| S _ { M } \widehat C _ { M , \lambda } ^ { - 1 / 2 } \big \| ^ { 2 } = \big \| \widehat C _ { M , \lambda } ^ { - 1 / 2 } S _ { M } ^ { * } S _ { M } \widehat C _ { M , \lambda } ^ { - 1 / 2 } \big \| = \big \| \widehat C _ { M , \lambda } ^ { - 1 / 2 } C _ { M } \widehat C _ { M , \lambda } ^ { - 1 / 2 } \big \| \leq \big \| \widehat C _ { M , \lambda } ^ { - 1 / 2 } C _ { M , \lambda } \widehat C _ { M , \lambda } ^ { - 1 / 2 } \big \| \leq \frac 8 5 ,
$$

where the first equality uses the relation $\| T \| ^ { 2 } = \| T ^ { * } T \|$ with $T = S _ { M } { \widehat { C } } _ { M , \lambda } ^ { - 1 / 2 }$ , and the last step uses (91). Thus we have

$$
\begin{array} { r } { \| S _ { M } \widehat { C } _ { M , \lambda } ^ { - 1 } C _ { M , \lambda } ^ { 1 / 2 } \| \leq \| S _ { M } \widehat { C } _ { M , \lambda } ^ { - 1 / 2 } \| \| \widehat { C } _ { M , \lambda } ^ { - 1 / 2 } C _ { M , \lambda } ^ { 1 / 2 } \| \leq \sqrt { \frac { 8 } { 5 } } \cdot \sqrt { \frac { 8 } { 5 } } = \frac { 8 } { 5 } . } \end{array}
$$

Finally, we combine the probabilities. The event of Lemma 23 has probability at least $1 - 2 \eta$ over the random features. Conditionally on this event, the bound (87) and the identity (88) each hold with probability at least $1 - \eta$ over the samples, so they hold simultaneously with probability at least $1 - 2 \eta$ . The bounds (91), (82), and (83) are deterministic consequences of (87) and (88), so they hold simultaneously with probability at least $( 1 - 2 \eta ) ( 1 - 2 \eta ) \geq 1 - 4 \eta$ over the features and the samples. Thus we complete the proof. □

Proposition 25. Under Assumptions $1 , \ 2 , \ 3 , \ 4$ and $\delta ,$ if $0 < \lambda \leq$ min $\{ 1 , \| L _ { K } \| \}$ and for any $\alpha _ { 0 } < \alpha \le 1$ , we assume

(i) (Misspecified case) $\begin{array} { r } { I f 0 < r < \frac { 1 } { 2 } } \end{array}$ and the number of random features satisfies

$$
M \geq \left\{ \begin{array} { l l } { C \lambda ^ { - \zeta } \log \frac { c n } { \lambda \eta } , } & { \zeta + \alpha \neq 1 , } \\ { \displaystyle C \lambda ^ { - \zeta } \log \frac { c n } { \lambda \eta } \left[ 1 + \log \frac { 1 + \| L _ { K } \| } { \lambda } \right] , } & { \zeta + \alpha = 1 . } \end{array} \right.\tag{92}
$$

(ii) (Well-specified case) $I f { \overset { 1 } { 2 } } \leq r \leq 1$ and the number of random features satisfies

$$
M \geq \left\{ \begin{array} { l l } { C \lambda ^ { - \left\{ \zeta + ( 1 + \gamma - \zeta ) ( 2 r - 1 ) \right\} } \log \frac { c n } { \lambda \eta } , } & { \zeta + \alpha \neq 1 , } \\ { C \lambda ^ { - \left\{ \zeta + ( 1 + \gamma - \zeta ) ( 2 r - 1 ) \right\} } \log \displaystyle \frac { c n } { \lambda \eta } \left[ 1 + \log \displaystyle \frac { 1 + \left. L _ { K } \right. } { \lambda } \right] , } & { \zeta + \alpha = 1 . } \end{array} \right.\tag{93}
$$

For both cases, we further assume the sample size n satisfies (81). Then for $\eta ~ \in ~ ( 0 , \frac { 1 } { 9 } )$ , with probability at least 1 − 9η over the features and the samples,

$$
\begin{array} { r l } & { \quad \left\| \widehat { C } _ { M , \lambda } ^ { - 1 / 2 } C _ { M , \lambda } ^ { 1 / 2 } \right\| ^ { 2 } \left\| C _ { M , \lambda } ^ { - 1 / 2 } \left( \overline { { S } } _ { M } ^ { * } \big ( f _ { \rho } - L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } \big ) - S _ { M } ^ { * } \big ( f _ { \rho } - L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } \big ) \right) \right\| _ { 2 } } \\ & { \leq C R \left( \sqrt { \frac { M _ { \alpha } \lambda ^ { 2 r - \alpha } } { n \eta } } + \lambda ^ { r } \right) . } \end{array}\tag{94}
$$

Proof. We first focus on $\left\| C _ { M , \lambda } ^ { - 1 / 2 } \left( \overline { { S } } _ { M } ^ { \ast } \big ( f _ { \rho } - L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } \big ) - S _ { M } ^ { \ast } \big ( f _ { \rho } - L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } \big ) \right) \right\| _ { 2 }$ . For this fixed random features realization, we define random variables

$$
z _ { i } = C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( { \bf x } _ { i } ) , \qquad d _ { i } = ( f _ { \rho } - L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } ) ( { \bf x } _ { i } ) , \qquad i = 1 , \ldots , n .
$$

By Definition 14, we have

$$
\begin{array} { l } { { \displaystyle C _ { M , \lambda } ^ { - 1 / 2 } \left( \overline { { S } } _ { M } ^ { * } \left( f _ { \rho } - L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } \right) - S _ { M } ^ { * } \left( f _ { \rho } - L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } \right) \right) = \frac 1 n \sum _ { i = 1 } ^ { n } ( z _ { i } d _ { i } - \mathbb { E } [ z _ { i } d _ { i } ] ) } } \\ { { \displaystyle = \frac 1 n \sum _ { i = 1 } ^ { n } \left( z _ { i } d _ { i } \mathbf { 1 } \{ x _ { i } \in \mathcal { A } _ { 1 } \} - \mathbb { E } \big [ z d \mathbf { 1 } \{ x \in \mathcal { A } _ { 1 } \} \big ] \right) + \frac 1 n \sum _ { i = 1 } ^ { n } \left( z _ { i } d _ { i } \mathbf { 1 } \{ x _ { i } \in \mathcal { A } _ { 2 } \} - \mathbb { E } \big [ z d \mathbf { 1 } \{ x \in \mathcal { A } _ { 2 } \} \big ] \right) , } } \end{array}
$$

where $\mathcal { A } _ { 1 }$ and $\mathcal { A } _ { 2 }$ are defined as in (73). We now proceed to bound the two terms separately. For the first term, we define

$$
\chi _ { i } = { \frac { 1 } { n } } { \Big ( } z _ { i } d _ { i } \mathbf { 1 } \{ x _ { i } \in { \mathcal { A } } _ { 1 } \} - \mathbb { E } { \big [ } z d \mathbf { 1 } \{ x \in { \mathcal { A } } _ { 1 } \} { \big ] } { \Big ) } , \quad i = 1 , \ldots , n .
$$

Then $\chi _ { 1 } , \ldots , \chi _ { n }$ are i.i.d. random vectors in $\mathbb { R } ^ { M }$ with $\mathbb { E } \chi _ { i } = 0$ . Moreover, we have

$$
\sum _ { i = 1 } ^ { n } \mathbb { E } \| \chi _ { i } \| _ { 2 } ^ { 2 } \leq \frac { 1 } { n } \mathbb { E } \bigg [ \| z \| _ { 2 } ^ { 2 } d ^ { 2 } \mathbf { 1 } \{ x \in \mathcal { A } _ { 1 } \} \bigg ] \leq \frac { 8 M _ { \alpha } \lambda ^ { - \alpha } } { n } \mathbb { E } \big [ d ^ { 2 } \mathbf { 1 } \{ x \in \mathcal { A } _ { 1 } \} \big ] \leq \frac { 8 M _ { \alpha } \lambda ^ { - \alpha } } { n } \mathbb { E } \big [ d ^ { 2 } \big ]
$$

where we use $\| z _ { i } \| _ { 2 } ^ { 2 } \leq 8 M _ { \alpha } \lambda ^ { - \alpha }$ on the set $\mathcal { A } _ { 1 }$ . Hence, by Chebyshev’s inequality, for any $t > 0$

$$
\mathbb { P } \bigg ( \bigg \| \sum _ { i = 1 } ^ { n } \chi _ { i } \bigg \| _ { 2 } > t \bigg ) \leq \frac { 1 } { t ^ { 2 } } \mathbb { E } \bigg [ \bigg \| \sum _ { i = 1 } ^ { n } \chi _ { i } \bigg \| _ { 2 } ^ { 2 } \bigg ] = \frac { 1 } { t ^ { 2 } } \sum _ { i = 1 } ^ { n } \mathbb { E } \| \chi _ { i } \| _ { 2 } ^ { 2 } \leq \frac { 8 M _ { \alpha } \lambda ^ { - \alpha } } { n t ^ { 2 } } \| d \| _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } ,
$$

where the second equality uses $\mathbb { E } \langle \chi _ { i } , \chi _ { j } \rangle = \langle \mathbb { E } \chi _ { i } , \mathbb { E } \chi _ { j } \rangle = 0$ for $i \neq j$ , which follows from the conditional independence and the centering $\mathbb { E } \chi _ { i } = 0$ . Setting the right-hand side equal to $\eta _ { \mathrm { : } }$ , we obtain that with probability at least $1 - \eta$

$$
\Bigl \| \sum _ { i = 1 } ^ { n } \chi _ { i } \Bigr \| _ { 2 } \leq \sqrt { \frac { 8 M _ { \alpha } \lambda ^ { - \alpha } \| \pmb { d } \| _ { L ^ { 2 } ( \rho _ { X } ) } ^ { 2 } } { n \eta } } .\tag{95}
$$

For the second term, on the event of Lemma 23, since $\rho _ { X } ( \mathcal { A } _ { 2 } ) \leq \eta / n$ , we have

$$
\mathbb { P } \Big \{ \bigcup _ { i = 1 } ^ { n } \{ { \pmb x } _ { i } \in \mathcal { A } _ { 2 } \} \Big \} \leq n \rho _ { X } ( \mathcal { A } _ { 2 } ) \leq \eta .
$$

Thus the complementary event, where none of the samples lie in $\mathcal { A } _ { 2 }$ , has probability at least $1 - \eta$ On this event, we have $\mathbf { 1 } \{ x _ { i } \in A _ { 1 } \} = 1$ for all $i = 1 , \ldots , n$ , so

$$
{ \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } z _ { i } d _ { i } \mathbf { 1 } \{ { \pmb x } _ { i } \in { \mathcal { A } } _ { 2 } \} = 0\tag{96}
$$

On the other hand, by Cauchy–Schwarz,

$$
\left\| \mathbb { E } \big [ z d \mathbf { 1 } \{ x \in \mathcal { A } _ { 2 } \} \big ] \right\| _ { 2 } \leq \| d \| _ { L ^ { 2 } ( \rho _ { X } ) } \Big ( \int _ { \mathcal { A } _ { 2 } } \| z \| _ { 2 } ^ { 2 } d \rho _ { X } ( x ) \Big ) ^ { 1 / 2 } \leq \frac { \| d \| _ { L ^ { 2 } ( \rho _ { X } ) } } { \sqrt { 8 } } ,\tag{97}
$$

where the last inequality uses the second conclusion of Lemma 23.

On the event of Lemma 17, we have $\| d \| _ { L ^ { 2 } ( \rho _ { X } ) } \leq C R \lambda ^ { r }$ , so Plugging this into (95) and (97), we get

$$
\left. C _ { M , \lambda } ^ { - 1 / 2 } \left( \overline { { S } } _ { M } ^ { * } \left( f _ { \rho } - L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } \right) - S _ { M } ^ { * } \left( f _ { \rho } - L _ { M } L _ { M , \lambda } ^ { - 1 } f _ { \rho } \right) \right) \right. _ { 2 } \leq C R \left( \sqrt { \frac { M _ { \alpha } \lambda ^ { 2 r - \alpha } } { n \eta } } + \lambda ^ { r } \right) .
$$

Combining the above inequality and Lemma 24 that $\begin{array} { r } { \| \widehat { C } _ { M , \lambda } ^ { - 1 / 2 } C _ { M , \lambda } ^ { 1 / 2 } \| \leq \sqrt { \frac { 8 } { 5 } } } \end{array}$ , we can obtain the desired bound for all scenarios. In the following, we verify the conditions on the number of random features, and the probability.

We first check the conditions on the number of random features. We show that the conditions (92) for $\textstyle 0 < r < { \frac { 1 } { 2 } }$ and (93) for $\textstyle { \frac { 1 } { 2 } } \leq r \leq 1$ covers all conditions on M required in the used lemmas. Recall $\begin{array} { r } { t _ { \eta } = \log \frac { 4 \bar { \kappa ^ { 2 } } } { \lambda \eta } } \end{array}$ from Proposition 18. Since $c n \geq 4 \kappa ^ { 2 }$ for c large enough, we have

$$
\log { \frac { c n } { \lambda \eta } } \geq \log { \frac { 4 \kappa ^ { 2 } } { \lambda \eta } } = t _ { \eta } .
$$

Moreover, (81) gives $\begin{array} { r } { n \ge \frac { 8 3 2 } { 3 } M _ { \alpha } \lambda ^ { - \alpha } } \end{array}$ and $\lambda \leq 1$ , hence $\begin{array} { r } { M _ { \alpha } \lambda ^ { 1 - \alpha } = \lambda \cdot M _ { \alpha } \lambda ^ { - \alpha } \leq \frac { 3 n } { 8 3 2 } } \end{array}$ <sup>3n</sup><sub>832</sub> , and with $\eta < 1 / 9$

$$
\log \frac { c ^ { \prime } ( n + M _ { \alpha } \lambda ^ { - \alpha } ) } { \eta ^ { 2 } } = \log \frac { c ^ { \prime } ( n \lambda + M _ { \alpha } \lambda ^ { 1 - \alpha } ) } { \lambda \eta ^ { 2 } } \leq \log \frac { c ^ { \prime } n \bigl ( 1 + \frac { 3 } { 8 3 2 } \bigr ) } { \lambda \eta ^ { 2 } } \leq 2 \log \frac { c n } { \lambda \eta }
$$

for c large enough, where $c ^ { \prime }$ is the constant in (74). Combining this with $\lambda ^ { - \zeta }$ in (92) and $\lambda ^ { - \{ \zeta + ( 1 + \bar { \gamma } - \zeta ) ( 2 r - 1 ) \} } \ge \lambda ^ { - \zeta }$ in (93), the conditions (92) and (93) imply the condition (74) in both cases of Lemma 23. The extra factor $\begin{array} { r } { \left[ 1 + \log \frac { 1 + \| L _ { K } \| } { \lambda } \right] } \end{array}$ when $\zeta + \alpha = 1$ matches that of (74). The same comparison, together with log $\frac { c n } { \lambda \eta } ~ \geq ~ t _ { \eta }$ , shows that (92) and (93) imply the conditions of Proposition 18, i.e. (39) for $\textstyle 0 < r < { \frac { 1 } { 2 } }$ and (40) for $\textstyle { \frac { 1 } { 2 } } \leq r \leq 1$ . Since (74) is also the condition required by Lemma 24, all conditions on M in the used lemmas are satisfied by (92) and (93).

Combining the probabilities, the bound holds with probability at least 1 − 9η over the features and the samples: 4η from the event of Proposition 18, 4η from Lemma 24, whose event already contains the event of Lemma 23 and the event that no sample lies in $\mathbf { \mathcal { A } } _ { 2 } , \mathbf { \mathcal { \eta } }$ from the Chebyshev bound (95). Thus we complete the proof.

## D Noise Error

Proposition 26. Under Assumptions $1 , \ 5 , \ 4$ and $6 , \ i f 0 < \lambda \leq 1$ and for any $\alpha _ { 0 } < \alpha \le 1$ , suppose that the number of random features M satisfies (74) and (37), and the sample size n satisfies (81). Then for $\eta \in ( 0 , 1 / 9 )$ , with probability at least 1 − 9η over the features and the samples,

$$
\begin{array} { r } { \left\| \hat { C } _ { M , \lambda } ^ { - 1 / 2 } C _ { M , \lambda } ^ { 1 / 2 } \right\| ^ { 2 } \left\| C _ { M , \lambda } ^ { - 1 / 2 } \widehat { S } _ { M } ^ { * } \hat { \pmb { \varepsilon } } \right\| _ { 2 } \leq C \left( \frac { B \sqrt { M _ { \alpha } \lambda ^ { - \alpha } } \log \frac { 2 } { \eta } } { n } + \sigma \sqrt { \frac { N ( \lambda ) \log \frac { 2 } { \eta } } { n } } \right) . } \end{array}\tag{98}
$$

Proof. We first focus on $\left\| C _ { M , \lambda } ^ { - 1 / 2 } \widehat { S } _ { M } ^ { * } \widehat { \varepsilon } \right\| _ { 2 }$ . For this fixed random features realization, we define random variables

$$
z _ { i } = C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( { \pmb x } _ { i } ) , \qquad \varepsilon _ { i } = y _ { i } - f _ { \rho } ( { \pmb x } _ { i } ) , \qquad i = 1 , \ldots , n .
$$

By Definition 14, we have

$$
C _ { M , \lambda } ^ { - 1 / 2 } \widehat { S } _ { M } ^ { * } \widehat { \varepsilon } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } z _ { i } \varepsilon _ { i } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } z _ { i } \varepsilon _ { i } \mathbf { 1 } \big \{ \mathbf { x } _ { i } \in \mathcal { A } _ { 1 } \big \} + \frac { 1 } { n } \sum _ { i = 1 } ^ { n } z _ { i } \varepsilon _ { i } \mathbf { 1 } \big \{ \mathbf { x } _ { i } \in \mathcal { A } _ { 2 } \big \} ,\tag{99}
$$

where $\mathcal { A } _ { 1 }$ and $\boldsymbol { A } _ { 2 }$ are defined as in (73). We now proceed to bound the two sums separately.

For the first sum, we define

$$
\xi _ { i } = \varepsilon _ { i } z _ { i } \mathbf { 1 } \{ x _ { i } \in A _ { 1 } \} , \quad i = 1 , \ldots , n .
$$

Then $\xi _ { 1 } , \ldots , \xi _ { n }$ are i.i.d. random vectors in $\mathbb { R } ^ { M }$ with $\mathbb { E } \xi _ { i } = 0$ , since $\mathbb { E } [ \varepsilon | \mathbf { x } ] = 0$ by Assumption 5. For every integer $q \geq 2$ , its q-th moment can be bounded as

$$
\begin{array} { r l r } {  { \mathbb { E } \| \xi _ { i } \| _ { 2 } ^ { q } = \mathbb { E } \Big [ | \varepsilon _ { i } | ^ { q } \| z _ { i } \| _ { 2 } ^ { q } \mathbf { 1 } \{ x _ { i } \in A _ { 1 } \} \Big ] \leq \big ( 8 M _ { \alpha } \lambda ^ { - \alpha } \big ) ^ { q / 2 - 1 } \mathbb { E } \Big [ \| z _ { i } \| _ { 2 } ^ { 2 } \mathbb { E } \big [ | \varepsilon _ { i } | ^ { q } | x _ { i } \big ] \Big ] } } \\ & { } & { \leq \big ( 8 M _ { \alpha } \lambda ^ { - \alpha } \big ) ^ { q / 2 - 1 } \cdot \frac { 1 } { 2 } q ! \sigma ^ { 2 } B ^ { q - 2 } \mathbb { E } \| z _ { i } \| _ { 2 } ^ { 2 } \leq \frac { 1 } { 2 } q ! \big ( B \sqrt { 8 M _ { \alpha } \lambda ^ { - \alpha } } \big ) ^ { q - 2 } \cdot 4 \sigma ^ { 2 } \mathcal { N } ( \lambda ) , } \end{array}
$$

where the first inequality follows from the fact that $\| z _ { i } \| _ { 2 } ^ { 2 } \leq 8 M _ { \alpha } \lambda ^ { - \alpha }$ on the set $\mathcal { A } _ { 1 }$ , the second inequality uses Assumption 5, and the last inequality uses the bound on $\mathbb { E } \Vert z _ { i } \Vert _ { 2 } ^ { 2 }$ derived below

$$
\begin{array} { r l } & { \mathbb { E } \| z \| _ { 2 } ^ { 2 } = \mathbb { E } \big [ \phi _ { M } ( \pmb { x } ) ^ { \top } C _ { M , \lambda } ^ { - 1 } \phi _ { M } ( \pmb { x } ) \big ] = \mathbb { E } \Big [ \mathrm { T r } \big ( C _ { M , \lambda } ^ { - 1 } \phi _ { M } ( \pmb { x } ) \phi _ { M } ( \pmb { x } ) ^ { \top } \big ) \Big ] } \\ & { \qquad = \mathrm { T r } \Big ( C _ { M , \lambda } ^ { - 1 } \mathbb { E } \big [ \phi _ { M } ( \pmb { x } ) \phi _ { M } ( \pmb { x } ) ^ { \top } \big ] \Big ) = \mathrm { T r } \big ( C _ { M , \lambda } ^ { - 1 } C _ { M } \big ) = \mathcal { N } _ { M } ( \lambda ) \leq 4 \mathcal { N } ( \lambda ) , } \end{array}
$$

where the equalities use the cyclicity of the trace and Definition 14, the last inequality is (c) of Lemma 17. Applying Lemma 28 to $\xi _ { 1 } , \ldots , \xi _ { n }$ with $\widetilde { B } = B \sqrt { 8 M _ { \alpha } \lambda ^ { - \alpha } }$ and $\widetilde { \sigma } ^ { 2 } = 4 \sigma ^ { 2 } \mathcal { N } ( \lambda )$ , we obtain with probability at least $1 - \eta$

$$
\left\| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \xi _ { i } \right\| _ { 2 } \leq \frac { 2 B \sqrt { 8 M _ { \alpha } \lambda ^ { - \alpha } } \log \frac { 2 } { \eta } } { n } + \sigma \sqrt { \frac { 8 \mathcal { N } ( \lambda ) \log \frac { 2 } { \eta } } { n } } .\tag{100}
$$

For the second sum, on the event of Lemma 23, since $\rho _ { X } ( \mathcal { A } _ { 2 } ) \leq \eta / n$ , we have

$$
\mathbb { P } \Big \{ \bigcup _ { i = 1 } ^ { n } \{ { \pmb x } _ { i } \in \mathcal { A } _ { 2 } \} \Big \} \leq n \rho _ { X } ( \mathcal { A } _ { 2 } ) \leq \eta .
$$

Thus the complementary event, where none of the samples lie in $\mathcal { A } _ { 2 }$ , has probability at least $1 - \eta$ On this event, we have $\mathbf { 1 } \{ x _ { i } \in \mathcal { A } _ { 2 } \} = 0$ for all $i = 1 , \ldots , n ,$ so

$$
{ \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } z _ { i } \varepsilon _ { i } { \bf 1 } \{ { \bf x } _ { i } \in { \mathcal { A } } _ { 2 } \} = 0 .\tag{101}
$$

Plugging (100) and (101) into the decomposition (99), and applying Lemma 24, we finally get the desired bound

$$
\left\| \widehat { C } _ { M , \lambda } ^ { - 1 / 2 } C _ { M , \lambda } ^ { 1 / 2 } \right\| ^ { 2 } \left\| C _ { M , \lambda } ^ { - 1 / 2 } \widehat { S } _ { M } ^ { * } \widehat { \varepsilon } \right\| _ { 2 } \leq \frac { 8 } { 5 } \Big \| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \xi _ { i } \Big \| _ { 2 } \leq C \left( \frac { B \sqrt { M _ { \alpha } \lambda ^ { - \alpha } } \log \frac { 2 } { \eta } } { n } + \sigma \sqrt { \frac { N ( \lambda ) \log \frac { 2 } { \eta } } { n } } \right) .
$$

The conditions on M are exactly what the used lemmas require: (74) is the condition of Lemma 23 and Lemma 24, and (37) is the condition of Lemma 17. Combining the probabilities, the bound holds with probability at least $1 - 9 \eta$ over the features and the samples: $4 \eta$ from the event of Lemma 17, 4η from Lemma 24, whose event already contains the event of Lemma 23 and the event that no sample lies in $\mathcal { A } _ { 2 }$ , and η from Lemma 28 in (100). Thus we complete the proof. □

## D.1 Noise Error under Heavy-tailed Conditions

Proposition 27. Under Assumptions 1, 4, 7 and 6, $i f 0 < \lambda \leq 1$ and for any $\alpha _ { 0 } < \alpha \le 1$ , suppose that the number of random features M satisfies (74) and (37), and the sample size n satisfies (81). Then for $\eta \in ( 0 , 1 / 9 )$ , with probability at least 1 − 9η over the features and the samples,

$$
\left\| \widehat { C } _ { M , \lambda } ^ { - 1 / 2 } C _ { M , \lambda } ^ { 1 / 2 } \right\| ^ { 2 } \left\| C _ { M , \lambda } ^ { - 1 / 2 } \widehat { S } _ { M } ^ { * } \widehat { \pmb { \varepsilon } } \right\| _ { 2 } \leq C \left( \frac { B ^ { \frac { 1 } { p } } \left( M _ { \alpha } \lambda ^ { - \alpha } \right) ^ { \frac { 1 } { 2 } - \frac { 1 } { p } } \mathscr { N } ( \lambda ) ^ { \frac { 1 } { p } } } { \eta ^ { \frac { 1 } { p } } n ^ { 1 - \frac { 1 } { p } } } + \sigma \sqrt { \frac { \mathscr { N } ( \lambda ) \log \frac { c } { \eta } } { n } } \right) .\tag{102}
$$

Proof. We first focus on $\left\| C _ { M , \lambda } ^ { - 1 / 2 } \widehat { S } _ { M } ^ { * } \widehat { \varepsilon } \right\| _ { 2 }$ . For this fixed random features realization, we define random variables

$$
z _ { i } = C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( { \pmb x } _ { i } ) , \qquad \varepsilon _ { i } = y _ { i } - f _ { \rho } ( { \pmb x } _ { i } ) , \qquad i = 1 , \ldots , n .
$$

By Definition 14, we have

$$
C _ { M , \lambda } ^ { - 1 / 2 } \widehat { S } _ { M } ^ { * } \widehat { \varepsilon } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } z _ { i } \varepsilon _ { i } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } z _ { i } \varepsilon _ { i } \mathbf { 1 } \{ \mathbf { x } _ { i } \in A _ { 1 } \} + \frac { 1 } { n } \sum _ { i = 1 } ^ { n } z _ { i } \varepsilon _ { i } \mathbf { 1 } \{ \mathbf { x } _ { i } \in A _ { 2 } \} ,\tag{103}
$$

where $\mathcal { A } _ { 1 }$ and $\boldsymbol { A } _ { 2 }$ are defined as in (73). We now proceed to bound the two sums separately.

For the first sum, we define

$$
\xi _ { i } = \varepsilon _ { i } z _ { i } \mathbf { 1 } \{ x _ { i } \in A _ { 1 } \} , \quad i = 1 , \ldots , n .
$$

Then $\xi _ { 1 } , \ldots , \xi _ { n }$ are i.i.d. random vectors in $\mathbb { R } ^ { M }$ with $\mathbb { E } \xi _ { i } = 0$ , since $\mathbb { E } [ \varepsilon | { \pmb x } ] = 0$ by Assumption 7. We first bound the second moment and the p-th moment of $\xi _ { i }$ . The same computation as in Proposition 26 gives

$$
\begin{array} { r l } & { \mathbb { E } \| z \| _ { 2 } ^ { 2 } = \mathbb { E } \big [ \phi _ { M } ( \pmb { x } ) ^ { \top } C _ { M , \lambda } ^ { - 1 } \phi _ { M } ( \pmb { x } ) \big ] = \mathbb { E } \Big [ \mathrm { T r } \big ( C _ { M , \lambda } ^ { - 1 } \phi _ { M } ( \pmb { x } ) \phi _ { M } ( \pmb { x } ) ^ { \top } \big ) \Big ] } \\ & { \qquad = \mathrm { T r } \Big ( C _ { M , \lambda } ^ { - 1 } \mathbb { E } \big [ \phi _ { M } ( \pmb { x } ) \phi _ { M } ( \pmb { x } ) ^ { \top } \big ] \Big ) = \mathrm { T r } \big ( C _ { M , \lambda } ^ { - 1 } C _ { M } \big ) = \mathcal { N } _ { M } ( \lambda ) \leq 4 \mathcal { N } ( \lambda ) , } \end{array}
$$

where the equalities use the cyclicity of the trace and Definition 14, and the last inequality is (c) of Lemma 17. Hence, by Assumption 7,

$$
\begin{array} { r } { \mathbb { E } \| \xi _ { i } \| _ { 2 } ^ { 2 } = \mathbb { E } \Big [ \varepsilon ^ { 2 } \| z \| _ { 2 } ^ { 2 } \mathbf { 1 } \{ x \in \mathcal { A } _ { 1 } \} \Big ] \leq \mathbb { E } \Big [ \| z \| _ { 2 } ^ { 2 } \mathbb { E } \big [ \varepsilon ^ { 2 } \big | x \big ] \Big ] \leq \sigma ^ { 2 } \mathbb { E } \| z \| _ { 2 } ^ { 2 } \leq 4 \sigma ^ { 2 } \mathcal { N } ( \lambda ) , } \end{array}
$$

and similarly, since $\| z \| _ { 2 } ^ { p - 2 } \mathbf { 1 } \{ \pmb { x } \in A _ { 1 } \} \leq \big ( 8 M _ { \alpha } \lambda ^ { - \alpha } \big ) ^ { p / 2 - 1 }$

$$
\begin{array} { r l } & { \mathbb { E } \| \xi _ { i } \| _ { 2 } ^ { p } = \mathbb { E } \Big [ | \varepsilon | ^ { p } \| z \| _ { 2 } ^ { p } \mathbf { 1 } \{ x \in \mathcal { A } _ { 1 } \} \Big ] \leq \big ( 8 M _ { \alpha } \lambda ^ { - \alpha } \big ) ^ { p / 2 - 1 } \mathbb { E } \Big [ \| z \| _ { 2 } ^ { 2 } \mathbb { E } \big [ | \varepsilon | ^ { p } \big | x \big ] \Big ] } \\ & { \qquad \leq B \big ( 8 M _ { \alpha } \lambda ^ { - \alpha } \big ) ^ { p / 2 - 1 } \mathbb { E } \| z \| _ { 2 } ^ { 2 } \leq 4 B \big ( 8 M _ { \alpha } \lambda ^ { - \alpha } \big ) ^ { p / 2 - 1 } \mathcal { N } ( \lambda ) . } \end{array}
$$

Applying Lemma 29 to $\xi _ { 1 } , \ldots , \xi _ { n }$ with $\widetilde { \sigma } ^ { 2 } = 4 \sigma ^ { 2 } \mathcal { N } ( \lambda )$ and $\widetilde { B } = 4 B \left( 8 M _ { \alpha } \lambda ^ { - \alpha } \right) ^ { p / 2 - 1 } \mathcal { N } ( \lambda )$ , we obtain with probability at least $1 - \eta _ { ; }$

$$
\left\| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \xi _ { i } \right\| _ { 2 } \leq \left( \frac { 8 c _ { p } B \left( 8 M _ { \alpha } \lambda ^ { - \alpha } \right) ^ { p / 2 - 1 } \mathcal { N } ( \lambda ) } { \eta n ^ { p - 1 } } \right) ^ { 1 / p } + \sigma \sqrt { \frac { 4 \mathcal { N } ( \lambda ) \log ( 2 c _ { p } / \eta ) } { c _ { p } ^ { \prime } n } } ,\tag{104}
$$

where $c _ { p } , c _ { p } ^ { \prime }$ are the constants in Lemma 29.

For the second sum, on the event of Lemma 23, since $\rho _ { X } ( \mathcal { A } _ { 2 } ) \leq \eta / n$ , we have

$$
\mathbb { P } \Big \{ \bigcup _ { i = 1 } ^ { n } \{ { \pmb x } _ { i } \in \mathcal { A } _ { 2 } \} \Big \} \leq n \rho _ { X } ( \mathcal { A } _ { 2 } ) \leq \eta .
$$

Thus the complementary event, where none of the samples lie in $\ A _ { 2 } ,$ has probability at least $1 - \eta$ On this event, we have $\mathbf { 1 } \{ x _ { i } \in \mathcal { A } _ { 2 } \} = 0$ for all $i = 1 , \ldots , n ,$ so

$$
{ \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } z _ { i } \varepsilon _ { i } { \bf 1 } \{ { \bf x } _ { i } \in { \mathcal { A } } _ { 2 } \} = 0 .\tag{105}
$$

Plugging (104) and (105) into the decomposition (103), and applying Lemma 24 that $\big \| \widehat { C } _ { M , \lambda } ^ { - 1 / 2 } \bar { C } _ { M , \lambda } ^ { 1 / 2 } \big \| ^ { 2 } \leq \frac { 8 } { 5 }$ , we obtain

$$
\begin{array} { r } { \left\| \widehat { C } _ { M , \lambda } ^ { - 1 / 2 } C _ { M , \lambda } ^ { 1 / 2 } \right\| ^ { 2 } \left\| C _ { M , \lambda } ^ { - 1 / 2 } \widehat { S } _ { M } ^ { * } \widehat { \varepsilon } \right\| _ { 2 } \leq C \left( \frac { B ^ { \frac { 1 } { p } } \left( M _ { \alpha } \lambda ^ { - \alpha } \right) ^ { \frac { 1 } { 2 } - \frac { 1 } { p } } \mathcal { N } ( \lambda ) ^ { \frac { 1 } { p } } } { \eta ^ { \frac { 1 } { p } } n ^ { 1 - \frac { 1 } { p } } } + \sigma \sqrt { \frac { \mathcal { N } ( \lambda ) \log \frac { c } { \eta } } { n } } \right) , } \end{array}
$$

The conditions on M are exactly what the used lemmas require: (74) is the condition of Lemma 23 and Lemma 24, and (37) is the condition of Lemma 17. Combining the probabilities, the bound holds with probability at least $1 - 9 \eta$ over the features and the samples: 4η from the event of Lemma 17, 4η from Lemma 24, whose event already contains the event of Lemma 23 and the event that no sample lies in $\boldsymbol { A } _ { 2 } .$ , and $\eta$ from Lemma 29 in (104). Thus we complete the proof. □

## E Proof of the Main Results

## E.1 Proof of Theorem 7

Proof. Since $2 r + \gamma > \alpha _ { 0 }$ , there exists some $\alpha > \alpha _ { 0 }$ and $2 r + \gamma > \alpha ,$ such that Propositions 18, 25, and 26 can be applied with this choice of α. Combining these, we obtain bounds on the three terms in the error decomposition (29)

$$
\begin{array} { r } { \left\| \widehat { f } _ { \lambda , M } - f _ { \rho } \right\| _ { L ^ { 2 } ( \rho _ { X } ) } \leq C R \left( \sqrt { \frac { M _ { \alpha } \lambda ^ { 2 r - \alpha } } { n \eta } } + \lambda ^ { r } \right) + C \left( \frac { B \sqrt { M _ { \alpha } \lambda ^ { - \alpha } } \log \frac { 2 } { \eta } } { n } + \sigma \sqrt { \frac { N ( \lambda ) \log \frac { 2 } { \eta } } { n } } \right) } \end{array}
$$

with probability at least $1 - 1 0 \eta = 1 - \delta \in ( 0 , 1 )$ over the features and the samples: 4η from the event of Lemma 17, 4η from Lemma 24, whose event already contains the event of Lemma 23 and the event that no sample lies in $\ A _ { 2 } , \ \eta$ from the Chebyshev bound (95) and η from the Bernstein bound (100). We next simplify the bounds using the specified choice of λ and the conditions in the theorem.

When n satisfies that $\begin{array} { r } { n \lambda ^ { \alpha } \ge \frac { M _ { \alpha } } { \eta \log ^ { 2 } \frac { 2 } { \eta } } } \end{array}$ , we have

$$
C R \sqrt { \frac { M _ { \alpha } \lambda ^ { 2 r - \alpha } } { n \eta } } = C R \lambda ^ { r } \sqrt { \frac { M _ { \alpha } } { n \lambda ^ { \alpha } \eta } } \leq C R \lambda ^ { r } \log \frac { 2 } { \eta } .
$$

When $2 r + \gamma > \alpha$ and $\lambda = n ^ { - \frac { 1 } { 2 r + \gamma } }$ , we have

$$
\frac { C B \sqrt { M _ { \alpha } \lambda ^ { - \alpha } } \log \frac { 2 } { \eta } } { n } \le C B \sqrt { M _ { \alpha } } \lambda ^ { 2 r + \gamma - \frac { \alpha } { 2 } } \log \frac { 2 } { \eta } \le C B \sqrt { M _ { \alpha } } \lambda ^ { r } \log \frac { 2 } { \eta } ,
$$

where the last inequality uses $\lambda ^ { r + \gamma - \alpha / 2 } < \lambda ^ { \gamma / 2 } < 1$ . By Assumption 3 that $\mathcal { N } ( \lambda ) \leq Q ^ { 2 } \lambda ^ { - \gamma }$ and $\lambda = n ^ { - \frac { 1 } { 2 r + \gamma } }$ , we have

$$
C \sigma \sqrt { \frac { \mathcal { N } ( \lambda ) \log \frac { 2 } { \eta } } { n } } \le C \sigma Q \lambda ^ { - \gamma / 2 } \lambda ^ { r + \gamma / 2 } \sqrt { \log \frac { 2 } { \eta } } \le C \sigma Q \lambda ^ { r } \log \frac { 2 } { \eta } ,
$$

where the last inequality uses log $\textstyle { \frac { 2 } { \eta } } > 1$ for $\eta \in ( 0 , 1 / 1 0 )$ .

Combining these three estimates yields

$$
\begin{array} { r } { \big \| \widehat { f } _ { \lambda , M } - f _ { \rho } \big \| _ { L ^ { 2 } ( \rho _ { X } ) } \leq ( C R + C B \sqrt { M _ { \alpha } } + C \sigma Q ) \lambda ^ { r } \log \frac 2 \eta = C _ { 1 } n ^ { - \frac { r } { 2 r + \gamma } } \log \frac { c _ { 1 } } \delta , } \end{array}
$$

where $C _ { 1 } = C R + C B \sqrt { M _ { \alpha } } + C \sigma Q$ and $c _ { 1 } = 2 0$ . In the following, we verify the conditions on the sample size, the number of random features.

We first verify the conditions on the sample size n. With the choice $\lambda = n ^ { - \frac { 1 } { 2 r + \gamma } }$ and $\begin{array} { r } { \eta = \frac { \delta } { 1 0 } } \end{array}$ 2 the conditions on n required above are the following three:

(i) $\lambda \leq \operatorname* { m i n } \{ 1 , \| L _ { K } \| \}$ in Proposition 25, which is equivalent to

$$
n \geq \operatorname* { m a x } \{ 1 , \| L _ { K } \| ^ { - ( 2 r + \gamma ) } \} .
$$

(ii) Condition (81) in Lemma 24, i.e., $\begin{array} { r } { n \ge \frac { 8 3 2 } { 3 } M _ { \alpha } \lambda ^ { - \alpha } \log \frac { 2 M } { \eta } } \end{array}$ . Note that there exist C large enough such that log $\begin{array} { r } { \frac { 2 M } { \eta } \leq C \log \frac { c n } { \delta } } \end{array}$ , so by $\lambda ^ { - \alpha } = n ^ { \frac { \alpha } { 2 r + \gamma } }$ it sufices that

$$
n ^ { \frac { 2 r + \gamma - \alpha } { 2 r + \gamma } } \geq C M _ { \alpha } \log \frac { c n } { \delta } .
$$

Since $\alpha < 2 r + \gamma$ , the exponent $\frac { 2 r + \gamma - \alpha } { 2 r + \gamma }$ is positive, so we can always solve for n in terms of the other parameters, i.e., $n \geq \widetilde n$ for some ne depending only on $M _ { \alpha } , \delta , r , \gamma _ { ; }$ , α and the constants C and c.

(iii) Condition $\begin{array} { r } { n \lambda ^ { \alpha } \ge \frac { M _ { \alpha } } { \eta \log ^ { 2 } \frac { 2 } { \eta } } } \end{array}$ used in the first estimate above, which is equivalent to

$$
n \ge \left( \frac { 1 0 M _ { \alpha } } { \delta \log ^ { 2 } \frac { 2 0 } { \delta } } \right) ^ { \frac { 2 r + \gamma } { 2 r + \gamma - \alpha } } .
$$

Therefore, all the conditions on n are satisfied whenever

$$
n \geq n _ { 0 } = \operatorname* { m a x } \left\{ 1 , \ \left. L _ { K } \right. ^ { - \left( 2 r + \gamma \right) } , \ \widetilde { n } , \ \left( \frac { 1 0 M _ { \alpha } } { \delta \log ^ { 2 } \frac { 2 0 } { \delta } } \right) ^ { \frac { 2 r + \gamma } { 2 r + \gamma - \alpha } } \right\} .
$$

We next verify the conditions on the number of random features M, i.e., (92) for $\textstyle 0 < r < { \frac { 1 } { 2 } }$ and (93) for $\textstyle { \frac { 1 } { 2 } } \leq r \leq 1$ in Proposition 25. With $\lambda = n ^ { - \frac { 1 } { 2 r + \gamma } }$ and $\begin{array} { r } { \eta = \frac { \delta } { 1 0 } . } \end{array}$ , we have

$$
\lambda ^ { - \zeta } = n ^ { \frac { \zeta } { 2 r + \gamma } } , \qquad \lambda ^ { - \{ \zeta + ( 1 + \gamma - \zeta ) ( 2 r - 1 ) \} } = n ^ { \frac { \zeta + ( 1 + \gamma - \zeta ) ( 2 r - 1 ) } { 2 r + \gamma } } ,
$$

and

$$
\log \frac { c n } { \lambda \eta } = \log \left( 1 0 c \delta ^ { - 1 } n ^ { 1 + \frac { 1 } { 2 r + \gamma } } \right) \leq C \log \frac { c n } { \delta } ,
$$

Moreover, when $\zeta + \alpha = 1$

$$
1 + \log \frac { 1 + \| L _ { K } \| } { \lambda } = 1 + \log \bigl ( ( 1 + \| L _ { K } \| ) n ^ { \frac { 1 } { 2 r + \gamma } } \bigr ) .
$$

Therefore, (92) and (93) are satisfied provided that

$$
\begin{array} { r } { M \geq \left\{ \begin{array} { l l } { C _ { 0 } n ^ { \frac { \zeta } { 2 r + \gamma } } \log ^ { \frac { c _ { 0 } n } { \delta } } , } & { 0 < r < \frac { 1 } { 2 } , \zeta + \alpha \neq 1 , } \\ { C _ { 0 } n ^ { \frac { \zeta } { 2 r + \gamma } } \log ^ { \frac { c _ { 0 } n } { \delta } } \Big [ 1 + \log \big ( ( 1 + \| L _ { K } \| ) n ^ { \frac { 1 } { 2 r + \gamma } } \big ) \Big ] , } & { 0 < r < \frac { 1 } { 2 } , \zeta + \alpha = 1 , } \\ { C _ { 0 } n ^ { \frac { \zeta + ( 1 + \gamma - \zeta ) ( 2 r - 1 ) } { 2 r + \gamma } } \log ^ { \frac { c _ { 0 } n } { \delta } } , } & { \frac { 1 } { 2 } \leq r \leq 1 , \zeta + \alpha \neq 1 , } \\ { C _ { 0 } n ^ { \frac { \zeta + ( 1 + \gamma - \zeta ) ( 2 r - 1 ) } { 2 r + \gamma } } \log ^ { \frac { c _ { 0 } n } { \delta } } \Big [ 1 + \log \big ( ( 1 + \| L _ { K } \| ) n ^ { \frac { 1 } { 2 r + \gamma } } \big ) \Big ] , } & { \frac { 1 } { 2 } \leq r \leq 1 , \zeta + \alpha = 1 , } \end{array} \right. } \end{array}
$$

Thus we complete the proof.

The proofs of Theorem 4 and Corollary 8 are the same as that of Theorem 7 with $\zeta = 1$ and $\zeta = \gamma$ . Thus we omit the details.

## E.2 Proof of Theorem 10

Proof. We follow a similar proof strategy as in Theorem 4. Plugging Propositions 18, 25, and 27 into the error decomposition (29), we obtain

$$
\begin{array} { r } { \| \widehat { f } _ { \lambda , M } - f _ { \rho } \| _ { L ^ { 2 } ( \rho _ { X } ) } \leq C R \left( \sqrt { \frac { M _ { \alpha } \lambda ^ { 2 r - \alpha } } { n \eta } } + \lambda ^ { r } \right) + C \left( \frac { B ^ { \frac { 1 } { p } } \left( M _ { \alpha } \lambda ^ { - \alpha } \right) ^ { \frac { 1 } { 2 } - \frac { 1 } { p } } \Lambda ( \lambda ) ^ { \frac { 1 } { p } } } { \eta ^ { \frac { 1 } { p } } n ^ { 1 - \frac { 1 } { p } } } + \sigma \sqrt { \frac { \Lambda ( \lambda ) \log \frac { c } { \eta } } { n } } \right) } \end{array}
$$

with probability at least $1 - 1 0 \eta$

It remains to control the additional term arising from the heavy-tailed noise. By Assumption 3 that $\mathcal { N } ( \lambda ) ~ \le ~ Q ^ { 2 } \lambda ^ { - \gamma }$ and $\textstyle { \frac { 1 } { p } } \ > \ 0$ , we have $\begin{array} { r } { \mathcal { N } ( \lambda ) ^ { \frac { 1 } { p } } \ \leq \ Q ^ { \frac { 2 } { p } } \lambda ^ { - \frac { \gamma } { p } } } \end{array}$ . Moreover, $\begin{array} { r } { \lambda = n ^ { - \frac { 1 } { 2 r + \gamma } } } \end{array}$ gives ${ \frac { 1 } { n ^ { 1 - { \frac { 1 } { p } } } } } = n ^ { - { \frac { p - 1 } { p } } } = \lambda ^ { \frac { ( 2 r + \gamma ) ( p - 1 ) } { p } }$ , ignoring the constant factors, we have

$$
\frac { \left( \lambda ^ { - \alpha } \right) ^ { \frac { 1 } { 2 } - \frac { 1 } { p } } \mathcal { N } ( \lambda ) ^ { \frac { 1 } { p } } } { \eta ^ { \frac { 1 } { p } } n ^ { 1 - \frac { 1 } { p } } } \le \lambda ^ { - \frac { \alpha ( p - 2 ) } { 2 p } - \frac { \gamma } { p } + \frac { ( 2 r + \gamma ) ( p - 1 ) } { p } } \eta ^ { - \frac { 1 } { p } } = \lambda ^ { r + \frac { ( p - 2 ) ( 2 r + 2 \gamma - \alpha ) } { 2 p } } \eta ^ { - \frac { 1 } { p } } \le \lambda ^ { r } ,
$$

where the equality uses

$$
- \frac { \alpha ( p - 2 ) } { 2 p } - \frac { \gamma } { p } + \frac { ( 2 r + \gamma ) ( p - 1 ) } { p } - r = \frac { ( p - 2 ) ( 2 r + 2 \gamma - \alpha ) } { 2 p } ,
$$

and the last inequality holds provided that

$$
\lambda ^ { \frac { ( p - 2 ) ( 2 r + 2 \gamma - \alpha ) } { 2 p } } \eta ^ { - \frac { 1 } { p } } \leq 1 .
$$

Since $p > 2$ and $2 r + 2 \gamma - \alpha > 2 r + \gamma - \alpha > 0$ , the exponent $\frac { ( p - 2 ) ( 2 r + 2 \gamma - \alpha ) } { 2 p }$ is positive. Substituting $\lambda = n ^ { - \frac { 1 } { 2 r + \gamma } }$ , the above condition is equivalent to

$$
n \geq \left( \eta ^ { - { \frac { 1 } { p } } } \right) ^ { { \frac { 2 p ( 2 r + \gamma ) } { ( p - 2 ) ( 2 r + 2 \gamma - \alpha ) } } } = \left( { \frac { 1 0 } { \delta } } \right) ^ { { \frac { 2 ( 2 r + \gamma ) } { ( p - 2 ) ( 2 r + 2 \gamma - \alpha ) } } } .\tag{106}
$$

Under the following condition on the sample size $n , \mathrm { i . e . }$ 2

$$
n \geq n _ { 1 } = \operatorname* { m a x } \left\{ 1 , \ \| L _ { K } \| ^ { - ( 2 r + \gamma ) } , \ \widetilde { n } , \ \left( \frac { 1 0 M _ { \alpha } } { \delta \log ^ { 2 } \frac { 2 0 } { \delta } } \right) ^ { \frac { 2 r + \gamma } { 2 r + \gamma - \alpha } } , \left( \frac { 1 0 } { \delta } \right) ^ { \frac { 2 ( 2 r + \gamma ) } { ( p - 2 ) ( 2 r + 2 \gamma - \alpha ) } } \right\} \geq n _ { 0 } .
$$

we can directly control the additional term arising from the heavy-tailed noise. Other terms in the error decomposition can be handled using techniques for light-tailed noise. Thus we can obtain the same error bound as in the light-tailed case with some diferent constants.

The proofs of Corollaries 12 and 11 are the same as that of Theorem 10 with $\zeta = 1$ and $\zeta = \gamma$ Thus we omit the details.

## F Concentration Inequalities in Hilbert Spaces

In this section, we give two concentration inequalities for sums of independent random variables taking values in a separable Hilbert space. They serve to bound the noise term $\left\| C _ { M , \lambda } ^ { - 1 / 2 } \widehat { S } _ { M } ^ { * } \widehat { \varepsilon } \right\| _ { 2 }$ under two diferent moment conditions on the noise. The first one, a Bennett inequality, applies to light-tailed noise and gives a bound with logarithmic dependence on the confidence level η. The second one, a Fuk-Nagaev inequality, only requires finite moments of order $p \geq 3$ and is therefore suited to heavy-tailed noise. Its bound contains an additional polynomial term in η.

## F.1 Concentration Inequalities in Hilbert Spaces under Light-tailed Conditions

The following Bennett inequality for vector-valued random variables is from Smale and Zhou (2007), and it can be derived from Theorem 3.4 in Pinelis (1994) and the elementary inequality $t \log ( 1 + t ) \geq$ $2 t - 2 \log ( 1 + t )$ for any $t > 0$

Lemma 28 (Bennett inequality for sum of random vectors). Let $z _ { 1 } , \dots , z _ { n }$ be a sequence of i.i.d. random variables on a separable Hilbert space H. If there exist $\tilde { \sigma } , \bar { B } \geq 0$ such that

$$
\mathbb { E } \| z - \mathbb { E } [ z ] \| _ { \mathcal { H } } ^ { p } \leq \frac { 1 } { 2 } p ! \widetilde { B } ^ { p - 2 } \widetilde { \sigma } ^ { 2 } , \quad \forall p \geq 2 ,\tag{107}
$$

then for any $\eta \in ( 0 , 1 )$ , there holds

$$
\left\| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } z _ { i } - \mathbb { E } z _ { i } \right\| _ { \mathcal { H } } \leq \frac { 2 \widetilde { B } \log ( 2 / \eta ) } { n } + \sqrt { \frac { 2 \widetilde { \sigma } ^ { 2 } \log ( 2 / \eta ) } { n } } ,
$$

with probability at least $1 - \eta$ . In particular, (107) holds if

$$
\begin{array} { r } { \| z \| _ { \mathcal { H } } \leq \widetilde { B } / 2 , \ a . s . \quad a n d \quad \mathbb { E } \| z \| _ { \mathcal { H } } ^ { 2 } \leq \widetilde { \sigma } ^ { 2 } . } \end{array}
$$

This lemma relies on a Bernstein-type moment condition (107) on the random variables. A suficient condition for (107) is that the variables are almost surely bounded or have light-tailed distributions $( \mathrm { e . g . } ,$ , Gaussian or sub-Gaussian). In our analysis it is applied to the whitened noise vectors $C _ { M , \lambda } ^ { - 1 / 2 } \phi _ { M } ( \pmb { x } _ { i } ) \hat { \varepsilon }$ and yields the high-probability bound for the noise term in Proposition 26, which is one of the three error terms in the decomposition of Proposition 15.

## F.2 Concentration Inequalities in Hilbert Spaces under Heavy-tailed Conditions

The following Fuk-Nagaev inequality for vector-valued random variables is from Yurinsky (2006) and Mollenhauer et al. (2026), which is a generalization of the classical real-valued Fuk-Nagaev inequality (Fuk, 1973; Nagaev, 1979) to separable Hilbert spaces.

Lemma 29 (Fuk-Nagaev inequality for sum of random vectors). Let $z _ { 1 } , \ldots , z _ { n }$ be a sequence of i.i.d. random variables on a separable Hilbert space $\mathcal { H } _ { \mathrm { : } }$ such that

$$
\mathbb { E } [ z ] = 0 , \quad \mathbb { E } [ \| z \| _ { \mathcal { H } } ^ { 2 } ] \leq \widetilde { \sigma } ^ { 2 } , \quad a n d \quad \mathbb { E } [ \| z \| _ { \mathcal { H } } ^ { p } ] \leq \widetilde { B } ,
$$

for some constants $\widetilde { \sigma } , \widetilde { B } > 0$ and $p \in \mathbb { N } , p \geq 3$ . Then there exist some constants $c _ { p } , c _ { p } ^ { \prime } > 0$ depending on p such that for any $t > 0$ , we have

$$
\mathbb { P } \left[ \left. \frac { 1 } { n } \sum _ { i = 1 } ^ { n } z _ { i } \right. _ { \mathcal { H } } > t \right] \leq c _ { p } \left( \frac { \widetilde { B } } { t ^ { p } n ^ { p - 1 } } + \exp \left( - c _ { p } ^ { \prime } \frac { t ^ { 2 } n } { \widetilde { \sigma } ^ { 2 } } \right) \right) .\tag{108}
$$

Equivalently, for any $\eta \in ( 0 , 1 )$ , we have

$$
\left\| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } z _ { i } \right\| _ { \mathcal { H } } \leq \left( \frac { 2 c _ { p } \widetilde { B } } { \eta n ^ { p - 1 } } \right) ^ { 1 / p } + \widetilde { \sigma } \sqrt { \frac { \log ( 2 c _ { p } / \eta ) } { c _ { p } ^ { \prime } n } }\tag{109}
$$

with probability at least $1 - \eta$

Compared to Lemma 28, this result assumes only finite p-th moments for some $p \geq 3$ , and thus can be applied to heavy-tailed random variables. It consists of two terms on the right-hand side of (109): a polynomial term (in n and $\eta )$ , capturing heavy tails, and a sub-Gaussian term with logarithmic dependence on $1 / \eta$ , analogous to the Bennett-type bound in Lemma 28. We use this inequality in place of Lemma 28 to control the same noise term under the heavy-tail assumption, which gives the noise bound in Proposition 27. Furthermore, once the sample size n is suficiently large, the polynomial term is dominated by the exponential terms, so the estimator attains the same rate as in the light-tailed case. For completeness, we give the proof of this result, following Yurinsky (2006) and Mollenhauer et al. (2026). Throughout, $C _ { p } , c _ { p } , c _ { p } ^ { \prime }$ denote positive constants depending only on the fixed integer $p \geq 3$ , their values may change from line to line.

Proof. We first prove that, for every $x > 0$

$$
\mathbb { P } \Big \{ \Big \| \sum _ { i = 1 } ^ { n } z _ { i } \Big \| _ { \mathcal { H } } > x \Big \} \leq C _ { p } \left( \frac { n \widetilde { B } } { x ^ { p } } + \exp { \Big [ - c _ { p } ^ { \prime } \frac { x ^ { 2 } } { n \widetilde { \sigma } ^ { 2 } } \Big ] } \right) .\tag{110}
$$

A centered bounded-sum inequality. Let $Y _ { 1 } , \dots , Y _ { n }$ be independent, centered, H-valued random variables satisfying $\| Y _ { i } \| _ { \mathcal { H } } \leq L$ almost surely, $\begin{array} { r } { \sum _ { i } \mathbb { E } \| Y _ { i } \| _ { \mathcal { H } } ^ { 2 } \leq n \widetilde { \sigma } ^ { 2 } } \end{array}$ , and $\begin{array} { r } { \sum _ { i } \mathbb { E } \| Y _ { i } \| _ { \mathcal { H } } ^ { p } \leq 2 ^ { p } n \widetilde B } \end{array}$ . The centered Hilbert-space cosh inequality (Pinelis, 1994; see also Mollenhauer et al., 2025, Appendix D.3.1) gives, for $h > 0$ 2

$$
\mathbb { E } \cosh \left( h \Big \Vert \sum _ { i } Y _ { i } \Big \Vert _ { \mathcal { H } } \right) \leq \prod _ { i } \mathbb { E } \Big [ e ^ { h \Vert Y _ { i } \Vert _ { \mathcal { H } } } - h \Vert Y _ { i } \Vert _ { \mathcal { H } } \Big ] \leq \exp \left( \sum _ { i } \mathbb { E } \Big [ e ^ { h \Vert Y _ { i } \Vert _ { \mathcal { H } } } - 1 - h \Vert Y _ { i } \Vert _ { \mathcal { H } } \Big ] \right) .
$$

For integers $2 \leq k \leq p$ , moment interpolation together with H¨older’s inequality on the disjoint union of the probability spaces yields

$$
\sum _ { i } \mathbb { E } \| Y _ { i } \| _ { \mathcal { H } } ^ { k } \leq \left( n \widetilde { \sigma } ^ { 2 } \right) ^ { \frac { p - k } { p - 2 } } \left( 2 ^ { p } n \widetilde { B } \right) ^ { \frac { k - 2 } { p - 2 } } .
$$

Consequently, by the weighted arithmetic–geometric mean inequality, the contribution of the kth power-series term is bounded by $C _ { p } ( h ^ { 2 } n \widetilde { \sigma } ^ { 2 } + h ^ { p } n \widetilde { B } )$ . For $k > p ,$ , the bound $\| Y _ { i } \| _ { \mathcal { H } } \leq L$ gives $\begin{array} { r } { \sum _ { i } \mathbb { E } \| Y _ { i } \| _ { \mathcal { H } } ^ { k } \leq L ^ { k - p } 2 ^ { p } n \widetilde { B } } \end{array}$ . Summing the exponential series shows

$$
\mathbb { E } \operatorname { c o s h } \left( h \Big \| \sum _ { i } Y _ { i } \Big \| _ { \mathcal { H } } \right) \leq \exp \left( C _ { p } h ^ { 2 } n \widetilde { \sigma } ^ { 2 } + C _ { p } h ^ { p } n \widetilde { B } e ^ { h L } \right) ,\tag{111}
$$

where the constants $C _ { p }$ are independent of $n , \ { \widetilde \sigma } ^ { 2 } , \ { \widetilde B } .$ , and L. Since $e ^ { u } \leq 2$ cosh(u) for $u \geq 0$ Markov’s inequality applied to (111) gives

$$
\mathbb { P } \left\{ \left\| \sum _ { i } Y _ { i } \right\| _ { \mathcal { H } } > u \right\} \le 2 \exp \Big ( - h u + C _ { p } h ^ { 2 } n \widetilde { \sigma } ^ { 2 } + C _ { p } h ^ { p } n \widetilde { B } e ^ { h L } \Big ) \qquad ( u , h > 0 ) .\tag{112}
$$

Truncation and correction of the mean. Fix $x > 0$ and set $\Delta = n \widetilde { B } / x ^ { p }$ . Choose $K \geq 1 6$ depending only on $p$ and to be specified below, and truncate at $L = x / K$

$$
\widetilde { z } _ { i } = z _ { i } \mathbf { 1 } _ { \{ \| z _ { i } \| _ { \mathcal { H } } \leq L \} } , \qquad Y _ { i } = \widetilde { z } _ { i } - \mathbb { E } \widetilde { z } _ { i } .
$$

In contrast to $\widetilde { z } _ { i } .$ , the variables $Y _ { i }$ are centered. By Markov’s inequality and a union bound,

$$
\begin{array} { r } { \mathbb { P } \Big \{ \displaystyle \sum _ { i = 1 } ^ { n } z _ { i } \neq \sum _ { i } \widetilde { z } _ { i } \Big \} \leq \sum _ { i } \frac { \mathbb { E } \| z _ { i } \| _ { \mathcal { H } } ^ { p } } { L ^ { p } } \leq K ^ { p } \Delta . } \end{array}\tag{113}
$$

Since $\mathbb { E } z _ { i } = 0$

$$
\Big \| \sum _ { i } \mathbb { E } \widetilde { z _ { i } } \Big \| _ { \mathcal { H } } \leq \sum _ { i } \mathbb { E } \Big [ \| z _ { i } \| _ { \mathcal { H } } \mathbf { 1 } _ { \{ \| z _ { i } \| _ { \mathcal { H } } > L \} } \Big ] \leq \frac { n \widetilde { B } } { L ^ { p - 1 } } = K ^ { p - 1 } \Delta x .\tag{114}
$$

Furthermore, $\| Y _ { i } \| _ { \mathcal { H } } \leq 2 L$

$$
\sum _ { i } \mathbb { E } \| Y _ { i } \| _ { \mathcal { H } } ^ { 2 } \leq n \widetilde { \sigma } ^ { 2 } , \qquad \sum _ { i } \mathbb { E } \| Y _ { i } \| _ { \mathcal { H } } ^ { p } \leq 2 ^ { p } n \widetilde { B } .
$$

The first inequality follows from the Hilbert-space variance identity $\mathbb { E } \Vert Y - \mathbb { E } Y \Vert _ { \mathcal { H } } ^ { 2 } = \mathbb { E } \Vert Y \Vert _ { \mathcal { H } } ^ { 2 } -$ $\| \mathbb { E } Y \| _ { \mathcal { H } } ^ { 2 } ;$ the second follows from Jensen’s inequality and $( a + b ) ^ { p } \leq 2 ^ { p - 1 } ( a ^ { p } + b ^ { p } )$

Choose a fixed $\Delta _ { 0 } \leq$ min $\{ e ^ { - 1 } , ( 4 K ^ { p - 1 } ) ^ { - 1 } \}$ . If $\Delta \ge \Delta _ { 0 }$ , then (110) follows from the trivial bound $\begin{array} { r } { \mathbb { P } \big \{ \big \| \sum _ { i } z _ { i } \big \| _ { \mathcal { H } } > x \big \} \leq 1 \stackrel { . . } { \leq } \Delta / \Delta _ { 0 } } \end{array}$ after enlarging $C _ { p } .$ . We may thus assume $\Delta < \Delta _ { 0 }$ . Then the right-hand side of $( 1 1 4 )$ is at most $x / 4$ . Combining (113) with (112), applied at $u = 3 x / 4$ and with the truncation level $2 L ,$ , yields, for every $h > 0$ 2

$$
\mathbb { P } \Big \{ \Big \lVert \sum _ { i = 1 } ^ { n } z _ { i } \Big \rVert _ { \mathcal { H } } > x \Big \} \le K ^ { p } \Delta + 2 \exp \left( - \frac { 3 } { 4 } h x + C _ { p } h ^ { 2 } n \widetilde { \sigma } ^ { 2 } + C _ { p } h ^ { p } n \widetilde { B } e ^ { 2 h x / K } \right) .\tag{115}
$$

Optimization in two deviation regimes. Write $H = x ^ { 2 } / ( n \widetilde { \sigma } ^ { 2 } )$ and $G = \log ( 1 / \Delta ) \geq 1$ . We take $D \geq 6 4 C _ { p }$ and then choose $K$ (which may be enlarged) so that $K \geq 1 6$ and $K \geq 4 a D$ , where $a = \operatorname* { m i n } \{ 1 , 3 / ( 1 6 C _ { p } ) \}$

If $H \leq D G ,$ choose $h = a x / ( n \widetilde { \sigma } ^ { 2 } )$ . Then

$$
\begin{array} { r } { - \frac 3 4 h x + C _ { p } h ^ { 2 } n \widetilde { \sigma } ^ { 2 } \le - \frac { 9 } { 1 6 } a H , \qquad C _ { p } h ^ { p } n \widetilde { B } e ^ { 2 h x / K } \le C _ { p } a ^ { p } D ^ { p } G ^ { p } e ^ { - G / 2 } \le C _ { p } ^ { \prime } , } \end{array}
$$

where the last bound is finite since $\mathrm { s u p } _ { G \geq 1 } G ^ { p } e ^ { - G / 2 } < \infty$ . Thus (115) gives

$$
\mathbb { P } \Big \{ \Big \| \sum _ { i = 1 } ^ { n } z _ { i } \Big \| _ { \mathcal { H } } > x \Big \} \leq K ^ { p } \Delta + C _ { p } ^ { \prime \prime } e ^ { - c _ { p } ^ { \prime } H } .
$$

If $H > D G$ , choose $h = 4 G / x$ . Then

$$
- \frac { 3 } { 4 } h x = - 3 G , \qquad C _ { p } h ^ { 2 } n \widetilde { \sigma } ^ { 2 } \leq \frac { 1 6 C _ { p } } { D } G \leq \frac { 1 } { 4 } G ,
$$

and, since $K \geq 1 6$

$$
C _ { p } h ^ { p } n \widetilde { B } e ^ { 2 h x / K } = C _ { p } 4 ^ { p } G ^ { p } \Delta e ^ { 8 G / K } \leq C _ { p } 4 ^ { p } G ^ { p } e ^ { - G / 2 } \leq C _ { p } ^ { \prime \prime } .
$$

Consequently, the exponential term in (115) is at most $C _ { p } e ^ { - 1 1 G / 4 } \leq C _ { p } \Delta$ , and we again obtain

$$
\mathbb { P } \Big \{ \Big \| \sum _ { i = 1 } ^ { n } z _ { i } \Big \| _ { \mathcal { H } } > x \Big \} \leq C _ { p } \left( \Delta + e ^ { - c _ { p } ^ { \prime } H } \right) .
$$

Combining the two regimes proves (110).

Finally, take $x = n t$ . Then $n \widetilde { B } / x ^ { p } = \widetilde { B } / ( n ^ { p - 1 } t ^ { p } )$ and $x ^ { 2 } / ( n \widetilde { \sigma } ^ { 2 } ) = n t ^ { 2 } / \widetilde { \sigma } ^ { 2 }$ , which proves (108) after renaming the constants and taking $c _ { p } \geq 1$

For the confidence bound, set

$$
u _ { 1 } = \left( \frac { 2 c _ { p } \widetilde { B } } { \eta n ^ { p - 1 } } \right) ^ { 1 / p } , \qquad u _ { 2 } = \widetilde { \sigma } \sqrt { \frac { \log ( 2 c _ { p } / \eta ) } { c _ { p } ^ { \prime } n } } , \qquad u = \operatorname* { m a x } \{ u _ { 1 } , u _ { 2 } \} .
$$

Evaluated at $t = u$ , each of the two terms on the right-hand side of (108) is at most $\eta / 2$ . Thus $\begin{array} { r } { \left\| { n } ^ { - 1 } \sum _ { i = 1 } ^ { n } { z _ { i } } \right\| _ { \mathcal { H } } \le u } \end{array}$ with probability at least $1 - \eta$ . Since $u \leq u _ { 1 } + u _ { 2 }$ , this proves (109). □

## G Embedding Condition

Lemma 30. Under Assumption $^ { 4 , }$ then for all $\lambda > 0$ and $\alpha _ { 0 } < \alpha \le 1$ , the following maximal capacity condition holds

$$
\mathcal { N } _ { \infty } ( \lambda ) = \mathop { \mathrm { e s s s u p } } _ { x } \sum _ { j } \frac { \mu _ { j } e _ { j } ( { \pmb x } ) ^ { 2 } } { \mu _ { j } + \lambda } \leq M _ { \alpha } \lambda ^ { - \alpha } .
$$

Proof. Recall the definition of the embedding condition (13), which states that there exists a constant $M _ { \alpha }$ for any $\alpha _ { 0 } < \alpha$ such that

$$
\sum _ { j } \mu _ { j } ^ { \alpha } e _ { j } ( { \pmb x } ) ^ { 2 } \le M _ { \alpha } , \quad \mathrm { f o r } \ \rho _ { X } \mathrm { - a l m o s t \ e v e r y } \ x .
$$

Without loss of generality, we assume that the constant $M _ { \alpha } \geq 1$ , since we can replace it with max $\{ M _ { \alpha } , 1 \}$ if necessary. Then, for any $\lambda > 0$ , we have

$$
\sum _ { j } \frac { \mu _ { j } e _ { j } ( \boldsymbol { x } ) ^ { 2 } } { \mu _ { j } + \lambda } \overset { } { \leq } \sum _ { j } \frac { \mu _ { j } ^ { 1 - \alpha } \mu _ { j } ^ { \alpha } e _ { j } ( \boldsymbol { x } ) ^ { 2 } } { \mu _ { j } + \lambda } \leq \sum _ { j } \mu _ { j } ^ { \alpha } e _ { j } ( \boldsymbol { x } ) ^ { 2 } \operatorname* { s u p } _ { j } \frac { \mu _ { j } ^ { 1 - \alpha } } { \mu _ { j } + \lambda } \leq M _ { \alpha } \lambda ^ { - \alpha } ,
$$

where the last inequality follows from $\begin{array} { r } { \operatorname* { s u p } _ { t \geq 0 } \frac { t ^ { s } } { t + \lambda } \leq \lambda ^ { s - 1 } } \end{array}$ for $0 \leq s \leq 1$

## H Useful Lemmas

Lemma 31 (Scalar Bernstein inequality). Let $U _ { 1 } , \dots , U _ { n }$ be independent real-valued random variables such that,

$$
0 \leq U _ { i } \leq b , \qquad \mathbb { E } U _ { i } \leq m , \qquad \operatorname { V a r } ( U _ { i } ) \leq v
$$

almost surely for some constants $b , m , v > 0$ . Then for every $u > 0$

$$
\mathbb { P } \Big ( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \big ( U _ { i } - \mathbb { E } U _ { i } \big ) > u \Big ) \le \exp \Big [ { - \frac { n u ^ { 2 } } { 2 \big ( v + b u / 3 \big ) } } \Big ] .\tag{116}
$$

Lemma 32 (Matrix Bernstein inequality). Let $Z _ { 1 } , \ldots , Z _ { n } \in \mathbb { R } ^ { d \times d }$ be independent self-adjoint random matrices such that,

$$
0 \leq \left\| Z _ { i } \right\| \leq B , \qquad \mathbb { E } Z _ { i } = 0 , \qquad \left\| { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \mathbb { E } \left[ Z _ { i } ^ { 2 } \right] \right\| \leq V
$$

almost surely for some constants $B , V > 0$ , where $\| \cdot \|$ denotes the operator norm. Then for every $U > 0$

$$
\mathbb { P } \Big ( \Big \| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } Z _ { i } \Big \| > U \Big ) \leq 2 d \exp \Big [ { - \frac { n U ^ { 2 } } { 2 \big ( V + B U / 3 \big ) } } \Big ] .\tag{117}
$$

## References

Amirhesam Abedsoltan, Mikhail Belkin, and Parthe Pandit. Toward large kernel models. In International Conference on Machine Learning, pages 61–78. PMLR, 2023.

Nachman Aronszajn. Theory of reproducing kernels. Transactions of the American mathematical society, 68(3):337–404, 1950.

Haim Avron, Kenneth L Clarkson, and David P Woodruf. Faster kernel ridge regression using sketching and preconditioning. SIAM Journal on Matrix Analysis and Applications, 38(4):1116– 1138, 2017.

Francis Bach. On the equivalence between kernel quadrature rules and random feature expansions. The Journal of Machine Learning Research, 18(1):714–751, 2017.

Gilles Blanchard and Nicole Kr¨amer. Optimal learning rates for kernel conjugate gradient regression. Advances in Neural Information Processing Systems, 23:226–234, 2010.

Stephen Boyd, Neal Parikh, Eric Chu, Borja Peleato, and Jonathan Eckstein. Distributed optimization and statistical learning via the alternating direction method of multipliers. Foundations and Trends® in Machine learning, 3(1):1–122, 2011.

Christian Brownlees, Emilien Joly, and G´abor Lugosi. Empirical risk minimization for heavy-tailed losses. The Annals of Statistics, 43(6):2507, 2015.

Andrea Caponnetto and Ernesto De Vito. Optimal rates for the regularized least-squares algorithm. Foundations of Computational Mathematics, 7:331–368, 2007.

Andreas Christmann and Ingo Steinwart. Consistency and robustness of kernel-based regression in convex risk minimization. Bernoulli, 13(3):799–819, 2007.

Felipe Cucker and Steve Smale. On the mathematical foundations of learning. Bulletin of the American Mathematical Society, 39(1):1–49, 2002.

Philip J. Davis and Philip Rabinowitz. Methods of Numerical Integration. Academic Press, 2 edition, 1984.

Ernesto De Vito, Andrea Caponnetto, and Lorenzo Rosasco. Model selection for regularized leastsquares algorithm in learning theory. Foundations of Computational Mathematics, 5(1):59–85, 2005.

Yunlong Feng and Qiang Wu. Learning under (1+ ϵ)-moment conditions. Applied and Computational Harmonic Analysis, 49(2):495–520, 2020.

Yunlong Feng and Qiang Wu. A statistical learning assessment of huber regression. Journal of Approximation Theory, 273:105660, 2022.

Simon Fischer and Ingo Steinwart. Sobolev norm learning rates for regularized least-squares algorithms. Journal of Machine Learning Research, 21(205):1–38, 2020.

D. H. Fuk. Some probabilistic inequalities for martingales. Siberian Mathematical Journal, 14: 131–137, 1973.

Gene H. Golub and John H. Welsch. Calculation of Gauss quadrature rules. Mathematics of Computation, 23(106):221–230, 1969.

Qiyang Han and Jon A Wellner. Convergence rates of least squares regression estimators with heavy-tailed errors. Annals of Statistics, 47(4):2286–2319, 2019.

Daniel Hsu and Sivan Sabato. Heavy-tailed regression with a generalized median-of-means estimator. In Proceedings of the 31st International Conference on Machine Learning, pages 37–45, 2014.

Peter J Huber and Elvezio M Ronchetti. Robust statistics. John Wiley & Sons, 2011.

Roger Koenker and Gilbert Bassett Jr. Regression quantiles. Econometrica, 46(1):33–50, 1978.

Arun K Kuchibhotla and Rohit K Patra. On least squares estimation under heteroscedastic and heavy-tailed errors. The Annals of Statistics, 50(1):277–302, 2022.

Samuel Lanthaler and Nicholas H Nelsen. Error bounds for learning with vector-valued random features. Advances in Neural Information Processing Systems, 36:71834–71861, 2023.

Jian Li and Yong Liu. Towards sharp analysis for distributed learning with random features. In Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, pages 3920–3928, 2023.

Jian Li, Yong Liu, and Weiping Wang. Optimal rates for agnostic distributed learning. IEEE Transactions on Information Theory, 70(4):2759–2778, 2023a.

Jian Li, Yong Liu, and Weiping Wang. Optimal convergence for agnostic kernel learning with random features. IEEE Transactions on Neural Networks and Learning Systems, 28:1–11, 2023b.

Youjuan Li, Yufeng Liu, and Ji Zhu. Quantile regression in reproducing kernel Hilbert spaces. Journal of the American Statistical Association, 102(477):255–268, 2007.

Zhu Li, Jean-Francois Ton, Dino Oglic, and Dino Sejdinovic. Towards a unified analysis of random fourier features. The Journal of Machine Learning Research, 22(1):4887–4937, 2021.

Junhong Lin and Volkan Cevher. Optimal convergence for distributed learning with stochastic gradient methods and spectral algorithms. The Journal of Machine Learning Research, 21(1): 5852–5914, 2020.

Junhong Lin and Lorenzo Rosasco. Optimal learning for multi-pass stochastic gradient methods. Advances in Neural Information Processing Systems, 29:4563–4571, 2016.

Junhong Lin, Alessandro Rudi, Lorenzo Rosasco, and Volkan Cevher. Optimal rates for spectral algorithms with least-squares regression over Hilbert spaces. Applied and Computational Harmonic Analysis, 48(3):868–890, 2020a.

Shao-Bo Lin, Xin Guo, and Ding-Xuan Zhou. Distributed learning with regularized least squares. The Journal of Machine Learning Research, 18(1):3202–3232, 2017.

Shao-Bo Lin, Di Wang, and Ding-Xuan Zhou. Distributed kernel ridge regression with communications. The Journal of Machine Learning Research, 21:3718–3755, 2020b.

Fanghui Liu, Xiaolin Huang, Yudong Chen, and Johan AK Suykens. Random features for kernel approximation: A survey on algorithms, theory, and beyond. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(10):7128–7148, 2021.

Jiamin Liu and Heng Lian. On optimal learning with random features. IEEE Transactions on Neural Networks and Learning Systems, 34(11):9536–9541, 2023.

Jiamin Liu and Heng Lian. Kernel-based adaptive Huber mean regression with heavy tails and contamination. Journal of the American Statistical Association, 2026. doi: 10.1080/01621459. 2026.2696486. Advance online publication.

Yong Liu, Jiankun Liu, and Shuqiang Wang. Efective distributed learning with random features: Improved bounds and algorithms. In International Conference on Learning Representations, pages 1–13, 2020.

Shahar Mendelson. Learning without concentration. Journal of the ACM (JACM), 62(3):1–25, 2015.

Shahar Mendelson. Upper bounds on product and multiplier empirical processes. Stochastic Processes and their Applications, 126(12):3652–3680, 2016.

Shahar Mendelson and Joseph Neeman. Regularization in kernel learning. The Annals of Statistics, 38(1):526–565, 2010.

Mattes Mollenhauer, Nicole M¨ucke, Dimitri Meunier, and Arthur Gretton. Regularized least squares learning with heavy-tailed noise is minimax optimal. Advances in Neural Information Processing Systems, 38:125703–125743, 2026.

Sergey V Nagaev. Large deviations of sums of independent random variables. The Annals of Probability, pages 745–789, 1979.

Iosif Pinelis. Optimum bounds for the distributions of martingales in banach spaces. The Annals of Probability, 22(4):1679–1706, 1994.

Svetlozar Todorov Rachev. Handbook of heavy tailed distributions in finance: Handbooks in finance, Book 1, volume 1. Elsevier, 2003.

Ali Rahimi and Benjamin Recht. Random features for large-scale kernel machines. Advances in Neural Information Processing Systems, 20:1177–1184, 2007.

Ali Rahimi and Benjamin Recht. Weighted sums of random kitchen sinks: Replacing minimization with randomization in learning. Advances in Neural Information Processing Systems, 21:1313– 1320, 2008.

Alessandro Rudi and Lorenzo Rosasco. Generalization properties of learning with random features. Advances in Neural Information Processing Systems, 30:3215–3225, 2017.

Alessandro Rudi, Rafaello Camoriano, and Lorenzo Rosasco. Less is more: Nystr¨om computational regularization. Advances in Neural Information Processing Systems, 28:1657–1665, 2015.

Alessandro Rudi, Luigi Carratino, and Lorenzo Rosasco. Falkon: An optimal large scale kernel method. Advances in Neural Information Processing Systems, 30:3891–3901, 2017.

Walter Rudin. Fourier analysis on groups. Courier Dover Publications, 2017.

Karl Sigman. A primer on heavy-tailed distributions. Queueing systems, 33(1-3):261, 1999.

Steve Smale and Ding-Xuan Zhou. Learning theory estimates via integral operators and their approximations. Constructive Approximation, 26(2):153–172, 2007.

Bharath Sriperumbudur and Zolt´an Szab´o. Optimal rates for random fourier features. Advances in Neural Information Processing Systems, 28:1144–1152, 2015.

Elias M. Stein and Rami Shakarchi. Fourier Analysis: An Introduction, volume 1 of Princeton Lectures in Analysis. Princeton University Press, 2003.

Ingo Steinwart and Andreas Christmann. Support vector machines. Springer Science & Business Media, 2008.

Ingo Steinwart and Clint Scovel. Mercer’s theorem on general domains: On the interaction between measures, kernels, and rkhss. Constructive Approximation, 35(3):363–417, 2012.

Ingo Steinwart, Clint Scovel, et al. Optimal rates for regularized least squares regression. In Proceedings of the Annual Conference on Learning Theory, 2009, pages 79–93, 2009.

Hongzhi Tong. Least squares regression under weak moment conditions. Journal of Computational and Applied Mathematics, 458:116336, 2025a.

Hongzhi Tong. Nonasymptotic analysis of robust regression with modified Huber’s loss. Journal of Complexity, 86:101878, 2025b.

Hongzhi Tong and Michael K. Ng. Ivanov regularized Huber regression in reproducing kernel Hilbert spaces. Analysis and Applications, 2025. doi: 10.1142/S0219530524410021.

Vladimir Vapnik. The nature of statistical learning theory. Springer Science & Business Media, 1999a.

Vladimir N Vapnik. An overview of statistical learning theory. IEEE Transactions on Neural Networks, 10(5):988–999, 1999b.

Grace Wahba. Spline models for observational data. SIAM, 1990.

Caixing Wang. Generalization properties of robust learning with random features. IEEE Transactions on Pattern Analysis and Machine Intelligence, pages 1–17, 2026. doi: 10.1109/TPAMI. 2026.3669168.

Caixing Wang and Xingdong Feng. Optimal kernel quantile learning with random features. In International Conference on Machine Learning, pages 50419–50452. PMLR, 2024.

Caixing Wang, Tao Li, Xinyi Zhang, Xingdong Feng, and Xin He. Communication-eficient nonparametric quantile regression via random features. Journal of Computational and Graphical Statistics, 33(4):1175–1184, 2024.

Caixing Wang, Zhibo Chen, and Yue Wang. Adaptive regularization for random features: A neighboring early-stopping rule with oracle-rate guarantees. arXiv preprint arXiv:2608.25513, 2026.

Wenjia Wang and Bing-Yi Jing. Gaussian process regression: Optimality, robustness, and relationship with kernel ridge regression. Journal of Machine Learning Research, 23(193):1–67, 2022.

Endre Weiszfeld and Frank Plastria. On the point for which the sum of the distances to n given points is minimum. Annals of Operations Research, 167(1):7–41, 2009.

Christopher Williams and Matthias Seeger. Using the nystroem method to speed up kernel machines. Advances in Neural Information Processing Systems, pages 682–688, 2001.

Qiang Wu, Yiming Ying, and Ding-Xuan Zhou. Learning rates of least-square regularized regression. Foundations of Computational Mathematics, 6(2):171–192, 2006.

Vadim Yurinsky. Sums and Gaussian vectors. Springer, 2006.

Haobo Zhang, Yicheng Li, Weihao Lu, and Qian Lin. On the optimality of misspecified kernel ridge regression. In Proceedings of the 40th International Conference on Machine Learning, volume 202, pages 41331–41353. PMLR, 2023.

Haobo Zhang, Yicheng Li, and Qian Lin. On the optimality of misspecified spectral algorithms. Journal of Machine Learning Research, 25(188):1–50, 2024.

Yuchen Zhang, John Duchi, and Martin Wainwright. Divide and conquer kernel ridge regression: A distributed algorithm with minimax optimal rates. The Journal of Machine Learning Research, 16(1):3299–3340, 2015.