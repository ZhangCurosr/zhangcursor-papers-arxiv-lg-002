# $\mathrm { N S } ^ { 3 } \mathrm { R }$ : A knowledge-driven nonconvex square-root regularization framework for joint group sparsity and structured smoothness

Wanling Xie<sup>a</sup>, Hu Yang<sup>b\*</sup>

<sup>a</sup>College of Mathematics and Statistics, Hunan University of Technology and Business, Changsha, Hunan 410205, China <sup>b</sup>College of Mathematics and Statistics, Chongqing University, Chongqing, 401331, China

## Abstract

Structured sparse learning is a leading paradigm for high-dimensional data modeling, as it enables rigorous integration of domain-validated structural priors into the learning process. Among these priors, pre-defined grouped structure and piecewise smoothness are ubiquitous in real-world applications. However, most existing structured regularization methods are built on squared loss and convex penalties, rarely jointly model these two complementary structural priors, and sufer from two key limitations. Their optimal tuning parameters are sensitive to the unknown noise level $\sigma$ due to the squared loss. Meanwhile, convex penalties induce inherent estimation bias and severe over-shrinkage of large magnitudes. To address these issues, we propose a knowledge-driven Nonconvex Structured Smooth Adjustment for Square-Root Regularization $( \mathrm { N S } ^ { 3 } \mathrm { R } )$ , which unifies the 𝜎-free square-root loss with a joint nonconvex $\ell _ { 2 , q }$ penalty $( 0 < q < 1 )$ to simultaneously encode group sparsity and structured smoothness priors. Theoretically, we establish rigorous non-asymptotic error bounds for the $\mathrm { N S } ^ { 3 } \mathrm { R }$ estimator under a mild $( 2 , q )$ -group restricted compatibility condition. Computationally, we develop an eficient double-screened IRLS with thresholding (DS-IRLS-th) algorithm, which adopts a nested double-screening strategy to reduce computational cost and improve numerical stability, with provable global convergence for the resulting nonconvex optimization problem. Extensive experiments on synthetic and real-world data demonstrate that $\mathrm { N S } ^ { 3 } \mathrm { R }$ consistently outperforms state-of-the-art methods in finite-sample settings.

Keywords: Structured sparse learning, Square-root regularization, Nonconvex $\ell _ { 2 , q }$ regularization, Structured smoothness, Global convergence

## 1. Introduction

High-dimensional sparse learning serves as a fundamental methodology in modern statistics and applied mathematics. The explosive growth of data dimensionality has created great challenges for statistical modeling and inference across a wide range of real-world scientific domains, including compressed sensing [1], genome-

wide association studies [2, 3], signal recovering [4, 5, 6], multi-task learning [7], econometric forecasting [8] and so on. In these practical applications, the relationship between the observed data and the target signal of interest can be well approximated by the linear system

$$
\boldsymbol { y } = \boldsymbol { X } \beta ^ { * } + \varepsilon\tag{1}
$$

In this system, $\ b { X } \in \mathbb { R } ^ { n \times p }$ denotes the measurement matrix, $\boldsymbol { y } \in \mathbb { R } ^ { n }$ corresponds to the noisy observation vector, $\beta ^ { * } \in \mathbb { R } ^ { p }$ is the unknown true signal vector we aim to recover, and 𝜀 is the random noise. Throughout this work, we adopt the standard Gaussian noise assumption, where 𝜀 has zero mean and unknown covariance $\sigma ^ { 2 } I _ { n }$ . A core property of high-dimensional scenarios is that the dimension $p$ is far greater than the number of observations 𝑛 $( \mathrm { i } . \mathrm { e } . , p \gg n )$ , which renders conventional model selection and estimation techniques inefective.

Building on the sparse learning framework widely used for high-dimensional data, structured sparse learning has emerged as the standard approach to address the limitations of traditional model selection methods under the $p \gg n$ setting. It moves beyond the entry-wise sparsity induced by the Lasso-type procedures [9] by leveraging inherent structural priors of the sparse signal we aim to recover, the most common and practically valuable of which are pre-defined group sparsity and fused sparsity. For group sparsity, the most widely used method is the group Lasso [10], which adopts a convex $\ell _ { 2 , 1 }$ penalty to induce group-level sparsity by encouraging elements in the same group to be either all non-zero or all zero simultaneously. Despite its popularity, the convex penalty brings excessive shrinkage on large non-zero coeficients resulting in non-negligible estimation bias, and tends to produce overly dense solutions leading to sub-optimal recovery performance in practice. To address these limitations, a range of nonconvex group penalties have been proposed, including the widely used group SCAD [11, 12, 13], group MCP [12, 13], and capped folded concave penalties [14, 15]. Also, the nonconvex $\ell _ { r , q }$ regularization has drawn extensive attention for its superior flexibility in modeling group sparsity [4, 6, 16], with its general form given by

$$
\| \beta \| _ { r , q } : = \left\{ \begin{array} { l l } { \left( \sum _ { j = 1 } ^ { J } \| \beta _ { \mathcal G _ { j } } \| _ { r } ^ { q } \right) ^ { \frac { 1 } { q } } , } & { q > 0 } \\ { \sum _ { j = 1 } ^ { J } \| \beta _ { \mathcal G _ { j } } \| _ { r } ^ { 0 } , } & { q = 0 } \end{array} \right.
$$

where $\beta _ { \mathcal { G } _ { j } }$ denotes the 𝑗-th group of $\beta$ (see section 2 for more details about group structure). As shown in [4, 6, 16], the nonconvex $\ell _ { r , q }$ penalties achieve a more favorable trade-of between sparsity induction and estimation bias reduction than convex alternatives, and empirical evidence from their simulation studies further shows that $\ell _ { r , q }$ regularization with $r = 2$ outperforms its general $\ell _ { r , q }$ counterparts in most cases.

Fused sparsity for ordered features is another widely used structural prior in high-dimensional statistical modeling. The foundational method for this prior is the fused Lasso [17], which adds a total variation penalty to the standard Lasso for modeling piecewise constant signal patterns over ordered features. It can adaptively identify contiguous blocks of identical coeficients without pre-specified group information, and has been widely adopted in signal processing, genomic analysis, and image denoising. Despite its popularity, the fused Lasso has clear limitations that restrict its practical utility. Its core design is tailored to capture step-function patterns rather than smooth local variations, which not only incurs a tremendous computational cost as the dimension increases, but also renders it fundamentally unsuitable for signals with smoothly varying coeficients. To tackle these limitations, multiple extensions have been developed. The Smooth Lasso [18] replaces the original $\ell _ { 1 }$ fusion penalty with a squared $\ell _ { 2 }$ term to better capture gradual smooth shifts in coeficients, whereas the Spline Lasso [19] draws on cubic spline smoothing ideas to model more flexible coeficient trajectories. Subsequent works have further extended this framework in model development, theoretical analysis and applied extensions [20, 21]. Nevertheless, most existing structured regularization methods focus only on either group structures or ordered structures, and cannot simultaneously incorporate both, which are ubiquitous in real-world applications.

Nearly all the structured sparse learning methods discussed above are built on the squared loss function. A critical limitation of these penalized least-squares is the challenge of regularization parameter tuning, as the theoretically optimal penalty values are closely tied to the unknown noise level $\sigma .$ . Accurate estimation of $\sigma$ is notoriously dificult in high-dimensional regimes, introducing practical hurdles and a heavy tuning burden in practical use. To address this critical issue, the square-root regularization framework was originally proposed. The classic square-root Lasso [22] replaces the squared loss with a square-root loss function, enabling scaleindependent parameter tuning as the optimal regularization parameter no longer depends on the unknown noise level. This $\sigma \cdot$ -free tuning property has motivated a series of extensions to structured sparse settings. The group square-root Lasso [23] extends the framework to group-level feature selection; the square-root fused Lasso [24] adapts the framework to capture local constant signal patterns; the SSASR model [5] combines the square-root loss with a convex $\ell _ { 2 , 1 }$ penalty to jointly model group sparsity and piecewise smoothness. Recent studies have further advanced this line of work by refining statistical theory and developing more eficient numerical algorithms [25, 26, 27, 28]. However, the vast majority of existing square-root regularized methods adopt convex penalty terms, leading to unavoidable estimation bias and sub-optimal recovery performance. To the best of our knowledge, no existing work has proposed a nonconvex square-root regularization framework that unifies group sparsity and structured smoothness priors. Meanwhile, this challenging nonconvex optimization problem lacks systematic theoretical guarantees and eficient convergent algorithms.

Motivated by the aforementioned limitations in existing literature, we develop a novel nonconvex structured smooth adjustment for square-root regularization $( \mathrm { N S } ^ { 3 } \mathrm { R } )$ framework, which combines the square-root loss function with a joint nonconvex $\ell _ { 2 , q }$ penalty for high-dimensional linear regression. It simultaneously encodes pre-defined group sparsity and structured smoothness priors in the sparse signal, while alleviating the common drawbacks of $\sigma \cdot$ -dependent tuning and over-shrinkage from convex regularization. The main contributions of this work are summarized as follows.

• Based on a joint $\ell _ { 2 , q }$ penalty, we propose a unified nonconvex square-root regularization framework to fill a critical gap in structured sparse learning. This framework enables joint modeling of group sparsity and structured smoothness while retaining the inherent 𝜎-free tuning property of square-root regularization. It further mitigates the inherent over-shrinkage and estimation bias of conventional convex regularizers, and thus achieves more reliable recovery of weak signals in high-dimensional regimes.

• For the $\mathrm { N S } ^ { 3 } \mathrm { R }$ estimator, we derive non-asymptotic Oracle inequalities under one of the mildest regularity conditions in existing literature. These results provide reliable theoretical guarantees for its practical application, and ensure stable finite-sample performance in high-dimensional regimes.

• To solve the challenging nonconvex, nonsmooth, and non-Lipschitz optimization problem, we develop an eficient double-screened IRLS with thresholding (DS-IRLS-th) algorithm. The proposed algorithm not only comes with a rigorous proof of global convergence under mild initial value conditions, but also reduces computational cost and improves numerical stability via its nested double-screening design. Comprehensive numerical experiments on both synthetic and real-world data further validate its fast convergence and favorable performance in $p \gg n$ settings.

The rest of the paper is organized as follows. In section 2, we present the nonconvex structured smooth adjustment for square-root regularization $( \mathrm { N S } ^ { 3 } \mathrm { R } )$ model and analyze its key theoretical properties. In section 3, we propose a double-screened IRLS with thresholding (DS-IRLS-th) algorithm admitting provable global convergence. In section 4 and section 5, we conduct simulation studies and an application to a microarray gene expression dataset to demonstrate the superior finite-sample performance of the proposed method compared with competing approaches. We conclude with final remarks in section 6. All technical proofs are deferred to the Appendix.

## 2. Methodology

## 2.1. The ${ \bf N S } ^ { 3 }$ R model

In this paper, we focus on recovering a sparse signal $\beta \in \mathbb { R } ^ { p }$ with a pre-defined grouped structure, i.e.,

$$
\boldsymbol { \beta } = \left( \underbrace { \beta _ { 1 , 1 } , . . . , \beta _ { 1 , T _ { 1 } } } _ { \beta _ { G _ { 1 } } } , \underbrace { \beta _ { 2 , 1 } , . . . , \beta _ { 2 , T _ { 2 } } } _ { \beta _ { G _ { 2 } } } , . . . , \underbrace { \beta _ { J , 1 } , . . . , \beta _ { J , T _ { J } } } _ { \beta _ { G _ { J } } } \right) ^ { T } .
$$

Here $\beta _ { \mathcal { G } _ { J } }$ denotes the sub-vector of $\beta$ formed by entries indexed in $\mathcal { G } _ { j }$ , where $\mathcal { G } _ { j }$ is an index subset of $\{ 1 , \ldots , p \}$ $J$ is the total number of groups, and $T _ { j } = | \boldsymbol { \mathcal { G } } _ { j } |$ denotes the size of the 𝑗-th group. For any group $\mathcal { G } _ { j } , \beta _ { \mathcal { G } _ { j } } = 0$ means $\beta _ { j , k } = 0$ for all $k \in \{ 1 , \ldots , T _ { j } \}$ , while $\beta _ { G _ { j } } \neq 0$ implies that all entries in the 𝑗-th group are non-zero. We assume in this paper that there is no overlap between any two groups, i.e., ${ \mathcal { G } } _ { i } \cap { \mathcal { G } } _ { k } = \emptyset$ for $i \neq k$

In practice, sparse signals often exhibit a piecewise (or blockwise) smooth structure, where the true entries within each block typically exhibit inherent similarity. Thus, incorporating within-group smoothness constraints into model design is well justified. Meanwhile, conventional convex regularization methods sufer from critical drawbacks $( \mathrm { e . g . }$ , inherent estimation bias and sub-optimal recovery performance) in terms of both theoretical guarantees and practical performance. To address these issues, we extend the group square-root regularization to a nonconvex fused framework. First, given $0 < q < 1$ and weights $\omega \in \mathbb { R } _ { + } ^ { J } \ ( \mathrm { i . e . , } \omega _ { j } \geq 0$ for all $1 \leq j \leq J )$ , for any vector $x \in \mathbb { R } ^ { p }$ with a pre-defined group structure, we denote the $\ell _ { 2 , q }$ regularization term as

$$
\| x \| _ { \omega - ( 2 , q ) } = \left( \sum _ { j = 1 } ^ { J } \omega _ { j } ^ { q } \| x _ { \mathcal { G } _ { j } } \| _ { 2 } ^ { q } \right) ^ { 1 / q }
$$

Then we propose a novel nonconvex structured smooth adjustment for square-root regularization $( \mathrm { N S } ^ { 3 } \mathrm { R } )$ as follows

$$
\hat { \beta } = \arg \operatorname* { m i n } _ { \beta \in \mathbb { R } ^ { p } } \frac { 1 } { \sqrt { n } } \| y - X \beta \| _ { 2 } + \frac { \lambda _ { 1 } } { n } \| \beta \| _ { \omega - ( 2 , q ) } ^ { q } + \frac { \lambda _ { 2 } } { n } \| D \beta \| _ { \omega - ( 2 , q ) } ^ { q }\tag{2}
$$

where $\lambda _ { 1 }$ and $\lambda _ { 2 }$ are tuning parameters that control the penalization in a diferent manner. To simplify notation, we let the overall regularization strength be $\lambda = \lambda _ { 1 } + \lambda _ { 2 }$ and define a regularization weight parameter $\alpha = \lambda _ { 1 } / \lambda$ that balances the relative contribution of the two penalization terms. Among the two nonconvex regularizers, the first term induces group-level sparsity in the signals, while the second is a nonconvex Inner Group Total Variation (IGTV) regularizer that imposes structured smoothness within groups, which is defined as

$$
\left. \boldsymbol { D } \boldsymbol { \beta } \right. _ { \omega - ( 2 , q ) } ^ { q } = \sum _ { j = 1 } ^ { J } \omega _ { j } ^ { q } \lVert D _ { \mathcal { G } _ { j } } \beta _ { \mathcal { G } _ { j } } \rVert _ { 2 } ^ { q } = \sum _ { j = 1 } ^ { J } \omega _ { j } ^ { q } \left[ \sum _ { k = 2 } ^ { T _ { j } } \left( \beta _ { j , k } - \beta _ { j , k - 1 } \right) ^ { 2 } \right] ^ { q / 2 }\tag{3}
$$

where $D$ is a block-diagonal matrix with the first-order diference matrices $D _ { \mathcal { G } _ { j } }$ for the 𝑗th group as each of its diagonal blocks, i.e., $D = d i a g ( D _ { \mathcal { G } _ { 1 } } , D _ { \mathcal { G } _ { 2 } } , . . . , D _ { \mathcal { G } _ { J } } ) \in \mathbb { R } ^ { ( p - J ) \times p }$ and

$$
D _ { \mathcal { G } _ { j } } = \left[ \begin{array} { c c c c c c } { 1 } & { - 1 } & { 0 } & { \cdots } & { 0 } & { 0 } \\ { 0 } & { 1 } & { - 1 } & { \cdots } & { 0 } & { 0 } \\ { \vdots } & { \vdots } & { \vdots } & { \ddots } & { \vdots } & { \vdots } \\ { 0 } & { 0 } & { 0 } & { \cdots } & { 1 } & { - 1 } \end{array} \right] _ { ( T _ { j } - 1 ) \times T _ { j } }
$$

There are many reasonable choices for the weights $\omega \in \mathbb { R } _ { + } ^ { J }$ , and a widely adopted choice in the literature is $\omega _ { j } = \sqrt { T _ { j } }$ . We therefore fix $\omega _ { j } = \sqrt { T _ { j } }$ throughout this paper. Then the problem in (2) is equivalent to the following optimization problem

$$
\hat { \beta } = \arg \operatorname* { m i n } _ { \beta \in \mathbb { R } ^ { p } } \frac { 1 } { \sqrt { n } } \| y - X \beta \| _ { 2 } + \frac { \lambda } { n } \left\{ \alpha \sum _ { j = 1 } ^ { J } T _ { j } ^ { q / 2 } \| \beta _ { G _ { j } } \| _ { 2 } ^ { q } + ( 1 - \alpha ) \sum _ { j = 1 } ^ { J } T _ { j } ^ { q / 2 } \| D _ { G _ { j } } \beta _ { G _ { j } } \| _ { 2 } ^ { q } \right\}\tag{4}
$$

Note that the objective function in (4) uses the square-root residual sum of squared errors as its loss function, with a joint $\ell _ { 2 , q }$ penalty that simultaneously encodes group sparsity and structured smoothness priors. In addition, problem (4) serves as a general nonconvex extension of the SSASR model in [5]. Compared to this convex counterpart, our nonconvex $\ell _ { 2 , q }$ regularization efectively alleviates the inherent estimation bias from convex penalties, and yields stronger sparsity induction and better smoothness modeling, especially for recovering weak signals in high-dimensional settings.

## 2.2. Theoretical Properties of $\mathbf { N S } ^ { 3 } \mathbf { R }$

To establish the theoretical properties of the $\mathrm { N S } ^ { 3 } \mathrm { R } .$ , we use the following notation and conventions throughout this paper. We assume 𝑋 is deterministic and normalized such that the diagonal entries of the Gram matrix $\Sigma : = X ^ { \top } X / n$ are 1. We use $\beta _ { \mathcal { G } _ { j } } = ( \beta _ { j , k } : k \in \{ 1 , . . . , T _ { j } \} )$ to denote the sub-vector of $\beta$ formed by entries indexed in ${ \mathcal { G } } _ { j } , X _ { { \mathcal { G } } _ { j } }$ to denote the submatrix of 𝑋 formed by columns in $\mathcal { G } _ { j }$ , and $D _ { \mathcal { G } _ { j } }$ to denote the first-order diference matrix of the 𝑗-th group. In what follows, whenever it does not cause confusion, we abbreviate $\beta _ { G _ { j } } , X _ { \mathcal { G } _ { j } }$ , and $D _ { \mathcal { G } _ { j } } : 0 \beta _ { j } , X _ { j }$ , and $D _ { j }$ , respectively. We define the group support set and support set of the true sparse signal $\beta ^ { * }$ as

$$
\mathbb { G } : = \operatorname { g s u p p } ( \boldsymbol { \beta } ^ { * } ) = \{ 1 \leq j \leq J : \boldsymbol { \beta } _ { j } ^ { * } \neq \mathbf { 0 } \} ,
$$

$$
S : = \bigcup _ { j \in \mathbb { G } } \mathcal { G } _ { j } ,
$$

with the group sparsity index $s ~ = ~ | \mathbb { G } |$ and the sparsity index $s ^ { * } ~ = ~ | \boldsymbol { S } |$ We use $\beta _ { S } \in \mathbb { R } ^ { s ^ { \ast } }$ (and similarly $\beta _ { S ^ { c } } \in \mathbb { R } ^ { p - s ^ { * } } )$ to denote the sub-vector of $\beta \in \mathbb { R } ^ { p }$ consisting of entries indexed in (or <sup>𝑐</sup>). Corresponding submatrix notation is used for 𝑋 and 𝐷. We also denote the maximum group size as $T _ { \operatorname* { m a x } } = \operatorname* { m a x } \{ T _ { j } : 1 \leq j \leq J \}$ and the minimum group size as $T _ { \operatorname* { m i n } } = \operatorname* { m i n } \{ T _ { j } : 1 \leq j \leq J \}$ . We write $a \lesssim b$ if $a \leq c b$ for some constant $c > 0$ and $a = O ( b )$ to indicate that 𝑎 and 𝑏 are of the same order up to a positive constant.

We now derive the global recovery bounds for $\mathrm { N S } ^ { 3 } \mathrm { R }$ , and begin by stating the necessary conditions for our results.

Definition 1 $( ( 2 , q )$ -Group Restricted Compatibility Condition $( ( 2 , q ) { \mathrm { - G R C C } } ) )$ . Let $0 < q \leq 1$ and $0 < \alpha \leq 1$

$\ b { X } \in \mathbb { R } ^ { n \times p }$ is said to satisfy the $( 2 , q )$ -Group Restricted Compatibility Condition for $\kappa > 0$ and $\gamma > 1 \ i f$

$$
\sum _ { j \in \mathbb { G } } \sqrt { T _ { j } } \| \Delta _ { j } \| _ { 2 } \leq \frac { \sqrt { s ^ { * } } \| X \Delta \| _ { 2 } } { \sqrt { n } \kappa }\tag{5}
$$

for all $\Delta \in \Delta _ { \gamma }$ , where

$$
\begin{array} { r l } & { \Delta _ { \gamma } : = \left\{ \Delta \in \mathbb { R } ^ { p } : \alpha \sum _ { j \in \mathbb { G } ^ { c } } T _ { j } ^ { q / 2 } \| \Delta _ { j } \| _ { 2 } ^ { 2 } + ( 1 - \alpha ) \sum _ { j \in \mathbb { G } ^ { c } } T _ { j } ^ { q / 2 } \| D _ { j } \Delta _ { j } \| _ { 2 } ^ { 2 } \right. } \\ & { \qquad \left. \leq \gamma \big ( \alpha \sum _ { j \in \mathbb { G } } T _ { j } ^ { q / 2 } \| \Delta _ { j } \| _ { 2 } ^ { 2 } + ( 1 - \alpha ) \sum _ { j \in \mathbb { G } } T _ { j } ^ { q / 2 } \| D _ { j } \Delta _ { j } \| _ { 2 } ^ { 2 } \big ) \right\} } \end{array}\tag{6}
$$

The $( 2 , q )$ -GRCC extends the Compatibility Condition (CC) adopted by the group square-root Lasso [23] to vectors whose structure is characterized by a joint $\ell _ { 2 , q }$ norm, and reduces to this original CC when $\alpha = 1$ and $q = 1$ . Existing literature has established that the 2, 1 -GRCC is much stronger than the 2, 𝑞 -GRCC for $q < 1$ , meaning the lower the $q ,$ <sup>,</sup> <sup>the</sup> <sup>weaker</sup> <sup>the</sup> (<sup>2,</sup> <sup>𝑞</sup>)<sup>-GRCC.</sup> <sup>The</sup> $( 2 , q )$ -GRCC therefore covers a broader range of cases than the standard CC. See [4, 29, 30, 31] and references therein for further discussion.

For the remainder of this paper, we assume that $\varepsilon _ { i }$ are i.i.d. Gaussian random variables following $N ( 0 , \sigma ^ { 2 } )$ , and that there exists a positive constant $\rho$ such that

$$
\| \beta ^ { * } \| _ { 2 , \infty } + \left( \frac { 1 - \alpha } { \alpha } \right) ^ { 1 / q } \| D \beta ^ { * } \| _ { 2 , \infty } \leq 2 ^ { - 2 / q } \rho .\tag{7}
$$

We define two random events associated with the linear regression model (1) as follows:

$$
\mathcal { A } = \left\{ \varepsilon : \frac { \| \varepsilon \| _ { 2 } } { \sqrt { n } } \leq \sigma \sqrt { 1 + t } \right\} ,\tag{8}
$$

and

$$
\mathcal { B } = \left\{ ( 2 \bar { \rho } ) ^ { 1 - q } V \leq \frac { \lambda _ { 1 } } { \bar { \gamma } } \right\} ,\tag{9}
$$

where the noise component 𝑉 is given by

$$
V = \operatorname* { m a x } _ { 1 \leq j \leq J } \left\{ \frac { \sqrt { n } \| X _ { j } ^ { \top } \varepsilon \| _ { 2 } } { \sqrt { T _ { j } } \| \varepsilon \| _ { 2 } } \right\} .\tag{10}
$$

Here, $\begin{array} { r } { \bar { \rho } = \left( \frac { \sigma n \sqrt { 1 + t } } { \lambda _ { 1 } } + \frac { 2 ^ { - q } s T _ { \mathrm { m a x } } ^ { q / 2 } \rho ^ { q } } { 2 } \right) ^ { 1 / q } } \end{array}$ and $\begin{array} { r } { \bar { \gamma } = \frac { \gamma + 1 } { \gamma - 1 } } \end{array}$ for $\gamma > 1$

With $\kappa > 0$ and $\gamma > 1$ given by the $( 2 , q ) { \mathrm { - G R C C } }$ above, we assume in what follows that the group sparsity

index 𝑠 and the sparsity index $s ^ { * }$ satisfy

$$
s ^ { * q / 2 } s ^ { 1 - q } \lesssim \frac { n } { \lambda } ,\tag{11}
$$

where $\lambda = \lambda _ { 1 } + \lambda _ { 2 }$ . We will show in Lemma A.4 that, in terms of orders of magnitude, the value of 𝜆 ensuring event $\mathcal { B }$ holds with high probability is bounded above by $\lambda = O \left( s ^ { \frac { 1 - q } { q } } { \sqrt { n \log J } } \right)$ . Therefore, the condition (11) is equivalent to

$$
s ^ { * q / 2 } s ^ { 1 / q - q } \lesssim \sqrt { \frac { n } { \log J } } .
$$

Intuitively, this means that when the number of groups 𝐽 greatly exceeds 𝑛, reliable performance can only be achieved if the true underlying model has substantially fewer parameters than 𝑛.

The following result demonstrates that the global solutions of the $\mathrm { N S } ^ { 3 } \mathrm { R }$ enjoy the dominant property with high probability.

Proposition 1. Let $\hat { \beta }$ be a global solution of criterion (4), and assume that condition (7) is satisfied. Then, on the event 𝒜 $\cap \mathcal { B }$ , thefollowing hold:

$$
\sum _ { j = 1 } ^ { J } \sqrt { T _ { j } } \| \hat { \beta } _ { j } - \beta _ { j } ^ { * } \| _ { 2 } \leq ( 2 \bar { \rho } ) ^ { 1 - q } \sum _ { j = 1 } ^ { J } T _ { j } ^ { q / 2 } \| \hat { \beta } _ { j } - \beta _ { j } ^ { * } \| _ { 2 } ^ { q }\tag{12}
$$

and

$$
\alpha \sum _ { j \in \mathbb { G } ^ { c } } T _ { j } ^ { q / 2 } \| \hat { \beta } _ { j } - \beta _ { j } ^ { * } \| _ { 2 } ^ { q } + ( 1 - \alpha ) \sum _ { j \in \mathbb { G } ^ { c } } T _ { j } ^ { q / 2 } \| D _ { j } ( \hat { \beta } _ { j } - \beta _ { j } ^ { * } ) \| _ { 2 } ^ { q }\tag{13}
$$

$$
\leq \gamma \Bigg [ \alpha \sum _ { j \in \mathbb { G } } T _ { j } ^ { q / 2 } \| \hat { \beta } _ { j } - \beta _ { j } ^ { * } \| _ { 2 } ^ { q } + ( 1 - \alpha ) \sum _ { j \in \mathbb { G } } T _ { j } ^ { q / 2 } \| D _ { j } ( \hat { \beta } _ { j } - \beta _ { j } ^ { * } ) \| _ { 2 } ^ { q } \Bigg ]
$$

where $\begin{array} { r } { \bar { \rho } = \left( \frac { \sigma n \sqrt { 1 + t } } { \lambda _ { 1 } } + \frac { 2 ^ { - q } s T _ { m a x } ^ { q / 2 } \rho ^ { q } } { 2 } \right) ^ { 1 / q } } \end{array}$

Remark 1. The result above implies that the estimation error $\Delta = \hat { \beta } - \beta ^ { * }$ lies in the set $\Delta _ { \gamma }$ (the feasible set of the $( 2 , q ) { \cdot } G R C C ,$ cf. Definition 1) whenever the event 𝒜  ℬ holds. By Lemma $A . 3$ and Lemma $A . 4 ,$ we have $\mathbb { P } ( \mathcal { A } ) \geq 1 - \delta _ { 1 }$ and $\mathbb { P } ( \mathcal { B } ) \geq 1 - \delta _ { 2 }$ under our assumptions, so that $\mathbb { P } ( \mathcal { A } \cap \mathcal { B } ) \geq 1 - \delta _ { 1 } - \delta _ { 2 }$ . This ensures $\Delta \in \Delta _ { \gamma }$ with probability at least $1 - \delta _ { 1 } - \delta _ { 2 }$ , rendering the subsequent theoretical results valid in a high-probability sense rather than just conditional on 𝒜 ℬ.

With the critical regularity condition established, we now state the main theorem of this section, which establishes the prediction and estimation properties of the $\mathrm { N S } ^ { 3 } \mathrm { R }$ under the weak $( 2 , q )$ -GRCC. This result generalizes [[23], Theorem II.1], where the group square-root Lasso is treated, corresponding in our set-up to the special case $q = 1$

Theorem 1 (Oracle Inequality). Assume that 𝑋 satisfies the $( 2 , q ) { \cdot } G R C C$ with $\kappa > 0$ and $\gamma > 1$ , and that (11)

$$
\lambda _ { 1 } \ge \frac { n } { \sqrt { n - T _ { m a x } } } T _ { m a x } ^ { \frac { 1 - q } { 2 } } \sqrt { 2 \zeta } s ^ { \frac { 1 - q } { q } } \rho ^ { 1 - q } \bar { \gamma } \left( 1 + \sqrt { \frac { 2 \log ( 2 J / \delta _ { 2 } ) } { T _ { m i n } } } \right) ,\tag{14}
$$

where $\begin{array} { r } { \zeta = \operatorname* { m a x } _ { j } \| X _ { j } \| _ { 2 } ^ { 2 } / n . } \end{array}$ . Then, with probability at least $1 - \delta _ { 1 } - \delta _ { 2 }$ , we have

$$
\frac { 1 } { \sqrt { n } } \| X ( \hat { \beta } - \beta ^ { * } ) \| _ { 2 } \lesssim \left( \sigma \frac { \lambda s ^ { 1 - q } s ^ { * q / 2 } } { n \kappa ^ { q } } \right) ^ { \frac { 1 } { 2 - q } }\tag{15}
$$

and

$$
\sum _ { j = 1 } ^ { J } T _ { j } ^ { q / 2 } \lVert \hat { \beta } _ { j } - \beta _ { j } ^ { * } \rVert _ { 2 } ^ { q } \lesssim \left( \sigma \frac { \lambda s ^ { 2 ( 1 / q - 1 ) } s ^ { * } } { n \kappa ^ { 2 } } \right) ^ { \frac { q } { 2 - q } } .\tag{16}
$$

Remark 2. (i) Theorem 1 establishes oracle inequalitiesfor the $N S ^ { 3 } R$ estimator under the weak $( 2 , q ) { \cdot } G R C C ,$ which stands as one of the mildest regularity conditions in the literature. As established in prior work [4, 31, 32], the compatibility condition and its related variants in the literature (e.g., q-restricted eigenvalue condition, group restricted eigenvalue condition) are strictly weaker than the Restricted Isometry Property (RIP) and are compatible with a wide range of design matrices, including Gaussian/sub-Gaussian random designs and grouped/block-wise correlated designs. Though the convergence rates for $q < 1$ are not superior to those of the classical $\ell _ { 2 , 1 }$ group Lasso, the required 2, 𝑞 -GRCC is substantially weaker: for problems where the 2, 𝑞 -GRCC holds but the standard group restricted eigenvalue condition fails, the group Lasso may yield inconsistent estimates, while our $N S ^ { 3 } R$ estimator still maintains valid recovery guarantees.

(ii) Notably, the tuning parameters $\lambda _ { 1 }$ and $\lambda _ { 2 }$ are 𝜎-free, i.e., independent of the unknown noise level $\sigma ,$ , a critical advantage in high-dimensional settings where 𝜎 is hard to estimate. In particular, when the maximum group size $T _ { \mathrm { m a x } } = { \cal { O } } ( 1 )$ , the estimator attains convergence rates of 𝑂 $\left( ( \sigma \lambda / n ) ^ { \frac { 1 } { 2 - q } } \sqrt { s } \right)$ for prediction and $O \left( ( \sigma \lambda / n ) ^ { \frac { q } { 2 - q } } s \right)$ for estimation, which coincide with the optimal rates for the $\ell _ { p , q }$ regularization problem discussed in [4] under the noise-free case. Unlike the penalized least-squares regularization that requires prior knowledge of 𝜎 for tuning, our 𝜎-independentframework avoids pre-estimating the noise level, a well-known challenge in ultra-high-dimensional settings. In practice, we canfix 𝜆 at its theoretical value and set $\lambda _ { 2 } = \theta \lambda _ { 1 }$ for a pre-specified constant $\theta > 0$ , or adaptively select the ratio 𝜃 via cross-validation (CV) or information criteria such as AIC or BIC to optimize predictive performance.

## 3. Algorithm and global convergence analysis

## 3.1. Subdiferential and optimality conditions

In this section, we establish the subdiferential (see Appendix A.1 for more details) and optimality conditions for the $\mathrm { N S } ^ { 3 } \mathrm { R }$ , both of which are critical for constructing an eficient iterative algorithm and analyzing its convergence properties. We begin with an equivalent reformulation of the original model (4) as follows

$$
\operatorname* { m i n } _ { \beta \in \mathbb { R } ^ { p } } F ( \beta ) = \| y - X \beta \| _ { 2 } + \lambda _ { 1 } \sum _ { j = 1 } ^ { J } T _ { j } ^ { q / 2 } \| \beta _ { j } \| _ { 2 } ^ { q } + \lambda _ { 2 } \sum _ { j = 1 } ^ { J } T _ { j } ^ { q / 2 } \| D _ { j } \beta _ { j } \| _ { 2 } ^ { q }\tag{17}
$$

The following proposition gives the explicit form of the subdiferential of $F ( \beta )$ , which is critical for deriving optimality conditions and analyzing the convergence of our algorithm.

Proposition 2. Assume that $y \ne X \beta$ over some set $C .$ . Then, on $C ,$ , the subdiferential of $F ( \beta )$ at $\beta$ is given groupwise for each group $j \in \{ 1 , \ldots , J \}$ by:

$$
( \partial F ( \beta ) ) _ { j } = \left\{ \begin{array} { l l } { - \frac { X _ { j } ^ { \top } \left( y - X \beta \right) } { \| y - X \beta \| _ { 2 } } + \lambda _ { 1 } T _ { j } ^ { q / 2 } q \| \beta _ { j } \| _ { 2 } ^ { q - 2 } \beta _ { j } } & { \beta _ { j } \not = 0 a n d D _ { j } \beta _ { j } \not = 0 , } \\ { + \lambda _ { 2 } T _ { j } ^ { q / 2 } q \| D _ { j } \beta _ { j } \| _ { 2 } ^ { q - 2 } D _ { j } ^ { \top } D _ { j } \beta _ { j } , } & \\ { ( - \infty , + \infty ) ^ { T _ { j } } , } & { o t h e r w i s e , } \end{array} \right.\tag{18}
$$

where $X _ { j } \in \mathbb { R } ^ { n \times T _ { j } }$ denotes the submatrix of 𝑋 corresponding to group 𝑗 with cardinality $T _ { j }$ , and $0 < q < 1$

In the framework of generalized subdiferentials, the well-known Fermat’s rule remains valid, that is, if $\beta \in \mathbb { R } ^ { p }$ is a local minimizer of $F ( \beta )$ then

$$
0 \in \partial { \cal F } ( \beta ) .\tag{19}
$$

A point $\beta \in \mathbb { R } ^ { p }$ that satisfies (19) is referred to as a stationary point of $F ( \beta )$ . For the optimization problem (17), if $\beta$ is a stationary point, then the componentwise optimality condition

$$
- \frac { X _ { j } ^ { \top } ( y - X \beta ) } { \| y - X \beta \| _ { 2 } } + \lambda _ { 1 } T _ { j } ^ { q / 2 } q \| \beta _ { j } \| _ { 2 } ^ { q - 2 } \beta _ { j } + \lambda _ { 2 } T _ { j } ^ { q / 2 } q \| D _ { j } \beta _ { j } \| _ { 2 } ^ { q - 2 } D _ { j } ^ { \top } D _ { j } \beta _ { j } = 0\tag{20}
$$

holds for all groups $j$ satisfying $\beta _ { j } \neq 0$ and $D _ { j } \beta _ { j } \neq 0$ . Further details can be found in Section 3.1 of [6].

## 3.2. Double-screened IRLS with thresholding algorithm

The objective function $F ( \beta )$ is nonconvex, nonsmooth, and non-Lipschitz, so most existing nonconvex algorithms either fail to guarantee convergence, or only ensure subsequential convergence under stringent conditions. Motivated by the IRLS with thresholding algorithm in [33], we propose a double-screened IRLS with thresholding (DS-IRLS-th) algorithm to solve the square-root optimization problem with a joint nonconvex $\ell _ { 2 , q }$ regularizer. Our algorithm adopts a majorization-minimization framework to iteratively solve a sequence of reweighted least squares problems, and incorporates a double-screening mechanism on both group-level coeficient magnitudes and within-group diference magnitudes. In practice, the DS-IRLS-th algorithm maintains two key sets at each iteration: the group support set $\mathbb { G } ^ { k }$ , including groups with coeficients exceeding the threshold $\tau _ { 1 } ;$ and the diference active set $\mathbb { A } ^ { k } .$ , a subset of $\mathbb { G } ^ { k }$ containing groups where within-group diferences exceed the threshold $\tau _ { 2 }$ . This nested structure guarantees a support-shrinking property across iterations, a critical requirement for proving convergence.

More precisely, let $\beta ^ { k }$ be the minimizer at the 𝑘-th iteration. By the nested double-screening mechanism, we only optimize the objective over the group support set $\mathbb { G } ^ { k }$ , with coeficients for groups outside $\mathbb { G } ^ { k }$ fixed to zero. For groups in $\mathbb { G } ^ { k } \setminus \mathbb { A } ^ { k }$ , all coeficients within the same group are constrained to a constant value. Equivalently, we solve the following constrained optimization problem at the 𝑘  1 th iteration:

$$
\widetilde { \beta } ^ { k + 1 } = \arg \operatorname* { m i n } _ { \beta \in \mathbb { R } ^ { p } } \left\{ \frac { 1 } { r _ { k } } \| y - X \beta \| _ { 2 } ^ { 2 } + \lambda _ { 1 } \sum _ { j \in \widetilde { \Omega } ^ { k } } T _ { j } ^ { q / 2 } w _ { 1 , j } ^ { k } \| \beta _ { j } \| _ { 2 } ^ { 2 } + \lambda _ { 2 } \sum _ { j \in \widetilde { \Omega } _ { k } ^ { k } } T _ { j } ^ { q / 2 } w _ { 2 , j } ^ { k } \| D _ { j } \beta _ { j } \| _ { 2 } ^ { 2 } \right\} ,\tag{21}
$$

$$
\begin{array} { r } { \mathrm { ~ . ~ } \beta _ { j } = 0 , \forall j \notin \mathbb { G } ^ { k } , \qquad D _ { j } \beta _ { j } = 0 , \forall j \in \mathbb { G } ^ { k } \setminus \mathbb { A } ^ { k } } \end{array}
$$

where $r _ { k } = \| y - X \beta ^ { k } \| _ { 2 }$ is the current residual norm, and the weights are defined by

$$
w _ { 1 , j } ^ { k } = q \| \beta _ { j } ^ { k } \| _ { 2 } ^ { q - 2 } , \forall j \in \mathbb { G } ^ { k } ; \quad w _ { 2 , j } ^ { k } = q \| D _ { j } \beta _ { j } ^ { k } \| _ { 2 } ^ { q - 2 } , \forall j \in \mathbb { A } ^ { k } .
$$

Our algorithm is summarized as follows.

Algorithm 1 DS-IRLS-th: Double-screened IRLS with Thresholding Algorithm   
Input: Threshold $\tau _ { 1 } , \tau _ { 2 } > 0 ;$ initial point $\boldsymbol { \beta } ^ { 0 } \in \mathbb { R } ^ { p }$ satisfying $\| \beta _ { j } ^ { 0 } \| _ { 2 } \ge \tau _ { 1 }$ and $\| D _ { j } \beta _ { j } ^ { 0 } \| _ { 2 } \ge \tau _ { 2 }$ for all $1 \leq j \leq J .$   
Initialize: Set the group support set $\mathbb { G } ^ { 0 } = \{ j : \| \boldsymbol { \beta } _ { j } ^ { 0 } \| _ { 2 } \geq \tau _ { 1 } \}$   
Iteration: For $k = 0 , 1 , 2 , \ldots$ do   
1. Compute the residual norm: $r _ { k } = \| y - X \beta ^ { k } \| _ { 2 } .$   
2. Set the diference active set: $\mathbb { A } ^ { k } = \{ j \in \mathbb { G } ^ { k } : \| D _ { j } \beta _ { j } ^ { k } \| _ { 2 } \geq \tau _ { 2 } \}$   
3. Compute the weights for each group in the group support set:   
$w _ { 1 , j } ^ { k } = q \| \beta _ { j } ^ { k } \| _ { 2 } ^ { q - 2 } , \forall j \in \mathbb { G } ^ { k } ; \quad w _ { 2 , j } ^ { k } = q \| D _ { j } \beta _ { j } ^ { k } \| _ { 2 } ^ { q - 2 } , \forall j \in \mathbb { A } ^ { k } .$

4. Solve the following constrained optimization problem:

$$
\widetilde { \beta } ^ { k + 1 } = \arg \operatorname* { m i n } _ { \beta \in \mathbb { R } ^ { p } } \left\{ \frac { 1 } { r _ { k } } \| y - X \beta \| _ { 2 } ^ { 2 } + \lambda _ { 1 } \sum _ { j \in \widetilde { \mathcal { Q } } ^ { k } } T _ { j } ^ { q / 2 } w _ { 1 , j } ^ { k } \| \beta _ { j } \| _ { 2 } ^ { 2 } + \lambda _ { 2 } \sum _ { j \in \mathring { \mathcal { Q } } _ { k } ^ { k } } T _ { j } ^ { q / 2 } w _ { 2 , j } ^ { k } \| D _ { j } \beta _ { j } \| _ { 2 } ^ { 2 } \right\} ,
$$

$$
\mathrm { s . t . } ~ \beta _ { j } = 0 , \forall j \notin \mathbb { G } ^ { k } , \quad D _ { j } \beta _ { j } = 0 , \forall j \in \mathbb { G } ^ { k } \setminus \mathbb { A } ^ { k } .
$$

or equivalently, denote $S ^ { k } = \cup _ { j \in \mathbb { G } ^ { k } } \mathcal { G } _ { j }$ , and define the group-wise transformation matrix:

$$
P _ { j } ^ { k } = \left\{ { \begin{array} { l l } { I _ { T _ { j } } , } & { j \in { \mathbb { A } } ^ { k } } \\ { \mathbf { 1 } _ { T _ { j } } , } & { j \in { \mathbb { G } } ^ { k } \setminus { \mathbb { A } } ^ { k } } \end{array} } \right.
$$

where $T _ { j }$ is the cardinality of the 𝑗-th group, $I _ { T _ { j } }$ is the $T _ { j }$ -dimensional identity matrix, and $\mathbf { 1 } _ { T _ { j } }$ is the $T _ { j }$ -dimensional vector with all components equal 1. Set $P = \mathrm { d i a g } ( P _ { \ i } ^ { k } , j \in \mathbb { G } ^ { k } ) , \tilde { X } _ { S ^ { k } } = X _ { S ^ { k } } P$ and block-diagonal weight matrix $\boldsymbol { W } ^ { k } = \mathrm { d i a g } ( W _ { i } ^ { k } , j \in \mathbb { G } ^ { k } )$ with

$$
\begin{array} { r } { W _ { j } ^ { k } = \left\{ \begin{array} { l l } { \lambda _ { 1 } T _ { j } ^ { q / 2 } w _ { 1 , j } ^ { k } I _ { T _ { j } } + \lambda _ { 2 } T _ { j } ^ { q / 2 } w _ { 2 , j } ^ { k } D _ { j } ^ { T } D _ { j } , } & { j \in \mathbb { A } ^ { k } } \\ { \lambda _ { 1 } T _ { j } ^ { q / 2 } w _ { 1 , j } ^ { k } \cdot T _ { j } , } & { j \in \mathbb { G } ^ { k } \setminus \mathbb { A } ^ { k } } \end{array} \right. } \end{array}
$$

Solve the following linear system:

$$
\left( \frac { 1 } { r _ { k } } \tilde { X } _ { S ^ { k } } ^ { T } \tilde { X } _ { S ^ { k } } + W ^ { k } \right) \beta ^ { r e d } = \frac { 1 } { r _ { k } } \tilde { X } _ { S ^ { k } } ^ { T } y
$$

where $\beta ^ { r e d }$ is the reduced variable, and the original coeficient is recovered by $\tilde { \beta } _ { S ^ { k } } ^ { k + 1 } = P \beta ^ { r e d }$

5. Thresholding: $\beta _ { j } ^ { k + 1 } = \left\{ \begin{array} { l l } { 0 , } & { \| \tilde { \beta } _ { j } ^ { k + 1 } \| _ { 2 } < \tau _ { 1 } } \\ { } & { } \\ { \tilde { \beta } _ { j } ^ { k + 1 } , } & { \mathrm { o t h e r w i s e } } \end{array} \right.$

6. Update the group support set: $\mathbb { G } ^ { k + 1 } = \{ j : \| \beta _ { j } ^ { k + 1 } \| _ { 2 } \geq \tau _ { 1 } \}$

end for

Output: The sequence $\{ \beta ^ { k } \}$

Remark 3. (i) The nested double-screening mechanism of the DS-IRLS-th algorithm restricts the diference active set $\mathbb { A } ^ { k }$ to groups with both non-negligible coeficients and non-flat within-group diferences. This avoids the numerical instability caused by excessively large weights from near-zero coeficients or within-group diferences in the reweighted least squares step. Importantly, all computations are restricted to the shrinking group support set, which significantly reduces the dimension of the linear system solved at each iteration.

Specifically, at the 𝑘 1 th iteration, we solve the reduced system

$$
\left( \frac { 1 } { r _ { k } } \tilde { X } _ { S ^ { k } } ^ { T } \tilde { X } _ { S ^ { k } } + W ^ { k } \right) \beta ^ { r e d } = \frac { 1 } { r _ { k } } \tilde { X } _ { S ^ { k } } ^ { T } y
$$

where $S ^ { k } = \cup _ { j \in \mathbb { G } ^ { k } } \mathcal { G } _ { j }$ denotes the support set, $\tilde { X } _ { S ^ { k } } = X _ { S ^ { k } } P$ is the reduced design matrix with transformation $P ,$ and $W ^ { k }$ is the block-diagonal weight matrix constructed from the estimate at the 𝑘-th iteration.

In early iterations, the relatively large support set makes direct matrix inversion computationally costly. For this case, we recommend the conjugate gradient method in [34], which is well-suited for large symmetric positive-definite systems. As the two nested sets $\mathbb { G } ^ { k }$ and $\mathbb { A } ^ { k }$ shrink rapidly and monotonically across iterations, the computational costfor solving the reduced linear system decreases substantially, giving our method a clear eficiency advantage over existing alternatives, especially for large-scale sparse recovery problems.

(ii) The DS-IRLS-th algorithm extends the IRLS-thframework in [33] by incorporating a double-screening mechanism designed for grouped and structured smoothness priors. This extension improves both sparse recovery accuracy and computational eficiency, driven by the rapid, monotonic shrinkage of the support set across iterations. A key advantage of this double-screening design is that it identifies pseudo-active groups, which have nearly flat within-group coeficients, and imposes equality constraints on all coeficients within these groups. This avoids numerical instability caused by near-zero within-group diferences in the reweighting step, while also greatly reducing the efective dimension ofthe optimization problem. For signals with inherent grouped and locally smooth structures, this leads to more reliable recovery of the true sparse pattern, and mitigates overfitting in smooth regions more efectively.

## 3.3. Global convergence analysis

In this subsection, we establish the global convergence result of the sequence $\{ \beta ^ { k } \}$ generated by the DS-IRLS-th algorithm. Our convergence analysis relies on the Kurdyka–Łojasiewicz (KL) inequality framework, which requires the objective function $F ( \beta )$ to satisfy the KL property. For completeness, we include the formal definitions of the KL property and semi-algebraic functions in Appendix A.2, both of which are classical results in nonconvex and nonsmooth optimization. We also verify in Appendix A.2 that our objective $F ( \beta )$ belongs to the class of semi-algebraic functions via standard closure properties of semi-algebraic sets and functions, which ensures that $F ( \beta )$ satisfies the KL property at all points relevant to our analysis. We prove convergence by verifying three key conditions for the KL framework, i.e., that $\{ \beta ^ { k } \}$ is bounded, that the function value sequence $\{ F ( \beta ^ { k } ) \}$ admits the suficient descent property, and that the subdiferential sequence $\{ \partial F ( { \boldsymbol { \beta } } ^ { k } ) \}$ is relatively bounded. Notably, our result holds without the proximal term required in most existing KL-based convergence analyses.

To apply the KL inequality framework, we verify the required regularity conditions in the two lemmas below. Lemma 1 characterizes the boundedness of the iterate sequence and the suficient descent property of the objective, while Lemma 2 establishes the relative boundedness of the subdiferential sequence.

Let $l _ { f }$ denote the local Lipschitz constant of the objective function $F ( \beta )$ around the KKT point, and let $\sigma _ { \mathrm { m i n } } ( D _ { j } ^ { \top } )$ denote the minimum singular value of the matrix $D _ { j } ^ { \top }$ . We now state our first lemma.

Lemma 1 (Boundedness and suficient descent of the objective function value). Let $\{ \beta ^ { k } \}$ be the sequence generated by the DS-IRLS-th algorithm. The following statements hold:

(i) Suppose the threshold satisfies $\tau _ { 2 } \leq \tilde { \tau } _ { 2 } = \left( \lambda _ { 2 } \sigma _ { \operatorname* { m i n } } ( D _ { j } ^ { \top } ) T _ { m i n } ^ { q / 2 } / l _ { f } \right) ^ { \frac { 1 } { 1 - q } }$ , then there exists a constant $C > \tau _ { 1 }$ such that

$$
C \geq \| \beta _ { j } ^ { k } \| _ { 2 } , \quad \forall 1 \leq j \leq J , k \geq 0 ;\tag{22}
$$

(ii) Suppose the thresholds satisfy $\tau _ { 1 } \le \tilde { \tau } _ { 1 } = \mathrm { m i }$ n $\left\{ \left( ( T _ { m i n } ^ { q / 2 } \lambda _ { 1 } ) / ( 2 \lVert X \rVert _ { 2 } ) \right) ^ { \frac { 1 } { 1 - q } } , ( \lambda _ { 1 } / ( 2 \tilde { a } ) ) ^ { \frac { 1 } { 2 - q } } \right\}$ , and $\tau _ { 2 } \leq \tilde { \tau } _ { 2 }$ , then there exists a constant $a > 0$ such that

$$
F ( \boldsymbol { \beta } ^ { k } ) - F ( \boldsymbol { \beta } ^ { k + 1 } ) \geq a \| \boldsymbol { \beta } ^ { k } - \boldsymbol { \beta } ^ { k + 1 } \| _ { 2 } ^ { 2 } ,\tag{23}
$$

holds for all $k \geq 0$

With the boundedness of the iterate sequence and the suficient descent property of the objective established, we next characterize the relative boundedness of the subdiferential sequence.

Lemma 2 (Relatively bounded subdiferential). For each $k \geq 0 ,$ suppose that $\| y - X \beta ^ { k } \| _ { 2 } > 0$ is uniformly boundedfrom below. Then there exists $u ^ { k + 1 } \in \partial { \cal F } ( \beta ^ { k + 1 } )$ such that

$$
\| u ^ { k + 1 } \| _ { 2 } \leq b \| \beta ^ { k } - \beta ^ { k + 1 } \| _ { 2 } ,\tag{24}
$$

where $^ b$ is a positive constant.

With the above two core properties of the DS-IRLS-th algorithm, we now establish its global convergence and finite-length property in the following theorem.

Theorem 2 (Global convergence and finite-length property). Let $\{ \beta ^ { k } \}$ be the sequence generated by the DS-IRLS-th algorithm. For each $k \geq 0 _ { : }$ , suppose that $\| y - X \beta ^ { k } \| _ { 2 } > 0$ is uniformly boundedfrom below, and the thresholds satisfy $\tau _ { 1 } \leq \tilde { \tau } _ { 1 } , \tau _ { 2 } \leq \tilde { \tau } _ { 2 } ( \tilde { \tau } _ { 1 } , \tilde { \tau } _ { 2 }$ are defined in Lemma 1). The following statements hold:

(i) The sequence $\{ \beta ^ { k } \}$ converges to a stationary point $\beta ^ { * }$ of the objective function $F ( \beta )$ , i.e.,

$$
\operatorname* { l i m } _ { k  \infty } \beta ^ { k } = \beta ^ { * } ;\tag{25}
$$

(ii) The sequence $\{ \beta ^ { k } \}$ has finite length, i.e.,

$$
\sum _ { k = 0 } ^ { \infty } \| { \boldsymbol { \beta } } ^ { k + 1 } - { \boldsymbol { \beta } } ^ { k } \| _ { 2 } < \infty .\tag{26}
$$

From Theorem 2, the proposed DS-IRLS-th algorithm globally converges to a stationary point of $F ( \beta )$ under mild regularity conditions. By contrast, most existing nonconvex algorithms for group-sparse regression only guarantee local (subsequence) convergence. This result holds for all iterations under the threshold conditions specified in Lemma 1.

## 4. Numerical experiments

Before investigating the finite-sample performance of the $\mathrm { N S } ^ { 3 } \mathrm { R }$ , we detail the parameter selection procedure for our algorithm. All numerical experiments are implemented in R and run on a personal desktop (Intel Core i7-14700K, 3.40 GHz, 64 GB of RAM).

We describe the general experimental setup below. We set the sample size $n \ : = \ : 2 5 0$ , the dimension $p = 1 0 0 0$ , the number of groups $J = 1 0 0$ , and the size of each group $T _ { j } = 1 0$ for all $1 \leq j \leq J $ . The sparsity level is defined as $s / J _ { \cdot }$ , where 𝑠 denotes the number of active groups in the true $\beta ^ { * }$ , and we vary 𝑠 across experiments. The coeficients within each active group are set to a constant value of 3, while all coeficients in inactive groups are fixed to 0. The covariates $( X _ { 1 } , X _ { 2 } , \ldots , X _ { p } )$ are independently generated from the standard normal distribution $N ( 0 , 1 )$ , and the response vector 𝑦 is generated by

$$
y = X { \boldsymbol { \beta } } ^ { * } + \sigma \cdot \operatorname { r a n d n } ( n , 1 ) ,
$$

where $\sigma$ is the standard deviation of the Gaussian noise. We fix $\sigma = 0 . 0 1$ in all experiments.

In our implementation, we first set the thresholds $\tau _ { 1 } = 1 0 ^ { - 2 } , \tau _ { 2 } = 1 0 ^ { - 4 }$ , and the exponent $q = 0 . 5$ for the proposed DS-IRLS-th algorithm. The algorithm is initialized with the ridge regression estimate, where the ridge penalty parameter is set to 0.01. For the regularization parameters, we adopt a modified version of the theoretical 𝜆-selection strategy proposed in Section IV of [23]. By the argument in [23], if $\begin{array} { r } { \lambda _ { 1 } \propto \sqrt { \frac { n \zeta \tau _ { 0 } } { T _ { \mathrm { m i n } } \tau _ { 0 } + n - T _ { \mathrm { m a x } } } } } \end{array}$ where

$$
\tau _ { 0 } = F _ { T _ { \mathrm { m i n } } , n - T _ { \mathrm { m a x } } } ^ { - 1 } \left( 1 - \alpha _ { \mathrm { s i g } } / J \right) , \quad \zeta = \operatorname* { m a x } _ { j } \frac { 1 } { n } \| X _ { j } \| _ { 2 } ^ { 2 } ,
$$

with $\alpha _ { \mathrm { s i g } }$ a pre-specified significance level, then the noise component 𝑉 is guaranteed to be dominated by $\lambda _ { 1 }$ . Our preliminary experiments find that a global scaling factor $c = 0 . 2$ yields stable performance across diferent sparsity levels. We thus set

$$
\lambda _ { 1 } = c \sqrt { \frac { n \zeta \tau _ { 0 } } { T _ { \mathrm { m i n } } \tau _ { 0 } + n - T _ { \mathrm { m a x } } } } .\tag{27}
$$

For the second regularization parameter, we set $\lambda _ { 2 } = \theta \lambda _ { 1 }$ , where the multiplier $\theta > 0$ is selected adaptively via the EBIC criterion [35] over the predefined grid $\{ 2 ^ { - 6 } , 2 ^ { - 5 } , \ldots , 2 ^ { 0 } \}$ . A fixed value of $\theta \left( \mathrm { e . g . , } \theta = 1 \right)$ may also be used as a simpler alternative.

For all experiments, we use the stopping criterion

$$
\frac { \| \beta ^ { k + 1 } - \beta ^ { k } \| _ { 2 } } { \| \beta ^ { k } \| _ { 2 } } < 1 0 ^ { - 4 } .
$$

For each sparsity level, we independently generate 100 datasets $( X , y , \beta ^ { * } )$ under the above setup, then run the algorithm on each dataset and average the results across 100 trials. We evaluate model selection performance using the metrics below.

The first metric we consider is the group-level false positive rate (FPR), defined as

$$
\frac { | \{ j : \beta _ { j } ^ { * } = 0 , \hat { \beta } _ { j } \neq 0 \} | } { | \{ j : \beta _ { j } ^ { * } = 0 \} | } ,
$$

where $\beta _ { j } ^ { * }$ and $\hat { \beta } _ { j }$ denote the true and estimated coeficient vectors for the 𝑗-th group, respectively.

The second metric is the success rate, defined as the proportion of the 100 trials where

$$
\frac { \| \hat { \beta } - \beta ^ { * } \| _ { 2 } } { \| \beta ^ { * } \| _ { 2 } } < 0 . 0 1 .
$$

## 4.1. Choice of $\tau _ { 1 } , \tau _ { 2 }$

In this subsection, we discuss the choice of $\tau _ { 1 } , \tau _ { 2 }$ . Following the discussion in Section 7.2 of [33], we fix $\tau _ { 1 } = 1 0 ^ { - 2 }$ . In our analysis, we test our algorithm with diferent $\tau _ { 2 }$ values from the set $\{ 1 0 ^ { - 6 } , 1 0 ^ { - 4 } , 1 0 ^ { - 2 } , 1 0 ^ { - 1 } , 1 \}$ across diferent sparsity levels. The success rates and average running times (over 100 independent trials) are given in Figure 1.

The results show that our algorithm with $\tau _ { 2 } = 1 0 ^ { - 6 }$ achieves the optimal balance between model selection accuracy and computational eficiency. Therefore, we fix $\tau _ { 1 } = 1 0 ^ { - 2 } { \mathrm { a n d } } \tau _ { 2 } = 1 0 ^ { - 6 }$ for all subsequent experiments.

## 4.2. Choice of $q$

In this subsection, we compare the performance of the DS-IRLS-th algorithm with diferent $q$ values from the set $\{ 0 . 1 , 0 . 3 , 0 . 5 , 0 . 7 , 0 . 9 \}$ across diferent sparsity levels. The resulting success rates and false positive rates are summarized in Figure 2, based on 100 independent replications.

The algorithm performs the worst when $q = 0 . 9$ . This poor performance stems from insuficient nonconvexity, which fails to efectively promote sparsity. In contrast, the algorithm achieves the best overall

![](images/8c0bbda34055cd3a7e794ff01cddf53e3daf6824974be35ce9756bef02bb16d7.jpg)  
(a)

![](images/41174dbd34c5d8795e562e3116e45d227ea6026e61319c0b54cccb9f4c84926c.jpg)  
(b)  
Fig. 1. The success rates and average running times of DS-IRLS-th algorithm with diferent threshold $\tau _ { 2 } .$

![](images/4a04cd11ea5b37ecbaec19607459b5856fd9d931885bcd45a089f24c71768d20.jpg)  
(a)

![](images/47454ad10af008549d1f448516db44cb2dc86c69d186e17470833f1d9c9bf518.jpg)  
(b)  
Fig. 2. The success rates and false positive rates of DS-IRLS-th algorithm with diferent exponent $q .$

performance when $q = 0 . 5$ , which aligns with the simulation results in [33]. We thus fix $q = 0 . 5$ for all subsequent experiments.

## 4.3. Convergence of the DS-IRLS-th algorithm

In this subsection, we analyze the convergence of the proposed DS-IRLS-th algorithm. Two sparsity levels, 1% and 20%, are considered in our experiments. Figure 3 presents the objective function values and relative errors across 50 iterations. We observe that, for both sparsity levels, the objective function values and

![](images/17db6f73dd5a50b7fc9bdcaa6d4090ed802ce7ccb34d9954c07da664dc8dced2.jpg)

![](images/796104ee56f51a4bb177e703b833fb152301aa0f1c1318cc0692d0a75f28d63d.jpg)

![](images/2da90b9cca78d1c965e170d3b9276ae63fe3fd4717cecc24571b99364044ac7b.jpg)  
(c)

![](images/6f66cd57c0086b6c462ede431d6f98b27ffb6f2371fcf98c9557798452efee51.jpg)  
(d)  
Fig. 3. Convergence performance of the DS-IRLS-th algorithm under sparsity levels 1% and 20%.

relative errors decrease monotonically and converge rapidly within just a few iterations. This demonstrates the efectiveness and fast convergence of the proposed DS-IRLS-th algorithm.

## 4.4. Comparisons to other methods

We conduct simulation studies to compare the performance of $\mathrm { N S } ^ { 3 } \mathrm { R }$ with several competing approaches in high-dimensional settings, including the smooth Lasso [18], the spline Lasso [19], the square-root fused Lasso [24] (SRFL), the group square-root Lasso [23] (GSRL), and the structured smooth adjustment for square-root regularization [5] (SSASR). We use the EBIC criterion from [35] for parameter tuning to ensure a fair comparison across all models. For methods requiring two tuning parameters, we set $\lambda _ { 2 } = \theta \lambda _ { 1 }$ , with the multiplier 𝜃 tuned over the predefined grid $\{ 2 ^ { - 6 } , 2 ^ { - 5 } , . . . , 2 ^ { - 1 } , 2 ^ { 0 } \}$ . To evaluate prediction accuracy, an additional test set of size $n _ { \mathrm { t e s t } } = 1 0 0 0 0$ is generated in each simulation, and model performance is assessed using the four metrics defined below:

• False discovery rate (FDR): The proportion of truly null features that are incorrectly selected among the

selected features, defined as

$$
\mathrm { F D R } = \frac { | \{ i : \beta _ { i } ^ { * } = 0 , \hat { \beta } _ { i } \neq 0 \} | } { | \{ i : \hat { \beta } _ { i } \neq 0 \} | } ,
$$

with the denominator set to 1 when no features are selected.

• False negative rate (FNR): The proportion of truly nonzero features that are missed by the selection procedure, defined as

$$
\mathrm { F N R } = \frac { | \{ i : \beta _ { i } ^ { * } \neq 0 , \hat { \beta } _ { i } = 0 \} | } { | \{ i : \beta _ { i } ^ { * } \neq 0 \} | } .
$$

• Estimation error (EE): The total squared error of the fitted regression entries, defined as

$$
\begin{array} { r } { \mathrm { E E } = ( \hat { \beta } - \beta ^ { * } ) ^ { \top } ( \hat { \beta } - \beta ^ { * } ) . } \end{array}
$$

• Prediction mean squared error (PMSE): The average squared prediction error on the independent test set, defined as

$$
\mathrm { P M S E } = \frac { 1 } { n _ { \mathrm { t e s t } } } \sum _ { i = 1 } ^ { n _ { \mathrm { t e s t } } } ( y _ { i } - \hat { y } _ { i } ) ^ { 2 } ,
$$

where $n _ { \mathrm { t e s t } }$ is the size of test set, $y _ { i }$ is the true response, and $\hat { y } _ { i }$ is the predicted response.

In all simulations, we fix the sample size $n = 2 5 0$ , dimension $p = 1 0 0 0$ , and group size 8 (resulting in 125 total groups). Among these groups, exactly 5 are designated as active (i.e., their associated entries are nonzero), with all entries in the remaining groups fixed at zero. The covariates $( X _ { 1 } , X _ { 2 } , \ldots , X _ { p } )$ are generated from a multivariate normal distribution $N ( 0 , \Sigma )$ , and the response 𝑦 is a linear function of the covariates with additive zero-mean Gaussian noise. To enable a comprehensive comparison, we consider two signal structures, two correlation structures, and two noise levels across our simulation scenarios, with each setting detailed in the following subsections.

## 4.4.1. Piecewise Constant Signals

Example 1 (Baseline) We generate piecewise constant signals to emulate the canonical setting where regression entries are homogeneous within each group and heterogeneous across groups. Among the 5 prespecified active groups, we assign a distinct constant magnitude to all entries within each group. To assess model performance in the weak-signal regime, we set entries in one active group to a low magnitude of 1, and the remaining four active groups to moderate magnitudes of 4, 5, 6, and 7. We then perturb these baseline entries with small zero-mean Gaussian noise of variance 0.01 to mimic real-world measurement noise. For this baseline setting, we fix the regression noise level at $\sigma = 1$ and use the identity correlation matrix $\Sigma = I$

Example 2 The setup is identical to Example 1, except that the noise level is set to $\sigma = 5$

Example 3 The setup is identical to Example 1, except that the correlation matrix is set to $\Sigma _ { j , k } = 0 . 5 ^ { | j - k | }$ Example 4 The setup is identical to Example 3, except that the noise level is set to $\sigma = 5$

## 4.4.2. Complex Smooth Signals

Example 5 (Baseline) We adopt the same group structure as Example 1, but generate complex smooth signals for active groups (rather than piecewise constant values) to emulate the intricate signal structures common in real-world applications. Each active group is assigned one of five distinct smooth patterns: linear increase, linear decrease, sinusoidal trend, quadratic growth, or exponential decay. We randomly parameterize each active group, adding small zero-mean Gaussian noise of variance 0.01 to mimic real-world measurement noise. For this baseline setting, we fix the regression noise level at $\sigma = 1$ and use the identity correlation matrix $\Sigma = I .$

Example 6 The setup is identical to Example 5, except that the noise level is set to $\sigma = 5$

Example 7 The setup is identical to Example 5, except that the correlation matrix is set to $\Sigma _ { j , k } = 0 . 5 ^ { | j - k | }$

Example 8 The setup is identical to Example 7, except that the noise level is set to $\sigma = 5$

## 4.4.3. Numerical Results

The proposed $\mathrm { N S } ^ { 3 } \mathrm { R }$ consistently outperforms all competing methods in variable selection, parameter estimation, and out-of-sample prediction across all simulation scenarios, and shows the highest stability by achieving the smallest standard deviations across replications. All the results for Examples 1–8 are summarized in Table 1, with values averaged over 100 independent replications and standard deviations reported in parentheses. As shown in Table 1, SSASR ranks second in all cases, while smooth Lasso and spline Lasso yield the worst performance in most cases. For Examples 1–4 with piecewise constant signals, SRFL underperforms the two convex group-regularized methods, GSRL and SSASR, across all metrics under low noise levels. When the noise level increases to $\sigma = 5$ , SRFL achieves comparable estimation and prediction performance to SSASR, owing to its inherent adaptability to piecewise constant structures. This comparable performance comes at the cost of lower variable selection accuracy, as it produces far denser models with substantially higher false discovery rates, and fails to efectively recover the true group-sparse structure.

Comparing Examples 1–2, 5–6 (independent covariates) with Examples 3–4, 7–8 (correlated covariates), the advantage of $\mathrm { N S } ^ { 3 } \mathrm { R }$ is more pronounced for independent covariates. In contrast, all competing methods face much more severe performance degradation relative to the correlated design, while $\mathrm { N S } ^ { 3 } \mathrm { R }$ maintains stable performance throughout. When the noise level increases from $\sigma = 1$ to $\sigma = 5$ , all methods degrade significantly as noise increases, whereas $\mathrm { N S } ^ { 3 } \mathrm { R }$ declines only mildly and maintains a clear advantage over all other approaches. In addition, $\mathrm { N S } ^ { 3 } \mathrm { R }$ consistently achieves exact recovery for low-magnitude signals, whereas all other methods cannot fully recover such signals under high noise levels. $\mathrm { N S } ^ { 3 } \mathrm { R }$ thus provides the best overall performance across all scenarios.

Table 1. Simulation results across all examples, evaluating FDR, FNR, EE, and PMSE across 100 replications.
<table><tr><td></td><td>Smooth lasso</td><td>Spline lasso</td><td>SRFL</td><td>GSRL</td><td>SSASR</td><td> $\overline { { \mathrm { N S } ^ { 3 } \mathrm { R } } }$ </td></tr><tr><td colspan="7">Example 1</td></tr><tr><td>FDR</td><td>47.00(0.02)</td><td>47.00(0.03)</td><td>5.54(0.05)</td><td>0.00(0.00)</td><td>0.00(0.00)</td><td>0.00(0.00)</td></tr><tr><td>FNR</td><td>1.28(0.02)</td><td>1.45(0.02)</td><td>0.00(0.00)</td><td>0.00(0.00)</td><td>10.20(0.10)</td><td>0.00(0.00)</td></tr><tr><td>EE</td><td>30.56(3.14)</td><td>30.71(3.32)</td><td>3.43(1.72)</td><td>2.64(0.92)</td><td>1.51(0.52)</td><td>0.20(0.05)</td></tr><tr><td>PMSE</td><td>31.77(3.16)</td><td>31.93(3.34)</td><td>4.44(1.73)</td><td>3.68(0.93)</td><td>2.53(0.52)</td><td>1.20(0.05)</td></tr><tr><td colspan="7">Example 2</td></tr><tr><td>FDR</td><td>71.26(0.02)</td><td>71.49(0.02)</td><td>2.91(0.04)</td><td>0.17(0.02)</td><td>0.00(0.00)</td><td>0.00(0.00)</td></tr><tr><td>FNR</td><td>8.50(0.03)</td><td>8.45(0.03)</td><td>8.15(0.09)</td><td>11.00(0.10)</td><td>10.20(0.10)</td><td>0.00(0.00)</td></tr><tr><td>EE</td><td>68.90(8.85)</td><td>67.06(8.45)</td><td>48.61(29.81)</td><td>63.43(22.12)</td><td>47.58(27.73)</td><td>4.70(3.34)</td></tr><tr><td>PMSE</td><td>94.00(8.92)</td><td>92.20(8.54)</td><td>73.96(30.05)</td><td>88.76(22.30)</td><td>72.88(27.92)</td><td>29.77(3.29)</td></tr><tr><td colspan="7">Example 3</td></tr><tr><td>FDR</td><td>14.87(0.05)</td><td>15.02(0.05)</td><td>11.36(0.09)</td><td>0.00(0.00)</td><td>0.00(0.00)</td><td>0.00(0.00)</td></tr><tr><td>FNR</td><td>0.00(0.00)</td><td>0.00(0.00)</td><td>0.00(0.00)</td><td>0.00(0.00)</td><td>0.00(0.00)</td><td>0.00(0.00)</td></tr><tr><td>EE</td><td>1.34(0.20)</td><td>1.36(0.21)</td><td>1.10(0.19)</td><td>0.60(0.11)</td><td>0.43(0.08)</td><td>0.30(0.08)</td></tr><tr><td>PMSE</td><td>2.62(0.19)</td><td>2.64(0.19)</td><td>2.07(0.31)</td><td>1.73(0.13)</td><td>1.57(0.11)</td><td>1.19(0.05)</td></tr><tr><td colspan="7">Example 4</td></tr><tr><td>FDR</td><td>15.35(0.05)</td><td>15.34(0.05)</td><td>13.93(0.08)</td><td>0.00(0.00)</td><td>0.00(0.00)</td><td>0.00(0.00)</td></tr><tr><td>FNR</td><td>0.95(0.01)</td><td>0.93(0.01)</td><td>0.00(0.00)</td><td>0.00(0.00)</td><td>0.00(0.00)</td><td>0.00(0.00)</td></tr><tr><td>EE</td><td>25.76(5.94)</td><td>25.12(5.44)</td><td>9.01(2.01)</td><td>10.10(2.01)</td><td>8.38(1.70)</td><td>6.40(1.85)</td></tr><tr><td>PMSE</td><td>58.45(7.78)</td><td>57.97(7.26)</td><td>41.61(3.95)</td><td>40.29(3.25)</td><td>39.21(2.86)</td><td>29.35(1.20)</td></tr><tr><td colspan="7">Example 5</td></tr><tr><td>FDR</td><td>24.95(0.03)</td><td>25.30(0.03)</td><td>4.54(0.05)</td><td>0.00(0.00)</td><td>0.00(0.00)</td><td>0.00(0.00)</td></tr><tr><td>FNR</td><td>0.68(0.01)</td><td>0.58(0.01)</td><td>0.63(0.01)</td><td>0.00(0.00)</td><td>0.00(0.00)</td><td>0.00(0.00)</td></tr><tr><td>EE</td><td>14.98(0.61)</td><td>14.89(0.61)</td><td>2.71(1.57)</td><td>1.15(0.27)</td><td>0.92(0.26)</td><td>0.19(0.05)</td></tr><tr><td>PMSE</td><td>15.78(0.60)</td><td>15.68(0.60)</td><td>3.76(1.59)</td><td>2.15(0.26)</td><td>1.92(0.25)</td><td>1.20(0.05)</td></tr><tr><td colspan="7">Example 6</td></tr><tr><td>FDR</td><td>53.39(0.08)</td><td>53.25(0.08)</td><td>2.69(0.03)</td><td>0.33(0.02)</td><td>0.00(0.00)</td><td>0.00(0.00)</td></tr><tr><td>FNR</td><td>6.30(0.02)</td><td>6.05(0.02)</td><td>4.33(0.02)</td><td>0.00(0.00)</td><td>0.00(0.00)</td><td>0.00(0.00)</td></tr><tr><td>EE</td><td>57.75(14.59)</td><td>55.40(12.93)</td><td>24.83(7.48)</td><td>22.58(5.61)</td><td>15.43(4.34)</td><td>4.58(1.74)</td></tr><tr><td>PMSE</td><td>82.01(14.53)</td><td>79.63(12.83)</td><td>50.20(7.64)</td><td>47.49(5.59)</td><td>40.47(4.32)</td><td>29.60(1.77)</td></tr><tr><td colspan="7">Example 7</td></tr><tr><td>FDR</td><td>19.60(0.07)</td><td>19.79(0.06)</td><td>9.48(0.04)</td><td>0.00(0.00)</td><td>0.00(0.00)</td><td>0.00(0.00)</td></tr><tr><td>FNR</td><td>0.83(0.01)</td><td>0.73(0.01)</td><td>2.50(0.00)</td><td>0.00(0.00)</td><td>0.00(0.00)</td><td>0.00(0.00)</td></tr><tr><td>EE</td><td>1.94(0.39)</td><td>1.87(0.40)</td><td>1.72(0.50)</td><td>0.76(0.16)</td><td>0.49(0.13)</td><td>0.28(0.08)</td></tr><tr><td>PMSE</td><td>3.11(0.48)</td><td>3.04(0.46)</td><td>3.05(0.93)</td><td>1.93(0.19)</td><td>1.65(0.18)</td><td>1.18(0.05)</td></tr><tr><td colspan="7">Example 8</td></tr><tr><td>FDR</td><td>18.57(0.07)</td><td>18.16(0.06)</td><td>10.25(0.05)</td><td>0.00(0.00)</td><td>0.00(0.00)</td><td>0.00(0.00)</td></tr><tr><td>FNR EE</td><td>4.80(0.02) 27.63(6.07)</td><td>4.78(0.02)</td><td>3.50(0.02)</td><td>0.00(0.00) 10.75(1.86)</td><td>0.00(0.00) 7.19(1.73)</td><td>0.00(0.00) 5.68(1.44)</td></tr><tr><td></td><td></td><td>26.67(6.07)</td><td>18.29(7.63)</td><td></td><td></td><td></td></tr><tr><td>PMSE</td><td>59.89(7.86)</td><td>59.30(7.85)</td><td>59.45(16.60)</td><td>41.62(3.27)</td><td>37.62(3.39)</td><td>29.14(1.11)</td></tr></table>

## 5. An application to gene expression microarray analysis

We apply our analytical framework to a gene expression dataset previously studied in [36]. This dataset has been deposited in the European Bioinformatics Institute (EMBL-EBI) Array-Express repository, ID: E-MEXP-1618. It contains expression microarray profiles of 22815 probe sets and bone biopsies from 84 women, with raw data normalized via the RMA method as detailed in [36]. For our analysis, bone mineral density (BMD) is the outcome of interest, quantified as hip T-scores from the participants and standardized to have zero mean and unit standard deviation. Gene expression levels are analyzed on a log2 scale. To incorporate domain-specific prior knowledge of bone metabolism into the model, we group all genes according to two sources of biological evidence. These include functionally validated bone metabolism-related genes from published studies [2, 3], and well-characterized bone metabolic pathways including bone resorption, osteoblast diferentiation and Wnt signaling cascades. We discard only the group of unannotated genes while retaining all other biologically informed grouping information, and sort all genes by their marginal importance relative to the bone density outcome. This strategy integrates prior biological insights into the leaning process, and improves the biological interpretability of the feature sets identified by our penalized regression model.

To illustrate the advantage of incorporating the structured smoothness prior, we compare our proposed NSR approach with two competing methods. These are the square-root fused Lasso (srfl) [24] and the external information tuned (xtune) Lasso [3]. For xtune Lasso, we use the xtune and glmnet [37] packages in R, and construct the external information matrix 𝑍 based on the predefined gene grouping structure. Each column of 𝑍 is a binary indicator for a specific group, with entries set to 1 for genes in the corresponding group and 0 otherwise. Regularization parameters for all methods are selected via the SCV-BIC criterion [38] to ensure consistency and fairness of the comparison. We randomly split the full dataset into a training set containing 80% of the 84 total observations (68 samples) and a held-out test set with the remaining 20% (18 samples). We fit all methods on the training data, and evaluate their predictive performance on the test set. This random split procedure is repeated for 100 independent simulations. Table 2 reports the mean number of selected genes and mean PMSE, with corresponding standard deviations in parentheses. Across all simulations, $\mathrm { N S } ^ { 3 } \mathrm { R }$ consistently selects a substantially smaller feature set than both xtune Lasso and srfl, with a markedly lower standard deviation that demonstrates superior stability in feature selection. Correspondingly, ${ \mathrm { N S } } ^ { 3 } { \mathrm { R } }$ achieves the lowest mean PMSE and outperforms both competing methods. This indicates that the sparse, stable feature sets identified by $\mathrm { N S } ^ { 3 } \mathrm { R }$ yield better predictive performance, while the larger, inconsistent gene sets selected by xtune Lasso and srfl f ail to deliver comparable improvements in prediction accuracy.

Notably, genes frequently selected by $\mathrm { N S } ^ { 3 } \mathrm { R }$ with a selection frequency of $\ge 3 0 \%$ across simulations are well-validated regulators of bone metabolism, and include core components of the Wnt signaling pathway such as DKK1, SOST, and WIF1. These core genes are robustly identified across all 100 simulations, confirming that $\mathrm { N S } ^ { 3 } \mathrm { R }$ can stably and reproducibly identify biologically meaningful signals relevant to bone metabolism. In comparison, xtune Lasso and srfl select a larger number of genes with lower selection frequencies. Most of the low-frequency genes selected by these competing methods have no validated functional link to bone metabolism, showing that these methods are more likely to capture false-positive features in high-dimensiona data.

In summary, these results demonstrate that $\mathrm { N S } ^ { 3 } \mathrm { R }$ achieves a favorable balance among sparse feature selection, predictive performance, and biological interpretability in high-dimensional genomic data analysis. By incorporating a structured smoothness prior adapted to the inherent grouping structure of genes, ${ \mathrm { N S } } ^ { 3 } { \mathrm { R } }$ provides a robust and biologically meaningful approach for feature selection in high-dimensional gene expression studies.

Table 2. The frequently selected genes, mean number of selected genes and mean PMSE across 100 replications on the test data.
<table><tr><td colspan="2">Method Frequently selected genes (≥ 30% in 100 simulations)</td><td>Number of se-PMSE lected genes</td><td></td></tr><tr><td>srfl</td><td>ACSL3 (100%), DKK1 (98%), C1orf61 (78%), IKBKG (70%), 21.75 (10.30) 0.75 (0.27) GATA1 (69%), HSD17B4 (69%), RRAS2 (68%), CTNNBIP1 (67%), FBN2 (67%), GNAS (66%), MIR18A (66%), WIF1 (57%), WWOX (57%), SOST (54%), IFITM1 (47%), SLC38A10 (47%), IL17A (43%), PTHLH (42%), CDH11 (39%), CARTPT (35%), SLC10A7 (34%), SNRNP200 (34%), TPH1 (34%), NFATC2 (33%), SMAD4 (32%), HIF1A (31%)</td><td></td><td></td></tr><tr><td>xtune</td><td>ACSL3 (98%), DKK1 (96%), WIF1 (96%), ELTD1 (93%), 17.89 (43.63) 0.68 (0.27) C1orf61 (88%), SOST (62%), SOX4 (58%), KDM4A (56%), PDGFA (32%)</td><td></td><td></td></tr><tr><td> $\mathrm { N S } ^ { 3 } \mathrm { R }$ </td><td>ACSL3 (100%), DKK1 (100%), ELTD1 (100%), SFRP1 (100%), 8.93 (2.41) SFRP4 (100%), SOST (100%), WIF1 (100%)</td><td></td><td>0.65 (0.24)</td></tr></table>

## 6. Conclusions and future works

In this work, we develop the nonconvex structured smooth adjustment for square-root regularization framework $( \mathrm { N S } ^ { 3 } \mathrm { R } )$ for high-dimensional structured sparse learning. It couples a joint nonconvex $\ell _ { 2 , q }$ penalty with a $\sigma \cdot$ -free square-root loss to simultaneously encode group sparsity and structured smoothness priors. This design mitigates the inherent over-shrinkage estimation bias from convex regularizers, and the dependency of tuning parameter selection on the unknown noise level 𝜎, which greatly facilitates the work of parameter tuning. We establish rigorous non-asymptotic estimation and prediction error bounds for the $\mathrm { N S } ^ { 3 } \mathrm { R }$ estimator under a mild regularity condition, providing reliable theoretical guarantees for its application in $p \gg n$ settings. For implementation, we develop an eficient double-screened IRLS with thresholding (DS-IRLS-th) algorithm for the resulting nonconvex, nonsmooth and non-Lipschitz optimization problem, with a rigorous proof of its global convergence under mild initial value conditions. Its nested double-screening design implements a support-shrinking strategy, which guarantees a rapid reduction of computational complexity across iterations. Numerical experiments show that the algorithm converges within few iterations, with a runtime below 1 second for moderate-sized datasets. Comprehensive experiments on synthetic and real-world gene expression datasets confirm that $\mathrm { N S } ^ { 3 } \mathrm { R }$ delivers consistently better finite-sample performance than state-of-the-art regularization methods.

In our future work, we will extend the proposed framework to accommodate more flexible structural priors and complex data settings. We will generalize the structured sparsity-inducing penalties to enable bi-level selection, enhancing its robustness against mis-specified grouping structures. We will also explore a broader class of nonconvex penalties, including group SCAD, group MCP and capped folded concave penalties, to achieve more accurate signal recovery in high-dimensional regimes. Further, we will extend the framework to graph-structured data to model label dependency across nodes via oriented incidence matrices, beyond simple piecewise smoothness assumptions. These extensions will provide a unified tool for high-dimensional data modeling in bioinformatics, signal processing, financial econometrics, image processing and other related application areas.

## Acknowledgments

This work is supported by the National Natural Science Foundation of China [Grant No. 12401345].

## Appendix A Preliminary Definitions and Supporting Lemmas

## A.1. Subdiferential

In this section, we recall the basic definition of the subdiferential.

Definition A.1 (Subdiferentials, [39]). Let $\phi : \mathbb { R } ^ { p }  ( - \infty , + \infty ]$ be a proper and lower semicontinuous function.

(i) The regular subdiferential of 𝜙 at $x \in$ dom $\phi = \left\{ x \in \mathbb { R } ^ { p } : \phi ( x ) < + \infty \right\}$ is defined as

$$
\hat { \partial } \phi ( x ) : = \left\{ u \in \mathbb { R } ^ { p } : \operatorname* { l i m } _ { y \neq x } \operatorname* { i n f } _ { \mathbf { \phi } } \frac { \phi ( y ) - \phi ( x ) - \langle u , y - x \rangle } { \| y - x \| _ { 2 } } \geq 0 \right\} ;
$$

(ii) The (limiting) subdiferential of 𝜙 at $x \in$ dom $\phi$ is defined as

$$
\partial \phi ( x ) : = \left\{ u \in \mathbb { R } ^ { p } : \exists x ^ { k } \to x , \ \phi ( x ^ { k } ) \to \phi ( x ) \ a n d \ \hat { \partial } \phi ( x ^ { k } ) \ \ni u ^ { k } \to u , \ a s \ k \to \infty \right\}
$$

## A.2. KL Property and Semi-Algebraic Functions

The Kurdyka-Łojasiewicz (KL) property plays a fundamental role in the convergence analysis of many optimization algorithms [40, 41]. A broad class of functions, including semi-algebraic, subanalytic, and log-exp functions, are KL functions. The definition of the KL property is given below.

Definition A.2 (KL property, [40]). The function $\phi : \mathbb { R } ^ { p }  ( - \infty , + \infty ]$ is said to have the Kurdyka-Łojasiewicz (KL) property at $\bar { x } \in d o m \partial \phi : = \{ x \in \mathbb { R } ^ { p } : \partial \phi ( x ) \neq 0 \}$ if there exists $\eta \in ( 0 , + \infty ]$ , a neighborhood 𝑈 of 𝑥¯, and a continuous concave function $\psi : [ 0 , \eta )  \mathbb { R } _ { + }$ such that

(i) $\psi ( 0 ) = 0 ;$

(ii) 𝜓 is continuously diferentiable on $( 0 , \eta )$

(iii) for all $t \in ( 0 , \eta ) , \psi ^ { \prime } ( t ) > 0 ,$

(iv) for all $x \in { U \cap \{ \phi ( \bar { x } ) < \phi ( x ) < \phi ( \bar { x } ) + \eta \} }$ , the KL inequality holds:

$$
\psi ^ { \prime } { \big ( } \phi ( x ) - \phi ( { \bar { x } } ) { \big ) } \operatorname { d i s t } { \big ( } 0 , \partial \phi ( x ) { \big ) } \geq 1 .
$$

$I f \phi$ satisfies the KL property at each point ofdom 𝜕𝜙, 𝜙 is called a KLfunction.

Semi-algebraic functions form an important subclass of KL functions, and their definition is given as follows.

Definition A.3 (Semi-algebraic sets and functions, [41]).

(i) A subset $o f \mathbb { R } ^ { p }$ is a real semi-algebraic set if there exists a finite number of real polynomialfunctions $g _ { i j } , h _ { i j } : \mathbb { R } ^ { p }  \mathbb { R }$ such that

$$
{ \mathcal { H } } = \bigcup _ { j } \bigcap _ { i } \left\{ u \in \mathbb { R } ^ { p } : g _ { i j } ( u ) = 0 a n d h _ { i j } ( u ) < 0 \right\} .
$$

(ii) A function 𝜙 : $\mathbb { R } ^ { p } \to ( - \infty , + \infty ]$ is called semi-algebraic if its graph

$$
\left\{ ( u , t ) \in \mathbb { R } ^ { p + 1 } : \phi ( u ) = t \right\}
$$

is a semi-algebraic subset of $\mathbb { R } ^ { p + 1 }$

The objective function $F ( \beta )$ defined in (17) is semi-algebraic. Indeed, the square-root loss $\| y - X \beta \| _ { 2 }$ is semi-algebraic (see [41, Example 2]), and for a rational exponent $q \in ( 0 , 1 )$ , the functions $\| \beta _ { j } \| _ { 2 } ^ { q }$ and $\| D _ { j } \beta _ { j } \| _ { 2 } ^ { q }$

are semi-algebraic. Since finite sums and compositions of semi-algebraic functions are semi-algebraic, 𝐹 is semi-algebraic and thus a KL function (see [41, Theorem 3]). This ensures that the KL property holds at every point of interest in our convergence analysis.

## A.3. Supporting Preliminary Lemmas

In order to prove our main results, it is necessary to present some technical lemmas. The following two lemmas are the key inequalities used in the proof.

Lemma A.1. Let $x , y \in \mathbb { R } ^ { p }$ . Then the following relations hold:

$$
( i ) \quad \| x \| _ { q _ { 2 } } \leq \| x \| _ { q _ { 1 } } \leq p ^ { \frac { 1 } { q _ { 1 } } - \frac { 1 } { q _ { 2 } } } \| x \| _ { q _ { 2 } } , f o r a n y 0 < q _ { 1 } \leq q _ { 2 } < + \infty .\tag{A.1}
$$

$$
( i i ) \ \| x \| _ { 2 , q } ^ { q } - \| y \| _ { 2 , q } ^ { q } \leq \| x - y \| _ { 2 , q } ^ { q } , f o r a n y 0 < q \leq 1 .\tag{A.2}
$$

(iii) Given $a > 0 , i f \| x \| _ { 2 } + \| y \| _ { 2 } \leq M$ , then

$$
\| x \| _ { 2 } ^ { q } + \| y \| _ { 2 } ^ { q } \leq 2 ^ { 1 - q } M ^ { q } , f o r a n y 0 < q \leq 1 .\tag{A.3}
$$

Proof. The left-hand side of (i) follows from the monotonicity of the $\ell _ { q }$ norms. For the right-hand side of (i), let $r = q _ { 2 } / q _ { 1 } \geq 1$ and let $r ^ { \prime } = r / ( r - 1 )$ be its Hölder conjugate. Applying Hölder’s inequality to the sequences $\textstyle ( | x _ { i } | ^ { q _ { 1 } } ) _ { i = 1 } ^ { p }$ and the constant sequence $( 1 ) _ { i = 1 } ^ { p }$ gives

$$
\sum _ { i = 1 } ^ { p } | x _ { i } | ^ { q _ { 1 } } \leq \left( \sum _ { i = 1 } ^ { p } | x _ { i } | ^ { q _ { 1 } r } \right) ^ { 1 / r } \left( \sum _ { i = 1 } ^ { p } 1 ^ { r ^ { \prime } } \right) ^ { 1 / r ^ { \prime } } = \left( \sum _ { i = 1 } ^ { p } | x _ { i } | ^ { q _ { 2 } } \right) ^ { q _ { 1 } / q _ { 2 } } p ^ { 1 - q _ { 1 } / q _ { 2 } } .
$$

Raising both sides to the power $1 / q _ { 1 }$ yields

$$
\| x \| _ { q _ { 1 } } = \left( \sum _ { i = 1 } ^ { p } | x _ { i } | ^ { q _ { 1 } } \right) ^ { 1 / q _ { 1 } } \leq \left( \sum _ { i = 1 } ^ { p } | x _ { i } | ^ { q _ { 2 } } \right) ^ { 1 / q _ { 2 } } p ^ { \frac { 1 } { q _ { 1 } } - \frac { 1 } { q _ { 2 } } } = p ^ { \frac { 1 } { q _ { 1 } } - \frac { 1 } { q _ { 2 } } } \| x \| _ { q _ { 2 } } .\tag{A.4}
$$

The second inequality is a direct application of Lemma 2 in [4]. For the third inequality, when $q = 1$ , the result follows immediately from the triangle inequality. When $0 < q < 1$ , let $u = \| x \| _ { 2 }$ and $\nu = \| y \| _ { 2 }$ . Then $u , \nu \geq 0$ and $u + \nu \leq M$ . Since the function $t \mapsto t ^ { q }$ is concave on $[ 0 , \infty )$ , by Jensen’s inequality we have

$$
\left( \frac { u + \nu } { 2 } \right) ^ { q } \geq \frac { u ^ { q } + \nu ^ { q } } { 2 } .\tag{A.5}
$$

Rearranging this inequality yields

$$
u ^ { q } + \nu ^ { q } \leq 2 ^ { 1 - q } ( u + \nu ) ^ { q } .
$$

Since $u + \nu \leq M$ and $s \mapsto s ^ { q }$ is strictly increasing on $( 0 , \infty )$ , we conclude

$$
\| x \| _ { 2 } ^ { q } + \| y \| _ { 2 } ^ { q } = u ^ { q } + \nu ^ { q } \leq 2 ^ { 1 - q } M ^ { q } .\tag{A.6}
$$

This completes the proof.

Lemma A.2. Given $0 < q < 1$ , for all 𝑥, $\boldsymbol { y } \in \mathbb { R } ^ { p }$ with $x \neq 0$

$$
\| x \| _ { 2 } ^ { q } - \| y \| _ { 2 } ^ { q } - q \| x \| _ { 2 } ^ { q - 2 } ( x - y ) ^ { \top } y \geq \frac { q } { 2 } \| x \| _ { 2 } ^ { q - 2 } \| x - y \| _ { 2 } ^ { 2 } .\tag{A.7}
$$

This lemma comes from Lemma 6.3 in [33].

Lemma A.3. For a given $\delta _ { 1 } \in ( 0 , 1 )$ , let $\begin{array} { r } { t = \sqrt { \frac { 4 \ln ( 1 / \delta _ { 1 } ) } { n } } + \frac { 4 \ln ( 1 / \delta _ { 1 } ) } { n } } \end{array}$ and define

$$
\mathcal { A } : = \left\{ \frac { \| \varepsilon \| _ { 2 } } { \sqrt { n } } \leq \sigma ^ { \sqrt { 1 + t } } \right\} .\tag{A.8}
$$

Then,

$$
\mathbb { P } ( \mathcal { A } ) \geq 1 - \delta _ { 1 } .\tag{A.9}
$$

The proof of this result is a direct application of Lemma 8.1 in [42].

Lemma A.4. Assume that $T _ { j } < n$ for all $1 \leq j \leq J .$ . Let $\zeta = \operatorname* { m a x } _ { j } \zeta _ { j }$ , where $\big \{ \zeta _ { j } = \| X _ { j } \| _ { 2 } ^ { 2 } / n \big \}$ , and $\delta _ { 2 } \in ( 0 , 1 )$ be given such that 16 log $\begin{array} { r } { \left( \frac { 2 J } { \delta _ { 2 } } \right) \le n - T _ { \operatorname* { m a x } } } \end{array}$ and $\begin{array} { r } { s ^ { 2 / q } \log \left( \frac { 2 J } { \delta _ { 2 } } \right) \geq \frac { \sigma ^ { 2 } ( 1 + t ) ( n - T _ { \operatorname* { m a x } } ) T _ { \operatorname* { m i n } } } { T _ { \operatorname* { m a x } } \zeta \rho ^ { 2 } } } \end{array}$ . If

$$
\lambda _ { 1 } \ge \frac { n } { \sqrt { n - T _ { \operatorname* { m a x } } } } T _ { \operatorname* { m a x } } ^ { \frac { 1 - q } { 2 } } \sqrt { 2 \zeta } s ^ { \frac { 1 - q } { q } } \rho ^ { 1 - q } \bar { \gamma } \left( 1 + \sqrt { \frac { 2 \log ( 2 J / \delta _ { 2 } ) } { T _ { \operatorname* { m i n } } } } \right) ,\tag{A.10}
$$

we have $\mathbb { P } ( \mathcal { B } ) \geq 1 - \delta _ { 2 }$

Proof. We first observe that

$$
\begin{array} { r l } & { \mathbb { P } ( \mathcal { B } ^ { c } ) = \mathbb { P } \left( ( 2 \bar { \rho } ) ^ { 1 - q } V > \displaystyle \frac { \lambda _ { 1 } } { \bar { \gamma } } \right) } \\ & { \qquad \leq \mathbb { P } \left( ( 2 \bar { \rho } ) ^ { 1 - q } V > \displaystyle \frac { n } { \sqrt { n - T _ { \operatorname* { m a x } } } } T _ { \operatorname* { m a x } } ^ { \frac { 1 - q } { 2 } } \sqrt { 2 \zeta } s ^ { \frac { 1 - q } { q } } \rho ^ { 1 - q } \left( 1 + \sqrt { \frac { 2 \log ( 2 J / \delta _ { 2 } ) } { T _ { \operatorname* { m i n } } } } \right) \right) } \\ & { \qquad \leq \mathbb { P } \left( \left( \displaystyle \frac { \sigma n \sqrt { 1 + t } } { 2 ^ { - q } \lambda _ { 1 } s T _ { \operatorname* { m a x } } ^ { q / 2 } \rho ^ { q } } + \displaystyle \frac { 1 } { 2 } \right) ^ { \frac { 1 - q } { q } } V > \displaystyle \frac { n } { \sqrt { n - T _ { \operatorname* { m a x } } } } \sqrt { 2 \zeta } \left( 1 + \sqrt { \frac { 2 \log ( 2 J / \delta _ { 2 } ) } { T _ { \operatorname* { m i n } } } } \right) \right) } \end{array}
$$

It follows from (A.10) that

$$
\begin{array} { r l r } {  { \frac { \sigma n \sqrt { 1 + t } } { 2 ^ { - q } \lambda _ { 1 } s T _ { \operatorname* { m a x } } ^ { q / 2 } \rho ^ { q } } \le \frac { \sigma n \sqrt { 1 + t } } { 2 ^ { - q } \frac { n } { \sqrt { n - T _ { \operatorname* { m a x } } } } \sqrt { 2 \zeta } T _ { \operatorname* { m a x } } ^ { 1 / 2 } s ^ { 1 / q } \rho \bar { \gamma } ( 1 + \sqrt { \frac { 2 \log ( 2 J / \delta _ { 2 } ) } { T _ { \operatorname* { m i n } } } } ) } } } \\ & { } & { \le \frac { \sigma n \sqrt { 1 + t } } { 2 ^ { - q } \frac { n } { \sqrt { n - T _ { \operatorname* { m a x } } } } \sqrt { 2 \zeta } T _ { \operatorname* { m a x } } ^ { 1 / 2 } s ^ { 1 / q } \rho \bar { \gamma } \sqrt { \frac { 2 \log ( 2 J / \delta _ { 2 } ) } { T _ { \operatorname* { m i n } } } } } \le \frac { 1 } { 2 } } \end{array}
$$

By Lemma II.1 in [23], we can obtain that

$$
\mathbb { P } ( \mathcal { B } ^ { c } ) \leq \mathbb { P } \left( V > \frac { n } { \sqrt { n - T _ { \operatorname* { m a x } } } } \sqrt { 2 \zeta } \left( 1 + \sqrt { \frac { 2 \log ( 2 J / \delta _ { 2 } ) } { T _ { \operatorname* { m i n } } } } \right) \right) \leq \delta _ { 2 }\tag{A.11}
$$

This concludes the proof.

Lemma A.5. In the proposed DS-IRLS-th algorithm, the group support set sequence $\{ \mathbb { G } ^ { k } \}$ and the diference active set sequence $\{ \mathbb { A } ^ { k } \}$ satisfies $\mathbb { G } ^ { k + 1 } \subseteq \mathbb { G } ^ { k }$ and $\mathbb { A } ^ { k + 1 } \subseteq \mathbb { A } ^ { k }$ for all $k \geq 0 ,$ , and thus both sequences converge. Suppose that the total number of groups 𝐽 is finite, then there exists $K \geq 0$ and ${ \mathbb { G } } ^ { * } , { \mathbb { A } } ^ { * } \subseteq \{ 1 , \dots , J \}$ such that for all $k \geq K$

$$
\mathbb { G } ^ { k } = \mathbb { G } ^ { * } , \quad \mathbb { A } ^ { k } = \mathbb { A } ^ { * } .\tag{A.12}
$$

Proof. Both $\{ \mathbb { G } ^ { k } \}$ and $\{ \mathbb { A } ^ { k } \}$ are non-increasing sequences $( \mathrm { i . e . , } \mathbb { G } ^ { k + 1 } \subseteq \mathbb { G } ^ { k } , \mathbb { A } ^ { k + 1 } \subseteq \mathbb { A } ^ { k } )$ , and the total number of groups 𝐽 is finite, thus they must converge to fixed sets within finite steps. 

## Appendix B Technical Proofs of Main Results

Proof of Proposition 1 By Lemma A.1 (iii) and (7), one has that

$$
\sum _ { j \in \mathbb { G } } T _ { j } ^ { q / 2 } \left[ \| \beta _ { j } ^ { * } \| _ { 2 } ^ { 2 } + \frac { 1 - \alpha } { \alpha } \| D _ { j } \beta _ { j } ^ { * } \| _ { 2 } ^ { 2 } \right] \leq T _ { \operatorname* { m a x } } ^ { q / 2 } s 2 ^ { 1 - q } \big ( 2 ^ { - 2 / q } \rho \big ) ^ { q } \leq 2 ^ { - ( q + 1 ) } s T _ { \operatorname* { m a x } } ^ { q / 2 } \rho ^ { q }\tag{B.1}
$$

Since $\hat { \beta }$ is a global solution of criterion (4), we have

$$
\frac { \lambda \alpha } { n } \sum _ { j = 1 } ^ { J } T _ { j } ^ { q / 2 } \| \hat { \beta } _ { j } \| _ { 2 } ^ { 2 } \leq \frac { 1 } { \sqrt { n } } \| y - X \beta ^ { * } \| _ { 2 } + \frac { \lambda \alpha } { n } \sum _ { j \in \mathbb { C } } T _ { j } ^ { q / 2 } \| \beta _ { j } ^ { * } \| _ { 2 } ^ { 2 } + \frac { \lambda ( 1 - \alpha ) } { n } \sum _ { j \in \mathbb { C } } T _ { j } ^ { q / 2 } \| D _ { j } \beta _ { j } ^ { * } \| _ { 2 } ^ { 2 }
$$

Since on $\mathcal { A }$ we have $\| \varepsilon \| _ { 2 } / { \sqrt { n } } \leq \sigma { \sqrt { 1 + t } }$ , then by (B.1), it follows that

$$
\begin{array} { l } { \displaystyle \sum _ { j = 1 } ^ { J } T _ { j } ^ { q / 2 } \| \hat { \beta } _ { j } \| _ { 2 } ^ { 2 } \leq \frac { n } { \sqrt { n } \lambda \alpha } \| y - X \beta ^ { * } \| _ { 2 } + \displaystyle \sum _ { j \in \mathbb { G } } T _ { j } ^ { q / 2 } \left[ \| \beta _ { j } ^ { * } \| _ { 2 } ^ { 2 } + \frac { 1 - \alpha } { \alpha } \| D _ { j } \beta _ { j } ^ { * } \| _ { 2 } ^ { 2 } \right] } \\ { \leq \frac { n } { \sqrt { n } \lambda \alpha } \| \varepsilon \| _ { 2 } + 2 ^ { - q } s T _ { \operatorname* { m a x } } ^ { q / 2 } \rho ^ { q } / 2 \leq \bar { \rho } ^ { q } } \end{array}
$$

where

$$
\bar { \rho } = \left( \sigma n \sqrt { 1 + t } / \lambda \alpha + 2 ^ { - q } s T _ { \mathrm { m a x } } ^ { q / 2 } \rho ^ { q } / 2 \right) ^ { 1 / q } .
$$

Note that $0 < q < 1$ and $T _ { j } \geq 1$ , then it follows from Lemma A.1 (i) that

$$
\begin{array} { r l } { \displaystyle \sum _ { j = 1 } ^ { J } \sqrt { T _ { j } } \| \hat { \beta } _ { j } - \beta _ { j } ^ { * } \| _ { 2 } \leq \displaystyle \sum _ { j = 1 } ^ { J } \sqrt { T _ { j } } \left[ \| \hat { \beta } _ { j } \| _ { 2 } + \| \beta _ { j } ^ { * } \| _ { 2 } \right] \leq \displaystyle \sum _ { j = 1 } ^ { J } \sqrt { T _ { j } } \| \hat { \beta } _ { j } \| _ { 2 } + \displaystyle \sum _ { j \in \mathfrak { S } } \sqrt { T _ { j } } \| \beta _ { j } ^ { * } \| _ { 2 } } & { } \\ { \leq \left[ \displaystyle \sum _ { j = 1 } ^ { J } T _ { j } ^ { q / 2 } \| \hat { \beta } _ { j } \| _ { 2 } ^ { q } \right] ^ { 1 / q } + \left[ \displaystyle \sum _ { j \in \mathfrak { S } } T _ { j } ^ { q / 2 } \| \beta _ { j } ^ { * } \| _ { 2 } ^ { q } \right] ^ { 1 / q } } & { } \\ { \leq \bar { \rho } + \left( 2 ^ { - q } s T _ { \operatorname* { m a x } } ^ { q / 2 } \rho ^ { q } / 2 \right) ^ { 1 / q } \leq 2 \bar { \rho } } \end{array}
$$

Therefore,

$$
\begin{array} { r l } { \displaystyle \sum _ { j = 1 } ^ { J } \sqrt { T _ { j } } \| \hat { \beta } _ { j } - \beta _ { j } ^ { * } \| _ { 2 } \leq \displaystyle \sum _ { j = 1 } ^ { J } T _ { j } ^ { ( 1 - q ) / 2 } \| \hat { \beta } _ { j } - \beta _ { j } ^ { * } \| _ { 2 } ^ { 1 - q } \cdot T _ { j } ^ { q / 2 } \| \hat { \beta } _ { j } - \beta _ { j } ^ { * } \| _ { 2 } ^ { q } } & { } \\ { \leq \displaystyle \operatorname* { m a x } _ { 1 \leq j \leq J } T _ { j } ^ { ( 1 - q ) / 2 } \| \hat { \beta } _ { j } - \beta _ { j } ^ { * } \| _ { 2 } ^ { 1 - q } \cdot \displaystyle \sum _ { j = 1 } ^ { J } T _ { j } ^ { q / 2 } \| \hat { \beta } _ { j } - \beta _ { j } ^ { * } \| _ { 2 } ^ { q } } & { } \\ { \leq ( 2 \bar { \rho } ) ^ { 1 - q } \displaystyle \sum _ { j = 1 } ^ { J } T _ { j } ^ { q / 2 } \| \hat { \beta } _ { j } - \beta _ { j } ^ { * } \| _ { 2 } ^ { q } } & { } \end{array}\tag{B.2}
$$

This completes the proof of (12). It remains to establish (13). Defining $\Delta _ { j } = \hat { \beta } _ { j } - \beta _ { j } ^ { * }$ for $1 \leq j \leq J _ { : }$ , it follows directly from the definition of $\hat { \beta }$ and Lemma A.1 (ii) that

$$
\begin{array} { l } { \displaystyle \frac { 1 } { \sqrt n } \left[ \| y - X \hat { \beta } \| _ { 2 } - \| y - X \beta ^ { * } \| _ { 2 } \right] } \\ { \displaystyle \quad \leq \frac { \lambda \alpha } { n } \sum _ { j = 1 } ^ { J } T _ { j } ^ { q / 2 } \left( \| \beta _ { j } ^ { * } \| _ { 2 } ^ { q } - \| \hat { \beta } _ { j } \| _ { 2 } ^ { q } \right) + \frac { \lambda ( 1 - \alpha ) } { n } \sum _ { j = 1 } ^ { J } T _ { j } ^ { q / 2 } \left( \| D _ { j } \beta _ { j } ^ { * } \| _ { 2 } ^ { q } - \| D _ { j } \hat { \beta } _ { j } \| _ { 2 } ^ { q } \right) } \\ { \displaystyle \quad \leq \frac { \lambda \alpha } { n } \sum _ { j \in \mathfrak { C } } T _ { j } ^ { q / 2 } \left[ \| \Delta _ { j } \| _ { 2 } ^ { q } + \frac { 1 - \alpha } { \alpha } \| D _ { j } \Delta _ { j } \| _ { 2 } ^ { q } \right] - \frac { \lambda \alpha } { n } \sum _ { j \in \mathfrak { C } } T _ { j } ^ { q / 2 } \left[ \| \Delta _ { j } \| _ { 2 } ^ { q } + \frac { 1 - \alpha } { \alpha } \| D _ { j } \Delta _ { j } \| _ { 2 } ^ { q } \right] } \end{array}\tag{B.3}
$$

Since

$$
\left. { \frac { \nabla \| y - X { \boldsymbol { \beta } } \| _ { 2 } } { \sqrt { n } } } \right| _ { \beta = \beta ^ { * } } = { \frac { - X ^ { T } \varepsilon } { { \sqrt { n } } \| \varepsilon \| _ { 2 } } }\tag{B.4}
$$

Then, on ${ \mathcal { A } } \cap { \mathcal { B } }$ , by convexity of the $\ell _ { 2 }$ norm and (B.2), we have

$$
\begin{array} { r l } { \displaystyle \frac { 1 } { \sqrt n } \left[ \| y - X \hat { \beta } \| _ { 2 } - \| y - X \beta ^ { * } \| _ { 2 } \right] \geq - \frac { | \varepsilon ^ { T } X \Delta | } { \sqrt n \| \varepsilon \| _ { 2 } } \geq - \displaystyle \sum _ { j = 1 } ^ { J } \frac { \| X _ { j } ^ { T } \varepsilon \| _ { 2 } \| \Delta _ { j } \| _ { 2 } } { \sqrt n \| \varepsilon \| _ { 2 } } } & { } \\ { \geq - ( 2 \bar { \rho } ) ^ { 1 - q } \displaystyle \frac { V } { n } \displaystyle \sum _ { j = 1 } ^ { J } \sqrt { T _ { j } } \| \Delta _ { j } \| _ { 2 } \geq - \frac { \lambda \alpha } { n \bar { \gamma } } \displaystyle \sum _ { j = 1 } ^ { J } T _ { j } ^ { q / 2 } \| \Delta _ { j } \| _ { 2 } ^ { q } } \end{array}\tag{B.5}
$$

Combining (B.3) and (B.5), we find

$$
\begin{array} { c } { \displaystyle - \frac { \lambda \alpha } { n \bar { \gamma } } \displaystyle \sum _ { j = 1 } ^ { J } { T _ { j } ^ { q / 2 } \| \Delta _ { j } \| _ { 2 } ^ { q } \leq \frac { \lambda \alpha } { n } \displaystyle \sum _ { j \in \mathbb { G } } T _ { j } ^ { q / 2 } \| \Delta _ { j } \| _ { 2 } ^ { q } + \frac { \lambda ( 1 - \alpha ) } { n } \displaystyle \sum _ { j \in \mathbb { G } } T _ { j } ^ { q / 2 } \| D _ { j } \Delta _ { j } \| _ { 2 } ^ { q } } } \\ { \displaystyle - \frac { \lambda \alpha } { n } \displaystyle \sum _ { j \in \mathbb { G } ^ { c } } T _ { j } ^ { q / 2 } \| \Delta _ { j } \| _ { 2 } ^ { q } - \frac { \lambda ( 1 - \alpha ) } { n } \displaystyle \sum _ { j \in \mathbb { G } ^ { c } } T _ { j } ^ { q / 2 } \| D _ { j } \Delta _ { j } \| _ { 2 } ^ { q } } \end{array}
$$

and thus

$$
\begin{array} { l } { \displaystyle \alpha \left( 1 - \frac { 1 } { \bar { \gamma } } \right) \sum _ { j \in \mathbb { G } ^ { c } } T _ { j } ^ { q / 2 } \| \Delta _ { j } \| _ { 2 } ^ { q } + ( 1 - \alpha ) \sum _ { j \in \mathbb { G } ^ { c } } T _ { j } ^ { q / 2 } \| D _ { j } \Delta _ { j } \| _ { 2 } ^ { q } } \\ { \displaystyle \leq \alpha \left( 1 + \frac { 1 } { \bar { \gamma } } \right) \sum _ { j \in \mathbb { G } } T _ { j } ^ { q / 2 } \| \Delta _ { j } \| _ { 2 } ^ { q } + ( 1 - \alpha ) \sum _ { j \in \mathbb { G } } T _ { j } ^ { q / 2 } \| D _ { j } \Delta _ { j } \| _ { 2 } ^ { q } } \end{array}
$$

Noting that $\begin{array} { r } { \bar { \gamma } = \frac { \gamma + 1 } { \gamma - 1 } } \end{array}$ and $\gamma > 1$ , we directly obtain the desired result:

$$
\alpha \sum _ { j \in \mathbb { C } ^ { c } } T _ { j } ^ { q / 2 } \| \Delta _ { j } \| _ { 2 } ^ { q } + ( 1 - \alpha ) \sum _ { j \in \mathbb { C } ^ { c } } T _ { j } ^ { q / 2 } \| D _ { j } \Delta _ { j } \| _ { 2 } ^ { q } \leq \gamma \left[ \alpha \sum _ { j \in \mathbb { C } } T _ { j } ^ { q / 2 } \| \Delta _ { j } \| _ { 2 } ^ { q } + ( 1 - \alpha ) \sum _ { j \in \mathbb { C } } T _ { j } ^ { q / 2 } \| D _ { j } \Delta _ { j } \| _ { 2 } ^ { q } \right] ,\tag{B.6}
$$

which completes the proof of this proposition.

Proofof Theorem 1 It follows from Lemma A.1 (i) and $( 2 , q )$ -GRCC that

$$
\sum _ { j \in \mathfrak { G } } T _ { j } ^ { q / 2 } \| \Delta _ { j } \| _ { 2 } ^ { q } \leq s ^ { 1 - q } \left( \sum _ { j \in \mathfrak { G } } \sqrt { T _ { j } } \| \Delta _ { j } \| _ { 2 } \right) ^ { q } \leq s ^ { 1 - q } \left( \frac { \sqrt { s ^ { * } } \| X \Delta \| _ { 2 } } { \sqrt { n } \kappa } \right) ^ { q } ,\tag{B.7}
$$

Since $\| D _ { j } \| _ { 2 } \leq 2$ , one has by (B.3) and (B.7) that

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { \sqrt n } \left[ \| y - X \hat { \beta } \| _ { 2 } - \| y - X \beta ^ { * } \| _ { 2 } \right] \leq \frac { \lambda \alpha } { n } \displaystyle \sum _ { j \in \mathbb { G } } T _ { j } ^ { q / 2 } \| \Delta _ { j } \| _ { 2 } ^ { q } + \frac { \lambda ( 1 - \alpha ) } { n } \displaystyle \sum _ { j \in \mathbb { G } } T _ { j } ^ { q / 2 } \| D _ { j } \Delta _ { j } \| _ { 2 } ^ { q } } \\ & { \quad \quad \quad \leq \frac { \lambda \alpha s ^ { 1 - q } } { n } \left( \displaystyle \sum _ { j \in \mathbb { G } } \sqrt { T _ { j } } \| \Delta _ { j } \| _ { 2 } \right) ^ { q } + \frac { \lambda ( 1 - \alpha ) 2 ^ { q } s ^ { 1 - q } } { n } \left( \displaystyle \sum _ { j \in \mathbb { G } } \sqrt { T _ { j } } \| \Delta _ { j } \| _ { 2 } \right) ^ { q } } \\ & { \quad \quad \quad \leq \frac { 2 \lambda s ^ { 1 - q } } { n } \left( \frac { \sqrt { s s } ^ { * } \| X \Delta \| _ { 2 } } { \sqrt { n } \kappa } \right) ^ { q } } \end{array}\tag{B.8}
$$

Besides

$$
\frac { \| y - X \hat { \beta } \| _ { 2 } ^ { 2 } } { n } - \frac { \| y - X \beta ^ { * } \| _ { 2 } ^ { 2 } } { n } = \frac { \| X \Delta - \varepsilon \| _ { 2 } ^ { 2 } } { n } - \frac { \| \varepsilon \| _ { 2 } ^ { 2 } } { n } = \frac { \| X \Delta \| _ { 2 } ^ { 2 } } { n } - \frac { 2 \varepsilon ^ { T } X \Delta } { n }\tag{B.9}
$$

Note that $\begin{array} { r } { \bar { \gamma } = \frac { \gamma + 1 } { \gamma - 1 } } \end{array}$ and $\gamma > 1$ , it follows from (B.5), (B.6) and (B.7) that

$$
\begin{array} { r l } & { \frac { \varepsilon ^ { T } X \Delta } { n } \leq \frac { \lambda \alpha } { n \widetilde { \gamma } } \frac { \| \varepsilon \| _ { 2 } } { \sqrt { n } } \displaystyle \sum _ { j = 1 } ^ { J } T _ { j } ^ { q / 2 } \| \Delta _ { j } \| _ { 2 } ^ { q } } \\ & { \qquad \leq \frac { \lambda } { n \widetilde { \gamma } } \frac { \| \varepsilon \| _ { 2 } } { \sqrt { n } } \left[ \alpha ( 1 + \gamma ) \displaystyle \sum _ { j \in \mathbb { Z } } T _ { j } ^ { q / 2 } \| \Delta _ { j } \| _ { 2 } ^ { q } + \gamma ( 1 - \alpha ) \displaystyle \sum _ { j \in \mathbb { G } } T _ { j } ^ { q / 2 } \| D _ { j } \Delta _ { j } \| _ { 2 } ^ { q } \right] } \\ & { \qquad \leq \frac { \lambda \| \varepsilon \| _ { 2 } } { n ^ { 3 / 2 } } \left( \frac { \alpha ( 1 + \gamma ) } { \widetilde { \gamma } } + \frac { 2 \gamma ( 1 - \alpha ) } { \widetilde { \gamma } } \right) \displaystyle \sum _ { j \in \mathbb { S } } T _ { j } ^ { q / 2 } \| \Delta _ { j } \| _ { 2 } ^ { q } } \\ & { \qquad \leq 2 \lambda \gamma \frac { \| \varepsilon \| _ { 2 } } { n ^ { 3 / 2 } } s ^ { 1 - q } \left( \frac { \sqrt { s } ^ { \ast } \| X \Delta \| _ { 2 } } { \sqrt { n } \kappa } \right) ^ { q } } \end{array}\tag{B.10}
$$

Combining (B.8), (B.9) and (B.10), we get

$$
\begin{array} { r l } & { \frac { \| X \Delta \| _ { 2 } ^ { 2 } } { n } = \frac { \| y - X \hat { \beta } \| _ { 2 } ^ { 2 } } { n } - \frac { \| y - X \beta ^ { * } \| _ { 2 } ^ { 2 } } { n } + \frac { 2 \varepsilon ^ { T } X \Delta } { n } } \\ & { \quad \quad = \left( \frac { \| y - X \hat { \beta } \| _ { 2 } } { \sqrt { n } } - \frac { \| y - X \beta ^ { * } \| _ { 2 } } { \sqrt { n } } \right) \left( \frac { \| y - X \hat { \beta } \| _ { 2 } } { \sqrt { n } } + \frac { \| y - X \beta ^ { * } \| _ { 2 } } { \sqrt { n } } \right) + \frac { 2 \varepsilon ^ { T } X \Delta } { n } } \\ & { \quad \quad \le \frac { 2 \lambda s ^ { 1 - q } } { n } \left( \frac { \sqrt { s ^ { * } } \| X \Delta \| _ { 2 } } { \sqrt { n } K } \right) ^ { q } \left[ 2 \frac { \| \varepsilon \| _ { 2 } } { \sqrt { n } } + \frac { 2 \lambda s ^ { 1 - q } } { n } \left( \frac { \sqrt { s ^ { * } } \| X \Delta \| _ { 2 } } { \sqrt { n } K } \right) ^ { q } \right] \quad + 4 \lambda \gamma \frac { \| \varepsilon \| _ { 2 } s ^ { 1 - q } } { n ^ { 3 / 2 } } \left( \frac { \sqrt { s ^ { * } } \| X \Delta \| _ { 2 } } { \sqrt { n } K } \right) ^ { q } } \\ & { \quad \quad \le 4 ( \gamma + 1 ) \frac { \| \varepsilon \| _ { 2 } } { \sqrt { n } } \frac { \lambda s ^ { 1 - q } s ^ { * q / 2 } } { n K ^ { q } } \left( \frac { \| X \Delta \| _ { 2 } } { \sqrt { n } } \right) ^ { q } + 4 \lambda ^ { 2 } \frac { s ^ { 2 ( 1 - q ) } s ^ { * q } } { n ^ { 2 } K ^ { 2 q } } \left( \frac { \| X \Delta \| _ { 2 } } { \sqrt { n } } \right) ^ { 2 q } } \end{array}
$$

By Lemma A.3 we have

$$
{ \frac { \| \varepsilon \| _ { 2 } } { \sqrt { n } } } \leq \sigma { \sqrt { 1 + t } }\tag{B.11}
$$

Then it is easy to obtain that

$$
\frac { \left\| X \Delta \right\| _ { 2 } } { \sqrt { n } } \lesssim \operatorname* { m a x } \left\{ \left( \sigma \frac { \lambda s ^ { 1 - q } s ^ { * q / 2 } } { n \kappa ^ { q } } \right) ^ { \frac { 1 } { 2 - q } } , \left( \frac { \lambda s ^ { 1 - q } s ^ { * q / 2 } } { n \kappa ^ { q } } \right) ^ { \frac { 1 } { 1 - q } } \right\}
$$

Invoking condition (11), the first statement of the theorem follows:

$$
{ \frac { \| X \Delta \| _ { 2 } } { \sqrt { n } } } \lesssim \left( \sigma { \frac { \lambda s ^ { 1 - q } s ^ { * q / 2 } } { n \kappa ^ { q } } } \right) ^ { \frac { 1 } { 2 - q } }\tag{B.12}
$$

For the second claim, it follows from Proposition 1 and (B.7) that

$$
\begin{array} { r l } { \displaystyle \sum _ { j = 1 } ^ { J } T _ { j } ^ { q / 2 } \| \Delta _ { j } \| _ { 2 } ^ { q } \leq ( \gamma + 1 ) \left[ \displaystyle \sum _ { j \in \mathbb { Q } } T _ { j } ^ { q / 2 } \| \Delta _ { j } \| _ { 2 } ^ { q } + \displaystyle \frac { ( 1 - \alpha ) } { \alpha } \displaystyle \sum _ { j \in \mathbb { Q } } T _ { j } ^ { q / 2 } \| D _ { j } \Delta _ { j } \| _ { 2 } ^ { q } \right] } & { } \\ { \lesssim s ^ { 1 - q } \left( \displaystyle \frac { \sqrt { s ^ { * } } \| X \Delta \| _ { 2 } } { \sqrt { n } \kappa } \right) ^ { q } \lesssim \frac { s ^ { 1 - q } s ^ { * q / 2 } } { \kappa ^ { q } } \cdot \left( \sigma \frac { \lambda s ^ { 1 - q } s ^ { * q / 2 } } { n \kappa ^ { q } } \right) ^ { \frac { q } { 2 - q } } } & { } \\ { \lesssim \left( \sigma \frac { \lambda s ^ { 2 ( 1 / q - 1 ) } s ^ { * } } { n \kappa ^ { 2 } } \right) ^ { \frac { q } { 2 - q } } } & { } \end{array}\tag{B.13}
$$

which concludes the proof of this theorem.

Proof of Proposition 2 Decompose the objective function as

$$
F ( \beta ) = f ( \beta ) + g ( \beta ) + h ( \beta ) ,\tag{B.14}
$$

where

$$
f ( { \boldsymbol { \beta } } ) = \| { \boldsymbol { y } } - X { \boldsymbol { \beta } } \| _ { 2 } , \quad { \boldsymbol { g } } ( { \boldsymbol { \beta } } ) = \lambda _ { 1 } \sum _ { j = 1 } ^ { J } T _ { j } ^ { q / 2 } \| { \boldsymbol { \beta } } _ { j } \| _ { 2 } ^ { q } , \quad { \boldsymbol { h } } ( { \boldsymbol { \beta } } ) = \lambda _ { 2 } \sum _ { j = 1 } ^ { J } T _ { j } ^ { q / 2 } \| { \boldsymbol { D } } _ { j } { \boldsymbol { \beta } } _ { j } \| _ { 2 } ^ { q } .
$$

Since $y \ne X \beta$ , the function $f$ is continuously diferentiable at $\beta ,$ with gradient

$$
\nabla f ( \beta ) = - \frac { X ^ { \top } ( y - X \beta ) } { \| y - X \beta \| _ { 2 } } .
$$

Accordingly, the 𝑗-th component of $\nabla f ( \beta )$ is

$$
\left( \nabla f ( \beta ) \right) _ { j } = - { \frac { x _ { j } ^ { \top } ( y - X \beta ) } { \| y - X \beta \| _ { 2 } } } .\tag{B.15}
$$

For each group 𝑗, define $\phi _ { j } ( \beta ) = \| \beta _ { j } \| _ { 2 } ^ { q }$ . By [6, Lemma 3.1], the subdiferential of $\phi _ { j }$ is

$$
\partial \phi _ { j } ( \beta ) = \left\{ \begin{array} { l l } { q \Vert \beta _ { j } \Vert _ { 2 } ^ { q - 2 } \beta _ { j } , } & { \beta _ { j } \neq 0 , } \\ { ( - \infty , + \infty ) ^ { T _ { j } } , } & { \beta _ { j } = 0 . } \end{array} \right.\tag{B.16}
$$

Thus, the contribution of 𝑔 to the 𝑗-th group is $\lambda _ { 1 } T _ { i } ^ { q / 2 } \partial \phi _ { j } ( \beta )$

Next, define $\psi _ { j } ( \beta ) = \| D _ { j } \beta _ { j } \| _ { 2 } ^ { q }$ . If $D _ { j } \beta _ { j } \neq 0 , \psi _ { j }$ is diferentiable by the chain rule, and its gradient is

$$
\begin{array} { r } { \nabla \psi _ { j } ( { \boldsymbol { \beta } } ) = q \| D _ { j } \beta _ { j } \| _ { 2 } ^ { q - 2 } D _ { j } ^ { \top } D _ { j } \beta _ { j } . } \end{array}\tag{B.17}
$$

If $D _ { j } \beta _ { j } = 0$ , the same argument as in [6, Lemma 3.1] applied to the composition $y \mapsto \| y \| _ { 2 } ^ { q }$ with the linear map $D _ { j }$ yields $\partial \psi _ { j } ( \beta ) = ( - \infty , + \infty ) ^ { T _ { j } }$

By [6, Lemma 3.2], $f , g .$ , and ℎ are all regular functions. From [39, Proposition 10.5] for separable functions, the sum function is also regular and

$$
\partial F ( \beta ) = \partial f ( \beta ) + \partial g ( \beta ) + \partial h ( \beta ) .\tag{B.18}
$$

Consequently, for a fixed group $j ,$ if $\beta _ { j } \neq 0$ and $D _ { j } \beta _ { j } \neq 0$ , each of $\partial f ( \beta ) , \partial g ( \beta )$ , and $\partial h ( \beta )$ is a singleton. Their sum is the vector given in (18). Otherwise, at least one of the subdiferentials $\partial \phi _ { j } ( \beta )$ or $\partial \psi _ { j } ( \beta )$ equals $\mathbb { R } ^ { T _ { j } }$ , so the sum $\left( \partial { \cal F } ( \beta ) \right) _ { j }$ is also $\mathbb { R } ^ { T _ { j } }$ . This completes the proof.

Proof of Lemma 1 (i) Let $k \geq 1$ . By Lemma $\mathrm { A . 5 } ,$ , we have $\mathbb { G } ^ { k + 1 } \subseteq \mathbb { G } ^ { k }$ and $\mathbb { A } ^ { k + 1 } \subseteq \mathbb { A } ^ { k }$ . When $\mathbb { G } ^ { k + 1 } = \mathbb { G } ^ { k }$ , we have $\beta ^ { k + 1 } = \tilde { \beta } ^ { k + 1 }$ , and by the first-order upper bound condition, we have

$$
\| y - X \beta \| _ { 2 } \leq \frac { 1 } { 2 r _ { k } } \| y - X \beta \| _ { 2 } ^ { 2 } + \frac { r _ { k } } { 2 } ,
$$

where $r _ { k } = \| y - X \beta ^ { k } \| _ { 2 }$ . Then we can introduce a majorization function

$$
M ( \beta \mid \beta ^ { k } ) = \frac { 1 } { 2 r _ { k } } \| y - X \beta \| _ { 2 } ^ { 2 } + \frac { r _ { k } } { 2 } + \lambda _ { 1 } \sum _ { j \in \Xi ^ { k } } T _ { j } ^ { q / 2 } \| \beta _ { j } \| _ { 2 } ^ { q } + \lambda _ { 2 } \sum _ { j \in \Xi ^ { k } } T _ { j } ^ { q / 2 } \| D _ { j } \beta _ { j } \| _ { 2 } ^ { q } .\tag{B.19}
$$

Then we have $F ( \beta ^ { k + 1 } ) \leq M ( \beta ^ { k + 1 } \mid \beta ^ { k } )$

From the constrained optimization problem (21), it follows that for $k \geq 1 , \beta _ { j } ^ { k } = 0 \mathrm { f o r } j \notin \mathbb { G } ^ { k } , D _ { j } \beta _ { j } ^ { k } \neq 0$

for $j \in \mathbb { A } ^ { k - 1 }$ , and thus

$$
\begin{array} { r l } & { \displaystyle F ( { \boldsymbol { \beta } } ^ { k } ) - M ( { \boldsymbol { \beta } } ^ { k + 1 } \mid { \boldsymbol { \beta } } ^ { k } ) = \frac { r _ { k } } { 2 } - \frac { 1 } { 2 r _ { k } } \| y - X { \boldsymbol { \beta } } ^ { k + 1 } \| _ { 2 } ^ { 2 } + \lambda _ { 1 } \displaystyle \sum _ { j \in \mathbb { S } ^ { k } } T _ { j } ^ { q / 2 } \left[ \| { \boldsymbol { \beta } } _ { j } ^ { k } \| _ { 2 } ^ { q } - \| { \boldsymbol { \beta } } _ { j } ^ { k + 1 } \| _ { 2 } ^ { q } \right] } \\ & { \qquad + \lambda _ { 2 } \displaystyle \sum _ { j \in \mathbb { S } ^ { k - 1 } } T _ { j } ^ { q / 2 } \| { \boldsymbol { D } } _ { j } { \boldsymbol { \beta } } _ { j } ^ { k } \| _ { 2 } ^ { q } - \lambda _ { 2 } \displaystyle \sum _ { j \in \mathbb { A } ^ { k } } T _ { j } ^ { q / 2 } \| { \boldsymbol { D } } _ { j } { \boldsymbol { \beta } } _ { j } ^ { k + 1 } \| _ { 2 } ^ { q } } \\ & { \displaystyle \geq \frac { r _ { k } } { 2 } - \frac { 1 } { 2 } - \frac { 1 } { 2 r _ { k } } \| y - X { \boldsymbol { \beta } } ^ { k + 1 } \| _ { 2 } ^ { 2 } + \lambda _ { 1 } \displaystyle \sum _ { j \in \mathbb { S } ^ { k } } T _ { j } ^ { q / 2 } \left[ \| { \boldsymbol { \beta } } _ { j } ^ { k } \| _ { 2 } ^ { q } - \| { \boldsymbol { \beta } } _ { j } ^ { k + 1 } \| _ { 2 } ^ { q } \right] } \\ &  \qquad + \lambda _ { 2 } \displaystyle \sum _ { j \in \mathbb { S } ^ { k - 1 } \backslash j ^ { k } } T _ { j } ^ { q / 2 } \| { \boldsymbol { D } } _ { j } { \boldsymbol { \beta } } _ { j } ^ { k } \| _ { 2 } ^ { q } + \lambda _ { 2 } \displaystyle \sum _  j \in \mathbb { S } ^  \end{array}\tag{B.20}
$$

Noting that

$$
\| y - X \beta ^ { k + 1 } \| _ { 2 } ^ { 2 } = \| y - X \beta ^ { k } \| _ { 2 } ^ { 2 } + \| X ( \beta ^ { k + 1 } - \beta ^ { k } ) \| _ { 2 } ^ { 2 } - 2 ( y - X \beta ^ { k } ) ^ { \top } X ( \beta ^ { k + 1 } - \beta ^ { k } ) ,
$$

we have

$$
\frac { r _ { k } } { 2 } - \frac { 1 } { 2 r _ { k } } \| y - X { \beta } ^ { k + 1 } \| _ { 2 } ^ { 2 } = - \frac { 1 } { 2 r _ { k } } \| X ( { \beta } ^ { k + 1 } - { \beta } ^ { k } ) \| _ { 2 } ^ { 2 } + \frac { 1 } { r _ { k } } ( y - X { \beta } ^ { k } ) ^ { \top } X ( { \beta } ^ { k + 1 } - { \beta } ^ { k } ) .\tag{B.21}
$$

Since $D _ { j } \beta _ { j } ^ { k + 1 } = 0$ for all $j \notin \mathbb { G } ^ { k } \setminus \mathbb { A } ^ { k }$ , by the KKT conditions of the constrained optimization problem (21), we have

$$
\frac { 1 } { r _ { k } } X _ { j } ^ { \top } ( X \beta ^ { k + 1 } - y ) + \lambda _ { 1 } T _ { j } ^ { q / 2 } w _ { 1 , j } ^ { k } \beta _ { j } ^ { k + 1 } + \lambda _ { 2 } T _ { j } ^ { q / 2 } w _ { 2 , j } ^ { k } D _ { j } ^ { \top } D _ { j } \beta _ { j } ^ { k + 1 } = 0 ,
$$

for all $j \in \mathbb { A } ^ { k }$ , and

$$
\frac { 1 } { r _ { k } } X _ { j } ^ { \top } ( X \beta ^ { k + 1 } - y ) + \lambda _ { 1 } T _ { j } ^ { q / 2 } w _ { 1 , j } ^ { k } \beta _ { j } ^ { k + 1 } + D _ { j } ^ { \top } \mu _ { j } = 0 ,
$$

for $j \in \mathbb { G } ^ { k } \setminus \mathbb { A } ^ { k }$ , where $w _ { 1 , j } ^ { k } = q \| \beta _ { j } ^ { k } \| _ { 2 } ^ { q - 2 } , w _ { 2 , j } ^ { k } = q \| D _ { j } \beta _ { j } ^ { k } \| _ { 2 } ^ { q - 2 }$ , and $\mu _ { j }$ is the Lagrange multiplier for the equality constraint of the 𝑗-th group. From Section 3.2, it follows that $D _ { j } \beta _ { j } ^ { k } = 0$ for all $j \notin \mathbb { A } ^ { k - 1 }$ and $D _ { j } \beta _ { j } ^ { k + 1 } = 0$ for all $j \notin \mathbb { A } ^ { k }$ . Taking the inner product with $( \beta _ { j } ^ { k + 1 } - \beta _ { j } ^ { k } )$ yields

$$
\begin{array} { r l } & { ( \forall j \in \mathbb { A } ^ { k } , \qquad 0 = \frac { 1 } { r _ { k } } ( \beta _ { j } ^ { k + 1 } - \beta _ { j } ^ { k } ) ^ { \top } X _ { j } ^ { \top } ( X \beta ^ { k + 1 } - y ) + \lambda _ { 1 } T _ { j } ^ { q / 2 } w _ { 1 , j } ^ { k } ( \beta _ { j } ^ { k + 1 } - \beta _ { j } ^ { k } ) ^ { \top } \beta _ { j } ^ { k + 1 }  } \\ & { \qquad + \lambda _ { 2 } T _ { j } ^ { q / 2 } w _ { 2 , j } ^ { k } ( \beta _ { j } ^ { k + 1 } - \beta _ { j } ^ { k } ) ^ { \top } D _ { j } ^ { \top } D _ { j } \beta _ { j } ^ { k + 1 } , \qquad } \\ { \forall j \in \mathbb { G } ^ { k } \setminus \mathbb { A } ^ { k - 1 } , \quad 0 = \frac { 1 } { r _ { k } } ( \beta _ { j } ^ { k + 1 } - \beta _ { j } ^ { k } ) ^ { \top } X _ { j } ^ { \top } ( X \beta ^ { k + 1 } - y ) + \lambda _ { 1 } T _ { j } ^ { q / 2 } w _ { 1 , j } ^ { k } ( \beta _ { j } ^ { k + 1 } - \beta _ { j } ^ { k } ) ^ { \top } \beta _ { j } ^ { k + 1 } , \qquad } \\ { \forall j \in \mathbb { A } ^ { k - 1 } \setminus \mathbb { A } ^ { k } , \quad 0 = \frac { 1 } { r _ { k } } ( \beta _ { j } ^ { k + 1 } - \beta _ { j } ^ { k } ) ^ { \top } X _ { j } ^ { \top } ( X \beta ^ { k + 1 } - y ) } \\ & { \qquad + \lambda _ { 1 } T _ { j } ^ { q / 2 } w _ { 1 , j } ^ { k } ( \beta _ { j } ^ { k + 1 } - \beta _ { j } ^ { k } ) ^ { \top } \beta _ { j } ^ { k + 1 } - \mu _ { j } ^ { \top } D _ { j } \beta _ { j } ^ { k } . } \end{array}
$$

Set $S ^ { k } = \cup _ { j \in \mathbb { G } ^ { k } } { \mathcal { G } } _ { j }$ . Since $\beta _ { j } ^ { k + 1 } = \beta _ { j } ^ { k } = 0$ for all $j \notin \mathbb { G } ^ { k }$ , we have ${ X } _ { S ^ { k } } ( \beta _ { S ^ { k } } ^ { k + 1 } - \beta _ { S ^ { k } } ^ { k } ) = { X } ( \beta ^ { k + 1 } - \beta ^ { k } )$ . Then

summing the above equalities over all $j ,$ we obtain

$$
\begin{array} { r l r } & { } & { 0 = \displaystyle \frac { 1 } { r _ { k } } ( X { \beta } ^ { k + 1 } - y ) ^ { \top } X ( { \beta } ^ { k + 1 } - { \beta } ^ { k } ) + \lambda _ { 1 } q \displaystyle \sum _ { j \in \mathbb { G } ^ { k } } T _ { j } ^ { q / 2 } \| { \beta } _ { j } ^ { k } \| _ { 2 } ^ { q - 2 } ( { \beta } _ { j } ^ { k + 1 } ) ^ { \top } ( { \beta } _ { j } ^ { k + 1 } - { \beta } _ { j } ^ { k } ) } \\ & { } & { + \lambda _ { 2 } q \displaystyle \sum _ { j \in \mathbb { A } ^ { k } } T _ { j } ^ { q / 2 } \| D _ { j } { \beta } _ { j } ^ { k } \| _ { 2 } ^ { q - 2 } ( D _ { j } { \beta } _ { j } ^ { k + 1 } ) ^ { \top } D _ { j } ( { \beta } _ { j } ^ { k + 1 } - { \beta } _ { j } ^ { k } ) - \displaystyle \sum _ { j \in \mathbb { A } ^ { k - 1 } \backslash \mathbb { A } ^ { k } } \mu _ { j } ^ { \top } D _ { j } { \beta } _ { j } ^ { k } . } \end{array}\tag{B.22}
$$

From (B.22), we have

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { r _ { k } } ( \boldsymbol { y } - \boldsymbol { X } { \beta } ^ { k } ) ^ { \top } \boldsymbol { X } ( { \beta } ^ { k + 1 } - { \beta } ^ { k } ) } \\ & { = \displaystyle \frac { 1 } { r _ { k } } ( \boldsymbol { y } - \boldsymbol { X } { \beta } ^ { k + 1 } ) ^ { \top } \boldsymbol { X } ( { \beta } ^ { k + 1 } - { \beta } ^ { k } ) + \displaystyle \frac { 1 } { r _ { k } } \| \boldsymbol { X } ( { \beta } ^ { k + 1 } - { \beta } ^ { k } ) \| _ { 2 } ^ { 2 } } \\ & { = \displaystyle \frac { 1 } { r _ { k } } \| \boldsymbol { X } ( { \beta } ^ { k + 1 } - { \beta } ^ { k } ) \| _ { 2 } ^ { 2 } + \lambda _ { 1 } q \displaystyle \sum _ { j \in \mathbb { G } ^ { k } } T _ { j } ^ { q / 2 } \| { \beta } _ { j } ^ { k } \| _ { 2 } ^ { q - 2 } ( { \beta } _ { j } ^ { k + 1 } ) ^ { \top } ( { \beta } _ { j } ^ { k + 1 } - { \beta } _ { j } ^ { k } ) } \\ & { \quad + \lambda _ { 2 } q \displaystyle \sum _ { j \in \mathbb { A } ^ { k } } T _ { j } ^ { q / 2 } \| D _ { j } { \beta } _ { j } ^ { k } \| _ { 2 } ^ { q - 2 } ( D _ { j } { \beta } _ { j } ^ { k + 1 } ) ^ { \top } D _ { j } ( { \beta } _ { j } ^ { k + 1 } - { \beta } _ { j } ^ { k } ) - \displaystyle \sum _ { j \in \mathbb { A } ^ { k - 1 } \backslash A ^ { k } } \mu _ { j } ^ { \top } D _ { j } { \beta } _ { j } ^ { k } . } \end{array}\tag{B.23}
$$

It follows from Lemma $\mathrm { A } . 2$ that

$$
\| { \boldsymbol { \beta } } _ { j } ^ { k } \| _ { 2 } ^ { q } - \| { \boldsymbol { \beta } } _ { j } ^ { k + 1 } \| _ { 2 } ^ { q } - q \| { \boldsymbol { \beta } } _ { j } ^ { k } \| _ { 2 } ^ { q - 2 } ( { \boldsymbol { \beta } } _ { j } ^ { k } - { \boldsymbol { \beta } } _ { j } ^ { k + 1 } ) ^ { \top } { \boldsymbol { \beta } } _ { j } ^ { k + 1 } \geq \frac { q } { 2 } \| { \boldsymbol { \beta } } _ { j } ^ { k } \| _ { 2 } ^ { q - 2 } \| { \boldsymbol { \beta } } _ { j } ^ { k + 1 } - { \boldsymbol { \beta } } _ { j } ^ { k } \| _ { 2 } ^ { 2 } ,
$$

for $j \in \mathbb { G } ^ { k }$ , and

$$
\| D _ { j } \beta _ { j } ^ { k } \| _ { 2 } ^ { q } - \| D _ { j } \beta _ { j } ^ { k + 1 } \| _ { 2 } ^ { q } - q \| D _ { j } \beta _ { j } ^ { k } \| _ { 2 } ^ { q - 2 } ( D _ { j } \beta _ { j } ^ { k } - D _ { j } \beta _ { j } ^ { k + 1 } ) ^ { \top } D _ { j } \beta _ { j } ^ { k + 1 } \geq \frac { q } { 2 } \| D _ { j } \beta _ { j } ^ { k } \| _ { 2 } ^ { q - 2 } \| D _ { j } ( \beta _ { j } ^ { k + 1 } - \beta _ { j } ^ { k } ) \| _ { 2 } ^ { 2 } ,
$$

for $j \in \mathbb { A } ^ { k }$ . Then we have

$$
\begin{array} { r l r } & { } & { \displaystyle \sum _ { j \in \mathbb G ^ { k } } { \cal T } _ { j } ^ { q / 2 } \left[ \| \beta _ { j } ^ { k } \| _ { 2 } ^ { q } - \| \beta _ { j } ^ { k + 1 } \| _ { 2 } ^ { q } \right] \geq q \displaystyle \sum _ { j \in \mathbb G ^ { k } } { \cal T } _ { j } ^ { q / 2 } \| \beta _ { j } ^ { k } \| _ { 2 } ^ { q - 2 } ( \beta _ { j } ^ { k } - \beta _ { j } ^ { k + 1 } ) ^ { \top } \beta _ { j } ^ { k + 1 } } \\ & { } & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad + \displaystyle \frac { q } { 2 } \displaystyle \sum _ { j \in \mathbb G ^ { k } } { \cal T } _ { j } ^ { q / 2 } \| \beta _ { j } ^ { k } \| _ { 2 } ^ { q - 2 } \| \beta _ { j } ^ { k + 1 } - \beta _ { j } ^ { k } \| _ { 2 } ^ { 2 } , } \end{array}\tag{B.24}
$$

$$
\begin{array} { r l } & { \displaystyle \sum _ { j \in \mathbb { A } ^ { k } } { \cal T } _ { j } ^ { q / 2 } \left[ \| D _ { j } \beta _ { j } ^ { k } \| _ { 2 } ^ { q } - \| D _ { j } \beta _ { j } ^ { k + 1 } \| _ { 2 } ^ { q } \right] \geq q \sum _ { j \in \mathbb { A } ^ { k } } { \cal T } _ { j } ^ { q / 2 } \| D _ { j } \beta _ { j } ^ { k } \| _ { 2 } ^ { q - 2 } ( D _ { j } \beta _ { j } ^ { k } - D _ { j } \beta _ { j } ^ { k + 1 } ) ^ { \top } D _ { j } \beta _ { j } ^ { k + 1 } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad + \frac { q } { 2 } \displaystyle \sum _ { j \in \mathbb { A } ^ { k } } { \cal T } _ { j } ^ { q / 2 } \| D _ { j } \beta _ { j } ^ { k } \| _ { 2 } ^ { q - 2 } \| D _ { j } ( \beta _ { j } ^ { k + 1 } - \beta _ { j } ^ { k } ) \| _ { 2 } ^ { 2 } . } \end{array}\tag{B.25}
$$

Substituting (B.21), (B.23), (B.24), and (B.25) into (B.20) yields

$$
\begin{array} { l } { { \displaystyle F ( { \boldsymbol { \beta } } ^ { k } ) - M ( { \boldsymbol { \beta } } ^ { k + 1 } \mid { \boldsymbol { \beta } } ^ { k } ) = - \frac { 1 } { 2 r _ { k } } \| { \boldsymbol { X } } ( { \boldsymbol { \beta } } ^ { k + 1 } - { \boldsymbol { \beta } } ^ { k } ) \| _ { 2 } ^ { 2 } + \frac { 1 } { r _ { k } } \| { \boldsymbol { X } } ( { \boldsymbol { \beta } } ^ { k + 1 } - { \boldsymbol { \beta } } ^ { k } ) \| _ { 2 } ^ { 2 } } } \\ { { \displaystyle \phantom { \frac { 1 } { 2 r _ { k } } } + \frac { \lambda _ { 1 } q } { 2 } \sum _ { j \in \mathbb { Z } ^ { k } } T _ { j } ^ { q / 2 } \| { \boldsymbol { \beta } } _ { j } ^ { k } \| _ { 2 } ^ { q - 2 } \| { \boldsymbol { \beta } } _ { j } ^ { k + 1 } - { \boldsymbol { \beta } } _ { j } ^ { k } \| _ { 2 } ^ { 2 } } } \\ { { \displaystyle \phantom { \frac { 1 } { 2 r _ { k } } } + \frac { \lambda _ { 2 } q } { 2 } \sum _ { j \in \mathbb { R } ^ { k } } T _ { j } ^ { q / 2 } \| { \boldsymbol { D } } _ { j } { \boldsymbol { \beta } } _ { j } ^ { k } \| _ { 2 } ^ { q - 2 } \| { \boldsymbol { D } } _ { j } ( { \boldsymbol { \beta } } _ { j } ^ { k + 1 } - { \boldsymbol { \beta } } _ { j } ^ { k } ) \| _ { 2 } ^ { 2 } } } \\   \displaystyle \phantom { \frac { 1 } { 2 r _ { k } } } + \lambda _ { 2 } \sum _ { j \in \mathbb { R } ^ { k - 1 } \backslash j _ { k } ^ { k } } T _ { j } ^ { q / 2 } \| { \boldsymbol { D } } _ { j } { \boldsymbol { \beta } } _ { j } ^ { k } \| _ { 2 } ^ { q } - \sum _ { j \in \mathbb { R } ^ { k - 1 } \backslash j _ { k } ^ { k } } \mu _ { j } ^ { \top } { \boldsymbol { D } } _  \end{array}\tag{B.26}
$$

For the constrained optimization problem (21), two key conditions hold at the KKT point: the Hessian of the Lagrangian is positive definite, and the gradients of the linear equality constraints are linearly independent $( \mathrm { i . e . }$ $\sigma _ { \operatorname* { m i n } } ( D _ { j } ^ { \top } ) > 0$ for each group 𝑗). By Chapter 10 in [43], the Lagrange multiplier $\mu _ { j }$ is unique. Meanwhile, by Theorem 4.3 in [44], $\mu _ { j }$ is bounded with the explicit upper bound

$$
\| \mu _ { j } \| _ { 2 } \leq \frac { l _ { f } } { \sigma _ { \operatorname* { m i n } } ( D _ { j } ^ { \top } ) } ,
$$

where $l _ { f }$ is the local Lipschitz constant of the objective function $F ( \beta )$ , and $\sigma _ { \mathrm { m i n } } ( D _ { j } ^ { \top } )$ is the minimum singular value of the matrix $D _ { j } ^ { \top }$

Recall that for $j \in \mathbb { A } ^ { k - 1 } \backslash \mathbb { A } ^ { k }$ , we have $0 < \| D _ { j } \beta _ { j } ^ { k } \| _ { 2 } < \tau _ { 2 }$ . By choosing $\tau _ { 2 } \leq \tilde { \tau } _ { 2 } = \left( \lambda _ { 2 } \sigma _ { \operatorname* { m i n } } ( D _ { j } ^ { \top } ) T _ { \operatorname* { m i n } } ^ { q / 2 } / l _ { f } \right) ^ { \frac { 1 } { 1 - q } }$ we have

$$
F ( \boldsymbol { \beta } ^ { k } ) - M ( \boldsymbol { \beta } ^ { k + 1 } \mid \boldsymbol { \beta } ^ { k } ) \geq \frac { \lambda _ { 1 } q } { 2 } \sum _ { j \in \mathbb { G } ^ { k } } T _ { j } ^ { q / 2 } \| \boldsymbol { \beta } _ { j } ^ { k } \| _ { 2 } ^ { q - 2 } \| \boldsymbol { \beta } _ { j } ^ { k + 1 } - \boldsymbol { \beta } _ { j } ^ { k } \| _ { 2 } ^ { 2 } .\tag{B.27}
$$

Since $F ( \beta ^ { k + 1 } ) \leq M ( \beta ^ { k + 1 } \mid \beta ^ { k } )$ , we have

$$
F ( \beta ^ { k } ) - F ( \beta ^ { k + 1 } ) \ge F ( \beta ^ { k } ) - M ( \beta ^ { k + 1 } \mid \beta ^ { k } ) \ge \frac { \lambda _ { 1 } q } { 2 } \sum _ { j \in \mathbb { S } ^ { k } } T _ { j } ^ { q / 2 } \| \beta _ { j } ^ { k } \| _ { 2 } ^ { q - 2 } \| \beta _ { j } ^ { k + 1 } - \beta _ { j } ^ { k } \| _ { 2 } ^ { 2 } \ge 0 .\tag{B.28}
$$

When $\mathbb { G } ^ { k + 1 } \subsetneq \mathbb { G } ^ { k }$ , we denote the truncated residual as $e ^ { k + 1 } = { \tilde { \beta } } ^ { k + 1 } - \beta ^ { k + 1 }$ , where

$$
e _ { j } ^ { k + 1 } = \left\{ \begin{array} { l l } { \tilde { \beta } _ { j } ^ { k + 1 } , } & { j \in \mathbb { G } ^ { k } \setminus \mathbb { G } ^ { k + 1 } , } \\ { 0 , } & { j \not \in \mathbb { G } ^ { k } \setminus \mathbb { G } ^ { k + 1 } . } \end{array} \right.
$$

Since

$$
F ( \beta ^ { k } ) - F ( \beta ^ { k + 1 } ) = \left[ F ( \beta ^ { k } ) - F ( \tilde { \beta } ^ { k + 1 } ) \right] - \left[ F ( \beta ^ { k + 1 } ) - F ( \tilde { \beta } ^ { k + 1 } ) \right] .\tag{B.29}
$$

Similar to (B.28), one has

$$
F ( { \boldsymbol { \beta } } ^ { k } ) - F ( \widetilde { \boldsymbol { \beta } } ^ { k + 1 } ) \geq \frac { \lambda _ { 1 } q } { 2 } \sum _ { j \in \mathbb { G } ^ { k } } T _ { j } ^ { q / 2 } \| { \boldsymbol { \beta } } _ { j } ^ { k } \| _ { 2 } ^ { q - 2 } \| \widetilde { \boldsymbol { \beta } } _ { j } ^ { k + 1 } - { \boldsymbol { \beta } } _ { j } ^ { k } \| _ { 2 } ^ { 2 } \geq 0 .\tag{B.30}
$$

Meanwhile, it is easy to obtain that

$$
\begin{array} { r l } & { F ( \beta ^ { k + 1 } ) - F ( \widetilde { \beta } ^ { k + 1 } ) \le \parallel y - X \beta ^ { k + 1 } \parallel _ { 2 } - \parallel y - X \widetilde { \beta } ^ { k + 1 } \parallel _ { 2 } } \\ & { \qquad - \lambda _ { 1 } \underset { j \in \mathbb { G } ^ { k } \backslash \left( \mathbb { G } ^ { k + 1 } \right. } { \sum } T _ { j } ^ { q / 2 } \Vert \widetilde { \beta } _ { j } ^ { k + 1 } \parallel _ { 2 } ^ { q } - \lambda _ { 2 } \underset { j \in \mathbb { G } ^ { k } \backslash \left( \mathbb { G } ^ { k + 1 } \right. } { \sum } T _ { j } ^ { q / 2 } \Vert D _ { j } \widetilde { \beta } _ { j } ^ { k + 1 } \Vert _ { 2 } ^ { q } . } \end{array}
$$

By the triangle inequality, we have

$$
\| y - X \beta ^ { k + 1 } \| _ { 2 } - \| y - X \widetilde { \beta } ^ { k + 1 } \| _ { 2 } \leq \| X e ^ { k + 1 } \| _ { 2 } \leq \| X \| _ { 2 } \| e ^ { k + 1 } \| _ { 2 } = \| X \| _ { 2 } \sqrt { \sum _ { j \in \mathbb { Q } ^ { k } \backslash ( \mathbb { Q } ^ { k + 1 } } \| e _ { j } ^ { k + 1 } \| _ { 2 } ^ { 2 } } .
$$

Since $\| e _ { j } ^ { k + 1 } \| _ { 2 } = \| \tilde { \beta } _ { j } ^ { k + 1 } \| _ { 2 } < \tau _ { 1 }$ for all $j \in \mathbb { G } ^ { k } \setminus \mathbb { G } ^ { k + 1 }$ , we get

$$
\begin{array} { r } { F ( \beta ^ { k + 1 } ) - F ( \widetilde { \beta } ^ { k + 1 } ) \leq \| X \| _ { 2 } \sqrt { | \mathbb { G } ^ { k } \setminus \mathbb { G } ^ { k + 1 } | \tau _ { 1 } ^ { 2 } } \leq | \mathbb { G } ^ { k } \setminus \mathbb { G } ^ { k + 1 } | \| X \| _ { 2 } \tau _ { 1 } , } \end{array}\tag{B.31}
$$

where $| \mathbb { G } ^ { k } \setminus \mathbb { G } ^ { k + 1 } |$ is the cardinality of the set $\mathbb { G } ^ { k } \setminus \mathbb { G } ^ { k + 1 }$ . Then for $k \geq 1$ , it follows that $F ( \beta ^ { k + 1 } ) - F ( \beta ^ { k } ) \leq 0$ when $\mathbb { G } ^ { k + 1 } = \mathbb { G } ^ { k }$ and $F ( \beta ^ { k + 1 } ) - F ( \beta ^ { k } ) \leq | { \mathbb G } ^ { k } \ \backslash \ { \mathbb G } ^ { k + 1 } | \| X \| _ { 2 } \tau _ { 1 }$ when $\mathbb { G } ^ { k + 1 } \subsetneq \mathbb { G } ^ { k }$ . This result clearly extends to the initial iteration $k = 0$ . Thus we have

$$
F ( { \boldsymbol { \beta } } ^ { k + 1 } ) - F ( { \boldsymbol { \beta } } ^ { 0 } ) \leq \sum _ { i = 0 } ^ { k } | { \mathbb G } ^ { i } \setminus { \mathbb G } ^ { i + 1 } | \| X \| _ { 2 } \tau _ { 1 } \leq J \| X \| _ { 2 } \tau _ { 1 } ,
$$

which means $F ( \beta ^ { k + 1 } )$ is bounded, and for each group $j ,$

$$
\| \beta _ { j } ^ { k + 1 } \| _ { 2 } ^ { q } \leq \sum _ { j = 1 } ^ { J } T _ { j } ^ { q / 2 } \| \beta _ { j } ^ { k + 1 } \| _ { 2 } ^ { q } \leq \frac { 1 } { \lambda _ { 1 } } F ( \beta ^ { k + 1 } ) .
$$

Then there exists $C > \tau _ { 1 }$ depending on $\beta ^ { 0 }$ such that $\| \beta _ { j } ^ { k } \| _ { 2 } \leq C$ for all $1 \leq j \leq J$ . Thus, the proof of statement (i) is completed.

(ii) By Lemma A.5, for all $k \geq K$ , we have $\mathbb { G } ^ { k } = \mathbb { G } ^ { k + 1 } = \mathbb { G } ^ { * }$ . Let $T _ { \operatorname* { m i n } } = \operatorname* { m i n } \{ T _ { j } : 1 \leq j \leq J \}$ . It follows from (B.28) that for $k \geq K , F ( \beta ^ { k } )$ is non-increasing with

$$
{ \cal F } ( { \boldsymbol \beta } ^ { k } ) - { \cal F } ( { \boldsymbol \beta } ^ { k + 1 } ) \geq \frac { \lambda _ { 1 } q } { 2 } C ^ { q - 2 } \sum _ { j \in \mathbb { Q } ^ { * } } T _ { j } ^ { q / 2 } \| { \boldsymbol \beta } _ { j } ^ { k + 1 } - { \boldsymbol \beta } _ { j } ^ { k } \| _ { 2 } ^ { 2 } \geq \frac { \lambda _ { 1 } q } { 2 } { \cal T } _ { \operatorname* { m i n } } ^ { q / 2 } C ^ { q - 2 } \| { \boldsymbol \beta } ^ { k + 1 } - { \boldsymbol \beta } ^ { k } \| _ { 2 } ^ { 2 } .
$$

Let $\begin{array} { r } { a = \frac { \lambda _ { 1 } q } { 2 } T _ { \mathrm { m i n } } ^ { q / 2 } C ^ { q - 2 } } \end{array}$ . Then inequality (23) holds for all $k \geq K$

We next prove that (23) holds for all $k \geq 0$ under the assumption

$$
\tau _ { 1 } \leq \tilde { \tau } _ { 1 } = \operatorname* { m i n } \Bigg \{ \Bigg ( \frac { T _ { \operatorname* { m i n } } ^ { q / 2 } \lambda _ { 1 } } { 2 \| X \| _ { 2 } } \Bigg ) ^ { \frac { 1 } { 1 - q } } , \Bigg ( \frac { \lambda _ { 1 } } { 2 \tilde { a } } \Bigg ) ^ { \frac { 1 } { 2 - q } } \Bigg \} ,
$$

where $\tilde { a } > 0$ is an arbitrary positive constant. When $\mathbb { G } ^ { k + 1 } \subseteq \mathbb { G } ^ { k }$ , for all $j \in \mathbb { G } ^ { k } \setminus \mathbb { G } ^ { k + 1 }$ , we have

$$
\| X \| _ { 2 } \| e _ { j } ^ { k + 1 } \| _ { 2 } ^ { 1 - q } + \tilde { a } T _ { j } ^ { q / 2 } \| e _ { j } ^ { k + 1 } \| _ { 2 } ^ { 2 - q } - \lambda _ { 1 } T _ { j } ^ { q / 2 } \leq 0 .
$$

Rearranging gives

$$
\| X \| _ { 2 } \| e _ { j } ^ { k + 1 } \| _ { 2 } - \lambda _ { 1 } T _ { j } ^ { q / 2 } \| e _ { j } ^ { k + 1 } \| _ { 2 } ^ { q } \leq - \tilde { a } T _ { j } ^ { q / 2 } \| e _ { j } ^ { k + 1 } \| _ { 2 } ^ { 2 } .
$$

We thus obtain

$$
F ( \beta ^ { k + 1 } ) - F ( \widetilde { \beta } ^ { k + 1 } ) \leq - \tilde { a } \sum _ { j \in \mathbb { G } ^ { k } \backslash \mathbb { G } ^ { k + 1 } } T _ { j } ^ { q / 2 } \| e _ { j } ^ { k + 1 } \| _ { 2 } ^ { 2 } .\tag{B.32}
$$

Note that $\beta _ { j } ^ { k + 1 } = \tilde { \beta } _ { j } ^ { k + 1 }$ for all $j \in \mathbb { G } ^ { k + 1 }$ . For $j \in \mathbb { G } ^ { k } \setminus \mathbb { G } ^ { k + 1 }$ , we have $\beta _ { j } ^ { k + 1 } = 0$ and $2 \| e _ { j } ^ { k + 1 } - \beta _ { j } ^ { k } \| _ { 2 } ^ { 2 } + 2 \| e _ { j } ^ { k + 1 } \| _ { 2 } ^ { 2 } \ge$ $\| \beta _ { j } ^ { k } \| _ { 2 } ^ { 2 }$ . Combining (B.29), (B.30), and (B.32), we obtain

$$
\begin{array} { r l } { F ( \beta ^ { k } ) - F ( \beta ^ { k + 1 } ) \ge \displaystyle \frac { 4 \Im { q } } { 2 } \sum _ { j \ge 0 } ^ { \infty / 2 } \| g _ { j } ^ { k } \| _ { 2 } ^ { \alpha - 2 } \| \beta _ { j } ^ { k + 1 } - \beta _ { j } ^ { k } \| _ { 2 } ^ { 2 } + \delta \sum _ { j \ge 0 } \sum _ { i \ge 0 } ^ { \infty / 2 } \| g _ { j } ^ { k + 1 } \| _ { 2 } ^ { 2 } } \\ & { \lesssim \displaystyle \frac { \Im { q } } { 2 } \sum _ { j \ge 0 } ^ { \infty / 2 } \sum _ { i } ^ { \beta ^ { k } / 2 } \| g _ { j } ^ { k } \| _ { 2 } ^ { \alpha - 2 } \| g _ { j } ^ { k + 1 } - \beta _ { j } ^ { k } \| _ { 2 } ^ { 2 } } \\ & { \quad + \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ { \ge \displaystyle \sum _ { j \ge 0 } ^ { \infty / 2 } \sum _ { i } ^ { \beta ^ { k + 1 } / 2 } T ^ { q / 2 } \left[ \frac { \lambda _ { 1 } \lambda _ { 2 } } { 2 } \| g _ { j } ^ { k } \| _ { 2 } ^ { \alpha - 2 } \| g _ { j } ^ { k + 1 } - \beta _ { j } ^ { k } \| _ { 2 } ^ { 2 } + \alpha \| g _ { j } ^ { k + 1 } \| _ { 2 } ^ { 2 } \right] } \\ & { \ge \displaystyle \sum _ { j \ge 0 } ^ { \infty } \frac { \lambda _ { 1 } \mathcal { G } _ { j } } { 2 } \sum _ { i = 0 } ^ { \infty / 2 } C ^ { - \beta - 1 } \| g _ { j } ^ { k + 1 } - \beta _ { j } ^ { k } \| _ { 2 } ^ { 2 } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \end{array}
$$

Let $\begin{array} { r } { a = \operatorname* { m i n } \left\{ \frac { \lambda _ { 1 } q } { 4 } T _ { \operatorname* { m i n } } ^ { q / 2 } C ^ { q - 2 } , \frac { \tilde { a } } { 2 } T _ { \operatorname* { m i n } } ^ { q / 2 } \right\} } \end{array}$ . If $\tau _ { 1 } \leq \tilde { \tau } , F ( \beta ^ { k } )$ is non-increasing for all $k \geq 0$ , with

$$
F ( \boldsymbol { \beta } ^ { k } ) - F ( \boldsymbol { \beta } ^ { k + 1 } ) \geq a \| \boldsymbol { \beta } ^ { k + 1 } - \boldsymbol { \beta } ^ { k } \| _ { 2 } ^ { 2 } .
$$

This completes the proof of statement (ii).

Proof of Lemma 2 For each $j = 1 , \dots , J ,$ , we construct a groupwise subgradient $u ^ { k + 1 } \in \partial { \cal F } ( \beta ^ { k + 1 } )$ following

Proposition 2, with

$$
u _ { j } ^ { k + 1 } = \left\{ \begin{array} { l l } { - \displaystyle \frac { X _ { j } ^ { \top } ( y - X \beta ^ { k + 1 } ) } { \| y - X \beta ^ { k + 1 } \| _ { 2 } } + \lambda _ { 1 } q T _ { j } ^ { q / 2 } \| \beta _ { j } ^ { k + 1 } \| _ { 2 } ^ { q - 2 } \beta _ { j } ^ { k + 1 } + \lambda _ { 2 } q T _ { j } ^ { q / 2 } \| D _ { j } \beta _ { j } ^ { k + 1 } \| _ { 2 } ^ { q - 2 } D _ { j } ^ { \top } D _ { j } \beta _ { j } ^ { k + 1 } , } & { j \in \mathbb { A } ^ { k + 1 } , } \\ { 0 , } & { j \not \in \mathbb { A } ^ { k + 1 } . } \end{array} \right.
$$

Let $r _ { k } = \| y - X \beta ^ { k } \| _ { 2 }$ and $r _ { k + 1 } = \| y - X \beta ^ { k + 1 } \| _ { 2 }$ . For any $j \in \mathbb { A } ^ { k + 1 }$ , the first-order optimality condition of subproblem (21) gives

$$
\frac { 1 } { r _ { k } } X _ { j } ^ { \top } ( X \widetilde { \beta } ^ { k + 1 } - y ) + \lambda _ { 1 } T _ { j } ^ { q / 2 } q \Vert \beta _ { j } ^ { k } \Vert _ { 2 } ^ { q - 2 } \widetilde { \beta } _ { j } ^ { k + 1 } + \lambda _ { 2 } T _ { j } ^ { q / 2 } q \Vert D _ { j } \beta _ { j } ^ { k } \Vert _ { 2 } ^ { q - 2 } D _ { j } ^ { \top } D _ { j } \widetilde { \beta } _ { j } ^ { k + 1 } = 0 .
$$

For $j \in \mathbb { A } ^ { k + 1 }$ , we have $\beta _ { j } ^ { k + 1 } = \tilde { \beta } _ { j } ^ { k + 1 }$ . Recall that $e ^ { k + 1 } = { \tilde { \beta } } ^ { k + 1 } - \beta ^ { k + 1 }$ , then the above equality can be rewritten as

$$
\frac { 1 } { r _ { k } } X _ { j } ^ { \top } ( X \beta ^ { k + 1 } + X e ^ { k + 1 } - y ) + \lambda _ { 1 } T _ { j } ^ { q / 2 } q \| \beta _ { j } ^ { k } \| _ { 2 } ^ { q - 2 } \beta _ { j } ^ { k + 1 } + \lambda _ { 2 } T _ { j } ^ { q / 2 } q \| D _ { j } \beta _ { j } ^ { k } \| _ { 2 } ^ { q - 2 } D _ { j } ^ { \top } D _ { j } \beta _ { j } ^ { k + 1 } = 0 .
$$

Substituting it into $u _ { j } ^ { k + 1 }$ yields

$$
\begin{array} { l } { { \displaystyle { u _ { j } ^ { k + 1 } = - \frac { X _ { j } ^ { \top } ( y - X \beta ^ { k + 1 } ) } { r _ { k + 1 } } + \lambda _ { 1 } T _ { j } ^ { q / 2 } q \| \beta _ { j } ^ { k + 1 } \| _ { 2 } ^ { q - 2 } \beta _ { j } ^ { k + 1 } + \lambda _ { 2 } T _ { j } ^ { q / 2 } q \| D _ { j } \beta _ { j } ^ { k + 1 } \| _ { 2 } ^ { q - 2 } D _ { j } ^ { \top } D _ { j } \beta _ { j } ^ { k + 1 } } } } \\ { { \displaystyle { \quad = \left( \frac { 1 } { r _ { k } } - \frac { 1 } { r _ { k + 1 } } \right) X _ { j } ^ { \top } ( y - X \beta ^ { k + 1 } ) - \frac { 1 } { r _ { k } } X _ { j } ^ { \top } X e ^ { k + 1 } + \lambda _ { 1 } T _ { j } ^ { q / 2 } q \left( \| \beta _ { j } ^ { k + 1 } \| _ { 2 } ^ { q - 2 } - \| \beta _ { j } ^ { k } \| _ { 2 } ^ { q - 2 } \right) \beta _ { j } ^ { k + 1 } } } } \\ { { \displaystyle { \quad \quad + \lambda _ { 2 } T _ { j } ^ { q / 2 } q \left( \| D _ { j } \beta _ { j } ^ { k + 1 } \| _ { 2 } ^ { q - 2 } - \| D _ { j } \beta _ { j } ^ { k } \| _ { 2 } ^ { q - 2 } \right) D _ { j } ^ { \top } D _ { j } \beta _ { j } ^ { k + 1 } } . } } \end{array}
$$

Then, we denote $d ^ { k + 1 } , \tilde { u } ^ { k + 1 } , \tilde { \nu } ^ { k + 1 } \in \mathbb { R } ^ { p }$ as

$$
d _ { j } ^ { k + 1 } = \left\{ \begin{array} { l l } { \displaystyle \left( \frac { 1 } { r _ { k } } - \frac { 1 } { r _ { k + 1 } } \right) X _ { j } ^ { \top } ( y - X \beta ^ { k + 1 } ) - \frac { 1 } { r _ { k } } X _ { j } ^ { \top } X e ^ { k + 1 } , } & { j \in \mathbb { A } ^ { k + 1 } , } \\ { 0 , } & { j \not \in \mathbb { A } ^ { k + 1 } , } \end{array} \right.
$$

$$
\tilde { u } _ { j } ^ { k + 1 } = \left\{ \begin{array} { l l } { \lambda _ { 1 } T _ { j } ^ { q / 2 } q \left( \lVert { \beta } _ { j } ^ { k + 1 } \rVert _ { 2 } ^ { q - 2 } - \lVert { \beta } _ { j } ^ { k } \rVert _ { 2 } ^ { q - 2 } \right) \beta _ { j } ^ { k + 1 } , } & { j \in \mathbb { A } ^ { k + 1 } , } \\ { 0 , } & { j \not \in \mathbb { A } ^ { k + 1 } , } \end{array} \right.
$$

$$
\begin{array} { r } { \tilde { \nu } _ { j } ^ { k + 1 } = \left\{ \begin{array} { l l } { \lambda _ { 2 } T _ { j } ^ { q / 2 } q \left( \| D _ { j } { \beta } _ { j } ^ { k + 1 } \| _ { 2 } ^ { q - 2 } - \| D _ { j } { \beta } _ { j } ^ { k } \| _ { 2 } ^ { q - 2 } \right) D _ { j } ^ { \top } D _ { j } { \beta } _ { j } ^ { k + 1 } , } & { j \in \mathbb { A } ^ { k + 1 } , } \\ { 0 , } & { j \not \in \mathbb { A } ^ { k + 1 } . } \end{array} \right. } \end{array}
$$

It follows that

$$
\begin{array} { r } { u ^ { k + 1 } = d ^ { k + 1 } + \tilde { u } ^ { k + 1 } + \tilde { \nu } ^ { k + 1 } . } \end{array}\tag{B.33}
$$

We now bound each component separately. First, by the triangle inequality, we have

$$
\begin{array} { r } { \lvert r _ { k + 1 } - r _ { k } \rvert = \big \lvert \lVert y - X \beta ^ { k + 1 } \rVert _ { 2 } - \lVert y - X \beta ^ { k } \rVert _ { 2 } \big \rvert \le \lVert X ( \beta ^ { k + 1 } - \beta ^ { k } ) \rVert _ { 2 } \le \lVert X \rVert _ { 2 } \lVert \beta ^ { k + 1 } - \beta ^ { k } \rVert _ { 2 } . } \end{array}
$$

Since $\| y - X \beta ^ { k } \| _ { 2 }$ is bounded below, there exists $\varsigma > 0$ such that $\| y - X \beta ^ { k } \| _ { 2 } > \varsigma$ for $k \geq 0$ . Then, we have

$$
\left| \frac { 1 } { r _ { k } } - \frac { 1 } { r _ { k + 1 } } \right| = \frac { | r _ { k + 1 } - r _ { k } | } { r _ { k } r _ { k + 1 } } \leq \frac { \| X \| _ { 2 } \| \beta ^ { k + 1 } - \beta ^ { k } \| _ { 2 } } { \varsigma ^ { 2 } } .
$$

Recall that $e _ { j } ^ { k + 1 } = 0$ for all $j \notin \mathbb { G } ^ { k } \setminus \mathbb { G } ^ { k + 1 }$ . For $j \in \mathbb { G } ^ { k } \setminus \mathbb { G } ^ { k + 1 }$ , we have $\beta _ { j } ^ { k + 1 } = 0 , \| e _ { j } ^ { k + 1 } \| _ { 2 } < \tau _ { 1 }$ , and $\| \beta _ { j } ^ { k } \| _ { 2 } \ge \tau _ { 1 }$ Let $S ^ { k } = \cup _ { j \in \mathbb { G } ^ { k } } { \mathcal { G } } _ { j }$ , we have

$$
\| e ^ { k + 1 } \| _ { 2 } \leq \| e _ { S ^ { k } \backslash S ^ { k + 1 } } ^ { k + 1 } \| _ { 2 } \leq \| \beta _ { S ^ { k } \backslash S ^ { k + 1 } } ^ { k } \| _ { 2 } = \| \beta _ { S ^ { k } \backslash S ^ { k + 1 } } ^ { k + 1 } - \beta _ { S ^ { k } \backslash S ^ { k + 1 } } ^ { k } \| _ { 2 } .
$$

It follows from Lemma 1 that $\| y - X \beta ^ { k } \| _ { 2 }$ is uniformly bounded above. Let $\begin{array} { r } { \tilde { \varsigma } = \operatorname* { s u p } _ { k } \| y - X \beta ^ { k } \| _ { 2 } < \infty } \end{array}$ , then by the triangle inequality we have

$$
\| d ^ { k + 1 } \| _ { 2 } \leq \frac { \| X \| _ { 2 } ^ { 2 } \tilde { \varsigma } } { \varsigma ^ { 2 } } \| \beta ^ { k + 1 } - \beta ^ { k } \| _ { 2 } + \frac { \| X \| _ { 2 } ^ { 2 } } { \varsigma } \| \beta _ { S ^ { k } \backslash S ^ { k + 1 } } ^ { k + 1 } - \beta _ { S ^ { k } \backslash S ^ { k + 1 } } ^ { k } \| _ { 2 } .
$$

For $j \in \mathbb { A } ^ { k + 1 }$ , it follows from Lemma 1 that $\tau _ { 1 } \leq \| \beta _ { j } ^ { k } \| _ { 2 } , \| \beta _ { j } ^ { k + 1 } \| _ { 2 } \leq C$ . Since $\phi ( t ) = t ^ { q - 2 }$ is Lipschitzcontinuous on $[ \tau _ { 1 } , C ]$ with Lipschitz constant $l _ { \tau _ { 1 } }$ , we obtain

$$
\begin{array} { r } { \left| \| { \beta } _ { j } ^ { k + 1 } \| _ { 2 } ^ { q - 2 } - \| { \beta } _ { j } ^ { k } \| _ { 2 } ^ { q - 2 } \right| \leq l _ { \tau _ { 1 } } \left| \| { \beta } _ { j } ^ { k + 1 } \| _ { 2 } - \| { \beta } _ { j } ^ { k } \| _ { 2 } \right| \leq l _ { \tau _ { 1 } } \| { \beta } _ { j } ^ { k + 1 } - { \beta } _ { j } ^ { k } \| _ { 2 } . } \end{array}
$$

Let $T _ { \mathrm { m a x } } = \mathrm { m a x } _ { 1 \leq j \leq J } T _ { j }$ . Since $\| \beta _ { j } ^ { k + 1 } \| _ { 2 } \leq C$ , we have

$$
\begin{array} { r } { \| \tilde { { \boldsymbol u } } _ { j } ^ { k + 1 } \| _ { 2 } \leq \lambda _ { 1 } q T _ { \operatorname* { m a x } } ^ { q / 2 } l _ { \tau _ { 1 } } C \| \beta _ { j } ^ { k + 1 } - \beta _ { j } ^ { k } \| _ { 2 } . } \end{array}
$$

Thus,

$$
\begin{array} { r l r } {  { \| \tilde { u } ^ { k + 1 } \| _ { 2 } \le \lambda _ { 1 } q T _ { \operatorname* { m a x } } ^ { q / 2 } l _ { \tau _ { 1 } } C \sqrt { \sum _ { j \in \mathbb { A } ^ { k + 1 } } \| \beta _ { j } ^ { k + 1 } - \beta _ { j } ^ { k } \| _ { 2 } ^ { 2 } } \le \lambda _ { 1 } q T _ { \operatorname* { m a x } } ^ { q / 2 } l _ { \tau _ { 1 } } C \sqrt { \sum _ { j \in \mathbb { B } ^ { k + 1 } } \| \beta _ { j } ^ { k + 1 } - \beta _ { j } ^ { k } \| _ { 2 } ^ { 2 } } } } \\ & { } & { \le \lambda _ { 1 } q T _ { \operatorname* { m a x } } ^ { q / 2 } l _ { \tau _ { 1 } } C \| \beta _ { S ^ { k + 1 } } ^ { k + 1 } - \beta _ { S ^ { k + 1 } } ^ { k } \| _ { 2 } . } \end{array}\tag{B.35}
$$

Similarly, since $D _ { j }$ is the first-order diference matrix, we have $\| D _ { j } \| _ { 2 } \leq 2$ and $\psi ( t ) = t ^ { q - 2 }$ is Lipschitz continuous on $[ \tau _ { 2 } , 2 C ]$ with constant $l _ { \tau _ { 2 } , D }$ . For any $j \in \mathbb { A } ^ { k + 1 }$ , we obtain

$$
\begin{array} { r } { \left| \| D _ { j } { \beta } _ { j } ^ { k + 1 } \| _ { 2 } ^ { q - 2 } - \| D _ { j } { \beta } _ { j } ^ { k } \| _ { 2 } ^ { q - 2 } \right| \leq l _ { \tau _ { 2 } , D } \left| \| D _ { j } { \beta } _ { j } ^ { k + 1 } \| _ { 2 } - \| D _ { j } { \beta } _ { j } ^ { k } \| _ { 2 } \right| \leq 2 l _ { \tau _ { 2 } , D } \| { \beta } _ { j } ^ { k + 1 } - { \beta } _ { j } ^ { k } \| _ { 2 } . } \end{array}
$$

Note that $\| D _ { j } ^ { \top } D _ { j } \beta _ { j } ^ { k + 1 } \| _ { 2 } \leq \| D _ { j } \| _ { 2 } ^ { 2 } \| \beta _ { j } ^ { k + 1 } \| _ { 2 } \leq 4 C$ . We thus obtain

$$
\begin{array} { r } { \| \tilde { \nu } ^ { k + 1 } \| _ { 2 } \leq 8 \lambda _ { 2 } q T _ { \operatorname* { m a x } } ^ { q / 2 } l _ { \tau _ { 2 } , D } C \| \beta _ { S ^ { k + 1 } } ^ { k + 1 } - \beta _ { S ^ { k + 1 } } ^ { k } \| _ { 2 } . } \end{array}\tag{B.36}
$$

Combining (B.35) and (B.36), we get

$$
\begin{array} { r } { \| \tilde { u } ^ { k + 1 } \| _ { 2 } + \| \tilde { \nu } ^ { k + 1 } \| _ { 2 } \leq \left( \lambda _ { 1 } q T _ { \operatorname* { m a x } } ^ { q / 2 } l _ { \tau _ { 1 } } C + 8 \lambda _ { 2 } q T _ { \operatorname* { m a x } } ^ { q / 2 } l _ { \tau _ { 2 } , D } C \right) \| \beta _ { S ^ { k + 1 } } ^ { k + 1 } - \beta _ { S ^ { k + 1 } } ^ { k } \| _ { 2 } \leq \tilde { b } \| \beta ^ { k + 1 } - \beta ^ { k } \| _ { 2 } , } \end{array}
$$

where $\tilde { b } = \lambda _ { 1 } q T _ { \mathrm { m a x } } ^ { q / 2 } l _ { \tau _ { 1 } } C + 8 \lambda _ { 2 } q T _ { \mathrm { m a x } } ^ { q / 2 } l _ { \tau _ { 2 } , D } C .$

Therefore,

$$
\begin{array} { l } { \| u ^ { k + 1 } \| _ { 2 } \le \| d ^ { k + 1 } \| _ { 2 } + \| \tilde { u } ^ { k + 1 } \| _ { 2 } + \| \tilde { \nu } ^ { k + 1 } \| _ { 2 } } \\ { \qquad \le \frac { \| X \| _ { 2 } ^ { 2 } \tilde { S } } { { \varsigma ^ { 2 } } } \| \beta ^ { k + 1 } - \beta ^ { k } \| _ { 2 } + \sqrt { 2 } \operatorname* { m a x } \left\{ \tilde { b } , \frac { \| X \| _ { 2 } ^ { 2 } } { \varsigma } \right\} \| \beta ^ { k + 1 } - \beta ^ { k } \| _ { 2 } \le b \| \beta ^ { k + 1 } - \beta ^ { k } \| _ { 2 } , } \end{array}\tag{B.37}
$$

where

$$
b = \frac { \| X \| _ { 2 } ^ { 2 } \tilde { S } } { \varsigma ^ { 2 } } + \sqrt { 2 } \operatorname* { m a x } \left\{ \tilde { b } , \frac { \| X \| _ { 2 } ^ { 2 } } { \varsigma } \right\} .
$$

This completes the proof.

Proof of Theorem 2 From Lemma 1 and 2, we observe that the function value sequence $\{ F ( \beta ^ { k } ) \}$ has suficient descent and the subdiferential sequence $\{ \partial F ( \beta ^ { k } ) \}$ is relatively bounded. Lemma 1 (i) further implies that the sequence $\{ \beta ^ { k } \}$ is bounded, hence there exists a subsequence $\{ \beta ^ { k _ { i } } \}$ converging to a point $\beta ^ { * }$ . Moreover, the objective function $F ( \beta )$ is semi-algebraic, and thus satisfies the Kurdyka–Łojasiewicz (KL) property on its domain. With all conditions of Theorem 2.9 in [45] satisfied, we conclude that the sequence $\{ \beta ^ { k } \}$ converges to a stationary point $\beta ^ { * }$ of $F ( \beta )$ , and the finite-length property holds. Thus, statements (i) and (ii) are proven.

## References

[1] Donoho, D. L. Compressed sensing. IEEE Transactions on Information Theory, 52(4):1289–1306, 2006.

[2] Tharmaratnam, K., Sperrin, M., Jaki, T., et al. Tilting the Lasso by knowledge-based post-processing. BMC Bioinformatics, 17(1):344, 2016.

[3] Zeng, C., Thomas, D. C., Lewinger, J. P. Incorporating prior knowledge into regularized regression. Bioinformatics, 37(4):514–521, 2021.

[4] Hu, Y., Li, C., Meng, K., et al. Group sparse optimization via $\ell _ { p , q }$ regularization. Journal of Machine Learning Research, 18(30):1–52, 2017.

[5] Xie, W., Yang, H. The structured smooth adjustment for square-root regularization: Theory, algorithm and applications. Knowledge-Based Systems, 207:106278, 2020.

[6] Xue, Y., Feng, Y., Wu, C. An Eficient and Globally Convergent Algorithm for $\ell _ { p , q } { - } \ell _ { r }$ Model in Group Sparse Optimization. Communications in Mathematical Sciences, 18(1), 2020.

[7] Chang, X., Zhou, M., Wang, X., et al. Informative relationship multi-task learning: Exploring pairwise contribution across tasks’ sharing knowledge. Knowledge-Based Systems, 301:112187, 2024.

[8] Saputra, W. H., Prastyo, D. D., Fithriasari, K. Knowledge-driven Nodes Selection with Modified LASSO Regularization for Neural Network Modeling in Economic Data Forecasting. Knowledge-Based Systems, 115273, 2026.

[9] Tibshirani, R. Regression shrinkage and selection via the Lasso. Journal of the Royal Statistical Society Series B: Statistical Methodology, 58(1):267–288, 1996.

[10] Yuan, M., Lin, Y. Model selection and estimation in regression with grouped variables. Journal of the Royal Statistical Society Series B: Statistical Methodology, 68(1):49–67, 2006.

[11] Wang, L., Chen, G., Li, H. Group SCAD regression analysis for microarray time course gene expression data. Bioinformatics, 23(12):1486–1494, 2007.

[12] Breheny, P., Huang, J. Group descent algorithms for nonconvex penalized linear and logistic regression models with grouped predictors. Statistics and Computing, 25(2):173–187, 2015.

[13] Lee, S., Oh, M., Kim, Y. Sparse optimization for nonconvex group penalized estimation. Journal of Statistical Computation and Simulation, 86(3):597–610, 2016.

[14] Pan, L., Chen, X. Group sparse optimization for images recovery using capped folded concave functions. SIAM Journal on Imaging Sciences, 14(1):1–25, 2021.

[15] Zhang, X., Peng, D. Solving constrained nonsmooth group sparse optimization via group Capped-ℓ<sub>1</sub> relaxation and group smoothing proximal gradient algorithm. Computational Optimization and Applications, 83(3):801–844, 2022.

[16] Rakotomamonjy, A., Flamary, R., Gasso, G., et al. $\ell _ { p } - \ell _ { q }$ Penalty for Sparse Linear and Sparse Multiple Kernel Multitask Learning. IEEE Transactions on Neural Networks, 22(8):1307–1320, 2011.

[17] Tibshirani, R., Saunders, M., Rosset, S., et al. Sparsity and smoothness via the fused Lasso. Journal of the Royal Statistical Society Series B: Statistical Methodology, 67(1):91–108, 2005.

[18] Hebiri, M., Sara, V. D. G. The Smooth-Lasso and other ℓ<sub>1</sub> ℓ<sub>2</sub>-penalized methods. Electronic Journal of Statistics, 5:1184–1226, 2011.

[19] Guo, J., Hu, J., Jing, B. Y., et al. Spline-Lasso in high-dimensional linear regression. Journal of the American Statistical Association, 111(513):288–297, 2016.

[20] Ma, S., Huang, J. A concave pairwise fusion approach to subgroup analysis. Journal of the American Statistical Association, 112(517):410–423, 2017.

[21] Wen, T., Chen, Z., Zhang, T., et al. Graph-based semi-supervised learning with non-convex graph total variation regularization. Expert Systems with Applications, 255:124709, 2024.

[22] Belloni, A., Chernozhukov, V., Wang, L. Square-root Lasso: pivotal recovery of sparse signals via conic programming. Biometrika, 98(4):791–806, 2011.

[23] Bunea, F., Lederer, J., She, Y. The group square-root Lasso: Theoretical properties and fast algorithms. IEEE Transactions on Information Theory, 60(2):1313–1325, 2013.

[24] Jiang, H., Luo, S., Dong, Y. Simultaneous feature selection and clustering based on square root optimization. European Journal ofOperational Research, 289(1):214–231, 2021.

[25] Zhao, C., Mao, X., Chen, M., et al. A fast algorithm for group square-root Lasso based group-sparse regression. Signal Processing, 187:108142, 2021.

[26] Chu, H. T. M., Toh, K. C., Zhang, Y. On regularized square-root regression problems: distributionally robust interpretation and fast computations. Journal of Machine Learning Research, 23(308):1–39, 2022.

[27] Wang, C., Tang, P. A dual semismooth Newton based augmented Lagrangian method for large-scale linearly constrained sparse group square-root Lasso problems. Journal of Scientific Computing, 96(2):45, 2023.

[28] Gai, Y., Meng, K., Xu, X. Online Variable Selection and Parameter Estimation for Massive Data via Square Root Lasso. Statistical Analysis and Data Mining: The ASA Data Science Journal, 18(2):e70016, 2025.

[29] Bickel, P. J., Ritov, Ya’acov, Tsybakov, A. B. Simultaneous analysis of Lasso and Dantzig selector. The Annals ofStatistics, 37(4):1705–1732, 2009.

[30] Meinshausen, N., Yu, B. Lasso-type recovery of sparse representations for high-dimensional data. Annals of Statistics, 37(1):246–270, 2009.

[31] Li, X., Hu, Y., Li, C., et al. Sparse estimation via lower-order penalty optimization methods in highdimensional linear regression. Journal ofGlobal Optimization, 85(2):315–349, 2023.

[32] Negahban, S. N., Ravikumar, P., Wainwright, M. J., et al. A unified framework for high-dimensional analysis of M-estimators with decomposable regularizers. Statistical Science, 538–557, 2012.

[33] Feng, X., Yan, S., Wu, C. The $\ell _ { 2 , q }$ regularized group sparse optimization: lower bound theory, recovery bound and algorithms. Applied and Computational Harmonic Analysis, 49(2):381–414, 2020.

[34] Saad, Y. Iterative methods for sparse linear systems. Society for Industrial and Applied Mathematics, 2003.

[35] Chen, J., Chen, Z. Extended Bayesian information criteria for model selection with large model spaces. Biometrika, 95(3):759–771, 2008.

[36] Reppe, S., Refvem, H., Gautvik, V. T., et al. Eight genes are highly associated with BMD variation in postmenopausal Caucasian women. Bone, 46(3):604–612, 2010.

[37] Friedman, J. H., Hastie, T., Tibshirani, R. Regularization paths for generalized linear models via coordinate descent. Journal of Statistical Software, 33:1–22, 2010.

[38] She, Y. An iterative algorithm for fitting nonconvex penalized generalized linear models with grouped predictors. Computational Statistics & Data Analysis, 56(10):2976–2990, 2012.

[39] Rockafellar, R. T., Wets, R. J. B. Variational analysis. Springer, 1998.

[40] Attouch, H., Bolte, J., Redont, P., et al. Proximal alternating minimization and projection methods for nonconvex problems: An approach based on the Kurdyka-Łojasiewicz inequality. Mathematics of Operations Research, 35(2):438–457, 2010.

[41] Bolte, J., Sabach, S., Teboulle, M. Proximal alternating linearized minimization for nonconvex and nonsmooth problems. Mathematical Programming, 146(1):459–494, 2014.

[42] Bühlmann, P., Van De Geer, S. Statistics for high-dimensional data: methods, theory and applications. Springer Science & Business Media, 2011.

[43] Boyd, S., Vandenberghe, L. Convex optimization. Cambridge University Press, 2004.

[44] Movahedian, N. Bounded Lagrange multiplier rules for general nonsmooth problems and application to mathematical programs with equilibrium constraints. Journal of Global Optimization, 67(4):829–850, 2017.

[45] Attouch, H., Bolte, J., Svaiter, B. F. Convergence of descent methods for semi-algebraic and tame problems: proximal algorithms, forward–backward splitting, and regularized Gauss–Seidel methods. Mathematical Programming, 137(1):91–129, 2013.