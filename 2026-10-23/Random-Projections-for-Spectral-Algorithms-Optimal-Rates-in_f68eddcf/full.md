# Random Projections for Spectral Algorithms: Optimal Rates in Sobolev Norms<sup>⋆</sup>

Caixing Wang<sup>a</sup>, Yue Wang<sup>b</sup>

<sup>a</sup>School ofStatistics and Data Science, Southeast University, , Nanjing, 210000, Jiangsu, China

<sup>b</sup>Faculty of Business for Science and Technology, School of Management, University of Science and Technology of China, , Hefei, 230000, Anhui, China

## A R T I C L E I N F O

<sub>Keywords:</sub>6   
<sub>Regularized</sub> <sub>algorithm</sub>2   
<sub>random projection</sub>0   
<sub>reproducing kernel Hilbert space</sub>2   
minimax rates   
mis-specified settingc

## A BS T RA C T

Random projections (RP) ofer an efective method for reducing computational and storage overhead while preserving the geometric structure of data. However, existing studies have primarily focused on the optimal generalization performance of specific kernel-regularized algorithms with RP in the well-specified setting and under restrictive conditions. In this paper, we provide a comprehensive and improved analysis of the generalization performance of RPbased spectral algorithms under general conditions without sacrificing statistical eficiency. By leveraging the embedding property of the reproducing kernel Hilbert space (RKHS) and a refined analysis of the operator similarity, we establish optimal learning rates in Sobolev norms that match the minimax lower bounds up to logarithmic factors. For two common implementations of RP, randomized sketches and Nyström sub-sampling (uniform or leverage-based), we show that the projection dimension required for optimality is proportional to the average or maximal efective dimension, yielding a significant reduction in computational cost while maintaining the statistical eficiency. Our results do not rely on the uniform boundedness assumption on the regression function and are valid across a wide range of source conditions, i.e., $s > \alpha _ { 0 } - 1 / \beta _ { }$ , where $s , \beta ,$ and $\alpha _ { 0 }$ denote the smoothness index, capacity index, and the embedding index, respectively. In the benign case when $\alpha _ { 0 } = 1 / \beta _ { : }$ the optimality holds for all $s \in ( 0 , 2 \tau ]$ with � and � denoting the qualification index and norm index. Experimental results demonstrate both the theoretical validity and the practical efectiveness of the proposed method.

## 1. Introduction

In statistical learning theory, the primary objective is to learn a function that approximates the underlying relationship between input and output variables based on a finite set of training data (Vapnik, $1 9 9 9 \mathrm { b , a ) }$ . Kernel methods, a class of nonparametric learning algorithms, have gained significant attention due to their ability to capture complex data structures and their strong generalization performance (Wahba, 1990; Schölkopf and Smola, 2002). These methods<sup>.</sup> operate in a reproducing kernel Hilbert space (RKHS) associated with a positive definite kernel function $K ( \cdot , \cdot )$ . To address the ill-posedness of such learning problems, regularization techniques are commonly employed, leading to a family of kernel-regularized algorithms, or spectral algorithms (Engl, Hanke and Neubauer, 1996). Prominent examples include Tikhonov regularization (also known as kernel ridge regression, KRR) (Smale and Zhou, 2007; Caponnetto and De Vito, 2007), spectral cutof (Blanchard, Bousquet and Zwald, 2007; Dicker, Foster and Hsu, 2017), and gradient-based methods (Yao, Rosasco and Caponnetto, 2007; Lin and Rosasco, 2016). Despite their solid theoretical guarantees, these algorithms often sufer high computational costs when applied to large-scale datasets. For instance, KRR typically requires $O ( n ^ { 3 } )$ time and $O ( n ^ { 2 } )$ space complexity, where � denotes the number of training samples. To mitigate this issue, various accelerating techniques have been developed, such as distributed algorithms (Zhang, Duchi and Wainwright, 2015; Guo, Lin and Zhou, 2017), random features (Rahimi and Recht, 2007; Rudi and Rosasco, 2017), randomized sketches (Yang, Pilanci and Wainwright, 2017; Zhang and Liao, 2019), and Nyström sub-sampling (Williams and Seeger, 2001; Kumar, Mohri and Talwalkar, 2012).

Random projections (RP), including randomized sketches and Nyström sub-sampling as special cases, have emerged as powerful techniques for dimensionality reduction. They ofer substantial computational savings while efectively preserving the geometric structure of data (Vempala, 2005). The core idea of RP is to estimate the regression function within a suitably chosen random subspace of the RKHS, thereby yielding spectral algorithms with RP. Specifically, for a chosen projection dimension $m \ll n ,$ we construct a random subspace $\mathcal { F } = \overline { { \operatorname { s p a n } \{ K _ { { \pmb { x } } _ { i _ { j } } } \} _ { j = 1 } ^ { m } } }$ , where $\{ i _ { 1 } , \dots , i _ { m } \} \subset \{ 1 , \dots , n \}$ are randomly selected indices from the input set $\{ { \pmb x } _ { i } \} _ { i = 1 } ^ { n }$ . Performing the spectral algorithm within this subspace  corresponds to the Nyström sub-sampling method. More generally, we may consider a random matrix $\mathbf { G } \in \mathbb { R } ^ { m \times n }$ and define the random subspace as $\begin{array} { r } { \mathcal { F } = \operatorname { s p a n } \{ \sum _ { j = 1 } ^ { n } \mathbf { G } _ { i j } K _ { x _ { i } } , i = 1 , \dots , m \} } \end{array}$ , which leads to the randomized sketching method. By choosing $m \ll n ,$ , we can achieve substantial reductions in both time and space complexity while maintaining the statistical eficiency of the original spectral algorithms (Williams and Seeger, 2001; Kumar et al., 2012; Mahoney et al., 2011).

Theoretical analysis of the generalization properties for kernel methods with specific random projection techniques has been extensively studied in the literature. In the fixed design, where the inputs $\{ { \pmb x } _ { i } \} _ { i = 1 } ^ { n }$ are deterministic, Yang et al. (2017) analyzed KRR with randomized sketches and established optimal learning rates in the worst case where the regression function $f _ { \rho }$ lies in the RKHS (� = 1 and $\gamma = 0 )$ . Similarly, sharp generalization error bounds in expectation for the Nyström sub-sampling method were derived in Bach (2013); Alaoui and Mahoney (2015). In the random design setting, where the inputs $\{ { \pmb x } _ { i } \} _ { i = 1 } ^ { n }$ are drawn i.i.d. from an unknown distribution, Rudi, Camoriano and Rosasco (2015); Myleiko, Pereverzyev Jr and Solodky (2019) focused on KRR and general spectral algorithms with Nyström sub-sampling, and established source and capacity dependent learning rates in high probability. However, these results are limited to the well-specified case where $f _ { \rho }$ lies in the RKHS $( s \in [ 1 , 2 \tau ]$ with � denoting the smoothness index and � denoting the qualification index), and the generalization error is measured via the $L _ { \rho } ^ { 2 }$ norm. The most closely related work is Lin and Cevher (2020a), which investigated spectral algorithms with random projections in the mis-specified setting where $f _ { \rho }$ may not lie in the RKHS. They derived optimal learning rates in various norms when $s \ge 1 - 1 / \beta$ $( \beta \in ( 1 , \infty )$ with $\beta$ denoting the capacity index), thus covering part of the mis-specified setting $( s \in ( 0 , 1 ) )$ . However, their analysis relies on restrictive assumptions, such as the Hölder-type source condition and uniform boundedness of $f _ { \rho } ,$ which may not hold in practice. Until now, it remains an open question whether optimal learning rates of the spectral algorithms with random projections can be achieved when $s < 1 - 1 / \beta$

In this paper, we provide a comprehensive and improved analysis of the generalization performance for the spectral algorithms with random projections (SARP) in the mis-specified setting under general conditions without sacrificing statistical eficiency. The following are the main contributions of this paper:

General conditions. Our analysis relies on five key assumptions: (i) Bernstein condition on the noise without requiring the uniform boundedness assumption on $f _ { \rho }$ (Assumption 1); (ii) capacity condition (Assumptions 2 and 5) and embedding condition (Assumption 3); (iii) general source condition on the regression function $f _ { \rho }$ (Assumption 4), of which the Hölder source condition is a special case. Compared with prior studies on RP-based spectral algorithms (Yang et al., 2017; Rudi et al., 2015; Lin and Cevher, 2020a), our assumptions are either more general or more readily satisfied in practice.

Comprehensive and improved learning rates. Under these more general assumptions, we first establish the minimax lower bounds in Sobolev norms for all $s > 0$ in Theorem 1. For the upper bounds, we combine the integral-operator technique with the embedding property of the RKHS to derive sharp learning rates for general SARP in Theorem 2. By appropriately choosing the regularization parameter �, we show that the upper bound is minimax optimal provided the projection error term is of smaller order than the typical approximation and empirical error terms. In Corollary 2, we further prove that this optimality is achievable for both randomized sketches and Nyström sub-sampling, when the projection dimension � is proportional to the average or maximal efective dimension (up to logarithmic factors). Compared to the state-of-the-art results in Lin and Cevher (2020a), our contribution primarily lies in extending the optimality range from $s \ge 1 - 1 / \beta$ to $s > \alpha _ { 0 } - 1 / \beta$ for $\alpha _ { 0 } \in [ 1 / \beta , 1 ]$ , rather than improving the convergence exponents already established, as summarized in Figure 1. Moreover, our analysis removes the uniform boundedness assumption on $f _ { \rho }$ by imposing the Bernstein condition directly on the noise rather than on the response variable. We also extend the analysis from traditional Hölder source conditions to more general source conditions. Regarding the technica analysis, the random projection constructions and their associated concentration bounds largely build on existing studies (Rudi et al., 2015; Lin and Cevher, 2020a). Our new technical contributions lie in the refined operator similarity and empirical error analysis, which enable optimal learning rates under the enlarged source range and weaker boundedness requirements. These improvements also yield smaller suficient projection dimensions for plain Nyström sampling in certain settings. Further details are provided in Section II of the Supplementary Material.

Eficient computational improvement. From a computational perspective, our results show that the projection dimension � required to achieve the optimal learning rates is no larger (and can even be smaller in some cases) than those required in Yang et al. (2017); Rudi et al. (2015); Lin and Cevher (2020a). This leads to a substantial reduction in computational cost while preserving statistical eficiency in a broader source range. The improvement stems from our more refined analysis of operator similarity and sample variance control.

![](images/a21819cb723166c16b9c4dfb1e63d118f4b975d356eb3b0e7103eb83c55a7add.jpg)

![](images/41809d8e6be617785cfc559236d1e501cb1c886b9ea3940bc49557ab2dd24468.jpg)

![](images/3bbb3eab8705610203519cfd8bbaf50467fa633f271fa1e8d656b73847fc870f.jpg)  
Figure 1: Illustration of the source range for optimality in our work. Panels from left to right represent the worst case $\left( \alpha = 1 \right)$ , general case $\left( \alpha \in [ 1 / \beta , 1 ] \right)$ , and benign case $\left( \alpha _ { 0 } = 1 / \beta \right)$ , respectively.

For a better understanding of our contribution and novelty, we provide a detailed comparison with existing methods in Section II of the Supplementary Material.

Notations. For two positive sequences $\{ a _ { n } \} _ { n \geq 1 }$ and $\{ b _ { n } \} _ { n > 1 }$ , we write $a _ { n } \lesssim b _ { n }$ if there exists a constant $C > 0$ such that $a _ { n } \leq C b _ { n }$ . We write $a _ { n } \asymp b _ { n }$ if both $a _ { n } \lesssim b _ { n }$ and $b _ { n } \lesssim a _ { n }$ hold. For $a , b \in \mathbb { R }$ , let $a \wedge b = \operatorname* { m i n } \{ a , b \}$ $a \vee b = \operatorname* { m a x } \{ a , b \}$ , and $a _ { + } = \operatorname* { m a x } \{ a , 0 \}$ . For a bounded linear operator �, we denote its operator norm, adjoint, and range by $\| A \| , A ^ { * }$ , and Ran(�), respectively. For a trace-class operator �, Tr(�) denotes its trace. The symbol � denotes the identity operator or matrix on the relevant space. For �, � in a Hilbert space , the rank-one operator � ⊗ � is defined by $( v \otimes u ) w = \langle u , w \rangle _ { \mathcal { H } } v$ for any $w \in \mathcal { H }$ . The constants � and � may vary from line to line.

## 2. Preliminaries

Let (, ) be a Borel measure space as the input space, and $\mathcal { V } \subset \mathbb { R }$ be the output space. Denote by � the joint distribution on $\mathcal { X } \times \mathcal { V }$ , which can be decomposed as $\rho ( { \pmb x } , y ) = \rho _ { \mathscr { X } } ( { \pmb x } ) \rho ( y | { \pmb x } )$ with $\rho ( y | \pmb { x } )$ being the conditional distribution of � given �. In a standard supervised learning framework, we observe a set of training samples $\mathbf { z } = \{ ( \boldsymbol { { \mathbf { x } } } _ { i } , \boldsymbol { { y } } _ { i } ) \} _ { i = 1 } ^ { n }$ drawn i.i.d. from the unknown distribution $\rho .$ The objective is to find a function $f$ that minimizes the expected risk over the space of all the measurable functions, which is defined as

$$
\mathcal { E } ( f ) = \int _ { \mathcal { X } \times \mathcal { Y } } \big ( f ( \pmb { x } ) - y \big ) ^ { 2 } d \rho ( \pmb { x } , y ) .\tag{1}
$$

The minimizer of the expected risk is the regression function $\begin{array} { r } { f _ { \rho } ( { \pmb x } ) = \int _ { \mathcal { Y } } y d \rho ( y | { \pmb x } ) } \end{array}$ . Consequently, we can define the following nonparametric regression model:

$$
y = f _ { \rho } ( x ) + \varepsilon ,\tag{2}
$$

where � is the noise satisfying $\mathbb { E } ( \varepsilon \vert \pmb { x } ) = 0$

## 2.1. RKHS and Operators

Consider a bounded and measurable kernel $K ~ : ~ \mathcal { X } \times \mathcal { X } \ \to \ \mathbb { R }$ that is symmetric and positive definite, i.e., sup<sub>�∈</sub> $K ( \pmb { x } , \pmb { x } ) \le \kappa ^ { 2 } , K ( \pmb { x } , \pmb { x } ^ { \prime } ) = K ( \pmb { x } ^ { \prime } , \pmb { x } )$ for $\pmb { x } , \pmb { x } ^ { \prime } \in \mathcal { X }$ , and $\begin{array} { r } { \sum _ { i , j = 1 } ^ { n } c _ { i } c _ { j } K ( { \pmb x } _ { i } , { \pmb x } _ { j } ) \geq 0 } \end{array}$ for any $n \in \mathbb { N } , \{ c _ { i } \} _ { i = 1 } ^ { n } \subset \mathbb { R }$ and $\{ \pmb { x } _ { i } \} _ { i = 1 } ^ { n } \subset \mathscr { X }$ . Without loss of generality, we take $\kappa \geq 1$ . This kernel is associated with a unique reproducing kernel Hilbert space (RKHS) $\mathcal { H } = \mathcal { H } _ { \mathcal { K } }$ equipped with inner product $\langle \cdot , \cdot \rangle _ { \mathscr { H } }$ and the norm $\| \cdot \| _ { \mathcal { H } }$ . The RKHS is the completion of the space of finite linear combinations of the kernel functions $\{ K _ { x } = K ( \pmb { x } , \cdot ) : \pmb { x } \in \pmb { \mathcal { X } } \}$ with respect to the inner product $\langle K _ { x } , K _ { x ^ { \prime } } \rangle _ { \mathcal { H } } = K ( \pmb { x } , \pmb { x } ^ { \prime } )$ . For any $f \in \mathcal { H } _ { K }$ , the reproducing property holds:

$$
f ( \pmb { x } ) = \langle f , K _ { \pmb { x } } \rangle _ { \mathcal { H } } , \quad \forall \pmb { x } \in \pmb { \mathcal { X } } .
$$

Since � is bounded, $K \in L ^ { 2 } ( \rho _ { X } \otimes \rho _ { X } )$ . Consequently, the integral operator $L _ { K }$ is compact, positive, and self-adjoint, and admits a spectral decomposition in $L _ { \rho _ { X } } ^ { 2 }$ . We do not require  to be compact or � to satisfy the classical Mercer conditions.

Let $L _ { \rho _ { \mathcal { X } } } ^ { 2 }$ be the space of square-integrable functions from  to ℝ with respect to $\rho _ { \mathcal { X } }$ , with its norm given by $\begin{array} { r } { \| f \| _ { \rho } = \sqrt { \int _ { \mathcal { X } } f ^ { 2 } ( \pmb { x } ) d \rho _ { \mathcal { X } } ( \pmb { x } ) } } \end{array}$ . Define the inclusion operator $S _ { K } : \mathcal { H } _ { K } \to L _ { \rho _ { \mathcal { X } } } ^ { 2 }$ as $S _ { K } f ( { \pmb x } ) = \langle f , K _ { \pmb x } \rangle _ { \mathcal { H } }$ , for any $f \in \mathcal { H } _ { K }$ and $\boldsymbol { x } \in \dot { \mathcal { X } }$ . We further assume that $S _ { K }$ is injective, i.e., ker $S _ { K } = \{ 0 \}$ . Throughout the paper, we identify $\mathcal { H } _ { K }$ with $S _ { K } ( \mathcal { H } _ { K } )$ in $L _ { \rho _ { \mathcal { X } } } ^ { 2 }$ . Its adjoint operator is denoted by $\begin{array} { r } { S _ { K } ^ { * } : L _ { \rho _ { \mathcal { X } } } ^ { 2 } \to \mathcal { H } _ { K } , S _ { K } ^ { * } g = \int _ { \mathcal { X } } K _ { x } g ( { \pmb x } ) d \rho _ { \mathcal { X } } ( { \pmb x } ) } \end{array}$ , for any $g \in L _ { \rho _ { \mathcal { X } } } ^ { 2 }$ and $\boldsymbol { x } \in \mathcal { X }$ . Then $S _ { K }$ and $S _ { K } ^ { * }$ are Hilbert-Schmidt operators and thus compact. Furthermore, we can define the integral operator $L _ { K }$ and the covariance operator $C _ { K }$ as follows:

$$
\begin{array} { l l l } { { \displaystyle { \cal L } _ { K } : { \cal L } _ { \rho _ { \mathcal { X } } } ^ { 2 }  { \cal L } _ { \rho _ { \mathcal { X } } } ^ { 2 } , } } & { { \displaystyle { \cal L } _ { K } g = \int _ { \mathcal { X } } K ( { \bf x } , \cdot ) g ( { \bf x } ) d \rho _ { \mathcal { X } } ( { \bf x } ) , } } \\ { { \displaystyle { \cal C } _ { K } : \mathcal { H } _ { K }  \mathcal { H } _ { K } , } } & { { \displaystyle { \cal C } _ { K } = \int _ { \mathcal { X } } \langle \cdot , K _ { { \bf x } } \rangle _ { \mathcal { H } } K _ { { \bf x } } d \rho _ { \mathcal { X } } ( { \bf x } ) = \int _ { \mathcal { X } } K _ { { \bf x } } \otimes K _ { { \bf x } } d \rho _ { \mathcal { X } } ( { \bf x } ) . } } \end{array}\tag{3}
$$

Since � is bounded on $\scriptstyle { \mathcal { X } } ,$ , the operators $L _ { K }$ and $C _ { K }$ are positive, self-adjoint, and trace-class (hence Hilbert–Schmidt and compact) (Steinwart and Christmann, 2008). They satisfy the identities ${ \cal L } _ { K } = S _ { K } S _ { K } ^ { * }$ and $C _ { K } = S _ { K } ^ { * } S _ { K }$ . Moreover, as $\begin{array} { r } { \operatorname* { s u p } _ { x \in \mathcal { X } } K ( \pmb { x } , \pmb { x } ) \le \kappa ^ { 2 } } \end{array}$ , we have $\| S _ { K } \| = \| S _ { K } ^ { * } \| \le \kappa$ and $\| L _ { K } \| = \| C _ { K } \| \leq \kappa ^ { 2 }$ . For any $f \in \mathcal { H } _ { K }$ , the following isometry property holds (Steinwart and Christmann, 2008):

$$
\Vert f \Vert _ { \rho } = \Vert S _ { K } f \Vert _ { \rho } = \Vert C _ { K } ^ { 1 / 2 } f \Vert _ { \mathcal { H } } .\tag{4}
$$

In addition, by the polar decomposition of $S _ { K } , \| L _ { K } ^ { - 1 / 2 } S _ { K } f \| _ { \rho } \leq \| f \| _ { \mathcal { H } }$

By the spectral theorem, there exists a sequence of non-negative and non-increasing eigenvalues $\{ \mu _ { i } \} _ { i \ge 1 }$ , orthonormal basis $\{ e _ { i } \} _ { i \ge 1 }$ of ${ \overline { { \mathrm { R a n } ( S _ { K } ) } } } \subseteq L _ { \rho _ { \mathcal { X } } } ^ { 2 }$ , and orthonormal basis $\{ \mu _ { i } ^ { 1 / 2 } e _ { i } \} _ { i \geq 1 }$ of $\mathcal { H } _ { K }$ such that

$$
L _ { K } = \sum _ { i = 1 } ^ { \infty } \mu _ { i } \langle \cdot , e _ { i } \rangle _ { \rho } e _ { i } , C _ { K } = \sum _ { i = 1 } ^ { \infty } \mu _ { i } \langle \cdot , \mu _ { i } ^ { 1 / 2 } e _ { i } \rangle _ { \mathcal { H } } \mu _ { i } ^ { 1 / 2 } e _ { i } .\tag{5}
$$

Moreover, the kernel admits the spectral representation $\begin{array} { r } { K ( x , x ^ { \prime } ) = \sum _ { i = 1 } ^ { \infty } \mu _ { i } e _ { i } ( x ) e _ { i } ( x ^ { \prime } ) } \end{array}$ with convergence in $L ^ { 2 } ( \rho _ { X } \otimes \rho _ { X } )$ Note that the eigenvalues $\{ \mu _ { i } \} _ { i \ge 1 }$ and eigenfunctions $\{ e _ { i } \} _ { i \ge 1 }$ depend on the kernel � and the marginal distribution $\rho _ { \mathcal { X } }$

## 2.2. Interpolation Space of RKHS

In this section, we introduce the interpolation space (or power space) of the RKHS $\varkappa _ { K }$ . For any $t \geq 0$ , we define the fractional integral operator $L _ { K } ^ { t }$ as $\begin{array} { r } { L _ { K } ^ { t } = \sum _ { i = 1 } ^ { \infty } \mu _ { i } ^ { t } \langle \cdot , e _ { i } \rangle _ { \rho } e _ { i } } \end{array}$

Definition 1. Given the RKHS $\mathcal { H } _ { K }$ associated with the kernel � and the integral operator $L _ { K }$ , for any $t \geq 0$ , we define the interpolation space $[ \mathcal { H } _ { K } ] _ { \rho } ^ { t }$ as

$$
[ \mathcal { H } _ { K } ] _ { \rho } ^ { t } : = { \mathrm { R a n } L } _ { K } ^ { t / 2 } = \left\{ \sum _ { i = 1 } ^ { \infty } a _ { i } \mu _ { i } ^ { t / 2 } e _ { i } : \sum _ { i = 1 } ^ { \infty } a _ { i } ^ { 2 } < \infty \right\} \subset L _ { \rho _ { \mathscr { X } } } ^ { 2 } .\tag{6}
$$

Let $f , g \ \in \ [ \mathcal { H } _ { K } ] _ { o } ^ { t }$ and $\begin{array} { r } { f \ = \ \sum _ { i = 1 } ^ { \infty } a _ { i } \mu _ { i } ^ { t / 2 } e _ { i } , g \ = \ \sum _ { i = 1 } ^ { \infty } b _ { i } \mu _ { i } ^ { t / 2 } e _ { i } , } \end{array}$ , define the inner product and norm of $[ \mathcal { H } _ { K } ] _ { \rho } ^ { t }$ as $\begin{array} { r } { \langle f , g \rangle _ { [ \mathcal { H } ] _ { \rho } ^ { t } } = \langle L _ { K } ^ { - t / 2 } f , L _ { K } ^ { - t / 2 } g \rangle _ { \rho } = \sum _ { i = 1 } ^ { \infty } a _ { i } b _ { i } , \| f \| _ { [ \mathcal { H } ] _ { \rho } ^ { t } } = \langle f , f \rangle _ { [ \mathcal { H } ] _ { \rho } ^ { t } } ^ { 1 / 2 } = ( \sum _ { i = 1 } ^ { \infty } a _ { i } ^ { 2 } ) ^ { 1 / 2 } f , } \end{array}$

The interpolation space $[ \mathcal { H } _ { K } ] _ { \rho } ^ { t }$ is also a separable Hilbert space with orthonormal basis $\{ \mu _ { i } ^ { t / 2 } e _ { i } \} _ { i \geq 1 }$ . It follows immediately that $[ \mathcal { H } _ { K } ] _ { \rho } ^ { 1 } \ = \ \mathcal { H } _ { K }$ and $[ \mathcal { H } _ { K } ] _ { \rho } ^ { 0 } \ \subseteq \ L _ { \rho _ { \mathcal { X } } } ^ { 2 }$ . For any $0 ~ \le ~ t _ { 1 } ~ \le ~ t _ { 2 }$ , there is a continuous embedding $[ \mathcal { H } _ { K } ] _ { \rho } ^ { t _ { 2 } } \hookrightarrow [ \mathcal { H } _ { K } ] _ { \rho } ^ { t _ { 1 } } \hookrightarrow [ \mathcal { H } _ { K } ] _ { \rho } ^ { 0 }$ (Fischer and Steinwart, 2020). The interpolation space $[ \mathcal { H } _ { K } ] _ { \rho } ^ { s }$ provides a natural framework for describing the regularity of the regression function $f _ { \rho } .$ . For example, if $f _ { \rho } \in [ \mathcal { H } _ { K } ] _ { \rho } ^ { t }$ for some $t > 1$ , then $f _ { \rho }$ is smoother than functions in the RKHS $\textstyle \mathcal { H } _ { \mathcal { K } } ;$ Conversely, when $0 < t < 1$ , the condition $f _ { \rho } \in [ \mathcal { H } _ { K } ] _ { \rho } ^ { t }$ is weaker and may allow $f _ { \rho } \notin \mathcal { H } _ { K }$

For $0 < t < 1$ , the interpolation space $[ \mathcal { H } _ { K } ] _ { \rho } ^ { t }$ can be characterized by the real interpolation method (Tsybakov, 2008; Steinwart and Scovel, 2012; Bitzer and Steinwart, 2025). Specifically, we have

$$
[ \mathcal { H } _ { K } ] _ { \rho } ^ { t } \cong [ L _ { \rho _ { \chi } } ^ { 2 } , [ \mathcal { H } _ { K } ] _ { \rho } ^ { 1 } ] _ { t , 2 } ,\tag{7}
$$

where $\cong$ denotes the isomorphism between the two spaces and $[ \cdot , \cdot ] _ { t , 2 }$ is the real interpolation functor of two function spaces.

## 2.3. General Source Condition

In statistical learning theory, it is standard to restrict the function class of the underlying problem. A common approach is to consider a family of probability measures ${ \cal P } ( \Omega )$ whose associated regression function satisfies $f _ { \rho } \in \Omega$ where Ω is a hypothesis set. Typically, Ω is compact and specified by regularity (smoothness) conditions. In the context of kernel methods, a widely used regularity condition is the Hölder source condition (Caponnetto and De Vito, 2007; Steinwart, Scovel et al., 2009; Guo et al., 2017) in terms of the integral operator $L _ { K }$

$$
f _ { \rho } \in \Omega _ { s , R } : = \left\{ f \in L _ { \rho _ { \mathscr { X } } } ^ { 2 } : f = L _ { K } ^ { s / 2 } g , \| g \| _ { \rho } \leq R \right\} ,\tag{8}
$$

where $s > 0$ is the smoothness parameter. When $s = 1$ , the source condition means that $f _ { \rho }$ admits a representative in $\mathcal { H } _ { K }$ , with minimum RKHS norm at most �. We denote the function class of $f _ { \rho }$ as $\mathcal { F } = L _ { K } ^ { s / 2 } ( L _ { \rho _ { \mathcal { X } } } ^ { 2 } )$ , and note that $\mathcal { H } _ { K } = L _ { K } ^ { 1 / 2 } ( L _ { \rho _ { \mathcal { X } } } ^ { 2 } ) .$ , and $L _ { K } ^ { s _ { 1 } / 2 } ( L _ { \rho _ { \mathcal { X } } } ^ { 2 } ) \subseteq L _ { K } ^ { s _ { 2 } / 2 } ( L _ { \rho _ { \mathcal { X } } } ^ { 2 } )$ (Steinwart and Christmann, 2008) for $s _ { 1 } \geq s _ { 2 }$ . When $s \geq 1$ , the functional class $\scriptstyle \mathbf { \dot { \boldsymbol { { F } } } }$ is a subset of the assumed RKHS $\mathcal { H } _ { K }$ , so we have $f _ { \rho } \in \mathcal { H } _ { K }$ (well-specified case). When $s \in ( 0 , 1 )$ , the functional class $_ { \mathcal { F } }$ is larger than the assumed RKHS $\mathcal { H } _ { K }$ , and there exist some cases where $f _ { \rho } \notin \mathcal { H } _ { K }$ (mis-specified case). By (6), the source condition in (8) is equivalent to $f _ { \rho } \in [ \mathcal { H } ] _ { \rho } ^ { s }$ and $\| f _ { \rho } \| _ { [ \mathcal { H } ] _ { o } ^ { s } } \leq R$

The Hölder source condition characterizes the regularity of $f _ { \rho }$ through the decay of the coeficients in the eigenbasis of $L _ { K }$ . Specifically, if $\{ ( \mu _ { i } , e _ { i } ) \} _ { i \geq 1 }$ are the eigenpairs of $L _ { K }$ , then $f _ { \rho }$ admits the expansion $\begin{array} { r } { f _ { \rho } = \sum _ { i = 1 } ^ { \infty } \langle f _ { \rho } , e _ { i } \rangle _ { \rho } e _ { i } } \end{array}$ . The Hölder source condition implies that the coeficients $\langle f _ { \rho } , e _ { i } \rangle _ { \rho }$ decay at a rate determined by �, i.e., $\begin{array} { r } { \sum _ { i = 1 } ^ { \infty } \frac { \langle f _ { \rho } , e _ { i } \rangle _ { \rho } ^ { 2 } } { \mu _ { i } ^ { s } } \leq R ^ { 2 } } \end{array}$ which is known as Picard’s criterion (Bauer, Pereverzev and Rosasco, 2007). A natural generalization is to replace the power function $\mu _ { i } ^ { s }$ by a more flexible index function $\phi ( \cdot )$ , leading to the more general condition $\begin{array} { r } { \sum _ { i = 1 } ^ { \infty } \frac { \langle f _ { \rho } , e _ { i } \rangle _ { \rho } ^ { 2 } } { \phi ^ { 2 } ( \mu _ { i } ) } \leq R ^ { 2 } } \end{array}$ Consequently, we define the general source condition as

$$
f _ { \rho } \in \Omega _ { \phi , s , R } : = \left\{ f \in L _ { \rho _ { \mathscr { X } } } ^ { 2 } : f = \phi ( L _ { K } ) g , \| g \| _ { \rho } \leq R , \phi \in C _ { s } \right\} ,
$$

where $c _ { s }$ is the index function class. In this paper, we consider

$C _ { s } = \{ \phi : [ 0 , \kappa ^ { 2 } ]  \mathbb { R } ^ { + } \bigg | \phi $ is continuous and non-decreasing, $\phi ( 0 ) = 0 , 0 < \phi ( t ) < \infty \mathrm { f o r } t \in ( 0 , \kappa ^ { 2 } ]$

and $\frac { \phi ^ { 2 } ( t ) } { t ^ { s } }$ is non-decreasing for some $s > 0 \}$

(9)

A typical example is the Hölder source condition, which corresponds to the choice $\phi ( t ) = t ^ { s / 2 } \in C _ { s }$

## 3. Spectral Algorithms and Random Projections

Spectral algorithms (SA) comprise a family of regularization methods that leverage the spectral operator through filter functions to stabilize learning problems (Engl et al., 1996). Initially developed for ill-posed linear inverse problems (De Vito, Rosasco, Caponnetto, De Giovannini and Odone, 2005), they have been subsequently extended to nonparametric regression via the connection between learning theory and inverse problems (Rosasco, De Vito and Verri, 2005; De Vito, Rosasco and Caponnetto, 2006; Guo et al., 2017).

Suppose that we have a training dataset $\{ ( \pmb { x } _ { i } , \pmb { y } _ { i } ) \} _ { i = 1 } ^ { n }$ drawn i.i.d. from an unknown distribution $\rho .$ Define the sampling operator

$$
S _ { K , n } : \mathcal { H } _ { K } \to \mathbb { R } ^ { n } , \quad S _ { K , n } f = \frac { 1 } { \sqrt { n } } \big ( \langle f , K _ { x _ { 1 } } \rangle _ { \mathcal { H } } , \dots , \langle f , K _ { x _ { n } } \rangle _ { \mathcal { H } } \big ) ,
$$

and its adjoint operator

$$
S _ { K , n } ^ { * } : \mathbb { R } ^ { n }  \mathcal { H } _ { K } , \quad S _ { K , n } ^ { * } \alpha = \frac { 1 } { \sqrt { n } } \sum _ { i = 1 } ^ { n } \alpha _ { i } K _ { x _ { i } } ,
$$

Several common spectral algorithms, along with their corresponding filter functions, qualifications, and the associated constants.
<table><tr><td>Method</td><td>Filter Function  $g _ { \lambda } ( t )$ </td><td>Parameters</td><td>Qualification τ</td><td> $\overline { { ( E , F _ { \tau } ) } }$ </td></tr><tr><td>Tikhonov Regularization (KRR)</td><td> $\frac { 1 } { t _ { \prime } + \lambda }$ </td><td> $\lambda > 0$ </td><td>1</td><td>(1, 1)</td></tr><tr><td>Iterative Regularization</td><td> $\sum _ { k = 1 } \lambda ^ { k - 1 } ( \lambda + t ) ^ { - k }$ </td><td> $l \in \mathbb N$ </td><td>1</td><td>(l, 1)</td></tr><tr><td>Gradient methods (GM)</td><td> $\sum _ { k = 1 } \eta ( 1 - \eta t ) ^ { l - k }$ </td><td> $\begin{array} { r } { \eta \in ( 0 , \kappa ^ { 2 } ] , \ \lambda = \frac { 1 } { \eta l } } \end{array}$ </td><td> $\mathsf { a n y } \tau > 0$ </td><td> $\left( 1 , \operatorname* { m a x } \{ 1 , ( \tau / e ) ^ { \tau } \} \right)$ </td></tr><tr><td>Spectral cutoff (SC)</td><td> $\left\{ \begin{array} { l l } { 1 / t , } & { t \geq \lambda , } \\ { 0 , } & { \mathrm { o t h e r w } } \end{array} \right.$  ise</td><td> $\lambda > 0$ </td><td> $\mathsf { a n y } \tau > 0$ </td><td>(1, 1)</td></tr></table>

for $\pmb { \alpha } = ( \alpha _ { 1 } , \alpha _ { 2 } , \ldots , \alpha _ { n } ) ^ { T } \in \mathbb { R } ^ { n }$ . The empirical covariance operator is then

$$
C _ { K , n } : \mathcal { H } _ { K } \to \mathcal { H } _ { K } , \quad C _ { K , n } = S _ { K , n } ^ { * } S _ { K , n } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } K _ { x _ { i } } \otimes K _ { x _ { i } } .
$$

Our goal is to learn an estimator $\hat { f } \in \mathcal { H } _ { K }$ from the training data that approximates the true regression function $f _ { \rho }$ defined in (2), i.e. the empirical risk minimizer, $\begin{array} { r } { \hat { f } = \operatorname * { a r g m i n } _ { f \in { \mathcal H } _ { K } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( f ( { x } _ { i } ) - { y } _ { i } ) ^ { 2 } } \end{array}$ . Using the operator notations introduced above, the minimizer should satisfy the normal equation $C _ { K , n } \hat { f } = S _ { K , n } ^ { * } \mathbf { y }$ , where $\begin{array} { r } { { \mathbf y } = \frac { 1 } { \sqrt { n } } ( y _ { 1 } , y _ { 2 } , \ldots , y _ { n } ) ^ { T } } \end{array}$ However, directly solving the normal equation is prone to overfitting, and the operator $C _ { K , n }$ is often singular.

To address these issues, spectral algorithms introduce a regularization mechanism by applying a filter function to the empirical covariance operator. This stabilizes the solution and mitigates overfitting through the regularization efect of the filter function. Specifically, a spectral algorithm produces an estimator $f _ { n , \lambda }$ of the form

$$
f _ { n , \lambda } = g _ { \lambda } ( C _ { K , n } ) S _ { K , n } ^ { * } { \bf y } ,\tag{10}
$$

where $\lambda > 0$ is a regularization parameter and $g _ { \lambda } : [ 0 , \kappa ^ { 2 } ] \to \mathbb { R }$ is a filter function satisfying certain conditions (see Definition 2 below).

Definition 2. A function $g _ { \lambda } ( \cdot ) : [ 0 , \kappa ^ { 2 } ] \to \mathbb { R } ^ { + }$ indexed by $\lambda \in \Lambda \subset \mathbb { R } ^ { + }$ is called a filter function with qualification $\tau \geq 1$ , if it satisfies the following conditions:

$$
\operatorname* { s u p } _ { \nu \in [ 0 , 1 ] } \operatorname* { s u p } _ { \lambda \in \Lambda } \operatorname* { s u p } _ { t \in [ 0 , \kappa ^ { 2 } ] } | t ^ { \nu } g _ { \lambda } ( t ) | \lambda ^ { 1 - \nu } \leq E , \quad \operatorname* { s u p } _ { \nu \in [ 0 , \tau ] } \operatorname* { s u p } _ { \lambda \in \Lambda } \operatorname* { s u p } _ { t \in [ 0 , \kappa ^ { 2 } ] } | ( 1 - t g _ { \lambda } ( t ) ) | t ^ { \nu } \lambda ^ { - \nu } \leq F _ { \tau } ,\tag{11}
$$

where � and $F _ { \tau }$ are positive constants independent of � and �.

Diferent choices of the filter function $g _ { \lambda }$ yield diferent spectral regularization schemes. In Table 1, we list several common spectral algorithms together with their corresponding filter functions, qualifications, and the associated constants $( E , F _ { \tau } )$

Despite the theoretical appeal of spectral algorithms, their practical implementation can be computationally intensive, especially for large-scale datasets. The primary computational bottleneck arises from computing and storing the empirica covariance operator $C _ { K , n } .$ , which involves handling an $n \times n$ kernel matrix $( \mathrm { i } . \mathrm { e } . , n \times n$ matrix inverse in KRR). This results in significant time and space complexity $( \mathrm { i } . \mathrm { e } . , O ( n ^ { 3 } )$ and $O ( n ^ { 2 } )$ for KRR), making it infeasible for large �. To alleviate this burden, various approximation techniques have been proposed to reduce the computational burden. One such technique is the use of random projections, which can efectively reduce the dimensionality of the data while preserving its essential structure (Vempala, 2005). The core idea of random projections is to restrict the original solution space to a lower-dimensional subspace using a projection operator �, satisfying $P = P ^ { * } = P ^ { 2 }$ . Incorporating random projections into spectral algorithms leads to the following estimator:

$$
f _ { n , \lambda } ^ { r p } = P g _ { \lambda } ( P C _ { K , n } P ) P S _ { K , n } ^ { * } \mathbf { y } .\tag{12}
$$

This approach can significantly reduce the computational complexity of spectral algorithms, making them more scalable and eficient for large datasets. In the next subsections, we will discuss some specific implementations of SARP, focusing on randomized sketches and Nyström sub-sampling.

## 3.1. Randomized Sketches for Spectral Algorithms

Randomized sketches involve a sketch matrix $\mathbf { G } \in \mathbb { R } ^ { m \times n }$ with $m \ll n ,$ and the range of the projection operator � is defined as the closure of the span of the rows of ${ \bf G } S _ { K , n } , \mathrm { i . e . }$ ., Range $\mathbf { \xi } ^ { \prime } P ) = \operatorname { s p a n } \{ S _ { K , n } ^ { * } \mathbf { G } ^ { T } \}$ . The sketch matrix � is drawn independently of the training sample and is isotropically normalized, $\mathrm { i . e . , } \mathbb { E } _ { G } [ G ^ { \top } G ] = I _ { n }$ . We assume that for any fixed nonzero � $\in \mathbb { R } ^ { n }$ and $t \in ( 0 , 1 )$ , there exist constants $C > 0$ and $b \geq 0$ , independent of �, �, and �, such that the following two-sided concentration inequality holds:

$$
\mathbb { P } _ { G } \left( \left| \| G \mathbf { a } \| _ { 2 } ^ { 2 } - \| \mathbf { a } \| _ { 2 } ^ { 2 } \right| \geq t \| \mathbf { a } \| _ { 2 } ^ { 2 } \right) \leq 2 \exp \left( - \frac { C m t ^ { 2 } } { ( \log n ) ^ { b } } \right) ,\tag{13}
$$

Common choices for � include sub-Gaussian random sketches, randomized orthogonal system (ROS) sketches, and sub-sampling sketches (Yang et al., 2017; Lin and Cevher, 2020a).

Sub-Gaussian random sketches. A sketch matrix $\mathbf { G } \in \mathbb { R } ^ { m \times n }$ is called a normalized sub-Gaussian sketch if its entries $\{ g _ { i j } \} _ { i , j = 1 } ^ { m , n }$ are independent, mean-zero random variables satisfying $\mathbb { E } [ g _ { i j } ^ { 2 } ] = 1 / m$ and the sub-Gaussian condition $\mathbb { E } \exp ( t g _ { i j } ) \le \exp \left( c ^ { 2 } t ^ { 2 } / 2 m \right) , t \in \mathbb { R }$ , for some universal constant $c > 0$ . The normalization ensures that $\mathbb { E } [ \mathbf { G } ^ { \top } \mathbf { G } ] = I _ { n }$ Such sketches satisfy the two-sided concentration inequality (13) with $b = 0$ � = 0 . Typical examples include matrices with i.i.d. Gaussian entries distributed as $\mathcal { N } ( 0 , 1 / m )$ or i.i.d. Rademacher (Bernoulli ±1) entries scaled by $1 / \sqrt { m }$ . Closely related constructions draw independent rows uniformly from a sphere of radius ${ \sqrt { n / m } }$ . These rows are isotropic and sub-Gaussian and provide comparable sketching guarantees (Yang et al., 2017).

Randomized orthogonal system (ROS) sketches. A ROS sketch matrix $\mathbf { G } \in \mathbb { R } ^ { m \times n }$ is constructed as $\mathbf { G } = \sqrt { n / m } \mathbf { R } \mathbf { H } \mathbf { D }$ where $\mathbf { D } \in \mathbb { R } ^ { n \times n }$ is a diagonal matrix with i.i.d. Rademacher entries, $\mathbf { H } \in \mathbb { R } ^ { n \times n }$ is an orthonormal matrix (e.g., Hadamard or Fourier matrix), and $\mathbf { R } \in \mathbb { R } ^ { m \times n }$ is a random sampling matrix that selects � rows uniformly at random from the $n \times n$ identity matrix. ROS sketches are computationally eficient due to the fast multiplication algorithms available for orthonormal matrices like Hadamard and Fourier transforms. Typically, a matrix-vector multiplication with a ROS sketch can be performed in $O ( n$ log �) time, which is significantly faster than the $O ( m n )$ time required for dense random matrices (Ailon and Liberty, 2009).

Sub-sampling sketches. A sub-sampling sketch matrix $\mathbf { G } \in \mathbb { R } ^ { m \times n }$ is obtained by selecting � rows uniformly at random without replacement from the $n \times n$ identity matrix. This means that each row of � contains exactly one entry equal to ${ \sqrt { n / m } }$ , and all other entries are zero. Unlike normalized sub-Gaussian and ROS sketches, coordinate subsampling does not generally satisfy (13) for arbitrary vectors. Its projection error is therefore analyzed separately using Nyström subsampling bounds.

## 3.2. Nyström Sub-sampling for Spectral Algorithms

Plain Nyström. Plain Nyström sub-sampling can be viewed as a sub-sampling sketch using matrix �, where the range of the projection operator � is defined as the closure of the span of a subset of the training data points. Specifically, we randomly and uniformly select a subset of � data points $\{ \pmb { x } _ { i _ { j } } \} _ { j = 1 } ^ { m }$ from the training dataset $\{ { \pmb x } _ { i } \} _ { i = 1 } ^ { n }$ without replacement, and define the range of � as Range $P ) = \overline { { \operatorname { s p a n } \{ K _ { { \pmb { x } } _ { i _ { j } } } \} _ { j = 1 } ^ { m } } } .$

Approximate leverage score (ALS) Nyström. Alaoui and Mahoney (2015) proposed the leverage score sampling for Nyström sub-sampling, which corresponds to a non-uniform sub-sampling sketch. The leverage scores quantify the importance of each data point in the dataset:

$$
( l _ { i } ( \lambda ) ) _ { i = 1 } ^ { n } , \quad l _ { i } ( \lambda ) = \left( \mathbf { K } ( \mathbf { K } + \lambda I ) ^ { - 1 } \right) _ { i i } , \quad i = 1 , 2 , \ldots , n .
$$

where $\mathbf { K } = S _ { K , n } S _ { K , n } ^ { * }$ denotes the Gram matrix. In practice, exact leverage scores are often replaced by eficiently computable approximations $\{ \hat { l } _ { i } ( \lambda ) \} _ { i = 1 } ^ { n }$ (Musco and Musco, 2017; Calandriello, Lazaric and Valko, 2017; Rudi, Calandriello, Carratino and Rosasco, 2018; Chen and Yang, 2021) that satisfy

$$
\hat { l } _ { i } ( \lambda ) / L \le l _ { i } ( \lambda ) \le L \hat { l } _ { i } ( \lambda ) \quad \mathrm { f o r ~ s o m e ~ } L \ge 1 .
$$

The ALS Nyström method selects � data points $\{ \widetilde { \pmb { x } } _ { i _ { j } } \} _ { j = \widetilde { \bot } } ^ { m }$ from the training dataset $\{ { \pmb x } _ { i } \} _ { i = 1 } ^ { n }$ according to the probability distribution $\{ q _ { i } \} _ { i = 1 } ^ { n }$ , where $\begin{array} { r } { q _ { i } = \hat { l } _ { i } ( \lambda ) / \sum _ { j = 1 } ^ { n } \hat { l } _ { j } ( \lambda ) } \end{array}$ . The range of the associated projection operator � is then defined by Range $( P ) = \overline { { \mathrm { s p a n } \{ K _ { \widetilde { x } _ { i _ { i } } } \} _ { j = 1 } ^ { m } } }$

To measure the computational complexity of the Nyström method under diferent sampling strategies, Alaoui and Mahoney (2015) introduced two empirical efective dimensions:

$$
d _ { \infty } ( \lambda ) = n \| \operatorname { d i a g } ( \mathbf { K } ( \mathbf { K } + \lambda I ) ^ { - 1 } ) \| _ { \infty } , \quad d _ { \mathrm { e f f } } ( \lambda ) = \operatorname { T r } \left( \mathbf { K } ( \mathbf { K } + \lambda I ) ^ { - 1 } \right) = \sum _ { i = 1 } ^ { n } l _ { i } ( \lambda ) .
$$

For any $\lambda > 0 .$ , it follows directly that $d _ { \mathrm { e f f } } ( \lambda ) \leq d _ { \infty } ( \lambda ) \leq \kappa ^ { 2 } / \lambda$ . Bach (2013) and Alaoui and Mahoney (2015) further showed that the minimum number of Nyström samples required to achieve accurate approximation scales proportionally to $d _ { \infty } ( \lambda )$ for plain Nyström, and to $d _ { \mathrm { e f f } } ( \lambda )$ when approximate leverage score Nyström is employed.

## 4. Main Results

In this section, we first establish a minimax lower bound for the learning rate in Sobolev norms under general source conditions, which ofers the fundamental learning limits of SARP. Then we show that the upper bound for the learning rate of SARP can match the lower bound under mild conditions. The analysis relies on the following definitions and assumptions.

Assumption 1 (Noise condition). There exist constants $\sigma \geq 0$ and $M \geq 0$ such that for all $p \geq 2 , \mathbb { E } ( \vert \varepsilon \vert ^ { p } \vert \pmb { x } ) \leq$ $\scriptstyle { \frac { 1 } { 2 } } p ! \sigma ^ { 2 } { \bar { M } } ^ { p - 2 }$

This assumption is the classical Bernstein condition imposed on the noise, which is widely used in kernel learning theory (Caponnetto and De Vito, 2007; Rastogi and Sampath, 2017). It is satisfied, for example, by bounded, Gaussian, or sub-Gaussian noise. In contrast, the closely related work (Lin and Cevher, 2020a) assumed that the regression function $f _ { \rho }$ is uniformly bounded. Such a requirement can be overly restrictive, and a uniform bound on $f _ { \rho }$ is very hard to verify from finite data in practice. Assumption 1 instead places the Bernstein condition solely on the noise � and does not require $f _ { \rho }$ to be uniformly bounded. In our synthetic experiment in Section 5, the regression function diverges when $s = 0 . 5 ;$ hence $f _ { \rho }$ is not uniformly bounded, yet SARP still achieves the optimal Sobolev learning rates. This confirms that a uniform boundedness assumption is unnecessary for obtaining optimal learning rates for SARP. We acknowledge Assumption 1 still imposes a relatively strong light-tailed requirement on the noise, which limits the applicability of our results to heavy-tailed settings. We also remark that recent work has established KRR theory under heavy-tailed noise with finitely many bounded moments (Mollenhauer, Mücke, Meunier and Gretton, 2026). Adapting our framework to this setting requires a separate and careful analysis, which we leave as promising future work. Further discussion on this extension is given in Section IV-A of the Supplementary Material.

Definition 3 (Efective dimension). For $\lambda > 0$ , define the random variable $\mathcal { N } _ { x } ( \lambda ) = \langle K _ { x } , ( C _ { K } + \lambda I ) ^ { - 1 } K _ { x } \rangle _ { \mathcal { H } }$ with $\textstyle { \boldsymbol { \mathbf { \mathit { x } } } } \in { \mathcal { X } }$ drawn from $\rho _ { \mathcal { X } }$ . Then the average efective dimension and maximal efective dimension of RKHS $\mathcal { H } _ { K }$ are defined as $\mathcal { N } ( \lambda ) = \mathbb { E } \overset { \cdot } { \mathcal { N } } _ { x } ( \lambda )$ and $\begin{array} { r } { \mathcal { N } _ { \infty } ( \lambda ) = \operatorname { e s s } \operatorname* { s u p } _ { x \in X } \mathcal { N } _ { x } ( \lambda ) } \end{array}$ , respectively.

The efective dimension characterizes the complexity of the RKHS $\mathcal { H } _ { K }$ and plays a crucial role in learning theory (Caponnetto and De Vito, 2007; Steinwart et al., 2009; Guo et al., 2017). It is easy to see that $\mathcal { N } ( \lambda ) =$ $\begin{array} { r } { \mathrm { T r } ( ( \dot { C } _ { K } + \dot { \lambda } I ) ^ { - 1 } C _ { K } ) = \sum _ { i = 1 } ^ { \infty } \frac { \mu _ { i } } { \mu _ { i } + \lambda } } \end{array}$ and $\mathcal { N } ( \lambda ) \leq \mathcal { N } _ { \infty } ( \lambda ) \leq \kappa ^ { 2 } \lambda ^ { - 1 }$ , where Tr(⋅) denotes the trace of the operator.

Assumption 2 (Capacity condition). For $\lambda > 0$ , there exist constants $Q > 0$ and $\beta \in ( 1 , \infty )$ such that $\mathcal { N } ( \lambda ) \leq Q ^ { 2 } \lambda ^ { - \frac { 1 } { \beta } }$

This assumption is known as the capacity condition. It can also be verified by the decay rate of the eigenvalues $\{ \mu _ { i } \} _ { i \ge 1 }$ . For example, if $\mu _ { i } \leq c i ^ { - \beta }$ for some $c > 0$ and $\beta > 1$ , then $\mathcal { N } ( \lambda ) \leq { c _ { \beta } } \lambda ^ { - \frac { 1 } { \beta } }$ for some constant $c _ { \beta } > 0$

Assumption 3 (Embedding condition). Assume that there exists an order $\alpha \in [ 1 / \beta , 1 ]$ and a constant $A > 0$ such that the canonical inclusion map $i ^ { \alpha } : [ \mathcal { H } _ { K } ] _ { o } ^ { \alpha } \hookrightarrow L _ { o } ^ { \infty } ( \mathcal { X } ) , i ^ { \alpha } ( f ) = f$ , is bounded, i.e. $\left\| f \right\| _ { \infty } \leq A \left\| f \right\| _ { [ \mathcal { H } ] _ { o } ^ { \alpha } } , \forall f \in [ \mathcal { H } _ { K } ] _ { \rho } ^ { \alpha } .$ Equivalently, $[ \mathcal { H } _ { K } ] _ { \rho } ^ { \alpha }$ is continuously embedded into $L _ { \rho } ^ { \infty } ( \mathcal { X } )$ and the operator norm of the embedding satisfies $\| i ^ { \alpha } \| \leq A$ We also define the embedding index as $\alpha _ { 0 } = \operatorname* { i n f } \big \{ \alpha \in \big [ 1 / \beta , 1 \big ] : \| \mathcal { H } | ^ { \alpha } \hookrightarrow L ^ { \infty } ( X ) \| < \infty \big \}$ and let $\alpha _ { 0 } \in [ 1 / \beta , 1 )$

The embedding property indicates that the interpolation space $[ \mathcal { H } ] _ { \rho } ^ { \alpha }$ can be continuously embedded into $L _ { \rho } ^ { \infty } ( \mathcal { X } )$ with operator norm bounded by �. By the inclusion relationship of interpolation spaces, we have $\mathcal { H } _ { K }$ has the embedding property of order $\alpha ^ { \prime }$ whenever it has the embedding property of order � for any $\alpha ^ { \prime } \geq \alpha$ . Typically, a larger � reflects a weaker embedding. The condition holds trivially when $\alpha = 1$ since the kernel � is bounded. The case $\alpha _ { 0 } = 1 / \beta$ corresponds to the benign RKHSs, in which the feature space is nearly homogeneous and contains no extreme leverage points or spatial spikes. Such benign cases can be referred to in Section III of the Supplementary Material. However, the embedding condition of order $\alpha _ { 0 }$ itself may not hold.

Remark 1. Assumption 3 yields the maximal capacity condition, i.e., $\mathcal { N } _ { \infty } ( \lambda ) \leq A ^ { 2 } \lambda ^ { - \alpha }$ , which is important to our analysis in the mis-specified setting. We provide the detailed proof in Lemma 11 of the Supplementary Material. However, the converse fails in general, and the embedding assumption is strictly stronger. In our theoretical framework, deriving a precise bound for the $L ^ { \infty }$ -norm is a critical step. The maximal capacity condition alone is insuficient as it cannot decouple the spatial supremum from the eigenvalue decay. The embedding assumption resolves this issue directly by establishing the required inequality, i.e., $\| f \| _ { \infty } \le A \| L _ { K } ^ { - \alpha / 2 } S _ { K } f \| _ { \rho }$ for any $f \in \mathcal { H } _ { K }$ . Therefore, the embedding condition is mathematically required for obtaining the required tight bound. Additional discussion and counterexamples appear in Section V-D of the Supplementary Material.

Assumption 4 (General source condition). For a fixed source index function $\phi \in { \mathcal { C } } _ { s }$ and $R > 0$ , where $C _ { s }$ is the index function class defined in (9), assume that

$$
f _ { \rho } \in \Omega _ { \phi , s , R } : = \left\{ f \in L _ { \rho _ { \mathscr { g } } } ^ { 2 } : f = \phi ( L _ { K } ) \boldsymbol { g } , \| \boldsymbol { g } \| _ { \rho } \leq R \right\} .\tag{14}
$$

For $\lambda \in ( 0 , \kappa ^ { 2 } ]$ , we further assume that there exists $c > 0$ such that $c \lambda ^ { \tau } / \phi ( \lambda ) \leq \operatorname* { i n f } _ { t \in [ \lambda , \kappa ^ { 2 } ] } t ^ { \tau } / \phi ( t )$

A typical example is the Hölder source condition, i.e., $\phi ( t ) = t ^ { s / 2 }$ for some $s > 0$ . In this case, $f _ { \rho } \in \Omega _ { \phi , s , R }$ implies that $f _ { \rho } \in [ \mathcal { H } _ { K } ] _ { \rho } ^ { s }$ and $\| f _ { \rho } \| _ { [ \mathcal { H } ] _ { o } ^ { s } } \leq R .$ , as also considered in Fischer and Steinwart (2020); Zhang, Li, Lu and Lin (2023); Zhang, Li and Lin (2024). The parameter � quantifies the smoothness of $f _ { \rho }$ relative to the RKHS $\mathcal { H } _ { K }$ . Specifically, if $s > 1$ , then $f _ { \rho }$ is smoother than the functions in $\mathcal { H } _ { K } \left( f _ { \rho } \in \mathcal { H } _ { K } \right)$ , whereas $0 < s < 1$ means $f _ { \rho }$ is less smooth than the functions in $\dot { \mathcal { H } } _ { K } ( f _ { \rho } \in [ \mathcal { H } _ { K } ] _ { \rho } ^ { s } )$ . Practically, one often employs a very smooth kernel (Gaussian or high-smoothness Matérn kernel) while the true regression function possesses only finite Sobolev regularity. Such "kernel too smooth, truth less smooth" situations are common, for instance, a Gaussian kernel is used as a default and still yields good empirical performance. Our theory will show that this is reasonable since SARP can still achieve the optimal rates in this mis-specified setting. In Corollary 2, we further present the Hölder-source rates in three smoothness regimes.

## 4.1. Minimax Lower Bound for Sobolev Norm Learning

In order to establish the lower bound, we introduce the following additional assumption, which can be viewed as an enhanced version of Assumption 2. Analogous assumptions appear in Fischer and Steinwart (2020); Zhang et al. (2024).

Assumption 5 (Eigenvalue decay condition). Assume that there exist some positive constants $0 < q _ { 1 } \leq q _ { 2 }$ and $\beta \in ( 1 , \infty )$ such that $q _ { 1 } k ^ { - \beta } \leq \mu _ { k } \leq q _ { 2 } k ^ { - \beta } , \forall k \geq 1$ , where $\mu _ { k }$ is the �-th eigenvalue of the integral operator $L _ { K }$ . Thus, it also holds that $\mathcal { N } ( \lambda ) \asymp \lambda ^ { - 1 / \beta }$

Theorem 1. Let the constants �, �, �, � be fixed. Given $s > 0$ and $\beta \in ( 1 , \infty ] ,$ , let $\mathcal { P } = \mathcal { P } ( \phi , s , \beta )$ be the set of all probability distributions � such that Assumptions 1, 5 and (14) hold. Thenfor any $\gamma \in [ 0 , s \wedge 1 ] , \delta \in ( 0 , 3 / 8 )$ , and all possible estimators �<sup>̂</sup>based on the � i.i.d. samples, there exists a probability measure $\rho ^ { * } \in \mathcal { P }$ such that with probability at least $1 - \delta ,$

$$
\begin{array} { r } { \| \widehat { f } - f _ { \rho ^ { * } } \| _ { { [ \mathcal { H } ] } _ { \rho ^ { * } } ^ { \gamma } } ^ { 2 } \geq C \delta \phi ^ { 2 } ( \Phi ^ { - 1 } ( n ^ { - 1 } ) ) ( \Phi ^ { - 1 } ( n ^ { - 1 } ) ) ^ { - \gamma } , } \end{array}\tag{15}
$$

where $\Phi ( t ) = ( \phi ( t ) / \phi ( 1 ) ) ^ { 2 } t ^ { 1 / \beta } , \Phi ^ { - 1 } ( \cdot )$ denotes thefunctional inverse, and � is a positive constant independent of�, �, �, whose explicit expression is given in the proof.

Theorem 1 establishes a minimax lower bound for the learning rate in Sobolev norms under general source conditions. In the existing literature, minimax lower bounds for the $L _ { \rho } ^ { 2 }$ norm or  norm are typically derived for the well-specified case where $f _ { \rho } \in \mathcal { H } _ { K }$ (Caponnetto and De Vito, 2007; Rastogi and Sampath, 2017), which corresponds to the source condition with $s \geq 1$ . For the mis-specified case where $f _ { \rho } \notin \mathcal { H } _ { K }$ , Fischer and Steinwart (2020); Zhang et al. (2023, 2024) provide the minimax lower bounds with $s > 0$ . However, these studies all assume the Hölder source condition, which is a special case of the more general source condition considered in this paper. To the best of our knowledge, this is the first work to establish minimax lower bounds under general source conditions for both well-specified and mis-specified cases. Furthermore, our analysis does not rely on the uniform boundedness assumption on $f _ { \rho }$ and accounts for multiple Sobolev norms. When $\phi ( t ) = t ^ { s / 2 }$ for some $s > 0 ,$ , the bound in Theorem 1 recovers the existing lower bounds of Fischer and Steinwart (2020); Zhang et al. (2023, 2024), as stated in the following corollary. We also want to emphasize that the minimax lower bound in Theorem 1 does not rely on the qualification parameter � and therefore does not require any restriction on �. So the lower bound is valid for any $s > 0$

Table 2  
Summary of technical parameters used in the main results.
<table><tr><td>Parameter</td><td>Concept</td><td>Intuitive Meaning</td></tr><tr><td> $\beta$ </td><td>Capacity condition</td><td>Measures the complexity of the RKHS (a larger  $\beta$  indicates a smaller space).</td></tr><tr><td>α</td><td>Embedding condition</td><td>Controls the continuous embedding of the RKHS interpolation space into  $L ^ { \infty }$  (a larger α reflects a weaker embedding).</td></tr><tr><td> $\alpha _ { 0 }$ </td><td>Embedding index</td><td>The smallest embedding exponent in the limiting sense.</td></tr><tr><td> $s$ </td><td>Source condition</td><td>Characterizes the smoothness of target function (a larger s implies a smoother function).</td></tr><tr><td> $\gamma$ </td><td>Norm index</td><td>Determines the error evaluation norm  $( \gamma = 0$  for standard  $L ^ { 2 }$  error,  $\gamma = 1$  for RKHS error).</td></tr></table>

Corollary 1. Under the same conditions of Theorem 1, $i f \phi ( t ) = t ^ { s / 2 }$ for some $s > 0 ,$ then with probability at least $1 - \delta ,$ , there holds that

$$
\| \widehat { f } - f _ { \rho ^ { * } } \| _ { [ \mathcal { H } ] _ { \rho ^ { * } } ^ { \gamma } } ^ { 2 } \geq C \delta n ^ { - \frac { ( s - \gamma ) \beta } { s \beta + 1 } } .\tag{16}
$$

Remark 2. The lower bound in Theorem 1 applies to standard spectral algorithms without random projections. Deriving a corresponding bound that explicitly incorporates the projection error is technically challenging and such lower bounds remain largely open. Nevertheless, Theorem 1 serves as a benchmark for the optimal rate without computational constraints. Our upper bounds explicitly quantify the projection efect through the projection dimension and the approximation error $\mathcal { A } _ { K , P } .$ , showing when the projected estimators match the minimax rate. Finally, since $f _ { n , \lambda } ^ { r p } \in \operatorname { R a n } ( P )$ , we have

$$
\| f _ { n , \lambda } ^ { r p } - f _ { \rho } \| _ { [ \mathcal { H } ] _ { \rho } ^ { \gamma } } ^ { 2 } \geq \operatorname* { i n f } _ { f \in \mathrm { R a n } ( P ) } \| f - f _ { \rho } \| _ { [ \mathcal { H } ] _ { \rho } ^ { \gamma } } ^ { 2 } .
$$

This inequality provides a projection-dependent lower bound that measures how well $f _ { \rho }$ can be approximated within the projected subspace. In particular, the approximation error can be zero when $f _ { \rho }$ belongs to Ran(�), even if $P \neq I$ Therefore, deriving a sharp minimax lower bound that explicitly accounts for random projections would require additional assumptions on the relationship between $f _ { \rho }$ and the projected subspace. We leave this question for future investigation.

## 4.2. Learning Rates for General Random Projections

In this section, we present our main theoretical results on the learning rates of SARP. For clarity, we introduce a projection operator term $\mathcal { A } _ { K , P } = | | C _ { K } ^ { 1 / 2 } ( I - P ) | |$ and the projection error $\Delta ,$ , which will be specified in the theorem below. Throughout the analysis, we assume $s \in ( 0 , 2 \tau ]$ . This range follows from the qualification of the spectral filter and is needed for the upper-bound analysis. Recall that $f _ { n , \lambda } ^ { r p }$ is given in (12). For ease of reference, we summarize the technical parameters used in the main results in Table 2.

Theorem 2. Under Assumptions 1-4, for any $\gamma \in [ 0 , 1 \land s ] , \lambda \in ( 0 , 1 ]$ and $\delta \in ( 0 , 1 )$ $i f n \gtrsim \lambda ^ { - \alpha }$ log(32�<sup>2</sup>∕��) ∨ $D _ { q , \delta } ^ { \frac { 2 \alpha } { s + \alpha } } \left( \lambda ^ { \alpha } \phi ^ { 2 } ( \lambda ) \right) ^ { - \frac { \alpha } { s + \alpha } }$ where $D _ { q , \delta }$ is defined in the proof, then with probability at least $1 - \delta ,$ the following holds:

(1) $I f \phi : [ 0 , \kappa ^ { 2 } ] \to \mathbb { R } ^ { + }$ is operator monotone with $\phi ( 0 ) = 0$ and $\phi ( \kappa ^ { 2 } ) < \infty ,$ , or $\phi : [ 0 , b ] \to \mathbb { R }$ is real analytic on an open neighborhood of [0, �] and satisfies $\phi ( 0 ) = 0 ,$ , then

$$
\| f _ { n , \lambda } ^ { r p } - f _ { \rho } \| _ { [ H ] _ { \rho } ^ { \gamma } } \lesssim \lambda ^ { - \frac { \gamma } { 2 } } \left[ \phi ( \lambda ) + \frac { 1 } { n \lambda ^ { \operatorname* { m a x } \{ \frac { \alpha } { 2 } , \alpha - \frac { s } { 2 } \} } } + \frac { 1 } { \sqrt { n } \lambda ^ { \frac { 1 } { 2 \beta } } } + \frac { \phi ( \lambda ) } { \sqrt { n } \lambda ^ { \frac { \alpha } { 2 } } } + \Delta \right] \log \frac { 8 } { \delta } .
$$

(2) $I f \phi = \psi \vartheta ,$ where $\psi : [ 0 , b ] \to \mathbb { R } _ { + }$ is operator monotone, where $b > 2 \kappa ^ { 2 } , \psi ( 0 ) = 0 ,$ , and $\psi ( b ) < \infty ,$ , and $\vartheta : [ 0 , \kappa ^ { 2 } ]  \mathbb { R } ^ { + }$ is non-decreasing, real analytic on an open neighborhood $o f [ 0 , b ]$ and satisfies $\vartheta ( 0 ) = 0 ,$ , and for $\forall \lambda \in ( 0 , \kappa ^ { 2 } ]$ , there exists $c ^ { \prime } > 0$ such that $\begin{array} { r } { c ^ { \prime } \frac { \lambda ^ { \tau } } { \vartheta ( \lambda ) \lambda ^ { 1 / 2 } } \leq \operatorname* { i n f } _ { t \in [ \lambda , \kappa ^ { 2 } ] } \frac { t ^ { \tau } } { \vartheta ( t ) t ^ { 1 / 2 } } } \end{array}$ , then

$$
\| f _ { n , \lambda } ^ { r p } - f _ { \rho } \| _ { [ H ] _ { \rho } ^ { r } } \lesssim \lambda ^ { - \frac { \gamma } { 2 } } \left[ \phi ( \lambda ) + \frac { 1 } { n \lambda ^ { \operatorname* { m a x } \{ \frac { \alpha } { 2 } , \alpha - \frac { \delta } { 2 } \} } } + \frac { 1 } { \sqrt { n } \lambda ^ { \frac { 1 } { 2 \beta } } } + \frac { \phi ( \lambda ) } { \sqrt { n } \lambda ^ { \frac { \alpha } { 2 } } } + \vartheta ( \lambda ) \psi \left( \frac { 1 } { \sqrt { n } } \right) + \Delta \right] \log \frac { 8 } { \delta } .
$$

Here, the projection error $\Delta = \left\{ \begin{array} { l l } { \lambda ^ { \frac { s } { 2 } - 1 } ( \mathcal { A } _ { K , P } ^ { 2 } + \lambda ^ { \frac { 1 } { 2 } } \mathcal { A } _ { K , P } + \lambda ^ { \frac { \gamma } { 2 } } \mathcal { A } _ { K , P } ^ { 2 - \gamma } ) + \lambda + \mathcal { A } _ { K , P } ^ { 2 } , ~ i f s < 2 ; } \\ { \mathcal { A } _ { K , P } ^ { 2 } + n ^ { - \frac { 1 } { 4 } } \mathcal { A } _ { K , P } + \lambda ^ { \frac { \gamma } { 2 } } \mathcal { A } _ { K , P } ^ { 2 - \gamma } + n ^ { - \frac { 1 } { 2 } } , ~ i f s \geq 2 . } \end{array} \right.$

Theorem $^ 2$ provides high-probability Sobolev norm error bounds for the SARP under general source conditions. Here we consider three types of source conditions: (1) a non-decreasing index function �; (i.e., the power index function or the Hölder source condition, $\phi ( t ) = t ^ { s / 2 } )$ (2) a real-analytic index function $\phi ; ( 3 )$ a composite index function $\phi = \psi \vartheta$ . Compared to the general source conditions proposed in Bauer et al. (2007); Lin and Cevher (2020b), our main contribution lies in the refined condition imposed on the index function, which is carefully exploited to achieve tight error bounds. Specifically, we propose a new lemma providing a sharp bound for $\| P \phi ( C _ { K , n } ) P - \phi ( P C _ { K , n } P ) \|$ using $\mathcal { A } _ { K , P } = | | C _ { K } ^ { 1 / 2 } ( I - P ) | |$ . To the best of our knowledge, this lemma and its proof have not appeared before. Our proof is based on a delicate block-operator decomposition of $C _ { K , n }$ together with the resolvent identity and Cauchy’s integral formula of $\phi ,$ rather than standard spectral perturbation or polynomial-truncation arguments, and thus appears to be genuinely new.

The derived learning rates can be decomposed into several terms, each reflecting diferent aspects of the learning process. The term $\lambda ^ { - \frac { \gamma } { 2 } } \phi ( \lambda )$ represents the approximation error, which depends on the regularization parameter � and the smoothness of the regression function $f _ { \rho }$ characterized by the index function $\phi .$ The intermediate terms involving �, $\lambda , \beta ,$ and � collectively describe the empirical error, which captures the efect of finite-sample variance. The last term $\Delta$ accounts for the impact of random projections on learning performance. When $P = I$ , our result recovers the existing bounds in Lin and Cevher (2020b) for spectral algorithms without RP. To achieve the optimal trade-of between these error terms, the regularization parameter � must be chosen appropriately, which leads to the following theorem.

Theorem 3. Under the same assumptions and conditions of Theorem 2, let $s > \alpha _ { 0 } - 1 / \beta ,$ , and $\lambda = \Phi ^ { - 1 } ( n ^ { - 1 } )$ with $\Phi ( t ) = ( \phi ( t ) / \phi ( 1 ) ) ^ { 2 } t ^ { 1 / \beta }$ and $\Phi ^ { - 1 } ( \cdot )$ denoting thefunctional inverse. For case (2), we additionally let $\lambda \ge n ^ { - 1 / 2 }$ , then with probability at least $1 - \delta ,$ there holds that

$$
\begin{array} { r } { \| f _ { n , \lambda } ^ { r p } - f _ { \rho } \| _ { [ \mathcal { H } ] _ { \rho } ^ { \gamma } } ^ { 2 } \lesssim ( \phi ( \Phi ^ { - 1 } ( n ^ { - 1 } ) ) + \Delta ) ^ { 2 } ( \Phi ^ { - 1 } ( n ^ { - 1 } ) ) ^ { - \gamma } \log ^ { 2 } \displaystyle \frac { 8 } { \delta } , } \end{array}
$$

where the projection error Δ is the same as that in Theorem 2.

Compared to the lower bound in Theorem 1, Theorem 3 shows that SARP attains the minimax-optimal learning rates up to a logarithmic factor whenever the projection error $\Delta$ is of a smaller order than $\phi ( \Phi ^ { - 1 } ( n ^ { - 1 } ) )$ . This requirement can be satisfied by choosing an appropriate projection dimension � for each type of random projection, as discussed in the next section. When $P = I$ , the resulting learning rates align with those in Bauer et al. (2007) and Lin and Cevher (2020b) for spectral algorithms under general source conditions. However, our results are more general as they encompass a broader range of source index $s > \alpha _ { 0 } - 1 / \beta$ with $\alpha _ { 0 } \in [ 1 / \beta , 1 )$ and Sobolev norms $\gamma \in [ 0 , 1 \land s ]$ , while the previous works primarily focus on the well-specified case with $s \geq 1$ (Bauer et al., 2007) or part of the mis-specified case $s \ge 1 - 1 / \beta$ (Lin and Cevher, 2020b).

Remark 3. The Sobolev norm provides a more general measure of learning performance than the traditional $L _ { \rho _ { X } } ^ { 2 }$ norm. For the squared loss, we have

$$
\begin{array} { r } { \mathcal { E } ( f ) - \mathcal { E } ( f _ { \rho } ) = \| f - f _ { \rho } \| _ { \rho } ^ { 2 } = \| f - f _ { \rho } \| _ { [ \mathcal { H } ] _ { \rho } ^ { 0 } } ^ { 2 } . } \end{array}
$$

Thus, setting $\gamma = 0$ in Theorems 2 and 3 directly yields the corresponding population excess risk bounds. In particular, when $s > \alpha _ { 0 } - 1 / \beta$ , SARP achieves the minimax-optimal excess risk rate up to logarithmic factors, provided that the projection error is suficiently small. For $\gamma > 0 .$ , the Sobolev norm provides stronger information by taking into account the smoothness of the estimated function. In particular, for Sobolev-type RKHSs, these norms can measure errors in fractional-order smoothness. For example, when $K ( x , x ^ { \prime } ) = \operatorname* { m i n } \{ x , x ^ { \prime } \}$ and $\gamma = 1$ , we have

$$
\| f - f _ { \rho } \| _ { [ \mathcal { H } ] _ { \rho } ^ { 1 } } ^ { 2 } = \int _ { 0 } ^ { 1 } | f ^ { \prime } ( x ) - f _ { \rho } ^ { \prime } ( x ) | ^ { 2 } d x ,
$$

provided that $f , f _ { \rho } \in \mathcal { H } _ { K }$ . This allows us to evaluate the recovery of derivatives, which cannot be directly measured by the population excess risk.

## 4.3. Learning Rates for Specific Random Projections

In this section, we apply Theorem 3 to three specific types of random projections introduced earlier: randomized sketches, plain, and ALS Nyström sub-sampling. We show how to choose the projection dimension � to efectively control the projection error Δ, thereby achieving optimality. For simplicity, we focus on the Hölder source condition with $\phi ( t ) = \bar { t } ^ { s / 2 }$ in the following corollary.

Corollary 2. Under the same assumptions and conditions ofTheorem 2, let $\phi ( t ) = \kappa ^ { - ( s - 2 ) _ { + } } t ^ { s / 2 }$ for some $s > 0 ;$ , and � is suficiently large. We consider thefollowing three cases of�:

(1) When $s \leq \alpha _ { 0 } - 1 / \beta ,$ , for any $\alpha > \alpha _ { 0 } ,$ , let $\lambda \asymp \left( \frac { C \log n } { n } \right) ^ { \frac { 1 } { \alpha } }$ for suficiently large �, and the projection dimension � satisfies

$m \gtrsim { \left\{ \begin{array} { l l } { n ^ { \frac { 1 } { \alpha \beta } } } \\ { n , } \end{array} \right. }$ , for Randomized Sketches and ALS Nyström; for Plain Nyström.

Then with probability at least $1 - \delta ,$ there holds that

$$
\| f _ { n , \lambda } ^ { r p } - f _ { \rho } \| _ { [ \mathcal { H } ] _ { \rho } ^ { \gamma } } ^ { 2 } \lesssim n ^ { - \frac { s - \gamma } { \alpha } } ( \log n ) ^ { \frac { s - \gamma } { \alpha } } \log ^ { 2 } \frac { 1 0 } { \delta } .
$$

(2) When $s > \alpha _ { 0 } - 1 / \beta$ and $s < 2 ,$ , let $\lambda \asymp n ^ { - \frac { \beta } { s \beta + 1 } }$ , and the projection dimension � satisfies

$m \gtrsim \left\{ \begin{array} { c } { { } } \\ { { n ^ { \frac { 1 } { s \beta + 1 } } } } \\ { { } } \\ { { \left( n ^ { \frac { \alpha \beta } { s \beta + 1 } } \right. } } \end{array} \right.$ , for Randomized Sketches and ALS Nyström; , for Plain Nyström.

Then with probability at least $1 - \delta ,$ there holds that

$$
\Vert f _ { n , \lambda } ^ { r p } - f _ { \rho } \Vert _ { [ \mathcal { H } ] _ { \rho } ^ { \gamma } } ^ { 2 } \lesssim n ^ { - \frac { ( s - \gamma ) \beta } { s \beta + 1 } } \log ^ { 2 } \frac { 1 0 } { \delta } .
$$

(3) When $s \geq 2 ,$ , let $\lambda \asymp n ^ { - \frac { \beta } { s \beta + 1 } }$ , and the projection dimension � satisfies

$m \gtrsim \left\{ \begin{array} { l l } { n ^ { \frac { s - \gamma } { ( 2 - \gamma ) ( s \beta + 1 ) } } , } \\ { n ^ { \frac { \alpha \beta ( s - \gamma ) } { ( 2 - \gamma ) ( s \beta + 1 ) } } . } \end{array} \right.$ , for Randomized Sketches and ALS Nyström; , for Plain Nyström.

Then with probability at least $1 - \delta ,$ there holds that

$$
\Vert f _ { n , \lambda } ^ { r p } - f _ { \rho } \Vert _ { [ \mathcal { H } ] _ { \rho } ^ { \gamma } } ^ { 2 } \lesssim n ^ { - \frac { ( s - \gamma ) \beta } { s \beta + 1 } } \log ^ { 2 } \frac { 1 0 } { \delta } .
$$

Corollary 2 provides capacity-dependent learning rates for spectral algorithms with diferent types of random projections under the Hölder source condition. The results demonstrate that by selecting an appropriate projection dimension � proportional to the average or maximal efective dimension, the projection error Δ can be controlled to be of a smaller order than the approximation and empirical error, thereby achieving the minimax optimal learning rates $n ^ { - ( s - \gamma ) \beta / ( s \beta + 1 ) }$ up to a logarithmic factor when $s > \alpha _ { 0 } - 1 / \beta$ . The learning rates degrade to $n ^ { - ( s - \gamma ) / \alpha }$ when $s \leq \alpha _ { 0 } - 1 / \beta$ and $\alpha > \alpha _ { 0 }$ . In the most benign case when $\alpha _ { 0 } = 1 / \beta$ , the optimal learning rates are achieved for all $s \in ( 0 , 2 \tau ]$ . For ease of notation, we show only the polynomial orders of the projection dimensions in Corollary 2. The full suficient conditions including all logarithmic factors are stated in Lemmas 21-23 of the Supplementary Material.

Compared with the existing results for spectral algorithms (or KRR) with randomized projection under the Hölder source condition, our guarantees extend to the mis-specified case with $s \leq \alpha _ { 0 } - 1 / \beta$ and Sobolev norms $\gamma \in [ 0 , 1 \land s ]$ For example, Yang et al. (2017) only considered the worst case with $s = 1$ and $\gamma = 0$ under fixed design; Rudi et al. (2015) established the optimal learning rates solely in the well-specified case with $s \geq 1$ and $\gamma = 0 ;$ the state-of-the-art results (Lin and Cevher, 2020a) established learning rates in various norms but only for part of the mis-specified case, requiring $s \ge 1 - 1 / \beta$ . Our extension comes from tighter estimates of (i) the similarity between the population covariance operator $C _ { K }$ and its empirical counterpart $C _ { K , n } .$ , which relaxes the sample-size requirement from $n \gtrsim \lambda ^ { - 1 }$ to $n \gtrsim \lambda ^ { - \alpha }$ , and (ii) the sample-variance term, reducing its order from $\mathcal { O } ( n ^ { - 1 } \lambda ^ { \operatorname* { m i n } \{ - 1 / 2 , s / 2 - 1 \} } + n ^ { - 1 / 2 } \lambda ^ { ( s - 1 ) / 2 } )$ to $\mathcal { O } ( n ^ { - 1 } \lambda ^ { \mathrm { m i n } \{ - \alpha / 2 , s / 2 - \alpha \} } + n ^ { - 1 / 2 } \lambda ^ { ( s - \alpha ) / 2 } )$ . Furthermore, diferent from the analysis in Lin and Cevher (2020a), we do not impose the uniform boundedness of $f _ { \rho } .$ , which has been discussed in Assumption 1.

Remark 4. The embedding condition yields the maximal capacity condition, i.e., $\mathcal { N } _ { \infty } ( \lambda ) \lesssim \lambda ^ { - \alpha }$ for any $\alpha > \alpha _ { 0 }$ , which determines the sample-size requirement $n \gtrsim \lambda ^ { - \alpha }$ up to logarithmic factors. In Theorem 2, the upper bounds explicitly rely on the embedding index � through the maximal capacity condition. For the Hölder source condition in Corollary 2, balancing the approximation and stochastic errors yields $\lambda \asymp n ^ { - \beta / ( s \beta + 1 ) }$ with $s > \alpha _ { 0 } - 1 / \beta$ , which in turn gives the final learning rate $n ^ { - s \beta / ( s \beta + 1 ) }$ up to logarithmic factors. The choice of � also satisfies the embedding-dependent sample-size requirement $n \gtrsim \lambda ^ { - \alpha }$ when $s > \alpha _ { 0 } - 1 / \beta$ and � is suficiently large. Consequently, although the final optimal learning rate appears independent of the embedding index �, the sample-size requirement implicitly depends on it, reflecting the influence of the embedding condition on the overall analysis. However, when $s \leq \alpha _ { 0 } - 1 / \beta$ , the embedding condition becomes the limiting factor, leading to the slower rate (log $n / n ) ^ { ( s - \gamma ) / \alpha }$ with $\lambda \asymp ( \log n / n ) ^ { 1 / \alpha }$ . Thus, a smaller embedding � enlarges the range in which minimax-optimal rates can be achieved. Moreover, � influences the projection dimension required for the learning rates, as shown in Corollary 2.

Remark 5. Ignoring the logarithmic terms, for randomized sketches and ALS Nyström methods, we can establish a sharp bound on $\mathcal { A } _ { K , P }$ provided $m \gtrsim \lambda ^ { - 1 / \beta }$ . For the plain Nyström method, it sufices that $m \gtrsim \lambda ^ { - \alpha }$ . These requirements follow directly from Assumption 2 that the average efective dimension satisfies $\mathcal { N } ( \lambda ) \lesssim \lambda ^ { - 1 / \beta }$ , and by Assumption 3, the maximal efective dimension satisfies $\mathcal { N } _ { \infty } ( \lambda ) \lesssim \lambda ^ { - \alpha }$ . The conditions above can be equivalently expressed as $m \gtrsim \mathcal { N } ( \lambda )$ for randomized sketches and ALS Nyström; � $\gtrsim \mathcal { N } _ { \infty } ( \lambda )$ for plain Nyström. In the proof of Corollary 2, we explicitly obtain the optimal rates by choosing � so that the projection term does not exceed the bias and variance terms under these conditions. Consequently, the suficient projection dimension is proportional to the (average or maximal) efective dimension to guarantee the optimal rates.

Remark 6. Computationally, randomized sketches and ALS Nyström require a projection dimension � that is significantly smaller than the sample size �, yielding substantial computational savings. In contrast, plain Nyström demands a larger projection dimension, which may limit its computational eficiency in certain scenarios $( s \le \alpha _ { 0 } - 1 / \beta )$ Compared to the optimal projection dimension derived in Lin and Cevher (2020a), the required projection dimension for the optimal rates remains the same as in Lin and Cevher (2020a) for randomized sketches and ALS Nyström, while the source range extends from $s \geq 1 - 1 / \beta \mathrm { t o } s > \alpha _ { 0 } - 1 / \beta$ . For plain Nyström, our results reduce the required projection dimension from $m \gtrsim n ^ { \beta / ( s \beta + 1 ) }$ when $s \in [ 1 - 1 / \beta , 2 )$ to $m \gtrsim n ^ { \alpha \beta / ( s \beta + 1 ) }$ when $s \in ( \alpha _ { 0 } - 1 / \beta , 2 )$ , from $m \gtrsim n ^ { \frac { \beta ( s - \gamma ) } { ( 2 - \gamma ) ( s \beta + 1 ) } }$ to � $\gtrsim n ^ { \frac { u p ( s - \gamma ) } { ( 2 - \gamma ) ( s \beta + 1 ) } }$ when $s \geq 2 .$ . In the most benign case $\alpha _ { 0 } = 1 / \beta$ , the required projection dimension further reduces to those in randomized sketches and ALS Nyström. These improvements stem primarily from our refined analysis of operator similarity and sample variance control.

Remark 7. For Tikhonov regularization with $\tau = 1$ , we note that when $s > 2 ,$ the saturation efect appears, where the learning rates do not depend on the source index � anymore and coincide with those obtained for $s = 2 , \operatorname { i . e . , } n ^ { - \frac { 2 \beta } { 2 \beta + 1 } }$ in

$L _ { \rho _ { \mathcal { X } } } ^ { 2 }$ norm. This phenomenon is well known in the theory of inverse problems (Bauer et al., 2007) and recently has been proved for KRR (Li, Zhang and Lin, 2023). By exploiting the smoothness of the regression function beyond $s = 2 .$ , Lu, Li, Lin et al. (2024) and Li, Gan, Shi and Lin (2026) further showed that, for spectral algorithms with finite qualification �, the learning rates in the $L _ { \rho _ { \mathcal { X } } } ^ { 2 }$ norm are $n ^ { - \frac { 2 \tau \beta } { 2 \tau \beta + 1 } }$ whenever $s > 2 \tau$ . To the best of our knowledge, there are currently no results on Sobolev norm learning rates for spectral algorithms in the regime $s > 2 \tau$ . We conjecture that a similar saturation efect occurs when $s > 2 \tau$ , but we leave this question for future work. In this paper, we therefore restrict our analysis to $s \in ( 0 , 2 \tau ]$

## 5. Numerical Experiments

In this section, we provide some simulated examples to validate our theoretical findings and real-world applications to demonstrate the efectiveness of SARP.

## 5.1. Simulated Examples

Our aim is to verify the optimal rates of SARP derived in Corollary 2. We consider three types of spectral algorithms:

(1) Kernel Ridge Regression (KRR) with $g _ { \lambda } ( t ) = ( t + \lambda ) ^ { - 1 }$ and $\tau = 1$

(2) Gradient Method (GM) with $\begin{array} { r } { g _ { \lambda } ( t ) = \sum _ { k = 1 } ^ { l } ( 1 - t ) ^ { l - k } , \lambda = \frac { 1 } { l } } \end{array}$ and � being any positive number;

(3) Spectral Cut-of (SC) with $\begin{array} { r } { g _ { \lambda } ( t ) = t ^ { - 1 } \mathbb { I } _ { [ t \geq \lambda ] } } \end{array}$ and � being any positive number.

For the simulation setup, we suppose the input space $\mathcal { X } = [ 0 , 1 ]$ and the marginal distribution $\rho _ { \mathcal { X } }$ is uniform on $\mathcal { X } .$ The kernel function is chosen as $K ( x , x ^ { \prime } ) = \operatorname* { m i n } \{ x , x ^ { \prime } \}$ . The corresponding RKHS is the first order Sobolev space $\mathcal { H } ^ { 1 } ( [ 0 , 1 ] )$ with eigenfunction-eigenvalue pairs given by

$$
e _ { k } ( x ) = \sqrt { 2 } \sin \frac { ( 2 k - 1 ) \pi x } { 2 } , \mu _ { k } = \left( \frac { 2 } { ( 2 k - 1 ) \pi } \right) ^ { 2 } , \forall k \geq 1 .
$$

It is easy to verify that Assumption 2 holds with $\beta = 2$ , and Section III of the Supplementary Material has shown that the RKHS is benign with $\alpha _ { 0 } = 1 / \beta = 0 . 5$ since the eigenfunctions are uniformly bounded. The regression function is constructed as

$$
f _ { \rho } ( x ) = \sum _ { k = 1 } ^ { \infty } { \frac { 1 } { k ^ { s + 0 . 5 } \log ( k + 1 ) } } e _ { 2 k - 1 } ( x ) \quad { \mathrm { f o r ~ s o m e ~ } } s > 0 .
$$

We can also verify that $f _ { \rho } \in [ \mathcal { H } _ { K } ] _ { \rho } ^ { s }$ . The output � is generated by the following model

$$
y = f _ { \rho } ( x ) + \varepsilon ,
$$

where � follows the standard Gaussian distribution.

We investigate three distinct scenarios: a mis-specified case with $s = 0 . 5$ and two well-specified cases with $s = 1 . 5$ and $s = 2$ . For each scenario, we vary the training sample size � across 10 values evenly from $\{ 1 0 0 0 , 2 0 0 0 , \ldots , 1 0 0 0 0 \}$ The regularization parameter is set according to the theoretical optimal form $\lambda = C _ { \lambda } n ^ { - \frac { \beta } { s \beta + 1 } }$ . For each generated dataset, we perform a grid search of 10 values for $C _ { \lambda }$ from $1 0 ^ { - 7 } ~ \mathrm { t o } ~ 1 \dot { 0 } ^ { 2 }$ . We select the value of $C _ { \lambda }$ that gives the smallest error on an independent validation set and report the corresponding performance. For the random projection methods, we use Gaussian sketching (GAU), ROS sketching (ROS), Nyström with uniform sampling (Ny-plain), and Nyström with approximate leverage scores (Ny-ALS) (Rudi et al., 2018). For each scenario, the projection dimension � is chosen based on the theoretical requirements outlined in Corollary 2. Here, we choose the projection dimension � according to Corollary 2 with the multiplicative constant fixed at $C _ { m } = 1$ , and round the resulting value up to the nearest integer. Since our experiments mainly aim to verify the theoretical learning rates, we do not further tune $C _ { m }$ to avoid additional computational costs. We further consider the full versions of KRR, GM, and SC as baseline methods. For a fair comparison, the full and projected estimators use the same spectral filters, kernel functions, data splits, and regularization parameter selection procedures. We further investigate the efect of projection dimension � on the performance of SARP in the real applications in the next section.

![](images/209dec7c241e3aac3f9bb6cb9e32f9f01d6be1c1b20e29aeaa32ad3345888db4.jpg)  
Figure 2: Log empirical learning rates of SARP with diferent spectral algorithms and projection methods for the mis-specified case when $s = 0 . 5$ averaged over 50 independent trials. The dashed lines represent the theoretical slopes. The gray lines represent the full spectral algorithms.

Suppose the solution of SARP is $\begin{array} { r } { f _ { n , \lambda } ^ { r p } = \sum _ { i = 1 } ^ { n } a _ { i } K ( \pmb { x } _ { i } , \cdot ) } \end{array}$ , then learning rates $\| f _ { n , \lambda } ^ { r p } - f _ { \rho } \| _ { [ \mathcal { H } ] _ { \rho } ^ { \gamma } } ^ { 2 }$ can be computed as

$$
\begin{array} { l } { \displaystyle \| f _ { n , \lambda } ^ { r p } - f _ { \rho } \| _ { [ H ] _ { p } ^ { \gamma } } ^ { 2 } = \| L _ { K } ^ { - \frac { \gamma } { 2 } } ( f _ { n , \lambda } ^ { r p } - f _ { \rho } ) \| _ { \rho } ^ { 2 } = \left\| L _ { K } ^ { - \frac { \gamma } { 2 } } \left( \displaystyle \sum _ { i = 1 } ^ { n } a _ { i } \displaystyle \sum _ { j = 1 } ^ { \infty } \mu _ { j } e _ { j } ( x _ { i } ) e _ { j } - \displaystyle \sum _ { k = 1 } ^ { \infty } \frac { 1 } { k ^ { s + 0 . 5 } \log ( k + 1 ) } e _ { 2 k - 1 } \right) \right\| _ { \rho } ^ { 2 } } \\ { \displaystyle = \sum _ { k = 1 } ^ { \infty } \left( \displaystyle \sum _ { i = 1 } ^ { n } a _ { i } e _ { 2 k - 1 } ( x _ { i } ) - \displaystyle \frac { 1 } { \mu _ { 2 k - 1 } k ^ { s + 0 . 5 } \log ( k + 1 ) } \right) ^ { 2 } \mu _ { 2 k - 1 } ^ { 2 - \gamma } + \displaystyle \sum _ { k = 1 } ^ { \infty } \left( \displaystyle \sum _ { i = 1 } ^ { n } a _ { i } e _ { 2 k } ( x _ { i } ) \right) ^ { 2 } \mu _ { 2 k } ^ { 2 - \gamma } . } \end{array}
$$

In this paper, we use the following truncation to approximate the learning rate

$$
\Vert f _ { n , \lambda } ^ { r p } - f _ { \rho } \Vert _ { [ H ] _ { p } ^ { \gamma } } ^ { 2 } \approx \sum _ { k = 1 } ^ { T } \left( \sum _ { i = 1 } ^ { n } a _ { i } e _ { 2 k - 1 } ( \mathbf { x } _ { i } ) - { \frac { 1 } { \mu _ { 2 k - 1 } k ^ { s + 0 . 5 } \log ( k + 1 ) } } \right) ^ { 2 } \mu _ { 2 k - 1 } ^ { 2 - \gamma } + \sum _ { k = 1 } ^ { T } \left( \sum _ { i = 1 } ^ { n } a _ { i } e _ { 2 k } ( \mathbf { x } _ { i } ) \right) ^ { 2 } \mu _ { 2 k } ^ { 2 - \gamma } .\tag{17}
$$

Here we set $T = 1 0 0 0 0$ . For the mis-specified case where $s = 0 . 5 .$ , we evaluate the Sobolev learning rates using $\gamma \in \{ 0 , 0 . 2 , 0 . 4 \}$ ; for the well-specified case $( s = 1 . 5$ and $s = 2 )$ , we use $\gamma \in \{ 0 , 0 . 4 , 1 \}$ . To empirically validate the learning rates derived in Corollary 2, we plot the test error against the training sample size � on a log-log scale. The theoretical learning rate has a slope of $- \frac { ( \bar { s } - \gamma ) \beta } { s \beta + 1 }$ on the log-log scale. Accordingly, we plot dashed reference lines with the theoretical slopes fixed and their intercepts determined by least-squares fitting to the log-transformed empirical results. All the results are averaged over 50 independent trials.

As shown in Figures 2, 3, and 4, the empirical learning rates of SARP align well with the theoretical findings in Corollary 2 for both mis-specified and well-specified cases. Since the theoretical learning rates are independent of the choice of spectral algorithms and projection methods once the projection dimension is chosen optimally, the empirical results also exhibit this consistency. For diferent Sobolev norms, we can see that the learning curves flatten as $\gamma$ increases, and the $L _ { \rho _ { \mathcal { X } } } ^ { 2 }$ error with $\gamma = 0$ converges the fastest, which is consistent with our theoretical findings. We also observe that SARP exhibits learning rates comparable to those of the corresponding full spectral algorithms across diferent projection methods and Sobolev norms.

![](images/d9ec649bb085791b15a3ddeecc656ac012b6b49f4cc7529516076093ae986753.jpg)  
Figure 3: Log empirical learning rates of SARP with diferent spectral algorithms and projection methods for the well-specified case when $s = 1 . 5$ averaged over 50 independent trials. The dashed lines represent the theoretical slopes. The gray lines represent the full spectral algorithms.

We note that the regression function $f _ { \rho }$ is not uniformly bounded when $s = 0 . 5$ , since $e _ { 2 k - 1 } ( 1 ) \equiv { \sqrt { 2 } } .$ This example confirms that the assumption of uniform boundedness on $f _ { \rho }$ is not necessary for deriving the optimal learning rates of SARP.

## 5.2. Real Applications

In this section, we apply SARP to two real-world datasets from the UCI repository: the Handwritten Digits USPS dataset (https://www.kaggle.com/datasets/bistaumanga/usps-dataset/data) and the cpusmall dataset (https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/regression.html) . The USPS dataset contains 7291 training and 2007 test images of size 16 × 16 pixels, while the cpusmall dataset consists of 8192 samples with 12 CPU characteristic features, where the target variable represents the percentage of time that CPUs run in user mode. For both datasets, we randomly select 80% of the samples for training and the remaining 20% for testing. Note that the USPS dataset corresponds to a classification task, whereas the cpusmall dataset corresponds to a regression task.

We implement SARP using a Gaussian RBF kernel with bandwidth 24, combined with KRR, GM, and SC as the spectral algorithms, and GAU, ROS, Ny-plain, and Ny-ALS as the projection methods. The regularization parameter � is chosen from $\{ 1 0 ^ { - 6 } , 1 0 ^ { - 5 } , \dots , 1 0 ^ { 0 } \}$ via the NNI toolbox (https://github.com/microsoft/nni). The projection dimension � takes values in {50, 100, 200, 400, 600, 800, 1000}. For each pair (�, �), we run SARP independently 20 times and report the average performance. The full SA implementations without projection are included as baselines.

![](images/ee2bb525a5838e59fc827d8276e6c2628792c85d072b1bc2aa2ac9a9fc969d34.jpg)  
Figure 4: Log empirical learning rates of SARP with diferent spectral algorithms and projection methods for the well-specified case when � = 2 averaged over 50 independent trials. The dashed lines represent the theoretical slopes. The gray lines represent the full spectral algorithms.

Classification error rate is used as the evaluation metric for the USPS dataset, while mean squared error (MSE) is used for the cpusmall dataset.

Figure 5 illustrates the performance of SARP with diferent spectral algorithms and projection methods on the USPS and cpusmall datasets. The results show that as the projection dimension � increases, SARP’s performance improves across all combinations of spectral algorithms and projection methods. Notably, when � reaches 600, SARP achieves stable performance, demonstrating its ability to reduce computational costs while maintaining high accuracy. Among the various projection methods, the sketching methods (GAU and ROS) and Nyström methods (Ny-plain and Ny-ALS) exhibit comparable performance, with no significant diference between them. However, Ny-ALS slightly outperforms Ny-plain, which is consistent with our theoretical findings that Ny-ALS requires a smaller projection dimension than Ny-plain to achieve optimal learning rates. As the projection dimension � increases, the classification error rates and MSEs of SARP gradually approach those of the corresponding full spectral algorithms. This is expected, since a larger projection dimension generally provides a more accurate approximation of the full kernel matrix, thereby reducing the approximation error introduced by random projection.

To evaluate the computational eficiency of diferent projection methods against the full SA algorithm without projection, we conduct a benchmark using the USPS dataset with projection dimension � = 500. All methods use a Gaussian RBF kernel with bandwidth 24 and a regularization parameter of $\lambda = 1 0 ^ { - 2 }$ . Each configuration is executed independently 20 times, and the average computational time and standard deviation are reported in Table 3. The results show that Ny-plain and Ny-ALS generally achieve lower computational costs than GAU and ROS. The SARP methods are substantially more eficient than their full SA counterparts, particularly for KRR and SC. However, for GM, GAU and ROS require more computational time than the full SA method. This may be attributed to the relatively low computational cost of the iterative GM algorithm, where the additional cost of constructing and applying random projections can outweigh their computational benefits. In contrast, Ny-plain and Ny-ALS remain more eficient than the full-kernel GM algorithm. Overall, these results suggest that suitable projection methods can significantly accelerate computation while maintaining comparable predictive performance.

![](images/ce9d018ce77467088d993fbb8203ce55795ce07889752316269ad8dd4a8debd2.jpg)  
Figure 5: Performance of SARP with diferent spectral algorithms and projection methods on the USPS and cpusmall datasets. The gray lines represent the full spectral algorithms.

Table 3  
Computational time (in seconds) of diferent spectral algorithms and projection methods for $m = 5 0 0$
<table><tr><td>Algorithm</td><td>GAU</td><td>ROS</td><td>NY</td><td>ALS</td><td>Full</td></tr><tr><td>KRR</td><td> $0 . 8 2 5 \pm 0 . 1 6 3$ </td><td> $1 . 3 3 9 \pm 0 . 6 1 2$ </td><td> $0 . 2 5 8 \pm 0 . 0 7 6$ </td><td> $0 . 2 3 8 \pm 0 . 0 1 0$ </td><td> $2 . 4 2 5 \pm 0 . 2 3 7$ </td></tr><tr><td>GM</td><td> $0 . 8 8 3 \pm 0 . 1 9 8$ </td><td> $1 . 3 3 2 \pm 0 . 6 2 8$ </td><td> $0 . 2 3 0 \pm 0 . 0 2 7$ </td><td> $0 . 2 7 4 \pm 0 . 0 0 2$ </td><td> $0 . 6 6 1 \pm 0 . 0 1 2$ </td></tr><tr><td>SC</td><td> $0 . 8 9 9 \pm 0 . 1 4 7$ </td><td> $1 . 3 9 1 \pm 0 . 5 9 0$ </td><td> $0 . 3 0 9 \pm 0 . 0 3 0$ </td><td> $0 . 2 6 7 \pm 0 . 0 1 1$ </td><td> $3 . 3 4 2 \pm 0 . 1 0 3$ </td></tr></table>

## 6. Conclusion

In this paper, we have established high-probability Sobolev norm error bounds for SARP under general source conditions. The results demonstrate that, by selecting an appropriate projection dimension and regularization parameter, these algorithms can achieve minimax optimal learning rates up to a logarithmic factor. Notably, our analysis extends to the mis-specified case with $s > \alpha _ { 0 } - 1 / \beta$ and Sobolev norms $\gamma \in [ 0 , 1 \land s ]$ , which were not fully addressed in previous works. Our findings contribute to a deeper understanding of the theoretical properties of SARP and provide practical insights for its implementation in large-scale learning tasks. Future research may proceed in several directions. First, our analysis techniques may apply to other scalable kernel learning algorithms, such as random features (Rudi and Rosasco, 2017; Wang and Feng, 2024) and stochastic gradient methods (Lin and Rosasco, 2016). Second, it is natural to extend our analysis to vector-valued settings, for example, in connection with conditional mean embeddings (Li, Meunier, Mollenhauer and Gretton, 2022). Third, one may relax the noise assumption to the heavy-tailed case and combine our SARP framework with robust concentration tools, which have recently been developed by Mollenhauer et al. (2026). Finally, a fully general information-theoretic lower bound on the projection dimension � for SARP remains an open problem.

## References

Ailon, N., Liberty, E., 2009. Fast dimension reduction using rademacher series on dual bch codes. Discrete & Computational Geometry 42, 615–630.

Alaoui, A., Mahoney, M.W., 2015. Fast randomized kernel ridge regression with statistical guarantees. Advances in Neural Information Processing Systems 28, 775–783.

Bach, F., 2013. Sharp analysis of low-rank kernel matrix approximations, in: Conference on Learning Theory, PMLR. pp. 185–209.

Bauer, F., Pereverzev, S., Rosasco, L., 2007. On regularization algorithms in learning theory. Journal of complexity 23, 52–72.

Bitzer, M., Steinwart, I., 2025. Spectral representations of interpolation spaces of reproducing kernel hilbert spaces. arXiv preprint arXiv:2508.16492

Blanchard, G., Bousquet, O., Zwald, L., 2007. Statistical properties of kernel principal component analysis. Machine Learning 66, 259–294.

Calandriello, D., Lazaric, A., Valko, M., 2017. Distributed adaptive sampling for kernel matrix approximation, in: Artificial Intelligence and Statistics, PMLR. pp. 1421–1429.

Caponnetto, A., De Vito, E., 2007. Optimal rates for the regularized least-squares algorithm. Foundations of Computational Mathematics 7, 331–368.

Chen, Y., Yang, Y., 2021. Fast statistical leverage score approximation in kernel ridge regression, in: International Conference on Artificial Intelligence and Statistics, PMLR. pp. 2935–2943.

De Vito, E., Rosasco, L., Caponnetto, A., 2006. Discretization error analysis for tikhonov regularization. Analysis and Applications 4, 81–99.

De Vito, E., Rosasco, L., Caponnetto, A., De Giovannini, U., Odone, F., 2005. Learning from examples as an inverse problem. Journal of Machine Learning Research 6, 883–904.

Dicker, L.H., Foster, D.P., Hsu, D., 2017. Kernel ridge vs. principal component regression: Minimax bounds and the qualification of regularization operators. Electronic Journal of Statistics 11, 1022–1047.

Engl, H.W., Hanke, M., Neubauer, A., 1996. Regularization of inverse problems. volume 375. Springer Science & Business Media.

Fischer, S., Steinwart, I., 2020. Sobolev norm learning rates for regularized least-squares algorithms. Journal of Machine Learning Research 21, 1–38.

Guo, Z.C., Lin, S.B., Zhou, D.X., 2017. Learning theory of distributed spectral algorithms. Inverse Problems 33, 074009.

Kumar, S., Mohri, M., Talwalkar, A., 2012. Sampling methods for the nyström method. The Journal of Machine Learning Research 13, 981–1006.

Li, Y., Gan, W., Shi, Z., Lin, Q., 2026. Generalization error curves for analytic spectral algorithms under power-law decay. Applied and Computationa Harmonic Analysis , 101920.

Li, Y., Zhang, H., Lin, Q., 2023. On the saturation efect of kernel ridge regression, in: The Eleventh International Conference on Learning Representations.

Li, Z., Meunier, D., Mollenhauer, M., Gretton, A., 2022. Optimal rates for regularized conditional mean embedding learning. Advances in Neural Information Processing Systems 35, 4433–4445.

Lin, J., Cevher, V., 2020a. Convergences of regularized algorithms and stochastic gradient methods with random projections. Journal of Machine Learning Research 21, 1–44.

Lin, J., Cevher, V., 2020b. Optimal convergence for distributed learning with stochastic gradient methods and spectral algorithms. The Journal of Machine Learning Research 21, 5852–5914.

Lin, J., Rosasco, L., 2016. Optimal learning for multi-pass stochastic gradient methods. Advances in Neural Information Processing Systems 29, 4563–4571.

Lu, W., Li, Y., Lin, Q., et al., 2024. On the saturation efects of spectral algorithms in large dimensions. Advances in Neural Information Processing Systems 37, 7011–7059.

Mahoney, M.W., et al., 2011. Randomized algorithms for matrices and data. Foundations and Trends® in Machine Learning 3, 123–224.

Mollenhauer, M., Mücke, N., Meunier, D., Gretton, A., 2026. Regularized least squares learning with heavy-tailed noise is minimax optimal. Advances in Neural Information Processing Systems 38, 125703–125743.

Musco, C., Musco, C., 2017. Recursive sampling for the nyström method. Advances in Neural Information Processing Systems 2017, 3834–3846.

Myleiko, G., Pereverzyev Jr, S., Solodky, S., 2019. Regularized nyström subsampling in regression and ranking problems under general smoothness i l i d li i 5 5

Rahimi, A., Recht, B., 2007. Random features for large-scale kernel machines. Advances in Neural Information Processing Systems 20, 1177–1184.

Rastogi, A., Sampath, S., 2017. Optimal rates for the regularized learning algorithms under general source condition. Frontiers in Applied Mathematics and Statistics 3, 3–17.

Rosasco, L., De Vito, E., Verri, A., 2005. Spectral methods for regularization in learning theory. DISI, Universita degli Studi di Genova, Italy, Technical Report DISI-TR-05-18 .

Rudi, A., Calandriello, D., Carratino, L., Rosasco, L., 2018. On fast leverage score sampling and optimal learning. Advances in Neural Information Processing Systems 31, 5677–5687.

Rudi, A., Camoriano, R., Rosasco, L., 2015. Less is more: Nyström computational regularization. Advances in Neural Information Processing Systems 28, 1657–1665.

Rudi, A., Rosasco, L., 2017. Generalization properties of learning with random features. Advances in Neural Information Processing Systems 30, 3215–3225.

Schölkopf, B., Smola, A.J., 2002. Learning with kernels: support vector machines, regularization, optimization, and beyond. MIT Press.

Smale, S., Zhou, D.X., 2007. Learning theory estimates via integral operators and their approximations. Constructive Approximation 26, 153–172.

Steinwart, I., Christmann, A., 2008. Support vector machines. Springer Science & Business Media.

Steinwart, I., Scovel, C., 2012. Mercer’s theorem on general domains: On the interaction between measures, kernels, and RKHSs. Constructive Approximation 35, 363–417.

Steinwart, I., Scovel, C., et al., 2009. Optimal rates for regularized least squares regression, in: Proceedings of the Annual Conference on Learning Theory, 2009, pp. 79–93.

Tsybakov, A.B., 2008. Nonparametric estimators, in: Introduction to Nonparametric Estimation. Springer, pp. 1–76.

Vapnik, V., 1999a. The nature of statistical learning theory. Springer Science & Business Media.

Vapnik, V.N., 1999b. An overview of statistical learning theory. IEEE Transactions on Neural Networks 10, 988–999.

Vempala, S.S., 2005. The random projection method. volume 65. American Mathematical Soc.

Wahba, G., 1990. Spline models for observational data. SIAM.

Wang, C., Feng, X., 2024. Optimal kernel quantile learning with random features, in: International Conference on Machine Learning, PMLR. pp. 50419–50452.

Williams, C., Seeger, M., 2001. Using the nystroem method to speed up kernel machines. Advances in Neural Information Processing Systems , 682–688.

Yang, Y., Pilanci, M., Wainwright, M.J., 2017. Randomized sketches for kernels: Fast and optimal nonparametric regression. The Annals of Statistics 45, 991 – 1023.

Yao, Y., Rosasco, L., Caponnetto, A., 2007. On early stopping in gradient descent learning. Constructive Approximation 26, 289–315.

Zhang, H., Li, Y., Lin, Q., 2024. On the optimality of misspecified spectral algorithms. Journal of Machine Learning Research 25, 1–50.

Zhang, H., Li, Y., Lu, W., Lin, Q., 2023. On the optimality of misspecified kernel ridge regression, in: Proceedings of the 40th International Conference on Machine Learning, PMLR. pp. 41331–41353.

Zhang, X., Liao, S., 2019. Incremental randomized sketching for online kernel learning, in: International Conference on Machine Learning, PMLR. pp. 7394–7403.

Zhang, Y., Duchi, J., Wainwright, M., 2015. Divide and conquer kernel ridge regression: A distributed algorithm with minimax optimal rates. The Journal of Machine Learning Research 16, 3299–3340.