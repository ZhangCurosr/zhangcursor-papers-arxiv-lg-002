# Tensor Network Kernel Machines for Nonlinear System Identification

Kim Batselier <sup>∗</sup>

Delft University of Technology, Delft, The Netherlands (e-mail: k.batselier@tudelft.nl).

Abstract: Over the past ten years Tensor Network Kernel Machines have enabled eficient nonlinear system identification through the use of low-rank tensor networks. The goal of this article is to provide the reader with a basic understanding of this model class in terms of its two defining properties. Furthermore, both the optimization and regularization of such models are discussed, together with recent developments and starting points for further exploration.

Keywords: Nonlinear system identification, Kernel methods, Tensors, Tensor decompositions, Tensor networks

## 1. INTRODUCTION

Kernel machines have a rich history of being applied to system identification problems <sup>1</sup> . The first occurrence of their use was in 1998 with Support Vector Machines (SVM) (Drezet and Harrison, 1998). The reformulation of SVMs into Least Squares-Support Vector Machines (LS-SVM) (Suykens et al., 2002) was a seminal result in the sense that it replaced a Quadratic Program with a simple linear least squares problem. Consequently, LS-SVMs[ quickly found their use in system identification. Among the applications are partially linear systems (Espinoza et al., 2005), Hammerstein systems (Goethals et al., 2005), Wiener-Hammerstein systems (Falck et al., 2009), and LPV-ARX models (T´oth et al., 2011), to name a few.

At around the same time when LS-SVMs were being developed, Gaussian Processes (GPs) attracted a lot of attention in the machine learning community (Williams and Rasmussen, 2006). The main diference between GPs and LS-SVMs comes from the former being derived from a probabilistic point of view. The main advantage of GPs is that confidence bounds on predictions become straightforward to compute. Just like their deterministic LS-SVM counterparts, GPs have found ample use in system identification (S¨arkk¨a, 2021).

The main advantage of kernel machines is their primaldual formulation, giving rise to a great amount of flexibility. The primal formulation considers a feature map $\varphi ( { \pmb x } )$ that maps $\pmb { x } \in \mathbb { R } ^ { D }$ to some P-dimensional (possibly infinite) feature space. The dual formulation on the other hand considers a kernel function $\kappa ( \pmb { x } _ { i } , \pmb { x } _ { j } )$ that computes some measure of similarity between data points $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ and $\mathbf { \delta } _ { \mathbf { x } _ { j } } .$ . The zero duality gap then implies that the solutions of the primal and dual problem are identical, thus giving rise to the aforementioned flexibility. When the sample size N is much larger than the dimension of the feature space $P ,$ then solving the primal problem is computationally advantageous with a complexity $O ( P ^ { 3 } )$ ). When P is much larger than N, then solving the dual problem is more advantageous with a complexity $O ( N ^ { 3 } )$

But what if both N and P are so large that neither the primal nor the dual problem can be solved? A Tensor Network Kernel Machine (TNKM) solves exactly this problem (Batselier, 2022). The key ingredient for TNKM’s success lies in the combination of a tensor network (TN) parameterization with a Kronecker product structured feature vector. By exploiting these structures the nonlinear system identification problem can be solved for $P = I ^ { D }$ parameters and N samples with a computational complexity of ${ \cal O } ( N D I R + I ^ { 3 } R ^ { 3 } )$ , rather than $\dot { O } ( I ^ { 3 D } )$ or $O ( N ^ { 3 } )$

The main content of this article is divided into three sections. In Section 2 the TNKM model is discussed in terms of its two defining characteristics: a Kronecker product structure combined with a TN parameterization. Section 3 goes into the optimization and regularization of TNKMs. Finally, Section 4 mentions recent TNKM developments.

## 2. TENSOR NETWORK KERNEL MACHINES

## 2.1 Notation

A tensor $\pmb { \mathcal { A } } \in \mathbb { R } ^ { I _ { 1 } \times I _ { 2 } \times \dots \times I _ { D } }$ is a multidimensional array, always denoted by a calligraphic uppercase boldface letter. Scalars are denoted by lowercase letters, vectors by lowercase boldface and matrices by uppercase boldface. The D entries of a vector x $\in \mathbb { R } ^ { D }$ are written as $x _ { 1 } , \ldots , x _ { D }$ . Onebased indexing is used and the maximal value an index can attain is always its corresponding uppercase letter. For example, the index i satisfies $1 \leq i \leq \bar { I }$

## 2.2 The Model

To keep notation simple we only consider the single output case throughout this article. Consider a nonlinear system identification problem with data $\textbf { \em D } : = \{ ( \boldsymbol { x } _ { n } , y _ { n } ) \} _ { n = 1 } ^ { N }$ The vector ${ \pmb x } _ { n }$ contains all past information up to time $n - 1$ , necessary to predict the output $y _ { n }$ at time n.

x<sub>n</sub> Model Structure u<sub>n−1</sub>, u<sub>n−2</sub>, . . . NFIR y<sub>n−1</sub>, y<sub>n−2</sub>, . . . NAR u<sub>n−1</sub>, u<sub>n−2</sub>, . . . , y<sub>n−1</sub>, y<sub>n−2</sub>, . . . NARX u<sub>n−1</sub>, u<sub>n−2</sub>, . . . , yˆ<sub>n−1</sub>, yˆ<sub>n−2</sub>, . . . NOE $u _ { n - 1 } , u _ { n - 2 } , \ldots , y _ { n - 1 } , y _ { n - 2 } , \ldots , e _ { n - 1 } , e _ { n - 2 } , \ldots .$ NARMAX

Table 1. Common choices for ${ \mathbf { \mathcal { x } } } _ { n }$ in a TNKM, resulting in diferent nonlinear model structures.

Before defining a TNKM, we first consider the $\mathrm { S V M / G P }$ regression model

$$
y _ { n } = { \hat { y } } ( \mathbf { x } _ { n } ) + e _ { n } = \varphi ( \mathbf { x } _ { n } ) ^ { \intercal } \mathbf { \Sigma } w + e _ { n } .\tag{1}
$$

The nonlinear 1-step ahead predictor ${ \hat { y } } ( \pmb { x } _ { n } )$ is parameterized as the dot product between a feature vector $\varphi ( { \pmb x } _ { n } )$ and model parameters w. The white noise disturbance process is denoted $e _ { n }$

A TNKM is defined by model equation (1) and the following two properties:

(1) $\varphi ( \pmb { x } _ { n } ) \in \mathbb { R } ^ { I ^ { D } }$ has a Kronecker product structure

$$
\varphi ( \pmb { x } _ { n } ) : = \varphi _ { D } \otimes \cdots \otimes \varphi _ { 2 } \otimes \varphi _ { 1 } .\tag{2}
$$

(2) $\boldsymbol { w } \in \mathbb { R } ^ { I ^ { D } }$ is parameterized as a tensor network.

These two properties complement one another, enabling TNKMs to identify highly-nonlinear dynamical systems very eficiently. The reason for this eficiency is that instead of solving the identification problem for w, only its tensor network components need to be identified instead. We unpack these two defining properties in the next two subsections.

## 2.3 Kronecker product feature vector $\varphi ( { \pmb x } _ { n } )$

The feature vector $\varphi ( { \pmb x } _ { n } )$ contains a collection of multivariate basis functions in the variables ${ \mathbf { x } } _ { n } .$ The nonlinear 1-step ahead predictor ${ \hat { y } } ( \pmb { x } _ { n } )$ is then obtained through a linear combination of these basis functions with appropriate weights w. The Kronecker product structure in (2) is merely a way to construct $I ^ { D }$ multivariate basis functions from products of I univariate basis functions.

Diferent choices of ${ \bf { x } } _ { n }$ result in diferent model structures. A direct feedthrough term can be easily added through the inclusion of $u _ { n }$ into x. An overview of common choices together with the resulting model structures are given in Table 1. All of these features can easily describe multiinput systems by replacing all scalars $u _ { n - 1 } , u _ { n - 2 }$ in x with their corresponding vectors. TNKMs are not restricted to input-output parameterizations. Nonlinear state space models of the form ${ \dot { \boldsymbol { x } } } = f ( { \boldsymbol { x } } )$ with fully observed state information are TNKMs when choosing $y = \dot { \pmb { x } } . \mathrm { ~ A ~ }$ welldeveloped model in this nonlinear state space setting that uses this idea is the multilinear time-invariant (MTI) model (Lichtenberg, 2011).

We now discuss some common choices for basis functions. For notational clarity we drop the discrete-time subscript notation such that ${ \pmb x } _ { n }$ is replaced with x from now on.

Monomial basis functions. The Stone-Weierstrass theorem (De Branges, 1959) guarantees that any continuous function on a locally compact domain can be approximated arbitrarily well by polynomials of increasing degree. This fact serves as a motivation to approximate the 1-step ahead predictor ${ \hat { y } } ( { \pmb x } )$ by a multivariate polynomial of total degree D. All monomials from total degree 0 up to D can be constructed through the Kronecker product (Batselier et al., 2017b)

$$
\varphi ( \pmb { x } ) ^ { \intercal } : = \underbrace { ( 1 \ \pmb { x } ) \otimes \dots \otimes ( 1 \ \pmb { x } ) } _ { D { \mathrm { ~ t i m e s } } } .\tag{3}
$$

Setting x equal to M past inputs $u _ { n - 1 } , \ldots , u _ { n - M }$ results in a truncated Volterra model (Batselier et al., 2017b). The monomial basis feature vector in (3) is equivalent with the polynomial kernel since $\varphi ( \pmb { x } ) ^ { \top } \varphi ( \pmb { x } ) = ( \hat { 1 } + \pmb { x } ^ { \top } \pmb { x } ) ^ { D }$ . In the context of TNKMs, however, we will not make further use of kernel functions since the whole idea will be to impose a tensor network structure onto w and exploit this structure for storage and computational gains.

Pure-power polynomial basis functions. Pure-power polynomial features (Chen et al., 2017) are an alternative to construct polynomial nonlinearities. For a vector $\pmb { x } \in \mathbb { R } ^ { D }$ they are defined as

$$
\varphi ( \pmb { x } ) ^ { \mathsf { T } } : = \bigotimes _ { d = 1 } ^ { D } \left( 1 ~ x _ { d } ~ x _ { d } ^ { 2 } ~ \cdots ~ x _ { d } ^ { I - 1 } \right) ,
$$

where $x _ { d }$ denotes the dth entry of x. Each factor in the Kronecker product therefore consists of a specific $x _ { d } ,$ in contrast to the monomial basis where all $x _ { d }$ components are present in each factor. Generating $I ^ { D }$ monomials using pure-power polynomial features results in more higherorder terms compared to the monomial basis functions. Similarly, the Stone-Weierstrass theorem applies.

Fourier Features. Two kinds of Kronecker product Fourier Features have been described in the literature, both approximate the Gaussian kernel function. The first option (Wesel and Batselier, 2021) considers the domain of the 1-step ahead predictor to be the hypercube $[ - U , U ] ^ { D }$ with vanishing boundary conditions. The ith entry of the dth feature vector in that case is defined as

$$
( \varphi _ { d } ) _ { i } : = \frac { 1 } { \sqrt { U } } p \left( \frac { i \pi } { 2 U } \right) \sin \bigg ( \frac { i \pi ( x _ { d } + U ) } { 2 U } \bigg ) ,
$$

where $p ( \cdot )$ is the spectral density of the one-dimensional Gaussian kernel. Each $\varphi _ { d }$ vector has I associated frequencies which are mixed via products into a total of $I ^ { \vec { D } }$ frequencies. Removing the zero boundary conditions on the domain requires adding I cosine contributions. Alternatively, an 1-step ahead predictor with period $L \in \mathbb { R }$ (Wesel and Batselier, 2024) can be modeled through the features

$$
( \varphi _ { d } ) _ { i } : = e ^ { 2 \pi j x _ { d } \frac { 2 + I } { 2 L } } \left( e ^ { - \frac { 2 \pi j x _ { d } } { L } } \right) ^ { ( i - 1 ) } ,
$$

with $j$ the imaginary unit and again I frequencies associated with each feature vector $\varphi _ { d }$

Quantized features. The pure-power polynomial and complex exponential Fourier features can be represented even more eficiently (Wesel and Batselier, 2024) when the total number of basis functions I satisfies $I = Q ^ { K }$ , with Q, K positive integers. The features in this case can be further decomposed into a Kronecker product, giving rise to so-called quantized features. For example, when $I = 2 ^ { 3 }$ then

$$
\left( 1 \ x _ { d } \ x _ { d } ^ { 2 } \ \cdot \cdot \cdot \ x _ { d } ^ { 7 } \right) = \left( 1 \ x _ { d } \right) \otimes \left( 1 \ x _ { d } ^ { 2 } \right) \otimes \left( 1 \ x _ { d } ^ { 4 } \right) .
$$

A similar relation holds for the complex exponential Fourier features. This result has major consequences for the parameterization of w. A theoretical derivation through the VC-dimension and various numerical experiments show that the parameterization of w that uses quantized features have better generalization properties compared to their conventional counterparts (Wesel and Batselier, 2024).

B-splines. B-splines are piecewise polynomials that can approximate the 1-step ahead predictor in a hypercube $[ \bar { - } \bar { U } , U ] ^ { D }$ up to any desired accuracy. Each dimension of the hypercube is first uniformly divided into M intervals $t _ { m } ~ \le ~ x ~ \le ~ t _ { m + 1 }$ also called the knot sequence $\left\{ t _ { 0 } , t _ { 1 } , \dots , t _ { M - 1 } , t _ { M } \right\}$ . This division determines the accuracy of the approximation, where larger M results in a better approximation. A B-spline basis function $B _ { i , \rho }$ is a polynomial of degree $\rho$ that remains nonzero over $\rho + 1$ consecutive intervals. The dth feature vector is hence defined as

$$
( \varphi _ { d } ) ^ { \intercal } : = ( B _ { 1 , \rho } ( x _ { d } ) \ B _ { 2 , \rho } ( x _ { d } ) \ \cdot \cdot \cdot \ B _ { I , \rho } ( x _ { d } ) ) ,
$$

with a total of $\begin{array} { r } { \begin{array} { l } { \displaystyle { I } } \end{array} = \begin{array} { l } { \displaystyle { M } \mathrm { ~ - ~ } \rho } \end{array} } \end{array}$ basis functions. A nice property of these basis functions is that the smoothness of the resulting 1-step ahead predictor can be directly controlled by limiting the diference between consecutive weights through regularization (Karagoz and Batselier, 2020). Furthermore, such regularization can be encoded as a low-rank tensor network, which enables an eficient implementation.

## 2.4 Tensor Network Parameterization of w

The second defining property of a TNKM is that $\boldsymbol { w } \in \mathbb { R } ^ { I ^ { D } }$ is parameterized by a tensor network. For the context of this article, a tensor network $\mathrm { T N } ( \cdot , \ldots , \cdot )$ is a multilinear function of D TN-cores $\boldsymbol { \nu } _ { 1 } , \ldots , \boldsymbol { \nu } _ { D }$ such that

$$
\pmb { w } = \mathrm { T N } ( \pmb { \mathcal { W } } _ { 1 } , \pmb { \mathcal { W } } _ { 2 } , \dots , \pmb { \mathcal { W } } _ { D } ) .\tag{4}
$$

TN-cores are small tensors that can be interpreted as fully describing w. A function is multilinear when it is a linear function in any $w _ { d }$ whilst keeping all other cores fixed. Therefore, equation (4) can always be rewritten as

$$
\pmb { w } = \pmb { U } _ { d } \mathrm { v e c } ( \pmb { \mathcal { W } } _ { d } ) ,\tag{5}
$$

for any of the D TN-cores $w _ { d } .$ . The matrix $U _ { d }$ depends on all other TN-cores $k \neq d ,$ and the vec(·) operation stacks all entries of $w _ { d }$ into a vector. The storage complexity of $\mathrm { ~ a ~ T N }$ is simply the storage complexity of the D TN-cores. For example, when each TN-core is a matrix of size $I \times R$ then the total storage cost of the TN is DIR. For low R, this constitutes a massive saving compared to the explicit storage of the $I ^ { D }$ entries of w.

By replacing w in ${ \hat { y } } _ { n }$ with a TN, the $I ^ { D }$ entries in w are never explicitly stored and the estimation of w becomes the estimation of the TN-cores $w _ { d }$ instead. In addition to obvious storage gains, significant computational gains can also be obtained through the development of optimization algorithms that exploit the tensor network structure of ${ \hat { y } } _ { n } .$ We now discuss the two most commonly-used tensor networks in TNKMs.

Canonical Polyadic Decomposition (CPD). The TNcores of a CPD are matrices $\hat { W } _ { d } \in \mathbb { R } ^ { I \times R }$ such that

$$
{ \pmb w } = \sum _ { r = 1 } ^ { R } { \pmb w } _ { D } ^ { r } \otimes \cdots \otimes { \pmb w } _ { 1 } ^ { r } ,\tag{6}
$$

where ${ \pmb w } _ { d } ^ { r }$ denotes the vector obtained from setting the index r in $W _ { d }$ to a fixed value, and R is called the CPD-rank (Hitchcock, 1927). The CPD-rank directly determines the “richness” of the set of w vectors that are described by the CPD. Increasing R always decreases the loss on identification data but at some point overfitting will occur. In Section 3 we will explain how overfitting can be countered through regularization.

One of the main benefits of using a CPD now becomes clear, the original $I ^ { D }$ storage cost is reduced to DIR, since only D matrices $W _ { d } .$ , each of size $I \times R ,$ need to be stored. The equivalent of equation (5) for the CPD is

$$
w = \underbrace { \left( \bigodot W _ { k } \otimes I _ { I } \right) } _ { U _ { d } } \mathrm { v e c } ( W _ { d } ) ,\tag{7}
$$

with ⊙ the column-wise Kronecker product and $\pmb { I } _ { I }$ the $I \ \times \ I$ identity matrix. Substituting equation (7) into equation (1) together with the Kronecker product for $\varphi$ results in an expression for the 1-step ahead predictor that is linear in $W _ { d }$

$$
\begin{array} { r l } & { \hat { \boldsymbol { y } } _ { n } = ( \varphi _ { D } \otimes \cdots \cdot \cdot \otimes \varphi _ { 1 } ) ^ { \intercal } \left( \underset { k \neq d } { \bigodot } W _ { k } \otimes I _ { I } \right) \mathrm { v e c } ( W _ { d } ) , } \\ & { \quad \quad = \left( \underset { k \neq d } { \bigodot } \varphi _ { k } ^ { \intercal } W _ { k } \otimes \varphi _ { d } ^ { \intercal } \right) \mathrm { v e c } ( W _ { d } ) , } \end{array}
$$

where $\circledast$ is the elementwise (Hadamard) product. The computational cost for evaluating the conventional predictor ${ \hat { y } } _ { n }$ for N samples is $O ( N I ^ { D } )$ , while for the TNKM it is $O ( \bar { N } D I R )$ (Lichtenberg, 2011; Wesel and Batselier, 2021). The exponential complexity has been completely resolved!

Tensor Train (TT). The TN-cores of a TT are 3rd-order tensors $\pmb { \mathcal { W } } _ { d } \in \dot { \mathbb { R } } ^ { R _ { d - 1 } \prime \times I \times R _ { d } }$ such that

$$
\pmb { w } = \sum _ { r _ { 1 } = 1 } ^ { R _ { 1 } } \cdot \cdot \cdot \sum _ { r _ { D - 1 } = 1 } ^ { R _ { D - 1 } } \pmb { w } _ { D } ^ { r _ { D - 1 } , r _ { D } } \otimes \cdot \cdot \cdot \otimes \pmb { w } _ { 1 } ^ { r _ { 0 } , r _ { 1 } } ,\tag{8}
$$

where ${ w _ { d } ^ { r _ { d - 1 } , r _ { d } } }$ denotes the vector obtained from setting the $r _ { d - 1 }$ and $r _ { d }$ indices in $w _ { d }$ to fixed values (Oseledets, 2011). In addition, boundary conditions $R _ { 0 } \stackrel { . } { = } R _ { D } = 1$ are required for the right-hand side in equation (8) to be a vector. The most notable diference with the CPD is that a TT has D −1 diferent TT-ranks $R _ { 1 } , R _ { 2 } , \ldots , R _ { D - 1 }$ The total storage cost of a TT is $O ( D I R ^ { 2 } )$ , assuming equal TT-ranks. This multitude of ranks gives the TT more flexibility in describing w, since a total of DIR TNparameters can be attained by possibly several diferent configurations of TT-ranks. In contrast, only a CPD-rank R results in a total number of DIR TN-parameters. The same argument on overfitting applies to TTs.

In order to derive equation (5) for TTs, we first need the following intermediate definitions

$$
\begin{array} { r l } & { { \cal W } _ { < d } : = \displaystyle \sum _ { r _ { 1 } = 1 } ^ { R _ { 1 } } \cdot \cdot \cdot \sum _ { r _ { d - 2 } = 1 } ^ { R _ { d - 2 } } { \pmb w } _ { d - 1 } ^ { r _ { d - 2 } , r _ { d - 1 } } \otimes \cdot \cdot \cdot \otimes { \pmb w } _ { 1 } ^ { r _ { 0 } , r _ { 1 } } , } \\ & { { \pmb W } _ { > d } : = \displaystyle \sum _ { r _ { d + 1 } = 1 } ^ { R _ { d + 1 } } \cdot \cdot \cdot \sum _ { r _ { D - 1 } = 1 } ^ { R _ { D - 1 } } { \pmb w } _ { D } ^ { r _ { D - 1 } , r _ { D } } \otimes \cdot \cdot \cdot \otimes { \pmb w } _ { d + 1 } ^ { r _ { d } , r _ { d + 1 } } , } \end{array}
$$

From these definitions it follows that $W _ { < d } \in \mathbb { R } ^ { I ^ { d - 1 } \times R _ { d - 1 } }$ and $W _ { > d } ~ \in ~ \mathbb { R } ^ { I ^ { D - d } \times R _ { d } }$ . Together with the boundary conditions $W _ { < 1 } = W _ { > d } = 1$ we can then write

$$
{ \pmb w } = ( { \pmb W } _ { > d } \otimes { \pmb I } _ { I } \otimes { \pmb W } _ { < d } ) \ \mathrm { v e c } ( { \pmb W } _ { d } )\tag{9}
$$

In order to derive the 1-step ahead predictor ${ \hat { y } } _ { n }$ using a TT parameterization of w we first need similar intermediate definitions for the feature vector

$$
\begin{array} { r l } & { \varphi _ { < d } : = \varphi _ { d - 1 } \otimes \cdots \otimes \varphi _ { 1 } \in \mathbb { R } ^ { I ^ { d - 1 } } , } \\ & { \varphi _ { > d } : = \varphi _ { D } \otimes \cdots \otimes \varphi _ { d + 1 } \in \mathbb { R } ^ { I ^ { D - d } } . } \end{array}
$$

These allow us to write

$$
\begin{array} { r l } & { \hat { y } _ { n } = \left( \varphi _ { > d } \otimes \varphi _ { d } \otimes \varphi _ { < d } \right) ^ { \top } ~ \left( W _ { > d } \otimes I _ { I } \otimes W _ { < d } \right) ~ \mathrm { v e c } ( \mathcal { W } _ { d } ) , } \\ & { \quad = \left( \varphi _ { > d } ^ { \top } W _ { > d } \otimes \varphi _ { d } ^ { \top } \otimes \varphi _ { < d } ^ { \top } W _ { < d } \right) ~ \mathrm { v e c } ( \mathcal { W } _ { d } ) . } \end{array}
$$

The TT parameterization of w allows the 1-step ahead predictor to be evaluated for N samples with a computational complexity O(NDIR<sup>2</sup>) (Batselier et al., 2017b).

## 3. OPTIMIZATION AND REGULARIZATION

Two choices are essential when identifying nonlinear systems as TNKMs. The first choice is the model structure, which is completely determined through x. The second choice is the TN parameterization of w. An estimate for w is found from solving

$$
\operatorname* { m i n } _ { \mathcal { W } _ { 1 } , \dots , \mathcal { W } _ { D } } \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \left\| y _ { n } - \hat { y } _ { n } \right\| _ { 2 } ^ { 2 } + \mathcal { R } ( \mathcal { W } _ { 1 } , \dots , \mathcal { W } _ { D } ) ,\tag{10}
$$

where both the model fit term and regularization term depend on the TN-cores $\mathcal { W } _ { 1 } , \ldots , \mathcal { W } _ { D }$ . The ℓ<sub>2</sub>-norm for the model fit term is the most common choice, but other choices are possible. In contrast to the conventional LS-SVM or GP model, the TNKM 1-step ahead predictor ${ \hat { y } } _ { n }$ is a multilinear function of the TN-cores, rendering the optimization problem nonlinear.

## 3.1 Optimization

In principle, one can use any of-the-shelf method for solving (10). Common choices when N is large are Stochastic Gradient Descent (Bottou et al., 2018) and the Adam optimizer (Kingma and Ba, 2014). These methods come with great flexibility and ease-of-use, but at the cost of not being specifically tailored to exploit the existing tensor network structure. Updating all TN-cores at once also becomes computationally expensive for larger D and TNranks R.

Exploiting this particular multilinear structure can be easily done with a nonlinear Gauss Seidel method, also called block coordinate descent. The main idea is to initialize the TN-cores and update only one keeping all others fixed. The multilinear structure turns this update into a simple linear least squares problem, which can be repeated for all TN-cores until convergence. The name for this procedure is ALS, which stands for alternating least squares or alternating linear scheme (Holtz et al., 2012; Uschmajew, 2012).

Derivation of the linear system for a TN-core update due to the model fit term in (10) is straightforward. Let $\Phi \in \mathbb { R } ^ { N \times I ^ { D } }$ be the matrix where row n is the feature vector $\varphi ( { \pmb x } _ { n } ) ^ { \top }$ , and $\boldsymbol { y } \in \mathbb { R } ^ { N }$ the vector of outputs. The model fit term is then

$$
\frac { 1 } { N } \left( \pmb { y } - \pmb { \Phi } \pmb { U } _ { d } \operatorname { v e c } ( \pmb { \mathcal { W } } _ { d } ) \right) ^ { \top } \ \left( ( \pmb { y } - \pmb { \Phi } \pmb { U } _ { d } \operatorname { v e c } ( \pmb { \mathcal { W } } _ { d } ) ) , \right.
$$

leading to the normal equations

$$
U _ { d } ^ { \mathsf { T } } \Phi ^ { \mathsf { T } } \Phi U _ { d } { \mathrm { ~ v e c } } ( \pmb { \mathcal { W } } _ { d } ) = U _ { d } ^ { \mathsf { T } } \Phi ^ { \mathsf { T } } \pmb { y } ,
$$

which can be solved analytically in $O ( I ^ { 3 } R ^ { 3 } )$ flops for the CPD and $O ( I ^ { 3 } R ^ { 6 } )$ flops for the TT.

Combining this ALS procedure with gradient boosting results in a fast automatic determination of the TNranks removing the need for cross-validation, together with reduced peak memory usage (Saiapin and Batselier, 2026b). The regularization term R adds an additional term to the left-hand side of the normal equations but does not afect the computational complexity in terms of O(·) flop count. We now discuss some commonly-used choices.

## 3.2 Regularization

Several diferent choices for the regularization term R have been described in the literature. From a Bayesian perspective, each of these can be interpreted as coming from specific priors.

Tikhonov. Tikhonov regularization is the most commonlyused regularization both in SVMs and GPs, defined as

$$
\begin{array} { r } { \mathcal { R } ( \mathcal { W } _ { 1 } , \dots , \mathcal { W } _ { D } ) : = { \left. { L \pmb { w } } \right. } _ { 2 } ^ { 2 } . } \end{array}\tag{11}
$$

Using this kind of regularization with $L : = \lambda I$ together with a kernel function is known as kernel ridge regression. From a Bayesian perspective equation (11) represents a zero-mean Gaussian prior on w with precision matrix $\pmb { L } \pmb { \intercal } \pmb { L }$ . As the total number of model parameters in a TNKM depends on the TN-ranks, overfitting becomes more likely with higher ranks. Tikhonov regularization alleviates this risk, making the final result less sensitive to “badly chosen” TN-ranks by adding the term

$$
U _ { d } ^ { \top } L ^ { \top } L U _ { d } \operatorname { v e c } ( { \mathcal { W } } _ { d } ) ,
$$

to the ALS normal equations (Chen et al., 2017). The $U _ { d }$ matrix needs to be recomputed after each update. When $L : = \lambda I .$ the non-uniqueness of a TT can be exploited to turn $U _ { d }$ into a semi-orthogonal matrix. This reduces the regularization term to $\lambda I \mathrm { v e c } (  { \boldsymbol { W } } _ { d } )$ and completely avoids recomputing $U _ { d }$

Independent cores. Instead of putting a prior on the whole w vector, one can also consider independent priors on each of the separate TN-cores $w _ { d } .$ The Tikhonov regularization term then becomes

$$
\mathcal { R } ( \mathcal { W } _ { 1 } , \ldots , \mathcal { W } _ { D } ) : = \sum _ { d = 1 } ^ { D } \| \pmb { L } _ { d } \operatorname { v e c } ( \pmb { \mathcal { W } } _ { d } ) \| _ { 2 } ^ { 2 } ,
$$

with $\pmb { L } _ { d } ^ { \top } \pmb { L } _ { d }$ vec $( \mathcal { W } _ { d } )$ the corresponding term in the ALS normal equations. Choosing $\bar { \cal L } _ { d } : = \bar { \lambda _ { d } \cal I }$ has been shown to make the final identified model insensitive to “badly chosen” TN-ranks (Memmel et al., 2023).

A particular useful choice for $\mathbf { { \mathit { L } } } _ { d }$ is the αth-order diference matrix in the case of B-spline basis functions (Karagoz and Batselier, 2020). Penalizing large diferences between consecutive TN-core parameters smoothens each of the univariate polynomials modeled by the B-splines. In this way, the smoothness of the predictor ${ \hat { y } } _ { n }$ as a whole can be ensured.

Each TN-core can also be made sparse by using an L1 regularization term (Zhang et al., 2018) of the form

$$
\mathcal { R } ( \mathcal { W } _ { 1 } , \ldots , \mathcal { W } _ { D } ) : = \sum _ { d = 1 } ^ { D } \lambda _ { d } \left\| \mathrm { v e c } ( \mathcal { W } _ { d } ) \right\| _ { 1 } ,
$$

which has been shown to make the TN-cores up to 60% more sparse without negatively impacting predictive performance.

Automatic Relevance Determination (ARD). The number of univariate basis functions I and TN-ranks are hyperparameters that are usually determined through cross-validation. In (Kilic and Batselier, 2025) a fully Bayesian approach was developed for CPD-based TNKMs. All parameters and hyperparameters (I, CP-rank R, variance of disturbance $e _ { n } ,$ prior hyperparameters) are modeled as random variables and learned through Bayesian inference. By assuming that the posterior distribution factors over all parameters and hyperparameters, one can derive a Bayesian ALS algorithm. The key diference with the conventional ALS is that in its Bayesian counterpart one TN-core is updated while replacing all other random variables with their expectation. The Bayesian ALS has the same computational cost $O ( D N I R + I ^ { 3 } R ^ { 3 } )$ as the conventional $\mathrm { A L } \dot { \mathrm { S } }$ using a CPD parameterization but comes with the benefit of providing uncertainty quantification. In this context a tensor-variate prior can be designed on $w _ { d }$ that learns the dimensions of this tensor. Taking a TT parameterization as an example (Kilic and Batselier, 2026), we consider the following zero-mean Gaussian prior

$$
p \left( \mathrm { v e c } ( \pmb { \mathcal { W } } _ { d } ) \right) = \mathcal { N } \left( \mathbf { 0 } , \pmb { \Lambda } _ { d } ^ { - 1 } \otimes \pmb { \Delta } _ { d } ^ { - 1 } \otimes \pmb { \Lambda } _ { d - 1 } ^ { - 1 } \right) .
$$

Each of the precision matrices $\mathbf { \Delta } \mathbf { \Lambda } _ { d - 1 } \in \mathbb { R } ^ { R _ { d - 1 } \times R _ { d - 1 } } , \mathbf { \Lambda } _ { d } \in$ $\mathbb { R } ^ { R _ { d } \times R _ { d } } , \pmb { \Delta } \in \mathbb { R } ^ { \times I }$ is a diagonal matrix of random variables which is updated in the Bayesian ALS. The $r _ { d } \mathrm { t h }$ diagonal entry of $\hat { \mathbf { A } } _ { d } \stackrel { * } { \sim } \mathrm { c o n t r o l s } ^ { \prime \prime }$ the r th matrix slice of the $R _ { d }$ dimension. Should one of these precisions become large, then that particular term of $R _ { d }$ is driven to zero, efectively truncating the rank. The same principle holds for all other TT-ranks and basis functions for each of the ${ \pmb x } _ { n }$ variables. In this way TNKMs become more interpretable, as the most relevant model components are automatically identified.

## 4. DEVELOPMENTS

This section presents a brief overview of some recent TNKM developments.

Rank-K CPD feature vector. The Kronecker product structure of the feature vector $\varphi ( { \pmb x } )$ is a rank-1 CPD. This can be generalized to a rank-K CPD

$$
\sum _ { k = 1 } ^ { K } \lambda _ { k } \varphi _ { D } ^ { k } \otimes \cdot \cdot \cdot \otimes \varphi _ { 2 } ^ { k } \otimes \varphi _ { 1 } ^ { k } ,\tag{12}
$$

where each $\varphi _ { d } ^ { k }$ is a diferent feature vector. In this way diferent choices of basis functions can be combined into a single feature vector and their individual “weights” $\lambda _ { k }$ can also be estimated eficiently from the identification data using ALS (Saiapin and Batselier, 2025). A major advantage of this approach is that cross-validation on K diferent choices for basis functions can be completely avoided. Identifying one TNKM with a rank-K feature vector as in (12) sufices, speeding up the whole identification process. In addition, a regularization term on the $\lambda _ { k }$ parameters efectively reduces the sensitivity of the final model to “badly chosen” basis functions.

Limiting Gaussian Processes. In (Wesel and Batselier, 2025) it is shown how the 1-step ahead predictor ${ \hat { y } } _ { n }$ of both the CPD-based and TT-based TNMKM converge to a Gaussian Process in the limit of large TN-ranks. A crucial assumption for this convergence is that ${ \pmb w } _ { d } ^ { r }$ for the CPDcase and ${ \pmb w } _ { d } ^ { r _ { d - 1 } , r _ { d } }$ for the $\mathrm { T T } .$ -case are independent and identically distributed random variables, which is satisfied for both independent cores and ARD regularization. An important result is that for equal number of basis functions I, number of features D and number of tensor network parameters, the TT-based 1-step predictor will converge at a polynomially faster rate than the CPD-based predictor, thus exhibiting GP behavior with a reduced number of model parameters.

Laplace approximation. An alternative for the fully Bayesian approach, present in the discussion on the Automatic Relevance Determination regularization, is to approximate the posterior of the TN-cores by a Gaussian distribution (Saiapin and Batselier, 2026a). This also allows TNKMs to provide principled uncertainty estimates. Combining this approximation with a linearized predictive distribution has been shown to consistently outperform Gaussian Processes and Bayesian Neural Networks.

Recursive identification with Kalman filter. Writing a conventional SVM model in state space form

$$
{ \pmb w } _ { n + 1 } = { \pmb w } _ { n } , y _ { n } = \varphi ( { \pmb x } _ { n } ) ^ { \intercal } { \pmb w } _ { n } + e _ { n } ,
$$

together with assuming Markovianity and a Gaussian prior $p ( \pmb { w } _ { 0 } )$ allows one to compute the filtering density $p ( \pmb { w } _ { n } | y _ { 1 : n } )$ exactly with a Kalman filter. Replacing the SVM model with a TNKM then implies that the Kalman filter update equations involve vectors and matrices of dimensions $I ^ { D }$ . Fortunately, a complete Kalman filter update step can be done eficiently by computing all quantities directly as tensor networks (Batselier et al., 2017a). This is yet another way to arrive at a probabilistic TNKM. More recently a square-root version of such a tensor network Kalman filter has been developed with improved numerical stability (Menzen et al., 2026).

## 5. FUTURE WORK AND CONCLUSION

TNKMs have been shown to be a flexible model class and eficient in terms of storage and computation. Upcoming work focuses on the derivation and implementation of an eficient TNKM-based prior for smooth and decaying highorder Volterra kernel identification and a proper initialization to avoid unstable gradients during identification.

Some pointers for future work. The identification of different colored noise models in the TNKM context remains largely unexplored. The parameterization of a completely separate noise dynamics such as in nonlinear Box-Jenkins models has also not been tackled yet. Another largely unexplored terrain is the use of TNKMs for nonlinear state space models where the state is partially or even completely hidden.

## ACKNOWLEDGEMENTS

This publication is part of the project Sustainable learning for Artificial Intelligence from noisy large-scale data (with project number VI.Vidi.213.017) which is financed by the Dutch Research Council (NWO).

## REFERENCES

Batselier, K. (2022). Low-rank tensor decompositions for nonlinear system identification: A tutorial with examples. IEEE Control Systems Magazine, 42(1), 54–74.

Batselier, K., Chen, Z., and Wong, N. (2017a). A Tensor Network Kalman filter with an application in recursive MIMO Volterra system identification. Automatica, 84, 17–25.

Batselier, K., Chen, Z., and Wong, N. (2017b). Tensor Network alternating linear scheme for MIMO Volterra system identification. Automatica, 84, 26–35.

Bottou, L., Curtis, F.E., and Nocedal, J. (2018). Optimization methods for large-scale machine learning. SIAM review, 60(2), 223–311.

Chen, Z., Batselier, K., Suykens, J.A., and Wong, N. (2017). Parallelized tensor train learning of polynomial classifiers. IEEE transactions on neural networks and learning systems, 29(10), 4621–4632.

De Branges, L. (1959). The Stone-Weierstrass Theorem. Proceedings of the American Mathematical Society, 10(5), 822–824. doi:10.2307/2033481.

Drezet, P.M. and Harrison, R.F. (1998). Support vector machines for system identification. In UKACC International Conference on Control (CONTROL’98), 688–692. IET.

Espinoza, M., Suykens, J.A., and De Moor, B. (2005). Kernel based partially linear models and nonlinear identification. IEEE Transactions on Automatic Control, 50(10), 1602–1606.

Falck, T., Pelckmans, K., Suykens, J.A., and De Moor, B. (2009). Identification of wiener-hammerstein systems using ls-svms. IFAC Proceedings Volumes, 42(10), 820– 825.

Goethals, I., Pelckmans, K., Suykens, J.A., and De Moor, B. (2005). Subspace identification of Hammerstein systems using least squares support vector machines. IEEE Transactions on Automatic Control, 50(10), 1509– 1519.

Hitchcock, F.L. (1927). The expression of a tensor or a polyadic as a sum of products. Journal of Mathematics and Physics, 6(1-4), 164–189.

Holtz, S., Rohwedder, T., and Schneider, R. (2012). The alternating linear scheme for tensor optimization in the tensor train format. SIAM Journal on Scientific Computing, 34(2), A683–A713.

Karagoz, R. and Batselier, K. (2020). Nonlinear system identification with regularized Tensor Network Bsplines. Automatica, 122, 109300.

Kilic, A. and Batselier, K. (2025). Interpretable Bayesian Tensor Network Kernel Machines with automatic rank and feature selection. arXiv preprint arXiv:2507.11136.

Kilic, A. and Batselier, K. (2026). Bayesian Tensor Train Kernel Machines with Automatic Model Complexity Inference. arXiv preprint arXiv:2611.07129.

Kingma, D.P. and Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.

Lichtenberg, G. (2011). Hybrid Tensor Systems. Habilitation thesis, Hamburg University of Technology.

Memmel, E., Menzen, C., and Batselier, K. (2023). Bayesian Framework for a MIMO Volterra Tensor Network. IFAC-PapersOnLine, 56(2), 7294–7299.

Menzen, C., Kok, M., and Batselier, K. (2026). Tensor network square root Kalman filter for online Gaussian process regression. Automatica, 183, 112694.

Oseledets, I.V. (2011). Tensor-train decomposition. SIAM Journal on Scientific Computing, 33(5), 2295–2317.

Saiapin, A. and Batselier, K. (2025). Tensor network based feature learning model. In Y. Li, S. Mandt, S. Agrawal, and E. Khan (eds.), Proceedings of The 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings of Machine Learning Research, 3277–3285. PMLR.

Saiapin, A. and Batselier, K. (2026a). Gradient Boosting for Eficient Rank Discovery in Tensor Network Kernel Machines. arXiv preprint arXiv:2611.08093.

Saiapin, A. and Batselier, K. (2026b). Laplace Approximation for Bayesian Tensor Network Kernel Machines. arXiv preprint arXiv:2604.26673.

S¨arkk¨a, S. (2021). Use of Gaussian processes in system identification. In Encyclopedia of systems and control, 2393–2402. Springer.

Suykens, J.A., Van Gestel, T., De Brabanter, J., De Moor, B., and Vandewalle, J. (2002). Least squares support vector machines, volume 4. World scientific, Singapore.

T´oth, R., Laurain, V., Zheng, W.X., and Poolla, K. (2011). Model structure learning: A support vector machine approach for LPV linear-regression models. In 2011 50th IEEE Conference on Decision and Control and European Control Conference, 3192–3197. IEEE.

Uschmajew, A. (2012). Local convergence of the alternating least squares algorithm for canonical tensor approximation. SIAM Journal on Matrix Analysis and Applications, 33(2), 639–652.

Wesel, F. and Batselier, K. (2021). Large-scale learning with Fourier features and tensor decompositions. Advances in neural information processing systems, 34, 17543–17554.

Wesel, F. and Batselier, K. (2024). Quantized Fourier and polynomial features for more expressive tensor network models. In International Conference on Artificial Intelligence and Statistics, 1261–1269. PMLR.

Wesel, F. and Batselier, K. (2025). Tensor Network-Constrained Kernel Machines as Gaussian Processes. In Proceedings of The 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings of Machine Learning Research, 2161–2169. PMLR.

Williams, C.K. and Rasmussen, C.E. (2006). Gaussian processes for machine learning, volume 2. MIT press, Cambridge, MA.

Zhang, Y., Ko, C.Y., Chen, C., Batselier, K., and Wong, N. (2018). Sparse tensor network system identification for nonlinear circuit macromodeling. In 2018 14th IEEE

International Conference on Solid-State and Integrated Circuit Technology (ICSICT), 1–4. IEEE.