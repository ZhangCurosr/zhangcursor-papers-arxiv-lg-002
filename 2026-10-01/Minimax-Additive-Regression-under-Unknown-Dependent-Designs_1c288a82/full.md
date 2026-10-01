# Minimax Additive Regression under Unknown Dependent Designs

Baptiste Ferrere ∗ Université de Toulouse EDF R&D

Fabrice Gamboa <sup>†</sup> Université de Toulouse ANITI

Jean-Michel Loubes <sup>†</sup> Université de Toulouse ANITI

## Abstract

We study additive regression under a potentially non-product random design on $[ 0 , \bar { 1 } ] ^ { d }$ , allowing the dimension d to grow with the sample size n. We introduce coupled smoothness classes that separately control the regularity of the marginal densities and the densityweighted additive components. To handle dependence, we adapt a Riesz-basis construction for functional ANOVA models and establish compatibility bounds with constants independent of the dimension under uniform bounds on the joint density. We construct thresholded least-squares estimators and establish matching minimax upper and lower bounds for prediction with known or unknown marginal densities, under suitable dimensiongrowth conditions. When the marginal densities are at least as smooth as the weighted components, the unknown-density problem attains the known-density minimax rate. When the densities are less smooth, their regularity determines the minimax rate over the coupled class. Finally, we show that the centered additive components can be recovered at the same aggregate upper rate, without an additional order of error.

## 1 INTRODUCTION

We consider the random-design nonparametric regression model

$$
Y _ { i } : = f ^ { \star } ( \mathbf { X } _ { i } ) + \xi _ { i } , \qquad i = 1 , \ldots , n ,\tag{1}
$$

where $\mathbf { X } _ { 1 } , \ldots , \mathbf { X } _ { n }$ are independent copies of a random vector $\mathbf { X } : = ( X _ { 1 } , \ldots , X _ { d } )$ supported on $[ 0 , 1 ] ^ { d }$ and distributed according to the probability measure P. We assume that $P$ admits a continuous density p with respect to the Lebesgue measure. The noise variables $\xi _ { 1 } , \ldots , \xi _ { n }$ are independent of the design and independently distributed according to ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ . Given the sample $\left( \mathbf { X } _ { i } , Y _ { i } \right) _ { 1 \leq i \leq n } ,$ our objective is to estimate $f ^ { \star }$ under the squared $\bar { L } ^ { 2 } ( P )$ loss. This is a classical setting in statistical learning theory and nonparametric regression (Tsybakov, 2009). In this work, we focus on additive models of the form

$$
f ^ { \star } ( { \bf x } ) = f _ { 0 } ^ { \star } + \sum _ { j = 1 } ^ { d } f _ { j } ^ { \star } ( x _ { j } ) , \quad { \bf x } \in [ 0 , 1 ] ^ { d } ,\tag{2}
$$

under the standard identifiability condition

$$
\mathbb { E } _ { P } [ f _ { j } ^ { \star } ( X _ { j } ) ] = 0 , \qquad j = 1 , \dots , d .\tag{3}
$$

Under this normalization, $f _ { 0 } ^ { \star } = \mathbb { E } _ { P } [ f ^ { \star } ( \mathbf { X } ) ]$ . Obviously, the intercept can be estimated at the parametric rate. We focus throughout on the centered case $f _ { 0 } ^ { \star } = 0$ Distribution-dependent centering has long been used in random-design additive regression and functional ANOVA models. Now, under a product design, it makes the univariate component spaces mutually orthogonal and is commonly imposed in the analysis of sparse additive models (Raskutti et al., 2012). More generally, it corresponds to the first-order hierarchical orthogonality constraint in functional ANOVA decompositions under dependent designs (Stone, 1994; Huang, 1998; Hooker, 2007). Under a general distribution $P ,$ the components remain orthogonal to the constants but are no longer mutually orthogonal. Consequently, the stability of the additive decomposition depends on the non-orthogonal geometry induced by the distribution.

Starting from this general distribution-dependent framework, we study the spectral and statistical structure of the resulting non-orthogonal geometry. To this end, we build on the recent Riesz representation of Ferrere et al. (2026). This representation provides stable coordinates for hierarchically orthogonal component spaces under non-product designs. To the best of our knowledge, the statistical estimation of functions in this representation has not been studied when the design distribution is unknown. In particular, existing analyses do not separate the smoothness of the regression function relative to a fixed design from the regularity required to learn the design-adapted representation. We summarize our key contributions below.

Contributions. Our main contribution is a rigorous theoretical analysis of additive regression under dependent random designs, allowing the dimension d to increase with the sample size n. We separate our analysis into two cases: known and unknown marginal densities. In both settings, we establish minimax optimal rates under suitable dimension-growth conditions.

• We derive explicit Riesz bounds for the additive representation of Ferrere et al. (2026), with stability constants independent of d under uniform joint-density bounds. These bounds control the geometry induced by dependence among the covariates.

• When the marginal densities are known, we show that thresholded least squares attains the classical additive minimax rate, with linear dependence on d. We establish a matching lower bound for each admissible fixed design distribution.

• When the marginal densities are unknown, we introduce an estimator based on sample splitting and thresholded least squares in an estimated dictionary. We control its prediction error and recover the centered additive components at the same aggregate upper rate.

• We develop a lower-bound construction for the rough-density regime that varies the design distribution while keeping the density-weighted components fixed. Together with the upper bounds, this establishes when unknown marginals preserve the known-density minimax rate and when their regularity determines the minimax rate.

Organization. The rest of the paper is organized as follows. Section 2 introduces all the mathematical objects necessary for the construction of our coupled smoothness class and the corresponding estimators. Section 3 studies the known-density setting, establishing the minimax optimal rate. Section 4 addresses unknown marginal densities. Section 4.1 presents the estimator, the prediction upper bound, and the componentrecovery result. Section 4.2 establishes matching lower bounds in the smooth and rough-density regimes. Section 5 discusses the main results, limitations, and future directions. All proofs are deferred to Appendix.

Related Work. Additive regression models were studied early on as a way to retain the flexibility of nonparametric regression while avoiding its full $d -$ dimensional complexity (Stone, 1985), and were subsequently formalized within the generalized additive model framework by Hastie and Tibshirani (1986). By reducing the estimation of a multivariate function to that of univariate components, additivity mitigates the curse of dimensionality: when every component has smoothness $\beta ,$ the classical rate $d n ^ { - 2 \bar { \beta } / ( 2 \beta + \bar { 1 } ) }$ can be attained, rather than the d-dimensional rate $n ^ { - 2 \beta / ( 2 \beta + d ) }$

Most closely related to our setting are backfitting and smooth-backfitting methods for additive regression under random design (Mammen et al., 1999; Horowitz et al., 2006). Under suitable regularity conditions on the joint design distribution, these methods recover individual additive components at their univariate oracle rates without requiring independent covariates. In this literature, assumptions on the marginal and pairwise design densities ensure the stability of the empirical projection operators, while component smoothness is defined—apart from the distribution-dependent centering constraints—in fixed Sobolev, spline, or kernel spaces. Our framework takes a diferent starting point: for each design distribution p, the associated Riesz representation defines a design-indexed function class $\mathcal { F } _ { \beta } ( \boldsymbol { p } )$ , in which $\beta$ controls the spectral decay of the Riesz coeficients, equivalently the Sobolev-type regularity of $p _ { j } f _ { j }$ . We then allow p to vary over a class ${ \mathcal P _ { \gamma } } ,$ where γ controls the regularity of the marginal densities and hence the accuracy with which the corresponding dictionary can be estimated. Thus, the design determines not only the centering constraints and projection geometry, but also the functional coordinates in which signal smoothness is measured.

In high-dimensional settings, additivity is often combined with the assumption that only a small number of components are active. This leads to sparse additive models literature and to estimators combining componentwise smoothness with variable selection (Meier et al., 2009; Ravikumar et al., 2009). Matching minimax bounds have been obtained over Sobolev and kernel classes (Raskutti et al., 2012; Yuan and Zhou, 2016; Tan and Zhang, 2019; Haris et al., 2022), as well as for more general low-order interaction models (Bhattacharya et al., 2024).

Additive models have also received renewed attention in machine learning because their componentwise structure provides a direct and visual form of interpretability (Rudin, 2019). This principle underlies Neural Additive Models (Agarwal et al., 2021), Neural Basis Models (Radenovic et al., 2022), NODE-GAM (Chang et al., 2021), and interpretable tree ensembles based on main efects and low-order interactions (Nori et al., 2019;

Lengerich et al., 2020; Bénard, 2025). These developments further motivate a precise understanding of the functional spaces to which additive components belong (defined later in equation (5)), particularly when the covariates are nonuniform and statistically dependent.

## 2 PRELIMINARIES

Notation. Let λ denote the Lebesgue measure on [0, 1]. Recall that for any $j \in \{ 1 , \ldots , d \}$ , we denote by $P _ { j }$ the marginal distribution of $X _ { j }$ and by $p _ { j }$ the corresponding density, such that $\begin{array} { r } { \frac { d P _ { j } } { d \lambda } = p _ { j } } \end{array}$ . We denote by $L ^ { 2 } ( P )$ the Hilbert space of square-integrable and $P -$ −measurable functions. For simplicity, we denote by $\| \cdot \|$ the $L ^ { 2 } ( P )$ −norm and $\| \cdot \| _ { 2 }$ the Euclidean norm. Throughout the paper, for two nonnegative quantities $\boldsymbol { a } _ { n , d }$ and $b _ { n , d } ,$ we write $a _ { n , d } \lesssim b _ { n , d }$ if there exists a constant $C > 0$ , independent of n and $d ,$ such that $a _ { n , d } ~ \leq ~ C b _ { n , d }$ We define $\gtrsim$ analogously and write $a _ { n , d } \asymp b _ { n , d }$ when both inequalities hold.

## 2.1 Identifiability condition

Without further condition, the additive representation in (2) is not unique, since constants can be transferred between the intercept and the component functions. Two structural assumptions are commonly made. A first possibility is the Lebesgue centering $\begin{array} { r l } { \int _ { 0 } ^ { 1 } f _ { j } ( x ) d x = } \end{array}$ 0, for all $j \in \{ 1 , \ldots , d \}$ . It is natural in Gaussian whitenoise formulations and when smoothness is defined in a fixed $L ^ { 2 } ( \lambda )$ geometry (Tsybakov, 2009; Bhattacharya et al., 2024; Liu, 2025). In this setting, it is natural to assume some Sobolev-type regularity where each component $f _ { j }$ can be expanded in an orthonormal basis of $L ^ { 2 } ( \lambda )$ or a kernel. Random-design additive regression instead commonly assumes the following identifiability constraint:

Assumption 2.1 (Population centering).

$$
\mathbb { E } _ { P } [ f _ { j } ( X _ { j } ) ] = 0 , \quad j = 1 , \dots , d ,\tag{4}
$$

see for example Mammen et al. (1999); Ravikumar et al. (2009); Raskutti et al. (2012); Horowitz et al. (2006). This condition is more natural since it is formulated in the Hilbert space equipped with the distribution of the true data. Of course it recovers the Lebesgue centering when each feature is uniform on [0, 1]. Under Assumption 2.1, the functions $f _ { j }$ act as the main efects of a generalized functional ANOVA decomposition (Stone, 1994; Huang, 1998; Hooker, 2007). More fundamentally, let $\mathcal { H } _ { \mathrm { 0 } }$ be the Hilbert space of constant functions of $L ^ { 2 } ( P )$ and, for every $j \in [ d ]$ , define the first-order generalized ANOVA space

$$
\begin{array} { r } { \mathcal { H } _ { j } : = \left\{ h \in L ^ { 2 } ( P _ { j } ) \vert \mathbb { E } _ { P } [ h ( X _ { j } ) ] = 0 \right\} . } \end{array}\tag{5}
$$

Each $\mathcal { H } _ { j }$ is a closed Hilbert subspace of $L ^ { 2 } ( P )$ . Under some conditions (Stone, 1994; Huang, 1998; Hooker, $2 0 0 7 ;$ Chastaing et al., 2012) ensuring existence and uniqueness of the generalized functional ANOVA decomposition—satisfied in particular under Assumption 2.3 defined later—the first-order component spaces form a direct sum:

$$
\mathcal { H } : = \mathcal { H } _ { 1 } \oplus \cdot \cdot \cdot \oplus \mathcal { H } _ { d }\tag{6}
$$

Here, the symbol ⊕ emphasizes that the decomposition is unique but not necessarily orthogonal. Consequently, studying centered additive functions satisfying Assumption 2.1 is equivalent to studying elements of H.

Remark 2.2. If $P$ is a product distribution, the component spaces $\mathcal { H } _ { 1 } , \ldots , \mathcal { H } _ { d }$ are mutually orthogonal, and (6) becomes an orthogonal direct sum. Under a dependent design, each $\mathcal { H } _ { j }$ remains orthogonal to $\mathcal { H } _ { \mathrm { 0 } }$ by population centering, but distinct first-order spaces are generally not mutually orthogonal.

Except in the uniform setting, where standard orthonormal bases provide an immediate representation, the joint geometry of these component spaces lacked an explicit stable coordinate system. To the best of our knowledge, Ferrere et al. (2026) are the first to provide such a characterization through a distribution-adapted Riesz basis based on what they call inverse marginaldensity weighting. We build on its first-order blocks to define the regularity classes used throughout this paper.

## 2.2 Representation of additive components

In order to adapt the construction of Ferrere et al. (2026), we introduce the following boundness condition on the joint density $p .$

Assumption 2.3. Let p be a density probability on $[ 0 , 1 ] ^ { d }$ . We denote by $p _ { \mathrm { m i n } } : = \operatorname* { i n f } _ { \mathbf { x } } p ( \mathbf { x } )$ and $p _ { \mathrm { m a x } } : =$ $\operatorname* { s u p } _ { \mathbf { x } } p ( \mathbf { x } )$ . We say that $p$ satisfies this assumption if

$$
0 < p _ { \operatorname* { m i n } } \le p \le p _ { \operatorname* { m a x } } < + \infty .\tag{7}
$$

This is a standard assumption in the literature of nonparametric regression, which will allow to achieve the standard minimax optimal rates without introducing additional dependencies in the dimension d.

We then denote by $( \phi _ { m } ) _ { m \in \mathbb { N } }$ the trigonometric basis on $[ 0 , 1 ]$ with $\phi _ { 0 } = 1$ (see for example Tsybakov (2009)). We can now define the object that will be central to our class of regularity.

Definition 2.4. Let $j \in \{ 1 , \ldots , d \}$ and m be a positive integer, we define the function $\psi _ { m } ^ { ( j ) }$ as

$$
\forall x \in [ 0 , 1 ] , \quad \psi _ { m } ^ { ( j ) } ( x ) : = \frac { \phi _ { m } ( x ) } { p _ { j } ( x ) }\tag{8}
$$

We introduce the notation $\Psi : = \left( ( \psi _ { m } ^ { ( j ) } ) _ { m \geq 1 } \right) _ { j = 1 , \ldots , d } .$

When working on [0, 1], it is more comfortable to use the trigonometric basis. We have the first technical result induced by Ψ.

Theorem 2.5. Fix a density p that satifies Assumption 2.3. For any finite sequence of real numbers $\textbf { \textit { a } } : =$ $\left( ( a _ { m } ^ { ( j ) } ) _ { m \geq 1 } \right) _ { j = 1 , \ldots , d } ,$ we define the function $f _ { a }$ as

$$
f _ { a } : = \sum _ { j = 1 } ^ { d } \sum _ { m \geq 1 } a _ { m } ^ { ( j ) } \psi _ { m } ^ { ( j ) } .\tag{9}
$$

The following inequality holds:

$$
C _ { \operatorname* { m i n } } ^ { p } \| \pmb { a } \| _ { 2 } ^ { 2 } \leq \| f _ { \pmb { a } } \| ^ { 2 } \leq C _ { \operatorname* { m a x } } ^ { p } \| \pmb { a } \| _ { 2 } ^ { 2 } ,\tag{10}
$$

where $C _ { \mathrm { m i n } } ^ { p } : = p _ { \mathrm { m i n } } / p _ { \mathrm { m a x } } ^ { 2 }$ and $C _ { \mathrm { m a x } } ^ { p } : = p _ { \mathrm { m a x } } / p _ { \mathrm { m i n } } ^ { 2 }$ . We say that Ψ is a Riesz sequence (Brézis, 2011).

This theorem will be useful to establish the upper bound inequalities. Indeed, it is a compatibility condition between the $L ^ { 2 } ( P )$ norm and the Euclidean norm. Furthermore, it is a key property to obtain the next result, which charaterizes the Hilbert spaces $\mathcal { H } _ { j }$

Theorem 2.6 (Representation theorem). For any $f \in$ H, there exists a unique sequence of real coeficients denoted $\pmb { \theta } : = ( ( \theta _ { m } ^ { ( j ) } ) _ { m \geq 1 } ) _ { j = 1 , \dots , d }$ such that each additive component is uniquely represented as

$$
\forall j \in \{ 1 , \dots , d \} , \quad f _ { j } = \sum _ { m = 1 } ^ { + \infty } \theta _ { m } ^ { ( j ) } \psi _ { m } ^ { ( j ) } .\tag{11}
$$

Note that the result of the Theorem 2.5 is not stated or proved in the paper of Ferrere et al. (2026), however the result of Theorem 2.6 is just an adaptation by replacing Legendre polynomials on [−1, 1] by the trigonometric basis on [0, 1].

## 2.3 Regularity classes

We distinguish signal regularity at a fixed and known design distribution from the regularity of the marginal densities used to construct the representation.

Definition 2.7 (Sobolev ball). For $\beta > 0$ and $R > 0$ define

$$
\Theta ( \beta , R ) : = \left\{ ( \theta _ { m } ) _ { m \geq 1 } \big | \sum _ { m \geq 1 } m ^ { 2 \beta } \theta _ { m } ^ { 2 } \leq R ^ { 2 } \right\} .\tag{12}
$$

Let $f \in \mathcal H$ . By Theorem 2.6, each component $f _ { j }$ has unique Riesz coeficients θ satisfying the decomposition

$$
g _ { j } : = p _ { j } f _ { j } = \sum _ { m = 1 } ^ { \infty } \theta _ { m } ^ { ( j ) } \phi _ { m } .\tag{13}
$$

Observe that $\theta _ { m } ^ { ( j ) } = \langle g _ { j } , \phi _ { m } \rangle _ { L ^ { 2 } ( \lambda ) }$ is the Fourier coeficient of frequency m of $g _ { j }$ in the trigonometric basis. Consequently, requiring $( \theta _ { m } ^ { ( j ) } ) _ { m \geq 1 } \in \Theta ( \beta , R )$ is exactly a periodic Sobolev ellipsoid constraint on the function $g _ { j }$ . The corresponding weighted coeficient norm is equivalent to the usual $H _ { \mathrm { p e r } } ^ { \beta } ( [ 0 , 1 ] )$ norm on its meanzero subspace (Tsybakov, 2009).

Remark 2.8. Weighted ellipsoid constraints on expansion coeficients are a standard way to encode regularity in nonparametric estimation. The same principle underlies kernel classes, where kernel eigenvalues determine the weights of the coeficient ellipsoid.

Here, we impose such a constraint in our Riesz coordinates. Since $\theta _ { m } ^ { ( j ) }$ are exactly the Fourier coeficients of $g _ { j } ,$ , this places $g _ { j }$ in a periodic Sobolev ellipsoid. For a fixed marginal density $p _ { j } .$ the resulting class of components $f _ { j }$ is therefore the image of this ellipsoid under the map $g \mapsto g / p _ { j }$ . It can be viewed as a Sobolev ellipsoid transformed by the design, whose elements need not retain the same ordinary Sobolev smoothness when $p _ { j }$ is irregular. Thus, regularity of $g _ { j }$ follows directly from our coeficient assumption.

Definition 2.9 (Design-indexed signal class). For a fixed admissible density $p ,$ we denote by $\mathcal { F } _ { \beta , R } ( \boldsymbol { p } )$ the strict subset of H that contains every function of $f$ of the form

$$
f = \sum _ { j = 1 } ^ { d } \sum _ { m = 1 } ^ { + \infty } \theta _ { m } ^ { ( j ) } \psi _ { m } ^ { ( j ) } ,\tag{14}
$$

where for all $j = 1 , \ldots , d ,$ the sequence $( \theta _ { m } ^ { ( j ) } ) _ { m }$ belongs to the ball $\Theta ( \beta , R )$ .

This class consists of population-centered additive functions whose weighted components $p _ { j } f _ { j }$ live in a standard Sobolev ellipsoid. The radius R and the smoothness $\beta$ are the same for each additive component. The dependence on the design density is part of the definition of the function class. To quantify the error incurred when estimating the marginal densities, we impose a separate Hölder regularity condition.

Definition 2.10 (Hölder regularity). Let $\gamma > 0 , L > 0 .$ and $s : = \lfloor \gamma \rfloor$ . A function $q : [ 0 , 1 ] \to \mathbb { R }$ is said to be $( \gamma , L )$ -Hölder if $q \in C ^ { s } ( [ 0 , 1 ] )$ and for all $x , y \in [ 0 , 1 ]$

$$
| q ^ { ( s ) } ( x ) - q ^ { ( s ) } ( y ) | \leq L | x - y | ^ { \gamma - s } .\tag{15}
$$

To define the class of all distributions of interest $p ,$ we need to parametrize it entirely with global and uniform constants.

Definition 2.11 (Distribution class). Let $\gamma > 0$ be a smoothness parameter and $L > 0$ a positive constant. Let also $\kappa _ { \mathrm { m i n } }$ and $\kappa _ { \mathrm { m a x } }$ be positive constants and free of dimension d such that $\kappa _ { \mathrm { m i n } } < 1 < \kappa _ { \mathrm { m a x } }$ . We define the regularity class $\mathcal { P } _ { \gamma , L }$ (where we omit the dependence in $\kappa _ { \mathrm { m i n } } , \kappa _ { \mathrm { m a x } }$ for simplicity) as the set of all densities $p$ defined on $[ 0 , 1 ] ^ { d }$ which satisfy:

1. $p _ { j }$ is (γ, L)−Hölder for all $j \in \{ 1 , \ldots , d \}$

$$
2 . \ K _ { \operatorname* { m i n } } \le p _ { \operatorname* { m i n } } \le p \le p _ { \operatorname* { m a x } } \le \kappa _ { \operatorname* { m a x } }
$$

Note that the second condition is a generalization of Assumption 2.3 where all the considered densities have the same bounds. Moreover, we can define the Riesz constants by $\begin{array} { r } { C _ { \mathrm { m i n } } : = \frac { \kappa _ { \mathrm { m i n } } } { \kappa _ { \mathrm { m a x } } ^ { 2 } } } \end{array}$ and $\begin{array} { r } { C _ { \mathrm { m a x } } : = \frac { \kappa _ { \mathrm { m a x } } } { \kappa _ { \mathrm { m i n } } ^ { 2 } } } \end{array}$ which are independent of d and n and uniform over all the considered distributions $p .$

Remark 2.12. Note that the class $\mathcal { P } _ { \gamma , L }$ for any $( \gamma , L )$ and any $\kappa _ { \operatorname* { m i n } { } } ~ < ~ 1 ~ < ~ \kappa _ { \operatorname* { m a x } { } }$ is not empty since the uniform density $p _ { \mathrm { u n i f } } \equiv 1$ always belongs to this class.

We finally introduce the following Assumption, which allows the dimension d to grow but at a rate slightly slower than the nonparametric rate of $n ^ { \frac { 2 \beta } { 2 \beta + 1 } }$

Assumption 2.13 (Growing dimension.). We suppose that d can grow with n at the following rate

$$
d = o \left( { \frac { n ^ { 2 \beta / ( 2 \beta + 1 ) } } { \log n } } \right) .\tag{16}
$$

Definition 2.14 (Coupled parameter class). For $\beta , \gamma , R , L , \kappa _ { \operatorname* { m i n } } , \kappa _ { \operatorname* { m a x } } > 0$ , define the coupled class

$$
\mathcal { C } _ { \beta , \gamma } : = \bigcup _ { p \in \mathcal { P } _ { \gamma , L } } \{ p \} \times \mathcal { F } _ { \beta , R } ( p ) .\tag{17}
$$

For the sake of simplicity, we explicit only the dependance in $\gamma$ and $\beta$

This is the global class considered in our statistical analysis. The parameter $\beta$ controls the spectral regularity of the signal in the Riesz representation associated with each fixed $p ,$ whereas $\gamma$ controls the regularity of the marginal densities defining that representation. Each admissible design therefore indexes its own signal class, and estimation guarantees are established uniformly over the resulting pairs $( p , f )$

## 3 KNOWN DENSITY REGIME

Let $p$ be a density satisfying Assumption 2.3, and suppose that its marginal densities are known. Write $\mathcal { F } = \mathcal { F } _ { \beta , R } ( \boldsymbol { p } )$ and define

$$
\mathfrak { M } ( n , d , \mathcal { F } ) = \operatorname* { i n f } _ { \widehat { f } } \operatorname* { s u p } _ { f \in \mathcal { F } } \mathbb { E } _ { p , f } \left[ \Vert f - \widehat { f } \Vert ^ { 2 } \right] ,\tag{18}
$$

where estimators may depend on $p ,$ the norm is taken in $L ^ { 2 } ( P )$ , and the expectation averages over both the random design and the regression noise. The underlying constants will make appear the Riesz constant of the Theorem 2.5, which explicitely depend of the distribution $p$ but not of n or d.

## 3.1 Minimax upper bound

In order to obtain an upper bound of the minimax risk, we construct an estimator that achives the desired rate of $d n ^ { - \frac { 2 \beta } { 2 \beta + 1 } }$ . For an integer $M \geq 1$ , let $\psi _ { M } ( \mathbf { x } ) \in$ $\mathbb { R } ^ { d M }$ collect the functions $\psi _ { m } ^ { ( j ) } ( x _ { j } )$ , ordered by the d coordinates and M frequencies. Let Ψ $\dot { M }$ be the design matrix with rows $\psi _ { M } ( \mathbf { X } _ { i } ) ^ { \top }$ , and define

$$
\mathbf { \Gamma } \mathbf { \Gamma } _ { M } = \mathbb { E } _ { p } \big [ \psi _ { M } ( \mathbf { X } ) \psi _ { M } ( \mathbf { X } ) ^ { \top } \big ] , \qquad \widehat { \mathbf { r } } _ { M } = \frac { 1 } { n } \Psi _ { M } ^ { \top } \Psi _ { M } .
$$

The Riesz property implies

$$
C _ { \operatorname* { m i n } } ^ { p } I _ { d M } \preceq \mathbf { { r } } _ { M } \preceq C _ { \operatorname* { m a x } } ^ { p } { { I _ { d M } } } .
$$

To ensure that least squares is well defined, introduce

$$
\zeta _ { n } = \left\{ \lambda _ { \operatorname* { m i n } } ( \widehat { \mathbf { { T } } } _ { M } ) \geq C _ { \operatorname* { m i n } } ^ { p } / 2 \right\} .
$$

Finally, we write $\pmb { Y } = ( Y _ { 1 } , \ldots , Y _ { n } ) ^ { \top }$

Definition 3.1. With the previous notations, we define our estimator ${ \widehat { f } } _ { M }$ for all $\mathbf { x } \in [ 0 , 1 ] ^ { d }$ as follows

$$
\widehat { f } _ { M } ( \mathbf { x } ) : = \psi _ { M } ( \mathbf { x } ) ^ { \top } \widehat { \pmb { \theta } } _ { M } ,\tag{19}
$$

where $\widehat { \pmb { \theta } } _ { M } : = ( \pmb { \Psi } _ { M } ^ { \top } \pmb { \Psi } _ { M } ) ^ { - 1 } \pmb { \Psi } _ { M } ^ { \top } \pmb { Y } \pmb { 1 } _ { \zeta _ { n } } \in \mathbb { R } ^ { d M } .$

Note that in this definition, the inverse is evaluated only on $\zeta _ { n }$ . The estimator uses the known marginal densities and the lower Riesz bound, without regularization. We show in Appendix E, that this construction leads to the desired upper bound formalized in the next theorem.

Theorem 3.2. under Assumption 2.13, we have the following minimax optimal upper bound

$$
\mathfrak { M } ( n , d , \mathcal { F } ) \lesssim d n ^ { - \frac { 2 \beta } { 2 \beta + 1 } } ,\tag{20}
$$

where all the underlying constants behind the symbol $\lesssim$ depend only on $\beta , R , \sigma , p _ { \mathrm { m i n } } , p _ { \mathrm { m a x } } .$

## 3.2 Minimax lower bound

For $\omega \in \{ - 1 , 1 \} ^ { d M }$ , we consider the standard following construction

$$
f _ { \omega } = a _ { M } \sum _ { j = 1 } ^ { d } \sum _ { m = M + 1 } ^ { 2 M } \omega _ { j m } \psi _ { m } ^ { ( j ) } , \qquad a _ { M } = a _ { 0 } M ^ { - \beta - 1 / 2 } .
$$

For suficiently small $a _ { 0 } > 0$ , all these functions belong to $\mathcal { F }$ and the lower Riesz inequality bounds their squared $L ^ { 2 } ( P )$ separation from below. By utilizing Assouad’s lemma (Tsybakov, 2009), we obtain the following theorem, which is proved in Appendix F.

Theorem 3.3. There exists an underlying constant which depends only on $\beta , R , \sigma ,$ p<sub>min</sub>, p<sub>max</sub> such that

$$
\mathfrak { M } ( n , d , \mathcal { F } ) \gtrsim d n ^ { - \frac { 2 \beta } { 2 \beta + 1 } } .\tag{21}
$$

## 4 UNKNOWN DENSITY REGIME

We now consider estimation over the coupled class $\mathcal { C } _ { \beta , \gamma }$ when the design distribution is unknown. Throughout this class, the joint densities satisfy $\kappa _ { \mathrm { m i n } } \leq p \leq \kappa _ { \mathrm { m a x } } .$ with the same known constants. We define the minimax risk by

$$
\mathfrak { M } ( n , d , \mathcal { C } _ { \beta , \gamma } ) : = \operatorname* { i n f } _ { \widehat { f } } \operatorname* { s u p } _ { ( p , f ) \in \mathcal { C } _ { \beta , \gamma } } \mathbb { E } _ { p , f } \left[ \Vert f - \widehat { f } \Vert ^ { 2 } \right] ,\tag{22}
$$

where the infimum ranges over all estimators measurable with respect to the observed sample. These estimators may depend on the fixed parameters defining the class, but not on the unknown pair $( p , f )$ . For each such pair, $\mathbb { E } _ { p , f }$ averages over the independent covariates $\mathbf { X } _ { 1 } , \ldots , \mathbf { X } _ { n }$ with common distribution $P$ of density $p$ and the independent Gaussian regression noise. The prediction loss is evaluated under that same distribution $P .$

## 4.1 Minimax upper bound

Unlike the fixed-design-distribution setting, the supremum in (22) allows both the design density p and the regression function $f \in \mathcal { F } _ { \beta , R } ( p )$ to vary. Thus, the sampling distribution, the prediction norm, and the admissible signal class all depend on the pair under consideration. The smoothness parameter $\beta$ controls the weighted components $p _ { j } f _ { j }$ , whereas $\gamma$ controls the marginal densities $p _ { j }$ . Our estimator uses sample splitting: the first subsample estimates the marginal densities, and the second fits the regression function in the resulting estimated dictionary by thresholded least squares. In the following, we take a pair $( p , f ) \in { \mathcal { C } } _ { \beta , \gamma }$ and provide a generic estimator that is data driven and independent of the particular choice of this pair. We fix the representation of f using the equation (11).

Construction of the estimator. Split the sample into two independent subsamples of sizes $n _ { 0 } = \lfloor n / 2 \rfloor$ and $n _ { 1 } = n - n _ { 0 }$ . Write $\mathcal { D } _ { n } ^ { 0 }$ for the first-subsample covariates, $\mathcal { D } _ { n } ^ { 1 }$ for the second one and $\mathcal { D } _ { n }$ for all observed covariates. Using $\mathcal { D } _ { n } ^ { 0 }$ , one can construct marginaldensity estimators $\widehat { p } _ { j }$ satisfying $\kappa _ { \mathrm { m i n } } \le \widehat { p } _ { j } \le \kappa _ { \mathrm { m a x } }$ and

$$
\operatorname* { m a x } _ { 1 \leq j \leq d } \operatorname* { s u p } _ { t \in [ 0 , 1 ] } \mathbb { E } \big [ | \widehat { p } _ { j } ( t ) - p _ { j } ( t ) | ^ { 2 } \big ] \leq \rho _ { n _ { 0 } } ,\tag{23}
$$

uniformly over $( p , f ) \in { \mathcal { C } } _ { \beta , \gamma }$ , where $\rho _ { n _ { 0 } } \lesssim n _ { 0 } ^ { - 2 \gamma / ( 2 \gamma + 1 ) }$ Such estimators can be obtained by piecewise polynomial projection followed by clipping; the construction and its risk bound are given in Appendix.

For all $j \in \{ 1 , \ldots , d \}$ and $m \geq 1$ , we define the functions $\widehat { \psi } _ { m } ^ { ( j ) }$ which replace the oracle densities $p _ { j }$ by their respective estimator and debiase the centering error due to the estimation of the densities.

$$
\widehat { \psi } _ { m } ^ { ( j ) } ( x ) : = \frac { \phi _ { m } ( x ) } { \widehat { p } _ { j } ( x ) } - \int _ { 0 } ^ { 1 } \frac { \phi _ { m } ( u ) } { \widehat { p } _ { j } ( u ) } d u , \qquad x \in [ 0 , 1 ] .\tag{24}
$$

Let $M \geq 1$ be a truncation level. We introduce $K : = 1 +$ dM and for any $\mathbf { x } \in [ 0 , 1 ] ^ { d }$ the K−dimensional vector $\widehat { \psi } _ { M } ( \mathbf { x } )$ which collects the constant function 1 and these dictionary elements, ordered by coordinate and then by frequency. The fitted constant compensates for the Lebesgue centering in (24); it is needed even though the true regression function is P−centered.

Let $\widehat { \Psi } _ { M } \in \mathbb { R } ^ { n _ { 1 } \times K }$ have rows $\widehat { \psi } _ { M } ( \mathbf { X } _ { i } ^ { 1 } ) ^ { \top }$ , where $\mathbf { X } _ { i } ^ { 1 }$ denotes a second-subsample covariate. Define the Gram matrices

$$
\widehat { \mathbf { r } } _ { M } \mathbf { \Psi } : = \begin{array} { r } { \frac { 1 } { n _ { 1 } } \widehat { \Psi } _ { M } ^ { \top } \widehat { \Psi } _ { M } , } \end{array}\tag{25}
$$

$$
\begin{array} { r l r } { { \mathbf { { r } } _ { M } } } & { : = } & { { \mathbb { E } } _ { p } \left[ \widehat { \psi } _ { M } ( { \mathbf { X } } ) \widehat { \psi } _ { M } ( { \mathbf { X } } ) ^ { \top } \mid \mathcal { D } _ { n } ^ { 0 } \right] , } \end{array}\tag{26}
$$

where the expectation is taken under $P$ and independently of $\mathcal { D } _ { n } ^ { 0 }$ . Define the event

$$
\zeta _ { n } = \left\{ \lambda _ { \operatorname* { m i n } } ( \widehat { \mathbf { { T } } } _ { M } ) \geq C _ { \operatorname* { m i n } } / 2 \right\} .
$$

We finally denote $\pmb { Y } ^ { 1 } : = ( Y _ { 1 } ^ { 1 } , \ldots , Y _ { n _ { 1 } } ^ { 1 } ) ^ { \top }$

Definition 4.1. With the previous notations, we define the estimator ${ \widehat { f } } _ { M }$ for all $\mathbf { x } \in [ 0 , 1 ] ^ { d }$ as follows

$$
\widehat { f } _ { M } ( \mathbf { x } ) : = \widehat { \psi } _ { M } ( \mathbf { x } ) ^ { \top } \widehat { \pmb { \eta } } _ { M } ,\tag{27}
$$

where $\widehat { \pmb { \eta } } _ { M } : = ( \widehat { \pmb { \Psi } } _ { M } ^ { \top } \widehat { \pmb { \Psi } } _ { M } ) ^ { - 1 } \widehat { \pmb { \Psi } } _ { M } ^ { \top } \pmb { Y } ^ { 1 } \ \mathbf { 1 } _ { \zeta _ { n } } \in \mathbb { R } ^ { K } .$

The inverse is evaluated only on $\zeta _ { n } .$ , where it exists.

Bias-Variance decomposition. The analysis combines truncation and dictionary estimation into a single approximation residual. We introduce the quantity $\begin{array} { r } { \mu _ { f } = \int _ { [ 0 , 1 ] ^ { d } } f ( x ) } \end{array}$ dx which is used in the proof but not needed to be known and define the key function $t _ { M }$ as follows

$$
t _ { M } : = \mu _ { f } + \sum _ { j = 1 } ^ { d } \sum _ { m = 1 } ^ { M } \theta _ { m } ^ { ( j ) } \widehat { \psi } _ { m } ^ { ( j ) } .\tag{28}
$$

Let also introduce the quantities $r _ { M } : = f - t _ { M }$ and $\pmb { r } _ { M } = ( r _ { M } ( \mathbf { X } _ { 1 } ^ { 1 } ) , \dots , r _ { M } ( \bar { \mathbf { X } } _ { n _ { 1 } } ^ { 1 } ) ) ^ { \top }$ . The comparison function $t _ { M }$ belongs to the fitted space and is used only in the analysis. Its coeficients are not required to compute the estimator. We introduce the following two quantities:

$$
\begin{array}{c} \begin{array} { r } { \{ \mathcal { B } _ { n } : = \| r _ { M } - \frac { 1 } { n _ { 1 } } \widehat { \psi } _ { M } ^ { \top } \widehat { \mathbf { T } } _ { M } ^ { - 1 } \widehat { \boldsymbol { \Psi } } _ { M } ^ { \top } \boldsymbol { r } _ { M } \| ^ { 2 } \mathbf { 1 } _ { \zeta _ { n } } , } \\ { \lfloor \mathcal { V } _ { n } : = \mathbb { E } [ \| \frac { 1 } { n _ { 1 } } \widehat { \psi } _ { M } ^ { \top } \widehat { \mathbf { T } } _ { M } ^ { - 1 } \widehat { \boldsymbol { \Psi } } _ { M } ^ { \top } \boldsymbol { \xi } \| ^ { 2 } \mathbf { 1 } _ { \zeta _ { n } } \mid \mathcal { D } _ { n } ] , } \end{array}   \end{array}\tag{29}
$$

$B _ { n }$ is the conditional bias and $\nu _ { n }$ the conditional variance. Given these two terms, we have the standard bias-variance decomposition, stated in the following.

Lemma 4.2. For a fixed pair $( p , f )$ , we have the $f o l -$ lowing decompositionof the conditional risk on the high probability event

$$
\mathbb { E } _ { p , f } \Big [ \| f - \widehat { f } _ { M } \| ^ { 2 } { \bf 1 } _ { \big < _ { n } } \ | \ \mathcal { D } _ { n } \Big ] = \mathcal { B } _ { n } + \mathcal { V } _ { n }\tag{30}
$$

Indeed, conditional on $\mathcal { D } _ { n } .$ the estimated dictionary and the residual are fixed. Expanding the least-squares estimator gives a residual contribution and a linear noise contribution. The latter has conditional mean zero, so their cross term vanishes.

Upper bound. We outline the four ingredients of the proof to obtain the desired upper bound, which are detailed in Appendix. First, the most technical one is the following Lemma, which makes clearly appear the density estimation term.

Lemma 4.3 (Approximation error).

$$
\mathbb { E } _ { p } \big [ \| r _ { M } \| ^ { 2 } \big ] \leq 2 \kappa _ { \operatorname* { m a x } } d R ^ { 2 } \left( \frac { M ^ { - 2 \beta } } { \kappa _ { \operatorname* { m i n } } ^ { 2 } } + \frac { \rho _ { n _ { 0 } } } { \kappa _ { \operatorname* { m i n } } ^ { 4 } } \right) .\tag{31}
$$

Then, we show in Appendix G.6 the following upper bound on the bias

$$
\mathbb { E } \left[ \boldsymbol { \mathcal { B } } _ { n } \mid \boldsymbol { \mathcal { D } } _ { n } ^ { 0 } \right] \leq \left( 2 + \frac { 4 C _ { \operatorname* { m a x } } } { C _ { \operatorname* { m i n } } } \right) \| \boldsymbol { r } \| ^ { 2 } ,\tag{32}
$$

and, we show in Appendix G.7 the following upper bound on the variance

$$
\mathbb { E } \left[ \mathcal { V } _ { n } \mid \mathcal { D } _ { n } ^ { 0 } \right] \leq \frac { 2 \sigma ^ { 2 } C _ { \operatorname* { m a x } } K } { n _ { 1 } C _ { \operatorname* { m i n } } } .\tag{33}
$$

Finally, by using the Matrix-Chernof inequality (Tropp (2015), recalled in our Theorem D.6), we obtain a bound on $\mathbb { P } ( \zeta _ { n } ^ { c } \mid \mathcal { D } _ { n } ^ { 0 } )$ , that is negligible. Then, by equilibrating the bias and the variance through $\stackrel { \cdot } { M } \stackrel { \cdot } { \asymp } n ^ { \bar { 1 } / ( 2 \beta + 1 ) }$ to obtain the desired rate, we provide the global upper bound on the class in the following theorem (proof in Appendix G).

Theorem 4.4. under Assumption ${ \it 2 . 1 3 , }$ we have the following minimax upper bound

$$
\mathfrak { M } ( n , d , \mathcal { C } _ { \beta , \gamma } ) \lesssim d n ^ { - \frac { 2 \beta } { 2 \beta + 1 } } + d n ^ { - \frac { 2 \gamma } { 2 \gamma + 1 } } ,\tag{34}
$$

where all the underlying constants behind the symbol ≲ depend only on $\beta , \gamma , L , R , \sigma , \kappa _ { \mathrm { m i n } } , \kappa _ { \mathrm { m a x } }$

In plain word, this result illustrates that learning the function and the geometry can be done at the rate of the two rates to do each task separately.

Recovering the true components. The centered coordinate blocks of the estimator given in Definition 4.1 estimate $\textstyle f _ { j } - \int _ { 0 } ^ { 1 } f _ { j } ( t ) d t$ , rather than the true components $f _ { j }$ . Nevertheless, these components can be recovered from the same fitted coeficients.

Let denote $\widehat { \theta } _ { m } ^ { ( j ) }$ denote the coeficient of frequency $m \in$ $\{ 1 , \dots , M \}$ and coordinate $j \in \{ 1 , \ldots , d \}$ of the vector $\widehat { \pmb { \eta } } _ { M } \in \mathbb { R } ^ { K }$ from the equation (27).

Definition 4.5. For all coordinate $j ,$ let define the debiased estimator $\bar { f } _ { j , M }$ of $f _ { j }$ as follows

$$
\bar { f } _ { j , M } : = \frac { 1 } { \widehat { p } _ { j } } \sum _ { m = 1 } ^ { M } \widehat { \theta } _ { m } ^ { ( j ) } \phi _ { m } .\tag{35}
$$

For a given $\widehat { \pmb { \eta } } _ { M } .$ , this estimator requires no additional fitting and natively satisfies $\begin{array} { r } { \int _ { 0 } ^ { 1 } \bar { f } _ { j , M } ( x ) \widehat { p } _ { j } ( x ) d x = 0 } \end{array}$ Thus, it is centered with respect to the estimated marginal weights. The following result shows that it consistently estimates the true component.

Theorem 4.6. under Assumption 2.13, choose $M \asymp$ n 2β+1 . Then, for all suficiently large n, the risk on the estimators $\bar { f } _ { j , M }$ achieve the following rate

$$
\sum _ { j = 1 } ^ { d } \mathbb { E } _ { p , f } \big [ \| \bar { f } _ { j , M } - f _ { j } \| ^ { 2 } \big ] \lesssim d n ^ { - \frac { 2 \beta } { ( 2 \beta + 1 ) } } + d n ^ { - \frac { 2 \gamma } { ( 2 \gamma + 1 ) } } .\tag{36}
$$

The implicit constants behind the symbol $\lesssim$ depend only on $\beta , \gamma , L , R , \sigma , \kappa _ { \mathrm { m i n } } , \kappa _ { \mathrm { m a x } } .$

The proof uses the uniform lower bound on the population Gram matrix to control the fitted coeficient error by the prediction and approximation errors. Combining this control with the componentwise approximation bounds underlying Lemma 4.3 yields the desired rate. A an immediate consequence, we can provide the following upper bound on the entire class

$$
\begin{array} { l } { \displaystyle \operatorname* { i n f } _ { \widehat { f } _ { 1 } , \ldots , \widehat { f } _ { d } } \operatorname* { s u p } _ { ( p , f ) \in { \mathscr C } _ { \beta , \gamma } } \sum _ { j = 1 } ^ { d } \mathbb { E } _ { p , f } \left[ \| \widehat { f } _ { j } - f _ { j } \| ^ { 2 } \right] } \\ { \lesssim d n ^ { - \frac { 2 \beta } { ( 2 \beta + 1 ) } } + d n ^ { - \frac { 2 \gamma } { ( 2 \gamma + 1 ) } } . } \end{array}\tag{37}
$$

The complete proof is provided in Appendix.

## 4.2 Minimax lower bound

To establish the minimax lower bound, we must distinguish two regimes. The smooth one and the rough one. If we are in the case where $\gamma \geq \beta$ , we can show that an admissible lower bound achieves the rate of $d n ^ { - \frac { 2 \beta } { 2 \beta + 1 } }$ In the case where $\gamma < \beta$ , the error of estimating the d marginal densities should be higher than $d n ^ { - \frac { \overline { { 2 \beta } } } { 2 \beta + 1 } }$

Smooth regime. In this paragraph, we study the case where $\gamma \geq \beta .$ . In the next theorem, we provide a result that is true for every $\gamma > 0$

Theorem 4.7. There exists an underlying constant which depends only on $\beta , R , \sigma$ such that

$$
\mathfrak { M } ( n , d , \mathcal { C } _ { \beta , \gamma } ) \gtrsim d n ^ { - \frac { 2 \beta } { 2 \beta + 1 } } .\tag{38}
$$

This result is proved in Appendix H and uses the fact that the uniform density $p _ { \mathrm { u n i f } } \equiv 1$ belongs to $\mathcal { P } _ { \gamma , L }$ for every $\gamma > 0 .$ . In particular, in the smooth regime and under Assumption 2.13, we have for all suficiently large n,

$$
\boxed { \mathfrak { M } ( n , d , \mathcal { C } _ { \beta , \gamma } ) \asymp d n ^ { - \frac { 2 \beta } { 2 \beta + 1 } } }\tag{39}
$$

which leads to the minimax optimality when $\gamma \geq \beta$

Rough regime. In this paragraph, we study the case where $\gamma < \beta .$ . The natural rate we want to obtain is $d n ^ { - \frac { 2 \gamma } { 2 \gamma + 1 } }$ . In order to have matching bounds which tend to zero, we make the following assumption on potentially diverging dimension in this regime.

Assumption 4.8. We suppose that d can grow with n at the following rate

$$
d = o \Big ( n ^ { \frac { 2 \gamma } { 2 \gamma + 1 } } \Big ) .\tag{40}
$$

We establish the following theorem.

Theorem 4.9. under Assumption $4 . 8 ,$ we have the following minimax lower bound

$$
\mathfrak { M } ( n , d , \mathcal { C } _ { \beta , \gamma } ) \gtrsim d n ^ { - \frac { 2 \gamma } { 2 \gamma + 1 } } ,\tag{41}
$$

where all the underlying constants behind the symbol ≲ depend only on $\beta , \gamma , L , R , \sigma , \kappa _ { \mathrm { m i n } } , \kappa _ { \mathrm { m a x } }$

In this rough regime, we have $d n ^ { - \frac { 2 \gamma } { 2 \gamma + 1 } } \to 0$ and $d n ^ { - \frac { 2 \beta } { 2 \beta + 1 } } = o ( d n ^ { - \frac { 2 \gamma } { 2 \gamma + 1 } } )$ which leads to the minimax optimality

$$
\boxed { \mathfrak { M } ( n , d , \mathcal { C } _ { \beta , \gamma } ) \asymp d n ^ { - \frac { 2 \gamma } { 2 \gamma + 1 } } }\tag{42}
$$

The proof of this theorem is deferred to Appendix I.

The main idea is to vary the marginal densities while keeping the weighted regression components fixed. Choose a simple smooth univariate function g with zero integral and satisfying the regular Sobolev smoothness assumption. For a sign array $\omega ,$ construct the following joint density

$$
p _ { \omega } ( \mathbf x ) : = 1 + \varepsilon \sin \left( \sum _ { j = 1 } ^ { d } u _ { \omega , j } ( x _ { j } ) \right) , \quad \mathbf x \in [ 0 , 1 ] ^ { d } ,\tag{43}
$$

where each $u _ { \omega , j }$ is a signed sum of disjoint, antisymmetric smooth bumps of width h and amplitude $a h ^ { \gamma }$ , with $a , \varepsilon > 0$ suficiently small. Then define the components $f _ { \omega , j }$ such that $p _ { \omega , j } f _ { \omega , j } = g$ , so the Sobolev constraint and the centering condition hold for every alternative.

The sine coupling keeps the joint density uniformly bounded, independently of d. Antisymmetry gives explicit marginals of the form $p _ { \omega , j } = 1 + t _ { h }$ sin $u _ { \omega , j } .$ The same symmetry yields an afine representation of $f _ { \omega }$ with order $d / h$ mutually orthogonal directions.

For adjacent sign arrays, the squared separation is of order $h ^ { 2 \bar { \gamma } + 1 }$ . Writing $Q _ { \omega }$ for the law of one observed pair and using standard information theory results (Cover and Thomas, 1991), we show that the information in both the covariates and the responses satisfies

$$
\begin{array} { r } { \mathrm { K L } \big ( Q _ { \omega } ^ { \otimes n } \| Q _ { \omega ^ { \prime } } ^ { \otimes n } \big ) \lesssim a ^ { 2 } n h ^ { 2 \gamma + 1 } . } \end{array}\tag{44}
$$

Taking $\begin{array} { r } { h ~ \asymp ~ n ^ { - 1 / ( 2 \gamma + 1 ) } } \end{array}$ keeps neighboring alternatives statistically indistinguishable. Assouad’s lemma (Tsybakov, 2009) then combines their separation over the $d / h$ directions, giving the lower bound $d h ^ { 2 \gamma } =$ $d n ^ { - 2 \stackrel { \prime } { \gamma } / ( 2 \gamma + 1 ) }$ . Uniform joint-density bounds allow the argument in $L ^ { 2 } ( \lambda _ { d } )$ to transfer to the prediction loss under each $P _ { \omega }$

## 5 DISCUSSION

Conclusion. We established minimax rates for additive regression under dependent random designs with growing dimension. Our coupled smoothness framework identifies when estimating the marginal densities preserves the known-density rate and when their regularity determines the minimax rate.

Limitations & future work. The main limitation is the assumption that the joint density is bounded above and away from zero uniformly in the dimension. Although density bounds are classical, this uniformity remains restrictive. Our guarantees also require dimension-growth conditions and do not cover sparse high-dimensional models. Relaxing the density bounds is an important direction.

A natural extension is to adapt the Riesz representation of Ferrere et al. (2026) to (potentially sparse) nonparametric interaction models, retaining dependent designs and diverging dimension. The aim would be to recover minimax rates with known marginals, then quantify the additional costs of estimating multivariate marginal densities and selecting active components. Kernel estimators extending Raskutti et al. (2012) or deep neural network estimators following Bhattacharya et al. (2024) provide two concrete approaches to investigate.

## Acknowledgements

The authors thank Nicolas Bousquet (EDF R&D) for his careful reading and valuable comments. This work was partially supported by the French Association Nationale de la Recherche et de la Technologie (ANRT) through a CIFRE PhD project at Électricité de France (EDF). Fabrice Gamboa and Jean-Michel Loubes acknowledge support from the ANR-3IA Artificial and Natural Intelligence Toulouse Institute (ANITI).

## References

Agarwal, R., Melnick, L., Frosst, N., Zhang, X., Lengerich, B., Caruana, R., and Hinton, G. E. (2021). Neural additive models: Interpretable machine learning with neural nets. Advances in neural information processing systems, 34:4699–4711.

Bénard, C. (2025). Tree Ensemble Explainability through the Hoefding Functional Decomposition and TreeHFD Algorithm. Advances in Neural Information Processing Systems.

Bhattacharya, S., Fan, J., and Mukherjee, D. (2024). Deep neural networks for nonparametric interaction models with diverging dimension. The Annals of Statistics, 52(6):2738–2766.

Brézis, H. (2011). Functional analysis, Sobolev spaces and partial diferential equations, volume 2. Springer.

Chang, C.-H., Caruana, R., and Goldenberg, A. (2021). NODE-GAM: Neural generalized additive model for interpretable deep learning. arXiv:2106.01613.

Chastaing, G., Gamboa, F., and Prieur, C. (2012). Generalized Hoefding-Sobol Decomposition for Dependent Variables – Application to Sensitivity Analysis. Electronic Journal of Statistics, 6:2420–2448.

Cover, T. M. and Thomas, J. A. (1991). Elements of information theory, volume 2. wiley New York.

Ferrere, B., Bousquet, N., Gamboa, F., and Loubes, J.-M. (2026). Generalized functional anova: A complete theoretical framework. arXiv preprint arXiv:2605.18422.

Haris, A., Simon, N., and Shojaie, A. (2022). Generalized sparse additive models. Journal of machine learning research, 23(70):1–56.

Hastie, T. and Tibshirani, R. (1986). Generalized additive models. Statistical science, 1(3):297–310.

Hooker, G. (2007). Generalized Functional ANOVA Diagnostics for High-Dimensional Functions of Dependent Variables. Journal of Computational and Graphical Statistics, 16(3):709–732.

Horowitz, J., Klemelä, J., and Mammen, E. (2006). Optimal estimation in additive regression models. Bernoulli, 12(2):271–298.

Huang, J. Z. (1998). Projection estimation in multiple regression with application to functional anova models. The annals of statistics, 26(1):242–272.

Lengerich, B., Tan, S., Chang, C.-H., Hooker, G., and Caruana, R. (2020). Purifying interaction efects with the functional ANOVA: An eficient algorithm for recovering identifiable additive models. In International Conference on Artificial Intelligence and Statistics, pages 2402–2412. PMLR.

Liu, S. (2025). Minimax-optimal univariate function selection in sparse additive models: Rates, adaptation, and the estimation-selection gap. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Mammen, E., Linton, O., and Nielsen, J. (1999). The existence and asymptotic properties of a backfitting projection algorithm under weak conditions. The Annals of Statistics, 27(5):1443–1490.

Meier, L., Van de Geer, S., and Bühlmann, P. (2009). High-dimensional additive modeling. The annals of Statistics.

Nori, H., Jenkins, S., Koch, P., and Caruana, R. (2019). InterpretML: A unified framework for machine learning interpretability. arXiv preprint arXiv:1909.09223.

Radenovic, F., Dubey, A., and Mahajan, D. (2022). Neural basis models for interpretability. Advances in Neural Information Processing Systems, 35:8414– 8426.

Raskutti, G., J Wainwright, M., and Yu, B. (2012). Minimax-optimal rates for sparse additive models over kernel classes via convex programming. Journal of machine learning research, 13(2).

Ravikumar, P., Laferty, J., Liu, H., and Wasserman, L. (2009). Sparse additive models. Journal of the Royal Statistical Society Series B: Statistical Methodology, 71(5):1009–1030.

Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nature Machine Intelligence, 1(5):206–215.

Stone, C. J. (1985). Additive regression and other nonparametric models. The annals of Statistics, pages 689–705.

Stone, C. J. (1994). The use of polynomial splines and their tensor products in multivariate function estimation. The Annals of Statistics, pages 118–171.

Szegő, G. (1939). Orthogonal polynomials, volume 23. American Mathematical Soc.

Tan, Z. and Zhang, C.-H. (2019). Doubly penalized estimation in additive regression with high-dimensional data. The Annals of Statistics.

Tropp, J. A. (2015). An introduction to matrix concentration inequalities. Foundations and trends® in machine learning, 8(1-2):1–230.

Tsybakov, A. B. (2009). Introduction to nonparametric estimation, volume 11. Springer.

Yuan, M. and Zhou, D.-X. (2016). Minimax optimal rates of estimation in high dimensional additive models. The Annals of Statistics.

## Appendix

A General notations 13   
B Proof of Theorem 2.5 13   
C Proof of Theorem 2.6 14   
D Technical lemmas 15   
D.1 Information theoretic lemmas 15   
D.2 Concentration inequality on matrices 16   
E Proof of Theorem 3.2 17   
E.1 Notation and preliminary bounds . 17   
E.2 Thresholded least squares and the error decomposition . 18   
E.3 Bounding the squared bias . 19   
E.4 Bounding the conditional variance 19   
E.5 Concentration of the empirical Gram matrix . 20   
E.6 Finite-sample risk bound . 20   
E.7 Obtaining minimax upper bound 20   
F Proof of Theorem 3.3 21   
F.1 Construction of a finite subfamily . 21   
F.2 Controlling neighboring observation laws . 22   
F.3 Reduction from prediction loss to Hamming loss 22   
F.4 Application of Assouad’s lemma . 23   
G Proof of Theorem 4.4 23   
G.1 Notations 23   
G.2 The estimator 25   
G.3 Proof of Lemma 4.2 25   
G.4 Uniform geometry of the estimated dictionary 26   
G.5 Proof of Lemma 4.3 28   
G.6 Bounding the conditional squared bias . 29   
G.7 Bounding the conditional variance 29   
G.8 Risk on the rejected event 30   
G.9 Combining the bounds . 30   
G.10 Construction of the marginal-density estimators 31   
G.11 Proof of Theorem 4.6 . 34   
H Proof of Theorem 4.7 35   
I Proof of Theorem 4.9 36   
I.1 Smooth functions construction 37   
I.2 Construction of alternatives 38   
I.3 Orthogonal separation of the regression functions 41   
I.4 Information in the covariates and the responses 42   
I.5 Assouad’s argument for the rough-density contribution . 43

## A GENERAL NOTATIONS

Let λ denote the Lebesgue measure on [0, 1]. We denote by $\lambda _ { d } : = \lambda ^ { \otimes d }$ the product Lebesgue measure on $[ 0 , 1 ] ^ { d } .$ Recall that for any $j \in \{ 1 , \ldots , d \}$ , we denote by $P _ { j }$ the marginal distribution of $X _ { j }$ and by $p _ { j }$ the corresponding density, such that $\begin{array} { r } { \frac { d P _ { j } } { d \lambda } = p _ { j } . } \end{array}$ . We denote by $L ^ { 2 } ( P )$ the Hilbert space of square integrable and P−measurable functions defined on $[ 0 , 1 ] ^ { d } .$ . For simplicity, we denote by $\| \cdot \|$ the $\bar { L } ^ { 2 } ( P )$ −norm instead of $\| \cdot \| _ { L ^ { 2 } ( P ) }$ and $\| \cdot \| _ { 2 }$ the Euclidean norm. When dealing with potentially infinite sequences, we denote by $\| \cdot \| _ { \ell ^ { 2 } }$ the corresponding $\dot { \ell } ^ { 2 }$ norm. When computing a norm under the Lebesgue measure, we will denote explicitly $\| \cdot \| _ { L ^ { 2 } ( \lambda ) }$ for the univariate case and $\| \cdot \| _ { L ^ { 2 } ( \lambda _ { d } ) }$ for the d−variate case.

Throughout this Appendix, logarithms are natural. For probability measures $P , Q$ on the same measurable space, we use the notations

$$
\operatorname { T V } ( P , Q ) : = \operatorname* { s u p } _ { A } | P ( A ) - Q ( A ) | ,\tag{45}
$$

for the total variation and

$$
\mathrm { K L } ( P \| Q ) : = \left\{ \int \log \left( { \frac { d P } { d Q } } \right) d P , \mathrm { i f } P \ll Q , \right.\tag{46}
$$

for the Kullback-Leibler divergence.

We recall the real trigonometric basis (Tsybakov, 2009), with $\phi _ { 0 } \equiv 1$ and, for every integer $m \geq 1$

$$
\phi _ { 2 m - 1 } ( x ) = \sqrt { 2 } \cos ( 2 \pi m x ) , \qquad \phi _ { 2 m } ( x ) = \sqrt { 2 } \sin ( 2 \pi m x ) , \qquad x \in [ 0 , 1 ] .\tag{47}
$$

The functions $( \phi _ { m } ) _ { m \geq 1 }$ form an orthonormal basis of the mean-zero subspace of $L ^ { 2 } ( \lambda )$ and satisfy

$$
\int _ { 0 } ^ { 1 } \phi _ { m } ( x ) d x = 0 , \qquad \| \phi _ { m } \| _ { L ^ { \infty } ( [ 0 , 1 ] ) } \leq \sqrt { 2 } .\tag{48}
$$

## B PROOF OF THEOREM 2.5

We want to show that Ψ is a Riesz sequence (Brézis, 2011). Let take a finite sequence $\pmb { a } : = \left( ( a _ { m } ^ { ( j ) } ) _ { m \geq 1 } \right) _ { j = 1 , \ldots , d } ,$ we define the function $f$ as

$$
f : = \sum _ { j = 1 } ^ { d } \sum _ { m \geq 1 } a _ { m } ^ { ( j ) } \psi _ { m } ^ { ( j ) } .\tag{49}
$$

Recall that we have for any function u, Var $\mathbf { \Sigma } _ { P } ( u ) = \operatorname* { i n f } _ { x \in \mathbb { R } } \int ( u - x ) ^ { 2 } d P$ , which leads to the fundamental inequality of the variance

$$
\begin{array} { r } { p _ { \operatorname* { m i n } } \operatorname { V a r } _ { \lambda ^ { \otimes d } } ( u ) \leq \operatorname { V a r } _ { P } ( u ) \leq p _ { \operatorname* { m a x } } \operatorname { V a r } _ { \lambda ^ { \otimes d } } ( u ) . } \end{array}\tag{50}
$$

In our setting, each component $f _ { j }$ is centered under $P _ { j }$ , so this previous inequality becomes:

$$
p _ { \operatorname* { m i n } } \sum _ { j = 1 } ^ { d } \mathrm { V a r } _ { \lambda } ( f _ { j } ) \leq \| f \| ^ { 2 } \leq p _ { \operatorname* { m a x } } \sum _ { j = 1 } ^ { d } \mathrm { V a r } _ { \lambda } ( f _ { j } )\tag{51}
$$

The same reasoning on each component yields

$$
\forall j \in \{ 1 , \dots , d \} , \qquad p _ { \operatorname* { m i n } } \mathrm { V a r } _ { \lambda } ( f _ { j } ) \leq \| f _ { j } \| ^ { 2 } \leq p _ { \operatorname* { m a x } } \mathrm { V a r } _ { \lambda } ( f _ { j } )\tag{52}
$$

By using this last inequality and summing over $j ,$ we have :

$$
{ \frac { 1 } { p _ { \operatorname* { m a x } } } } \sum _ { j = 1 } ^ { d } \| f _ { j } \| ^ { 2 } \leq \sum _ { j = 1 } ^ { d } \operatorname { V a r } _ { \lambda } ( f _ { j } ) \leq { \frac { 1 } { p _ { \operatorname* { m i n } } } } \sum _ { j = 1 } ^ { d } \| f _ { j } \| ^ { 2 }\tag{53}
$$

By injecting this in equation (51), we obtain

$$
{ \frac { p _ { \operatorname* { m i n } } } { p _ { \operatorname* { m a x } } } } \sum _ { j = 1 } ^ { d } \| f _ { j } \| ^ { 2 } \leq \| f \| ^ { 2 } \leq { \frac { p _ { \operatorname* { m a x } } } { p _ { \operatorname* { m i n } } } } \sum _ { j = 1 } ^ { d } \| f _ { j } \| ^ { 2 }\tag{54}
$$

Finally, observe that by definition

$$
f _ { j } = \sum _ { m } \frac { a _ { m } ^ { ( j ) } \phi _ { m } ^ { ( j ) } } { p _ { j } } ,\tag{55}
$$

so the corresponding $\| \cdot \| ^ { 2 }$ norm satisfies

$$
\frac { 1 } { p _ { \operatorname* { m a x } } } \sum _ { m } \left( a _ { m } ^ { ( j ) } \right) ^ { 2 } \leq \| f _ { j } \| ^ { 2 } = \int \left( \sum _ { m } a _ { m } ^ { ( j ) } \phi _ { m } ^ { ( j ) } \right) ^ { 2 } \frac { 1 } { p _ { j } } d \lambda \leq \frac { 1 } { p _ { \operatorname* { m i n } } } \sum _ { m } \left( a _ { m } ^ { ( j ) } \right) ^ { 2 } ,\tag{56}
$$

which leads to the desired conclusion.

## C PROOF OF THEOREM 2.6

The Theorem 2.5 shows that the collection of function Ψ is a Riesz sequence of $L ^ { 2 } ( P )$ , so in order to have the result, it sufices to show that it is a Riesz basis (Brézis, 2011) of H.

Let take a function $f = f _ { 1 } + \cdot \cdot \cdot + f _ { d } \in \mathcal { H }$ and set for every j, $g _ { j } = p _ { j } f _ { j }$ . The $g _ { j }$ are centered and belong to $L ^ { 2 } ( \lambda )$ Indeed, we have

$$
\int _ { [ 0 , 1 ] } g _ { j } ( x ) d x = \int _ { [ 0 , 1 ] } f _ { j } ( x ) p _ { j } ( x ) d x = \mathbb { E } _ { P _ { j } } [ f _ { j } ( X ) ] = 0 ,\tag{57}
$$

and under Assumtion 2.3

$$
\| g _ { j } \| _ { L ^ { 2 } ( \lambda ) } ^ { 2 } = \int _ { [ 0 , 1 ] } g _ { j } ^ { 2 } ( x ) d x = \int _ { [ 0 , 1 ] } f _ { j } ^ { 2 } ( x ) p _ { j } ^ { 2 } ( x ) d x \leq p _ { \operatorname* { m a x } } \| f _ { j } \| ^ { 2 } < \infty .\tag{58}
$$

Using that the trigonometric basis $\left( \phi _ { m } \right)$ is a Hilbert basis of $L ^ { 2 } ( \lambda )$ (Brézis, 2011; Tsybakov, 2009), any centered function of $L ^ { 2 } ( \lambda )$ expresses as $\sum _ { m \ge 1 } \alpha _ { m } \phi _ { m }$ , which yields the following fundamental expansion of $g _ { j }$

$$
g _ { j } = \sum _ { m = 1 } ^ { \infty } \theta _ { m } ^ { ( j ) } \phi _ { m } , \qquad j = 1 , \ldots , d ,\tag{59}
$$

where $\theta _ { m } ^ { ( j ) } : = \langle g _ { j } , \phi _ { m } \rangle _ { L ^ { 2 } ( \lambda ) }$ . Now, let define the residual $R _ { M } ^ { ( j ) }$ of order $M \geq 1$ as

$$
R _ { M } ^ { ( j ) } : = f _ { j } - \sum _ { m = 1 } ^ { M } \theta _ { m } ^ { ( j ) } \psi _ { m } ^ { ( j ) } .\tag{60}
$$

We have

(61)

$$
\begin{array} { l } { \displaystyle \left\| R _ { M } ^ { ( j ) } \right\| _ { L ^ { 2 } ( P _ { j } ) } ^ { 2 } = \left\| f _ { j } - \sum _ { m = 1 } ^ { M } \theta _ { m } ^ { ( j ) } \psi _ { m } ^ { ( j ) } \right\| ^ { 2 } } \\ { \displaystyle = \int _ { [ 0 , 1 ] } \left( f _ { j } ( x ) - \sum _ { m = 1 } ^ { M } \theta _ { m } ^ { ( j ) } \psi _ { m } ^ { ( j ) } ( x ) \right) ^ { 2 } p _ { j } ( x ) d x } \\ { \displaystyle = \int _ { [ 0 , 1 ] } \left( \frac { g _ { j } ( x ) - \sum _ { m = 1 } ^ { M } \theta _ { m } ^ { ( j ) } \phi _ { m } ( x ) } { p _ { j } ( x ) } \right) ^ { 2 } p _ { j } ( x ) d x } \end{array}\tag{62}
$$

(63)

$$
\leq \frac { 1 } { p _ { \mathrm { m i n } } } \int _ { [ 0 , 1 ] } \left( g _ { j } ( x ) - \sum _ { m = 1 } ^ { M } \theta _ { m } ^ { ( j ) } \phi _ { m } ( x ) \right) ^ { 2 } d x\tag{64}
$$

$$
\leq \frac { 1 } { p _ { \mathrm { m i n } } } \left\| g _ { j } - \sum _ { m = 1 } ^ { M } \theta _ { m } ^ { ( j ) } \phi _ { m } \right\| _ { L ^ { 2 } ( \lambda ) } ^ { 2 }\tag{65}
$$

This last quantity converges to 0 when $M \to \infty$ thanks to the expansion of $g _ { j }$ in the trigonometric basis, which ends the proof and allows to obtain the following expansion for any $f \in \mathcal H$

$$
\boxed { f = f _ { 1 } + \dots + f _ { d } = \sum _ { j = 1 } ^ { d } \sum _ { m = 1 } ^ { \infty } \underbrace { \left( \int _ { [ 0 , 1 ] } f _ { j } ( t ) \phi _ { m } ( t ) p _ { j } ( t ) d t \right) } _ { \theta _ { m } ^ { ( j ) } } \cdot \psi _ { m } ^ { ( j ) } . }\tag{66}
$$

## D TECHNICAL LEMMAS

## D.1 Information theoretic lemmas

Lemma D.1 (Elementary KL bounds, Cover and Thomas (1991)). For every measurable set A, with $t = Q ( A )$ and $s = Q ^ { \prime } ( A )$

$$
\mathrm { K L } ( Q \| Q ^ { \prime } ) \geq t \log ( t / s ) + ( 1 - t ) \log ( ( 1 - t ) / ( 1 - s ) ) ,\tag{67}
$$

with the usual continuous or infinite boundary values. If $q ^ { \prime } > 0$ wherever $q > 0$ , then

$$
\mathrm { K L } ( Q \| Q ^ { \prime } ) \leq \int \frac { ( q - q ^ { \prime } ) ^ { 2 } } { q ^ { \prime } } .\tag{68}
$$

Lemma D.2 (Pinsker’s inequality, Cover and Thomas (1991)). For any probability measures $Q , Q ^ { \prime }$

$$
\operatorname { T V } ( Q , Q ^ { \prime } ) \leq \sqrt { \operatorname { K L } ( Q \| Q ^ { \prime } ) / 2 } .\tag{69}
$$

Lemma D.3 (Assouad’s lemma for Hamming loss, (Tsybakov, 2009)). Let $N \geq 1$ , and let $\{ P _ { \omega } : \omega \in \{ - 1 , + 1 \} ^ { N } \}$ be a family of probability measures on the same measurable space. Suppose that an observation Z has distribution $P _ { \omega }$

Define the Hamming distance by

$$
\rho ( \omega , \omega ^ { \prime } ) : = \sum _ { k = 1 } ^ { N } \mathbf { 1 } \{ \omega _ { k } \neq \omega _ { k } ^ { \prime } \} ,\tag{70}
$$

and let $\omega ^ { ( k ) }$ denote the vector obtained by reversing only the k-th sign of ω. If, for some $\eta \in [ 0 , 1 ]$ 2

$$
\operatorname* { m a x } _ { \omega \in \{ - 1 , + 1 \} ^ { N } } \operatorname* { m a x } _ { 1 \leq k \leq N } \mathrm { T V } ( P _ { \omega } , P _ { \omega ^ { ( k ) } } ) \leq \eta ,\tag{71}
$$

then

$$
\operatorname* { i n f } _ { \widehat { \omega } } \operatorname* { s u p } _ { \omega \in \{ - 1 , + 1 \} ^ { N } } \mathbb { E } _ { \omega } \big [ \rho ( \widehat { \omega } ( Z ) , \omega ) \big ] \geq \frac { N } { 2 } ( 1 - \eta ) ,\tag{72}
$$

where the infimum ranges over all measurable estimators taking values in $\{ - 1 , + 1 \} ^ { N }$

Lemma D.4 (Orthogonal Assouad bound). Let $\begin{array} { r } { F _ { \omega } = F _ { 0 } + \sum _ { k = 1 } ^ { N } \omega _ { k } H _ { k } , \omega \in \{ - 1 , 1 \} ^ { N } } \end{array}$ , where the $H _ { k }$ are orthogonal in a real Hilbert space and $\| H _ { k } \| ^ { 2 } = v ^ { 2 } > 0$ . Let $Q _ { \omega }$ be the observation law. If adjacent vertices satisfy

$$
\operatorname* { m a x } _ { \omega , k } \mathrm { K L } ( Q _ { \omega } | | Q _ { \omega ^ { ( k ) } } ) \leq \alpha < 2 ,\tag{73}
$$

then

$$
\operatorname* { i n f } _ { \widehat { F } } \operatorname* { s u p } _ { \omega } \mathbb { E } _ { \omega } \| \widehat { F } - F _ { \omega } \| ^ { 2 } \geq \frac { N v ^ { 2 } } { 2 } \left( 1 - \sqrt { \alpha / 2 } \right) .\tag{74}
$$

Here $\omega ^ { ( k ) }$ is obtained by reversing sign $k .$

Proof. For any measurable Hilbert-space-valued estimator ${ \widehat { F } } _ { : }$ define

$$
\widehat { \omega } _ { k } = \left\{ \begin{array} { l l } { 1 , } & { \langle \widehat { F } - F _ { 0 } , H _ { k } \rangle \geq 0 , } \\ { - 1 , } & { \langle \widehat { F } - F _ { 0 } , H _ { k } \rangle < 0 . } \end{array} \right.\tag{75}
$$

By orthogonality and Bessel’s inequality,

$$
\| \widehat { F } - F _ { \omega } \| ^ { 2 } \geq \sum _ { k = 1 } ^ { N } \frac { | \langle \widehat { F } - F _ { 0 } , H _ { k } \rangle - \omega _ { k } v ^ { 2 } | ^ { 2 } } { v ^ { 2 } } \geq v ^ { 2 } \sum _ { k = 1 } ^ { N } \mathbf { 1 } _ { \{ \widehat { \omega } _ { k } \neq \omega _ { k } \} } .\tag{76}
$$

Let $\rho$ denote the Hamming distance. Assouad’s inequality in its Kullback–Leibler form (Tsybakov, 2009) gives

$$
\operatorname* { i n f } _ { \widetilde { \omega } } \operatorname* { s u p } _ { \omega } \mathbb { E } _ { \omega } \rho ( \widetilde { \omega } , \omega ) \geq \frac { N } { 2 } \left( 1 - \sqrt { \alpha / 2 } \right) .\tag{77}
$$

The theorem is stated on $\{ 0 , 1 \} ^ { N }$ , which is equivalent to $\{ - 1 , 1 \} ^ { N }$ under a coordinatewise relabeling. Combining the last two inequalities and taking the infimum over $\widehat F$ proves the claim. □

Lemma D.5 (Gaussian KL, Cover and Thomas (1991)). Let Q and $Q ^ { \prime }$ be the laws of one observation with design densities $p , p ^ { \prime }$ and conditional distributions $\mathcal { N } ( f ( \mathbf { x } ) , \sigma ^ { 2 } )$ and $\mathcal { N } ( f ^ { \prime } ( \mathbf { x } ) , \sigma ^ { 2 } )$ , respectively. When the right-hand side is finite,

$$
\operatorname { K L } ( Q ^ { \otimes n } \| ( Q ^ { \prime } ) ^ { \otimes n } ) = n \operatorname { K L } ( P \| P ^ { \prime } ) + { \frac { n } { 2 \sigma ^ { 2 } } } \| f - f ^ { \prime } \| ^ { 2 } .\tag{78}
$$

## D.2 Concentration inequality on matrices

Theorem D.6 (Matrix Chernof, Theorem 5.1.1, (Tropp, 2015)). Let $\boldsymbol { S } _ { 1 } , \ldots , \boldsymbol { S } _ { N }$ be independent random Hermitian $K \times K$ matrices such that $0 \preceq S _ { i } \preceq b _ { 0 } I _ { K }$ almost surely for every i, where $b _ { 0 } > 0$ . Let define

$$
\mu _ { \mathrm { m i n } } : = \lambda _ { \mathrm { m i n } } \Bigg ( \sum _ { i = 1 } ^ { N } \mathbb { E } [ S _ { i } ] \Bigg ) ,\tag{79}
$$

where the operator $\lambda _ { \mathrm { m i n } } ( \cdot )$ denotes the smallest eigen value. For every $0 \leq \varepsilon < 1$ ，

$$
\mathbb { P } \Bigg ( \lambda _ { \operatorname* { m i n } } \Bigg ( \sum _ { i = 1 } ^ { N } S _ { i } \Bigg ) \leq ( 1 - \varepsilon ) \mu _ { \operatorname* { m i n } } \Bigg ) \leq K \left( \frac { e ^ { - \varepsilon } } { ( 1 - \varepsilon ) ^ { 1 - \varepsilon } } \right) ^ { \mu _ { \operatorname* { m i n } } / b _ { 0 } } .\tag{80}
$$

In particular, taking $\varepsilon = 1 / 2$ yields

$$
\mathbb { P } \left( \lambda _ { \operatorname* { m i n } } \left( \sum _ { i = 1 } ^ { N } S _ { i } \right) \leq \frac { \mu _ { \operatorname* { m i n } } } { 2 } \right) \leq K \exp \left( - \frac { 1 - \log 2 } { 2 } \frac { \mu _ { \operatorname* { m i n } } } { b _ { 0 } } \right) .\tag{81}
$$

## E PROOF OF THEOREM 3.2

Throughout this section, the design density p is fixed and known. We write P for the corresponding probability distribution on $[ 0 , 1 ] ^ { d }$ , and $\mathcal { F }$ for the associated additive function class. Expectations under the regression model with design density p and regression function f are denoted by $\mathbb { E } _ { p , f }$

We prove a finite-sample risk bound for the thresholded estimator defined in (19), and then choose the truncation level M to obtain the minimax upper bound. The argument does not require independence between the coordinates of the design.

## E.1 Notation and preliminary bounds

Recall that

$$
Y _ { i } = f ( \mathbf { X } _ { i } ) + \xi _ { i } , \qquad i = 1 , \ldots , n ,\tag{82}
$$

where the design vectors are independent with common distribution $P ,$ and the errors are independent ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ random variables, independent of the design. For each $f \in { \mathcal { F } }$ , write

$$
f ( \mathbf { x } ) = \sum _ { j = 1 } ^ { d } \sum _ { m \geq 1 } \theta _ { m } ^ { ( j ) } \psi _ { m } ^ { ( j ) } ( x _ { j } ) , \qquad \psi _ { m } ^ { ( j ) } ( x _ { j } ) = \frac { \phi _ { m } ^ { ( j ) } ( x _ { j } ) } { p _ { j } ( x _ { j } ) } .\tag{83}
$$

The expansions are in $L ^ { 2 } ( P )$ . By the definition of the function class,

$$
\sum _ { m \ge 1 } m ^ { 2 \beta } \big ( \theta _ { m } ^ { ( j ) } \big ) ^ { 2 } \le R ^ { 2 } , \qquad j = 1 , \ldots , d .\tag{84}
$$

Recall also that the trigonometric functions are all bounded by ${ \sqrt { 2 } } .$

For an integer $M \geq 1$ , define the truncated function

$$
f _ { M } ( \mathbf { x } ) : = \sum _ { j = 1 } ^ { d } \sum _ { m = 1 } ^ { M } \theta _ { m } ^ { ( j ) } \psi _ { m } ^ { ( j ) } ( x _ { j } ) = \psi _ { M } ( \mathbf { x } ) ^ { \top } \pmb { \theta } _ { M } ,\tag{85}
$$

${ \pmb \theta } _ { M } \in \mathbb { R } ^ { d M }$ contains the first M coeficients of each component, and the residual

$$
r _ { M } : = f - f _ { M } .\tag{86}
$$

The Riesz bounds from Theorem 2.5 imply

$$
\begin{array} { r } { C _ { \operatorname* { m i n } } ^ { p } \| \pmb { a } \| _ { 2 } ^ { 2 } \leq \| \psi _ { M } ^ { \top } \pmb { a } \| ^ { 2 } = \pmb { a } ^ { \top } \mathbf { r } _ { M } \pmb { a } \leq C _ { \operatorname* { m a x } } ^ { p } \| \pmb { a } \| _ { 2 } ^ { 2 } } \end{array}\tag{87}
$$

for every $\pmb { a } \in \mathbb { R } ^ { d M }$

Remark E.1. The identity $\pmb { a } ^ { \top } \pmb { \Gamma } _ { M } \pmb { a } = \| \pmb { \psi } _ { M } ^ { \top } \pmb { a } \| ^ { 2 }$ is crucial but its proof is straightforward, indeed one has

$$
\begin{array} { r } { \pmb { a } ^ { \top } \Gamma _ { M } \pmb { a } = \pmb { a } ^ { \top } \mathbb { E } _ { p } \left[ \psi _ { M } ( \mathbf { X } ) \psi _ { M } ( \mathbf { X } ) ^ { \top } \right] \pmb { a } , } \end{array}\tag{88}
$$

$$
\mathbf { \tau } = \mathbb { E } _ { p } \left[ \pmb { a } ^ { \top } \psi _ { M } ( \mathbf { X } ) \psi _ { M } ( \mathbf { X } ) ^ { \top } \pmb { a } \right] ,\tag{89}
$$

$$
\mathbf { \Sigma } = \mathbb { E } _ { p } \left[ \left( \psi _ { M } ( \mathbf { X } ) ^ { \top } \pmb { a } \right) ^ { 2 } \right] ,\tag{90}
$$

$$
\mathbf { \Psi } = \int ( \psi _ { M } ^ { \top } \pmb { a } ) ^ { 2 } p d \lambda _ { d } ,\tag{91}
$$

$$
\mathbf { \Psi } = \| \psi _ { M } ^ { \top } \pmb { a } \| ^ { 2 } .\tag{92}
$$

In particular, applying the upper Riesz bound to the tail of the expansion and using (84), we obtain

$$
\| r _ { M } \| ^ { 2 } \leq C _ { \operatorname* { m a x } } ^ { p } \sum _ { j = 1 } ^ { d } \sum _ { m > M } \left( \theta _ { m } ^ { ( j ) } \right) ^ { 2 } ,\tag{93}
$$

$$
\leq C _ { \mathrm { m a x } } ^ { p } M ^ { - 2 \beta } \sum _ { j = 1 } ^ { d } \sum _ { m > M } m ^ { 2 \beta } { \left( \theta _ { m } ^ { ( j ) } \right) } ^ { 2 } ,\tag{94}
$$

$$
\leq C _ { \mathrm { m a x } } ^ { p } d R ^ { 2 } M ^ { - 2 \beta } .\tag{95}
$$

Similarly, since $m ^ { 2 \beta } \geq 1$

$$
\| f \| ^ { 2 } \leq C _ { \operatorname* { m a x } } ^ { p } \sum _ { j = 1 } ^ { d } \sum _ { m \geq 1 } \left( \theta _ { m } ^ { ( j ) } \right) ^ { 2 } \leq C _ { \operatorname* { m a x } } ^ { p } d R ^ { 2 } .\tag{96}
$$

## E.2 Thresholded least squares and the error decomposition

Recall the acceptance event and the empirical Gram matrix

$$
\zeta _ { n } = \left\{ \lambda _ { \operatorname* { m i n } } ( \widehat { \mathbf { T } } _ { M } ) \geq C _ { \operatorname* { m i n } } ^ { p } / 2 \right\} , \qquad \widehat { \mathbf { T } } _ { M } = \frac 1 n \Psi _ { M } ^ { \top } \Psi _ { M } .\tag{97}
$$

For clarity, recall the definition of the coeficient estimator

$$
\widehat { \pmb { \theta } } _ { M } = \left\{ \begin{array} { l l } { ( \pmb { \Psi } _ { M } ^ { \top } \pmb { \Psi } _ { M } ) ^ { - 1 } \pmb { \Psi } _ { M } ^ { \top } \pmb { Y } , } & { \mathrm { o n ~ } \zeta _ { n } , } \\ { \pmb { 0 } , } & { \mathrm { o n ~ } \zeta _ { n } ^ { c } . } \end{array} \right.\tag{98}
$$

Thus no inverse is evaluated on the rejection event. On $\zeta _ { n } .$ the design matrix has full column rank, which in particular requires $d M \leq n$ . Finally, recall that the thresholded estimator $\widehat { f } _ { M }$ is defined as follows

$$
\widehat { f } _ { M } ( \cdot ) = \psi _ { M } ( \cdot ) ^ { \top } \widehat { \pmb { \theta } } _ { M }\tag{99}
$$

We introduce the random matrix

$$
\begin{array} { r } { B _ { M } : = \left\{ \begin{array} { l l } { ( \Psi _ { M } ^ { \top } \Psi _ { M } ) ^ { - 1 } \Psi _ { M } ^ { \top } , } & { \mathrm { o n ~ } \zeta _ { n } , } \\ { { \bf 0 } , } & { \mathrm { o n ~ } \zeta _ { n } ^ { c } , } \end{array} \right. } \end{array}\tag{100}
$$

and the vectors

$$
\pmb { r } _ { M } : = \big ( r _ { M } ( \mathbf { X } _ { 1 } ) , \ldots , r _ { M } ( \mathbf { X } _ { n } ) \big ) ^ { \top } , \qquad \pmb { \xi } : = ( \xi _ { 1 } , \ldots , \xi _ { n } ) ^ { \top } .\tag{101}
$$

The regression model can then be written as

$$
Y : = \Psi _ { M } \pmb { \theta } _ { M } + \pmb { r } _ { M } + \pmb { \xi } .\tag{102}
$$

Consequently, on $\zeta _ { n }$ , we have

$$
\widehat { \pmb { \theta } } _ { M } = \pmb { \theta } _ { M } + \pmb { B } _ { M } \pmb { r } _ { M } + \pmb { B } _ { M } \pmb { \xi }\tag{103}
$$

which leads to the following decomposition of $f - \widehat { f } _ { M }$

$$
f - \widehat { f } _ { M } = r _ { M } - \psi _ { M } ^ { \top } B _ { M } \pmb { r } _ { M } - \psi _ { M } ^ { \top } B _ { M } \pmb { \xi } .\tag{104}
$$

Let denote $\mathcal { D } _ { n }$ the dataset $\mathbf { X } _ { 1 } , \ldots , \mathbf { X } _ { n }$ . The event $\zeta _ { n } ,$ the matrix $\pmb { B } _ { M }$ , and the vector $\mathbf { \nabla } _ { \mathbf { \pmb { r } } _ { M } }$ are measurable with respect to $\mathcal { D } _ { n }$ . Moreover,

$$
\mathbb { E } _ { p , f } [ \pmb { \xi } | \mathcal { D } _ { n } ] = \mathbf { 0 } , \qquad \mathbb { E } _ { p , f } [ \pmb { \xi } \pmb { \xi } ^ { \top } | \mathcal { D } _ { n } ] = \sigma ^ { 2 } I _ { n } .
$$

The cross term involving the noise therefore has conditional expectation zero. It follows the standard bias-variance decomposition

$$
\begin{array} { r } { \bigg \lvert \mathbb { E } _ { p , f } \Big [ \big \lVert f - \widehat { f } _ { M } \big \rVert ^ { 2 } \mathbf { 1 } _ { \zeta _ { n } } \ \big \lvert \mathcal { D } _ { n } \Big ] = \left. r _ { M } - \psi _ { M } ^ { \top } B _ { M } r _ { M } \right. ^ { 2 } \mathbf { 1 } _ { \zeta _ { n } } + \mathbb { E } _ { p , f } \left[ \left. \psi _ { M } ^ { \top } B _ { M } \xi \right. ^ { 2 } \big \lvert \mathcal { D } _ { n } \right] \mathbf { 1 } _ { \zeta _ { n } } . } \end{array}\tag{105}
$$

Recall that here, $\psi _ { M }$ is a function from $\mathbb { R } ^ { d }$ to $\mathbb { R } ^ { d M }$

On $\zeta _ { n } ^ { c }$ , the estimator is zero, so

$$
\begin{array} { r } { \boxed { \mathbb { E } _ { p , f } \Big [ \| f - \widehat { f } _ { M } \| ^ { 2 } \mathbf { 1 } _ { \zeta _ { n } ^ { c } } \Big ] = \| f \| ^ { 2 } \mathbb { P } _ { p } ( \zeta _ { n } ^ { c } ) . } } \end{array}\tag{106}
$$

We next bound the two terms in (105) and the rejection probability.

## E.3 Bounding the squared bias

On $\zeta _ { n } ,$ we have $\Psi _ { M } ^ { \top } \Psi _ { M } \succeq n C _ { \operatorname* { m i n } } ^ { p } / 2$ , which implies

$$
B _ { M } \pmb { B } _ { M } ^ { \top } = ( \pmb { \Psi } _ { M } ^ { \top } \pmb { \Psi } _ { M } ) ^ { - 1 } \preceq \frac { 2 } { n C _ { \operatorname* { m i n } } ^ { p } } \pmb { I } _ { d M } .\tag{107}
$$

Hence

$$
\| B _ { M } \| _ { \mathrm { o p } } ^ { 2 } \leq \frac { 2 } { n C _ { \mathrm { m i n } } ^ { p } } \qquad \mathrm { o n } ~ \zeta _ { n } .\tag{108}
$$

Using $\| u - v \| ^ { 2 } \leq 2 \| u \| ^ { 2 } + 2 \| v \| ^ { 2 }$ , (87), and (108), we obtain, on $\zeta _ { n }$

$$
\begin{array} { r l r } & { } & { \left\| \boldsymbol { r } _ { M } - \boldsymbol { \psi } _ { M } ^ { \top } \boldsymbol { B } _ { M } \boldsymbol { r } _ { M } \right\| ^ { 2 } \leq 2 \| \boldsymbol { r } _ { M } \| ^ { 2 } + 2 C _ { \operatorname* { m a x } } ^ { p } \| \boldsymbol { B } _ { M } \boldsymbol { r } _ { M } \| _ { 2 } ^ { 2 } } \\ & { } & { \qquad \leq 2 \| \boldsymbol { r } _ { M } \| ^ { 2 } + \frac { 4 C _ { \operatorname* { m a x } } ^ { p } } { n C _ { \operatorname* { m i n } } ^ { p } } \| \boldsymbol { r } _ { M } \| _ { 2 } ^ { 2 } . \qquad } \end{array}\tag{109}
$$

Since $r _ { M }$ is a fixed function,

$$
\mathbb { E } _ { p , f } \left[ \| r _ { M } \| _ { 2 } ^ { 2 } \right] = \sum _ { i = 1 } ^ { n } \mathbb { E } _ { p } [ r _ { M } ( \mathbf { X } _ { i } ) ^ { 2 } ] = n \| r _ { M } \| ^ { 2 } .\tag{110}
$$

Multiplying (109) by $\mathbf { 1 } _ { \zeta _ { n } } ,$ , taking expectations, and using $\mathbf { 1 } _ { \zeta _ { n } } \leq 1$ yields

$$
| \mathbb { E } _ { p , f } [ \| r _ { M } - \psi _ { M } ^ { \top } B _ { M } r _ { M } \| ^ { 2 } \mathbf { 1 } _ { \mathcal { O } _ { n } } ] \leq ( 2 + \frac { 4 C _ { \operatorname* { m a x } } ^ { p } } { C _ { \operatorname* { m i n } } ^ { p } } ) \| r _ { M } \| ^ { 2 } \leq C _ { \operatorname* { m a x } } ^ { p } ( 2 + \frac { 4 C _ { \operatorname* { m a x } } ^ { p } } { C _ { \operatorname* { m i n } } ^ { p } } ) d R ^ { 2 } M ^ { - 2 \beta } .\tag{111}
$$

## E.4 Bounding the conditional variance

Conditional on $\mathcal { D } _ { n } .$ , the matrix $\pmb { B } _ { M }$ is fixed. By the definition of the population Gram matrix,

$$
\begin{array} { r l } & { \mathbb { E } _ { p , f } \left[ \left\| \psi _ { M } ^ { \top } B _ { M } \pmb { \xi } \right\| ^ { 2 } \bigm | \mathcal { D } _ { n } \right] } \\ & { \quad = \mathbb { E } _ { p , f } \left[ \pmb { \xi } ^ { \top } \pmb { B } _ { M } ^ { \top } \pmb { \Gamma } _ { M } \pmb { B } _ { M } \pmb { \xi } \bigm | \mathcal { D } _ { n } \right] } \\ & { \quad = \mathrm { T r } \big ( \pmb { B } _ { M } ^ { \top } \pmb { \Gamma } _ { M } \pmb { B } _ { M } \mathbb { E } _ { p , f } \left[ \pmb { \xi } \pmb { \xi } ^ { \top } \bigm | \mathcal { D } _ { n } \right] \big ) } \\ & { \quad = \sigma ^ { 2 } \mathrm { T r } \big ( \pmb { \Gamma } _ { M } \pmb { B } _ { M } \pmb { B } _ { M } ^ { \top } \bigm ) . } \end{array}\tag{112}
$$

On $\zeta _ { n } .$ this is

$$
\frac { \sigma ^ { 2 } } { n } \mathrm { T r } \Big ( \Gamma _ { M } \widehat { \Gamma } _ { M } ^ { - 1 } \Big ) .
$$

Because

$$
{ \bf \widehat { \mathbf { F } } } _ { M } \preceq C _ { \mathrm { m a x } } ^ { p } { \cal I } _ { d M } , \qquad { \widehat { \mathbf { F } } } _ { M } ^ { - 1 } \preceq { \frac { 2 } { C _ { \mathrm { m i n } } ^ { p } } } { \cal I } _ { d M } \quad \mathrm { o n } ~ \zeta _ { n } ,
$$

we obtain

$$
\mathbb { E } _ { p , f } \left[ \| \psi _ { M } ^ { \top } B _ { M } \pmb { \xi } \| ^ { 2 } | \mathcal { D } _ { n } \right] \le \frac { 2 \sigma ^ { 2 } C _ { \operatorname* { m a x } } ^ { p } } { C _ { \operatorname* { m i n } } ^ { p } } \frac { d M } { n } \qquad \mathrm { o n } \ \zeta _ { n } .
$$

Here we used the fact that $\operatorname { T r } ( U V ) \geq 0$ for positive semidefinite matrices $U , V ;$ the matrices need not commute. Since $\zeta _ { n }$ is $\mathcal { D } _ { n }$ -measurable, the tower property gives

$$
\left| \mathbb { E } _ { p , f } \left[ \| \psi _ { M } ^ { \top } B _ { M } \pmb { \xi } \| ^ { 2 } \mathbf { 1 } _ { \zeta _ { n } } \right] \right| \overset { } { = } \frac { 2 \sigma ^ { 2 } C _ { \operatorname* { m a x } } ^ { p } } { C _ { \operatorname* { m i n } } ^ { p } } \frac { d M } { n } .\tag{113}
$$

## E.5 Concentration of the empirical Gram matrix

We use the lower-tail form of the matrix Chernof inequality (see Theorem D.6). To apply this result, write

$$
{ \widehat { \bf \Gamma } } _ { M } = \sum _ { i = 1 } ^ { n } S _ { i } , \qquad S _ { i } = { \frac { 1 } { n } } \psi _ { M } ( { \bf X } _ { i } ) \psi _ { M } ( { \bf X } _ { i } ) ^ { \top } .\tag{114}
$$

These matrices are independent, symmetric, and positive semidefinite. Since a rank-one matrix $\mathbf { \Delta } _ { v v } \top$ has operator norm $\| \pmb { v } \| _ { 2 } ^ { 2 }$ , we have

$$
\lambda _ { \operatorname* { m a x } } ( { \cal S } _ { i } ) = \frac { 1 } { n } \| \psi _ { M } ( { \bf X } _ { i } ) \| _ { 2 } ^ { 2 } .\tag{115}
$$

The lower bound $p _ { j } \ge p _ { \mathrm { m i n } }$ and the uniform bound on the trigonometric functions imply

$$
\begin{array} { l } { \| \psi _ { M } ( \mathbf { x } ) \| _ { 2 } ^ { 2 } = \displaystyle \sum _ { j = 1 } ^ { d } \sum _ { m = 1 } ^ { M } \frac { | \phi _ { m } ^ { ( j ) } ( x _ { j } ) | ^ { 2 } } { p _ { j } ( x _ { j } ) ^ { 2 } } } \\ { \leq \displaystyle \frac { 2 d M } { p _ { \mathrm { m i n } } ^ { 2 } } . } \end{array}\tag{116}
$$

We may therefore take $K = d M$ and $\begin{array} { r } { L = \frac { 2 d M } { n p _ { \mathrm { m i n } } ^ { 2 } } } \end{array}$ . Moreover,

$$
\sum _ { i = 1 } ^ { n } \mathbb { E } _ { p } [ { \pmb S } _ { i } ] = \mathbf { \Gamma } _ { M } , \qquad \mu _ { \mathrm { m i n } } = \lambda _ { \mathrm { m i n } } ( \mathbf { \Gamma } _ { M } ) \geq C _ { \mathrm { m i n } } ^ { p } .\tag{117}
$$

Theorem D.6 gives

$$
\mathbb { P } _ { p } \Big ( \lambda _ { \operatorname* { m i n } } ( \widehat { \mathbf { T } } _ { M } ) \leq \frac { \mu _ { \operatorname* { m i n } } } { 2 } \Big ) \leq d M \exp \left( - \frac { 1 - \log 2 } { 2 } \frac { \mu _ { \operatorname* { m i n } } } { L } \right) .\tag{118}
$$

Since $C _ { \mathrm { m i n } } ^ { p } \leq \mu _ { \mathrm { m i n } }$ , we have the following bound on the probability

$$
\begin{array} { r } { \boxed { \mathbb { P } _ { p } ( \zeta _ { n } ^ { c } ) \leq d M \exp \left( - a _ { p } \frac { n } { d M } \right) , \qquad a _ { p } : = \frac { p _ { \operatorname* { m i n } } ^ { 2 } C _ { \operatorname* { m i n } } ^ { p } ( 1 - \log 2 ) } { 4 } . } } \end{array}\tag{119}
$$

## E.6 Finite-sample risk bound

Combining (106), (96), and (119) yields

$$
\mathbb { E } _ { p , f } \left[ \| f - \widehat { f } _ { M } \| ^ { 2 } \mathbf { 1 } _ { \zeta _ { n } ^ { c } } \right] \leq C _ { \operatorname* { m a x } } ^ { p } d R ^ { 2 } \left[ d M \exp \left( - a _ { p } \frac { n } { d M } \right) \right] .\tag{120}
$$

On the acceptance event, we use (105), (111), and (113). Adding the two contributions gives

$$
\begin{array} { r l } { \displaystyle \operatorname* { s u p } _ { f \in \mathcal { F } } \mathbb { E } _ { p , f } \Big [ \| f - \widehat { f } _ { M } \| ^ { 2 } \Big ] \leq C _ { \operatorname* { m a x } } ^ { p } \left( 2 + \frac { 4 C _ { \operatorname* { m a x } } ^ { p } } { C _ { \operatorname* { m i n } } ^ { p } } \right) d R ^ { 2 } M ^ { - 2 \beta } } & { } \\ { + \frac { 2 \sigma ^ { 2 } C _ { \operatorname* { m a x } } ^ { p } } { C _ { \operatorname* { m i n } } ^ { p } } \frac { d M } { n } } & { } \\ { + C _ { \operatorname* { m a x } } ^ { p } d R ^ { 2 } \left[ d M \exp \left( - a _ { p } \frac { n } { d M } \right) \right] . } \end{array}\tag{121}
$$

All bounds are uniform over $f \in { \mathcal { F } }$ . The three terms in (121) correspond to approximation, variance, and rejection and there is no regularization bias.

## E.7 Obtaining minimax upper bound

We typically choose $M = \left\lceil n ^ { 1 / ( 2 \beta + 1 ) } \right\rceil$ to match the bias and the variance. Thus the first two terms in (121) are bounded by a constant multiple of $\displaystyle { \dot { d n } } ^ { - \frac { 2 \beta } { 2 \beta + 1 } }$ . Recall that under Assumption 2.13, one has

$$
d \log n = o \Big ( n ^ { 2 \beta / ( 2 \beta + 1 ) } \Big ) .\tag{122}
$$

Under this condition,

$$
{ \frac { n } { d M \log n } } \longrightarrow \infty ,\tag{123}
$$

and $d M \leq n$ for all suficiently large n. It follows that

$$
d M \exp \Bigl ( - a _ { p } \frac { n } { d M } \Bigr ) = o \Bigl ( n ^ { - \frac { 2 \beta } { 2 \beta + 1 } } \Bigr ) ,\tag{124}
$$

so the rejection contribution is negligible relative to the target risk.

Finally, the estimator is measurable with respect to the sample and uses only the known density and the specified class parameters. It does not depend on the unknown regression function. By the definition of the minimax risk,

$$
\mathfrak { M } ( n , d , \mathcal { F } ) = \operatorname* { i n f } _ { \widehat { f } } \operatorname* { s u p } _ { f \in \mathcal { F } } \mathbb { E } _ { p , f } \left[ \Vert f - \widehat { f } \Vert ^ { 2 } \right]\tag{125}
$$

$$
\leq \operatorname* { s u p } _ { f \in { \mathcal { F } } } { \mathbb { E } } _ { p , f } \left[ \| f - { \widehat { f } } _ { M } \| ^ { 2 } \right]\tag{126}
$$

$$
\lesssim d n ^ { - \frac { 2 \beta } { 2 \beta + 1 } } .\tag{127}
$$

Using the bounds on $C _ { \mathrm { m i n } } ^ { p }$ and $C _ { \mathrm { m a x } } ^ { p }$ supplied by Theorem 2.5, the implicit constant depends only on $\beta , R , \sigma , p _ { \mathrm { m i n } } , p _ { \mathrm { m a x } }$ , and not on $n , d ,$ or f.

## F PROOF OF THEOREM 3.3

Throughout this section, the design density $p$ is fixed and known. We write P for the corresponding distribution on $[ 0 , 1 ] ^ { d }$ , and $\mathcal { F }$ for the associated additive function class. We assume $\beta > 0 , R > 0$ , and $\sigma > 0$

We prove that

$$
\mathfrak { M } ( n , d , \mathcal { F } ) : = \operatorname* { i n f } _ { \widehat { f } } \operatorname* { s u p } _ { f \in \mathcal { F } } \mathbb { E } _ { p , f } \left[ \Vert \widehat { f } - f \Vert ^ { 2 } \right] \gtrsim d n ^ { - \frac { 2 \beta } { 2 \beta + 1 } } .
$$

The infimum includes all estimators based on the observations and the known density $p .$

Our argument uses the Riesz inequalities from Theorem 2.5. The constants are independent of the number of retained frequencies. Their uniform control in terms of $p _ { \mathrm { m i n } }$ and $p _ { \mathrm { m a x } }$ will give the claimed dependence of the final constant.

## F.1 Construction of a finite subfamily

For $n \geq 1$ , set $M = \left\lceil n ^ { \frac { 1 } { 2 \beta + 1 } } \right\rceil$ and $a _ { M } = a _ { 0 } M ^ { - \beta - 1 / 2 }$ where

$$
a _ { 0 } ^ { 2 } = \operatorname* { m i n } \left\{ 2 ^ { - 2 \beta } R ^ { 2 } , \frac { \sigma ^ { 2 } } { 1 6 C _ { \operatorname* { m a x } } ^ { p } } \right\} .\tag{128}
$$

In particular, $a _ { 0 } > 0$ and does not depend on n or M. Let define

$$
\begin{array} { r } { \mathcal { Z } _ { M } = \{ ( j , m ) : 1 \leq j \leq d , ~ M + 1 \leq m \leq 2 M \} , \qquad | \mathcal { Z } _ { M } | = d M . } \end{array}\tag{129}
$$

For each sign vector $\omega \in \{ - 1 , + 1 \} ^ { \mathcal { T } _ { M } }$ , define

$$
f _ { \omega } ( \mathbf { x } ) = \sum _ { j = 1 } ^ { d } f _ { \omega , j } ( x _ { j } ) , \qquad f _ { \omega , j } ( x _ { j } ) = a _ { M } \sum _ { m = M + 1 } ^ { 2 M } \omega _ { j m } \psi _ { m } ^ { ( j ) } ( x _ { j } ) .\tag{130}
$$

Observe that the reference-basis coeficients of $p _ { j } f _ { \omega , j }$ equal $a _ { M } \omega _ { j m }$ for $M + 1 \leq m \leq 2 M$ and vanish otherwise. Hence

$$
\begin{array} { c } { { \displaystyle { \sum _ { m = M + 1 } ^ { 2 M } m ^ { 2 \beta } ( a _ { M } \omega _ { j m } ) ^ { 2 } \leq M ( 2 M ) ^ { 2 \beta } a _ { M } ^ { 2 } } } } \\ { { { } } } \\ { { = 2 ^ { 2 \beta } a _ { 0 } ^ { 2 } } } \\ { { { } \leq { R } ^ { 2 } . } } \end{array}\tag{131}
$$

It follows that

$$
\left\{ f _ { \omega } : \omega \in \{ - 1 , + 1 \} ^ { \mathcal { T } _ { M } } \right\} \subseteq \mathcal { F } .\tag{132}
$$

For later use, define the following Hamming distance

$$
\rho ( \omega , \omega ^ { \prime } ) : = \sum _ { ( j , m ) \in \mathbb { Z } _ { M } } \mathbf { 1 } _ { \{ \omega _ { j m } \neq \omega _ { j m } ^ { \prime } \} } .\tag{133}
$$

The lower Riesz inequality immediately gives

$$
\| f _ { \omega } - f _ { \omega ^ { \prime } } \| ^ { 2 } \geq 4 C _ { \operatorname* { m i n } } ^ { p } a _ { M } ^ { 2 } \rho ( \omega , \omega ^ { \prime } ) .\tag{134}
$$

## F.2 Controlling neighboring observation laws

Let $Q _ { \omega }$ denote the joint law of the n observations under design distribution P and regression function $f _ { \omega }$ . Write $\mathbb { E } _ { \omega }$ for expectation under this law.

For $( j , m ) \in \mathcal { I } _ { M }$ , let $\omega ^ { ( j , m ) }$ be obtained from ω by reversing only the sign $\omega _ { j m }$ . Then

$$
f _ { \omega } - f _ { \omega ^ { ( j , m ) } } = 2 a _ { M } \omega _ { j m } \psi _ { m } ^ { ( j ) } .\tag{135}
$$

By the upper Riesz inequality,

$$
\begin{array} { r } { \big \| \ b { f _ { \omega } } - \ b { f _ { \omega ^ { ( j , m ) } } } \big \| ^ { 2 } \leq 4 C _ { \operatorname* { m a x } } ^ { p } a _ { M } ^ { 2 } . } \end{array}\tag{136}
$$

The design distribution is identical at every vertex of the hypercube. The Gaussian regression identity in Lemma D.5 therefore yields

$$
\begin{array} { r l } {  { \mathrm { K L } ( Q _ { \omega } \| Q _ { \omega ^ { ( j , m ) } } ) = n \times 0 + \frac { n } { 2 \sigma ^ { 2 } } \| f _ { \omega } - f _ { \omega ^ { ( j , m ) } } \| ^ { 2 } } } \\ & { \leq \frac { 2 n C _ { \operatorname* { m a x } } ^ { p } a _ { M } ^ { 2 } } { \sigma ^ { 2 } } } \\ & { = \frac { 2 C _ { \operatorname* { m a x } } ^ { p } a _ { 0 } ^ { 2 } } { \sigma ^ { 2 } } \frac { n } { M ^ { 2 \beta + 1 } } } \\ & { \leq \frac { 1 } { 8 } . } \end{array}\tag{137}
$$

The last inequality follows from $M ^ { 2 \beta + 1 } \geq n$ and (128).

Pinsker’s inequality (Lemma D.2) now gives

$$
\operatorname* { m a x } _ { \omega } \operatorname* { m a x } _ { ( j , m ) \in \mathbb { Z } _ { M } } \mathrm { T V } \big ( Q _ { \omega } , Q _ { \omega ^ { ( j , m ) } } \big ) \leq \sqrt { \frac { 1 } { 2 } \cdot \frac { 1 } { 8 } } = \frac { 1 } { 4 } .\tag{138}
$$

## F.3 Reduction from prediction loss to Hamming loss

Fix an arbitrary estimator ${ \widehat { f } } .$ If its maximum risk over the finite subfamily is infinite, the desired lower bound is immediate. We may therefore assume that $\widehat { f }$ takes values in $L ^ { 2 } ( P )$ almost surely under every $Q _ { \omega }$ . Consider the finite-dimensional vector subspace

$$
\mathcal { V } _ { M } = \operatorname { s p a n } \left\{ \psi _ { m } ^ { ( j ) } : ( j , m ) \in \mathbb { Z } _ { M } \right\} ,\tag{139}
$$

and let $\Pi _ { M }$ be the orthogonal projection onto $\nu _ { M }$ in $L ^ { 2 } ( P )$ . The lower Riesz bound guarantees linear independence of the functions defining $\nu _ { M }$ . Consequently, there is a unique coeficient vector $\widehat { \pmb { b } } = ( \widehat { b } _ { j m } ) _ { ( j , m ) \in \mathcal { T } _ { M } }$ such that

$$
\Pi _ { M } \widehat { f } = \sum _ { ( j , m ) \in \mathbb { Z } _ { M } } \widehat { b } _ { j m } \psi _ { m } ^ { ( j ) } .\tag{140}
$$

The coeficient map is continuous on $\nu _ { M }$ , so these coeficients are measurable functions of the observations. Define the induced sign estimator by

$$
\widehat { \omega } _ { j m } = \left\{ \begin{array} { l l } { + 1 , } & { \widehat { b } _ { j m } \geq 0 , } \\ { - 1 , } & { \widehat { b } _ { j m } < 0 . } \end{array} \right.\tag{141}
$$

Because $f _ { \omega } \in \mathcal { V } _ { M }$ , orthogonal projection gives

$$
\begin{array} { r l } {  { \Vert \widehat { f } - f _ { \omega } \Vert ^ { 2 } = \Vert \widehat { f } - \Pi _ { M } \widehat { f } \Vert ^ { 2 } + \Vert \Pi _ { M } \widehat { f } - f _ { \omega } \Vert ^ { 2 } } } \\ & { \geq \Vert \Pi _ { M } \widehat { f } - f _ { \omega } \Vert ^ { 2 } } \\ & { \geq C _ { \operatorname* { m i n } } ^ { p } \displaystyle \sum _ { ( j , m ) \in \mathbb { Z } _ { M } } ( \widehat { b } _ { j m } - a _ { M } \omega _ { j m } ) ^ { 2 } . } \end{array}\tag{142}
$$

Now, let suppose that $\widehat { \omega } _ { j m } \neq \omega _ { j m }$ , because $a _ { M } > 0$ , we have

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { \omega _ { j m } = + 1 \implies \widehat { \omega } _ { j m } = - 1 \implies \widehat { b } _ { j m } < 0 \implies \lvert \widehat { b } _ { j m } - a _ { M } \omega _ { j m } \rvert = a _ { M } - \widehat { b } _ { j m } \geq a _ { M } } \\ { \omega _ { j m } = - 1 \implies \widehat { \omega } _ { j m } = + 1 \implies \widehat { b } _ { j m } \geq 0 \implies \lvert \widehat { b } _ { j m } - a _ { M } \omega _ { j m } \rvert = \widehat { b } _ { j m } + a _ { M } \geq a _ { M } } \end{array} \right. } \end{array}\tag{143}
$$

We finally can write

$$
\begin{array} { r } { \left( \widehat { b } _ { j m } - a _ { M } \omega _ { j m } \right) ^ { 2 } \geq a _ { M } ^ { 2 } \mathbf { 1 } _ { \{ \widehat { \omega } _ { j m } \neq \omega _ { j m } \} } , } \end{array}\tag{144}
$$

which leads to the last inequality

$$
\Vert \widehat { f } - f _ { \omega } \Vert ^ { 2 } \geq C _ { \operatorname* { m i n } } ^ { p } a _ { M } ^ { 2 } \rho ( \widehat { \omega } , \omega ) .\tag{145}
$$

## F.4 Application of Assouad’s lemma

By (138), Lemma D.3 applies with $N = d M$ and $\eta = 1 / 4$ . Therefore every measurable sign estimator satisfies

$$
\operatorname* { s u p } _ { \omega } \mathbb { E } _ { \omega } [ \rho ( \widehat { \omega } , \omega ) ] \geq \frac { d M } { 2 } \left( 1 - \frac { 1 } { 4 } \right) = \frac { 3 } { 8 } d M .\tag{146}
$$

Since the finite subfamily is contained in ${ \mathcal F } ,$ (142) implies

$$
\begin{array} { r l } & { \underset { f \in \mathcal { F } } { \operatorname* { s u p } } \mathbb { E } _ { p , f } \Big [ \| \widehat { f } - f \| ^ { 2 } \Big ] \geq \underset { \omega } { \operatorname* { s u p } } \mathbb { E } _ { \omega } \Big [ \| \widehat { f } - f _ { \omega } \| ^ { 2 } \Big ] } \\ & { \qquad \quad \geq C _ { \operatorname* { m i n } } ^ { p } a _ { M } ^ { 2 } \underset { \omega } { \operatorname* { s u p } } \mathbb { E } _ { \omega } [ \rho ( \widehat { \omega } , \omega ) ] } \\ & { \qquad \quad \geq \frac { 3 } { 8 } C _ { \operatorname* { m i n } } ^ { p } a _ { M } ^ { 2 } d M } \\ & { \qquad \quad \geq \frac { 3 } { 8 } C _ { \operatorname* { m i n } } ^ { p } a _ { 0 } ^ { 2 } d M ^ { - 2 \beta } . } \end{array}\tag{147}
$$

This holds for every estimator ${ \widehat { f } } .$ Taking the infimum over estimators yields

$$
\mathfrak { M } ( n , d , \mathcal { F } ) \ge \frac { 3 } { 8 } C _ { \mathrm { m i n } } ^ { p } a _ { 0 } ^ { 2 } d M ^ { - 2 \beta } .\tag{148}
$$

Finally, by recalling that for $n \geq 1 , M = \left\lceil n ^ { \frac { 1 } { 2 \beta + 1 } } \right\rceil$ , we have the desired result

$$
\mathfrak { M } ( n , d , \mathcal { F } ) \gtrsim d n ^ { - \frac { 2 \beta } { 2 \beta + 1 } } ,\tag{149}
$$

where the positive underlying constant is equal to $\frac { 3 C _ { \mathrm { m i n } } ^ { p } } { 2 ^ { 2 \beta + 3 } }$ min $\begin{array} { r } { \left\{ 2 ^ { - 2 \beta } R ^ { 2 } , \frac { \sigma ^ { 2 } } { 1 6 C _ { \mathrm { m a x } } ^ { p } } \right\} } \end{array}$ and does not depend on n. Remark F.1. The lower-bound argument itself imposes no restriction on the growth of $d ,$ provided the Riesz constants remain uniformly controlled. The dimension condition used for the upper bound is needed to establish the matching upper rate.

## G PROOF OF THEOREM 4.4

## G.1 Notations

We recall that, throughout the class $\mathcal { C } _ { \beta , \gamma }$ , the joint densities satisfy

$$
0 < \kappa _ { \mathrm { m i n } } \leq p \leq \kappa _ { \mathrm { m a x } } < \infty .\tag{150}
$$

Where constants $\kappa _ { \mathrm { m i n } }$ and $\kappa _ { \mathrm { m a x } }$ are known parameters of the class and satisfy

$$
\kappa _ { \operatorname* { m i n } } < 1 < \kappa _ { \operatorname* { m a x } } .\tag{151}
$$

As a direct consequence, the class is nonempty, since the uniform distribution belongs to the admissible distributions. Integrating (150) over all but one coordinate gives $\kappa _ { \mathrm { m i n } } \le p _ { j } \le \kappa _ { \mathrm { m a x } }$ for every $j .$ No independence between the coordinates of a covariate is assumed.

Fix a pair $( p , f ) \in { \mathcal { C } } _ { \beta , \gamma }$ . The regression function is additive and centered componentwise. We use the representations

$$
f _ { j } = \sum _ { m \ge 1 } \theta _ { m } ^ { ( j ) } \psi _ { m } ^ { ( j ) } , \qquad g _ { j } : = p _ { j } f _ { j } = \sum _ { m \ge 1 } \theta _ { m } ^ { ( j ) } \phi _ { m } , \qquad 1 \le j \le d .\tag{152}
$$

The Sobolev condition implies

$$
\sum _ { m \geq 1 } m ^ { 2 \beta } \big ( \theta _ { m } ^ { ( j ) } \big ) ^ { 2 } \leq R ^ { 2 } , \qquad 1 \leq j \leq d ,\tag{153}
$$

where $\beta > 0$ and $R > 0$ are fixed. Thus Sobolev regularity is understood in the periodic Fourier sense specified by (153). The expansions in (152) hold in $L ^ { 2 } ( \lambda )$ for $g _ { j }$ and in $L ^ { 2 } ( p _ { j } d \lambda )$ for $f _ { j }$ . No assumption $\beta > 1 / 2$ is made.

The marginal densities belong are all $( \gamma , L ) { \mathrm { - H } } { \mathrm { { \ddot { o l d e r } } } }$ (see Definition 2.10) and uniformly bounded between the class parameters $\kappa _ { \mathrm { m i n } }$ and $\kappa _ { \mathrm { m a x } }$ . The notation $\mathcal { C } _ { \beta , \gamma }$ suppresses the fixed parameters $R , L , \kappa _ { \operatorname* { m i n } } , \kappa _ { \operatorname* { m a x } }$ . Both $p$ and $f$ may vary in this class, and d may increase with n at the rate of Assumption 2.13.

We observe an i.i.d. sample $( \mathbf { X } _ { 1 } , Y _ { 1 } ) , \ldots , ( \mathbf { X } _ { n } , Y _ { n } )$ satisfying

$$
Y _ { i } = f ( \mathbf { X } _ { i } ) + \xi _ { i } , \qquad \mathbf { X } _ { i } \sim P , \qquad \xi _ { i } \sim { \mathcal { N } } ( 0 , \sigma ^ { 2 } ) , \qquad 1 \le i \le n ,\tag{154}
$$

where the noises are independent of all covariates and $\sigma ^ { 2 } > 0$ is fixed. For $n \geq 4 ,$ let

$$
n _ { 0 } : = \lfloor n / 2 \rfloor , \qquad n _ { 1 } : = n - n _ { 0 } .\tag{155}
$$

We split the covariates into

$$
\mathcal { D } _ { n } ^ { 0 } : = ( \mathbf { X } _ { 1 } ^ { 0 } , \ldots , \mathbf { X } _ { n _ { 0 } } ^ { 0 } ) = ( \mathbf { X } _ { 1 } , \ldots , \mathbf { X } _ { n _ { 0 } } ) ,\tag{156}
$$

$$
\mathscr { D } _ { n } ^ { 1 } : = ( \mathbf { X } _ { 1 } ^ { 1 } , \ldots , \mathbf { X } _ { n _ { 1 } } ^ { 1 } ) = ( \mathbf { X } _ { n _ { 0 } + 1 } , \ldots , \mathbf { X } _ { n } ) .\tag{157}
$$

These samples are independent and have sizes of order n. The first sample is used to estimate the marginal densities; the second, together with its responses, is used to estimate $f .$ We write $Y _ { i } ^ { 1 } : = Y _ { n _ { 0 } + i }$ and $\xi _ { i } ^ { 1 } : = \xi _ { n _ { 0 } + i }$ Conditioning on a dataset means conditioning on the sigma-field generated by its covariates. In particular, $\mathcal { D } _ { n } : = ( \mathcal { D } _ { n } ^ { 0 } , \mathcal { D } _ { n } ^ { 1 } )$ contains the two designs, but not the responses.

Let $\widehat { p } _ { 1 } , \ldots , \widehat { p } _ { d }$ be jointly measurable estimators constructed from $\mathcal { D } _ { n } ^ { 0 }$ alone. We will require

$$
\kappa _ { \operatorname* { m i n } } \leq \widehat { p } _ { j } ( x ) \leq \kappa _ { \operatorname* { m a x } } \quad \mathrm { f o r ~ e v e r y ~ } x \in [ 0 , 1 ] , ~ 1 \leq j \leq d , \quad \mathrm { a l m o s t ~ s u r e l y } .\tag{158}
$$

The marginal-density estimators constructed satisfy

$$
\operatorname* { m a x } _ { 1 \leq j \leq d } \operatorname* { s u p } _ { x \in [ 0 , 1 ] } \mathbb { E } \Big [ \big ( \widehat { p } _ { j } ( x ) - p _ { j } ( x ) \big ) ^ { 2 } \Big ] \leq \rho _ { n _ { 0 } } ,\tag{159}
$$

where $\rho _ { n _ { 0 } }$ is a deterministic bound depending only on the sample size $n _ { 0 }$ and the fixed parameters of the class, and not on the particular pair $( p , f )$ or the dimension d. Subsection G.10 constructs estimators satisfying these conditions with $\rho _ { n _ { 0 } } \leq C _ { \gamma , L , \kappa _ { \mathrm { m a x } } } n _ { 0 } ^ { - 2 \gamma / ( 2 \gamma + 1 ) }$ . The estimators of diferent marginal densities may be dependent.

For these estimators and an integer $M \geq 1$ , define the estimated dictionary elements centered with respect to Lebesgue measure:

$$
\widehat { \psi } _ { m } ^ { ( j ) } ( x _ { j } ) : = \frac { \phi _ { m } ( x _ { j } ) } { \widehat { p } _ { j } ( x _ { j } ) } - \int _ { 0 } ^ { 1 } \frac { \phi _ { m } ( t ) } { \widehat { p } _ { j } ( t ) } d t , \qquad 1 \leq j \leq d , \quad 1 \leq m \leq M .\tag{160}
$$

For $\mathbf { x } \in [ 0 , 1 ] ^ { d }$ , let

$$
\begin{array} { r } { \widehat { \psi } ( \mathbf { x } ) : = \left( 1 , \widehat { \psi } _ { 1 } ^ { ( 1 ) } ( x _ { 1 } ) , \ldots , \widehat { \psi } _ { M } ^ { ( 1 ) } ( x _ { 1 } ) , \ldots , \widehat { \psi } _ { 1 } ^ { ( d ) } ( x _ { d } ) , \ldots , \widehat { \psi } _ { M } ^ { ( d ) } ( x _ { d } ) \right) ^ { \top } \in \mathbb { R } ^ { K } , \qquad K : = 1 + d M . } \end{array}\tag{161}
$$

The order of all coeficient vectors below matches this order of the dictionary elements. The design matrix $\widehat { \Psi } \in \mathbb { R } ^ { n _ { 1 } \times K }$ has ith row $\widehat { \psi } ( \mathbf { X } _ { i } ^ { 1 } ) ^ { \top }$ . Its empirical Gram matrix and conditional population counterpart are

$$
\widehat { \mathbf { r } } : = \frac { 1 } { n _ { 1 } } \widehat { \boldsymbol { \Psi } } ^ { \top } \widehat { \boldsymbol { \Psi } } , \qquad \mathbf { r } : = \mathbb { E } \Big [ \widehat { \psi } ( \mathbf { X } ) \widehat { \psi } ( \mathbf { X } ) ^ { \top } \mid \mathcal { D } _ { n } ^ { 0 } \Big ] ,\tag{162}
$$

where $\mathbf { X } \sim P$ is independent of $\mathcal { D } _ { n } ^ { 0 }$

For simplicity, we omit the dependence in M all previous notations.

We define 4 constants which depend only of $\kappa _ { \mathrm { m i n } }$ and $\kappa _ { \mathrm { m a x } } .$

$$
\begin{array} { r l r } { C _ { \operatorname* { m i n } } : = \frac { \kappa _ { \operatorname* { m i n } } } { \kappa _ { \operatorname* { m a x } } ^ { 2 } } , } & { C _ { \operatorname* { m a x } } : = \frac { \kappa _ { \operatorname* { m a x } } } { \kappa _ { \operatorname* { m i n } } ^ { 2 } } , } & \\ { c _ { 1 } : = \frac { C _ { \operatorname* { m i n } } } { 2 } , } & { c _ { 2 } : = \frac { ( 1 - \log 2 ) \kappa _ { \operatorname* { m i n } } ^ { 3 } } { 1 6 \kappa _ { \operatorname* { m a x } } ^ { 2 } } > 0 . } \end{array}\tag{163}
$$

For a matrix, $\| \cdot \| _ { \mathrm { o p } }$ denotes its matrix operator norm, and ⪯ denotes the Loewner order on symmetric matrices.

## G.2 The estimator

Fix $M \geq 1$ and marginal density estimators satisfying (158)–(159). Define the event

$$
\zeta _ { n } : = \{ \lambda _ { \operatorname* { m i n } } ( { \widehat { \Gamma } } ) \geq c _ { 1 } \} ,\tag{164}
$$

where $\lambda _ { \mathrm { m i n } }$ denotes the smallest eigenvalue. Let

$$
\begin{array} { r } { Y ^ { 1 } : = ( Y _ { 1 } ^ { 1 } , \ldots , Y _ { n _ { 1 } } ^ { 1 } ) ^ { \top } . } \end{array}\tag{165}
$$

The thresholded least-squares estimator is

$$
\begin{array} { r } { \widehat { f } _ { M } ( \mathbf { x } ) : = \left\{ \begin{array} { l l } { \widehat { \psi } ( \mathbf { x } ) ^ { \top } ( \widehat { \pmb { \Psi } } ^ { \top } \widehat { \pmb { \Psi } } ) ^ { - 1 } \widehat { \pmb { \Psi } } ^ { \top } \pmb { Y } ^ { 1 } , } & { \mathrm { o n ~ } \zeta _ { n } , } \\ { 0 , } & { \mathrm { o n ~ } \zeta _ { n } ^ { c } . } \end{array} \right. } \end{array}\tag{166}
$$

Remark G.1 (Well-definedness and the fitted constant). On $\zeta _ { n }$ ,

$$
\widehat { \Psi } ^ { \top } \widehat { \Psi } = n _ { 1 } \widehat { \mathbf { I } } \succeq n _ { 1 } c _ { 1 } I _ { K } ,\tag{167}
$$

so the inverse in the estimator exists. On the complement, no inverse is evaluated. In particular, $\zeta _ { n }$ is empty if $K > n _ { 1 }$ . The estimator uses only the observed data and the known density bounds; it has no ridge parameter.

The fitted space is

$$
\operatorname { s p a n } \left( 1 , \left\{ \frac { \phi _ { m } ( x _ { j } ) } { \widehat { p _ { j } } ( x _ { j } ) } : 1 \leq j \leq d , \ 1 \leq m \leq M \right\} \right) .\tag{168}
$$

The constant allows the estimator to account for the means removed in (160). It does not introduce a nonzero intercept into the true model (2). Centering under P does not, in general, imply centering under $\lambda _ { d } .$ The integrals in (160) are known functionals of the estimated densities. The theoretical estimator uses these exact integrals; no closed-form antiderivative is assumed.

## G.3 Proof of Lemma 4.2

For $M \geq 1$ , introduce the truncated numerators and the Lebesgue mean

$$
h _ { j } : = \sum _ { m = 1 } ^ { M } \theta _ { m } ^ { ( j ) } \phi _ { m } , \qquad \mu _ { f } : = \int _ { [ 0 , 1 ] ^ { d } } f d \lambda _ { d } .\tag{169}
$$

Define the comparison function

$$
t _ { M } ( \mathbf { x } ) : = \mu _ { f } + \sum _ { j = 1 } ^ { d } \left\{ \frac { h _ { j } ( x _ { j } ) } { \widehat { p } _ { j } ( x _ { j } ) } - \int _ { 0 } ^ { 1 } \frac { h _ { j } ( t ) } { \widehat { p } _ { j } ( t ) } d t \right\} .\tag{170}
$$

This function incorporates both truncation and the estimated dictionary. It is generally random through $\mathcal { D } _ { n } ^ { 0 }$ Although $\mathbb { E } _ { P } f = 0$ , the coeficient $\mu _ { f }$ need not vanish when $P$ is nonuniform. Neither $\mu _ { f }$ nor the coeficients of the true function are used by the estimator.

Let define

$$
\pmb { \eta } ^ { \star } : = \left( \mu _ { f } , \theta _ { 1 } ^ { ( 1 ) } , \ldots , \theta _ { M } ^ { ( 1 ) } , \ldots , \theta _ { 1 } ^ { ( d ) } , \ldots , \theta _ { M } ^ { ( d ) } \right) ^ { \top } \in \mathbb { R } ^ { K } .\tag{171}
$$

Then $t _ { M } ( \cdot ) = \widehat \psi ( \cdot ) ^ { \top } \pmb \eta ^ { \star }$ . Indeed, by the definition of the centered dictionary and linearity of the integral,

$$
\begin{array} { l } { \displaystyle \widehat { \psi } ( \mathbf { x } ) ^ { \top } \eta ^ { \star } = \mu _ { f } + \displaystyle \sum _ { j = 1 } ^ { d } \sum _ { m = 1 } ^ { M } \theta _ { m } ^ { ( j ) } \widehat \psi _ { m } ^ { ( j ) } ( x _ { j } ) } \\ { = \mu _ { f } + \displaystyle \sum _ { j = 1 } ^ { d } \left\{ \frac { \sum _ { m = 1 } ^ { M } \theta _ { m } ^ { ( j ) } \phi _ { m } ( x _ { j } ) } { \widehat { p } _ { j } ( x _ { j } ) } - \int _ { 0 } ^ { 1 } \frac { \sum _ { m = 1 } ^ { M } \theta _ { m } ^ { ( j ) } \phi _ { m } ( t ) } { \widehat { p } _ { j } ( t ) } d t \right\} } \\ { = t _ { M } ( \mathbf { x } ) . } \end{array}
$$

Set

$$
\begin{array} { r l } & { r : = f - t _ { M } , } \\ & { r : = ( r ( \mathbf { X } _ { 1 } ^ { 1 } ) , \ldots , r ( \mathbf { X } _ { n _ { 1 } } ^ { 1 } ) ) ^ { \top } , } \\ & { \xi : = ( \xi _ { 1 } ^ { 1 } , \ldots , \xi _ { n _ { 1 } } ^ { 1 } ) ^ { \top } . } \end{array}\tag{172}
$$

The responses satisfy the exact identity

$$
\pmb { Y } ^ { 1 } = \widehat { \pmb { \Psi } } \pmb { \eta } ^ { \star } + \pmb { r } + \pmb { \xi } .\tag{173}
$$

Consequently, on $\zeta _ { n } .$

$$
\widehat { f } _ { M } ( \mathbf { x } ) = t _ { M } ( \mathbf { x } ) + \widehat { \psi } ( \mathbf { x } ) ^ { \top } ( \widehat { \boldsymbol { \Psi } } ^ { \top } \widehat { \boldsymbol { \Psi } } ) ^ { - 1 } \widehat { \boldsymbol { \Psi } } ^ { \top } \boldsymbol { r } + \widehat { \psi } ( \mathbf { x } ) ^ { \top } ( \widehat { \boldsymbol { \Psi } } ^ { \top } \widehat { \boldsymbol { \Psi } } ) ^ { - 1 } \widehat { \boldsymbol { \Psi } } ^ { \top } \boldsymbol { \xi } .\tag{174}
$$

Equivalently,

$$
f ( \mathbf { x } ) - \widehat { f } _ { M } ( \mathbf { x } ) = r ( \mathbf { x } ) - \frac { 1 } { n _ { 1 } } \widehat { \psi } ( \mathbf { x } ) ^ { \top } \widehat \mathbf { T } ^ { - 1 } \widehat { \Psi } ^ { \top } r - \frac { 1 } { n _ { 1 } } \widehat { \psi } ( \mathbf { x } ) ^ { \top } \widehat \mathbf { T } ^ { - 1 } \widehat { \Psi } ^ { \top } \pmb { \xi } .\tag{175}
$$

Conditional on the two designs $\mathcal { D } _ { n }$ , the dictionary, $r , r ,$ and $\zeta _ { n }$ are fixed, whereas

$$
\mathbb { E } [ \pmb { \xi } \mid \mathcal { D } _ { n } ] = 0 , \qquad \mathbb { E } [ \pmb { \xi } \pmb { \xi } ^ { \top } \mid \mathcal { D } _ { n } ] = \sigma ^ { 2 } \mathbf { I } _ { n _ { 1 } } .\tag{176}
$$

The cross term involving the noise therefore has conditional expectation zero. On $\zeta _ { n } .$ , we obtain the bias-variance decomposition

$$
\left| \mathbb { E } \left[ \left\| f - \widehat { f } _ { M } \right\| ^ { 2 } \mathbf { 1 } _ { \mathcal { G } _ { n } } \mid \mathcal { D } _ { n } \right] = \underbrace { \left\| r - \frac { 1 } { n _ { 1 } } \widehat { \psi } ^ { \top } \widehat { \mathbf { T } } ^ { - 1 } \widehat { \Psi } ^ { \top } r \right\| ^ { 2 } \mathbf { 1 } _ { \mathcal { G } _ { n } } } _ { \mathrm { c o n d i t i o n a l ~ s q u a r e d ~ b i a s } } + \underbrace { \mathbb { E } \left[ \left\| \frac { 1 } { n _ { 1 } } \widehat { \psi } ^ { \top } \widehat { \mathbf { T } } ^ { - 1 } \widehat { \Psi } ^ { \top } \xi \right\| ^ { 2 } \mathbf { 1 } _ { \mathcal { G } _ { n } } \mid \mathcal { D } _ { n } \right] } _ { \mathrm { c o n d i t i o n a l ~ v a r i a n c e } } . \right|\tag{177}
$$

Here $\widehat { \psi }$ denotes the vector-valued function of the test covariate inside the $L ^ { 2 } ( P )$ norm. On $\zeta _ { n } ^ { c } , \widehat { f } _ { M } = 0$ so the conditional risk is just given by

$$
\bigg | \mathbb { E } \Big [ \| f - \widehat { f } _ { M } \| ^ { 2 } \mathbf { 1 } _ { \zeta _ { n } ^ { c } } \mid \mathcal { D } _ { n } \Big ] = \| f \| ^ { 2 } \mathbf { 1 } _ { \zeta _ { n } ^ { c } } .\tag{178}
$$

## G.4 Uniform geometry of the estimated dictionary

We first introduce the standard lemma which bridges the population Gram matrix Γ and the function $\widehat { \psi } .$ Lemma G.2. Let $\pmb { \eta } \in \mathbb { R } ^ { K }$ , we have the following identity

$$
\| \widehat \psi \eta \| ^ { 2 } = \eta ^ { \top } \Gamma \eta .\tag{179}
$$

Proof. It sufices to start from the definition of Γ and compute $\eta ^ { \intercal } \mathbf { T } \eta$ , indeed we have

$$
\eta ^ { \top } \Gamma \eta = \eta ^ { \top } \mathbb { E } \Big [ \widehat { \psi } ( \mathbf { X } ) \widehat { \psi } ( \mathbf { X } ) ^ { \top } \mid \mathcal { D } _ { n } ^ { 0 } \Big ] \eta\tag{180}
$$

$$
= \mathbb { E } \left[ \pmb { \eta } ^ { \top } \widehat { \pmb { \psi } } ( \mathbf { X } ) \widehat { \pmb { \psi } } ( \mathbf { X } ) ^ { \top } \pmb { \eta } \mid \mathcal { D } _ { n } ^ { 0 } \right]
$$

$$
\mathbf { \eta } = \mathbb { E } \left[ \left( \eta ^ { \top } \widehat { \psi } ( \mathbf { X } ) \right) ^ { 2 } | \mathbf { \mathcal { D } } _ { n } ^ { 0 } \right]\tag{181}
$$

$$
= \int \left( \eta ^ { \top } \widehat { \psi } ( \mathbf { x } ) \right) ^ { 2 } p ( \mathbf { x } ) d \mathbf { x }\tag{182}
$$

$$
= \| \hat { \psi } \eta \| ^ { 2 } .\tag{183}
$$

(184)

Lemma G.3 (Population Gram matrix and row norm). For every realization of the first sample satisfying (158),

$$
C _ { \operatorname* { m i n } } { I _ { K } } \preceq { \bf { \Gamma } } \preceq C _ { \operatorname* { m a x } } { I _ { K } } .\tag{185}
$$

Moreover,

$$
\operatorname* { s u p } _ { \mathbf x \in [ 0 , 1 ] ^ { d } } \| \widehat \psi ( \mathbf x ) \| _ { 2 } ^ { 2 } \leq 1 + \frac { 8 d M } { \kappa _ { \operatorname* { m i n } } ^ { 2 } } \leq \frac { 8 K } { \kappa _ { \operatorname* { m i n } } ^ { 2 } } .\tag{186}
$$

These bounds require no event on which the estimated densities are close to the true densities.

Proof. Fix $\mathcal { D } _ { n } ^ { 0 }$ . For $\pmb { u } = ( u _ { 1 } , \ldots , u _ { M } ) ^ { \top }$ , set

$$
h = \sum _ { m = 1 } ^ { M } u _ { m } \phi _ { m } , \qquad v = \frac { h } { \widehat { p _ { j } } } - \int _ { 0 } ^ { 1 } \frac { h ( t ) } { \widehat { p _ { j } } ( t ) } d t .\tag{187}
$$

Since $\textstyle \int h d \lambda = 0$ and $\widehat { p } _ { j } \leq \kappa _ { \mathrm { m a x } }$

$$
\langle h , v \rangle _ { L ^ { 2 } ( \lambda ) } = \int _ { 0 } ^ { 1 } \frac { h ( t ) ^ { 2 } } { \widehat { p _ { j } } ( t ) } d t \geq \frac { 1 } { \kappa _ { \operatorname* { m a x } } } \| h \| _ { L ^ { 2 } ( \lambda ) } ^ { 2 } .\tag{188}
$$

Cauchy–Schwarz yields $\| v \| _ { L ^ { 2 } ( \lambda ) } \ge \| h \| _ { L ^ { 2 } ( \lambda ) } / \kappa _ { \operatorname* { m a x } } ;$ the statement is immediate if $h = 0$ . Conversely, subtracting the Lebesgue mean is an orthogonal projection in $L ^ { 2 } ( \lambda )$ , so

$$
\| v \| _ { L ^ { 2 } ( \lambda ) } \le \| h / \widehat { p } _ { j } \| _ { L ^ { 2 } ( \lambda ) } \le \frac { 1 } { \kappa _ { \operatorname* { m i n } } } \| h \| _ { L ^ { 2 } ( \lambda ) } .\tag{189}
$$

By orthonormality of the $\phi _ { m }$ , we have therefore proved

$$
\frac { 1 } { \kappa _ { \mathrm { m a x } } ^ { 2 } } \| \pmb { u } \| _ { 2 } ^ { 2 } \leq \| \boldsymbol { v } \| _ { L ^ { 2 } ( \lambda ) } ^ { 2 } \leq \frac { 1 } { \kappa _ { \mathrm { m i n } } ^ { 2 } } \| \pmb { u } \| _ { 2 } ^ { 2 } .\tag{190}
$$

For any $\pmb { \eta } \in \mathbb { R } ^ { K }$ , let compute $\begin{array} { r } { \widehat { \psi } ^ { \top } \eta : } \end{array}$

$$
\begin{array} { r l r } {  { \widehat { \psi } ^ { \top } \pmb { \eta } = \eta _ { 0 } + \sum _ { j = 1 } ^ { d } \displaystyle \sum _ { m = 1 } ^ { M } \eta _ { m } ^ { ( j ) } \widehat { \psi } _ { m } ^ { ( j ) } } } \\ & { } & { = \eta _ { 0 } + \sum _ { j = 1 } ^ { d } \displaystyle \sum _ { m = 1 } ^ { M } \eta _ { m } ^ { ( j ) } ( \frac { \phi _ { m } } { \widehat { p } _ { j } } - \int _ { [ 0 , 1 ] } \frac { \phi _ { m } ( t ) } { \widehat { p } _ { j } ( t ) } d t ) . } \\ & { } & \end{array}\tag{191}
$$

(192)

We can write $\begin{array} { r } { \widehat { \psi } ^ { \top } \eta = \eta _ { 0 } + \sum _ { j = 1 } ^ { d } v _ { j } } \end{array}$ , where the $v _ { j }$ ’s are centered, univariate and mutually orthogonal under the product measure $\lambda _ { d }$ . Thus

$$
\| \widehat { \psi } ^ { \top } \pmb { \eta } \| _ { L ^ { 2 } ( \lambda _ { d } ) } ^ { 2 } = \eta _ { 0 } ^ { 2 } + \sum _ { j = 1 } ^ { d } \| v _ { j } \| _ { L ^ { 2 } ( \lambda ) } ^ { 2 } .\tag{193}
$$

The equation (190) yields

$$
\eta _ { 0 } ^ { 2 } + \frac { 1 } { \kappa _ { \operatorname* { m a x } } ^ { 2 } } \sum _ { j = 1 } ^ { d } \sum _ { m = 1 } ^ { M } \left( \eta _ { m } ^ { ( j ) } \right) ^ { 2 } \leq \| \widehat { \psi } ^ { \top } \eta \| _ { L ^ { 2 } ( \lambda _ { d } ) } ^ { 2 } \leq \eta _ { 0 } ^ { 2 } + \frac { 1 } { \kappa _ { \operatorname* { m i n } } ^ { 2 } } \sum _ { j = 1 } ^ { d } \sum _ { m = 1 } ^ { M } \left( \eta _ { m } ^ { ( j ) } \right) ^ { 2 } .\tag{194}
$$

By using that $\kappa _ { \operatorname* { m i n } } < 1 < \kappa _ { \operatorname* { m a x } }$ , we obtain

$$
\frac { 1 } { \kappa _ { \operatorname* { m a x } } ^ { 2 } } \| \pmb { \eta } \| _ { 2 } ^ { 2 } \leq \| \widehat { \psi } ^ { \top } \pmb { \eta } \| _ { L ^ { 2 } ( \lambda _ { d } ) } ^ { 2 } \leq \frac { 1 } { \kappa _ { \operatorname* { m i n } } ^ { 2 } } \| \pmb { \eta } \| _ { 2 } ^ { 2 } .\tag{195}
$$

Finally, since $\kappa _ { \mathrm { m i n } } \le p \le \kappa _ { \mathrm { m a x } }$ , we have

$$
\begin{array} { r } { C _ { \operatorname* { m i n } } \| \pmb { \eta } \| _ { 2 } ^ { 2 } \leq \| \widehat { \psi } ^ { \top } \pmb { \eta } \| ^ { 2 } = \pmb { \eta } ^ { \top } \mathbf { \Gamma } \pmb { \Gamma } \pmb { \eta } \leq C _ { \operatorname* { m a x } } \| \pmb { \eta } \| _ { 2 } ^ { 2 } , } \end{array}\tag{196}
$$

which proves (185). For (186), use $| \widehat { \psi } _ { m } ^ { ( j ) } | \leq 2 \sqrt { 2 } / \kappa _ { \operatorname* { m i n } }$ , sum the squared coordinates, and recall that $\kappa _ { \operatorname* { m i n } } < 1$ .

## G.5 Proof of Lemma 4.3

Fix $( p , f )$ and $\mathcal { D } _ { n } ^ { 0 }$ , and define $w _ { j } : = f _ { j } - h _ { j } / \widehat { p } _ { j }$ . Since $\begin{array} { r } { \mu _ { f } = \sum _ { j = 1 } ^ { d } \int _ { 0 } ^ { 1 } f _ { j } d \lambda } \end{array}$

$$
r ( \mathbf { x } ) = \sum _ { j = 1 } ^ { d } \left( w _ { j } ( x _ { j } ) - \int _ { 0 } ^ { 1 } w _ { j } ( x ) d x \right) .\tag{197}
$$

The summands in (197) are centered and orthogonal under $\lambda _ { d } ,$ for every realization of $\mathcal { D } _ { n } ^ { 0 }$ . Consequently,

$$
\begin{array} { r l r } {  { \| r \| _ { L ^ { 2 } ( P ) } ^ { 2 } \le \kappa _ { \operatorname* { m a x } } \| r \| _ { L ^ { 2 } ( \lambda _ { d } ) } ^ { 2 } } } \\ & { } & { \quad \le \kappa _ { \operatorname* { m a x } } \displaystyle \sum _ { j = 1 } ^ { d } \| w _ { j } - \int _ { 0 } ^ { 1 } w _ { j } d \lambda \| _ { L ^ { 2 } ( \lambda ) } ^ { 2 } } \end{array}\tag{198}
$$

$$
\leq \kappa _ { \operatorname* { m a x } } \sum _ { j = 1 } ^ { d } \| w _ { j } \| _ { L ^ { 2 } ( \lambda ) } ^ { 2 } .\tag{199}
$$

Using $f _ { j } = g _ { j } / p _ { j }$ , we decompose each summand as

$$
w _ { j } = \frac { g _ { j } - h _ { j } } { p _ { j } } + h _ { j } \left( \frac { 1 } { p _ { j } } - \frac { 1 } { \widehat { p _ { j } } } \right) .\tag{200}
$$

The density lower bounds and $( u + v ) ^ { 2 } \leq 2 u ^ { 2 } + 2 v ^ { 2 }$ yield

$$
\| w _ { j } \| _ { L ^ { 2 } ( \lambda ) } ^ { 2 } \le \frac { 2 } { \kappa _ { \operatorname* { m i n } } ^ { 2 } } \| g _ { j } - h _ { j } \| _ { L ^ { 2 } ( \lambda ) } ^ { 2 } + \frac { 2 } { \kappa _ { \operatorname* { m i n } } ^ { 4 } } \int _ { 0 } ^ { 1 } h _ { j } ( x ) ^ { 2 } | \widehat { p } _ { j } ( x ) - p _ { j } ( x ) | ^ { 2 } d t .\tag{201}
$$

By Parseval’s identity and (153),

$$
\| g _ { j } - h _ { j } \| _ { L ^ { 2 } ( \lambda ) } ^ { 2 } = \sum _ { m > M } ( \theta _ { m } ^ { ( j ) } ) ^ { 2 } \leq R ^ { 2 } M ^ { - 2 \beta } , \qquad \| h _ { j } \| _ { L ^ { 2 } ( \lambda ) } ^ { 2 } \leq R ^ { 2 } .\tag{202}
$$

For the fixed pair $( p , f ) , h _ { j }$ is deterministic. Tonelli’s theorem and (159) therefore give

$$
\begin{array} { r l r } {  { \mathbb { E } [ \int _ { 0 } ^ { 1 } h _ { j } ( x ) ^ { 2 } | \widehat { p } _ { j } ( x ) - p _ { j } ( x ) | ^ { 2 } d t ] = \int _ { 0 } ^ { 1 } h _ { j } ( x ) ^ { 2 } \mathbb { E } [ | \widehat { p } _ { j } ( x ) - p _ { j } ( x ) | ^ { 2 } ] d x } } \\ & { } & { \leq \rho _ { n _ { 0 } } \| h _ { j } \| _ { L ^ { 2 } ( \lambda ) } ^ { 2 } } \\ & { } & { \leq R ^ { 2 } \rho _ { n _ { 0 } } . } \end{array}\tag{203}
$$

Substituting (202) and (203) into (201), and summing in (198), proves the claim.

Remark G.4 (The dependence on dimension and the density norm). The centering identity (197) is responsible for the linear dependence on $d \colon$ it avoids bounding the squared norm of a sum by d times the sum of squared norms. The calculation uses $\begin{array} { r } { \operatorname* { s u p } _ { t } \mathbb { E } | \widehat { p } _ { j } ( t ) - p _ { j } ( t ) | ^ { 2 } } \end{array}$ , not $\mathbb { E } \| \widehat { p } _ { j } - p _ { j } \| _ { L ^ { \infty } } ^ { 2 }$ . Indeed, the deterministic weights $h _ { j } ^ { 2 }$ allow the pointwise risk bound to be integrated directly. This uses only $\| h _ { j } \| _ { L ^ { 2 } ( \lambda ) } \leq R .$ , so it applies to every $\beta > 0$ and allows dependence between the diferent marginal estimators.

## G.6 Bounding the conditional squared bias

In this subsection, we prove the inequality of equation (32).

On $\zeta _ { n } .$ , define

$$
\begin{array} { r } { \pmb { A } : = \widehat { \Psi } ^ { \top } \widehat { \pmb { \Psi } } = n _ { 1 } \widehat { \Gamma } , \qquad \pmb { B } : = \pmb { A } ^ { - 1 } \widehat { \pmb { \Psi } } ^ { \top } . } \end{array}\tag{204}
$$

We extend B by zero on $\zeta _ { n } ^ { c }$ when writing expressions with an indicator. On $\zeta _ { n }$

$$
B B ^ { \top } = A ^ { - 1 } \widehat \Psi ^ { \top } \widehat \Psi A ^ { - 1 } = A ^ { - 1 } ,\tag{205}
$$

and therefore

$$
\| B \| _ { \mathrm { o p } } ^ { 2 } = \| A ^ { - 1 } \| _ { \mathrm { o p } } \leq \frac { 1 } { n _ { 1 } c _ { 1 } } .\tag{206}
$$

The triangle inequality and the population norm bound (196) implies, on this event,

$$
\begin{array} { l } { \displaystyle \| r - \widehat \psi ^ { \top } B r \| ^ { 2 } \leq 2 \| r \| ^ { 2 } + 2 \| \widehat \psi ^ { \top } B r \| ^ { 2 } } \\ { \leq 2 \| r \| ^ { 2 } + 2 C _ { \operatorname* { m a x } } \| B r \| _ { 2 } ^ { 2 } } \\ { \leq 2 \| r \| ^ { 2 } + \displaystyle \frac { 2 C _ { \operatorname* { m a x } } } { n _ { 1 } c _ { 1 } } \sum _ { i = 1 } ^ { n _ { 1 } } r ( \mathbf { X } _ { i } ^ { 1 } ) ^ { 2 } . } \end{array}\tag{207}
$$

Conditional on $\mathcal { D } _ { n } ^ { 0 } .$ , r is fixed and the second-sample covariates are i.i.d. with distribution P. Hence

$$
\mathbb { E } \left[ \sum _ { i = 1 } ^ { n _ { 1 } } r ( \mathbf { X } _ { i } ^ { 1 } ) ^ { 2 } \mid \mathcal { D } _ { n } ^ { 0 } \right] = n _ { 1 } \| r \| ^ { 2 } .\tag{208}
$$

Multiplying (207) by $\mathbf { 1 } _ { \zeta _ { n } }$ , discarding the indicator on its nonnegative right-hand side, and using (208) gives

$$
\mathbb { E } \Big [ \| r - \widehat { \psi } ^ { \top } B r \| ^ { 2 } \mathbf { 1 } _ { \zeta _ { n } } \mid \mathcal { D } _ { n } ^ { 0 } \Big ] \leq \left( 2 + \frac { 2 C _ { \operatorname* { m a x } } } { c _ { 1 } } \right) \| r \| ^ { 2 } .\tag{209}
$$

In particular, no concentration inequality for the approximation residual is needed. Also, the two terms inside the norm on the left are not asserted to be orthogonal in $L ^ { 2 } ( P )$ ).

## G.7 Bounding the conditional variance

In this subsection, we prove the inequality of equation (33).

The prediction loss additionally involves the population Gram matrix. Recall also that $B B ^ { \top } = ( \widehat \Psi ^ { \top } \widehat \Psi ) ^ { - 1 }$ on $\zeta _ { n } .$

$$
\begin{array} { r l } & { \mathbb { E } \Big [ \| \hat { \psi } ^ { \top } B \xi \| ^ { 2 } \bigm | \mathcal { D } _ { n } \Big ] = \mathbb { E } \big [ ( B \xi ) ^ { \top } \mathbf { \Gamma } ( B \xi ) \bigm | \mathcal { D } _ { n } \Big ] } \\ & { \qquad = \mathbb { E } \left[ \xi ^ { \top } B ^ { \top } \mathbf { \Gamma } B \xi \bigm | \mathcal { D } _ { n } \right] } \\ & { \qquad = \mathrm { T r } \big ( \mathbb { E } \left[ \xi ^ { \top } B ^ { \top } \mathbf { \Gamma } B \xi \bigm | \mathcal { D } _ { n } \right] \bigm ) } \\ & { \qquad = \mathrm { T r } \big ( B ^ { \top } \mathbf { \Gamma } \mathbf { } \mathbf { { \cal T } } B \mathbb { E } \left[ \xi \xi ^ { \top } \bigm | \mathcal { D } _ { n } \right] \bigm ) } \\ & { \qquad = \sigma ^ { 2 } \mathrm { T r } \big ( \mathbf { \Gamma } B B ^ { \top } \bigm ) } \\ & { \qquad = \sigma ^ { 2 } \mathrm { T r } \big ( \mathbf { \Gamma } ( \hat { \Psi } ^ { \top } \hat { \Psi } ) ^ { - 1 } \bigm ) . } \end{array}\tag{210}
$$

Using that $\mathbf { r } \preceq C _ { \mathrm { m a x } } \pmb { I } _ { K }$ and recalling that on $\zeta _ { n }$ we have $\widehat \Psi ^ { \top } \widehat \Psi \succeq n _ { 1 } c _ { 1 } I _ { K }$ , this leads to the desired result. And because $\zeta _ { n }$ is measurable with respect to $\mathcal { D } _ { n }$ , the tower property gives

$$
\mathbb { E } \Big [ \| \widehat { \psi } ^ { \top } B \xi \| ^ { 2 } \mathbf { 1 } _ { \zeta _ { n } } \mid \mathcal { D } _ { n } ^ { 0 } \Big ] \leq \frac { \sigma ^ { 2 } C _ { \operatorname* { m a x } } K } { n _ { 1 } c _ { 1 } } .\tag{211}
$$

## G.8 Risk on the rejected event

We use only the following lower-tail form of the matrix Chernof inequality. We state the precise result and its parameters to make the dependence on K explicit.

Lemma G.5 (Rejection probability). Let define the following quantity

$$
\Delta _ { n , K } : = \operatorname* { m i n } \left\{ 1 , K \exp \left( - c _ { 2 } \frac { n _ { 1 } } { K } \right) \right\} .\tag{212}
$$

Almost surely with respect to the first sample, we have the following upper bound

$$
\begin{array} { r } { \mathbb { P } _ { p , f } ( \zeta _ { n } ^ { c } \mid \mathcal { D } _ { n } ^ { 0 } ) \leq \Delta _ { n , K } . } \end{array}\tag{213}
$$

Proof. Condition on $\mathcal { D } _ { n } ^ { 0 }$ and set

$$
\begin{array} { r } { S _ { i } : = \widehat { \psi } ( \mathbf { X } _ { i } ^ { 1 } ) \widehat { \psi } ( \mathbf { X } _ { i } ^ { 1 } ) ^ { \top } , \quad \quad 1 \leq i \leq n _ { 1 } . } \end{array}\tag{214}
$$

These matrices are conditionally independent and positive semidefinite. Their sum is $n _ { 1 } \widehat { \mathbf { I } }$ . The only nonzero eigenvalue of a matrix $\mathbf { \boldsymbol { v } } \mathbf { \boldsymbol { v } } ^ { \top }$ is $\| \pmb { v } \| _ { 2 } ^ { 2 }$ . Thus Lemma G.3 verifies the hypotheses of Theorem D.6, conditionally on $\mathcal { D } _ { n } ^ { 0 }$ , with

$$
N = n _ { 1 } , \qquad b _ { 0 } = \frac { 8 K } { \kappa _ { \mathrm { m i n } } ^ { 2 } } , \qquad \mu _ { \mathrm { m i n } } = n _ { 1 } \lambda _ { \mathrm { m i n } } ( \mathbf { T } ) \geq n _ { 1 } C _ { \mathrm { m i n } } .\tag{215}
$$

Since $c = C _ { \mathrm { m i n } } / 2$ , we have the inclusion

$$
\zeta _ { n } ^ { c } \subseteq \left\{ \lambda _ { \operatorname* { m i n } } \left( \sum _ { i = 1 } ^ { n _ { 1 } } S _ { i } \right) \leq \mu _ { \operatorname* { m i n } } / 2 \right\} .\tag{216}
$$

Applying (81) and using (215),

$$
\mathbb { P } ( \zeta _ { n } ^ { c } \mid \mathcal { D } _ { n } ^ { 0 } ) \le K \exp \left( - \frac { 1 - \log 2 } { 2 } \frac { n _ { 1 } C _ { \mathrm { m i n } } \kappa _ { \mathrm { m i n } } ^ { 2 } } { 8 K } \right)\tag{217}
$$

$$
= K \exp \biggl ( - c _ { 2 } \frac { n _ { 1 } } { K } \biggr ) .\tag{218}
$$

Taking the minimum with 1 proves the claim.

The Riesz bounds and the Sobolev smoothness yield the following inequality for every $( p , f ) \in { \mathcal { C } } _ { \beta , \gamma }$

$$
\| f \| ^ { 2 } \leq C _ { \operatorname* { m a x } } d R ^ { 2 } .\tag{219}
$$

Because $\widehat { f } _ { M } = 0$ on $\zeta _ { n } ^ { c }$ , Lemmas G.5 and this last inequality give, conditionally on $\mathcal { D } _ { n } ^ { 0 }$

$$
\mathbb { E } \Big [ \| f - \widehat { f } _ { M } \| ^ { 2 } { \mathbf 1 } _ { \zeta _ { n } ^ { c } } \ | \ \mathcal { D } _ { n } ^ { 0 } \Big ] = \| f \| ^ { 2 } \mathbb { P } ( \zeta _ { n } ^ { c } \ | \ \mathcal { D } _ { n } ^ { 0 } )\tag{220}
$$

$$
\leq C _ { \mathrm { m a x } } d R ^ { 2 } \Delta _ { n , K } .\tag{221}
$$

In particular, the rejection contribution involves the squared $L ^ { 2 } ( P )$ norm of the signal, rather than a supremum norm.

## G.9 Combining the bounds

In this subsection we combine all the bound and complete the proof of Theorem 4.4.

First, by integrating the equation (177) over $\mathcal { D } _ { n } ^ { 1 }$ , we obtain the following identity

$$
\mathbb { E } \Big [ \| f - \widehat f _ { M } \| ^ { 2 } \mathbf { 1 } _ { \zeta _ { n } } \ | \ \mathcal { D } _ { n } ^ { 0 } \Big ] = \mathbb { E } \Big [ \| r - \widehat \psi ^ { \top } B r \| ^ { 2 } \mathbf { 1 } _ { \zeta _ { n } } \ | \ \mathcal { D } _ { n } ^ { 0 } \Big ] + \mathbb { E } \Big [ \| \widehat \psi ^ { \top } B \xi \| ^ { 2 } \mathbf { 1 } _ { \zeta _ { n } } \ | \ \mathcal { D } _ { n } ^ { 0 } \Big ] \ .\tag{222}
$$

Equations (209) and (211) imply

$$
{  { \mathbb E } } \Big [ \| f - \widehat f _ { M } \| ^ { 2 } { \mathbf 1 } _ { \zeta _ { n } } \ | \ {  { \mathcal D } } _ { n } ^ { 0 } \Big ] \leq \left( 2 + \frac { 2 C _ { \operatorname* { m a x } } } { c _ { 1 } } \right) \| r \| ^ { 2 } + \frac { \sigma ^ { 2 } C _ { \operatorname* { m a x } } K } { n _ { 1 } c _ { 1 } } .\tag{223}
$$

Adding (220) and integrating over $\mathcal { D } _ { n } ^ { 0 }$ yields

$$
\mathbb { E } \Big [ \| f - \widehat { f } _ { M } \| ^ { 2 } \Big ] \leq \bigg ( 2 + \frac { 4 C _ { \operatorname* { m a x } } } { C _ { \operatorname* { m i n } } } \bigg ) \mathbb { E } _ { p , f } [ \| r \| ^ { 2 } ] + \frac { 2 \sigma ^ { 2 } C _ { \operatorname* { m a x } } K } { n _ { 1 } C _ { \operatorname* { m i n } } } + C _ { \operatorname* { m a x } } d R ^ { 2 } \Delta _ { n , K } .\tag{224}
$$

Substitute the approximation bound of Lemma 4.3 yields this final upper bound

$$
\left. \mathbb { E } \left[ \Vert f - \widehat { f } _ { M } \Vert ^ { 2 } \right] \leq \left( 2 + \frac { 4 C _ { \operatorname* { m a x } } } { C _ { \operatorname* { m i n } } } \right) 2 \kappa _ { \operatorname* { m a x } } d R ^ { 2 } \left( \frac { M ^ { - 2 \beta } } { \kappa _ { \operatorname* { m i n } } ^ { 2 } } + \frac { \rho _ { n _ { 0 } } } { \kappa _ { \operatorname* { m i n } } ^ { 4 } } \right) + \frac { 2 \sigma ^ { 2 } C _ { \operatorname* { m a x } } K } { n _ { 1 } C _ { \operatorname* { m i n } } } + C _ { \operatorname* { m a x } } d R ^ { 2 } \Delta _ { n , K } \right.\tag{225}
$$

Every term on the resulting right-hand side is independent of the particular pair $( p , f )$ , which justifies taking the supremum over the whole class and allows us to obtain a minimax upper bound. Combining this inequality with $n _ { 0 } , n _ { 1 } \asymp n$ and $\rho _ { n _ { 0 } } \lesssim n ^ { - \frac { 2 \gamma } { 2 \gamma + 1 } }$ yields an upper bound of the following form

$$
\operatorname* { s u p } _ { ( p , f ) \in \mathcal { C } _ { \beta , \gamma } } \mathbb { E } _ { p , f } [ \| \widehat { f } _ { M } - f \| ^ { 2 } ] \lesssim d M ^ { - 2 \beta } + \frac { d M } { n } + d n ^ { - \frac { 2 \gamma } { 2 \gamma + 1 } } + d \Delta _ { n , K } .\tag{226}
$$

This bound is valid without a growth condition on $d ;$ the rejection term is retained explicitly.

Balancing the Sobolev approximation term and the variance gives $M \asymp n ^ { \frac { 1 } { 2 \beta + 1 } }$ . Under Assumption 2.13,

$$
d \log n = o \bigg ( n ^ { \frac { 2 \beta } { 2 \beta + 1 } } \bigg ) ,\tag{227}
$$

and therefore

$$
K \log n \asymp d M \log n = o ( n ) .\tag{228}
$$

Since $n _ { 1 } \asymp n$ and $c _ { 2 } > 0$ is fixed, it follows that

$$
\frac { c _ { 2 } n _ { 1 } } { K \log n } \longrightarrow \infty .\tag{229}
$$

Moreover, $K \leq n$ for all suficiently large n. Using $\Delta _ { n , K } \leq K \exp ( - c _ { 2 } n _ { 1 } / K )$ , we obtain

$$
n ^ { \frac { 2 \beta } { 2 \beta + 1 } } \Delta _ { n , K } \leq \exp \left( { \frac { 2 \beta } { 2 \beta + 1 } } \log n + \log K - { \frac { c _ { 2 } n _ { 1 } } { K } } \right)\tag{230}
$$

$$
\leq \exp \left[ \log n \left( 1 + \frac { 2 \beta } { 2 \beta + 1 } - \frac { c _ { 2 } n _ { 1 } } { K \log n } \right) \right] \longrightarrow 0 .\tag{231}
$$

Consequently,

$$
d \Delta _ { n , K } = o \bigg ( d n ^ { - \frac { 2 \beta } { 2 \beta + 1 } } \bigg ) .\tag{232}
$$

We finally obtain the desired minimax upperbound

$$
\left| \operatorname* { i n f } _ { \widetilde f } \operatorname* { s u p } _ { ( p , f ) \in { \mathscr C } _ { \beta , \gamma } } \mathbb E _ { p , f } [ \| \widetilde f - f \| ^ { 2 } ] \lesssim d n ^ { - \frac { 2 \beta } { 2 \beta + 1 } } + d n ^ { - \frac { 2 \gamma } { 2 \gamma + 1 } } . \right|\tag{233}
$$

## G.10 Construction of the marginal-density estimators

We now establish the density-estimation guarantee used in the regression analysis. We use piecewise polynomial projection followed by pointwise clipping. This construction handles the boundary of [0, 1] without extending the marginal densities beyond their domain.

Throughout this subsection, write

$$
s = \lfloor \gamma \rfloor , \qquad \alpha = \gamma - s \in [ 0 , 1 ) .\tag{234}
$$

Under our definition of $( \gamma , L )$ -Hölder regularity, each marginal density $p _ { j }$ belongs to $C ^ { s } ( [ 0 , 1 ] )$ and satisfies

$$
| p _ { j } ^ { ( s ) } ( u ) - p _ { j } ^ { ( s ) } ( v ) | \leq L | u - v | ^ { \alpha } , \qquad u , v \in [ 0 , 1 ] , \quad u \neq v .\tag{235}
$$

Here $p _ { j } ^ { ( 0 ) } = p _ { j }$ . When $\gamma$ is an integer, $\alpha = 0$ and this condition bounds the oscillation of $p _ { j } ^ { ( s ) }$ by L. The marginal density bounds $\kappa _ { \mathrm { m i n } } \le p _ { j } \le \kappa _ { \mathrm { m a x } }$ follow from the joint density bounds defining $\mathcal { C } _ { \beta , \gamma }$

Lemma G.6 (Uniform pointwise density risk). Let $s = \lfloor \gamma \rfloor$ and $n _ { 0 } \geq 1$ . For every integer $J \geq 1$ , there exist estimators $\widehat { p } _ { j }$ , depending on J and constructed from $\mathcal { D } _ { n } ^ { 0 }$ , satisfying (158) and

$$
\operatorname* { m a x } _ { 1 \leq j \leq d } \operatorname* { s u p } _ { t \in [ 0 , 1 ] } \mathbb { E } _ { p } \big [ | \widehat { p } _ { j } ( t ) - p _ { j } ( t ) | ^ { 2 } \big ] \leq \left( \frac { ( s + 2 ) L } { s ! } \right) ^ { 2 } J ^ { - 2 \gamma } + \kappa _ { \operatorname* { m a x } } ( s + 1 ) ^ { 2 } \frac { J } { n _ { 0 } } .\tag{236}
$$

In particular, choosing $J = \lceil n _ { 0 } ^ { \frac { 1 } { 2 \gamma + 1 } } \rceil$ provides the guarantee (159) with a deterministic bound $\rho _ { n _ { 0 } }$ satisfying

$$
\rho _ { n _ { 0 } } \le C _ { \gamma , L , \kappa _ { \operatorname* { m a x } } } n _ { 0 } ^ { - \frac { 2 \gamma } { 2 \gamma + 1 } } \lesssim n _ { 0 } ^ { - \frac { 2 \gamma } { 2 \gamma + 1 } }\tag{237}
$$

where the underlying constant is independent of $n , d$ and the pair $( p , f )$

Proof. Fix $( p , f ) \in { \mathcal { C } } _ { \beta , \gamma }$ . The estimators below depend only on the first-subsample covariates, so their distributions depend on p but not on f and the expectations will be taken over $p .$

Let be a integer $J \geq 1$ , partition [0, 1] into intervals of length $1 / J$ by setting for every $k \in \{ 1 , \ldots , J - 1 \}$

$$
I _ { k } : = \left. \frac { k - 1 } { J } , \frac { k } { J } \right. ,\tag{238}
$$

and

$$
I _ { J } : = \left[ \frac { J - 1 } { J } , 1 \right] .\tag{239}
$$

Thus every point belongs to exactly one interval, including the endpoints.

Let $\mathcal { L } _ { 0 } , \ldots , \mathcal { L } _ { s }$ be the shifted Legendre polynomials normalized to form an orthonormal basis of the polynomials of degree at most s in $L ^ { 2 } ( \lambda )$ . They satisfy

$$
| { \mathcal { L } } _ { r } ( u ) | \leq { \sqrt { 2 r + 1 } } , \qquad \sum _ { r = 0 } ^ { s } { \mathcal { L } } _ { r } ( u ) ^ { 2 } \leq ( s + 1 ) ^ { 2 } , \qquad u \in [ 0 , 1 ] ,\tag{240}
$$

see for example Brézis (2011) or Szegő (1939). Define the piecewise polynomial basis functions

$$
\varphi _ { k r } ( t ) : = \sqrt { J } \mathcal L _ { r } ( J t - k + 1 ) \mathbf 1 _ { I _ { k } } ( t ) , \qquad 1 \leq k \leq J , \quad 0 \leq r \leq s .\tag{241}
$$

These functions form an orthonormal basis in $L ^ { 2 } ( \lambda )$ of the space of functions that are polynomial of degree at most s on each interval. Its projection kernel is

$$
K _ { J } ( t , u ) : = \sum _ { k = 1 } ^ { J } \sum _ { r = 0 } ^ { s } \varphi _ { k r } ( t ) \varphi _ { k r } ( u ) .\tag{242}
$$

For each coordinate $j ,$ define

$$
\begin{array} { l } { \displaystyle \widetilde { p } _ { j } ( t ) : = \frac { 1 } { n _ { 0 } } \sum _ { i = 1 } ^ { n _ { 0 } } K _ { J } ( t , X _ { i j } ^ { 0 } ) , } \\ { \displaystyle \widehat { p } _ { j } ( t ) : = \mathrm { c l i p } \left( \kappa _ { \mathrm { m i n } } , \widetilde { p } _ { j } ( t ) , \kappa _ { \mathrm { m a x } } \right) , } \end{array}\tag{243}
$$

which imposes that $\kappa _ { \mathrm { m i n } } \le \widehat { p } _ { j } \le \kappa _ { \mathrm { m a x } }$ (which are known parameters of the class $\mathcal { C } _ { \beta , \gamma } )$ . Here $X _ { i j } ^ { 0 }$ is the jth coordinate of $\mathbf { X } _ { i } ^ { 0 }$ . The clipped estimator is pointwise contractive, i.e.

$$
| \widehat { p } _ { j } ( t ) - p _ { j } ( t ) | \leq | \widetilde { p } _ { j } ( t ) - p _ { j } ( t ) | .\tag{244}
$$

Indeed, there are three cases. First, $\widetilde { p } _ { j } ( t ) = \widehat { p } _ { j } ( t )$ which implies the equality. Second, we have $\widehat { p _ { j } } ( t ) = \kappa _ { \mathrm { m i n } }$ which implies by clipping that $\widehat { p } _ { j } ( t ) \geq \widetilde { p } _ { j } ( t )$ which leads to

$$
\begin{array} { r } { \widetilde { p } _ { j } ( t ) - p _ { j } ( t ) \le \widehat { p } _ { j } ( t ) - p _ { j } ( t ) = \kappa _ { \operatorname* { m i n } } - p _ { j } ( t ) \le 0 . } \end{array}\tag{245}
$$

By applying the absolute value we obtain the desired inequality. Observe that the third case is symmetric. It therefore sufices to bound the squared bias and variance of $\widetilde { p } _ { j } ( t )$

Bias. Let $\Pi _ { J }$ denote the orthogonal projection associated with $K _ { J }$ , with pointwise representative

$$
( \Pi _ { J } v ) ( t ) = \int _ { 0 } ^ { 1 } K _ { J } ( t , u ) v ( u ) d u = \sum _ { k } \sum _ { r } \langle \varphi _ { k r } , v \rangle _ { L ^ { 2 } ( \lambda ) } \varphi _ { k r } ( t ) .\tag{246}
$$

Since each $X _ { i j } ^ { 0 }$ has density $p _ { j }$ , we can write

$$
\mathbb { E } _ { p } [ \widetilde { p } _ { j } ( t ) ] = \int _ { 0 } ^ { 1 } K _ { J } ( t , u ) p _ { j } ( u ) d u = ( \Pi _ { J } p _ { j } ) ( t ) .\tag{247}
$$

For $t \in I _ { k }$ , the function $K _ { J } ( t , \cdot )$ vanishes outside $I _ { k }$ . By Cauchy–Schwarz, orthonormality, and equation (240), we have

$$
\int _ { I _ { k } } \left| K _ { J } ( t , u ) \right| d u \leq J ^ { - 1 / 2 } \left( \int _ { I _ { k } } K _ { J } ( t , u ) ^ { 2 } d u \right) ^ { 1 / 2 }\tag{248}
$$

$$
\leq J ^ { - 1 / 2 } \left( \sum _ { r = 0 } ^ { s } \varphi _ { k r } ( t ) ^ { 2 } \right) ^ { 1 }\tag{249}
$$

$$
\leq s + 1 .\tag{250}
$$

Let $a _ { k } : = ( k - 1 ) / J ,$ and let

$$
T _ { j k } ( t ) : = \sum _ { r = 0 } ^ { s } \frac { p _ { j } ^ { ( r ) } ( a _ { k } ) } { r ! } ( t - a _ { k } ) ^ { r }\tag{251}
$$

be the Taylor degree s polynomial of $p _ { j }$ at point $a _ { k }$ . We will show that

$$
\operatorname* { s u p } _ { t \in I _ { k } } | p _ { j } ( t ) - T _ { j k } ( t ) | \leq \frac { L } { s ! } J ^ { - \gamma } .\tag{252}
$$

If $s = 0$ , then $0 < \gamma < 1$ and $T _ { j k } = p _ { j } ( a _ { k } )$ . The Hölder condition directly gives

$$
| p _ { j } ( t ) - T _ { j k } ( t ) | \leq L | t - a _ { k } | ^ { \gamma } \leq L J ^ { - \gamma } .\tag{253}
$$

If $s \geq 1$ , Taylor’s integral formula gives

$$
p _ { j } ( t ) - T _ { j k } ( t ) = \frac { 1 } { ( s - 1 ) ! } \int _ { a _ { k } } ^ { t } ( t - u ) ^ { s - 1 } \big ( p _ { j } ^ { ( s ) } ( u ) - p _ { j } ^ { ( s ) } ( a _ { k } ) \big ) d u .\tag{254}
$$

For $u \in [ a _ { k } , t ]$ , the derivative diference is bounded by $L J ^ { - \alpha }$ . When $\alpha = 0$ , this follows from the oscillation bound; when $\alpha > 0$ , it follows from $| u - a _ { k } | \leq 1 / J$ and the Hölder condition. Consequently,

$$
| p _ { j } ( t ) - T _ { j k } ( t ) | \leq { \frac { L J ^ { - \alpha } } { ( s - 1 ) ! } } \int _ { a _ { k } } ^ { t } ( t - u ) ^ { s - 1 } d u\tag{255}
$$

$$
= \frac { L J ^ { - \alpha } } { s ! } ( t - a _ { k } ) ^ { s }\tag{256}
$$

$$
\leq \frac { L } { s ! } J ^ { - ( s + \alpha ) }\tag{257}
$$

$$
= \frac { L } { s ! } J ^ { - \gamma } .\tag{258}
$$

This proves (252) for every $\gamma > 0$ , including integer values.

The projection reproduces polynomials of degree at most s on each cell. Therefore, for $t \in I _ { k }$ , we have

$$
\Pi _ { J } T _ { j k } ( t ) = T _ { j k } ( t ) ,\tag{259}
$$

which yields

$$
\begin{array} { r l } { | ( \Pi _ { J } p _ { j } ) ( t ) - p _ { j } ( t ) | = | ( \Pi _ { J } p _ { j } ) ( t ) - ( \Pi _ { J } T _ { j k } ) ( t ) + T _ { j k } ( t ) - p _ { j } ( t ) | } & { } \\ { \displaystyle } & { = \left| \int _ { I _ { k } } K _ { J } ( t , u ) \big ( p _ { j } ( u ) - T _ { j k } ( u ) \big ) d u - \big ( p _ { j } ( t ) - T _ { j k } ( t ) \big ) \right| } \\ { \displaystyle } & { \le \left( 1 + \int _ { I _ { k } } | K _ { J } ( t , u ) | d u \right) \underset { u \in I _ { k } } { \operatorname* { s u p } } | p _ { j } ( u ) - T _ { j k } ( u ) | } \\ { \displaystyle } & { \le \frac { ( s + 2 ) L } { s ! } J ^ { - \gamma } . } \end{array}
$$

Taking the supremum over all cells gives

$$
\operatorname* { s u p } _ { t \in [ 0 , 1 ] } | \mathbb { E } _ { p } [ \widetilde { p } _ { j } ( t ) ] - p _ { j } ( t ) | \leq \frac { ( s + 2 ) L } { s ! } J ^ { - \gamma } .\tag{260}
$$

Variance. For each fixed coordinate $j ,$ the observations $X _ { 1 j } ^ { 0 } , \ldots , X _ { n _ { 0 } j } ^ { 0 }$ are independent and have density $p _ { j }$ We have

$$
\operatorname { V a r } _ { p } ( \widetilde { p } _ { j } ( t ) ) = \frac { 1 } { n _ { 0 } } \operatorname { V a r } _ { p } ( K _ { J } ( t , X _ { 1 j } ^ { 0 } ) )\tag{261}
$$

$$
\leq \frac { 1 } { n _ { 0 } } \int _ { 0 } ^ { 1 } K _ { J } ( t , u ) ^ { 2 } p _ { j } ( u ) d u\tag{262}
$$

$$
\leq \frac { \kappa _ { \operatorname* { m a x } } } { n _ { 0 } } \int _ { 0 } ^ { 1 } K _ { J } ( t , u ) ^ { 2 } d u\tag{263}
$$

$$
\leq \frac { \kappa _ { \mathrm { m a x } } } { n _ { 0 } } \sum _ { k = 1 } ^ { J } \sum _ { r = 0 } ^ { s } \varphi _ { k r } ( t ) ^ { 2 }\tag{264}
$$

$$
\leq \kappa _ { \mathrm { m a x } } ( s + 1 ) ^ { 2 } \frac { J } { n _ { 0 } } .\tag{265}
$$

The last inequality uses the fact that exactly one cell contains $t ,$ together with (240).

Risk bound. Combining the clipping inequality with the standard bias–variance decomposition yields

$$
\mathbb { E } _ { p , f } \big [ | \widehat { p } _ { j } ( t ) - p _ { j } ( t ) | ^ { 2 } \big ] \leq \mathbb { E } _ { p } \big [ | \widetilde { p } _ { j } ( t ) - p _ { j } ( t ) | ^ { 2 } \big ]\tag{266}
$$

$$
= \left( \mathbb { E } _ { p } [ \widetilde { p } _ { j } ( t ) ] - p _ { j } ( t ) \right) ^ { 2 } + \operatorname { V a r } _ { p } ( \widetilde { p } _ { j } ( t ) )\tag{267}
$$

$$
\leq \left( \frac { ( s + 2 ) L } { s ! } \right) ^ { 2 } J ^ { - 2 \gamma } + \kappa _ { \operatorname* { m a x } } ( s + 1 ) ^ { 2 } \frac { J } { n _ { 0 } } .\tag{268}
$$

The right-hand side is independent of $t , j$ , and the pair $( p , f )$ . Taking the corresponding suprema proves (236). Finally, for $J = \lceil n _ { 0 } ^ { \frac { 1 } { 2 \gamma + 1 } } \rceil$ and $n _ { 0 } \geq 1$

$$
J ^ { - 2 \gamma } \le n _ { 0 } ^ { - \frac { 2 \gamma } { 2 \gamma + 1 } } , \qquad \frac { J } { n _ { 0 } } \le 2 n _ { 0 } ^ { - \frac { 2 \gamma } { 2 \gamma + 1 } } .
$$

Thus (237) holds with

$$
C _ { \gamma , L , \kappa _ { \mathrm { m a x } } } = \left( \frac { ( s + 2 ) L } { s ! } \right) ^ { 2 } + 2 \kappa _ { \mathrm { m a x } } ( s + 1 ) ^ { 2 } .
$$

Since $s = \lfloor \gamma \rfloor$ , this constant depends only on γ, $L ,$ and $\kappa _ { \mathrm { m a x } }$

## G.11 Proof of Theorem 4.6

Let first $\widehat { \pmb { \theta } } _ { M }$ collect the fitted coeficients $\widehat { \theta } _ { m } ^ { ( j ) }$ in dictionary order, and let $\pmb { \theta } _ { M }$ collect the corresponding true coeficients. With these notations, recall that we have

$$
\widehat { \eta } _ { M } = \left( \frac { \widehat { \eta } _ { 0 } } { \widehat { \theta } _ { M } } \right) , \qquad \eta _ { M } ^ { \star } = \left( \mu _ { f } ^ { \mu _ { f } } \right) , \qquad \mu _ { f } = \int _ { [ 0 , 1 ] ^ { d } } f ( x ) d x ,\tag{269}
$$

where $\widehat { \eta } _ { 0 }$ is the $0 ^ { \mathrm { t h } }$ component of the K−dimensional vector $\widehat { \pmb { \eta } } _ { M }$ defined in (27). By construction, recall that we have

$$
\widehat { f } _ { M } = \widehat { \psi } ^ { \top } \widehat { \eta } _ { M } , \qquad t _ { M } = \widehat { \psi } ^ { \top } \pmb { \eta } _ { M } ^ { \star } , \qquad r = f - t _ { M } .\tag{270}
$$

For every realization of the full sample, using the Riesz bounds of equation (196), we always have

$$
\begin{array} { r l r } {  { \| \widehat { \pmb \theta } _ { M } - \pmb \theta _ { M } \| _ { 2 } ^ { 2 } \leq \| \widehat { \pmb \eta } _ { M } - \pmb \eta _ { M } ^ { \star } \| _ { 2 } ^ { 2 } } } \\ & { } & { \leq \displaystyle \frac { 1 } { C _ { \operatorname* { m i n } } } \| \widehat { \pmb f } _ { M } - \pmb t _ { M } \| ^ { 2 } } \\ & { } & { \leq \displaystyle \frac { 2 } { C _ { \operatorname* { m i n } } } ( \| \widehat { \pmb f } _ { M } - \pmb f \| ^ { 2 } + \| r \| ^ { 2 } ) . } \end{array}\tag{271}
$$

For each component $j ,$ recall that the rectified estimator of $f _ { j }$ is defined as

$$
\bar { f } _ { j , M } : = \frac { 1 } { \widehat { p } _ { j } } \sum _ { m = 1 } ^ { M } \widehat { \theta } _ { m } ^ { ( j ) } \phi _ { m } .\tag{272}
$$

and recall the notations

$$
g _ { j } = p _ { j } f _ { j } , \qquad h _ { j } = \sum _ { m = 1 } ^ { M } \theta _ { m } ^ { ( j ) } \phi _ { m } , \qquad \widehat { h } _ { j } = \sum _ { m = 1 } ^ { M } \widehat { \theta } _ { m } ^ { ( j ) } \phi _ { m } .\tag{273}
$$

Set $w _ { j } = f _ { j } - h _ { j } / \widehat { p } _ { j }$ in order to obtain

$$
f _ { j } - \widehat { f } _ { j , M } = w _ { j } + \frac { h _ { j } - \widehat { h } _ { j } } { \widehat { p _ { j } } } .\tag{274}
$$

Since $\kappa _ { \mathrm { m i n } } \leq p _ { j } , \widehat { p _ { j } } \leq \kappa _ { \mathrm { m a x } }$ , orthonormality of the trigonometric basis under Lebesgue measure gives

$$
\sum _ { j = 1 } ^ { d } \| f _ { j } - \bar { f } _ { j , M } \| ^ { 2 } \leq 2 \sum _ { j = 1 } ^ { d } \| w _ { j } \| ^ { 2 } + 2 \frac { \kappa _ { \operatorname* { m a x } } } { \kappa _ { \operatorname* { m i n } } ^ { 2 } } \| \widehat { \pmb { \theta } } _ { M } - \pmb { \theta } _ { M } \| _ { 2 } ^ { 2 } .\tag{275}
$$

And the upper bound on $\lVert \widehat { \pmb { \theta } } _ { M } - \pmb { \theta } _ { M } \rVert _ { 2 } ^ { 2 }$ yields finally:

$$
\sum _ { j = 1 } ^ { d } \mathbb { E } _ { p , f } \left[ \Vert f _ { j } - \bar { f } _ { j , M } \Vert ^ { 2 } \right] \leq 2 \kappa _ { \operatorname* { m a x } } \sum _ { j = 1 } ^ { d } \mathbb { E } _ { p } \left[ \Vert w _ { j } \Vert _ { L ^ { 2 } ( \lambda ) } ^ { 2 } \right] + 4 \frac { \kappa _ { \operatorname* { m a x } } } { \kappa _ { \operatorname* { m i n } } ^ { 2 } C _ { \operatorname* { m i n } } } \left( \mathbb { E } _ { p , f } \left[ \Vert \hat { f } _ { M } - f \Vert ^ { 2 } \right] + \mathbb { E } _ { p } \left[ \Vert r \Vert ^ { 2 } \right] \right) .\tag{276}
$$

Observe that this last expression involves known bounds. Indeed, we showed in subsection G.5 that

$$
\mathbb { E } _ { p } \left[ \| w _ { j } \| _ { L ^ { 2 } ( \lambda ) } ^ { 2 } \right] \le 2 R ^ { 2 } \left( \frac { M ^ { - 2 \beta } } { \kappa _ { \operatorname* { m i n } } ^ { 2 } } + \frac { \rho _ { n _ { 0 } } } { \kappa _ { \operatorname* { m i n } } ^ { 4 } } \right) , \qquad \mathbb { E } _ { p } \left[ \| r \| ^ { 2 } \right] \le 2 \kappa _ { \operatorname* { m a x } } d R ^ { 2 } \left( \frac { M ^ { - 2 \beta } } { \kappa _ { \operatorname* { m i n } } ^ { 2 } } + \frac { \rho _ { n _ { 0 } } } { \kappa _ { \operatorname* { m i n } } ^ { 4 } } \right)\tag{277}
$$

By combining the bounds and using that

$$
M \asymp n ^ { \frac { 1 } { 2 \beta + 1 } } , \qquad \rho _ { n _ { 0 } } \lesssim n ^ { - \frac { 2 \gamma } { 2 \gamma + 1 } } ,\tag{278}
$$

we obtain the desired rate in $d n ^ { - \frac { 2 \beta } { 2 \beta + 1 } } + d n ^ { - \frac { 2 \gamma } { 2 \gamma + 1 } }$

## H PROOF OF THEOREM 4.7

Let $p _ { \mathrm { u n i f } } \equiv 1$ denote the uniform density on $[ 0 , 1 ] ^ { d }$ . Since $p _ { \mathrm { u n i f } } \in \mathcal { P } _ { \gamma , L }$ , the definition of the coupled class gives

$$
\left\{ p _ { \mathrm { u n i f } } \right\} \times \mathcal { F } _ { \beta , R } ( p _ { \mathrm { u n i f } } ) \subseteq \mathcal { C } _ { \beta , \gamma } .\tag{279}
$$

For every estimator ${ \widehat { f } } ,$ restricting the supremum to this subclass yields

$$
\operatorname* { s u p } _ { ( p , f ) \in \mathcal { C } _ { \beta , \gamma } } \mathbb { E } _ { p , f } \left[ \| f - \widehat { f } \| ^ { 2 } \right] \geq \operatorname* { s u p } _ { f \in \mathcal { F } _ { \beta , R } ( p _ { \mathrm { u n i f } } ) } \mathbb { E } _ { p _ { \mathrm { u n i f } } , f } \left[ \| f - \widehat { f } \| _ { L ^ { 2 } ( \lambda _ { d } ) } ^ { 2 } \right] .\tag{280}
$$

Taking the infimum over estimators, and allowing the estimator in the restricted problem to know $p _ { \mathrm { u n i f } } ,$ gives

$$
\mathfrak { M } ( n , d , \mathcal { C } _ { \beta , \gamma } ) \geq \mathfrak { M } ( n , d , \mathcal { F } _ { \beta , R } ( p _ { \mathrm { u n i f } } ) ) .\tag{281}
$$

Under $p _ { \mathrm { u n i f } }$ , all marginal densities equal one, and hence $\psi _ { m } ^ { ( j ) } = \phi _ { m }$ . The class ${ \mathcal { F } } _ { \beta , R } ( p _ { \mathrm { u n i f } } )$ is therefore the classical additive Sobolev class (Tsybakov, 2009), with radius R for each component. Moreover, both Riesz constants equal one. The fixed-density lower bound established in Section 3 consequently implies

$$
\Re ( n , d , \mathcal { F } _ { \beta , R } ( p _ { \mathrm { u n i f } } ) ) \gtrsim d n ^ { - \frac { 2 \beta } { 2 \beta + 1 } } .\tag{282}
$$

where the underlying constant depends only on $\beta , R , \sigma .$ . Combining this inequality with (281) proves the Theorem 4.7.

## I PROOF OF THEOREM 4.9

Recall that the minimax expected prediction risk by

$$
\mathfrak { M } ( n , d , \mathcal { C } _ { \beta , \gamma } ) = \operatorname* { i n f } _ { \widehat { f } } \operatorname* { s u p } _ { ( p , f ) \in \mathcal { C } _ { \beta , \gamma } } \mathbb { E } _ { p , f } \big [ \Vert \widehat { f } - f \Vert ^ { 2 } \big ] .\tag{283}
$$

The estimator is one measurable function of the observations, used for every pair in the class. It need not be additive. Both the observation law and the norm in the loss vary with $( p , f )$ . The proof is long and technical, that is why we present in the following a sketch of proof.

We construct a finite family $\{ ( p _ { \omega } , F _ { \omega } ) \} _ { \omega } \subset \mathcal { C } _ { \beta , \gamma }$ whose regression functions are suficiently separated, while neighboring observation laws remain dificult to distinguish. The main challenge is to make this separation depend on γ while preserving the Sobolev constraint on the density-weighted components. To this end, we fix a smooth mean-zero function g with a nonzero plateau and define

$$
F _ { \omega , j } = \frac { g } { p _ { \omega , j } } , \qquad F _ { \omega } = \sum _ { j = 1 } ^ { d } F _ { \omega , j } .\tag{284}
$$

Thus $p _ { \omega , j } F _ { \omega , j } = g$ , so the weighted components satisfy the Sobolev constraint and the centering condition uniformly over the family. All variation is introduced through the design densities. These are constructed using a bounded sinusoidal coupling of localized odd bumps of width h and amplitude proportional to $h ^ { \gamma }$ . This construction preserves the joint density bounds independently of $d$ and produces γ-Hölder marginals. The dimension assumption ensures that the marginal perturbations retain the required amplitude.

Let $P _ { \omega }$ denote the design distribution and $Q _ { \omega }$ the law of one observed pair. For adjacent vertices, Lemma D.5 gives

$$
\operatorname { K L } \bigl ( Q _ { \omega } ^ { \otimes n } \lVert Q _ { \omega ^ { ( k ) } } ^ { \otimes n } \bigr ) = n \operatorname { K L } ( P _ { \omega } \lVert P _ { \omega ^ { ( k ) } } ) + \frac { n } { 2 \sigma ^ { 2 } } \lVert F _ { \omega } - F _ { \omega ^ { ( k ) } } \rVert _ { L ^ { 2 } ( P _ { \omega } ) } ^ { 2 } .\tag{285}
$$

The uniform lower density bound and Lemma D.1 control the design divergence by the squared $L ^ { 2 } ( \lambda _ { d } )$ distance between the densities. The construction makes this distance, as well as the neighboring regression distance, of order at most $h ^ { 2 \gamma + 1 }$ . Consequently, choosing $h \asymp n ^ { - \frac { 1 } { 2 \gamma + 1 } }$ and a suficiently small fixed bump amplitude constant bounds every neighboring sample divergence by $1 / 8 .$

Finally, the odd bumps and the plateau of g yield an exact afine representation of the regression family with $N \asymp d / h$ mutually orthogonal directions in $L ^ { 2 } ( \lambda _ { d } )$ , each having squared norm of order $h ^ { 2 \gamma + 1 }$ . Since $p _ { \omega } \ge \kappa _ { \mathrm { m i n } }$ the prediction loss uniformly dominates the squared $L ^ { 2 } ( \lambda _ { d } )$ loss. Applying Lemma D.4 therefore gives

$$
\mathfrak { M } ( n , d , \mathcal { C } _ { \beta , \gamma } ) \gtrsim \frac { d } { h } h ^ { 2 \gamma + 1 } \asymp d n ^ { - \frac { 2 \gamma } { 2 \gamma + 1 } } ,\tag{286}
$$

under the stated dimension assumption. This establishes the density-driven lower bound in the rough regime $\gamma < \beta$

Remark I.1 (An alternative approach via Fano’s inequality). A standard approach to minimax lower bounds is to combine a Varshamov–Gilbert packing with Fano’s inequality (see for example Tsybakov (2009)). We instead use Assouad’s lemma, which directly exploits the hypercube structure of our construction.

The same family could also support a proof based on Fano’s inequality. Writing N for the number of sign coordinates, one would first select an exponentially large subset $\Omega \subset \{ - 1 , + 1 \} ^ { N }$ whose distinct elements have Hamming distance of order N. Orthogonality would then give a squared $L ^ { 2 } ( \lambda _ { d } )$ separation of order $N v _ { h } ^ { 2 }$ between the corresponding regression functions. The additional step would be to control the sample KL divergence between arbitrary selected alternatives, rather than only between neighboring vertices. Specifically, one could establish

$$
\begin{array} { r } { \mathrm { K L } \bigl ( Q _ { \omega } ^ { \otimes n } \bigr | \bigr | Q _ { \omega ^ { \prime } } ^ { \otimes n } \bigr ) \leq C a ^ { 2 } n h ^ { 2 \gamma + 1 } \rho ( \omega , \omega ^ { \prime } ) , } \end{array}\tag{287}
$$

where $\rho$ is the Hamming distance and a is the bump-amplitude constant. With $n h ^ { 2 \gamma + 1 } \lesssim 1$ and a suficiently small, these divergences would be bounded by a small multiple of log $| \Omega | \asymp N$ , allowing Fano’s inequality to yield the same lower rate. Assouad’s lemma avoids this packing step and requires only the neighboring-divergence bounds proved below.

## I.1 Smooth functions construction

In this subsection, we show how to construct functions with γ−Hölder smoothness.

Lemma I.2. There exist a nonzero $b \in C ^ { \infty } ( ( 0 , 1 ) )$ and a smooth periodic real function w such that

$$
b ( 1 - t ) = - b ( t ) , \qquad \| b \| _ { \infty } \leq 1 , \qquad B _ { 2 } : = \int _ { 0 } ^ { 1 } b ( t ) ^ { 2 } d t > 0 ,\tag{288}
$$

and

$$
\int _ { 0 } ^ { 1 } w ( x ) d x = 0 , \qquad w ( x ) = 1 \quad o n [ 1 / 4 , 1 / 2 ] .\tag{289}
$$

For every $\begin{array} { r } { \beta > 0 , \sum _ { m > 1 } m ^ { 2 \beta } | \langle w , \phi _ { m } \rangle | ^ { 2 } < \infty } \end{array}$ . Consequently, for every $R > 0$ , there exists $\tau > 0$ such that $g = \tau w$ belongs to the Sobolev ellipsoid of radius R.

Proof. Choose a nonzero nonnegative function $\eta \in C ^ { \infty } ( ( 1 / 8 , 3 / 8 ) )$ , extended by zero to R, and define

$$
b ( t ) : = \frac { \eta ( t ) - \eta ( 1 - t ) } { \| \eta \| _ { \infty } } .\tag{290}
$$

The two terms have disjoint supports. Hence $b \in C ^ { \infty } ( ( 0 , 1 ) )$ is nonzero, $b ( 1 - t ) = - b ( t )$ , and $\| b \| _ { \infty } \leq 1$ . In particular, $B _ { 2 } > 0$

Choose a smooth cutof $\chi \in C ^ { \infty } ( ( 1 / 8 , 5 / 8 ) )$ satisfying $\chi = 1$ on $[ 1 / 4 , 1 / 2 ]$ , and a nonnegative function $\rho \in$ $C ^ { \infty } ( ( 3 / 4 , 1 ) )$ satisfying $\begin{array} { r } { \int _ { 0 } ^ { 1 } \rho ( x ) d x = 1 } \end{array}$ . Such functions exist by the standard smooth cutof construction. Define on [0, 1]

$$
w ( x ) = \chi ( x ) - \left( \int _ { 0 } ^ { 1 } \chi ( t ) d t \right) \rho ( x ) ,\tag{291}
$$

and extend w periodically. Since w vanishes in neighborhoods of both endpoints, its periodic extension is smooth. Moreover, $\begin{array} { r } { \int _ { 0 } ^ { 1 } w ( x ) d x = 0 } \end{array}$ and $w = 1$ on $[ 1 / 4 , 1 / 2 ]$ . For every integer $\ell \geq 1$ , repeated integration by part in the trigonometric Fourier coeficients gives

$$
| \langle w , \phi _ { m } \rangle | \leq C _ { \ell } m ^ { - \ell } , \qquad m \geq 1 ,\tag{292}
$$

where the boundary terms vanish by periodicity. Choosing $\ell > \beta + 1 / 2$ yields

$$
S _ { \beta } : = \sum _ { m \ge 1 } m ^ { 2 \beta } | \langle w , \phi _ { m } \rangle | ^ { 2 } < \infty .\tag{293}
$$

Since w is nonzero and has mean zero, $S _ { \beta } > 0$ . Taking $\tau = R / \sqrt { S _ { \beta } }$ proves the last assertion.

From now, we fix a univariate centered function g, which belongs to the Sobolev ellipsoid of radius R and $^ { g , }$ smoothness $\beta$ and is constant to a real value τ on the compact $[ 1 / 4 , 1 / 2 ]$

Lemma I.3. Let $0 < h \leq 1 , 0 < a \leq 1$ , and let define for all integer $k \geq 1$

$$
b _ { k } ( x ) : = b \left( \frac { x - \nu _ { k } } { h } \right) ,\tag{294}
$$

which have disjoint cells of length $h ,$ , with b from Lemma I.2, extended by zero. For arbitrary signs $\omega _ { k }$ , let define the function $u _ { \omega }$ as follows

$$
u _ { \omega } : = a h ^ { \gamma } \sum _ { k } \omega _ { k } b _ { k } .\tag{295}
$$

For each integer $\ell \geq 0$

$$
\begin{array} { r } { \| ( \sin u _ { \omega } ) ^ { ( \ell ) } \| _ { \infty } \leq C _ { \ell } a h ^ { \gamma - \ell } . } \end{array}\tag{296}
$$

We also have

$$
\operatorname* { s u p } _ { x \neq y } \frac { \big | ( \sin u _ { \omega } ) ^ { ( \lfloor \gamma \rfloor ) } ( x ) - ( \sin u _ { \omega } ) ^ { ( \lfloor \gamma \rfloor ) } ( y ) \big | } { | x - y | ^ { \gamma - \lfloor \gamma \rfloor } } \leq C _ { \gamma } a .\tag{297}
$$

These constants depend only on $b ,$ the indicated derivative order, and $\gamma _ { ; }$ and not on the number of cells or their signs.

Proof. On a single cell, write $z = a h ^ { \gamma } \omega _ { k }$ . For $| z | \le 1$

$$
\sin ( z b ( t ) ) = \int _ { 0 } ^ { z } b ( t ) \cos ( v b ( t ) ) d v .\tag{298}
$$

Every derivative with respect to t of the integrand is uniformly bounded for $| v | \leq 1$ , because b is smooth with compact support. Diferentiation under this finite integral therefore bounds its ℓth derivative by $C _ { \ell } | z |$ . Rescaling $t = ( x - \nu _ { k } ) / h$ multiplies this bound by $h ^ { - \ell }$ . All derivatives vanish at cell boundaries, and supports are disjoint, so the same supremum bound holds globally. For noninteger $\gamma ,$ put $r = \lfloor \gamma \rfloor$ and $\delta = \gamma - r \in ( 0 , 1 )$ . The derivative of order r has both supremum bound $C a h ^ { \delta }$ and Lipschitz constant $C a h ^ { \delta - 1 }$ . Consequently,

$$
| ( \sin u _ { \omega } ) ^ { ( r ) } ( x ) - ( \sin u _ { \omega } ) ^ { ( r ) } ( y ) | \leq C a \operatorname* { m i n } \{ h ^ { \delta } , h ^ { \delta - 1 } | x - y | \} \leq C a | x - y | ^ { \delta } .\tag{299}
$$

The last inequality follows separately from $| x - y | \geq h$ and $| x - y | \le h$ . This proves the bound. At integer γ, the already proved derivative estimate of order $\gamma$ is $C a$ , so its oscillation is at most $2 C a$ □

## I.2 Construction of alternatives

In this subsection, we explicitly construct a finite subfamil $\{ ( p _ { \omega } , F _ { \omega } ) \} _ { \omega } \subset \mathcal { C } _ { \beta , \gamma }$ built on the previous subsection. Lemma I.4 (Sine coupling and its exact marginals). Let $u _ { 1 } , \ldots , u _ { s }$ be real functions on [0, 1] such that $\scriptstyle \int$ sin $u _ { j } = 0$ and R cos $u _ { j } = q$ for every $j$ . For $s \leq d$ and $0 < \varepsilon < 1$ , set

$$
p ( \mathbf { x } ) = 1 + \varepsilon \sin \left( \sum _ { j = 1 } ^ { s } u _ { j } ( x _ { j } ) \right) .\tag{300}
$$

Then $p$ is a density with $1 - \varepsilon \le p \le 1 + \varepsilon$ , and its marginals are

$$
\begin{array} { r } { \left\{ p _ { j } = 1 + \varepsilon q ^ { s - 1 } \sin u _ { j } , \qquad ( j \leq s ) , \right. } \\ { p _ { j } = 1 , \qquad ( j > s ) . } \end{array}\tag{301}
$$

Furthermore, $i f$ we have $1 \ge q \ge 1 - A a ^ { 2 } h ^ { 2 \gamma } , A a ^ { 2 } h ^ { 2 \gamma } \le 1 / 2$ , and $s h ^ { 2 \gamma } \leq 1$ , then

$$
\varepsilon e ^ { - 2 A a ^ { 2 } } \leq \varepsilon q ^ { s - 1 } \leq \varepsilon .
$$

Proof. First, Fubini’s theorem and the identity exp(i $\begin{array} { r } { \sum _ { j } u _ { j } ) = \prod _ { j } \exp ( \mathrm { i } u _ { j } ) } \end{array}$ give

$$
\int _ { [ 0 , 1 ] ^ { d } } \exp \left( { \mathrm { i } } \sum _ { j = 1 } ^ { s } u _ { j } ( x _ { j } ) \right) d \mathbf { x } = q ^ { s } ,\tag{302}
$$

which is real. Its imaginary part is zero, so $\textstyle \int p = 1$

Second, the pointwise bounds follow from $| \sin | \leq 1$

Third, integrating over all coordinates except an active coordinate j gives imaginary part $q ^ { s - 1 }$ sin $u _ { j } ( x )$ ; integration over all active coordinates gives zero for an inactive marginal.

Finally, log $( 1 - z ) \geq - 2 z \ \mathrm { o n } \ [ 0 , 1 / 2 ] \colon$ : its diference from −2z has derivative $( 1 - 2 z ) / ( 1 - z ) \geq 0$ and value zero at zero. Apply this inequality to $z = 1 - q$ to obtain $q ^ { s - 1 } \geq \exp ( - 2 A a ^ { 2 } ( s - 1 ) h ^ { 2 \gamma } ) \geq e ^ { - 2 A a ^ { 2 } }$ □

In what follows, we fix $b , w , \tau , g = \tau w$ from Lemma I.2. In particular, $g = \tau \ \mathrm { o n } \ [ 1 / 4 , 1 / 2 ] , \int g = 0$ , and $g$ is in the Sobolev ellipsoid of radius R. Choose once and for all

$$
0 < \varepsilon < \operatorname* { m i n } \{ 1 / 2 , 1 - \kappa _ { \operatorname* { m i n } } , \kappa _ { \operatorname* { m a x } } - 1 \} .\tag{303}
$$

For integers $n , d \geq 1$ , set

$$
M = \lceil n ^ { 1 / ( 2 \gamma + 1 ) } \rceil , \qquad h = \frac { 1 } { 4 M } , \qquad s = \operatorname* { m i n } \{ d , \lfloor h ^ { - 2 \gamma } \rfloor \} .\tag{304}
$$

Here $s \ge 1 , M h = 1 / 4$ , and $s h ^ { 2 \gamma } \leq 1$ . For any integer $k \geq 1$ , let define

$$
\nu _ { k } : = \frac { 1 } { 4 } + ( k - 1 ) h ,\tag{305}
$$

and set the function

$$
b _ { k } ( \cdot ) : = b \left( \frac { \cdot - \nu _ { k } } { h } \right) , \qquad 1 \leq k \leq M ,\tag{306}
$$

with zero extensions. All cells lie inside the plateau interval. Observe that if we define the bump supports $\boldsymbol { B } _ { k }$ as follows

$$
\mathcal { B } _ { k } : = \left[ \frac { 1 } { 4 } \left( \frac { k - 1 } { M } + 1 \right) , \frac { 1 } { 4 } \left( \frac { k } { M } + 1 \right) \right[ ,\tag{307}
$$

we have $\begin{array} { r } { \frac { x - \nu _ { k } } { h } \in [ 0 , 1 [ \Longleftrightarrow \ x \in B _ { k } } \end{array}$ which implies that for any k, the functions $b _ { k }$ reproduce the same function b on disjoint intervals $\boldsymbol { B } _ { k }$ and for any $x \notin B _ { k } \implies b _ { k } ( x ) = 0$

For a constant $a \in ( 0 , 1 ]$ to be selected below, and for a binary matrix $\omega \in \{ - 1 , 1 \} ^ { s \times M }$ , define

$$
u _ { \omega , j } ( x ) = a h ^ { \gamma } \sum _ { k = 1 } ^ { M } \omega _ { j k } b _ { k } ( x ) , \qquad p _ { \omega } ( \mathbf x ) : = 1 + \varepsilon \sin \left( \sum _ { j = 1 } ^ { s } u _ { \omega , j } ( x _ { j } ) \right) .\tag{308}
$$

Lemma I.5. We have the two following key results for all j:

$$
\begin{array} { r c l } { \displaystyle \int _ { [ 0 , 1 ] } \sin \big ( u _ { \omega , j } ( x ) \big ) \ d x } & { = } & { 0 , } \\ { \displaystyle \int _ { 0 } ^ { 1 } \cos \big ( u _ { \omega , j } ( x ) \big ) \ d x } & { = } & { \underbrace { 1 - \frac { 1 } { 4 } \int _ { 0 } ^ { 1 } [ 1 - \cos \bigl ( a h ^ { \gamma } b ( t ) \bigr ) ] \ d t } _ { : = q _ { h } } . } \end{array}\tag{309}
$$

(310)

Proof. Let ω be a vector of signs and define

$$
u _ { \omega } : = a h ^ { \gamma } \sum _ { k = 1 } ^ { M } \omega _ { k } b _ { k } .\tag{311}
$$

Observe that $u _ { \omega }$ can be written as follows, using the bumps $\boldsymbol { B } _ { k }$ :

$$
u _ { \omega } ( x ) = a h ^ { \gamma } \sum _ { k = 1 } ^ { M } \omega _ { k } b _ { k } ( x ) \mathbf { 1 } _ { x \in \mathcal { B } _ { k } } .\tag{312}
$$

Using that the $\boldsymbol { B } _ { k }$ are disjoint and sin $0 = 0$ , we have

$$
\sin ( u _ { \omega } ( x ) ) = \sum _ { k = 1 } ^ { M } \omega _ { k } \sin \left( a h ^ { \gamma } b _ { k } ( x ) \right) \mathbf { 1 } _ { x \in \mathcal { B } _ { k } } .\tag{313}
$$

Observe that the function b is defined on $[ 0 , 1 ]$ , centered at $1 / 2$ and odd. Indeed, using $b ( 1 - t ) = - b ( t )$ we have $b ( 1 / 2 ) = 0$ and we also have $b ( x + 1 / 2 ) = b ( 1 - ( 1 / 2 - x ) ) = - b ( 1 / 2 - x )$ . By translation, for each $k ,$ the function $b _ { k }$ is odd on the bump $\boldsymbol { B } _ { k }$ and centered on its middle, which gives the first equality.

Moreover, we have

$$
\cos ( u _ { \omega } ( x ) ) = \sum _ { k = 1 } ^ { M } \cos \left( a h ^ { \gamma } b _ { k } ( x ) \right) \mathbf { 1 } _ { x \in \mathcal { B } _ { k } } + \prod _ { k = 1 } ^ { M } \mathbf { 1 } _ { x \not \in \mathcal { B } _ { k } } .\tag{314}
$$

In plain word, if $x \in B _ { k }$ , the cosine equals cos $( a h ^ { \gamma } b _ { k } ( x ) )$ , else it equals 1. Observe that

$$
B _ { 1 } \cup \cdots \cup B _ { M } = \left[ \frac { 1 } { 4 } , \frac { 1 } { 2 } \right[ ,\tag{315}
$$

so the volume of the set where the function equals 1 is $3 / 4$ , which yields

$$
\int _ { [ 0 , 1 ] } \cos ( u _ { \omega } ( x ) ) d x = \frac { 3 } { 4 } + \sum _ { k = 1 } ^ { M } \int _ { { \mathcal { B } } _ { k } } \cos \left( a h ^ { \gamma } b _ { k } ( x ) \right) d x .\tag{316}
$$

Computing the integral $\begin{array} { r } { \int _ { B _ { k } } \cos { \left( a h ^ { \gamma } b _ { k } ( x ) \right) } } \end{array}$ dx is explicit by translating to $[ 0 , 1 ]$ , which leads to the result.

The basic inequality $1 - \cos z \leq z ^ { 2 } / 2$ easily gives

$$
0 \leq 1 - q _ { h } \leq \frac { a ^ { 2 } B _ { 2 } } { 8 } h ^ { 2 \gamma } \leq \frac { 1 } { 8 } .\tag{317}
$$

Applying Lemma $\mathrm { I . 4 } ,$ we obtain valid densities with the required strict joint bounds and exact marginals

$$
p _ { \omega , j } ( x ) = \left\{ \begin{array} { l l } { 1 + t _ { h } \sin u _ { \omega , j } ( x ) , } & { j \leq s , } \\ { 1 , } & { j > s , } \end{array} \right. \quad t _ { h } : = \varepsilon q _ { h } ^ { s - 1 } , \quad \quad \varepsilon e ^ { - a ^ { 2 } B _ { 2 } / 4 } \leq t _ { h } \leq \varepsilon .\tag{318}
$$

Lemma I.3 gives the following uniform Hölder bound

$$
\forall x , y , \quad \left| p _ { \omega , j } ^ { ( \lfloor \gamma \rfloor ) } ( x ) - p _ { \omega , j } ^ { ( \lfloor \gamma \rfloor ) } ( y ) \right| \le C _ { \gamma } \varepsilon a | x - y | ^ { \gamma - \lfloor \gamma \rfloor } ,\tag{319}
$$

where the constant $C _ { \gamma }$ is fixed from now. Select $a > 0$ small enough that this is at most $L ;$ the further restriction needed for testing will be imposed below. Thus every marginal belongs to $\mathcal { H } ^ { \gamma } ( L )$

Define the regression components by

$$
f _ { \omega , j } = g / p _ { \omega , j } \quad ( j \leq s ) , \qquad f _ { \omega , j } = 0 \quad ( j > s ) , \qquad f _ { \omega } = \sum _ { j = 1 } ^ { d } f _ { \omega , j } .\tag{320}
$$

For each active coordinate $j ,$ we have $p _ { \omega , j } f _ { \omega , j } = g$ and

$$
\int _ { 0 } ^ { 1 } f _ { \omega , j } ( x ) p _ { \omega , j } ( x ) d x = \int _ { 0 } ^ { 1 } g ( x ) d x = 0 .\tag{321}
$$

Inactive weighted components are zero. All weighted components therefore satisfy the Sobolev constraint, and every $( p _ { \omega } , f _ { \omega } )$ lies in $\mathcal { C } _ { \beta , \gamma }$ . The weighted signals remain fixed across the entire subfamily.

## I.3 Orthogonal separation of the regression functions

Let define for all $x \in [ 0 , 1 ]$ and $k \geq 1$ the following functions

$$
z _ { k } ( x ) : = \sin ( a h ^ { \gamma } b _ { k } ( x ) ) , \qquad H _ { k } ( x ) : = { \frac { \tau t _ { h } z _ { k } ( x ) } { 1 - t _ { h } ^ { 2 } z _ { k } ( x ) ^ { 2 } } } .\tag{322}
$$

Only one bump can be nonzero at any x. On its support $g = \tau .$ , and identity $a ^ { 2 } - b ^ { 2 } = ( a - b ) ( a + b )$ gives

$$
\frac { 1 } { 1 + t _ { h } \omega _ { j k } z _ { k } } = \frac { 1 } { 1 - t _ { h } ^ { 2 } z _ { k } ^ { 2 } } - \frac { \omega _ { j k } t _ { h } z _ { k } } { 1 - t _ { h } ^ { 2 } z _ { k } ^ { 2 } } ,\tag{323}
$$

which implies for all $\mathbf { x } \in [ 0 , 1 ] ^ { d }$

$$
f _ { \omega } ( \mathbf { x } ) = F _ { 0 } ( \mathbf { x } ) - \sum _ { j = 1 } ^ { s } \sum _ { k = 1 } ^ { M } \omega _ { j k } H _ { k } ( x _ { j } ) ,\tag{324}
$$

where the sign-independent function is explicitly

$$
F _ { 0 } ( \mathbf { x } ) : = \sum _ { j = 1 } ^ { s } \left[ g ( x _ { j } ) + \tau \sum _ { k = 1 } ^ { M } \frac { t _ { h } ^ { 2 } z _ { k } ( x _ { j } ) ^ { 2 } } { 1 - t _ { h } ^ { 2 } z _ { k } ( x _ { j } ) ^ { 2 } } \right] .\tag{325}
$$

The map $z \mapsto \tau t _ { h } z / ( 1 - t _ { h } ^ { 2 } z ^ { 2 } )$ is odd. Therefore, each $H _ { k }$ is antisymmetric about the midpoint of its cell and satisfies

$$
\int _ { 0 } ^ { 1 } H _ { k } ( x ) d x = 0 .\tag{326}
$$

The lifted functions $\mathbf { x } \mapsto H _ { k } ( x _ { j } )$ , indexed by $1 \leq j \leq s$ and $1 \leq k \leq M$ , are pairwise orthogonal in $L ^ { 2 } ( \lambda _ { d } )$ Indeed, for a fixed coordinate $j ,$ distinct functions $H _ { k } ( x _ { j } )$ have disjoint supports. For distinct coordinates $j \neq j ^ { \prime }$ Fubini’s theorem gives

$$
\int _ { [ 0 , 1 ] ^ { d } } H _ { k } ( x _ { j } ) H _ { k ^ { \prime } } ( x _ { j ^ { \prime } } ) d \mathbf { x } = \left( \int _ { 0 } ^ { 1 } H _ { k } ( t ) d t \right) \left( \int _ { 0 } ^ { 1 } H _ { k ^ { \prime } } ( t ) d t \right) = 0 .\tag{327}
$$

Their squared norms are independent of k and are equal to

$$
v _ { h } ^ { 2 } : = \tau ^ { 2 } t _ { h } ^ { 2 } h \int _ { 0 } ^ { 1 } \frac { \sin ^ { 2 } ( a h ^ { \gamma } b ( t ) ) } { ( 1 - t _ { h } ^ { 2 } \sin ^ { 2 } ( a h ^ { \gamma } b ( t ) ) ) ^ { 2 } } d t .\tag{328}
$$

Since $| a h ^ { \gamma } b | \leq 1$ , the inequalities | sin z $| \geq | z | / 2$ for $| z | \le 1$ and $| \sin z | \le | z |$ give

$$
\frac { \tau ^ { 2 } \varepsilon ^ { 2 } e ^ { - B _ { 2 } / 2 } a ^ { 2 } B _ { 2 } } { 4 } h ^ { 2 \gamma + 1 } \leq v _ { h } ^ { 2 } \leq \frac { \tau ^ { 2 } \varepsilon ^ { 2 } a ^ { 2 } B _ { 2 } } { ( 1 - \varepsilon ^ { 2 } ) ^ { 2 } } h ^ { 2 \gamma + 1 } .\tag{329}
$$

For clarity, the lower sine inequality follows by integrating cos $t \geq$ cos $1 > 1 / 2$ between 0 and $| z | ;$ the upper one follows by integrating | cos $t | \le 1$ . The lower exponential factor in (329) uses $a \leq 1$ and the lower bound for $t _ { h }$ in (318).

## I.4 Information in the covariates and the responses

Let $Q _ { \omega }$ denote the law of one observed pair under $( p _ { \omega } , f _ { \omega } )$ . Suppose $\omega ^ { \prime }$ difers from ω only at $( j , k )$ . The sine function is 1-Lipschitz, and hence

$$
\left| p _ { \omega } ( \mathbf { x } ) - p _ { \omega ^ { \prime } } ( \mathbf { x } ) \right| = \left| \varepsilon \sin \left( \sum _ { j = 1 } ^ { s } u _ { \omega , j } ( x _ { j } ) \right) - \varepsilon \sin \left( \sum _ { j = 1 } ^ { s } u _ { \omega ^ { \prime } , j } ( x _ { j } ) \right) \right| ,\tag{330}
$$

$$
= \varepsilon \left| \sin \left( \sum _ { j = 1 } ^ { s } u _ { \omega , j } ( x _ { j } ) \right) - \sin \left( \sum _ { j = 1 } ^ { s } u _ { \omega ^ { \prime } , j } ( x _ { j } ) \right) \right| ,\tag{331}
$$

$$
\leq \varepsilon | | \sum _ { j = 1 } ^ { s } ( u _ { \omega , j } ( x _ { j } ) - u _ { \omega ^ { \prime } , j } ( x _ { j } ) ) | ,\tag{332}
$$

$$
\leq \varepsilon \left| \sum _ { j = 1 } ^ { s } \left( a h ^ { \gamma } \sum _ { k = 1 } ^ { M } \omega _ { j k } b _ { k } ( x _ { j } ) - a h ^ { \gamma } \sum _ { k = 1 } ^ { M } \omega _ { j k } ^ { \prime } b _ { k } ( x _ { j } ) \right) \right| ,\tag{333}
$$

$$
\begin{array} { r l } & { \leq \varepsilon a h ^ { \gamma } \left| \omega _ { j k } - \omega _ { j k } ^ { \prime } \right| \left| b _ { k } ( x _ { j } ) \right| , } \\ & { \leq 2 \varepsilon a h ^ { \gamma } | b _ { k } ( x _ { j } ) | . } \end{array}\tag{334}
$$

(335)

Since $p _ { \omega ^ { \prime } } \geq 1 - \varepsilon .$ , Lemma D.1 yields

$$
\mathrm { K L } ( P _ { \omega } \| P _ { \omega ^ { \prime } } ) \leq \int _ { [ 0 , 1 ] ^ { d } } \frac { \left( p _ { \omega } ( \mathbf { x } ) - p _ { \omega ^ { \prime } } ( \mathbf { x } ) \right) ^ { 2 } } { p _ { \omega ^ { \prime } } ( \mathbf { x } ) } d \lambda _ { d } ( \mathbf { x } ) ,\tag{336}
$$

$$
\leq \int _ { B _ { k } } { \frac { \left( 2 \varepsilon a h ^ { \gamma } b _ { k } ( x ) \right) ^ { 2 } } { 1 - \varepsilon } } d x ,\tag{337}
$$

$$
\leq \frac { 4 \varepsilon ^ { 2 } a ^ { 2 } B _ { 2 } } { 1 - \varepsilon } h ^ { 2 \gamma + 1 } .\tag{338}
$$

Moreover, (324) gives exactly $f _ { \omega } - f _ { \omega ^ { \prime } } = \pm 2 H _ { k } ( x _ { j } )$ . Thus

$$
\| f _ { \omega } - f _ { \omega ^ { \prime } } \| _ { L ^ { 2 } ( P _ { \omega } ) } ^ { 2 } \leq 4 \kappa _ { \operatorname* { m a x } } v _ { h } ^ { 2 } .\tag{339}
$$

The Gaussian KL identity in Lemma D.5 and the previous inequalities give

$$
\mathrm { K L } ( Q _ { \omega } ^ { \otimes n } | | Q _ { \omega ^ { \prime } } ^ { \otimes n } ) \leq \frac { 4 n \varepsilon ^ { 2 } a ^ { 2 } B _ { 2 } } { 1 - \varepsilon } h ^ { 2 \gamma + 1 } + \frac { 2 n \kappa _ { \mathrm { m a x } } v _ { h } ^ { 2 } } { \sigma ^ { 2 } } .\tag{340}
$$

Then, the inequality (329) gives

$$
\mathrm { K L } ( Q _ { \omega } ^ { \otimes n } \| Q _ { \omega ^ { \prime } } ^ { \otimes n } ) \leq a ^ { 2 } n h ^ { 2 \gamma + 1 } \underbrace { \left[ { \frac { 4 \varepsilon ^ { 2 } B _ { 2 } } { 1 - \varepsilon } } + { \frac { 2 \kappa _ { \operatorname* { m a x } } \tau ^ { 2 } \varepsilon ^ { 2 } B _ { 2 } } { \sigma ^ { 2 } ( 1 - \varepsilon ^ { 2 } ) ^ { 2 } } } \right] } _ { C }\tag{341}
$$

$$
\leq C n h ^ { 2 \gamma + 1 } a ^ { 2 } .\tag{342}
$$

Moreover, by the choice of M and the definition of $h ,$ we have

$$
\begin{array} { l } { n h ^ { 2 \gamma + 1 } = n \left( \displaystyle \frac { 1 } { 4 M } \right) ^ { 2 \gamma + 1 } } \\ { \leq 4 ^ { - ( 2 \gamma + 1 ) } \times n \times n ^ { - \frac { 2 \gamma + 1 } { 2 \gamma + 1 } } } \\ { \leq 1 } \end{array}
$$

All quantities inside the brackets are fixed constants. We finally select a as follows

$$
0 < a \leq \operatorname* { m i n } \left\{ 1 , \frac { L } { C _ { \gamma } \varepsilon } , \frac { 1 } { \sqrt { 8 C } } \right\} ,\tag{343}
$$

which ensures that

$$
\mathrm { K L } ( Q _ { \omega } ^ { \otimes n } | | Q _ { \omega ^ { \prime } } ^ { \otimes n } ) \leq \frac { 1 } { 8 } .\tag{344}
$$

## I.5 Assouad’s argument for the rough-density contribution

For every estimator and vertex, the uniform bounds on p give

$$
\| \widehat { f } - f _ { \omega } \| _ { L ^ { 2 } ( P _ { \omega } ) } ^ { 2 } \geq \kappa _ { \operatorname* { m i n } } \| \widehat { f } - f _ { \omega } \| _ { L ^ { 2 } ( \lambda _ { d } ) } ^ { 2 } .\tag{345}
$$

If an estimator has infinite risk at a vertex, the desired lower bound already holds. Otherwise it can be regarded, up to null sets, as an $L ^ { 2 } ( \lambda _ { d } )$ -valued estimator on this finite family. The observation laws here are mutually absolutely continuous, so the finitely many exceptional null sets can be removed simultaneously.

Apply Lemma D.4 to (324), using directions $- H _ { k } ( x _ { j } ) , N = s M$ , and $\alpha = 1 / 8$ . Restricting the supremum to this finite admissible family gives

$$
\mathfrak { M } ( n , d , \mathcal { C } _ { \beta , \gamma } ) \geq \frac { 3 \kappa _ { \operatorname* { m i n } } } { 8 } s M v _ { h } ^ { 2 } \geq c s h ^ { 2 \gamma } ,
$$

where $M h = 1 / 4$ and (329) were used. For $x \geq 1 , \ \lfloor x \rfloor \geq x / 2$ . Therefore we have

$$
s h ^ { 2 \gamma } \geq \frac { 1 } { 2 } \operatorname* { m i n } \{ d h ^ { 2 \gamma } , 1 \} .\tag{346}
$$

Finally, $n ^ { \frac { 1 } { 2 \gamma + 1 } } \leq M \leq 2 n ^ { \frac { 1 } { 2 \gamma + 1 } }$ for every $n \geq 1$ , so

$$
8 ^ { - 2 \gamma } n ^ { - \frac { 2 \gamma } { 2 \gamma + 1 } } \leq h ^ { 2 \gamma } \leq 4 ^ { - 2 \gamma } n ^ { - \frac { 2 \gamma } { 2 \gamma + 1 } } .
$$

Combining the last three displays proves, for all integers $n , d \geq 1$

$$
\mathfrak { M } ( n , d , \mathcal { C } _ { \beta , \gamma } ) \gtrsim \operatorname* { m i n } \{ d n ^ { - \frac { 2 \gamma } { 2 \gamma + 1 } } , 1 \} .\tag{347}
$$

A suficient condition to obtain the desired rate of $\mathfrak { M } ( n , d , \mathcal { C } _ { \beta , \gamma } ) \gtrsim d n ^ { - \frac { 2 \gamma } { 2 \gamma + 1 } }$ is having $d n ^ { - \frac { 2 \gamma } { 2 \gamma + 1 } } \leq 1$ for n large enough, so it sufices to assume

$$
d = o \left( n ^ { \frac { 2 \gamma } { 2 \gamma + 1 } } \right) ,\tag{348}
$$

which leads to the desired rate and tends to 0.