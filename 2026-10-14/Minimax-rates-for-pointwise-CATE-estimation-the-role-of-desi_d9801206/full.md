# Minimax rates for pointwise CATE estimation: the role of design smoothness

Yuxuan Hou<sup>1</sup> and Zhenghan Li<sup>2</sup>

<sup>1</sup>Qiuzhen College, Tsinghua University <sup>2</sup>School of Statistics, University of Minnesota

October 14, 2026

## Abstract

We study the role of design smoothness in the minimax estimation of conditional average treatment efects at a fixed covariate point. For design densities bounded above and away from zero, without further regularity, we derive upper and lower risk bounds with an arbitrarily small gap in the convergence-rate exponent. Under uniform design, we obtain matching upper and lower bounds at interior points in the regime of low nuisance smoothness. These results show that rough covariate designs can lead to slower rates than uniform designs, even under the same smoothness assumptions on the treatment efect and nuisance functions, highlighting the role of the covariate distribution in pointwise causal estimation.

Keywords: conditional average treatment efect; minimax rate; nonparametric estimation; causal inference.

MSC2020 subject classifications: Primary 62G08, 62C20; secondary 62G20.

## 1 Introduction

The conditional average treatment efect (CATE) describes how the average efect of a treatment varies with observed covariates. We study its estimation at a fixed covariate value, with accuracy measured by mean absolute error. Existing approaches include matching [1, 2, 4, 5, 9, 12] and doubly robust or orthogonal estimation [3, 6, 8, 10, 11]. Our focus is on how the covariate distribution afects the accuracy attainable by any estimator.

For pointwise CATE estimation over Hölder classes, Kennedy et al. [7] established minimax lower bounds and constructed a higher-order local polynomial R-learner that attains these bounds under suitable conditions on the design density. Their lower-bound construction uses a design density that vanishes on parts of the covariate domain. This leaves open whether the same lower bound holds under a uniform density. A second question is what rate is attainable when the density is only bounded above and away from zero, without the regularity needed for accurate density estimation.

Under uniform design, we establish the lower bound of Kennedy et al. [7] at interior covariate points in the regime of low nuisance smoothness. The nuisance functions are the conditional treatment probability and the control outcome regression. We also derive a matching upper bound under conditions on the nuisance estimators and the accuracy of design density estimation, and verify these conditions with high probability under uniform design. For rough design, where the density is bounded above and away from zero without smoothness restrictions, we give upper and lower risk bounds with an arbitrarily small gap in the convergence-rate exponent. These bounds identify low-smoothness regimes in which a rough design slows pointwise estimation.

Our conditional upper bound follows the higher-order local polynomial R-learner framework of Kennedy et al. [7]. The construction combines local polynomial estimation with a second-order projection correction based on a design density estimated from independent training data. For rough design, we use weighted contrasts of nearby observations to cancel low-order polynomials without estimating the density.

Section 2 introduces the model and the two design settings. We present the rough-design bounds in Section 3 and the uniform-design results in Section 4. Section 5 discusses extensions.

## 2 Setup

## 2.1 Model and parameter classes

We estimate the conditional mean diference $\tau ( x ) = \mu _ { 1 } ( x ) - \mu _ { 0 } ( x )$ , where $\mu _ { a } ( x ) = \operatorname { \mathbb { E } } ( Y \mid X =$ $x , A = a )$ for $a \in \{ 0 , 1 \}$ . Under consistency, conditional exchangeability, and the overlap condition below, this conditional mean diference identifies the CATE.

Hölder classes. For $\zeta > 0$ let $\underline { { \zeta } } = \lceil \zeta \rceil - 1$ and $\theta _ { \zeta } = \zeta - \underline { { \zeta } } \in ( 0 , 1 ]$ . For $g : [ 0 , 1 ] ^ { d } \to \mathbb { R }$ let

$$
\| g \| _ { H ^ { \zeta } } = \operatorname* { m a x } _ { | a | \leq \underline { { \zeta } } } \| \partial ^ { a } g \| _ { \infty } + \operatorname* { m a x } _ { | a | = \underline { { \zeta } } } \operatorname* { s u p } _ { x \neq y } \frac { | \partial ^ { a } g ( x ) - \partial ^ { a } g ( y ) | } { \| x - y \| ^ { \theta _ { \zeta } } } ,
$$

so that for integer $\zeta$ the top derivative is Lipschitz, and let $H ^ { \zeta }$ be the set of $g$ with $\| g \| _ { H ^ { \zeta } } < \infty$ Here $a \in  { \mathbb { N } } _ { 0 } ^ { d }$ is a multi-index, $\begin{array} { r } { | a | = \sum _ { j } a _ { j } , a ! = \prod _ { j } a _ { j } ! } \end{array}$ , and $\partial ^ { a }$ denotes the corresponding partial derivative. The norm $\lVert \cdot \rVert$ on $\mathbb { R } ^ { d }$ is Euclidean, and $\left\| \cdot \right\| _ { \infty }$ is the supremum norm. Write $H ^ { \zeta } ( L ) = \{ g$ $\| g \| _ { H ^ { \zeta } } \le L \}$ for the ball of radius L.

Model. We observe independent and identically distributed (i.i.d.) copies of $Z = ( X , A , Y )$ with $X \in [ 0 , 1 ] ^ { d } , A \in \{ 0 , 1 \}$ , and

$$
Y = \mu _ { 0 } ( X ) + A \tau ( X ) + \varepsilon , \qquad \mathbb { E } ( \varepsilon \mid X , A ) = 0 , \qquad \pi ( x ) = \mathbb { P } ( A = 1 \mid X = x ) .
$$

The covariate X has density f. We assume $\pi \in H ^ { \alpha } , \mu _ { 0 } \in H ^ { \beta }$ , and $\tau \in H ^ { \gamma }$ , where $\alpha , \beta , \gamma > 0$ . The bounds on the Hölder norms and the outcome moments are specified for each model class below. Define

$$
s = \frac { \alpha + \beta } 2 , \qquad s _ { 1 } = \frac { ( \alpha \wedge 1 ) + ( \beta \wedge 1 ) } 2 , \qquad \bar { \alpha } = \operatorname * { m a x } \{ \alpha , \beta \} ,
$$

and

$$
\rho _ { \mathrm { T S } } = { \frac { 2 ( s + s _ { 1 } ) } { d + 4 s _ { 1 } + { \frac { 2 d s _ { 1 } } { \gamma } } } } , \qquad \rho _ { \mathrm { S } } = { \frac { 1 } { 1 + d / ( 2 \gamma ) + d / ( 4 s ) } } .
$$

The low-smoothness regime is $s < d \gamma / ( 4 \gamma + 2 d )$ . The subscript TS denotes the two-scale construction used for rough design.

## 2.2 Two design settings: rough and uniform

The rough-design class. For rough design, fix density bounds $0 < f \leq 1 \leq \overline { { f } } < \infty$ and assume $\underline { { f } } \le f \le \overline { { f } }$ almost everywhere on $[ 0 , 1 ] ^ { d }$ . No smoothness is imposed on $f .$ Let $\mathcal { P } _ { \mathrm { r o u g h } } ( \alpha , \beta , \gamma )$ denote the laws in the model above satisfying this density bound and

$$
\begin{array} { r l } & { \epsilon \leq \pi \leq 1 - \epsilon , \qquad \pi \in H ^ { \alpha } ( L _ { \pi } ) , \quad \mu _ { 0 } \in H ^ { \beta } ( L _ { \mu } ) , \quad \tau \in H ^ { \gamma } ( L _ { \tau } ) , } \\ & { \qquad \quad \displaystyle \mathrm { e s s s u p } { \mathbb { E } } ( Y ^ { 2 } \mid X = x , A = a ) \leq M _ { 2 } . } \end{array}
$$

Here $\epsilon \in ( 0 , 1 / 2 )$ is the overlap constant, which bounds treatment probabilities away from zero and one. The constants $f , \overline { { { f } } } , L _ { \pi } , L _ { \mu } , L _ { \tau } , M _ { 2 }$ are positive and fixed as n varies; their dependence is suppressed in the class notation.

The uniform-design setting. We also consider the uniform design, with $f \equiv 1 \mathrm { ~ o n ~ } [ 0 , 1 ] ^ { d }$ . The estimator is constructed without using the uniformity assumption.

## 3 Rough design

We work over $\mathcal { P } _ { \mathrm { r o u g h } } ( \alpha , \beta , \gamma )$ . We first construct an estimator and state its risk bound, then give the corresponding minimax lower bound.

## 3.1 Upper bound

Stencil construction. Fix an integer $q > \alpha + \beta + \gamma + 1$ and set $D _ { q } = { \binom { d + q } { q } }$ and $m _ { \mathrm { s t } } = D _ { q } + 1$ Choose $D _ { q }$ points $z _ { 1 } = 0 , z _ { 3 } , . . . , z _ { m _ { \mathrm { s t } } }$ at which evaluation uniquely determines every polynomial of total degree at most $q .$ Choose suficiently small neighborhoods $U _ { i }$ of $z _ { i } , 3 \leq i \leq m _ { \mathrm { s t } }$ , so that this property holds for every selection of one point from each box together with the origin.

Split the n observations $Z _ { i } = ( X _ { i } , A _ { i } , Y _ { i } )$ into $m _ { \mathrm { s t } }$ disjoint groups $I _ { 1 } , \ldots , I _ { m _ { \mathrm { s t } } }$ of approximately equal size. For a kernel K of one observation from each group, define the product-sample average

$$
\mathbb { U } _ { n , m _ { \mathrm { s t } } } K = \frac { 1 } { \prod _ { j = 1 } ^ { m _ { \mathrm { s t } } } \left| I _ { j } \right| } \sum _ { i _ { j } \in I _ { j } , 1 \le j \le m _ { \mathrm { s t } } } K ( Z _ { i _ { 1 } } , . . . , Z _ { i _ { m _ { \mathrm { s t } } } } )
$$

Write $B ( x , t ) = [ 0 , 1 ] ^ { d } \cap \{ u : \| u - x \| _ { \infty } \leq t \}$ . Within this subsection, $\widehat { Q } , \varphi _ { h }$ , and the sample subsets $I _ { j }$ refer to the rough-design estimator.

An ordered tuple $( x _ { 1 } , \ldots , x _ { m _ { \mathrm { s t } } } )$ belongs to the stencil event $\mathcal { C } _ { h , \ell , r } ( x _ { 0 } )$ if

$$
\begin{array} { l } { x _ { 1 } \in B ( x _ { 0 } , h ) } \\ { x _ { 2 } \in B ( x _ { 1 } , \ell ) } \\ { x _ { i } \in x _ { 1 } + r U _ { i } \quad 3 \leq i \leq m _ { \mathrm { s t } } } \end{array}
$$

For an accepted tuple, let $V ( x _ { 1 : m _ { \mathrm { s t } } } )$ be the $D _ { q } \times m _ { \mathrm { s t } }$ polynomial evaluation matrix. Its rows are indexed by monomials of total degree at most q, evaluated at $( x _ { j } - x _ { 1 } ) / r$ . The matrix has rank $D _ { q }$ Define the polynomial map as

$$
\varphi _ { h } ( x ) = \left( \frac { 1 } { a ! } \left( \frac { x - x _ { 0 } } { h } \right) ^ { a } \right) _ { a \in \mathcal { A } _ { q } } \in \mathbb { R } ^ { D _ { q } }
$$

where $\mathcal { A } _ { q } = \{ a \in \mathbb { N } _ { 0 } ^ { d } : | a | \leq q \}$ . Order the coordinates with $a = 0$ first, and let $\boldsymbol { e } _ { 0 } \in \mathbb { R } ^ { D _ { q } }$ select this coordinate. Let

$$
w = ( w _ { 1 } , \ldots , w _ { m _ { \mathrm { s t } } } ) ^ { \top }
$$

be the unit null vector of $V ( x _ { 1 : m _ { \mathrm { s t } } } )$ , with the sign fixed by $w _ { 1 } > 0$

Estimator. Define

$$
C _ { A } = \sum _ { j = 1 } ^ { m _ { \mathrm { s t } } } w _ { j } A _ { j } \quad C _ { Y } = \sum _ { j = 1 } ^ { m _ { \mathrm { s t } } } w _ { j } Y _ { j }
$$

$$
C _ { A \varphi } = \sum _ { j = 1 } ^ { m _ { \mathrm { s t } } } w _ { j } A _ { j } \varphi _ { h } ( X _ { j } ) \quad \overline { { \varphi } } _ { w } = \sum _ { j = 1 } ^ { m _ { \mathrm { s t } } } w _ { j } ^ { 2 } \varphi _ { h } ( X _ { j } )
$$

Write $\mathcal { C } = \mathcal { C } _ { h , \ell , r } ( x _ { 0 } )$ . Define

$$
\begin{array} { r } { \hat { b } = \mathbb { U } _ { n , m _ { \mathrm { s t } } } [ 1 _ { \mathcal { C } } \overline { { \varphi } } _ { w } C _ { A } C _ { Y } ] } \\ { \hat { Q } = \mathbb { U } _ { n , m _ { \mathrm { s t } } } [ 1 _ { \mathcal { C } } \overline { { \varphi } } _ { w } C _ { A } C _ { A \varphi } ^ { \top } ] } \end{array}
$$

Let $s _ { \mathrm { m i n } }$ denote the smallest singular value. Let $q _ { 0 } = h ^ { d } \ell ^ { d } r ^ { d ( m _ { \mathrm { s t } } - 2 ) }$ , and let $\kappa _ { Q } ~ > ~ 0$ be a fixed constant such that $s _ { \operatorname* { m i n } } ( \mathbb { E } _ { P } \widehat { Q } ) \geq \kappa _ { Q } q _ { 0 }$ uniformly over the model class for the bandwidths used below. The final estimator is

$$
\begin{array} { r } { \widetilde { \theta } = \left\{ \begin{array} { l l } { \widehat { Q } ^ { - 1 } \widehat { b } , } & { s _ { \mathrm { m i n } } ( \widehat { Q } ) \geq \kappa _ { Q } q _ { 0 } / 2 , } \\ { 0 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. \quad \quad \widehat { \tau } _ { h , \ell , r } ( x _ { 0 } ) = \mathrm { c l i p } _ { [ - L _ { \tau } , L _ { \tau } ] } \left( e _ { 0 } ^ { \top } \widetilde { \theta } \right) . } \end{array}
$$

Here $\mathrm { c l i p } _ { [ a , b ] } ( t ) = \operatorname* { m a x } \{ a , \operatorname* { m i n } \{ t , b \} \}$ , and $\| \tau \| _ { \infty } \leq L _ { \tau }$ by the model-class definition.

Risk bound. The following theorem gives the risk bound for this estimator.

Theorem 3.1 (Rough-design upper bound). For suitable bandwidths $h , \ell , r$ and all suficiently large n, the estimator defined above satisfies

$$
\operatorname* { s u p } _ { P \in \mathcal { P } _ { \mathrm { r o u g h } } ( \alpha , \beta , \gamma ) } \mathbb { E } _ { P } | \widehat { \tau } _ { h , \ell , r } ( x _ { 0 } ) - \tau _ { P } ( x _ { 0 } ) | \lesssim \left\{ \begin{array} { l l } { n ^ { - \rho _ { \mathrm { T S } } } , } & { s < \frac { d \gamma } { 4 \gamma + 2 d } , } \\ { n ^ { - \gamma / ( 2 \gamma + d ) } , } & { s \geq \frac { d \gamma } { 4 \gamma + 2 d } . } \end{array} \right.
$$

## 3.2 Lower bound

We next give a lower bound on the worst-case risk of any estimator.

Theorem 3.2 (Rough-design lower bound). For every $\omega > 0$ and all suficiently large n,

$$
\operatorname* { i n f } _ { \hat { T } } \operatorname* { s u p } _ { P \in \mathcal { P } _ { \mathrm { r o u g h } } ( \alpha , \beta , \gamma ) } \mathbb { E } _ { P } | \widehat { \tau } ( x _ { 0 } ) - \tau _ { P } ( x _ { 0 } ) | \gtrsim \left\{ \begin{array} { l l } { n ^ { - ( \rho _ { \mathrm { T S } } + \omega ) } , } & { s < \frac { d \gamma } { 4 \gamma + 2 d } , } \\ { n ^ { - \gamma / ( 2 \gamma + d ) } , } & { s \geq \frac { d \gamma } { 4 \gamma + 2 d } . } \end{array} \right.
$$

The infimum is over all measurable estimators based on n i.i.d. observations.

## 4 Uniform design

We first derive a conditional upper bound for general design densities and then verify its conditions under the uniform design. The lower bound is established under the uniform design.

## 4.1 Upper bound

We describe a second-order local polynomial R-learner [7] for $\tau ( x _ { 0 } )$ at a fixed point $x _ { 0 } \in [ 0 , 1 ] ^ { d }$ The design distribution and the nuisance functions are estimated on training data T independent of the estimation sample. The results are stated for a general design density; the uniform design is the special case $f \equiv 1$

Assumptions on the distribution. Assume $\epsilon \le \pi \le 1 - \epsilon , \pi \in H ^ { \alpha } , \mu _ { 0 } \in H ^ { \beta } , \tau \in H ^ { \gamma }$ with $\| \pi \| _ { H ^ { \alpha } } + \| \mu _ { 0 } \| _ { H ^ { \beta } } + \| \tau \| _ { H ^ { \gamma } } \leq L _ { \mathrm { m o d } } , \| \mu _ { 0 } \| _ { \infty } + \| \tau \| _ { \infty } \leq M$ , and

$$
{ \mathbb E } \{ \exp ( t \varepsilon ) \mid X , A \} \leq \exp ( \sigma ^ { 2 } t ^ { 2 } / 2 ) , \qquad t \in { \mathbb R } .
$$

The marginal law F of X has a density f with $0 ~ < ~ f ~ \le ~ f ~ \le ~ \overline { { f } } ~ < ~ \infty \quad$ on $[ 0 , 1 ] ^ { d }$ . Define $\mathcal { P } _ { \operatorname { u n i f } } ( \alpha , \beta , \gamma ; \epsilon , L , M )$ as the laws in the model of section 2.1 with $f \equiv 1 , \epsilon \leq \pi \leq 1 - \epsilon , \| \pi \| _ { H ^ { \alpha } } +$ $\left. \mu _ { 0 } \right. _ { H ^ { \beta } } + \left. \tau \right. _ { H ^ { \gamma } } \leq L , \left. \mu _ { 0 } \right. _ { \infty } + \left. \tau \right. _ { \infty } \leq 3 M$ , and $\vert \varepsilon \vert \le 3 M$ almost surely, where $L , M > 0$ are fixed. Every law in this class satisfies these assumptions with $\underline { { f } } = \overline { { f } } = 1 , L _ { \mathrm { m o d } } = L$ , and with M and σ both replaced by 3M.

Kernel and local polynomial basis. Let $K _ { h } ( x _ { 0 } , u ) = h ^ { - d } 2 ^ { - d } \mathbf { 1 } \{ \| u - x _ { 0 } \| _ { \infty } \leq h \} , r _ { \gamma } = \lfloor \gamma \rfloor$ $\mathcal { A } _ { r _ { \gamma } } = \{ a \in \mathbb { N } _ { 0 } ^ { d } : \vert a \vert \leq r _ { \gamma } \} , p = \vert \mathcal { A } _ { r _ { \gamma } } \vert$ , and

$$
\varphi _ { h , x _ { 0 } } ( u ) = \Big ( \frac { 1 } { a ! } \prod _ { j = 1 } ^ { d } \Big ( \frac { u _ { j } - x _ { 0 , j } } { h } \Big ) ^ { a _ { j } } \Big ) _ { a \in A _ { r \gamma } } \in \mathbb { R } ^ { p } , \qquad \varphi ( t ) = ( t ^ { a } / a ! ) _ { a \in A _ { r \gamma } } ,
$$

with the coordinate $a = 0$ first and $e _ { 0 } \in \mathbb { R } ^ { p }$ the corresponding unit vector. Let $B _ { x _ { 0 } } ( h ) = [ 0 , 1 ] ^ { d } \cap$ $\Pi _ { j } [ x _ { 0 , j } - h , x _ { 0 , j } + h ]$ . For $\varsigma \in \{ - 1 , 1 \} ^ { d }$ let $\begin{array} { r } { I _ { \varsigma } = \prod _ { j } I _ { \varsigma _ { j } } } \end{array}$ with $I _ { 1 } = [ 0 , 1 / 2 ]$ and $I _ { - 1 } = [ - 1 / 2 , 0 ]$ , and set

$$
\lambda _ { * } = \operatorname* { m i n } _ { \substack { \varsigma \in \{ - 1 , 1 \} ^ { d } } } \lambda _ { \operatorname* { m i n } } \Big ( \int _ { I _ { \varsigma } } \varphi ( t ) \varphi ( t ) ^ { \top } d t \Big ) > 0 , \qquad c _ { Q } = 2 ^ { - d } \underline { { f } } _ { \ c } \epsilon ( 1 - \epsilon ) \lambda _ { * } .
$$

Local mesh and estimated projection kernel. Let $0 < \ell \leq h \leq 1$ . Partition each coordinate interval of $B _ { x _ { 0 } } ( h )$ into intervals of length in $[ \ell / 2 , \ell ]$ and take the induced partition of $B _ { x _ { 0 } } ( h )$ into cells. Fix $q _ { \mathrm { p o l y } } \geq \operatorname* { m a x } \{ \lfloor \alpha \rfloor , \lfloor \beta \rfloor \}$ , let $W _ { x _ { 0 } , \ell }$ be the space of functions supported on $B _ { x _ { 0 } } ( h )$ that are polynomials of degree at most $q _ { \mathrm { p o l y } }$ on each cell, and let

$$
V _ { x _ { 0 } , \ell } = W _ { x _ { 0 } , \ell } + \sum _ { a \in A _ { r _ { \gamma } } } \varphi _ { h , x _ { 0 } , a } W _ { x _ { 0 } , \ell } .
$$

Let $\widehat F$ be a finite nonnegative measure on $[ 0 , 1 ] ^ { d }$ built from $\tau ,$ and $\widehat { \mu } _ { x _ { 0 } } ( d u ) = K _ { h } ( x _ { 0 } , u ) d \widehat { F } ( u )$ . Fix a spanning set b of $V _ { x _ { 0 } , \ell } ,$ , remove redundant vectors with respect to $L _ { 2 } ( \widehat { \mu } _ { x _ { 0 } } )$ by a fixed rule, and write b for the kept vectors. Define

$$
\widehat { \Omega } _ { x _ { 0 } } = \int b ( u ) b ( u ) ^ { \top } d \widehat { \mu } _ { x _ { 0 } } ( u ) , \qquad \widehat { P } _ { x _ { 0 } } ( u , v ) = b ( u ) ^ { \top } \widehat { \Omega } _ { x _ { 0 } } ^ { - 1 } b ( v ) .
$$

Only the restriction of $\widehat F$ to $B _ { x _ { 0 } } ( h )$ enters this projection; Fb need not have total mass one. $\widehat F$

Nuisance estimators and the estimator. Let $\widehat { \pi }$ and $\widehat { \mu } _ { 0 }$ be estimators built from $\tau _ { \ast }$ and write $\Delta _ { \pi } = \pi - \widehat { \pi }$ and $\Delta _ { \mu } = \mu _ { 0 } - \widehat { \mu } _ { 0 }$ . In this upper-bound construction, n denotes the size of the estimation sample, excluding training data; $I _ { 1 } , I _ { 2 }$ and $\widehat { Q }$ are defined for the estimator in this section. Split the estimation sample into independent parts $I _ { 1 } , I _ { 2 }$ with $\underline { { c } } _ { I } n \le | I _ { j } | \le \overline { { c } } _ { I } n$ . For $i \in I _ { 1 } , j \in I _ { 2 }$ , write $\varphi _ { i } = \varphi _ { h , x _ { 0 } } ( X _ { i } ) , K _ { i } = K _ { h } ( x _ { 0 } , X _ { i } )$ , and likewise for $j$ . Let

$$
\begin{array} { l } { { \displaystyle \displaystyle \hat { Q } = \frac { 1 } { | I _ { 1 } | } \sum _ { i \in I _ { 1 } } K _ { i } \varphi _ { i } \varphi _ { i } ^ { \top } A _ { i } \{ A _ { i } - \widehat { \pi } ( X _ { i } ) \} } } \\ { ~ - \frac { 1 } { | I _ { 1 } | | I _ { 2 } | } \sum _ { i \in I _ { 1 } } \sum _ { j \in I _ { 2 } } K _ { i } \varphi _ { i } \{ A _ { i } - \widehat { \pi } ( X _ { i } ) \} K _ { j } \widehat { P } _ { x _ { 0 } } ( X _ { i } , X _ { j } ) A _ { j } \varphi _ { j } ^ { \top } , }  \\ { { \displaystyle \hat { R } = \frac { 1 } { | I _ { 1 } | } \sum _ { i \in I _ { 1 } } K _ { i } \varphi _ { i } \{ A _ { i } - \widehat { \pi } ( X _ { i } ) \} \{ Y _ { i } - \widehat { \mu } _ { 0 } ( X _ { i } ) \} } } \\ { { \displaystyle ~ - \frac { 1 } { | I _ { 1 } | | I _ { 2 } | } \sum _ { i \in I _ { 1 } } \sum _ { j \in I _ { 2 } } K _ { i } \varphi _ { i } \{ A _ { i } - \widehat { \pi } ( X _ { i } ) \} K _ { j } \widehat { P } _ { x _ { 0 } } ( X _ { i } , X _ { j } ) \{ Y _ { j } - \widehat { \mu } _ { 0 } ( X _ { j } ) \} } . }  \end{array}
$$

Fix $M _ { \mathrm { c l i p } } \geq \| \tau \| _ { \infty }$ uniformly over the model class. We take $M _ { \mathrm { c l i p } } = M$ under the general distribution assumptions and $M _ { \mathrm { c l i p } } = 3 M$ for $\mathcal { P } _ { \mathrm { u n i f } } ( \alpha , \beta , \gamma ; \epsilon , L , M )$ . Let $G = \{ s _ { \operatorname* { m i n } } ( { \widehat { Q } } ) \geq c _ { Q } / 2 \}$ ，

$$
\widehat { \theta } = \left\{ \begin{array} { l l } { \widehat { Q } ^ { - 1 } \widehat { R } , } & { G , } \\ { 0 , } & { G ^ { c } , } \end{array} \right. \quad \widehat { \tau } _ { F } ( x _ { 0 } ) = e _ { 0 } ^ { \top } \widehat { \theta } , \qquad \widetilde { \tau } _ { F } ( x _ { 0 } ) = \operatorname* { m a x } \{ - M _ { \mathrm { c l i p } } , \operatorname* { m i n } \{ \widehat { \tau } _ { F } ( x _ { 0 } ) , M _ { \mathrm { c l i p } } \} \} .
$$

Constants. Let

$$
\mathfrak { C } = \{ d , \alpha , \beta , \gamma , \epsilon , \underline { { f } } , \overline { { f } } , L _ { \mathrm { m o d } } , L _ { \mathrm { n u i s } } , M , M _ { \mathrm { c l i p } } , \sigma , \underline { { c _ { I } } } , \overline { { c } } _ { I } , q _ { \mathrm { p o l y } } \} ,
$$

where $L _ { \mathrm { n u i s } }$ is the constant in condition (2) below. Constants implicit in $\lesssim$ depend only on C and on $C _ { \mathrm { F 2 } }$ in condition (F2) below, and not on $n , h , \ell , x _ { 0 } , P$ , or the training sample. For $g$ on $[ 0 , 1 ] ^ { d }$ and $x \in [ 0 , 1 ] ^ { d }$ write $\begin{array} { r } { \| g \| _ { x , h } ^ { 2 } = \int K _ { h } ( x , u ) g ( u ) ^ { 2 } d F ( u ) } \end{array}$

Theorem 4.1 (Conditional risk bound). Fix $x _ { 0 } \in [ 0 , 1 ] ^ { d }$ and $0 < \ell \leq h \leq 1$ . Suppose the following hold uniformly over the distribution class.

(1) The assumptions on the distribution above hold with the stated constants.

(2) Conditionally on T , $\epsilon / 2 \le \widehat { \pi } \le 1 - \epsilon / 2$ and $\| \Delta _ { \pi } \| _ { H ^ { \alpha } } + \| \Delta _ { \mu } \| _ { H ^ { \beta } } \leq L _ { \mathrm { n u i s } }$

(3) $\ell ^ { \alpha }$ is small enough that $C _ { \mathrm { s t a b } } \ell ^ { \alpha } \leq c _ { Q } / 8$ , where $C _ { \mathrm { s t a b } } = C _ { \mathrm { s t a b } } ( \mathfrak { C } )$ is a fixed positive stability constant depending only on C.

(4) Conditionally on $\mathcal { T } , \widehat { F } \ll F$ , and with $r _ { F } = d \widehat { F } / d F$ and $\begin{array} { r } { \Delta _ { F } = \| r _ { F } - 1 \| _ { \infty } , } \end{array}$

(F1) $\Delta _ { F } \leq c _ { F }$ , where $c _ { F } = c _ { F } ( \mathfrak { C } ) \in ( 0 , 1 / 2 ]$ satisfies $C _ { \mathrm { s t a b } } c _ { F } \le c _ { Q } / 8 _ { ; }$ ;

(F2) with $a _ { \pi } = \| \Delta _ { \pi } \| _ { x _ { 0 } , h }$ and $a _ { \mu } = \| \Delta _ { \mu } \| _ { x _ { 0 } , h }$

$$
\Delta _ { F } a _ { \pi } ( a _ { \mu } + h ^ { \gamma } ) \leq C _ { \mathrm { F 2 } } ( h ^ { \gamma } + \ell ^ { 2 s } ) .
$$

Let $\mathbb { E } ^ { \mathrm { f i n } }$ denote expectation over the estimation sample only, with $( \widehat { F } , \widehat { \pi } , \widehat { \mu } _ { 0 } )$ fixed. Then there is a constant C, depending only on C and $C _ { \mathrm { F 2 } }$ , such that

$$
\mathbb { E } ^ { \mathrm { f i n } } | \widehat { \tau } _ { F } ( x _ { 0 } ) - \tau ( x _ { 0 } ) | ^ { 2 } \leq C \Big ( h ^ { 2 \gamma } + \ell ^ { 4 s } + \frac { 1 } { n h ^ { d } } + \frac { 1 } { n ^ { 2 } h ^ { d } \ell ^ { d } } \Big ) .
$$

Consequently, with $h ^ { \gamma } = n ^ { - \rho _ { \mathrm { S } } }$ and $\ell = h ^ { \gamma / ( 2 s ) }$ , in the low-smoothness regime,

$$
\mathbb { E } ^ { \mathrm { f i n } } | \widehat { \tau } _ { F } ( x _ { 0 } ) - \tau ( x _ { 0 } ) | ^ { 2 } \lesssim n ^ { - 2 \rho _ { \mathrm { S } } } .
$$

Both bounds are uniform in $x _ { 0 } \in [ 0 , 1 ] ^ { d }$ , in $P _ { i }$ , and in the training realization.

The proof uses conditions (2) and (4) only on $B _ { x _ { 0 } } ( h )$ . So theorem 4.1 remains true if the Hölder norms in (2) and the supremum in $\Delta _ { F }$ are taken over any set containing $B _ { x _ { 0 } } ( h )$ instead of $[ 0 , 1 ] ^ { d }$ Remark 4.2 (Known design distribution). If $F$ is known, take ${ \widehat { F } } = F$ . Then $\Delta _ { F } = 0$ and (F1), (F2) hold trivially.

Unconditional bound. Define the event

$$
\mathcal { E } _ { \mathrm { t r } } = \left\{ \widehat { F } \ll F , \ \mathrm { ( F 1 ) ~ a n d ~ ( F 2 ) ~ h o l d } , \ \epsilon / 2 \leq \widehat { \pi } \leq 1 - \epsilon / 2 , \ \lVert \Delta _ { \pi } \rVert _ { H ^ { \alpha } } + \lVert \Delta _ { \mu } \rVert _ { H ^ { \beta } } \leq L _ { \mathrm { n u i s } } \right\} ,
$$

which is measurable with respect to $\tau _ { \ast }$ , and assume that uniformly over $P ,$

$$
\mathbb { P } _ { P } ( \mathcal { E } _ { \mathrm { t r } } ^ { c } ) \leq \delta _ { \mathrm { t r } }\tag{1}
$$

for a deterministic $\delta _ { \mathrm { t r } }$ , which may depend on the sizes of the training samples.

Theorem 4.3 (Unconditional risk bound). Suppose conditions $( 1 )$ and (3) of theorem $\it 4 . 1$ hold uniformly over the model class and (1) holds. Let $R _ { n , h , \ell } = h ^ { 2 \gamma } + \ell ^ { 4 s } + ( n h ^ { d } ) ^ { - 1 } + ( n ^ { 2 } h ^ { d } \ell ^ { d } ) ^ { - 1 }$ . Then, with the constant C of theorem $4 . 1 ,$

$$
\begin{array} { r } { \mathbb { P } _ { P } \left[ \mathbb { E } _ { P } \{ | \widetilde { \tau } _ { F } ( x _ { 0 } ) - \tau ( x _ { 0 } ) | ^ { 2 } \ | \ T \} \le C R _ { n , h , \ell } \right] \ge 1 - \delta _ { \mathrm { t r } } , } \end{array}
$$

and

$$
\begin{array} { r } { \mathbb { E } _ { P } | \widetilde { \tau } _ { F } ( x _ { 0 } ) - \tau ( x _ { 0 } ) | ^ { 2 } \le C R _ { n , h , \ell } + 4 M _ { \mathrm { c l i p } } ^ { 2 } \delta _ { \mathrm { t r } } . } \end{array}
$$

Both bounds are uniform in $x _ { 0 } \in [ 0 , 1 ] ^ { d }$ and in $P .$

Uniform but unknown design. We now show that conditions (2) and (4) of theorem 4.1 hold with high probability when the design is uniform, for estimators that do not use this fact. Fix $x _ { 0 } \in ( 0 , 1 ) ^ { d }$ and $r _ { 0 } > 0$ with dis $_ { \infty } ( x _ { 0 } , \partial [ 0 , 1 ] ^ { d } ) \geq 2 r _ { 0 }$ , and let $B _ { 0 } = x _ { 0 } + [ - r _ { 0 } , r _ { 0 } ] ^ { d }$ . For $g$ on $B _ { 0 }$ $\| g \| _ { H ^ { \zeta } ( B _ { 0 } ) }$ is the Hölder norm of Section 2.1 with the suprema taken over $B _ { 0 }$ . The training data consist of two independent samples, $\mathcal { T } = ( \mathcal { T } _ { N } , \mathcal { D } _ { f } )$ , where $\mathcal { T } _ { N } = \{ ( X _ { k } ^ { \mathrm { t r } } , A _ { k } ^ { \mathrm { t r } } , Y _ { k } ^ { \mathrm { t r } } ) : k \leq N \}$ is used for $\widehat { \pi }$ and $\widehat { \mu } _ { 0 }$ and $\mathcal { D } _ { f } = \{ X _ { k } ^ { f } : k \leq m _ { f } \}$ is a covariate sample used for ${ \widehat F } .$ . Fix a kernel $K _ { \mathrm { t r } } \in C ^ { J + 1 } ( \mathbb R ^ { d } )$ $J = \left\lceil { \bar { \alpha } } \right\rceil$ , with $K _ { \mathrm { t r } } \ge 0 , \int K _ { \mathrm { t r } } = 1$ , and support in $[ - 1 , 1 ] ^ { d }$ , and write $K _ { \lambda } ( t ) = \lambda ^ { - d } K _ { \mathrm { t r } } ( t / \lambda )$ . Define

$$
\widehat { f } ( \boldsymbol { u } ) = \frac { 1 } { m _ { f } } \sum _ { k = 1 } ^ { m _ { f } } K _ { \lambda _ { f } } ( \boldsymbol { u } - \boldsymbol { X } _ { k } ^ { f } ) , \qquad S _ { \lambda } [ \xi ] ( \boldsymbol { x } ) = \frac { 1 } { N } \sum _ { k = 1 } ^ { N } K _ { \lambda } ( \boldsymbol { x } - \boldsymbol { X } _ { k } ^ { \mathrm { t r } } ) \xi ( Z _ { k } ^ { \mathrm { t r } } )
$$

for bounded functions $\xi$ of $Z = ( X , A , Y )$ , let $d \widehat { F } ( u ) = \mathbf { 1 } \{ u \in [ 0 , 1 ] ^ { d } \} \widehat { f } ( u )$ du without renormaliza tion, and let

$$
\widehat { \pi } ( x ) = \operatorname* { m a x } \Big \{ \frac { \epsilon } { 2 } , \operatorname* { m i n } \Big \{ \frac { S _ { \lambda } [ A ] ( x ) } { S _ { \lambda } [ 1 ] ( x ) } , 1 - \frac { \epsilon } { 2 } \Big \} \Big \} , \qquad \widehat { \mu } _ { 0 } ( x ) = \frac { S _ { \lambda _ { \mu } } [ ( 1 - A ) Y ] ( x ) } { S _ { \lambda _ { \mu } } [ 1 - A ] ( x ) } ,
$$

with $\widehat { \pi } = \epsilon / 2$ where $S _ { \lambda } [ 1 ] = 0 , \widehat { \mu } _ { 0 } = 0$ where its denominator is zero, and $\lambda _ { \mu } = \lambda \mathrm { i f } \alpha \ge \beta$ and $\lambda _ { \mu } = r _ { 0 } { \mathrm { ~ i f ~ } } \alpha < \beta$

Theorem 4.4 (Verification under uniform design). Let $P \in \mathcal { P } _ { \operatorname* { u n i f } } ( \alpha , \beta , \gamma ; \epsilon , L , M )$ , let $0 < \ell \leq h \leq$ $r _ { 0 }$ , and let $0 < \delta \le 1 / 2$ . Take

$$
\lambda = N ^ { - \eta } ~ w i t h ~ 0 < \eta < { \frac { 1 } { d + 2 \bar { \alpha } } } , \qquad \lambda _ { f } \in [ 1 / m _ { f } , r _ { 0 } ] .
$$

There are constants $C _ { 1 } , C _ { 2 } , L _ { \mathrm { n u i s } } ^ { * } ,$ depending only on $( d , \alpha , \beta , \gamma , \epsilon , L , M , q _ { \mathrm { p o l y } } , K _ { \mathrm { t r } } , r _ { 0 } )$ , such that if

$$
N ^ { 1 - \eta ( d + 2 \bar { \alpha } ) } \geq C _ { 1 } \log ( N / \delta ) , \qquad m _ { f } \lambda _ { f } ^ { d } \geq C _ { 2 } h ^ { - 2 \gamma } \log ( m _ { f } / \delta ) ,
$$

then with probability at least $1 - \delta$ over $\tau _ { \cdot }$ , the following hold with all norms and suprema taken over $B _ { 0 }$ :

(a) $\epsilon / 2 \le \widehat { \pi } \le 1 - \epsilon / 2$ and $\| \Delta _ { \pi } \| _ { H ^ { \alpha } ( B _ { 0 } ) } + \| \Delta _ { \mu } \| _ { H ^ { \beta } ( B _ { 0 } ) } \leq L _ { \mathrm { n u i s } } ^ { * }$

$$
\begin{array} { r } { ( b ) ~ \widehat { F } \ll F , ~ \Delta _ { F } = \operatorname* { s u p } _ { B _ { 0 } } | \widehat { f } - 1 | \leq c _ { F } , ~ a n d ~ \Delta _ { F } a _ { \pi } ( a _ { \mu } + h ^ { \gamma } ) \leq h ^ { \gamma } . } \end{array}
$$

That $i s ,$ conditions $( { \mathcal { Q } } )$ and $( 4 )$ of theorem $\it 4 . 1$ hold on $B _ { 0 }$ with $L _ { \mathrm { n u i s } } = L _ { \mathrm { n u i s } } ^ { \ast }$ and $C _ { \mathrm { F 2 } } = 1$ . The probability bound is uniform over $P \in \mathcal { P } _ { \operatorname* { u n i f } } ( \alpha , \beta , \gamma ; \epsilon , L , M )$

Since the design density is one, on $B _ { 0 }$ the mean of $S _ { \lambda } [ \xi ]$ is the convolution $K _ { \lambda } * g$ with $g ( u ) =$ $\mathbb { E } ( \xi \ | \ X = \ u )$ Convolution does not increase Hölder norms, so the Hölder norms of the ratios defining $\widehat { \pi }$ and $\widehat { \mu } _ { 0 }$ are controlled by those of $\pi$ and $\mu _ { 0 }$ , and the stochastic parts have Hölder norm of order $\lambda ^ { - \bar { \alpha } } \sqrt { \log ( N / \delta ) / ( N \lambda ^ { d } ) }$ , which the condition on η makes small. Likewise $\widehat { f }$ is unbiased on $B _ { 0 }$ , so $\Delta _ { F }$ is of order $\sqrt { \log ( m _ { f } / \delta ) / ( m _ { f } \lambda _ { f } ^ { d } ) }$ . This is where uniformity of the design is used, and it is why $x _ { 0 }$ is taken in the interior.

Risk bound. Take $h ^ { \gamma } = n ^ { - \rho _ { \mathrm { S } } } , \ell = h ^ { \gamma / ( 2 s ) }$ , and $\delta = n ^ { - 2 \rho _ { \mathrm { S } } }$ in theorem 4.4, with $N = m _ { f } = n$ and $\lambda _ { f } = r _ { 0 }$ . In the low-smoothness regime $\rho _ { \mathrm { S } } < 1 / 2 .$ , so both sample size conditions of theorem 4.4 hold for large n. Then theorem 4.3, with $\delta _ { \mathrm { t r } } = n ^ { - 2 \rho _ { \mathrm { S } } }$ and with the conditions localized to $B _ { 0 }$ as noted after theorem 4.1, gives the following result.

Corollary 4.5 (Uniform-design upper bound). Suppose $s < d \gamma / ( 4 \gamma + 2 d )$ and $x _ { 0 } \in ( 0 , 1 ) ^ { d }$ . With the bandwidths and training sample sizes chosen above, for all suficiently large $n ,$

$$
\operatorname* { s u p } _ { P \in \mathcal { P } _ { \mathrm { u n i f } } ( \alpha , \beta , \gamma ; \epsilon , L , M ) } \mathbb { E } _ { P } | \widetilde \tau _ { F } ( x _ { 0 } ) - \tau ( x _ { 0 } ) | ^ { 2 } \leq C n ^ { - 2 \rho _ { \mathrm { S } } } .
$$

The estimator does not use the design density, and $M _ { \mathrm { c l i p } } = 3 M$

The total sample size is $n _ { \mathrm { t o t } } = n + N + m _ { f } = 3 n ;$ replacing n by $n _ { \mathrm { t o t } }$ changes only the constant in the rate.

## 4.2 Lower bound

We next establish a lower bound under uniform design, using a binary-outcome subclass. Define this subclass by

$$
\begin{array} { r } { \mathcal { P } _ { \operatorname* { u n i f } } ^ { \mathrm { b i n } } ( \alpha , \beta , \gamma ; \epsilon ) = \{ P \in \mathcal { P } _ { \operatorname* { u n i f } } ( \alpha , \beta , \gamma ; \epsilon , 1 , 1 ) : P ( Y \in \{ 0 , 1 \} ) = 1 \} . } \end{array}
$$

Theorem 4.6 (Uniform-design lower bound). Let $d \geq 1$ , let $\alpha , \beta , \gamma > 0$ satisfy $s < d \gamma / ( 4 \gamma + 2 d )$ let $\epsilon \in ( 0 , 1 / 2 )$ , and let $x _ { 0 } \in ( 0 , 1 ) ^ { d }$ . There is a constant $c > 0$ , depending only on $( d , \alpha , \beta , \gamma , \epsilon , x _ { 0 } )$ such that for all $n _ { \ ! }$ ,

$$
\underset { \ b { \hat { \tau } } } { \operatorname* { i n f } } \ \underset { { P \in \mathcal { P } _ { \mathrm { u n i f } } ^ { \mathrm { b i n } } ( \alpha , \beta , \gamma ; \epsilon ) } } { \operatorname* { s u p } } \ \mathbb { E } _ { P } \big | \hat { \tau } ( x _ { 0 } ) - \tau _ { P } ( x _ { 0 } ) \big | \ \geq \ c n ^ { - \rho _ { \mathrm { S } } } ,
$$

where the infimum is over all measurable functions of n i.i.d. observations.

Since

$$
\begin{array} { r } { \mathcal { P } _ { \operatorname* { u n i f } } ^ { \mathrm { b i n } } ( \alpha , \beta , \gamma ; \epsilon ) \subseteq \mathcal { P } _ { \operatorname* { u n i f } } ( \alpha , \beta , \gamma ; \epsilon , L , M ) \qquad \mathrm { f o r ~ a l l ~ } L \geq 1 , \ M \geq 1 , } \end{array}
$$

the same lower bound holds over $\mathcal { P } _ { \operatorname { u n i f } } ( \alpha , \beta , \gamma ; \epsilon , L , M )$

theorem 4.6 shows that no estimator can do better than $n ^ { - \rho _ { \mathrm { S } } }$ in mean absolute error. Together with corollary 4.5, this shows that in the low-smoothness regime $s < d \gamma / ( 4 \gamma + 2 d )$ the minimax rate for $\tau ( x _ { 0 } )$ at an interior point under uniform design is $n ^ { - \rho _ { \mathrm { S } } }$ with $\rho _ { \mathrm { S } } = 1 / \{ 1 + d / ( 2 \gamma ) + d / ( 4 s ) \}$

## 5 Discussion

Our results show that the minimax rate for estimating the CATE at a point depends on the covariate design.

Future work could characterize the minimax rate when $f \in H ^ { \nu } ( L _ { f } )$ for fixed $\nu , L _ { f } > 0$ , where ν measures design-density smoothness. This would describe the intermediate setting between rough and smooth designs. Another direction is to extend the analysis to average treatment efect (ATE) estimation and other causal estimands.

## Proofs

Lean 4 code for the proofs is available on GitHub at https: $: / / \mathrm { g } \dot { \bf 1 }$ thub.com/houyx22/causal\_ lowerbound. Parts of the proofs were developed with AI assistance.

## References

[1] Alberto Abadie and Guido W. Imbens. Large sample properties of matching estimators for average treatment efects. Econometrica, 74(1):235–267, 2006. doi: 10.1111/j.1468-0262.2006. 00655.x.

[2] Alberto Abadie and Guido W. Imbens. Bias-corrected matching estimators for average treatment efects. Journal of Business & Economic Statistics, 29(1):1–11, 2011. doi: 10.1198/jbes. 2009.07333.

[3] Heejung Bang and James M. Robins. Doubly robust estimation in missing data and causal inference models. Biometrics, 61(4):962–973, 2005. doi: 10.1111/j.1541-0420.2005.00377.x.

[4] Zijun Gao and Yanjun Han. Minimax optimal nonparametric estimation of heterogeneous treatment efects. In Advances in Neural Information Processing Systems, volume 33, pages 21751–21762. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/ 2020/hash/f75b757d3459c3e93e98ddab7b903938-Abstract.html.

[5] James J. Heckman, Hidehiko Ichimura, and Petra Todd. Matching as an econometric evaluation estimator. The Review of Economic Studies, 65(2):261–294, 1998. doi: 10.1111/1467-937X. 00044.

[6] Edward H. Kennedy. Towards optimal doubly robust estimation of heterogeneous causal efects. Electronic Journal of Statistics, 17(2):3008–3049, 2023. doi: 10.1214/23-EJS2157.

[7] Edward H. Kennedy, Sivaraman Balakrishnan, James M. Robins, and Larry Wasserman. Minimax rates for heterogeneous causal efect estimation. The Annals of Statistics, 52(2):793–816, 2024. doi: 10.1214/24-AOS2369.

[8] Sokbae Lee, Ryo Okui, and Yoon-Jae Whang. Doubly robust uniform confidence band for the conditional average treatment efect function. Journal of Applied Econometrics, 32(7): 1207–1225, 2017. doi: 10.1002/jae.2574.

[9] Zhexiao Lin, Peng Ding, and Fang Han. Estimation based on nearest neighbor matching: From density ratio to average treatment efect. Econometrica, 91(6):2187–2217, 2023. doi: 10.3982/ECTA20598.

[10] Xinkun Nie and Stefan Wager. Quasi-oracle estimation of heterogeneous treatment efects. Biometrika, 108(2):299–319, 2021. doi: 10.1093/biomet/asaa076.

[11] James M. Robins, Andrea Rotnitzky, and Lue Ping Zhao. Estimation of regression coeficients when some regressors are not always observed. Journal of the American Statistical Association, 89(427):846–866, 1994. doi: 10.1080/01621459.1994.10476818.

[12] Paul R. Rosenbaum and Donald B. Rubin. The central role of the propensity score in observational studies for causal efects. Biometrika, 70(1):41–55, 1983. doi: 10.1093/biomet/70.1.41.