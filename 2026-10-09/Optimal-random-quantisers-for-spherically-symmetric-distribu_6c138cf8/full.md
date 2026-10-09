# Optimal random quantisers for spherically symmetric distributions

Luc Pronzato<sup>∗</sup> and Anatoly Zhigljavsky<sup>†</sup>

October 9, 2026

## Abstract

Zador’s celebrated theorem is a cornerstone of optimal quantisation: it establishes both the weak limit of the empirical distribution of an optimal n-point quantiser in $\mathbb { R } ^ { d }$ and the decay rate of the associated L -mean quantisation error. In large dimension, however, observing this asymptotic behaviour requires an astronomically large sample size. We prove that, for spherically symmetric target distributions, optimisation over all spherically symmetric distributions is a convex problem and derive an equivalence theorem that both characterises global optimality and yields a constructive algorithm. We show that, for moderate n, random quantisers uniformly distributed on a sphere of suitably chosen radius R perform exceptionally well and, over a broad range of values of $n ,$ are numerically certified to be optimal among all random quantisers. Their expected distortion has an explicit integral representation that can be evaluated to arbitrary precision, and we prove concentration across random quantisers: the distortion variance tends to zero as n → ∞ for fixed d. For $s = 2$ , both the optimal radius and the associated minimum expected distortion admit exact expressions. For general s, the optimal radius can be determined eficiently, and extreme-value theory provides useful approximations when n grows with d. Depending on this growth rate, R either converges to zero or approaches a positive limit that is independent of s.

keywords: quantisation, distortion, spherically symmetric distribution, Zador’s theorem, extreme-value theory

MSC: primary 28-08; secondary 60D99, 62K99

## 1 Introduction

For $\mu$ a measure on $\mathcal { X } \subseteq \mathbb { R } ^ { d }$ (or a d-dimensional Riemannian manifold), with density $\varphi$ and finite moment of order $s > 0$ , the (µ, s)-distortion on an n-point set ${ \bf X } _ { n } = \{ { \bf x } _ { 1 } , \ldots , { \bf x } _ { n } \} \in \mathbb { R } ^ { n \times d }$ is $D _ { \mu , s } ( \mathbf { X } _ { n } ) = \mathsf { E } _ { \mu } \{ \operatorname* { m i n } _ { i = 1 , \ldots , n } \left\| U - \mathbf { x } _ { i } \right\| ^ { s } \}$ , where $U \stackrel { d } { \sim } \mu .$ . Zador’s celebrated theorem [17] states that, when $\mu$ has finite moment of order $s ^ { \prime } > s .$ , the empirical distribution of an n-point optimal quantiser $\mathbf { X } _ { n } ^ { * }$ minimising $D _ { \mu , s } ( \mathbf { X } _ { n } )$ converges weakly, as $n  \infty$ , to the probability measure with density proportional to $\varphi ^ { d / ( d + s ) }$ . It also characterises the asymptotic minimum of the normalised distortion $n ^ { s / d } D _ { \mu , s } ( \mathbf { X } _ { n } ^ { * } )$ , or equivalently of the normalised quantisation error $n ^ { 1 / d } E _ { \mu , s } ( \mathbf { X } _ { n } ^ { * } )$ . These asymptotic considerations are informative only when $n ^ { 1 / d }$ is suficiently large; in large dimension, this requires astronomically large values of n.

The purpose of this paper is to show that for $\mu$ spherically symmetric, random quantisers with a suitable spherically symmetric distribution can achieve exceptional performance. When $n$ is large but $n ^ { 1 / d }$ is small, the best random quantisers we obtain are for designs $\mathbf { X } _ { n }$ uniform on a sphere $S _ { d - 1 } ( R )$ with suitable radius $R ,$ and the variability of $D _ { \mu , s } ( \mathbf { X } _ { n } )$ vanishes as n increases (Theorem 4.1). In the particular case $s = 2$ , the optimal R is proportional to $\mathsf { E } _ { \mu } \{ \| U \| \}$ with a constant depending only on n and $d ,$ and the best expected $( \mu , 2 )$ )-distortion is $\mathsf { E } _ { \mu } \{ \| U \| ^ { 2 } \} - R ^ { 2 }$ ; see Corollary 2.1. For general s, using extreme-value theory, we derive approximations for $R$ and for the best expected $( \mu , s )$ -distortion. We then identify diferent asymptotic regimes where $d$ increases and n grows with $d .$ When $\mu$ possesses a norm-concentration property and concentrates on a sphere of radius r as $d \to \infty$ the optimal radius $R$ of the random quantiser exhibits distinct behaviours depending on the growth rate of $n$ relative to $d \colon$ for a sub-exponential growth, $R \to 0 ;$ for a super-exponential growth, $R \to r ;$ for an exponential growth $n = \lambda ^ { d } ( 1 + o ( 1 ) )$ , $R  r \sqrt { 1 - 1 / \lambda ^ { 2 } } ;$ ; see Theorem $4 . 2 .$ . Overall, the paper provides a framework for generating nearly optimal quantisers in the large $n$ and $d$ situation for spherically symmetric distributions. Sequences of nested designs with low quantisation error can thus be eficiently generated, ofering a compelling alternative to the classical greedy-packing algorithm for space-filling design (see, e.g., [15]). While this paper focuses exclusively on spherically symmetric distributions, the conclusion highlights that extending this approach to the more common case of quantising the uniform measure in the d-dimensional hypercube yields promising numerical results.

The paper is organised as follows. Section 2 introduces the key concepts used throughout the paper. It also derives an explicit integral expression for the expected $( \mu , s )$ )-distortion of a random quantiser whose n points are independent and identically distributed (i.i.d.) according to a spherically symmetric distribution $\mathbb { P } .$ In Section $^ { 3 , }$ we exploit the convexity of the expected $( \mu , s )$ -distortion with respect to $\mathbb { P }$ to derive a necessary and suficient condition for optimality of a quantiser distribution $\mathbb { P } ^ { * }$ and propose an optimisation algorithm. Three emblematic cases are presented: (i) $\mu$ is uniform on the unit sphere, $( i i ) ~ \mu$ is uniform in the unit ball, and (iii) $\mu$ follows a multivariate spherically symmetric normal distribution. In the latter two scenarios, in addition to the case where $\mathbb { P }$ is uniform on a sphere, we also analyse the cases where $\mathbb { P }$ is uniform in a ball and where $\mathbb { P }$ is a multivariate spherically symmetric normal distribution. Section 4 applies extreme-value theory to derive asymptotic results $( n  \infty )$ for random quantisers uniform on a sphere. Section 5 provides a brief conclusion to the paper. Several illustrative examples are presented, using the Matlab scripts available at https: //github.com/lpronzato-prog/Spherically-symmetric-random-designs. Additional numerical examples are presented in the supplementary material.

We denote by $\mathcal { B } _ { d } ( \mathbf { x } , r )$ the d-dimensional (closed) Euclidean ball with centre $\mathbf { x }$ and radius $r ;$ we write $\mathcal { B } _ { d } ( r ) ~ = ~ \mathcal { B } _ { d } ( \mathbf { 0 } _ { d } , r )$ , and $S _ { d - 1 } ( r )$ denotes its $( d - 1 ) \cdot$ dimensional boundary sphere; $\| \cdot \|$ is the ℓ<sub>2</sub>-norm; $\mathbf { u } ^ { \top }$ is the transpose of the column vector u; $\mathbf { I } _ { d }$ is the d-dimensional identity matrix.

## 2 Distance c.d.f., quantisation error and distortion

## 2.1 Definitions and basic properties

For any n-point set ${ \bf X } _ { n }$ we denote by $d ( \cdot , \mathbf { X } _ { n } )$ the distance function defined by $\begin{array} { r } { \mathbf { x } \in \mathbb { R } ^ { d } \mapsto d ( \mathbf { x } , \mathbf { X } _ { n } ) = \operatorname* { m i n } _ { \mathbf { x } _ { i } \in \mathbf { X } _ { n } } \| \mathbf { x } - \mathbf { x } _ { i } \| ; } \end{array}$ ν denotes a probability measure on $\mathbb { R } ^ { d }$

Definition 2.1. For ${ \bf X } _ { n }$ a fixed n-point set in $\mathbb { R } ^ { d }$ , the distance c.d.f. $F ( \cdot ; \mathbf { X } _ { n } , \nu )$

is the $c . d . f .$ of the random variable $d ( U , \mathbf { X } _ { n } )$ when U is distributed with $\nu :$

$$
F ( t ; \mathbf { X } _ { n } , \nu ) = \nu \left\{ U \in \bigcup _ { i = 1 } ^ { n } \mathcal { B } _ { d } ( \mathbf { x } _ { i } , t ) \right\} = \nu \{ d ( U , \mathbf { X } _ { n } ) \leq t \} .
$$

When ${ \bf X } _ { n } = { \bf R } _ { n }$ is a random n-point set generated with some probability measure $\mathsf { P } _ { n }$ on $\mathbb { R } ^ { n \times d }$ , we define the mean distance c.d.f. $F ( \cdot ; \mathsf { P } _ { n } , \nu )$ of the random variable $d ( U , \mathbf { R } _ { n } )$ by $F ( t ; \mathsf { P } _ { n } , \nu ) = \mathsf { E } _ { \mathsf { P } _ { n } } \{ F ( t ; \mathbf { R } _ { n } , \nu ) \}$

If ν is the delta measure $\delta _ { \mathbf { u } }$ concentrated at $\mathbf { u } \in \mathbb { R } ^ { d }$ , the mean distance c.d.f. is simply $F ( t ; \mathsf { P } _ { n } , \delta _ { \mathbf { u } } ) = \mathsf { P } _ { n } \left\{ d ( \mathbf { u } , \mathbf { R } _ { n } ) \leq t \right\}$

When $\mathcal { X }$ is a compact subset of $\mathbb { R } ^ { d }$ , we denote by $\begin{array} { r } { \mathsf { C R } ( \mathbf { X } _ { n } ) = \operatorname* { m a x } _ { \mathbf { x } \in \mathcal { X } } d ( \mathbf { x } , \mathbf { X } _ { n } ) } \end{array}$ the covering radius of $\mathbf { X } _ { n }$ . If ν is equivalent to the Lebesgue measure on $\mathcal { X }$ $F ( \cdot ; \mathbf { X } _ { n } , \nu )$ has a density on $\left[ 0 , { \mathsf { C R } } ( \mathbf { X } _ { n } ) \right]$ , which we denote by $f ( \cdot ; \mathbf { X } _ { n } , \nu )$ , and the essential supremum of the random variable $d ( U , \mathbf { X } _ { n } )$ equals $\mathsf { C R } ( \mathbf { X } _ { n } )$ When ν is the uniform measure on $\mathcal { X }$ , we simply denote the distance c.d.f. by $F ( \cdot ; \mathbf { X } _ { n } )$ : the random variable $d ( U , \mathbf { X } _ { n } )$ is supported on $[ 0 , { \mathsf { C R } } ( \mathbf { X } _ { n } ) ]$ and $F ( \cdot ; \mathbf { X } _ { n } )$ contains all information about the filling of $\mathcal { X }$ by $\mathbf { X } _ { n }$ . In particular, the γ-quantiles $q _ { \gamma } ( \mathbf { X } _ { n } )$ of $F ( \cdot ; \mathbf { X } _ { n } )$ provide useful space-filling characteristics. When $\mathcal { X }$ is connected and coincides with the closure of its interior, the density $f ( \cdot ; \mathbf { X } _ { n } )$ is strictly positive on $( 0 , \mathsf { C R } ( \mathbf { X } _ { n } ) )$ and $q _ { \gamma } ( \mathbf { X } _ { n } )$ is defined by

$$
F \lbrack q _ { \gamma } ( \mathbf { X } _ { n } ) ; \mathbf { X } _ { n } \rbrack = \gamma \mathrm { f o r ~ a n y } \gamma \in ( 0 , 1 ) ,
$$

with $q _ { 0 } ( \mathbf { X } _ { n } ) = 0$ and $q _ { 1 } ( \mathbf { X } _ { n } ) = \mathsf { C R } ( \mathbf { X } _ { n } )$

Definition 2.2. The L -mean quantisation error of a general probability measure $\mu$ on $\mathbb { R } ^ { d }$ with finite $s { - } t h$ moment $E \{ \| X \| ^ { s } \}$ for a given n-point set ${ \bf X } _ { n }$ is the $L _ { s } ( \mu )$ mean (and, when $s \geq 1$ , the $L _ { s } ( \mu ) \ – n o r m )$ of the distance function $d ( \cdot , \mathbf { X } _ { n } )$

$$
E _ { \mu , s } ( \mathbf { X } _ { n } ) = \| d ( \cdot , \mathbf { X } _ { n } ) \| _ { L _ { s } } = \left[ \int _ { \mathcal { X } } d ^ { s } ( \mathbf { x } , \mathbf { X } _ { n } ) \mu ( \mathrm { d } \mathbf { x } ) \right] ^ { 1 / s } , \quad s > 0 .
$$

The quantity $D _ { \mu , s } ( { \mathbf { X } } _ { n } ) = E _ { \mu , s } ^ { s } ( { \mathbf { X } } _ { n } )$ is called the $( \mu , s )$ -distortion related to ${ \bf X } _ { n } \ [ 1 2 ]$ When ${ \mathbf X } _ { n } = { \mathbf R } _ { n }$ is random, we define the expected $( \mu , s )$ -distortion as $D _ { \mu , s } ( { \mathsf { P } } _ { n } ) =$ $\mathsf E _ { \mathsf P _ { n } } \{ E _ { \mu , s } ^ { s } ( { \bf R } _ { n } ) \}$

For any $s > 0$ , the $( \mu , s )$ -distortion satisfies $( \mathrm { s e e } , \mathrm { e . g . } , [ 5 , \mathrm { p . ~ } 1 5 0 ] )$ ):

$$
D _ { \mu , s } ( { \mathbf { X } _ { n } } ) = { \mathbb E } _ { \mu } \{ d ^ { s } ( U , { \mathbf { X } _ { n } } ) \} = s \int _ { 0 } ^ { \infty } { t ^ { s - 1 } \left[ 1 - F ( t ; { \mathbf { X } _ { n } } , \mu ) \right] } \mathrm { d } t .\tag{1}
$$

The result is easily obtained when $\mu$ is such that $F ( \cdot ; \mathbf { X } _ { n } , \mu )$ has a density $f ( \cdot ; \mathbf { X } _ { n } , \mu )$ Indeed, by swapping the order of integration we get

$$
\begin{array} { r l r } { \mathsf { E } _ { \mu } \{ d ( U , { \bf X } _ { n } ) \} } & { = } & { \displaystyle \int _ { 0 } ^ { \infty } \tau f ( \tau ; { \bf X } _ { n } , \mu ) \mathrm { d } \tau = \int _ { 0 } ^ { \infty } \left( \int _ { 0 } ^ { \tau } \mathrm { d } t \right) f ( \tau ; { \bf X } _ { n } , \mu ) \mathrm { d } \tau } \\ & { = } & { \displaystyle \int _ { 0 } ^ { \infty } \left( \int _ { t } ^ { \infty } f ( \tau ; { \bf X } _ { n } , \mu ) \mathrm { d } \tau \right) \mathrm { d } t = \int _ { 0 } ^ { \infty } [ 1 - F ( t ; { \bf X } _ { n } , \mu ) ] \mathrm { d } t . } \end{array}
$$

Now, for any $s > 0 , G ^ { ( s ) } ( \cdot ; \mathbf { X } _ { n } , \mu )$ , defined by $G ^ { ( s ) } ( t ; { \bf X } _ { n } , \mu ) = F ( t ^ { 1 / s } ; { \bf X } _ { n } , \mu )$ for any $t \geq 0 ,$ , is the c.d.f. of $d ^ { s } ( U , \mathbf { X } _ { n } )$ , and we obtain

$$
\mathsf E _ { \mu } \{ d ^ { s } ( U , \mathbf X _ { n } ) \} = \int _ { 0 } ^ { \infty } [ 1 - F ( t ^ { 1 / s } ; \mathbf X _ { n } , \mu ) ] \mathrm d t = s \int _ { 0 } ^ { \infty } t ^ { s - 1 } \left[ 1 - F ( t ; \mathbf X _ { n } , \mu ) \right] \mathrm d t .
$$

Remark 2.1. For any probability measure µ such that $\mathsf { E } \{ \| X \| ^ { 2 s } \} < \infty$ , the $( \mu , s )$ distortion of an arbitrary n-point set $\mathbf { X } _ { n }$ can be approximated by the Monte-Carlo estimator $\begin{array} { r } { D _ { \mu _ { N } , s } ( \mathbf { X } _ { n } ) = ( 1 / N ) \sum _ { i = 1 } ^ { N } d ^ { s } ( U _ { i } , \mathbf { X } _ { n } ) } \end{array}$ , where the $U _ { i }$ are i.i.d. with distribution $\mu .$ . This estimator satisfies the classical central limit theorem

$$
\sqrt { N } \left[ D _ { \mu _ { N } , s } ( { \mathbf X } _ { n } ) - D _ { \mu , s } ( { \mathbf X } _ { n } ) \right] \stackrel { \mathrm { d } } { \to } { \mathcal N } ( 0 , V ( { \mathbf X } _ { n } ) ) , \quad N \to \infty ,
$$

where $V ( { \bf X } _ { n } ) = D _ { \mu , 2 s } ( { \bf X } _ { n } ) - D _ { \mu , s } ^ { 2 } ( { \bf X } _ { n } )$ . When $\mu$ is supported on $\mathcal { X }$ compact, $\mathsf { C R } ( \mathbf { X } _ { n } ) <$ ∞ and a direct application of Hoefding’s inequality gives:

for all $\alpha \in ( 0 , 1 )$ , Prob {|D<sub>µN,s</sub>(X<sub>n</sub>) − D<sub>µ,s</sub>(X<sub>n</sub>)| > α CR<sup>s</sup>(X<sub>n</sub>)} < 2 e<sup>−2</sup> <sup>Nα2</sup> ,

showing the exponentially fast concentration of $D _ { \mu _ { N } , s } ( { \mathbf { X } } _ { n } )$ to its mean $D _ { \mu , s } ( \mathbf { X } _ { n } )$ as N increases. When µ has unbounded support, depending on its tail properties, other concentration inequalities can also be derived. ◁

## 2.2 Quantisation error of random quantisers

Let ${ \bf R } _ { n } \ = \ \{ { \bf x } _ { 1 } , \ldots , { \bf x } _ { n } \}$ denote a random n-point set generated according to a probability measure $\mathsf { P } _ { n }$ on $\mathbb { R } ^ { n \times d }$ . Our objective is to choose $\mathsf { P } _ { n }$ to minimise the expected $( \mu , s )$ -distortion $D _ { \mu , s } ( \mathsf { P } _ { n } )$ defined in Definition 2.2.

From (1), we can write

$$
D _ { \mu , s } ( \mathsf { P } _ { n } ) = s \int _ { t \geq 0 } t ^ { s - 1 } [ 1 - F ( t ; \mathsf { P } _ { n } , \mu ) ] \mathrm { d } t\tag{2}
$$

where $F ( \cdot ; \mathsf { P } _ { n } , \mu ) = \mathsf { E } _ { \mathsf { P } _ { n } } \{ F ( \cdot ; \mathbf { R } _ { n } , \mu ) \}$ is the mean distance c.d.f. of Definition $2 . 1 ;$ that is, the c.d.f. of the random variable $d ( U , \mathbf { R } _ { n } )$ where both $U$ and ${ \mathbf { R } } _ { n }$ are random. The expected $( \mu , s )$ -distortion $D _ { \mu , s } ( \mathsf { P } _ { n } )$ is the $( \mu , s )$ -distortion for the c.d.f. $F ( \cdot ; \mathsf { P } _ { n } , \mu )$ . In this paper we focus on distortion and quantisation error, but we could have also considered the γ-quantiles of $F ( \cdot ; \mathsf { P } _ { n } , \mu )$ ; see Remark 4.2. Note that Jensen inequality yields the following upper bound on the expected quantisation error: $\mathsf E _ { \mathsf P _ { n } } \{ E _ { \mu , s } ( \mathbf { R } _ { n } ) \} \leq [ D _ { \mu , s } ( \mathsf { P } _ { n } ) ] ^ { 1 / s }$ for $s \geq 1$

By swapping the order of expectations, we get

$$
\begin{array} { l l l } { F ( t ; { \mathsf { P } } _ { n } , \mu ) } & { = } & { { \mathsf { E } } _ { { \mathsf { P } } _ { n } } \left\{ { \mathsf { E } } _ { \mu } \left\{ \mathbb { 1 } _ { \{ { \mathbf { u } } \in \mathbb { R } ^ { d } : d ( { \mathbf { u } } , { \mathbf { R } } _ { n } ) \leq t \} } ( U ) \right\} \right\} } \\ & { = } & { { \mathsf { E } } _ { \mu } \left\{ { \mathsf { E } } _ { { \mathsf { P } } _ { n } } \left\{ \mathbb { 1 } _ { \{ { \mathbf { u } } \in \mathbb { R } ^ { d } : d ( { \mathbf { u } } , { \mathbf { R } } _ { n } ) \leq t \} } ( U ) \right\} \right\} } \\ & { = } & { { \mathsf { E } } _ { \mu } \left\{ F ( t ; { \mathsf { P } } _ { n } , \delta _ { U } ) \right\} , } \end{array}\tag{3}
$$

where $F ( t ; \mathsf { P } _ { n } , \delta _ { \mathbf { u } } ) = \mathsf { P } _ { n } \left\{ d ( \mathbf { u } , \mathbf { R } _ { n } ) \leq t \right\}$

In the following, we call random quantiser an n-point set ${ \mathbf { R } } _ { n }$ with i.i.d. points having some distribution $\mathbb { P } _ { : }$ , so that $\begin{array} { r } { \mathsf { P } _ { n } ( \mathrm { d } \mathbf { x } _ { 1 } , \ldots , \mathrm { d } \mathbf { x } _ { n } ) = \prod _ { i = 1 } ^ { n } \mathbb { P } ( \mathrm { d } \mathbf { x } _ { i } ) ; } \end{array}$ we shall denote this distribution $\mathsf { P } _ { n } = \mathbb { P } ^ { [ n ] }$ . In Sections 2.3 and 2.4 we show that $F ( t ; \mathbb { P } ^ { [ n ] } , \mu )$ can be expressed in the form of a double integral when P is spherically symmetric. This will rely on the following property which is true for any $\mathbb { P }$

For any fixed $\textbf { u } \in \ \mathbb { R } ^ { d }$ , any $t \geq 0$ and any ${ \mathbf { R } } _ { n }$ with i.i.d. points having the arbitrary distribution P, we have

$$
\begin{array} { r c l } { F ( t ; \mathbb { P } ^ { [ n ] } , \delta _ { \mathbf { u } } ) } & { = } & { 1 - \displaystyle \prod _ { j = 1 } ^ { n } \mathbb { P } \left\{ \mathbf { u } \notin \mathcal { B } _ { d } ( \mathbf { x } _ { j } , t ) \right\} } \\ { } & { = } & { 1 - \displaystyle \prod _ { j = 1 } ^ { n } \left( 1 - \mathbb { P } \left\{ \mathbf { u } \in \mathcal { B } _ { d } ( \mathbf { x } _ { j } , t ) \right\} \right) = 1 - \left( 1 - \mathbb { P } \left\{ \left\| X - \mathbf { u } \right\| \leq t \right\} \right) ^ { n } , } \end{array}\tag{4}
$$

where $X \stackrel { d } { \sim } \mathbb { P } .$

The expression (4) indicates that the calculations of the mean distance c.d.f. $F ( t ; \mathbb { P } ^ { [ n ] } , \mu )$ and then of the expected $( \mu , s )$ -distortion require the calculation of the probability

$$
\mathbb { P } \left\{ \| X - \mathbf { u } \| \leq t \right\} = \mathbb { P } \left\{ \mathbf { u } \in \mathcal { B } _ { d } ( X , t ) \right\} ,
$$

which depends on P, u, and t. Exchanging the order of integration in (3) thus leads us to consider another distance c.d.f., in which u is fixed and plays the role of a one-point quantiser, whereas X is random. This symmetry disappears for random n-point sets, whose n i.i.d. points are accounted for explicitly in (4). It is noticeable that this is the only place where the size n appears.

## 2.3 P $\{ \| X - \mathbf { u } \| \leq t \}$ when $\mathbb { P }$ is spherically symmetric

By definition, the random vector $X = ( X _ { 1 } , \ldots , X _ { d } )$ is spherically symmetric if it can be represented as $X \ { \overset { d } { = } } \ R \cdot Z ^ { ( d ) }$ , where $R = \| X \| , Z ^ { ( d ) } = ( Z _ { 1 } , . . . , Z _ { d } )$ is uniformly distributed on the unit sphere $S _ { d - 1 } ( 1 )$ , and R and $Z ^ { ( d ) }$ are independent.

When P is spherically symmetric, the distribution of $\| X - \mathbf { u } \|$ for $X \ { \overset { d } { \sim } } \ \mathbb { P }$ only depends on ∥u∥ and d. This distribution is specified in Proposition 2.1 below. In the following, $\beta _ { a , b }$ denotes a random variable with the Beta-density

$$
\begin{array} { r } { \varphi _ { a , b } ( t ) = t ^ { a - 1 } ( 1 - t ) ^ { b - 1 } / B ( a , b ) \mathrm { f o r } 0 \leq t \leq 1 , } \end{array}
$$

where $B ( a , b ) = \Gamma ( a ) \Gamma ( b ) / \Gamma ( a + b )$ is the Beta-function, $a , b > 0$ . The c.d.f. of $\beta _ { a , b }$ is Prob $\{ \beta _ { a , b } \leq t \} = I _ { t } ( a , b )$ , where $I _ { t } ( a , b )$ is the regularised incomplete Beta-function, for which we use the convention

$$
I _ { t } ( \cdot , \cdot ) = \left\{ \begin{array} { l l } { { 0 } } & { { \mathrm { f o r } t \leq 0 } } \\ { { 1 } } & { { \mathrm { f o r } t \geq 1 . } } \end{array} \right.\tag{5}
$$

The following lemma (see, $\mathrm { { e . g . , \ [ 3 . } }$ , Sect. 2.2]) will be used several times.

Lemma 2.1. Let $X = ( X _ { 1 } , \dots , X _ { d } ) \overset { d } { = } R \cdot Z ^ { ( d ) }$ be a spherically symmetric random vector in $\mathbb { R } ^ { d }$ . Then, for any $m \in \{ 1 , \ldots , d - 1 \}$ , we have

$$
\begin{array} { r } { X ^ { ( m ) } = ( X _ { 1 } , \ldots , X _ { m } ) \stackrel { d } { = } R \cdot \sqrt { \beta _ { m / 2 , ( d - m ) / 2 } } \cdot Z ^ { ( m ) } , } \end{array}
$$

where $R = \| X \| , \ \beta _ { m / 2 , ( d - m ) / 2 }$ and $Z ^ { ( m ) }$ are independent. Moreover, the joint density of $V ^ { ( m ) } = X ^ { ( m ) } / R$ is

$$
\varphi _ { m } ( v _ { 1 } , \ldots , v _ { m } ) = { \frac { \Gamma ( d / 2 ) } { \pi ^ { m / 2 } \Gamma ( ( d - m ) / 2 ) } } \left( 1 - \sum _ { i = 1 } ^ { m } v _ { i } ^ { 2 } \right) ^ { ( d - m ) / 2 - 1 } f o r \sum _ { i = 1 } ^ { m } v _ { i } ^ { 2 } \leq 1 .
$$

The next proposition is the key ingredient in the derivation of exact and asymptotic results for the mean distance c.d.f. and the $( \mu , s )$ -distortion.

Proposition 2.1. Assume that d $\geq 2$ and $X \ { \overset { d } { \sim } } \ \mathbb { P }$ with P spherically symmetric. Then, for any fixed $\mathbf { u } \in \mathbb { R } ^ { d }$ we have

$$
\begin{array} { r } { \| X - \mathbf { u } \| ^ { 2 } \overset { d } { = } ( \| \mathbf { u } \| - R ) ^ { 2 } + 4 \left\| \mathbf { u } \right\| R \beta _ { \delta , \delta } , } \end{array}\tag{6}
$$

where $\delta = ( d - 1 ) / 2$ and the random variables $R = \| X \|$ and $\beta _ { \delta , \delta }$ are independent. Moreover, when $\mathbf { R } _ { n } \sim \mathsf { P } _ { n } = \mathbb { P } _ { a } ^ { [ n ] }$ with $\mathbb { P } _ { a }$ uniform on $\begin{array} { r } { S _ { d - 1 } ( a ) , a \geq 0 } \end{array}$ , we have

$$
\begin{array} { r } { \begin{array} { r l } & { d ^ { 2 } ( \mathbf { u } , \mathbf { R } _ { n } ) \stackrel { d } { = } ( \left\| \mathbf { u } \right\| - a ) ^ { 2 } + 4 a \left\| \mathbf { u } \right\| \zeta ( n , d ) , } \end{array} } \end{array}\tag{7}
$$

where $\begin{array} { r } { \zeta ( n , d ) = \operatorname* { m i n } _ { i = 1 , \dots , n } \zeta _ { i } } \end{array}$ and the $\zeta _ { i } \overset { d } { = } \beta _ { \delta , \delta }$ are i.i.d.

Proof. As X is spherically symmetric, the distribution of $\| X - \mathbf { u } \|$ only depends on u through $r = \| \mathbf { u } \|$ , and without any loss of generality we can assume that $\mathbf { u } = ( r , 0 , \ldots , 0 )$ . We thus have

$$
\left\| X - \mathbf { u } \right\| ^ { 2 } \overset { d } { = } ( r - X _ { 1 } ) ^ { 2 } + \sum _ { j = 2 } ^ { d } X _ { j } ^ { 2 } = r ^ { 2 } - 2 r X _ { 1 } + R ^ { 2 } = ( R - r ) ^ { 2 } + 2 r R ( 1 - Z _ { 1 } ) ,
$$

where we have denoted $Z _ { 1 } = X _ { 1 } / R$ . From Lemma 2.1, R and $Z _ { 1 }$ are independent and the expression of $\varphi _ { 1 } ( \cdot )$ gives $( 1 - Z _ { 1 } ) / 2 \overset { d } { = } \beta _ { \delta , \delta }$ , which yields (6). Since the n points of ${ \mathbf { R } } _ { n }$ are i.i.d. with $\mathbb { P } _ { a }$ , the representation (6) yields (7). □

From (6), for all $t \geq 0$ we have

$$
\begin{array} { r c l } { \mathbb { P } \left\{ \| X - \mathbf { u } \| \leq t \right\} } & { = } & { \mathsf { P r o b } \left\{ ( R - r ) ^ { 2 } + 4 R r \beta _ { \delta , \delta } \leq t ^ { 2 } \right\} , } \end{array}
$$

with $r = \| \mathbf { u } \| > 0$ , and thus

$$
\mathbb { P } \left\{ \| X - \mathbf { u } \| \leq t \right\} \ = \ \mathsf { P r o b } \left\{ \beta _ { \delta , \delta } \leq \frac { t ^ { 2 } - ( R - r ) ^ { 2 } } { 4 R r } \right\} .
$$

By conditioning on the random variable $R = \| X \|$ and denoting Φ(·) the c.d.f. of $R = \| X \|$ , we obtain

$$
\begin{array} { r c l } { \displaystyle \mathbb { P } \left\{ \| X - \mathbf { u } \| \le t \right\} } & { = } & { \displaystyle \int _ { \rho \ge 0 } \mathsf { P r o b } \left\{ \beta _ { \delta , \delta } \le \frac { t ^ { 2 } - ( \rho - r ) ^ { 2 } } { 4 \rho r } \right\} \mathrm { d } \Phi ( \rho ) } \\ & { = } & { \displaystyle \int _ { \rho \ge 0 } I _ { \upsilon } ( \delta , \delta ) \mathrm { d } \Phi ( \rho ) , } \end{array}\tag{8}
$$

where $\delta = ( d - 1 ) / 2$

$$
v = v ( t , \rho , r ) = [ t ^ { 2 } - ( \rho - r ) ^ { 2 } ] / ( 4 \rho r )\tag{9}
$$

and $I _ { v } ( \cdot , \cdot )$ is the regularised incomplete beta-function with the added convention (5). The integration in (8) is over the support of the distribution of $R = \| X \|$ If $r = \| \mathbf { u } \| = 0$ , then (8) reduces to

$$
\mathbb { P } \left\{ \| X - \mathbf { u } \| \leq t \right\} = \mathbb { P } \left\{ \| X \| \leq t \right\} = \int _ { 0 } ^ { t } \mathrm { d } \Phi ( \rho ) .
$$

When $\mathbb { P } = \mathbb { P } _ { a }$ , the uniform distribution on the sphere $\cal { S } _ { d - 1 } ( a )$ , the expression (8) with the convention (5) can be simplified as follows:

$$
\begin{array} { r } { \mathbb { P } _ { a } \left\{ \left\| X - \mathbf { u } \right\| \leq t \right\} = \left\{ \begin{array} { l l } { 0 } & { \mathrm { i f ~ } t \leq \left| r - a \right| , } \\ { I _ { v } ( \delta , \delta ) } & { \mathrm { i f ~ } \left| r - a \right| \leq t \leq a + r , } \\ { 1 } & { \mathrm { i f ~ } a + r \leq t , } \end{array} \right. } \end{array}\tag{10}
$$

where $r = \| \mathbf { u } \|$ and $v = v ( t , a , r )$ is given by (9).

2.4 $F ( t ; \mathbb { P } ^ { [ n ] } , \mu )$ and $D _ { \mu , s } ( \mathbb { P } ^ { [ n ] } )$ when P is spherically symmetric From (3) and (4), the mean distance c.d.f. $F ( \cdot ; \mathbb { P } ^ { [ n ] } , \mu )$ can be calculated explicitly as

$$
F ( t ; \mathbb { P } ^ { [ n ] } , \mu ) = 1 - \mathsf { E } _ { \mu } \left\{ \left( 1 - \mathbb { P } \left\{ \left\| X - U \right\| \leq t \right\} \right) ^ { n } \right\} ,\tag{11}
$$

where $U \ \stackrel { d } { \sim } \ \mu , \ X \ \stackrel { d } { \sim } \ \mathbb { P } .$ and X and $U$ are independent. When $\mathbb { P }$ is spherically symmetric, P $\{ \| X - \mathbf { u } \| \leq t \}$ is given by (8) which only depends on u through $r =$ $\lvert \lvert \mathbf { u } \rvert \rvert$ , and the expectation with respect to U in (11) is reduced to an expectation with respect to the distribution of $\| U \|$ . The calculation of $F ( t ; \mathbb { P } ^ { [ n ] } , \mu )$ thus amounts to a double integration, with respect to the distributions of $\| X \|$ and $\| U \|$ . From (2), the calculation of $D _ { \mu , s } ( \mathbb { P } ^ { [ n ] } )$ requires an additional integration with respect to t,

$$
\begin{array} { r c l } { { { \cal D } _ { \mu , s } ( { \mathbb P } ^ { [ n ] } ) } } & { { = } } & { { s \displaystyle \int _ { t \geq 0 } t ^ { s - 1 } { \sf E } _ { \mu } \left\{ \left( 1 - { \mathbb P } \left\{ \left\| X - U \right\| \leq t \right\} \right) ^ { n } \right\} { \sf d } t , } } \\ { { } } & { { = } } & { { s \displaystyle \int _ { t \geq 0 } t ^ { s - 1 } \displaystyle \int _ { r \geq 0 } H ( r , t ; \Phi , n ) { \sf d } \Psi ( r ) { \sf d } t , } } \end{array}\tag{12}
$$

where

$$
H ( r , t ; \Phi , n ) = \left( 1 - \int _ { \rho \geq 0 } I _ { v } ( \delta , \delta ) \mathrm { d } \Phi ( \rho ) \right) ^ { n } ,
$$

with $\delta = ( d - 1 ) / 2 , \upsilon = \upsilon ( t , \rho , r )$ given by (9), $\Psi ( \cdot )$ the $\mathrm { c . d . f . }$ of $\| U \|$ for $U \overset { d } { \sim } \mu$ and $\Phi ( \cdot )$ the c.d.f. of $\| X \|$ for $X \stackrel { d } { \sim } \mathbb { P }$

For given $\Psi ( \cdot )$ and $\Phi ( \cdot )$ , the integrals required to compute $D _ { \mu , s } ( \mathbb { P } ^ { [ n ] } )$ can be evaluated with arbitrary precision. The measure $\mu$ does not need to be spherically symmetric; however, $\Psi ( \cdot )$ can be challenging to derive in general cases. When $\mu$ is spherically symmetric, a distribution with minimum expected distortion is also spherically symmetric; see Remark 3.1. In the next section, we demonstrate that determining such a distribution forms a convex problem, for which we derive a necessary and suficient optimality condition and propose an algorithm inspired by optimal design theory.

Remark 2.2. The expression (12) gives the expected $( \mu , s )$ -distortion for random quantisers. Therefore, according to the paradigm of the so-called “probabilistic method” (see, e.g., [1]), for any P there exists at least one non-random n-point set ${ \bf X } _ { n }$ with $( \mu , s )$ -distortion $D _ { \mu , s } ( \mathbf { X } _ { n } ) \leq D _ { \mu , s } ( \mathbb { P } ^ { [ n ] } )$ . See Section 1.3 of the supplementary material for an illustration involving full factorial designs. ◁

## 2.5 An exact result for $s = 2$ when $\mathbb { P }$ is uniform on $\cal { S } _ { d - 1 } ( a )$

When $s = 2$ and $\mathbb { P }$ is the uniform measure $\mathbb { P } _ { a }$ on a sphere $\cal { S } _ { d - 1 } ( a )$ , Proposition 2.1 can be directly applied to express both the expected distortion $D _ { \mu , 2 } ( \mathbb { P } _ { a } ^ { [ n ] } )$ and the optimal radius $a ^ { * } ( d , n , 2 )$

Corollary 2.1. Assume that $d \geq 2$ , that $\mu$ is any probability measure on $\mathbb { R } ^ { d }$ with finite second moment, and that $\mathbf { R } _ { n } \ \overset { d } { \sim } \ \mathbb { P } _ { a } ^ { [ n ] }$ with $\mathbb { P } _ { a }$ uniform on $S _ { d - 1 } ( a ) , ~ a ~ \geq ~ 0$ Then the expected $( \mu , 2 )$ -distortion is exactly

$$
\begin{array} { r } { D _ { \mu , 2 } ( \mathbb { P } _ { a } ^ { [ n ] } ) = \mathsf { E } _ { \mu } \{ \| U \| ^ { 2 } \} + a ^ { 2 } - 2 a \mathsf { E } _ { \mu } \{ \| U \| \} \left( 1 - 2 m _ { n , d } \right) , } \end{array}\tag{13}
$$

where

$$
m _ { n , d } = \int _ { 0 } ^ { 1 } \left[ 1 - I _ { z } ( \delta , \delta ) \right] ^ { n } \mathrm { d } z , \qquad \delta = \frac { d - 1 } { 2 } .\tag{14}
$$

The unique minimiser of $D _ { \mu , 2 } ( \mathbb { P } _ { a } ^ { [ n ] } )$ with respect to $a \geq 0$ is

$$
a ^ { * } ( d , n , 2 ) = \mathsf E _ { \mu } \{ \| U \| \} ( 1 - 2 m _ { n , d } )\tag{15}
$$

and the corresponding minimum expected distortion is

$$
D _ { \mu , 2 } ( \mathbb { P } _ { a ^ { * } } ^ { [ n ] } ) = \mathsf E _ { \mu } \{ \| U \| ^ { 2 } \} - [ a ^ { * } ( d , n , 2 ) ] ^ { 2 } .\tag{16}
$$

Proof. Conditioning on $U = \mathbf { u }$ in (7) and then averaging with respect to U give

$$
\begin{array} { r l r } { D _ { \mu , 2 } ( \mathbb { P } _ { a } ^ { [ n ] } ) } & { = } & { \mathsf E _ { \mu } \left\{ ( \| U \| - a ) ^ { 2 } + 4 a \| U \| m _ { n , d } \right\} , } \end{array}
$$

with $\begin{array} { r } { m _ { n , d } = \mathsf { E } \{ \zeta ( n , d ) \} = \int _ { 0 } ^ { 1 } \left[ 1 - I _ { z } ( \delta , \delta ) \right] ^ { n } } \end{array}$ dz, which yields (13). Moreover, $m _ { n , d } \leq$ $\mathsf E \{ \zeta _ { 1 } \} = 1 / 2$ , so that the minimiser of this strictly convex quadratic function of a is nonnegative and is given by (15). Substitution gives (16). Note that the dependence of the optimal radius on $\mu$ is entirely through the mean radius $\mathsf { E } _ { \mu } \{ \| U \| \}$ □

## 3 Optimisation over all spherically symmetric distributions

## 3.1 A necessary and suficient condition for optimality

For P a spherically symmetric distribution with c.d.f. $\Phi ( \cdot )$ , denote

$$
A ( \Phi ; r , t ) = \int _ { \rho \geq 0 } I _ { v } ( \delta , \delta ) \mathrm { d } \Phi ( \rho )
$$

with $\delta = ( d - 1 ) / 2$ and $v = v ( t , \rho , r )$ given by (9), so that the expected distortion $D _ { \mu , s } ( \mathbb { P } ^ { [ n ] } )$ equals

$$
\mathcal { I } _ { \mu , s } ( \Phi ; n ) = s \int _ { t \geq 0 } t ^ { s - 1 } \int _ { r \geq 0 } \left[ 1 - A ( \Phi ; r , t ) \right] ^ { n } \mathrm { d } \Psi ( r ) \mathrm { d } t\tag{17}
$$

with $\Psi ( \cdot )$ the c.d.f. of $\| U \|$ for $U \ \stackrel { d } { \sim } \ \mu ,$ see (12). We denote by ${ \mathcal { E } } _ { \mu , s } ^ { \infty } ( \Phi , n )$ the expected-distortion eficiency

$$
\mathcal { E } _ { \mu , s } ( \Phi ; n ) = \frac { \mathcal { I } _ { \mu , s } ( \Phi ^ { * } ; n ) } { \mathcal { I } _ { \mu , s } ( \Phi ; n ) } ,\tag{18}
$$

where $\Phi ^ { * }$ is an optimal c.d.f. minimising $\mathcal { I } _ { \mu , s } ( \Phi ; n )$ , with $\mathcal { E } _ { \mu , s } ( \Phi , n ) \in [ 0 , 1 ]$

Definition 3.1. For any $n \geq 1$ and $s > 0 _ { i }$ , and for $\Phi ( \cdot )$ the $c . d . f . \ o f \ \| X \|$ with $X \sim \mathbb { P }$ spherically symmetric, we define the sensitivity function associated with the distortion $D _ { \mu , s } ( \mathbb { P } ^ { [ n ] } )$ by

$$
S _ { \mu , s } \bigl ( \rho ; \Phi , n \bigr ) = n s \int _ { t \geq 0 } t ^ { s - 1 } \int _ { r \geq 0 } \left[ 1 - A \bigl ( \Phi ; r , t \bigr ) \right] ^ { n - 1 } I _ { v } \bigl ( \delta , \delta \bigr ) \mathrm { d } \Psi \bigl ( r \bigr ) \mathrm { d } t ,\tag{19}
$$

where $\delta = ( d - 1 ) / 2 , \upsilon = \upsilon ( t , \rho , r )$ given by (9) and $\Psi ( \cdot )$ is the c.d.f. of $\| U \|$ with $U \overset { d } { \sim } \mu$

The functional $\mathcal { I } _ { \mu , s } ( \cdot ; n )$ and the associated sensitivity function $S _ { \mu , s } ( \cdot ; \Phi , n )$ satisfy the following properties (the denomination “Equivalence Theorem” is a tribute to the result of Kiefer and Wolfowitz [7] in optimal design of experiments).

Theorem 3.1 (Equivalence Theorem). For every $n \geq 1$ and $s > 0$ , the functional $\Phi \mapsto { \mathcal { I } } _ { \mu , s } ( \Phi ; n )$ is convex. $I f \mu$ is supported in $\mathcal { B } _ { d } ( r _ { \operatorname* { m a x } } )$ , then the infimum distortion over all spherically symmetric distributions is attained by a probability measure such that Φ is supported on $[ 0 , r _ { \mathrm { m a x } } ]$ . A spherically symmetric distribution $\mathbb { P } ^ { * }$ such that $\Phi ^ { * }$ is supported on this interval is globally optimal among all spherically symmetric distributions if and only if

$$
S _ { \mu , s } ( \rho ; \Phi ^ { * } , n ) \le \overline { { S } } _ { \mu , s } ( \Phi ^ { * } , n ) \quad f o r \ e v e r y \ \rho \in [ 0 , r _ { \operatorname * { m a x } } ] ,\tag{20}
$$

where

$$
\overline { { S } } _ { \mu , s } ( \Phi , n ) = \int _ { \rho \geq 0 } S _ { \mu , s } ( \rho ; \Phi , n ) \mathrm { d } \Phi ( \rho ) .
$$

In (20), equality holds for $\Phi ^ { * }$ -almost every $\rho$ and the computable quantity

$$
g _ { \mu , s } ( \Phi ; n ) = \operatorname* { m a x } _ { 0 \leq \rho \leq r _ { \operatorname* { m a x } } } S _ { \mu , s } ( \rho ; \Phi , n ) - \overline { { S } } _ { \mu , s } ( \Phi , n )\tag{21}
$$

provides a global optimality certificate in terms of expected-distortion eficiency (18):

$$
1 \geq \mathcal { E } _ { \mu , s } ( \Phi ; n ) \geq 1 - \frac { g _ { \mu , s } ( \Phi ; n ) } { \mathcal { T } _ { \mu , s } ( \Phi ; n ) } .\tag{22}
$$

Proof. The map $\Phi \mapsto A ( \Phi ; r , t )$ is linear, while $x \mapsto ( 1 - x ) ^ { n }$ is convex on $[ 0 , 1 ]$ Integration in $( 1 7 )$ preserves convexity. $\operatorname { I f } \rho > r _ { \operatorname* { m a x } } .$ , replacing a design point $\mathbf { x } = \rho \mathbf { z }$ with $\| \mathbf { z } \| = 1$ by $\mathbf { x } ^ { \prime } = r _ { \mathrm { m a x } } \mathbf { z }$ cannot increase its distance from any $\mathbf { u } \in \mathcal { B } _ { d } ( r _ { \operatorname* { m a x } } )$ 2 since

$$
\| \mathbf { u } - \rho \mathbf { z } \| ^ { 2 } - \| \mathbf { u } - r _ { \operatorname* { m a x } } \mathbf { z } \| ^ { 2 } = ( \rho - r _ { \operatorname* { m a x } } ) [ \rho + r _ { \operatorname* { m a x } } - 2 \mathbf { u } ^ { \top } \mathbf { z } ] \geq 0 .
$$

Hence every spherically symmetric distribution can be projected onto $[ 0 , r _ { \mathrm { m a x } } ]$ without increasing the distortion. The set of probability measures on this compact interval is weakly compact, and the distortion is continuous under weak convergence, a minimiser thus exists.

For two c.d.f. Φ and $\Phi ^ { \prime }$ , diferentiation along $\Phi _ { \epsilon } = ( 1 - \epsilon ) \Phi + \epsilon \Phi ^ { \prime }$ gives

$$
\left. \frac { \mathrm { d } } { \mathrm { d } \epsilon } \mathcal { I } _ { \mu , s } ( \Phi _ { \epsilon } ; n ) \right| _ { \epsilon = 0 + } = \overline { { S } } _ { \mu , s } ( \Phi , n ) - \int _ { \rho \geq 0 } S _ { \mu , s } ( \rho ; \Phi , n ) \mathrm { d } \Phi ^ { \prime } ( \rho ) .\tag{23}
$$

Convexity implies that $\Phi ^ { * }$ is optimal if and only if this derivative is non-negative for every $\Phi ^ { \prime }$ . Taking $\Phi ^ { \prime }$ to be a point mass at $\rho$ shows that the condition (20) is necessary. Conversely, the linearity of (23) in $\Phi - \Phi ^ { \prime }$ implies that when (20) is satisfied, then $\mathrm { d } { \mathcal { I } _ { \mu , s } } [ ( 1 - \epsilon ) \Phi ^ { * } + \epsilon \Phi ^ { \prime } ; n ] / \mathrm { d } { \epsilon } { | _ { \epsilon = 0 + } } \ge 0$ for any $\Phi ^ { \prime }$ , so that $\Phi ^ { * }$ is optimal. The condition (20) is thus suficient. Since its left-hand side has $\Phi ^ { * }$ -average $\overline { { S } } _ { \mu , s } ( \Phi ^ { * } , n )$ , equality must hold $\Phi ^ { * }$ -almost everywhere. Finally, the optimality of $\Phi ^ { * }$ and convexity imply, for any Φ,

$$
\begin{array} { r l } & { \mathcal { I } _ { \mu , s } ( \Phi ; n ) \geq \mathcal { I } _ { \mu , s } ( \Phi ^ { * } ; n ) \geq \mathcal { I } _ { \mu , s } ( \Phi ; n ) + \mathrm { d } \mathcal { I } _ { \mu , s } [ ( 1 - \epsilon ) \Phi + \epsilon \Phi ^ { * } ; n ] / \mathrm { d } \epsilon \big | _ { \epsilon = 0 + } } \\ & { \phantom { \mathcal { I } _ { \mu , s } } \geq \mathcal { I } _ { \mu , s } ( \Phi ; n ) + \underset { 0 \leq \rho \leq r _ { \operatorname* { m a x } } } { \operatorname* { m i n } } \left[ \overline { { S } } _ { \mu , s } ( \Phi , n ) - S _ { \mu , s } ( \rho ; \Phi , n ) \right] , } \end{array}
$$

that is, (22).

Remark 3.1. Spherical symmetry of $\mathbb { P }$ is not required for convexity. Indeed, the expressions in (2), (3), and (4) show that the functional $\mathbb { P } \mapsto D _ { \mu , s } ( { \bar { \mathbb { P } } } ^ { [ n ] } )$ is convex for any measure $\mu$ on $\mathcal { X } _ { i }$ , any $n \geq 1$ and $s > 0$ . However, exploiting convexity to derive optimal distributions is significantly simplified under the spherical symmetry condition. This is not restrictive when $\mu$ is spherically symmetric. Indeed, when $\mu$ is invariant under rotations, since $D _ { \mu , s } ( \mathbb { P } ^ { [ n ] } )$ is convex in $\mathbb { P }$ and invariant under rotations of P, by a standard symmetrisation argument there exists an optimal $\mathbb { P } ^ { * }$ which is rotationally invariant. ◁

As stated in the next corollary, (20) provides a direct numerical test of whether optimisation over a single sphere has lost anything relative to optimisation over arbitrary radial mixtures.

Corollary 3.1. The distribution concentrated at one radius $a \in [ 0 , { r _ { \operatorname* { m a x } } } ]$ is globally optimal among all radial distributions if and only if

$$
S _ { \mu , s } ( \rho ; \delta _ { a } , n ) \leq S _ { \mu , s } ( a ; \delta _ { a } , n ) \quad f o r \ a l l \ \rho \in [ 0 , r _ { \operatorname * { m a x } } ] .
$$

For unbounded target distributions, the equivalence condition of Theorem 3.1 remains valid whenever the diferentiations above are justified by the relevant moment bounds. Once an optimum $\Phi ^ { * }$ has been determined on $[ 0 , r _ { \mathrm { m a x } } ]$ for a large enough r<sub>max</sub>, optimality over $\mathbb { R } ^ { + }$ is guaranteed when (20) is satisfied for all $\rho \geq r _ { \operatorname* { m a x } }$

## 3.2 An optimisation algorithm

Theorem 3.1 yields a constructive procedure. Starting from a finitely supported measure with c.d.f. $\Phi _ { k } .$ , calculate

$$
\rho _ { k } \in \operatorname { A r g } _ { 0 \leq \rho \leq r _ { \operatorname* { m a x } } } S _ { \mu , s } ( \rho ; \Phi _ { k } , n ) .\tag{24}
$$

If $g _ { \mu , s } ( \Phi _ { k } ; n )$ is below a prescribed tolerance, (22) certifies near-optimality. Otherwise set

$$
\Phi _ { k + 1 } = ( 1 - \gamma _ { k } ) \Phi _ { k } + \gamma _ { k } \delta _ { \rho _ { k } } , \qquad \gamma _ { k } \in \mathrm { A r g } \operatorname* { m i n } _ { 0 \leq \gamma \leq 1 } \mathcal { I } _ { \mu , s } [ ( 1 - \gamma ) \Phi _ { k } + \gamma \delta _ { \rho _ { k } } ; n ] .\tag{25}
$$

This is the classical conditional-gradient, or vertex-direction, algorithm on the space of probability measures used in optimal design [4, 16]. Each iteration requires only a one-dimensional maximisation and a one-dimensional line search, preserves finite support, decreases the distortion, and supplies the rigorous stopping certificate $g _ { \mu , s } ( \Phi _ { k } ; n )$ . It can therefore determine whether the best radial law is a single sphere or a mixture of a small number of spheres, without imposing that structure in advance. Several variants of the algorithm have been proposed in the literature; see, $\mathrm { { e . g . , \ [ 1 4 } }$ , Sect. 9.1]. In the Matlab script Optimal Sphere Mixture.m available at https://github.com/lpronzato-prog/Spherically-symmetric-random-designs we exploit the linearity in Φ of the map $\Phi \mapsto A ( \Phi ; r , t )$ to update $\mathcal { I } _ { \mu , s } ( \Phi _ { k } ; n )$ and $S _ { \mu , s } ( \rho ; \Phi _ { k } , n )$ , and alternate between vertex-direction iterations (25) and vertexexchange iterations defined by:

$$
\Phi _ { k + 1 } = \Phi _ { k } + \gamma _ { k } ( \delta _ { \rho _ { k } } - \delta _ { \rho _ { k } ^ { - } } ) ,
$$

where $\gamma _ { k } \in \mathrm { ~ A ~ }$ rg min $^ { 1 } 0 \le \gamma \le w _ { k } ^ { - } \ { \mathcal I } _ { \mu , s } [ \Phi _ { k } + \gamma ( \delta _ { \rho _ { k } } - \delta _ { \rho _ { k } ^ { - } } ) ; n ]$ . Here, $\rho _ { k }$ is given by (24), $\rho _ { k } ^ { - } ~ \in$ Arg min ${ \mathsf { \iota } } _ { \rho \in \operatorname { S u p p } ( \Phi _ { k } ) } S _ { \mu , s } ( \rho ; \Phi _ { k } , n )$ , and $w _ { k } ^ { - }$ is the weight of $\rho _ { k } ^ { - }$ in $\Phi _ { k }$ . Note that when $\gamma = w _ { k } ^ { - }$ , the support point $\rho _ { k }$ replaces $\rho _ { k } ^ { - }$ . Typically, the search domain for including new support points (here, the interval $\lbrack 0 , r _ { \mathrm { m a x } } \rbrack )$ is discretized. In Optimal Sphere Mixture.m, we start with a finite set of candidates for $\rho _ { k }$ . However, if two support points $\rho _ { i } ^ { - }$ and $\mathring { \rho _ { j } } ^ { - }$ of $\Phi _ { k }$ are separated by less than some ϵ, we periodically consider replacing $\Phi _ { k }$ with $\Phi _ { k } - w _ { i } ^ { - } \delta _ { \rho _ { i } ^ { - } } - w _ { j } ^ { - } \delta _ { \rho _ { i } ^ { - } } + ( w _ { i } ^ { - } + w _ { j } ^ { - } ) \delta _ { \rho _ { i } }$ , where $\rho$ is optimally chosen in the interval $[ \rho _ { i } ^ { - } , \rho _ { j } ^ { - } ]$ (and is included in the set of candidates for subsequent iterations). We also periodically remove all support points of $\Phi _ { k }$ that receive negligible weights. We stop the algorithm when $g _ { \mu , s } ( \Phi _ { k } ; n ) / \mathcal { T } _ { \mu , s } ( \Phi _ { k } ; n ) < \epsilon$ for a small $\epsilon ;$ by (22), this certifies that $\mathcal { E } _ { \mu , s } ( \Phi _ { k } ; n ) > 1 - \epsilon$

## 3.3 Three examples

## 3.3.1 $\mu$ is uniform on the unit sphere $\cal { S } _ { d - 1 } ( 1 )$

Here $\mu$ is the spherical measure (uniform on $S _ { d - 1 } ( 1 ) )$ , and the distribution of $\| U \|$ for $U \overset { d } { \sim } \mu$ is the delta measure at $r = 1$ . When $n = 1$ and ${ \bf X } _ { n } = \{ { \bf 0 } _ { d } \}$ , the distance c.d.f. $F ( t ; \mathbf { X } _ { n } , \mu )$ equals 0 for $t < 1$ and 1 for $t \geq 1$

Using the algorithm described above, we obtain that, given s and $d ,$ for very small values of $n > 1$ the optimal distribution is a mixture of the delta measure at the origin and the uniform measure on $\cal { S } _ { d - 1 } ( a )$ for some $a < 1$ . Figure 1 shows $n _ { * } ( d , s )$ , the smallest values of n such that the distribution $\mathbb { P } _ { a ^ { * } }$ ∗ uniform on a single sphere $S _ { d - 1 } ( a ^ { * } )$ is optimal, as a function of $d = 2 , \ldots , 1 5$ for diferent $s \ ( \mathrm { f o r } \ n = 1 $ the degenerate case $\mathbb { P } = \delta _ { \mathbf { 0 } _ { d } }$ is optimal). For all $n \geq n _ { * } ( d , s )$ , the application of Corollary 3.1 shows that the optimal distribution is concentrated (and uniform) on a single sphere $S _ { d - 1 } ( a ^ { * } )$ : we simply need to plot $S _ { \mu , s } ( \rho ; \delta _ { a ^ { * } } , n )$ for $\rho \in [ 0 , 1 ]$ . This is to be contrasted with the situation met in Sections 3.3.2 and 3.3.3. In the special case $s = 2$ and $d = 3 .$ , the optimality of $\mathbb { P } _ { a ^ { * } }$ for any $n \geq 3$ can be proved formally (as then $\delta = 1$ and $I _ { v } ( 1 , 1 ) = v$ for $v \in [ 0 , 1 ] )$ .

![](images/d517e0d2840209f03974011df81862e07cc12e8cdbb4fc7d94f157f854bf5576.jpg)  
Figure 1: Sphere. Smallest $n = n ( d , s )$ such that $\mathbb { P }$ uniform on $S _ { d - 1 } ( a ^ { * } )$ is optimal, for $s = 2 \ ( \ast ) , s = 4 \ ( \star )$ and $s = 1 0 \ ( \bullet )$

The value $a ^ { * } = a ^ { * } ( d , n , s )$ that minimises $D _ { \mu , s } ( \mathbb { P } _ { a } ^ { [ n ] } )$ with respect to a can be obtained numerically with arbitrary precision for any given $d , n$ and s. The expression (10) can be simplified as follows:

$$
\mathbb { P } _ { a } \left\{ \left\| X - \mathbf { u } \right\| \leq t \right\} = \left\{ \begin{array} { l l } { 0 } & { \mathrm { i f ~ } t \leq 1 - a , } \\ { I _ { \upsilon } ( \delta , \delta ) } & { \mathrm { i f ~ } 1 - a \leq t \leq 1 + a , } \\ { 1 } & { \mathrm { i f ~ } 1 + a \leq t , } \end{array} \right.
$$

where $\upsilon = [ t ^ { 2 } - ( 1 - a ) ^ { 2 } ] / ( 4 a )$ . Formula (12) then gives

$$
D _ { \mu , s } ( \mathbb { P } _ { a } ^ { [ n ] } ) = ( 1 - a ) ^ { s } + s \int _ { 1 - a } ^ { 1 + a } t ^ { s - 1 } \left[ 1 - I _ { v } ( \delta , \delta ) \right] ^ { n } \mathrm { d } t .\tag{26}
$$

In the special case $d = 3 , \delta = 1$ and $I _ { v } ( 1 , 1 ) = v \mathrm { ~ f o r ~ } v \in [ 0 , 1 ]$ , so that $D _ { \mu , s } ( \mathbb { P } _ { a } ^ { [ n ] } )$ can be calculated explicitly for s even. It is given by a polynomial in a of degree s, with for example $D _ { \mu , 2 } ( \mathbb { P } _ { a } ^ { [ n ] } ) = ( 1 - a ) ^ { 2 } + 4 a / ( n + 1 )$ ) and $D _ { \mu , 4 } ( \mathbb { P } _ { a } ^ { [ n ] } ) = ( 1 - a ) ^ { 4 } + 8 a [ n ( 1 -$ $a ) ^ { 2 } + 2 + 2 a ^ { 2 } ] / [ ( n + 1 ) ( n + 2 ) ]$ , which gives $a ^ { * } ( 3 , n , 2 ) = ( n - 1 ) / ( n + 1 )$ and $a ^ { * } ( 3 , n , 4 )$ as the root of a cubic equation, satisfying $a ^ { * } ( 3 , n , 4 ) = 1 - 4 / n + 1 6 / n ^ { 2 } + \mathcal { O } ( 1 / n ^ { 3 } )$ when $n \to \infty$ . We thus have $a ^ { * } ( 3 , 3 , 2 ) = 1 / 2$ , and $\mathbb { P } _ { 1 / 2 }$ is globally optimal among all distributions on $\mathbb { R } ^ { 3 }$ , whereas $a ^ { * } ( 3 , 2 , 2 ) \ = \ 1 / 3$ with $\mathbb { P } _ { 1 / 3 }$ being only optimal among all distributions concentrated on a sphere (as $n _ { * } ( 3 , 2 ) = 3$ , see Figure 1).

The left column of Figure 2 presents $a ^ { * } ( d , n , s )$ as a function of d for diferent n and $s = 2 ~ ( \mathrm { t o p } )$ and $s = 1 0$ (bottom). For fixed s and $d , a ^ { * }$ increases with $n ,$ and asymptotically, when n tends to infinity, $a ^ { * }$ tends to 1 (see, e.g., [6, Chap. 9]). Intuitively the choice $a \ : = \ : 1$ seems natural for all n and s. However, Figure 2 indicates that $a = 1$ can be a poor choice, especially in high dimension. For any fixed d and $n , a ^ { * } ( d , n , s )$ is a decreasing function of s, with $a ^ { * } ( d , n , \infty ) = 0$ if n is small enough. Indeed, for $n \leq d , \mathsf { C R } ( \mathbf { X } _ { n } ) \geq 1$ for any $\mathbf { X } _ { n }$ , and the inequality is strict if ${ \bf 0 } _ { d } \not \in { \bf X } _ { n }$ . To see this, choose a unit vector orthogonal to the linear span of the at most d points in $\mathbf { X } _ { n }$ . This vector, viewed as a point of $S _ { d - 1 } ( 1 )$ , is at distance at least one from every point of $\mathbf { X } _ { n } ,$ and at distance strictly greater than one if none of those points is the origin. This implies that $a ^ { * } ( d , n , s ) = 0$ for large enough s when $n \leq d .$ . Also, for fixed n and $s , a ^ { * } ( d , n , s )$ decreases with d. The evolution of $D _ { \mu , s } ( \mathbb { P } _ { a ^ { * } } ^ { [ n ] } )$ as a function of d for diferent n and $s = 1$ , 10 is presented in the supplementary material.

The right column of Figure 2 presents the eficiency of the naive quantiser with n points i.i.d. on $S _ { d - 1 } ( 1 )$ relative to optimised random quantisers with n points on $S _ { d - 1 } ( a ^ { * } )$ . It shows that the benefit of using an appropriate a is significant when d is large or n is small.

![](images/f9981767dc8f8e7c5d6b9d94c33faf4a84920ad16dbb4e621694e29878ecac81.jpg)

![](images/bf385d3b0df3dab73995321a7363e18ccfe6bfa8fe98af0cfcab260180270a77.jpg)

![](images/3419bad11eac33699f6bc301a6aca80e85f18c7f84fcdb705582da1cfe1c480f.jpg)

![](images/fba19069034c7025d9c1087e32843dff2f0160abae11f52213722ff6b8ff86c3.jpg)  
Figure 2: Sphere. Value $a ^ { * }$ minimising $D _ { \mu , s } ( \mathbb { P } _ { a } ^ { [ n ] } )$ (left column) and ratio $D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { a ^ { * } } ^ { [ n ] } ) / D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { 1 } ^ { [ n ] } )$ (right column) as functions of $d { \mathrm { ~ f o r ~ } } s = 2$ (top row) and $s = 1 0$ (bottom row) and diferent values of n: $n = 1 0 \ ( \bullet ) , n = 1 0 ^ { 2 } \ ( \circ ) , n = 1 0 ^ { 3 } \ ( \overline { { \boldsymbol { \eta } } } ) , n = 1 0 ^ { 4 }$ $( \star )$ , and $n = 1 0 ^ { 5 } ~ ( = )$ . On the top-left panel, the dashed lines correspond to the extremevalue approximation (36) of Corollary 4.1.

## 3.3.2 $\mu$ is uniform in the unit ball $\mathcal { B } _ { d } ( 1 )$

For µ uniform in $\mathcal { B } _ { d } ( 1 )$ , the density of $r = \| U \|$ for $U \overset { d } { \sim } \mu$ is $\psi ( \tau ) = d \tau ^ { d - 1 }$ . When $\mathbb { P } = \mathbb { P } _ { a }$ uniform on $s _ { d - 1 } ( a )$ , (12) gives

$$
D _ { \mu , s } ( \mathbb { P } _ { a } ^ { [ n ] } ) = s \int _ { t \geq 0 } t ^ { s - 1 } \int _ { r \geq 0 } H ( r , t ; \Phi _ { a } , n ) \psi ( r ) \mathrm { d } r \mathrm { d } t ,\tag{27}
$$

where $H ( r , t ; \Phi _ { a } , n ) = \left( 1 - \mathbb { P } _ { a } \left\{ \left\| X - \mathbf { u } \right\| \le t \right\} \right) ^ { n }$ with $\Phi _ { a } ( \cdot )$ the c.d.f. of the delta measure $\delta _ { a }$ and $r = \| \mathbf { u } \|$ , so that $\mathbb { P } _ { a } \{ \| X - \mathbf { u } \| \leq t \}$ is given by (10). We also consider random quantisers whose n points are i.i.d. with $\mathbb { P } = \mathbb { P } _ { 0 , b }$ uniform in $\mathcal { B } _ { d } ( b )$ . The c.d.f. $\Phi ( \cdot )$ of $R = \| X \|$ has then the density $\phi _ { 0 , b } ( \rho ) = d \rho ^ { d - 1 } / b ^ { d }$ for $\rho \in [ 0 , b ]$ . Plots of $b ^ { * }$ minimising $D _ { \mu , s } ( \mathbb { P } _ { 0 , b } ^ { [ n ] } )$ and of $D _ { \mu , s } ( \mathbb { P } _ { 0 , b ^ { * } } ^ { [ n ] } )$ as functions of d for diferent n and $s = 1 , 1 0$ are presented in the supplementary material.

For n small enough, a distribution $\mathbb { P } _ { a ^ { * } }$ with $a ^ { * } = a ^ { * } ( d , n , s )$ , uniform on a single sphere, is optimal. However, Zador’s theorem indicates that $\mathbb { P } = \mu$ is asymptotically optimal. Consequently, distributions concentrated on a single sphere cease to be optimal for suficiently large n. Below we show that this n grows exponentially fast with d.

The algorithm described in Section 3.2 produces optimal $\mathbb { P } ^ { * }$ that are mixtures of uniform distributions on concentric spheres with radii $\rho _ { i }$ . The number of spheres increases (slowly) with $n ,$ and the weight $w _ { i }$ allocated to each $\rho _ { i }$ increases with $\rho _ { i }$ (as expected, since $\mathbb { P } ^ { * }$ should increasingly resemble $\mu$ as n grows); see Figure 4 for an illustration.

We first consider the case $s = 2$ and $d = 8$ . The algorithm is stopped when $\mathcal { E } _ { \mu , s } ( \Phi _ { k } ; n ) > 0 . 9 9 9 9$ , see (22); we therefore identify the computed distribution with the optimum $\mathbb { P } ^ { * }$ in the discussion below (for comparison, the expected-distortion eficiency for $\mathbb { P } = \mu$ is only about 0.9296 for $n = 1 0 0 0$ and 0.9631 for $n = 1 0 0 0 0 0 )$ Figure 3 shows the sensitivity function $S _ { \mu , s } ( \rho ; \Phi ^ { * } , n )$ as a function of $\rho \in [ 0 , 1 ]$ when $n = 1 0 0 0$ (two concentric spheres) and $n = 1 0 0 0 0 0$ (four spheres). The vertical dashed lines indicate the positions of the support points $\rho _ { i }$ of $\Phi ^ { * }$ (for which $S _ { \mu , s } ( \rho _ { i } ; \Phi ^ { * } , n ) = \overline { { S } } _ { \mu , s } ( \Phi ^ { * } , n )$ ; see Theorem 3.1). Figure 4 presents the optimal c.d.f. $\Phi ^ { * }$ for both design sizes.

![](images/bf8c93199d53be23dc661ae6be66d583b3d43bd573162749794e5a1ebf05b688.jpg)

![](images/9455be26e0267115ead260ac28809a3c98b1192f550bf401a4eff5f6beb837c0.jpg)  
Figure 3: Ball, $d = 8 .$ . Sensitivity function $S _ { \mu , 2 } ( \rho ; \Phi ^ { * } , n )$ when $n = 1 0 0 0$ (left) and $n = 1 0 0 0 0 0 \ \mathrm { ( r i g h t ) }$ .

Below we address the following questions:

• for which $n = n _ { 1 } ( d , s )$ does $\mathbb { P } _ { a ^ { * } }$ , the uniform distribution on a sphere with optimal radius $a ^ { * } ( d , n , s )$ , cease being optimal?

• for which $n = n _ { 2 } ( d , s )$ does $\mathbb { P } _ { a } .$ ∗ cease to perform better than the uniform distribution $\mathbb { P } _ { 0 , b ^ { * } }$ on a ball with optimised radius $b ^ { * } ( d , n , s ) ?$

• for which $n = n _ { 3 } ( d , s )$ does $\mathbb { P } _ { a } .$ ∗ cease to perform better than the uniform distribution $\mathbb { P } _ { 0 , 1 } = \mu$ (which is asymptotically optimal according to Zador’s theorem)?

![](images/faed538f7ad95eaa2da816e41f770b7a496795a6d20fcfdb85839a994d154e68.jpg)

![](images/91f8fa0b4ab3071f00c4be4729a751743b7976d142857adfd8899832594f5fd4.jpg)  
Figure 4: Ball, $s = 2 , d = 8 . \ \Phi ^ { * } \ ( - \bullet )$ and c.d.f. for $\mathbb { P } = \mu \ ( - \ - \ \llap { / } \llap { / } )$ when n = 1 000 (left) and $n = 1 0 0 0 0 0$ (right). The locations of the support points of $\Phi ^ { * }$ are indicated by •.

The left panel of Figure 5 presents the evolution of $n _ { 1 } ( d , s ) , n _ { 2 } ( d , s )$ and $n _ { 3 } ( d , s )$ (on a logarithmic scale, obtained by binary search) as functions of d for $s = 2$ (see the supplementary material for plots of $n _ { 2 } ( d , s )$ for other values of s). The behaviour is qualitatively similar for other values of s, with a slightly slower increase with d as s grows. Notably, the plot of $n _ { 1 } ( d , 2 )$ almost perfectly coincides with the dashed line representing $2 ^ { d + 1 }$ . For moderate design sizes, the optimal distribution is uniform on a single sphere. This distribution remains preferable to a uniform distribution on a ball, and hence to $\mu$ itself, until n reaches astronomically large values when d is large.

![](images/29e4069148322b98d5c3997879c7a264e94ec50da5d69af544c99fc27be1e7c7.jpg)

![](images/0909faa01a6c17957d9818962062fc3a88809d2fcd2e62ac607a4d15a1725077.jpg)  
Figure 5: $n _ { 1 } ( d , 2 )$ (♦), $n _ { 2 } ( d , 2 )$ (•) and $n _ { 3 } ( d , 2 ) ~ ( { \star } )$ as functions of $d ,$ for $\mu$ uniform on $\mathcal { B } _ { d } ( 1 )$ (left) and $\mu$ the normal distribution $\mathcal { N } ( \mathbf { 0 } _ { d } , \mathbf { I } _ { d } / d )$ (right).

## 3.3.3 $\mu$ is a spherically symmetric normal distribution

Any spherically symmetric normal distribution in $\mathbb { R } ^ { d }$ can be renormalised as $\mathcal { N } ( \mathbf { 0 } _ { d } , \mathbf { I } _ { d } / d )$ In this section, we assume that $\mu$ is the corresponding probability measure. When $U \stackrel { d } { \sim } \mu , d \| U \| ^ { 2 }$ has the chi-square distribution with d degrees of freedom, so that the density of $\| U \|$ is

$$
\psi ( r ) = \frac { d ^ { d / 2 } } { 2 ^ { d / 2 - 1 } \Gamma ( d / 2 ) } r ^ { d - 1 } \mathrm { e } ^ { - d r ^ { 2 } / 2 } , r \geq 0 .\tag{28}
$$

The distribution of $r$ is more and more concentrated around 1 as d increases. Its k-th moment is

$$
M _ { \psi , k } = \mathsf E _ { \mu } \{ \| U \| ^ { k } \} = ( 2 ^ { k / 2 } / d ^ { k / 2 } ) \Gamma ( ( k + d ) / 2 ) / \Gamma ( d / 2 ) , k = 1 , 2 , . . .
$$

so that its mean is smaller than 1 for any d and its variance is $1 / ( 2 d ) - 1 / ( 8 d ^ { 2 } ) +$ $\mathcal { O } ( 1 / d ^ { 3 } ) , d \to \infty$

When $\mathbb { P } = \mathbb { P } _ { a }$ uniform on $\cal { S } _ { d - 1 } ( a )$ , the expected $( \mu , s )$ -distortion $D _ { \mu , s } ( \mathbb { P } _ { a } ^ { [ n ] } )$ is given by (27), with $\psi ( \cdot )$ given by (28). When $\mathbb { P } = \mu _ { \sigma }$ is the probability measure of $\mathcal { N } ( \mathbf { 0 } _ { d } , \sigma ^ { 2 } \mathbf { I } _ { d } / d ) , R = \| X \|$ has the p.d.f. $\phi _ { \sigma } ( \rho ) = ( 1 / \sigma ) \psi ( \rho / \sigma )$ , where $\psi ( \cdot )$ is given by (28), and the expected $( \mu , s )$ -distortion $D _ { \mu , s } ( \mu _ { \sigma } ^ { [ n ] } )$ is given by (12) where

$$
H ( r , t ; \Phi , n ) = \left( 1 - \frac { 1 } { \sigma } \int _ { \rho \geq 0 } I _ { v } ( \delta , \delta ) \psi ( \rho / \sigma ) \mathrm { d } \rho \right) ^ { n } ,
$$

with $v = v ( t , \rho , r )$ given by (9).

We denote by $\sigma ^ { * } = \sigma ^ { * } ( d , n , s )$ the value of $\sigma$ that minimises the distortion $D _ { \mu , s } ( \mu _ { \sigma } ^ { \left[ n \right] } )$ and by $a ^ { * } = a ^ { * } ( d , n , s )$ the value of a that minimises $D _ { \mu , s } ( \mathbb { P } _ { a } ^ { [ n ] } )$ , as in previous sections. The empirical distribution of an optimal quantiser $\mathbf { X } _ { n } ^ { * }$ minimising $E _ { \mu , s } ( \mathbf { X } _ { n } )$ weakly converges to $\mathcal { N } ( \mathbf { 0 } _ { d } , ( \sigma _ { \infty } ^ { * } ) ^ { 2 } \mathbf { I } _ { d } / d )$ as $n \to \infty$ , with $\sigma _ { \infty } ^ { * } = \sqrt { 1 + s / d } ;$ see [6, Table 7.1].

We take $d = 1 0$ and $s = 5$ . The left panel of Figure 6 presents the sensitivity function $S _ { \mu , s } ( \rho ; \Phi ^ { * } , n )$ as a function of $\rho \in [ 0 , 2 ]$ when $n = 1 0 0 0$ , showing that the optimal distribution is a mixture of two uniform distributions on concentric spheres; their radii are indicated by vertical dashed lines. The right panel shows the c.d.f. obtained when $\mathcal { E } _ { \mu , s } ( \Phi _ { k } ; n ) > 0 . 9 9 9 9$ in the algorithm of Section 3.2; see (22). In the same situation, the expected-distortion eficiency for $\mathbb { P } = \mu _ { \sigma _ { \infty } ^ { * } }$ is only about 0.7828.

![](images/5b4a1c16d6c5731b971845a11c5f2ec3829ddad6aba8dcc5b18025aa4635e24e.jpg)

![](images/7136f9dada304b945c5f519c2d6b134666cc57a804de9ad2950cbfe2fac88dfc.jpg)  
Figure 6: Normal distribution, $s = 5 , d = 1 0 , n = 1 0 0 0$ . Left: Sensitivity function $S _ { \mu , 5 } ( \rho ; \Phi ^ { * } , n )$ . Right: $\Phi ^ { * } \left( - * \right)$ and c.d.f. for $\mathbb { P } = \mu _ { \sigma _ { \infty } ^ { * } } \mathrm { ~ ( - ~ -- ~ } \emptyset \mathrm { ) }$ ; the locations of the support points of $\Phi ^ { * }$ are indicated by •.

The left panel of Figure 7 illustrates that $\sigma ^ { * } = \sigma ^ { * } ( d , n , s )$ increases with $s ,$ similarly to $\sigma _ { \infty } ^ { * }$ (shown as a dashed line), and that $a ^ { * } = a ^ { * } ( d , n , s )$ also increases with s, unlike in Section 3.3.1. The reason is that large values of $r = \| U \|$ receive increasing weight as s increases, while $r = 1$ in Section 3.3.1. The right panel shows that $\sigma ^ { * } - a ^ { * }$ tends to zero as d increases, a consequence of $\mu _ { \sigma }$ tending to $\mathbb { P } _ { \sigma }$ as $d \to \infty$

Finally, similarly to Section 3.3.2, we compute the values of $n _ { 1 } ( d , s ) , n _ { 2 } ( d , s )$ and $n _ { 3 } ( d , s )$ at which the following transitions occur, respectively: $\mathbb { P } _ { a ^ { * } }$ , the uniform distribution on a sphere with optimal radius $a ^ { * } ( d , n , s )$ , ceases to be optimal; $\mathbb { P } _ { a ^ { * } }$ ceases to outperform an optimised normal distribution $\mu _ { \sigma ^ { * } } ; \mathbb { P } _ { a ^ { * } }$ ceases to outperform the asymptotically optimal distribution $\mu _ { \sigma _ { \infty } ^ { * } }$ . Their values for $s = 2$ are plotted as functions of d on the right panel of Figure 5 (see the supplementary material for other values of $s )$ . While the rate of increase with d is smaller than for $\mu$ uniform in $\mathcal { B } _ { d } ( 1 )$ (left panel), the qualitative behaviour remains similar.

![](images/e79145abd493a9d7e5f9fd4f4bfdc944738348cea44c0403da32bfead87b7d3a.jpg)

![](images/311996a3d0c89f3d598aa05d53962c948e861918375762e5c08833a56e0d003b.jpg)  
Figure 7: Optimal values $a ^ { * } ( d , n , s )$ (•) and $\sigma ^ { * } ( d , n , s )$ (♦) as functions of $s$ for $d = 1 0$ and $n = 1$ 000 (left panel, with $\sigma _ { \infty } ^ { * }$ shown as a dashed line), and as functions of d for $s = 2$ and $n = 1 0 0 0 0$ (right panel).

## 4 Extreme-value approximations

In this section, extreme-value theory is used to derive approximations of the $( \mu , s ) \dash$ distortion $D _ { \mu , s } ( \mathbb { P } _ { a } ^ { [ n ] } )$ , where $\mathbb { P } _ { a }$ is uniform on $\cal { S } _ { d - 1 } ( a )$ (Section 4.1). In Section 4.2, three asymptotic regimes are identified that govern the limiting behaviour of $D _ { \mu , s } ( \mathbb { P } _ { a } ^ { [ n ] } )$ when n increases with d. This is illustrated in Section 4.3 for the case where $\mu$ is uniform on $S _ { d - 1 } ( 1 )$ . Further examples, including the case where $\mu$ is spherically normal, are provided in the supplementary material.

## 4.1 Uniform quantisers on a sphere

Consider random quantisers $\mathbf { R } _ { n }$ with distribution $\mathsf { P } _ { n } = \mathbb { P } _ { a } ^ { [ n ] }$ where $\mathbb { P } _ { a }$ is uniform on $\cal { S } _ { d - 1 } ( a )$ . For large $n ,$ the distribution of $\zeta ( n , d )$ in $( 7 )$ can be approximated using standard results from extreme-value theory. Here we assume $d \geq 3$ to avoid singularity in beta distributions.

Lemma 4.1. Assume that $d \geq 3$ and $\mathsf { P } _ { n } \ = \ \mathbb { P } _ { a } ^ { [ n ] }$ , with $a \ > \ 0$ . For any fixed u $E \in \mathbb { R } ^ { d } \backslash \{ \mathbf { 0 } _ { d } \}$ , we have

$$
\frac 1 { 4 a \left\| \mathbf { u } \right\| \kappa _ { n , d } } \left[ d ^ { 2 } ( \mathbf { u } , \mathbf { R } _ { n } ) - ( \left\| \mathbf { u } \right\| - a ) ^ { 2 } \right] \overset { \mathrm { d } } { \to } \xi _ { d } , n \to \infty ,\tag{29}
$$

where $\xi _ { d }$ is a random variable with Weibull $c . d . f . \ F _ { d } ( t ) = 1 - \exp ( - t ^ { \delta } ) , t \geq 0$ , and moments $\mathsf E \{ \xi _ { d } ^ { k } \} = \Gamma ( 1 + k / \delta )$ with $\delta = ( d - 1 ) / 2$ . Here $\kappa _ { n , d }$ is the $1 / n$ quantile $o f$ $\beta _ { \delta , \delta }$ , defined by $I _ { \kappa _ { n , d } } ( \delta , \delta ) = 1 / n$

Proof. From standard extreme-value theory $( \sec , \mathrm { e . g . } , [ 8 , \mathrm { p . ~ 5 9 } ] )$ , the random variable $\zeta ( n , d ) / \kappa _ { n , d }$ in (7) converges in distribution to the random variable $\xi _ { d } . \quad \boxed { \begin{array} { r l } \end{array} }$

Remark 4.1. Lemma 4.1 yields the following approximations for the s-moment of $d ( \mathbf { u } , \mathbf { R } _ { n } )$ for large n and any given u:

$$
M _ { s } ( d , n , a , r ) = \int _ { \zeta \geq 0 } \left[ ( r - a ) ^ { 2 } + 4 a r \kappa _ { n , d } \zeta \right] ^ { s / 2 } \mathrm { d } F _ { d } ( \zeta ) ,
$$

with $r = \| \mathbf { u } \|$ . Explicit expressions are easily obtained for even s. Assuming that $\| \mathbf { u } \| = 1$ , this gives in particular the approximations

$$
\begin{array} { r c l } { { \mathsf { v a r } _ { 2 } } } & { { = } } & { { 1 6 a ^ { 2 } \kappa _ { n , d } ^ { 2 } \mathsf { v a r } \{ \xi _ { d } \} = 1 6 a ^ { 2 } \kappa _ { n , d } ^ { 2 } \left[ \Gamma ( 1 + 2 / \delta ) - \Gamma ^ { 2 } ( 1 + 1 / \delta ) \right] , } } \\ { { \mathsf { v a r } _ { 4 } } } & { { = } } & { { 6 4 a ^ { 2 } \kappa _ { n , d } ^ { 2 } \left\{ ( 1 - a ) ^ { 4 } \mathsf { v a r } \{ \xi _ { d } \} + 4 a ^ { 2 } \kappa _ { n , d } ^ { 2 } \mathsf { v a r } \{ \xi _ { d } ^ { 2 } \} \right. } } \\ { { } } & { { } } & { { \left. + 4 ( 1 - a ) ^ { 2 } a \kappa _ { n , d } \left[ \Gamma ( 1 + 3 / \delta ) - \Gamma ( 1 + 1 / \delta ) \Gamma ( 1 + 2 / \delta ) \right] \right\} . } } \end{array}
$$

$f o r v a r { _ s } = { \mathsf { v a r } } \{ d ^ { s } ( { \mathbf { u } } , { \mathbf { R } } _ { n } ) \}$ when $s = 2$ and $s = 4$ , respectively.

For each fixed u, equation (29) gives the limiting distribution of the suitably normalised nearest-neighbour distance. Averaging this pointwise extreme-value approximation over a spherically symmetric target $\mu$ motivates the following approximation of $D _ { \mu , s } ( \mathbb { P } _ { a } ^ { [ n ] } )$ ).

Definition 4.1. Assume that $d \geq 3$ and that $\mu$ is spherically symmetric. $T h e$ extreme-value approximation of the expected $( \mu , s )$ -distortion $D _ { \mu , s } ( \mathbb { P } _ { a } ^ { [ n ] } )$ is defined $b y$

$$
\widehat { E } _ { \mu , s } ^ { s } ( n ; a ) = \int _ { r \geq 0 } \int _ { \zeta \geq 0 } \left[ ( r - a ) ^ { 2 } + 4 a r \kappa _ { n , d } \zeta \right] ^ { s / 2 } \mathrm { d } F _ { d } ( \zeta ) \mathrm { d } \Psi ( r ) ,\tag{30}
$$

where $\Psi ( \cdot )$ is the $c . d . f .$ of $\| U \|$ with $U \overset { d } { \sim } \mu , F _ { d } ( t ) = 1 - \exp ( - t ^ { \delta } )$ with $\delta = ( d - 1 ) / 2 $ and $\kappa _ { n , d }$ is the $1 / n$ quantile $o f \beta _ { \delta , \delta }$ defined by $I _ { \kappa _ { n , d } } ( \delta , \delta ) = 1 / n$ . The extreme-value approximation $\widehat { D } _ { \mu , s } ( \mathbb { P } _ { a } ^ { [ n ] } )$ of $D _ { \mu , s } ( \mathbb { P } _ { a } ^ { [ n ] } )$ is defined by $\widehat { D } _ { \mu , s } ( \mathbb { P } _ { a } ^ { [ n ] } ) = \widehat { E } _ { \mu , s } ^ { s } ( n ; a )$

The approximation (30) is derived from the following considerations. Since $\mu$ is spherically symmetric, we can write $U \stackrel { d } { \sim } r \cdot U ^ { ( d ) }$ where $U ^ { ( d ) }$ is uniformly distributed on $S _ { d - 1 } ( 1 ) , r$ has the c.d.f. $\Psi ( \cdot )$ and the two variables are independent. Replacing the conditional distribution of $d ^ { 2 } ( U , { \bf R } _ { n } )$ , given $r = \| U \|$ , by the extreme-value limit in (29), and then averaging with respect to r and the Weibull variable, yields (30). This is an approximation, not an exact identity of means. A formal proof that the variability of $E _ { \mu , s } ^ { s } ( \mathbf { R } _ { n } )$ vanishes asymptotically is provided below in Theorem 4.1. We first introduce some notation.

Let $Z _ { 1 } , \ldots , Z _ { n }$ be i.i.d. uniform on $S _ { d - 1 } ( 1 )$ and let ${ \bf R } _ { n , a }$ be a random quantiser on $\cal { S } _ { d - 1 } ( a )$

$$
\mathbf { R } _ { n , a } = \{ a Z _ { 1 } , \ldots , a Z _ { n } \} \stackrel { d } { \sim } \mathbb { P } _ { a } ^ { [ n ] } .
$$

Denote $Q _ { n , a } = E _ { \mu , s } ^ { s } ( \mathbf { R } _ { n , a } )$ the realised distortion of this random quantiser, so that $\mathsf E \{ Q _ { n , a } \} = D _ { \mu , s } ( \mathbb { P } _ { a } ^ { [ n ] } )$ . For $U \stackrel { d } { \sim } \mu .$ define

$$
B _ { \mu , s } ( a ) = \mathsf E _ { \mu } \{ ( \| U \| + a ) ^ { s } \} .
$$

For $n \geq 2$ , let $V _ { i , n }$ be the spherical Voronoi cell of $Z _ { i }$ and let $p _ { i , n }$ be its normalised spherical measure (ties may be resolved arbitrarily, since they form a set of spherical measure zero almost surely). Denote by $v _ { d } ( \theta )$ the normalised surface measure of a spherical cap of angular radius $\theta ,$ and by $N _ { d } ( \theta )$ the smallest number of such caps required to cover $S _ { d - 1 } ( 1 )$

Theorem 4.1. For any $d \geq 2 , n \geq 2 , s > 0 , a > 0$ , and $\mu$ such that $B _ { \mu , s } ( a ) < \infty$ we have

$$
{ \mathsf { v a r } } \{ Q _ { n , a } \} \leq B _ { \mu , s } ^ { 2 } ( a ) \mathsf { E } \left\{ \sum _ { i = 1 } ^ { n } p _ { i , n } ^ { 2 } \right\} \leq B _ { \mu , s } ^ { 2 } ( a ) \mathsf { E } \left\{ \operatorname* { m a x } _ { 1 \leq i \leq n } p _ { i , n } \right\} ,\tag{31}
$$

and

$$
{ \mathsf { v a r } } \{ Q _ { n , a } \} \leq B _ { \mu , s } ^ { 2 } ( a ) B ( d , n ) ,\tag{32}
$$

where

$$
B ( d , n ) = \operatorname* { i n f } _ { \theta \in ( 0 , \pi ) } \left\{ v _ { d } ( \theta ) + N _ { d } ( \theta / 2 ) \exp [ - n v _ { d } ( \theta / 2 ) ] \right\} .
$$

Therefore,

$$
\mathsf { P r o b } \{ | Q _ { n , a } - D _ { \mu , s } ( \mathbb { P } _ { a } ^ { [ n ] } ) | \ge t \} \le \frac { B _ { \mu , s } ^ { 2 } ( a ) B ( d , n ) } { t ^ { 2 } } ,
$$

and for every fixed d,

$$
\begin{array} { r } { Q _ { n , a } - D _ { \mu , s } ( { \mathbb P } _ { a } ^ { [ n ] } ) \to 0 \quad i n L _ { 2 } \ a n d \ i n \ p r o b a b i l i t y \ a s \ n \to \infty . } \end{array}\tag{33}
$$

Proof. Let $Q _ { n , a } ^ { ( - i ) }$ denote the distortion obtained after deleting the point $a Z _ { i }$ . This random variable is independent of $Z _ { i }$ conditionally on the other $n - 1$ points. Deleting $a Z _ { i }$ changes the nearest-neighbour distance only for directions in $V _ { i , n }$ . Since the distance between a point of norm r and any point of norm a is at most $r + a ,$ spherical symmetry and independence of the radial and angular components $\mathrm { g i v e }$

$$
0 \leq Q _ { n , a } ^ { ( - i ) } - Q _ { n , a } \leq B _ { \mu , s } ( a ) p _ { i , n } .
$$

The deletion form of the Efron–Stein inequality, see [2], therefore yields

$$
\begin{array} { r l r } {  { \mathsf { v a r } \{ Q _ { n , a } \} \le \sum _ { i = 1 } ^ { n } \mathsf { E } \{ ( Q _ { n , a } ^ { ( - i ) } - Q _ { n , a } ) ^ { 2 } \} } \ } \le  & { { \cal B } _ { \mu , s } ^ { 2 } ( a ) \mathsf { E } \{ \sum _ { i = 1 } ^ { n } p _ { i , n } ^ { 2 } \} } \\ & { } & \\ & { } & { \le B _ { \mu , s } ^ { 2 } ( a ) \mathsf { E } \{ \underset { 1 \leq i \leq n } { \operatorname* { m a x } } p _ { i , n } \} , } \end{array}
$$

where the last inequality follows from $\textstyle \sum _ { i } p _ { i , n } = 1$ . This completes the proof of (31).

Let $\omega _ { n } = \mathrm { m a x } _ { z \in S _ { d - 1 } ( 1 ) }$ min arccos $( z ^ { \top } Z _ { i } )$ be the angular covering radius. Every Voronoi cell is contained in a cap of radius $\omega _ { n }$ , and hence ma $\mathrm { x } _ { i } p _ { i , n } ~ \leq ~ v _ { d } ( \omega _ { n } )$ Now take a covering of the sphere by $N _ { d } ( \theta / 2 )$ caps of radius $\theta / 2$ and centres $c _ { k } .$ $k = 1 , \dots , N _ { d } ( \theta / 2 )$ . Assume that each cap contains at least one $Z _ { i }$ . Then, any $z \in  { S } _ { d - 1 } ( 1 )$ belongs to some cap with centre $c _ { k }$ and radius $\theta / 2$ and this cap also contains a $Z _ { i }$ , implying that $\omega _ { n } \leq \theta .$ . A union bound gives

$$
\mathsf { P r o b } \{ \omega _ { n } > \theta \} \le N _ { d } ( \theta / 2 ) [ 1 - v _ { d } ( \theta / 2 ) ] ^ { n } \le N _ { d } ( \theta / 2 ) e ^ { - n v _ { d } ( \theta / 2 ) } .
$$

It follows that $\mathsf { E } \{ \mathsf { m a x } _ { i } p _ { i , n } \} \leq v _ { d } ( \theta ) + \mathsf { P r o b } \{ \omega _ { n } > \theta \}$ , proving (32).

For fixed d, the i.i.d. directions are almost surely dense on the sphere, so $\omega _ { n }  0$ and max<sub>i</sub> $p _ { i , n } \to 0$ almost surely. Dominated convergence in (31) proves (33).

For s even, $\widehat { E } _ { \mu , s } ^ { s } ( n ; a )$ can be expressed explicitly in terms of moments of $\| U \|$ $\begin{array} { r } { M _ { \Psi , k } \ = \ \mathsf E _ { \mu } \{ \| U \| ^ { k } \} \ = \ \int _ { r > 0 } r ^ { k } \mathrm { d } \Psi ( r ) } \end{array}$ , and moments ${ \sf E } \{ \zeta ^ { k } \} = \Gamma ( 1 + k / \delta )$ of the Weibull distribution. In particular, for $s = 2$ and 4 we get the following extremevalue approximations for the $( \mu , s )$ -distortions:

$$
\begin{array} { r l r } { \widehat { E } _ { \mu , 2 } ^ { 2 } ( n ; a ) } & { = } & { M _ { \Psi , 2 } - 2 a M _ { \Psi , 1 } + a ^ { 2 } + 4 a \kappa _ { n , d } M _ { \Psi , 1 } \Gamma ( 1 + 1 / \delta ) , } \end{array}\tag{34}
$$

$$
\begin{array} { r c l } { { \widehat { E } _ { \mu , 4 } ^ { 4 } ( n ; a ) } } & { { = } } & { { M _ { \Psi , 4 } - 4 a M _ { \Psi , 3 } + 6 a ^ { 2 } M _ { \Psi , 2 } - 4 a ^ { 3 } M _ { \Psi , 1 } + a ^ { 4 } } } \\ { { } } & { { } } & { { + 8 a \kappa _ { n , d } \Gamma ( 1 + 1 / \delta ) \left( M _ { \Psi , 3 } - 2 a M _ { \Psi , 2 } + a ^ { 2 } M _ { \Psi , 1 } \right) } } \\ { { } } & { { } } & { { + 1 6 a ^ { 2 } \kappa _ { n , d } ^ { 2 } M _ { \Psi , 2 } \Gamma ( 1 + 2 / \delta ) . } } \end{array}\tag{35}
$$

Denote by $\widehat { a ^ { * } } = \widehat { a ^ { * } } ( d , n , s )$ the value of a that minimises $\widehat { E } _ { \mu , s } ( n ; a ) \mathrel { \mathop : }$ ; it is an extreme-value approximation of the exact minimiser $a ^ { * }$ of $D _ { \mu , s } ( \mathbb { P } _ { a } ^ { [ n ] } )$ . For $s = 2$ Lemma 4.1 gives $m _ { n , d } \simeq \kappa _ { n , d } \Gamma ( 1 + 1 / \delta )$ in (14). The next corollary is therefore the extreme-value counterpart of Corollary 2.1.

Corollary 4.1. For $d \geq 3 , \mu$ spherically symmetric and $\mathbf { R } _ { n } \overset { d } { \sim } \mathbb { P } _ { a } ^ { [ n ] }$ , we have

$$
\begin{array} { r c l } { { \widehat { a ^ { * } } = \widehat { a ^ { * } } ( d , n , 2 ) } } & { { = } } & { { M _ { \Psi , 1 } \left[ 1 - 2 \kappa _ { n , d } \Gamma ( 1 + 1 / \delta ) \right] , } } \\ { { \widehat { E } _ { \mu , 2 } ^ { 2 } ( n ; \widehat { a ^ { * } } ) } } & { { = } } & { { M _ { \Psi , 2 } - ( \widehat { a ^ { * } } ) ^ { 2 } . } } \end{array}\tag{36}
$$

The top-left panel of Figure 2 presents plots of $a ^ { * } ( d , n , 2 )$ and $\widehat { a ^ { * } } ( d , n , 2 )$ for various n and d when $\mu$ is uniform on $S _ { d - 1 } ( 1 )$ , showing the fast convergence of $\widehat { a ^ { * } }$ to $a ^ { * }$ as n increases. On the top-right panel, the plots of $D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { a ^ { * } } ^ { [ n ] } ) / D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { 1 } ^ { [ n ] } )$ and $D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { a ^ { * } } ^ { [ n ] } ) / D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { 1 } ^ { [ n ] } )$ are indistinguishable. In the right panel of Figure 7, where $\mu$ is normal, the plots of $a ^ { * } ( d , n , 2 )$ and $\widehat { a ^ { * } } ( d , n , 2 )$ are also indistinguishable. Other illustrations are provided in the supplementary material.

The expressions for $\widehat { a ^ { * } } = \widehat { a ^ { * } } ( d , \widehat { n , } 4 )$ and $\widehat { E } _ { \mu , 4 } ^ { 4 } ( n ; \widehat { a ^ { * } } )$ can also be written in analytic form (by finding the roots of a third-degree polynomial), but the expressions are cumbersome.

## 4.2 Three asymptotic regimes for n growing with d

In view of Lemma 4.1, the asymptotic behaviour of $\kappa _ { n , d }$ is the key element in understanding that of $d ^ { 2 } ( \mathbf { u } , \mathbf { R } _ { n } )$ . For $d = 3 , \beta _ { 1 , 1 }$ is uniform on $[ 0 , 1 ]$ and we simply have $\kappa _ { n , 3 } = 1 / n$ . For $d > 3$ there is no explicit formula for $\kappa _ { n , d }$ , but it can easily be computed numerically by solving $I _ { \kappa _ { n , d } } ( \delta , \delta ) = 1 / n$ . For fixed $t \le 1 / 2 , I _ { t } ( \delta , \delta )$ decreases as δ increases. Therefore, $\kappa _ { n , d }$ is an increasing function of d for any fixed $n \geq 2$ (it is also a decreasing function of n for any fixed $d )$ . When n tends to infinity with d we have the following properties.

Proposition 4.1. For fixed $d \geq 3$ , the 1/n quantile $\kappa _ { n , d }$ of $\beta _ { \delta , \delta }$ with $\delta = ( d - 1 ) / 2$ tends to zero as n tends to infinity. When $d \to \infty$ and $n = n ( d )$ grows with $d ,$ we have the following three cases:

(i) $i f n ^ { 1 / d } \to \infty$ , then lim $d {  } \infty \kappa _ { n , d } = 0 ;$

(ii) $i f n ^ { 1 / d } \to \lambda$ with $\lambda > 1$ , then

$$
\operatorname * { l i m } _ { d \to \infty } \kappa _ { n , d } = \frac 1 2 \left[ 1 - \sqrt { 1 - 1 / \lambda ^ { 2 } } \right] ;\tag{37}
$$

(iii) $i f \left( \log n \right) / d \to 0$ , then lim $\begin{array} { r } { { \bf \nabla } \cdot d  \infty \kappa _ { n , d } = \frac { 1 } { 2 } } \end{array}$

Proof. Since $x ^ { \delta - 1 } ( 1 - t ) ^ { \delta } \leq x ^ { \delta - 1 } ( 1 - t ) ^ { \delta - 1 } \leq x ^ { \delta - 1 } ( 1 - x ) ^ { \delta - 1 }$ for $\delta \geq 1$ and all $t \in [ 0 , 1 )$ and $x \in [ 0 , t ]$ , we have

$$
\frac { t ^ { \delta } ( 1 - t ) ^ { \delta } } { \delta B ( \delta , \delta ) } \leq I _ { t } ( \delta , \delta ) .\tag{38}
$$

The derivation of an exploitable upper bound is more delicate. Assume that $d \geq 3$ and let $\delta ^ { \prime } = \left\lfloor \delta \right\rfloor$ , the largest integer smaller than or equal to δ. As $n \geq 2$ , we are only interested in values of t less than $1 / 2$ . Since $\delta ^ { \prime } \leq \delta $ , we have $I _ { t } ( \delta , \delta ) \le I _ { t } ( \delta ^ { \prime } , \delta ^ { \prime } )$ , which can be calculated explicitly by successive integration by parts. Direct calculations yield the following special case of the well-known relation between the c.d.f.

of the beta and binomial distributions:

$$
{ \cal I } _ { t } ( \delta ^ { \prime } , \delta ^ { \prime } ) = \sum _ { k = 0 } ^ { \delta ^ { \prime } - 1 } { \binom { 2 \delta ^ { \prime } - 1 } { k } } t ^ { 2 \delta ^ { \prime } - 1 - k } ( 1 - t ) ^ { k } ,\tag{39}
$$

$$
\begin{array} { r l } { = } & { { } t ^ { \delta ^ { \prime } } ( 1 - t ) ^ { \delta ^ { \prime } - 1 } \displaystyle \sum _ { k = 0 } ^ { \delta ^ { \prime } - 1 } \binom { 2 \delta ^ { \prime } - 1 } { k } \left( \frac { t } { 1 - t } \right) ^ { \delta ^ { \prime } - 1 - k } . } \end{array}\tag{40}
$$

Using the property $t / ( 1 - t ) \leq 1$ for $t \leq 1 / 2$ , with $\begin{array} { r } { \sum _ { k = 0 } ^ { \delta ^ { \prime } - 1 } { \binom { 2 \delta ^ { \prime } - 1 } { k } } = 4 ^ { \delta ^ { \prime } - 1 } } \end{array}$ , we thus obtain

$$
I _ { t } ( \delta , \delta ) \leq I _ { t } ( \delta ^ { \prime } , \delta ^ { \prime } ) \leq t ^ { \delta ^ { \prime } } [ 4 ( 1 - t ) ] ^ { \delta ^ { \prime } - 1 } \leq \frac { 1 } { 2 } \left[ 4 t ( 1 - t ) \right] ^ { \delta ^ { \prime } - 1 } , t \leq 1 / 2 .\tag{41}
$$

Together with (38), this implies, for $n \geq 2$ and $d \geq 5 ,$

$$
\frac { 1 } { 2 } \left[ 1 - \sqrt { 1 - ( 2 / n ) ^ { 1 / ( \delta ^ { \prime } - 1 ) } } \right] \leq \kappa _ { n , d } \leq \frac { 1 } { 2 } \left[ 1 - \sqrt { ( 1 - 4 c _ { d } n ^ { - 1 / \delta } ) _ { + } } \right] ,\tag{42}
$$

where $( x ) _ { + } = \operatorname* { m a x } \{ x , 0 \}$ and $c _ { d } = [ \delta B ( \delta , \delta ) ] ^ { 1 / \delta } = 1 / 4 + ( \log d ) / ( 4 d ) + { \cal O } ( 1 / d )$ $d \to \infty$ (with, moreover, $( 4 c _ { d } ) ^ { \delta } < d + 2$ for all $d \geq 2$ , implying that the right-hand side of (42) is strictly smaller than $1 / 2$ for $n \geq d + 2 )$

When d is fixed and $n  \infty ,$ , or $n ^ { 1 / d } \to \infty { \mathrm { ~ a s ~ } } d \to \infty , n ^ { - 1 / d } \to 0$ implying that the right-hand side of (42) tends to zero. Therefore, $\kappa _ { n , d }  0$

In case $( i i ) , n ^ { - 1 / d } \to \lambda ^ { - 1 }$ . Taking limits on both sides of (42), and using   
$\delta / d  1 / 2 , \delta ^ { \prime } / d  1 / 2$ , and $c _ { d } \to 1 / 4$ , we get lim $_ { \cdot d  \infty } \kappa _ { n , d } = ( 1 / 2 ) [ 1 - \sqrt { 1 - 1 / \lambda ^ { 2 } } ]$ When log $n = o ( d )$ , (case $( i i i ) ) , n ^ { - 1 / d }  1$ and the left-hand side of (42) tends   
to $1 / 2$ □

The left panel of Figure 8 plots $\kappa _ { n , d }$ as a function of d for diferent $n \mathrm { : }$ the continuation of each curve (with n fixed) goes to the limiting value $1 / 2$ in view of Proposition 4.1-(iii). The bounds in (42) are asymptotically accurate for large $n ,$ particularly the upper bound, but lack precision for smaller n. For illustration, only the envelope for $n = 1 0 ^ { 3 }$ is shown.

The asymptotic behaviour of $\kappa _ { n , d }$ as $n ( d )$ grows with d governs that of the extreme-value approximation of the $( \mu , s )$ -distortion from Definition 4.1 and of the extreme-value approximation $\widehat { a ^ { * } }$ of the minimiser; see Corollary 4.1. Under suitable (non-restrictive) assumptions on $\mu ,$ we show below (Theorem 4.2) that the optimal radius $a ^ { * } ( d , n ( d ) , s )$ converges to a limit $a _ { \infty }$ that depends on the growth regime of $n ( d )$ but not on s. We also derive the limit of the associated expected $( \mu , s ) \cdot$ distortion.

Define

$$
\kappa _ { \infty } = \left\{ \begin{array} { l l } { 0 } & { \mathrm { i f ~ } n ^ { 1 / d } \to \infty , } \\ { \frac { 1 } { 2 } \left( 1 - \sqrt { 1 - 1 / \lambda ^ { 2 } } \right) } & { \mathrm { i f ~ } n ^ { 1 / d } \to \lambda \in ( 1 , \infty ) , } \\ { 1 / 2 } & { \mathrm { i f ~ } ( \log n ) / d \to 0 . } \end{array} \right.\tag{43}
$$

Theorem 4.2. For $s > 0$ and each $d ,$ let $\mu _ { d }$ be a probability measure on $\mathbb { R } ^ { d }$ , let $U _ { d } \overset { d } { \sim } \mu _ { d } , R _ { d } = \| U _ { d } \|$ , and assume that, for some $r > 0$ and $\eta > 0$

$$
R _ { d } \to r \ i n \ p r o b a b i l i t y \ a s \ d \to \infty \ a n d \ \operatorname* { s u p } _ { d } \mathsf { E } \{ R _ { d } ^ { s + \eta } \} < \infty .\tag{44}
$$

Let $n = n ( d ) \to \infty$ satisfy one of the three regimes in (43). $I f a _ { d }  a \geq 0$ , then

$$
D _ { \mu _ { d } , s } ( { \mathbb P } _ { a _ { d } } ^ { [ n ( d ) ] } )  [ ( r - a ) ^ { 2 } + 4 a r \kappa _ { \infty } ] ^ { s / 2 } , d  \infty .\tag{45}
$$

Let $a _ { d } ^ { * } = a ^ { * } ( d , n ( d ) , s )$ be any minimiser of $D _ { \mu _ { d } , s } ( \mathbb { P } _ { a } ^ { [ n ( d ) ] } )$ over $a \geq 0$ . Then

$$
a _ { d } ^ { * }  r ( 1 - 2 \kappa _ { \infty } ) = \{ \begin{array} { l l } { r } & { i f n ^ { 1 / d }  \infty , } \\ { r \sqrt { 1 - 1 / \lambda ^ { 2 } } } & { i f n ^ { 1 / d }  \lambda \in ( 1 , \infty ) , } \\ { 0 } & { i f ( \log n ) / d  0 , } \end{array}\tag{46}
$$

and

$$
\operatorname* { m i n } _ { a \geq 0 } D _ { \mu _ { d } , s } ( \mathbb { P } _ { a } ^ { [ n ( d ) ] } ) \to \left[ 4 r ^ { 2 } \kappa _ { \infty } ( 1 - \kappa _ { \infty } ) \right] ^ { s / 2 } = \left\{ \begin{array} { l l } { 0 } & { i f n ^ { 1 / d } \to \infty , } \\ { ( r / \lambda ) ^ { s } } & { i f n ^ { 1 / d } \to \lambda \in ( 1 , \infty ) , } \\ { r ^ { s } } & { i f \left( \log n \right) / d \to 0 . } \end{array} \right.\tag{47}
$$

Proof. In each of the three regimes considered, Proposition 4.1 gives $\kappa _ { n ( d ) , d }  \kappa _ { \infty }$ Consider the random variable $Z _ { d } = \zeta ( n ( d ) , d )$ in $( 7 )$ . We first show that, for every $p > 0 , Z _ { d }  \kappa _ { \infty }$ in $L ^ { p }$ (and thus in particular, $m _ { n ( d ) , d } = \mathsf E \{ Z _ { d } \} \to \kappa _ { \infty } )$

Fix $M > 1$ and define

$$
n _ { d } ^ { + } = \lceil M n ( d ) \rceil \quad \mathrm { a n d } \quad n _ { d } ^ { - } = \lfloor n ( d ) / M \rfloor .
$$

Multiplication of $n ( d )$ by a fixed positive constant does not change its asymptotic regime. Therefore, $\kappa _ { n _ { d } ^ { + } , d } \to \kappa _ { \infty }$ and $\kappa _ { n _ { d } ^ { - } , d }  \kappa _ { \infty }$ . For every $\epsilon > 0$ and all suficiently large d we have

$$
\mathsf { P r o b } \{ Z _ { d } \leq \kappa _ { \infty } - \epsilon \} \quad \leq \quad \mathsf { P r o b } \{ Z _ { d } \leq \kappa _ { n _ { d } ^ { + } , d } \} = 1 - \left( 1 - \frac { 1 } { n _ { d } ^ { + } } \right) ^ { n ( d ) } \leq \frac { 1 } { M } ,
$$

$$
\mathsf { P r o b } \{ Z _ { d } > \kappa _ { \infty } + \epsilon \} \quad \le \quad \mathsf { P r o b } \{ Z _ { d } > \kappa _ { n _ { d } ^ { - } , d } \} = \left( 1 - \frac { 1 } { n _ { d } ^ { - } } \right) ^ { n ( d ) } \le \exp ( - M ) .
$$

Letting first $d \to \infty$ and then $M \to \infty$ proves convergence in probability. Since $0 \le Z _ { d } \le 1$ , convergence holds in $L ^ { p }$ for every $p > 0$ , which also gives $m _ { n ( d ) , d } =$ ${ \mathsf E } \{ Z _ { d } \} \to \kappa _ { \infty }$

Now, consider (7) again. Conditioning on $U _ { d }$ gives the exact representation

$$
D _ { \mu _ { d } , s } ( { \mathbb P } _ { a } ^ { [ n ( d ) ] } ) = { \sf E } \left\{ \left[ ( R _ { d } - a ) ^ { 2 } + 4 a R _ { d } Z _ { d } \right] ^ { s / 2 } \right\} ,\tag{48}
$$

where $Z _ { d }$ may be taken independent of $R _ { d }$ . As $Z _ { d } \to \kappa _ { \infty }$ in probability, if $a _ { d } \to a$ then

$$
( R _ { d } - a _ { d } ) ^ { 2 } + 4 a _ { d } R _ { d } Z _ { d }  ( r - a ) ^ { 2 } + 4 a r \kappa _ { \infty }
$$

in probability. Moreover,

$$
0 \le \left[ ( R _ { d } - a _ { d } ) ^ { 2 } + 4 a _ { d } R _ { d } Z _ { d } \right] ^ { s / 2 } \le \left( R _ { d } + a _ { d } \right) ^ { s } .
$$

Assumption (44) therefore gives uniform integrability, and (45) follows. Using truncation of $R _ { d }$ and uniform continuity on compact sets, we obtain that the convergence in (45) is locally uniform in a. The minimisers $a _ { d } ^ { * }$ form a bounded sequence: indeed,

$$
D _ { \mu _ { d } , s } ( \mathbb { P } _ { a } ^ { [ n ( d ) ] } ) \geq \mathsf E \{ | R _ { d } - a | ^ { s } \} ,
$$

whereas the choice $a = 0$ gives $D _ { \mu _ { d } , s } ( { \mathbb P } _ { 0 } ^ { [ n ( d ) ] } ) = \mathsf E \{ R _ { d } ^ { s } \}$ , which is uniformly bounded. The standard argmin argument can therefore be applied to the limiting function

$$
L ( a ) = \left[ ( r - a ) ^ { 2 } + 4 a r \kappa _ { \infty } \right] ^ { s / 2 } .
$$

Since $s > 0$ , its unique minimiser is $a = r ( 1 - 2 \kappa _ { \infty } )$ , and its minimum is $[ 4 r ^ { 2 } \kappa _ { \infty } ( 1 -$ $\kappa _ { \infty } ) ] ^ { s / 2 }$ . Substitution of the three values in (43) gives (46) and (47). □

Under the assumptions of Theorem 4.2, the extreme-value approximation $\widehat { D } _ { \mu _ { d } , s } ( \mathbb { P } _ { a } ^ { [ n ( d ) ] } )$ of Definition 4.1 also satisfies $\widehat { D } _ { \mu _ { d } , s } ( \mathbb { P } _ { a } ^ { [ n ( d ) ] } ) \to \left\lceil ( r - a ) ^ { 2 } + 4 a r \kappa _ { \infty } \right\rceil ^ { s / 2 }$ as $d \to \infty$ and $\widehat { a ^ { * } } ( d , n ( d ) , s )$ satisfies (46): compared with the proof of Theorem 4.2, we now consider the random variable $T _ { n , d } = ( R _ { d } - a ) ^ { 2 } + 4 a R _ { d } \kappa _ { n , d } \zeta$ in (30), where again $\kappa _ { n , d }  \kappa _ { \infty }$

The assumptions (44) on $\mu$ are satisfied when $\mu$ possesses a norm-concentration property, i.e., when it satisfies

$$
{ \mathrm { f o r ~ a l l ~ } } \epsilon > 0 , \quad \mu \left\{ | \| U \| - r | \geq \epsilon \right\} \leq 2 \exp ( - c d \epsilon ^ { 2 } )
$$

when $U \sim \mu .$ , for some constant c not depending on $d .$ This is the case in particular for $\mu$ uniform in the unit ball $\mathcal { B } _ { d } ( 1 )$ as in Section 3.3.2 and for $\mu$ normal as in Section 3.3.3. The numerical evolution of $n _ { 1 } ( d , s )$ in Figure 5 suggests that, in these two situations and for suficiently small $\lambda > 1$ , quantisers with distribution $\mathbb { P } _ { \sqrt { 1 - 1 / \lambda ^ { 2 } } }$ may be asymptotically optimal among all random quantisers when $n ^ { 1 / d } \to \lambda .$ . This conclusion is empirical and is not established by Theorem 4.2, which optimises only over single-sphere distributions.

When $\mu$ is uniform on $S _ { d - 1 } ( 1 )$ , the same arguments as those leading to Definition 4.1 indicate that, for any $\mathbf { R } _ { n } \overset { d } { \sim } \mathbb { P } _ { a } ^ { [ n ] }$ , the extreme-value approximation of the $\gamma \cdot$ -quantile of the distance c.d.f. $F ( t ; \mathbf { R } _ { n } , \mu )$ is $\widehat { q } _ { \gamma } = \sqrt { ( 1 - a ) ^ { 2 } + 4 a \kappa _ { n , d } t _ { \gamma } } ,$ with $t _ { \gamma } = [ - \log ( 1 - \gamma ) ] ^ { 1 / \delta }$ the γ-quantile of the Weibull distribution with c.d.f. $F _ { d } ( t ) = 1 - \exp ( - t ^ { \delta } ) , \delta = ( d - 1 ) / 2$

The minimum of $\widehat { q } _ { \gamma }$ with respect to $a \geq 0$ is obtained for $\widehat { a _ { * } } ( d , n , \gamma ) = \left[ 1 - \right.$ $2 \kappa _ { n , d } t _ { \gamma } ] _ { + } .$ Since $t _ { \gamma }  1$ as $d \to \infty$ , for any fixed $\gamma , \widehat { a _ { * } } ( d , n ( d ) , \gamma )$ has the same limits as $\widehat { a ^ { * } } ( d , n ( d ) , \widehat { s } )$ , independently of $\gamma ,$ when $n = n ( d ) \to \infty$ according to one of the three regimes in (43).

Remark 4.2. Note that $\widehat { q } _ { \gamma } = \widehat { E } _ { \mu , 2 } ( n ; a )$ for any a when $\gamma = \gamma ( d , 2 ) = F _ { d } [ \Gamma ( 1 +$ $1 / \delta ) ]$ , showing that for large $n _ { : }$ , minimising the expected $( \mu , 2 )$ )-distortion is equivalent to minimising the γ-quantile of the mean distance $c . d . f .$ for

$$
\gamma = \gamma ( d , 2 ) = 1 - { \mathrm e } ^ { - { \mathrm e } ^ { - \gamma } } + \frac { \pi ^ { 2 } } { 6 d } { \mathrm e } ^ { - \gamma - { \mathrm e } ^ { - \gamma } } + { \cal O } ( d ^ { - 2 } ) , d \to \infty { , }
$$

where $\gamma$ is $\mathit { E u l e r } ^ { \prime } s$ constant and $1 - \mathrm { e } ^ { - \mathrm { e } ^ { - \gamma } } \simeq 0 . 4 2 9 6 2 4$ This suggests that $L _ { 2 } -$ quantisation is roughly equivalent to minimising the median of the distance $c . d . f . ,$ a phenomenon that can also be observed in other situations. ◁

## 4.3 Example: $\mu$ uniform on the unit sphere $S _ { d - 1 } ( 1 )$

When $U \stackrel { d } { \sim } \mu$ uniform on $S _ { d - 1 } ( 1 )$ , for any $\mathbf { R } _ { n } \overset { d } { \sim } \mathbb { P } _ { a } ^ { [ n ] }$ the approximations (34) and (35) become

$$
\begin{array} { l } { { \widehat { E } _ { \mu , 2 } ( n ; a ) = \left[ ( 1 - a ) ^ { 2 } + 4 a \kappa _ { n , d } \Gamma ( 1 + 1 / \delta ) \right] ^ { 1 / 2 } , } } \\ { { \widehat { E } _ { \mu , 4 } ( n ; a ) = \left[ ( 1 - a ) ^ { 4 } + 1 6 a ^ { 2 } \kappa _ { n , d } ^ { 2 } \Gamma ( 1 + 2 / \delta ) + 8 a ( 1 - a ) ^ { 2 } \kappa _ { n , d } \Gamma ( 1 + 1 / \delta ) \right] ^ { 1 / 4 } . } } \end{array}
$$

For $s = 2$ , Corollary 4.1 gives $\begin{array} { r } { \widehat { a ^ { * } } = \widehat { a ^ { * } } ( d , n , 2 ) = 1 \mathrm { - } 2 \kappa _ { n , d } \Gamma ( 1 \mathrm { + } 1 / \delta ) } \end{array}$ and $\widehat { E } _ { \mu , 2 } ^ { 2 } ( n ; \widehat { a ^ { * } } ) =$ $1 - ( \widehat { a ^ { * } } ) ^ { 2 }$ . The right panel of Figure 8 plots $\widehat { a ^ { * } } ( d , n , 2 ) - a ^ { * } ( d , n , 2 )$ as a function of d for diferent values of n: $\widehat { a ^ { * } } - a ^ { * } > 0$ and the accuracy of the approximation of $a ^ { * }$ by $\widehat { a ^ { * } }$ tends to decrease with d and increase with n.

The left and right panels of Figure 9 show that the two approximations $\widehat { D } _ { \mu , 2 } ( \mathbb { P } _ { a } ^ { [ n ] } )$ and $\widehat { D } _ { \mu , 4 } ( \mathbb { P } _ { a } ^ { [ n ] } )$ are quite accurate when n is large enough for extreme value theory to be applicable. For large $d ,$ convergence to the extreme-value distribution is slow and $\widehat { D } _ { \mu , s } ^ { 1 / { s } } ( \mathbb { P } _ { a } ^ { [ n ] } )$ slightly overestimates the true value $D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { a } ^ { [ n ] } )$

![](images/cd6282d5c5d7dcee2f9f6be36e8a59355acbf4f69fd25441852d4159287401fd.jpg)

![](images/4de7ded1bf7e54ff4ab654c47b983b62697f4a17b0c1f5737f30ab2a6f443ec5.jpg)  
Figure 8: Values of $\kappa _ { n , d } ~ ( \mathrm { l e f t } )$ and of $\widehat { a ^ { * } } ( d , n , 2 ) - a ^ { * } ( d , n , 2 )$ (right) as functions of d for diferent values of n: $n = 1 0 ^ { 2 } ~ ( \diamond )$ , n = 10<sup>3</sup> (▼), n = 10<sup>4</sup> (⋆) and $n = 1 0 ^ { 5 } ~ ( = )$

The left panel of Figure 10 provides another illustration of the approximation accuracy. It presents normalised values $n ^ { 1 / d } \widehat { D } _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { a } ^ { [ n ] } )$ and $n ^ { 1 / d } D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { a } ^ { [ n ] } )$ as functions of $n = 2 ^ { m } , m = 6 , 7 , \ldots , 2 0$ , for $s = 4$ and $d = 2 0$ , both for fixed $a = 0 . 7 5$ and for the optimal value $a ^ { * } ( d , n , s )$ (the plots with $\widehat { a ^ { * } } ( d , n , s )$ substituted for $a ^ { * } ( d , n , s )$ are visually indistinguishable). In the right panel, 100 numerically computed<sup>1</sup> values of $n ^ { 1 / d } E _ { \mu , s } ( { \bf R } _ { n } )$ are shown for each $n _ { \mathrm { : } }$ using independent random quantisers $\mathbf { R } _ { n } \overset { d } { \sim } \mathbb { P } _ { a ^ { * } } ^ { [ n ] }$ . The simulations agree closely with the exact mean $n ^ { 1 / d } D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { a ^ { * } } ^ { [ n ] } )$ and show very little variability across quantisers, in agreement with Theorem 4.1.

![](images/35b803247a801940bd86d01c774c0f815a468507d3a7802b4cd8cf748e17b413.jpg)

![](images/e072d50dd39c6381f4f8259fce5a6116d54c43f08e81d872a7206fe0a5deff0e.jpg)  
Figure 9: Relative error $1 - \widehat { D } _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { a } ^ { [ n ] } ) / D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { a } ^ { [ n ] } )$ of the approximation of $D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { a } ^ { [ n ] } )$ (left for $s = 2$ and right for $s = 4 )$ based on extreme-value theory, as a function of d for diferent values of n: $n = 1 0 ^ { 2 } ~ ( \diamond )$ $n = 1 0 ^ { 3 } \ ( \ " )$ 2 $n = 1 0 ^ { 4 } ~ ( \star )$ and $n = 1 0 ^ { 5 } ~ ( \equiv )$ ; a is the optimal value $\boldsymbol { a } ^ { * } ( d , n , s )$

## 5 Conclusions

![](images/85a52491e9a9d5c41992257f640b9cad9ae119873bf8bf357c8fe22c1c371a7d.jpg)

![](images/b1325faff43870d0e631f2c6574f797cbde769b8ba31f7bef384af75e659c6d5.jpg)  
Figure 10: $n ^ { 1 / d } \widehat { D } _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { a } ^ { [ n ] } )$ (♦) and $n ^ { 1 / d } D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { a } ^ { [ n ] } )$ (•) as functions of n for $s = 4$ and $d = 2 0$ . Left: $a = 0 . 7 5$ for top curves and $a = a ^ { * } ( d , n , s )$ for bottom curves. Right: $a =$ $a ^ { * } ( d , n , s )$ , the values of $n ^ { 1 / d } E _ { \mu , s } ( { \bf R } _ { n } )$ computed numerically for 100 random quantisers $\mathbf { R } _ { n } \overset { d } { \sim } \mathbb { P } _ { a ^ { * } } ^ { [ n ] }$ are shown as magenta dots.

We have demonstrated that, for a spherically symmetric measure $\mu$ in $\mathbb { R } ^ { d }$ , unless the sample size n is extremely large, a random quantiser uniformly distributed on a sphere of suitable radius can significantly outperform a random quantiser whose distribution follows the asymptotic limit predicted by Zador’s theorem. The optimal radius can be determined numerically by minimising a triple integral, which can be evaluated with arbitrary precision. Additionally, an approximation of the optimal radius is available, derived from extreme-value theory. Since the variability of the distortion for such random quantisers rapidly diminishes as n increases, this approach is highly practical for constructing high-performance quantisers.

While the restriction to spherically symmetric distributions may appear limiting, it provides a robust foundation for broader applications. For instance, quantising the uniform measure on the d-dimensional hypercube $[ - 1 , 1 ] ^ { d }$ presents a compelling and challenging problem, particularly in space-filling design for computer experiments (see, e.g., [13]). While greedy quantisation ofers an attractive approach for the incremental construction of designs in low dimensions [9, 10], it becomes computationally infeasible for large d. In such cases, recent results from [11] allow the uniform measure to be approximated by a spherically symmetric distribution. Similarly, one can consider random quantisers distributed according to a product measure, which can itself be approximated by a spherically symmetric distribution. Preliminary findings then suggest that quantisers distributed on the vertices of a smaller hypercube exhibit promising performance, paving the way for further advances in this direction.

## References

[1] N. Alon and J.H. Spencer. The Probabilistic Method. Wiley, 2000. Second edition.

[2] S. Boucheron, G. Lugosi, and O. Bousquet. Concentration inequalities. In O. Bousquet, U. von Luxburg, and G. R¨atsch, editors, Advanced Lectures on Machine Learning, pages 208–240. Springer, 2004.

[3] K.-T. Fang, S. Kotz, and K.W. Ng. Symmetric Multivariate and Related Distributions. Chapman and Hall/CRC, 1990.

[4] V.V. Fedorov. Theory of Optimal Experiments. Academic Press, New York, 1972.

[5] W. Feller. An Introduction to Probability Theory and Its Applications, vol. 2. Wiley, New York, 1971. Second edition.

[6] S. Graf and H. Luschgy. Foundations of Quantization for Probability Distributions. Springer, Berlin, 2000.

[7] J. Kiefer and J. Wolfowitz. The equivalence of two extremum problems. Canadian Journal of Mathematics, 12:363–366, 1960.

[8] S. Kotz and S. Nadarajah. Extreme Value Distributions: Theory and Applications. Imperial College Press, 2000.

[9] H. Luschgy and G. Pag\`es. Greedy vector quantization. Journal of Approximation Theory, 198:111–131, 2015.

[10] A. Nogales G´omez, L. Pronzato, and M.-J. Rendas. Incremental space-filling design based on coverings and spacings: improving upon low discrepancy sequences. Journal of Statistical Theory and Practice, 15(4):77, 2021.

[11] J. Noonan and A. Zhigljavsky. High-Dimensional Optimization: Set Exploration in the Non-Asymptotic Regime. Springer, 2024.

[12] G. Pag\`es. A space quantization method for numerical integration. Journal of Computational and Applied Mathematics, 89(1):1–38, 1997.

[13] L. Pronzato and W. M¨uller. Design of computer experiments: space filling and beyond. Statistics and Computing, 22:681–701, 2012.

[14] L. Pronzato and A. P´azman. Design of Experiments in Nonlinear Models. Asymptotic Normality, Optimality Criteria and Small-Sample Properties. Springer, LNS 212, New York, 2013.

[15] L. Pronzato and A.A. Zhigljavsky. Quasi-uniform designs with asymptotically optimal and near-optimal uniformity constant. Journal of Approximation Theory, 294(105931), 2023.

[16] H.P. Wynn. The sequential generation of D-optimum experimental designs. Annals of Math. Stat., 41:1655–1664, 1970.

[17] P.L. Zador. Asymptotic quantization error of continuous signals and the quantization dimension. IEEE Trans. Inform. Theory, 28:139–149, 1982.

# Supplementary Material for the paper Optimal random quantisers for spherically symmetric distributions

Luc Pronzato<sup>2</sup> and Anatoly Zhigljavsky<sup>3</sup>

## Abstract

This document contains additional material for the parent paper Optimal random quantisers for spherically symmetric distributions. We consider the three cases (i) µ is uniform on the sphere $S _ { d - 1 } ( 1 ) , \ ( i i ) \ \mu$ is uniform in the ball $\mathcal { B } _ { d } ( 1 )$ , and (iii) $\mu$ is spherically normal, for which we provide numerical examples that complement those in the parent paper and illustrate some of the results given there. In case (i) we consider random quantisers whose distribution is a mixture of the uniform distribution on a sphere $S _ { d - 1 } ( a )$ and the delta measure at $\mathbf { 0 } _ { d } .$ . We also compare the quantisation performance of fullfactorial $2 ^ { d }$ designs with that of random quantisers of the same size that are uniformly distributed on a sphere. In case (ii), we compare the performance of random quantisers uniformly distributed, respectively, on a sphere and in a ball, both with optimised radii. In case $( i i i )$ , we compare the performance of random quantisers uniformly distributed on spheres whose radii are derived from asymptotic considerations when $n = n ( d )$ grows exponentially with d.

The notation follows that of the parent paper. Equation numbers of the form $\left( \mathrm { x x } ^ { * } \right)$ refer to equation (xx) in the parent paper. Similarly, references such as $\mathrm { x x ^ { * } }$ are used for figures, definitions, theorems, and other numbered elements from the parent paper.

## 1 $\mu$ is uniform on the sphere $S _ { d - 1 } ( 1 )$

Here $\mu$ is the spherical measure (uniform on $S _ { d - 1 } ( 1 ) )$ , and the distribution of $\| U \|$ for $U \overset { d } { \sim } \mu$ is the delta measure at $r = 1$ ; see Section 3.3.1\*.

## 1.1 $\mathbb { P } = \mathbb { P } _ { a }$ uniform on $S _ { d - 1 } ( a )$

We first consider random quantisers with distribution $\mathbb { P } = \mathbb { P } _ { a }$ uniform on ${ \cal S } _ { d - 1 } ( a )$ and minimise the expected $( \mu , s )$ -distortion $D _ { \mu , s } ( \mathbb { P } _ { a } ^ { [ n ] } )$ , given by $( 2 6 ^ { * } )$ , with respect to $a \in [ 0 , 1 ]$ . The value $a ^ { * } = a ^ { * } ( d , n , s )$ that minimises $D _ { \mu , s } ( \mathbb { P } _ { a } ^ { [ n ] } )$ with respect to a can be obtained numerically with arbitrary precision for any given $d , n$ and s. The left column of Figure 1 presents $a ^ { * } ( d , n , s )$ as a function of d for diferent n and $s = 1 ~ ( \mathrm { t o p } )$ and $s = 1 0 \ ( \mathrm { b o t t o m } )$ . For fixed s and $d , a ^ { * }$ increases with $n ,$ and $a ^ { * }$ tends to 1 when n tends to infinity. For any fixed d and $n , a ^ { * } ( d , n , s )$ is a decreasing function of s, with $a ^ { * } ( d , n , \infty ) = 0$ if n is small enough.

The central column of Figure 1 shows that, as expected, $D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { a ^ { * } } ^ { [ n ] } )$ decreases with n and increases with d and s. The right column presents the eficiency of the naive quantiser with n points i.i.d. on $S _ { d - 1 } ( 1 )$ relative to optimised random quantisers with n points on $S _ { d - 1 } ( a ^ { * } )$ . It shows that the asymptotic optimum $a = 1$ can be a poor choice for large d: indeed, the benefit of using an appropriate a is significant when d is large or n is small.

![](images/a552cf5a9e9f5dd5466aee3eb160acb673db027215d792c8cb094ca52d8a4cf0.jpg)  
Figure 1: Sphere. Value $a ^ { * }$ minimising $D _ { \mu , s } ( \mathbb { P } _ { a } ^ { [ n ] } )$ (left column), $D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { a ^ { * } } ^ { [ n ] } )$ (central column) and ratio $D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { a ^ { * } } ^ { [ n ] } ) / D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { 1 } ^ { [ n ] } )$ (right column) as functions of d for $s = 1$ (top row) and $s = 1 0$ (bottom row) and diferent values of n: $n = 1 0 { \mathit { \Omega } } ( \bullet ) , n = 1 0 ^ { 2 } { \mathit { \Omega } } ( \ast ) , n = 1 0 ^ { 3 }$ $( \boldsymbol { \Psi } ) , n = 1 0 ^ { 4 } ~ ( \star )$ , and $n = 1 0 ^ { 5 } ~ ( = )$

## 1.2 Mixture of $\mathbb { P } = \mathbb { P } _ { a }$ uniform on $S _ { d - 1 } ( a )$ and $\delta _ { \mathbf { 0 } _ { d } }$

The next figure (Figure 2, where $s = 1 0 , d = 1 0 )$ shows that adding a delta measure at $\mathbf { 0 } _ { d }$ to $\mathbb { P } _ { 1 }$ may also yield a decrease of the expected distortion for small n. Denote $\mathbb { Q } _ { \alpha , a } = \alpha \mathbb { P } _ { a } + ( 1 - \alpha ) \delta _ { \mathbf { 0 } _ { d } }$ . Similarly to $( 2 6 ^ { * } )$ , we obtain

$$
\begin{array} { r c l } { { { \cal D } _ { \mu , s } ( \mathbb Q _ { \alpha , a } ^ { [ n ] } ) } } & { { = } } & { { ( 1 - a ) ^ { s } + s \displaystyle \int _ { 1 - a } ^ { 1 } t ^ { s - 1 } \left[ 1 - \alpha I _ { v } ( \delta , \delta ) \right] ^ { n } \mathrm { d } t } } \\ { { } } & { { } } & { { \displaystyle + s \alpha ^ { n } \displaystyle \int _ { 1 } ^ { 1 + a } t ^ { s - 1 } \left[ 1 - I _ { v } ( \delta , \delta ) \right] ^ { n } \mathrm { d } t , } } \end{array}
$$

where $\upsilon = [ t ^ { 2 } - ( 1 - a ) ^ { 2 } ] / ( 4 a ) , \delta = ( d - 1 ) / 2$ and $I _ { t } ( \cdot , \cdot )$ is the regularised incomplete beta function with the convention (5\*). The left panel presents the optimal $\alpha ^ { * }$ minimising $D _ { \mu , s } ( \mathbb { Q } _ { \alpha , 1 } ^ { [ n ] } )$ as a function of n: unsurprisingly, $\alpha ^ { * }$ quickly approaches 1 as n increases. One may observe that, for all $n , ( 1 - \alpha ^ { * } ) > 1 / n$ . Next, for a given n $( n = 2 0 )$ , we use the corresponding optimal $\alpha ^ { * }$ minimising $D _ { \mu , s } ( \mathbb { Q } _ { \alpha , 1 } ^ { [ n ] } ) \ ( \alpha ^ { * } \simeq 0 . 8 4 6 )$ and plot $D _ { \mu , s } ^ { 1 / s } ( \mathbb { Q } _ { \alpha ^ { * } , a } ^ { [ n ] } )$ and $D _ { \mu , s } ^ { 1 / s } ( \mathbb { Q } _ { 1 , a } ^ { [ n ] } ) = D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { a } ^ { [ n ] } )$ as functions of a (central panel): the efect of introducing a delta measure at $\mathbf { 0 } _ { d }$ is significant for a close to one, but choosing a suitable radius a has a larger impact. Finally, we consider the eficiencies of random quantisers with $\mathbb { Q } _ { \alpha ^ { * } , 1 }$ with respect to quantisers distributed with $\mathbb { P } _ { 1 }$ and $\mathbb { P } _ { a } .$ ∗ (with $\alpha ^ { * }$ and $a ^ { * }$ depending on $n )$ as functions of $n$ (right panel): when a is set to 1, the choice of a suitable α has some positive efect for small $n ,$ but the impact of choosing a suitable $a$ is stronger for all $n .$ . In Section 3.3.1\*, we show numerically that for n larger than some $n _ { * } ( d , s ) , \mathbb { P } _ { a ^ { * } }$ uniform on a single sphere is optimal among all spherically symmetric distributions (see Figure 1\*).

## 1.3 Comparison between random quantisers uniform on a sphere and optimised full factorial $2 ^ { d }$ designs

Let $n = 2 ^ { d }$ and ${ \bf X } _ { n } ( b )$ be the full factorial $2 ^ { d }$ design with points $\mathbf { x } _ { i }$ having coordinates ±b, $b \in ( 0 , 1 ] $ : the $\mathbf { x } _ { i }$ are vertices of the cube $[ - b , b ] ^ { d }$ ; they also belong to the sphere $\cal { S } _ { d - 1 } ( a )$ with $a \ = \ { \sqrt { d } } b$ . For $\mathcal { X } =  { S _ { d - 1 } } ( 1 )$ , the Voronoi regions

![](images/4449955aaca19db8e7e4c837369a5a3c937e67f3a8d546d36c94ffa7d1f50a10.jpg)

![](images/8cc6d41e7ba051d223fb0bdd59bda316b2f66f77d7549f14bca43b6f6417857e.jpg)

![](images/5e58cb15a06bd9a832e395b919f6cbf92b6b147633082870a8cd9f86cca1a376.jpg)  
Figure 2: Left: value $\alpha ^ { * }$ minimising $D _ { \mu , s } ( \mathbb { Q } _ { \alpha , 1 } ^ { [ n ] } )$ as a function of n. Centre: $D _ { \mu , s } ^ { 1 / s } ( \mathbb { Q } _ { \alpha ^ { * } , a } ^ { [ n ] } )$ $\mathrm { ~ ( ~ - ~ \varsigma ~ - ~ ) ~ }$ and $D _ { \mu , s } ^ { 1 / s } ( \mathbb { Q } _ { 1 , a } ^ { [ n ] } )$ (—) as functions of a for $n \ = \ 2 0$ and $\alpha ^ { * } ~ = ~ 0 . 8 4 6$ . Right: $D _ { \mu , s } ^ { 1 / s } ( \mathbb { Q } _ { \alpha ^ { * } , 1 } ^ { [ n ] } ) / D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { 1 } ^ { [ n ] } )$ (•) and $D _ { \mu , s } ^ { 1 / s } ( \mathbb { Q } _ { \alpha ^ { * } , 1 } ^ { [ n ] } ) / D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { a ^ { * } } ^ { [ n ] } )$ (♦) as functions of $n ; s = 1 0$ $d = 1 0$

$\begin{array} { r } { \mathcal { V } ( \mathbf { x } _ { i } ) = \{ \mathbf { x } \in \mathcal { X } : \| \mathbf { x } - \mathbf { x } _ { i } \| = \operatorname* { m i n } _ { \mathbf { x } _ { j } \in \mathbf { X } _ { n } ( b ) } \| \mathbf { x } - \mathbf { x } _ { j } \| \} } \end{array}$ are all identical up to a permutation of coordinates and do not depend on the value of b.

Consider first the case $s = 2$ . For $\mu$ uniform on $\begin{array} { r } { \mathcal { X } = S _ { d - 1 } ( 1 ) , D _ { { \mu } , 2 } [ { \mathbf { X } } _ { n } ( b ) ] = } \end{array}$ $\mathsf { E } \{ \| V -  { \mathbf { b } } \| ^ { 2 } \}$ where $\mathbf { b } = ( b , \dots , b ) \in \mathbb { R } ^ { d }$ and $V = \left( V _ { 1 } , \ldots , V _ { d } \right)$ is uniform in $\mathcal { V } ( { \bf b } ) =$ $S _ { d - 1 } ( 1 ) \cap \mathbb { R } _ { + } ^ { d }$ . Therefore

$$
D _ { \mu , 2 } [ { \mathbf { X } } _ { n } ( b ) ] = { \mathsf { E } } \{ \| V \| ^ { 2 } - 2 b \sum _ { i = 1 } ^ { d } V _ { i } + d b ^ { 2 } \} = 1 - 2 b d { \mathsf { E } } \{ V _ { 1 } \} + d b ^ { 2 } .
$$

Minimisation of $D _ { \mu , 2 } [ \mathbf { X } _ { n } ( b ) ]$ with respect to b yields $b _ { 2 } ^ { * } = \mathsf E \{ V _ { 1 } \}$ and the optima $( \mu , 2 )$ -distortion of a $2 ^ { d }$ factorial design is $D _ { \mu , 2 } [ \mathbf { X } _ { n } ( b _ { 2 } ^ { * } ) ] = 1 - \dot { d } ( b _ { 2 } ^ { * } ) ^ { 2 }$ . For ${ \bar { U } } ^ { ( d ) } =$ $( U _ { 1 } , \dots , U _ { d } ) \stackrel { d } { \sim } \mu$ , the first component $U _ { 1 }$ has the density $\varphi _ { 1 } ( \cdot )$ of Lemma $2 . 1 ^ { * }$ on $[ - 1 , 1 ]$ , therefore, $V _ { 1 }$ has the density $2 \varphi _ { 1 } ( \cdot )$ on [0, 1], so that

$$
\begin{array} { r c l } { { b _ { 2 } ^ { * } = \mathsf E \{ V _ { 1 } \} } } & { { = } } & { { \displaystyle \frac { \Gamma ( d / 2 ) } { \sqrt \pi \Gamma ( ( d + 1 ) / 2 ) } , } } \\ { { { \cal D } _ { \mu , 2 } [ { \bf X } _ { n } ( b _ { 2 } ^ { * } ) ] } } & { { = } } & { { \displaystyle 1 - \frac { d } { \pi } \left( \frac { \Gamma ( d / 2 ) } { \Gamma ( ( d + 1 ) / 2 ) } \right) ^ { 2 } . } } \end{array}
$$

More generally, the analytic expression for the $( \mu , s )$ -distortion $D _ { \mu , s } [ \mathbf { X } _ { n } ( b ) ]$ can be derived for any even $s ,$ but the calculations become more complicated as s increases. For $s = 4$ we obtain

$$
\begin{array} { l l l } { { { \cal D } _ { \mu , 4 } [ { \bf X } _ { n } ( b ) ] } } & { { = } } & { { \displaystyle { \mathbb E } \{ \| V - { \bf b } \| ^ { 4 } \} } } \\ { { } } & { { = } } & { { \displaystyle 1 + 4 b ^ { 2 } { \mathbb E } \left\{ \left( \sum _ { i = 1 } ^ { d } V _ { i } \right) ^ { 2 } \right\} + d ^ { 2 } b ^ { 4 } + 2 d b ^ { 2 } - 4 b d { \mathbb E } \{ V _ { 1 } \} ( 1 + d b ^ { 2 } ) . } } \end{array}
$$

Using ${ \ E \{ ( \sum _ { i = 1 } ^ { d } V _ { i } ) ^ { 2 } \} } = 1 + d ( d - 1 ) { \mathsf E } \{ V _ { 1 } V _ { 2 } \}$ , where $\mathsf E \{ V _ { 1 } V _ { 2 } \} = 2 / ( \pi d )$ from $\varphi _ { 2 } ( \cdot , \cdot )$ of Lemma $2 . 1 ^ { * }$ , together with the expression above for $\mathsf { E } \{ V _ { 1 } \}$ , we can express $D _ { \mu , 4 } [ \mathbf { X } _ { n } ( b ) ]$ as a fourth-degree polynomial in b with coeficients depending on $d .$ The optimal value $b _ { 4 } ^ { * }$ can be obtained explicitly.

Denote by $a _ { s } ^ { * } = \sqrt { d } b _ { s } ^ { * }$ the radius of the sphere on which the optimised full factorial design ${ \bf X } _ { n } ( b _ { s } ^ { * } )$ lies. The values of $a _ { 2 } ^ { * }$ and $a _ { 4 } ^ { * }$ are very close for large $d ,$ since $a _ { j } ^ { * } = \sqrt { d } b _ { j } ^ { * } = \sqrt { 2 / \pi } + { \mathcal O } ( 1 / d )$ for $j = 2 , 4$

For $s \mathbf { \sigma } ^ { \prime } = \infty , \ E _ { \mu , \infty } [ \mathbf { X } _ { n } ( b ) ] \ = \ { \mathsf { C R } } [ \mathbf { X } _ { n } ( b ) ]$ , the covering radius of ${ \bf X } _ { n } ( b )$ , with $( \mathsf { C R } [ \mathbf { X } _ { n } ( b ) ] ) ^ { 2 } = 1 + d b ^ { 2 } - 2 b$ being minimised at $b _ { \infty } ^ { * } = 1 / d$ , which gives $\begin{array} { r l } { \mathsf { C R } [ \mathbf { X } _ { n } ( b _ { \infty } ^ { * } ) ] = } & { { } } \end{array}$ ${ \sqrt { 1 - 1 / d } } .$

Figure 3 compares the optimal full factorial designs ${ \bf X } _ { n } ( b _ { s } ^ { * } )$ for $s = 2$ and 4 with optimised random quantisers $\mathbb { P } _ { a ^ { * } } ^ { [ n ] }$ on $S _ { d - 1 } ( 1 )$ with the same sample size $n = 2 ^ { d }$ ， with $a ^ { * } = a ^ { * } ( d , n , s )$ ; see Section 1.1. The left panel shows $a _ { s } ^ { * }$ and $a ^ { * }$ as functions of $d .$ The right panel presents $D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { a ^ { * } } ^ { [ n ] } )$ for the random quantisers and $E _ { \mu , s } [ \mathbf { X } _ { n } ( b _ { s } ^ { * } ) ]$ for the full factorial designs, for $s = 2$ and 4. For both types of point sets, the behaviours for $s = 2$ and $s = 4$ are similar. As d tends to infinity, for the full factorial designs $a _ { s } ^ { * } = \sqrt { d } b _ { s } ^ { * }$ tends to the limit $\sqrt { 2 / \pi } \simeq 0$ .797885 from above. For random quantisers, $a ^ { * } ( s ) = a ^ { * } ( d , n , s )$ tends (from below) to the larger limiting value $\sqrt { 3 } / 2 \simeq 0 . 8 6 6 0$ ; see Theorem $4 . 2 ^ { * }$ . The expected $( \mu , s )$ -distortion of random quantisers shows little sensitivity to the choice of a in the neighbourhood of $a ^ { * } ( s )$ In particular, the plots of $D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { a _ { s } ^ { * } } ^ { [ n ] } )$ (not shown) and $D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { a ^ { * } ( s ) } ^ { [ n ] } )$ almost coincide for $d > 3 .$

For both values of $s , a _ { s } ^ { * } \approx a ^ { * } ( s )$ for $d = 7 .$ . In the numerical comparisons shown here, this is the dimension at which random quantisers start exhibiting a smaller quantisation error than full factorial designs. As a consequence of the probabilistic method, the strict inequality between the mean random-quantiser distortion and the factorial-design distortion implies that for every dimension $d \ge 7$ there exist non-random quantisers with $n = 2 ^ { d }$ that have smaller quantisation errors for $s = 2$ and 4 than optimised full factorial designs. The numerical optimality certificates in Section $3 . 3 . 1 ^ { * }$ further indicate that $\mathbb { P } _ { a ^ { * } }$ is optimal among i.i.d. random-quantiser distributions in the cases considered. The concentration result of Theorem $4 . 1 ^ { * }$ explains why typical realisations are close to their mean performance, but it is not needed for the probabilistic-method existence argument.

![](images/acff89e4396211dbc50a0c91f8b1960913b7f3640d8508329c0e412fa3b82645.jpg)

![](images/fd3eddfb6d3a9af832e7a16c57ac1b52b3a8547b08f09392b21bcb5d4ff73861.jpg)  
Figure 3: Left: Values $a ^ { * } ( s )$ minimising $D _ { \mu , s } ( \mathbb { P } _ { a } ^ { [ n ] } )$ for $n = 2 ^ { d }$ with $s = 2 \ ( \diamond )$ and $s = 4$ $( { \star } )$ , and values $a _ { s } ^ { * } = \sqrt { d } b _ { s } ^ { * }$ minimising $E _ { \mu , s } [ \mathbf { X } _ { n } ( b ) ]$ for $s = 2 \ ( \bullet )$ and $s = 4 \ ( \mathbf { \hat { u } } )$ , as functions of d. Right: $D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { a ^ { * } } ^ { [ n ] } )$ for $n = 2 ^ { d }$ with $a ^ { * } = a ^ { * } ( d , n , s )$ $s = 2 \ ( \diamond )$ and $s = 4$ $( { \star } )$ , and values $E _ { \mu , s } [ \mathbf { X } _ { n } ( b _ { s } ^ { * } ) ]$ for $s = 2 \ ( \bullet )$ and $s = 4 ~ ( \mathbf { u } )$ , as functions of d.

## 2 $\mu$ is uniform in the unit ball $\mathcal { B } _ { d } ( 1 )$

Here $\mu$ is uniform in $\mathcal { B } _ { d } ( 1 )$ , so that the density of $r = \| U \|$ for $U \stackrel { d } { \sim } \mu$ is $\psi ( \tau ) =$ $d \tau ^ { d - 1 } $ ; see Section $3 . 3 . 2 ^ { * }$

When the random quantiser has the distribution $\mathbb { P } = \mathbb { P } _ { a }$ uniform on $\cal { S } _ { d - 1 } ( a )$ ， $D _ { \mu , s } ( \mathbb { P } _ { a } ^ { [ n ] } )$ is given by $( 2 7 ^ { * } )$ . When $\mathbb { P } = \mathbb { P } _ { 0 , b }$ uniform in $\mathcal { B } _ { d } ( b )$ , the c.d.f. Φ(·) of $R = \| X \|$ has the density $\phi _ { 0 , b } ( \rho ) = d \rho ^ { d - 1 } / b ^ { d }$ for $\rho \in [ 0 , b ]$ . More general spherically symmetric distributions P could also be used. Here, the class $\mathbb { P } _ { 0 , b }$ is considered as a simple one-parameter benchmark; it does not contain the globally optimal radial distribution, which consists of a mixture of uniform distributions on concentric spheres (see Section 3.3.2\*). The expected $( \mu , s )$ -distortion $D _ { \mu , s } ( \mathbb { P } _ { 0 , b } ^ { [ n ] } )$ is given by (12\*) where

$$
H ( r , t ; \Phi , n ) = \left( 1 - \int _ { 0 } ^ { b } I _ { v } ( \delta , \delta ) \phi _ { 0 , b } ( \rho ) \mathrm { d } \rho \right) ^ { n } ,
$$

with $v = v ( t , \rho , r )$ given by (9\*).

Figure 4 presents the same information as Figure 1 for the ball configuration: we consider the distribution $\mathbb { P } _ { 0 , b ^ { * } }$ , which is optimal within the one-parameter class $\{ \mathbb { P } _ { 0 , b } : b \ge 0 \}$ , where $b ^ { * } = b ^ { * } ( d , n , s )$ minimises $D _ { \mu , s } ( \mathbb { P } _ { 0 , b } ^ { [ n ] } )$ , and the right column compares the performances of $\mathbb { P } _ { 0 , b ^ { * } }$ and $\mathbb { P } _ { 0 , 1 }$

![](images/82254ff6430a41c5b1d210316f70ba86fce44c0e3f9d78c15c093f6cf2d348a1.jpg)  
Figure 4: Ball. Value $b ^ { * }$ minimising $D _ { \mu , s } ( \mathbb { P } _ { 0 , b } ^ { [ n ] } )$ (left column), $D _ { \mu , s } ^ { 1 / s } \big ( \mathbb { P } _ { 0 , b ^ { * } } ^ { [ n ] } \big )$ (central column) and ratio $D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { 0 , b ^ { * } } ^ { [ n ] } ) / D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { 0 , 1 } ^ { [ n ] } )$ (right column) as functions of d for $s = 1$ (top row) and $s = 1 0$ (bottom row) and diferent values of n: $n = 1 0$ (•), $n = 1 0 ^ { 2 }$ (♦), $n = 1 0 ^ { 3 } ~ ( \tau )$ $n = 1 0 ^ { 4 } \ ( \star )$ , and $n = 1 0 ^ { 5 } ~ ( = )$ .

The quantities considered behave qualitatively as in Figure 1. Note, however, that $b ^ { * }$ is larger than $a ^ { * }$ plotted in Figure 1 (left columns; in particular, on the bottom row with $s = 1 0 , b ^ { * }$ is much larger than $a ^ { * } )$ , that the displayed values of $D _ { \mu , s } ( \mathbb { P } _ { 0 , b ^ { * } } ^ { [ n ] } )$ for the ball target are smaller than the corresponding displayed values of $D _ { \mu , s } ( \mathbb { P } _ { a ^ { * } } ^ { [ n ] } )$ ) for the sphere target (central columns), and that the eficiency of a “classical” (i.e., asymptotically optimal) random design compared to an optimised design is slightly larger for the ball than for the sphere (right columns). These distortion values correspond to diferent target measures and therefore should not be interpreted as a direct performance comparison between the two quantisers. For fixed $d ,$ the empirical measure of an optimal quantiser minimising $E _ { \mu , s } ( \mathbf { X } _ { n } )$ converges weakly to the uniform probability measure $\mathbb { P } _ { 0 , 1 }$ as $n \to \infty$ (see [6, Table 7.1]). Consistent with this result, we empirically observe that the optimal parameters $b ^ { * }$ and $a ^ { * }$ both converge to 1.

Figure 5 displays the optimal values $b ^ { * } ( d , n , s )$ and $a ^ { * } ( d , n , s )$ , respectively minimising $D _ { \mu , s } ( \mathbb { P } _ { 0 , b } ^ { [ n ] } )$ and $D _ { \mu , s } ( \mathbb { P } _ { a } ^ { [ n ] } )$ , plotted as functions of d for $s = 2$ and two distinct values of $n .$ The figure shows that $b ^ { * } > a ^ { * }$ , and the gap $b ^ { * } - a ^ { * }$ narrows as d increases, reflecting the observed convergence of both $a ^ { * }$ and $b ^ { * }$ to zero as $d \to \infty$ The behaviour remains qualitatively unchanged for other values of $s ,$ especially when the dimension d is large, in agreement with the results in Section $4 . 2 ^ { * }$

![](images/cee3a61c9d8a4c351783ced00236cdbab2e62512d634e26c5ef94599e1c4bc17.jpg)

![](images/a5c6083f82bf8bc0f8b8faefab6662c04f8102285252a969a8ff3e4c12e6f5a3.jpg)  
Figure 5: $a ^ { \ast } ( d , n , 2 ) ~ ( \circ ) , \widehat { a ^ { \ast } } ( d , n , 2 )$ (⋆) and $b ^ { * } ( d , n , 2 )$ (♦) as functions of d for $n = 1$ 000 (left) and n = 100 000 (right).

Given the observations in Figures 4 and $5 ,$ two key patterns emerge: $( i )$ for large dimensions $d ,$ the values of $a ^ { * } ( d , n , s )$ and $b ^ { * } ( d , n , s )$ become very close; (ii) the quantiser $\mathbb { P } _ { 0 , b ^ { * } } ^ { [ n ] }$ significantly outperforms $\mathbb { P } _ { 0 , 1 } ^ { [ n ] }$ when d is suficiently large. Moreover, we also observe that $\mathbb { P } _ { a ^ { * } } ^ { [ n ] }$ outperforms $\mathbb { P } _ { 0 , b } ^ { [ n ] }$ <sub>∗</sub> for reasonable design sizes n; see Figure 6.

![](images/22a4e85314439c0be9d28161117b22cae0411d865dcdf3331d4e9cfee97d9c6f.jpg)

![](images/e3de119092d638155e12a2cba75600f511f0ae67053c5593d420c0af2af25202.jpg)  
Figure 6: Eficiencies $D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { a } ^ { [ n ] } ) / D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { 0 , b ^ { * } } ^ { [ n ] } )$ as functions of d: $a = a ^ { \ast } ( d , n , 2 ) ~ ( \bullet ) , ~ a =$ $\widehat { a ^ { * } } ( d , n , 2 ) \ ( \star ) ; n = 1 0 0 0$ (left) and $n = 1 0 0 0 0 0$ (right).

Since $b ^ { * }$ converges to 1 as $n  \infty$ , with $\mathbb { P } _ { 0 , 1 } ^ { [ n ] }$ being asymptotically optimal, a natural question arises: for a fixed dimension $d ,$ at what sample size $n = n _ { 2 } ( d , s )$ does $\mathbb { P } _ { a ^ { * } } ^ { [ n ] }$ stop being preferable to $\mathbb { P } _ { 0 , b ^ { * } } ^ { [ n ] } ?$ Empirical analysis reveals that this transition occurs only for extremely large values of n. The left panel of Figure 7 presents $n _ { 2 } ( d , s )$ on a logarithmic scale (computed by bisection) as a function of $d = 3 , \ldots , 2 0$ for three values of s. Over this range, the results suggest a super-exponential increase of $n _ { 2 } ( d , s )$ with d.

In the non-asymptotic regime, the extreme-value approximation $\widehat { a ^ { * } } ( d , n , 2 )$ of $a ^ { * } ( d , n , 2 )$ is given by $( 3 6 ^ { * } )$ for $s \ = \ 2 ;$ for $s = 4 , \widehat { a ^ { * } } ( d , n , 4 )$ is a root of the third-degree polynomial corresponding to the derivative of $( 3 5 ^ { * } )$ . Figure 5 displays $\widehat { a ^ { * } } ( d , n , 2 )$ as a function of d for two values of n: it is practically indistinguishable from $a ^ { * } ( d , n , 2 )$ . The eficiencies $D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { \widehat { a ^ { * } } } ^ { [ n ] } ) / D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { 0 , b ^ { * } } ^ { [ n ] } )$ and $D _ { \mu , s } ^ { 1 / { \bar { s } } } ( \mathbb { P } _ { a ^ { * } } ^ { [ n ] } ) / D _ { \mu , s } ^ { \bar { 1 } / s } ( \mathbb { P } _ { 0 , b ^ { * } } ^ { [ n ] } )$ are virtually indistinguishable in Figure $6 ;$ both $\mathbb { P } _ { a ^ { * } } ^ { [ n ] }$ and $\mathbb { P } _ { a ^ { * } } ^ { [ n ] }$ outperform $\mathbb { P } _ { 0 , b } ^ { [ n ] } { } _ { }$ <sub>∗</sub> for the values of n and d considered. Moreover, $\mathbb { P } _ { 0 , b ^ { * } } ^ { [ n ] }$ <sub>∗</sub> significantly outperforms $\mathbb { P } _ { 0 , 1 } ^ { [ n ] }$ ; see

![](images/a34657033a355e579821955f6c07d97317f3cb0ad3688382d881fe524a6861c2.jpg)

![](images/0a03c87e31187ebb68a622f1908c411366d0cdf9a2d8c67e21a0a2dbb9dba75d.jpg)  
Figure 7: $n _ { 2 } ( d , s )$ as a function of d for $s = 2 \ ( \diamond )$ , s = 4 (⋆) and $s = 1 0 \ ( \bullet )$ . Left: $\mu$ is uniform in $\mathcal { B } _ { d } ( 1 )$ . Right: µ is spherically normal.

Figure 4.

## 3 $\mu$ is a spherically symmetric normal distribution

Here $\mu$ is the probability measure of the normal distribution $\mathcal { N } ( \mathbf { 0 } _ { d } , \mathbf { I } _ { d } / d )$ , with $\mathbf { I } _ { d }$ the d-dimensional identity matrix; see Section 3.3.3\*.

Let $\sigma ^ { * } = \sigma ^ { * } ( d , n , s )$ denote the value of σ that minimises the distortion $D _ { \mu , s } ( \mu _ { \sigma } ^ { [ n ] } )$ as previously $a ^ { * } = a ^ { * } ( d , n , s )$ is the value of a that minimises $D _ { \mu , s } ( \mathbb { P } _ { a } ^ { [ n ] } )$

The empirical distribution of an optimal quantiser $\mathbf { X } _ { n } ^ { * }$ minimising $E _ { \mu , s } ( \mathbf { X } _ { n } )$ converges weakly to $\mathcal { N } ( \mathbf { 0 } _ { d } , ( \sigma _ { \infty } ^ { * } ) ^ { 2 } \mathbf { I } _ { d } / d )$ as $n  \infty$ , where $\sigma _ { \infty } ^ { * } = \sqrt { 1 + s / d } ;$ see [6, Table 7.1]. The left panel of Figure 8 (for $d = 1 0$ and $n = 1 0 0 0 0 0 )$ illustrates that $\sigma ^ { * } = \sigma ^ { * } ( d , n , s )$ increases with $s ,$ as does $\sigma _ { \infty } ^ { * } .$ , and that $a ^ { * } = a ^ { * } ( d , n , s )$ also increases with s (see Figure 7\*-left for the case $n = 1 0 0 0 )$ ).

![](images/fe58dfccc854eedda16ff3f0414beb433911dc09a3aed133661e61b2f6c5b22c.jpg)

![](images/cc831f857b3da73b65d58da801a7cf40451e88e5d7ce4a7dd6c589927bd1751b.jpg)  
Figure 8: Left: optimal values $a ^ { \ast } ( d , n , s ) \ ( \bullet )$ and $\sigma ^ { * } ( d , n , s ) \left( \diamond \right)$ as functions of s for $d = 1 0$ and $n = 1 0 0 0 0 0$ . Right: ratios $D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { a ^ { * } } ^ { [ n ] } ) / D _ { \mu , s } ^ { 1 / s } ( \mu _ { \sigma _ { \infty } ^ { * } } ^ { [ n ] } ) \ ( - \ - \ - )$ and $D _ { \mu , s } ^ { 1 / s } ( \mu _ { \sigma ^ { * } } ^ { \left[ n \right] } ) / D _ { \mu , s } ^ { 1 / s } ( \mu _ { \sigma _ { \infty } ^ { * } } ^ { \left[ n \right] } )$ (—) as functions of d for $s = 2$ and $n = 1 0 0 0 0$

Figure 9 presents $D _ { \mu , s } ^ { 1 / s } ( \mu _ { \sigma } ^ { [ n ] } )$ and $D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { a = \sigma } ^ { [ n ] } )$ as functions of σ for $s = 2 , n =$ 1 000 and $d = 5$ , 10, and 20. For $d = 5$ (left panel), the asymptotically optimal scale is $\sigma _ { \infty } ^ { * } = \sqrt { 1 . 4 } \simeq 1 . 1 8 3 2$ . For $d \geq 1 0$ (centre and right panels), the plots of $D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { \sigma } ^ { [ n ] } )$ and $D _ { \mu , s } ^ { 1 / s } ( \mu _ { \sigma } ^ { [ n ] } )$ are close, especially near their minima. The behaviour is similar when using other values of s.

The right panel of Figure 8 presents the eficiencies of the asymptotically optimal distribution $\mu _ { \sigma _ { \infty } ^ { * } }$ relative to $\mathbb { P } _ { a }$ ∗ and $\mu _ { \sigma ^ { * } }$ , for $n = 1 0 0 0 0$ , as functions of $d .$ These eficiencies are always less than 1, meaning that the asymptotic regime is not yet reached for the values of d considered. $\mathrm { A t } d = 1 0$ , however, we can observe that, contrary to the central panel of Figure 9 where $n = 1 0 0 0$ , now $D _ { \mu , 2 } ( \mathbb { P } _ { a ^ { * } } ^ { [ n ] } ) > D _ { \mu , 2 } ( \mu _ { \sigma ^ { * } } ^ { [ n ] } )$ with $n = 1 0 0 0 0$ points we approach the asymptotic regime and $\mathbb { P } _ { a ^ { * } }$ is dominated by a suitably chosen normal distribution. The smallest $n = n _ { 2 } ( d , s )$ at which this occurs is plotted in the right panel of Figure 7 for $s = 2 , 4$ , 10 (the case $s = 2$ is also presented in Figure 5\*). These plots suggest that $D _ { \mu , s } ( \mathbb { P } _ { a ^ { * } } ^ { [ n ] } ) < D _ { \mu , s } ( \mu _ { \sigma ^ { * } } ^ { [ n ] } )$ for n $\lesssim n _ { 0 } \lambda _ { 0 } ^ { d }$ for some $n _ { 0 }$ and $\lambda _ { 0 }$ . We now investigate the behaviour of $D _ { \mu , s } ( \mathbb { P } _ { a } ^ { [ n ] } )$ for other choices of a in a similar asymptotic regime where $n = n ( d )$ grows exponentially fast with d. We take $n ( d ) = 2 ^ { d }$ . According to the numerical comparisons, $\mathbb { P } _ { a ^ { * } } ^ { [ n ( d ) ] }$ is then not optimal among all spherically symmetric distributions, but it still outperforms the optimised normal distribution $\mu _ { \sigma ^ { * } } ^ { [ n ( d ) ] }$

![](images/77d6e5af41bf5c4d5b7d5b0ad405b7e15fcf7e984bae82e13e5d0b6e0f9d0328.jpg)

![](images/71905ab9b5268d027071227f3f78517224d9e28f7f1bd1ab0d5630332660dd1d.jpg)

![](images/ceaf4129f9be3950fd108c57de3b6104d5e3b7f26ad1bd805f52a8d21566ae17.jpg)  
Figure 9: $D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { a = \sigma } ^ { [ n ] } ) \ ( - \ - \ - )$ and $D _ { \mu , s } ^ { 1 / s } ( \mu _ { \sigma } ^ { \left[ n \right] } ) ~ ( - )$ as functions of σ for diferent d with $n = 1 0 0 0$ and $s = 2$

Given that $M _ { \psi , 1 } = \mathsf E \{ \| U \| \}  1$ and var $\{ \| U \| \} \to 0$ as $d \to \infty$ when $U \sim \mu$ , we analyse the following cases for a. (i) Uniform distribution on $S _ { d - 1 } ( 1 )$ : we consider $a = \widetilde { a ^ { * } } ( d , n , s )$ which minimises $D _ { \widetilde { \mu } , s } ^ { 1 / s } ( \mathbb { P } _ { a } ^ { [ n ] } )$ for $\widetilde { \mu }$ uniform on $S _ { d - 1 } ( 1 )$ , as discussed in Section 1. (ii) Scaled uniform distribution: we also consider $a = M _ { \psi , 1 } \widetilde { a ^ { * } } ( d , n , s )$ ， obtained when $\widetilde { \mu }$ is uniform on $S _ { d - 1 } ( M _ { \psi , 1 } )$ . As we consider the asymptotic regime $n = 2 ^ { d }$ , we additionally examine $( i i i )$ the limiting value $\widehat { a _ { \lambda } ^ { * } } = \sqrt { 3 } / 2$ , see $( 4 6 ^ { * } )$ ; $( i v )$ the value $\widehat { a ^ { * } } ( d , s )$ , obtained by minimising the extreme-value approximation $\widehat { E } _ { \mu , s } ^ { s } ( n ; a )$ defined in $( 3 0 ^ { * } )$ after the quantile $\kappa _ { n , d }$ is replaced by its asymptotic value $( 1 / 2 ) ( 1 - { \sqrt { 3 } } / 2 )$ . (Formula $( 3 6 ^ { * } )$ applies only when $s = 2 . )$

The left panel of Figure 10 displays the four parameters $\sigma ^ { * } ( d , n , s ) , a ^ { * } ( d , n , s )$ 2 $\widetilde { a ^ { * } } ( d , n , s )$ , and $M _ { \psi , 1 } \widetilde { a ^ { * } } ( d , n , s )$ as functions of d for $s = 4$ , together with $\widehat { a _ { \lambda } ^ { * } } \ ( \mathrm { i n - }$ dicated by a horizontal line) and $\widehat { a ^ { * } } ( d , s )$ The right panel in the same figure shows the eficiency of the optimal normal distribution $\mu _ { \sigma }$ ∗ compared to $\mathbb { P } _ { a }$ uniform on $\cal { S } _ { d - 1 } ( a )$ , that is, $D _ { \mu , s } ^ { \hat { 1 } / s } ( \mathbb { P } _ { a } ^ { [ n ] } ) / D _ { \mu , s } ^ { 1 / s } ( \mu _ { \sigma ^ { * } } ^ { [ n ] } )$ ), for the five choices of a considered: these five quantiser distributions $\mathbb { P } _ { a }$ perform better than $\mu _ { \sigma ^ { * } }$ (for $d \geq 4$ when $a = M _ { \psi , 1 } \widetilde { a ^ { * } } ( d , n , s ) )$ . Note that the eficiencies for $\widehat { a _ { \lambda } ^ { * } }$ (purple triangles) and $a ^ { * } ( d , n , s )$ (blue dots) almost coincide for $d \geq 5$

Remark 3.1. From the results in Section $4 . 2 ^ { * } ,$ the four quantities $a ^ { * } ( d , n , s )$ $\widetilde { a ^ { * } } ( d , n , s )$ , $M _ { \psi , 1 } \widetilde { a ^ { * } } ( d , n , s )$ and $\widehat { a ^ { * } } ( d , s )$ tend to $\widehat { a _ { \lambda } ^ { * } } = \sqrt { 3 } / 2$ when $d \to \infty$ . Our numerical results suggest that, in the considered asymptotic regime, $\sigma ^ { * } ( d , n , s )$ also converges to the limiting value $\widehat { a _ { \lambda } ^ { * } } = \sqrt { 3 } / 2$ . However, a rigorous theoretical analysis is still required to confirm this observation. The same is true for the asymptotic behaviour of $b ^ { * } ( d , n , s )$ in the optimised quantiser $\mathbb { P } _ { 0 , b ^ { * } } ^ { [ n ] }$ when $\mu$ is uniform in $\mathcal { B } _ { d } ( 1 )$ . From $( 6 ^ { * } )$ in Proposition ${ \mathit { 2 . 1 } } ^ { * } ,$ we have to analyse the behaviour of $d ( \mathbf { u } , \mathbf { R } _ { n } ) = \operatorname* { m i n } _ { i = 1 , . . . , n } \left[ ( \left| \left| \mathbf { u } \right| \right| - R _ { i } ) ^ { 2 } + 4 \left| \left| \mathbf { u } \right| \right| R _ { i } \zeta _ { i } \right]$ when $n \to \infty$ , where the $R _ { i }$ are $i . i . d .$ , the $\zeta _ { i }$ are $i . i . d .$ with $\zeta _ { i } \overset { d } { = } \beta _ { \delta , \delta }$ , and the two families are mutually independent. A norm-concentration property for $R = \| X \|$ as $d \to \infty$ does not by itself imply an extreme-value limit $f o r d ( { \mathbf { u } } , { \mathbf { R } } _ { n } )$ . Even the cases where $\mathbb { P } = \mathbb { P } _ { 0 , b }$ is uniform on $\mathcal { B } _ { d } ( b )$ or where $\mathbb { P } = \mu _ { \sigma }$ is multivariate normal pose significant challenges. ◁

![](images/9149a716df3f092113220f554ddcf64189389a03c85389a6a2f5bff0f635041d.jpg)

![](images/5f40809de1fc4420558a05e96a795ff4213def98b0e01d3da4f5c6683e5585be.jpg)  
Figure 10: Left: $\sigma ^ { * } ( d , n , s )$ (♦), $\boldsymbol a ^ { * } ( d , n , s )$ (•), $\widetilde { a ^ { * } } ( d , n , s )$ (▼), $M _ { \psi , 1 } \widetilde { a ^ { * } } ( d , n , s ) \ ( \ " ) , \ \widehat { a _ { \lambda } ^ { * } } =$ ${ \sqrt { 3 } } / 2 \ ( \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } )$ and $\widehat { a ^ { * } } ( d , s ) \ ( \star )$ as functions of d. Right: eficiencies $D _ { \mu , s } ^ { 1 / s } ( \mathbb { P } _ { a } ^ { [ n ] } ) / D _ { \mu , s } ^ { 1 / s } ( \mu _ { \sigma ^ { * } } ^ { [ n ] } )$ as functions of d for the choices of a indicated on the left panel $( n = 2 ^ { d } , s = 4 )$