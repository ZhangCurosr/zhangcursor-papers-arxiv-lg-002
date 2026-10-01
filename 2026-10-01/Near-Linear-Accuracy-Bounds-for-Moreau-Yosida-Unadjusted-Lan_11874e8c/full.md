# Near-Linear Accuracy Bounds for Moreau–Yosida Unadjusted Langevin Sampling

Yuchen Xin<sup>\*</sup>, Zhihua Zhang<sup>†</sup>

## Abstract

We establish near-linear accuracy bounds for the classical Moreau–Yosida unadjusted Langevin algorithm (MYULA). The target is $\pi \propto e ^ { - f - g }$ , where $f \in C ^ { 2 } (  { \mathbb { R } } ^ { d } )$ is m-strongly convex with Lipschitz gradient and $g$ is convex and globally Lipschitz. Under an explicit parameter-dependent step-size condition, we bound the invariant-measure bias relative to the Moreau-smoothed target by $\widetilde O ( h )$ , with only logarithmic dependence on the inverse smoothing parameter in the error coeficient. Combining this estimate with the Moreau approximation bias and Wasserstein contraction gives $\tilde { O } ( \varepsilon ^ { - 1 } )$ iterations to make the Nth-iterate law $\mu _ { N }$ satisfy $\sqrt { m } W _ { 2 } ( \mu _ { N } , \pi ) \leq \varepsilon$ for fixed model parameters and initialization. We bound the stationary error directly, without assuming third derivatives or a Lipschitz Hessian. Each iteration uses one gradient evaluation and one exact proximal evaluation. The key idea in our analysis is to convert a second-order stationary residual into a Wasserstein bound using a Poisson-based estimate.

## Contents

1 Introduction   
1.1 Contribution 3   
1.2 Related Work 3   
2 Preliminaries 4   
2.1 Weak derivatives 5   
2.2 Moreau regularization 5   
2.3 Drift maps, pushforwards, and density evolution 5   
2.4 Poisson equations and transport 7   
2.5 Entropy and density-ratio moments . 9   
3 The Moreau–Yosida Unadjusted Langevin Sampling 9   
3.1 Problem formulation and assumptions 9   
3.2 Main results . 10   
4 Proof of the Main Results 11   
4.1 The analytic estimates . 11   
4.2 The stationary residual . 12   
4.3 A residual-to-transport estimate 16   
4.4 Proof of the fixed-smoothing theorem 18   
4.5 Proof of the end-to-end complexity theorem 20   
5 Stationary Flux Estimates 20   
5.1 The averaged drift flux . 21   
5.2 A finite heat expansion . 21   
5.3 Proof of the flux bounds 23   
6 Conclusion 24   
A Target Moments and Normalized Operator Estimates 24   
A.1 Target drift moments . . 24   
A.2 A Gibbs change of variables for the heat operator . 25   
A.3 Gaussian derivatives without an extra dimension factor 26   
A.4 Pushforward estimates by an integrated Jacobian . 28   
B Finite-Order Stationary Density Ratios 29   
B.1 Weighted energy for the density ratio . 29   
B.2 Gaussian integration by parts with the density-ratio weight 31   
B.3 Proof of the density-ratio bound 33   
C Poisson Regularity and Negative Sobolev Norms 38   
C.1 Existence and second-order energy 38   
C.2 Approximation by compactly supported tests 39   
C.3 Equivalent test classes for the negative Sobolev norm . 40   
D Weak Regularity and Integrability Before Dissipation 41   
D.1 Integration by parts and smooth approximation . 41   
D.2 Finiteness before the R´enyi calculation 42   
D.3 Removing the auxiliary mollification 50   
References 52

## 1 Introduction

We study a sampling problem from composite log-concave distributions

$$
\pi ( \mathrm { d } x ) \propto e ^ { - f ( x ) - g ( x ) } \mathrm { d } x , \qquad x \in \mathbb { R } ^ { d } ,
$$

where $f \in C ^ { 2 } (  { \mathbb { R } } ^ { d } )$ is m-strongly convex with L<sub>f</sub>-Lipschitz gradient, and $g$ is convex and globally G-Lipschitz. The nonsmooth term can represent penalties such as $\ell _ { 1 }$ regularization. The Moreau– Yosida unadjusted Langevin algorithm (MYULA) of Durmus et al. [3] replaces $g$ by its Moreau envelope $g _ { \lambda }$ and uses the update

$$
\begin{array} { r } { \widehat { X } _ { k + 1 } = \widehat { X } _ { k } - h \nabla ( f + g _ { \lambda } ) ( \widehat { X } _ { k } ) + \sqrt { 2 h } \xi _ { k + 1 } , \qquad \xi _ { k + 1 } \overset { \mathrm { i . i . d . } } { \sim } N ( 0 , I _ { d } ) . } \end{array}
$$

The identity $\nabla g _ { \lambda } = \lambda ^ { - 1 } ( I - \operatorname { p r o x } _ { \lambda g } )$ allows each iteration to use one evaluation of $\nabla f$ and one exact proximal evaluation of $^ { g , }$ without an acceptance step.

The central issue is how smoothing afects discretization accuracy. Reducing λ improves the approximation of $\pi$ by $\pi _ { \lambda } \propto e ^ { - f - g _ { \lambda } }$ , but increases the global smoothness bound $L _ { \lambda } = L _ { f } + \lambda ^ { - 1 }$ Thus, a small-step bias estimate for a fixed λ does not by itself give a sharp complexity bound for the original target: its dependence on $\lambda$ must also be controlled.

## 1.1 Contribution

In this work, we would address the above issue and present error bounds for MYULA. We denote

$$
\tau _ { f } : = \operatorname * { s u p } _ { x } \mathrm { t r } [ \nabla ^ { 2 } f ( x ) ] , \qquad R : = \frac { G + \sqrt { G ^ { 2 } + 4 \tau _ { f } } } { 2 } ,
$$

and let $\ell _ { h , \lambda } : = 1 + \log ( e + ( m h ) ^ { - 1 } + L _ { \lambda } / m )$ . Under the step-size condition (3.11), Theorem 3.4 proves that the MYULA invariant law satisfies

$$
\sqrt { m } W _ { 2 } ( \widehat { \pi } _ { \lambda , h } , \pi _ { \lambda } ) \leq C h R ^ { 2 } \ell _ { h , \lambda } ^ { 2 } .
$$

The error coeficient depends on $\lambda ^ { - 1 }$ only through a logarithm, although the suficient step-size restriction is stronger than $h L _ { \lambda } \leq c .$

For $0 < \varepsilon \le 1$ , choose $\lambda = \varepsilon / G ^ { 2 }$ and the fixed step size in Theorem 3.5. Combining the invariant-measure estimate with the Moreau bias bound of Xin and Zhang [8, Proposition 5.9] gives the suficient iteration complexity

$$
N = \widetilde { O } \biggl [ \frac { L _ { f } \tau _ { f } } { m ^ { 2 } } + \frac { L _ { f } R } { m ^ { 3 / 2 } } + \frac { 1 } { \varepsilon } \left( \frac { R ^ { 2 } } { m } + \frac { G ^ { 3 } R } { m ^ { 2 } } \right) \biggr ]
$$

for $\sqrt { m } W _ { 2 } ( \mu _ { N } , \pi ) \le \varepsilon$ , where $\mu _ { N } = \mathcal { L } ( \widehat { X } _ { N } )$ and $\widetilde O$ suppresses logarithmic factors. For fixed model parameters and initialization, this improves the $\widetilde { \cal O } ( \varepsilon ^ { - 4 / 3 } )$ guarantee of Xin and Zhang [9] to $\widetilde { O } ( \varepsilon ^ { - 1 } )$ . For fixed $L _ { f } / m$ and $G / \sqrt { m }$ , the bound also gives $\bar { O } ( d / \varepsilon )$ iterations.

Proof strategy. The core strategy of our analysis is a Poisson-based estimate that converts a second-order stationary residual into a Wasserstein bound (Section 4.3). By testing the residual against a Poisson solution, we place derivatives on the test function rather than on the residual fields. Weighted energy estimates control the resulting terms while retaining the Hessian as a weight, instead of replacing it by its global bound. In particular, for MYULA, exact drift–heat identities express the residual through vector and matrix fluxes (Section 4.2). Finite-order density-ratio estimates and short-time heat estimates control these fluxes (Section 5), allowing the curvature contribution to be bounded through its average under $\pi _ { \lambda }$ . Together, these steps yield a near-linear invariant-measure bias with only logarithmic dependence on $\lambda ^ { - 1 }$ in its coeficient, without additional higher-order smoothness assumptions (Section 4.4). Adding the smoothing and convergence errors gives the complexity bound in Section 4.5.

## 1.2 Related Work

MYULA and averaged curvature. Durmus et al. [3] introduced MYULA and established asymptotic and nonasymptotic convergence guarantees for nonsmooth log-concave targets. More recently, Xin and Zhang [8] analyzed the weak Moreau Hessian through its average along a reference heat path. For the structured penalties treated there, curvature localization yields $\widetilde { O } ( \varepsilon ^ { - 2 } )$ end-to-end complexity. The closest result is Xin and Zhang [9], who consider the same assumptions and MYULA transition as this paper. Their discrete Poisson-corrector analysis gives an invariant-measure bias of $O ( h ) + \widetilde O ( h ^ { 3 / 4 } )$ , with only logarithmic smoothing dependence in its coeficients, and consequently $\widetilde { O } ( \varepsilon ^ { - 4 / 3 } )$ complexity. Our stationary-density analysis replaces this bias bound by $\widetilde O ( h )$ under a stronger step-size restriction. The improvement concerns the fixed-parameter accuracy exponent; it does not assert uniformly better dependence on every model parameter.

First-order ULA bounds and the smoothing parameter. For a smooth, m-strongly convex potential with L-Lipschitz gradient, Pedrotti and Whalley [5, Theorem 1] prove

$$
\sqrt { m } W _ { 2 } ( \widehat { \pi } _ { h } , \pi ) \leq 6 h L \sqrt { d } , \qquad 0 < h L \leq 1 .
$$

Applying this estimate to $f + g _ { \lambda }$ , with the $C ^ { 1 , 1 }$ case obtained by mollification, and choosing $\lambda \times \varepsilon / G ^ { 2 }$ gives the suficient MYULA complexity

$$
\tilde { O } \Bigg [ \frac { \sqrt { d } } { m } \left( \frac { L _ { f } } { \varepsilon } + \frac { G ^ { 2 } } { \varepsilon ^ { 2 } } \right) \Bigg ] .
$$

Thus, first-order bias for a fixed smooth target does not directly imply near-linear accuracy complexity after Moreau smoothing. The gain is in the joint smoothing–discretization analysis, not in establishing first-order ULA bias at fixed smoothness.

Other proximal samplers. Diferent transitions can achieve higher accuracy without a fixed Moreau approximation. For example, Fan et al. [4, Proposition 5] obtain polylogarithmic accuracy dependence in $W _ { 2 }$ for strongly convex composite targets, under their initialization and subroutine assumptions. Their method uses a restricted Gaussian sampling step implemented by approximate rejection sampling, rather than a deterministic proximal update. These guarantees concern a diferent algorithm and computational model. Our result is an iteration bound for classical MYULA with one exact proximal evaluation per step.

## 2 Preliminaries

All measures are defined on $\mathbb { R } ^ { d }$ , and $\mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ denotes the probability measures with finite second moment. For $\mu , \nu \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ ,

$$
W _ { 2 } ^ { 2 } ( \mu , \nu ) : = \operatorname* { i n f } _ { \mathcal { L } ( X ) = \mu , \mathcal { L } ( Y ) = \nu } \mathbb { E } \left\| X - Y \right\| ^ { 2 } .\tag{2.1}
$$

The same symbol denotes a measure and its Lebesgue density when a density exists. We use the Euclidean norm and inner product for vectors, the Frobenius norm $\begin{array} { r } { \| M \| _ { F } ^ { 2 } = \sum _ { i , j } M _ { i j } ^ { 2 } } \end{array}$ for matrices, and $\begin{array} { r } { \langle M , N \rangle = \mathrm { t r } ( M ^ { \top } N ) = \sum _ { i , j } M _ { i j } N _ { i j } } \end{array}$ . For symmetric matrices, $M \preceq N$ means $v ^ { \top } M v \leq v ^ { \top } N v$ for every v. For a scalar, vector, or matrix field $F _ { ; }$ define

$$
\| F \| _ { L ^ { s } ( \nu ) } : = \left( \int \| F ( x ) \| ^ { s } \nu ( \mathrm { d } x ) \right) ^ { 1 / s } , \qquad 1 \le s < \infty .\tag{2.2}
$$

The pointwise norm in (2.2) is the absolute value for scalars, the Euclidean norm for vectors, and the Frobenius norm for matrices. Thus, for a matrix field $F .$

$$
\| F \| _ { L ^ { 2 } ( \nu ) } ^ { 2 } = \int \| F ( x ) \| _ { F } ^ { 2 } \nu ( \mathrm { d } x ) = \sum _ { i , j } \int | F _ { i j } ( x ) | ^ { 2 } \nu ( \mathrm { d } x ) .
$$

In particular, for a positive density ν and a scalar, vector, or matrix density $F$

$$
\| F / \nu \| _ { L ^ { s } ( \nu ) } ^ { s } = \int \| F ( x ) \| ^ { s } \nu ( x ) ^ { 1 - s } \mathrm { d } x .
$$

For $v , w \in \mathbb { R } ^ { d }$ , the outer product is the matrix

$$
( v \otimes w ) _ { i j } : = v _ { i } w _ { j } , \qquad \| v \otimes w \| _ { F } = \| v \| \left\| w \right\| .\tag{2.3}
$$

## 2.1 Weak derivatives

For a locally integrable function u, its distributional derivative is defined by

$$
\langle \partial _ { i } u , \phi \rangle = - \int u \partial _ { i } \phi \mathrm { d } x , \qquad \phi \in C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { d } ) ,
$$

where $C _ { c } ^ { \infty }$ denotes smooth functions with compact support. If this derivative is represented by a locally integrable function v, then v is the weak derivative of u. Lipschitz functions have almosteverywhere derivatives, meaning their weak derivatives. An equality of distributions implies an equality against every such test function. When both sides are locally integrable functions, this is equivalent to equality almost everywhere; it does not assert the existence of classical derivatives at every point.

For a vector field H and a matrix field $E ,$ our divergence convention is

$$
\operatorname { d i v } H = \sum _ { i } \partial _ { i } H _ { i } , \qquad ( \operatorname { d i v } E ) _ { i } = \sum _ { j } \partial _ { j } E _ { i j } , \qquad \operatorname { d i v } \operatorname { d i v } E = \sum _ { i , j } \partial _ { i } \partial _ { j } E _ { i j } .\tag{2.4}
$$

In particular,

$$
\langle \operatorname { d i v } H , \phi \rangle = - \int \langle \nabla \phi , H \rangle ~ \mathrm { d } x , \qquad \langle \operatorname { d i v } \operatorname { d i v } E , \phi \rangle = \int \langle E , \nabla ^ { 2 } \phi \rangle \mathrm { d } x .\tag{2.5}
$$

These formulas require only local integrability of the fields. They allow density equations to be used through integration by parts without diferentiating the densities classically. Hessians are written explicitly as $\nabla ^ { 2 } f , \nabla ^ { \bar { 2 } } g _ { \lambda } \bar { }$ , or $\nabla ^ { 2 } U _ { \lambda } \mathbf { ; }$ ; the letter H denotes a vector flux. Constants $c , C > 0$ are universal and may change from line to line. The notation $\widetilde O$ suppresses fixed powers of logarithms, but no polynomial parameter factors.

## 2.2 Moreau regularization

For a convex, globally G-Lipschitz function $g : \mathbb { R } ^ { d }  \mathbb { R }$ and $\lambda > 0$ , let

$$
g _ { \lambda } ( x ) : = \operatorname* { i n f } _ { y \in \mathbb { R } ^ { d } } \left\{ g ( y ) + { \frac { \| x - y \| ^ { 2 } } { 2 \lambda } } \right\} .\tag{2.6}
$$

The unique minimizer is denoted by $\mathrm { p r o x } _ { \lambda g } ( x )$ . The Moreau properties used below are

$$
\nabla g _ { \lambda } ( x ) = \lambda ^ { - 1 } \big ( x - \mathrm { p r o x } _ { \lambda g } ( x ) \big ) , \quad \| \nabla g _ { \lambda } ( x ) \| \leq G , \quad 0 \preceq \nabla ^ { 2 } g _ { \lambda } \preceq \lambda ^ { - 1 } I \quad \mathrm { a . e . }\tag{2.7}
$$

In particular, $g _ { \lambda } \in C ^ { 1 , 1 }$ , meaning that its gradient is globally Lipschitz. The last inequality concerns its weak Hessian, not a continuous second derivative. We use these facts in the form recalled in Xin and Zhang [9, Section 3.2 and Appendix $\mathrm { A } ]$

## 2.3 Drift maps, pushforwards, and density evolution

The potential used for MYULA will be specified in Section 3.1. For the remainder of this section, we use only its regularity and convexity properties: fix $\lambda > 0$ and let $U _ { \lambda } \in C ^ { 1 , 1 } ( \mathbb { R } ^ { d } )$ be m-strongly convex with L -Lipschitz gradient, where $0 < m \le L _ { \lambda } < \infty$ . Write $\pi _ { \lambda }$ for the probability density proportional to $e ^ { - U _ { \lambda } }$ . Its weak Hessian satisfies

$$
m I \preceq \nabla ^ { 2 } U _ { \lambda } \preceq L _ { \lambda } I { \mathrm { a l m o s t ~ e v e r y w h e r e . } }\tag{2.8}
$$

The deterministic drift step is

$$
T _ { t } ( x ) : = x - t \nabla U _ { \lambda } ( x ) .\tag{2.9}
$$

For an integrable scalar, vector, or matrix density $F _ { ; }$ , the pushforward of F dx assigns to each Borel set the integral of $F$ over its inverse image under $T _ { t }$ . This defines a unique weighted measure, componentwise. If it has a Lebesgue density, we denote that density by $( T _ { t } ) _ { \# } F \colon$ ; equivalently,

$$
\int \phi ( y ) ( T _ { t } ) _ { \# } F ( y ) \mathrm { d } y = \int \phi ( T _ { t } x ) F ( x ) \mathrm { d } x \quad \mathrm { f o r ~ e v e r y ~ } \phi \in C _ { c } ^ { \infty } ( \mathbb { R } ^ { d } ) .\tag{2.10}
$$

For a probability density, the pushforward is the law of the transformed random variable $T _ { t } X$ . For vector and matrix densities, the pushforward is defined componentwise.

Proposition 2.1 (Drift pushforward). $I f 0 \leq t L _ { \lambda } < 1$ , then $T _ { t }$ is a global bi-Lipschitz bijection and

$$
\mathrm { L i p } ( T _ { t } ^ { - 1 } ) \leq ( 1 - t L _ { \lambda } ) ^ { - 1 } .
$$

For every integrable scalar, vector, or matrix density $F _ { \mathrm { { ; } } }$ , its pushforward has the density

$$
( T _ { t } ) _ { \# } F ( y ) = \frac { F ( T _ { t } ^ { - 1 } y ) } { \operatorname * { d e t } \bigl ( I - t \nabla ^ { 2 } U _ { \lambda } ( T _ { t } ^ { - 1 } y ) \bigr ) } \quad f o r \ a . e . \ y .\tag{2.11}
$$

In particular, $\| ( T _ { t } ) _ { \# } F \| _ { L ^ { 1 } ( \mathbb { R } ^ { d } ) } = \| F \| _ { L ^ { 1 } ( \mathbb { R } ^ { d } ) }$

Proof. For each $y ,$ the map $x \mapsto y + t \nabla U _ { \lambda } ( x )$ is a contraction on $\mathbb { R } ^ { d }$ . Its unique fixed point is $T _ { t } ^ { - 1 } y$ The fixed-point equations for two values of y give the stated inverse Lipschitz bound, and $T _ { t }$ is Lipschitz by definition. Its derivative is $D T _ { t } = I - t \nabla ^ { 2 } U _ { \lambda }$ almost everywhere, with

$$
\operatorname* { d e t } ( I - t \nabla ^ { 2 } U _ { \lambda } ) \geq ( 1 - t L _ { \lambda } ) ^ { d } > 0 .
$$

The weighted area formula [7, Chapter 2, Theorem 3.3 and Remark $3 . 4 ( 2 ) ]$ applied to the Lipschitz bijection $T _ { t }$ gives

$$
\int q ( x ) \operatorname* { d e t } ( I - t \nabla ^ { 2 } U _ { \lambda } ( x ) ) \mathrm { d } x = \int q ( T _ { t } ^ { - 1 } y ) \mathrm { d } y
$$

for every nonnegative measurable scalar function q. Applying this formula with the integrand divided by the positive Jacobian, and then taking positive and negative parts componentwise, yields

$$
\int \phi ( T _ { t } x ) F ( x ) \mathrm { d } x = \int \phi ( y ) { \frac { F ( T _ { t } ^ { - 1 } y ) } { \operatorname* { d e t } ( I - t \nabla ^ { 2 } U _ { \lambda } ( T _ { t } ^ { - 1 } y ) ) } } \mathrm { d } y .
$$

Both $T _ { t }$ and $T _ { t } ^ { - 1 }$ preserve Lebesgue null sets because they are Lipschitz, so the almost-everywhere derivatives sufice for these formulas. The last identity proves (2.11). Changing variables in its pointwise norm gives the stated $L ^ { 1 }$ identity. □

The free heat semigroup is

$$
\forall _ { t } F ( x ) : = ( 4 \pi t ) ^ { - d / 2 } \int e ^ { - \| x - y \| ^ { 2 } / ( 4 t ) } F ( y ) \mathrm { d } y , \qquad t > 0 , \qquad \mathsf { H } _ { 0 } F = F .\tag{2.12}
$$

It acts componentwise and adds an independent $N ( 0 , 2 t I _ { d } )$ increment to a probability density. For $F \in L ^ { 1 } ( \mathbb { R } ^ { d } )$ , it preserves the integral and satisfies $\| \mathsf { H } _ { t } F \| _ { L ^ { 1 } } \leq \| F \| _ { L ^ { 1 } }$ . Moreover, $\mathsf { H } _ { t } F \to F$ in $L ^ { 1 }$ as $t \downarrow 0 .$ , since the Gaussian kernels form an approximate identity.

We consider a gradient bound and an integrated form of the heat equation. For scalar $F \in L ^ { 1 } ( \mathbb { R } ^ { d } )$ diferentiation of the Gaussian kernel, the triangle inequality, and Tonelli’s theorem give

$$
\begin{array} { r l r } & { } & { \| \nabla \mathsf { H } _ { t } F \| _ { L ^ { 1 } ( \mathbb { R } ^ { d } ) } \leq \| F \| _ { L ^ { 1 } ( \mathbb { R } ^ { d } ) } \displaystyle \int _ { \mathbb { R } ^ { d } } \frac { \| z \| } { 2 t } ( 4 \pi t ) ^ { - d / 2 } e ^ { - \| z \| ^ { 2 } / ( 4 t ) } \mathrm { d } z } \\ & { } & { \qquad \leq \displaystyle \frac { C _ { d } } { \sqrt { t } } \| F \| _ { L ^ { 1 } ( \mathbb { R } ^ { d } ) } , \qquad C _ { d } : = \sqrt { d / 2 } . } \end{array}\tag{2.13}
$$

The second inequality is Cauchy–Schwarz and the second moment 2td of this Gaussian density. In particular, $t \left\| \nabla \mathsf { H } _ { t } F \right\| _ { L ^ { 1 } }$ is integrable near $t = 0$

For $\phi \in C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { d } )$ , symmetry of the Gaussian kernel gives

$$
\int \phi \mathsf { H } _ { t } F \mathrm { d } x = \int F \mathsf { H } _ { t } \phi \mathrm { d } x .
$$

For $t > 0$ , diferentiation of the kernel and integration by parts against the smooth function $\phi$ yield $\partial _ { t } \mathsf { H } _ { t } \phi = \mathsf { H } _ { t } \Delta { \phi }$ . Since

$$
| F ( x ) \partial _ { t } \mathsf { H } _ { t } \phi ( x ) | \leq | F ( x ) | \| \Delta \phi \| _ { \infty } ,
$$

diferentiation under the integral is justified by dominated convergence. Hence

$$
\frac { \mathrm { d } } { \mathrm { d } t } \int \phi \mathsf { H } _ { t } F \mathrm { d } x = \int ( \Delta \phi ) \mathsf { H } _ { t } F \mathrm { d } x , \qquad t > 0 .\tag{2.14}
$$

The derivative is bounded by $\| F \| _ { L ^ { 1 } } \left\| \Delta \phi \right\| _ { \infty }$ . Integrating first over a positive-time interval and then using $\mathsf { H } _ { t } F \to F$ in $L ^ { 1 }$ at the left endpoint gives

$$
\int \phi ( \mathsf { H } _ { b } F - \mathsf { H } _ { a } F ) \mathrm { d } x = \int _ { a } ^ { b } \int ( \Delta \phi ) \mathsf { H } _ { t } F \mathrm { d } x \mathrm { d } t , \qquad 0 \leq a \leq b < \infty .\tag{2.15}
$$

Thus, for $F \in L ^ { 1 } ( \mathbb { R } ^ { d } )$ and $\phi \in C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { d } )$ , the map $t \mapsto \int \phi { \sf H } _ { t } F$ dx is absolutely continuous on [0, T] for every $T > 0$

The Langevin generator and its formal adjoint with respect to Lebesgue measure are defined by

$$
\begin{array} { r } { \mathcal { L } _ { \lambda } \phi = \Delta \phi - \left. \nabla U _ { \lambda } , \nabla \phi \right. , \qquad \mathcal { L } _ { \lambda } ^ { \ast } \rho = \Delta \rho + \mathrm { d i v } ( \rho \nabla U _ { \lambda } ) . } \end{array}\tag{2.16}
$$

Here $\mathcal { L } _ { \lambda } ^ { * } \rho$ is understood distributionally: $\begin{array} { r } { \langle \mathcal { L } _ { \lambda } ^ { * } \rho , \phi \rangle = \int \rho \mathcal { L } _ { \lambda } \phi } \end{array}$ dx for $\phi \in C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { d } )$ . Because $\nabla \pi _ { \lambda } =$ $- \pi \chi U _ { \lambda }$ , we have $\mathcal { L } _ { \lambda } ^ { * } \pi _ { \lambda } = 0$

## 2.4 Poisson equations and transport

The Poisson equation will be used to convert a residual in the stationary density equation into a diference of expectations. The weighted Sobolev space $H ^ { 1 } ( \pi _ { \lambda } )$ consists of functions whose value and weak gradient belong to $L ^ { 2 } ( \pi _ { \lambda } )$ . Strong convexity gives the Poincar´e inequality

$$
\int _ { \mathrm { \Omega } } \left| v - \int v \mathrm { d } \pi _ { \lambda } \right| ^ { 2 } \mathrm { d } \pi _ { \lambda } \leq \frac { 1 } { m } \int \left\| \nabla v \right\| ^ { 2 } \mathrm { d } \pi _ { \lambda } ;\tag{2.17}
$$

see, for example, Chewi [1]. For $\phi \in L ^ { 2 } ( \pi _ { \lambda } )$ , the mean-zero Poisson equation is

$$
{ \mathcal L } _ { \lambda } \psi = \phi - \int \phi \mathrm { d } \pi _ { \lambda } , \qquad \int \psi \mathrm { d } \pi _ { \lambda } = 0 .\tag{2.18}
$$

A weak solution is a function $\psi \in H ^ { 1 } ( \pi _ { \lambda } )$ satisfying

$$
\int \langle \nabla \psi , \nabla v \rangle ~ \mathrm { d } \pi _ { \lambda } = - \int \left( \phi - \int \phi \mathrm { d } \pi _ { \lambda } \right) v \mathrm { d } \pi _ { \lambda } , \qquad v \in H ^ { 1 } ( \pi _ { \lambda } ) .\tag{2.19}
$$

Here $\phi$ is prescribed and $\psi$ is the unknown function; the zero-mean condition removes the freedom to add a constant. For $k = 1 , 2$ , the unweighted Sobolev space $H ^ { k } ( \mathbb { R } ^ { d } )$ consists of functions whose weak derivatives of order at most k belong to $L ^ { 2 } ( \mathbb { R } ^ { d } )$ with respect to Lebesgue measure. We write $u \in H _ { \mathrm { l o c } } ^ { 2 } (  { \mathbb { R } } ^ { d } )$ if $\eta u \in H ^ { 2 } ( \mathbb { R } ^ { d } )$ for every $\eta \in C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { d } )$ . The next lemma shows the regularity and approximation needed below.

Lemma 2.2 (Poisson regularity and approximation). Let $U _ { \lambda } \in C ^ { 1 , 1 } ( \mathbb { R } ^ { d } )$ be m-strongly convex with L -Lipschitz gradient, where $0 < m \le L _ { \lambda } < \infty$ . Let $\pi _ { \lambda } \propto e ^ { - U _ { \lambda } }$ and let $\mathcal { L } _ { \lambda }$ be the generator in (2.16). For every $\phi \in L ^ { 2 } ( \pi _ { \lambda } )$ , the Poisson equation (2.18) has a unique mean-zero weak solution $\psi \in H ^ { 1 } ( \pi _ { \lambda } )$ . Moreover, $\psi \in H _ { \mathrm { l o c } } ^ { 2 } ( \mathbb { R } ^ { d } )$ $I f \phi \in C _ { c } ^ { \infty } ( \mathbb { R } ^ { d } )$ , then $\nabla ^ { 2 } \psi \in L ^ { 2 } ( \pi _ { \lambda } )$ and

$$
\begin{array} { r l } & { \displaystyle \int \left\| \nabla ^ { 2 } \psi \right\| _ { F } ^ { 2 } \mathrm { d } \pi _ { \lambda } + \int \nabla \psi ^ { \top } ( \nabla ^ { 2 } U _ { \lambda } ) \nabla \psi \mathrm { d } \pi _ { \lambda } } \\ & { \quad \leq \displaystyle \int \left. \phi - \int \phi \mathrm { d } \pi _ { \lambda } \right. ^ { 2 } \mathrm { d } \pi _ { \lambda } \leq \frac { 1 } { m } \displaystyle \int \left\| \nabla \phi \right\| ^ { 2 } \mathrm { d } \pi _ { \lambda } . } \end{array}\tag{2.20}
$$

For such $\phi$ , there exist $\psi _ { n } \in C _ { c } ^ { \infty } ( \mathbb { R } ^ { d } )$ such that

$$
\| \mathcal { L } _ { \lambda } \psi _ { n } - \mathcal { L } _ { \lambda } \psi \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } + \left\| \nabla ^ { 2 } \psi _ { n } - \nabla ^ { 2 } \psi \right\| _ { L ^ { 2 } ( \pi _ { \lambda } ) } + \| \nabla \psi _ { n } - \nabla \psi \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } \longrightarrow 0 .\tag{2.21}
$$

The proof is given in Appendix C. The estimate (2.20) controls the weak derivatives of the Poisson solution, whereas (2.21) allows it to be used in residual identities initially stated only for compactly supported smooth tests. Neither assertion requires a continuous Hessian or a third derivative of $U _ { \lambda }$

For an integrable signed density $\sigma$ with R σ $\mathrm { d } x = 0$ , define

$$
\| \sigma \| _ { \dot { H } ^ { - 1 } ( \pi _ { \lambda } ) } : = \operatorname* { s u p } _ { \phi \in C _ { c } ^ { \infty } ( \mathbb { R } ^ { d } ) \atop \| \nabla \phi \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } \leq 1 } \left| \int \phi \sigma \mathrm { d } x \right| .\tag{2.22}
$$

Here $\sigma$ represents the signed measure with Lebesgue density $\sigma$ , not a density relative to $\pi _ { \lambda }$ . In particular, the function–measure pairing is

$$
\left. \phi , \sigma ( x ) \mathrm { d } x \right. = \int \phi \sigma \mathrm { d } x = \int \phi { \frac { \sigma } { \pi _ { \lambda } } } \mathrm { d } \pi _ { \lambda } .\tag{2.23}
$$

If $\sigma / \pi _ { \lambda } \in L ^ { 2 } ( \pi _ { \lambda } )$ , the same norm is obtained using either weighted Sobolev tests or $C ^ { 1 }$ tests:

$$
\| \sigma \| _ { \dot { H } ^ { - 1 } ( \pi _ { \lambda } ) } = \operatorname* { s u p } _ { \stackrel { \phi \in H ^ { 1 } ( \pi _ { \lambda } ) } { \| \nabla \phi \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } \leq 1 } } \left| \int \phi \sigma { \mathrm { d } } x \right| = \operatorname* { s u p } _ { \stackrel { \phi \in C ^ { 1 } ( \mathbb { R } ^ { d } ) } { \| \nabla \phi \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } \leq 1 } } \left| \int \phi \sigma { \mathrm { d } } x \right| .\tag{2.24}
$$

The approximation and integrability needed for these equalities are proved in Section C.3. Consequently, (2.22) agrees, in this setting, with the norm of the signed measure $\sigma ( x )$ dx in Peyre [6, Section 1.1].

For a probability density $\nu \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ with $\nu / \pi _ { \lambda } \in L ^ { 2 } ( \pi _ { \lambda } )$ , apply Peyre [6, Theorem 1] to the measures $\pi _ { \lambda } ( x )$ dx and $\nu ( x )$ dx. By (2.23) and (2.24), this gives

$$
W _ { 2 } ( \nu , \pi _ { \lambda } ) \leq 2 \left. \nu - \pi _ { \lambda } \right. _ { \dot { H } ^ { - 1 } ( \pi _ { \lambda } ) } .\tag{2.25}
$$

## 2.5 Entropy and density-ratio moments

For $F \geq 0$ , write

$$
\mathrm { E n t } _ { \nu } F : = \int F \log F \mathrm { d } \nu - \left( \int F \mathrm { d } \nu \right) \log \int F \mathrm { d } \nu .
$$

Strong convexity also gives the log-Sobolev inequality [1]

$$
\operatorname { E n t } _ { \pi _ { \lambda } } ( w ^ { 2 } ) \leq \frac { 2 } { m } \int \| \nabla w \| ^ { 2 } ~ \mathrm { d } \pi _ { \lambda } .\tag{2.26}
$$

For $\alpha > 1$ and a probability density $\nu ,$ the R´enyi divergence is

$$
\mathcal { R } _ { \alpha } ( \nu \| \pi _ { \lambda } ) : = \frac { 1 } { \alpha - 1 } \log \int \left( \frac { \nu } { \pi _ { \lambda } } \right) ^ { \alpha } \mathrm { d } \pi _ { \lambda } .\tag{2.27}
$$

Equivalently, $\| \nu / \pi _ { \lambda } \| _ { L ^ { \alpha } ( \pi _ { \lambda } ) } = \exp \{ ( \alpha - 1 ) \mathcal { R } _ { \alpha } ( \nu \| \pi _ { \lambda } ) / \alpha \}$

## 3 The Moreau–Yosida Unadjusted Langevin Sampling

In this section we study theoretical analysis for error bounds of the Moreau–Yosida unadjusted Langevin algorithm (MYULA). We first give some assumptions with the problem description, and then present our main theoretical results.

## 3.1 Problem formulation and assumptions

We consider the probability distribution

$$
\pi ( \mathrm { d } \boldsymbol { x } ) = Z ^ { - 1 } \exp \{ - f ( \boldsymbol { x } ) - g ( \boldsymbol { x } ) \} \mathrm { d } \boldsymbol { x } , \qquad \boldsymbol { x } \in \mathbb { R } ^ { d } .\tag{3.1}
$$

The assumptions and normalization below follow those of Xin and Zhang [9].

Assumption 3.1 (Smooth component). The function $f \in C ^ { 2 } (  { \mathbb { R } } ^ { d } )$ satisfies

$$
m I \preceq \nabla ^ { 2 } f ( x ) \preceq L _ { f } I , \qquad x \in \mathbb { R } ^ { d } , \qquad 0 < m \leq L _ { f } < \infty .\tag{3.2}
$$

Write

$$
\tau _ { f } : = \operatorname* { s u p } _ { \boldsymbol { x } \in \mathbb { R } ^ { d } } \operatorname { t r } [ \nabla ^ { 2 } f ( \boldsymbol { x } ) ] .\tag{3.3}
$$

In particular, md $\leq \tau _ { f } \leq d L _ { f }$

Assumption 3.2 (Nonsmooth component). The function $g \colon  { \mathbb { R } ^ { d } } \to  { \mathbb { R } }$ is convex and globally G-Lipschitz, for some $G > 0$

For the Moreau envelope in (2.6), set

$$
U _ { \lambda } : = f + g _ { \lambda } , \qquad \pi _ { \lambda } ( \mathrm { d } x ) : = Z _ { \lambda } ^ { - 1 } e ^ { - U _ { \lambda } ( x ) } \mathrm { d } x , \qquad L _ { \lambda } : = L _ { f } + \lambda ^ { - 1 } .\tag{3.4}
$$

By Assumptions 3.1 and 3.2 and (2.7), this choice of $U _ { \lambda }$ belongs to $C ^ { 1 , 1 } ( \mathbb { R } ^ { d } )$ and satisfies (2.8).   
Thus the operator constructions and Poisson results in Sections 2.3 and 2.4 apply.

MYULA is the fixed-step recursion

$$
\widehat { X } _ { k + 1 } = \widehat { X } _ { k } - h \nabla U _ { \lambda } ( \widehat { X } _ { k } ) + \sqrt { 2 h } \xi _ { k + 1 } , \qquad \xi _ { k + 1 } \overset { \mathrm { i . i . d . } } { \sim } N ( 0 , I _ { d } ) .\tag{3.5}
$$

Its transition kernel is $Q _ { \lambda , h } .$ , and $\mu _ { k } : = \mathcal { L } ( \widehat { X } _ { k } )$ . Under the step-size condition below, its invariant law is denoted by $\widehat { \pi } _ { \lambda , h }$ . The initial law $\mu _ { 0 }$ belongs to $\mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ . We measure accuracy by $\sqrt { m } W _ { 2 } ( \mu _ { k } , \pi )$ Each iteration uses one evaluation of $\nabla f$ and one exact proximal evaluation of $g$ .

Let

$$
R : = \frac { G + \sqrt { G ^ { 2 } + 4 \tau _ { f } } } { 2 } .\tag{3.6}
$$

Thus, $R ^ { 2 } = \tau _ { f } + G R , R \geq G$ , and $R ^ { 2 } \geq \tau _ { f } \geq m d .$

Proposition 3.3 (Invariant law and contraction). Under Assumptions 3.1 and 3.2, $i f 0 < h L _ { \lambda } < 1$ then $Q _ { \lambda , h }$ has a unique invariant law in $\mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ , with a smooth strictly positive density $\widehat { \pi } _ { \lambda , h }$ . For $\mu , \nu \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ 2

$$
W _ { 2 } ( \mu Q _ { \lambda , h } , \nu Q _ { \lambda , h } ) \leq ( 1 - m h ) W _ { 2 } ( \mu , \nu ) .\tag{3.7}
$$

Moreover,

$$
\widehat { \pi } _ { \lambda , h } = \mathsf { H } _ { h } \big ( ( T _ { h } ) _ { \# } \widehat { \pi } _ { \lambda , h } \big ) .\tag{3.8}
$$

Proof. For a smooth potential, integration of the Hessian along a segment gives $\| T _ { h } x - T _ { h } y \| \le$ $( 1 - m h ) \parallel x - y \parallel$ . Mollification preserves the Hessian bounds and gives the same inequality for $U _ { \lambda } \in C ^ { 1 , 1 }$ by uniform convergence of its gradient. Using the same Gaussian increment for two Euler updates proves (3.7). The linear growth of the drift implies that the kernel maps $\mathcal { P } _ { 2 }$ into itself. The contraction theorem on the complete metric space $( \mathcal { P } _ { 2 } , W _ { 2 } )$ therefore gives the unique invariant law in that space. An Euler update consists of the pushforward by $T _ { h }$ followed by the heat operator $\mathsf { H } _ { h }$ which proves (3.8). Convolution of a probability measure with the Gaussian kernel has a smooth strictly positive density. □

## 3.2 Main results

The first theorem controls the invariant-measure bias at fixed smoothing; the second gives a single step size for the original target. Set

$$
\ell _ { h , \lambda } : = 1 + \log \left( e + \frac { 1 } { m h } + \frac { L _ { \lambda } } { m } \right) ,\tag{3.9}
$$

$$
\ell _ { \varepsilon } : = 1 + \log \left( e + \frac { L _ { f } } { m } + \frac { \tau _ { f } } { m } + \frac { G ^ { 2 } } { m } + \frac { 1 } { \varepsilon } \right) .\tag{3.10}
$$

Theorem 3.4 (Invariant-measure bias at fixed smoothing). Under Assumptions 3.1 and 3.2, there exist universal constants $c , C > 0$ such that, for every $\lambda , h > 0$ satisfying

$$
h \ell _ { h , \lambda } ^ { 2 } \left[ \frac { L _ { f } \tau _ { f } } { m } + \frac { L _ { f } R } { \sqrt { m } } + \frac { G R } { m \lambda } + R ^ { 2 } + \frac { 1 } { \lambda } \right] \leq c ,\tag{3.11}
$$

the kernel $Q _ { \lambda , h }$ has a unique invariant law $\widehat { \pi } _ { \lambda , h } \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ , and

$$
\sqrt { m } W _ { 2 } ( \widehat { \pi } _ { \lambda , h } , \pi _ { \lambda } ) \leq C h R ^ { 2 } \ell _ { h , \lambda } ^ { 2 } .\tag{3.12}
$$

The proof of Theorem 3.4 is given in Section 4.4. The suficient restriction (3.11) is stronger than $h L _ { \lambda } \leq c$

Theorem 3.5 (End-to-end complexity). Under Assumptions 3.1 and 3.2, let $0 < \varepsilon \le 1$ and choose

$$
\lambda _ { \varepsilon } : = \frac { \varepsilon } { G ^ { 2 } } ,\tag{3.13}
$$

$$
h _ { \varepsilon } : = \frac { c } { \ell _ { \varepsilon } ^ { 2 } } \left[ \frac { L _ { f } \tau _ { f } } { m } + \frac { L _ { f } R } { \sqrt { m } } + \frac { R ^ { 2 } + G ^ { 3 } R / m } { \varepsilon } \right] ^ { - 1 } ,\tag{3.14}
$$

where $c > 0$ is a suficiently small universal constant. For every $\mu _ { 0 } \in \mathcal P _ { 2 } ( \mathbb { R } ^ { d } )$ , if

$$
N \geq \frac { C } { m h _ { \varepsilon } } \log \left( 2 + \frac { \sqrt { m } W _ { 2 } ( \mu _ { 0 } , \pi ) } { \varepsilon } \right) ,\tag{3.15}
$$

then

$$
\sqrt { m } W _ { 2 } \big ( \mu _ { 0 } Q _ { \lambda _ { \varepsilon } , h _ { \varepsilon } } ^ { N } , \pi \big ) \leq \varepsilon .\tag{3.16}
$$

Moreover, the smallest integer satisfying (3.15) obeys

$$
N \leq 1 + \frac { C \ell _ { \varepsilon } ^ { 2 } } { m } \left[ \frac { L _ { f } \tau _ { f } } { m } + \frac { L _ { f } R } { \sqrt { m } } + \frac { R ^ { 2 } + G ^ { 3 } R / m } { \varepsilon } \right] \log \left( 2 + \frac { \sqrt { m } W _ { 2 } ( \mu _ { 0 } , \pi ) } { \varepsilon } \right) .\tag{3.17}
$$

In particular, for fixed model parameters and fixed initialization, $N ( \varepsilon ) = \widetilde { O } ( \varepsilon ^ { - 1 } )$

The proof of Theorem 3.5 is given in Section 4.5. The smoothing step uses the Moreau approximation estimate of Xin and Zhang [8, Proposition 5.9], also used in Xin and Zhang [9, Section 6.5]:

$$
\sqrt { m } W _ { 2 } ( \pi _ { \lambda } , \pi ) \leq \frac { G ^ { 2 } \lambda } { 4 } .\tag{3.18}
$$

The choices (3.13)–(3.14) leave this bias budget unchanged.

Remark 3.6 (Parameter dependence). Using $R \leq G + { \sqrt { \tau _ { f } } }$ , the suficient iteration bound (3.17) implies

$$
N = \widetilde { \cal O } \left[ \frac { L _ { f } \tau _ { f } } { m ^ { 2 } } + \frac { L _ { f } G } { m ^ { 3 / 2 } } + \frac 1 \varepsilon \left( \frac { \tau _ { f } } { m } + \frac { G ^ { 3 } \sqrt { \tau _ { f } } } { m ^ { 2 } } + \frac { G ^ { 4 } } { m ^ { 2 } } \right) \right] ,\tag{3.19}
$$

where $\widetilde O$ suppresses the logarithmic factors in (3.17). For fixed $L _ { f } / m$ and $G / \sqrt { m }$ , the inequality $\tau _ { f } \leq d L _ { f }$ gives $N = \widetilde { O } ( d / \varepsilon )$

## 4 Proof of the Main Results

This section proves Theorems 3.4 and 3.5. We first use stationarity and a Poisson equation to bound $W _ { 2 } ( \widehat { \pi } _ { \lambda , h } , \pi _ { \lambda } )$ , using the flux estimates proved in Section 5. We then combine this bound with the Moreau approximation error and contraction of the chain to obtain the iteration bound.

## 4.1 The analytic estimates

Lemma 4.1 (Target drift moments). Under Assumptions 3.1 and 3.2,

$$
\int \| \nabla U _ { \lambda } \| ^ { 2 } ~ \mathrm { d } \pi _ { \lambda } = \int \operatorname { t r } ( \nabla ^ { 2 } U _ { \lambda } ) \mathrm { d } \pi _ { \lambda } \leq R ^ { 2 } ,\tag{4.1}
$$

$$
\| \nabla U _ { \lambda } \| _ { L ^ { s } ( \pi _ { \lambda } ) } \le 3 \sqrt s R , \qquad s \ge 2 ,\tag{4.2}
$$

$$
\int \exp \left( \frac { \| \nabla U _ { \lambda } \| ^ { 2 } } { 2 5 6 R ^ { 2 } } \right) \mathrm { d } \pi _ { \lambda } \leq 2 .\tag{4.3}
$$

The second-moment identity and bound follow from Xin and Zhang [9, Lemma 7.1 and Proposition $\mathrm { A . 4 } ]$ . The higher-moment and exponential estimates are proved in Section A.1. These estimates control the mean curvature and the drift moments using G and $\tau _ { f } .$ , uniformly in λ.

Lemma 4.2 (Finite-order stationary density ratios). Let $\alpha \geq 2$ . Suppose $h L _ { \lambda } \leq c$ and

$$
h \frac { L _ { f } \tau _ { f } } { m } \leq \frac { c } { \alpha ^ { 2 } } , \qquad h \frac { L _ { f } R } { \sqrt { m } } \leq \frac { c } { \alpha } , \qquad \frac { h } { \lambda } \frac { G R } { m } \leq \frac { c } { \alpha ^ { 2 } } .\tag{4.4}
$$

Then the stationary Euler interpolation

$$
\nu _ { t } : = \mathsf { H } _ { t } ( ( T _ { t } ) _ { \# } \widehat { \pi } _ { \lambda , h } ) , \qquad 0 \leq t \leq h , \qquad \nu _ { 0 } = \nu _ { h } = \widehat { \pi } _ { \lambda , h } ,\tag{4.5}
$$

satisfies

$$
\operatorname* { s u p } _ { 0 \leq t \leq h } \| \nu _ { t } / \pi _ { \lambda } \| _ { L ^ { \alpha } ( \pi _ { \lambda } ) } \leq e ^ { 1 / 4 } .\tag{4.6}
$$

The proof is given in Appendix B. The prior integrability needed for diferentiation is established independently in Section D.2. The equal endpoint laws close a R´enyi dissipation estimate over one Euler step using the log-Sobolev inequality for $\pi _ { \lambda }$

Lemma 4.3 (Normalized heat and pushforward estimates). The estimates below apply to measurable scalar, vector, or matrix densities F, as indicated, whenever the norm on the right-hand side is finite. For $s \geq 2 , t > 0$ , and $t s ^ { 2 } R ^ { 2 } \leq c ,$

$$
\| \mathsf { H } _ { t } F / \pi _ { \lambda } \| _ { L ^ { s } ( \pi _ { \lambda } ) } \leq C \| F / \pi _ { \lambda } \| _ { L ^ { 2 s } ( \pi _ { \lambda } ) } ,\tag{4.7}
$$

$$
\| \nabla \mathsf { H } _ { t } F / \pi _ { \lambda } \| _ { L ^ { s } ( \pi _ { \lambda } ) } \leq \frac { C } { \sqrt { t } } \| F / \pi _ { \lambda } \| _ { L ^ { 2 s } ( \pi _ { \lambda } ) }
$$

$$
f o r \ s c a l a r \ F ,\tag{4.8}
$$

$$
\| \nabla \operatorname { d i v } \mathsf { H } _ { t } F / \pi _ { \lambda } \| _ { L ^ { s } ( \pi _ { \lambda } ) } \leq \frac { C \sqrt { s } } { t } \| F / \pi _ { \lambda } \| _ { L ^ { 4 s } ( \pi _ { \lambda } ) }
$$

$$
f o r \ v e c t o r \ F .\tag{4.9}
$$

Moreover, if $s t ( L _ { \lambda } + R ^ { 2 } ) \leq c ,$ , then

$$
\| ( T _ { t } ) _ { \# } F / \pi _ { \lambda } \| _ { L ^ { s } ( \pi _ { \lambda } ) } \leq C \| F / \pi _ { \lambda } \| _ { L ^ { 2 s } ( \pi _ { \lambda } ) } .\tag{4.10}
$$

The zero-order and pushforward estimates apply to scalar, vector, and matrix densities and extend to $t = 0$

The proof is given in Sections A.2 to A.4.

## 4.2 The stationary residual

Assume $h L _ { \lambda } < 1$ and let $\widehat { \pi } _ { \lambda , h }$ be the invariant density from Proposition 3.3. Starting from $\widehat { \pi } _ { \lambda , h }$ , a full drift step followed by heat evolution for time h returns to $\widehat { \pi } _ { \lambda , h }$ . We compare this endpoint with the average along the heat path. Define

$$
J : = \frac { 1 } { h } \int _ { 0 } ^ { h } ( T _ { s } ) _ { \# } ( \nabla U _ { \lambda } \widehat { \pi } _ { \lambda , h } ) \mathrm { d } s ,\tag{4.11}
$$

and

$$
\rho _ { t } : = \mathsf { H } _ { t } ( ( T _ { h } ) _ { \# } \widehat { \pi } _ { \lambda , h } ) , \qquad \bar { \rho } : = \frac { 1 } { h } \int _ { 0 } ^ { h } \rho _ { t } \mathrm { d } t , \qquad 0 \leq t \leq h .\tag{4.12}
$$

By stationarity,

$$
\rho _ { 0 } = ( T _ { h } ) _ { \# } \widehat { \pi } _ { \lambda , h } , \qquad \rho _ { h } = \widehat { \pi } _ { \lambda , h } .
$$

Unlike the Euler interpolation in (4.5), this path keeps $T _ { h }$ fixed as t varies. Define also

$$
H : = \frac 1 h \int _ { 0 } ^ { h } t \nabla \rho _ { t } \mathrm d t ,\tag{4.13}
$$

$$
E : = - \frac { 1 } { h } \int _ { 0 } ^ { h } ( h - s ) ( T _ { s } ) _ { \# } \big ( ( \nabla U _ { \lambda } \otimes \nabla U _ { \lambda } ) \widehat { \pi } _ { \lambda , h } \big ) \mathrm { d } s .\tag{4.14}
$$

The gradient in (4.13) is used only for $t > 0$ . The following lemma proves that these time integrals define $L ^ { 1 }$ fields and identifies their weak derivatives.

Lemma 4.4 (Exact stationary identities). Under $h L _ { \lambda } < 1$ , the fields in (4.11)–(4.14) are well defined in $L ^ { 1 } ( \mathbb { R } ^ { d } )$ and satisfy

$$
\begin{array} { r } { \widehat { \pi } _ { \lambda , h } - \bar { \rho } = \mathrm { d i v } H , } \\ { \nabla U _ { \lambda } \widehat { \pi } _ { \lambda , h } - J = \mathrm { d i v } E , } \\ { \Delta \bar { \rho } + \mathrm { d i v } J = 0 . \qquad } \end{array}\tag{4.15}
$$

Consequently,

$$
{ \mathcal { L } } _ { \lambda } ^ { * } ( { \bar { \rho } } - \pi _ { \lambda } ) = { \mathrm { d i v } } { \mathrm { d i v } } ( E - \nabla U _ { \lambda } \otimes H ) + { \mathrm { d i v } } ( ( \nabla ^ { 2 } U _ { \lambda } ) H ) .\tag{4.16}
$$

All these identities hold in the sense of distributions. In particular, $\Delta \bar { \rho }$ denotes the distributional Laplacian; it has the $L ^ { 1 }$ representative $( \widehat { \pi } _ { \lambda , h } - \rho _ { 0 } ) / h$

Proof. Integrability of the fields. Since $\nabla U _ { \lambda }$ has at most linear growth and $\widehat { \pi } _ { \lambda , h } \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ 2

$$
\int \bigl ( \| \nabla U _ { \lambda } \| + \| \nabla U _ { \lambda } \| ^ { 2 } \bigr ) \widehat { \pi } _ { \lambda , h } \mathrm { d } x < \infty .\tag{4.17}
$$

The pushforward and heat-kernel formulas give jointly measurable versions of the integrands in (4.11)–(4.14). Since $\rho _ { 0 }$ is a probability density, so are $\rho _ { t }$ and ${ \bar { \rho } } .$ By Proposition 2.1 and (2.13),

$$
\begin{array} { l } { \displaystyle \| J \| _ { L ^ { 1 } } \leq \int \| \nabla U _ { \lambda } \| \widehat { \pi } _ { \lambda , h } \mathrm { d } x , } \\ { \displaystyle \| E \| _ { L ^ { 1 } } \leq \frac { h } { 2 } \int \| \nabla U _ { \lambda } \| ^ { 2 } \widehat { \pi } _ { \lambda , h } \mathrm { d } x , } \\ { \displaystyle \| H \| _ { L ^ { 1 } } \leq \frac { 1 } { h } \int _ { 0 } ^ { h } t \| \nabla \rho _ { t } \| _ { L ^ { 1 } } \mathrm { d } t \leq \frac { C _ { d } } { h } \int _ { 0 } ^ { h } \sqrt { t } \mathrm { d } t = \frac { 2 C _ { d } } { 3 } \sqrt { h } . } \end{array}
$$

All $L ^ { 1 }$ norms here are with respect to Lebesgue measure. These estimates justify the time integrals and their pairing with bounded test functions.

The drift balance. We first prove $\rho _ { 0 } - { \widehat \pi } _ { \lambda , h } = h$ div J. Fix $\phi \in \ C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { d } )$ . For $\varepsilon \neq 0$ with $s , s + \varepsilon \in [ 0 , h ]$ , the mean value theorem gives

$$
\left. \frac { \phi ( T _ { s + \varepsilon } x ) - \phi ( T _ { s } x ) } { \varepsilon } \right. \widehat { \pi } _ { \lambda , h } ( x ) \leq \| \nabla \phi \| _ { \infty } \| \nabla U _ { \lambda } ( x ) \| \widehat { \pi } _ { \lambda , h } ( x ) .
$$

The right-hand side is integrable by (4.17) and independent of $s , \varepsilon .$ . Dominated convergence therefore permits diferentiation under the integral in (2.10), giving

$$
\begin{array} { l } { \displaystyle \frac { \mathrm { d } } { \mathrm { d } s } \int \phi ( y ) ( T _ { s } ) _ { \# } \widehat { \pi } _ { \lambda , h } ( y ) \mathrm { d } y = - \int \left. \nabla \phi ( T _ { s } x ) , \nabla U _ { \lambda } ( x ) \right. \widehat { \pi } _ { \lambda , h } ( x ) \mathrm { d } x } \\ { \displaystyle \qquad = - \int \left. \nabla \phi , ( T _ { s } ) _ { \# } ( \nabla U _ { \lambda } \widehat { \pi } _ { \lambda , h } ) \right. \mathrm { d } x . } \end{array}\tag{4.18}
$$

The same domination gives continuity of this derivative in s. Integrating (4.18) over [0, h] yields

$$
\int \phi ( \rho _ { 0 } - { \widehat { \pi } } _ { \lambda , h } ) \mathrm { d } x = - h \int \langle \nabla \phi , J \rangle \mathrm { d } x = h \langle \operatorname { d i v } J , \phi \rangle .
$$

Thus

$\rho _ { 0 } - \widehat { \pi } _ { \lambda , h } = ( T _ { h } ) _ { \# } \widehat { \pi } _ { \lambda , h } - \widehat { \pi } _ { \lambda , h } = h$ div J in the sense of distributions.

(4.19)

The heat balance and the endpoint diference. To prove the third identity in (4.15), we compute the distributional Laplacian through its definition. For $\phi \in \ C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { d } )$ , Fubini’s theorem applies because

$$
\int _ { 0 } ^ { h } \int \rho _ { t } | \Delta \phi | \mathrm { d } x \mathrm { d } t \leq h \left\| \Delta \phi \right\| _ { \infty } < \infty .
$$

The definition of the distributional Laplacian and (2.15) therefore give

$$
\begin{array} { l } { \displaystyle \langle \Delta \bar { \rho } , \phi \rangle : = \int \bar { \rho } \Delta \phi \mathrm { d } x = \frac { 1 } { h } \int _ { 0 } ^ { h } \int \rho _ { t } \Delta \phi \mathrm { d } x \mathrm { d } t } \\ { \displaystyle \qquad = \frac { 1 } { h } \int \phi ( \rho _ { h } - \rho _ { 0 } ) \mathrm { d } x } \\ { \displaystyle \qquad = \frac { 1 } { h } \int \phi ( \widehat { \pi } _ { \lambda , h } - \rho _ { 0 } ) \mathrm { d } x = - \langle \mathrm { d i v } J , \phi \rangle . } \end{array}\tag{4.20}
$$

Here (2.15) is used with $F = \rho _ { 0 } , a = 0$ , and $b = h$ , and the last equality is (4.19). This proves $\Delta \bar { \rho } + \mathrm { d i v } \ : J = 0$ and the claimed $L ^ { 1 }$ representative of $\Delta \bar { \rho } .$ . No classical Laplacian of $\bar { \rho }$ was assumed. We next prove $\widehat { \pi } _ { \lambda , h } - \bar { \rho } = \mathrm { d i v } H$ . The integrability already established for $t \nabla \rho _ { t }$ permits Fubini’s theorem. Spatial integration by parts for $t > 0$ , followed by (2.14), gives

$$
\begin{array} { l } { { \displaystyle \left. \mathrm { d i v } { \cal H } , \phi \right. = - \frac 1 h \int _ { 0 } ^ { h } t \int \left. \nabla \phi , \nabla \rho _ { t } \right. \mathrm { d } x \mathrm { d } t } \ ~ } \\ { { \displaystyle ~ = \frac 1 h \int _ { 0 } ^ { h } t \int \rho _ { t } \Delta \phi \mathrm { d } x \mathrm { d } t } } \\ { { \displaystyle ~ = \frac 1 h \int _ { 0 } ^ { h } t \frac { \mathrm { d } } { \mathrm { d } t } \left( \int \phi \rho _ { t } \mathrm { d } x \right) \mathrm { d } t } } \\ { { \displaystyle ~ = \frac 1 h \left[ t \int \phi \rho _ { t } \mathrm { d } x \right] _ { 0 } ^ { h } - \frac 1 h \int _ { 0 } ^ { h } \int \phi \rho _ { t } \mathrm { d } x \mathrm { d } t } } \\ { { \displaystyle ~ = \int \phi ( \widehat \pi _ { \lambda h } - \bar { \rho } ) \mathrm { d } x } . } \end{array}\tag{4.21}
$$

The time integration by parts is valid because $t \mapsto \int \phi \rho _ { t }$ dx is absolutely continuous by (2.15). Its boundary term at zero vanishes since $\begin{array} { r } { \left| t \int \phi \rho _ { t } \mathrm { d } x \right| \leq t \left\| \phi \right\| _ { \infty } } \end{array}$

The matrix flux. We prove the second identity in (4.15) componentwise. For each fixed i, the diference quotients for the pairing in (2.10) with density $( \partial _ { i } U _ { \lambda } ) \widehat { \pi } _ { \lambda , h }$ are dominated by the integrable function $\| \nabla \phi \| _ { \infty } \| \nabla U _ { \lambda } \| ^ { 2 } \widehat { \pi } _ { \lambda , h }$ . We may therefore diferentiate under the integral. Keeping the initial weight $\partial _ { i } U _ { \lambda } ( x )$ fixed gives

$$
\begin{array} { l } { \displaystyle \frac { \mathrm { d } } { \mathrm { d } s } \int \phi ( y ) \big [ ( T _ { s } ) _ { \# } ( \nabla U _ { \lambda } \widehat \pi _ { \lambda , h } ) \big ] _ { i } ( y ) \mathrm { d } y } \\ { = - \displaystyle \sum _ { j } \int \partial _ { j } \phi ( T _ { s } x ) \partial _ { i } U _ { \lambda } ( x ) \partial _ { j } U _ { \lambda } ( x ) \widehat \pi _ { \lambda , h } ( x ) \mathrm { d } x } \\ { = - \displaystyle \sum _ { j } \int \partial _ { j } \phi ( y ) \big [ ( T _ { s } ) _ { \# } \big ( ( \nabla U _ { \lambda } \otimes \nabla U _ { \lambda } ) \widehat \pi _ { \lambda , h } \big ) \big ] _ { i j } ( y ) \mathrm { d } y . } \end{array}\tag{4.22}
$$

The same second-moment bound justifies Fubini’s theorem for the two time integrals below, since

$$
\int _ { 0 } ^ { h } \int _ { 0 } ^ { s } \int \left\| \nabla U _ { \lambda } \right\| ^ { 2 } \widehat { \pi } _ { \lambda , h } \mathrm { d } x \mathrm { d } u \mathrm { d } s = \frac { h ^ { 2 } } { 2 } \int \left\| \nabla U _ { \lambda } \right\| ^ { 2 } \widehat { \pi } _ { \lambda , h } \mathrm { d } x < \infty .
$$

Integrating (4.22) first from 0 to s and then averaging over $s \in [ 0 , h ]$ , we obtain

$$
\begin{array} { r l } & { \displaystyle \int \phi \big ( ( \partial _ { i } U _ { \lambda } ) \widehat { \pi } _ { \lambda , h } - J _ { i } \big ) \mathrm { d } x } \\ & { \displaystyle \quad = - \frac { 1 } { h } \int _ { 0 } ^ { h } \int _ { 0 } ^ { s } \frac { \mathrm { d } } { \mathrm { d } u } \left( \int \phi ( y ) \big [ ( T _ { u } ) _ { \# } ( \nabla U _ { \lambda } \widehat { \pi } _ { \lambda , h } ) \big ] _ { i } ( y ) \mathrm { d } y \right) \mathrm { d } u \mathrm { d } s } \\ & { \displaystyle \quad = \frac { 1 } { h } \sum _ { j } \int _ { 0 } ^ { h } ( h - u ) \int \partial _ { j } \phi ( y ) \big [ ( T _ { u } ) _ { \# } ( ( \nabla U _ { \lambda } \otimes \nabla U _ { \lambda } ) \widehat { \pi } _ { \lambda , h } ) \big ] _ { i j } ( y ) \mathrm { d } y \mathrm { d } u } \\ & { \displaystyle \quad = - \sum _ { j } \int \partial _ { j } \phi E _ { i j } \mathrm { d } x = \langle ( \mathrm { d i v } E ) _ { i } , \phi \rangle . } \end{array}
$$

This proves $\nabla U _ { \lambda } { \widehat { \pi } } _ { \lambda , h } - J = \operatorname { d i v } E$ and completes (4.15).

The residual equation. We use the product rule

$$
\operatorname { d i v } ( \nabla U _ { \lambda } \otimes H ) = \nabla U _ { \lambda } \left( { \widehat { \pi } } _ { \lambda , h } - { \bar { \rho } } \right) + ( \nabla ^ { 2 } U _ { \lambda } ) H .\tag{4.23}
$$

To justify it, test div $H = \widehat { \pi } _ { \lambda , h } - \bar { \rho }$ against smooth approximations of $( \partial _ { i } U _ { \lambda } ) \phi$ . Mollifications of $\partial _ { i } U _ { \lambda }$ converge locally uniformly; their gradients are uniformly bounded and converge almost everywhere to its weak gradient. Since H and $\widehat { \pi } _ { \lambda , h } - \bar { \rho }$ are locally integrable, dominated convergence gives

$$
- \int ( \partial _ { i } U _ { \lambda } ) \langle H , \nabla \phi \rangle \mathrm { d } x = \int ( \partial _ { i } U _ { \lambda } ) ( \widehat { \pi } _ { \lambda , h } - \bar { \rho } ) \phi \mathrm { d } x + \int \phi \sum _ { j } ( \partial _ { j } \partial _ { i } U _ { \lambda } ) H _ { j } \mathrm { d } x ,
$$

which is the ith component of (4.23). Now $\mathcal { L } _ { \lambda } ^ { * } \pi _ { \lambda } = 0$ , (4.15), and (4.23) give

$$
\begin{array} { r l } & { \mathcal { L } _ { \lambda } ^ { * } ( \bar { \rho } - \pi _ { \lambda } ) = \Delta \bar { \rho } + \mathrm { d i v } ( \nabla U _ { \lambda } \bar { \rho } ) } \\ & { \quad \quad \quad \quad = \mathrm { d i v } ( \nabla U _ { \lambda } \bar { \rho } - J ) } \\ & { \quad \quad \quad = \mathrm { d i v } \big ( \mathrm { d i v } E - \nabla U _ { \lambda } ( \widehat { \pi } _ { \lambda , h } - \bar { \rho } ) \big ) } \\ & { \quad \quad \quad = \mathrm { d i v } \big ( \mathrm { d i v } E - \mathrm { d i v } ( \nabla U _ { \lambda } \otimes H ) + ( \nabla ^ { 2 } U _ { \lambda } ) H \big ) } \\ & { \quad \quad \quad = \mathrm { d i v } \mathrm { d i v } ( E - \nabla U _ { \lambda } \otimes H ) + \mathrm { d i v } ( ( \nabla ^ { 2 } U _ { \lambda } ) H ) . } \end{array}
$$

All derivatives in this calculation are distributional. Equivalently, the resulting identity means that

$$
\int ( \bar { \rho } - \pi _ { \lambda } ) \mathcal { L } _ { \lambda } \phi \mathrm { d } x = \int \left. E - \nabla U _ { \lambda } \otimes H , \nabla ^ { 2 } \phi \right. \mathrm { d } x - \int \left. ( \nabla ^ { 2 } U _ { \lambda } ) H , \nabla \phi \right. \mathrm { d } x
$$

for every $\phi \in C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { d } )$ . This proves (4.16).

For the quantitative flux bounds below and in Section 5, choose the moment order

$$
r : = 4 + 2 \log ( 1 + L _ { \lambda } / m ) .\tag{4.24}
$$

Lemma 4.5 (Stationary flux bounds). Under (3.11), the fields in (4.11)–(4.14) satisfy

$$
\| H / \pi _ { \lambda } \| _ { L ^ { r } ( \pi _ { \lambda } ) } \leq C h r R \log \frac { e } { m h } ,\tag{4.25}
$$

$$
\| E / \pi _ { \lambda } \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } \le C h R ^ { 2 } ,\tag{4.26}
$$

$$
\| ( E - \nabla U _ { \lambda } \otimes H ) / \pi _ { \lambda } \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } \leq C h r R ^ { 2 } \log \frac { e } { m h } .\tag{4.27}
$$

The proof is given in Section 5.

## 4.3 A residual-to-transport estimate

Proposition 4.6 (Second-order residual estimate). Let $\nu , \rho \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ be densities with $\nu / \pi _ { \lambda } , \rho / \pi _ { \lambda } \in$ $L ^ { 2 } ( \pi _ { \lambda } )$ . Suppose a vector density H and a matrix density E satisfy

$$
\nu - \rho = \operatorname { d i v } H ,\tag{4.28}
$$

$$
{ \mathcal L } _ { \lambda } ^ { * } ( \rho - \pi _ { \lambda } ) = \mathrm { d i v } \mathrm { d i v } ( E - \nabla U _ { \lambda } \otimes H ) + \mathrm { d i v } ( ( \nabla ^ { 2 } U _ { \lambda } ) H )\tag{4.29}
$$

in the sense of distributions. If the norms on the right below are finite, then

$$
\begin{array} { l } { \sqrt { m } W _ { 2 } ( \nu , \pi _ { \lambda } ) \leq 2 \sqrt { m } \| H / \pi _ { \lambda } \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } + 2 \| ( E - \nabla U _ { \lambda } \otimes H ) / \pi _ { \lambda } \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } } \\ { \displaystyle \qquad +  2 ( \int \frac { H ^ { \top } ( \nabla ^ { 2 } U _ { \lambda } ) H } { \pi _ { \lambda } } \mathrm { d } x ) ^ { 1 / 2 } . } \end{array}\tag{4.30}
$$

For clarity, (4.28) and (4.29) mean that, for every $v \in C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { d } )$

$$
\begin{array} { l } { \displaystyle \int v ( \boldsymbol \nu - \boldsymbol \rho ) \mathrm { d } x = - \int \left. \nabla v , H \right. \mathrm { d } x , } \\ { \displaystyle \int ( \boldsymbol \rho - \boldsymbol \pi _ { \lambda } ) \mathcal { L } _ { \lambda } v \mathrm { d } x = \int \left. E - \nabla U _ { \lambda } \otimes H , \nabla ^ { 2 } v \right. \mathrm { d } x - \int \left. ( \nabla ^ { 2 } U _ { \lambda } ) H , \nabla v \right. \mathrm { d } x . } \end{array}
$$

All derivatives in these integral identities act on the test function. In particular, no classical derivatives of E or of the weak Hessian $\nabla ^ { 2 } U _ { \lambda }$ are required.

Proof of Proposition 4.6. Fix $\phi \in C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { d } )$ and let $\psi$ be its Poisson solution from Lemma 2.2. We first prove the pairing identity

$$
\begin{array} { l } { \displaystyle \int \phi ( \rho - \pi _ { \lambda } ) \mathrm { d } x = \int ( \rho - \pi _ { \lambda } ) \mathcal { L } _ { \lambda } \psi \mathrm { d } x } \\ { \displaystyle \qquad = \int \left. E - \nabla U _ { \lambda } \otimes H , \nabla ^ { 2 } \psi \right. \mathrm { d } x - \int \left. ( \nabla ^ { 2 } U _ { \lambda } ) H , \nabla \psi \right. \mathrm { d } x . } \end{array}\tag{4.31}
$$

The residual equation (4.29) is initially available only for compactly supported smooth tests. We therefore take the sequence $\psi _ { n } \in C _ { c } ^ { \infty } ( \mathbb { R } ^ { d } )$ from (2.21). For every $n ,$ the weak formulation gives

$$
\int ( \rho - \pi _ { \lambda } ) \mathcal { L } _ { \lambda } \psi _ { n } \mathrm { d } x = \int \left. E - \nabla U _ { \lambda } \otimes H , \nabla ^ { 2 } \psi _ { n } \right. \mathrm { d } x - \int \left. ( \nabla ^ { 2 } U _ { \lambda } ) H , \nabla \psi _ { n } \right. \mathrm { d } x .\tag{4.32}
$$

We now justify passage to the limit in each integral of (4.32).

For the curvature integral, positivity of $\nabla ^ { 2 } U _ { \lambda }$ and Cauchy–Schwarz give, for every $v \in H ^ { 1 } ( \pi _ { \lambda } )$

$$
\left| \int \left. ( \nabla ^ { 2 } U _ { \lambda } ) H , \nabla v \right. \mathrm { d } x \right| \leq \left( \int { \frac { H ^ { \top } ( \nabla ^ { 2 } U _ { \lambda } ) H } { \pi _ { \lambda } } } \mathrm { d } x \right) ^ { 1 / 2 } \left( \int \nabla v ^ { \top } ( \nabla ^ { 2 } U _ { \lambda } ) \nabla v \mathrm { d } \pi _ { \lambda } \right) ^ { 1 / 2 } .\tag{4.33}
$$

Using Cauchy–Schwarz for the density and matrix integrals, and (4.33) with $v = \psi _ { n } - \psi$ for the curvature integral, we obtain

$$
\left| \int ( \rho - \pi _ { \lambda } ) ( \mathcal { L } _ { \lambda } \psi _ { n } - \mathcal { L } _ { \lambda } \psi ) \mathrm { d } x \right| \leq \| \rho / \pi _ { \lambda } - 1 \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } \| \mathcal { L } _ { \lambda } \psi _ { n } - \mathcal { L } _ { \lambda } \psi \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } \longrightarrow 0 ,
$$

$$
\mathopen { } \mathclose \bgroup \left| \int \left. E - \nabla U _ { \lambda } \otimes H , \nabla ^ { 2 } \psi _ { n } - \nabla ^ { 2 } \psi \right. \mathrm { d } x \aftergroup \egroup \right| \leq \| ( E - \nabla U _ { \lambda } \otimes H ) / \pi _ { \lambda } \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } \mathopen { } \mathclose \bgroup \left\| \nabla ^ { 2 } \psi _ { n } - \nabla ^ { 2 } \psi \aftergroup \egroup \right\| _ { L ^ { 2 } ( \pi _ { \lambda } ) } \longrightarrow 0 ,
$$

$$
\left| \int \left. ( \nabla ^ { 2 } U _ { \lambda } ) H , \nabla \psi _ { n } - \nabla \psi \right. \mathrm { d } x \right| \leq { \sqrt { L _ { \lambda } } } \left( \int { \frac { H ^ { \top } ( \nabla ^ { 2 } U _ { \lambda } ) H } { \pi _ { \lambda } } } \mathrm { d } x \right) ^ { 1 / 2 } \| \nabla \psi _ { n } - \nabla \psi \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } \longrightarrow 0 .\tag{4.34}
$$

The coeficient norms are finite by the hypotheses of the proposition. The three approximation errors tend to zero by (2.21); the last estimate also uses $\nabla ^ { 2 } U _ { \lambda } \preceq L _ { \lambda } I$ . Thus (4.34) allows us to pass to the limit in (4.32), proving (4.31).

We next estimate the right-hand side of (4.31). For the matrix integral, Cauchy–Schwarz uses the Frobenius inner product; for the curvature integral, apply (4.33) with $v = \psi$ . Consequently,

$$
\begin{array} { r l } & { \displaystyle \left. \int \phi ( \rho - \pi _ { \lambda } ) \mathrm { d } x \right. \leq \| ( E - \nabla U _ { \lambda } \otimes H ) / \pi _ { \lambda } \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } \left( \int \| \nabla ^ { 2 } \psi \| _ { F } ^ { 2 } \mathrm { d } \pi _ { \lambda } \right) ^ { 1 / 2 } } \\ & { \quad \quad \quad \quad \quad \quad \quad + \left( \displaystyle \int \frac { H ^ { \top } ( \nabla ^ { 2 } U _ { \lambda } ) H } { \pi _ { \lambda } } \mathrm { d } x \right) ^ { 1 / 2 } \left( \int \nabla \psi ^ { \top } ( \nabla ^ { 2 } U _ { \lambda } ) \nabla \psi \mathrm { d } \pi _ { \lambda } \right) ^ { 1 / 2 } . } \end{array}\tag{4.35}
$$

From (2.20),

$$
\begin{array} { r } { \left( \displaystyle \int \left\| \nabla ^ { 2 } \psi \right\| _ { F } ^ { 2 } \mathrm { d } \pi _ { \lambda } \right) ^ { 1 / 2 } \leq \frac { 1 } { \sqrt { m } } \left\| \nabla \phi \right\| _ { L ^ { 2 } \left( \pi _ { \lambda } \right) } , } \\ { \left( \displaystyle \int \nabla \psi ^ { \top } ( \nabla ^ { 2 } U _ { \lambda } ) \nabla \psi \mathrm { d } \pi _ { \lambda } \right) ^ { 1 / 2 } \leq \frac { 1 } { \sqrt { m } } \left\| \nabla \phi \right\| _ { L ^ { 2 } \left( \pi _ { \lambda } \right) } . } \end{array}\tag{4.36}
$$

Substituting (4.36) into (4.35) yields

$$
\left| \int \phi ( \rho - \pi _ { \lambda } ) \mathrm { d } x \right| \leq \frac { \| \nabla \phi \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } } { \sqrt { m } } \left[ \| ( E - \nabla U _ { \lambda } \otimes H ) / \pi _ { \lambda } \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } + \left( \int \frac { H ^ { \top } ( \nabla ^ { 2 } U _ { \lambda } ) H } { \pi _ { \lambda } } \mathrm { d } x \right) ^ { 1 / 2 } \right] .\tag{4.37}
$$

The first residual identity (4.28), tested directly with $\phi ,$ gives

$$
\left| \int \phi ( \nu - \rho ) \mathrm { d } x \right| = \left| \int \langle \nabla \phi , H \rangle \mathrm { d } x \right| \leq \| H / \pi _ { \lambda } \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } \| \nabla \phi \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } .\tag{4.38}
$$

Since $\nu - \pi _ { \lambda } = ( \nu - \rho ) + ( \rho - \pi _ { \lambda } )$ , the triangle inequality, (4.38), and (4.37) imply

$$
\begin{array} { r l r } {  {  \int \phi ( \boldsymbol { \nu } - \boldsymbol { \pi } _ { \lambda } ) \mathrm { d } \boldsymbol { x }  \leq  \int \phi ( \boldsymbol { \nu } - \boldsymbol { \rho } ) \mathrm { d } \boldsymbol { x }  +  \int \phi ( \boldsymbol { \rho } - \boldsymbol { \pi } _ { \lambda } ) \mathrm { d } \boldsymbol { x }  } } \\ & { } & { \leq \| \nabla \phi \| _ { L ^ { 2 } ( \boldsymbol { \pi } _ { \lambda } ) } [ \| H / \pi _ { \lambda } \| _ { L ^ { 2 } ( \boldsymbol { \pi } _ { \lambda } ) } + \frac { 1 } { \sqrt { m } } \| ( E - \nabla U _ { \lambda } \otimes H ) / \pi _ { \lambda } \| _ { L ^ { 2 } ( \boldsymbol { \pi } _ { \lambda } ) }  } \\ & { } & { \quad  + \frac { 1 } { \sqrt { m } } ( \int \frac { H ^ { \top } ( \nabla ^ { 2 } U _ { \lambda } ) H } { \pi _ { \lambda } } \mathrm { d } x ) ^ { 1 / 2 } ] . } \end{array}\tag{4.39}
$$

Taking the supremum in (4.39) over the test functions in (2.22) gives

$$
\begin{array} { l } { \displaystyle \| \nu - \pi _ { \lambda } \| _ { \dot { H } ^ { - 1 } ( \pi _ { \lambda } ) } = \operatorname* { s u p } _ { \phi \in C _ { c } ^ { \infty } ( \mathbb { R } ^ { d } ) } \left| \int \phi ( \nu - \pi _ { \lambda } ) \mathrm { d } x \right| } \\ { \displaystyle \| \nabla \phi \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } \le 1 } \\ { \displaystyle \le \| H / \pi _ { \lambda } \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } + \frac { 1 } { \sqrt { m } } \| ( E - \nabla U _ { \lambda } \otimes H ) / \pi _ { \lambda } \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } } \\ { \displaystyle \qquad + \frac { 1 } { \sqrt { m } } \left( \int \frac { H ^ { \top } ( \nabla ^ { 2 } U _ { \lambda } ) H } { \pi _ { \lambda } } \mathrm { d } x \right) ^ { 1 / 2 } . } \end{array}\tag{4.40}
$$

Finally, $\nu / \pi _ { \lambda } \in L ^ { 2 } ( \pi _ { \lambda } )$ by assumption, so (2.25) applies. Using that comparison and then (4.40), we conclude that

$$
\begin{array} { r l } & { \sqrt { m } W _ { 2 } ( \nu , \pi _ { \lambda } ) \leq 2 \sqrt { m } \ \| \nu - \pi _ { \lambda } \| _ { \dot { H } ^ { - 1 } ( \pi _ { \lambda } ) } } \\ & { \qquad \leq 2 \sqrt { m } \ \| H / \pi _ { \lambda } \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } + 2 \left\| ( E - \nabla U _ { \lambda } \otimes H ) / \pi _ { \lambda } \right\| _ { L ^ { 2 } ( \pi _ { \lambda } ) } } \\ & { \qquad + \ 2 \left( \displaystyle \int \frac { H ^ { \top } ( \nabla ^ { 2 } U _ { \lambda } ) H } { \pi _ { \lambda } } \mathrm { d } x \right) ^ { 1 / 2 } . } \end{array}
$$

This is (4.30).

## 4.4 Proof of the fixed-smoothing theorem

Proof of Theorem 3.4. Assume (3.11). Since $\tau _ { f } \ge m d \ge m$ and $L _ { \lambda } = L _ { f } + \lambda ^ { - 1 }$ , we have

$$
h ( L _ { \lambda } + R ^ { 2 } ) \leq h \left[ \frac { L _ { f } \tau _ { f } } { m } + R ^ { 2 } + \frac { 1 } { \lambda } \right] \leq \frac { c } { \ell _ { h , \lambda } ^ { 2 } } \leq c .\tag{4.41}
$$

Taking the universal constant in (3.11) suficiently small gives $h L _ { \lambda } \leq 1 / 2$ . By Proposition 3.3, $Q _ { \lambda , h }$ therefore has a unique invariant law $\widehat { \pi } _ { \lambda , h } \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ with a strictly positive smooth density.

Let r be as in (4.24), and take ${ \bar { \rho } } , H$ , E from (4.12)–(4.14). We will apply Proposition 4.6 with $\nu = \widehat { \pi } _ { \lambda , h }$ and $\rho = \bar { \rho } .$ . We first verify its density assumptions and then estimate the three terms in (4.30).

The density assumptions. (3.11) gives

$$
6 4 h { \frac { L _ { f } \tau _ { f } } { m } } + 8 h { \frac { L _ { f } R } { \sqrt { m } } } + 6 4 h { \frac { G R } { m \lambda } } \leq 6 4 h \left[ { \frac { L _ { f } \tau _ { f } } { m } } + { \frac { L _ { f } R } { \sqrt { m } } } + { \frac { G R } { m \lambda } } \right] \leq { \frac { 6 4 c } { \ell _ { h , \lambda } ^ { 2 } } } .\tag{4.42}
$$

Together with (4.41), these bounds imply the conditions of Lemma 4.2 with $\alpha = 8 $ , after decreasing the universal constant in (3.11). Evaluating (4.6) at $t = 0$ yields

$$
\| \widehat { \pi } _ { \lambda , h } / \pi _ { \lambda } \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } \leq \| \widehat { \pi } _ { \lambda , h } / \pi _ { \lambda } \| _ { L ^ { 8 } ( \pi _ { \lambda } ) } \leq e ^ { 1 / 4 } .\tag{4.43}
$$

For $0 < t \leq h$ , apply (4.7) with output order $s = 2$ and (4.10) with output order $s = 4$ . Their time restrictions hold because

$$
4 t R ^ { 2 } \le 4 h R ^ { 2 } \le 4 c , \qquad 4 h ( L _ { \lambda } + R ^ { 2 } ) \le 4 c
$$

by (4.41). Thus, after the same choice of a suficiently small constant,

$$
\begin{array} { r } { \| \rho _ { t } / \pi _ { \lambda } \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } = \| \mathsf { H } _ { t } ( ( T _ { h } ) _ { \# } \widehat { \pi } _ { \lambda , h } ) / \pi _ { \lambda } \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } } \\ { \leq C \| ( T _ { h } ) _ { \# } \widehat { \pi } _ { \lambda , h } / \pi _ { \lambda } \| _ { L ^ { 4 } ( \pi _ { \lambda } ) } } \\ { \leq C \| \widehat { \pi } _ { \lambda , h } / \pi _ { \lambda } \| _ { L ^ { 8 } ( \pi _ { \lambda } ) } \leq C . } \end{array}\tag{4.44}
$$

Jensen’s inequality for the time average, followed by Tonelli’s theorem and (4.44), gives

$$
\begin{array} { r } { \| \bar { \rho } / \pi _ { \lambda } \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } ^ { 2 } = \displaystyle \int \left( \frac { 1 } { h } \int _ { 0 } ^ { h } \frac { \rho _ { t } ( x ) } { \pi _ { \lambda } ( x ) } \mathrm { d } t \right) ^ { 2 } \mathrm { d } \pi _ { \lambda } ( x ) } \\ { \leq \displaystyle \frac { 1 } { h } \int _ { 0 } ^ { h } \| \rho _ { t } / \pi _ { \lambda } \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } ^ { 2 } \mathrm { d } t < \infty . } \end{array}\tag{4.45}
$$

The heat-kernel representation also gives

$$
\int \| y \| ^ { 2 } { \bar { \rho } } ( y ) \mathrm { d } y = \int \| T _ { h } ( x ) \| ^ { 2 } { \widehat { \pi } } _ { \lambda , h } ( x ) \mathrm { d } x + d h < \infty .
$$

Indeed, the Gaussian increment at time t contributes 2td to the second moment, and $T _ { h }$ has at most linear growth. Hence $\bar { \rho } \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ . Equations (4.43) and (4.45) verify the two $L ^ { 2 }$ density-ratio assumptions of Proposition 4.6.

The curvature term. The flux bounds (4.25) and (4.27) are supplied by Lemma 4.5. It remains to estimate the integral involving $\nabla ^ { 2 } U _ { \lambda }$ in (4.30). Since $\nabla ^ { 2 } U _ { \lambda }$ is positive semidefinite, its quadratic form is bounded by its trace times the squared Euclidean norm. Applying this pointwise and then using H¨older with conjugate exponents $r / 2$ and $r / ( r - 2 )$ gives

$$
\begin{array} { l } { \displaystyle \int \frac { H ^ { \top } ( \nabla ^ { 2 } U _ { \lambda } ) H } { \pi _ { \lambda } } \mathrm { d } x = \int \left( \frac { H } { \pi _ { \lambda } } \right) ^ { \top } ( \nabla ^ { 2 } U _ { \lambda } ) \left( \frac { H } { \pi _ { \lambda } } \right) \mathrm { d } \pi _ { \lambda } } \\ { \displaystyle \qquad \leq \int \| H / \pi _ { \lambda } \| ^ { 2 } \mathrm { t r } ( \nabla ^ { 2 } U _ { \lambda } ) \mathrm { d } \pi _ { \lambda } } \\ { \displaystyle \qquad \leq \| H / \pi _ { \lambda } \| _ { L ^ { r } ( \pi _ { \lambda } ) } ^ { 2 } \left( \int [ \mathrm { t r } ( \nabla ^ { 2 } U _ { \lambda } ) ] ^ { r / ( r - 2 ) } \mathrm { d } \pi _ { \lambda } \right) ^ { ( r - 2 ) / r } . } \end{array}\tag{4.46}
$$

To estimate the last integral, use $0 \leq \mathrm { t r } ( \nabla ^ { 2 } U _ { \lambda } ) \leq d L _ { \lambda }$ almost everywhere and (4.1). Since $r / ( r - 2 ) = 1 + 2 / ( r - 2 )$ 2

$$
\begin{array} { r l } {  { \int [ \mathrm { t r } ( \nabla ^ { 2 } U _ { \lambda } ) ] ^ { r / ( r - 2 ) } \mathrm { d } \pi _ { \lambda } = \int \mathrm { t r } ( \nabla ^ { 2 } U _ { \lambda } ) [ \mathrm { t r } ( \nabla ^ { 2 } U _ { \lambda } ) ] ^ { 2 / ( r - 2 ) } \mathrm { d } \pi _ { \lambda } } } \\ & { \le ( d L _ { \lambda } ) ^ { 2 / ( r - 2 ) } \int \mathrm { t r } ( \nabla ^ { 2 } U _ { \lambda } ) \mathrm { d } \pi _ { \lambda } } \\ & { \le ( d L _ { \lambda } ) ^ { 2 / ( r - 2 ) } R ^ { 2 } . } \end{array}\tag{4.47}
$$

Moreover, $R ^ { 2 } \geq$ md and the choice of r in (4.24) imply

$$
\left( \frac { d L _ { \lambda } } { R ^ { 2 } } \right) ^ { 1 / r } \leq \left( \frac { L _ { \lambda } } { m } \right) ^ { 1 / r } = \exp \left\{ \frac { \log ( L _ { \lambda } / m ) } { 4 + 2 \log ( 1 + L _ { \lambda } / m ) } \right\} \leq \sqrt { e } .\tag{4.48}
$$

Substituting (4.47) into (4.46), taking square roots, and using (4.48), we obtain

$$
\begin{array} { r l r } & { } & { \left( \displaystyle \int \frac { H ^ { \top } ( \nabla ^ { 2 } U _ { \lambda } ) H } { \pi _ { \lambda } } \mathrm { d } x \right) ^ { 1 / 2 } \leq R \left( \frac { d L _ { \lambda } } { R ^ { 2 } } \right) ^ { 1 / r } \| H / \pi _ { \lambda } \| _ { L ^ { r } ( \pi _ { \lambda } ) } } \\ & { } & { \leq \sqrt { e } R \| H / \pi _ { \lambda } \| _ { L ^ { r } ( \pi _ { \lambda } ) } . } \end{array}\tag{4.49}
$$

In particular, this integral is finite by (4.25).

Application of the residual estimate. The identities (4.15) and (4.16) are precisely the weak identities required in Proposition 4.6 for $\nu = \widehat { \pi } _ { \lambda , h }$ and $\rho \ = \ { \bar { \rho } } .$ Its density assumptions have been verified above, and its weighted norms are finite by (4.25), (4.27), and (4.49). Since $r \geq 4$ $\| H / \pi _ { \lambda } \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } \le \| H / \pi _ { \lambda } \| _ { L ^ { r } ( \pi _ { \lambda } ) }$ . Applying Proposition 4.6 therefore gives

$$
\begin{array} { r l } & { \sqrt { m } W _ { 2 } ( \widehat { \pi } _ { \lambda , h } , \pi _ { \lambda } ) \leq 2 \sqrt { m } \left\| H / \pi _ { \lambda } \right\| _ { L ^ { 2 } ( \pi _ { \lambda } ) } + 2 \left\| ( E - \nabla U _ { \lambda } \otimes H ) / \pi _ { \lambda } \right\| _ { L ^ { 2 } ( \pi _ { \lambda } ) } } \\ & { \qquad + 2 \left( \displaystyle \int \frac { H ^ { \top } ( \nabla ^ { 2 } U _ { \lambda } ) H } { \pi _ { \lambda } } \mathrm { d } x \right) ^ { 1 / 2 } } \\ & { \qquad \leq 2 ( \sqrt { m } + \sqrt { e } R ) \left\| H / \pi _ { \lambda } \right\| _ { L ^ { r } ( \pi _ { \lambda } ) } + 2 \left\| ( E - \nabla U _ { \lambda } \otimes H ) / \pi _ { \lambda } \right\| _ { L ^ { 2 } ( \pi _ { \lambda } ) } } \\ & { \qquad \leq C h r R ^ { 2 } \log \displaystyle \frac { e } { m h } . } \end{array}\tag{4.50}
$$

The last inequality uses (4.25), (4.27), and ${ \sqrt { m } } \leq R$

Finally, the definitions (4.24) and (3.9) give

$$
r \leq 4 \ell _ { h , \lambda } , \log \frac { e } { m h } = 1 + \log \frac { 1 } { m h } \leq \ell _ { h , \lambda } .\tag{4.51}
$$

Combining (4.50) and (4.51) yields

$$
\sqrt { m } W _ { 2 } ( \widehat { \pi } _ { \lambda , h } , \pi _ { \lambda } ) \leq C h R ^ { 2 } \ell _ { h , \lambda } ^ { 2 } .
$$

This proves (3.12).

## 4.5 Proof of the end-to-end complexity theorem

Proof of Theorem 3.5. We allocate $\varepsilon / 4$ to the Moreau bias, $\varepsilon / 4$ to the invariant-measure bias, and $\varepsilon / 2$ to convergence from the initial law. The choice (3.13) gives the first budget by (3.18). We verify that (3.14) gives the second and that (3.15) gives the third. Substitute $\lambda = \varepsilon / G ^ { 2 }$ into the bracket in (3.11). Because $R \geq G$ and $\varepsilon \leq 1$ 2

$$
\frac { L _ { f } \tau _ { f } } { m } + \frac { L _ { f } R } { \sqrt { m } } + \frac { G R } { m \lambda } + R ^ { 2 } + \frac { 1 } { \lambda } \leq 2 \left[ \frac { L _ { f } \tau _ { f } } { m } + \frac { L _ { f } R } { \sqrt { m } } + \frac { R ^ { 2 } + G ^ { 3 } R / m } { \varepsilon } \right] .\tag{4.52}
$$

Also $R / \sqrt { m } \leq G / \sqrt { m } + \sqrt { \tau _ { f } / m }$ . The reciprocal of $m h _ { \varepsilon }$ in (3.14) is consequently bounded above by a fixed polynomial in the dimensionless ratios appearing in (3.10), multiplied by $\ell _ { \varepsilon } ^ { 2 } / c$ . It follows that

$$
\ell _ { h _ { \varepsilon } , \lambda _ { \varepsilon } } \leq C \ell _ { \varepsilon } + \log ( 1 / c ) .\tag{4.53}
$$

Choose the absolute c small enough, using $c ( 1 + \log ( 1 / c ) ) ^ { 2 } \to 0$ . Equations (4.52) and (4.53) show that (3.11) holds and that

$$
\sqrt { m } W _ { 2 } ( \widehat { \pi } _ { \lambda _ { \varepsilon } , h _ { \varepsilon } } , \pi _ { \lambda _ { \varepsilon } } ) \leq \varepsilon / 4 .\tag{4.54}
$$

Equation (3.18) gives another $\varepsilon / 4$

Synchronous coupling of (3.5) gives

$$
W _ { 2 } ( \mu _ { 0 } Q _ { \lambda , h } ^ { N } , \widehat { \pi } _ { \lambda , h } ) \leq ( 1 - m h ) ^ { N } W _ { 2 } ( \mu _ { 0 } , \widehat { \pi } _ { \lambda , h } ) .\tag{4.55}
$$

The triangle inequality and the two bias budgets imply

$$
\sqrt { m } W _ { 2 } ( \mu _ { 0 } , \widehat { \pi } _ { \lambda _ { \varepsilon } , h _ { \varepsilon } } ) \leq \sqrt { m } W _ { 2 } ( \mu _ { 0 } , \pi ) + \varepsilon / 2 .
$$

Thus (3.15) makes the contraction error at most $\varepsilon / 2$ . The triangle inequality gives

$$
\begin{array} { r l } & { \sqrt { m } W _ { 2 } ( \mu _ { 0 } Q _ { \lambda _ { \varepsilon } , h _ { \varepsilon } } ^ { N } , \pi ) \leq \sqrt { m } W _ { 2 } ( \mu _ { 0 } Q _ { \lambda _ { \varepsilon } , h _ { \varepsilon } } ^ { N } , \widehat { \pi } _ { \lambda _ { \varepsilon } , h _ { \varepsilon } } ) } \\ & { \qquad + \sqrt { m } W _ { 2 } ( \widehat { \pi } _ { \lambda _ { \varepsilon } , h _ { \varepsilon } } , \pi _ { \lambda _ { \varepsilon } } ) + \sqrt { m } W _ { 2 } ( \pi _ { \lambda _ { \varepsilon } } , \pi ) \leq \varepsilon . } \end{array}
$$

This proves (3.16); substituting (3.14) into (3.15) proves (3.17).

This completes the proof of Theorem 3.5.

## 5 Stationary Flux Estimates

This section proves the bounds on H and E in Lemma 4.5; the fields are defined in Section 4.2. We first bound the averaged drift J, then use a finite heat expansion to estimate the gradient of the invariant density $\widehat { \pi } _ { \lambda , h }$ and obtain the bound on H. The bound on E follows from the drift moment and pushforward estimates.

## 5.1 The averaged drift flux

Assume (3.11). Since $\tau _ { f } \geq m$ , this condition implies $h L _ { \lambda } \leq c ,$ , so Proposition 3.3 supplies the invariant density $\widehat { \pi } _ { \lambda , h }$

By (4.24) and $( 3 . 9 ) , r \leq 4 \ell _ { h , \lambda }$ , and hence (3.11) implies

$$
h ( 3 2 r ) ^ { 2 } \left[ \frac { L _ { f } \tau _ { f } } { m } + \frac { G R } { m \lambda } \right] + h ( 3 2 r ) \frac { L _ { f } R } { \sqrt { m } } \leq C h \ell _ { h , \lambda } ^ { 2 } \left[ \frac { L _ { f } \tau _ { f } } { m } + \frac { L _ { f } R } { \sqrt { m } } + \frac { G R } { m \lambda } \right] \leq C c .
$$

Taking the universal constant in (3.11) suficiently small therefore ensures (4.4) with $\alpha = 3 2 r$ . By (4.6) with $\alpha = 3 2 r$ and $t = 0$ ，

$$
\| \widehat { \pi } _ { \lambda , h } / \pi _ { \lambda } \| _ { L ^ { 3 2 r } ( \pi _ { \lambda } ) } \leq e ^ { 1 / 4 } .\tag{5.1}
$$

For the next estimate, we use (4.10) with output order $s = 8 r$ and times $0 \leq t \leq h$ . Its restriction is satisfied after decreasing the constant in (3.11), since $L _ { \lambda } \leq L _ { f } \tau _ { f } / m + \lambda ^ { - 1 }$ and $8 r h ( L _ { \lambda } + R ^ { 2 } ) \leq C c$ Minkowski’s inequality for the time average in (4.11), followed by (4.10) and H¨older’s inequality, gives

$$
\begin{array} { r l } {  { \| J / \pi _ { \lambda } \| _ { L ^ { 8 r } ( \pi _ { \lambda } ) } \leq \displaystyle \frac { 1 } { h } \int _ { 0 } ^ { h } \| ( T _ { u } ) _ { \# } ( \nabla U _ { \lambda } \widehat \pi _ { \lambda , h } ) / \pi _ { \lambda } \| _ { L ^ { 8 r } ( \pi _ { \lambda } ) } \mathrm { d } u } } \\ & { \leq C \| \nabla U _ { \lambda } \widehat \pi _ { \lambda , h } / \pi _ { \lambda } \| _ { L ^ { 1 6 r } ( \pi _ { \lambda } ) } } \\ & { \leq C \| \nabla U _ { \lambda } \| _ { L ^ { 3 2 r } ( \pi _ { \lambda } ) } \| \widehat \pi _ { \lambda , h } / \pi _ { \lambda } \| _ { L ^ { 3 2 r } ( \pi _ { \lambda } ) } \leq C \sqrt { r } R . } \end{array}\tag{5.2}
$$

The last step uses (4.2) at order 32r and (5.1).

## 5.2 A finite heat expansion

We next derive the finite heat expansion. By (4.19), the distributional divergence of J has the $L ^ { 1 }$ representative

$$
\operatorname { d i v } J = \frac { ( T _ { h } ) _ { \# } \widehat { \pi } _ { \lambda , h } - \widehat { \pi } _ { \lambda , h } } { h } .
$$

Since $J \in L ^ { 1 } ( \mathbb { R } ^ { d } ; \mathbb { R } ^ { d } )$ , both J and div J can therefore be convolved with the heat kernel. For every $t > 0$

$$
\mathsf { H } _ { t } ( \mathrm { d i v } J ) = \mathrm { d i v } ( \mathsf { H } _ { t } J ) .\tag{5.3}
$$

To verify this identity, take $\phi \in C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { d } )$ . Symmetry of the heat kernel and $\nabla \mathsf { H } _ { t } \phi = \mathsf { H } _ { t } \nabla \phi$ give

$$
\begin{array} { r l } { \displaystyle \int \phi \mathsf { H } _ { t } ( \mathrm { d i v } J ) \mathrm { d } x = \int ( \mathsf { H } _ { t } \phi ) \mathrm { d i v } J \mathrm { d } x } \\ { \displaystyle } & { = - \int \left. \nabla \mathsf { H } _ { t } \phi , J \right. \mathrm { d } x } \\ { \displaystyle } & { = - \int \left. \nabla \phi , \mathsf { H } _ { t } J \right. \mathrm { d } x = \int \phi \mathrm { d i v } \big ( \mathsf { H } _ { t } J \big ) \mathrm { d } x . } \end{array}
$$

For the integration by parts with $\mathsf { H } _ { t } \phi$ , first multiply it by smooth cutofs $\chi _ { k }$ equal to one on the ball of radius $k ,$ supported in the ball of radius 2k, and satisfying $\| \nabla \chi _ { k } \| _ { \infty } \leq C / k$ . The cutof error is bounded by

$$
\left\| \mathsf { H } _ { t } \phi \right\| _ { \infty } \left\| \nabla \chi _ { k } \right\| _ { \infty } \left\| J \right\| _ { L ^ { 1 } } \longrightarrow 0 .
$$

The remaining terms converge by dominated convergence, since $J ,$ div $J \in L ^ { 1 }$ and $\mathsf { H } _ { t } \phi , \nabla \mathsf { H } _ { t } \phi$ are bounded. This proves (5.3).

Applying $\mathsf { H } _ { h }$ to (4.19), and using stationarity (3.8) and (5.3), yields

$$
\begin{array} { r l } & { \widehat { \boldsymbol { \pi } } _ { \lambda , h } = \mathsf { H } _ { h } ( ( T _ { h } ) _ { \# } \widehat { \boldsymbol { \pi } } _ { \lambda , h } ) } \\ & { \qquad = \mathsf { H } _ { h } \widehat { \boldsymbol { \pi } } _ { \lambda , h } + h \mathsf { H } _ { h } ( \mathrm { d i v } \boldsymbol { J } ) } \\ & { \qquad = \mathsf { H } _ { h } \widehat { \boldsymbol { \pi } } _ { \lambda , h } + h \mathrm { d i v } \mathsf { H } _ { h } \boldsymbol { J } . } \end{array}\tag{5.4}
$$

For $k = 0 , \ldots , n - 1$ , the semigroup property and (5.3) then give

$$
\mathsf { H } _ { k h } \widehat { \pi } _ { \lambda , h } - \mathsf { H } _ { ( k + 1 ) h } \widehat { \pi } _ { \lambda , h } = h \mathrm { d i v } \mathsf { H } _ { ( k + 1 ) h } J .
$$

Summing these equalities gives the exact finite identity

$$
\widehat { \pi } _ { \lambda , h } = \mathsf { H } _ { n h } \widehat { \pi } _ { \lambda , h } + h \sum _ { j = 1 } ^ { n } \mathrm { d i v } \mathsf { H } _ { j h } J , \qquad n \geq 1 .\tag{5.5}
$$

For positive time, convolution of an $L ^ { 1 }$ function with the Gaussian kernel is smooth: every spatial derivative can be placed on the kernel, whose derivatives are bounded and integrable. Thus the right-hand side of (5.5) is smooth. The identity, initially obtained distributionally, consequently holds pointwise for the smooth representative of $\widehat { \pi } _ { \lambda , h }$

Choose

$$
\tau : = \frac { c _ { 1 } } { r ^ { 2 } R ^ { 2 } } , \qquad n : = \left\lfloor \frac { \tau } { h } \right\rfloor ,
$$

where $0 < c _ { 1 } \leq 1$ is a suficiently small universal constant. Since $r \leq 4 \ell _ { h , \lambda }$ , (3.11) gives $h r ^ { 2 } R ^ { 2 } \leq 1 6 c$ Decreasing the constant c in (3.11) relative to $c _ { 1 }$ therefore ensures

$$
h \leq \frac { \tau } { 2 } , \qquad \frac { \tau } { 2 } \leq n h \leq \tau .
$$

We will apply (4.8) and (4.9) with output order $s = 2 r$ , at times $t = n h$ and $t = j h$ , respectively. Their common time restriction holds because, for $1 \leq j \leq n$ 2

$$
( j h ) ( 2 r ) ^ { 2 } R ^ { 2 } \leq \tau ( 2 r ) ^ { 2 } R ^ { 2 } = 4 c _ { 1 } .\tag{5.6}
$$

All terms in (5.5) are smooth, and its sum is finite. We may therefore diferentiate it to obtain

$$
\nabla \widehat { \pi } _ { \lambda , h } = \nabla \mathsf { H } _ { n h } \widehat { \pi } _ { \lambda , h } + h \sum _ { j = 1 } ^ { n } \nabla \operatorname { d i v } \mathsf { H } _ { j h } { J } .\tag{5.7}
$$

Taking the weighted $L ^ { 2 r }$ norm in (5.7), and using (4.8), (4.9), and $n h \ge \tau / 2$ , gives

$$
\begin{array} { r l } { \displaystyle \| \nabla \widehat { \pi } _ { \lambda , h } / \pi _ { \lambda } \| _ { L ^ { 2 r } ( \pi _ { \lambda } ) } \leq \| \nabla \mathsf { H } _ { n h } \widehat { \pi } _ { \lambda , h } / \pi _ { \lambda } \| _ { L ^ { 2 r } ( \pi _ { \lambda } ) } + h \displaystyle \sum _ { j = 1 } ^ { n } \| \nabla \mathrm { d i v } \mathsf { H } _ { j h } J / \pi _ { \lambda } \| _ { L ^ { 2 r } ( \pi _ { \lambda } ) } } & { } \\ { \displaystyle \leq C \tau ^ { - 1 / 2 } \| \widehat { \pi } _ { \lambda , h } / \pi _ { \lambda } \| _ { L ^ { 4 r } ( \pi _ { \lambda } ) } + C h \displaystyle \sum _ { j = 1 } ^ { n } \frac { \sqrt { r } } { j h } \| J / \pi _ { \lambda } \| _ { L ^ { 8 r } ( \pi _ { \lambda } ) } } & { } \\ { \displaystyle \leq C r R \left( 1 + \displaystyle \sum _ { j = 1 } ^ { n } \frac { 1 } { j } \right) \leq C r R ( 1 + \log n ) \leq C r R \log \displaystyle \frac { e } { m h } . } \end{array}\tag{5.8}
$$

For the third inequality in (5.8), use (5.2), $\tau ^ { - 1 / 2 } = r R / \sqrt { c _ { 1 } }$ , and (5.1). The last two inequalities use $\textstyle \sum _ { j = 1 } ^ { n } j ^ { - 1 } \leq 1 + \log n$ and $n \leq \tau / h \leq 1 / ( m h )$ , since $\bar { R } ^ { 2 } \geq m$ and $c _ { 1 } \leq 1$

## 5.3 Proof of the flux bounds

Proof of Lemma $4 . 5 .$ To estimate H, we express $\nabla \rho _ { t }$ in terms of $\nabla \widehat { \pi } _ { \lambda , h }$ and J. We first justify the commutation identity

$$
\nabla ( \mathsf { H } _ { t } \widehat { \pi } _ { \lambda , h } ) = \mathsf { H } _ { t } ( \nabla \widehat { \pi } _ { \lambda , h } ) , \qquad t > 0 .\tag{5.9}
$$

By (5.8) and H¨older’s inequality,

$$
\int \| \nabla \widehat { \pi } _ { \lambda , h } \| \mathrm { d } x = \int \| \nabla \widehat { \pi } _ { \lambda , h } / \pi _ { \lambda } \| \mathrm { d } \pi _ { \lambda } \leq \| \nabla \widehat { \pi } _ { \lambda , h } / \pi _ { \lambda } \| _ { L ^ { 2 r } ( \pi _ { \lambda } ) } < \infty .
$$

Thus both $\widehat { \pi } _ { \lambda , h }$ and $\nabla \widehat { \pi } _ { \lambda , h }$ belong to $L ^ { 1 } ( \mathbb { R } ^ { d } )$ . For each $i ,$ diferentiation of the Gaussian kernel and integration by parts give

$$
\begin{array} { r l } & { \partial _ { i } ( \mathsf { H } _ { t } \widehat { \pi } _ { \lambda , h } ) ( x ) = \displaystyle \int \partial _ { x _ { i } } \left[ \frac { e ^ { - \| x - y \| ^ { 2 } / ( 4 t ) } } { ( 4 \pi t ) ^ { d / 2 } } \right] \widehat { \pi } _ { \lambda , h } ( y ) \mathrm { d } y } \\ & { \qquad = - \displaystyle \int \partial _ { y _ { i } } \left[ \frac { e ^ { - \| x - y \| ^ { 2 } / ( 4 t ) } } { ( 4 \pi t ) ^ { d / 2 } } \right] \widehat { \pi } _ { \lambda , h } ( y ) \mathrm { d } y } \\ & { \qquad = \displaystyle \int \frac { e ^ { - \| x - y \| ^ { 2 } / ( 4 t ) } } { ( 4 \pi t ) ^ { d / 2 } } \partial _ { i } \widehat { \pi } _ { \lambda , h } ( y ) \mathrm { d } y = \mathsf { H } _ { t } ( \partial _ { i } \widehat { \pi } _ { \lambda , h } ) ( x ) . } \end{array}
$$

For fixed $t > 0 ,$ the kernel and its first derivatives are bounded. Diferentiation under the integral therefore follows from $\widehat { \pi } _ { \lambda , h } \in L ^ { 1 }$ . The integration by parts is justified by the cutof argument used for (5.3), now using $\widehat { \pi } _ { \lambda , h } , \nabla \widehat { \pi } _ { \lambda , h } \in L ^ { 1 }$ . This proves (5.9).

For $0 < t \leq h$ , the definition of $\rho _ { t }$ , (4.19), and (5.3) give

$$
\rho _ { t } = \mathsf { H } _ { t } \bigl ( \widehat { \pi } _ { \lambda , h } + h \operatorname { d i v } J \bigr ) = \mathsf { H } _ { t } \widehat { \pi } _ { \lambda , h } + h \operatorname { d i v } \mathsf { H } _ { t } J .
$$

Diferentiating this identity between smooth functions and using (5.9), we obtain

$$
\begin{array} { r l } & { \nabla \rho _ { t } = \nabla \mathsf { H } _ { t } \widehat { \pi } _ { \lambda , h } + h \nabla \mathrm { d i v } \mathsf { H } _ { t } J } \\ & { \qquad = \mathsf { H } _ { t } ( \nabla \widehat { \pi } _ { \lambda , h } ) + h \nabla \mathrm { d i v } \mathsf { H } _ { t } J , \qquad 0 < t \leq h . } \end{array}\tag{5.10}
$$

Apply (4.7) to the vector field $\nabla \widehat { \pi } _ { \lambda , h }$ and (4.9) to J, both with output order $s = r$ . The time restrictions hold because $t r ^ { 2 } R ^ { 2 } \leq h r ^ { 2 } R ^ { 2 } \leq c _ { 1 } / 2$ , as verified in Section 5.2. Using (5.2), (5.8), and $\| J / \pi _ { \lambda } \| _ { L ^ { 4 r } ( \pi _ { \lambda } ) } \le \| J / \pi _ { \lambda } \| _ { L ^ { 8 r } ( \pi _ { \lambda } ) }$ , we get

$$
\begin{array} { l } { \displaystyle \| \nabla \rho _ { t } / \pi _ { \lambda } \| _ { L ^ { r } ( \pi _ { \lambda } ) } \leq C \| \nabla \widehat { \pi } _ { \lambda , h } / \pi _ { \lambda } \| _ { L ^ { 2 r } ( \pi _ { \lambda } ) } + \frac { C h \sqrt { r } } { t } \| J / \pi _ { \lambda } \| _ { L ^ { 4 r } ( \pi _ { \lambda } ) } } \\ { \leq C r R \log \displaystyle \frac { e } { m h } + \frac { C h r R } { t } , \qquad 0 < t \leq h . } \end{array}
$$

The weight t in (4.13) cancels the apparent singularity at zero, proving (4.25). For (4.26), apply (4.10) with output order $s = 2$ and time $u \in [ 0 , h ]$ . Its time restriction is weaker than the one already verified with output order 8r before (5.2). Using (2.3) and H¨older’s inequality, we obtain

$$
\begin{array} { r l } {  { \| ( T _ { u } ) _ { \# } ( ( \nabla U _ { \lambda } \otimes \nabla U _ { \lambda } ) \widehat \pi _ { \lambda , h } ) / \pi _ { \lambda } \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } } \quad } & { } \\ & { \leq C \| ( \nabla U _ { \lambda } \otimes \nabla U _ { \lambda } ) \widehat \pi _ { \lambda , h } / \pi _ { \lambda } \| _ { L ^ { 4 } ( \pi _ { \lambda } ) } } \\ & { \leq C \| \nabla U _ { \lambda } \| _ { L ^ { 1 6 } ( \pi _ { \lambda } ) } ^ { 2 } \| \widehat \pi _ { \lambda , h } / \pi _ { \lambda } \| _ { L ^ { 8 } ( \pi _ { \lambda } ) } \leq C R ^ { 2 } , \qquad 0 \leq u \leq h . } \end{array}\tag{5.11}
$$

The last inequality follows from (4.2) at order 16 and (5.1). Minkowski’s inequality in (4.14), followed by (5.11), now gives

$$
\begin{array} { r l r } {  { \| E / \pi _ { \lambda } \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } \le \displaystyle \frac { 1 } { h } \int _ { 0 } ^ { h } ( h - u ) \| ( T _ { u } ) _ { \# } ( ( \nabla U _ { \lambda } \otimes \nabla U _ { \lambda } ) \widehat { \pi } _ { \lambda , h } ) / \pi _ { \lambda } \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } \mathrm { d } u } } \\ & { } & { \le \displaystyle \frac { C R ^ { 2 } } { h } \int _ { 0 } ^ { h } ( h - u ) \mathrm { d } u \le C h R ^ { 2 } . } \end{array}
$$

This proves (4.26). Finally, $r \geq 4$ and (4.2) imply

$$
\| \nabla U _ { \lambda } \otimes H / \pi _ { \lambda } \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } \leq \| \nabla U _ { \lambda } \| _ { L ^ { 4 } ( \pi _ { \lambda } ) } \| H / \pi _ { \lambda } \| _ { L ^ { 4 } ( \pi _ { \lambda } ) } \leq C R \| H / \pi _ { \lambda } \| _ { L ^ { r } ( \pi _ { \lambda } ) } .
$$

Combining this with (4.26) proves (4.27). This proves (4.25), (4.26), and (4.27), completing the proof of Lemma 4.5. □

## 6 Conclusion

We established a $\widetilde O ( h )$ invariant-measure bias bound relative to the smoothed target under an explicit step-size condition, with only logarithmic dependence on $\lambda ^ { - 1 }$ in the error coeficient. Combining this bound with Moreau approximation and Wasserstein contraction gives $\widetilde { O } ( \varepsilon ^ { - 1 } )$ iterations to achieve $\sqrt { m } W _ { 2 } ( \mu _ { N } , \pi ) \le \varepsilon$ for fixed model parameters and initialization. These guarantees require no additional higher-order smoothness assumptions and leave the classical MYULA algorithm unchanged.

## A Target Moments and Normalized Operator Estimates

## A.1 Target drift moments

Proof of Lemma $4 . 1 .$ The identity and bound in (4.1) follow from Xin and Zhang [9, Lemma 7.1 and Proposition A.4]. We prove the higher-moment and exponential estimates.

Since $\pi _ { \lambda }$ has Gaussian tails and $\nabla f$ has at most linear growth, all moments of $\nabla f$ are finite. Integration by parts with polynomially growing vector fields is justified by spatial cutofs; see Section D.1.

For $s \geq 2$ , apply integration by parts to the $C ^ { 1 }$ test field $\| \nabla f \| ^ { s - 2 } \nabla f$ . Using $\nabla U _ { \lambda } = \nabla f + \nabla g _ { \lambda }$ we obtain

$$
\begin{array} { r l } & { \displaystyle \int \| \nabla f \| ^ { s } \mathrm { d } \pi _ { \lambda } = \int \| \nabla f \| ^ { s - 2 } \Delta f \mathrm { d } \pi _ { \lambda } } \\ & { \quad \quad \quad + ( s - 2 ) \int \| \nabla f \| ^ { s - 4 } ( \nabla f ) ^ { \top } \nabla ^ { 2 } f \nabla f \mathrm { d } \pi _ { \lambda } } \\ & { \quad \quad \quad - \displaystyle \int \| \nabla f \| ^ { s - 2 } \langle \nabla f , \nabla g _ { \lambda } \rangle \mathrm { d } \pi _ { \lambda } } \\ & { \quad \quad \quad \le [ \tau _ { f } + ( s - 2 ) \operatorname* { m i n } ( L _ { f } , \tau _ { f } ) ] \displaystyle \int \| \nabla f \| ^ { s - 2 } \mathrm { d } \pi _ { \lambda } } \\ & { \quad \quad \quad + G \displaystyle \int \| \nabla f \| ^ { s - 1 } \mathrm { d } \pi _ { \lambda } . } \end{array}\tag{A.1}
$$

The inequality uses $\Delta f \le \tau _ { f } , 0 \le \nabla ^ { 2 } f \preceq \operatorname* { m i n } ( L _ { f } , \tau _ { f } ) I .$ , and $\| \nabla g _ { \lambda } \| \le G$ . The term with coeficient $s - 2$ is zero when $s = 2 ;$ for $s > 2$ , its integrand extends continuously by zero at $\nabla f = 0$

H¨older’s inequality gives

$$
\int \| \nabla f \| ^ { s - 2 } \mathrm { d } \pi _ { \lambda } \leq \| \nabla f \| _ { L ^ { s } ( \pi _ { \lambda } ) } ^ { s - 2 } , \qquad \int \| \nabla f \| ^ { s - 1 } \mathrm { d } \pi _ { \lambda } \leq \| \nabla f \| _ { L ^ { s } ( \pi _ { \lambda } ) } ^ { s - 1 } .
$$

$\mathrm { I f ~ } \| \nabla f \| _ { L ^ { s } ( \pi _ { \lambda } ) } > 0$ , substituting these bounds into (A.1) and dividing by $\| \nabla f \| _ { L ^ { s } ( \pi _ { \lambda } ) } ^ { s - 2 }$ yields

$$
\| \nabla f \| _ { L ^ { s } ( \pi _ { \lambda } ) } ^ { 2 } \le \tau _ { f } + ( s - 2 ) \operatorname * { m i n } ( L _ { f } , \tau _ { f } ) + G \left\| \nabla f \right\| _ { L ^ { s } ( \pi _ { \lambda } ) } .
$$

Solving this quadratic inequality gives

$$
\| \nabla f \| _ { L ^ { s } ( \pi _ { \lambda } ) } \leq G + \sqrt { \tau _ { f } + ( s - 2 ) \operatorname* { m i n } ( L _ { f } , \tau _ { f } ) } .\tag{A.2}
$$

Since $\| \nabla g _ { \lambda } \| \le G , G \le R$ , and $\tau _ { f } \leq R ^ { 2 }$ , it follows that

$$
\begin{array} { r } { \| \nabla U _ { \lambda } \| _ { L ^ { s } ( \pi _ { \lambda } ) } \leq \| \nabla f \| _ { L ^ { s } ( \pi _ { \lambda } ) } + G \leq 2 G + \sqrt { ( s - 1 ) \tau _ { f } } \leq 3 \sqrt { s } R , } \end{array}
$$

which proves (4.2).

Finally, (4.2) gives $\begin{array} { r } { \int \| \nabla U _ { \lambda } \| ^ { 2 k } \ \mathrm { d } \pi _ { \lambda } \leq ( 1 8 k R ^ { 2 } ) ^ { k } } \end{array}$ . Expanding the exponential and using $k ! \geq$ $( k / e ) ^ { k }$ , we obtain

$$
\int e ^ { \| \nabla U _ { \lambda } \| ^ { 2 } / ( 2 5 6 R ^ { 2 } ) } \mathrm { d } \pi _ { \lambda } \leq 1 + \sum _ { k \geq 1 } ( 1 8 e / 2 5 6 ) ^ { k } \leq 2 .
$$

This proves (4.3).

## A.2 A Gibbs change of variables for the heat operator

This and the following two subsections prove Lemma 4.3. We use the exponential moment bound (4.3):

$$
\int e ^ { \| \nabla U _ { \lambda } \| ^ { 2 } / ( 2 5 6 R ^ { 2 } ) } \mathrm { d } \pi _ { \lambda } \leq 2 .\tag{A.3}
$$

Let $X \sim \pi _ { \lambda }$ and $Z _ { t } \sim N ( 0 , 2 t I _ { d } )$ be independent, and write $\varphi _ { 2 t I _ { d } }$ for the density of $Z _ { t }$ . The Gibbs density satisfies

$$
\pi _ { \lambda } ( x ) = \pi _ { \lambda } ( y ) e ^ { U _ { \lambda } ( y ) - U _ { \lambda } ( x ) } .
$$

For every $a \geq 0$ , independence and the substitution $y = x + z$ therefore give

$$
\begin{array} { r l } {  { \mathbb { E } e ^ { a [ U _ { \lambda } ( X + Z _ { t } ) - U _ { \lambda } ( X ) ] } } } \\ & { = \int \int \pi _ { \lambda } ( x ) \varphi _ { 2 t { \cal I } _ { d } } ( y - x ) e ^ { a [ U _ { \lambda } ( y ) - U _ { \lambda } ( x ) ] } \mathrm { d } x \mathrm { d } y } \\ & { = \displaystyle \int \int \pi _ { \lambda } ( y ) \varphi _ { 2 t { \cal I } _ { d } } ( y - x ) e ^ { ( a + 1 ) [ U _ { \lambda } ( y ) - U _ { \lambda } ( x ) ] } \mathrm { d } x \mathrm { d } y } \\ & { \le \displaystyle \int e ^ { ( a + 1 ) ^ { 2 } t \| \nabla U _ { \lambda } ( y ) \| ^ { 2 } } \pi _ { \lambda } ( y ) \mathrm { d } y . } \end{array}\tag{A.4}
$$

For the last inequality, convexity gives

$$
U _ { \lambda } ( y ) - U _ { \lambda } ( x ) \le \langle \nabla U _ { \lambda } ( y ) , y - x \rangle ,
$$

and the Gaussian integral in x is

$$
\int \varphi _ { 2 t I _ { d } } ( y - x ) e ^ { ( a + 1 ) \left. \nabla U _ { \lambda } ( y ) , y - x \right. } \mathrm { d } x = e ^ { ( a + 1 ) ^ { 2 } t \| \nabla U _ { \lambda } ( y ) \| ^ { 2 } } .
$$

Thus, by (A.3), the expectation in (A.4) is at most 2 whenever $( a + 1 ) ^ { 2 } t R ^ { 2 } \leq 1 / 2 5 6$

Write $w = F / \pi _ { \lambda }$ . Since the heat kernel has total mass one, Jensen’s inequality gives

$$
\| \mathsf { H } _ { t } F ( y ) \| ^ { s } = \left\| \int \varphi _ { 2 t I _ { d } } ( y - x ) F ( x ) \mathrm { d } x \right\| ^ { s } \leq \int \varphi _ { 2 t I _ { d } } ( y - x ) \| F ( x ) \| ^ { s } \mathrm { d } x .
$$

Multiply by $\pi _ { \lambda } ( y ) ^ { 1 - s }$ and integrate in y. Using Tonelli’s theorem and $F ( x ) = \pi _ { \lambda } ( x ) w ( x )$ , we obtain

$$
\begin{array} { l } { \displaystyle | | \mathsf { H } _ { t } F / \pi _ { \lambda } | | _ { L ^ { s } ( \pi _ { \lambda } ) } ^ { s } = \int | | \mathsf { H } _ { t } F ( y ) | | ^ { s } \pi _ { \lambda } ( y ) ^ { 1 - s } \mathrm { d } y } \\ { \displaystyle \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ { \displaystyle \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ { \displaystyle \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ { \displaystyle \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ { \displaystyle \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ { \displaystyle \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ { \displaystyle \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ { \displaystyle \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ { \displaystyle \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\  \displaystyle \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \ \end{array}\tag{A.5}
$$

Cauchy–Schwarz then yields

$$
\begin{array} { r } { \| \mathsf { H } _ { t } F / \pi _ { \lambda } \| _ { L ^ { s } ( \pi _ { \lambda } ) } ^ { s } \le \| w \| _ { L ^ { 2 s } ( \pi _ { \lambda } ) } ^ { s } \left( \mathbb { E } e ^ { 2 ( s - 1 ) [ U _ { \lambda } ( X + Z _ { t } ) - U _ { \lambda } ( X ) ] } \right) ^ { 1 / 2 } . } \end{array}
$$

Apply (A.4) with $a = 2 ( s - 1 )$ . Together with (A.3), this bounds the expectation by 2 whenever $( 2 s - 1 ) ^ { 2 } t R ^ { 2 } \leq 1 / 2 5 6$ . Since $2 s - 1 \leq 2 s$ , this condition follows from $t s ^ { 2 } R ^ { 2 } \leq c$ for a suficiently small universal constant c. Taking sth roots proves (4.7). The argument applies to scalar, vector, and matrix densities, using the absolute value, Euclidean norm, and Frobenius norm, respectively.

## A.3 Gaussian derivatives without an extra dimension factor

Let F be scalar with $\| F / \pi _ { \lambda } \| _ { L ^ { 2 s } ( \pi _ { \lambda } ) } < \infty$ . Since $\pi _ { \lambda }$ is a bounded probability density,

$$
\int | F | \mathrm { d } x \leq \| F \big / \pi _ { \lambda } \| _ { L ^ { 2 s } ( \pi _ { \lambda } ) } , \qquad \int | F | ^ { 2 } \mathrm { d } x \leq \| \pi _ { \lambda } \| _ { \infty } \| F \big / \pi _ { \lambda } \| _ { L ^ { 2 s } ( \pi _ { \lambda } ) } ^ { 2 } .
$$

For fixed $t > 0 ,$ , every derivative of the Gaussian kernel is bounded. Thus diferentiation under the integral is justified by $F \in L ^ { 1 } ( \mathbb { R } ^ { d } )$ , and ${ \sf H } _ { t } F$ is smooth.

For a unit vector e, diferentiating the kernel gives

$$
\begin{array} { r l } & { e \cdot \nabla { \mathsf { H } } _ { t } F ( x ) = \displaystyle \int e \cdot \nabla _ { x } \varphi _ { 2 t I _ { d } } ( x - y ) F ( y ) \mathrm { d } y } \\ & { \qquad = - \displaystyle \int \frac { e \cdot ( x - y ) } { 2 t } \varphi _ { 2 t I _ { d } } ( x - y ) F ( y ) \mathrm { d } y } \\ & { \qquad = - \mathbb { E } \left[ \frac { e \cdot Z _ { t } } { 2 t } F ( x - Z _ { t } ) \right] . } \end{array}
$$

Using $\mathbb { E } ( e \cdot Z _ { t } ) ^ { 2 } = 2 t$ , Cauchy–Schwarz yields

$$
\begin{array} { r l } & { \| \nabla \mathsf { H } _ { t } F ( x ) \| = \displaystyle \operatorname* { s u p } _ { \| e \| = 1 } \bigg | \mathbb { E } \left[ \frac { e \cdot Z _ { t } } { 2 t } F ( x - Z _ { t } ) \right] \bigg | } \\ & { \qquad \leq ( 2 t ) ^ { - 1 / 2 } \big ( \mathsf { H } _ { t } | F | ^ { 2 } ( x ) \big ) ^ { 1 / 2 } . } \end{array}
$$

For $s \geq 2$ , Jensen then gives $\| \nabla \mathsf { H } _ { t } F \| ^ { s } \leq ( 2 t ) ^ { - s / 2 } \mathsf { H } _ { t } | F | ^ { s }$ . Multiply by $\pi _ { \lambda } ^ { 1 - s }$ , integrate, and use the same endpoint exchange as in $\left( \mathrm { { A . 5 } } \right)$ . This proves the scalar gradient estimate (4.8).

To estimate the divergence of a vector field, let $s ^ { \prime } = s / ( s - 1 )$ and $a = 2 s / ( 2 s - 1 )$ within this calculation. Initially take $\phi \in C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { d } )$ with $\| \phi \| _ { L ^ { s ^ { \prime } } ( \pi _ { \lambda } ) } \leq 1$ . Integration by parts against the compactly supported $\phi$ and symmetry of the Gaussian kernel give

$$
\begin{array} { r l } & { \displaystyle \int \phi \mathrm { d i v } \mathsf { H } _ { t } F \mathrm { d } x = - \int \left. \nabla \phi , \mathsf { H } _ { t } F \right. \mathrm { d } x } \\ & { \quad \quad \quad = - \int \left. \mathsf { H } _ { t } ( \nabla \phi ) , F \right. \mathrm { d } x } \\ & { \quad \quad \quad = - \int \left. \nabla \mathsf { H } _ { t } \phi , F / \pi _ { \lambda } \right. \mathrm { d } \pi _ { \lambda } . } \end{array}
$$

The exchange of integrals is justified by $F \in L ^ { 1 }$ and the boundedness of $\nabla \phi .$ . The last equality also uses $\mathsf { H } _ { t } ( \nabla \phi ) = \nabla \mathsf { H } _ { t } \phi$ . Since $1 / ( 2 s ) + 1 / a = 1$ , H¨older’s inequality with respect to $\pi _ { \lambda }$ yields

$$
\left| \int \phi \mathrm { d i v } \mathsf { H } _ { t } F \mathrm { d } x \right| \leq \| F / \pi _ { \lambda } \| _ { L ^ { 2 s } ( \pi _ { \lambda } ) } \| \nabla \mathsf { H } _ { t } \phi \| _ { L ^ { a } ( \pi _ { \lambda } ) } .\tag{A.6}
$$

For every unit vector e, the scalar $e \cdot Z _ { t }$ has distribution N(0, 2t), and hence

$$
\left( \mathbb { E } | e \cdot Z _ { t } | ^ { 2 s } \right) ^ { 1 / ( 2 s ) } \leq C { \sqrt { s t } } .
$$

Since $1 / ( 2 s ) + 1 / a = 1$ , the derivative formula above and H¨older’s inequality give

$$
\begin{array} { r l } {  { \| \nabla \mathsf { H } _ { t } \phi ( x ) \| = \operatorname* { s u p } _ { \| e \| = 1 } | \mathbb { E } [ \frac { e \cdot Z _ { t } } { 2 t } \phi ( x - Z _ { t } ) ] | } } \\ & { \leq \frac { 1 } { 2 t } \operatorname* { s u p } _ { \| e \| = 1 } ( \mathbb { E } | e \cdot Z _ { t } | ^ { 2 s } ) ^ { 1 / ( 2 s ) } ( \mathbb { E } | \phi ( x - Z _ { t } ) | ^ { a } ) ^ { 1 / a } } \\ & { \leq C \sqrt { s / t } ( \mathsf { H } _ { t } | \phi | ^ { a } ( x ) ) ^ { 1 / a } . } \end{array}
$$

To integrate this bound, use symmetry of the heat kernel and H¨older with conjugate exponents $s ^ { \prime } / a = ( 2 s - 1 ) / ( 2 s - 2 )$ and $2 s - 1 \colon$

$$
\begin{array} { r l r } {  { \int \mathsf { H } _ { t } | \phi | ^ { a } \mathrm { d } \pi _ { \lambda } = \int | \phi | ^ { a } \frac { \mathsf { H } _ { t } \pi _ { \lambda } } { \pi _ { \lambda } } \mathrm { d } \pi _ { \lambda } } } \\ & { } & { \leq \| \phi \| _ { L ^ { s ^ { \prime } } ( \pi _ { \lambda } ) } ^ { a } \| \mathsf { H } _ { t } \pi _ { \lambda } / \pi _ { \lambda } \| _ { L ^ { 2 s - 1 } ( \pi _ { \lambda } ) } . } \end{array}
$$

Consequently,

$$
\| \nabla \mathsf { H } _ { t } \phi \| _ { L ^ { a } ( \pi _ { \lambda } ) } \leq C \sqrt { s / t } \left\| \phi \right\| _ { L ^ { s ^ { \prime } } ( \pi _ { \lambda } ) } \| \mathsf { H } _ { t } \pi _ { \lambda } / \pi _ { \lambda } \| _ { L ^ { 2 s - 1 } ( \pi _ { \lambda } ) } ^ { 1 / a } \leq C \sqrt { s / t } .
$$

The last inequality uses $\| \phi \| _ { L ^ { s ^ { \prime } } ( \pi _ { \lambda } ) } \le 1$ and (4.7) with input $\pi _ { \lambda }$ and order $2 s - 1$ . Since $2 s - 1 \leq 2 s$ 2 its time restriction follows from $t s ^ { 2 } R ^ { 2 } \leq c$ after decreasing the universal constant c. Duality and density of $C _ { c } ^ { \infty } ( \mathbb { R } ^ { d } )$ in $L ^ { s ^ { \prime } } ( \pi _ { \lambda } )$ therefore yield

$$
\begin{array} { r } { \| \mathrm { d i v } \mathsf { H } _ { t } F / \pi _ { \lambda } \| _ { L ^ { s } ( \pi _ { \lambda } ) } \leq C \sqrt { s / t } \| F / \pi _ { \lambda } \| _ { L ^ { 2 s } ( \pi _ { \lambda } ) } . } \end{array}\tag{A.7}
$$

Finally, ∇ div $\mathsf { H } _ { t } = \nabla \mathsf { H } _ { t / 2 } \circ \mathrm { d i v } \mathsf { H } _ { t / 2 }$ . Apply the scalar gradient bound and then (A.7), with order 2s in the latter. The result is (4.9).

## A.4 Pushforward estimates by an integrated Jacobian

Recall that $0 ~ \preceq ~ \nabla ^ { 2 } U _ { \lambda } ~ \preceq ~ L _ { \lambda } I$ almost everywhere. For $u \geq 0$ , the expansion map $S _ { u } ( x ) =$ $x + u \nabla U _ { \lambda } ( x )$ is globally bi-Lipschitz. Indeed, $S _ { u } ^ { - 1 } y$ is the unique minimizer of the strongly convex function $x \mapsto { \textstyle { \frac { 1 } { 2 } } } \left\| x - y \right\| ^ { 2 } + u U _ { \lambda } ( x )$ ; monotonicity of $\nabla U _ { \lambda }$ gives a Lipschitz inverse. For $u > 0$ , set $y = S _ { u } ( x ) = x + u \nabla U _ { \lambda } ( x )$ . Monotonicity of $\nabla U _ { \lambda }$ gives

$$
\begin{array} { r } { \| \nabla U _ { \lambda } ( x ) \| ^ { 2 } \leq \langle \nabla U _ { \lambda } ( y ) , \nabla U _ { \lambda } ( x ) \rangle \leq \| \nabla U _ { \lambda } ( y ) \| \| \nabla U _ { \lambda } ( x ) \| , } \end{array}
$$

so $\lVert \nabla U _ { \lambda } ( x ) \rVert \leq \lVert \nabla U _ { \lambda } ( y ) \rVert$ . Convexity therefore implies

$$
\begin{array} { r } { U _ { \lambda } ( y ) - U _ { \lambda } ( x ) \leq \langle \nabla U _ { \lambda } ( y ) , y - x \rangle = u \langle \nabla U _ { \lambda } ( y ) , \nabla U _ { \lambda } ( x ) \rangle \leq u \| \nabla U _ { \lambda } ( y ) \| ^ { 2 } . } \end{array}
$$

Changing variables $y = S _ { u } ( x )$ and using the Gibbs density ratio, we obtain

$$
\begin{array} { l } { \displaystyle \int \operatorname* { d e t } ( I + { \boldsymbol { u } } \nabla ^ { 2 } U _ { \lambda } ( { \boldsymbol { x } } ) ) \pi _ { \lambda } ( { \boldsymbol { x } } ) \mathrm { d } { \boldsymbol { x } } = \int \pi _ { \lambda } ( S _ { \boldsymbol { u } } ^ { - 1 } { \boldsymbol { y } } ) \mathrm { d } { \boldsymbol { y } } } \\ { \displaystyle \qquad = \int e ^ { U _ { \lambda } ( { \boldsymbol { y } } ) - U _ { \lambda } ( S _ { \boldsymbol { u } } ^ { - 1 } { \boldsymbol { y } } ) } \mathrm { d } \pi _ { \lambda } ( { \boldsymbol { y } } ) } \\ { \displaystyle \qquad \leq \int e ^ { u \| \nabla U _ { \lambda } ( { \boldsymbol { y } } ) \| ^ { 2 } } \mathrm { d } \pi _ { \lambda } ( { \boldsymbol { y } } ) . } \end{array}\tag{A.8}
$$

The case $u = 0$ is immediate. By $\left( \mathrm { { A . 3 } } \right)$ , the last integral is at most 2 whenever u $R ^ { 2 } \leq 1 / 2 5 6$

Under $s t ( L _ { \lambda } + R ^ { 2 } ) \leq c ,$ one has $t L _ { \lambda } < 1$ , so the contraction map $T _ { t } = I - t \nabla U _ { \lambda }$ is bi-Lipschitz by Proposition 2.1. Write $J _ { t } ( x ) =$ det $( I - t \nabla ^ { 2 } U _ { \lambda } ( x ) ) > 0$ almost everywhere. Using (2.11) and changing variables $y = T _ { t } x$ , we obtain

$$
\begin{array} { l } { \displaystyle \| ( T _ { t } ) _ { \# } F / \pi _ { \lambda } \| _ { L ^ { s } ( \pi _ { \lambda } ) } ^ { s } = \int \left\| \frac { F ( x ) } { J _ { t } ( x ) } \right\| ^ { s } \pi _ { \lambda } ( T _ { t } x ) ^ { 1 - s } J _ { t } ( x ) \mathrm { d } x } \\ { \displaystyle \qquad = \int \left\| \frac { F ( x ) } { \pi _ { \lambda } ( x ) } \right\| ^ { s } e ^ { ( s - 1 ) [ U _ { \lambda } ( T _ { t } x ) - U _ { \lambda } ( x ) ] } J _ { t } ( x ) ^ { 1 - s } \mathrm { d } \pi _ { \lambda } ( x ) . } \end{array}\tag{A.9}
$$

Since $t L _ { \lambda } < 1$ , the smooth descent inequality gives

$$
{ U _ { \lambda } } ( T _ { t } x ) \le { U _ { \lambda } } ( x ) - { t } \left( 1 - \frac { { t } L _ { \lambda } } { 2 } \right) \left\| { \nabla } { U _ { \lambda } } ( x ) \right\| ^ { 2 } \le { U _ { \lambda } } ( x ) .
$$

Thus the exponential factor in (A.9) is at most one, and Cauchy–Schwarz yields

$$
\| ( T _ { t } ) _ { \# } F / \pi _ { \lambda } \| _ { L ^ { s } ( \pi _ { \lambda } ) } ^ { s } \le \| F / \pi _ { \lambda } \| _ { L ^ { 2 s } ( \pi _ { \lambda } ) } ^ { s } \left( \int J _ { t } ^ { - 2 ( s - 1 ) } \mathrm { d } \pi _ { \lambda } \right) ^ { 1 / 2 } .
$$

For $k \geq 1$ and $0 \leq z \leq 1 / ( 4 k )$ , Bernoulli’s inequality gives $( 1 - z ) ^ { k } \geq 1 - k z$ , and hence

$$
( 1 - z ) ^ { - k } \leq { \frac { 1 } { 1 - k z } } = 1 + { \frac { k z } { 1 - k z } } \leq 1 + 2 k z .
$$

Set $k = 2 ( s - 1 )$ . For almost every $x ,$ let $\mu _ { 1 } , \ldots , \mu _ { d }$ be the eigenvalues of $\nabla ^ { 2 } U _ { \lambda } ( x )$ . If $k t L _ { \lambda } \leq 1 / 4$ applying the preceding inequality to $z = t \mu _ { i }$ gives

$$
J _ { t } ( x ) ^ { - k } = \prod _ { i = 1 } ^ { d } ( 1 - t \mu _ { i } ) ^ { - k } \leq \prod _ { i = 1 } ^ { d } ( 1 + 2 k t \mu _ { i } ) = \operatorname * { d e t } ( I + 2 k t \nabla ^ { 2 } U _ { \lambda } ( x ) ) .
$$

Since $k \leq 2 s$ , the condition $s t ( L _ { \lambda } + R ^ { 2 } ) \leq c$ ensures both $k t L _ { \lambda } \leq 1 / 4$ and $2 k t R ^ { 2 } \leq 1 / 2 5 6$ when c is suficiently small. Applying (A.8) with $u = 2 k t$ therefore yields

$$
\int J _ { t } ^ { - k } \mathrm { d } \pi _ { \lambda } \leq \int \operatorname* { d e t } ( I + 2 k t \nabla ^ { 2 } U _ { \lambda } ) \mathrm { d } \pi _ { \lambda } \leq 2 .
$$

Combining this with the preceding Cauchy–Schwarz bound gives

$$
\begin{array} { r } { \| ( T _ { t } ) _ { \# } F / \pi _ { \lambda } \| _ { L ^ { s } ( \pi _ { \lambda } ) } \leq 2 ^ { 1 / ( 2 s ) } \| F / \pi _ { \lambda } \| _ { L ^ { 2 s } ( \pi _ { \lambda } ) } , } \end{array}
$$

which proves (4.10). The changes of variables above are justified by the area formula for bi-Lipschitz maps; see Proposition 2.1. This completes the proof of Lemma 4.3.

## B Finite-Order Stationary Density Ratios

This appendix proves the stationary density-ratio estimate in Lemma 4.2. The goal is to control

$$
\operatorname* { s u p } _ { 0 \leq t \leq h } \| \nu _ { t } / \pi _ { \lambda } \| _ { L ^ { \alpha } ( \pi _ { \lambda } ) }
$$

under the step restrictions (4.4).

The proof has three steps. In Section B.1, we establish the weighted drift and curvature moments needed for the calculation. In Section B.2, these moments give the drift-increment bound (B.7). In Section B.3, we insert that bound into the R´enyi dissipation identity and use the stationary endpoint condition to obtain (B.12).

The integrability needed for diferentiation and integration by parts is stated in Lemma B.1 and proved independently in Section D.2. The regularization and limiting arguments are supplied in Appendix D. The interpolation method follows the standard R´enyi analysis of Langevin algorithms [2].

## B.1 Weighted energy for the density ratio

Fix $\alpha \geq 2$ and $\lambda , h$ satisfying the restrictions of Lemma 4.2. We first work with an auxiliary smooth regularization. Let $\varrho \in C _ { c } ^ { \infty } ( \mathbb { R } ^ { d } )$ be nonnegative and symmetric, with $\textstyle \int { \varrho \mathrm { d } x } = 1$ , and set

$$
\varrho _ { \delta } ( x ) : = \delta ^ { - d } \varrho ( x / \delta ) , \qquad f ^ { ( \delta ) } : = f \ast \varrho _ { \delta } , \qquad g _ { \lambda } ^ { ( \delta ) } : = g _ { \lambda } \ast \varrho _ { \delta } , \qquad \delta > 0 .
$$

Define

$$
U _ { \lambda } ^ { ( \delta ) } : = f ^ { ( \delta ) } + g _ { \lambda } ^ { ( \delta ) } , \qquad \pi _ { \lambda } ^ { ( \delta ) } \propto e ^ { - U _ { \lambda } ^ { ( \delta ) } } , \qquad T _ { t } ^ { ( \delta ) } ( x ) : = x - t \nabla U _ { \lambda } ^ { ( \delta ) } ( x ) .
$$

The parameter $\delta$ is an auxiliary convolution scale, distinct from the Moreau parameter $\lambda .$ It will tend to zero with $\lambda$ and h fixed.

Let $Q _ { \lambda , h } ^ { ( \delta ) }$ be the Euler kernel associated with $U _ { \lambda } ^ { ( \delta ) }$ , and let $\widehat { \pi } _ { \lambda , h } ^ { ( \delta ) }$ be its invariant density. The corresponding stationary interpolation is

$$
\nu _ { t } ^ { ( \delta ) } : = \mathsf { H } _ { t } \big ( ( T _ { t } ^ { ( \delta ) } ) \# \widehat { \pi } _ { \lambda , h } ^ { ( \delta ) } \big ) , \qquad \nu _ { 0 } ^ { ( \delta ) } = \nu _ { h } ^ { ( \delta ) } = \widehat { \pi } _ { \lambda , h } ^ { ( \delta ) } .
$$

The structural bounds needed below hold with the same parameters $m , L _ { f } , \tau _ { f } , G$ , and $\lambda ;$ their preservation is verified in Section D.1.

For the calculation below, fix $\delta > 0$ and omit the superscript (δ) from the potentials and all associated densities, kernels, and maps. In particular, $\widehat { \pi } _ { \lambda , h }$ below denotes the invariant density of

the regularized Euler kernel. We first establish the integrability needed for the calculation at this fixed δ. We then obtain a quantitative estimate independent of $\delta ,$ and pass to the original potential in Section D.3.

Let $X \sim \widehat { \pi } _ { \lambda , h }$ and $Z \sim N ( 0 , I _ { d } )$ be independent, and, for $0 \leq t \leq h$ , let

$$
Y = X - t \nabla U \lambda ( X ) + { \sqrt { 2 t } } Z .
$$

The law of $Y$ is $\nu _ { t }$ . In this appendix only, suppressing the time argument, define

$$
\begin{array} { c } { u : = \displaystyle \frac { \nu _ { t } } { \pi _ { \lambda } } , \qquad s : = \nabla \log u , } \\ { e _ { t } ( y ) : = \nabla U _ { \lambda } ( y ) - \mathbb { E } [ \nabla U _ { \lambda } ( X ) \mid Y = y ] . } \end{array}
$$

The density ratio u is the quantity whose α-moment we seek to control. Its relative score $s = \nabla$ log u appears in the dissipation of that moment. The field $e _ { t }$ measures the discrepancy between the current drift $\nabla U _ { \lambda } ( y )$ and the drift frozen at the starting point X. The mixed term involving $e _ { t }$ and s will be estimated in Section B.3 using the drift-increment bound proved in Section B.2.

Before carrying out these calculations, we record the integrability statement that makes them legitimate.

Lemma B.1 (Integrability before dissipation). Consider the regularized interpolation defined above, with $\delta > 0$ fixed. Under the restrictions of Lemma 4.2, $\nu _ { t }$ is positive, locally $C ^ { 1 , 2 }$ in $( t , y )$ for $0 < t < h$ , and pointwise continuous at both time endpoints. Moreover,

$$
\operatorname* { s u p } _ { 0 \leq t \leq h } \int \big [ u ^ { \alpha } ( 1 + \| s \| ^ { 2 } + \| e _ { t } \| ^ { 2 } ) + u ^ { 3 \alpha / 2 } \big ] \mathrm { d } \pi _ { \lambda } < \infty .
$$

For every integer $k \geq 0$ , one also has

$$
\operatorname* { s u p } _ { 0 \leq t \leq h } \mathbb { E } \Big [ u ( Y ) ^ { \alpha - 1 } ( 1 + \| X \| + \| Y \| ) ^ { k } \Big ] < \infty .
$$

These bounds are uniform in $t \in [ 0 , h ]$ . Their finite values may depend on the fixed parameters and on $\delta ;$ no uniform bound as $\delta \downarrow 0$ is asserted in this lemma.

The proof of Lemma B.1 is given in Section D.2. It establishes the required integrability independently of the dissipation calculation. The finite bounds from that proof justify the operations below; their numerical values do not enter the quantitative estimate.

We now prepare two moment bounds for the calculation in Section B.2: an $L ^ { 2 }$ bound on $\nabla f$ and a bound on the mean of $\Delta g _ { \lambda }$ under the appropriate density-ratio weight. We define

$$
\omega ( \mathrm { d } x ) : = \frac { u ( x ) ^ { \alpha } \pi _ { \lambda } ( x ) \mathrm { d } x } { \int u ^ { \alpha } \mathrm { d } \pi _ { \lambda } } , \qquad I : = \mathbb { E } _ { \omega } \left\| s \right\| ^ { 2 } .\tag{B.1}
$$

A weighted moment of $\nabla f$ . Recall that $\nabla U _ { \lambda } = \nabla f + \nabla g _ { \lambda }$ and ∇ log $\omega = \alpha s - \nabla U _ { \lambda }$ . Integration by parts gives

$$
\begin{array} { r l } & { \mathbb { E } _ { \omega } \Delta f = - \mathbb { E } _ { \omega } \left. \nabla f , \nabla \log \omega \right. } \\ & { \quad \quad \quad = \mathbb { E } _ { \omega } \left\| \nabla f \right\| ^ { 2 } + \mathbb { E } _ { \omega } \left. \nabla f , \nabla g _ { \lambda } \right. - \alpha \mathbb { E } _ { \omega } \left. \nabla f , s \right. . } \end{array}
$$

The integration by parts is justified by spatial cutofs and the moment and score bounds in Lemma B.1. Rearranging the preceding identity, using $\Delta f \le \tau _ { f }$ and $\| \nabla g _ { \lambda } \| \le G$ , and applying Cauchy–Schwarz, we obtain

$$
\begin{array} { r l } & { \| \nabla f \| _ { L ^ { 2 } ( \omega ) } ^ { 2 } = \mathbb { E } _ { \omega } \Delta f - \mathbb { E } _ { \omega } \left. \nabla f , \nabla g _ { \lambda } \right. + \alpha \mathbb { E } _ { \omega } \left. \nabla f , s \right. } \\ & { \qquad \leq \tau _ { f } + G \mathbb { E } _ { \omega } \| \nabla f \| + \alpha \| \nabla f \| _ { L ^ { 2 } ( \omega ) } \sqrt { I } } \\ & { \qquad \leq \tau _ { f } + ( G + \alpha \sqrt { I } ) \| \nabla f \| _ { L ^ { 2 } ( \omega ) } . } \end{array}
$$

Using $R ^ { 2 } = \tau _ { f } + G R$ , we solve this inequality and obtain

$$
\| \nabla f \| _ { L ^ { 2 } ( \omega ) } \leq R + \alpha \sqrt { I } .\tag{B.2}
$$

The weighted mean of the Moreau curvature. We next estimate $\mathbb { E } _ { \omega } \Delta g _ { \lambda }$ . Applying integration by parts directly to $\nabla g _ { \lambda }$ gives

$$
\begin{array} { r l } & { \mathbb { E } _ { \omega } \Delta g _ { \lambda } = - \mathbb { E } _ { \omega } \left. \nabla g _ { \lambda } , \nabla \log \omega \right. } \\ & { \quad \quad \quad = \mathbb { E } _ { \omega } \left. \nabla g _ { \lambda } , \nabla f + \nabla g _ { \lambda } - \alpha s \right. } \\ & { \quad \quad \quad \leq G \left\| \nabla f \right\| _ { L ^ { 2 } ( \omega ) } + G ^ { 2 } + \alpha G \sqrt { I } } \\ & { \quad \quad \quad \leq G R + G ^ { 2 } + 2 \alpha G \sqrt { I } } \\ & { \quad \quad \quad \leq 2 G R + 2 \alpha G \sqrt { I } . } \end{array}\tag{B.3}
$$

The first inequality uses $\| \nabla g _ { \lambda } \| \le G$ and Cauchy–Schwarz. The next line uses (B.2), and the last uses $G ^ { 2 } \leq G R$

## B.2 Gaussian integration by parts with the density-ratio weight

This subsection proves (B.7).

For an integrable function $\Phi ( X , Y )$ , define the weighted joint expectation

$$
\mathbb { E } _ { \alpha } [ \Phi ( X , Y ) ] : = \frac { \mathbb { E } \big [ u ( Y ) ^ { \alpha - 1 } \Phi ( X , Y ) \big ] } { \int u ^ { \alpha } \mathrm { d } \pi _ { \lambda } } .
$$

Since Y has density $\nu _ { t } = u \pi _ { \lambda }$ , the denominator normalizes this expectation, and its Y marginal is ω:

$$
\mathbb { E } _ { \alpha } [ \phi ( Y ) ] = \frac { \int \phi ( y ) u ( y ) ^ { \alpha } \pi _ { \lambda } ( y ) \mathrm { d } y } { \int u ^ { \alpha } \mathrm { d } \pi _ { \lambda } } = \mathbb { E } _ { \omega } \phi
$$

for every integrable test function $\phi .$

Fix $0 < t \leq h$ . Let w be the gradient of a convex function with L-Lipschitz gradient, and write

$$
\Delta w : = w ( Y ) - w ( X ) .
$$

Cocoercivity and $Y - X = - t \nabla U _ { \lambda } ( X ) + \sqrt { 2 t } Z$ give

$$
\begin{array} { r } { \mathbb { E } _ { \alpha } \left\| \Delta w \right\| ^ { 2 } \leq - t L \mathbb { E } _ { \alpha } \left. \Delta w , \nabla U _ { \lambda } ( X ) \right. + L \mathbb { E } _ { \alpha } \left. \Delta w , \sqrt { 2 t } Z \right. . } \end{array}
$$

To evaluate the Gaussian term, condition on X and integrate by parts under the original standard Gaussian law of Z, keeping $u ( Y ) ^ { \alpha - 1 }$ in the integrand. The identities

$$
\partial _ { Z _ { i } } Y _ { j } = \sqrt { 2 t } \delta _ { i j } , \qquad \nabla _ { y } u ( y ) ^ { \alpha - 1 } = ( \alpha - 1 ) u ( y ) ^ { \alpha - 1 } s ( y )
$$

yield

$$
\begin{array} { r } { \mathbb E _ { \alpha } \left. \Delta w , \sqrt { 2 t } Z \right. = 2 t \mathbb E _ { \omega } \mathrm { d i v } w + 2 t ( \alpha - 1 ) \mathbb E _ { \alpha } \left. \Delta w , s ( Y ) \right. . } \end{array}
$$

The joint weighted moment bound with $k = 2$ in Lemma B.1 and $\| \Delta w \| \leq L \| Y - X \|$ give ${ \mathbb E } _ { \alpha } \left\| \Delta w \right\| ^ { 2 } < \infty$ . The same lemma gives $I < \infty$ . Since the Y marginal is $\omega ,$ , Cauchy–Schwarz yields

$$
\mathbb { E } _ { \alpha } \left| \left. \Delta w , s ( Y ) \right. \right| \leq \left( \mathbb { E } _ { \alpha } \left\| \Delta w \right\| ^ { 2 } \right) ^ { 1 / 2 } \sqrt { I } < \infty .
$$

The remaining terms are integrable by the joint moment bound and the bounded derivative of w.   
These bounds justify removing the cutofs in the Gaussian integration by parts.

Substitution gives

$$
\begin{array} { r l } & { \mathbb { E } _ { \alpha } \left\| w ( Y ) - w ( X ) \right\| ^ { 2 } \leq t L \Big [ - \mathbb { E } _ { \alpha } \left. w ( Y ) - w ( X ) , \nabla U _ { \lambda } ( X ) \right. + 2 \mathbb { E } _ { \omega } \normalfont \mathrm { d i v } w } \\ & { \qquad + \left. 2 ( \alpha - 1 ) \mathbb { E } _ { \alpha } \left. w ( Y ) - w ( X ) , s ( Y ) \right. \right] . } \end{array}\tag{B.4}
$$

The smooth component. We first estimate the squared increment of $\nabla f$ . Define

$$
D _ { f } : = \mathbb { E } _ { \alpha } \left\| \nabla f ( Y ) - \nabla f ( X ) \right\| ^ { 2 } .
$$

To use (B.4), we also need a weighted second moment of $\nabla U _ { \lambda } ( X )$ . The identity

$$
\nabla U _ { \lambda } ( X ) = \nabla f ( Y ) - \left( \nabla f ( Y ) - \nabla f ( X ) \right) + \nabla g _ { \lambda } ( X )
$$

and the fact that the Y marginal of $\mathbb { E } _ { \alpha }$ is ω give

$$
\begin{array} { r } { \left( \mathbb { E } _ { \alpha } \left\| \nabla U _ { \lambda } ( X ) \right\| ^ { 2 } \right) ^ { 1 / 2 } \leq \left\| \nabla f \right\| _ { L ^ { 2 } ( \omega ) } + \sqrt { D _ { f } } + G \leq 2 R + \alpha \sqrt { I } + \sqrt { D _ { f } } , } \end{array}
$$

where the last line uses (B.2) and $G \leq R$

Apply (B.4) with $w = \nabla f$ and $L = L _ { f }$ . Using div $\cdot ( \nabla f ) = \Delta f \leq \tau _ { f }$ and Cauchy–Schwarz, we obtain

$$
\begin{array} { r l } & { D _ { f } \leq t L _ { f } \Big [ \sqrt { D _ { f } } \big ( 2 R + \alpha \sqrt { I } + \sqrt { D _ { f } } \big ) + 2 \tau _ { f } + 2 ( \alpha - 1 ) \sqrt { D _ { f } } \sqrt { I } \Big ] } \\ & { \qquad \leq t L _ { f } \Big [ D _ { f } + ( 2 R + 3 \alpha \sqrt { I } ) \sqrt { D _ { f } } + 2 \tau _ { f } \Big ] . } \end{array}
$$

Young’s inequality gives

$$
t L _ { f } ( 2 R + 3 \alpha \sqrt { I } ) \sqrt { D _ { f } } \le \frac { 1 } { 4 } D _ { f } + 8 t ^ { 2 } L _ { f } ^ { 2 } R ^ { 2 } + 1 8 t ^ { 2 } L _ { f } ^ { 2 } \alpha ^ { 2 } I .
$$

For $t L _ { f } \leq 1 / 4$ , moving the terms involving $D _ { f }$ to the left therefore yields

$$
\frac { 1 } { 2 } D _ { f } \leq 2 t L _ { f } \tau _ { f } + 8 t ^ { 2 } L _ { f } ^ { 2 } R ^ { 2 } + 1 8 t ^ { 2 } L _ { f } ^ { 2 } \alpha ^ { 2 } I .
$$

In particular, we may use

$$
D _ { f } \leq 6 t L _ { f } \tau _ { f } + 1 6 t ^ { 2 } L _ { f } ^ { 2 } R ^ { 2 } + 3 6 t ^ { 2 } L _ { f } ^ { 2 } \alpha ^ { 2 } I .\tag{B.5}
$$

Since $\tau _ { f } \leq R ^ { 2 }$ and $t L _ { f } \leq 1 / 4$ , the preceding estimates also give

$$
\left( \mathbb { E } _ { \alpha } \left\| \nabla U _ { \lambda } ( X ) \right\| ^ { 2 } \right) ^ { 1 / 2 } \leq 6 R + 3 \alpha \sqrt { I } .
$$

The Moreau component. We next estimate

$$
D _ { g } : = \mathbb { E } _ { \alpha } \left\| \nabla g _ { \lambda } ( Y ) - \nabla g _ { \lambda } ( X ) \right\| ^ { 2 } .
$$

Apply (B.4) with $w = \nabla g _ { \lambda }$ and $L = \lambda ^ { - 1 }$ . The bounded-gradient property gives

$$
\| \nabla g _ { \lambda } ( Y ) - \nabla g _ { \lambda } ( X ) \| \leq 2 G .
$$

It follows that

$$
\begin{array} { r l } & { { D _ { g } } \leq \displaystyle \frac { t } { \lambda } \Big [ 2 G \big ( \mathbb { E } _ { \alpha } \| \nabla U _ { \lambda } ( X ) \| ^ { 2 } \big ) ^ { 1 / 2 } + 2 \mathbb { E } _ { \omega } \Delta g _ { \lambda } + 4 ( \alpha - 1 ) G \sqrt { I } \Big ] } \\ & { \quad \leq \displaystyle \frac { t } { \lambda } \Big [ 2 G ( 6 R + 3 \alpha \sqrt { I } ) + 2 ( 2 G R + 2 \alpha G \sqrt { I } ) + 4 ( \alpha - 1 ) G \sqrt { I } \Big ] . } \end{array}
$$

The second line uses the bound above and (B.3). Collecting terms gives

$$
D _ { g } \leq \frac { t } { \lambda } \big ( 1 6 G R + 1 4 \alpha G \sqrt { I } \big ) .\tag{B.6}
$$

Combining the two components. Since $\nabla U _ { \lambda } = \nabla f + \nabla g _ { \lambda }$ ，

$$
\begin{array} { r } { \mathbb { E } _ { \alpha } \| \nabla U _ { \lambda } ( Y ) - \nabla U _ { \lambda } ( X ) \| ^ { 2 } \leq 2 D _ { f } + 2 D _ { g } . } \end{array}
$$

Equations (B.5) and (B.6) therefore yield

$$
\begin{array} { r l } & { \mathbb { E } _ { \alpha } \left\| \nabla U _ { \lambda } ( Y ) - \nabla U _ { \lambda } ( X ) \right\| ^ { 2 } \leq 1 2 t L _ { f } \tau _ { f } + 3 2 t ^ { 2 } L _ { f } ^ { 2 } R ^ { 2 } + 7 2 t ^ { 2 } L _ { f } ^ { 2 } \alpha ^ { 2 } I } \\ & { \qquad + 3 2 ( t / \lambda ) G R + 2 8 ( t / \lambda ) \alpha G \sqrt { I } . } \end{array}\tag{B.7}
$$

## B.3 Proof of the density-ratio bound

This subsection proves (B.12), which implies (4.6).

Step 1. Evolution equation for $\nu _ { t } \mathbf { : }$ (B.8). Recall that

$$
Y = X - t \nabla U \lambda ( X ) + { \sqrt { 2 t } } Z .
$$

For $\phi \in C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { d } )$ and $0 < t < h$ , diferentiation under the expectation gives

$$
\frac { \mathrm { d } } { \mathrm { d } t } \int \phi ( y ) \nu _ { t } ( y ) \mathrm { d } y = \mathbb { E } \left. \nabla \phi ( Y ) , - \nabla U _ { \lambda } ( X ) + Z / \sqrt { 2 t } \right. .
$$

This diferentiation is valid on compact subintervals of $( 0 , h )$ because $\nabla \phi$ is bounded, the drift grows at most linearly, and X has finite moments by Lemma B.1.

For each coordinate $i ,$ Gaussian integration by parts, first conditional on $X$ , gives

$$
\mathbb { E } [ Z _ { i } \partial _ { i } \phi ( Y ) ] = \sqrt { 2 t } \mathbb { E } [ \partial _ { i i } \phi ( Y ) ] .
$$

Consequently,

$$
\begin{array} { r l } & { \displaystyle \frac { \mathrm { d } } { \mathrm { d } t } \int \phi ( y ) \nu _ { t } ( y ) \mathrm { d } y = \mathbb { E } \big [ \Delta \phi ( Y ) - \langle \nabla U _ { \lambda } ( X ) , \nabla \phi ( Y ) \rangle \big ] } \\ & { \quad \quad \quad = \displaystyle \int \big [ \Delta \phi ( y ) - \langle \nabla U _ { \lambda } ( y ) - e _ { t } ( y ) , \nabla \phi ( y ) \rangle \big ] \nu _ { t } ( y ) \mathrm { d } y , } \end{array}
$$

where the second equality uses

$$
\mathbb { E } [ \nabla U _ { \lambda } ( X ) \mid Y = y ] = \nabla U _ { \lambda } ( y ) - e _ { t } ( y ) .
$$

Thus, in the weak sense,

$$
\partial _ { t } \nu _ { t } = \Delta \nu _ { t } + \operatorname { d i v } ( \nu _ { t } \nabla U _ { \lambda } ) - \operatorname { d i v } ( \nu _ { t } e _ { t } ) .
$$

Since $\nabla \log \pi _ { \lambda } = - \nabla U _ { \lambda }$ , we have

$$
\nabla \nu _ { t } + \nu _ { t } \nabla U _ { \lambda } = \nu _ { t } \nabla \log \frac { \nu _ { t } } { \pi _ { \lambda } } .
$$

This proves

$$
\partial _ { t } \nu _ { t } = \operatorname { d i v } \left( \nu _ { t } \nabla \log \frac { \nu _ { t } } { \pi _ { \lambda } } \right) - \operatorname { d i v } ( \nu _ { t } e _ { t } ) .\tag{B.8}
$$

Step 2. Derivative of the R´enyi divergence: (B.9). By (2.27), with $u = \nu _ { t } / \pi _ { \lambda }$

$$
\mathcal { R } _ { \alpha } ( \nu _ { t } \| \pi _ { \lambda } ) = \frac { 1 } { \alpha - 1 } \log \int \boldsymbol { u } ^ { \alpha } \mathrm { d } \pi _ { \lambda } .
$$

We first compute the derivative of the integral on the right-hand side.

Choose $\chi _ { n } \in C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { d } )$ such that $0 \leq \chi _ { n } \leq 1 , \chi _ { n } = 1$ on the ball of radius $n , \chi _ { n } = 0$ outside the ball of radius 2n, and $\| \nabla \chi _ { n } \| _ { \infty } \leq C / n$ . Fix $0 < t _ { 0 } < t _ { 1 } < h$ . For each fixed $t \in [ t _ { 0 } , t _ { 1 } ]$ , Step 1 gives

$$
\int \phi \partial _ { t } \nu _ { t } \mathrm { d } x = - \int \left. \nabla \phi , s - e _ { t } \right. \nu _ { t } \mathrm { d } x , \qquad \phi \in C _ { c } ^ { \infty } ( \mathbb { R } ^ { d } ) .
$$

By Lemma B.1, u is positive and locally $C ^ { 1 , 2 } , \partial _ { t } \nu _ { t }$ is locally continuous, and $\nu _ { t } ( s - e _ { t } )$ is locally integrable. Approximation in $C ^ { 1 }$ , with support in a fixed compact set, therefore extends this identity to $\phi \in C _ { c } ^ { 1 } (  { \mathbb { R } } ^ { d } )$

At this fixed time, we may take $\phi = \chi _ { n } u ^ { \alpha - 1 }$ . Applying the chain rule first, and then the spatial identity above, gives

$$
\begin{array} { r l r } {  { \frac { \mathrm { d } } { \mathrm { d } t } \int \chi _ { n } u ^ { \alpha } \mathrm { d } \pi _ { \lambda } = \alpha \int \chi _ { n } u ^ { \alpha - 1 } \partial _ { t } \nu _ { t } \mathrm { d } x } } \\ & { } & { = - \alpha \int  \nabla ( \chi _ { n } u ^ { \alpha - 1 } ) , s - e _ { t }  \nu _ { t } \mathrm { d } x . } \end{array}
$$

Since

$$
\begin{array} { r } { \nabla \big ( \chi _ { n } u ^ { \alpha - 1 } \big ) = u ^ { \alpha - 1 } \nabla \chi _ { n } + ( \alpha - 1 ) \chi _ { n } u ^ { \alpha - 1 } s , } \end{array}
$$

we obtain

$$
\begin{array} { r l } { \displaystyle \frac { \mathrm { d } } { \mathrm { d } t } \int \chi _ { n } u ^ { \alpha } \mathrm { d } \pi _ { \lambda } = } & { - \alpha \big ( \alpha - 1 \big ) \int \chi _ { n } u ^ { \alpha } \left\| s \right\| ^ { 2 } \mathrm { d } \pi _ { \lambda } } \\ & { \phantom { \frac { \mathrm { d } } { \mathrm { d } t } \int \chi _ { n } u ^ { \alpha } \left( \boldsymbol { e } _ { t } , s \right) \mathrm { d } \pi _ { \lambda } } + \alpha \big ( \alpha - 1 \big ) \int \chi _ { n } u ^ { \alpha } \left. \boldsymbol { e } _ { t } , s \right. \mathrm { d } \pi _ { \lambda } } \\ & { - \alpha \displaystyle \int u ^ { \alpha } \left. s - \boldsymbol { e } _ { t } , \nabla \chi _ { n } \right. \mathrm { d } \pi _ { \lambda } . } \end{array}
$$

Keep $t _ { 0 }$ and $t _ { 1 }$ fixed and integrate the preceding identity in time. Writing $u ( t _ { j } ) = \nu _ { t _ { j } } / \pi _ { \lambda }$ at the endpoints gives

$$
\begin{array} { l } { \displaystyle \int \chi _ { n } u ( t _ { 1 } ) ^ { \alpha } \mathrm { d } \pi _ { \lambda } - \int \chi _ { n } u ( t _ { 0 } ) ^ { \alpha } \mathrm { d } \pi _ { \lambda } } \\ { \displaystyle \quad = - \alpha ( \alpha - 1 ) \int _ { t _ { 0 } } ^ { t _ { 1 } } \int \chi _ { n } u ^ { \alpha } \left. s \right. ^ { 2 } \mathrm { d } \pi _ { \lambda } \mathrm { d } t } \\ { \displaystyle \quad + \alpha ( \alpha - 1 ) \int _ { t _ { 0 } } ^ { t _ { 1 } } \int \chi _ { n } u ^ { \alpha } \left. e _ { t } , s \right. \mathrm { d } \pi _ { \lambda } \mathrm { d } t } \\ { \displaystyle \quad - \alpha \int _ { t _ { 0 } } ^ { t _ { 1 } } \int u ^ { \alpha } \left. s - e _ { t } , \nabla \chi _ { n } \right. \mathrm { d } \pi _ { \lambda } \mathrm { d } t . } \end{array}
$$

We now remove the spatial cutof while keeping $t _ { 0 }$ and $t _ { 1 }$ fixed. The last term satisfies

$$
\alpha \left| \int _ { t _ { 0 } } ^ { t _ { 1 } } \int u ^ { \alpha } \left. s - e _ { t } , \nabla \chi _ { n } \right. \mathrm { d } \pi _ { \lambda } \mathrm { d } t \right| \leq \frac { C \alpha } { n } \int _ { t _ { 0 } } ^ { t _ { 1 } } \int u ^ { \alpha } \bigl ( 1 + \| s \| ^ { 2 } + \| e _ { t } \| ^ { 2 } \bigr ) \mathrm { d } \pi _ { \lambda } \mathrm { d } t .
$$

The time integral on the right is finite and independent of n by Lemma B.1. Moreover,

$$
\boldsymbol { u } ^ { \alpha } \vert \langle e _ { t } , s \rangle \vert \leq \frac { 1 } { 2 } \boldsymbol { u } ^ { \alpha } \big ( \Vert e _ { t } \Vert ^ { 2 } + \Vert s \Vert ^ { 2 } \big ) ,
$$

so the same lemma provides an integrable bound for each of the other time integrands. We may therefore let $n \to \infty$ by dominated convergence, obtaining

$$
\begin{array} { l } { \displaystyle \int u ( t _ { 1 } ) ^ { \alpha } \mathrm { d } \pi _ { \lambda } - \int u ( t _ { 0 } ) ^ { \alpha } \mathrm { d } \pi _ { \lambda } } \\ { \displaystyle \quad = - \alpha ( \alpha - 1 ) \int _ { t _ { 0 } } ^ { t _ { 1 } } \int u ^ { \alpha } \| s \| ^ { 2 } \mathrm { d } \pi _ { \lambda } \mathrm { d } t } \\ { \displaystyle \qquad + \alpha ( \alpha - 1 ) \int _ { t _ { 0 } } ^ { t _ { 1 } } \int u ^ { \alpha } \langle e _ { t } , s \rangle \mathrm { d } \pi _ { \lambda } \mathrm { d } t . } \end{array}
$$

It remains to include the time endpoints. By Lemma B.1, both time integrands in the last identity belong to $L ^ { 1 } ( 0 , h )$ . The endpoint limits (D.10), proved in Section D.2, show that $\begin{array} { r } { t \mapsto \int u ( t ) ^ { \alpha } \mathrm { d } \pi _ { \lambda } } \end{array}$ is continuous at $t = 0$ and $t = h$ . We can therefore let $t _ { 0 } \downarrow 0$ or $t _ { 1 } \uparrow h$ in this identity, and then include both endpoints. The same identity consequently holds for every $0 \leq t _ { 0 } < t _ { 1 } \leq h$

Since its right-hand side is the time integral of an $L ^ { 1 } ( 0 , h )$ function, this identity proves that $\begin{array} { r } { t \mapsto \int u ( t ) ^ { \alpha } \mathrm { d } \pi _ { \lambda } } \end{array}$ is absolutely continuous on $[ 0 , h ]$ . Its derivative is therefore

$$
\frac { \mathrm { d } } { \mathrm { d } t } \int u ^ { \alpha } \mathrm { d } \pi _ { \lambda } = - \alpha ( \alpha - 1 ) \int u ^ { \alpha } \left\| s \right\| ^ { 2 } \mathrm { d } \pi _ { \lambda } + \alpha ( \alpha - 1 ) \int u ^ { \alpha } \left. e _ { t } , s \right. \mathrm { d } \pi _ { \lambda }
$$

for almost every $t \in ( 0 , h )$

Because $\textstyle \int u ^ { \alpha } \mathrm { d } \pi _ { \lambda } \geq 1$ , the R´enyi divergence is also absolutely continuous on $[ 0 , h ]$ . The chain rule gives

$$
\frac { \mathrm { d } } { \mathrm { d } t } \mathcal { R } _ { \alpha } ( \nu _ { t } \| \pi _ { \lambda } ) = \frac { \frac { \mathrm { d } } { \mathrm { d } t } \int u ^ { \alpha } \mathrm { d } \pi _ { \lambda } } { \left( \alpha - 1 \right) \int u ^ { \alpha } \mathrm { d } \pi _ { \lambda } } = - \alpha I + \alpha \mathbb { E } _ { \omega } \left. e _ { t } , s \right. .
$$

The weight defining $\mathbb { E } _ { \alpha }$ depends only on $Y _ { i }$ , so it leaves the conditional law of X given Y unchanged. Hence

$$
e _ { t } ( Y ) = \mathbb { E } _ { \alpha } [ \nabla U _ { \lambda } ( Y ) - \nabla U _ { \lambda } ( X ) \mid Y ] .
$$

Conditional Jensen’s inequality and the fact that the Y marginal is ω give

$$
\begin{array} { r } { \mathbb { E } _ { \omega } \left\| e _ { t } \right\| ^ { 2 } \leq \mathbb { E } _ { \alpha } \left\| \nabla U _ { \lambda } ( Y ) - \nabla U _ { \lambda } ( X ) \right\| ^ { 2 } . } \end{array}
$$

Using

$$
\langle e _ { t } , s \rangle \leq \frac { 1 } { 2 } \left. s \right. ^ { 2 } + \frac { 1 } { 2 } \left. e _ { t } \right. ^ { 2 } ,
$$

we conclude that, for almost every $t ,$

$$
\begin{array} { r l r } {  { \frac { \mathrm { d } } { \mathrm { d } t } \mathcal { R } _ { \alpha } ( \nu _ { t } \| \pi _ { \lambda } ) = - \alpha I + \alpha \mathbb { E } _ { \omega }  e _ { t } , s  } } \\ & { } & { \leq - \frac { \alpha } { 2 } I + \frac { \alpha } { 2 } \mathbb { E } _ { \omega } \| e _ { t } \| ^ { 2 } } \\ & { } & { \leq - \frac { \alpha } { 2 } I + \frac { \alpha } { 2 } \mathbb { E } _ { \alpha } \| \nabla U _ { \lambda } ( Y ) - \nabla U _ { \lambda } ( X ) \| ^ { 2 } . } \end{array}\tag{B.9}
$$

Step 3. Applying the drift bound: (B.10). Substituting (B.7) into (B.9) gives

$$
\begin{array} { l } { \displaystyle \frac { \mathrm { d } } { \mathrm { d } t } \mathcal { R } _ { \alpha } ( \nu _ { t } \| \pi _ { \lambda } ) \leq - \frac { \alpha } { 2 } I + 6 \alpha t L _ { f } \tau _ { f } + 1 6 \alpha t ^ { 2 } L _ { f } ^ { 2 } R ^ { 2 } } \\ { \displaystyle + 3 6 \alpha ^ { 3 } t ^ { 2 } L _ { f } ^ { 2 } I + 1 6 \alpha ( t / \lambda ) G R + 1 4 \alpha ^ { 2 } ( t / \lambda ) G \sqrt { I } . } \end{array}
$$

Since $\tau _ { f } \geq m$ , the first condition in (4.4) implies $h L _ { f } \le c / \alpha ^ { 2 }$ . Taking the universal constant c small enough gives, for $0 \leq t \leq h$

$$
3 6 \alpha ^ { 3 } t ^ { 2 } L _ { f } ^ { 2 } I \le \frac { \alpha } { 8 } I .
$$

Also, Young’s inequality gives

$$
1 4 \alpha ^ { 2 } ( t / \lambda ) G \sqrt { I } \leq \frac { \alpha } { 8 } I + 3 9 2 \alpha ^ { 3 } ( t / \lambda ) ^ { 2 } G ^ { 2 } .
$$

Combining these estimates proves

$$
\frac { \mathrm { d } } { \mathrm { d } t } \mathcal { R } _ { \alpha } ( \nu _ { t } \| \pi _ { \lambda } ) \leq - \frac { \alpha } { 4 } I + C \alpha \Big [ t L _ { f } \tau _ { f } + t ^ { 2 } L _ { f } ^ { 2 } R ^ { 2 } + ( t / \lambda ) G R + \alpha ^ { 2 } ( t / \lambda ) ^ { 2 } G ^ { 2 } \Big ] .\tag{B.10}
$$

Step 4. A lower bound for I: (B.11). By Lemma B.1,

$$
\int u ^ { \alpha } \mathrm { d } \pi _ { \lambda } < \infty , \int \left\| \nabla u ^ { \alpha / 2 } \right\| ^ { 2 } \mathrm { d } \pi _ { \lambda } = \frac { \alpha ^ { 2 } } { 4 } \int u ^ { \alpha } \left\| s \right\| ^ { 2 } \mathrm { d } \pi _ { \lambda } < \infty .
$$

We may therefore apply (2.26) to $w = u ^ { \alpha / 2 }$ . Dividing by $\int u ^ { \alpha } \mathrm { d } \pi _ { \lambda }$ gives

$$
\frac { \mathrm { E n t } _ { \pi _ { \lambda } } ( u ^ { \alpha } ) } { \int u ^ { \alpha } \mathrm { d } \pi _ { \lambda } } \leq \frac { 2 } { m } \frac { \int \left. \nabla u ^ { \alpha / 2 } \right. ^ { 2 } \mathrm { d } \pi _ { \lambda } } { \int u ^ { \alpha } \mathrm { d } \pi _ { \lambda } } = \frac { \alpha ^ { 2 } } { 2 m } I .
$$

We next bound the left-hand side from below. Since $\textstyle \int u \mathrm { d } \pi _ { \lambda } = 1$ , the definition of ω gives

$$
\mathbb E _ { \omega } [ u ^ { 1 - \alpha } ] = \frac { \int u \mathrm { d } \pi _ { \lambda } } { \int u ^ { \alpha } \mathrm { d } \pi _ { \lambda } } = \frac { 1 } { \int u ^ { \alpha } \mathrm { d } \pi _ { \lambda } } .
$$

The $3 \alpha / 2$ moment in Lemma B.1 ensures that $\mathbb { E } _ { \omega } | \log u | < \infty \colon u ^ { \alpha } |$ | log u| is bounded when $0 < u \leq 1$ ， and is at most $C _ { \alpha } u ^ { 3 \alpha / 2 }$ when $u \geq 1$ . Jensen’s inequality for the logarithm therefore gives

$$
\begin{array} { r l r } {  { ( 1 - \alpha ) \mathbb { E } _ { \omega } \log { u } = \mathbb { E } _ { \omega } [ \log ( u ^ { 1 - \alpha } ) ] } } \\ & { } & { \leq \log \mathbb { E } _ { \omega } [ u ^ { 1 - \alpha } ] = - \log \int u ^ { \alpha } \mathrm { d } \pi _ { \lambda } . } \end{array}
$$

Because $\alpha > 1$ , this implies

$$
\mathbb { E } _ { \omega } \log u \geq \frac { 1 } { \alpha - 1 } \log \int u ^ { \alpha } \mathrm { d } \pi _ { \lambda } = \mathcal { R } _ { \alpha } ( \nu _ { t } \| \pi _ { \lambda } ) .
$$

Consequently,

$$
\begin{array} { r l } {  { \frac { \operatorname { E n t } _ { \pi _ { \lambda } } ( u ^ { \alpha } ) } { \int u ^ { \alpha } \mathrm { d } \pi _ { \lambda } } = \alpha \mathbb { E } _ { \omega } \log u - \log \int u ^ { \alpha } \mathrm { d } \pi _ { \lambda } } } \\ & { \geq \alpha \mathcal { R } _ { \alpha } ( \nu _ { t } \| \pi _ { \lambda } ) - ( \alpha - 1 ) \mathcal { R } _ { \alpha } ( \nu _ { t } \| \pi _ { \lambda } ) } \\ & { = \mathcal { R } _ { \alpha } ( \nu _ { t } \| \pi _ { \lambda } ) \mathrm { . } } \end{array}
$$

Combining the upper and lower bounds proves

$$
I \geq \frac { 2 m } { \alpha ^ { 2 } } \mathcal { R } _ { \alpha } ( \nu _ { t } \| \pi _ { \lambda } ) .\tag{B.11}
$$

Step 5. Uniform $L ^ { \alpha }$ bound for the density ratio: (B.12) and $( 4 . 6 )$ . In (B.10), use (B.11) and bound t by h in the four nonnegative terms. Multiplying the resulting inequality by $e ^ { m t / ( 2 \alpha ) }$ and integrating from 0 to t gives, after enlarging the universal constant $C ,$

$$
\begin{array} { l } { \displaystyle \mathcal { R } _ { \alpha } ( \nu _ { t } \| \pi _ { \lambda } ) \le e ^ { - m t / ( 2 \alpha ) } \mathcal { R } _ { \alpha } ( \nu _ { 0 } \| \pi _ { \lambda } ) } \\ { \displaystyle \qquad + \frac { C \alpha ^ { 2 } } { m } \big ( 1 - e ^ { - m t / ( 2 \alpha ) } \big ) \Big [ h L _ { f } \tau _ { f } + h ^ { 2 } L _ { f } ^ { 2 } R ^ { 2 } + ( h / \lambda ) G R + \alpha ^ { 2 } ( h / \lambda ) ^ { 2 } G ^ { 2 } \Big ] . } \end{array}
$$

Taking $t = h$ and using $\nu _ { h } = \nu _ { 0 }$ , we move $e ^ { - m h / ( 2 \alpha ) } \mathcal { R } _ { \alpha } ( \nu _ { 0 } \| \pi _ { \lambda } )$ to the left-hand side and cancel the positive factor $1 - e ^ { - m h / ( 2 \alpha ) }$ . This bounds $\mathcal { R } _ { \alpha } ( \nu _ { 0 } \| \pi _ { \lambda } )$ by $C \alpha ^ { 2 } / m$ times the bracket above. Substituting this bound back into the inequality for general t proves

$$
\operatorname* { s u p } _ { 0 \leq t \leq h } \mathcal { R } _ { \alpha } ( \nu _ { t } \| \pi _ { \lambda } ) \leq \frac { C \alpha ^ { 2 } } { m } \Big [ h L _ { f } \tau _ { f } + h ^ { 2 } L _ { f } ^ { 2 } R ^ { 2 } + ( h / \lambda ) G R + \alpha ^ { 2 } ( h / \lambda ) ^ { 2 } G ^ { 2 } \Big ] .\tag{B.12}
$$

The conditions in (4.4), together with $R ^ { 2 } \geq m$ , give

$$
\begin{array} { r l } & { \quad \quad \frac { \alpha ^ { 2 } } { m } h L _ { f } \tau _ { f } \leq c , } \\ & { \quad \quad \frac { \alpha ^ { 2 } } { m } h ^ { 2 } L _ { f } ^ { 2 } R ^ { 2 } = \left( \frac { \alpha h L _ { f } R } { \sqrt { m } } \right) ^ { 2 } \leq c ^ { 2 } , } \\ & { \quad \quad \displaystyle \frac { \alpha ^ { 2 } } { m } ( h / \lambda ) G R \leq c , } \\ & { \quad \quad \displaystyle \frac { \alpha ^ { 4 } } { m } ( h / \lambda ) ^ { 2 } G ^ { 2 } = \frac { m } { R ^ { 2 } } \left( \frac { \alpha ^ { 2 } } { m } ( h / \lambda ) G R \right) ^ { 2 } \leq c ^ { 2 } . } \end{array}
$$

Taking the universal constant c small enough makes the right-hand side of (B.12) at most $1 / 4$ Finally, (2.27) gives

$$
\operatorname* { s u p } _ { 0 \leq t \leq h } \| \nu _ { t } / \pi _ { \lambda } \| _ { L ^ { \alpha } ( \pi _ { \lambda } ) } = \exp \left\{ \frac { \alpha - 1 } { \alpha } \operatorname* { s u p } _ { 0 \leq t \leq h } \mathcal { R } _ { \alpha } ( \nu _ { t } \| \pi _ { \lambda } ) \right\} \leq e ^ { 1 / 4 } .
$$

This is (4.6).

All quantitative constants above are independent of the auxiliary smoothing scale. Passing to the limit as described in Appendix D gives (4.6) for the original potential. This completes the proof of Lemma 4.2.

## C Poisson Regularity and Negative Sobolev Norms

This appendix proves Lemma 2.2 and the test-function equivalence (2.24). Throughout, λ is fixed, $U _ { \lambda } \in C ^ { 1 , 1 } ( \mathbb { R } ^ { d } )$ , and m $I \preceq \nabla ^ { 2 } U _ { \lambda } \preceq L _ { \lambda } I$ almost everywhere.

## C.1 Existence and second-order energy

Proof of the existence and energy assertions in Lemma 2.2. By (2.17), the gradient norm is a complete Hilbert norm on the mean-zero subspace of $H ^ { 1 } ( \pi _ { \lambda } )$ . For $\phi \in L ^ { 2 } ( \pi _ { \lambda } )$ , the functional

$$
v \longmapsto - \int \left( \phi - \int \phi { \mathrm { d } } \pi _ { \lambda } \right) v { \mathrm { d } } \pi _ { \lambda }
$$

is continuous in that norm, with norm at most $\begin{array} { r } { m ^ { - 1 / 2 } \left. \phi - \int \phi \mathrm { d } \pi _ { \lambda } \right. _ { L ^ { 2 } ( \pi _ { \lambda } ) } } \end{array}$ . The Riesz representation theorem gives a unique mean-zero $\psi \in H ^ { 1 } ( \pi _ { \lambda } )$ satisfying (2.19). The identity extends from meanzero tests to every $v \in H ^ { 1 } ( \pi _ { \lambda } )$ because both sides are unchanged when a constant is added to v.

On compact sets, $\pi _ { \lambda }$ is bounded above and below by positive constants and $\nabla U _ { \lambda }$ is bounded. The weak equation therefore gives, in the sense of distributions,

$$
\Delta \psi = \phi - \int \phi \mathrm { d } \pi _ { \lambda } + \langle \nabla U _ { \lambda } , \nabla \psi \rangle \in L _ { \mathrm { l o c } } ^ { 2 } ( \mathbb { R } ^ { d } ) .
$$

For $\eta \in C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { d } )$ , the function ηψ belongs to ordinary $H ^ { 1 } ( \mathbb { R } ^ { d } )$ and

$$
\Delta ( \eta \psi ) = \eta \Delta \psi + 2 \left. \nabla \eta , \nabla \psi \right. + \psi \Delta \eta \in L ^ { 2 } ( \mathbb { R } ^ { d } ) .
$$

Set $u = \eta \psi$ and let $u _ { \varepsilon } = u * \varrho _ { \varepsilon }$ , where $\varrho _ { \varepsilon } ( x ) = \varepsilon ^ { - d } \varrho ( x / \varepsilon ) , \varrho \in C _ { c } ^ { \infty } ( B ( 0 , 1 ) ) , \varrho \ge 0$ , and $\textstyle \int { \varrho \mathrm { d } x } = 1$ Then $u _ { \varepsilon } \in C _ { c } ^ { \infty } ( \mathbb { R } ^ { d } ) , u _ { \varepsilon } \to$ u in $H ^ { 1 } ( \mathbb { R } ^ { d } )$ , and

$$
\Delta u _ { \varepsilon } = ( \Delta u ) * \varrho _ { \varepsilon } \longrightarrow \Delta u \mathrm { i n } L ^ { 2 } ( \mathbb { R } ^ { d } ) .
$$

For every $w \in C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { d } )$ , two integrations by parts give

$$
\int \left\| \nabla ^ { 2 } w \right\| _ { F } ^ { 2 } \mathrm { d } x = \sum _ { i , j } \int ( \partial _ { i i } w ) ( \partial _ { j j } w ) \mathrm { d } x = \int | \Delta w | ^ { 2 } \mathrm { d } x .
$$

Applying this identity to $w = u _ { \varepsilon } - u _ { \varepsilon ^ { \prime } }$ shows that $\nabla ^ { 2 } u _ { \varepsilon }$ is Cauchy in $L ^ { 2 } ( \mathbb R ^ { d } )$ . Its limit is the weak Hessian of $u ,$ as follows by passing to the limit in

$$
\int \partial _ { i j } u _ { \varepsilon } \zeta \mathrm { d } x = \int u _ { \varepsilon } \partial _ { i j } \zeta \mathrm { d } x , \qquad \zeta \in C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { d } ) .
$$

Thus $u \in H ^ { 2 } ( \mathbb { R } ^ { d } )$ . Since η was arbitrary, $\psi \in H _ { \mathrm { l o c } } ^ { 2 } ( \mathbb { R } ^ { d } )$

Now let $\phi \in C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { d } )$ . For $u \in C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { d } )$ , diferentiation of the generator in the weak sense gives

$$
\nabla \mathcal { L } _ { \lambda } u = \mathcal { L } _ { \lambda } ( \nabla u ) - ( \nabla ^ { 2 } U _ { \lambda } ) \nabla u ,
$$

where $\mathcal { L } _ { \lambda }$ acts componentwise on $\nabla u$ . Weighted integration by parts then yields

$$
\begin{array} { r l } & { \displaystyle \int ( \mathcal { L } _ { \lambda } u ) ^ { 2 } \mathrm { d } \pi _ { \lambda } = - \int \langle \nabla u , \nabla \mathcal { L } _ { \lambda } u \rangle \mathrm { d } \pi _ { \lambda } } \\ & { \quad \quad \quad = \displaystyle \int \left\| \nabla ^ { 2 } u \right\| _ { F } ^ { 2 } \mathrm { d } \pi _ { \lambda } + \int \nabla u ^ { \top } ( \nabla ^ { 2 } U _ { \lambda } ) \nabla u \mathrm { d } \pi _ { \lambda } . } \end{array}\tag{C.1}
$$

Only the bounded weak derivative of $\nabla U _ { \lambda }$ is used. For a compactly supported $u \in H ^ { 2 } ( \mathbb { R } ^ { d } )$ , let $u _ { \varepsilon } = u * \varrho _ { \varepsilon }$ . Mollification commutes with weak derivatives through order two, so $u _ { \varepsilon } \to u$ in $H ^ { 2 } ( \mathbb R ^ { d } )$ For $0 < \varepsilon < 1$ , these functions are supported in a fixed compact set, on which $\pi _ { \lambda }$ and $\nabla U _ { \lambda }$ are bounded. Consequently,

$$
\| \mathcal { L } _ { \lambda } u _ { \varepsilon } - \mathcal { L } _ { \lambda } u \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } + \big \| \nabla ^ { 2 } u _ { \varepsilon } - \nabla ^ { 2 } u \big \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } + \| \nabla u _ { \varepsilon } - \nabla u \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } \longrightarrow 0 .
$$

Together with the boundedness of $\nabla ^ { 2 } U _ { \lambda }$ , these convergences justify passage to the limit in (C.1).   
The identity therefore holds for compactly supported $H ^ { 2 } ( \mathbb R ^ { d } )$ functions.

Choose smooth cutofs $0 \leq \chi _ { n } \leq 1$ , equal to one for $\| x \| \leq n$ and zero for $\left\| x \right\| \geq 2 n$ , such that $\| \nabla \chi _ { n } \| \leq C / n$ and $\left. \nabla ^ { 2 } \chi _ { n } \right. _ { F } \leq C _ { d } / n ^ { 2 }$ . Local regularity implies that $\chi _ { n } \psi$ is a compactly supported $H ^ { 2 }$ function. Since $\nabla U _ { \lambda }$ has at most linear growth, $\mathcal { L } _ { \lambda \chi _ { n } }$ is bounded uniformly in n on its annular support, with the model parameters fixed. Consequently,

$$
\begin{array} { r l } & { \mathcal { L } _ { \lambda } ( \chi _ { n } \psi ) = \chi _ { n } \left( \phi - \displaystyle \int \phi { \mathrm { d } } \pi _ { \lambda } \right) + 2 \left. \nabla \chi _ { n } , \nabla \psi \right. + \psi \mathcal { L } _ { \lambda } \chi _ { n } } \\ & { \qquad \longrightarrow \phi - \displaystyle \int \phi { \mathrm { d } } \pi _ { \lambda } = \mathcal { L } _ { \lambda } \psi \qquad \mathrm { i n ~ } L ^ { 2 } ( \pi _ { \lambda } ) . } \end{array}
$$

The first term converges by dominated convergence; the other two tend to zero because $\psi , \nabla \psi \in$ $L ^ { 2 } ( \pi _ { \lambda } )$ and the derivatives of $\chi _ { n }$ are supported where $\| x \| \geq n$

Apply (C.1) to $\chi _ { n } \psi$ . Since $\nabla ^ { 2 } U _ { \lambda } \succeq 0$ , both energy terms are nonnegative. Moreover, $\chi _ { n } \psi = \psi$ on $B ( 0 , n )$ , so their derivatives agree there. Fatou’s lemma and the convergence of $\mathcal { L } _ { \lambda } ( \chi _ { n } \psi )$ give

$$
\begin{array} { r l } { \left. { \int \big \| \nabla ^ { 2 } \psi \big \| _ { F } ^ { 2 } \ \mathrm { d } \pi _ { \lambda } + \int \nabla \psi ^ { \top } \nabla ^ { 2 } U _ { \lambda } \nabla \psi \mathrm { d } \pi _ { \lambda } } \right.} \\ & { \leq \operatorname* { l i m i n f } _ { n  \infty } \int | \mathcal { L } _ { \lambda } ( \chi _ { n } \psi ) | ^ { 2 } \mathrm { d } \pi _ { \lambda } } \\ & { = \displaystyle \int \left| \phi - \int \phi \mathrm { d } \pi _ { \lambda } \right| ^ { 2 } \mathrm { d } \pi _ { \lambda } < \infty . } \end{array}
$$

In particular, $\nabla ^ { 2 } \psi \in L ^ { 2 } ( \pi _ { \lambda } )$ . This proves the first inequality in (2.20); the second follows from (2.17) applied to ϕ. □

## C.2 Approximation by compactly supported tests

Proof of the approximation assertion in Lemma 2.2. Let $\phi \in C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { d } )$ and use the cutofs from the preceding proof. The convergence of $\mathcal { L } _ { \lambda } ( \chi _ { n } \psi )$ to ${ \mathcal { L } } _ { \lambda } \psi$ in $L ^ { 2 } ( \pi _ { \lambda } )$ has already been established. Now

that (2.20) gives global weighted integrability of $\nabla ^ { 2 } \psi$ , the product formula

$$
\begin{array} { r l } & { \nabla ^ { 2 } ( \chi _ { n } \psi ) - \nabla ^ { 2 } \psi = ( \chi _ { n } - 1 ) \nabla ^ { 2 } \psi + \nabla \chi _ { n } \otimes \nabla \psi } \\ & { \qquad + \nabla \psi \otimes \nabla \chi _ { n } + \psi \nabla ^ { 2 } \chi _ { n } } \end{array}
$$

implies convergence to zero in $L ^ { 2 } ( \pi _ { \lambda } )$ . Likewise,

$$
\nabla ( \chi _ { n } \psi ) - \nabla \psi = ( \chi _ { n } - 1 ) \nabla \psi + \psi \nabla \chi _ { n } \longrightarrow 0 \quad \mathrm { i n ~ } L ^ { 2 } ( \pi _ { \lambda } ) .
$$

For each fixed $n ,$ mollify $\chi _ { n } \psi$ . The mollifications converge in ordinary $H ^ { 2 } ( \mathbb R ^ { d } )$ and remain supported in a fixed compact set. On that set, $\pi _ { \lambda }$ is bounded above and below by positive constants and $\nabla U _ { \lambda }$ is bounded. Thus they also converge in each of the three norms in (2.21). Choose the mollification scale so that the sum of these three errors relative to $\chi _ { n } \psi$ is at most $1 / n$ . The resulting sequence $\psi _ { n } \in C _ { c } ^ { \infty } ( \mathbb { R } ^ { d } )$ satisfies (2.21). □

## C.3 Equivalent test classes for the negative Sobolev norm

Proof of (2.24). Let $\sigma$ be an integrable signed density with $\textstyle \int \sigma \mathrm { d } x = 0$ and $\sigma / \pi _ { \lambda } \in L ^ { 2 } ( \pi _ { \lambda } )$ . We first compare compactly supported smooth tests with $H ^ { 1 } ( \pi _ { \lambda } )$ tests.

For $v \in H ^ { 1 } ( \pi _ { \lambda } )$ , choose smooth cutofs $0 \leq \chi _ { n } \leq 1$ , equal to one on the ball of radius $n _ { : }$ supported in the ball of radius $2 n .$ , and satisfying $\| \nabla \chi _ { n } \| \leq C / n$ . Then

$$
\begin{array} { c } { \displaystyle \| \chi _ { n } v - v \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } \longrightarrow 0 , } \\ { \displaystyle \| \nabla ( \chi _ { n } v ) - \nabla v \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } \le \| ( \chi _ { n } - 1 ) \nabla v \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } + \displaystyle \frac { C } { n } \| v \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } \longrightarrow 0 . } \end{array}
$$

On each compact set, $\pi _ { \lambda }$ is bounded above and below by positive constants. Mollifying $\chi _ { n } v$ and choosing the smoothing scales diagonally therefore gives $v _ { n } \in C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { d } )$ with

$$
\| v _ { n } - v \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } + \| \nabla v _ { n } - \nabla v \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } \longrightarrow 0 .
$$

The pairing is continuous under this convergence, since

$$
\left| \int ( v _ { n } - v ) \sigma \mathrm { d } x \right| \leq \| \sigma / \pi _ { \lambda } \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } \| v _ { n } - v \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } \longrightarrow 0 .
$$

$\mathrm { I f ~ } \| \nabla v \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } \leq 1$ , replace $v _ { n }$ by

$$
\frac { v _ { n } } { \operatorname* { m a x } \{ 1 , \| \nabla v _ { n } \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } \} } .
$$

This preserves convergence to v in $H ^ { 1 } ( \pi _ { \lambda } )$ and makes every approximation admissible in (2.22). Thus the supremum over $H ^ { 1 } ( \pi _ { \lambda } )$ equals the supremum over $C _ { c } ^ { \infty } ( \mathbb { R } ^ { d } )$

It remains to compare with $C ^ { 1 }$ tests. Let $v \in C ^ { 1 } (  { \mathbb { R } } ^ { d } )$ satisfy $\| \nabla v \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } < \infty$ . To verify that $v \in H ^ { 1 } ( \pi _ { \lambda } )$ , define the bounded truncations

$$
v _ { k } : = \operatorname* { m a x } \{ - k , \operatorname* { m i n } \{ v , k \} \} .
$$

They belong to $H ^ { 1 } ( \pi _ { \lambda } )$ and satisfy $\| \nabla v _ { k } \| \leq \| \nabla v \|$ almost everywhere. The Poincar´e inequality gives

$$
\int \left| v _ { k } - \int v _ { k } \mathrm { d } \pi _ { \lambda } \right| ^ { 2 } \mathrm { d } \pi _ { \lambda } \leq \frac { 1 } { m } \left\| \nabla v \right\| _ { L ^ { 2 } ( \pi _ { \lambda } ) } ^ { 2 } .
$$

For $k > \operatorname* { s u p } _ { \| x \| \leq 1 } | v ( x ) |$ , one has $v _ { k } = v$ on the unit ball. Restricting the preceding variance bound to that ball gives

$$
\left| \int v _ { k } \mathrm { d } \pi _ { \lambda } \right| \leq \operatorname* { s u p } _ { \| x \| \leq 1 } | v ( x ) | + \frac { \| \nabla v \| _ { L ^ { 2 } ( \pi _ { \lambda } ) } } { \sqrt { m \pi _ { \lambda } ( \{ x : \| x \| \leq 1 \} ) } } .
$$

The denominator is positive. Hence the means of $v _ { k } ,$ and therefore their $L ^ { 2 } ( \pi _ { \lambda } )$ norms, are uniformly bounded. Since $v _ { k } \to v$ pointwise, Fatou’s lemma yields $v \in L ^ { 2 } ( \pi _ { \lambda } )$ . Together with the assumed gradient bound, this proves $v \in H ^ { 1 } ( \pi _ { \lambda } )$

Consequently, every admissible $C ^ { 1 }$ test is an admissible $H ^ { 1 } ( \pi _ { \lambda } )$ test. Since $C _ { c } ^ { \infty } ( \mathbb { R } ^ { d } ) \subset C ^ { 1 } ( \mathbb { R } ^ { d } )$ and the $C _ { c } ^ { \infty }$ and $H ^ { 1 }$ suprema have already been shown equal, the $C ^ { 1 }$ supremum has the same value. This proves (2.24). □

## D Weak Regularity and Integrability Before Dissipation

This appendix supplies the technical arguments used in Appendix B. We first verify the properties of the auxiliary mollification. We then prove the integrability statement Lemma B.1 independently of the dissipation calculation. Finally, Section D.3 passes the quantitative estimate to the original potential.

## D.1 Integration by parts and smooth approximation

We first justify the integrations by parts in Section A.1. Let Φ be a locally Lipschitz vector field such that Φ and its weak divergence have at most polynomial growth. Choose $\chi _ { n } \in C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { d } )$ with $0 \leq \chi _ { n } \leq 1$ , equal to one on the ball of radius $n ,$ supported on the ball of radius 2n, and satisfying $\| \nabla \chi _ { n } \| _ { \infty } \leq C / n$ . Integration by parts with compact support gives

$$
\int \chi _ { n } \operatorname { d i v } \Phi \operatorname { d } \pi _ { \lambda } = \int \chi _ { n } \left. \Phi , \nabla U _ { \lambda } \right. \operatorname { d } \pi _ { \lambda } - \int \left. \nabla \chi _ { n } , \Phi \right. \operatorname { d } \pi _ { \lambda } .
$$

Strong convexity gives Gaussian tails for $\pi _ { \lambda } .$ , and $\nabla U _ { \lambda }$ grows at most linearly. The first two terms therefore converge by dominated convergence, while

$$
\left| \int \langle \nabla \chi _ { n } , \Phi \rangle ~ \mathrm { d } \pi _ { \lambda } \right| \leq { \frac { C } { n } } \int \left\| \Phi \right\| ~ \mathrm { d } \pi _ { \lambda } \longrightarrow 0 .
$$

Thus

$$
\int \mathrm { d i v } \Phi \mathrm { d } \pi _ { \lambda } = \int \left. \Phi , \nabla U _ { \lambda } \right. \mathrm { d } \pi _ { \lambda } .
$$

The vector fields used in Section A.1 satisfy these conditions.

Consider the mollified potentials defined in Section B.1. Convolution with the nonnegative kernel $\varrho _ { \delta }$ preserves the structural bounds:

$$
\begin{array} { r l } & { m I \preceq \nabla ^ { 2 } f ^ { ( \delta ) } \preceq L _ { f } I , \qquad \mathrm { t r } \big ( \nabla ^ { 2 } f ^ { ( \delta ) } \big ) \leq \tau _ { f } , } \\ & { \quad \left\| \nabla g _ { \lambda } ^ { ( \delta ) } \right\| \leq G , \qquad 0 \preceq \nabla ^ { 2 } g _ { \lambda } ^ { ( \delta ) } \preceq \lambda ^ { - 1 } I . } \end{array}
$$

Thus the same parameters and step restrictions apply for every $\delta > 0$ . For each fixed $\delta ,$ the mollified potentials are smooth and all their derivatives of order at least two are bounded. These derivative bounds may depend on $\delta .$

The function $g _ { \lambda } ^ { ( \delta ) }$ need not be the Moreau envelope of the original $g .$ . The estimates in Appendix B use the convexity, gradient, and Hessian bounds displayed above. The convergence of the regularized potentials and their associated laws is treated in Section D.3.

## D.2 Finiteness before the R´enyi calculation

Proof of Lemma B.1. Fix $\alpha \geq 2 , \lambda ,$ , and h satisfying the restrictions of Lemma 4.2, and fix an auxiliary scale $\delta > 0$ . Throughout this proof, the potentials, densities, kernels, and maps are the regularized objects defined in Section B.1; their superscripts (δ) are omitted. In particular, the function $f$ and its minimizer introduced below belong to this regularized problem. In this subsection only, $C$ and its subscripted versions may depend on the fixed problem data and on $\alpha , \lambda , h , \delta$ . The step-size constant c remains universal.

We first establish an exponential moment of the invariant law, then derive density and gradient bounds uniform in $t \in [ 0 , h ]$ , and finally control the weighted score and joint moments. The finite bounds obtained in this proof may depend on the fixed parameters and on $\delta .$ Their values are used only to justify the analytic operations in Appendix B and do not enter the quantitative step restrictions. For $h L _ { \lambda } < c < 1$ , the map $T _ { h }$ is $( 1 - m h )$ -Lipschitz. The Euler kernel therefore contracts $\mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ in $W _ { 2 }$ and has a unique invariant law.

An exponential moment of the invariant law. We prove the exponential moment bound (D.5) for $\widehat { \pi } _ { \lambda , h }$ . Let $x _ { f } = \mathrm { a r g }$ min f, $F = f - f ( x _ { f } )$ , and $\eta = 1 - 1 / ( 8 \alpha )$ , only within this subsection. Strong convexity and smoothness give

$$
F ( x ) \geq { \frac { m } { 2 } } \left\| x - x _ { f } \right\| ^ { 2 } , \qquad 2 m F ( x ) \leq \left\| \nabla F ( x ) \right\| ^ { 2 } \leq 2 L _ { f } F ( x ) .\tag{D.1}
$$

For

$$
Y = x - h \nabla U _ { \lambda } ( x ) + \sqrt { 2 h } Z ,
$$

the $L _ { f }$ -smoothness of $f$ gives

$$
F ( Y ) - F ( x ) \leq \langle \nabla f ( x ) , Y - x \rangle + { \frac { L _ { f } } { 2 } } \left. Y - x \right. ^ { 2 } .
$$

Substituting $Y - x = - h \nabla U _ { \lambda } ( x ) + \sqrt { 2 h } Z$ and expanding the square yields

$$
\begin{array} { l } { F ( \boldsymbol { Y } ) - F ( \boldsymbol { x } ) \leq \displaystyle - h \left. \nabla f ( \boldsymbol { x } ) , \nabla U _ { \lambda } ( \boldsymbol { x } ) \right. + \frac { h ^ { 2 } L _ { f } } { 2 } \left. \nabla U _ { \lambda } ( \boldsymbol { x } ) \right. ^ { 2 } } \\ { \displaystyle + \sqrt { 2 h } \left. \nabla f ( \boldsymbol { x } ) - h L _ { f } \nabla U _ { \lambda } ( \boldsymbol { x } ) , Z \right. + h L _ { f } \left. Z \right. ^ { 2 } . } \end{array}
$$

For $\xi \in \mathbb { R } ^ { d }$ and $\theta < 1 / 2$ , completing the square gives

$$
- \frac { 1 } { 2 } \left\| z \right\| ^ { 2 } + \langle \xi , z \rangle + \theta \left\| z \right\| ^ { 2 } = - \frac { 1 - 2 \theta } { 2 } \left\| z - \frac { \xi } { 1 - 2 \theta } \right\| ^ { 2 } + \frac { \left\| \xi \right\| ^ { 2 } } { 2 ( 1 - 2 \theta ) } .
$$

Integrating against the standard Gaussian density therefore gives

$$
\mathbb { E } e ^ { \langle \xi , Z \rangle + \theta \| Z \| ^ { 2 } } = ( 1 - 2 \theta ) ^ { - d / 2 } \exp \left\{ \frac { \| \xi \| ^ { 2 } } { 2 ( 1 - 2 \theta ) } \right\} .\tag{D.2}
$$

The step-size restrictions ensure $2 \eta h L _ { f } < 1$ . Apply (D.2) with

$$
\theta = \eta h L _ { f } , \qquad \xi = \eta \sqrt { 2 h } \big ( \nabla f ( x ) - h L _ { f } \nabla U _ { \lambda } ( x ) \big ) .
$$

Since

$$
\frac { Q _ { \lambda , h } e ^ { \eta F } ( x ) } { e ^ { \eta F ( x ) } } = \mathbb { E } e ^ { \eta ( F ( Y ) - F ( x ) ) } ,
$$

we obtain

$$
\begin{array} { l } { \displaystyle \log \frac { Q _ { \lambda , h } e ^ { \eta F } ( \boldsymbol { x } ) } { \epsilon ^ { \eta F ( \boldsymbol { x } ) } } \le - \frac { d } { 2 } \log ( 1 - 2 \eta h L _ { f } ) - \eta h \left. \nabla f ( \boldsymbol { x } ) , \nabla U _ { \lambda } ( \boldsymbol { x } ) \right. } \\ { \displaystyle \quad + \frac { \eta h ^ { 2 } L _ { f } } { 2 } \left\| \nabla U _ { \lambda } ( \boldsymbol { x } ) \right\| ^ { 2 } + \frac { \eta ^ { 2 } h } { 1 - 2 \eta h L _ { f } } \left\| \nabla f ( \boldsymbol { x } ) - h L _ { f } \nabla U _ { \lambda } ( \boldsymbol { x } ) \right\| ^ { 2 } } \\ { \displaystyle = - \frac { d } { 2 } \log ( 1 - 2 \eta h L _ { f } ) } \\ { \displaystyle \quad + \frac { \eta h } { 1 - 2 \eta h L _ { f } } \left[ \eta \left\| \nabla f ( \boldsymbol { x } ) \right\| ^ { 2 } - \left. \nabla f ( \boldsymbol { x } ) , \nabla U _ { \lambda } ( \boldsymbol { x } ) \right. + \frac { h L _ { f } } { 2 } \left\| \nabla U _ { \lambda } ( \boldsymbol { x } ) \right\| ^ { 2 } \right] . } \end{array}
$$

Finally, substitute $\nabla U _ { \lambda } = \nabla f + \nabla g _ { \lambda }$ and use $\| \nabla g _ { \lambda } \| \le G$ to obtain

$$
\begin{array} { l } { \log \displaystyle \frac { Q _ { \lambda , h } e ^ { \eta F } ( \boldsymbol { x } ) } { e ^ { \eta F ( \boldsymbol { x } ) } } \le - \displaystyle \frac { d } { 2 } \log ( 1 - 2 \eta h L _ { f } ) } \\ { \displaystyle \qquad + \frac { \eta h } { 1 - 2 \eta h L _ { f } } \Big [ ( - ( 1 - \eta ) + h L _ { f } / 2 ) \left. \nabla f ( \boldsymbol { x } ) \right. ^ { 2 } } \\ { \displaystyle \qquad - \left( 1 - h L _ { f } \right) \left. \nabla f ( \boldsymbol { x } ) , \nabla g _ { \lambda } ( \boldsymbol { x } ) \right. + ( h L _ { f } / 2 ) G ^ { 2 } \Big ] . } \end{array}\tag{D.3}
$$

The first condition in (4.4), together with $\tau _ { f } \geq m$ , gives $h L _ { f } \le c / \alpha ^ { 2 }$ . Since $1 - \eta = 1 / ( 8 \alpha )$ , taking the universal constant c small enough ensures

$$
h L _ { f } \leq \frac { 1 - \eta } { 2 } , \qquad 1 - 2 \eta h L _ { f } \geq \frac { 1 } { 2 } .
$$

Hence

$$
- ( 1 - \eta ) + \frac { h L _ { f } } { 2 } \leq - \frac { 3 } { 4 } ( 1 - \eta ) .
$$

For the mixed term, Young’s inequality gives

$$
- ( 1 - h L _ { f } ) \left. \nabla f ( x ) , \nabla g _ { \lambda } ( x ) \right. \leq G \left\| \nabla f ( x ) \right\| \leq \frac { 1 - \eta } { 4 } \left\| \nabla f ( x ) \right\| ^ { 2 } + \frac { G ^ { 2 } } { 1 - \eta } .
$$

The bracket in (D.3) is therefore at most

$$
- \frac { 1 - \eta } { 2 } \left\| \nabla f ( x ) \right\| ^ { 2 } + \left( \frac { 1 } { 1 - \eta } + \frac { h L _ { f } } { 2 } \right) G ^ { 2 } .
$$

Also, $- \log ( 1 - z ) \leq 2 z$ for $0 \le z \le 1 / 2$ gives

$$
- \frac { d } { 2 } \log ( 1 - 2 \eta h L _ { f } ) \leq 2 \eta d h L _ { f } .
$$

Using $\| \nabla f ( x ) \| ^ { 2 } \geq 2 m F ( x )$ , we conclude that

$$
\log \frac { Q _ { \lambda , h } e ^ { \eta F } ( x ) } { e ^ { \eta F ( x ) } } \leq - \eta ( 1 - \eta ) m h F ( x ) + C _ { \alpha } h .
$$

Thus, with $c _ { \alpha } = \eta ( 1 - \eta ) > 0$

$$
{ Q } _ { { \lambda } , h } e ^ { \eta { \cal F } } \leq e ^ { \eta { \cal F } } \exp \{ - c _ { \alpha } m h { \cal F } + C _ { \alpha } h \} .\tag{D.4}
$$

Choose a finite $K > 0$ such that $c _ { \alpha } m K \geq C _ { \alpha } + 1$ . For $F ( x ) \geq K , ( \mathrm { D } . 4 )$ gives

$$
Q _ { \lambda , h } e ^ { \eta F } ( x ) \leq e ^ { - h } e ^ { \eta F ( x ) } .
$$

For $F ( x ) < K$ , it gives

$$
Q _ { \lambda , h } e ^ { \eta F } ( x ) \leq e ^ { \eta K + C _ { \alpha } h } .
$$

Combining the two bounds, we obtain, for every $x ,$

$$
\begin{array} { r } { Q _ { \lambda , h } e ^ { \eta F } ( x ) \le e ^ { - h } e ^ { \eta F ( x ) } + e ^ { \eta K + C _ { \alpha } h } . } \end{array}
$$

Start the Euler chain at $X _ { 0 } = x _ { f }$ . Since $F ( x _ { f } ) = 0$ , taking expectations and iterating the preceding inequality gives

$$
\mathbb { E } e ^ { \eta F ( X _ { n } ) } \le e ^ { - n h } + e ^ { \eta K + C _ { \alpha } h } \sum _ { j = 0 } ^ { n - 1 } e ^ { - j h } \le 1 + \frac { e ^ { \eta K + C _ { \alpha } h } } { 1 - e ^ { - h } } .
$$

Each expectation is finite by induction, and the bound is independent of $n .$ . The $W _ { 2 }$ contraction gives convergence of the law of $X _ { n }$ to $\widehat { \pi } _ { \lambda , h }$ . Since $e ^ { \eta F }$ is continuous and nonnegative, lower semicontinuity yields

$$
\int e ^ { \eta F } \widehat { \pi } _ { \lambda , h } \mathrm { d } x \leq \operatorname* { l i m } _ { n  \infty } \operatorname { l i m } _ { } \mathcal { \Lambda } e ^ { \eta F ( X _ { n } ) } < \infty .\tag{D.5}
$$

Bounds for the density and its gradient. We bound $\nu _ { t }$ and its gradient uniformly for $0 \leq t \leq h$ as stated in (D.9).

We first prove the comparison (D.6) between $F ( y )$ and $F ( z )$ . For $0 < \beta < \gamma$ , Taylor’s upper bound gives

$$
\beta F ( y ) - \gamma F ( z ) \leq - ( \gamma - \beta ) F ( z ) + \beta \left. \nabla F ( z ) , y - z \right. + \frac { \beta L _ { f } } { 2 } \left. y - z \right. ^ { 2 } .
$$

By Young’s inequality,

$$
\begin{array} { r l r } {  { \beta  \nabla F ( z ) , y - z  \le \frac { \gamma - \beta } { 2 L _ { f } } \| \nabla F ( z ) \| ^ { 2 } + \frac { \beta ^ { 2 } L _ { f } } { 2 ( \gamma - \beta ) } \| y - z \| ^ { 2 } } } \\ & { } & { \le ( \gamma - \beta ) F ( z ) + \frac { \beta ^ { 2 } L _ { f } } { 2 ( \gamma - \beta ) } \| y - z \| ^ { 2 } , } \end{array}
$$

where the second inequality uses $\Vert \nabla F ( z ) \Vert ^ { 2 } \leq 2 L _ { f } F ( z )$ . Substitution gives

$$
\beta F ( y ) - \gamma F ( z ) \leq \frac { \beta \gamma L _ { f } } { 2 ( \gamma - \beta ) } \left. y - z \right. ^ { 2 } .\tag{D.6}
$$

We also need an upper bound for $F ( T _ { t } x )$ . For $0 \leq t \leq h$ , Taylor’s upper bound gives

$$
\begin{array} { r l } & { \displaystyle F ( T _ { t } x ) - F ( x ) \leq - t \langle \nabla f ( x ) , \nabla U _ { \lambda } ( x ) \rangle + \frac { t ^ { 2 } L _ { f } } { 2 } \| \nabla U _ { \lambda } ( x ) \| ^ { 2 } } \\ & { \quad \quad \quad \quad = - \displaystyle \frac { t } { 2 } \| \nabla f ( x ) \| ^ { 2 } + \frac { t } { 2 } \| \nabla g _ { \lambda } ( x ) \| ^ { 2 } - \frac { t ( 1 - t L _ { f } ) } { 2 } \| \nabla U _ { \lambda } ( x ) \| ^ { 2 } } \\ & { \quad \quad \quad \leq - \displaystyle \frac { t } { 2 } \| \nabla f ( x ) \| ^ { 2 } + \frac { t } { 2 } G ^ { 2 } \leq \frac { t } { 2 } G ^ { 2 } , } \end{array}
$$

using $t L _ { f } \leq 1$ and $\| \nabla g _ { \lambda } \| \le G$ . Thus

$$
F ( T _ { t } x ) \leq F ( x ) + { \frac { t } { 2 } } G ^ { 2 } , \qquad 0 \leq t \leq h .\tag{D.7}
$$

Choose the two intermediate exponents

$$
\beta _ { 1 } : = 1 - \frac { 1 } { 4 \alpha } , \qquad \beta _ { 2 } : = 1 - \frac { 1 } { 2 \alpha } ,
$$

and decrease the universal step constant so that hα $L _ { f } \leq 1 / 3 2$

The stationary identity and (2.12) give

$$
\widehat { \pi } _ { \lambda , h } ( y ) = ( 4 \pi h ) ^ { - d / 2 } \int e ^ { - \| y - T _ { h } x \| ^ { 2 } / ( 4 h ) } \widehat { \pi } _ { \lambda , h } ( x ) \mathrm { d } x .
$$

Diferentiating the Gaussian kernel gives

$$
\nabla \widehat { \pi } _ { \lambda , h } ( y ) = - ( 4 \pi h ) ^ { - d / 2 } \int \frac { y - T _ { h } x } { 2 h } e ^ { - \| y - T _ { h } x \| ^ { 2 } / ( 4 h ) } \widehat { \pi } _ { \lambda , h } ( x ) \mathrm { d } x .
$$

This diferentiation is justified because the kernel and its spatial derivatives are bounded for fixed $h > 0$ . Since

$$
( 4 \pi h ) ^ { - d / 2 } \left( 1 + \frac { \| y - T _ { h } x \| } { 2 h } \right) e ^ { - \| y - T _ { h } x \| ^ { 2 } / ( 4 h ) } \leq C _ { h } e ^ { - \| y - T _ { h } x \| ^ { 2 } / ( 8 h ) } ,
$$

the density and its gradient can be estimated using the same Gaussian bound. In (D.6), the choice $( \gamma , \beta ) = ( \eta , \beta _ { 1 } )$ gives a coeficient at most 4α $L _ { f }$ . Thus

$$
\begin{array} { r l } & { \displaystyle e ^ { \beta _ { 1 } F ( y ) } \big ( \widehat \pi _ { \lambda , h } ( y ) + \| \nabla \widehat \pi _ { \lambda , h } ( y ) \| \big ) } \\ & { \displaystyle \quad \leq C _ { h } \int e ^ { \eta F ( T _ { h } x ) } \exp \biggl \{ - \left( \frac 1 { 8 h } - 4 \alpha L _ { f } \right) \| y - T _ { h } x \| ^ { 2 } \biggr \} \widehat \pi _ { \lambda , h } ( x ) \mathrm { d } x } \\ & { \displaystyle \quad \leq C _ { h } e ^ { \eta h G ^ { 2 } / 2 } \int e ^ { \eta F ( x ) } \widehat \pi _ { \lambda , h } ( x ) \mathrm { d } x < \infty . } \end{array}
$$

The last inequality uses hα $L _ { f } \le 1 / 3 2 , ( \mathrm { D } . 7 )$ with $t = h ,$ , and (D.5). The bound is independent of $y .$ The change-of-variables formula gives

$$
( ( T _ { t } ) _ { \# } \widehat { \pi } _ { \lambda , h } ) ( y ) = \widehat { \pi } _ { \lambda , h } ( T _ { t } ^ { - 1 } y ) | \operatorname* { d e t } D T _ { t } ^ { - 1 } ( y ) | .
$$

By Proposition 2.1,

$$
\begin{array} { r } { D T _ { t } ^ { - 1 } ( y ) = \big ( I - t \nabla ^ { 2 } U _ { \lambda } ( T _ { t } ^ { - 1 } y ) \big ) ^ { - 1 } , \qquad \big \| D T _ { t } ^ { - 1 } ( y ) \big \| _ { \mathrm { o p } } \leq ( 1 - h L _ { \lambda } ) ^ { - 1 } . } \end{array}
$$

The determinant is positive. Diferentiating the density formula gives

$$
\begin{array} { r l } & { \nabla ( ( T _ { t } ) _ { \# } \widehat { \pi } _ { \lambda , h } ) ( y ) = \operatorname* { d e t } D T _ { t } ^ { - 1 } ( y ) [ D T _ { t } ^ { - 1 } ( y ) ] ^ { \mathsf { T } } \nabla \widehat { \pi } _ { \lambda , h } ( T _ { t } ^ { - 1 } y ) } \\ & { \qquad + \widehat { \pi } _ { \lambda , h } ( T _ { t } ^ { - 1 } y ) \nabla \operatorname* { d e t } D T _ { t } ^ { - 1 } ( y ) . } \end{array}
$$

At the fixed smoothing scale, the third derivatives of $U _ { \lambda }$ are bounded. Diferentiating the inversematrix formula above therefore shows that det $D T _ { t } ^ { - 1 }$ and $\nabla$ det $D T _ { t } ^ { - 1 }$ are uniformly bounded for $0 \leq t \leq h$

The preceding bounds on $\widehat { \pi } _ { \lambda , h }$ and $\nabla \widehat { \pi } _ { \lambda , h }$ now imply

$$
( ( T _ { t } ) _ { \# } \widehat { \pi } _ { \lambda , h } ) ( y ) + \| \nabla ( ( T _ { t } ) _ { \# } \widehat { \pi } _ { \lambda , h } ) ( y ) \| \leq C e ^ { - \beta _ { 1 } F ( T _ { t } ^ { - 1 } y ) } .
$$

Applying (D.7) with $x = T _ { t } ^ { - 1 } y$ gives

$$
F ( y ) \leq F ( T _ { t } ^ { - 1 } y ) + t G ^ { 2 } / 2 .
$$

Consequently,

$$
\begin{array} { r } { ( ( T _ { t } ) _ { \# } \widehat { \pi } _ { \lambda , h } ) ( y ) + \| \nabla ( ( T _ { t } ) _ { \# } \widehat { \pi } _ { \lambda , h } ) ( y ) \| \leq C e ^ { - \beta _ { 1 } F ( y ) } , \qquad 0 \leq t \leq h . } \end{array}\tag{D.8}
$$

with one constant for the whole time interval.

For the final heat convolution, apply (D.6) with $( \gamma , \beta ) = ( \beta _ { 1 } , \beta _ { 2 } )$ , whose coeficient is at most 2α $L _ { f } .$ . By (D.8) and (D.1), both the pushed density and its gradient are integrable. Spatial integration by parts therefore gives

$$
\begin{array} { r } { \nabla \mathsf { H } _ { t } ( ( T _ { t } ) _ { \# } \widehat { \pi } _ { \lambda , h } ) = \mathsf { H } _ { t } \nabla ( ( T _ { t } ) _ { \# } \widehat { \pi } _ { \lambda , h } ) . } \end{array}
$$

For $t > 0$ it follows that

$$
\begin{array} { r l r } {  { e ^ { \beta _ { 2 } F ( y ) } \big ( \nu _ { t } ( y ) + \| \nabla \nu _ { t } ( y ) \| \big ) } } \\ & { } & { \leq C ( 4 \pi t ) ^ { - d / 2 } \int \exp \biggl \{ - ( \frac { 1 } { 4 t } - 2 \alpha L _ { f } ) \| y - z \| ^ { 2 } \biggr \} ~ \mathrm { d } z } \\ & { } & { = C ( 1 - 8 \alpha t L _ { f } ) ^ { - d / 2 } \leq C ^ { \prime } . } \end{array}
$$

The case $t = 0$ follows from $\left( \mathrm { D } . 8 \right)$ , since $T _ { 0 }$ is the identity, $F \geq 0$ , and $\beta _ { 1 } > \beta _ { 2 }$ . Consequently, with a constant uniform for $0 \leq t \leq h$ 2

$$
\nu _ { t } ( y ) + \| \nabla \nu _ { t } ( y ) \| \leq C _ { \alpha , h } \exp \{ - ( 1 - 1 / ( 2 \alpha ) ) F ( y ) \} .\tag{D.9}
$$

Regularity and endpoint continuity. We show that the first time derivative and the spatial derivatives up to order two of $\nu _ { t }$ exist and are continuous for $0 < t < h$ . We then check continuity at the two time endpoints.

For $t > 0$ , write the Gaussian representation as

$$
\nu _ { t } ( y ) = \int \kappa _ { t } ( y , x ) \widehat { \pi } _ { \lambda , h } ( x ) \mathrm { d } x , \qquad \kappa _ { t } ( y , x ) = ( 4 \pi t ) ^ { - d / 2 } \exp \left\{ - \frac { \| y - T _ { t } x \| ^ { 2 } } { 4 t } \right\} .
$$

Here $T _ { t } x = x - t \nabla U _ { \lambda } ( x )$ , while $\widehat { \pi } _ { \lambda , h }$ is fixed throughout the interpolation.

The spatial derivatives of the kernel are

$$
\begin{array} { l } { { \nabla _ { y } \kappa _ { t } ( y , x ) = - \displaystyle \frac { y - T _ { t } x } { 2 t } \kappa _ { t } ( y , x ) , } } \\ { { \nabla _ { y } ^ { 2 } \kappa _ { t } ( y , x ) = \displaystyle \left[ \frac { ( y - T _ { t } x ) ( y - T _ { t } x ) ^ { \top } } { 4 t ^ { 2 } } - \frac { I _ { d } } { 2 t } \right] \kappa _ { t } ( y , x ) . } } \end{array}
$$

For the time derivative, note that $\partial _ { t } ( y - T _ { t } x ) = \nabla U _ { \lambda } ( x )$ . Diferentiating both the prefactor and the exponent gives

$$
\partial _ { t } \kappa _ { t } ( y , x ) = \left[ - \frac { d } { 2 t } + \frac { \left. y - T _ { t } x \right. ^ { 2 } } { 4 t ^ { 2 } } - \frac { \langle y - T _ { t } x , \nabla U _ { \lambda } ( x ) \rangle } { 2 t } \right] \kappa _ { t } ( y , x ) .
$$

Fix $0 < \varepsilon < h$ . The functions

$$
z \longmapsto ( 1 + \| z \| + \| z \| ^ { 2 } ) e ^ { - \| z \| ^ { 2 } / 4 }
$$

are bounded on $\mathbb { R } ^ { d }$ . Applying this observation with $z = ( y - T _ { t } x ) / \sqrt { t }$ to the displayed derivatives gives, for $\varepsilon \le t \le h$ and all $x , y .$

$$
\begin{array} { r l } & { | \kappa _ { t } ( y , x ) | + \| \nabla _ { y } \kappa _ { t } ( y , x ) \| + \left\| \nabla _ { y } ^ { 2 } \kappa _ { t } ( y , x ) \right\| + | \partial _ { t } \kappa _ { t } ( y , x ) | } \\ & { \qquad \leq C _ { \varepsilon } \big ( 1 + \| \nabla U _ { \lambda } ( x ) \| \big ) . } \end{array}
$$

This bound is independent of t and $y$ in the stated range.

The drift grows at most linearly, and the quadratic lower bound in (D.1) implies

$$
1 + \| \nabla U _ { \lambda } ( x ) \| \leq C ( 1 + \| x - x _ { f } \| ) \leq C ^ { \prime } e ^ { \eta F ( x ) } .
$$

Thus (D.5) makes the common bound integrable against $\widehat { \pi } _ { \lambda , h } ( \boldsymbol { x } ) \mathrm { d } \boldsymbol { x }$

We may therefore diferentiate the integral once in time and twice in space:

$$
\begin{array} { r l } & { \displaystyle \partial _ { t } \nu _ { t } ( y ) = \int \partial _ { t } \kappa _ { t } ( y , x ) \widehat { \pi } _ { \lambda , h } ( x ) \mathrm { d } x , } \\ & { \displaystyle \nabla _ { y } ^ { j } \nu _ { t } ( y ) = \int \nabla _ { y } ^ { j } \kappa _ { t } ( y , x ) \widehat { \pi } _ { \lambda , h } ( x ) \mathrm { d } x , \qquad j = 1 , 2 . } \end{array}
$$

For each fixed $x ,$ the kernel and all the displayed derivatives are continuous in $( t , y )$ . Dominated convergence, using the same common bound, shows that their integrals are also continuous in $( t , y )$ Since $\varepsilon > 0$ is arbitrary, ν is locally $C ^ { 1 , 2 }$ on $( 0 , h ) \times \mathbb { R } ^ { d }$

Moreover, $\kappa _ { t } ( y , x ) > 0$ for every $t > 0$ and every $x , y$ . Since $\widehat { \pi } _ { \lambda , h }$ is a probability density, this gives $\nu _ { t } ( y ) > 0$ . The same dominated-convergence argument applies as $t \uparrow h ,$ , so

$$
\nu _ { t } ( y ) \longrightarrow \nu _ { h } ( y ) = \widehat { \pi } _ { \lambda , h } ( y ) .
$$

We now prove continuity at $t = 0$ . First, the inverse-map bound in Proposition 2.1 gives

$$
\begin{array} { l } { \displaystyle \left\| T _ { t } ^ { - 1 } y - y \right\| = \left\| T _ { t } ^ { - 1 } y - T _ { t } ^ { - 1 } ( T _ { t } y ) \right\| } \\ { \displaystyle \leq \frac { 1 } { 1 - t L _ { \lambda } } \| y - T _ { t } y \| } \\ { \displaystyle = \frac { t } { 1 - t L _ { \lambda } } \| \nabla U _ { \lambda } ( y ) \| \longrightarrow 0 . } \end{array}
$$

The inverse derivative also satisfies

$$
D T _ { t } ^ { - 1 } ( y ) = \big ( I - t \nabla ^ { 2 } U _ { \lambda } ( T _ { t } ^ { - 1 } y ) \big ) ^ { - 1 } , \qquad \big \| D T _ { t } ^ { - 1 } ( y ) - I _ { d } \big \| _ { \mathrm { o p } } \leq \frac { t L _ { \lambda } } { 1 - t L _ { \lambda } } \longrightarrow 0 .
$$

Using the change-of-variables formula (2.11) and the continuity of $\widehat { \pi } _ { \lambda , h } = \nu _ { h }$ , we obtain

$$
\begin{array} { r } { ( ( T _ { t } ) _ { \# } \widehat { \pi } _ { \lambda , h } ) ( y ) = \widehat { \pi } _ { \lambda , h } ( T _ { t } ^ { - 1 } y ) \operatorname* { d e t } D T _ { t } ^ { - 1 } ( y ) \longrightarrow \widehat { \pi } _ { \lambda , h } ( y ) . } \end{array}
$$

We next estimate the efect of the Gaussian increment. Equation (D.8) and $F \geq 0$ give

$$
\operatorname* { s u p } _ { 0 \leq r \leq h } \left\| \nabla ( ( T _ { r } ) _ { \# } \widehat { \pi } _ { \lambda , h } ) \right\| _ { \infty } \leq C .
$$

Thus these densities are Lipschitz with the same constant for all $0 \leq r \leq h$ . By (2.12) and $\nu _ { t } = \mathsf { H } _ { t } ( ( T _ { t } ) _ { \# } \widehat { \pi } _ { \lambda , h } )$ ，

$$
\begin{array} { r l } & { \left| \nu _ { t } ( y ) - \left( ( T _ { t } ) _ { \# } \widehat { \pi } _ { \lambda , h } \right) ( y ) \right| } \\ & { \quad = \left| \mathbb E \left[ ( ( T _ { t } ) _ { \# } \widehat { \pi } _ { \lambda , h } ) ( y + \sqrt { 2 t } Z ) - ( ( T _ { t } ) _ { \# } \widehat { \pi } _ { \lambda , h } ) ( y ) \right] \right| } \\ & { \quad \leq C \sqrt { 2 t } \mathbb E \left\| Z \right\| \longrightarrow 0 . } \end{array}
$$

Together with $( ( T _ { t } ) _ { \# } \widehat { \pi } _ { \lambda , h } ) ( y ) \to \widehat { \pi } _ { \lambda , h } ( y )$ , this proves

$$
\nu _ { t } ( y ) \longrightarrow \widehat { \pi } _ { \lambda , h } ( y ) = \nu _ { 0 } ( y ) \qquad \mathrm { a s ~ } t \downarrow 0 .
$$

Density-ratio and score integrals. Since $g _ { \lambda }$ is G-Lipschitz,

$$
\pi _ { \lambda } ( y ) \geq C ^ { - 1 } \exp \{ - F ( y ) - G \| y - x _ { f } \| \} .
$$

Combining this with (D.9) gives

$$
\begin{array} { r l } & { u ( y ) ^ { \alpha } \pi _ { \lambda } ( y ) = \nu _ { t } ( y ) ^ { \alpha } \pi _ { \lambda } ( y ) ^ { 1 - \alpha } } \\ & { \qquad \leq C \exp \left\{ - \alpha \left( 1 - \displaystyle \frac { 1 } { 2 \alpha } \right) F ( y ) + ( \alpha - 1 ) \big ( F ( y ) + G \| y - x _ { f } \| \big ) \right\} } \\ & { \qquad = C \exp \left\{ - \displaystyle \frac { 1 } { 2 } F ( y ) + ( \alpha - 1 ) G \| y - x _ { f } \| \right\} . } \end{array}
$$

Similarly,

$$
u ( y ) ^ { 3 \alpha / 2 } \pi _ { \lambda } ( y ) \leq C \exp \left\{ - { \frac { 1 } { 4 } } F ( y ) + \left( { \frac { 3 \alpha } { 2 } } - 1 \right) G \left\| y - x _ { f } \right\| \right\} .
$$

Both bounds are integrable because

$$
F ( y ) \geq { \frac { m } { 2 } } \left\| y - x _ { f } \right\| ^ { 2 } .
$$

Their constants are uniform for $0 \leq t \leq h$

For the score, use $s = \nabla$ log $\nu _ { t } + \nabla U _ { \lambda }$ to obtain

$$
u ^ { \alpha } \pi _ { \lambda } \left\| s \right\| ^ { 2 } \leq 2 \pi _ { \lambda } ^ { 1 - \alpha } \nu _ { t } ^ { \alpha - 2 } \left\| \nabla \nu _ { t } \right\| ^ { 2 } + 2 \pi _ { \lambda } ^ { 1 - \alpha } \nu _ { t } ^ { \alpha } \left\| \nabla U _ { \lambda } \right\| ^ { 2 } .
$$

Since $\alpha \geq 2 \quad$ , the factor $\nu _ { t } ^ { \alpha - 2 }$ can be bounded using (D.9). The same equation bounds $\nabla \nu _ { t }$ , and $\nabla U _ { \lambda }$ grows at most linearly. Thus the sum of the two terms is bounded by

$$
C \bigl ( 1 + \| y - x _ { f } \| ^ { 2 } \bigr ) \exp \left\{ - \frac { 1 } { 2 } F ( y ) + ( \alpha - 1 ) G \left\| y - x _ { f } \right\| \right\} .
$$

This bound is integrable and uniform in time. We have therefore proved

$$
\operatorname* { s u p } _ { 0 \leq t \leq h } \int \left[ u ^ { \alpha } ( 1 + \| s \| ^ { 2 } ) + u ^ { 3 \alpha / 2 } \right] \mathrm { d } \pi _ { \lambda } < \infty .
$$

The pointwise endpoint continuity of $\nu _ { t }$ and the integrable bound for $u ^ { \alpha } \pi _ { \lambda }$ give the following limits by dominated convergence:

$$
\begin{array} { l } { \displaystyle \operatorname* { l i m } _ { t \downarrow 0 } \int u ( t ) ^ { \alpha } \mathrm { d } \pi _ { \lambda } = \int \left( \frac { \widehat \pi _ { \lambda , h } } { \pi _ { \lambda } } \right) ^ { \alpha } \mathrm { d } \pi _ { \lambda } , } \\ { \displaystyle \operatorname* { l i m } _ { t \uparrow h } \int u ( t ) ^ { \alpha } \mathrm { d } \pi _ { \lambda } = \int \left( \frac { \widehat \pi _ { \lambda , h } } { \pi _ { \lambda } } \right) ^ { \alpha } \mathrm { d } \pi _ { \lambda } . } \end{array}\tag{D.10}
$$

Since $\nu _ { 0 } = \nu _ { h } = \widehat { \pi } _ { \lambda , h }$ , this proves continuity of $\begin{array} { r } { t \mapsto \int u ( t ) ^ { \alpha } \mathrm { d } \pi _ { \lambda } } \end{array}$ at both endpoints.

Joint weighted moments. We prove the joint moment bound in Lemma B.1. The upper bound for $\nu _ { t }$ and the lower bound for $\pi _ { \lambda }$ give

$$
u ( y ) ^ { \alpha - 1 } \leq C \exp \left\{ \frac { \alpha - 1 } { 2 \alpha } F ( y ) + ( \alpha - 1 ) G \left. y - x _ { f } \right. \right\} .
$$

Strong convexity and Young’s inequality give

$$
( \alpha - 1 ) G \parallel y - x _ { f } \parallel \leq { \frac { 1 } { 8 } } F ( y ) + { \frac { 4 ( \alpha - 1 ) ^ { 2 } G ^ { 2 } } { m } } .
$$

Since $( \alpha - 1 ) / ( 2 \alpha ) \le 1 / 2$ , it follows that

$$
u ( y ) ^ { \alpha - 1 } \leq C e ^ { 5 F ( y ) / 8 } .
$$

For every integer $k \geq 0$ , the quadratic lower bound in (D.1) gives

$$
( 1 + \Vert Y \Vert ) ^ { k } \leq C _ { k } e ^ { F ( Y ) / 3 2 } .
$$

Consequently,

$$
\begin{array} { r l } & { u ( Y ) ^ { \alpha - 1 } ( 1 + \| X \| + \| Y \| ) ^ { k } } \\ & { \qquad \leq C _ { k } ( 1 + \| X \| ) ^ { k } e ^ { ( 5 / 8 + 1 / 3 2 ) F ( Y ) } } \\ & { \qquad = C _ { k } ( 1 + \| X \| ) ^ { k } e ^ { 2 1 F ( Y ) / 3 2 } . } \end{array}
$$

Conditional on X, Taylor’s upper bound gives

$$
F ( Y ) \leq F ( T _ { t } X ) + \sqrt { 2 t } \left. \nabla F ( T _ { t } X ) , Z \right. + t L _ { f } \left. Z \right. ^ { 2 } .
$$

Applying (D.2) yields

$$
\begin{array} { r l r } {  { \mathbb { E } [ e ^ { 2 1 F ( Y ) / 3 2 } \mid X ] \le ( 1 - \frac { 2 1 } { 1 6 } t L _ { f } ) ^ { - d / 2 } \exp \{ \frac { 2 1 } { 3 2 } F ( T _ { t } X ) + \frac { ( 2 1 / 3 2 ) ^ { 2 } t } { 1 - \frac { 2 1 } { 1 6 } t L _ { f } } \| \nabla F ( T _ { t } X ) \| ^ { 2 } \} } } \\ & { } & { \le ( 1 - \frac { 2 1 } { 1 6 } t L _ { f } ) ^ { - d / 2 } \exp \{ \frac { 2 1 F ( T _ { t } X ) } { 3 2 ( 1 - \frac { 2 1 } { 1 6 } t L _ { f } ) } \} , } \end{array}
$$

where the second inequality uses $\| \nabla F \| ^ { 2 } \leq 2 L _ { f } F .$

The condition hα $L _ { f } \leq 1 / 3 2$ and $\alpha \geq 2$ imply $t L _ { f } \leq 1 / 6 4$ . In particular,

$$
\frac { 2 1 } { 3 2 ( 1 - \frac { 2 1 } { 1 6 } t L _ { f } ) } \leq \frac { 3 } { 4 } ,
$$

and the Gaussian prefactor is bounded uniformly for $0 \leq t \leq h$ . Using (D.7) with $x = X$ , we obtain

$$
{ \mathbb E } \Big [ e ^ { 2 1 F ( Y ) / 3 2 } \ | \ X \Big ] \leq C e ^ { 3 F ( X ) / 4 } , \qquad 0 \leq t \leq h .
$$

Finally, $\eta = 1 - 1 / ( 8 \alpha ) \geq 1 5 / 1 6$ gives $\eta - 3 / 4 \geq 3 / 1 6$ . The quadratic lower bound in (D.1) therefore gives

$$
( 1 + \| X \| ) ^ { k } e ^ { 3 F ( X ) / 4 } \leq C _ { k } e ^ { \eta F ( X ) } .
$$

Combining the preceding estimates gives

$$
\begin{array} { r l } & { \mathbb { E } \Big [ u ( Y ) ^ { \alpha - 1 } ( 1 + \| X \| + \| Y \| ) ^ { k } \Big ] } \\ & { \qquad \leq C _ { k } \mathbb { E } \Big [ ( 1 + \| X \| ) ^ { k } e ^ { 3 F ( X ) / 4 } \Big ] } \\ & { \qquad \leq C _ { k } ^ { \prime } \displaystyle \int e ^ { \eta F ( x ) } \widehat \pi _ { \lambda , h } ( x ) \mathrm { d } x < \infty . } \end{array}
$$

The last inequality is (D.5), and the bound is uniform for $0 \leq t \leq h$

It remains to bound the integral containing $e _ { t }$ . By its definition,

$$
e _ { t } ( Y ) = \mathbb { E } [ \nabla U _ { \lambda } ( Y ) - \nabla U _ { \lambda } ( X ) \mid Y ] .
$$

Conditional Jensen’s inequality gives

$$
\begin{array} { r l } {  { \int u ^ { \alpha } \| e _ { t } \| ^ { 2 } \mathrm { d } \pi _ { \lambda } = \mathbb { E } \Big [ u ( Y ) ^ { \alpha - 1 } \| e _ { t } ( Y ) \| ^ { 2 } \Big ] } } \\ & { \qquad \le \mathbb { E } \Big [ u ( Y ) ^ { \alpha - 1 } \| \nabla U _ { \lambda } ( Y ) - \nabla U _ { \lambda } ( X ) \| ^ { 2 } \Big ] } \\ & { \qquad \le L _ { \lambda } ^ { 2 } \mathbb { E } \Big [ u ( Y ) ^ { \alpha - 1 } \| Y - X \| ^ { 2 } \Big ] . } \end{array}
$$

The last expression is uniformly finite by the joint moment bound with $k = 2$

This completes the proof of Lemma B.1.

## D.3 Removing the auxiliary mollification

We now restore the superscripts (δ) and let $\delta \downarrow 0$ , with $\alpha , \lambda .$ , and h fixed. The global Lipschitz bound on $\nabla U _ { \lambda }$ gives

$$
\varepsilon _ { \delta } : = \left\| \nabla U _ { \lambda } ^ { ( \delta ) } - \nabla U _ { \lambda } \right\| _ { \infty } \longrightarrow 0 .
$$

By the definitions of $f ^ { ( \delta ) }$ and $g _ { \lambda } ^ { ( \delta ) }$ ,

$$
U _ { \lambda } ^ { ( \delta ) } ( x ) = ( U _ { \lambda } * \varrho _ { \delta } ) ( x ) = \int U _ { \lambda } ( x - \delta z ) \varrho ( z ) \mathrm { d } z .
$$

The symmetry of $\varrho$ gives

$$
\int z \varrho ( z ) \mathrm { d } z = 0 .
$$

Since $\nabla U _ { \lambda }$ is $L _ { \lambda } .$ -Lipschitz, Taylor’s remainder satisfies

$$
\left| U _ { \lambda } ( \boldsymbol { x } - \delta \boldsymbol { z } ) - U _ { \lambda } ( \boldsymbol { x } ) + \delta \left. \nabla U _ { \lambda } ( \boldsymbol { x } ) , \boldsymbol { z } \right. \right| \leq \frac { L _ { \lambda } \delta ^ { 2 } } { 2 } \left\| \boldsymbol { z } \right\| ^ { 2 } .
$$

Using the zero first moment, we can write

$$
U _ { \lambda } ^ { ( \delta ) } ( x ) - U _ { \lambda } ( x ) = \int \Bigl [ U _ { \lambda } ( x - \delta z ) - U _ { \lambda } ( x ) + \delta \left. \nabla U _ { \lambda } ( x ) , z \right. \Bigr ] \varrho ( z ) \mathrm { d } z .
$$

Integrating the remainder bound and taking the supremum over x therefore gives

$$
\left\| { U _ { \lambda } ^ { ( \delta ) } - U _ { \lambda } } \right\| _ { \infty } \le \frac { L _ { \lambda } \delta ^ { 2 } } { 2 } \int \left\| { z } \right\| ^ { 2 } \varrho ( z ) \mathrm { d } z = O ( L _ { \lambda } \delta ^ { 2 } ) .
$$

The implicit constant depends only on the fixed kernel $\varrho .$ Consequently, $\pi _ { \lambda } ^ { ( \delta ) } / \pi _ { \lambda }$ converges uniformly to one. In particular, $\pi _ { \lambda } ^ { ( \delta ) } \leq 2 \pi _ { \lambda }$ for all suficiently small δ. Since $\pi _ { \lambda }$ has a finite second moment, dominated convergence gives both weak convergence and convergence of the second moments. By the standard characterization of $W _ { 2 }$ convergence,

$$
W _ { 2 } ( \pi _ { \lambda } ^ { ( \delta ) } , \pi _ { \lambda } ) \longrightarrow 0 .
$$

Recall that

$$
\begin{array} { r } { \varepsilon _ { \delta } : = \left\| \nabla U _ { \lambda } ^ { ( \delta ) } - \nabla U _ { \lambda } \right\| _ { \infty } , \qquad T _ { h } ^ { ( \delta ) } x : = x - h \nabla U _ { \lambda } ^ { ( \delta ) } ( x ) . } \end{array}
$$

The smoothed and original potentials satisfy the same lower and upper Hessian bounds. Hence the contraction argument in Proposition 3.3 gives

$$
\left\| T _ { h } ^ { ( \delta ) } x - T _ { h } ^ { ( \delta ) } y \right\| \leq \left( 1 - m h \right) \| x - y \| .
$$

Also,

$$
\left\| T _ { h } ^ { ( \delta ) } x - T _ { h } x \right\| \leq h \varepsilon _ { \delta } .
$$

Choose an optimal coupling $( X _ { \delta } , X )$ of $\widehat { \pi } _ { \lambda , h } ^ { ( \delta ) }$ and $\widehat { \pi } _ { \lambda , h }$ , so that

$$
\begin{array} { r } { \Big ( \mathbb { E } \left\| X _ { \delta } - X \right\| ^ { 2 } \Big ) ^ { 1 / 2 } = W _ { 2 } \big ( \widehat { \pi } _ { \lambda , h } ^ { ( \delta ) } , \widehat { \pi } _ { \lambda , h } \big ) . } \end{array}
$$

Let $Z \sim N ( 0 , I _ { d } )$ be independent of this pair. By invariance,

$$
T _ { h } ^ { ( \delta ) } X _ { \delta } + \sqrt { 2 h } Z \sim \widehat { \pi } _ { \lambda , h } ^ { ( \delta ) } , \qquad T _ { h } X + \sqrt { 2 h } Z \sim \widehat { \pi } _ { \lambda , h } .
$$

These two updated random variables therefore give another coupling of the same invariant laws. Their Gaussian increments cancel in the diference. The definition of $W _ { 2 }$ and the $L ^ { 2 }$ triangle inequality give

$$
\begin{array} { r l } { W _ { 2 } \big ( \widehat { \pi } _ { \lambda , h } ^ { ( \delta ) } , \widehat { \pi } _ { \lambda , h } \big ) } & { \leq \bigg ( \mathbb { E } \left\| T _ { h } ^ { ( \delta ) } X _ { \delta } - T _ { h } X \right\| ^ { 2 } \bigg ) ^ { 1 / 2 } } \\ & { \leq \bigg ( \mathbb { E } \left\| T _ { h } ^ { ( \delta ) } X _ { \delta } - T _ { h } ^ { ( \delta ) } X \right\| ^ { 2 } \bigg ) ^ { 1 / 2 } + \bigg ( \mathbb { E } \left\| T _ { h } ^ { ( \delta ) } X - T _ { h } X \right\| ^ { 2 } \bigg ) ^ { 1 / 2 } } \\ & { \leq ( 1 - m h ) \bigg ( \mathbb { E } \left\| X _ { \delta } - X \right\| ^ { 2 } \bigg ) ^ { 1 / 2 } + h \varepsilon _ { \delta } } \\ & { = ( 1 - m h ) W _ { 2 } \big ( \widehat { \pi } _ { \lambda , h } ^ { ( \delta ) } , \widehat { \pi } _ { \lambda , h } \big ) + h \varepsilon _ { \delta } . } \end{array}
$$

Rearranging and using $h > 0$ gives

$$
W _ { 2 } \bigl ( \widehat { \pi } _ { \lambda , h } ^ { ( \delta ) } , \widehat { \pi } _ { \lambda , h } \bigr ) \leq \frac { \varepsilon _ { \delta } } { m } \longrightarrow 0 \qquad \mathrm { a s ~ } \delta \downarrow 0 .
$$

The same coupling for the interpolations, together with $\mathrm { L i p } ( T _ { t } ^ { ( \delta ) } ) \leq 1 - m t$ , yields

$$
\begin{array} { c } { { W _ { 2 } ( \nu _ { t } ^ { ( \delta ) } , \nu _ { t } ) \leq ( 1 - m t ) W _ { 2 } \left( \widehat { \pi } _ { \lambda , h } ^ { ( \delta ) } , \widehat { \pi } _ { \lambda , h } \right) + t \varepsilon _ { \delta } } } \\ { { \leq \displaystyle \frac { \varepsilon _ { \delta } } { m } , \qquad 0 \leq t \leq h . } } \end{array}
$$

Thus the regularized interpolations converge to the original interpolation uniformly in time in $W _ { 2 }$

Under the restrictions of Lemma 4.2, the argument in Appendix B gives

$$
\mathcal R _ { \alpha } ( \nu _ { t } ^ { ( \delta ) } | | \pi _ { \lambda } ^ { ( \delta ) } ) \leq \frac { 1 } { 4 } \qquad \mathrm { f o r ~ e v e r y ~ } \delta > 0 \mathrm { ~ a n d ~ } 0 \leq t \leq h .
$$

For each fixed $t \in [ 0 , h ]$ , the $W _ { 2 }$ convergence established above implies the weak convergence

$$
\nu _ { t } ^ { ( \delta ) } \Rightarrow \nu _ { t } , \qquad \pi _ { \lambda } ^ { ( \delta ) } \Rightarrow \pi _ { \lambda } .
$$

Joint lower semicontinuity of the R´enyi divergence, together with the uniform bound for the regularized problem, therefore gives

$$
\mathcal { R } _ { \alpha } ( \nu _ { t } \| \pi _ { \lambda } ) \leq \operatorname* { l i m } _ { \delta \downarrow 0 } \operatorname* { i n f } _ { } \mathcal { R } _ { \alpha } ( \nu _ { t } ^ { ( \delta ) } \| \pi _ { \lambda } ^ { ( \delta ) } ) \leq \frac { 1 } { 4 } .
$$

The same bound holds for every $t \in [ 0 , h ]$ , so taking the supremum in time and using (2.27) proves (4.6). The same argument also preserves the full quantitative bound (B.12).

## References

[1] Sinho Chewi. Log-Concave Sampling. Book manuscript, 2026. URL https://chewisinho. github.io/main.pdf. Final draft, August 20, 2026.

[2] Sinho Chewi, Murat A. Erdogdu, Mufan Bill Li, Ruoqi Shen, and Matthew Zhang. Analysis of Langevin Monte Carlo from Poincar´e to Log-Sobolev. arXiv preprint arXiv:2112.12662v2, 2024. URL https://arxiv.org/abs/2112.12662v2.

[3] Alain Durmus, Eric Moulines, and Marcelo Pereyra. Eficient Bayesian computation by proximal <sup>´</sup> Markov chain Monte Carlo: When Langevin meets Moreau. SIAM Journal on Imaging Sciences, 11(1):473–506, 2018. doi: 10.1137/16M1108340.

[4] Jiaojiao Fan, Bo Yuan, and Yongxin Chen. Improved dimension dependence of a proximal algorithm for sampling. In Proceedings of Thirty Sixth Conference on Learning Theory, volume 195 of Proceedings of Machine Learning Research, pages 1473–1521. PMLR, 2023. URL https: //proceedings.mlr.press/v195/fan23a.html.

[5] Francesco Pedrotti and Peter A. Whalley. Wasserstein mixing time of the unadjusted Langevin algorithm. arXiv preprint arXiv:2608.02430, 2026.

[6] R´emi Peyre. Comparison between W<sub>2</sub> distance and $\dot { H } ^ { - 1 }$ norm, and localisation of Wasserstein distance. arXiv preprint arXiv:1104.4631v2, 2016. URL https://arxiv.org/abs/1104.4631v2.

[7] Leon Simon. Introduction to geometric measure theory. Lecture notes, 2018. URL https: //math.stanford.edu/ lms/ntu-gmt-text.pdf. NTU lectures, revised March 7, 2018.

[8] Yuchen Xin and Zhihua Zhang. Active-trace complexity bounds for Moreau–Yosida unadjusted Langevin sampling. arXiv preprint arXiv:2608.13467v1, 2026. URL https://arxiv.org/abs/ 2608.13467v1.

[9] Yuchen Xin and Zhihua Zhang. Poisson-corrector complexity bounds for Moreau–Yosida unadjusted Langevin sampling. arXiv preprint arXiv:2609.12594v1, 2026. URL https://arxiv. org/abs/2609.12594v1.