# On the gradient conjecture at infinity: Finite spherical length for gradient trajectories of definable potentials

Thang Do<sup>1</sup>, Arnulf Jentzen<sup>2,3</sup>, and Benno Kuckuck<sup>4</sup>

<sup>1</sup> School of Data Science, The Chinese University of Hong Kong, Shenzhen (CUHK-Shenzhen), China, e-mail: minhthangdo@link.cuhk.edu.cn

<sup>2</sup> School of Data Science, The Chinese University of Hong Kong, Shenzhen (CUHK-Shenzhen), China, e-mail: ajentzen@cuhk.edu.cn

Applied Mathematics: Institute for Analysis and Numerics, Faculty of Mathematics and Computer Science, University of M¨unster, Germany, e-mail: ajentzen@uni-muenster.de

<sup>4</sup> Applied Mathematics: Institute for Analysis and Numerics, Faculty of Mathematics and Computer Science, University of M¨unster, Germany, e-mail: bkuckuck@uni-muenster.de

October 22, 2026

## Abstract

We prove that for every open $U \subseteq \mathbb { R } ^ { n }$ the spherical projection of every gradient trajectory $\theta ( t )$ $t \geq 0 ,$ , of a definable $C ^ { 1 }$ -potential $F \colon U $ R has finite length and, in particular, converges (directional convergence). In other words, every gradient trajectory of a definable $C ^ { 1 }$ -potential has finite total angular motion. Here, definable means definable in an arbitrary o-minimal structure expanding the real field.

Notably, the gradient trajectory in this result is an arbitrary solution of the gradient flow differential equation (whose solutions may not be unique) with no further assumptions: The gradient trajectory may be unbounded and diverge to infinity or it may be convergent. The established theorem therefore provides afirmative answers to both the gradient conjecture at infinity for definable potentials and the definable Thom gradient conjecture.

We apply our theorem to the training of deep neural networks (DNNs) to conclude that, in the case of definable and continuously diferentiable activation and loss functions, the spherical projection of the DNN training flow must always converge (directional convergence).

## Contents

1 Introduction and main result 3   
1.1 Finite spherical length of gradient flow (GF) trajectories 3   
1.2 Related works and key novelty of this article 4   
1.3 Definability, o-minimal structures, and rectifiable curves 5   
1.4 Connections to the training of deep neural networks (DNNs) 6   
1.5 Outline of the proof and structure of this article 7   
2 Regularity properties of radial upper bounds for definable C<sup>1</sup>-scalar fields 9   
2.1 Embedded submanifolds 10   
2.2 Smooth functions on embedded submanifolds 12   
2.3 Integrability, smoothness, and definability properties of radial upper bounds 18   
3 Construction of the radial-energy regions and the energy transformations 25   
3.1 Smooth cylindrical definable cell decompositions 25   
3.2 Upper and lower bounds for the energy-derivatives of the back-transformations 26   
3.3 Upper and lower bounds for the radial-derivatives of the back-transformations 34   
4 Construction and regularity of the truncation of the transformed energy 38   
4.1 Vanishing of the derivatives of absolutely continuous functions on their level sets 38   
4.2 Construction and uniform regularity of the truncation of the transformed energy 40   
5 Lyapunov-type stability for the radial-angular diferential equation (DE) system 45   
5.1 Regularity properties for one-parameter primitives of definable two-parameter functions 45   
5.2 Lyapunov-type stability for the radial-angular dynamics 48   
5.3 Upper bounds for the error in the Lyapunov-type estimate 57   
5.4 Upper bounds for parts of the length of the radial component in the DE system 61   
5.5 Finite length of the angular component in the DE system 67   
5.6 Uniform upper bounds for the length of the angular component in the DE system 70   
6 Spherical length at finite points and at infinity 74   
6.1 Uniform upper bounds for the spherical length of GF trajectories close to the origin 75   
6.2 Finite spherical length for GF trajectories converging to finite points 77   
6.3 Finite spherical length for general open domains 82   
6.4 Finite spherical length for general time intervals 85   
6.5 Finite spherical length for locally definable potentials 87   
6.6 Infinite spherical length for an analytic potential 90   
6.7 Finite spherical length and directional convergence for DNN training flows 92

## 1 Introduction and main result

The study of solutions of gradient flow (GF) diferential equations (DEs) has a long history in mathematics (see, for instance, $[ 1 , 4 , 3 2 , 3 3 , 3 8 , 4 5 ]$ and Subsection 1.2 below).

In the main result of this work, Theorem 1.1 below, we prove for every natural number $n \in \mathbb N$ = $\{ 1 , 2 , 3 , \ldots \}$ and every open $U \subseteq \mathbb { R } ^ { n }$ that the spherical projection of every GF trajectory $\theta ( t ) , t \geq 0$ , of a C<sup>1</sup>-potential $F \colon U  \mathbb { R }$ definable in an arbitrary o-minimal structure over the reals has finite length and must, in particular, be convergent (directional convergence).

In the following, definability of F means that the graph of the function F is a definable set in an arbitrary fixed o-minimal structure over the reals (see Subsection 1.3 below). A GF trajectory is an arbitrary solution of the GF DE. It does not need to be bounded or have a convergent subsequence but may diverge to infinity.

Our findings have strong connections to the gradient conjecture [1,33,45], the definable Thom gradient conjecture [34, 43], the gradient conjecture at infinity [12, 20], and also to deep learning (DL) [7, 26, 30] and we refer to Subsection 1.2 below for a short literature review.

## 1.1 Finite spherical length of gradient flow (GF) trajectories

We now present Theorem 1.1 in a self-contained way and, thereafter, we explain the conclusion and consequences of Theorem 1.1 and also relate it to the state-of-the-art in the current literature.

```latex
Theorem 1.1. Let $n \in \mathbb { N } , \vartheta \in \mathbb { R } ^ { n }$ , let $U \subseteq \mathbb { R } ^ { n }$ be open, let $F \in C ^ { 1 } ( U , \mathbb { R } )$ be definable, let $I \subseteq \mathbb { R }$ be
convex with nonempty interior, let $\theta \in C ^ { 1 } ( I , U )$ satisfy for all $t \in I$ that
$\theta ^ { \prime } ( t ) = - ( \nabla F ) ( \theta ( t ) ) ,$ (1.1)
and let $P \colon U \to \mathbb { R } ^ { n }$ satisfy<sup>1</sup> for all $v \in U \backslash \{ \vartheta \}$ that $P ( v ) = | v - \vartheta | ^ { - 1 } ( v - \vartheta )$ . Then $P \circ \theta$ has finite
length.
```

Theorem 1.1 is proved as Theorem 6.8 in Subsection 6.4 below. The conclusion of Theorem 1.1 can be viewed as follows. The point ϑ in Theorem 1.1 serves as the reference point of polar coordinates, with every point $v \in \mathbb { R } ^ { n } \backslash \{ \vartheta \}$ having a unique representation by the polar coordinate pair consisting of the distance $| v - \vartheta | \in ( 0 , \infty )$ to the reference point of the polar coordinates and its angular component<sup>2</sup>, i.e., its spherical projection $| \boldsymbol { v } - \boldsymbol { \vartheta } | ^ { - 1 } ( \boldsymbol { v } - \boldsymbol { \vartheta } )$ . Theorem 1.1 then proves for every definable $C ^ { 1 } { \mathrm { - p o t e n t i a l } }$ $F \colon U $ R that every GF trajectory $\theta \colon I  U$ of $F$ has finite total angular motion over its entire time interval I, that is, the length of $P \circ \theta$ is finite. Clearly, the fact that $P \circ \theta$ has finite length implies that there exists a unique $v \in \mathbb { R } ^ { n }$ such that lim $\begin{array} { r } { \operatorname* { s u p } _ { t \to \operatorname* { s u p } ( I ) } | ( P \circ \theta ) ( t ) - v | = 0 } \end{array}$ (directional convergence); see, for instance, Lemma 6.12.

Note that the time interval I in Theorem 1.1 may be bounded or unbounded, for example, $I = [ 0 , \infty )$ and the endpoints of I may or may not be elements of I. Theorem 1.1 covers the situation where lim $\begin{array} { r } { \operatorname* { i n f } _ { t  \operatorname* { s u p } ( I ) } | \theta ( t ) | = \infty } \end{array}$ (divergence to infinity) as well as the situation where lim in $\dot { \cdot } _ { t  \mathrm { s u p } ( I ) } | \theta ( t ) | < \infty$ (cf., for instance, Proposition 6.9 below).

Concerning smoothness, Theorem 1.1 only assumes that the potential $F \colon U  \mathbb { R }$ is continuously diferentiable (an element of $C ^ { 1 } ( U , \mathbb { R } ) )$ and thus only imposes that the gradient $\nabla F$ is continuous but not, e.g., that it is locally Lipschitz continuous. Solutions of (1.1) may thus not be unique. For example, in the case of the time interval $I = [ 0 , \infty )$ and the potential $F = ( \mathbb { R } \ni v \mapsto - \textstyle \frac { 2 } { 3 } | v | ^ { 3 / 2 } \in \mathbb { R } )$ we have for every $c \in I$ that $I \ni t \mapsto - ( \operatorname* { m a x } \{ 0 , ( t - c ) / 2 \} ) ^ { 2 } \in \mathbb { R }$ is a solution of (1.1).

Theorem 1.1 assumes that the potential is definable with respect to some arbitrary given o-minimal structure. In Definition 1.6 in Subsection 1.3 below, we briefly recall the notion of a definable function. Note that all polynomials and all semialgebraic functions (see, for example, [3, Section 2.2]) are definable functions in every o-minimal structure but not every analytic function is a definable function (for instance, the function $\mathbb { R } \ni x \mapsto \sin ( x ) \in \mathbb { R }$ is not definable in any o-minimal structure). The conclusion of Theorem 1.1 even fails to hold in general if the definability assumption is replaced by analyticity (see Proposition 6.11 below and cf. also [12, Theorem 1.1, page 592]).

However, the regularity assumption does allow the following weakening: In the case where the GF trajectory $\theta ( t ) , t \in I ,$ , in Theorem 1.1 has accumulation points in U at both ends of the time interval I, the conclusion of Theorem 1.1 also holds when the potential is only locally definable. This statement is the subject of the following Corollary 1.2, which is a direct consequence of Theorem 1.1 (see Corollary 6.10 in Subsection 6.5 below). In Definition 1.7 in Subsection 1.3 below we recall the precise definition of a locally definable function.

Corollary 1.2. Let $n \in \mathbb { N } , \vartheta \in \mathbb { R } ^ { n }$ , let $U \subseteq \mathbb { R } ^ { n }$ be open, let $F \in C ^ { 1 } ( U , \mathbb { R } )$ be locally definable, let $I \subseteq \mathbb { R }$ be convex with nonempty interior, let $\theta \in C ^ { 1 } ( I , U )$ satisfy for all $t \in I$ that

$$
\theta ^ { \prime } ( t ) = - { \bigl ( } \nabla F { \bigr ) } { \bigl ( } \theta ( t ) { \bigr ) } ,\tag{1.2}
$$

let $a , b \in U$ satisfy lim $\begin{array} { r } { \operatorname* { i n f } _ { t \to \operatorname* { i n f } ( I ) } | \theta ( t ) - a | = \operatorname* { l i m } \operatorname* { i n f } _ { t \to \operatorname* { s u p } ( I ) } | \theta ( t ) - b | = 0 , } \end{array}$ , and let $P \colon U \to \mathbb { R } ^ { n }$ satisfy for all $v \in U \backslash \{ \vartheta \}$ that $P ( v ) = | v - \vartheta | ^ { - 1 } ( v - \vartheta )$ . Then $P \circ \theta$ has finite length.

Corollary 1.2 is proved as Corollary 6.10 in Subsection 6.5 below. Corollary 1.2 assumes that the potential is a locally definable function. Analytic, locally bounded semianalytic, and locally bounded subanalytic functions on open sets (cf., for example, [2, Definitions 2.1, 3.1, and 3.2]) are all examples of locally $\mathbb { R } _ { \mathrm { a n } }$ -definable functions (see Definition 1.7 below). Here, $\mathbb { R } _ { \mathrm { a n } }$ is the structure generated by all restricted real analytic functions, whose definable sets are precisely the globally subanalytic sets (see, for instance, [48, Section 3.1, page 137]). O-minimality of this structure was established in [17, 46]. Moreover, every definable function is also locally definable.

## 1.2 Related works and key novelty of this article

In the case where the potential is not only locally definable (as in Corollary 1.2) but analytic, the gradient conjecture (also known as the gradient conjecture of R. Thom) asked whether the spherical projection of a GF trajectory converging to the reference point is convergent; see, for example, [45, Section III.B, page 204], [1, Problem 18, page 64], and [35, Section IV.9, page 1592].

Finite length of the spherical projection of the GF trajectories is a stronger conclusion than mere convergence as asserted by the gradient conjecture: finite length controls the total angular motion, not only the limit of the angular motion. In the case of analytic potentials, finite spherical length of GF trajectories that converge to the reference point in the domain U (and thereby a complete solution to the classical gradient conjecture of R. Thom) was established in Kurdyka, Mostowski, & Parusi´nski [33, Theorem 7.1]. Before the complete solution of the classical gradient conjecture in [33], special cases of it (such as the two-dimensional case $n = 2 )$ had been established in [31, Theorem 3, page 60], [23, Section 1, page 134], and [40, Section 3, page 450].

It is natural to ask whether the conclusion of the gradient conjecture also holds for (locally) definable potential functions $F \colon U  \mathbb { R }$ and this extension of the gradient conjecture has sometimes been referred to as the definable Thom gradient conjecture (cf., for instance, [41] and [43, Section 7.8.3.2]).

The definable Thom gradient conjecture and finite spherical length for convergent GF trajectories have been established in the two-dimensional case $n = 2 \ [ 3 4$ , Item (2) (page 139) and §8] as well as for polynomially bounded o-minimal structures [34, Item (3) (page 139) and §9] (where an o-minimal structure is called polynomially bounded if every function from R to R that is definable with respect to this o-minimal structure grows at most polynomially).

In analogy to R. Thom’s gradient conjecture that deals with gradient trajectories that are bounded and converge to a point, there is also the gradient conjecture at infinity which asserts that the spherical projection $| \theta ( t ) | ^ { - 1 } \theta ( t )$ converges as t tends to T for every GF trajectory $\theta ( t ) , t \in [ 0 , T )$ , that escapes to infinity in the sense that lim $\begin{array} { r } { \operatorname* { i n f } _ { t  T } | \theta ( t ) | = \infty } \end{array}$ where $T \in ( 0 , \infty ]$ ; cf., for example, [20, Section 6]. In particular, in the case where the potential is $C ^ { 2 }$ and semialgebraic, [20, Theorem 5.1] not only solves the gradient conjecture at infinity but also establishes finite length of the spherical projection of every GF trajectory that escapes to infinity.

The work [12, Theorem 1.1, page 592] explicitly specifies for every $k \in \mathbb N$ a convex $C ^ { k }$ -potential $F \colon  { \mathbb { R } } ^ { 2 } \to$ R (whose restriction to $\mathbb { R } ^ { 2 } \backslash \{ 0 \}$ is analytic) for which the gradient conjecture at infinity fails to hold for the positive gradient flow of F (or, equivalently, for the negative gradient flow of the concave potential −F), with escape to infinity occurring at a finite maximal time. This does not contradict Theorem 1.1 above as the potentials in [12] are not definable. Moreover, in the setting of suitable non-Euclidean Riemannian metrics, the work [21, Proposition 5.1, page 76] specifies a smooth semialgebraic potential for which the gradient conjecture at infinity also fails to hold.

For bounded GF trajectories, Kurdyka- Lojasiewicz (KL)-type inequalities near critical points are powerful tools for establishing finite length and convergence of GF trajectories [32]. In a related direction, Bochnak– Lojasiewicz-type gradient inequalities at infinity, near asymptotic critical values, are studied in [10, Proposition 2.7, page 497], [11, Proposition 3.2, page 42], and [37, Proposition 6, page 69]. In particular, [10, Proposition 2.7, page 497] in conjunction with [37, Remark 9, page 69] establishes such an inequality for polynomially bounded o-minimal structures.

Theorem 1.1 provides afirmative answers to the definable Thom gradient conjecture (DTGC) and to the definable gradient conjecture at infinity and it not only establishes directional convergence but also finite spherical length for both regimes (at a critical point and at infinity) for any arbitrary $C ^ { 1 }$ -potential that is definable with respect to an arbitrary o-minimal structure over the reals.

In contrast to the classical Thom gradient conjecture and the DTGC, the strong gradient conjecture due to R. Moussu (which is also known as the non-oscillation conjecture) (see, for instance, [42, Section 1], [16, Theorem 5.1, page 99], [19, Theorem 2.1, item 5, pages 2519–2520], [32, Conjecture (F), page 781], and [22, Theorem 1, page 1685]) remains an open problem of research in most cases, even for analytic potential functions $F \colon U  \mathbb { R }$

For further reading and further references on the topic of the dynamics of the gradient vector field of an analytic or definable function (potential) we refer to the survey article [43].

## 1.3 Definability, o-minimal structures, and rectifiable curves

For the convenience of the reader, in this subsection, we briefly recall those concepts from analysis and ominimal geometry that are required to accurately retrace the statements of Theorem 1.1 and Corollary 1.2 above. For further reading in the area of o-minimal geometry we refer, for example, to [47, Chapter 1] and [9, Chapter 1].

Definition 1.3 (Structures on the real numbers). We say that $s$ is a structure if and only if it holds that $\begin{array} { r } { S = ( S _ { n } ) _ { n \in \mathbb { N } } \colon \mathbb { N } \to ( \bigcup _ { n = 1 } ^ { \infty } \mathcal { P } ( \mathcal { P } ( \mathbb { R } ^ { n } ) ) ) } \end{array}$ is a function from N to $\begin{array} { r } { ( \bigcup _ { n = 1 } ^ { \infty } \mathcal { P } ( \mathcal { P } ( \mathbb { R } ^ { n } ) ) ) } \end{array}$ which satisfies that

(i) it holds for every $n \in \mathbb { N }$ , every polynomial $p \colon  { \mathbb { R } } ^ { n } \to  { \mathbb { R } }$ , and $\mathrm { e v e r y ^ { 3 } }$ interval $I \subseteq \mathbb { R }$ that $p ^ { - 1 } ( I ) \in S _ { n }$

(ii) it holds for all $n \in \mathbb { N } , A , B \in { \mathcal { S } } _ { n }$ that $\{ A \cap B , A \cup B , A \backslash B \} \subseteq S _ { n } \subseteq { \mathcal { P } } ( \mathbb { R } ^ { n } )$

(iii) it holds for all $n , m \in  { \mathbb { N } } , A \in  { \mathcal { S } } _ { n } , B \in  { \mathcal { S } } _ { m }$ that $A \times B \in S _ { n + m }$ , and

(iv) it holds for all $n \in \mathbb { N } , A \in S _ { n + 1 }$ that $\textstyle \bigcup _ { x = ( x _ { 1 } , \ldots , x _ { n + 1 } ) \in A } \{ \left( x _ { 1 } , \ldots , x _ { n } \right) \} \in S _ { n }$

Definition 1.4 (O-minimal structures). Let S be a structure (cf. Definition 1.3). Then we say that S is o-minimal if and only if it holds for every $A \in S _ { 1 }$ that A is a finite union of intervals.

Definition 1.5 (Definable sets). Let S be an o-minimal structure (cf. Definitions 1.3 and 1.4). Then we say that A is an S-definable set (we say that A is definable) if and only if there exists $n \in \mathbb N$ such that $A \in S _ { n }$

Definition 1.6 (Definable functions). Let S be an o-minimal structure (cf. Definitions 1.3 and 1.4). Then we say that $f$ is an S-definable function (we say that $f$ is definable) if and only if it $\mathrm { \ h o l d s ^ { 4 } }$ that $\operatorname { G r a p h } ( f )$ is an S-definable set.

In our formulation of Corollary 1.2 above we also employ the concept of a locally definable function, which we present in Definition 1.7 below. For further details on locally definable functions we refer, for instance, to [27, Section 2].

Definition 1.7 (Locally definable functions). Let S be an o-minimal structure (cf. Definitions 1.3 and 1.4). Let $n \in \mathbb N$ , let $U \subseteq \mathbb { R } ^ { n }$ be open, and let $f \colon U \to$ R be a function. Then we say that f is a locally definable function with respect to $s$ (we say that $f$ is locally definable) if and only if for all $x \in U$ there exists $\varepsilon \in ( 0 , \infty )$ such that $f | _ { \{ y \in U : | x - y | < \varepsilon \} }$ is definable with respect to ${ \mathcal { S } } .$

In the next step we briefly recall the concept of the length of a function from an interval to a subset of R<sup>n</sup>. For further reading on curve lengths we refer, for example, to [6, Section 2.3.1].

Definition 1.8 (Length of a curve). Let $I \subseteq \mathbb { R }$ be an interval, let $n \in \mathbb { N } ,$ , let $U \subseteq \mathbb { R } ^ { n }$ be a set, and let $\gamma \colon I  U$ be a function. Then we denote by len $\mathsf { \iota } ( \gamma ) \in [ 0 , \infty ]$ the extended real number given by

$$
\begin{array} { r } { \begin{array} { c } { \mathrm { l e n } ( \gamma ) = \operatorname* { s u p } \Big ( \big \{ 0 \big \} \cup \Big \{ r \in \mathbb { R } \colon \exists m \in \mathbb { N } , t _ { 0 } , t _ { 1 } , \dots , t _ { m } \in I \colon \Big ( t _ { 0 } < t _ { 1 } < \dots < t _ { m } } \\ { \qquad \mathrm { ~ a n d ~ } r = \displaystyle \sum _ { k = 1 } ^ { m } | \gamma ( t _ { k } ) - \gamma ( t _ { k - 1 } ) | \Big ) \Big \} \Big ) , } \end{array} } \end{array}\tag{1.3}
$$

we call len(γ) the length of $\gamma _ { \mathrm { { i } } }$ , and we say that $\gamma$ has finite length if and only if $\mathrm { l e n } ( \gamma ) < \infty$

## 1.4 Connections to the training of deep neural networks (DNNs)

Even though Theorem 1.1 is a result in the area of pure mathematics, it also has strong connections to applied mathematics, optimization, and computer science, specifically, to DL, that is, to the training of deep neural networks (DNNs) by means of gradient descent based optimization algorithms.

GF trajectories are the basis for many optimization procedures in the sense that, under suitable assumptions, the solutions $\theta ( t ) , t \geq 0 ,$ of (1.1) (with the time interval being $I = [ 0 , \infty ) )$ converge to minimizers of the potential $F \colon U  \mathbb { R }$ as t tends to infinity. KL-type inequalities and related techniques are also commonly used to establish convergence of discretizations of (1.1) (gradient based optimization methods) to critical points of F under, among other hypotheses, the assumption that the considered optimization process is bounded (remains in a compactum).

For training of DNNs by means of stochastic gradient descent (SGD) based optimization methods (such as standard SGD, adaptive moment estimation SGD (Adam) [29], and momentum orthogonalized by Newton-Schulz SGD (Muon) [28]) the situation is, however, more subtle as extensive numerical simulations [13, 24, 30], complemented by analytical results in specific settings [7, 8, 18, 26, 30, 39, 44], strongly suggest that, in most cases of DNN training, the GF trajectories and their discretizations by means of SGD methods are not bounded or converge to a critical point but diverge to infinity (though note that the divergence to infinity of GF trajectories can only occur at a rather slow speed of divergence; see, for instance, [18, Proposition 2.54]).

In such a diverging scenario it is, of course, no longer possible to conclude convergence of the GF trajectory but Theorem 1.1 allows us in broad generality (definable and continuously diferentiable activation and loss) to conclude directional convergence (convergence of the spherical projection) for the DNN training flow. This is precisely the subject of the next result, Corollary 1.9, which is an immediate consequence of Theorem 1.1.

Corollary 1.9 (Directional convergence for DNN training). Let m, $L \in \mathbb { N } , \ \ell = ( \ell _ { 0 } , \ell _ { 1 } , \dots , \ell _ { L } ) \in$ $\mathbb { N } ^ { L + 1 }$ , let $\begin{array} { r } { n = \sum _ { i = 1 } ^ { L } \ell _ { i } ( \ell _ { i - 1 } + 1 ) } \end{array}$ , let $A \in C ^ { 1 } ( \mathbb { R } , \mathbb { R } ) , \mathcal { L } \in C ^ { 1 } ( \mathbb { R } ^ { \ell _ { L } } \times \mathbb { R } ^ { \ell _ { L } } \times \mathbb { R } ^ { n } , [ 0 , \infty ) )$ be definable, for every $\theta = ( \theta _ { 1 } , \overrightarrow { \dots } , \hat { \theta } _ { n } ) \in \mathbb { R } ^ { n } \ l e t \ N _ { \theta } ^ { v } = ( \ N _ { \theta , 1 } ^ { v } , \dots , \ N _ { \theta , \ell _ { n } } ^ { v } ) \colon \mathbb { R } ^ { \ell _ { 0 } } \to \mathbb { R } ^ { \ell _ { v } } , \ v \in \{ 0 , 1 , \dots , L \}$ , satisfy for all $v \in \{ 0 , 1 , \ldots , L - 1 \} , x = ( x _ { 1 } , \ldots , x _ { \ell _ { 0 } } ) \in \mathbb { R } ^ { \ell _ { 0 } } , i \in \{ 1 , 2 , \ldots , \ell _ { v + 1 } \}$ that

$$
\begin{array} { l } { { \displaystyle \mathcal { N } _ { \theta , i } ^ { v + 1 } ( x ) = \theta _ { \ell _ { v + 1 } \ell _ { v } + i + \sum _ { h = 1 } ^ { v } \ell _ { h } ( \ell _ { h - 1 } + 1 ) } } } \\ { { \displaystyle \qquad + \sum _ { j = 1 } ^ { \ell _ { v } } \theta _ { ( i - 1 ) \ell _ { v } + j + \sum _ { h = 1 } ^ { v } \ell _ { h } ( \ell _ { h - 1 } + 1 ) } \Big ( x _ { \operatorname* { m i n } \{ j , \ell _ { 0 } \} } \mathbb { 1 } _ { \{ 0 \} } ( v ) + A ( \mathcal { N } _ { \theta , j } ^ { v } ( x ) ) \mathbb { 1 } _ { \mathbb { R } } ( v ) \Big ) , } } \end{array}\tag{1.4}
$$

<sup>let</sup> <sub>all</sub> $\vartheta \in \mathbb { R } ^ { n } , x _ { 1 } , x _ { 2 } , \ldots , x _ { m } \in \mathbb { R } ^ { \ell _ { 0 } } , y _ { 1 } , y _ { 2 } , \ldots , y _ { m } \in \mathbb { R } ^ { \ell _ { L } } , l e t F \in C ( \mathbb { R } ^ { n } , \mathbb { R } )$ and $P \colon \mathbb { R } ^ { n } \to \mathbb { R } ^ { n }$ satisfy for $\theta \in \mathbb { R } ^ { n } \backslash \{ \vartheta \}$ that

$$
F ( \theta ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \mathcal { L } \big ( \mathcal { N } _ { \theta } ^ { L } ( x _ { i } ) , y _ { i } , \theta \big )\tag{1.5}
$$

and $P ( \theta ) = | \theta - \vartheta | ^ { - 1 } ( \theta - \vartheta )$ , and let $\theta \in C ^ { 1 } ( [ 0 , \infty ) , \mathbb { R } ^ { n } )$ satisfy for all $t \in [ 0 , \infty )$ that $\theta ^ { \prime } ( t ) =$ $- \left( \nabla F \right) \left( \theta ( t ) \right)$ . Then there exists a unique $v \in \mathbb { R } ^ { n }$ such that lim $\begin{array} { r } { \operatorname* { s u p } _ { t  \infty } | ( P \circ \theta ) ( t ) - v | = 0 } \end{array}$

## 1.5 Outline of the proof and structure of this article

In this subsection we provide a brief outline of the proof of Theorem 1.1 and describe the structure of the remainder of this article. Theorem 1.1 is proved as Theorem 6.8 in Section 6 below. In the first step of the proof of Theorem 6.8 we reduce Theorem 6.8 (by separately considering the gradient trajectories $\theta ( t )$ and $\theta ( - t )$ of the potentials $F$ and −F, respectively) to the special case where the time interval I is equal to the half-open interval $I = [ \tau , T )$ for some $T \in ( \tau , \infty ] , \tau \in \mathbb { R }$ . This special case is proved as Corollary 6.6 in Subsection 6.3 below.

Corollary 6.6, in turn, is proved by reducing it to the special case where the GF trajectory $\theta ( t )$ $t \in [ \tau , T )$ , never hits the reference point ϑ for the spherical projection in the sense that $\{ t \in [ \tau , T ) \colon \theta ( t ) =$ $\vartheta \} = \varnothing .$ This special case is proved as Proposition 6.4 in Subsection 6.3. The challenging cases in the proof of Proposition 6.4 are the case in which lim su $\mathrm { p } _ { t  T } | \theta ( t ) - \vartheta | = 0$ (convergence to the reference point $\vartheta \in \mathbb { R } ^ { n } ) \ -$ the case of the DTGC – and the case in which lim inf $ { \dot { \mathbf { \xi } } } _ { t  T } | \theta ( t ) | = \infty$ (divergence to infinity) – the case of the gradient conjecture at infinity. All other cases can be handled by employing the well-established classical KL theory [32, Theorem 1] (see Proposition 6.3 in Subsection 6.2 below for details).

We treat these two challenging cases in the proof of Proposition 6.4 jointly by applying an appropriate transformation to the GF DE in (1.1) above. Specifically, we transform (1.1) to DE systems on definable $C ^ { 2 }$ -submanifolds of $\mathbb { R } ^ { n + 1 }$ of the form

$$
\begin{array} { r } { r ^ { \prime } ( t ) = - \alpha ( t ) ( r ( t ) ) ^ { 2 } ( \partial _ { r } h ) ( r ( t ) , x ( t ) ) \qquad \mathrm { a n d } \qquad x ^ { \prime } ( t ) = - \alpha ( t ) ( \nabla _ { M } h ) ( r ( t ) , x ( t ) ) } \end{array}\tag{1.6}
$$

for Lebesgue-almost all $t \in I = [ \tau , T )$ and it is precisely the subject of Section 5 to study the dynamics of DE systems of the form (1.6). Here, $\alpha \colon I  [ 0 , \infty )$ is a Lebesgue measurable function, $h \colon \mathcal { O } \to \mathbb { R }$ is a definable $C ^ { 1 }$ -function on a relatively open subset O of $( 0 , \Re ) \times M$ where $\Re \in ( 0 , \infty )$ is a positive real number and where $M \subseteq \mathbb { R } ^ { n }$ is a compact and definable $C ^ { 2 } .$ -submanifold of $\mathbb { R } ^ { n }$ , and $r \colon I  ( 0 , \Re )$ and $x \colon I \to M$ are locally absolutely continuous functions such that the solution pair $( r ( t ) , x ( t ) ) , t \in I$ , takes values in O.

In the setting of Proposition 6.4 in Section 6 below and Theorem 1.1 above we have that the submanifold $M = \{ v \in \mathbb { R } ^ { n } \colon | v | = 1 \}$ is the unit sphere in $\mathbb { R } ^ { n }$ , we have that $x ( t ) = P ( \theta ( t ) )$ is the spherical projection of $\theta ( t )$ , we have that $\dot { \boldsymbol { r } } ( t ) = | \theta ( t ) - \vartheta | ^ { \nu }$ with $\nu = 1$ in the case where lim s $\begin{array} { r } { \operatorname * { l p } _ { t  T } | \theta ( t ) - \vartheta | = 0 } \end{array}$ and with $\nu = - 1$ in the case where lim inf $\dot { \mathbf { \xi } } _ { t  T } | \theta ( t ) | = \infty$ , we have that $\alpha ( t ) = ( r ( t ) ) ^ { - 2 \nu } ( \bar { 1 } + ( F ( \theta ( t ) ) ) ^ { 2 } ) ^ { 3 / 2 }$ for $t \in I .$ , we have that $\mathcal { O } = \{ ( s , u ) \in ( 0 , \Re ) \times M : \vartheta + s ^ { \nu } u \in U \}$ , and we have that

$$
h ( s , u ) = \varphi ( F ( \vartheta + s ^ { \nu } u ) )\tag{1.7}
$$

for $( s , u ) \in \mathcal { O }$ , where $\varphi \colon \mathbb { R }  ( - 1 , 1 )$ satisfies for all $e \in \mathbb { R }$ that $\varphi ( e ) = ( 1 + e ^ { 2 } ) ^ { - 1 / 2 } e$ . We note that in both cases, $\nu = 1$ and $\nu = - 1$ , we have that lim su $\begin{array} { r } { \mathsf { l p } _ { t \to T } | r ( t ) | = 0 } \end{array}$

As $r ( t )$ represents the radius $| \theta ( t ) - \vartheta |$ (in the case $\nu = 1 )$ and its inverse (in the case $\nu = - 1 )$ and as $x ( t )$ represents the angular component of $\theta ( t )$ , we refer to (1.6) as a radial-angular DE system and it is the key objective of Section 5 to establish that $x ( t )$ in (1.6) has finite length. Specifically, item (ii) in Corollary 5.7 proves under suitable assumptions that $\begin{array} { r } { \int _ { \tau } ^ { T } | x ^ { \prime } ( t ) | \mathrm { d } t < \infty } \end{array}$ and this is used in the proof of Proposition 6.4 to establish that $P \circ \theta$ has finite length. Our proof of Corollary 5.7 is based on applications of Proposition 5.3 in Subsection 5.2 below, Lemma 5.5 in Subsection 5.3 below, and Proposition 5.6 in Subsection 5.4 below.

On a technical level, the most fundamental contribution of this work is Proposition 5.3, which establishes a certain Lyapunov-type stability for solutions $( r ( t ) , x ( t ) ) \in \mathcal { O } , t \in I .$ of the radial-angular DE systems. Specifically, (5.41) in Proposition 5.3 shows that there exists $\eta \in ( 0 , \Re )$ such that for all $a , b \in I$ with $a \leq b$ and $r ( [ a , b ] ) \subseteq ( 0 , \eta )$ it holds that

$$
Q ( r ( b ) , x ( b ) ) - Q ( r ( a ) , x ( a ) ) \leq - \int _ { a } ^ { b } | x ^ { \prime } ( t ) | \mathrm { d } t + \int _ { a } ^ { b } m ( r ( t ) ) | r ^ { \prime } ( t ) | \mathrm { d } t .\tag{1.8}
$$

The function m: $( 0 , \eta )  [ 0 , \infty )$ in (1.8) is a continuous, integrable, and definable function and the function $Q \colon \mathcal { O } \cap ( ( 0 , \eta ) \times M ) \to [ 0 , \infty )$ in (1.8) is continuous and bounded. Note that (1.8) shows that $Q$ serves as a Lyapunov-type function for the radial-angular DE system in (1.6) up to the radial error term in the second integral on the right hand side of (1.8). Furthermore, we observe that rearranging (1.8) shows that for all $a , b \in I$ with $a \leq b$ and $r ( [ a , b ] ) \subseteq ( 0 , \eta )$ it holds that

$$
\int _ { a } ^ { b } | x ^ { \prime } ( t ) | \mathrm { d } t \leq \int _ { a } ^ { b } m ( r ( t ) ) | r ^ { \prime } ( t ) | \mathrm { d } t + \left[ \operatorname* { s u p } _ { v \in \mathcal { O } \cap ( ( 0 , \eta ) \times M ) } Q ( v ) \right] .\tag{1.9}
$$

Note that (1.9) is only applicable to subintervals $[ a , b ] \subseteq I$ of I on which the radial component r is smaller than η in the sense that $r ( [ a , b ] ) \subseteq ( 0 , \eta )$ . However, as r converges to zero in the sense that lim $\begin{array} { r } { \operatorname* { s u p } _ { t \to T } | r ( t ) | = 0 . } \end{array}$ , we obtain that there exists $\sigma \in I$ which satisfies $r ( I \cap [ \sigma , \infty ) ) \subseteq ( 0 , \eta )$ . In the proof of Corollary 5.7 we also employ Lemma 5.5 and Proposition 5.6 to show, under suitable assumptions, that the integral on the right hand side of (1.9) can be uniformly bounded. Combining this with (1.9) and the fact that Q is bounded then proves the finite length estimate for the tail $\begin{array} { r } { \int _ { \sigma } ^ { T } | x ^ { \prime } ( t ) | \mathrm { d } t < \infty } \end{array}$ . This and the fact that for every locally absolutely continuous y : $I \to \mathbb { R } ^ { n }$ we have that $\begin{array} { r } { \int _ { \tau } ^ { \sigma } \vert y ^ { \prime } ( t ) \vert \mathrm { d } t < \infty } \end{array}$ then allow us to establish, in item (ii) in Corollary 5.7, the finite length estimate $\begin{array} { r } { \int _ { \tau } ^ { T } | x ^ { \prime } ( t ) | \dot { \mathrm { d } } t = \int _ { \tau } ^ { \sigma } | x ^ { \prime } ( t ) | \mathrm { d } t + \int _ { \sigma } ^ { T } | x ^ { \prime } ( t ) | \mathrm { d } t < } \end{array}$ ∞.

The central task in the proof of Proposition 5.3 is thus to construct the Lyapunov-type function $Q$ so that (1.8) holds. One could try to directly employ the function h in (1.6) as a Lyapunov-type function and applying the fundamental theorem of calculus and the chain rule to the Lyapunov-type energy $h ( r ( t ) , x ( t ) ) , t \in I$ , shows that for all $a , b \in I$ with $a < b$ it holds that

$$
\begin{array} { l } { \displaystyle h ( r ( b ) , x ( b ) ) - h ( r ( a ) , x ( a ) ) = \int _ { a } ^ { b } \big ( ( \partial _ { r } h ) ( r ( t ) , x ( t ) ) r ^ { \prime } ( t ) + \langle ( \nabla _ { M } h ) ( r ( t ) , x ( t ) ) , x ^ { \prime } ( t ) \rangle \big ) \mathrm { d } t } \\ { \displaystyle = - \int _ { a } ^ { b } \alpha ( t ) \big ( ( r ( t ) ) ^ { 2 } | ( \partial _ { r } h ) ( r ( t ) , x ( t ) ) | ^ { 2 } + | ( \nabla _ { M } h ) ( r ( t ) , x ( t ) ) | ^ { 2 } \big ) \mathrm { d } t } \\ { \displaystyle \leq - \int _ { a } ^ { b } \alpha ( t ) | ( \nabla _ { M } h ) ( r ( t ) , x ( t ) ) | ^ { 2 } \mathrm { d } t . } \end{array}\tag{1.10}
$$

This shows that the Lyapunov energy $h ( r ( t ) , x ( t ) ) , \ t \ \in \ I _ { : }$ , is non-increasing and that h is a Lyapunov function for (1.6) but the dissipation $\begin{array} { r } { \int _ { a } ^ { b } \alpha ( t ) | ( \nabla _ { M } h ) ( r ( t ) , x ( t ) ) | ^ { 2 } \mathrm { d } t \ < \ \infty } \end{array}$ in (1.10) associated to the Lyapunov function h is not large enough to directly establish finiteness of the length of x, i.e., $\begin{array} { r } { \int _ { \tau } ^ { T } | x ^ { \prime } ( t ) | \mathrm { d } t = \int _ { \tau } ^ { T } \alpha ( t ) | ( \nabla _ { M } h ) ( r ( t ) , x ( t ) ) | } \end{array}$ dt (see the second equality in (1.6)). The problem is that the norm of the gradient $( \nabla _ { M } h ) ( r ( t ) , x ( t ) )$ appears quadratically inside the integral in (1.10) but in the length the norm of the gradient $( \nabla _ { M } h ) ( r ( t ) , x ( t ) )$ ) appears only linearly inside the integral and if the norm of the gradient $| ( \nabla _ { M } h ) ( r ( t ) , x ( t ) ) |$ is small, then $| ( \nabla _ { M } h ) ( r ( t ) , x ( t ) ) | ^ { 2 }$ is much smaller than $| ( \nabla _ { M } h ) ( r ( t ) , x ( t ) ) |$

We want to make use of (1.10) and our objective is to transform the energy $h ( r ( t ) , x ( t ) )$ so that the dissipation of the transformed energy/the transformed Lyapunov-type function is large enough to control – up to a radial error (see (1.8) and (1.9) above) – the length $\begin{array} { r l r } { \int _ { \tau } ^ { \bar { T } } \vert x ^ { \prime } ( t ) \vert \mathrm { d } t } \end{array}$ . The transformation may, however, depend not only on the Lyapunov energy $h ( r ( t ) , x ( t ) )$ but also on the value of the radial component r(t) $r ( t )$ and we consider Lyapunov-type functions of the form $Q ( r , x ) = \Phi ( r , h ( r , x ) )$ for some suitable function Φ. Specifically, the approach initiated by Lemma 3.5 in Section 3 below is to construct a natural number $N \in \mathbb N$ , strictly positive reals $\rho \in ( 0 , \Re ) , H \in ( 0 , \infty )$ , definable $[ 0 , H ]$ -valued transformation functions $\mathcal { F } _ { 0 } , \mathcal { F } _ { 1 } , \ldots , \mathcal { F } _ { N - 1 }$ , definable back-transformation functions $E _ { 1 } , E _ { 2 } , \dots , E _ { N }$ , and definable energy boundary functions $\mathfrak { e } _ { 0 } , \mathfrak { e } _ { 1 } , \ldots , \mathfrak { e } _ { N }$ that specify pairwise disjoint regions in the radial-energy plane $\mathbf { H } _ { 0 } , \mathbf { H } _ { 1 } , \dots , \mathbf { H } _ { N - 1 }$ through $\forall i \in \{ 0 , 1 , \ldots , N - 1 \} : \mathbf { H } _ { i } = \{ ( \mathbf { r } , \mathfrak { h } ) \in ( 0 , \rho ) \times \mathbb { R } \colon \mathfrak { e } _ { i } ( \mathbf { r } ) < \mathfrak { h } < \mathfrak { e } _ { i + 1 } ( \mathbf { r } ) \}$ and to consider the transformed energy

$$
\mathcal { F } _ { i } \big ( r ( t ) , h \big ( r ( t ) , x ( t ) \big ) \big )\tag{1.11}
$$

in the regime where $( r ( t ) , h ( r ( t ) , x ( t ) ) ) \in \mathbf { H } _ { i }$ for $t \in I , i \in \{ 0 , 1 , \ldots , N - 1 \}$ . In the following we denote for every $i \in \{ 0 , 1 , \ldots , N - 1 \}$ by $Z _ { i } ( r , x ) = \mathcal { F } _ { i } ( r , h ( r , x ) )$ the transformed energy associated to radial component r and angular component x, with $Z _ { i } \colon { \mathcal { O } } \cap ( ( 0 , \rho ) \times M ) \to [ 0 , H ]$ mapping from $\mathcal { O } \cap ( ( 0 , \rho ) \times M )$ to [0, H]. In Lemma 3.5 we also supply crucial (lower and) upper bounds for the partial derivative with respect to the transformed energy variable $( \partial _ { z } E _ { i } ) ( r , z )$ of the back-transformations $E _ { i } , \ : i \in \left\{ 1 , 2 , \ldots , N \right\}$ Specifically, the fundamental theorem of calculus, the chain rule, the inverse function diferentiation rule, the second equation in (1.6), and Lemma 3.5 show that for all $i \in \{ 0 , 1 , \ldots , N - 1 \} , a , b \in I$ with $a < b$ and $\bigcup _ { t \in [ a , b ] } \{ { \bigl ( } r ( t ) , h ( r ( t ) , x ( t ) ) { \bigr ) } \} \subseteq \mathbf { H } _ { i }$ it holds that

$$
\begin{array} { r l } & { \displaystyle Z _ { i } ( r ( b ) , x ( b ) ) - Z _ { i } ( r ( a ) , x ( a ) ) } \\ & { \displaystyle \quad = \int _ { a } ^ { b } \big ( ( \partial _ { r } Z _ { i } ) ( r ( t ) , x ( t ) ) r ^ { \prime } ( t ) + \langle ( \nabla _ { M } Z _ { i } ) ( r ( t ) , x ( t ) ) , x ^ { \prime } ( t ) \rangle \big ) \mathrm { d } t } \\ & { \displaystyle \quad = \int _ { a } ^ { b } ( \partial _ { r } Z _ { i } ) ( r ( t ) , x ( t ) ) r ^ { \prime } ( t ) \mathrm { d } t - \int _ { a } ^ { b } \frac { \vert ( \nabla _ { M } h ) ( r ( t ) , x ( t ) ) \vert \vert x ^ { \prime } ( t ) \vert } { ( \partial _ { z } E _ { i + 1 } ) ( r ( t ) , Z _ { i } ( r ( t ) , x ( t ) ) ) } \mathrm { d } t } \\ & { \displaystyle \quad \leq \int _ { a } ^ { b } ( \partial _ { r } Z _ { i } ) ( r ( t ) , x ( t ) ) r ^ { \prime } ( t ) \mathrm { d } t - \frac { 1 } { c } \int _ { a } ^ { b } \vert x ^ { \prime } ( t ) \vert \mathrm { d } t . } \end{array}\tag{1.12}
$$

Here the final inequality uses $0 < ( \partial _ { z } E _ { i + 1 } ) ( r ( t ) , Z _ { i } ( r ( t ) , x ( t ) ) ) \leq c | ( \nabla _ { M } h ) ( r ( t ) , x ( t ) ) |$ with a strictly positive real constant $c \in ( 0 , \infty )$ according to items (v) and (vi) in Lemma 3.5. The inequality in (1.12) does not, however, directly provide us with (1.8) as (1.12) only applies to solutions with the radial component and the energy staying in one fixed radial-energy region $\mathbf { H } _ { i }$ and as the dependence of the transformation functions $\mathcal { F } _ { 0 } , \mathcal { F } _ { 1 } , \ldots , \mathcal { F } _ { N - 1 }$ not only on the energy but also on the radial component produces an additional radial error term (see the first integral on the right hand side of (1.12)).

To this end, we establish in Lemma 3.7 in Section 3 below upper bounds for the absolute diference between the radial partial derivative of the energy $\partial _ { r } h$ and the radial partial derivative of the backtransformations $\partial _ { r } E _ { i + 1 }$ involving nonnegative, definable, and integrable functions $\mathbf { m } _ { i } ( r )$ . We use these estimates to explicitly specify suitable definable $C ^ { 1 }$ weight functions $\omega _ { i } \colon ( 0 , \eta ) \times ( 0 , H ) \to ( 0 , 1 ] , i \ \in$ $\{ 0 , 1 , \ldots , N - 1 \}$ , (see (5.63) in the proof of Proposition 5.3 below) that balance the radial error term (see the first integral on the right hand side of (1.12)) against the angular dissipation term with the right sign (see the second integral on the right hand side of (1.12)), where $\eta \in ( 0 , \rho )$ is a suficiently small strictly positive real number. For every $i \in \{ 0 , 1 , \ldots , N - 1 \}$ we denote by $P _ { i } \colon ( 0 , \eta ) \times [ 0 , H ] \to [ 0 , H ]$ the one-parameter primitive (with respect to the transformed energy variable) of the weight function $\omega _ { i }$ in the sense that for all $r \in ( 0 , \eta ) , z \in [ 0 , H ]$ we have that $\begin{array} { r } { P _ { i } ( r , z ) = \int _ { 0 } ^ { z } \omega _ { i } ( r , u ) } \end{array}$ du and the Lyapunov-type function in (1.8) is then, for every $( r , x ) \in \mathcal { O } \cap ( ( 0 , \eta ) \times M )$ , given by

$$
Q ( r , x ) = \kappa \sum _ { i = 0 } ^ { N - 1 } P _ { i } ( r , Z _ { i } ( r , x ) ) = \kappa \sum _ { i = 0 } ^ { N - 1 } P _ { i } \big ( r , \mathcal { F } _ { i } ( r , h ( r , x ) ) \big )\tag{1.13}
$$

where $\kappa \in ( 0 , \infty )$ is a suficiently large real number. Lemma 5.1 in Section 5 provides regularity properties for $Q$ and, in particular, allows us to conclude that $Q$ is continuous. Moreover, the fact that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ we have that im $' P _ { i } ) \subseteq [ 0 , H ]$ shows for all $( r , x ) \in \mathcal { O } \cap ( ( 0 , \eta ) \times M )$ that $0 \leq$ $Q ( r , x ) \leq \kappa N H$ . Hence, Q is a bounded and continuous Lyapunov-type function.

To prove (1.8), we also need to handle the boundaries between the radial-energy regions, at which the energy transformation functions $\mathcal { F } _ { 0 } , \mathcal { F } _ { 1 } , \ldots , \mathcal { F } _ { N - 1 }$ need not be diferentiable. To solve this issue we employ a truncation and approximation procedure for the transformed energy $Z _ { i } ( r , x )$ . Specifically, we approximate $Z _ { i } ( r , x )$ by max{ε, min{ $\langle H - \varepsilon , Z _ { i } ( r , x ) \rangle \}$ for $\varepsilon \in ( 0 , H / 2 )$ and, thereby, also obtain approximations for the Lyapunov-type function Q in (1.13) and in Lemma 4.3 in Section 4 we establish regularity properties and estimates for the approximated quantities and functions. These regularity properties and estimates allow us to apply the fundamental theorem of calculus for absolutely continuous functions to the approximated quantities and, thereafter, to pass to the limit as the truncation parameter $\varepsilon$ tends to zero. For the times $t \in I$ spent exactly at the boundaries of the radial-energy regions in the sense that $h ( r ( t ) , x ( t ) ) = \mathfrak { e } _ { i } ( r ( t ) )$ , we apply the elementary result in Lemma 4.1 (which shows for every $c \in \mathbb { R }$ that the derivative of an absolutely continuous function vanishes almost everywhere on its c-level set) to the real valued function $t \mapsto h ( r ( t ) , x ( t ) ) - \mathfrak { e } _ { i } ( r ( t ) )$ ).

In the proof of Proposition 5.3 we actually perform the arguments and transformation procedures (1.11)–(1.13) outlined above not directly with the energy function $h \colon \mathcal { O } \to$ R from (1.6) but with the normalized energy function $\mathbf { h } = \varphi \circ h \colon \mathcal { O } \to ( - 1 , 1 )$ . (This is possible as a solution $( r ( t ) , x ( t ) ) \in \mathcal { O } , t \in I ,$ of (1.6) is also a solution of (1.6) but with $h \colon \mathcal { O } \to \mathbb { R }$ replaced by h: $\mathcal { O }  ( - 1 , 1 )$ and $\alpha \colon I  [ 0 , \infty )$ replaced by a modified $\widetilde { \alpha } \colon I  [ 0 , \infty )$ ; see (5.124) in the proof of Proposition 5.3 for details.) However, if $h$ emerges from the polar coordinate type transformation in the proof of Proposition 6.4, in which case we have for all $( s , u ) \in \mathcal { O }$ that $h ( s , u ) = \varphi ( F ( \vartheta + s ^ { \nu } u ) )$ ) so that h already takes values in (−1, 1) (see (1.7) above), then this additional normalization step is not needed and the procedure (1.11)–(1.13) is directly applicable as described above.

## 2 Regularity properties of radial upper bounds for definable C<sup>1</sup>-scalar fields

In the main results of this section, Lemma 2.11 and Lemma 2.13 in Subsection 2.3 below, we establish integrability, smoothness, and definability properties of radial upper bounds for definable $C ^ { 1 }$ -scalar fields on definable $C ^ { r }$ -submanifolds of $\mathbb { R } ^ { n }$

In particular, in Lemma 2.11 we consider a definable $C ^ { 1 }$ -scalar field $Z = ( Z ( r , x ) ) _ { ( r , x ) \in U } : U $ R on a relatively open subset $U \subseteq ( 0 , \mathfrak { r } _ { 0 } ) \times M$ of the Cartesian product of an open interval $( 0 , \mathfrak { r } _ { 0 } )$ and a bounded and definable $C ^ { 1 } .$ -submanifold M of $\mathbb { R } ^ { n }$ and establish an upper bound for the partial derivative with respect to the radial variable $( \partial _ { r } Z ) ( r , x )$ using a definable and (locally) integrable $C ^ { 1 } .$ -function $\eta ( r )$ . We employ Lemma 2.11 in the proof of Lemma 3.7 (which establishes lower and upper bounds for the partial derivative with respect to the radial variable of the energy back-transformations) in Section 3 below and (through its consequence in Lemma 2.12) in the proof of Lemma 5.1 (which establishes regularity properties for one-parameter primitives of definable two-parameter functions, in particular, for the weight functions $\omega _ { i } \colon ( 0 , \tau ) \times ( 0 , H ) \to ( 0 , 1 ] \mathrm { ~ f o r ~ } i \in \{ 0 , 1 , \ldots , N - 1 \} \mathrm { ~ }$ ; cf. (1.13)) in Section 5 below. Lemma $3 . 7$ and Lemma 5.1 are both used in the proof of Proposition 5.3 to establish Lyapunov-type stability for the radial-angular DE system.

In Lemma 2.13 we consider a definable $C ^ { 1 }$ -function $a \colon ( 0 , \mathfrak { r } _ { 0 } ) \to \mathbb { R }$ and a definable $C ^ { 1 }$ -scalar field $h = ( h ( r , x ) ) _ { ( r , x ) \in \mathcal { O } } : \mathcal { O } \to \mathbb { R }$ on a relatively open subset $\mathcal { O } \subseteq ( 0 , \mathfrak { r } _ { 0 } ) \times M \ \mathrm { o f } \ ( 0 , \mathfrak { r } _ { 0 } ) \times M$ , where $M \subseteq \mathbb { R } ^ { n }$ is a bounded and definable $C ^ { 1 }$ -submanifold of $\mathbb { R } ^ { n }$ , and we establish an upper bound of the form $| ( \partial _ { r } h ) ( r , x ) -$ $a ^ { \prime } ( r ) | \leq \beta ( r ) | ( \nabla _ { M } h ) ( r , x ) |$ with the constraint $h ( r , x ) = a ( r )$ for suficiently small r using a definable and (locally) integrable $C ^ { 1 }$ -function $\beta ( r )$ . We employ Lemma 2.13 in the proof of Proposition 5.3 (which establishes Lyapunov-type stability for the radial-angular DE system) to control the radial error (cf. the second integral on the right hand side of (1.8)) at times t at which the energy $h ( r ( t ) , x ( t ) )$ ) coincides with an energy boundary value in the sense that $\begin{array} { r } { h ( r ( t ) , x ( t ) ) \in \bigcup _ { i = 0 } ^ { N } \{ \mathfrak { e } _ { i } ( r ( t ) ) \} } \end{array}$

Before we formulate and prove Lemma 2.11 and Lemma 2.13, we first recall several well-known diferential geometric preliminaries in Subsections 2.1 and 2.2 below (cf., for example, [5, Chapter 3]). We include proofs for the statements in Subsections 2.1 and 2.2 for completeness. In Subsection 2.3 we then use some of the auxiliary results from Subsection 2.2 to prove Lemma 2.11 and Lemma 2.13.

## 2.1 Embedded submanifolds

In the following notion, Definition 2.1, we recall the standard concept of an embedded m-dimensional C<sup>r</sup>-submanifold of $\mathbb { R } ^ { n }$ . For further reading we refer, for instance, to [5, Definition 3.10 and Theorem 3.12].

Definition 2.1 (Embedded submanifolds with specified dimension). Let $r , n \in \mathbb { N } ,$ , m $\in \ \mathbb { N } _ { 0 }$ and let $M \subseteq \mathbb { R } ^ { n }$ be a set. Then we say that M is an m-dimensional $C ^ { r }$ -submanifold of $\mathbb { R } ^ { n }$ if and only if for every $x \in M$ there exist an open $U \subseteq \mathbb { R } ^ { m }$ , an open neighbourhood $V \subseteq \mathbb { R } ^ { n }$ of $x ,$ and a homeomorphism $\phi \in C ^ { r } ( U , V \cap M )$ such that for all $u \in U$ it holds that

$$
\operatorname { r a n k } ( \phi ^ { \prime } ( u ) ) = m .\tag{2.1}
$$

Definition 2.2 (Embedded submanifolds with unspecified dimension). Let $r , n \in \mathbb { N }$ and let $M \subseteq \mathbb { R } ^ { n }$ be a set. Then we say that M is $\mathbf { a } ~ C ^ { r }$ -submanifold of $\mathbb { R } ^ { n }$ if and only if there exists $m \in  { \mathbb { N } } _ { 0 }$ such that M is a m-dimensional $C ^ { r }$ -submanifold of $\mathbb { R } ^ { n }$ (cf. Definition 2.1).

In the next notion we recall the concept of the tangent space of a submanifold at a point (cf., for example, [5, Definition 3.14] for further reading).

Definition 2.3 (Tangent spaces). Let $n \in \mathbb { N }$ , let $M \subseteq \mathbb { R } ^ { n }$ be a $C ^ { 1 }$ -submanifold of $\mathbb { R } ^ { n }$ , and let $x \in M$ (cf. Definition 2.2). Then we denote by $T _ { x } M$ the set given by

$$
T _ { x } M = \left\{ v \in \mathbb { R } ^ { n } \colon [ \exists \varepsilon \in ( 0 , \infty ) , \gamma \in C ^ { 1 } ( ( - \varepsilon , \varepsilon ) , M ) \colon \left( ( \gamma ( 0 ) = x ) \wedge ( \gamma ^ { \prime } ( 0 ) = v ) \right) ] \right\} .\tag{2.2}
$$

In the next lemma we recall a well-known local representation of a $C ^ { 1 }$ -submanifold of $\mathbb { R } ^ { n }$ by a difeomorphism (cf., for instance, [5, Theorem 3.12]).

Lemma 2.4 (Local normal forms of embedded submanifolds). Let $n \in  { \mathbb { N } } , \ m \in  { \mathbb { N } } _ { 0 }$ , let $M \subseteq \mathbb { R } ^ { n }$ be an m-dimensional $C ^ { 1 }$ -submanifold of R<sup>n</sup>, and let $x \in M \ / c f .$ Definition $\it { 2 . 1 ) }$ . Then there exist an open neighbourhood $U \subseteq \mathbb { R } ^ { n }$ of x, an open $V \subseteq \mathbb { R } ^ { n }$ , and a C<sup>1</sup>-difeomorphism $\Phi \colon U  V$ such that

$$
\Phi ( x ) = 0 \qquad { a n d } \qquad \Phi ( M \cap U ) = V \cap ( \mathbb { R } ^ { m } \times \{ 0 \} ^ { n - m } ) .\tag{2.3}
$$

Proof of Lemma $\it { 2 . 4 } .$ . Throughout this proof let $P \colon \mathbb { R } ^ { n }  \mathbb { R } ^ { n - m }$ and $Q \colon \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ satisfy for all $y =$ $( y _ { 1 } , \dots , y _ { n } ) \in \mathbb { R } ^ { n }$ that

$$
P ( y _ { 1 } , \ldots , y _ { n } ) = ( y _ { m + 1 } , \ldots , y _ { n } ) \qquad { \mathrm { a n d } } \qquad Q ( y _ { 1 } , \ldots , y _ { n } ) = ( y _ { 1 } , \ldots , y _ { m } ) .\tag{2.4}
$$

Observe that the assumption that M is a m-dimensional $C ^ { 1 }$ -submanifold of $\mathbb { R } ^ { n }$ ensures that there exist an open $A \subseteq \mathbb { R } ^ { m }$ , a point $\xi \in A$ , a set $O \subseteq M$ open in $M .$ , and an injective $\phi \in C ^ { 1 } ( A , \mathbb { R } ^ { n } )$ which satisfy that $\phi ( \xi ) = x \in M , \phi ( A ) = O , \phi \colon A \to O$ is a homeomorphism, and

$$
\forall u \in A \colon \operatorname { r a n k } \phi ^ { \prime } ( u ) = m .\tag{2.5}
$$

Note that (2.5) implies that there exist distinct $i _ { 1 } , i _ { 2 } , \ldots , i _ { m } \in \{ 1 , 2 , \ldots , n \}$ and $\pi \colon \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ which satisfy that for all $y = ( y _ { 1 } , y _ { 2 } , \ldots , y _ { n } ) \in \mathbb { R } ^ { n }$ it holds that

$$
\pi ( y ) = ( y _ { i _ { 1 } } , y _ { i _ { 2 } } , \ldots , y _ { i _ { m } } )\tag{2.6}
$$

and that

$$
\pi \circ \phi ^ { \prime } ( \xi )\tag{2.7}
$$

is a linear isomorphism. Observe that (2.7) shows that there exist distinct $j _ { 1 } , j _ { 2 } , \ldots , j _ { n - m } \in$ $\{ 1 , 2 , \ldots , n \} \backslash \{ i _ { 1 } , i _ { 2 } , \ldots , i _ { m } \}$ and a linear isomorphism $L \colon \mathbb { R } ^ { n } \ \to \ \mathbb { R } ^ { m } \ \times \ \mathbb { R } ^ { n - m }$ which satisfy that for all $y = ( y _ { 1 } , y _ { 2 } , \ldots , y _ { n } ) \in \mathbb { R } ^ { n }$ it holds that

$$
L ( y ) = \left( ( y _ { i _ { 1 } } , y _ { i _ { 2 } } , \ldots , y _ { i _ { m } } ) , ( y _ { j _ { 1 } } , y _ { j _ { 2 } } , \ldots , y _ { j _ { n - m } } ) \right)\tag{2.8}
$$

and that

$$
( Q \circ L \circ \phi ) ^ { \prime } ( \xi ) = \pi \circ \phi ^ { \prime } ( \xi )\tag{2.9}
$$

is a linear isomorphism. In the following let $h = Q \circ L \circ \phi$ . Note that (2.9) and the fact that $h = Q \circ L \circ \phi$ prove that $h \in C ^ { 1 } ( A , \mathbb { R } ^ { m } )$ and its derivative

$$
h ^ { \prime } ( \xi ) = ( Q \circ L \circ \phi ) ^ { \prime } ( \xi ) \colon \mathbb { R } ^ { m } \to \mathbb { R } ^ { m }\tag{2.10}
$$

is a linear isomorphism. Combining the fact that A is open and contains ξ and the $C ^ { 1 }$ inverse-function theorem applied to h at ξ therefore demonstrates that there exist open $W , B \subseteq \mathbb { R } ^ { m }$ which satisfy that $\xi \in W \subseteq A , h ( \xi ) \in B , h ( W ) = B$ , and the restriction

$$
\mathbf { h } = h | _ { W } \colon W \to B\tag{2.11}
$$

is bijective with h $\in C ^ { 1 } ( W , B )$ and $\mathbf { h } ^ { - 1 } \in C ^ { 1 } ( B , W )$ . In the following let $a = \mathbf { h } ( \xi ) \in B , b = P ( L ( x ) ) \in$ $\mathbb { R } ^ { n - m }$ , and let $f \colon B \to \mathbb { R } ^ { n - m }$ satisfy $f = P \circ L \circ \phi \circ \mathbf { h } ^ { - 1 }$ . Observe that (2.8), (2.11), the fact that $\phi ( \xi ) = x ,$ and the fact that $h = Q \circ L$ ◦ ϕ establish that

$$
L ( x ) = ( \mathbf { h } ( \xi ) , P ( L ( x ) ) ) = ( a , b ) .\tag{2.12}
$$

Moreover, note that the fact that $a = \mathbf { h } ( \xi )$ and the fact that $\mathbf { h } ^ { - 1 } \circ \mathbf { h } = \mathrm { I d } _ { W }$ ensure that

$$
\mathbf { h } ^ { - 1 } ( a ) = \mathbf { h } ^ { - 1 } ( \mathbf { h } ( \boldsymbol { \xi } ) ) = \boldsymbol { \xi } .\tag{2.13}
$$

This, the fact that $P \in C ^ { 1 } ( \mathbb { R } ^ { n } , \mathbb { R } ^ { n - m } ) , L \in C ^ { 1 } ( \mathbb { R } ^ { n } , \mathbb { R } ^ { m } \times \mathbb { R } ^ { n - m } ) , \phi \in C ^ { 1 } ( A , \mathbb { R } ^ { n } )$ , and $\mathbf { h } ^ { - 1 } \in C ^ { 1 } ( B , W )$ the fact that $\phi ( \xi ) = x$ , the fact that $b = P ( L ( x ) )$ , and the fact that $f = P \circ L \circ \phi \circ \mathbf { h } ^ { - 1 }$ imply that $f \in C ^ { 1 } ( B , \mathbb { R } ^ { n - m } )$ and that

$$
f ( a ) = P ( L ( \phi ( \mathbf { h } ^ { - 1 } ( a ) ) ) ) = P ( L ( x ) ) = b .\tag{2.14}
$$

In the next step we combine (2.11), the fact that $\mathbf { h } \circ \mathbf { h } ^ { - 1 } = \mathrm { I d } _ { B }$ , the fact that for all $u \in A$ it holds that $h ( u ) = Q ( L ( \phi ( u ) ) ,$ ), the fact that $f = P \circ L \circ \phi \circ \mathbf { h } ^ { - 1 }$ , and the fact that for all $y \in \mathbb { R } ^ { n }$ it holds that $( Q ( y ) , P ( y ) ) = y$ to obtain that for all $u \in B$ it holds that

$$
\begin{array} { r } { ( L \circ \phi \circ \mathbf { h } ^ { - 1 } ) ( u ) = \left( \mathbf { h } ( \mathbf { h } ^ { - 1 } ( u ) ) , f ( u ) \right) = ( u , f ( u ) ) . } \end{array}\tag{2.15}
$$

Furthermore, observe that the fact that W is open in A, the fact that ϕ: $A  O$ is a homeomorphism, the fact that O is open in M, the fact that $\xi \in W$ , and the fact that $\phi ( \xi ) = x$ show that there exists an open $C \subseteq \mathbb { R } ^ { n }$ which satisfies

$$
x \in C \qquad \mathrm { a n d } \qquad M \cap C = \phi ( W ) .\tag{2.16}
$$

Note that (2.12), (2.16), the fact that L is a homeomorphism, the fact that $a \in B ,$ and the fact that B is open prove that there exist open $K \subseteq B , H \subseteq \mathbb { R } ^ { n - m }$ which satisfy

$$
( a , b ) \in K \times H \subseteq L ( C ) .\tag{2.17}
$$

In the following let $G \subseteq B , U \subseteq \mathbb { R } ^ { n }$ satisfy

$$
G = K \cap f ^ { - 1 } ( H ) \qquad { \mathrm { a n d } } \qquad U = L ^ { - 1 } ( G \times H ) .\tag{2.18}
$$

Observe that (2.12), (2.14), (2.17), (2.18), the fact that B and K are open, the fact that H is open, the fact that f is continuous, and the fact that $L \colon \mathbb { R } ^ { n }  \mathbb { R } ^ { m } \times \mathbb { R } ^ { n - m }$ is a homeomorphism demonstrate that G is open, U is open, and

$$
a \in G \subseteq B , \qquad x \in U \subseteq C , \qquad { \mathrm { a n d } } \qquad f ( G ) \subseteq H .\tag{2.19}
$$

Combining (2.15), (2.16), (2.18), the fact that $\mathbf { h } ^ { - 1 } ( B ) = W ,$ and the fact that L is bijective hence establishes that

$$
L ( M \cap U ) = L ( \phi ( W ) ) \cap ( G \times H ) = \{ ( u , f ( u ) ) \in G \times H \colon u \in G \} .\tag{2.20}
$$

In the following let

$$
V = \{ ( u - a , v - f ( u ) ) \in \mathbb { R } ^ { m } \times \mathbb { R } ^ { n - m } \colon ( u , v ) \in G \times H \}\tag{2.21}
$$

and let $\Phi \colon U  V$ satisfy for all $( u , v ) \in G \times H$ that

$$
\Phi ( L ^ { - 1 } ( u , v ) ) = ( u - a , v - f ( u ) ) .\tag{2.22}
$$

Note that (2.12), (2.21), (2.22), the fact that $b = f ( a )$ , the fact that $a \in G \subseteq B$ , the fact that G is open and H is open, and the fact that $f \in C ^ { 1 } ( B , \mathbb { R } ^ { n - m } )$ ensure that

$$
V = \big \{ ( s , t ) \in \mathbb { R } ^ { m } \times \mathbb { R } ^ { n - m } \colon \big [ ( s + a \in G ) \wedge ( t + f ( s + a ) \in H ) \big ] \big \} ,\tag{2.23}
$$

V is open, $0 \in V , \Phi ( x ) = 0$ , and $\Phi \in C ^ { 1 } ( U , V )$ . In the following let $\Theta \colon V  U$ satisfy for all $( s , t ) \in V$ that

$$
\Theta ( s , t ) = L ^ { - 1 } { \big ( } s + a , t + f ( s + a ) { \big ) } .\tag{2.24}
$$

Observe that (2.23), the fact that $U = L ^ { - 1 } ( G \times H )$ , the fact that $G \subseteq B .$ , and the fact that $f \in$ $C ^ { 1 } ( B , \mathbb { R } ^ { n - m } )$ imply that $\Theta \in C ^ { 1 } ( V , U )$ . Next, note that (2.22) and (2.24) show that for all $( u , v ) \in G \times H$ and $( s , t ) \in V$ it holds that

$$
\Theta \big ( \Phi \big ( L ^ { - 1 } ( u , v ) \big ) \big ) = L ^ { - 1 } ( u , v ) \qquad \mathrm { a n d } \qquad \Phi ( \Theta ( s , t ) ) = ( s , t ) .\tag{2.25}
$$

This and the fact that $\Phi \in C ^ { 1 } ( U , V )$ and $\Theta \in C ^ { 1 } ( V , U )$ prove that $\Phi \colon U  V$ is a $C ^ { 1 }$ difeomorphism and it holds for all $( s , t ) \in V$ that

$$
\Phi ^ { - 1 } ( s , t ) = \Theta ( s , t ) = L ^ { - 1 } { \big ( } s + a , t + f ( s + a ) { \big ) } .\tag{2.26}
$$

Next we combine the fact that $f ( G ) \subseteq H$ , the fact that $U = L ^ { - 1 } ( G \times H )$ , and (2.22) to obtain that for all $u \in G$ it holds that $( u , f ( u ) ) \in G \times H , L ^ { - 1 } ( u , f ( u ) ) \in U$ , and

$$
\Phi ( L ^ { - 1 } ( u , f ( u ) ) ) = ( u - a , f ( u ) - f ( u ) ) = ( u - a , 0 ) .\tag{2.27}
$$

Combining (2.20) and (2.21) therefore demonstrates that

$$
\Phi ( M \cap U ) = V \cap ( \mathbb { R } ^ { m } \times \{ 0 \} ^ { n - m } ) .\tag{2.28}
$$

The proof of Lemma 2.4 is thus complete.

## 2.2 Smooth functions on embedded submanifolds

In the following notion we recall the concept of the set of all S-valued r-times continuously diferentiable functions on an m-dimensional C<sup>r</sup>-submanifold M of $\mathbb { R } ^ { n }$ (cf., for example, [5, Definition 8.5 and Remark 8.6]).

Definition 2.5 (Smooth functions on embedded submanifolds). Let $r , k , n \in \mathbb { N } , m \in \mathbb { N } _ { 0 }$ , let $S \subseteq \mathbb { R } ^ { k }$ be a set, and let $M \subseteq \mathbb { R } ^ { n }$ be a m-dimensional C<sup>r</sup>-submanifold of $\mathbb { R } ^ { n }$ (cf. Definition 2.1). Then we denote by $\mathbf { C } ^ { r } ( M , S )$ the set of all functions $f \colon M \to S$ such that for every open $U \subseteq \mathbb { R } ^ { m }$ , every open $V \subseteq \mathbb { R } ^ { n }$ and every $\phi \in C ^ { r } ( U , V \cap M )$ with the property that for all $u \in U$ it holds that rank $\begin{array} { r } { ( \phi ^ { \prime } ( u ) ) = m } \end{array}$ we have that

$$
f \circ \phi \in C ^ { r } ( U , S ) .\tag{2.29}
$$

Lemma 2.6 (Existence and uniqueness of the angular gradient). Let $n \in \mathbb { N } , \mathfrak { r } \in ( 0 , \infty )$ , let $M \subseteq \mathbb { R } ^ { n }$ be a C<sup>1</sup>-submanifold of $\mathbb { R } ^ { n }$ , let $U \ \subseteq \ ( 0 , \mathfrak { r } ) \times M$ be open in $( 0 , \mathfrak { r } ) \times M$ , and let $Z \in \mathbf { C } ^ { 1 } ( U , \mathbb { R } )$ (cf. Definitions 2.2 and 2.5). Then for all $( r , x ) \in U$ there exists a unique $v \in T _ { x } M$ such that for all $\varepsilon \ \in \ ( 0 , \infty ) , \ \gamma \in C ^ { 1 } ( ( - \varepsilon , \varepsilon ) , M ) , \ \sigma \ \in \ C ^ { 1 } ( ( - \varepsilon , \varepsilon ) , \mathbb { R } )$ with $\gamma ( 0 ) \ = \ x , \ \bigcup _ { t \in ( - \varepsilon , \varepsilon ) } \{ ( r , \gamma ( t ) ) \} \ \subseteq \ U$ , and $\forall t \in ( - \varepsilon , \varepsilon ) \colon \sigma ( t ) = Z ( r , \gamma ( t ) )$ it holds that

$$
\langle v , \gamma ^ { \prime } ( 0 ) \rangle = \sigma ^ { \prime } ( 0 )\tag{2.30}
$$

(cf. Definition 2.3).

Proof of Lemma 2.6. We first prove that for all $( r , x ) \in U$ it holds that there exists $v \in T _ { x } M$ such that for all $\varepsilon \in ( 0 , \infty ) , \gamma \in C ^ { 1 } ( ( - \varepsilon , \varepsilon ) , M ) , \sigma \in C ^ { 1 } ( ( - \varepsilon , \varepsilon ) , \mathbb { R } )$ with $\gamma ( 0 ) = x , \{ r \} \times \gamma ( ( - \varepsilon , \varepsilon ) ) \subseteq U$ , and $\forall t \in ( - \varepsilon , \varepsilon ) \colon \sigma ( t ) = Z ( r , \gamma ( t ) )$ that

$$
\langle v , \gamma ^ { \prime } ( 0 ) \rangle = \sigma ^ { \prime } ( 0 ) .\tag{2.31}
$$

To prove (2.31), let $( r , x ) \in U$ . In the following let $m \in  { \mathbb { N } } _ { 0 }$ satisfy that M is an m-dimensional $C ^ { 1 } \mathbf { \Phi } .$ submanifold of $\mathbb { R } ^ { n }$ , and let $Q \colon \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ satisfy for all $y = ( y _ { 1 } , \ldots , y _ { n } ) \in \mathbb { R } ^ { n }$ that

$$
Q ( y ) = ( y _ { 1 } , \dots , y _ { m } ) .\tag{2.32}
$$

Observe that Lemma 2.4 establishes that there exist open $O , W \subseteq \mathbb { R } ^ { n }$ and a $C ^ { 1 }$ -difeomorphism $\Phi \colon O \to$ W which satisfy that

$$
x \in O , \qquad \Phi ( x ) = 0 , \qquad \mathrm { a n d } \qquad \Phi ( M \cap O ) = W \cap ( \mathbb { R } ^ { m } \times \{ 0 \} ^ { n - m } ) .\tag{2.33}
$$

In the following let

$$
D = \{ a \in \mathbb { R } ^ { m } \colon ( a , 0 ) \in W , ~ ( r , \Phi ^ { - 1 } ( a , 0 ) ) \in U \}\tag{2.34}
$$

and let $\phi \colon D \to M$ and $f \colon D $ R satisfy for all $a \in D$ that

$$
\phi ( a ) = \Phi ^ { - 1 } ( a , 0 ) \qquad \mathrm { a n d } \qquad f ( a ) = Z ( r , \phi ( a ) ) .\tag{2.35}
$$

Note that (2.32), (2.33), (2.34), (2.35), the fact that U is open in $( 0 , \mathfrak { r } ) \times M$ , and the chain rule ensure that D is open, that $0 \in D$ , that $\phi \in C ^ { 1 } ( D , M )$ , that $\phi ( D )$ is open in M, that

$$
\phi ( 0 ) = x \qquad \mathrm { a n d } \qquad \{ r \} \times \phi ( D ) \subseteq U ,\tag{2.36}
$$

and that for all $a \in D , y \in \phi ( D )$ it holds that

$$
Q ( \Phi ( \phi ( a ) ) ) = a \qquad \mathrm { a n d } \qquad \phi ( Q ( \Phi ( y ) ) ) = y .\tag{2.37}
$$

This, (2.32), (2.33), (2.35), the fact that D is open and $\phi ( D )$ is open in M, and the chain rule imply that for all $a \in D$ it holds that

$$
\phi ^ { \prime } ( a ) \colon \mathbb { R } ^ { m } \to T _ { \phi ( a ) } M { \mathrm { ~ i s ~ a ~ l i n e a r ~ i s o m o r p h i s m } } .\tag{2.38}
$$

In the following let

$$
A = \{ ( s , a ) \in ( 0 , \mathfrak { r } ) \times D \colon ( s , \phi ( a ) ) \in U \}\tag{2.39}
$$

and let $\Psi \colon A \to U$ satisfy for all $( s , a ) \in A$ that

$$
\Psi ( s , a ) = ( s , \phi ( a ) ) .\tag{2.40}
$$

Observe that (2.36), (2.38), (2.39), (2.40), and the fact that U is open in $( 0 , \mathfrak { r } ) \times M$ show that A is open, that $\{ r \} \times D \subseteq A$ , that $\Psi \in C ^ { 1 } ( A , U )$ , and that for all $( s , a ) \in A$ it holds that

$$
\operatorname { r a n k } ( \Psi ^ { \prime } ( s , a ) ) = m + 1 .\tag{2.41}
$$

Combining this, (2.35), (2.40), the fact that $Z \in \mathbf { C } ^ { 1 } ( U , \mathbb { R } )$ , and the chain rule proves that $Z \circ \Psi \in C ^ { 1 } ( A , \mathbb { R } )$ and that $f \in C ^ { 1 } ( D _ { \cdot }$ , R). In addition, note that $( 2 . 3 6 ) , ( 2 . 3 8 )$ , the fact that $f \in C ^ { 1 } ( D , \mathbb { R } )$ , the fact that $f ^ { \prime } ( 0 ) \circ ( \phi ^ { \prime } ( 0 ) ) ^ { - 1 }$ is a linear functional on $T _ { x } M$ , and the finite-dimensional Riesz representation theorem demonstrate that there exists a unique $v \in T _ { x } M$ which satisfies for al $a \in \mathbb { R } ^ { m }$ that

$$
\langle v , \phi ^ { \prime } ( 0 ) a \rangle = f ^ { \prime } ( 0 ) a .\tag{2.42}
$$

We claim that for all $\varepsilon \in ( 0 , \infty ) , \gamma \in C ^ { 1 } ( ( - \varepsilon , \varepsilon ) , M ) , \sigma \in C ^ { 1 } ( ( - \varepsilon , \varepsilon ) , \mathbb { R } ) { \mathrm { ~ w i t h ~ } } \gamma ( 0 ) = x , \{ r \} \times \gamma ( ( - \varepsilon , \varepsilon ) ) \subseteq C ^ { 1 } ( ( - \varepsilon , \varepsilon ) , \mathbb { R } ) { \mathrm { ~ w i t h ~ } } \gamma ( 0 ) = x .$ $U ,$ and $\forall t \in ( - \varepsilon , \varepsilon ) \colon \sigma ( t ) = Z ( r , \gamma ( t ) )$ it holds that

$$
\langle v , \gamma ^ { \prime } ( 0 ) \rangle = \sigma ^ { \prime } ( 0 ) .\tag{2.43}
$$

To prove (2.43), let $\varepsilon \in ( 0 , \infty ) , \gamma \in C ^ { 1 } ( ( - \varepsilon , \varepsilon ) , M ) , \sigma \in C ^ { 1 } ( ( - \varepsilon , \varepsilon ) , \mathbb { R } )$ satisfy for all $t \in ( - \varepsilon , \varepsilon )$ that

$$
\gamma ( 0 ) = x , \qquad ( r , \gamma ( t ) ) \in U , \qquad { \mathrm { a n d } } \qquad \sigma ( t ) = Z ( r , \gamma ( t ) ) .\tag{2.44}
$$

Observe that (2.44), (2.36), the fact that $\phi ( D )$ is open in M, and the fact that $\gamma$ is continuous establish that there exists $\eta \in ( 0 , \varepsilon )$ which satisfies that

$$
\gamma ( ( - \eta , \eta ) ) \subseteq \phi ( D ) .\tag{2.45}
$$

In the following let $\alpha \colon ( - \eta , \eta )  D$ satisfy for all $t \in ( - \eta , \eta )$ that

$$
\alpha ( t ) = Q ( \Phi ( \gamma ( t ) ) ) .\tag{2.46}
$$

Note that (2.44), (2.32), (2.33), (2.35), (2.36), (2.37), (2.45), (2.46), and the chain rule ensure that α $\in C ^ { 1 } ( ( - \eta , \eta ) , D )$ , that $\alpha ( 0 ) = 0$ , and that for all $t \in ( - \eta , \eta )$ it holds that

$$
\gamma ( t ) = \phi ( \alpha ( t ) ) \qquad \mathrm { a n d } \qquad \sigma ( t ) = f ( \alpha ( t ) ) .\tag{2.47}
$$

This and the chain rule imply that

$$
\gamma ^ { \prime } ( 0 ) = \phi ^ { \prime } ( 0 ) \alpha ^ { \prime } ( 0 ) \qquad \mathrm { a n d } \qquad \sigma ^ { \prime } ( 0 ) = f ^ { \prime } ( 0 ) \alpha ^ { \prime } ( 0 ) .\tag{2.48}
$$

Combining this, (2.42), and the fact that the choices of $\varepsilon , \gamma$ , and σ were arbitrary proves (2.43). Observe that (2.42), (2.43), and the fact that the choice of $( r , x )$ was arbitrary establish (2.31).

Next we prove that for all $( \mathbf { r } , \mathbf { x } ) \ \in \ U$ , w, $\textbf { w } \in \ T _ { \mathbf { x } } M$ with the property that for all $\varepsilon \in \mathsf { \Gamma } ( 0 , \infty )$ $\gamma \in C ^ { 1 } ( ( - \varepsilon , \varepsilon ) , M ) , \sigma \in C ^ { 1 } ( ( - \varepsilon , \varepsilon ) , \mathbb { R } )$ with $\gamma ( 0 ) = \mathbf { x } , \ \{ \mathbf { r } \} \times \gamma ( ( - \varepsilon , \varepsilon ) ) \subseteq U$ , and $\forall t \in ( - \varepsilon , \varepsilon ) \colon \sigma ( t ) =$ $Z ( \mathbf { r } , \gamma ( t ) )$ it holds that $\langle w , \gamma ^ { \prime } ( 0 ) \rangle = \langle \mathbf { w } , \gamma ^ { \prime } ( 0 ) \rangle = \sigma ^ { \prime } ( 0 )$ we have that

$$
w = \mathbf { w } .\tag{2.49}
$$

To prove (2.49), let $( \mathbf { r } , \mathbf { x } ) \in U , \ w , \mathbf { w } \in T _ { \mathbf { x } } M$ satisfy for all $\varepsilon \in ( 0 , \infty ) , \gamma \in C ^ { 1 } ( ( - \varepsilon , \varepsilon ) , M )$ $\sigma \in$ $C ^ { 1 } ( ( - \varepsilon , \varepsilon ) , \mathbb { R } )$ with $\gamma ( 0 ) = \mathbf { x } , \ \{ \mathbf { r } \} \times \gamma ( ( - \varepsilon , \varepsilon ) ) \subseteq U$ , and $\forall t \in ( - \varepsilon , \varepsilon ) \colon \sigma ( t ) = Z ( \mathbf { r } , \gamma ( t ) )$ that

$$
\begin{array} { r } { \langle w , \gamma ^ { \prime } ( 0 ) \rangle = \langle \mathbf { w } , \gamma ^ { \prime } ( 0 ) \rangle = \sigma ^ { \prime } ( 0 ) . } \end{array}\tag{2.50}
$$

In the following let $m \in \mathbb { N } _ { 0 }$ satisfy that M is an m-dimensional $C ^ { 1 }$ -submanifold of $\mathbb { R } ^ { n }$ . Note that Lemma 2.4 shows that there exist open $O , W \subseteq \mathbb { R } ^ { n }$ and a C<sup>1</sup>-difeomorphism Φ: $O \to W$ which satisfy that

$$
\mathbf { x } \in O , \qquad \Phi ( \mathbf { x } ) = 0 , \qquad \mathrm { a n d } \qquad \Phi ( M \cap O ) = W \cap ( \mathbb { R } ^ { m } \times \{ 0 \} ^ { n - m } ) .\tag{2.51}
$$

In the following let

$$
A = \{ ( s , a ) \in ( 0 , \mathfrak { r } ) \times \mathbb { R } ^ { m } \colon ( a , 0 ) \in W , \ ( s , \Phi ^ { - 1 } ( a , 0 ) ) \in U \} \quad \mathrm { a n d } \quad D = \{ a \in \mathbb { R } ^ { m } \colon ( \mathbf { r } , a ) \in A \} ,\tag{2.52}
$$

let $\Psi \colon A \to U$ satisfy for all $( s , a ) \in A$ that

$$
\Psi ( s , a ) = ( s , \Phi ^ { - 1 } ( a , 0 ) ) ,\tag{2.53}
$$

and let $\phi \colon D \to M$ and $f \colon D $ R satisfy for all $a \in D$ that

$$
\phi ( a ) = \Phi ^ { - 1 } ( a , 0 ) \qquad \mathrm { a n d } \qquad f ( a ) = Z ( \mathbf { r } , \phi ( a ) ) .\tag{2.54}
$$

Observe that (2.51), (2.52), (2.53), (2.54), the fact that U is open in $( 0 , \mathfrak { r } ) \times M ,$ , and the chain rule demonstrate that A and D are open, that $0 \in \textit { D }$ , that $\Psi \in C ^ { 1 } ( A , U )$ , that $\phi \in C ^ { 1 } ( D , M )$ , that $\phi ^ { \prime } ( 0 ) \colon \mathbb { R } ^ { m } \to T _ { \mathbf { x } } M$ is a linear isomorphism, that

$$
\phi ( 0 ) = \mathbf { x } \qquad { \mathrm { a n d } } \qquad \{ \mathbf { r } \} \times \phi ( D ) \subseteq U ,\tag{2.55}
$$

and that for all $( s , a ) \in A$ it holds that

$$
\operatorname { r a n k } ( \Psi ^ { \prime } ( s , a ) ) = m + 1 .\tag{2.56}
$$

This and the fact that $Z \in \mathbf { C } ^ { 1 } ( U , \mathbb { R } )$ ensure that $Z \circ \Psi \in C ^ { 1 } ( A , \mathbb { R } )$ . Moreover, note that (2.52), (2.53), and (2.54) imply that for all $a \in D$ it holds that

$$
( \mathbf { r } , a ) \in A \qquad { \mathrm { a n d } } \qquad f ( a ) = Z ( \mathbf { r } , \phi ( a ) ) = ( Z \circ \Psi ) ( \mathbf { r } , a ) .\tag{2.57}
$$

Combining this, the fact that $Z \circ \Psi \in C ^ { 1 } ( A , \mathbb { R } )$ , the fact that $( D \ni a \mapsto ( \mathbf { r } , a ) \in A ) \in C ^ { 1 } ( D , A )$ , and the chain rule proves that $f \in C ^ { 1 } ( D , \mathbb { R } )$ . We claim that for all $a \in \mathbb { R } ^ { m }$ it holds that

$$
\langle w , \phi ^ { \prime } ( 0 ) a \rangle = \langle { \bf w } , \phi ^ { \prime } ( 0 ) a \rangle = f ^ { \prime } ( 0 ) a .\tag{2.58}
$$

To prove (2.58), let $a \in \mathbb { R } ^ { m }$ . Observe that the fact that $D$ is open and $0 \in D$ establishes that there exists $\tau \in ( 0 , \infty )$ which satisfies for all $t \in ( - \tau , \tau )$ that

$$
t a \in D .\tag{2.59}
$$

In the following let $\beta \colon ( - \tau , \tau ) \to M$ and $\chi \colon ( - \tau , \tau ) $ R satisfy for all $t \in ( - \tau , \tau )$ that

$$
\beta ( t ) = \phi ( t a ) \qquad \mathrm { a n d } \qquad \chi ( t ) = f ( t a ) .\tag{2.60}
$$

Note that (2.54), (2.55), (2.59), (2.60), the fact that $\phi \in C ^ { 1 } ( D , M )$ and $f \in C ^ { 1 } ( D , \mathbb { R } )$ , and the chain rule show that $\beta \in C ^ { 1 } ( ( - \tau , \tau ) , M )$ , that $\chi \in C ^ { 1 } ( ( - \tau , \tau ) , \mathbb { R } )$ , that $\beta ( 0 ) = \mathbf { x } ,$ that for all $t \in ( - \tau , \tau )$ it holds that

$$
( \mathbf { r } , \beta ( t ) ) \in U , \qquad \chi ( t ) = Z ( \mathbf { r } , \beta ( t ) ) , \qquad \beta ^ { \prime } ( 0 ) = \phi ^ { \prime } ( 0 ) a \qquad \mathrm { a n d } \qquad \chi ^ { \prime } ( 0 ) = f ^ { \prime } ( 0 ) a .\tag{2.61}
$$

This, (2.50), and the fact that $a \in \mathbb { R } ^ { m }$ was arbitrary prove (2.58). Observe that (2.55), (2.58), and the fact that the choices of $( \mathbf { r } , \mathbf { x } ) , \ w ,$ , and w were arbitrary establish (2.49). Note that (2.31) and (2.49) prove (2.30). The proof of Lemma 2.6 is thus complete. □

Definition 2.7 (Gradients with respect to embedded submanifolds). Let $n \in  { \mathbb { N } } , \mathfrak { r } _ { 0 } \in ( 0 , \infty )$ , let $M \subseteq \mathbb { R } ^ { n }$ be a definable C<sup>1</sup>-submanifold of $\mathbb { R } ^ { n }$ , let $U \subseteq ( 0 , \mathfrak { r } _ { 0 } ) \times M$ be definable and open in $( 0 , \mathfrak { r } _ { 0 } ) \times M$ , and let $Z \in \mathbf { C } ^ { 1 } ( U , \mathbb { R } )$ be definable (cf. Definitions 2.2 and 2.5). Then we denote by $\nabla _ { M } Z \colon U  \mathbb { R } ^ { n }$ the function which satisfies for all $( r , x ) \in U$ that

(i) it holds that $( \nabla _ { M } Z ) ( r , x ) \in T _ { x } M$ and

(ii) it holds for all $\varepsilon \in ( 0 , \infty ) , \gamma \in C ^ { 1 } ( ( - \varepsilon , \varepsilon ) , M ) , \sigma \in C ^ { 1 } ( ( - \varepsilon , \varepsilon ) , \mathbb { R } ) { \mathrm { ~ w i t h ~ } } \gamma ( 0 ) = x$

$$
\cup _ { t \in ( - \varepsilon , \varepsilon ) } \{ ( r , \gamma ( t ) ) \} \subseteq U , { \mathrm { a n d ~ } } \forall t \in ( - \varepsilon , \varepsilon ) : \sigma ( t ) = Z ( r , \gamma ( t ) ) { \mathrm { ~ t h a t ~ } }
$$

$$
\langle ( \nabla _ { M } Z ) ( r , x ) , \gamma ^ { \prime } ( 0 ) \rangle = \sigma ^ { \prime } ( 0 )\tag{2.62}
$$

(cf. Definition 2.3).

Lemma 2.8 (Continuity and definability of the partial derivative with respect to the radial variable). Let $n \in \mathbb { N } , \mathfrak { r } \in ( 0 , \infty )$ , let $M \subseteq \mathbb { R } ^ { n }$ be a definable $C ^ { 1 }$ -submanifold $o f \mathbb { R } ^ { n }$ , let $U \subseteq ( 0 , \mathfrak { r } ) \times M$ be definable and open in $( 0 , \mathfrak { r } ) \times M$ , and let $f = ( f ( r , x ) ) _ { ( r , x ) \in U } \in \mathbf { C } ^ { 1 } ( U , \mathbb { R } )$ be definable (cf. Definitions 1.5, 1.6, 2.2, and 2.5). Then

(i) it holds for all $x \in M$ that $\{ s \in ( 0 , \mathfrak { r } ) \colon ( s , x ) \in U \} \ni r \mapsto f ( r , x ) \in \mathbb { R }$ is diferentiable and

(ii) it holds that $U \ni ( r , x ) \mapsto ( \partial _ { r } f ) ( r , x ) \in \mathbb { R }$ is continuous and definable.

Proof of Lemma 2.8. Throughout this proof let $m \in \mathbb { N } _ { 0 }$ satisfy that M is a m-dimensional $C ^ { 1 } \mathfrak { - }$ submanifold of $\mathbb { R } ^ { n }$ (cf. Definition 2.1). Observe that the fact that M is an m-dimensional $C ^ { 1 }$ submanifold of R<sup>n</sup> demonstrates that for every $p \in M$ there exist an open $W _ { p } \subseteq \mathbb { R } ^ { m }$ , a set $O _ { p } \subseteq M$ open in M, and a homeomorphism $\phi _ { p } \in C ^ { 1 } ( W _ { p } , O _ { p } )$ which satisfy that $p \in O _ { p }$ and that for all $u \in W _ { p }$ it holds that rank $\dot { \cdot } ( \phi _ { p } ) ^ { \prime } ( u ) = m$ . In the following for every $p \in M$ let $A _ { p } = \{ ( r , \bar { u } ) \in ( 0 , \mathfrak { r } ) \times W _ { p } \colon ( r , \phi _ { p } ( \bar { u } ) ) \in U \}$ and let $\Phi _ { p } \colon \dot { \boldsymbol { A } } _ { p } \to U \cap ( ( 0 , \mathfrak { r } ) \times O _ { p } )$ and $g _ { p } = ( g _ { p } ( r , u ) ) _ { ( r , u ) \in A _ { p } } \colon A _ { p } \to$ R satisfy for all $( r , u ) \in A _ { p }$ that

$$
\Phi _ { p } ( r , u ) = ( r , \phi _ { p } ( u ) ) \qquad \mathrm { a n d } \qquad g _ { p } ( r , u ) = f ( r , \phi _ { p } ( u ) ) .\tag{2.63}
$$

Note that (2.63), the assumption that U is open in $( 0 , \mathfrak { r } ) \times M$ , the fact that for all $p \in M$ it holds that $A _ { p }$ is open, the fact that for all $p \in M$ it holds that $\Phi _ { p } \in C ^ { 1 } ( A _ { p } , U \cap ( ( 0 , \mathfrak { r } ) \times O _ { p } ) )$ is a homeomorphism, the fact that for all $p \in M , ( r , u ) \in A _ { p }$ it holds that rank $\mathbf { \nabla } : ( \Phi _ { p } ) ^ { \prime } ( r , u ) = m + 1$ , and the fact that $f \in { \bf C } ^ { 1 } ( U , \mathbb { R } )$ ensure that for all $p \in M$ it holds that $g _ { p } \in C ^ { 1 } ( A _ { p } , \mathbb { R } )$ . Combining this, (2.63), and the fact that for all $p \in M$ it holds that $p \in O _ { p }$ establishes item (i). Observe that (2.63), item (i), and the fact that for all $p \in M$ it holds that $g _ { p } \in \hat { C } ^ { 1 } ( A _ { p } , \mathbb { R } )$ imply that for all $p \in M , ( r , u ) \in A _ { p }$ it holds that

$$
( \partial _ { r } f ) ( r , \phi _ { p } ( u ) ) = \operatorname* { l i m } _ { t \to 0 } { \frac { f ( r + t , \phi _ { p } ( u ) ) - f ( r , \phi _ { p } ( u ) ) } { t } } = \operatorname* { l i m } _ { t \to 0 } { \frac { g _ { p } ( r + t , u ) - g _ { p } ( r , u ) } { t } } = ( \partial _ { r } g _ { p } ) ( r , u ) .\tag{2.64}
$$

This and the fact that for all $p \in M$ it holds that $\phi _ { p }$ is a homeomorphism show that for all $p \in M$ $( r , y ) \in U \cap ( ( 0 , \mathfrak { r } ) \times O _ { p } )$ it holds that

$$
( \partial _ { r } f ) ( r , y ) = ( \partial _ { r } g _ { p } ) ( r , \phi _ { p } ^ { - 1 } ( y ) ) .\tag{2.65}
$$

Combining this, the fact that for all $p \in M$ it holds that $\partial _ { r } g _ { p }$ is continuous, the fact that for all $p \in M$ it holds that $U \cap ( ( 0 , \mathfrak { r } ) \times O _ { p } ) \ni ( r , y ) \mapsto ( r , \phi _ { p } ^ { - 1 } ( y ) ) \in A _ { p }$ is continuous, and the fact that $( U \cap ( ( 0 , { \mathfrak { r } } ) \times O _ { p } ) ) _ { p \in M }$ is an open cover of U proves that $\partial _ { r } f$ is continuous. In the next step we note that item (i) demonstrates that

$$
\begin{array}{c} \begin{array} { r l } & { \left\{ ( a , r , y ) \in \mathbb { R } \times U \colon a = ( \partial _ { r } f ) ( r , y ) \right\} } \\ & { = \Bigg \{ ( a , r , y ) \in \mathbb { R } \times U \colon \Bigg [ \forall \varepsilon \in ( 0 , \infty ) \exists \delta \in ( 0 , \infty ) \colon \big ( \forall s \in ( 0 , \mathfrak { r } ) , ( s , y ) \in U , 0 < | s - r | < \delta \colon } \\ & { \qquad | f ( s , y ) - f ( r , y ) - a ( s - r ) | < \varepsilon | s - r | \big ) } \end{array} \Bigg ] \Bigg \} .  \end{array}\tag{2.66}
$$

This, the fact that $f$ is definable, the fact that U is definable, and [9, Theorem 1.13] ensure that the graph of $\partial _ { r } f$ is definable. Combining this and the fact that $\partial _ { r } f$ is continuous establishes item (ii). The proof of Lemma 2.8 is thus complete. □

Lemma 2.9 (Continuity and definability of the gradient with respect to the angular variable). Let $n \in \mathbb { N } , \mathfrak { r } \in ( 0 , \infty )$ , let $M \subseteq \mathbb { R } ^ { n }$ be a definable $C ^ { 1 }$ -submanifold $o f \mathbb { R } ^ { n }$ , let $U \subseteq ( 0 , \mathfrak { r } ) \times M$ be definable and open in $( 0 , \mathfrak { r } ) \times M$ , and let $Z \in \mathbf { C } ^ { 1 } ( U , \mathbb { R } )$ be definable (cf. Definitions 2.2 and 2.5). Then $\nabla _ { M } Z \colon U  \mathbb { R } ^ { n }$ is continuous and definable (cf. Definition $\it { 2 . 7 ) }$

Proof of Lemma 2.9. Throughout this proof let $m \ \in \ \mathbb { N } _ { 0 }$ satisfy that M is an m-dimensional $C ^ { 1 } .$ submanifold of $\mathbb { R } ^ { n }$ . In our proof we distinguish between the case $M = \emptyset$ , the case $( M \neq \emptyset \ \mathrm { a n d } \ m = 0 )$ and the case $( M \neq \emptyset$ and $m \geq 1 \}$ ). We first prove that $\nabla _ { M } Z$ is continuous and definable in the case that

$$
M = \varnothing .\tag{2.67}
$$

Observe that (2.67) and the fact that $U \subseteq ( 0 , \mathfrak { r } ) \times M$ imply that $U = \emptyset$ . Hence, we obtain that $\nabla _ { M } Z$ is continuous and definable in the case $M = \emptyset$ . We now prove that $\nabla _ { M } Z$ is continuous and definable in the case that

$$
M \neq \emptyset \qquad { \mathrm { a n d } } \qquad m = 0 .\tag{2.68}
$$

Note that (2.68), the fact that M is an m-dimensional C<sup>1</sup>-submanifold of $\mathbb { R } ^ { n }$ , and the fact that for all $( r , x ) \in U$ it holds that $( \nabla _ { M } Z ) ( r , x ) \in T _ { x } M$ show that for all $( r , x ) \in U$ it holds that $T _ { x } M = \{ 0 \}$ and $( \nabla _ { M } Z ) ( r , x ) = 0$ . This and the fact that U is definable prove that $\nabla _ { M } Z$ is continuous and definable in the case $M \neq \emptyset$ and $m = 0$ . We now prove that $\nabla _ { M } Z$ is continuous and definable in the case that

$$
M \neq \emptyset \qquad { \mathrm { a n d } } \qquad m \geq 1 .\tag{2.69}
$$

We claim that for all $x \in M$ there exist a definable set $V \subseteq M$ open in M with $x \in V$ and a coordinate projection $\pi \colon \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ such that $\pi ( V )$ is definable and open and that

$$
\pi | _ { V } : V \to \pi ( V ) { \mathrm { ~ i s ~ a ~ d e f i n a b l e ~ } } C ^ { 1 } { \mathrm { - d i f f e o m o r p h i s m . } }\tag{2.70}
$$

To prove (2.70), let $x \in M$ . Observe that Lemma 2.4 and the inverse function theorem demonstrate that there exist a set $W \subseteq M$ open in M with $x \in W$ and a coordinate projection $\pi \colon \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ which satisfy that π| is a $C ^ { 1 }$ -difeomorphism onto an open subset of $\mathbb { R } ^ { m }$ . Note that the fact that W is open in M and $x \in W$ ensures that there exists $\delta \in ( 0 , \infty )$ which satisfies that

$$
\left\{ y \in M \colon | y - x | < \delta \right\} \subseteq W .\tag{2.71}
$$

In the following let

$$
V = \{ y \in M \colon | y - x | < \delta \} .\tag{2.72}
$$

Observe that (2.71), (2.72), and the fact that M is definable establish that V is open in M, that $\pi ( V )$ is definable and open, and that π| : $V \to \pi ( V )$ is a $C ^ { 1 }$ -difeomorphism. Furthermore, note that the fact that $V$ is definable and the fact that $\pi | _ { V } \colon V \to \pi ( V )$ is bijective imply that

$$
\operatorname { g r a p h } ( ( \pi | _ { V } ) ^ { - 1 } ) = \{ ( \pi ( y ) , y ) \colon y \in V \}\tag{2.73}
$$

is definable and that $( \pi | _ { V } ) ^ { - 1 }$ is definable. Next, observe that the fact that V is definable and open in M and $x \in V$ , the fact that π is a coordinate projection and $\pi ( V )$ is definable and open, the fact that $( \pi | _ { V } ) ^ { - 1 } \colon \pi ( V ) \to V$ is a definable $C ^ { 1 }$ -difeomorphism, and the fact that $x \in M$ was arbitrary prove (2.70).

Note that (2.70), the fact that M is definable, and the existence of a finite definable $C ^ { 1 }$ -atlas for every definable $C ^ { 1 } { \mathrm { - s u b m a n i f o l d } }$ (see [15, Section 5.2]) show that there exist $N \in \mathbb { N }$ , definable sets $\mathcal { O } _ { j } \subseteq M$ open in $M ,$ definable open $\mathcal { D } _ { j } \subseteq \mathbb { R } ^ { m }$ , and definable $C ^ { 1 }$ -difeomorphisms $\phi _ { j } \colon \mathscr { D } _ { j }  \mathscr { O } _ { j } , j \in \{ 1 , \ldots , N \}$ which satisfy that

$$
\begin{array} { r } { M = \bigcup _ { j = 1 } ^ { N } \mathcal { O } _ { j } } \end{array}\tag{2.74}
$$

and which satisfy for all $j \in \{ 1 , 2 , \dots , N \} , u \in \mathcal { D } _ { j }$ that $( \phi _ { j } ) ^ { \prime } ( u ) \colon \mathbb { R } ^ { m } \to T _ { \phi _ { j } ( u ) } M$ is a linear isomorphism. In the following for every $j \in \{ 1 , 2 , \dots , N \}$ let

$$
\mathcal { U } _ { j } = \{ ( r , u ) \in ( 0 , \mathfrak { r } ) \times \mathcal { D } _ { j } \colon ( r , \phi _ { j } ( u ) ) \in U \}\tag{2.75}
$$

and let $z _ { j } : \mathcal { U } _ { j } $ R satisfy for all $( r , u ) \in \mathcal { U } _ { j }$ that $z _ { j } ( r , u ) = Z ( r , \phi _ { j } ( u ) )$ . Observe that the fact that U is definable and open in $( 0 , \mathfrak { r } ) \times M$ , the fact that for all $j \in \{ 1 , 2 , \dots , N \}$ it holds that $\phi _ { j }$ is definable and continuously diferentiable, and the fact that $Z \in \mathbf { C } ^ { 1 } ( U , \mathbb { R } )$ is definable demonstrate that for all $j \in \{ 1 , \ldots , N \}$ it holds that $\mathcal { U } _ { j }$ is definable and open and that $z _ { j } \in C ^ { 1 } ( \mathcal { U } _ { j } , \mathbb { R } )$ is definable. In the following for all $j \in \{ 1 , \ldots , N \}$ let $A _ { j } \colon \mathcal { D } _ { j } \to \mathbb { R } ^ { n \times m }$ and $G _ { j } \colon \mathcal { D } _ { j }  \mathring { \mathbb { R } } ^ { m \times m }$ satisfy for all $u \in \mathcal { D } _ { \mathcal { T } }$ <sub>j</sub> that

$$
A _ { j } ( u ) = ( \phi _ { j } ) ^ { \prime } ( u ) \qquad \mathrm { a n d } \qquad G _ { j } ( u ) = ( A _ { j } ( u ) ) ^ { \mathsf { T } } A _ { j } ( u ) .\tag{2.76}
$$

Note that (2.76) and the fact that for all $j \in \{ 1 , 2 , \dots , N \} , u \in \mathcal { D } _ { j }$ it holds that $( \phi _ { j } ) ^ { \prime } ( u ) \colon { \mathbb { R } } ^ { m } \to T _ { \phi _ { i } ( u ) } M$ is a linear isomorphism ensure that for all $j \in \{ 1 , 2 , \dots , N \}$ , u $\in \mathcal { D } _ { j }$ it holds that $G _ { j } ( u )$ is positive definite and invertible. In the following for all $j \in \{ 1 , 2 , \dots , N \}$ let $g _ { j } \colon \mathcal { U } _ { j } \to \mathbb { R } ^ { n }$ satisfy for all $( r , u ) \in \mathcal { U } _ { j }$ that

$$
\begin{array} { r } { g _ { j } ( r , u ) = A _ { j } ( u ) ( G _ { j } ( u ) ) ^ { - 1 } ( \nabla _ { u } z _ { j } ) ( r , u ) . } \end{array}\tag{2.77}
$$

Observe that the fact that for all $j \in \{ 1 , 2 , \dots , N \}$ it holds that $\phi _ { j }$ and $z _ { j }$ are continuously diferentiable and definable, item (ii) in Lemma 2.8, and (2.77) establish that for all $j \in \{ 1 , 2 , \dots , N \}$ it holds that $g _ { j }$ is continuous and definable and that for all $( r , u ) \in \mathcal { U } _ { j }$ it holds that $g _ { j } ( r , u ) \in T _ { \phi _ { j } ( u ) } M$ . We claim that for all $j \in \{ 1 , \dots , N \} , ( r , u ) \in \mathcal { U } _ { j }$ it holds that

$$
\begin{array} { r } { ( \nabla _ { M } Z ) ( r , \phi _ { j } ( u ) ) = g _ { j } ( r , u ) . } \end{array}\tag{2.78}
$$

To prove (2.78), let $j \in \{ 1 , 2 , \ldots , N \} , ( r , u ) \in \mathcal { U } _ { j } , \varepsilon \in ( 0 , \infty ) , \gamma \in C ^ { 1 } ( ( - \varepsilon , \varepsilon ) , M )$ , and $\sigma \in C ^ { 1 } ( ( - \varepsilon , \varepsilon ) , \mathbb { R } )$ satisfy for all $t \in ( - \varepsilon , \varepsilon )$ that

$$
\gamma ( 0 ) = \phi _ { j } ( u ) , \qquad ( r , \gamma ( t ) ) \in U , \qquad \mathrm { a n d } \qquad \sigma ( t ) = Z ( r , \gamma ( t ) ) .\tag{2.79}
$$

Note that (2.79), the fact that γ is continuous, the fact that $\phi _ { j } ( u ) \in { \mathcal { O } } _ { j }$ , and the fact that ${ \mathcal { O } } _ { j }$ is open in M imply that there exists $\eta \in ( 0 , \varepsilon )$ which satisfies that

$$
\gamma ( ( - \eta , \eta ) ) \subseteq { \mathcal { O } } _ { j } .\tag{2.80}
$$

Observe that (2.79), (2.80), the fact that $\gamma \in C ^ { 1 } ( ( - \varepsilon , \varepsilon ) , M )$ , the fact that $\phi _ { j } : { \mathcal { D } } _ { j }  { \mathcal { O } } _ { j }$ is a $C ^ { 1 } \mathfrak { - }$ difeomorphism, the fact that for all $( s , v ) \in \mathcal { U } _ { j }$ it holds that $z _ { j } ( s , v ) = Z ( s , \phi _ { j } ( v ) )$ , and the chain rule prove that there exists $\alpha \in C ^ { 1 } ( ( - \eta , \eta ) , D _ { j } )$ which satisfies $\alpha ( 0 ) = u$ and satisfies for all $t \in ( - \eta , \eta )$ that

$$
\gamma ( t ) = \phi _ { j } ( \alpha ( t ) ) \qquad \mathrm { a n d } \qquad \sigma ( t ) = z _ { j } ( r , \alpha ( t ) ) .\tag{2.81}
$$

Note that (2.76), (2.77), (2.81), and the chain rule show that

$$
\begin{array} { r l } & { \langle g _ { j } ( r , u ) , \gamma ^ { \prime } ( 0 ) \rangle = \langle A _ { j } ( u ) ( G _ { j } ( u ) ) ^ { - 1 } ( \nabla _ { u } z _ { j } ) ( r , u ) , A _ { j } ( u ) \alpha ^ { \prime } ( 0 ) \rangle } \\ & { \qquad = \langle ( \nabla _ { u } z _ { j } ) ( r , u ) , \alpha ^ { \prime } ( 0 ) \rangle = \sigma ^ { \prime } ( 0 ) . } \end{array}\tag{2.82}
$$

Combining this, the fact that $g _ { j } ( r , u ) \in T _ { \phi _ { j } ( u ) } M$ , the fact that the choices of $j , ( r , u ) , \varepsilon , \gamma$ , and σ were arbitrary, and Lemma 2.6 establishes $( 2 . 7 8 { \bar { ) } }$

Observe that (2.74), (2.78), and the fact that for all $j \in \{ 1 , 2 , \dots , N \}$ it holds that $g _ { j }$ and $\phi _ { i } ^ { - 1 }$ are continuous demonstrate that $\nabla _ { M } Z$ is continuous. Note that (2.74), (2.78), and the fact that for all $j \in \{ 1 , \ldots , N \}$ it holds that $\mathcal { U } _ { j } , \phi _ { j }$ , and $g _ { j }$ are definable ensure that

$$
\begin{array} { r } { \operatorname { g r a p h } ( \nabla _ { M } Z ) = \bigcup _ { j = 1 } ^ { N } \{ ( ( r , \phi _ { j } ( u ) ) , g _ { j } ( r , u ) ) \colon ( r , u ) \in \mathcal { U } _ { j } \} } \end{array}\tag{2.83}
$$

is definable. This and the fact that $\nabla _ { M } Z$ is continuous prove that $\nabla _ { M } Z$ is continuous and definable in the case $M \neq \emptyset$ and $m \geq 1$ . The proof of Lemma 2.9 is thus complete. □

## 2.3 Integrability, smoothness, and definability properties of radial upper bounds

In the following elementary Lemma 2.10 we recall the well-known fact that for every $r _ { 0 } \in ( 0 , \infty )$ and every bounded and definable curve $u \colon ( 0 , r _ { 0 } ) \to \mathbb { R } ^ { n }$ we have that u has finite length near zero. Only for completeness, we include a detailed proof for Lemma 2.10 here.

Lemma 2.10. Let $n , p \in  { \mathbb { N } } , \mathfrak { r } _ { 0 } \in ( 0 , \infty )$ and let $u \colon ( 0 , \mathfrak { r } _ { 0 } ) \to \mathbb { R } ^ { n }$ be bounded and definable. Then there exists $\mathfrak { r } _ { 1 } \in ( 0 , \mathfrak { r } _ { 0 } )$ such that

(i) it holds that $u | _ { ( 0 , \mathfrak { r } _ { 1 } ) } \in C ^ { p } ( ( 0 , \mathfrak { r } _ { 1 } ) , \mathbb { R } ^ { n } )$ ，

(ii) there exists $v \in \mathbb { R } ^ { n }$ such that lim $\begin{array} { r } { \operatorname* { s u p } _ { s  0 } | u ( s ) - v | = 0 } \end{array}$ , and

(iii) it holds that $\begin{array} { r } { \int _ { 0 } ^ { \tau _ { 1 } } \left| u ^ { \prime } ( r ) \right| \mathrm { d } r < \infty } \end{array}$

Proof of Lemma 2.10. Throughout this proof, for every $j \in \{ 1 , \ldots , n \}$ , let $\pi _ { j } \colon \mathbb { R } ^ { n } \to \mathbb { R }$ be the jth coordinate projection. Observe that the assumption that u is definable, the fact that for all $j \in \{ 1 , \ldots , n \}$ it holds that $\pi _ { j }$ is definable, and [9, Theorem 2.1 and Corollary 6.5] imply that for all $j \in \{ 1 , \ldots , n \}$ there exists $\Re _ { j } \in ( 0 , \mathfrak { r } _ { 0 } )$ which satisfies that $( \pi _ { j } \circ u ) | _ { ( 0 , \Re _ { j } ) } \in C ^ { p } ( ( 0 , \Re _ { j } )$ , R) and that $( \pi _ { j } \circ u ) | _ { ( 0 , \Re _ { j } ) }$ is strictly increasing, strictly decreasing, or constant. In the following let $\rho \in \mathbb R$ satisfy

$$
\rho = \operatorname* { m i n } \{ \mathfrak { R } _ { 1 } , \mathfrak { R } _ { 2 } , . ~ . ~ . , \mathfrak { R } _ { n } \} .\tag{2.84}
$$

Note that (2.84) shows that for all $j \in \{ 1 , \ldots , n \}$ it holds that $( \pi _ { j } \circ u ) | _ { ( 0 , \rho ) } \in C ^ { p } ( ( 0 , \rho ) , \mathbb { R } )$ and that $( \pi _ { j } \circ u ) | _ { ( 0 , \rho ) }$ is strictly increasing, strictly decreasing, or constant. In the following let $\mathfrak { r } _ { 1 } \in ( 0 , \rho )$ . In addition, observe that the fact that for all $j \in \{ 1 , \ldots , n \}$ it holds that $( \pi _ { j } \circ u ) | _ { ( 0 , \mathfrak { r } _ { 1 } ) } \in C ^ { p } ( ( 0 , \mathfrak { r } _ { 1 } ) , \mathbb { R } )$ establishes that

$$
u | _ { ( 0 , \mathfrak { r } _ { 1 } ) } \in C ^ { p } ( ( 0 , \mathfrak { r } _ { 1 } ) , \mathbb { R } ^ { n } ) .\tag{2.85}
$$

This proves item (i). Note that the fact that u is bounded and the fact that for all $j \in \{ 1 , \ldots , n \}$ it holds that $( \pi _ { j } \circ u ) | _ { ( 0 , \mathfrak { r } _ { 1 } ) }$ is strictly increasing, strictly decreasing, or constant demonstrate that for all $j \in \{ 1 , \ldots , n \}$ there exists $v _ { j } \in \mathbb { R }$ which satisfies that

$$
\operatorname* { l i m } _ { r \to 0 } ( \pi _ { j } \circ u ) ( r ) = v _ { j } .\tag{2.86}
$$

In the following let $v = ( v _ { 1 } , \ldots , v _ { n } ) \in \mathbb { R } ^ { n }$ . Observe that (2.86) ensures that

$$
\operatorname* { l i m } _ { r \to 0 } u ( r ) = v .\tag{2.87}
$$

This establishes item (ii). Note that the fact that $p \in \mathbb { N } .$ , the fact that for all $j \in \{ 1 , \ldots , n \}$ it holds that $( \pi _ { j } \circ u ) | _ { ( 0 , \mathfrak { r } _ { 1 } ) } \in C ^ { 1 } ( ( 0 , \mathfrak { r } _ { 1 } ) , \mathbb { R } )$ and that $( \pi _ { j } \circ u ) | _ { ( 0 , \mathfrak { r } _ { 1 } ) }$ is strictly increasing, strictly decreasing, or constant, and the fundamental theorem of calculus imply that for all $j \in \{ 1 , \ldots , n \} , a , b \in ( 0 , \mathfrak { r } _ { 1 } )$ with $a < b$ it holds that

$$
\int _ { a } ^ { b } | ( \pi _ { j } \circ u ) ^ { \prime } ( r ) | \mathrm { d } r = | ( \pi _ { j } \circ u ) ( b ) - ( \pi _ { j } \circ u ) ( a ) | .\tag{2.88}
$$

Combining the fact that for all $j \in \{ 1 , \ldots , n \}$ it holds that $\pi _ { j } \circ u$ is bounded on $( 0 , \mathfrak { r } _ { 1 } )$ and the triangle inequality hence shows that for all $j \in \{ 1 , \ldots , n \} , a , b \in ( 0 , \mathfrak { r } _ { 1 } )$ with $a < b$ it holds that

$$
\int _ { a } ^ { b } | ( \pi _ { j } \circ u ) ^ { \prime } ( r ) | \mathrm { d } r \leq 2 \operatorname* { s u p } _ { s \in ( 0 , \mathfrak { r } _ { 1 } ) } | ( \pi _ { j } \circ u ) ( s ) | < \infty .\tag{2.89}
$$

This and the monotone convergence theorem prove that for all $j \in \{ 1 , \ldots , n \}$ it holds that

$$
\int _ { 0 } ^ { \tau _ { 1 } } \big | ( \pi _ { j } \circ u ) ^ { \prime } ( r ) \big | \mathrm { d } r < \infty .\tag{2.90}
$$

Combining this and the fact that for all $r \in ( 0 , \mathfrak { r } _ { 1 } )$ it holds that $\begin{array} { r } { | u ^ { \prime } ( r ) | \leq \sum _ { j = 1 } ^ { n } | ( \pi _ { j } \circ u ) ^ { \prime } ( r ) | } \end{array}$ demonstrates that

$$
\int _ { 0 } ^ { \tau _ { 1 } } | u ^ { \prime } ( r ) | \mathrm { d } r \leq \sum _ { j = 1 } ^ { n } \int _ { 0 } ^ { \tau _ { 1 } } | ( \pi _ { j } \circ u ) ^ { \prime } ( r ) | \mathrm { d } r < \infty .\tag{2.91}
$$

This establishes item (iii). The proof of Lemma 2.10 is thus complete.

Lemma 2.11. Let $n \in  { \mathbb { N } } , \ \mathfrak { r } _ { 0 } \in \left( 0 , \infty \right)$ , let $M \subseteq \mathbb { R } ^ { n }$ be a bounded and definable $C ^ { 1 }$ -submanifold of $\mathbb { R } ^ { n }$ , let $U \subseteq ( 0 , \mathfrak { r } _ { 0 } ) \times M$ be $\mathit { d e f i n a b l e }$ , assume that U is open in $( 0 , \mathfrak { r } _ { 0 } ) \times M$ , let $\Omega \subseteq U$ be $\begin{array} { r } { d e \hbar n a b l e , } \end{array}$ $l e t ~ Z = \big ( Z ( r , x ) \big ) _ { ( r , x ) \in U } \in \mathbf { C } ^ { 1 } ( U , \mathbb { R } )$ be definable, assume $\mathrm { s u p } _ { v \in \Omega } | Z ( v ) | < \infty$ , for every $r \in ( 0 , \mathfrak { r } _ { 0 } )$ let $\mathcal { F } _ { r } = \{ x \in M : ( r , x ) \in \Omega \}$ , and let $\eta \colon \left( 0 , \mathfrak { r } _ { 0 } \right) \to \left[ 0 , \right.$ ∞] satisfy for all $r \in ( 0 , \mathfrak { r } _ { 0 } )$ that

$$
\eta ( r ) = \operatorname* { s u p } \left( \{ 0 \} \cup \left( \bigcup _ { x \in \mathcal { F } _ { r } } \left\{ \frac { \lvert ( \partial _ { r } Z ) ( r , x ) \rvert } { 1 + \lvert ( \nabla _ { M } Z ) ( r , x ) \rvert } \right\} \right) \right)\tag{2.92}
$$

(cf. Definitions 2.2, 2.5, and 2.7). Then there exists $\mathfrak { r } _ { 1 } \in ( 0 , \mathfrak { r } _ { 0 } )$ such that

(i) it holds that $\eta ( ( 0 , \mathfrak { r } _ { 1 } ) ) \subseteq \mathbb { R }$ and

(ii) it holds that $( 0 , \mathfrak { r } _ { 1 } ) \ni r \mapsto \eta ( r ) \in \mathbb { R }$ is continuously diferentiable, integrable, and definable.

Proof of Lemma 2.11. Throughout this proof let $q \colon \Omega \to [ 0 , \infty )$ satisfy for all $( r , x ) \in \Omega$ that

$$
q ( r , x ) = \frac { | ( \partial _ { r } Z ) ( r , x ) | } { 1 + | ( \nabla _ { M } Z ) ( r , x ) | }\tag{2.93}
$$

and let $I = \{ r \in ( 0 , \mathfrak { r } _ { 0 } ) \colon \eta ( r ) = + \infty \}$ . Observe that the fact that $Z \in \mathbf { C } ^ { 1 } ( U , \mathbb { R } )$ is definable, Lemma 2.8, and Lemma 2.9 ensure that $\partial _ { r } Z \colon U $ R and $\nabla _ { M } Z \colon U  \mathbb { R } ^ { n }$ are definable. Combining this, (2.93), the fact that Ω is definable, and the fact that for all $( r , x ) \in \Omega$ it holds that $1 + | ( \nabla _ { M } Z ) ( r , x ) | > 0$ implies that $q \colon \Omega \to [ 0 , \infty )$ is definable. This and the fact that Ω is definable show that the set

$$
H = \left\{ ( r , \lambda , x ) \in ( 0 , \mathfrak { r } _ { 0 } ) \times [ 0 , \infty ) \times M \colon \left[ ( ( r , x ) \in \Omega ) \wedge ( q ( r , x ) > \lambda ) \right] \right\}\tag{2.94}
$$

is definable. Hence, we obtain that the set

$$
E = \left\{ ( r , \lambda ) \in ( 0 , \mathfrak { r } _ { 0 } ) \times [ 0 , \infty ) \colon \left[ \exists x \in M \colon ( r , \lambda , x ) \in H \right] \right\}\tag{2.95}
$$

is definable. This proves that the set

$$
J = \left\{ r \in ( 0 , \mathfrak { r } _ { 0 } ) \colon \left[ \exists \lambda \in [ 0 , \infty ) \colon ( r , \lambda ) \notin E \right] \right\}\tag{2.96}
$$

is definable. Combining this, (2.92), (2.95), (2.94), and the fact that $I = \{ r \in ( 0 , { \mathfrak { r } } _ { 0 } ) \colon \eta ( r ) = + \infty \}$ demonstrates that

$$
\begin{array} { r l } & { I = \left\{ r \in ( 0 , \mathfrak { r } _ { 0 } ) \colon \left[ \forall \lambda \in [ 0 , \infty ) \exists x \in M \colon ( ( r , x ) \in \Omega ) \wedge ( q ( r , x ) > \lambda ) \right] \right\} } \\ & { \ = \left\{ r \in ( 0 , \mathfrak { r } _ { 0 } ) \colon \left[ \forall \lambda \in [ 0 , \infty ) \exists x \in M \colon ( r , \lambda , x ) \in H \right] \right\} } \\ & { \ = \left\{ r \in ( 0 , \mathfrak { r } _ { 0 } ) \colon \left[ \forall \lambda \in [ 0 , \infty ) \colon ( r , \lambda ) \in E \right] \right\} = ( 0 , \mathfrak { r } _ { 0 } ) \backslash J } \end{array}\tag{2.97}
$$

is definable. In the following let

$$
\begin{array} { r l } & { G = \big \{ ( r , 0 ) \colon \big [ ( r \in J ) \wedge ( \mathcal { F } _ { r } = \emptyset ) \big ] \big \} } \\ & { \quad \cup \big \{ ( r , y ) \in J \times [ 0 , \infty ) \colon \big [ \big ( \forall x \in \mathcal { F } _ { r } \colon q ( r , x ) \le y \big ) \wedge \big ( \forall \varepsilon \in ( 0 , \infty ) \exists x \in \mathcal { F } _ { r } \colon q ( r , x ) > y - \varepsilon \big ) \big ] \big \} . } \end{array}\tag{2.98}
$$

Note that (2.97), (2.98), and the fact that Ω and q are definable establish that G is definable. Next we combine (2.92), (2.97), and (2.98) to obtain that

$$
G = \mathrm { g r a p h } \big ( \eta | _ { ( 0 , \mathfrak { r } _ { 0 } ) \backslash I } \big ) .\tag{2.99}
$$

This and the fact that $G$ is definable ensure that $\eta | _ { ( 0 , \mathfrak { r } _ { 0 } ) \backslash I }$ is definable. Moreover, observe that (2.97) and the fact that for all definable $A \subseteq \mathbb { R }$ it holds that A is a finite union of singletons and open intervals imply that

$$
\big ( \exists \rho \in ( 0 , \mathfrak { r } _ { 0 } ) \colon I \cap ( 0 , \rho ) = \emptyset \big ) \quad \mathrm { o r } \quad \big ( \exists \rho \in ( 0 , \mathfrak { r } _ { 0 } ) \colon ( 0 , \rho ) \subseteq I \big ) .\tag{2.100}
$$

In the following we claim that for all $\rho \in ( 0 , \mathfrak { r } _ { 0 } )$ it holds that

$$
( 0 , \rho ) \not \subseteq I .\tag{2.101}
$$

We prove (2.101) by contradiction. In the following assume for contradiction that $\left( \exists \rho \in \left( 0 , \mathfrak { r } _ { 0 } \right) \colon \left( 0 , \rho \right) \subseteq \right.$ $I )$ and let ${ \boldsymbol { \rho } } _ { 2 } \in ( 0 , { \mathfrak { r } } _ { 0 } )$ satisfy $( 0 , \rho _ { 2 } ) \subseteq I$ . Note that the assumption that $( 0 , \rho _ { 2 } ) \subseteq I _ { \ O }$ , the fact that $\tilde { I ^ { - } } = \{ r \in ( 0 , \mathfrak { r } _ { 0 } ) \colon \eta ( r ) = + \infty \}$ , and (2.92) show that for every $r \in ( 0 , \rho _ { 2 } )$ there exists $x \in { \mathcal { F } } _ { r }$ such that

$$
\frac { | ( \partial _ { r } Z ) ( r , x ) | } { 1 + | ( \nabla _ { M } Z ) ( r , x ) | } \geq r ^ { - 2 } .\tag{2.102}
$$

In the following let

$$
W = \big \{ ( r , x ) \in \Omega \colon \big [ ( r \in ( 0 , \pmb { \rho } _ { 2 } ) ) \wedge ( q ( r , x ) \geq r ^ { - 2 } ) \big ] \big \} .\tag{2.103}
$$

Observe that (2.103), the fact that $q$ is definable, and the fact that Ω is definable prove that $W$ is definable. In the next step we combine $\left( 2 . 1 0 2 \right)$ , (2.103), and the definition of $q$ to obtain that for all $r \in ( 0 , \rho _ { 2 } )$ it holds that $\mathcal { A } _ { r } = \{ x \in M \colon ( r , x ) \in W \}$ is nonempty. Furthermore, note that the fact that for all $r \in ( 0 , \rho _ { 2 } )$ it holds that A<sub>r</sub> is nonempty and the fact that $W \subseteq ( 0 , \rho _ { 2 } ) \times M$ demonstrate that

$$
\operatorname { p r o j } _ { 1 } ( W ) = ( 0 , \rho _ { 2 } ) .\tag{2.104}
$$

Combining this, the fact that W is definable, and the definable choice theorem [9, Theorem 3.1] establishes that there exists a definable function $\chi \colon ( 0 , \rho _ { 2 } )  M$ which satisfies for every $r \in ( 0 , \rho _ { 2 } )$ that $( r , \chi ( r ) ) \in$ $W .$ . Observe that (2.93), (2.103), and the fact that for all $r \in ( 0 , \rho _ { 2 } )$ it holds that $( r , \chi ( r ) ) \in W$ ensure that for all $r \in ( 0 , \rho _ { 2 } )$ it holds that $\chi ( r ) \in \mathcal { F } _ { r }$ and

$$
\frac { | ( \partial _ { r } Z ) ( r , \chi ( r ) ) | } { 1 + | ( \nabla _ { M } Z ) ( r , \chi ( r ) ) | } \geq r ^ { - 2 } .\tag{2.105}
$$

In the following let $z _ { * } \colon ( 0 , \rho _ { 2 } ) \to  { \mathbb { R } }$ satisfy for all $r \in ( 0 , \rho _ { 2 } )$ that $z _ { * } ( \boldsymbol { r } ) = Z ( r , \chi ( r ) )$ . Note that the fact that M is bounded, the fact that $\chi ( ( 0 , \pmb { \rho } _ { 2 } ) ) \subseteq M$ , the fact that $Z$ is bounded on Ω, and the fact that for all $r \in ( 0 , \rho _ { 2 } )$ it holds that $( r , \chi ( r ) ) \in \Omega$ imply that $\chi$ and $z _ { * }$ are bounded and definable. Lemma 2.10 therefore shows that there exists $\rho _ { 3 } \in ( 0 , \rho _ { 2 } )$ which satisfies that

$$
\chi | _ { ( 0 , \rho _ { 3 } ) } \in C ^ { 1 } ( ( 0 , \rho _ { 3 } ) , M ) , \quad \ z _ { * } | _ { ( 0 , \rho _ { 3 } ) } \in C ^ { 1 } ( ( 0 , \rho _ { 3 } ) , \mathbb { R } ) , \quad \mathrm { a n d } \quad \int _ { 0 } ^ { \rho _ { 3 } } \bigl ( | \chi ^ { \prime } ( r ) | + | ( z _ { * } ) ^ { \prime } ( r ) | \bigr ) \mathrm { d } r < \infty .\tag{2.106}
$$

Observe that (2.106) and the fact that for all $r \in ( 0 , \pmb { \rho } _ { 3 } ) , \varepsilon \in ( 0 , \infty )$ with $( r - \varepsilon , r + \varepsilon ) \subseteq ( 0 , \rho _ { 3 } )$ it holds that $\chi | _ { ( r - \varepsilon , r + \varepsilon ) } \in C ^ { 1 } ( ( r - \varepsilon , r + \varepsilon )$ , M) and $( \chi | _ { ( r - \varepsilon , r + \varepsilon ) } ) ^ { \prime } ( r ) = \chi ^ { \prime } ( r )$ prove that for all $r \in ( 0 , \rho _ { 3 } )$ it holds that

$$
\chi ^ { \prime } ( r ) \in T _ { \chi ( r ) } M .\tag{2.107}
$$

This, the fact that for all $r \in ( 0 , \rho _ { 3 } )$ it holds that $z _ { * } ( \boldsymbol { r } ) = Z ( r , \chi ( r ) )$ , the fact that for all $r \in ( 0 , \rho _ { 3 } )$ it holds that $( r , \chi ( r ) ) \in U$ , the fact that $Z \in \mathbf { C } ^ { 1 } ( U , \mathbb { R } )$ , and the chain rule demonstrate that for all $r \in ( 0 , \rho _ { 3 } )$ it holds that

$$
\begin{array} { r } { ( z _ { * } ) ^ { \prime } ( r ) = ( \partial _ { r } Z ) ( r , \chi ( r ) ) + \langle ( \nabla _ { M } Z ) ( r , \chi ( r ) ) , \chi ^ { \prime } ( r ) \rangle . } \end{array}\tag{2.108}
$$

Combining the triangle inequality and the Cauchy–Schwarz inequality hence establishes that for all $r \in ( 0 , \rho _ { 3 } )$ it holds that

$$
\frac { | ( \partial _ { r } Z ) ( r , \chi ( r ) ) | } { 1 + | ( \nabla _ { M } Z ) ( r , \chi ( r ) ) | } \le | ( z _ { * } ) ^ { \prime } ( r ) | + | \chi ^ { \prime } ( r ) | .\tag{2.109}
$$

Next, note that (2.105) and (2.109) ensure that for all $r \in ( 0 , \rho _ { 3 } )$ it holds that

$$
r ^ { - 2 } \leq | ( z _ { * } ) ^ { \prime } ( r ) | + | \chi ^ { \prime } ( r ) | .\tag{2.110}
$$

This and the fact that $\textstyle \int _ { 0 } ^ { \rho _ { 3 } } r ^ { - 2 } \mathrm { d } r = + \infty$ imply that

$$
\int _ { 0 } ^ { \rho _ { 3 } } \big ( | \chi ^ { \prime } ( r ) | + | ( z _ { * } ) ^ { \prime } ( r ) | \big ) \mathrm { d } r = + \infty .\tag{2.111}
$$

Combining this and (2.106) proves that

$$
\infty = \int _ { 0 } ^ { \rho _ { 3 } } \big ( | \chi ^ { \prime } ( r ) | + | ( z _ { * } ) ^ { \prime } ( r ) | \big ) \mathrm { d } r < \infty .\tag{2.112}
$$

This contradiction proves (2.101).

Observe that (2.100) and (2.101) show that

$$
\exists \rho \in ( 0 , \mathfrak { r } _ { 0 } ) \colon I \cap ( 0 , \rho ) = \emptyset .\tag{2.113}
$$

This and the fact that $I = \{ r \in ( 0 , { \mathfrak { r } } _ { 0 } ) \colon \eta ( r ) = + \infty \}$ demonstrate that there exists $\rho \in ( 0 , \mathfrak { r } _ { 0 } )$ which satisfies for all $r \in ( 0 , \rho )$ that

$$
\eta ( r ) < \infty .\tag{2.114}
$$

Note that (2.114) and the fact that $\eta | _ { ( 0 , \mathfrak { r } _ { 0 } ) \backslash I }$ is definable establish that $\eta | _ { ( 0 , \rho ) } \colon ( 0 , \rho ) \to [ 0 , \infty )$ is definable. Combining this and the fact that the zero set of $\eta | _ { ( 0 , \rho ) }$ is definable ensures that there exists $\rho _ { 1 } \in ( 0 , \rho )$ which satisfies that

$$
\left( \forall r \in ( 0 , \pmb { \rho } _ { 1 } ) \colon \eta ( r ) = 0 \right) \quad \mathrm { o r } \quad \left( \forall r \in ( 0 , \pmb { \rho } _ { 1 } ) \colon \eta ( r ) > 0 \right) .\tag{2.115}
$$

In the following we prove that there exists $\varrho \in ( 0 , \rho _ { 1 } )$ such that

$$
\int _ { 0 } ^ { \varrho } \eta ( r ) \mathrm { d } r < \infty .\tag{2.116}
$$

In our proof of (2.116) we distinguish between the case that for all $r \in ( 0 , \rho _ { 1 } )$ it holds that $\eta ( r ) = 0$ and the case that for all $r \in ( 0 , \rho _ { 1 } )$ it holds that $\eta ( r ) > 0$ . We first prove (2.116) in the case that

$$
\forall r \in ( 0 , \pmb { \rho } _ { 1 } ) \colon \eta ( r ) = 0 .\tag{2.117}
$$

Observe that (2.117) implies that

$$
\int _ { 0 } ^ { \rho _ { 1 } } \eta ( r ) \mathrm { d } r = 0 < \infty .\tag{2.118}
$$

This proves (2.116) in the case that for all $r \in ( 0 , \rho _ { 1 } )$ it holds that $\eta ( r ) = 0$ . We now prove (2.116) in the case that

$$
\forall r \in ( 0 , \pmb { \rho } _ { 1 } ) \colon \eta ( r ) > 0 .\tag{2.119}
$$

Note that (2.92) and (2.119) prove that for every $r \in ( 0 , \rho _ { 1 } )$ there exists $x \in \mathcal { F } _ { r }$ such that

$$
\frac { | ( \partial _ { r } Z ) ( r , x ) | } { 1 + | ( \nabla _ { M } Z ) ( r , x ) | } \ge \frac { 1 } { 2 } \eta ( r ) .\tag{2.120}
$$

In the following let

$$
\begin{array} { r } { K = \big \{ ( r , x ) \in \Omega \colon \big [ ( r \in ( 0 , \rho _ { 1 } ) ) \wedge ( q ( r , x ) \geq \frac { 1 } { 2 } \eta ( r ) ) \big ] \big \} . } \end{array}\tag{2.121}
$$

Observe that the fact that $q , \eta | _ { ( 0 , \rho _ { 1 } ) } , \Omega$ , and $( 0 , \rho _ { 1 } )$ are definable and (2.121) show that K is definable. In addition, note that (2.93), (2.120), and (2.121) demonstrate that for all $r \in ( 0 , \rho _ { 1 } )$ it holds that $\mathcal { Q } _ { r } = \{ x \in M \colon ( r , x ) \in K \}$ is nonempty. This and the fact that $K \subseteq ( 0 , \rho _ { 1 } ) \times M$ establish that

$$
\operatorname { p r o j } _ { 1 } ( K ) = ( 0 , \pmb { \rho } _ { 1 } ) .\tag{2.122}
$$

Combining the fact that K is definable and the definable choice theorem [9, Theorem 3.1] therefore ensures that there exists a definable function $\xi \colon ( 0 , \pmb { \rho } _ { 1 } )  M$ which satisfies for all $r \in ( 0 , \rho _ { 1 } )$ that $( r , \xi ( r ) ) \in K$ . Observe that (2.93), (2.121), and the fact that for all $r \in ( 0 , \rho _ { 1 } )$ it holds that $( r , \xi ( r ) ) \in K$ imply that for all $r \in ( 0 , \rho _ { 1 } )$ it holds that

$$
\xi ( r ) \in { \mathcal { F } } _ { r } \qquad \mathrm { a n d } \qquad \frac { | ( \partial _ { r } Z ) ( r , \xi ( r ) ) | } { 1 + | ( \nabla _ { M } Z ) ( r , \xi ( r ) ) | } \geq \frac { 1 } { 2 } \eta ( r ) .\tag{2.123}
$$

In the following let $\zeta \colon ( 0 , \rho _ { 1 } ) $ R satisfy for all $r \in ( 0 , \rho _ { 1 } )$ that $\zeta ( r ) = Z ( r , \xi ( r ) )$ . Note that the fact that M is bounded, the fact that Z is bounded on Ω, the fact that for all $r \in ( 0 , \rho _ { 1 } )$ it holds that $( r , \xi ( r ) ) \in \Omega$ , and Lemma 2.10 prove that there exists $\pmb { \rho } _ { 4 } \in ( 0 , \pmb { \rho } _ { 1 } )$ which satisfies that

$$
\xi | _ { ( 0 , \rho _ { 4 } ) } \in C ^ { 1 } ( ( 0 , \rho _ { 4 } ) , M ) , \quad \zeta | _ { ( 0 , \rho _ { 4 } ) } \in C ^ { 1 } ( ( 0 , \rho _ { 4 } ) , \mathbb { R } ) , \quad \mathrm { a n d } \quad \int _ { 0 } ^ { \rho _ { 4 } } \bigl ( | \xi ^ { \prime } ( r ) | + | \zeta ^ { \prime } ( r ) | \bigr ) \mathrm { d } r < \infty .\tag{2.124}
$$

Observe that (2.123), (2.124), the fact that $Z \in \mathbf { C } ^ { 1 } ( U , \mathbb { R } )$ , the fact that for all $r \in ( 0 , \rho _ { 4 } )$ it holds that $\zeta ( r ) = Z ( r , \xi ( r ) )$ , the chain rule, the triangle inequality, and the Cauchy–Schwarz inequality show that for all $r \in ( 0 , \rho _ { 4 } )$ it holds that

$$
\frac { | ( \partial _ { r } Z ) ( r , \xi ( r ) ) | } { 1 + | ( \nabla _ { M } Z ) ( r , \xi ( r ) ) | } \le | \zeta ^ { \prime } ( r ) | + | \xi ^ { \prime } ( r ) | .\tag{2.125}
$$

This and (2.123) demonstrate that for all $r \in ( 0 , \rho _ { 4 } )$ it holds that

$$
\eta ( r ) \leq 2 | \zeta ^ { \prime } ( r ) | + 2 | \xi ^ { \prime } ( r ) | .\tag{2.126}
$$

Combining this, (2.124), and the fact that η is nonnegative establishes that

$$
\int _ { 0 } ^ { \pmb \rho _ { 4 } } \eta ( r ) \mathrm { d } r < \infty .\tag{2.127}
$$

This proves (2.116) in the case that for all $r \in ( 0 , \rho _ { 1 } )$ it holds that $\eta ( r ) > 0$

Note that the fact that $\eta | _ { ( 0 , \rho ) } \colon ( 0 , \rho ) \to [ 0 , \infty )$ is definable, the fact that $0 < \varrho < \rho _ { ; }$ , and [9, Corollary 6.5] ensure that there exists ${ \mathfrak { r } } _ { 1 } \in ( 0 , \varrho )$ such that $\eta | _ { ( 0 , \mathfrak { r } _ { 1 } ) } \in C ^ { 1 } ( ( 0 , \mathfrak { r } _ { 1 } ) , \mathbb { R } )$ is locally bounded. This, (2.114), (2.116), and the fact that $\eta | _ { ( 0 , \rho ) }$ is definable prove item (i) and item (ii). The proof of Lemma 2.11 is thus complete. □

In the following result we specialize Lemma 2.11 to the case where $M = ( 0 , H )$ is the 1-dimensional $C ^ { 1 }$ -submanifold (0, H) of R for some strictly positive real number $H \in ( 0 , \infty )$

Lemma 2.12. Let r, $H \in ( 0 , \infty )$ , let $U \subseteq ( 0 , \mathfrak { r } ) \times ( 0 , H )$ be definable and open, let $\Omega \subseteq U$ be $\mathit { d e f i n a b l e }$ let $Z = ( Z ( r , x ) ) _ { ( r , x ) \in U } \in C ^ { 1 } ( U , \mathbb { R } )$ be $\begin{array} { l } { { d e f i n a b l e , } } \end{array}$ assume $\begin{array} { r } { \operatorname* { s u p } _ { v \in \Omega } | Z ( v ) | < \infty , } \end{array}$ and let $\eta \colon ( 0 , \mathfrak { r } )  [ 0 , \infty ]$ satisfy that for all $r \in ( 0 , \mathfrak { r } )$ it holds that

$$
\eta ( r ) = \operatorname* { s u p } \left( \{ 0 \} \cup \left\{ z \in \mathbb { R } \colon \left[ \exists x \in ( 0 , H ) \colon \left( \left[ ( r , x ) \in \Omega \right] \wedge \left[ z = \frac { | ( \partial _ { r } Z ) ( r , x ) | } { 1 + | ( \partial _ { x } Z ) ( r , x ) | } \right] \right) \right] \right\} \right) .\tag{2.128}
$$

Then there exists $\rho \in ( 0 , \mathfrak { r } )$ such that

(i) it holds that $\eta ( ( 0 , \rho ) ) \subseteq \mathbb { R }$ and

(ii) it holds that $( 0 , \rho ) \ni r \mapsto \eta ( r ) \in \mathbb { R }$ is continuously diferentiable, integrable, and definable.

Proof of Lemma 2.12. Observe that the fact that $( 0 , H )$ is a bounded and definable 1-dimensional $C ^ { 1 } \cdot$ submanifold of R, the fact that for all $( r , x ) \in U$ it holds that $( \nabla _ { ( 0 , H ) } Z ) ( r , x ) = ( \partial _ { x } Z ) ( r , x ) , ( 2 . 1 2 8 )$ and Lemma 2.11 (applied with $n \frown 1 , \mathfrak { r } _ { 0 } \frown \mathfrak { r } , M \frown ( 0 , H )$ , and $\mathcal { F } _ { r } \cap \left\{ x \in ( 0 , H ) \colon ( r , x ) \in \Omega \right\}$ for every $r \in ( 0 , \mathfrak { r } )$ in the notation of Lemma 2.11) establish item (i) and item (ii). The proof of Lemma 2.12 is thus complete. □

Lemma 2.13. Let $n \in  { \mathbb { N } } , \ \mathfrak { r } _ { 0 } \in \left( 0 , \infty \right)$ , let $M \subseteq \mathbb { R } ^ { n }$ be a bounded and definable $C ^ { 1 }$ -submanifold of R<sup>n</sup>, let ${ \mathcal { O } } \subseteq ( 0 , \mathfrak { r } _ { 0 } ) \times M$ be $\mathit { d e f i n a b l e }$ and open $i n \ ( 0 , \mathfrak { r } _ { 0 } ) \times M _ { ; }$ , let $h = ( h ( r , x ) ) _ { ( r , x ) \in \mathcal { O } } \in \mathbf { C } ^ { 1 } ( \mathcal { O } , \mathbb { R } )$ $a \in C ^ { 1 } ( ( 0 , \mathfrak { r } _ { 0 } ) , \mathbb { R } )$ be definable, and let $\beta \colon ( 0 , \mathfrak { r } _ { 0 } )  [ 0 , \infty ]$ satisfy for all $r \in ( 0 , \mathfrak { r } _ { 0 } )$ that

$$
\beta ( r ) = \operatorname* { s u p } \Big ( \{ 0 \} \cup \Big \{ z \in \mathbb { R } \colon \Big [ \exists x \in \big \{ y \in M \colon ( r , y ) \in \mathcal { O } \cap h ^ { - 1 } ( \{ a ( r ) \} ) \big \} \colon
$$

$$
z | ( \nabla _ { M } h ) ( r , x ) | = | ( \partial _ { r } h ) ( r , x ) - a ^ { \prime } ( r ) | > 0 \Big ] \Big \} \Big )\tag{2.129}
$$

(cf. Definitions 2.2 and 2.5). Then there exists $\mathfrak { r } _ { 1 } \in ( 0 , \mathfrak { r } _ { 0 } )$ such that

(i) it holds that $\beta ( ( 0 , \mathfrak { r } _ { 1 } ) ) \subseteq \mathbb { R }$ and

(ii) it holds that $( 0 , \mathfrak { r } _ { 1 } ) \ni r \mapsto \beta ( r ) \in \mathbb { R }$ is continuously diferentiable, integrable, and definable.

Proof of Lemma 2.13. Throughout this proof let $G \colon { \mathcal { O } }  \mathbb { R }$ satisfy for all $v \in \mathcal { O }$ that $G ( v ) = | ( \nabla _ { M } h ) ( v ) |$ 2 let

$$
S = \big \{ ( r , x ) \in \mathcal { O } \colon \big [ ( h ( r , x ) = a ( r ) ) \wedge ( G ( r , x ) > 0 ) \big ] \big \} ,\tag{2.130}
$$

let $q \colon S \to [ 0 , \infty )$ satisfy that for all $( r , x ) \in S$ it holds that

$$
q ( r , x ) = \frac { | ( \partial _ { r } h ) ( r , x ) - a ^ { \prime } ( r ) | } { G ( r , x ) } ,\tag{2.131}
$$

and let $I = \{ r \in ( 0 , { \mathfrak { r } } _ { 0 } ) \colon \beta ( r ) = + \infty \}$ . Note that the fact that $h \in { \bf C } ^ { 1 } ( { \mathcal { O } } , \mathbb { R } )$ is definable, the fact that O is definable and open in $( 0 , \mathfrak { r } _ { 0 } ) \times M .$ , Lemma 2.8, and Lemma 2.9 imply that $\partial _ { r } h \colon { \mathcal { O } } \to \mathbb { R }$ and $\nabla _ { M } h \colon { \mathcal { O } } \to \mathbb { R } ^ { n }$ are definable. Combining this and the fact that for all $( r , x ) \in { \mathcal { O } }$ it holds that $G ( r , x ) = | ( \nabla _ { M } h ) ( r , x ) |$ shows that $G \colon { \mathcal { O } }  \mathbb { R }$ is definable. Next we combine the fact that $a \in C ^ { 1 } ( ( 0 , \mathfrak { r } _ { 0 } ) , \mathbb { R } )$ is definable and Lemma 2.8 (applied with n $\curvearrowleft 1 , \texttt { r e r } _ { 0 } , \ M \curvearrowleft\{ 0 \} , \ U \curvearrowleft ( 0 , \mathfrak { r } _ { 0 } ) \times \{ 0 \} , \ f \curvearrowleft( ( 0 , \mathfrak { r } _ { 0 } ) \times \{ 0 \} \ni ( r , 0 ) \mapsto 0 )$ $a ( r ) \in \mathbb { R } )$ in the notation of Lemma 2.8) to obtain that $a ^ { \prime } \colon ( 0 , \mathfrak { r } _ { 0 } )  \mathbb { R }$ is definable. Moreover, observe that the fact that $h , a ,$ and G are definable demonstrates that S is definable. This, the fact that $\partial _ { r } h ,$ $a ^ { \prime } { . }$ , and G are definable, the fact that for all $( r , x ) \in S$ it holds that $G ( r , x ) > 0$ , and (2.131) ensure that $q \colon S \to [ 0 , \infty )$ is well-defined and definable. Combining this and (2.129) proves that I is definable. This and [9, Theorem 2.10] establish that there exists $R \in ( 0 , { \mathfrak { r } } _ { 0 } )$ which satisfies

$$
( 0 , R ) \subseteq I \quad { \mathrm { o r } } \quad ( 0 , R ) \cap I = \varnothing .\tag{2.132}
$$

We claim that

$$
( 0 , R ) \not \subseteq I .\tag{2.133}
$$

To prove this claim, assume for contradiction that

$$
( 0 , R ) \subseteq I .\tag{2.134}
$$

Note that (2.129), (2.134), and the fact that $I = \{ r \in ( 0 , \mathfrak { r } _ { 0 } ) \colon \beta ( r ) = + \infty \}$ imply that for all $r \in ( 0 , R )$ there exists $x \in M$ such that $( r , x ) \in S$ and $q ( r , x ) \geq r ^ { - 2 }$ . Combining this and the definable choice theorem [9, Theorem 3.1] shows that there exists a definable function $\chi \colon ( 0 , R ) \to M$ which satisfies for all $r \in ( 0 , R )$ that

$$
( r , \chi ( r ) ) \in S \quad { \mathrm { a n d } } \quad q ( r , \chi ( r ) ) \geq r ^ { - 2 } .\tag{2.135}
$$

Observe that (2.135), the fact that M is bounded, and Lemma 2.10 demonstrate that there exists $\Re _ { x } \in ( 0 , R )$ which satisfies that

$$
\chi | _ { ( 0 , \Re _ { x } ) } \in C ^ { 1 } ( ( 0 , \Re _ { x } ) , M ) \quad \mathrm { a n d } \quad \int _ { 0 } ^ { \Re _ { x } } | \chi ^ { \prime } ( r ) | \mathrm { d } r < \infty .\tag{2.136}
$$

Note that (2.130) and (2.135) ensure that for all $r \in ( 0 , \Re _ { x } )$ it holds that

$$
h ( r , \chi ( r ) ) = a ( r ) .\tag{2.137}
$$

This, the fact that $h \in { \bf C } ^ { 1 } ( \mathcal { O } , \mathbb { R } ) , a \in C ^ { 1 } ( ( 0 , \mathfrak { r } _ { 0 } ) , \mathbb { R } )$ , and $\chi | _ { ( 0 , \mathfrak { R } _ { x } ) } \in C ^ { 1 } ( ( 0 , \mathfrak { R } _ { x } ) , M )$ , and the chain rule prove that for all $r \in ( 0 , \Re _ { x } )$ it holds that

$$
0 = \frac { \mathrm { d } } { \mathrm { d } r } ( h ( r , \chi ( r ) ) - a ( r ) ) = ( \partial _ { r } h ) ( r , \chi ( r ) ) + \langle ( \nabla _ { M } h ) ( r , \chi ( r ) ) , \chi ^ { \prime } ( r ) \rangle - a ^ { \prime } ( r ) .\tag{2.138}
$$

Hence, we obtain that for all $r \in ( 0 , \Re _ { x } )$ it holds that

$$
\begin{array} { r } { ( \partial _ { r } h ) ( r , \chi ( r ) ) - a ^ { \prime } ( r ) = - \langle ( \nabla _ { M } h ) ( r , \chi ( r ) ) , \chi ^ { \prime } ( r ) \rangle . } \end{array}\tag{2.139}
$$

Combining this, (2.131), the Cauchy–Schwarz inequality, the fact that $G ( r , x ) = | ( \nabla _ { M } h ) ( r , x ) |$ , and the fact that for al $r \in ( 0 , R )$ it holds that $G ( r , \chi ( r ) ) > 0$ , by (2.135) establishes that for all $r \in ( 0 , \Re _ { x } )$ it holds that

$$
\begin{array} { l } { \displaystyle q ( \boldsymbol { r } , \chi ( \boldsymbol { r } ) ) = \frac { \left| ( \partial _ { \boldsymbol { r } } h ) ( \boldsymbol { r } , \chi ( \boldsymbol { r } ) ) - a ^ { \prime } ( \boldsymbol { r } ) \right| } { G ( \boldsymbol { r } , \chi ( \boldsymbol { r } ) ) } = \frac { | \left. ( \nabla _ { M } h ) ( \boldsymbol { r } , \chi ( \boldsymbol { r } ) ) , \chi ^ { \prime } ( \boldsymbol { r } ) \right. | } { | ( \nabla _ { M } h ) ( \boldsymbol { r } , \chi ( \boldsymbol { r } ) ) | } } \\ { \displaystyle \qquad \leq \frac { | ( \nabla _ { M } h ) ( \boldsymbol { r } , \chi ( \boldsymbol { r } ) ) | \left| \chi ^ { \prime } ( \boldsymbol { r } ) \right| } { | ( \nabla _ { M } h ) ( \boldsymbol { r } , \chi ( \boldsymbol { r } ) ) | } = | \chi ^ { \prime } ( \boldsymbol { r } ) | . } \end{array}\tag{2.140}
$$

This and (2.135) imply that for all $r \in ( 0 , \Re _ { x } )$ it holds that

$$
r ^ { - 2 } \leq | \chi ^ { \prime } ( r ) | .\tag{2.141}
$$

Combining this and the fact that $\begin{array} { r } { \int _ { 0 } ^ { \mathfrak { R } _ { x } } r ^ { - 2 } \mathrm { d } r = + \infty } \end{array}$ shows that

$$
\int _ { 0 } ^ { \infty } | \chi ^ { \prime } ( r ) | \mathrm { d } r = + \infty .\tag{2.142}
$$

This and (2.136) demonstrate that the assumption in (2.134) leads to a contradiction. This proves (2.133).

Observe that (2.132), (2.133), and the fact that $I = \{ r \in ( 0 , { \mathfrak { r } } _ { 0 } ) \colon \beta ( r ) = + \infty \}$ ensure that for all $r \in ( 0 , R )$ it holds that $\beta ( r ) < + \infty$ . Combining this, (2.129), (2.131), and the fact that q is definable establishes that $\beta | _ { ( 0 , R ) } \colon ( 0 , R ) \to [ 0 , \infty )$ is definable. In the next step we note that the fact that $\beta | _ { ( 0 , R ) }$ is definable, [9, Theorem 2.10], and the fact that $\beta$ is nonnegative imply that there exists $\Re _ { \beta } \in ( 0 , R )$ which satisfies

$$
\beta | _ { ( 0 , \mathfrak { R } _ { \beta } ) } = 0 \quad \mathrm { o r } \quad \forall r \in ( 0 , \mathfrak { R } _ { \beta } ) \colon \beta ( r ) > 0 ;\tag{2.143}
$$

We claim that

$$
\exists \mathfrak { R } _ { \mathrm { i n t } } \in ( 0 , \mathfrak { R } _ { \beta } ) \colon \int _ { 0 } ^ { \mathfrak { R } _ { \mathrm { i n t } } } \beta ( r ) \mathrm { d } r < \infty .\tag{2.144}
$$

We prove (2.144) by contradiction. In the following we thus assume that for all $\rho \in ( 0 , \Re _ { \beta } )$ it holds that

$$
\int _ { 0 } ^ { \rho } \beta ( r ) \mathrm { d } r = \infty .\tag{2.145}
$$

Observe that (2.143) and (2.145) show that for all $r \in ( 0 , \Re _ { \beta } )$ it holds that

$$
\beta ( r ) > 0 .\tag{2.146}
$$

This and (2.129) demonstrate that for all $r \in ( 0 , \Re _ { \beta } )$ there exists $y \in M$ such that $( r , y ) \in S$ and $q ( r , y ) \ge { \textstyle \frac { 1 } { 2 } } \beta ( r )$ . Combining this and the definable choice theorem [9, Theorem 3.1] proves that there exists a definable function $\psi \colon ( 0 , \Re _ { \beta } ) \to M$ which satisfies that for all $r \in ( 0 , \Re _ { \beta } )$ it holds that

$$
( r , \psi ( r ) ) \in S \quad { \mathrm { a n d } } \quad q ( r , \psi ( r ) ) \geq { \frac { 1 } { 2 } } \beta ( r ) ;\tag{2.147}
$$

Note that (2.147), the fact that M is bounded, and Lemma 2.10 ensure that there exists $\Re _ { y } \in ( 0 , \Re _ { \beta } )$ which satisfies that

$$
\psi | _ { ( 0 , \mathfrak { R } _ { y } ) } \in C ^ { 1 } ( ( 0 , \mathfrak { R } _ { y } ) , M ) \quad \mathrm { a n d } \quad \int _ { 0 } ^ { \mathfrak { R } _ { y } } | \psi ^ { \prime } ( r ) | \mathrm { d } r < \infty .\tag{2.148}
$$

Observe that (2.130) and (2.147) establish that for all $r \in ( 0 , \Re _ { y } )$ it holds that

$$
h ( r , \psi ( r ) ) = a ( r ) .\tag{2.149}
$$

This, the fact that $h \in { \bf C } ^ { 1 } ( { \mathcal { O } } , \mathbb { R } ) , a \in C ^ { 1 } ( ( 0 , \mathfrak { r } _ { 0 } ) , \mathbb { R } )$ , and $\psi | _ { ( 0 , \mathfrak { R } _ { y } ) } \in C ^ { 1 } ( ( 0 , \mathfrak { R } _ { y } ) , M )$ , and the chain rule imply that for all $r \in ( 0 , \Re _ { y } )$ it holds that

$$
0 = \frac { \mathrm { d } } { \mathrm { d } r } ( h ( r , \psi ( r ) ) - a ( r ) ) = ( \partial _ { r } h ) ( r , \psi ( r ) ) + \langle ( \nabla _ { M } h ) ( r , \psi ( r ) ) , \psi ^ { \prime } ( r ) \rangle - a ^ { \prime } ( r ) .\tag{2.150}
$$

Hence, we obtain that for all $r \in ( 0 , \Re _ { y } )$ it holds that

$$
( \partial _ { r } h ) ( r , \psi ( r ) ) - a ^ { \prime } ( r ) = - \langle ( \nabla _ { M } h ) ( r , \psi ( r ) ) , \psi ^ { \prime } ( r ) \rangle .\tag{2.151}
$$

Furthermore, note that (2.130) and (2.147) show that for all $r \in ( 0 , \Re _ { \beta } )$ it holds that

$$
G ( r , \psi ( r ) ) > 0 .\tag{2.152}
$$

Combining (2.131), (2.151), the Cauchy–Schwarz inequality, and the fact that for all $( r , x ) \in { \mathcal { O } }$ it holds that $G ( r , x ) = | ( \nabla _ { M } h ) ( r , x ) |$ hence demonstrates that for all $r \in ( 0 , \Re _ { y } )$ it holds that

$$
\begin{array} { r l } & { q ( r , \psi ( r ) ) = \frac { \left| \left( \partial _ { r } h \right) \left( r , \psi ( r ) \right) - a ^ { \prime } ( r ) \right| } { G ( r , \psi ( r ) ) } = \frac { \left| \left. \left( \nabla _ { M } h \right) \left( r , \psi ( r ) \right) , \psi ^ { \prime } ( r ) \right. \right| } { \left| \left( \nabla _ { M } h \right) \left( r , \psi ( r ) \right) \right| } } \\ & { \qquad \leq \frac { \left| \left( \nabla _ { M } h \right) \left( r , \psi ( r ) \right) \right| \left| \psi ^ { \prime } ( r ) \right| } { \left| \left( \nabla _ { M } h \right) \left( r , \psi ( r ) \right) \right| } = | \psi ^ { \prime } ( r ) | . } \end{array}\tag{2.153}
$$

This and (2.147) prove that for all $r \in ( 0 , \Re _ { y } )$ it holds that

$$
\beta ( r ) \leq 2 | \psi ^ { \prime } ( r ) | .\tag{2.154}
$$

Combining this, (2.145), (2.148), and the fact that $\beta$ is nonnegative ensures that

$$
\infty = \int _ { 0 } ^ { \Re _ { y } } \beta ( r ) \mathrm { d } r \le 2 \int _ { 0 } ^ { \Re _ { y } } | \psi ^ { \prime } ( r ) | \mathrm { d } r < \infty .\tag{2.155}
$$

This contradiction shows (2.144).

Observe that (2.144) establishes that there exists $\Re _ { \mathrm { i n t } } \in ( 0 , \Re _ { \beta } )$ which satisfies that

$$
\int _ { 0 } ^ { \infty } \beta ( r ) \mathrm { d } r < \infty .\tag{2.156}
$$

Note that (2.156), the fact that $\beta | _ { ( 0 , \mathfrak { R } _ { \mathrm { i n t } } ) }$ has finite values and is definable, [9, Corollary 6.5], and the fact that $\beta$ is nonnegative prove items (i) and (ii). The proof of Lemma 2.13 is thus complete. □

## 3 Construction of the radial-energy regions and the energy transformations

As outlined in Subsection 1.5 above, the key issue in the proof of Theorem 1.1 is to establish Lyapunovtype stability (in the precise sense of (1.8) in Subsection 1.5) for the radial-angular DE system (see (1.6) in Subsection 1.5) as asserted in Proposition 5.3 in Section 5 below. The system in (1.6) already admits a Lyapunov function, namely, the scalar field h itself. However, the dissipation associated to this Lyapunov function and its Lyapunov energy $h ( r ( t ) , x ( t ) )$ is not large enough to directly allow us to conclude finiteness of the length of the angular component (see (1.10) in Subsection 1.5).

It is the key subject of this section to construct and study suitable transformation functions that we apply to the energy $h ( r ( t ) , x ( t ) )$ to obtain new Lyapunov-type functions with larger dissipation. In Subsection 3.1 we briefly recall some well-known concepts from the area of o-minimal geometry that we employ in the construction of the energy transformations. In Lemma 3.5 in Subsection 3.2 we construct the desired transformations $\mathcal { F } _ { 0 } , \mathcal { F } _ { 1 } , \ldots , \mathcal { F } _ { N - 1 }$ and their associated back-transformations $E _ { 1 } , E _ { 2 } , \dots , E _ { N }$ and establish lower and upper bounds for the partial derivative with respect to the energy variable of the back-transformations $( \partial _ { z } E _ { i } ) ( r , z )$ . In Lemma 3.7 in Subsection 3.3 we establish lower and upper bounds for the partial derivative with respect to the radial variable of the back-transformations $( \partial _ { r } E _ { i } ) ( r , z )$ Lemma 3.6 in Subsection 3.3 establishes in item (ii) a basic growth bound near the origin for definable and integrable functions, which we employ in the proof of Lemma 3.7.

## 3.1 Smooth cylindrical definable cell decompositions

In this subsection we recall some standard concepts from o-minimal geometry. We refer, for instance, to [9, Definition 2.4 and Section 6.2] for further reading.

Definition 3.1 (Definable partitions). Let $d \in \mathbb { N }$ , let $A \subseteq \mathbb { R } ^ { d }$ be a set, and let $\mathcal { P }$ be a set. Then we say that $\mathcal { P }$ is a definable partition of A if and only if

(i) it holds that $\mathcal { P }$ is a finite set,

(ii) it holds that $\textstyle \bigcup _ { B \in { \mathcal { P } } } B = A .$

(iii) it holds for all $B \in { \mathcal { P } }$ that B is nonempty and definable, and

(iv) it holds for all $B , C \in { \mathcal { P } }$ with $B \neq C$ that $B \cap C = \emptyset$

(cf. Definition 1.5).

Definition 3.2 (Smooth cylindrical definable cell decompositions). Let $k \in \mathbb N$ . Then

(i) for every set $\mathcal { P }$ we say that $\mathcal { P }$ is a cylindrical definable $C ^ { k }$ cell decomposition of R if and only if

(a) it holds that $\mathcal { P }$ is a definable partition of R and

(b) it holds for all $A \in { \mathcal { P } }$ that A is a singleton or a non-empty open interval,

(ii) for every set $\mathcal { P }$ and every $d \in \mathbb { N }$ we say that $\mathcal { P }$ is a cylindrical definable $C ^ { k }$ cell decomposition of $\mathbb { R } ^ { d + 1 }$ if and only if

(a) it holds that $\mathcal { P }$ is a definable partition of $\mathbb { R } ^ { d + 1 }$

(b) it holds that $\bigcup _ { E \in { \mathcal { P } } } \left\{ \left\{ x = ( x _ { 1 } , \ldots , x _ { d } ) \in \mathbb { R } ^ { d } \colon ( \exists v \in \mathbb { R } \colon ( x _ { 1 } , \ldots , x _ { d } , v ) \in E ) \right\} \right\}$ is a cylindrical definable $C ^ { k }$ cell decomposition of R<sup>d</sup>, and

(c) it holds for all $\begin{array} { r } { B \in \bigcup _ { E \in { \mathcal { P } } } \big \{ \{ x = ( x _ { 1 } , \dotsc , x _ { d } ) \in { \mathbb { R } } ^ { d } \colon ( \exists v \in { \mathbb { R } } \colon ( x _ { 1 } , \dotsc , x _ { d } , v ) \in E ) \} \big \} } \end{array}$ that there exist $q \in  { \mathbb { N } } _ { 0 }$ and $\zeta _ { 0 } , \zeta _ { 1 } , \dots , \zeta _ { q + 1 } \colon B \to [ - \infty , \infty ]$ such that

(I) it holds for all $x \in B { \mathrm { ~ t h a t ~ } } - \infty = \zeta _ { 0 } ( x ) < \zeta _ { 1 } ( x ) < \ldots < \zeta _ { q } ( x ) < \zeta _ { q + 1 } ( x ) = \infty$

(II) it holds for all $i \in \{ 1 , 2 , \ldots , q \}$ that $B \ni x \mapsto \zeta _ { i } ( x ) \in \mathbb { R }$ is definable,

(III) it holds for all $i \in \{ 1 , 2 , \dots , q \} , \xi \in B$ that there exists an open $U \subseteq \mathbb { R } ^ { d }$ and $F \in C ^ { k } ( U , \mathbb { R } )$ such that $\xi \in U$ and $F | _ { B \cap U } = \zeta _ { i } | _ { B \cap U }$ , and

(IV) it holds for all $V \in \mathcal { P }$ with $V \subseteq B \times \mathbb { R }$ that there exists $j \in \{ 0 , 1 , \ldots , q \}$ such that

$$
V \in \left\{ \mathrm { G r a p h } ( \zeta _ { j } ) , \left\{ ( x _ { 1 } , \dots , x _ { d } , v ) \in B \times \mathbb { R } \colon \zeta _ { j } ( x _ { 1 } , \dots , x _ { d } ) < v < \zeta _ { j + 1 } ( x _ { 1 } , \dots , x _ { d } ) \right\} \right\}\tag{3.1}
$$

(cf. Definitions 1.6 and 3.1).

Definition 3.3 (Compatible definable partitions). Let $d \in \mathbb { N } .$ , let $A \subseteq \mathbb { R } ^ { d }$ be a set, and let $\mathcal { P }$ be a definable partition of $\mathbb { R } ^ { d }$ (cf. Definition 3.1). Then we say that $\mathcal { P }$ is compatible with A if and only if there exists ${ \mathcal { A } } \subseteq { \mathcal { P } }$ such that $A = \bigcup _ { B \in { \mathcal { A } } } B$

Definition 3.4 (Refinements of definable partitions). Let $d \in \mathbb { N }$ and let $\mathcal { P }$ and $\mathcal { R }$ be definable partitions of $\mathbb { R } ^ { d }$ (cf. Definition 3.1). Then we say that $\mathcal { R }$ is a refinement of $\mathcal { P }$ if and only if for all $A \in { \mathcal { R } }$ there exists $B \in { \mathcal { P } }$ such that $A \subseteq B$

## 3.2 Upper and lower bounds for the energy-derivatives of the backtransformations

The next result, Lemma 3.5, is a key tool for the construction of the Lyapunov-type function Q for the radial-angular DE system $( r ( t ) , x ( t ) ) , t \in I ,$ in Proposition 5.3 in Section 5 below. Naturally, the function h in Proposition 5.3 also serves as the Lyapunov function for the radial-angular DE system and the associated Lyapunov energy $h ( r ( t ) , x ( t ) ) , t \in I ,$ is non-increasing over time but the dissipation associated to the Lyapunov function h is not large enough to directly provide an estimate for the length of $x ( t ) , t \in I$ . Instead the approach initiated by Lemma 3.5 is to employ transformations using the functions $\mathcal { F } _ { 0 } , \mathcal { F } _ { 1 } , \ldots , \mathcal { F } _ { N - 1 }$ in Lemma 3.5 and to consider the transformed Lyapunov-type energy

$$
\mathcal { F } _ { i } \big ( r ( t ) , ( \varphi \circ h ) ( r ( t ) , x ( t ) ) \big )\tag{3.2}
$$

for $t \in I , i \in \{ 0 , 1 , \ldots , N - 1 \}$ , where $\varphi \colon \mathbb { R } $ R is the normalization function which satisfies for all $e \in \mathbb { R }$ that $\varphi ( e ) \bar { = } ( 1 + e ^ { 2 } ) \bar { } ^ { - 1 / 2 } e$ . The functions $E _ { 1 } , E _ { 2 } , \dots , E _ { N }$ serve as the back-transformations in the sense that for all $i \in \{ 1 , 2 , \ldots , N \} , r \in ( 0 , \rho ) , z \in ( 0 , H )$ it holds that $\mathcal { F } _ { i - 1 } ( r , E _ { i } ( r , z ) ) = z \ ( \mathrm { s e e }$ item (v) in Lemma 3.5).

Lemma 3.5. Let $D \in \mathbb { N } , \ R , c \in ( 0 , \infty )$ , let $M \subseteq \mathbb { R } ^ { D }$ be a nonempty, compact, and definable $C ^ { 2 }$ submanifold of $\mathbb { R } ^ { D }$ , let $\Omega \subseteq ( 0 , R ) \times M$ be definable and open in $( 0 , R ) \times M$ , let $h \in { \bf C } ^ { 1 } ( \Omega , \mathbb { R } )$ be definable, let $a , b \colon ( 0 , R ) \to \mathbb { R }$ be definable, assume for all $r \in ( 0 , R )$ that $a ( r ) < b ( r )$ , assume for all $( r , x ) \in \Omega$ that

$$
a ( r ) \leq h ( r , x ) \leq b ( r ) ,\tag{3.3}
$$

and let $G \colon \Omega  \mathbb { R }$ satisfy for all $( r , x ) \in \Omega$ that $G ( r , x ) = | ( \nabla _ { M } h ) ( r , x ) |$ (cf. Definitions $\it { 2 . 2 , 2 . 5 , }$ and $\it { 2 . 7 ) }$ . Then there exist $\rho \in ( 0 , R )$ , N ∈ N, $H \in \mathsf { \Gamma } ( 0 , \infty )$ and definable $\mathfrak { e } _ { 0 } , \mathfrak { e } _ { 1 } , \ldots , \mathfrak { e } _ { N } \in C ^ { 2 } ( ( 0 , \rho ) , \mathbb { R } )$ $\mathcal { F } _ { 0 } , \mathcal { F } _ { 1 } , \dotsc , \mathcal { F } _ { N } \ \in \ C ( ( 0 , \rho ) \times  { \mathbb { R } } , [ 0 , H ] ) , E _ { 1 } , E _ { 2 } , \dotsc , E _ { N } \ \in \ C ^ { 2 } ( ( 0 , \rho ) \times ( 0 , H ) ,  { \mathbb { R } } ) , T _ { 1 } , T _ { 2 } , \dotsc , T _ { N } \ \in$ $C ^ { 1 } ( ( 0 , \rho ) \times ( 0 , H ) , ( 0 , \infty ) )$ such that

(i) for all $r \in ( 0 , \rho )$ it holds that $a ( r ) = \mathfrak { e } _ { 0 } ( r ) < \mathfrak { e } _ { 1 } ( r ) < \ldots < \mathfrak { e } _ { N } ( r ) = b ( r )$

(ii) for all $\begin{array} { r l } & { r \in ( 0 , \rho ) , e \in \mathbb { R } \ w i t h \ i n f \big ( \{ \infty \} \cup \{ z \in \mathbb { R } : \exists x \in M : ( r , x ) \in \Omega \cap h ^ { - 1 } ( \{ e \} ) \cap G ^ { - 1 } ( \{ z \} ) \} \big ) = 0 } \\ & { s t h a t e \in \{ \mathfrak { e } _ { 0 } ( r ) , \mathfrak { e } _ { 1 } ( r ) , \dots , \mathfrak { e } _ { N } ( r ) \} , } \end{array}$ it hold

(iii) for all $i \in \{ 0 , 1 , \ldots , N - 1 \} , \ r \in ( 0 , \rho ) , \ u \in ( - \infty , \mathfrak { e } _ { i } ( r ) ] , \ v \in [ \mathfrak { e } _ { i + 1 } ( r ) , \infty ) , \ z , w \in ( \mathfrak { e } _ { i } ( r ) , \mathfrak { e } _ { i + 1 } ( r ) )$ with $z <$ w it holds that $0 = \mathcal { F } _ { i } ( r , u ) < \mathcal { F } _ { i } ( r , z ) < \mathcal { F } _ { i } ( r , w ) < \mathcal { F } _ { i } ( r , v ) = H ,$

(iv) for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that $\mathcal { F } _ { i } \big | _ { \{ ( r , e ) \in ( 0 , \rho ) \times \mathbb { R } : \mathscr { e } _ { i } ( r ) < e < \mathfrak { e } _ { i + 1 } ( r ) \} }$ is twice continuously diferentiable,

(v) for all $i \in \{ 1 , 2 , \ldots , N \} , r \in ( 0 , \rho ) , z \in ( 0 , H )$ it holds that

$$
\begin{array} { r } { \mathfrak { e } _ { i - 1 } ( r ) < E _ { i } ( r , z ) < \mathfrak { e } _ { i } ( r ) , \quad \mathcal { F } _ { i - 1 } ( r , E _ { i } ( r , z ) ) = z , \qquad a n d \qquad T _ { i } ( r , z ) = \partial _ { z } E _ { i } ( r , z ) > 0 , } \end{array}\tag{3.4}
$$

and

(vi) for all $i \in \{ 1 , 2 , \ldots , N \} , ( r , x ) \in \Omega \cap ( ( 0 , \rho ) \times M ) \mathrm { ~ } w i t h \mathrm { ~ } \mathfrak { e } _ { i - 1 } ( r ) < h ( r , x ) < \mathfrak { e } _ { i } ( r )$ it holds that

$$
T _ { i } ( r , { \mathcal { F } } _ { i - 1 } ( r , h ( r , x ) ) ) \leq c G ( r , x ) .\tag{3.5}
$$

Proof of Lemma 3.5. Throughout this proof let $\pi _ { j } \colon \mathbb { R } ^ { D } \  \ \mathbb { R } , \ j \ \in \ \{ 1 , \ldots , D \}$ , satisfy for all $j \in$ $\{ 1 , \dots , D \} , x = ( x _ { 1 } , \dots , x _ { D } ) \in \mathbb { R } ^ { D }$ that $\pi _ { j } ( x ) = x _ { j }$ and let $B \in [ 0 , \infty ) , \ : H \in ( 0 , \infty ) , \ : \mathcal { D } \subseteq ( 0 , R ) \times \mathbb { R }$ satisfy

$$
\begin{array} { r l } & { B = \displaystyle \sum _ { j = 1 } ^ { D } \bigg ( \operatorname* { m a x } _ { \boldsymbol { x } \in \boldsymbol { M } } \pi _ { j } ( \boldsymbol { x } ) - \operatorname* { m i n } _ { \boldsymbol { x } \in \boldsymbol { M } } \pi _ { j } ( \boldsymbol { x } ) \bigg ) , \qquad \boldsymbol { H } = 1 + 2 B / c , } \\ { \mathrm { a n d } \qquad } & { \mathcal { D } = \big \{ ( r , e ) \in ( 0 , R ) \times \mathbb { R } \colon \big [ \exists \boldsymbol { x } \in \boldsymbol { M } \colon ( ( r , \boldsymbol { x } ) \in \Omega ) \wedge ( h ( r , \boldsymbol { x } ) = e ) \big ] \big \} . } \end{array}\tag{3.6}
$$

We claim that for all $u , v \in \mathbb { R }$ with $u < v$ and every definable function $\psi \colon ( u , v ) \times ( 0 , 1 ) \to M$ there exists $A \in ( 0 , \infty )$ such that for all $\delta \in ( 0 , 1 )$ there exists a finite set $\mathcal { E } \subseteq ( u , v )$ such that $\psi ( \cdot , \delta ) | _ { ( u , v ) \backslash \mathcal { E } } \in$ $C ^ { 1 } ( ( u , v ) \backslash \mathcal { E } , M )$ and that

$$
\int _ { ( u , v ) \backslash \mathcal { E } } \left| \left( \partial _ { e } \psi \right) \left( e , \delta \right) \right| \mathrm { d } e \leq A .\tag{3.7}
$$

To prove this claim, let $u , v \in$ R satisfy $u \ < \ v .$ , and let $\psi \colon ( u , v ) \times ( 0 , 1 ) \to M$ be definable. In the following let $\phi _ { j } \colon ( 0 , 1 ) \times ( u , v )  \mathbb { R } , j \in \{ 1 , \ldots , D \}$ , satisfy for all $j \in \{ 1 , \ldots , D \} , ( \delta , e ) \in ( 0 , 1 ) \times ( u , v )$ that

$$
\phi _ { j } ( \delta , e ) = \pi _ { j } ( \psi ( e , \delta ) ) .\tag{3.8}
$$

Observe that (3.8), the fact that ψ is definable, and [9, Theorem $6 . 7 ]$ imply that there exist definable partitions $\mathcal { V } _ { j } , j \in \{ 1 , \ldots , D \}$ , of $( 0 , 1 ) \times ( u , v )$ which satisfy that for all $j \in \{ 1 , \dots , D \} , S \in \mathcal { V } _ { j }$ it holds that S is a $C ^ { 1 }$ submanifold of $\mathbb { R } ^ { 2 }$ and $\phi _ { j } | _ { S } \in \mathbf { C } ^ { 1 } ( S , \mathbb { R } )$ (cf. Definition 3.1). This and [9, Theorem $6 . 6 ]$ show that there exists a cylindrical definable $C ^ { 2 }$ cell decomposition $\mathcal { P }$ of $\mathbb { R } ^ { 2 }$ which is compatible with $( 0 , 1 ) \times ( u , v )$ and which satisfies for all $j \in \{ 1 , \dots , D \} , S \in \mathcal { V } _ { j }$ that

$$
\begin{array} { r } { S = \bigcup _ { C \in \mathcal { P } , C \subseteq S } C } \end{array}\tag{3.9}
$$

(cf. Definitions 3.2 and 3.3). In the following let $\mathcal { U } \subseteq ( 0 , 1 ) \times ( u , v )$ satisfy

$$
\begin{array} { r } { \mathcal { U } = \big \{ ( \delta , e ) \in ( 0 , 1 ) \times ( u , v ) \colon \big [ \exists C \in \mathcal { P } , ~ \eta \in ( 0 , \infty ) \colon \{ \delta \} \times ( e - \eta , e + \eta ) \subseteq C \big ] \big \} . } \end{array}\tag{3.10}
$$

Note that (3.9), the fact that for all $j \in \{ 1 , \ldots , D \}$ it holds that $\mathcal { V } _ { j }$ is a definable partition of $( 0 , 1 ) \times ( u , v )$ and the fact that for all $j \in \{ 1 , \dots , D \} , S \in \mathcal { V } _ { j }$ it holds that $\phi _ { j } ^ { \mathsf { ^ { \prime } } } | _ { S } \in \mathbf { C } ^ { 1 } ( S , \mathbb { R } )$ demonstrate that for all $C \in \mathcal { P } , \ j \in \{ 1 , \ldots , D \}$ with $C \ \subseteq \ ( 0 , 1 ) \times ( u , v )$ there exists $S \ \in \ { \mathcal { V } } _ { j }$ such that $C \subseteq S$ and that $\phi _ { j } | _ { C } \in \mathbf { C } ^ { 1 } ( C , \mathbb { R } )$ . Combining (3.10), the fact that $\mathcal { P }$ is a definable partition of $\mathbb { R } ^ { 2 }$ , the fact that $\mathcal { P }$ is compatible with $( 0 , 1 ) \times ( u , v )$ , the chain rule, and item (ii) in Lemma 2.8 therefore ensures that U is definable and that there exist definable $d _ { j } \colon \mathcal { U } \to \mathbb { R } , j \in \{ 1 , \dots , D \}$ , which satisfy for all $j \in \{ 1 , \ldots , D \}$ $( \delta , e ) \in \mathcal { U }$ that

$$
\begin{array} { r } { d _ { j } ( \delta , e ) = \left( \partial _ { e } \phi _ { j } \right) ( \delta , e ) . } \end{array}\tag{3.11}
$$

In the following let $\mathcal { L } _ { j } , \mathcal { Z } _ { j } , \mathcal { W } _ { j } \subseteq \mathcal { U } , j \in \{ 1 , \dots , D \}$ , satisfy for all $j \in \{ 1 , \ldots , D \}$ that

$$
{ \mathcal { L } } _ { j } = \{ ( \delta , e ) \in { \mathcal { U } } \colon d _ { j } ( \delta , e ) < 0 \} , \qquad { \mathcal { Z } } _ { j } = \{ ( \delta , e ) \in { \mathcal { U } } \colon d _ { j } ( \delta , e ) = 0 \} ,\tag{3.12}
$$

$$
\mathrm { a n d } \qquad \mathcal { W } _ { j } = \{ ( \delta , e ) \in \mathcal { U } : d _ { j } ( \delta , e ) > 0 \} .\tag{3.13}
$$

Observe that (3.12), the fact that U is definable, the fact that for all $j \in \{ 1 , \ldots , D \}$ it holds that $d _ { j }$ is definable, the fact that $\mathcal { P }$ is a definable partition of $\mathbb { R } ^ { 2 }$ , and [9, Theorem $6 . 6 ]$ establish that there exists a cylindrical definable $C ^ { 2 }$ cell decomposition $\mathcal { S }$ of $\mathbb { R } ^ { 2 }$ which is a refinement of $\mathcal { P }$ , which is compatible with $u ,$ and which satisfies for all $j \in \{ 1 , \dots , D \} , S \in \{ \mathcal { L } _ { j } , \mathcal { Z } _ { j } , \mathcal { W } _ { j } \}$ that

$$
\begin{array} { r } { S = \bigcup _ { C \in \mathcal { S } , C \subseteq S } C } \end{array}\tag{3.14}
$$

(cf. Definition 3.4). Note that (3.12), (3.14), and the fact that $\mathcal { S }$ is a definable partition of $\mathbb { R } ^ { 2 }$ prove that for all $C \in \mathcal { S } , j \in \{ 1 , \dots , D \} , \xi , \eta \in C$ with $C \subseteq { \mathcal { U } }$ it holds that

$$
\mathrm { s i g n } \big ( d _ { j } ( \xi ) \big ) = \mathrm { s i g n } \big ( d _ { j } ( \eta ) \big ) .\tag{3.15}
$$

Next, observe that (3.10), the fact that $\mathcal { S }$ is a refinement of $\mathcal { P } .$ , and the fact that $\mathcal { S }$ is compatible with U imply that for all $C \in \mathcal { S } , \delta \in ( 0 , 1 ) , p , q \in ( u , v )$ with $C \subseteq ( 0 , 1 ) \times ( u , v ) , \{ \delta \} \times ( p , q ) \subseteq C$ , and $p < q$ it holds that

$$
C \subseteq U .\tag{3.16}
$$

This, $( 3 . 8 ) , ( 3 . 1 1 )$ , (3.15), the fact that for all $C \in \mathcal { P } , j \in \left\{ 1 , \dots , D \right\}$ with $C \subseteq ( 0 , 1 ) \times ( u , v )$ it holds that $\phi _ { j } | _ { C } \in \mathbf { C } ^ { 1 } ( C , \mathbb { R } )$ , the fact that $\mathcal { S }$ is a refinement of $\mathcal { P } _ { : }$ , the fact that for all $\delta \in ( 0 , 1 ) , C \in \mathcal { S }$ it holds that $\{ e \in \mathbb { R } \colon ( \delta , e ) \in C \}$ is empty, a singleton, or an open interval, the fact that $\mathcal { S }$ is compatible with $( 0 , 1 ) \times ( u , v )$ , and the mean value theorem show that for all $\delta \in ( 0 , 1 )$ there exist $k \in \{ 1 , \ldots , \# \mathcal { S } \}$ and $i = ( t _ { 0 } , \ldots , t _ { k } ) \in \mathbb { R } ^ { k + 1 }$ with $u = t _ { 0 } < \cdot \cdot \cdot < t _ { k } = v$ such that for all $\ell \in \{ 1 , \dots , k \} , j \in \{ 1 , \dots , D \}$ it holds that $\psi ( \cdot , \delta ) | _ { ( t _ { \ell - 1 } , t _ { \ell } ) } \in C ^ { 1 } ( ( t _ { \ell - 1 } , t _ { \ell } ) , M )$ and that $\phi _ { j } ( \delta , \cdot ) \big | _ { ( t _ { \ell - 1 } , t _ { \ell } ) }$ is non-decreasing or non-increasing. In the following let $A \in ( 0 , \infty )$ satisfy $A = 1 + B \left( \# { \mathcal { S } } \right)$ . We claim that for all $\delta \in ( 0 , 1 )$ there exists a finite set $\mathcal { E } \subseteq ( u , v )$ such that $\psi ( \cdot , \delta ) | _ { ( u , v ) \backslash \mathcal { E } } \in C ^ { 1 } ( ( u , v ) \backslash \mathcal { E } , M )$ and that

$$
\int _ { ( u , v ) \backslash \mathcal { E } } \left| \left( \partial _ { e } \psi \right) \left( e , \delta \right) \right| \mathrm { d } e \leq A .\tag{3.17}
$$

To prove (3.17), let $\delta \in ( 0 , 1 ) , k \in \{ 1 , \dots , \# \mathcal { S } \}$ , and $t = ( t _ { 0 } , \ldots , t _ { k } ) \in \mathbb { R } ^ { k + 1 }$ , assume that $u = t _ { 0 } < \cdots <$ $t _ { k } = v$ and that for all $\ell \in \{ 1 , \dots , k \} , \ j \in \{ 1 , \dots , D \}$ it holds that $\psi ( \cdot , \delta ) | _ { ( t _ { \ell - 1 } , t _ { \ell } ) } \in C ^ { 1 } ( ( t _ { \ell - 1 } , t _ { \ell } ) , M )$ and that $\phi _ { j } ( \delta , \cdot ) \big | _ { ( t _ { \ell - 1 } , t _ { \ell } ) }$ is non-decreasing or non-increasing, and let $\mathcal { E } = \{ t _ { 1 } , \ldots , t _ { k - 1 } \}$ . Note that the fact that for all $\ell \in \{ 1 , \dots , k \} , j \in \{ 1 , \dots , D \}$ it holds that $\phi _ { j } ( \delta , \cdot ) | _ { ( t _ { \ell - 1 } , t _ { \ell } ) } \in C ^ { 1 } ( ( t _ { \ell - 1 } , t _ { \ell } ) , \mathbb { R } )$ and that $\phi _ { j } ( \delta , \cdot ) \big | _ { ( t _ { \ell - 1 } , t _ { \ell } ) }$ is either non-decreasing or non-increasing, the fundamental theorem of calculus, (3.8), and the fact that ψ takes values in $M$ demonstrate that for all $\ell \in \{ 1 , \ldots , k \} , j \in \{ 1 , \ldots , D \} , p , q \in ( t _ { \ell - 1 } , t _ { \ell } )$ with $p < q$ it holds that

$$
\int _ { p } ^ { q } \big | \big ( \partial _ { e } \phi _ { j } \big ) \big ( \delta , e \big ) \big | \mathrm { d } e = \lvert \phi _ { j } ( \delta , q ) - \phi _ { j } ( \delta , p ) \rvert \le \operatorname* { m a x } _ { x \in M } \pi _ { j } ( x ) - \operatorname* { m i n } _ { x \in M } \pi _ { j } ( x ) .\tag{3.18}
$$

Combining this and the monotone convergence theorem ensures that for all $\ell \in \{ 1 , \dots , k \} , j \in \{ 1 , \dots , D \}$ it holds that

$$
\int _ { t _ { \ell - 1 } } ^ { t _ { \ell } } \big | \big ( \partial _ { e } \phi _ { j } \big ) \big ( \delta , e \big ) \big | \mathrm { d } e \leq \operatorname* { m a x } _ { x \in M } \pi _ { j } ( x ) - \operatorname* { m i n } _ { x \in M } \pi _ { j } ( x ) .\tag{3.19}
$$

In addition, observe that (3.8) and the fact that for all $\ell \in \{ 1 , \ldots , k \}$ it holds that $\psi ( \cdot , \delta ) | _ { ( t _ { \ell - 1 } , t _ { \ell } ) } \in$ $C ^ { 1 } ( ( t _ { \ell - 1 } , t _ { \ell } ) , M )$ establish that for all $e \in ( u , v ) \backslash \mathcal { E }$ it holds that

$$
\left| \left( \partial _ { e } \psi \right) \left( e , \delta \right) \right| = \sqrt { \sum _ { j = 1 } ^ { D } ( \left( \partial _ { e } \phi _ { j } \right) \left( \delta , e \right) ) ^ { 2 } } \leq \sum _ { j = 1 } ^ { D } \bigl | \bigl ( \partial _ { e } \phi _ { j } \bigr ) \bigl ( \delta , e \bigr ) \bigr | .\tag{3.20}
$$

This, (3.6), (3.19), the fact that $( u , v ) \backslash \mathcal { E }$ is the disjoint union of $( t _ { \ell - 1 } , t _ { \ell } ) , \ell \in \{ 1 , \dots , k \}$ , and the fact that $k \leq \# \mathcal { S }$ prove that

$$
\begin{array} { r l } & { \displaystyle \int _ { ( u , v ) \setminus \xi } \left. \left( \partial _ { e } \psi \right) \left( e , \delta \right) \right. \mathrm { d } e \leq \displaystyle \sum _ { \ell = 1 } ^ { k } \displaystyle \sum _ { j = 1 } ^ { D } \int _ { t _ { \ell - 1 } } ^ { t _ { \ell } } \left. \left( \partial _ { e } \phi _ { j } \right) \left( \delta , e \right) \right. \mathrm { d } e } \\ & { \qquad \leq \displaystyle \sum _ { \ell = 1 } ^ { k } \displaystyle \sum _ { j = 1 } ^ { D } \left( \operatorname* { m a x } _ { x \in M } \pi _ { j } ( x ) - \operatorname* { m i n } _ { x \in M } \pi _ { j } ( x ) \right) = k B \leq B \left( \# \mathcal { S } \right) < A . } \end{array}\tag{3.21}
$$

Combining the fact that $\mathcal { E } \subseteq ( u , v )$ is finite, the fact that $\psi ( \cdot , \delta ) | _ { ( u , v ) \backslash \mathcal { E } } \in C ^ { 1 } ( ( u , v ) \backslash \mathcal { E } , M )$ , and the fact that $\delta \in ( 0 , 1 )$ was arbitrary hence establishes (3.17). Note that $( 3 . 1 7 )$ , the fact that $A \in ( 0 , \infty )$ , and the fact that u, v, ψ were arbitrary prove $( 3 . 7 )$

In the following let $\mu \colon { \mathcal { D } } \to [ 0 , \infty )$ and $\mathcal { C } \subseteq \mathcal { D }$ satisfy for all $( r , e ) \in \mathcal { D }$ that

$$
\mu ( r , e ) = \operatorname* { i n f } \left\{ G ( r , x ) \colon \left[ \left( ( r , x ) \in \Omega \right) \wedge \left( h ( r , x ) = e \right) \right] \right\} \qquad { \mathrm { a n d } } \qquad \mathcal { C } = \left\{ ( s , v ) \in \mathcal { D } \colon \mu ( s , v ) = 0 \right\} .\tag{3.22}
$$

Observe that (3.6), the fact that h is definable, and the fact that Ω is definable imply that

$$
\mathcal { D } = \{ ( r , h ( r , x ) ) \colon ( r , x ) \in \Omega \}\tag{3.23}
$$

and that D is definable. This and (3.22) show that for all $( r , e ) \in \mathcal { D }$ there exists $x \in M$ such that $( r , x ) \in \Omega$ and

$$
h ( r , x ) = e \qquad \mathrm { a n d } \qquad 0 \leq \mu ( r , e ) \leq G ( r , x ) < \infty .\tag{3.24}
$$

In the following let ${ \mathcal { I } } \subseteq { \mathcal { D } } \times [ 0 , \infty )$ satisfy

$$
\mathcal { I } = ( \mathcal { D } \times [ 0 , \infty ) ) \backslash \big \{ ( r , h ( r , x ) , z ) \colon \big [ ( ( r , x ) \in \Omega ) \wedge ( z \in [ 0 , \infty ) ) \wedge ( G ( r , x ) < z ) \big ] \big \} .\tag{3.25}
$$

Note that (3.25), the fact that D and Ω are definable, and the fact that h and G are definable demonstrate that $\mathcal { I }$ is definable. Moreover, observe that (3.25) ensures that for all $( r , e , z ) \in \mathcal { I } , x \in M$ with $( r , x ) \in \Omega$ and $h ( r , x ) = e$ it holds that

$$
z \leq G ( r , x ) .\tag{3.26}
$$

Combining this and (3.22) establishes that for all $( r , e , z ) \in \mathcal { I }$ it holds that

$$
z \leq \mu ( r , e ) .\tag{3.27}
$$

Next we combine (3.22), (3.24), and (3.25) to obtain that for all $( r , e ) \in \mathcal { D }$ it holds that

$$
( r , e , \mu ( r , e ) ) \in \mathcal { I } .\tag{3.28}
$$

In the following let $\Gamma \subseteq \mathcal { I }$ satisfy

$$
\Gamma = \mathcal { I } \backslash \big \{ ( r , e , z ) \in \mathcal { I } \colon \big [ \exists v \in [ 0 , \infty ) \colon \big ( ( r , e , v ) \in \mathcal { I } \big ) \wedge ( z < v ) \big ] \big \} .\tag{3.29}
$$

Note that (3.29) and the fact that J is definable prove that Γ is definable. In the next step we combine (3.27), (3.28), and (3.29) to obtain that for all $( r , e ) \in \mathcal { D }$ it holds that

$$
( r , e , \mu ( r , e ) ) \in \Gamma .\tag{3.30}
$$

Furthermore, observe that (3.28), (3.29), and the fact that for all $( r , e , z ) \in \Gamma , v \in [ 0 , \infty )$ with $( r , e , v ) \in \mathcal { I }$ it holds that $v \leq z$ imply that for all $( r , e , z ) \in \Gamma$ it holds that

$$
\mu ( r , e ) \leq z .\tag{3.31}
$$

This, (3.27), (3.30), and the fact that $\Gamma \subseteq \mathcal { I }$ show that

$$
\Gamma = \{ ( r , e , \mu ( r , e ) ) \colon ( r , e ) \in \mathcal { D } \} .\tag{3.32}
$$

Combining this and the fact that Γ is definable demonstrates that $\mu$ is definable. This and (3.22) ensure that ${ \mathcal { C } } = \mu ^ { - 1 } ( \{ 0 \} )$ is definable. We claim that for all $r \in ( 0 , R )$ it holds that

$$
\# \{ e \in \mathbb { R } \colon ( r , e ) \in \mathcal { C } \} < \infty .\tag{3.33}
$$

To prove (3.33), let $r \in ( 0 , R )$ . We prove (3.33) by contradiction. We thus assume that $\# \{ e \in \mathbb { R } \colon ( r , e ) \in$ ${ \mathcal { C } } \} = \infty$ . Note that the fact that C is definable and the fact that $r \in ( 0 , R )$ establish that $\{ e \in \mathbb { R } \colon ( r , e ) \in$ $\mathcal { C } \}$ is definable. Combining this, the assumption that $\# \{ e \in \mathbb { R } \colon ( r , e ) \in \mathcal { C } \} = \infty .$ , and the fact that definable subsets of R are finite unions of points and intervals proves that there exist $u , v \in$ R with $u < v$ which satisfy

$$
( u , v ) \subseteq \{ e \in \mathbb { R } \colon ( r , e ) \in \mathcal { C } \} .\tag{3.34}
$$

Observe that (3.22) and (3.34) imply that for all $e \in ( u , v )$ it holds that

$$
( r , e ) \in { \mathcal { D } } \qquad \mathrm { a n d } \qquad \mu ( r , e ) = 0 .\tag{3.35}
$$

This and (3.27) show that for all $( e , \delta ) \in ( u , v ) \times ( 0 , 1 )$ it holds that

$$
( r , e , \delta ) \notin { \mathcal { I } } .\tag{3.36}
$$

Combining this, (3.25), and (3.35) demonstrates that for all $( e , \delta ) \in ( u , v ) \times ( 0 , 1 )$ there exists $x \in M$ such that

$$
( r , x ) \in \Omega , \qquad h ( r , x ) = e , \qquad \mathrm { a n d } \qquad G ( r , x ) < \delta .\tag{3.37}
$$

In the following let $\mathcal { H } \subseteq ( u , v ) \times ( 0 , 1 ) \times M$ satisfy

$$
\begin{array} { r } { \mathcal { H } = \big \{ ( e , \delta , x ) \in ( u , v ) \times ( 0 , 1 ) \times M \colon \big [ ( ( r , x ) \in \Omega ) \wedge ( h ( r , x ) = e ) \wedge ( G ( r , x ) < \delta ) \big ] \big \} . } \end{array}\tag{3.38}
$$

Note that (3.37), (3.38), and the fact that $M , \Omega , h ,$ and $G$ are definable ensure that H is definable and that

$$
\left\{ ( e , \delta ) \in ( u , v ) \times ( 0 , 1 ) \colon \left[ \exists x \in M \colon ( e , \delta , x ) \in \mathcal { H } \right] \right\} = ( u , v ) \times ( 0 , 1 ) .\tag{3.39}
$$

This, (3.38), the fact that H is definable, and [9, Theorem 3.1] establish that there exists a definable function $\chi \colon ( u , v ) \times ( 0 , 1 ) \to M$ which satisfies for all $( e , \delta ) \in ( u , v ) \times ( 0 , 1 )$ that

$$
( r , \chi ( e , \delta ) ) \in \Omega , \quad h ( r , \chi ( e , \delta ) ) = e , \qquad \mathrm { a n d } \qquad G ( r , \chi ( e , \delta ) ) < \delta .\tag{3.40}
$$

Observe that (3.7) and the fact that $\chi$ is definable prove that there exists $A \in ( 0 , \infty )$ which satisfies that for all $\delta \in ( 0 , 1 )$ there exists a finite set $\mathcal { E } \subseteq ( u , v )$ such that $\chi ( \cdot , \delta ) | _ { ( u , v ) \backslash \mathcal { E } } \in C ^ { 1 } ( ( u , v ) \backslash \mathcal { E } , M )$ and

$$
\int _ { ( u , v ) \backslash \mathcal { E } } \left| \left( \partial _ { e } \chi \right) ( e , \delta ) \right| \mathrm { d } e \leq A .\tag{3.41}
$$

In the following let $\delta \in ( 0 , 1 )$ satisfy

$$
\delta = \operatorname* { m i n } \biggl \{ \frac { 1 } { 2 } , \frac { v - u } { 2 A } \biggr \} ,\tag{3.42}
$$

and let $\mathcal { E } \subseteq ( u , v )$ be finite with the property that $\chi ( \cdot , \delta ) | _ { ( u , v ) \backslash \mathcal { E } } \in C ^ { 1 } ( ( u , v ) \backslash \mathcal { E } , M )$ and with the property that

$$
\int _ { ( u , v ) \backslash \mathcal { E } } \left| \left( \partial _ { e } \chi \right) ( e , \delta ) \right| \mathrm { d } e \leq A .\tag{3.43}
$$

Note that (3.40), the fact that E is finite, the fact that $\chi ( \cdot , \delta ) | _ { ( u , v ) \backslash \mathcal { E } } \in C ^ { 1 } ( ( u , v ) \backslash \mathcal { E } , M )$ , the fact that $h \in { \bf C } ^ { 1 } ( \Omega , \mathbb { R } )$ , the fact that for all $( s , x ) \in \Omega$ it holds that $G ( s , x ) = | ( \nabla _ { M } h ) ( s , x ) |$ , the chain rule, and the Cauchy–Schwarz inequality imply that for all $e \in ( u , v ) \backslash \mathcal { E }$ it holds that

$$
\begin{array} { l } { \displaystyle 1 = \frac { \mathrm { d } } { \mathrm { d } e } h ( r , \chi ( e , \delta ) ) = \left. ( \nabla _ { M } h ) ( r , \chi ( e , \delta ) ) , \left( \partial _ { e } \chi \right) ( e , \delta ) \right. } \\ { \displaystyle \quad \leq G ( r , \chi ( e , \delta ) ) \big \vert \left( \partial _ { e } \chi \right) ( e , \delta ) \big \vert \leq \delta \big \vert \left( \partial _ { e } \chi \right) \left( e , \delta \right) \big \vert . } \end{array}\tag{3.44}
$$

Combining this, (3.43), and the fact that $\mathcal { E }$ is finite shows that

$$
v - u = \int _ { ( u , v ) \backslash \mathcal { E } } 1 \mathrm { d } e \leq \delta \int _ { ( u , v ) \backslash \mathcal { E } } \left| \left( \partial _ { e } \chi \right) \left( e , \delta \right) \right| \mathrm { d } e \leq \delta A .\tag{3.45}
$$

This, (3.42), and the fact that $v - u > 0$ demonstrate that $\begin{array} { r } { v - u \le \delta A \le \frac { v - u } { 2 } < v - u } \end{array}$ . This contradiction and the fact that $r \in ( 0 , R )$ was arbitrary prove (3.33).

Observe that (3.22) and (3.24) ensure that for all $( r , e ) \in \mathcal { D } \backslash \mathcal { C }$ it holds that

$$
0 < \mu ( r , e ) < 2 \mu ( r , e ) < \infty .\tag{3.46}
$$

Combining this and (3.27) establishes that for all $( r , e ) \in \mathcal { D } \backslash \mathcal { C }$ it holds that

$$
( r , e , 2 \mu ( r , e ) ) \not \in \mathcal { I } .\tag{3.47}
$$

This, (3.25), and (3.46) prove that for all $( r , e ) \in \mathcal { D } \backslash \mathcal { C }$ there exists $x \in M$ such that

$$
( r , x ) \in \Omega , \qquad h ( r , x ) = e , \qquad \mathrm { a n d } \qquad G ( r , x ) < 2 \mu ( r , e ) .\tag{3.48}
$$

In the following let ${ \mathcal { K } } \subseteq ( { \mathcal { D } } \backslash { \mathcal { C } } ) \times M$ satisfy

$$
{ \mathrm { \AA } } = \left\{ ( r , e , x ) \in ( { \mathcal { D } } \backslash { \mathcal { C } } ) \times M : \left[ ( ( r , x ) \in \Omega ) \wedge ( h ( r , x ) = e ) \wedge ( G ( r , x ) < 2 \mu ( r , e ) ) \right] \right\} . \qquad\tag{3.49}
$$

Note that (3.48), (3.49), the fact that $\mathcal { D } , \mathcal { C } , M .$ and Ω are definable, and the fact that $h , G ,$ and $\mu$ are definable imply that K is definable and that

$$
\left\{ ( r , e ) \in { \mathcal { D } } \backslash { \mathcal { C } } \colon \left[ \exists x \in M : ( r , e , x ) \in { \mathcal { K } } \right] \right\} = { \mathcal { D } } \backslash { \mathcal { C } } .\tag{3.50}
$$

Combining this, (3.49), the fact that K is definable, and [9, Theorem 3.1] shows that there exists a definable function $\zeta \colon \mathcal { D } \backslash \mathcal { C }  M$ which satisfies for all $( r , e ) \in \mathcal { D } \backslash \mathcal { C }$ that

$$
( r , \zeta ( r , e ) ) \in \Omega , \quad h ( r , \zeta ( r , e ) ) = e , \qquad \mathrm { a n d } \qquad G ( r , \zeta ( r , e ) ) < 2 \mu ( r , e ) .\tag{3.51}
$$

Observe that (3.51), the fact that $\zeta$ is definable, the fact that for all $j \in \{ 1 , \ldots , D \}$ it holds that $\pi _ { j }$ is definable, and [9, Theorem 6.7] demonstrate that there exist definable partitions $\mathcal { X } _ { j } , j \in \{ 1 , \ldots , D \}$ , of $\mathcal { D } \backslash \mathcal { C }$ which satisfy that for all $\mathbf { \bar { \chi } } _ { j } \in \{ 1 , \dots , D \} , S \in \mathcal { X } _ { j }$ it holds that $S$ is a $C ^ { 2 }$ submanifold of $\mathbb { R } ^ { 2 }$ and $( \pi _ { j } \circ \zeta ) | _ { S } \in \mathbf { C } ^ { 2 } ( S , \mathbb { R } )$ . Note that the fact that a and b are definable and [9, Corollary 6.5] ensure that there exists $\kappa \in ( 0 , R )$ which satisfies that $a | _ { ( 0 , \kappa ) } , b | _ { ( 0 , \kappa ) } \in C ^ { 2 } ( ( 0 , \kappa ) , \mathbb { R } )$ . This, the fact that D and C are definable, the fact that a and b are definable, the fact that for all $j \in \{ 1 , \ldots , D \}$ it holds that $\mathcal { X } _ { j }$ is a definable partition of ${ \mathcal { D } } \backslash { \mathcal { C } } .$ , and [9, Theorem 6.6] establish that there exists a cylindrical definable $C ^ { 2 }$ cell decomposition $\mathcal { R }$ of $\mathbb { R } ^ { 2 }$ which is compatible with $( 0 , \kappa / 2 ) \times \mathbb { R } , \mathcal { D } , \mathcal { C } , \{ ( r , a ( r ) ) \colon r \in ( 0 , \kappa / 2 ) \}$ , and $\left\{ \left( r , b ( r ) \right) : r \in \left( 0 , \kappa / 2 \right) \right\}$ , and which satisfies for all $j \in \{ 1 , \dots , D \} , S \in \mathcal { X } _ { j }$ that

$$
\begin{array} { r } { S = \bigcup _ { C \in \mathcal { R } , C \subseteq S } C . } \end{array}\tag{3.52}
$$

In the following let $\mathcal { O } \subseteq \mathcal { D } \backslash \mathcal { C }$ satisfy

$$
\begin{array} { r } { \mathcal { O } = \bigcup _ { C \in \mathcal { R } , C \subseteq \mathcal { D } \backslash \mathcal { C } , C \mathrm { ~ o p e n } } C . } \end{array}\tag{3.53}
$$

Observe that (3.52), (3.53), the fact that for all $j \in \{ 1 , \ldots , D \}$ it holds that $\mathcal { X } _ { j }$ is a definable partition of $\mathcal { D } \backslash \mathcal { C }$ , and the fact that for all $j \in \{ 1 , \dots , D \} , S \in \mathcal { X } _ { j }$ it holds that $( \pi _ { j } \circ \zeta ) | _ { S } \overset { \cdot } { \in } \mathbf { C } ^ { 2 } ( S , \mathbb { R } )$ prove that O is definable and open and that $\zeta | _ { \mathcal { O } } \in C ^ { 2 } ( \mathcal { O } , M )$ . Combining this and the fact that $\zeta$ is definable implies that there exist definable functions $\nu _ { j } \in C ^ { 1 } ( \mathcal { O } , \mathbb { R } ) , j \in \{ 1 , \dots , D \}$ , which satisfy for all $j \in \{ 1 , \ldots , D \}$ $( r , e ) \in \mathcal { O }$ that

$$
\nu _ { j } ( r , e ) = \left( \partial _ { e } ( \pi _ { j } \circ \zeta ) \right) ( r , e ) .\tag{3.54}
$$

In the following let $\mathcal { V } _ { j , s } \subseteq O , j \in \{ 1 , \ldots , D \} , s \in \{ - 1 , 0 , 1 \}$ , satisfy for all $j \in \{ 1 , \ldots , D \} , s \in \{ - 1 , 0 , 1 \}$ that

$$
\mathcal { V } _ { j , s } = \{ ( r , e ) \in \mathcal { O } \colon \mathrm { s i g n } ( \nu _ { j } ( r , e ) ) = s \} .\tag{3.55}
$$

Note that (3.55), the fact that O is definable, the fact that for all $j \in \{ 1 , \ldots , D \}$ it holds that $\nu _ { j }$ is definable, the fact that $\mathcal { R }$ is a definable partition of $\mathbb { R } ^ { 2 }$ , and [9, Theorem $6 . 6 ]$ show that there exists a cylindrical definable $C ^ { 2 }$ cell decomposition $\mathcal { T }$ of $\mathbb { R } ^ { 2 }$ which is a refinement of R, which is compatible with ${ \mathcal { O } } ,$ and which satisfies for all $j \in \{ 1 , \ldots , D \} , s \in \{ - 1 , 0 , 1 \}$ that

$$
\begin{array} { r } { \mathcal { Y } _ { j , s } = \bigcup _ { C \in \mathcal { T } , C \subseteq \mathcal { V } _ { j , s } } C . } \end{array}\tag{3.56}
$$

This, (3.55), and the fact that $\mathcal { T }$ is a definable partition of $\mathbb { R } ^ { 2 }$ demonstrate that for all $C \in \mathcal { T }$ $j \in \{ 1 , \dots , D \} , \xi , \eta \in C$ with $C \subseteq \mathcal { O }$ it holds that

$$
\mathrm { s i g n } ( \nu _ { j } ( \xi ) ) = \mathrm { s i g n } ( \nu _ { j } ( \eta ) ) .\tag{3.57}
$$

Next, observe that (3.53), the fact that $\mathcal { T }$ is a refinement of ${ \mathcal { R } } .$ , the fact that $\mathcal { R }$ is compatible with D and ${ \mathcal { C } } ,$ and the fact that for all $C \in { \mathcal { R } }$ with nonempty interior in $\mathbb { R } ^ { 2 }$ it holds that $C$ is open ensure that for all $C \in \mathcal { T }$ with $C \subseteq { \mathcal { D } } \backslash { \mathcal { C } }$ and $C$ open it holds that

$$
C \subseteq O .\tag{3.58}
$$

In addition, note that (3.33), the fact that $\mathcal { T }$ is compatible with ${ \mathcal { C } } ,$ and the fact that for all $C \in \mathcal { T }$ with $C \subseteq { \mathcal { C } }$ and $C$ open it holds that there exist $r \in ( 0 , R ) , p , q \in \mathbb { R }$ with $p < q$ and $\{ r \} \times ( p , q ) \subseteq C$ establish that for all open $C \in \mathcal { T }$ it holds that

$$
C \cap { \mathcal { C } } = \emptyset .\tag{3.59}
$$

Moreover, observe that (3.3), (3.6), and (3.22) prove that

$$
\mathcal { C } \subseteq \mathcal { D } \subseteq \{ ( r , e ) \in ( 0 , R ) \times \mathbb { R } \colon a ( r ) \leq e \leq b ( r ) \} .\tag{3.60}
$$

In the next step we note that the fact that $\mathcal { T }$ is a cylindrical definable $C ^ { 2 }$ cell decomposition of $\mathbb { R } ^ { 2 }$ implies that there exists a definable partition $\mathcal { I }$ of R which satisfies that for all $J \in { \mathcal { S } }$ it holds that J is a singleton or an open interval and that

$$
{ \mathcal { I } } = \left\{ \{ r \in \mathbb { R } \colon [ \exists e \in \mathbb { R } \colon ( r , e ) \in C ] \} \colon C \in { \mathcal { T } } \right\} .\tag{3.61}
$$

Combining the fact that $\mathcal { R }$ is compatible with $( 0 , \kappa / 2 ) \times \mathbb { R }$ and the fact that $\mathcal { T }$ is a refinement of $\mathcal { R }$ therefore shows that

$$
\{ 0 \} \in { \mathcal { S } } \qquad \mathrm { a n d } \qquad \{ \kappa / 2 \} \in { \mathcal { I } } .\tag{3.62}
$$

This, the fact that $\mathcal { I }$ is a definable partition of R, and the fact that for all $J \in \mathcal { S }$ it holds that J is a singleton or an open interval demonstrate that there exists $\varepsilon \in ( 0 , \kappa )$ which satisfies

$$
\varepsilon = \operatorname* { m i n } \{ s \in ( 0 , \infty ) \colon \{ s \} \in { \mathcal { I } } \} \leq \kappa / 2 \qquad { \mathrm { a n d ~ } } \qquad ( 0 , \varepsilon ) \in { \mathcal { I } } .\tag{3.63}
$$

Combining this, (3.61), the fact that $\mathcal { T }$ is a cylindrical definable $C ^ { 2 }$ cell decomposition of $\mathbb { R } ^ { 2 }$ , the fact that $\mathcal { T }$ is compatible with $\left. \left( r , a ( r ) \right) : r \in \left( 0 , \kappa / 2 \right) \right.$ and $\left\{ \left( r , b ( r ) \right) : r \in \left( 0 , \kappa / 2 \right) \right\}$ , and the fact that for all $r \in ( 0 , R )$ it holds that $a ( r ) < b ( r )$ ensures that there exist m $\in \mathbb { N }$ with $m \geq 2$ and definable functions $\mathbf { f } _ { j } \in C ^ { 2 } ( ( 0 , \varepsilon ) , \mathbb { R } ) , j \in \{ 1 , \dots , m \}$ , which satisfy for all $r \in ( 0 , \varepsilon )$ that

$$
\mathbf { f } _ { 1 } ( r ) < \cdots < \mathbf { f } _ { m } ( r ) ,\tag{3.64}
$$

and which satisfy

$$
\big \{ C \in \mathcal { T } \colon \big [ ( C \subseteq ( 0 , \varepsilon ) \times \mathbb { R } ) \wedge ( C \mathrm { ~ i s ~ n o t ~ o p e n } ) \big ] \big \} = \big \{ \{ ( r , \mathbf { f } _ { j } ( r ) ) \colon r \in ( 0 , \varepsilon ) \} \colon j \in \{ 1 , \ldots , m \} \big \} .\tag{3.65}
$$

This, (3.64), and the fact that $\mathcal { T }$ is a cylindrical definable $C ^ { 2 }$ cell decomposition of $\mathbb { R } ^ { 2 }$ establish that for all $j \in \left\{ 1 , \dots , m - 1 \right\}$ it holds that

$$
\{ ( r , e ) \in ( 0 , \varepsilon ) \times \mathbb { R } \colon { \mathbf { f } } _ { j } ( r ) < e < { \mathbf { f } } _ { j + 1 } ( r ) \} \in \mathcal { T } .\tag{3.66}
$$

Next we combine (3.64), (3.65), the fact that $\mathcal { T }$ is compatible with $\{ ( r , a ( r ) ) \colon r \ \in \ ( 0 , \kappa / 2 ) \}$ and $\{ ( r , b ( r ) ) \colon r \in ( 0 , \kappa / 2 ) \}$ , the fact that for all open $C \in \mathcal { T } , \ r \in ( 0 , \varepsilon )$ with $C \subseteq ( 0 , \varepsilon ) \times$ R it holds that $\# \{ e \in \mathbb { R } \colon ( r , e ) \in C \} = \infty$ , and the fact that for all $r \in ( 0 , R )$ it holds that $a ( r ) < b ( r )$ to obtain that there exist $p , q \in \{ 1 , \ldots , m \}$ with $p < q$ which satisfy for all $r \in ( 0 , \varepsilon )$ that

$$
a ( r ) = \mathbf { f } _ { p } ( r ) \qquad { \mathrm { a n d } } \qquad b ( r ) = \mathbf { f } _ { q } ( r ) .\tag{3.67}
$$

In the following let $N \in \mathbb { N }$ satisfy $N = q - p$ , let $\mathfrak { e } _ { j } \colon ( 0 , \varepsilon )  \mathbb { R } , \ j \in \{ 0 , 1 , \ldots , N \}$ , satisfy for all $j \in \{ 0 , 1 , \ldots , N \} , r \in ( 0 , \varepsilon )$ that

$$
\begin{array} { r } { \mathfrak { e } _ { j } ( r ) = \mathbf { f } _ { p + j } ( r ) , } \end{array}\tag{3.68}
$$

and let $\mathcal { B } _ { i } \subseteq ( 0 , \varepsilon ) \times \mathbb { R } , i \in \{ 0 , 1 , \dots , N - 1 \}$ , satisfy for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ that

$$
\mathcal { B } _ { i } = \{ ( r , e ) \in ( 0 , \varepsilon ) \times \mathbb { R } \colon \mathfrak { e } _ { i } ( r ) < e < \mathfrak { e } _ { i + 1 } ( r ) \} .\tag{3.69}
$$

Observe that (3.64), (3.66), (3.68), (3.69), the fact that $N = q - p $ , and the fact that for all $j \in \{ 1 , \dots , m \}$ it holds that $\mathbf { f } _ { j }$ is definable and $\mathbf { f } _ { j } \in C ^ { 2 } ( ( 0 , \varepsilon ) , \mathbb { R } )$ prove that for all $j \in \{ 0 , 1 , \ldots , N \}$ it holds that ${ \mathfrak { e } } _ { j }$ is definable and $\mathfrak { e } _ { j } \in C ^ { 2 } ( ( 0 , \varepsilon ) , \mathbb { R } )$ and that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that $\textstyle B _ { i }$ is open and

$$
B _ { i } \in { \mathcal { T } } .\tag{3.70}
$$

Combining this, (3.59), and the fact that $\mathcal { T }$ is compatible with D implies that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that

$$
B _ { i } \cap { \mathcal { C } } = \emptyset \qquad { \mathrm { a n d } } \qquad \left( B _ { i } \cap { \mathcal { D } } = \emptyset \quad { \mathrm { o r } } \quad B _ { i } \subseteq { \mathcal { D } } \backslash { \mathcal { C } } \right) .\tag{3.71}
$$

Furthermore, note that (3.64), (3.67), (3.68), (3.69), and the fact that $N = q - p$ show that

$$
\begin{array} { r } { \{ ( r , e ) \in ( 0 , \varepsilon ) \times {  { \mathbb E } } \colon a ( r ) \le e \le b ( r ) \} = \bigcup _ { j = 0 } ^ { N } \{ ( r , \mathfrak { e } _ { j } ( r ) ) \colon r \in ( 0 , \varepsilon ) \} \cup \Big ( \bigcup _ { i = 0 } ^ { N - 1 } \mathcal { B } _ { i } \Big ) . } \end{array}\tag{3.72}
$$

This, (3.60), (3.64), (3.67), (3.68), and (3.71) demonstrate that for all $r \in ( 0 , \varepsilon )$ it holds that

$$
a ( r ) = \mathfrak { e } _ { 0 } ( r ) < \cdots < \mathfrak { e } _ { N } ( r ) = b ( r ) \qquad \mathrm { ~ a n d ~ } \qquad \{ e \in \mathbb { R } \colon ( r , e ) \in \mathcal { C } \} \subseteq \{ \mathfrak { e } _ { 0 } ( r ) , \ldots , \mathfrak { e } _ { N } ( r ) \} .\tag{3.73}
$$

Next, observe that (3.58), (3.70), and the fact that $\zeta | _ { \mathcal { O } } \in C ^ { 2 } ( \mathcal { O } , M )$ ensure that for all $i \in \{ 0 , 1 , \ldots , N { - } 1 \}$ with $B _ { i } \subseteq { \mathcal { D } } \backslash { \mathcal { C } }$ it holds that $\boldsymbol { B } _ { i } \subseteq \mathcal { O }$ and that $\zeta | _ { B _ { i } } \in C ^ { 2 } ( B _ { i } , M )$ . Combining this, (3.54), (3.57), and (3.70) establishes that for all $i \in \{ 0 , 1 , \ldots , N - 1 \} , j \in \{ 1 , \ldots , D \} , \xi , \eta \in \mathcal { B } _ { i }$ with $B _ { i } \subseteq { \mathcal { D } } \backslash { \mathcal { C } }$ it holds that

$$
\operatorname { s i g n } \bigl ( \bigl ( \partial _ { e } ( \pi _ { j } \circ \zeta ) \bigr ) ( \xi ) \bigr ) = \operatorname { s i g n } \bigl ( \bigl ( \partial _ { e } ( \pi _ { j } \circ \zeta ) \bigr ) ( \eta ) \bigr ) .\tag{3.74}
$$

In the following let A = {i ∈ {0, 1, . . . , N − 1} : B<sub>i</sub> ⊆ D\C} and for every $i \in \mathcal { A } , \ j \in \{ 1 , \ldots , D \}$ let $\sigma _ { i j } \in \{ - 1 , 0 , 1 \}$ satisfy that for all $( r , e ) \in B _ { i }$ it holds that

$$
\sigma _ { i j } = \operatorname { s i g n } \big ( \big ( \partial _ { e } ( \pi _ { j } \circ \zeta ) \big ) ( r , e ) \big )\tag{3.75}
$$

and let $\mathfrak { v } _ { i } \colon B _ { i } \to \mathbb { R }$ satisfy for all $( r , e ) \in B _ { i }$ that $\begin{array} { r } { \mathfrak { v } _ { i } ( r , e ) = \sum _ { i = 1 } ^ { D } \sigma _ { i j } \pi _ { j } ( \zeta ( r , e ) ) } \end{array}$ . Note that (3.51), the chain rule, and the Cauchy–Schwarz inequality prove that for all $i \in \mathcal { A } , \left( r , e \right) \in \mathcal { B } _ { i }$ it holds that

$$
1 = \left. { ( \nabla _ { M } h ) ( r , \zeta ( r , e ) ) , \left( \partial _ { e } \zeta \right) ( r , e ) } \right. \leq 2 \mu ( r , e ) | \big ( \partial _ { e } \zeta \big ) \left( r , e \right) |\tag{3.76}
$$

$$
\mathrm { a n d } \qquad \left( \partial _ { e } \mathfrak { v } _ { i } \right) \left( r , e \right) = \sum _ { j = 1 } ^ { D } | \left( \partial _ { e } ( \pi _ { j } \circ \zeta ) \right) \left( r , e \right) | \geq \frac { 1 } { 2 \mu ( r , e ) } > 0 .\tag{3.77}
$$

This, (3.6), and [9, Corollary 6.5] imply that there exist $\rho \in ( 0 , \varepsilon )$ and definable functions $\alpha _ { i } , \beta _ { i } , \Delta _ { i } \in$ $C ^ { 2 } ( ( 0 , \rho ) , \mathbb { R } ) , i \in \mathcal { \bar { A } }$ , which satisfy for all $i \in \mathcal { A } , r \in ( 0 , \rho )$ that

$$
\alpha _ { i } ( r ) = \operatorname* { l i m } _ { e \searrow _ { \mathfrak { c } _ { i } \mathfrak { c } _ { i } } ( r ) } \mathfrak { v } _ { i } ( r , e ) , \qquad \beta _ { i } ( r ) = \operatorname* { l i m } _ { e \nearrow _ { \mathfrak { c } _ { i + 1 } ( r ) } \mathfrak { v } _ { i } ( r , e ) , }\tag{3.78}
$$

$$
\quad { \mathrm { a n d } } \qquad 0 < \Delta _ { i } ( r ) = \beta _ { i } ( r ) - \alpha _ { i } ( r ) \leq B .\tag{3.79}
$$

Observe that (3.73) and the fact that $\rho < \varepsilon$ show that for all $r \in ( 0 , \rho )$ it holds that

$$
a ( r ) = \mathfrak { e } _ { 0 } ( r ) < \mathfrak { e } _ { 1 } ( r ) < \cdot \cdot \cdot < \mathfrak { e } _ { N } ( r ) = b ( r ) .\tag{3.80}
$$

In addition, note that (3.6), (3.22), (3.73), and the fact that $\rho < \varepsilon$ demonstrate that for all $r \in ( 0 , \rho )$ $e \in \mathbb { R }$ with in $\begin{array} { r } { \big ( \{ \infty \} \cup \{ z \in \mathbb { R } \colon \exists x \in M \colon ( r , x ) \in \Omega \cap h ^ { - 1 } ( \{ e \} ) \cap G ^ { - 1 } ( \{ z \} ) \} \big ) = 0 } \end{array}$ it holds that

$$
e \in \{ \mathfrak { e } _ { 0 } ( r ) , \mathfrak { e } _ { 1 } ( r ) , \dots , \mathfrak { e } _ { N } ( r ) \} .\tag{3.81}
$$

In the following let $\mathcal { F } _ { N } \colon ( 0 , \rho ) \times  { \mathbb { R } } \to [ 0 , H ]$ satisfy $\mathcal { F } _ { N } ~ = ~ 0$ and for every $i \in \{ 0 , 1 , \ldots , N - 1 \}$ let $\mathcal { F } _ { i } \colon ( 0 , \rho ) \times  { \mathbb { R } } \to [ 0 , H ]$ satisfy for all $( r , e ) \in ( 0 , \rho ) \times \mathbb { R }$ that

$$
\mathcal { F } _ { i } ( r , e ) = \left\{ \begin{array} { l l } { 0 } & { : e \leq \mathfrak { e } _ { i } ( r ) , } \\ { H ( \mathfrak { v } _ { i } ( r , e ) - \alpha _ { i } ( r ) ) / \Delta _ { i } ( r ) } & { : \mathfrak { e } _ { i } ( r ) < e < \mathfrak { e } _ { i + 1 } ( r ) , \ i \in \mathcal { A } , } \\ { H ( e - \mathfrak { e } _ { i } ( r ) ) / ( \mathfrak { e } _ { i + 1 } ( r ) - \mathfrak { e } _ { i } ( r ) ) } & { : \mathfrak { e } _ { i } ( r ) < e < \mathfrak { e } _ { i + 1 } ( r ) , \ i \not \in \mathcal { A } , } \\ { H } & { : e \geq \mathfrak { e } _ { i + 1 } ( r ) } \end{array} \right.\tag{3.82}
$$

Observe that (3.76), (3.78), and (3.82) ensure that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that $\mathcal { F } _ { i }$ is definable and that $\mathcal { F } _ { i } | _ { \mathcal { B } _ { i } \cap ( ( 0 , \rho ) \times \mathbb { R } ) } \in C ^ { 2 } ( \mathcal { B } _ { i } \cap ( ( 0 , \rho ) \times \mathbb { R } ) , { \mathrm { 1 } }$ R), that for all $i \in \{ 0 , 1 , \ldots , N - 1 \} , r \in ( 0 , \rho )$ it holds that R $\cdot \ni e \mapsto \mathcal { F } _ { i } ( r , e ) \in [ 0 , H ]$ is continuous and non-decreasing, and that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ $r \in ( 0 , \rho ) , e \in ( \mathfrak { e } _ { i } ( r ) , \mathfrak { e } _ { i + 1 } ( r ) )$ it holds that

$$
\left( \partial _ { e } \mathcal { F } _ { i } \right) \left( r , e \right) > 0 .\tag{3.83}
$$

Moreover, note that (3.69) and the fact that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that $\mathcal { F } _ { i } \big | _ { B _ { i } \cap ( ( 0 , \rho ) \times \mathbb { R } ) } \in$ $C ^ { 2 } ( B _ { i } \cap ( ( 0 , \rho ) \times \mathbb { R } )$ , R) establish that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that

$$
\begin{array} { r } { \mathcal { F } _ { i } \big | \{ ( r , e ) \in ( 0 , \rho ) \times \mathbb { R } \colon \mathfrak { e } _ { i } ( r ) < e < \mathfrak { e } _ { i + 1 } ( r ) \} \quad \mathrm { i s ~ t w i c e ~ c o n t i n u o u s l y ~ d i f f e r e n t i a b l e } . } \end{array}\tag{3.84}
$$

Combining (3.82), (3.83), and the mean value theorem hence proves that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ $r \in ( 0 , \rho ) , u \in ( - \infty , \mathfrak { e } _ { i } ( r ) ] , v \in [ \mathfrak { e } _ { i + 1 } ( r ) , \infty ) , z , w \in ( \mathfrak { e } _ { i } ( r ) , \mathfrak { e } _ { i + 1 } ( r ) )$ with $z < w$ it holds that

$$
0 = \mathcal { F } _ { i } ( r , u ) < \mathcal { F } _ { i } ( r , z ) < \mathcal { F } _ { i } ( r , w ) < \mathcal { F } _ { i } ( r , v ) = H .\tag{3.85}
$$

This, (3.82), the fact that for all $i \in \{ 0 , 1 , \ldots , N - 1 \} , r \in ( 0 , \rho )$ it holds that $\mathbb { R } \ni e \mapsto \mathcal { F } _ { i } ( r , e ) \in [ 0 , H ]$ is continuous, the intermediate value theorem, the fact that for all $j \in \{ 0 , 1 , \ldots , N \}$ it holds that ${ \mathfrak { e } } _ { j }$ is continuous, and the fact that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that ${ \mathcal { F } } _ { i }$ is continuous on $\boldsymbol { B } _ { i } \cap ( ( 0 , \rho ) \times \mathbb { R } )$ imply that for all $i \in \{ 0 , 1 , . . . , N - 1 \} , r \in ( 0 , \rho ) , \eta \in ( 0 , H )$ there exist $u , v \in ( \mathfrak { e } _ { i } ( r ) , \mathfrak { e } _ { i + 1 } ( r ) )$ with $u < v$ and an open $J \subseteq ( 0 , \rho )$ with $r \in J$ such that for all $s \in J , e , f \in \mathbb { R }$ with $e \leq u$ and $f \geq v$ it holds that

$$
0 \leq \mathcal { F } _ { i } ( s , e ) \leq \mathcal { F } _ { i } ( s , u ) < \eta \qquad \mathrm { a n d } \qquad 0 \leq H - \mathcal { F } _ { i } ( s , f ) \leq H - \mathcal { F } _ { i } ( s , v ) < \eta .\tag{3.86}
$$

Combining this, (3.82), the fact that for all $j \in \{ 0 , 1 , \ldots , N \}$ it holds that ${ \mathfrak { e } } _ { j }$ is continuous, and the fact that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that ${ \mathcal { F } } _ { i }$ is continuous on $B _ { i } \cap ( ( \bar { 0 } , \rho ) \times \mathbb { R } )$ shows that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that ${ \mathcal { F } } _ { i }$ is continuous on $( 0 , \rho ) \times \mathbb { R }$ . This, (3.83), (3.84), (3.85), the fact that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that ${ \mathcal { F } } _ { i }$ is definable, the intermediate value theorem, and the inverse function theorem demonstrate that there exist definable $E _ { 1 } , E _ { 2 } , \ldots , E _ { N } \in C ^ { 2 } ( ( 0 , \rho ) \times ( 0 , H ) , \mathbb { R } )$ and $T _ { 1 } , T _ { 2 } , \ldots , T _ { N } \in C ^ { 1 } ( ( 0 , \rho ) \times ( 0 , H ) , ( 0 , \infty ) )$ which satisfy for all $i \in \{ 1 , 2 , \ldots , N \} , r \in ( 0 , \rho ) , z \in ( 0 , H )$ that

$$
\mathfrak { c } _ { i - 1 } ( r ) < E _ { i } ( r , z ) < \mathfrak { e } _ { i } ( r ) , \quad \mathcal { F } _ { i - 1 } ( r , E _ { i } ( r , z ) ) = z , \qquad \mathrm { a n d } \qquad T _ { i } ( r , z ) = \partial _ { z } E _ { i } ( r , z ) > 0 ,\tag{3.87}
$$

and which satisfy for all $i \in \{ 0 , 1 , \ldots , N - 1 \} , ( r , e ) \in \mathcal { B } _ { i }$ with $r < \rho$ that

$$
T _ { i + 1 } ( r , \mathcal { F } _ { i } ( r , e ) ) = \frac { 1 } { \left( \partial _ { e } \mathcal { F } _ { i } \right) \left( r , e \right) } .\tag{3.88}
$$

Combining this, (3.6), (3.22), (3.76), (3.78), and (3.82) ensures that for all $i \in \mathcal { A } , \left( r , x \right) \in \Omega$ with $r < \rho$ and $\mathfrak { e } _ { i } ( r ) < h ( r , x ) < \mathfrak { e } _ { i + 1 } ( r )$ it holds that

$$
T _ { i + 1 } ( r , \mathcal { F } _ { i } ( r , h ( r , x ) ) ) \leq \frac { 2 \Delta _ { i } ( r ) } { H } \mu ( r , h ( r , x ) ) \leq \frac { 2 B } { 1 + 2 B / c } G ( r , x ) \leq c G ( r , x ) .\tag{3.89}
$$

This, (3.6), (3.69), (3.88), and the fact that for all $i \in \{ 0 , 1 , \ldots , N - 1 \} \backslash { \cal A }$ it holds that $B _ { i } \cap \mathcal { D } = \varnothing$ establish that for all $i \in \{ 1 , 2 , . . . , N \} , ( r , x ) \in \Omega \cap ( ( 0 , \rho ) \times M )$ with $\mathfrak { e } _ { i - 1 } ( r ) < h ( r , x ) < \mathfrak { e } _ { i } ( r )$ it holds that

$$
T _ { i } ( r , { \mathcal { F } } _ { i - 1 } ( r , h ( r , x ) ) ) \leq c G ( r , x ) .\tag{3.90}
$$

Combining (3.80), (3.81), (3.84), (3.85), and (3.87) therefore proves item (i), item (ii), item (iii), item (iv), item (v), and item (vi). The proof of Lemma 3.5 is thus complete. □

## 3.3 Upper and lower bounds for the radial-derivatives of the back-transformations

Lemma 3.6. Let $\mathfrak { r } _ { 0 } \in ( 0 , \infty )$ . Then

(i) for every definable and integrable n: $( 0 , \mathfrak { r } _ { 0 } ) \to [ 0 , \infty )$ there exists $\mathfrak { r } _ { 1 } \in \left( 0 , \mathfrak { r } _ { 0 } \right)$ such that $n | _ { ( 0 , \mathfrak { r } _ { 1 } ) }$ is continuous and

$$
\mathrm { l i m \ s u p } _ { s \to 0 } | s n ( s ) | = 0\tag{3.91}
$$

and

(ii) for every $\varepsilon \in ( 0 , \infty ) , N \in \mathbb { N }$ and all definable and integrable ${ \mathfrak { n } } _ { 1 } , { \mathfrak { n } } _ { 2 } , \ldots , { \mathfrak { n } } _ { N } \colon ( 0 , { \mathfrak { r } } _ { 0 } ) \to [ 0 , \infty )$ there exists $R \in ( 0 , \mathfrak { r } _ { 0 } )$ such that for all $i \in \{ 1 , 2 , \ldots , N \} , s \in ( 0 , R )$ it holds that

$$
s ^ { 2 } | \mathfrak { n } _ { i } ( s ) | ^ { 2 } \leq \varepsilon\tag{3.92}
$$

(cf. Definition 1.6).

Proof of Lemma 3.6. In the following we prove item (i). To prove item (i), let $n \colon ( 0 , \mathfrak { r } _ { 0 } ) \to [ 0 , \infty )$ be definable and integrable (cf. Definition 1.6). Observe that the assumption that n is definable and [9, Theorem 2.1 and Corollary 6.5] imply that there exists $R \in ( 0 , \mathfrak { r } _ { 0 } )$ which satisfies that ${ n } | _ { ( 0 , R ) }$ is continuous and is strictly increasing, strictly decreasing, or constant. In the following let ${ \mathfrak { r } } _ { 1 } \in ( 0 , R )$ . Note that the fact that $n | _ { ( 0 , R ) }$ is continuous and the fact that ${ \mathfrak { r } } _ { 1 } < R$ show that $n | _ { ( 0 , \mathfrak { r } _ { 1 } ) }$ is continuous. In our proof of item (i) we distinguish between the case that $n | _ { ( 0 , R ) }$ is non-decreasing and the case that $n | _ { ( 0 , R ) }$ is non-increasing. In the following we prove item (i) in the case that

$$
n { \big \vert } _ { ( 0 , R ) } \quad { \mathrm { i s ~ n o n \mathrm { - } d e c r e a s i n g } } .\tag{3.93}
$$

Observe that the assumption that $n | _ { ( 0 , R ) }$ is non-decreasing and the fact that n is nonnegative demonstrate that for all $r \in ( 0 , \mathfrak { r } _ { 1 } )$ it holds that

$$
0 \leq r n ( r ) \leq r n ( \mathfrak { r } _ { 1 } ) .\tag{3.94}
$$

Hence, we obtain that

$$
\operatorname* { l i m } _ { r  0 } r n ( r ) = 0 .\tag{3.95}
$$

This shows item (i) in the case that $n | _ { ( 0 , R ) }$ is non-decreasing. We now prove item (i) in the case that

$$
n { \big \vert } _ { ( 0 , R ) } \quad { \mathrm { i s ~ n o n - i n c r e a s i n g } } .\tag{3.96}
$$

Note that the assumption that $n | _ { ( 0 , R ) }$ is non-increasing and the fact that n is nonnegative ensure that for all $r \in ( 0 , \mathfrak { r } _ { 1 } )$ it holds that

$$
0 \leq { \frac { r } { 2 } } n ( r ) \leq \int _ { r / 2 } ^ { r } n ( s ) \mathrm { d } s \leq \int _ { 0 } ^ { r } n ( s ) \mathrm { d } s .\tag{3.97}
$$

In the next step we combine the assumption that n is non-negative and integrable on $( 0 , \mathfrak { r } _ { 0 } )$ and the dominated convergence theorem to obtain that

$$
\operatorname* { l i m } _ { r \to 0 } \int _ { 0 } ^ { r } n ( s ) \mathrm { d } s = 0 .\tag{3.98}
$$

This and (3.97) establish that

$$
\operatorname* { l i m } _ { r  0 } r n ( r ) = 0 .\tag{3.99}
$$

This establishes item (i) in the case that ${ n } | _ { ( 0 , R ) }$ is non-increasing. This completes the proof of item (i). In the following we prove item (ii). To prove item (ii), let $N ~ \in ~ \mathbb { N } , ~ \varepsilon ~ \in ~ ( 0 , \infty )$ , and let ${ \mathfrak { n } } _ { 1 } , \ldots , { \mathfrak { n } } _ { N } \colon ( 0 , { \mathfrak { r } } _ { 0 } ) \ \to \ [ 0 , \infty )$ be definable and integrable. Observe that item (i) proves that for all $i \in \{ 1 , \ldots , N \}$ it holds that

$$
\operatorname * { l i m } _ { r  0 } r ^ { 2 } ( \mathfrak { n } _ { i } ( r ) ) ^ { 2 } = 0 .\tag{3.100}
$$

Combining this and the assumption that $\varepsilon \ > \ 0$ implies that for every $i \in \{ 1 , \ldots , N \}$ there exists $\rho _ { i } \in ( 0 , \mathfrak { r } _ { 0 } )$ which satisfies for all $r \in ( 0 , \rho _ { i } )$ it holds that

$$
r ^ { 2 } ( \mathfrak { n } _ { i } ( r ) ) ^ { 2 } \leq \varepsilon .\tag{3.101}
$$

In the following let $\mathfrak { r } _ { \varepsilon } = \mathrm { m i n } _ { i \in \{ 1 , 2 , \ldots , N \} } \pmb { \rho } _ { i }$ . Note that the fact that $N \in \mathbb { N } .$ the fact that for all $i \in$ $\{ 1 , \ldots , N \}$ it holds that $\pmb { \rho } _ { i } \in ( 0 , \mathfrak { r } _ { 0 } )$ , and the fact that $\mathfrak { r } _ { \varepsilon } = \mathrm { m i n } _ { 1 \leq i \leq N } \rho _ { i }$ show that $\mathfrak { r } _ { \varepsilon } \in ( 0 , \mathfrak { r } _ { 0 } )$ and that for all $i \in \{ 1 , \ldots , N \}$ it holds that ${ \mathfrak { r } } _ { \varepsilon } \leq \rho _ { i }$ . This and (3.101) establish item (ii). The proof of Lemma 3.6 is thus complete. □

Lemma 3.7. Let $D , N \in \mathbb { N } , \ \mathfrak { r } _ { 0 } , \mathfrak { c } , H \in ( 0 , \infty ) , \ \mathfrak { r } _ { 1 } \ \in \ ( 0 , \mathfrak { r } _ { 0 } )$ , let $M \subseteq \mathbb { R } ^ { D }$ be a nonempty, compact, and definable C<sup>2</sup>-submanifold of $\mathbb { R } ^ { D }$ , let ${ \mathcal { O } } \subseteq ( 0 , \mathfrak { r } _ { 0 } ) \times M$ be definable and open in $( 0 , \mathfrak { r } _ { 0 } ) \times M$ , let $h \in { \bf C } ^ { 1 } ( \mathcal { O } , \mathbb { R } ) , \mathfrak { e } _ { 0 } , \mathfrak { e } _ { 1 } , \ldots , \mathfrak { e } _ { N } \in C ^ { 2 } ( ( 0 , \mathfrak { r } _ { 1 } ) , \mathbb { R } )$ be definable, assume for all $r \in ( 0 , \mathfrak { r } _ { 1 } )$ that

$$
\mathfrak { e } _ { 0 } ( r ) < \mathfrak { e } _ { 1 } ( r ) < \ldots < \mathfrak { e } _ { N } ( r ) ,\tag{3.102}
$$

for every $i \in \{ 1 , 2 , \ldots , N \}$ let $\mathcal { F } _ { i } \in C ( ( 0 , \mathfrak { r } _ { 1 } ) \times \mathbb { R } , [ 0 , H ] ) , E _ { i } = ( E _ { i } ( r , z ) ) _ { ( r , z ) \in ( 0 , \mathfrak { r } _ { 1 } ) \times ( 0 , H ) } \in C ^ { 2 } ( ( 0 , \mathfrak { r } _ { 1 } ) \times \mathbb { R } ^ { 3 } )$ $( 0 , H ) , \mathbb { R } )$ be definable, and assume for all $i \in \{ 1 , 2 , \ldots , N \} , r \in ( 0 , \mathfrak { r } _ { 1 } ) , u , v \in ( \mathfrak { e } _ { i - 1 } ( r ) , \mathfrak { e } _ { i } ( r ) ) , z \in ( 0 , H )$ with $u < v$ that

$$
\begin{array} { r } { \mathcal { F } _ { i } ( r , u ) < \mathcal { F } _ { i } ( r , v ) , \quad \mathfrak { e } _ { i - 1 } ( r ) < E _ { i } ( r , z ) < \mathfrak { e } _ { i } ( r ) , \quad \mathcal { F } _ { i } ( r , E _ { i } ( r , z ) ) = z , \quad a n d \quad \partial _ { z } E _ { i } ( r , z ) > 0 } \end{array}\tag{3.103}
$$

(cf. Definitions 2.2 and 2.5). Then there exist $\mathfrak { r } _ { 2 } \in \mathsf { \Gamma } ( 0 , \mathfrak { r } _ { 1 } )$ and definable $\mathbf { m } _ { 1 } , \mathbf { m } _ { 2 } , \dots , \mathbf { m } _ { N } \in$ $C ^ { 1 } ( ( 0 , \mathfrak { r } _ { 2 } ) , [ 1 , \infty ) )$ ) such that

(i) for all $i \in \{ 1 , 2 , \ldots , N \} , r \in ( 0 , \mathfrak { r } _ { 2 } )$ it holds that $\begin{array} { r } { \int _ { 0 } ^ { \mathfrak { r } _ { 2 } } \mathbf { m } _ { i } ( s ) \mathrm { d } s < \infty } \end{array}$ and $r ^ { 2 } ( \mathbf { m } _ { i } ( r ) ) ^ { 2 } \leq \mathbf { c }$ and

(ii) for all $i \in \{ 1 , 2 , \ldots , N \} , ( r , x ) \in \mathcal { O } \cap ( ( 0 , \mathfrak { r } _ { 2 } ) \times M ) \ w i t h \ \mathfrak { e } _ { i - 1 } ( r ) < h ( r , x ) < \mathfrak { e } _ { i } ( r )$ it holds that

$$
\big | \partial _ { r } h ( r , x ) - ( \partial _ { r } E _ { i } ) ( r , \mathscr { F } _ { i } ( r , h ( r , x ) ) ) \big | \leq \mathbf { m } _ { i } ( r ) \Big ( ( \partial _ { z } E _ { i } ) ( r , \mathscr { F } _ { i } ( r , h ( r , x ) ) ) + \mathbf { c } | ( \nabla _ { M } h ) ( r , x ) | \Big )\tag{3.104}
$$

(cf. Definition 2.7).

Proof of Lemma 3.7. Throughout this proof let $G \colon { \mathcal { O } }  \mathbb { R }$ satisfy for all $( r , x ) \in { \mathcal { O } }$ that $G ( r , x ) =$ $| ( \nabla _ { M } h ) ( r , x ) |$ and for every $i \in \{ 1 , 2 , \ldots , N \}$ let $\Omega _ { i } \subseteq \mathbb { R } ^ { D + 1 }$ satisfy

$$
\Omega _ { i } = \{ ( r , x ) \in \mathcal { O } \cap ( ( 0 , \mathfrak { r } _ { 1 } ) \times M ) \colon \mathfrak { e } _ { i - 1 } ( r ) < h ( r , x ) < \mathfrak { e } _ { i } ( r ) \} ,\tag{3.105}
$$

let $c _ { i } , T _ { i } \colon ( 0 , \mathfrak { r } _ { 1 } ) \times ( 0 , H )  \mathbb { R } \mathrm { ~ s a t i s f y ~ f o r ~ a l l ~ } r \in ( 0 , \mathfrak { r } _ { 1 } ) , z \in ( 0 , H )$ that

$$
c _ { i } ( r , z ) = \partial _ { r } E _ { i } ( r , z ) \qquad \mathrm { a n d } \qquad T _ { i } ( r , z ) = \partial _ { z } E _ { i } ( r , z ) ,\tag{3.106}
$$

and let $Z _ { i } \colon \mathcal { O } \cap ( ( 0 , \mathfrak { r } _ { 1 } ) \times M ) \to [ 0 , H ]$ satisfy for all $( r , x ) \in \mathcal { O } \cap \left( ( 0 , \mathfrak { r } _ { 1 } ) \times M \right)$ that

$$
Z _ { i } ( r , x ) = \mathcal { F } _ { i } ( r , h ( r , x ) )\tag{3.107}
$$

(cf. Definition 2.7). Observe that (3.103) demonstrates that for all $i \in \{ 1 , 2 , \ldots , N \} , r \in ( 0 , \mathfrak { r } _ { 1 } ) , e \in$ $( \mathfrak { e } _ { i - 1 } ( r ) , \mathfrak { e } _ { i } ( r ) )$ it holds that

$$
0 < \mathcal { F } _ { i } ( r , e ) < H .\tag{3.108}
$$

Combining this and (3.103) ensures that for all $i \in \{ 1 , 2 , \ldots , N \} , r \in ( 0 , \mathfrak { r } _ { 1 } ) , e \in ( \mathfrak { e } _ { i - 1 } ( r ) , \mathfrak { e } _ { i } ( r ) )$ it holds that

$$
E _ { i } ( r , { \mathcal { F } } _ { i } ( r , e ) ) = e .\tag{3.109}
$$

Next we combine (3.105), the fact that O is definable and open in $( 0 , \mathfrak { r } _ { 0 } ) \times M$ , and the fact that for all $i \in \{ 1 , 2 , \ldots , N \}$ it holds that $\boldsymbol { h } , \mathsf { e } _ { i - 1 }$ , and ${ \mathfrak { e } } _ { i }$ are definable and continuous to obtain that for all $i \in \{ 1 , 2 , \ldots , N \}$ it holds that $\Omega _ { i }$ is definable and is open in $( 0 , \mathfrak { r } _ { 1 } ) \times M$ . This, $( 3 . 1 0 3 ) , ( 3 . 1 0 5 ) , ( 3 . 1 0 7 )$ (3.108), (3.109), the fact that for all $i \in \{ 1 , 2 , \ldots , N \}$ it holds that $E _ { i } \in C ^ { 2 } ( ( 0 , \mathfrak { r } _ { 1 } ) \times ( 0 , H ) , \mathbb { R } )$ , the inverse function theorem, the fact that h is definable and $h \in { \bf C } ^ { 1 } ( \mathcal { O } , \mathbb { R } )$ , and the chain rule prove that for all $i \in \{ 1 , 2 , \ldots , N \}$ it holds that $\mathcal { F } _ { i } \big | _ { \{ ( r , e ) \in ( 0 , \mathfrak { r } _ { 1 } ) \times \mathbb { R } : \mathfrak { e } _ { i - 1 } ( r ) < e < \mathfrak { e } _ { i } ( r ) \} }$ is twice continuously diferentiable, $Z _ { i } | _ { \Omega _ { i } } \colon \Omega _ { i } \to \mathbb { R }$ is definable, $Z _ { i } | _ { \Omega _ { i } } \in { \bf C } ^ { 1 } ( \dot { \Omega } _ { i } , \dot { \mathbb { R } } )$ , and for all $( r , x ) \in \Omega _ { i }$ it holds that

$$
\begin{array} { r } { \varepsilon _ { i - 1 } ( r ) < h ( r , x ) < \mathfrak { e } _ { i } ( r ) \qquad \mathrm { ~ a n d ~ } \qquad 0 < Z _ { i } ( r , x ) < H . } \end{array}\tag{3.110}
$$

In the following for every $i \in \{ 1 , 2 , \ldots , N \}$ let $\mathbf { Z } _ { i } = ( \mathbf { Z } _ { i } ( r , x ) ) _ { ( r , x ) \in \Omega _ { i } } = Z _ { i } | _ { \Omega _ { i } }$ . Note that (3.103), (3.106), (3.107), (3.109), and (3.110) imply that for all $i \in \{ 1 , 2 , \ldots , N \} , ( r , x ) \in \Omega _ { i }$ it holds that

$$
h ( r , x ) = E _ { i } ( r , Z _ { i } ( r , x ) ) \qquad \mathrm { a n d } \qquad T _ { i } ( r , Z _ { i } ( r , x ) ) > 0 .\tag{3.111}
$$

Combining this and the fact that for all $i \in \{ 1 , 2 , \ldots , N \}$ it holds that $\Omega _ { i }$ is open in $( 0 , \mathfrak { r } _ { 1 } ) \times M$ shows that for all $i \in \{ 1 , 2 , \ldots , N \} , ( r , x ) \in \Omega _ { i }$ there exists $\delta \in ( 0 , \operatorname* { m i n } \{ r , \mathfrak { r } _ { 1 } - r \} )$ which satisfies that for all $s \in ( r - \delta , r + \delta )$ it holds that

$$
( s , x ) \in \Omega _ { i } \qquad \mathrm { a n d } \qquad h ( s , x ) = E _ { i } ( s , Z _ { i } ( s , x ) ) .\tag{3.112}
$$

This, the fact that for all $i \in \{ 1 , 2 , \ldots , N \}$ it holds that $\mathbf { Z } _ { i } \in \mathbf { C } ^ { 1 } ( \Omega _ { i } , \mathbb { R } )$ and $E _ { i } \in C ^ { 2 } ( ( 0 , \mathfrak { r } _ { 1 } ) \times ( 0 , H ) , \mathbb { R } )$ the fact that $h \in { \bf C } ^ { 1 } ( { \mathcal { O } } , \mathbb { R } )$ , and the chain rule establish that for all $i \in \{ 1 , 2 , \ldots , N \} , ( r , x ) \in \Omega _ { i }$ it holds that

$$
\partial _ { r } h ( r , x ) = ( \partial _ { r } E _ { i } ) ( r , Z _ { i } ( r , x ) ) + \big ( \partial _ { z } E _ { i } \big ) \big ( r , Z _ { i } ( r , x ) \big ) ( \partial _ { r } { \bf Z } _ { i } ) ( r , x ) .\tag{3.113}
$$

Furthermore, observe that (3.111) and the fact that for all $i \in \{ 1 , 2 , \ldots , N \}$ it holds that $\Omega _ { i }$ is open in $( 0 , \mathfrak { r } _ { 1 } ) \times M$ demonstrate that for every $i \in \{ 1 , 2 , \ldots , N \} , ( r , x ) \in \Omega _ { i } , \varepsilon \in ( 0 , \infty )$ and every $\gamma \in$ $C ^ { 1 } ( ( - \varepsilon , \varepsilon ) , M )$ with $\gamma ( 0 ) = x$ there exists $\delta \in ( 0 , \varepsilon )$ which satisfies that for all $t \in ( - \delta , \delta )$ it holds that

$$
( r , \gamma ( t ) ) \in \Omega _ { i } \qquad \mathrm { a n d } \qquad hslash ( r , \gamma ( t ) ) = E _ { i } ( r , Z _ { i } ( r , \gamma ( t ) ) ) .\tag{3.114}
$$

Combining this, the fact that for all $i \in \{ 1 , 2 , \ldots , N \}$ it holds that $\mathbf Z _ { i } \in { \bf C } ^ { 1 } ( \Omega _ { i } , \mathbb { R } )$ and $E _ { i } \in C ^ { 2 } ( ( 0 , \mathfrak { r } _ { 1 } ) \times$ $( 0 , H ) , \mathbb { R } )$ , the fact that $h \in { \bf C } ^ { 1 } ( { \mathcal { O } } , \mathbb { R } )$ , and the chain rule ensures that for every $i \in \{ 1 , 2 , \ldots , N \}$ $( r , x ) \in \Omega _ { i } , \varepsilon \in ( 0 , \infty )$ and every $\gamma \in C ^ { 1 } ( ( - \varepsilon , \varepsilon ) , M )$ with $\gamma ( 0 ) = x$ it holds that

$$
\begin{array} { l } { \displaystyle \langle ( \nabla _ { M } h ) ( \boldsymbol { r } , \boldsymbol { x } ) , \gamma ^ { \prime } ( 0 ) \rangle = \frac { \mathrm { d } } { \mathrm { d } t } h ( \boldsymbol { r } , \gamma ( t ) ) \big | _ { t = 0 } = \left( \partial _ { z } E _ { i } \right) \left( \boldsymbol { r } , Z _ { i } ( \boldsymbol { r } , \boldsymbol { x } ) \right) \frac { \mathrm { d } } { \mathrm { d } t } \mathbf { Z } _ { i } ( \boldsymbol { r } , \gamma ( t ) ) \big | _ { t = 0 } } \\ { = \left( \partial _ { z } E _ { i } \right) \left( \boldsymbol { r } , Z _ { i } ( \boldsymbol { r } , \boldsymbol { x } ) \right) \langle ( \nabla _ { M } \mathbf { Z } _ { i } ) ( \boldsymbol { r } , \boldsymbol { x } ) , \gamma ^ { \prime } ( 0 ) \rangle . } \end{array}\tag{3.115}
$$

Next, note that the fact that for all $i \in \{ 1 , 2 , \ldots , N \} , ( r , x ) \in \Omega _ { i }$ it holds that $( \nabla _ { M } \mathbf { Z } _ { i } ) ( r , x ) \in T _ { x } M$ , the fact that for all $i \in \{ 1 , 2 , \ldots , N \} , ( r , x ) \in \Omega _ { i }$ it holds that $\left( \partial _ { z } E _ { i } \right) \left( r , Z _ { i } ( r , x ) \right) \in \mathbb { R }$ , and the fact that for all $x \in M$ it holds that $T _ { x } M$ is a linear subspace of $\mathbb { R } ^ { D }$ prove that for all $i \in \{ 1 , 2 , \ldots , N \} , ( r , x ) \in \Omega _ { i }$ it holds that

$$
\left( \partial _ { z } E _ { i } \right) \left( r , Z _ { i } ( r , x ) \right) \left( \nabla _ { M } \mathbf { Z } _ { i } \right) \left( r , x \right) \in T _ { x } M\tag{3.116}
$$

(cf. Definition 2.3). This, (3.115), and Lemma 2.6 imply that for all $i \in \{ 1 , 2 , \ldots , N \} , ( r , x ) \in \Omega _ { i }$ it holds that

$$
\begin{array} { r } { ( \nabla _ { M } h ) ( r , x ) = \left( \partial _ { z } E _ { i } \right) \big ( r , Z _ { i } ( r , x ) \big ) ( \nabla _ { M } \mathbf { Z } _ { i } ) ( r , x ) . } \end{array}\tag{3.117}
$$

Combining (3.106) and (3.113) hence shows that for all $i \ \in \ \{ 1 , 2 , \ldots , N \} , \ ( r , x ) \ \in \ \Omega _ { i }$ it holds that $( \partial _ { r } h ) ( r , x ) = c _ { i } ( r , Z _ { i } ( r , x ) ) + T _ { i } ( r , Z _ { i } ( r , x ) ) ( \partial _ { r } { \bf Z } _ { i } ) ( r , x )$ and

$$
\begin{array} { r } { ( \nabla _ { M } h ) ( r , x ) = T _ { i } ( r , Z _ { i } ( r , x ) ) ( \nabla _ { M } \mathbf { Z } _ { i } ) ( r , x ) . } \end{array}\tag{3.118}
$$

This, (3.111), and the fact that for all $( r , x ) \in { \mathcal { O } }$ it holds that $G ( r , x ) = | ( \nabla _ { M } h ) ( r , x ) |$ establish that for all $i \in \{ 1 , 2 , \ldots , N \} , ( r , x ) \in \Omega _ { i }$ it holds that

$$
\frac { \left| ( \partial _ { r } \mathbf { Z } _ { i } ) ( r , x ) \right| } { 1 + \left| ( \nabla _ { M } \mathbf { Z } _ { i } ) ( r , x ) \right| } = \frac { \left| ( \partial _ { r } h ) ( r , x ) - c _ { i } ( r , Z _ { i } ( r , x ) ) \right| } { T _ { i } ( r , Z _ { i } ( r , x ) ) + G ( r , x ) } .\tag{3.119}
$$

In the following for every $i \in \{ 1 , 2 , \ldots , N \} , r \in ( 0 , \mathfrak { r } _ { 1 } )$ let $\mathcal { X } _ { i , r } \subseteq M$ satisfy

$$
\mathcal { X } _ { i , r } = \{ x \in M : ( r , x ) \in \Omega _ { i } \}\tag{3.120}
$$

and for every $i \in \{ 1 , 2 , \ldots , N \}$ let $\eta _ { i } \colon \left( 0 , \mathfrak { r } _ { 1 } \right) \to \left[ 0 , \infty \right]$ satisfy that for all $r \in ( 0 , \mathfrak { r } _ { 1 } )$ it holds that

$$
\eta _ { i } ( r ) = \operatorname* { s u p } _ { x \in \mathcal { X } _ { i , r } } \frac { \left| ( \partial _ { r } \mathbf { Z } _ { i } ) ( r , x ) \right| } { 1 + | ( \nabla _ { M } \mathbf { Z } _ { i } ) ( r , x ) | } ,\tag{3.121}
$$

where the supremum of the empty set is defined to be zero. Observe that (3.110), (3.120), (3.121), the fact that M is a compact definable C<sup>2</sup>-submanifold of $\mathbb { R } ^ { D }$ , the fact that for all $i \in \{ 1 , 2 , \ldots , N \}$ it holds that $\Omega _ { i }$ is definable and is open in $( 0 , \mathfrak { r } _ { 1 } ) \times M$ , the fact that for all $i \in \{ 1 , 2 , \ldots , N \}$ it holds that $\mathbf { Z } _ { i }$ is definable and $\mathbf { Z } _ { i } \in { \bf C } ^ { 1 } ( \Omega _ { i } , \mathbb { R } )$ , and Lemma 2.11 (applied for every $i \in \{ 1 , 2 , \ldots , N \}$ with $n \backsim D .$ $\mathfrak { r } _ { 0 } \ \cap \ \mathfrak { r } _ { 1 } , U \frown \Omega _ { i } , \ \Omega \frown \ \Omega _ { i } , Z \frown \mathbf { Z } _ { i } , \eta \frown \eta _ { i }$ in the notation of Lemma 2.11) demonstrate that for all $i \in \{ 1 , 2 , \ldots , N \}$ there exists $\pmb { \rho } _ { i } \in ( 0 , { \mathfrak { r } } _ { 1 } )$ which satisfies that

(I) it holds that $\eta _ { i } ( ( 0 , \pmb { \rho } _ { i } ) ) \subseteq \mathbb { R }$ and

(II) it holds that $( 0 , \pmb { \rho } _ { i } ) \ni r \mapsto \eta _ { i } ( r ) \in \mathbb { R }$ is continuously diferentiable, definable, and integrable.

$$
R \in ( 0 , \mathfrak { r } _ { 1 } )
$$

$$
R = \frac { 1 } { 2 } \operatorname* { m i n } \{ \rho _ { i } \colon i \in \{ 1 , 2 , \ldots , N \} \} .\tag{3.122}
$$

Note that item $( \mathrm { I } ) , ( 3 . 1 1 1 ) , ( 3 . 1 1 9 ) , ( 3 . 1 2 0 ) , ( 3 . 1 2 1 )$ , (3.122), and the fact that G is nonnegative ensure that for all $i \in \{ 1 , 2 , \ldots , N \} , ( r , x ) \in \Omega _ { i } \cap ( ( 0 , R ) \times M )$ it holds that

$$
\bigl | ( \partial _ { r } h ) ( r , x ) - c _ { i } ( r , Z _ { i } ( r , x ) ) \bigr | \leq \eta _ { i } ( r ) \bigl ( T _ { i } ( r , Z _ { i } ( r , x ) ) + G ( r , x ) \bigr ) .\tag{3.123}
$$

In the following for every $i \in \{ 1 , 2 , \ldots , N \}$ let $\mathbf { n } _ { i } \colon ( 0 , R ) \to [ 1 , \infty )$ satisfy that for all $r \in ( 0 , R )$ it holds that

$$
\mathbf { n } _ { i } ( r ) = 1 + \operatorname* { m a x } \{ 1 , \mathbf { c } ^ { - 1 } \} \eta _ { i } ( r ) .\tag{3.124}
$$

Observe that item (II), (3.122), (3.124), the fact that for all $i \in \{ 1 , 2 , \ldots , N \}$ it holds that $\eta _ { i } | _ { ( 0 , \rho _ { i } ) }$ is definable, and the fact that for all $i \in \{ 1 , 2 , \ldots , N \}$ it holds that $\eta _ { i }$ is nonnegative prove that for all $i \in \{ 1 , 2 , \ldots , N \}$ it holds that $\mathbf { n } _ { i }$ is definable, that $\mathbf { n } _ { i } \in C ^ { 1 } ( ( 0 , R ) , [ 1 , \infty ) )$ , and that

$$
\int _ { 0 } ^ { R } \mathbf { n } _ { i } ( r ) \mathrm { d } r = R + \operatorname* { m a x } \{ 1 , \mathbf { c } ^ { - 1 } \} \int _ { 0 } ^ { R } \eta _ { i } ( r ) \mathrm { d } r < \infty .\tag{3.125}
$$

Combining the fact that for all $i \in \{ 1 , 2 , \ldots , N \}$ it holds that ${ \bf n } _ { i }$ is definable and nonnegative and Lemma $3 . 6$ (applied with $\mathfrak { r } _ { 0 } \ \cap \ R , \ ( \mathfrak { n } _ { 1 } , \dots , \mathfrak { n } _ { N } ) \ \cap \ ( \mathfrak { n } _ { 1 } , \dots , \mathfrak { n } _ { N } ) , \ \varepsilon \ \frown \ \mathfrak { c }$ in the notation of Lemma 3.6) therefore implies that there exists ${ \mathfrak { r } } _ { 2 } \in ( 0 , R )$ which satisfies for all $i \in \{ 1 , 2 , \ldots , N \} , r \in ( 0 , \mathfrak { r } _ { 2 } )$ that

$$
r ^ { 2 } ( { \bf n } _ { i } ( r ) ) ^ { 2 } \leq { \bf c } .\tag{3.126}
$$

In the following for every $i \in \{ 1 , 2 , \ldots , N \}$ let $\mathbf { m } _ { i } \colon ( 0 , \mathfrak { r } _ { 2 } ) \to [ 1 , \infty )$ satisfy for all $r \in ( 0 , \mathfrak { r } _ { 2 } )$ that

$$
\mathbf { m } _ { i } ( r ) = \mathbf { n } _ { i } ( r ) = 1 + \operatorname* { m a x } \{ 1 , \mathbf { c } ^ { - 1 } \} \eta _ { i } ( r ) .\tag{3.127}
$$

Note that (3.125), (3.127), the fact that $\mathfrak { r } _ { 2 } \in ( 0 , R ) \subseteq ( 0 , \mathfrak { r } _ { 1 } )$ , and the fact that for all $i \in \{ 1 , 2 , \ldots , N \}$ it holds that ${ \bf n } _ { i }$ is definable and n<sub>i</sub> ∈ C<sup>1</sup>((0, R), [1, ∞)) show that for all $i \in \{ 1 , 2 , \ldots , N \}$ it holds that $\mathbf { m } _ { i }$ is definable, that $\mathbf { m } _ { i } \in C ^ { 1 } ( ( 0 , \mathfrak { r } _ { 2 } ) , [ 1 , \infty ) )$ , and that

$$
\int _ { 0 } ^ { \tau _ { 2 } } \mathbf { m } _ { i } ( r ) \mathrm { d } r \leq \int _ { 0 } ^ { R } \mathbf { n } _ { i } ( r ) \mathrm { d } r < \infty .\tag{3.128}
$$

This, (3.126), and (3.127) establish item (i).

Observe that (3.127), the fact that for all $i \in \{ 1 , 2 , \ldots , N \}$ it holds that $\eta _ { i }$ is nonnegative, the fact that max $\{ 1 , \mathbf { c } ^ { - 1 } \} \geq 1$ , and the fact that c max $\{ 1 , \mathbf { c } ^ { - 1 } \} \geq 1$ demonstrate that for all $i \in \{ 1 , 2 , \ldots , N \}$ $r \in ( 0 , \mathfrak { r } _ { 2 } )$ it holds that

$$
\eta _ { i } ( r ) \leq \mathbf { m } _ { i } ( r ) \qquad \mathrm { a n d } \qquad \eta _ { i } ( r ) \leq \mathbf { c } \mathbf { m } _ { i } ( r ) .\tag{3.129}
$$

Combining this, $( 3 . 1 1 1 ) , ( 3 . 1 2 3 )$ , the fact that G is nonnegative, and the fact that $( 0 , \mathfrak { r } _ { 2 } ) \subseteq ( 0 , R )$ ensures that for all $i \in \{ 1 , 2 , \ldots , N \} , ( r , x ) \in \Omega _ { i } \cap ( ( 0 , \mathfrak { r } _ { 2 } ) \times M )$ it holds that

$$
\begin{array} { r l } & { \left| ( \partial _ { r } h ) ( r , x ) - c _ { i } ( r , Z _ { i } ( r , x ) ) \right| \leq \eta _ { i } ( r ) T _ { i } ( r , Z _ { i } ( r , x ) ) + \eta _ { i } ( r ) G ( r , x ) } \\ & { \qquad \leq \mathbf { m } _ { i } ( r ) T _ { i } ( r , Z _ { i } ( r , x ) ) + \mathbf { c m } _ { i } ( r ) G ( r , x ) } \\ & { \qquad = \mathbf { m } _ { i } ( r ) \bigl ( T _ { i } ( r , Z _ { i } ( r , x ) ) + \mathbf { c } G ( r , x ) \bigr ) . } \end{array}\tag{3.130}
$$

This, (3.105), (3.106), (3.107), and the fact that for all $( r , x ) \in { \mathcal { O } }$ it holds that $G ( r , x ) = | ( \nabla _ { M } h ) ( r , x ) |$ prove item (ii). The proof of Lemma 3.7 is thus complete. □

## 4 Construction and regularity of the truncation of the transformed energy

As outlined in Subsection 1.5 above, the key issue in the proof of Theorem 1.1 is to establish Lyapunovtype stability (in the precise sense of (1.8) in Subsection 1.5) for the radial-angular DE system (see (1.6) in Subsection 1.5) as asserted in Proposition 5.3 in Section 5 below. A dificulty that arises in the construction of $Q$ in (1.13) in Subsection 1.5 is that the transformed energy function $Z _ { i }$ is $C ^ { 1 }$ on the set where $( r , h ( r , x ) ) \in \mathbf { H } _ { i }$ , but need not be continuously diferentiable on its entire domain and its composition with a locally absolutely continuous trajectory need thus not be locally absolutely continuous.

We solve this issue by employing a truncation and approximation procedure for the transformed energy $Z _ { i } ( r , x )$ . Specifically, we approximate $Z _ { i } ( r , x )$ by max{ε, min $\{ H - \varepsilon , Z _ { i } ( r , x ) \} \}$ for $\varepsilon \in ( 0 , H / 2 )$ and it is precisely the subject of Lemma 4.3 in this section to establish regularity properties and estimates for the truncated transformed energy.

## 4.1 Vanishing of the derivatives of absolutely continuous functions on their level sets

In the following elementary Lemma 4.1 we recall the well-known fact that for every $c \in \mathbb { R }$ and every absolutely continuous function $f$ we have that $f ^ { \prime }$ vanishes almost everywhere on the c-level set $f ^ { - 1 } ( \{ c \} )$ . Only for completeness, we include a detailed proof for Lemma 4.1 here.

Lemma 4.1. Let $I \subseteq \mathbb { R }$ be a compact interval, let $c \in \mathbb { R }$ , and let $f \colon I  \mathbb { R }$ be absolutely continuous. Then

$$
\int _ { f ^ { - 1 } ( \{ c \} ) } | f ^ { \prime } ( t ) | \mathrm { d } t = 0 .\tag{4.1}
$$

Proof of Lemma $\it 4 . 1$ . Throughout this proof, let $\mu$ denote Lebesgue measure on $\mathbb { R } .$ let $E = f ^ { - 1 } ( \{ c \} )$ , let $A \subseteq E$ be the set of isolated points of $E ,$ let $B \subseteq { \mathrm { i n t } } ( I )$ satisfy

$$
B = \{ t \in \operatorname { i n t } ( I ) \colon f { \mathrm { ~ i s ~ n o t ~ d i f f e r e n t i a b l e ~ a t ~ } } t \} ,\tag{4.2}
$$

and let $Z = A \cup B \cup ( I \backslash \operatorname { i n t } ( I ) )$ . Note that the fact that A is the set of isolated points of E implies that for all $t \in A$ there exist $p , q \in \mathbb { Q }$ with $p < t < q$ such that

$$
E \cap ( p , q ) = \{ t \} .\tag{4.3}
$$

Combining this and the fact that $\mathbb { Q } ^ { 2 }$ is countable shows that A is at most countable. Hence, we obtain that A is Lebesgue measurable and that

$$
\mu ( A ) = 0 .\tag{4.4}
$$

In addition, observe that (4.2), the fact that $f$ is absolutely continuous, and the fundamental theorem of calculus for absolutely continuous functions establish that B is Lebesgue measurable and that

$$
\mu ( B ) = 0 .\tag{4.5}
$$

This, (4.4), and the fact that $\mu ( I \backslash \operatorname { i n t } ( I ) ) = 0$ demonstrate that $Z$ is Lebesgue measurable and that

$$
0 \leq \mu ( Z ) \leq \mu ( A ) + \mu ( B ) + \mu ( I \backslash \operatorname { i n t } ( I ) ) = 0 .\tag{4.6}
$$

We claim that for all $t \in E \backslash Z$ it holds that

$$
f ^ { \prime } ( t ) = 0 .\tag{4.7}
$$

To prove this claim, let $t \in E \backslash Z$ . Note that the fact that $t \in E \backslash Z$ ensures that f is diferentiable at t and there exists $( s _ { k } ) _ { k \in \mathbb { N } } \subseteq E \backslash \{ t \}$ which satisfies that

$$
\operatorname* { l i m } _ { k \to \infty } s _ { k } = t .\tag{4.8}
$$

Observe that (4.8), the fact that f is diferentiable at $t ,$ and the fact that $t \in E$ prove that

$$
f ^ { \prime } ( t ) = \operatorname* { l i m } _ { k \to \infty } { \frac { f ( s _ { k } ) - f ( t ) } { s _ { k } - t } } = \operatorname* { l i m } _ { k \to \infty } { \frac { c - c } { s _ { k } - t } } = 0 .\tag{4.9}
$$

Combining this and the fact that $t \in E \backslash Z$ was chosen arbitrarily establishes (4.7).

Note that the fact that $f$ is absolutely continuous and the fundamental theorem of calculus for absolutely continuous functions imply that $f ^ { \prime }$ is Lebesgue measurable on I and that

$$
\int _ { I } | f ^ { \prime } ( t ) | \mathrm { d } t < \infty .\tag{4.10}
$$

Moreover, observe that the fact that $f$ is continuous shows that $E$ is compact and Lebesgue measurable. This, $( 4 . 6 )$ , and the fact that $Z$ is Lebesgue measurable demonstrate that $E \backslash Z$ and $E \cap Z$ are Lebesgue measurable and that

$$
0 \leq \mu ( E \cap Z ) \leq \mu ( Z ) = 0 .\tag{4.11}
$$

Combining this and (4.7) ensures that

$$
\int _ { E \setminus Z } | f ^ { \prime } ( t ) | { \mathrm { d } } t = \int _ { E \setminus Z } 0 { \mathrm { d } } t = 0 .\tag{4.12}
$$

In the next step we note that (4.10), (4.11), and the fact that $E \cap Z \subseteq I$ is Lebesgue measurable prove that

$$
\int _ { E \cap Z } | f ^ { \prime } ( t ) | \mathrm { d } t = 0 .\tag{4.13}
$$

This, (4.10), and (4.12) establish that

$$
\int _ { E } | f ^ { \prime } ( t ) | \mathrm { d } t = \int _ { E \setminus Z } | f ^ { \prime } ( t ) | \mathrm { d } t + \int _ { E \cap Z } | f ^ { \prime } ( t ) | \mathrm { d } t = 0 .\tag{4.14}
$$

This proves (4.1). The proof of Lemma 4.1 is thus complete.

## 4.2 Construction and uniform regularity of the truncation of the transformed energy

In the following well-known Lemma 4.2 we recall two elementary properties of $C ^ { 1 }$ -submanifolds, namely, that a relatively open subset of an m-dimensional $C ^ { 1 }$ -submanifold is again an m-dimensional $C ^ { 1 } \cdot$ submanifold and that the Cartesian product of an m-dimensional $C ^ { 1 }$ -submanifold and an n-dimensional C<sup>1</sup>-submanifold is an $( m + n )$ )-dimensional $C ^ { 1 }$ -submanifold. Only for completeness, we include a proof for Lemma 4.2 here.

Lemma 4.2 (Products and subsets open in embedded manifolds). Let $d , \mathfrak { d } \in \mathbb { N } , m , n \in \mathbb { N } _ { 0 }$ , let $X \subseteq \mathbb { R } ^ { d }$ be a m-dimensional $C ^ { 1 }$ -submanifold of $\mathbb { R } ^ { d } .$ , let $Y \subseteq \mathbb { R } ^ { \mathfrak { d } }$ be an n-dimensional $C ^ { 1 }$ -submanifold $o f \mathbb { R } ^ { \mathfrak { d } }$ , and let $U \subseteq X$ be open in X (cf. Definition 2.1). Then

(i) it holds that U is a m-dimensional $C ^ { 1 }$ -submanifold of $\mathbb { R } ^ { d }$ and

(ii) it holds that $X \times Y$ is $a \ ( m + n )$ -dimensional $C ^ { 1 }$ -submanifold of $\mathbb { R } ^ { d + \mathfrak { d } }$

Proof of Lemma $4 . 2 .$ To prove item (i), let $p \in U$ . Observe that the fact that $p \in U \subseteq X$ and the assumption that X is a m-dimensional $C ^ { 1 }$ -submanifold of $\mathbb { R } ^ { d }$ imply that there exist an open $W \subseteq \mathbb { R } ^ { m }$ a set $O \subseteq X$ open in $X$ , and a homeomorphism $\phi \in C ^ { 1 } ( W , O )$ which satisfy that $p \in O$ and that for all $u \in W$ it holds that

$$
\operatorname { r a n k } \phi ^ { \prime } ( u ) = m .\tag{4.15}
$$

In the following let $G = \phi ^ { - 1 } ( U )$ and let $\chi \colon G \to \mathbb { R } ^ { d }$ satisfy $\chi = \phi | _ { G }$ . Note that the fact that U is open in X and the fact that $O \subseteq X$ show that $O \cap U$ is open in O. Combining this, (4.15), the fact that $\phi \colon W \to O$ is a homeomorphism, and the fact that $G = \phi ^ { - 1 } ( U )$ demonstrates that $G = \phi ^ { - 1 } ( O \cap U )$ is open in W. This and the fact that W is open ensure that G is open. Combining this, (4.15), the fact that $\chi = \phi | _ { G }$ , and the fact that ϕ: $W  O$ is a homeomorphism establishes that $\chi \in C ^ { 1 } ( G , O \cap U )$ is a homeomorphism and that for all $u \in G$ it holds that

$$
\operatorname { r a n k } \chi ^ { \prime } ( u ) = \operatorname { r a n k } \phi ^ { \prime } ( u ) = m .\tag{4.16}
$$

Next we combine (4.15), the fact that $p \in O \cap U$ , the fact that $G = \phi ^ { - 1 } ( U )$ , and the fact that $\chi = \phi | _ { G }$ to obtain that there exists $w \in G$ which satisfies that

$$
\chi ( w ) = \phi ( w ) = p .\tag{4.17}
$$

Observe that (4.16), (4.17), the fact that G is open, the fact that $\chi \in C ^ { 1 } ( G , \mathbb { R } ^ { d } )$ is injective, the fact that $\chi ( G ) = O \cap U$ , the fact that $G \ni g \mapsto \chi ( g ) \in O \cap U$ is a homeomorphism, the fact that $p \in O \cap U .$ the fact that $O \cap U$ is open in U, and the fact that $p \in U$ was chosen arbitrarily prove item (i). To prove item (ii), let $x \in X , y \in Y$ . Note that the fact that $x \in X$ and $y \in Y$ , the assumption that X is m-dimensional C<sup>1</sup>-submanifold of $\mathbb { R } ^ { d }$ , and the assumption that Y is n-dimensional $C ^ { 1 }$ -submanifold of R<sup>d</sup> imply that there exist open $A \subseteq \mathbb { R } ^ { m }$ and $B \subseteq \mathbb { R } ^ { n }$ , a set $S \subseteq X$ open in X, a set $T \subseteq Y$ open in $Y$ , and homeomorphisms $\psi \in C ^ { 1 } ( A , S ) , \eta \in C ^ { 1 } ( B , T )$ which satisfy that $x \in S , y \in T$ and that for all $a \in A , b \in B$ it holds that

$$
\operatorname { r a n k } \psi ^ { \prime } ( a ) = m \qquad { \mathrm { a n d } } \qquad \operatorname { r a n k } \eta ^ { \prime } ( b ) = n .\tag{4.18}
$$

In the following let $\Theta \colon A \times B \to \mathbb { R } ^ { d + \mathfrak { o } }$ satisfy for all $a \in A , b \in B$ that

$$
\Theta ( a , b ) = ( \psi ( a ) , \eta ( b ) ) .\tag{4.19}
$$

Observe that (4.19) and the fact that $\psi \colon A \  \ S$ and $\eta \colon B \to T$ are homeomorphisms show that Θ: $A \times B \to S \times T$ is a homeomorphism and that for all $s \in S , t \in T$ it holds that

$$
\Theta ^ { - 1 } ( s , t ) = ( \psi ^ { - 1 } ( s ) , \eta ^ { - 1 } ( t ) ) .\tag{4.20}
$$

Furthermore, note that the fact that A is open, the fact that B is open, the fact that $S$ is open in $X$ the fact that $T$ is open in Y, and the fact that $x \in S$ and $y \in T$ demonstrate that $A \times B$ is open and that $S \times T$ is open in $X \times Y$ and contains $( x , y )$ . This, $( 4 . 1 8 ) , ( 4 . 1 9 )$ , and the fact that $\psi \in C ^ { 1 } ( A , S )$ and $\eta \in C ^ { 1 } ( B , T )$ ensure that $\Theta \in C ^ { 1 } ( A \times B , \mathbb { R } ^ { d + \mathfrak { d } } )$ and that for all $a \in A , b \in B$ it holds that

$$
\Theta ^ { \prime } ( a , b ) = \left( \begin{array} { c c } { { \psi ^ { \prime } ( a ) } } & { { 0 } } \\ { { 0 } } & { { \eta ^ { \prime } ( b ) } } \end{array} \right) \qquad \mathrm { a n d } \qquad \mathrm { r a n k } \Theta ^ { \prime } ( a , b ) = m + n .\tag{4.21}
$$

Combining (4.20), the fact that $A \times B$ is open, the fact that Θ is a homeomorphism, the fact that $( x , y ) \in S \times T$ , the fact that $S \times T$ is open in $X \times Y$ , and the fact that $x \in X$ and $y \in Y$ were chosen arbitrarily hence establishes item (ii). The proof of Lemma 4.2 is thus complete. □

Lemma 4.3. Let $d \in \mathbb { N } , \ \Re , H \in ( 0 , \infty ) , \ \varepsilon \in ( 0 , H / 2 )$ , let $M \subseteq \mathbb { R } ^ { d }$ be a C<sup>1</sup>-submanifold $o f \mathbb { R } ^ { d }$ , let ${ \mathcal { O } } \subseteq ( 0 , \Re ) \times M$ be open in $( 0 , \Re ) \times M$ , let $Z \in C ( \mathcal { O } , [ 0 , H ] )$ , let $\Omega \subseteq { \mathcal { O } }$ satisfy

$$
\Omega = \{ ( \rho , y ) \in { \mathcal { O } } \colon 0 < Z ( \rho , y ) < H \} ,\tag{4.22}
$$

assume $Z | _ { \Omega } \in { \bf C } ^ { 1 } ( \Omega , [ 0 , H ] )$ , let $\omega \in C ( ( 0 , \Re ) \times ( 0 , H ) , [ 0 , 1 ] )$ , m, ${ \mathfrak { b } } \in C ( ( 0 , \mathfrak { R } ) , [ 0 , \infty ) )$ , let $P \colon ( 0 , \Re ) \times$ $[ 0 , H ] \to \mathbb { R }$ satisfy for all $s , t \in ( 0 , \Re ) , z \in [ 0 , H ]$ that

$$
P ( s , z ) = \int _ { ( 0 , z ) } \omega ( s , u ) \mathrm { d } u \qquad a n d \qquad | P ( t , z ) - P ( s , z ) | \leq \int _ { \operatorname* { m i n } \{ s , t \} } ^ { \operatorname* { m a x } \{ s , t \} } \mathfrak { b } ( u ) \mathrm { d } u ,\tag{4.23}
$$

let $J \subseteq \mathbb { R }$ be an interval, let $\varrho \colon J  ( 0 , \Re )$ and $\xi \colon J  M$ be locally absolutely continuous, assume $\begin{array} { r } { \bigcup _ { t \in J } \{ ( \varrho ( t ) , \xi ( t ) ) \} \subseteq \mathcal { O } _ { } } \end{array}$ , let $\alpha , \gamma \colon J \to [ 0 , \infty )$ be Lebesgue measurable, assume that $J \ni t \mapsto \alpha ( t ) \gamma ( t ) \in \mathbb { R }$ is locally integrable, let v : $J  [ 0 , H ]$ satisfy for all $t \in J$ that

$$
v ( t ) = Z ( \varrho ( t ) , \xi ( t ) ) ,\tag{4.24}
$$

assume $f o r$ Lebesgue-almost al $l ~ t \in \{ s \in J \colon 0 < v ( s ) < H \}$ that

$$
\begin{array} { r } { \omega ( \varrho ( t ) , v ( t ) ) v ^ { \prime } ( t ) \leq \operatorname* { m i n } \big \{ 2 \mathbf { m } ( \varrho ( t ) ) | \varrho ^ { \prime } ( t ) | , 5 \mathbf { m } ( \varrho ( t ) ) | \varrho ^ { \prime } ( t ) | - \frac { 1 } { 4 } \alpha ( t ) \gamma ( t ) \big \} , } \end{array}\tag{4.25}
$$

and let $\mathbf { Z } \colon \mathcal { O } \to [ \varepsilon , H - \varepsilon ] , q \colon \mathcal { O } \to \mathbb { R } , \zeta \colon J \to [ \varepsilon , H - \varepsilon ]$ , and $Q \colon J $ R satisfy for all $( \rho , y ) \in \mathcal { O } , t \in J$ that

$$
{ \bf Z } ( \rho , y ) = \operatorname* { m a x } \{ \varepsilon , \operatorname* { m i n } \{ Z ( \rho , y ) , H - \varepsilon \} \} , \qquad q ( \rho , y ) = P ( \rho , { \bf Z } ( \rho , y ) ) ,\tag{4.26}
$$

$$
\zeta ( t ) = \mathbf { Z } ( \varrho ( t ) , \xi ( t ) ) , \qquad a n d \qquad Q ( t ) = q ( \varrho ( t ) , \xi ( t ) )\tag{4.27}
$$

(cf. Definitions 2.2 and $2 . 5 )$ . Then

(i) it holds that Z and q are locally Lipschitz continuous,

(ii) it holds that ζ and Q are locally absolutely continuous,

(iii) it holds for Lebesgue-almost all $t \in J$ that $Q ^ { \prime } ( t ) \leq { \bigl ( } 2 \mathbf { m } ( \varrho ( t ) ) + \mathfrak { b } ( \varrho ( t ) ) { \bigr ) } | \varrho ^ { \prime } ( t ) |$

(iv) it holds for Lebesgue-almost all $t \in \{ s \in J \colon \varepsilon < v ( s ) < H - \varepsilon \}$ that

$$
\begin{array} { r } { Q ^ { \prime } ( t ) \leq \big ( 5 \mathbf { m } ( \varrho ( t ) ) + \mathfrak { b } ( \varrho ( t ) ) \big ) | \varrho ^ { \prime } ( t ) | - \frac { 1 } { 4 } \alpha ( t ) \gamma ( t ) , } \end{array}\tag{4.28}
$$

and

(v) it holds for Lebesgue-almost all $t \in J$ that

$$
\left| Q ^ { \prime } ( t ) - \omega ( \varrho ( t ) , \zeta ( t ) ) \zeta ^ { \prime } ( t ) \right| \leq \mathfrak { h } ( \varrho ( t ) ) | \varrho ^ { \prime } ( t ) | .\tag{4.29}
$$

Proof of Lemma 4.3. Throughout this proof let $\mu$ denote Lebesgue measure on $\mathbb { R } .$ , let $\mathfrak { B } \colon ( 0 , \mathfrak { R } ) \to \mathbb { R }$ satisfy that for all $\rho \in ( 0 , \Re )$ it holds that

$$
{ \mathfrak { B } } ( \rho ) = \int _ { { \mathfrak { R } } / 2 } ^ { \rho } { \mathfrak { b } } ( u ) \mathrm { d } u ,\tag{4.30}
$$

let $\mathcal { A } = \{ t \in J : \varepsilon < v ( t ) < H - \varepsilon \}$ }, and let ${ \mathcal { D } } \subseteq \operatorname { i n t } ( J )$ satisfy

$$
{ \mathcal { D } } = \{ t \in \operatorname { i n t } ( J ) \colon \varrho , \zeta , Q \mathrm { ~ a r e ~ d i f f e r e n t i a b l e ~ a t ~ } t \} .\tag{4.31}
$$

Observe that (4.23), the fact that $0 \leq \omega \leq 1$ , and the triangle inequality prove that for all $\rho \in ( 0 , \Re )$ ${ \displaystyle \mathfrak { z } _ { 1 } , \mathfrak { z } _ { 2 } \in \left[ 0 , H \right] }$ it holds that

$$
| P ( \rho , \mathfrak { z } _ { 2 } ) - P ( \rho , \mathfrak { z } _ { 1 } ) | \leq | \mathfrak { z } _ { 2 } - \mathfrak { z } _ { 1 } | .\tag{4.32}
$$

This, (4.23), and the triangle inequality imply that for all $s , t \in ( 0 , \Re ) , \ : _ { 3 1 } , \ i _ { 2 } \in [ 0 , H ]$ it holds that

$$
\begin{array} { r l } & { | P ( t , \mathfrak { z } _ { 2 } ) - P ( s , \mathfrak { z } _ { 1 } ) | \le | P ( t , \mathfrak { z } _ { 2 } ) - P ( s , \mathfrak { z } _ { 2 } ) | + | P ( s , \mathfrak { z } _ { 2 } ) - P ( s , \mathfrak { z } _ { 1 } ) | } \\ & { \qquad \le \left( \int _ { \operatorname* { m i n } \{ s , t \} } ^ { \operatorname* { m a x } \{ s , t \} } \mathfrak { b } ( w ) \mathrm { d } w \right) + | \mathfrak { z } _ { 2 } - \mathfrak { z } _ { 1 } | . } \end{array}\tag{4.33}
$$

Combining this and the fact that b is continuous shows that for all $\rho _ { 1 } , \rho _ { 2 } \in ( 0 , \Re )$ with $\rho _ { 1 } < \rho _ { 2 }$ there exists $L \in [ 1 , \infty )$ which satisfies for all $s , t \in [ \rho _ { 1 } , \rho _ { 2 } ] , \mathtt { 3 1 } , \mathtt { 3 2 } \in [ 0 , H ]$ that

$$
| P ( t , \mathfrak { z } _ { 2 } ) - P ( s , \mathfrak { z } _ { 1 } ) | \leq L \sqrt { ( t - s ) ^ { 2 } + ( \mathfrak { z } _ { 2 } - \mathfrak { z } _ { 1 } ) ^ { 2 } } .\tag{4.34}
$$

Note that (4.22), the fact that $Z$ is continuous, the fact that $\Omega = Z ^ { - 1 } ( ( 0 , H ) )$ , the fact that $\mathcal { O }$ is open in $( 0 , \Re ) \times M$ , the fact that (0, R) is a 1-dimensional $C ^ { 1 }$ -submanifold of $\mathbb { R } ^ { 1 }$ , the assumption that M is a $C ^ { 1 } .$ -submanifold of $\mathbb { R } ^ { d } .$ and Lemma 4.2 demonstrate that Ω is open in O and is a $C ^ { 1 }$ -submanifold of R<sup>d+1</sup>. Lemma 2.4 therefore ensures that for every $p \in \Omega$ there exist $n \in \{ 0 , 1 , \ldots , d + 1 \}$ , open $A , B \subseteq \mathbb { R } ^ { d + 1 }$ and $C ^ { 1 }$ difeomorphism $\Psi \colon A  B$ such that

$$
p \in A , \qquad 0 \in B , \qquad \Psi ( p ) = 0 , \qquad \mathrm { a n d } \qquad \Psi ( \Omega \cap A ) = B \cap ( \mathbb { R } ^ { n } \times \{ 0 \} ^ { d + 1 - n } ) .\tag{4.35}
$$

We claim that for all $p \in \Omega$ there exist an open $U \subseteq \mathbb { R } ^ { d + 1 }$ and $F \in C ^ { 1 } ( U , \mathbb { R } )$ which satisfy that

$$
p \in U \qquad { \mathrm { a n d } } \qquad F | _ { U \cap \Omega } = Z | _ { U \cap \Omega } .\tag{4.36}
$$

To prove this claim, let $p \in \Omega$ , let $n \in \{ 0 , 1 , \ldots , d + 1 \}$ , let $A \subseteq \mathbb { R } ^ { d + 1 }$ be open, let $B \subseteq \mathbb { R } ^ { d + 1 }$ be open, let $\Psi \colon A  B$ be a $C ^ { 1 }$ difeomorphism, and assume

$$
p \in A , \qquad 0 \in B , \qquad \Psi ( p ) = 0 , \qquad \mathrm { a n d } \qquad \Psi ( \Omega \cap A ) = B \cap ( \mathbb { R } ^ { n } \times \{ 0 \} ^ { d + 1 - n } ) .\tag{4.37}
$$

Observe that the fact that $0 \in B$ and the fact that B is open establish that there exists $\delta \in ( 0 , \infty )$ which satisfies that

$$
( - \delta , \delta ) ^ { d + 1 } \subseteq B .\tag{4.38}
$$

Note that (4.37) and (4.38) prove that for all $a \in ( - \delta , \delta ) ^ { n }$ it holds that

$$
\qquad ( a , 0 ) \in B \cap \left( \mathbb { R } ^ { n } \times \{ 0 \} ^ { d + 1 - n } \right) \qquad { \mathrm { a n d } } \qquad \Psi ^ { - 1 } ( a , 0 ) \in \Omega \cap A .\tag{4.39}
$$

In the following let $U = \Psi ^ { - 1 } ( ( - \delta , \delta ) ^ { d + 1 } )$ , let $\pi \colon \mathbb { R } ^ { d + 1 }  \mathbb { R } ^ { n }$ satisfy for all $a \in \mathbb { R } ^ { n } , b \in \mathbb { R } ^ { d + 1 - n }$ that

$$
\pi ( a , b ) = a ,\tag{4.40}
$$

and let $g \colon ( - \delta , \delta ) ^ { n }  ]$ R satisfy for all $a \in ( - \delta , \delta ) ^ { n }$ that

$$
g ( a ) = Z { \bigl ( } \Psi ^ { - 1 } ( a , 0 ) { \bigr ) } .\tag{4.41}
$$

Observe that (4.37), (4.38), the fact that $U = \Psi ^ { - 1 } ( ( - \delta , \delta ) ^ { d + 1 } )$ , the fact that $\Psi \colon A  B$ is a homeomorphism, and the fact that A is open imply that U is open and that

$$
p \in U \qquad { \mathrm { a n d } } \qquad \pi ( \Psi ( U ) ) = ( - \delta , \delta ) ^ { n } .\tag{4.42}
$$

Next, note that (4.37), (4.38), the fact that $U = \Psi ^ { - 1 } ( ( - \delta , \delta ) ^ { d + 1 } )$ , and the fact that $\Psi \colon A  B$ is bijective show that

$$
\Psi ( \Omega \cap U ) = \Psi ( \Omega \cap A ) \cap ( - \delta , \delta ) ^ { d + 1 } = ( - \delta , \delta ) ^ { n } \times \{ 0 \} ^ { d + 1 - n } .\tag{4.43}
$$

In the following let $\mathbf { T } \colon ( - \delta , \delta ) ^ { n } \to \Omega \cap U$ satisfy that for all $a \in ( - \delta , \delta ) ^ { n }$ it holds that

$$
\mathbf { T } ( a ) = \Psi ^ { - 1 } ( a , 0 ) .\tag{4.44}
$$

Observe that (4.37), (4.40), (4.43), (4.44), the fact that $\Psi \colon A  B$ is a homeomorphism, the fact that $\mathbb { R } ^ { n } \ni a \mapsto ( a , 0 ) \in \mathbb { R } ^ { d + 1 }$ is continuous, and the fact that π is continuous demonstrate that T : $( - \delta , \delta ) ^ { n } $ $\Omega \cap U$ is a homeomorphism and that for all $y \in \Omega \cap U$ it holds that

$$
\mathbf { T } ^ { - 1 } ( y ) = { \boldsymbol { \pi } } ( \Psi ( y ) ) \qquad { \mathrm { a n d } } \qquad \mathbf { T } ( 0 ) = \Psi ^ { - 1 } ( 0 ) = p .\tag{4.45}
$$

In addition, note that (4.38), (4.44), the fact that $\Psi ^ { - 1 } \in C ^ { 1 } ( B , A )$ , the fact that for all $z \in B$ it holds that $( \Psi ^ { - 1 } ) ^ { \prime } ( z )$ is invertible, the fact that $\mathbb { R } ^ { n } \ni a \mapsto ( a , 0 ) \in \mathbb { R } ^ { d + 1 }$ is an injective linear function, and the chain rule ensure that $\mathbf { T } \in C ^ { 1 } ( ( - \delta , \delta ) ^ { n } , \mathbb { R } ^ { d + 1 } )$ and that for all $a \in ( - \delta , \delta ) ^ { n } , h \in \mathbb { R } ^ { n }$ it holds that

$$
\mathbf { T } ^ { \prime } ( a ) h = ( \Psi ^ { - 1 } ) ^ { \prime } ( a , 0 ) ( h , 0 ) \qquad { \mathrm { a n d } } \qquad { \mathrm { r a n k } } \mathbf { T } ^ { \prime } ( a ) = n .\tag{4.46}
$$

This, (4.41), (4.44), the fact that U is open, the fact that $\Omega \cap U$ is open in Ω, the fact that T: $( - \delta , \delta ) ^ { n } $ $\Omega \cap U$ is a homeomorphism, the assumption that $Z | _ { \Omega } \in { \bf C } ^ { 1 } ( \Omega , [ 0 , H ] )$ , and (2.29) establish that

$$
g = ( Z | _ { \Omega } ) \circ \mathbf { T } \in C ^ { 1 } ( ( - \delta , \delta ) ^ { n } , \mathbb { R } ) .\tag{4.47}
$$

Combining (4.42), the fact that U is open, the fact that $\Psi \in C ^ { 1 } ( A , B )$ , and the fact that $\pi$ is linear hence proves that there exists $F \in C ^ { 1 } ( \bar { U } , \mathbb { R } )$ which satisfies for all $y \in U$ that

$$
F ( y ) = g { \bigl ( } \pi ( \Psi ( y ) ) { \bigr ) } .\tag{4.48}
$$

Observe that (4.37), (4.41), (4.48), the fact that $U \subseteq A$ , and the fact that for all $y \in \Omega \cap U$ it holds that $\Psi ( y ) = ( \pi ( \Psi ( y ) ) , 0 )$ imply that for all $y \in \Omega \cap U$ it holds that

$$
F ( y ) = Z { \bigl ( } \Psi ^ { - 1 } ( \pi ( \Psi ( y ) ) , 0 ) { \bigr ) } = Z { \bigl ( } \Psi ^ { - 1 } ( \Psi ( y ) ) { \bigr ) } = Z ( y ) .\tag{4.49}
$$

This, (4.42), the fact that $U$ is open, the fact that $F \in C ^ { 1 } ( U , \mathbb { R } )$ , and the fact that $p \in \Omega$ was chosen arbitrarily establish (4.36).

Note that (4.36) shows that $Z | _ { \Omega }$ is locally Lipschitz continuous. Combining this, (4.26), and the fact that $\mathbb { R } \ni u \mapsto$ max{ε, min $\{ u , H - \varepsilon \} \} \in \mathbb { R }$ is Lipschitz continuous demonstrates that $\mathbf { Z } | _ { \Omega }$ is locally Lipschitz continuous. Moreover, observe that (4.26) and the assumption that $Z$ is continuous ensure that $Z ^ { - 1 } ( ( - \infty , \varepsilon ) )$ and $Z ^ { - 1 } ( ( H - \varepsilon , \infty ) )$ ) are open in O and that for all $( \rho , y ) \in Z ^ { - 1 } ( ( - \infty , \varepsilon ) ) \cup Z ^ { - 1 } ( ( H - \varepsilon , \infty ) )$ it holds that

$$
\mathbf { Z } ( \rho , y ) = \left\{ { \begin{array} { l l } { \varepsilon } & { : Z ( \rho , y ) < \varepsilon , } \\ { H - \varepsilon } & { : Z ( \rho , y ) > H - \varepsilon } \end{array} } . \right.\tag{4.50}
$$

This, the fact that $\mathbf { Z } | _ { \Omega }$ is locally Lipschitz continuous, and the fact that $\Omega \cup Z ^ { - 1 } ( ( - \infty , \varepsilon ) ) \cup Z ^ { - 1 } ( ( H -$ $\varepsilon , \infty ) ) = \mathcal { O }$ prove that Z is locally Lipschitz continuous. Combining this, (4.26), and (4.34) establishes item (i). To prove item (ii), let $I \subseteq J$ be a compact interval and let $K = \{ ( \varrho ( t ) , \xi ( t ) ) \colon t \in I \}$ . Note that the fact that $\varrho \vert _ { I }$ and $\xi | _ { I }$ are continuous and the assumption that $\begin{array} { r } { \bigcup _ { t \in J } \{ ( \varrho ( t ) , \xi ( t ) ) \} \subseteq \mathcal { O } } \end{array}$ imply that K is a compact subset of O. This, item (i), and [36, Proposition A.48] show that $\mathbf { Z } | _ { \mathcal { K } }$ and q|<sub>K</sub> are Lipschitz continuous and that there exists $\Lambda \in ( 0 , \infty )$ which satisfies for all $x , y \in \kappa$ that

$$
| \mathbf { Z } ( x ) - \mathbf { Z } ( y ) | + | q ( x ) - q ( y ) | \leq \Lambda | x - y | .\tag{4.51}
$$

Combining this, (4.27), and the fact that for all $s , t \ \in \ I$ it holds that $| ( \varrho ( t ) , \xi ( t ) ) - ( \varrho ( s ) , \xi ( s ) ) | \ \leq$ $| \varrho ( t ) - \varrho ( s ) | + | \xi ( t ) - \xi ( s ) |$ demonstrates that for all $N \in \mathbb { N } , \ a _ { 1 } , a _ { 2 } , \dotsc , a _ { N } , b _ { 1 } , b _ { 2 } , \dotsc , b _ { N } \ \in \ I$ with $a _ { 1 } < b _ { 1 } \leq a _ { 2 } < b _ { 2 } \leq . . . \leq a _ { N } < b _ { N }$ it holds that

$$
\sum _ { j = 1 } ^ { N } \bigl ( | \zeta ( b _ { j } ) - \zeta ( a _ { j } ) | + | Q ( b _ { j } ) - Q ( a _ { j } ) | \bigr ) \leq \Lambda \sum _ { j = 1 } ^ { N } \bigl ( | \varrho ( b _ { j } ) - \varrho ( a _ { j } ) | + | \xi ( b _ { j } ) - \xi ( a _ { j } ) | \bigr ) .\tag{4.52}
$$

This and the assumption that $\varrho \vert _ { I }$ and $\xi | _ { I }$ are absolutely continuous ensure that $\zeta | _ { I }$ and $Q | _ { I }$ are absolutely continuous. Combining this and the fact that the compact interval $I \subseteq J$ was chosen arbitrarily proves item (ii). Observe that (4.30), the assumption that b is continuous and nonnegative, and the fundamental theorem of calculus establish that $\mathfrak { B } \in C ^ { 1 } ( ( 0 , \mathfrak { R } ) , \mathbb { R } )$ and that for all $\rho , s , t \in ( 0 , \Re )$ it holds that

$$
\mathfrak { B } ^ { \prime } ( \rho ) = \mathfrak { b } ( \rho ) \qquad \mathrm { ~ a n d ~ } \qquad \int _ { \operatorname* { m i n } \{ s , t \} } ^ { \operatorname* { m a x } \{ s , t \} } \mathfrak { b } ( u ) \mathrm { d } u = | \mathfrak { B } ( t ) - \mathfrak { B } ( s ) | .\tag{4.53}
$$

In the next step we combine item (ii), the fact that $\varrho$ is locally absolutely continuous, and the fundamental theorem of calculus for absolutely continuous functions to obtain that for every compact interval $I \subseteq J$ it holds that for Lebesgue-almost all $t \in \operatorname { i n t } ( I )$ it holds that $\varrho , \zeta ,$ and Q are diferentiable at t. This, (4.31), the fact that int(J) is covered by the interiors of countably many compact intervals contained in $^ { J , }$ and the fact that $J \backslash \operatorname { i n t } ( J )$ has at most two points imply that D is Lebesgue measurable and that

$$
\mu ( J \backslash { \cal D } ) = 0 .\tag{4.54}
$$

We claim that for all $t \in \mathcal { D }$ it holds that

$$
\left| Q ^ { \prime } ( t ) - \omega ( \varrho ( t ) , \zeta ( t ) ) \zeta ^ { \prime } ( t ) \right| \leq \mathfrak { h } ( \varrho ( t ) ) | \varrho ^ { \prime } ( t ) | .\tag{4.55}
$$

To prove this claim, let $t \in \mathcal { D }$ . Note that (4.26) and (4.27) show that for all $h \in ( 0 , \infty )$ with $t + h \in J$ it holds that

$$
\begin{array} { r l } & { Q ( t + h ) - Q ( t ) = P ( \varrho ( t + h ) , \zeta ( t + h ) ) - P ( \varrho ( t ) , \zeta ( t + h ) ) } \\ & { \qquad + P ( \varrho ( t ) , \zeta ( t + h ) ) - P ( \varrho ( t ) , \zeta ( t ) ) . } \end{array}\tag{4.56}
$$

Combining this, (4.23), and (4.53) demonstrates that for all $h \in ( 0 , \infty )$ with $t + h \in J$ it holds that

$$
\left| \frac { Q ( t + h ) - Q ( t ) } { h } - \frac { P ( \varrho ( t ) , \zeta ( t + h ) ) - P ( \varrho ( t ) , \zeta ( t ) ) } { h } \right| \leq \frac { | \mathfrak { B } ( \varrho ( t + h ) ) - \mathfrak { B } ( \varrho ( t ) ) | } { h } .\tag{4.57}
$$

Next we combine (4.23), the fact that $\omega$ is continuous, and the fundamental theorem of calculus to obtain that for all $( \rho , z ) \in ( 0 , \Re ) \times ( 0 , H )$ it holds that

$$
\left( \partial _ { z } P \right) ( \rho , z ) = \omega ( \rho , z ) .\tag{4.58}
$$

This, the fact that $t \in \mathcal { D }$ , the fact that $\zeta ( t ) \in [ \varepsilon , H - \varepsilon ] \subseteq ( 0 , H )$ , and the chain rule ensure that

$$
\operatorname* { l i m } _ { h \to 0 } \frac { P ( \varrho ( t ) , \zeta ( t + h ) ) - P ( \varrho ( t ) , \zeta ( t ) ) } { h } = \omega ( \varrho ( t ) , \zeta ( t ) ) \zeta ^ { \prime } ( t ) .\tag{4.59}
$$

Furthermore, observe that (4.53), the fact that $t \in \mathcal { D }$ , the fact that $\varrho$ is continuous, and the chain rule prove that

$$
\operatorname* { l i m } _ { h \searrow 0 } { \frac { | \mathfrak { B } ( \varrho ( t + h ) ) - \mathfrak { B } ( \varrho ( t ) ) | } { h } } = \mathfrak { b } ( \varrho ( t ) ) | \varrho ^ { \prime } ( t ) | .\tag{4.60}
$$

Combining (4.57), (4.59), and the fact that $Q$ is diferentiable at t therefore establishes that

$$
\left| Q ^ { \prime } ( t ) - \omega ( \varrho ( t ) , \zeta ( t ) ) \zeta ^ { \prime } ( t ) \right| \leq \mathfrak { h } ( \varrho ( t ) ) | \varrho ^ { \prime } ( t ) | .\tag{4.61}
$$

This and the fact that $t \in \mathcal { D }$ was chosen arbitrarily prove (4.55).

Note that (4.24), (4.26), (4.27), the fact that $Z , \varrho .$ , and ξ are continuous, and the fact that $\mathcal { A } = \{ t \in$ $J \colon \varepsilon < v ( t ) < H - \varepsilon \}$ imply that A is open in J and that for all $t \in { \mathcal { A } }$ it holds that

$$
\zeta ( t ) = v ( t ) .\tag{4.62}
$$

Combining (4.25) and the fact that $\mathcal { A }$ is open in $J$ hence shows that for Lebesgue-almost all $t \in { \mathcal { A } }$ it holds that

$$
\omega ( \varrho ( t ) , \zeta ( t ) ) \zeta ^ { \prime } ( t ) \leq 2 \mathbf { m } ( \varrho ( t ) ) | \varrho ^ { \prime } ( t ) | \qquad \mathrm { a n d } \qquad \omega ( \varrho ( t ) , \zeta ( t ) ) \zeta ^ { \prime } ( t ) \leq - \frac { \alpha ( t ) \gamma ( t ) } { 4 } + 5 \mathbf { m } ( \varrho ( t ) ) | \varrho ^ { \prime } ( t ) | .\tag{4.63}
$$

This, (4.54), and (4.55) demonstrate that for Lebesgue-almost all $t \in { \mathcal { A } }$ it holds that

$$
\begin{array} { r } { Q ^ { \prime } ( t ) \leq \big ( 2 \mathbf { m } ( \varrho ( t ) ) + \mathfrak { b } ( \varrho ( t ) ) \big ) | \varrho ^ { \prime } ( t ) | \qquad \mathrm { a n d } \qquad Q ^ { \prime } ( t ) \leq - \frac { \alpha ( t ) \gamma ( t ) } { 4 } + \big ( 5 \mathbf { m } ( \varrho ( t ) ) + \mathfrak { b } ( \varrho ( t ) ) \big ) | \varrho ^ { \prime } ( t ) | . } \end{array}\tag{4.64}
$$

Next, observe that (4.24), (4.26), (4.27), the fact that $\mathcal { A } = \left\{ t \in J \colon \varepsilon < v ( t ) < H - \varepsilon \right\}$ , and the fact that $\varepsilon < H - \varepsilon$ ensure that

$$
J \backslash { \cal A } = \{ t \in J \colon \zeta ( t ) = \varepsilon \} \cup \{ t \in J \colon \zeta ( t ) = H - \varepsilon \} .\tag{4.65}
$$

Combining this, item (ii), and Lemma 4.1 establishes that for every compact interval $I \subseteq J$ it holds that

$$
\int _ { \{ t \in I \colon \zeta ( t ) = \varepsilon \} } | \zeta ^ { \prime } ( t ) | \mathrm { d } t = 0 \qquad \mathrm { a n d ~ } \qquad \int _ { \{ t \in I \colon \zeta ( t ) = H - \varepsilon \} } | \zeta ^ { \prime } ( t ) | \mathrm { d } t = 0 .\tag{4.66}
$$

This, (4.65), the fact that $\zeta$ is continuous, the fundamental theorem of calculus for absolutely continuous functions, and the assumption that J is an interval prove that for Lebesgue-almost all $t \in J \backslash A$ it holds that

$$
\zeta ^ { \prime } ( t ) = 0 .\tag{4.67}
$$

Combining this, (4.54), (4.55), and the fact that m is nonnegative implies that for Lebesgue-almost all $t \in J { \backslash } A$ it holds that

$$
\begin{array} { r } { Q ^ { \prime } ( t ) \leq \mathfrak { b } ( \varrho ( t ) ) | \varrho ^ { \prime } ( t ) | \leq \big ( 2 \mathbf { m } ( \varrho ( t ) ) + \mathfrak { b } ( \varrho ( t ) ) \big ) | \varrho ^ { \prime } ( t ) | . } \end{array}\tag{4.68}
$$

This, (4.31), (4.54), (4.64), and the fact that $J = \mathcal { A } \cup ( J \backslash \mathcal { A } )$ establish item (iii). Note that (4.31), (4.54), (4.64), and the fact that $\mathcal { A } = \{ t \in J \colon \varepsilon < v ( t ) < H - \varepsilon \}$ prove item (iv). Observe that (4.31), (4.54), and (4.55) establish item (v). The proof of Lemma 4.3 is thus complete. □

## 5 Lyapunov-type stability for the radial-angular diferential equation (DE) system

In Theorem 6.8 in Section 6 below we establish that, on an arbitrary time interval $I \subseteq \mathbb { R }$ with non-empty interior, the spherical projection of every GF trajectory of a definable C<sup>1</sup>-potential has finite length. One step in our proof of Theorem 6.8 is to perform an appropriate coordinate transform (polar coordinates, at infinity combined with the inversion method $\mathbb { R } ^ { n } \backslash \{ 0 \} \ni x \mapsto | x | ^ { - 2 } x \in \mathbb { R } ^ { n } \backslash \{ 0 \} )$ to the GF DE to obtain an appropriate transformed DE system that consists of a radial component (which is $r ( t ) = | \theta ( t ) - \vartheta | ^ { \nu }$ for $\nu \in \{ - 1 , 1 \} ,$ ) and the angular component (which is the spherical projection) that takes values in the sphere of $\mathbb { R } ^ { n }$ (see (6.86) and (6.87) in the proof of Proposition 6.4 for details).

It is precisely the subject of this section to study the dynamics of such radial-angular DE systems in an abstract form, where a solution of the DE system $( r ( t ) , x ( t ) ) \in \mathcal { O } , t \in I$ , takes values in a definable set ${ \mathcal { O } } \subseteq ( 0 , \Re ) \times M$ which is open in $( 0 , \Re ) \times M$ , where $\Re \in ( 0 , \infty )$ is a suficiently small real number, where M is a compact and definable $\dot { C } ^ { 2 }$ -submanifold of $\mathbb { R } ^ { D }$ , and where $D \in \mathbb { N }$

In particular, in item (ii) in Corollary 5.7 in Subsection 5.5 below we prove that the length of the angular variable $x ( t ) , t \in I ,$ of such radial-angular DE systems $( r ( t ) , x ( t ) ) \in \mathcal { O } , t \in I .$ is finite provided that the radial variable $r ( t )$ converges to zero as t tends to sup(I) and provided that the time interval I agrees with the half-open interval $I = [ \tau , T )$ for some $T \in ( \tau , \infty ] , \tau \in \mathbb { R }$ . Item (ii) in Corollary 5.7 is applied in the proof of Proposition 6.4 to establish finite length of the spherical projection of the GF trajectory and, thereby, to complete the proof of Theorem 6.8. Our proof of Corollary 5.7 is based on applications of Proposition 5.3 in Subsection 5.2 below, Lemma 5.5 in Subsection 5.3 below, and Proposition 5.6 in Subsection 5.4 below.

On a technical level, the most fundamental contribution of this work is Proposition 5.3, which establishes a certain Lyapunov-type stability for solutions $( r ( t ) , x ( t ) ) \in \mathcal { O } , t \in I .$ , of the radial-angular DE systems. Specifically, (5.41) in Proposition 5.3 shows that for all $a , b \in I$ with $a \leq b$ it holds that

$$
Q ( r ( b ) , x ( b ) ) - Q ( r ( a ) , x ( a ) ) \leq - \int _ { a } ^ { b } | x ^ { \prime } ( t ) | \mathrm { d } t + \int _ { a } ^ { b } m ( r ( t ) ) | r ^ { \prime } ( t ) | \mathrm { d } t .\tag{5.1}
$$

The function m in (5.1) is a nonnegative, continuous, integrable, and definable function and the function $Q$ in (5.1) is a nonnegative, continuous, and bounded function. We note that (5.1) shows that $Q$ serves as a Lyapunov-type function for the radial-angular DE system. If $c \in [ 0 , \infty )$ is an upper bound for $Q ,$ then rearranging (5.1) shows that for all $a , b \in I$ with $a \leq b$ it holds that

$$
\int _ { a } ^ { b } | x ^ { \prime } ( t ) | \mathrm { d } t \leq 2 c + \int _ { a } ^ { b } m ( r ( t ) ) | r ^ { \prime } ( t ) | \mathrm { d } t .\tag{5.2}
$$

In the proof of Corollary 5.7 we employ Lemma 5.5 and Proposition 5.6 to show, under suitable assumptions, that the integral on the right hand side of (5.2) can be uniformly bounded. Then, combining this with the consequence of the Lyapunov-type estimate in Proposition 5.3 in (5.2) allows us to establish the finite length estimate $\begin{array} { r } { \int _ { \tau } ^ { T } | x ^ { \prime } ( t ) | \mathrm { d } t < \infty } \end{array}$ in item (ii) in Corollary 5.7.

## 5.1 Regularity properties for one-parameter primitives of definable twoparameter functions

In the next result, Lemma 5.1 below, we establish some basic properties for the one-parameter primitive (see (5.3) below) of a definable and continuously diferentiable two-parameter function. We employ Lemma 5.1 in our proof of the Lyapunov-type estimate in Proposition 5.3 in Subsection 5.2 below. Proposition 5.3, in turn, is used to establish the finite length estimate for the angular component in Corollary 5.7 in Subsection 5.5 below. Corollary 5.7, in turn, is employed in Proposition 6.4 in Section 6 to prove the main result of this work; see Theorem 6.8 in Section 6.

Lemma 5.1. Let R, $H \in ( 0 , \infty )$ , let $\omega \in C ^ { 1 } ( ( 0 , \Re ) \times ( 0 , H ) , [ 0 , 1 ] )$ be definable, and let $P \colon ( 0 , \Re ) \times$ $[ 0 , H ] $ R satisfy for all $r \in ( 0 , \Re ) , z \in [ 0 , H ]$ that

$$
P ( r , z ) = \int _ { 0 } ^ { z } \omega ( r , u ) \mathrm { d } u .\tag{5.3}
$$

Then there exist $\varrho \in ( 0 , \Re )$ and a definable $b \in C ( ( 0 , \varrho ) , [ 0 , \infty ) )$ such that

(i) for all $r \in ( 0 , \Re ) , z \in [ 0 , H ]$ it holds that $P ( r , 0 ) = 0$ and $0 \le P ( r , z ) \le z \le H$

(ii) for all $r \in ( 0 , \Re ) , z _ { 1 } , z _ { 2 } \in [ 0 , H ]$ it holds that $| P ( r , z _ { 1 } ) - P ( r , z _ { 2 } ) | \leq | z _ { 1 } - z _ { 2 } |$

(iii) for all $r \in ( 0 , \Re )$ it holds that $( 0 , H ) \ni z \mapsto P ( r , z ) \in \mathbb { R }$ is diferentiable,

(iv) for all $r \in ( 0 , \Re ) , z \in ( 0 , H )$ it holds that $\partial _ { z } P ( r , z ) = \omega ( r , z )$

(v) for all $s , t \in ( 0 , \varrho ) , z \in [ 0 , H ]$ it holds that

$$
| P ( s , z ) - P ( t , z ) | \leq \left| \int _ { s } ^ { t } b ( r ) \mathrm { d } r \right| \leq \int _ { 0 } ^ { \varrho } b ( r ) \mathrm { d } r < \infty ,\tag{5.4}
$$

and

(vi) for all $\rho _ { 1 } , \rho _ { 2 } \in ( 0 , \varrho )$ with $\rho _ { 1 } ~ < ~ \rho _ { 2 }$ there exists $L \ \in \ \mathbb { R }$ which satisfies for all $s _ { 1 } , s _ { 2 } \in [ \rho _ { 1 } , \rho _ { 2 } ] ;$ $z _ { 1 } , z _ { 2 } \in [ 0 , H ]$ that

$$
| P ( s _ { 1 } , z _ { 1 } ) - P ( s _ { 2 } , z _ { 2 } ) | \leq L { \sqrt { ( s _ { 1 } - s _ { 2 } ) ^ { 2 } + ( z _ { 1 } - z _ { 2 } ) ^ { 2 } } } .\tag{5.5}
$$

Proof of Lemma 5.1. Note that (5.3), the fact that ω is continuous, and the fact that for all $( r , u ) \in$ $( 0 , \mathfrak { R } ) \times ( 0 , H )$ it holds that $0 \leq \omega ( r , u ) \leq 1$ prove item (i). Observe that (5.3), the fact that for all $( r , u ) \in ( 0 , \Re ) \times ( 0 , H )$ it holds that $0 \leq \omega ( r , u ) \leq 1$ , and the triangle inequality show that for all $r \in ( 0 , \Re ) , a , b \in [ 0 , H ]$ with $a < b$ it holds that

$$
| P ( r , a ) - P ( r , b ) | = \left| \int _ { a } ^ { b } \omega ( r , u ) \mathrm { d } u \right| \leq b - a = | a - b | .\tag{5.6}
$$

This establishes item (ii). Note that (5.3), the fact that ω is continuous, and the fundamental theorem of calculus prove item (iii). In addition, observe that item (iii), (5.3), the fact that ω is continuous, and the fundamental theorem of calculus establish item (iv). In the following let $W = ( 0 , \Re ) \times ( 0 , H )$ and let $\sigma \colon ( 0 , \Re ) \to [ 0 , \infty ]$ satisfy for all $r \in ( 0 , \Re )$ that

$$
\sigma ( r ) = \operatorname* { s u p } _ { u \in ( 0 , H ) } \frac { \big | ( \partial _ { r } \omega ) ( r , u ) \big | } { 1 + \big | \left( \partial _ { z } \omega \right) \left( r , u \right) \big | } .\tag{5.7}
$$

Note that (5.7), the fact that W is definable and open, the fact that $\omega \in C ^ { 1 } ( W , [ 0 , 1 ] )$ is definable and bounded, and Lemma 2.12 (applied with $\mathfrak { r } \cap \mathfrak { R } , U \cap W , \Omega \cap W , Z \cap \omega , \eta \cap \sigma$ in the notation of Lemma 2.12) demonstrate that there exists $\varrho \in ( 0 , \Re )$ which satisfies that

(I) it holds that $\sigma ( ( 0 , \varrho ) ) \subseteq \mathbb { R }$ and

(II) it holds that $( 0 , \varrho ) \ni r \mapsto \sigma ( r ) \in \mathbb { R }$ is continuously diferentiable, definable, and integrable.

In the following for every $\delta \in \{ - 1 , 1 \}$ let $S _ { \delta } \subseteq W$ and for every $r \in ( 0 , \Re )$ let $\mathcal { V } _ { \delta , r } \subseteq ( 0 , H )$ satisfy

$$
S _ { \delta } = \big \{ ( r , z ) \in W \colon \delta \big ( \partial _ { z } \omega \big ) \big ( r , z \big ) > 0 \big \} \quad \mathrm { a n d } \qquad \mathcal { V } _ { \delta , r } = \big \{ z \in ( 0 , H ) \colon ( r , z ) \in S _ { \delta } \big \} .\tag{5.8}
$$

Observe that (5.8), the fact that $\omega \in C ^ { 1 } ( W , [ 0 , 1 ] )$ is definable, and Lemma 2.8 (applied with $n \frown 1$ ${ \mathfrak { r } } \longleftrightarrow H , M \frown ( 0 , \Re ) , U \curvearrowright ( 0 , H ) \times ( 0 , \Re ) , f \curvearrowright ( ( 0 , H ) \times ( 0 , \Re ) \ni ( z , r ) \mapsto \omega ( r , z ) \in \mathbb { R } )$ in the notation of Lemma 2.8) ensure that for all $\delta \in \{ - 1 , 1 \}$ it holds that $S _ { \delta }$ is definable and open. Combining this, (5.8), and the fact that for every open and definable $A \subseteq \mathbb { R }$ it holds that A is a finite union of pairwise disjoint open intervals implies that for all $r \in ( 0 , \Re ) , \delta \in \{ - 1 , 1 \}$ there exist $m _ { r , \delta } \in \mathbb { N } _ { 0 }$ and pairwise disjoint nonempty open intervals $I _ { r , \delta , j } \subseteq ( 0 , H ) , j \in \{ 1 , 2 , . . . , m _ { r , \delta } \}$ , which satisfy

$$
\begin{array} { r } { \mathcal { V } _ { \delta , r } = \bigcup _ { j = 1 } ^ { m _ { r , \delta } } I _ { r , \delta , j } . } \end{array}\tag{5.9}
$$

Note that (5.8), (5.9), the fact that for all $\delta \in \{ - 1 , 1 \}$ it holds that $S _ { \delta }$ is definable, and [9, Theorem 3.12] show that there exists $N \in \mathbb N$ which satisfies that for all $r \in ( 0 , \Re )$ it holds that

$$
m _ { r , - 1 } + m _ { r , 1 } \leq N .\tag{5.10}
$$

Observe that (5.8), (5.9), the fact that $\omega \in C ^ { 1 } ( W , [ 0 , 1 ] )$ , and the fundamental theorem of calculus prove that for all $r \in ( 0 , \Re ) , \delta \in \{ - 1 , 1 \} , j \in \{ 1 , \dots , m _ { r , \delta } \} , a , b \in I _ { r , \delta , j }$ with $a < b$ it holds that

$$
\int _ { a } ^ { b } \left| \left( \partial _ { z } \omega \right) \left( r , u \right) \right| \mathrm { d } u = \delta \left( \omega ( r , b ) - \omega ( r , a ) \right) \leq 1 .\tag{5.11}
$$

This, the fact that for all $r \in ( 0 , \Re ) , \delta \in \{ - 1 , 1 \} , j \in \{ 1 , \ldots , m _ { r , \delta } \}$ it holds that $I _ { r , \delta , j }$ is an open interval, and the monotone convergence theorem establish that for all $r \in ( 0 , \Re ) , \delta \in \{ - 1 , 1 \} , j \in \{ 1 , \ldots , m _ { r , \delta } \}$ it holds that

$$
\int _ { I _ { r , \delta , j } } \left. \left( \partial _ { z } \omega \right) \left( r , u \right) \right. \mathrm { d } u \leq 1 .\tag{5.12}
$$

Combining this, (5.8), (5.9), (5.10), and the fact that for all $r \in ( 0 , \Re ) , u \in ( 0 , H ) \backslash ( \mathcal { V } _ { - 1 , r } \cup \mathcal { V } _ { 1 , r } )$ it holds that $\left( \partial _ { z } \omega \right) \left( r , u \right) = 0$ demonstrates that for all $r \in ( 0 , \Re )$ it holds that

$$
\int _ { ( 0 , H ) } \left| \left( \partial _ { z } \omega \right) \left( r , u \right) \right| \mathrm { d } u \leq m _ { r , - 1 } + m _ { r , 1 } \leq N .\tag{5.13}
$$

Moreover, note that (5.7) and item (I) ensure that for all $( r , u ) \in ( 0 , \varrho ) \times ( 0 , H )$ it holds that

$$
\left| ( \partial _ { r } \omega ) ( r , u ) \right| \leq \sigma ( r ) \big ( 1 + \big | \big ( \partial _ { z } \omega \big ) ( r , u ) \big | \big ) .\tag{5.14}
$$

This, (5.13), and the fact that σ is nonnegative imply that for all $r \in ( 0 , \varrho )$ it holds that

$$
\int _ { ( 0 , H ) } \left| ( \partial _ { r } \omega ) ( r , u ) \right| \mathrm { d } u \leq ( H + N ) \sigma ( r ) .\tag{5.15}
$$

In the following let $b \colon ( 0 , \varrho )  [ 0 , \infty )$ satisfy for all $r \in ( 0 , \varrho )$ that

$$
b ( r ) = ( H + N ) \sigma ( r ) .\tag{5.16}
$$

Observe that item (II), (5.16), and the fact that $H + N > 0$ show that b is definable and continuous and satisfies

$$
\int _ { 0 } ^ { \varrho } b ( r ) \mathrm { d } r = ( H + N ) \int _ { 0 } ^ { \varrho } \sigma ( r ) \mathrm { d } r < \infty .\tag{5.17}
$$

In the next step we note that the fact that $\omega \in C ^ { 1 } ( W , [ 0 , 1 ] )$ and the fundamental theorem of calculus prove that for all $s , t \in ( 0 , \varrho ) , u \in ( 0 , H )$ with $s < t$ it holds that

$$
| \omega ( s , u ) - \omega ( t , u ) | = \left| \int _ { s } ^ { t } ( \partial _ { r } \omega ) ( r , u ) \mathrm { d } r \right| \leq \int _ { s } ^ { t } | ( \partial _ { r } \omega ) ( r , u ) | \mathrm { d } r .\tag{5.18}
$$

Combining (5.3), (5.15), (5.16), and Tonelli’s theorem therefore establishes that for all $s , t \in ( 0 , \varrho )$ $z \in [ 0 , H ]$ with $s < t$ it holds that

$$
\begin{array} { r l r } {  { | P ( s , z ) - P ( t , z ) | \leq \int _ { 0 } ^ { z } | \omega ( s , u ) - \omega ( t , u ) | \mathrm { d } u \leq \int _ { 0 } ^ { z } \int _ { s } ^ { t } | ( \partial _ { r } \omega ) ( r , u ) | \mathrm { d } r \mathrm { d } u } } \\ & { } & { = \int _ { s } ^ { t } \int _ { 0 } ^ { z } | ( \partial _ { r } \omega ) ( r , u ) | \mathrm { d } u \mathrm { d } r \leq \int _ { s } ^ { t } b ( r ) \mathrm { d } r . ~ } \end{array}\tag{5.19}
$$

This and (5.17) prove item (v). Observe that item (II), (5.16), and the extreme value theorem demonstrate that for all $\rho _ { 1 } , \rho _ { 2 } \in ( 0 , \varrho )$ with $\rho _ { 1 } < \rho _ { 2 }$ there exists $K _ { \rho _ { 1 } , \rho _ { 2 } } \in [ 0 , \infty )$ which satisfies for all $r \in [ \rho _ { 1 } , \rho _ { 2 } ]$ that

$$
b ( r ) \leq K _ { \rho _ { 1 } , \rho _ { 2 } } .\tag{5.20}
$$

Combining this, item (ii), item (v), the triangle inequality, and the Cauchy–Schwarz inequality ensures that for all $\rho _ { 1 } , \rho _ { 2 } \in ( 0 , \varrho ) , ( \mathfrak { s } _ { 1 } , \mathfrak { z } _ { 1 } ) , ( \mathfrak { s } _ { 2 } , \mathfrak { z } _ { 2 } ) \in [ \rho _ { 1 } , \rho _ { 2 } ] \times [ 0 , H ]$ with $\rho _ { 1 } < \rho _ { 2 }$ it holds that

$$
\begin{array} { r l } & { | P ( \mathfrak { s } _ { 1 } , \mathfrak { z } _ { 1 } ) - P ( \mathfrak { s } _ { 2 } , \mathfrak { z } _ { 2 } ) | \leq | P ( \mathfrak { s } _ { 1 } , \mathfrak { z } _ { 2 } ) - P ( \mathfrak { s } _ { 2 } , \mathfrak { z } _ { 2 } ) | + | P ( \mathfrak { s } _ { 1 } , \mathfrak { z } _ { 1 } ) - P ( \mathfrak { s } _ { 1 } , \mathfrak { z } _ { 2 } ) | } \\ & { \qquad \leq K _ { \rho _ { 1 } , \rho _ { 2 } } | \mathfrak { s } _ { 1 } - \mathfrak { s } _ { 2 } | + | \mathfrak { z } _ { 1 } - \mathfrak { z } _ { 2 } | } \\ & { \qquad \leq \sqrt { ( K _ { \rho _ { 1 } , \rho _ { 2 } } ) ^ { 2 } + 1 } \sqrt { ( \mathfrak { s } _ { 1 } - \mathfrak { s } _ { 2 } ) ^ { 2 } + ( \mathfrak { z } _ { 1 } - \mathfrak { z } _ { 2 } ) ^ { 2 } } } \end{array} .\tag{5.21}
$$

This establishes item (vi). The proof of Lemma 5.1 is thus complete.

## 5.2 Lyapunov-type stability for the radial-angular dynamics

In (5.41) in Proposition 5.3 below we establish a crucial Lyapunov-type estimate for the radial-angular DE system. The function $Q \colon \mathcal { O } \cap ( ( 0 , \eta ) \times M ) \to [ 0 , \infty )$ in Proposition 5.3 serves as the Lyapunov-type function for the radial-angular DE system in (5.40). In our proof of Proposition 5.3 we employ the elementary estimates for real numbers in the following Lemma 5.2.

Lemma 5.2. Let $r , T \in ( 0 , \infty ) , g \in [ 0 , \infty ) , n \in [ 1 , \infty ) , v , c , D , B \in \mathbb { R }$ satisfy

$$
\begin{array} { r } { | v - c | \leq n ( T + g ) , \quad r ^ { 2 } n ^ { 2 } \leq \frac { 1 } { 6 4 } , \quad D = T + 8 r ^ { 2 } n ( c ^ { 2 } + T ^ { 2 } ) ^ { 1 / 2 } , \quad a n d \quad \mathcal { B } = - D ^ { - 1 } ( g ^ { 2 } + r ^ { 2 } v ( v - c ) ) . } \end{array}\tag{5.22}
$$

Then the following two statements hold:

(i) It holds that $B \leq 2 n r ^ { 2 } | \boldsymbol { v } |$

(ii) If $g \geq T$ , then $\begin{array} { r } { B \leq 5 n r ^ { 2 } | v | - \frac { g } { 4 } } \end{array}$

Proof of Lemma 5.2. Throughout this proof, let $Y , P \in \mathbb { R }$ satisfy

$$
Y = r ^ { 2 } n | c | \qquad { \mathrm { a n d } } \qquad P = g ^ { 2 } + r ^ { 2 } v ( v - c ) .\tag{5.23}
$$

Note that (5.22) and the fact that $r > 0$ and $n \geq 1$ imply that

$$
0 < r ^ { 2 } n \leq r ^ { 2 } n ^ { 2 } \leq { \frac { 1 } { 6 4 } } .\tag{5.24}
$$

Combining this, (5.22), (5.23), the fact that $T > 0$ , and the fact that $| c | \leq \sqrt { c ^ { 2 } + T ^ { 2 } } \leq | c | + T$ shows that

$$
D \geq T > 0 , \qquad D \geq 8 Y ,\tag{5.25}
$$

and

$$
D \leq T + 8 r ^ { 2 } n ( \vert c \vert + T ) = ( 1 + 8 r ^ { 2 } n ) T + 8 Y \leq \frac { 9 } { 8 } T + 8 Y .\tag{5.26}
$$

This, (5.22), the fact that $g ^ { 2 } \geq 0$ and $r ^ { 2 } > 0$ , and the fact that $- v ( v - c ) \leq | v | | v - c |$ prove that

$$
\mathcal { B } = - \frac { g ^ { 2 } } { D } - \frac { r ^ { 2 } v ( v - c ) } { D } \leq \frac { r ^ { 2 } | v | | v - c | } { D } .\tag{5.27}
$$

Next we combine (5.22) and the fact that $n \geq 1$ to obtain that if $g < T$ , then it holds that

$$
| v - c | \leq n ( T + g ) \leq 2 n T .\tag{5.28}
$$

Combining (5.25), (5.27), and the fact that $n r ^ { 2 } | v | \geq 0$ hence demonstrates that if $g < T$ , then it holds that

$$
B \leq \frac { r ^ { 2 } | v | | v - c | } { D } \leq 2 n r ^ { 2 } | v | \frac { T } { D } \leq 2 n r ^ { 2 } | v | .\tag{5.29}
$$

Furthermore, observe that (5.22) and the fact that $T > 0$ and $n \geq 1$ ensure that if $g \geq T$ , then it holds that

$$
g \geq T > 0 \qquad \mathrm { a n d } \qquad | v - c | \leq n ( T + g ) \leq 2 n g .\tag{5.30}
$$

This, (5.23), and the fact that $c ( v - c ) \geq - | c | | v - c |$ establish that if $g \geq T ,$ , then it holds that

$$
P = g ^ { 2 } + r ^ { 2 } ( v - c ) ^ { 2 } + r ^ { 2 } c ( v - c ) \geq g ^ { 2 } - r ^ { 2 } | c | | v - c | \geq g ^ { 2 } - 2 Y g .\tag{5.31}
$$

Combining (5.25), (5.26), and the fact that $T > 0$ therefore implies that if $g \geq T$ and $g \ge 8 Y$ , then it holds that

$$
P \geq g ^ { 2 } - 2 Y g \geq \frac 3 4 g ^ { 2 } > 0 \qquad \mathrm { a n d } \qquad 0 < D \leq \frac 9 8 T + 8 Y \leq \frac { 1 7 } { 8 } g .\tag{5.32}
$$

This, (5.22), and (5.23) show that if $g \geq T$ and $g \geq 8 Y$ , then it holds that

$$
B = - \frac { P } { D } \leq - \frac { 3 g ^ { 2 } } { 4 D } \leq - \frac { 6 } { 1 7 } g \leq - \frac { g } { 4 } \leq 0 .\tag{5.33}
$$

Next, note that (5.25), (5.27), and (5.30) prove that if $g \geq T$ and $g < 8 Y$ , then it holds that

$$
D \geq 8 Y > g \geq T > 0 \qquad \mathrm { a n d } \qquad B \leq \frac { r ^ { 2 } | v | | v - c | } { D } \leq 2 n r ^ { 2 } | v | \frac { g } { D } \leq 2 n r ^ { 2 } | v | .\tag{5.34}
$$

Combining this, (5.29), and (5.33) establishes item (i). Observe that (5.22), (5.23), (5.30), and the fact that $n \geq 1$ demonstrate that if $g \geq T$ and $g < 8 Y$ , then it holds that

$$
| v - c | \leq 2 n g < 1 6 n Y = 1 6 r ^ { 2 } n ^ { 2 } | c | \leq { \frac { | c | } { 4 } } .\tag{5.35}
$$

This and the triangle inequality ensure that if $g \geq T$ and $g < 8 Y$ , then it holds that

$$
| v | \geq | c | - | v - c | \geq { \frac { 3 } { 4 } } | c | .\tag{5.36}
$$

Combining (5.23) and the fact that $n r ^ { 2 } > 0$ hence implies that if $g \geq T$ and $g < 8 Y$ , then it holds that

$$
\frac { g } { 4 } < 2 Y = 2 n r ^ { 2 } | c | \leq \frac { 8 } { 3 } n r ^ { 2 } | v | .\tag{5.37}
$$

This and item (i) show that if $g \geq T$ and $g < 8 Y$ , then it holds that

$$
\mathcal { B } \leq 2 n r ^ { 2 } | v | = - \frac { g } { 4 } + 2 n r ^ { 2 } | v | + \frac { g } { 4 } \leq - \frac { g } { 4 } + \frac { 1 4 } { 3 } n r ^ { 2 } | v | \leq - \frac { g } { 4 } + 5 n r ^ { 2 } | v | .\tag{5.38}
$$

Combining this, (5.33), and the fact that $5 n r ^ { 2 } | v | \geq 0$ proves item (ii). The proof of Lemma 5.2 is thus complete. □

Proposition 5.3 (Lyapunov-type estimate). Let $D \in \mathbb { N } , \ \Re , { \mathsf { a } } \in ( 0 , \infty ) , \ { \mathsf { b } } \in [ { \mathsf { a } } , \infty )$ , let $M \subseteq \mathbb { R } ^ { D }$ be a nonempty, compact, and definable $C ^ { 2 }$ -submanifold $o f \mathbb { R } ^ { D }$ , let ${ \mathcal { O } } \subseteq ( 0 , \Re ) \times M$ be definable and open in $( 0 , \Re ) \times M$ , and let $h = ( h ( r , x ) ) _ { ( r , x ) \in \mathcal { O } } \in \mathbf { C } ^ { 1 } ( \mathcal { O } , \mathbb { R } )$ be definable (cf. Definitions 1.5, 1.6, 2.2, and $2 . 5 )$ Then there exist $\eta \in ( 0 , \Re )$ , a bounded $Q \in C ( \mathcal { O } \cap ( ( 0 , \eta ) \times M ) , [ 0 , \infty ) )$ , and a definable and integrable m $\cdot \in C ( ( 0 , \eta ) , [ 0 , \infty ) )$ ) such that for every interval $I \subseteq \mathbb { R }$ , all locally absolutely continuous r : $I \to ( 0 , \eta )$ and $x \colon I \to M$ , and all Lebesgue measurable $\alpha \colon I  [ 0 , \infty )$ and $A \colon \mathsf { \bar { I } } \to \mathbb { R } ^ { D \times \bar { D } }$ with $\textstyle \bigcup _ { t \in I } \{ ( r ( t ) , x ( t ) ) \} \subseteq \mathcal { O }$ and with the property that for Lebesgue-almost every $t \in I$ and every $v \in T _ { x ( t ) } M$ it holds that

$$
A ( t ) v \in T _ { x ( t ) } M , \qquad \langle A ( t ) v , v \rangle \geq \mathsf { a } | v | ^ { 2 } , \qquad | A ( t ) v | \leq \mathsf { b } | v | ,\tag{5.39}
$$

$$
\begin{array} { r } { r ^ { \prime } ( t ) = - \alpha ( t ) ( r ( t ) ) ^ { 2 } ( \partial _ { r } h ) ( r ( t ) , x ( t ) ) , \qquad a n d \qquad x ^ { \prime } ( t ) = - \alpha ( t ) A ( t ) ( \nabla _ { M } h ) ( r ( t ) , x ( t ) ) } \end{array}\tag{5.40}
$$

we have that for all $a , b \in I$ with $a \leq b$ it holds that

$$
Q ( r ( b ) , x ( b ) ) - Q ( r ( a ) , x ( a ) ) \leq - \int _ { a } ^ { b } | x ^ { \prime } ( t ) | \mathrm { d } t + \int _ { a } ^ { b } m ( r ( t ) ) | r ^ { \prime } ( t ) | \mathrm { d } t\tag{5.41}
$$

(cf. Definitions 2.3 and 2.7).

Proof of Proposition 5.3. Throughout this proof let $\varphi \colon \mathbb { R }  ( - 1 , 1 )$ and h: $\mathcal { O }  ( - 1 , 1 )$ satisfy for all $e \in \mathbb { R }$ that $\varphi ( e ) = e / \sqrt { 1 + e ^ { 2 } }$ and $\mathbf { h } = \varphi \circ h$ . Note that the fact that $h \in { \bf C } ^ { 1 } ( { \mathcal { O } } , \mathbb { R } )$ is definable and the chain rule establish that $\varphi \in C ^ { 1 } ( \mathbb { R } , ( - 1 , 1 ) )$ and h $\in { \bf C } ^ { 1 } ( { \mathcal { O } } , ( - 1 , 1 ) )$ are definable, that for all $e \in \mathbb { R } { \mathrm { ~ i t ~ } }$ holds that

$$
\varphi ^ { \prime } ( e ) = ( 1 + e ^ { 2 } ) ^ { - 3 / 2 } > 0 ,\tag{5.42}
$$

and that for all $( s , y ) \in \mathcal { O }$ it holds that

$$
( \partial _ { r } \mathbf { h } ) ( s , y ) = \varphi ^ { \prime } ( h ( s , y ) ) ( \partial _ { r } h ) ( s , y ) \qquad { \mathrm { a n d } } \qquad ( \nabla _ { M } \mathbf { h } ) ( s , y ) = \varphi ^ { \prime } ( h ( s , y ) ) ( \nabla _ { M } h ) ( s , y ) .\tag{5.43}
$$

In addition, observe that the fact that for all $e \in \mathbb { R }$ it holds that $\varphi ( e ) = e / \sqrt { 1 + e ^ { 2 } }$ and the fact that for all $( s , y ) \in \mathcal { O }$ it holds that $\mathbf { h } ( s , y ) = \varphi ( h ( s , y ) )$ demonstrate that for all $( s , y ) \in \mathcal { O }$ it holds that

$$
- 1 < \mathbf { h } ( s , y ) < 1 .\tag{5.44}
$$

In the following let ${ \mathsf { c } } , \Lambda , \kappa \in ( 0 , \infty )$ satisfy

$$
\mathsf { c } = \sqrt { \mathsf { a } } , \qquad \Lambda = \frac { \mathsf { b } } { \mathsf { c } } , \qquad \kappa = 4 \Lambda ,\tag{5.45}
$$

and let $G \colon { \mathcal { O } } \to [ 0 , \infty )$ satisfy for all $( s , y ) \in \mathcal { O }$ that

$$
G ( s , y ) = | ( \nabla _ { M } \mathbf { h } ) ( s , y ) | .\tag{5.46}
$$

(cf. Definition 2.7). Note that (5.44), (5.45), (5.46), and Lemma 3.5 (applied with $R \frown \Re , \Omega \frown \mathcal { O }$ $h \curvearrowleft \mathbf { h } , a \curvearrowleft ( ( 0 , \Re ) \ni s \mapsto - 1 ) , b \curvearrowleft ( ( 0 , \Re ) \ni s \mapsto 1 )$ , c $\yen 4,4$ in the notation of Lemma 3.5) ensure that there exist $\theta \in ( 0 , \Re ) , N \in \mathbb { N } , H \in ( 0 , \infty )$ and definable $\mathfrak { e } _ { 0 } , \mathfrak { e } _ { 1 } , \dots , \mathfrak { e } _ { N } \in C ^ { 2 } ( ( 0 , \theta ) , \mathbb { R } ) , \mathcal { F } _ { 0 } , \mathcal { F } _ { 1 } , \dots , \mathcal { F } _ { N } \in$ $C ( ( 0 , \theta ) \times \mathbb { R } , [ 0 , H ] ) , E _ { 1 } , E _ { 2 } , \ldots , E _ { N } \in C ^ { 2 } \bigl ( ( 0 , \theta ) \times ( 0 , H ) , \mathbb { R } \bigr ) , T _ { 1 } , T _ { 2 } , \ldots , T _ { N } \in C ^ { 1 } \bigl ( ( 0 , \theta ) \times ( 0 , H ) , ( 0 , \infty ) \bigr )$ such that

(I) for all $s \in ( 0 , \theta )$ it holds that

$$
- 1 = \mathfrak { e } _ { 0 } ( s ) < \cdots < \mathfrak { e } _ { N } ( s ) = 1 ,\tag{5.47}
$$

(II) for all $s \in ( 0 , \theta ) , e \in \mathbb { R } \mathrm { ~ w i t h ~ i n f } \bigl ( \{ \infty \} \cup \{ z \in \mathbb { R } \colon \exists y \in M \colon ( s , y ) \in \mathcal { O } \cap \mathbf { h } ^ { - 1 } ( \{ e \} ) \cap G ^ { - 1 } ( \{ z \} ) \} \bigr ) = 0$ it holds that $e \in \{ \mathfrak { e } _ { 0 } ( s ) , \mathfrak { e } _ { 1 } ( s ) , \dots , \mathfrak { e } _ { N } ( s ) \}$

(III) for all $i \in \{ 0 , 1 , \ldots , N - 1 \} , s \in ( 0 , \theta ) , u \in ( - \infty , \mathfrak { e } _ { i } ( s ) ] , v \in [ \mathfrak { e } _ { i + 1 } ( s ) , \infty ) , z , w \in ( \mathfrak { e } _ { i } ( s ) , \mathfrak { e } _ { i + 1 } ( s ) )$ with $z < w$ it holds that

$$
0 = \mathcal { F } _ { i } ( s , u ) < \mathcal { F } _ { i } ( s , z ) < \mathcal { F } _ { i } ( s , w ) < \mathcal { F } _ { i } ( s , v ) = H ,\tag{5.48}
$$

(IV) for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that $\mathcal { F } _ { i } \big | _ { \{ ( s , u ) \in ( 0 , \theta ) \times \mathbb { R } : ~ \mathfrak { e } _ { i } ( s ) < u < \mathfrak { e } _ { i + 1 } ( s ) \} }$ is twice continuously diferentiable,

(V) for all $i \in \{ 1 , 2 , \ldots , N \} , s \in ( 0 , \theta ) , z \in ( 0 , H )$ it holds that

$$
\mathfrak { c } _ { i - 1 } ( s ) < E _ { i } ( s , z ) < \mathfrak { e } _ { i } ( s ) , \qquad \mathcal { F } _ { i - 1 } ( s , E _ { i } ( s , z ) ) = z , \qquad \mathrm { a n d } \qquad T _ { i } ( s , z ) = \partial _ { z } E _ { i } ( s , z ) > 0 ,\tag{5.49}
$$

and

(VI) for all $i \in \{ 1 , 2 , . . . , N \} , ( s , y ) \in \mathcal { O } \cap \left( \left( 0 , \theta \right) \times M \right)$ with $\mathfrak { e } _ { i - 1 } ( s ) < \mathbf { h } ( s , y ) < \mathfrak { e } _ { i } ( s )$ it holds that

$$
T _ { i } ( s , { \mathcal { F } } _ { i - 1 } ( s , \mathbf { h } ( s , y ) ) ) \leq { \tt c } G ( s , y ) .\tag{5.50}
$$

In the following for every $i \in \{ 0 , 1 , \ldots , N - 1 \}$ let $Z _ { i } \colon \mathcal { O } \cap ( ( 0 , \theta ) \times M ) \to [ 0 , H ] , S _ { i } \colon ( 0 , \theta ) \times ( 0 , H ) \to \mathbb { R }$ and $\Omega _ { i } \subseteq \mathcal { O } \cap ( ( 0 , \theta ) \times M )$ satisfy that for all $( s , y ) \in \mathcal { O } \cap ( ( 0 , \theta ) \times M )$ it holds that

$$
Z _ { i } ( s , y ) = \mathcal { F } _ { i } ( s , \mathbf { h } ( s , y ) ) ,\tag{5.51}
$$

that for all $( s , z ) \in ( 0 , \theta ) \times ( 0 , H )$ it holds that

$$
S _ { i } ( s , z ) = ( \partial _ { s } E _ { i + 1 } ) ( s , z ) ,\tag{5.52}
$$

and that

$$
\begin{array} { r } { \Omega _ { i } = \{ ( s , y ) \in \mathcal { O } \cap ( ( 0 , \theta ) \times M ) \colon \mathfrak { e } _ { i } ( s ) < \mathbf { h } ( s , y ) < \mathfrak { e } _ { i + 1 } ( s ) \} . } \end{array}\tag{5.53}
$$

Observe that (5.51), (5.52), the fact that for al $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that h and $\mathcal { F } _ { i }$ are continuous and definable, and the fact that $( \mathcal { O } \cap ( ( 0 , \theta ) \times M ) ) \ni ( s , y ) \mapsto s \in ( 0 , \theta )$ is continuous and definable imply that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that $Z _ { i }$ is continuous and definable. Moreover, note that (5.48), (5.51), (5.52), and (5.53) show that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that

$$
\Omega _ { i } = \{ ( s , y ) \in \mathcal { O } \cap ( ( 0 , \theta ) \times M ) \colon 0 < Z _ { i } ( s , y ) < H \} .\tag{5.54}
$$

This, the fact that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that $Z _ { i }$ is continuous and definable, the fact that M is a C<sup>2</sup>-submanifold of $\mathbb { R } ^ { D }$ , the fact that O is definable and open in $( 0 , \Re ) \times M$ , and Lemma 4.2 prove that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ } it holds that $\Omega _ { i }$ is definable, open in $( 0 , \theta ) \times M$ , and a $C ^ { 1 }$ -submanifold of $\mathbb { R } ^ { D + 1 }$ . We claim that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that

$$
Z _ { i } | _ { \Omega _ { i } } \in { \bf C } ^ { 1 } ( \Omega _ { i } , [ 0 , H ] ) .\tag{5.55}
$$

To prove this claim, let $i \in \{ 0 , 1 , \ldots , N - 1 \}$ , let $n \in \{ 0 , 1 , \ldots , D \}$ satisfy that M is a n-dimensional C<sup>2</sup>-submanifold of $\mathbb { R } ^ { D }$ , let $\bar { W } \subseteq \mathbb { R } ^ { n + 1 }$ be open, and let $\boldsymbol \phi = ( \phi _ { 0 } , \dots , \phi _ { D } ) \in C ^ { 1 } ( W , \Omega _ { i } )$ satisfy for all $w \in W$ that rank $\phi ^ { \prime } ( w ) = n + 1$ . Observe that (5.53), the fact that M is a n-dimensional $C ^ { 2 } .$ submanifold of $\mathbb { R } ^ { D }$ , the fact that $\phi \in C ^ { 1 } ( W , \Omega _ { i } )$ , the fact that $\Omega _ { i } \subseteq \mathcal { O }$ , the fact that for all $w \in W$ it holds that rank $\phi ^ { \prime } ( w ) = n + 1$ , and the fact that $ { \mathbf { h } } \in  { \mathbf { C } } ^ { 1 } (  { \mathcal { O } } , ( - 1 , 1 ) )$ establish that $\phi _ { 0 } \in C ^ { 1 } ( W , \mathbb { R } )$ and h $\circ \phi \in C ^ { 1 } ( W , ( - 1 , 1 ) )$ and that for all $w \in W$ it holds that

$$
\mathfrak { e } _ { i } \bigl ( \phi _ { 0 } ( w ) \bigr ) < \mathbf { h } ( \phi ( w ) ) < \mathfrak { e } _ { i + 1 } \bigl ( \phi _ { 0 } ( w ) \bigr ) .\tag{5.56}
$$

Combining this, (5.51), (5.52), the fact that $\phi _ { 0 } \in C ^ { 1 } ( W , \mathbb { R } )$ and h $\circ \phi \in C ^ { 1 } ( W , ( - 1 , 1 ) )$ ), the fact that $\mathcal { F } _ { i } \big | _ { \{ ( s , u ) \in ( 0 , \theta ) \times  { { \mathbb R } } : \textup { \texttt { e } } _ { i } ( s ) < u < \mathsf { e } _ { i + 1 } ( s ) \} } \in C ^ { 2 } ( \{ ( s , u ) \in ( 0 , \theta ) \times  { { \mathbb R } } : \mathsf { e } _ { i } ( s ) < u < \mathsf { e } _ { i + 1 } ( s ) \} , [ 0 , H ] )$ , and the chain rule demonstrates that $Z _ { i } \circ \phi \colon W \to [ 0 , H ]$ is diferentiable and that for all w $\in W$ it holds that

$$
\begin{array} { r } { \left( Z _ { i } \circ \phi \right) ^ { \prime } ( w ) = \left( \partial _ { s } \mathcal { F } _ { i } \right) \left( \phi _ { 0 } ( w ) , \mathbf { h } ( \phi ( w ) ) \right) ( \phi _ { 0 } ) ^ { \prime } ( w ) + \left( \partial _ { u } \mathcal { F } _ { i } \right) \left( \phi _ { 0 } ( w ) , \mathbf { h } ( \phi ( w ) ) \right) ( \mathbf { h } \circ \phi ) ^ { \prime } ( w ) . } \end{array}\tag{5.57}
$$

This, (5.56), the fact that $\phi _ { 0 } .$ , h $\circ \phi , ( \phi _ { 0 } ) ^ { \prime }$ , and $( { \bf h } \circ \phi ) ^ { \prime }$ are continuous on $W ,$ and the fact that $\partial _ { s } \mathcal { F } _ { i }$ and $\partial _ { u } \mathcal { F } _ { i }$ are continuous on $\{ ( s , u ) \in ( 0 , \theta ) \times \mathbb { R } \colon \mathfrak { e } _ { i } ( s ) < u < \mathfrak { e } _ { i + 1 } ( s ) \}$ ensure that $( Z _ { i } \circ \phi ) ^ { \prime }$ is continuous on W. Combining this and the fact that $Z _ { i } \circ \phi$ is diferentiable on W implies that $Z _ { i } \circ \phi \in C ^ { 1 } ( W , [ 0 , H ] )$ This and the arbitrariness of $i , W$ , and ϕ show (5.55).

Note that (5.48), (5.51), (5.52), and (5.53) prove that for all $i \in \{ 0 , 1 , \ldots , N - 1 \} , ( s , y ) \in \mathcal { O } \cap ( ( 0 , \theta ) \times$ M) it holds that

$$
Z _ { i } ( s , y ) = \left\{ \begin{array} { l l } { 0 } & { : \mathbf { h } ( s , y ) \leq \mathfrak { e } _ { i } ( s ) , } \\ { H } & { : \mathbf { h } ( s , y ) \geq \mathfrak { e } _ { i + 1 } ( s ) , } \\ { \mathcal { F } _ { i } ( s , \mathbf { h } ( s , y ) ) } & { : ( s , y ) \in \Omega _ { i } . } \end{array} \right.\tag{5.58}
$$

In the next step we combine (5.48), (5.49), (5.50), (5.51), (5.52), (5.53), and (5.54) to obtain that for all $i \in \{ 0 , 1 , \ldots , N - 1 \} , ( s , y ) \in \Omega _ { i } , z \in ( 0 , H )$ it holds that

$$
\mathbf { h } ( s , y ) = E _ { i + 1 } ( s , Z _ { i } ( s , y ) ) , \qquad 0 < T _ { i + 1 } ( s , Z _ { i } ( s , y ) ) \leq c G ( s , y ) , \qquad \mathrm { a n d } \qquad T _ { i + 1 } ( s , z ) = \left( \partial _ { z } E _ { i + 1 } \right) \left( s , z \right) .\tag{5.59}
$$

Next we combine $( 5 . 4 5 ) , ( 5 . 4 6 ) , ( 5 . 4 7 ) , ( 5 . 4 8 ) , ( 5 . 4 9 ) , ( 5 . 5 1 ) , ( 5 . 5 2 ) , ( 5 . 5 3 )$ , the fact that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that $\mathcal { F } _ { i } \big | _ { \{ ( s , u ) \in ( 0 , \theta ) \times \mathbb { R } \colon \mathfrak { e } _ { i } ( s ) < u < \mathfrak { e } _ { i + 1 } ( s ) \} } \in C ^ { 2 } ( \{ ( s , u ) \ \in \ ( 0 , \theta ) \times \mathbb { R } \colon \mathfrak { e } _ { i } ( s ) \ < \varnothing \} ) ,$ $u \ < \ \mathfrak { e } _ { i + 1 } ( s ) \} , [ 0 , H ] )$ , and Lemma 3.7 (applied with ${ \mathfrak { r } } _ { 0 } \ \cap \ \{ \mathfrak { R }  , \ { \mathfrak { r } } _ { 1 } \ \cap \ \theta , \ h \ \cap$ h, c ↶ min{c, 1/64}, $( { \mathcal { F } } _ { 1 } , \ldots , { \mathcal { F } } _ { N } ) \ \curvearrowright \ ( { \mathcal { F } } _ { 0 } , { \mathcal { F } } _ { 1 } , \ldots , { \mathcal { F } } _ { N - 1 } )$ , and $( \mathbf { m } _ { 1 } , \ldots , \mathbf { m } _ { N } ) \ \curvearrowleft \ ( \mathbf { m } _ { 0 } , \mathbf { m } _ { 1 } , \ldots , \mathbf { m } _ { N - 1 } )$ in the notation of Lemma 3.7) to obtain that there exist $\tau \in ( 0 , \theta )$ and definable $\mathbf { m } _ { 0 } , \mathbf { m } _ { 1 } , \ldots , \mathbf { m } _ { N - 1 } \in C ^ { 1 } ( ( 0 , \tau ) , [ 1 , \infty ) )$ ， which satisfy that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that

$$
\int _ { 0 } ^ { \tau } \mathbf { m } _ { i } ( s ) \mathrm { d } s < \infty ,\tag{5.60}
$$

that for all $i \in \{ 0 , 1 , \ldots , N - 1 \} , ( s , y ) \in \Omega _ { i } \cap ( ( 0 , \tau ) \times M )$ it holds that

$$
\begin{array} { r } { \big \vert ( \partial _ { r } \mathbf { h } ) ( s , y ) - S _ { i } ( s , Z _ { i } ( s , y ) ) \big \vert \le \mathbf { m } _ { i } ( s ) \big ( T _ { i + 1 } ( s , Z _ { i } ( s , y ) ) + \mathtt { c } G ( s , y ) \big ) , } \end{array}\tag{5.61}
$$

and that for all $i \in \{ 0 , 1 , \ldots , N - 1 \} , s \in ( 0 , \tau )$ it holds that

$$
s ^ { 2 } ( \mathbf { m } _ { i } ( s ) ) ^ { 2 } \leq \operatorname* { m i n } \{ \mathsf { c } , 1 / 6 4 \} \leq \frac { 1 } { 6 4 } .\tag{5.62}
$$

In the following for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ let $\omega _ { i } \colon ( 0 , \tau ) \times ( 0 , H ) \to ( 0 , 1 ]$ and $P _ { i } \colon ( 0 , \tau ) \times [ 0 , H ] \to \mathbb { R }$ satisfy that for all $s \in ( 0 , \tau ) , z \in ( 0 , H ) , v \in [ 0 , H ]$ it holds that

$$
\omega _ { i } ( s , z ) = \frac { T _ { i + 1 } ( s , z ) } { T _ { i + 1 } ( s , z ) + 8 s ^ { 2 } \mathbf { m } _ { i } ( s ) \sqrt { ( S _ { i } ( s , z ) ) ^ { 2 } + ( T _ { i + 1 } ( s , z ) ) ^ { 2 } } } \qquad \mathrm { a n d } \qquad P _ { i } ( s , v ) = \int _ { 0 } ^ { v } \omega _ { i } ( s , u ) \mathrm { d } u .\tag{5.63}
$$

Observe that $( 5 . 5 1 ) , ~ ( 5 . 5 2 ) , ~ ( 5 . 6 3 )$ , the fact that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that $E _ { i + 1 } \in$ $C ^ { 2 } ( ( 0 , \theta ) \times ( 0 , H )$ , R) is definable, the fact that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that $T _ { i + 1 } \in C ^ { 1 } ( ( 0 , \theta ) \times$ $( 0 , H ) , ( 0 , \infty ) )$ is definable and positive, the fact that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that $\mathbf { m } _ { i } \in$ $C ^ { 1 } ( ( 0 , \tau ) , [ 1 , \infty ) )$ is definable, and Lemma 2.8 establish that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that $\omega _ { i } \in C ^ { 1 } ( ( 0 , \tau ) \times ( 0 , H ) , ( 0 , 1 ] )$ is definable and that for all $i \in \{ 0 , 1 , \ldots , N - 1 \} , ( s , z ) \in ( 0 , \tau ) \times ( 0 , H )$ it holds that

$$
0 < \omega _ { i } ( s , z ) \leq 1 .\tag{5.64}
$$

Combining (5.45), (5.46), (5.63), items $( \mathrm { i } ) , ( \mathrm { i i } )$ , and $\mathrm { ( v ) }$ in Lemma 5.1, Lemma 2.13 (applied for every $j \in$ $\{ 0 , 1 , \ldots , N \}$ with $\begin{array} { r } { n \curvearrow D , \mathfrak { r } _ { 0 } \curvearrowleft { \tau } , \mathcal { O } \curvearrowleft { \mathcal { O } } \cap ( ( 0 , \tau ) \times M ) , h \curvearrowleft { \mathbf { h } } \middle \vert _ { \mathcal { O } \cap ( ( 0 , \tau ) \times M ) } , a \curvearrowleft { \mathfrak { e } } _ { j } \middle \vert ( 0 , \tau ) \middle \vert _ { ( j , \tau ) }  } \end{array}$ in the notation of Lemma 2.13), and the fact that for all $j \in \{ 0 , 1 , \ldots , N \}$ it holds that $\mathfrak { e } _ { j } \in C ^ { 2 } ( ( 0 , \theta )$ , R) is definable therefore demonstrates that there exist $\eta \in ( 0 , \tau )$ , definable $\mathfrak { b } _ { i } \in C ( ( 0 , \eta ) , [ 0 , \infty ) ) , i \in \{ 0 , 1 , . . . , N - 1 \}$ and $\beta _ { j } \colon ( 0 , \eta ) \to [ 0 , \infty ) , \ j \in \{ 0 , 1 , \ldots , N \}$ , which satisfy that for all $j \in \{ 0 , 1 , \ldots , N \}$ it holds that $( 0 , \eta ) \ni s \mapsto \beta _ { j } ( s ) \in \mathbb { R }$ is definable, continuously diferentiable, and integrable, that

$$
\sum _ { i = 0 } ^ { N - 1 } \int _ { 0 } ^ { \eta } \mathfrak { b } _ { i } ( s ) \mathrm { d } s + \sum _ { j = 0 } ^ { N } \int _ { 0 } ^ { \eta } \beta _ { j } ( s ) \mathrm { d } s < \infty ,\tag{5.65}
$$

that for all $i \in \{ 0 , 1 , . . . , N - 1 \} , s , t \in ( 0 , \eta ) , z , v \in [ 0 , H ]$ it holds that

$$
0 \leq P _ { i } ( s , z ) \leq H , \quad | P _ { i } ( s , z ) - P _ { i } ( s , v ) | \leq | z - v | , \quad \mathrm { a n d } \quad | P _ { i } ( s , z ) - P _ { i } ( t , z ) | \leq \left| \int _ { s } ^ { t } \mathfrak { b } _ { i } ( u ) \mathrm { d } u \right| ,\tag{5.66}
$$

and that for all $j \in \{ 0 , 1 , \ldots , N \} , ( s , y ) \in \mathcal { O } \cap ( ( 0 , \eta ) \times M )$ with $\begin{array} { r } { \mathbf { h } ( s , y ) = \mathfrak { e } _ { j } ( s ) } \end{array}$ and $G ( s , y ) > 0$ it holds that

$$
\left| ( \partial _ { r } \mathbf { h } ) ( s , y ) - ( \mathfrak { e } _ { j } ) ^ { \prime } ( s ) \right| \le \beta _ { j } ( s ) G ( s , y ) .\tag{5.67}
$$

In the following let M, $m \colon ( 0 , \eta )  [ 0 , \infty )$ satisfy that for all $s \in ( 0 , \eta )$ it holds that

$$
\mathbb { M } ( s ) = \kappa \sum _ { i = 0 } ^ { N - 1 } \bigl ( 5 \mathbf { m } _ { i } ( s ) + \mathfrak { b } _ { i } ( s ) \bigr ) \qquad \mathrm { a n d } \qquad m ( s ) = \mathbb { M } ( s ) + \frac { \Lambda } { \mathbf { c } } \sum _ { j = 0 } ^ { N } \beta _ { j } ( s )\tag{5.68}
$$

and let $Q \colon \mathcal { O } \cap ( ( 0 , \eta ) \times M ) \to [ 0 , \infty )$ satisfy that for all $( s , y ) \in \mathcal { O } \cap ( ( 0 , \eta ) \times M )$ it holds that

$$
Q ( s , y ) = \kappa \sum _ { i = 0 } ^ { N - 1 } P _ { i } ( s , Z _ { i } ( s , y ) ) .\tag{5.69}
$$

Note that (5.66), the fact that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that ${ \mathfrak { b } } _ { i }$ is nonnegative, and the triangle inequality ensure that for all $i \in \{ 0 , 1 , \ldots , N - 1 \} , s , t \in ( 0 , \eta ) , z , v \in [ 0 , H ]$ it holds that

$$
\vert P _ { i } ( t , v ) - P _ { i } ( s , z ) \vert \leq \vert P _ { i } ( t , v ) - P _ { i } ( t , z ) \vert + \vert P _ { i } ( t , z ) - P _ { i } ( s , z ) \vert \leq \vert v - z \vert + \int _ { \operatorname* { m i n } \{ s , t \} } ^ { \operatorname* { m a x } \{ s , t \} } \mathfrak { b } _ { i } ( u ) \mathrm { d } u .\tag{5.70}
$$

This, the fact that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that ${ \mathfrak { b } } _ { i }$ is continuous on $( 0 , \eta )$ , and the fundamental theorem of calculus imply that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that $P _ { i } | _ { ( 0 , \eta ) \times \lbrack 0 , H \rbrack }$ is continuous. Combining this, (5.66), (5.69), the fact that $\kappa > 0$ , and the fact that for all $i \in \{ 0 , 1 , \ldots , \overset { } { N } - 1 \}$ it holds that $Z _ { i }$ is continuous shows that $Q$ is continuous and satisfies that for all $( s , y ) \in \mathcal { O } \cap ( ( 0 , \eta ) \times M )$ it holds that

$$
0 \leq Q ( s , y ) \leq \kappa N H .\tag{5.71}
$$

Furthermore, observe that (5.68), (5.69), the fact that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that m and ${ \mathfrak { b } } _ { i }$ are definable, continuous, and nonnegative, and the fact that for all $j \in \{ 0 , 1 , \ldots , N \}$ it holds that $\beta _ { j }$ is definable, continuous, and nonnegative prove that m is definable, continuous, nonnegative, and Lebesgue measurable. This, (5.60), (5.65), (5.68), and (5.69) establish that $\begin{array} { r } { \int _ { 0 } ^ { \eta } m ( s ) \mathrm { d } s < \infty . } \end{array}$ . We claim that for every interval $I \subseteq \mathbb { R }$ , all locally absolutely continuous $r \colon I \to ( 0 , \eta )$ and $x \colon I  M$ , and all Lebesgue measurable $\alpha \colon I  [ 0 , \infty )$ and $\mathbf { \boldsymbol { A } } \colon I \to \mathbb { R } ^ { \breve { D } \times D }$ with $\textstyle \bigcup _ { t \in I } \{ ( r ( t ) , { \dot { x } } ( t ) ) \} \subseteq { \mathcal { O } }$ and with the property that for Lebesgue-almost all $t \in I$ and every $v \in T _ { x ( t ) } M$ it holds that

$$
A ( t ) v \in T _ { x ( t ) } M , \qquad \langle A ( t ) v , v \rangle \geq \mathsf { a } | v | ^ { 2 } , \qquad | A ( t ) v | \leq \mathsf { b } | v | ,\tag{5.72}
$$

$$
\begin{array} { r } { r ^ { \prime } ( t ) = - \alpha ( t ) ( r ( t ) ) ^ { 2 } ( \partial _ { r } \mathbf { h } ) ( r ( t ) , x ( t ) ) , \qquad \mathrm { a n d } \qquad x ^ { \prime } ( t ) = - \alpha ( t ) A ( t ) ( \nabla _ { M } \mathbf { h } ) ( r ( t ) , x ( t ) ) } \end{array}\tag{5.73}
$$

we have that for all $a , b \in I$ with $a \leq b$ it holds that

$$
Q ( r ( b ) , x ( b ) ) - Q ( r ( a ) , x ( a ) ) \leq - \int _ { a } ^ { b } | x ^ { \prime } ( t ) | \mathrm { d } t + \int _ { a } ^ { b } m ( r ( t ) ) | r ^ { \prime } ( t ) | \mathrm { d } t\tag{5.74}
$$

(cf. Definition 2.3).

To prove (5.74), let $I \subseteq \mathbb { R }$ be an interval, let $r \colon I \to ( 0 , \eta )$ and $x \colon I  M$ be locally absolutely continuous, let α : $I  [ 0 , \infty )$ and $A \colon I  \mathbb { R } ^ { D \times D }$ be Lebesgue measurable, assume $\begin{array} { r l } { \bigcup _ { t \in I } \{ ( r ( t ) , x ( t ) ) \} \subseteq } & { { } } \end{array}$ O, assume that for Lebesgue-almost all $t \in I$ and every $v \in T _ { x ( t ) } M$ it holds that

$$
A ( t ) v \in T _ { x ( t ) } M , \qquad \langle A ( t ) v , v \rangle \geq \mathsf { a } | v | ^ { 2 } , \qquad | A ( t ) v | \leq \mathsf { b } | v | ,\tag{5.75}
$$

$$
\begin{array} { r } { r ^ { \prime } ( t ) = - \alpha ( t ) ( r ( t ) ) ^ { 2 } ( \partial _ { r } { \bf h } ) ( r ( t ) , x ( t ) ) , \qquad \mathrm { a n d } \qquad x ^ { \prime } ( t ) = - \alpha ( t ) A ( t ) ( \nabla _ { M } { \bf h } ) ( r ( t ) , x ( t ) ) , } \end{array}\tag{5.76}
$$

and let $p \colon I \to \mathbb { R } ^ { D } , w \colon I \to \mathbb { R } , \gamma , \ell \colon I \to [ 0 , \infty )$ , and $\xi _ { i } \colon I \to [ 0 , H ] , i \in \{ 0 , 1 , \ldots , N - 1 \}$ , satisfy for all $t \in I , i \in \{ 0 , 1 , \ldots , N - 1 \}$ that

$$
\begin{array} { r l } & { p ( t ) = ( \nabla _ { M } \mathbf { h } ) ( r ( t ) , x ( t ) ) , \qquad w ( t ) = ( \partial _ { r } \mathbf { h } ) ( r ( t ) , x ( t ) ) , \qquad \gamma ( t ) = \sqrt { \operatorname* { m a x } \{ 0 , \langle A ( t ) p ( t ) , p ( t ) \rangle \} } , } \\ & { \qquad \ell ( t ) = \alpha ( t ) | A ( t ) p ( t ) | , \qquad \mathrm { a n d } \qquad \xi _ { i } ( t ) = Z _ { i } ( r ( t ) , x ( t ) ) . } \end{array}\tag{5.77}
$$

Note that (5.45), (5.46), (5.75), (5.77), and the Cauchy–Schwarz inequality demonstrate that for Lebesgue-almost all $t \in I$ it holds that

$$
\begin{array} { r } { c | p ( t ) | \leq \gamma ( t ) , \qquad ( \gamma ( t ) ) ^ { 2 } \leq | A ( t ) p ( t ) | | p ( t ) | , \qquad \mathrm { a n d } \qquad | A ( t ) p ( t ) | \leq \mathsf { b } | p ( t ) | . } \end{array}\tag{5.78}
$$

Combining (5.45), (5.46), (5.76), (5.77), and the fact that for Lebesgue-almost all $t \in I$ with $\gamma ( t ) = 0$ it holds that $p ( t ) = 0$ hence ensures that for Lebesgue-almost all $t \in I$ it holds that

$$
\mathsf { c } \alpha ( t ) \gamma ( t ) \le \ell ( t ) = | x ^ { \prime } ( t ) | \le \Lambda \alpha ( t ) \gamma ( t ) .\tag{5.79}
$$

This and the fact that x is locally absolutely continuous imply that for all $a , b \in I$ with $a < b$ it holds that

$$
\int _ { a } ^ { b } \alpha ( t ) \gamma ( t ) \mathrm { d } t \leq \frac { 1 } { c } \int _ { a } ^ { b } | x ^ { \prime } ( t ) | \mathrm { d } t < \infty .\tag{5.80}
$$

Next, observe that $( 5 . 7 5 ) , ( 5 . 7 6 ) , ( 5 . 7 7 )$ , the fact that $ { \mathbf { h } } \in  { \mathbf { C } } ^ { 1 } (  { \mathcal { O } } , ( - 1 , 1 ) )$ , the fact that for all compact subintervals $J \subseteq I$ it holds that $\{ ( r ( t ) , x ( t ) ) \colon t \in J \}$ is a compact subset of O, and the chain rule show that $I \ni t \mapsto \mathbf { h } ( r ( t ) , x ( t ) ) \in ]$ R is locally absolutely continuous and that for Lebesgue-almost all $t \in I$ it holds that

$$
\begin{array} { r l r } & { } & { \frac { \mathrm { d } } { \mathrm { d } t } \mathbf { h } ( r ( t ) , x ( t ) ) = ( \partial _ { r } \mathbf { h } ) ( r ( t ) , x ( t ) ) r ^ { \prime } ( t ) + \langle ( \nabla _ { M } \mathbf { h } ) ( r ( t ) , x ( t ) ) , x ^ { \prime } ( t ) \rangle = w ( t ) r ^ { \prime } ( t ) + \langle p ( t ) , x ^ { \prime } ( t ) \rangle } \\ & { } & { = - \alpha ( t ) ( r ( t ) ) ^ { 2 } ( w ( t ) ) ^ { 2 } - \alpha ( t ) \langle A ( t ) p ( t ) , p ( t ) \rangle = - \alpha ( t ) \big ( ( \gamma ( t ) ) ^ { 2 } + ( r ( t ) ) ^ { 2 } ( w ( t ) ) ^ { 2 } \big ) . } \end{array}\tag{5.81}
$$

In addition, note that (5.54), (5.77), the fact that r and x are continuous, and the fact that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that $Z _ { i }$ is continuous prove that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that $\xi _ { i }$ is continuous, that $\{ u \in I \colon 0 < \xi _ { i } ( u ) < H \}$ is open in $I ,$ and that

$$
\{ u \in I \colon 0 < \xi _ { i } ( u ) < H \} = \{ u \in I \colon ( r ( u ) , x ( u ) ) \in \Omega _ { i } \} .\tag{5.82}
$$

Combining this, (5.55), (5.77), and the fact that r and x are locally absolutely continuous establishes that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that $\xi _ { i } \big | _ { \{ u \in I : 0 < \xi _ { i } ( u ) < H \} }$ is locally absolutely continuous. This, (5.51), (5.52), (5.59), (5.77), (5.82), the fact that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that $E _ { i + 1 } \in$ $C ^ { 2 } ( ( 0 , \theta ) \times ( 0 , H )$ , R), and the fact that r is locally absolutely continuous demonstrate that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that for Lebesgue-almost all $t \in \{ u \in I \colon 0 < \xi _ { i } ( u ) < H \}$ it holds that $\xi _ { i }$ is diferentiable at t and that

$$
\begin{array} { l } { \displaystyle \frac { \mathrm d } { \mathrm d t } \mathbf { h } ( r ( t ) , x ( t ) ) = \frac { \mathrm d } { \mathrm d t } E _ { i + 1 } ( r ( t ) , \xi _ { i } ( t ) ) = \left( E _ { i + 1 } \right) ^ { \prime } ( r ( t ) , \xi _ { i } ( t ) ) \big ( r ^ { \prime } ( t ) , ( \xi _ { i } ) ^ { \prime } ( t ) \big ) } \\ { = \big ( \partial _ { s } E _ { i + 1 } \big ) \big ( r ( t ) , \xi _ { i } ( t ) \big ) r ^ { \prime } ( t ) + \big ( \partial _ { z } E _ { i + 1 } \big ) \big ( r ( t ) , \xi _ { i } ( t ) \big ) ( \xi _ { i } ) ^ { \prime } ( t ) } \\ { = S _ { i } ( r ( t ) , \xi _ { i } ( t ) ) r ^ { \prime } ( t ) + T _ { i + 1 } ( r ( t ) , \xi _ { i } ( t ) ) ( \xi _ { i } ) ^ { \prime } ( t ) . } \end{array}\tag{5.83}
$$

Combining this, (5.59), (5.76), (5.77), and (5.81) ensures that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that for Lebesgue-almost all $t \in \{ u \in I \colon 0 < \xi _ { i } ( u ) < H \}$ it holds that

$$
\begin{array} { l } { \displaystyle ( \xi _ { i } ) ^ { \prime } ( t ) = \frac { \frac { \mathrm { d } } { \mathrm { d } t } \mathbf { h } ( r ( t ) , x ( t ) ) - S _ { i } ( r ( t ) , \xi _ { i } ( t ) ) r ^ { \prime } ( t ) } { T _ { i + 1 } ( r ( t ) , \xi _ { i } ( t ) ) } } \\ { \displaystyle \quad = - \frac { \alpha ( t ) } { T _ { i + 1 } ( r ( t ) , \xi _ { i } ( t ) ) } \big ( ( \gamma ( t ) ) ^ { 2 } + ( r ( t ) ) ^ { 2 } w ( t ) [ w ( t ) - S _ { i } ( r ( t ) , \xi _ { i } ( t ) ) ] \big ) . } \end{array}\tag{5.84}
$$

Moreover, observe that (5.45), (5.46), (5.59), (5.77), (5.78), and (5.82) imply that for all $i \in \{ 0 , 1 , \ldots , N -$ 1} it holds that for Lebesgue-almost all $t \in \{ u \in I \colon 0 < \xi _ { i } ( u ) < H \}$ it holds that

$$
0 < T _ { i + 1 } ( r ( t ) , \xi _ { i } ( t ) ) \leq \mathsf { c } G ( r ( t ) , x ( t ) ) = \mathsf { c } | p ( t ) | \leq \gamma ( t ) .\tag{5.85}
$$

This, (5.61), (5.77), (5.82), the fact that for all $t \in I$ it holds that $0 < r ( t ) < \eta < \tau .$ , and the fact that for all $i \in \{ 0 , 1 , \ldots , N - 1 \} , s \in ( 0 , \tau )$ it holds that $\mathbf { m } _ { i } ( s ) \geq 1$ show that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that for Lebesgue-almost all $t \in \{ u \in I \colon 0 < \xi _ { i } ( u ) < H \}$ it holds that

$$
\begin{array} { r l } & { \left| w ( t ) - S _ { i } ( r ( t ) , \xi _ { i } ( t ) ) \right| \le \mathbf { m } _ { i } ( r ( t ) ) \big ( T _ { i + 1 } ( r ( t ) , \xi _ { i } ( t ) ) + \mathsf { c } G ( r ( t ) , x ( t ) ) \big ) } \\ & { \qquad \le \mathbf { m } _ { i } ( r ( t ) ) \big ( T _ { i + 1 } ( r ( t ) , \xi _ { i } ( t ) ) + \gamma ( t ) \big ) . } \end{array}\tag{5.86}
$$

In the next step we note that (5.62), the fact that for all $t \in I$ it holds that $0 < r ( t ) < \eta < \tau$ , and the fact that for all $i \in \{ 0 , 1 , \ldots , N - 1 \} , s \in ( 0 , \tau )$ it holds that $\mathbf { m } _ { i } ( s ) \geq 1$ prove that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ $t \in I$ it holds that

$$
r ( t ) > 0 , \qquad \mathbf { m } _ { i } ( r ( t ) ) \geq 1 , \qquad \mathrm { a n d } \qquad ( r ( t ) ) ^ { 2 } ( \mathbf { m } _ { i } ( r ( t ) ) ) ^ { 2 } \leq \frac { 1 } { 6 4 } .\tag{5.87}
$$

Combining (5.63), (5.76), (5.77), (5.84), (5.85), and the fact that for all $t \in I$ it holds that $\alpha ( t ) \geq 0$ therefore establishes that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that for Lebesgue-almost all $t \in \{ u \in$ $I \colon 0 < \xi _ { i } ( u ) < H \}$ it holds that

$$
\begin{array} { r l } & { \omega _ { i } ( r ( t ) , \xi _ { i } ( t ) ) ( \xi _ { i } ) ^ { \prime } ( t ) } \\ & { = - \alpha ( t ) \frac { ( \gamma ( t ) ) ^ { 2 } + ( r ( t ) ) ^ { 2 } w ( t ) [ w ( t ) - S _ { i } ( r ( t ) , \xi _ { i } ( t ) ) ] } { T _ { i + 1 } ( r ( t ) , \xi _ { i } ( t ) ) + 8 ( r ( t ) ) ^ { 2 } \mathbf { m } _ { i } ( r ( t ) ) \sqrt { ( S _ { i } ( r ( t ) , \xi _ { i } ( t ) ) ) ^ { 2 } + ( T _ { i + 1 } ( r ( t ) , \xi _ { i } ( t ) ) ) ^ { 2 } } } } \end{array}\tag{5.88}
$$

and

$$
| r ^ { \prime } ( t ) | = \left| - \alpha ( t ) ( r ( t ) ) ^ { 2 } w ( t ) \right| = \alpha ( t ) ( r ( t ) ) ^ { 2 } | w ( t ) | .\tag{5.89}
$$

This, $( 5 . 8 8 ) , ( 5 . 8 5 ) , ( 5 . 8 6 ) , ( 5 . 8 7 ) $ , items (i) and (ii) in Lemma 5.2 (applied with $r \backsim r ( t ) , T \backsim$ $T _ { i + 1 } ( r ( t ) , \xi _ { i } ( t ) ) , \ g \ \curvearrow \ \gamma ( t ) , \ n \frown \mathbf { m } _ { i } ( r ( t ) ) , \ v \frown w ( t ) , \ c \frown \ S _ { i } ( r ( t ) , \xi _ { i } ( t ) )$ in the notation of Lemma $5 . 2 )$ and the fact that for all $t \in I$ it holds that $\alpha ( t ) \geq 0$ demonstrate that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that for Lebesgue-almost all $t \in \{ u \in I \colon 0 < \xi _ { i } ( u ) < H \}$ it holds that

$$
\omega _ { i } ( r ( t ) , \xi _ { i } ( t ) ) ( \xi _ { i } ) ^ { \prime } ( t ) \leq 2 \alpha ( t ) \mathbf { m } _ { i } ( r ( t ) ) ( r ( t ) ) ^ { 2 } | w ( t ) | = 2 \mathbf { m } _ { i } ( r ( t ) ) | r ^ { \prime } ( t ) |\tag{5.90}
$$

and

$$
\omega _ { i } ( r ( t ) , \xi _ { i } ( t ) ) ( \xi _ { i } ) ^ { \prime } ( t ) \leq 5 \alpha ( t ) \mathbf { m } _ { i } ( r ( t ) ) ( r ( t ) ) ^ { 2 } | w ( t ) | - \frac { \alpha ( t ) \gamma ( t ) } { 4 } = 5 \mathbf { m } _ { i } ( r ( t ) ) | r ^ { \prime } ( t ) | - \frac { \alpha ( t ) \gamma ( t ) } { 4 } .\tag{5.91}
$$

In the following for every $i \in \{ 0 , 1 , \ldots , N - 1 \} , \varepsilon \in ( 0 , ^ { H / 2 } )$ let $Y _ { i } ^ { \varepsilon } \colon { \mathcal { O } } \cap ( ( 0 , \eta ) \times M ) \to [ \varepsilon , H - \varepsilon ]$ and $q _ { i } ^ { \varepsilon } \colon \mathcal { O } \cap ( ( 0 , \eta ) \times M ) \to \mathbb { R }$ satisfy for all $( s , y ) \in \mathcal { O } \cap ( ( 0 , \eta ) \times M )$ that

$$
Y _ { i } ^ { \varepsilon } ( s , y ) = \operatorname* { m a x } \{ \varepsilon , \operatorname* { m i n } \{ Z _ { i } ( s , y ) , H - \varepsilon \} \} \mathrm { ~ a n d ~ } q _ { i } ^ { \varepsilon } ( s , y ) = P _ { i } ( s , Y _ { i } ^ { \varepsilon } ( s , y ) ) ,\tag{5.92}
$$

for every $\varepsilon \in ( 0 , H / 2 )$ let $V _ { \varepsilon } \colon \mathcal { O } \cap ( ( 0 , \eta ) \times M ) \to$ R and $\mathcal { U } _ { \varepsilon } \subseteq \mathcal { O } \cap \left( ( 0 , \eta ) \times M \right)$ satisfy for all $( s , y ) \in$ $\mathcal { O } \cap ( ( 0 , \eta ) \times M )$ that

$$
V _ { \varepsilon } ( s , y ) = \kappa \sum _ { i = 0 } ^ { N - 1 } q _ { i } ^ { \varepsilon } ( s , y ) \qquad \mathrm { a n d } \qquad \mathcal { U } _ { \varepsilon } = \bigcup _ { i = 0 } ^ { N - 1 } \{ ( u , v ) \in \mathcal O \cap ( ( 0 , \eta ) \times M ) \colon \varepsilon < Z _ { i } ( u , v ) < H - \varepsilon \} ,\tag{5.93}
$$

and let $\Gamma \subseteq O \cap ( ( 0 , \eta ) \times M )$ satisfy

$$
\Gamma = \bigcup _ { j = 0 } ^ { N } \{ ( s , y ) \in \mathcal O \cap ( ( 0 , \eta ) \times M ) \colon \mathbf h ( s , y ) = \mathfrak { e } _ { j } ( s ) \} .\tag{5.94}
$$

Observe that (5.66), (5.68), (5.69), (5.92), (5.93), the triangle inequality, the fact that $\kappa > 0 .$ and the fact that for all $z \in [ 0 , H ] , \ : \varepsilon \in ( 0 , H / 2 )$ it holds that $\left| \operatorname* { m a x } \{ \varepsilon , \operatorname* { m i n } \{ z , H - \varepsilon \} \} - z \right| \leq \varepsilon$ ensure that for all $\varepsilon \in ( 0 , ^ { H / 2 } ) , ( s , y ) \in \mathcal { O } \cap ( ( 0 , \eta ) \times M )$ it holds that

$$
\begin{array} { l l } { \displaystyle \left. V _ { \varepsilon } ( s , y ) - Q ( s , y ) \right. = \kappa \Biggl \lvert \displaystyle \sum _ { i = 0 } ^ { N - 1 } \left( P _ { i } ( s , Y _ { i } ^ { \varepsilon } ( s , y ) ) - P _ { i } ( s , Z _ { i } ( s , y ) ) \right) \Biggr \rvert } \\ { \displaystyle \leq \kappa \displaystyle \sum _ { i = 0 } ^ { N - 1 } \left. P _ { i } ( s , Y _ { i } ^ { \varepsilon } ( s , y ) ) - P _ { i } ( s , Z _ { i } ( s , y ) ) \right. \leq \kappa \displaystyle \sum _ { i = 0 } ^ { N - 1 } \left. Y _ { i } ^ { \varepsilon } ( s , y ) - Z _ { i } ( s , y ) \right. \leq \kappa \displaystyle \sum _ { i = 0 } ^ { N - 1 } \varepsilon = \kappa N \varepsilon . } \end{array}\tag{5.95}
$$

Next we combine (5.44), (5.47), (5.54), (5.58), (5.93), and (5.94) to obtain that for all $\varepsilon , \varepsilon ^ { \prime } \in ( 0 , { H } / { 2 } )$ with $\varepsilon ^ { \prime } < \varepsilon$ it holds that $\mathcal { U } _ { \varepsilon } \subseteq \subseteq \mathcal { U } _ { \varepsilon } ,$ and that

$$
\textstyle \bigcup _ { \varepsilon \in ( 0 , H / 2 ) } \mathcal { U } _ { \varepsilon } = ( \mathcal { O } \cap ( ( 0 , \eta ) \times M ) ) \backslash \Gamma .\tag{5.96}
$$

Furthermore, note that (5.54), (5.66), (5.80), (5.90), (5.91), (5.92), item (ii) in Lemma 4.3 (applied to $Z _ { i }$ on $\mathcal { O } \cap ( ( 0 , \eta ) \times M ) )$ , and the fact that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that $\mathbf { m } _ { i }$ and ${ \mathfrak { b } } _ { i }$ are continuous imply that for all $i \in \{ 0 , 1 , \ldots , N - 1 \} , \varepsilon \in ( 0 , \dot { H } / 2 )$ it holds that $I \ni t \mapsto q _ { i } ^ { \varepsilon } ( r ( t ) , x ( t ) ) \in \mathbb { R }$ is locally absolutely continuous. Next, observe that (5.54), (5.66), (5.80), (5.90), (5.91), (5.92), and item (iii) in Lemma 4.3 show that for all $i \in \{ 0 , 1 , \ldots , N - 1 \} , \varepsilon \in ( 0 ,  H / 2 )$ it holds that for Lebesgue-almost all $t \in I$ it holds that

$$
\frac { \mathrm { d } } { \mathrm { d } t } q _ { i } ^ { \varepsilon } ( r ( t ) , x ( t ) ) \leq \big ( 2 \mathbf { m } _ { i } ( r ( t ) ) + \mathfrak { b } _ { i } ( r ( t ) ) \big ) | r ^ { \prime } ( t ) | .\tag{5.97}
$$

In addition, note that (5.54), (5.66), (5.80), (5.90), (5.91), (5.92), and item $( \mathrm { i v } )$ in Lemma 4.3 prove that for all $i \in \{ 0 , 1 , \ldots , N - 1 \} , \varepsilon \in ( 0 , ^ { H / 2 } )$ it holds that for Lebesgue-almost all $t \in \{ u \in I \colon \varepsilon < \xi _ { i } ( u ) <$ $H - \varepsilon \}$ it holds that

$$
\frac { \mathrm { d } } { \mathrm { d } t } q _ { i } ^ { \varepsilon } ( r ( t ) , x ( t ) ) \leq \big ( 5 \mathbf { m } _ { i } ( r ( t ) ) + \mathfrak { b } _ { i } ( r ( t ) ) \big ) | r ^ { \prime } ( t ) | - \frac { 1 } { 4 } \alpha ( t ) \gamma ( t ) .\tag{5.98}
$$

Moreover, observe that (5.93) and (5.97) establish that for all $\varepsilon \in ( 0 , H / 2 )$ and Lebesgue-almost all $t \in I$ it holds that

$$
\frac { \mathrm { d } } { \mathrm { d } t } V _ { \varepsilon } ( r ( t ) , x ( t ) ) = \kappa \sum _ { i = 0 } ^ { N - 1 } \frac { \mathrm { d } } { \mathrm { d } t } q _ { i } ^ { \varepsilon } ( r ( t ) , x ( t ) ) .\tag{5.99}
$$

Combining this, (5.97), (5.98), the fact that for all $i \in \{ 0 , 1 , \ldots , N - 1 \}$ it holds that $\mathbf { m } _ { i }$ is nonnegative, and the fact that $\kappa > 0$ demonstrates that for all $\varepsilon \in ( 0 , H / 2 )$ and Lebesgue-almost all $t \in I$ it holds that

$$
\frac { \mathrm { d } } { \mathrm { d } t } V _ { \varepsilon } ( r ( t ) , x ( t ) ) \le - \frac { \kappa \alpha ( t ) \gamma ( t ) } { 4 } \sum _ { i = 0 } ^ { N - 1 } \mathbf { 1 } _ { ( \varepsilon , H - \varepsilon ) } ( \xi _ { i } ( t ) ) + \kappa \sum _ { i = 0 } ^ { N - 1 } \big ( 5 \mathbf { m } _ { i } ( r ( t ) ) + \mathfrak { b } _ { i } ( r ( t ) ) \big ) | r ^ { \prime } ( t ) | .\tag{5.100}
$$

In the next step we combine (5.77) and (5.93) to obtain that for all $\varepsilon \in ( 0 , H / 2 ) , t \in I$ it holds that

$$
\mathbf { 1 } _ { \boldsymbol { U } _ { \varepsilon } } ( r ( t ) , \boldsymbol { x } ( t ) ) = \operatorname* { m i n } \biggl \{ 1 , \sum _ { i = 0 } ^ { N - 1 } \mathbf { 1 } _ { ( \varepsilon , H - \varepsilon ) } ( \xi _ { i } ( t ) ) \biggr \} \leq \sum _ { i = 0 } ^ { N - 1 } \mathbf { 1 } _ { ( \varepsilon , H - \varepsilon ) } ( \xi _ { i } ( t ) ) .\tag{5.101}
$$

This, the fact that for all $t \in I$ it holds that $\alpha ( t ) \geq 0$ and $\gamma ( t ) \geq 0 .$ , and the fact that $\kappa > 0$ ensure that for all $\varepsilon \in ( 0 , H / 2 ) , t \in I$ it holds that

$$
- \frac { \kappa \alpha ( t ) \gamma ( t ) } { 4 } \sum _ { i = 0 } ^ { N - 1 } \mathbf { 1 } _ { ( \varepsilon , H - \varepsilon ) } ( \xi _ { i } ( t ) ) \leq - \frac { \kappa \alpha ( t ) \gamma ( t ) } { 4 } \mathbf { 1 } _ { \mathcal { U } _ { \varepsilon } } ( r ( t ) , x ( t ) ) .\tag{5.102}
$$

Next we combine (5.45), (5.46), (5.68), (5.69), and the fact that for all $t \in I$ it holds that $r ( t ) \in ( 0 , \eta )$ to obtain that for all $t \in I$ it holds that

$$
\kappa \sum _ { i = 0 } ^ { N - 1 } \big ( 5 \mathbf { m } _ { i } ( r ( t ) ) + \mathfrak { b } _ { i } ( r ( t ) ) \big ) = \mathbb { M } ( r ( t ) ) \qquad \mathrm { a n d } \qquad \frac { \kappa } { 4 } = \Lambda .\tag{5.103}
$$

Combining (5.100) and (5.102) hence implies that for all $\varepsilon \in ( 0 , H / 2 )$ and Lebesgue-almost all $t \in I$ it holds that

$$
\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } t } V _ { \varepsilon } ( r ( t ) , x ( t ) ) \leq - \frac { \kappa \alpha ( t ) \gamma ( t ) } { 4 } \mathbf { 1 } _ { \mathcal { U } _ { \varepsilon } } ( r ( t ) , x ( t ) ) + \mathbb { M } ( r ( t ) ) | r ^ { \prime } ( t ) | } \\ { = - \Lambda \alpha ( t ) \gamma ( t ) \mathbf { 1 } _ { \mathcal { U } _ { \varepsilon } } ( r ( t ) , x ( t ) ) + \mathbb { M } ( r ( t ) ) | r ^ { \prime } ( t ) | . } \end{array}\tag{5.104}
$$

This, (5.79), and the fact that for all $\varepsilon \in ( 0 , H / 2 ) , t \in I$ it holds that $\mathbf { 1 } _ { \mathcal { U } _ { \varepsilon } } ( r ( t ) , x ( t ) ) \geq 0$ show that for all $\varepsilon \in ( 0 , H / 2 )$ and Lebesgue-almost all $t \in I$ it holds that

$$
\frac { \mathrm { d } } { \mathrm { d } t } V _ { \varepsilon } ( r ( t ) , x ( t ) ) \leq - \ell ( t ) \mathbf { 1 } _ { \mathcal { U } _ { \varepsilon } } ( r ( t ) , x ( t ) ) + \mathbb { M } ( r ( t ) ) | r ^ { \prime } ( t ) | .\tag{5.105}
$$

Combining (5.93), the fact that for all $i \ \in \ \{ 0 , 1 , \ldots , N - 1 \} , \ \varepsilon \ \in \ ( 0 , ^ { H / 2 } )$ it holds that ${ \textit { I } } \ni { \textit { t } } \mapsto$ $q _ { i } ^ { \varepsilon } ( r ( t ) , x ( t ) ) \in \mathbb { I }$ R is locally absolutely continuous, and the fundamental theorem of calculus therefore proves that for all $\varepsilon \in ( 0 , H / 2 ) , a , b \in I$ with $a < b$ it holds that

$$
V _ { \varepsilon } ( r ( b ) , x ( b ) ) - V _ { \varepsilon } ( r ( a ) , x ( a ) ) \leq - \int _ { a } ^ { b } \ell ( t ) { \mathbf 1 } _ { \mathscr { U } _ { \varepsilon } } ( r ( t ) , x ( t ) ) \mathrm { d } t + \int _ { a } ^ { b } { \mathbb { M } } ( r ( t ) ) | r ^ { \prime } ( t ) | \mathrm { d } t .\tag{5.106}
$$

Furthermore, note that (5.95) establishes that for all $( s , y ) \in \mathcal { O } \cap ( ( 0 , \eta ) \times M )$ it holds that

$$
\operatorname * { l i m } _ { \varepsilon  0 } V _ { \varepsilon } ( s , y ) = Q ( s , y ) .\tag{5.107}
$$

Hence, we obtain that for all $a , b \in I$ with $a < b$ it holds that

$$
\operatorname * { l i m } _ { \varepsilon \to 0 } ( V _ { \varepsilon } ( r ( b ) , x ( b ) ) - V _ { \varepsilon } ( r ( a ) , x ( a ) ) ) = Q ( r ( b ) , x ( b ) ) - Q ( r ( a ) , x ( a ) ) .\tag{5.108}
$$

Next, observe that (5.79), the fact that ℓ is nonnegative and measurable, the fact that M is continuous and nonnegative, the fact that r and x are locally absolutely continuous, the fundamental theorem of calculus for absolutely continuous functions, and the extreme value theorem demonstrate that for all $a , b \in I$ with $a < b$ it holds that

$$
0 \leq \int _ { a } ^ { b } \ell ( t ) \mathrm { d } t = \int _ { a } ^ { b } | x ^ { \prime } ( t ) | \mathrm { d } t < \infty \qquad \mathrm { a n d } \qquad 0 \leq \int _ { a } ^ { b } \mathbb { M } ( r ( t ) ) | r ^ { \prime } ( t ) | \mathrm { d } t < \infty .\tag{5.109}
$$

This, (5.93), (5.96), the fact that ℓ is nonnegative and measurable, and the monotone convergence theorem ensure that for all $a , b \in I$ with $a < b$ it holds that

$$
\operatorname* { l i m } _ { \varepsilon \to 0 } \int _ { a } ^ { b } \ell ( t ) { \mathbf 1 } _ { \mathcal { U } _ { \varepsilon } } ( r ( t ) , x ( t ) ) \mathrm { d } t = \int _ { a } ^ { b } \ell ( t ) { \mathbf 1 } _ { ( \mathcal { O } \cap ( ( 0 , \eta ) \times M ) ) \setminus } ( r ( t ) , x ( t ) ) \mathrm { d } t < \infty .\tag{5.110}
$$

Combining (5.106), (5.108), and (5.109) hence implies that for all $a , b \in I$ with $a < b$ it holds that

$$
Q ( r ( b ) , x ( b ) ) - Q ( r ( a ) , x ( a ) ) \leq - \int _ { a } ^ { b } \ell ( t ) { \mathbf 1 } _ { ( O \cap ( ( 0 , \eta ) \times M ) ) \setminus \Gamma } ( r ( t ) , x ( t ) ) \mathrm d t + \int _ { a } ^ { b } \mathbb { M } ( r ( t ) ) | r ^ { \prime } ( t ) | \mathrm d t .\tag{5.111}
$$

In the following for all $j \in \{ 0 , 1 , \ldots , N \}$ let $f _ { j } \colon I $ R satisfy that for all $t \in I$ it holds that

$$
f _ { j } ( t ) = \mathbf { h } ( r ( t ) , x ( t ) ) - \mathfrak { e } _ { j } ( r ( t ) ) .\tag{5.112}
$$

Note that (5.76), (5.77), (5.81), (5.112), the fact that for all $j ~ \in ~ \{ 0 , 1 , \ldots , N \}$ it holds that ${ \mathfrak { e } } _ { j } \ \in$ $C ^ { 1 } ( ( 0 , \theta ) , \mathbb { R } )$ , Lemma 4.1, and the chain rule show that for all $j \in \{ 0 , 1 , \ldots , N \}$ it holds that for Lebesguealmost all $t \in f _ { j } ^ { - 1 } ( \{ 0 \} )$ it holds that

$$
0 = ( f _ { j } ) ^ { \prime } ( t ) = - \alpha ( t ) \big ( ( \gamma ( t ) ) ^ { 2 } + ( r ( t ) ) ^ { 2 } w ( t ) [ w ( t ) - ( \mathfrak { e } _ { j } ) ^ { \prime } ( r ( t ) ) ] \big ) .\tag{5.113}
$$

This, (5.76), (5.77), and the fact that for all $t \in I$ it holds that $\alpha ( t ) \geq 0$ prove that for all $j \in \{ 0 , 1 , \ldots , N \}$ it holds that for Lebesgue-almost all $t \in f _ { j } ^ { - 1 } ( \{ 0 \} )$ it holds that

$$
\begin{array} { r l } & { \alpha ( t ) ( \gamma ( t ) ) ^ { 2 } = - \alpha ( t ) ( r ( t ) ) ^ { 2 } w ( t ) [ w ( t ) - ( \mathfrak { e } _ { j } ) ^ { \prime } ( r ( t ) ) ] = [ w ( t ) - ( \mathfrak { e } _ { j } ) ^ { \prime } ( r ( t ) ) ] r ^ { \prime } ( t ) } \\ & { \qquad = | [ w ( t ) - ( \mathfrak { e } _ { j } ) ^ { \prime } ( r ( t ) ) ] r ^ { \prime } ( t ) | = | w ( t ) - ( \mathfrak { e } _ { j } ) ^ { \prime } ( r ( t ) ) | | r ^ { \prime } ( t ) | . } \end{array}\tag{5.114}
$$

In addition, observe that (5.112) establishes that for all $j \in \{ 0 , 1 , \dots , N \} , t \in f _ { j } ^ { - 1 } ( \{ 0 \} )$ it holds that

$$
\begin{array} { r } { \mathbf { h } ( r ( t ) , x ( t ) ) = f _ { j } ( t ) + \mathfrak { e } _ { j } ( r ( t ) ) = \mathfrak { e } _ { j } ( r ( t ) ) . } \end{array}\tag{5.115}
$$

Combining (5.67) and (5.77) therefore demonstrates that for all $j \in \{ 0 , 1 , \dots , N \} , \ t \in f _ { j } ^ { - 1 } ( \{ 0 \} )$ with $G ( r ( t ) , x ( t ) ) > 0$ it holds that

$$
| w ( t ) - ( \mathfrak { e } _ { j } ) ^ { \prime } ( r ( t ) ) | = \big | ( \partial _ { r } \mathbf { h } ) ( r ( t ) , x ( t ) ) - ( \mathfrak { e } _ { j } ) ^ { \prime } ( r ( t ) ) \big | \le \beta _ { j } ( r ( t ) ) G ( r ( t ) , x ( t ) ) .\tag{5.116}
$$

Moreover, note that (5.45), (5.46), (5.77), and (5.78) ensure that for Lebesgue-almost all $t \in I$ it holds that

$$
\mathsf { c } G ( r ( t ) , x ( t ) ) = \mathsf { c } | p ( t ) | \leq \gamma ( t ) .\tag{5.117}
$$

This, (5.114), (5.116), the fact that ${ \mathsf c } > 0$ , and the fact that for all $j \in \{ 0 , 1 , \ldots , N \}$ it holds that $\beta _ { j }$ is nonnegative imply that for all $j \in \{ 0 , 1 , \ldots , N \}$ it holds that for Lebesgue-almost all $t \in f _ { j } ^ { - 1 } ( \{ 0 \} )$ with $G ( r ( t ) , x ( t ) ) > 0$ it holds that

$$
\alpha ( t ) ( { \gamma } ( t ) ) ^ { 2 } \le \beta _ { j } ( r ( t ) ) G ( r ( t ) , x ( t ) ) | r ^ { \prime } ( t ) | \le \frac { \beta _ { j } ( r ( t ) ) } { \mathtt { c } } { \gamma } ( t ) | r ^ { \prime } ( t ) | .\tag{5.118}
$$

Combining this, (5.78), and (5.79) shows that for all $j \in \{ 0 , 1 , \ldots , N \}$ it holds that for Lebesgue-almost all $t \in f _ { j } ^ { - 1 } ( \{ 0 \} )$ it holds that

$$
\ell ( t ) \leq \frac { \Lambda } { \mathsf { c } } \beta _ { j } ( r ( t ) ) | r ^ { \prime } ( t ) | .\tag{5.119}
$$

This, (5.94), (5.112), and the fact that for all $j \in \{ 0 , 1 , \ldots , N \}$ it holds that $\beta _ { j }$ is nonnegative prove that for all $a , b \in I$ with $a < b$ it holds that

$$
\int _ { a } ^ { b } \ell ( t ) \mathbf { 1 } _ { \Gamma } ( r ( t ) , x ( t ) ) \mathrm { d } t \leq { \frac { \Lambda } { c } } \int _ { a } ^ { b } \sum _ { j = 0 } ^ { N } \beta _ { j } ( r ( t ) ) | r ^ { \prime } ( t ) | \mathrm { d } t .\tag{5.120}
$$

Combining this, (5.68), (5.69), (5.79), (5.109), (5.111), the fact that r is locally absolutely continuous, the fundamental theorem of calculus for absolutely continuous functions, and the extreme value theorem establishes that for all $a , b \in I$ with $a < b$ it holds that

$$
\begin{array} { r l } & { Q ( \tau ( t ) , \tau ( s ) ) - Q ( \tau ( s ) , \tau ( c ) ) } \\ & { \quad \le - \displaystyle \int _ { 0 } ^ { t } \boldsymbol { \hat { \sigma } } ( t ) \mathbf { 1 } _ { C ( \mathcal { C } ( s ) , \Psi ( s ) , \tau ( s ) ) , \mathbf { M } } \boldsymbol { ( \tau ( \hat { \sigma } ) , \boldsymbol { x } ( s ) ) } \mathrm { d } t + \displaystyle \int _ { 0 } ^ { s } \mathbf { 1 } _ { C ( \mathcal { C } ( s ) ) , \tau ( s ) } \boldsymbol { \hat { \sigma } } ( t ) \mathrm { d } t } \\ & { \quad = - \displaystyle \int _ { 0 } ^ { t } \boldsymbol { \hat { \sigma } } ( t ) \mathbf { d } t + \displaystyle \int _ { 0 } ^ { s } \boldsymbol { \hat { \sigma } } ( t ) \mathbf { 1 } _ { C } ( \boldsymbol { \hat { \sigma } } ( t ) , \boldsymbol { x } ( t ) ) \mathrm { d } t + \displaystyle \int _ { 0 } ^ { s } \mathbf { 1 } _ { C ( \mathcal { C } ( s ) ) , \tau ( s ) } \boldsymbol { \hat { \sigma } } ( t ) \mathrm { d } t } \\ & { \quad \le - \displaystyle \int _ { 0 } ^ { t } \boldsymbol { \hat { \sigma } } ( t ) \mathbf { 1 } \cdot \mathbf { 1 } _ { C } + \displaystyle \int _ { 0 } ^ { s } \int _ { 0 } ^ { t } \sum _ { t = 0 } ^ { N } \boldsymbol { \hat { \sigma } } ( t ) \mathrm { d } t + \displaystyle \int _ { 0 } ^ { s } \mathbf { 1 } _ { C } ( t ) \| \boldsymbol { \hat { \sigma } } ( t ) \| ^ { \tau } ( \boldsymbol { \hat { \sigma } } ) \| ^ { \tau } ( \boldsymbol { \hat { \sigma } } ) \| \mathrm { d } t } \\ &  \quad = - \displaystyle \int _ { 0 } ^ { t } \boldsymbol { \hat { \sigma } } ( t ) \mathbf { d } t + \displaystyle \int _ { 0 } ^ { s } \Big \langle \mathbf { 1 } \big ( \mathbf { r } ( t ) \boldsymbol { \hat { \sigma } } ( t ) \big ) + \frac { \mathbf { 1 } } { C } \sum _ { t = 0 } ^ { N } \boldsymbol { \hat { \sigma } } ( t  \end{array}\tag{5.121}
$$

This and the arbitrariness of $I , r , x , \alpha ,$ and A prove (5.74).

To prove (5.41), let $I \subseteq \mathbb { R }$ be an interval, let $r \colon I \to ( 0 , \eta )$ and $x \colon I  M$ be locally absolutely continuous, let α : $I  [ 0 , \infty )$ and A: $I  \mathbb { R } ^ { D \times D }$ be Lebesgue measurable, assume $\begin{array} { r l } { \bigcup _ { t \in I } \{ ( r ( t ) , x ( t ) ) \} \subseteq } & { { } } \end{array}$ O, and assume that for Lebesgue-almost all $t \in I$ and every $v \in T _ { x ( t ) } M$ it holds that

$$
A ( t ) v \in T _ { x ( t ) } M , \qquad \langle A ( t ) v , v \rangle \geq \mathsf { a } | v | ^ { 2 } , \qquad | A ( t ) v | \leq \mathsf { b } | v | ,\tag{5.122}
$$

$$
\begin{array} { r } { r ^ { \prime } ( t ) = - \alpha ( t ) ( r ( t ) ) ^ { 2 } ( \partial _ { r } h ) ( r ( t ) , x ( t ) ) , \qquad \mathrm { a n d } \qquad x ^ { \prime } ( t ) = - \alpha ( t ) A ( t ) ( \nabla _ { M } h ) ( r ( t ) , x ( t ) ) . } \end{array}\tag{5.123}
$$

In the following let $\lambda \colon I  [ 0 , \infty )$ satisfy that for all $t \in I$ it holds that

$$
\lambda ( t ) = \frac { \alpha ( t ) } { \varphi ^ { \prime } ( h ( r ( t ) , x ( t ) ) ) } = \alpha ( t ) \big ( 1 + ( h ( r ( t ) , x ( t ) ) ) ^ { 2 } \big ) ^ { 3 / 2 } .\tag{5.124}
$$

Observe that (5.124), the fact that h is continuous, the fact that α is nonnegative and Lebesgue measurable, and the fact that r and x are locally absolutely continuous demonstrate that r and x are continuous and that λ is nonnegative and Lebesgue measurable. In the next step we note that (5.42), (5.43), (5.123), and (5.124) ensure that for Lebesgue-almost all $t \in I$ it holds that

and

$$
\begin{array} { r l } & { { \boldsymbol r } ^ { \prime } ( t ) = - \alpha ( t ) ( r ( t ) ) ^ { 2 } ( \partial _ { r } h ) ( r ( t ) , \boldsymbol { x } ( t ) ) = - \lambda ( t ) ( r ( t ) ) ^ { 2 } ( \partial _ { r } \mathbf { h } ) ( r ( t ) , \boldsymbol { x } ( t ) ) } \\ & { { \boldsymbol x } ^ { \prime } ( t ) = - \alpha ( t ) A ( t ) ( \nabla _ { M } h ) ( r ( t ) , \boldsymbol { x } ( t ) ) = - \lambda ( t ) A ( t ) ( \nabla _ { M } \mathbf { h } ) ( r ( t ) , \boldsymbol { x } ( t ) ) . } \end{array}\tag{5.125}
$$

Combining this, (5.74), (5.122), the fact that λ is nonnegative and Lebesgue measurable, the fact that A is Lebesgue measurable, the fact that r and x are locally absolutely continuous, and the fact that $\textstyle \bigcup _ { t \in I } \{ ( r ( t ) , x ( t ) ) \} \subseteq \mathcal { O }$ implies that for all $a , b \in I$ with $a \leq b$ it holds that

$$
Q ( r ( b ) , x ( b ) ) - Q ( r ( a ) , x ( a ) ) \leq - \int _ { a } ^ { b } | x ^ { \prime } ( t ) | \mathrm { d } t + \int _ { a } ^ { b } m ( r ( t ) ) | r ^ { \prime } ( t ) | \mathrm { d } t .\tag{5.126}
$$

This and the arbitrariness of $I , r , x , \alpha ,$ , and A establish (5.41). The proof of Proposition 5.3 is thus complete. □

## 5.3 Upper bounds for the error in the Lyapunov-type estimate

Lemma 5.4. Let $\varrho , \mathfrak { C } \in ( 0 , \infty )$ , let $I \subseteq \mathbb { R }$ be an interval, let r : $I \to ( 0 , \infty )$ be locally absolutely continuous, for every $R \in ( 0 , \varrho )$ let $\nu _ { R } \colon B ( ( 0 , R ] ) \to [ 0 , \infty ]$ satisfy for all $B \in B ( ( 0 , R ] )$ that

$$
\nu _ { R } ( B ) = \int _ { r ^ { - 1 } ( B ) } | r ^ { \prime } ( t ) | \mathrm { d } t ,\tag{5.127}
$$

let v : $I  [ 0 , \infty )$ be Lebesgue measurable, assume for Lebesgue-almost every $t \in I$ that $| r ^ { \prime } ( t ) | \leq \mathfrak { v } ( t )$ , and assume for all $\rho \in ( 0 , \varrho )$ that $\int _ { r ^ { - 1 } ( [ \rho / 2 , \rho ] ) } { \mathfrak { v } } ( t ) \mathrm { d } t \leq { \mathfrak { C } } \rho$ . Then

(i) for all $\rho \in ( 0 , \varrho )$ it holds that

$$
\int _ { r ^ { - 1 } ( ( 0 , \rho ] ) } { \mathfrak { v } } ( t ) \mathrm { d } t \leq 2 { \mathfrak { C } } \rho ,\tag{5.128}
$$

(ii) for all $R \in ( 0 , \varrho )$ it holds that $\nu _ { R }$ is a finite measure, and

(iii) for all $R \in ( 0 , \varrho ) , \rho \in ( 0 , R ]$ it holds that $\nu _ { R } ( ( 0 , \rho ] ) \leq 2 \mathfrak { C } \rho$

Proof of Lemma ${ 5 . 4 } .$ Throughout this proof for every $\rho \in ( 0 , \varrho ) , j \in \mathbb { N } _ { 0 }$ let $\mathcal { T } _ { \rho , j } \subseteq I$ satisfy

$$
\mathcal { T } _ { \rho , j } = r ^ { - 1 } ( ( 2 ^ { - j - 1 } \rho , 2 ^ { - j } \rho ] ) .\tag{5.129}
$$

Observe that (5.129) and the fact that r is continuous show that for all $\rho \in ( 0 , \varrho )$ it holds that $\mathcal { T } _ { \rho , j }$ $j \in  { \mathbb { N } } _ { 0 }$ , are Lebesgue measurable, are pairwise disjoint, and satisfy that

$$
\begin{array} { r } { \bigcup _ { j \in \mathbb { N } _ { 0 } } \mathcal { T } _ { \rho , j } = r ^ { - 1 } ( ( 0 , \rho ] ) . } \end{array}\tag{5.130}
$$

Next we combine the assumption that for all $\rho \in ( 0 , \varrho )$ it holds that $\begin{array} { r } { \int _ { r ^ { - 1 } ( [ \rho / 2 , \rho ] ) } \mathfrak { v } ( t ) \mathrm { d } t \leq \mathfrak { C } \rho } \end{array}$ , (5.129), and the fact that v is nonnegative to obtain that for all $\rho \in ( 0 , \varrho ) , j \in \mathbb { N } _ { 0 }$ it holds that

$$
\int _ { \mathcal T _ { \rho , j } } \mathfrak v ( t ) \mathrm d t \leq \int _ { r ^ { - 1 } ( [ 2 ^ { - j - 1 } \rho , 2 ^ { - j } \rho ] ) } \mathfrak v ( t ) \mathrm d t \leq \mathfrak C 2 ^ { - j } \rho .\tag{5.131}
$$

Combining this, (5.130), and the fact that for all $\rho \in ( 0 , \varrho )$ it holds that $\mathcal { T } _ { \rho , j } , j \in \mathbb { N } _ { 0 }$ , are pairwise disjoint proves that for all $\rho \in ( 0 , \varrho )$ it holds that

$$
\int _ { r ^ { - 1 } ( ( 0 , \rho ] ) } { \mathfrak { v } } ( t ) \mathrm { d } t = \sum _ { j = 0 } ^ { \infty } \int _ { \mathcal { T } _ { \rho , j } } { \mathfrak { v } } ( t ) \mathrm { d } t \leq { \mathfrak { C } } \rho \sum _ { j = 0 } ^ { \infty } 2 ^ { - j } = 2 { \mathfrak { C } } \rho .\tag{5.132}
$$

This establishes item (i). Note that (5.127) and the fact that $r$ is Borel measurable demonstrate that for every $R \in ( 0 , \varrho )$ and every sequence $( B _ { j } ) _ { j \in  { \mathbb { N } } _ { 0 } }$ of pairwise disjoint Borel subsets of $( 0 , R ]$ it holds that $\nu _ { R } ( \emptyset ) = 0$ and

$$
\begin{array} { r } { \nu _ { R } \big ( \bigcup _ { j \in \mathbb { N } _ { 0 } } B _ { j } \big ) = \int _ { \bigcup _ { j \in \mathbb { N } _ { 0 } } r ^ { - 1 } ( B _ { j } ) } | r ^ { \prime } ( t ) | \mathrm { d } t = \sum _ { j = 0 } ^ { \infty } \int _ { r ^ { - 1 } ( B _ { j } ) } | r ^ { \prime } ( t ) | \mathrm { d } t = \sum _ { j = 0 } ^ { \infty } \nu _ { R } ( B _ { j } ) . } \end{array}\tag{5.133}
$$

Combining this, the assumption that for Lebesgue-almost all $t \in I$ it holds that $| r ^ { \prime } ( t ) | \leq \mathfrak { v } ( t )$ , (5.127), and (5.128) ensures that for all $R \in ( 0 , \varrho )$ it holds that $\nu _ { R }$ is a nonnegative Borel measure on $( 0 , R ]$ and satisfies for al $\rho \in ( 0 , R ]$ that

$$
\nu _ { R } ( ( 0 , \rho ] ) = \int _ { r ^ { - 1 } ( ( 0 , \rho ] ) } | r ^ { \prime } ( t ) | \mathrm { d } t \leq \int _ { r ^ { - 1 } ( ( 0 , \rho ] ) } \mathfrak { v } ( t ) \mathrm { d } t \leq 2 \mathfrak { C } \rho .\tag{5.134}
$$

This proves item (ii). This establishes item (iii). The proof of Lemma 5.4 is thus complete.

Lemma 5.5. Let $R \in ( 0 , \infty )$ and let $m \colon ( 0 , R ] \to [ 0 , \infty )$ be locally bounded, definable, and integrable $( c f .$ Definition 1.6). Then

(i) for every $S \in ( 0 , R ] , K \in ( 0 , \infty )$ and every finite measure µ: ${ \cal B } ( ( 0 , S ] )  [ 0 , \infty ]$ with the property that for all $\rho \in \left( 0 , S \right]$ it holds that $\mu ( ( 0 , \rho ] ) \leq K \rho$ we have for every non-increasing integrable $w \colon ( 0 , S ] \to [ 0 , \infty )$ that

$$
\int _ { 0 } ^ { S } w ( s ) \mu ( \mathrm { d } s ) \leq K \int _ { 0 } ^ { S } w ( s ) \mathrm { d } s ,\tag{5.135}
$$

(ii) there exist $\delta \in ( 0 , R )$ and a definable, non-increasing, and integrable ψ : $( 0 , \delta ]  [ 0 , \infty )$ such that for all $s \in ( 0 , \delta ]$ it holds that

$$
m ( s ) \leq \psi ( s ) ,\tag{5.136}
$$

(iii) for every $\kappa \in ( 0 , \infty ) , \eta \in ( 0 , R ]$ and every finite measure ν : $B ( ( 0 , R ] )  [ 0 , \infty ]$ with the property that for all $\rho \in ( 0 , \eta ]$ it holds that $\nu ( ( 0 , \rho ] ) \leq \kappa \rho$ we have that

$$
\int _ { 0 } ^ { R } m ( s ) \nu ( \mathrm { d } s ) < \infty ,\tag{5.137}
$$

and

(iv) for every $\delta \in ( 0 , R ) , \ \varrho , \mathfrak { C } \in ( 0 , \infty )$ , every definable, non-increasing, and integrable $\psi \colon ( 0 , \delta ] \to$ $[ 0 , \infty )$ , every interval $I \subseteq \mathbb { R }$ , and every locally absolutely continuous $r \colon I \  \ ( 0 , \infty )$ with the property that for all $s \in ( 0 , \delta ]$ it holds that $m ( s ) \leq \psi ( s )$ , and with the property that for all $u \in ( 0 , \varrho )$ it holds that

$$
\int _ { r ^ { - 1 } ( [ u / 2 , u ] ) } | r ^ { \prime } ( t ) | \mathrm { d } t \leq \mathfrak { C } u\tag{5.138}
$$

we have for all $S \in ( 0 , \delta ] \cap ( 0 , \varrho )$ that

$$
\int _ { r ^ { - 1 } ( ( 0 , S ] ) } m ( r ( t ) ) | r ^ { \prime } ( t ) | \mathrm { d } t \leq 2 \mathfrak { C } \int _ { 0 } ^ { S } \psi ( s ) \mathrm { d } s < \infty .\tag{5.139}
$$

Proof of Lemma 5.5. Throughout this proof let λ be Lebesgue measure on R. We first claim that for every $S \in ( 0 , R ] , K \in ( 0 , \infty )$ , every finite nonnegative Borel measure µ on $( 0 , S ]$ , and every non-increasing integrable function $w \colon ( 0 , S ] \to [ 0 , \infty )$ with the property that for all $\rho \in ( 0 , S ]$ it holds that $\mu ( ( 0 , \rho ] ) \leq K \rho$ we have that

$$
\int _ { ( 0 , S ] } w ( s ) \mathrm { d } \mu ( s ) \leq K \int _ { 0 } ^ { S } w ( s ) \mathrm { d } s .\tag{5.140}
$$

To prove (5.140), let $S \in ( 0 , R ] , K \in ( 0 , \infty )$ , let $\mu$ be a finite nonnegative Borel measure on (0, S] which satisfies for all $\rho \in ( 0 , S ]$ that

$$
\mu ( ( 0 , \rho ] ) \leq K \rho ,\tag{5.141}
$$

let $w \colon ( 0 , S ] \to [ 0 , \infty )$ be non-increasing and Lebesgue integrable, for every $a \in [ 0 , \infty )$ let $A _ { a } \subseteq ( 0 , S ]$ satisfy

$$
{ \cal A } _ { a } = \{ s \in ( 0 , S ] \colon w ( s ) > a \} ,\tag{5.142}
$$

and let $F \colon ( 0 , S ] \times [ 0 , \infty ) \to \{ 0 , 1 \}$ satisfy for all $( s , a ) \in ( 0 , S ] \times [ 0 , \infty )$ that

$$
F ( s , a ) = \mathbf { 1 } _ { [ 0 , w ( s ) ) } ( a ) .\tag{5.143}
$$

Observe that (5.142) and the fact that w is non-increasing imply that w is Borel measurable and that for every $a \in [ 0 , \infty )$ there exists $b \in [ 0 , S ]$ such that

$$
( 0 , b ) \subseteq A _ { a } \subseteq ( 0 , b ] \qquad { \mathrm { a n d ~ } } \qquad \lambda ( A _ { a } ) = b .\tag{5.144}
$$

Combining this, (5.141), and the fact that $\mu ( \varnothing ) = 0$ shows that for all $a \in [ 0 , \infty )$ it holds that

$$
\mu ( A _ { a } ) \leq K \lambda ( A _ { a } ) .\tag{5.145}
$$

Furthermore, note that (5.142), (5.143), and the fact that w is Borel measurable demonstrate that $F$ is nonnegative and measurable and that for all $( s , a ) \in ( 0 , S ] \times [ 0 , \infty )$ it holds that

$$
F ( s , a ) = \mathbf { 1 } _ { A _ { a } } ( s ) .\tag{5.146}
$$

Next, observe that (5.143) and the fact that for all $s \in ( 0 , S ]$ it holds that $w ( s ) \in [ 0 , \infty )$ ensure that for all $s \in ( 0 , S ]$ it holds that

$$
\int _ { 0 } ^ { \infty } F ( s , a ) \mathrm { d } a = \lambda ( [ 0 , w ( s ) ) ) = w ( s ) .\tag{5.147}
$$

This, (5.146), the fact that $F$ is nonnegative and measurable, the fact that $\mu$ is finite and $\lambda | _ { [ 0 , \infty ) }$ is sigma-finite, and Tonelli’s theorem prove that

$$
\begin{array} { r } { \int _ { ( 0 , S ] } w ( s ) \mathrm { d } \mu ( s ) = \int _ { ( 0 , S ] } \left( \int _ { 0 } ^ { \infty } F ( s , a ) \mathrm { d } a \right) \mathrm { d } \mu ( s ) = \int _ { 0 } ^ { \infty } \left( \int _ { ( 0 , S ] } F ( s , a ) \mathrm { d } \mu ( s ) \right) \mathrm { d } a = \int _ { 0 } ^ { \infty } \mu ( A _ { a } ) \mathrm { d } a . } \end{array}\tag{5.148}
$$

In addition, note that (5.146), (5.147), the fact that F is nonnegative and measurable, the fact that $\lambda | _ { ( 0 , S ] }$ is finite and $\lambda | _ { [ 0 , \infty ) }$ is sigma-finite, and Tonelli’s theorem establish that

$$
\int _ { 0 } ^ { \infty } \lambda ( A _ { a } ) \mathrm { d } a = \int _ { 0 } ^ { \infty } \left( \int _ { 0 } ^ { S } F ( s , a ) \mathrm { d } s \right) \mathrm { d } a = \int _ { 0 } ^ { S } \left( \int _ { 0 } ^ { \infty } F ( s , a ) \mathrm { d } a \right) \mathrm { d } s = \int _ { 0 } ^ { S } w ( s ) \mathrm { d } s .\tag{5.149}
$$

Combining this, (5.145), (5.148), and the fact that w is integrable implies that

$$
\int _ { ( 0 , S ] } w ( s ) \mathrm { d } \mu ( s ) = \int _ { 0 } ^ { \infty } \mu ( A _ { a } ) \mathrm { d } a \leq K \int _ { 0 } ^ { \infty } \lambda ( A _ { a } ) \mathrm { d } a = K \int _ { 0 } ^ { S } w ( s ) \mathrm { d } s < \infty .\tag{5.150}
$$

This and the arbitrariness of $S , K , \mu ,$ and w prove item (i). Observe that the fact that m is definable and [9, Theorem 2.1] show that there exists $\varepsilon \in ( 0 , R )$ which satisfies that $m | _ { ( 0 , \varepsilon ) }$ is non-increasing or non-decreasing. In the following let $\delta \in ( 0 , \varepsilon )$ and let ψ : $( 0 , \delta ]  [ 0 , \infty )$ satisfy for all $s \in ( 0 , \delta ]$ that

$$
\psi ( s ) = { \left\{ \begin{array} { l l } { m ( s ) } & { : m | _ { ( 0 , \varepsilon ) } { \mathrm { ~ i s ~ n o n - i n c r e a s i n g , } } } \\ { m ( \delta ) } & { : m | _ { ( 0 , \varepsilon ) } { \mathrm { ~ i s ~ n o t ~ n o n - i n c r e a s i n g } } } \end{array} \right. } .\tag{5.151}
$$

Note that (5.151), the fact that m is definable, the fact that $m | _ { ( 0 , \varepsilon ) }$ is non-increasing or non-decreasing, and the fact that $0 < \delta < \varepsilon < R$ demonstrate that ψ is definable and non-increasing and that for all $s \in ( 0 , \delta ]$ it holds that

$$
0 \leq m ( s ) \leq \psi ( s ) \leq m ( s ) + m ( \delta ) .\tag{5.152}
$$

Combining this, the assumption that m is integrable, and the fact that $\psi$ is nonnegative and nonincreasing ensures that

$$
\int _ { 0 } ^ { \delta } \psi ( s ) \mathrm { d } s \leq \left( \int _ { 0 } ^ { \delta } m ( s ) \mathrm { d } s \right) + \delta m ( \delta ) < \infty .\tag{5.153}
$$

This, (5.152), the fact that $\delta \in ( 0 , R )$ , and the fact that ψ is definable and non-increasing establish item (ii). We next prove item (iii). In the following let $\kappa \in ( 0 , \infty ) , \eta \in ( 0 , R ]$ , let $\nu$ be a finite nonnegative Borel measure on $( 0 , R ]$ which satisfies for all $\rho \in ( 0 , \eta ]$ that

$$
\begin{array} { r } { \nu ( ( 0 , \rho ] ) \leq \kappa \rho , } \end{array}\tag{5.154}
$$

and let $\zeta \in ( 0 , R ) , B \in [ 0 , \infty )$ satisfy

$$
\zeta = \operatorname* { m i n } \{ \delta , \eta \} \qquad \mathrm { a n d } \qquad B = \operatorname* { s u p } _ { s \in [ \zeta , R ] } m ( s ) .\tag{5.155}
$$

Observe that item (i), (5.152), (5.153), (5.154), (5.155), and the fact that $\psi | _ { ( 0 , \zeta ] }$ is non-increasing imply that

$$
\int _ { 0 } ^ { \zeta } m ( s ) \nu ( \mathrm { d } s ) \leq \int _ { 0 } ^ { \zeta } \psi ( s ) \nu ( \mathrm { d } s ) \leq \kappa \int _ { 0 } ^ { \zeta } \psi ( s ) \mathrm { d } s < \infty .\tag{5.156}
$$

Moreover, note that the assumption that m is locally bounded, the fact that $[ \zeta , R ]$ is a compact subset of (0, R], (5.155), and the fact that ν is finite prove that

$$
\int _ { ( \zeta , R ] } m ( s ) \mathrm { d } \nu ( s ) \leq B \nu ( ( 0 , R ] ) < \infty .\tag{5.157}
$$

Combining (5.156) and the arbitrariness of $\kappa , \eta ,$ and ν hence establishes item (iii). We finally prove item (iv). In the following let $\theta \in ( 0 , R ) , \varrho , \mathfrak { C } \in ( 0 , \infty )$ , let $\varphi \colon ( 0 , \theta ] \to [ 0 , \infty )$ be definable, non-increasing, and integrable and satisfy for all $s \in ( 0 , \theta ]$ that

$$
m ( s ) \leq \varphi ( s ) ,\tag{5.158}
$$

let $I \subseteq \mathbb { R }$ be an interval, let $r \colon I \to ( 0 , \infty )$ be locally absolutely continuous, and assume for all $u \in ( 0 , \varrho )$ it holds that

$$
\int _ { r ^ { - 1 } ( [ u / 2 , u ] ) } | r ^ { \prime } ( t ) | \mathrm { d } t \leq \mathfrak { C } u ,\tag{5.159}
$$

let $\sigma \in ( 0 , \theta ] \cap ( 0 , \varrho )$ , and let $L \colon B ( ( 0 , \sigma ] ) \to [ 0 , \infty ]$ satisfy for all $D \in B ( ( 0 , \sigma ] )$ that

$$
L ( D ) = \int _ { r ^ { - 1 } ( D ) } | r ^ { \prime } ( t ) | { \mathrm { d } } t .\tag{5.160}
$$

Observe that (5.159), (5.160), the fact that $r$ is locally absolutely continuous, and items (ii) and (iii) in Lemma 5.4 show that L is a finite nonnegative Borel measure on $( 0 , \sigma ]$ and that for all $\rho \in ( 0 , \sigma ]$ it holds that

$$
L ( ( 0 , \rho ] ) \leq 2 \mathfrak { C } \rho .\tag{5.161}
$$

In the next step we combine (5.160), the fact that m is nonnegative and Borel measurable, and the monotone convergence theorem to obtain that

$$
\int _ { r ^ { - 1 } ( ( 0 , \sigma ] ) } m ( r ( t ) ) | r ^ { \prime } ( t ) | \mathrm { d } t = \int _ { ( 0 , \sigma ] } m ( s ) \mathrm { d } L ( s ) .\tag{5.162}
$$

This, item (i), (5.158), (5.161), and the fact that $\varphi | _ { ( 0 , \sigma ] }$ is non-increasing and integrable demonstrate that

$$
\int _ { r ^ { - 1 } ( ( 0 , \sigma ] ) } m ( r ( t ) ) | r ^ { \prime } ( t ) | \mathrm { d } t \leq \int _ { 0 } ^ { \sigma } \varphi ( s ) L ( \mathrm { d } s ) \leq 2 \mathfrak { C } \int _ { 0 } ^ { \sigma } \varphi ( s ) \mathrm { d } s < \infty .\tag{5.163}
$$

Combining this and the arbitrariness of $\theta , \varphi , \varrho , \mathfrak { C } , I , r ,$ and σ proves item (iv). The proof of Lemma 5.5 is thus complete. □

## 5.4 Upper bounds for parts of the length of the radial component in the DE system

Proposition 5.6. Let $D \in \mathbb { N } , \ \Re , { \mathsf { a } } \in ( 0 , \infty ) , \ { \mathsf { b } } \in [ { \mathsf { a } } , \infty )$ , let $M \subseteq \mathbb { R } ^ { D }$ be a nonempty, compact, and definable C<sup>2</sup>-submanifold $o f \mathbb { R } ^ { D }$ , let $\Omega \subseteq ( 0 , \Re ) \times M$ be definable and open in $( 0 , \Re ) \times M ,$ and let $\boldsymbol { h } = ( h ( r , \boldsymbol { x } ) ) _ { ( r , \boldsymbol { x } ) \in \Omega } \in \mathbf { C } ^ { 1 } ( \Omega , \mathbb { R } )$ be definable $( c f .$ Definitions $1 . 5 , \ 1 . 6 , \ 2 . 2 ,$ and ${ \it 2 . 5 ) }$ . Then there exists $\mathfrak { C } \in \mathbb { R }$ such that for every interval $I \subseteq \mathbb { R }$ , all locally absolutely continuous $r \colon I  ( 0 , \Re )$ and x: $I  M _ { : }$ and all Lebesgue measurable α: $I  [ 0 , \infty )$ and $A \colon I  \mathbb { R } ^ { D \times D }$ with $\textstyle \bigcup _ { t \in I } \{ ( r ( t ) , x ( t ) ) \} \subseteq \Omega$ and with the property that for Lebesgue-almost every $t \in I$ and every $v \in T _ { x ( t ) } M$ it holds that

$$
A ( t ) v \in T _ { x ( t ) } M , \qquad \langle A ( t ) v , v \rangle \geq \mathsf { a } | v | ^ { 2 } , \qquad | A ( t ) v | \leq \mathsf { b } | v | ,\tag{5.164}
$$

$$
\begin{array} { r } { r ^ { \prime } ( t ) = - \alpha ( t ) ( r ( t ) ) ^ { 2 } ( \partial _ { r } h ) ( r ( t ) , x ( t ) ) , \qquad a n d \qquad x ^ { \prime } ( t ) = - \alpha ( t ) A ( t ) ( \nabla _ { M } h ) ( r ( t ) , x ( t ) ) } \end{array}\tag{5.165}
$$

we have for all $\rho \in ( 0 , \Re )$ that

$$
\int _ { r ^ { - 1 } ( [ \rho / 2 , \rho ] ) } \sqrt { | r ^ { \prime } ( t ) | ^ { 2 } + ( r ( t ) ) ^ { 2 } | x ^ { \prime } ( t ) | ^ { 2 } } \mathrm { d } t \leq \mathfrak { C } \rho\tag{5.166}
$$

(cf. Definitions 2.3 and 2.7).

Proof of Proposition 5.6. Throughout this proof let $K \subseteq \mathbb { R } ^ { D + 1 }$ and $\mathcal { W } \subseteq \mathbb { R } ^ { 2 } \times M$ satisfy

$$
K = [ 1 / 2 , 1 ] \times M \quad \quad { \mathrm { a n d } } \quad \quad { \mathcal { W } } = \{ ( \rho , s , y ) \in ( 0 , \Re ) \times ( 0 , \infty ) \times M : ( \rho s , y ) \in \Omega \} ,\tag{5.167}
$$

let $f \colon \mathcal { W } $ R and $H \colon \mathcal { W } \to [ 0 , \infty )$ satisfy that for all $( \rho , s , y ) \in \mathcal { W }$ it holds that

$$
f ( \rho , s , y ) = h ( \rho s , y ) \qquad \mathrm { a n d } \qquad H ( \rho , s , y ) = \sqrt { \rho ^ { 2 } \big ( ( \partial _ { r } h ) ( \rho s , y ) \big ) ^ { 2 } + | ( \nabla _ { M } h ) ( \rho s , y ) | ^ { 2 } } ,\tag{5.168}
$$

let D ⊆ (0, R) × K satisfy

$$
\mathcal { D } = \mathcal { W } \cap ( ( 0 , \Re ) \times K ) ,\tag{5.169}
$$

let $\mathcal { E } \subseteq ( 0 , \Re ) \times \mathbb { R }$ satisfy

$$
\mathcal { E } = \big \{ ( \rho , e ) \in ( 0 , \mathfrak { R } ) \times \mathbb { R } \colon \big [ \exists u \in K \colon ( ( \rho , u ) \in \mathcal { D } ) \wedge ( f ( \rho , u ) = e ) \big ] \big \} ,\tag{5.170}
$$

let $\kappa , \Lambda , \mathbf { B } \in ( 0 , \infty )$ satisfy $\kappa = \operatorname* { m i n } \{ 1 / 4 , \mathsf { a } \} , \Lambda = \operatorname* { m a x } \{ 1 , \mathsf { b } \}$ , and

$$
\mathbf { B } = { \frac { 1 } { 2 } } + \sum _ { k = 1 } ^ { D } { \binom { \operatorname* { s u p } } { y = ( y _ { 1 } , \dots , y _ { D } ) \in M } } y _ { k } - \operatorname* { i n f } _ { y = ( y _ { 1 } , \dots , y _ { D } ) \in M } y _ { k } { \Biggr ) } ,\tag{5.171}
$$

let $\mu \colon { \mathcal { E } } \to [ 0 , \infty )$ satisfy for all $( \rho , e ) \in \mathcal { E }$ that

$$
\mu ( \rho , e ) = \operatorname* { i n f } \big \{ H ( \rho , u ) \colon \big [ ( ( \rho , u ) \in \mathcal { D } ) \wedge ( f ( \rho , u ) = e ) \big ] \big \} ,\tag{5.172}
$$

and let $\nu \subseteq \mathcal { E }$ satisfy

$$
\mathcal { V } = \{ ( \rho , e ) \in \mathcal { E } \colon \mu ( \rho , e ) = 0 \} .\tag{5.173}
$$

Note that (5.167), the assumptions that M is nonempty, compact, and definable, and the assumptions that Ω is definable and open in $( 0 , \Re ) \times M$ ensure that K is nonempty, compact, and definable and that W is definable and open in $( 0 , \Re ) \times ( 0 , \infty ) \times M$ . This, (5.168), and the assumption that $h \in { \bf C } ^ { 1 } ( \Omega , \mathbb { R } )$ is definable imply that $f \in { \bf C } ^ { 1 } ( \mathcal { W } , \mathbb { R } )$ is definable. Next we combine the assumption that $h \in { \bf C } ^ { 1 } ( \Omega , \mathbb { R } )$ is definable, the assumptions that Ω is definable and open in $( 0 , \Re ) \times M$ , Lemma 2.8, and Lemma 2.9 to obtain that $\partial _ { r } h \colon \Omega \to$ R and $\nabla _ { M } h \colon \Omega \to \mathbb { R } ^ { D }$ are continuous and definable. Combining (5.167) and (5.168) therefore establishes that H is continuous and definable on W. Furthermore, observe that (5.169), (5.170), the fact that $K , \mathcal { W } ,$ and f are definable, and [9, Theorem 1.13] show that D and E are definable and that for all $( \rho , e ) \in \mathcal { E }$ it holds that

$$
\left\{ H ( \rho , u ) \colon \left[ ( ( \rho , u ) \in { \mathcal { D } } ) \wedge ( f ( \rho , u ) = e ) \right] \right\} \neq \emptyset .\tag{5.174}
$$

This, (5.172), the fact that H is finite and nonnegative, the fact that $\mathcal { D } , \ f ,$ and H are definable, and [9, Theorem 1.13] demonstrate that $\mu$ is finite and definable. Combining this, (5.172), the fact that $\mathcal { D } , \ f , \ H$ , and µ are definable, and [9, Theorem 3.1] proves that there exists a definable function $\psi = ( \psi _ { 0 } , \psi _ { 1 } , \dots , \psi _ { D } ) \colon \mathcal { E } \to K$ which satisfies that for all $( \rho , e ) \in \mathcal { E }$ it holds that

$$
( \rho , \psi ( \rho , e ) ) \in \mathcal { D } , \qquad f ( \rho , \psi ( \rho , e ) ) = e , \qquad \mathrm { a n d } \qquad \mu ( \rho , e ) \leq H ( \rho , \psi ( \rho , e ) ) ,\tag{5.175}
$$

and which satisfies that for all $( \rho , e ) \in \mathcal { E }$ with $\mu ( \rho , e ) > 0$ it holds that

$$
H ( \rho , \psi ( \rho , e ) ) < 2 \mu ( \rho , e ) .\tag{5.176}
$$

Next, note that (5.172), (5.173), the fact that $\mathcal { D } , \ f , \ H ,$ and $\mu$ are definable, and [9, Theorem 3.1] ensure that V is definable and that there exists a definable function $\chi = ( \chi _ { 0 } , \chi _ { 1 } , \ldots , \chi _ { D } ) \colon \{ ( \rho , \eta , e ) \in$ $( 0 , \Re ) \times ( 0 , \infty ) \times \Re \colon ( \rho , e ) \in \mathcal { V } \} \to K$ which satisfies that for all $( \rho , \eta , e ) \in ( 0 , \Re ) \times ( 0 , \infty ) \times \mathbb { R }$ with $( \rho , e ) \in \mathcal { V }$ it holds that

$$
( \rho , \chi ( \rho , \eta , e ) ) \in \mathcal { D } , \qquad f ( \rho , \chi ( \rho , \eta , e ) ) = e , \qquad \mathrm { a n d } \qquad H ( \rho , \chi ( \rho , \eta , e ) ) < \eta .\tag{5.177}
$$

This, the fact that $\chi$ is definable, and [9, Theorems 6.6 and 6.7] imply that there exists $P \in \mathbb { N }$ which satisfies that for all $\rho \in ( 0 , \Re ) , \eta \in ( 0 , \infty )$ there exists a definable partition $\mathcal { P }$ of $\{ e \in \mathbb { R } \colon ( \rho , e ) \in \mathcal { V } \}$ such that

(I) it holds that $\# \mathcal { P } \leq P$ and for all $J \in \mathcal { P }$ it holds that J is a singleton or an open interval,

(II) for all open intervals $J \in \mathcal { P }$ it holds that $\chi ( \rho , \eta , \cdot ) | _ { J } \in C ^ { 1 } ( J , K )$ , and

(III) for all open intervals $J ~ \in ~ \mathcal { P }$ there exists $\varsigma = ( \varsigma _ { 0 } , \varsigma _ { 1 } , \ldots , \varsigma _ { D } ) \in \{ - 1 , 1 \} ^ { D + 1 }$ such that for all $j \in \{ 0 , 1 , \dots , D \} , e \in J$ it holds that

$$
\varsigma _ { j } \left( \partial _ { e } \chi _ { j } \right) \left( \rho , \eta , e \right) \geq 0\tag{5.178}
$$

(cf. Definition 3.1). Combining this, (5.167), (5.171), the fact that $\chi \big ( \{ ( \rho , \eta , e ) \in ( 0 , \Re ) \times ( 0 , \infty ) \times $ $\mathbb { R } \colon ( \rho , e ) \in \mathcal { V } \} \subseteq K$ , the fundamental theorem of calculus, and the monotone convergence theorem establishes that for every $\rho \in ( 0 , \Re ) , \eta \in ( 0 , \infty ) , a , b \in \mathbb { R }$ with $a < b$ and $\{ \rho \} \times ( a , b ) \subseteq \mathcal { V }$ it holds that

$$
\int _ { a } ^ { b } \left| \left( \partial _ { e } \chi \right) \left( \rho , \eta , e \right) \right| \mathrm { d } e \leq \sum _ { k = 0 } ^ { D } \int _ { a } ^ { b } \left| \left( \partial _ { e } \chi _ { k } \right) \left( \rho , \eta , e \right) \right| \mathrm { d } e \leq P \mathbf { B } .\tag{5.179}
$$

$\eta \in ( 0 , \infty ) , a , b \in \mathbb { R }$ 7), th<sub>with</sub> $a < b$ rule,<sub>and</sub> $\{ \rho \} \times ( a , b ) \subseteq \mathcal { V }$ <sup>y–Schwarz</sup> <sup>inequality</sup> <sup>show</sup> <sup>that</sup> <sup>for</sup> <sup>al</sup><sub>it holds that for Lebesgue-almost all</sub> $e \in ( a , b )$ $\rho \in ( 0 , \Re )$ it holds that

$$
1 = \partial _ { e } \big ( f ( \rho , \chi ( \rho , \eta , e ) ) \big ) \leq H ( \rho , \chi ( \rho , \eta , e ) ) \big | \big ( \partial _ { e } \chi \big ) \big ( \rho , \eta , e \big ) \big | \leq \eta \big | \big ( \partial _ { e } \chi \big ) \big ( \rho , \eta , e \big ) \big | .\tag{5.180}
$$

Combining this and (5.179) demonstrates that for all ${ \boldsymbol { \rho } } \in ( 0 , \Re ) , \eta \in ( 0 , \infty ) , \boldsymbol { a } , \boldsymbol { b } \in \mathbb { R }$ with $a < b$ and $\{ \rho \} \times ( a , b ) \subseteq \mathcal { V }$ it holds that

$$
b - a \leq \eta \int _ { a } ^ { b } \lvert \left( \partial _ { e } \chi \right) \left( \rho , \eta , e \right) \rvert \mathrm { d } e \leq \eta P \mathbf { B } .\tag{5.181}
$$

This, the fact that $P \mathbf { B } \in ( 0 , \infty )$ , and the fact that V is definable prove that for all $\rho \in ( 0 , \Re )$ it holds that

$$
\# \{ e \in \mathbb { R } \colon ( \rho , e ) \in \mathcal { V } \} < \infty .\tag{5.182}
$$

In addition, observe that the fact that $\mathcal { E }$ is definable, the fact that for all $j \in \{ 0 , 1 , \ldots , D \}$ it holds that $\psi _ { j } : { \mathcal { E } }  \mathbb { R }$ is definable, and [9, Theorem 6.7] ensure that there exist definable partitions $\mathcal { T } _ { j }$

$j \in \{ 0 , 1 , \ldots , D \} ,$ of $\mathcal { E }$ which satisfy that for all $j \in \{ 0 , 1 , \ldots , D \} , \ T \in \mathcal { T } _ { j }$ it holds that $T$ is a $C ^ { 1 }$ submanifold of $\bar { \mathbb { R } } ^ { 2 }$ and

$$
\psi _ { j } | _ { T } \in { \bf C } ^ { 1 } ( T , \mathbb { R } ) .\tag{5.183}
$$

Combining the fact that E and V are definable and [9, Theorem $6 . 6 ]$ hence implies that there exists a cylindrical definable $C ^ { 2 }$ cell decomposition $\mathcal { P }$ of $\mathbb { R } ^ { 2 }$ which is compatible with $( 0 , \Re ) \times \mathbb { R } , \mathcal { E } , \mathcal { V } _ { : }$ and every element of $\mathcal { T } _ { i }$ for every $j \in \{ 0 , 1 , \ldots , D \}$ (cf. Definitions 3.2 and 3.3). In the following let $\mathcal { Q } \subseteq \mathcal { P }$ be the set of all $\boldsymbol { Q } \in \mathcal { P }$ with $Q \subseteq { \mathcal { E } }$ such that for all $\rho \in ( 0 , \Re )$ it holds that $\{ e \in \mathbb { R } \colon ( \rho , e ) \in Q \}$ is empty or an open interval. Note that the fact that $\mathcal { P }$ is a definable partition of $\mathbb { R } ^ { 2 }$ , the fact that for all $Q \in { \mathcal { Q } }$ it holds that $Q \in { \mathcal { P } }$ and $Q \subseteq { \mathcal { E } } ,$ the fact that for all $j \in \{ 0 , 1 , \ldots , D \}$ it holds that $\mathcal { T } _ { j }$ is a definable partition of $\mathcal { E } ,$ and the fact that for all $j \in \{ 0 , 1 , \ldots , D \} , T \in \mathcal { T } _ { j }$ it holds that $T$ is a union of members of $\mathcal { P }$ establish that for all $Q \in \mathcal { Q } , j \in \{ 0 , 1 , \ldots , D \}$ it holds that

$$
\exists T \in { \mathcal { T } } _ { j } : Q \subseteq T .\tag{5.184}
$$

This, (5.183), the fact that for all $Q \in { \mathcal { Q } }$ it holds that $Q \subseteq ( 0 , \Re ) \times \mathbb { R }$ , the fact that for all $Q \in { \mathcal { Q } }$ $\rho \in ( 0 , \Re )$ it holds that $\{ e \in \mathbb { R } \colon ( \rho , e ) \in Q \}$ is empty or an open interval, the fact that for all $\rho \in$ R it holds that $( \mathbb { R } \ni e \mapsto ( \rho , e ) \in \mathbb { R } ^ { 2 } ) \in C ^ { 1 } ( \mathbb { R } , \mathbb { R } ^ { 2 } )$ , and the chain rule show that for all $Q \in \mathcal { Q } , \rho \in ( 0 , \Re )$ with $\{ e \in \mathbb { R } \colon ( \rho , e ) \in Q \} \neq \emptyset$ it holds that

$$
\psi ( \rho , \cdot ) | _ { \{ e \in \mathbb { R } : \ ( \rho , e ) \in Q \} } \in C ^ { 1 } \big ( \{ e \in \mathbb { R } \colon ( \rho , e ) \in Q \} , \mathbb { R } ^ { D + 1 } \big ) .\tag{5.185}
$$

Combining the fact that for all $Q \in { \mathcal { Q } }$ it holds that $Q$ is definable, the fact that for all $j \in \{ 0 , 1 , \ldots , D \}$ it holds that $\psi _ { j } \colon { \mathcal { E } }  \mathbb { R }$ is definable, and the fact that derivatives of definable functions are definable therefore demonstrates that for all $Q \in \mathcal { Q } , j \in \left\{ 0 , 1 , \ldots , D \right\}$ it holds that $\left( Q \ni \left( \rho , e \right) \mapsto \left( \partial _ { e } \psi _ { j } \right) \left( \rho , e \right) \in \mathbb { R } \right)$ is definable. In the following for all $Q \in \mathcal { Q } , j \in \{ 0 , 1 , . . . , D \} , \tau \in \{ - 1 , 0 , 1 \}$ let $S _ { Q , j } ^ { \tau } \subseteq \mathsf { \bar { Q } }$ satisfy

$$
S _ { Q , j } ^ { \tau } = \big \{ ( \rho , e ) \in Q \colon \mathrm { s i g n } \big ( \big ( \partial _ { e } \psi _ { j } \big ) \big ( \rho , e \big ) \big ) = \tau \big \} .\tag{5.186}
$$

Observe that (5.186), the fact that for all $Q \in \mathcal { Q } , j \in \{ 0 , 1 , \ldots , D \}$ it holds that $( \rho , e ) \mapsto \left( \partial _ { e } \psi _ { j } \right) ( \rho , e )$ is definable on $Q ,$ the fact that $\mathcal { P }$ is finite, and [9, Theorem 6.6] prove that there exists a cylindrical definable $C ^ { 2 }$ cell decomposition R of $\mathbb { R } ^ { 2 }$ which is a refinement of $\mathcal { P }$ and which satisfies that for all $Q \in \mathcal { Q } , j \in \{ 0 , 1 , . . . , D \} , \tau \in \{ - 1 , 0 , 1 \}$ it holds that $\mathcal { R }$ is compatible with $S _ { Q , j } ^ { \tau }$ (cf. Definition 3.4). In foll

$$
{ \mathcal { A } } = \left\{ \left\{ \rho \in \mathbb { R } \colon \left[ \exists e \in \mathbb { R } \colon ( \rho , e ) \in U \right] \right\} \colon U \in \mathcal { R } \right\}\tag{5.187}
$$

and let $L \in \mathbb { N } , \mathfrak { C } \in ( 0 , \infty )$ satisfy

$$
L = 1 + \# \mathcal { R } \qquad \mathrm { a n d } \qquad \mathfrak { C } = \operatorname* { m a x } \{ 1 , 2 ( \Lambda / \kappa ) L \mathbf { B } \} .\tag{5.188}
$$

Note that (5.187) and the fact that $\mathcal { R }$ is compatible with $( 0 , \Re ) \times \mathbb { R }$ ensure that $\mathcal { A }$ is a definable partition of R and that for all $J \in { \mathcal { A } }$ it holds that J is a singleton or an open interval and that either $J \subseteq ( 0 , \Re )$ or $J \cap ( 0 , \Re ) = \emptyset$ . We claim that for every $J \in { \mathcal { A } }$ with $J \subseteq ( 0 , \Re )$ , every interval $I \subseteq \mathbb { R } .$ , all locally absolutely continuous $r \colon I  ( 0 , \Re )$ and $x \colon I \to M$ , and all Lebesgue measurable α : $I \to [ 0 , \infty ) , A \colon \bar { I } \to \mathbb { R } ^ { D \times D }$ and v: $I  [ 0 , \infty )$ with $\textstyle \bigcup _ { t \in I } \{ ( r ( t ) , x ( t ) ) \} \subseteq \Omega$ , with the property that for Lebesgue-almost all $t \in I$ and every $v \in T _ { x ( t ) } M$ it holds that

$$
\begin{array} { r } { A ( t ) v \in T _ { x ( t ) } M , \qquad \langle A ( t ) v , v \rangle \geq \mathsf { a } | v | ^ { 2 } , \qquad \mathrm { a n d } \qquad | A ( t ) v | \leq \mathsf { b } | v | , } \end{array}\tag{5.189}
$$

with the property that for Lebesgue-almost all $t \in I$ it holds that

$$
\begin{array} { r } { r ^ { \prime } ( t ) = - \alpha ( t ) ( r ( t ) ) ^ { 2 } ( \partial _ { r } h ) ( r ( t ) , x ( t ) ) \qquad \mathrm { a n d } \qquad x ^ { \prime } ( t ) = - \alpha ( t ) A ( t ) ( \nabla _ { M } h ) ( r ( t ) , x ( t ) ) , } \end{array}\tag{5.190}
$$

and with the property that for Lebesgue-almost all $t \in I$ it holds that

$$
\mathfrak { v } ( t ) = \sqrt { | r ^ { \prime } ( t ) | ^ { 2 } + ( r ( t ) ) ^ { 2 } | x ^ { \prime } ( t ) | ^ { 2 } }\tag{5.191}
$$

we have for all $\rho \in J , \tau , \theta \in I$ with $\tau < \theta$ that

$$
\int _ { [ \tau , \theta ] \cap r ^ { - 1 } ( [ \rho / 2 , \rho ] ) } \mathfrak { v } ( t ) \mathrm { d } t \leq \mathfrak { C } \rho .\tag{5.192}
$$

To prove (5.192), let $J \in \mathcal { A }$ satisfy $J \subseteq ( 0 , \Re )$ , let $I \subseteq \mathbb { R }$ be an interval, let $r \colon I \  \ ( 0 , \Re )$ and $x \colon I  M$ be locally absolutely continuous, let $\rho \in J , \tau , \theta \in I$ satisfy $\tau < \theta$ , and let $\alpha \colon I  [ 0 , \infty )$ $A \colon I  \mathbb { R } ^ { D \times D }$ , and ${ \mathfrak { v } } \colon I \to [ 0 , \infty )$ be Lebesgue measurable, assume that $\textstyle \bigcup _ { t \in I } \{ ( r ( t ) , x ( t ) ) \} \subseteq \Omega$ , assume that for Lebesgue-almost all $t \in I$ and every $v \in T _ { x ( t ) } M$ it holds that

$$
\begin{array} { r } { A ( t ) v \in T _ { x ( t ) } M , \qquad \langle A ( t ) v , v \rangle \geq \mathsf { a } | v | ^ { 2 } , \qquad \mathrm { a n d } \qquad | A ( t ) v | \leq \mathsf { b } | v | , } \end{array}\tag{5.193}
$$

assume that for Lebesgue-almost all $t \in I$ it holds that

$$
\begin{array} { r } { r ^ { \prime } ( t ) = - \alpha ( t ) ( r ( t ) ) ^ { 2 } ( \partial _ { r } h ) ( r ( t ) , x ( t ) ) \qquad \mathrm { a n d } \qquad x ^ { \prime } ( t ) = - \alpha ( t ) A ( t ) ( \nabla _ { M } h ) ( r ( t ) , x ( t ) ) , } \end{array}\tag{5.194}
$$

and assume that for Lebesgue-almost all $t \in I$ it holds that

$$
\mathfrak { v } ( t ) = \sqrt { | r ^ { \prime } ( t ) | ^ { 2 } + ( r ( t ) ) ^ { 2 } | x ^ { \prime } ( t ) | ^ { 2 } } .\tag{5.195}
$$

Observe that (5.182), (5.185), (5.186), (5.187), (5.188), the fact that $\mathcal { R }$ is compatible with $\mathcal { E }$ and $\nu ,$ and the fact that for all $Q \in \mathcal { Q } , ~ j \in \{ 0 , 1 , . . . , D \} , \omega \in \{ - 1 , 0 , 1 \}$ it holds that $\mathcal { R }$ is compatible with ${ \cal S } _ { { \cal Q } , j } ^ { \omega }$ imply that there exist $N \in \{ 1 , \ldots , L \}$ , a set $\mathcal { L } \subseteq \{ 0 , 1 , \ldots , N - 1 \}$ , definable functions $\mathfrak { e } _ { i } \colon J  \mathbb { R }$ $j \in \{ 1 , \dots , N - 1 \}$ , functions $\mathfrak { e } _ { 0 } , \mathfrak { e } _ { N } \colon J \to \mathbb { R } \cup \{ - \infty , \infty \}$ , and $\sigma _ { i k } \in \{ - 1 , 0 , 1 \} , i \in \mathcal { L } , k \in \{ 0 , \overset { \cdot } { 1 } , \dots , D \}$ , which satisfy

a) for all $\rho \in J$ it holds that

$$
\begin{array} { r l } & { \quad - \infty = \mathfrak { e } _ { 0 } ( \rho ) < \cdot \cdot \cdot < \mathfrak { e } _ { N } ( \rho ) = \infty , } \\ & { \quad \quad \quad \{ \mathfrak { e } \in \mathbb { R } \colon ( \rho , \mathfrak { e } ) \in \mathcal { V } \} \subseteq \{ \mathfrak { e } _ { j } ( \rho ) \colon j \in \{ 1 , \ldots , N - 1 \} \} , } \\ { \mathrm { a n d } \quad } & { \quad \{ \mathfrak { e } \in \mathbb { R } \colon ( \rho , \mathfrak { e } ) \in \mathcal { E } \} \backslash \{ \mathfrak { e } _ { j } ( \rho ) \colon j \in \{ 1 , \ldots , N - 1 \} \} = \bigcup _ { i \in \mathcal { L } } ( \mathfrak { e } _ { i } ( \rho ) , \mathfrak { e } _ { i + 1 } ( \rho ) ) , } \end{array}\tag{5.196}
$$

b) for all $j \in \{ 1 , \dots , N - 1 \} , \rho \in J$ it holds that ${ \mathfrak { e } } _ { j } ( \rho ) \in \mathbb { R }$

c) for all $i \in \mathcal { L } , \rho \in J$ it holds that $\psi ( \rho , \cdot ) | _ { ( \mathfrak { e } _ { i } ( \rho ) , \mathfrak { e } _ { i + 1 } ( \rho ) ) } \in C ^ { 1 } ( ( \mathfrak { e } _ { i } ( \rho ) , \mathfrak { e } _ { i + 1 } ( \rho ) ) , K )$ , and

d) for all $i \in \mathcal { L } , k \in \{ 0 , 1 , . . . , D \} , \rho \in J , e \in ( \mathfrak { e } _ { i } ( \rho ) , \mathfrak { e } _ { i + 1 } ( \rho ) )$ it holds that

$$
( \rho , e ) \in \mathcal { E } \backslash \mathcal { V } , \qquad \mu ( \rho , e ) > 0 , \qquad \mathrm { a n d } \qquad \mathrm { s i g n } \big ( \big ( \partial _ { e } \psi _ { k } \big ) \big ( \rho , e \big ) \big ) = \sigma _ { i k } .\tag{5.197}
$$

In the following let $\mathcal { O } _ { i } \subseteq J \times \mathbb { R } , i \in \mathcal { L } .$ , satisfy

$$
\mathcal { O } _ { i } = \{ ( \rho , e ) \in J \times \mathbb { R } \colon \mathfrak { e } _ { i } ( \rho ) < e < \mathfrak { e } _ { i + 1 } ( \rho ) \} ,\tag{5.198}
$$

and let $z _ { i } \colon \mathcal { O } _ { i } \to \mathbb { R } , i \in \mathcal { L } .$ satisfy that for all $i \in \mathcal { L } , ( \rho , e ) \in \mathcal { O } _ { i }$ it holds that

$$
\begin{array} { r } { z _ { i } ( \rho , e ) = 2 \sum _ { k = 0 } ^ { D } \sigma _ { i k } \psi _ { k } ( \rho , e ) . } \end{array}\tag{5.199}
$$

Note that (5.168), (5.175), (5.176), (5.197), (5.199), the chain rule, the fact that for all $i \in \mathcal { L } , ( \rho , e ) \in \mathcal { O } _ { i }$ it holds that $\big ( \big ( \partial _ { e } \psi _ { j } \big ) \big ( \rho , e \big ) \big ) _ { j = 1 } ^ { D } \in T _ { ( \psi _ { j } ( \rho , e ) ) _ { j = 1 } ^ { D } } M ,$ and the Cauchy–Schwarz inequality establish that for all $i \in \mathcal { L } , ( \rho , e ) \in \mathcal { O } _ { i } , ( s , y ) \in K , ( \nu _ { 1 } , \dots , \nu _ { D } ) \in \mathbb { R } ^ { D }$ with $\psi ( \rho , e ) = ( s , y )$ and $( \nu _ { 1 } , \ldots , \nu _ { D } ) = ( \nabla _ { M } h ) ( \rho s , y )$ it holds that

$$
\begin{array} { r l } & { 1 = \partial _ { e } \big ( f ( \rho , \psi ( \rho , e ) ) \big ) = \rho ( \partial _ { r } h ) ( \rho s , y ) \big ( \partial _ { e } \psi _ { 0 } \big ) \big ( \rho , e \big ) + \sum _ { k = 1 } ^ { D } \nu _ { k } \big ( \partial _ { e } \psi _ { k } \big ) \big ( \rho , e \big ) } \\ & { \quad \leq H ( \rho , \psi ( \rho , e ) ) | \big ( \partial _ { e } \psi \big ) \big ( \rho , e \big ) | \leq 2 \mu ( \rho , e ) \sum _ { k = 0 } ^ { D } | \big ( \partial _ { e } \psi _ { k } \big ) \big ( \rho , e \big ) | = \mu ( \rho , e ) \big ( \partial _ { e } z _ { i } \big ) \big ( \rho , e \big ) } \end{array}\tag{5.200}
$$

(cf. Definition 2.3). This, (5.167), (5.171), (5.197), (5.199), the fact that $\psi$ takes values in $K .$ , and the mean value theorem show that for all $i \in \mathcal { L } , \rho \in J , \xi , \zeta \in ( \mathfrak { e } _ { i } ( \rho ) , \mathfrak { e } _ { i + 1 } ( \rho ) )$ with $\xi < \zeta$ it holds that

$$
\begin{array} { r } { 0 < z _ { i } ( \rho , \zeta ) - z _ { i } ( \rho , \xi ) = 2 \sum _ { k = 0 } ^ { D } \lvert \psi _ { k } ( \rho , \zeta ) - \psi _ { k } ( \rho , \xi ) \rvert \leq 2 \mathbf { B } . } \end{array}\tag{5.201}
$$

Combining this and (5.199) demonstrates that for all $i \in \mathcal { L } , \rho \in J$ it holds that

$$
0 < \operatorname* { l i m } _ { e \mathcal { J } ^ { \bullet } { e _ { i + 1 } ( \rho ) } } z _ { i } ( \rho , e ) - \operatorname* { l i m } _ { e \searrow { e _ { i } ( \rho ) } } z _ { i } ( \rho , e ) \leq 2 \mathbf { B } .\tag{5.202}
$$

In the following let $\beta _ { i } ^ { - } , \beta _ { i } ^ { + } , d _ { i } \colon J  \mathbb { R } , i \in \mathcal { L }$ , satisfy that for all $i \in \mathcal { L } , \rho \in J$ it holds that

$$
\beta _ { i } ^ { - } ( \rho ) = \operatorname* { l i m } _ { e \searrow e _ { i } ( \rho ) } z _ { i } ( \rho , e ) , \qquad \beta _ { i } ^ { + } ( \rho ) = \operatorname* { l i m } _ { e \nearrow e _ { i + 1 } ( \rho ) } z _ { i } ( \rho , e ) , \qquad \mathrm { a n d } \qquad d _ { i } ( \rho ) = \beta _ { i } ^ { + } ( \rho ) - \beta _ { i } ^ { - } ( \rho )\tag{5.203}
$$

and let $q _ { i } \colon J \times \mathbb { R } \to \mathbb { R } , i \in \mathcal { L }$ , satisfy that for all $i \in \mathcal { L } , ( \rho , e ) \in J \times \mathbb { R }$ it holds that

$$
q _ { i } ( \rho , e ) = \left\{ \begin{array} { l l } { 0 } & { : e \leq \mathfrak { e } _ { i } ( \rho ) , } \\ { z _ { i } ( \rho , e ) - \beta _ { i } ^ { - } ( \rho ) } & { : \mathfrak { e } _ { i } ( \rho ) < e < \mathfrak { e } _ { i + 1 } ( \rho ) , } \\ { d _ { i } ( \rho ) } & { : e \geq \mathfrak { e } _ { i + 1 } ( \rho ) } \end{array} \right.\tag{5.204}
$$

Observe that (5.172), (5.175), $( 5 . 1 9 7 ) , ( 5 . 2 0 0 ) , ( 5 . 2 0 1 ) , ( 5 . 2 0 2 )$ , (5.203), and (5.204) prove that for all $i \in \mathcal { L } , \rho \in J$ it holds that $q _ { i } ( \rho , \cdot ) | _ { ( \mathfrak { e } _ { i } ( \rho ) , \mathfrak { e } _ { i + 1 } ( \rho ) ) } \in C ^ { 1 } ( ( \mathfrak { e } _ { i } ( \rho ) , \mathfrak { e } _ { i + 1 } ( \rho ) )$ , R) and that $q _ { i } ( \rho , \cdot )$ is continuous and non-decreasing on R, is strictly increasing on $( \mathfrak { e } _ { i } ( \rho ) , \mathfrak { e } _ { i + 1 } ( \rho ) )$ , satisfies

$$
\begin{array} { r } { \{ q _ { i } ( \rho , e ) \colon e \in \mathbb { R } \} \subseteq [ 0 , d _ { i } ( \rho ) ] \qquad \mathrm { a n d } \qquad \{ q _ { i } ( \rho , e ) \colon e \in ( \mathfrak { e } _ { i } ( \rho ) , \mathfrak { e } _ { i + 1 } ( \rho ) ) \} = ( 0 , d _ { i } ( \rho ) ) , } \end{array}\tag{5.205}
$$

and satisfies that for all $u \in K$ with $( \rho , u ) \in \mathcal { D }$ and $( \rho , f ( \rho , u ) ) \in \mathcal { O } _ { i }$ it holds that

$$
\left( \partial _ { e } q _ { i } \right) \left( \rho , f ( \rho , u ) \right) \geq { \frac { 1 } { \mu ( \rho , f ( \rho , u ) ) } } \geq { \frac { 1 } { H ( \rho , u ) } } > 0 .\tag{5.206}
$$

In the following let $E , w \colon I  \mathbb { R }$ and $\mathbf { g } \colon I \to  { \mathbb { R } } ^ { D }$ satisfy for all $t \in I$ that

$$
E ( t ) = h ( r ( t ) , x ( t ) ) , \qquad w ( t ) = ( \partial _ { r } h ) ( r ( t ) , x ( t ) ) , \qquad \mathrm { a n d } \qquad \mathbf { g } ( t ) = ( \nabla _ { M } h ) ( r ( t ) , x ( t ) ) ,\tag{5.207}
$$

let $\delta \in ( 0 , \infty )$ satisfy

$$
\delta = \frac { 1 } { 2 } \operatorname* { m i n } \bigl ( \{ 1 \} \cup \{ d _ { i } ( \rho ) \colon i \in \mathcal { L } \} \bigr ) ,\tag{5.208}
$$

let $\mathbf { q } _ { i } ^ { \varepsilon } : \mathbb { R }  \mathbb { R } , i \in \mathcal { L } , \varepsilon \in ( 0 , \delta )$ , satisfy for all $i \in \mathcal { L } , \varepsilon \in ( 0 , \delta ) , e \in$ R that

$$
\begin{array} { r } { \mathbf q _ { i } ^ { \varepsilon } ( e ) = \operatorname* { m i n } \{ d _ { i } ( \rho ) - \varepsilon , \operatorname* { m a x } \{ \varepsilon , q _ { i } ( \rho , e ) \} \} , } \end{array}\tag{5.209}
$$

let $\mathcal { T } _ { i } ^ { \varepsilon } \subseteq [ \tau , \theta ] , i \in \mathcal { L } , \varepsilon \in ( 0 , \delta )$ , satisfy for all $i \in \mathcal { L } , \varepsilon \in ( 0 , \delta )$ that

$$
T _ { i } ^ { \varepsilon } = \big \{ t \in [ \tau , \theta ] \colon \big [ ( \rho / 2 \leq r ( t ) \leq \rho ) \wedge ( \varepsilon < q _ { i } ( \rho , E ( t ) ) < d _ { i } ( \rho ) - \varepsilon ) \big ] \big \} ,\tag{5.210}
$$

and let $\mathcal { U } ^ { \varepsilon } \subseteq [ \tau , \theta ] , \varepsilon \in ( 0 , \delta )$ , satisfy for all $\varepsilon \in ( 0 , \delta )$ that

$$
\begin{array} { r } { \mathcal { U } ^ { \varepsilon } = \bigcup _ { i \in \mathcal { L } } \mathcal { T } _ { i } ^ { \varepsilon } . } \end{array}\tag{5.211}
$$

Note that (5.193), (5.194), $( 5 . 2 0 7 )$ , the fact that for all $t \in I$ it holds that $\mathbf { g } ( t ) \in T _ { x ( t ) } M$ , the fact that for all compact subintervals $[ a , b ] \subseteq I$ it holds that $\{ ( r ( t ) , x ( t ) ) \colon t \in [ a , b ] \}$ is a compact subset of Ω, the assumption that $h \in { \bf C } ^ { 1 } ( \Omega , \mathbb { R } )$ , and the chain rule ensure that E is locally absolutely continuous and non-increasing on $I ,$ and satisfies that for Lebesgue-almost all $t \in I$ it holds that

$$
\begin{array} { r l } & { - E ^ { \prime } ( t ) = - ( \partial _ { r } h ) ( r ( t ) , x ( t ) ) r ^ { \prime } ( t ) - \left. ( \nabla _ { M } h ) ( r ( t ) , x ( t ) ) , x ^ { \prime } ( t ) \right. } \\ & { \qquad = \alpha ( t ) \big ( ( r ( t ) ) ^ { 2 } ( w ( t ) ) ^ { 2 } + \left. A ( t ) \mathbf { g } ( t ) , \mathbf { g } ( t ) \right. \big ) \geq \alpha ( t ) \big ( ( r ( t ) ) ^ { 2 } ( w ( t ) ) ^ { 2 } + \mathsf { a } | \mathbf { g } ( t ) | ^ { 2 } \big ) \geq 0 . } \end{array}\tag{5.212}
$$

Moreover, observe that (5.168), (5.207), and the fact that $\kappa = \operatorname* { m i n } \{ 1 / 4 , \mathsf { a } \}$ imply that for all $t \in [ \tau , \theta ]$ with $\rho / 2 \leq r ( t ) \leq \rho$ it holds that

$$
\begin{array} { r l } & { ( r ( t ) ) ^ { 2 } ( w ( t ) ) ^ { 2 } + | \mathbf { g } ( t ) | ^ { 2 } \leq \big ( H ( \rho , r ( t ) / \rho , x ( t ) ) \big ) ^ { 2 } = \rho ^ { 2 } ( w ( t ) ) ^ { 2 } + | \mathbf { g } ( t ) | ^ { 2 } } \\ & { \qquad \leq 4 ( r ( t ) ) ^ { 2 } ( w ( t ) ) ^ { 2 } + | \mathbf { g } ( t ) | ^ { 2 } \leq \frac { 1 } { \kappa } \big ( ( r ( t ) ) ^ { 2 } ( w ( t ) ) ^ { 2 } + \mathsf { a } | \mathbf { g } ( t ) | ^ { 2 } \big ) . } \end{array}\tag{5.213}
$$

This, (5.212), and the fact that for all $t \in I$ it holds that $\alpha ( t ) \geq 0$ establish that for Lebesgue-almost all $t \in [ \tau , \theta ]$ with $\rho / 2 \leq r ( t ) \leq \rho$ it holds that

$$
- E ^ { \prime } ( t ) \geq \alpha ( t ) { \big ( } ( r ( t ) ) ^ { 2 } ( w ( t ) ) ^ { 2 } + \mathsf { a } | \mathbf { g } ( t ) | ^ { 2 } { \big ) } \geq \kappa \alpha ( t ) { \big ( } H ( \rho , r ( t ) / \rho , x ( t ) ) { \big ) } ^ { 2 } .\tag{5.214}
$$

In the next step we note that (5.193), (5.194), (5.195), (5.207), (5.213), the fact that for all $t \in I$ it holds that $\mathbf { g } ( t ) \in T _ { x ( t ) } M$ and $\alpha ( t ) \geq 0$ , and the fact that $\Lambda = \operatorname* { m a x } \{ 1 , \mathsf { b } \}$ show that for Lebesgue-almost all $t \in [ \tau , \theta ]$ with $\rho / 2 \leq r ( t ) \leq \rho$ it holds that

$$
\begin{array} { r l } & { \mathfrak { v } ( t ) = \alpha ( t ) \sqrt { ( r ( t ) ) ^ { 4 } ( w ( t ) ) ^ { 2 } + ( r ( t ) ) ^ { 2 } | A ( t ) \mathbf { g } ( t ) | ^ { 2 } } = \alpha ( t ) r ( t ) \sqrt { ( r ( t ) ) ^ { 2 } ( w ( t ) ) ^ { 2 } + | A ( t ) \mathbf { g } ( t ) | ^ { 2 } } } \\ & { \qquad \leq \alpha ( t ) r ( t ) \sqrt { ( r ( t ) ) ^ { 2 } ( w ( t ) ) ^ { 2 } + ( \mathfrak { b } ) ^ { 2 } | \mathbf { g } ( t ) | ^ { 2 } } \leq \Lambda \alpha ( t ) r ( t ) \sqrt { ( r ( t ) ) ^ { 2 } ( w ( t ) ) ^ { 2 } + | \mathbf { g } ( t ) | ^ { 2 } } } \\ & { \qquad \leq \Lambda \alpha ( t ) \rho H ( \rho , r ( t ) / \rho , x ( t ) ) . } \end{array}\tag{5.215}
$$

Next we combine the fact that $E | _ { [ \tau , \theta ] }$ is absolutely continuous and Lemma 4.1 (applied for every $j \in$ $\{ 1 , \ldots , N - 1 \}$ with $I \curvearrowleft [ \tau , \theta ] , c \curvearrowleft \Vert \rho ) , f \curvearrowleft E  _ { [ \tau , \theta ] }$ in the notation of Lemma 4.1) to obtain that for all $j \in \{ 1 , \dots , N - 1 \}$ it holds that

$$
\int _ { \{ t \in [ \tau , \theta ] : E ( t ) = \mathfrak { e } _ { j } ( \rho ) \} } | E ^ { \prime } ( t ) | \mathrm { d } t = 0 .\tag{5.216}
$$

Combining this and (5.212) demonstrates that for all $j \in \{ 1 , \ldots , N - 1 \}$ it holds that there exists a Lebesgue measurable set $\mathcal { Z } _ { j } ~ \subseteq ~ [ \tau , \theta ]$ of Lebesgue measure zero such that for all $t \in [ \tau , \theta ] \backslash \mathcal { Z } _ { j }$ with $E ( t ) = \mathfrak { e } _ { j } ( \rho )$ it holds that

$$
0 = - E ^ { \prime } ( t ) \geq \alpha ( t ) \big ( ( r ( t ) ) ^ { 2 } ( w ( t ) ) ^ { 2 } + \mathsf { a } | \mathbf { g } ( t ) | ^ { 2 } \big ) \geq 0 .\tag{5.217}
$$

This, (5.194), (5.207), the fact that $\mathsf { a } > 0$ , and the fact that for all $t \in I$ it holds that $r ( t ) > 0$ and $\alpha ( t ) \geq 0$ prove that for all $j \in \{ 1 , \dots , N - 1 \}$ it holds that there exists a Lebesgue measurable set $\mathcal { Z } _ { j } \subseteq [ \tau , \theta ]$ of Lebesgue measure zero such that for all $t \in [ \tau , \theta ] \backslash \mathcal { Z } _ { j }$ with $E ( t ) = \mathfrak { e } _ { j } ( \rho )$ it holds that r and x are diferentiable at t and that $\alpha ( t ) w ( t ) = 0$

$$
\alpha ( t ) \mathbf { g } ( t ) = 0 , \qquad r ^ { \prime } ( t ) = - ( r ( t ) ) ^ { 2 } \alpha ( t ) w ( t ) = 0 , \qquad \mathrm { a n d } \qquad x ^ { \prime } ( t ) = - A ( t ) \bigl ( \alpha ( t ) \mathbf { g } ( t ) \bigr ) = 0 .\tag{5.218}
$$

Combining (5.195), the fact that $E | _ { [ \tau , \theta ] }$ is continuous, and the fact that v is Lebesgue measurable hence ensures that for all $j \in \{ 1 , \dots , N - \mathrm { { \bar { 1 } } } \}$ it holds that

$$
\int _ { \{ t \in [ \tau , \theta ] : \ E ( t ) = \mathfrak { e } _ { j } ( \rho ) \} } \mathfrak { v } ( t ) \mathrm { d } t = 0 .\tag{5.219}
$$

Furthermore, observe that (5.202), (5.203), (5.204), (5.208), and (5.209) imply that for all $i \in \mathcal { L } , \varepsilon \in ( 0 , \delta )$ it holds that $\{ e \in \mathbb { R } \colon \varepsilon \leq q _ { i } ( \rho , e ) \leq d _ { i } ( \rho ) - \varepsilon \} \subseteq ( \mathfrak { e } _ { i } ( \rho ) , \mathfrak { e } _ { i + 1 } ( \rho ) )$ and is a compact interval and that $\mathbf { q } _ { i } ^ { \varepsilon }$ is globally Lipschitz and non-decreasing. This, (5.209), and (5.212) establish that for all $i \in \mathcal { L } , \varepsilon \in ( 0 , \delta )$ it holds that $\mathbf { q } _ { i } ^ { \varepsilon } \circ E$ is absolutely continuous and non-increasing on $[ \tau , \theta ]$ and that for Lebesgue-almost all $t \in [ \tau , \theta ]$ it holds that

$$
- { \frac { \mathrm { d } } { \mathrm { d } t } } \mathbf { q } _ { i } ^ { \varepsilon } ( E ( t ) ) \geq 0 .\tag{5.220}
$$

Next, note that (5.167), (5.168), (5.169), (5.198), (5.204), (5.207), (5.210), and the assumption that $\textstyle \bigcup _ { t \in I } \{ ( r ( t ) , x ( t ) ) \} \subseteq \Omega$ show that for all $i \in \mathcal { L } , \varepsilon \in ( 0 , \delta ) , t \in \mathcal { T } _ { i } ^ { \varepsilon }$ it holds that

$$
( \rho , r ( t ) / \rho , x ( t ) ) \in \mathcal { D } , \quad f ( \rho , r ( t ) / \rho , x ( t ) ) = E ( t ) , \quad \mathfrak { e } _ { i } ( \rho ) < E ( t ) < \mathfrak { e } _ { i + 1 } ( \rho ) , \quad \mathrm { a n d } \quad ( \rho , E ( t ) ) \in \mathcal { O } _ { i } [ \rho , X _ { i } ( t ) ] .\tag{5.221}
$$

Combining this and (5.206) demonstrates that for all $i \in \mathcal { L } , \varepsilon \in ( 0 , \delta ) , t \in \mathcal { T } _ { i } ^ { \varepsilon }$ it holds that

$$
\left( \partial _ { e } q _ { i } \right) \left( \rho , E ( t ) \right) \geq { \frac { 1 } { \mu ( \rho , E ( t ) ) } } \geq { \frac { 1 } { H ( \rho , r ( t ) / \rho , x ( t ) ) } } > 0 .\tag{5.222}
$$

In addition, observe that (5.209) and the fact that for all $i \in \mathcal { L }$ it holds that $q _ { i } ( \rho , \cdot ) | _ { ( \mathfrak { e } _ { i } ( \rho ) , \mathfrak { e } _ { i + 1 } ( \rho ) ) } \ \in$ $C ^ { 1 } ( ( \mathfrak { e } _ { i } ( \rho ) , \mathfrak { e } _ { i + 1 } ( \rho ) ) , \mathbb { R } )$ prove that for all $i \in \mathcal { L } , \varepsilon \in ( 0 , \delta ) , e \in ( \mathfrak { e } _ { i } ( \rho ) , \mathfrak { e } _ { i + 1 } ( \rho ) )$ with $\varepsilon < q _ { i } ( \rho , e ) < d _ { i } ( \rho ) - \varepsilon$ it holds that $\mathbf { q } _ { i } ^ { \varepsilon }$ is diferentiable at e and satisfies

$$
{ \bf q } _ { i } ^ { \varepsilon } ( e ) = q _ { i } ( \rho , e ) \qquad \mathrm { a n d } \qquad ( { \bf q } _ { i } ^ { \varepsilon } ) ^ { \prime } ( e ) = \bigl ( \partial _ { e } q _ { i } \bigr ) \bigl ( \rho , e \bigr ) .\tag{5.223}
$$

This, (5.210), (5.212), (5.214), (5.215), (5.221), (5.222), and the chain rule ensure that for all $i \in { \mathcal { L } }$ $\varepsilon \in ( 0 , \delta )$ it holds that for Lebesgue-almost all $t \in \mathcal { T } _ { i } ^ { \varepsilon }$ it holds that E and $\mathbf { q } _ { i } ^ { \varepsilon } \circ E$ are diferentiable at t and that

$$
- \frac { \mathrm { d } } { \mathrm { d } t } \mathbf { q } _ { i } ^ { \varepsilon } ( E ( t ) ) = \left( \partial _ { \epsilon } q _ { i } \right) \left( \rho , E ( t ) \right) \left( - E ^ { \prime } ( t ) \right) \geq \frac { - E ^ { \prime } ( t ) } { H \left( \rho , r ( t ) / \rho , x ( t ) \right) } \geq \kappa \alpha ( t ) H ( \rho , r ( t ) / \rho , x ( t ) ) \geq \frac { \kappa } { \Lambda \rho } \mathfrak { v } ( t ) .\tag{5.224}
$$

Combining (5.210), (5.211), (5.220), the fact that for all $\varepsilon \in ( 0 , \delta )$ it holds that $\mathcal { T } _ { i } ^ { \varepsilon } , i \in \mathcal { L }$ , are pairwise disjoint, and the fundamental theorem of calculus therefore implies that for all $\varepsilon \in ( 0 , \delta )$ it holds that

$$
\int _ { \mathcal { U } ^ { \varepsilon } } { \mathfrak { v } } ( t ) \mathrm { d } t \leq \frac { \Lambda \rho } { \kappa } \sum _ { i \in \mathcal { L } } \int _ { \tau } ^ { \theta } - \frac { \mathrm { d } } { \mathrm { d } t } { \mathfrak { q } } _ { i } ^ { \varepsilon } ( E ( t ) ) \mathrm { d } t = \frac { \Lambda \rho } { \kappa } \sum _ { i \in \mathcal { L } } \bigl ( { \mathbf { \mathfrak { q } } } _ { i } ^ { \varepsilon } ( E ( \tau ) ) - { \mathbf { \mathfrak { q } } } _ { i } ^ { \varepsilon } ( E ( \theta ) ) \bigr ) .\tag{5.225}
$$

This, (5.188), (5.202), (5.203), and (5.209) establish that for all $\varepsilon \in ( 0 , \delta )$ it holds that

$$
\int _ { \mathcal { U } ^ { \varepsilon } } \mathfrak { v } ( t ) \mathrm { d } t \leq \frac { \Lambda \rho } { \kappa } \sum _ { i \in \mathcal { L } } d _ { i } ( \rho ) \leq \frac { 2 \Lambda } { \kappa } N \mathbf { B } \rho \leq \frac { 2 \Lambda } { \kappa } L \mathbf { B } \rho \leq \mathfrak { C } \rho .\tag{5.226}
$$

Moreover, note that (5.204), (5.210), and (5.211) show that for all $v , w \in ( 0 , \delta )$ with $w < v$ it holds that $\mathcal { U } ^ { v } \subseteq \mathcal { U } ^ { w }$ and that

$$
\begin{array} { r } { \bigcup _ { \varepsilon \in ( 0 , \delta ) } \mathcal { U } ^ { \varepsilon } = \big \{ t \in [ \tau , \theta ] \colon \big [ ( \rho / 2 \leq r ( t ) \leq \rho ) \wedge \big ( E ( t ) \in \bigcup _ { i \in \mathcal { L } } ( \mathfrak { e } _ { i } ( \rho ) , \mathfrak { e } _ { i + 1 } ( \rho ) ) \big ) \big ] \big \} . } \end{array}\tag{5.227}
$$

Combining this, item a), (5.219), (5.226), the monotone convergence theorem, and the fact that for all $t \in [ \tau , \theta ]$ with $\rho / 2 \leq r ( t ) \leq \rho$ it holds that $( \rho , r ( t ) / \rho , x ( t ) ) \in \mathcal { D }$ and $( \rho , E ( t ) ) \in \mathcal { E }$ demonstrates that

$$
\int _ { [ \tau , \theta ] \cap r ^ { - 1 } ( [ \rho / 2 , \rho ] ) } \mathfrak { v } ( t ) \mathrm { d } t \leq \mathfrak { C } \rho .\tag{5.228}
$$

This and the arbitrariness of $J , I , r , x , \alpha , A , \mathfrak { v } , \rho , \tau$ , and θ prove (5.192).

Observe that (5.192), the fact that A is compatible with (0, R), and the monotone convergence theorem establish (5.166). The proof of Proposition 5.6 is thus complete. □

## 5.5 Finite length of the angular component in the DE system

Corollary 5.7. Let $D \in \mathbb { N } , \ \Re , { \mathfrak { a } } \in ( 0 , \infty ) , \ { \mathfrak { b } } \in [ { \mathfrak { a } } , \infty )$ , let $M \subseteq \mathbb { R } ^ { D }$ be a nonempty, compact, and definable C<sup>2</sup>-submanifold of $\mathbb { R } ^ { D }$ , let ${ \mathcal { O } } \subseteq ( 0 , \Re ) \times M$ be definable and open in $( 0 , \Re ) \times M$ , and let $h = ( h ( r , x ) ) _ { ( r , x ) \in \mathcal { O } } \in \mathbf { C } ^ { 1 } ( \mathcal { O } , \mathbb { R } )$ be definable (cf. Definitions 1.5, $\it { 1 . 6 , } \ 2 . 2 ,  $ and 2.5). Then there exist $\delta \in ( 0 , \Re ) , \notin \in \mathbb { R }$ , and a definable, non-increasing, and integrable $\psi \colon ( 0 , \delta ] \  \ [ 0 , \infty )$ such that for every interval $I \subseteq \mathbb { R }$ , all locally absolutely continuous r : $I  ( 0 , \Re )$ and x: $I  M$ , and all Lebesgue measurable α: $I  [ 0 , \infty )$ and $\bar { \boldsymbol { A } } \colon \boldsymbol { I }  \mathbb { R } ^ { D \times D }$ with $\textstyle \bigcup _ { t \in I } \{ ( r ( t ) , x ( t ) ) \} \subseteq \mathcal { O }$ and with the property that for Lebesgue-almost every $t \in I$ and every $v \in T _ { x ( t ) } M$ it holds that

$$
A ( t ) v \in T _ { x ( t ) } M , \qquad \langle A ( t ) v , v \rangle \geq \mathsf { a } | v | ^ { 2 } , \qquad | A ( t ) v | \leq \mathsf { b } | v | ,\tag{5.229}
$$

$$
\begin{array} { r } { r ^ { \prime } ( t ) = - \alpha ( t ) ( r ( t ) ) ^ { 2 } ( \partial _ { r } h ) ( r ( t ) , x ( t ) ) , \qquad a n d \qquad x ^ { \prime } ( t ) = - \alpha ( t ) A ( t ) ( \nabla _ { M } h ) ( r ( t ) , x ( t ) ) } \end{array}\tag{5.230}
$$

we have that

(i) for all $R \in ( 0 , \delta ]$ with $r ( I ) \subseteq ( 0 , R ]$ it holds that

$$
\int _ { I } \sqrt { | r ^ { \prime } ( t ) | ^ { 2 } + | x ^ { \prime } ( t ) | ^ { 2 } } \mathrm { d } t \leq \mathfrak { C } + \mathfrak { C } \int _ { 0 } ^ { R } \psi ( s ) \mathrm { d } s < \infty ,\tag{5.231}
$$

and

(ii) for all $\tau \in \mathbb { R } , T \in ( \tau , \infty ]$ with $I = [ \tau , T )$ and lim su $\mathrm { p } _ { t  T } | r ( t ) | = 0$ it holds that

$$
\int _ { \tau } ^ { T } | x ^ { \prime } ( t ) | \mathrm { d } t < \infty\tag{5.232}
$$

(cf. Definitions 2.3 and 2.7).

Proof of Corollary 5.7. Note that Proposition 5.3 ensures that there exist $\eta \in ( 0 , \Re )$ , a bounded $Q \in$ $C ( \mathcal { O } \cap ( ( 0 , \eta ) \times M ) , [ 0 , \infty ) )$ , and a definable and integrable $m \in C ( ( 0 , \eta ) , [ 0 , \infty ) )$ ) which satisfy that

(I) for all $a , b \in ( 0 , \eta )$ with $a \leq b$ it holds that

$$
\operatorname* { s u p } _ { s \in [ a , b ] } m ( s ) < \infty \qquad \mathrm { a n d } \qquad \int _ { 0 } ^ { \eta } m ( s ) \mathrm { d } s < \infty\tag{5.233}
$$

and

(II) for every interval $I \subseteq \mathbb { R }$ , all locally absolutely continuous $r \colon I \to ( 0 , \eta )$ and $x \colon I  M .$ and all Lebesgue measurable $\alpha \colon I  [ 0 , \infty )$ and $A \colon I  \mathbb { R } ^ { D \times D }$ with $\begin{array} { r } { \bigcup _ { t \in I } \{ ( r ( t ) , x ( t ) ) \} \subseteq \mathcal { O } } \end{array}$ and with the property that for Lebesgue-almost all $t \in I$ and every $v \in T _ { x ( t ) } M$ it holds that

$$
A ( t ) v \in T _ { x ( t ) } M , \qquad \langle A ( t ) v , v \rangle \geq \mathsf { a } | v | ^ { 2 } , \qquad | A ( t ) v | \leq \mathsf { b } | v | ,\tag{5.234}
$$

$$
\begin{array} { r } { r ^ { \prime } ( t ) = - \alpha ( t ) ( r ( t ) ) ^ { 2 } ( \partial _ { r } h ) ( r ( t ) , x ( t ) ) , \qquad \mathrm { a n d } \qquad x ^ { \prime } ( t ) = - \alpha ( t ) A ( t ) ( \nabla _ { M } h ) ( r ( t ) , x ( t ) ) } \end{array}\tag{5.235}
$$

we have that for all $a , b \in I$ with $a \leq b$ it holds that

$$
Q ( r ( b ) , x ( b ) ) - Q ( r ( a ) , x ( a ) ) \leq - \int _ { a } ^ { b } | x ^ { \prime } ( t ) | \mathrm { d } t + \int _ { a } ^ { b } m ( r ( t ) ) | r ^ { \prime } ( t ) | \mathrm { d } t\tag{5.236}
$$

(cf. Definitions 2.3 and 2.7). Observe that Proposition 5.6 (applied with $\Omega \curvearrowleft \langle \mathcal { O }$ in the notation of Proposition 5.6) implies that there exist $\varrho \in ( 0 , \Re )$ and $\mathfrak { R } \in ( 0 , \infty )$ which satisfy that for every interval $I \subseteq \mathbb { R } .$ , all locally absolutely continuous $r \colon I  ( 0 , \Re )$ and $x \colon I  M$ , and all Lebesgue measurable α : $I  [ 0 , \infty )$ and A : $I  \mathbf { \mathbb { R } } ^ { D \times D }$ with $\begin{array} { r } { \bigcup _ { t \in I } \{ ( r ( t ) , x ( t ) ) \} \subseteq \mathcal { O } } \end{array}$ , with the property that for Lebesguealmost all $t \in I$ and every $v \in T _ { x ( t ) }$ M it holds that

$$
\begin{array} { r } { A ( t ) v \in T _ { x ( t ) } M , \qquad \langle A ( t ) v , v \rangle \geq \mathsf { a } | v | ^ { 2 } , \qquad \mathrm { a n d } \qquad | A ( t ) v | \leq \mathsf { b } | v | , } \end{array}\tag{5.237}
$$

and with the property that for Lebesgue-almost all $t \in I$ it holds that

$$
\begin{array} { r } { r ^ { \prime } ( t ) = - \alpha ( t ) ( r ( t ) ) ^ { 2 } ( \partial _ { r } h ) ( r ( t ) , x ( t ) ) \qquad \mathrm { a n d } \qquad x ^ { \prime } ( t ) = - \alpha ( t ) A ( t ) ( \nabla _ { M } h ) ( r ( t ) , x ( t ) ) } \end{array}\tag{5.238}
$$

we have for all $u \in ( 0 , \varrho )$ that

$$
\int _ { r ^ { - 1 } ( [ u / 2 , u ] ) } \sqrt { | r ^ { \prime } ( t ) | ^ { 2 } + ( r ( t ) ) ^ { 2 } | x ^ { \prime } ( t ) | ^ { 2 } } \mathrm { d } t \leq \Re u .\tag{5.239}
$$

Note that (5.236) shows that for every interval $I \subseteq \mathbb { R }$ , all locally absolutely continuous r : $I  ( 0 , \Re )$ and $x \colon I \to M$ , and all Lebesgue measurable α: $I  [ 0 , \infty )$ and $A \colon I  \mathbb { R } ^ { D \times D }$ with $\begin{array} { r } { \bigcup _ { t \in I } \{ ( r ( t ) , x ( t ) ) \} \subseteq \mathcal { O } } \end{array}$ with the property that for Lebesgue-almost all $t \in I$ and every $v \in T _ { x ( t ) } M$ it holds that

$$
\begin{array} { r } { A ( t ) v \in T _ { x ( t ) } M , \qquad \langle A ( t ) v , v \rangle \geq \mathsf { a } | v | ^ { 2 } , \qquad \mathrm { a n d } \qquad | A ( t ) v | \leq \mathsf { b } | v | , } \end{array}\tag{5.240}
$$

and with the property that for Lebesgue-almost all $t \in I$ it holds that

$$
\begin{array} { r } { r ^ { \prime } ( t ) = - \alpha ( t ) ( r ( t ) ) ^ { 2 } ( \partial _ { r } h ) ( r ( t ) , x ( t ) ) \qquad \mathrm { a n d } \qquad x ^ { \prime } ( t ) = - \alpha ( t ) A ( t ) ( \nabla _ { M } h ) ( r ( t ) , x ( t ) ) } \end{array}\tag{5.241}
$$

we have that for all $a , b \in I$ with $a < b$ and $r ( [ a , b ] ) \subseteq ( 0 , \eta )$ it holds that

$$
Q ( r ( b ) , x ( b ) ) - Q ( r ( a ) , x ( a ) ) \leq - \int _ { a } ^ { b } | x ^ { \prime } ( t ) | \mathrm { d } t + \int _ { a } ^ { b } m ( r ( t ) ) | r ^ { \prime } ( t ) | \mathrm { d } t .\tag{5.242}
$$

In the following let $\zeta , B \in \mathbb { R }$ satisfy

$$
\zeta = \frac { 1 } { 2 } \operatorname* { m i n } \{ \eta , \varrho \} \qquad \mathrm { a n d } \qquad B = \operatorname* { s u p } \bigl ( \{ 0 \} \cup Q ( \mathcal { O } \cap ( ( 0 , \eta ) \times M ) ) \bigr ) .\tag{5.243}
$$

Observe that (5.233), (5.243), the fact that $m | _ { ( 0 , \zeta ] }$ is nonnegative, locally bounded, and definable, and item (ii) in Lemma 5.5 (applied with $R \cap \zeta , m \cap m | _ { ( 0 , \zeta ] }$ in the notation of Lemma 5.5) demonstrate that there exist $\delta \in ( 0 , \zeta )$ and a definable, non-increasing, and integrable $\varphi \colon ( 0 , \delta ] \to [ 0 , \infty )$ which satisfy that for all $s \in ( 0 , \delta ]$ it holds that

$$
m ( s ) \leq \varphi ( s ) .\tag{5.244}
$$

Note that (5.243) and the fact that $Q$ is nonnegative prove that for all $( s , y ) \in \mathcal { O } \cap ( ( 0 , \eta ) \times M )$ it holds that

$$
0 \leq Q ( s , y ) \leq B .\tag{5.245}
$$

We first prove that for every interval $I \subseteq \mathbb { R }$ , all locally absolutely continuous r : $I  ( 0 , \Re )$ and $x \colon I \to M$ and all Lebesgue measurable $\alpha \colon I  [ 0 , \infty )$ and $\boldsymbol { A } \colon \boldsymbol { I } \to \mathbb { R } ^ { D \times D }$ with $\begin{array} { r } { \bigcup _ { t \in I } \{ ( r ( t ) , x ( t ) ) \} \subseteq \mathcal { O } } \end{array}$ , with the property that for Lebesgue-almost all $t \in I$ and every $v \in T _ { x ( t ) } M$ it holds that

$$
\begin{array} { r } { A ( t ) v \in T _ { x ( t ) } M , \qquad \langle A ( t ) v , v \rangle \geq \mathsf { a } | v | ^ { 2 } , \qquad \mathrm { a n d } \qquad | A ( t ) v | \leq \mathsf { b } | v | , } \end{array}\tag{5.246}
$$

and with the property that for Lebesgue-almost all $t \in I$ it holds that

$$
\begin{array} { r } { r ^ { \prime } ( t ) = - \alpha ( t ) ( r ( t ) ) ^ { 2 } ( \partial _ { r } h ) ( r ( t ) , x ( t ) ) \qquad \mathrm { a n d } \qquad x ^ { \prime } ( t ) = - \alpha ( t ) A ( t ) ( \nabla _ { M } h ) ( r ( t ) , x ( t ) ) } \end{array}\tag{5.247}
$$

we have that for every $R \in ( 0 , \delta ]$ with $\forall t \in I \colon r ( t ) \leq R$ it holds that

$$
\int _ { I } \sqrt { | r ^ { \prime } ( t ) | ^ { 2 } + | x ^ { \prime } ( t ) | ^ { 2 } } \mathrm { d } t \leq 2 \Re R + B + 2 \Re \int _ { 0 } ^ { R } \varphi ( s ) \mathrm { d } s < \infty .\tag{5.248}
$$

To prove (5.248), let $I \subseteq \mathbb { R }$ be an interval, let $r \colon I  ( 0 , \Re )$ and $x \colon I  M$ be locally absolutely continuous, let $\alpha \colon I \to [ 0 , \infty ) , A \colon I \to \mathbb { R } ^ { D \times D }$ , and v: $I  [ 0 , \infty )$ be Lebesgue measurable, assume that $\textstyle \bigcup _ { t \in I } \{ ( r ( t ) , x ( t ) ) \} \subseteq \mathcal { O }$ , assume that for Lebesgue-almost all $t \in I$ and all $v \in T _ { x ( t ) } M$ it holds that

$$
\begin{array} { r } { A ( t ) v \in T _ { x ( t ) } M , \qquad \langle A ( t ) v , v \rangle \geq \mathsf { a } | v | ^ { 2 } , \qquad \mathrm { a n d } \qquad | A ( t ) v | \leq \mathsf { b } | v | , } \end{array}\tag{5.249}
$$

and assume that for Lebesgue-almost all $t \in I$ it holds that

$$
\begin{array} { r l } & { r ^ { \prime } ( t ) = - \alpha ( t ) ( r ( t ) ) ^ { 2 } ( \partial _ { r } h ) ( r ( t ) , x ( t ) ) , \qquad x ^ { \prime } ( t ) = - \alpha ( t ) A ( t ) ( \nabla _ { M } h ) ( r ( t ) , x ( t ) ) , } \\ & { \qquad \mathrm { a n d } \qquad \mathfrak { v } ( t ) = \sqrt { | r ^ { \prime } ( t ) | ^ { 2 } + ( r ( t ) ) ^ { 2 } | x ^ { \prime } ( t ) | ^ { 2 } } . } \end{array}\tag{5.250}
$$

Observe that (5.239), (5.249), (5.250), and the fact that $\begin{array} { r } { \bigcup _ { t \in I } \{ ( r ( t ) , x ( t ) ) \} \ \subseteq \ \mathcal { O } } \end{array}$ establish that for Lebesgue-almost all $t \in I$ it holds that $| r ^ { \prime } ( t ) | \leq \mathfrak { v } ( t )$ and that for all $u \in ( 0 , \varrho )$ it holds that

$$
\int _ { r ^ { - 1 } ( [ u / 2 , u ] ) } { \mathfrak { v } } ( t ) \mathrm { d } t \leq \mathfrak { R } u .\tag{5.251}
$$

Combining this, (5.243), (5.250), the fact that $\delta < \zeta < \varrho .$ , the fact that r is locally absolutely continuous, and item (i) in Lemma 5.4 (applied with $\mathfrak { C } \frown \mathfrak { K } , r \frown \big ( I \ni t \mapsto r ( t ) \in ( 0 , \infty ) \big ) , \rho \ldots R$ in the notation of Lemma 5.4) ensures that for all $R \in ( 0 , \delta ]$ it holds that

$$
\int _ { r ^ { - 1 } ( ( 0 , R ] ) } | r ^ { \prime } ( t ) | \mathrm { d } t \leq \int _ { r ^ { - 1 } ( ( 0 , R ] ) } \mathfrak { v } ( t ) \mathrm { d } t \leq 2 \mathfrak { R } R .\tag{5.252}
$$

In the next step we combine (5.233), (5.243), (5.244), (5.250), (5.251), the fact that $m | _ { ( 0 , \zeta ] }$ is nonnegative, locally bounded, and definable, the fact that $\varphi$ is definable, non-increasing, and integrable, the fact that r is locally absolutely continuous, the fact that $\delta < \zeta < \varrho .$ , the monotonicity of the Lebesgue integral, and item (iv) in Lemma 5.5 (applied with $R \curvearrowleft , m \curvearrowleft m | _ { ( 0 , \zeta ] } , \mathfrak { C } \curvearrowleft , \mathfrak { R } , \psi \curvearrowleft \varphi , r  ( I  \ni t  \mapsto r ( t ) \in ( 0 , \infty ) )$ $S \curvearrowleft R$ in the notation of Lemma 5.5) to obtain that for all $R \in ( 0 , \delta ]$ it holds that

$$
\int _ { r ^ { - 1 } ( ( 0 , R ] ) } m ( r ( t ) ) | r ^ { \prime } ( t ) | \mathrm { d } t \leq 2 \Re \int _ { 0 } ^ { R } \varphi ( s ) \mathrm { d } s < \infty .\tag{5.253}
$$

This, (5.242), (5.245), the fact that $\begin{array} { r } { \bigcup _ { t \in I } \{ ( r ( t ) , x ( t ) ) \} \subseteq \mathcal { O } } \end{array}$ , the fact that $\delta < \eta ,$ and the fact that m is nonnegative imply that for every $R \in ( 0 , \delta ]$ with the property that for all $t \in I$ it holds that $r ( t ) \leq R$ we have for all $a , b \in I$ with $a < b$ that

$$
\int _ { a } ^ { b } | x ^ { \prime } ( t ) | \mathrm { d } t \leq Q ( r ( a ) , x ( a ) ) - Q ( r ( b ) , x ( b ) ) + \int _ { a } ^ { b } m ( r ( t ) ) | r ^ { \prime } ( t ) | \mathrm { d } t \leq B + 2 \Re \int _ { 0 } ^ { R } \varphi ( s ) \mathrm { d } s .\tag{5.254}
$$

Combining the fact that I is an interval and the monotone convergence theorem hence shows that for every $R \in ( 0 , \delta ]$ with the property that for all $t \in I$ it holds that $r ( t ) \leq R$ we have that

$$
\int _ { I } \lvert x ^ { \prime } ( t ) \rvert \mathrm d t \leq B + 2 \Re \int _ { 0 } ^ { R } \varphi ( s ) \mathrm d s < \infty .\tag{5.255}
$$

This, (5.252), and the fact that for all $a , b \in [ 0 , \infty )$ it holds that ${ \sqrt { a ^ { 2 } + b ^ { 2 } } } \leq a + b$ demonstrate that for every $R \in ( 0 , \delta ]$ with the property that for all $t \in I$ it holds that $r ( t ) \leq R$ we have that

$$
\int _ { I } \sqrt { | r ^ { \prime } ( t ) | ^ { 2 } + | x ^ { \prime } ( t ) | ^ { 2 } } \mathrm { d } t \leq \int _ { I } | r ^ { \prime } ( t ) | \mathrm { d } t + \int _ { I } | x ^ { \prime } ( t ) | \mathrm { d } t \leq 2 \Re R + B + 2 \Re \int _ { 0 } ^ { R } \varphi ( s ) \mathrm { d } s < \infty .\tag{5.256}
$$

Combining this and the arbitrariness of $I , \ r , \ x , \ \alpha ,$ and A proves (5.248). We next prove that for every interval $I \subseteq \mathbb { R }$ , all locally absolutely continuous r : $I  ( 0 , \Re )$ and $x \colon I  M$ , and all Lebesgue measurable $\alpha \colon I  [ 0 , \infty )$ and $A \colon I   { \mathbb { R } } ^ { \setminus D \times D }$ with $\textstyle \bigcup _ { t \in I } \{ ( r ( t ) , x ( t ) ) \} \subseteq \mathcal { O }$ , with the property that for Lebesgue-almost all $t \in I$ and every $v \in T _ { x ( t ) } M$ it holds that

$$
\begin{array} { r } { A ( t ) v \in T _ { x ( t ) } M , \qquad \langle A ( t ) v , v \rangle \geq \mathsf { a } | v | ^ { 2 } , \qquad \mathrm { a n d } \qquad | A ( t ) v | \leq \mathsf { b } | v | , } \end{array}\tag{5.257}
$$

and with the property that for Lebesgue-almost all $t \in I$ it holds that

$$
\begin{array} { r } { r ^ { \prime } ( t ) = - \alpha ( t ) ( r ( t ) ) ^ { 2 } ( \partial _ { r } h ) ( r ( t ) , x ( t ) ) \qquad \mathrm { a n d } \qquad x ^ { \prime } ( t ) = - \alpha ( t ) A ( t ) ( \nabla _ { M } h ) ( r ( t ) , x ( t ) ) } \end{array}\tag{5.258}
$$

we have that for every $\tau \in \mathbb { R } , T \in ( \tau , \infty ]$ with $I = [ \tau , T )$ and lim $\mathbf { \nabla } _ { t \nearrow T } r ( t ) = 0$ it holds that

$$
\int _ { \tau } ^ { T } | x ^ { \prime } ( t ) | \mathrm { d } t < \infty .\tag{5.259}
$$

To prove (5.259), let $I \subseteq \mathbb { R }$ be an interval, let $r \colon I  ( 0 , \Re )$ and $x \colon I  M$ be locally absolutely continuous, let $\alpha \colon I  [ 0 , \infty )$ and $A \colon I  \mathbb { R } ^ { D \times D }$ be Lebesgue measurable, let $\tau \in \mathbb { R } , T \in ( \tau , \infty ]$ , assume that $\textstyle \bigcup _ { t \in I } \{ ( r ( t ) , x ( t ) ) \} \subseteq \mathcal { O }$ , assume that

$$
I = [ \tau , T ) \qquad \mathrm { a n d } \qquad \operatorname * { l i m } _ { t \nearrow T } r ( t ) = 0 ,\tag{5.260}
$$

assume that for Lebesgue-almost all $t \in I$ and all $v \in T _ { x ( t ) } M$ it holds that

$$
\begin{array} { r } { A ( t ) v \in T _ { x ( t ) } M , \qquad \langle A ( t ) v , v \rangle \geq \mathsf { a } | v | ^ { 2 } , \qquad \mathrm { a n d } \qquad | A ( t ) v | \leq \mathsf { b } | v | , } \end{array}\tag{5.261}
$$

and assume that for Lebesgue-almost all $t \in I$ it holds that

$$
\begin{array} { r } { r ^ { \prime } ( t ) = - \alpha ( t ) ( r ( t ) ) ^ { 2 } ( \partial _ { r } h ) ( r ( t ) , x ( t ) ) \qquad \mathrm { a n d } \qquad x ^ { \prime } ( t ) = - \alpha ( t ) A ( t ) ( \nabla _ { M } h ) ( r ( t ) , x ( t ) ) . } \end{array}\tag{5.262}
$$

Note that (5.260), the fact that lim $\mathbf { \boldsymbol { \mathbf { \rho } } } _ { 1 _ { t } \nearrow T } \boldsymbol { r } ( t ) = 0$ , and the fact that $\delta > 0$ establish that there exists $\sigma \in [ \tau , T )$ such that for all $t \in [ \sigma , T )$ it holds that

$$
0 < r ( t ) \leq \delta .\tag{5.263}
$$

This, (5.248), (5.260), (5.261), (5.262), and the fact that $\textstyle \bigcup _ { t \in I } \{ ( r ( t ) , x ( t ) ) \} \subseteq \mathcal { O }$ ensure that there exists $\sigma \in [ \tau , T )$ which satisfies that

$$
\int _ { \sigma } ^ { T } | x ^ { \prime } ( t ) | \mathrm { d } t \leq \int _ { \sigma } ^ { T } \sqrt { | r ^ { \prime } ( t ) | ^ { 2 } + | x ^ { \prime } ( t ) | ^ { 2 } } \mathrm { d } t \leq 2 \Re \delta + B + 2 \Re \int _ { 0 } ^ { \delta } \varphi ( s ) \mathrm { d } s < \infty .\tag{5.264}
$$

Observe that (5.264) and the fact that x is locally absolutely continuous imply that

$$
\int _ { \tau } ^ { T } | x ^ { \prime } ( t ) | \mathrm { d } t = \int _ { \tau } ^ { \sigma } | x ^ { \prime } ( t ) | \mathrm { d } t + \int _ { \sigma } ^ { T } | x ^ { \prime } ( t ) | \mathrm { d } t < \infty .\tag{5.265}
$$

Combining this and the arbitrariness of $I , \ r , \ x , \ \alpha , \ A , \ \tau ,$ and T proves (5.259). In the following let $\mathfrak { C } = \operatorname* { m a x } \{ 2 \mathbb { A } , B \}$ , and let $\psi \colon ( 0 , \delta ] \to [ 0 , \infty )$ satisfy for all $s \in ( 0 , \delta ]$ that $\psi ( s ) = 1 + \varphi ( s )$ . Note that the fact that φ is definable, non-increasing, and integrable, the fact that for all $s \in ( 0 , \delta ]$ it holds that $\psi ( s ) = 1 + \varphi ( s )$ , the fact that $\delta \in ( 0 , \infty )$ , and the linearity of the integral show that ψ is definable, non-increasing, and integrable and that for all $R \in ( 0 , \delta ]$ it holds that $\begin{array} { r } { \int _ { 0 } ^ { R } \psi ( s ) \mathrm { d } s = R + \int _ { 0 } ^ { R } \varphi ( s ) \mathrm { d } s } \end{array}$ . This, (5.248), (5.259), the fact that $\mathfrak { C } = \operatorname* { m a x } \{ 2 \mathfrak { R } , B \} > 0$ , and the fact that φ is nonnegative establish items (i) and (ii). The proof of Corollary 5.7 is thus complete. □

## 5.6 Uniform upper bounds for the length of the angular component in the DE system

Proposition 5.8. Let $D , n \in \mathbb { N } , \ \Re , { \mathfrak { a } } \in ( 0 , \infty ) , \ \mathsf { b } \in [ \mathsf { a } , \infty )$ , let $M \subseteq \mathbb { R } ^ { D }$ be a nonempty, compact, connected, and definable n-dimensional $C ^ { 2 }$ -submanifold of $\mathbb { R } ^ { D }$ , and let $h \ = \ \big ( h ( r , x ) \big ) _ { ( r , x ) \in ( 0 , \Re ) \times M } \ \in$ $\mathbf { C } ^ { 1 } ( ( 0 , \Re ) \times M , \mathbb { R } )$ be definable (cf. Definitions $1 . 5 , \ 1 . 6 , \ 2 . 1 ,$ and 2.5). Then there exist $R \in ( 0 , \Re )$ $L \in \mathbb { R }$ such that for every interval $I \subseteq \mathbb { R }$ , all locally absolutely continuous $r \colon I  ( 0 , \Re )$ and $x \colon I \to M$ and all Lebesgue measurable α: $I  [ 0 , \infty )$ and A: $I  \mathbb { R } ^ { D \times D }$ with the property that for Lebesgue-almost every $t \in I$ and every $v \in T _ { x ( t ) } M$ it holds that

$$
A ( t ) v \in T _ { x ( t ) } M , \qquad \langle A ( t ) v , v \rangle \geq { \mathsf { a } } | v | ^ { 2 } , \qquad a n d \qquad | A ( t ) v | \leq { \mathsf { b } } | v | ,\tag{5.266}
$$

$$
\begin{array} { r } { r ^ { \prime } ( t ) = - \alpha ( t ) ( r ( t ) ) ^ { 2 } ( \partial _ { r } h ) ( r ( t ) , x ( t ) ) , \qquad a n d \qquad x ^ { \prime } ( t ) = - \alpha ( t ) A ( t ) ( \nabla _ { M } h ) ( r ( t ) , x ( t ) ) } \end{array}\tag{5.267}
$$

we have that

$$
\int _ { r ^ { - 1 } ( ( 0 , R ] ) } | x ^ { \prime } ( t ) | \mathrm { d } t \leq L\tag{5.268}
$$

(cf. Definitions 2.3 and 2.7).

Proof of Proposition 5.8. Observe that item (i) in Corollary 5.7 (applied with $\mathcal { O } \frown ( 0 , \Re ) \times M , R \curvearrowright \delta$ in the notation of Corollary 5.7) and Proposition 5.6 (applied with $\Omega \frown \left( 0 , \Re \right) \times M$ in the notation of Proposition 5.6) demonstrate that there exist $\delta , \varrho \in ( 0 , \Re ) , \mathfrak { C } \in ( 0 , \infty )$ , and $K \in [ 0 , \infty )$ such that for every interval $I \subseteq \mathbb { R }$ , all locally absolutely continuous $r \colon I  ( 0 , \Re )$ and $x \colon I  M .$ and all Lebesgue measurable α: $I  [ 0 , \infty )$ and $\mathbf { \bar { \Phi } } _ { A : I  \mathbb { R } ^ { \tilde { D } \times D } }$ with the property that for Lebesgue-almost all $t \in I$ and every $v \in T _ { x ( t ) } M$ it holds that

$$
\begin{array} { r } { A ( t ) v \in T _ { x ( t ) } M , \qquad \langle A ( t ) v , v \rangle \geq \mathsf { a } | v | ^ { 2 } , \qquad \mathrm { a n d } \qquad | A ( t ) v | \leq \mathsf { b } | v | , } \end{array}\tag{5.269}
$$

and with the property that for Lebesgue-almost all $t \in I$ it holds that

$$
\begin{array} { r } { r ^ { \prime } ( t ) = - \alpha ( t ) ( r ( t ) ) ^ { 2 } ( \partial _ { r } h ) ( r ( t ) , x ( t ) ) \qquad \mathrm { a n d } \qquad x ^ { \prime } ( t ) = - \alpha ( t ) A ( t ) ( \nabla _ { M } h ) ( r ( t ) , x ( t ) ) , } \end{array}\tag{5.270}
$$

we have that

(I) if for all $t \in I$ it holds that $r ( t ) \leq \delta ,$ , then

$$
\int _ { I } | x ^ { \prime } ( t ) | \mathrm { d } t \leq \int _ { I } \sqrt { | r ^ { \prime } ( t ) | ^ { 2 } + | x ^ { \prime } ( t ) | ^ { 2 } } \mathrm { d } t \leq K ,\tag{5.271}
$$

and

(II) for all $u \in ( 0 , \varrho )$ it holds that

$$
\int _ { r ^ { - 1 } ( [ u / 2 , u ] ) } | r ^ { \prime } ( t ) | \mathrm { d } t \leq \int _ { r ^ { - 1 } ( [ u / 2 , u ] ) } \sqrt { | r ^ { \prime } ( t ) | ^ { 2 } + ( r ( t ) ) ^ { 2 } | x ^ { \prime } ( t ) | ^ { 2 } } \mathrm { d } t \leq \mathfrak { C } u\tag{5.272}
$$

(cf. Definitions 2.3 and 2.7). In the following let $S \in ( 0 , \Re )$ and $L \in [ 0 , \infty )$ satisfy

$$
S = { \frac { 1 } { 2 } } \operatorname* { m i n } \{ \delta , \varrho \} \qquad { \mathrm { a n d } } \qquad L = ( 1 + 2 \mathfrak { C } ) K .\tag{5.273}
$$

We claim that for every interval $I \subseteq \mathbb { R }$ , all locally absolutely continuous r : $I  ( 0 , \Re )$ and $x \colon I \to M$ , and all Lebesgue measurable α : $I  [ 0 , \infty )$ and $\dot { A } \dot { : } I  \mathbb { R } ^ { D \times } \dot { D }$ with the property that for Lebesgue-almost all $t \in I$ and every $v \in T _ { x ( t ) } M$ it holds that

$$
\begin{array} { r } { A ( t ) v \in T _ { x ( t ) } M , \qquad \langle A ( t ) v , v \rangle \geq \mathsf { a } | v | ^ { 2 } , \qquad \mathrm { a n d } \qquad | A ( t ) v | \leq \mathsf { b } | v | , } \end{array}\tag{5.274}
$$

and with the property that for Lebesgue-almost all $t \in I$ it holds that

$$
\begin{array} { r } { r ^ { \prime } ( t ) = - \alpha ( t ) ( r ( t ) ) ^ { 2 } ( \partial _ { r } h ) ( r ( t ) , x ( t ) ) \qquad \mathrm { a n d } \qquad x ^ { \prime } ( t ) = - \alpha ( t ) A ( t ) ( \nabla _ { M } h ) ( r ( t ) , x ( t ) ) } \end{array}\tag{5.275}
$$

we have for all $a , b \in I$ with $a < b$ that

$$
\int _ { [ a , b ] \cap r ^ { - 1 } ( ( 0 , s / 2 ] ) } | x ^ { \prime } ( t ) | { \mathrm { d } } t \leq L .\tag{5.276}
$$

To prove this claim, let $I \subseteq \mathbb { R }$ be an interval, let $r \colon I  ( 0 , \Re )$ and $x \colon I  M$ be locally absolutely continuous, let α : $I  [ 0 , \infty )$ and $A \colon I  \mathbb { R } ^ { D \times D }$ be Lebesgue measurable, assume that for Lebesguealmost all $t \in I$ and every $v \in T _ { x ( t ) } M$ it holds that

$$
\begin{array} { r } { A ( t ) v \in T _ { x ( t ) } M , \qquad \langle A ( t ) v , v \rangle \geq \mathsf { a } | v | ^ { 2 } , \qquad \mathrm { a n d } \qquad | A ( t ) v | \leq \mathsf { b } | v | , } \end{array}\tag{5.277}
$$

assume that for Lebesgue-almost all $t \in I$ it holds that r and x are diferentiable at t and that

$$
\begin{array} { r } { r ^ { \prime } ( t ) = - \alpha ( t ) ( r ( t ) ) ^ { 2 } ( \partial _ { r } h ) ( r ( t ) , x ( t ) ) \qquad \mathrm { a n d } \qquad x ^ { \prime } ( t ) = - \alpha ( t ) A ( t ) ( \nabla _ { M } h ) ( r ( t ) , x ( t ) ) , } \end{array}\tag{5.278}
$$

and let $a , b \in I$ satisfy $a < b$ . In the following let $U , E \subseteq [ a , b ]$ satisfy

$$
U = \{ t \in [ a , b ] \colon r ( t ) < S \} \qquad { \mathrm { a n d } } \qquad E = [ a , b ] \cap r ^ { - 1 } ( ( 0 , { \cal S } / 2 ] ) .\tag{5.279}
$$

Note that the fact that $r | _ { [ a , b ] }$ is absolutely continuous and the fact that for every interval $J \subseteq \mathbb { R }$ and every absolutely continuous $\overset { \cdot } { f } : J \to \mathbb { R }$ it holds that f is continuous ensure that $r | _ { [ a , b ] }$ is continuous. Combining (5.279) and the fact that $( - \infty , S )$ is open therefore implies that U is open in $[ a , b ]$ . Next we combine (5.279) and the fact that $S > 0$ to obtain that

$$
E = \left\{ t \in [ a , b ] \colon 0 < r ( t ) \le { \mathit { s } } / { \mathit { 2 } } \right\} \subseteq \left\{ t \in [ a , b ] \colon r ( t ) < S \right\} = U .\tag{5.280}
$$

In the following let $W _ { t } \subseteq U , t \in E$ , satisfy for all $t \in E$ that

$$
W _ { t } = \bigcup \big \{ V \subseteq U \colon \big [ ( V { \mathrm { ~ i s ~ a n ~ i n t e r v a l } } ) \wedge ( t \in V ) \big ] \big \} .\tag{5.281}
$$

Observe that (5.280) and (5.281) prove that for all $t \in E$ it holds that $W _ { t }$ is an interval and that

$$
\{ t \} \subseteq W _ { t } \subseteq U .\tag{5.282}
$$

This and (5.281) show that for every $t \in E$ and every interval V with $W _ { t } \subseteq V \subseteq U$ it holds that

$$
V = W _ { t } .\tag{5.283}
$$

Furthermore, note that (5.281) and (5.282) establish that for every nonempty interval $J \subseteq U$ with the property that $J \cap E \neq \emptyset$ and with the property that for every interval V with $J \subseteq V \subseteq U$ it holds that $V = J$ we have for all $t \in J \cap E$ that

$$
J = W _ { t } .\tag{5.284}
$$

Combining (5.282), (5.283), and the fact that for all $t \in E$ it holds that $t \in W _ { t } \cap E$ hence demonstrates that there exists a set $\mathcal { J }$ which satisfies

$$
\begin{array} { r l } & { \mathcal { J } = \{ W _ { t } : t \in E \} } \\ & { \quad = \big \{ J \subseteq U : \big [ ( J \cap E \neq \emptyset ) \wedge ( J \mathrm { ~ i s ~ a n ~ i n t e r v a l } ) \wedge \big ( \forall V \mathrm { ~ i n t e r v a l } , \ J \subseteq V \subseteq U : V = J \big ) \big ] \big \} . } \end{array}\tag{5.285}
$$

In the following let $\mathcal { H } \subseteq \mathcal { L }$ satisfy

$$
{ \mathcal { H } } = \{ J \in { \mathcal { J } } \colon a \not \in J \} .\tag{5.286}
$$

Observe that (5.285) ensures that for all $J \in { \mathcal { J } }$ it holds that J is an interval. This and (5.285) imply that for all $J , V \in { \mathcal { L } }$ with $J \cap V \neq \emptyset$ it holds that $J = J \cup V = V$ . Hence, we obtain that for all $J , V \in { \mathcal { J } }$ with $J \neq V$ it holds that

$$
J \cap V = \emptyset .\tag{5.287}
$$

Next, note that (5.285) proves that for all $t \in E$ it holds that

$$
W _ { t } \in { \mathcal { J } } .\tag{5.288}
$$

Combining this and (5.282) shows that

$$
E \subseteq \bigcup _ { J \in { \mathcal { J } } } J .\tag{5.289}
$$

In addition, observe that (5.286) establishes that

$$
{ \mathcal { J } } \backslash { \mathcal { H } } = \{ J \in { \mathcal { J } } : a \in J \} .\tag{5.290}
$$

This, (5.287), and the fact that for all $J , V \in { \mathcal { J } } \backslash { \mathcal { H } }$ it holds that $a \in J \cap V$ demonstrate that for all $J , V \in { \mathcal { J } } \backslash { \mathcal { H } }$ it holds that $J = V$ . Combining this and (5.289) ensures that

$$
\begin{array} { r } { E \subseteq \bigcup _ { J \in { \mathcal { J } } } J \qquad \mathrm { a n d } \qquad \# ( { \mathcal { J } } \backslash { \mathcal { H } } ) \leq 1 . } \end{array}\tag{5.291}
$$

In the following let $\sigma _ { J } , \tau _ { J } \in [ a , b ] , J \in \mathcal { H } .$ satisfy for all $J \in \mathcal { H }$ that

$$
\sigma _ { J } = \operatorname* { i n f } J \qquad { \mathrm { a n d } } \qquad \tau _ { J } \in J \cap E .\tag{5.292}
$$

Note that (5.279), (5.285), (5.286), (5.292), and the fact that $S > 0$ imply that for all $J \in \mathcal { H }$ it holds that

$$
a < \tau _ { J } \leq b \qquad \mathrm { a n d } \qquad r ( \tau _ { J } ) \leq \frac { S } { 2 } < S .\tag{5.293}
$$

This, (5.279), and the fact that $r | _ { [ a , b ] }$ is continuous prove that for all $J \in \mathcal { H }$ there exists $s \in [ a , \tau _ { J } )$ such that

$$
[ s , \tau _ { J } ] \subseteq U .\tag{5.294}
$$

Combining (5.285), (5.286), and (5.292) therefore shows that for all $J \in \mathcal { H }$ there exists $s \in [ a , \tau _ { J } )$ such that

$$
[ s , \tau _ { J } ] \subseteq J .\tag{5.295}
$$

This and (5.292) establish that for all $J \in \mathcal { H }$ it holds that

$$
\sigma _ { J } < \tau _ { J } .\tag{5.296}
$$

Combining (5.279), (5.285), (5.286), and (5.292) hence demonstrates that for all $J \in \mathcal { H }$ it holds that

$$
a \leq \sigma _ { J } < \tau _ { J } \leq b , \quad \quad r ( \sigma _ { J } ) = S , \quad \quad r ( \tau _ { J } ) \leq \frac { S } { 2 } , \quad \quad \mathrm { a n d } \quad \quad ( \sigma _ { J } , \tau _ { J } ] \subseteq J .\tag{5.297}
$$

This, the fact that $r | _ { [ a , b ] }$ is continuous, and the intermediate value theorem ensure that for all $J \in \mathcal { H }$ it holds that $\{ t \in [ \sigma _ { J } , \dot { \tau } _ { J } ] \colon r ( t ) = s / { 2 } \}$ is nonempty and compact. In the following let $\mathsf { b } _ { J } \in [ a , b ] , \ J \in \mathcal { H } .$ satisfy for all $J \in \mathcal { H }$ that

$$
\mathfrak { b } _ { J } = \operatorname* { i n f } \{ t \in [ \sigma _ { J } , \tau _ { J } ] \colon r ( t ) = S / 2 \} .\tag{5.298}
$$

Observe that (5.297), (5.298), the fact that $r | _ { [ a , b ] }$ is continuous, and the intermediate value theorem imply that for all $J \in \mathcal { H }$ it holds that

$$
\sigma _ { J } < \mathfrak { b } _ { J } \leq \tau _ { J } , \qquad r ( \mathfrak { b } _ { J } ) = S / 2 , \qquad [ \sigma _ { J } , \mathfrak { b } _ { J } ] \subseteq r ^ { - 1 } ( [ S / 2 , S ] ) , \qquad \mathrm { a n d } \qquad ( \sigma _ { J } , \mathfrak { b } _ { J } ) \subseteq J .\tag{5.299}
$$

Combining this, (5.297), the fact that $r | _ { [ a , b ] }$ is absolutely continuous, the fundamental theorem of calculus for absolutely continuous functions, and the triangle inequality proves that for all $J \in \mathcal { H }$ it holds that

$$
\frac { S } { 2 } = \left| r ( \mathfrak { b } _ { J } ) - r ( \sigma _ { J } ) \right| = \left| \int _ { \sigma _ { J } } ^ { \mathfrak { b } _ { J } } r ^ { \prime } ( t ) \mathrm { d } t \right| \leq \int _ { \sigma _ { J } } ^ { \mathfrak { b } _ { J } } | r ^ { \prime } ( t ) | \mathrm { d } t .\tag{5.300}
$$

This, (5.272), (5.273), (5.299), and the fact that for all $J , V \in { \mathcal { H } }$ with $J \neq V$ it holds that $J \cap V = \emptyset$ show that for every finite set $\mathcal { F } \subseteq \mathcal { H }$ it holds that

$$
\frac { S } { 2 } ( \# \mathcal { F } ) = \sum _ { J \in \mathcal { F } } \frac { S } { 2 } \leq \sum _ { J \in \mathcal { F } } \int _ { \sigma _ { J } } ^ { \mathfrak { b } _ { J } } | r ^ { \prime } ( t ) | \mathrm { d } t \leq \int _ { r ^ { - 1 } ( [ s / 2 , S ] ) } | r ^ { \prime } ( t ) | \mathrm { d } t \leq \mathfrak { C } S .\tag{5.301}
$$

Combining this, (5.291), and the fact that $S > 0$ establishes that

$$
\# \mathcal { H } \leq 2 \mathfrak { C } \qquad \mathrm { a n d } \qquad \# \mathcal { J } = \# \mathcal { H } + \# ( \mathcal { J } \backslash \mathcal { H } ) \leq 2 \mathfrak { C } + 1 .\tag{5.302}
$$

Moreover, note that (5.279), (5.285), and the fact that I is an interval demonstrate that for all $J \in { \mathcal { J } }$ it holds that J is a nonempty interval and that

$$
J \subseteq U \subseteq [ a , b ] \subseteq I .\tag{5.303}
$$

This, (5.273), and (5.279) ensure that for all $J \in { \mathcal { J } }$ and all $t \in J$ it holds that

$$
0 < ( r | \boldsymbol { \ j } ( t ) ( t ) = r ( t ) < S \leq \delta .\tag{5.304}
$$

In the next step we observe that (5.278) and (5.303) imply that for all $J \in { \mathcal { J } }$ it holds that for Lebesguealmost all $t \in J$ it holds that $r , x , r | _ { J } ,$ and $x | _ { J }$ are diferentiable at t and that

$$
( r | \boldsymbol { \jmath } ) ^ { \prime } ( t ) = r ^ { \prime } ( t ) \qquad \mathrm { a n d } \qquad ( x | \boldsymbol { \jmath } ) ^ { \prime } ( t ) = x ^ { \prime } ( t ) .\tag{5.305}
$$

Next we combine (5.277) and (5.303) to obtain that for all $J \in { \mathcal { J } }$ it holds that for Lebesgue-almost all $t \in J$ it holds that for all $v \in T _ { ( x \mid J ) } M$ it holds that

$$
( A | \ j ) ( t ) v \in T _ { ( x | \ j ) ( t ) } M , \qquad \langle ( A | \ j ) ( t ) v , v \rangle \geq \mathsf { a } | v | ^ { 2 } , \qquad \mathrm { a n d } \qquad | ( A | \ j ) ( t ) v | \leq \mathsf { b } | v | .\tag{5.306}
$$

Furthermore, note that (5.278) and (5.305) prove that for all $J \in { \mathcal { J } }$ it holds that for Lebesgue-almost all $t \in J$ it holds that

$$
( r | \ j ) ^ { \prime } ( t ) = - ( \alpha | \ j ) ( t ) \big ( ( r | \ j ) ( t ) \big ) ^ { 2 } ( \partial _ { r } h ) ( ( r | \ j ) ( t ) , ( x | \ j ) ( t ) )\tag{5.307}
$$

$$
\begin{array} { r l } { \mathrm { a n d } \quad } & { ( x | \boldsymbol { J } ) ^ { \prime } ( t ) = - ( \alpha | \boldsymbol { J } ) ( t ) ( A | \boldsymbol { J } ) ( t ) ( \nabla _ { M } h ) ( ( r | \boldsymbol { J } ) ( t ) , ( x | \boldsymbol { J } ) ( t ) ) . } \end{array}\tag{5.308}
$$

Combining (5.271) (applied for every $J \in \mathcal { L }$ with $I \cap J , r \cap r | _ { J } , x \curvearrowright \neg { x } | _ { J } , \alpha \curvearrowright \neg { \alpha } | _ { J } , A \curvearrowright A | _ { J } )$ (5.303), (5.304), (5.306), the fact that for all $J \in { \mathcal { J } }$ it holds that $r \vert _ { J }$ and $x | _ { J }$ are locally absolutely continuous, and the fact that for all $J \in { \mathcal { J } }$ it holds that $\alpha | _ { J }$ and $A | _ { J }$ are Lebesgue measurable therefore shows that for all $J \in { \mathcal { J } }$ it holds that

$$
\int _ { J } | ( x | _ { J } ) ^ { \prime } ( t ) | \mathrm { d } t \leq \int _ { J } \sqrt { | ( r | _ { J } ) ^ { \prime } ( t ) | ^ { 2 } + | ( x | _ { J } ) ^ { \prime } ( t ) | ^ { 2 } } \mathrm { d } t \leq K .\tag{5.309}
$$

This and (5.305) establish that for all $J \in { \mathcal { J } }$ it holds that

$$
\int _ { J } | x ^ { \prime } ( t ) | { \mathrm { d } } t \leq K .\tag{5.310}
$$

Combining this, (5.273), (5.279), (5.291), and (5.302) demonstrates that

$$
\int _ { [ a , b ] \cap r ^ { - 1 } ( ( 0 , s / 2 ] ) } | x ^ { \prime } ( t ) | \mathrm { d } t = \int _ { E } | x ^ { \prime } ( t ) | \mathrm { d } t \leq \sum _ { J \in \mathcal { J } } \int _ { J } | x ^ { \prime } ( t ) | \mathrm { d } t \leq K ( \# \mathcal { J } ) \leq ( 1 + 2 \mathfrak { C } ) K = L .\tag{5.311}
$$

This and the fact that $I , r , x , \alpha , A , a .$ , and b were arbitrary subject to the stated assumptions prove (5.276).

In the following let $R = S / 2$ . Observe that (5.276), the fact that $R = S / 2 \in ( 0 , \Re )$ , and the monotone convergence theorem establish (5.268). The proof of Proposition 5.8 is thus complete. □

## 6 Spherical length at finite points and at infinity

In this section we prove in Theorem 6.8 in Subsection 6.4 below that on an arbitrary time interval with non-empty interior $I \subseteq \mathbb { R }$ we have that the spherical projection of every GF trajectory $\theta ( t ) , t \in I ,$ , of a definable $C ^ { 1 }$ -potential $F \colon U $ R defined on an open subset U of $\mathbb { R } ^ { n }$ has finite length. Theorem 1.1 in the introduction essentially coincides with Theorem 6.8.

In the first step of our proof of Theorem 6.8 we reduce Theorem 6.8 (by separately considering the GF trajectories $t \mapsto \theta ( t )$ and $t \mapsto \theta ( - t )$ of the potentials F and −F, respectively) to the special case where the time interval I is equal to the half-open interval $I = [ \tau , T )$ for some $T \in ( \tau , \infty ] , \tau \in \mathbb { R }$ . This special case is proved as Corollary 6.6 in Subsection 6.3 below.

Corollary 6.6 is proved by reducing it to the special case where the GF trajectory $\theta ( t ) , t \in [ \tau , T )$ , never hits the reference point ϑ for the spherical projection in the sense that $\{ t \in [ \tau , T ) \colon \theta ( t ) = \vartheta \} = \varnothing$ . This special case is proved as Proposition 6.4 in Subsection 6.3. Loosely speaking, the proof of Proposition 6.4 can be divided into three cases, namely,

(A) the case in which there exists $v \in \mathbb { R } ^ { n } \backslash \{ \vartheta \}$ such that lim su $\mathrm { p } _ { t  T } | \theta ( t ) - v | = 0$ (convergence to a finite point $v \in \mathbb { R } ^ { n }$ that is diferent from the reference point ϑ),

(B) the case in which lim su $\mathrm { p } _ { t  T } | \theta ( t ) - \vartheta | = 0$ (convergence to the reference point $\vartheta \in \mathbb { R } ^ { n } )$ – the case of the DTGC, and

(C) the case in which lim in $\mathrm { f } _ { t  T } | \theta ( t ) | = \infty$ (divergence to infinity) – the case of the gradient conjecture at infinity.

The case in item (A) is straightforward as the spherical length can be controlled through the length of the GF trajectory and is handled here by item (ii) in Proposition 6.3 in Subsection 6.2 below (cf. also Corollary 6.1 in Subsection 6.1 below). The key dificulty lies in the cases in items (B) and (C).

A central step in the proof of Proposition 6.4 in the case of items (B) and (C) is to perform an appropriate coordinate transform to the GF DE to obtain a transformed DE system that consists of a radial variable $r ( t )$ (which is $r ( t ) = | \theta ( t ) - \vartheta | ^ { \nu }$ with $\nu = 1$ in the case of item (B) and $\nu = - 1$ in the case of item (C)) and the angular variable $\omega ( t )$ (which is the spherical projection) that takes values in a regular submanifold of $\mathbb { R } ^ { n }$ (cf. Definition 2.2), namely, in the setting of Section 6, the sphere in $\mathbb { R } ^ { n }$ The transformation is thus nothing but polar coordinates in the case of item (B) and polar coordinates combined with the inversion method $( \bar { \mathbb { R } } ^ { n } \backslash \{ 0 \} \ni x \mapsto | x | ^ { - 2 } x \in \mathbb { R } ^ { n } \backslash \{ 0 \} )$ in the case of item (C), even though the transformed DE system appears slightly more complicated.

The resulting DE system is precisely in the form required by the machinery in Section 5 – specifically, item (ii) in Corollary $5 . 7 \mathrm { ~ - ~ } \mathrm { t o }$ establish finite length of the angular variable on the submanifold and, therefore, to establish finite spherical length of the original GF trajectories. A useful feature of the transformation and the resulting DE system is that after the transformation the radial variable always converges to zero as both cases, item (B) (convergence to the reference point ϑ) and item (C) (divergence to infinity), are handled in the regime where the transformed radial variable approaches zero.

## 6.1 Uniform upper bounds for the spherical length of GF trajectories close to the origin

In the following Corollary 6.1 we establish a uniform upper bound for the length of the spherical projection of a GF trajectory where we count only those time periods in which the GF trajectory is in a suficiently small ball around the origin. The potential $F \colon V  \mathbb { R }$ in Corollary 6.1 is assumed to be continuously diferentiable and definable.

Corollary 6.1. Let $n \in \mathbb { N } \backslash \{ 1 \} , \ \Re \in ( 0 , \infty )$ , let $V = \{ z \in \mathbb { R } ^ { n } \colon 0 < | z | < \Re \}$ , and let $F \in C ^ { 1 } ( V , \mathbb { R } )$ be definable (cf. Definition 1.6). Then there exist $\varepsilon \in ( 0 , \Re ) , \mathfrak { C } \in \mathbb { R }$ such that for every interval $I \subseteq \mathbb { R }$ , every locally absolutely continuous $\theta \colon I  V$ , every Lebesgue measurable λ: $I  [ 0 , \infty )$ , and every $\omega \in C ( I , \mathbb { R } ^ { n } )$ with the property that for all $t \in I$ it holds that $\omega ( t ) = | \theta ( t ) | ^ { - 1 } \theta ( t )$ and $\begin{array} { r } { \int _ { I } \vert \theta ^ { \prime } ( s ) + \lambda ( s ) ( \nabla F ) ( \theta ( s ) ) \vert \mathrm { d } s = 0 } \end{array}$ we have that

(i) it holds that $\omega$ is locally absolutely continuous and

(ii) it holds for all $\rho \in ( 0 , \varepsilon ]$ that

$$
\int _ { \{ t \in I \colon 0 < | \theta ( t ) | \leq \rho \} } | \theta ^ { \prime } ( t ) | \mathrm { d } t \leq \mathfrak { C } \rho \qquad a n d \qquad \int _ { \{ t \in I \colon 0 < | \theta ( t ) | \leq \varepsilon \} } | \omega ^ { \prime } ( t ) | \mathrm { d } t \leq \mathfrak { C } .\tag{6.1}
$$

Proof of Corollary 6.1. Throughout this proof let $M \subseteq \mathbb { R } ^ { n }$ satisfy

$$
M = \{ u \in \mathbb { R } ^ { n } : | u | = 1 \} ,\tag{6.2}
$$

and let $h \colon ( 0 , \Re ) \times M \to \mathbb { I }$ R satisfy for all $s \in ( 0 , \Re ) , u \in M$ that

$$
h ( s , u ) = F ( s u ) .\tag{6.3}
$$

Note that (6.2), (6.3), the fact that $n \geq 2$ , and the assumption that $F \in C ^ { 1 } ( V , \mathbb { R } )$ is definable ensure that M is a nonempty, compact, connected, and definable $( n - 1 )$ -dimensional $\dot { C } ^ { 2 }$ -submanifold of $\mathbb { R } ^ { n }$ and that $h \in { \bf C } ^ { 1 } ( ( 0 , \Re ) \times M , \mathbb { R } )$ is definable (cf. Definitions 1.5, 2.1, and 2.5). Proposition 5.6 (applied with $D \curvearrowleft { \cap } { n , \Omega } \curvearrowleft { \left( 0 , \Re \right) \times M , \textsf { a } } \curvearrowleft { 1 , \textsf { b } } \curvearrowleft { 1 , \ A ( t ) \curvearrowleft \mathrm { I d } _ { n } }$ in the notation of Proposition 5.6), the monotonicity of the Lebesgue integral, and Proposition 5.8 (applied with $D \cap n , n \cap n - 1 , \mathsf { a } \cap 1 , \mathsf { b } \cap 1 , R \cap \varepsilon ,$ $A ( t ) \cap { \mathrm { I d } } _ { n }$ in the notation of Proposition 5.8) hence imply that there exist $\varepsilon \in ( 0 , \Re ) , K , L \in ( 0 , \infty )$ which satisfy for every interval $I \subseteq \mathbb { R } .$ , all locally absolutely continuous $r \colon I  ( 0 , \Re )$ and $x \colon I \to M .$ and every Lebesgue measurable α: $I  [ 0 , \infty )$ with the property that for Lebesgue-almost all $t \in I \ { \mathrm { i t } }$ holds that

$$
\begin{array} { r } { r ^ { \prime } ( t ) = - \alpha ( t ) ( r ( t ) ) ^ { 2 } ( \partial _ { r } h ) ( r ( t ) , x ( t ) ) \qquad \mathrm { a n d } \qquad x ^ { \prime } ( t ) = - \alpha ( t ) ( \nabla _ { M } h ) ( r ( t ) , x ( t ) ) , } \end{array}\tag{6.4}
$$

we have for all $\rho \in ( 0 , \Re )$ that

$$
\int _ { r ^ { - 1 } ( [ \rho / 2 , \rho ] ) } | r ^ { \prime } ( t ) | \mathrm { d } t \leq K \rho \qquad \mathrm { a n d ~ } \qquad \int _ { r ^ { - 1 } ( ( 0 , \varepsilon ] ) } | x ^ { \prime } ( t ) | \mathrm { d } t \leq L\tag{6.5}
$$

(cf. Definition 2.7). In the following let ${ \mathfrak { C } } \in ( 0 , \infty )$ satisfy

$$
{ \mathfrak { C } } = 2 K + L .\tag{6.6}
$$

We claim that for every interval $I \subseteq \mathbb { R } .$ , every locally absolutely continuous $y \colon I  V$ , every Lebesgue measurable λ: $I  [ 0 , \infty )$ , and every ω : $I \to \mathbb { R } ^ { n }$ with the property that for Lebesgue-almost all $t \in I$ it holds that

$$
y ^ { \prime } ( t ) = - \lambda ( t ) ( \nabla F ) ( y ( t ) ) ,\tag{6.7}
$$

and with the property that for all $t \in I$ it holds that

$$
\omega ( t ) = { \frac { y ( t ) } { | y ( t ) | } }\tag{6.8}
$$

we have that $\omega$ is locally absolutely continuous and that for all $\rho \in ( 0 , \varepsilon ]$ it holds that

$$
\int _ { \{ t \in I \colon 0 < | y ( t ) | \leq \rho \} } | y ^ { \prime } ( t ) | \mathrm { d } t \leq ( 2 K + L ) \rho \qquad \mathrm { a n d ~ } \qquad \int _ { \{ t \in I \colon 0 < | y ( t ) | \leq \varepsilon \} } | \omega ^ { \prime } ( t ) | \mathrm { d } t \leq L .\tag{6.9}
$$

To prove this claim, let $I \subseteq \mathbb { R }$ be an interval, let $\rho \in ( 0 , \varepsilon ]$ , let $y \colon I  V$ be locally absolutely continuous, let $\lambda \colon I  [ 0 , \infty )$ be Lebesgue measurable, assume that for Lebesgue-almost all $t \in I$ it holds that

$$
y ^ { \prime } ( t ) = - \lambda ( t ) ( \nabla F ) ( y ( t ) ) ,\tag{6.10}
$$

and let ω : $I \to \mathbb { R } ^ { n }$ satisfy for all $t \in I$ that

$$
\omega ( t ) = \frac { y ( t ) } { | y ( t ) | } .\tag{6.11}
$$

In the following let $r \colon I  ( 0 , \Re )$ and $\alpha \colon I  [ 0 , \infty )$ satisfy for all $t \in I$ that

$$
r ( t ) = | y ( t ) | \qquad \mathrm { a n d } \qquad \alpha ( t ) = { \frac { \lambda ( t ) } { ( r ( t ) ) ^ { 2 } } } .\tag{6.12}
$$

Observe that (6.2), (6.11), (6.12), the fact that y is locally absolutely continuous, and the assumption that λ is Lebesgue measurable show that $\omega ( I ) \subseteq M$ , that α is Lebesgue measurable, and that r and $\omega$ are locally absolutely continuous. Next, note that (6.3) and the chain rule demonstrate that for all $s \in ( 0 , \Re ) , u \in M$ it holds that

$$
( \partial _ { r } h ) ( s , u ) = \langle ( \nabla F ) ( s u ) , u \rangle .\tag{6.13}
$$

In addition, observe that (6.2), (6.3), and the chain rule prove that for all $s \in ( 0 , \Re ) , u \in M , \eta \in ( 0 , \infty )$ 2 $\gamma \in C ^ { 1 } ( ( - \eta , \eta ) , M )$ with $\gamma ( 0 ) = u$ it holds that

$$
\langle u , \gamma ^ { \prime } ( 0 ) \rangle = \frac { 1 } { 2 } \frac { \mathrm { d } } { \mathrm { d } t } | \gamma ( t ) | ^ { 2 } \big | _ { t = 0 } = 0\tag{6.14}
$$

and

$$
\frac { \mathrm { d } } { \mathrm { d } t } h ( s , \gamma ( t ) ) \big | _ { t = 0 } = \frac { \mathrm { d } } { \mathrm { d } t } F ( s \gamma ( t ) ) \big | _ { t = 0 } = s \langle \bigl ( \nabla F \bigr ) \bigl ( s u \bigr ) , \gamma ^ { \prime } ( 0 ) \rangle .\tag{6.15}
$$

Combining (6.2), the fact that for all $u \in M$ it holds that $T _ { u } M = \{ v \in \mathbb { R } ^ { n } \colon \langle u , v \rangle = 0 \}$ , and the fact that for all $u \in M , w \in \mathbb { R } ^ { n }$ it holds that $\langle w - \langle w , u \rangle u , u \rangle = \langle w , u \rangle ( 1 - | u | ^ { 2 } ) = 0$ therefore establishes that for all $s \in ( 0 , \Re ) , u \in M$ it holds that

$$
s ( ( \nabla F ) ( s u ) - \langle ( \nabla F ) ( s u ) , u \rangle u ) \in T _ { u } M\tag{6.16}
$$

and that for all $s \in ( 0 , \Re ) , u \in M , \eta \in ( 0 , \infty ) , \gamma \in C ^ { 1 } ( ( - \eta , \eta ) , M )$ with $\gamma ( 0 ) = u$ it holds that

$$
\left. s \big ( \big ( \nabla F \big ) \big ( s u \big ) - \langle \big ( \nabla F \big ) \big ( s u \big ) , u \rangle u \big ) , \gamma ^ { \prime } ( 0 ) \right.
$$

$$
= s \langle \left( \nabla F \right) \left( s u \right) , \gamma ^ { \prime } ( 0 ) \rangle - s \langle \left( \nabla F \right) \left( s u \right) , u \rangle \langle u , \gamma ^ { \prime } ( 0 ) \rangle = \frac { \mathrm { d } } { \mathrm { d } t } h ( s , \gamma ( t ) ) \big | _ { t = 0 }\tag{6.17}
$$

(cf. Definition 2.3). This, (6.13), and Lemma 2.6 ensure that for all $s \in ( 0 , \Re ) , u \in M$ it holds that

$$
\begin{array} { r } { ( \partial _ { r } h ) ( s , u ) = \langle \left( \nabla F \right) ( s u ) , u \rangle \qquad \mathrm { a n d } \qquad ( \nabla _ { M } h ) ( s , u ) = s \big ( \big ( \nabla F \big ) \big ( s u \big ) - \langle \left( \nabla F \right) \big ( s u \big ) , u \rangle u \big ) . } \end{array}\tag{6.18}
$$

Moreover, note that (6.11), (6.12), the fact that y is locally absolutely continuous, and the chain rule imply that for Lebesgue-almost all $t \in I$ it holds that

$$
r ^ { \prime } ( t ) = \langle \omega ( t ) , y ^ { \prime } ( t ) \rangle \qquad { \mathrm { a n d } } \qquad \omega ^ { \prime } ( t ) = { \frac { y ^ { \prime } ( t ) - r ^ { \prime } ( t ) \omega ( t ) } { r ( t ) } } .\tag{6.19}
$$

In the next step we combine (6.11) and (6.12) to obtain that for all $t \in I$ it holds that

$$
y ( t ) = r ( t ) \omega ( t ) \qquad \mathrm { a n d } \qquad \lambda ( t ) = \alpha ( t ) ( r ( t ) ) ^ { 2 } .\tag{6.20}
$$

Combining (6.10), (6.18), and (6.19) hence shows that for Lebesgue-almost all $t \in I$ it holds that

$$
\begin{array} { r l } & { r ^ { \prime } ( t ) = \langle \omega ( t ) , y ^ { \prime } ( t ) \rangle = - \lambda ( t ) \langle \left( \nabla F \right) \big ( y ( t ) \big ) , \omega ( t ) \rangle = - \lambda ( t ) \langle \left( \nabla F \right) \big ( r ( t ) \omega ( t ) \big ) , \omega ( t ) \rangle } \\ & { \qquad = - \lambda ( t ) ( \partial _ { r } h ) ( r ( t ) , \omega ( t ) ) = - \alpha ( t ) ( r ( t ) ) ^ { 2 } ( \partial _ { r } h ) ( r ( t ) , \omega ( t ) ) . } \end{array}\tag{6.21}
$$

This, (6.10), (6.18), (6.20), and the fact that for all $t \in I$ it holds that $r ( t ) > 0$ demonstrate that for Lebesgue-almost all $t \in I$ it holds that

$$
y ^ { \prime } ( t ) - r ^ { \prime } ( t ) \omega ( t ) = - \lambda ( t ) \big ( \big ( \nabla F \big ) \big ( y ( t ) \big ) - \langle \big ( \nabla F \big ) \big ( y ( t ) \big ) , \omega ( t ) \rangle \omega ( t ) \big ) = - \frac { \lambda ( t ) } { r ( t ) } ( \nabla _ { M } h ) ( r ( t ) , \omega ( t ) ) .\tag{6.22}
$$

Combining (6.12) and (6.19) therefore proves that for Lebesgue-almost all $t \in I$ it holds that

$$
\omega ^ { \prime } ( t ) = \frac { y ^ { \prime } ( t ) - r ^ { \prime } ( t ) \omega ( t ) } { r ( t ) } = - \frac { \lambda ( t ) } { ( r ( t ) ) ^ { 2 } } ( \nabla _ { M } h ) ( r ( t ) , \omega ( t ) ) = - \alpha ( t ) ( \nabla _ { M } h ) ( r ( t ) , \omega ( t ) ) .\tag{6.23}
$$

Next we combine (6.19) and the fact that for all $t \in I$ it holds that $r ( t ) > 0$ to obtain that for Lebesguealmost all $t \in I$ it holds that

$$
y ^ { \prime } ( t ) = r ^ { \prime } ( t ) \omega ( t ) + r ( t ) \omega ^ { \prime } ( t ) .\tag{6.24}
$$

Furthermore, observe that (6.5), the fact that $\omega ( I ) \subseteq M$ , the fact that α is Lebesgue measurable, the fact that r and ω are locally absolutely continuous, (6.12), (6.21), (6.23), and item (i) in Lemma 5.4 (applied with $\varrho \curvearrowright \mathparagraph \mathparagraph$ and ${ \mathfrak { C } } \qquad K )$ establish that

$$
\int _ { r ^ { - 1 } ( ( 0 , \rho ] ) } | r ^ { \prime } ( t ) | \mathrm { d } t \leq 2 K \rho \qquad \mathrm { a n d } \qquad \int _ { r ^ { - 1 } ( ( 0 , \varepsilon ] ) } | \omega ^ { \prime } ( t ) | \mathrm { d } t \leq L .\tag{6.25}
$$

This, (6.2), (6.12), (6.24), the fact that $\omega ( I ) \subseteq M$ , the fact that $0 < \rho \le \varepsilon$ , and the triangle inequality ensure that

$$
\begin{array} { r l } & { \displaystyle \int _ { \{ t \in I \colon 0 < | y ( t ) | \leq \rho \} } | y ^ { \prime } ( t ) | \mathrm { d } t = \int _ { r ^ { - 1 } ( ( 0 , \rho ] ) } | y ^ { \prime } ( t ) | \mathrm { d } t } \\ & { \qquad \leq \displaystyle \int _ { r ^ { - 1 } ( ( 0 , \rho ] ) } \left| r ^ { \prime } ( t ) \right| \mathrm { d } t + \rho \int _ { r ^ { - 1 } ( ( 0 , \varepsilon ] ) } \left| \omega ^ { \prime } ( t ) \right| \mathrm { d } t \leq ( 2 K + L ) \rho . } \end{array}\tag{6.26}
$$

Combining this, (6.12), (6.25), the fact that ω is locally absolutely continuous, and the fact that $I , \rho ,$ y, $\lambda ,$ and $\omega$ were arbitrary subject to the stated assumptions proves (6.9).

Note that (6.5), (6.6), (6.9), the fact that $K \in ( 0 , \infty )$ , and the fact that $L \in [ 0 , \infty )$ establish (6.1). The proof of Corollary 6.1 is thus complete. □

## 6.2 Finite spherical length for GF trajectories converging to finite points

In the proofs of our results on finite spherical length in Proposition 6.3, Proposition 6.4, and Corollary 6.6 below, we use the well-known fact, which we recall in the following Lemma 6.2, that the length of a locally absolutely continuous curve coincides with the integral of the norm of the (almost everywhere defined) derivative of the curve. Only for the sake of completeness, we include a detailed proof for Lemma 6.2 here.

Lemma 6.2. Let $n \in \mathbb N$ , let $I \subseteq \mathbb { R }$ be an interval, and let $\gamma \colon I  \mathbb { R } ^ { n }$ be locally absolutely continuous. Then

$$
\mathrm { l e n } ( \gamma ) = \int _ { I } \lvert \gamma ^ { \prime } ( t ) \rvert \mathrm { d } t\tag{6.27}
$$

(cf. Definition 1.8).

Proof of Lemma 6.2. Observe that the fact that $\gamma$ is locally absolutely continuous and the fundamental theorem of calculus for absolutely continuous functions imply that for all $k \in \mathbb { N } , \ t _ { 0 } , \ldots , t _ { k } \in I$ with $t _ { 0 } < t _ { 1 } < . . . < t _ { k }$ it holds that

$$
\begin{array} { r l } & { \displaystyle \sum _ { j = 1 } ^ { k } | \gamma ( t _ { j } ) - \gamma ( t _ { j - 1 } ) | = \sum _ { j = 1 } ^ { k } \left| \int _ { t _ { j - 1 } } ^ { t _ { j } } \gamma ^ { \prime } ( u ) \mathrm { d } u \right| \leq \sum _ { j = 1 } ^ { k } \int _ { t _ { j - 1 } } ^ { t _ { j } } | \gamma ^ { \prime } ( u ) | \mathrm { d } u } \\ & { \quad \quad \quad = \int _ { t _ { 0 } } ^ { t _ { k } } | \gamma ^ { \prime } ( u ) | \mathrm { d } u \leq \int _ { I } | \gamma ^ { \prime } ( u ) | \mathrm { d } u . } \end{array}\tag{6.28}
$$

Hence, we obtain that

$$
\mathrm { l e n } ( \gamma ) \leq \int _ { I } \lvert \gamma ^ { \prime } ( t ) \rvert \mathrm { d } t\tag{6.29}
$$

(cf. Definition 1.8). We claim that for all $a , b \in I$ with $a < b$ it holds that

$$
\int _ { a } ^ { b } | \gamma ^ { \prime } ( t ) | \mathrm { d } t \leq \mathrm { l e n } ( \gamma ) .\tag{6.30}
$$

To prove (6.30), let $a , b \in I$ satisfy $a < b$ . In the following let $q _ { k , j } \in [ a , b ] , k \in \mathbb { N } , j \in \{ 0 , 1 , \dots , k \}$ , satisfy for all $k \in \mathbb { N } , j \in \{ 0 , 1 , \ldots , k \}$ that

$$
q _ { k , j } = a + \frac { j } { k } ( b - a )\tag{6.31}
$$

and let $g _ { k } \colon [ a , b ) \to [ 0 , \infty ) , k \in \mathbb { N }$ , satisfy for all $k \in \mathbb { N } , j \in \{ 0 , 1 , \dots , k - 1 \} , t \in \left[ q _ { k , j } , q _ { k , j + 1 } \right)$ that

$$
g _ { k } ( t ) = \frac { \lvert \gamma \left( q _ { k , j + 1 } \right) - \gamma \left( q _ { k , j } \right) \rvert } { q _ { k , j + 1 } - q _ { k , j } } .\tag{6.32}
$$

Note that (6.31), (6.32), and the fact that $[ a , b ] \subseteq I$ show that for all $k \in \mathbb N$ it holds that

$$
\int _ { [ a , b ) } g _ { k } ( t ) \mathrm { d } t = \sum _ { j = 0 } ^ { k - 1 } \lvert \gamma ( q _ { k , j + 1 } ) - \gamma ( q _ { k , j } ) \rvert \le \mathrm { l e n } ( \gamma ) .\tag{6.33}
$$

Next, observe that (6.31), (6.32), the fact that $\gamma | _ { [ a , b ] }$ is absolutely continuous, and the fact that for Lebesgue-almost all $t \in ( a , b )$ it holds that $\gamma$ is diferentiable at t demonstrate that for Lebesgue-almost all $t \in ( a , b )$ it holds that

$$
\operatorname* { l i m } _ { k \to \infty } g _ { k } ( t ) = | \gamma ^ { \prime } ( t ) | .\tag{6.34}
$$

This, (6.33), the fact that for all $k \in \mathbb N$ it holds that $g _ { k }$ is nonnegative and Lebesgue measurable, and Fatou’s lemma ensure that

$$
\int _ { a } ^ { b } | \gamma ^ { \prime } ( t ) | \mathrm { d } t \leq \operatorname* { l i m i n f } _ { k \to \infty } \int _ { [ a , b ) } g _ { k } ( t ) \mathrm { d } t \leq \mathrm { l e n } ( \gamma ) .\tag{6.35}
$$

Combining this and the fact that $a , b \in I$ with $a < b$ were arbitrary proves (6.30).

Note that (6.30), the fact that I is an interval, the fact that $| \gamma ^ { \prime } |$ is nonnegative and Lebesgue measurable, and the monotone convergence theorem establish that

$$
\int _ { I } \lvert \gamma ^ { \prime } ( t ) \rvert \mathrm d t = \operatorname* { s u p } \biggl ( \{ 0 \} \cup \left\{ \int _ { c } ^ { d } \lvert \gamma ^ { \prime } ( t ) \rvert \mathrm d t : c , d \in I , \ c < d \right\} \biggr ) \leq \mathrm { l e n } ( \gamma ) .\tag{6.36}
$$

This and (6.29) prove that

$$
\mathrm { l e n } ( \gamma ) = \int _ { I } \lvert \gamma ^ { \prime } ( t ) \rvert \mathrm { d } t .\tag{6.37}
$$

The proof of Lemma 6.2 is thus complete.

Proposition 6.3. Let $n \in \mathbb { N } , \ \vartheta , p \in \mathbb { R } ^ { n } , \ \tau \in \mathbb { R } , \ T \in ( \tau , \infty ]$ , let $U \subseteq \mathbb { R } ^ { n }$ be definable and open, let $F \in C ^ { 1 } ( U , \mathbb { R } )$ be definable, let $\theta \in C ^ { 1 } ( [ \tau , T ) , U )$ satisfy

$$
\theta ^ { \prime } = - ( \nabla F ) \circ \theta ,\tag{6.38}
$$

let ω : $[ \tau , T ) \to \mathbb { R } ^ { n }$ satisfy for all $t \in [ \tau , T )$ with $\theta ( t ) \neq p$ that $\omega ( t ) = | \theta ( t ) - p | ^ { - 1 } ( \theta ( t ) - p )$ , and let $\mathfrak { t } = \operatorname* { s u p } ( \{ - \infty \} \cup \theta ^ { - 1 } ( \{ p \} ) )$ . Then the following statements hold:

(i) If lim $\begin{array} { r } { \operatorname* { i n f } _ { t  T } | \theta ( t ) | < \infty } \end{array}$ , then there exists $v \in \mathbb { R } ^ { n }$ such that lim su $\begin{array} { r } { \operatorname { l p } _ { t  T } | \theta ( t ) - v | = 0 } \end{array}$

(ii) If lim in $\mathrm { f } _ { t  T } | \theta ( t ) - \vartheta | = 0$ and there exists an open $O \subseteq \mathbb { R } ^ { n }$ such that $( \{ \vartheta \} \cap \{ p \} ) \subseteq O \subseteq ( U \cup \{ \vartheta \} )$ then for every $s \in [ \tau , T ) \cap ( \mathsf { t } , \infty )$ the function $\omega | _ { [ s , T ) }$ has finite length

(cf. Definition 1.8).

Proof of Proposition 6.3. Throughout this proof let $J \subseteq [ \tau , T )$ satisfy

$$
J = [ \tau , T ) \cap ( { \mathsf { t } } , \infty ) .\tag{6.39}
$$

Observe that (6.38) and the chain rule imply that $F \circ \theta$ is non-increasing and that for all $t \in ( \tau , T )$ it holds that

$$
\frac { \mathrm { d } } { \mathrm { d } t } F ( \theta ( t ) ) = \langle \left( \nabla F \right) \left( \theta ( t ) \right) , \theta ^ { \prime } ( t ) \rangle = - | \left( \nabla F \right) \left( \theta ( t ) \right) | ^ { 2 } = - | \theta ^ { \prime } ( t ) | ^ { 2 } .\tag{6.40}
$$

In addition, note that (6.39), the fact that ${ \mathfrak { t } } = \operatorname* { s u p } ( \{ - \infty \} \cup \theta ^ { - 1 } ( \{ p \} ) )$ , the assumption that for all $t \in [ \tau , T )$ with $\theta ( t ) \neq p$ it holds that $\omega ( t ) = ( \theta ( t ) - p ) / | \theta ( t ) - p |$ , the assumption that $\theta \in C ^ { 1 } ( [ \tau , T ) , U )$ 2 and the chain rule show that for all $s \in J$ it holds that $\omega | _ { [ s , T ) } \in C ^ { 1 } ( [ s , T ) , \mathbb { R } ^ { n } )$ and that for all $t \in J$ it holds that

$$
\theta ( t ) \neq p , \qquad | \omega ( t ) | = 1 , \qquad \mathrm { a n d } \qquad \omega ^ { \prime } ( t ) = \frac { \theta ^ { \prime } ( t ) - \langle \omega ( t ) , \theta ^ { \prime } ( t ) \rangle \omega ( t ) } { | \theta ( t ) - p | } .\tag{6.41}
$$

Combining this and (6.39) demonstrates that for all $t , s , b \in J$ with $s < b$ it holds that min $v \in [ s , b ]  \vert \theta ( v ) - p \vert >$ 0,

$$
\begin{array} { r l } & { | \theta ^ { \prime } ( t ) - \langle \omega ( t ) , \theta ^ { \prime } ( t ) \rangle \omega ( t ) | ^ { 2 } = | \theta ^ { \prime } ( t ) | ^ { 2 } - 2 \langle \omega ( t ) , \theta ^ { \prime } ( t ) \rangle \langle \theta ^ { \prime } ( t ) , \omega ( t ) \rangle + | \langle \omega ( t ) , \theta ^ { \prime } ( t ) \rangle | ^ { 2 } | \omega ( t ) | ^ { 2 } } \\ & { = | \theta ^ { \prime } ( t ) | ^ { 2 } - 2 | \langle \omega ( t ) , \theta ^ { \prime } ( t ) \rangle | ^ { 2 } + | \langle \omega ( t ) , \theta ^ { \prime } ( t ) \rangle | ^ { 2 } = | \theta ^ { \prime } ( t ) | ^ { 2 } - | \langle \omega ( t ) , \theta ^ { \prime } ( t ) \rangle | ^ { 2 } \leq | \theta ^ { \prime } ( t ) | ^ { 2 } , } \end{array}\tag{6.42}
$$

and

$$
\begin{array} { r } { \int _ { s } ^ { b } | \omega ^ { \prime } ( t ) | \mathrm { d } t = \int _ { s } ^ { b } \frac { | \theta ^ { \prime } ( t ) - \langle \omega ( t ) , \theta ^ { \prime } ( t ) \rangle \omega ( t ) | } { | \theta ( t ) - p | } \mathrm { d } t \leq \int _ { s } ^ { b } \frac { | \theta ^ { \prime } ( t ) | } { | \theta ( t ) - p | } \mathrm { d } t } \\ { \leq \frac { \int _ { s } ^ { b } | \theta ^ { \prime } ( t ) | \mathrm { d } t } { \operatorname* { m i n } _ { t \in [ s , b ] } | \theta ( t ) - p | } \leq \frac { ( b - s ) \operatorname* { m a x } _ { t \in [ s , b ] } | \theta ^ { \prime } ( t ) | } { \operatorname* { m i n } _ { t \in [ s , b ] } | \theta ( t ) - p | } < \infty . } \end{array}\tag{6.43}
$$

We claim that if lim in $\mathrm { \Delta } \mathrm { f } _ { t  T } | \theta ( t ) | < \infty$ , then

$$
\int _ { \tau } ^ { T } | \theta ^ { \prime } ( t ) | \mathrm { d } t < \infty .\tag{6.44}
$$

To prove this claim, assume that lim in $\mathrm { | } _ { t  T } | \theta ( t ) | ~ < ~ \infty$ Observe that the assumption that lim $\begin{array} { r } { \operatorname* { i n f } _ { t  T } | \theta ( t ) | < \infty } \end{array}$ ensures that there exists $\mathbf { v } \in \mathbb { R } ^ { n }$ which satisfies

$$
\operatorname* { l i m } _ { t  T } \operatorname* { i n f } _ { } | \theta ( t ) - \mathbf { v } | = 0 .\tag{6.45}
$$

In the following let $\chi \colon \mathbb { R } \to ( - 1 , 1 ) , H \colon U \to ( - 1 , 1 )$ , and $e \colon [ \tau , T )  ( - 1 , 1 )$ satisfy for all $u \in \mathbb { R } , z \in U$ ， $t \in [ \tau , T )$ that

$$
\chi ( u ) = { \frac { u } { \sqrt { 1 + u ^ { 2 } } } } , \qquad H ( z ) = \chi ( F ( z ) ) , \qquad { \mathrm { a n d } } \qquad e ( t ) = H ( \theta ( t ) ) .\tag{6.46}
$$

Note that (6.46), the assumption that $F \in C ^ { 1 } ( U , \mathbb { R } )$ is definable, and the chain rule establish that $H \in C ^ { 1 } ( U , ( - 1 , 1 ) )$ is definable and that for all $u \in \mathbb { R } , z \in U$ it holds that

$$
\chi ^ { \prime } ( u ) = ( 1 + u ^ { 2 } ) ^ { - 3 / 2 } > 0 \qquad \mathrm { a n d } \qquad \big ( \nabla H \big ) \big ( z \big ) = \chi ^ { \prime } \big ( F ( z ) \big ) \big ( \nabla F \big ) \big ( z \big ) .\tag{6.47}
$$

This, (6.38), (6.40), (6.46), and the chain rule prove that e is non-increasing and that for all $t \in ( \tau , T )$ it holds that

$$
e ^ { \prime } ( t ) = - \chi ^ { \prime } ( F ( \theta ( t ) ) ) | \theta ^ { \prime } ( t ) | ^ { 2 } = - | \bigl ( \nabla H \bigr ) \bigl ( \theta ( t ) \bigr ) | | \theta ^ { \prime } ( t ) | \leq 0 .\tag{6.48}
$$

Combining this and the fact that $e ( [ \tau , T ) ) \subseteq ( - 1 , 1 )$ implies that there exists $\ell \in [ - 1 , 1 )$ which satisfies

$$
\operatorname* { l i m } _ { t \to T } e ( t ) = \ell \qquad { \mathrm { a n d } } \qquad \operatorname* { i n f } _ { t \in [ \tau , T ) } e ( t ) = \ell .\tag{6.49}
$$

In our proof of (6.44) we distinguish between the case $e ^ { - 1 } ( \{ \ell \} ) \neq \emptyset$ and the case that for all $t \in [ \tau , T )$ it holds that $e ( t ) > \ell$ . We first prove (6.44) in the case $e ^ { - 1 } ( \{ \ell \} ) \neq \emptyset$ . Observe that $( 6 . 4 7 ) , ( 6 . 4 8 )$ , (6.49), and the fundamental theorem of calculus show that for every $a \in [ \tau , T )$ with $e ( a ) = \ell$ it holds that $e | _ { [ a , T ) }$ and $\theta | _ { [ a , T ) }$ are constant and that

$$
\int _ { \tau } ^ { T } | \theta ^ { \prime } ( t ) | \mathrm { d } t = \int _ { \tau } ^ { a } | \theta ^ { \prime } ( t ) | \mathrm { d } t < \infty .\tag{6.50}
$$

This establishes $\left( 6 . 4 4 \right)$ in the case $e ^ { - 1 } ( \{ \ell \} ) \neq \varnothing$ . We now prove (6.44) in the case that for all $t \in [ \tau , T )$ it holds that $e ( t ) > \ell .$ In the following let $W = \left\{ z \in U \colon \left[ ( | z - \mathbf { v } | < 1 ) \wedge ( H ( z ) > \ell ) \right] \right\}$ , and let $f \colon W \to ( 0 , \infty )$ satisfy for all $z \in W$ that

$$
f ( z ) = H ( z ) - \ell .\tag{6.51}
$$

Note that (6.51), the fact that W is open, bounded, and definable, the fact that $f \in C ^ { 1 } ( W , ( 0 , \infty ) )$ is positive and definable, and [32, Theorem 1] demonstrate that there exist $c , \eta \in ( 0 , \infty )$ and a strictly increasing $\varphi \in C ^ { 1 } ( ( 0 , \eta ) , ( 0 , \infty ) )$ ) which satisfy for all $z \in W$ with $f ( z ) < \eta$ that

$$
\varphi ^ { \prime } ( f ( z ) ) | \big ( \nabla H \big ) ( z ) | = | \big ( \nabla ( \varphi \circ f ) \big ) ( z ) | \geq c .\tag{6.52}
$$

In the following let $\psi : [ 0 , \eta )  [ 0 , \infty )$ satisfy $\psi ( 0 ) = 0$ and for all $u \in ( 0 , \eta )$ that

$$
\psi ( u ) = \frac { \varphi ( u ) - \operatorname* { i n f } \varphi ( ( 0 , \eta ) ) } { c } .\tag{6.53}
$$

Observe that (6.52), (6.53), and the fact that lim $_ { u \to 0 } \varphi ( u ) = \operatorname* { i n f } \varphi ( ( 0 , \eta ) ) \in [ 0 , \infty )$ ensure that ψ is continuous and strictly increasing on $[ 0 , \eta )$ , that $\psi | _ { ( 0 , \eta ) } \in C ^ { 1 } ( ( 0 , \eta ) , [ 0 , \infty ) )$ , and that for all $z \in W$ with $f ( z ) < \eta$ it holds that

$$
\psi ^ { \prime } ( f ( z ) ) | \big ( \nabla H \big ) \big ( z \big ) | \geq 1 \qquad \mathrm { a n d } \qquad \operatorname* { l i m } _ { u \to 0 } \psi ( u ) = 0 .\tag{6.54}
$$

Combining this, (6.45), (6.49), and the assumption that for all $t \in [ \tau , T )$ it holds that $e ( t ) > \ell$ proves that there exists $a \in ( \tau , T )$ which satisfies

$$
| \theta ( a ) - \mathbf { v } | < { \frac { 1 } { 4 } } , \qquad 0 < e ( a ) - \ell < \eta , \qquad \mathrm { a n d } \qquad \psi ( e ( a ) - \ell ) < { \frac { 1 } { 4 } } .\tag{6.55}
$$

Note that (6.46), (6.51), (6.55), the fact that e is non-increasing, the fact that $\theta ( [ \tau , T ) ) \subseteq U$ , the assumption that for all $t \in [ \tau , T )$ it holds that $e ( t ) > \ell ,$ , and the fact that $W = \{ z \ \in U \colon \lceil ( | z - \mathbf { v } | <$ $1 ) \wedge ( H ( z ) > \ell ) ] \}$ imply that for all $t \in [ a , T )$ with $| \theta ( t ) - \mathbf { v } | < 1$ it holds that

$$
\theta ( t ) \in W \qquad { \mathrm { a n d } } \qquad 0 < f ( \theta ( t ) ) = e ( t ) - \ell \leq e ( a ) - \ell < \eta .\tag{6.56}
$$

This and (6.54) show that for all $t \in [ a , T )$ with $| \theta ( t ) - \mathbf { v } | < 1$ it holds that

$$
\psi ^ { \prime } ( f ( \theta ( t ) ) ) | \bigl ( \nabla H \bigr ) \bigl ( \theta ( t ) \bigr ) | \geq 1 .\tag{6.57}
$$

Combining this, (6.48), (6.56), the fact that for all $t \in [ a , T )$ it holds that $| \theta ^ { \prime } ( t ) | \geq 0$ , and the chain rule establishes that for all $t \in [ a , T )$ with $| \theta ( t ) - \mathbf { v } | < 1$ it holds that

$$
- \frac { \mathrm { d } } { \mathrm { d } t } \psi ( e ( t ) - \ell ) = - \psi ^ { \prime } ( e ( t ) - \ell ) e ^ { \prime } ( t ) = \psi ^ { \prime } ( f ( \theta ( t ) ) ) | \big ( \nabla H \big ) \big ( \theta ( t ) \big ) | | \theta ^ { \prime } ( t ) | \geq | \theta ^ { \prime } ( t ) | .\tag{6.58}
$$

This, (6.55), the fact that ψ is nonnegative, and the fundamental theorem of calculus demonstrate that for all $b \in ( a , T )$ with $\theta ( [ a , b ] ) \subseteq \{ z \in \bar { \mathbb { R } } ^ { n } \colon | z - \mathbf { v } | < 1 \}$ it holds that

$$
\int _ { a } ^ { b } | \theta ^ { \prime } ( t ) | \mathrm { d } t \leq \psi ( e ( a ) - \ell ) - \psi ( e ( b ) - \ell ) \leq \psi ( e ( a ) - \ell ) < \frac { 1 } { 4 } .\tag{6.59}
$$

Combining (6.55), the fact that θ is continuous, and the fact that for all $b \in ( a , T )$ with $b = \operatorname { i n f } \{ t \in$ $[ a , T ) \colon | \theta ( t ) - \mathbf { v } | \geq 1 / 2 \}$ it holds that

$$
| \theta ( b ) - \mathbf { v } | = 1 / 2 , \qquad \theta ( [ a , b ] ) \subseteq \{ z \in \mathbb { R } ^ { n } \colon | z - \mathbf { v } | < 1 \} ,\tag{6.60}
$$

$$
{ \mathrm { a n d ~ } } \qquad { \frac { 1 } { 4 } } < | \theta ( b ) - \theta ( a ) | \leq \int _ { a } ^ { b } | \theta ^ { \prime } ( t ) | \mathrm { d } t\tag{6.61}
$$

hence ensures that

$$
\theta ( [ a , T ) ) \subseteq \{ z \in \mathbb { R } ^ { n } \colon | z - \mathbf { v } | < 1 / 2 \} .\tag{6.62}
$$

This, (6.59), the fact that $\theta | _ { [ \tau , a ] } \in C ^ { 1 } ( [ \tau , a ] , U )$ , and the monotone convergence theorem prove that

$$
\int _ { \tau } ^ { T } | \theta ^ { \prime } ( t ) | { \mathrm { d } } t = \left( \int _ { \tau } ^ { a } | \theta ^ { \prime } ( t ) | { \mathrm { d } } t \right) + \left( \int _ { a } ^ { T } | \theta ^ { \prime } ( t ) | { \mathrm { d } } t \right) \leq \left( \int _ { \tau } ^ { a } | \theta ^ { \prime } ( t ) | { \mathrm { d } } t \right) + \psi ( e ( a ) - \ell ) < \infty .\tag{6.63}
$$

This establishes (6.44) in the case that for all $t \in [ \tau , T )$ it holds that $e ( t ) > \ell .$ This completes our proof of (6.44).

In the following we prove item (i). We thus assume that lim in $\dot { \tau } _ { t  T } | \theta ( t ) | < \infty$ . Observe that (6.44), the assumption that lim $\begin{array} { r } { \operatorname* { i n f } _ { t  T } | \theta ( t ) | < \infty } \end{array}$ , and the fact that for all $b , d \in [ \tau , T )$ with $b < d$ it holds that $\begin{array} { r } { | \theta ( d ) - \theta ( b ) | \leq \int _ { b } ^ { d } | \theta ^ { \prime } ( t ) | \mathrm { d } t } \end{array}$ imply that there exists $\mathbf { v } \in \mathbb { R } ^ { n }$ such that

$$
\operatorname* { l i m } _ { t  T } | \theta ( t ) - \mathbf { v } | = 0 .\tag{6.64}
$$

This proves item (i).

In the following we prove item (ii). We thus assume that lim in $\mathrm { f } _ { t  T } | \theta ( t ) - \vartheta | = 0$ , let $\mathbb { O } \subseteq \mathbb { R } ^ { n }$ be open with $\{ \vartheta \} \cap \{ p \} \subseteq \mathbb { O } \subseteq U \cup \{ \vartheta \}$ , and let $s \in J$ . Note that item (i), (6.44), the assumption that lim $\begin{array} { r } { \operatorname* { i n f } _ { t  T } | \theta ( t ) - \vartheta | = 0 } \end{array}$ , and the fact that lim in $\mathrm { \Delta } [ \ _ { t  T } | \theta ( t ) | \leq | \vartheta | <$ ∞ show that

$$
\int _ { \tau } ^ { T } | \theta ^ { \prime } ( t ) | \mathrm { d } t < \infty \qquad \mathrm { a n d } \qquad \operatorname* { l i m } _ { t \to T } | \theta ( t ) - \vartheta | = 0 .\tag{6.65}
$$

In our proof of item (ii) we distinguish between the case ϑ $\neq p ,$ the case $( \vartheta = p$ and $n = 1 )$ , and the case $( \vartheta = p$ and $n \geq 2 )$ . We first prove item (ii) in the case $\vartheta \neq p .$ . Observe that (6.41), (6.42), (6.43), $_ { ( 6 . 6 5 ) }$ , the assumption that ϑ $\neq p ,$ , and the triangle inequality demonstrate that there exists $\zeta \in ( s , T )$ such that for all $t \in [ \zeta , T )$ it holds that $| \theta ( t ) - p | \geq 1 / 2 | \vartheta - p |$ and that

$$
\begin{array} { r l } { \displaystyle \int _ { s } ^ { T } | \omega ^ { \prime } ( t ) | \mathrm { d } t = \left( \int _ { s } ^ { \zeta } | \omega ^ { \prime } ( t ) | \mathrm { d } t \right) + \left( \int _ { \zeta } ^ { T } | \omega ^ { \prime } ( t ) | \mathrm { d } t \right) \leq \left( \int _ { s } ^ { \zeta } | \omega ^ { \prime } ( t ) | \mathrm { d } t \right) + \left( \int _ { \zeta } ^ { T } \frac { | \theta ^ { \prime } ( t ) | } { | \theta ( t ) - p | } \mathrm { d } t \right) } & { } \\ { \displaystyle \leq \left( \int _ { s } ^ { \zeta } | \omega ^ { \prime } ( t ) | \mathrm { d } t \right) + \frac { 2 } { | \vartheta - p | } \left( \int _ { \zeta } ^ { T } | \theta ^ { \prime } ( t ) | \mathrm { d } t \right) } & { } \\ { \displaystyle \leq \frac { \int _ { s } ^ { \zeta } | \theta ^ { \prime } ( t ) | \mathrm { d } t } { \operatorname* { m i n } _ { t \in [ s , \zeta ] } | \theta ( t ) - p | } + \frac { 2 } { | \vartheta - p | } \left( \int _ { \zeta } ^ { T } | \theta ^ { \prime } ( t ) | \mathrm { d } t \right) < \infty . } & { } \end{array}\tag{6.66}
$$

Combining this, the fact that $\omega | _ { [ s , T ) } \in C ^ { 1 } ( [ s , T ) , \mathbb { R } ^ { n } )$ , and Lemma 6.2 (applied with $I \cap [ s , T ) , \gamma \cap$ $\omega | _ { [ s , T ) }$ in the notation of Lemma 6.2) ensures that

$$
\mathrm { l e n } ( \omega | _ { [ s , T ) } ) = \int _ { [ s , T ) } | ( \omega | _ { [ s , T ) } ) ^ { \prime } ( t ) | \mathrm { d } t = \int _ { s } ^ { T } | \omega ^ { \prime } ( t ) | \mathrm { d } t < \infty\tag{6.67}
$$

(cf. Definition 1.8). This and the fact that $s \in J$ was arbitrary establish item (ii) in the case $\vartheta \neq p .$ . We now prove item (ii) in the case $\vartheta = p$ and $n = 1$ . Note that (6.41), the assumption that $n = 1$ , and the fact that $\omega | _ { [ s , T ) }$ is continuous imply that $\omega | _ { [ s , T ) }$ is constant. Hence, we obtain that

$$
\begin{array} { r } { \mathrm { l e n } ( \omega | _ { [ s , T ) } ) = 0 . } \end{array}\tag{6.68}
$$

Combining this and the fact that $s \in J$ was arbitrary proves item (ii) in the case $\vartheta = p$ and $n = 1$ . We now prove item (ii) in the case $\vartheta = p$ and $n \geq 2$ . Observe that (6.41), (6.65), the assumption that O is open, and the fact that $p \in \mathbb { O } \subseteq U \cup \{ p \}$ show that there exist $\Re \in ( 0 , \infty )$ and $\kappa \in ( s , T )$ which satisfy

$$
\{ z \in \mathbb { R } ^ { n } : 0 < | z - p | < \Re \} \subseteq U \qquad \mathrm { a n d } \qquad \theta ( [ \kappa , T ) ) \subseteq \{ z \in \mathbb { R } ^ { n } : 0 < | z - p | < \Re \} .\tag{6.69}
$$

In the following let $\mathcal { V } = \left. z \in \mathbb { R } ^ { n } \colon 0 < | z | < \Re \right.$ , and let $G \colon \mathcal { V }  \mathbb { R }$ and $\xi \colon [ \kappa , T )  \mathcal { V }$ satisfy for all $z \in \mathcal { V } , t \in [ \kappa , T )$ that

$$
G ( z ) = F ( p + z ) \qquad \mathrm { a n d } \qquad \xi ( t ) = \theta ( t ) - p .\tag{6.70}
$$

Note that (6.38), (6.41), (6.69), (6.70), the assumption that $F \in C ^ { 1 } ( U , \mathbb { R } )$ is definable, the assumption that $\theta \ \in \ { \dot { C } } ^ { 1 } ( [ \tau , { \dot { T } } ) , { \dot { U } } )$ , and the chain rule demonstrate that $G \in C ^ { 1 } ( \mathcal { V } , \mathbb { R } )$ is definable, that $\xi \in C ^ { 1 } ( [ \kappa , T ) , \mathcal { V } )$ , and that for all $t \in [ \kappa , T )$ it holds that

$$
\xi ^ { \prime } ( t ) = - ( \nabla F ) ( \theta ( t ) ) = - ( \nabla G ) ( \xi ( t ) ) \qquad { \mathrm { a n d } } \qquad \omega ( t ) = { \frac { \xi ( t ) } { | \xi ( t ) | } } .\tag{6.71}
$$

This, the fact that $G \in C ^ { 1 } ( \mathcal { V } , \mathbb { R } )$ is definable, the fact that $\xi \in C ^ { 1 } ( [ \kappa , T ) , \mathcal { V } )$ , and Corollary 6.1 (applied V $\mathrm { w i t h } \ F \ \curvearrow \ G , \ I \ \curvearrow \ [ \kappa , T ) , \ \theta \ \curvearrow \ \xi , \ \lambda ( t ) \ \curvearrow \ 1$ in the notation of Corollary 6.1) ensure that there exist $\delta \in ( 0 , \Re )$ and $\mathbf { L } \in [ 0 , \infty )$ which satisfy

$$
\int _ { \{ t \in [ \kappa , T ) : 0 < | \xi ( t ) | \leq \delta \} } | \omega ^ { \prime } ( t ) | \mathrm { d } t \leq \mathbf { L } .\tag{6.72}
$$

Observe that (6.43), (6.65), (6.70), (6.72), and the assumption that $\vartheta = p$ establish that there exists $\mu \in ( \kappa , T )$ such that $\forall t \in [ \mu , T ) \colon | \xi ( t ) | \leq \delta$ and that

$$
\int _ { s } ^ { T } | \omega ^ { \prime } ( t ) | { \mathrm { d } } t = \left( \int _ { s } ^ { \mu } | \omega ^ { \prime } ( t ) | { \mathrm { d } } t \right) + \left( \int _ { \mu } ^ { T } | \omega ^ { \prime } ( t ) | { \mathrm { d } } t \right) \leq \left( \int _ { s } ^ { \mu } | \omega ^ { \prime } ( t ) | { \mathrm { d } } t \right) + \mathbf { L } < \infty .\tag{6.73}
$$

Combining this, the fact that $\omega | _ { [ s , T ) } \in C ^ { 1 } ( [ s , T ) , \mathbb { R } ^ { n } )$ , and Lemma 6.2 (applied with $I \cap [ s , T ) , \gamma \cap$ $\omega | _ { [ s , T ) }$ in the notation of Lemma 6.2) implies that

$$
\mathrm { l e n } ( \omega | _ { [ s , T ) } ) = \int _ { [ s , T ) } | ( \omega | _ { [ s , T ) } ) ^ { \prime } ( t ) | \mathrm { d } t = \int _ { s } ^ { T } | \omega ^ { \prime } ( t ) | \mathrm { d } t < \infty .\tag{6.74}
$$

This and the fact that $s \in J$ was arbitrary prove item (ii) in the case $\vartheta = p$ and $n \geq 2 .$ . This completes our proof of item (ii). The proof of Proposition 6.3 is thus complete. □

## 6.3 Finite spherical length for general open domains

Proposition 6.4 (Finite spherical length for GF trajectories that avoid the reference point). $L e t n \in \mathbb { N } _ { : }$ $\vartheta \in \mathbb { R } ^ { n } , \tau \in \mathbb { R } , T \in ( \tau , \infty ]$ , let $U \subseteq \mathbb { R } ^ { n }$ be open, let $F \in C ^ { 1 } ( U , \mathbb { R } )$ be definable, let $\theta \in C ^ { 1 } ( [ \tau , T ) , U \backslash \{ \vartheta \} )$ satisfy for all $t \in [ \tau , T )$ that

$$
\theta ^ { \prime } ( t ) = - ( \nabla F ) ( \theta ( t ) ) ,\tag{6.75}
$$

and let $P \colon U \backslash \{ \vartheta \}  \mathbb { R } ^ { n }$ satisfy for all $v \in U \backslash \{ \vartheta \}$ that $P ( v ) = | v - \vartheta | ^ { - 1 } ( v - \vartheta )$ . Then $P \circ \theta$ has finite length (cf. Definition 1.8).

Proof of Proposition $6 . 4$ . Throughout this proof let $\mathbb { S } = \{ x \in \mathbb { R } ^ { n } \colon | x | = 1 \}$ , let $\phi \colon \mathbb { R }  ( - 1 , 1 )$ satisfy for all e ∈ R that $\phi ( e ) = e / \sqrt { 1 + e ^ { 2 } }$ , and let $\omega \colon [ \tau , T ) \to  { \mathbb { R } } ^ { n }$ satisfy

$$
\omega = P \circ \theta .\tag{6.76}
$$

Note that the assumption that for all $v \in U \backslash \{ \vartheta \}$ it holds that $P ( v ) = | v - \vartheta | ^ { - 1 } ( v - \vartheta )$ , (6.76), and the fact that $\theta ( [ \tau , T ) ) \subseteq U \backslash \{ \vartheta \}$ show that for all $t \in [ \tau , T )$ it holds that

$$
\omega ( t ) = \frac { \theta ( t ) - \vartheta } { | \theta ( t ) - \vartheta | } .\tag{6.77}
$$

Moreover, observe that the assumption that F is definable demonstrates that U is definable. In our proof of Proposition 6.4 we distinguish between the case that there exists $v \in \mathbb { R } ^ { n } \backslash \{ \vartheta \}$ such that lim su $\rvert _ { t  T } \lvert \theta ( t ) - v \rvert = 0$ and the case that there exists no $v \in \mathbb { R } ^ { n } \backslash \{ \vartheta \}$ such that lim $\begin{array} { r } { \operatorname* { u p } _ { t  T } | \theta ( t ) - v | = } \end{array}$ 0. We first prove that $P \circ \theta$ has finite length in the case that there exists $v \in \mathbb { R } ^ { n } \backslash \{ \vartheta \}$ such that lim sup $\boldsymbol { \mathbf { \ell } } _ { t  T } | \theta ( t ) - \boldsymbol { v } | = 0$ . In the following let $v \in \mathbb { R } ^ { n } \backslash \{ \vartheta \}$ satisfy lim $\begin{array} { r } { \operatorname* { s u p } _ { t  T } | \theta ( t ) - v | = 0 } \end{array}$ Note that item (ii) in Proposition 6.3 (applied with $\vartheta \curvearrowleft v , p \curvearrowleft \vartheta , O \curvearrowleft \varnothing$ in the notation of Proposition 6.3), (6.75), (6.76), (6.77), the fact that lim s $\begin{array} { r } { \operatorname { 1 p } _ { t  T } | \theta ( t ) - v | = 0 , } \end{array}$ , the fact that $\{ v \} \cap \{ \vartheta \} = \varnothing .$ , and the fact that $\theta ^ { - 1 } ( \{ \vartheta \} ) = \varnothing$ establish that $P \circ \theta$ has finite length in the case that there exists $v \in \mathbb { R } ^ { n } \backslash \{ \vartheta \}$ such that lim su $\begin{array} { r } { \mathsf { \nabla } \mathsf { l p } _ { t \to T } | \theta ( t ) - v | = 0 } \end{array}$ . We now prove that P ◦ θ has finite length in the case that there exists no $v \in \mathbb { R } ^ { n } \backslash \{ \vartheta \}$ such that lim su $\rvert _ { t  T } \lvert \theta ( t ) - v \rvert = 0$ . In the following we thus assume that there exists no $v \in \mathbb { R } ^ { n } \backslash \{ \vartheta \}$ such that lim $\begin{array} { r } { \operatorname* { s u p } _ { t  T } | \theta ( t ) - v | = 0 } \end{array}$ . Observe that item (i) in Proposition 6.3 (applied with $p \curvearrowleft $ in the notation of Proposition 6.3), (6.75), (6.77), the fact that U is definable and open, the fact that lim inf $_ { t  T } | \theta ( t ) | \in [ 0 , \infty ]$ , and the assumption that there exists no $v \in \mathbb { R } ^ { n } \backslash \{ \vartheta \}$ such that lim s $\begin{array} { r } { \mathsf { \nabla } \mathsf { l p } _ { t \to T } | \theta ( t ) - v | = 0 } \end{array}$ ensure that lim $\iota _ { t  T } \theta ( t ) = \vartheta$ or that $\begin{array} { r } { \operatorname* { l i m } _ { t  T } | \theta ( t ) | = \infty } \end{array}$ . In the following let $\nu \in \{ - 1 , 1 \}$ satisfy

$$
\nu = \left\{ \begin{array} { l l } { 1 } & { : \operatorname* { l i m } _ { t \to T } \theta ( t ) = \vartheta , } \\ { - 1 } & { : \operatorname* { l i m } _ { t \to T } \vert \theta ( t ) \vert = \infty } \end{array} , \right.\tag{6.78}
$$

let Γ, $r \colon [ \tau , T )  ( 0 , \infty )$ satisfy for all $t \in [ \tau , T )$ that

$$
\Gamma ( t ) = | \theta ( t ) - \vartheta | \qquad \mathrm { a n d } \qquad r ( t ) = ( \Gamma ( t ) ) ^ { \nu } ,\tag{6.79}
$$

let

$$
\Omega = \{ ( s , x ) \in ( 0 , 1 ) \times \mathbb { S } \colon \vartheta + s ^ { \nu } x \in U \} ,\tag{6.80}
$$

and let $z \colon \Omega \to U , h = ( h ( s , x ) ) _ { ( s , x ) \in \Omega } \colon \Omega \to ( - 1 , 1 )$ , and $\eta \colon \Omega \to ( 0 , \infty )$ satisfy for all $( s , x ) \in \Omega$ that

$$
z ( s , x ) = \vartheta + s ^ { \nu } x , \qquad h ( s , x ) = \phi ( F ( z ( s , x ) ) ) , \qquad \mathrm { a n d } \qquad \eta ( s , x ) = \phi ^ { \prime } ( F ( z ( s , x ) ) ) .\tag{6.81}
$$

Note that (6.75), (6.77), (6.78), (6.79), the assumption that $\theta \in C ^ { 1 } ( [ \tau , T ) , U \backslash \{ \vartheta \} )$ , and the chain rule imply that Γ, $r \in C ^ { 1 } ( [ \tau , T ) , ( 0 , \infty ) )$ , that $\omega \in C ^ { 1 } ( [ \tau , T ) , \mathbb { R } ^ { n } )$ , that $\omega ( [ \tau , T ) ) \subseteq \mathbb { S }$ , that lim $\mathsf { \Pi } _ { 1 : \to T } r ( t ) = 0$ and that there exists $c \in [ \tau , T )$ which satisfies for all $t \in [ c , T )$ that

$$
0 < r ( t ) < 1 , \qquad ( r ( t ) , \omega ( t ) ) \in \Omega , \qquad \mathrm { a n d } \qquad z ( r ( t ) , \omega ( t ) ) = \theta ( t ) .\tag{6.82}
$$

In the following let $\alpha \colon [ c , T ) \to ( 0 , \infty )$ satisfy for all $t \in [ c , T )$ that

$$
\alpha ( t ) = \frac { ( r ( t ) ) ^ { - 2 \nu } } { \eta ( r ( t ) , \omega ( t ) ) } .\tag{6.83}
$$

Observe that (6.80), (6.81), the fact that U is definable and open, the assumption that $F \in C ^ { 1 } ( U , \mathbb { R } )$ the assumption that F is definable, and the fact that for all $e \in \mathbb { R }$ it holds that $\phi ^ { \prime } ( e ) = ( 1 + e ^ { 2 } ) ^ { - 3 / 2 } > 0$ prove that Ω is definable and open in $( 0 , 1 ) \times \mathbb { S }$ , that $h \in { \bf C } ^ { 1 } ( \Omega , ( - 1 , 1 ) )$ is definable, and that α is continuous (cf. Definition 2.5). In the next step we note that (6.81), the chain rule, and the fact that for all $x \in \mathbb { S }$ it holds that $T _ { x } \mathbb { S } = \{ w \in \mathbb { R } ^ { n } \colon \langle w , x \rangle = 0 \}$ show that for all $( s , x ) \in \Omega$ it holds that

$$
\begin{array} { r l } & { ( \partial _ { s } h ) ( s , x ) = \nu s ^ { \nu - 1 } \eta ( s , x ) \langle ( \nabla F ) ( z ( s , x ) ) , x \rangle } \\ { \mathrm { a n d } \quad } & { ( \nabla _ { \mathbb { S } } h ) ( s , x ) = s ^ { \nu } \eta ( s , x ) ( ( \nabla F ) ( z ( s , x ) ) - \langle ( \nabla F ) ( z ( s , x ) ) , x \rangle x ) } \end{array}\tag{6.84}
$$

(cf. Definitions 2.3 and 2.7). Next we combine (6.75), (6.77), (6.79), and the chain rule to obtain that for all $t \in [ \tau , T )$ it holds that

$$
\Gamma ^ { \prime } ( t ) = \frac { \langle \theta ( t ) - \vartheta , \theta ^ { \prime } ( t ) \rangle } { \Gamma ( t ) } = \langle \omega ( t ) , \theta ^ { \prime } ( t ) \rangle = - \langle \left( \nabla F \right) \left( \theta ( t ) \right) , \omega ( t ) \rangle .\tag{6.85}
$$

Combining this, (6.79), (6.83), (6.82), (6.84), and the fact that $\nu ^ { 2 } = 1$ demonstrates that for all $t \in [ c , T )$ it holds that

$$
r ^ { \prime } ( t ) = \nu ( \Gamma ( t ) ) ^ { \nu - 1 } \Gamma ^ { \prime } ( t ) = - \nu ( r ( t ) ) ^ { 1 - \nu } \langle ( \nabla F ) ( \theta ( t ) ) , \omega ( t ) \rangle = - \alpha ( t ) ( r ( t ) ) ^ { 2 } ( \partial _ { s } h ) ( r ( t ) , \omega ( t ) ) .\tag{6.86}
$$

Furthermore, observe that (6.75), (6.77), (6.79), (6.83), (6.82), (6.84), and (6.85) establish that for all $t \in [ c , T )$ it holds that

$$
\begin{array} { r l } & { \omega ^ { \prime } ( t ) = \frac { \theta ^ { \prime } ( t ) } { \Gamma ( t ) } - \frac { \theta ( t ) - \vartheta } { ( \Gamma ( t ) ) ^ { 2 } } \Gamma ^ { \prime } ( t ) = \frac { \theta ^ { \prime } ( t ) - \Gamma ^ { \prime } ( t ) \omega ( t ) } { \Gamma ( t ) } } \\ & { \qquad = - ( r ( t ) ) ^ { - \nu } ( ( \nabla F ) ( \theta ( t ) ) - \langle ( \nabla F ) ( \theta ( t ) ) , \omega ( t ) \rangle \omega ( t ) ) = - \alpha ( t ) ( \nabla _ { \mathbb { S } } h ) ( r ( t ) , \omega ( t ) ) . } \end{array}\tag{6.87}
$$

This, (6.82), (6.86), the fact that $r \in C ^ { 1 } ( [ \tau , T ) , ( 0 , \infty ) )$ , the fact that ω $\in C ^ { 1 } ( [ \tau , T ) , \mathbb { R } ^ { n } )$ , the fact that $\omega ( [ \tau , T ) ) \subseteq \mathbb { S } .$ , the fact that S is a nonempty, compact, and definable C<sup>2</sup>-submanifold of $\mathbb { R } ^ { n }$ , the fact that α is continuous, the fact that lim $\mathbf { \rho } _ { ( t  T } r ( t ) = 0$ , and item (ii) in Corollary 5.7 (applied with $D \neq n ,$ R ↶ 1, M ↶ S, O ↶ Ω, a ↶ 1, b ↶ 1, $I \curvearrowleft[ c , T ) , r \curvearrowleft r | _ { [ c , T ) } , x \curvearrowleft \omega | _ { [ c , T ) } , A ( t ) \curvearrowleft \mathrm { I d } _ { n } , \tau \curvearrowleft c $ in the notation of Corollary 5.7) ensure that

$$
\int _ { c } ^ { T } | \omega ^ { \prime } ( t ) | \mathrm { d } t < \infty\tag{6.88}
$$

(cf. Definition 2.2). Combining this, $( 6 . 7 6 )$ , the fact that $\omega \in C ^ { 1 } ( [ \tau , T ) , \mathbb { R } ^ { n } )$ , the fact that $[ \tau , c ]$ is compact, and Lemma 6.2 proves that $P \circ \theta$ has finite length in the case that there exists no $v \in \mathbb { R } ^ { n } \backslash \{ \vartheta \}$ such that lim $\begin{array} { r } { \operatorname* { s u p } _ { t  T } | \theta ( t ) - v | = 0 } \end{array}$ . This completes the proof of Proposition 6.4. □

In the proof of the finite spherical length result in Corollary 6.6 below we employ the well-known fact, which we recall in Lemma 6.5 below, that the fibres/level sets of GF trajectories are convex. Only for completeness, we include a proof for Lemma 6.5 here.

Lemma 6.5 (Convexity of the level sets of $\mathrm { G F }$ trajectories). Let $n \in \mathbb { N } , \vartheta \in \mathbb { R } ^ { n }$ , let $U \subseteq \mathbb { R } ^ { n }$ be open, let $I \subseteq \mathbb { R }$ be convex with nonempty interior, and let $F \in C ^ { 1 } ( U , \mathbb { R } ) , \theta \in C ^ { 1 } ( I , U )$ satisfy for all $t \in I$ that

$$
\theta ^ { \prime } ( t ) = - ( \nabla F ) ( \theta ( t ) ) .\tag{6.89}
$$

Then it holds that $\{ t \in I \colon \theta ( t ) = \vartheta \}$ is convex.

Proof of Lemma 6.5. Throughout this proof let $A \subseteq I$ satisfy

$$
A = \{ t \in I \colon \theta ( t ) = \vartheta \} .\tag{6.90}
$$

We claim that for all $s , t \in A$ with $s < t$ it holds that

$$
[ s , t ] \subseteq A .\tag{6.91}
$$

To prove (6.91), let $s , t \in A$ satisfy $s < t$ . Note that (6.90), the fact that $s , t \in A$ , the fact that $s < t ,$ and the assumption that I is convex imply that

$$
[ s , t ] \subseteq I \qquad { \mathrm { a n d } } \qquad \theta ( s ) = \theta ( t ) = \vartheta .\tag{6.92}
$$

This, the assumption that $F \in C ^ { 1 } ( U , \mathbb { R } )$ , the assumption that $\theta \in C ^ { 1 } ( I , U )$ , the assumption that for all $u \in I$ it holds that $\theta ^ { \prime } ( u ) = - { \bigl ( } \nabla F { \bigr ) } { \bigl ( } \theta ( u ) { \bigr ) }$ , and the chain rule show that $( F \circ \theta ) | _ { [ s , t ] } \in C ^ { 1 } ( [ s , t ] , \mathbb { R } )$ and that for all $u \in ( s , t )$ it holds that

$$
\frac { \mathrm { d } } { \mathrm { d } u } F ( \theta ( u ) ) = \langle \left( \nabla F \right) \left( \theta ( u ) \right) , \theta ^ { \prime } ( u ) \rangle = - | \theta ^ { \prime } ( u ) | ^ { 2 } .\tag{6.93}
$$

Combining (6.92), the fact that $( F \circ \theta ) | _ { [ s , t ] } \in C ^ { 1 } ( [ s , t ] , \mathbb { R } )$ , and the fundamental theorem of calculus therefore demonstrates that

$$
\int _ { s } ^ { t } \lvert \theta ^ { \prime } ( u ) \rvert ^ { 2 } \mathrm { d } u = F ( \theta ( s ) ) - F ( \theta ( t ) ) = 0 .\tag{6.94}
$$

This, (6.92), the assumption that $\theta \in C ^ { 1 } ( I , U )$ , and the fact that $s < t$ establish that for all $u \in [ s , t ]$ it holds that

$$
\theta ^ { \prime } ( u ) = 0 .\tag{6.95}
$$

Combining this, (6.92), the assumption that $\theta \in C ^ { 1 } ( I , U )$ , and the fundamental theorem of calculus ensures that for all $u \in [ s , t ]$ it holds that

$$
\theta ( u ) = \theta ( s ) + \int _ { s } ^ { u } \theta ^ { \prime } ( v ) \mathrm { d } v = \vartheta .\tag{6.96}
$$

This, (6.90), (6.92), and the fact that $s , t \in A$ with $s < t$ were arbitrary prove (6.91).

Observe that (6.90) and (6.91) establish that $\{ t \in I \colon \theta ( t ) = \vartheta \}$ is convex. The proof of Lemma 6.5 is thus complete. □

Corollary 6.6 (Finite spherical length for GF trajectories on a general open set). Let $n \in \mathbb { N } , \vartheta \in \mathbb { R } ^ { n }$ ， $\tau \in \mathbb { R } , T \in ( \tau , \infty ]$ , let $U \subseteq \mathbb { R } ^ { n }$ be open, let $F \in C ^ { 1 } ( U , \mathbb { R } )$ be definable, let $\theta \in C ^ { 1 } ( [ \tau , T ) , U )$ satisfy for all $t \in [ \tau , T )$ that

$$
\theta ^ { \prime } ( t ) = - ( \nabla F ) ( \theta ( t ) ) ,\tag{6.97}
$$

and let $P \colon U \to \mathbb { R } ^ { n }$ satisfy for all $v \in U \backslash \{ \vartheta \}$ that $P ( v ) = | v - \vartheta | ^ { - 1 } ( v - \vartheta )$ . Then $P \circ \theta$ has finite length.

Proof of Corollary 6.6. Note that Proposition 6.4 implies that if $\vartheta \notin U$ , then $P \circ \theta$ has finite length. In the following assume that $\vartheta \in U$ . Throughout this proof let $A = \left\{ t \in [ \tau , T ) \colon \theta ( t ) = \vartheta \right\}$ . Observe that Lemma 6.5 (applied with $I \frown [ \tau , T )$ in the notation of Lemma 6.5), the fact that $[ \tau , T )$ is convex with nonempty interior, and the fact that $A = \{ t \in [ \tau , T ) \colon \theta ( t ) = \vartheta \}$ show that A is convex. In the following let J satisfy

$$
\mathcal { I } = \big \{ J \subseteq [ \tau , T ) \backslash A \colon \big [ ( J \mathrm { ~ i s ~ c o n v e x } ) \wedge \big ( \forall V \mathrm { ~ c o n v e x } , J \subseteq V \subseteq [ \tau , T ) \backslash A \colon V = J \neq ( \emptyset ) \big ] \big \} .\tag{6.98}
$$

Note that (6.98), the fact that A is convex, and the fact that A is closed in $[ \tau , T )$ demonstrate that J contains at most two intervals. We claim that for all $J \in \mathcal { I }$ it holds that

$$
\mathrm { l e n } ( ( P \circ \theta ) | _ { J } ) < \infty\tag{6.99}
$$

To prove (6.99), let $J \in \mathcal { I } , c \in J$ , and let $a = \operatorname { i n f } J$ and $b = \operatorname* { s u p } J .$ . Observe that Proposition 6.4 (applied with $\tau \wedge c , T \wedge b , P \wedge P | _ { U \setminus \{ \vartheta \} } , \theta \frown \theta | _ { [ c , b ) }$ in the notation of Proposition 6.4) and the fact that $[ c , b ) \subseteq J$ ensure that

$$
\mathrm { l e n } ( ( P \circ \theta ) | _ { [ c , b ) } ) < \infty\tag{6.100}
$$

(cf. Definition 1.8). Next, note that Lemma 6.2 and the fact that $( P \circ \theta ) | _ { J } \in C ^ { 1 } ( J , \mathbb { R } ^ { n } )$ prove that if $a \in J$ , then it holds that

$$
\begin{array} { r } { \mathrm { l e n } \big ( ( P \circ \theta ) | _ { [ a , c ] } \big ) < \infty . } \end{array}\tag{6.101}
$$

In addition, observe that the assumption that $\theta ^ { \prime } = - ( \nabla F ) \circ \theta$ and the chain rule establish that if a ${ \mathrm { ~ , ~ } } \notin J ,$ then for all $t \in [ - c , - a )$ it holds that

$$
{ \frac { \mathrm { d } } { \mathrm { d } t } } \theta ( - t ) = - \theta ^ { \prime } ( - t ) = \left( \nabla F \right) \left( \theta ( - t ) \right) = - { \bigl ( } \nabla ( - F ) { \bigr ) } { \bigl ( } \theta ( - t ) { \bigr ) } .\tag{6.102}
$$

Proposition 6.4 (applied with $\tau \nleftarrow - c , T \nleftarrow - a , F \land - F , P \land P | _ { U \setminus \{ \varnothing \} } , \theta \land ( [ - c , - a ) \ni t \mapsto \theta ( - t ) \in$ U) in the notation of Proposition 6.4) hence implies that if a $\notin J ,$ then it holds that

$$
\operatorname { l e n } ( ( P \circ \theta ) | _ { ( a , c ] } ) < \infty .\tag{6.103}
$$

Combining (6.100), (6.101), the triangle inequality, and the fact that $J \in \mathcal { I }$ was arbitrary therefore proves (6.99).

Note that (6.99), the fact that J contains at most two intervals, the fact that for all $t \in A$ it holds that $P ( \theta ( t ) ) = P ( \vartheta )$ , the fact that for all $t \in [ \tau , T ) \backslash A$ it holds that $| P ( \theta ( t ) ) | = 1$ , and the triangle inequality show that for all $k \in \mathbb { N } , t _ { 0 } , t _ { 1 } , \ldots , t _ { k } \in [ \tau , T )$ with $t _ { 0 } < t _ { 1 } < . . . < t _ { k }$ it holds that

$$
\sum _ { j = 1 } ^ { k } \lvert P ( \theta ( t _ { j } ) ) - P ( \theta ( t _ { j - 1 } ) ) \rvert \leq \sum _ { J \in \mathcal { I } } \mathrm { l e n } ( ( P \circ \theta ) \lvert _ { J } ) + 2 \big ( 1 + \lvert P ( \vartheta ) \rvert \big ) < \infty .\tag{6.104}
$$

Hence, we obtain that len $( P \circ \theta ) < \infty$ . The proof of Corollary 6.6 is thus complete.

## 6.4 Finite spherical length for general time intervals

In the proof of the finite spherical length result in Theorem 6.8 below we employ the elementary length bound for decomposed curves without the boundary points in the following Lemma 6.7. Only for completeness, we include a proof for Lemma 6.7 here.

Lemma 6.7. Let $n \in \mathbb { N } , B \in \mathbb { R }$ , let $I \subseteq \mathbb { R }$ be convex, let $c \in \operatorname { i n t } ( I )$ , let $a = \operatorname { i n f } ( I )$ , let $b = \operatorname* { s u p } ( I )$ , and let $\gamma \colon I  \mathbb { R } ^ { n }$ satisfy for all $t \in I$ that

$$
| \gamma ( t ) | \leq B .\tag{6.105}
$$

Then

$$
\mathrm { l e n } ( \gamma ) \le \mathrm { l e n } ( \gamma | _ { ( a , c ] } ) + \mathrm { l e n } ( \gamma | _ { [ c , b ) } ) + 4 B\tag{6.106}
$$

(cf. Definition 1.8).

Proof of Lemma 6.7. Observe that the assumption that I is convex, the fact that a = inf I and $b = \operatorname { s u p } I$ and the fact that $c \in$ int(I) demonstrate that

$$
a < c < b , \qquad ( a , b ) \subseteq I , \qquad { \mathrm { a n d } } \qquad I \backslash ( a , b ) \subseteq \{ a , b \} \cap \mathbb { R } .\tag{6.107}
$$

Moreover, note that (6.105) and the triangle inequality ensure that for all $s , t \in I$ it holds that

$$
| \gamma ( t ) - \gamma ( s ) | \leq | \gamma ( t ) | + | \gamma ( s ) | \leq 2 B .\tag{6.108}
$$

In the next step we combine (6.107) and the triangle inequality to obtain that for all $k \in \mathbb { N } , t =$ $( t _ { 0 } , \ldots , t _ { k } ) \in ( a , b ) ^ { k + 1 }$ with $t _ { 0 } < t _ { 1 } < . . . < t _ { k }$ it holds that

$$
\sum _ { j = 1 } ^ { k } \lvert \gamma ( t _ { j } ) - \gamma ( t _ { j - 1 } ) \rvert \leq \mathrm { l e n } ( \gamma \lvert _ { ( a , c ] } ) + \mathrm { l e n } ( \gamma \lvert _ { [ c , b ) } )\tag{6.109}
$$

(cf. Definition 1.8). This, (6.107), (6.108), and the fact that for all $k \in \mathbb { N } , t = ( t _ { 0 } , t _ { 1 } , \ldots , t _ { k } ) \in I ^ { k + 1 }$ with $t _ { 0 } < t _ { 1 } < \ldots < t _ { k }$ it holds that $\{ t _ { 0 } , t _ { 1 } , \ldots , t _ { k } \} \backslash ( a , b ) \ \subseteq \ \{ t _ { 0 } , t _ { k } \}$ establish that for all $k \in \mathbb N$ $t = ( t _ { 0 } , t _ { 1 } , \ldots , t _ { k } ) \in I ^ { k + 1 }$ with $t _ { 0 } < t _ { 1 } < . . . < t _ { k }$ it holds that

$$
\begin{array} { l } { \displaystyle \sum _ { j = 1 } ^ { k } \lvert \gamma ( t _ { j } ) - \gamma ( t _ { j - 1 } ) \rvert \le \biggl ( \sum _ { j \in \{ 1 , \dots , k \} , ( t _ { j - 1 } , t _ { j } ) \in ( a , b ) ^ { 2 } } \lvert \gamma ( t _ { j } ) - \gamma ( t _ { j - 1 } ) \rvert \biggr ) + 4 B } \\ { \displaystyle \qquad \le \mathrm { l e n } ( \gamma \vert _ { ( a , c ] } ) + \mathrm { l e n } ( \gamma \vert _ { [ c , b ) } ) + 4 B . } \end{array}\tag{6.110}
$$

Hence, we obtain that

$$
\begin{array} { r } { \mathrm { l e n } ( \gamma ) \leq \mathrm { l e n } ( \gamma | _ { ( a , c ] } ) + \mathrm { l e n } ( \gamma | _ { [ c , b ) } ) + 4 B . } \end{array}\tag{6.111}
$$

The proof of Lemma 6.7 is thus complete.

Theorem 6.8. Let $n \in \mathbb { N } , \vartheta \in \mathbb { R } ^ { n }$ , let $U \subseteq \mathbb { R } ^ { n }$ be open, let $F \in C ^ { 1 } ( U , \mathbb { R } )$ be definable, let $I \subseteq \mathbb { R } \ b e$ convex with nonempty interior, let $\theta \in C ^ { 1 } ( I , U )$ satisfy

$$
\theta ^ { \prime } = - ( \nabla F ) \circ \theta ,\tag{6.112}
$$

and let $P \colon U \to \mathbb { R } ^ { n }$ satisfy for all $v \in U \backslash \{ \vartheta \}$ that $P ( v ) = | v - \vartheta | ^ { - 1 } ( v - \vartheta )$ . Then $P \circ \theta$ has finite length. Proof of Theorem 6.8. Throughout this proof let a = inf I, b = sup I, c ∈ int(I), let $\mathbf { P } \colon \mathbb { R } ^ { n } \to \mathbb { R } ^ { n }$ satisfy that $\mathbf { P } | _ { U } = P$ and that for all $v \in \mathbb { R } ^ { n } \backslash \{ \vartheta \}$ it holds that

$$
\mathbf { P } ( v ) = { \frac { v - \vartheta } { | v - \vartheta | } } ,\tag{6.113}
$$

let $B \in ( 0 , \infty )$ satisfy $B = 1 + | \mathbf { P } ( \vartheta ) |$ |, and let $\gamma \colon [ - c , - a )  U$ satisfy for all $t \in [ - c , - a )$ that

$$
\gamma ( t ) = \theta ( - t ) .\tag{6.114}
$$

Observe that the assumption that I is convex with nonempty interior, the fact that a = inf I and $b = \operatorname* { s u p } I ,$ and the fact that $c \in \operatorname { i n t } ( I )$ imply that

$$
- \infty \leq a < c < b \leq \infty , \qquad ( a , b ) \subseteq I , \qquad \mathrm { a n d } \qquad I \backslash ( a , b ) \subseteq \{ a , b \} \cap \mathbb { R } .\tag{6.115}
$$

Next we combine (6.113) and the fact that $B = 1 + | \mathbf { P } ( \vartheta ) |$ to obtain that for all $v \in \mathbb { R } ^ { n }$ it holds that

$$
| \mathbf { P } ( v ) | \leq B .\tag{6.116}
$$

Furthermore, note that (6.113), (6.115), the assumption that U is open, the assumption that $F \in$ $C ^ { 1 } ( U , \mathbb { R } )$ , the fact that F is definable, the assumption that $\theta \in C ^ { 1 } ( I , U )$ , the assumption that for all $t \in I$ it holds that $\theta ^ { \prime } ( t ) = - { \bigl ( } \nabla F { \bigr ) } { \bigl ( } \theta ( t ) { \bigr ) }$ , the fact that $\mathbf { P } | _ { U } = P$ , and Corollary 6.6 (applied with $\tau \land c ,$ $T \frown b , \theta \frown \theta | _ { [ c , b ) } , P \curvearrowright \mathbf { P }$ in the notation of Corollary 6.6) prove that

$$
\mathrm { l e n } ( ( P \circ \theta ) | _ { [ c , b ) } ) < \infty\tag{6.117}
$$

(cf. Definition 1.8). Next, observe that (6.114), (6.115), the assumption that $F \in C ^ { 1 } ( U , \mathbb { R } )$ , the fact that F is definable, the assumption that $\theta \ \in \ C ^ { 1 } ( I , U )$ , the assumption that for all $t \in I$ it holds that $\theta ^ { \prime } ( t ) = - ( \nabla F ) \big ( \theta ( t ) \big )$ , and the chain rule show that $- F \in C ^ { 1 } ( U , \mathbb { R } )$ , that $- \boldsymbol { F }$ is definable, that $\gamma \in C ^ { 1 } ( [ - c , - a ) , U )$ , and that for all $t \in [ - c , - a )$ it holds that

$$
\gamma ^ { \prime } ( t ) = - \theta ^ { \prime } ( - t ) = \left( \nabla F \right) \left( \gamma ( t ) \right) = - \left( \nabla ( - F ) \right) \left( \gamma ( t ) \right) .\tag{6.118}
$$

Combining this, (6.113), (6.115), the assumption that U is open, the fact that $- F \in C ^ { 1 } ( U , \mathbb { R } )$ , the fact that $- \boldsymbol { F }$ is definable, the fact that $\gamma \in C ^ { 1 } ( [ - c , - a ) , U )$ , and Corollary 6.6 (applied with $\tau \frown - c ,$ $T \frown - a , F \curvearrowright - F , \theta \curvearrowright \gamma , P \curvearrowright \mathbf { P }$ in the notation of Corollary 6.6) demonstrates that

$$
\mathrm { l e n } ( \mathbf { P } \circ \gamma ) < \infty .\tag{6.119}
$$

This, (6.114), and the fact that $\mathbf { P } | _ { U } = P$ ensure that

$$
\begin{array} { r } { \mathrm { l e n } ( ( P \circ \theta ) | _ { ( a , c ] } ) = \mathrm { l e n } ( \mathbf { P } \circ \gamma ) < \infty . } \end{array}\tag{6.120}
$$

Combining this, (6.116), (6.117), the fact that $\mathbf { P } | _ { U } = P ,$ the fact that $\theta ( I ) \subseteq U$ , and Lemma 6.7 (applied with $\gamma \curvearrowleft P \circ \theta$ in the notation of Lemma 6.7) establishes that $P \circ \theta$ has finite length. The proof of Theorem 6.8 is thus complete. □

## 6.5 Finite spherical length for locally definable potentials

In Corollary 6.10 below we establish finite spherical length of all GF trajectories with limit points in the domain of a locally definable potential. In our proof of Corollary 6.10 we employ the wellknown consequence of the KL inequality, which we recall in the following Proposition 6.9, that such GF trajectories are also convergent. The proof of Proposition 6.9 is based on an application of Kurdyka [32, Theorem 1]. For completeness, we include a proof for Proposition 6.9 here.

Proposition 6.9 (Convergence of GF trajectories). Let $n \in \mathbb { N } .$ , let $U \subseteq \mathbb { R } ^ { n }$ be open, let $F \in C ^ { 1 } ( U , \mathbb { R } )$ be locally $\mathop { d e f i n a b l e }$ , let $I \subseteq \mathbb { R }$ be convex with nonempty interior, let $\theta \in C ^ { 1 } ( I , U )$ satisfy for all $t \in I$ that $\theta ^ { \prime } ( t ) = - \bigl ( \nabla F \bigr ) \bigl ( \theta ( t ) \bigr )$ , and let $b \in U$ satisfy lim in $\boldsymbol { \mathrm { f } } _ { t  \mathrm { s u p } ( I ) } | \theta ( t ) - b | = 0$ . Then lim $\begin{array} { r } { \operatorname* { s u p } _ { t \to \operatorname* { s u p } ( I ) } | \theta ( t ) - b | = } \end{array}$ 0.

Proof of Proposition 6.9. Throughout this proof let $T = \operatorname* { s u p } ( I )$ , let h : $I \to \mathbb { R }$ satisfy $h = F \circ \theta .$ , and let $( t _ { j } ) _ { j \in \mathbb { N } } \subseteq I$ satisfy

$$
\operatorname* { l i m } _ { j \to \infty } t _ { j } = T \qquad { \mathrm { a n d } } \qquad \operatorname* { l i m } _ { j \to \infty } \theta ( t _ { j } ) = b .\tag{6.121}
$$

We claim that

$$
\operatorname* { l i m } _ { t \to T } | \theta ( t ) - b | = 0 .\tag{6.122}
$$

In our proof of (6.122) we distinguish between the case $T \in I$ and the case $T \not \in I .$ We first prove (6.122) in the case $T \in I$ . Note that the fact that θ is continuous and the assumption that lim in $ { \dot { \mathbf { \rho } } } _ { t  T } | \theta ( t ) - b | = 0$ imply that

$$
\operatorname* { l i m } _ { t \to T } | \theta ( t ) - b | = | \theta ( T ) - b | = \operatorname* { l i m } _ { t \to T } | \theta ( t ) - b | = 0 .\tag{6.123}
$$

This proves (6.122) in the case $T \in I .$ . We now prove (6.122) in the case $T \not \in I .$ Observe that the fact that $h = F \circ \theta$ , the assumption that $F \in C ^ { 1 } ( U , \mathbb { R } )$ , the assumption that $\theta \in C ^ { 1 } ( I , U )$ , the assumption that for all $t \in I$ it holds that $\theta ^ { \prime } ( t ) = - { \bigl ( } \nabla F { \bigr ) } { \bigl ( } \theta ( t ) { \bigr ) }$ , and the chain rule show that for all $t \in I$ it holds that

$$
h ^ { \prime } ( t ) = \langle \left( \nabla F \right) \left( \theta ( t ) \right) , \theta ^ { \prime } ( t ) \rangle = - | \left( \nabla F \right) \left( \theta ( t ) \right) | ^ { 2 } = - | \theta ^ { \prime } ( t ) | ^ { 2 } \leq 0 .\tag{6.124}
$$

Combining this, (6.121), the fact that I is convex, and the fact that F is continuous demonstrates that h is non-increasing and that for all $t \in I$ it holds that

$$
h ( t ) \geq \operatorname* { l i m } _ { s  T } h ( s ) = \operatorname* { l i m } _ { j  \infty } h ( t _ { j } ) = F ( b ) .\tag{6.125}
$$

In the following we distinguish between the case $h ^ { - 1 } ( \{ F ( b ) \} ) \neq \emptyset$ and the case that for all $t \in I$ it holds that $h ( t ) > F ( b )$ We first prove (6.122) in the case $h ^ { - 1 } ( \{ F ( b ) \} ) \neq \emptyset$ . Note that (6.124), (6.125), the fact that h is non-increasing, the fact that I is convex, and the fundamental theorem of calculus ensure that for all $s , u \in I$ with $s \leq u$ and $h ( s ) = F ( b )$ it holds that

$$
\int _ { s } ^ { u } | \theta ^ { \prime } ( t ) | ^ { 2 } \mathrm { d } t = h ( s ) - h ( u ) = 0 .\tag{6.126}
$$

This, the fact that $\theta ^ { \prime }$ is continuous, and the fundamental theorem of calculus establish that for all $s , u \in I$ with $s \leq u$ and $h ( s ) = F ( b )$ it holds that

$$
\theta ( u ) = \theta ( s ) + \int _ { s } ^ { u } \theta ^ { \prime } ( t ) \mathrm { d } t = \theta ( s ) .\tag{6.127}
$$

Combining (6.121), the fact that $T = \operatorname* { s u p } ( I ) \not \in I ,$ and the assumption that $h ^ { - 1 } ( \{ F ( b ) \} ) \neq \emptyset$ hence proves (6.122) in the case $h ^ { - 1 } ( \{ F ( b ) \} ) \neq \emptyset$ . We now prove (6.122) in the case that for all $t \in I$ it holds that $h ( t ) > F ( b )$ . Observe that the fact that F is locally definable, the fact that $b \in U$ , and the fact that U is open imply that there exist $\rho \in ( 0 , \infty ) , V \subseteq U$ which satisfy that

$$
\{ x \in \mathbb { R } ^ { n } \colon | x - b | \leq \rho \} \subseteq U , \quad \quad V = \{ x \in \mathbb { R } ^ { n } \colon | x - b | < \rho \} , \quad \quad \mathrm { a n d } \quad \quad F | _ { V } { \mathrm { ~ i s ~ d e f i n a b l e . } }\tag{6.128}
$$

In the following let $W \subseteq V$ satisfy

$$
W = \{ x \in V \colon F ( x ) > F ( b ) \}\tag{6.129}
$$

and let $f \colon W \to ( 0 , \infty )$ satisfy that for all $x \in W$ it holds that

$$
f ( x ) = F ( x ) - F ( b ) .\tag{6.130}
$$

Note that (6.128), (6.129), (6.130), and the assumption that $F \in C ^ { 1 } ( U , \mathbb { R } )$ show that W is open, bounded, and definable, that f is definable, and that $f \in C ^ { 1 } ( W , ( 0 , \infty ) )$ . This, (6.129), (6.130), [32, Theorem 1], and the chain rule demonstrate that there exist $c , \eta \in \mathsf { \Gamma } ( 0 , \infty )$ and a strictly increasing $\psi \in C ^ { 1 } ( ( 0 , \dot { \eta } ) , ( 0 , \infty ) )$ which satisfy for all $x \in V$ with $0 < F ( x ) - F ( b ) < \eta$ that

$$
\psi ^ { \prime } ( F ( x ) - F ( b ) ) | \bigl ( \nabla F \bigr ) ( x ) | \geq c .\tag{6.131}
$$

In the following let $\varphi \colon [ 0 , \eta )  [ 0 , \infty )$ satisfy that $\varphi ( 0 ) = 0$ and that for all $s \in ( 0 , \eta )$ it holds that

$$
\varphi ( s ) = { \frac { \psi ( s ) - \operatorname* { i n f } \psi ( ( 0 , \eta ) ) } { c } } .\tag{6.132}
$$

Observe that (6.132) and the fact that $\psi$ is positive and strictly increasing ensure that

$$
\operatorname* { l i m } _ { s \to 0 } \psi ( s ) = \operatorname* { i n f } \psi ( ( 0 , \eta ) ) \in [ 0 , \infty ) \qquad { \mathrm { a n d ~ } } \qquad \operatorname* { l i m } _ { s \to 0 } \varphi ( s ) = 0 .\tag{6.133}
$$

Combining this, (6.131), (6.132), the fact that $\psi \in C ^ { 1 } ( ( 0 , \eta ) , ( 0 , \infty ) )$ , the fact that $c > 0 .$ , and the fact that $\varphi ( 0 ) = 0$ establishes that $\varphi$ is continuous, that $\varphi | _ { ( 0 , \eta ) } \in C ^ { 1 } ( ( 0 , \eta ) , \mathbb { R } )$ , and that for all $x \in V$ with $0 < F ( x ) - F ( b ) < \eta$ it holds that

$$
\varphi ^ { \prime } ( F ( x ) - F ( b ) ) | \bigl ( \nabla F \bigr ) ( x ) | \geq 1 .\tag{6.134}
$$

We claim that for all $s , u \in I$ with $s \leq u , 0 < h ( s ) - F ( b ) < \eta ,$ and $| \theta ( s ) - b | + \varphi ( h ( s ) - F ( b ) ) < \rho$ it holds that

$$
| \theta ( u ) - b | \leq | \theta ( s ) - b | + \varphi ( h ( s ) - F ( b ) ) .\tag{6.135}
$$

To prove $\left( 6 . 1 3 5 \right)$ , let $s , u \in I .$ , assume $s \leq u$ , assume $0 < h ( s ) - F ( b ) < \eta .$ , and assume $| \theta ( s ) - b | + \varphi ( h ( s ) -$ $F ( b ) ) < \rho .$ Note that the fact that h is non-increasing, the assumption that for all $t \in I$ it holds that $h ( t ) > F ( b )$ , the fact that $h ( s ) - F ( b ) < \eta$ , and the fact that I is convex prove that $[ s , u ] \subseteq I$ and that for all $t \in [ s , u ]$ it holds that

$$
0 < h ( t ) - F ( b ) \leq h ( s ) - F ( b ) < \eta .\tag{6.136}
$$

This, (6.124), (6.134), the fact that $h = F \circ \theta$ , the fact that $\varphi | _ { ( 0 , \eta ) } \in C ^ { 1 } ( ( 0 , \eta ) , \mathbb { R } )$ , and the chain rule imply that for all $t \in [ s , u ]$ with $\theta ( t ) \in V$ it holds that

$$
- { \frac { \mathrm { d } } { \mathrm { d } t } } \varphi ( h ( t ) - F ( b ) ) = \varphi ^ { \prime } ( h ( t ) - F ( b ) ) | \theta ^ { \prime } ( t ) | ^ { 2 } \geq | \theta ^ { \prime } ( t ) | .\tag{6.137}
$$

Combining this, the fact that $\varphi ( [ 0 , \eta ) ) \subseteq [ 0 , \infty )$ , the fundamental theorem of calculus, and the triangle inequality shows that for all $v \in [ s , u ]$ with $\theta ( [ s , v ] ) \subseteq V$ it holds that

$$
\begin{array} { r l } & { | \theta ( v ) - b | \le | \theta ( s ) - b | + \displaystyle \int _ { s } ^ { v } | \theta ^ { \prime } ( t ) | \mathrm { d } t \le | \theta ( s ) - b | + \varphi ( h ( s ) - F ( b ) ) - \varphi ( h ( v ) - F ( b ) ) } \\ & { \qquad \le | \theta ( s ) - b | + \varphi ( h ( s ) - F ( b ) ) < \rho . } \end{array}\tag{6.138}
$$

This, (6.128), the fact that $| \theta ( s ) - b | < \rho$ , and the fact that θ is continuous establish that

$$
\theta ( [ s , u ] ) \subseteq V .\tag{6.139}
$$

Combining (6.138) and the fact that $s , u \in I$ with $s \leq u , 0 < h ( s ) - F ( b ) < \eta$ , and $| \theta ( s ) - b | + \varphi ( h ( s ) -$ $F ( b ) ) < \rho$ were arbitrary therefore proves $\left( 6 . 1 3 5 \right)$

Observe that (6.121), (6.125), (6.133), and the assumption that for all $t \in I$ it holds that $h ( t ) > F ( b )$ demonstrate that there exists $J \in$ N which satisfies for all $j \in \mathbb { N } \cap [ J , \infty )$ that

$$
0 < h ( t _ { j } ) - F ( b ) < \eta \qquad \mathrm { a n d } \qquad | \theta ( t _ { j } ) - b | + \varphi ( h ( t _ { j } ) - F ( b ) ) < \rho .\tag{6.140}
$$

Note that (6.135), (6.140), and the fact that $T = \operatorname* { s u p } ( I ) \notin I$ ensure that for all $j \in \mathbb { N } \cap [ J , \infty )$ it holds that

$$
0 \leq \operatorname* { l i m } _ { t  T } \lvert \theta ( t ) - b \rvert \leq \lvert \theta ( t _ { j } ) - b \rvert + \varphi ( h ( t _ { j } ) - F ( b ) ) .\tag{6.141}
$$

In addition, observe that (6.121) and (6.125) imply that

$$
\operatorname* { l i m } _ { j \to \infty } | \theta ( t _ { j } ) - b | = 0 \qquad \mathrm { a n d } \qquad \operatorname* { l i m } _ { j \to \infty } ( h ( t _ { j } ) - F ( b ) ) = 0 .\tag{6.142}
$$

This, (6.125), and (6.133) show that

$$
\operatorname* { l i m } _ { j \to \infty } \varphi ( h ( t _ { j } ) - F ( b ) ) = \operatorname* { l i m } _ { s \to 0 } \varphi ( s ) = 0 .\tag{6.143}
$$

Combining this and (6.142) establishes that

$$
\operatorname* { l i m } _ { j \to \infty } \big ( | \theta ( t _ { j } ) - b | + \varphi ( h ( t _ { j } ) - F ( b ) ) \big ) = 0 .\tag{6.144}
$$

This and (6.141) prove (6.122) in the case that for all $t \in I$ it holds that $h ( t ) > F ( b )$ . This completes the proof of (6.122). The proof of Proposition 6.9 is thus complete. □

Corollary 6.10 (Finite spherical length for GF trajectories of locally definable potentials). Let $n \in \mathbb { N }$ $\vartheta \in \mathbb { R } ^ { n }$ , let $U \subseteq \mathbb { R } ^ { n }$ be open, let $F \in C ^ { 1 } ( U , \mathbb { R } )$ be locally definable, let $I \subseteq \mathbb { R }$ be convex with nonempty interior, let $\theta \in C ^ { 1 } ( I , U )$ satisfy

$$
\theta ^ { \prime } = - ( \nabla F ) \circ \theta ,\tag{6.145}
$$

let $a , b \in U$ satisfy lim in $\mathrm { f } _ { t  \mathrm { i n f } ( I ) } | \theta ( t ) - a | =$ lim in $ { \dot { \bar { \cdot } } } _ { t \to \operatorname { s u p } ( I ) } | \theta ( t ) - b | = 0$ , and let $P \colon U \to \mathbb { R } ^ { n }$ satisfy for all $v \in U \backslash \{ \vartheta \}$ that $P ( v ) = | v - \partial ^ { } | ^ { - 1 } ( v - \vartheta )$ . Then $P \circ \theta$ has finite length.

Proof of Corollary 6.10. Throughout this proof let $G \colon U  \mathbb { R }$ satisfy $G = - F$ , let $J \subseteq \mathbb { R } , K \subseteq U$ satisfy

$$
J = \{ - t \colon t \in I \} \qquad \mathrm { a n d } \qquad K = \theta ( I ) \cup \{ a , b \} ,\tag{6.146}
$$

and let $\gamma \colon J  U$ satisfy for all $u \in J$ that

$$
\gamma ( u ) = \theta ( - u ) .\tag{6.147}
$$

Note that Proposition 6.9, the assumption that U is open, the assumption that $F \in C ^ { 1 } ( U , \mathbb { R } )$ , the assumption that F is locally definable, the assumption that I is convex with nonempty interior, the assumption that $\theta \in C ^ { 1 } ( I , U )$ , the assumption that for all $t \in I$ it holds that $\theta ^ { \prime } ( t ) = - ( \nabla F ) \big ( \theta ( t ) \big )$ , the fact that $b \in U .$ , and the assumption that lim in $\mathrm { f } _ { t  \mathrm { s u p } ( I ) } | \theta ( t ) - b | = 0$ demonstrate that

$$
\operatorname* { l i m } _ { t \to \operatorname* { s u p } ( I ) } | \theta ( t ) - b | = 0 .\tag{6.148}
$$

Moreover, observe that (6.146), (6.147), the fact that $G = - F .$ , the assumption that I is convex with nonempty interior, the assumption that $F \in C ^ { 1 } ( U , \mathbb { R } )$ , the assumption that $F$ is locally definable, the assumption that $\theta \in C ^ { 1 } ( I , U )$ , and the assumption that for all $t \in I$ it holds that $\theta ^ { \prime } ( t ) = - { \bigl ( } \nabla F { \bigr ) } { \bigl ( } \theta ( t ) { \bigr ) }$ ensure that J is convex with nonempty interior, that $G \in C ^ { 1 } ( U , \mathbb { R } )$ , that G is locally definable, that $\gamma \in C ^ { 1 } ( J , U )$ , and that for all $u \in J$ it holds that

$$
\gamma ^ { \prime } ( u ) = - \theta ^ { \prime } ( - u ) = \left( \nabla F \right) \left( \theta ( - u ) \right) = - \bigl ( \nabla G \bigr ) \bigl ( \gamma ( u ) \bigr ) .\tag{6.149}
$$

In the next step we note that (6.146), (6.147), and the assumption that lim $\begin{array} { r } { \operatorname* { i n f } _ { t \to \operatorname* { i n f } ( I ) } | \theta ( t ) - a | = 0 } \end{array}$ imply that

$$
\operatorname* { s u p } ( J ) = - \operatorname* { i n f } ( I ) \qquad { \mathrm { a n d } } \qquad \operatorname* { l i m i n f } _ { u \to \operatorname* { s u p } ( J ) } | \gamma ( u ) - a | = \operatorname* { l i m i n f } _ { t \to \operatorname* { i n f } ( I ) } | \theta ( t ) - a | = 0 .\tag{6.150}
$$

Combining this, (6.146), (6.147), (6.149), the fact that J is convex with nonempty interior, the fact that $G \in C ^ { 1 } ( U , \mathbb { R } )$ , the fact that G is locally definable, the fact that $\gamma \in C ^ { 1 } ( J , U )$ , the fact that $a \in U$ , and Proposition 6.9 (applied with $F \cap G , I \cap J , \theta \cap \gamma , b \cap a$ in the notation of Proposition 6.9) shows that

$$
\operatorname* { l i m } _ { t \to \operatorname* { i n f } ( I ) } | \theta ( t ) - a | = \operatorname* { l i m } _ { u \to \operatorname* { s u p } ( J ) } | \gamma ( u ) - a | = 0 .\tag{6.151}
$$

This, (6.146), (6.148), and the fact that θ is continuous establish that K is compact. Combining the fact that $K \subseteq U$ , the assumption that U is open, and the assumption that F is locally definable hence proves that there exist $N \in \mathbb { N } , x = ( x _ { 1 } , \ldots , x _ { N } ) \in K ^ { N } , \rho = ( \rho _ { 1 } , \ldots , \rho _ { N } ) \in ( 0 , \infty ) ^ { N }$ , and open $B _ { j } \subseteq U$ $j \in \{ 1 , \ldots , N \}$ , which satisfy that

(i) for all $j \in \{ 1 , \ldots , N \}$ it holds that $B _ { j } = \{ y \in \mathbb { R } ^ { n } \colon | y - x _ { j } | < \rho _ { j } \}$ and that $F | _ { B _ { j } }$ is definable and

(ii) it holds that

$$
\begin{array} { r } { K \subseteq \bigcup _ { j = 1 } ^ { N } B _ { j } \subseteq U . } \end{array}\tag{6.152}
$$

In the following let $V \subseteq U$ satisfy

$$
\begin{array} { r } { V = \bigcup _ { j = 1 } ^ { N } B _ { j } . } \end{array}\tag{6.153}
$$

Observe that (6.153) demonstrates that

$$
\begin{array} { r } { \mathrm { G r a p h } ( F | _ { V } ) = \bigcup _ { j = 1 } ^ { N } \mathrm { G r a p h } ( F | _ { B _ { j } } ) . } \end{array}\tag{6.154}
$$

This, (6.146), (6.152), (6.153), the fact that for all $j \in \{ 1 , \ldots , N \}$ it holds that $B _ { j }$ is open and that $F | _ { B _ { j } }$ is definable, the assumption that $F \in C ^ { 1 } ( U , \mathbb { R } )$ , and the assumption that $\theta \in \dot { C } ^ { 1 } ( I , U )$ ensure that $\check { V }$ is open, that $F | _ { V } \in C ^ { 1 } ( V , \mathbb { R } )$ , that $F | _ { V }$ is definable, and that $\theta \in C ^ { 1 } ( I , V )$ . Combining this, (6.153), the assumption that I is convex with nonempty interior, the assumption that for all $t \in I$ it holds that $\theta ^ { \prime } ( t ) = - { \bigl ( } \nabla F { \bigr ) } { \bigl ( } \theta ( t ) { \bigr ) }$ , the assumption that for all $v \in U \backslash \{ \vartheta \}$ it holds that $P ( v ) = ( v - \vartheta ) / | v - \vartheta |$ , and Theorem 6.8 (applied with $U \cap V , F \cap F | _ { V } , P \cap P | _ { V }$ in the notation of Theorem 6.8) establishes that

$$
\ln ( P \circ \theta ) = \ln ( ( P | _ { V } ) \circ \theta ) < \infty\tag{6.155}
$$

(cf. Definition 1.8). The proof of Corollary 6.10 is thus complete.

## 6.6 Infinite spherical length for an analytic potential

Theorem 1.1 above, in particular, proves for every definable potential $F \colon  { \mathbb { R } } ^ { n } \to  { \mathbb { R } }$ that the conclusion of the gradient conjecture at infinity applies. In the following elementary result, Proposition 6.11 below, we provide an explicit example of an analytic potential $F \colon  { \mathbb { R } ^ { \bar { 2 } } } \to$ R for which the conclusion of the gradient conjecture at infinity fails to hold (cf., for example, also [12, Theorem 1.1, page 592]). In particular, for every o-minimal structure $s$ we have that $F \colon  { \mathbb { R } } ^ { 2 } \to  { \mathbb { F } }$ in Proposition 6.11 fails to be S-definable. However, we have that F in Proposition 6.11 is locally $\mathbb { R } _ { \mathrm { a n } ^ { - } } \mathrm { d e f i n a b l e }$ (cf. Corollary 1.2 above).

Proposition 6.11 (An analytic potential with a spiralling gradient trajectory). Let $F \colon  { \mathbb { R } } ^ { 2 } \to$ R satisfy for all $x , y \in \mathbb { R }$ that

$$
\begin{array} { c } { { F ( x , y ) = - ( x ^ { 2 } + y ^ { 2 } ) - \frac 1 3 ( x ^ { 2 } + y ^ { 2 } ) ^ { 3 } } } \\ { { + ( x ^ { 2 } + y ^ { 2 } ) \big [ ( x ^ { 2 } - y ^ { 2 } ) \mathrm { s i n } ( x ^ { 2 } + y ^ { 2 } ) - 2 x y \mathrm { c o s } ( x ^ { 2 } + y ^ { 2 } ) \big ] } } \end{array}\tag{6.156}
$$

let $\theta \colon [ 0 , \infty )  \mathbb { R } ^ { 2 }$ satisfy for all $t \in [ 0 , \infty )$ that

$$
\theta ( t ) = e ^ { 2 t } \bigl ( \cos ( \textstyle { \frac { 1 } { 2 } } e ^ { 4 t } ) , \sin ( \textstyle { \frac { 1 } { 2 } } e ^ { 4 t } ) \bigr ) ,\tag{6.157}
$$

and let $P \colon \mathbb { R } ^ { 2 }  \mathbb { R } ^ { 2 }$ satisfy for all $v \in \mathbb { R } ^ { 2 } \backslash \{ 0 \}$ that $P ( v ) = | v | ^ { - 1 } v$ . Then

(i) it holds that F and θ are analytic,

(ii) it holds for all $t \in [ 0 , \infty )$ that $| \theta ( t ) | = e ^ { 2 t }$ 2

(iii) it holds for all $t \in [ 0 , \infty )$ that $( P \circ \theta ) ( t ) = \left( \cos ( \frac { 1 } { 2 } e ^ { 4 t } ) , \sin ( \frac { 1 } { 2 } e ^ { 4 t } ) \right)$

(iv) it holds for all $v \in \mathbb { R } ^ { 2 }$ that lim $\begin{array} { r } { \operatorname* { s u p } _ { t  \infty } | ( P \circ \theta ) ( t ) - v | > 0 } \end{array}$

(v) it holds that len $( P \circ \theta ) = \infty$ , and

(vi) it holds for all $t \in [ 0 , \infty )$ that $\theta ^ { \prime } ( t ) = - ( \nabla F ) ( \theta ( t ) )$

(cf. Definition 1.8).

Proof of Proposition 6.11. Throughout this proof let $E , T \colon \mathbb { R } \to \mathbb { R } ^ { 2 }$ satisfy for all $t \in \mathbb { R }$ that

$$
E ( t ) = ( \cos ( t ) , \sin ( t ) ) \qquad { \mathrm { a n d } } \qquad T ( t ) = ( - \sin ( t ) , \cos ( t ) ) ,\tag{6.158}
$$

let $r , \alpha \colon [ 0 , \infty ) \to ( 0 , \infty )$ satisfy for all $t \in [ 0 , \infty )$ that

$$
r ( t ) = e ^ { 2 t } \qquad \mathrm { a n d } \qquad \alpha ( t ) = { \frac { 1 } { 2 } } e ^ { 4 t } ,\tag{6.159}
$$

let $H = ( H ( \rho , t ) ) _ { ( \rho , t ) \in ( 0 , \infty ) \times \mathbb { R } } \colon ( 0 , \infty ) \times \mathbb { R } \to$ R satisfy for all $\rho \in ( 0 , \infty ) , t \in \mathbb { R }$ that

$$
H ( \rho , t ) = F ( \rho E ( t ) ) ,\tag{6.160}
$$

and let $( s _ { j } ) _ { j \in \mathbb { N } } \subseteq [ 0 , \infty )$ satisfy for all $j \in \mathbb N$ that

$$
s _ { j } = \frac { 1 } { 4 } \log ( 2 \pi j ) .\tag{6.161}
$$

Note that (6.156) and (6.157) prove item (i). Observe that (6.157), (6.159), and the fact that for all $t \in \mathbb { R }$ it holds that $( \cos ( t ) ) ^ { 2 } + ( \sin ( t ) ) ^ { 2 } = 1$ imply that for all $t \in [ 0 , \infty )$ it holds that

$$
| \theta ( t ) | = r ( t ) \sqrt { ( \cos ( \alpha ( t ) ) ) ^ { 2 } + ( \sin ( \alpha ( t ) ) ) ^ { 2 } } = r ( t ) = e ^ { 2 t } > 0 .\tag{6.162}
$$

This establishes item (ii). Note that item (ii), (6.157), and the assumption that for all $v \in \mathbb { R } ^ { 2 } \backslash \{ 0 \}$ it holds that $P ( v ) = | v | ^ { - 1 } v$ prove item (iii). Observe that item (iii), (6.159), (6.161), and the fact that for all $j \in \mathbb N$ it holds that cos $( \pi j ) = ( - 1 ) ^ { j }$ and sin $( \pi j ) = 0$ show that for all $j \in \mathbb N$ it holds that

$$
0 < s _ { j } < s _ { j + 1 } , \qquad \alpha ( s _ { j } ) = \pi j , \qquad ( P \circ \theta ) ( s _ { j } ) = ( ( - 1 ) ^ { j } , 0 ) , \qquad \mathrm { a n d } \qquad \operatorname* { l i m } _ { k \to \infty } s _ { k } = \infty .\tag{6.163}
$$

Combining this and the triangle inequality demonstrates that for all $v \in \mathbb { R } ^ { 2 }$ it holds that

$$
\operatorname* { l i m s u p } _ { t \to \infty } \lvert ( P \circ \theta ) ( t ) - v \rvert \geq \operatorname* { m a x } \{ \lvert ( 1 , 0 ) - v \rvert , \lvert ( - 1 , 0 ) - v \rvert \} \geq \frac { 1 } { 2 } \lvert ( 1 , 0 ) - ( - 1 , 0 ) \rvert = 1 > 0 .\tag{6.164}
$$

This establishes item (iv). Note that (6.163) ensures that for all $m \in \mathbb { N }$ it holds that

$$
\displaystyle \mathrm { l e n } ( P \circ \theta ) \geq \sum _ { j = 1 } ^ { m } \lvert ( P \circ \theta ) ( s _ { j + 1 } ) - ( P \circ \theta ) ( s _ { j } ) \rvert = \sum _ { j = 1 } ^ { m } 2 = 2 m\tag{6.165}
$$

(cf. Definition 1.8). This proves item (v). Observe that (6.158) and the fact that for all $t \in \mathbb { R }$ it holds that $( \cos ( t ) ) ^ { 2 } + ( \sin ( t ) ) ^ { 2 } = 1$ imply that for all $t \in \mathbb { R }$ it holds that

$$
E ^ { \prime } ( t ) = T ( t ) , \qquad | E ( t ) | = | T ( t ) | = 1 , \qquad \mathrm { a n d } \qquad \langle E ( t ) , T ( t ) \rangle = 0 .\tag{6.166}
$$

Next we combine (6.157), (6.158), (6.159), and the chain rule to obtain that for all $t \in [ 0 , \infty )$ it holds that

$$
\theta ( t ) = r ( t ) E ( \alpha ( t ) ) , \qquad 2 \alpha ( t ) = ( r ( t ) ) ^ { 2 } , \qquad r ^ { \prime } ( t ) = 2 r ( t ) , \qquad \mathrm { a n d } \qquad \alpha ^ { \prime } ( t ) = 2 e ^ { 4 t } = 2 ( r ( t ) ) ^ { 2 } .\tag{6.167}
$$

Furthermore, note that (6.156), (6.158), and (6.160) show that for all $\rho \in ( 0 , \infty ) , t \in \mathbb { R }$ it holds that

$$
\begin{array} { l } { { \displaystyle { \cal H } ( \rho , t ) = - \rho ^ { 2 } - \frac { \rho ^ { 6 } } { 3 } + \rho ^ { 4 } \big ( \cos ( 2 t ) \sin ( \rho ^ { 2 } ) - \sin ( 2 t ) \cos ( \rho ^ { 2 } ) \big ) } } \\ { { \displaystyle ~ = - \rho ^ { 2 } - \frac { \rho ^ { 6 } } { 3 } - \rho ^ { 4 } \sin ( 2 t - \rho ^ { 2 } ) . } } \end{array}\tag{6.168}
$$

Combining this and the chain rule demonstrates that for all $\rho \in ( 0 , \infty ) , t \in \mathbb { R }$ it holds that

$$
\begin{array} { r } { \left( \partial _ { \rho } H \right) \left( \rho , t \right) = - 2 \rho - 2 \rho ^ { 5 } - 4 \rho ^ { 3 } \sin ( 2 t - \rho ^ { 2 } ) + 2 \rho ^ { 5 } \cos ( 2 t - \rho ^ { 2 } ) \quad \mathrm { a n d } \quad \left( \partial _ { t } H \right) \left( \rho , t \right) = - 2 \rho ^ { 4 } \cos ( 2 t - \rho ^ { 2 } ) . } \end{array}\tag{6.169}
$$

This and (6.167) establish that for all $u \in [ 0 , \infty )$ it holds that

$$
\left( \partial _ { P } H \right) \left( r ( u ) , \alpha ( u ) \right) = - 2 r ( u ) - 2 ( r ( u ) ) ^ { 5 } + 2 ( r ( u ) ) ^ { 5 } = - 2 r ( u ) \quad \mathrm { a n d } \quad \left( \partial _ { t } H \right) \left( r ( u ) , \alpha ( u ) \right) = - 2 ( r ( u ) ) ^ { 4 } .\tag{6.170}
$$

Next, observe that item (i), (6.160), (6.166), and the chain rule ensure that for all $\rho \in ( 0 , \infty ) , t \in \mathbb { R } \mathrm { ~ i ~ }$ t holds that

$$
\left( \partial _ { \rho } H \right) \left( \rho , t \right) = \langle \left( \nabla F \right) \left( \rho E ( t ) \right) , E ( t ) \rangle \qquad { \mathrm { a n d } } \qquad \left( \partial _ { t } H \right) \left( \rho , t \right) = \rho \langle \left( \nabla F \right) \left( \rho E ( t ) \right) , T ( t ) \rangle .\tag{6.171}
$$

Combining (6.166) and the fact that for all $t \in \mathbb { R }$ it holds that $E ( t ) , T ( t )$ form an orthonormal basis of $\mathbb { R } ^ { 2 }$ therefore proves that for all $\rho \in ( 0 , \infty ) , t \in \mathbb { R }$ it holds that

$$
\left( \nabla F \right) \left( \rho E ( t ) \right) = \left( \partial _ { \rho } H \right) ( \rho , t ) E ( t ) + \frac { 1 } { \rho } \big ( \partial _ { t } H \big ) ( \rho , t ) T ( t ) .\tag{6.172}
$$

This, (6.167), and (6.170) imply that for all $u \in [ 0 , \infty )$ it holds that

$$
- \bigl ( \nabla F \bigr ) \bigl ( \theta ( u ) \bigr ) = 2 r ( u ) E \bigl ( \alpha ( u ) \bigr ) + 2 ( r ( u ) ) ^ { 3 } T ( \alpha ( u ) ) .\tag{6.173}
$$

In addition, note that (6.166), (6.167), and the chain rule show that for all $u \in [ 0 , \infty )$ it holds that

$$
\theta ^ { \prime } ( u ) = r ^ { \prime } ( u ) E ( \alpha ( u ) ) + r ( u ) \alpha ^ { \prime } ( u ) T ( \alpha ( u ) ) = 2 r ( u ) E ( \alpha ( u ) ) + 2 ( r ( u ) ) ^ { 3 } T ( \alpha ( u ) ) .\tag{6.174}
$$

Combining this and (6.173) establishes item (vi). The proof of Proposition 6.11 is thus complete.

## 6.7 Finite spherical length and directional convergence for DNN training flows

In Corollary 6.13 below we establish finite spherical length and directional convergence of DNN training flows. Our proof of Corollary 6.13 uses the elementary and well-known fact, which we recall in the following Lemma 6.12, that every curve with finite length is also convergent. Only for completeness, we include a proof for Lemma 6.12 here.

Lemma 6.12 (Convergence of curves with finite length). Let $n \in \mathbb N$ , let $U \subseteq \mathbb { R } ^ { n }$ be a set, let $I \subseteq \mathbb { R }$ be convex with nonempty interior, and let $\gamma \colon I  U$ have finite length (cf. Definition 1.8). Then there exists a unique $v \in \overline { { U } }$ such that

$$
\operatorname* { l i m } _ { t \to \operatorname* { s u p } ( I ) } | \gamma ( t ) - v | = 0 .\tag{6.175}
$$

Proof of Lemma 6.12. Throughout this proof let $a \in \mathrm { i n t } ( I ) , b \in \mathbb { R } \cup \{ \infty \} , J \subseteq I$ satisfy

$$
b = \operatorname* { s u p } ( I ) \qquad { \mathrm { a n d } } \qquad J = I \cap [ a , b ) ,\tag{6.176}
$$

let $\ell \colon J  [ 0 , \infty )$ satisfy for all $t \in J$ that

$$
\ell ( t ) = \mathrm { l e n } ( \gamma | _ { [ a , t ] } ) ,\tag{6.177}
$$

and let $L \in [ 0 , \infty ]$ satisfy

$$
L = \operatorname* { s u p } \{ \ell ( t ) \colon t \in J \} .\tag{6.178}
$$

Observe that (6.176), the fact that $a \in \operatorname { i n t } ( I )$ , and the fact that I is convex demonstrate that

$$
a < b , \qquad J = [ a , b ) , \qquad \mathrm { a n d } \qquad \operatorname { s u p } ( J ) = b .\tag{6.179}
$$

This, (6.177), and the assumption that γ has finite length ensure that for all $u , t \in J$ with $u \leq t$ it holds that

$$
0 = \ell ( a ) \leq \ell ( u ) \leq \ell ( t ) \leq \mathrm { l e n } ( \gamma ) < \infty .\tag{6.180}
$$

Combining this and (6.177) proves that for all $u , t \in J$ with $u \leq t$ it holds that

$$
| \gamma ( t ) - \gamma ( u ) | \leq \ell ( t ) - \ell ( u ) .\tag{6.181}
$$

Moreover, note that (6.178), (6.179), and (6.180) imply that

$$
L \in [ 0 , \operatorname { l e n } ( \gamma ) ] \qquad { \mathrm { a n d } } \qquad \operatorname * { l i m } _ { t \to b } \ell ( t ) = L .\tag{6.182}
$$

This and (6.179) show that there exists $( s _ { j } ) _ { j \in \mathbb { N } } \subseteq J$ which satisfies for all $j \in \mathbb N$ that

$$
s _ { j } < s _ { j + 1 } , \qquad \operatorname* { l i m } _ { k \to \infty } { s _ { k } } = b , \qquad { \mathrm { a n d } } \qquad \operatorname* { l i m } _ { k \to \infty } \ell ( s _ { k } ) = L .\tag{6.183}
$$

Observe that (6.180), (6.181), (6.182), and (6.183) establish that for all $\varepsilon \in ( 0 , \infty )$ there exists $K \in \mathbb N$ such that for all $j , k \in \mathbb { N } \cap [ K , \infty )$ it holds that

$$
| \gamma ( s _ { j } ) - \gamma ( s _ { k } ) | \leq \ell ( \operatorname* { m a x } \{ s _ { j } , s _ { k } \} ) - \ell ( \operatorname* { m i n } \{ s _ { j } , s _ { k } \} ) \leq L - \ell ( s _ { K } ) < \varepsilon .\tag{6.184}
$$

Combining this and the fact that for all $j \in \mathbb N$ it holds that $\gamma ( s _ { j } ) \in U$ demonstrates that there exists $v \in \overline { { U } }$ which satisfies

$$
\operatorname* { l i m } _ { j  \infty } \gamma ( s _ { j } ) = v .\tag{6.185}
$$

Note that (6.179), (6.181), (6.183), and (6.185) ensure that for all $t \in J$ it holds that

$$
| \gamma ( t ) - v | = \operatorname* { l i m } _ { j \to \infty } | \gamma ( t ) - \gamma ( s _ { j } ) | \leq \operatorname* { l i m } _ { j \to \infty } \big ( \ell ( s _ { j } ) - \ell ( t ) \big ) = L - \ell ( t ) .\tag{6.186}
$$

This, (6.176), (6.179), and (6.182) prove that

$$
0 \leq \operatorname* { l i m } _ { t  b } \operatorname* { s u p } \lvert \gamma ( t ) - v \rvert \leq \operatorname* { l i m } _ { t  b } \bigl ( L - \ell ( t ) \bigr ) = 0 .\tag{6.187}
$$

Combining this and (6.176) establishes that

$$
\operatorname* { l i m } _ { t \to \operatorname* { s u p } ( I ) } | \gamma ( t ) - v | = 0 .\tag{6.188}
$$

This, (6.176), and the triangle inequality imply that for all $w \in \overline { { U } }$ with lim $\begin{array} { r } { \operatorname* { s u p } _ { t  b } | \gamma ( t ) - w | = 0 } \end{array}$ it holds that

$$
0 \leq | v - w | \leq \operatorname* { l i m } _ { t \to b } \operatorname* { s u p } _ { } \big ( | v - \gamma ( t ) | + | \gamma ( t ) - w | \big ) = 0 .\tag{6.189}
$$

Combining this and (6.176) proves that v is the unique element of $\overline { U }$ which satisfies

$$
\operatorname* { l i m } _ { t \to \operatorname* { s u p } ( I ) } | \gamma ( t ) - v | = 0 .\tag{6.190}
$$

The proof of Lemma 6.12 is thus complete.

Corollary 6.13 (Finite spherical length for DNN training flows). Let $m , L \in \mathbb { N } , \ \ell = ( \ell _ { 0 } , \ell _ { 1 } , \dots , \ell _ { L } ) \in$ $\mathbb { N } ^ { L + 1 }$ , let $\begin{array} { r } { n = \sum _ { i = 1 } ^ { L } \ell _ { i } ( \ell _ { i - 1 } + 1 ) } \end{array}$ , let $A \in C ^ { 1 } ( \mathbb { R } , \mathbb { R } ) , \mathcal { L } \in C ^ { 1 } ( \mathbb { R } ^ { \ell _ { L } } \times \mathbb { R } ^ { \ell _ { L } } \times \mathbb { R } ^ { n } , [ 0 , \infty ) )$ be definable, for every $\theta = ( \theta _ { 1 } , \dots , \bar { \theta } _ { n } ) \in \mathbb { R } ^ { n } \ l e t \ N _ { \theta } ^ { v } = ( N _ { \theta , 1 } ^ { v } , \dots , N _ { \theta , \ell _ { n } } ^ { v } ) \colon \mathbb { R } ^ { \ell _ { 0 } } \to \mathbb { R } ^ { \ell _ { v } } , \ v \in \{ 0 , 1 , \dots , L \}$ , satisfy for all $v \in \{ 0 , 1 , \ldots , L - 1 \} , x = ( x _ { 1 } , \ldots , x _ { \ell _ { 0 } } ) \in \mathbb { R } ^ { \ell _ { 0 } } , i \in \{ 1 , 2 , \ldots , \ell _ { v + 1 } \}$ that

$$
\begin{array} { r l } & { \mathcal { N } _ { \theta , i } ^ { v + 1 } ( x ) = \theta _ { \ell _ { v + 1 } \ell _ { v } + i + \sum _ { h = 1 } ^ { v } \ell _ { h } ( \ell _ { h - 1 } + 1 ) } } \\ & { \qquad + \displaystyle \sum _ { j = 1 } ^ { \ell _ { v } } \theta _ { ( i - 1 ) \ell _ { v } + j + \sum _ { h = 1 } ^ { v } \ell _ { h } ( \ell _ { h - 1 } + 1 ) } \Big ( x _ { \operatorname* { m i n } \{ j , \ell _ { 0 } \} } \mathbb { 1 } _ { \{ 0 \} } ( v ) + A ( \mathcal { N } _ { \theta , j } ^ { v } ( x ) ) \mathbb { 1 } _ { \mathbb { R } } ( v ) \Big ) , } \end{array}\tag{6.191}
$$

let $\vartheta \in \mathbb { R } ^ { n } , x _ { 1 } , x _ { 2 } , \ldots , x _ { m } \in \mathbb { R } ^ { \ell _ { 0 } } , y _ { 1 } , y _ { 2 } , \ldots , y _ { m } \in \mathbb { R } ^ { \ell _ { L } }$ , let $F \colon  { \mathbb { R } } ^ { n } \to$ R satisfy for all $\boldsymbol { \theta } \in \mathbb { R } ^ { n }$ that

$$
F ( \theta ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \mathcal { L } \big ( \mathcal { N } _ { \theta } ^ { L } ( x _ { i } ) , y _ { i } , \theta \big ) ,\tag{6.192}
$$

and let $P \colon \mathbb { R } ^ { n } \to \mathbb { R } ^ { n }$ satisfy for all $\theta \in \mathbb { R } ^ { n } \backslash \{ \vartheta \}$ that $P ( \theta ) = | \theta - \vartheta | ^ { - 1 } ( \theta - \vartheta )$ . Then

(i) it holds that $F \in C ^ { 1 } ( \mathbb { R } ^ { n } , \mathbb { R } )$ is definable,

(ii) it holds for all $p \in \mathbb { R } ^ { n }$ that there exists $\theta \in C ^ { 1 } ( [ 0 , \infty ) , \mathbb { R } ^ { n } )$ such that for all $t \in [ 0 , \infty )$ it holds that $\theta ( 0 ) = p \ a n d \ \theta ^ { \prime } ( t ) = - \bigl ( \nabla F \bigr ) \bigl ( \theta ( t ) \bigr )$

(iii) it holds for all $\theta \in C ^ { 1 } ( [ 0 , \infty ) , \mathbb { R } ^ { n } )$ with $\forall t \in [ 0 , \infty ) \colon \theta ^ { \prime } ( t ) = - \bigl ( \nabla F \bigr ) \bigl ( \theta ( t ) \bigr )$ that $P \circ \theta$ has finite length, and

(iv) it holds for all $\theta \in C ^ { 1 } ( [ 0 , \infty ) , \mathbb { R } ^ { n } )$ with $\forall t \in [ 0 , \infty ) \colon \theta ^ { \prime } ( t ) = - \bigl ( \nabla F \bigr ) \bigl ( \theta ( t ) \bigr )$ that there exists a unique $v \in \mathbb { R } ^ { n }$ such that lim $\begin{array} { r } { \operatorname* { s u p } _ { t  \infty } | ( P \circ \theta ) ( t ) - v | = 0 } \end{array}$

Proof of Corollary 6.13. Throughout this proof let ${ \cal Q } _ { v } = ( Q _ { v } ^ { 1 } , \ldots , Q _ { v } ^ { \ell _ { v } } ) \colon \mathbb { R } ^ { n } \times \mathbb { R } ^ { \ell _ { 0 } }  \mathbb { R } ^ { \ell _ { v } } , v \in \{ 1 , \ldots , L \}$ satisfy for all $v \in \{ 1 , \ldots , L \} , w \in \mathbb { R } ^ { n } , z \in \mathbb { R } ^ { \ell _ { 0 } }$ that

$$
\begin{array} { r } { Q _ { v } ( w , z ) = \mathcal { N } _ { w } ^ { v } ( z ) . } \end{array}\tag{6.193}
$$

Observe that (6.193), the assumption that for all $w \in \mathbb { R } ^ { n }$ it holds that $\begin{array} { r } { F ( w ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \mathcal { L } (  { N _ { w } ^ { L } } ( x _ { i } ) , y _ { i } , w ) } \end{array}$ and the assumption that $\mathcal { L }$ takes values in $[ 0 , \infty )$ show that for all $w \in \mathbb { R } ^ { n }$ it holds that

$$
F ( w ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \mathcal { L } ( Q _ { L } ( w , x _ { i } ) , y _ { i } , w ) \geq 0 .\tag{6.194}
$$

In the next step we combine (6.191) and (6.193) to obtain that for all $w = ( w _ { 1 } , \dots , w _ { n } ) \in \mathbb { R } ^ { n } , z =$ $( z _ { 1 } , \dots , z _ { \ell _ { 0 } } ) \in \mathbb { R } ^ { \ell _ { 0 } } , i \in \{ 1 , \dots , \ell _ { 1 } \}$ it holds that

$$
Q _ { 1 } ^ { i } ( w , z ) = w _ { \ell _ { 1 } \ell _ { 0 } + i } + \sum _ { j = 1 } ^ { \ell _ { 0 } } w _ { ( i - 1 ) \ell _ { 0 } + j } z _ { j } .\tag{6.195}
$$

This, (6.191), (6.193), the fact that $A \in C ^ { 1 } ( \mathbb { R } , \mathbb { R } )$ is definable, and induction demonstrate that for all $v \in \{ 1 , \ldots , L \}$ it holds that $Q _ { v } \in C ^ { 1 } ( \mathbb { R } ^ { n } \times \mathbb { R } ^ { \ell _ { 0 } } , \mathbb { R } ^ { \ell _ { v } } )$ is definable. Combining (6.194) and the assumption that $\mathcal { L } \in C ^ { 1 } \overset { \cdot } { ( } \mathbb { R } ^ { \ell _ { L } } \times \mathbb { R } ^ { \ell _ { L } } \times \mathbb { R } ^ { n } , [ 0 , \infty ) \overset { ) } { ) }$ is definable hence establishes item (i).

In the following we prove item (ii). To prove item (ii), let $p \in \mathbb { R } ^ { n }$ . Note that the fact that $F \in$ $C ^ { 1 } ( \mathbb { R } ^ { n } , \mathbb { R } )$ and Peano’s existence theorem ensure that there exist $T \in ( 0 , \infty ] , \xi \in C ^ { 1 } ( [ 0 , T ) , \mathbb { R } ^ { n } )$ which satisfy that

(I) it holds that $\xi ( 0 ) = p ,$

(II) for all $t \in [ 0 , T )$ it holds that $\xi ^ { \prime } ( t ) = - \bigl ( \nabla F \bigr ) \bigl ( \xi ( t ) \bigr )$ , and

(III) there exist no $S \in ( T , \infty ] , \chi \in C ^ { 1 } ( [ 0 , S ) , \mathbb { R } ^ { n } )$ such that $\chi | _ { [ 0 , T ) } = \xi$ and that for all $u \in [ 0 , S )$ it holds that $\chi ^ { \prime } ( u ) = - \bigl ( \nabla F \bigr ) \bigl ( \chi ( u ) \bigr )$

Observe that item (II), the fact that $F \in C ^ { 1 } ( \mathbb { R } ^ { n } , \mathbb { R } )$ , and the chain rule imply that for all $t \in [ 0 , T )$ it holds that

$$
\begin{array} { r } { ( F \circ \xi ) ^ { \prime } ( t ) = \langle \left( \nabla F \right) \left( \xi ( t ) \right) , \xi ^ { \prime } ( t ) \rangle = - | \xi ^ { \prime } ( t ) | ^ { 2 } . } \end{array}\tag{6.196}
$$

This, (6.194), item (I), and the fundamental theorem of calculus prove that for all $s , t \in [ 0 , T )$ with $s \leq t$ it holds that

$$
0 \leq \int _ { s } ^ { t } | \xi ^ { \prime } ( u ) | ^ { 2 } \mathrm { d } u \leq \int _ { 0 } ^ { t } | \xi ^ { \prime } ( u ) | ^ { 2 } \mathrm { d } u = F ( p ) - F ( \xi ( t ) ) \leq F ( p ) .\tag{6.197}
$$

Combining the fact that $\xi \in C ^ { 1 } ( [ 0 , T ) , \mathbb { R } ^ { n } )$ , the fundamental theorem of calculus, and the Cauchy– Schwarz inequality therefore shows that for all $s , t \in [ 0 , T )$ with $s \leq t$ it holds that

$$
| \xi ( t ) - \xi ( s ) | \leq \int _ { s } ^ { t } | \xi ^ { \prime } ( u ) | \mathrm { d } u \leq \sqrt { t - s } \biggl ( \int _ { s } ^ { t } | \xi ^ { \prime } ( u ) | ^ { 2 } \mathrm { d } u \biggr ) ^ { 1 / 2 } \leq \sqrt { F ( p ) } \sqrt { t - s } .\tag{6.198}
$$

We claim that $T = \infty$ . We prove this claim by contradiction. In the following we thus assume that $T < \infty$ . Note that (6.198) and the assumption that $T < \infty$ demonstrate that there exists $q \in \mathbb { R } ^ { n }$ which satisfies

$$
\operatorname* { l i m } _ { t \nearrow T } \xi ( t ) = q .\tag{6.199}
$$

Observe that the fact that $q \in \mathbb { R } ^ { n }$ , the fact that $F \in C ^ { 1 } ( \mathbb { R } ^ { n } , \mathbb { R } )$ , and Peano’s existence theorem establish that there exist $\varepsilon \in ( 0 , \infty ) , \eta \in C ^ { 1 } ( [ T , T + \varepsilon ) , \mathbb { R } ^ { n } )$ which satisfy for all $t \in [ T , T + \varepsilon )$ that

$$
\eta ( T ) = q \qquad \mathrm { a n d } \qquad \eta ^ { \prime } ( t ) = - \bigl ( \nabla F \bigr ) \bigl ( \eta ( t ) \bigr ) .\tag{6.200}
$$

This, item (II), (6.199), and the fact that $\nabla F$ is continuous ensure that

$$
\operatorname* { l i m } _ { t \nearrow T } \xi ^ { \prime } ( t ) = - \big ( \nabla F \big ) ( q ) = \eta ^ { \prime } ( T ) .\tag{6.201}
$$

In the following let $\zeta \colon [ 0 , T + \varepsilon )  \mathbb { R } ^ { n }$ satisfy for all $t \in [ 0 , T + \varepsilon )$ that

$$
\zeta ( t ) = \left\{ \begin{array} { l l } { \xi ( t ) } & { : t \in [ 0 , T ) , } \\ { \eta ( t ) } & { : t \in [ T , T + \varepsilon ) } \end{array} \right. .\tag{6.202}
$$

Note that item (II), (6.199), (6.200), (6.201), (6.202), and the fact that $\xi \in C ^ { 1 } ( [ 0 , T ) , \mathbb { R } ^ { n } )$ and $\eta \in$ $C ^ { 1 } ( [ T , T + \varepsilon ) , \mathbb { R } ^ { n } )$ imply that $\zeta \in C ^ { 1 } ( [ 0 , T + \varepsilon )$ , R<sup>n</sup>) and that for all $t \in [ 0 , T + \varepsilon )$ it holds that

$$
\zeta | _ { [ 0 , T ) } = \xi \qquad \mathrm { a n d } \qquad \zeta ^ { \prime } ( t ) = - \bigl ( \nabla F \bigr ) \bigl ( \zeta ( t ) \bigr ) .\tag{6.203}
$$

Combining item (III) and the fact that $\zeta \in C ^ { 1 } ( [ 0 , T + \varepsilon ) , \mathbb { R } ^ { n } )$ hence proves that $T \geq T + \varepsilon > T$ . This contradiction proves that $T = \infty$ . Observe that the fact that $T = \infty ,$ , item (I), item (II), and the fact that $p \in \mathbb { R } ^ { n }$ was arbitrary establish item (ii).

In the following we prove item (iii) and item (iv). To prove item (iii) and item (iv), let $\theta \in$ $C ^ { 1 } ( [ 0 , \infty ) , \mathbb { R } ^ { n } )$ and assume for all $t \in [ 0 , \infty )$ that

$$
\theta ^ { \prime } ( t ) = - \bigl ( \nabla F \bigr ) \bigl ( \theta ( t ) \bigr ) .\tag{6.204}
$$

Note that (6.204), the fact that $F \in C ^ { 1 } ( \mathbb { R } ^ { n } , \mathbb { R } )$ is definable, the assumption that for all $w \in \mathbb { R } ^ { n } \backslash \{ \vartheta \}$ it holds that $P ( w ) = | w - \vartheta | ^ { - 1 } ( w - \vartheta )$ , and Theorem 6.8 (applied with $U \nleftarrow \mathbb { R } ^ { n } , I \nleftarrow [ 0 , \infty )$ in the notation of Theorem 6.8) show that

$$
\operatorname { l e n } ( P \circ \theta ) < \infty\tag{6.205}
$$

(cf. Definition 1.8). This and the fact that θ was arbitrary prove item (iii). Observe that (6.205) and Lemma 6.12 (applied with $U \curvearrowleft \mathbb { R } ^ { n } , I \curvearrowleft [ 0 , \infty ) , \gamma \curvearrowleft P \circ \theta$ in the notation of Lemma 6.12) demonstrate that there exists a unique $v \in \mathbb { R } ^ { n }$ such that

$$
\operatorname* { l i m } _ { t \to \infty } \operatorname* { s u p } _ { | ( P \circ \theta ) ( t ) - v | } = 0 .\tag{6.206}
$$

Combining this and the fact that θ was arbitrary establishes item (iv). The proof of Corollary 6.13 is thus complete. □

## Acknowledgements

This work has been partially supported by the National Natural Science Foundation of China (NSFC) under grant number W2531010. This work has also been supported by the Ministry of Culture and Science NRW as part of the Lamarr Fellow Network. We also gratefully acknowledge the Cluster of Excellence EXC 2044/2-390685587, Mathematics M¨unster: Dynamics-Geometry-Structure funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation).

## Use of large language models

The authors gratefully acknowledge extensive assistance from OpenAI’s GPT-5.6 Sol and GPT-6 Astra/Sol in the development of this work. These large language models were used via Chat-GPT/Codex as research collaborators for developing mathematical arguments and ideas, exploring proof strategies, testing arguments, identifying gaps and possible counterexamples, refining proofs and exposition, and reviewing the literature. All mathematical statements, claims, and proofs were independently checked by the authors and all of the sentences, statements, and arguments in this article are the sole responsibility of the authors.

## References

[1] V. I. Arnold. Some open problems in the theory of singularities. In Singularities, Part 1 (Arcata, Calif., 1981), volume 40 of Proc. Sympos. Pure Math., pages 57–69. Amer. Math. Soc., Providence, RI, 1983. Translated from the Russian.

[2] E. Bierstone and P. D. Milman. Semianalytic and subanalytic sets. Inst. Hautes Etudes Sci. Publ. Math., (67):5–42, 1988.

[3] J. Bochnak, M. Coste, and M.-F. Roy. Real algebraic geometry, volume 36 of Ergebnisse der Mathematik und ihrer Grenzgebiete (3) [Results in Mathematics and Related Areas (3)]. Springer-Verlag, Berlin, 1998. Translated from the 1987 French original, Revised by the authors.

[4] J. Bolte, A. Daniilidis, and A. Lewis. The Lojasiewicz inequality for nonsmooth subanalytic functions with applications to subgradient dynamical systems. SIAM J. Optim., 17(4):1205–1223, 2006.

[5] N. Boumal. An introduction to optimization on smooth manifolds. Cambridge University Press, Cambridge, 2023.

[6] D. Burago, Y. Burago, and S. Ivanov. A Course in Metric Geometry, volume 33 of Graduate Studies in Mathematics. American Mathematical Society, Providence, RI, 2001. doi:10.1090/gsm/033.

[7] Y. Cai, K. Zhou, J. Wu, S. Mei, M. Lindsey, and P. L. Bartlett. Implicit bias of gradient descent for non-homogeneous deep networks. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 6426–6504. PMLR, 2025. https://proceedings.mlr.press/v267/cai25m.html.

[8] L. Chizat and F. Bach. Implicit bias of gradient descent for wide two-layer neural networks trained with the logistic loss. In Proceedings of the 33rd Conference on Learning Theory, volume 125 of Proceedings of Machine Learning Research, pages 1305–1338. PMLR, 2020. https://proceeding s.mlr.press/v125/chizat20a.html.

[9] M. Coste. An introduction to O-minimal geometry. Lecture notes, November 1999. https://univ -rennes.hal.science/hal-05413940/document.

[10] D. D’Acunto and V. Grandjean. A gradient inequality at infinity for tame functions. Rev. Mat. Complut., 18(2):493–501, 2005. doi:10.5209/rev REMA.2005.v18.n2.16699.

[11] D. D’Acunto and V. Grandjean. On gradient at infinity of semialgebraic functions. Ann. Polon. Math., 87:39–49, 2005. doi:10.4064/ap87-0-4.

[12] A. Daniilidis, M. Haddou, and O. Ley. A convex function satisfying the lojasiewicz inequality but failing the gradient conjecture both at zero and infinity. Bull. Lond. Math. Soc., 54(2):590–608, 2022. doi:10.1112/blms.12586.

[13] S. Dereich, A. Jentzen, and A. Riekert. Averaged Adam accelerates stochastic optimization in the training of deep neural network approximations for partial diferential equation and optimal control problems. arXiv:2501.06081, 2025. https://arxiv.org/abs/2501.06081.

[14] T. Do, A. Jentzen, and A. Riekert. Non-convergence to the optimal risk for Adam and stochastic gradient descent optimization in the training of deep neural networks. arXiv:2503.01660, 2025. https://arxiv.org/abs/2503.01660.

[15] A. Fischer. Peano diferentiable extensions in o-minimal structures. Adv. Geom., 10(1):31–49, 2010.

[16] P. Fortuny and F. Sanz. Gradient vector fields do not generate twister dynamics. J. Diferential Equations, 174(1):91–100, 2001. doi:10.1006/jdeq.2000.3926.

[17] A. M. Gabrielov. Projections of semi-analytic sets. Functional Analysis and Its Applications, 2(4):282–291, 1968.

[18] D. Gallon, A. Jentzen, and F. Lindner. Blow up phenomena for gradient descent optimization methods in the training of artificial neural networks. arXiv:2211.15641, 2022. https://arxiv.or g/abs/2211.15641.

[19] P. Goldstein. Gradient flow of a harmonic function in R<sup>3</sup>. J. Diferential Equations, 247(9):2517– 2557, 2009.

[20] V. Grandjean. On the limit set at infinity of a gradient trajectory of a semialgebraic function. J. Diferential Equations, 233(1):22–41, 2007. doi:10.1016/j.jde.2006.10.009.

[21] V. Grandjean. Gradient trajectories for plane singular metrics I: Oscillating trajectories. Demonstr. Math., 47(1):69–78, 2014. doi:10.2478/dema-2014-0006.

[22] V. Grandjean and F. Sanz. On restricted analytic gradients on analytic isolated surface singularities. J. Diferential Equations, 255(7):1684–1708, 2013. doi:10.1016/j.jde.2013.05.020.

[23] F. Ichikawa. Thom’s conjecture on singularities of gradient vector fields. Kodai Math. J., 15(1):134– 140, 1992. doi:10.2996/kmj/1138039532.

[24] A. Jentzen, J. Kranz, and A. Riekert. PADAM: Parallel averaged Adam reduces the error for stochastic optimization in scientific machine learning. arXiv:2505.22085, 2025. Accepted for publication in J. Comput. Math.. https://arxiv.org/abs/2505.22085.

[25] A. Jentzen, B. Kuckuck, and P. von Wurstemberger. Mathematical introduction to deep learning: Methods, implementations, and theory. arXiv:2310.20360v3, 2025. https://arxiv.org/abs/2310 .20360.

[26] Z. Ji and M. Telgarsky. Directional convergence and alignment in deep learning. In Advances in Neural Information Processing Systems, volume 33, pages 17176–17186, 2020. https://proceedi ngs.neurips.cc/paper\_files/paper/2020/hash/c76e4b2fa54f8506719a5c0dc14c2eb9-Abstr act.html.

[27] G. Jones, J. Kirby, and T. Servi. Local interdefinability of Weierstrass elliptic functions. J. Inst. Math. Jussieu, 15(4):673–691, 2016.

[28] K. Jordan, Y. Jin, V. Boza, J. You, F. Cesista, L. Newhouse, and J. Bernstein. Muon: An optimizer for hidden layers in neural networks. Blog post, December 2024. https://kellerjordan.github. io/posts/muon/.

[29] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. In Proceedings of the 3rd International Conference on Learning Representations, 2015. https://arxiv.org/abs/1412.6980.

[30] J. Kranz, D. Gallon, S. Dereich, and A. Jentzen. SAD neural networks: Divergent gradient flows and asymptotic optimality via o-minimal structures. In Advances in Neural Information Processing Systems, volume 38, 2025. https://arxiv.org/abs/2505.09572.

[31] N. H. Kuiper. On gradient curves of an analytic function near a critical point. IHES Preprint M/91/36, 1991. https://omeka.ihes.fr/document/M\_91\_36.pdf.

[32] K. Kurdyka. On gradients of functions definable in o-minimal structures. Ann. Inst. Fourier (Grenoble), 48(3):769–783, 1998. doi:10.5802/aif.1638.

[33] K. Kurdyka, T. Mostowski, and A. Parusi´nski. Proof of the gradient conjecture of R. Thom. Ann. of Math. (2), 152(3):763–792, 2000. doi:10.2307/2661354.

[34] K. Kurdyka and A. Parusi´nski. Quasi-convex decomposition in o-minimal structures. Application to the gradient conjecture. In Singularity theory and its applications, volume 43 of Adv. Stud. Pure Math., pages 137–177. Math. Soc. Japan, Tokyo, 2006. doi:10.2969/aspm/04310137.

[35] S. L ojasiewicz. Sur la g´eom´etrie semi- et sous-analytique. Ann. Inst. Fourier (Grenoble), 43(5):1575– 1595, 1993.

[36] J. M. Lee. Introduction to smooth manifolds, volume 218 of Graduate Texts in Mathematics. Springer, New York, second edition, 2013.

[37] T. L. Loi. lojasiewicz inequalities in o-minimal structures. Manuscripta Math., 150(1-2):59–72, 2016.

[38] S. Lojasiewicz. Sur les trajectoires du gradient d’une fonction analytique. In Geometry seminars, 1982–1983 (Bologna, 1982/1983), pages 115–117. Univ. Stud. Bologna, Bologna, 1984.

[39] K. Lyu and J. Li. Gradient descent maximizes the margin of homogeneous neural networks. In Proceedings of the 8th International Conference on Learning Representations, 2020. https://arxi v.org/abs/1906.05890.

[40] R. Moussu. Sur la dynamique des gradients. Existence de vari´et´es invariantes. Math. Ann., 307(3):445–460, 1997. doi:10.1007/s002080050043.

[41] A. Parusi´nski. On the gradient conjecture for definable functions. In M. J. Edmundo, D. Richardson, and A. Wilkie, editors, Lecture Notes in Real Algebraic and Analytic Geometry, pages 143–158. Cuvillier Verlag, G¨ottingen, 2005. https://cuvillier.de/get/ebook/3812/9783865375575.pdf.

[42] F. Sanz. Non-oscillating solutions of analytic gradient vector fields. Ann. Inst. Fourier (Grenoble), 48(4):1045–1067, 1998. doi:10.5802/aif.1648.

[43] F. Sanz S´anchez. Half a century with the problem of the gradient of an analytic function. In Handbook of geometry and topology of singularities V. Foliations, pages 335–395. Springer, Cham, [2024] ©2024.

[44] D. Soudry, E. Hofer, M. S. Nacson, S. Gunasekar, and N. Srebro. The implicit bias of gradient descent on separable data. J. Mach. Learn. Res., 19:Paper No. 70, 57, 2018.

[45] R. Thom. Probl\`emes rencontr´es dans mon parcours math´ematique: un bilan. Inst. Hautes Etudes<sup>´</sup> Sci. Publ. Math., (70):199–214, 1989.

[46] L. van den Dries. A generalization of the Tarski–Seidenberg theorem, and some nondefinability results. Bulletin of the American Mathematical Society (New Series), 15(2):189–193, 1986.

[47] L. van den Dries. Tame topology and o-minimal structures, volume 248 of London Mathematical Society Lecture Note Series. Cambridge University Press, Cambridge, 1998. doi:10.1017/CBO9780511525919.

[48] A. J. Wilkie. o-minimal structures. Number 326, pages Exp. No. 985, vii, 131–142. 2009. S´eminaire Bourbaki. Vol. 2007/2008.