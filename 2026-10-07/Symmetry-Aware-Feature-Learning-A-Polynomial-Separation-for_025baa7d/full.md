3 Main results 7   
3.1 Weak directional recovery (p ≥ 3) . . 7   
3.2 Weak subspace recovery (p = 2) . . 9

# Symmetry-Aware Feature Learning: A Polynomial Separation for Multi-Index Models

Jivan Waber<sup>1</sup>, Vanessa Piccolo<sup>1</sup>, Yatin Dandi<sup>1,2</sup>, Florent Krzakala<sup>1</sup>

<sup>1</sup> Information, Learning and Physics Laboratory

<sup>2</sup> Statistical Physics of Computation Laboratory École Polytechnique Fédérale de Lausanne (EPFL)

## Abstract

We establish a polynomial sample complexity separation between symmetry-aware and symmetry-agnostic feature learning. We study growing-rank multi-index models with highdimensional Gaussian covariates in $\mathbb { R } ^ { d }$ and $r ~ = \Theta ( d ^ { \delta } )$ teacher directions forming a cyclic symmetry orbit, where $0 < \delta < 1 / 2$ . We compare three ways of exploiting this structure: architectural weight sharing, data augmentation over the full symmetry group, and learning without access to the symmetry. In particular, we analyze a symmetry-tied convolutional network, an untied network, and the same untied network trained with full-group data augmentation, using spherical online SGD with correlation loss. For a class of polynomial links with information exponent $p \geq 3 .$ , we prove matching sample complexity bounds up to logarithmic factors: the tied and augmented learners achieve weak directional recovery in $\widetilde { \Theta } ( d ^ { p - 1 } )$ samples, whereas the symmetry-agnostic learner requires $\widetilde { \Theta } ( r d ^ { p - 1 } )$ . For the pure quadratic Hermite link, the same separation holds for weak recovery of the teacher subspace, with sample complexities $\widetilde { \Theta } ( d )$ and $\widetilde { \Theta } ( r d )$ , respectively. Thus, full-group data augmentation matches the sample eficiency of architectural weight sharing, and both provide a polynomial advantage over training without symmetry. For $p \geq 3$ , the proof reveals a two-stage mechanism: fluctuations at initialization select one direction in the teacher orbit, after which localized growth amplifies its overlap to the weak recovery scale while competing overlaps remain near their initialization scale.

## Contents

## 1 Introduction

2 Setting 5   
2.1 Teacher model 5   
2.2 Learning procedures 6

## 4 Related work

5 Proof outline 12   
5.1 Proof overview 12   
5.2 Upper bounds for directional recovery 14   
5.3 Upper bounds for quadratic subspace recovery 15   
5.4 Non-escape and lower bounds 16   
5.5 Proofs of the main results 17   
6 Correlation dynamics and auxiliary estimates 17   
6.1 Correlation recursion and population drift 18   
6.2 Coordinate noise estimates 20   
6.3 Uniform control of the stochastic gradient 21   
7 Proof of weak recovery for the tied student 21   
7.1 Proof strategy . 21   
7.2 Microscopic selection . 23   
7.3 Growth to a macroscopic level . 27   
8 Proofs for the augmented and untied students 31   
8.1 Componentwise reductions . 31   
8.2 One-weight untied dynamics . 32   
8.3 Proof of the untied weak recovery bound . 33   
8.4 Proof of weak orbit coverage . 35   
9 Quadratic link and weak subspace recovery 38   
9.1 Proof strategy . . 38   
9.2 Total overlap and initialization 39   
9.3 Recursion and population growth 40   
9.4 Noise and gradient moments . . 42   
9.5 Growth and proof of the tied upper bound . 43   
9.6 Augmented and untied upper bounds . 47   
10 Lower bounds for weak recovery 50   
10.1 Proof strategy . . 50   
10.2 Initialization and escape times 50   
10.3 An upper comparison for the correlations 51   
10.4 Noise before escape . 53   
10.5 Proof of the non-escape theorem 54   
11 Conclusion 56   
References 57   
A Teacher geometry and initialization 61   
B Proofs of stochastic gradient estimates 67   
B.1 Feature-gradient covariance 67   
B.2 Coordinate noise moments . 67   
B.3 Uniform control of the stochastic gradient 70

## 1 Introduction

Symmetry is one of the most efective inductive biases in machine learning. When transformations of the input preserve the task, the same feature need not be learned independently at every transformed location. Convolutional neural networks exploit this principle through weight sharing, while groupequivariant architectures extend it to more general transformations (Cohen & Welling, 2016; Kondor & Trivedi, 2018; Bronstein et al., 2021). More broadly, geometric deep learning views symmetry and geometry as organizing principles for designing neural architectures (Bronstein et al., 2021).

There are two common ways to exploit this prior knowledge. One is architectural: constrain the model through weight sharing or equivariance. The other is algorithmic: keep the model unconstrained but train on transformed copies of each observation, as in data augmentation. Group averaging provides a natural mathematical description of the latter (Chen et al., 2020; Lyle et al., 2020). A substantial literature has shown that invariance can improve variance, complexity, and generalization (Elesedy & Zaidi, 2021; Mei et al., 2021), and has quantified advantages of convolutional structure in infinite-width regimes (Cagnetta et al., 2023). Here we study how symmetry changes the sample complexity of feature learning by stochastic gradient descent (SGD). In particular, we ask

How much does symmetry reduce the number of samples that SGD needs to discover a relevant feature, and how do weight sharing and data augmentation compare?

Answering this question mathematically requires analyzing the high-dimensional, nonconvex SGD dynamics from random initialization, rather than comparing hypothesis classes or estimators after optimization.

Feature learning in single- and multi-index models To study this question, we focus on single- and multi-index models of the form

$$
f ^ { * } ( x ) = g \left( \langle w _ { 1 } ^ { * } , x \rangle , \dots , \langle w _ { r } ^ { * } , x \rangle \right) ,
$$

where $g \colon  { \mathbb { R } } ^ { r } \to  { \mathbb { R } }$ is the link function and $w _ { 1 } ^ { * } , \ldots , w _ { r } ^ { * }$ are the latent directions. These models have been used to study feature recovery, computational barriers, and the dynamics of gradient-based learning, see e.g. Dudeja & Hsu (2018); Barbier et al. (2019); Ben Arous et al. (2021); Arnaboldi et al. (2023); Abbe et al. (2023); Dandi et al. (2024b); Troiani et al. (2025); Şimşek et al. (2025). In particular, for Gaussian single-index models, the degree of the first nonzero Hermite coeficient of the link function–its information exponent–determines the time scale on which online SGD escapes the nearly uninformative initialization regime (Ben Arous et al., 2021).

Interactions between several latent directions produce richer dynamics. Finite-rank analyses of online SGD show that competing correlations can suppress or select one another, leading to sequential feature recovery. In multi-spiked tensor PCA, Ben Arous et al. (2026) characterize this through a greedy maximum selection mechanism, which determines the order in which latent directions are recovered. Related selection mechanisms have also been identified in shallow multiindex models with growing numbers of features (Ren et al., 2025), while other recent works study growing collections of approximately independent or diverse features (Oko et al., 2024; Ren & Lee, 2025). Our setting combines the dificulty of interacting directions with that of growing rank: the number of competing directions grows with the dimension, while the directions are strongly dependent because they are symmetry-related copies of a single hidden feature. This structure raises a question that is absent from fixed-rank models: can a learner avoid paying separately for every direction by exploiting their common origin? A more detailed literature review is given in Section 4.

A symmetric growing-rank model We study a growing-rank model designed to isolate this efect:

$$
f ^ { * } ( x ) = \frac { 1 } { \sqrt { r } } \sum _ { k = 0 } ^ { r - 1 } \sigma \left( \langle \Pi ^ { k } w ^ { * } , x \rangle \right) ,
$$

where Π generates a group action of size r. Thus the r latent directions $w ^ { * } , \Pi w ^ { * } , \ldots , \Pi ^ { r - 1 } w ^ { * }$ are not independent parameters: they form the orbit of a single direction $w ^ { * }$ . For cyclic shifts, this is a convolutional teacher. We consider the growing-rank regime $r \asymp d ^ { \delta }$ with $\begin{array} { r } { 0 < \delta < \frac { 1 } { 2 } } \end{array}$

We analyze two regimes for the link function σ. Our main results concern polynomial links with information exponent $p \geq 3$ . We also treat the pure quadratic Hermite link $\sigma = h _ { 2 } .$ , for which the individual orbit directions are generally not identifiable and the natural recovery target is instead the subspace spanned by the teacher orbit.

We compare three ways of learning this target with spherical online SGD. The tied procedure (T) uses a single trainable weight shared across its entire orbit. The augmented procedure (A) uses an untied two-layer model but averages each sample loss over the full group orbit. The untied procedure (U) uses the same untied model without augmentation. The latter two therefore have the same parametrization; their only diference is whether the symmetry is used during training or not.

An r-fold separation Our main result shows that exploiting the symmetry reduces the sample complexity of feature recovery by a factor r. For polynomial links with information exponent $p \geq 3$ , we consider weak directional recovery: some learned weight must reach a fixed, non-vanishing correlation with an element of the identifiable teacher orbit. For the dimension dependent step sizes prescribed in our analysis, we prove matching upper and lower bounds (up to logarithmic factors):

$$
n _ { \mathsf { T } } = \widetilde \Theta ( d ^ { p - 1 } ) , \qquad n _ { \mathsf { A } } = \widetilde \Theta ( d ^ { p - 1 } ) , \qquad n _ { \mathsf { U } } = \widetilde \Theta ( r d ^ { p - 1 } ) .
$$

The scale $\widetilde { \Theta } ( d ^ { p - 1 } )$ is the weak recovery scale of vanilla online SGD for Gaussian single-index models with information exponent $p$ (Ben Arous et al., 2021; Damian et al., 2023). Thus, despite the teacher rank growing as $r \asymp d ^ { \delta }$ , weight sharing and exact data augmentation recover a teacher feature at the single-index scale, whereas the symmetry-agnostic untied learner requires an additional factor equal to the number of orbit directions. Since $r \asymp d ^ { \delta }$ , this yields a polynomial separation in the dimension. We further show that, with width $s \gtrsim r$ log r, the untied and augmented procedures cover the entire teacher orbit within their respective weak recovery sample complexity scales.

For the pure Hermite-2 link, individual orbit directions are generally unidentifiable. Thus, we instead consider weak subspace recovery: some learned weight must acquire a fixed, non-vanishing fraction of its squared norm in the subspace spanned by the teacher orbit. We obtain the same r-fold separation: the tied and augmented procedures require $\widetilde { \Theta } ( d )$ samples, whereas the untied learner requires $\widetilde { \Theta } ( r d )$

Proof mechanism For $p \geq 3$ , the proof reveals a two-step selection-and-growth mechanism. At random initialization, one of the $r \asymp d ^ { \delta }$ correlations with the teacher orbit is slightly larger than the others. During an initial microscopic phase, the advantage of this leading correlation over the competing correlations is amplified. In a subsequent macroscopic phase, the selected correlation grows to a macroscopic level while the competitors remain microscopic. For the pure quadratic Hermite link, individual orbit directions are not identifiable. Rather than tracking a single leading correlation, we track the squared norm of the projection onto the teacher subspace, which undergoes multiplicative growth.

The three procedures difer in how symmetry afects these dynamics. Weight sharing ties all transformed copies of the feature to a single trainable parameter, whereas data augmentation averages the stochastic gradient over the orbit. Although augmentation leaves the population objective unchanged, orbit averaging reduces stochastic fluctuations. With the corresponding learning-rate rescaling, the tied and augmented procedures have the same efective correlation dynamics. In the symmetry-agnostic untied procedure, the corresponding efective dynamics are slower by a factor $^ { r , }$ yielding the r-fold separation in both recovery regimes. See Subsection 5.1 for a fuller proof overview.

## 2 Setting

## 2.1 Teacher model

Let $d \geq 1$ , and let $r \geq 1$ be an integer dividing d. Define $\Delta : = d / r .$ and let $\Pi \in \mathbb { R } ^ { d \times d }$ denote the permutation matrix corresponding to a cyclic shift by $\Delta$ coordinates. Thus $\Pi ^ { r } = \Pi _ { d } .$ , and the cyclic group generated by Π is

$$
\mathcal { G } : = \left\{ \mathrm { I } _ { d } , \Pi , \ldots , \Pi ^ { r - 1 } \right\} .
$$

We observe a sequence of samples $( x _ { i } , y _ { i } ) _ { i = 1 } ^ { n }$ , where $x _ { i } \overset { \mathrm { i . i . d . } } { \sim } \mathcal { N } ( 0 , \mathrm { I } _ { d } )$ and $y _ { i } = f ^ { * } ( x _ { i } )$ , where

$$
f ^ { * } ( x ) : = \frac { 1 } { \sqrt { r } } \sum _ { k = 0 } ^ { r - 1 } \sigma \left( \langle \Pi ^ { k } w ^ { * } , x \rangle \right) .\tag{1}
$$

Here $\boldsymbol { w } ^ { * } \in \mathbb { S } ^ { d - 1 }$ is the teacher direction and $\sigma : \mathbb { R }  \mathbb { R }$ is a known nonlinear link function. The teacher depends on $w ^ { * }$ only through its orbit $\{ \Pi ^ { k } w ^ { * } \colon 0 \leq k < r \}$ . Thus the statistical model is invariant under $w ^ { * } \mapsto \Pi ^ { k } w ^ { * }$

We analyze two regimes for the link function.

Assumption 1 (Link function). The link σ belongs to one of the following two classes.

(i) Polynomial links with information exponent $p \geq 3$ . The link is a fixed polynomial with finite Hermite expansion

$$
\sigma ( z ) = \sum _ { j = p } ^ { P } a _ { j } h _ { j } ( z ) , \qquad 3 \leq p \leq P < \infty , \qquad a _ { p } \neq 0 ,
$$

where $( h _ { j } ) _ { j \geq 0 }$ is the orthonormal Hermite basis of $L ^ { 2 } ( \mathcal { N } ( 0 , 1 ) )$ ). The integers $p , P$ and the coeficients $( a _ { j } ) _ { j = p } ^ { P }$ are fixed independently of d. If $p$ is even, we also assume that $\sigma$ is even.

(ii) Pure quadratic link.

$$
\sigma ( z ) = h _ { 2 } ( z ) = \frac { 1 } { \sqrt { 2 } } ( z ^ { 2 } - 1 ) .
$$

In case of Assumption 1(i), we call $p = p _ { \sigma } : = \operatorname* { m i n } \{ j \geq 1 \colon a _ { j } \neq 0 \}$ the information exponent of the link. Since $p \geq 3$ , the Hermite coeficients of degrees $1 , \ldots , p - 1$ vanish.

Remark 2.1 (On the link assumptions). The finite Hermite expansion in Assumption $1 ( \mathrm { i } )$ is mainly technical and could be relaxed under suitable regularity and moment-growth conditions on the link. The parity condition for $p$ ensures that the symmetry of the leading-order dynamics agrees with the exact symmetry of the statistical model. This condition can be removed by considering paired opposite initializations; we do not establish this extension here.

For $u \in \mathbb { S } ^ { d - 1 }$ , define the correlations

$$
m _ { k } ( u ) : = \left. \Pi ^ { k } w ^ { * } , u \right. , \qquad 0 \leq k < r .\tag{2}
$$

The teacher direction $w ^ { * }$ is identifiable only up to cyclic shifts. When $\sigma$ is even, there is an additional global sign ambiguity. We therefore define the identifiable teacher orbit by

$$
\mathcal { O } _ { \sigma } ( w ^ { * } ) : = \left\{ \begin{array} { l l } { \{ \pm \Pi ^ { k } w ^ { * } : 0 \leq k < r \} , } & { \sigma \mathrm { ~ i s ~ e v e n , } } \\ { \{ \Pi ^ { k } w ^ { * } : 0 \leq k < r \} , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.
$$

For $u \in \mathbb { S } ^ { d - 1 }$ , define its correlation with the identifiable teacher orbit by

$$
M _ { \sigma } ( u ) : = \operatorname* { m a x } _ { v \in \mathcal { O } _ { \sigma } ( w ^ { * } ) } \langle v , u \rangle .\tag{3}
$$

The quadratic case in Assumption 1(ii) is qualitatively diferent: the individual directions in the teacher orbit are generally not identifiable. Accordingly, weak recovery in this case will instead be formulated in terms of the subspace spanned by the teacher orbit (see Definition 3.6).

## 2.2 Learning procedures

We compare three learning procedures: a tied convolutional student, an untied student, and the same untied student trained with data augmentation. Throughout, let

$$
\ell ( y , z ) : = - y z
$$

denote the correlation loss.

Definition 2.2 (Tied convolutional student). The tied student is parametrized by the weight $w \in \mathbb { S } ^ { d - 1 }$ and is defined by

$$
f _ { w } ^ { \mathsf { T } } ( x ) : = { \frac { 1 } { \sqrt { r } } } \sum _ { k = 0 } ^ { r - 1 } \sigma \left( \langle \Pi ^ { k } w , x \rangle \right) .
$$

Its sample loss is

$$
\begin{array} { r } { L ^ { \mathsf { T } } ( w ; x , y ) : = \ell \left( y , f _ { w } ^ { \mathsf { T } } ( x ) \right) . } \end{array}
$$

Definition 2.3 (Untied student). Let $W = ( w ^ { 1 } , \dots , w ^ { s } ) \in ( \mathbb { S } ^ { d - 1 } ) ^ { s }$ with $s = s ( d ) \geq 1$ . The untied student is parametrized by W and defined by

$$
f _ { W } ^ { \mathsf { U } } ( x ) : = { \frac { 1 } { \sqrt { s } } } \sum _ { j = 1 } ^ { s } \sigma \left( \langle w ^ { j } , x \rangle \right) .
$$

Its sample loss is

$$
L ^ { \mathsf { U } } ( W ; x , y ) : = \ell \left( y , f _ { W } ^ { \mathsf { U } } ( x ) \right) .
$$

Unlike the tied student, $w ^ { 1 } , \ldots , w ^ { s }$ are not constrained to belong to a common orbit under ${ \mathcal { G } } .$ Definition 2.4 (Untied student with data augmentation). The augmented procedure uses the same model class $f _ { W } ^ { \mathsf { U } }$ , but replaces the sample loss by its group average:

$$
L ^ { \mathsf { A } } ( W ; x , y ) : = { \frac { 1 } { r } } \sum _ { k = 0 } ^ { r - 1 } \ell \left( y , f _ { W } ^ { U } ( \Pi ^ { k } x ) \right) .
$$

Set $s _ { \mathsf { T } } : = 1$ and $s _ { \mathsf { U } } = s _ { \mathsf { A } } : = s \in \mathbb { N }$ , with $s \geq 1$ . For $\mathsf { k } \in \{ \mathsf { T } , \mathsf { U } , \mathsf { A } \}$ , define $\mathcal { M } _ { \sf k } : = ( \mathbb { S } ^ { d - 1 } ) ^ { s _ { \sf k } }$ , so that

$$
\mathcal { M } _ { \sf T } = \mathbb { S } ^ { d - 1 } , \qquad \mathcal { M } _ { \sf U } = \mathcal { M } _ { \sf A } = ( \mathbb { S } ^ { d - 1 } ) ^ { s } .
$$

For $\theta \in \mathcal { M } _ { \sf k }$ , let

$$
\Phi ^ { \mathsf { k } } ( \theta ) : = \mathbb { E } \left[ L ^ { \mathsf { k } } ( \theta ; x , y ) \right]\tag{4}
$$

denote the corresponding population objective. Since the Gaussian input distribution is invariant under Π and the teacher satisfies $f ^ { * } ( \Pi ^ { k } x ) = f ^ { * } ( x )$ , the augmented and unaugmented untied procedures have the same population objective:

$$
\Phi ^ { \mathsf { A } } ( W ) = \Phi ^ { \mathsf { U } } ( W ) .
$$

Thus data augmentation leaves the population drift unchanged.

Online spherical SGD For $\theta = ( w ^ { 1 } , \ldots , w ^ { s _ { \boldsymbol { \mathrm { k } } } } ) \in \mathcal { M } _ { \boldsymbol { \mathsf { k } } }$ , write

$$
\nabla _ { w ^ { j } } L ( \theta ) : = ( \mathrm { I } _ { d } - w ^ { j } w ^ { j \top } ) \widehat \nabla _ { w ^ { j } } L ( \theta )
$$

for the spherical gradient, where $\widehat { \nabla } _ { w ^ { j } }$ denotes the Euclidean gradient. Given an initial condition $\theta _ { 0 } \in \mathcal { M } _ { \sf k }$ and a step size $\eta _ { d } ^ { \mathsf { k } } > 0$ , single-pass spherical SGD is defined componentwise by

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { \widetilde { w } _ { t } ^ { j } = w _ { t - 1 } ^ { j } - \eta _ { d } ^ { \mathsf { k } } \nabla _ { w ^ { j } } L ^ { \mathsf { k } } ( \theta _ { t - 1 } ; x _ { t } , y _ { t } ) , } \\ { w _ { t } ^ { j } = \frac { \widetilde { w } _ { t } ^ { j } } { \| \widetilde { w } _ { t } ^ { j } \| } , } \end{array} \right. \qquad \mathrm { ~ 1 \le ~ } j \le s _ { \mathsf { k } } . } \end{array}\tag{5}
$$

For the tied procedure we identify $\theta _ { t } = w _ { t } ^ { 1 , \mathsf { T } }$ with $w _ { t }$ . For the untied and augmented procedures, $\theta _ { t } = W _ { t } = ( w _ { t } ^ { 1 } , \ldots , w _ { t } ^ { s } )$ . At each iteration, the algorithm uses one new independent observation $( x _ { t } , y _ { t } )$ . In the augmented procedure, this same observation additionally generates the r transformed inputs $\Pi ^ { k } x _ { t }$ for $0 \leq k < r$ . Hence the number of SGD iterations equals the number of fresh samples, although augmentation incurs a larger per-iteration computational cost.

## 3 Main results

We now state our main results on weak recovery by online spherical SGD. We work on a probability space $( \Omega , \mathcal { F } , \mathbb { P } )$ supporting the teacher direction $w ^ { * }$ , all parameter initializations, and an i.i.d. sequence of Gaussian covariates $( x _ { i } ) _ { i \geq 1 }$ used by online SGD.

Assumption 2 (Random teacher and initialization). The teacher direction is sampled according to $w ^ { * } \sim \mathrm { U n i f } ( \mathbb { S } ^ { d - 1 } )$ . For each learning procedure k $\in \{ \mathsf { T } , \mathsf { U } , \mathsf { A } \}$ , initialize

$$
w _ { 0 } ^ { 1 , \ k } , \ldots , w _ { 0 } ^ { s _ { \ k } , \ k \ \stackrel { \mathrm { i . i . d . } } { \sim } } \operatorname { U n i f } ( \mathbb { S } ^ { d - 1 } ) ,
$$

where $s \tau = 1$ and $s _ { \mathsf { U } } = s _ { \mathsf { A } } = s$ . The teacher, all parameter initializations, and the Gaussian covariates $( x _ { i } ) _ { i \geq 1 }$ are mutually independent.

Unless stated otherwise, P denotes probability with respect to all of these sources of randomness. We work throughout in the following high-dimensional regime.

Assumption 3 (High-dimensional regime). Fix $\begin{array} { r } { 0 < \delta < \frac { 1 } { 2 } } \end{array}$ and $\kappa \geq 0$ . We work along a sequence of dimensions $d \to \infty$ for which

$$
r = r ( d ) = d ^ { \delta } , \qquad 1 \leq s = s ( d ) \leq d ^ { \kappa } ,
$$

with $r ( d )$ an integer divisor of d.

## 3.1 Weak directional recovery $\left( p \geq 3 \right)$

We first consider Assumption 1(i), namely polynomial links with information exponent $p \geq 3$ . For $\rho \in ( 0 , 1 )$ and $\mathsf { k } \in \{ \mathsf { T } , \mathsf { U } , \mathsf { A } \}$ , define the event

$$
\mathcal { R } _ { n } ^ { \mathrm { k } } ( \rho ) : = \left\{ \operatorname* { m a x } _ { 0 \leq t \leq n } \operatorname* { m a x } _ { 1 \leq j \leq s _ { \mathrm { k } } } M _ { \sigma } ( w _ { t } ^ { j , \mathrm { k } } ) \geq \rho \right\} ,
$$

where $s _ { \mathsf { T } } = 1 , s _ { \mathsf { U } } = s _ { \mathsf { A } } = s ,$ and $w _ { t } ^ { 1 , \top } = w _ { t } ^ { \top }$

Definition 3.1 (Weak recovery). Fix $\rho \in \left( 0 , 1 \right)$ , independently of d. We say that procedure $\mathsf { k } \in \{ \mathsf { T } , \mathsf { U } , \mathsf { A } \}$ achieves weak recovery at level $\rho$ within $n = n ( d )$ samples if

$$
\mathbb { P } ( \mathcal { R } _ { n } ^ { \mathrm { k } } ( \rho ) ) \to 1 \qquad \mathrm { a s ~ } d \to \infty .
$$

Thus, weak recovery requires the tied procedure’s single weight, or at least one weight of the untied or augmented procedure, to attain a fixed non-vanishing correlation with the identifiable teacher orbit. When $p$ is even, recovery is only up to cyclic shift and global sign, since both are exact symmetries of the model. When $p$ is odd, only the cyclic shift remains unidentifiable, and recovery requires positive correlation with the teacher orbit.

For $\mathsf { k } \in \{ \mathsf { T } , \mathsf { U } , \mathsf { A } \}$ , let $n _ { \mathsf { k } } ( \rho )$ denote the sample complexity of achieving weak recovery at level $\rho$ in the sense of Definition 3.1. Throughout, $\widetilde { \Theta }$ denotes matching upper and lower sample complexity bounds up to logarithmic factors in d. The upper bounds give recovery with probability $1 - o ( 1 )$ by the stated scale, whereas below the lower-bound scale, recovery does not occur with probability $1 - o ( 1 )$ . Our main results identify the sample complexity for the three procedures.

Theorem 3.2 (Sample complexity of weak recovery, $p \geq 3 )$ . Suppose that Assumptions $\begin{array} { r } { 1 ( i ) , \ 2 , } \end{array}$ and 3 hold. Fix $\rho \in ( 0 , 1 )$ ), independently of d. Then there exists $A _ { * } = A _ { * } ( p , P ) > 0$ such that, for every fixed $A > A _ { * }$ , the following holds. Set

$$
\eta _ { d } = d ^ { - p / 2 } ( \log d ) ^ { - A } ,
$$

and run the three online spherical SGD procedures (5) with step sizes

$$
\eta _ { d } ^ { \mathsf { T } } = \eta _ { d } , \qquad \eta _ { d } ^ { \mathsf { U } } = \sqrt { \frac { s } { r } } \eta _ { d } , \qquad \eta _ { d } ^ { \mathsf { A } } = \sqrt { r s } \eta _ { d } .\tag{6}
$$

Then the sample complexities for weak recovery at level $\rho$ satisfy

$$
n _ { \mathsf { T } } ( \rho ) , \ n _ { \mathsf { A } } ( \rho ) = \widetilde \Theta \left( \frac { d ^ { ( p - 2 ) / 2 } } { \eta _ { d } } \right) = \widetilde \Theta ( d ^ { p - 1 } ) , \qquad n _ { \mathsf { U } } ( \rho ) = \widetilde \Theta \left( \frac { r d ^ { ( p - 2 ) / 2 } } { \eta _ { d } } \right) = \widetilde \Theta ( r d ^ { p - 1 } ) .
$$

Thus, up to logarithmic factors, weight tying and exact data augmentation achieve weak recovery at the same $d ^ { p - 1 }$ scale as vanilla Gaussian single-index online SGD (Ben Arous et al., 2021; Damian et al., 2023), while the unaugmented untied procedure requires an additional factor r. The upper bounds in Theorem 3.2 are stronger than Definition 3.1 requires: any fixed weight of the untied or augmented procedure achieves weak recovery at the corresponding sample scale, so the proof does not rely on selecting the best among s initializations. Conversely, the lower bounds hold uniformly over all s weights. The proof combines the upper bounds of Theorem 5.1 with the lower bounds of Theorem $5 . 4 ;$ see Subsection 5.5.

Figure 1 illustrates the predicted separation empirically for the pure Hermite-4 link at $\delta = 1 / 2$ Since the upper bounds hold for any fixed weight, we set $s = 1$ and use the corresponding rescaled learning rates with a common initialization across procedures. This isolates the single-weight dynamics and controls for diferences in initial alignment that can strongly afect recovery times at small dimensions but do not change the asymptotic sample complexity scales established above.

Remark 3.3 (Choice of learning rates). The reference rate $\eta _ { d } = d ^ { - p / 2 } ( \log d )$ <sup>−A</sup> has the largest polynomial order supported by our perturbative analysis. Up to logarithmic factors, the accumulated stochastic and normalization errors, relative to the relevant growth scale, are controlled by $( \eta _ { d } d ^ { p / 2 } ) ^ { 1 / 2 }$ and $\eta _ { d } d ^ { p / 2 }$ , respectively. Choosing the constant A suficiently large makes both contributions negligible and allows comparison with the population dynamics.

Orbit coverage For the tied student, recovery of one orbit direction automatically represents the entire orbit through weight sharing. For the untied and augmented students, weak recovery guarantees only that one weight aligns with one orbit direction. We therefore ask the stronger question of whether the collection of s weights can recover every orbit direction. Define the relevant correlation

$$
q _ { k } ( w ) : = { \left\{ \begin{array} { l l } { | m _ { k } ( w ) | , } & { p { \mathrm { ~ e v e n } } , } \\ { m _ { k } ( w ) , } & { p { \mathrm { ~ o d d } } . } \end{array} \right. }\tag{7}
$$

For ${ \sf k } \in \{ { \sf U } , { \sf A } \}$ , define the full-orbit coverage event

$$
\mathcal { C } _ { n } ^ { \mathrm { k } } ( \rho ) : = \left\{ \operatorname* { m i n } _ { 0 \leq k < r } \operatorname* { m a x } _ { 1 \leq j \leq s _ { \mathrm { k } } } \operatorname* { m a x } _ { 0 \leq t \leq n } q _ { k } ( w _ { t } ^ { j , \mathrm { k } } ) \geq \rho \right\} .
$$

Definition 3.4 (Weak orbit coverage). We say that procedure $\mathsf { k } \in \{ \mathsf { U } , \mathsf { A } \}$ weakly covers the full teacher orbit at level $\rho$ within $n = n ( d )$ samples if

$$
\mathbb { P } ( \mathcal { C } _ { n } ^ { \boldsymbol { \mathsf { k } } } ( \rho ) )  \boldsymbol { 1 } .
$$

Write $n _ { \mathsf { k } } ^ { \mathrm { c o v } } ( \rho )$ for the corresponding sample complexity scale.

Equivalently, Definition 3.4 requires every identifiable teacher-orbit direction to be recovered by at least one weight at some time before n. With suficiently large width $s ,$ this stronger requirement can be achieved without changing the sample complexity scale.

Corollary 3.5 (Weak orbit coverage). Under the assumptions and learning rates of Theorem 3.2, $\mathit { f i x } \varepsilon > 0$ and suppose $s \geq ( 1 + \varepsilon ) \eta$ r log r. Then

$$
n _ { \mathsf { A } } ^ { \mathrm { c o v } } ( \rho ) = \widetilde { \Theta } ( d ^ { p - 1 } ) , \qquad n _ { \mathsf { U } } ^ { \mathrm { c o v } } ( \rho ) = \widetilde { \Theta } ( r d ^ { p - 1 } ) .
$$

Thus, once the width is suficient to cover the r orbit directions, requiring recovery of the entire orbit does not increase the sample complexity scale relative to weak recovery of a single direction. The proof is given in Section 8.4.

## 3.2 Weak subspace recovery $( p = 2 )$

We next consider Assumption 1(ii), namely the pure quadratic link. In this case the individual directions in the teacher orbit are generally not identifiable, and the natural recovery target is instead the subspace

$$
S ^ { * } : = \operatorname { s p a n } \{ \Pi ^ { k } w ^ { * } : 0 \leq k < r \} .
$$

For $w \in \mathbb { S } ^ { d - 1 }$ , let $w ^ { \parallel }$ denote its orthogonal projection onto ${ \boldsymbol { S } } ^ { * }$ . Since $\lVert \boldsymbol { w } \rVert = 1$ , the quantity $\| w ^ { \| } \| ^ { 2 }$ measures the fraction of the squared norm of w contained in the teacher subspace.

Definition 3.6 (Weak subspace recovery). Fix $\rho \in ( 0 , 1 )$ , independently of d. We say that procedure $\mathsf { k } \in \{ \mathsf { T } , \mathsf { U } , \mathsf { A } \}$ achieves weak subspace recovery at level $\rho$ within $n = n ( d )$ samples if

$$
\mathbb { P } \left( \operatorname* { m a x } _ { 0 \leq t \leq n } \operatorname* { m a x } _ { 1 \leq j \leq s _ { \mathrm { k } } } \left\| ( w _ { t } ^ { j , \mathrm { k } } ) ^ { \parallel } \right\| ^ { 2 } \geq \rho \right) \to 1 \qquad \mathrm { a s ~ } d \to \infty ,
$$

where $s _ { \mathsf { T } } = 1 , s _ { \mathsf { U } } = s _ { \mathsf { A } } = s ,$ and $w _ { t } ^ { 1 , \top } = w _ { t } ^ { \top }$

Let $n _ { \boldsymbol { \mathsf { k } } } ^ { ( 2 ) } ( \rho )$ denote the sample complexity of weak subspace recovery, with upper and lower bounds interpreted as for $n _ { \mathsf { k } } ( \rho )$

![](images/e937787e731aca79d3cbcf5db2d4274579c0073e6e1ebed9d12c1c4ce17a223d.jpg)  
Figure 1: Recovery-time scaling. Spherical online SGD with correlation loss, Hermite-4 link, and $r = { \sqrt { d } }$ symmetries. Points show median samples to reach $M _ { \sigma } ( w ) \geq 0 . 5$ over 10 paired seeds; bars are pointwise 95% bootstrap intervals. Colored dashed lines show power-law fits; gray dotted lines show reference powers. We report equivalent single-weight dynamics with $s _ { \mathsf { T } } = s _ { \mathsf { U } } = s _ { \mathsf { A } } = 1$ , shared initialization and samples, $\eta _ { d } = 0 . 1 d ^ { - 2 }$ , and the rescalings in (6). Tied and augmented trajectories coincide.

![](images/36feceff9a5472003a5213761331a551bce35b0a36f6bc6b88b27651e8c7b67e.jpg)  
Figure 2: Selection of an orbit direction in population flow. Population gradient flow with correlation loss for the tied student with the Hermite-4 link, dimension $d \ = \ 2 0 2 5$ , and cyclic symmetries $r ~ = ~ { \sqrt { d } } ~ = ~ 4 5$ Solid blue follows the correlation $\widetilde { m } _ { i _ { * } } ( w _ { t } ) { \ : = \ } \mathrm { s g n } ( m _ { i _ { * } } ( w _ { 0 } ) ) m _ { i _ { * } } ( w _ { t } )$ with index $i _ { * } = \arg \operatorname* { m a x } _ { k } | m _ { k } ( w _ { 0 } ) |$ . Dashed orange shows max $_ { \cdot k \neq i _ { * } } \left| m _ { k } ( w _ { t } ) \right|$ . The inset magnifies the early gap amplification. Time is normalized by $\tau _ { 0 . 5 }$ , the first time the population flow reaches orbit correlation 0.5.

Theorem 3.7 (Sample complexity of weak subspace recovery). Suppose Assumption $1 ( i i ) , \ 2 ,$ and 3 hold. Fix $\rho \in ( 0 , 1 )$ , independently of d. Then there exists a universal constant $A _ { 0 } > 0$ such that, for every fixed $A > A _ { 0 }$ , the following holds. Set

$$
\eta _ { d } = d ^ { - 1 } ( \log d ) ^ { - A } ,\tag{8}
$$

and use the same relative learning-rate scaling across the three procedures as in (6). Then the sample complexities for weak subspace recovery at level ρ satisfy

$$
n _ { \mathsf { T } } ^ { ( 2 ) } ( \rho ) , n _ { \mathsf { A } } ^ { ( 2 ) } ( \rho ) = \widetilde \Theta \biggl ( \frac { 1 } { \eta _ { d } } \biggr ) = \widetilde \Theta ( d ) , \qquad n _ { \mathsf { U } } ^ { ( 2 ) } ( \rho ) = \widetilde \Theta \biggl ( \frac { r } { \eta _ { d } } \biggr ) = \widetilde \Theta ( r d ) .
$$

Thus, the pure quadratic case exhibits the same r-fold separation between symmetry-aware and symmetry-agnostic learning as for $p \geq 3 \mathrm { : }$ weight tying and data augmentation achieve weak subspace recovery at scale $\widetilde { \Theta } ( d )$ , whereas the untied procedure requires $\widetilde { \Theta } ( r d )$ samples. The proof is provided in Subsection 5.5, combining the upper bounds of Theorem 5.3 with the lower bounds of Theorem 5.4.

Remark 3.8 (General links with information exponent 2). For general centered polynomial links with information exponent 2, our lower bound still applies, but our matching upper bound does not: we prove the upper bound only for the pure quadratic link. Higher-order Hermite components can change the appropriate recovery notion and the resulting dynamics. Related results for untied models have been obtained by Ren & Lee (2025).

## 4 Related work

Our work connects to several strands of a broad and rapidly developing literature, which we review below.

Learning low-dimensional representations Multi-index models provide a tractable setting for studying how learning algorithms discover low-dimensional structure in high-dimensional data. Chen & Meka (2020) developed algorithms for Gaussian polynomial multi-index models using filtered PCA and gradient-based refinement. Neural-network representation learning has been established under second-order nondegeneracy conditions (Damian et al., 2022), while Mousavi-Hosseini et al. (2023) showed that SGD with weight decay suppresses weight components orthogonal to the teacher subspace. These approaches investigate how information about the index subspace can be extracted through moment-based methods or through the evolution of network weights.

Staircase structure and leap complexity Hierarchical structure can allow previously learned features to facilitate the discovery of new ones. For Boolean inputs, Abbe et al. (2021) introduced the staircase property and established hierarchical learning guarantees under a layer-wise training procedure. For Boolean functions depending on a fixed number of coordinates, Abbe et al. (2022) identified the merged-staircase property as necessary and nearly suficient for learning with O(d) samples in the two-layer mean-field setting. Abbe et al. (2023) extended this perspective through leap complexity, which quantifies how many new coordinates must be learned together, counting Hermite multiplicities for Gaussian inputs. They proved learning guarantees for a structured class of Gaussian targets under a modified SGD procedure, showing how the dynamics move between successive saddles as additional coordinates are recovered.

Gradient dynamics and subspace discovery For Gaussian single-index models, Ben Arous et al. (2021) identified the information exponent governing the initial search phase of online SGD. Ba et al. (2022) quantified representation learning after one gradient step, highlighting the role of learning-rate scaling. For Gaussian multi-index models, Dandi et al. (2024a) showed that successive gradient steps with fresh batches of O(d) samples can reveal additional directions through a staircase mechanism. Bietti et al. (2025) established global convergence and characterized successive learning phases for a two-timescale population gradient flow, with timescales determined by the target’s Hermite decomposition. Related hierarchical subspace learning and weak recovery thresholds were analyzed through approximate message passing by Troiani et al. (2025). For orthogonal additive teachers, Şimşek et al. (2025) studied directional selection under correlation-loss gradient flow. Competition between signal directions also appears in multi-spiked tensor PCA, where Ben Arous et al. (2026) characterize a greedy maximum selection mechanism leading to the sequential recovery of individual spikes.

Dependence on the learning procedure Information exponent rates depend on the updates being analyzed. Loss smoothing (Damian et al., 2023), batch reuse (Dandi et al., 2024b), and alternating layer-wise updates with suitable learning-rate scales (Tsiolis et al., 2025) can improve upon the rates of the corresponding vanilla dynamics. Batch size and the loss also afect online learning times (Arnaboldi et al., 2024). Complementary algorithmic results introduce generative exponents and generative leap complexity for fixed-rank Gaussian models, accounting for information accessible through nonlinear label processing and sequential subspace estimation (Damian et al.,

2024; 2025). These works establish lower bounds within statistical-query or low-degree frameworks and develop algorithms that attain the corresponding scales under their assumptions.

Growing rank and subspace recovery When the number of teacher directions grows with dimension, the rank dependence of learning guarantees becomes essential. Oko et al. (2024) analyzed learning sums of nearly orthogonal ridge functions with growing rank, and Ren et al. (2025) related individual feature recovery times to prediction-risk scaling laws. In particular, the sample complexity of the untied learner in Theorem 3.2 has the same leading dependence on r and d as the upper bound of Oko et al. (2024), where they consider learning an untied teacher instead of a symmetry-tied one as in the present work. For quadratic targets, Ben Arous et al. (2025) studied growing-rank networks trained by orthogonality-constrained online SGD. Subspace and directional recovery can also occur at diferent stages: Ren & Lee (2025) showed how second-order information can reveal the index subspace before higher-order components distinguish individual teacher directions.

Weight sharing and data augmentation Group-equivariant architectures (Cohen & Welling, 2016; Kondor & Trivedi, 2018) and group-averaged training (Chen et al., 2020; Lyle et al., 2020) provide diferent ways to exploit symmetry. Statistical benefits have been quantified for equivariant, random-feature, and kernel models (Elesedy & Zaidi, 2021; Mei et al., 2021; Cagnetta et al., 2023). Beyond these settings, Lahoti et al. (2024) established sample complexity separations between convolutional, locally connected, and fully connected networks in a structured image model, distinguishing the benefits of locality and weight sharing.

Our work studies how a known symmetry changes online feature learning when the teacher rank grows with dimension. We compare weak directional and subspace recovery under weight sharing, full-group augmentation, and untied training without augmentation. The matching lower bounds are non-escape bounds for the specified spherical SGD updates and learning rates; they do not assert optimality over alternative learning algorithms.

## 5 Proof outline

We first explain the mechanisms underlying our recovery results, then state the precise upper and lower bounds and combine them to prove Theorems 3.2 and 3.7. Throughout, the three procedures use the learning-rate scaling in (6), expressed in terms of a reference rate $\eta _ { d }$ . Its value is specified in each theorem below.

## 5.1 Proof overview

The proofs of our main results are based on tracking the correlations $m _ { k } ( w _ { t } )$ defined in (2) between the SGD iterates and the teacher orbit. For $p \geq 3$ , we track the relevant correlations $q _ { k } ( w _ { t } )$ from (7), so that

$$
M _ { \sigma } ( w _ { t } ) = \operatorname* { m a x } _ { 0 \leq k < r } q _ { k } ( w _ { t } ) .
$$

Recall that $q _ { k } = m _ { k }$ when $p$ is odd and $q _ { k } = | m _ { k } |$ when $p$ is even. We show that random initialization produces, with high probability, a unique leading relevant correlation separated from its competitors by a gap large enough to dominate the error, and that SGD dynamics preserves and amplifies this gap.

Initialization, interactions, and correlation growth At random initialization, each correlation $m _ { k } ( w _ { 0 } )$ is typically of order $d ^ { - 1 / 2 }$ . Since the teacher orbit is nearly orthogonal, the correlation vector is approximately Gaussian with covariance close to $d ^ { - 1 } \mathrm { I } _ { r }$ . In particular, its coordinates behave approximately as independent $N ( 0 , d ^ { - 1 } )$ variables, and their largest relevant value $q _ { k } ( w _ { 0 } )$ is of order

$$
\gamma _ { d } : = \sqrt { \frac { \log r } { d } } .
$$

Moreover, with high probability, there is a unique leading relevant correlation, separated from all competing correlations by a quantifiable gap. This initial gap breaks the symmetry between the teacher directions and is then amplified by the SGD dynamics. Hermite orthogonality gives

$$
{ \Phi } ^ { \mathsf { T } } ( w ) = - \sum _ { k = 0 } ^ { r - 1 } \phi ( m _ { k } ( w ) ) , \qquad \phi ( z ) = \sum _ { \ell = p } ^ { P } a _ { \ell } ^ { 2 } z ^ { \ell } ,
$$

with $\phi ^ { \prime } ( m ) = p a _ { p } ^ { 2 } m ^ { p - 1 } + O ( | m | ^ { p } )$ near the origin. Writing $m _ { k } = m _ { k } ( w )$ , the spherical population drift of the kth correlation is

$$
- \langle \Pi ^ { k } w ^ { * } , \nabla \Phi ^ { \mathsf { T } } ( w ) \rangle = ( 1 - m _ { k } ^ { 2 } ) \phi ^ { \prime } ( m _ { k } ) + \sum _ { i \neq k } \langle \Pi ^ { k } w ^ { * } , \Pi ^ { i } w ^ { * } \rangle \phi ^ { \prime } ( m _ { i } ) - m _ { k } \sum _ { i \neq k } m _ { i } \phi ^ { \prime } ( m _ { i } ) .
$$

The first term gives the self-amplifying growth of $m _ { k }$ , while the remaining terms describe the interactions with the other teacher directions and the efect of the spherical constraint.

The main technical dificulty is to control these interaction terms uniformly when the number r of teacher directions grows with d. Together with the stochastic fluctuations and normalization errors, we show that their cumulative efect remains smaller than the initialization gap throughout the microscopic phase. As a result, the relevant correlation that is largest at initialization remains the leader: no competitor overtakes it, and its advantage grows over time. The dynamics can therefore be reduced to an efectively 1D growth mechanism along the selected teacher direction (see Fig. 2).

From microscopic selection to weak recovery Introduce the intermediate scales $\alpha _ { d } , \beta _ { d }$ satisfying

$$
\gamma _ { d } \ll \alpha _ { d } \ll \beta _ { d } \ll 1 .
$$

During the first, microscopic phase, all correlations grow but at diferent rates; thus, the initial gap between the leader and its competitors widens. As a result, the correlation that is largest at initialization reaches the threshold $\beta _ { d }$ , while all competing correlations remain at scale $O ( \alpha _ { d } )$ This phase takes $O ( ( \eta _ { d } \gamma _ { d } ^ { p - 2 } ) ^ { - 1 } )$ steps. During the second phase, the population drift drives the selected correlation from $\beta _ { d }$ to the weak recovery level $\rho \in ( 0 , 1 )$ , while the spherical dynamics keep the competing correlations small. This phase is faster and takes $O ( ( \eta _ { d } \beta _ { d } ^ { p - 2 } ) ^ { - 1 } )$ steps. Figure 3 illustrates the sketch of proof with this two-phase argument and Section 7 gives the proof for the tied procedure.

Untied and augmented procedures For the untied and augmented learners, it is enough for the upper bound to follow one fixed weight $w ^ { j }$ . Under the correlation loss, the population objective is additive across weights, so the gradient with respect to $w ^ { j }$ depends only on $w ^ { j } ;$

$$
\nabla _ { w ^ { j } } \Phi ^ { \mathsf { U } } ( W ) = \nabla _ { w ^ { j } } \Phi ^ { \mathsf { A } } ( W ) = \frac { 1 } { \sqrt { r s } } \nabla \Phi ^ { \mathsf { T } } ( w ^ { j } ) .
$$

Hence, the same correlation argument applies weight by weight. At the prescribed learning rates, an augmented weight has the same efective drift and fluctuation scales as the tied trajectory. For an untied weight, the efective drift is smaller by a factor r, while the stochastic fluctuations are smaller only by ${ \sqrt { r } } .$ . The same selection-and-growth mechanism therefore takes r times as many samples, producing the separation in Theorem 3.2. Section 8 makes these reductions precise.

Subspace recovery for the pure quadratic link For $\sigma = h _ { 2 }$ , we seek to grow $\| w _ { t } ^ { \parallel } \| ^ { 2 }$ to a fixed positive level. Because the teacher orbit is nearly orthonormal, the squared norm of the projection onto its span is, up to $\mathrm { ~ a ~ } 1 + o ( 1 )$ factor, the sum of the squared correlations:

$$
\| w _ { t } ^ { \parallel } \| ^ { 2 } = ( 1 + o ( 1 ) ) \sum _ { k = 0 } ^ { r - 1 } m _ { k } ( w _ { t } ) ^ { 2 } .
$$

We therefore track this sum, denoted by $R _ { t }$ , with $R _ { 0 } ~ \asymp ~ r / d$ Its population drift produces multiplicative growth. Controlling the accumulated stochastic and normalization errors shows that $\| w _ { t } ^ { \parallel } \| ^ { 2 }$ reaches the weak recovery level $\rho \in \mathsf { \Gamma } ( 0 , 1 )$ within $O ( \eta _ { d } ^ { - 1 } \log ( d / r ) )$ samples for the tied and augmented procedures, and r times as many for the untied procedure (see Section 9).

![](images/0bb1972978f1b066e9f24615d9b6aed287820b82a6c094d62501712607acb52f.jpg)  
Figure 3: Proof sketch: two phases of recovery. Blue follows the initial leader; orange shows the largest competitor, whose index may change. Their initial gap widens during both phases. The shaded band bounds competing absolute overlaps. Axes and trajectories are schematic; the macroscopic phase is expanded for visibility.

Matching lower bounds For every $p \geq 2$ , our drift and noise estimates imply that, with high probability, all correlations remain $O ( \gamma _ { d } )$ for at least $c / ( \eta _ { d } \gamma _ { d } ^ { p - 2 } )$ steps for the tied and augmented procedures, and for r times as many steps under the untied procedure. Here $c > 0$ is suficiently small. Since $r \gamma _ { d } ^ { 2 } = o ( 1 )$ , this excludes both directional and subspace recovery, uniformly over polynomially many weights (see Section 10).

## 5.2 Upper bounds for directional recovery

We retain the microscopic correlation scale $\gamma _ { d } = \sqrt { \log r / d }$ introduced in Subsection 5.1. Recall the orbit correlation $M _ { \sigma } ( w )$ defined in (3). For $\mathsf { k } \in \{ \mathsf { T } , \mathsf { U } , \mathsf { A } \} , a \in ( 0 , 1 )$ , and $1 \leq j \leq s _ { \mathsf { k } }$ , define the hitting time

$$
\tau _ { a } ^ { j , \mathsf { k } } : = \operatorname* { i n f } \left\{ t \geq 0 \colon M _ { \sigma } ( w _ { t } ^ { j , \mathsf { k } } ) \geq a \right\} ,\tag{9}
$$

with the convention inf $\varnothing = \infty$ . For the tied trajectory, we write $w _ { t } ^ { \mathsf { T } } = w _ { t } ^ { 1 , \mathsf { T } }$ and $\tau _ { a } ^ { \mathsf { T } } = \tau _ { a } ^ { 1 , \mathsf { T } }$

Theorem 5.1 (Weak recovery for $p \geq 3 )$ . Suppose that Assumptions $\begin{array} { r } { 1 ( i ) , \ 2 , } \end{array}$ and 3 hold. Fix $\rho \in ( 0 , 1 )$ . There exists $A _ { * } = A _ { * } ( p , P ) \geq ( p - 2 ) / 2$ such that the following holds for every $A > A _ { * }$ Set

$$
\eta _ { d } = d ^ { - p / 2 } ( \log d ) ^ { - A } ,
$$

and use the procedure-specific step sizes (6).

(T) If tied spherical SGD is run with $\eta _ { d } ^ { \mathsf { T } }$ , then there exists $C _ { T } = C _ { T } ( \sigma , \delta , \rho ) < \infty$ such that

$$
\mathbb { P } ( \tau _ { \rho } ^ { \mathsf { T } } \leq \frac { C _ { T } } { \eta _ { d } \gamma _ { d } ^ { p - 2 } } )  1 .
$$

(U) For any fixed $j ~ \in ~ \{ 1 , \dotsc , s \}$ , if untied spherical SGD is run with $\eta _ { d } ^ { \mathsf { U } }$ , then there exists $C _ { U } = C _ { U } ( \sigma , \delta , \rho ) < \infty$ such that

$$
\mathbb { P } ( \tau _ { \rho } ^ { j , \mathsf { U } } \leq \frac { C _ { U } r } { \eta _ { d } \gamma _ { d } ^ { p - 2 } } )  1 .
$$

(A) For any fixed $j \in \{ 1 , \ldots , s \}$ , if augmented spherical $S G D$ is run with $\eta _ { d } ^ { \mathsf { A } }$ , then there exists $C _ { A } = C _ { A } ( \sigma , \delta , \rho ) < \infty$ such that

$$
\mathbb { P } ( \tau _ { \rho } ^ { j , \mathsf { A } } \leq \frac { C _ { A } } { \eta _ { d } \gamma _ { d } ^ { p - 2 } } )  1 .
$$

Thus the tied and augmented procedures achieve weak recovery within

$$
\frac { 1 } { \eta _ { d } \gamma _ { d } ^ { p - 2 } } = O \left( d ^ { p - 1 } \operatorname { p o l y l o g } ( d ) \right)
$$

samples, whereas the untied procedure requires at most $O \left( r d ^ { p - 1 } \operatorname { p o l y l o g } ( d ) \right)$ samples. Theorem 5.1 gives the upper bounds stated in Theorem 3.2. The matching lower bounds follow from Theorem 5.4 below.

Remark 5.2 (Fixed-weight upper bounds). For the untied and augmented procedures, the upper bound holds for any fixed weight index $j .$ . In particular, the proof does not exploit the best among the s random initializations. The width s enters only through the learning-rate rescaling in (6).

## 5.3 Upper bounds for quadratic subspace recovery

Recall the teacher orbit subspace ${ \boldsymbol { S } } ^ { * }$ and the orthogonal projection $w ^ { \parallel }$ used in Definition 3.6. For $a \in ( 0 , 1 )$ and $\mathsf { k } \in \{ \mathsf { T } , \mathsf { U } , \mathsf { A } \}$ , define

$$
\tau _ { a } ^ { ( 2 ) , \mathrm { k } } : = \operatorname* { i n f } \left\{ t \geq 0 \colon \operatorname* { m a x } _ { 1 \leq j \leq s _ { \mathrm { k } } } \| ( w _ { t } ^ { j , \mathrm { k } } ) ^ { \| } \| ^ { 2 } \geq a \right\} ,\tag{10}
$$

with inf $\varnothing = \infty$ . For the tied trajectory, we write $\tau _ { a } ^ { ( 2 ) } = \tau _ { a } ^ { ( 2 ) , \top }$

Theorem 5.3 (Weak subspace recovery for the quadratic link). Suppose that Assumptions $1 ( i i ) , \ 2 ,$ and 3 hold. Fix $\rho \in ( 0 , 1 )$ independently of d. There exists a universal constant $A _ { 0 } > 0$ such that the following holds for every $A > A _ { 0 }$ . Set

$$
\eta _ { d } = d ^ { - 1 } ( \log d ) ^ { - A } ,
$$

and use the procedure-specific step sizes (6).

(T) If tied spherical SGD is run with $\eta _ { d } ^ { \mathsf { T } }$ , then there exists $C _ { T } = C _ { T } ( \delta , \rho ) < \infty$ such that

$$
\mathbb { P } ( \tau _ { \rho } ^ { ( 2 ) , \mathsf { T } } \leq \frac { C _ { T } \log ( d / r ) } { \eta _ { d } } )  1 .
$$

(U) If untied spherical SGD is run with $\eta _ { d } ^ { \mathsf { U } }$ , then there exists $C _ { U } = C _ { U } ( \delta , \rho ) < \infty$ such that

$$
\mathbb { P } ( \tau _ { \rho } ^ { ( 2 ) , \mathsf { U } } \leq \frac { C _ { U } r \log ( d / r ) } { \eta _ { d } } )  1 .
$$

(A) If augmented spherical SGD is run with $\eta _ { d } ^ { \mathsf { A } }$ , then there exists $C _ { A } = C _ { A } ( \delta , \rho ) < \infty$ such that

$$
\mathbb { P } ( \tau _ { \rho } ^ { ( 2 ) , \mathsf { A } } \leq \frac { C _ { A } \log ( d / r ) } { \eta _ { d } } )  1 .
$$

Since lo $\smash { \{ d / r \} = ( 1 - \delta ) }$ log d, these bounds are $\widetilde O ( d )$ for the tied and augmented procedures, and $\widetilde { O } ( r d )$ for the untied.

## 5.4 Non-escape and lower bounds

We next give a non-escape estimate that yields the matching lower bounds for both recovery notions above. The result applies to any fixed centered polynomial link with information exponent $p \geq 2$ For k $\in \{ \mathsf { T } , \mathsf { U } , \mathsf { A } \}$ , define

$$
\tau _ { \mathrm { e s c } } ^ { \mathbf { k } } ( C ) : = \operatorname* { i n f } \left\{ t \geq 0 \colon \operatorname* { m a x } _ { 1 \leq j \leq s _ { \mathrm { k } } 0 \leq \ell < r } | m _ { \ell } ( w _ { t } ^ { j , \mathbf { k } } ) | \geq C \gamma _ { d } \right\} ,\tag{11}
$$

with inf $\varnothing = \infty$ . Thus $\tau _ { \mathrm { e s c } } ^ { \mathbf { k } }$ is the first time that any weight of procedure k develops a correlation larger than a constant multiple of the initialization scale $\gamma _ { d } .$

Theorem 5.4 (Non-escape for the three learning procedures). Suppose that Assumptions 2 and 3 hold, and let

$$
\sigma ( z ) = \sum _ { j = p } ^ { P } a _ { j } h _ { j } ( z ) , \qquad 2 \leq p \leq P < \infty , \qquad a _ { p } \neq 0 ,
$$

where P and the coeficients are fixed independently of d. Fix $A > 0$ , set

$$
\eta _ { d } = d ^ { - p / 2 } ( \log d ) ^ { - A } ,
$$

and use the procedure-specific step sizes (6). Then there exist constants $c , C _ { \mathrm { e s c } } > 0$ , depending only on $\sigma , \delta , \kappa ,$ such that, with

$$
T _ { d } : = \frac { c } { \eta _ { d } \gamma _ { d } ^ { p - 2 } } ,
$$

we have

$$
\mathbb { P } ( \tau _ { \mathrm { e s c } } ^ { \mathsf { T } } ( C _ { \mathrm { e s c } } ) > T _ { d } , \tau _ { \mathrm { e s c } } ^ { \mathsf { A } } ( C _ { \mathrm { e s c } } ) > T _ { d } , \tau _ { \mathrm { e s c } } ^ { \mathsf { U } } ( C _ { \mathrm { e s c } } ) > r T _ { d } )  1 .
$$

On the event in Theorem 5.4, every weight remains at microscopic correlation with every teacher direction throughout the corresponding time interval. Hence, for $p \geq 3$ , weak recovery is impossible on these intervals. For the quadratic link, the same event also rules out subspace weak recovery. Indeed, on the teacher-geometry event of Lemma A.1,

$$
\| ( w _ { t } ^ { j , \mathsf { k } } ) ^ { \| } \| ^ { 2 } \leq \frac { \sum _ { \ell = 0 } ^ { r - 1 } m _ { \ell } ( w _ { t } ^ { j , \mathsf { k } } ) ^ { 2 } } { 1 - r \mu _ { * } } \leq \frac { C _ { \mathrm { e s c } } ^ { 2 } r \gamma _ { d } ^ { 2 } } { 1 - r \mu _ { * } } = o ( 1 ) ,
$$

uniformly over all weights and times in the corresponding intervals. Since each iteration uses one fresh observation, Theorem 5.4 yields sample complexity lower bounds $\widetilde \Omega ( d ^ { p - 1 } )$ for the tied and augmented procedures and $\widetilde \Omega ( r d ^ { p - 1 } )$ for the untied procedure, for every polynomial link covered by Theorem 5.4.

## 5.5 Proofs of the main results

We combine the upper bounds with the non-escape estimate to prove Theorems 3.2 and 3.7.

Proof of Theorem 3.2. Fix $A > A _ { * }$ , where $A _ { * }$ is given by Theorem 5.1. For the upper bounds, apply Theorem 5.1 with $j = 1$ . Since recovery of one weight is suficient for Definition 3.1, weak recovery occurs with probability $1 - o ( 1 )$ within $C / ( \eta _ { d } \gamma _ { d } ^ { p - 2 } )$ samples for the tied and augmented procedures, and within $C r / ( \eta _ { d } \gamma _ { d } ^ { p - 2 } )$ samples for the untied procedure.

For the lower bounds, Theorem 5.4 applies under the same assumptions and learning rates. Since $M _ { \sigma } ( w ) \leq \operatorname* { m a x } _ { 0 \leq \ell < r } | m _ { \ell } ( w ) |$ and $C _ { \mathrm { e s c } } \gamma _ { d } < \rho$ for all suficiently large d, its non-escape event excludes weak recovery by every weight up to times $c / ( \eta _ { d } \gamma _ { d } ^ { p - 2 } )$ for the tied and augmented procedures, and $c r / ( \eta _ { d } \gamma _ { d } ^ { p - 2 } )$ for the untied procedure.

Finally,

$$
\frac { 1 } { \eta _ { d } \gamma _ { d } ^ { p - 2 } } = \frac { d ^ { ( p - 2 ) / 2 } } { \eta _ { d } ( \log r ) ^ { ( p - 2 ) / 2 } } = \widetilde \Theta \biggl ( \frac { d ^ { ( p - 2 ) / 2 } } { \eta _ { d } } \biggr ) = \widetilde \Theta ( d ^ { p - 1 } ) ,
$$

where we used that $r = d ^ { \delta }$ and $\eta _ { d } = d ^ { - p / 2 } ( \log d ) ^ { - A }$ . This proves the claim.

Proof of Theorem 3.7. Fix $A > A _ { 0 }$ , where $A _ { 0 }$ is given by Theorem 5.3. By (10), the event in Definition 3.6 is exactly $\{ \tau _ { \rho } ^ { ( 2 ) , \sf k } \leq n \}$ . Theorem 5.3 therefore gives upper bounds of $C \log ( d / r ) / \eta _ { d }$ samples for the tied and augmented procedures and of $C r \log ( d / r ) / \eta _ { d }$ samples for the untied procedure.

For the lower bounds, apply Theorem 5.4 with $\sigma = h _ { 2 }$ and $p = 2$ . Then $T _ { d } = c / \eta _ { d }$ , and the subspace estimate above shows that, with probability $1 - o ( 1 )$ , every weight satisfies $\| ( w _ { t } ^ { j , \mathsf { k } } ) \| _ { \mathsf { l } } \| ^ { 2 } = o ( 1 )$ throughout the corresponding time interval. Since $\rho > 0$ is fixed, weak subspace recovery is therefore impossible before $c / \eta _ { d }$ for the tied and augmented procedures, and before $c r / \eta _ { d }$ for the untied procedure.

The upper and lower bounds difer only by the logarithmic factor $\log ( d / r ) = ( 1 - \delta )$ log d. Since $\eta _ { d } ^ { - 1 } = d ( \log d ) ^ { A }$ , we obtain

$$
n _ { \mathsf { T } } ^ { ( 2 ) } ( \rho ) , n _ { \mathsf { A } } ^ { ( 2 ) } ( \rho ) = \widetilde \Theta ( \eta _ { d } ^ { - 1 } ) = \widetilde \Theta ( d ) , \qquad n _ { \mathsf { U } } ^ { ( 2 ) } ( \rho ) = \widetilde \Theta ( r \eta _ { d } ^ { - 1 } ) = \widetilde \Theta ( r d ) .
$$

## 6 Correlation dynamics and auxiliary estimates

The purpose of this section is to collect the dynamical identities and stochastic estimates used in the weak recovery proofs. We derive them for the tied student, for which the correlation dynamics take their cleanest form. Section 7 uses these estimates to prove weak recovery for the tied student, while Section 8 shows how the same structure changes under augmentation and untied training.

Throughout this section, we suppress the superscript T and write

$$
w _ { t } : = w _ { t } ^ { \mathsf { T } } , \qquad L : = L ^ { \mathsf { T } } , \qquad \Phi : = \Phi ^ { \mathsf { T } } , \qquad \eta _ { d } : = \eta _ { d } ^ { \mathsf { T } } .
$$

We also work on the teacher-geometry event $\mathcal { G } _ { d }$ introduced in Lemma A.1. All estimates below that condition on the teacher are uniform over $w ^ { \ast } \in \mathcal { G } _ { d }$

## 6.1 Correlation recursion and population drift

We begin by deriving the exact evolution of the correlations

$$
m _ { k } ( w _ { t } ) = \langle \Pi ^ { k } w ^ { \ast } , w _ { t } \rangle , \qquad 0 \leq k < r ,
$$

under spherical SGD. We first identify the population contribution. For $k \in \{ 0 , \ldots , r - 1 \}$ , define the teacher correlations

$$
c _ { k } : = \langle w ^ { * } , \Pi ^ { k } w ^ { * } \rangle ,
$$

where indices are understood modulo r. By cyclicity, $\langle \Pi ^ { k } w ^ { * } , \Pi ^ { j } w ^ { * } \rangle = c _ { j - k }$

Lemma 6.1 (Population drift). For every $w \in \mathbb { S } ^ { d - 1 }$ and $k \in \{ 0 , \ldots , r - 1 \}$ ,

$$
- \langle \Pi ^ { k } w ^ { * } , \nabla \Phi ( w ) \rangle = \phi ^ { \prime } ( m _ { k } ( w ) ) + \Gamma _ { k } ( w ) ,\tag{12}
$$

where

$$
\Gamma _ { k } ( w ) : = \sum _ { j \neq k } c _ { j - k } \phi ^ { \prime } ( m _ { j } ( w ) ) - m _ { k } ( w ) \sum _ { j = 0 } ^ { r - 1 } m _ { j } ( w ) \phi ^ { \prime } ( m _ { j } ( w ) ) .\tag{13}
$$

Proof. On the unit sphere, Hermite orthogonality gives

$$
\Phi ( w ) = - \sum _ { j = 0 } ^ { r - 1 } \phi ( m _ { j } ( w ) ) .
$$

The spherical gradient of the population objective is

$$
\nabla \Phi ( w ) = (  { \mathrm { I } _ { d } } - w w ^ { \top } ) \widehat { \nabla } \Phi ( w ) = - \sum _ { j = 0 } ^ { r - 1 } \phi ^ { \prime } ( m _ { j } ( w ) ) \big ( \Pi ^ { j } w ^ { * } - m _ { j } ( w ) w \big ) .
$$

Taking the scalar product with $\Pi ^ { k } w ^ { \ast } \mathrm { ~ y ~ }$ ields

$$
- \langle \Pi ^ { k } w ^ { * } , \nabla \Phi ( w ) \rangle = \sum _ { j = 0 } ^ { r - 1 } \phi ^ { \prime } ( m _ { j } ( w ) ) \left( \langle \Pi ^ { k } w ^ { * } , \Pi ^ { j } w ^ { * } \rangle - m _ { k } ( w ) m _ { j } ( w ) \right) .
$$

Separating the term $j = k$ , using $\langle \Pi ^ { k } w ^ { * } , \Pi ^ { k } w ^ { * } \rangle = 1$ , and recalling $\langle \Pi ^ { k } w ^ { \ast } , \Pi ^ { j } w ^ { \ast } \rangle = c _ { j } .$ <sub>−k</sub>, gives (12).

We now combine this population identity with the sample-level SGD update. Define

$$
G _ { t } : = \nabla L ( w _ { t - 1 } ; x _ { t } , y _ { t } ) ,
$$

and introduce the sample-wise centered loss

$$
H _ { t } ( w ) : = L ( w ; x _ { t } , y _ { t } ) - \Phi ( w ) .
$$

Then $G _ { t } = \nabla \Phi ( w _ { t - 1 } ) + \nabla H _ { t } ( w _ { t - 1 } )$ . We also define the coordinate-wise stochastic fluctuation

$$
\begin{array} { r } { \xi _ { k , t } : = - \left. \Pi ^ { k } w ^ { * } , \nabla H _ { t } ( w _ { t - 1 } ) \right. . } \end{array}\tag{14}
$$

Lemma 6.2 (Exact correlation evolution). For every $0 \leq k < r$ and $t \geq 1$ ,

$$
m _ { k } ( w _ { t } ) = m _ { k } ( w _ { 0 } ) + p a _ { p } ^ { 2 } \eta _ { d } \sum _ { \ell = 1 } ^ { t } m _ { k } ( w _ { \ell - 1 } ) ^ { p - 1 } + E _ { k , t } ,\tag{15}
$$

where

$$
\begin{array} { l } { { \displaystyle E _ { k , t } : = \eta _ { d } \sum _ { \ell = 1 } ^ { t } R _ { \sigma } ( m _ { k } ( w _ { \ell - 1 } ) ) + v \eta _ { d } \sum _ { \ell = 1 } ^ { t } \Gamma _ { k } ( w _ { \ell - 1 } ) + \eta _ { d } \sum _ { \ell = 1 } ^ { t } \xi _ { k , \ell } } } \\ { { \displaystyle ~ - \sum _ { \ell = 1 } ^ { t } \left( \sqrt { 1 + \eta _ { d } ^ { 2 } \| G _ { \ell } \| ^ { 2 } } - 1 \right) m _ { k } ( w _ { \ell } ) } , } \end{array}\tag{16}
$$

with

$$
R _ { \sigma } ( m ) : = \sum _ { k = p + 1 } ^ { P } k a _ { k } ^ { 2 } m ^ { k - 1 } .
$$

Proof. By the spherical SGD update (5),

$$
w _ { t } = \frac { w _ { t - 1 } - \eta _ { d } G _ { t } } { \Vert w _ { t - 1 } - \eta _ { d } G _ { t } \Vert } .
$$

Since $G _ { t }$ belongs to the tangent space of $\mathbb { S } ^ { d - 1 }$ at $w _ { t - 1 }$ , we have $\langle G _ { t } , w _ { t - 1 } \rangle = 0$ , and therefore

$$
\lVert \boldsymbol { w } _ { t - 1 } - \eta _ { d } \boldsymbol { G } _ { t } \rVert ^ { 2 } = 1 + \eta _ { d } ^ { 2 } \lVert \boldsymbol { G } _ { t } \rVert ^ { 2 } .
$$

Taking the scalar product of $w _ { t }$ with $\Pi ^ { k } w ^ { * }$ gives

$$
m _ { k } ( w _ { t } ) = \langle \Pi ^ { k } w ^ { \ast } , w _ { t } \rangle = \frac { m _ { k } ( w _ { t - 1 } ) - \eta _ { d } \langle \Pi ^ { k } w ^ { \ast } , G _ { t } \rangle } { \| w _ { t - 1 } - \eta _ { d } G _ { t } \| } ,
$$

or equivalently,

$$
\sqrt { 1 + \eta _ { d } ^ { 2 } \| G _ { t } \| ^ { 2 } } m _ { k } ( w _ { t } ) = m _ { k } ( w _ { t - 1 } ) - \eta _ { d } \langle \Pi ^ { k } w ^ { * } , G _ { t } \rangle .
$$

Using $G _ { t } = \nabla \Phi ( w _ { t - 1 } ) + \nabla H _ { t } ( w _ { t - 1 } )$ , Lemma 6.1, and the definition of $\xi _ { k , t }$ , we obtain

$$
\sqrt { 1 + \eta _ { d } ^ { 2 } \| G _ { t } \| ^ { 2 } } m _ { k } ( w _ { t } ) = m _ { k } ( w _ { t - 1 } ) + \eta _ { d } \phi ^ { \prime } ( m _ { k } ( w _ { t - 1 } ) ) + \eta _ { d } \Gamma _ { k } ( w _ { t - 1 } ) + \eta _ { d } \xi _ { k , t } .
$$

Therefore

$$
\begin{array} { r l } & { m _ { k } ( w _ { t } ) = m _ { k } ( w _ { t - 1 } ) + \eta _ { d } \phi ^ { \prime } ( m _ { k } ( w _ { t - 1 } ) ) + \eta _ { d } \Gamma _ { k } ( w _ { t - 1 } ) + \eta _ { d } \xi _ { k , t } } \\ & { \qquad - \left( \sqrt { 1 + \eta _ { d } ^ { 2 } \| G _ { t } \| ^ { 2 } } - 1 \right) m _ { k } ( w _ { t } ) . } \end{array}\tag{17}
$$

Finally,

$$
\phi ^ { \prime } ( m ) = \sum _ { k = p } ^ { P } k a _ { k } ^ { 2 } m ^ { k - 1 } = p a _ { p } ^ { 2 } m ^ { p - 1 } + R _ { \sigma } ( m ) .
$$

Summing (17) over $\ell = 1 , \ldots , t$ and separating the leading Hermite contribution in (17) gives (15).

Since $\begin{array} { r } { \phi ( z ) = \sum _ { \ell = p } ^ { P } a _ { \ell } ^ { 2 } z ^ { \ell } } \end{array}$ , termwise estimates give, for a constant $C _ { \sigma } < \infty$

$$
| \phi ^ { \prime } ( z ) | \leq C _ { \sigma } | z | ^ { p - 1 } , \qquad | R _ { \sigma } ( z ) | \leq C _ { \sigma } | z | ^ { p } , \qquad | z | \leq 1 .\tag{18}
$$

Moreover, the coeficients of $\phi$ are nonnegative, so

$$
\phi ^ { \prime } ( z ) \geq p a _ { p } ^ { 2 } z ^ { p - 1 } , \qquad 0 \leq z \leq 1 .\tag{19}
$$

The four terms in $E _ { k , t }$ have distinct origins: they are, respectively, the higher-Hermite remainder, the interaction with the other teacher directions, the centered stochastic fluctuation, and the spherical normalization error. The deterministic remainder and interaction terms will be controlled directly from the correlation bounds and the geometry of the teacher orbit. The remaining two contributions require probabilistic estimates on $\xi _ { k , t }$ and $\| G _ { t } \|$ , which we establish in the next two subsections.

## 6.2 Coordinate noise estimates

Define the filtration

$$
\mathcal { F } _ { t } : = \sigma ( w ^ { \ast } , w _ { 0 } , ( x _ { \ell } , y _ { \ell } ) _ { 1 \le \ell \le t } ) .
$$

Since $( x _ { t } , y _ { t } )$ is fresh conditionally on $\mathcal { F } _ { t - 1 }$ 2

$$
\mathbb { E } [ G _ { t } \mid { \mathcal { F } } _ { t - 1 } ] = \nabla \Phi ( w _ { t - 1 } ) .
$$

Therefore,

$$
\xi _ { k , t } = - \left. \Pi ^ { k } w ^ { \ast } , G _ { t } - \mathbb { E } [ G _ { t } \mid \mathcal { F } _ { t - 1 } ] \right.
$$

is a martingale diference with respect to $( \mathcal { F } _ { t } ) _ { t \geq 0 }$

We first derive an identity for the second moments of the feature gradient. Besides being used below to control $\xi _ { k , t }$ , this identity will also be useful in the proof of the uniform gradient bound.

Lemma 6.3 (Feature-gradient covariance). Let $w \in \mathbb { S } ^ { d - 1 }$ and $v \perp w ,$ , and define $a _ { j } ( w ) : = \langle w , \Pi ^ { j } w \rangle$ Then

$$
\mathbb { E } \left[ \langle v , \widehat { \nabla } f _ { w } ( x ) \rangle ^ { 2 } \right] = \sum _ { j = 0 } ^ { r - 1 } \left( \phi ^ { \prime } ( a _ { j } ( w ) ) \langle v , \Pi ^ { j } v \rangle + \phi ^ { \prime \prime } ( a _ { j } ( w ) ) \langle w , \Pi ^ { j } v \rangle \langle \Pi ^ { j } w , v \rangle \right) .
$$

The proof of Lemma 6.3 is given in Appendix B.1.

We now control the conditional moments of the coordinate noise.

Lemma 6.4 (Coordinate noise moments). On the teacher-geometry event $\mathcal { G } _ { d } .$ , there exists $C _ { \sigma } < \infty$ such that, for all suficiently large d, all $0 \leq k < r , t \geq 1$ , and $\nu \geq 2$ ，

$$
\mathbb { E } [ \xi _ { k , t } ^ { 2 } \mid \mathcal { F } _ { t - 1 } ] \le C _ { \sigma } \left( 1 + r m _ { k } ( w _ { t - 1 } ) ^ { 2 } \right) ,\tag{20}
$$

$$
( \mathbb { E } [ | \xi _ { k , t } | ^ { \nu } \mid \mathcal { F } _ { t - 1 } ] ) ^ { 1 / \nu } \leq C _ { \sigma } \nu ^ { P } \left( 1 + \sqrt { r } | m _ { k } ( w _ { t - 1 } ) | \right) .\tag{21}
$$

The bounds hold uniformly over $w _ { t - 1 } \in \mathbb { S } ^ { d - 1 }$ and over $w ^ { \ast } \in \mathcal G _ { d }$

The proof of Lemma 6.4 is given in Appendix B.2.

## 6.3 Uniform control of the stochastic gradient

We finally establish a high-probability bound on the norm of the stochastic gradient, uniformly over polynomially many SGD iterations.

Lemma 6.5. For every fixed $C > 0$ there exists $C _ { \sigma , C } < \infty$ such that, for every deterministic horizon $T \leq d ^ { C }$

$$
\mathbb { P } \left( \operatorname* { m a x } _ { 1 \leq t \leq T } \| G _ { t } \| > C _ { \sigma , C } ( \log d ) ^ { P } \sqrt { d } \right) = o ( 1 ) .
$$

The proof of Lemma 6.5 is given in Appendix B.3.

## 7 Proof of weak recovery for the tied student

In this section we prove part (T) of Theorem 5.1. The proof shows that the leading polynomial dependence of the recovery time is determined by the information exponent $p \mathrm { : }$ higher Hermite components afect the constants and logarithmic factors, but not the leading $d ^ { p - 1 }$ sample complexity exponent.

## 7.1 Proof strategy

The proof proceeds in two main steps. The first step selects and amplifies one of the r teacher correlations from its random initialization scale until the trajectory enters a suitable intermediate region. The second step starts from any point in this region and propagates the selected correlation to the fixed recovery level $\rho .$

Fix $\begin{array} { r } { B > \frac { 2 } { p - 2 } } \end{array}$ , depending only on $p ,$ and define the two intermediate scales

$$
\alpha _ { d } : = \gamma _ { d } ( \log d ) ^ { 2 / ( p - 2 ) } , \qquad \beta _ { d } : = \gamma _ { d } ( \log d ) ^ { B } .
$$

Then

$$
\gamma _ { d } \ll \alpha _ { d } \ll \beta _ { d } \ll 1 .
$$

We next introduce notation for the correlation selected at initialization. Define

$$
i _ { * } : = \operatorname* { m i n } \underset { 0 \leq k < r } { \arg \operatorname* { m a x } } q _ { k } ( w _ { 0 } ) ,
$$

where $q _ { k } ( w )$ denotes the relevant correlation defined in (7). Define the corresponding oriented correlation by

$$
\widetilde { \boldsymbol { m } } _ { i _ { * } } ( \boldsymbol { w } _ { t } ) : = \left\{ \begin{array} { l l } { \mathrm { s g n } ( \boldsymbol { m } _ { i _ { * } } ( \boldsymbol { w } _ { 0 } ) ) \boldsymbol { m } _ { i _ { * } } ( \boldsymbol { w } _ { t } ) , } & { p \mathrm { ~ e v e n } , } \\ { \boldsymbol { m } _ { i _ { * } } ( \boldsymbol { w } _ { t } ) , } & { p \mathrm { ~ o d d } , } \end{array} \right.\tag{22}
$$

with the convention $\operatorname { s g n } ( 0 ) = 1$ . Lemma A.3 shows that, with probability tending to one, the maximizer is unique, $q _ { i _ { * } } ( w _ { 0 } ) \asymp \gamma _ { d }$ , and is separated from all competing relevant correlations by a quantitative relative gap.

Let

$$
\mathcal { F } _ { t } : = \sigma \left( w ^ { * } , w _ { 0 } , ( x _ { \ell } , y _ { \ell } ) _ { 1 \le \ell \le t } \right)
$$

denote the natural filtration. Then $i _ { * }$ is ${ \mathcal { F } } _ { 0 } .$ -measurable, $w _ { t }$ is $\mathcal { F } _ { t } .$ -measurable, and each hitting time $\tau _ { a }$ is an $( \mathcal { F } _ { t } ) _ { t \geq 0 ^ { - \mathrm { { s t o p p i n g } } } }$ time.

The first phase shows that the small advantage present at initialization survives the stochastic fluctuations and interaction errors and is amplified until the selected correlation reaches the scale $\beta _ { d }$ , while all competing correlations remain at the smaller scale $\alpha _ { d }$

Proposition 7.1 (Microscopic selection). Suppose that Assumptions $\begin{array} { r } { 1 ( i ) , \ 2 , } \end{array}$ and 3 hold. There exists $A _ { * } = A _ { * } ( p , P ) \geq ( p - 2 ) / 2$ such that the following holds for every fixed $A > A _ { * }$ . Run tied spherical SGD (5) with step size

$$
\eta _ { d } = d ^ { - p / 2 } ( \log d ) ^ { - A } .
$$

Then there exist constants $C _ { \alpha } , C _ { \beta } < \infty$ , depending only on the fixed model parameters, such that, with probability $1 - o ( 1 )$ 2

$$
\tau _ { \beta _ { d } } \leq \frac { C _ { \beta } } { \eta _ { d } \gamma _ { d } ^ { p - 2 } } ,
$$

and

$$
\beta _ { d } \leq \tilde { m } _ { i _ { * } } ( w _ { \tau _ { \beta _ { d } } } ) \leq 2 \beta _ { d } , \qquad \operatorname* { m a x } _ { k \neq i _ { * } } | m _ { k } ( w _ { \tau _ { \beta _ { d } } } ) | \leq C _ { \alpha } \alpha _ { d } .
$$

Moreover, $i _ { * }$ remains the unique relevant leader up to time $\tau _ { \beta _ { d } }$

The proof of Proposition 7.1 is given in Section 7.2.

Since $\tau _ { \beta _ { d } }$ is a stopping time and the SGD recursion is driven by i.i.d. observations, the strong Markov property implies that, conditionally on ${ \mathcal { F } } _ { { \tau } _ { \beta _ { d } } }$ , the post- $\cdot \tau _ { \beta _ { d } }$ trajectory has the law of tied spherical SGD started from $w _ { \tau _ { \beta _ { d } } }$ and driven by fresh observations, with $w ^ { * }$ fixed. This allows us to restart the analysis at $\tau _ { \beta _ { d } }$

Proposition 7.2 (Growth to macroscopic size). Suppose that Assumptions $\begin{array} { r } { { 1 } ( i ) , \ 2 , } \end{array}$ and 3 hold. Fix $\rho \in ( 0 , 1 )$ , independently of d. There exists $A _ { * } = A _ { * } ( p , P ) \geq ( p - 2 ) / 2$ such that the following holds for every fixed $A > A _ { * }$ . Run tied spherical SGD (5) with step size

$$
\eta _ { d } = d ^ { - p / 2 } ( \log d ) ^ { - A } .
$$

Then there exist $C _ { \rho } < \infty$ and a deterministic sequence $\varepsilon _ { d } \to 0$ such that, almost surely on

$$
\mathcal { G } _ { d } \cap \left\{ \beta _ { d } \leq \widetilde { m } _ { i _ { * } } ( w _ { \tau _ { \beta _ { d } } } ) \leq 2 \beta _ { d } , \quad \underset { k \neq i _ { * } } { \operatorname* { m a x } } | m _ { k } ( w _ { \tau _ { \beta _ { d } } } ) | \leq C _ { \alpha } \alpha _ { d } \right\} ,
$$

it holds that

$$
\mathbb { P } \left( \tau _ { \rho } - \tau _ { \beta _ { d } } \leq \frac { C _ { \rho } } { \eta _ { d } \beta _ { d } ^ { p - 2 } } \middle | \mathcal { F } _ { \tau _ { \beta _ { d } } } \right) \geq 1 - \varepsilon _ { d } ,
$$

where $\mathcal { G } _ { d }$ is the teacher-geometry event of Lemma A.1. Moreover, with the same conditional probability lower bound, $i _ { * }$ remains the unique favorable leader up to time $\tau _ { \rho } ,$ while all competing overlaps remain microscopic.

The proof of Proposition 7.2 is given in Section 7.3. We now provide the proof of part (T) of Theorem 5.1.

Proof of part (T) of Theorem 5.1. Let

$$
\mathcal { E } _ { d } : = \mathcal { G } _ { d } \cap \left\{ \tau _ { \beta _ { d } } \leq \frac { C _ { \beta } } { \eta _ { d } \gamma _ { d } ^ { p - 2 } } , \quad \beta _ { d } \leq \widetilde { m } _ { i _ { * } } ( w _ { \tau _ { \beta _ { d } } } ) \leq 2 \beta _ { d } , \quad \operatorname* { m a x } _ { k \neq i _ { * } } \vert m _ { k } ( w _ { \tau _ { \beta _ { d } } } ) \vert \leq C _ { \alpha } \alpha _ { d } \right\} .
$$

By Proposition 7.1 and Lemma $\mathrm { A . 1 } , \mathbb { P } ( \mathcal { E } _ { d } ) = 1 - o ( 1 )$ . Moreover, $\mathcal { E } _ { d } \in \mathcal { F } _ { \tau _ { \beta _ { d } } }$

By Proposition 7.2 and the tower property,

$$
\mathbb { P } \left( \mathcal { E } _ { d } \cap \left\{ \tau _ { \rho } - \tau _ { \beta _ { d } } > \frac { C _ { \rho } } { \eta _ { d } \beta _ { d } ^ { p - 2 } } \right\} \right) = \mathbb { E } \left[ \mathbf { 1 } _ { \mathcal { E } _ { d } } \mathbb { P } \left( \tau _ { \rho } - \tau _ { \beta _ { d } } > \frac { C _ { \rho } } { \eta _ { d } \beta _ { d } ^ { p - 2 } } \Bigg | \mathcal { F } _ { \tau _ { \beta _ { d } } } \right) \right] \leq \varepsilon _ { d } .
$$

Hence, with probability $1 - o ( 1 )$

$$
\tau _ { \rho } \leq \frac { C _ { \beta } } { \eta _ { d } \gamma _ { d } ^ { p - 2 } } + \frac { C _ { \rho } } { \eta _ { d } \beta _ { d } ^ { p - 2 } } .
$$

Since $\beta _ { d } = \gamma _ { d } ( \log d ) ^ { B }$ , we have

$$
\frac { 1 } { \eta _ { d } \beta _ { d } ^ { p - 2 } } = ( \log d ) ^ { - B ( p - 2 ) } \frac { 1 } { \eta _ { d } \gamma _ { d } ^ { p - 2 } } = o \left( \frac { 1 } { \eta _ { d } \gamma _ { d } ^ { p - 2 } } \right) .
$$

Therefore, after increasing the constant C if necessary,

$$
\tau _ { \rho } \leq \frac { C } { \eta _ { d } \gamma _ { d } ^ { p - 2 } }
$$

with probability $1 - o ( 1 )$ (1).

## 7.2 Microscopic selection

We now study the dynamics while all correlations remain microscopic and prove Proposition 7.1. According to (15), we have the evolution

$$
m _ { k } ( w _ { t } ) = m _ { k } ( w _ { 0 } ) + p a _ { p } ^ { 2 } \eta _ { d } \sum _ { \ell = 1 } ^ { t } m _ { k } ( w _ { \ell - 1 } ) ^ { p - 1 } + E _ { k , t } ,
$$

where $E _ { k , t }$ is defined in (16). Set

$$
T _ { 0 } : = \left\lceil \frac { C _ { 0 } } { \eta _ { d } \gamma _ { d } ^ { p - 2 } } \right\rceil ,\tag{23}
$$

where $C _ { 0 } > 0$ is fixed and suficiently large, and define

$$
\hat { \tau } _ { \beta _ { d } } : = \operatorname* { i n f } \left\{ t \geq 0 \colon \operatorname* { m a x } _ { 0 \leq k \leq r - 1 } | m _ { k } ( w _ { t } ) | \geq \beta _ { d } \right\} .
$$

For even $p , \hat { \tau } _ { \beta _ { d } } = \tau _ { \beta _ { d } }$ . For odd $p$ we prove below that the first absolute crossing is necessarily positive, and hence the two stopping times agree up to $T _ { 0 }$

Lemma 7.3. Suppose $\beta _ { d } = \gamma _ { d } ( \log d ) ^ { B }$ with $B > 2 / ( p - 2 )$ and $\eta _ { d } = d ^ { - p / 2 } ( \log d ) ^ { - A }$ with $A >$ $A _ { * } ( p , P )$ . Then, with probability $1 - o ( 1 )$ ,

$$
\operatorname* { m a x } _ { 0 \leq k \leq r - 1 } \operatorname* { m a x } _ { 0 \leq t \leq \hat { \tau } _ { \beta _ { d } } \wedge T _ { 0 } } | E _ { k , t } | = o \big ( ( \log d ) ^ { - 2 } \gamma _ { d } \big ) .
$$

Corollary 7.4. With probability $1 - o ( 1 )$

$$
\hat { \tau } _ { \beta _ { d } } \wedge T _ { 0 } = \tau _ { \beta _ { d } } \wedge T _ { 0 } .
$$

Consequently Lemma 7.3 remains valid with $\hat { \tau } _ { \beta _ { d } }$ replaced by $\tau _ { \beta _ { d } }$ .

Proof. For even $p$ the two stopping times coincide by definition. Suppose $p$ is odd. Then $p - 1$ is even, so the leading term in (15) is nonnegative. Hence, for $t \leq \hat { \tau } _ { \beta _ { d } } \wedge T _ { 0 }$

$$
m _ { k } ( w _ { t } ) \geq m _ { k } ( w _ { 0 } ) - \vert E _ { k , t } \vert .
$$

On the initialization event, max $_ k \left| m _ { k } ( w _ { 0 } ) \right| \leq C \gamma _ { d }$ , while Lemma 7.3 gives max $\dot { \cdot } _ { k , t } | E _ { k , t } | = o ( \gamma _ { d } )$ Thus

$$
m _ { k } ( w _ { t } ) \ge - C \gamma _ { d } - o ( \gamma _ { d } ) > - \beta _ { d }
$$

for all suficiently large $d ,$ uniformly in k and in the stopped time interval. No coordinate can therefore exit through $- \beta _ { d }$ before $T _ { 0 }$ , proving the claim. □

Lemma 7.5. With probability $1 - o ( 1 )$

$$
\operatorname* { m a x } _ { 0 \leq k \leq r - 1 0 \leq t \leq \hat { \tau } _ { \beta _ { d } } \wedge T _ { 0 } } \left| \eta _ { d } \sum _ { \ell = 1 } ^ { t } \xi _ { k , \ell } \right| \leq C \eta _ { d } \left( \sqrt { T _ { 0 } \log d } + ( \log d ) ^ { P + 1 } \right) .
$$

Proof. For $1 \leq \ell \leq T _ { 0 }$ set

$$
Y _ { k , \ell } = \mathbf { 1 } _ { \{ \ell \leq \hat { \tau } _ { \beta _ { d } } \} } \xi _ { k , \ell } .
$$

Then $( Y _ { k , \ell } ) _ { \ell }$ is a martingale diference sequence and

$$
\operatorname* { m a x } _ { t \leq \hat { \tau } _ { \beta _ { d } } \wedge T _ { 0 } } \left. \sum _ { \ell = 1 } ^ { t } \xi _ { k , \ell } \right. = \operatorname* { m a x } _ { t \leq T _ { 0 } } \left. \sum _ { \ell = 1 } ^ { t } Y _ { k , \ell } \right. .
$$

On $\{ \ell \le \hat { \tau } _ { \beta _ { d } } \} , | m _ { k } ( w _ { \ell - 1 } ) | < \beta _ { d }$ . Since $r \beta _ { d } ^ { 2 } = o ( 1 )$ , Lemma 6.4 gives, for all suficiently large $d ,$

$$
\begin{array} { r } { \mathbb E [ Y _ { k , \ell } ^ { 2 } \mid \mathcal F _ { \ell - 1 } ] \le C _ { \sigma } , \qquad ( \mathbb E [ | Y _ { k , \ell } | ^ { q } \mid \mathcal F _ { \ell - 1 } ] ) ^ { 1 / q } \le C _ { \sigma } q ^ { P } . } \end{array}\tag{24}
$$

Choose $q _ { d } = \lceil c \log d \rceil$ and $R _ { d } = C ( \log d ) ^ { P }$ , with $c , C$ suficiently large. Conditional Markov’s inequality gives

$$
\begin{array} { r } { \mathbb { P } ( | Y _ { k , \ell } | > R _ { d } \ | \ \mathcal { F } _ { \ell - 1 } ) \le 4 ^ { - q _ { d } } . } \end{array}
$$

Since $T _ { 0 } = d ^ { p - 1 } \operatorname { p o l y l o g } ( d ) \leq d ^ { p }$ and $r \leq d ,$ a union bound shows that, with probability $1 - o ( 1 )$

$$
| Y _ { k , \ell } | \leq R _ { d } \qquad { \mathrm { f o r ~ a l l ~ } } k < r , \ \ell \leq T _ { 0 } .
$$

Define the truncated and recentered increments

$$
\widehat { Y } _ { k , \ell } = Y _ { k , \ell } \mathbf { 1 } _ { \left\{ | Y _ { k , \ell } | \leq R _ { d } \right\} } - \mathbb { E } \big [ Y _ { k , \ell } \mathbf { 1 } _ { \left\{ | Y _ { k , \ell } | \leq R _ { d } \right\} } \mid \mathcal { F } _ { \ell - 1 } \big ] .
$$

Then $| \widehat { Y } _ { k , \ell } | \leq 2 R _ { d }$ and the predictable quadratic variation up to $T _ { 0 }$ is at most $C _ { \sigma } T _ { 0 }$ . Freedman’s maximal inequality therefore gives, after a union bound over $k < r$

$$
\operatorname* { m a x } _ { k < r } \operatorname* { m a x } _ { t \leq T _ { 0 } } \left. \sum _ { \ell = 1 } ^ { t } \widehat { Y } _ { k , \ell } \right. \leq C \left( \sqrt { T _ { 0 } \log d } + R _ { d } \log d \right)
$$

with probability $1 - o ( 1 )$ ).

Finally, the truncation bias satisfies

$$
\left| \mathbb { E } \left[ Y _ { k , \ell } \mathbf { 1 } _ { \left\{ | Y _ { k , \ell } | \leq R _ { d } \right\} } \mid \mathcal { F } _ { \ell - 1 } \right] \right| \leq R _ { d } 4 ^ { - q _ { d } } ,
$$

so $T _ { 0 } R _ { d } 4 ^ { - q _ { d } } = o ( 1 ) \quad$ after increasing c. Since $R _ { d } = C ( \log d ) ^ { P }$ , the claim follows after multiplying by $\eta _ { d }$ □

Proof of Lemma 7.3. Set $\nu _ { d } = ( \log d ) ^ { - 2 }$ . We control the four terms in (16).

Higher-Hermite remainder. For $\ell \leq \hat { \tau } _ { \beta _ { d } } \wedge T _ { 0 }$ , (18) gives

$$
| R _ { \sigma } ( m _ { k } ( w _ { \ell - 1 } ) ) | \leq C _ { \sigma } \beta _ { d } ^ { p } .
$$

Therefore

$$
\frac { \underset { k , t } { \operatorname* { m a x } } \left| \eta _ { d } \sum _ { \ell = 1 } ^ { t } R _ { \sigma } ( m _ { k } ( w _ { \ell - 1 } ) ) \right| } { \nu _ { d } \gamma _ { d } } \leq C _ { \sigma } \frac { \eta _ { d } T _ { 0 } \beta _ { d } ^ { p } } { \nu _ { d } \gamma _ { d } } \lesssim \frac { \gamma _ { d } } { \nu _ { d } } \left( \frac { \beta _ { d } } { \gamma _ { d } } \right) ^ { p } = o ( 1 ) .
$$

Interaction term. On the same stopped interval, (13) and (18) imply

$$
| \Gamma _ { k } ( w _ { \ell - 1 } ) | \le C _ { \sigma } ( r \mu _ { * } + r \beta _ { d } ^ { 2 } ) \beta _ { d } ^ { p - 1 } .
$$

Hence

$$
\frac { \displaystyle \operatorname* { m a x } _ { k , t } \bigg | \eta _ { d } \sum _ { \ell = 1 } ^ { t } \Gamma _ { k } ( w _ { \ell - 1 } ) \bigg | } { \nu _ { d } \gamma _ { d } } \le C _ { \sigma } \big ( r \mu _ { * } + r \beta _ { d } ^ { 2 } \big ) \nu _ { d } ^ { - 1 } \left( \frac { \beta _ { d } } { \gamma _ { d } } \right) ^ { p - 1 } = o ( 1 ) ,
$$

because $r \mu _ { * }$ and $r \beta _ { d } ^ { 2 }$ decay polynomially in $d ,$ whereas all remaining factors are polylogarithmic.

Normalization term. Since $T _ { 0 }$ is polynomial in $d ,$ Lemma 6.5 gives, with probability $1 - o ( 1 )$

$$
\operatorname* { m a x } _ { 1 \leq \ell \leq T _ { 0 } } \| G _ { \ell } \| \leq C _ { \sigma } \sqrt { d } ( \log d ) ^ { P } .\tag{25}
$$

Our choice of A ensures $C _ { \sigma } \sqrt { d } \left( \log d \right) ^ { P } = o ( \beta _ { d } )$ . Thus, also at the exit step,

$$
\begin{array} { r } { | m _ { k } ( w _ { \ell } ) | \leq \beta _ { d } + C _ { \sigma } \eta _ { d } \sqrt { d } \left( \log d \right) ^ { P } \leq 2 \beta _ { d } . } \end{array}
$$

Using $\sqrt { 1 + x } - 1 \leq x / 2$

$$
\operatorname* { m a x } _ { k , t } \left. \sum _ { \ell = 1 } ^ { t } ( \sqrt { 1 + \eta _ { d } ^ { 2 } \| G _ { \ell } \| ^ { 2 } } - 1 ) m _ { k } ( w _ { \ell } ) \right. \leq C \eta _ { d } ^ { 2 } T _ { 0 } d ( \log d ) ^ { 2 P } \beta _ { d } .
$$

After division by $\nu _ { d } \gamma _ { d }$ this is at most

$$
C _ { \sigma } ( \log d ) ^ { - A - ( p - 2 ) / 2 + 2 P + B + 2 } ,
$$

which is $o ( 1 )$ for $A > A _ { * } ( p , P )$

Martingale term. By Lemma 7.5, after division by $\nu _ { d } \gamma _ { d }$ the first stochastic contribution is

$$
\frac { \eta _ { d } \sqrt { T _ { 0 } \log d } } { \nu _ { d } \gamma _ { d } } \lesssim ( \log d ) ^ { - A / 2 + 5 / 2 - p / 4 } = o ( 1 )
$$

for A suficiently large, while

$$
\frac { \eta _ { d } ( \log d ) ^ { P + 1 } } { \nu _ { d } \gamma _ { d } } \lesssim d ^ { - ( p - 1 ) / 2 } ( \log d ) ^ { - A + P + 5 / 2 } = o ( 1 ) .
$$

Combining the four estimates proves

$$
\operatorname* { m a x } _ { k < r } \operatorname* { m a x } _ { t \leq \hat { \tau } _ { \beta _ { d } } \wedge T _ { 0 } } | E _ { k , t } | = o ( \nu _ { d } \gamma _ { d } )
$$

with probability $1 - o ( 1 )$ , uniformly over admissible teachers and initializations.

We now give the proof of Proposition 7.1.

Proof of Proposition 7.1. Set $\nu _ { d } = ( \log d ) ^ { - 2 }$ and work on the intersection of the high-probability events from Lemma A.3, Lemma 7.3, Corollary 7.4, and (25). Let

$$
m _ { 0 } : = M _ { \sigma } ( w _ { 0 } ) , \qquad c \gamma _ { d } \leq m _ { 0 } \leq C \gamma _ { d } ,
$$

and

$$
e _ { d } : = \operatorname* { m a x } _ { k < r } \operatorname* { m a x } _ { t \leq \tau _ { \beta _ { d } } \wedge T _ { 0 } } | E _ { k , t } | = o ( \nu _ { d } m _ { 0 } ) .
$$

Then, for every $t \leq \tau _ { \beta _ { d } } \wedge T _ { 0 }$

$$
m _ { k } ( w _ { 0 } ) + p a _ { p } ^ { 2 } \eta _ { d } \sum _ { \ell = 0 } ^ { t - 1 } m _ { k } ( w _ { \ell } ) ^ { p - 1 } - e _ { d } \le m _ { k } ( w _ { t } ) \le m _ { k } ( w _ { 0 } ) + p a _ { p } ^ { 2 } \eta _ { d } \sum _ { \ell = 0 } ^ { t - 1 } m _ { k } ( w _ { \ell } ) ^ { p - 1 } + e _ { d } .\tag{26}
$$

Recall the oriented correlation $\tilde { m } _ { i _ { * } } ( w _ { t } )$ defined in (22). Then $\tilde { m } _ { i _ { * } } ( w _ { 0 } ) = m _ { 0 }$ and

$$
\widetilde { m } _ { i _ { * } } ( w _ { t } ) = m _ { 0 } + p a _ { p } ^ { 2 } \eta _ { d } \sum _ { \ell = 0 } ^ { t - 1 } \widetilde { m } _ { i _ { * } } ( w _ { \ell } ) ^ { p - 1 } + \overline { { E } } _ { * , t } , \qquad | \overline { { E } } _ { * , t } | \leq e _ { d } .
$$

Define

$$
L _ { t } : = m _ { 0 } - e _ { d } + p a _ { p } ^ { 2 } \eta _ { d } \sum _ { \ell = 0 } ^ { t - 1 } \widetilde { m } _ { i _ { * } } ( w _ { \ell } ) ^ { p - 1 } .
$$

Then $\tilde { m } _ { i _ { * } } ( w _ { t } ) \geq L _ { t } , L _ { 0 } > 0$ , and

$$
L _ { t } - L _ { t - 1 } = p a _ { p } ^ { 2 } \eta _ { d } \widetilde { m } _ { i _ { * } } ( w _ { t - 1 } ) ^ { p - 1 } \geq p a _ { p } ^ { 2 } \eta _ { d } L _ { t - 1 } ^ { p - 1 } .\tag{27}
$$

For a favorable competitor, set $b _ { k , t } = | m _ { k } ( w _ { t } ) |$ when $p$ is even. When $p$ is odd, first consider competitors with $m _ { k } ( w _ { 0 } ) \ge 0$ and set $b _ { k , t } = m _ { k } ( w _ { t } )$ as long as this remains nonnegative. Define

$$
U _ { k , t } : = b _ { k , 0 } + e _ { d } + p a _ { p } ^ { 2 } \eta _ { d } \sum _ { \ell = 0 } ^ { t - 1 } b _ { k , \ell } ^ { p - 1 } .
$$

Then $b _ { k , t } \leq U _ { k , t }$ and

$$
U _ { k , t } - U _ { k , t - 1 } \leq p a _ { p } ^ { 2 } \eta _ { d } U _ { k , t - 1 } ^ { p - 1 } .
$$

The initialization gap gives

$$
L _ { 0 } - U _ { k , 0 } \geq \nu _ { d } m _ { 0 } - 2 e _ { d } > 0 ,
$$

so monotonicity of $x \mapsto x ^ { p - 1 }$ on $\mathbb { R } _ { + }$ implies inductively that the selected coordinate remains strictly ahead of every favorable competitor.

Time to reach $\beta _ { d } .$ . Set $q = p - 2$ and

$$
t _ { \beta } : = \operatorname* { i n f } \{ t \geq 0 : L _ { t } \geq \beta _ { d } \} .
$$

Before this time, the mean value theorem and (27) give

$$
L _ { t - 1 } ^ { - q } - L _ { t } ^ { - q } \ge \frac { q p a _ { p } ^ { 2 } \eta _ { d } } { ( 1 + p a _ { p } ^ { 2 } \eta _ { d } \beta _ { d } ^ { q } ) ^ { q + 1 } } .
$$

Hence

$$
t _ { \beta } \leq 1 + \frac { ( 1 + p a _ { p } ^ { 2 } \eta _ { d } \beta _ { d } ^ { q } ) ^ { q + 1 } } { q p a _ { p } ^ { 2 } \eta _ { d } L _ { 0 } ^ { q } } \left[ 1 - \left( \frac { L _ { 0 } } { \beta _ { d } } \right) ^ { q } \right] .\tag{28}
$$

Since $p a _ { p } ^ { 2 } \eta _ { d } \beta _ { d } ^ { q } = o ( \nu _ { d } )$ and $L _ { 0 } \asymp \gamma _ { d } .$

$$
t _ { \beta } \lesssim \frac { 1 } { \eta _ { d } \gamma _ { d } ^ { p - 2 } } .
$$

Taking $C _ { 0 }$ suficiently large gives $t _ { \beta } \leq T _ { 0 }$ , and therefore $\tau _ { \beta _ { d } } \leq t _ { \beta }$

Size of the competitors. The initialization gap implies

$$
\frac { U _ { k , 0 } } { L _ { 0 } } \leq 1 - ( 1 - o ( 1 ) ) \nu _ { d } .
$$

Iterating the upper scalar recursion gives

$$
U _ { k , t } \leq U _ { k , 0 } \left( 1 - q p a _ { p } ^ { 2 } \eta _ { d } U _ { k , 0 } ^ { q } t \right) ^ { - 1 / q }
$$

whenever the denominator is positive. Combining this with (28), and using the extra +1 term only through $q p a _ { p } ^ { 2 } \eta _ { d } U _ { k , 0 } ^ { q } = o ( \nu _ { d } )$ , yields

$$
\begin{array} { r } { 1 - q p a _ { p } ^ { 2 } \eta _ { d } U _ { k , 0 } ^ { q } t \geq c _ { p } \nu _ { d } , \qquad t \leq \tau _ { \beta _ { d } } , } \end{array}
$$

for some $c _ { p } > 0$ . Consequently

$$
\operatorname* { m a x } _ { t \leq \tau _ { \beta _ { d } } } | m _ { k } ( w _ { t } ) | \lesssim \gamma _ { d } \nu _ { d } ^ { - 1 / ( p - 2 ) } = \alpha _ { d } = o ( \beta _ { d } ) .\tag{29}
$$

Suppose now that $p$ is odd and $m _ { k } ( w _ { 0 } ) < 0$ . Let

$$
\nu _ { k } : = \operatorname* { i n f } \{ t \geq 0 \colon m _ { k } ( w _ { t } ) \geq 0 \} .
$$

Before $\nu _ { k } ,$ , (15) gives $| m _ { k } ( w _ { t } ) | \lesssim \gamma _ { d }$ . If $\nu _ { k } \leq \tau _ { \beta _ { d } }$ and $m _ { k } ( w _ { \nu _ { k } - 1 } ) = - u$ , then $u < \beta _ { d }$ and

$$
- u + p a _ { p } ^ { 2 } \eta _ { d } u ^ { p - 1 } = - u ( 1 - p a _ { p } ^ { 2 } \eta _ { d } u ^ { p - 2 } ) \leq 0 .
$$

The crossing overshoot is therefore at most $2 e _ { d }$ . Starting from this size, the positive scalar envelope remains $O ( e _ { d } )$ because

$$
p a _ { p } ^ { 2 } \eta _ { d } e _ { d } ^ { p - 2 } \tau _ { \beta _ { d } } \lesssim \left( \frac { e _ { d } } { \gamma _ { d } } \right) ^ { p - 2 } = o ( 1 ) .
$$

This proves (29) for initially negative competitors as well. The same argument handles later sign changes.

Finally, the preceding comparison shows that $i _ { * }$ is the coordinate that reaches the threshold. Immediately before the crossing its favorable magnitude is below $\beta _ { d }$ , and the exact normalized update gives

$$
\begin{array} { r } { \beta _ { d } \leq M _ { \sigma } ( w _ { \tau _ { \beta _ { d } } } ) \leq \beta _ { d } + \eta _ { d } \| G _ { \tau _ { \beta _ { d } } } \| . } \end{array}
$$

By (25), $\eta _ { d } \| G _ { \tau _ { \beta _ { d } } } \| = o ( \beta _ { d } )$ , so

$$
M _ { \sigma } ( w _ { \tau _ { \beta _ { d } } } ) = \beta _ { d } ( 1 + o ( 1 ) ) .
$$

This proves Proposition 7.1.

## 7.3 Growth to a macroscopic level

We now prove Proposition 7.2. By Proposition 7.1, with probability $1 - o ( 1 )$ there is a unique index $i _ { * }$ such that

$$
\beta _ { d } \leq \tilde { m } _ { i _ { * } } ( w _ { \tau _ { \beta _ { d } } } ) \leq 2 \beta _ { d } , \qquad \operatorname* { m a x } _ { k \neq i _ { * } } | m _ { k } ( w _ { \tau _ { \beta _ { d } } } ) | \leq C _ { \alpha } \alpha _ { d } ,
$$

where $\tilde { m } _ { i _ { * } } ( w _ { t } )$ is the oriented correlation defined in (22). We may thus orient all correlations according to the selected direction and suppress the bars below, so that $m _ { i _ { * } } ( w _ { t } ) = \widetilde { m } _ { i _ { * } } ( w _ { t } ) > 0$

The competitor dynamics contain a contractive factor generated by the selected overlap. We first record the deterministic estimate used to exploit this damping.

Lemma 7.6 (Damping preserves partial-sum bounds). Suppose real sequences satisfy

$$
d _ { t } x _ { t + 1 } = q _ { t } x _ { t } + F _ { t } + N _ { t + 1 } - N _ { t } , \qquad 0 \leq q _ { t } \leq 1 , \quad d _ { t } \geq 1 , \quad N _ { 0 } = 0 .
$$

Then, for every $t \geq 0$

$$
| x _ { t } | \leq | x _ { 0 } | + \sum _ { \ell = 0 } ^ { t - 1 } | F _ { \ell } | + 2 \operatorname* { m a x } _ { 0 \leq \ell \leq t } | N _ { \ell } | .
$$

Proof. Iterating the recursion, an increment introduced at time ℓ is multiplied at time t by

$$
\omega _ { \ell , t } : = \frac { 1 } { d _ { \ell } } \prod _ { j = \ell + 1 } ^ { t - 1 } \frac { q _ { j } } { d _ { j } } , \qquad 0 \le \ell < t .
$$

The weights lie in [0, 1] and are nondecreasing in ℓ. Summation by parts therefore gives

$$
\left| \sum _ { \ell = 0 } ^ { t - 1 } \omega _ { \ell , t } ( N _ { \ell + 1 } - N _ { \ell } ) \right| \leq 2 \operatorname* { m a x } _ { 0 \leq \ell \leq t } | N _ { \ell } | .
$$

The coeficients multiplying $x _ { 0 }$ and the forcing terms are at most one, which proves the claim.

Proof of Proposition 7.2. Condition on ${ \mathcal { F } } _ { { \tau } _ { \beta _ { d } } }$ , then

$$
\beta _ { d } \leq \widetilde { m } _ { i _ { * } } ( w _ { 0 } ) \leq 2 \beta _ { d } , \qquad \operatorname* { m a x } _ { k \neq i _ { * } } \vert m _ { k } ( w _ { 0 } ) \vert \leq C _ { \alpha } \alpha _ { d } .
$$

Set

$$
c : = \frac { p a _ { p } ^ { 2 } ( 1 - \rho ^ { 2 } ) } { 4 } , \qquad \varepsilon : = \frac 1 4 , \qquad T : = \left\lceil \frac { C _ { T } } { \eta _ { d } \beta _ { d } ^ { p - 2 } } \right\rceil ,
$$

where $C _ { T }$ is chosen below. Define

$$
D _ { t } : = c \eta _ { d } \sum _ { \ell = 0 } ^ { t - 1 } m _ { i * } ( w _ { \ell } ) ^ { p - 1 }
$$

and

$$
\tau : = T \wedge \operatorname* { i n f } \left\{ t \geq 0 : \ \operatorname* { m a x } _ { \bf k \neq i _ { * } } | m _ { k } ( w _ { t } ) | \geq 2 C _ { \alpha } \alpha _ { d } , \ \right\} .
$$

For $t < \tau , 0 < m _ { i _ { * } } ( w _ { t } ) < \rho$ and $D _ { t \wedge \tau } \leq 1 + c \eta _ { d }$

Step 1: Progress of the selected correlation. From Lemma 6.1,

$$
\begin{array} { r l } & { - \langle \Pi ^ { i _ { * } } w ^ { * } , \nabla \Phi ( w _ { t } ) \rangle = ( 1 - m _ { i _ { * } } ( w _ { t } ) ^ { 2 } ) \phi ^ { \prime } ( m _ { i _ { * } } ( w _ { t } ) ) } \\ & { \qquad + \displaystyle \sum _ { j \ne i _ { * } } c _ { j - i _ { * } } \phi ^ { \prime } ( m _ { j } ( w _ { t } ) ) - m _ { i _ { * } } ( w _ { t } ) \sum _ { j \ne i _ { * } } m _ { j } ( w _ { t } ) \phi ^ { \prime } ( m _ { j } ( w _ { t } ) ) . } \end{array}
$$

For $m \in [ 0 , \rho ]$ , (19) gives $\phi ^ { \prime } ( m ) \geq p a _ { p } ^ { 2 } m ^ { p - 1 }$ . For the competitors, (18) gives $| \phi ^ { \prime } ( m _ { j } ) | \leq C _ { \sigma } | m _ { j } | ^ { p - 1 }$ Hence, for $t < \tau$

$$
- \langle \Pi ^ { i _ { * } } w ^ { * } , \nabla \Phi ( w _ { t } ) \rangle \geq p a _ { p } ^ { 2 } ( 1 - \rho ^ { 2 } ) m _ { i _ { * } } ( w _ { t } ) ^ { p - 1 } - C _ { \sigma } r \mu _ { * } \alpha _ { d } ^ { p - 1 } - C _ { \sigma } r \alpha _ { d } ^ { p } m _ { i _ { * } } ( w _ { t } ) .
$$

Since $m _ { i _ { * } } ( w _ { t } ) \geq \beta _ { d } / 2$ , both error terms are $o ( m _ { i _ { * } } ( w _ { t } ) ^ { p - 1 } )$ . Thus, for all suficiently large d,

$$
- \langle \Pi ^ { i _ { * } } w ^ { * } , \nabla \Phi ( w _ { t } ) \rangle \geq 2 c m _ { i _ { * } } ( w _ { t } ) ^ { p - 1 } .\tag{30}
$$

Lemma 6.5 applies uniformly after the restart. Since $T$ is polynomial in d, with conditional probability $1 - o ( 1 )$ ,

$$
\operatorname* { m a x } _ { 1 \leq t \leq T } \| G _ { t } \| \leq C _ { \sigma } \sqrt { d } \big ( \log d \big ) ^ { P } .\tag{31}
$$

Our choice of A ensures

$$
\frac { C _ { \sigma } \eta _ { d } \sqrt { d } ( \log d ) ^ { P } } { \beta _ { d } } = o ( 1 ) , \qquad \frac { C _ { \sigma } ^ { 2 } \eta _ { d } d ( \log d ) ^ { 2 P } } { \beta _ { d } ^ { p - 2 } } = o ( 1 ) .\tag{32}
$$

The normalization correction is therefore bounded by

$$
\left| ( \sqrt { 1 + \eta _ { d } ^ { 2 } \| G _ { t + 1 } \| ^ { 2 } } - 1 ) m _ { i _ { * } } ( w _ { t + 1 } ) \right| \leq c \eta _ { d } m _ { i _ { * } } ( w _ { t } ) ^ { p - 1 } .
$$

Combining with (30) gives

$$
m _ { i _ { * } } ( w _ { t + 1 } ) \geq m _ { i _ { * } } ( w _ { t } ) + c \eta _ { d } m _ { i _ { * } } ( w _ { t } ) ^ { p - 1 } + \eta _ { d } \xi _ { i _ { * } , t + 1 } , \qquad t < \tau .\tag{33}
$$

Define the stopped martingales

$$
N _ { k , t } : = \eta _ { d } \sum _ { \ell = 0 } ^ { t - 1 } \mathbf { 1 } _ { \left\{ \ell < \tau \right\} } \xi _ { k , \ell + 1 } .
$$

For the selected coordinate,

$$
\sum _ { \ell < \tau } m _ { i _ { * } } ( w _ { \ell } ) ^ { 2 } \leq \frac { C } { \eta _ { d } \beta _ { d } ^ { p - 3 } } ,
$$

because $m _ { i _ { * } } ( w _ { \ell } ) \geq \beta _ { d } / 2$ and $D _ { \tau } \leq 1 + c \eta _ { d }$ . Lemma 6.4 yields

$$
\mathbb { E } [ N _ { i * , T } ^ { 2 } ] \le C _ { \sigma } \eta _ { d } ^ { 2 } \left( T + \frac { r } { \eta _ { d } \beta _ { d } ^ { p - 3 } } \right) \le C _ { \sigma } \eta _ { d } ^ { 2 } T ( 1 + r \beta _ { d } ) = o ( \beta _ { d } ^ { 2 } ) .
$$

Hence Doob’s maximal inequality implies

$$
\operatorname* { m a x } _ { 0 \leq n \leq T } | N _ { i _ { * } , n } | \leq \varepsilon \beta _ { d }\tag{34}
$$

with conditional probability $1 - o ( 1 )$ . Summing (33) gives, for every $t \leq \tau ,$

$$
m _ { i * } ( w _ { t } ) \geq ( 1 - \varepsilon ) \beta _ { d } + D _ { t } .\tag{35}
$$

Thus neither the lower boundary $\beta _ { d } / 2$ nor the cutof $D _ { t } = 1$ can trigger τ .

Step 2: Persistence of the microscopic separation. For $k \neq i _ { * }$ and $\ell < \tau , | m _ { k } ( w _ { \ell } ) | < 2 C _ { \alpha } \alpha _ { d } ,$ so Lemma 6.4 gives uniformly

$$
\begin{array} { r } { \mathbb { E } [ ( \mathbf { 1 } _ { \{ \ell < \tau \} } \xi _ { k , \ell + 1 } ) ^ { 2 } \mid \mathcal { F } _ { \ell } ] \leq C _ { \sigma } , \qquad \| \mathbf { 1 } _ { \{ \ell < \tau \} } \xi _ { k , \ell + 1 } \| _ { L ^ { q } ( \cdot | \mathcal { F } _ { \ell } ) } \leq C _ { \sigma } q ^ { P } . } \end{array}
$$

Repeating the truncation–Freedman argument from Lemma 7.5 gives, with conditional probability $1 - o ( 1 )$

$$
\operatorname* { m a x } _ { k \neq i _ { * } } \operatorname* { m a x } _ { 0 \leq t \leq T } | N _ { k , t } | \leq C \eta _ { d } \left( \sqrt { T \log d } + ( \log d ) ^ { P + 1 } \right) = o ( \alpha _ { d } ) .\tag{36}
$$

For $t < \tau$ and $k \neq i _ { * }$ , isolate the damping generated by the selected coordinate in the exact recursion:

$$
\sqrt { 1 + \eta _ { d } ^ { 2 } \| G _ { t + 1 } \| ^ { 2 } } m _ { k } ( w _ { t + 1 } ) = q _ { t } m _ { k } ( w _ { t } ) + \eta _ { d } F _ { k , t } + \eta _ { d } \xi _ { k , t + 1 } ,
$$

where

$$
q _ { t } : = 1 - \eta _ { d } m _ { i _ { * } } ( w _ { t } ) \phi ^ { \prime } ( m _ { i _ { * } } ( w _ { t } ) )
$$

and

$$
F _ { k , t } : = \phi ^ { \prime } ( m _ { k } ( w _ { t } ) ) + \sum _ { j \neq k } c _ { j - k } \phi ^ { \prime } ( m _ { j } ( w _ { t } ) ) - m _ { k } ( w _ { t } ) \sum _ { j \neq i _ { * } } m _ { j } ( w _ { t } ) \phi ^ { \prime } ( m _ { j } ( w _ { t } ) ) .
$$

Since $m _ { i _ { * } } > 0$ and $\phi ^ { \prime } ( m _ { i _ { * } } ) > 0$ , while $\eta _ { d }  0$ , we have $0 \leq q _ { t } \leq 1$ for all suficiently large d. Moreover, (18) implies

$$
\begin{array} { r } { | F _ { k , t } | \leq C _ { \sigma } \left( \alpha _ { d } ^ { p - 1 } + \mu _ { * } m _ { i _ { * } } ( w _ { t } ) ^ { p - 1 } + r \alpha _ { d } ^ { p + 1 } \right) \leq C _ { \sigma } \left( \alpha _ { d } ^ { p - 1 } + \mu _ { * } m _ { i _ { * } } ( w _ { t } ) ^ { p - 1 } \right) , } \end{array}
$$

where we used $r \alpha _ { d } ^ { 2 } = o ( 1 )$ . Lemma 7.6 therefore gives, for $t \leq \tau$

$$
\left. m _ { k } ( w _ { t } ) \right. \leq C _ { \alpha } \alpha _ { d } + C _ { \sigma } \eta _ { d } T \alpha _ { d } ^ { p - 1 } + C _ { \sigma } \mu _ { * } \eta _ { d } \sum _ { \ell < t } m _ { i _ { * } } ( w _ { \ell } ) ^ { p - 1 } + 2 \operatorname* { m a x } _ { \ell \leq t } \left. N _ { k , \ell } \right. .
$$

Now

$$
\eta _ { d } T \alpha _ { d } ^ { p - 2 } = O \left( \left( \frac { \alpha _ { d } } { \beta _ { d } } \right) ^ { p - 2 } \right) = o ( 1 ) ,
$$

$D _ { t } \leq 1 + c \eta _ { d } , \mu _ { * } = o ( \alpha _ { d } )$ , and (36) is $o ( \alpha _ { d } )$ . Hence

$$
| m _ { k } ( w _ { t } ) | < 2 C _ { \alpha } \alpha _ { d }
$$

for every $k \neq i _ { * }$ and $t \leq \tau .$ , so no competitor can trigger the stopping time.

Step 3: Reaching the target. By (35),

$$
m _ { i _ { * } } ( w _ { t } ) \geq ( 1 - \varepsilon ) \beta _ { d } + c \eta _ { d } \sum _ { \ell = 0 } ^ { t - 1 } m _ { i _ { * } } ( w _ { \ell } ) ^ { p - 1 } = : L _ { t } .
$$

The standard discrete comparison for $x ^ { \prime } = c x ^ { p - 1 }$ gives

$$
t _ { \rho } : = \operatorname* { i n f } \{ n : L _ { n } \geq \rho \} \leq \frac { C _ { \sigma , \rho } } { \eta _ { d } ( ( 1 - \varepsilon ) \beta _ { d } ) ^ { p - 2 } } .
$$

Choosing $C _ { T }$ suficiently large ensures $t _ { \rho } \leq T$ . Since the other stopping mechanisms have already been excluded, the selected coordinate reaches $\rho$ before $T ,$ and

$$
\tau _ { \rho } \leq \left\lceil \frac { C _ { \rho } } { \eta _ { d } \beta _ { d } ^ { p - 2 } } \right\rceil .
$$

Throughout this interval,

$$
m _ { i _ { * } } ( w _ { t } ) \geq \frac { \beta _ { d } } { 2 } , \qquad \operatorname* { m a x } _ { k \neq i _ { * } } | m _ { k } ( w _ { t } ) | < 2 C _ { \alpha } \alpha _ { d } .
$$

Since $\alpha _ { d } = o ( \beta _ { d } ) , i ,$ <sub>∗</sub> remains the unique favorable leader. Therefore $M _ { \sigma } ( w _ { \tau _ { \rho } } ) \geq \rho$ . This proves Proposition 7.2. □

## 8 Proofs for the augmented and untied students

We now prove parts (U) and (A) of Theorem 5.1. The key observation is that, because the correlation loss is linear in the predictor, the gradient with respect to one untied weight does not involve the other weights. Consequently, both procedures reduce to one-vector dynamics.

## 8.1 Componentwise reductions

For $u \in \mathbb { S } ^ { d - 1 }$ , define the single-weight loss

$$
L ^ { \mathrm { S } } ( u ; x , y ) : = - y \sigma ( \left. u , x \right. ) ,\tag{37}
$$

and its population counterpart

$$
\Psi ( u ) : = \mathbb { E } [ L ^ { \mathrm { S } } ( u ; x , y ) ] .
$$

By linearity of the untied predictor,

$$
\nabla _ { w ^ { j } } L ^ { \mathsf { U } } ( W ; x , y ) = \frac { 1 } { \sqrt { s } } \nabla L ^ { \mathrm { S } } ( w ^ { j } ; x , y ) , \qquad 1 \le j \le s .\tag{38}
$$

For the augmented procedure, using $\langle w , \Pi ^ { k } x \rangle = \langle \Pi ^ { - k } w , x \rangle$ and the fact that $k \mapsto - k$ (mod r) permutes the orbit,

$$
L ^ { \mathsf { A } } ( W ; x , y ) = - { \frac { y } { r { \sqrt { s } } } } \sum _ { j = 1 } ^ { s } \sum _ { k = 0 } ^ { r - 1 } \sigma ( \langle w ^ { j } , \Pi ^ { k } x \rangle ) = { \frac { 1 } { \sqrt { r s } } } \sum _ { j = 1 } ^ { s } L ^ { \mathsf { T } } ( w ^ { j } ; x , y ) .\tag{39}
$$

Hence

$$
\nabla _ { w ^ { j } } L ^ { \mathsf { A } } ( W ; x , y ) = { \frac { 1 } { \sqrt { r s } } } \nabla L ^ { \mathsf { T } } ( w ^ { j } ; x , y ) .\tag{40}
$$

Using the step sizes in (6) and the identities in (38) and (40), we have

$$
\widetilde { w } _ { t } ^ { j , \mathsf { U } } = w _ { t - 1 } ^ { j , \mathsf { U } } - \frac { \eta _ { d } } { \sqrt { r } } \nabla L ^ { \mathrm { S } } ( w _ { t - 1 } ^ { j , \mathsf { U } } ; x _ { t } , y _ { t } ) ,\tag{41}
$$

$$
\widetilde { w } _ { t } ^ { j , \mathsf { A } } = w _ { t - 1 } ^ { j , \mathsf { A } } - \eta _ { d } \nabla L ^ { \mathsf { T } } ( w _ { t - 1 } ^ { j , \mathsf { A } } ; x _ { t } , y _ { t } ) .\tag{42}
$$

The normalization to the sphere is then performed componentwise as in (5). Note that (42) is an exact identity: conditionally on its initialization, every augmented weight follows precisely the tied SGD trajectory with reference step size $\eta _ { d } .$ , driven by the same stream of fresh samples.

Proof of part (A) of Theorem 5.1. Fix any weight index $1 \leq j \leq s$ . By (42), the trajectory of $w _ { t } ^ { j , \mathsf { A } }$ has the same law as the tied trajectory analyzed in part (T), with an independent uniform initialization. Therefore, with probability $1 - o ( 1 )$ , there exists

$$
t \leq \frac { C } { \eta _ { d } \gamma _ { d } ^ { p - 2 } }
$$

such that $M _ { \sigma } ( w _ { t } ^ { j , \mathsf { A } } ) \geq \rho .$ . This proves the required bound on $\tau _ { \rho } ^ { j , \mathsf { A } }$

It remains to analyze the unaugmented untied student. Again it is enough to follow one weight, which we denote by $\overset { \cdot } { u _ { t } } : = w _ { t } ^ { 1 , \mathsf { U } }$

## 8.2 One-weight untied dynamics

The population loss of the single-weight model is

$$
\Psi ( u ) = - \mathbb { E } \big [ y \sigma ( \langle u , x \rangle ) \big ] = - \frac { 1 } { \sqrt { r } } \sum _ { k = 0 } ^ { r - 1 } \phi ( m _ { k } ( u ) ) .\tag{43}
$$

Thus the population landscape is the same scalar overlap landscape as for the tied model, but its magnitude is smaller by a factor $r ^ { - 1 / 2 }$

Let

$$
\mathcal { F } _ { t } ^ { \mathrm { S } } : = \sigma ( w ^ { \ast } , u _ { 0 } , ( x _ { \ell } , y _ { \ell } ) _ { 1 \leq \ell \leq t } )
$$

be the natural filtration of the one-weight process. Set

$$
G _ { t } ^ { \mathrm { S } } : = \nabla L ^ { \mathrm { S } } ( u _ { t - 1 } ; x _ { t } , y _ { t } ) , \qquad H _ { t } ^ { \mathrm { S } } ( u ) : = L ^ { \mathrm { S } } ( u ; x _ { t } , y _ { t } ) - \Psi ( u ) ,
$$

and define

$$
\xi _ { k , t } ^ { \mathrm { S } } : = - \langle \Pi ^ { k } w ^ { * } , \nabla H _ { t } ^ { \mathrm { S } } ( u _ { t - 1 } ) \rangle .
$$

Exactly as in Lemma 6.1, diferentiating (43) gives

$$
\langle \Pi ^ { k } w ^ { * } , \nabla \Psi ( u ) \rangle = \frac { 1 } { \sqrt { r } } \big ( \phi ^ { \prime } ( m _ { k } ( u ) ) + \Gamma _ { k } ( u ) \big ) ,\tag{44}
$$

where $\Gamma _ { k }$ is the same function as in (13). Using the efective update (41), the exact overlap recursion is therefore

$$
\sqrt { 1 + \frac { \eta _ { d } ^ { 2 } } { r } \| G _ { t } ^ { \mathrm { S } } \| ^ { 2 } } m _ { k } ( u _ { t } ) = m _ { k } ( u _ { t - 1 } ) + \frac { \eta _ { d } } { r } ( \phi ^ { \prime } ( m _ { k } ( u _ { t - 1 } ) ) + \Gamma _ { k } ( u _ { t - 1 } ) ) + \frac { \eta _ { d } } { \sqrt { r } } \xi _ { k , t } ^ { \mathrm { S } } .\tag{45}
$$

Consequently,

$$
m _ { k } ( u _ { t } ) = m _ { k } ( u _ { 0 } ) + \frac { p a _ { p } ^ { 2 } \eta _ { d } } { r } \sum _ { \ell = 1 } ^ { t } m _ { k } ( u _ { \ell - 1 } ) ^ { p - 1 } + E _ { k , t } ^ { \cup } ,\tag{46}
$$

where

$$
\begin{array} { r l r } {  { E _ { k , t } ^ { \mathsf { U } } : = \frac { \eta _ { d } } { r } \sum _ { \ell = 1 } ^ { t } R _ { \sigma } ( m _ { k } ( u _ { \ell - 1 } ) ) + \frac { \eta _ { d } } { r } \sum _ { \ell = 1 } ^ { t } \Gamma _ { k } ( u _ { \ell - 1 } ) } } \\ & { } & { \quad + \frac { \eta _ { d } } { \sqrt { r } } \sum _ { \ell = 1 } ^ { t } \xi _ { k , \ell } ^ { \mathrm { S } } - \sum _ { \ell = 1 } ^ { t } ( \sqrt { 1 + \frac { \eta _ { d } ^ { 2 } } { r } \| G _ { \ell } ^ { \mathrm { S } } \| ^ { 2 } } - 1 ) m _ { k } ( u _ { \ell } ) . } \end{array}\tag{47}
$$

The stochastic estimates needed for (47) are simpler than in the tied model.

Lemma 8.1 (Single-weight stochastic bounds). Condition on a teacher satisfying the geometry event of Lemma A.1. There exists $C _ { \sigma } < \infty$ such that, uniformly over $u \in \mathbb { S } ^ { d - 1 } , k < r _ { \ast }$ , and $t \geq 1$ 2

$$
\mathbb { E } [ ( \xi _ { k , t } ^ { \mathrm { S } } ) ^ { 2 } \mid \mathcal { F } _ { t - 1 } ^ { \mathrm { S } } ] \leq C _ { \sigma } ,\tag{48}
$$

$$
\left( \mathbb { E } [ | \xi _ { k , t } ^ { \mathrm { S } } | ^ { a } \mid \mathcal { F } _ { t - 1 } ^ { \mathrm { S } } ] \right) ^ { 1 / a } \leq C _ { \sigma } a ^ { P } , \qquad a \geq 2 .\tag{49}
$$

Moreover, for every fixed $K < \infty$ , there exists $C _ { \sigma , K } < \infty$ such that, for every deterministic horizon $T \leq d ^ { K }$ 2

$$
\mathbb { P } \left( \operatorname* { m a x } _ { 1 \leq t \leq T } \| G _ { t } ^ { \mathrm { S } } \| > C _ { \sigma , K } \sqrt { d } ( \log d ) ^ { P } \right) = o ( 1 ) .\tag{50}
$$

Proof. Fix $u \in \mathbb { S } ^ { d - 1 }$ and set $v = ( \mathrm { I } _ { d } - u u ^ { \top } ) \Pi ^ { k } w ^ { \ast }$ , so $\| v \| \leq 1$ . Then

$$
\langle \Pi ^ { k } w ^ { * } , G ^ { \mathrm { S } } \rangle = - y \sigma ^ { \prime } ( \langle u , x \rangle ) \langle v , x \rangle .
$$

On the teacher-geometry event, $\| y \| _ { L ^ { a } } \leq C _ { \sigma } a ^ { P / 2 }$ for every $a \ge 2$ , by the same $L ^ { 2 }$ bound and Gaussian hypercontractivity used in Lemma 6.4. The factor $\sigma ^ { \prime } ( \langle u , x \rangle ) \langle v , x \rangle$ is a Gaussian polynomial of degree at most $P$ with uniformly bounded $L ^ { 2 } \mathrm { - n o r m ; }$ hence hypercontractivity and Cauchy–Schwarz give

$$
\| \langle \Pi ^ { k } w ^ { * } , G ^ { \mathrm { S } } \rangle \| _ { L ^ { a } } \leq C _ { \sigma } a ^ { P } .
$$

Centering can only decrease the variance and changes the $L ^ { a } .$ -norm by at most a factor two, proving (48)–(49).

For the gradient norm,

$$
\begin{array} { r } { G ^ { \mathrm { S } } = - y \sigma ^ { \prime } ( \langle u , x \rangle ) ( \mathrm { I } _ { d } - u u ^ { \top } ) x . } \end{array}
$$

Choosing an orthonormal basis of $u ^ { \perp }$ , applying the preceding polynomial moment estimate coordinatewise, and using Minkowski’s inequality yields

$$
\begin{array} { r } { \Big ( \mathbb { E } [ \| G ^ { \mathrm { S } } \| ^ { a } ] \Big ) ^ { 1 / a } \leq C _ { \sigma } \sqrt { d } a ^ { P } , \qquad a \geq 2 , } \end{array}
$$

uniformly in u. Taking $a \asymp$ log d, applying Markov’s inequality conditionally on the past, and then taking a union bound over $t \leq T \leq d ^ { K }$ proves (50). □

## 8.3 Proof of the untied weak recovery bound

We now compare (46) with the tied decomposition (15). The deterministic drift has been reduced by a factor r, while the centered stochastic increment has been reduced by a factor $\sqrt { r }$ . These two changes exactly balance when the time horizon is multiplied by r.

Fix the same scales $\alpha _ { d } , \beta _ { d }$ as in the tied proof, set $\nu _ { d } = ( \log d ) ^ { - 2 }$ , and define

$$
T _ { 0 } ^ { \mathsf { U } } : = \left\lceil \frac { C _ { 0 } r } { \eta _ { d } \gamma _ { d } ^ { p - 2 } } \right\rceil , \qquad \widehat { \tau } _ { \beta _ { d } } ^ { \mathsf { U } } : = \operatorname* { i n f } \left\{ t \geq 0 : \operatorname* { m a x } _ { k < r } | m _ { k } ( u _ { t } ) | \geq \beta _ { d } \right\} .
$$

The initialization Lemma A.3 applies unchanged to $u _ { 0 }$

We first record the analogue of Lemma 7.3.

Lemma 8.2 (Untied microscopic error). For $A > A _ { * } ( p , P )$ suficiently large,

$$
\operatorname* { m a x } _ { k < r } \operatorname* { m a x } _ { t \leq \mathcal { T } _ { \beta _ { d } } ^ { \mathsf { U } } \wedge T _ { 0 } ^ { \mathsf { U } } } | E _ { k , t } ^ { \mathsf { U } } | = o ( \nu _ { d } \gamma _ { d } ) ,
$$

with probability $1 - o ( 1 )$

Proof. The higher-Hermite and interaction terms are identical in size to their tied counterparts because

$$
{ \frac { \eta _ { d } } { r } } T _ { 0 } ^ { \mathsf { U } } \asymp \gamma _ { d } ^ { - ( p - 2 ) } .
$$

Thus the calculations in the proof of Lemma 7.3 apply verbatim.

For the martingale term, Lemma 8.1 and the same truncation–Freedman argument as in Lemma 7.5 give

$$
\operatorname* { m a x } _ { k < r } \operatorname* { m a x } _ { t \leq \bar { \tau } _ { \beta _ { d } } ^ { \mathsf { U } } \wedge T _ { 0 } ^ { \mathsf { U } } } \left| \frac { \eta _ { d } } { \sqrt { r } } \sum _ { \ell = 1 } ^ { t } \xi _ { k , \ell } ^ { \mathrm { S } } \right| \leq C \frac { \eta _ { d } } { \sqrt { r } } \left( \sqrt { T _ { 0 } ^ { \mathsf { U } } \log d } + ( \log d ) ^ { P + 1 } \right)
$$

with probability $1 - o ( 1 )$ . Since

$$
\frac { \eta _ { d } } { \sqrt { r } } \sqrt { T _ { 0 } ^ { \cup } \log d } \asymp \sqrt { \frac { \eta _ { d } \log d } { \gamma _ { d } ^ { p - 2 } } } ,
$$

this is exactly the leading martingale scale in the tied proof, while the truncation term is smaller by $r ^ { - 1 / 2 }$ . Hence the whole term is $o ( \nu _ { d } \gamma _ { d } )$

Finally, on the gradient event (50),

$$
\sqrt { 1 + \frac { \eta _ { d } ^ { 2 } } { r } \| G _ { \ell } ^ { \mathrm { S } } \| ^ { 2 } } - 1 \leq \frac { \eta _ { d } ^ { 2 } } { 2 r } \| G _ { \ell } ^ { \mathrm { S } } \| ^ { 2 } .
$$

Therefore the cumulative normalization error is bounded by

$$
C \frac { \eta _ { d } ^ { 2 } } { r } T _ { 0 } ^ { \mathsf { U } } d ( \log d ) ^ { 2 P } \beta _ { d } ,
$$

which coincides with the tied bound after using the definition of $T _ { 0 } ^ { \mathsf { U } }$ . This is again $o ( \nu _ { d } \gamma _ { d } )$ . Combining the four estimates proves the claim. □

If $p$ is odd, the analogue of Corollary 7.4 also holds: before $T _ { 0 } ^ { \mathsf { U } }$ , no coordinate can be the first to leave the microscopic region through $- \beta _ { d }$ . Indeed, the leading drift in (46) is nonnegative and Lemma 8.2 gives a cumulative error $o ( \gamma _ { d } )$ , whereas $\gamma _ { d } = o ( \beta _ { d } )$

On this event, the scalar comparison proof of Proposition 7.1 applies without any structural change, with $\eta _ { d }$ in the leading scalar recursion replaced by $\eta _ { d } / r$ . Consequently, for a unique favorable index $i _ { * }$

$$
\tau _ { \beta _ { d } } ^ { ( 1 ) , \mathsf { U } } \leq \frac { C _ { \beta } r } { \eta _ { d } \gamma _ { d } ^ { p - 2 } } , \qquad \beta _ { d } \leq \widetilde { m } _ { i _ { * } } ( u _ { \tau _ { \beta _ { d } } ^ { ( 1 ) , \mathsf { U } } } ) \leq 2 \beta _ { d } ,\tag{51}
$$

and

$$
\operatorname* { m a x } _ { k \neq i _ { * } } | m _ { k } ( u _ { \tau _ { \beta _ { d } } ^ { ( 1 ) , \mathsf { U } } } ) | \leq C _ { \alpha } \alpha _ { d } ,
$$

with probability $1 - o ( 1 )$ . Here $\tau _ { a } ^ { ( 1 ) , \mathsf { U } } : = \operatorname* { i n f } \{ t : M _ { \sigma } ( u _ { t } ) \geq a \}$

The continuation argument rescales in the same way. Starting from the configuration in (51), define

$$
T ^ { \mathsf { U } } : = \left\lceil \frac { C _ { T } r } { \eta _ { d } \beta _ { d } ^ { p - 2 } } \right\rceil .
$$

The selected population drift in (45) is $r ^ { - 1 }$ times the corresponding drift per tied SGD step, the martingale increment is multiplied by $r ^ { - 1 / 2 }$ , and the normalization coeficient by $r ^ { - 1 }$ . The cumulative scales over $T ^ { \mathsf { U } }$ are therefore the same as, or smaller than, those in the tied continuation proof. Indeed,

$$
\frac { \eta _ { d } } { r } T ^ { \mathsf { U } } \asymp \beta _ { d } ^ { - ( p - 2 ) } , \qquad \frac { \eta _ { d } } { \sqrt { r } } \sqrt { T ^ { \mathsf { U } } \log d } \asymp \sqrt { \frac { \eta _ { d } \log d } { \beta _ { d } ^ { p - 2 } } } ,
$$

and

$$
\frac { \eta _ { d } ^ { 2 } } { r } T ^ { \mathsf { U } } \asymp \frac { \eta _ { d } } { \beta _ { d } ^ { p - 2 } } .
$$

Thus the deterministic forcing, martingale fluctuations, and normalization errors satisfy exactly the bounds used in Proposition 7.2; the selected-coordinate noise is in fact simpler because (48) is uniform in the overlap. The damping argument for the competitors is unchanged after replacing

each occurrence of $\eta _ { d }$ in the population drift by $\eta _ { d } / r$ . Consequently, the selected overlap reaches $\rho$ within an additional

$$
\frac { C _ { \rho } r } { \eta _ { d } \beta _ { d } ^ { p - 2 } } = o \left( \frac { r } { \eta _ { d } \gamma _ { d } ^ { p - 2 } } \right)
$$

steps, while all competitors remain microscopic. Therefore

$$
\tau _ { \rho } ^ { ( 1 ) , \mathsf { U } } \leq \frac { C r } { \eta _ { d } \gamma _ { d } ^ { p - 2 } }\tag{52}
$$

with probability $1 - o ( 1 )$ ).

Proof of part (U) of Theorem 5.1. For any fixed $1 \leq j \leq s ,$ the jth untied weight has the same trajectory law as the first weight analyzed above. Thus (52) gives

$$
\tau _ { \rho } ^ { j , \mathsf { U } } \leq \frac { C r } { \eta _ { d } \gamma _ { d } ^ { p - 2 } }
$$

with probability $1 - o ( 1 )$ . Since $\gamma _ { d } = \sqrt { \delta \log d / d } ,$ this is

$$
O \left( r d ^ { p - 1 } ( \log d ) ^ { A - ( p - 2 ) / 2 } \right) ,
$$

which proves the theorem.

## 8.4 Proof of weak orbit coverage

We now turn to the proof of Corollary 3.5, which describes how full weak orbit coverage is achieved by the augmented and untied students given enough weights.

Retain the assumptions and learning rates of Theorem 5.1, and fix $\rho \in ( 0 , 1 )$ . The preceding untied argument already shows that any fixed weight recovers its initial leader, by (51) and the continuation leading to (52). We now quantify its stochastic failure probability to apply this conclusion simultaneously to all weights satisfying the initialization conditions.

Recall the teacher-geometry event $\mathcal { G } _ { d }$ from Lemma A.1. Moreover, recall the initialization conditions in Lemmas A.2 and A.3: for a weight u with leader $i _ { * } = \arg \operatorname* { m a x } _ { k < r } q _ { k } ( u )$ ,

$$
\operatorname* { m a x } _ { k < r } | m _ { k } ( u ) | \leq C \gamma _ { d } , \qquad q _ { i _ { * } } ( u ) \geq c \gamma _ { d } , \qquad \operatorname* { m a x } _ { k \neq i _ { * } } q _ { k } ( u ) \leq ( 1 - \nu _ { d } ) q _ { i _ { * } } ( u ) ,\tag{53}
$$

where $\nu _ { d } = ( \log d ) ^ { - 2 }$ . Let $\mathcal { E } _ { d } ^ { j }$ denote the event that (53) holds for $u = w _ { 0 } ^ { j , \mathsf { U } }$ , and write $i _ { * } ( j )$ for its initial leader. We break ties by the smallest index; on $\mathcal { E } _ { d } ^ { j }$ the leader is unique. For a constant $C _ { U } > 0$ , set

$$
T _ { \mathsf { U } } : = \left\lceil \frac { C _ { U } r } { \eta _ { d } \gamma _ { d } ^ { p - 2 } } \right\rceil .
$$

Lemma 8.3 (Simultaneous untied recovery of initial leaders). There exists $C _ { U } = C _ { U } ( \sigma , \delta , \rho ) < \infty$ such that, for every fixed $D > 0$ and all suficiently large $d ,$ uniformly over $w ^ { \ast } \in \mathcal G _ { d }$ 7

$$
\mathbb { P } \left( \bigcup _ { j = 1 } ^ { s } \left( \mathcal { E } _ { d } ^ { j } \cap \left\{ \operatorname* { m a x } _ { 0 \leq t \leq T _ { \mathrm { { l } } } } q _ { i _ { * } ( j ) } ( w _ { t } ^ { j , \mathsf { U } } ) < \rho \right\} \right) \Bigg | w ^ { * } \right) \leq s d ^ { - D } .
$$

In particular, taking $D > \kappa$ in Assumption ${ \mathcal { B } } ,$ with conditional probability $1 - o ( 1 )$ every weight satisfying (53) recovers its initial leader by $T _ { \mathsf { U } }$

Proof. Fix a teacher in $\mathcal { G } _ { d }$ and a weight index $j ,$ , and condition further on an initialization $u _ { 0 } = w _ { 0 } ^ { j , \mathsf { U } }$ satisfying (53). By (41), its trajectory $u _ { t } = w _ { t } ^ { j , \mathsf { U } }$ follows the one-weight dynamics. The samples remain fresh under this conditioning by Assumption 2. The conditional moment estimates of Lemma 8.1 and its proof, together with the truncation argument in the proof of Lemma 7.5, give an event of conditional probability at least $1 - d ^ { - D }$ on which

$$
\frac { \eta _ { d } } { \sqrt { r } } \operatorname* { m a x } _ { \ell < r } \operatorname* { m a x } _ { 0 \le n \le T _ { \mathbb { U } } } \left| \sum _ { t = 1 } ^ { n } \xi _ { \ell , t } ^ { \mathrm { S } } \right| \le C _ { \sigma , D } \frac { \eta _ { d } } { \sqrt { r } } \left( \sqrt { T _ { \mathbb { U } } \log d } + ( \log d ) ^ { P + 1 } \right) = o ( \nu _ { d } \gamma _ { d } ) ,\tag{54}
$$

and

$$
\operatorname* { m a x } _ { 1 \leq t \leq T _ { \mathsf { U } } } \| G _ { t } ^ { \mathrm { S } } \| \leq C _ { \sigma , D } \sqrt { d } ( \log d ) ^ { P } .
$$

Indeed, choose the moment orders proportional to log d so that the truncation, concentration, and gradient estimates have combined failure probability at most

$$
C _ { \sigma , b } r ( T _ { \mathsf { U } } + 1 ) d ^ { - b } \leq d ^ { - D } ,
$$

where $b > D + p + 1$ is fixed. The last inequality holds for suficiently large $d ,$ since $T _ { \mathsf { U } } \leq d ^ { p }$ and $r \leq d .$ The estimates hold over the full horizon because the untied conditional moments are uniform over the current weight. The little-o bound in (54) follows from the same calculation as in the proof of Lemma 8.2.

Any noise sum over an interval inside $[ 0 , T _ { \mathsf { U } } ]$ is the diference of two partial sums, so its magnitude is at most twice the maximum in (54). Thus the same event also controls the noise after the microscopic stopping time. The deterministic error bounds from Lemma 8.2 and the comparisons leading to (51) and (52) now apply with $i _ { * } = i _ { * } ( j )$ . They bring this initial leader to $\rho$ within

$$
\frac { C _ { \beta } r } { \eta _ { d } \gamma _ { d } ^ { p - 2 } } + \frac { C _ { \rho } r } { \eta _ { d } \beta _ { d } ^ { p - 2 } } \le T _ { \mathsf { U } } ,
$$

for suficiently large $C _ { U }$ and $d ,$ since $\beta _ { d } / \gamma _ { d }  \infty$ . Here $\beta _ { d }$ is the microscopic threshold from Section $^ { 7 , }$ also used in Lemma 8.2. Increasing D changes the concentration constants and the dimension threshold, but neither the little-o comparisons nor the choice of $C _ { U }$

The estimates are uniform over the conditioned initialization. Integrating over $u _ { 0 }$ therefore gives, for every $j ,$

$$
\mathbb { P } \left( \mathcal { E } _ { d } ^ { j } \cap \left\{ \operatorname* { m a x } _ { 0 \leq t \leq T _ { \mathrm { U } } } q _ { i _ { * } ( j ) } ( w _ { t } ^ { j , \mathsf { U } } ) < \rho \right\} \bigg | w ^ { * } \right) \leq d ^ { - D } .
$$

A union bound over $j \le s$ proves the lemma. This step requires no independence between the trajectories, which share the sample sequence. □

Fix $\varepsilon > 0$ and suppose $s \geq ( 1 + \varepsilon ) r \log r$ . Recall the relevant correlations $q _ { k }$ from $( 7 )$ and the coverage events ${ \mathcal { C } } _ { n } ^ { \boldsymbol { \mathsf { k } } } ( \rho )$ from Definition 3.4. Use the horizon $T _ { \mathsf { U } }$ above, with $C _ { U }$ chosen according to Lemma 8.3. Choose $C _ { A }$ as in part (A) of Theorem 5.1, and set

$$
T _ { \mathsf { A } } : = \left\lceil \frac { C _ { A } } { \eta _ { d } \gamma _ { d } ^ { p - 2 } } \right\rceil .
$$

Proof of Corollary 3.5. We first prove the coverage upper bounds. According to (2) and (7), we have the shift identity

$$
q _ { k } ( \Pi ^ { a } u ) = q _ { k - a } ( u ) , \qquad 0 \leq a , k < r ,\tag{55}
$$

where indices are understood modulo r.

Untied procedure. Condition on a teacher $w ^ { \ast } \in \mathcal { G } _ { d }$ . Using the initialization events $\mathcal { E } _ { d } ^ { j }$ , write

$$
p _ { \mathrm { i n i t } } : = \mathbb { P } ( \mathcal { E } _ { d } ^ { 1 } \mid w ^ { * } ) = 1 - o ( 1 ) .
$$

The last equality follows from the joint proof of Lemmas A.2 and A.3 in Appendix A, uniformly over $w ^ { \ast } \in \mathcal { G } _ { d }$

By (55), shifting an initial weight by $\Pi ^ { a }$ preserves (53) and shifts its unique leader by a. Since uniform initialization on the sphere is invariant under these shifts, for every $j \le s$ and $k < r ,$

$$
\mathbb { P } \big ( \mathcal { E } _ { d } ^ { j } \cap \{ i _ { * } ( j ) = k \} \mid w ^ { * } \big ) = \frac { p _ { \mathrm { i n i t } } } { r } .
$$

The initializations are independent by Assumption 2. Consequently, for a fixed teacher direction $k ,$ , the probability that no initial weight satisfies (53) and has leader k is $( 1 - p _ { \mathrm { i n i t } } / r ) ^ { s }$ . Taking a union bound over $k < r ,$ the probability that at least one teacher direction has no such initial weight is at most

$$
r \left( 1 - { \frac { p _ { \mathrm { i n i t } } } { r } } \right) ^ { s } \leq r \exp \left( - { \frac { s p _ { \mathrm { i n i t } } } { r } } \right) \leq r ^ { - \varepsilon + o ( 1 ) } = o ( 1 ) .
$$

If every teacher direction has such an initial weight and all these weights recover their initial leaders, full orbit coverage holds. Choose $D > \kappa$ , so sd<sup>−</sup> $\begin{array} { r } { \mathbf { \nabla } \cdot D _ { \mathbf { \nabla } } = o ( 1 ) } \end{array}$ by Assumption 3. Combining the preceding bound on missing initial leaders with the recovery failure bound in Lemma 8.3 gives

$$
\begin{array} { r } { \mathbb { P } \left( ( \mathcal { C } _ { T _ { \mathsf { U } } } ^ { \mathsf { U } } ( \rho ) ) ^ { c } \right) \leq \mathbb { P } ( \mathcal { G } _ { d } ^ { c } ) + r ^ { - \varepsilon + o ( 1 ) } + s d ^ { - D } = o ( 1 ) , } \end{array}
$$

The three terms account, respectively, for failure of the teacher geometry event, a teacher direction with no suitable initial weight, and a recovery failure among weights satisfying (53). The first term tends to zero by Lemma A.1.

Augmented procedure. Here we condition on the teacher and the samples through the recovery horizon. Define

$$
\begin{array} { r } { \mathcal { D } _ { \mathsf { A } } : = \sigma ( w ^ { * } , ( x _ { t } , y _ { t } ) _ { 1 \leq t \leq T _ { \mathsf { A } } } ) . } \end{array}
$$

Under this conditioning, the initial weights remain independent and uniform by Assumption 2. The reduction (42) therefore makes their trajectories conditionally independent and identically distributed.

For every fixed sample, the tied loss from Definition 2.2 satisfies

$$
\begin{array} { r } { L ^ { \mathsf { T } } ( \Pi ^ { a } u ; x , y ) = L ^ { \mathsf { T } } ( u ; x , y ) , \qquad \nabla L ^ { \mathsf { T } } ( \Pi ^ { a } u ; x , y ) = \Pi ^ { a } \nabla L ^ { \mathsf { T } } ( u ; x , y ) . } \end{array}
$$

The normalization in (5) also commutes with $\Pi ^ { a }$ . Hence, by (42), shifting an initial weight shifts its entire augmented trajectory by the same amount. Together with (55), this implies that the conditional probability of recovering direction k by time $T _ { \mathsf { A } }$ is the same for every k.

Using the orbit recovery time defined in (9), let

$$
p _ { \mathrm { r e c } } : = \mathbb { P } \left( \tau _ { \rho } ^ { 1 , \mathsf { A } } \leq T _ { \mathsf { A } } \Big | \mathscr { D } _ { \mathsf { A } } \right) .
$$

By (3) and (7), $M _ { \sigma } ( u ) = \operatorname* { m a x } _ { k < r } q _ { k } ( u )$ . The event defining $p _ { \mathrm { r e c } }$ is therefore the union of the r direction-recovery events. Since these events have equal conditional probabilities, a union bound gives

$$
\mathbb { P } \left( \operatorname* { m a x } _ { 0 \le t \le T _ { \mathsf { A } } } q _ { k } ( w _ { t } ^ { 1 , \mathsf { A } } ) \ge \rho \bigg | \mathcal { D } _ { \mathsf { A } } \right) \ge \frac { p _ { \mathrm { r e c } } } { r } , \qquad 0 \le k < r .
$$

This bound allows a single weight to recover more than one direction.

For any fixed direction, conditional independence across weights bounds the probability that all s trajectories miss it by $( 1 - p _ { \mathrm { r e c } } / r ) ^ { s }$ . A union bound over directions consequently gives

$$
\mathbb { P } \left( \left( \mathcal { C } _ { T _ { \mathsf { A } } } ^ { \mathsf { A } } ( \rho ) \right) ^ { c } \Big | \mathcal { D } _ { \mathsf { A } } \right) \leq r \left( 1 - \frac { p _ { \mathrm { r e c } } } { r } \right) ^ { s } \leq r \exp \left( - \frac { s p _ { \mathrm { r e c } } } { r } \right) .
$$

We now remove the conditioning. Part (A) of Theorem 5.1 and the tower property imply

$$
\mathbb { E } [ 1 - p _ { \mathrm { r e c } } ] = \mathbb { P } \left( \tau _ { \rho } ^ { 1 , \mathsf { A } } > T _ { \mathsf { A } } \right) = o ( 1 ) .
$$

Fix any constant $\theta \in ( ( 1 + \varepsilon ) ^ { - 1 } , 1 )$ . On $\{ p _ { \mathrm { r e c } } \geq \theta \}$ , the conditional coverage failure probability is at most $\bar { r ^ { 1 - ( 1 + \varepsilon ) \theta } }$ . On the complementary event, Markov’s inequality gives

$$
\mathbb { P } ( p _ { \mathrm { r e c } } < \theta ) \leq \frac { \mathbb { E } [ 1 - p _ { \mathrm { r e c } } ] } { 1 - \theta } .
$$

Averaging the conditional coverage estimate thus yields

$$
\mathbb { P } \left( \left( { \mathcal { C } } _ { T _ { \mathtt { A } } } ^ { \mathtt { A } } ( \rho ) \right) ^ { c } \right) \leq { \frac { \mathbb { E } [ 1 - p _ { \mathrm { r e c } } ] } { 1 - \theta } } + r ^ { 1 - ( 1 + \varepsilon ) \theta } = o ( 1 ) .
$$

Sample complexity and matching lower bounds. The preceding estimates prove coverage by $T _ { \mathsf { U } }$ and $T _ { \mathsf { A } }$ for the respective procedures. Diferent directions may be recovered at diferent times, as allowed by Definition 3.4. Each iteration of (5) uses one fresh observation. Using $\eta _ { d } = d ^ { - p / 2 } ( \log d ) ^ { - A }$ from Theorem 5.1 and $\gamma _ { d } = \sqrt { \log r / d }$ from Section A, the horizons satisfy

$$
T _ { \mathsf { A } } = \widetilde O ( d ^ { p - 1 } ) , \qquad T _ { \mathsf { U } } = \widetilde O ( r d ^ { p - 1 } ) .
$$

For the lower bounds, note that by Definitions 3.1 and 3.4, we have

$$
{ \mathcal { C } } _ { n } ^ { \mathrm { k } } ( \rho ) \subseteq { \mathcal { R } } _ { n } ^ { \mathrm { k } } ( \rho ) , \qquad \mathrm { k } \in \{ \mathrm { U } , \mathrm { A } \} .
$$

Theorem 5.4 therefore supplies the matching lower bounds, giving

$$
n _ { \mathsf { A } } ^ { \mathrm { c o v } } ( \rho ) = \widetilde { \Theta } ( d ^ { p - 1 } ) , \qquad n _ { \mathsf { U } } ^ { \mathrm { c o v } } ( \rho ) = \widetilde { \Theta } ( r d ^ { p - 1 } ) ,
$$

which proves the corollary.

## 9 Quadratic link and weak subspace recovery

In this section, we prove Theorem 5.3 with $\sigma = h _ { 2 }$ . We retain the assumptions and recovery times from Section 5, with the learning-rate scaling in (6). We first analyze the tied trajectory, using the notation of Section 6, and then treat the augmented and untied procedures using the reductions from Section 8.

## 9.1 Proof strategy

For $p \geq 3$ , the microscopic argument in Section 7 amplifies the initial gap between the leading correlation and its competitors. For the pure quadratic link, the leading coordinate drift is linear, and we instead study the growth of the total mass in the teacher orbit subspace.

Recall that w<sup>∥</sup> denotes the orthogonal projection of w onto $S ^ { * }$ . We track its squared norm through the total overlap

$$
R ( w ) : = \sum _ { k = 0 } ^ { r - 1 } m _ { k } ( w ) ^ { 2 } , \qquad R _ { t } : = R ( w _ { t } ) .
$$

Subsection 9.2 shows that, on the teacher-geometry event $\mathcal { G } _ { d }$ defined in Lemma $\mathrm { A . 1 }$

$$
R ( w ) = ( 1 + o ( 1 ) ) \| w ^ { \| } \| ^ { 2 }
$$

uniformly over unit weights, and $R _ { 0 } \times r / d$ with probability $1 - o ( 1 )$ . Thus weak subspace recovery follows once $R _ { t }$ reaches a macroscopic level.

The identity $\Phi = - R$ gives a direct description of the population growth: the population contribution to the one-step increase of $R _ { t }$ is $\eta _ { d } \lVert \nabla \Phi ( w _ { t - 1 } ) \rVert ^ { 2 }$ . Below any fixed threshold $a < 1$ , this quantity is comparable to $\eta _ { d } R _ { t - 1 }$ on $\mathcal { G } _ { d }$ . Subsection 9.3 establishes this comparison and the exact stochastic recursion.

To turn this proportional drift into a growth estimate, we study log $R _ { t }$ . Its increments have a positive drift of order $\eta _ { d }$ , together with relative noise and discretization errors. The relative noise involves division by $R _ { t - 1 }$ , so we first stop the process when $R _ { t }$ either reaches $a$ or falls below a fixed fraction of $R _ { 0 }$ . Before this exit, the denominator remains bounded below at the initialization scale. Using the moment estimates from Subsection 9.4, Subsection 9.5 shows that the accumulated relative noise, logarithmic remainders, and spherical normalization errors are $o ( 1 )$ over a horizon of order $\eta _ { d } ^ { - 1 } \log ( d / r )$ . The resulting lower bound on $\log ( R _ { t } / R _ { 0 } )$ rules out the lower exit and forces $R _ { t }$ to reach a within this horizon. Choosing $a \in ( \rho , 1 )$ then yields weak subspace recovery.

Finally, Subsection 9.6 treats the other procedures. Under the prescribed learning rates, each augmented weight follows the tied update exactly. For an untied weight, the drift carries a factor $\eta _ { d } / r$ and the centered stochastic increment a factor $\eta _ { d } / \sqrt { r }$ . The same comparison therefore applies over a horizon longer by a factor $^ { r , }$ with the accumulated errors still negligible.

## 9.2 Total overlap and initialization

We work on the teacher-geometry event $\mathcal { G } _ { d }$ . Recall the teacher Gram matrix $W ^ { * }$ from the proof of Lemma A.3 in Appendix A. On this event,

$$
\| W ^ { * } - \operatorname { I } _ { r } \| _ { \mathrm { o p } } \leq r \mu _ { * } = o ( 1 ) .
$$

To relate total overlap to projection mass, define

$$
B ^ { * } : = \sum _ { k = 0 } ^ { r - 1 } ( \Pi ^ { k } w ^ { * } ) ( \Pi ^ { k } w ^ { * } ) ^ { \top } .
$$

This matrix vanishes on $( S ^ { * } ) ^ { \perp }$ , and

$$
R ( w ) = \langle w , B ^ { * } w \rangle = \langle w ^ { \parallel } , B ^ { * } w ^ { \parallel } \rangle .
$$

For suficiently large $d , W ^ { * }$ is positive definite. The restriction of $B ^ { * }$ to $S ^ { * }$ has the same eigenvalues as $W ^ { * }$ , all lying in $[ 1 - r \mu _ { * } , 1 + r \mu _ { * } ]$ . Consequently, uniformly over $w \in \mathbb { S } ^ { d - 1 }$ ，

$$
( 1 - r \mu _ { * } ) \| w ^ { \| } \| ^ { 2 } \leq R ( w ) \leq ( 1 + r \mu _ { * } ) \| w ^ { \| } \| ^ { 2 } .\tag{56}
$$

Thus it sufices to control the total overlap. We next establish its initial scale.

Lemma 9.1 (Initial total overlap). Condition on a teacher $w ^ { \ast } \in \mathcal { G } _ { d } ,$ , and let $w _ { 0 }$ be uniform on $\mathbb { S } ^ { d - 1 }$ , independently of the teacher. There exist constants $c , C > 0$ such that, for all suficiently large d and every $\varepsilon \in ( 0 , 1 )$

$$
\mathbb { P } \left( \left| R _ { 0 } - \frac { r } { d } \right| > \varepsilon \frac { r } { d } \middle | w ^ { * } \right) \leq C \exp ( - c \varepsilon ^ { 2 } r ) .
$$

The constants are uniform over $w ^ { \ast } \in \mathcal G _ { d }$ . In particular, $R _ { 0 } \times r / d$ with conditional probability $1 - o ( 1 )$

Proof. Write $w _ { 0 } = h / \lVert h \rVert$ , where $h \sim \mathcal { N } ( 0 , \mathrm { I } _ { d } )$ is independent of the teacher. Then

$$
R _ { 0 } = \frac { h ^ { \top } B ^ { * } h } { \| h \| ^ { 2 } } .
$$

The matrix $B ^ { * }$ is positive semidefinite and satisfies

$$
\begin{array} { r } { \operatorname { T r } ( B ^ { * } ) = r , \qquad \| B ^ { * } \| _ { \mathrm { o p } } = \| W ^ { * } \| _ { \mathrm { o p } } \leq 1 + r \mu _ { * } \leq C , } \end{array}
$$

and hence

$$
\| B ^ { * } \| _ { \mathrm { F } } ^ { 2 } = \mathrm { T r } ( ( B ^ { * } ) ^ { 2 } ) \leq \| B ^ { * } \| _ { \mathrm { o p } } \mathrm { T r } ( B ^ { * } ) \leq C r .
$$

For $\theta \in ( 0 , 1 )$ , apply the Hanson–Wright inequality (77) to $B ^ { * }$ and $\operatorname { I } _ { d } . \mathrm { ~ A ~ }$ union bound gives

$$
\begin{array} { r l } & { \mathbb { P } \Big ( | h ^ { \top } B ^ { * } h - r | > \theta r ~ \mathrm { o r } ~ | \| h \| ^ { 2 } - d | > \theta d ~ \Big | ~ w ^ { * } \Big ) } \\ & { \qquad \leq C e ^ { - c \theta ^ { 2 } r } + C e ^ { - c \theta ^ { 2 } d } \leq C e ^ { - c \theta ^ { 2 } r } , } \end{array}
$$

where we used $r \leq d .$ . Outside this event,

$$
\frac { 1 - \theta } { 1 + \theta } \frac { r } { d } \leq R _ { 0 } \leq \frac { 1 + \theta } { 1 - \theta } \frac { r } { d } .
$$

Taking θ to be a suficiently small universal multiple of ε proves the concentration bound. The final assertion follows by fixing $\varepsilon \in ( 0 , 1 )$ and using $r = d ^ { \delta } \to \infty$ □

Combining Lemma 9.1, (56), and $\mathbb { P } ( \mathcal { G } _ { d } ) = 1 - o ( 1 )$ yields

$$
\| w _ { 0 } ^ { \parallel } \| ^ { 2 } \asymp R _ { 0 } \asymp \frac { r } { d }
$$

with probability $1 - o ( 1 )$ . The same initialization estimate applies to any fixed weight of the untied or augmented student.

## 9.3 Recursion and population growth

Recall that $R ( w ) = \langle w , B ^ { * } w \rangle$ and $\Phi = - R$ . The spherical gradient calculation in Lemma 6.1, specialized to $\phi ( z ) = z ^ { 2 }$ , gives

$$
\nabla \Phi ( w ) = - 2 \big ( B ^ { * } w - R ( w ) w \big ) .
$$

This identity allows us to express the population contribution to the growth of R as the squared norm of the spherical gradient.

Using the coordinate increments from (14), define

$$
\Xi _ { t } : = \sum _ { k = 0 } ^ { r - 1 } m _ { k } ( w _ { t - 1 } ) \xi _ { k , t } = - \langle B ^ { * } w _ { t - 1 } , \nabla H _ { t } ( w _ { t - 1 } ) \rangle .
$$

The coeficients $m _ { k } ( w _ { t - 1 } )$ are $\mathcal { F } _ { t - 1 }$ -measurable, so

$$
\mathbb { E } [ \Xi _ { t } \mid { \mathcal { F } } _ { t - 1 } ] = 0 .
$$

Lemma 9.2 (Exact recursion for the total overlap). For every $t \geq 1$

$$
( 1 + \eta _ { d } ^ { 2 } \| G _ { t } \| ^ { 2 } ) R _ { t } = R _ { t - 1 } + \eta _ { d } \| \nabla \Phi ( w _ { t - 1 } ) \| ^ { 2 } + 2 \eta _ { d } \Xi _ { t } + \eta _ { d } ^ { 2 } \langle G _ { t } , B ^ { * } G _ { t } \rangle .\tag{57}
$$

Proof. Since $G _ { t } \perp w _ { t - 1 }$ , the spherical update (5) yields

$$
( 1 + \eta _ { d } ^ { 2 } \| G _ { t } \| ^ { 2 } ) R _ { t } = R _ { t - 1 } - 2 \eta _ { d } \langle B ^ { * } w _ { t - 1 } , G _ { t } \rangle + \eta _ { d } ^ { 2 } \langle G _ { t } , B ^ { * } G _ { t } \rangle .
$$

Using $G _ { t } = \nabla \Phi ( w _ { t - 1 } ) + \nabla H _ { t } ( w _ { t - 1 } )$ and the spherical gradient formula above,

$$
\begin{array} { r l } & { - 2 \langle B ^ { * } w _ { t - 1 } , G _ { t } \rangle = \langle \nabla \Phi ( w _ { t - 1 } ) , G _ { t } \rangle } \\ & { \qquad = \| \nabla \Phi ( w _ { t - 1 } ) \| ^ { 2 } + 2 \Xi _ { t } . } \end{array}
$$

Substitution proves the identity.

The population contribution is therefore nonnegative. Teacher geometry gives the quantitative lower bound needed to amplify a small total overlap.

Proposition 9.3 (Population growth of the total overlap). On the teacher-geometry event $\mathcal { G } _ { d }$ for every fixed $a \in ( 0 , 1 )$ there exist constants $c _ { a } , C _ { a } > 0$ , depending only on a, such that, for all suficiently large d, uniformly over $w \in \mathbb { S } ^ { d - 1 }$ with $0 < R ( w ) \leq a .$

$$
c _ { a } R ( w ) \leq \| \nabla \Phi ( w ) \| ^ { 2 } \leq C _ { a } R ( w ) .
$$

Proof. Since $\langle w , B ^ { * } w \rangle = R ( w )$ and $\| w \| = 1$ ，

$$
\begin{array} { r } { \| \nabla \Phi ( w ) \| ^ { 2 } = 4 \big ( \| B ^ { * } w \| ^ { 2 } - R ( w ) ^ { 2 } \big ) . } \end{array}\tag{58}
$$

By the spectral comparison established in Subsection 9.2, the nonzero eigenvalues of $B ^ { * }$ belong to $[ 1 - r \mu _ { * } , 1 + r \mu _ { * } ]$ . Hence,

$$
( 1 - r \mu _ { * } ) R ( w ) \leq \| B ^ { * } w \| ^ { 2 } \leq ( 1 + r \mu _ { * } ) R ( w ) .
$$

Combining these bounds with (58) gives

$$
4 R ( w ) ( 1 - R ( w ) - r \mu _ { * } ) \leq \| \nabla \Phi ( w ) \| ^ { 2 } \leq 4 R ( w ) ( 1 - R ( w ) + r \mu _ { * } ) .
$$

On $\mathcal { G } _ { d } .$ , we have $r \mu _ { * } = o ( 1 )$ uniformly over the teacher. Since $R ( w ) \leq a < 1$ , the claimed bounds follow for suficiently large d. □

The matrix $B ^ { * }$ is positive semidefinite, so the last term in (57) is nonnegative. Discarding it and applying Proposition 9.3, we obtain, on $\mathcal { G } _ { d }$ and for suficiently large $d ,$

$$
( 1 + \eta _ { d } ^ { 2 } \| G _ { t } \| ^ { 2 } ) R _ { t } \geq ( 1 + c _ { a } \eta _ { d } ) R _ { t - 1 } + 2 \eta _ { d } \Xi _ { t }\tag{59}
$$

whenever $0 < R _ { t - 1 } < a$ . The next subsection controls the centered increment $\Xi _ { t }$ and the normalization term $\eta _ { d } ^ { 2 } \lVert G _ { t } \rVert ^ { 2 }$

## 9.4 Noise and gradient moments

We work on the teacher-geometry event $\mathcal { G } _ { d }$ . The coordinate and gradient estimates from Sections 6 and 8, specialized to $p = P = 2$ , give the stochastic bounds needed for (59).

For the untied procedure, recall the single-weight trajectory $u _ { t } = w _ { t } ^ { 1 , \mathsf { U } }$ and the quantities $G _ { t } ^ { \mathrm { S } }$ $\xi _ { k , t } ^ { \mathrm { S } } .$ , and $\mathcal { F } _ { t } ^ { \mathrm { S } }$ from Section 8. Define

$$
\Xi _ { t } ^ { \mathrm { { S } } } : = \sum _ { k = 0 } ^ { r - 1 } m _ { k } ( u _ { t - 1 } ) \xi _ { k , t } ^ { \mathrm { { S } } } .
$$

As for the tied process, $\mathbb { E } [ \Xi _ { t } ^ { \mathrm { S } } \mid \mathcal { F } _ { t - 1 } ^ { \mathrm { S } } ] = 0$

Corollary 9.4 (Noise and gradient moments). There exists a constant $C > 0$ such that, for all suficiently large d, every $t \geq 1$ , and every $\nu \geq 2$

$$
\mathbb { E } [ \Xi _ { t } ^ { 2 } \mid \mathcal { F } _ { t - 1 } ] \le C r R _ { t - 1 } ,\tag{60}
$$

$$
( \mathbb { E } [ | \Xi _ { t } | ^ { \nu } | \mathcal { F } _ { t - 1 } ] ) ^ { 1 / \nu } \leq C \nu ^ { 2 } \sqrt { r R _ { t - 1 } } ,\tag{61}
$$

$$
( \mathbb { E } [ \| G _ { t } \| ^ { \nu } | \mathcal { F } _ { t - 1 } ] ) ^ { 1 / \nu } \leq C \nu ^ { 2 } \sqrt { d } .\tag{62}
$$

The estimates are uniform over the current weight and over teachers in $\mathcal { G } _ { d }$ . They also hold for the untied single-weight process, with

$$
\begin{array} { r l } { ( G _ { t } , \Xi _ { t } , R _ { t - 1 } , \mathcal { F } _ { t - 1 } ) } & { r e p l a c e d \ b y \quad ( G _ { t } ^ { \mathrm { S } } , \Xi _ { t } ^ { \mathrm { S } } , R ( u _ { t - 1 } ) , \mathcal { F } _ { t - 1 } ^ { \mathrm { S } } ) . } \end{array}
$$

Proof. By (56),

$$
R ( w ) \leq 1 + r \mu _ { * } \leq C
$$

uniformly over unit weights on $\mathcal { G } _ { d }$ , for suficiently large d.

Condition on $\mathcal { F } _ { t - 1 }$ . Conditional Minkowski’s inequality and Lemma 6.4, with $P = 2$ , give

$$
\begin{array} { r l } {  { \big ( \mathbb { E } [ | \Xi _ { t } | ^ { \nu } \ | \ \mathcal { F } _ { t - 1 } ] \big ) ^ { 1 / \nu } \leq C \nu ^ { 2 } \sum _ { k = 0 } ^ { r - 1 } | m _ { k } ( w _ { t - 1 } ) | \big ( 1 + \sqrt { r } \big | m _ { k } ( w _ { t - 1 } ) \big | \big ) } \quad } & { } \\ & { \leq C \nu ^ { 2 } \big ( \sqrt { r R _ { t - 1 } } + \sqrt { r } R _ { t - 1 } \big ) } \\ & { \leq C \nu ^ { 2 } \sqrt { r R _ { t - 1 } } . } \end{array}
$$

Here the second inequality uses Cauchy–Schwarz, and the last uses the uniform bound on R. This proves (61); taking $\nu = 2$ also gives (60). The gradient estimate (62) is the conditional moment bound (99) established in the proof of Lemma 6.5 in Appendix B.3, specialized to $P = 2$

For the untied process, Lemma 8.1 gives

$$
\left( \mathbb { E } [ | \xi _ { k , t } ^ { \mathrm { S } } | ^ { \nu } | \mathcal { F } _ { t - 1 } ^ { \mathrm { S } } ] \right) ^ { 1 / \nu } \leq C \nu ^ { 2 } .
$$

Consequently, the same conditional Minkowski and Cauchy–Schwarz inequalities yield

$$
\left( \mathbb { E } [ | \Xi _ { t } ^ { \mathrm { S } } | ^ { \nu } \ | \ \mathcal { F } _ { t - 1 } ^ { \mathrm { S } } ] \right) ^ { 1 / \nu } \leq C \nu ^ { 2 } \sum _ { k = 0 } ^ { r - 1 } | m _ { k } ( u _ { t - 1 } ) | \leq C \nu ^ { 2 } \sqrt { r R ( u _ { t - 1 } ) } .
$$

Taking $\nu = 2$ gives the corresponding variance bound. The gradient-moment calculation in the proof of Lemma 8.1, with $P = 2$ , gives the asserted estimate for $G _ { t } ^ { \mathrm { S } }$ . All these bounds are uniform over the current weight and the conditioned teacher. □

We record two consequences for the growth argument. First, for every fixed $c _ { 0 } > 0$ , whenever $R _ { t - 1 } \geq c _ { 0 } r / d .$

$$
( \mathbb { E } [ | \frac { \Xi _ { t } } { R _ { t - 1 } } | ^ { \nu } | \mathcal { F } _ { t - 1 } ] ) ^ { 1 / \nu } \leq C _ { c _ { 0 } } \nu ^ { 2 } \sqrt { d } , \qquad \nu \geq 2 .\tag{63}
$$

Indeed, $R _ { t - 1 }$ is measurable with respect to the conditioning, and $\sqrt { r / R _ { t - 1 } } \le \sqrt { d / c _ { 0 } }$ . This estimate controls the relative fluctuations of the total overlap.

Second, for every fixed $B , D > 0$ , there exists $C _ { B , D } < \infty$ such that, for every deterministic integer horizon $1 \leq T \leq d ^ { B }$

$$
\mathbb { P } \left( \operatorname* { m a x } _ { 1 \leq t \leq T } \| G _ { t } \| ^ { 2 } > C _ { B , D } d ( \log d ) ^ { 4 } \bigg | w ^ { * } , w _ { 0 } \right) \leq d ^ { - D } .\tag{64}
$$

This holds for suficiently large $d ,$ uniformly over $w ^ { \ast } \in \mathcal G _ { d }$ and $w _ { 0 } \in \mathbb { S } ^ { d - 1 }$ . To see this, take ν to be a suficiently large multiple of log d in (62). Conditional Markov’s inequality makes the failure probability at each step at most $d ^ { - \bar { B } - D - 1 }$ , and a union bound over $t \leq T$ proves the claim.

Both (63) and (64) hold for the untied process with the substitutions in Corollary 9.4, conditioning on $w ^ { * } , u _ { 0 }$ in the latter bound.

## 9.5 Growth and proof of the tied upper bound

We combine the preceding estimates to prove exponential growth of the total overlap until it reaches a fixed threshold. Fix $a , \theta , c _ { \mathrm { i n i t } } \in ( 0 , 1 )$ and $C _ { 0 } > 0$ , and define

$$
\tau _ { a } ^ { R } : = \operatorname* { i n f } \{ t \geq 0 : R _ { t } \geq a \} , \quad \quad T _ { 2 } : = \left\lceil \frac { C _ { 0 } \log ( d / r ) } { \eta _ { d } } \right\rceil .
$$

We first condition on a teacher $w ^ { \ast } \in \mathcal { G } _ { d }$ and an initialization satisfying

$$
R _ { 0 } \geq c _ { \mathrm { i n i t } } { \frac { r } { d } } .\tag{65}
$$

By Lemma 9.1, a uniform independent initialization satisfies this condition with conditional probability at least $1 - C e ^ { - c r }$ , uniformly over $w ^ { \ast } \in \mathcal { G } _ { d }$

Proposition 9.5 (Growth of the total overlap). There exist a universal constant $A _ { 0 } > 0$ and a constant $c _ { a } > 0$ , depending only on $^ { a , }$ such that, for every fixed $A > A _ { 0 }$ and $D > 0$ , the following holds for suficiently large d.

For every fixed teacher $w ^ { \ast } \in \mathcal { G } _ { d }$ and initialization satisfying (65), with probability at least $1 - d ^ { - D }$ over the SGD samples, simultaneously for every integer $0 \leq n \leq T _ { 2 }$

$$
R _ { n \wedge \tau _ { a } ^ { R } } \geq \theta R _ { 0 } \exp \bigl ( c _ { a } \eta _ { d } ( n \wedge \tau _ { a } ^ { R } ) \bigr ) .\tag{66}
$$

To apply the relative-noise estimate, fix $\beta \in ( 0 , \theta )$ and introduce

$$
\begin{array} { r } { \widehat { \tau } _ { a } ^ { R } : = \operatorname* { i n f } \{ t \geq 0 : R _ { t } \leq \beta R _ { 0 } \mathrm { ~ o r ~ } R _ { t } \geq a \} . } \end{array}
$$

For $1 \leq t \leq \widehat { \tau } _ { a } ^ { R }$

$$
R _ { t - 1 } > \beta R _ { 0 } \geq \beta c _ { \mathrm { i n i t } } { \frac { r } { d } } ,
$$

so (63) applies. We will show that the lower boundary cannot be reached before $T _ { 2 }$

Lemma 9.6 (Stopped relative-noise bounds). Fix a teacher $w ^ { \ast } \in \mathcal G _ { d }$ and an initialization satisfying (65). For every fixed $D > 0$ , there exists $C _ { D } < \infty$ such that, for suficiently large $d ,$ with conditional probability at least $1 - d ^ { - D }$ over the SGD samples,

$$
\operatorname* { m a x } _ { 0 \le n \le T _ { 2 } } \left| 2 \eta _ { d } \sum _ { t = 1 } ^ { n \wedge \widehat { \tau } _ { a } ^ { R } } \frac { \Xi _ { t } } { R _ { t - 1 } } \right| \le C _ { D } \eta _ { d } \sqrt { d T _ { 2 } \log d } ,
$$

and, simultaneously,

$$
\mathopen { } \mathclose \bgroup \left| \frac { \Xi _ { t } } { R _ { t - 1 } } \aftergroup \egroup \right| \leq C _ { D } \sqrt { d } ( \log d ) ^ { 2 } , \qquad 1 \leq t \leq \widehat { \tau } _ { a } ^ { R } \wedge T _ { 2 } .
$$

The constant may depend on the fixed parameters, but is uniform over the conditioned teachers and initializations.

Proof. Fix the teacher and initialization as in the statement. All probabilities below are conditional on this pair.

Step 1: Stopping and conditional moments.

For $1 \leq t \leq T _ { 2 }$ , define

$$
Y _ { t } : = \left\{ \begin{array} { l l } { \Xi _ { t } } & { t \leq \widehat { \tau } _ { a } ^ { R } , } \\ { \sqrt { d } R _ { t - 1 } } & { \tau > \widehat { \tau } _ { a } ^ { R } . } \\ { 0 , } & { t > \widehat { \tau } _ { a } ^ { R } . } \end{array} \right.
$$

The event $\{ t \leq \widehat \tau _ { a } ^ { R } \}$ belongs to $\mathcal { F } _ { t - 1 }$ , and on this event,

$$
R _ { t - 1 } > \beta R _ { 0 } \geq \beta c _ { \mathrm { i n i t } } \frac { r } { d } .
$$

Thus the denominator is positive and $\mathbb { E } [ Y _ { t } \mid \mathcal { F } _ { t - 1 } ] = 0$ . By (63), there exists $K < \infty$ , uniform over the conditioned teacher and initialization, such that

$$
\mathbb { E } [ Y _ { t } ^ { 2 } \mid { \mathcal F } _ { t - 1 } ] \le K , \qquad ( \mathbb { E } [ | Y _ { t } | ^ { \nu } \mid { \mathcal F } _ { t - 1 } ] ) ^ { 1 / \nu } \le K \nu ^ { 2 } , \qquad \nu \ge 2 .
$$

Step 2: Truncation and recentering.

Fix $b > D + 2$ and set

$$
q _ { d } : = \lceil b \log d \rceil , \qquad L _ { d } : = e K q _ { d } ^ { 2 } .
$$

Conditional Markov’s inequality gives

$$
\mathbb { P } \big ( | Y _ { t } | > L _ { d } | \mathcal { F } _ { t - 1 } \big ) \le \frac { \mathbb { E } [ | Y _ { t } | ^ { q _ { d } } | \mathcal { F } _ { t - 1 } ] } { L _ { d } ^ { q _ { d } } } \le e ^ { - q _ { d } } .
$$

Since $T _ { 2 } \asymp d ( \log d ) ^ { A + 1 } \leq d ^ { 2 }$ for suficiently large $d ,$ a union bound yields

$$
\mathbb { P } \left( \operatorname* { m a x } _ { 1 \leq t \leq T _ { 2 } } | Y _ { t } | > L _ { d } \right) \leq T _ { 2 } e ^ { - q _ { d } } .
$$

Define the truncated and recentered increments

$$
\widehat { Y } _ { t } : = Y _ { t } \mathbf { 1 } _ { \{ | Y _ { t } | \leq L _ { d } \} } - \mathbb { E } \big [ Y _ { t } \mathbf { 1 } _ { \{ | Y _ { t } | \leq L _ { d } \} } \mid \mathcal { F } _ { t - 1 } \big ] .
$$

They are martingale diferences and satisfy $| \widehat { Y } _ { t } | \leq 2 L _ { d }$ . To control the centering correction, use $\mathbb { E } [ Y _ { t } \mid \mathcal { F } _ { t - 1 } ] = 0$ and the same moment bound:

$$
\begin{array} { r l } & { \left| { \mathbb E } \big [ Y _ { t } \mathbf { 1 } _ { \{ | Y _ { t } | \leq L _ { d } \} } \mid { \mathcal F } _ { t - 1 } \big ] \right| = \left| { \mathbb E } \big [ Y _ { t } \mathbf { 1 } _ { \{ | Y _ { t } | > L _ { d } \} } \mid { \mathcal F } _ { t - 1 } \big ] \right| } \\ & { ~ \leq \frac { { \mathbb E } \left[ \mid Y _ { t } \mid ^ { q _ { d } } \mid { \mathcal F } _ { t - 1 } \right] } { L _ { d } ^ { q _ { d } - 1 } } } \\ & { ~ \leq L _ { d } e ^ { - q _ { d } } . } \end{array}
$$

Consequently, on the event that no increment is truncated,

$$
\operatorname* { m a x } _ { 0 \leq n \leq T _ { 2 } } \left. \sum _ { t = 1 } ^ { n } Y _ { t } \right. \leq \operatorname* { m a x } _ { 0 \leq n \leq T _ { 2 } } \left. \sum _ { t = 1 } ^ { n } \widehat { Y } _ { t } \right. + T _ { 2 } L _ { d } e ^ { - q _ { d } } .
$$

Step 3: Maximal concentration.

Centering decreases the second moment, so

$$
\mathbb { E } [ \widehat { Y } _ { t } ^ { 2 } \mid \mathcal { F } _ { t - 1 } ] \leq \mathbb { E } \big [ Y _ { t } ^ { 2 } \mathbf { 1 } _ { \{ | Y _ { t } | \leq L _ { d } \} } \mid \mathcal { F } _ { t - 1 } \big ] \leq K .
$$

The predictable quadratic variation is therefore at most $K T _ { 2 }$ . Applying Freedman’s maximal inequality (Freedman, 1975) to both signs of the martingale gives, for every $z > 0$ 2

$$
\mathbb { P } \left( \operatorname* { m a x } _ { 0 \le n \le T _ { 2 } } \left| \sum _ { t = 1 } ^ { n } \widehat { Y } _ { t } \right| \ge z \right) \le 2 \exp \left( - \frac { z ^ { 2 } } { 2 ( K T _ { 2 } + \frac { 2 } { 3 } L _ { d } z ) } \right) .
$$

Take

$$
z _ { d } : = \sqrt { 2 K T _ { 2 } q _ { d } } + \frac { 4 } { 3 } L _ { d } q _ { d } .
$$

This choice satisfies

$$
z _ { d } ^ { 2 } \ge 2 q _ { d } \biggl ( K T _ { 2 } + \frac { 2 } { 3 } L _ { d } z _ { d } \biggr ) ,
$$

so the preceding failure probability is at most $2 e ^ { - q _ { d } }$

Combining this event with the absence of truncation, with probability at least

$$
1 - ( T _ { 2 } + 2 ) e ^ { - q _ { d } } \geq 1 - d ^ { - D }
$$

for suficiently large $d ,$ we have simultaneously

$$
\operatorname* { m a x } _ { 0 \leq n \leq T _ { 2 } } \left| \sum _ { t = 1 } ^ { n } Y _ { t } \right| \leq z _ { d } + T _ { 2 } L _ { d } e ^ { - q _ { d } } , \qquad \operatorname* { m a x } _ { 1 \leq t \leq T _ { 2 } } | Y _ { t } | \leq L _ { d } .
$$

Here

$$
z _ { d } \leq C _ { D } \big ( \sqrt { T _ { 2 } \log d } + ( \log d ) ^ { 3 } \big ) , \qquad T _ { 2 } L _ { d } e ^ { - q _ { d } } \leq C _ { D } ( \log d ) ^ { 2 } d ^ { 2 - b } = o ( 1 ) .
$$

Since $( \log d ) ^ { 3 } = o ( { \sqrt { T _ { 2 } \log d } } )$ , multiplying the partial-sum bound by $2 \eta _ { d } \sqrt { d }$ proves the first assertion. The second follows from $| \Xi _ { t } / R _ { t - 1 } | = \sqrt { d } | Y _ { t } |$ before and including the stopped update, and $L _ { d } \leq$ $C _ { D } ( \log d ) ^ { 2 }$ □

Proof of Proposition 9.5. Fix a teacher $w ^ { \ast } \in \mathcal G _ { d }$ and an initialization satisfying (65). All probabilities below are conditional on this pair. If $R _ { 0 } \geq a .$ , then $\tau _ { a } ^ { R } = 0$ and the conclusion is immediate. Assume henceforth that $R _ { 0 } < a$

Step 1: Control of the stochastic terms.

Fix $D ^ { \prime } > D$ . Apply Lemma 9.6 and (64), both with failure exponent $D ^ { \prime }$ . With probability at least $1 - 2 d ^ { - D ^ { \prime } }$ , the following bounds hold simultaneously:

$$
\operatorname* { m a x } _ { 0 \le n \le T _ { 2 } } \left| 2 \eta _ { d } \sum _ { t = 1 } ^ { n \wedge \widehat { \tau } _ { a } ^ { R } } \frac { \Xi _ { t } } { R _ { t - 1 } } \right| \le C \eta _ { d } \sqrt { d T _ { 2 } \log d } ,\tag{67}
$$

$$
\operatorname* { m a x } _ { 1 \leq t \leq \widehat { \tau } _ { a } ^ { R } \wedge T _ { 2 } } \left. \frac { \Xi _ { t } } { R _ { t - 1 } } \right. \leq C \sqrt { d } ( \log d ) ^ { 2 } ,\tag{68}
$$

$$
\operatorname* { m a x } _ { 1 \leq t \leq T _ { 2 } } \| G _ { t } \| ^ { 2 } \leq C d ( \log d ) ^ { 4 } .\tag{69}
$$

We work on this intersection. The first estimate controls the accumulated linear noise term. The second ensures that each logarithmic increment is well defined and bounds its quadratic remainder. The third controls the spherical normalization.

Step 2: Logarithmic comparison.

For $1 \leq t \leq \widehat { \tau } _ { a } ^ { R } \wedge T _ { 2 }$ , we have $0 < R _ { t - 1 } < a$ . Therefore (59) gives

$$
\frac { R _ { t } } { R _ { t - 1 } } \geq \frac { 1 + c _ { a } \eta _ { d } + 2 \eta _ { d } \Xi _ { t } / R _ { t - 1 } } { 1 + \eta _ { d } ^ { 2 } \| G _ { t } \| ^ { 2 } } ,
$$

where $c _ { a }$ is the lower-bound constant from Proposition 9.3. By (68),

$$
\left| c _ { a } \eta _ { d } + 2 \eta _ { d } \frac { \Xi _ { t } } { R _ { t - 1 } } \right| \leq C \eta _ { d } \sqrt { d } ( \log d ) ^ { 2 } = o ( 1 ) .
$$

Thus the numerator is positive for suficiently large $d ,$ including at the update reaching either stopping boundary. In particular, $R _ { t } > 0$ throughout these updates, so we may take logarithms.

Using log $( 1 + x ) \geq x - C x ^ { 2 }$ for suficiently small $| x |$ and log $( 1 + y ) \leq y$ for $y \geq 0$ , we obtain

$$
\log \frac { R _ { t } } { R _ { t - 1 } } \geq c _ { a } \eta _ { d } + 2 \eta _ { d } \frac { \Xi _ { t } } { R _ { t - 1 } } - C \eta _ { d } ^ { 2 } \left( 1 + \bigg | \frac { \Xi _ { t } } { R _ { t - 1 } } \bigg | ^ { 2 } \right) - \eta _ { d } ^ { 2 } \| G _ { t } \| ^ { 2 } .
$$

Applying (68) and (69) therefore gives

$$
\log \frac { R _ { t } } { R _ { t - 1 } } \geq c _ { a } \eta _ { d } + 2 \eta _ { d } \frac { \Xi _ { t } } { R _ { t - 1 } } - C \eta _ { d } ^ { 2 } d ( \log d ) ^ { 4 } .\tag{70}
$$

Summing the logarithmic increments up to n $\wedge \widehat { \tau } _ { a } ^ { R }$ and using (67), we find, simultaneously for every $0 \leq n \leq T _ { 2 }$

$$
\log \frac { R _ { n \wedge \widehat { \tau } _ { a } ^ { R } } } { R _ { 0 } } \geq c _ { a } \eta _ { d } \big ( n \wedge \widehat { \tau } _ { a } ^ { R } \big ) - C \left[ \eta _ { d } \sqrt { d T _ { 2 } \log d } + \eta _ { d } ^ { 2 } d T _ { 2 } ( \log d ) ^ { 4 } \right] .
$$

Since $T _ { 2 } \asymp \eta _ { d } ^ { - 1 }$ log d,

$$
\begin{array} { r l } & { \eta _ { d } \sqrt { d T _ { 2 } \log d } = O ( \sqrt { \eta _ { d } d } \log d ) = O ( ( \log d ) ^ { 1 - A / 2 } ) , } \\ & { \eta _ { d } ^ { 2 } d T _ { 2 } ( \log d ) ^ { 4 } = O ( \eta _ { d } d ( \log d ) ^ { 5 } ) = O ( ( \log d ) ^ { 5 - A } ) . } \end{array}
$$

Both errors tend to zero for a suficiently large universal choice of $A _ { 0 }$ . For suficiently large $d ,$ their total contribution, including the constant $C ,$ is at most − log θ. Exponentiating yields

$$
R _ { n \wedge \widehat { \tau } _ { a } ^ { R } } \geq \theta R _ { 0 } \exp \bigl ( c _ { a } \eta _ { d } \bigl ( n \wedge \widehat { \tau } _ { a } ^ { R } \bigr ) \bigr ) , \qquad 0 \leq n \leq T _ { 2 } .
$$

## Step 3: Exclusion of the lower exit.

Suppose that the auxiliary process exits through its lower boundary at some $\hat { \tau } _ { a } ^ { R } \leq T _ { 2 }$ . Evaluating the preceding inequality at this time gives

$$
R _ { \widehat { \tau } _ { a } ^ { R } } \geq \theta R _ { 0 } \exp ( c _ { a } \eta _ { d } \widehat { \tau } _ { a } ^ { R } ) \geq \theta R _ { 0 } > \beta R _ { 0 } .
$$

This contradicts the condition $R _ { \widehat { \tau } _ { \alpha } ^ { R } } \leq \beta R _ { 0 }$ defining a lower exit. Therefore the auxiliary stopping time can only record an upper exit before $T _ { 2 }$ , and

$$
\widehat { \tau } _ { a } ^ { R } \wedge T _ { 2 } = \tau _ { a } ^ { R } \wedge T _ { 2 } .
$$

Replacing the auxiliary stopping time by $\tau _ { a } ^ { R }$ in the growth estimate proves (66).

The event used in the proof has conditional probability at least $1 - 2 d ^ { - D ^ { \prime } } \geq 1 - d ^ { - D }$ for suficiently large d. All constants are uniform over the fixed teacher and initialization under consideration.

We now translate total-overlap growth into subspace recovery.

Proof of part (T) of Theorem 5.3. Choose $a = a ( \rho ) \in ( \rho , 1 )$ and fix $\theta , c _ { \mathrm { i n i t } } \in ( 0 , 1 )$ . Let $c _ { a }$ be the constant from Proposition 9.5, and choose $C _ { 0 } = C _ { 0 } ( \rho )$ such that $c _ { a } C _ { 0 } > 1$

By Lemmas A.1 and 9.1, the teacher belongs to $\mathcal { G } _ { d }$ and $R _ { 0 } \geq c _ { \mathrm { i n i t } } r / d$ with probability $1 - o ( 1 )$ The uniform conditional estimate in Proposition 9.5 then shows that (66) also holds up to $T _ { 2 }$ with probability $1 - o ( 1 )$ . Work on this intersection.

If $\tau _ { a } ^ { R } > T _ { 2 } ,$ then

$$
\begin{array} { r l } & { a > R _ { T _ { 2 } } } \\ & { \phantom { \sum } \geq \theta c _ { \mathrm { i n i t } } \frac { r } { d } \exp ( c _ { a } \eta _ { d } T _ { 2 } ) } \\ & { \phantom { \sum } \geq \theta c _ { \mathrm { i n i t } } \left( \frac { d } { r } \right) ^ { c _ { a } C _ { 0 } - 1 } . } \end{array}
$$

The last expression diverges because $d / r  \infty$ and $c _ { a } C _ { 0 } > 1$ , a contradiction. Thus $\tau _ { a } ^ { R } \leq T _ { 2 }$

At that time, (56) gives

$$
\| w _ { \tau _ { a } ^ { R } } ^ { \parallel } \| ^ { 2 } \geq \frac { R _ { \tau _ { a } ^ { R } } } { 1 + r \mu _ { * } } \geq \frac { a } { 1 + r \mu _ { * } } > \rho
$$

for suficiently large d. Consequently,

$$
\tau _ { \rho } ^ { ( 2 ) , \mathsf { T } } \leq \tau _ { a } ^ { R } \leq \left\lceil \frac { C _ { 0 } \log ( d / r ) } { \eta _ { d } } \right\rceil .
$$

Absorbing the ceiling into a constant $C _ { T } ( \rho )$ proves part (T).

## 9.6 Augmented and untied upper bounds

We complete the proof of Theorem 5.3 using the componentwise reductions from Section 8. For each procedure, it sufices to follow one fixed weight.

Proof of part (A) of Theorem 5.3. By (42), the first augmented weight follows the tied update with reference step size $\eta _ { d }$ . Its initialization has the same distribution as the tied initialization, so part (T) gives, with probability $1 - o ( 1 )$ ),

$$
\operatorname* { m a x } _ { 0 \leq t \leq C _ { T } \log ( d / r ) / \eta _ { d } } \| ( w _ { t } ^ { 1 , \mathsf { A } } ) ^ { \parallel } \| ^ { 2 } \geq \rho .
$$

Recovery of this weight implies recovery of the student. Thus part (A) holds with $C _ { A } = C _ { T }$ □

Proof of part (U) of Theorem 5.3. Step 1: Efective recursion. Retain the single-weight trajectory $u _ { t } = w _ { t } ^ { 1 , \mathsf { U } }$ and the quantities $G _ { t } ^ { \mathrm { S } }$ and $\Xi _ { t } ^ { \mathrm { S } }$ from Subsection 9.4, and write

$$
R _ { t } ^ { \mathsf { U } } : = R ( u _ { t } ) .
$$

By (43), $\Psi = \Phi / \sqrt { r }$ . Applying the quadratic-form calculation of Lemma 9.2 to the efective update (41) therefore gives

$$
\left( 1 + \frac { \eta _ { d } ^ { 2 } } { r } \Vert G _ { t } ^ { \mathrm { S } } \Vert ^ { 2 } \right) R _ { t } ^ { \mathrm { U } } = R _ { t - 1 } ^ { \mathrm { U } } + \frac { \eta _ { d } } { r } \Vert \nabla \Phi ( u _ { t - 1 } ) \Vert ^ { 2 } + \frac { 2 \eta _ { d } } { \sqrt { r } } \Xi _ { t } ^ { \mathrm { S } } + \frac { \eta _ { d } ^ { 2 } } { r } \langle G _ { t } ^ { \mathrm { S } } , B ^ { \ast } G _ { t } ^ { \mathrm { S } } \rangle .\tag{71}
$$

Since $B ^ { * }$ is positive semidefinite, Proposition 9.3 yields

$$
\left( 1 + \frac { \eta _ { d } ^ { 2 } } { r } \Vert G _ { t } ^ { \mathrm { S } } \Vert ^ { 2 } \right) R _ { t } ^ { \mathsf { U } } \geq \left( 1 + \frac { c _ { a } \eta _ { d } } { r } \right) R _ { t - 1 } ^ { \mathsf { U } } + \frac { 2 \eta _ { d } } { \sqrt { r } } \Xi _ { t } ^ { \mathrm { S } }\tag{72}
$$

whenever $0 < R _ { t - 1 } ^ { \mathsf { U } } < a$ , on $\mathcal { G } _ { d }$ and for suficiently large d. Thus the drift and normalization coeficients are divided by $r ,$ while the noise coeficient is divided by $\sqrt { r }$

Step 2: Stopped logarithmic growth. Choose $a = a ( \rho ) \in ( \rho , 1 )$ , fix $\theta , c _ { \mathrm { i n i t } } \in ( 0 , 1 )$ and $\beta \in ( 0 , \theta )$ and choose $C _ { 0 } = C _ { 0 } ( \rho )$ such that $c _ { a } C _ { 0 } > 1$ . Set

$$
T _ { 2 } ^ { \mathsf { U } } : = \left\lceil \frac { C _ { 0 } r \log ( d / r ) } { \eta _ { d } } \right\rceil .
$$

Condition on a teacher $w ^ { \ast } \in \mathcal G _ { d }$ and an initialization satisfying $R _ { 0 } ^ { \mathrm { U } } \geq c _ { \mathrm { i n i t } } r / d .$ . If $R _ { 0 } ^ { \mathsf { U } } \geq a$ , recovery already holds for suficiently large d by (56). Otherwise, define

$$
\widehat { \tau } _ { a } ^ { R , \mathsf { U } } : = \operatorname* { i n f } \{ t \geq 0 : R _ { t } ^ { \mathsf { U } } \leq \beta R _ { 0 } ^ { \mathsf { U } } \mathrm { ~ o r ~ } R _ { t } ^ { \mathsf { U } } \geq a \} .
$$

Before and including the update reaching this stopping time,

$$
R _ { t - 1 } ^ { \mathsf { U } } > \beta c _ { \mathrm { i n i t } } \frac { r } { d } .
$$

Hence the untied relative-noise estimate from Subsection 9.4 applies.

The variables $\Xi _ { t } ^ { \mathrm { S } } / ( \sqrt { d } R _ { t - 1 } ^ { \mathsf { U } } )$ , set to zero after ${ \widehat { \tau } } _ { a } ^ { R , \mathsf { U } }$ , are martingale diferences with conditional L<sup>ν</sup>-norm at most $C \nu ^ { 2 }$ . The stopping indicator includes the exit update and is measurable with respect to $\mathcal { F } _ { t - 1 } ^ { \mathrm { S } }$ . Since $r = d ^ { \delta }$ with $\delta < 1 / 2$ and A is fixed,

$$
T _ { 2 } ^ { \mathsf { U } } \asymp d ^ { 1 + \delta } ( \log d ) ^ { A + 1 } \leq d ^ { 2 }
$$

for suficiently large d. Thus the truncation and concentration argument in the proof of Lemma 9.6 applies with $T _ { 2 }$ replaced by $T _ { 2 } ^ { \mathsf { U } }$ . It controls both the cumulative stopped relative noise and the

individual increments over this horizon. Together with the untied version of (64), these estimates give the logarithmic comparison below.

More precisely, the one-step perturbation of the numerator in (72), after division by $R _ { t - 1 } ^ { \mathsf { U } }$ satisfies

$$
\left| \frac { c _ { a } \eta _ { d } } { r } + \frac { 2 \eta _ { d } } { \sqrt { r } } \frac { \Xi _ { t } ^ { \mathrm { S } } } { R _ { t - 1 } ^ { \cup } } \right| \le C _ { D } \frac { \eta _ { d } } { \sqrt { r } } \sqrt { d } ( \log d ) ^ { 2 } = o ( 1 ) .
$$

Thus the numerator remains positive through the exit update. Expanding the logarithm and summing, as in Subsection 9.5, gives, for every fixed $D > 0$ , with conditional probability at least $1 - d ^ { - D }$ , simultaneously for $0 \leq n \leq T _ { 2 } ^ { \mathsf { U } }$ ,

$$
\log \frac { R _ { n \wedge \widehat \tau _ { a } ^ { R , \mathsf { u } } } ^ { \mathsf { U } } } { R _ { 0 } ^ { \mathsf { U } } } \geq \frac { c _ { a } \eta _ { d } } { r } ( n \wedge \widehat \tau _ { a } ^ { R , \mathsf { U } } ) - C _ { D } \left[ \frac { \eta _ { d } } { \sqrt { r } } \sqrt { d T _ { 2 } ^ { \mathsf { U } } \log d } + \frac { \eta _ { d } ^ { 2 } } { r } d T _ { 2 } ^ { \mathsf { U } } ( \log d ) ^ { 4 } \right] .
$$

All conditional estimates are uniform over the teacher and initialization under consideration.

Since $T _ { 2 } ^ { \mathsf { U } } \asymp r \eta _ { d } ^ { - 1 }$ log d, the accumulated errors satisfy

$$
\frac { \eta _ { d } } { \sqrt { r } } \sqrt { d T _ { 2 } ^ { \mathrm { U } } \log d } = O \bigl ( ( \log d ) ^ { 1 - A / 2 } \bigr ) , \qquad \frac { \eta _ { d } ^ { 2 } } { r } d T _ { 2 } ^ { \mathrm { U } } ( \log d ) ^ { 4 } = O \bigl ( ( \log d ) ^ { 5 - A } \bigr ) .
$$

These are the same orders as in the tied proof. For the same suficiently large universal choice of $A _ { 0 }$ , their total contribution tends to zero. Consequently, for suficiently large $d ,$

$$
R _ { n \wedge \widehat \tau _ { a } ^ { R , \mathsf { u } } } ^ { \mathsf { U } } \geq \theta R _ { 0 } ^ { \mathsf { U } } \exp \left( \frac { c _ { a } \eta _ { d } } { r } ( n \wedge \widehat \tau _ { a } ^ { R , \mathsf { U } } ) \right) , \qquad 0 \leq n \leq T _ { 2 } ^ { \mathsf { U } } .
$$

Step 3: Reaching the recovery threshold.

Since $\theta > \beta ,$ , this excludes an exit through the lower boundary before $T _ { 2 } ^ { \mathsf { U } }$ , exactly as in the tied proof. Moreover, if $\hat { \tau } _ { a } ^ { R , \cup } > T _ { 2 } ^ { \cup }$ , then

$$
a > R _ { T _ { 2 } ^ { \mathsf { U } } } ^ { \mathsf { U } } \geq \theta c _ { \mathrm { i n i t } } \left( \frac { d } { r } \right) ^ { c _ { a } C _ { 0 } - 1 } ,
$$

which is impossible for suficiently large d. Therefore the process reaches $R _ { t } ^ { \mathsf { U } } \geq a$ at some $t \leq T _ { 2 } ^ { \mathsf { U } }$ At that time,

$$
\| u _ { t } ^ { \| } \| ^ { 2 } \geq \frac { R _ { t } ^ { \cup } } { 1 + r \mu _ { * } } \geq \frac { a } { 1 + r \mu _ { * } } > \rho .
$$

Finally, Lemma 9.1 bounds the excluded initialization probability by $C e ^ { - c r }$ , uniformly over $w ^ { \ast } \in \mathcal G _ { d }$ . Averaging the conditional estimates and using Lemma A.1 shows that, with probability $1 - o ( 1 )$

$$
\tau _ { \rho } ^ { ( 2 ) , \mathsf { U } } \leq T _ { 2 } ^ { \mathsf { U } } \leq \frac { C _ { U } ( \rho ) r \log ( d / r ) } { \eta _ { d } } ,
$$

after absorbing the ceiling into $C _ { U } ( \rho )$ . This proves part (U).

Together with part (T), a finite union bound gives the three recovery guarantees simultaneously with probability $1 - o ( 1 )$ ).

Remark 9.7 (Shifted tied weights). The subspaces ${ \boldsymbol { S } } ^ { * }$ and $( S ^ { * } ) ^ { \perp }$ are invariant under Π. Consequently,

$$
\begin{array} { r } { ( \Pi ^ { k } w ) ^ { \parallel } = \Pi ^ { k } w ^ { \parallel } , \qquad \lVert ( \Pi ^ { k } w ) ^ { \parallel } \rVert ^ { 2 } = \lVert w ^ { \parallel } \rVert ^ { 2 } . } \end{array}
$$

Thus, when the tied weight achieves subspace recovery, all its cyclic shifts have the same projection mass in the teacher subspace.

## 10 Lower bounds for weak recovery

In this section, we prove Theorem 5.4 under its assumptions on the teacher, initialization, and scaling. The link is a fixed centered polynomial of degree P with information exponent $p \geq 2$ . The population identities and componentwise reductions from Sections 6 and 8 apply to this entire class: their derivations use the finite Hermite expansion and do not require a parity assumption. Throughout, $A > 0$ is fixed, the reference learning rate is

$$
\eta _ { d } = d ^ { - p / 2 } ( \log d ) ^ { - A } ,
$$

and the three procedures use the step sizes in (6). We retain the notation $w _ { t } ^ { j , \mathsf { k } }$ and $s _ { \boldsymbol { \mathrm { k } } }$ for the iterates and numbers of weights.

## 10.1 Proof strategy

We show that all absolute correlations remain at their initialization scale $O ( \gamma _ { d } )$ for a suficiently small constant multiple of the microscopic growth time. This excludes both weak directional recovery and weak subspace recovery. The argument applies simultaneously to every weight, including when the procedures share their observations.

The proof has two ingredients. First, we bound the absolute population drift by the largest absolute correlation raised to the power $p - 1$ . Lemma 7.6 then gives a pathwise upper comparison that retains the exact spherical normalization. Second, we control the accumulated coordinate noise until the first escape from the initialization scale, uniformly over all weights and teacher directions.

For the tied and augmented procedures, the relevant horizon is of order $( \eta _ { d } \gamma _ { d } ^ { p - 2 } ) ^ { - 1 }$ . For the untied procedure, it is longer by a factor r. On these horizons, the stochastic contribution is $o ( \gamma _ { d } )$ while a suficiently small time constant makes the accumulated drift smaller than the distance to the escape threshold. Evaluating the comparison at a hypothetical first exit gives the desired contradiction.

## 10.2 Initialization and escape times

Define the largest absolute correlation of a weight and of a procedure by

$$
\widehat { M } ( w ) : = \operatorname* { m a x } _ { 0 \leq k < r } | m _ { k } ( w ) | , \qquad \widehat { M } _ { t } ^ { \mathrm { k } } : = \operatorname* { m a x } _ { 1 \leq j \leq s _ { \mathrm { k } } } \widehat { M } ( w _ { t } ^ { j , \mathrm { k } } ) .
$$

We control absolute correlations for every $p \geq 2$

Lemma A.2 establishes that $\widehat { M } ( w _ { 0 } ) = O ( \gamma _ { d } )$ with probability $1 - o ( 1 )$ for a single random weight. Here we need this upper bound simultaneously over all initial weights, so we use a quantitative tail estimate before taking the union bound.

Fix a teacher $w ^ { * }$ . By Assumption 2, each initial weight is conditionally uniform on $\mathbb { S } ^ { d - 1 }$ . The standard spherical-cap bound therefore gives, for every procedure, weight, and teacher direction,

$$
\begin{array} { r } { \mathbb { P } \left( | m _ { k } ( w _ { 0 } ^ { j , \mathsf { k } } ) | > z \middle | w ^ { * } \right) \leq C e ^ { - c d z ^ { 2 } } , \qquad 0 < z < 1 , } \end{array}
$$

with universal constants $c , C > 0$ . Since $\gamma _ { d } = o ( 1 )$ , we may take $z = C _ { \mathrm { i n i t } } \gamma _ { d }$ for any fixed $C _ { \mathrm { i n i t } } > 0$ and suficiently large d. A union bound over the $( 1 + 2 s ) r$ initial correlations then yields

$$
\mathbb { P } \left( \operatorname* { m a x } _ { { \bf k } \in \{ \mathsf { T } , \mathsf { U } , \mathsf { A } \} } \widehat { M } _ { 0 } ^ { \mathsf { k } } > C _ { \mathrm { i n i t } } \gamma _ { d } \bigg | w ^ { * } \right) \le C ( 1 + 2 s ) r e ^ { - c C _ { \mathrm { i n i t } } ^ { 2 } \log r } \le C d ^ { \kappa + \delta - c C _ { \mathrm { i n i t } } ^ { 2 } \delta } .\tag{73}
$$

The last inequality uses $s \leq d ^ { \kappa }$ and $r = d ^ { \delta }$ from Assumption 3. Choose $C _ { \mathrm { i n i t } } = C _ { \mathrm { i n i t } } ( \delta , \kappa )$ large enough that $c C _ { \mathrm { i n i t } } ^ { 2 } \delta > \kappa + \delta$ . Then (73) is $o ( 1 )$ , uniformly over the teacher.

Combining this estimate with Lemma A.1, we obtain, with probability $1 - o ( 1 )$

$$
w ^ { * } \in \mathcal { G } _ { d } , \qquad \operatorname* { m a x } _ { \mathsf { k } \in \{ \mathsf { T } , \mathsf { U } , \mathsf { A } \} } \widehat { M } _ { 0 } ^ { \mathsf { k } } \leq C _ { \mathrm { i n i t } } \gamma _ { d } .
$$

We will first establish the subsequent estimates conditionally on a teacher and initial weights satisfying these conditions.

Fix any constant $C _ { \mathrm { e s c } } > C _ { \mathrm { i n i t } }$ and write $\tau _ { \mathrm { e s c } } ^ { \mathrm { k } } : = \tau _ { \mathrm { e s c } } ^ { \mathrm { k } } ( C _ { \mathrm { e s c } } )$ for the escape times defined in (11). On the initialization event above, $\tau _ { \mathrm { e s c } } ^ { \mathsf { k } } \geq 1$ . For every integer $1 \leq t \leq \tau _ { \mathrm { e s c } } ^ { \mathsf { k } }$

$$
\operatorname* { m a x } _ { 1 \le j \le s _ { \mathsf { k } } } \widehat M ( w _ { t - 1 } ^ { j , \mathsf { k } } ) < C _ { \mathrm { e s c } } \gamma _ { d } .
$$

Thus bounds evaluated at the preceding iterate remain valid through the update producing the first exit. We will use this observation to apply the upper comparison at a hypothetical exit time and obtain a contradiction.

## 10.3 An upper comparison for the correlations

Throughout this subsection, Φ denotes the tied population objective. The finite Hermite expansion implies

$$
| \phi ^ { \prime } ( z ) | \leq C _ { \sigma } | z | ^ { p - 1 } , \qquad | z | \leq 1 .
$$

We combine this bound with Lemma 6.1 and the overlap-energy estimate (90) from Appendix B.2. We combine this bound with Lemma 6.1 and the overlap-energy estimate (90) from Appendix B.2.

Let $\mathcal { F } _ { t }$ denote the joint filtration generated by the teacher, all initial weights, and all observations used through time t. For $\mathsf { k } \in \{ \mathsf { T } , \mathsf { A } \}$ , let $G _ { t } ^ { j , \mathsf { k } }$ be the tied sample gradient evaluated at $w _ { t - 1 } ^ { j , \mathsf { k } }$ . For $\mathsf { k } = \mathsf { U }$ , let it be the single-weight sample gradient associated with (37). Define

$$
\begin{array} { r } { \xi _ { k , t } ^ { j , \mathsf { k } } : = - \left. \Pi ^ { k } w ^ { \ast } , G _ { t } ^ { j , \mathsf { k } } - \mathbb { E } [ G _ { t } ^ { j , \mathsf { k } } \mid \mathcal { F } _ { t - 1 } ] \right. . } \end{array}
$$

These are the coordinate fluctuations already used in Sections 6 and 8, now indexed by the procedure and weight. Fresh-sample independence gives

$$
\mathbb { E } [ G _ { t } ^ { j , \mathsf { k } } \mid \mathcal { F } _ { t - 1 } ] = \left\{ \nabla \Phi ( w _ { t - 1 } ^ { j , \mathsf { k } } ) , \quad \mathsf { k } \in \{ \mathsf { T } , \mathsf { A } \} , \right.
$$

where Ψ is the single-weight population objective from (43). By the definition of the centered fluctuations,

$$
\mathbb { E } [ \xi _ { k , t } ^ { j , \mathsf { k } } \mid \mathcal { F } _ { t - 1 } ] = 0 .
$$

Lemma 10.1 (Upper comparison before escape). On $\mathcal { G } _ { d }$ , there exists $C _ { \sigma } < \infty$ such that, for all suficiently large d and every $w \in \mathbb { S } ^ { d - 1 }$ ,

$$
\operatorname* { m a x } _ { 0 \leq k < r } \left| \left. \Pi ^ { k } w ^ { * } , \nabla \Phi ( w ) \right. \right| \leq C _ { \sigma } \widehat { M } ( w ) ^ { p - 1 } .
$$

Moreover, for $\mathsf { k } \in \{ \mathsf { T } , \mathsf { A } \}$ and every integer $0 \leq n \leq \tau _ { \mathrm { e s c } } ^ { \mathsf { k } }$ 7

$$
\widehat { M } _ { n } ^ { \mathbf { k } } \leq \widehat { M } _ { 0 } ^ { \mathbf { k } } + C _ { \sigma } \eta _ { d } n ( C _ { \mathrm { e s c } } \gamma _ { d } ) ^ { p - 1 } + 2 \eta _ { d } \operatorname* { m a x } _ { 1 \leq j \leq s _ { \mathbf { k } } 0 \leq h \leq n } \operatorname* { m a x } _ { \left| t = 1 \right. } \xi _ { k , t } ^ { j , \mathbf { k } } \bigg | .\tag{74}
$$

For the untied procedure and every integer $0 \leq n \leq \tau _ { \mathrm { e s c } } ^ { \mathrm { U } }$

$$
\widehat { M } _ { n } ^ { \mathsf { U } } \leq \widehat { M } _ { 0 } ^ { \mathsf { U } } + C _ { \sigma } \frac { \eta _ { d } } { r } n ( C _ { \mathrm { e s c } } \gamma _ { d } ) ^ { p - 1 } + \frac { 2 \eta _ { d } } { \sqrt { r } } \operatorname* { m a x } _ { 1 \leq j \leq s } \operatorname* { m a x } _ { 0 \leq h \leq n } \left. \sum _ { t = 1 } ^ { h } \xi _ { k , t } ^ { j , \mathsf { U } } \right. .\tag{75}
$$

Both comparisons are pathwise and include the first exit step.

Proof. Population drift. Fix $w \in \mathbb { S } ^ { d - 1 }$ and write $M = \widehat { M } ( w ) . \mathrm { ~ I f ~ } M = 0$ , then every correlation vanishes and $\phi ^ { \prime } ( 0 ) = 0$ , so Lemma 6.1 gives $\nabla \Phi ( w ) = 0$ . Suppose henceforth that $M > 0$ . Since $p \geq 2$

$$
\sum _ { \ell = 0 } ^ { r - 1 } | m _ { \ell } ( w ) | ^ { p } \leq M ^ { p - 2 } \sum _ { \ell = 0 } ^ { r - 1 } m _ { \ell } ( w ) ^ { 2 } .
$$

By Lemma 6.1, the derivative bound above, and the definition of $\mu _ { * }$

$$
\begin{array} { r l } {  { |  \Pi ^ { k } w ^ { * } , \nabla \Phi ( w )  | \le C _ { \sigma } [ ( 1 + r \mu _ { * } ) M ^ { p - 1 } + M \sum _ { \ell = 0 } ^ { r - 1 } | m _ { \ell } ( w ) | ^ { p } ] } } \\ & { \le C _ { \sigma } M ^ { p - 1 } [ 1 + r \mu _ { * } + \sum _ { \ell = 0 } ^ { r - 1 } m _ { \ell } ( w ) ^ { 2 } ] } \\ & { \le C _ { \sigma } M ^ { p - 1 } . } \end{array}
$$

The last inequality uses $r \mu _ { * } = o ( 1 )$ on $\mathcal { G } _ { d }$ and the uniform overlap-energy bound (90). After enlarging $C _ { \sigma }$ , this proves the first assertion, including the case $p = 2$

Tied and augmented weights. Fix $\mathsf { k } \in \{ \mathsf { T } , \mathsf { A } \}$ , a weight index $j ,$ and a teacher direction k. The exact coordinate update from the proof of Lemma 6.2, together with (42), has the form

$$
d _ { t } x _ { t + 1 } = x _ { t } + F _ { t } + N _ { t + 1 } - N _ { t } ,
$$

where

$$
x _ { t } = m _ { k } ( w _ { t } ^ { j , \mathbf { k } } ) , \qquad d _ { t } = \sqrt { 1 + \eta _ { d } ^ { 2 } \| G _ { t + 1 } ^ { j , \mathbf { k } } \| ^ { 2 } } ,
$$

and

$$
F _ { t } = - \eta _ { d } \left. \Pi ^ { k } w ^ { * } , \nabla \Phi ( w _ { t } ^ { j , \mathbf { k } } ) \right. ,
$$

$$
N _ { t } = \eta _ { d } \sum _ { \ell = 1 } ^ { t } \xi _ { k , \ell } ^ { j , \ k } .
$$

Thus $d _ { t } \geq 1$ and $N _ { 0 } = 0 .$ , so Lemma 7.6 applies with $q _ { t } = 1$

For every integer $n \leq \tau _ { \mathrm { e s c } } ^ { \mathsf { k } } ,$ each $t < n$ lies strictly before escape. The population bound just proved therefore gives

$$
\sum _ { t = 0 } ^ { n - 1 } | F _ { t } | \leq C _ { \sigma } \eta _ { d } \sum _ { t = 0 } ^ { n - 1 } \widehat { M } ( w _ { t } ^ { j , \mathsf { k } } ) ^ { p - 1 } \leq C _ { \sigma } \eta _ { d } n ( C _ { \mathrm { e s c } } \gamma _ { d } ) ^ { p - 1 } .
$$

Consequently, Lemma 7.6 yields

$$
| m _ { k } ( w _ { n } ^ { j , { \bf k } } ) | \leq | m _ { k } ( w _ { 0 } ^ { j , { \bf k } } ) | + C _ { \sigma } \eta _ { d } n ( C _ { \mathrm { e s c } } \gamma _ { d } ) ^ { p - 1 } + 2 \eta _ { d } \operatorname* { m a x } _ { 0 \leq h \leq n } \left| \sum _ { \ell = 1 } ^ { h } \xi _ { k , \ell } ^ { j , { \bf k } } \right| .
$$

Taking the maximum over $j$ and k proves (74).

Untied weights. By (43), $\nabla \Psi = \nabla \Phi / \sqrt { r }$ . For a fixed untied weight $j$ and teacher direction $k ,$ the exact update (45) therefore has the same damping form, now with

$$
x _ { t } = m _ { k } ( w _ { t } ^ { j , \mathsf { U } } ) , \qquad d _ { t } = \sqrt { 1 + \frac { \eta _ { d } ^ { 2 } } { r } \| G _ { t + 1 } ^ { j , \mathsf { U } } \| ^ { 2 } } ,
$$

and

$$
F _ { t } = - \frac { \eta _ { d } } { r } \left. \Pi ^ { k } w ^ { * } , \nabla \Phi ( w _ { t } ^ { j , \mathsf { U } } ) \right. ,
$$

$$
N _ { t } = \frac { \eta _ { d } } { \sqrt { r } } \sum _ { \ell = 1 } ^ { t } \xi _ { k , \ell } ^ { j , \mathsf { U } } .
$$

Again $d _ { t } \geq 1$ , and for $n \leq \tau _ { \mathrm { e s c } } ^ { \mathsf { U } }$

$$
\sum _ { t = 0 } ^ { n - 1 } | F _ { t } | \leq C _ { \sigma } \frac { \eta _ { d } } { r } n ( C _ { \mathrm { e s c } } \gamma _ { d } ) ^ { p - 1 } .
$$

Applying Lemma 7.6 with $q _ { t } = 1$ and taking the maximum over all weights and teacher directions gives (75).

Both applications are pathwise. In particular, all drift terms used to bound the correlation at time n are evaluated at times $t < n$ . The comparisons therefore remain valid when n is the first exit time. □

## 10.4 Noise before escape

The moment estimates of Lemma 6.4 apply to the tied and augmented weights, while Lemma 8.1 applies to the untied weights. Their proofs use only the finite Hermite expansion and $p \geq 2$ , so the estimates hold for every polynomial link considered in Theorem 5.4. Before escape,

$$
r m _ { k } ( w _ { t - 1 } ^ { j , { \ k } } ) ^ { 2 } \leq C _ { \mathrm { e s c } } ^ { 2 } r \gamma _ { d } ^ { 2 } = o ( 1 ) .
$$

Thus the stopped coordinate fluctuations have bounded conditional variance and conditional $L ^ { \nu } .$ -norm of order $\nu ^ { P }$

Lemma 10.2 (Noise before escape). Fix a teacher $w ^ { \ast } \in \mathcal { G } _ { d }$ and initial weights satisfying

$$
\operatorname* { m a x } _ { \mathsf { k } \in \{ \mathsf { T } , \mathsf { U } , \mathsf { A } \} } \widehat { M } _ { 0 } ^ { \mathsf { k } } \leq C _ { \mathrm { i n i t } } \gamma _ { d } .
$$

For every fixed $D > 0$ , there exists $C _ { \sigma , D } < \infty$ such that the following holds for all suficiently large d. For any deterministic integer horizons $1 \leq T _ { \mathsf { k } } \leq d ^ { p }$ , with probability at least $1 - d ^ { - D }$ over the SGD samples, simultaneously for $\mathsf { k } \in \{ \mathsf { T } , \mathsf { U } , \mathsf { A } \}$

$$
\operatorname* { m a x } _ { 1 \leq j \leq s _ { \mathrm { k } } 0 \leq n \leq T _ { \mathrm { k } } } \left. \sum _ { t = 1 } ^ { n \wedge \tau _ { \mathrm { e s c } } ^ { \mathrm { k } } } \xi _ { k , t } ^ { j , \mathrm { k } } \right. \leq C _ { \sigma , D } \left( \sqrt { T _ { \mathrm { k } } \log d } + ( \log d ) ^ { P + 1 } \right) .\tag{76}
$$

The constant may also depend on $\delta , \kappa , C _ { \mathrm { i n i t } } , C _ { \mathrm { e s c } }$ . The constants and dimension threshold are uniform over the conditioned teacher and initial weights.

Proof. Fix the teacher and initial weights as in the statement, and work under the corresponding conditional law.

For each procedure, weight, and teacher direction, set

$$
Y _ { t } : = \mathbf { 1 } _ { \{ t \leq \tau _ { \mathrm { e s c } } ^ { \mathrm { k } } \} } \xi _ { k , t } ^ { j , \mathrm { k } } , \qquad 1 \leq t \leq T _ { \mathrm { k } } ,
$$

suppressing the coordinate indices in $Y _ { t } .$ . The stopping indicator is $\mathscr { F } _ { t - 1 } { \mathrm { - m e a s u r a b l e } }$ , so $( Y _ { t } )$ is a martingale diference sequence. On $\{ t \leq \tau _ { \mathrm { e s c } } ^ { \mathsf { k } } \}$ ,

$$
r m _ { k } ( w _ { t - 1 } ^ { j , { \ k } } ) ^ { 2 } \leq C _ { \mathrm { e s c } } ^ { 2 } r \gamma _ { d } ^ { 2 } = o ( 1 ) .
$$

Consequently, Lemmas 6.4 and 8.1 give, uniformly over all coordinate processes,

$$
\mathbb { E } [ Y _ { t } ^ { 2 } \mid { \mathcal F } _ { t - 1 } ] \le C _ { \sigma } , \qquad ( \mathbb { E } [ | Y _ { t } | ^ { \nu } \mid { \mathcal F } _ { t - 1 } ] ) ^ { 1 / \nu } \le C _ { \sigma } \nu ^ { P } , \qquad \nu \ge 2 .
$$

The earlier moment calculations apply to the joint filtration because the observation at time t is fresh conditionally on $\mathcal { F } _ { t - 1 }$

These are exactly the conditional moment bounds used in the proof of Lemma 7.5. Apply its truncation, recentering, and maximal-concentration argument with moment order proportional to log d and cutof proportional to $( \log d ) ^ { P }$ . For any fixed $b > p ,$ , choosing the proportionality constants suficiently large gives

$$
\mathbb { P } \left( \operatorname* { m a x } _ { 0 \leq n \leq T _ { \mathsf { k } } } \left| \sum _ { t = 1 } ^ { n } Y _ { t } \right| > C _ { \sigma , b } \left[ \sqrt { T _ { \mathsf { k } } \log d } + ( \log d ) ^ { P + 1 } \right] \right) \leq C ( T _ { \mathsf { k } } + 1 ) d ^ { - b } .
$$

Here the factor $T _ { \mathsf { k } }$ accounts for the truncation events, while the maximal martingale bound contributes $C d ^ { - b }$ . The accumulated centering correction is at most

$$
\begin{array} { r } { C _ { \sigma , b } T _ { \mathsf { k } } ( \log d ) ^ { P } d ^ { - b } \leq C _ { \sigma , b } d ^ { p - b } ( \log d ) ^ { P } = o ( 1 ) , } \end{array}
$$

and is absorbed into the displayed threshold.

Choose $b > D + \kappa + \delta + p .$ . A union bound over the $( 1 + 2 s ) r$ coordinate processes bounds the total failure probability by

$$
C ( 1 + 2 s ) r ( d ^ { p } + 1 ) d ^ { - b } \leq C d ^ { \kappa + \delta + p - b } \leq d ^ { - D }
$$

for suficiently large d. No independence between these processes is needed.

Finally,

$$
\sum _ { t = 1 } ^ { n } Y _ { t } = \sum _ { t = 1 } ^ { n \wedge \tau _ { \mathrm { e s c } } ^ { \mathrm { k } } } \xi _ { k , t } ^ { j , \mathrm { k } } ,
$$

so the simultaneous partial-sum bounds give (76). The stopping convention includes the exitproducing increment, and all estimates are uniform over the conditioned teacher and initial weights.

## 10.5 Proof of the non-escape theorem

We now combine Lemmas 10.1 and 10.2 to prove Theorem 5.4.

Proof of Theorem $5 . 4 .$ Take $C _ { \mathrm { i n i t } }$ as in Subsection 10.2 and fix $C _ { \mathrm { e s c } } > C _ { \mathrm { i n i t } }$ depending only on $\delta , \kappa$ Let $C _ { \sigma }$ be the constant in Lemma 10.1, and choose $c > 0$ so that

$$
C _ { \mathrm { i n i t } } + C _ { \sigma } c C _ { \mathrm { e s c } } ^ { p - 1 } < C _ { \mathrm { e s c } } .
$$

With $T _ { d }$ as in the theorem, define

$$
\begin{array} { r } { T _ { \mathsf { T } } = T _ { \mathsf { A } } : = \lfloor T _ { d } \rfloor , \qquad T _ { \mathsf { U } } : = \lfloor r T _ { d } \rfloor . } \end{array}
$$

Since

$$
\frac { 1 } { \eta _ { d } \gamma _ { d } ^ { p - 2 } } = \frac { d ^ { p - 1 } ( \log d ) ^ { A } } { ( \log r ) ^ { ( p - 2 ) / 2 } } \asymp d ^ { p - 1 } ( \log d ) ^ { A - ( p - 2 ) / 2 } ,
$$

all three horizons diverge. Moreover,

$$
T _ { \mathsf { U } } = O \Bigl ( d ^ { p - 1 + \delta } ( \log d ) ^ { A - ( p - 2 ) / 2 } \Bigr ) = o ( d ^ { p } ) .
$$

Thus $1 \leq T _ { \mathsf { k } } \leq d ^ { p }$ for suficiently large $d ,$ and Lemma 10.2 applies.

Step 1: The accumulated noise is smaller than initialization. Fix a teacher $w ^ { \ast } \in \mathcal G _ { d }$ and initial weights satisfying ma $\mathrm { x } _ { \mathsf { k } } \widehat { M } _ { 0 } ^ { \mathsf { k } } \leq C _ { \mathrm { i n i t } } \gamma _ { d }$ . Apply Lemma 10.2 with the horizons above and any fixed $D > 0 .$ . On its event, which has conditional probability at least $1 - d ^ { - D }$ , the stochastic terms in (74) and (75) are $o ( \gamma _ { d } )$ simultaneously for all procedures. Indeed, the definitions of the horizons give

$$
\begin{array} { r } { \operatorname* { m a x } \left\{ \frac { \eta _ { d } \sqrt { T _ { \mathsf { T } } \log d } } { \gamma _ { d } } , \frac { \eta _ { d } \sqrt { T _ { \mathsf { U } } \log d } } { \sqrt { r } \gamma _ { d } } \right\} \leq \sqrt { c \eta _ { d } \gamma _ { d } ^ { - p } \log d } = O \Big ( ( \log d ) ^ { - A / 2 - ( p - 2 ) / 4 } \Big ) = o ( 1 ) . } \end{array}
$$

For the remaining term,

$$
\frac { \eta _ { d } ( \log d ) ^ { P + 1 } } { \gamma _ { d } } = { \cal O } \Bigl ( d ^ { - ( p - 1 ) / 2 } ( \log d ) ^ { P + 1 / 2 - A } \Bigr ) = o ( 1 ) .
$$

The corresponding untied term is smaller by $r ^ { - 1 / 2 }$ . Both estimates hold for every fixed $A > 0$ and $2 \leq p \leq P < \infty$

## Step 2: Excluding the first exit.

Work on this common noise event. For $n \leq \tau _ { \mathrm { e s c } } ^ { \mathrm { k } } \wedge T _ { \mathrm { k } }$ , every partial sum with $h \leq n$ agrees with its stopped counterpart in (76). For $\mathsf { k } \in \{ \mathsf { T } , \mathsf { A } \}$ , Lemma 10.1 therefore gives

$$
\widehat { M } _ { n } ^ { \mathbf { k } } \leq C _ { \mathrm { i n i t } } \gamma _ { d } + C _ { \sigma } \eta _ { d } T _ { \mathbf { k } } ( C _ { \mathrm { e s c } } \gamma _ { d } ) ^ { p - 1 } + o ( \gamma _ { d } ) \leq \left( C _ { \mathrm { i n i t } } + C _ { \sigma } c C _ { \mathrm { e s c } } ^ { p - 1 } + o ( 1 ) \right) \gamma _ { d } .
$$

For the untied procedure, the same comparison yields

$$
\widehat { M } _ { n } ^ { \mathsf { U } } \leq C _ { \mathrm { i n i t } } \gamma _ { d } + C _ { \sigma } \frac { \eta _ { d } } { r } T _ { \mathsf { U } } ( C _ { \mathrm { e s c } } \gamma _ { d } ) ^ { p - 1 } + o ( \gamma _ { d } ) \leq \left( C _ { \mathrm { i n i t } } + C _ { \sigma } c C _ { \mathrm { e s c } } ^ { p - 1 } + o ( 1 ) \right) \gamma _ { d } .
$$

By the choice of $^ { c , }$ these upper bounds are strictly smaller than $C _ { \mathrm { e s c } } \gamma _ { d }$ for suficiently large d. If any procedure had $\tau _ { \mathrm { e s c } } ^ { \mathsf { k } } \leq T _ { \mathsf { k } }$ , applying its comparison at $n = \tau _ { \mathrm { e s c } } ^ { \mathsf { k } }$ would contradict the definition of the exit time. Consequently, simultaneously for all three procedures,

$$
\tau _ { \mathrm { e s c } } ^ { \mathsf { k } } > T _ { \mathsf { k } } , \qquad \operatorname* { m a x } _ { 0 \leq t \leq T _ { \mathsf { k } } } \widehat M _ { t } ^ { \mathsf { k } } < C _ { \mathrm { e s c } } \gamma _ { d } .
$$

The conditional estimates are uniform over the teacher and initial weights under consideration. Lemma A.1 and (73) show that these conditioning requirements hold with probability $1 - o ( 1 )$ ). Averaging gives the same unconditional probability for the simultaneous non-escape bounds. Since the escape times are integer-valued, they exceed the unrounded horizon $T _ { d }$ for the tied and augmented procedures and $r T _ { d }$ for the untied procedure. This proves the theorem, with c and $C _ { \mathrm { e s c } }$ depending only on $\sigma , \delta , \kappa$ □

## 11 Conclusion

We established an r-fold separation, up to logarithmic factors, in the sample complexity of symmetryaware and symmetry-agnostic feature learning. For spherical online SGD with correlation loss, this separation holds for weak directional recovery when $p \geq 3$ and weak subspace recovery for the pure quadratic Hermite link. In this setting, full-group data augmentation matches the statistical benefit of weight sharing: exploiting symmetry through either the architecture or the training procedure improves the balance between informative drift and stochastic fluctuations. We conjecture that the same separation persists for $d ^ { 1 / 2 } \lesssim r \ll d ,$ even though the initial largest correlation no longer remains dominant until weak recovery, and a diferent proof mechanism is therefore needed.

Several questions remain open. A natural next step is to study general links with information exponent two, extending the analysis of Ren & Lee (2025) to our symmetric setting. Here, quadratic subspace amplification and the directional information carried by higher-order terms may interact, raising the question of whether the same symmetry advantage persists. It would also be useful to establish strong recovery and vanishing prediction error. Beyond the recovery criterion, other losses and augmentation schemes deserve investigation: squared loss introduces interactions between filters, while using only a subset of group transformations raises a tradeof between sample eficiency and computation.

## Acknowledgments

We acknowledge funding from the Swiss National Science Foundation grants OperaGOST (grant number 200021\_200390) and DSGIANGO (grant number 225837). This work was supported by the Simons Collaboration on the Physics of Learning and Neural Computation via the Simons Foundation grant (#1257412) and by the EPFL AI Center PhD Fellowship Program 2026.

## AI use statement

We used generative AI tools to assist with formulating and refining mathematical statements, exploring proof strategies, and drafting and revising parts of the proofs. We also used AI assistance to identify relevant literature and references, improve the organization and exposition of the manuscript, and prepare and revise text, figure captions, and layouts. AI assistance was also used to develop and debug the experimental and analysis code and to prepare plots from the numerical results. The mathematical arguments and exposition were developed through iterative discussion, critical review, and substantive revision by the authors. The authors take responsibility for the final content of this work, including all AI-assisted mathematical claims, proofs, text, and artifacts.

## References

Emmanuel Abbe, Enric Boix-Adserà, Matthew S. Brennan, Guy Bresler, and Dheeraj Nagaraj. The staircase property: How hierarchical structure can guide deep learning. In Advances in Neural Information Processing Systems, volume 34, pp. 26989–27002. Curran Associates, Inc., 2021. URL https://proceedings.neurips.cc/paper\_files/paper/2021/hash/e2db7186375992e7291 65726762cb4c1-Abstract.html.

Emmanuel Abbe, Enric Boix-Adserà, and Theodor Misiakiewicz. The merged-staircase property: A necessary and nearly suficient condition for SGD learning of sparse functions on two-layer neural networks. In Proceedings of the 35th Conference on Learning Theory, volume 178 of Proceedings of Machine Learning Research, pp. 4782–4887. PMLR, 2022. URL https://proceedings.mlr. press/v178/abbe22a.html.

Emmanuel Abbe, Enric Boix-Adserà, and Theodor Misiakiewicz. SGD learning on neural networks: Leap complexity and saddle-to-saddle dynamics. In Proceedings of the 36th Conference on Learning Theory, volume 195 of Proceedings of Machine Learning Research, pp. 2552–2623. PMLR, 2023. URL https://proceedings.mlr.press/v195/abbe23a.html.

Luca Arnaboldi, Ludovic Stephan, Florent Krzakala, and Bruno Loureiro. From high-dimensional & mean-field dynamics to dimensionless ODEs: A unifying approach to SGD in two-layers networks. In The Thirty Sixth Annual Conference on Learning Theory, pp. 1199–1227. PMLR, 2023.

Luca Arnaboldi, Yatin Dandi, Florent Krzakala, Bruno Loureiro, Luca Pesce, and Ludovic Stephan. Online learning and information exponents: The importance of batch size & Time/Complexity tradeofs. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 1730–1762. PMLR, 2024. URL https: //proceedings.mlr.press/v235/arnaboldi24a.html.

Jimmy Ba, Murat A. Erdogdu, Taiji Suzuki, Zhichao Wang, Denny Wu, and Greg Yang. Highdimensional asymptotics of feature learning: How one gradient step improves the representation. In Advances in Neural Information Processing Systems, volume 35, pp. 37932–37946. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper\_files/paper/2022/ha sh/f7e7fabd73b3df96c54a320862afcb78-Abstract-Conference.html.

Jean Barbier, Florent Krzakala, Nicolas Macris, Léo Miolane, and Lenka Zdeborová. Optimal errors and phase transitions in high-dimensional generalized linear models. Proceedings of the National Academy of Sciences, 116(12):5451–5460, 2019.

Gérard Ben Arous, Reza Gheissari, and Aukosh Jagannath. Online stochastic gradient descent on non-convex losses from high-dimensional inference. Journal of Machine Learning Research, 22 (106):1–51, 2021. URL https://www.jmlr.org/papers/v22/20-1288.html.

Gérard Ben Arous, Murat A. Erdogdu, Nuri Mert Vural, and Denny Wu. Learning quadratic neural networks in high dimensions: SGD dynamics and scaling laws. In Advances in Neural Information Processing Systems, volume 38, Main Conference, pp. 146725–146812. Curran Associates, Inc., 2025. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/d7ce06e9293 c3d8e6cb3f80b4157f875-Abstract-Conference.html.

Gérard Ben Arous, Cédric Gerbelot, and Vanessa Piccolo. Stochastic gradient descent in high dimensions for multi-spiked tensor PCA. Comm. Pure Appl. Math., 79(10):2291–2369, 2026. ISSN 0010-3640,1097-0312. doi: 10.1002/cpa.70056. URL https://doi.org/10.1002/cpa.70056.

Alberto Bietti, Joan Bruna, and Loucas Pillaud-Vivien. On learning Gaussian multi-index models with gradient flow part I: General properties and two-timescale learning. Communications on Pure and Applied Mathematics, 78(12):2354–2435, 2025. doi: 10.1002/cpa.70006. URL https://doi.org/10.1002/cpa.70006.

Michael M. Bronstein, Joan Bruna, Taco Cohen, and Petar Veličković. Geometric deep learning: Grids, groups, graphs, geodesics, and gauges. arXiv preprint arXiv:2104.13478, 2021. URL https://arxiv.org/abs/2104.13478.

Francesco Cagnetta, Alessandro Favero, and Matthieu Wyart. What can be learnt with wide convolutional neural networks? In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 3347–3379. PMLR, 2023. URL https://proceedings.mlr.press/v202/cagnetta23a.html.

Shuxiao Chen, Edgar Dobriban, and Jane H. Lee. A group-theoretic framework for data augmentation. Journal of Machine Learning Research, 21(245):1–71, 2020. URL https://www.jmlr.org/paper s/v21/20-163.html.

Sitan Chen and Raghu Meka. Learning polynomials in few relevant dimensions. In Proceedings of the 33rd Conference on Learning Theory, volume 125 of Proceedings of Machine Learning Research, pp. 1161–1227. PMLR, 2020. URL https://proceedings.mlr.press/v125/chen20a.html.

Taco Cohen and Max Welling. Group equivariant convolutional networks. In Proceedings of the 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, pp. 2990–2999. PMLR, 2016. URL https://proceedings.mlr.press/v48/cohenc16 .html.

Alex Damian, Jason D. Lee, and Mahdi Soltanolkotabi. Neural networks can learn representations with gradient descent. In Proceedings of the 35th Conference on Learning Theory, volume 178 of Proceedings of Machine Learning Research, pp. 5413–5452. PMLR, 2022. URL https: //proceedings.mlr.press/v178/damian22a.html.

Alex Damian, Eshaan Nichani, Rong Ge, and Jason D. Lee. Smoothing the landscape boosts the signal for SGD: Optimal sample complexity for learning single index models. In Advances in Neural Information Processing Systems, volume 36, pp. 752–784. Curran Associates, Inc., 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/hash/02763667a5761ff 92bb15d8751bcd223-Abstract-Conference.html.

Alex Damian, Loucas Pillaud-Vivien, Jason D. Lee, and Joan Bruna. Computational-statistical gaps in Gaussian single-index models (extended abstract). In Proceedings of the 37th Conference on Learning Theory, volume 247 of Proceedings of Machine Learning Research, pp. 1262. PMLR, 2024. URL https://proceedings.mlr.press/v247/damian24a.html.

Alex Damian, Jason D. Lee, and Joan Bruna. The generative leap: Tight sample complexity for eficiently learning Gaussian multi-index models. In Advances in Neural Information Processing Systems, volume 38, Main Conference, pp. 28276–28311. Curran Associates, Inc., 2025. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/290056cc784d4825eb8 cf5c72f808ebb-Abstract-Conference.html.

Yatin Dandi, Florent Krzakala, Bruno Loureiro, Luca Pesce, and Ludovic Stephan. How two-layer neural networks learn, one (giant) step at a time. Journal of Machine Learning Research, 25(349): 1–65, 2024a. URL https://jmlr.org/papers/v25/23-1543.html.

Yatin Dandi, Emanuele Troiani, Luca Arnaboldi, Luca Pesce, Lenka Zdeborová, and Florent Krzakala. The benefits of reusing batches for gradient descent in two-layer networks: Breaking the curse of information and leap exponents. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 9991–10016. PMLR, 2024b. URL https://proceedings.mlr.press/v235/dandi24a.html.

Rishabh Dudeja and Daniel Hsu. Learning single-index models in gaussian space. In Proceedings of the 31st Conference on Learning Theory, volume 75 of Proceedings of Machine Learning Research, pp. 1887–1930. PMLR, 2018. URL https://proceedings.mlr.press/v75/dudeja18a.html.

Bryn Elesedy and Sheheryar Zaidi. Provably strict generalisation benefit for equivariant models. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 2959–2969. PMLR, 2021. URL https://proceedings.mlr. press/v139/elesedy21a.html.

David A. Freedman. On tail probabilities for martingales. The Annals of Probability, 3(1):100–118, 1975. doi: 10.1214/aop/1176996452.

Risi Kondor and Shubhendu Trivedi. On the generalization of equivariance and convolution in neural networks to the action of compact groups. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 2747–2755. PMLR, 2018. URL https://proceedings.mlr.press/v80/kondor18a.html.

Aakash Sunil Lahoti, Stefani Karp, Ezra Winston, Aarti Singh, and Yuanzhi Li. Role of locality and weight sharing in image-based tasks: A sample complexity separation between CNNs, LCNs, and FCNs. In The Twelfth International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/hash/71b17f00017da0d73823cc f7fbce2d4f-Abstract-Conference.html.

Clare Lyle, Mark van der Wilk, Marta Kwiatkowska, Yarin Gal, and Benjamin Bloem-Reddy. On the benefits of invariance in neural networks. arXiv preprint arXiv:2005.00178, 2020. URL https://arxiv.org/abs/2005.00178.

Song Mei, Theodor Misiakiewicz, and Andrea Montanari. Learning with invariances in random features and kernel models. In Proceedings of the 34th Conference on Learning Theory, volume 134 of Proceedings of Machine Learning Research, pp. 3351–3418. PMLR, 2021. URL https: //proceedings.mlr.press/v134/mei21a.html.

Alireza Mousavi-Hosseini, Sejun Park, Manuela Girotti, Ioannis Mitliagkas, and Murat A. Erdogdu. Neural networks eficiently learn low-dimensional representations with SGD. In The Eleventh International Conference on Learning Representations, 2023. URL https://iclr.cc/virtual/ 2023/poster/11421.

H. N. Nagaraja, Karthik Bharath, and Fangyuan Zhang. Spacings around an order statistic. Ann. Inst. Statist. Math., 67(3):515–540, 2015. ISSN 0020-3157,1572-9052. doi: 10.1007/s10463-014-0466-9. URL https://doi.org/10.1007/s10463-014-0466-9.

Kazusato Oko, Yujin Song, Taiji Suzuki, and Denny Wu. Learning sum of diverse features: Computational hardness and eficient gradient-based training for ridge combinations. In Proceedings of the 37th Conference on Learning Theory, volume 247 of Proceedings of Machine Learning Research, pp. 4009–4081. PMLR, 2024. URL https://proceedings.mlr.press/v247/oko24a. html.

Yunwei Ren and Jason D. Lee. Learning orthogonal multi-index models: A fine-grained information exponent analysis. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, Main Conference, pp. 165869–165920. Curran Associates, Inc., 2025. doi: 10.52202/085713-5528. URL https://proceedings.neurips.cc/paper\_files/paper/2025/file/f24777ce35e5a1ef111 336dc85c5d9c0-Paper-Conference.pdf.

Yunwei Ren, Eshaan Nichani, Denny Wu, and Jason D. Lee. Emergence and scaling laws in sgd learning of shallow neural networks. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, Main Conference, pp. 38227–38309. Curran Associates, Inc., 2025. doi: 10.52202/085713-1280. URL https://proceedings.neurips.cc/paper\_files/paper/2025/file/36d13b71487b299 65a2d3f5fcc109afc-Paper-Conference.pdf.

Berfin Şimşek, Amire Bendjeddou, and Daniel Hsu. Learning gaussian multi-index models with gradient flow: Time complexity and directional convergence. In Yingzhen Li, Stephan Mandt, Shipra Agrawal, and Emtiyaz Khan (eds.), Proceedings of The 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings of Machine Learning Research, pp. 4204–4212. PMLR, 2025. URL https://proceedings.mlr.press/v258/simsek25a.html.

Emanuele Troiani, Yatin Dandi, Leonardo Defilippis, Lenka Zdeborová, Bruno Loureiro, and Florent Krzakala. Fundamental computational limits of weak learnability in high-dimensional multi-index models. In Proceedings of the 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings of Machine Learning Research, pp. 2467–2475. PMLR, 2025. URL https://proceedings.mlr.press/v258/troiani25a.html.

Konstantinos Christopher Tsiolis, Alireza Mousavi-Hosseini, and Murat A. Erdogdu. From information to generative exponent: Learning rate induces phase transitions in SGD. In Advances in Neural Information Processing Systems, volume 38, Main Conference, pp. 83037–83071. Curran Associates, Inc., 2025. URL https://proceedings.neurips.cc/paper\_files/paper/2025/ha sh/77b7d565f2370979b898d76d7ea27288-Abstract-Conference.html.

Roman Vershynin. High-dimensional probability, volume 47 of Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, Cambridge, 2018. ISBN 978-1-108-41519-4. doi: 10.1017/9781108231596. URL https://doi.org/10.1017/9781108231596. An introduction with applications in data science, With a foreword by Sara van de Geer.

## A Teacher geometry and initialization

The purpose of this section is to characterize the random geometry of the teacher orbit and the scale of the correlations at initialization. For a fixed unit vector $v \in \mathbb { S } ^ { d - 1 }$ and an independent vector $u \sim \mathrm { U n i f } ( \mathbb { S } ^ { d - 1 } )$ , the correlation $\langle v , u \rangle$ is typically of order $d ^ { - 1 / 2 }$ . In our model, however, we consider the correlations of a random direction with the r shifted teacher directions $w ^ { * } , \Pi w ^ { * } , \ldots , \Pi ^ { r - 1 } w ^ { * }$ Because these directions are nearly orthogonal, the corresponding correlations behave approximately as $r$ independent Gaussian coordinates. Their largest relevant value is therefore enhanced by the usual extreme-value factor and occurs at the scale

$$
\gamma _ { d } : = \sqrt { \frac { \log r } { d } } .
$$

We first show that the teacher orbit is nearly orthogonal. We then establish the initialization scale $\gamma _ { d }$ and a quantitative gap between the largest relevant correlation and its closest competitor, where in the even-p case correlations are compared in absolute value. This gap provides the initial asymmetry that is amplified by the nonlinear SGD dynamics.

Recall that the teacher direction is drawn uniformly at random, $w ^ { * } \sim \mathrm { U n i f } ( \mathbb { S } ^ { d - 1 } )$ . We begin by controlling the correlations among its cyclic shifts. Define the maximum correlation within the teacher orbit by

$$
\mu _ { * } : = \operatorname* { m a x } _ { 1 \leq k \leq r - 1 } \left| \left. w ^ { * } , \Pi ^ { k } w ^ { * } \right. \right| .
$$

Lemma A.1 (Geometry of the teacher orbit). Under Assumptions 2 and 3, there exists a constant $C _ { * } < \infty$ such that, for the event

$$
\mathcal G _ { d } : = \{ \mu _ { * } \leq C _ { * } \gamma _ { d } \} ,
$$

it holds that

$$
\mathbb { P } ( { \mathcal { G } } _ { d } ) \to 1 .
$$

Moreover, on $\mathcal { G } _ { d }$ $r \mu _ { * } = o \left( ( \log d ) ^ { - 2 } \right)$

Proof. Assumption 2 allows us to write $w ^ { * } = g / \lVert g \rVert$ with $g \sim \mathcal { N } ( 0 , \mathrm { I } _ { d } )$ , so that

$$
\mu _ { * } = \operatorname* { m a x } _ { 1 \leq a \leq r - 1 } \frac { \Big | g ^ { \top } \Pi ^ { a } g \Big | } { \| g \| ^ { 2 } } .
$$

Thus we need an upper bound on max $\langle 1 { \leq } a { \leq } r { - } 1 \left| g ^ { \top } \Pi ^ { a } g \right|$ and a lower bound on $\| g \| ^ { 2 }$ . By the Hanson– Wright inequality (see $\mathrm { e . g . }$ (Vershynin, 2018, Theorem 6.2.1)), for every symmetric matrix $A .$

$$
\mathbb { P } \left( \left| g ^ { \top } A g - \mathbb { E } [ g ^ { \top } A g ] \right| > t \right) \leq 2 \exp \left( - c \operatorname* { m i n } \left\{ \frac { t ^ { 2 } } { \| A \| _ { \mathrm { F } } ^ { 2 } } , \frac { t } { \| A \| _ { \mathrm { o p } } } \right\} \right) , \qquad t > 0 .\tag{77}
$$

Note that $\mathbb { E } [ g ^ { \top } A g ] = \mathrm { T r } A$ . Fix $1 \leq a \leq r - 1$ . Since $a \Delta = a d / r$ is not a multiple of $d ,$ the cyclic shift $\Pi ^ { a }$ has no fixed coordinates and hence $\operatorname { T r } ( \Pi ^ { a } ) = 0$ . Since the matrix $\Pi ^ { a }$ is not necessarily symmetric, we consider its symmetric part

$$
A _ { a } : = \frac { \Pi ^ { a } + \Pi ^ { - a } } { 2 } .
$$

Then

$$
\mathrm { T r } ( A _ { a } ) = 0 , \qquad \| A _ { a } \| _ { \mathrm { o p } } \leq 1 , \qquad \| A _ { a } \| _ { \mathrm { F } } ^ { 2 } \leq d .
$$

Moreover, since the skew-symmetric part does not contribute to a quadratic form, $g ^ { \top } \Pi ^ { a } g = g ^ { \top } A _ { a } g$ Hence, setting $t = C { \sqrt { d \log r } }$ in (77) gives

$$
\begin{array} { r } { \mathbb { P } \left( \left| g ^ { \top } \Pi ^ { a } g \right| > C \sqrt { d \log r } \right) \leq 2 \exp \left( - c \operatorname* { m i n } \left\{ C ^ { 2 } \log r , C \sqrt { d \log r } \right\} \right) . } \end{array}
$$

Since log $r = \delta$ log $d = o ( d )$ , for all suficiently large $d ,$

$$
\begin{array} { r } { \mathbb { P } \left( \left| g ^ { \top } \Pi ^ { a } g \right| > C \sqrt { d \log r } \right) \leq 2 e ^ { - c C ^ { 2 } \log r } = 2 r ^ { - c C ^ { 2 } } . } \end{array}
$$

Taking a union bound over $a = 1 , \ldots , r - 1$ , we obtain

$$
\mathbb { P } \left( \operatorname* { m a x } _ { 1 \leq a \leq r - 1 } \left| \boldsymbol { g } ^ { \top } \Pi ^ { a } \boldsymbol { g } \right| > C \sqrt { d \log r } \right) \leq 2 r ^ { 1 - c C ^ { 2 } } .\tag{78}
$$

Choosing C suficiently large, the right-hand side is $o ( 1 )$ . For the lower bound on $\| g \| ^ { 2 }$ , we apply again (77) to $A = \operatorname { I } _ { d }$ , for which $\mathbb { E } \| g \| ^ { 2 } = d ;$ with $t = d / 2$ . This gives

$$
\mathbb { P } \left( \| g \| ^ { 2 } < \frac { d } { 2 } \right) \le \mathbb { P } \left( \left| \| g \| ^ { 2 } - d \right| > \frac { d } { 2 } \right) \le 2 e ^ { - c d } .\tag{79}
$$

Therefore, combining (78) and (79), with probability at least $1 - 2 r ^ { 1 - c C ^ { 2 } } - 2 e ^ { - c d } = 1 - o ( 1 )$ (1), we have

$$
\mu _ { * } = \operatorname* { m a x } _ { 1 \leq a \leq r - 1 } \frac { \left| g ^ { \top } \Pi ^ { a } g \right| } { \| g \| ^ { 2 } } \leq 2 C \sqrt { \frac { \log r } { d } } .
$$

Finally, since $r = d ^ { \delta }$ with $\delta < 1 / 2$ by Assumption 3,

$$
r \mu _ { * } \leq 2 C d ^ { \delta - \frac { 1 } { 2 } } \sqrt { \log r } = 2 C \sqrt { \delta } d ^ { \delta - \frac { 1 } { 2 } } \sqrt { \log d } = o \left( ( \log d ) ^ { - 2 } \right) .
$$

This completes the proof.

We next introduce a generic initialization $u \sim \mathrm { U n i f } ( \mathbb { S } ^ { d - 1 } )$ , independent of $w ^ { * }$ , and study its correlations with the teacher orbit. They can therefore be applied to the tied initialization and, separately, to any fixed component of the untied or augmented initialization. Recall that

$$
m _ { k } ( u ) = \langle \Pi ^ { k } w ^ { * } , u \rangle , \qquad q _ { k } ( u ) = \left\{ \begin{array} { l l } { { | m _ { k } ( u ) | , } } & { { p \mathrm { ~ e v e n } , } } \\ { { m _ { k } ( u ) , } } & { { p \mathrm { ~ o d d } . } } \end{array} \right.
$$

Lemma A.2 (Initialization scale). Suppose that Assumption 3 holds. Let $w ^ { * }$ , u be independent and uniformly distributed on $\mathbb { S } ^ { d - 1 }$ . Then

$$
\operatorname* { m a x } _ { 0 \leq k < r } | m _ { k } ( u ) | = \left( \sqrt { 2 } + o _ { \mathbb { P } } ( 1 ) \right) \gamma _ { d } .
$$

Moreover,

$$
\operatorname* { m a x } _ { 0 \leq k < r } q _ { k } ( u ) = \left( \sqrt { 2 } + o _ { \mathbb { P } } ( 1 ) \right) \gamma _ { d } .
$$

In particular, there exist constants $0 < c < C < \infty$ such that, with probability $1 - o ( 1 )$

$$
\operatorname* { m a x } _ { 0 \leq k < r } | m _ { k } ( u ) | \leq C \gamma _ { d } , \qquad \operatorname* { m a x } _ { 0 \leq k < r } q _ { k } ( u ) \geq c \gamma _ { d } .
$$

We prove Lemma $\mathrm { A . 2 }$ below. We next show that the largest relevant correlation is separated from all competing correlations by a relative gap of order $( \log d ) ^ { - 2 }$

Lemma A.3 (Initialization gap). Suppose that Assumption 3 holds. Let $w ^ { * }$ , u be independent and uniformly distributed on $\mathbb { S } ^ { d - 1 }$ , and set $\nu _ { d } : = ( \log d ) ^ { - 2 }$ . With probability $1 - o ( 1 )$ , the maximizer

$$
i _ { * } : = \arg \operatorname* { m a x } _ { 0 \leq k < r } q _ { k } ( u )
$$

is unique and satisfies

$$
\operatorname* { m a x } _ { k \neq i _ { * } } q _ { k } ( u ) \leq ( 1 - \nu _ { d } ) q _ { i _ { * } } ( u ) .
$$

We now prove Lemmas A.2 and A.3 together.

Proof of Lemmas A.2 and A.3. We first record the extreme-value estimates used below. Let $Z _ { 0 } , \ldots , Z _ { r - 1 }$ be i.i.d. N(0, 1). Denote by $A _ { 1 } \geq A _ { 2 }$ the two largest values among $| Z _ { 0 } | , \ldots , | Z _ { r - 1 } |$ and by $P _ { 1 } \geq P _ { 2 }$ the two largest values among $Z _ { 0 } , \ldots , Z _ { r - 1 }$ . Standard Gaussian tail estimates imply

$$
\frac { A _ { 1 } } { \sqrt { 2 \log r } } \stackrel { \mathbb { P } } { \to } 1 , \qquad \frac { P _ { 1 } } { \sqrt { 2 \log r } } \stackrel { \mathbb { P } } { \to } 1 .\tag{80}
$$

Moreover, applying the asymptotics of extreme spacings in the Gumbel domain (Nagaraja et al., 2015, Theorem 5) to the standard normal and half-normal distributions gives

$$
{ \sqrt { 2 \log r } } ( A _ { 1 } - A _ { 2 } ) \ { \xrightarrow { d } } \ \operatorname { E x p } ( 1 ) , \qquad { \sqrt { 2 \log r } } \ ( P _ { 1 } - P _ { 2 } ) \ { \xrightarrow { d } } \ \operatorname { E x p } ( 1 ) .\tag{81}
$$

We now deduce the relative gap needed below. Recall that $\nu _ { d } = ( \log d ) ^ { - 2 }$ . Since $r \ : = \ : d ^ { \delta }$ by Assumption 3, $\nu _ { d }$ log $r = \delta ( \log d ) ^ { - 1 } \to 0$ . Moreover, by (80),

$$
\nu _ { d } \sqrt { 2 \log r } A _ { 1 } = ( 1 + o _ { \mathbb { P } } ( 1 ) ) 2 \nu _ { d } \log r \stackrel { \mathbb { P } } { \to } 0 .
$$

For every fixed $a > 0$ , setting

$$
U _ { d } : = \sqrt { 2 \log r } ( A _ { 1 } - A _ { 2 } ) , \quad \quad V _ { d } : = a \nu _ { d } \sqrt { 2 \log r } A _ { 1 } ,
$$

we have $U _ { d } \stackrel { d } { \to } \mathrm { E x p } ( 1 )$ and $V _ { d } \stackrel { \mathbb { P } } {  } 0$ . For every $\varepsilon > 0$

$$
\mathbb { P } ( U _ { d } \leq V _ { d } ) \leq \mathbb { P } ( U _ { d } \leq \varepsilon ) + \mathbb { P } ( V _ { d } > \varepsilon ) .
$$

Taking the limit superior and then letting $\varepsilon \downarrow 0$ gives

$$
\mathbb { P } ( U _ { d } \leq V _ { d } )  0 ,
$$

since an $\exp ( 1 )$ random variable is strictly positive almost surely. Hence, for every fixed $a > 0$

$$
\mathbb { P } ( A _ { 1 } - A _ { 2 } \leq a \nu _ { d } A _ { 1 } )  0 .
$$

Equivalently,

$$
A _ { 2 } \leq ( 1 - a \nu _ { d } ) A _ { 1 }\tag{82}
$$

with probability $1 - o ( 1 )$ . The same argument yields

$$
P _ { 2 } \leq ( 1 - a \nu _ { d } ) P _ { 1 }\tag{83}
$$

with probability $1 - o ( 1 )$

We now transfer the preceding extreme-value estimates from independent Gaussian coordinates to the actual correlations with the teacher orbit. Define the teacher-geometry event of Lemma A.1:

$$
\mathcal G _ { d } : = \{ \mu _ { * } \leq C \gamma _ { d } \} .
$$

By that lemma, $\mathbb { P } \left( \mathcal { G } _ { d } \right) = 1 - o ( 1 )$ . Fix $w ^ { \ast } \in \mathcal G _ { d }$ . In what follows, probabilities are taken conditionally on this realization of $w ^ { * }$ . Write

$$
u = \frac { h } { \| h \| } , \qquad h \sim \mathcal { N } ( 0 , \mathrm { I } _ { d } ) ,
$$

where h is independent of $w ^ { * }$ , and define

$$
X _ { k } : = \left. \Pi ^ { k } w ^ { * } , h \right. , \qquad 0 \leq k < r .
$$

Indeed, for $0 \leq i , j < r _ { \cdot }$

$$
\mathbb { E } [ X _ { i } X _ { j } \mid w ^ { * } ] = \left. \Pi ^ { i } w ^ { * } , \Pi ^ { j } w ^ { * } \right. .
$$

Conditionally on $w ^ { * }$ , the vector $X : = ( X _ { 0 } , \ldots , X _ { r - 1 } )$ is centered Gaussian with covariance matrix

$$
W ^ { * } : = \left( \left. \Pi ^ { i } w ^ { * } , \Pi ^ { j } w ^ { * } \right. \right) _ { 0 \leq i , j \leq r - 1 } .
$$

For a matrix $A = ( A _ { i j } ) _ { i , j = 1 } ^ { r }$ , let

$$
\| A \| _ { \infty \to \infty } : = \operatorname* { s u p } _ { \| x \| _ { \infty } \leq 1 } \| A x \| _ { \infty } = \operatorname* { m a x } _ { 1 \leq i \leq r } \sum _ { j = 1 } ^ { r } | A _ { i j } |
$$

denote the operator norm induced by the vector $\ell _ { \infty } { \mathrm { - n o r m } }$ . The diagonal entries of $W ^ { * }$ are equal to one. Moreover, if $i \neq j$ , then

$$
\left| \left. \Pi ^ { i } w ^ { * } , \Pi ^ { j } w ^ { * } \right. \right| = \left| \left. w ^ { * } , \Pi ^ { j - i } w ^ { * } \right. \right| \leq \mu _ { * } ,
$$

where the exponent is understood modulo r. Hence, on $\mathcal { G } _ { d }$

$$
\| W ^ { * } - \mathrm { I } _ { r } \| _ { \infty \to \infty } \leq ( r - 1 ) \mu _ { * } \leq C r \gamma _ { d } .
$$

Moreover, $b _ { d } : = C r \gamma _ { d } = o ( \nu _ { d } )$ according to Lemma A.1. Since $W ^ { * } - \mathrm { I } _ { r }$ is symmetric,

$$
\| W ^ { * } - \operatorname { I } _ { r } \| _ { \mathrm { o p } } \leq \| W ^ { * } - \operatorname { I } _ { r } \| _ { \infty \to \infty } \leq b _ { d } = o ( 1 ) .
$$

Therefore, for all suficiently large $d ,$ every eigenvalue of $W ^ { * }$ belongs to $[ 1 - b _ { d } , 1 + b _ { d } ] \subset ( 0 , \infty )$ , so $W ^ { * }$ is positive definite. Set $B : = W ^ { * } - \mathrm { I } _ { r }$ . Then, for all suficiently large $d , \| B \| _ { \infty \to \infty } \leq b _ { d } < 1 / 2$ Hence the binomial series for the principal square root converges absolutely in the $\| \cdot \| _ { \infty \to \infty ^ { - \mathrm { { n o r m } : } } }$

$$
( W ^ { * } ) ^ { 1 / 2 } = ( \mathrm { I } _ { r } + B ) ^ { 1 / 2 } = \mathrm { I } _ { r } + \sum _ { m = 1 } ^ { \infty } \binom { 1 / 2 } { m } B ^ { m } .
$$

By submultiplicativity,

$$
\| ( W ^ { * } ) ^ { 1 / 2 } - \mathbf { I } _ { r } \| _ { \infty \to \infty } \leq \sum _ { m = 1 } ^ { \infty } \left| { \binom { 1 / 2 } { m } } \right| \| B \| _ { \infty \to \infty } ^ { m } \leq C \| B \| _ { \infty \to \infty } \leq C b _ { d } ,
$$

where $C < \infty$ is a universal constant. Therefore, if we set

$$
\varepsilon _ { d } ( w ^ { * } ) : = \| ( W ^ { * } ) ^ { 1 / 2 } -  { \mathrm { I } _ { r } } \| _ { \infty \to \infty } ,
$$

then, uniformly over $w ^ { \ast } \in \mathcal G _ { d }$

$$
\varepsilon _ { d } ( w ^ { * } ) \leq C b _ { d } = o ( \nu _ { d } ) .\tag{84}
$$

Now define $Z : = ( W ^ { * } ) ^ { - 1 / 2 } X$ . Conditionally on $w ^ { * } , Z \sim \mathcal { N } ( 0 , \mathrm { I } _ { r } )$ , so its coordinates are independent standard Gaussian variables. Let $A _ { 1 } \geq A _ { 2 }$ and $P _ { 1 } \geq P _ { 2 }$ denote the corresponding absolute and signed order statistics, as above. Since $X = ( W ^ { * } ) ^ { 1 / 2 } Z$ , we have

$$
\begin{array} { r } { \| X - Z \| _ { \infty } \leq \varepsilon _ { d } ( w ^ { * } ) \| Z \| _ { \infty } = \varepsilon _ { d } ( w ^ { * } ) A _ { 1 } . } \end{array}\tag{85}
$$

Absolute maximum. Let $k _ { * }$ be the almost surely unique index satisfying $| Z _ { k _ { * } } | = A _ { 1 }$ . Taking $a = 2$ in (82), with conditional probability $1 - o ( 1 )$

$$
A _ { 2 } \leq ( 1 - 2 \nu _ { d } ) A _ { 1 } .
$$

On this event,

$$
| X _ { k * } | \geq A _ { 1 } - \| X - Z \| _ { \infty } \geq ( 1 - \varepsilon _ { d } ) A _ { 1 } ,
$$

while

$$
\operatorname* { m a x } _ { k \neq k _ { * } } | X _ { k } | \leq A _ { 2 } + \| X - Z \| _ { \infty } \leq ( 1 - 2 \nu _ { d } + \varepsilon _ { d } ) A _ { 1 } .
$$

By (84), $\varepsilon _ { d } = o ( \nu _ { d } )$ uniformly on $\mathcal { G } _ { d }$ . Hence, for all suficiently large $d ,$

$$
1 - 2 \nu _ { d } + \varepsilon _ { d } \leq ( 1 - \nu _ { d } ) ( 1 - \varepsilon _ { d } ) .
$$

It follows that

$$
\operatorname* { m a x } _ { k \neq k _ { * } } | X _ { k } | \leq ( 1 - \nu _ { d } ) | X _ { k _ { * } } | .\tag{86}
$$

In particular, $k _ { * }$ is also the unique maximizer of $| X _ { k } |$ . Furthermore,

$$
\left| \operatorname* { m a x } _ { k } | X _ { k } | - A _ { 1 } \right| \leq \| X - Z \| _ { \infty } \leq \varepsilon _ { d } A _ { 1 } .
$$

Since $\varepsilon _ { d } = o ( 1 )$ uniformly on $\mathcal { G } _ { d }$ , it follows from (80) that, conditionally on $w ^ { \ast } \in \mathcal { G } _ { d }$

$$
\operatorname* { m a x } _ { 0 \leq k < r } | X _ { k } | = ( 1 + o _ { \mathbb { P } } ( 1 ) ) \sqrt { 2 \log r } ,\tag{87}
$$

where the $o _ { \mathbb { P } } ( 1 )$ is uniform over $w ^ { \ast } \in \mathcal { G } _ { d }$

Signed maximum. Let $j _ { * }$ be the almost surely unique index satisfying $Z _ { j _ { * } } = P _ { 1 }$ . By (80),

$$
{ \frac { A _ { 1 } } { P _ { 1 } } } \stackrel { \mathbb { P } } {  } 1 .
$$

In particular, with probability $1 - o ( 1 ) , A _ { 1 } \leq 2 P _ { 1 }$ . Intersect this event with the spacing event $P _ { 2 } \leq ( 1 - 2 \nu _ { d } ) P _ { 1 }$ , which also has probability $1 - o ( 1 )$ by (83). On their intersection,

$$
X _ { j _ { * } } \geq P _ { 1 } - \varepsilon _ { d } A _ { 1 } \geq ( 1 - 2 \varepsilon _ { d } ) P _ { 1 } ,
$$

while

$$
\operatorname* { m a x } _ { k \neq j _ { * } } X _ { k } \leq P _ { 2 } + \varepsilon _ { d } A _ { 1 } \leq ( 1 - 2 \nu _ { d } + 2 \varepsilon _ { d } ) P _ { 1 } .
$$

Since $\varepsilon _ { d } = o ( \nu _ { d } )$ , for all suficiently large $d ,$

$$
1 - 2 \nu _ { d } + 2 \varepsilon _ { d } \leq ( 1 - \nu _ { d } ) ( 1 - 2 \varepsilon _ { d } ) .
$$

Therefore

$$
\operatorname* { m a x } _ { k \neq j _ { * } } X _ { k } \leq ( 1 - \nu _ { d } ) X _ { j _ { * } } .\tag{88}
$$

Thus $j _ { * }$ is also the unique maximizer of the signed coordinates of X. Furthermore,

$$
\left| \operatorname* { m a x } _ { 0 \leq k < r } X _ { k } - P _ { 1 } \right| \leq \| X - Z \| _ { \infty } \leq \varepsilon _ { d } A _ { 1 } .
$$

Since $A _ { 1 } / P _ { 1 }  1$ in probability and $\varepsilon _ { d } = o ( 1 )$ , we obtain

$$
\operatorname* { m a x } _ { k } X _ { k } = ( 1 + o _ { \mathbb { P } } ( 1 ) ) \sqrt { 2 \log r } ,\tag{89}
$$

again uniformly over $w ^ { \ast } \in \mathcal { G } _ { d }$

Normalization and conclusion. Finally,

$$
m _ { k } ( u ) = \left. \Pi ^ { k } w ^ { * } , u \right. = \frac { X _ { k } } { \lVert h \rVert } .
$$

Since $\| h \| / \sqrt { d } \overset { \mathbb { P } } {  } 1$ , combining this with (87) yields

$$
\operatorname* { m a x } _ { 0 \leq k < r } | m _ { k } ( u ) | = ( 1 + o _ { \mathbb { P } } ( 1 ) ) \sqrt { \frac { 2 \log r } { d } } = ( \sqrt { 2 } + o _ { \mathbb { P } } ( 1 ) ) \gamma _ { d } .
$$

Similarly, (89) gives

$$
\operatorname* { m a x } _ { 0 \leq k < r } m _ { k } ( u ) = ( \sqrt { 2 } + o _ { \mathbb { P } } ( 1 ) ) \gamma _ { d } .
$$

Consequently,

$$
\operatorname* { m a x } _ { 0 \leq k < r } q _ { k } ( u ) = ( \sqrt { 2 } + o _ { \mathbb { P } } ( 1 ) ) \gamma _ { d } ,
$$

which proves Lemma A.2. The normalization by the common positive factor $\| h \|$ does not change either the maximizing index or any relative gap. Hence (86) gives, when $p$ is even,

$$
\operatorname* { m a x } _ { k \neq i _ { * } } q _ { k } ( u ) = \operatorname* { m a x } _ { k \neq i _ { * } } | m _ { k } ( u ) | \leq ( 1 - \nu _ { d } ) | m _ { i _ { * } } ( u ) | = ( 1 - \nu _ { d } ) q _ { i _ { * } } ( u ) ,
$$

whereas (88) gives, when $p$ is odd,

$$
\operatorname* { m a x } _ { k \neq i _ { * } } q _ { k } ( u ) = \operatorname* { m a x } _ { k \neq i _ { * } } m _ { k } ( u ) \leq ( 1 - \nu _ { d } ) m _ { i _ { * } } ( u ) = ( 1 - \nu _ { d } ) q _ { i _ { * } } ( u ) .
$$

In both cases the maximizer is unique. All the preceding conditional estimates hold uniformly over $w ^ { \ast } \in \mathcal G _ { d }$ . Therefore, if $\mathcal { E } _ { d }$ denotes the event on which the conclusions of Lemmas A.2 and A.3 hold, then

$$
\mathbb { P } ( \mathcal { E } _ { d } ^ { \mathrm { c } } ) \leq \mathbb { P } ( \mathcal { G } _ { d } ^ { \mathrm { c } } ) + \operatorname* { s u p } _ { w ^ { * } \in \mathcal { G } _ { d } } \mathbb { P } ( \mathcal { E } _ { d } ^ { \mathrm { c } } \mid w ^ { * } ) = o ( 1 ) .
$$

This completes the proofs of Lemmas A.2 and A.3.

## B Proofs of stochastic gradient estimates

We use the notation of Section 6 and the teacher-geometry event of Lemma A.1.

## B.1 Feature-gradient covariance

Proof of Lemma 6.3. Computing the Euclidean gradient of $f _ { w } ( x )$ defined in Definition 2.2 gives

$$
\widehat { \nabla } f _ { w } ( x ) = \frac { 1 } { \sqrt { r } } \sum _ { i = 0 } ^ { r - 1 } \sigma ^ { \prime } \left( \langle \Pi ^ { i } w , x \rangle \right) \Pi ^ { - i } x .
$$

Set $Z _ { i } = \langle \Pi ^ { i } w , x \rangle$ and $V _ { i } = \langle \Pi ^ { i } v , x \rangle$ . Then

$$
\langle v , \widehat \nabla f _ { w } ( x ) \rangle = \frac { 1 } { \sqrt { r } } \sum _ { i = 0 } ^ { r - 1 } \sigma ^ { \prime } ( Z _ { i } ) V _ { i } ,
$$

and therefore, by cyclic invariance,

$$
\mathbb { E } \left[ \langle v , \widehat { \nabla } f _ { w } ( x ) \rangle ^ { 2 } \right] = \sum _ { j = 0 } ^ { r - 1 } \mathbb { E } \left[ \sigma ^ { \prime } ( Z _ { 0 } ) \sigma ^ { \prime } ( Z _ { j } ) V _ { 0 } V _ { j } \right] .
$$

For the jointly Gaussian vector $( Z _ { 0 } , Z _ { j } , V _ { 0 } , V _ { j } )$ , since

$$
\mathrm { C o v } ( V _ { 0 } , Z _ { 0 } ) = \mathrm { C o v } ( V _ { j } , Z _ { j } ) = \langle v , w \rangle = 0 ,
$$

Gaussian integration by parts gives

$$
\begin{array} { r } { \mathbb E \left[ \sigma ^ { \prime } ( Z _ { 0 } ) \sigma ^ { \prime } ( Z _ { j } ) V _ { 0 } V _ { j } \right] = \langle v , \Pi ^ { j } v \rangle \mathbb E [ \sigma ^ { \prime } ( Z _ { 0 } ) \sigma ^ { \prime } ( Z _ { j } ) ] + \langle \Pi ^ { j } w , v \rangle \langle w , \Pi ^ { j } v \rangle \mathbb E [ \sigma ^ { \prime \prime } ( Z _ { 0 } ) \sigma ^ { \prime \prime } ( Z _ { j } ) ] . } \end{array}
$$

Using

$$
\mathbb { E } [ \sigma ^ { \prime } ( Z _ { 0 } ) \sigma ^ { \prime } ( Z _ { j } ) ] = \phi ^ { \prime } ( a _ { j } ( w ) ) , \qquad \mathbb { E } [ \sigma ^ { \prime \prime } ( Z _ { 0 } ) \sigma ^ { \prime \prime } ( Z _ { j } ) ] = \phi ^ { \prime \prime } ( a _ { j } ( w ) ) ,
$$

yields the announced statement.

## B.2 Coordinate noise moments

Proof of Lemma $6 . 4 \cdot$ Fix a teacher $w ^ { \ast } \in \mathcal G _ { d }$ and condition on $\mathcal { F } _ { t - 1 }$ . Then $w : = w _ { t - 1 }$ is fixed and the only remaining randomness comes from the fresh Gaussian input $x _ { t }$ . Throughout the proof, all expectations are with respect to this fresh sample. The constants below are uniform over $w \in \mathbb { S } ^ { d - }$ −1 and $w ^ { \ast } \in \mathcal { G } _ { d }$

We first claim that

$$
\sum _ { k = 0 } ^ { r - 1 } m _ { k } ( w ) ^ { 2 } \leq 2 .\tag{90}
$$

Let

$$
\begin{array} { r } { \boldsymbol { U } ^ { * } : = [ \boldsymbol { w } ^ { * } , \boldsymbol { \Pi } \boldsymbol { w } ^ { * } , \ldots , \boldsymbol { \Pi } ^ { r - 1 } \boldsymbol { w } ^ { * } ] \in \mathbb { R } ^ { d \times r } , } \end{array}
$$

so that $( m _ { 0 } ( w ) , \dots , m _ { r - 1 } ( w ) ) = ( U ^ { * } ) ^ { \top } w$ . Its Gram matrix is $W ^ { * } = ( U ^ { * } ) ^ { \top } U ^ { * }$ . On $\mathcal { G } _ { d } .$ the diagonal entries of $W ^ { * }$ are equal to one and the absolute sum of the of-diagonal entries in each row is at most

$$
( r - 1 ) \mu _ { * } \leq r \mu _ { * } = o ( 1 ) .
$$

Hence, for all suficiently large $d ,$

$$
\begin{array} { r } { \| W ^ { * } \| _ { \mathrm { o p } } \leq \| W ^ { * } \| _ { \infty  \infty } \leq 1 + ( r - 1 ) \mu _ { * } \leq 2 . } \end{array}
$$

This implies that

$$
\sum _ { k = 0 } ^ { r - 1 } m _ { k } ( w ) ^ { 2 } = \| ( U ^ { * } ) ^ { \top } w \| ^ { 2 } \leq \| U ^ { * } ( U ^ { * } ) ^ { \top } \| _ { \mathrm { { o p } } } = \| W ^ { * } \| _ { \mathrm { { o p } } } \leq 2 ,
$$

thus proving (90).

Recall that $c _ { a } = \langle w ^ { * } , \Pi ^ { a } w ^ { * } \rangle$ and $\begin{array} { r } { \phi ( c ) = \sum _ { k = p } ^ { P } a _ { k } ^ { 2 } c ^ { k } } \end{array}$ . By (1),

$$
\mathbb { E } [ y ^ { 2 } ] = \frac { 1 } { r } \sum _ { i , j = 0 } ^ { r - 1 } \phi ( c _ { j - i } ) = \sum _ { a = 0 } ^ { r - 1 } \phi ( c _ { a } ) ,
$$

because each diference $a = j - i ($ mod r) occurs exactly $r$ times. For every $| c | \le 1$ ，

$$
| \phi ( c ) | \leq \sum _ { k = p } ^ { P } a _ { k } ^ { 2 } | c | ^ { k } \leq \left( \sum _ { k = p } ^ { P } a _ { k } ^ { 2 } \right) | c | ^ { p } = \| \sigma \| _ { L ^ { 2 } } ^ { 2 } | c | ^ { p } .
$$

Therefore

$$
\mathbb { E } [ y ^ { 2 } ] \leq \phi ( 1 ) + \sum _ { a = 1 } ^ { r - 1 } | \phi ( c _ { a } ) | \leq C _ { \sigma } ( 1 + r \mu _ { * } ^ { p } ) \leq C _ { \sigma } ,
$$

where the last inequality holds uniformly on $\mathcal { G } _ { d }$ for all suficiently large d. Indeed, $\mu _ { * } \leq 1$ and $r \mu _ { * } ~ = ~ o ( 1 )$ . Since y is a polynomial of degree at most $P$ in the Gaussian input, Gaussian hypercontractivity gives

$$
\| y \| _ { L ^ { 4 } } \leq C _ { \sigma } .\tag{91}
$$

Fix $0 \leq k < r$ , set $m : = m _ { k } ( w )$ , and define the tangent projection of the kth teacher direction by

$$
\begin{array} { r } { v : = ( \mathrm { I } _ { d } - w w ^ { \top } ) \Pi ^ { k } w ^ { * } = \Pi ^ { k } w ^ { * } - m w . } \end{array}
$$

Then $v \perp w$ and $\| v \| ^ { 2 } = 1 - m ^ { 2 } \leq 1$ . Moreover, since $G = - y \nabla f _ { w } ( x )$ , we have

$$
\langle \Pi ^ { k } w ^ { \ast } , G \rangle = - y \langle v , \widehat { \nabla } f _ { w } ( x ) \rangle .
$$

Set $Q : = \langle v , \widehat { \nabla } f _ { w } ( x ) \rangle$ and $a _ { j } ( w ) : = \langle w , \Pi ^ { j } w \rangle$ . By Lemma 6.3,

$$
\mathbb { E } Q ^ { 2 } = \sum _ { j = 0 } ^ { r - 1 } \Big ( \phi ^ { \prime } ( a _ { j } ( w ) ) \langle v , \Pi ^ { j } v \rangle + \phi ^ { \prime \prime } ( a _ { j } ( w ) ) \langle w , \Pi ^ { j } v \rangle \langle \Pi ^ { j } w , v \rangle \Big ) .\tag{92}
$$

For $j = 0$

$$
\begin{array} { r } { \langle v , \Pi ^ { 0 } v \rangle = \| v \| ^ { 2 } \leq 1 , \qquad \langle w , v \rangle = 0 , } \end{array}
$$

so the $j = 0$ contribution to (92) is bounded by a constant depending only on $\sigma .$ . For $j \neq 0$ , with all indices understood modulo $r ,$

$$
\langle v , \Pi ^ { j } v \rangle = c _ { j } - m \bigl ( m _ { k + j } ( w ) + m _ { k - j } ( w ) \bigr ) + m ^ { 2 } a _ { j } ( w ) ,
$$

$$
\langle w , \Pi ^ { j } v \rangle \langle \Pi ^ { j } w , v \rangle = \left( m _ { k + j } ( w ) - m a _ { j } ( w ) \right) \left( m _ { k - j } ( w ) - m a _ { j } ( w ) \right) .
$$

Using (90), the Cauchy–Schwarz inequality, and $| a _ { j } ( w ) | \leq 1$ , we obtain

$$
\sum _ { j \neq 0 } | \langle v , \Pi ^ { j } v \rangle | \leq r \mu _ { * } + 2 | m | \sum _ { j = 0 } ^ { r - 1 } | m _ { j } ( w ) | + r m ^ { 2 } \leq r \mu _ { * } + 2 \sqrt { 2 r } | m | + r m ^ { 2 } ,\tag{93}
$$

and

$$
\begin{array} { r l r } {  { \sum _ { j \not = 0 } | \langle w , \Pi ^ { j } v \rangle \langle \Pi ^ { j } w , v \rangle | \leq \sum _ { j = 0 } ^ { r - 1 } | m _ { k + j } ( w ) m _ { k - j } ( w ) | + 2 | m | \sum _ { j = 0 } ^ { r - 1 } | m _ { j } ( w ) | + r m ^ { 2 } } } \\ & { } & { \leq 2 + 2 \sqrt { 2 r } | m | + r m ^ { 2 } . } \end{array}\tag{94}
$$

Here, for the first term in the last display, Cauchy–Schwarz and (90) give

$$
\sum _ { j = 0 } ^ { r - 1 } | m _ { k + j } ( w ) m _ { k - j } ( w ) | \leq \left( \sum _ { j } m _ { k + j } ( w ) ^ { 2 } \right) ^ { 1 / 2 } \left( \sum _ { j } m _ { k - j } ( w ) ^ { 2 } \right) ^ { 1 / 2 } \leq 2 .
$$

Since $\phi ^ { \prime }$ and $\phi ^ { \prime \prime }$ are bounded on [−1, 1], combining (92)–(94) gives

$$
\begin{array} { r } { \mathbb { E } Q ^ { 2 } \leq C _ { \sigma } \left( 1 + r \mu _ { * } + \sqrt { r } \left| m \right| + r m ^ { 2 } \right) . } \end{array}
$$

On $\mathcal { G } _ { d } , r \mu _ { * } = o ( 1 )$ , and $\sqrt { r } \left| m \right| \leq ( 1 + r m ^ { 2 } ) / 2$ . Therefore,

$$
\mathbb { E } Q ^ { 2 } \le C _ { \sigma } ( 1 + r m ^ { 2 } ) .\tag{95}
$$

The polynomial $Q$ has degree at most P in the Gaussian input. Hence, by Gaussian hypercontractivity,

$$
\| Q \| _ { L ^ { 4 } } \leq 3 ^ { P / 2 } \| Q \| _ { L ^ { 2 } } .
$$

Using Cauchy–Schwarz, (91), and (95),

$$
\begin{array} { r } { \mathbb { E } \left[ \langle \Pi ^ { k } w ^ { * } , G \rangle ^ { 2 } \right] = \mathbb { E } [ y ^ { 2 } Q ^ { 2 } ] \leq \| y \| _ { L ^ { 4 } } ^ { 2 } \| Q \| _ { L ^ { 4 } } ^ { 2 } \leq C _ { \sigma } ( 1 + r m ^ { 2 } ) . } \end{array}
$$

Now

$$
\xi _ { k , t } = - \left( \langle \Pi ^ { k } w ^ { * } , G _ { t } \rangle - \mathbb { E } [ \langle \Pi ^ { k } w ^ { * } , G _ { t } \rangle \mid \mathcal { F } _ { t - 1 } ] \right) .
$$

Centering can only decrease the second moment, and therefore

$$
\mathbb { E } [ \xi _ { k , t } ^ { 2 } \mid \mathcal { F } _ { t - 1 } ] \le C _ { \sigma } \left( 1 + r m _ { k } ( w _ { t - 1 } ) ^ { 2 } \right) ,
$$

which proves (20).

Conditionally on $\mathcal { F } _ { t - 1 }$ , the random variable $\langle \Pi ^ { k } w ^ { * } , G _ { t } \rangle$ is a polynomial of degree at most $2 P$ in the fresh Gaussian input $x _ { t }$ . Subtracting its conditional expectation does not increase its degree, so $\xi _ { k , t }$ is also $\mathrm { a }$ Gaussian polynomial of degree at most $2 P$ . Hence, for every $\nu \geq 2$ , Gaussian hypercontractivity yields

$$
\begin{array} { r } { \| \xi _ { k , t } \| _ { L ^ { \nu } ( \cdot | \mathcal F _ { t - 1 } ) } \leq ( \nu - 1 ) ^ { P } \| \xi _ { k , t } \| _ { L ^ { 2 } ( \cdot | \mathcal F _ { t - 1 } ) } . } \end{array}
$$

Using (20),

$$
\begin{array} { r } { ( \mathbb { E } [ | \xi _ { k , t } | ^ { \nu } \ | \ \mathcal { F } _ { t - 1 } ] ) ^ { 1 / \nu } \leq C _ { \sigma } \nu ^ { P } \sqrt { 1 + r { m _ { k } ( w _ { t - 1 } ) ^ { 2 } } } \leq C _ { \sigma } \nu ^ { P } \left( 1 + \sqrt { r } \left| { m _ { k } ( w _ { t - 1 } ) } \right| \right) , } \end{array}
$$

which proves (21).

## B.3 Uniform control of the stochastic gradient

Proof of Lemma 6.5. We first work conditionally on a teacher $w ^ { \ast } \in \mathcal { G } _ { d }$ and establish the uniform second-moment bound

$$
\operatorname* { s u p } _ { w \in \mathbb { S } ^ { d - 1 } } \mathbb { E } \| G ( w ; x , y ) \| ^ { 2 } \leq C _ { \sigma } d .\tag{96}
$$

All expectations in this first part are with respect to a fresh input $x \sim \mathcal { N } ( 0 , \mathrm { I } _ { d } )$ . Fix w and write $\mathrm { P } _ { w } : = \mathrm { I } _ { d } - w w ^ { \top }$ . Since

$$
G ( w ; x , y ) = - y \nabla _ { w } f _ { w } ( x ) = - y \mathrm { P } _ { w } \widehat { \nabla } _ { w } f _ { w } ( x ) ,
$$

Cauchy–Schwarz inequality gives

$$
\begin{array} { r } { \mathbb { E } \| G \| ^ { 2 } = \mathbb { E } \left[ y ^ { 2 } \| \mathrm { P } _ { w } \widehat { \nabla } _ { w } f _ { w } ( x ) \| ^ { 2 } \right] \leq \| y \| _ { L ^ { 4 } } ^ { 2 } \left( \mathbb { E } \| \mathrm { P } _ { w } \widehat { \nabla } _ { w } f _ { w } ( x ) \| ^ { 4 } \right) ^ { 1 / 2 } . } \end{array}\tag{97}
$$

By (91), $\| y \| _ { L ^ { 4 } } \leq C _ { \sigma }$ , uniformly over teachers in $\mathcal { G } _ { d } .$ . Choose an orthonormal basis $( v _ { 1 } , \ldots , v _ { d - 1 } )$ of $w ^ { \perp }$ , and set $Q _ { i } : = \langle v _ { i } , \widehat { \nabla } _ { w } f _ { w } ( x ) \rangle$ ⟩. Then

$$
\| \mathrm { P } _ { w } \widehat { \nabla } _ { w } f _ { w } ( x ) \| ^ { 2 } = \sum _ { i = 1 } ^ { d - 1 } Q _ { i } ^ { 2 } .
$$

Hence, by Minkowski’s inequality in $L ^ { 2 }$ ,

$$
\Big ( \mathbb { E } \| \mathrm { P } _ { w } \widehat { \nabla } _ { w } f _ { w } ( x ) \| ^ { 4 } \Big ) ^ { 1 / 2 } = \left\| \sum _ { i = 1 } ^ { d - 1 } Q _ { i } ^ { 2 } \right\| _ { L ^ { 2 } } \leq \sum _ { i = 1 } ^ { d - 1 } \Big \| Q _ { i } ^ { 2 } \Big \| _ { L ^ { 2 } } = \sum _ { i = 1 } ^ { d - 1 } \| Q _ { i } \| _ { L ^ { 4 } } ^ { 2 } .
$$

Each $Q _ { i }$ is a polynomial of degree at most $P$ in the Gaussian input x. Gaussian hypercontractivity therefore yields

$$
\| Q _ { i } \| _ { L ^ { 4 } } \leq 3 ^ { P / 2 } \| Q _ { i } \| _ { L ^ { 2 } } ,
$$

and consequently

$$
\left( \mathbb { E } \| \mathrm { P } _ { w } \widehat { \nabla } _ { w } f _ { w } ( x ) \| ^ { 4 } \right) ^ { 1 / 2 } \leq 3 ^ { P } \sum _ { i = 1 } ^ { d - 1 } \mathbb { E } \left[ Q _ { i } ^ { 2 } \right] .\tag{98}
$$

It remains to control the sum on the right-hand side of (98). By Lemma 6.3,

$$
\sum _ { i = 1 } ^ { d - 1 } \mathbb E \left[ Q _ { i } ^ { 2 } \right] = \sum _ { j = 0 } ^ { r - 1 } \left( \phi ^ { \prime } ( a _ { j } ( w ) ) \sum _ { i = 1 } ^ { d - 1 } \langle v _ { i } , \Pi ^ { j } v _ { i } \rangle + \phi ^ { \prime \prime } ( a _ { j } ( w ) ) \sum _ { i = 1 } ^ { d - 1 } \langle w , \Pi ^ { j } v _ { i } \rangle \langle \Pi ^ { j } w , v _ { i } \rangle \right) ,
$$

where $a _ { j } ( w ) = \langle w , \Pi ^ { j } w \rangle$ . For the first sum, since $\begin{array} { r } { \mathrm { P } _ { w } = \sum _ { i = 1 } ^ { d - 1 } v _ { i } v _ { i } ^ { \top } } \end{array}$ , we have

$$
\sum _ { i = 1 } ^ { d - 1 } \langle v _ { i } , \Pi ^ { j } v _ { i } \rangle = \operatorname { T r } ( \mathrm { P } _ { w } \Pi ^ { j } ) .
$$

$\mathrm { I f } \ j = 0$ , this equals $d - 1 . { \mathrm { ~ I f ~ } } 1 \leq j < r$ , then $\mathrm { T r } ( \Pi ^ { j } ) = 0$ , and therefore

$$
\begin{array} { r } { \mathrm { T r } ( \mathrm { P } _ { w } \Pi ^ { j } ) = - \langle w , \Pi ^ { j } w \rangle = - a _ { j } ( w ) . } \end{array}
$$

As a result,

$$
\sum _ { j = 0 } ^ { r - 1 } \left| \sum _ { i = 1 } ^ { d - 1 } \langle v _ { i } , \Pi ^ { j } v _ { i } \rangle \right| \leq d + r .
$$

For the second sum,

$$
\sum _ { i = 1 } ^ { d - 1 } \langle w , \Pi ^ { j } v _ { i } \rangle \langle \Pi ^ { j } w , v _ { i } \rangle = \langle \mathrm { P } _ { w } \Pi ^ { - j } w , \mathrm { P } _ { w } \Pi ^ { j } w \rangle ,
$$

and hence its absolute value is at most one. Since $\phi ^ { \prime }$ and $\phi ^ { \prime \prime }$ are uniformly bounded on $[ - 1 , 1 ]$

$$
\sum _ { i = 1 } ^ { d - 1 } \mathbb { E } \left[ Q _ { i } ^ { 2 } \right] \leq C _ { \sigma } ( d + r ) \leq C _ { \sigma } d ,
$$

where we used $r = d ^ { \delta } \leq d .$ . Combining this estimate with (98) and (97) proves (96).

We now derive higher conditional moments. Fix $t \geq 1$ and condition on $\mathcal { F } _ { t - 1 }$ . Then $w _ { t - 1 }$ is fixed and the only randomness comes from the fresh Gaussian input $x _ { t }$ . Choose an orthonormal basis $( v _ { 1 } , \ldots , v _ { d - 1 } )$ of $w _ { t - 1 } ^ { \perp }$ , and set $R _ { i } : = \langle v _ { i } , G _ { t } \rangle$ . Each $R _ { i }$ is a polynomial of degree at most $2 P$ in $x _ { t }$ . Thus, for every $\nu \geq 2$ , Gaussian hypercontractivity gives

$$
\| R _ { i } \| _ { L ^ { \nu } ( \cdot | \mathcal F _ { t - 1 } ) } \leq ( \nu - 1 ) ^ { P } \| R _ { i } \| _ { L ^ { 2 } ( \cdot | \mathcal F _ { t - 1 } ) } .
$$

Using Minkowski in $L ^ { \nu / 2 }$ 2

$$
\begin{array} { r l } {  { ( \mathbb { E } [ \| G _ { t } \| ^ { \nu } \mid \mathcal { F } _ { t - 1 } ] ) ^ { 2 / \nu } = \| \overset { d - 1 } { \sum _ { i = 1 } ^ { n } } R _ { i } ^ { 2 } \| _ { L ^ { \nu / 2 } } } } \\ & { \leq \sum _ { i = 1 } ^ { d - 1 } \| R _ { i } \| _ { L ^ { \nu } } ^ { 2 } } \\ & { \leq ( \nu - 1 ) ^ { 2 P } \sum _ { i = 1 } ^ { d - 1 } \| R _ { i } \| _ { L ^ { 2 } } ^ { 2 } } \\ & { = ( \nu - 1 ) ^ { 2 P } \mathbb { E } [ \| G _ { t } \| ^ { 2 } \mid \mathcal { F } _ { t - 1 } ] } \\ & { \leq C _ { \sigma } d \nu ^ { 2 P } , } \end{array}
$$

where the last inequality follows from (96). Therefore

$$
\begin{array} { r } { ( \mathbb { E } [ \| G _ { t } \| ^ { \nu } | \mathcal { F } _ { t - 1 } ] ) ^ { 1 / \nu } \leq C _ { \sigma } \sqrt { d } \nu ^ { P } . } \end{array}\tag{99}
$$

Take $\nu _ { d } = \lceil c \log d \rceil$ for a fixed $c > 0$ . By conditional Markov’s inequality and (99), for every $D > 0$ one can choose $L = L ( \sigma , D ) < \infty$ suficiently large so that

$$
\mathbb { P } \left( \Vert G _ { t } \Vert > L \sqrt { d } \left( \log d \right) ^ { P } \Big | \mathcal { F } _ { t - 1 } \right) \leq d ^ { - D }
$$

uniformly in $t ,$ for all suficiently large $d ,$ on the event $\mathcal { G } _ { d }$ . Choose $D > C + 1$ . Taking expectations and applying a union bound over $1 \leq t \leq T \leq d ^ { C }$ gives

$$
\mathbb { P } \left( \left\{ \underset { 1 \leq t \leq T } { \operatorname* { m a x } } \left\| \boldsymbol { G } _ { t } \right\| > L \sqrt { d } \left( \log d \right) ^ { P } \right\} \cap \mathcal { G } _ { d } \right) \leq d ^ { C - D } = o ( 1 ) .
$$

Finally, $\mathbb { P } ( \mathcal { G } _ { d } ^ { \mathrm { c } } ) = o ( 1 )$ by Lemma A.1, so that

$$
\mathbb { P } \left( \operatorname* { m a x } _ { 1 \leq t \leq T } \| G _ { t } \| > L \sqrt { d } ( \log d ) ^ { P } \right) = o ( 1 ) .
$$

This proves the lemma.